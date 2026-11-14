//! Decode operands inside bounded matrix tiles; never construct an expanded expert bank.

use std::collections::BTreeMap;

use crate::engine::{MlxError, TensorRef, TranslationContext};
use crate::mlx::{Array, metal::Kernel};
use crate::sys::mlx;

pub(crate) struct Routing {
    pub indices: mlx::mlx_array,
    pub values: Vec<i64>,
    pub selections: i32,
}

pub(crate) fn read_routing(
    ctx: &mut TranslationContext,
    input: &TensorRef,
    groups: i32,
) -> Result<Routing, MlxError> {
    let indices = ctx.resolve(input)?;
    let shape = ctx.shape_of(indices);
    if shape.len() != 2
        || !matches!(
            ctx.dtype_of(indices),
            mlx::mlx_dtype__MLX_INT32 | mlx::mlx_dtype__MLX_INT64
        )
    {
        return Err("GroupedMatMul: supply an int32 or int64 [M,S] group_indices tensor".into());
    }
    let values = ctx.read_ints_eval(input)?;
    if values.len() != (i64::from(shape[0]) * i64::from(shape[1])) as usize {
        return Err(
            "GroupedMatMul: incomplete routing input; supply all [M,S] group_indices".into(),
        );
    }
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|&(_, &v)| v < 0 || v >= i64::from(groups))
    {
        return Err(format!(
            "GroupedMatMul: group_indices[{index}]={value} is outside [0,{groups}); correct the expert IDs to match the weight bank"
        ));
    }
    Ok(Routing {
        indices: ctx.contiguous(indices)?,
        values,
        selections: shape[1],
    })
}

struct Batches {
    routes: mlx::mlx_array,
    experts: mlx::mlx_array,
    starts: mlx::mlx_array,
    count: i32,
    maximum: i32,
}

fn batches(ctx: &mut TranslationContext, routing: &Routing) -> Batches {
    let mut groups = BTreeMap::<i32, Vec<i32>>::new();
    for (route, &expert) in routing.values.iter().enumerate() {
        groups.entry(expert as i32).or_default().push(route as i32);
    }
    let count = groups.len() as i32;
    let maximum = groups.values().map(Vec::len).max().unwrap_or(0) as i32;
    let mut experts = Vec::with_capacity(groups.len());
    let mut routes = Vec::with_capacity(routing.values.len());
    let mut starts = vec![0];
    for (expert, entries) in groups {
        experts.push(expert);
        routes.extend(entries);
        starts.push(routes.len() as i32);
    }
    let mut array = |values: &[i32]| {
        ctx.keep(Array::from_data(
            values.as_ptr().cast(),
            &[values.len() as i32],
            mlx::mlx_dtype__MLX_INT32,
        ))
    };
    Batches {
        routes: array(&routes),
        experts: array(&experts),
        starts: array(&starts),
        count,
        maximum,
    }
}

pub(crate) enum KernelKind {
    Matvec,
    Tiled,
}

pub(crate) struct Matmul<'a> {
    /// x, w, xscale, wscale, xzero, wzero; unused parameters are scalar placeholders.
    pub operands: [mlx::mlx_array; 6],
    pub m: i32,
    pub k: i32,
    pub n: i32,
    pub routing: Option<&'a Routing>,
    /// Load-order hint; the reader still maps every element to its source address.
    pub k_contiguous: bool,
    pub header: &'a str,
    pub shape: &'a [i32],
    pub dtype: mlx::mlx_dtype,
}

impl Matmul<'_> {
    pub fn run(
        &self,
        ctx: &mut TranslationContext,
    ) -> Result<Option<(mlx::mlx_array, KernelKind)>, MlxError> {
        let s = self.routing.map_or(1, |r| r.selections);
        let Some(routes) = self.m.checked_mul(s) else {
            return Ok(None);
        };
        let Some(total) = routes.checked_mul(self.n) else {
            return Ok(None);
        };
        if total == 0 || self.k == 0 {
            return Ok(None);
        }
        let mut inputs = ["x", "w", "xscale", "wscale", "xzero", "wzero"]
            .into_iter()
            .zip(self.operands)
            .collect::<Vec<_>>();
        let mut templates = vec![
            ("M", self.m),
            ("K", self.k),
            ("N", self.n),
            ("S", s),
            ("GROUPED", i32::from(self.routing.is_some())),
            ("K_CONTIGUOUS", i32::from(self.k_contiguous)),
        ];
        let vector = self.k_contiguous && routes <= 4 && total.checked_mul(32).is_some();
        let (grid, group, name, source, kind) = if vector {
            let indices = self
                .routing
                .map_or_else(|| ctx.scalar_i32(0), |r| r.indices);
            inputs.push(("indices", indices));
            let threads = total * 32;
            (
                [threads, 1, 1],
                [threads.min(256), 1, 1],
                c"mlx_weight_matvec",
                include_str!("weight_matvec.metal"),
                KernelKind::Matvec,
            )
        } else {
            let batches = if let Some(routing) = self.routing {
                batches(ctx, routing)
            } else {
                let zero = ctx.scalar_i32(0);
                Batches {
                    routes: zero,
                    experts: zero,
                    starts: zero,
                    count: 1,
                    maximum: self.m,
                }
            };
            let Some(grid_x) = ((self.n - 1) / 32 + 1).checked_mul(128) else {
                return Ok(None);
            };
            let tile_m = if batches.maximum >= 16 { 32 } else { 8 };
            templates.push(("TILE_M", tile_m));
            inputs.extend([
                ("routes", batches.routes),
                ("experts", batches.experts),
                ("starts", batches.starts),
            ]);
            (
                [grid_x, (batches.maximum - 1) / tile_m + 1, batches.count],
                [128, 1, 1],
                c"mlx_weight_tiled_matmul",
                include_str!("weight_matmul.metal"),
                KernelKind::Tiled,
            )
        };
        // BF16's exponent range must not be restricted to FP16.
        let tile_type = if self.dtype == mlx::mlx_dtype__MLX_FLOAT16 {
            "half"
        } else {
            "float"
        };
        let header = format!("using mlx_weight_tile = {tile_type};\n{}", self.header);
        let y = Kernel {
            name,
            source,
            header: &header,
            templates: &templates,
            shape: self.shape,
            dtype: self.dtype,
            grid,
            group,
        }
        .run(ctx, &inputs)?;
        Ok(Some((y, kind)))
    }
}
