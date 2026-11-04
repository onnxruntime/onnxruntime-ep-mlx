//! Exact gated-delta state/replay and pullback Scan recognition.
//! Metal loops carry state in registers, with four value columns per group.
//! Reverse query/key/decay/beta partials reduce across independent value tiles.
//! Other bodies retain ordinary Scan execution.
use crate::engine::{MlxError, NodeDesc, Src, SubgraphDesc, TensorRef, TranslationContext};
use crate::sys::{mlx, ort};
use std::collections::HashMap;
use std::ffi::CString;

const F32: mlx::mlx_dtype = mlx::mlx_dtype__MLX_FLOAT32;
pub const REVERSE_KERNEL_NAME: &str = "mlx_gated_delta_reverse_scan";
pub const NO_FUSE_ENV: &str = "ONNXRUNTIME_EP_MLX_NO_GATED_DELTA_REVERSE";
pub fn reverse_disabled() -> bool {
    std::env::var_os(NO_FUSE_ENV)
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

enum Literal {
    Axes(Vec<i64>),
    Float(f32),
}
fn literal(r: &TensorRef, producers: &HashMap<&str, &NodeDesc>) -> Option<Literal> {
    if r.source == Src::CtxInput && !r.constant {
        return None;
    }
    let read = |bytes: &[u8], count: usize, dt| {
        if dt == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && count == 1
            && bytes.len() >= 4
        {
            let v = f32::from_le_bytes(bytes[..4].try_into().ok()?);
            return v.is_finite().then_some(Literal::Float(v));
        }
        if dt == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            && (1..=4).contains(&count)
            && bytes.len() >= count * 8
        {
            return Some(Literal::Axes(
                bytes[..count * 8]
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|v| i64::from_le_bytes(*v))
                    .collect(),
            ));
        }
        None
    };
    if let Some(init) = &r.init
        && (r.source == Src::Initializer || r.constant)
    {
        let width = if init.dtype
            == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            && init.count == 1
        {
            4
        } else if init.dtype == ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
            && (1..=4).contains(&init.count)
        {
            8
        } else {
            return None;
        };
        if init.data.is_null() {
            return None;
        }
        return read(
            // The plan retains initializer storage; the dtype/count checks above bound this read.
            unsafe { std::slice::from_raw_parts(init.data as *const u8, init.count * width) },
            init.count,
            init.dtype,
        );
    }
    let n = producers.get(r.name.as_str())?;
    if n.op_type != "Constant" || !n.domain.is_empty() {
        return None;
    }
    let t = n.tensors.get("value")?;
    read(&t.data, t.count, t.dtype)
}
fn op(name: &str, mut xs: Vec<String>) -> String {
    if name == "Mul" || name == "Add" {
        xs.sort();
    }
    format!("{name}({})", xs.join(","))
}
fn bin(name: &str, x: &str, y: &str) -> String {
    op(name, vec![x.into(), y.into()])
}
fn axes(x: &str, name: &str, a: &[i64]) -> String {
    op(name, vec![x.into(), format!("a{a:?}")])
}
fn reduce(x: &str, keep: i64) -> String {
    axes(x, &format!("ReduceSum{keep}"), &[-1])
}
fn outer(x: &str, y: &str) -> String {
    bin(
        "Mul",
        &axes(x, "Unsqueeze", &[-1]),
        &axes(y, "Unsqueeze", &[-2]),
    )
}
fn matvec(x: &str, y: &str) -> String {
    axes(
        &bin("MatMul", x, &axes(y, "Unsqueeze", &[-1])),
        "Squeeze",
        &[-1],
    )
}
fn retrieve(x: &str, y: &str) -> String {
    axes(
        &bin("MatMul", &axes(y, "Unsqueeze", &[-2]), x),
        "Squeeze",
        &[-2],
    )
}

struct Matcher<'a> {
    body: &'a SubgraphDesc,
    producers: HashMap<&'a str, &'a NodeDesc>,
    memo: HashMap<String, String>,
}
impl Matcher<'_> {
    fn expr(&mut self, name: &str) -> Option<String> {
        if let Some(i) = self.body.input_names.iter().position(|n| n == name) {
            return Some(format!("i{i}"));
        }
        if let Some(e) = self.memo.get(name) {
            return Some(e.clone());
        }
        let n = *self.producers.get(name)?;
        if n.outputs.len() != 1 || !n.domain.is_empty() {
            return None;
        }
        let mut xs = Vec::new();
        for r in &n.inputs {
            let s = match literal(r, &self.producers) {
                Some(Literal::Axes(a)) => format!("a{a:?}"),
                Some(Literal::Float(v)) => format!("f{:08x}", v.to_bits()),
                None => self.expr(&r.name)?,
            };
            xs.push(s);
        }
        let tag = match n.op_type.as_str() {
            "Mul" | "Add" | "Sub" | "MatMul" | "Unsqueeze" | "Squeeze" if xs.len() == 2 => {
                n.op_type.clone()
            }
            "Exp" if xs.len() == 1 => n.op_type.clone(),
            "ReduceSum" if xs.len() == 2 => {
                format!("ReduceSum{}", n.ints.get("keepdims").copied().unwrap_or(1))
            }
            _ => return None,
        };
        let out = op(&tag, xs);
        if out.len() > 16 * 1024 {
            return None;
        }
        self.memo.insert(name.into(), out.clone());
        Some(out)
    }
}

/// Match all six output expressions, including static axes and the actual scale.
/// Names, node ordering and common-subexpression sharing do not affect recognition.
pub fn match_reverse_body(body: &SubgraphDesc, num_state: usize, num_scan: usize) -> Option<f32> {
    if num_state != 1
        || num_scan != 7
        || body.input_names.len() != 8
        || body.output_names.len() != 6
        || body.nodes.len() > 128
    {
        return None;
    }
    let producers: HashMap<_, _> = body
        .nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |r| (r.name.as_str(), n)))
        .collect();
    let dq = producers.get(body.output_names[1].as_str())?;
    if dq.op_type != "Mul" || !dq.domain.is_empty() {
        return None;
    }
    let scale = dq
        .inputs
        .iter()
        .find_map(|r| match literal(r, &producers) {
            Some(Literal::Float(v)) => Some(v),
            _ => None,
        })?;
    let factor = op("Exp", vec![axes("i5", "Unsqueeze", &[-1])]);
    let faded = bin("Mul", "i1", &factor);
    let residual = bin("Sub", "i4", &retrieve(&faded, "i3"));
    let rate = axes("i6", "Unsqueeze", &[-1, -2]);
    let present = bin("Add", &faded, &bin("Mul", &rate, &outer("i3", &residual)));
    let scalar = format!("f{:08x}", scale.to_bits());
    let total = bin("Add", "i0", &outer(&bin("Mul", "i2", &scalar), "i7"));
    let product = retrieve(&total, "i3");
    let dv = bin("Mul", &product, &axes("i6", "Unsqueeze", &[-1]));
    let dfaded = bin("Sub", &total, &outer("i3", &dv));
    let expected = [
        bin("Mul", &factor, &dfaded),
        bin("Mul", &matvec(&present, "i7"), &scalar),
        bin(
            "Sub",
            &bin(
                "Mul",
                &matvec(&total, &residual),
                &axes("i6", "Unsqueeze", &[-1]),
            ),
            &matvec(&faded, &dv),
        ),
        dv,
        reduce(&reduce(&bin("Mul", &dfaded, &faded), 0), 1),
        reduce(&bin("Mul", &product, &residual), 0),
    ];
    let mut m = Matcher {
        body,
        producers,
        memo: HashMap::new(),
    };
    for (name, want) in body.output_names.iter().zip(expected) {
        if m.expr(name)? != want {
            return None;
        }
    }
    // Do not erase unrelated or random body computations.
    if body.nodes.iter().any(|n| {
        n.op_type != "Constant"
            && (n.outputs.len() != 1 || !m.memo.contains_key(&n.outputs[0].name))
    }) {
        return None;
    }
    Some(scale)
}

const SOURCE: &str = r#"
uint ki = thread_index_in_threadgroup;
uint subgroup = ki / 32u;
uint lane = ki % 32u;
uint tile = threadgroup_position_in_grid.x % (uint)NT;
uint bh = threadgroup_position_in_grid.x / (uint)NT;
const uint NS = (uint)((K < 32 ? 32 : K) / 32);
threadgroup float scratch[4*4*2];
threadgroup float sums[4*2];
threadgroup float gparts[4];
float carry[4];
for (uint j=0;j<4;++j) {
 uint vi=tile*4u+j;
 carry[j]=(ki<(uint)K && vi<(uint)V)?c0[(bh*(uint)K+ki)*(uint)V+vi]:0.0f;
}
for (uint step=0;step<(uint)T;++step) {
 uint t=IN_REVERSE ? (uint)T-1u-step:step;
 uint o=OUT_REVERSE ? (uint)T-1u-step:step;
 uint tv=t*(uint)(B*H)+bh;
 uint ov=o*(uint)(B*H)+bh;
 float key=ki<(uint)K?k[tv*(uint)K+ki]:0.0f;
 float query=ki<(uint)K?q[tv*(uint)K+ki]:0.0f;
 float factor=precise::exp(g[tv]);
 float rate=beta[tv];
 float faded[4],total[4],dy[4];
 for (uint j=0;j<4;++j) {
  uint vi=tile*4u+j;
  dy[j]=vi<(uint)V?gy[tv*(uint)V+vi]:0.0f;
  faded[j]=(ki<(uint)K && vi<(uint)V)?previous[(tv*(uint)K+ki)*(uint)V+vi]*factor:0.0f;
  total[j]=carry[j]+(query*SCALE)*dy[j];
  float a=simd_sum(faded[j]*key);
  float b=simd_sum(total[j]*key);
  if (lane==0) { scratch[j*8u+subgroup]=a;scratch[j*8u+4u+subgroup]=b; }
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if (ki<4) {
  float a=0.0f,b=0.0f;
  for(uint s=0;s<NS;++s) {a+=scratch[ki*8u+s];b+=scratch[ki*8u+4u+s];}
  sums[ki*2u]=a;sums[ki*2u+1u]=b;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float dqv=0.0f,dkv=0.0f,dgv=0.0f,dbv=0.0f;
 for (uint j=0;j<4;++j) {
  uint vi=tile*4u+j;
  float residual=(vi<(uint)V?v[tv*(uint)V+vi]:0.0f)-sums[j*2u];
  float product=sums[j*2u+1u];
  float dr=rate*product;
  float present=faded[j]+rate*(key*residual);
  dqv+=present*dy[j];
  dkv+=(total[j]*residual)*rate-faded[j]*dr;
  float dfaded=total[j]-key*dr;
  dgv+=dfaded*faded[j];
  dbv+=product*residual;
  carry[j]=factor*dfaded;
  if(ki==0 && vi<(uint)V) {dv[ov*(uint)V+vi]=dr;}
 }
 if(ki<(uint)K) {
  uint z=(ov*(uint)NT+tile)*(uint)K+ki;
  dq[z]=dqv*SCALE;dk[z]=dkv;
 }
 float gsum=simd_sum(dgv);
 if(lane==0) {gparts[subgroup]=gsum;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(ki==0) {
  float sum=0.0f;for(uint s=0;s<NS;++s) {sum+=gparts[s];}
  dg[ov*(uint)NT+tile]=sum;db[ov*(uint)NT+tile]=dbv;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint j=0;j<4;++j) {
 uint vi=tile*4u+j;
 if(ki<(uint)K && vi<(uint)V) {cfinal[(bh*(uint)K+ki)*(uint)V+vi]=carry[j];}
}
"#;

struct Strings(mlx::mlx_vector_string);
impl Strings {
    fn new(values: &[&str]) -> Result<Self, MlxError> {
        let raw = unsafe { mlx::mlx_vector_string_new() };
        for value in values {
            let s = CString::new(*value).map_err(|_| "GatedDelta: invalid name".to_string())?;
            unsafe { mlx::mlx_vector_string_append_value(raw, s.as_ptr()) };
        }
        Ok(Self(raw))
    }
}
impl Drop for Strings {
    fn drop(&mut self) {
        unsafe { mlx::mlx_vector_string_free(self.0) };
    }
}

/// Emit one register-carried recurrence and reduce the independent value tiles.
pub fn emit_reverse(
    ctx: &mut TranslationContext,
    inputs: &[mlx::mlx_array],
    scale: f32,
    in_reverse: bool,
    out_reverse: bool,
) -> Result<Option<Vec<mlx::mlx_array>>, MlxError> {
    if inputs.len() != 8 || inputs.iter().any(|&x| ctx.dtype_of(x) != F32) {
        return Ok(None);
    }
    let shapes: Vec<_> = inputs.iter().map(|&x| ctx.shape_of(x)).collect();
    if shapes[0].len() != 4 || shapes[2].len() != 4 {
        return Ok(None);
    }
    let (b, h, k, v) = (shapes[0][0], shapes[0][1], shapes[0][2], shapes[0][3]);
    let t = shapes[2][0];
    if b <= 0 || h <= 0 || t <= 0 || ![16, 32, 64, 128].contains(&k) || v <= 0 || v > 128 {
        return Ok(None);
    }
    let want = [
        vec![b, h, k, v],
        vec![t, b, h, k, v],
        vec![t, b, h, k],
        vec![t, b, h, k],
        vec![t, b, h, v],
        vec![t, b, h, 1],
        vec![t, b, h],
        vec![t, b, h, v],
    ];
    if shapes != want {
        return Ok(None);
    }
    let nt = (v + 3) / 4;
    let elements = t as i128 * b as i128 * h as i128 * k as i128 * v as i128;
    let grid = b as i128 * h as i128 * nt as i128 * k.max(32) as i128;
    if elements > u32::MAX as i128 || grid > i32::MAX as i128 {
        return Ok(None);
    }
    let name = CString::new(format!(
        "onnxrt_gated_delta_reverse_{:08x}",
        scale.to_bits()
    ))
    .unwrap();
    // Embed the exact FP32 bits inside the kernel body, without global preprocessor state.
    let literal = format!("as_type<float>(0x{:08x}u)", scale.to_bits());
    let header = CString::new("").unwrap();
    let source = CString::new(SOURCE.replace("SCALE", &literal)).unwrap();
    let inn = Strings::new(&["c0", "previous", "q", "k", "v", "g", "beta", "gy"])?;
    let outn = Strings::new(&["cfinal", "dq", "dk", "dv", "dg", "db"])?;
    let kernel = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            inn.0,
            outn.0,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        )
    };
    if kernel.ctx.is_null() {
        return Ok(None);
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    let mut rc = 0;
    for (key, value) in [
        ("T", t),
        ("B", b),
        ("H", h),
        ("K", k),
        ("V", v),
        ("NT", nt),
        ("IN_REVERSE", in_reverse as i32),
        ("OUT_REVERSE", out_reverse as i32),
    ] {
        let key = CString::new(key).unwrap();
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, key.as_ptr(), value)
        };
    }
    let outs = [
        vec![b, h, k, v],
        vec![t, b, h, nt, k],
        vec![t, b, h, nt, k],
        vec![t, b, h, v],
        vec![t, b, h, nt, 1],
        vec![t, b, h, nt],
    ];
    for shape in &outs {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                shape.as_ptr(),
                shape.len(),
                F32,
            )
        };
    }
    let threads = k.max(32);
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, b * h * nt * threads, 1, 1) };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_thread_group(config, threads, 1, 1) };
    let mut args = crate::mlx::VectorArray::new();
    for &x in inputs {
        args.append(x);
    }
    let mut outputs = crate::mlx::VectorArray::new();
    if rc == 0 {
        rc = unsafe {
            mlx::mlx_fast_metal_kernel_apply(
                outputs.as_mut_ptr(),
                kernel,
                args.as_raw(),
                config,
                ctx.stream(),
            )
        };
    }
    unsafe {
        mlx::mlx_fast_metal_kernel_config_free(config);
        mlx::mlx_fast_metal_kernel_free(kernel);
    }
    if rc != 0 || outputs.size() != 6 {
        return Ok(None);
    }
    let mut result = Vec::new();
    for i in 0..6 {
        let value = ctx.keep(outputs.get(i));
        if [1, 2, 4, 5].contains(&i) {
            let axis = 3;
            result.push(
                ctx.emit(|out, s| unsafe { mlx::mlx_sum_axes(out, value, &axis, 1, false, s) })?,
            );
        } else {
            result.push(value);
        }
    }
    Ok(Some(result))
}

pub const STATE_KERNEL_NAME: &str = "mlx_gated_delta_state_scan";
pub const REPLAY_KERNEL_NAME: &str = "mlx_gated_delta_replay_scan";

/// Recognize the state transition and optional previous-state output exactly.
pub fn match_state_body(body: &SubgraphDesc, num_state: usize, num_scan: usize) -> Option<bool> {
    if num_state != 1
        || num_scan != 4
        || body.input_names.len() != 5
        || !(1..=2).contains(&body.output_names.len())
        || body.nodes.len() > 64
    {
        return None;
    }
    let producers: HashMap<_, _> = body
        .nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |r| (r.name.as_str(), n)))
        .collect();
    let factor = op("Exp", vec![axes("i3", "Unsqueeze", &[-1])]);
    let faded = bin("Mul", "i0", &factor);
    let residual = bin("Sub", "i2", &retrieve(&faded, "i1"));
    let present = bin(
        "Add",
        &faded,
        &bin(
            "Mul",
            &axes("i4", "Unsqueeze", &[-1, -2]),
            &outer("i1", &residual),
        ),
    );
    let mut m = Matcher {
        body,
        producers,
        memo: HashMap::new(),
    };
    if m.expr(&body.output_names[0])? != present {
        return None;
    }
    let history = body.output_names.len() == 2;
    if history && body.output_names[1] != body.input_names[0] {
        let node = m.producers.get(body.output_names[1].as_str())?;
        if !node.domain.is_empty()
            || node.op_type != "Identity"
            || node.inputs.len() != 1
            || node.inputs[0].name != body.input_names[0]
            || node.outputs.len() != 1
        {
            return None;
        }
        m.memo.insert(body.output_names[1].clone(), "i0".into());
    }
    if body.nodes.iter().any(|node| {
        node.op_type != "Constant"
            && (node.outputs.len() != 1 || !m.memo.contains_key(&node.outputs[0].name))
    }) {
        return None;
    }
    Some(history)
}

const STATE_SOURCE: &str = r#"
uint ki=thread_index_in_threadgroup;
uint lane=ki%32u, subgroup=ki/32u;
const uint NS=(uint)((K<32?32:K)/32);
uint tile=threadgroup_position_in_grid.x%(uint)NT;
uint bh=threadgroup_position_in_grid.x/(uint)NT;
threadgroup float partials[4*4];
float carry[4];
for(uint j=0;j<4;++j) {
 uint vi=tile*4u+j;
 carry[j]=(ki<(uint)K && vi<(uint)V)?initial[(bh*(uint)K+ki)*(uint)V+vi]:0.0f;
}
for(uint step=0;step<(uint)T;++step) {
 uint ti=IN_REVERSE?(uint)T-1u-step:step;
 uint tbh=ti*(uint)(B*H)+bh;
 HISTORY_WRITE
 float key=ki<(uint)K?keys[tbh*(uint)K+ki]:0.0f;
 float factor=ki<(uint)K?exp(decays[tbh*(uint)G+(G==1?0u:ki)]):1.0f;
 float faded[4];
 for(uint j=0;j<4;++j) {
  faded[j]=carry[j]*factor;
  float sum=simd_sum(key*faded[j]);
  if(lane==0) {partials[j*NS+subgroup]=sum;}
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float rate=betas[tbh];
 for(uint j=0;j<4;++j) {
  uint vi=tile*4u+j;
  float sum=0.0f;
  for(uint s=0;s<NS;++s) {sum+=partials[j*NS+s];}
  float residual=(vi<(uint)V?values[tbh*(uint)V+vi]:0.0f)-sum;
  carry[j]=faded[j]+rate*(key*residual);
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint j=0;j<4;++j) {
 uint vi=tile*4u+j;
 if(ki<(uint)K && vi<(uint)V) {final[(bh*(uint)K+ki)*(uint)V+vi]=carry[j];}
}
"#;
const HISTORY_WRITE: &str = r#"
 uint oi=OUT_REVERSE?(uint)T-1u-step:step;
 for(uint j=0;j<4;++j) {
  uint vi=tile*4u+j;
  if(ki<(uint)K && vi<(uint)V) {
   previous[((oi*(uint)(B*H)+bh)*(uint)K+ki)*(uint)V+vi]=carry[j];
  }
 }
"#;

#[derive(Debug, PartialEq)]
struct StateGeometry {
    b: i32,
    h: i32,
    k: i32,
    v: i32,
    t: i32,
    g: i32,
    nt: i32,
    grid: i32,
}
fn state_geometry(shapes: &[Vec<i32>], history: bool) -> Option<StateGeometry> {
    if shapes.len() != 5 || shapes[0].len() != 4 || shapes[1].len() != 4 {
        return None;
    }
    let (b, h, k, v) = (shapes[0][0], shapes[0][1], shapes[0][2], shapes[0][3]);
    let t = shapes[1][0];
    if b <= 0 || h <= 0 || t <= 0 || ![16, 32, 64, 128].contains(&k) || !(1..=128).contains(&v) {
        return None;
    }
    let g = *shapes[3].get(3)?;
    if ![1, k].contains(&g)
        || shapes[1] != [t, b, h, k]
        || shapes[2] != [t, b, h, v]
        || shapes[3] != [t, b, h, g]
        || shapes[4] != [t, b, h]
    {
        return None;
    }
    let nt = (v + 3) / 4;
    let state = b as i128 * h as i128 * k as i128 * v as i128;
    let sequence = t as i128 * b as i128 * h as i128 * k.max(v) as i128;
    let grid = b as i128 * h as i128 * nt as i128 * k.max(32) as i128;
    if state > u32::MAX as i128
        || sequence > u32::MAX as i128
        || (history && state * t as i128 > u32::MAX as i128)
        || grid > i32::MAX as i128
    {
        return None;
    }
    Some(StateGeometry {
        b,
        h,
        k,
        v,
        t,
        g,
        nt,
        grid: grid as i32,
    })
}

/// Recompute state-only or previous-state histories without expanding time steps.
pub fn emit_state(
    ctx: &mut TranslationContext,
    inputs: &[mlx::mlx_array],
    history: bool,
    in_reverse: bool,
    out_reverse: bool,
) -> Result<Option<Vec<mlx::mlx_array>>, MlxError> {
    if inputs.iter().any(|&x| ctx.dtype_of(x) != F32) {
        return Ok(None);
    }
    let shapes: Vec<_> = inputs.iter().map(|&x| ctx.shape_of(x)).collect();
    let Some(c) = state_geometry(&shapes, history) else {
        return Ok(None);
    };
    let name = CString::new(if history {
        "onnxrt_gated_delta_replay_scan"
    } else {
        "onnxrt_gated_delta_state_scan"
    })
    .unwrap();
    let header = CString::new("").unwrap();
    let source = CString::new(
        STATE_SOURCE.replace("HISTORY_WRITE", if history { HISTORY_WRITE } else { "" }),
    )
    .unwrap();
    let inn = Strings::new(&["initial", "keys", "values", "decays", "betas"])?;
    let outn = Strings::new(if history {
        &["final", "previous"]
    } else {
        &["final"]
    })?;
    let kernel = unsafe {
        mlx::mlx_fast_metal_kernel_new(
            name.as_ptr(),
            inn.0,
            outn.0,
            source.as_ptr(),
            header.as_ptr(),
            true,
            false,
        )
    };
    if kernel.ctx.is_null() {
        return Ok(None);
    }
    let config = unsafe { mlx::mlx_fast_metal_kernel_config_new() };
    let mut rc = 0;
    for (key, value) in [
        ("T", c.t),
        ("B", c.b),
        ("H", c.h),
        ("K", c.k),
        ("V", c.v),
        ("G", c.g),
        ("NT", c.nt),
        ("IN_REVERSE", in_reverse as i32),
        ("OUT_REVERSE", out_reverse as i32),
    ] {
        let key = CString::new(key).unwrap();
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_int(config, key.as_ptr(), value)
        };
    }
    let mut outs = vec![vec![c.b, c.h, c.k, c.v]];
    if history {
        outs.push(vec![c.t, c.b, c.h, c.k, c.v]);
    }
    for shape in &outs {
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_output_arg(
                config,
                shape.as_ptr(),
                shape.len(),
                F32,
            )
        };
    }
    let threads = c.k.max(32);
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config, c.grid, 1, 1) };
    rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_thread_group(config, threads, 1, 1) };
    let mut args = crate::mlx::VectorArray::new();
    for &x in inputs {
        args.append(x);
    }
    let mut outputs = crate::mlx::VectorArray::new();
    if rc == 0 {
        rc = unsafe {
            mlx::mlx_fast_metal_kernel_apply(
                outputs.as_mut_ptr(),
                kernel,
                args.as_raw(),
                config,
                ctx.stream(),
            )
        };
    }
    unsafe {
        mlx::mlx_fast_metal_kernel_config_free(config);
        mlx::mlx_fast_metal_kernel_free(kernel);
    }
    if rc != 0 {
        return Err("gated-delta state Scan kernel failed".into());
    }
    Ok(Some(
        (0..outs.len())
            .map(|index| ctx.keep(outputs.get(index)))
            .collect(),
    ))
}

#[cfg(test)]
mod state_tests {
    use super::state_geometry;
    fn shapes(t: i32, k: i32, v: i32, g: i32) -> Vec<Vec<i32>> {
        vec![
            vec![1, 2, k, v],
            vec![t, 1, 2, k],
            vec![t, 1, 2, v],
            vec![t, 1, 2, g],
            vec![t, 1, 2],
        ]
    }
    #[test]
    fn covered_state_geometry_and_tails() {
        for k in [16, 32, 64, 128] {
            for v in [1, 9, 128] {
                for g in [1, k] {
                    for history in [false, true] {
                        let c = state_geometry(&shapes(32, k, v, g), history).unwrap();
                        assert_eq!(c.nt, (v + 3) / 4);
                        assert_eq!(c.grid, 2 * c.nt * k.max(32));
                    }
                }
            }
        }
    }
    #[test]
    fn uncovered_and_overflowing_geometry_declines() {
        for s in [
            shapes(0, 32, 9, 1),
            shapes(7, 17, 9, 1),
            shapes(7, 32, 129, 1),
            shapes(7, 32, 9, 2),
        ] {
            assert!(state_geometry(&s, true).is_none());
        }
        let mut s = shapes(i32::MAX, 128, 128, 1);
        assert!(state_geometry(&s, true).is_none());
        s[0][0] = 0;
        assert!(state_geometry(&s, false).is_none());
        assert!(state_geometry(&[], false).is_none());
    }
}
