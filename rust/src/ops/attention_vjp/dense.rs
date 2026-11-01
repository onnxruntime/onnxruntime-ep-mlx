//! GEMM pullback for score matrices within an explicit workspace budget.
use super::Config;
use crate::engine::MlxError;
use crate::mlx::Array;
use crate::sys::mlx;

type Raw = mlx::mlx_array;
type Binary = unsafe extern "C" fn(*mut Raw, Raw, Raw, mlx::mlx_stream) -> i32;
struct Math {
    stream: mlx::mlx_stream,
    owned: Vec<Array>,
}
impl Math {
    fn keep(&mut self, a: Array) -> Raw {
        let raw = a.as_raw();
        self.owned.push(a);
        raw
    }
    fn emit(&mut self, f: impl FnOnce(*mut Raw, mlx::mlx_stream) -> i32) -> Result<Raw, MlxError> {
        let mut raw = unsafe { mlx::mlx_array_new() };
        let rc = f(&mut raw, self.stream);
        let a = Array::from_raw(raw);
        if rc != 0 {
            return Err("AttentionVJP dense operation failed".into());
        }
        Ok(self.keep(a))
    }
    fn scalar(&mut self, value: f32) -> Raw {
        self.keep(Array::from_raw(unsafe {
            mlx::mlx_array_new_float32(value)
        }))
    }
    fn binary(&mut self, op: Binary, a: Raw, b: Raw) -> Result<Raw, MlxError> {
        self.emit(|out, s| unsafe { op(out, a, b, s) })
    }
    fn reshape(&mut self, a: Raw, shape: &[i32]) -> Result<Raw, MlxError> {
        self.emit(|out, s| unsafe { mlx::mlx_reshape(out, a, shape.as_ptr(), shape.len(), s) })
    }
    fn transpose(&mut self, a: Raw) -> Result<Raw, MlxError> {
        self.emit(|out, s| unsafe { mlx::mlx_swapaxes(out, a, -1, -2, s) })
    }
    fn sum(&mut self, a: Raw, axis: i32, keep: bool) -> Result<Raw, MlxError> {
        self.emit(|out, s| unsafe { mlx::mlx_sum_axis(out, a, axis, keep, s) })
    }
    fn where_(&mut self, c: Raw, a: Raw, b: Raw) -> Result<Raw, MlxError> {
        self.emit(|out, s| unsafe { mlx::mlx_where(out, c, a, b, s) })
    }
    fn take(&mut self, a: Raw) -> Array {
        let i = self
            .owned
            .iter()
            .position(|x| x.as_raw().ctx == a.ctx)
            .unwrap();
        self.owned.swap_remove(i)
    }
}

pub(super) fn execute(
    inputs: &[Raw],
    geometry: [i32; 3],
    mask_shape: &[i32],
    c: &Config,
    stream: mlx::mlx_stream,
) -> Result<Vec<Array>, MlxError> {
    let [b, sq, sk] = geometry;
    let (q, k, v, y, g, mask) = (
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
    );
    let mut m = Math {
        stream,
        owned: Vec::new(),
    };
    let groups = c.hq / c.hk;
    let q = m.reshape(q, &[b, c.hk, groups, sq, c.d])?;
    let k = m.reshape(k, &[b, c.hk, 1, sk, c.d])?;
    let v = m.reshape(v, &[b, c.hk, 1, sk, c.d])?;
    let g = m.reshape(g, &[b, c.hk, groups, sq, c.d])?;
    let root = m.scalar(c.scale.sqrt());
    let qs = m.binary(mlx::mlx_multiply, q, root)?;
    let ks = m.binary(mlx::mlx_multiply, k, root)?;
    let kt = m.transpose(ks)?;
    let raw = m.binary(mlx::mlx_matmul, qs, kt)?;
    let mut scores = m.reshape(raw, &[b, c.hq, sq, sk])?;
    let zero = m.scalar(0.);
    let neg = m.scalar(f32::NEG_INFINITY);
    if c.mask != 0 {
        let mut bias = if c.mask == 1 {
            m.where_(mask, zero, neg)?
        } else {
            mask
        };
        let width = *mask_shape.last().unwrap();
        if width != 1 && width < sk {
            let axis = [mask_shape.len() as i32 - 1];
            let low = [0];
            let high = [sk - width];
            bias = m.emit(|out, s| unsafe {
                mlx::mlx_pad(
                    out,
                    bias,
                    axis.as_ptr(),
                    1,
                    low.as_ptr(),
                    1,
                    high.as_ptr(),
                    1,
                    neg,
                    c"constant".as_ptr(),
                    s,
                )
            })?;
        }
        scores = m.binary(mlx::mlx_add, scores, bias)?;
    }
    if c.causal != 0 {
        let qi = m.emit(|out, s| unsafe {
            mlx::mlx_arange(out, 0., f64::from(sq), 1., mlx::mlx_dtype__MLX_INT32, s)
        })?;
        let qi = m.reshape(qi, &[sq, 1])?;
        let ki = m.emit(|out, s| unsafe {
            mlx::mlx_arange(out, 0., f64::from(sk), 1., mlx::mlx_dtype__MLX_INT32, s)
        })?;
        let allowed = m.binary(mlx::mlx_less_equal, ki, qi)?;
        scores = m.where_(allowed, scores, neg)?;
    }
    let maximum = m.emit(|out, s| unsafe { mlx::mlx_max_axis(out, scores, -1, true, s) })?;
    let empty = m.binary(mlx::mlx_equal, maximum, neg)?;
    let safe = m.where_(empty, zero, scores)?;
    let probability = m.emit(|out, s| unsafe { mlx::mlx_softmax_axis(out, safe, -1, true, s) })?;
    let p = m.where_(empty, zero, probability)?;
    let p = m.reshape(p, &[b, c.hk, groups, sq, sk])?;
    let pt = m.transpose(p)?;
    let dv = m.binary(mlx::mlx_matmul, pt, g)?;
    let dv = m.sum(dv, 2, false)?;
    let vt = m.transpose(v)?;
    let dp = m.binary(mlx::mlx_matmul, g, vt)?;
    let y = m.reshape(y, &[b, c.hk, groups, sq, c.d])?;
    let yg = m.binary(mlx::mlx_multiply, y, g)?;
    let delta = m.sum(yg, -1, true)?;
    let centered = m.binary(mlx::mlx_subtract, dp, delta)?;
    let ds = m.binary(mlx::mlx_multiply, p, centered)?;
    let dq = m.binary(mlx::mlx_matmul, ds, ks)?;
    let dq = m.binary(mlx::mlx_multiply, dq, root)?;
    let dq = m.reshape(dq, &[b, c.hq, sq, c.d])?;
    let dst = m.transpose(ds)?;
    let dk = m.binary(mlx::mlx_matmul, dst, qs)?;
    let dk = m.binary(mlx::mlx_multiply, dk, root)?;
    let dk = m.sum(dk, 2, false)?;
    // Only the three gradients escape. Intermediate score buffers can be freed
    // after their final consumer instead of entering the session-wide arena.
    Ok(vec![m.take(dq), m.take(dk), m.take(dv)])
}
