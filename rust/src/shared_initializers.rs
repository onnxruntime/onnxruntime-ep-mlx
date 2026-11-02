//! Weak interning of owned, immutable initializer arrays on the evaluator's run thread.
//!
//! A plan holds both a private C handle and a strong owner. The index holds only weak references,
//! so closing the last session releases the storage. Hashing includes dtype and shape; names and
//! ORT pointers are not identities. Inputs with default initializers never enter this path.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use sha2::{Digest, Sha256};

use crate::engine::{InitData, MlxError, mlx_dtype_from_onnx};
use crate::mlx::Array;
use crate::sys::mlx;

#[derive(Hash, PartialEq, Eq)]
struct Key {
    dtype: u32,
    shape: Vec<i32>,
    digest: [u8; 32],
}

thread_local! {
    // The run gate already pins each plan's execution to its first run thread. No raw array
    // handle crosses threads through this index. Arc permits atomic owner teardown even when
    // ORT releases a session on a different thread; it does not make Array Send or Sync.
    static ARRAYS: RefCell<HashMap<Key, Weak<Array>>> = RefCell::new(HashMap::new());
}

pub(crate) fn get(init: &InitData, shape: &[i32]) -> Result<Option<Arc<Array>>, MlxError> {
    if std::env::var_os("ONNXRUNTIME_EP_MLX_NO_SHARED_INITIALIZERS")
        .is_some_and(|value| value != "0")
    {
        return Ok(None);
    }
    let dtype = onnx_runtime_ir::DataType::from_onnx(init.dtype as i32)
        .ok_or("MLX: unsupported shared initializer dtype")?;
    let bytes = dtype
        .checked_storage_bytes(init.count)
        .ok_or("MLX: shared initializer byte count overflows")?;
    // Keep small constants on the existing path; interning primarily targets model weights.
    if bytes < 4096 {
        return Ok(None);
    }
    if init.data.is_null() || bytes > isize::MAX as usize {
        return Err("MLX: invalid shared initializer storage".into());
    }
    // InitData describes ORT/session-owned immutable bytes, valid throughout this Compute call.
    let data = unsafe { std::slice::from_raw_parts(init.data.cast::<u8>(), bytes) };
    let key = Key {
        dtype: init.dtype,
        shape: shape.to_vec(),
        digest: Sha256::digest(data).into(),
    };
    ARRAYS.with(|index| {
        if let Some(existing) = index.borrow().get(&key).and_then(Weak::upgrade) {
            return Ok(Some(existing));
        }
        // Own the payload: a borrowed ORT buffer could disappear when the originating session
        // closes while another plan still uses the shared array. Publish only evaluated leaves.
        let array = Array::from_data(init.data, shape, mlx_dtype_from_onnx(init.dtype));
        if unsafe { mlx::mlx_array_eval(array.as_raw()) } != 0 {
            return Err("MLX: could not evaluate shared initializer".into());
        }
        #[allow(clippy::arc_with_non_send_sync)] // Thread-local access; atomic teardown, see above.
        let owner = Arc::new(array);
        let mut index = index.borrow_mut();
        // Remove expired keys on insert, so repeated model loads do not retain model identities.
        index.retain(|_, value| value.strong_count() != 0);
        index.insert(key, Arc::downgrade(&owner));
        Ok(Some(owner))
    })
}
