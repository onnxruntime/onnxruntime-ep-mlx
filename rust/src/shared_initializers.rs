//! Weak interning of owned, immutable initializer arrays on the evaluator's run thread.
//!
//! A plan holds both a private C handle and a strong owner. The index holds only weak references,
//! so closing the last session releases the storage. Hashing includes dtype and shape; names and
//! ORT pointers are not identities. Inputs with default initializers never enter this path.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use sha2::{Digest, Sha256};

use crate::engine::{InitData, MlxError, mlx_dtype_from_onnx};
use crate::mlx::Array;
use crate::sys::mlx;

#[derive(Hash, PartialEq, Eq)]
struct Key {
    dtype: u32,
    shape: Vec<i64>,
    digest: [u8; 32],
}

type BufferIndex = HashMap<Key, Weak<Vec<u8>>>;
static BUFFERS: OnceLock<Mutex<BufferIndex>> = OnceLock::new();

fn enabled() -> bool {
    !std::env::var_os("ONNXRUNTIME_EP_MLX_NO_SHARED_INITIALIZERS").is_some_and(|value| value != "0")
}

/// Intern the plan's owned host bytes before session-local copies are retained.
/// This pool contains ordinary immutable Rust bytes and is safe across compile threads.
pub(crate) fn own(data: &[u8], dtype: u32, shape: &[i64]) -> Result<Arc<Vec<u8>>, MlxError> {
    if data.len() < 4096 || !enabled() {
        return Ok(Arc::new(data.to_vec()));
    }
    let key = Key {
        dtype,
        shape: shape.to_vec(),
        digest: Sha256::digest(data).into(),
    };
    let mut index = BUFFERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| "MLX: initializer buffer index is poisoned")?;
    if let Some(existing) = index.get(&key).and_then(Weak::upgrade) {
        return Ok(existing);
    }
    let owner = Arc::new(data.to_vec());
    index.retain(|_, value| value.strong_count() != 0);
    index.insert(key, Arc::downgrade(&owner));
    Ok(owner)
}

#[derive(Hash, PartialEq, Eq)]
struct ArrayKey {
    data: usize,
    dtype: u32,
    shape: Vec<i32>,
}

/// Owns both the MLX leaf and its host-buffer identity. A copy fallback can finish with
/// the managed payload immediately, while the array remains indexed by that host address.
pub(crate) struct SharedInitializer {
    pub(crate) array: Array,
    _storage: Arc<Vec<u8>>,
}

thread_local! {
    // The run gate already pins each plan's execution to its first run thread. No raw array
    // handle crosses threads through this index. Arc permits atomic owner teardown even when
    // ORT releases a session on a different thread; it does not make Array Send or Sync.
    static ARRAYS: RefCell<HashMap<ArrayKey, Weak<SharedInitializer>>> = RefCell::new(HashMap::new());
}

pub(crate) fn get(
    init: &InitData,
    shape: &[i32],
) -> Result<Option<Arc<SharedInitializer>>, MlxError> {
    get_typed(init, shape, mlx_dtype_from_onnx(init.dtype))
}

pub(crate) fn get_view(
    init: &InitData,
    shape: &[i32],
    dtype: mlx::mlx_dtype,
) -> Result<Option<Arc<SharedInitializer>>, MlxError> {
    if std::env::var_os("ONNXRUNTIME_EP_MLX_NO_SHARED_QUANT_VIEWS")
        .is_some_and(|value| value != "0")
    {
        return Ok(None);
    }
    get_typed(init, shape, dtype)
}

pub(crate) fn get_derived(
    data: &[u8],
    shape: &[i32],
    dtype: u32,
) -> Result<Option<Arc<SharedInitializer>>, MlxError> {
    let count = shape.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(usize::try_from(dim).ok()?)
    });
    let bytes = count.and_then(|count| {
        onnx_runtime_ir::DataType::from_onnx(dtype as i32)
            .and_then(|dtype| dtype.checked_storage_bytes(count))
    });
    if bytes != Some(data.len()) {
        return Err(format!(
            "MLX: derived initializer has {} bytes for shape {shape:?} and ONNX dtype {dtype}; \
             provide storage matching the tensor shape and dtype",
            data.len()
        ));
    }
    if !enabled()
        || data.len() < 4096
        || std::env::var_os("ONNXRUNTIME_EP_MLX_NO_SHARED_QUANT_VIEWS")
            .is_some_and(|value| value != "0")
    {
        return Ok(None);
    }
    let logical_shape: Vec<i64> = shape.iter().map(|&dim| i64::from(dim)).collect();
    let storage = own(data, dtype, &logical_shape)?;
    let init = InitData {
        data: storage.as_ptr().cast(),
        shape: logical_shape,
        dtype,
        count: count.expect("validated derived initializer element count"),
        owned: Some(storage),
    };
    get(&init, shape)
}

fn get_typed(
    init: &InitData,
    shape: &[i32],
    dtype: mlx::mlx_dtype,
) -> Result<Option<Arc<SharedInitializer>>, MlxError> {
    if !enabled() {
        return Ok(None);
    }
    let Some(storage) = &init.owned else {
        return Ok(None);
    };
    if storage.len() < 4096 {
        return Ok(None);
    }
    // The entry's explicit storage owner keeps this content-interned address stable for
    // its entire indexed lifetime, including MLX's copy fallback and foreign-thread teardown.
    let key = ArrayKey {
        data: storage.as_ptr() as usize,
        dtype,
        shape: shape.to_vec(),
    };
    ARRAYS.with(|index| {
        if let Some(existing) = index.borrow().get(&key).and_then(Weak::upgrade) {
            return Ok(Some(existing));
        }
        // Own the payload: a borrowed ORT buffer could disappear when the originating session
        // closes while another plan still uses the shared array. Publish only evaluated leaves.
        let array = Array::from_owned_data(storage.clone(), shape, dtype);
        if unsafe { mlx::mlx_array_eval(array.as_raw()) } != 0 {
            return Err("MLX: could not evaluate shared initializer".into());
        }
        #[allow(clippy::arc_with_non_send_sync)] // Thread-local access; atomic teardown, see above.
        let owner = Arc::new(SharedInitializer {
            array,
            _storage: storage.clone(),
        });
        let mut index = index.borrow_mut();
        // Remove expired keys on insert, so repeated model loads do not retain model identities.
        index.retain(|_, value| value.strong_count() != 0);
        index.insert(key, Arc::downgrade(&owner));
        Ok(Some(owner))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_pool_distinguishes_metadata_and_contents() {
        let data = vec![17; 4096];
        let first = own(&data, 1, &[1024]).unwrap();
        let same = own(&data, 1, &[1024]).unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        assert!(!Arc::ptr_eq(&first, &own(&data, 6, &[1024]).unwrap()));
        assert!(!Arc::ptr_eq(&first, &own(&data, 1, &[32, 32]).unwrap()));
        assert!(!Arc::ptr_eq(
            &first,
            &own(&vec![18; 4096], 1, &[1024]).unwrap()
        ));
        let weak = Arc::downgrade(&first);
        drop(first);
        assert!(weak.upgrade().is_some());
        drop(same);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn copied_array_keeps_its_host_identity_until_the_entry_drops() {
        let data = vec![37; 8192];
        let dtype = crate::sys::ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        let storage = own(&data, dtype, &[8192]).unwrap();
        let weak = Arc::downgrade(&storage);
        // A copied MLX leaf deliberately has no managed-payload reference to the source.
        // This models the copy fallback independently of allocator alignment or callback timing.
        let entry = SharedInitializer {
            array: Array::from_data(storage.as_ptr().cast(), &[8192], mlx_dtype_from_onnx(dtype)),
            _storage: storage.clone(),
        };
        drop(storage);
        let again = own(&data, dtype, &[8192]).unwrap();
        assert!(Arc::ptr_eq(&again, &weak.upgrade().unwrap()));
        drop(again);
        assert!(weak.upgrade().is_some());
        drop(entry);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn concurrent_compilation_uses_one_host_owner() {
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let workers = (0..2)
            .map(|_| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let buffer = own(&vec![91; 8192], 1, &[2048]).unwrap();
                    barrier.wait();
                    buffer
                })
            })
            .collect::<Vec<_>>();
        let owners = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert!(Arc::ptr_eq(&owners[0], &owners[1]));
    }

    #[test]
    fn typed_views_share_only_matching_dtype_and_shape() {
        let dtype = crate::sys::ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        let storage = own(&vec![19; 8192], dtype, &[8192]).unwrap();
        let init = InitData {
            data: storage.as_ptr().cast(),
            shape: vec![8192],
            dtype,
            count: 8192,
            owned: Some(storage),
        };
        let first = get_view(&init, &[2048], mlx::mlx_dtype__MLX_UINT32)
            .unwrap()
            .unwrap();
        let same = get_view(&init, &[2048], mlx::mlx_dtype__MLX_UINT32)
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        let other_dtype = get_view(&init, &[2048], mlx::mlx_dtype__MLX_FLOAT32)
            .unwrap()
            .unwrap();
        let other_shape = get_view(&init, &[1024, 2], mlx::mlx_dtype__MLX_UINT32)
            .unwrap()
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &other_dtype));
        assert!(!Arc::ptr_eq(&first, &other_shape));
        drop(init);
        assert!(Arc::ptr_eq(&first, &same));
    }

    #[test]
    fn derived_initializers_share_content_and_validate_storage() {
        let data = vec![19; 8192];
        let first = get_derived(&data, &[2048], 1).unwrap().unwrap();
        let same = get_derived(&data.clone(), &[2048], 1).unwrap().unwrap();
        assert!(Arc::ptr_eq(&first, &same));
        let different = get_derived(&vec![20; 8192], &[2048], 1).unwrap().unwrap();
        assert!(!Arc::ptr_eq(&first, &different));
        drop(data);
        assert!(Arc::ptr_eq(&first, &same));
        assert!(get_derived(&[0; 8], &[2048], 1).is_err());
        assert!(get_derived(&[0; 8], &[-1], 1).is_err());
    }
}
