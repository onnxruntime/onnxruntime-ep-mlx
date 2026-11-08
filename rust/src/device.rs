//! Public ORT device allocations retaining MLX arrays instead of copying boundary outputs.
//! Address reservations support ORT's offset-based memory patterns. They are identifiers, not
//! host-accessible payloads; access goes through the registered ORT data-transfer implementation.

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString, c_void};
use std::ptr;
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;

use crate::engine::{MlxError, dim_i32, mlx_dtype_from_onnx};
use crate::factory::ORT_API_VERSION;
use crate::mlx::{Array, VectorArray};
use crate::sys::{mlx, ort};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DefaultStreamKind {
    Cpu,
    Gpu,
}

fn default_stream_kind(dtype: u32, stored_dtype: mlx::mlx_dtype) -> DefaultStreamKind {
    if crate::registry::is_float64(dtype) || stored_dtype == mlx::mlx_dtype__MLX_FLOAT64 {
        DefaultStreamKind::Cpu
    } else {
        DefaultStreamKind::Gpu
    }
}

fn default_stream(kind: DefaultStreamKind) -> crate::mlx::Stream {
    match kind {
        DefaultStreamKind::Cpu => crate::mlx::Stream::new_default_cpu(),
        DefaultStreamKind::Gpu => crate::mlx::Stream::new_default_gpu(),
    }
}

struct Loaded {
    array: Array,
    stream_kind: DefaultStreamKind,
}

struct Stored {
    handle: usize,
    host_views: HashMap<(u32, Vec<i64>), usize>,
    bytes: usize,
    thread: ThreadId,
}

impl Drop for Stored {
    fn drop(&mut self) {
        // As with plan teardown, releasing a handle may happen on ORT's teardown thread.
        unsafe {
            for &handle in self.host_views.values() {
                mlx::mlx_array_free(mlx::mlx_array {
                    ctx: handle as *mut c_void,
                });
            }
            mlx::mlx_array_free(mlx::mlx_array {
                ctx: self.handle as *mut c_void,
            })
        };
    }
}

struct Allocation {
    layout: Layout,
    entries: BTreeMap<usize, Stored>,
}

static ALLOCATIONS: OnceLock<Mutex<BTreeMap<usize, Allocation>>> = OnceLock::new();

fn allocations() -> &'static Mutex<BTreeMap<usize, Allocation>> {
    ALLOCATIONS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("ONNXRUNTIME_EP_MLX_DEVICE_TENSORS")
            .is_some_and(|v| v != "0" && !v.is_empty())
    })
}

pub(crate) unsafe fn memory_info(api: &ort::OrtApi) -> Result<*mut ort::OrtMemoryInfo, MlxError> {
    let mut info = ptr::null_mut();
    check(
        api,
        unsafe {
            (api.CreateMemoryInfo_V2.unwrap())(
                c"MLX".as_ptr(),
                ort::OrtMemoryInfoDeviceType_OrtMemoryInfoDeviceType_GPU,
                0x106B,
                0,
                ort::OrtDeviceMemoryType_OrtDeviceMemoryType_DEFAULT,
                64,
                ort::OrtAllocatorType_OrtDeviceAllocator,
                &mut info,
            )
        },
        "CreateMemoryInfo_V2",
    )?;
    Ok(info)
}

pub(crate) fn check(
    api: &ort::OrtApi,
    status: *mut ort::OrtStatus,
    operation: &str,
) -> Result<(), MlxError> {
    if status.is_null() {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr((api.GetErrorMessage.unwrap())(status)) }
        .to_string_lossy()
        .into_owned();
    unsafe { (api.ReleaseStatus.unwrap())(status) };
    Err(format!("MLX device tensors: {operation} failed: {message}"))
}

pub(crate) fn status(api: &ort::OrtApi, error: MlxError) -> *mut ort::OrtStatus {
    let error = CString::new(error.replace('\0', "\\0")).unwrap();
    unsafe { (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, error.as_ptr()) }
}

fn is_device_tensor(api: &ort::OrtApi, value: *const ort::OrtValue) -> Result<bool, MlxError> {
    let mut info = ptr::null();
    check(
        api,
        unsafe { (api.GetTensorMemoryInfo.unwrap())(value, &mut info) },
        "GetTensorMemoryInfo",
    )?;
    let mut kind = ort::OrtMemoryInfoDeviceType_OrtMemoryInfoDeviceType_CPU;
    unsafe { (api.MemoryInfoGetDeviceType.unwrap())(info, &mut kind) };
    if kind != ort::OrtMemoryInfoDeviceType_OrtMemoryInfoDeviceType_GPU
        || unsafe { (api.MemoryInfoGetVendorId.unwrap())(info) } != 0x106B
    {
        return Ok(false);
    }
    let mut device = -1;
    check(
        api,
        unsafe { (api.MemoryInfoGetId.unwrap())(info, &mut device) },
        "MemoryInfoGetId",
    )?;
    Ok(device == 0
        && unsafe { (api.MemoryInfoGetDeviceMemType.unwrap())(info) }
            == ort::OrtDeviceMemoryType_OrtDeviceMemoryType_DEFAULT)
}

fn metadata(
    api: &ort::OrtApi,
    value: *const ort::OrtValue,
) -> Result<(*const c_void, Vec<i64>, u32, usize), MlxError> {
    let (shape, dtype) = crate::engine::read_ort_tensor_metadata(api, value)?;
    let mut bytes = 0;
    check(
        api,
        unsafe { (api.GetTensorSizeInBytes.unwrap())(value, &mut bytes) },
        "GetTensorSizeInBytes",
    )?;
    let mut data = ptr::null();
    check(
        api,
        unsafe { (api.GetTensorData.unwrap())(value, &mut data) },
        "GetTensorData",
    )?;
    Ok((data, shape, dtype, bytes))
}

fn load(data: *const c_void, bytes: usize, dtype: u32, shape: &[i64]) -> Result<Loaded, MlxError> {
    if bytes == 0 {
        let (shape, mlx_dtype) = if crate::registry::is_movable(dtype) {
            (
                shape
                    .iter()
                    .map(|&d| dim_i32(d))
                    .collect::<Result<Vec<_>, _>>()?,
                mlx_dtype_from_onnx(dtype),
            )
        } else if crate::engine::packed4_signed(dtype).is_some() {
            (vec![0], mlx::mlx_dtype__MLX_UINT8)
        } else {
            return Err(
                "MLX device tensors require a native numeric dtype; use CPU I/O for this tensor type"
                    .into(),
            );
        };
        return Ok(Loaded {
            array: Array::from_data(ptr::null(), &shape, mlx_dtype),
            stream_kind: default_stream_kind(dtype, mlx_dtype),
        });
    }
    let address = data as usize;
    let index = allocations()
        .lock()
        .map_err(|_| "MLX device allocation index is poisoned")?;
    let (&base, allocation) = index.range(..=address).next_back().ok_or(
        "MLX cannot import an Apple GPU tensor that is not backed by its public device allocator; copy it to CPU with the owning execution provider before passing it to MLX",
    )?;
    let offset = address - base;
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > allocation.layout.size())
    {
        return Err(
            "MLX device tensor exceeds its allocation; check shape and output bindings".into(),
        );
    }
    let entry = allocation.entries.get(&offset)
        .filter(|entry| entry.bytes == bytes)
        .ok_or("MLX device tensor has no initialized payload; upload data or bind a producer output first")?;
    if entry.thread != std::thread::current().id() {
        return Err("MLX device tensors must be consumed on their producer's run thread; keep inference and device-tensor transfers on one thread".into());
    }
    let borrowed = std::mem::ManuallyDrop::new(Array::from_raw(mlx::mlx_array {
        ctx: entry.handle as *mut c_void,
    }));
    let mut array = borrowed.retained()?;
    let stream_kind = default_stream_kind(dtype, array.dtype());
    if crate::registry::is_movable(dtype) && array.dtype() != mlx_dtype_from_onnx(dtype) {
        let actual_dtype = array.dtype();
        let declared_dtype = mlx_dtype_from_onnx(dtype);
        let mut viewed = Array::new();
        let stream = default_stream(stream_kind);
        if unsafe {
            mlx::mlx_view(
                viewed.as_mut_ptr(),
                array.as_raw(),
                declared_dtype,
                stream.as_raw(),
            )
        } != 0
        {
            return Err(format!(
                "MLX device byte transfer cannot reinterpret stored dtype {actual_dtype:?} as declared dtype {declared_dtype:?}; keep device bindings on the dtype used to create the OrtValue"
            ));
        }
        array = viewed;
    }
    if crate::registry::is_movable(dtype) && array.shape() != shape {
        let shape: Vec<i32> = shape
            .iter()
            .map(|&v| dim_i32(v))
            .collect::<Result<_, _>>()?;
        let mut reshaped = Array::new();
        let stream = default_stream(stream_kind);
        if unsafe {
            mlx::mlx_reshape(
                reshaped.as_mut_ptr(),
                array.as_raw(),
                shape.as_ptr(),
                shape.len(),
                stream.as_raw(),
            )
        } != 0
        {
            return Err("MLX device tensor view cannot preserve its declared shape".into());
        }
        return Ok(Loaded {
            array: reshaped,
            stream_kind,
        });
    }
    Ok(Loaded { array, stream_kind })
}

fn store(data: *const c_void, bytes: usize, _dtype: u32, array: &Array) -> Result<(), MlxError> {
    if array.size().checked_mul(array.itemsize()) != Some(bytes) {
        return Err(
            "MLX device output byte count differs from the ORT tensor; check dtype and shape"
                .into(),
        );
    }
    if bytes == 0 {
        return Ok(());
    }
    let address = data as usize;
    let mut index = allocations()
        .lock()
        .map_err(|_| "MLX device allocation index is poisoned")?;
    let (&base, allocation) = index.range_mut(..=address).next_back().ok_or(
        "MLX device output is not allocator-owned; allocate it using the EP's memory_info",
    )?;
    let offset = address - base;
    let end = offset
        .checked_add(bytes)
        .filter(|&end| end <= allocation.layout.size())
        .ok_or("MLX device output exceeds its allocation; check output bindings")?;
    allocation
        .entries
        .retain(|&start, old| start >= end || start + old.bytes <= offset);
    let retained = std::mem::ManuallyDrop::new(array.retained()?);
    allocation.entries.insert(
        offset,
        Stored {
            handle: retained.as_raw().ctx as usize,
            host_views: HashMap::new(),
            bytes,
            thread: std::thread::current().id(),
        },
    );
    Ok(())
}

fn loaded_array(
    api: &ort::OrtApi,
    value: *const ort::OrtValue,
) -> Result<Option<Loaded>, MlxError> {
    if !enabled() {
        return Ok(None);
    }
    if !is_device_tensor(api, value)? {
        return Ok(None);
    }
    let (data, shape, dtype, bytes) = metadata(api, value)?;
    Ok(Some(load(data, bytes, dtype, &shape)?))
}

pub(crate) fn array(
    api: &ort::OrtApi,
    value: *const ort::OrtValue,
) -> Result<Option<Array>, MlxError> {
    Ok(loaded_array(api, value)?.map(|loaded| loaded.array))
}

pub(crate) fn host_data(
    api: &ort::OrtApi,
    value: *const ort::OrtValue,
) -> Result<Option<*const c_void>, MlxError> {
    let Some(loaded) = loaded_array(api, value)? else {
        return Ok(None);
    };
    let array = loaded.array;
    let (data, shape, dtype, bytes) = metadata(api, value)?;
    if bytes == 0 {
        return Ok(Some(ptr::null()));
    }
    let address = data as usize;
    let key = (dtype, shape);
    {
        let index = allocations()
            .lock()
            .map_err(|_| "MLX device allocation index is poisoned")?;
        if let Some((&base, allocation)) = index.range(..=address).next_back()
            && let Some(entry) = allocation.entries.get(&(address - base))
            && let Some(&handle) = entry.host_views.get(&key)
        {
            let borrowed = std::mem::ManuallyDrop::new(Array::from_raw(mlx::mlx_array {
                ctx: handle as *mut c_void,
            }));
            return Ok(Some(borrowed.data_bytes().cast()));
        }
    }
    let mut contiguous = Array::new();
    let stream = default_stream(loaded.stream_kind);
    if unsafe {
        mlx::mlx_contiguous(
            contiguous.as_mut_ptr(),
            array.as_raw(),
            false,
            stream.as_raw(),
        )
    } != 0
    {
        return Err(
            "MLX device host read cannot materialize a contiguous tensor; check GPU availability"
                .into(),
        );
    }
    let mut arrays = VectorArray::new();
    arrays.append(contiguous.as_raw());
    crate::mlx::eval(&arrays)?;
    let pointer = contiguous.data_bytes().cast();
    let mut index = allocations()
        .lock()
        .map_err(|_| "MLX device allocation index is poisoned")?;
    let (&base, allocation) = index.range_mut(..=address).next_back().ok_or(
        "MLX device tensor was freed during a host read; retain the OrtValue until Run completes",
    )?;
    let entry = allocation
        .entries
        .get_mut(&(address - base))
        .ok_or("MLX device payload changed during a host read; do not mutate inputs during Run")?;
    let contiguous = std::mem::ManuallyDrop::new(contiguous);
    entry
        .host_views
        .insert(key, contiguous.as_raw().ctx as usize);
    Ok(Some(pointer))
}

pub(crate) fn assign(
    api: &ort::OrtApi,
    value: *const ort::OrtValue,
    array: &Array,
) -> Result<bool, MlxError> {
    if !enabled() {
        return Ok(false);
    }
    if !is_device_tensor(api, value)? {
        return Ok(false);
    }
    let (data, _, dtype, bytes) = metadata(api, value)?;
    store(data, bytes, dtype, array)?;
    Ok(true)
}

#[repr(C)]
struct Allocator {
    base: ort::OrtAllocator,
    api: *const ort::OrtApi,
    info: *mut ort::OrtMemoryInfo,
}

unsafe extern "C" fn allocate(_: *mut ort::OrtAllocator, size: usize) -> *mut c_void {
    let Ok(layout) = Layout::from_size_align(size.max(1), 64) else {
        log::error!(
            "MLX device allocation size {size} exceeds address space; reduce tensor capacity"
        );
        return ptr::null_mut();
    };
    let data = unsafe { alloc(layout) };
    if data.is_null() {
        log::error!("MLX device allocation of {size} bytes failed; reduce tensor capacity");
        return ptr::null_mut();
    }
    let Ok(mut index) = allocations().lock() else {
        unsafe { dealloc(data, layout) };
        log::error!("MLX device allocation index is poisoned; recreate the inference process");
        return ptr::null_mut();
    };
    index.insert(
        data as usize,
        Allocation {
            layout,
            entries: BTreeMap::new(),
        },
    );
    data.cast()
}

unsafe extern "C" fn free(_: *mut ort::OrtAllocator, data: *mut c_void) {
    if data.is_null() {
        return;
    }
    match allocations().lock() {
        Ok(mut index) => {
            if let Some(allocation) = index.remove(&(data as usize)) {
                let layout = allocation.layout;
                drop(allocation);
                unsafe { dealloc(data.cast(), layout) };
            } else {
                log::error!(
                    "MLX device Free received an unknown allocation; check tensor ownership"
                );
            }
        }
        Err(_) => {
            log::error!("MLX device allocation index is poisoned; recreate the inference process")
        }
    }
}

unsafe extern "C" fn info(allocator: *const ort::OrtAllocator) -> *const ort::OrtMemoryInfo {
    unsafe { (*(allocator.cast::<Allocator>())).info }
}

pub(crate) unsafe fn create_allocator(
    api: *const ort::OrtApi,
) -> Result<*mut ort::OrtAllocator, MlxError> {
    let memory = unsafe { memory_info(&*api) }?;
    let mut base: ort::OrtAllocator = unsafe { std::mem::zeroed() };
    base.version = ORT_API_VERSION;
    base.Alloc = Some(allocate);
    base.Reserve = Some(allocate);
    base.Free = Some(free);
    base.Info = Some(info);
    Ok(Box::into_raw(Box::new(Allocator {
        base,
        api,
        info: memory,
    }))
    .cast())
}

pub(crate) unsafe fn release_allocator(allocator: *mut ort::OrtAllocator) {
    if allocator.is_null() {
        return;
    }
    let allocator = unsafe { Box::from_raw(allocator.cast::<Allocator>()) };
    unsafe { ((*allocator.api).ReleaseMemoryInfo.unwrap())(allocator.info) };
}

#[repr(C)]
struct Transfer {
    base: ort::OrtDataTransferImpl,
    api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    memory_info: *mut ort::OrtMemoryInfo,
    memory_device: *const ort::OrtMemoryDevice,
}

unsafe extern "C" fn release_transfer(transfer: *mut ort::OrtDataTransferImpl) {
    let transfer = unsafe { Box::from_raw(transfer.cast::<Transfer>()) };
    if !transfer.memory_info.is_null() {
        unsafe { ((*transfer.api).ReleaseMemoryInfo.unwrap())(transfer.memory_info) };
    }
}

unsafe extern "C" fn can_copy(
    transfer: *const ort::OrtDataTransferImpl,
    source: *const ort::OrtMemoryDevice,
    destination: *const ort::OrtMemoryDevice,
) -> bool {
    let transfer = unsafe { &*(transfer.cast::<Transfer>()) };
    let ep = unsafe { &*transfer.ep_api };
    let is_cpu = |device| unsafe {
        (ep.MemoryDevice_GetDeviceType.unwrap())(device)
            == ort::OrtMemoryInfoDeviceType_OrtMemoryInfoDeviceType_CPU
    };
    let is_mlx =
        |device| unsafe { (ep.MemoryDevice_AreEqual.unwrap())(device, transfer.memory_device) };
    let source_cpu = is_cpu(source);
    let destination_cpu = is_cpu(destination);
    let source_mlx = is_mlx(source);
    let destination_mlx = is_mlx(destination);
    (source_mlx && (destination_cpu || destination_mlx)) || (destination_mlx && source_cpu)
}

unsafe extern "C" fn copy_tensors(
    transfer: *mut ort::OrtDataTransferImpl,
    sources: *mut *const ort::OrtValue,
    destinations: *mut *mut ort::OrtValue,
    _: *mut *mut ort::OrtSyncStream,
    count: usize,
) -> *mut ort::OrtStatus {
    let transfer = unsafe { &*transfer.cast::<Transfer>() };
    let api = unsafe { &*transfer.api };
    unsafe {
        crate::guard_ffi_status(transfer.api, "device tensor transfer", || {
            let result = (|| {
                for i in 0..count {
                    let source = *sources.add(i);
                    let destination = *destinations.add(i);
                    let (data, shape, dtype, bytes) = metadata(api, source)?;
                    let (dst, dst_shape, dst_dtype, dst_bytes) = metadata(api, destination)?;
                    if shape != dst_shape || dtype != dst_dtype || bytes != dst_bytes {
                        return Err(
                            "MLX device transfer requires matching shapes and dtypes".into()
                        );
                    }
                    if let Some(array) = array(api, source)? {
                        if !assign(api, destination, &array)? {
                            let tracer = crate::trace::tracer();
                            let mut arrays = VectorArray::new();
                            arrays.append(array.as_raw());
                            {
                                let _phase = tracer.phase("eval");
                                let _region = tracer.eval_region();
                                crate::mlx::eval(&arrays)?;
                            }
                            let started = tracer.active().then(std::time::Instant::now);
                            if bytes > 0 {
                                ptr::copy(array.data_bytes(), dst.cast_mut().cast(), bytes);
                            }
                            if let Some(started) = started {
                                tracer.record_copyout(false, bytes as u64, started.elapsed());
                            }
                        }
                    } else {
                        let (shape, dtype) = if crate::registry::is_movable(dtype) {
                            (
                                shape
                                    .iter()
                                    .map(|&v| dim_i32(v))
                                    .collect::<Result<Vec<_>, _>>()?,
                                mlx_dtype_from_onnx(dtype),
                            )
                        } else if crate::engine::packed4_signed(dtype).is_some() {
                            (
                                vec![
                                    i32::try_from(bytes)
                                        .map_err(|_| "MLX packed transfer exceeds int32")?,
                                ],
                                mlx::mlx_dtype__MLX_UINT8,
                            )
                        } else {
                            return Err("MLX device tensors require numeric data; use CPU I/O for string tensors".into());
                        };
                        let array = Array::from_data(data, &shape, dtype);
                        if !assign(api, destination, &array)? {
                            return Err(
                                "MLX transfer destination must use the MLX device allocator".into(),
                            );
                        }
                    }
                }
                Ok(())
            })();
            result.map_or_else(|e| status(api, e), |()| ptr::null_mut())
        })
    }
}

fn create_transfer_with_enabled(
    api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    enabled: bool,
) -> Result<*mut ort::OrtDataTransferImpl, MlxError> {
    if !enabled {
        return Ok(ptr::null_mut());
    }
    let memory_info = unsafe { memory_info(&*api)? };
    let memory_device = unsafe { ((*ep_api).MemoryInfo_GetMemoryDevice.unwrap())(memory_info) };
    if memory_device.is_null() {
        unsafe { ((*api).ReleaseMemoryInfo.unwrap())(memory_info) };
        return Err(
            "MLX device transfer could not obtain its registered memory-device identity; use ONNX Runtime 1.31 or newer"
                .into(),
        );
    }
    let mut base: ort::OrtDataTransferImpl = unsafe { std::mem::zeroed() };
    base.ort_version_supported = ORT_API_VERSION;
    base.Release = Some(release_transfer);
    base.CanCopy = Some(can_copy);
    base.CopyTensors = Some(copy_tensors);
    Ok(Box::into_raw(Box::new(Transfer {
        base,
        api,
        ep_api,
        memory_info,
        memory_device,
    }))
    .cast())
}

pub(crate) fn create_transfer(
    api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
) -> Result<*mut ort::OrtDataTransferImpl, MlxError> {
    create_transfer_with_enabled(api, ep_api, enabled())
}

#[cfg(test)]
mod tests {
    use super::*;

    static CPU_TOKEN: u8 = 1;
    static MLX_TOKEN: u8 = 2;
    static FOREIGN_TOKEN: u8 = 3;
    static EQUIVALENT_TOKEN: u8 = 4;

    fn device(token: &'static u8) -> *const ort::OrtMemoryDevice {
        std::ptr::from_ref(token).cast()
    }

    fn cpu() -> *const ort::OrtMemoryDevice {
        device(&CPU_TOKEN)
    }

    fn mlx() -> *const ort::OrtMemoryDevice {
        device(&MLX_TOKEN)
    }

    fn foreign() -> *const ort::OrtMemoryDevice {
        device(&FOREIGN_TOKEN)
    }

    fn equivalent() -> *const ort::OrtMemoryDevice {
        device(&EQUIVALENT_TOKEN)
    }

    unsafe extern "C" fn mock_device_type(
        device: *const ort::OrtMemoryDevice,
    ) -> ort::OrtMemoryInfoDeviceType {
        if device == cpu() {
            ort::OrtMemoryInfoDeviceType_OrtMemoryInfoDeviceType_CPU
        } else {
            ort::OrtMemoryInfoDeviceType_OrtMemoryInfoDeviceType_GPU
        }
    }

    unsafe extern "C" fn mock_device_equal(
        left: *const ort::OrtMemoryDevice,
        right: *const ort::OrtMemoryDevice,
    ) -> bool {
        left == right
            || (left == equivalent() && right == mlx())
            || (left == mlx() && right == equivalent())
    }

    #[test]
    fn disabled_device_tensors_do_not_create_a_data_transfer() {
        let transfer = create_transfer_with_enabled(ptr::null(), ptr::null(), false).unwrap();
        assert!(transfer.is_null());
    }

    #[test]
    fn can_copy_uses_ort_memory_device_identity() {
        let mut ep_api: ort::OrtEpApi = unsafe { std::mem::zeroed() };
        ep_api.MemoryDevice_GetDeviceType = Some(mock_device_type);
        ep_api.MemoryDevice_AreEqual = Some(mock_device_equal);
        let transfer = Transfer {
            base: unsafe { std::mem::zeroed() },
            api: ptr::null(),
            ep_api: &ep_api,
            memory_info: ptr::null_mut(),
            memory_device: mlx(),
        };
        let transfer = (&transfer as *const Transfer).cast::<ort::OrtDataTransferImpl>();

        assert!(unsafe { can_copy(transfer, cpu(), mlx()) });
        assert!(unsafe { can_copy(transfer, mlx(), cpu()) });
        assert!(unsafe { can_copy(transfer, mlx(), mlx()) });
        assert!(!unsafe { can_copy(transfer, cpu(), cpu()) });
        assert!(!unsafe { can_copy(transfer, cpu(), foreign()) });
        assert!(!unsafe { can_copy(transfer, foreign(), cpu()) });

        // OrtMemoryDevice identity has no allocator name or owner. An equivalent descriptor can
        // pass CanCopy, so CopyTensors must still validate MLX allocation ownership.
        assert!(unsafe { can_copy(transfer, equivalent(), cpu()) });
    }

    #[test]
    fn foreign_gpu_pointer_is_rejected_before_mlx_host_import() {
        let result = load(
            std::ptr::dangling::<u8>().cast(),
            std::mem::size_of::<f32>(),
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &[1],
        );
        let Err(error) = result else {
            panic!("foreign GPU pointer unexpectedly imported");
        };
        assert!(error.contains("not backed by its public device allocator"));
        assert!(error.contains("copy it to CPU with the owning execution provider"));
    }

    #[test]
    fn float64_device_materialization_uses_cpu_streams() {
        assert_eq!(
            default_stream_kind(
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                mlx::mlx_dtype__MLX_FLOAT64,
            ),
            DefaultStreamKind::Cpu
        );
        assert_eq!(
            default_stream_kind(
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                mlx::mlx_dtype__MLX_FLOAT64,
            ),
            DefaultStreamKind::Cpu
        );
        assert_eq!(
            default_stream_kind(
                ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                mlx::mlx_dtype__MLX_FLOAT32,
            ),
            DefaultStreamKind::Gpu
        );
    }
}
