//! `MlxEpFactory` — our implementation of the ORT `OrtEpFactory` C-ABI vtable.
//!
//! The ORT struct is embedded as the FIRST field so a `*OrtEpFactory` handed to
//! ORT is pointer-identical to our `*MlxEpFactory` (repr(C), offset 0). We fill
//! only the entry points we need; everything else stays `None` (zeroed).

use std::ffi::{CString, c_char, c_void};
use std::ptr;

use crate::ep::MlxEp;
use crate::sys::ort;

pub const ORT_API_VERSION: u32 = 31;

const _: () = assert!(
    ORT_API_VERSION == ort::ORT_API_VERSION,
    "Build onnxruntime-ep-mlx with ONNX Runtime 1.31 headers"
);

#[repr(C)]
pub struct MlxEpFactory {
    base: ort::OrtEpFactory,
    pub ort_api: *const ort::OrtApi,
    pub ep_api: *const ort::OrtEpApi,
    name: CString,
    vendor: CString,
    version: CString,
    allocator_infos: std::sync::Mutex<Vec<usize>>,
}

impl MlxEpFactory {
    pub fn new(
        registration_name: *const c_char,
        ort_api: *const ort::OrtApi,
        ep_api: *const ort::OrtEpApi,
    ) -> Box<MlxEpFactory> {
        let name = unsafe {
            if registration_name.is_null() || *registration_name == 0 {
                c"MLXExecutionProvider".to_owned()
            } else {
                std::ffi::CStr::from_ptr(registration_name).to_owned()
            }
        };
        let mut base: ort::OrtEpFactory = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetVendor = Some(get_vendor);
        base.GetVendorId = Some(get_vendor_id);
        base.GetVersion = Some(get_version);
        base.GetSupportedDevices = Some(get_supported_devices);
        base.CreateEp = Some(create_ep);
        base.ReleaseEp = Some(release_ep);
        base.CreateAllocator = Some(create_allocator);
        base.ReleaseAllocator = Some(release_allocator);
        base.CreateDataTransfer = Some(create_data_transfer);
        base.CreateSyncStreamForDevice = Some(create_sync_stream_for_device);
        base.IsStreamAware = Some(is_stream_aware);

        Box::new(MlxEpFactory {
            base,
            ort_api,
            ep_api,
            name,
            vendor: c"onnxruntime-mlx".to_owned(),
            // Single-sourced from [package].version so the EP reports its real version to ORT
            // (GetVersion). Our version scheme encodes ORT compat: 0.<ORT_API_VERSION>.<patch>,
            // so 0.31.x requires ORT 1.31 or newer (ORT_API_VERSION 31).
            version: CString::new(env!("CARGO_PKG_VERSION"))
                .expect("CARGO_PKG_VERSION contains an interior nul"),
            allocator_infos: std::sync::Mutex::new(Vec::new()),
        })
    }

    pub fn as_ptr(self: Box<Self>) -> *mut ort::OrtEpFactory {
        Box::into_raw(self) as *mut ort::OrtEpFactory
    }
}

#[inline]
unsafe fn this(p: *const ort::OrtEpFactory) -> *const MlxEpFactory {
    p as *const MlxEpFactory
}

unsafe extern "C" fn get_name(p: *const ort::OrtEpFactory) -> *const c_char {
    unsafe { (*this(p)).name.as_ptr() }
}
unsafe extern "C" fn get_vendor(p: *const ort::OrtEpFactory) -> *const c_char {
    unsafe { (*this(p)).vendor.as_ptr() }
}
unsafe extern "C" fn get_vendor_id(_p: *const ort::OrtEpFactory) -> u32 {
    0x106B // Apple
}
unsafe extern "C" fn get_version(p: *const ort::OrtEpFactory) -> *const c_char {
    unsafe { (*this(p)).version.as_ptr() }
}
unsafe extern "C" fn is_stream_aware(_p: *const ort::OrtEpFactory) -> bool {
    false
}

unsafe extern "C" fn get_supported_devices(
    p: *mut ort::OrtEpFactory,
    devices: *const *const ort::OrtHardwareDevice,
    num_devices: usize,
    ep_devices: *mut *mut ort::OrtEpDevice,
    max_ep_devices: usize,
    num_ep_devices: *mut usize,
) -> *mut ort::OrtStatus {
    let api = unsafe { (*this(p)).ort_api };
    unsafe {
        crate::guard_ffi_status(api, "get_supported_devices", || {
            get_supported_devices_impl(
                p,
                devices,
                num_devices,
                ep_devices,
                max_ep_devices,
                num_ep_devices,
            )
        })
    }
}

unsafe fn get_supported_devices_impl(
    p: *mut ort::OrtEpFactory,
    devices: *const *const ort::OrtHardwareDevice,
    num_devices: usize,
    ep_devices: *mut *mut ort::OrtEpDevice,
    max_ep_devices: usize,
    num_ep_devices: *mut usize,
) -> *mut ort::OrtStatus {
    unsafe {
        let f = &*this(p);
        let ort_api = &*f.ort_api;
        let ep_api = &*f.ep_api;
        *num_ep_devices = 0;

        let hw_type = ort_api.HardwareDevice_Type.unwrap();
        let mut gpu: *const ort::OrtHardwareDevice = ptr::null();
        let mut cpu: *const ort::OrtHardwareDevice = ptr::null();
        for i in 0..num_devices {
            let dev = *devices.add(i);
            let t = hw_type(dev);
            if t == ort::OrtHardwareDeviceType_OrtHardwareDeviceType_GPU && gpu.is_null() {
                gpu = dev;
            } else if t == ort::OrtHardwareDeviceType_OrtHardwareDeviceType_CPU && cpu.is_null() {
                cpu = dev;
            }
        }
        let selected = if !gpu.is_null() { gpu } else { cpu };
        if selected.is_null() || max_ep_devices < 1 {
            return ptr::null_mut();
        }

        let mut ep_device: *mut ort::OrtEpDevice = ptr::null_mut();
        let create = ep_api.CreateEpDevice.unwrap();
        let st = create(p, selected, ptr::null(), ptr::null(), &mut ep_device);
        if !st.is_null() {
            return st;
        }
        if crate::device::enabled() {
            let mut owners =
                match f.allocator_infos.lock() {
                    Ok(owners) => owners,
                    Err(_) => return crate::device::status(
                        ort_api,
                        "MLX allocator metadata lock is poisoned; recreate the inference process"
                            .into(),
                    ),
                };
            let info = match crate::device::memory_info(ort_api) {
                Ok(info) => info,
                Err(error) => return crate::device::status(ort_api, error),
            };
            let status = (ep_api.EpDevice_AddAllocatorInfo.unwrap())(ep_device, info);
            if !status.is_null() {
                (ort_api.ReleaseMemoryInfo.unwrap())(info);
                return status;
            }
            owners.push(info as usize);
        }
        *ep_devices.add(0) = ep_device;
        *num_ep_devices = 1;
        // Device-bind notice: env-gated through the tracer so a normal (traced-off) run stays
        // silent on stderr; surfaced as a trace instant + verbose/summary line otherwise.
        let tr = crate::trace::tracer();
        if tr.active() {
            let dev = if !gpu.is_null() { "GPU" } else { "CPU" };
            log::info!("GetSupportedDevices: bound to {dev} device");
        }
        ptr::null_mut()
    }
}

unsafe extern "C" fn create_ep(
    p: *mut ort::OrtEpFactory,
    devices: *const *const ort::OrtHardwareDevice,
    ep_metadata: *const *const ort::OrtKeyValuePairs,
    num_devices: usize,
    session_options: *const ort::OrtSessionOptions,
    logger: *const ort::OrtLogger,
    ep: *mut *mut ort::OrtEp,
) -> *mut ort::OrtStatus {
    let api = unsafe { (*this(p)).ort_api };
    unsafe {
        crate::guard_ffi_status(api, "create_ep", || {
            create_ep_impl(
                p,
                devices,
                ep_metadata,
                num_devices,
                session_options,
                logger,
                ep,
            )
        })
    }
}

unsafe fn create_ep_impl(
    p: *mut ort::OrtEpFactory,
    _devices: *const *const ort::OrtHardwareDevice,
    _ep_metadata: *const *const ort::OrtKeyValuePairs,
    num_devices: usize,
    session_options: *const ort::OrtSessionOptions,
    logger: *const ort::OrtLogger,
    ep: *mut *mut ort::OrtEp,
) -> *mut ort::OrtStatus {
    unsafe {
        let f = &*this(p);
        *ep = ptr::null_mut();
        if num_devices != 1 {
            return ((*f.ort_api).CreateStatus.unwrap())(
                ort::OrtErrorCode_ORT_INVALID_ARGUMENT,
                c"MLXExecutionProvider expects exactly one device".as_ptr(),
            );
        }
        let enable_compile =
            match crate::config::boolean(&*f.ort_api, session_options, c"ep.mlx.enable_compile") {
                Ok(value) => value.unwrap_or(true),
                Err(status) => return status,
            };
        let status = crate::memory::configure_cache(&*f.ort_api, session_options);
        if !status.is_null() {
            return status;
        }
        let memory = if crate::device::enabled() {
            match crate::device::memory_info(&*f.ort_api) {
                Ok(info) => info,
                Err(error) => return crate::device::status(&*f.ort_api, error),
            }
        } else {
            ptr::null_mut()
        };
        let mlx_ep = MlxEp::new(f.ort_api, f.ep_api, &f.name, logger, enable_compile, memory);
        *ep = mlx_ep.as_ptr();
        ptr::null_mut()
    }
}

unsafe extern "C" fn release_ep(_p: *mut ort::OrtEpFactory, ep: *mut ort::OrtEp) {
    if !ep.is_null() {
        drop(unsafe { Box::from_raw(ep as *mut MlxEp) });
    }
}

unsafe extern "C" fn create_allocator(
    p: *mut ort::OrtEpFactory,
    memory_info: *const ort::OrtMemoryInfo,
    _options: *const ort::OrtKeyValuePairs,
    allocator: *mut *mut ort::OrtAllocator,
) -> *mut ort::OrtStatus {
    unsafe { *allocator = ptr::null_mut() };
    let api = unsafe { &*(*this(p)).ort_api };
    let mut name = ptr::null();
    let status = unsafe { (api.MemoryInfoGetName.unwrap())(memory_info, &mut name) };
    if !status.is_null() {
        return status;
    }
    if !name.is_null() && unsafe { std::ffi::CStr::from_ptr(name) } == c"MLX" {
        match unsafe { crate::device::create_allocator(api) } {
            Ok(value) => unsafe { *allocator = value },
            Err(error) => return crate::device::status(api, error),
        }
    }
    ptr::null_mut()
}
unsafe extern "C" fn release_allocator(
    _p: *mut ort::OrtEpFactory,
    allocator: *mut ort::OrtAllocator,
) {
    unsafe { crate::device::release_allocator(allocator) };
}
unsafe extern "C" fn create_data_transfer(
    p: *mut ort::OrtEpFactory,
    data_transfer: *mut *mut ort::OrtDataTransferImpl,
) -> *mut ort::OrtStatus {
    let factory = unsafe { &*this(p) };
    unsafe { *data_transfer = crate::device::create_transfer(factory.ort_api, factory.ep_api) };
    ptr::null_mut()
}
unsafe extern "C" fn create_sync_stream_for_device(
    _p: *mut ort::OrtEpFactory,
    _memory_device: *const ort::OrtMemoryDevice,
    _options: *const ort::OrtKeyValuePairs,
    stream: *mut *mut ort::OrtSyncStreamImpl,
) -> *mut ort::OrtStatus {
    unsafe { *stream = ptr::null_mut() };
    ptr::null_mut()
}

/// Free a factory created in `CreateEpFactories`.
///
/// # Safety
/// `p` must be a pointer returned by `MlxEpFactory::as_ptr`.
pub unsafe fn release_factory(p: *mut ort::OrtEpFactory) {
    if !p.is_null() {
        let factory = unsafe { Box::from_raw(p as *mut MlxEpFactory) };
        if let Ok(owners) = factory.allocator_infos.lock() {
            for &info in owners.iter() {
                unsafe {
                    ((*factory.ort_api).ReleaseMemoryInfo.unwrap())(info as *mut ort::OrtMemoryInfo)
                };
            }
        } else {
            log::error!("MLX allocator metadata lock is poisoned during factory teardown");
        }
    }
}

// Silence "field never read directly" — accessed via the ABI base pointer.
const _: fn() = || {
    let _ = std::mem::size_of::<*mut c_void>();
};
