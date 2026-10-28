//! ORT Plugin EP ABI and lifecycle shell.
//!
//! This module is deliberately small: it owns the `OrtEp` vtable, performs the
//! minimal ABI casts and FFI panic guards, and delegates capability policy,
//! compilation state, and execution to their focused modules. Compute state owns
//! streams created on the inference thread; individual plans never retain ORT pointers.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use crate::factory::ORT_API_VERSION;
use crate::sys::ort;

#[repr(C)]
pub struct MlxEp {
    base: ort::OrtEp,
    ort_api: *const ort::OrtApi,
    ep_api: *const ort::OrtEpApi,
    name: CString,
}

impl MlxEp {
    pub fn new(
        ort_api: *const ort::OrtApi,
        ep_api: *const ort::OrtEpApi,
        name: &CStr,
        _logger: *const ort::OrtLogger,
    ) -> Box<Self> {
        let mut base: ort::OrtEp = unsafe { std::mem::zeroed() };
        base.ort_version_supported = ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetCapability = Some(get_capability);
        base.Compile = Some(compile);
        base.ReleaseNodeComputeInfos = Some(release_node_compute_infos);
        base.GetDefaultMemoryDevice = Some(get_default_memory_device);
        Box::new(Self {
            base,
            ort_api,
            ep_api,
            name: name.to_owned(),
        })
    }

    pub fn as_ptr(self: Box<Self>) -> *mut ort::OrtEp {
        Box::into_raw(self).cast()
    }
}

// Trace export is process-cumulative and is finalized on EP teardown.
impl Drop for MlxEp {
    fn drop(&mut self) {
        let tracer = crate::trace::tracer();
        tracer.log_slowest_ops();
        tracer.log_summary();
        tracer.export();
    }
}

#[inline]
unsafe fn this(ep: *const ort::OrtEp) -> *const MlxEp {
    ep.cast()
}

unsafe extern "C" fn get_name(ep: *const ort::OrtEp) -> *const c_char {
    unsafe { (*this(ep)).name.as_ptr() }
}

unsafe extern "C" fn get_default_memory_device(
    _ep: *const ort::OrtEp,
    device: *mut *const ort::OrtMemoryDevice,
) -> *mut ort::OrtStatus {
    unsafe {
        // I/O remains on ORT's CPU allocator in unified memory.
        *device = ptr::null();
        ptr::null_mut()
    }
}

unsafe extern "C" fn get_capability(
    ep: *mut ort::OrtEp,
    graph: *const ort::OrtGraph,
    support: *mut ort::OrtEpGraphSupportInfo,
) -> *mut ort::OrtStatus {
    let ep = unsafe { &*this(ep) };
    unsafe {
        crate::guard_ffi_status(ep.ort_api, "get_capability", || {
            crate::capability::get_capability(ep.ort_api, ep.ep_api, graph, support)
        })
    }
}

unsafe extern "C" fn compile(
    ep: *mut ort::OrtEp,
    graphs: *mut *const ort::OrtGraph,
    fused_nodes: *mut *const ort::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut ort::OrtNodeComputeInfo,
    _ep_context_nodes: *mut *mut ort::OrtNode,
) -> *mut ort::OrtStatus {
    let ep = unsafe { &*this(ep) };
    unsafe {
        crate::guard_ffi_status(ep.ort_api, "compile", || {
            crate::runtime::compile(ep.ort_api, graphs, fused_nodes, count, node_compute_infos)
        })
    }
}

unsafe extern "C" fn release_node_compute_infos(
    _ep: *mut ort::OrtEp,
    infos: *mut *mut ort::OrtNodeComputeInfo,
    count: usize,
) {
    unsafe { crate::runtime::release_node_compute_infos(infos, count) }
}
