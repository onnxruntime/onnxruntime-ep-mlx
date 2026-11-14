use std::ffi::{CStr, CString};

use crate::engine::{MlxError, TranslationContext};
use crate::mlx::VectorArray;
use crate::sys::mlx;

pub(crate) fn is_gpu(ctx: &TranslationContext) -> Result<bool, MlxError> {
    let mut device = unsafe { mlx::mlx_device_new() };
    let mut kind = mlx::mlx_device_type__MLX_CPU;
    let mut rc = unsafe { mlx::mlx_stream_get_device(&mut device, ctx.stream()) };
    if rc == 0 {
        rc = unsafe { mlx::mlx_device_get_type(&mut kind, device) };
    }
    unsafe { mlx::mlx_device_free(device) };
    if rc != 0 {
        return Err(
            "MLX Metal: cannot inspect the execution stream; verify the MLX device and Metal setup"
                .into(),
        );
    }
    Ok(kind == mlx::mlx_device_type__MLX_GPU)
}

struct KernelHandle(mlx::mlx_fast_metal_kernel);

impl Drop for KernelHandle {
    fn drop(&mut self) {
        unsafe { mlx::mlx_fast_metal_kernel_free(self.0) };
    }
}

struct ConfigHandle(mlx::mlx_fast_metal_kernel_config);

impl Drop for ConfigHandle {
    fn drop(&mut self) {
        unsafe { mlx::mlx_fast_metal_kernel_config_free(self.0) };
    }
}

pub(crate) struct Kernel<'a> {
    pub name: &'static CStr,
    pub source: &'static str,
    pub header: &'a str,
    pub templates: &'a [(&'static str, i32)],
    pub shape: &'a [i32],
    pub dtype: mlx::mlx_dtype,
    pub grid: [i32; 3],
    pub group: [i32; 3],
}

impl Kernel<'_> {
    pub fn run(
        &self,
        ctx: &mut TranslationContext,
        inputs: &[(&str, mlx::mlx_array)],
    ) -> Result<mlx::mlx_array, MlxError> {
        let names = inputs.iter().map(|(name, _)| *name).collect::<Vec<_>>();
        let names = VectorString::new(&names)?;
        let outputs = VectorString::new(&["y"])?;
        let source = CString::new(self.source).expect("static shader has no NUL");
        let header = CString::new(self.header).expect("generated numeric layout has no NUL");
        let kernel = KernelHandle(unsafe {
            mlx::mlx_fast_metal_kernel_new(
                self.name.as_ptr(),
                names.as_raw(),
                outputs.as_raw(),
                source.as_ptr(),
                header.as_ptr(),
                true,
                false,
            )
        });
        if kernel.0.ctx.is_null() {
            return Err("MLX Metal: cannot create kernel; check the MLX/Metal installation".into());
        }
        let config = ConfigHandle(unsafe { mlx::mlx_fast_metal_kernel_config_new() });
        if config.0.ctx.is_null() {
            return Err(
                "MLX Metal: cannot allocate dispatch configuration; check available memory".into(),
            );
        }
        let mut rc = unsafe {
            mlx::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                config.0,
                c"WT".as_ptr(),
                self.dtype,
            )
        };
        for &(name, value) in self.templates {
            let name = CString::new(name).expect("static template name has no NUL");
            rc |= unsafe {
                mlx::mlx_fast_metal_kernel_config_add_template_arg_int(
                    config.0,
                    name.as_ptr(),
                    value,
                )
            };
        }
        rc |= unsafe {
            mlx::mlx_fast_metal_kernel_config_add_output_arg(
                config.0,
                self.shape.as_ptr(),
                self.shape.len(),
                self.dtype,
            )
        };
        let [x, y, z] = self.grid;
        rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_grid(config.0, x, y, z) };
        let [x, y, z] = self.group;
        rc |= unsafe { mlx::mlx_fast_metal_kernel_config_set_thread_group(config.0, x, y, z) };
        let mut arrays = VectorArray::new();
        for &(_, array) in inputs {
            let array = if ctx.ndim(array) == 0 {
                ctx.reshape(array, &[1])?
            } else {
                array
            };
            arrays.append(array);
        }
        let mut output = VectorArray::new();
        if rc == 0 {
            rc = unsafe {
                mlx::mlx_fast_metal_kernel_apply(
                    output.as_mut_ptr(),
                    kernel.0,
                    arrays.as_raw(),
                    config.0,
                    ctx.stream(),
                )
            };
        }
        if rc != 0 || output.size() != 1 {
            return Err(format!(
                "MLX Metal kernel {} dispatch failed (status {rc}); check the preceding compiler/device error",
                self.name.to_string_lossy()
            ));
        }
        Ok(ctx.keep(output.get(0)))
    }
}

pub(crate) struct VectorString(mlx::mlx_vector_string);

impl VectorString {
    pub(crate) fn new(items: &[&str]) -> Result<Self, MlxError> {
        let result = Self(unsafe { mlx::mlx_vector_string_new() });
        for item in items {
            let name = CString::new(*item).map_err(|_| {
                "MLX Metal: argument name contains a NUL byte; use a valid kernel identifier"
                    .to_string()
            })?;
            if unsafe { mlx::mlx_vector_string_append_value(result.0, name.as_ptr()) } != 0 {
                return Err(
                    "MLX Metal: cannot allocate argument names; check available memory".into(),
                );
            }
        }
        Ok(result)
    }

    pub(crate) fn as_raw(&self) -> mlx::mlx_vector_string {
        self.0
    }
}

impl Drop for VectorString {
    fn drop(&mut self) {
        unsafe { mlx::mlx_vector_string_free(self.0) };
    }
}
