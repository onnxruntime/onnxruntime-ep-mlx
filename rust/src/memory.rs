//! Explicit process-wide MLX free-cache and GPU-residency budgets.

use std::ffi::CString;
use std::ptr;

use crate::sys::{mlx, ort};

type SetLimit = unsafe extern "C" fn(*mut usize, usize) -> std::ffi::c_int;

fn parse_limit(value: &str) -> Result<usize, &'static str> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err("must be a nonnegative decimal byte count");
    }
    value
        .parse()
        .map_err(|_| "byte count exceeds usize capacity")
}

unsafe fn invalid(api: &ort::OrtApi, label: &str, detail: &str) -> *mut ort::OrtStatus {
    let message = CString::new(format!("MLX {label} budget: {detail}")).unwrap();
    unsafe { (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_INVALID_ARGUMENT, message.as_ptr()) }
}

/// Apply only explicitly requested settings. Session config takes precedence over the environment.
/// MLX's allocator is shared by all EP sessions in this process; session teardown does not reset it.
pub(crate) unsafe fn configure(
    api: &ort::OrtApi,
    options: *const ort::OrtSessionOptions,
) -> *mut ort::OrtStatus {
    let settings: [(&std::ffi::CStr, &str, &str, SetLimit); 2] = [
        (
            c"ep.mlx.wired_limit_bytes",
            "ONNXRUNTIME_EP_MLX_WIRED_LIMIT_BYTES",
            "wired memory",
            mlx::mlx_set_wired_limit,
        ),
        (
            c"ep.mlx.cache_limit_bytes",
            "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES",
            "allocator cache",
            mlx::mlx_set_cache_limit,
        ),
    ];
    let mut limits = Vec::new();
    for (session_key, env_key, label, setter) in settings {
        let session_value = match unsafe { crate::config::value(api, options, session_key) } {
            Ok(value) => value,
            Err(status) => return status,
        };
        let value = match session_value {
            Some(value) => Some(value),
            None => match std::env::var(env_key) {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(_) => {
                    return unsafe {
                        invalid(
                            api,
                            label,
                            &format!("set {env_key} to UTF-8 decimal digits"),
                        )
                    };
                }
            },
        };
        if let Some(value) = value {
            let limit = match parse_limit(&value) {
                Ok(limit) => limit,
                Err(detail) => {
                    return unsafe {
                        invalid(
                            api,
                            label,
                            &format!("{env_key} or {} {detail}", session_key.to_string_lossy()),
                        )
                    };
                }
            };
            limits.push((env_key, label, setter, limit));
        }
    }
    for (env_key, label, setter, limit) in limits {
        let mut previous = 0;
        if unsafe { setter(&mut previous, limit) } != 0 {
            let message = CString::new(format!(
                "MLX {label} budget could not apply {limit} bytes from {env_key} or its session setting; \
                 choose a device-supported budget (wired memory requires macOS 15+ and must fit the \
                 system wired limit), then recreate the session"
            ))
            .unwrap();
            return unsafe {
                (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, message.as_ptr())
            };
        }
        log::debug!("MLX {label} budget: {limit} bytes (previous {previous})");
    }
    ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::parse_limit;

    #[test]
    fn cache_budget_accepts_zero_and_byte_counts() {
        assert_eq!(parse_limit("0"), Ok(0));
        assert_eq!(parse_limit("536870912"), Ok(512 * 1024 * 1024));
        assert_eq!(parse_limit(&usize::MAX.to_string()), Ok(usize::MAX));
    }

    #[test]
    fn cache_budget_rejects_invalid_or_overflowing_values() {
        for value in [
            "",
            "-1",
            "+1",
            "1.5",
            " 32",
            "32MiB",
            "184467440737095516160",
        ] {
            assert!(parse_limit(value).is_err(), "{value}");
        }
    }
}
