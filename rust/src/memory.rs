//! Explicit process-wide MLX allocator cache budget.

use std::ffi::CString;
use std::ptr;

use crate::sys::{mlx, ort};

const SESSION_KEY: &std::ffi::CStr = c"ep.mlx.cache_limit_bytes";
const ENV_KEY: &str = "ONNXRUNTIME_EP_MLX_CACHE_LIMIT_BYTES";

fn parse_limit(value: &str) -> Result<usize, &'static str> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return Err("must be a nonnegative decimal byte count");
    }
    value
        .parse()
        .map_err(|_| "byte count exceeds usize capacity")
}

unsafe fn invalid(api: &ort::OrtApi, detail: &str) -> *mut ort::OrtStatus {
    let message = CString::new(format!("MLX allocator cache budget: {detail}")).unwrap();
    unsafe { (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_INVALID_ARGUMENT, message.as_ptr()) }
}

/// Apply only explicitly requested settings. Session config takes precedence over the environment.
/// MLX's allocator is shared by all EP sessions in this process; session teardown does not reset it.
pub(crate) unsafe fn configure_cache(
    api: &ort::OrtApi,
    options: *const ort::OrtSessionOptions,
) -> *mut ort::OrtStatus {
    let session_value = match unsafe { crate::config::value(api, options, SESSION_KEY) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let value = match session_value {
        Some(value) => Some(value),
        None => match std::env::var(ENV_KEY) {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => {
                return unsafe { invalid(api, "environment value must be UTF-8 decimal digits") };
            }
        },
    };
    let Some(value) = value else {
        return ptr::null_mut();
    };
    let limit = match parse_limit(&value) {
        Ok(limit) => limit,
        Err(detail) => return unsafe { invalid(api, detail) },
    };
    let mut previous = 0;
    if unsafe { mlx::mlx_set_cache_limit(&mut previous, limit) } != 0 {
        let message = c"MLX allocator cache budget could not be applied";
        return unsafe {
            (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_EP_FAIL, message.as_ptr())
        };
    }
    log::debug!("MLX allocator cache budget: {limit} bytes (previous {previous})");
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
