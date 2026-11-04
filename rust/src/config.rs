//! Session-local options read while ORT owns the session configuration.

use std::ffi::{CStr, CString};
use std::ptr;

use crate::sys::ort;

unsafe fn invalid(api: &ort::OrtApi, key: &CStr, detail: &str) -> *mut ort::OrtStatus {
    let message = CString::new(format!(
        "MLX session option {}: {detail}",
        key.to_string_lossy()
    ))
    .unwrap();
    unsafe { (api.CreateStatus.unwrap())(ort::OrtErrorCode_ORT_INVALID_ARGUMENT, message.as_ptr()) }
}

/// Copy a session setting; no borrowed ORT storage escapes the call.
pub(crate) unsafe fn value(
    api: &ort::OrtApi,
    options: *const ort::OrtSessionOptions,
    key: &CStr,
) -> Result<Option<String>, *mut ort::OrtStatus> {
    if options.is_null() {
        return Ok(None);
    }
    let mut present = 0;
    let status =
        unsafe { (api.HasSessionConfigEntry.unwrap())(options, key.as_ptr(), &mut present) };
    if !status.is_null() {
        return Err(status);
    }
    if present == 0 {
        return Ok(None);
    }
    let get = api.GetSessionConfigEntry.unwrap();
    let mut size = 0;
    let status = unsafe { get(options, key.as_ptr(), ptr::null_mut(), &mut size) };
    if !status.is_null() {
        return Err(status);
    }
    let mut bytes = vec![0u8; size];
    let status = unsafe { get(options, key.as_ptr(), bytes.as_mut_ptr().cast(), &mut size) };
    if !status.is_null() {
        return Err(status);
    }
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    match std::str::from_utf8(&bytes[..end]) {
        Ok(value) => Ok(Some(value.to_owned())),
        Err(_) => Err(unsafe { invalid(api, key, "value must be UTF-8") }),
    }
}

fn parse_bool(value: &str) -> Result<bool, &'static str> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err("expected 0 or 1"),
    }
}

pub(crate) unsafe fn boolean(
    api: &ort::OrtApi,
    options: *const ort::OrtSessionOptions,
    key: &CStr,
) -> Result<Option<bool>, *mut ort::OrtStatus> {
    match unsafe { value(api, options, key)? } {
        None => Ok(None),
        Some(value) => parse_bool(&value)
            .map(Some)
            .map_err(|detail| unsafe { invalid(api, key, detail) }),
    }
}

#[cfg(test)]
mod tests {
    use super::parse_bool;

    #[test]
    fn boolean_settings_accept_only_zero_and_one() {
        assert_eq!(parse_bool("0"), Ok(false));
        assert_eq!(parse_bool("1"), Ok(true));
        for value in ["", "true", "false", "2", "-1", " 1", "01"] {
            assert!(parse_bool(value).is_err(), "{value:?}");
        }
    }
}
