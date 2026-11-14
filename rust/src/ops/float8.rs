//! ONNX 8-bit float formats. MLX has no FP8 dtype, so payloads travel as uint8 bit patterns and
//! are decoded exactly to FLOAT on the device: by table lookup for standalone DequantizeLinear and
//! inline in fused weight readers.

use crate::engine::{MlxError, TranslationContext};
use crate::mlx::Array;
use crate::sys::{mlx, ort};

/// Exact FLOAT values of FP8 bit patterns via a device-side 256-entry table lookup.
pub(crate) fn decode(
    ctx: &mut TranslationContext,
    bits: mlx::mlx_array,
    format: Format,
) -> Result<mlx::mlx_array, MlxError> {
    let table = format.table();
    let table = ctx.keep(Array::from_data(
        table.as_ptr().cast(),
        &[256],
        mlx::mlx_dtype__MLX_FLOAT32,
    ));
    ctx.emit(|res, s| unsafe { mlx::mlx_take(res, table, bits, s) })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Format {
    E4M3Fn,
    E4M3FnUz,
    E5M2,
    E5M2FnUz,
}

impl Format {
    pub(crate) fn from_onnx(dtype: ort::ONNXTensorElementDataType) -> Option<Self> {
        Some(match dtype {
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN => {
                Self::E4M3Fn
            }
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ => {
                Self::E4M3FnUz
            }
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 => Self::E5M2,
            ort::ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ => {
                Self::E5M2FnUz
            }
            _ => return None,
        })
    }

    /// (mantissa bits, exponent bias)
    fn layout(self) -> (u32, u32) {
        match self {
            Self::E4M3Fn => (3, 7),
            Self::E4M3FnUz => (3, 8),
            Self::E5M2 => (2, 15),
            Self::E5M2FnUz => (2, 16),
        }
    }

    fn unsigned_zero(self) -> bool {
        matches!(self, Self::E4M3FnUz | Self::E5M2FnUz)
    }

    /// Exact FP32 value of one encoding.
    pub(crate) fn decode(self, bits: u8) -> f32 {
        let (mantissa_bits, bias) = self.layout();
        let magnitude = u32::from(bits & 0x7f);
        let exponent = magnitude >> mantissa_bits;
        let mantissa = magnitude & ((1 << mantissa_bits) - 1);
        if self.unsigned_zero() && bits == 0x80
            || self == Self::E4M3Fn && magnitude == 0x7f
            || self == Self::E5M2 && exponent == 31 && mantissa != 0
        {
            return f32::NAN;
        }
        let value = if self == Self::E5M2 && exponent == 31 {
            f32::INFINITY
        } else if exponent == 0 {
            mantissa as f32 * f32::from_bits((128 - bias - mantissa_bits) << 23)
        } else {
            f32::from_bits(((exponent + 127 - bias) << 23) | (mantissa << (23 - mantissa_bits)))
        };
        if bits & 0x80 != 0 { -value } else { value }
    }

    pub(crate) fn table(self) -> [f32; 256] {
        std::array::from_fn(|bits| self.decode(bits as u8))
    }

    /// Branch-free Metal decoder `float mlx_fp8(uint bits)` matching [`Format::decode`].
    pub(crate) fn metal(self) -> String {
        let (mantissa_bits, bias) = self.layout();
        let special = match self {
            Self::E4M3Fn => "(b & 0x7fu) == 0x7fu ? NAN : v",
            Self::E4M3FnUz | Self::E5M2FnUz => "b == 0x80u ? NAN : v",
            Self::E5M2 => "e == 31u ? (m == 0u ? copysign(INFINITY, v) : NAN) : v",
        };
        format!(
            r#"
inline float mlx_fp8(uint b) {{
    uint e = (b >> {mantissa_bits}) & {exponent_mask}u;
    uint m = b & {mantissa_mask}u;
    float v = e == 0u ? float(m) * as_type<float>({subnormal}u)
                      : as_type<float>(((e + {rebias}u) << 23) | (m << {shift}));
    v = as_type<float>(as_type<uint>(v) | ((b & 0x80u) << 24));
    return {special};
}}
"#,
            exponent_mask = (1u32 << (7 - mantissa_bits)) - 1,
            mantissa_mask = (1u32 << mantissa_bits) - 1,
            subnormal = (128 - bias - mantissa_bits) << 23,
            rebias = 127 - bias,
            shift = 23 - mantissa_bits,
        ) + &self.metal_x4()
    }

    /// `mlx_fp8x4`: decodes four little-endian payload bytes. E4M3 and E5M2 map exactly onto
    /// fp16 bit patterns (E4M3 scaled by a power of two), so two bytes decode per half2 convert.
    fn metal_x4(self) -> String {
        let pair = |body: &str, finish: &str| {
            format!(
                r#"
inline float4 mlx_fp8x4(uint raw) {{
    uint lo = raw & 0x00ff00ffu, hi = (raw >> 8) & 0x00ff00ffu;
    float2 a = float2(as_type<half2>({lo})), c = float2(as_type<half2>({hi}));
    float4 v = float4(a.x, c.x, a.y, c.y);
    return {finish};
}}
"#,
                lo = body.replace('#', "lo"),
                hi = body.replace('#', "hi"),
            )
        };
        let e4m3 = "((# & 0x007f007fu) << 7) | ((# & 0x00800080u) << 8)";
        match self {
            // NaN (S.1111.111) is the only pattern decoding to |v| = 1.875 before the 2^8 rescale.
            Self::E4M3Fn => pair(e4m3, "select(v * 256.0f, float4(NAN), abs(v) == 1.875f)"),
            // 0x80 is NaN and the only pattern producing -0.
            Self::E4M3FnUz => pair(
                e4m3,
                "select(v * 128.0f, float4(NAN), as_type<uint4>(v) == 0x80000000u)",
            ),
            Self::E5M2 => pair("# << 8", "v"),
            // FNUZ exponent 31 is finite, so fp16 reinterpretation does not apply.
            Self::E5M2FnUz => r#"
inline float4 mlx_fp8x4(uint raw) {
    return float4(mlx_fp8(raw & 0xffu), mlx_fp8((raw >> 8) & 0xffu),
                  mlx_fp8((raw >> 16) & 0xffu), mlx_fp8(raw >> 24));
}
"#
            .to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Format;

    #[test]
    fn decodes_boundaries() {
        let cases: &[(Format, u8, f32)] = &[
            (Format::E4M3Fn, 0x7e, 448.0),
            (Format::E4M3Fn, 0x01, 2f32.powi(-9)),
            (Format::E4M3Fn, 0x08, 2f32.powi(-6)),
            (Format::E4M3Fn, 0xb8, -1.0),
            (Format::E4M3FnUz, 0x7f, 240.0),
            (Format::E4M3FnUz, 0x01, 2f32.powi(-10)),
            (Format::E5M2, 0x7b, 57344.0),
            (Format::E5M2, 0x7c, f32::INFINITY),
            (Format::E5M2, 0xfc, f32::NEG_INFINITY),
            (Format::E5M2, 0x01, 2f32.powi(-16)),
            (Format::E5M2FnUz, 0x7f, 57344.0),
            (Format::E5M2FnUz, 0x01, 2f32.powi(-17)),
        ];
        for &(format, bits, value) in cases {
            assert_eq!(format.decode(bits), value, "{format:?} {bits:#x}");
        }
        assert!(Format::E4M3Fn.decode(0x80).is_sign_negative());
        assert_eq!(Format::E4M3Fn.decode(0x80), 0.0);
        for (format, nan) in [
            (Format::E4M3Fn, 0x7f),
            (Format::E4M3Fn, 0xff),
            (Format::E4M3FnUz, 0x80),
            (Format::E5M2, 0x7d),
            (Format::E5M2FnUz, 0x80),
        ] {
            assert!(format.decode(nan).is_nan(), "{format:?} {nan:#x}");
        }
    }
}
