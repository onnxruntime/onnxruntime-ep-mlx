//! MX and NV FP4 block decoders.

use super::super::PackedFormat;
use super::unpack_bit_planes;

const FP4: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

pub(super) fn decode(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    match format {
        PackedFormat::Mxfp4 => decode_mxfp4(block),
        PackedFormat::Nvfp4 => decode_nvfp4(block),
        _ => unreachable!("non-FP4 format routed to FP4 decoder"),
    }
}

fn decode_mxfp4(block: &[u8]) -> Vec<f32> {
    let exponent = block[0];
    let bits = if exponent <= 1 {
        0x0020_0000u32 << exponent
    } else {
        (exponent as u32 - 1) << 23
    };
    let d = f32::from_bits(bits);
    unpack_bit_planes(&block[1..], 4, 16)
        .into_iter()
        .map(|value| d * FP4[value as usize])
        .collect()
}

fn decode_nvfp4(block: &[u8]) -> Vec<f32> {
    let mut output = Vec::with_capacity(64);
    for group in 0..4 {
        let encoded = block[group];
        let exponent = (encoded >> 3) & 15;
        let mantissa = encoded & 7;
        let d = if encoded == 0 || encoded == 127 {
            0.0
        } else if exponent == 0 {
            (mantissa as f32) * 2f32.powi(-10)
        } else {
            (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 8)
        };
        output.extend(
            unpack_bit_planes(&block[4 + 8 * group..12 + 8 * group], 4, 8)
                .into_iter()
                .map(|value| d * FP4[value as usize]),
        );
    }
    output
}
