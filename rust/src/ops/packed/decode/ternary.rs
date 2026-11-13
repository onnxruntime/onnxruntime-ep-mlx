//! Ternary TQ block decoders.

use super::super::PackedFormat;
use super::{read_f16, unpack_bit_planes};

pub(super) fn decode(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    match format {
        PackedFormat::Tq1_0 => decode_tq1_0(block),
        PackedFormat::Tq2_0 => decode_tq2_0(block),
        _ => unreachable!("non-ternary format routed to ternary decoder"),
    }
}

fn decode_tq1_0(block: &[u8]) -> Vec<f32> {
    let mut q = Vec::with_capacity(256);
    for (start, width, planes) in [(0usize, 32usize, 5usize), (32, 16, 5), (48, 4, 4)] {
        for plane in 0..planes {
            let factor = 3u16.pow(plane as u32);
            for index in 0..width {
                let wrapped = (block[start + index] as u16 * factor) & 255;
                q.push((wrapped * 3 / 256) as i32 - 1);
            }
        }
    }
    let d = read_f16(block, 52);
    q.into_iter().map(|value| d * value as f32).collect()
}

fn decode_tq2_0(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 64);
    unpack_bit_planes(&block[..64], 2, 32)
        .into_iter()
        .map(|value| d * (value - 1) as f32)
        .collect()
}
