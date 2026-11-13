//! K-quant decoders; each helper keeps its 256-element block offsets local.

use super::super::PackedFormat;
use super::{read_f16, read_f32, unpack_bit_planes};

pub(super) fn decode(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    match format {
        PackedFormat::Q2K => decode_q2_k(block),
        PackedFormat::Q3K => decode_q3_k(block),
        PackedFormat::Q4K | PackedFormat::Q5K => decode_q4_q5_k(format, block),
        PackedFormat::Q6K => decode_q6_k(block),
        PackedFormat::Q8K => decode_q8_k(block),
        _ => unreachable!("non-K format routed to K-quant decoder"),
    }
}

fn decode_q2_k(block: &[u8]) -> Vec<f32> {
    let q = unpack_bit_planes(&block[16..80], 2, 32);
    let d = read_f16(block, 80);
    let dmin = read_f16(block, 82);
    q.into_iter()
        .enumerate()
        .map(|(index, value)| {
            let scale = block[index / 16];
            (d * (scale & 15) as f32) * value as f32 - dmin * (scale >> 4) as f32
        })
        .collect()
}

fn decode_q3_k(block: &[u8]) -> Vec<f32> {
    let mut q = unpack_bit_planes(&block[32..96], 2, 32);
    let mask = unpack_bit_planes(&block[..32], 1, 32);
    for index in 0..256 {
        q[index] -= 4 * (1 - mask[index]);
    }
    let low = unpack_bit_planes(&block[96..104], 4, 8);
    let high = unpack_bit_planes(&block[104..108], 2, 4);
    let scales: Vec<_> = low
        .into_iter()
        .zip(high)
        .map(|(low, high)| (low | (high << 4)) - 32)
        .collect();
    let d = read_f16(block, 108);
    q.into_iter()
        .enumerate()
        .map(|(index, value)| (d * scales[index / 16] as f32) * value as f32)
        .collect()
}

fn decode_q4_q5_k(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    let packed_scales = &block[4..16];
    let mut scales = [0i32; 8];
    let mut minimums = [0i32; 8];
    for index in 0..8 {
        if index < 4 {
            scales[index] = (packed_scales[index] & 63) as i32;
            minimums[index] = (packed_scales[index + 4] & 63) as i32;
        } else {
            scales[index] =
                ((packed_scales[index + 4] & 15) | ((packed_scales[index - 4] >> 6) << 4)) as i32;
            minimums[index] =
                ((packed_scales[index + 4] >> 4) | ((packed_scales[index] >> 6) << 4)) as i32;
        }
    }
    let q_offset = if format == PackedFormat::Q4K { 16 } else { 48 };
    let mut q = unpack_bit_planes(&block[q_offset..], 4, 32);
    if format == PackedFormat::Q5K {
        let high = unpack_bit_planes(&block[16..48], 1, 32);
        for index in 0..256 {
            q[index] |= high[index] << 4;
        }
    }
    let d = read_f16(block, 0);
    let dmin = read_f16(block, 2);
    q.into_iter()
        .enumerate()
        .map(|(index, value)| {
            (d * scales[index / 32] as f32) * value as f32 - dmin * minimums[index / 32] as f32
        })
        .collect()
}

fn decode_q6_k(block: &[u8]) -> Vec<f32> {
    let low = unpack_bit_planes(&block[..128], 4, 64);
    let high = unpack_bit_planes(&block[128..192], 2, 32);
    let d = read_f16(block, 208);
    (0..256)
        .map(|index| {
            let q = (low[index] | (high[index] << 4)) - 32;
            let scale = block[192 + index / 16] as i8;
            (d * scale as f32) * q as f32
        })
        .collect()
}

fn decode_q8_k(block: &[u8]) -> Vec<f32> {
    let d = read_f32(block, 0);
    block[4..260]
        .iter()
        .map(|&value| d * (value as i8) as f32)
        .collect()
}
