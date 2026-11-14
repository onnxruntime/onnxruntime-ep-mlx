//! Legacy Q blocks with scale/offset metadata followed by packed quantized lanes.

use super::super::PackedFormat;
use super::{read_f16, read_le_uint, unpack_bit_planes};

pub(super) fn decode(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    match format {
        PackedFormat::Q4_0 | PackedFormat::Q4_1 | PackedFormat::Q5_0 | PackedFormat::Q5_1 => {
            decode_q4_q5(format, block)
        }
        PackedFormat::Q8_0 | PackedFormat::Q8_1 => decode_q8(format, block),
        PackedFormat::Q1_0 => decode_q1_0(block),
        PackedFormat::Q2_0 => decode_q2_0(block),
        _ => unreachable!("non-legacy format routed to legacy decoder"),
    }
}

fn decode_q4_q5(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    let affine = matches!(format, PackedFormat::Q4_1 | PackedFormat::Q5_1);
    let metadata_bytes = if affine { 4 } else { 2 };
    let mut q = if matches!(format, PackedFormat::Q5_0 | PackedFormat::Q5_1) {
        let high = read_le_uint(block, metadata_bytes, 4);
        let mut q = unpack_bit_planes(&block[metadata_bytes + 4..], 4, 16);
        for (index, value) in q.iter_mut().enumerate() {
            *value |= (((high >> index) & 1) << 4) as i32;
        }
        q
    } else {
        unpack_bit_planes(&block[metadata_bytes..], 4, 16)
    };
    let d = read_f16(block, 0);
    if !affine {
        let bias = if format == PackedFormat::Q5_0 { 16 } else { 8 };
        q.iter_mut().for_each(|value| *value -= bias);
    }
    let m = if affine { read_f16(block, 2) } else { 0.0 };
    q.into_iter()
        .map(|value| {
            let scaled = d * value as f32;
            if affine { scaled + m } else { scaled }
        })
        .collect()
}

fn decode_q8(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    let offset = if format == PackedFormat::Q8_0 { 2 } else { 4 };
    let d = read_f16(block, 0);
    block[offset..offset + 32]
        .iter()
        .map(|&value| d * (value as i8) as f32)
        .collect()
}

fn decode_q1_0(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 0);
    (0..128)
        .map(|index| {
            if (block[2 + index / 8] >> (index % 8)) & 1 != 0 {
                d
            } else {
                -d
            }
        })
        .collect()
}

fn decode_q2_0(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 0);
    unpack_bit_planes(&block[2..], 2, 1)
        .into_iter()
        .map(|value| d * (value - 1) as f32)
        .collect()
}
