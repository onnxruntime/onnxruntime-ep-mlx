//! Byte-exact host decoders grouped by the normative packed-format families.

use half::f16;

use super::PackedFormat;

mod fp4;
mod iq;
mod k_quants;
mod legacy;
mod ternary;

fn read_f16(bytes: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]])).to_f32()
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_bits(u32::from_le_bytes(
        bytes[offset..offset + 4].try_into().unwrap(),
    ))
}

fn read_le_uint(bytes: &[u8], offset: usize, width: usize) -> u64 {
    bytes[offset..offset + width]
        .iter()
        .enumerate()
        .fold(0u64, |value, (shift, &byte)| {
            value | ((byte as u64) << (8 * shift))
        })
}

fn unpack_bit_planes(data: &[u8], bits: usize, group: usize) -> Vec<i32> {
    let mut output = Vec::with_capacity(data.len() * 8 / bits);
    let mask = (1u8 << bits) - 1;
    for bytes in data.chunks_exact(group) {
        for shift in (0..8).step_by(bits) {
            output.extend(bytes.iter().map(|&byte| ((byte >> shift) & mask) as i32));
        }
    }
    output
}

pub(super) fn block(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    match format {
        PackedFormat::Q4_0
        | PackedFormat::Q4_1
        | PackedFormat::Q5_0
        | PackedFormat::Q5_1
        | PackedFormat::Q8_0
        | PackedFormat::Q8_1
        | PackedFormat::Q1_0
        | PackedFormat::Q2_0 => legacy::decode(format, block),
        PackedFormat::Q2K
        | PackedFormat::Q3K
        | PackedFormat::Q4K
        | PackedFormat::Q5K
        | PackedFormat::Q6K
        | PackedFormat::Q8K => k_quants::decode(format, block),
        PackedFormat::Iq2Xxs
        | PackedFormat::Iq2Xs
        | PackedFormat::Iq3Xxs
        | PackedFormat::Iq1S
        | PackedFormat::Iq4Nl
        | PackedFormat::Iq3S
        | PackedFormat::Iq2S
        | PackedFormat::Iq4Xs
        | PackedFormat::Iq1M => iq::decode(format, block),
        PackedFormat::Tq1_0 | PackedFormat::Tq2_0 => ternary::decode(format, block),
        PackedFormat::Mxfp4 | PackedFormat::Nvfp4 => fp4::decode(format, block),
    }
}
