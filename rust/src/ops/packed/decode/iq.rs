//! IQ decoders split by codebook family, preserving full-width code, sign, and scale fields.

use half::f16;

use super::super::{PackedFormat, iq_tables};
use super::{read_f16, read_le_uint, unpack_bit_planes};

const IQ4: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

pub(super) fn decode(format: PackedFormat, block: &[u8]) -> Vec<f32> {
    match format {
        PackedFormat::Iq2Xxs => decode_iq2_xxs(block),
        PackedFormat::Iq2Xs => decode_iq2_xs(block),
        PackedFormat::Iq2S => decode_iq2_s(block),
        PackedFormat::Iq3Xxs => decode_iq3_xxs(block),
        PackedFormat::Iq3S => decode_iq3_s(block),
        PackedFormat::Iq1S => decode_iq1_s(block),
        PackedFormat::Iq1M => decode_iq1_m(block),
        PackedFormat::Iq4Nl => decode_iq4_nl(block),
        PackedFormat::Iq4Xs => decode_iq4_xs(block),
        _ => unreachable!("non-IQ format routed to IQ decoder"),
    }
}

fn signs(mut index: u32, parity: bool) -> [f32; 8] {
    if parity {
        index |= (index.count_ones() & 1) << 7;
    }
    std::array::from_fn(|bit| 1.0 - 2.0 * ((index >> bit) & 1) as f32)
}

fn decode_iq2_xxs(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 0);
    let mut output = vec![0.0f32; 256];
    for group in 0..8 {
        let aux = read_le_uint(block, 6 + 8 * group, 4) as u32;
        for lane in 0..4 {
            let i = 4 * group + lane;
            let values = iq_tables::IQ2_XXS.row(block[2 + 8 * group + lane] as usize);
            let signs = signs((aux >> (7 * lane)) & 127, true);
            let scale = (d * (0.5 + (aux >> 28) as f32)) * 0.25;
            for j in 0..8 {
                output[8 * i + j] = (scale * values[j]) * signs[j];
            }
        }
    }
    output
}

fn decode_iq2_xs(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 0);
    let mut output = vec![0.0f32; 256];
    for group in 0..8 {
        for lane in 0..4 {
            let i = 4 * group + lane;
            let word = read_le_uint(block, 2 + 2 * i, 2) as u16;
            let values = iq_tables::IQ2_XS.row((word & 511) as usize);
            let signs = signs((word >> 9) as u32, true);
            let scale = (d * (0.5 + ((block[66 + group] >> (4 * (lane / 2))) & 15) as f32)) * 0.25;
            for j in 0..8 {
                output[8 * i + j] = (scale * values[j]) * signs[j];
            }
        }
    }
    output
}

fn decode_iq2_s(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 0);
    let mut output = vec![0.0f32; 256];
    for group in 0..8 {
        for lane in 0..4 {
            let i = 4 * group + lane;
            let index =
                (block[2 + i] as usize) | ((((block[66 + group] >> (2 * lane)) & 3) as usize) << 8);
            let values = iq_tables::IQ2_S.row(index);
            let signs = signs(block[34 + i] as u32, false);
            let scale = (d * (0.5 + ((block[74 + group] >> (4 * (lane / 2))) & 15) as f32)) * 0.25;
            for j in 0..8 {
                output[8 * i + j] = (scale * values[j]) * signs[j];
            }
        }
    }
    output
}

fn decode_iq3_xxs(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 0);
    let mut output = vec![0.0f32; 256];
    for group in 0..8 {
        let aux = read_le_uint(block, 66 + 4 * group, 4) as u32;
        for lane in 0..4 {
            let i = 4 * group + lane;
            let first = iq_tables::IQ3_XXS.row(block[2 + 2 * i] as usize);
            let second = iq_tables::IQ3_XXS.row(block[3 + 2 * i] as usize);
            let signs = signs((aux >> (7 * lane)) & 127, true);
            let scale = (d * (0.5 + (aux >> 28) as f32)) * 0.5;
            for j in 0..4 {
                output[8 * i + j] = (scale * first[j]) * signs[j];
                output[8 * i + 4 + j] = (scale * second[j]) * signs[4 + j];
            }
        }
    }
    output
}

fn decode_iq3_s(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 0);
    let mut output = vec![0.0f32; 256];
    for group in 0..8 {
        for lane in 0..4 {
            let i = 4 * group + lane;
            let first_index = (block[2 + 2 * i] as usize)
                | ((((block[66 + group] >> (2 * lane)) & 1) as usize) << 8);
            let second_index = (block[3 + 2 * i] as usize)
                | ((((block[66 + group] >> (2 * lane + 1)) & 1) as usize) << 8);
            let first = iq_tables::IQ3_S.row(first_index);
            let second = iq_tables::IQ3_S.row(second_index);
            let signs = signs(block[74 + i] as u32, false);
            let scale = d * (1 + 2 * ((block[106 + group / 2] >> (4 * (group % 2))) & 15)) as f32;
            for j in 0..4 {
                output[8 * i + j] = (scale * first[j]) * signs[j];
                output[8 * i + 4 + j] = (scale * second[j]) * signs[4 + j];
            }
        }
    }
    output
}

fn decode_iq1_s(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 0);
    let mut output = vec![0.0f32; 256];
    for group in 0..8 {
        let high = read_le_uint(block, 34 + 2 * group, 2) as u16;
        for lane in 0..4 {
            let i = 4 * group + lane;
            let index = (block[2 + i] as usize) | ((((high >> (3 * lane)) & 7) as usize) << 8);
            let scale = d * (2 * ((high >> 12) & 7) + 1) as f32;
            let delta = if high & 0x8000 != 0 { -0.125 } else { 0.125 };
            let values = iq_tables::IQ1_S.row(index);
            for j in 0..8 {
                output[8 * i + j] = scale * (values[j] + delta);
            }
        }
    }
    output
}

fn decode_iq1_m(block: &[u8]) -> Vec<f32> {
    let mut scales = [0u16; 4];
    for (index, scale) in scales.iter_mut().enumerate() {
        *scale = read_le_uint(block, 48 + 2 * index, 2) as u16;
    }
    let scale_bits = scales
        .iter()
        .enumerate()
        .fold(0u16, |bits, (index, &scale)| {
            bits | ((scale >> 12) << (4 * index))
        });
    let d = f16::from_bits(scale_bits).to_f32();
    let mut output = vec![0.0f32; 256];
    for group in 0..8 {
        for lane in 0..4 {
            let i = 4 * group + lane;
            let high = block[32 + i / 2] >> (4 * (i % 2));
            let index = (block[i] as usize) | (((high & 7) as usize) << 8);
            let scale = d
                * (2 * ((scales[group / 2] >> (6 * (group % 2) + 3 * (lane / 2))) & 7) + 1) as f32;
            let delta = if high & 8 != 0 { -0.125 } else { 0.125 };
            let values = iq_tables::IQ1_S.row(index);
            for j in 0..8 {
                output[8 * i + j] = scale * (values[j] + delta);
            }
        }
    }
    output
}

fn decode_iq4_nl(block: &[u8]) -> Vec<f32> {
    let d = read_f16(block, 0);
    unpack_bit_planes(&block[2..], 4, 16)
        .into_iter()
        .map(|value| d * IQ4[value as usize])
        .collect()
}

fn decode_iq4_xs(block: &[u8]) -> Vec<f32> {
    let high = read_le_uint(block, 2, 2);
    let scales: Vec<_> = (0..8)
        .map(|index| {
            let low = (block[4 + index / 2] >> (4 * (index % 2))) & 15;
            (low | (((high >> (2 * index)) as u8 & 3) << 4)) as i32 - 32
        })
        .collect();
    let q = unpack_bit_planes(&block[8..], 4, 16);
    let d = read_f16(block, 0);
    q.into_iter()
        .enumerate()
        .map(|(index, value)| (d * scales[index / 32] as f32) * IQ4[value as usize])
        .collect()
}
