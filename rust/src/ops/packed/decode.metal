inline float mlx_packed_half(device const uchar* b, uint offset) {
    ushort bits = ushort(b[offset]) | (ushort(b[offset + 1]) << 8);
    return float(as_type<half>(bits));
}

template <int FORMAT>
inline int mlx_packed_code(device const uchar* b, uint i) {
    if (FORMAT <= 3) {
        constexpr bool affine = FORMAT == 1 || FORMAT == 3;
        constexpr bool five = FORMAT == 2 || FORMAT == 3;
        constexpr uint metadata = affine ? 4 : 2;
        constexpr uint offset = metadata + (five ? 4 : 0);
        int q = (int(b[offset + i % 16]) >> (4 * (i / 16))) & 15;
        if (five) {
            uint high = uint(b[metadata]) | (uint(b[metadata + 1]) << 8)
                | (uint(b[metadata + 2]) << 16) | (uint(b[metadata + 3]) << 24);
            q |= ((high >> i) & 1) << 4;
        }
        return affine ? q : q - (five ? 16 : 8);
    } else if (FORMAT <= 5) {
        constexpr uint offset = FORMAT == 4 ? 2 : 4;
        return int(char(b[offset + i]));
    } else {
        constexpr uint offset = FORMAT == 8 ? 16 : 48;
        uint q = (b[offset + (i / 64) * 32 + i % 32] >> (4 * ((i / 32) % 2))) & 15;
        if (FORMAT == 9) { q |= ((b[16 + i % 32] >> (i / 32)) & 1) << 4; }
        return int(q);
    }
}

template <int FORMAT>
inline float2 mlx_packed_parameters(device const uchar* b, uint i) {
    if (FORMAT <= 5) {
        return float2(mlx_packed_half(b, 0),
            FORMAT == 1 || FORMAT == 3 ? mlx_packed_half(b, 2) : 0.0f);
    }
    uint group = i / 32;
    uint scale, minimum;
    if (group < 4) {
        scale = b[4 + group] & 63;
        minimum = b[8 + group] & 63;
    } else {
        scale = (b[8 + group] & 15) | ((b[group] >> 6) << 4);
        minimum = (b[8 + group] >> 4) | ((b[4 + group] >> 6) << 4);
    }
    return float2(
        precise::fma(mlx_packed_half(b, 0), float(scale), -0.0f),
        precise::fma(mlx_packed_half(b, 2), float(minimum), -0.0f));
}

template <int FORMAT, int BLOCK, int BYTES>
inline float mlx_packed_value(device const uchar* packed, size_t index) {
    device const uchar* b = packed + (index / BLOCK) * BYTES;
    uint i = uint(index % BLOCK);
    if (FORMAT <= 3) {
        constexpr bool affine = FORMAT == 1 || FORMAT == 3;
        int q = mlx_packed_code<FORMAT>(b, i);
        // A separate rounded multiplication must precede the affine addition.
        // -0, unlike +0, preserves the sign of a zero multiplication in fma.
        float scaled = precise::fma(mlx_packed_half(b, 0), float(q), -0.0f);
        return affine ? precise::fma(1.0f, scaled, mlx_packed_half(b, 2)) : scaled;
    } else if (FORMAT <= 5) {
        return precise::fma(mlx_packed_half(b, 0), float(mlx_packed_code<FORMAT>(b, i)), -0.0f);
    } else if (FORMAT == 6) {
        uint scale = b[i / 16];
        uint q = (b[16 + (i / 128) * 32 + i % 32] >> (2 * ((i / 32) % 4))) & 3;
        float d = precise::fma(mlx_packed_half(b, 80), float(scale & 15), -0.0f);
        float value = precise::fma(d, float(q), -0.0f);
        float minimum = precise::fma(mlx_packed_half(b, 82), float(scale >> 4), -0.0f);
        return precise::fma(-1.0f, minimum, value);
    } else if (FORMAT == 7) {
        int q = (b[32 + (i / 128) * 32 + i % 32] >> (2 * ((i / 32) % 4))) & 3;
        q -= 4 * (1 - ((b[i % 32] >> (i / 32)) & 1));
        uint group = i / 16;
        uint low = (b[96 + group % 8] >> (4 * (group / 8))) & 15;
        uint high = (b[104 + group % 4] >> (2 * (group / 4))) & 3;
        float d = precise::fma(mlx_packed_half(b, 108), float(int(low | (high << 4)) - 32), -0.0f);
        return precise::fma(d, float(q), -0.0f);
    } else if (FORMAT == 8 || FORMAT == 9) {
        float2 parameters = mlx_packed_parameters<FORMAT>(b, i);
        float value = precise::fma(parameters.x, float(mlx_packed_code<FORMAT>(b, i)), -0.0f);
        return precise::fma(-1.0f, parameters.y, value);
    } else {
        uint low = (b[(i / 128) * 64 + i % 64] >> (4 * ((i / 64) % 2))) & 15;
        uint high = (b[128 + (i / 128) * 32 + i % 32] >> (2 * ((i / 32) % 4))) & 3;
        float d = precise::fma(mlx_packed_half(b, 208), float(char(b[192 + i / 16])), -0.0f);
        return precise::fma(d, float(int(low | (high << 4)) - 32), -0.0f);
    }
}
