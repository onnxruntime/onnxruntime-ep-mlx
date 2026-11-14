uint word = thread_position_in_grid.x;
if (word < WORDS) {
    size_t start = size_t(word) * 32;
    size_t first = start / BITS;
    uint leading = uint(start - first * BITS);
    uint value = 0;
    for (size_t index = first; index <= (start + 31) / BITS; ++index) {
        device const uchar* block = packed + (index / BLOCK) * BYTES;
        uint code = uint(mlx_packed_code<FORMAT>(block, uint(index % BLOCK)) + CENTER);
        int shift = int((index - first) * BITS) - int(leading);
        value |= shift < 0 ? code >> uint(-shift) : code << uint(shift);
    }
    y[word] = value;
}
