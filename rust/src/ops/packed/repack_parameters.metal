uint group = thread_position_in_grid.x;
if (group < GROUPS) {
    size_t index = size_t(group) * 32;
    device const uchar* block = packed + (index / BLOCK) * BYTES;
    float2 parameters = mlx_packed_parameters<FORMAT>(block, uint(index % BLOCK));
    y[group] = parameters.x;
    y[GROUPS + group] = FORMAT == 1 || FORMAT == 3 ? parameters.y
        : FORMAT == 8 || FORMAT == 9 ? -parameters.y : -float(CENTER) * parameters.x;
}
