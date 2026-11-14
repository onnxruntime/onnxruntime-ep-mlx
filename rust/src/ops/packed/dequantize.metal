size_t index = size_t(thread_position_in_grid.y) * GRID_X + thread_position_in_grid.x;
size_t size = (size_t(packed_shape[0]) / BYTES) * BLOCK;
if (index < size) {
    y[index] = WT(mlx_packed_value<FORMAT, BLOCK, BYTES>(packed, index));
}
