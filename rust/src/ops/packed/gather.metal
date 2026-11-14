size_t out = thread_position_in_grid.x;
if (out < SIZE) {
    int row = int(indices[out / WIDTH]);
    if (row < 0) { row += ROWS; }
    size_t wi = size_t(row) * WIDTH + out % WIDTH;
    y[out] = WT(mlx_packed_value<FORMAT, BLOCK, BYTES>(packed, mlx_packed_source_index(wi)));
}
