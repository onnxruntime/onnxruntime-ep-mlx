uint tid = thread_index_in_threadgroup;
uint simd = simdgroup_index_in_threadgroup;
uint tile_n = threadgroup_position_in_grid.x * 32;
uint tile_m = threadgroup_position_in_grid.y * TILE_M;
uint batch = threadgroup_position_in_grid.z;
uint expert = GROUPED ? uint(experts[batch]) : 0;
uint start = GROUPED ? uint(starts[batch]) : 0;
uint count = GROUPED ? uint(starts[batch + 1]) - start : M;
if (tile_m >= count) { return; }

threadgroup mlx_weight_tile a_tile[TILE_M * 32];
threadgroup mlx_weight_tile b_tile[32 * 32];
threadgroup float out_tile[TILE_M * 32];
simdgroup_float8x8 acc[TILE_M / 8];
for (uint row = 0; row < TILE_M / 8; ++row) {
    acc[row] = simdgroup_float8x8(0.0f);
}
for (uint base = 0; base < K; base += 32) {
    for (uint i = tid; i < TILE_M * 32; i += 128) {
        uint r = tile_m + i / 32;
        uint k = base + i % 32;
        uint route = GROUPED && r < count ? uint(routes[start + r]) : r;
        a_tile[i] = r < count && k < K
            ? mlx_weight_tile(mlx_weight_x(x, xscale, xzero, route / S, k))
            : mlx_weight_tile(0.0f);
    }
    for (uint i = tid; i < 32 * 32; i += 128) {
        uint ki = K_CONTIGUOUS ? i % 32 : i / 32;
        uint ni = K_CONTIGUOUS ? i / 32 : i % 32;
        uint k = base + ki;
        uint col = tile_n + ni;
        float weight = 0.0f;
        if (k < K && col < N) {
            weight = float(WT(mlx_weight_w(w, wscale, wzero, expert, k, col)));
        }
        b_tile[ki * 32 + ni] = mlx_weight_tile(weight);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < 32; step += 8) {
        simdgroup_matrix<mlx_weight_tile, 8, 8> b;
        simdgroup_load(b, b_tile + step * 32 + simd * 8, 32);
        for (uint row = 0; row < TILE_M / 8; ++row) {
            simdgroup_matrix<mlx_weight_tile, 8, 8> a;
            simdgroup_load(a, a_tile + row * 8 * 32 + step, 32);
            simdgroup_multiply_accumulate(acc[row], a, b, acc[row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint row = 0; row < TILE_M / 8; ++row) {
    simdgroup_store(acc[row], out_tile + row * 8 * 32 + simd * 8, 32);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < TILE_M * 32; i += 128) {
    uint r = tile_m + i / 32;
    uint col = tile_n + i % 32;
    if (r < count && col < N) {
        uint route = GROUPED ? uint(routes[start + r]) : r;
        y[size_t(route) * N + col] = WT(out_tile[i]);
    }
}
