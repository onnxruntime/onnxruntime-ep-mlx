// FP8 decode GEMV over K-contiguous weights [E?,N,K] (Gemm/GroupedMatMul transB=1).
// Each simdgroup owns ROWS outputs; each lane decodes 8 consecutive bytes per step and reuses
// them across TILE_M activation rows; grid z tiles routes (grouped) or M. Requires K % 8 == 0 and an 8-byte aligned payload.
uint lane = thread_index_in_simdgroup;
uint n0 = (threadgroup_position_in_grid.x * SIMDGROUPS + simdgroup_index_in_threadgroup) * ROWS;
uint route = threadgroup_position_in_grid.z;
uint expert = GROUPED ? uint(indices[route]) : 0;
uint xrow = GROUPED ? route / S : route * TILE_M;
size_t pe = size_t(expert / WD0) * WS0;
float acc[TILE_M][ROWS];
for (uint m = 0; m < TILE_M; ++m) {
    for (uint r = 0; r < ROWS; ++r) { acc[m][r] = 0.0f; }
}
for (uint k = lane * 8; k < K; k += 256) {
    float xv[TILE_M][8];
    for (uint m = 0; m < TILE_M; ++m) {
        for (uint j = 0; j < 8; ++j) { xv[m][j] = float(x[size_t(min(xrow + m, uint(M) - 1)) * K + k + j]); }
    }
    for (uint r = 0; r < ROWS; ++r) {
        uint n = n0 + r;
        if (n >= N) { break; }
        uint2 raw = *((const device uint2*)(w + (size_t(expert) * N + n) * K + k));
        size_t pn = pe + size_t(n / WD1) * WS1;
        float4 lo = mlx_fp8x4(raw.x), hi = mlx_fp8x4(raw.y);
        float dv[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        for (uint j = 0; j < 8; ++j) {
            size_t p = pn + size_t((CHUNK_SCALE ? k : k + j) / WD2) * WS2;
            float d = dv[j];
            if (W_ZERO) { d -= mlx_fp8(uint(wzero[p])); }
            float wv = mlx_weight_round(precise::fma(d, float(wscale[p]), -0.0f));
            for (uint m = 0; m < TILE_M; ++m) { acc[m][r] = precise::fma(xv[m][j], wv, acc[m][r]); }
        }
    }
}
for (uint m = 0; m < TILE_M; ++m) {
    for (uint r = 0; r < ROWS; ++r) {
        float sum = simd_sum(acc[m][r]);
        if (lane == 0 && n0 + r < N && (GROUPED || xrow + m < M)) {
            y[size_t(GROUPED ? route : xrow + m) * N + n0 + r] = sum;
        }
    }
}
