// FP8 decode GEMV over N-contiguous weights [E?,K,N] (MatMul / transB=0).
// Lanes own 4 adjacent columns so each row read is a coalesced 128-byte segment; simdgroups and
// grid.y split K. Partials are reduced in threadgroup memory, then across SPLIT by the caller.
// Requires N % 4 == 0 and a 4-byte aligned payload.
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint n0 = threadgroup_position_in_grid.x * 128 + lane * 4;
uint split = threadgroup_position_in_grid.y;
uint route = threadgroup_position_in_grid.z;
uint expert = GROUPED ? uint(indices[route]) : 0;
uint xrow = GROUPED ? route / S : 0;
size_t pe = size_t(expert / WD0) * WS0;
bool active = n0 < N;
float acc[TILE_M][4];
for (uint m = 0; m < TILE_M; ++m) {
    for (uint j = 0; j < 4; ++j) { acc[m][j] = 0.0f; }
}
if (active) {
    for (uint k = split * SIMDGROUPS + sg; k < K; k += SPLIT * SIMDGROUPS) {
        uint raw = *((const device uint*)(w + (size_t(expert) * K + k) * N + n0));
        size_t pk = pe + size_t(k / WD1) * WS1;
        float xv[TILE_M];
        for (uint m = 0; m < TILE_M; ++m) { xv[m] = float(x[size_t(xrow + m) * K + k]); }
        float4 dv = mlx_fp8x4(raw);
        for (uint j = 0; j < 4; ++j) {
            size_t p = pk + size_t((CHUNK_SCALE ? n0 : n0 + j) / WD2) * WS2;
            float d = dv[j];
            if (W_ZERO) { d -= mlx_fp8(uint(wzero[p])); }
            float wv = precise::fma(d, float(wscale[p]), -0.0f);
            for (uint m = 0; m < TILE_M; ++m) { acc[m][j] = precise::fma(xv[m], wv, acc[m][j]); }
        }
    }
}
threadgroup float partial[SIMDGROUPS * 128];
uint rows = GROUPED ? M * S : M;
for (uint m = 0; m < TILE_M; ++m) {
    for (uint j = 0; j < 4; ++j) { partial[sg * 128 + lane * 4 + j] = acc[m][j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && active) {
        for (uint j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for (uint g = 0; g < SIMDGROUPS; ++g) { sum += partial[g * 128 + lane * 4 + j]; }
            y[(size_t(split) * rows + (GROUPED ? route : m)) * N + n0 + j] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
