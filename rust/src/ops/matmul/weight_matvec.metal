size_t out = thread_position_in_grid.x / 32;
uint lane = thread_position_in_grid.x % 32;
if (out >= size_t(M) * S * N) { return; }
uint col = out % N;
size_t route = out / N;
uint row = route / S;
uint expert = GROUPED ? uint(indices[route]) : 0;
float sum = 0.0f;
for (uint k = lane; k < K; k += 32) {
    float weight = float(WT(mlx_weight_w(w, wscale, wzero, expert, k, col)));
    sum = precise::fma(mlx_weight_x(x, xscale, xzero, row, k), weight, sum);
}
sum = simd_sum(sum);
if (lane == 0) { y[out] = WT(sum); }
