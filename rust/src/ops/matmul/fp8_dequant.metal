// Decodes a [E*R1, C] FP8 payload to fp32 with ONNX per-element rounding; VEC columns per thread.
uint c = thread_position_in_grid.x * VEC, r = thread_position_in_grid.y;
if (c >= C) { return; }
uint e = r / R1, wr = r % R1;
size_t base = size_t(r) * C + c;
size_t pr = size_t(e / WD0) * WS0 + size_t(wr / WD1) * WS1;
float4 dv = VEC == 4 ? mlx_fp8x4(*((device const uint*)(w + base))) : float4(mlx_fp8(uint(w[base])));
for (uint j = 0; j < VEC; ++j) {
    size_t p = pr + size_t((c + j) / WD2) * WS2;
    float d = dv[j];
    if (W_ZERO) { d -= mlx_fp8(uint(wzero[p])); }
    y[base + j] = precise::fma(d, float(wscale[p]), -0.0f);
}
