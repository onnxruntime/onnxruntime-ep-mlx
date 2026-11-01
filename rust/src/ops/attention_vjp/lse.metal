const uint tid = thread_index_in_threadgroup, sg = tid / 32;
const uint qb = threadgroup_position_in_grid.x * 16,
           h = threadgroup_position_in_grid.y,
           b = threadgroup_position_in_grid.z, kh = h / (HQ / HK);
const float scale = as_type<float>(SCALE);
threadgroup float scores[16 * 64], maximum[16], normalizer[16];
for (uint i = tid; i < 16; i += 128) {
  maximum[i] = -INFINITY;
  normalizer[i] = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint end = CAUSAL ? min(uint(SK), qb + 16) : uint(SK);
for (uint kb = 0; kb < end; kb += 64) {
  for (uint tile = sg; tile < 16; tile += 4) {
    uint qr = tile / 8 * 8, kr = tile % 8 * 8;
    simdgroup_float8x8 sc(0);
    if (qb + qr + 8 <= SQ && kb + kr + 8 <= SK) {
      for (uint d = 0; d < D; d += 8) {
        simdgroup_float8x8 q, k;
        simdgroup_load(q,
                       Q + b * Q_strides[0] + h * Q_strides[1] +
                           (qb + qr) * Q_strides[2] + d,
                       Q_strides[2]);
        simdgroup_load(k,
                       K + b * K_strides[0] + kh * K_strides[1] +
                           (kb + kr) * K_strides[2] + d,
                       K_strides[2], ulong2(0), true);
        simdgroup_multiply_accumulate(sc, q, k, sc);
      }
      simdgroup_store(sc, scores + qr * 64 + kr, 64);
    } else
      for (uint cell = tid % 32; cell < 64; cell += 32) {
        uint q = qb + qr + cell / 8, k = kb + kr + cell % 8;
        float sum = 0;
        if (q < SQ && k < SK)
          for (uint d = 0; d < D; ++d)
            sum +=
                Q[b * Q_strides[0] + h * Q_strides[1] + q * Q_strides[2] + d] *
                K[b * K_strides[0] + kh * K_strides[1] + k * K_strides[2] + d];
        scores[(qr + cell / 8) * 64 + kr + cell % 8] = sum;
      }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint row = sg; row < 16; row += 4) {
    uint q = qb + row;
    float score = -INFINITY;
    for (uint col = tid % 32; col < 64; col += 32) {
      uint k = kb + col;
      float x =
          q < SQ && k < SK
              ? scores[row * 64 + col] * scale +
                    onnax_attention_bias(Mask, Mask_shape, Mask_strides,
                                         Mask_ndim, b, h, q, k, MASK, CAUSAL)
              : -INFINITY;
      scores[row * 64 + col] = x;
      score = max(score, x);
    }
    float mx = simd_max(score);
    float next = max(mx, maximum[row]);
    float sum = 0;
    if (isfinite(next))
      for (uint col = tid % 32; col < 64; col += 32)
        sum += exp(scores[row * 64 + col] - next);
    sum = simd_sum(sum);
    if (tid % 32 == 0) {
      normalizer[row] =
          (isfinite(maximum[row]) ? normalizer[row] * exp(maximum[row] - next)
                                  : 0) +
          sum;
      maximum[row] = next;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint row = sg; row < 16; row += 4) {
  uint q = qb + row;
  if (q < SQ) {
    float delta = 0;
    for (uint d = tid % 32; d < D; d += 32)
      delta += Y[b * Y_strides[0] + h * Y_strides[1] + q * Y_strides[2] + d] *
               G[b * G_strides[0] + h * G_strides[1] + q * G_strides[2] + d];
    delta = simd_sum(delta);
    if (tid % 32 == 0) {
      LS[(b * HQ + h) * SQ + q] =
          normalizer[row] > 0 ? maximum[row] + log(normalizer[row]) : -INFINITY;
      Delta[(b * HQ + h) * SQ + q] = delta;
    }
  }
}
