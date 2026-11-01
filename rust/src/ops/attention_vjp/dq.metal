const uint tid = thread_index_in_threadgroup, sg = tid / 32;
const uint qb = threadgroup_position_in_grid.x * 16,
           h = threadgroup_position_in_grid.y,
           b = threadgroup_position_in_grid.z;
const uint kh = h / (HQ / HK);
const float scale = as_type<float>(SCALE);
threadgroup float score[16 * 32], dp[16 * 32], ds[16 * 32], accum[16 * D];
for (uint i = tid; i < 16 * D; i += 128)
  accum[i] = 0;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint end = CAUSAL ? min(uint(SK), qb + 16) : uint(SK);
for (uint kb = 0; kb < end; kb += 32) {
  for (uint tile = sg; tile < 8; tile += 4) {
    const uint qr = tile / 4 * 8, kr = tile % 4 * 8;
    simdgroup_float8x8 sc(0), dd(0);
    // Tail loads use small shared staging; the common full-tile path reads
    // views directly.
    if (qb + qr + 8 <= SQ && kb + kr + 8 <= SK) {
      for (uint d = 0; d < D; d += 8) {
        simdgroup_float8x8 q, k, g, v;
        simdgroup_load(q,
                       Q + b * Q_strides[0] + h * Q_strides[1] +
                           (qb + qr) * Q_strides[2] + d,
                       Q_strides[2]);
        simdgroup_load(k,
                       K + b * K_strides[0] + kh * K_strides[1] +
                           (kb + kr) * K_strides[2] + d,
                       K_strides[2], ulong2(0), true);
        simdgroup_load(g,
                       G + b * G_strides[0] + h * G_strides[1] +
                           (qb + qr) * G_strides[2] + d,
                       G_strides[2]);
        simdgroup_load(v,
                       V + b * V_strides[0] + kh * V_strides[1] +
                           (kb + kr) * V_strides[2] + d,
                       V_strides[2], ulong2(0), true);
        simdgroup_multiply_accumulate(sc, q, k, sc);
        simdgroup_multiply_accumulate(dd, g, v, dd);
      }
      simdgroup_store(sc, score + qr * 32 + kr, 32);
      simdgroup_store(dd, dp + qr * 32 + kr, 32);
    } else {
      for (uint cell = tid % 32; cell < 64; cell += 32) {
        uint q = qb + qr + cell / 8, k = kb + kr + cell % 8;
        float a = 0, z = 0;
        if (q < SQ && k < SK)
          for (uint d = 0; d < D; ++d) {
            a += Q[b * Q_strides[0] + h * Q_strides[1] + q * Q_strides[2] + d] *
                 K[b * K_strides[0] + kh * K_strides[1] + k * K_strides[2] + d];
            z += G[b * G_strides[0] + h * G_strides[1] + q * G_strides[2] + d] *
                 V[b * V_strides[0] + kh * V_strides[1] + k * V_strides[2] + d];
          }
        score[(qr + cell / 8) * 32 + kr + cell % 8] = a;
        dp[(qr + cell / 8) * 32 + kr + cell % 8] = z;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 16 * 32; i += 128) {
    uint q = qb + i / 32, k = kb + i % 32;
    float p = 0;
    if (q < SQ && k < SK) {
      float ls = LS[(b * HQ + h) * SQ + q];
      float s = score[i] * scale +
                onnax_attention_bias(Mask, Mask_shape, Mask_strides, Mask_ndim,
                                     b, h, q, k, MASK, CAUSAL);
      if (isfinite(ls))
        p = exp(s - ls);
    }
    ds[i] = p > 0 ? p * (dp[i] - Delta[(b * HQ + h) * SQ + q]) * scale : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint tile = sg; tile < 2 * (D / 8); tile += 4) {
    uint qr = tile / (D / 8) * 8, d = tile % (D / 8) * 8;
    simdgroup_float8x8 out;
    simdgroup_load(out, accum + qr * D + d, D);
    if (kb + 32 <= SK) {
      for (uint k = 0; k < 32; k += 8) {
        simdgroup_float8x8 a, z;
        simdgroup_load(a, ds + qr * 32 + k, 32);
        simdgroup_load(z,
                       K + b * K_strides[0] + kh * K_strides[1] +
                           (kb + k) * K_strides[2] + d,
                       K_strides[2]);
        simdgroup_multiply_accumulate(out, a, z, out);
      }
      simdgroup_store(out, accum + qr * D + d, D);
    } else {
      simdgroup_store(out, accum + qr * D + d, D);
      for (uint cell = tid % 32; cell < 64; cell += 32) {
        float sum = 0;
        uint row = qr + cell / 8, col = d + cell % 8;
        for (uint k = 0; kb + k < SK; ++k)
          sum += ds[row * 32 + k] * K[b * K_strides[0] + kh * K_strides[1] +
                                      (kb + k) * K_strides[2] + col];
        accum[row * D + col] += sum;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint i = tid; i < 16 * D; i += 128) {
  uint q = qb + i / D;
  if (q < SQ)
    DQ[((b * HQ + h) * SQ + q) * D + i % D] = accum[i];
}
