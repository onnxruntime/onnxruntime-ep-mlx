const uint tid = thread_index_in_threadgroup, sg = tid / 32;
const uint kb = threadgroup_position_in_grid.x * KV_ROWS,
           kh = threadgroup_position_in_grid.y,
           b = threadgroup_position_in_grid.z;
const float scale = as_type<float>(SCALE);
threadgroup float score[KV_ROWS * 32], dp[KV_ROWS * 32], prob[KV_ROWS * 32], ds[KV_ROWS * 32],
    dk[KV_ROWS * D], dv[KV_ROWS * D];
for (uint i = tid; i < KV_ROWS * D; i += 128) {
  dk[i] = 0;
  dv[i] = 0;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint h = kh * (HQ / HK); h < (kh + 1) * (HQ / HK); ++h) {
  uint start = CAUSAL ? kb / 32 * 32 : 0;
  for (uint qb = start; qb < SQ; qb += 32) {
    for (uint tile = sg; tile < (KV_ROWS / 8) * 4; tile += 4) {
      uint kr = tile / 4 * 8, qr = tile % 4 * 8;
      simdgroup_float8x8 sc(0), dd(0);
      if (kb + kr + 8 <= SK && qb + qr + 8 <= SQ) {
        for (uint d = 0; d < D; d += 8) {
          simdgroup_float8x8 k, q, v, g;
          simdgroup_load(k,
                         K + b * K_strides[0] + kh * K_strides[1] +
                             (kb + kr) * K_strides[2] + d,
                         K_strides[2]);
          simdgroup_load(q,
                         Q + b * Q_strides[0] + h * Q_strides[1] +
                             (qb + qr) * Q_strides[2] + d,
                         Q_strides[2], ulong2(0), true);
          simdgroup_load(v,
                         V + b * V_strides[0] + kh * V_strides[1] +
                             (kb + kr) * V_strides[2] + d,
                         V_strides[2]);
          simdgroup_load(g,
                         G + b * G_strides[0] + h * G_strides[1] +
                             (qb + qr) * G_strides[2] + d,
                         G_strides[2], ulong2(0), true);
          simdgroup_multiply_accumulate(sc, k, q, sc);
          simdgroup_multiply_accumulate(dd, v, g, dd);
        }
        simdgroup_store(sc, score + kr * 32 + qr, 32);
        simdgroup_store(dd, dp + kr * 32 + qr, 32);
      } else
        for (uint cell = tid % 32; cell < 64; cell += 32) {
          uint k = kb + kr + cell / 8, q = qb + qr + cell % 8;
          float a = 0, z = 0;
          if (k < SK && q < SQ)
            for (uint d = 0; d < D; ++d) {
              a +=
                  K[b * K_strides[0] + kh * K_strides[1] + k * K_strides[2] +
                    d] *
                  Q[b * Q_strides[0] + h * Q_strides[1] + q * Q_strides[2] + d];
              z +=
                  V[b * V_strides[0] + kh * V_strides[1] + k * V_strides[2] +
                    d] *
                  G[b * G_strides[0] + h * G_strides[1] + q * G_strides[2] + d];
            }
          score[(kr + cell / 8) * 32 + qr + cell % 8] = a;
          dp[(kr + cell / 8) * 32 + qr + cell % 8] = z;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < KV_ROWS * 32; i += 128) {
      uint k = kb + i / 32, q = qb + i % 32;
      float p = 0;
      if (k < SK && q < SQ) {
        float ls = LS[(b * HQ + h) * SQ + q];
        float s = score[i] * scale +
                  onnax_attention_bias(Mask, Mask_shape, Mask_strides,
                                       Mask_ndim, b, h, q, k, MASK, CAUSAL);
        if (isfinite(ls))
          p = exp(s - ls);
      }
      prob[i] = p;
      ds[i] = p > 0 ? p * (dp[i] - Delta[(b * HQ + h) * SQ + q]) * scale : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint tile = sg; tile < (KV_ROWS / 8) * (D / 8); tile += 4) {
      uint kr = tile / (D / 8) * 8, d = tile % (D / 8) * 8;
      simdgroup_float8x8 ka, va;
      simdgroup_load(ka, dk + kr * D + d, D);
      simdgroup_load(va, dv + kr * D + d, D);
      if (qb + 32 <= SQ) {
        for (uint q = 0; q < 32; q += 8) {
          simdgroup_float8x8 a, p, z, g;
          simdgroup_load(a, ds + kr * 32 + q, 32);
          simdgroup_load(p, prob + kr * 32 + q, 32);
          simdgroup_load(z,
                         Q + b * Q_strides[0] + h * Q_strides[1] +
                             (qb + q) * Q_strides[2] + d,
                         Q_strides[2]);
          simdgroup_load(g,
                         G + b * G_strides[0] + h * G_strides[1] +
                             (qb + q) * G_strides[2] + d,
                         G_strides[2]);
          simdgroup_multiply_accumulate(ka, a, z, ka);
          simdgroup_multiply_accumulate(va, p, g, va);
        }
        simdgroup_store(ka, dk + kr * D + d, D);
        simdgroup_store(va, dv + kr * D + d, D);
      } else {
        simdgroup_store(ka, dk + kr * D + d, D);
        simdgroup_store(va, dv + kr * D + d, D);
        for (uint cell = tid % 32; cell < 64; cell += 32) {
          uint row = kr + cell / 8, col = d + cell % 8;
          float a = 0, z = 0;
          for (uint q = 0; qb + q < SQ; ++q) {
            a += ds[row * 32 + q] * Q[b * Q_strides[0] + h * Q_strides[1] +
                                      (qb + q) * Q_strides[2] + col];
            z += prob[row * 32 + q] * G[b * G_strides[0] + h * G_strides[1] +
                                        (qb + q) * G_strides[2] + col];
          }
          dk[row * D + col] += a;
          dv[row * D + col] += z;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
for (uint i = tid; i < KV_ROWS * D; i += 128) {
  uint k = kb + i / D;
  if (k < SK) {
    DK[((b * HK + kh) * SK + k) * D + i % D] = dk[i];
    DV[((b * HK + kh) * SK + k) * D + i % D] = dv[i];
  }
}
