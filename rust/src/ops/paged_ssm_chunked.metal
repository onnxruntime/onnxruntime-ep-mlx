// The SIMD matrix layout and block recurrence follow MLX's MIT-licensed
// gated_delta_update.h. See NOTICE. Decay products here are unclamped and zero-safe.
uint lane = thread_position_in_grid.x % 32;
uint column = thread_position_in_grid.x / 32;
if (column >= size_t(B) * H * (V / 8)) return;
uint vb = (column % (V / 8)) * 8;
uint h = (column / (V / 8)) % H;
uint b = column / ((V / 8) * H), kh = h / (H / HQ);
uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
uint fn = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
int start = offsets[b], end = offsets[b + 1];
if (offsets[0] != 0 || offsets[B] != N ||
    start < 0 || end < start || end > N) return;
// CAPTURE_INIT
simdgroup_float8x8 s[K / 8];
#pragma clang loop unroll(full)
for (uint ki = 0; ki < K; ki += 8) {
    simdgroup_float8x8 tile;
    tile.thread_elements()[0] = initial[((size_t(b)*H+h)*V+vb+fn)*K+ki+fm];
    tile.thread_elements()[1] = initial[((size_t(b)*H+h)*V+vb+fn+1)*K+ki+fm];
    s[ki/8] = tile;
}
for (int t = start; t < end; t += 8) {
    int valid = min(8, end-t);
    float prefix = 1.0f, between0 = 1.0f, between1 = 1.0f;
    float tail0 = 1.0f, tail1 = 1.0f, total = 1.0f;
    #pragma clang loop unroll(full)
    for (uint j=0; j<8; ++j) {
        float g = j<uint(valid) ? gates_rates[size_t(t+j)*H+h] : 1.0f;
        if (j<=fm) prefix *= g;
        if (j>fn && j<=fm) between0 *= g;
        if (j>fn+1 && j<=fm) between1 *= g;
        if (j>fn) tail0 *= g;
        if (j>fn+1) tail1 *= g;
        total *= g;
    }
    float beta_i = fm<uint(valid) ? gates_rates[(size_t(N)+t+fm)*H+h] : 0.0f;
    simdgroup_float8x8 kk = make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 qk = make_filled_simdgroup_matrix<float,8>(0.0f);
    simdgroup_float8x8 base = make_filled_simdgroup_matrix<float,8>(0.0f);
    #pragma clang loop unroll(full)
    for (uint ki=0; ki<K; ki+=8) {
        simdgroup_float8x8 k, kt, q;
        k.thread_elements()[0] = fm<uint(valid) ? normalized_qk[((size_t(N)+t+fm)*HQ+kh)*K+ki+fn] : 0.0f;
        k.thread_elements()[1] = fm<uint(valid) ? normalized_qk[((size_t(N)+t+fm)*HQ+kh)*K+ki+fn+1] : 0.0f;
        kt.thread_elements()[0] = fn<uint(valid) ? normalized_qk[((size_t(N)+t+fn)*HQ+kh)*K+ki+fm] : 0.0f;
        kt.thread_elements()[1] = fn+1<uint(valid) ? normalized_qk[((size_t(N)+t+fn+1)*HQ+kh)*K+ki+fm] : 0.0f;
        q.thread_elements()[0] = fm<uint(valid) ? normalized_qk[(size_t(t+fm)*HQ+kh)*K+ki+fn] : 0.0f;
        q.thread_elements()[1] = fm<uint(valid) ? normalized_qk[(size_t(t+fm)*HQ+kh)*K+ki+fn+1] : 0.0f;
        simdgroup_multiply_accumulate(kk,k,kt,kk);
        simdgroup_multiply_accumulate(qk,q,kt,qk);
        q.thread_elements()[0] *= prefix;
        q.thread_elements()[1] *= prefix;
        simdgroup_multiply_accumulate(base,q,s[ki/8],base);
    }
    simdgroup_float8x8 identity = make_filled_simdgroup_matrix<float,8>(0.0f);
    identity.thread_elements()[0] = fm==fn ? 1.0f : 0.0f;
    identity.thread_elements()[1] = fm==fn+1 ? 1.0f : 0.0f;
    simdgroup_float8x8 power = kk, inverse;
    power.thread_elements()[0] *= fn<fm ? beta_i : 0.0f;
    power.thread_elements()[1] *= fn+1<fm ? beta_i : 0.0f;
    inverse.thread_elements()[0] = identity.thread_elements()[0]-power.thread_elements()[0];
    inverse.thread_elements()[1] = identity.thread_elements()[1]-power.thread_elements()[1];
    #pragma clang loop unroll(full)
    for (int step=0; step<2; ++step) {
        simdgroup_multiply(power,power,power);
        simdgroup_multiply_accumulate(inverse,inverse,power,inverse);
    }
    simdgroup_float8x8 ws = make_filled_simdgroup_matrix<float,8>(0.0f);
    #pragma clang loop unroll(full)
    for (uint ki=0; ki<K; ki+=8) {
        simdgroup_float8x8 k, w;
        k.thread_elements()[0] = fm<uint(valid) ? normalized_qk[((size_t(N)+t+fm)*HQ+kh)*K+ki+fn]*beta_i : 0.0f;
        k.thread_elements()[1] = fm<uint(valid) ? normalized_qk[((size_t(N)+t+fm)*HQ+kh)*K+ki+fn+1]*beta_i : 0.0f;
        simdgroup_multiply(w,inverse,k);
        w.thread_elements()[0] *= prefix;
        w.thread_elements()[1] *= prefix;
        simdgroup_multiply_accumulate(ws,w,s[ki/8],ws);
    }
    inverse.thread_elements()[0] *= fn<=fm ? between0 : 0.0f;
    inverse.thread_elements()[1] *= fn+1<=fm ? between1 : 0.0f;
    simdgroup_float8x8 values, u, delta, y;
    values.thread_elements()[0] = fm<uint(valid) ? float(qkv[size_t(t+fm)*WIDTH+2*HQ*K+h*V+vb+fn])*beta_i : 0.0f;
    values.thread_elements()[1] = fm<uint(valid) ? float(qkv[size_t(t+fm)*WIDTH+2*HQ*K+h*V+vb+fn+1])*beta_i : 0.0f;
    simdgroup_multiply(u,inverse,values);
    delta.thread_elements()[0] = u.thread_elements()[0]-ws.thread_elements()[0];
    delta.thread_elements()[1] = u.thread_elements()[1]-ws.thread_elements()[1];
    // CAPTURE_STEP
    qk.thread_elements()[0] *= fn<=fm ? between0 : 0.0f;
    qk.thread_elements()[1] *= fn+1<=fm ? between1 : 0.0f;
    simdgroup_multiply_accumulate(y,qk,delta,base);
    if (fm<uint(valid)) {
        output[(size_t(t+fm)*H+h)*V+vb+fn] = T(y.thread_elements()[0]*as_type<float>(uint(SCALE_BITS)));
        output[(size_t(t+fm)*H+h)*V+vb+fn+1] = T(y.thread_elements()[1]*as_type<float>(uint(SCALE_BITS)));
    }
    #pragma clang loop unroll(full)
    for (uint ki=0; ki<K; ki+=8) {
        simdgroup_float8x8 kt, update;
        kt.thread_elements()[0] = fn<uint(valid) ? normalized_qk[((size_t(N)+t+fn)*HQ+kh)*K+ki+fm]*tail0 : 0.0f;
        kt.thread_elements()[1] = fn+1<uint(valid) ? normalized_qk[((size_t(N)+t+fn+1)*HQ+kh)*K+ki+fm]*tail1 : 0.0f;
        simdgroup_multiply(update,kt,delta);
        s[ki/8].thread_elements()[0] = total*s[ki/8].thread_elements()[0]+update.thread_elements()[0];
        s[ki/8].thread_elements()[1] = total*s[ki/8].thread_elements()[1]+update.thread_elements()[1];
    }
}
#pragma clang loop unroll(full)
for (uint ki=0; ki<K; ki+=8) {
    final[((size_t(b)*H+h)*V+vb+fn)*K+ki+fm] = s[ki/8].thread_elements()[0];
    final[((size_t(b)*H+h)*V+vb+fn+1)*K+ki+fm] = s[ki/8].thread_elements()[1];
}
