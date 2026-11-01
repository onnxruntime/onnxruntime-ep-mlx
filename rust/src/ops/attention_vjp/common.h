#include <metal_simdgroup_matrix>
using namespace metal;

template <typename Mask>
inline float onnax_attention_bias(Mask mask, constant int *shape,
                                  constant int64_t *strides, uint ndim, uint b,
                                  uint h, uint q, uint k, uint kind,
                                  bool causal) {
  if (causal && k > q)
    return -INFINITY;
  if (kind == 0)
    return 0.0f;
  const uint width = shape[ndim - 1];
  if (width != 1 && k >= width)
    return -INFINITY;
  int64_t offset = (width == 1 ? 0 : k) * strides[ndim - 1];
  if (ndim >= 2)
    offset += (shape[ndim - 2] == 1 ? 0 : q) * strides[ndim - 2];
  if (ndim >= 3)
    offset += (shape[ndim - 3] == 1 ? 0 : h) * strides[ndim - 3];
  if (ndim >= 4)
    offset += (shape[ndim - 4] == 1 ? 0 : b) * strides[ndim - 4];
  return kind == 1 ? (mask[offset] ? 0.0f : -INFINITY) : float(mask[offset]);
}
