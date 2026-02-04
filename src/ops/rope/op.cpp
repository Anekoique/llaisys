#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {
namespace {
template <typename T>
void rope_impl(std::byte *out_ptr,
               const std::byte *in_ptr,
               const int64_t *pos_ids,
               float theta,
               size_t seqlen,
               size_t nhead,
               size_t dim) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *in = reinterpret_cast<const T *>(in_ptr);
    const size_t half = dim / 2;

    for (size_t s = 0; s < seqlen; ++s) {
        float pos = static_cast<float>(pos_ids[s]);
        for (size_t h = 0; h < nhead; ++h) {
            size_t base = (s * nhead + h) * dim;
            for (size_t j = 0; j < half; ++j) {
                float a = llaisys::utils::cast<float>(in[base + j]);
                float b = llaisys::utils::cast<float>(in[base + half + j]);
                float phi = pos / std::pow(theta, (2.0f * static_cast<float>(j)) / static_cast<float>(dim));
                float cos_v = std::cos(phi);
                float sin_v = std::sin(phi);

                out[base + j] = llaisys::utils::cast<T>(a * cos_v - b * sin_v);
                out[base + half + j] = llaisys::utils::cast<T>(b * cos_v + a * sin_v);
            }
        }
    }
}
} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(out->ndim() == 3 && in->ndim() == 3, "RoPE: out/in must be 3D.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D.");
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "RoPE: pos_ids length mismatch.");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids dtype must be int64.");
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "RoPE: out/in dtype mismatch.");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "RoPE: head dim must be even.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");

    const size_t seqlen = in->shape()[0];
    const size_t nhead = in->shape()[1];
    const size_t dim = in->shape()[2];
    const auto *pos = reinterpret_cast<const int64_t *>(pos_ids->data());

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_impl<float>(out->data(), in->data(), pos, theta, seqlen, nhead, dim);
    case LLAISYS_DTYPE_F16:
        return rope_impl<llaisys::fp16_t>(out->data(), in->data(), pos, theta, seqlen, nhead, dim);
    case LLAISYS_DTYPE_BF16:
        return rope_impl<llaisys::bf16_t>(out->data(), in->data(), pos, theta, seqlen, nhead, dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
