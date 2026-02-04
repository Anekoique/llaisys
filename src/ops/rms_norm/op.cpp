#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {
namespace {
template <typename T>
void rms_norm_impl(std::byte *out_ptr,
                   const std::byte *in_ptr,
                   const std::byte *weight_ptr,
                   float eps,
                   size_t rows,
                   size_t dim) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *in = reinterpret_cast<const T *>(in_ptr);
    const auto *weight = reinterpret_cast<const T *>(weight_ptr);

    for (size_t r = 0; r < rows; ++r) {
        float square_sum = 0.0f;
        size_t base = r * dim;
        for (size_t c = 0; c < dim; ++c) {
            float x = llaisys::utils::cast<float>(in[base + c]);
            square_sum += x * x;
        }
        T mean = llaisys::utils::cast<T>(square_sum / static_cast<float>(dim));
        T mean_eps = llaisys::utils::cast<T>(llaisys::utils::cast<float>(mean) + eps);
        T inv_rms = llaisys::utils::cast<T>(1.0f / std::sqrt(llaisys::utils::cast<float>(mean_eps)));
        for (size_t c = 0; c < dim; ++c) {
            float x = llaisys::utils::cast<float>(in[base + c]);
            float w = llaisys::utils::cast<float>(weight[c]);
            T normed = llaisys::utils::cast<T>(x * llaisys::utils::cast<float>(inv_rms));
            out[base + c] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(normed) * w);
        }
    }
}
} // namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2, "RmsNorm: out/in must be 2D.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(weight->ndim() == 1, "RmsNorm: weight must be 1D.");
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "RmsNorm: weight length mismatch.");
    CHECK_ARGUMENT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(), "RmsNorm: dtype mismatch.");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RmsNorm: all tensors must be contiguous.");

    const size_t rows = in->shape()[0];
    const size_t dim = in->shape()[1];
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_impl<float>(out->data(), in->data(), weight->data(), eps, rows, dim);
    case LLAISYS_DTYPE_F16:
        return rms_norm_impl<llaisys::fp16_t>(out->data(), in->data(), weight->data(), eps, rows, dim);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_impl<llaisys::bf16_t>(out->data(), in->data(), weight->data(), eps, rows, dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
