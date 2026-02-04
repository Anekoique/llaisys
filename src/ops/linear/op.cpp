#include "op.hpp"

#include "../../utils.hpp"

namespace llaisys::ops {
namespace {
template <typename T>
void linear_impl(std::byte *out_ptr,
                 const std::byte *in_ptr,
                 const std::byte *w_ptr,
                 const std::byte *b_ptr,
                 size_t m,
                 size_t k,
                 size_t n) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *in = reinterpret_cast<const T *>(in_ptr);
    const auto *w = reinterpret_cast<const T *>(w_ptr);
    const auto *b = (b_ptr == nullptr) ? nullptr : reinterpret_cast<const T *>(b_ptr);

    for (size_t i = 0; i < m; ++i) {
        for (size_t o = 0; o < n; ++o) {
            float acc = (b == nullptr) ? 0.0f : llaisys::utils::cast<float>(b[o]);
            const size_t in_base = i * k;
            const size_t w_base = o * k;
            for (size_t j = 0; j < k; ++j) {
                acc += llaisys::utils::cast<float>(in[in_base + j]) * llaisys::utils::cast<float>(w[w_base + j]);
            }
            out[i * n + o] = llaisys::utils::cast<T>(acc);
        }
    }
}
} // namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2, "Linear: out/in/weight must be 2D.");
    CHECK_ARGUMENT(in->shape()[0] == out->shape()[0], "Linear: batch size mismatch.");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear: in_features mismatch.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "Linear: out_features mismatch.");
    CHECK_ARGUMENT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(), "Linear: out/in/weight dtype mismatch.");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: out/in/weight must be contiguous.");

    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias must be 1D.");
        CHECK_ARGUMENT(bias->shape()[0] == out->shape()[1], "Linear: bias shape mismatch.");
        CHECK_ARGUMENT(bias->dtype() == out->dtype(), "Linear: bias dtype mismatch.");
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    const size_t m = out->shape()[0];
    const size_t n = out->shape()[1];
    const size_t k = in->shape()[1];
    const std::byte *bias_ptr = (bias == nullptr) ? nullptr : bias->data();

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_impl<float>(out->data(), in->data(), weight->data(), bias_ptr, m, k, n);
    case LLAISYS_DTYPE_F16:
        return linear_impl<llaisys::fp16_t>(out->data(), in->data(), weight->data(), bias_ptr, m, k, n);
    case LLAISYS_DTYPE_BF16:
        return linear_impl<llaisys::bf16_t>(out->data(), in->data(), weight->data(), bias_ptr, m, k, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
