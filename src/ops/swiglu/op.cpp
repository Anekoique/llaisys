#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>

namespace llaisys::ops {
namespace {
template <typename T>
void swiglu_impl(std::byte *out_ptr, const std::byte *gate_ptr, const std::byte *up_ptr, size_t n) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *gate = reinterpret_cast<const T *>(gate_ptr);
    const auto *up = reinterpret_cast<const T *>(up_ptr);

    for (size_t i = 0; i < n; ++i) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        float silu = g / (1.0f + std::exp(-g));
        out[i] = llaisys::utils::cast<T>(u * silu);
    }
}
} // namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous.");

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_impl<float>(out->data(), gate->data(), up->data(), out->numel());
    case LLAISYS_DTYPE_F16:
        return swiglu_impl<llaisys::fp16_t>(out->data(), gate->data(), up->data(), out->numel());
    case LLAISYS_DTYPE_BF16:
        return swiglu_impl<llaisys::bf16_t>(out->data(), gate->data(), up->data(), out->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
