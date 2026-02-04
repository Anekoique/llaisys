#include "op.hpp"

#include "../../utils.hpp"

#include <limits>

namespace llaisys::ops {
namespace {
template <typename T>
void argmax_impl(std::byte *max_idx_ptr, std::byte *max_val_ptr, const std::byte *vals_ptr, size_t n) {
    auto *max_idx = reinterpret_cast<int64_t *>(max_idx_ptr);
    auto *max_val = reinterpret_cast<T *>(max_val_ptr);
    const auto *vals = reinterpret_cast<const T *>(vals_ptr);

    size_t best_idx = 0;
    float best_val = llaisys::utils::cast<float>(vals[0]);
    for (size_t i = 1; i < n; ++i) {
        float cur = llaisys::utils::cast<float>(vals[i]);
        if (cur > best_val) {
            best_val = cur;
            best_idx = i;
        }
    }
    max_idx[0] = static_cast<int64_t>(best_idx);
    max_val[0] = llaisys::utils::cast<T>(best_val);
}
} // namespace

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor.");
    CHECK_ARGUMENT(max_idx->ndim() == 1 && max_idx->numel() == 1, "Argmax: max_idx must be shape [1].");
    CHECK_ARGUMENT(max_val->ndim() == 1 && max_val->numel() == 1, "Argmax: max_val must be shape [1].");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx dtype must be int64.");
    CHECK_ARGUMENT(max_val->dtype() == vals->dtype(), "Argmax: max_val dtype must match vals dtype.");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: all tensors must be contiguous.");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax: vals must be non-empty.");

    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32:
        return argmax_impl<float>(max_idx->data(), max_val->data(), vals->data(), vals->numel());
    case LLAISYS_DTYPE_F16:
        return argmax_impl<llaisys::fp16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel());
    case LLAISYS_DTYPE_BF16:
        return argmax_impl<llaisys::bf16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }
}
} // namespace llaisys::ops
