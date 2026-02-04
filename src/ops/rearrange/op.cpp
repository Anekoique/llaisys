#include "op.hpp"

#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "Rearrange: dtype mismatch.");
    CHECK_ARGUMENT(out->numel() == in->numel(), "Rearrange: numel mismatch.");
    ASSERT(out->isContiguous() && in->isContiguous(), "Rearrange: only contiguous tensors are supported.");
    std::memcpy(out->data(), in->data(), out->numel() * out->elementSize());
}
} // namespace llaisys::ops
