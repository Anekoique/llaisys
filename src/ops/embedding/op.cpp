#include "op.hpp"

#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index must be 1D.");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: out must be 2D.");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index dtype must be int64.");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "Embedding: out dtype must match weight dtype.");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0], "Embedding: out rows must equal index size.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "Embedding: out cols must match embedding dim.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");

    const auto *idx = reinterpret_cast<const int64_t *>(index->data());
    const size_t rows = index->shape()[0];
    const size_t vocab = weight->shape()[0];
    const size_t dim = weight->shape()[1];
    const size_t row_bytes = dim * llaisys::utils::dsize(weight->dtype());
    auto *dst = out->data();
    const auto *src = weight->data();

    for (size_t i = 0; i < rows; ++i) {
        CHECK_ARGUMENT(idx[i] >= 0 && static_cast<size_t>(idx[i]) < vocab, "Embedding: index out of range.");
        std::memcpy(dst + i * row_bytes, src + static_cast<size_t>(idx[i]) * row_bytes, row_bytes);
    }
}
} // namespace llaisys::ops
