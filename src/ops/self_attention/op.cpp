#include "op.hpp"

#include "../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops {
namespace {
template <typename T>
void self_attention_impl(std::byte *out_ptr,
                         const std::byte *q_ptr,
                         const std::byte *k_ptr,
                         const std::byte *v_ptr,
                         float scale,
                         size_t qlen,
                         size_t kvlen,
                         size_t nh,
                         size_t nkvh,
                         size_t hd) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *q = reinterpret_cast<const T *>(q_ptr);
    const auto *k = reinterpret_cast<const T *>(k_ptr);
    const auto *v = reinterpret_cast<const T *>(v_ptr);

    const size_t kv_group = nh / nkvh;
    const ptrdiff_t causal_offset = static_cast<ptrdiff_t>(kvlen) - static_cast<ptrdiff_t>(qlen);
    std::vector<float> logits(kvlen);
    std::vector<float> probs(kvlen);

    for (size_t qi = 0; qi < qlen; ++qi) {
        for (size_t h = 0; h < nh; ++h) {
            const size_t kh = h / kv_group;
            float max_logit = -std::numeric_limits<float>::infinity();

            for (size_t kj = 0; kj < kvlen; ++kj) {
                bool masked = static_cast<ptrdiff_t>(kj) > causal_offset + static_cast<ptrdiff_t>(qi);
                if (masked) {
                    logits[kj] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float dot = 0.0f;
                const size_t q_base = (qi * nh + h) * hd;
                const size_t k_base = (kj * nkvh + kh) * hd;
                for (size_t d = 0; d < hd; ++d) {
                    dot += llaisys::utils::cast<float>(q[q_base + d]) * llaisys::utils::cast<float>(k[k_base + d]);
                }
                logits[kj] = dot * scale;
                if (logits[kj] > max_logit) {
                    max_logit = logits[kj];
                }
            }

            float denom = 0.0f;
            for (size_t kj = 0; kj < kvlen; ++kj) {
                if (std::isinf(logits[kj]) && logits[kj] < 0.0f) {
                    probs[kj] = 0.0f;
                } else {
                    probs[kj] = std::exp(logits[kj] - max_logit);
                    denom += probs[kj];
                }
            }

            const size_t out_base = (qi * nh + h) * hd;
            if (denom == 0.0f) {
                for (size_t d = 0; d < hd; ++d) {
                    out[out_base + d] = llaisys::utils::cast<T>(0.0f);
                }
                continue;
            }
            for (size_t d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (size_t kj = 0; kj < kvlen; ++kj) {
                    if (probs[kj] == 0.0f) {
                        continue;
                    }
                    const size_t v_base = (kj * nkvh + kh) * hd;
                    acc += (probs[kj] / denom) * llaisys::utils::cast<float>(v[v_base + d]);
                }
                out[out_base + d] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}
} // namespace

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "SelfAttention: all tensors must be 3D.");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2] && q->shape()[2] == v->shape()[2],
                   "SelfAttention: head_dim mismatch.");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1],
                   "SelfAttention: k/v shape mismatch.");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0] && attn_val->shape()[1] == q->shape()[1] && attn_val->shape()[2] == q->shape()[2],
                   "SelfAttention: attn_val shape mismatch.");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "SelfAttention: num_heads must be divisible by num_kv_heads.");
    CHECK_ARGUMENT(attn_val->dtype() == q->dtype() && q->dtype() == k->dtype() && q->dtype() == v->dtype(),
                   "SelfAttention: dtype mismatch.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    const size_t qlen = q->shape()[0];
    const size_t kvlen = k->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t nkvh = k->shape()[1];
    const size_t hd = q->shape()[2];

    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_impl<float>(attn_val->data(), q->data(), k->data(), v->data(), scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_F16:
        return self_attention_impl<llaisys::fp16_t>(attn_val->data(), q->data(), k->data(), v->data(), scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_BF16:
        return self_attention_impl<llaisys::bf16_t>(attn_val->data(), q->data(), k->data(), v->data(), scale, qlen, kvlen, nh, nkvh, hd);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}
} // namespace llaisys::ops
