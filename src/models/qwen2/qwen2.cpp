#include "qwen2.hpp"

#include "../../llaisys/llaisys_tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>

namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, const int *device_ids, int ndevice)
    : _meta(meta), _device(device), _maxseq(0), _cur_pos(0), _scale(1.0f) {
    CHECK_ARGUMENT(device == LLAISYS_DEVICE_CPU, "Qwen2Model: only CPU device is supported currently.");
    if (device_ids != nullptr && ndevice > 0) {
        _device_ids.assign(device_ids, device_ids + ndevice);
    } else {
        _device_ids.push_back(0);
    }

    _attn_norm_w.resize(_meta.nlayer, nullptr);
    _attn_q_w.resize(_meta.nlayer, nullptr);
    _attn_q_b.resize(_meta.nlayer, nullptr);
    _attn_k_w.resize(_meta.nlayer, nullptr);
    _attn_k_b.resize(_meta.nlayer, nullptr);
    _attn_v_w.resize(_meta.nlayer, nullptr);
    _attn_v_b.resize(_meta.nlayer, nullptr);
    _attn_o_w.resize(_meta.nlayer, nullptr);
    _mlp_norm_w.resize(_meta.nlayer, nullptr);
    _mlp_gate_w.resize(_meta.nlayer, nullptr);
    _mlp_up_w.resize(_meta.nlayer, nullptr);
    _mlp_down_w.resize(_meta.nlayer, nullptr);

    _weights.in_embed = nullptr;
    _weights.out_embed = nullptr;
    _weights.out_norm_w = nullptr;
    _weights.attn_norm_w = _attn_norm_w.data();
    _weights.attn_q_w = _attn_q_w.data();
    _weights.attn_q_b = _attn_q_b.data();
    _weights.attn_k_w = _attn_k_w.data();
    _weights.attn_k_b = _attn_k_b.data();
    _weights.attn_v_w = _attn_v_w.data();
    _weights.attn_v_b = _attn_v_b.data();
    _weights.attn_o_w = _attn_o_w.data();
    _weights.mlp_norm_w = _mlp_norm_w.data();
    _weights.mlp_gate_w = _mlp_gate_w.data();
    _weights.mlp_up_w = _mlp_up_w.data();
    _weights.mlp_down_w = _mlp_down_w.data();

    _scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
}

Qwen2Model::~Qwen2Model() {
    _k_cache.clear();
    _v_cache.clear();
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return &_weights;
}

void Qwen2Model::reset(size_t maxseq) {
    core::context().setDevice(_device, _device_ids[0]);

    _maxseq = maxseq;
    _cur_pos = 0;

    _k_cache.clear();
    _v_cache.clear();
    _k_cache.reserve(_meta.nlayer);
    _v_cache.reserve(_meta.nlayer);

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _k_cache.push_back(Tensor::create({_maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_ids[0]));
        _v_cache.push_back(Tensor::create({_maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_ids[0]));
    }

    _token_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_ids[0]);
    _pos_id = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_ids[0]);

    _x = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_ids[0]);
    _x_norm = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_ids[0]);
    _q = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_ids[0]);
    _k = Tensor::create({1, _meta.nkvh * _meta.dh}, _meta.dtype, _device, _device_ids[0]);
    _v = Tensor::create({1, _meta.nkvh * _meta.dh}, _meta.dtype, _device, _device_ids[0]);
    _q_rope = Tensor::create({1, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_ids[0]);
    _k_rope = Tensor::create({1, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_ids[0]);
    _attn = Tensor::create({1, _meta.nh, _meta.dh}, _meta.dtype, _device, _device_ids[0]);
    _attn_proj = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_ids[0]);
    _res1 = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_ids[0]);
    _mlp_norm = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_ids[0]);
    _gate = Tensor::create({1, _meta.di}, _meta.dtype, _device, _device_ids[0]);
    _up = Tensor::create({1, _meta.di}, _meta.dtype, _device, _device_ids[0]);
    _swiglu = Tensor::create({1, _meta.di}, _meta.dtype, _device, _device_ids[0]);
    _mlp_out = Tensor::create({1, _meta.hs}, _meta.dtype, _device, _device_ids[0]);
    _logits = Tensor::create({1, _meta.voc}, _meta.dtype, _device, _device_ids[0]);
    _max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_ids[0]);
    _max_val = Tensor::create({1}, _meta.dtype, _device, _device_ids[0]);
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2ModelInfer: token_ids is null.");
    CHECK_ARGUMENT(ntoken > 0, "Qwen2ModelInfer: ntoken must be > 0.");
    CHECK_ARGUMENT(_maxseq > 0, "Qwen2ModelInfer: call reset() before infer.");
    CHECK_ARGUMENT(ntoken <= _maxseq, "Qwen2ModelInfer: ntoken exceeds maxseq.");

    core::context().setDevice(_device, _device_ids[0]);

    if (_cur_pos > ntoken) {
        _cur_pos = 0;
    }

    for (size_t i = _cur_pos; i < ntoken; ++i) {
        run_one_token(token_ids[i]);
        _cur_pos++;
    }

    // final norm + lm head
    tensor_t out_norm_w = unwrap(_weights.out_norm_w);
    tensor_t out_embed = unwrap(_weights.out_embed);
    ops::rms_norm(_x_norm, _x, out_norm_w, _meta.epsilon);
    ops::linear(_logits, _x_norm, out_embed, nullptr);

    auto logits_1d = _logits->view({_meta.voc});
    ops::argmax(_max_idx, _max_val, logits_1d);
    int64_t next = *reinterpret_cast<int64_t *>(_max_idx->data());
    return next;
}

tensor_t Qwen2Model::unwrap(llaisysTensor_t handle) const {
    CHECK_ARGUMENT(handle != nullptr, "Qwen2Model: weight tensor is null.");
    return handle->tensor;
}

void Qwen2Model::run_one_token(int64_t token) {
    CHECK_ARGUMENT(_weights.in_embed != nullptr, "Qwen2Model: in_embed is null.");

    _token_idx->load(&token);
    ops::embedding(_x, _token_idx, unwrap(_weights.in_embed));

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        ops::rms_norm(_x_norm, _x, unwrap(_weights.attn_norm_w[layer]), _meta.epsilon);

        ops::linear(_q, _x_norm, unwrap(_weights.attn_q_w[layer]), unwrap(_weights.attn_q_b[layer]));
        ops::linear(_k, _x_norm, unwrap(_weights.attn_k_w[layer]), unwrap(_weights.attn_k_b[layer]));
        ops::linear(_v, _x_norm, unwrap(_weights.attn_v_w[layer]), unwrap(_weights.attn_v_b[layer]));

        auto q3 = _q->view({1, _meta.nh, _meta.dh});
        auto k3 = _k->view({1, _meta.nkvh, _meta.dh});
        auto v3 = _v->view({1, _meta.nkvh, _meta.dh});

        int64_t pos = static_cast<int64_t>(_cur_pos);
        _pos_id->load(&pos);
        ops::rope(_q_rope, q3, _pos_id, _meta.theta);
        ops::rope(_k_rope, k3, _pos_id, _meta.theta);

        auto k_slice = _k_cache[layer]->slice(0, _cur_pos, _cur_pos + 1);
        auto v_slice = _v_cache[layer]->slice(0, _cur_pos, _cur_pos + 1);
        ops::rearrange(k_slice, _k_rope);
        ops::rearrange(v_slice, v3);

        auto k_all = _k_cache[layer]->slice(0, 0, _cur_pos + 1);
        auto v_all = _v_cache[layer]->slice(0, 0, _cur_pos + 1);

        ops::self_attention(_attn, _q_rope, k_all, v_all, _scale);

        auto attn2 = _attn->view({1, _meta.hs});
        ops::linear(_attn_proj, attn2, unwrap(_weights.attn_o_w[layer]), nullptr);
        ops::add(_res1, _x, _attn_proj);

        ops::rms_norm(_mlp_norm, _res1, unwrap(_weights.mlp_norm_w[layer]), _meta.epsilon);
        ops::linear(_gate, _mlp_norm, unwrap(_weights.mlp_gate_w[layer]), nullptr);
        ops::linear(_up, _mlp_norm, unwrap(_weights.mlp_up_w[layer]), nullptr);
        ops::swiglu(_swiglu, _gate, _up);
        ops::linear(_mlp_out, _swiglu, unwrap(_weights.mlp_down_w[layer]), nullptr);
        ops::add(_x, _res1, _mlp_out);
    }
}

} // namespace llaisys::models
