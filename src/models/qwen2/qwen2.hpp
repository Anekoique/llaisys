#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <vector>

namespace llaisys::models {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, const int *device_ids, int ndevice);
    ~Qwen2Model();

    LlaisysQwen2Weights *weights();
    void reset(size_t maxseq);
    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    void run_one_token(int64_t token);
    tensor_t unwrap(llaisysTensor_t handle) const;

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    std::vector<int> _device_ids;

    LlaisysQwen2Weights _weights;
    std::vector<llaisysTensor_t> _attn_norm_w;
    std::vector<llaisysTensor_t> _attn_q_w;
    std::vector<llaisysTensor_t> _attn_q_b;
    std::vector<llaisysTensor_t> _attn_k_w;
    std::vector<llaisysTensor_t> _attn_k_b;
    std::vector<llaisysTensor_t> _attn_v_w;
    std::vector<llaisysTensor_t> _attn_v_b;
    std::vector<llaisysTensor_t> _attn_o_w;
    std::vector<llaisysTensor_t> _mlp_norm_w;
    std::vector<llaisysTensor_t> _mlp_gate_w;
    std::vector<llaisysTensor_t> _mlp_up_w;
    std::vector<llaisysTensor_t> _mlp_down_w;

    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;

    size_t _maxseq;
    size_t _cur_pos;
    float _scale;

    tensor_t _token_idx;
    tensor_t _pos_id;
    tensor_t _x;
    tensor_t _x_norm;
    tensor_t _q;
    tensor_t _k;
    tensor_t _v;
    tensor_t _q_rope;
    tensor_t _k_rope;
    tensor_t _attn;
    tensor_t _attn_proj;
    tensor_t _res1;
    tensor_t _mlp_norm;
    tensor_t _gate;
    tensor_t _up;
    tensor_t _swiglu;
    tensor_t _mlp_out;
    tensor_t _logits;
    tensor_t _max_idx;
    tensor_t _max_val;
};

} // namespace llaisys::models
