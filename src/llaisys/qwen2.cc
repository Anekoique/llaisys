#include "llaisys/models/qwen2.h"

#include "../models/qwen2/qwen2.hpp"

__C {
    struct LlaisysQwen2Model {
        llaisys::models::Qwen2Model *impl;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        if (meta == nullptr) {
            return nullptr;
        }
        auto *model = new LlaisysQwen2Model{
            new llaisys::models::Qwen2Model(*meta, device, device_ids, ndevice)};
        return model;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (model == nullptr) {
            return;
        }
        delete model->impl;
        model->impl = nullptr;
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        if (model == nullptr) {
            return nullptr;
        }
        return model->impl->weights();
    }

    void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model, size_t maxseq) {
        if (model == nullptr) {
            return;
        }
        model->impl->reset(maxseq);
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        if (model == nullptr) {
            return -1;
        }
        return model->impl->infer(token_ids, ntoken);
    }
}
