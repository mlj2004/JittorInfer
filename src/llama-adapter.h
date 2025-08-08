#include "ggml.h"

struct llama_adapter_cvec {
    int valid = 1;

    struct ggml_tensor * apply_to(struct ggml_tensor * cur) const {
        GGML_ASSERT(valid == 1);
        return cur;
    }
};

struct llama_adapter_lora {};
