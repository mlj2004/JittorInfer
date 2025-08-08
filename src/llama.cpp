#include "llama.h"

#include "llama-context.h"
#include "llama-impl.h"

bool llama_supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr || llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context *   ctx    = ggml_init(params);
        ggml_free(ctx);
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

struct llama_perf_context_data llama_perf_context(const struct llama_context * ctx) {
    struct llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data.t_start_ms  = 1e-3 * ctx->t_start_us;
    data.t_load_ms   = 1e-3 * ctx->t_load_us;
    data.t_p_eval_ms = 1e-3 * ctx->t_p_eval_us;
    data.t_eval_ms   = 1e-3 * ctx->t_eval_us;
    data.n_p_eval    = std::max(1, ctx->n_p_eval);
    data.n_eval      = std::max(1, ctx->n_eval);

    return data;
}

void llama_perf_context_print(const struct llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                   __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval,
                   1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                   __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval,
                   1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms),
                   (data.n_p_eval + data.n_eval));
}

void llama_perf_context_reset(struct llama_context * ctx) {
    ctx->t_start_us = ggml_time_us();
    ctx->t_eval_us = ctx->n_eval = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

struct llama_kv_cache_view llama_kv_cache_view_init(const struct llama_context * ctx, int32_t n_seq_max) {
    return llama_kv_cache_view_init(ctx->kv_self, n_seq_max);
}

void llama_kv_cache_view_update(const struct llama_context * ctx, struct llama_kv_cache_view * view) {
    llama_kv_cache_view_update(view, ctx->kv_self);
}

bool llama_kv_cache_seq_rm(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return llama_kv_cache_seq_rm(ctx->kv_self, seq_id, p0, p1);
}

void llama_kv_cache_seq_cp(struct llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0,
                           llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }
    llama_kv_cache_seq_cp(ctx->kv_self, seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_seq_add(struct llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1,
                            llama_pos delta) {
    if (delta == 0) {
        return;
    }

    llama_kv_cache_seq_add(ctx->kv_self, seq_id, p0, p1, delta);
}
