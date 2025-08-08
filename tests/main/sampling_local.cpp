#include "sampling_local.h"

#include <cmath>

#include "common_local.h"
#include "ggml.h"
#include "llama-context.h"

struct common_sampler_local {
    common_params_local_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * chain;

    // only the last token is needed.
    std::vector<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    void set_logits(struct llama_context * ctx, int idx) {
        const auto logits = llama_get_logits_ith(ctx, idx);

        const int k = logits.len;
        cur.resize(k);
        if (logits.type == llama_logits::LLAMA_LOGITS_TYPE_TOPK) {
            for (int i = 0; i < k; i++) {
                cur[i] = llama_token_data{ (int) logits.indices[i], logits.values[i], 0.0f };
            }
        } else {
            for (int i = 0; i < k; i++) {
                cur[i] = llama_token_data{ i, logits.values[i], 0.0f };
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }
};

struct common_sampler_local * common_sampler_local_init(const struct llama_model *                  model,
                                                        const struct common_params_local_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    struct llama_sampler * grmr;
    if (params.grammar.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", params.grammar.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif  // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<const char *> trigger_words;
        trigger_words.reserve(params.grammar_trigger_words.size());
        for (const auto & str : params.grammar_trigger_words) {
            trigger_words.push_back(str.word.c_str());
        }

        GGML_ASSERT(!params.grammar_lazy);
        grmr = llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
    }

    auto * result = new common_sampler_local{
        /* .params = */ params,
        /* .grmr   = */ grmr,
        /* .chain  = */ llama_sampler_chain_init(lparams),
        /* .prev   = */ std::vector<llama_token>(0),
        /* .cur    = */ {},
        /* .cur_p  = */ {},
    };

    llama_sampler_chain_add(
        result->chain,
        llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), params.logit_bias.size(), params.logit_bias.data()));

    for (const auto & cnstr : params.samplers) {
        switch (cnstr) {
            case COMMON_SAMPLER_TYPE_LOCAL_DRY:
                {
                    std::vector<const char *> c_breakers;
                    c_breakers.reserve(params.dry_sequence_breakers.size());
                    for (const auto & str : params.dry_sequence_breakers) {
                        c_breakers.push_back(str.c_str());
                    }

                    llama_sampler_chain_add(
                        result->chain,
                        llama_sampler_init_dry(vocab, llama_model_n_ctx_train(model), params.dry_multiplier,
                                               params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n,
                                               c_breakers.data(), c_breakers.size()));
                }
                break;
            case COMMON_SAMPLER_TYPE_LOCAL_TOP_K:
                llama_sampler_chain_add(result->chain, llama_sampler_init_top_k(params.top_k));
                break;
            case COMMON_SAMPLER_TYPE_LOCAL_TOP_P:
                llama_sampler_chain_add(result->chain, llama_sampler_init_top_p(params.top_p, params.min_keep));
                break;
            case COMMON_SAMPLER_TYPE_LOCAL_MIN_P:
                llama_sampler_chain_add(result->chain, llama_sampler_init_min_p(params.min_p, params.min_keep));
                break;
            case COMMON_SAMPLER_TYPE_LOCAL_XTC:
                llama_sampler_chain_add(
                    result->chain,
                    llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                break;
            case COMMON_SAMPLER_TYPE_LOCAL_TYPICAL_P:
                llama_sampler_chain_add(result->chain, llama_sampler_init_typical(params.typ_p, params.min_keep));
                break;
            case COMMON_SAMPLER_TYPE_LOCAL_TEMPERATURE:
                llama_sampler_chain_add(result->chain, llama_sampler_init_temp_ext(params.temp, params.dynatemp_range,
                                                                                   params.dynatemp_exponent));
                break;
            case COMMON_SAMPLER_TYPE_LOCAL_INFILL:
                GGML_ABORT("llama_sampler_init_infill is not implemented");
                break;
            case COMMON_SAMPLER_TYPE_LOCAL_PENALTIES:
                llama_sampler_chain_add(result->chain,
                                        llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat,
                                                                     params.penalty_freq, params.penalty_present));
                break;
            default:
                GGML_ASSERT(false && "unknown sampler type");
        }
    }
    llama_sampler_chain_add(result->chain, llama_sampler_init_dist(params.seed));

    return result;
}

void common_sampler_local_free(struct common_sampler_local * gsmpl) {
    if (gsmpl) {
        llama_sampler_free(gsmpl->grmr);

        llama_sampler_free(gsmpl->chain);

        delete gsmpl;
    }
}

void common_sampler_accept(struct common_sampler_local * gsmpl, llama_token token, bool accept_grammar) {
    if (accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    if (gsmpl->prev.empty()) {
        gsmpl->prev.push_back(token);
    } else {
        gsmpl->prev[0] = token;
    }
}

void common_sampler_reset(struct common_sampler_local * gsmpl) {
    llama_sampler_reset(gsmpl->grmr);

    llama_sampler_reset(gsmpl->chain);
}

llama_token common_sampler_sample(struct common_sampler_local * gsmpl, struct llama_context * ctx, int idx,
                                  bool grammar_first) {
    gsmpl->set_logits(ctx, idx);

    auto & grmr  = gsmpl->grmr;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p;  // initialized by set_logits

    if (grammar_first) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    const llama_token id = cur_p.data[cur_p.selected].id;

    if (grammar_first) {
        return id;
    }

    // check if it the sampled token fits the grammar
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(grmr, &cur_p);
    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during re-sampling - check your sampling configuration");

    return cur_p.data[cur_p.selected].id;
}

// helpers

llama_token common_sampler_local_last(const struct common_sampler_local * gsmpl) {
    GGML_ASSERT(!gsmpl->prev.empty());
    return gsmpl->prev[0];
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler_local * gsmpl) {
    // TODO: measure grammar performance

    if (gsmpl) {
        llama_perf_sampler_print(gsmpl->chain);
    }
    if (ctx) {
        llama_perf_context_print(ctx);
    }
}
