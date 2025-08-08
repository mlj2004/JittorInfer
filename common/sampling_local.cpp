#include "sampling_local.h"

#include <cmath>
#include <unordered_map>

#include "common_local.h"
#include "llama-context.h"

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template <typename T> struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos       = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first   = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz    = 0;
        first = 0;
        pos   = 0;
    }

    bool empty() const { return sz == 0; }

    size_t size() const { return sz; }

    size_t         capacity = 0;
    size_t         sz       = 0;
    size_t         first    = 0;
    size_t         pos      = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

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

struct common_sampler * common_sampler_init(const struct llama_model *            model,
                                            const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    struct llama_sampler * grmr;

    std::vector<const char *> trigger_words;
    trigger_words.reserve(params.grammar_trigger_words.size());
    for (const auto & str : params.grammar_trigger_words) {
        trigger_words.push_back(str.word.c_str());
    }

    grmr = llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");

    auto * result = new common_sampler{
        /* .params = */ params,
        /* .grmr   = */ grmr,
        /* .chain  = */ llama_sampler_chain_init(lparams),
        /* .prev   = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur    = */ {},
        /* .cur_p  = */ {},
    };

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

    llama_sampler_chain_add(
        result->chain,
        llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), params.logit_bias.size(), params.logit_bias.data()));

    llama_sampler_chain_add(result->chain, llama_sampler_init_dist(params.seed));
    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (gsmpl) {
        llama_sampler_free(gsmpl->grmr);

        llama_sampler_free(gsmpl->chain);

        delete gsmpl;
    }
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar) {
    if (accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    llama_sampler_reset(gsmpl->grmr);

    llama_sampler_reset(gsmpl->chain);
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx,
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
