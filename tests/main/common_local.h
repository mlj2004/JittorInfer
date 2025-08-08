#pragma once

#include <set>
#include <string>
#include <vector>

#include "llama.h"

enum common_params_local_type {
    COMMON_SAMPLER_TYPE_LOCAL_NONE        = 0,
    COMMON_SAMPLER_TYPE_LOCAL_DRY         = 1,
    COMMON_SAMPLER_TYPE_LOCAL_TOP_K       = 2,
    COMMON_SAMPLER_TYPE_LOCAL_TOP_P       = 3,
    COMMON_SAMPLER_TYPE_LOCAL_MIN_P       = 4,
    //COMMON_SAMPLER_TYPE_LOCAL_TFS_Z       = 5,
    COMMON_SAMPLER_TYPE_LOCAL_TYPICAL_P   = 6,
    COMMON_SAMPLER_TYPE_LOCAL_TEMPERATURE = 7,
    COMMON_SAMPLER_TYPE_LOCAL_XTC         = 8,
    COMMON_SAMPLER_TYPE_LOCAL_INFILL      = 9,
    COMMON_SAMPLER_TYPE_LOCAL_PENALTIES   = 10,
};

enum common_conversation_mode {
    COMMON_CONVERSATION_MODE_DISABLED = 0,
    COMMON_CONVERSATION_MODE_ENABLED  = 1,
    COMMON_CONVERSATION_MODE_AUTO     = 2,
};

// note: defines object's lifetime
struct common_init_result {
    llama_model_ptr   model;
    llama_context_ptr context;

    // std::vector<llama_adapter_lora_ptr> lora;
};

struct common_grammar_local_trigger {
    std::string word;
    bool        at_start;
};

struct common_adapter_lora_info {
    std::string path;
    float       scale;

    struct llama_adapter_lora * ptr;
};

// sampling parameters
struct common_params_local_sampling {
    uint32_t seed = LLAMA_DEFAULT_SEED;  // the seed used to initialize llama_sampler

    int32_t n_prev            = 64;      // number of previous tokens to remember
    int32_t n_probs           = 0;       // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep          = 0;       // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t top_k             = 40;      // <= 0 to use vocab size
    float   top_p             = 0.95f;   // 1.0 = disabled
    float   min_p             = 0.05f;   // 0.0 = disabled
    float   xtc_probability   = 0.00f;   // 0.0 = disabled
    float   xtc_threshold     = 0.10f;   // > 0.5 disables XTC
    float   typ_p             = 1.00f;   // typical_p, 1.0 = disabled
    float   temp              = 0.80f;   // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   dynatemp_range    = 0.00f;   // 0.0 = disabled
    float   dynatemp_exponent = 1.00f;   // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penalty_last_n    = 64;      // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penalty_repeat    = 1.00f;   // 1.0 = disabled
    float   penalty_freq      = 0.00f;   // 0.0 = disabled
    float   penalty_present   = 0.00f;   // 0.0 = disabled
    float   dry_multiplier    = 0.0f;    // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    float   dry_base =
        1.75f;  // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    int32_t dry_allowed_length = 2;  // tokens extending repetitions beyond this receive penalty
    int32_t dry_penalty_last_n =
        -1;                          // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
    int32_t mirostat         = 0;    // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   top_n_sigma      = -1.00f;  // -1.0 = disabled
    float   mirostat_tau     = 5.00f;   // target entropy
    float   mirostat_eta     = 0.10f;   // learning rate
    bool    ignore_eos       = false;
    bool    no_perf          = false;   // disable performance metrics
    bool    timing_per_token = false;

    std::vector<std::string> dry_sequence_breakers = { "\n", ":", "\"", "*" };  // default sequence breakers for DRY

    std::vector<enum common_params_local_type> samplers = {
        COMMON_SAMPLER_TYPE_LOCAL_PENALTIES, COMMON_SAMPLER_TYPE_LOCAL_DRY,         COMMON_SAMPLER_TYPE_LOCAL_TOP_K,
        COMMON_SAMPLER_TYPE_LOCAL_TYPICAL_P, COMMON_SAMPLER_TYPE_LOCAL_TOP_P,       COMMON_SAMPLER_TYPE_LOCAL_MIN_P,
        COMMON_SAMPLER_TYPE_LOCAL_XTC,       COMMON_SAMPLER_TYPE_LOCAL_TEMPERATURE,
    };

    std::string                               grammar;                // optional BNF-like grammar to constrain sampling
    bool                                      grammar_lazy = false;
    std::vector<common_grammar_local_trigger> grammar_trigger_words;  // optional trigger words to trigger lazy grammar
    std::vector<llama_token>
        grammar_trigger_tokens;  // optional trigger tokens to trigger lazy grammar and print trigger special tokens.
    std::set<llama_token> preserved_tokens;

    std::vector<llama_logit_bias> logit_bias;  // logit biases to apply

    // print the parameters into a string
    std::string print() const;
};

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special = true);

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special = true);

// call once at the start of a program if it uses libcommon
// initializes the logging system and prints info about the build
bool set_process_priority(enum ggml_sched_priority prio);

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> common_tokenize(const struct llama_context * ctx, const std::string & text, bool add_special,
                                         bool parse_special = false);

std::vector<llama_token> common_tokenize(const struct llama_vocab * vocab, const std::string & text, bool add_special,
                                         bool parse_special = false);

void string_process_escapes(std::string & input);

//
// Batch utils
//

void common_batch_clear(struct llama_batch & batch);

void common_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos,
                      const std::vector<llama_seq_id> & seq_ids, bool logits);

size_t common_llama_batch_max_comm_size(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max);
size_t common_dump_batch(const struct llama_batch & batch, int32_t embd, char * data);
void   common_load_batch(struct llama_batch & batch, int32_t embd, const char * data);

//
// KV cache utils
//

// Dump the KV cache view showing individual sequences in each cell (long output).
void common_kv_cache_dump_view_seqs(const llama_kv_cache_view & view, int row_size = 40);

int get_batch_index(int index, int n_thread, int n_tasks);
