#include "common_local.h"

#include <sys/resource.h>
#include <sys/types.h>

#include <cerrno>   // for errno
#include <cstdint>
#include <cstring>  // for strerror
#include <string>
#include <unordered_map>

#include "llama-kv-cache.h"

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }

    return piece;
}

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:
            p = 0;
            break;
        case GGML_SCHED_PRIO_MEDIUM:
            p = -5;
            break;
        case GGML_SCHED_PRIO_HIGH:
            p = -10;
            break;
        case GGML_SCHED_PRIO_REALTIME:
            p = -20;
            break;
    }

    if (!setpriority(PRIO_PROCESS, 0, p)) {
        fprintf(stderr, "failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

std::vector<llama_token> common_tokenize(const struct llama_context * ctx, const std::string & text, bool add_special,
                                         bool parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(const struct llama_vocab * vocab, const std::string & text, bool add_special,
                                         bool parse_special) {
    // upper limit for the number of tokens
    int                      n_tokens = text.length() + (2 * add_special);
    std::vector<llama_token> result(n_tokens);
    n_tokens =
        llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check =
            llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

void string_process_escapes(std::string & input) {
    std::size_t input_len  = input.length();
    std::size_t output_idx = 0;

    for (std::size_t input_idx = 0; input_idx < input_len; ++input_idx) {
        if (input[input_idx] == '\\' && input_idx + 1 < input_len) {
            switch (input[++input_idx]) {
                case 'n':
                    input[output_idx++] = '\n';
                    break;
                case 'r':
                    input[output_idx++] = '\r';
                    break;
                case 't':
                    input[output_idx++] = '\t';
                    break;
                case '\'':
                    input[output_idx++] = '\'';
                    break;
                case '\"':
                    input[output_idx++] = '\"';
                    break;
                case '\\':
                    input[output_idx++] = '\\';
                    break;
                case 'x':
                    // Handle \x12, etc
                    if (input_idx + 2 < input_len) {
                        const char x[3]  = { input[input_idx + 1], input[input_idx + 2], 0 };
                        char *     err_p = nullptr;
                        const long val   = std::strtol(x, &err_p, 16);
                        if (err_p == x + 2) {
                            input_idx += 2;
                            input[output_idx++] = char(val);
                            break;
                        }
                    }
                    // fall through
                default:
                    input[output_idx++] = '\\';
                    input[output_idx++] = input[input_idx];
                    break;
            }
        } else {
            input[output_idx++] = input[input_idx];
        }
    }

    input.resize(output_idx);
}

//
// Batch Utils
//

void common_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void common_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos,
                      const std::vector<llama_seq_id> & seq_ids, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token[batch.n_tokens]    = id;
    batch.pos[batch.n_tokens]      = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;

    batch.n_tokens++;
}

static void copy_move(const void * src, char * dst, size_t & off, size_t size) {
    memcpy(dst + off, src, size);
    off += size;
}

size_t common_dump_batch(const struct llama_batch & batch, int32_t embd, char * data) {
    size_t        now_id        = 0;
    const int32_t n_tokens      = batch.n_tokens;
    const size_t  token_size    = (embd ? sizeof(float) * embd : sizeof(llama_token)) * n_tokens;
    const size_t  pos_size      = sizeof(llama_pos) * n_tokens;
    const size_t  n_seq_id_size = sizeof(int32_t) * n_tokens;
    const size_t  logits_size   = sizeof(int8_t) * n_tokens;
    copy_move((void *) &batch.n_tokens, data, now_id, sizeof(batch.n_tokens));
    copy_move((void *) batch.token, data, now_id, token_size);
    copy_move((void *) batch.pos, data, now_id, pos_size);
    copy_move((void *) batch.n_seq_id, data, now_id, n_seq_id_size);
    copy_move((void *) batch.logits, data, now_id, logits_size);
    for (int i = 0; i < n_tokens; ++i) {
        copy_move((void *) batch.seq_id[i], data, now_id, sizeof(llama_seq_id) * batch.n_seq_id[i]);
    }
    return now_id;
}

static void read_move(void * dst, const char * src, size_t & off, size_t size) {
    memcpy(dst, src + off, size);
    off += size;
}

void common_load_batch(struct llama_batch & batch, int32_t embd, const char * data) {
    size_t now_id = 0;
    read_move((void *) &batch.n_tokens, data, now_id, sizeof(batch.n_tokens));
    const int32_t n_tokens      = batch.n_tokens;
    const size_t  token_size    = (embd ? sizeof(float) * embd : sizeof(llama_token)) * n_tokens;
    const size_t  pos_size      = sizeof(llama_pos) * n_tokens;
    const size_t  n_seq_id_size = sizeof(int32_t) * n_tokens;
    const size_t  logits_size   = sizeof(int8_t) * n_tokens;
    read_move((void *) batch.token, data, now_id, token_size);
    read_move((void *) batch.pos, data, now_id, pos_size);
    read_move((void *) batch.n_seq_id, data, now_id, n_seq_id_size);
    read_move((void *) batch.logits, data, now_id, logits_size);
    for (int i = 0; i < n_tokens; ++i) {
        read_move((void *) batch.seq_id[i], data, now_id, sizeof(llama_seq_id) * batch.n_seq_id[i]);
    }
}

size_t common_llama_batch_max_comm_size(int32_t n_tokens_alloc, int32_t embd, int32_t n_seq_max) {
    const int32_t n_tokens      = n_tokens_alloc;
    const size_t  n_tokens_size = sizeof(n_tokens);
    const size_t  token_size    = (embd ? sizeof(float) * embd : sizeof(llama_token)) * n_tokens;
    const size_t  pos_size      = sizeof(llama_pos) * n_tokens;
    const size_t  n_seq_id_size = sizeof(int32_t) * n_tokens;
    const size_t  logits_size   = sizeof(int8_t) * n_tokens;
    const size_t  seq_id_size   = sizeof(llama_seq_id) * n_seq_max;
    return n_tokens_size + token_size + pos_size + n_seq_id_size + logits_size + seq_id_size;
}

//
// KV cache utils
//

void common_kv_cache_dump_view_seqs(const llama_kv_cache_view & view, int row_size) {
    static const char slot_chars[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    printf(
        "=== Dumping KV cache. total cells %d, max sequences per cell %d, populated cells %d, total tokens in cache "
        "%d, largest empty slot=%d @ %d\n",
        view.n_cells, view.n_seq_max, view.used_cells, view.token_count, view.max_contiguous, view.max_contiguous_idx);

    std::unordered_map<llama_seq_id, size_t> seqs;
    llama_kv_cache_view_cell *               c_curr  = view.cells;
    llama_seq_id *                           cs_curr = view.cells_sequences;

    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] < 0) {
                continue;
            }
            if (seqs.find(cs_curr[j]) == seqs.end()) {
                if (seqs.size() + 1 >= sizeof(slot_chars)) {
                    break;
                }
                const size_t sz  = seqs.size();
                seqs[cs_curr[j]] = sz;
            }
        }
        if (seqs.size() + 1 >= sizeof(slot_chars)) {
            break;
        }
    }

    printf("=== Sequence legend: ");
    for (const auto & it : seqs) {
        printf("%zu=%d, ", it.second, it.first);
    }
    printf("'+'=other sequence ids");

    c_curr  = view.cells;
    cs_curr = view.cells_sequences;
    for (int i = 0; i < view.n_cells; i++, c_curr++, cs_curr += view.n_seq_max) {
        if (i % row_size == 0) {
            printf("\n%5d: ", i);
        }
        for (int j = 0; j < view.n_seq_max; j++) {
            if (cs_curr[j] >= 0) {
                const auto & it = seqs.find(cs_curr[j]);
                putchar(it != seqs.end() ? int(slot_chars[it->second]) : '+');
            } else {
                putchar('.');
            }
        }
        putchar(' ');
    }

    printf("\n=== Done dumping\n");
}

int get_batch_index(int index, int n_thread, int n_tasks) {
    if (index < 0) {
        return -1;
    }
    int stride = n_tasks / n_thread;
    return std::min((index - index % stride) / stride, n_thread - 1);
}
