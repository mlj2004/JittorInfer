#include <cassert>
#include <cmath>
#include <cstring>

#include "ggml.h"
#include "llama-context.h"
#include "llama-graph-builder.h"
#include "llama-impl.h"
#include "llama.h"

// find holes from the beginning of the KV cache and fill them by moving data from the end of the cache
static void llama_kv_cache_defrag_impl(struct llama_context & lctx) {
    auto & kv_self = lctx.kv_self;

    const auto & hparams = lctx.model.hparams;

    const uint32_t n_layer = hparams.n_layer;

    const uint32_t n_kv   = llama_kv_cache_cell_max(kv_self);
    const uint32_t n_used = kv_self.used;

    assert(n_used <= n_kv);

    //const int64_t t_start = ggml_time_us();

    // number of cells moved
    uint32_t n_moves = 0;

    // each move requires 6*n_layer tensors (see build_defrag)
    //   - source view, destination view, copy operation
    //   - x2 for keys and values
    //const uint32_t max_moves = model.max_nodes()/(6*n_layer);
    // TODO: tmp fix https://github.com/ggerganov/llama.cpp/issues/6685#issuecomment-2057579516
    const uint32_t max_moves = (lctx.model.max_nodes() - 2 * n_layer) / (6 * n_layer);

    // determine which KV cells to move where
    //
    //  cell i moves to ids[i]
    //
    //  if ids[i] == i || ids[i] == n_kv, then cell i is not moved
    //
    std::vector<uint32_t> ids(n_kv, n_kv);

    for (uint32_t i0 = 0; i0 < n_used; ++i0) {
        const auto & cell0 = kv_self.cells[i0];

        if (!cell0.is_empty()) {
            ids[i0] = i0;

            continue;
        }

        // found a hole - fill it with data from the end of the cache

        uint32_t nh = 1;

        // determine the size of the hole
        while (i0 + nh < n_used && kv_self.cells[i0 + nh].is_empty()) {
            nh++;
        }

        uint32_t nf = 0;
        uint32_t is = n_kv - 1;

        // starting from the end, find nh non-empty cells
        for (; is > i0; --is) {
            const auto & cell1 = kv_self.cells[is];

            if (cell1.is_empty() || ids[is] != n_kv) {
                continue;
            }

            // non-empty cell which is not yet moved
            nf++;

            if (nf == nh) {
                break;
            }
        }

        // this can only happen if `n_used` is not accurate, which would be a bug
        GGML_ASSERT(nf == nh && "KV defrag bug: nf != nh");

        nf = 0;

        uint32_t i1 = is;

        // are we moving a continuous block of memory?
        bool cont = false;

        // should we stop searching for the next move?
        bool stop = false;

        // go back and move the nf cells to the hole
        for (; i1 < n_kv; ++i1) {
            auto & cell1 = kv_self.cells[i1];

            if (cell1.is_empty() || ids[i1] != n_kv) {
                if (n_moves == max_moves) {
                    stop = true;
                    break;
                }

                cont = false;
                continue;
            }

            // this cell goes to (i0 + nf)
            ids[i1] = i0 + nf;

            // move the cell meta data
            kv_self.cells[i0 + nf] = cell1;

            // clear the old cell and move the head there
            cell1        = llama_kv_cell();
            kv_self.head = n_used;

            if (!cont) {
                n_moves++;
                cont = true;
            }

            nf++;

            if (nf == nh) {
                break;
            }
        }

        if (stop || n_moves == max_moves) {
            break;
        }

        //LLAMA_LOG_INFO("(tmp log) KV defrag: move [%u, %u) to [%u, %u)\n", is, i1 + 1, i0, i0 + nh);

        i0 += nh - 1;
    }

    if (n_moves == 0) {
        return;
    }
    // ggml_graph defrag

    ggml_backend_sched_reset(lctx.sched.get());

    ggml_cgraph * gf = llama_graph_builder::llama_build_graph_defrag(lctx, ids);

    llama_graph_builder::llama_graph_compute(lctx, gf, lctx.sched.get(), lctx.cparams.n_threads, lctx.threadpool);
}

static void llama_kv_cache_update_impl(struct llama_context & lctx) {
    bool need_reserve = false;

    // Simplifed: we do not support shift
    GGML_ASSERT(!lctx.kv_self.has_shift);

    // defragment the KV cache if needed
    if (lctx.kv_self.do_defrag) {
        llama_kv_cache_defrag_impl(lctx);

        need_reserve = true;

        lctx.kv_self.do_defrag = false;
    }

    // reserve a worst case graph again
    if (need_reserve && !lctx.cparams.enable_ge) {
        // TODO: extract to a function
        // build worst-case graph
        uint32_t    n_seqs   = 1;  // TODO: worst-case number of sequences
        uint32_t    n_tokens = std::min(lctx.cparams.n_ctx, lctx.cparams.n_ubatch);
        llama_token token =
            lctx.model.vocab
                .token_bos();  // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph
        llama_ubatch  ubatch = { true,    n_tokens, n_tokens / n_seqs, n_seqs,  &token,
                                 nullptr, nullptr,  nullptr,           nullptr, nullptr };
        ggml_cgraph * gf     = llama_graph_builder::llama_build_graph(lctx, lctx.buf_compute_meta, ubatch, true);

        // initialize scheduler with the worst-case graph
        ggml_backend_sched_reset(lctx.sched.get());
        if (!ggml_backend_sched_reserve(lctx.sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        }
    }
}

void llama_kv_cache_update(struct llama_context * ctx) {
    llama_kv_cache_update_impl(*ctx);
}
