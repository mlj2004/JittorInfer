#include <cmath>

#include "llama-context.h"
#include "llama-graph-utils.h"

class llm_defrag_context : public llm_build_context {
    const llama_kv_cache & kv_self;
    const llama_model &    model;
    const llama_hparams &  hparams;
    const llama_cparams &  cparams;
    const int              n_layer;
    const bool             flash_attn;
  public:
    llm_defrag_context(llama_context & lctx) :
        llm_build_context(lctx),
        kv_self(lctx.kv_self),
        model(lctx.model),
        hparams(model.hparams),
        cparams(lctx.cparams),
        n_layer(hparams.n_layer),
        flash_attn(cparams.flash_attn) {}

    struct ggml_cgraph * build_defrag(const std::vector<uint32_t> & ids) {
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(), false);

        // const int head_split = (hparams.enable_tensor_parallel && !hparams.enable_data_parallel) ? hparams.num_parallel : 1;
        for (uint32_t i = 0; i < ids.size(); ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == ids.size()) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < ids.size() && ids[i + nm] == id + nm) {
                nm++;
            }

            for (int il = 0; il < n_layer; ++il) {
                const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
                const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

                ggml_tensor * view_k_src = ggml_view_2d(ctx0, kv_self.k_l[il], n_embd_k_gqa, nm,
                                                        ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                                                        ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa * i));

                ggml_tensor * view_k_dst = ggml_view_2d(ctx0, kv_self.k_l[il], n_embd_k_gqa, nm,
                                                        ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                                                        ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa * id));

                ggml_tensor * view_v_src;
                ggml_tensor * view_v_dst;

                if (flash_attn) {
                    // NOTE: the V cache is not transposed when using flash attention
                    view_v_src = ggml_view_2d(ctx0, kv_self.v_l[il], n_embd_v_gqa, nm,
                                              ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                                              ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa * i));

                    view_v_dst = ggml_view_2d(ctx0, kv_self.v_l[il], n_embd_v_gqa, nm,
                                              ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa),
                                              ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa * id));
                } else {
                    view_v_src = ggml_view_2d(ctx0, kv_self.v_l[il], nm, n_embd_v_gqa,
                                              ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                                              ggml_row_size(kv_self.v_l[il]->type, i));

                    view_v_dst = ggml_view_2d(ctx0, kv_self.v_l[il], nm, n_embd_v_gqa,
                                              ggml_row_size(kv_self.v_l[il]->type, kv_self.size),
                                              ggml_row_size(kv_self.v_l[il]->type, id));
                }

                ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_k_src, view_k_dst));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, view_v_src, view_v_dst));
            }

            i += nm - 1;
        }

        //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);

        return gf;
    }
};

struct ggml_cgraph * llm_build_defrag(llama_context & lctx, const std::vector<uint32_t> & ids) {
    llm_defrag_context llm(lctx);

    llm.init();

    struct ggml_cgraph * result = llm.build_defrag(ids);

    llm.free();

    return result;
}
