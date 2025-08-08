#include "llama-batch.h"
#include "llama.h"

namespace llama_graph_builder {
enum ggml_status llama_graph_compute(llama_context & lctx, ggml_cgraph * gf, ggml_backend_sched_t sched, int n_threads,
                                     ggml_threadpool * threadpool);
struct ggml_cgraph * llama_build_graph(llama_context & lctx, std::vector<uint8_t> & buf_compute_meta,
                                       const llama_ubatch & ubatch, bool worst_case, int print_layer = -1);
void                 llama_update_graph(llama_context & lctx, const llama_ubatch & ubatch, bool worst_case);

struct ggml_cgraph * llama_build_graph_defrag(llama_context & lctx, const std::vector<uint32_t> & ids);
}  // namespace llama_graph_builder
