#ifndef LLAMA_GRAPH_DEFRAG_H
#define LLAMA_GRAPH_DEFRAG_H

#include "llama-graph-utils.h"

struct ggml_cgraph * llm_build_defrag(llama_context & lctx, const std::vector<uint32_t> & ids);

#endif  // LLAMA_GRAPH_DEFRAG_H
