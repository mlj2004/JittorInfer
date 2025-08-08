#include <vector>
#include <cstdint>
#include <string>

void flash_attn_cpu(
    const std::vector<float> &query,
    const std::vector<float> &key,
    const std::vector<float> &value,
    const std::vector<int8_t> &attn_mask,
    std::vector<float> &output,
    int64_t batch_size,
    int64_t num_heads,
    int64_t head_dims_kq,
    int64_t head_dims_v,
    int64_t key_num_heads,
    int64_t sequence_lenth_q,
    int64_t sequence_lenth_kv,
    float scaleValue,
    const std::string &layerOut = "BNSD"
);
