#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>

#include "ggml-backend.h"
#include "ggml-cann.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-impl.h"

void attention_cpu(const std::vector<float> & query, const std::vector<float> & key, const std::vector<float> & value,
                   const std::vector<int8_t> & attn_mask, std::vector<float> & output, int64_t batch_size,
                   int64_t num_heads, int64_t head_dims_kq, int64_t head_dims_v, int64_t key_num_heads,
                   int64_t sequence_lenth_q, int64_t sequence_lenth_kv, float scaleValue) {
    // 初始化输出向量
    output.resize(batch_size * num_heads * sequence_lenth_q * head_dims_v, 0.0f);

    // 使用类似wkv_b_post_process的方式实现
    struct ggml_init_params init_params = { /* .mem_size */ ggml_tensor_overhead() * 32 +
                                                1024 * 1024 * 128,  // 分配足够的内存
                                            /* .mem_buffer */ NULL,
                                            /* .no_alloc */ false };

    // 创建GGML上下文
    ggml_context * ctx = ggml_init(init_params);
    if (!ctx) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        return;
    }

    // 创建输入张量
    // 查询张量: [batch_size, num_heads, sequence_lenth_q, head_dims]
    ggml_tensor * q_tensor =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_kq, sequence_lenth_q, num_heads, batch_size);
    // 键张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor * k_tensor =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_kq, sequence_lenth_kv, key_num_heads, batch_size);
    // 值张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor * v_tensor =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_v, sequence_lenth_kv, key_num_heads, batch_size);

    // 掩码张量: [batch_size, 1, sequence_lenth_q, sequence_lenth_kv]
    ggml_tensor * mask_tensor =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sequence_lenth_kv, sequence_lenth_q, 1, batch_size);

    // 输出张量: [batch_size, num_heads, sequence_lenth_q, head_dims]
    ggml_tensor * output_tensor =
        ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dims_v, sequence_lenth_q, num_heads, batch_size);

    // 复制数据到张量
    memcpy(q_tensor->data, query.data(), query.size() * sizeof(float));
    memcpy(k_tensor->data, key.data(), key.size() * sizeof(float));
    memcpy(v_tensor->data, value.data(), value.size() * sizeof(float));

    std::vector<float> mask_data(attn_mask.size());
    for (size_t i = 0; i < attn_mask.size(); ++i) {
        mask_data[i] = attn_mask[i] ? -INFINITY : 0.0f;
    }
    memcpy(mask_tensor->data, mask_data.data(), attn_mask.size() * sizeof(float));

    // 构建计算图
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 100, false);

    ggml_tensor * q_cur = q_tensor;

    // 2. 计算 Q*K^T
    ggml_tensor * kq = ggml_mul_mat(ctx, k_tensor, q_cur);  // 矩阵乘法

    // 3. 应用缩放因子
    // kq = ggml_scale(ctx, kq, scaleValue);

    // 4. 应用注意力掩码（如果有）并计算softmax
    ggml_tensor * kq_soft_max = ggml_soft_max_ext(ctx, kq, mask_tensor, scaleValue, 0.0f);
    // if (mask_tensor && !attn_mask.empty()) {
    //     kq_soft_max = ggml_soft_max_ext(ctx, kq, mask_tensor, 1.0f, 0.0f);
    // } else {
    //     kq_soft_max = ggml_soft_max(ctx, kq);
    // }

    // 5. 计算 Attention * V
    ggml_tensor * v_cur = ggml_cont(ctx, ggml_permute(ctx, v_tensor, 1, 0, 2, 3));
    ggml_tensor * kqv   = ggml_mul_mat(ctx, v_cur, kq_soft_max);

    // 6. 重新排列结果
    ggml_tensor * kqv_merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // BDSH -> BHSD

    // 7. 转换为连续内存布局
    ggml_tensor * result = ggml_cpy(ctx, kqv_merged, output_tensor);

    // 添加到计算图中
    ggml_build_forward_expand(gf, result);

    // 执行计算
    ggml_build_forward_expand(gf, result);
    struct ggml_cplan cplan = ggml_graph_plan(gf, 1, NULL);

    // 为 cplan 分配 work_data
    if (cplan.work_size > 0) {
        void * work_memory = malloc(cplan.work_size);
        if (!work_memory) {
            std::cerr << "Failed to allocate work memory for cplan" << std::endl;
            ggml_free(ctx);
            return;
        }
        cplan.work_data = (uint8_t *) work_memory;
    }

    ggml_status status = ggml_graph_compute(gf, &cplan);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("attention_cpu failed with status : %d\n", status);
        GGML_ABORT("attention_cpu failed");
    }

    // 释放 work_data
    if (cplan.work_size > 0 && cplan.work_data) {
        free(cplan.work_data);
    }
    // 复制结果到输出向量
    memcpy(output.data(), output_tensor->data, output.size() * sizeof(float));

    // 释放资源
    ggml_free(ctx);
}

int main() {
    std::cout << "Starting Flash Attention test..." << std::endl;

    // Define tensor dimensions
    int64_t batch_size        = 1;
    int64_t num_heads         = 16;   // Reduced from 128 to a smaller value
    int64_t head_dims_kq      = 192;  // Reduced from 192 to a smaller value
    int64_t head_dims_v       = 128;  // Reduced from 192 to a smaller value
    int64_t key_num_heads     = num_heads;
    int64_t sequence_lenth_q  = 64;
    int64_t sequence_lenth_kv = 128;
    float   scaleValue        = 1.0f / std::sqrt(static_cast<float>(head_dims_kq));

    std::cout << "Dimensions: batch_size=" << batch_size << ", num_heads=" << num_heads
              << ", head_dims_kq=" << head_dims_kq << ", head_dims_v=" << head_dims_v << ", seq_q=" << sequence_lenth_q
              << ", seq_kv=" << sequence_lenth_kv << std::endl;

    // Calculate sizes for tensors
    int64_t query_size     = batch_size * num_heads * sequence_lenth_q * head_dims_kq;
    int64_t key_size       = batch_size * key_num_heads * sequence_lenth_kv * head_dims_kq;
    int64_t value_size     = batch_size * key_num_heads * sequence_lenth_kv * head_dims_v;
    int64_t attn_mask_size = batch_size * 1 * sequence_lenth_q * sequence_lenth_kv;
    int64_t output_size    = batch_size * num_heads * sequence_lenth_q * head_dims_v;

    std::cout << "Allocating memory for tensors..." << std::endl;
    std::cout << "Query size: " << query_size << std::endl;
    std::cout << "Key size: " << key_size << std::endl;
    std::cout << "Value size: " << value_size << std::endl;
    std::cout << "Attention mask size: " << attn_mask_size << std::endl;
    std::cout << "Output size: " << output_size << std::endl;

    // Initialize tensors
    std::vector<float>  query_host(query_size);
    std::vector<float>  key_host(key_size);
    std::vector<float>  value_host(value_size);
    std::vector<int8_t> attn_mask(attn_mask_size, 0);  // Initialize all to 0

    // 随机设置一些 attn_mask 元素为 1（表示需要被屏蔽的位置）
    // 使用与初始化张量相同的随机数生成器
    std::random_device          mask_rd;
    std::mt19937                mask_gen(mask_rd());
    std::bernoulli_distribution mask_dist(0.3);  // 30% 的概率设置为 1
    for (int64_t i = 0; i < attn_mask_size; ++i) {
        attn_mask[i] = mask_dist(mask_gen) ? 1 : 0;
    }

    // 打印部分 attn_mask 信息
    std::cout << "Attention mask sample:" << std::endl;
    for (int i = 0; i < std::min(static_cast<int64_t>(16), static_cast<int64_t>(attn_mask_size)); ++i) {
        std::cout << static_cast<int>(attn_mask[i]) << " ";
    }
    std::cout << std::endl;

    std::vector<float> output_host(output_size);

    // Random initialization
    std::cout << "Initializing tensors with random values..." << std::endl;
    std::random_device                    rd;
    std::mt19937                          gen(rd());
    std::uniform_real_distribution<float> dis(-0.1, 0.1);  // Smaller range for numerical stability

    for (auto & val : query_host) {
        val = dis(gen);
    }
    for (auto & val : key_host) {
        val = dis(gen);
    }
    for (auto & val : value_host) {
        val = dis(gen);
    }

    // memset(output_host.data(), 0, output_size * sizeof(float));
    // attention_cpu(query_host, key_host, value_host, attn_mask, output_host,
    //     batch_size, num_heads, head_dims_kq, head_dims_v, key_num_heads,
    //     sequence_lenth_q, sequence_lenth_kv, scaleValue);

    // for (int i = 0; i < std::min(static_cast<int64_t>(10), output_size); i++) {
    //     std::cout << std::fixed << std::setprecision(4) << static_cast<float>(output_host[i]) << " ";
    // }
    std::cout << std::endl;
    std::vector<float> output_host_cann(output_size);
    memset(output_host_cann.data(), 0, output_size * sizeof(float));

    // Call flash attention function
    std::cout << "Calling flash attention function..." << std::endl;
    int result = ggml_cann_prompt_flash_attention(query_host, key_host, value_host, attn_mask, output_host_cann,
                                                  batch_size, num_heads, head_dims_kq, head_dims_v, key_num_heads,
                                                  sequence_lenth_q, sequence_lenth_kv, scaleValue);

    if (result != 0) {
        std::cerr << "Flash attention failed with error code: " << result << std::endl;
        return 1;
    }

    std::cout << "Flash attention completed successfully." << std::endl;
    std::cout << "Output tensor sample (first few values):" << std::endl;

    // Print only a small sample of the output to avoid flooding the console
    for (int i = 0; i < std::min(static_cast<int64_t>(10), output_size); i++) {
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(output_host_cann[i]) << " ";
    }
    std::cout << std::endl;

    // 比较 CPU 和 CANN 实现的结果
    std::cout << "\nComparing CPU and CANN implementations:" << std::endl;

    double max_abs_error     = 0.0;
    double max_rel_error     = 0.0;
    int    max_abs_error_idx = 0;
    int    max_rel_error_idx = 0;

    for (int64_t i = 0; i < output_size; ++i) {
        double abs_error = std::fabs(output_host[i] - output_host_cann[i]);
        double rel_error = 0.0;

        // 计算相对误差，避免除以0
        if (std::fabs(output_host[i]) > 1e-10) {
            rel_error = abs_error / std::fabs(output_host[i]);
        } else if (std::fabs(output_host_cann[i]) > 1e-10) {
            rel_error = abs_error / std::fabs(output_host_cann[i]);
        }

        if (abs_error > max_abs_error) {
            max_abs_error     = abs_error;
            max_abs_error_idx = i;
        }

        if (rel_error > max_rel_error) {
            max_rel_error     = rel_error;
            max_rel_error_idx = i;
        }
    }

    std::cout << "Maximum absolute error: " << max_abs_error << " at index " << max_abs_error_idx
              << " (CPU: " << output_host[max_abs_error_idx] << ", CANN: " << output_host_cann[max_abs_error_idx] << ")"
              << std::endl;

    std::cout << "Maximum relative error: " << max_rel_error << " at index " << max_rel_error_idx
              << " (CPU: " << output_host[max_rel_error_idx] << ", CANN: " << output_host_cann[max_rel_error_idx] << ")"
              << std::endl;

    return 0;
}
