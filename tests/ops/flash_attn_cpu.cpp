#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "flash_attn_cpu.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstring>

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
    const std::string &layerOut
) {
    // 初始化输出向量
    output.resize(batch_size * num_heads * sequence_lenth_q * head_dims_v, 0.0f);
    
    // 使用类似wkv_b_post_process的方式实现
    struct ggml_init_params init_params = {
        /* .mem_size */   ggml_tensor_overhead() * 32 + 1024 * 1024 * 128,  // 分配足够的内存
        /* .mem_buffer */ NULL,
        /* .no_alloc */   false
    };
    
    // 创建GGML上下文
    ggml_context* ctx = ggml_init(init_params);
    if (!ctx) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        return;
    }
    
    // 创建输入张量
    // 查询张量: [batch_size, num_heads, sequence_lenth_q, head_dims]
    GGML_ASSERT(layerOut == "BNSD");
    ggml_tensor* q_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                            head_dims_kq, sequence_lenth_q, num_heads, batch_size);
    // 键张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor* k_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                            head_dims_kq, sequence_lenth_kv, key_num_heads, batch_size);
    // 值张量: [batch_size, key_num_heads, sequence_lenth_kv, head_dims]
    ggml_tensor* v_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                            head_dims_v, sequence_lenth_kv, key_num_heads, batch_size);
    
    // 掩码张量: [batch_size, 1, sequence_lenth_q, sequence_lenth_kv]
    ggml_tensor* mask_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                    sequence_lenth_kv, sequence_lenth_q, 1, batch_size);
    
    // 输出张量: [batch_size, num_heads, sequence_lenth_q, head_dims]
    ggml_tensor* output_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                head_dims_v, sequence_lenth_q, num_heads, batch_size);
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
    
    ggml_tensor* q_cur = q_tensor;
    
    // 2. 计算 Q*K^T
    ggml_tensor* kq = ggml_mul_mat(ctx, k_tensor, q_cur);  // 矩阵乘法
    
    // 3. 应用缩放因子
    // kq = ggml_scale(ctx, kq, scaleValue);
    
    // 4. 应用注意力掩码（如果有）并计算softmax
    ggml_tensor* kq_soft_max = ggml_soft_max_ext(ctx, kq, mask_tensor, scaleValue, 0.0f);
    // if (mask_tensor && !attn_mask.empty()) {
    //     kq_soft_max = ggml_soft_max_ext(ctx, kq, mask_tensor, 1.0f, 0.0f);
    // } else {
    //     kq_soft_max = ggml_soft_max(ctx, kq);
    // }
    
    // 5. 计算 Attention * V
    ggml_tensor* v_cur = ggml_cont(ctx, ggml_permute(ctx, v_tensor, 1, 0, 2, 3));
    ggml_tensor* kqv = ggml_mul_mat(ctx, v_cur, kq_soft_max);
    
    // 6. 重新排列结果
    // ggml_tensor* kqv_merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));  // BDSH -> BHSD
    
    // 7. 转换为连续内存布局
    ggml_tensor* result = ggml_cpy(ctx, kqv, output_tensor);
    // ggml_tensor* result = ggml_cpy(ctx, kqv_merged, output_tensor);
    
    // 添加到计算图中
    ggml_build_forward_expand(gf, result);
    
    // 执行计算
    ggml_build_forward_expand(gf, result);
    struct ggml_cplan cplan = ggml_graph_plan(gf, 1, NULL);
    
    // 为 cplan 分配 work_data
    if (cplan.work_size > 0) {
        void* work_memory = malloc(cplan.work_size);
        if (!work_memory) {
            std::cerr << "Failed to allocate work memory for cplan" << std::endl;
            ggml_free(ctx);
            return;
        }
        cplan.work_data = (uint8_t*)work_memory;
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
