#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include "ops_test_case.h"
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>
#include <utility>
#include <vector>

const int max_graph_nodes = 128;
void init_tensor(ggml_tensor* tensor, float min, float max){
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    for (auto &val : data){
        val = dis(gen);
    }
    std::cout << "Initializing tensor sample (first few values):" << std::endl;
    for (int i = 0; i < std::min(static_cast<size_t>(10), data.size()); i++) {
        std::cout << std::fixed << std::setprecision(4) << static_cast<float>(data[i]) << " ";
    }
    std::cout << std::endl;
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        memcpy(tensor->data, data.data(), data.size() * sizeof(float));
    }else{
        // unimplemented
    }
}
test_case::test_case(std::string name):name(std::move(name)){
    struct ggml_init_params init_params = {
    /* .mem_size */   ggml_tensor_overhead() * 32 + 1024 * 1024 * 128,  // 分配足够的内存
    /* .mem_buffer */ NULL,
    /* .no_alloc */   false
    };
    ctx = ggml_init(init_params);
    if (!ctx) {
        std::cerr << "Failed to initialize GGML context" << std::endl;
        return;
    }
    output_size = 0;
}
test_case::~test_case(){
    ggml_free(ctx);
}
void test_case_add::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_add::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_add(compute_ctx,src_tensors[0],src_tensors[1]);
}
void test_case_sub::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_sub::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_sub(compute_ctx,src_tensors[0],src_tensors[1]);
}
void test_case_mul::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_mul::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_mul(compute_ctx,src_tensors[0],src_tensors[1]);
}
void test_case_div::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_div::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_div(compute_ctx,src_tensors[0],src_tensors[1]);
}
void test_case_sqr::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_sqr::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_sqr(compute_ctx,src_tensors[0]);
}
void test_case_sum_rows::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = 1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_sum_rows::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_sum_rows(compute_ctx,src_tensors[0]);
}
void test_case_acc::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a_dim1, a_dim2, a_dim3, a_dim4));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, b_dim1, b_dim2, b_dim3, b_dim4));
    output_size = a_dim1 * a_dim2 * a_dim3 * a_dim4;
}
ggml_tensor* test_case_acc::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_acc(compute_ctx,src_tensors[0],src_tensors[1],
        src_tensors[0]->nb[1],src_tensors[0]->nb[2],src_tensors[0]->nb[3],src_tensors[1]->nb[1]);
}
void test_case_norm::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_norm::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_norm(compute_ctx,src_tensors[0],eps);
}
void test_case_group_norm::init_src_size(){
    assert(src.empty());
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim1, dim2, dim3, dim4));
    output_size = dim1 * dim2 * dim3 * dim4;
}
ggml_tensor* test_case_group_norm::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_group_norm(compute_ctx,src_tensors[0],num_group,eps);
}
void test_case_concat::init_src_size(){
    assert(src.empty());
    std::vector<int> ne_b = ne_a;
    std::vector<int> ne_c = ne_a;
    ne_b[dim] = ne_b_d;
    ne_c[dim] += ne_b_d;
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0],ne_a[1],ne_a[2],ne_a[3]));
    src.push_back(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_b[0],ne_b[1],ne_b[2],ne_b[3]));
    output_size = ne_c[0] * ne_c[1] * ne_c[2] * ne_c[3];
}
ggml_tensor* test_case_concat::build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const{
    return ggml_concat(compute_ctx,src_tensors[0],src_tensors[1],dim);
}