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