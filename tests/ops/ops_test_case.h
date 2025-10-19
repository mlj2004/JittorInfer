#pragma once
#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include <cstdint>
#include <iostream>
#include <random>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>

void init_tensor(ggml_tensor* tensor, float min = -1.0f, float max = 1.0f);

struct test_case{
    ggml_context* ctx;
    std::vector <ggml_tensor*> src;
    int64_t output_size;

    std::string name;
    // 初始化算子源矩阵
    virtual void init_src_size()=0;
    // 构建计算图关系
    virtual ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const=0;
    // 输出
    void print(){}
    // 初始化矩阵
    
    test_case(std::string name);
    virtual ~test_case();
};

struct test_case_add:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_add(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("ADD"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_mul:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_mul(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("MUL"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_sub:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_sub(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("SUB"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_div:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_div(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("DIV"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};