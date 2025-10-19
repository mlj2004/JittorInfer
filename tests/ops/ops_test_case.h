#pragma once
#include "ggml-cann.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "llama-impl.h"
#include <cstdint>
#include <cstdio>
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
struct test_case_sqr:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_sqr(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("SQR"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_sum_rows:test_case{
    int dim1,dim2,dim3,dim4;
    test_case_sum_rows(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),test_case("SUM_ROWS"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_acc:test_case{
    int a_dim1,a_dim2,a_dim3,a_dim4,
    b_dim1,b_dim2,b_dim3,b_dim4,
    nb1,nb2,nb3,offset;
    test_case_acc(int a_dim1 = 256,int a_dim2 = 17,int a_dim3 = 1, int a_dim4 = 1,
        int b_dim1 = 256,int b_dim2 = 16,int b_dim3 = 1, int b_dim4 = 1):
        a_dim1(a_dim1),a_dim2(a_dim2),a_dim3(a_dim3),a_dim4(a_dim4),
        b_dim1(b_dim1),b_dim2(b_dim2),b_dim3(b_dim3),b_dim4(b_dim4),
        test_case("ACC"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_norm:test_case{
    int dim1,dim2,dim3,dim4;
    float eps;
    test_case_norm(int dim1 = 64,int dim2 = 64,int dim3 =4, int dim4 = 4,float eps = 1e-6f):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),eps(eps),test_case("NORM"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_group_norm:test_case{
    int dim1,dim2,dim3,dim4;
    int32_t num_group;
    float eps;
    test_case_group_norm(int dim1 = 64,int dim2 = 64,int dim3 =320, int dim4 = 1,int32_t n_group = 32,float eps = 1e-6f):
        dim1(dim1),dim2(dim2),dim3(dim3),dim4(dim4),num_group(n_group),eps(eps),test_case("GROUP_NORM"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};
struct test_case_concat:test_case{
    std::vector<int> ne_a;
    int ne_b_d;
    int dim;
    test_case_concat(int dim1 = 64,int dim2 = 64,int dim3 =5, int dim4 = 5, int ne_b_d = 6, int dim = 2):
        ne_a({dim1,dim2,dim3,dim4}),dim(dim),ne_b_d(ne_b_d),test_case("CONCAT"){}
    void init_src_size();
    ggml_tensor* build_graph(ggml_context* compute_ctx, std::vector<ggml_tensor*>& src_tensors)const;
};