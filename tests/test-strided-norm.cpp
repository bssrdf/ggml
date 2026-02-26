#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
//#include <cuda_runtime.h>
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model &, int,  int, int, int, int, int, bool);
struct ggml_cgraph * build_graph_0(const test_model&);
struct ggml_cgraph * build_graph_1(const test_model&);

void load_model(test_model & model, bool use_gpu = false ) {
    // create data
    
    int M = 3072, N = 2048;
    // srand(time(NULL));

    // printf(" input: IC = %d, OC = %d, IW = %d, IH = %d \n ", IC, OC, IW, IH);

    // Initialize adata
    std::vector<float> adata(M*3*N);
    for (int i = 0; i < M*3*N; i++) {
        // adata[i] = 2.f;
        // adata[i] = (float)(i%KW)-1.f;
        // adata[i] = (float)((i+1)%KW+1)/10.0;
        // adata[i] = (float)(i%100);
        // adata[i] = (rand() % 255) / 255.0;
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        adata[i] = r;
    }

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> hadata(M*3*N);
    ggml_fp32_to_fp16_row(adata.data(), hadata.data(), M*3*N);

    size_t buffer_size = 0;
    {
        // buffer_size += KW * KH * IC * OC * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += M * 3 * N * ggml_type_size(GGML_TYPE_F16); // tensor a
        // buffer_size += IW * IH * IC * N  * ggml_type_size(GGML_TYPE_F32); // tensor b
        // buffer_size += IW * IH * IC * N  * ggml_type_size(GGML_TYPE_F16); // tensor b
        buffer_size += 1024; // overhead
    }

    // printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    // printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 1;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (use_gpu) {
        // fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#else
    GGML_UNUSED(use_gpu);
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#else
    GGML_UNUSED(use_gpu);
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  M*3, N, 1, 1);
    // model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32,  KW, KH, IC, OC);
    // model.b = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, IW, IH, IC, N);
    // model.b = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, IW, IH, IC, N);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, hadata.data(), ggml_nbytes(model.a));
        // memcpy(model.a->data, adata.data(), ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, hadata.data(), 0, ggml_nbytes(model.a));
        // ggml_backend_tensor_set(model.a, adata.data(), 0, ggml_nbytes(model.a));
    }

}

typedef struct ggml_cgraph* (*build_graph_t)(const test_model& model);


struct ggml_cgraph * build_graph_1(const test_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);


    struct ggml_tensor* a =  model.a;
    struct ggml_tensor* b = ggml_view_4d(ctx0, a, a->ne[0]/3, a->ne[1], 1, 1, a->nb[1], a->nb[2], a->nb[3], (a->nb[0])*a->ne[0]/3);
    struct ggml_tensor* b0 = ggml_cont(ctx0, b);
    struct ggml_tensor* b1 = ggml_reshape_4d(ctx0, b0, 128, a->ne[0]/3/128, a->ne[1], 1);
    struct ggml_tensor* c = ggml_rms_norm(ctx0, b1, 1.e-6f);
    ggml_set_name(c, "cont_res");
    ggml_build_forward_expand(gf, c);

    struct ggml_tensor* d0 = ggml_view_4d(ctx0, a, 128, a->ne[0]/3/128, a->ne[1], 1, a->nb[0]*128, a->nb[1], a->nb[2], (a->nb[0])*a->ne[0]/3);

    struct ggml_tensor* d = ggml_rms_norm(ctx0, d0, 1.e-6f);
    ggml_set_name(d, "view_res");
    ggml_build_forward_expand(gf, d);
    // ne = wino_res->ne;
    // printf("wino: (%zu, %zu, %zu, %zu) \n", ne[0], ne[1], ne[2], ne[3]);
    ggml_free(ctx0);
    return gf;
}

std::vector<std::vector<ggml_fp16_t>> compute_graph(const test_model &, ggml_gallocr_t,
            build_graph_t, int, double *);


std::vector<std::vector<ggml_fp16_t>> compute_graph(const test_model & model, ggml_gallocr_t allocr,
            build_graph_t build_graph, int iters, double *t) {
    struct ggml_cgraph * gf = build_graph(model);


    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    ggml_backend_synchronize(model.backend);

    int64_t start_time = ggml_time_us();

    for(int iter=0; iter<iters; iter++){
        ggml_backend_graph_compute(model.backend, gf);
        ggml_backend_synchronize(model.backend);
    }

    int64_t end_time = ggml_time_us();
    double time_us = end_time - start_time;

    time_us = time_us/iters;

    //ggml_graph_print(gf);

    struct ggml_tensor *res0 = NULL;
    struct ggml_tensor *res1 = NULL;

    for(int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if(strcmp(ggml_get_name(ggml_graph_node(gf, i)), "cont_res") == 0) {
            res0 = ggml_graph_node(gf, i);
        } else if(strcmp(ggml_get_name(ggml_graph_node(gf, i)), "view_res") == 0) {
            res1 = ggml_graph_node(gf, i);
        }
    }

    std::vector<std::vector<ggml_fp16_t>> results;

    std::vector<ggml_fp16_t> data0(ggml_nelements(res0));
    ggml_backend_tensor_get(res0, data0.data(), 0, ggml_nbytes(res0));
    std::vector<ggml_fp16_t> data1(ggml_nelements(res1));
    ggml_backend_tensor_get(res1, data1.data(), 0, ggml_nbytes(res1));

    results.push_back(data0);
    results.push_back(data1);

    *t = time_us/1000;
    return results;

}


int main(void)
{
    ggml_time_init();

    double time_iter0 = 0.0, time_iter1 = 0.0;

    int k = 0;

    // for (auto c : configs_sdxl_1024){
    test_model model;
    load_model(model, true);

    ggml_gallocr_t allocr = NULL;
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    //create the worst case graph for memory usage estimation
    struct ggml_cgraph * gf = build_graph_1(model);

    // compute the required memory
    ggml_gallocr_reserve(allocr, gf);
    size_t mem_size0 = ggml_gallocr_get_buffer_size(allocr, 0);
    // fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);


    int iterations = 0;

    double run_time0;
   std::vector<std::vector<ggml_fp16_t>> data = compute_graph(model, allocr, build_graph_1, iterations, &run_time0);
    std::vector<float>  f32_data0(data[0].size());
    std::vector<float>  f32_data1(data[1].size());
    ggml_fp16_to_fp32_row(data[0].data(), f32_data0.data(), data[0].size());
    ggml_fp16_to_fp32_row(data[1].data(), f32_data1.data(), data[1].size());
    // int i = 2048;
    // for(int i = 0; i < ggml_nelements(wino_res); i++) {
    // for(int i = 0; i < 26*38; i++) {
    for(int i = 0; i < f32_data0.size(); i++) {
        float diff = fabs(f32_data0[i] - f32_data1[i]);
        if(diff > 0.5) {
                printf("(%f, %f, %d) \n",
                f32_data0[i], f32_data1[i], i);
            // break;
        }
    }

    ggml_free(model.ctx);
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);


    // printf("\nPerforming test:\n");
    return 0;
}
