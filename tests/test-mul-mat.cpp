#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
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
    struct ggml_tensor * b2;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    // int KW = 3, KH = 3, IC = 10, OC = 10;
    // int IW = 8, IH = 6, N = 1;
    int M = 1028, N = 1, K = 2816;

    // Initialize adata
    std::vector<float> adata(M*K);
    for (int i = 0; i < M*K; i++) {
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        adata[i] = r;
    }

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> hadata(M*K);
    ggml_fp32_to_fp16_row(adata.data(), hadata.data(), M*K);

    // Initialize bdata
    std::vector<float> bdata(N*K);
    for (int i = 0; i < N*K; i++) {
        // bdata[i] = 1.5f;
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        bdata[i] = r;
    }

    // printf("bdata: [");
    // for(int j=0; j < K; ++j){
    //    printf("%.3f,", bdata[j]);
    // }
    // printf("]\n");

    std::vector<ggml_fp16_t> hbdata(N*K);
    ggml_fp32_to_fp16_row(bdata.data(), hbdata.data(), N*K);


    size_t buffer_size = 0;
    {
        buffer_size += M * K * ggml_type_size(GGML_TYPE_F16); // tensor a
        buffer_size += N * K * ggml_type_size(GGML_TYPE_F16); // tensor b
        buffer_size += N * K * ggml_type_size(GGML_TYPE_F32); // tensor b2
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 3;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    ggml_log_set(ggml_log_callback_default, nullptr);

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = ggml_backend_alloc_buffer(model.backend, buffer_size);

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, K, M, 1, 1);
    model.b = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, K, N, 1, 1);
    model.b2 = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, K, N, 1, 1);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, hadata.data(), ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, hadata.data(), 0, ggml_nbytes(model.a));
    }

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.b);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b->data, hbdata.data(), ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, hbdata.data(), 0, ggml_nbytes(model.b));
    }


     ggml_tallocr_alloc(&alloc, model.b2);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b2->data, bdata.data(), ggml_nbytes(model.b2));
    } else {
        ggml_backend_tensor_set(model.b2, bdata.data(), 0, ggml_nbytes(model.b2));
    }
}

struct ggml_cgraph * build_graph(const test_model& model) {
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


    // split conv2d in fundamental methods for test unit
    struct ggml_tensor* res = ggml_mul_mat(ctx0, model.a, model.b);
    ggml_set_name(res, "gemm_res");
    ggml_build_forward_expand(gf, res);

    struct ggml_tensor* res2 = ggml_mul_mat(ctx0, model.a, model.b2);
    ggml_set_name(res2, "gemm_res2");
    ggml_build_forward_expand(gf, res2);

    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * compute_graph(const test_model & model, ggml_gallocr_t allocr) {
    struct ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    //ggml_graph_print(gf);

    return gf;
}

int main(void)
{
    ggml_time_init();

    test_model model;
    load_model(model, true);

    ggml_gallocr_t allocr = NULL;

    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model);

        // compute the required memory
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
    }

    struct ggml_cgraph * gf_res = compute_graph(model, allocr);

    struct ggml_tensor * gemm_res = NULL;
    struct ggml_tensor * gemm_res2 = NULL;

    for(int i = 0; i < ggml_graph_n_nodes(gf_res); ++i) {
        if(strcmp(ggml_get_name(ggml_graph_node(gf_res, i)), "gemm_res") == 0) {
            gemm_res = ggml_graph_node(gf_res, i);
        }else if(strcmp(ggml_get_name(ggml_graph_node(gf_res, i)), "gemm_res2") == 0) {
            gemm_res2 = ggml_graph_node(gf_res, i);
        }
    }

    std::vector<ggml_fp16_t> gemm_data(ggml_nelements(gemm_res));
    std::vector<float> gemm_data2(ggml_nelements(gemm_res2));

    ggml_backend_tensor_get(gemm_res, gemm_data.data(), 0, ggml_nbytes(gemm_res));
    ggml_backend_tensor_get(gemm_res2, gemm_data2.data(), 0, ggml_nbytes(gemm_res2));

    for(int i = 0; i < gemm_data.size(); i++) {
        printf("%d: %f, %f\n", i, gemm_data2[i], ggml_fp16_to_fp32(gemm_data[i]));
    }

    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
