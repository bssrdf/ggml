#include "ggml.h"
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
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    int KW = 3, KH = 3, IC = 15;

    // Initialize adata
    float * adata = new float[KW * KH * IC];
    for (int i = 0; i < KW * KH * (IC / 3) ; i++) {        
        adata[i] = 2.5f;
    }
    for (int i = KW * KH * (IC / 3); i < KW * KH * IC * 2 / 3; i++) {        
        adata[i] = 3.5f;
    }

    for (int i = KW * KH * (IC * 2 / 3); i < KW * KH * IC; i++) {        
        adata[i] = 4.5f;
    }

    

    size_t buffer_size = 0;
    {
        buffer_size += KW * KH * IC * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

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
        ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
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
    model.a = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32,  KW, KH, IC);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, adata, ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, adata, 0, ggml_nbytes(model.a));
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
    struct ggml_tensor *a = model.a;
    int64_t offset = a->nb[2] * a->ne[2] / 3;    
    struct ggml_tensor* half1 = ggml_view_3d(ctx0, a, a->ne[0], a->ne[1], a->ne[2]/3, a->nb[1], a->nb[2], offset * 0);
    ggml_set_name(half1, "half1_res");
    ggml_build_forward_expand(gf, half1);

    // recalculate for avoid fragmentation
    struct ggml_tensor* half2 = ggml_view_3d(ctx0, a, a->ne[0], a->ne[1], a->ne[2]/3, a->nb[1], a->nb[2], offset * 1);
    ggml_set_name(half2, "half2_res");
    ggml_build_forward_expand(gf, half2);

    struct ggml_tensor* half3 = ggml_view_3d(ctx0, a, a->ne[0], a->ne[1], a->ne[2]/3, a->nb[1], a->nb[2], offset * 2);
    ggml_set_name(half3, "half3_res");
    ggml_build_forward_expand(gf, half3);

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

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads);
    }
#endif

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

    struct ggml_tensor * h1_res = NULL;
    struct ggml_tensor * h2_res = NULL;
    struct ggml_tensor * h3_res = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
        if(strcmp(ggml_get_name(gf_res->nodes[i]), "half1_res") == 0) {
            h1_res = gf_res->nodes[i];
        } else if(strcmp(ggml_get_name(gf_res->nodes[i]), "half2_res") == 0) {
            h2_res = gf_res->nodes[i];
        } else if(strcmp(ggml_get_name(gf_res->nodes[i]), "half3_res") == 0) {
            h3_res = gf_res->nodes[i];
        }
    }

    float* h1_data = new float[ggml_nelements(h1_res)];
    float* h2_data = new float[ggml_nelements(h2_res)];
    float* h3_data = new float[ggml_nelements(h3_res)];

    ggml_backend_tensor_get(h1_res, h1_data, 0, ggml_nbytes(h1_res));
    ggml_backend_tensor_get(h2_res, h2_data, 0, ggml_nbytes(h2_res));
    ggml_backend_tensor_get(h3_res, h3_data, 0, ggml_nbytes(h3_res));

    for(int k = 0; k < 5; k++){
        for(int j = 0; j < 3; j++){
            for(int i = 0; i < 3; i++){
               printf("%f, ", h1_data[i+3*j+k*3*3]);
            }
            printf("\n");
        }
        printf("==================\n");
    }

    printf("*************************\n");

    for(int k = 0; k < 5; k++){
        for(int j = 0; j < 3; j++){
            for(int i = 0; i < 3; i++){
               printf("%f, ", h2_data[i+3*j+k*3*3]);
            }
            printf("\n");
        }
        printf("==================\n");
    }

    printf("*************************\n");

    for(int k = 0; k < 5; k++){
        for(int j = 0; j < 3; j++){
            for(int i = 0; i < 3; i++){
               printf("%f, ", h3_data[i+3*j+k*3*3]);
            }
            printf("\n");
        }
        printf("==================\n");
    }       
    
    

    
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
