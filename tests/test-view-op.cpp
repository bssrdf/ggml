#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

//#define GGML_USE_CUBLAS // uncomment this to use cuda backend, make sure build ggml lib with GGML_CUBLAS=ON

#ifdef GGML_USE_CUBLAS
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

struct test_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, float* a, int M, int N, int K, bool use_gpu = false) {
    size_t buffer_size = 0;
    {
        buffer_size += (M * N * K) * ggml_type_size(GGML_TYPE_F32); // tensor a
        
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %d bytes\n", __func__, (int) buffer_size);

    int num_tensors = 2;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUBLAS
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
    model.a = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, M, N, K);
    printf("Matrix A: [%i, %i, %i]\n", M, N, K);
    

    // create a allocator
    ggml_tallocr_t alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.a->data, a, ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a)); // cuda requires copy the data directly to device
    }    

    ggml_tallocr_free(alloc);
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

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // zT = x @ yT
    struct ggml_tensor * a = model.a;
    struct ggml_tensor * b = ggml_view_2d(ctx0, a, a->ne[0], a->ne[2], a->nb[2], a->nb[1]);
    // struct ggml_tensor * b = ggml_view_2d(ctx0, a, a->ne[0], a->ne[1], a->nb[0], 0);
    // struct ggml_tensor * b = ggml_view_2d(ctx0, a, 1, 2, a->nb[1], a->nb[0]*1);
    struct ggml_tensor * result = ggml_scale(ctx0, ggml_cont(ctx0, b), 1.f);
    int64_t *ne = result->ne;
    printf("res: [%zu, %zu, %zu, %zu] \n", ne[0], ne[1], ne[2], ne[3]);
    
    // z = (zT)T
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor* compute(const test_model & model, ggml_gallocr_t allocr) {
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

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}



int main(void)
{
    ggml_time_init();
    const int M = 4, N = 2, K = 3;  // a conv2d expected matrix multiplication

    // matrix A (4 X 36)
    float matrixA[M * N * K] = {0.f};

    // for (int k = 0; k < K; k++) {   
    //     int t = 0;     
    //     for (int i = 0; i < N; i++) {
    //         for (int j = 0; j < M; j++) {
    //             t++;
    //             matrixA[k*M*N+i*M+j] = (float)(k+1)*t;
    //         }
    //     }
    // }

    for (int i = 0; i < N; i++) {
        int t = 0;     
        for (int k = 0; k < K; k++) {   
            for (int j = 0; j < M; j++) {
                t++;
                matrixA[k*M*N+i*M+j] = (float)(i+1)*t;
            }
        }
    }

    for (int k = 0; k < K; k++) {   
        printf("[");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                printf("%f,", matrixA[k*M*N+i*M+j]);
            }
            printf("\n");
        }
        printf("]\n");
    }



       

    test_model model;
    load_model(model, matrixA, M, N, K, true);

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

    printf("\nPerforming ggml view test:\n");
    struct ggml_tensor * result = compute(model, allocr);
    printf("\nPerforming ggml view test done\n");

    float* out_data = new float[ggml_nelements(result)];

    ggml_backend_tensor_get(result, out_data, 0, ggml_nbytes(result));


    
    for (int i = 0; i < K; i++) {
        printf("[");
        for (int j = 0; j < M; j++) {
            printf("%f, ", out_data[i * M + j]);
        }
        printf("\n");
    }

    

   // free memory
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
