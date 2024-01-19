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

void load_model(test_model & model, float* a, float* b, int M, int N, int K, bool use_gpu = false) {
    size_t buffer_size = 0;
    {
        buffer_size += (M * N) * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += (N * K) * ggml_type_size(GGML_TYPE_F32); // tensor b
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
        ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
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
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, M, K);
    printf("Matrix A: [%i, %i]\n", K, M);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, N, K);
    printf("Matrix B: [%i, %i]\n", K, N);

    // create a allocator
    ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer);

    // alloc memory
    ggml_allocr_alloc(alloc, model.a);

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

    // alloc memory
    ggml_allocr_alloc(alloc, model.b);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b->data, b, ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));  // cuda requires copy the data directly to device
    }

    ggml_allocr_free(alloc);
}

struct ggml_cgraph * build_graph(const test_model& model, struct ggml_allocr * allocr) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // zT = x @ yT
    struct ggml_tensor * result = ggml_out_prod(ctx0, model.a, model.b);

    struct ggml_tensor * h = model.a;
    int64_t *ne = h->ne;
    printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    h = model.b;
    ne = h->ne;
    printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    h = result;
    ne = h->ne;
    printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
 


    // z = (zT)T
    ggml_build_forward_expand(gf,  result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor* compute(const test_model & model, struct ggml_allocr * allocr) {
    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = build_graph(model, allocr);

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);
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

void perform_gemm_test(float* a, float* b, float* expected, int M, int N, int K) {
    for(int jj = 0; jj < N; jj++){       
        for(int ii = 0; ii < M; ii++) {      
            expected[jj*M + ii] = 0.f;
            for (int k = 0; k < K; k++) {
                expected[jj*M + ii] += a[k*M+ii] * b[k*N+jj];
            }
        }
    }

    
}

int main(void)
{
    ggml_time_init();
    const int M = 3, N = 4, K = 2;  // a conv2d expected matrix multiplication

    // matrix A (4 X 36)
    float matrixA[M * K] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };

    // matrix B (16 X 36)
    float matrixB[N * K] = {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };

    
    float expected_results[M * N] = {0.f};

    bool passed = true;

    perform_gemm_test(matrixA, matrixB, expected_results, M, N, K);

    test_model model;
    load_model(model, matrixA, matrixB, M, N, K, true);

    ggml_backend_buffer_t buf_compute; // for compute
    struct ggml_allocr * allocr = NULL;

    {
        allocr = ggml_allocr_new_measure_from_backend(model.backend);

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model, allocr);
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);
        ggml_allocr_free(allocr);

        // compute the required memory
        buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
        allocr = ggml_allocr_new_from_buffer(buf_compute);
        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }

    struct ggml_tensor * result = compute(model, allocr);

    float* out_data = new float[ggml_nelements(result)];

    ggml_backend_tensor_get(result, out_data, 0, ggml_nbytes(result));

    printf("\nPerforming ggml_mul_mat test:\n");

    passed = true;
    for(int i = 0; i < M * N; i++) {
        if(out_data[i] != expected_results[i]) {
            passed = false;
            break;
        }
    }

    // for (int j = 0; j < N; j++) {
    //    for (int i = 0; i < M; i++) {        
    //         printf("%.1f ", out_data[j * M + i]);
    //     }
    //     printf("\n");
    // }
    for (int i = 0; i < M*N; i++) {       
        printf("%.1f ", out_data[i]);
        if((i+1) % M == 0)       
            printf("\n");
    }
    printf("==================\n");
    for (int i = 0; i < M*N; i++) {       
        printf("%.1f ", expected_results[i]);
        if((i+1) % M == 0)       
            printf("\n");
    }

    printf("ggml_out_prod (%d): %s\n", (int) ggml_nelements(result), passed && (ggml_nelements(result) == M * N) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

   // free memory
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_buffer_free(buf_compute);
    ggml_backend_free(model.backend);
    return 0;
}
