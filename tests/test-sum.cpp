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
    struct ggml_tensor * w;
    struct ggml_tensor * b;
    struct ggml_tensor * input;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, float* w, float* b, int M, int N, int K, bool use_gpu = false) {
    size_t buffer_size = 0;
    {
        buffer_size += (M * N) * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += (N ) * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += (M*K) * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %d bytes\n", __func__, (int) buffer_size);

    int num_tensors = 3+2;
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
    // M = 3, N = 4 
    model.b = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, N);
    model.w = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, N, M);
    model.input = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, M, K);
    ggml_set_name(model.input, "input");

    // create a allocator
    ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer);

    // alloc memory
    ggml_allocr_alloc(alloc, model.w);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.w->data, w, ggml_nbytes(model.w));
    } else {
        ggml_backend_tensor_set(model.w, w, 0, ggml_nbytes(model.w)); // cuda requires copy the data directly to device
    }

    // alloc memory
    ggml_allocr_alloc(alloc, model.b);
    
    ggml_allocr_alloc(alloc, model.input);


    ggml_allocr_free(alloc);
}

static struct ggml_tensor * get_tensor_from_graph(struct ggml_cgraph * gf, const char *name){

    struct ggml_tensor * res = NULL;
    for(int i = 0; i < gf->n_nodes; i++) {
        printf("%d, %s \n", i, gf->nodes[i]->name);
        if(strcmp(ggml_get_name(gf->nodes[i]), name) == 0) {
            res = gf->nodes[i];
            break;
        } 
    }
    for(int i = 0; i < gf->n_leafs; i++) {
        printf("%d, %s \n", i, gf->leafs[i]->name);
        if(strcmp(ggml_get_name(gf->leafs[i]), name) == 0) {
            res = gf->leafs[i];
            break;
        } 
    }
    return res;
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
    struct ggml_tensor * result = ggml_soft_max(ctx0, model.w);
    
    // struct ggml_tensor * result = ggml_add(ctx0, wx, model.b);

    struct ggml_tensor * h = model.w;
    int64_t *ne = h->ne;
    printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    h = model.b;
    ne = h->ne;
    printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    h = model.input;
    ne = h->ne;
    printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    h = result;
    ne = h->ne;
    printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);

    // z = (zT)T
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

struct ggml_tensor* compute(const test_model & model,  struct ggml_allocr * allocr) {
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


int main(void)
{
    ggml_time_init();
    const int M = 2, N = 32, K = 2;  // a conv2d expected matrix multiplication

    // matrix A (4 X 36)
    float matrixA[M*N];
    int j = 0;
    for (int i = 0; i < N; i++)
        matrixA[j*N+i] = (float)(j*N+i)*0.01;
    j = 1;
    for (int i = N; i >=0; i--)
        matrixA[j*N+N-i] = (float)(j*N+i)*0.01;



    // matrix B (16 X 36)
    float matrixB[M*N] = {1., 2.f, 3.f, 4.f};
    

  


   float expected_result[K * N] = {15.f, 34.f, 53.f, 72.0, 
                                   21.f, 55.f, 89.f, 123.f
                                  };
   
   
    bool passed = true;
   

    test_model model;
    load_model(model, matrixA, matrixB, M, N, K, true);

    fprintf(stderr, "%s: after load model: \n", __func__);
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

    printf("\nPerforming ggml_mlp_layer test:\n");

   
    for (int i = 0; i < M*N; i++) {       
        printf("%f ", out_data[i]);
        if((i+1) % N == 0)       
            printf("\n");
    }
    printf("==================\n");

    for (int i = 0; i < M*N; i++) {       
        printf("%f ", matrixA[i]);
        if((i+1) % N == 0)       
            printf("\n");
    }
    printf("==================\n");

  

   // free memory
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_buffer_free(buf_compute);
    ggml_backend_free(model.backend);
    return 0;
}
