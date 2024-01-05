#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"

// #define GGML_USE_CUBLAS

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
#include <array>


using dim_t = std::array<int, 2>;


int nx = 8;
int ny = 8;

static void ggml_log_callba1ck_default(ggml_log_level level, const char * text, void * user_data) {
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

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    int KW = 3, KH = 3, IC = 10, OC = 10;
    int IW = 8, IH = 6, N = 1;

    using scalar_type = float;
    using input_type = scalar_type;    

   
    dim_t fft_size = {nx, ny};
    unsigned int batch_size = 256;



    // Initialize adata
    float * adata = new float[nx*ny*batch_size];
    for (int i = 0; i < nx*ny*batch_size; i++) {
        adata[i] = (scalar_type)i+1.0f;
    }

    for(int i = 0; i < nx*ny*batch_size; i++){
        printf(" %f, ", adata[i]);
        if((i+1) % nx == 0)
           printf("\n");
        if((i+1) % (nx*ny) ==0)
           printf("==============================\n");

    }

    std::vector<ggml_fp16_t> hadata(nx*ny*batch_size);
    ggml_fp32_to_fp16_row(adata, hadata.data(), nx*ny*batch_size);

    // Initialize bdata
    float * bdata =  new float[1];
    for (int i = 0; i < 1; i++) {
        bdata[i] = 0.6;
    }

    size_t buffer_size = 0;
    {
        buffer_size += nx * ny * batch_size * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %d bytes\n", __func__, buffer_size);
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

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
    model.a = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32,  nx, ny, batch_size);
    // model.a = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F16,  nx, ny, batch_size);
    model.b = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 1);

    // model.b->op_params[0] = 1;

    // create a allocator
    ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer);

    // alloc memory
    ggml_allocr_alloc(alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, adata, ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, adata, 0, ggml_nbytes(model.a));
        // ggml_backend_tensor_set(model.a, hadata.data(), 0, ggml_nbytes(model.a));
    }

    // alloc memory
    ggml_allocr_alloc(alloc, model.b);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.b->data, bdata, ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, bdata, 0, ggml_nbytes(model.b));
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

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    // split conv2d in fundamental methods for test unit
    // struct ggml_tensor* fft2d = ggml_fft_filter(ctx0, model.a, model.b);
    struct ggml_tensor* fft2d = ggml_freeu_backbone(ctx0, model.a, nullptr);
    // struct ggml_tensor* fft2d = ggml_sum_rows(ctx0, model.a);
    fft2d->op_params[0] = 2;
    float scale = 1.4;
    memcpy(fft2d->op_params+1,  &scale, sizeof(float));
    ggml_set_name(fft2d, "fft2d_res");
    ggml_build_forward_expand(gf, fft2d);
    

    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * compute_graph(const test_model & model, struct ggml_allocr * allocr) {
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

    return gf;
}

int main(void)
{
    ggml_time_init();

    test_model model;
    load_model(model, true);

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
        fprintf(stderr, "%s: compute buffer size: %d bytes\n", __func__, mem_size);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0f/1024.0f);
    }

    struct ggml_cgraph * gf_res = compute_graph(model, allocr);

    struct ggml_tensor * fft2d_res = NULL;    

    for(int i = 0; i < gf_res->n_nodes; i++) {
        if(strcmp(ggml_get_name(gf_res->nodes[i]), "fft2d_res") == 0) {
            fft2d_res = gf_res->nodes[i];
        } 
    }
    
    float* fft2d_data = new float[ggml_nelements(fft2d_res)];
    
    ggml_backend_tensor_get(fft2d_res, fft2d_data, 0, ggml_nbytes(fft2d_res));

    for(int i = 0; i < ggml_nelements(fft2d_res); i++){
        printf(" %f, ", fft2d_data[i]);
        if((i+1) % nx == 0)
           printf("\n");
    }

    
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_buffer_free(buf_compute);
    ggml_backend_free(model.backend);
    return 0;
}
