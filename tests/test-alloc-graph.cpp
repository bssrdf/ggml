#include "ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"
#include "train.h"

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

#include <map>
#include <float.h>



static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    struct ggml_tensor * encode1_weight;
    struct ggml_tensor * encode1_bias;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};


static void load_data(ggml_backend_t backend, struct ggml_tensor * dst, struct ggml_tensor * src){
    if(ggml_backend_is_cpu(backend)) {
        memcpy(dst->data, src->data, ggml_nbytes(dst));
    } else {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(dst));
    }

}

void load_model(test_model & model, float* a, float* b, int M, int N, int K, int L, bool use_gpu = false) {
    // M=784, N=20, K=100, L=400
    size_t buffer_size = 0;
    {
        buffer_size += (M * K) * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += (N * K) * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += (M * L) * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += (L) * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }


    
    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %d bytes\n", __func__, (int) buffer_size);

    int num_tensors = 6;
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

    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, M, K); 
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, N, K); 
    model.encode1_weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, M, L); 
    model.encode1_bias   = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, L);     

    // create a allocator
    ggml_allocr * alloc = ggml_allocr_new_from_buffer(model.buffer);

    // alloc memory
    ggml_allocr_alloc(alloc, model.a);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, a, ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a)); // cuda requires copy the data directly to device
    }

    // alloc memory
    ggml_allocr_alloc(alloc, model.b);

    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.b->data, b, ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));  // cuda requires copy the data directly to device
    }

    ggml_allocr_alloc(alloc, model.encode1_bias);
    ggml_allocr_alloc(alloc, model.encode1_weight);

    struct ggml_init_params lparams = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    

    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.b->data, b, ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));  // cuda requires copy the data directly to device
    }

    float *rn = new float[M*L];    
    for(int i = 0; i < M*L; i++){            
        // rnds[i] = frand_normal(rnd);         
        rn[i] = -(float)1/2.f + i*0.0005;
    }

    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.encode1_weight->data, rn, ggml_nbytes(model.encode1_weight));
    } else {
        ggml_backend_tensor_set(model.encode1_weight, rn, 0, ggml_nbytes(model.encode1_weight)); // cuda requires copy the data directly to device
    }

    float* en_data = new float[K*L];

    for(int k = 0; k < K; k++){
        for(int l = 0; l < L; l++){
            float sum = 0.f;
            for(int i = 0; i < M; i++){
                // float *r = (float *)((char*)rn->data + 
                sum += rn[l*M+i] * a[k*M+i];
            }
            en_data[k*L+l] = sum;
        }
    }
    printf("should have the followign numbers \n");
    for (int i = 0; i < 10; ++i)
        printf("%f, ", en_data[i]);
    printf("\n");   

    delete(en_data);


   
    // free_random_normal_distribution(rnd0);

    // ggml_print_objects(ctx0);
    delete(rn);

    rn = new float[L];    
    for(int i = 0; i < L; i++){            
        // rnds[i] = frand_normal(rnd);         
        rn[i] = -(float)1/2.f + i*0.03;
    }
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.encode1_bias->data, rn, ggml_nbytes(model.encode1_bias));
    } else {
        ggml_backend_tensor_set(model.encode1_bias, rn, 0, ggml_nbytes(model.encode1_bias)); // cuda requires copy the data directly to device
    }

    delete(rn);
   


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

    // struct ggml_tensor* h = ggml_mul_mat(ctx0, model.encode1_weight, ggml_cont(ctx0, model.a)); 
    struct ggml_tensor* h = ggml_mul_mat(ctx0, model.encode1_weight, model.a); 
    ggml_set_name(h, "encode1_w");

    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model.encode1_bias, h)); 
    ggml_set_name(h, "encode1_b");
    
    struct ggml_tensor* result = ggml_relu(ctx0, h); 
    ggml_set_name(result, "result");

    // z = (zT)T
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// struct ggml_tensor* compute(const test_model & model, struct ggml_allocr * allocr) {
struct ggml_cgraph * compute(const test_model & model, struct ggml_allocr * allocr) {
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
    // return gf->nodes[gf->n_nodes - 1];
    return gf;

}

typedef unsigned char uint8;

#define COUNT_TRAIN 60000

int read_data(uint8 *data, const unsigned long count)
{
    FILE *fp_image = fopen("models/mnist/train-images-idx3-ubyte", "rb");
    if (!fp_image) return 1;
    if(fseek(fp_image, 16, SEEK_SET) != 0)
       return 1;
    size_t r = fread(data, 1, 28ll*28ll*count, fp_image);     
    if (ferror(fp_image)){
        fprintf(stderr, "%s: error afetr read \n", __func__);       
        return 1;
    }
    fclose(fp_image);
    return 0;
}


int main(void)
{
    ggml_time_init();    

    bool passed = true;

    test_model model;
    int n_input = 784;  
    int n_latent = 20;  
    int n_batch = 100;
    int n_feature = 400;  

    uint8 *train_data = (uint8 *)malloc(COUNT_TRAIN * 28 * 28 * sizeof(uint8));

    if(read_data(train_data, COUNT_TRAIN) > 0){
        fprintf(stderr, "Error in read in Mnist training data! \n");
        exit(1);
    }

    uint8_t *buf = train_data;
    std::vector<float> digit;

    digit.resize(n_batch*28*28);

    for (int ib = 0; ib < n_batch; ib++) 
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                digit[ib*28*28 + row*28 + col] = ((float)buf[ib*28*28 + row*28 + col])/255.f;
            }
        }
    // struct random_normal_distribution * rnd = init_random_normal_distribution(1337, 0, 1.f, -FLT_MAX, FLT_MAX);

    float *rnds = new float[n_latent*n_batch];    
    for(int i = 0; i < n_latent*n_batch; i++){            
        // rnds[i] = frand_normal(rnd);         
        rnds[i] = -(float)1/2.f + i*0.01;
    }

    load_model(model, digit.data(), rnds, n_input, n_latent, n_batch, n_feature, true);

    ggml_backend_buffer_t buf_compute; // for compute
    struct ggml_allocr * allocr = NULL;

    {
        allocr = ggml_allocr_new_measure_from_backend(model.backend);

        //create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model, allocr);
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);
        // ggml_graph_dump_dot(gf, NULL, "test-alloc-graph-forward.dot");

        std::map<void *, struct ggml_tensor*> gf_map;  
        std::map<void *, struct ggml_tensor*>::iterator it;
        for(int i = 0; i < gf->n_nodes; ++i){
            struct ggml_tensor *node = gf->nodes[i];
            it = gf_map.find((void *)node->data);
            if (it != gf_map.end()){ 
                printf( "%s 's data addr already allocated for %s \n", node->name, gf_map[(void *)node->data]->name);
            }
            gf_map[(void *)node->data] = node;
        }

        ggml_allocr_free(allocr);

        // compute the required memory
        buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
        allocr = ggml_allocr_new_from_buffer(buf_compute);
        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }

    // struct ggml_tensor * result = compute(model, allocr);
    struct ggml_cgraph * gf = compute(model, allocr);

    struct ggml_tensor * result = NULL;
    struct ggml_tensor * encode1_w = NULL;
    for(int i = 0; i < gf->n_nodes; i++) {
        if(strcmp(ggml_get_name(gf->nodes[i]), "result") == 0) {
            result = gf->nodes[i];
        } else if(strcmp(ggml_get_name(gf->nodes[i]), "encode1_w") == 0) {
            encode1_w = gf->nodes[i];
        }
    }

    float* out_data = new float[ggml_nelements(result)];
    float* en_data = new float[ggml_nelements(encode1_w)];

    ggml_backend_tensor_get(result, out_data, 0, ggml_nbytes(result));
    ggml_backend_tensor_get(encode1_w, en_data, 0, ggml_nbytes(encode1_w));

    for (int i = 0; i < 10; ++i)
        printf("%f, ", out_data[i]);
    printf("\n");   
    printf("instead have  \n");
    for (int i = 0; i < 10; ++i)
        printf("%f, ", en_data[i]);
    printf("\n");   

    
   // free memory
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_buffer_free(buf_compute);
    ggml_backend_free(model.backend);
    return 0;
}
