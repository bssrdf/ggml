#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"
#include "train.h"

#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>


#include <stdlib.h>
#include <float.h>


#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif


#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

typedef unsigned char uint8;
typedef uint8 image[28][28];

constexpr float rms_norm_eps = 5e-6f;

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }
    ggml_graph_compute(graph, &plan);
}

static struct ggml_tensor * randomize_tensor(
    struct ggml_tensor * tensor, int ndims, const int64_t ne[], float fmin, float fmax
) {
    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)tensor->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)tensor->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)tensor->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)tensor->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    }

    return tensor;
}


static struct ggml_tensor * get_tensor_from_graph(struct ggml_cgraph * gf, const char *name){

    struct ggml_tensor * res = NULL;
    for(int i = 0; i < gf->n_nodes; i++) {
        if(strcmp(ggml_get_name(gf->nodes[i]), name) == 0) {
            res = gf->nodes[i];
            break;
        } 
    }
    return res;
}

// default hparams
struct mnist_hparams {
    int32_t n_input   = 784;
    

    // int32_t n_latent  = 30;    
    // int32_t enc1_out  = 1000;
    // int32_t enc2_out  = 500;
    // int32_t enc3_out  = 250;
    // int32_t dec4_out  = 250;
    // int32_t dec3_out  = 500;
    // int32_t dec2_out  = 1000;

    int32_t n_latent  = 10;
    int32_t enc1_out  = 100;
    int32_t enc2_out  = 50;
    int32_t enc3_out  = 25;
    int32_t dec4_out  = 25;
    int32_t dec3_out  = 50;
    int32_t dec2_out  = 100;
     

};

struct mnist_vae_model {
    mnist_hparams hparams;

    struct ggml_tensor * input;
    struct ggml_tensor * noise;

    struct ggml_tensor * encode1_weight;
    struct ggml_tensor * encode1_bias;
    struct ggml_tensor * encode2_weight;
    struct ggml_tensor * encode2_bias;
    struct ggml_tensor * encode3_weight;
    struct ggml_tensor * encode3_bias;    


    struct ggml_tensor * logsd_weight;
    struct ggml_tensor * logsd_bias;
    struct ggml_tensor * mu_weight;
    struct ggml_tensor * mu_bias;    
   
    struct ggml_tensor * decode1_weight;
    struct ggml_tensor * decode1_bias;
    struct ggml_tensor * decode2_weight;
    struct ggml_tensor * decode2_bias;
    struct ggml_tensor * decode3_weight;
    struct ggml_tensor * decode3_bias;
    struct ggml_tensor * decode4_weight;
    struct ggml_tensor * decode4_bias;

    ggml_backend_t backend = NULL;
    
    ggml_backend_buffer_t compute_buffer = NULL;
    size_t compute_buffer_size = 0;

    ggml_backend_buffer_t params_buffer = NULL;
    size_t params_buffer_size = 0;
    struct ggml_allocr* compute_allocr   = NULL;

    struct ggml_context * ctx;

    size_t calculate_mem_size() {
        double mem_size = 0;
        mem_size += (hparams.n_input  * hparams.enc1_out + hparams.enc1_out) * ggml_type_sizef(GGML_TYPE_F32);  // encode1_w+b
        mem_size += (hparams.enc1_out * hparams.enc2_out + hparams.enc2_out) * ggml_type_sizef(GGML_TYPE_F32);  // encode2_w+b
        mem_size += (hparams.enc2_out * hparams.enc3_out + hparams.enc3_out) * ggml_type_sizef(GGML_TYPE_F32);  // encode3_w+b
        mem_size += (hparams.enc3_out * hparams.n_latent + hparams.n_latent) * ggml_type_sizef(GGML_TYPE_F32);  // logsd
        mem_size += (hparams.enc3_out * hparams.n_latent + hparams.n_latent) * ggml_type_sizef(GGML_TYPE_F32);  // mu
        mem_size += (hparams.n_latent *  hparams.dec4_out + hparams.dec4_out)* ggml_type_sizef(GGML_TYPE_F32);  // decode4_w+b
        mem_size += (hparams.dec4_out *  hparams.dec3_out + hparams.dec3_out)* ggml_type_sizef(GGML_TYPE_F32);  // decode3_w+b
        mem_size += (hparams.dec3_out *  hparams.dec2_out + hparams.dec2_out)* ggml_type_sizef(GGML_TYPE_F32);  // decode2_w+b
        mem_size += (hparams.dec2_out *  hparams.n_input  + hparams.n_input) * ggml_type_sizef(GGML_TYPE_F32);  // decode1_w+b
        return static_cast<size_t>(mem_size);
    }

    size_t get_num_tensors() {
        return 18;
    }
    
};


static void init_model(struct mnist_vae_model * model, bool use_gpu = false, int32_t n_batch=100) {
    const auto & hparams = model->hparams;

    const int32_t n_input   = hparams.n_input;
    const int32_t n_latent  = hparams.n_latent;    
    const int32_t enc1_out  = hparams.enc1_out;
    const int32_t enc2_out  = hparams.enc2_out;
    const int32_t enc3_out  = hparams.enc3_out;
    const int32_t dec2_out  = hparams.dec2_out;
    const int32_t dec3_out  = hparams.dec3_out;
    const int32_t dec4_out  = hparams.dec4_out;
 
    
    size_t buffer_size = 0;
    {
        buffer_size += (n_input  * n_batch ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (n_latent * n_batch ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (n_input  * enc1_out + enc1_out ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (enc1_out * enc2_out + enc2_out ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (enc2_out * enc3_out + enc3_out ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += 2 * (enc3_out * n_latent + n_latent ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (n_latent * dec4_out + dec4_out ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (dec4_out * dec3_out + dec3_out ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (dec3_out * dec2_out + dec2_out ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (dec2_out * n_input  + n_input ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (dec2_out * n_input  + n_input ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += 1024; // overhead
    }
    model->compute_buffer_size = buffer_size;

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %d bytes\n", __func__, (int) buffer_size);

    int num_tensors = 20 * 2; // *2 to acount for their grads
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * (num_tensors + 2),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // initialize the backend
#ifdef GGML_USE_CUBLAS
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model->backend = ggml_backend_cuda_init(0);
        if (!model->backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif
    if(!model->backend) {
        // fallback to CPU backend
        model->backend = ggml_backend_cpu_init();
    }

    model->compute_buffer = ggml_backend_alloc_buffer(model->backend, model->compute_buffer_size);

    // create context
    model->ctx = ggml_init(params);
    struct ggml_context * ctx = model->ctx;

    // create a allocator
    ggml_allocr * alloc = ggml_allocr_new_from_buffer(model->compute_buffer);

    // alloc memory
    // ggml_allocr_alloc(alloc, model.a);

    model->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, n_batch); 
    model->noise = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_latent, n_batch); 
    ggml_allocr_alloc(alloc, model->input);
    ggml_allocr_alloc(alloc, model->noise);
    fprintf(stderr, "%s: A \n", __func__);

    model->encode1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, enc1_out); 
    model->encode1_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, enc1_out);          
    model->encode2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc1_out, enc2_out); 
    model->encode2_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, enc2_out);          
    model->encode3_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc2_out, enc3_out); 
    model->encode3_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, enc3_out);          
    ggml_allocr_alloc(alloc, model->encode1_weight);
    ggml_allocr_alloc(alloc, model->encode2_weight);
    ggml_allocr_alloc(alloc, model->encode3_weight);
    ggml_allocr_alloc(alloc, model->encode1_bias);
    ggml_allocr_alloc(alloc, model->encode2_bias);
    ggml_allocr_alloc(alloc, model->encode3_bias);

    fprintf(stderr, "%s: B \n", __func__);

    model->logsd_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc3_out, n_latent); 
    model->logsd_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_latent);          
    ggml_allocr_alloc(alloc, model->logsd_weight);
    ggml_allocr_alloc(alloc, model->logsd_bias);

    model->mu_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc3_out, n_latent); 
    model->mu_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_latent);          
    ggml_allocr_alloc(alloc, model->mu_weight);
    ggml_allocr_alloc(alloc, model->mu_bias);

    fprintf(stderr, "%s: C \n", __func__);

    model->decode4_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_latent, dec4_out); 
    model->decode4_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec4_out);          
    model->decode3_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dec4_out, dec3_out); 
    model->decode3_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec3_out);          
    model->decode2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dec3_out, dec2_out); 
    model->decode2_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec2_out);          

    model->decode1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dec2_out, n_input); 
    model->decode1_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_input);          

    ggml_allocr_alloc(alloc, model->decode4_weight);
    ggml_allocr_alloc(alloc, model->decode3_weight);
    ggml_allocr_alloc(alloc, model->decode2_weight);
    ggml_allocr_alloc(alloc, model->decode1_weight);
    fprintf(stderr, "%s: C.1 \n", __func__);
    ggml_allocr_alloc(alloc, model->decode4_bias);
    ggml_allocr_alloc(alloc, model->decode3_bias);
    fprintf(stderr, "%s: C.2 \n", __func__);
    ggml_allocr_alloc(alloc, model->decode2_bias);
    fprintf(stderr, "%s: C.3 \n", __func__);
    ggml_allocr_alloc(alloc, model->decode1_bias);

    fprintf(stderr, "%s: D \n", __func__);
    ggml_allocr_free(alloc);

}

static void set_param_model(struct mnist_vae_model * model) {

    struct ggml_context* ctx = model->ctx;

    ggml_set_param(ctx, model->encode1_weight);
    ggml_set_param(ctx, model->encode1_bias);
    ggml_set_param(ctx, model->encode2_weight);
    ggml_set_param(ctx, model->encode2_bias);
    ggml_set_param(ctx, model->encode3_weight);
    ggml_set_param(ctx, model->encode3_bias);

    ggml_set_param(ctx, model->decode1_weight);
    ggml_set_param(ctx, model->decode1_bias);
    ggml_set_param(ctx, model->decode2_weight);
    ggml_set_param(ctx, model->decode2_bias);
    ggml_set_param(ctx, model->decode3_weight);
    ggml_set_param(ctx, model->decode3_bias);
    ggml_set_param(ctx, model->decode4_weight);
    ggml_set_param(ctx, model->decode4_bias);

    ggml_set_param(ctx, model->logsd_weight);
    ggml_set_param(ctx, model->logsd_bias);
    ggml_set_param(ctx, model->mu_weight);
    ggml_set_param(ctx, model->mu_bias);
    
}

static void load_data(ggml_backend_t backend, struct ggml_tensor * dst, struct ggml_tensor * src){
    if(ggml_backend_is_cpu(backend)) {
        memcpy(dst->data, src->data, ggml_nbytes(dst));
    } else {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(dst));
    }

}


static void randomize_bias( struct mnist_vae_model * model, 
                            struct ggml_context    * ctx0,
                            struct random_uniform_distribution * rnd,
                            struct ggml_tensor     * w,
                            struct ggml_tensor     * b){
        int64_t fan_in = w->ne[0];
        float scale = 1./sqrt((float)fan_in);        
        struct ggml_tensor * rn = ggml_dup_tensor(ctx0, b);        
        randomize_tensor_uniform(rn, rnd);
        rn = ggml_scale(ctx0, rn, scale);
        load_data(model->backend, b, rn);
}

static void zero_bias_model(struct mnist_vae_model * model, int seed){

    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct random_uniform_distribution * rnd = init_random_uniform_distribution(seed, -1.f, 1.f);

    struct ggml_context * ctx0 = ggml_init(params);
    randomize_bias(model, ctx0, rnd, model->decode1_weight, model->decode1_bias);
    randomize_bias(model, ctx0, rnd, model->decode2_weight, model->decode2_bias);
    randomize_bias(model, ctx0, rnd, model->decode3_weight, model->decode3_bias);
    randomize_bias(model, ctx0, rnd, model->decode4_weight, model->decode4_bias);
    randomize_bias(model, ctx0, rnd, model->encode1_weight, model->encode1_bias);
    randomize_bias(model, ctx0, rnd, model->encode2_weight, model->encode2_bias);
    randomize_bias(model, ctx0, rnd, model->encode3_weight, model->encode3_bias);
    randomize_bias(model, ctx0, rnd, model->mu_weight, model->mu_bias);
    randomize_bias(model, ctx0, rnd, model->logsd_weight, model->logsd_bias);
    
    ggml_free(ctx0);
    free_random_uniform_distribution(rnd);


}

static void print_row(struct ggml_tensor * probs, int i) {
    for (int k = 0; k < probs->ne[0]; ++k) {
        float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
        printf(" %.6f", p);
    }
    printf("\n");
}

static void print_matrix(struct ggml_tensor * probs) {
    assert(ggml_is_matrix(probs));
    for (int i = 0; i < probs->ne[1]; ++i) {
        for (int k = 0; k < probs->ne[0]; ++k) {
            float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
            printf(" %.6f", p);
        }
        printf("\n");
    }
}



static void randomize_model(struct mnist_vae_model * model, int seed, float mean, float std, float min, float max) {
    const auto & hparams = model->hparams;
    
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct random_normal_distribution * rnd = init_random_normal_distribution(seed, mean, std, min, max);

    struct ggml_tensor * rn = NULL;
    // fprintf(stderr, "%s: before dup model tensor \n", __func__); 
    rn = ggml_dup_tensor(ctx0, model->encode1_weight);
    // fprintf(stderr, "%s: after dup model tensor \n", __func__); 
    // int ndim = ggml_n_dims(rn);
    // if (rn->backend == GGML_BACKEND_GPU) 
    //     fprintf(stderr, "%s: rn is  on GPU \n", __func__); 
    // else
    //     fprintf(stderr, "%s: rn is  on CPU with %d dims\n", __func__, ndim); 
    // ndim = ggml_n_dims(model->encode1_weight);
    // if (model->encode1_weight->backend == GGML_BACKEND_GPU) 
    //     fprintf(stderr, "%s: model weight is  on GPU \n", __func__); 
    // else{
    //     int64_t *ne = model->encode1_weight->ne;
    //     fprintf(stderr, "%s: model weight is  on CPU with (%d, %d, %d, %d) dims \n", 
    //          __func__, ne[0], ne[1], ne[2], ne[3]);     
    // }
    rn = randomize_tensor_normal(rn, rnd);
    // print_matrix(rn);
    // fprintf(stderr, "%s: after randmize tensor \n", __func__); 
    load_data(model->backend, model->encode1_weight, rn);
    // fprintf(stderr, "%s: after load into model \n", __func__); 

    rn = ggml_dup_tensor(ctx0, model->encode2_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->encode2_weight, rn);

    rn = ggml_dup_tensor(ctx0, model->encode3_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->encode3_weight, rn);

    rn = ggml_dup_tensor(ctx0, model->decode1_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->decode1_weight, rn);

    rn = ggml_dup_tensor(ctx0, model->decode2_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->decode2_weight, rn);

    rn = ggml_dup_tensor(ctx0, model->decode3_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->decode3_weight, rn);

    rn = ggml_dup_tensor(ctx0, model->decode4_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->decode4_weight, rn);

    rn = ggml_dup_tensor(ctx0, model->logsd_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->logsd_weight, rn);

    rn = ggml_dup_tensor(ctx0, model->mu_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->mu_weight, rn);


    // struct random_normal_distribution * rnd0 = init_random_normal_distribution(seed, 0, 1.f, min, max);

    // rn = ggml_dup_tensor(ctx0, model->noise);
    // randomize_tensor_normal(rn, rnd0);
    // load_data(model->backend, model->noise, rn);



    // randomize_tensor_normal(model->encode1_weight, rnd);


    // randomize_tensor_normal(model->encode2_weight, rnd);
    // randomize_tensor_normal(model->encode3_weight, rnd);
    // randomize_tensor_normal(model->decode1_weight, rnd);
    // randomize_tensor_normal(model->decode2_weight, rnd);
    // randomize_tensor_normal(model->decode3_weight, rnd);
    // randomize_tensor_normal(model->decode4_weight, rnd);
    // randomize_tensor_normal(model->logsd_weight, rnd);
    // randomize_tensor_normal(model->mu_weight, rnd);
    free_random_normal_distribution(rnd);
    // free_random_normal_distribution(rnd0);

    // ggml_print_objects(ctx0);

    ggml_free(ctx0);
}



static void train_forward_batch(
    struct mnist_vae_model   * model,
    struct ggml_context      * ctx0,
    const  int32_t             n_batch
) {
  
    /*
    for training, do not use *_inplace operators as they don't allow
    backpropagation
    */
    int32_t n_input = model->hparams.n_input;

    ggml_set_name(model->input, "input");
    ggml_set_name(model->noise, "noise");
    struct ggml_tensor* h = ggml_mul_mat(ctx0, model->encode1_weight, ggml_cont(ctx0, model->input)); 
    ggml_set_name(h, "encode1_w");

    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->encode1_bias, h)); 
    ggml_set_name(h, "encode1_b");
    
    h = ggml_relu(ctx0, h); 
    ggml_set_name(h, "encode1_relu");
    // fprintf(stderr, "%s: done with first build\n", __func__);       

    // fprintf(stderr, "%s: build 2 %d\n", __func__, __LINE__);       
    h = ggml_mul_mat(ctx0, model->encode2_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->encode2_bias, h)); 
    ggml_set_name(h, "encode2_b");
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->encode2_weight, h), ggml_repeat(ctx0, model->encode2_bias, );
    h = ggml_relu(ctx0, h); 
    // fprintf(stderr, "%s: build 3 %d\n", __func__, __LINE__);       
    h = ggml_mul_mat(ctx0, model->encode3_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->encode3_bias, h)); 
    ggml_set_name(h, "encode3_b");
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->encode3_weight, h), model->encode3_bias);
    h = ggml_relu(ctx0, h); 
        

    struct ggml_tensor* h1 = ggml_mul_mat(ctx0, model->mu_weight, h);
    h1 = ggml_add(ctx0, h1, ggml_repeat(ctx0, model->mu_bias, h1)); 
    struct ggml_tensor* mu = h1;

    struct ggml_tensor* logsd =  ggml_mul_mat(ctx0, model->logsd_weight, h);
    logsd = ggml_add(ctx0, logsd, ggml_repeat(ctx0, model->logsd_bias, logsd)); 
    ggml_set_name(logsd, "logsd");

    h1 = ggml_sqr(ctx0, h1);
    ggml_set_name(h1, "meansq");
    struct ggml_tensor* sd = ggml_exp(ctx0, logsd);
    ggml_set_name(sd, "sd");
    struct ggml_tensor* var = ggml_sqr(ctx0, sd);
    ggml_set_name(var, "var");
    struct ggml_tensor* h3 = ggml_add(ctx0, ggml_scale(ctx0, h1, 0.5f), 
                                            ggml_scale(ctx0, sd, 0.5f));
    if(ggml_backend_is_cpu(model->backend)){ 
       h3 = ggml_sub(ctx0, h3, ggml_scale(ctx0, logsd, 0.5f));
    }else{
        h3 = ggml_add(ctx0, h3, ggml_scale(ctx0, logsd, -1.f));
    }
    ggml_set_name(h3, "kldiv_plus_half");
    if(ggml_backend_is_cpu(model->backend)){ 
       h3 = ggml_add1(ctx0, h3, ggml_new_f32(ctx0, -0.5f));
    }else{
       h3 = ggml_add1(ctx0, h3, ggml_new_f32(ctx0, -0.5f));
    }    
    ggml_set_name(h3, "kldiv");   
    h3 = ggml_scale(ctx0, ggml_sum(ctx0, h3), 1.f/(float)n_batch);
    // h3 = ggml_sum(ctx0, h3);
    struct ggml_tensor* klloss = h3;
    ggml_set_name(h3, "klloss");   

    h3 = ggml_mul(ctx0, model->noise, sd);
    ggml_set_name(h3, "sdnoise");
    h3 = ggml_add(ctx0, h3, mu);
    ggml_set_name(h3, "sample");

    h = ggml_mul_mat(ctx0, model->decode4_weight, h3); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode4_bias, h)); 
    h = ggml_relu(ctx0, h);
    h = ggml_mul_mat(ctx0, model->decode3_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode3_bias, h)); 
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->decode3_weight, h), model->decode3_bias);
    h = ggml_relu(ctx0, h);
    h = ggml_mul_mat(ctx0, model->decode2_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode2_bias, h)); 
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->decode2_weight, h), model->decode2_bias);
    h = ggml_relu(ctx0, h);
    h = ggml_mul_mat(ctx0, model->decode1_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode1_bias, h)); 
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->decode1_weight, h), model->decode1_bias);
    
    // run the computation
    // ggml_build_forward_expand(gf, inpL);
    // *l2 = h; // 2nd loss is sigmoid cross entropy
    ggml_set_name(h, "src1_sigloss");
    struct ggml_tensor* x = h;
    struct ggml_tensor* z = model->input;
    h = ggml_sub(ctx0, ggml_relu(ctx0, x), ggml_mul(ctx0, x, z));
    h = ggml_add(ctx0, h, ggml_log(ctx0, 
                             ggml_add1(ctx0,  
                                  ggml_exp(ctx0, ggml_neg(ctx0, ggml_abs(ctx0, x))),
                                  ggml_new_f32(ctx0, 1.f))));
    // h = ggml_scale(ctx0, ggml_sum(ctx0, h), 1.f/((float)(n_batch*n_input)));
    h = ggml_scale(ctx0, ggml_sum(ctx0, h), 1.f/((float)(n_batch*n_input)));
    // h = ggml_sum(ctx0, h);
    ggml_set_name(h, "sigloss");

    h = ggml_add(ctx0, h, klloss);
    ggml_set_name(h, "totloss");

    ggml_set_name(model->decode1_bias, "decode1_bias");       
    ggml_set_name(model->decode1_weight, "decode1_weight");       
    ggml_set_name(model->decode2_bias, "decode2_bias");       
    ggml_set_name(model->decode2_weight, "decode2_weight");       
    ggml_set_name(model->decode3_bias, "decode3_bias");       
    ggml_set_name(model->decode3_weight, "decode3_weight");       
    ggml_set_name(model->decode4_bias, "decode4_bias");       
    ggml_set_name(model->decode4_weight, "decode4_weight");       
    
    ggml_set_name(model->encode1_bias, "encode1_bias");       
    ggml_set_name(model->encode1_weight, "encode1_weight");       
    ggml_set_name(model->encode2_bias, "encode2_bias");       
    ggml_set_name(model->encode2_weight, "encode2_weight");       
    ggml_set_name(model->encode3_bias, "encode3_bias");       
    ggml_set_name(model->encode3_weight, "encode3_weight");       

    ggml_set_name(model->mu_bias, "mu_bias");       
    ggml_set_name(model->mu_weight, "mu_weight");       
    
    ggml_set_name(model->logsd_bias, "logsd_bias");       
    ggml_set_name(model->logsd_weight, "logsd_weight");       

    return;
}

struct ggml_cgraph* build_train_graph_batch(struct mnist_vae_model * model, struct ggml_allocr * allocr,
// static void build_train_graph_batch(struct mnist_vae_model * model, 
                                //    struct ggml_context * ctx0,
                                   int32_t n_batch) {
    // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead() * 1024 + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    const auto & hparams = model->hparams;


    struct ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf.data(),
        /*.no_alloc   =*/true,  // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };
    // LOG_DEBUG("mem_size %u ", params.mem_size);

    struct ggml_context* ctx0 = ggml_init(params);

    // struct ggml_context* ctx0 = model->ctx;

    struct ggml_cgraph* gf = ggml_new_graph(ctx0);


    train_forward_batch(model, ctx0, n_batch);

    // ggml_build_forward_expand(gf, l1);
    // ggml_build_forward_expand(gf, ggml_get_tensor(ctx0, "klloss"));
    // ggml_build_forward_expand(gf, ggml_get_tensor(ctx0, "sigloss"));
    ggml_build_forward_expand(gf, ggml_get_tensor(ctx0, "totloss"));

    // struct ggml_tensor * t = get_tensor_from_graph(gf, "node_38");
    // if(t != NULL){
    //     printf("node_38 found \n ");
    //     if( t->src[0])
    //         printf("src0: %s \n ", t->src[0]->name);
    //     if(t->src[1])
    //         printf("src1: %s \n ", t->src[1]->name);
    // } 
    // else{
    //     printf("node_38 not found \n ");
    // }


    ggml_free(ctx0);

    return gf;
}


struct ggml_cgraph * compute_graph_batch(struct mnist_vae_model * model, 
                                         struct ggml_allocr * allocr, 
                                         int32_t n_batch) {
    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = build_train_graph_batch(model, allocr, n_batch);

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);
    int n_threads = 1;

    if (ggml_backend_is_cpu(model->backend)) {
        ggml_backend_cpu_set_n_threads(model->backend, n_threads);
    }

    ggml_backend_graph_compute(model->backend, gf);

    //ggml_graph_print(gf);

    return gf;
}


static void lshift_examples(struct ggml_tensor * tokens_input, struct ggml_tensor * targets, int n_shift) {
    int n_tokens = tokens_input->ne[0];
    int n_vocab = targets->ne[0];
    for (int i=0; i<n_tokens-n_shift; ++i) {
        ggml_set_i32_1d(tokens_input, i, ggml_get_i32_1d(tokens_input, i + n_shift));
        for (int k=0; k<n_vocab; ++k) {
            ggml_set_f32_1d(targets, i*n_vocab + k, ggml_get_f32_1d(targets, (i + n_shift)*n_vocab + k));
        }
    }
}



static struct ggml_tensor * square_error_loss(
    struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b
) {
    // todo: instead of a-b: a[1:]-b[:-1]
    return ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, a, b)));
}

static struct ggml_tensor * cross_entropy_loss(
    struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b
) {
    const float eps = 1e-3f;
    return
        ggml_sum(ctx,
            ggml_neg(ctx,
                ggml_sum_rows(ctx,
                    ggml_mul(ctx,
                        ggml_soft_max(ctx, a),
                        ggml_log(ctx,
                            ggml_add1(ctx,
                                ggml_soft_max(ctx, b),
                                ggml_new_f32(ctx, eps)))))));
}

#define COUNT_TRAIN 60000ll

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


int main(int argc, char ** argv) {
    

    struct mnist_vae_model model;    

    bool use_gpu = false;
    int n_batch = 1;
    
    


    struct ggml_init_params lcparams;
    lcparams.mem_size   = static_cast<size_t>(model.get_num_tensors() * ggml_tensor_overhead()) + 1 * 1024 * 1024;
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = true;

    // model.params_buffer_size = 10 * 1024 * 1024;  // 10 MB, for padding
    // model.params_buffer_size += model.calculate_mem_size();

    init_model(&model, use_gpu, n_batch);

    printf(" model inited\n");
  
    set_param_model(&model);
    printf(" mdoel params set \n");

    randomize_model(&model, 1337, 0.0f, 0.1f, -1.0f, +1.0f);
    // randomize_model(&model, 1337, 0.0f, .1f, -FLT_MAX, FLT_MAX);

    // struct ggml_tensor * h = model.encode1_weight;
    // int64_t *ne = h->ne;
    // printf("h is %s \n", h->name);
    // printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    
    printf("modle initialzed with random numbers \n");
    zero_bias_model(&model, 1337);
    // h = model.encode1_weight;
    // ne = h->ne;
    // printf("h is %s \n", h->name);
    // printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    // printf("after zero bias \n"); 
    printf("modle bias zeroed \n");
    


    
   
    uint8 *train_data = (uint8 *)malloc(COUNT_TRAIN * 28 * 28 * sizeof(uint8));

    if(read_data(train_data, COUNT_TRAIN) > 0){
        fprintf(stderr, "Error in read in Mnist training data! \n");
        exit(1);
    }

    uint8_t *buf = train_data;
    std::vector<float> digit;

    digit.resize(n_batch*28*28);

    int32_t num_batches = COUNT_TRAIN / n_batch;

    std::vector<uint8_t> work_buffer;
    
    // h = model.encode1_weight;
    // ne = h->ne;
    // printf("h is %s \n", h->name);
    // printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);

    // h = ggml_get_tensor(model.ctx, "input");
    // ne = h->ne;
    // printf("h is %s \n", h->name);
    // printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    

    // for(int i = 0; i < gf->n_nodes; i++){
    //     h = gf->nodes[i];
    //     ne = h->ne;
    //     printf("%s: (%ld, %ld, %ld, %ld) \n", h->name, ne[0], ne[1], ne[2], ne[3]);
    // }
    // for(int i = 0; i < gf->n_leafs; i++){
    //     h = gf->leafs[i];
    //     ne = h->ne;
    //     printf("%s: (%ld, %ld, %ld, %ld) \n", h->name, ne[0], ne[1], ne[2], ne[3]);
    // }  

    ggml_backend_buffer_t graph_compute; // for compute
    struct ggml_allocr * allocr = NULL;

    {
        allocr = ggml_allocr_new_measure_from_backend(model.backend);

        //create the worst case graph for memory usage estimation
         printf("try build graph \n");   
        struct ggml_cgraph * gf = build_train_graph_batch(&model, allocr, n_batch);
         printf("after try \n");   
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);
        ggml_allocr_free(allocr);

        // compute the required memory
        graph_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
        allocr = ggml_allocr_new_from_buffer(graph_compute);
        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }
   

    size_t    compute_size = 1024ll*1024ll*1024ll;
    uint8_t * compute_addr = new uint8_t[compute_size];

    struct random_normal_distribution * rnd = init_random_normal_distribution(1337, 0, 1.f, -FLT_MAX, FLT_MAX);

    float *rnds = new float[model.hparams.n_latent*n_batch];

    for (int ex=0; ex<num_batches; ++ex) {
        printf(" enter loop %d \n", ex);
        struct ggml_init_params optparams = {
            /*.mem_size   =*/ compute_size,
            /*.mem_buffer =*/ compute_addr,
            /*.no_alloc   =*/ false,
        };

        for (int ib = 0; ib < n_batch; ib++) 
            for (int row = 0; row < 28; row++) {
                for (int col = 0; col < 28; col++) {
                    digit[ib*28*28 + row*28 + col] = ((float)buf[ib*28*28 + row*28 + col])/255.f;
                    // printf(" %.2f,  ", digit[ib*28*28 + row*28 + col]);
                }
                // printf("\n");
            }


        buf += n_batch*28*28;  
                
        struct ggml_context * ctx0 = ggml_init(optparams);
        // printf("this is adam optimizer ctx %p \n ", (void *)ctx0);
        // struct ggml_context * ctxm = model.ctx;
        // printf("mode ctx is %p \n ", (void *)ctxm );
        // ggml_print_objects(ctxm);
        struct ggml_tensor * input_batch = ggml_get_tensor(model.ctx, "input");
        struct ggml_tensor * noise_batch = ggml_get_tensor(model.ctx, "noise");
        
        float mi = FLT_MAX, mx = -FLT_MAX;  
        for(int i = 0; i < model.hparams.n_latent*n_batch; i++){            
            rnds[i] = frand_normal(rnd);
            if(mi > rnds[i])
                mi = rnds[i];
            if(mx < rnds[i])
                mx = rnds[i];
            // printf(" %f, ", rnds[i]);
            // if((i+1)%model.hparams.n_latent == 0)
            //    printf("\n");
        }
        printf("min,max of noise %f, %f\n", mi, mx);
        
        
        

        if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
        ) {
            memcpy(input_batch->data, digit.data(), ggml_nbytes(input_batch));
            memcpy(noise_batch->data, rnds, ggml_nbytes(noise_batch));
            // memcpy(model.b->data, b, ggml_nbytes(model.b));
        } else {
            ggml_backend_tensor_set(input_batch, digit.data(), 0, ggml_nbytes(input_batch));  // cuda requires copy the data directly to device
            ggml_backend_tensor_set(noise_batch, rnds, 0, ggml_nbytes(noise_batch));  // cuda requires copy the data directly to device
        } 
        // printf("copy to input \n", ex);

        
        // printf("after compute graph \n", ex);
        
        struct ggml_cgraph * gf_res = compute_graph_batch(&model, allocr, n_batch);
        // ggml_graph_dump_dot(gf_res, NULL, "mnist-vae-forward.dot");
        // struct ggml_tensor * err_kl = get_tensor_from_graph(gf_res, "klloss");
        // struct ggml_tensor * err_sig = get_tensor_from_graph(gf_res, "sigloss");
        struct ggml_tensor * err_tot = get_tensor_from_graph(gf_res, "totloss");
        // if(!err_kl){
        //     fprintf(stderr, "something wrong with graph compute \n");
        //     exit(1);
        // }
        // if(!err_sig){
        //     fprintf(stderr, "something wrong with graph compute \n");
        //     exit(1);
        // }
        if(!err_tot){
            fprintf(stderr, "something wrong with graph compute \n");
            exit(1);
        }

        // int64_t *ne = err_kl->ne;
        // printf(" loss ne = (%d, %d, %d, %d)\n", ne[0], ne[1], ne[2], ne[3]);

        // float error_before_opt0 = ggml_get_f32_1d(err_kl, 0);
        // float error_before_opt1 = ggml_get_f32_1d(err_sig, 0);
        float error_before_opt1 = ggml_get_f32_1d(err_tot, 0);
        // printf(" after compute error is  %f,%f \n", error_before_opt0, error_before_opt1);
        printf(" after compute error is  %f\n", error_before_opt1);

        struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_ADAM);
        // struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_LBFGS);
        opt_params.print_backward_graph = false;
        opt_params.print_forward_graph = false;
        opt_params.adam.n_iter = 16;
        // opt_params.lbfgs.n_iter = 16;
        // opt_params.lbfgs.max_linesearch = 50;
        

        // printf(" before opt  \n");
        // ggml_opt(ctx0, opt_params, err_kl);        
        // ggml_opt(ctx0, opt_params, err_sig);
        ggml_opt(ctx0, opt_params, err_tot);
        // exit(1);
        // printf(" after opt  \n");


        //
        // ggml_build_forward_expand(gf, e);        

        // float error_after_opt = ggml_get_f32_1d(e, 0);

        ggml_free(ctx0);
    }

    printf("done\n");

    // ggml_free(kv_self.ctx);
    // ggml_free(model_lora.ctx);
    ggml_free(model.ctx);
    free_random_normal_distribution(rnd);

    return 0;
}
