#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"
#include "ggml/ggml-backend.h"
#include "train.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <map>
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

static uint8 float2pixel(float f){
   return (uint8)((f >= 1.0 ? 255 : (f <= 0.0 ? 0 : (int)floor(f * 256.0))));
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
            return gf->nodes[i];
            break;
        } 
    }
    for(int i = 0; i < gf->n_leafs; i++) {
        if(strcmp(ggml_get_name(gf->leafs[i]), name) == 0) {
            return gf->leafs[i];
            break;
        } 
    }
    return res;
}

// default hparams
struct mnist_hparams {
    int32_t n_input   = 784;
    

    int32_t n_latent  = 20;    
    int32_t enc1_out  = 400;    
    int32_t dec2_out  = 400;


    // int32_t n_latent  = 5;    
    // int32_t enc1_out  = 10;    
    // int32_t dec2_out  = 10;

    // int32_t n_latent  = 10;
    // int32_t enc1_out  = 100;
    // int32_t enc2_out  = 50;
    // int32_t enc3_out  = 25;
    // int32_t dec4_out  = 25;
    // int32_t dec3_out  = 50;
    // int32_t dec2_out  = 100;
     

};

struct mnist_vae_model {
    mnist_hparams hparams;

    struct ggml_tensor * input;
    struct ggml_tensor * noise;

    struct ggml_tensor * encode1_weight;
    struct ggml_tensor * encode1_bias;    


    struct ggml_tensor * logsd_weight;
    struct ggml_tensor * logsd_bias;
    struct ggml_tensor * mu_weight;
    struct ggml_tensor * mu_bias;    
   
    struct ggml_tensor * decode1_weight;
    struct ggml_tensor * decode1_bias;
    struct ggml_tensor * decode2_weight;
    struct ggml_tensor * decode2_bias;
   
    ggml_backend_t backend = NULL;
    
    ggml_backend_buffer_t compute_buffer = NULL;
    size_t compute_buffer_size = 0;

    ggml_backend_buffer_t params_buffer = NULL;
    size_t params_buffer_size = 0;
    struct ggml_allocr* compute_allocr   = NULL;

    struct ggml_context * ctx;

    std::map<void *, struct ggml_tensor*> data_map;

    size_t calculate_mem_size() {
        double mem_size = 0;
        mem_size += (hparams.n_input  * hparams.enc1_out + hparams.enc1_out) * ggml_type_sizef(GGML_TYPE_F32);  // encode1_w+b
        
        mem_size += (hparams.enc1_out * hparams.n_latent + hparams.n_latent) * ggml_type_sizef(GGML_TYPE_F32);  // logsd
        mem_size += (hparams.enc1_out * hparams.n_latent + hparams.n_latent) * ggml_type_sizef(GGML_TYPE_F32);  // mu
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
    const int32_t dec2_out  = hparams.dec2_out;
    
 
    
    size_t buffer_size = 0;
    {
        buffer_size += (n_input  * n_batch ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (n_latent * n_batch ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (n_input  * enc1_out + enc1_out ) * ggml_type_size(GGML_TYPE_F32);        
        buffer_size += 2 * (enc1_out * n_latent + n_latent ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (n_latent * dec2_out + dec2_out ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (dec2_out * n_input  + n_input ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += (dec2_out * n_input  + n_input ) * ggml_type_size(GGML_TYPE_F32); 
        buffer_size += 1024; // overhead
    }
    model->compute_buffer_size = buffer_size;

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %d bytes\n", __func__, (int) buffer_size);

    int num_tensors = 10 * 2 + 2; // *2 to acount for their grads
    struct ggml_init_params params {
            // /*.mem_size   =*/ ggml_tensor_overhead() * (num_tensors + 2),
            /*.mem_size   =*/ ggml_tensor_overhead() * 1024 + ggml_graph_overhead(),
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

    // model->compute_buffer = ggml_backend_alloc_buffer(model->backend, model->compute_buffer_size);

    // create context
    model->ctx = ggml_init(params);
    struct ggml_context * ctx = model->ctx;

    // create a allocator
    // ggml_allocr * alloc = ggml_allocr_new_from_buffer(model->compute_buffer);

    // alloc memory
    // ggml_allocr_alloc(alloc, model.a);

    model->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, n_batch); 
    model->noise = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_latent, n_batch); 
    

    model->encode1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, enc1_out); 
    model->encode1_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, enc1_out);          


    model->logsd_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc1_out, n_latent); 
    model->logsd_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_latent);          

    model->mu_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc1_out, n_latent); 
    model->mu_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_latent);          


    model->decode2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_latent, dec2_out); 
    model->decode2_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec2_out);          

    model->decode1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dec2_out, n_input); 
    model->decode1_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_input);          


}

static void print_compnent(struct mnist_vae_model * model, struct ggml_tensor * t){
    fprintf(stderr, " name: %s, addr: %p, buffer: %p \n", t->name, (void *)t->data, (void *)t->buffer);
    model->data_map[(void *)t->data] = t;
}

static void print_model_data_addr(struct mnist_vae_model * model){
    fprintf(stderr, " ********************************* \n");
    print_compnent(model, model->input);
    print_compnent(model, model->noise);
    print_compnent(model, model->encode1_weight);
    print_compnent(model, model->encode1_bias);
    print_compnent(model, model->decode1_weight);
    print_compnent(model, model->decode1_bias);
    print_compnent(model, model->decode2_weight);
    print_compnent(model, model->decode2_bias);
    print_compnent(model, model->mu_weight);
    print_compnent(model, model->mu_bias);
    print_compnent(model, model->logsd_weight);
    print_compnent(model, model->logsd_bias);
    fprintf(stderr, " ********************************* \n");
}


static void set_param_model(struct mnist_vae_model * model) {

    struct ggml_context* ctx = model->ctx;

    ggml_set_param(ctx, model->encode1_weight);
    ggml_set_param(ctx, model->encode1_bias);
   

    ggml_set_param(ctx, model->decode1_weight);
    ggml_set_param(ctx, model->decode1_bias);
    ggml_set_param(ctx, model->decode2_weight);
    ggml_set_param(ctx, model->decode2_bias);
    
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
        float scale = sqrt(1./(float)fan_in);        
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
    randomize_bias(model, ctx0, rnd, model->encode1_weight, model->encode1_bias);
    randomize_bias(model, ctx0, rnd, model->mu_weight, model->mu_bias);
    randomize_bias(model, ctx0, rnd, model->logsd_weight, model->logsd_bias);
    
    ggml_free(ctx0);
    free_random_uniform_distribution(rnd);
}



static void print_row(struct ggml_tensor * probs, int i) {
    if(probs->backend != GGML_BACKEND_CPU){
        const int64_t ne = ggml_nelements(probs) ;
        int64_t ne0 = probs->ne[0];
        float *g = new float[ne0]; 
        int64_t bytes = ggml_nbytes(probs);
        ggml_backend_tensor_get(probs, g, i*sizeof(float)*ne0, sizeof(float)*ne0);
        for (int k = 0; k < ne0; ++k) {
            printf(" %f", g[k]);
        }
        printf("\n");  
        delete g;
    }
    else{
        for (int k = 0; k < probs->ne[0]; ++k) {
            float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
            printf(" %.f", p);
        }
        printf("\n");
    }
}

static void print_matrix(struct ggml_tensor * p) {
    assert(ggml_is_matrix(p));
    const int64_t ne = ggml_nelements(p) ;
    if(p->backend != GGML_BACKEND_CPU){
        float *g = new float[ne]; 
        int64_t bytes = ggml_nbytes(p);
        ggml_backend_tensor_get(p, g, 0, bytes);
        for (int i = 0; i < p->ne[1]; ++i) {
            for (int k = 0; k < p->ne[0]; ++k) {
                printf(" %f", g[i*p->ne[0] + k]);
            }
            printf("\n");
        }
        delete g;
    }else{            
        // TODO: add function to get all elements at once
        for (int i = 0; i < p->ne[1]; ++i) {
            for (int k = 0; k < p->ne[0]; ++k) {
                float f = ggml_get_f32_1d(p, i*p->ne[0] + k);
                printf(" %f", f);
            }
            printf("\n");
        }
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

    

    rn = ggml_dup_tensor(ctx0, model->decode1_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->decode1_weight, rn);

    rn = ggml_dup_tensor(ctx0, model->decode2_weight);
    randomize_tensor_normal(rn, rnd);
    load_data(model->backend, model->decode2_weight, rn);
    

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
    // struct ggml_tensor* h = ggml_mul_mat(ctx0, model->encode1_weight, ggml_cont(ctx0, model->input)); 
    struct ggml_tensor* h = ggml_mul_mat(ctx0, model->encode1_weight, model->input); 
    ggml_set_name(h, "encode1_w");

    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->encode1_bias, h)); 
    ggml_set_name(h, "encode1_b");
    
    h = ggml_relu(ctx0, h); 
    ggml_set_name(h, "encode1_relu");
    // fprintf(stderr, "%s: done with first build\n", __func__);       
   
        

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
    // struct ggml_tensor* var = ggml_sqr(ctx0, sd);
    // ggml_set_name(var, "var");
    // struct ggml_tensor* h3 = ggml_add(ctx0, ggml_scale(ctx0, h1, 0.5f), 
    //                                         ggml_scale(ctx0, sd, 0.5f));
    // h3 = ggml_add(ctx0, h3, ggml_scale(ctx0, logsd, -0.5f));
    struct ggml_tensor* h3 = ggml_add(ctx0, h1, sd);
    h3 = ggml_sub(ctx0, h3, logsd);
    h3 = ggml_scale(ctx0, h3, 0.5f);
    ggml_set_name(h3, "kldiv_plus_half");
    h3 = ggml_add1(ctx0, h3, ggml_new_f32(ctx0, -0.5f));
    ggml_set_name(h3, "kldiv");   
    // h3 = ggml_scale(ctx0, ggml_sum(ctx0, h3), 1.f/(float)n_batch);
    h3 = ggml_sum(ctx0, h3);
    struct ggml_tensor* klloss = h3;
    ggml_set_name(klloss, "klloss");   

    h3 = ggml_mul(ctx0, model->noise, ggml_exp(ctx0, ggml_scale(ctx0, logsd, 0.5f)));
    ggml_set_name(h3, "sdnoise");
    h3 = ggml_add(ctx0, h3, mu);
    ggml_set_name(h3, "sample");

    
    h = ggml_mul_mat(ctx0, model->decode2_weight, h3); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode2_bias, h)); 
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->decode2_weight, h), model->decode2_bias);
    h = ggml_relu(ctx0, h);
    ggml_set_name(h, "decode2_relu");
    h = ggml_mul_mat(ctx0, model->decode1_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode1_bias, h)); 
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->decode1_weight, h), model->decode1_bias);
    
    // 2nd loss is sigmoid cross entropy
    ggml_set_name(h, "src1_sigloss");
    struct ggml_tensor* x = h;
    struct ggml_tensor* recon = ggml_sigmoid(ctx0, x);
    ggml_set_name(recon, "reconstructed");

    struct ggml_tensor* z = model->input;
    
    h = ggml_sub(ctx0, ggml_relu(ctx0, x), ggml_mul(ctx0, x, z));
    h = ggml_add(ctx0, h, ggml_log(ctx0, 
                             ggml_add1(ctx0,  
                                  ggml_exp(ctx0, ggml_neg(ctx0, ggml_abs(ctx0, x))),
                                  ggml_new_f32(ctx0, 1.f))));
    // h = ggml_scale(ctx0, ggml_sum(ctx0, h), 1.f/((float)(n_batch*n_input)));
    // h = ggml_scale(ctx0, ggml_sum(ctx0, h), 1.f/((float)(n_batch)));
    h = ggml_sum(ctx0, h);
    ggml_set_name(h, "sigloss");

    // h = ggml_add(ctx0, h, klloss);
    struct ggml_tensor* hf = ggml_add(ctx0, klloss, h);
    // h = ggml_neg(ctx0, h);
    ggml_set_name(hf, "totloss");

    ggml_set_name(model->decode1_bias, "decode1_bias");       
    ggml_set_name(model->decode1_weight, "decode1_weight");       
    ggml_set_name(model->decode2_bias, "decode2_bias");       
    ggml_set_name(model->decode2_weight, "decode2_weight");       
    
    ggml_set_name(model->encode1_bias, "encode1_bias");       
    ggml_set_name(model->encode1_weight, "encode1_weight");       
    
    ggml_set_name(model->mu_bias, "mu_bias");       
    ggml_set_name(model->mu_weight, "mu_weight");       
    
    ggml_set_name(model->logsd_bias, "logsd_bias");       
    ggml_set_name(model->logsd_weight, "logsd_weight");       

    return;
}

struct ggml_cgraph* build_train_graph_batch(struct mnist_vae_model * model,                                          
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

    // struct ggml_cgraph* gf = ggml_new_graph(model->ctx);
    struct ggml_cgraph* gf  = ggml_new_graph_custom(model->ctx, GGML_DEFAULT_GRAPH_SIZE, true);

    train_forward_batch(model, model->ctx, n_batch);
    
    ggml_build_forward_expand(gf, ggml_get_tensor(model->ctx, "totloss"));

    ggml_free(ctx0);

    return gf;
}

static void sample_forward_batch(
    struct mnist_vae_model   * model,
    struct ggml_context      * ctx0,
    const  int32_t             n_batch
) {
  
    struct ggml_tensor* h = ggml_mul_mat(ctx0, model->decode2_weight, model->noise); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode2_bias, h)); 
    h = ggml_relu(ctx0, h);
    h = ggml_mul_mat(ctx0, model->decode1_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode1_bias, h));    
    struct ggml_tensor* recon = ggml_sigmoid(ctx0, h);
    ggml_set_name(recon, "sample");
    return;

}

struct ggml_cgraph* build_sample_graph_batch(struct mnist_vae_model * model, 
                                             struct ggml_context* ctx0,                                         
                                             int32_t n_batch) {
    
    struct ggml_cgraph* gf  = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, false);

    sample_forward_batch(model, ctx0, n_batch);
    
    ggml_build_forward_expand(gf, ggml_get_tensor(ctx0, "sample"));

    return gf;
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

static void check_data_buffer(struct ggml_cgraph* gf){

    std::map<void *, struct ggml_tensor*> gf_map;  
    std::map<void *, struct ggml_tensor*>::iterator it;
    for(int i = 0; i < gf->n_nodes; ++i){
        struct ggml_tensor *node = gf->nodes[i];
        // printf("%d, checking %s (%s) \n", i, node->name, ggml_op_desc(node));
        it = gf_map.find((void *)node->data);
        if (it != gf_map.end()){ 
            printf( "%s 's data addr already allocated for %s \n", node->name, gf_map[(void *)node->data]->name);
        }
        gf_map[(void *)node->data] = node;
    } 
}

static void check_op_suppport(struct mnist_vae_model * model, struct ggml_cgraph* gf){

    for(int i = 0; i < gf->n_nodes; ++i){
        struct ggml_tensor *node = gf->nodes[i];
        if(!ggml_backend_supports_op(model->backend, node)){
            fprintf(stderr, "%s: node %s 's op (%s) is not supported by the backend\n",
            __func__, node->name, ggml_op_desc(node));
        }        

    } 
}


static void check_backend(struct mnist_vae_model * model, struct ggml_cgraph* gf){

    for(int i = 0; i < gf->n_nodes; ++i){
        struct ggml_tensor *node = gf->nodes[i];
        if(node->backend == GGML_BACKEND_CPU){
            fprintf(stderr, "%s: node %s is on CPU\n", __func__, node->name);
        }  
        else if (node->backend == GGML_BACKEND_GPU){
            fprintf(stderr, "%s: node %s is on GPU\n", __func__, node->name);
        }
        else{
            fprintf(stderr, "%s: node %s is on Others\n", __func__, node->name);
        }

    }
    for(int i = 0; i < gf->n_leafs; ++i){
        struct ggml_tensor *node = gf->leafs[i];
        if(node->backend == GGML_BACKEND_CPU){
            fprintf(stderr, "%s: leaf %s is on CPU\n", __func__, node->name);
        }  
        else if (node->backend == GGML_BACKEND_GPU){
            fprintf(stderr, "%s: leaf %s is on GPU\n", __func__, node->name);
        }
        else{
            fprintf(stderr, "%s: leaf %s is on Others\n", __func__, node->name);
        }
    } 
}


static void ggml_opt_set_grad_to_one( struct ggml_tensor *f ){
    GGML_ASSERT(ggml_is_scalar(f));
    if(f->backend != GGML_BACKEND_CPU){
        float one = 1.f;
        ggml_backend_tensor_set(f->grad, &one, 0, ggml_nbytes(f->grad));
    }else{
        ggml_set_f32      (f->grad, 1.0f);
    }

}

static bool output_images(const std::string &filename, const float *data, int nr, int nc){    
    int n = nr * nc;
    uint8 *pixels = new uint8[28*28*n];
    for (int i = 0; i < nr; i++){ 
        for (int j = 0; j < nc; j++) {
            for (int row = 0; row < 28; row++){
                for (int col = 0; col < 28; col++){
                    int idx = (i*nc+j)*28*28+row*28 + col;
                    int idx0 = (i*28+row)*28*nc+ j*28+col; 
                    // if(i == 0 && j == 1)
                    //     printf("accessing %d, %d, %d, %d - %d : %d \n", i, j, row, col, idx0, idx);
                    pixels[idx0] = float2pixel(data[idx]);
                }
            }
        }
    }
    if (!stbi_write_png(filename.c_str(), 28*nc, 28*nr, 1, pixels, 28*nc)) {
        printf("%s: failed to write mask %s\n", __func__, filename.c_str());
        return false;
    }

    delete pixels;
    printf("%s: image %s was written\n", __func__, filename.c_str());
    return true;
}


#define COUNT_TRAIN 60000ll
#define COUNT_TEST  10000ll

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

int read_test_data(uint8 *data, const unsigned long count)
{
    FILE *fp_image = fopen("models/mnist/t10k-images.idx3-ubyte", "rb");
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

static int log_interval = 0;
static int64_t counter = 0;
static int num_batches = 0;
static int64_t *indices = NULL;
static float *rnds = NULL;
struct mnist_vae_model model;  
struct ggml_tensor * input_batch = NULL;
struct ggml_tensor * noise_batch = NULL;

static double time_us = 0.f;



struct random_normal_distribution * rnd = NULL;


void loss_print(int iter, float loss){
    int n_bat = COUNT_TRAIN / num_batches;
    if((iter-1) % log_interval == 0)
        printf("Epoch: %d [ %d/%ld (%05.2f%%)], TOTAL Loss: %f \n",  iter/num_batches+1,
                        ((iter-1) % num_batches)*n_bat, COUNT_TRAIN,
                    100. * ((iter-1) % num_batches)/num_batches,
                    loss/(float)n_bat);

}

void opt_callback(void * data, int accum_step, float * sched, bool * cancel){
    int n_bat = COUNT_TRAIN / num_batches;
    int cnt = counter % num_batches;
    // const int64_t t_start_us = ggml_time_us();
    if(counter % num_batches == 0){
        for(int i = 0; i < num_batches; i++){
           indices[i] = i*n_bat*28*28;
        }   
        for(int i = 0; i < num_batches; i++){
            std::random_device     rand_dev;
            std::mt19937   generator(rand_dev());
            std::uniform_int_distribution<int>  distr(i, num_batches-1);
            int j = distr(generator);
            int64_t tmp = indices[i];
            indices[i] =  indices[j];
            indices[j] =  tmp;
        }
    }
    
    for(int i = 0; i < model.hparams.n_latent*n_bat; i++){            
        rnds[i] = frand_normal(rnd);
    }

        // load input and noise data  
    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
            || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(input_batch->data, (char *)data+indices[cnt], ggml_nbytes(input_batch));
        memcpy(noise_batch->data, rnds, ggml_nbytes(noise_batch));
    } else {
        ggml_backend_tensor_set(input_batch, (char *)data+indices[cnt], 0, ggml_nbytes(input_batch));  // cuda requires copy the data directly to device
        ggml_backend_tensor_set(noise_batch, rnds, 0, ggml_nbytes(noise_batch));  // cuda requires copy the data directly to device
        // ggml_backend_tensor_set_async(model.backend, input_batch, (char *)data+indices[cnt], 0, ggml_nbytes(input_batch));  // cuda requires copy the data directly to device
        // ggml_backend_tensor_set_async(model.backend,  noise_batch, rnds, 0, ggml_nbytes(noise_batch));  // cuda requires copy the data directly to device
    }
    const int64_t t_feed_us = ggml_time_us() ;
    // time_us = t_feed_us - t_start_us;
    // if (counter % 10 == 0){
    //   printf("  counter:  %d  %8.2f \n", counter, time_us/10);
    //   time_us = 0.f;
    // }
    counter++;

}


void test_callback(int cnt, void *cc_data) {

    if (cnt > 1 and cnt % num_batches == 0){
        int epoch = cnt / num_batches;
        struct mnist_vae_model *m = (struct mnist_vae_model *)cc_data;
        int n_bat = COUNT_TRAIN / num_batches;
        {
            struct ggml_init_params sparams {
                // /*.mem_size   =*/ ggml_tensor_overhead() * (num_tensors + 2),
                /*.mem_size   =*/ ggml_tensor_overhead() * 1024 + ggml_graph_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            struct ggml_context * ctxs = ggml_init(sparams);
            struct ggml_cgraph* gs = build_sample_graph_batch(m, ctxs, n_bat); 
            ggml_backend_buffer_t sample_buffer = ggml_backend_alloc_ctx_tensors(ctxs, model.backend);

            for(int i = 0; i < m->hparams.n_latent*n_bat; i++){            
                rnds[i] = frand_normal(rnd);         
            }
            struct ggml_tensor * noise_batch = ggml_get_tensor(m->ctx, "noise");
            // load noise data  
            if(ggml_backend_is_cpu(model.backend)
    #ifdef GGML_USE_METAL
                    || ggml_backend_is_metal(model.backend)
    #endif
            ) {
                memcpy(noise_batch->data, rnds, ggml_nbytes(noise_batch));
            } else {
                ggml_backend_tensor_set(noise_batch, rnds, 0, ggml_nbytes(noise_batch));  // cuda requires copy the data directly to device
            } 

            if (ggml_backend_is_cpu(m->backend)) {
                ggml_backend_cpu_set_n_threads(m->backend, 1);
            }
            GGML_ASSERT(gs != NULL);
            ggml_backend_graph_compute(m->backend, gs);
            struct ggml_tensor * sample = get_tensor_from_graph(gs, "sample");
            
            float* out_data = new float[ggml_nelements(sample)];    
            ggml_backend_tensor_get(sample, out_data, 0, ggml_nbytes(sample));

            std::string filename = "mnist-sample-epoch_" + std::to_string(epoch) + ".png";
            output_images(filename, out_data, 4, 16);
            ggml_graph_clear(gs);
            ggml_backend_buffer_free(sample_buffer);
            ggml_free(ctxs);   
            delete out_data;
        }
    }


}


struct run_params {

    int n_epoch;
    int n_batch;

    // float f_norm_rms_eps;
    // float rope_freq_base;
    // float rope_freq_scale;
};


static struct run_params get_default_run_params() {
    struct run_params params;
    params.n_epoch    =  10;
    params.n_batch    =  64;

    return params;
}

static void run_print_usage(int argc, char ** argv, const struct run_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                 show this help message and exit\n");
    fprintf(stderr, "  --epochs N                 number of epochs to train (default %d)\n", params->n_epoch);
    fprintf(stderr, "  --batch_size N             number of epochs to train (default %d)\n", params->n_batch);
}



static bool run_params_parse(int argc, char ** argv, struct run_params * params) {
    bool invalid_param = false;
    std::string arg;
    struct run_params default_params = get_default_run_params();
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "--epochs") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_epoch = std::stoi(argv[i]);
        }else if(arg == "--batch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_batch = std::stoi(argv[i]);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            run_print_usage(argc, argv, &default_params);
            exit(1);
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        run_print_usage(argc, argv, &default_params);
        exit(1);
    }

    return true;
}


int main(int argc, char ** argv) {
    

    bool use_gpu = true;
    // int n_batch = 64;
    int n_threads = 1;
    log_interval = 10;

    struct run_params rparams = get_default_run_params();

    if (!run_params_parse(argc, argv, &rparams)) {
        return 1;
    }

    int n_batch = rparams.n_batch;
    


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

   
    


    
   
    uint8 *train_data = (uint8 *)malloc(COUNT_TRAIN * 28 * 28 * sizeof(uint8));
    float *digit = (float *)malloc(COUNT_TRAIN * 28 * 28 * sizeof(float));

    if(read_data(train_data, COUNT_TRAIN) > 0){
        fprintf(stderr, "Error in read in Mnist training data! \n");
        exit(1);
    }
    
    uint8_t *buf = train_data;

    for (int ib = 0; ib < COUNT_TRAIN; ib++) 
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                digit[ib*28*28 + row*28 + col] = ((float)buf[ib*28*28 + row*28 + col])/255.f;
            }
        }

    free(train_data);


    num_batches = COUNT_TRAIN / n_batch;

    rnds = new float[model.hparams.n_latent*n_batch];

    std::vector<uint8_t> work_buffer;

    size_t    compute_size = 1024ll*1024ll*1024ll;
    uint8_t * compute_addr = new uint8_t[compute_size];
    struct ggml_init_params optparams = {
            /*.mem_size   =*/ compute_size,
            /*.mem_buffer =*/ compute_addr,
            /*.no_alloc   =*/ false,
        };

    

    struct ggml_cgraph* gf = NULL; 
    struct ggml_cgraph* gb = NULL; 

    
    gf = build_train_graph_batch(&model, n_batch);
    gb = ggml_graph_dup(model.ctx, gf);           
    printf("build backward graph \n");
    // ggml_build_backward_expand(model.ctx, gf, gb, true);
    ggml_build_backward_expand(model.ctx, gf, gb, false);
    printf("finished build backward graph \n");
    // ggml_graph_dump_dot(gf, NULL, "mnist-vae-forward.dot");
    
    // printf("graph dumped \n");
    model.compute_buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    // if(ggml_backend_is_cpu(model.backend))
    //     check_data_buffer(gf);
    // check_backend(&model, gf);
    // ggml_graph_dump_dot(gb, gf,  "mnist-vae-cpu-backward.dot");    
    randomize_model(&model, 1337, 0.0f, 0.1f, -1.0f, +1.0f);
    // randomize_model(&model, 1337, 0.0f, .1f, -FLT_MAX, FLT_MAX);
    printf("modle initialzed with random numbers \n");
    zero_bias_model(&model, 1337);
    printf("modle bias zeroed \n");            

    indices = new int64_t[num_batches];

    
    rnd = init_random_normal_distribution(1337, 0, 1.f, -FLT_MAX, FLT_MAX);     


                
    struct ggml_context * ctx0 = ggml_init(optparams);
    input_batch = ggml_get_tensor(model.ctx, "input");
    noise_batch = ggml_get_tensor(model.ctx, "noise");
    static struct ggml_tensor * err_tot = ggml_get_tensor(model.ctx, "totloss");
        
   
        

    struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_ADAM);
    // struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_LBFGS);
    opt_params.print_backward_graph = false;
    opt_params.print_forward_graph = false;
    opt_params.adam.n_iter = rparams.n_epoch * num_batches;
    // opt_params.adam.gclip = 1.f;
    opt_params.max_no_improvement = 0;
    // opt_params.lbfgs.n_iter = 16;
    // opt_params.lbfgs.max_linesearch = 50;
    opt_params.gf = gf;
    opt_params.gb = gb;
    opt_params.customer_callback  = test_callback;
    opt_params.customer_data = (void *)&model;
    opt_params.log_callback = loss_print;
    
  
    int ret = ggml_opt(ctx0, opt_params, err_tot, opt_callback, digit);
    ggml_graph_print(gb);



    
    ggml_free(ctx0);

    delete indices;
    delete rnds;
    free(digit);
    // free(test_data);
    ggml_free(model.ctx);
    free_random_normal_distribution(rnd);

    return 0;
}
