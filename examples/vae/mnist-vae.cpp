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

// default hparams
struct mnist_hparams {
    int32_t n_input   = 784;
    // int32_t n_latent  = 30;
    // int32_t n_batch   = 100;
    // int32_t enc1_out  = 1000;
    // int32_t enc2_out  = 500;
    // int32_t enc3_out  = 250;
    // int32_t dec4_out  = 250;
    // int32_t dec3_out  = 500;
    // int32_t dec2_out  = 1000;

    int32_t n_latent  = 5;
    int32_t n_batch   = 100;
    int32_t enc1_out  = 100;
    int32_t enc2_out  = 50;
    int32_t enc3_out  = 25;
    int32_t dec4_out  = 25;
    int32_t dec3_out  = 50;
    int32_t dec2_out  = 100;
     

};

struct mnist_vae_model {
    mnist_hparams hparams;

    // struct ggml_tensor * input;
    // struct ggml_tensor * noise;

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


static void init_model(struct mnist_vae_model * model) {
    const auto & hparams = model->hparams;

    const int32_t n_input   = hparams.n_input;
    const int32_t n_latent  = hparams.n_latent;    
    const int32_t enc1_out  = hparams.enc1_out;
    const int32_t enc2_out  = hparams.enc2_out;
    const int32_t enc3_out  = hparams.enc3_out;
    const int32_t dec2_out  = hparams.dec2_out;
    const int32_t dec3_out  = hparams.dec3_out;
    const int32_t dec4_out  = hparams.dec4_out;

    struct ggml_context * ctx = model->ctx;

    model->encode1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_input, enc1_out); 
    model->encode1_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, enc1_out);          
    model->encode2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc1_out, enc2_out); 
    model->encode2_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, enc2_out);          
    model->encode3_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc2_out, enc3_out); 
    model->encode3_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, enc3_out);          

    model->logsd_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc3_out, n_latent); 
    model->logsd_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_latent);          

    model->mu_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, enc3_out, n_latent); 
    model->mu_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_latent);          

    model->decode4_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_latent, dec4_out); 
    model->decode4_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec4_out);          
    model->decode3_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dec4_out, dec3_out); 
    model->decode3_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec3_out);          
    model->decode2_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dec3_out, dec2_out); 
    model->decode2_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dec2_out);          

    model->decode1_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dec2_out, n_input); 
    model->decode1_bias   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_input);          
    
}

static void set_param_model(struct mnist_vae_model * model) {
    const auto& hparams = model->hparams;

    // const uint32_t n_layer = hparams.n_layer;

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

static void zero_bias_model(struct mnist_vae_model * model){
    if(ggml_backend_is_cpu(model->backend)) {
        ggml_set_zero(model->decode1_bias);
        ggml_set_zero(model->decode2_bias);
        ggml_set_zero(model->decode3_bias);
        ggml_set_zero(model->decode4_bias);
        ggml_set_zero(model->encode1_bias);
        ggml_set_zero(model->encode2_bias);
        ggml_set_zero(model->encode3_bias);
        ggml_set_zero(model->mu_bias);
        ggml_set_zero(model->logsd_bias);
    }
    else{
        struct ggml_init_params params = {
                    .mem_size   = 128*1024*1024,
                    .mem_buffer = NULL,
                    .no_alloc   = false,
                    };

        struct ggml_context * ctx0 = ggml_init(params);
        struct ggml_tensor * rn = NULL;
        // fprintf(stderr, "%s: before dup model tensor \n", __func__); 
        rn = ggml_dup_tensor(ctx0, model->encode1_bias);
        rn = ggml_set_zero(rn);
        ggml_backend_tensor_set(model->encode1_bias, rn->data, 0, ggml_nbytes(model->encode1_bias));
        rn = ggml_dup_tensor(ctx0, model->encode2_bias);
        rn = ggml_set_zero(rn);
        ggml_backend_tensor_set(model->encode2_bias, rn->data, 0, ggml_nbytes(model->encode2_bias));
        rn = ggml_dup_tensor(ctx0, model->encode3_bias);
        rn = ggml_set_zero(rn);
        ggml_backend_tensor_set(model->encode3_bias, rn->data, 0, ggml_nbytes(model->encode3_bias));
        rn = ggml_dup_tensor(ctx0, model->decode1_bias);
        rn = ggml_set_zero(rn);
        ggml_backend_tensor_set(model->decode1_bias, rn->data, 0, ggml_nbytes(model->decode1_bias));
        rn = ggml_dup_tensor(ctx0, model->decode2_bias);
        rn = ggml_set_zero(rn);
        ggml_backend_tensor_set(model->decode2_bias, rn->data, 0, ggml_nbytes(model->decode2_bias));
        rn = ggml_dup_tensor(ctx0, model->decode3_bias);
        rn = ggml_set_zero(rn);
        ggml_backend_tensor_set(model->decode3_bias, rn->data, 0, ggml_nbytes(model->decode3_bias));
        rn = ggml_dup_tensor(ctx0, model->decode4_bias);
        rn = ggml_set_zero(rn);
        ggml_backend_tensor_set(model->decode4_bias, rn->data, 0, ggml_nbytes(model->decode4_bias));
        rn = ggml_dup_tensor(ctx0, model->mu_bias);
        rn = ggml_set_zero(rn);
        ggml_backend_tensor_set(model->mu_bias, rn->data, 0, ggml_nbytes(model->mu_bias));
        rn = ggml_dup_tensor(ctx0, model->logsd_bias);
        rn = ggml_set_zero(rn);
        ggml_backend_tensor_set(model->logsd_bias, rn->data, 0, ggml_nbytes(model->logsd_bias));
        ggml_free(ctx0);
    }

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

static void load_data(ggml_backend_t backend, struct ggml_tensor * dst, struct ggml_tensor * src){
    if(ggml_backend_is_cpu(backend)) {
        memcpy(dst->data, src->data, ggml_nbytes(dst));
    } else {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(dst));
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
    fprintf(stderr, "%s: before dup model tensor \n", __func__); 
    rn = ggml_dup_tensor(ctx0, model->encode1_weight);
    fprintf(stderr, "%s: after dup model tensor \n", __func__); 
    int ndim = ggml_n_dims(rn);
    if (rn->backend == GGML_BACKEND_GPU) 
        fprintf(stderr, "%s: rn is  on GPU \n", __func__); 
    else
        fprintf(stderr, "%s: rn is  on CPU with %d dims\n", __func__, ndim); 
    ndim = ggml_n_dims(model->encode1_weight);
    if (model->encode1_weight->backend == GGML_BACKEND_GPU) 
        fprintf(stderr, "%s: model weight is  on GPU \n", __func__); 
    else{
        int64_t *ne = model->encode1_weight->ne;
        fprintf(stderr, "%s: model weight is  on CPU with (%d, %d, %d, %d) dims \n", 
             __func__, ne[0], ne[1], ne[2], ne[3]);     
    }
    rn = randomize_tensor_normal(rn, rnd);
    // print_matrix(rn);
    fprintf(stderr, "%s: after randmize tensor \n", __func__); 
    load_data(model->backend, model->encode1_weight, rn);
    fprintf(stderr, "%s: after load into model \n", __func__); 

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

    // ggml_print_objects(ctx0);

    ggml_free(ctx0);
}



static void train_forward_batch(
    struct mnist_vae_model   * model,
    struct ggml_context      * ctx0,
    struct ggml_cgraph       * gf,
    struct ggml_tensor       * input,
    struct ggml_tensor       * noise,    
    const  int32_t             n_batch
) {

    ggml_set_name(input, "input");
    struct ggml_tensor* h = ggml_mul_mat(ctx0, model->encode1_weight, ggml_cont(ctx0, input)); 
    ggml_set_name(h, "encode1_w");

    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->encode1_bias, h)); 
    ggml_set_name(h, "encode1_b");
    
    h = ggml_relu_inplace(ctx0, h); 
    ggml_set_name(h, "encode1_relu");
    // fprintf(stderr, "%s: done with first build\n", __func__);       

    // fprintf(stderr, "%s: build 2 %d\n", __func__, __LINE__);       
    h = ggml_mul_mat(ctx0, model->encode2_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->encode2_bias, h)); 
    ggml_set_name(h, "encode2_b");
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->encode2_weight, h), ggml_repeat(ctx0, model->encode2_bias, );
    h = ggml_relu_inplace(ctx0, h); 
    // fprintf(stderr, "%s: build 3 %d\n", __func__, __LINE__);       
    h = ggml_mul_mat(ctx0, model->encode3_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->encode3_bias, h)); 
    ggml_set_name(h, "encode3_b");
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->encode3_weight, h), model->encode3_bias);
    h = ggml_relu_inplace(ctx0, h); 
        

    struct ggml_tensor* h1 = ggml_mul_mat(ctx0, model->mu_weight, h);
    h1 = ggml_add(ctx0, h1, ggml_repeat(ctx0, model->mu_bias, h1)); 
    struct ggml_tensor* mu = h1;

    struct ggml_tensor* logsd =  ggml_mul_mat(ctx0, model->logsd_weight, h);
    logsd = ggml_add(ctx0, logsd, ggml_repeat(ctx0, model->logsd_bias, logsd)); 
    ggml_set_name(logsd, "logsd");

    h1 = ggml_sqr_inplace(ctx0, h1);
    ggml_set_name(h1, "meansq");
    struct ggml_tensor* sd = ggml_exp(ctx0, logsd);
    ggml_set_name(sd, "sd");
    struct ggml_tensor* var = ggml_sqr(ctx0, sd);
    ggml_set_name(var, "var");
    struct ggml_tensor* h3 = ggml_add(ctx0, ggml_scale_inplace(ctx0, h1, 0.5f), 
                                            ggml_scale_inplace(ctx0, var, 0.5f));
    h3 = ggml_sub(ctx0, h3, logsd);
    ggml_set_name(h3, "kldiv_plus_half");
    h3 = ggml_add1_inplace(ctx0, h3, ggml_new_f32(ctx0, -0.5f));
    ggml_set_name(h3, "kldiv");    

    h3 = ggml_mul(ctx0, noise, sd);
    ggml_set_name(h3, "sdnoise");
    h3 = ggml_add(ctx0, h3, mu);
    ggml_set_name(h3, "sample");

    h = ggml_mul_mat(ctx0, model->decode4_weight, h3); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode4_bias, h)); 
    h = ggml_relu_inplace(ctx0, h);
    h = ggml_mul_mat(ctx0, model->decode3_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode3_bias, h)); 
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->decode3_weight, h), model->decode3_bias);
    h = ggml_relu_inplace(ctx0, h);
    h = ggml_mul_mat(ctx0, model->decode2_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode2_bias, h)); 
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->decode2_weight, h), model->decode2_bias);
    h = ggml_relu_inplace(ctx0, h);
    h = ggml_mul_mat(ctx0, model->decode1_weight, h); 
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, model->decode1_bias, h)); 
    // h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->decode1_weight, h), model->decode1_bias);
       

    // run the computation
    // ggml_build_forward_expand(gf, inpL);
    // *l2 = h; // 2nd loss is sigmoid cross entropy
    ggml_set_name(h, "for_cren");
    return;
}

struct ggml_cgraph* build_train_graph_batch(struct mnist_vae_model * model, 
                                            struct ggml_tensor* z, 
                                            struct ggml_tensor* z0,
                                            int32_t n_batch
                                           ) {
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

    // struct ggml_context* ctx0 = ggml_init(params);

    struct ggml_context* ctx0 = model->ctx;

    struct ggml_cgraph* gf = ggml_new_graph(ctx0);

    struct ggml_tensor* z_ = NULL;
    struct ggml_tensor* z0_ = NULL;


    // it's performing a compute, check if backend isn't cpu
    if (!ggml_backend_is_cpu(model->backend)) {
        // pass input tensors to gpu memory
        z_ = ggml_dup_tensor(ctx0, z);
        ggml_allocr_alloc(model->compute_allocr, z_);

        z0_ = ggml_dup_tensor(ctx0, z0);
        ggml_allocr_alloc(model->compute_allocr, z0_);

        // pass data to device backend
        if (!ggml_allocr_is_measure(model->compute_allocr)) {
            ggml_backend_tensor_set(z_, z->data, 0, ggml_nbytes(z));
            ggml_backend_tensor_set(z0_, z0->data, 0, ggml_nbytes(z0));
        }
    } else {
        z_ = z;
        z0_ = z0;
    }

    train_forward_batch(model, ctx0, gf, z_, z0_, n_batch);

    // ggml_build_forward_expand(gf, l1);
    // ggml_build_forward_expand(gf, l2);


    // ggml_free(ctx0);

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
 
    if(!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }


    struct ggml_init_params lcparams;
    lcparams.mem_size   = static_cast<size_t>(model.get_num_tensors() * ggml_tensor_overhead()) + 1 * 1024 * 1024;
    lcparams.mem_buffer = NULL;
    lcparams.no_alloc   = true;

    model.params_buffer_size = 10 * 1024 * 1024;  // 10 MB, for padding
    model.params_buffer_size += model.calculate_mem_size();

    model.ctx = ggml_init(lcparams);
    if (!model.ctx) {
        fprintf(stderr, "ggml_init() failed");
        exit(1);
    }
    printf("init model\n");

    model.params_buffer = ggml_backend_alloc_buffer(model.backend, model.params_buffer_size);
    
    printf("alloc params_buffer\n");
    
    init_model(&model);

    printf(" model inited\n");
    
    // if(model.compute_buffer_size == 0)

    // {
    //     model.compute_allocr = ggml_allocr_new_measure_from_backend(model.backend);

    //     // create the worst case graph for memory usage estimation
    //     struct ggml_tensor * z = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_input);
    //     struct ggml_cgraph * gf = build_graph(model, z);

    //     // compute the required memory
    //     model.compute_buffer_size = ggml_allocr_alloc_graph(model.compute_allocr, gf) + 1024 * 1024;
    //     // recreate the allocator with the required memory
    //     ggml_allocr_free(model.compute_allocr);

    //     model.compute_buffer = ggml_backend_alloc_buffer(model.backend, model.compute_buffer_size);
    //     model.compute_allocr = ggml_allocr_new_from_buffer(model.compute_buffer);

    //     fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, model.compute_buffer_size/1024.0/1024.0);
    // }

    // allocate the compute buffer
    model.compute_buffer_size = 2*1024ll*1024ll*1024ll; // 2 GB
    model.compute_buffer = ggml_backend_alloc_buffer(model.backend, model.compute_buffer_size);
    model.compute_allocr = ggml_allocr_new_from_buffer(model.compute_buffer);

    set_param_model(&model);
    printf(" mdoel params set \n");

    


    int n_batch = 100;
   
    uint8 *train_data = (uint8 *)malloc(COUNT_TRAIN * 28 * 28 * sizeof(uint8));

    if(read_data(train_data, COUNT_TRAIN) > 0){
        fprintf(stderr, "Error in read in Mnist training data! \n");
        exit(1);
    }

    uint8_t *buf = train_data;
    std::vector<float> digit;

    digit.resize(n_batch*28*28);

    struct ggml_init_params tmparms = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(tmparms);

    int32_t num_batches = COUNT_TRAIN / n_batch;
    for (int ib = 0; ib < n_batch; ib++) 
        for (int row = 0; row < 28; row++) 
            for (int col = 0; col < 28; col++) 
                digit[ib*28*28 + row*28 + col] = ((float)buf[ib*28*28 + row*28 + col])/255.f;
    struct ggml_tensor * input_batch = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, model.hparams.n_input, n_batch);
    struct ggml_tensor * noise_batch = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, model.hparams.n_latent, n_batch);
    

    
    
    // ggml_allocr_reset(model.compute_allocr);
    fprintf(stderr, "%s: to build first graph \n", __func__);   
    struct ggml_cgraph* gf = build_train_graph_batch(&model, input_batch, noise_batch, n_batch);
    // ggml_allocr_alloc_graph(model.compute_allocr, gf); 
    model.compute_buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    memcpy(input_batch->data, digit.data(), ggml_nbytes(input_batch)); 
    noise_batch = ggml_set_f32(noise_batch, 0.f);
    model.compute_allocr = ggml_allocr_new_from_buffer(model.compute_buffer);

    // ggml_print_objects(model.ctx);

    randomize_model(&model, 1337, 0.0f, 1.f, -1.0f, +1.0f);
    // struct ggml_tensor * h = model.encode1_weight;
    // int64_t *ne = h->ne;
    // printf("h is %s \n", h->name);
    // printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    
    printf("modle initialzed with random numbers \n");
    zero_bias_model(&model);
    // h = model.encode1_weight;
    // ne = h->ne;
    // printf("h is %s \n", h->name);
    // printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    // printf("after zero bias \n"); 


    std::vector<uint8_t> work_buffer;

    
    
    struct ggml_tensor * loss_kl = ggml_get_tensor(model.ctx, "kldiv");

    ggml_build_forward_expand(gf, loss_kl);
    // h = model.encode1_weight;
    // ne = h->ne;
    // printf("h is %s \n", h->name);
    // printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);

    // h = ggml_get_tensor(model.ctx, "input");
    // ne = h->ne;
    // printf("h is %s \n", h->name);
    // printf("(%ld, %ld, %ld, %ld) \n", ne[0], ne[1], ne[2], ne[3]);
    // printf("after forward expand \n");   

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

    // ggml_graph_compute_helper(work_buffer, gf, /*n_threads*/ 1);
    ggml_backend_graph_compute(model.backend, gf);

    ggml_free(ctx0);


    for (int ex=0; ex<num_batches; ++ex) {
        printf(" enter loop %d \n", ex);

        for (int ib = 0; ib < n_batch; ib++) 
            for (int row = 0; row < 28; row++) 
                for (int col = 0; col < 28; col++) 
                    digit[ib*28*28 + row*28 + col] = ((float)buf[ib*28*28 + row*28 + col])/255.f;

        buf += n_batch*28*28;  
        
        struct ggml_context * ctx0 = ggml_init(tmparms);
        
        memcpy(input_batch->data, digit.data(), ggml_nbytes(input_batch)); 
        printf("copy to input \n", ex);
        
        struct ggml_cgraph* gf = build_train_graph_batch(&model, input_batch, noise_batch, n_batch);
        printf(" after gf build \n");
        loss_kl = ggml_get_tensor(model.ctx, "kldiv");
        if (!loss_kl){
            fprintf(stderr, "%s: kl losss not returned properly \n", __func__);
            exit(1);
        }           
      
        struct ggml_tensor * e = ggml_sum(model.ctx, loss_kl);
        struct ggml_tensor * err_kl = ggml_scale(model.ctx, e, 1.f/(float)n_batch);
        // ggml_allocr_alloc(model.compute_allocr, err_kl); 
        printf(" after err_kl build \n");
        ggml_build_forward_expand(gf, err_kl);
        ggml_allocr_alloc_graph(model.compute_allocr, gf); 
        ggml_graph_dump_dot(gf, NULL, "vae-1-forward.dot");
        // ggml_build_forward_expand(gf, loss_sigmoid);
        // ggml_allocr_alloc_graph(model.compute_allocr, gf); 
        // ggml_graph_dump_dot(gf, NULL, "vae-2-forward.dot");
        printf(" compute buffer alloced \n");
        // ggml_graph_compute_helper(work_buffer, gf, /*n_threads*/ 1);
        ggml_backend_graph_compute(model.backend, gf);
        printf(" after compute \n");
        // struct ggml_tensor * t = ggml_get_tensor(model.ctx, "meansq");
        // print_matrix(t);
        // printf("===============================================\n");
        // t = ggml_get_tensor(model.ctx, "var");
        // print_matrix(t);
        // printf("===============================================\n");
        // t = model.encode1_weight;
        // print_matrix(t);
        // printf("===============================================\n");
        // t = ggml_get_tensor(model.ctx, "encode2_b");
        // print_matrix(t);
        // printf("===============================================\n");
        // t = ggml_get_tensor(model.ctx, "encode3_b");
        // print_matrix(t);



        float error_before_opt = ggml_get_f32_1d(e, 0);

        printf(" after compute error is  %f \n", error_before_opt);

        struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_ADAM);
        opt_params.print_backward_graph = false;
        opt_params.print_forward_graph = false;
        opt_params.adam.n_iter = 16;

        ggml_opt(model.ctx, opt_params, err_kl);


        //
        // ggml_build_forward_expand(gf, e);        

        // float error_after_opt = ggml_get_f32_1d(e, 0);

        ggml_free(ctx0);
    }

    printf("done\n");

    // ggml_free(kv_self.ctx);
    // ggml_free(model_lora.ctx);
    ggml_free(model.ctx);

    return 0;
}
