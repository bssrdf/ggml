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

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

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
    int32_t n_hidden  = 500;
    int32_t n_latent  = 30;
    int32_t n_batch   = 100;
    int32_t enc1_out  = 1000;
    int32_t enc2_out  = 500;
    int32_t enc3_out  = 250;
    int32_t dec4_out  = 250;
    int32_t dec3_out  = 500;
    int32_t dec2_out  = 1000;
     

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



static void randomize_model(struct mnist_vae_model * model, int seed, float mean, float std, float min, float max) {
    const auto & hparams = model->hparams;
    

    struct random_normal_distribution * rnd = init_random_normal_distribution(seed, mean, std, min, max);

    randomize_tensor_normal(model->encode1_weight, rnd);
    randomize_tensor_normal(model->encode2_weight, rnd);
    randomize_tensor_normal(model->encode3_weight, rnd);
    randomize_tensor_normal(model->decode1_weight, rnd);
    randomize_tensor_normal(model->decode2_weight, rnd);
    randomize_tensor_normal(model->decode3_weight, rnd);
    randomize_tensor_normal(model->decode4_weight, rnd);
    randomize_tensor_normal(model->logsd_weight, rnd);
    randomize_tensor_normal(model->mu_weight, rnd);
    free_random_normal_distribution(rnd);
}



static struct ggml_tensor * train_forward_batch(
    struct mnist_vae_model   * model,
    struct ggml_context      * ctx0,
    struct ggml_cgraph       * gf,
    struct ggml_tensor       * input,
    struct ggml_tensor       * noise,    
    const  int32_t             n_batch
) {
    // const int N = n_input;

    
    // struct ggml_tensor * tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N*n_batch);
    
    

    struct ggml_tensor* h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->encode1_weight, input), 
                                           model->encode1_bias); 
    h = ggml_relu_inplace(ctx0, h); 
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->encode2_weight, h), model->encode2_bias);
    h = ggml_relu_inplace(ctx0, h); 
    h = ggml_add(ctx0, ggml_mul_mat(ctx0, model->encode3_weight, h), model->encode3_bias);
    h = ggml_relu_inplace(ctx0, h); 
    struct ggml_tensor* h1 = ggml_add(ctx0, ggml_mul_mat(ctx0, model->mu_weight, h), model->mu_bias);
    struct ggml_tensor* logsd = ggml_add(ctx0, ggml_mul_mat(ctx0, model->logsd_weight, h), model->logsd_bias);

    h1 = ggml_sqr_inplace(ctx0, h1);

    struct ggml_tensor* sd = ggml_elu() 



    

    // run the computation
    ggml_build_forward_expand(gf, inpL);

    return inpL;
}

struct ggml_cgraph* build_train_graph_batch(struct mnist_vae_model * model, 
                                            struct ggml_tensor* z, 
                                            struct ggml_tensor* z0,
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

        struct ggml_tensor* out =  train_forward_batch(model, ctx0, gf, z_, z0_, n_batch);

        ggml_build_forward_expand(gf, out);
        ggml_free(ctx0);

        return gf;
    }







static void print_row(struct ggml_tensor * probs, int i) {
    for (int k = 0; k < probs->ne[0]; ++k) {
        float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
        printf(" %.2f", p);
    }
    printf("\n");
}

static void print_matrix(struct ggml_tensor * probs) {
    assert(ggml_is_matrix(probs));
    for (int i = 0; i < probs->ne[1]; ++i) {
        for (int k = 0; k < probs->ne[0]; ++k) {
            float p = ggml_get_f32_1d(probs, i*probs->ne[0] + k);
            printf(" %.2f", p);
        }
        printf("\n");
    }
}

static void print_token(int token, int n_vocab) {
    for (int k = 0; k < token; ++k) {
        printf(" ");
    }
    printf("X");
    for (int k = token+1; k < n_vocab; ++k) {
        printf(" ");
    }
    printf("\n");
}

static void print_tokens(struct ggml_tensor * tokens, int n_vocab) {
    for (int i=0; i<tokens->ne[0]; ++i) {
        int token = ggml_get_i32_1d(tokens, i);
        print_token(token, n_vocab);
    }
}



static void get_example_targets_batch(
    struct ggml_context * ctx, int example_id, struct ggml_tensor * tokens_input, struct ggml_tensor * targets
) {
    GGML_ASSERT(ggml_is_matrix(tokens_input));
    GGML_ASSERT(ggml_is_3d(targets));
    int n_tokens = tokens_input->ne[0];
    int n_batch  = tokens_input->ne[1];
    GGML_ASSERT(n_tokens == targets->ne[1]);
    GGML_ASSERT(n_batch  == targets->ne[2]);

    for (int k=0; k<n_batch; ++k) {
        struct ggml_tensor * tokens_input_k = ggml_view_1d(ctx,
                                                tokens_input,
                                                tokens_input->ne[0],
                                                k*tokens_input->nb[1]);
        struct ggml_tensor * targets_k    = ggml_view_2d(ctx,
                                                targets,
                                                targets->ne[0],
                                                targets->ne[1],
                                                targets->nb[1],
                                                k*targets->nb[2]);
        
    }
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

int main(int argc, char ** argv) {
    if (argc < 1) {
        fprintf(stderr, "usage: %s\n", argv[0]);

        return 1;
    }

    struct mnist_vae_model model;    

    bool use_gpu = true;
    
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
        exit(0);
    }
    printf("init model\n");

    model.params_buffer = ggml_backend_alloc_buffer(model.backend, model.params_buffer_size);


    

    
    
    
    init_model(&model);
    

    int n_batch = 8;
    
    // struct ggml_allocr * compute_allocr = NULL;
    if(model.compute_buffer_size == 0)
    // allocate the compute buffer
    {
        model.compute_allocr = ggml_allocr_new_measure_from_backend(model.backend);

        // create the worst case graph for memory usage estimation
        struct ggml_tensor * z = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.hparams.n_input);
        struct ggml_cgraph * gf = build_graph(model, z);

        // compute the required memory
        model.compute_buffer_size = ggml_allocr_alloc_graph(model.compute_allocr, gf) + 1024 * 1024;
        // recreate the allocator with the required memory
        ggml_allocr_free(model.compute_allocr);

        model.compute_buffer = ggml_backend_alloc_buffer(model.backend, model.compute_buffer_size);
        model.compute_allocr = ggml_allocr_new_from_buffer(model.compute_buffer);

        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, model.compute_buffer_size/1024.0/1024.0);
    }

    

    set_param_model(&model);

    randomize_model(&model, 1337, 0.0f, 1.0f, -1.0f, +1.0f);


    int n_examples = 256;
   

    std::vector<uint8_t> work_buffer;

    for (int ex=0; ex<n_examples; ++ex) {
        struct ggml_init_params params = {
            /*.mem_size   =*/ compute_size,
            /*.mem_buffer =*/ compute_addr,
            /*.no_alloc   =*/ false,
        };

        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * after_opt_best_samples  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * after_opt_probs         = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab, n_tokens, n_batch);
        struct ggml_tensor * tokens_input            = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_batch);
        struct ggml_tensor * targets                 = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_vocab, n_tokens, n_batch);

        int n_past = 0;

        ggml_cgraph gf = {};

        get_example_targets_batch(ctx0, 64*ex+0,  tokens_input, targets);

        struct ggml_tensor * logits = forward_batch(&model, &kv_self, ctx0, &gf, tokens_input, n_tokens, n_past, n_batch);
        // struct ggml_tensor * e = cross_entropy_loss(ctx0, targets, logits);
        struct ggml_tensor * e = square_error_loss(ctx0, targets, logits);

        ggml_build_forward_expand(&gf, e);
        ggml_graph_compute_helper(work_buffer, &gf, /*n_threads*/ 1);

        float error_before_opt = ggml_get_f32_1d(e, 0);

        struct ggml_opt_params opt_params_lbfgs = ggml_opt_default_params(GGML_OPT_LBFGS);
        opt_params_lbfgs.print_forward_graph = false;
        opt_params_lbfgs.print_backward_graph = false;
        opt_params_lbfgs.lbfgs.n_iter = 16;
        ggml_opt(ctx0, opt_params_lbfgs, e);
        //
        ggml_build_forward_expand(&gf, e);
        ggml_graph_compute_helper(work_buffer, &gf, /*n_threads*/ 1);

        float error_after_opt = ggml_get_f32_1d(e, 0);

        if (ex % 8 == 0) {
            printf("Example %d\n", (ex+1));
            printf("error_before_opt: %.2f\n", error_before_opt);
            printf("error_after_opt:  %.2f\n", error_after_opt);
        }

        if (ex % 64 == 0) {
            sample_softmax_batch(ctx0, logits, after_opt_probs, after_opt_best_samples);
            // printf("probabilities after optimization:\n");
            // print_matrix(after_opt_probs);
            printf("best samples after optimization:\n");
            print_tokens(after_opt_best_samples, n_vocab);
        }

        ggml_free(ctx0);
    }

    {
        int n_gen = 128;
        int sample_ctx = n_tokens-n_tokens/8;

        printf("Generating %d tokens.\n", n_gen);

        struct ggml_tensor * tokens_input = ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, n_tokens);
        struct ggml_tensor * targets      = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, n_vocab, n_tokens);

        get_example_targets(137, tokens_input, targets);
        for (int i=sample_ctx; i<n_tokens; ++i) {
            ggml_set_i32_1d(tokens_input, i, n_vocab/2);
        }

        for (int i=0; i<sample_ctx-1; ++i) {
            print_token(ggml_get_i32_1d(tokens_input, i), n_vocab);
        }
        printf("---\n");
        for (int i=0; i<n_gen; ++i) {
            struct ggml_init_params params = {
                /*.mem_size   =*/ compute_size,
                /*.mem_buffer =*/ compute_addr,
                /*.no_alloc   =*/ false,
            };
            struct ggml_context * ctx0 = ggml_init(params);

            ggml_cgraph gf = {};

            int n_past = 0;
            struct ggml_tensor * logits = forward(&model, &kv_self, ctx0, &gf, tokens_input, sample_ctx, n_past);

            ggml_build_forward_expand(&gf, logits);
            ggml_graph_compute_helper(work_buffer, &gf, /*n_threads*/ 1);

            struct ggml_tensor * best_samples = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, sample_ctx);
            struct ggml_tensor * probs        = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, sample_ctx);

            sample_softmax(logits, probs, best_samples);

            // int sample_at = n_tokens-1;
            int token = ggml_get_i32_1d(best_samples, sample_ctx-1);

            // print_row(probs, sample_at);
            print_token(token, n_vocab);

            lshift_examples(tokens_input, targets, 1);
            ggml_set_i32_1d(tokens_input, 0, 0);
            ggml_set_i32_1d(tokens_input, sample_ctx-1, token);

            ggml_free(ctx0);
        }
    }

    print_matrix(model.tok_embeddings);
    printf("done\n");

    // ggml_free(kv_self.ctx);
    // ggml_free(model_lora.ctx);
    ggml_free(model.ctx);

    return 0;
}
