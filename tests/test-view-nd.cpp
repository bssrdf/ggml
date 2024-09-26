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
    struct ggml_tensor * b;
    struct ggml_tensor * c;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;
};

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    int KW = 3, KH = 3, IC = 10;
    int OW = 2, OH = 3, OC = 12;
    // Initialize adata
    float * adata = new float[KW * KH * IC];
    for (int i = 0; i < KW * KH * IC  ; i++) {        
        adata[i] = (float)i;
    }
    // for (int i = KW * KH * (IC / 2); i < KW * KH * IC; i++) {        
    //     adata[i] = (float)i;
    // }

    // for (int i = KW * KH * (IC * 2 / 3); i < KW * KH * IC; i++) {        
    //     adata[i] = 4.5f;
    // }

    float * bdata = new float[OW * OH * OC];
    for (int i = 0; i < OW * OH * OC; i++) {
        bdata[i] = (float)i;
    }

    

    size_t buffer_size = 0;
    {
        buffer_size += KW * KH * IC * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += OW * OH * OC * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += OW * OH * OC * ggml_type_size(GGML_TYPE_F32); // tensor c
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 3;
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
    model.a = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32,  IC, KW, KH);
    model.b = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32,  OW, OH, OC);
    model.c = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32,  OC, OH, OW);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.a);
    ggml_tallocr_alloc(&alloc, model.b);
    ggml_tallocr_alloc(&alloc, model.c);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.a->data, adata, ggml_nbytes(model.a));
    } else {
        ggml_backend_tensor_set(model.a, adata, 0, ggml_nbytes(model.a));
    }

    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.b->data, bdata, ggml_nbytes(model.b));
    } else {
        ggml_backend_tensor_set(model.b, bdata, 0, ggml_nbytes(model.b));
    }

    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.c->data, bdata, ggml_nbytes(model.c));
    } else {
        ggml_backend_tensor_set(model.c, bdata, 0, ggml_nbytes(model.c));
    }
    
}

std::vector<struct ggml_tensor*> chunk_half(struct ggml_context* ctx,
                                struct ggml_tensor* x){
        auto t = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));   
        for(int i=0; i < 4; i++){
           printf("chunk, %d, %d \n", t->ne[i], t->nb[i]);
        }
        int64_t n = t->ne[2] / 2;
        int64_t offset = t->nb[2] * n; 
        auto k = ggml_view_3d(ctx, t, t->ne[0], t->ne[1], n, t->nb[1], t->nb[2], offset*0);
        auto v = ggml_view_3d(ctx, t, t->ne[0], t->ne[1], n, t->nb[1], t->nb[2], offset*1);
        return {ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3)),
                ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3))};  
    }

    std::vector<struct ggml_tensor*> chunk_half1(struct ggml_context* ctx,
                                struct ggml_tensor* x){

        // auto tlo = ggml_view_4d(ctx, x, x->ne[0]/2, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], 0);
        // auto tli = ggml_view_4d(ctx, x, x->ne[0]/2, x->ne[1], x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3], x->nb[0]*x->ne[0]/2);
        auto tlo = ggml_view_4d(ctx, x, x->ne[0]/2, x->ne[1], x->ne[2], x->ne[3], x->nb[1]/2, x->nb[2]/2, x->nb[3]/2, 0);
        auto tli = ggml_view_4d(ctx, x, x->ne[0]/2, x->ne[1], x->ne[2], x->ne[3], x->nb[1]/2, x->nb[2]/2, x->nb[3]/2, x->nb[0]*x->ne[0]/2);
        return {ggml_cont(ctx, tlo),
                ggml_cont(ctx, tli)};  
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
    for(int i=0; i < 4; i++){
        printf("%d, %d \n", a->ne[i], a->nb[i]);
    }
    int64_t offset = a->nb[0]*a->ne[0]/2 ;    
    // struct ggml_tensor* half1 = ggml_view_3d(ctx0, a, a->ne[0]/2, a->ne[1], a->ne[2], 
    //                            a->nb[1], a->nb[2], offset * 0);
    std::vector<struct ggml_tensor*> chunks =  chunk_half(ctx0, a);
    struct ggml_tensor* half1 = chunks[0];
    a = half1;
    for(int i=0; i < 4; i++){
        printf("half1, %d, %d \n", a->ne[i], a->nb[i]);
    }
    ggml_set_name(half1, "half1_res");
    ggml_build_forward_expand(gf, half1);

    // recalculate for avoid fragmentation
    // struct ggml_tensor* half2 = ggml_view_3d(ctx0, a, a->ne[0]/2, a->ne[1], a->ne[2], 
    //                            a->nb[1], a->nb[2], offset * 1);
    struct ggml_tensor* half2 = chunks[1];
    ggml_set_name(half2, "half2_res");
    ggml_build_forward_expand(gf, half2);

    // struct ggml_tensor* half3 = ggml_view_3d(ctx0, a, a->ne[0], a->ne[1], a->ne[2]/3, a->nb[1], a->nb[2], offset * 2);
    // ggml_set_name(half3, "half3_res");
    // ggml_build_forward_expand(gf, half3);

    struct ggml_tensor *b = model.b;
    int heads = 4;
    /*******************
    example of getting a 4D view of a 3D tensor
    ********************/
    struct ggml_tensor* x = ggml_view_4d(ctx0, b, b->ne[0], b->ne[1], heads, b->ne[2]/heads, 
                            b->nb[1], b->nb[2], b->nb[3], 0);
    ggml_set_name(x, "r1_res");
    ggml_build_forward_expand(gf, x);

    struct ggml_tensor *c = model.c;
    struct ggml_tensor* y = ggml_soft_max(ctx0, c);
    ggml_set_name(y, "s1_res");
    ggml_build_forward_expand(gf, y);


    struct ggml_tensor* xdup = ggml_cont(ctx0, x); 
    ggml_set_name(xdup, "x1_res");
    ggml_build_forward_expand(gf, xdup);

    struct ggml_tensor* z = ggml_cont(ctx0, ggml_view_2d(ctx0, c, c->ne[0], c->ne[2], c->nb[2], 0));
    ggml_set_name(z, "z1_res");
    ggml_build_forward_expand(gf, z);

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
    struct ggml_tensor * r1_res = NULL;
    struct ggml_tensor * s1_res = NULL;
    struct ggml_tensor * x1_res = NULL;
    struct ggml_tensor * z1_res = NULL;

    for(int i = 0; i < gf_res->n_nodes; i++) {
        if(strcmp(ggml_get_name(gf_res->nodes[i]), "half1_res") == 0) {
            h1_res = gf_res->nodes[i];
        } else if(strcmp(ggml_get_name(gf_res->nodes[i]), "half2_res") == 0) {
            h2_res = gf_res->nodes[i];
        // } else if(strcmp(ggml_get_name(gf_res->nodes[i]), "half3_res") == 0) {
        //     h3_res = gf_res->nodes[i];
        } else if(strcmp(ggml_get_name(gf_res->nodes[i]), "r1_res") == 0) {
            r1_res = gf_res->nodes[i];
        } else if(strcmp(ggml_get_name(gf_res->nodes[i]), "s1_res") == 0) {
            s1_res = gf_res->nodes[i];
        } else if(strcmp(ggml_get_name(gf_res->nodes[i]), "x1_res") == 0) {
            x1_res = gf_res->nodes[i];
        }   else if(strcmp(ggml_get_name(gf_res->nodes[i]), "z1_res") == 0) {
            z1_res = gf_res->nodes[i];
        }
    }


    float* h1_data = new float[ggml_nelements(h1_res)];
    float* h2_data = new float[ggml_nelements(h2_res)];
    // float* h3_data = new float[ggml_nelements(h3_res)];
    float* r1_data = new float[ggml_nelements(r1_res)];
    float* s1_data = new float[ggml_nelements(s1_res)];
    float* x1_data = new float[ggml_nelements(x1_res)];
    float* z1_data = new float[ggml_nelements(z1_res)];

    ggml_backend_tensor_get(h1_res, h1_data, 0, ggml_nbytes(h1_res));

    ggml_backend_tensor_get(h2_res, h2_data, 0, ggml_nbytes(h2_res));

    // ggml_backend_tensor_get(h3_res, h3_data, 0, ggml_nbytes(h3_res));
    ggml_backend_tensor_get(r1_res, r1_data, 0, ggml_nbytes(r1_res));
    ggml_backend_tensor_get(s1_res, s1_data, 0, ggml_nbytes(s1_res));
    ggml_backend_tensor_get(x1_res, x1_data, 0, ggml_nbytes(x1_res));
    ggml_backend_tensor_get(z1_res, z1_data, 0, ggml_nbytes(z1_res));
    
    printf("==================\n");

    for(int k = 0; k < 3; k++){
        for(int j = 0; j < 3; j++){
            for(int i = 0; i < 5; i++){
               printf("%f, ", h1_data[i+5*j+k*5*3]);
            }
            printf("\n");
        }
        printf("==================\n");
    }

    printf("*************************\n");

    for(int k = 0; k < 3; k++){
        for(int j = 0; j < 3; j++){
            for(int i = 0; i < 5; i++){
               printf("%f, ", h2_data[i+5*j+k*5*3]);
            }
            printf("\n");
        }
        printf("==================\n");
    }

    printf("*************************\n");

    // for(int k = 0; k < 5; k++){
    //     for(int j = 0; j < 3; j++){
    //         for(int i = 0; i < 3; i++){
    //            printf("%f, ", h3_data[i+3*j+k*3*3]);
    //         }
    //         printf("\n");
    //     }
    //     printf("==================\n");
    // }     

    for(int i = 0; i < ggml_nelements(x1_res); i++){
           printf("%f, ", x1_data[i]);
    }  
    printf("\n");

    // for(int l = 0; l < 2; l++){
    //     for(int k = 0; k < 4; k++){
    //         for(int j = 0; j < 3; j++){
    //             for(int i = 0; i < 3; i++){
    //             printf("%f, ", r1_data[i+3*j+k*3*3+l*4*3*3]);
    //             }
    //             printf("\n");
    //         }
    //         printf("==================\n");
    //     }
    //     printf("+++++++++++++++++++++++\n");

    // }


    for(int k = 0; k < 2; k++){
        for(int j = 0; j < 3; j++){
            for(int i = 0; i < 12; i++){
               printf("%f, ", s1_data[i+12*j+k*3*12]);
            }
            printf("\n");
        }
        printf("==================\n");
    }

    printf("+++++++++++++++++++++++\n");
    
        for(int j = 0; j < 2; j++){
            for(int i = 0; i < 12; i++){
               printf("%f, ", z1_data[i+12*j]);
            }
            printf("\n");
        }
    printf("==================\n");

    
    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
