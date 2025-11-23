#include "ggml.h"
#include "ggml-cpu.h"
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
#include <array>
#include <random>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct test_model {
    struct ggml_tensor * q32;
    struct ggml_tensor * k32;
    struct ggml_tensor * v32;
    struct ggml_tensor * q16;
    struct ggml_tensor * k16;
    struct ggml_tensor * v16;

    struct ggml_tensor * m;
    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx;

    int64_t hsk; // K head size
    int64_t hsv; // V head size
    int64_t nh; // num heads
    const std::array<int64_t, 2> nr23{1,1}; // repeat in dim 2 and 3, tests for grouped-query attention
    int64_t kv; // kv size
    int64_t nb; // batch size

    bool mask; // use mask
    bool sinks; // use sinks

    const float max_bias = 0.f; // ALiBi
    const float logit_softcap= 0.f; // Gemma 2

    ggml_prec prec;
    ggml_type type_KV;
    std::array<int32_t, 4> permute;
};


static void init_tensor_kq_mask(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS( int32_t, ne, tensor, ne);

    std::vector<float>       data_f32(ne0*ne1*ne2*ne3);
    std::vector<ggml_fp16_t> data_f16(ne0*ne1*ne2*ne3);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (size_t i = 0; i < data_f32.size(); i++) {
        data_f32[i] = dis(gen);
    }

    // block size
    const int blck0 = 128;
    const int blck1 = 64;

    // number of INF blocks
    const int n_inf_blocks = 0.1*(ne0*ne1*ne2*ne3)/(blck0*blck1);

    for (int b = 0; b < n_inf_blocks; b++) {
        const int p3 = (rd() % ne3);
        const int p2 = (rd() % ne2);
        const int p1 = (rd() % ne1);
        const int p0 = (rd() % ne0);

        for (int i1 = 0; i1 < blck1 && p1 + i1 < ne1; i1++) {
            const int idx = p3*ne2*ne1*ne0 + p2*ne1*ne0 + (p1 + i1)*ne0 + p0;

            for (int i0 = 0; i0 < blck0 && p0 + i0 < ne0; i0++) {
                data_f32[idx + i0] = -INFINITY;
            }
        }
    }

    ggml_fp32_to_fp16_row(data_f32.data(), data_f16.data(), ne0*ne1*ne2*ne3);

    ggml_backend_tensor_set(tensor, data_f16.data(), 0, data_f16.size()*sizeof(ggml_fp16_t));
}

void load_model(test_model & model, bool use_gpu = false) {
    // create data
    // int KW = 3, KH = 3, IC = 10, OC = 10;
    // int IW = 8, IH = 6, N = 1;
    int M = 1028, N = 1, K = 2816;

    model.hsk = 128;
    model.hsv = 128;
    model.nh = 32;
    model.kv = 96;
    model.nb = 8;
    model.type_KV = GGML_TYPE_F16;

    int64_t kv = 96, nb = 8;
    int64_t nh = 32;

    const int64_t hsk_padded = GGML_PAD(model.hsk, ggml_blck_size(model.type_KV));
    const int64_t hsv_padded = GGML_PAD(model.hsv, ggml_blck_size(model.type_KV));


    // Initialize adata
    std::vector<float> qdata(hsk_padded*nb*nh*model.nr23[0]*model.nr23[1]);
    for (int i = 0; i < hsk_padded*nb*nh*model.nr23[0]*model.nr23[1]; i++) {
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        qdata[i] = r;
    }

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> hqdata(hsk_padded*nb*nh*model.nr23[0]*model.nr23[1]);
    ggml_fp32_to_fp16_row(qdata.data(), hqdata.data(), hsk_padded*nb*nh*model.nr23[0]*model.nr23[1]);

    // Initialize bdata
    std::vector<float> kdata(hsk_padded*kv*nh*model.nr23[1]);
    for (int i = 0; i < hsk_padded*kv*nh*model.nr23[1]; i++) {
        // bdata[i] = 1.5f;
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        kdata[i] = r;
    }

    std::vector<ggml_fp16_t> hkdata(hsk_padded*kv*nh*model.nr23[1]);
    ggml_fp32_to_fp16_row(kdata.data(), hkdata.data(), hsk_padded*kv*nh*model.nr23[1]);

    std::vector<float> vdata(hsv_padded*kv*nh*model.nr23[1]);
    for (int i = 0; i < hsv_padded*kv*nh*model.nr23[1]; i++) {
        // bdata[i] = 1.5f;
        float r = -1.f + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1.f-(-1.f))));
        vdata[i] = r;
    }

    std::vector<ggml_fp16_t> hvdata(hsv_padded*kv*nh*model.nr23[1]);
    ggml_fp32_to_fp16_row(vdata.data(), hvdata.data(), hsv_padded*kv*nh*model.nr23[1]);


    size_t buffer_size = 0;
    {
        // buffer_size += M * K * ggml_type_size(GGML_TYPE_F16); // tensor a
        // buffer_size += N * K * ggml_type_size(GGML_TYPE_F16); // tensor b
        // buffer_size += N * K * ggml_type_size(GGML_TYPE_F32); // tensor b2
        buffer_size += hsk_padded*nb*nh*model.nr23[0]*model.nr23[1]*ggml_type_size(GGML_TYPE_F32);
        buffer_size += hsk_padded*nb*nh*model.nr23[0]*model.nr23[1]*ggml_type_size(GGML_TYPE_F16);
        buffer_size += hsk_padded*kv*nh*model.nr23[1]*ggml_type_size(GGML_TYPE_F32);
        buffer_size += hsk_padded*kv*nh*model.nr23[1]*ggml_type_size(GGML_TYPE_F16);
        buffer_size += hsv_padded*kv*nh*model.nr23[1]*ggml_type_size(GGML_TYPE_F32);
        buffer_size += hsv_padded*kv*nh*model.nr23[1]*ggml_type_size(GGML_TYPE_F16);
        buffer_size += kv*GGML_PAD(nb, GGML_KQ_MASK_PAD)*model.nr23[1] * ggml_type_size(GGML_TYPE_F16); // mask
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size/ 1024.f/ 1024.f));

    int num_tensors = 7;
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    ggml_log_set(ggml_log_callback_default, nullptr);


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
    // model.a = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, K, M, 1, 1);
    // model.b = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, K, N, 1, 1);
    // model.b2 = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, K, N, 1, 1);
    model.q32 = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32,  hsk_padded, nb, nh*model.nr23[0], model.nr23[1]);
    model.k32 = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32,  hsk_padded, kv, nh, model.nr23[1]);
    model.v32 = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32,  hsv_padded, kv, nh, model.nr23[1]);

    model.q16 = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  hsk_padded, nb, nh*model.nr23[0], model.nr23[1]);
    model.k16 = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  hsk_padded, kv, nh, model.nr23[1]);
    model.v16 = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16,  hsv_padded, kv, nh, model.nr23[1]);

    model.m = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F16, kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, model.nr23[1]);

    // create a allocator
    struct ggml_tallocr alloc = ggml_tallocr_new(model.buffer);

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.q32);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.q32->data, qdata.data(), ggml_nbytes(model.q32));
    } else {
        ggml_backend_tensor_set(model.q32, qdata.data(), 0, ggml_nbytes(model.q32));
    }

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.q16);

    // load data to buffer
    if(ggml_backend_is_cpu(model.backend)) {
        memcpy(model.q16->data, hqdata.data(), ggml_nbytes(model.q16));
    } else {
        ggml_backend_tensor_set(model.q16, hqdata.data(), 0, ggml_nbytes(model.q16));
    }

    // alloc memory
    ggml_tallocr_alloc(&alloc, model.k32);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.k32->data, kdata.data(), ggml_nbytes(model.k32));
    } else {
        ggml_backend_tensor_set(model.k32, kdata.data(), 0, ggml_nbytes(model.k32));
    }

    ggml_tallocr_alloc(&alloc, model.k16);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.k16->data, hkdata.data(), ggml_nbytes(model.k16));
    } else {
        ggml_backend_tensor_set(model.k16, hkdata.data(), 0, ggml_nbytes(model.k16));
    }


    // alloc memory
    ggml_tallocr_alloc(&alloc, model.v32);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.v32->data, vdata.data(), ggml_nbytes(model.v32));
    } else {
        ggml_backend_tensor_set(model.v32, vdata.data(), 0, ggml_nbytes(model.v32));
    }

    ggml_tallocr_alloc(&alloc, model.v16);

    if(ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
#endif
    ) {
        memcpy(model.v16->data, hvdata.data(), ggml_nbytes(model.v16));
    } else {
        ggml_backend_tensor_set(model.v16, hvdata.data(), 0, ggml_nbytes(model.v16));
    }

     ggml_tallocr_alloc(&alloc, model.m);

    init_tensor_kq_mask(model.m);

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
    // struct ggml_tensor* res = ggml_mul_mat(ctx0, model.a, model.b);
    ggml_tensor * out32 = ggml_flash_attn_ext(ctx0, model.q32, model.k32, model.v32, model.m,
             1.0f/sqrtf(model.hsk), model.max_bias, model.logit_softcap);
    ggml_set_name(out32, "res32");
    ggml_build_forward_expand(gf, out32);

    ggml_tensor * out16 = ggml_flash_attn_ext(ctx0, model.q16, model.k16, model.v16, model.m,
             1.0f/sqrtf(model.hsk), model.max_bias, model.logit_softcap);
    ggml_set_name(out16, "res16");
    ggml_build_forward_expand(gf, out16);

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

    struct ggml_tensor * gemm_res = NULL;
    struct ggml_tensor * gemm_res2 = NULL;

    for(int i = 0; i < ggml_graph_n_nodes(gf_res); ++i) {
        if(strcmp(ggml_get_name(ggml_graph_node(gf_res, i)), "res16") == 0) {
            gemm_res = ggml_graph_node(gf_res, i);
        }else if(strcmp(ggml_get_name(ggml_graph_node(gf_res, i)), "res32") == 0) {
            gemm_res2 = ggml_graph_node(gf_res, i);
        }
    }

    std::vector<ggml_fp16_t> gemm_data(ggml_nelements(gemm_res));
    std::vector<float> gemm_data2(ggml_nelements(gemm_res2));

    ggml_backend_tensor_get(gemm_res, gemm_data.data(), 0, ggml_nbytes(gemm_res));
    ggml_backend_tensor_get(gemm_res2, gemm_data2.data(), 0, ggml_nbytes(gemm_res2));

    for(int i = 0; i < gemm_data.size(); i++) {
        printf("%d: %f, %f\n", i, gemm_data2[i], ggml_fp16_to_fp32(gemm_data[i]));
    }

    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    ggml_gallocr_free(allocr);
    return 0;
}
