#include "ggml/ggml.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, const char ** argv) {
    const int n_threads = 1;

    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    {
        struct ggml_tensor * x = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

        ggml_set_param(ctx0, x);

        struct ggml_tensor * a = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        struct ggml_tensor * b = ggml_mul(ctx0, x, x);
        struct ggml_tensor * f = ggml_mul(ctx0, b, a);

        // a*x^2
        // 2*a*x

        ggml_print_objects(ctx0);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, f);
        printf("forward built \n");
        struct ggml_cgraph * gb = ggml_graph_dup(ctx0, gf);
        ggml_build_backward_expand(ctx0, gf, gb, false);

        printf(" forward order is %d \n", gf->order);
        printf("backward order is %d \n", gb->order);

        ggml_set_f32(x, 2.0f);
        ggml_set_f32(a, 3.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(f->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("f     = %f\n", ggml_get_f32_1d(f, 0));
        printf("df/dx = %f\n", ggml_get_f32_1d(x->grad, 0));

        GGML_ASSERT(ggml_get_f32_1d(f, 0)       == 12.0f);
        GGML_ASSERT(ggml_get_f32_1d(x->grad, 0) == 12.0f);

        ggml_set_f32(x, 3.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(f->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("f     = %f\n", ggml_get_f32_1d(f, 0));
        printf("df/dx = %f\n", ggml_get_f32_1d(x->grad, 0));

        GGML_ASSERT(ggml_get_f32_1d(f, 0)       == 27.0f);
        GGML_ASSERT(ggml_get_f32_1d(x->grad, 0) == 18.0f);
        for(int i=0; i < gf->n_nodes; i++){
            struct ggml_tensor * node = gf->nodes[i];
            printf(" forward %d, %s, %p \n ", i, node->name, (void *)node);
        }
        for(int i=0; i < gb->n_nodes; i++){
            struct ggml_tensor * node = gb->nodes[i];
            printf("backward %d, %s, %p, %p \n ", i, node->name, (void *)node, (void*)(node->grad));
        }
        for(int i=0; i < gb->n_leafs; i++){
            struct ggml_tensor * node = gb->leafs[i];
            printf("backward %d, %s, %p \n ", i, node->name, (void *)node);
        }

        ggml_graph_dump_dot(gf, NULL, "test1-1-forward.dot");
        ggml_graph_dump_dot(gb, gf,   "test1-1-backward.dot");
    }

    
    ///////////////////////////////////////////////////////////////

    {
        struct ggml_tensor * x1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        struct ggml_tensor * x2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        struct ggml_tensor * x3 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

        ggml_set_f32(x1, 4.0f);
        ggml_set_f32(x2, 2.0f);
        ggml_set_f32(x3, 0.0f);

        ggml_set_param(ctx0, x1);
        ggml_set_param(ctx0, x2);

        // struct ggml_tensor * y = ggml_add(ctx0, ggml_mul(ctx0, x1, x1), ggml_mul(ctx0, ggml_mul(ctx0, x2, x2), x2));
        struct ggml_tensor * y = ggml_add(ctx0, ggml_mul(ctx0, x1, x1), ggml_mul(ctx0, x1, x2));

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, y);
        struct ggml_cgraph * gb = ggml_graph_dup(ctx0, gf);
        ggml_build_backward_expand(ctx0, gf, gb, false);

        ggml_graph_reset(gf);
        ggml_set_f32(y->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("y      = %f\n", ggml_get_f32_1d(y, 0));
        printf("df/dx1 = %f\n", ggml_get_f32_1d(x1->grad, 0));
        printf("df/dx2 = %f\n", ggml_get_f32_1d(x2->grad, 0));

        // GGML_ASSERT(ggml_get_f32_1d(y, 0)        == 12.0f);
        // GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) == 7.0f);
        // GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) == 3.0f);

        for(int i=0; i < gf->n_nodes; i++){
            struct ggml_tensor * node = gf->nodes[i];
            printf(" forward %d, %s, %p \n ", i, node->name, (void *)node);
        }
        for(int i=0; i < gf->n_leafs; i++){
            struct ggml_tensor * node = gf->leafs[i];
            printf(" forward %d, %s, %p \n ", i, node->name, (void *)node);
        }
        for(int i=0; i < gb->n_nodes; i++){
            struct ggml_tensor * node = gb->nodes[i];
            printf("backward %d, %s, %p, (%p, %p) \n ", i, node->name, (void *)node, (void*)(node->src[0]), (void*)(node->src[1]));
        }
        for(int i=0; i < gb->n_leafs; i++){
            struct ggml_tensor * node = gb->leafs[i];
            printf("backward %d, %s, %p \n ", i, node->name, (void *)node);
        }

        ggml_graph_dump_dot(gf, NULL, "test1-2-forward.dot");
        ggml_graph_dump_dot(gb, gf,   "test1-2-backward.dot");

        struct ggml_tensor * g1 = x1->grad;
        struct ggml_tensor * g2 = x2->grad;

        struct ggml_cgraph * gbb = ggml_graph_dup(ctx0, gb);

        ggml_build_backward_expand(ctx0, gb, gbb, true);

        ggml_graph_reset(gb);
        ggml_set_f32(g1->grad, 1.0f);
        ggml_set_f32(g2->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gbb, n_threads);

        printf("H * [1, 1] = [ %f %f ]\n", ggml_get_f32_1d(x1->grad, 0), ggml_get_f32_1d(x2->grad, 0));

        for(int i=0; i < gbb->n_nodes; i++){
            struct ggml_tensor * node = gbb->nodes[i];
            printf("backbackward %d, %s, %p, (%p:%f, %p:%f) \n ", i, node->name, 
            (void *)node, (void*)(node->src[0]), node->src[0]?ggml_get_f32_1d(node->src[0], 0):-1.f, 
            (void *)(node->src[1]), node->src[1]?ggml_get_f32_1d(node->src[1], 0):-1.f);
        }
        for(int i=0; i < gbb->n_leafs; i++){
            struct ggml_tensor * node = gbb->leafs[i];
            printf("backbackward %d, %s, %p \n ", i, node->name, (void *)node);
        }
        struct ggml_tensor * node = x1->grad;
        printf("x1: %s, %p, (%p, %p) \n ", node->name, (void *)node, (void*)(node->src[0]), (void*)(node->src[1]));
        node = x2->grad;
        printf("x2: %s, %p, (%p, %p) \n ", node->name, (void *)node, (void*)(node->src[0]), (void*)(node->src[1]));


        printf("x1 = [ %f %f ]\n", ggml_get_f32_1d(gbb->nodes[19], 0), ggml_get_f32_1d(gbb->nodes[20], 0));
        printf("x1 = [ %f %f ]\n", ggml_get_f32_1d(gbb->nodes[17], 0), ggml_get_f32_1d(gbb->nodes[18], 0));

        printf("x2 = [ %f %f ]\n", ggml_get_f32_1d(gbb->nodes[22], 0), ggml_get_f32_1d(gbb->nodes[23], 0));
        

        // GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) == 3.0f);
        // GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) == 1.0f);

        // ggml_graph_dump_dot(gf, NULL, "test1-2-forward.dot");
        // ggml_graph_dump_dot(gb, gf,   "test1-2-backward.dot");
        ggml_graph_dump_dot(gbb, gb,  "test1-2-backward2.dot");
    }


    ///////////////////////////////////////////////////////////////

    {
        struct ggml_tensor * x1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        struct ggml_tensor * x2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

        ggml_set_param(ctx0, x1);
        ggml_set_param(ctx0, x2);

        struct ggml_tensor * y = ggml_mul(ctx0, ggml_add(ctx0, ggml_mul(ctx0, x1, x1), ggml_mul(ctx0, x1, x2)), x1);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, y);
        struct ggml_cgraph * gb = ggml_graph_dup(ctx0, gf);
        ggml_build_backward_expand(ctx0, gf, gb, false);

        ggml_set_f32(x1, 3.0f);
        ggml_set_f32(x2, 4.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(y->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("y      = %f\n", ggml_get_f32_1d(y, 0));
        printf("df/dx1 = %f\n", ggml_get_f32_1d(x1->grad, 0));
        printf("df/dx2 = %f\n", ggml_get_f32_1d(x2->grad, 0));

        GGML_ASSERT(ggml_get_f32_1d(y, 0)        == 63.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) == 51.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) == 9.0f);

        ggml_graph_dump_dot(gf, NULL, "test1-3-forward.dot");
        ggml_graph_dump_dot(gb, gf,   "test1-3-backward.dot");
    }
    ///////////////////////////////////////////////////////////////

    {
        struct ggml_tensor * x1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        struct ggml_tensor * x2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        struct ggml_tensor * x3 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);

        ggml_set_param(ctx0, x1);
        ggml_set_param(ctx0, x2);
        ggml_set_param(ctx0, x3);

        struct ggml_tensor * y = ggml_mul(ctx0, ggml_mul(ctx0, ggml_mul(ctx0, x1, x1), ggml_mul(ctx0, x2, x2)), x3);

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, y);
        struct ggml_cgraph * gb = ggml_graph_dup(ctx0, gf);
        ggml_build_backward_expand(ctx0, gf, gb, false);

        ggml_set_f32(x1, 1.0f);
        ggml_set_f32(x2, 2.0f);
        ggml_set_f32(x3, 3.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(y->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("y      = %f\n", ggml_get_f32_1d(y, 0));
        printf("df/dx1 = %f\n", ggml_get_f32_1d(x1->grad, 0));
        printf("df/dx2 = %f\n", ggml_get_f32_1d(x2->grad, 0));
        printf("df/dx3 = %f\n", ggml_get_f32_1d(x3->grad, 0));

        GGML_ASSERT(ggml_get_f32_1d(y, 0)        == 12.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) == 24.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) == 12.0f);
        GGML_ASSERT(ggml_get_f32_1d(x3->grad, 0) == 4.0f);

        struct ggml_tensor * g1 = x1->grad;
        struct ggml_tensor * g2 = x2->grad;
        struct ggml_tensor * g3 = x3->grad;

        struct ggml_cgraph * gbb = ggml_graph_dup(ctx0, gb);

        ggml_build_backward_expand(ctx0, gb, gbb, true);

        ggml_graph_reset(gb);
        ggml_set_f32(g1->grad, 1.0f);
        ggml_set_f32(g2->grad, 1.0f);
        ggml_set_f32(g3->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gbb, n_threads);

        printf("H * [1, 1, 1]^T = [ %f %f %f ]\n",
                ggml_get_f32_1d(x1->grad, 0),
                ggml_get_f32_1d(x2->grad, 0),
                ggml_get_f32_1d(x3->grad, 0));

        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) == 56.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) == 34.0f);
        GGML_ASSERT(ggml_get_f32_1d(x3->grad, 0) == 12.0f);

        ggml_graph_dump_dot(gf, NULL, "test1-4-forward.dot");
        ggml_graph_dump_dot(gb, gf,   "test1-4-backward.dot");
    }
    
    ///////////////////////////////////////////////////////////////

    {
        struct ggml_tensor * x1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);
        struct ggml_tensor * x2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);

        ggml_set_param(ctx0, x1);
        ggml_set_param(ctx0, x2);

        struct ggml_tensor * y = ggml_sum(ctx0, ggml_mul(ctx0, x1, x2));

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, y);
        struct ggml_cgraph * gb = ggml_graph_dup(ctx0, gf);
        ggml_build_backward_expand(ctx0, gf, gb, false);

        ggml_set_f32(x1, 3.0f);
        ggml_set_f32(x2, 5.0f);

        printf("x1 = %f %f %f\n",
                ggml_get_f32_1d(x1, 0),
                ggml_get_f32_1d(x1, 1),
                ggml_get_f32_1d(x1, 2));

        ggml_graph_reset(gf);
        ggml_set_f32(y->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("y      = %f\n", ggml_get_f32_1d(y, 0));
        printf("df/dx1 = %f %f %f\n",
                ggml_get_f32_1d(x1->grad, 0),
                ggml_get_f32_1d(x1->grad, 1),
                ggml_get_f32_1d(x1->grad, 2));
        printf("df/dx2 = %f %f %f\n",
                ggml_get_f32_1d(x2->grad, 0),
                ggml_get_f32_1d(x2->grad, 1),
                ggml_get_f32_1d(x2->grad, 2));

        GGML_ASSERT(ggml_get_f32_1d(y, 0)        == 45.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) == 5.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) == 3.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 1) == 5.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 1) == 3.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 2) == 5.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 2) == 3.0f);

        ggml_graph_dump_dot(gf, NULL, "test1-5-forward.dot");
        ggml_graph_dump_dot(gb, gf,   "test1-5-backward.dot");
    }


    
    ///////////////////////////////////////////////////////////////

    {
        struct ggml_tensor * x1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);
        struct ggml_tensor * x2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);

        ggml_set_param(ctx0, x1);
        ggml_set_param(ctx0, x2);

        struct ggml_tensor * y =
            ggml_sum(ctx0,
                    ggml_add(ctx0,
                        ggml_mul(ctx0, x1, x2),
                        ggml_mul(ctx0,
                            ggml_repeat(ctx0, ggml_new_f32(ctx0, -2.0f), x1),
                            ggml_mul(ctx0, x1, x1)
                            )
                        )
                    );

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, y);
        struct ggml_cgraph * gb = ggml_graph_dup(ctx0, gf);
        ggml_build_backward_expand(ctx0, gf, gb, false);

        ggml_set_f32(x1, 3.0f);
        ggml_set_f32(x2, 5.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(y->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("y      = %f\n", ggml_get_f32_1d(y, 0));
        printf("df/dx1 = %f %f %f\n",
                ggml_get_f32_1d(x1->grad, 0),
                ggml_get_f32_1d(x1->grad, 1),
                ggml_get_f32_1d(x1->grad, 2));
        printf("df/dx2 = %f %f %f\n",
                ggml_get_f32_1d(x2->grad, 0),
                ggml_get_f32_1d(x2->grad, 1),
                ggml_get_f32_1d(x2->grad, 2));

        GGML_ASSERT(ggml_get_f32_1d(y, 0)              == -9.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) == -7.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 1) == -7.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 2) == -7.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) ==  3.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 1) ==  3.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 2) ==  3.0f);

        ggml_graph_dump_dot(gf, NULL, "test1-6-forward.dot");
        ggml_graph_dump_dot(gb, gf,   "test1-6-backward.dot");
    }
 
    /*
    ///////////////////////////////////////////////////////////////

    {
        struct ggml_tensor * x1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);
        struct ggml_tensor * x2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);

        ggml_set_param(ctx0, x1);
        ggml_set_param(ctx0, x2);

        struct ggml_tensor * y =
            ggml_sum(ctx0,
                    ggml_sub(ctx0,
                        ggml_mul(ctx0, x1, x2),
                        ggml_mul(ctx0,
                            ggml_mul(ctx0, x1, x1),
                            ggml_repeat(ctx0, ggml_new_f32(ctx0, -2.0f), x1)
                            )
                        )
                    );

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, y);
        struct ggml_cgraph * gb = ggml_graph_dup(ctx0, gf);
        ggml_build_backward_expand(ctx0, gf, gb, false);

        ggml_set_f32(x1, 3.0f);
        ggml_set_f32(x2, 5.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(y->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("y      = %f\n", ggml_get_f32_1d(y, 0));
        printf("df/dx1 = %f %f %f\n",
                ggml_get_f32_1d(x1->grad, 0),
                ggml_get_f32_1d(x1->grad, 1),
                ggml_get_f32_1d(x1->grad, 2));
        printf("df/dx2 = %f %f %f\n",
                ggml_get_f32_1d(x2->grad, 0),
                ggml_get_f32_1d(x2->grad, 1),
                ggml_get_f32_1d(x2->grad, 2));

        GGML_ASSERT(ggml_get_f32_1d(y, 0)        == 99.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) == 17.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 1) == 17.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 2) == 17.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) ==  3.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 1) ==  3.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 2) ==  3.0f);

        ggml_graph_dump_dot(gf, NULL, "test1-7-forward.dot");
        ggml_graph_dump_dot(gb, gf,   "test1-7-backward.dot");
    }

    ///////////////////////////////////////////////////////////////

    {
        struct ggml_tensor * x1 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);
        struct ggml_tensor * x2 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 3);

        ggml_set_param(ctx0, x1);
        ggml_set_param(ctx0, x2);

        struct ggml_tensor * y =
            ggml_abs(ctx0,
                    ggml_sub(ctx0, x1, x2)
                    );

        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, y);
        struct ggml_cgraph * gb = ggml_graph_dup(ctx0, gf);
        ggml_build_backward_expand(ctx0, gf, gb, false);

        ggml_set_f32(x1, 3.0f);
        ggml_set_f32(x2, 5.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(y->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("y      = %f\n", ggml_get_f32_1d(y, 0));
        printf("df/dx1 = %f %f %f\n",
                ggml_get_f32_1d(x1->grad, 0),
                ggml_get_f32_1d(x1->grad, 1),
                ggml_get_f32_1d(x1->grad, 2));
        printf("df/dx2 = %f %f %f\n",
                ggml_get_f32_1d(x2->grad, 0),
                ggml_get_f32_1d(x2->grad, 1),
                ggml_get_f32_1d(x2->grad, 2));

        GGML_ASSERT(ggml_get_f32_1d(y, 0)        ==  2.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) == -1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 1) == -1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 2) == -1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) ==  1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 1) ==  1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 2) ==  1.0f);

        ggml_set_f32(x1, 7.0f);
        ggml_set_f32(x2, 5.0f);

        ggml_graph_reset(gf);
        ggml_set_f32(y->grad, 1.0f);

        ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

        printf("y      = %f\n", ggml_get_f32_1d(y, 0));
        printf("df/dx1 = %f %f %f\n",
                ggml_get_f32_1d(x1->grad, 0),
                ggml_get_f32_1d(x1->grad, 1),
                ggml_get_f32_1d(x1->grad, 2));
        printf("df/dx2 = %f %f %f\n",
                ggml_get_f32_1d(x2->grad, 0),
                ggml_get_f32_1d(x2->grad, 1),
                ggml_get_f32_1d(x2->grad, 2));

        GGML_ASSERT(ggml_get_f32_1d(y, 0)        ==  2.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 0) ==  1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 1) ==  1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x1->grad, 2) ==  1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 0) == -1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 1) == -1.0f);
        GGML_ASSERT(ggml_get_f32_1d(x2->grad, 2) == -1.0f);

        ggml_graph_dump_dot(gf, NULL, "test1-8-forward.dot");
        ggml_graph_dump_dot(gb, gf,   "test1-8-backward.dot");
    }
    */
    ggml_free(ctx0);

    return 0;
}
