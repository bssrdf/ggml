#include "common.cuh"
#include <type_traits>

#if __CUDA_ARCH__ >= CC_AMPERE
#include "mma.h"
#endif

// #define BC 8
// #define BN 32
// #define BK 64
// #define TW 32
// #define TH 4

#define BN_p 138

__constant__ int access_f_s[2][32];
__constant__ int access_s[2][32];


#if __CUDA_ARCH__ >= CC_AMPERE

#define wmmaM  16
#define wmmaN  16
#define wmmaK  16

__constant__ int access_f_f[2][32];
__constant__ int access_t[2][32];
__constant__ int access_o[2][2];
__constant__ int access_p[2];

#define PADDING 1

// access_f_s
const int aux[2][32] = {
                        {0,2,4,6,0,2,4,6,
                         1,3,5,7,1,3,5,7,
                         0,2,4,6,0,2,4,6,
                         1,3,5,7,1,3,5,7},
                        {1,3,5,7,1,3,5,7,
                         0,2,4,6,0,2,4,6,
                         1,3,5,7,1,3,5,7,
                         0,2,4,6,0,2,4,6}
                        };
// access_f_f
const int aux1[2][32] = {
                        { 0, 0, 0, 0, 1, 1, 1, 1,
                          2, 2, 2, 2, 3, 3, 3, 3,
                          4, 4, 4, 4, 5, 5, 5, 5,
                          6, 6, 6, 6, 7, 7, 7, 7},                         
                        { 8, 8, 8, 8, 9, 9, 9, 9,
                         10,10,10,10,11,11,11,11,
                         12,12,12,12,13,13,13,13,
                         14,14,14,14,15,15,15,15}                          
                        };
// access_s
const int aux2[2][32] = {
                         {0,2,4,6,1,3,5,7,
                          0,2,4,6,1,3,5,7,
                          0,2,4,6,1,3,5,7,
                          0,2,4,6,1,3,5,7},                         
                         {1,3,5,7,0,2,4,6,
                          1,3,5,7,0,2,4,6,
                          1,3,5,7,0,2,4,6,
                          1,3,5,7,0,2,4,6}
                        };        
// access_t
const int aux3[2][32] = {
                         {0,2,4,6,0,2,4,6,
                          0,2,4,6,0,2,4,6,
                          8,10,12,14,8,10,12,14,
                          8,10,12,14,8,10,12,14},                          
                         {8,10,12,14,8,10,12,14,
                          8,10,12,14,8,10,12,14,
                          0,2,4,6,0,2,4,6,
                          0,2,4,6,0,2,4,6}                          
                        };       
const int aux_offset[2][2] = {{0, 4}, {16, 20}}; 
const int aux_offset1[2] = {0, BN_p}; 

#else

__constant__ int tileid[2][32];

// access_f_s
const int aux[2][32] = {
                        {0,0,1,1,2,2,3,3,4,4,5,5,6,6,
                            7,7,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7},
                        {8,8,9,9,10,10,11,11,12,12,13,13,
                            14,14,15,15,8,8,9,9,10,10,11,11,12,12,
                            13,13,14,14,15,15}
                        };
// access_s
const int aux2[2][32] = {
                         {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,
                          3,2,3,2,3,2,3,2,3,2,3,2,3,2,3},                         
                         {4,5,4,5,4,5,4,5,4,5,4,
                            5,4,5,4,5,6,7,6,7,6,7,6,7,
                            6,7,6,7,6,7,6,7}
                        }; 

const int tid[2][32] = {
                        {0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29,
                         0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29},
                        {2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31,
                         2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31}
                        };       

#endif


void ggml_cuda_op_winograd_stage0(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_winograd_stage1(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

