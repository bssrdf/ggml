#include "conv-winograd.cuh"
#include "convert.cuh"
#include <cuda_runtime.h>

template<typename T>
__device__ void __inline__ outer_product(float4* input_frag, float4* filter_frag, float4 accumulator_in[][16]){
    T *accumulator = (T *)accumulator_in;
    T *x           = (T *)input_frag;
    T *y           = (T *)filter_frag;

    *(accumulator)        += (*(x+0)) * (*y);
    *(accumulator+1)      += (*(x+1)) * (*(y));
    *(accumulator+2)      += (*(x+2)) * (*(y));
    *(accumulator+3)      += (*(x+3)) * (*(y));

    *(accumulator+4)      += (*(x+4)) * (*(y));
    *(accumulator+5)      += (*(x+5)) * (*(y));
    *(accumulator+6)      += (*(x+6)) * (*(y));
    *(accumulator+7)      += (*(x+7)) * (*(y));

    *(accumulator+8)      += (*(x+0)) * (*(y+1));
    *(accumulator+9)      += (*(x+1)) * (*(y+1));
    *(accumulator+10)      += (*(x+2)) * (*(y+1));
    *(accumulator+11)      += (*(x+3)) * (*(y+1));

    *(accumulator+12)      += (*(x+4)) * (*(y+1));
    *(accumulator+13)      += (*(x+5)) * (*(y+1));
    *(accumulator+14)      += (*(x+6)) * (*(y+1));
    *(accumulator+15)      += (*(x+7)) * (*(y+1));

    *(accumulator+16)      += (*(x+0)) * (*(y+2));
    *(accumulator+17)      += (*(x+1)) * (*(y+2));
    *(accumulator+18)      += (*(x+2)) * (*(y+2));
    *(accumulator+19)      += (*(x+3)) * (*(y+2));

    *(accumulator+20)      += (*(x+4)) * (*(y+2));
    *(accumulator+21)      += (*(x+5)) * (*(y+2));
    *(accumulator+22)      += (*(x+6)) * (*(y+2));
    *(accumulator+23)      += (*(x+7)) * (*(y+2));

    *(accumulator+24)      += (*(x+0)) * (*(y+3));
    *(accumulator+25)      += (*(x+1)) * (*(y+3));
    *(accumulator+26)      += (*(x+2)) * (*(y+3));
    *(accumulator+27)      += (*(x+3)) * (*(y+3));

    *(accumulator+28)      += (*(x+4)) * (*(y+3));
    *(accumulator+29)      += (*(x+5)) * (*(y+3));
    *(accumulator+30)      += (*(x+6)) * (*(y+3));
    *(accumulator+31)      += (*(x+7)) * (*(y+3));

   //
    *(accumulator+32)      += (*(x+0)) * (*(y+4));
    *(accumulator+33)      += (*(x+1)) * (*(y+4));
    *(accumulator+34)      += (*(x+2)) * (*(y+4));
    *(accumulator+35)      += (*(x+3)) * (*(y+4));

    *(accumulator+36)      += (*(x+4)) * (*(y+4));
    *(accumulator+37)      += (*(x+5)) * (*(y+4));
    *(accumulator+38)      += (*(x+6)) * (*(y+4));
    *(accumulator+39)      += (*(x+7)) * (*(y+4));

    *(accumulator+40)      += (*(x+0)) * (*(y+5));
    *(accumulator+41)      += (*(x+1)) * (*(y+5));
    *(accumulator+42)      += (*(x+2)) * (*(y+5));
    *(accumulator+43)      += (*(x+3)) * (*(y+5));

    *(accumulator+44)      += (*(x+4)) * (*(y+5));
    *(accumulator+45)      += (*(x+5)) * (*(y+5));
    *(accumulator+46)      += (*(x+6)) * (*(y+5));
    *(accumulator+47)      += (*(x+7)) * (*(y+5));

    *(accumulator+48)      += (*(x+0)) * (*(y+6));
    *(accumulator+49)      += (*(x+1)) * (*(y+6));
    *(accumulator+50)      += (*(x+2)) * (*(y+6));
    *(accumulator+51)      += (*(x+3)) * (*(y+6));

    *(accumulator+52)      += (*(x+4)) * (*(y+6));
    *(accumulator+53)      += (*(x+5)) * (*(y+6));
    *(accumulator+54)      += (*(x+6)) * (*(y+6));
    *(accumulator+55)      += (*(x+7)) * (*(y+6));

    *(accumulator+56)      += (*(x+0)) * (*(y+7));
    *(accumulator+57)      += (*(x+1)) * (*(y+7));
    *(accumulator+58)      += (*(x+2)) * (*(y+7));
    *(accumulator+59)      += (*(x+3)) * (*(y+7));

    *(accumulator+60)      += (*(x+4)) * (*(y+7));
    *(accumulator+61)      += (*(x+5)) * (*(y+7));
    *(accumulator+62)      += (*(x+6)) * (*(y+7));
    *(accumulator+63)      += (*(x+7)) * (*(y+7));

    //////

    *(accumulator+64)      += (*(x+8)) * (*(y+8));
    *(accumulator+65)      += (*(x+9)) * (*(y+8));
    *(accumulator+66)      += (*(x+10)) * (*(y+8));
    *(accumulator+67)      += (*(x+11)) * (*(y+8));
    *(accumulator+68)      += (*(x+12)) * (*(y+8));
    *(accumulator+69)      += (*(x+13)) * (*(y+8));
    *(accumulator+70)      += (*(x+14)) * (*(y+8));
    *(accumulator+71)      += (*(x+15)) * (*(y+8));

    *(accumulator+72)      += (*(x+8)) * (*(y+9));
    *(accumulator+73)      += (*(x+9)) * (*(y+9));
    *(accumulator+74)      += (*(x+10)) * (*(y+9));
    *(accumulator+75)      += (*(x+11)) * (*(y+9));
    *(accumulator+76)      += (*(x+12)) * (*(y+9));
    *(accumulator+77)      += (*(x+13)) * (*(y+9));
    *(accumulator+78)      += (*(x+14)) * (*(y+9));
    *(accumulator+79)      += (*(x+15)) * (*(y+9));

    *(accumulator+80)      += (*(x+8)) * (*(y+10));
    *(accumulator+81)      += (*(x+9)) * (*(y+10));
    *(accumulator+82)      += (*(x+10)) * (*(y+10));
    *(accumulator+83)      += (*(x+11)) * (*(y+10));
    *(accumulator+84)      += (*(x+12)) * (*(y+10));
    *(accumulator+85)      += (*(x+13)) * (*(y+10));
    *(accumulator+86)      += (*(x+14)) * (*(y+10));
    *(accumulator+87)      += (*(x+15)) * (*(y+10));

    *(accumulator+88)      += (*(x+8)) * (*(y+11));
    *(accumulator+89)      += (*(x+9)) * (*(y+11));
    *(accumulator+90)      += (*(x+10)) * (*(y+11));
    *(accumulator+91)      += (*(x+11)) * (*(y+11));
    *(accumulator+92)      += (*(x+12)) * (*(y+11));
    *(accumulator+93)      += (*(x+13)) * (*(y+11));
    *(accumulator+94)      += (*(x+14)) * (*(y+11));
    *(accumulator+95)      += (*(x+15)) * (*(y+11));
  
    //

    *(accumulator+96)      += (*(x+8)) * (*(y+12));
    *(accumulator+97)      += (*(x+9)) * (*(y+12));
    *(accumulator+98)      += (*(x+10)) * (*(y+12));
    *(accumulator+99)      += (*(x+11)) * (*(y+12));
    *(accumulator+100)      += (*(x+12)) * (*(y+12));
    *(accumulator+101)      += (*(x+13)) * (*(y+12));
    *(accumulator+102)      += (*(x+14)) * (*(y+12));
    *(accumulator+103)      += (*(x+15)) * (*(y+12));

    *(accumulator+104)      += (*(x+8)) * (*(y+13));
    *(accumulator+105)      += (*(x+9)) * (*(y+13));
    *(accumulator+106)      += (*(x+10)) * (*(y+13));
    *(accumulator+107)      += (*(x+11)) * (*(y+13));
    *(accumulator+108)      += (*(x+12)) * (*(y+13));
    *(accumulator+109)      += (*(x+13)) * (*(y+13));
    *(accumulator+110)      += (*(x+14)) * (*(y+13));
    *(accumulator+111)      += (*(x+15)) * (*(y+13));

    *(accumulator+112)      += (*(x+8)) * (*(y+14));
    *(accumulator+113)      += (*(x+9)) * (*(y+14));
    *(accumulator+114)      += (*(x+10)) * (*(y+14));
    *(accumulator+115)      += (*(x+11)) * (*(y+14));
    *(accumulator+116)      += (*(x+12)) * (*(y+14));
    *(accumulator+117)      += (*(x+13)) * (*(y+14));
    *(accumulator+118)      += (*(x+14)) * (*(y+14));
    *(accumulator+119)      += (*(x+15)) * (*(y+14));

    *(accumulator+120)      += (*(x+8)) * (*(y+15));
    *(accumulator+121)      += (*(x+9)) * (*(y+15));
    *(accumulator+122)      += (*(x+10)) * (*(y+15));
    *(accumulator+123)      += (*(x+11)) * (*(y+15));
    *(accumulator+124)      += (*(x+12)) * (*(y+15));
    *(accumulator+125)      += (*(x+13)) * (*(y+15));
    *(accumulator+126)      += (*(x+14)) * (*(y+15));
    *(accumulator+127)      += (*(x+15)) * (*(y+15));

  }

// extern "C"
// {

#if __CUDA_ARCH__ >= CC_AMPERE

__device__ __forceinline__ void  transform_output_tile(float *pOutputs, float2 *C_tile, float2 *At, 
    int round, int c_tensor, int c_glb_offset, int id, unsigned short mask, int out_w)
{                     
  // c_tensor += (((round)/2)*32 + ((round)%2)*2)*c_glb_offset/2;  
  // c_tensor +=  round * 16 * c_glb_offset; //each round moves 16 (= 64/4) K
  // c_tensor +=  16 * c_glb_offset; //each round moves 16 (= 64/4) K
  // int c_tensor1 = c_tensor + 16 * c_glb_offset;
  int x, x1;

  #pragma unroll
  for(int j=0; j<4; j++){

    At[j].x = C_tile[j].x + C_tile[4+j].x + C_tile[8+j].x;
    At[j].y = C_tile[j].y + C_tile[4+j].y + C_tile[8+j].y;

    At[4+j].x = C_tile[4+j].x - C_tile[8+j].x - C_tile[12+j].x;
    At[4+j].y = C_tile[4+j].y - C_tile[8+j].y - C_tile[12+j].y;
    
  }  

  #pragma unroll
  for(int i=0; i<2; i++){
    x = i*4;
    // x1 = i*((tiles_dim-(out_w%2)) + (out_w%2)/2);
    x1 = i*((out_w-(out_w%2)) + (out_w%2)/2);    

    if(mask&(1<<(i*2))){
      pOutputs[x1 + c_tensor + id] = At[x].x + At[x+1].x + At[x+2].x;
    }
    if(mask&(1<<(i*2))){
      pOutputs[x1 + c_tensor + id + c_glb_offset] = At[x].y + At[x+1].y + At[x+2].y;
      
    }
    if(mask&(1<<(i*2+1))){
      pOutputs[x1 + c_tensor + id + 1] = At[x+1].x - At[x+2].x - At[x+3].x;
    }
    if(mask&(1<<(i*2+1))){
      pOutputs[x1 + c_tensor + c_glb_offset + id + 1] = At[x+1].y - At[x+2].y - At[x+3].y;
    }
  } 
}

#else

__device__ __forceinline__ void  transform_output_tile(float * __restrict__ pOutputs, float2 *C_tile, float2 *At, 
    int round, int c_tensor, int c_glb_offset, int i1, int i2,
    unsigned int mask1, unsigned int mask2, int out_w)
{                     

  c_tensor += (((round)>>1)*32 + ((round)&1)*2)*c_glb_offset;
  int x, x1;

  #pragma unroll
  for(int j=0; j<4; j++){

    At[j].x = C_tile[j].x + C_tile[4+j].x + C_tile[8+j].x;
    At[j].y = C_tile[j].y + C_tile[4+j].y + C_tile[8+j].y;

    At[4+j].x = C_tile[4+j].x - C_tile[8+j].x - C_tile[12+j].x;
    At[4+j].y = C_tile[4+j].y - C_tile[8+j].y - C_tile[12+j].y;
    
  }

  #pragma unroll
  for(int i=0; i<2; i++){
    x = i*4;
    x1 = i*((out_w-(out_w&1)) + (out_w&1)/2);

    if(mask1&(1<<(i*2))){
      pOutputs[x1 + c_tensor + i1] = At[x].x + At[x+1].x + At[x+2].x;
    }
    if(mask2&(1<<(i*2))){
      pOutputs[x1 + c_tensor + i2] = At[x].y + At[x+1].y + At[x+2].y;
    }
    if(mask1&(1<<(i*2+1))){
      pOutputs[x1 + c_tensor + i1 + 1] = At[x+1].x - At[x+2].x - At[x+3].x;
    }
    if(mask2&(1<<(i*2+1))){
      pOutputs[x1 + c_tensor + i2 + 1] = At[x+1].y - At[x+2].y - At[x+3].y;
    }
  } 
}

#endif

template<int TW, int TH>
__device__ __forceinline__ unsigned int get_mask(int idd, int tiles_dim_w, int tiles_dim_h, 
         int tw, int th, int out_w, int out_h){

  unsigned int mask = 0x000F;
  // if((blockIdx.y/tiles_dim)==(tiles_dim-1) && out_w%2) mask&=0x0003; // pad bottom row
  // if(!((blockIdx.y+1)%tiles_dim) && out_w%2)           mask&=0X0005; // pad right col
  // if(blockIdx.y==gridDim.y-1 && (idd / tw) == th-1 && out_h%2)  mask&=0x0003; // pad bottom row
  // if(blockIdx.x==gridDim.x-1 && (idd % tw) == tw-1 && out_w%2)  mask&=0X0005; // pad right col
  if((tiles_dim_w & (tw-1)) == 0 && (tiles_dim_h & (th-1)) == 0){
    if(blockIdx.y==gridDim.y-1 && (idd / tw)     == th-1 && (out_h&1))  mask&=0x0003; // pad bottom row
    if(blockIdx.x==gridDim.x-1 && (idd & (tw-1)) == tw-1 && (out_w&1))  mask&=0X0005; // pad right col
  }else if((tiles_dim_w & (tw-1)) == 0){
    int k = out_h & (TH-1);
    int k1 =  k & 1 ? (k+1)>>1 : k>>1; // there could be 4*k1 tiles
    if(blockIdx.y==gridDim.y-1 && (idd / tw) == k1-1 && (k&1))  mask&=0x0003; // pad bottom row
    if(blockIdx.y==gridDim.y-1 && (idd / tw) > k1-1) mask &= 0x0; //pad all zeros since this tile does not exist
  }else if((tiles_dim_h & (th-1)) == 0){
    int k = out_w & (TW-1);
    int k1 =  k & 1 ? (k+1) >> 1 : k >> 1; // there could be 4*k1 tiles
    if(blockIdx.x==gridDim.x-1 && (idd & (tw-1)) == k1-1 && (k&1))  mask&=0X0005; // pad right col
    if(blockIdx.x==gridDim.x-1 && (idd & (tw-1)) > k1-1)  mask&=0X0; // pad all zeroes
  }else{
    int kh = out_h & (TH-1);
    int kw = out_w & (TW-1);
    int kh1 =  kh & 1 ? (kh+1) >> 1 : kh >> 1; // there could be kh1*kw1 tiles
    int kw1 =  kw & 1 ? (kw+1) >> 1 : kw >> 1;
    if(blockIdx.y==gridDim.y-1 && (idd / tw) == kh1-1 && (kh&1))  mask&=0x0003; // pad bottom row
    if(blockIdx.x==gridDim.x-1 && (idd & (tw-1)) == kw1-1 && (kw&1))  mask&=0X0005; // pad right col
    if(blockIdx.y==gridDim.y-1 && (idd / tw) > kh1-1)  mask &= 0x0; //pad all zeros since this tile does not exist
    if(blockIdx.x==gridDim.x-1 && (idd & (tw-1)) > kw1-1)  mask &= 0X0; // pad all zeroes
  }
  return mask;
}


#if __CUDA_ARCH__ >= CC_AMPERE

__device__ __forceinline__ int loc(int st, int k){
  
  int t = (st % 8) * 4;
  t += (k%8) / 2;
  int accum1 = ((t%8)/2)*34 + t%2 + (t/16)*2 + ((t/8)%2)*8;
  int sst = st < 16 ? st : st-16;
  accum1 += access_o[sst/8][k/8];
  accum1 += access_p[st/16];

  // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == 0  && threadIdx.y == 4){
  //   printf("AA %d, %d, %d, %d, %d \n", accum1, access_o[sst/8][k/8], access_p[st/16], st, k);
  // }
  return accum1; 

}

template<int TW, int TH, int BN, int BK, int BC>
__device__ __forceinline__ void store_output_tile(float *Accum, unsigned char* shared_mem, float *C, 
                       int out_h, int out_w, int tiles_dim_w, int tiles_dim_h,  int tw, int th){
  
  float2 *output_smem = (float2 *) shared_mem;
  float2 *accumulator = (float2 *) Accum;
  // float2 *C_out = (float2*)C;

  // float2 *C_tile = (float2*) input_frag_mem;
  // float2 *At = (float2*) filter_frag_mem;

  float2 C_tile[16]; 
  float2 At[16]; 

  int warpid = threadIdx.y;
  int laneid = threadIdx.x;
  unsigned short mask1 = get_mask(laneid, tiles_dim_w, tiles_dim_h, tw, th, out_w, out_h);
  int id1 = (laneid % tw) * 2 + (laneid / tw) * out_w * 2;
  
  int acumm1 = ((threadIdx.x%8)/2)*34 + threadIdx.x%2 + (threadIdx.x/16)*2 + ((threadIdx.x/8)%2)*8;
  int acumm2 = acumm1+4;
                       
  
  // For transformating
  int offset = BN_p*2; //*2/2
  int offsetp = warpid*BN_p*2; //*2/2

  int idx = BN/wmmaM*BK/wmmaN*4;
  // int init = ( (threadIdx.y/4)*BN_p*16 + (threadIdx.y%4)*(32+2) ) *2 + threadIdx.x;

  int c_glb_offset = out_h*out_w;
  

  int c_tensor = blockIdx.z*c_glb_offset*BK + blockIdx.x * TW  + blockIdx.y * out_w * TH +
                //  (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * out_w * 2 + 
                 warpid*2*c_glb_offset;


  #pragma unroll                                  
  for(int round=0; round<4; round++){

    // each round will output 16(E)x32(T)x16(K)
    // 32T is divided into 2x16(T)
    // each thread needs to write 8 floats into smem.

    
    *((float2 *)(output_smem + offsetp + acumm1))      = accumulator[round*4+0];
    *((float2 *)(output_smem + offsetp + acumm2))      = accumulator[round*4+1];
    *((float2 *)(output_smem + offsetp + acumm1 + 16)) = accumulator[round*4+2];
    *((float2 *)(output_smem + offsetp + acumm2 + 16)) = accumulator[round*4+3];
      
    

    *((float2 *)(output_smem+offsetp + BN_p + acumm1))      = accumulator[BK/4+round*4+0];
    *((float2 *)(output_smem+offsetp + BN_p + acumm2))      = accumulator[BK/4+round*4+1];
    *((float2 *)(output_smem+offsetp + BN_p + acumm1 + 16)) = accumulator[BK/4+round*4+2];
    *((float2 *)(output_smem+offsetp + BN_p + acumm2 + 16)) = accumulator[BK/4+round*4+3];
    
    
    *((float2 *)(output_smem+BN_p*16 + offsetp + acumm1))      =  accumulator[idx + round*4 + 0];
    *((float2 *)(output_smem+BN_p*16 + offsetp + acumm2))      =  accumulator[idx + round*4 + 1];
    *((float2 *)(output_smem+BN_p*16 + offsetp + acumm1 + 16)) =  accumulator[idx + round*4 + 2];
    *((float2 *)(output_smem+BN_p*16 + offsetp + acumm2 + 16)) =  accumulator[idx + round*4 + 3];



    *((float2 *)(output_smem+BN_p*16 + offsetp + BN_p + acumm1))      = accumulator[idx + BK/4 + round*4 + 0];
    *((float2 *)(output_smem+BN_p*16 + offsetp + BN_p + acumm2))      = accumulator[idx + BK/4 + round*4 + 1];
    *((float2 *)(output_smem+BN_p*16 + offsetp + BN_p + acumm1 + 16)) = accumulator[idx + BK/4 + round*4 + 2];
    *((float2 *)(output_smem+BN_p*16 + offsetp + BN_p + acumm2 + 16)) = accumulator[idx + BK/4 + round*4 + 3];

    __syncthreads();


    // for output transformation, the role of threadIdx.y changes again:
    // in the main loop, different threadIdx.y deal with different element of the 4x4 tile 
    // here, they are for 4 different groups of lane ids from optSTS64 layout
    // for init (and init+32), we need to identify its tile number (0-31) within the supertile     
    // first, from init, find out from which threadIdx.x it comes.
    // int idy = init - threadIdx.x;
    // if(idy > 204) idy -= BN_p*16*2; 
    // int idx = idy + threadIdx.x;
    // if(idx % 2 == 0)
    //     idx = idx / 2;
    // else
    //     idx = (idx-1) / 2;
    // int l = laneid[idx];
    // now we got l, which is the land id which computed accumulated sum for the tile element 
    // each lane id (or threadIdx.x) computed 8 tiles which are distributed into 4 locations spreading
    // over the smem. We need to find which of the 8 the current tile is.   
    // use tileid table to figure out
    // int id1 = tileid[0][l];


    // for 2nd tile
    // idx = idy + threadIdx.x + 32;
    // if(idx % 2 == 0)
    //     idx = idx / 2;
    // else
    //     idx = (idx-1) / 2;
    // l = laneid[idx];
    // int id2 = tileid[1][l];


    // int tx = 0, ty=0; 
    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&  threadIdx.x == tx  && threadIdx.y == ty)      
    //   printf("round, %d, [", round);
    for(int i=0; i<16; i++){
      C_tile[i] = output_smem[i*offset + loc(laneid, warpid*2)];
      
    }
    

    // transform output tiles
    // transform_output_tile(C, C_tile, At, tiles_dim, round, c_tensor, c_glb_offset, id1, id2, mask, out_w);
    transform_output_tile(C, C_tile, At, round, c_tensor, c_glb_offset, id1, mask1, out_w);
    __syncthreads();

    c_tensor +=  16 * c_glb_offset;

  }
}

#else

template<int TW, int TH, int BN, int BK, int BC>
__device__ __forceinline__ void store_output_tile(float4 acumm_smem[][16], float *shared_mem, float * __restrict__ C, 
int out_h, int out_w, int tiles_dim_w, int tiles_dim_h,  int tw, int th, 
float4 *input_frag_mem, float4* filter_frag_mem){
  
  float2 *output_smem = (float2 *) shared_mem;
  float2 *accumulator = (float2 *) acumm_smem;
  // float2 *C_out = (float2*)C;

  float2 *C_tile = (float2*) input_frag_mem;
  float2 *At = (float2*) filter_frag_mem;
  // for output transformation, the role of threadIdx.y changes again:
    // in the main loop, different threadIdx.y deal with different element of the 4x4 tile 
    // here, they are for 4 different groups of lane ids from optSTS64 layout
    // for init (and init+32), we need to identify its tile number (0-31) within the supertile     
    // first, from init, find out from which threadIdx.x it comes.

    // now we got l, which is the land id which computed accumulated sum for the tile element 
    // each lane id (or threadIdx.x) computed 8 tiles which are distributed into 4 locations spreading
    // over the smem. We need to find which of the 8 the current tile is.   
    // use tileid table to figure out    

   // for 2nd tile

  int idd1 = tileid[0][threadIdx.x];
  int id1 = (idd1 & (tw-1)) * 2 + (idd1 / tw) * out_w * 2;
  int idd2 = tileid[1][threadIdx.x];
  int id2 = (idd2 & (tw-1)) * 2 + (idd2 / tw) * out_w * 2;

  // unsigned short mask1 = 0x000F;
  unsigned int mask1 = get_mask<TW, TH>(idd1, tiles_dim_w, tiles_dim_h, tw, th, out_w, out_h);
  unsigned int mask2 = get_mask<TW, TH>(idd2, tiles_dim_w, tiles_dim_h, tw, th, out_w, out_h);
  
  // output transpose step
  int t=0;
  int acumm1, acumm2;
  // For transposing
  //acumm1 = access_s_out[Inx]; //* 4
  acumm1 = ((threadIdx.x&7)>>1)*34 + (threadIdx.x&1) + (threadIdx.x>>4)*2 + ((threadIdx.x>>3)&1)*8;
  acumm2 = acumm1+4;
                       
  int acumm4 = BN_p*16 ; //*4
  int idx  = threadIdx.y * BN_p;
  int idx2 = idx + BN_p*8; //(BN_p*2 *8)/2

  // For transformating
  int offset = BN_p *2; //*2/2
  int init = ( (threadIdx.y>>2)*BN_p*16 + (threadIdx.y&3)*(32+2) ) *2 + threadIdx.x;

  int c_glb_offset = out_h*out_w;
  // int c_tensor = blockIdx.z*c_glb_offset*BK + (blockIdx.y%tiles_dim)*2 + (blockIdx.y/tiles_dim)*out_w*2 + 
  //               blockIdx.x*BN + (threadIdx.x%16)*2+
  //               ((threadIdx.x/16)*16 + (threadIdx.y%4)*4 + threadIdx.y/4)*c_glb_offset;

  // int c_tile = blockIdx.x * tx  + blockIdx.y * in_w * ty; 
  // int c_tensor = c_tile + (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * in_w * 2 + 
  //               threadIdx.y*(in_h*in_w) - (in_w+1);

  int c_tensor = blockIdx.z*c_glb_offset*BK + blockIdx.x * TW  + blockIdx.y * out_w * TH +
                //  (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * out_w * 2 + 
                 ((threadIdx.x>>4)*16 + (threadIdx.y&3)*4 + (threadIdx.y>>2))*c_glb_offset;

  #pragma unroll                                  
  for(int round=0; round<4; round++){

    *( (float2*) (output_smem + idx + acumm1) )  = *(accumulator+t);
    *( (float2*) (output_smem + idx + acumm1 + 16) )  = *(accumulator+t+1); // float 4, t
    *( (float2*) (output_smem + idx + acumm2) )  = *(accumulator+t+2);
    *( (float2*) (output_smem + idx + acumm2 + 16) )  = *(accumulator+t+3); // float 4, t+1


    *( (float2*) (output_smem + idx2 + acumm1) ) = *(accumulator+t+32);
    *( (float2*) (output_smem + idx2 + acumm1 + 16) ) = *(accumulator+t+33); // float 4, t+16
    *( (float2*) (output_smem + idx2 + acumm2) ) = *(accumulator+t+34);
    *( (float2*) (output_smem + idx2 + acumm2 + 16) ) = *(accumulator+t+35); // float 4, t+17

    // the above 8 float2 will be consumed by theadIdx.y = [0,1,2,3]

    // the following 8 float2 will be consumed by theadIdx.y = [4,5,6,7]

    *( (float2*) (output_smem + idx + acumm4 + acumm1) )  = *(accumulator+t+4); 
    *( (float2*) (output_smem + idx + acumm4 + acumm1 + 16) )  = *(accumulator+t+5); // float 4, t+2
    *( (float2*) (output_smem + idx + acumm4 + acumm2) )  = *(accumulator+t+6);
    *( (float2*) (output_smem + idx + acumm4 + acumm2 + 16) )  = *(accumulator+t+7); // float 4, t+3

    *( (float2*) (output_smem + idx2 + acumm4 + acumm1) ) = *(accumulator+t+36);
    *( (float2*) (output_smem + idx2 + acumm4 + acumm1 + 16) ) = *(accumulator+t+37); // float 4, t+18
    *( (float2*) (output_smem + idx2 + acumm4 + acumm2) ) = *(accumulator+t+38);
    *( (float2*) (output_smem + idx2 + acumm4 + acumm2 + 16) ) = *(accumulator+t+39); // float 4, t+19
    
    

    t+=8;

    __syncthreads();

    
  
    
    for(int i=0; i<16; i++){
      C_tile[i].x = shared_mem[i*offset + init];
      C_tile[i].y = shared_mem[i*offset + init + 32];
    
    }
    

    // transform output tiles    
    transform_output_tile(C, C_tile, At, round, c_tensor, c_glb_offset, id1, id2, mask1, mask2, out_w);
    __syncthreads();
  }
}

#endif


#if __CUDA_ARCH__ >= CC_AMPERE

// Set of functions per row in Gw product
__device__ half f_row1(const half * __restrict__ Gw, int j){
    return Gw[j];
  }
  __device__ half f_row2(const half * __restrict__ Gw, int j){
    return __float2half(0.5f)*(Gw[j] + Gw[6+j] + Gw[3+j]);
  }
  __device__ half f_row3(const half * __restrict__ Gw, int j){
    return __float2half(0.5f)*(Gw[j] + Gw[6+j] - Gw[3+j]);
  }
  __device__ half f_row4(const half * __restrict__ Gw, int j){
    return Gw[6+j];
  }
  // Set of functions per column in GwGt product
  __device__ half f_col1(const half * __restrict__ Gw, int j){
    return Gw[j];
  }
  __device__ half f_col2(const half * __restrict__ Gw, int j){
    return __float2half(0.5f)*(Gw[j] + Gw[j+2] + Gw[j+1]);
  }
  __device__ half f_col3(const half * __restrict__ Gw, int j){
    return __float2half(0.5f)*(Gw[j] + Gw[j+2] - Gw[j+1]);
  }
  __device__ half f_col4(const half * __restrict__ Gw, int j){
    return Gw[j+2];
  }
  
  typedef half(*pointFunction_t)(const half *, int);

  template<int BN, int BK, int BC>
  __global__ void FX(const half * __restrict__ pInputs, half * __restrict__ pOutputs, int filt_k, 
                      int filt_c, int filt_h, int filt_w){
    int Inx = threadIdx.x, Iny = threadIdx.y;
    int TileX = blockIdx.x, TileY = blockIdx.y;
  
    int c_glb_offset = filt_k*filt_h*filt_w;
    int c_kernel = TileY*BC*c_glb_offset + TileX*BK + Iny*c_glb_offset + Inx;
    // int c_glb_offset_s = filt_c*4*4;
    int c_glb_offset_s = filt_c*filt_k;
    int c_kernel_s = TileY*BC + TileX*BK*filt_c + Iny + Inx * filt_c;
  
    half Gw[21]; //9+12. In registers
    half *Gw_buffer = Gw+9;
  
    pointFunction_t func1[4] = {f_row1, f_row2, f_row3, f_row4};
    pointFunction_t func2[4] = {f_col1, f_col2, f_col3, f_col4};
  
    for(int bk=0; bk<BK; bk+=blockDim.x){
      for(int i=0; i<9; i++){
        Gw[i] = pInputs[c_kernel + i*filt_k];
      }
  
      int aux;
      for(int i=0; i<4; i++){
        aux = i*3;
        for(int j=0; j<3; j++){
          Gw_buffer[j+aux] = (*func1[i])(Gw, j);
        }
      }
  
      int aux2;
      for(int i=0; i<4; i++){
        aux = i*3; aux2 = i<<2;
        for(int j=0; j<4; j++){
          pOutputs[c_kernel_s+aux2*c_glb_offset_s+j*c_glb_offset_s] = (*func2[j])(Gw_buffer, aux);
        }
      }
  
      c_kernel   += blockDim.x;
      c_kernel_s += blockDim.x*filt_c;
    }
  }

#else

template <typename T>
  static __device__ T __forceinline__ fx_const (float val) {
      return static_cast<T>(val);
  }
template <>
  __device__ half __forceinline__ fx_const<half>(float val) {
      return __float2half(val);
  }

// Set of functions per row in Gw product
template <typename T>
  __device__ T f_row1(const T * __restrict__ G, int j){
    return G[j];
  }
template <typename T>
  __device__ T f_row2(const T * __restrict__ G, int j){
    return fx_const<T>(0.5f)*(G[j] + G[6+j] + G[3+j]);
  }
template <typename T>
  __device__ T f_row3(const T * __restrict__ G, int j){
    return fx_const<T>(0.5f)*(G[j] + G[6+j] - G[3+j]);
  }
template <typename T>
  __device__ T f_row4(const T * __restrict__ G, int j){
    return G[6+j];
  }

  // Set of functions per column in GwGt product
template <typename T>  
  __device__ T f_col1(const T * __restrict__ G, int j){
    return G[j];
  }
template <typename T>
  __device__ T f_col2(const T * __restrict__ G, int j){
    return fx_const<T>(0.5f)*(G[j] + G[j+2] + G[j+1]);
  }
template <typename T>
  __device__ T f_col3(const T * __restrict__ G, int j){
    return fx_const<T>(0.5f)*(G[j] + G[j+2] - G[j+1]);
  }
template <typename T>
  __device__ T f_col4(const T * __restrict__ G, int j){
    return G[j+2];
  }

  template <typename T>
  static __device__ __forceinline__ float t2f32(T val) {
      return (float) val;
  }

  template <>
  __device__ float __forceinline__ t2f32<half>(half val) {
      return __half2float(val);
  }

  

  template<typename T, int BN, int BK, int BC>
  __global__ void FX(const T * __restrict__ pInputs, T * __restrict__ pOutputs, int filt_k, 
                      int filt_c, int filt_h, int filt_w){

  typedef T(*pointFunction_t)(const T *, int);

    // assumes KCHW layout
    int Inx = threadIdx.x, Iny = threadIdx.y;
    int TileX = blockIdx.x, TileY = blockIdx.y;
  
    // int c_glb_offset = filt_k*filt_h*filt_w;
    // int c_kernel = TileY*BC*c_glb_offset + TileX*BK + Iny*c_glb_offset + Inx;
    int c_glb_offset = filt_h*filt_w;
    // int c_kernel = TileY*BC*c_glb_offset + TileX*BK*filt_c*c_glb_offset + Iny*c_glb_offset+ Inx*filt_c*c_glb_offset;
    int c_kernel = (TileY*BC + (TileX*BK+Inx)*filt_c + Iny)*c_glb_offset;
    int c_glb_offset_s = filt_k*4*4;
    int c_kernel_s = TileY*BC*c_glb_offset_s + TileX*BK + Iny*c_glb_offset_s + Inx;
  
    T Gw[21]; //9+12. In registers
    T *Gw_buffer = Gw+9;
  
    pointFunction_t func1[4] = {f_row1, f_row2, f_row3, f_row4};
    pointFunction_t func2[4] = {f_col1, f_col2, f_col3, f_col4};
  
    for(int bk=0; bk<BK; bk+=blockDim.x){      
      for(int i=0; i<9; i++){
        // Gw[i] = t2f32(pInputs[c_kernel + i]);
        Gw[i] = pInputs[c_kernel + i];
      }      
  
      int aux;
      for(int i=0; i<4; i++){
        aux = i*3;
        for(int j=0; j<3; j++){
          Gw_buffer[j+aux] = (*func1[i])(Gw, j);
        }
      }

      int aux2;
      for(int i=0; i<4; i++){
        aux = i*3; aux2 = i<<2;
        for(int j=0; j<4; j++){
          pOutputs[c_kernel_s+aux2*filt_k+j*filt_k] = (*func2[j])(Gw_buffer, aux);
        }
      }
  
      c_kernel   += blockDim.x*(filt_c*c_glb_offset);
      c_kernel_s += blockDim.x;
    }
  }

#endif


#define d(input, i, j) ( input[(i<<2) + (j)] )
#if __CUDA_ARCH__ >= CC_AMPERE

// smem layout for input tile
// ___________32T(C0)______32T(C1)____... _____32T(C15) E0
// ___________32T(C0)______32T(C1)____... _____32T(C15) E1
// .....
// .....
// ___________32T(C0)______32T(C1)____... _____32T(C15) E15

__device__ __forceinline__ void load_and_transform_input_tile(half *Btd, half *pOutputs){

  half2 workspace[3]; 
  int c_offset = 4*(64+PADDING);
  int c_tensor = (threadIdx.x/8)*(64+PADDING) + (threadIdx.x%8)*4 + (threadIdx.y/4)*32 + (threadIdx.y%4);
  // int offset = 0;
  half2 *ptr = (half2 *)pOutputs;
  half2 *Btd2 = (half2 *)Btd;
  // for(int k=0; k<2; k++){
  #pragma unroll
  for(int j=0; j<4; j++){
    workspace[0] = Btd2[j];
    workspace[1] = Btd2[j+4];
    workspace[2] = Btd2[j+8];

    Btd2[j]    = workspace[0] - workspace[2];
    Btd2[j+4]  = workspace[1] + workspace[2];
    Btd2[j+8]  = workspace[2] - workspace[1];
    Btd2[j+12] = workspace[1] - Btd2[j+12];
  }      
    // int offset1 = ((threadIdx.x % 2) ^ k) * (BN+PADDING);
  #pragma unroll
  for(int i=0; i<4; i++){ // prefetch 1 input tile/thread
    ptr[c_tensor+i*c_offset*4] = d(Btd2, i, 0) - d(Btd2, i, 2);  
    ptr[c_tensor+i*c_offset*4+c_offset] = d(Btd2, i, 1) + d(Btd2, i, 2);
    ptr[c_tensor+i*c_offset*4+2*c_offset] = d(Btd2, i, 2) - d(Btd2, i, 1);
    ptr[c_tensor+i*c_offset*4+3*c_offset] = d(Btd2, i, 1) - d(Btd2, i, 3);
  }     
}

#else

template <int BN, int BK, int BC>
__device__ __forceinline__ void load_and_transform_input_tile(float *Btd, float * __restrict__ pOutputs){

  float workspace[3]; 

  #pragma unroll
  for(int j=0; j<4; j++){
    workspace[0] = Btd[j];
    workspace[1] = Btd[j+4];
    workspace[2] = Btd[j+8];

    Btd[j]    = workspace[0] - workspace[2];
    Btd[j+4]  = workspace[1] + workspace[2];
    Btd[j+8]  = workspace[2] - workspace[1];
    Btd[j+12] = workspace[1] - Btd[j+12];
  }
  
  int c_offset = BC*BN;
  int c_tensor = threadIdx.y*BN + threadIdx.x;
  
  #pragma unroll
  for(int i=0; i<4; i++){ // prefetch 1 input tile/thread
    pOutputs[c_tensor+i*c_offset*4] = d(Btd, i, 0) - d(Btd, i, 2);  
    pOutputs[c_tensor+i*c_offset*4+c_offset] = d(Btd, i, 1) + d(Btd, i, 2);
    pOutputs[c_tensor+i*c_offset*4+2*c_offset] = d(Btd, i, 2) - d(Btd, i, 1);
    pOutputs[c_tensor+i*c_offset*4+3*c_offset] = d(Btd, i, 1) - d(Btd, i, 3);    
  }     

}

#endif

template <int BN, int BK, int BC>
__device__ __forceinline__ void load_filter_tile(float *tiles, float * __restrict__ pOutputs, 
                                int filt_c, int filt_k){
 
  int c_tensor_s = threadIdx.y*BK + threadIdx.x;
  int c_offset_s = BK*BC;
  // if(threadIdx.y >= BC) return;
  
  // each thread in row 0 puts its first element of 1st filter tile(loaded by the thread) in smem
  // taking 32 slots 
  // then puts its first element of 2nd filter tile immediately after, taking another 32 slots
  // then followed by threads in row 1, 2.. until 7

  // Note the next element is BK*BC (8*64) slots away, then another BK*BC ....
  // for every 64 values, the first 32 belongs to filter tile 1, the next 32 for filter tile 2 

  for(int k=0; k<2; k++){ // prefetch 2 filter tiles/thread
    for(int i=0; i<4; i++){
      #pragma unroll
      for(int j=0; j<4; j++){
        pOutputs[c_tensor_s + i*c_offset_s*4 + j*c_offset_s] = tiles[k*16 + i*4 + j];
      }
    }
    // 2nd tile right behind the 1st?
    c_tensor_s += BN; // BN has nothing to do with input tiles
  }
  
}

template<typename T, int BN, int BK>
__device__ __forceinline__ void prefetch_filter_tile(const T * __restrict__ pInputs, float * __restrict__ tiles, int filt_k){

  int c_tensor = blockIdx.z*BK + threadIdx.y*(filt_k<<4) + threadIdx.x; // Iny*filt_k*4*4
  // each threadIdx.y corresponds to one channel; there are 8 different threadIdx.y so 8 channels 
  
  //each thread (32 threads in x direction) loads 2 kernel tiles (32 in K direction apart)
  // save the two tiles in a float[32] register, float[16] for each  
  
  int acumm;
  #pragma unroll  
  for(int i=0; i<4; i++){
      acumm = i*(filt_k<<2);
      #pragma unroll
      for(int j=0; j<4; j++){
          tiles[(i<<2) + j] = t2f32(pInputs[acumm + j*filt_k + c_tensor]);
          tiles[16 + (i<<2) + j] = t2f32(pInputs[acumm + j*filt_k + c_tensor+BN]);
      }
  }
}

#if __CUDA_ARCH__ >= CC_AMPERE

template<int TW, int TH>
__device__ __forceinline__ void prefetch_input_tile(const half *pInputs, half *tile, int in_h, 
                       int in_w, int tw, int th, unsigned short mask){
  
  // load two input tiles per thread
  // int tx = in_w / gridDim.x, ty = in_h / gridDim.y;  
  int c_offset = in_h*in_w; 
  int c_tile = blockIdx.x * TW  + blockIdx.y * in_w * TH; 
  int c_tensor = c_tile + (threadIdx.x % tw) * 2 + (threadIdx.x / tw) * in_w * 2 + 
                threadIdx.y*2*c_offset - (in_w+1);

      // + threadIdx.y*(in_h*in_w) + (in_w+1);
  // if(threadIdx.x/in_n != 0){
  //   printf(" %d, %d, %d, %d \n", blockIdx.x, blockIdx.y,  threadIdx.x, threadIdx.y);
  // }
  int acumm,x;
  //short x1,x2;     

  // if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 
  //     && threadIdx.x == 31 && threadIdx.y == 0){
  //         printf("X, %hu \n", mask);   
  //   }
           
  if(mask==0xFFFF){
    #pragma unroll
    for(int i=0; i<4; i++){
      acumm = i*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        x = (i<<2) + j;
        tile[2*x + 0] = pInputs[acumm + j + c_tensor]; //1st channel
        tile[2*x + 1] = pInputs[acumm + j + c_tensor + c_offset];//2nd channel
        // if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 
        //   && threadIdx.x == 31 && threadIdx.y == 0){
        //      printf("A, %d, %d, %d, %f, %d\n", i, j, acumm+j, tile[(i<<2) + j],acumm + j + c_tensor);   
        // }
      }
    }

  } else {
    for(int i=0; i<4; i++){
      acumm = i*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        x = (i<<2) + j;
        tile[2*x+0] = 0.f;
        tile[2*x+1] = 0.f;
        if(mask&(1<<x)){
          tile[2*x + 0]=pInputs[acumm + j + c_tensor];
          tile[2*x + 1]=pInputs[acumm + j + c_tensor + c_offset];
        }
          // if(blockIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0 
          //   && threadIdx.x == 0 && threadIdx.y == 0){
          //      printf("B, %d, %d, %d, %d, %d, %d, %hu, %s, %f, %f, %d, %d\n", i, j, x, acumm+j, c_tensor, c_offset, mask, mask&(1<<x)?"t":"f",
          //       __half2float(tile[2*x+0]), __half2float(tile[2*x+1]), acumm + j + c_tensor, acumm + j + c_tensor + c_offset);   
          // }        
      }
    }
  }
}

// smem layout for transformed filter weights 
// ___________16C(K0)______16C(K1)____... _____16C(K31) E0
// ___________16C(K0)______16C(K1)____... _____16C(K31) E1
// .....
// .....
// ___________16C(K0)______16C(K1)____... _____16C(K31) E15
// -- B_Frag1
template<int BK, int BC>
__device__ __forceinline__ void prefetch_filter_tile_async(const half *pInputs, half *smem, int filt_c, int filt_k, int ko){

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int c_offset = filt_c*filt_k;
  int c_tensor = blockIdx.z*BK*filt_c + ty*c_offset + (tx/2)*filt_c + (tx%2)*8 + ko * 32 * filt_c; // Iny*filt_k*4*4
  int c_tensor_s = ty*(BC*32) + tx * 8;

  // each threadIdx.y corresponds to 2 channels; there are 8 different threadIdx.y so 16 channels 
  // each threadx load 16 filters in K 
  //each thread (32 threads in x direction) loads 4 kernel tiles (2 for each channel and 32 in K direction apart)
  
  
  // int tid = ty*32+tx;
  // int cid = ((ty*32+tx) % 128) / 8;   
  // int kid = tx % 8;
  // tx = tx % 16;
  // int eid = (tx / 4) * (filt_k<<2) + (tx % 4) * filt_k;  


   // swizzling, but not necessary here
  // int c = tx % 8;
  // int s = tx / 8;
  // int row =  (c & 1) | ((c >> 1) & 2);
  // int bank = ((c << 1) & 4) | s ^ row;
  #pragma unroll
  for(int k = 0; k < 2; k++){ // each cp.async can load 16 bytes = 8 halfs, we need to load 16 halfs
    // load 8 tile elements, each ty loads 16Kx16C 
    // each tx loads 8 halfs (16 bytes)
    void *ptr = (void *)(smem + k*8*(BC*32) + c_tensor_s);
    // void *ptr = (void *)(smem + k*8*(BC*16) + ty*(BC*16) + (row * 8 + bank) * 8);
    unsigned int smem_ptr;

    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                "l"(&pInputs[c_tensor + k * 8 * c_offset]),
                "n"(16));
    
    ptr = (void *)(smem + k*8*(BC*32) + c_tensor_s + BC*16);    

    asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
        "%0, smem_ptr; }\n"
        : "=r"(smem_ptr)
        : "l"(ptr));

    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                "l"(&pInputs[c_tensor + 16*filt_c + k * 8 * c_offset]),
                "n"(16));
  }
}

#else

template<int TW, int TH>
__device__ __forceinline__ void prefetch_input_tile(const float * __restrict__ pInputs, float *tile, int in_h, 
                       int in_w, int tw, int th, unsigned short mask){
  
  // each thread loads two input tiles to fill a half2 buffer   
  int c_offset = in_h*in_w;
  int c_tile = blockIdx.x * TW  + blockIdx.y * in_w * TH; 
  int c_tensor = c_tile + ((threadIdx.x & (tw-1)) << 1) + (threadIdx.x / tw ) * (in_w << 1) + 
                threadIdx.y*c_offset - (in_w+1);
  
  int acumm,x;
  
           
  if(mask==0xFFFF){
    #pragma unroll
    for(int i=0; i<4; i++){
      acumm = i*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        tile[(i<<2) + j] = pInputs[acumm + j + c_tensor];
      }
    }

  } else {
    for(int i=0; i<4; i++){
      acumm = i*in_w;   
      #pragma unroll
      for(int j=0; j<4; j++){
        x = (i<<2) + j;
        tile[x] = 0.f;
        if(mask&(1<<x))
          tile[x] = pInputs[acumm + j + c_tensor];
      }
    }
  }
}

#endif 

// this remains the same as 32x64x8 case
__device__  __forceinline__ void prefetch_filter_frag(float4 *filter_frag, float4 *B_frag, int f_frag_offset, int offset1, int offset2){

  // from the land id table, 32 threads are actually divided into 2 big groups
  // first 16 and the last 16
  // each big group further divides into 8 pairs
  // threads within each pair load the same filter value   

  // the 2nd group just duplicates the 1st

  *((float4*) (filter_frag))     = *(B_frag + offset1); 
  *((float4*) (filter_frag + 1)) = *(B_frag + offset2); // + 32 floats (8 float4)

 // the next 8 floats are for the next next tile element 
  *((float4*) (filter_frag + 2)) = *(B_frag + f_frag_offset + offset1);
  *((float4*) (filter_frag + 3)) = *(B_frag + f_frag_offset + offset2);
}


__device__  __forceinline__ void prefetch_input_frag(float4* input_frag, float4 *A_frag, int frag_offset, int offset1, int offset2){  

  *((float4*) (input_frag))     = *(A_frag + offset1); //ld_shared(A_frag + offset1);
  *((float4*) (input_frag + 1)) = *(A_frag + offset2);

  *((float4*) (input_frag + 2)) = *(A_frag + frag_offset + offset1);
  *((float4*) (input_frag + 3)) = *(A_frag + frag_offset + offset2); //3=2+1
}

#if __CUDA_ARCH__ >= CC_AMPERE

__device__ void loadFragA(unsigned int *frag, half *smem, int ki)
{
    // load 32x16    
    // we use mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 to do 16x16x16 mm,
    // so we need to fill 2 16x8 A matrices;
    // for each 16x8 matrix A, each thread loads 4 elements (a0, a1, a2, a3) and they are
    // row 0, col 0,1 and row 8 col 0,1
    // so from the point of view the 16x16 matrix, all 8 elemets for thread 0 are
    // row/tile 0, col/channel (0, 1, 8, 9) and row/tile 8, col/chennel (0, 1, 8, 9)
    // to avoid bank conflicts, we can make threads in a warp coordinate the loading by using 
    // specially designed offsets
    // T0, T4, T8,..T28 all 8 threads load the same channels (0 and 1) and successive super tiles,
    // which results in bank conflicts. We let them load in an interleaving way:
    // first (0, 1, 0, 1, 0, 1, 0, 1)
    // then  (1, 0, 1, 0, 1, 0, 1, 0)
    // so in each round, successive threads load different channels and avoid conflicts
    // similarly for the otehr 24 threads
    int tx = threadIdx.x;
    // int ty = threadIdx.y;
    // half2 *fragA = (half2 *)frag;
    // half2 *input = (half2 *)smem;
    unsigned int *fragA = frag;
    unsigned int *input = (unsigned int *)smem;
    #pragma unroll
    for (int i = 0; i < 2; ++i){        
      // for (int k = 0; k < 2; ++k){              
        //                      |   channel          |   |     super tile      |
        fragA[i*4+0] = input[i*2*(64+PADDING) +                     tx];
        fragA[i*4+1] = input[i*2*(64+PADDING) +                32 + tx];
        fragA[i*4+2] = input[i*2*(64+PADDING) + (64+PADDING)      + tx];
        fragA[i*4+3] = input[i*2*(64+PADDING) + (64+PADDING) + 32 + tx];
        
    }
}

template<int BN, int BC>
__device__ void loadFragB(unsigned int *frag, half *smem, int ki)
{
    // // load 16x16    
    // // for (int i = 0; i < 2; ++i)
    // // {      
    //   nvcuda::wmma::load_matrix_sync(frag, smem + threadIdx.y*(wmmaN*wmmaK)+ ki * 8 *(wmmaN*wmmaK) , 16);
    // // }
    // load 16x16    
    // we use mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 to do 16x16x16 mm,
    // so we need to fill 4 8x8 B matrices;
    // for each 8x8 matrix B, thread 0 loads 2 elements (a0, a1) and they are
    // row 0,1 col 0
    // so from the point of view of the 16x16 matrix, all 8 elements for thread 0 are
    // row/channel (0, 1) col/K 0 , row/channel (8, 9), col/K 0
    // row/channel (0, 1) col/K 8 , row/channel (8, 9), col/K 8
    // to avoid bank conflicts, we can make threads in a warp coordinate the loading by using 
    // specially designed offsets
    // T0, T4, T8,..T28 all 8 threads load the same channels (0 and 1) and successive filters in K,
    // which results in bank conflicts. We let them load in an interleaving way:
    // first (0, 1, 0, 1, 0, 1, 0, 1)
    // then  (1, 0, 1, 0, 1, 0, 1, 0)
    // so in each round, successive threads load different channels and avoid conflicts
    // similarly for the other 24 threads
    // note the code is very similar to loadFragA
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // half *fragB = (half *)frag;
    unsigned int * ptr;
    #pragma unroll
    for (int k = 0; k < 2; ++k){
      //                  | tile element  |   |   channel          |  |       K      |
      // fragB[k*4+0] = smem[(ki*8+ty)*(BC*BC) + BC*access_s[0][tx]     + tx / 4 + k * 8];
      // fragB[k*4+1] = smem[(ki*8+ty)*(BC*BC) + BC*access_s[1][tx]     + tx / 4 + k * 8];
      // fragB[k*4+2] = smem[(ki*8+ty)*(BC*BC) + BC*(access_s[0][tx]+8) + tx / 4 + k * 8];
      // fragB[k*4+3] = smem[(ki*8+ty)*(BC*BC) + BC*(access_s[1][tx]+8) + tx / 4 + k * 8];
      //                                             | tile element  |   |        K          |   | channel   |
      // ptr =  reinterpret_cast<unsigned int *>(smem + (ki*8+ty)*(BC*BC) + BC * (tx / 4 + k * 8) +  (tx%4)*2    );
      ptr =  reinterpret_cast<unsigned int *>(smem + (ki*8+ty)*(BC*BN) + BC * (tx / 4 + k * 8) +  access_t[0][tx] );
      frag[k*2+0] = ptr[0];
      // ptr =  reinterpret_cast<unsigned int *>(smem + (ki*8+ty)*(BC*BC) + BC * (tx / 4 + k * 8) +  (tx%4)*2 + 8);
      ptr =  reinterpret_cast<unsigned int *>(smem + (ki*8+ty)*(BC*BN) + BC * (tx / 4 + k * 8) +  access_t[1][tx]);
      frag[k*2+1] = ptr[0];
    }
    #pragma unroll
    for (int k = 0; k < 2; ++k){
      //                                             | tile element  |   |        K          |   | channel   |
      // ptr =  reinterpret_cast<unsigned int *>(smem + (ki*8+ty)*(BC*BC) + BC * (tx / 4 + k * 8) +  (tx%4)*2    );
      ptr =  reinterpret_cast<unsigned int *>(smem + (ki*8+ty)*(BC*BN) + BC*16 + BC * (tx / 4 + k * 8) +  access_t[0][tx] );
      frag[4+k*2+0] = ptr[0];
      // ptr =  reinterpret_cast<unsigned int *>(smem + (ki*8+ty)*(BC*BC) + BC * (tx / 4 + k * 8) +  (tx%4)*2 + 8);
      ptr =  reinterpret_cast<unsigned int *>(smem + (ki*8+ty)*(BC*BN) + BC*16 + BC * (tx / 4 + k * 8) +  access_t[1][tx]);
      frag[4+k*2+1] = ptr[0];
    }    
}


// Fragments layouts for A and B
// used by mmaSync
// each mma.sync.aligned.m16n8k8 takes 2 FragA and 1 FragB
//                FragA
//   ______________________________
//  |              |               |
//  |      0       |        1      |
//  |              |               |
//  |______________|_______________|
//  |              |               |
//  |              |               |
//  |      2       |        3      |
//  |              |               |
//  |______________|_______________|


//                FragB
//   ______________________________
//  |              |               |
//  |      0       |        2      |
//  |              |               |
//  |______________|_______________|
//  |              |               |
//  |              |               |
//  |      1       |        3      |
//  |              |               |
//  |______________|_______________|


__device__ void mmaSync(unsigned int *fragA, unsigned int *fragB, float *accum)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
        : "r"(fragA[0]), "r"(fragA[2]),
          "r"(fragB[0]),
          "f"(accum[0]), "f"(accum[1]), "f"(accum[4]), "f"(accum[5]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
        : "r"(fragA[1]), "r"(fragA[3]),
          "r"(fragB[1]),
          "f"(accum[0]), "f"(accum[1]), "f"(accum[4]), "f"(accum[5]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
        : "r"(fragA[0]), "r"(fragA[2]),
          "r"(fragB[2]),
          "f"(accum[2]), "f"(accum[3]), "f"(accum[6]), "f"(accum[7]));

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5},"
        "{%6},"
        "{%7,  %8,  %9,  %10};\n"
        : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
        : "r"(fragA[1]), "r"(fragA[3]),
          "r"(fragB[3]),
          "f"(accum[2]), "f"(accum[3]), "f"(accum[6]), "f"(accum[7]));
}

template<int TW, int TH, int BN, int BK, int BC>
__global__ void Winograd_kernel(half *A, half *B, float *C,
                    int tiles_dim_w, int tiles_dim_h,
                    int in_c, int in_h, int in_w,
                    int tile_size, int X, int Y,
                    int filt_k, int filt_c,
                    int out_c,
                    int out_h, int out_w){

  extern __shared__ unsigned char shared_mem[];
  half *input_smem  = reinterpret_cast<half *>(shared_mem);
  // half *filter_smem = input_smem + 16*BC*BN;

  unsigned short m = 0xFFFF;
  // if((blockIdx.y/tiles_dim)==0)   m&=0xFFF0;
  // if((blockIdx.y/tiles_dim)==(tiles_dim-1)) m &= (!(in_w%2))?(0x0FFF):(0x00FF);
  // if(!((blockIdx.y+1)%tiles_dim)) m &= (!(in_w%2))?(0x7777):(0x3333);
  // if(!((blockIdx.y)%tiles_dim))   m&=0xeeee;

  if(blockIdx.y==0 && (threadIdx.x / X) == 0)   m &= 0xFFF0;  // pad top row
  if(tiles_dim_w % X == 0 && tiles_dim_h % Y == 0){
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == Y-1) m &= (!(in_h%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x % X) == X-1) m &= (!(in_w%2))?(0x7777):(0x3333); // pad right col or right 2 cols
  }else if(tiles_dim_w % X == 0){
    int k = in_h % TH; 
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 4*k1 tiles
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x % X) == X-1) m &= (!(in_w%2))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == k1-1) m &= (!(k%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X > k1-1) m &= 0x0; //pad all zeros since this tile does not exist
  }else if(tiles_dim_h % Y == 0){
    int k = in_w % TW;   
    int k1 =  k % 2 ? (k+1)/2 : k/2; // there could be 8*k1 tiles
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == Y-1) m &= (!(in_h%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X == k1-1) m &= (!(k%2))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X > k1-1) m &= 0x0; //pad all zeros since this tile does not exist 
  }else{
    int kh = in_h % TH; 
    int kw = in_w % TW;   
    int kh1 =  kh % 2 ? (kh+1)/2 : kh/2; // there could be kh1*kw1 tiles
    int kw1 =  kw % 2 ? (kw+1)/2 : kw/2; 
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == kh1-1) m &= (!(kh%2))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X > kh1-1) m &= 0x0; //pad all zeros since this tile does not exist
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X == kw1-1) m &= (!(kw%2))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.x==gridDim.x-1 && threadIdx.x % X > kw1-1) m &= 0x0; //pad all zeros since this tile does not exist
  }  
  if(blockIdx.x==0 && (threadIdx.x % X) == 0)   m &=0xeeee;  // pad left col
  
  half img_tile[32]; // Prefetch input from GMEM
  // half filter_tile[64]; // Prefetch filter from GMEM

  
  half *A_frag; // Input data pointer

  // half *B_frag; // Filter data pointer  
  // half *B_frag1 =  filter_smem;
  half *B_frag1 = input_smem + 16*4*(64+PADDING)*2;
  half *B_frag2 =  B_frag1 + 16*BC*BK/2;  // 16*BC*BK/4 = 4*BC*BK
  

  // unsigned int FragA[2 * BN / wmmaM * 4];      //  4 int32 = 8 half
  unsigned int *FragA = (unsigned int *)img_tile;      //  4 int32 = 8 half
  unsigned int FragB[2 * BN / wmmaN * 4];      // 4 int32 = 8 half
  float Accum[2 * BN / wmmaM * BK / wmmaN * 8] = {0.0}; // [4, 2, 8]

  prefetch_input_tile<TW, TH>(A, img_tile, in_h, in_w, X, Y, m);
  prefetch_filter_tile_async<BK, BC>(B, B_frag1, filt_c, filt_k, 0);  
  asm volatile("cp.async.commit_group;\n" ::);
  // prefetch_filter_tile_async(B, B_frag2, filt_c, filt_k, 1);  
  // asm volatile("cp.async.commit_group;\n" ::);
  // prefetch_filter_tile_async(B, B_frag3, filt_c, filt_k, 2);  
  // asm volatile("cp.async.commit_group;\n" ::);
  // int ko = 0;
  

  // Mainloop - iterates over the entire K dimension - not unrolled

  // wee need to do 16-batched 32x16x64 MM, each wmma will do 16x16x16 so 
  // we need to do 16 2x4 wmmas's 
  // we allocate 2 FragA and 4 FragB and 16 Accum, then in a loop of 2 iterations 
  // reuse 2 FragA and 4 FragB
  //    
  for(int iter=0; iter<in_c-BC; iter+=BC){ // Current iteration

   

    load_and_transform_input_tile(img_tile, input_smem);
    // load_filter_tile(filter_tile, filter_smem, filt_c, filt_k);

    __syncthreads();

    for(int k = 0; k < 2; k++){
      // A_frag = input_smem  + threadIdx.y*(BN+PADDING)*BC + k*8*(BN+PADDING)*BC;
      A_frag = input_smem  + threadIdx.y*(64+PADDING)*4*2 + k*8*(64+PADDING)*4*2;
      // B_frag = filter_smem + threadIdx.y*BC*BK + k*8*BC*BK;      
      loadFragA(FragA + k * BN / wmmaM * 4, A_frag, k);
    }
  

    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();   
    // now both input and filter tiles are in smem, we can load wmma frags and do wmma computation  
    // if(iter<(in_c-BC)){ // ???should there be a if here
    prefetch_filter_tile_async<BK,BC>(B, B_frag2, filt_c, filt_k, 1);  
    asm volatile("cp.async.commit_group;\n" ::);

    for(int k = 0; k < 2; k++){
      loadFragB(FragB + k * BN / wmmaN * 4, B_frag1, k);
    }
    
    for(int k = 0; k < 2; k++){  
      for(int mii = 0; mii < BN / wmmaM; mii++){
        for(int nii = 0; nii < BN / wmmaN; nii++){
            // 16x16x16 for each wmma
             mmaSync(&FragA[k * BN / wmmaM * 4 + mii * 4], &FragB[k * BN / wmmaN * 4 + nii * 4], &Accum[k*(BN / wmmaM * BK / wmmaN) * 8 + mii * (BK / wmmaN) * 8 + nii * 8]);
        }
      }     
    }

    // __syncthreads();
    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();   

    B += BC; 

    prefetch_filter_tile_async<BK,BC>(B, B_frag1, filt_c, filt_k, 0);  
    asm volatile("cp.async.commit_group;\n" ::);

    for(int k = 0; k < 2; k++){
      loadFragB(FragB + k * BN / wmmaN * 4, B_frag2, k);
    }
    for(int k = 0; k < 2; k++){
      for(int mii = 0; mii < BN / wmmaM; mii++){
        for(int nii = 0; nii < BN / wmmaN; nii++){
            // 16x16x16 for each wmma
            mmaSync(&FragA[k * BN / wmmaM * 4 + mii * 4], &FragB[k * BN / wmmaN * 4 + nii * 4], &Accum[k*(BN / wmmaM * BK / wmmaN) * 8 + mii * (BK / wmmaN) * 8 + nii * 8 + 16]);
        }
      }     
    }
    
    A += BC*in_w*in_h;
    // B += filt_k*BC*4*4;

    // if(iter<(in_c-BC)){
    prefetch_input_tile<TW,TH>(A, img_tile, in_h, in_w, X, Y, m);
      // prefetch_filter_tile(B, filter_tile, filt_k);

    __syncthreads();
  }

  // last iteration 
  load_and_transform_input_tile(img_tile, input_smem);  
  __syncthreads(); 

  for(int k = 0; k < 2; k++){
    // A_frag = input_smem  + threadIdx.y*(BN+PADDING)*BC + k*8*(BN+PADDING)*BC;      
    A_frag = input_smem  + threadIdx.y*(64+PADDING)*4*2 + k*8*(64+PADDING)*4*2;
    loadFragA(FragA + k * BN / wmmaM * 4, A_frag, k);
  }

  asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
  __syncthreads();   
    
  prefetch_filter_tile_async<BK,BC>(B, B_frag2, filt_c, filt_k, 1);  
  asm volatile("cp.async.commit_group;\n" ::);  
  
  for(int k = 0; k < 2; k++){
    loadFragB(FragB + k * BN / wmmaN * 4, B_frag1, k);
  }
    
  for(int k = 0; k < 2; k++){  
    for(int mii = 0; mii < BN / wmmaM; mii++){
      for(int nii = 0; nii < BN / wmmaN; nii++){
          // 16x16x16 for each wmma
          mmaSync(&FragA[k * BN / wmmaM * 4 + mii * 4], &FragB[k * BN / wmmaN * 4 + nii * 4], &Accum[k*(BN / wmmaM * BK / wmmaN) * 8 + mii * (BK / wmmaN) * 8 + nii * 8]);
      }
    }     
  }

  

  asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
  __syncthreads();   

  for(int k = 0; k < 2; k++){
    loadFragB(FragB + k * BN / wmmaN * 4, B_frag2, k);
  }

  for(int k = 0; k < 2; k++){
    for(int mii = 0; mii < BN / wmmaM; mii++){
      for(int nii = 0; nii < BN / wmmaN; nii++){
          // 16x16x16 for each wmma
          mmaSync(&FragA[k * BN / wmmaM * 4 + mii * 4], &FragB[k * BN / wmmaN * 4 + nii * 4], &Accum[k*(BN / wmmaM * BK / wmmaN) * 8 + mii * (BK / wmmaN) * 8 + nii * 8 + 16]);
      }    
    }     
  }

  // Transpose, transform and store accumulated result
  store_output_tile<TW,TH,BN,BK,BC>(Accum, shared_mem, C, out_h, out_w, tiles_dim_w, tiles_dim_h, X, Y);
                     
}

#else

template<typename T, int TW, int TH, int BN, int BK, int BC>
__global__ void Winograd_kernel(const float *A, const T *B, float *C,
                    int tiles_dim_w, int tiles_dim_h,
                    int in_c, int in_h, int in_w,
                    int tile_size, int X, int Y,
                    int filt_k, int filt_c,
                    int out_c,
                    int tile_2d_s, int out_h, int out_w){

  extern __shared__ float shared_mem[];
  float *input_smem  = (float*)shared_mem;
  float *filter_smem = (float*)&shared_mem[16*BC*BN];

  unsigned int m = 0xFFFF;  

  if(blockIdx.y==0 && (threadIdx.x / X) == 0)   m &= 0xFFF0;  // pad top row
  if(tiles_dim_w & (X-1) == 0 && (tiles_dim_h & (Y-1)) == 0){
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == Y-1) m &= (!(in_h&1))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x & (X-1)) == X-1) m &= (!(in_w&1))?(0x7777):(0x3333); // pad right col or right 2 cols
  }else if((tiles_dim_w & (X-1)) == 0){
    int k = in_h & (TH-1); 
    int k1 =  k & 1 ? (k+1)>>1 : (k>>1); // there could be 4*k1 tiles
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x & (X-1)) == X-1) m &= (!(in_w&1))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == k1-1) m &= (!(k&1))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X > k1-1) m &= 0x0; //pad all zeros since this tile does not exist
  }else if((tiles_dim_h & (Y-1)) == 0){
    int k = in_w & (TW-1);   
    int k1 =  k & 1 ? (k+1)>>1 : k>>1; // there could be 8*k1 tiles
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == Y-1) m &= (!(in_h&1))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x & (X-1)) == k1-1) m &= (!(k&1))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x & (X-1)) > k1-1) m &= 0x0; //pad all zeros since this tile does not exist 
  }else{
    int kh = in_h & (TH-1); 
    int kw = in_w & (TW-1);   
    int kh1 =  kh & 1 ? (kh+1)>>1 : kh>>1; // there could be kh1*kw1 tiles
    int kw1 =  kw & 1 ? (kw+1)>>1 : kw>>1; 
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X == kh1-1) m &= (!(kh&1))?(0x0FFF):(0x00FF); //pad bottom row or bottom 2 rows
    if(blockIdx.y==gridDim.y-1 && threadIdx.x / X > kh1-1) m &= 0x0; //pad all zeros since this tile does not exist
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x & (X-1)) == kw1-1) m &= (!(kw&1))?(0x7777):(0x3333); // pad right col or right 2 cols
    if(blockIdx.x==gridDim.x-1 && (threadIdx.x & (X-1)) > kw1-1) m &= 0x0; //pad all zeros since this tile does not exist
  }  
  if(blockIdx.x==0 && (threadIdx.x & (X-1)) == 0)   m &=0xeeee;  // pad left col
  
  float img_tile[16]; // Prefetch input from GMEM
  float filter_tile[32]; // Prefetch filter from GMEM

  float4 input_frag_mem[8];  //2*2(2*8/4) Data to do Outer Product + prefetch f. SMEM (double_buffer)
  float4 filter_frag_mem[8]; //2*2 Data to do Outer Product + prefetch f. SMEM (double_buffer)
  float4 accumulator[2][16] = {0.0f};  // Accumulators 

  float4 *A_frag; // Input data pointer
  int frag_offset = 2 * (BN*BC); // (2=8/4) SMEM input read offset

  float4 *B_frag; // Filter data pointer  
  int f_frag_offset = 2 * (BC*BK); // (2=8/4 with 4 being float4) SMEM filter read offset 
        

  float4 *input_frag  = (float4*) input_frag_mem;
  float4 *filter_frag = (float4*) filter_frag_mem;

  float4 *swap_filter;
  float4 *swap_input;

  prefetch_input_tile<TW, TH>(A, img_tile, in_h, in_w, X, Y, m);
  prefetch_filter_tile<T, BN, BK>(B, filter_tile, filt_k);

  float4 *input_frag_buffer  = (float4*) (input_frag+4);
  float4 *filter_frag_buffer = (float4*) (filter_frag+4);
  
  // Mainloop - iterates over the entire K dimension - not unrolled
  for(int iter=0; iter<in_c; iter+=BC){ // Current iteration

    A_frag = (float4*) (input_smem  + threadIdx.y*BN*BC);
    B_frag = (float4*) (filter_smem + threadIdx.y*BC*BK);

    load_and_transform_input_tile<BN, BK,BC>(img_tile, input_smem);
    load_filter_tile<BN, BK,BC>(filter_tile, filter_smem, filt_c, filt_k);

    __syncthreads();

    prefetch_input_frag(input_frag, A_frag, frag_offset, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
    prefetch_filter_frag(filter_frag, B_frag, f_frag_offset, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);

    
    #pragma unroll
    for(int i=0; i<BC; i++){

      if(i<(BC-1)){
        A_frag += BN/4;     // This actually moves 32 float (A_frag is float4*)
                          // 32 float is also of size of supertile of one input channel   
        B_frag += BK/4;   // This actually moves 16*4=64 floats (B_frag is float4*), 
                          // 64 floats is also of size of one filter channel 

        prefetch_input_frag(input_frag_buffer, A_frag, frag_offset, access_s[0][threadIdx.x], access_s[1][threadIdx.x]);
        prefetch_filter_frag(filter_frag_buffer, B_frag, f_frag_offset, access_f_s[0][threadIdx.x], access_f_s[1][threadIdx.x]);
      }
     
      outer_product<float>(input_frag, filter_frag, accumulator);

      swap_input = input_frag;
      input_frag = input_frag_buffer;
      input_frag_buffer = swap_input;

      swap_filter = filter_frag;
      filter_frag = filter_frag_buffer;
      filter_frag_buffer = swap_filter;
      
    }
    
    A += BC*in_w*in_h;
    B += filt_k*BC*4*4;

    if(iter<(in_c-BC)){
      prefetch_input_tile<TW,TH>(A, img_tile, in_h, in_w, X, Y, m);
      prefetch_filter_tile<T,BN,BK>(B, filter_tile, filt_k);
    }

    __syncthreads();
  }

  // Transpose, transform and store accumulated result
  store_output_tile<TW, TH, BN, BK, BC>(accumulator, shared_mem, C, out_h, out_w, tiles_dim_w, tiles_dim_h, X, Y,
                  input_frag_mem, filter_frag_mem);
                     
}
#endif

// }

template<typename T, int BN, int BK, int BC>
static void conv_winograd_stage0_cuda(        
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,        
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const T * src0, T * dst,
        cudaStream_t stream) {
    // printf("doing FX\n");
#if __CUDA_ARCH__ >= CC_AMPERE
  FX<BN,BK,BC><<<dim3(src0_ne3/BK, src0_ne2/BC), dim3(32, BC), 0, stream>>>(src0, dst, src0_ne3, src0_ne2, src0_ne1, src0_ne0);
#else
  FX<T,BN,BK,BC><<<dim3(src0_ne3/BK, src0_ne2/BC), dim3(32, BC), 0, stream>>>(src0, dst, src0_ne3, src0_ne2, src0_ne1, src0_ne0);
#endif
    
}

#if __CUDA_ARCH__ >= CC_AMPERE
template<int BN, int BK, int BC>
static void conv_winograd_stage1_cuda(int tiles_dim_w, int tiles_dim_h, int X, int Y,   
        int tile_size, int tile_2d_s,    
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const T * src0, const half * src1,  float * dst,
        cudaStream_t stream) {

    int64_t filt_k = src0_ne0; 
    int64_t in_c   = src1_ne2;
    int64_t in_h   = src1_ne1;
    int64_t in_w   = src1_ne0;
    int64_t filt_c = src0_ne3;
    int64_t out_c  = filt_k;
    int64_t out_h  = in_h;
    int64_t out_w  = in_w;
    int smem_size = (16*BN*BC + 16*BC*BK)*4;
    int max_size = 65536; // 64 KB
    cudaFuncSetAttribute(Winograd_kernel<32,4,BN,BK,BC>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_size);

    Winograd_kernel<32,4,BN,BK,BC><<<dim3((tiles_dim_w+X-1)/X, (tiles_dim_h+Y-1)/Y, filt_k/BK), dim3(BN, 8), smem_size, stream>>>(src1, src0, dst,
               tiles_dim_w, tiles_dim_h, in_c, in_h, in_w, tile_size, X, Y, 
               filt_k, filt_c, out_c, tile_2d_s, out_h, out_w);    
}

#else

template<typename T, int BN, int BK, int BC>
static void conv_winograd_stage1_cuda(int tiles_dim_w, int tiles_dim_h, int X, int Y,   
        int tile_size, int tile_2d_s,    
        const int src0_ne0, const int src0_ne1, const int src0_ne2, const int src0_ne3,
        const int src1_ne0, const int src1_ne1, const int src1_ne2, const int src1_ne3,
        const int dst_ne0, const int dst_ne1, const int dst_ne2, const int dst_ne3,
        const T * src0, const float * src1,  float * dst,
        cudaStream_t stream) {

    int64_t filt_k = src0_ne0; 
    int64_t in_c   = src1_ne2;
    int64_t in_h   = src1_ne1;
    int64_t in_w   = src1_ne0;
    int64_t filt_c = src0_ne3;
    int64_t out_c  = filt_k;
    int64_t out_h  = in_h;
    int64_t out_w  = in_w;
    int smem_size = (16*BN*BC + 16*BC*BK)*4;
    int max_size = 65536; // 64 KB
    cudaFuncSetAttribute(Winograd_kernel<T,32,4,BN,BK,BC>, cudaFuncAttributeMaxDynamicSharedMemorySize, max_size);

    Winograd_kernel<T,32,4,BN,BK,BC><<<dim3((tiles_dim_w+X-1)/X, (tiles_dim_h+Y-1)/Y, filt_k/BK), dim3(BN, 8), smem_size, stream>>>(src1, src0, dst,
               tiles_dim_w, tiles_dim_h, in_c, in_h, in_w, tile_size, X, Y, 
               filt_k, filt_c, out_c, tile_2d_s, out_h, out_w);    
}


#endif

void ggml_cuda_op_winograd_stage0(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    // const half * src0_d = (const half *)src0->data;
    if(src0 == NULL){
        return;
    }
    // float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();
    // int id = ggml_cuda_get_device();

    // GGML_ASSERT(src0->type == GGML_TYPE_F16);
    // GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
    // if (src0->type != GGML_TYPE_F32) {
    //     const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
    //     GGML_ASSERT(to_fp32_cuda != nullptr);
    //     int64_t nle = ggml_nelements(src0);
    //     src0_ddq_as_f32.alloc(nle);
    //     const char * src0_dd = (char *)src0->data;
    //     to_fp32_cuda(src0_dd, src0_ddq_as_f32.get(), nle, stream);
    // }

    // // GGML_ASSERT(ggml_is_contiguous(src0));
    // const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *)src0->data : src0_ddq_as_f32.get();
    if(src0->type == GGML_TYPE_F32){
      const float* src0_d = (const float *)src0->data;
      float * dst_d = (float *)dst->data;
      // conv_winograd_stage0_f32_f32_cuda(src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
      conv_winograd_stage0_cuda<float, 32,64,8>(src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
          dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
           src0_d, dst_d, stream);    
    }else{
      const half * src0_d = (const half *)src0->data;
      half * dst_d = (half *)dst->data;
      // conv_winograd_stage0_f16_f32_cuda(src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
#if __CUDA_ARCH__ >= CC_AMPERE
      conv_winograd_stage0_cuda<half, 32, 64, 16>(src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
          dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
           src0_d, dst_d, stream);
#else      
      conv_winograd_stage0_cuda<half, 32, 64, 8>(src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
          dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
           src0_d, dst_d, stream);
#endif
    }
}


void ggml_cuda_op_winograd_stage1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    // const float * src0_d = (const float *)src0->data;

    const ggml_tensor * src1 = dst->src[1];
    

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    // GGML_ASSERT(src0->type == GGML_TYPE_F32);
    // GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));


    const int m         = 2;
    const int r         = 3;
    const int tile_size = m+r-1; 
    int tiles_dim_w, tiles_dim_h;  
  
    tiles_dim_w = ceil(ceil((double)(src1->ne[0]+2)/2)-1);
    tiles_dim_h = ceil(ceil((double)(src1->ne[1]+2)/2)-1);

    int tile_2d_s = tile_size*tile_size;

    cudaMemcpyToSymbol(access_f_s, aux, 64*sizeof(int));
    cudaMemcpyToSymbol(access_s, aux2, 64*sizeof(int));  
  #if __CUDA_ARCH__ >= CC_AMPERE
    cudaMemcpyToSymbol(access_t, aux3, 64*sizeof(int));
    cudaMemcpyToSymbol(access_f_f, aux1, 64*sizeof(int));
    cudaMemcpyToSymbol(access_o, aux_offset, 4*sizeof(int));
    cudaMemcpyToSymbol(access_p, aux_offset1, 2*sizeof(int));
  #else
    cudaMemcpyToSymbol(tileid, tid, 64*sizeof(int));
  #endif  

#if __CUDA_ARCH__ >= CC_AMPERE
      GGML_ASSERT(src1->type == GGML_TYPE_F16);
      const half * src1_d = (const half *)src1->data;
      conv_winograd_stage1_cuda<32,64,16>(tiles_dim_w, tiles_dim_h, 16, 2,
          tile_size, tile_2d_s,
          src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
          src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
          dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
          src0_d, src1_d, dst_d, stream);        
#else  
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const float * src1_d = (const float *)src1->data;
   
    if(src0->type == GGML_TYPE_F32){
      
      const float * src0_d = (const float *)src0->data;
      // const float * src1_d = (const float *)src1->data;      
      conv_winograd_stage1_cuda<float,32,64,8>(tiles_dim_w, tiles_dim_h, 16, 2,
          tile_size, tile_2d_s,
          src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
          src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
          dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
          src0_d, src1_d, dst_d, stream);
    } else{
      const half * src0_d = (const half *)src0->data;
      // const half * src1_d = (const half *)src1->data;      
    
      conv_winograd_stage1_cuda<half,32,64,8>(tiles_dim_w, tiles_dim_h, 16, 2,
          tile_size, tile_2d_s,
          src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
          src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
          dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
          src0_d, src1_d, dst_d, stream);
    }
#endif

    
}


