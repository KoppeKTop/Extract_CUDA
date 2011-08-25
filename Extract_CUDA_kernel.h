#ifndef EXTRACT_CUDA_KERNEL_H_

#define EXTRACT_CUDA_KERNEL_H_

#include <curand_kernel.h>

__global__ void ca_step(char * cube, int odd, curandState * rands, unsigned int blockIdz, dim3 ca_dim
#ifdef _DEBUG
, unsigned int * dbg
#endif
);

__global__ void clear_cells(char * cube, dim3 ca_dim);
__global__ void clear_top(char * cube, dim3 ca_dim);
__global__ void clear_floor(char * cube, dim3 ca_dim);

// __global__ void count_cells(unsigned int * part_cnt, char * cube, unsigned int blockIdz, const unsigned int thick, 
// const unsigned int height, dim3 ca_dim
// #ifdef _DEBUG
// , unsigned int * debug_sum
// #endif
// );

// __global__ void sum_array(unsigned int * d_array, unsigned int *d_sum, int cnt);

#endif