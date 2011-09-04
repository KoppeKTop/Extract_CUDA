#ifndef _COUNTING_CUH_
#define _COUNTING_CUH_

#include <cuda.h>

extern __shared__ int counts[];

template <typename TVal>
__global__ void count_points_by_layer(TVal * points, const TVal s_val, const dim3 vol_dim, int * res)
// GPU version
// thread/block configuration:
// 1 block count 1 layer
// configurate kernel for shared mem = blockDim.x * sizeof(int)
// minimum threads count = 64
{
//  const int currZ = blockIdx.x;
    if (blockIdx.x >= vol_dim.z)
        return;
    counts[threadIdx.x] = 0; 
    for (int currY = 0; currY < vol_dim.y; ++currY)
    {
        for (int currX = threadIdx.x; currX < vol_dim.x; currX += blockDim.x)
        {
            if (points[currX + currY * vol_dim.x + blockIdx.x * vol_dim.x * vol_dim.y] == s_val)
                counts[threadIdx.x]++;
        }
    }

    __syncthreads();
    
    // now reduce results

    for ( int s = blockDim.x / 2; s > 32; s >>= 1 ) 
    {
        if ( threadIdx.x < s ) 
            counts [threadIdx.x] += counts [threadIdx.x + s];
    }

    __syncthreads ();
    if ( threadIdx.x < 32 ) // unroll last iterations 
    {
        counts [threadIdx.x] += counts [threadIdx.x + 32]; 
        counts [threadIdx.x] += counts [threadIdx.x + 16]; 
        counts [threadIdx.x] += counts [threadIdx.x + 8]; 
        counts [threadIdx.x] += counts [threadIdx.x + 4]; 
        counts [threadIdx.x] += counts [threadIdx.x + 2]; 
        counts [threadIdx.x] += counts [threadIdx.x + 1];
    }

    if (threadIdx.x == 0)
        res[blockIdx.x] = counts[0];
}

template <typename TVal>
__host__ void count_points_by_layer_cpu(TVal * points, const TVal s_val, const dim3 vol_dim, int * res)
// CPU version
{
    for(int currZ = 0; currZ < vol_dim.z; currZ ++)
    {
        res[currZ] = 0;
        for (int currY = 0; currY < vol_dim.y; currY ++)
            for (int currX = 0; currX < vol_dim.x; currX ++)
            {
                if (points[currX + currY * vol_dim.x + currZ * vol_dim.x * vol_dim.y] == s_val)
                    res[currZ]++;
            }
    }
}


#endif
