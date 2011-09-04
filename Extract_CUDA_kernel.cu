#include <curand_kernel.h>
#include "Extract_CUDA_kernel.h"

#define TO_COODRS(x,y,z) ((x) + (y)*ca_dim.x + (z)*ca_dim.x*ca_dim.y)
#define TO_CUBE_COODRS(x,y,z) ((x) + (y)*cube_CELLS + (z)*cube_CELLS*cube_CELLS)
#define CUBE_CELLS_CNT cube_CELLS*cube_CELLS*cube_CELLS
#define THICKNESS thickness


///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////
__constant__ int indexs[2][3][8] = {{{2, 0, 3, 1, 6, 4, 7, 5},
                        {4, 0, 6, 2, 5, 1, 7, 3},
                        {2, 3, 6, 7, 0, 1, 4, 5}},
                {{1, 3, 0, 2, 5, 7, 4, 6},
                        {1, 5, 3, 7, 0, 4, 2, 6},
                        {4, 5, 0, 1, 6, 7, 2, 3}}};


#define ROTATE_CYCLE_2(ind)    \
        cube_element1 = cube[coords[(ind)]];\
        cube_element2 = cube[coords[(ind)+1]];\
        if (cube_element1 >= 2 || cube_element2 >= 2)\
            return;\
        index = indexs[rotate_angle][rotate_axis][(ind)];\
        rotate_element[index] = cube_element1;\
        index = indexs[rotate_angle][rotate_axis][(ind)+1];\
        rotate_element[index] = cube_element2;

#define ROTATE_CYCLE(ind)    \
        if (cube[coords[(ind)]] >= 2)\
            return;\
        index = indexs[rotate_angle][rotate_axis][(ind)];\
        rotate_element[index] = cube[coords[(ind)]];

#define RETURN_CUBE_TO_MEM(ind)    cube[coords[(ind)]] = rotate_element[(ind)];

__global__ void ca_step(char * cube, int odd, curandState * rands, unsigned int blockIdz, dim3 ca_dim
#ifdef _DEBUG
, unsigned int * dbg
#endif
)
{
    int i, j, k;
    int i1, j1; // coordinates of diagonal cell in block, (k1 == k + 1 - there is no toroid in that direction)
    char rotate_element[8];
    int coords[8];
    char cube_element1;
    char cube_element2;
    int rotate_axis, rotate_angle;
    int index;
    short onBoarder = 0;

    i =  (blockIdx.x * blockDim.x + threadIdx.x);
    j =  (blockIdx.y * blockDim.y + threadIdx.y);
    k = (blockIdz * blockDim.z + threadIdx.z);

    if ( (i*2+odd) >= ca_dim.x ||  (j*2+odd) >= ca_dim.y || (k*2+odd+1) >= ca_dim.z))
    {
        return;
    }

    if ( (i*2+odd) == ca_dim.x ||  (j*2+odd) == ca_dim.y)
    {
        onBoarder = 1;
    }
    
    int cube_idx = i + j * (ca_dim.x/2) + k * (ca_dim.x/2) * (ca_dim.y/2);
    curandState local_rnd_state = rands[cube_idx];
    
    rotate_angle = (int)(curand_uniform(&local_rnd_state) * 3);
    if ( rotate_angle >= 2 ) // choose not to rotate
        return;
    rotate_axis = (int)(curand_uniform(&local_rnd_state) * 3);
    rands[cube_idx] = local_rnd_state;
    if (rotate_axis == 3)   rotate_axis = 2; // uniform distribution includes 1.0, so...
    i = i*2 + odd;
    j = j*2 + odd;
    k = k*2 + odd;
    i1 = (i+1 == ca_dim.x)? 0: i+1;
    j1 = (j+1 == ca_dim.y)? 0: j+1;
#ifdef _DEBUG
    const int all_cells = ca_dim.x*ca_dim.y*ca_dim.z;
    if (TO_COODRS(i+1,j+1,k+1) > all_cells) 
    {
      atomicAdd(dbg, 1);
      return;
    }
#endif
    coords[0] = TO_COODRS(i,j,k);
    coords[1] = TO_COODRS(i1,j,k);
    coords[2] = TO_COODRS(i,j1,k);
    coords[3] = TO_COODRS(i1,j1,k);
    coords[4] = TO_COODRS(i,j,k+1);
    coords[5] = TO_COODRS(i1,j,k+1);
    coords[6] = TO_COODRS(i,j1,k+1);
    coords[7] = TO_COODRS(i1,j1,k+1);

    ROTATE_CYCLE_2(0)
    ROTATE_CYCLE_2(2)
    ROTATE_CYCLE_2(4)
    ROTATE_CYCLE_2(6)

//    for (l = 0; l < 8; l++)    cube[coords[l]] = rotate_element[l];
    RETURN_CUBE_TO_MEM(0)
    RETURN_CUBE_TO_MEM(1)
    RETURN_CUBE_TO_MEM(2)
    RETURN_CUBE_TO_MEM(3)
    RETURN_CUBE_TO_MEM(4)
    RETURN_CUBE_TO_MEM(5)
    RETURN_CUBE_TO_MEM(6)
    RETURN_CUBE_TO_MEM(7)
}

__global__ void clear_cells(char * cube, dim3 ca_dim)
{
    const int i =  (blockIdx.x * blockDim.x + threadIdx.x)*2;
    const int j =  (blockIdx.y * blockDim.y + threadIdx.y)*2;
    
    cube[TO_COODRS(0, i , j)] = 0;
    cube[TO_COODRS(0, i+1 , j)] = 0;
    cube[TO_COODRS(0, i , j+1)] = 0;
    cube[TO_COODRS(0, i+1 , j+1)] = 0;

    cube[TO_COODRS(i, 0 , j)] = 0;
    cube[TO_COODRS(i+1, 0 , j)] = 0;
    cube[TO_COODRS(i, 0 , j+1)] = 0;
    cube[TO_COODRS(i+1, 0 , j+1)] = 0;

    cube[TO_COODRS(CELLS - 2, i , j)] = 0;
    cube[TO_COODRS(CELLS - 1, i , j)] = 0;
    cube[TO_COODRS(CELLS - 2, i+1, j)] = 0;
    cube[TO_COODRS(CELLS - 1, i+1, j)] = 0;
    cube[TO_COODRS(CELLS - 2, i , j+1)] = 0;
    cube[TO_COODRS(CELLS - 1, i , j+1)] = 0;
    cube[TO_COODRS(CELLS - 2, i+1 , j+1)] = 0;
    cube[TO_COODRS(CELLS - 1, i+1 , j+1)] = 0;

    cube[TO_COODRS(i, CELLS - 2, j)] = 0;
    cube[TO_COODRS(i+1, CELLS - 2, j)] = 0;
    cube[TO_COODRS(i, CELLS - 2, j+1)] = 0;
    cube[TO_COODRS(i+1, CELLS - 2, j+1)] = 0;
    cube[TO_COODRS(i, CELLS - 1 , j)] = 0;
    cube[TO_COODRS(i+1, CELLS - 1, j)] = 0;
    cube[TO_COODRS(i, CELLS - 1, j+1)] = 0;
    cube[TO_COODRS(i+1, CELLS - 1, j+1)] = 0;
}

__global__ void clear_top(char * cube, dim3 ca_dim)
{
    const int i =  (blockIdx.x * blockDim.x + threadIdx.x)*2;
    const int j =  (blockIdx.y * blockDim.y + threadIdx.y)*2;

    cube[TO_COODRS(i, j , 0)] = 1;
    cube[TO_COODRS(i+1, j , 0)] = 1;
    cube[TO_COODRS(i, j+1, 0)] = 1;
    cube[TO_COODRS(i+1, j+1 , 0)] = 1;
}

__global__ void clear_floor(char * cube, dim3 ca_dim)
{
    const int i =  (blockIdx.x * blockDim.x + threadIdx.x)*2;
    const int j =  (blockIdx.y * blockDim.y + threadIdx.y)*2;

    cube[TO_COODRS(i, j , CELLS - 2)] = 0;
    cube[TO_COODRS(i+1, j , CELLS - 2)] = 0;
    cube[TO_COODRS(i, j+1, CELLS - 2)] = 0;
    cube[TO_COODRS(i+1, j+1 , CELLS - 2)] = 0;
    cube[TO_COODRS(i, j , CELLS - 1)] = 0;
    cube[TO_COODRS(i+1, j , CELLS - 1)] = 0;
    cube[TO_COODRS(i, j+1, CELLS - 1)] = 0;
    cube[TO_COODRS(i+1, j+1 , CELLS - 1)] = 0;
}

__global__ void init_rnd(curandState * state, int randCnt, unsigned int randSeed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < randCnt)
    {
        curand_init(randSeed, idx, 0, state + idx);
        idx += gridDim.x * blockDim.x;
    }
}


// __global__ void count_cells(unsigned int * part_cnt, char * cube, unsigned int * blockIdz, const unsigned int thick, 
// const unsigned int height, dim3 ca_dim
// #ifdef _DEBUG
// , unsigned int * debug_sum
// #endif
// )
// {
//     const int cells = 2 * blockDim.x * gridDim.x+1;
//     const unsigned int thickness = thick;
//     const unsigned int h = height;
//     const int i =  (blockIdx.x * blockDim.x + threadIdx.x)*2;
//     const int j =  (blockIdx.y * blockDim.y + threadIdx.y)*2;
//     const int k = ((*blockIdz) * blockDim.z + threadIdx.z)*2;
//     const int thd = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

//     if ((i < THICKNESS-1) or (j < THICKNESS-1) or (h + k < THICKNESS - 1)) return;
//     if ((i > cells - THICKNESS - 2) or (j > cells - THICKNESS - 2) or (h + k > cells - THICKNESS - 2)) return;

//     unsigned int cnt = 0;

//     if (cube[TO_COODRS(i, j, k)] == 1) cnt++;
//     if (cube[TO_COODRS(i+1, j, k)] == 1) cnt++;
//     if (cube[TO_COODRS(i, j+1, k)] == 1) cnt++;
//     if (cube[TO_COODRS(i+1, j+1, k)] == 1) cnt++;
//     if (cube[TO_COODRS(i, j, k+1)] == 1) cnt++;
//     if (cube[TO_COODRS(i+1, j, k+1)] == 1) cnt++;
//     if (cube[TO_COODRS(i, j+1, k+1)] == 1) cnt++;
//     if (cube[TO_COODRS(i+1, j+1, k+1)] == 1) cnt++;

//     atomicAdd(&(part_cnt[thd]), cnt);

/*    cnt_sh[thd] = cnt;
    __syncthreads();
    y = 0;
    while (y < 512)
    {
        if (cnt_sh[y] != 0) break;
        y++;
    }
    __syncthreads();

    if (thd == y)
    {
        for (x = y; x < 512; x++) res += cnt_sh[x];
        part_cnt[blockIdx.x + blockIdx.y * gridDim.x] = res;
    }
*/
// #ifdef _DEBUG
//     atomicAdd(&(debug_sum[thd]), cnt);
// #endif

// }

// __global__ void sum_array(unsigned int * d_array, unsigned int *d_sum, int cnt)
// {
//     int res = 0;
//     for (int i = 0; i < cnt; i++)
//         res += d_array[i];
//     d_sum[0] += res;
// }

