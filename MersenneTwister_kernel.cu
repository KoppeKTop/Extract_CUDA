/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include "MersenneTwister.h"


__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];


//Load twister configurations
void loadMTGPU(const char *fname){
    FILE *fd = fopen(fname, "rb");
    if(!fd){
        printf("initMTGPU(): failed to open %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    if( !fread(h_MT, sizeof(h_MT), 1, fd) ){
        printf("initMTGPU(): failed to load %s\n", fname);
        printf("TEST FAILED\n");
        exit(0);
    }
    fclose(fd);
}

//Initialize/seed twister for current GPU context
void seedMTGPU(unsigned int seed){
    int i;
    //Need to be thread-safe
    mt_struct_stripped *MT = (mt_struct_stripped *)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

    for(i = 0; i < MT_RNG_COUNT; i++){
        MT[i]      = h_MT[i];
        MT[i].seed = seed;
    }
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT)) );

    free(MT);
}


////////////////////////////////////////////////////////////////////////////////
// Write MT_RNG_COUNT vertical lanes of NPerRng random numbers to *d_Random.
// For coalesced global writes MT_RNG_COUNT should be a multiple of warp size.
// Initial states for each generator are the same, since the states are
// initialized from the global seed. In order to improve distribution properties
// on small NPerRng supply dedicated (local) seed to each twister.
// The local seeds, in their turn, can be extracted from global seed
// by means of any simple random number generator, like LCG.
////////////////////////////////////////////////////////////////////////////////
__global__ void RandomGPU(
    float *d_Random,
    int NPerRng
){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    int iState, iState1, iStateM, iOut;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN];

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N){
        //Load bit-vector Mersenne Twister parameters
        mt_struct_stripped config = ds_MT[iRng];

        //Initialize current state
        mt[0] = config.seed;
        for(iState = 1; iState < MT_NN; iState++)
            mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

        iState = 0;
        mti1 = mt[0];
        for(iOut = 0; iOut < NPerRng; iOut++){
            //iState1 = (iState +     1) % MT_NN
            //iStateM = (iState + MT_MM) % MT_NN
            iState1 = iState + 1;
            iStateM = iState + MT_MM;
            if(iState1 >= MT_NN) iState1 -= MT_NN;
            if(iStateM >= MT_NN) iStateM -= MT_NN;
            mti  = mti1;
            mti1 = mt[iState1];
            mtiM = mt[iStateM];

            x    = (mti & MT_UMASK) | (mti1 & MT_LMASK);
            x    =  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
            mt[iState] = x;
            iState = iState1;

            //Tempering transformation
            x ^= (x >> MT_SHIFT0);
            x ^= (x << MT_SHIFTB) & config.mask_b;
            x ^= (x << MT_SHIFTC) & config.mask_c;
            x ^= (x >> MT_SHIFT1);

            //Convert to (0, 1] float and write to global memory
            d_Random[iRng + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
        }
    }
}



////////////////////////////////////////////////////////////////////////////////
// Transform each of MT_RNG_COUNT lanes of NPerRng uniformly distributed 
// random samples, produced by RandomGPU(), to normally distributed lanes
// using Cartesian form of Box-Muller transformation.
// NPerRng must be even.
////////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979f
__device__ void BoxMuller(float& u1, float& u2){
    float   r = sqrtf(-2.0f * logf(u1));
    float phi = 2 * PI * u2;
    u1 = r * __cosf(phi);
    u2 = r * __sinf(phi);
}

__global__ void BoxMullerGPU(float *d_Random, int NPerRng){
    const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int THREAD_N = blockDim.x * gridDim.x;

    for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N)
        for(int iOut = 0; iOut < NPerRng; iOut += 2)
            BoxMuller(
                d_Random[iRng + (iOut + 0) * MT_RNG_COUNT],
                d_Random[iRng + (iOut + 1) * MT_RNG_COUNT]
            );
}
/*
#define CELLS cells
#define cube_CELLS cube_cells
#define TO_COODRS(x,y,z) ((z) + (y)*CELLS + (x)*CELLS*CELLS)
#define TO_CUBE_COODRS(x,y,z) ((z) + (y)*cube_CELLS + (x)*cube_CELLS*cube_CELLS)
#define CUBE_CELLS_CNT cube_CELLS*cube_CELLS*cube_CELLS
#define THICKNESS thickness


///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////
// const int    PATH_N = iDivUp((CELLS-1) * (CELLS-1) * (CELLS-1), 4);
// const int N_PER_RNG = iAlignUp(iDivUp(PATH_N, MT_RNG_COUNT), 2);
// const int    RAND_N = MT_RNG_COUNT * N_PER_RNG;

__constant__ int indexs[2][3][8] = {{{2, 0, 3, 1, 6, 4, 7, 5},
                        {4, 0, 6, 2, 5, 1, 7, 3},
                        {2, 3, 6, 7, 0, 1, 4, 5}},
                {{1, 3, 0, 2, 5, 7, 4, 6},
                        {1, 5, 3, 7, 0, 4, 2, 6},
                        {4, 5, 0, 1, 6, 7, 2, 3}}};

#define ROTATE_CYCLE(ind)	\
		coord = coords[(ind)];\
		cube_element = cube[coord];\
		if (cube_element >= 2)\
		        return;\
		index = indexs[rotate_angle][rotate_axis][(ind)];\
		rotate_element[index] = cube_element;

#define RETURN_CUBE_TO_MEM(ind)	cube[coords[(ind)]] = rotate_element[(ind)];

#ifdef _DEBUG
__global__ void ca_step(char * cube, int odd, float * rands, unsigned int * blockIdz, unsigned int * dbg)
#else
__global__ void ca_step(char * cube, int odd, float * rands, unsigned int * blockIdz)
#endif
{
//const int indexs[2][3][8] = {{{2, 0, 3, 1, 6, 4, 7, 5},
//                        {4, 0, 6, 2, 5, 1, 7, 3},
//                        {2, 3, 6, 7, 0, 1, 4, 5}},
//                {{1, 3, 0, 2, 5, 7, 4, 6},
//                        {1, 5, 3, 7, 0, 4, 2, 6},
//                        {4, 5, 0, 1, 6, 7, 2, 3}}};
	int i, j, k;
//	int l;
//	int x, y, z;
	int rotate_element[8];
	int coords[8];
	char cube_element;
	int rotate_axis, rotate_angle;
	int index;
	int coord;

	const int cells = 2 * blockDim.x * gridDim.x+1;
	const int cube_cells = blockDim.x * gridDim.x;

	i =  (blockIdx.x * blockDim.x + threadIdx.x);
	j =  (blockIdx.y * blockDim.y + threadIdx.y);
	k = ((*blockIdz) * blockDim.z + threadIdx.z);

	rotate_angle = (int)(rands[TO_CUBE_COODRS(i,j,k)]*3);
	if (rotate_angle == 2)
        	return;
	rotate_axis = (int)(rands[TO_CUBE_COODRS(i,j,k) + CUBE_CELLS_CNT]*3);
	i = i*2 + odd;
	j = j*2 + odd;
	k = k*2 + odd;
	#ifdef _DEBUG
	const int all_cells = cells*cells*cells;
	if (TO_COODRS(i+1,j+1,k+1) > all_cells) 
	{
	  atomicAdd(dbg, 1);
	  return;
	}
	#endif

//	l = 0;
//	for (x = i; x < i+2; x++)
//	{
//        	for (y = j; y < j+2; y++)
//        	{
//		        for (z = k; z < k+2; z++)
//        		{
//		                coords[l] = TO_COODRS(x,y,z);
//              		l++;
//		        };
//	        };
//	};

	coords[0] = TO_COODRS(i,j,k);                
	coords[1] = TO_COODRS(i+1,j,k);
	coords[2] = TO_COODRS(i,j+1,k);
	coords[3] = TO_COODRS(i+1,j+1,k);
	coords[4] = TO_COODRS(i,j,k+1);
	coords[5] = TO_COODRS(i+1,j,k+1);
	coords[6] = TO_COODRS(i,j+1,k+1);
	coords[7] = TO_COODRS(i+1,j+1,k+1);

	ROTATE_CYCLE(0)
	ROTATE_CYCLE(1)
	ROTATE_CYCLE(2)
	ROTATE_CYCLE(3)
	ROTATE_CYCLE(4)
	ROTATE_CYCLE(5)
	ROTATE_CYCLE(6)
	ROTATE_CYCLE(7)

//	for (l = 0; l < 8; l++)    cube[coords[l]] = rotate_element[l];
	RETURN_CUBE_TO_MEM(0)
	RETURN_CUBE_TO_MEM(1)
	RETURN_CUBE_TO_MEM(2)
	RETURN_CUBE_TO_MEM(3)
	RETURN_CUBE_TO_MEM(4)
	RETURN_CUBE_TO_MEM(5)
	RETURN_CUBE_TO_MEM(6)
	RETURN_CUBE_TO_MEM(7)
}

#define _FAST
#undef _FAST

__global__ void clear_cells(char * cube)
{
	//int l;
	const int cells = 2 * blockDim.x * gridDim.x+1;
	//const int all_cells = cells*cells*cells;
	const int i =  (blockIdx.x * blockDim.x + threadIdx.x)*2;
	const int j =  (blockIdx.y * blockDim.y + threadIdx.y)*2;
	//const int k = ((*blockIdz) * blockDim.z + threadIdx.z)*2;

	
       	cube[TO_COODRS(0, i , j)] = 0;
        cube[TO_COODRS(0, i+1 , j)] = 0;
       	cube[TO_COODRS(0, i , j+1)] = 0;
        cube[TO_COODRS(0, i+1 , j+1)] = 0;

#ifdef _FAST
       	*((short *)&(cube[TO_COODRS(i+1, 0 , j)])) = 0;
       	*((short *)&(cube[TO_COODRS(i+1, 0 , j+1)])) = 0;
       	*((short *)&(cube[TO_COODRS(i+1, j , 0)])) = 0;
       	*((short *)&(cube[TO_COODRS(i+1, j+1, 0)])) = 0;

       	*((short *)&(cube[TO_COODRS(CELLS - 1, i , j)])) = 0;
	*((short *)&(cube[TO_COODRS(CELLS - 1, i+1 , j)])) = 0;
	*((short *)&(cube[TO_COODRS(CELLS - 1, i , j+1)])) = 0;
       	*((short *)&(cube[TO_COODRS(CELLS - 1, i+1 , j+1)])) = 0;

	*((short *)&(cube[TO_COODRS(i+1, CELLS - 2, j)])) = 0;
	*((short *)&(cube[TO_COODRS(i+1, CELLS - 2, j)])) = 0;
	*((short *)&(cube[TO_COODRS(i+1, CELLS - 1, j+1)])) = 0;
	*((short *)&(cube[TO_COODRS(i+1, CELLS - 1, j+1)])) = 0;

      	*((short *)&(cube[TO_COODRS(i+1, j , CELLS - 2)])) = 0;
       	*((short *)&(cube[TO_COODRS(i+1, j+1, CELLS - 2)])) = 0;
       	*((short *)&(cube[TO_COODRS(i+1, j , CELLS - 1)])) = 0;
       	*((short *)&(cube[TO_COODRS(i+1, j+1, CELLS - 1)])) = 0;
#else
	cube[TO_COODRS(i, 0 , j)] = 0;
	cube[TO_COODRS(i+1, 0 , j)] = 0;
	cube[TO_COODRS(i, 0 , j+1)] = 0;
	cube[TO_COODRS(i+1, 0 , j+1)] = 0;
	cube[TO_COODRS(i, j , 0)] = 0;
	cube[TO_COODRS(i+1, j , 0)] = 0;
	cube[TO_COODRS(i, j+1, 0)] = 0;
	cube[TO_COODRS(i+1, j+1 , 0)] = 0;

       	cube[TO_COODRS(CELLS - 2, i , j)] = 0;
	cube[TO_COODRS(CELLS - 2, i+1, j)] = 0;
	cube[TO_COODRS(CELLS - 2, i , j+1)] = 0;
       	cube[TO_COODRS(CELLS - 2, i+1 , j+1)] = 0;
	cube[TO_COODRS(CELLS - 1, i , j)] = 0;
	cube[TO_COODRS(CELLS - 1, i+1 , j)] = 0;
	cube[TO_COODRS(CELLS - 1, i , j+1)] = 0;
	cube[TO_COODRS(CELLS - 1, i+1 , j+1)] = 0;

	cube[TO_COODRS(i, CELLS - 2, j)] = 0;
	cube[TO_COODRS(i+1, CELLS - 2, j)] = 0;
	cube[TO_COODRS(i, CELLS - 2, j+1)] = 0;
	cube[TO_COODRS(i+1, CELLS - 2, j+1)] = 0;
	cube[TO_COODRS(i, CELLS - 1 , j)] = 0;
	cube[TO_COODRS(i+1, CELLS - 1, j)] = 0;
	cube[TO_COODRS(i, CELLS - 1, j+1)] = 0;
	cube[TO_COODRS(i+1, CELLS - 1, j+1)] = 0;

       	cube[TO_COODRS(i, j , CELLS - 2)] = 0;
       	cube[TO_COODRS(i+1, j , CELLS - 2)] = 0;
       	cube[TO_COODRS(i, j+1, CELLS - 2)] = 0;
       	cube[TO_COODRS(i+1, j+1 , CELLS - 2)] = 0;
       	cube[TO_COODRS(i, j , CELLS - 1)] = 0;
       	cube[TO_COODRS(i+1, j , CELLS - 1)] = 0;
       	cube[TO_COODRS(i, j+1, CELLS - 1)] = 0;
       	cube[TO_COODRS(i+1, j+1 , CELLS - 1)] = 0;
#endif
}

#ifdef _DEBUG
__global__ void count_cells(unsigned int * part_cnt, char * cube, unsigned int * blockIdz, const unsigned int thick, unsigned int * debug_sum)
#else
__global__ void count_cells(unsigned int * part_cnt, char * cube, unsigned int * blockIdz, const unsigned int thick)
#endif
{
	int x, y, z;
	const int cells = 2 * blockDim.x * gridDim.x+1;
	const unsigned int thickness = thick;
	const int i =  (blockIdx.x * blockDim.x + threadIdx.x)*2;
	const int j =  (blockIdx.y * blockDim.y + threadIdx.y)*2;
	const int k = ((*blockIdz) * blockDim.z + threadIdx.z)*2;
	const int thd = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;


//	__shared__ unsigned int cnt_sh[512];
//	__syncthreads();
//	if (threadIdx.x+threadIdx.y+threadIdx.z == 0)
//	cnt_sh[thd] = 0;
//	__syncthreads();
//	atomicAdd((&part_cnt[blockIdx.x + blockIdx.y * gridDim.x]), cnt);
//	__syncthreads();
//

	if ((i < THICKNESS-1) or (j < THICKNESS-1) or (k < THICKNESS-1)) return;
	if ((i > cells - THICKNESS - 2) or (j > cells - THICKNESS - 2) or (k > cells - THICKNESS - 2)) return;

	unsigned int cnt = 0;
//	unsigned int res = 0;

	for (x = i; x < i+2; x++)
	    for (y = j; y < j+2; y++)
        	for (z = k; z < k+2; z++)
	        {
	                if (cube[TO_COODRS(x,y,z)] == 1) cnt++;
        	}

	atomicAdd(&(part_cnt[thd]), cnt);

//	cnt_sh[thd] = cnt;
//	__syncthreads();
//	y = 0;
//	while (y < 512)
//	{
//		if (cnt_sh[y] != 0) break;
//		y++;
//	}
//	__syncthreads();
//
//	if (thd == y)
//	{
//		for (x = y; x < 512; x++) res += cnt_sh[x];
//		part_cnt[blockIdx.x + blockIdx.y * gridDim.x] = res;
//	}

	#ifdef _DEBUG
		atomicAdd(&(debug_sum[thd]), cnt);
	#endif

}

__global__ void sum_array(unsigned int * d_array, unsigned int *d_sum, int cnt)
{
	int res = 0;
	for (int i = 0; i < cnt; i++)
		res += d_array[i];
	d_sum[0] += res;
}

__global__ void null_array(void * d_array, int cnt)
{
	for (int i = 0; i < cnt; i++)
		((int * )d_array)[i] = 0;
}
*/
/*
__global__ void end_step(unsigned int * part_cnt, char * cube, unsigned int * blockIdz)
{
int i, j, k, l;
int x,y,z;
unsigned int cnt;
int cells = 2 * blockDim.x * gridDim.x+1;
int all_cells = cells*cells*cells;
i =  (blockIdx.x * blockDim.x + threadIdx.x)*2;
j =  (blockIdx.y * blockDim.y + threadIdx.y)*2;
k = ((*blockIdz) * blockDim.z + threadIdx.z)*2;

unsigned char ret = 0;

if (i == 0)
{
        cube[TO_COODRS(0, j , k)] = 0;
        cube[TO_COODRS(0, j+1 , k)] = 0;
        cube[TO_COODRS(0, j , k+1)] = 0;
        cube[TO_COODRS(0, j+1 , k+1)] = 0;
        ret = 1;
}

if (j == 0)
{
        cube[TO_COODRS(i, 0 , k)] = 0;
        cube[TO_COODRS(i+1, 0 , k)] = 0;
        cube[TO_COODRS(i, 0 , k+1)] = 0;
        cube[TO_COODRS(i+1, 0 , k+1)] = 0;
        ret = 1;
}

if (k == 0)
{
        cube[TO_COODRS(i, j , 0)] = 0;
        cube[TO_COODRS(i+1, j , 0)] = 0;
        cube[TO_COODRS(i, j+1, 0)] = 0;
        cube[TO_COODRS(i+1, j+1 , 0)] = 0;
        ret = 1;
}

if (i == CELLS-3)
{
        for (l = 0; l < 2; l++)
        {
        cube[TO_COODRS(CELLS - 2 + l, j , k)] = 0;
        cube[TO_COODRS(CELLS - 2 + l, j+1 , k)] = 0;
        cube[TO_COODRS(CELLS - 2 + l, j , k+1)] = 0;
        cube[TO_COODRS(CELLS - 2 + l, j+1 , k+1)] = 0;
        }
        ret = 1;
}

if (j == CELLS-3)
{
        for (l = 0; l < 2; l++)
        {        
        cube[TO_COODRS(i, CELLS - 2 + l , k)] = 0;
        cube[TO_COODRS(i+1, CELLS - 2 + l , k)] = 0;
        cube[TO_COODRS(i, CELLS - 2 + l , k+1)] = 0;
        cube[TO_COODRS(i+1, CELLS - 2 + l , k+1)] = 0;
        }
        ret = 1;
}

if (k == CELLS-3)
{
        for (l = 0; l < 2; l++)
        {
        cube[TO_COODRS(i, j , CELLS - 2 + l)] = 0;
        cube[TO_COODRS(i+1, j , CELLS - 2 + l)] = 0;
        cube[TO_COODRS(i, j+1, CELLS - 2 + l)] = 0;
        cube[TO_COODRS(i+1, j+1 , CELLS - 2 + l)] = 0;
        }
        ret = 1;
}

if (ret) 
{
  return;
}

if ((i < THICKNESS-1) or (j < THICKNESS-1) or (k < THICKNESS-1)) return;
if ((i > CELLS - THICKNESS - 2) or (j > CELLS - THICKNESS - 2) or (k > CELLS - THICKNESS - 2)) return;

cnt = 0;
for (x = i; x < i+2; x++)
    for (y = j; y < j+2; y++)
        for (z = k; z < k+2; z++)
        {
                //if ((x < THICKNESS) or (y < THICKNESS) or (z < THICKNESS)) continue;
                //if ((x > CELLS - THICKNESS - 2) or (y > CELLS - THICKNESS - 2) or (z > CELLS - THICKNESS - 2)) continue;
                if (cube[TO_COODRS(x,y,z)] == 1) cnt++;
        }
atomicAdd(part_cnt, cnt);
}
*/
