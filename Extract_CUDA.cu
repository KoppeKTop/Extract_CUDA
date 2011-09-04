#define _DEBUG
#undef _DEBUG

#define _SPEED_TEST
#undef _SPEED_TEST

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <cutil.h>
#include <vector.h>
#include <multithreading.h>
#include "Extract_CUDA.h"
#include "cuda_helper.h"
#include "counting.cuh"
#include <curand_kernel.h>


///////////////////////////////////////////////////////////////////////////////
// Common host and device function 
///////////////////////////////////////////////////////////////////////////////
//ceil(a / b)
extern "C" int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

//floor(a / b)
extern "C" int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
extern "C" int iAlignUp(int a, int b){
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
extern "C" int iAlignDown(int a, int b){
    return a - a % b;
}


extern "C" int getCube(char * cube, unsigned int &drug, t_params * params);
extern "C" int getParams(t_params * params, char * filename);
extern "C" void dump_cube(char *cube, int dim, char *fname);
extern "C" int load_dump(char *cube, int dim, char *fname);

// PThreads syncronization (sync_pthreads.cpp)
extern "C" void sync_pthreads();
extern "C" void init_sync(int N);

///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////
const int    PATH_N = 2*264*264*264;
const int N_PER_RNG = iAlignUp(iDivUp(PATH_N, MT_RNG_COUNT), 2);
const int    RAND_N = MT_RNG_COUNT * N_PER_RNG;

int strlen(char * s)
{
    int N;
    for (N = 0; *s != '\0'; s++)
        N++;
    return N;
}

void reverse(char * s)
{
    int c, i, j;
    for (i = 0, j = strlen(s)-1; i < j; i++, j--)
    {
        c = s[i];
        s[i]=s[j];
        s[j] = c;
    }
}

char * itoa(int N)
{
    int i, sign;
    char * s;
    s = (char *)malloc(10 * sizeof(char));
    if ((sign = N) < 0)
        N = -N;
    i = 0;
    do
    {
        s[i++] = N % 10 + '0';
    } while((N /= 10) >0);
    if (sign < 0) s[i++] = '-';
    s[i] = '\0';
    reverse(s);
    return s;
}

void myPrint(char * msg, char * fname)
{
    FILE * fp = fopen(fname, "a");
    if (fp)
    {
        fprintf(fp, msg);
        fprintf(fp, "\n");
        fclose(fp);
    }
}

void SpecialDump(const char * file_name, const char * d_cube, const dim3 & vol_dim, const vector<int *> & results_distrs) 
{
    // we save:
    // dim.x, dim.y, dim.z (12 bytes)
    // cube cells (x*y*z bytes)
    // N - number of results (4 bytes)
    // distributions (N * z * 4 bytes)
    FILE * out = fopen(file_name, "wb");
    char * h_cube = NULL;
    int results_cnt = (int) results_distrs.size();
    fwrite(&(vol_dim.x), sizeof(vol_dim.x), 1, out);
    fwrite(&(vol_dim.y), sizeof(vol_dim.x), 1, out);
    fwrite(&(vol_dim.z), sizeof(vol_dim.x), 1, out);
    size_t cells_cnt = vol_dim.x * vol_dim.y * vol_dim.z;
    h_cube = (char *)malloc(cells_cnt * sizeof(char));
    CUDA_SAFE_CALL( cudaMemcpy(h_cube, d_cube, cells_cnt * sizeof(char), cudaMemcpyDeviceToHost));
    fwrite(h_cube, sizeof(char), cells_cnt, out);
    free(h_cube);
    fwrite(&results_cnt, sizeof(int), 1, out);
    for (int res_idx = 0; res_idx < results_distrs.size(); ++res_idx)
    {
        int * curr_res = results_distrs[res_idx];
        fwrite(curr_res, sizeof(int), vol_dim.z, out);
    }
    fclose(out);
}

void LoadSpecialDump(const char * file_name, char ** h_cube, dim3 * vol_dim, vector<int *> & results_distrs)
{
    FILE * in = fopen(file_name, "rb");
    fread(&(vol_dim.x), sizeof(vol_dim.x), 1, in);
    fread(&(vol_dim.y), sizeof(vol_dim.x), 1, in);
    fread(&(vol_dim.z), sizeof(vol_dim.x), 1, in);
    size_t cells_cnt = vol_dim.x * vol_dim.y * vol_dim.z;
    *h_cube = (char *)malloc(cells_cnt*sizeof(char));
    fread(*h_cube, sizeof(char), cells_cnt, in);
    int results_cnt = 0;
    fread(&results_cnt, sizeof(int), 1, in);
    while(results_cnt-- != 0)
    {
        int * curr_res = (int *)malloc(vol_dim.z * sizeof(int));
        fread(curr_res, sizeof(int), vol_dim.z, in);
        results_distrs.push_back(curr_res);
    }
    fclose(in);
}

bool distrs_eq(int * distr1, int * distr2, int cnt)
{
    bool isEq = true;
    for (int idx = 0; idx < cnt && isEq; ++idx)
    {
        isEq = (distr1[idx] == distr2[idx]);
    }
    return res;
}


int ca_calc(const t_params & params, char * h_cube)
{
    // Plan:
    // load cube from file
    // send it to gpu
    // generate initial randoms
    // send them to gpu
    // go to iterations:
    // even step
    // odd step
    // count layers
    // every N iterations save distribution from gpu to ram
    // do it while distribution is change
    int err = 0;
    vector <int *> results_distrs;
    dim3 vol_dim = params->vol_dim;
    int cells_cnt = vol_dim.x * vol_dim.y * vol_dim.z;
    int block_cnt = (int)(vol_dim.x/2) * (int)(vol_dim.y/2) * (int)(vol_dim.z/2);
    int cells_sz = cells_cnt * sizeof(char);
    char * d_cube;
    curandState * d_rnd;
    int *d_distr, *h_distr; // distributions
    int distr_iter_sz = vol_dim.z * sizeof(int);
    int *h_distr_one_iter;
    int iters_on_gpu = 1000; // we will store this iterations directly on gpu
    int curr_gpu_iter = 0;
    int curr_iter = 0;
    
    dim3 gen_rnd_block(128, 1), gen_rnd_grid(32000, 1);
    dim3 step_block(8,8,8), step_grid;
    dim3 count_grid(vol_dim.z, 1), count_block(128,1);
    
    step_grid.x = (vol_dim.x / (step_block.x * 2)) + 1;
    step_grid.y = (vol_dim.y / (step_block.y * 2)) + 1;
    
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_cube, cells_sz));
    CUDA_SAFE_CALL( cudaMemcpy(d_cube, h_cube, cells_sz, cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_rnd, block_cnt * sizeof(curandState)));
    init_rnd<<<gen_rnd_grid, gen_rnd_block>>>(d_rnd, block_cnt, (unsigned int) time(NULL));
    CUDA_SAFE_CALL( cudaThreadSynchronize() );

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_distr, iters_on_gpu * distr_iter_sz));

    bool can_exit = false;
    while (!can_exit)
    {
        for (int odd = 0; odd < 2; ++odd)
        {
            for (int z_coord = 0; z_coord < vol_dim.z; z_coord += step_block.z * 2)
            {                
                ca_step<<<step_grid, step_block>>>(d_cube, odd, d_rnd, z_coord, vol_dim);
            }
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
            count_points_by_layer<<<count_grid, count_block, count_block.x * sizeof(int)>>>
            (d_cube, (char)1, vol_dim, d_distr + curr_gpu_iter * distr_iter_sz);
            CUDA_SAFE_CALL( cudaThreadSynchronize() );
            if (++curr_gpu_iter == iters_on_gpu)
            {
                // read results to host
                h_distr = (int *)malloc(iters_on_gpu * distr_iter_sz);
                CUDA_SAFE_CALL( cudaMemcpy(h_distr, d_distr, iters_on_gpu * distr_iter_sz, cudaMemcpyDeviceToHost));
                for (int curr_copy = 0; curr_copy < iters_on_gpu; ++curr_copy)
                {
                    int * one_iter = (int *)malloc(distr_iter_sz);
                    memcpy(one_iter, h_distr + vol_dim.z*curr_copy, distr_iter_sz);
                    results_distrs.push_back(one_iter);
                }
                free(h_distr);
                can_exit = !distrs_eq(*results_distrs.end(), *(results_distrs.end() - iters_on_gpu/2), vol_dim.z);
                curr_gpu_iter = 0;
            }
        }
        ++curr_iter;
        if (curr_iter % params.dump_every == 0)
        {
            SpecialDump(params.dump_to, d_cube, vol_dim, results_distrs);
        }
    }
    SpecialDump(params.dump_to, d_cube, vol_dim, results_distrs);
    for (int res_idx = 0; res_idx < results_distrs.size(); ++res_idx)
    {
        free(results_distrs[res_idx]);
    }
    CUDA_SAFE_CALL(cudaFree(d_cube));
    CUDA_SAFE_CALL(cudaFree(d_rnd));
    CUDA_SAFE_CALL(cudaFree(d_distr));
    return err;
}

///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){

    float
        *d_Rand;

    double
        gpuTime;

    int i,j;
    unsigned int hTimer, hSumTimer;
    char ** iniFilename;
    iniFilename = new (char *);
    iniFilename[0] = new char[256];

    t_params params;
    cutGetCmdLineArgumentstr( argc, (const char **)argv, "ini", iniFilename);
    if (iniFilename[0] == NULL) return 1;

    if (getParams(&params, iniFilename[0]) != 0)
    {
        printf("Oooops!\n");
        return 1;
    }

    dim3 vol_dim = params.vol_dim;
    cells_cnt = vol_dim.x * vol_dim.y * vol_dim.z;
    char *cube = (char *)malloc(cells_cnt);//, *cube_fin;
    if (getCube(cube, drug, &params) < 0) return 1;
    ca_calc(params, cube);
    cout << "Done!\n";
    free(cube);
    delete [] iniFilename[0];
    delete iniFilename;
    return 0

//     cube = 0;
//     char *cube_gpu;
//     char *s;

//     unsigned int *part_cnt;
//     unsigned int *part_cnt_gpu, *z_gpu;
//     unsigned int *sum_gpu;
//     unsigned int drug = 0;
//     dim3 vol_dim = params.vol_dim;
//     // const int cells = params.vol_dim;
//     const int all_cells = vol_dim.x * vol_dim.y * vol_dim.z;
//     const int thickness = params.thickness;


//     dim3 dimBlock(8, 8, 8);
//     dim3 dimBlock_clear(8, 8);
//     int grid_dim_x = vol_dim.x/16 + ( vol_dim.x % 16 == 0 )?0:1;
//     int grid_dim_y = vol_dim.y/16 + ( vol_dim.y % 16 == 0 )?0:1;
//     int grid_dim_z = vol_dim.z/16 + ( vol_dim.z % 16 == 0 )?0:1;
//     dim3 dimGrid(grid_dim_x, grid_dim_y);

//     CUT_DEVICE_INIT(argc, argv);
//     CUT_SAFE_CALL( cutCreateTimer(&hTimer) );
//     CUT_SAFE_CALL( cutCreateTimer(&hSumTimer) );

//     printf("Initializing data\n", PATH_N);
//     part_cnt = (unsigned int *)malloc(sizeof(unsigned int));
//     cube     = (char *)malloc(all_cells * sizeof(char));

//     CUDA_SAFE_CALL( cudaMalloc((void **)&cube_gpu, all_cells * sizeof(char)) );
//     CUDA_SAFE_CALL( cudaMalloc((void **)&d_Rand, RAND_N * sizeof(float)) );
//     CUDA_SAFE_CALL( cudaMalloc((void **)&part_cnt_gpu, sizeof(int)) );
//     CUDA_SAFE_CALL( cudaMalloc((void **)&sum_gpu, 512 * sizeof(int)) );
//     CUDA_SAFE_CALL( cudaMalloc((void **)&z_gpu, sizeof(int)) );
// #ifdef _DEBUG
//     unsigned int *debug_gpu, *dbg_cpu;
//     unsigned int dbg_sum;
//     CUDA_SAFE_CALL( cudaMalloc((void **)&debug_gpu, 512*sizeof(int)) );
//     dbg_cpu = (unsigned int *)malloc(512*sizeof(unsigned int));
//     for (i = 0; i < 512; i++) dbg_cpu[i] = 0;
// #endif

//     if (params.dump_from)
//     {
//         if (load_dump(cube, vol_dim, params.dump_from) == 0) return 1;
//     }
//     else 
//         if (getCube(cube, drug, &params) < 0) return 1;
//         else dump_cube(cube, vol_dim, params.dump_to);

//     CUDA_SAFE_CALL( cudaMemcpy(cube_gpu, cube, all_cells * sizeof(char), cudaMemcpyHostToDevice));
    
//     printf("Loading CPU and GPU twisters configurations...\n");
//     srand((unsigned int) time(NULL));

//     myPrint("Let's try some iterations...\n", params.print_to);
//     myPrint((s=itoa((drug))), params.print_to);
//     free(s);
//     CUDA_SAFE_CALL( cudaThreadSynchronize() );
//     CUT_SAFE_CALL( cutResetTimer(hTimer) );
//     CUT_SAFE_CALL( cutResetTimer(hSumTimer) );
//     CUT_SAFE_CALL( cutStartTimer(hSumTimer) );

//     for (i=1; i <= params.max_iter; i++)
//     {
//     #ifdef _DEBUG
//         CUDA_SAFE_CALL( cudaMemcpy(debug_gpu, dbg_cpu, sizeof(int), cudaMemcpyHostToDevice));
//     #endif

//     #ifdef _SPEED_TEST
//         CUT_SAFE_CALL( cutStartTimer(hTimer) );
//     #endif

//         for (j = 0; j < grid_dim; j++)
//         {
//         #ifdef _DEBUG
//             ca_step<<<dimGrid, dimBlock>>>(cube_gpu, 0, d_Rand, j, debug_gpu);
//         #else
//             ca_step<<<dimGrid, dimBlock>>>(cube_gpu, 0, d_Rand, j);
//         #endif
//             CUT_CHECK_ERROR("Even step execution failed\n");
//             CUDA_SAFE_CALL( cudaThreadSynchronize() );
//         }

//     #ifdef _SPEED_TEST
//         cutStopTimer(hTimer);
//         printf("Even step time: %f ms\n\n", cutGetTimerValue(hTimer));
//         cutResetTimer(hTimer);
//     #endif

//     #ifdef _DEBUG
//         CUDA_SAFE_CALL( cudaMemcpy(dbg_cpu, debug_gpu, sizeof(int), cudaMemcpyDeviceToHost));
//         if (dbg_cpu[0] != 0)
//         {
//             printf("Borders crossed %i times!!!\n", dbg_cpu[0]);
//             dbg_cpu[0] = 0;
//         }
//     #endif

//     #ifdef _DEBUG
//         CUDA_SAFE_CALL( cudaMemcpy(debug_gpu, dbg_cpu, sizeof(int), cudaMemcpyHostToDevice));
//     #endif

//     #ifdef _SPEED_TEST
//         CUT_SAFE_CALL( cutStartTimer(hTimer) );
//     #endif

//         for (j = 0; j < grid_dim; j++)
//         {
//     #ifdef _DEBUG
//             ca_step<<<dimGrid, dimBlock>>>(cube_gpu, 1, d_Rand, j, debug_gpu);
//     #else
//             ca_step<<<dimGrid, dimBlock>>>(cube_gpu, 1, d_Rand, z_gpu);
//     #endif
//             CUT_CHECK_ERROR("Odd step execution failed\n");
//             CUDA_SAFE_CALL( cudaThreadSynchronize() );
//         }

//     #ifdef _SPEED_TEST
//         cutStopTimer(hTimer);
//         printf("Odd step time: %f ms\n\n", cutGetTimerValue(hTimer));
//         cutResetTimer(hTimer);
//     #endif


//     #ifdef _DEBUG
//         CUDA_SAFE_CALL( cudaMemcpy(dbg_cpu, debug_gpu, sizeof(int), cudaMemcpyDeviceToHost));
//         if (dbg_cpu[0] != 0)
//         {
//             printf("Borders crossed %i times!!!\n", dbg_cpu[0]);
//             dbg_cpu[0] = 0;
//         }
//     #endif

//     #ifdef _SPEED_TEST
//         CUT_SAFE_CALL( cutStartTimer(hTimer) );
//     #endif
            
//         //for (j = 0; j < grid_dim; j++)
//             //{
//                 //CUDA_SAFE_CALL( cudaMemcpy(z_gpu, &j, sizeof(int), cudaMemcpyHostToDevice));
//         clear_cells<<<dimGrid, dimBlock_clear>>>(cube_gpu);
//         CUT_CHECK_ERROR("Clearing failed\n");
//         CUDA_SAFE_CALL( cudaThreadSynchronize() );                
//         //}

//     #ifdef _SPEED_TEST
//         cutStopTimer(hTimer);
//         printf("Clar cells time: %f ms\n\n", cutGetTimerValue(hTimer));
//         cutResetTimer(hTimer);
//     #endif


//         if (i % params.dump_every == 0)
//         {
//             CUDA_SAFE_CALL( cudaMemcpy(cube, cube_gpu, all_cells * sizeof(char), cudaMemcpyDeviceToHost));
//             dump_cube(cube, cells, params.dump_to);
//         }
//         printf("%i-th iteration end\n", i);
//     }
//     CUDA_SAFE_CALL( cudaThreadSynchronize() );
//     CUT_SAFE_CALL( cutStopTimer(hSumTimer) );
//     gpuTime = cutGetTimerValue(hSumTimer);
//     printf("Iteration time  : %f \n", (double)gpuTime/i);

//     myPrint("exiting", params.print_to);
//     printf("Shutting down...\n");
//     CUDA_SAFE_CALL( cudaFree(d_Rand) );
//     CUDA_SAFE_CALL( cudaFree(part_cnt_gpu) );
//     CUDA_SAFE_CALL( cudaFree(cube_gpu) );
//     CUDA_SAFE_CALL( cudaFree(sum_gpu) );
//     CUDA_SAFE_CALL( cudaFree(z_gpu) );
// #ifdef _DEBUG
//     CUDA_SAFE_CALL( cudaFree(debug_gpu) );
//     free(dbg_cpu);
// #endif
//     free(part_cnt);
//     free(cube);

//     CUT_SAFE_CALL( cutDeleteTimer( hTimer) );

//     CUT_EXIT(argc, argv);
}

