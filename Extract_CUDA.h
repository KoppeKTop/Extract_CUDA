#ifndef EXTRACT_CUDA_H
#define EXTRACT_CUDA_H
#ifndef extract_cuda_h
#define extract_cuda_h

#include <cuda.h>

typedef struct
{
    double stop_part;
    unsigned int max_iter;
    dim3 vol_dim;
    unsigned int thickness;
    char * cube_filename;
    char * dump_to;
    char * dump_from;
    char * print_to;
    unsigned int dump_every;
    unsigned int count_every;
} t_params;

typedef struct{
    //Device ID for multi-GPU version
    int device;
    // Number of current thread
    unsigned int th_id;

    // threads count
    unsigned int th_cnt;

    // Host-side data source and result destination
    TOptionData  *optionData;
    TOptionValue *callValue;
    //Intermediate device-side buffers
    void *d_Buffer;

    //(Pseudo/quasirandom) number generator seed
    unsigned int seed;
    //(Pseudo/quasirandom) samples count
    int pathN;
    //(Pseudo/quasirandom) samples device storage
    float *d_Samples;

    //Time stamp
    float time;
} TOptionPlan;

#endif
#endif

