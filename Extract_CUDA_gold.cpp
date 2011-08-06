
#ifdef _WIN32
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
//#include "MersenneTwister.h"
#include "Extract_CUDA.h"
#include "dci.h"
#include <iostream.h>
#include <fstream.h>



// static mt_struct MT[MT_RNG_COUNT];
// static uint32_t state[MT_NN];



// extern "C" void initMTRef(const char *fname){
    
//     FILE *fd = fopen(fname, "rb");
//     if(!fd){
//         printf("initMTRef(): failed to open %s\n", fname);
//         printf("TEST FAILED\n");
//         exit(0);
//     }

//     for (int i = 0; i < MT_RNG_COUNT; i++){
//         //Inline structure size for compatibility,
//         //since pointer types are 8-byte on 64-bit systems (unused *state variable)
//         if( !fread(MT + i, 16 /* sizeof(mt_struct) */ * sizeof(int), 1, fd) ){
//             printf("initMTRef(): failed to load %s\n", fname);
//             printf("TEST FAILED\n");
//             exit(0);
//         }
//     }

//     fclose(fd);
// }


// extern "C" void RandomRef(
//     float *h_Random,
//     int NPerRng,
//     unsigned int seed
// ){
//     int iRng, iOut;

//     for(iRng = 0; iRng < MT_RNG_COUNT; iRng++){
//         MT[iRng].state = state;
//         sgenrand_mt(seed, &MT[iRng]);

//         for(iOut = 0; iOut < NPerRng; iOut++)
//            h_Random[iRng * NPerRng + iOut] = ((float)genrand_mt(&MT[iRng]) + 1.0f) / 4294967296.0f;
//     }
// }


#define PI 3.14159265358979f
// void BoxMuller(float& u1, float& u2){
//     float   r = sqrtf(-2.0f * logf(u1));
//     float phi = 2 * PI * u2;
//     u1 = r * cosf(phi);
//     u2 = r * sinf(phi);
// }

// extern "C" void BoxMullerRef(float *h_Random, int NPerRng){
//     int i;

//     for(i = 0; i < MT_RNG_COUNT * NPerRng; i += 2)
//         BoxMuller(h_Random[i + 0], h_Random[i + 1]);
// }

extern "C" int getCube(char * cube, unsigned int &drug, t_params * params)
{
	char ch;
	ifstream fin(params->cube_filename);
	int i = 0;
	int cells = params->cells;
	int all_cells = cells*cells*cells;
	int map[10];
	//cube = new char[all_cells];
	for(i=0; i<10; i++) map[i] = 0;
	if (fin)
	{
		i = 0;
		while(fin.get(ch))
		{
			i++;
		}
		if (i != all_cells)
		{
			printf("file corrupted!\n");
			printf("Readed %i, not %i!\n", i, all_cells);
			fin.close();
			return -1;
		}
	}
	else
	{
		cout << "cant load cube from file " << params->cube_filename << endl;
		return -2;
	}

	printf("file is OK\n");
	fin.close();

	fin.open(params->cube_filename);
	if (fin)
	{
		i = 0;
		while(fin.get(ch))
		{
			cube[i] = (unsigned char)(ch - '0');
			map[(int)(ch - '0')] += 1;
			i++;
		}
		printf("\n");
	}
	fin.close();
	printf("Cube loaded!\n");
	for (i=0; i<10; i++) printf("%i: %i\n", i, map[i]);
        drug = map[1];
	return 0;
}

#include "iniparser.h"
extern "C" char * iniparser_getstring(dictionary * d, const char * key, char * def);
extern "C" int iniparser_getint(dictionary * d, const char * key, int notfound);
extern "C" double iniparser_getdouble(dictionary * d, char * key, double notfound);
extern "C" dictionary * iniparser_load(const char * ininame);
extern "C" void iniparser_freedict(dictionary * d);

extern "C" int getParams(t_params * params, char * filename)
{
    dictionary * ini;
    ini = iniparser_load(filename);

    if (ini == NULL) {
        fprintf(stderr, "cannot parse file: %s\n", filename);
        return 1;
    }
    
    params->stop_part = iniparser_getdouble(ini, "GENERAL:stop_part", 0);
    params->max_iter = iniparser_getint(ini, "GENERAL:max_iter", 0);
    params->cells = iniparser_getint(ini, "GENERAL:cells", 0);

    char * dump_from = iniparser_getstring(ini, "GENERAL:dump_from", NULL);
    if (dump_from != NULL && strlen(dump_from) <= 255) {

        printf("Will be loaded from %s\n", dump_from);

        params->dump_from = new char[256];
        strcpy(params->dump_from, dump_from);
    }
    else params->dump_from = NULL;

    char * cube_filename = iniparser_getstring(ini, "GENERAL:cube_filename", NULL);
    if ((cube_filename != NULL  && strlen(cube_filename) <= 255) && params->dump_from == NULL)
    {
        printf("Correct cube_filename: %s\n", cube_filename);
        params->cube_filename = new char[256];
        strcpy(params->cube_filename, cube_filename);
    }
    else if (params->dump_from == NULL) {
        fprintf(stderr, "Bad cube_filename\n");
        return 1;
    } else params->cube_filename = 0;
        

    char * print_to = iniparser_getstring(ini, "GENERAL:print_to", NULL);
    if (print_to == NULL || strlen(print_to) > 255) {
        fprintf(stderr, "Bad print_to\n");
        return 1;
    }
    params->print_to = new char[256];
    strcpy(params->print_to, print_to);

    char * dump_to = iniparser_getstring(ini, "GENERAL:dump_to", NULL);
    if (dump_to != NULL && strlen(dump_to) <= 255) {
        params->dump_to = new char[256];
        strcpy(params->dump_to, dump_to);
        params->dump_every = iniparser_getint(ini, "GENERAL:dump_every", 0xFFFFFFFF);
    }
    else 
    {
        params->dump_to = 0;
        params->dump_every = 0xFFFFFFFF;
    }

    params->thickness = iniparser_getint(ini, "GENERAL:thickness", 0);
    params->count_every = iniparser_getint(ini, "GENERAL:count_every", 0);

    if (params->stop_part && params->max_iter && params->cells && params->thickness && params->count_every)
    {
        iniparser_freedict(ini);
        printf("Ini readed\n");
        return 0;
    }
    fprintf(stderr, "Bad params file\n");
    return 1;
}

extern "C" void dump_cube(char *cube, int dim, char *fname)
{
    printf("Dumping... ");
    FILE *fd = fopen(fname, "w");
    fwrite(cube, 1, sizeof(char)*dim*dim*dim, fd);
    fclose(fd);
    printf("Done\n");
}

extern "C" int load_dump(char *cube, int dim, char *fname)
{
    printf("Loading... ");
    FILE *fd = fopen(fname, "r");
    if (!fd) {
        fprintf(stderr, "cannot open file: %s\n", fname);
        return 0;
    }
    int res = fread(cube, 1, sizeof(char)*dim*dim*dim, fd);
    if (res != dim*dim*dim) {
        fprintf(stderr, "cannot load dump from file: %s\n", fname);
        fclose(fd);
        return 0;
    }
    fclose(fd);
    printf("Done\n");
    return res;
}

