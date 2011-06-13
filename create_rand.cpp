#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <iostream.h>
#include <fstream.h>
#include <time.h>

int calc_active_cnt(int cnt_ag, double dens_ag, double M_ac, double mass_part)
{
    double m_ag = dens_ag / 1e21;
    double Na = 6.02e23;
    double m_ac = M_ac / Na;
    double sum_m_ag = m_ag * cnt_ag;
    double sum_m_ac = sum_m_ag * mass_part / (1-mass_part);
    return int(sum_m_ac/m_ac + 1);
}

void dump_cube(char *cube, int dim, char *fname)
{
    printf("Dumping... ");
    FILE *fd = fopen(fname, "w");
    fwrite(cube, 1, sizeof(char)*dim*dim*dim, fd);
    fclose(fd);
    printf("Done\n");
}

char * create_cube(int size, int structure, int cnt, int cnt_av)
{
    char * cube = new char[size*size*size];
    if (!cube)
    {
        fprintf(stderr, "Can\'t allocate memory\n");
        return NULL;
    }

    srand((unsigned int) time(NULL));

    int thick = (size - structure - 1)/2;
    int i, j, k, num;

    for (i=0; i < size*size*size; i++) cube[i] = 0;

    printf("Generate structure: %i cells\n", cnt);
    while(cnt != 0)
    {
        i = (int)(((double)rand()/RAND_MAX) * structure) + thick;
        j = (int)(((double)rand()/RAND_MAX) * structure) + thick;
        k = (int)(((double)rand()/RAND_MAX) * structure) + thick;

        num = i * size * size + j * size + k;
        if (cube[num] == 0) 
        {
            cube[num] = 2;
            cnt--;
        }
    }

    printf("Incapsulate active component: %i cells\n", cnt_av);
    while(cnt_av != 0)
    {
        i = (int)(((double)rand()/RAND_MAX) * structure) + thick;
        j = (int)(((double)rand()/RAND_MAX) * structure) + thick;
        k = (int)(((double)rand()/RAND_MAX) * structure) + thick;

        num = i * size * size + j * size + k;
        if (cube[num] == 0)
        {
            cube[num] = 1;
            cnt_av--;
            //printf("Ost: %i\n", cnt_av);
        }
    }
    printf("Done\n");

    return cube;
}

int main()
{
    double porosity = 0.9;
    int size = 529;
    int structure = 500;
    int cnt_ag = structure*structure*structure*(1-porosity);
    int cnt_av = calc_active_cnt(cnt_ag, 2.2, 254, 0.2);

    printf("Ag: %i, AC: %i\n", cnt_ag, cnt_av);
    if (cnt_av + cnt_ag >= structure*structure*structure)
    {
        printf("Too much cells...\n");
        return 1;
    }

    char *cube = create_cube(size, structure, cnt_ag, cnt_av);
    if (!cube) return 1;
    dump_cube(cube, size, "rand_cube.dat");
    return 0;
}

