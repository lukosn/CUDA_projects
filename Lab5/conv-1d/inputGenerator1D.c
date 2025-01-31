#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RND_SEED 13 // for tests reproducibility
#define MAX_MASK_VALUE 10
#define MAX_SIGNAL_VALUE 200

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Wrong number of arguments: exactly 3 arguments needed (length of mask, length of signal and output file name)\n");
        return 1;
    }

    // test reproducibility
    srand(RND_SEED);

    // read sizes
    int maskSize = atoi(argv[1]);
    int size = atoi(argv[2]);

    // open file
    FILE *fp;
    fp = fopen(argv[3], "w");
    if (fp == NULL)
    {
        printf("%s: cannot open file.\n", argv[3]);
        return 2;
    }

    // generate & write data
    fprintf(fp, "%d %d\n", maskSize, size);

    int n = maskSize / 2;
    for (int i = 0; i < maskSize; ++i)
    {
        float M = fmod(rand() / 100.0, (float)MAX_MASK_VALUE);
        fprintf(fp, "%.2f ", M);
    }
    fprintf(fp, "\n");

    for (int i = 0; i < size; ++i)
    {
        float N = fmod(rand() / 100.0, (float)MAX_SIGNAL_VALUE);
        fprintf(fp, "%.2f ", N);
    }

    fclose(fp);

    return 0;
}