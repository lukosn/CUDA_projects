#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RND_SEED 13 // for tests reproducibility

const int MAX_VAL = 100;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Wrong number of arguments: exactly 2 arguments needed (length of vector and filename)\n");
        return 1;
    }

    // read sizes
    int lengthSize = atoi(argv[1]);

    // test reproducibility for given size
    srand(RND_SEED ^ lengthSize);

    // open file
    FILE *fp;
    fp = fopen(argv[2], "w");
    if (fp == NULL)
    {
        printf("%s: cannot open file.\n", argv[3]);
        return 2;
    }

    // generate & write data
    fprintf(fp, "%d\n", lengthSize);
    for (int i = 0; i < lengthSize; ++i)
    {
        float N = rand() % MAX_VAL;
        fprintf(fp, "%.2f ", N);
    }

    fclose(fp);

    return 0;
}