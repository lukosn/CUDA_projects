#ifndef _MATUTILS_H_
#define _MATUTILS_H_
#include <stdio.h>
#include <stdlib.h>

#define MAT_SUCCESS 0
#define MAT_NO_FILE 1
#define MAT_MEM_NOT_ALLOCATED 2
#define MAT_OVERSIZE 3

#define RND_SEED 13 // for tests reproducibility

int getMatSize(const char *fileName, int *matrixSize)
{
    // Open input file
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        return MAT_NO_FILE;
    }

    // Read matrix size
    fscanf(fp, "%d", matrixSize);

    // Close file
    fclose(fp);
    return MAT_SUCCESS;
}

int readMat(const char *fileName, float *mat, int matSize)
{
    // Check matrix memory
    if (mat == NULL)
    {
        return MAT_MEM_NOT_ALLOCATED;
    }

    // Open file
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        return MAT_NO_FILE;
    }

    // Check matrix size
    int matSizeRead = 0;
    fscanf(fp, "%d", &matSizeRead);
    if (matSize > matSizeRead)
    {
        return MAT_OVERSIZE;
    }

    // Read matrix values
    for (int i = 0; i < matSize; ++i)
    {
        fscanf(fp, "%f ", mat + i);
    }

    // Close file
    fclose(fp);
    return MAT_SUCCESS;
}

int generateMat(const char *fileName, float *mat, int matSize)
{
    // Check matrix memory
    if (mat == NULL)
    {
        return MAT_MEM_NOT_ALLOCATED;
    }

    // Generate random matrix values
    for (int i = 0; i < matSize; ++i)
    {
        mat[i] = (rand() % 20) + 50;
    }

    // Open file
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL)
    {
        return MAT_NO_FILE;
    }

    // Write matrix size and values
    fprintf(fp, "%d\n", matSize);
    for (int i = 0; i < matSize; ++i)
    {
        fprintf(fp, "%.0f ", mat[i]);
    }

    // Close file
    fclose(fp);
    return MAT_SUCCESS;
}

#endif