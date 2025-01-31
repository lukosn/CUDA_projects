#include <stdio.h>
#include <stdlib.h>

#define SUCCESS 0
#define NO_FILE 1
#define NO_MEMO 2

int getSize(const char *fileName, int *dataSize)
{
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    fscanf(fp, "%d", dataSize);

    fclose(fp);
    return SUCCESS;
}

int readData(const char *fileName, float **inputValues, int *inputSize)
{
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    // Read size
    fscanf(fp, "%d", inputSize);

    // Allocate memory
    *inputValues = (float *)malloc((*inputSize) * sizeof(float));
    if ((*inputValues) == NULL)
    {
        return NO_MEMO;
    }

    // Read values
    for (int i = 0; i < (*inputSize); ++i)
    {
        fscanf(fp, "%f ", (*inputValues) + i);
    }

    fclose(fp);
    return SUCCESS;
}

int writeData(const char *fileName, const float *data, const int dataLen)
{
    FILE *fp;

    fp = fopen(fileName, "w");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    for (int i = 0; i < dataLen; ++i)
    {
        fprintf(fp, "%.2f ", data[i]);
    }

    fclose(fp);
    return SUCCESS;
}