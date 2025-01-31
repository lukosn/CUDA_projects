#include <stdio.h>

#define SUCCESS 0
#define NO_FILE 1

/*** Convolution 1D ***/

int getSizes1D(const char *fileName, int *maskSize, int *signalSize)
{
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    fscanf(fp, "%d %d", maskSize, signalSize);

    fclose(fp);
    return SUCCESS;
}

int getValues1D(const char *fileName, float *maskValues, float *signalValues)
{
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    int maskSize, signalSize;
    fscanf(fp, "%d %d", &maskSize, &signalSize);

    // Read mask
    for (int i = 0; i < maskSize; ++i)
    {
        fscanf(fp, "%f ", maskValues + i);
    }

    // Read signal
    for (int i = 0; i < signalSize; ++i)
    {
        fscanf(fp, "%f ", signalValues + i);
    }

    fclose(fp);
    return SUCCESS;
}

int writeData1D(const char *fileName, const float *data, const int dataLen)
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

/*** Convolution 2D ***/
int getSizes2D(const char *fileName, int *maskSize, int *signalWidth, int *signalPitch, int *signalHeight)
{
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    fscanf(fp, "%d", maskSize);
    fscanf(fp, "%d %d %d", signalWidth, signalPitch, signalHeight);

    fclose(fp);
    return SUCCESS;
}

int getValues2D(const char *fileName, float *maskValues, float *signalValues)
{
    FILE *fp;

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    int maskSize, signalWidth, signalPitch, signalHeight;
    fscanf(fp, "%d", &maskSize);
    fscanf(fp, "%d %d %d", &signalWidth, &signalPitch, &signalHeight);

    // Read mask
    for (int i = 0; i < maskSize * maskSize; ++i)
    {
        fscanf(fp, "%f ", maskValues + i);
    }

    // Read signal
    for (int i = 0; i < signalHeight * signalPitch; ++i)
    {
        fscanf(fp, "%f ", signalValues + i);
    }

    fclose(fp);
    return SUCCESS;
}

int writeData2D(const char *fileName, const float *data, const int dataHeight, const int dataPitch)
{
    FILE *fp;

    fp = fopen(fileName, "w");
    if (fp == NULL)
    {
        return NO_FILE;
    }

    for (int i = 0; i < dataHeight; ++i)
    {
        for (int j = 0; j < dataPitch; ++j)
        {
            fprintf(fp, "%.2f ", data[i * dataPitch + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return SUCCESS;
}
