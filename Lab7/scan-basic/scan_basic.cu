#include "scanFiles.h"

#include <stdio.h>

#define SECTION_SIZE 512

void scanSequential(float *output, float *input, int width)
{
    float accumulator = input[0];
    output[0] = accumulator;
    for (int i = 1; i < width; ++i)
    {
        accumulator += input[i];
        output[i] = accumulator;
    }
}

__global__ void scanKernel(float *input, float *output, int width)
{
    //@@ INSERT CODE HERE
    int tid = threadIdx.x;

    extern __shared__ float XY[];
    
    if(tid < width)
    {
        XY[tid] = input[tid];
    }
    __syncthreads();

    for(int stride = 1; stride < width; stride *= 2)
    {
        int index = tid + stride;
        if(index < width)
        {
            XY[index] += XY[index - stride];
        }
        __syncthreads();
    }
    if(tid < width)
    {
        output[tid] = XY[tid];
    }
}

void launchScanKernel(float *h_output, float *h_input, int width)
{
    //@@ INSERT CODE HERE
     float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc((void **)&d_input, width * sizeof(float));
    cudaMalloc((void **)&d_output, width * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = SECTION_SIZE;
    int numBlocks = 1;

    int sharedMemSize = blockSize * sizeof(float);
    scanKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_output, width);

    // Copy output data to host

    cudaMemcpy(h_output, d_output, width * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char *argv[])
{

    // check if number of input args is correct: input and output image filename
    if (argc != 2)
    {
        printf("Wrong number of arguments: exactly 1 arguments needed (input .txt filename)\n");
        return 1;
    }

    // read data
    int inputSize;
    float *inputData = NULL;
    int status = readData(argv[1], &inputData, &inputSize);
    if (status == NO_FILE)
    {
        fprintf(stderr, "%s: No such file or directory.\n", argv[1]);
        return 2;
    }
    else if (status == NO_MEMO)
    {
        fprintf(stderr, "Cannot allocate memory for the input data.\n");
        return 3;
    }

    // reference output
    float *outputRef = (float *)malloc(inputSize * sizeof(float));
    scanSequential(outputRef, inputData, inputSize);

    // launch kernel
    float *outputScan = (float *)malloc(inputSize * sizeof(float));
    launchScanKernel(outputScan, inputData, inputSize);

    // check results
    int nErr = 0;
    for (int i = 0; i < inputSize; ++i)
    {
        if (outputRef[i] != outputScan[i])
        {
            nErr++;
            printf("Error at [%d]: %f seq vs %f par\n", i, outputRef[i], outputScan[i]);
        }
    }
    if (nErr == 0)
    {
        printf("Scan Kernel OK!\n");
    }
    else
    {
        printf("Scan Kernel FAIL! %d/%d errors detected.\n", nErr, inputSize);
    }

    // write output data
    writeData("outSequential.txt", outputRef, inputSize);
    writeData("outParallel.txt", outputScan, inputSize);

    // clean
    free(inputData);
    free(outputRef);
    free(outputScan);

    return 0;
}