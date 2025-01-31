#include "reductionFiles.h"

#include <stdio.h>

#define BLOCK_SIZE 128

float reductionSequential(float *input, int width)
{
    float sum = 0.0f;
    for (int i = 0; i < width; ++i)
    {
        sum += input[i];
    }

    return sum;
}

__global__ void reductionKernelBasic(float *input, float *output, int width)
{
    //@@ INSERT CODE HERE
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory with input data
    sharedData[tid] = (i < width) ? input[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;

        if (index < blockDim.x)
        {
            sharedData[index] += sharedData[index + s];
        }
        __syncthreads();
    }

    // Write the result to the output array
    if (tid == 0)
    {
        output[blockIdx.x] = sharedData[0];
    }
}

__global__ void reductionKernelOp(float *input, float *output, int width)
{
    //@@ INSERT CODE HERE
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory with input data
    sharedData[tid] = (i < width) ? input[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    // Write the result to the output array
    if (tid == 0)
    {
        output[blockIdx.x] = sharedData[0];
    }
}

float launchReductionKernelBasic(float *h_input, int width)
{
    //@@ INSERT CODE HERE
     float *d_input, *d_output;
    int numBlocks = static_cast<int>(std::ceil(static_cast<float>(width) / BLOCK_SIZE));
    

    // Allocate device memory
    cudaMalloc((void **)&d_input, width * sizeof(float));
    cudaMalloc((void **)&d_output, numBlocks * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    reductionKernelBasic<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, width);
    cudaDeviceSynchronize();

    // Copy output data to host
    float *h_output = (float *)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    float finalResult = 0.0f;
    for (int i = 0; i < numBlocks; ++i)
    {
        finalResult += h_output[i];
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return finalResult;
}

float launchReductionKernelOp(float *h_input, int width)
{
    //@@ INSERT CODE HERE
    float *d_input, *d_output;
    int numBlocks = static_cast<int>(std::ceil(static_cast<float>(width) / BLOCK_SIZE));

    // Allocate device memory
    cudaMalloc((void **)&d_input, width * sizeof(float));
    cudaMalloc((void **)&d_output, numBlocks * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, width * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    reductionKernelOp<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, width);
    cudaDeviceSynchronize();

    // Copy output data to host
    float *h_output = (float *)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on the host
    float finalResult = 0.0f;
    for (int i = 0; i < numBlocks; ++i)
    {
        finalResult += h_output[i];
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_output);

    return finalResult;
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
    float refVal = reductionSequential(inputData, inputSize);
    printf("Reference output: %.2f\n", refVal);

    // launch basic kernel
    float outputBasic = launchReductionKernelBasic(inputData, inputSize);
    if (refVal == outputBasic)
    {
        printf("Basic Kernel OK!\n");
    }
    else
    {
        printf("Basic Kernel FAIL! Output: %.2f\n", outputBasic);
    }

    // launch optimised kernel
    float outputOp = launchReductionKernelOp(inputData, inputSize);
    if (refVal == outputOp)
    {
        printf("Optimised Kernel OK!\n");
    }
    else
    {
        printf("Optimised Kernel FAIL! Output: %.2f\n", outputOp);
    }

    // write output data
    writeData("outBasic.txt", &outputBasic, 1);
    writeData("outOp.txt", &outputOp, 1);

    // clean
    free(inputData);

    return 0;
}