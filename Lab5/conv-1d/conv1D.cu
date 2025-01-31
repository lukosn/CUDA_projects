#include "convFiles.h"
#include <stdio.h>

#define TILE_SIZE 1024
#define MAX_MASK_WIDTH 15

__constant__ float c_M[MAX_MASK_WIDTH];

__global__ void basicConvolution1D(float *N, float *M, float *P, const int width, const int maskWidth)
{

    //@@ INSERT CODE HERE
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < width)
    {
        float Pvalue = 0;

        for(int j = 0; j < maskWidth; j++)
        {
            int N_index = i + j - maskWidth / 2;

            if(N_index >= 0 && N_index < width)
            {
                Pvalue += N[N_index] * c_M[j];
            }
        }

        P[i] = Pvalue;
    }

}

__global__ void tiledConvolution1D(float *N, float *P, const int width, const int maskWidth)
{
    //@@ INSERT CODE HERE
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sharedN[TILE_SIZE - 2 + MAX_MASK_WIDTH];
    int localIndex = threadIdx.x + MAX_MASK_WIDTH;

    if (i < width)
    {
        sharedN[localIndex] = N[i];
        
        if (threadIdx.x < MAX_MASK_WIDTH)
        {
            int leftIndex = i - MAX_MASK_WIDTH;
            int rightIndex = i + blockDim.x;

            if (leftIndex >= 0)
                sharedN[threadIdx.x] = N[leftIndex];
            else
                sharedN[threadIdx.x] = 0.0f;

            if (rightIndex < width)
                sharedN[threadIdx.x + blockDim.x + MAX_MASK_WIDTH] = N[rightIndex];
            else
                sharedN[threadIdx.x + blockDim.x + MAX_MASK_WIDTH] = 0.0f;
        }

        __syncthreads();

        float Pvalue = 0.0f;

        
        for (int j = 0; j < maskWidth; j++)
        {
            Pvalue += sharedN[localIndex + j - maskWidth / 2] * c_M[j];
        }

        P[i] = Pvalue;
    }
}

void launchBasicConvolution1D(float *h_N, float *h_M, float *h_P, const int width, const int maskWidth)
{
    //@@ INSERT CODE HERE
    float *d_N, *d_P;
    float d_M[MAX_MASK_WIDTH];

    cudaMalloc((void **)&d_N, width * sizeof(float));
    cudaMalloc((void **)&d_P, width * sizeof(float));
    
    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_M, h_M, maskWidth * sizeof(float));

    int blockSize = TILE_SIZE;
    int gridSize = (width + blockSize - 1) / blockSize;

    basicConvolution1D<<<gridSize, blockSize>>>(d_N, d_M, d_P, width, maskWidth);

    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_N);
    cudaFree(d_P);
}

void launchTiledConvolution1D(float *h_N, float *h_P, const int width, const int maskWidth)
{
    //@@ INSERT CODE HERE
    float *d_N, *d_P;

    cudaMalloc((void **)&d_N, width * sizeof(float));
    cudaMalloc((void **)&d_P, width * sizeof(float));

    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = TILE_SIZE;
    int gridSize = (width + blockSize - 1) / blockSize;

    tiledConvolution1D<<<gridSize, blockSize>>>(d_N, d_P, width, maskWidth);

    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_N);
    cudaFree(d_P);
}

int main(int argc, char *argv[])
{
    // check if number of input args is correct: input filename
    if (argc != 2)
    {
        printf("Wrong number of arguments: exactly 1 argument needed (input .txt filename)\n");
        return 1;
    }

    // output names
    char nameOutBasic[] = "out_1D_basic.txt";
    char nameOutTiled[] = "out_1D_tiled.txt";

    // read sizes
    int width, maskWidth;
    int status = getSizes1D(argv[1], &maskWidth, &width);
    if (status == NO_FILE)
    {
        printf("%s: No such file or directory,\n", argv[1]);
        return 2;
    }

    // read data
    float *N = (float *)malloc(width * sizeof(float));
    float *M = (float *)malloc(maskWidth * sizeof(float));
    getValues1D(argv[1], M, N);

    // for the output data
    float *P = (float *)malloc(width * sizeof(float));

    // basic kernel
    launchBasicConvolution1D(N, M, P, width, maskWidth);
    writeData1D(nameOutBasic, P, width);

    // tiled kernel
    //@@ INSERT CODE HERE
    launchTiledConvolution1D(N, P, width, maskWidth);
    writeData1D(nameOutTiled, P, width);

    free(P);
    free(N);
    free(M);

    return 0;
}
