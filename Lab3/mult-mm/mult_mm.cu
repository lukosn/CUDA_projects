#include <stdio.h>
#include "matUtils.h"
using namespace std;

#define RND_SEED 13 // for tests reproducibility

// Compute C = A * B general matrix-matrix multiply
__global__ void standardMatrixMult(float *A, float *B, float *C, int numARows,
                                   int numAColumns, int numBRows, int numBColumns)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns)
    {
        float sum = 0;
        for (int ii = 0; ii < numAColumns; ii++)
        {
            sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

#define TILE_WIDTH 16

//@@ INSERT CODE HERE
// Compute C = A * B tiled matrix-matrix multiply
__global__ void matrixMultiply(float* A, float* B, float* C, int numAColumns, int numARows, int numBColumns,
int numBRows, int numCColumns, int numCRows)
{
    __device__ __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __device__ __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    const unsigned int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const unsigned int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int p=0; p<ceil((float)numAColumns/TILE_WIDTH); p++) 
    {
        if (threadIdx.x+p*TILE_WIDTH<numAColumns && row<numARows) 
        {
            ds_A[threadIdx.y][threadIdx.x] = A[(row*numAColumns) + (p*TILE_WIDTH+threadIdx.x)];
        } 
            else 
            {
                ds_A[threadIdx.y][threadIdx.x] = 0.0f;   
            }  

        if ((threadIdx.y+p*TILE_WIDTH)<numBRows && col<numBColumns) 
        {
            ds_B[threadIdx.y][threadIdx.x] = B[(p*TILE_WIDTH+threadIdx.y) * numBColumns+col];
        }
            else 
            {
              ds_B[threadIdx.y][threadIdx.x] = 0.0f;   
            }

        __syncthreads();
    
       for (int cnt=0; cnt<TILE_WIDTH; cnt++) 
       {
            sum += ds_A[threadIdx.y][cnt] * ds_B[cnt][threadIdx.x];
        }

        __syncthreads();  
    }
    if (row<numCRows && col<numCColumns) 
    {
        C[row*numCColumns+col] = sum;
    }
} 
//

void generateRandomFlattenMatrix(float *M, unsigned int size)
{
    for (int i = 0; i < size; ++i)
    {
        M[i] = (rand() % 20) + 50;
    }
}

int main(int argc, char **argv)
{
    // check if number of input args is correct
    if (argc < 4 || argc > 7)
    {
        printf("Wrong number of arguments: 3 mandatory arguments needed (width A, height A and width B)\n");
        printf("If 4th argument is --read then input matrix are read from the files given as 5th and 6th arguments.\n");
        printf("Example: ./mult_mm.out 5 8 13 --read ./inputA.txt ./inputB.txt");
        return 0;
    }
    int widthA = atoi(argv[1]);
    int heightA = atoi(argv[2]);
    int widthB = atoi(argv[3]);
    int heightB = widthA;
    int widthC = widthB;
    int heightC = heightA;


    int readFile = 0;
    int matAsize = widthA * heightA;
    int matBsize = widthB * widthA;
    int matCsize = widthC * heightC;
    if (argc > 4)
    {
        // Check matrix A size
        int status = getMatSize(argv[5], &matAsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: getMatSize for matrix A, status: %d\n", status);
            return 0;
        }
        if (matAsize != widthA * heightA)
        {
            printf("Matrix A size mismtach: %d vs %d.\n", matAsize, widthA * heightA);
            return 0;
        }

        // Check matrix B size
        status = getMatSize(argv[6], &matBsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: getMatSize for matrix B, status: %d\n", status);
            return 0;
        }
        if (matBsize != widthB * widthA)
        {
            printf("Matrix B size mismtach: %d vs %d.\n", matBsize, widthB * widthA);
            return 0;
        }

        readFile = 1;
    }

    float *h_A = (float *)malloc(matAsize * sizeof(float));
    float *h_B = (float *)malloc(matBsize * sizeof(float));
    float *h_C = (float *)malloc(matCsize * sizeof(float));
    float *h_C_standard = (float *)malloc(matCsize * sizeof(float));

    if (!readFile)
    {
        srand(RND_SEED);

        // Generate matrix A
        int status = generateMat("./inputA.txt", h_A, matAsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: generateMat for matrix A, status: %d\n", status);
            return 0;
        }

        // Generate matrix B
        status = generateMat("./inputB.txt", h_B, matBsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: generateMat for matrix B, status: %d\n", status);
            return 0;
        }
    }
    else
    {
        // Read matrix A
        int status = readMat(argv[5], h_A, matAsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: readMat for matrix A, status: %d\n", status);
            return 0;
        }

        // Read matrix B
        status = readMat(argv[6], h_B, matBsize);
        if (status != MAT_SUCCESS)
        {
            printf("Error: readMat for matrix B, status: %d\n", status);
            return 0;
        }
    }

    //create inputs for device
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    float *d_C_standard = NULL;

    cudaMalloc((void **)&d_A, matAsize * sizeof(float));
    cudaMalloc((void **)&d_B, matBsize * sizeof(float));
    cudaMalloc((void **)&d_C, matCsize * sizeof(float));
    cudaMalloc((void **)&d_C_standard, matCsize * sizeof(float));

    cudaMemcpy(d_A, h_A, matAsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matBsize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(ceil((float) widthB / TILE_WIDTH), ceil((float) heightA / TILE_WIDTH), 1);

    printf("> C=A*B using tailing method\n");
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, widthA, heightA, widthB, heightB, widthC, heightC);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, matCsize * sizeof(float), cudaMemcpyDeviceToHost);

    printf("> C=A*B using standard method\n");
    standardMatrixMult<<<dimGrid, dimBlock>>>(d_A, d_B, d_C_standard, heightA, widthA, heightB, widthB);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C_standard, d_C_standard, matCsize * sizeof(float), cudaMemcpyDeviceToHost);

    ///////////////////////////////////////////////////////
    //@@ INSERT CODE HERE

    // save output values

    bool is_matched=true;

    for (int l = 0; l < heightA * widthB; ++l)
        {
            if (h_C[l]!=h_C_standard[l])
            {
                is_matched=false;
            }
        }

    char filename[100];
    sprintf(filename, "mult_mm_out_%d_%d_%d.txt", widthA, heightA, widthB);
    if (is_matched==true){
        FILE *fp = fopen(filename, "w");
        if (fp == NULL)
        {
            fprintf(stderr, "Cannot open output file!\n");
        }
        else
        {
            printf("Generating output file... ");
            for (int i = 0; i < heightA * widthB; ++i)
            {
                fprintf(fp, "%.0f ", h_C_standard[i]);
            }
            printf("DONE! \n");
            fclose(fp);
        }
    }

    ///////////////////////////////////////////////////////

    return 0;
}
