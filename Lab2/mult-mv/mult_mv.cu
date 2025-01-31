#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define RND_SEED 13 // for tests reproducibility
#define TITLE_WIDTH 16

int createInputs(float **A, float **x, int size)
{
	// input test
	if (size <= 0)
	{
		fprintf(stderr, "Size must be greater than 0.\n");
		return 1;
	}

	// allocate memory
	*A = (float *)malloc(size * size * sizeof(float));
	if (*A == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for matrix A.\n");
		return 2;
	}

	*x = (float *)malloc(size * sizeof(float));
	if (*x == NULL)
	{
		fprintf(stderr, "Cannot allocate memory for vector x.\n");
		return 2;
	}

	// fill with pseudo-random values
	srand(RND_SEED);
	for (int i = 0; i < size * size; ++i)
	{
		(*A)[i] = rand();
		if (i < size)
		{
			(*x)[i] = rand();
		}
	}

	return 0;
}

//@@ INSERT CODE HERE
__global__ void multMatrixVector(float *b, float *A, float *x, unsigned int size)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if( idx < size){
		float sum = 0; 
		for(int i = 0; i<size; i++){
			sum += A[idx*size + i] * x[i];
		}
		b[idx] = sum;
	}
}
//

int main(int argc, char **argv)
{
	// check if number of input args is correct
	if (argc != 2)
	{
		printf("Wrong number of arguments: exactly 1 argument needed (vector length)\n");
		return 0;
	}
	int length = atoi(argv[1]);

	// create input data
	float *h_A = NULL;
	float *h_x = NULL;
	int status = createInputs(&h_A, &h_x, length);
	if (status != 0)
	{
		return status;
	}

	///////////////////////////////////////////////////////
	//@@ INSERT CODE HERE
	float *deviceInputVect;
	float *deviceInputMatrix;
	float *deviceOutputVect;
	
	cudaMalloc(&deviceInputVect, length * sizeof(float));
	cudaMalloc(&deviceInputMatrix, length * length * sizeof(float));
	cudaMalloc(&deviceOutputVect, length * sizeof(float));

	cudaMemcpy(deviceInputVect, h_x, length * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputMatrix, h_A, length * length * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil((float)length / TITLE_WIDTH), 1, 1);
	dim3 dimBlock(TITLE_WIDTH, 1, 1);

	multMatrixVector<<<dimGrid, dimBlock>>>(deviceOutputVect, deviceInputMatrix, deviceInputVect, length);

	float *hostOutputVect = (float *)malloc(length * sizeof(float));
	cudaMemcpy(hostOutputVect, deviceOutputVect, length * sizeof(float), cudaMemcpyDeviceToHost);

	// save output values to file
	FILE *fp = fopen("mult_mv_out.txt", "w");
	if (fp == NULL)
	{
		fprintf(stderr, "Cannot open output file!\n");
	}
	else
	{
		for (int i = 0; i < length; ++i)
		{
			fprintf(fp, "%.0f ", hostOutputVect[i]);
		}
		fclose(fp);
	}

	// free memory
	free(h_A);
	free(h_x);
	free(hostOutputVect);
	cudaFree(deviceInputVect);
	cudaFree(deviceInputMatrix);
	cudaFree(deviceOutputVect);

	///////////////////////////////////////////////////////

	return 0;
}
