#include <stdio.h>
#include <cuda.h>

#define RND_SEED 13 // for tests reproducibility

int generateRandomFlattenMatrix(float *M, int size)
{
	if (size <= 0)
	{
		fprintf(stderr, "Matrix size must be greater than 0.\n");
		return 1;
	}

	srand(RND_SEED);
	for (int i = 0; i < size; ++i)
	{
		M[i] = (rand() % 20) + 50;
	}

	return 0;
}

//@@ INSERT CODE HERE
// 1 thread 1 element
__global__ void addMMv1(float *A, float *B, float *C, int size)
{
}

// 1 thread 1 col
__global__ void addMMv2(float *A, float *B, float *C, int size)
{
}

// 1 thread 1 row
__global__ void addMMv3(float *A, float *B, float *C, int size)
{
}
//

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("Wrong number of arguments: exactly 1 argument needed (size of the matrices)\n");
		return 0;
	}
	int size = atoi(argv[1]);

	///////////////////////////////////////////////////////
	//@@ INSERT CODE HERE

	// print v1 results
	// FILE *fp = fopen("add_mm_out.txt", "w");
	// if (fp == NULL)
	// {
	// 	fprintf(stderr, "Cannot open output file!\n");
	// }
	// else
	// {
	// 	fprintf(fp, "--- v1 ---\n");
	// 	for (int i = 0; i < size * size; ++i)
	// 	{
	// 		fprintf(fp, "%f ", h_A[i]);
	// 	}
	// 	fprintf(fp, "\n");
	// 	fclose(fp);
	// }

	///////////////////////////////////////////////////////

	return 0;
}
