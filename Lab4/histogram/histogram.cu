#include "histogramUtils.h"
#include <stdio.h>

#define N_LETTERS 26

//@@ INSERT CODE HERE
// Histogram - basic parallel implementation
__global__ void histogram_1(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	int binWidth = ceil (26.0/ nBins ) ;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < size) {
        int alphabetPosition = buffer[tid] - 'a';
        if (alphabetPosition >= 0 && alphabetPosition < 26) {
            atomicAdd(&histogram[alphabetPosition / binWidth], 1);
        }
        tid += blockDim.x * gridDim.x;
    }
}


//@@ INSERT CODE HERE
// Histogram - interleaved partitioning
__global__ void histogram_2(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int binWidth = (N_LETTERS + nBins - 1) / nBins;

    while (tid < size)
    {
        int alphabetPosition = buffer[tid] - 'a';
        if (alphabetPosition >= 0 && alphabetPosition < 26)
        {
            atomicAdd(&histogram[alphabetPosition / binWidth], 1);
        }
        tid += blockDim.x * gridDim.x;
    }
}

//@@ INSERT CODE HERE
// Histogram - interleaved partitioning + privatisation
__global__ void histogram_3(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
	extern __shared__ unsigned int ds_histogram[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int binWidth = (N_LETTERS + nBins - 1) / nBins;

    int localThreadId = threadIdx.x;
    for (int i = localThreadId; i < nBins; i += blockDim.x)
    {
        ds_histogram[i] = 0;
    }

    __syncthreads();

    while (tid < size)
    {
        int alphabetPosition = buffer[tid] - 'a';
        if (alphabetPosition >= 0 && alphabetPosition < 26)
        {
            atomicAdd(&ds_histogram[alphabetPosition / binWidth], 1);
        }
        tid += blockDim.x * gridDim.x;
    }

    __syncthreads();

    for (int i = localThreadId; i < nBins; i += blockDim.x)
    {
        atomicAdd(&histogram[i], ds_histogram[i]);
    }
}

//@@ EXTRA: INSERT CODE HERE
// Extra: Histogram - interleaved partitioning + privatisation + aggregation
__global__ void histogram_4(unsigned char *buffer, long size, unsigned int *histogram, unsigned int nBins)
{
}

int main(int argc, char **argv)
{
	// check if number of input args is correct: input text filename
	if (argc < 2 || argc > 3)
	{
		printf("Wrong number of arguments! Expecting 1 mandatory argument (input .txt filename) and 1 optional argument (number of bins). \n");
		return 0;
	}

	// read input string
	long size = getCharsNo(argv[1]) + 1;
	unsigned char *h_buffer1 = (unsigned char *)malloc(size * sizeof(unsigned char));
	unsigned char *h_buffer2 = (unsigned char *)malloc(size * sizeof(unsigned char));
	unsigned char *h_buffer3 = (unsigned char *)malloc(size * sizeof(unsigned char));

	readFile(argv[1], size, h_buffer1);
	readFile(argv[1], size, h_buffer2);
	readFile(argv[1], size, h_buffer3);
	printf("Input string size: %ld\n", size);

	// set number of bins
	int nBins = 7;
	if (argc > 2)
	{
		int inBinsVal = atoi(argv[2]);
		if (inBinsVal > 0 && inBinsVal <= N_LETTERS)
		{
			nBins = inBinsVal;
		}
		else
		{
			fprintf(stderr, "Incorrect input number of bins: %d. Proceeding with default value: %d.\n", inBinsVal, nBins);
		}
	}

	///////////////////////////////////////////////////////
	//@@ INSERT CODE HERE
	 // Allocate memory for the histogram on the host
    unsigned int *h_histogram1 = (unsigned int *)malloc(N_LETTERS * sizeof(unsigned int));
	unsigned int *h_histogram2 = (unsigned int *)malloc(N_LETTERS * sizeof(unsigned int));
	unsigned int *h_histogram3 = (unsigned int *)malloc(N_LETTERS * sizeof(unsigned int));

    // Allocate memory for the input text buffer and histogram on the device
    unsigned char *d_buffer1;
    unsigned int *d_histogram1;
    cudaMalloc((void **)&d_buffer1, size * sizeof(unsigned char));
    cudaMalloc((void **)&d_histogram1, N_LETTERS * sizeof(unsigned int));

	unsigned char *d_buffer2;
    unsigned int *d_histogram2;
    cudaMalloc((void **)&d_buffer2, size * sizeof(unsigned char));
    cudaMalloc((void **)&d_histogram2, N_LETTERS * sizeof(unsigned int));

	unsigned char *d_buffer3;
    unsigned int *d_histogram3;
    cudaMalloc((void **)&d_buffer3, size * sizeof(unsigned char));
    cudaMalloc((void **)&d_histogram3, N_LETTERS * sizeof(unsigned int));

    // Copy the input text buffer from the host to the device
    cudaMemcpy(d_buffer1, h_buffer1, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_buffer2, h_buffer2, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_buffer3, h_buffer3, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Zero out the histogram on the device
    cudaMemset(d_histogram1, 0, N_LETTERS * sizeof(unsigned int));
	cudaMemset(d_histogram2, 0, N_LETTERS * sizeof(unsigned int));
	cudaMemset(d_histogram3, 0, N_LETTERS * sizeof(unsigned int));

    // Define grid and block dimensions as needed
    int blockSize = 256;  // You can adjust this based on your GPU's capabilities
	int gridSize = 60;

	// Launch the histogram_1
	histogram_1<<<gridSize, blockSize>>>(d_buffer1, size, d_histogram1, nBins);

	// Launch the histogram_2
	histogram_2<<<gridSize, blockSize>>>(d_buffer2, size, d_histogram2, nBins);

	// Calculate shared memory size for histogram_3 (equal to the number of bins)
    int sharedMemorySize = nBins * sizeof(unsigned int);

	// Launch the histogram_3 
	histogram_3<<<gridSize, blockSize, sharedMemorySize>>>(d_buffer3, size, d_histogram3, nBins);


    // Copy the histogram data back from the device to the host
    cudaMemcpy(h_histogram1, d_histogram1, N_LETTERS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_histogram2, d_histogram2, N_LETTERS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_histogram3, d_histogram3, N_LETTERS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Write the histogram data to an output file
    // Write the histogram data to an output file
	writeFile(const_cast<char*>("histogram1_output.txt"), h_histogram1, nBins);
	writeFile(const_cast<char*>("histogram2_output.txt"), h_histogram2, nBins);
	writeFile(const_cast<char*>("histogram3_output.txt"), h_histogram3, nBins);

	printf("Histogram 1:\n");
	for (int i = 1; i <= nBins; i++)
	{
		printf("Bin %d: %d\n", i, h_histogram1[i - 1]);
	}

	printf("Histogram 2:\n");
	for (int i = 1; i <= nBins; i++)
	{
		printf("Bin %d: %d\n", i, h_histogram2[i - 1]);
	}

	printf("Histogram 3:\n");
	for (int i = 1; i <= nBins; i++)
	{
		printf("Bin %d: %d\n", i, h_histogram3[i - 1]);
	}


    // Free allocated memory on both the host and the device
    free(h_buffer1);
    free(h_histogram1);
    cudaFree(d_buffer1);
    cudaFree(d_histogram1);

	free(h_buffer2);
    free(h_histogram2);
    cudaFree(d_buffer2);
    cudaFree(d_histogram2);

	free(h_buffer3);
    free(h_histogram3);
    cudaFree(d_buffer3);
    cudaFree(d_histogram3);
	///////////////////////////////////////////////////////

	return 0;
}
