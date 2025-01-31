#include <stdio.h>
#include <assert.h>

#include "histogramUtils.h"

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

// Histogram based on strategy II with privatization
__global__ void histogram(unsigned char *buffer, long size, unsigned int *histogram, unsigned int bins)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int bin_size = (int)(26 - 1) / bins + 1;

	// Privatized bins
	extern __shared__ unsigned int s_histogram[];
	for (unsigned int binIdx = threadIdx.x; binIdx < bins; binIdx += blockDim.x)
	{
		s_histogram[binIdx] = 0u;
	}
	__syncthreads();

	// Histogram
	for (unsigned int t = i; t < size; t += blockDim.x * gridDim.x)
	{
		int alphabet_position = buffer[t] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26)
		{
			atomicAdd(&(s_histogram[alphabet_position / bin_size]), 1);
		}
	}
	__syncthreads();

	// Commit to global memory
	for (unsigned int binIdx = threadIdx.x; binIdx < bins; binIdx += blockDim.x)
	{
		// printf("%d\n", s_histogram[binIdx]);
		atomicAdd(&(histogram[binIdx]), s_histogram[binIdx]);
	}
}

#define NUMBER_OF_STREAMS 10
#define NUMBER_OF_BINS 7

int main(int argc, char **argv)
{
	// check if number of input args is correct: input and output image filename
	if (argc != 2)
	{
		printf("Wrong number of arguments: exactly 1 arguments needed (input .txt filename)\n");
		return 0;
	}

	long size = getCharsNo(argv[1]) + 1;
	unsigned char *h_buffer = (unsigned char *)malloc(size * sizeof(unsigned char));
	readFile(argv[1], size, h_buffer);

	///////////////////////////////////////////////////////
	//@@ INSERT CODE HERE
	// alloc pinned host memory for input data and histograms
    unsigned int *h_histogram;
    cudaHostAlloc((void **)&h_histogram, NUMBER_OF_STREAMS * NUMBER_OF_BINS * sizeof(unsigned int), cudaHostAllocDefault);
    memset(h_histogram, 0 ,NUMBER_OF_STREAMS * NUMBER_OF_BINS * sizeof(unsigned int));

    unsigned char *ph_buffer;
    cudaHostAlloc((void **)&ph_buffer, size * sizeof(unsigned char), cudaHostAllocDefault);
    memcpy(ph_buffer, h_buffer, size*sizeof(unsigned char));

	// alloc device data
    unsigned char *d_buffer;
    unsigned int *d_histogram;
    cudaMalloc((void **)&d_buffer, size * sizeof(unsigned char));
    cudaMalloc((void **)&d_histogram, NUMBER_OF_STREAMS * NUMBER_OF_BINS * sizeof(unsigned int));

	// create timing events
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float ms;

	// baseline case - sequential transfer and execute
    checkCuda (cudaEventRecord(startEvent, 0));
    
    cudaMemcpy(d_buffer, ph_buffer, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    histogram<<<256, 512, NUMBER_OF_BINS * sizeof(unsigned int)>>>(d_buffer, size, d_histogram, NUMBER_OF_BINS);
    cudaMemcpy(h_histogram, d_histogram, NUMBER_OF_STREAMS * NUMBER_OF_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // stop recording and print the measured time
    checkCuda (cudaEventRecord(stopEvent, 0));
    checkCuda (cudaEventSynchronize(stopEvent));
    checkCuda (cudaEventElapsedTime(& ms , startEvent , stopEvent));
    printf ("\n Time for sequential transfer and execute (ms): %f\n", ms);
	for (int i = 0; i < NUMBER_OF_BINS; i++)
    {
        printf("%d ", h_histogram[i]);
    }

    //reset histogram
    cudaMemset(d_histogram, 0, NUMBER_OF_STREAMS * NUMBER_OF_BINS * sizeof(unsigned int));
    memset(h_histogram, 0, NUMBER_OF_STREAMS * NUMBER_OF_BINS * sizeof(unsigned int));
	// create streams
    cudaStream_t streams[NUMBER_OF_STREAMS];
    for (int i = 0; i < NUMBER_OF_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    // create streamSize
    long streamSize = ceil((float)size / NUMBER_OF_STREAMS);

	// asynchronous version
    checkCuda (cudaEventRecord(startEvent, 0));
    for (int i = 0; i < NUMBER_OF_STREAMS; i++)
    {
        long front = min(i*streamSize, size);
        long end = min((i+1)*streamSize, size);
        cudaMemcpyAsync(d_buffer + front, ph_buffer + front, (end-front) * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[i]);
        histogram<<<256, 512, NUMBER_OF_BINS * sizeof(unsigned int), streams[i]>>>(d_buffer+front, end-front, d_histogram + i * NUMBER_OF_BINS, NUMBER_OF_BINS);
        cudaMemcpyAsync(h_histogram + i * NUMBER_OF_BINS, d_histogram + i * NUMBER_OF_BINS, NUMBER_OF_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[i]);
    }
    //stream syncho
    for (int i=0; i<NUMBER_OF_STREAMS; i++)
    {
        cudaStreamSynchronize(streams[i]);
    }
    checkCuda (cudaEventRecord(stopEvent, 0)) ;
    checkCuda (cudaEventSynchronize(stopEvent));
    checkCuda (cudaEventElapsedTime(& ms, startEvent, stopEvent));
    printf ("\n Time for asynchronous transfer and execute (ms): %f\n", ms);
    
	// accumulate histograms
    for (int i=1; i<NUMBER_OF_STREAMS; i++)
    {
        for (int j=0; j<NUMBER_OF_BINS;j++)
        {
            h_histogram[j] += h_histogram[i*NUMBER_OF_BINS+j];
        }
    }
    for (int i=0; i<NUMBER_OF_BINS; i++)
    {
        printf("%i ", h_histogram[i]);
    }
    for (int i=0; i<NUMBER_OF_STREAMS; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    printf("\n");
	// cleanup
	free(h_buffer);
	cudaFreeHost(h_histogram);
   	cudaFreeHost(ph_buffer);
    cudaFree(d_buffer);
	cudaFree(d_histogram);


	///////////////////////////////////////////////////////

	return 0;
}
