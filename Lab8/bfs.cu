#include "bfsFiles.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 128
// maximum number of elements that can be inserted into a block queue
#define BLOCK_QUEUE_SIZE 8192

void BFSSequential(int source, int *edges, int *dest, int *label, int nodes)
{
	int *cFrontier = (int *)malloc(nodes * sizeof(int));
	int cFrontierTail = 0;
	int *pFrontier = (int *)malloc(nodes * sizeof(int));
	int pFrontierTail = 0;

	pFrontier[pFrontierTail++] = source;
	while (pFrontierTail > 0)
	{
		// visit all previous frontier vertices
		for (int f = 0; f < pFrontierTail; f++)
		{
			// pick up one of the previous frontier vertices
			int cVertex = pFrontier[f];
			// for all its edges
			for (int i = edges[cVertex]; i < edges[cVertex + 1]; i++)
			{
				// the dest vertex has not been visited
				if (label[dest[i]] == -1)
				{
					cFrontier[cFrontierTail++] = dest[i];
					label[dest[i]] = label[cVertex] + 1;
				}
			}
		}
		// swap previous and current
		int *temp = cFrontier;
		cFrontier = pFrontier;
		pFrontier = temp;
		pFrontierTail = cFrontierTail;
		cFrontierTail = 0;
	}

	free(cFrontier);
	free(pFrontier);
}

__global__ void BFSKernelGlobalQueue(int *edges, int *dest, int *label, int *pFrontier, int *cFrontier, int *pFrontierTail, int *cFrontierTail)
{
	//@@ INSERT KERNEL CODE HERE

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    for (int idx = threadId; idx < *pFrontierTail; idx += blockDim.x * gridDim.x) 
	{
        int cVertex = pFrontier[idx];

        for (int i = edges[cVertex]; i < edges[cVertex + 1]; i++) 
		{
            int neighbor = dest[i];
            if (atomicCAS(&label[neighbor], -1, label[cVertex] + 1) == -1) 
			{
                int pos = atomicAdd(cFrontierTail, 1);
                cFrontier[pos] = neighbor;
            }
        }
    }
}

__global__ void BFSKernelBlockQueue(int *edges, int *dest, int *label, int *pFrontier, int *cFrontier, int *pFrontierTail, int *cFrontierTail)
{
	//@@ INSERT KERNEL CODE HERE
}

void BFSHost(int mode, int *h_edges, int *h_dest, int *h_label, int nodes, int *h_pFrontier, int *h_pFrontierTail)
{
	//@@ INSERT HOST CODE HERE

	// allocate edges, dest, label in device global memory
	int *d_edges;
	int *d_dest;
	int *d_label;
	int *h_cFrontier = new int[nodes];
	int h_cFrontierTail = 0;

	cudaMalloc((void **)&d_edges, (nodes+1) * sizeof(int));
	cudaMalloc((void **)&d_dest, h_edges[nodes] * sizeof(int));
	cudaMalloc((void **)&d_label, nodes * sizeof(int));

	// allocate pFrontier, cFrontier, cFrontierTail, pFrontierTail in device global memory
	int *d_pFrontier;
	int *d_cFrontier;
	int *d_pFrontierTail;
	int *d_cFrontierTail;

	cudaMalloc((void **)&d_pFrontier, nodes * sizeof(int));
	cudaMalloc((void **)&d_cFrontier, nodes * sizeof(int));
	cudaMalloc((void **)&d_pFrontierTail, sizeof(int));
	cudaMalloc((void **)&d_cFrontierTail, sizeof(int));

	// copy the data from host to device
	cudaMemcpy(d_edges, h_edges, (nodes+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dest, h_dest, h_edges[nodes] * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_label, h_label, nodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pFrontier, h_pFrontier, nodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cFrontier, h_cFrontier, nodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pFrontierTail, h_pFrontierTail, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cFrontierTail, h_pFrontierTail, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(d_cFrontierTail, 0, sizeof(int));

	// launch a kernel in a loop
	int gridSize = ceil((float)*h_pFrontierTail/BLOCK_SIZE);
	while (*h_pFrontierTail > 0)
	{
		if (mode == 0)
			BFSKernelGlobalQueue<<<gridSize, BLOCK_SIZE>>>(d_edges, d_dest, 
			d_label, d_pFrontier, d_cFrontier, d_pFrontierTail, d_cFrontierTail);

		// read the current frontier and copy it from device to host
		cudaMemcpy(h_cFrontier, d_cFrontier, nodes * sizeof(int), cudaMemcpyDeviceToHost);

		// swap the roles of the frontiers
		int *temp = h_cFrontier;
		cudaMemcpy(d_cFrontier, h_pFrontier, nodes * sizeof(int), cudaMemcpyHostToDevice);
		h_pFrontier = temp;
		cudaMemcpy(d_pFrontier, h_pFrontier, nodes * sizeof(int), cudaMemcpyHostToDevice);

		// set pFrontierTail and cFrontierTail
		cudaMemcpy(&h_cFrontierTail, d_cFrontierTail, sizeof(int), cudaMemcpyDeviceToHost);
		*h_pFrontierTail = h_cFrontierTail;

		cudaMemcpy(d_pFrontierTail, h_pFrontierTail, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_cFrontierTail, 0, sizeof(int));
	}

	// copy data to label
	cudaMemcpy(h_label, d_label, nodes * sizeof(int), cudaMemcpyDeviceToHost);

	// free device memory
	cudaFree(d_edges);
	cudaFree(d_dest);
	cudaFree(d_label);
	cudaFree(d_pFrontier);
	cudaFree(d_cFrontier);
	cudaFree(d_pFrontierTail);
	cudaFree(d_cFrontierTail);
}

int main(int argc, char *argv[])
{
	// check if number of input args is correct: input and output image filename
	if (argc != 5)
	{
		printf("Wrong number of arguments: exactly 4 arguments needed (1 input and 3 output .txt filenames, suggested: out_seq.txt, out_global.txt, out_block.txt)\n");
		return 1;
	}

	int source;
	int *edges;
	int *dest;
	int *labelSeq;
	int *labelGlobal;
	int *labelBlock;
	int nodes;
	int *pFrontierGlobal;
	int *pFrontierBlock;
	int pFrontierTailGlobal;
	int pFrontierTailBlock;

	int status = readFile(argv[1], &source, &edges, &dest, &labelSeq, &labelGlobal, &labelBlock, &nodes, &pFrontierGlobal, &pFrontierBlock, &pFrontierTailGlobal, &pFrontierTailBlock);
	if (status != SUCCESS)
	{
		printf("Cannot read from file!\n");
		return 2;
	}

	BFSSequential(source, edges, dest, labelSeq, nodes);
	BFSHost(0, edges, dest, labelGlobal, nodes, pFrontierGlobal, &pFrontierTailGlobal);
	BFSHost(1, edges, dest, labelBlock, nodes, pFrontierBlock, &pFrontierTailBlock);

	writeFile(argv[2], &labelSeq, &nodes);
	writeFile(argv[3], &labelGlobal, &nodes);
	writeFile(argv[4], &labelBlock, &nodes);

	free(pFrontierBlock);
	free(pFrontierGlobal);
	free(labelBlock);
	free(labelGlobal);
	free(labelSeq);
	free(dest);
	free(edges);

	return 0;
}
