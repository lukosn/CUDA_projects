#include "ppmIO.h"
#include <stdio.h>

//@@ INSERT CODE HERE

#define TILE_WIDTH 16
__global__ void rgb2gray(float *grayImage, float *rgbImage, int channels, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        // get 1D coordinate for the grayscale image
        int grayOffset = y * width + x;
        // one can think of the RGB image having
        // CHANNEL times columns than the gray scale image
        int rgbOffset = grayOffset * channels;
        float r = rgbImage[rgbOffset];     // red value for pixel
        float g = rgbImage[rgbOffset + 1]; // green value for pixel
        float b = rgbImage[rgbOffset + 2]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        grayImage[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

//@@

int main(int argc, char *argv[])
{

    // check if number of input args is correct: input and output image filename
    if (argc != 3)
    {
        printf("Wrong number of arguments: exactly 2 arguments needed (input and output .ppm filename)\n");
        return 0;
    }

    // get size of input image
    unsigned int width, height;
    getPPMSize(argv[1], &width, &height);
    // read input image to a host variable
    float *hostInputImageData = (float *)malloc(width * height * 3 * sizeof(float));
    readPPM(argv[1], hostInputImageData);

    // allocate input and output images in the device
    float *deviceInputImageData;
    float *deviceOutputImageData;
    cudaMalloc((void **)&deviceInputImageData, width * height * 3 * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData, width * height * sizeof(float));

    // copy image to the device
    cudaMemcpy(deviceInputImageData, hostInputImageData, width * height * 3 * sizeof(float), cudaMemcpyHostToDevice);

    //  ///////////////////////////////////////////////////////
    //  //@@ INSERT CODE HERE

    dim3 dimGrid(ceil((float)width / TILE_WIDTH), ceil((float)height / TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    rgb2gray<<<dimGrid, dimBlock>>>(deviceOutputImageData, deviceInputImageData, 3, width, height);

    //  ///////////////////////////////////////////////////////

    float *hostOutputImageData = (float *)malloc(width * height * sizeof(float));
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    writePPM(argv[2], hostOutputImageData, width, height, 1);

    free(hostInputImageData);
    free(hostOutputImageData);
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

    return 0;
}
