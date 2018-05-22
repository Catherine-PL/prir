#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
using namespace cv;

//method used to catch error from CUDA
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
static int HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		std::cout << "Error" << cudaGetErrorString(err) <<" at line: " << line << std::endl;
        return EXIT_FAILURE;
	}
	return 0;
}

int GAUSS[25] = {1, 1, 2, 1, 1,
                 1, 2, 4, 2, 1,
                 2, 4, 8, 4, 2,
                 1, 2, 4, 2, 1,
                 1, 1, 2, 1, 1
                };

__constant__ int gaussMask[25];
__constant__ int gaussSum;

__device__ int getGauss(unsigned char* image, int channel, int id , int size) {
    int value = 0;
    int outOfBounds = 0;

    //a -12 to 12 because mask is 25
    for (int a = -12; a <= 12; a++) {
    	// *3 because there are 3 channels
    	int current = id + a*3;
    	if (current < 0 | current >= size * 3) {
    		outOfBounds += gaussMask[12 + a];
    	} else {
    		//TODO this doesnt work
    		//but returning value = image[current + channel] seems to get correct pixels. what's wrong??????
    		value += image[current + channel]*gaussMask[12 + a];
    	}
    }
    return value / (gaussSum - outOfBounds);
}

__global__ void filter(unsigned char* inputImage, unsigned char* outputImage, int size) {
	 int tid = (threadIdx.x + blockIdx.x * blockDim.x)*3;

	 outputImage[tid] = getGauss(inputImage, 0, tid, size);
	 outputImage[tid + 1] = getGauss(inputImage, 1, tid, size);
	 outputImage[tid + 2] = getGauss(inputImage, 2, tid, size);
}

int main(int argc, char **argv) {
    if (argc != 3 ) {
        std::cout << "Invalid number of arguments!" << std::endl;
        return EXIT_FAILURE;
    }
    Mat inputImage = imread(argv[1], IMREAD_COLOR);
    if (inputImage.empty() ) {
        std::cout << "File doesn't exist: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    Mat outputImage = inputImage.clone();

    //GAUSS ARRAY SUM
    int *arraySum;
        for(int count = 0; count < 25; count++) {
            arraySum += GAUSS[count];
        }

    // GAUSS MASK SIZE
    HANDLE_ERROR(cudaMemcpyToSymbol(gaussSum, &arraySum, sizeof(int)));

    //GAUSS MASK
    HANDLE_ERROR(cudaMemcpyToSymbol(gaussMask, GAUSS, sizeof(int) * 25));

    //INPUT AND OUTPUT IMAGES
    //unsigned char from Mat is a 1D array with each pixel description. i.e. b1,g1,r1,b2,g2,r2,…
    unsigned char *devInputImage, *devOutputImage;
    int imageMemSize = inputImage.rows*inputImage.step;
    HANDLE_ERROR(cudaMalloc<unsigned char>(&devInputImage, imageMemSize));
    HANDLE_ERROR(cudaMalloc<unsigned char>(&devOutputImage, imageMemSize));
    HANDLE_ERROR(cudaMemcpy(devInputImage, inputImage.ptr(), imageMemSize, cudaMemcpyHostToDevice));


    //GRIDS AND THREADS SIZES
	int blockSize = 512;
	int grid = (inputImage.rows*inputImage.cols + blockSize - 1) / blockSize;

	//MAIN CALL TO CUDA
	filter <<<grid, blockSize >>>(devInputImage, devOutputImage, inputImage.rows * inputImage.cols);

	//COPY BACK THE IMAGE
	HANDLE_ERROR(cudaMemcpy(outputImage.ptr(), devOutputImage, imageMemSize, cudaMemcpyDeviceToHost));
	//TODO measure time


  
    imwrite(argv[2], outputImage);
    std::cout << "Finished!" << std::endl;
    return EXIT_SUCCESS;
}
