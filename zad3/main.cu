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

__device__ int getGauss(unsigned char* image, int channel, int id , int cols, int rows, int arraySum) {
    int value = 0;
    int outOfBounds = 0;

    for (int i = -2; i <= 2; i++) {
    	for (int j = -2; j <= 2; j++) {
    		// multiplyiing by 3 because there are 3 channels
    		int current = id + (i*cols + j)*3;
    		//adding 12 to neutrilize negative values of a mask which has size 12
    		int maskIndex = 12 + 5*i+j;
    		if (current < 0 || current >= cols*rows * 3) {
    		    	outOfBounds += gaussMask[maskIndex];
    		} else {
    		    	value += image[current + channel]*gaussMask[maskIndex];
    		}
    	}
    }
    return value / (arraySum - outOfBounds);
}

__global__ void filter(unsigned char* inputImage, unsigned char* outputImage, int cols, int rows, int arraySum) {
	 int tid = (threadIdx.x + blockIdx.x * blockDim.x)*3;
	 if (tid <  cols*rows * 3 - 3) {
		 outputImage[tid] = getGauss(inputImage, 0, tid, cols, rows, arraySum);
		 outputImage[tid + 1] = getGauss(inputImage, 1, tid, cols, rows, arraySum);
		 outputImage[tid + 2] = getGauss(inputImage, 2, tid, cols, rows, arraySum); 
	 }
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
    int arraySum = 0;
        for(int count = 0; count < 25; count++) {
            arraySum += GAUSS[count];
        }

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

	//TIME MEASUREMENT INIT
	float totalTime = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	cudaEventCreate(&stop);
	//MAIN CALL TO CUDA
	filter <<<grid, blockSize >>>(devInputImage, devOutputImage, inputImage.rows, inputImage.cols, arraySum);

	//CALCULATE ELAPSED TIME
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&totalTime, start, stop);
	//COPY BACK THE IMAGE
	HANDLE_ERROR(cudaMemcpy(outputImage.ptr(), devOutputImage, imageMemSize, cudaMemcpyDeviceToHost));

    imwrite(argv[2], outputImage);
    std::cout << "Time: " << totalTime << "ms" << std::endl;
    
    cudaFree(devInputImage);
    cudaFree(devOutputImage);
    return EXIT_SUCCESS;
}
