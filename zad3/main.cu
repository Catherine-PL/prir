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

//TODO add arguments: input, output, arraySum
__global__ void filter(unsigned char* inputImage, unsigned char* outputImage) {
	//TODO i and j
	//outputImage.at<cv::Vec3b>(i, j)[0] = getGauss(inputImage, 0, i, j, arraySum);
	//outputImage.at<cv::Vec3b>(i, j)[1] = getGauss(inputImage, 1, i, j, arraySum);
	//outputImage.at<cv::Vec3b>(i, j)[2] = getGauss(inputImage, 2, i, j, arraySum);
	//TODO

}

//TODO this will run on each thread in each block
/*__device__ int getGauss(Mat image, int channel, int i, int j, int arraySum) {
    int value = 0;
    int outOfBounds = 0;
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <=2; y++) {
            if (i + x < 0 || i + x >= image.rows || j + y < 0 || j + y > image.cols ) {
                outOfBounds += gaussMask[5 * (2 + x) + 2 + y];
            } else {
                value += image.at<Vec3b>(i + x, j + y)[channel]*gaussMask[5 * (2 + x) + 2 + y];
            }
        }
    }
    return value / (arraySum - outOfBounds);
}*/

int main(int argc, char **argv) {
    if (argc != 3 ) {
        std::cout << "Invalid number of arguments!" << argc << std::endl;
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
/*
 * commented because I use constant gaussSum instead
    int *devArraySum = NULL;
    HANDLE_ERROR(cudaMalloc((void**) &devArraySum, sizeof(unsigned int)));
    HANDLE_ERROR(cudaMemcpy(devArraySum, arraySum, sizeof(unsigned int), cudaMemcpyHostToDevice));
*/

    // GAUSS MASK SIZE
    HANDLE_ERROR(cudaMemcpyToSymbol(gaussSum, &arraySum, sizeof(int)));

    //GAUSS MASK
    HANDLE_ERROR(cudaMemcpyToSymbol(gaussMask, GAUSS, sizeof(int) * 25));

    //INPUT AND OUTPUT IMAGES
    unsigned char *devInputImage, *devOutputImage;
    int imageMemSize = inputImage.rows*inputImage.step;
    HANDLE_ERROR(cudaMalloc<unsigned char>(&devInputImage, imageMemSize));
    HANDLE_ERROR(cudaMalloc<unsigned char>(&devOutputImage, imageMemSize));

    //step – Number of bytes each matrix row occupies. The value includes the padding bytes at the end of each row, if any
    HANDLE_ERROR(cudaMemcpy(devInputImage, inputImage.ptr(), imageMemSize, cudaMemcpyHostToDevice));


    //GRIDS AND THREADS SIZES
	int blockSide = 128;
	int gridX = (inputImage.rows + blockSide - 1) / blockSide;
	int gridY = (inputImage.cols + blockSide - 1) / blockSide;

	dim3 grids(gridX, gridY);
	dim3 threads(blockSide, blockSide);

	//MAIN CALL TO CUDA
	filter <<<grids, threads >>>(devInputImage, devOutputImage);
	// it should call _device_ method for each thread in each block
	// copy content of output image back to C
	//measure time


  
    imwrite(argv[2], outputImage);
    return EXIT_SUCCESS;
}
