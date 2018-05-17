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

//TODO add arguments: input, output, arraySum
__global__ void filter() {
	//outputImage.at<cv::Vec3b>(i, j)[0] = getGauss(inputImage, 0, i, j, arraySum);
	//outputImage.at<cv::Vec3b>(i, j)[1] = getGauss(inputImage, 1, i, j, arraySum);
	//outputImage.at<cv::Vec3b>(i, j)[2] = getGauss(inputImage, 2, i, j, arraySum);
	//TODO

}

int GAUSS[25] = {1, 1, 2, 1, 1,
                 1, 2, 4, 2, 1,
                 2, 4, 8, 4, 2,
                 1, 2, 4, 2, 1,
                 1, 1, 2, 1, 1
                };

//TODO this will run on each thread in each block
int getGauss(Mat image, int channel, int i, int j, int arraySum) {
    int value = 0;
    int outOfBounds = 0;
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <=2; y++) {
            if (i + x < 0 || i + x >= image.rows || j + y < 0 || j + y > image.cols ) {
                outOfBounds += GAUSS[5 * (2 + x) + 2 + y];
            } else {
                value += image.at<Vec3b>(i + x, j + y)[channel]*GAUSS[5 * (2 + x) + 2 + y];
            }
        }
    }
    return value / (arraySum - outOfBounds);
}

int main(int argc, char **argv) {
    if (argc != 2 ) {
        std::cout << "Invalid number of arguments!" << std::endl;
        return EXIT_FAILURE;
    }
    Mat inputImage = imread(argv[0], IMREAD_COLOR);
    if (inputImage.empty() ) {
        std::cout << "File doesn't exist: " << argv[0] << std::endl;
        return EXIT_FAILURE;
    }
    int numberOfThreads = atoi(argv[1]);
    Mat outputImage = inputImage.clone();

    // get gauss weights sum
    int arraySum = 0;
    for(int count = 0; count < 25; count++) {
        arraySum += GAUSS[count];
    }

	//TODO
	// assing memory in CUDA for the input image, output image, arraySum
	// call the CUDA method

	int blockSide = 128;
	int gridX = (inputImage.rows + blockSide - 1) / blockSide;
	int gridY = (inputImage.cols + blockSide - 1) / blockSide;

	dim3 grids(gridX, gridY);
	dim3 threads(blockSide, blockSide);
	filter << <grids, threads >> >();
	// it should call _device_ method for each thread in each block
	// copy content of output image back to C
	//measure time


  
    imwrite(argv[1], outputImage);
    return EXIT_SUCCESS;
}
