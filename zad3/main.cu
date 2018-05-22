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


//TODO this will run on each thread in each block
__device__ int getGauss(unsigned char* image, int channel, int i, int j , int rows, int cols, int step, int offset) {
    int value = 0;
    int outOfBounds = 0;
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <=2; y++) {
            if (i + x < 0 || i + x >= rows || j + y < 0 || j + y > cols ) {
                outOfBounds += gaussMask[5 * (2 + x) + 2 + y];
            } else {

        /*    	int b = inputImage[step * y + x ];
            		 int g = inputImage[step * y + x + 1];
            		 int r = inputImage[step * y + x + 2];*/
            	//[step*j + i + channel]
                value += image[offset + channel ]*gaussMask[5 * (2 + x) + 2 + y];
            }
        }
    }
    return value / (gaussSum - outOfBounds);

    /*for(int xChange = 0 ; xChange < MASK_SIZE ; xChange++) {
    				for(int yChange = 0 ; yChange< MASK_SIZE ; yChange++) {
    					currentPixelVal+=
    					d_mask[xChange][yChange] * inputImage[thIdx + ((yChange - 2) * size) + ((xChange - 2) * channelNo)];
    				}
    			}
    			outputImage[thIdx] = (unsigned char) (currentPixelVal/d_weight);*/
}

//TODO add arguments: input, output, arraySum
__global__ void filter(unsigned char* inputImage, unsigned char* outputImage, int rows, int cols, int step) {
	 int x = threadIdx.x + blockIdx.x * blockDim.x;
	 int y = threadIdx.y + blockIdx.y * blockDim.y;
	 //todo maybe step should be used instead of cols
	 int offset = (x + y * blockDim.x * gridDim.x)*3;



	 //TODO assign to output
	 // outputImage[step * y * blockDim.x * gridDim.x + x ]
	 outputImage[offset ] = inputImage[offset ];//getGauss(inputImage, 0, x, y, rows, cols, step, offset);
	 outputImage[offset + 1] = inputImage[offset + 1 ];//getGauss(inputImage, 1, x, y, rows, cols, step, offset);
	 outputImage[offset + 2] = inputImage[offset + 2];//getGauss(inputImage, 2, x, y, rows, cols, step, offset);

	//outputImage.at<cv::Vec3b>(i, j)[0] = getGauss(inputImage, 0, i, j, arraySum);
	//outputImage.at<cv::Vec3b>(i, j)[1] = getGauss(inputImage, 1, i, j, arraySum);
	//outputImage.at<cv::Vec3b>(i, j)[2] = getGauss(inputImage, 2, i, j, arraySum);
	//TODO

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
    //unsigned char from Mat: 1D array. i.e. b1,g1,r1,b2,g2,r2,…
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

	//IMAGE SIZES

	dim3 grids(gridX, gridY);
	dim3 threads(blockSide, blockSide);

	//MAIN CALL TO CUDA
	filter <<<grids, threads >>>(devInputImage, devOutputImage, inputImage.rows, inputImage.cols, inputImage.step);
	// it should call _device_ method for each thread in each block
	// copy content of output image back to Cpp
	HANDLE_ERROR(cudaMemcpy(outputImage.ptr(), devOutputImage, imageMemSize, cudaMemcpyDeviceToHost));
	//measure time


  
    imwrite(argv[2], outputImage);
    std::cout << "Finished!" << std::endl;
    return EXIT_SUCCESS;
}
