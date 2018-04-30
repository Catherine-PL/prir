#include <iostream>
#include <chrono>
#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace MPI;

int GAUSS[25] = {1, 1, 2, 1, 1,
                 1, 2, 4, 2, 1,
                 2, 4, 8, 4, 2,
                 1, 2, 4, 2, 1,
                 1, 1, 2, 1, 1
                };

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
	MPI_Init(NULL, NULL);

    if (argc != 3 ) {
        std::cout << "Invalid number of arguments!" << std::endl;
        return EXIT_FAILURE;
    }
  

    int numberOfThreads, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfThreads);

    // get gauss weights sum
    int arraySum = 0;
    for(int count = 0; count < 25; count++) {
        arraySum += GAUSS[count];
    }
    //if only 1 thread then just do it in here
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = MPI_Wtime();
    if (numberOfThreads == 1) {
    	Mat inputImage = imread(argv[1], IMREAD_COLOR);
    	Mat outputImage = inputImage.clone();
    	int rows = inputImage.rows;
    	int cols = inputImage.cols;
    	for (int i=0; i < rows; i++) {
    	    for (int j = 0; j < cols; j++) {
    	    	outputImage.at<cv::Vec3b>(i,j)[0] = getGauss(inputImage, 0, i, j, arraySum);
    	    	outputImage.at<cv::Vec3b>(i,j)[1] = getGauss(inputImage, 1, i, j, arraySum);
    	    	outputImage.at<cv::Vec3b>(i,j)[2] = getGauss(inputImage, 2, i, j, arraySum);
    	    	}
    	}
    	imwrite(argv[2], outputImage);
    } else if (rank == 0) {
    	 Mat inputImage = imread(argv[1], IMREAD_COLOR);
    	 int rows = inputImage.rows;
    	 int cols = inputImage.cols;
    	 
    	    if (inputImage.empty() ) {
    	        std::cout << "File doesn't exist: " << argv[1] << std::endl;
    	        return EXIT_FAILURE;
    	    }
    	 Mat outputImage;
    	//cut image and send
    	 int top, bottom, blockRows;
    	 int *blockSize = new int[3];
    	 blockRows = rows/(numberOfThreads - 1);
    	 for (int i = 1; i < numberOfThreads; i++) {
    		 top = (i-1) * blockRows;
    		 if (i == (numberOfThreads - 1)) {
    			 bottom = inputImage.rows;
    		 }else {
    			 bottom = i * blockRows;
    		 }
    		 Mat block = Mat::zeros(cv::Size(cols, bottom-top), CV_8UC3);
    		 Rect r(0, top, cols, bottom-top);
    		 inputImage(r).copyTo(block);

    		 blockSize[0] = block.rows;
    		 blockSize[1] = block.cols;
    		 blockSize[2] = block.channels();
    		 COMM_WORLD.Send(blockSize, 3, MPI_INT, i, 0);
    		 COMM_WORLD.Send(block.data,block.rows * block.cols * block.channels(), MPI_BYTE, i, 1);
    		 
    		 // receive parts of images
    		 Mat outputBlock = block.clone();
    		 COMM_WORLD.Recv(outputBlock.data, outputBlock.cols * outputBlock.rows * outputBlock.channels(), MPI_BYTE, i, 3);
    		 outputImage.push_back(outputBlock);
    		 imwrite(argv[2], outputImage);
    	 }
    } else {
    	int *blockSize = new int[3];
    	COMM_WORLD.Recv(blockSize, 3, MPI_INT, 0, 0);
    	Mat block = Mat(blockSize[0], blockSize[1], CV_8UC3);
    	COMM_WORLD.Recv(block.data, blockSize[0] * blockSize[1] * 3, MPI_BYTE, 0,1);
    	
    	// blur
    	Mat outputBlock = block.clone();
    	
    	for (int i=0; i < blockSize[0]; i++) {
    	    for (int j = 0; j < blockSize[1]; j++) {
    	    	outputBlock.at<cv::Vec3b>(i,j)[0] = getGauss(block, 0, i, j, arraySum);
    	    	outputBlock.at<cv::Vec3b>(i,j)[1] = getGauss(block, 1, i, j, arraySum);
    	    	outputBlock.at<cv::Vec3b>(i,j)[2] = getGauss(block, 2, i, j, arraySum);
    	    }
    	}
    	
    	// send
		COMM_WORLD.Send(outputBlock.data, outputBlock.cols * outputBlock.rows * outputBlock.channels(), MPI_BYTE, 0, 3);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = MPI_Wtime();
    if (rank == 0) {
    	std::cout << "Time: " << (end - start) * 1000 << "ms" << std::endl;
    }
    MPI_Finalize();
    
    return EXIT_SUCCESS;
}
