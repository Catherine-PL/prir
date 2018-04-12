#include <iostream>
#include <chrono>
#include <omp.h>
#include <opencv2/opencv.hpp>
using namespace cv;

int GAUSS[25] = {1, 1, 2, 1, 1,
				  1, 2, 4, 2, 1,
				  2, 4, 8, 4, 2,
				  1, 2, 4, 2, 1,
				  1, 1, 2, 1, 1};

int getGauss(Mat image, int channel, int i, int j, int arraySum) {
	int value = 0;
	for (int x = -2; x <= 2; x++) {
	   for (int y = -2; y <=2; y++) {
		   	  value+=image.at<Vec3b>(i+x,j+y)[channel]*GAUSS[5*(2+x) + 2 + y];
	   }
	}
	return value/arraySum;
}

int main(int argc, char **argv) {
    if (argc != 4 ){
        std::cout << "Invalid number of arguments!" << std::endl;
        return EXIT_FAILURE;
    }
    Mat inputImage = imread(argv[2], IMREAD_COLOR);
       if (inputImage.empty() ){
           std::cout << "File doesn't exist: " << argv[2] << std::endl;
           return EXIT_FAILURE;
       }
    int numberOfThreads = atoi(argv[1]);
    Mat outputImage = inputImage.clone();
    int startrow, startcol;
        startrow = 2;//middle pixel of filter mask
        startcol = 2;
    Size s = inputImage.size();
    int rows = s.height;
    int cols = s.width;

    // get gauss weights sum
    int arraySum=0;
    for(int count=0;count<25;count++){
        	arraySum+=GAUSS[count];
    }
    auto start = std::chrono::system_clock::now();
    #pragma omp parallel num_threads(numberOfThreads)
    {
    	for (int i = startrow; i <= rows - startrow -1; i++) {
    	        for (int j = startcol; j <= cols - startcol -1; j++) {
    	        	outputImage.at<cv::Vec3b>(i,j)[0] = getGauss(inputImage, 0, i, j, arraySum);
    	        	outputImage.at<cv::Vec3b>(i,j)[1] = getGauss(inputImage, 1, i, j, arraySum);
    	        	outputImage.at<cv::Vec3b>(i,j)[2] = getGauss(inputImage, 2, i, j, arraySum);
    	        }
    	    }
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedMiliseconds = (end - start) * 1000;
    std::cout << "Elapsed time: " << elapsedMiliseconds.count()<< "ms" << std::endl;

    imwrite(argv[3], outputImage);
    return EXIT_SUCCESS;
}












