#include <iostream>
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
    Init(argc, argv);

    if (argc != 3 ) {
        std::cout << "Invalid number of arguments!" << std::endl;
        Finalize();
        return EXIT_FAILURE;
    }

    int rank = COMM_WORLD.Get_rank();
    int numberOfProcesses = COMM_WORLD.Get_size();

    // get gauss weights sum
    int arraySum = 0;
    for(int count = 0; count < 25; count++) {
        arraySum += GAUSS[count];
    }
    Mat inputImage, outputImage;
    double start;
    COMM_WORLD.Barrier();

    if (rank == 0) {
        inputImage = imread(argv[1], IMREAD_COLOR);
        if (inputImage.empty() ) {
            std::cout << "File doesn't exist: " << argv[1] << std::endl;
            Finalize();
            return EXIT_FAILURE;
        }
        int rows = inputImage.rows;
        int cols = inputImage.cols;

        //if only 1 process then just do it in here
        if (numberOfProcesses == 1) {
            outputImage = inputImage.clone();
            start = Wtime();
            for (int i=0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    outputImage.at<cv::Vec3b>(i,j)[0] = getGauss(inputImage, 0, i, j, arraySum);
                    outputImage.at<cv::Vec3b>(i,j)[1] = getGauss(inputImage, 1, i, j, arraySum);
                    outputImage.at<cv::Vec3b>(i,j)[2] = getGauss(inputImage, 2, i, j, arraySum);
                }
            }
        } else { // numberOfProcesses > 1
            //cut image and send
            int top, bottom, blockRows;
            int *blockSize = new int[3];
            blockRows = rows/(numberOfProcesses - 1);
            start = Wtime();
            for (int i = 1; i < numberOfProcesses; i++) {
                top = (i-1) * blockRows -2;
                if (top < 0) {
                    top = 0;
                }

                if (i == (numberOfProcesses - 1)) {
                    bottom = inputImage.rows;
                } else {
                    bottom = i * blockRows + 2;
                    if (bottom > inputImage.rows) {
                        bottom = inputImage.rows;
                    }
                }
                Mat block = Mat::zeros(cv::Size(cols, bottom-top), CV_8UC3);
                Rect r(0, top, cols, bottom-top);
                inputImage(r).copyTo(block);

                blockSize[0] = block.rows;
                blockSize[1] = block.cols;
                blockSize[2] = block.channels();
                COMM_WORLD.Send(blockSize, 3, INT, i, 0);
                COMM_WORLD.Send(block.data,block.rows * block.cols * block.channels(), BYTE, i, 1);
            }
            for (int i = 1; i < numberOfProcesses; i++) {
                // receive parts of images
                COMM_WORLD.Recv(blockSize, 3, INT, i, 3);
                Mat outputBlock = Mat::zeros(cv::Size(blockSize[1], blockSize[0]), CV_8UC3);
                COMM_WORLD.Recv(outputBlock.data, outputBlock.cols * outputBlock.rows * outputBlock.channels(), BYTE, i, 4);
                if (i != 1) { // cut top 2 rows
                    outputBlock = outputBlock.rowRange(2, outputBlock.rows).clone();
                }
                if (i != numberOfProcesses - 1) { // cut bottom 2 rows
                    outputBlock = outputBlock.rowRange(0, outputBlock.rows-2).clone();
                }
                outputImage.push_back(outputBlock);
            }
            delete [] blockSize;
        }
    } else { // rank != 0
        int *blockSize = new int[3];
        COMM_WORLD.Recv(blockSize, 3, INT, 0, 0);
        Mat block = Mat(blockSize[0], blockSize[1], CV_8UC3);
        COMM_WORLD.Recv(block.data, blockSize[0] * blockSize[1] * 3, BYTE, 0,1);

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
        COMM_WORLD.Send(blockSize, 3, INT, 0, 3);
        COMM_WORLD.Send(outputBlock.data, outputBlock.cols * outputBlock.rows * outputBlock.channels(), BYTE, 0, 4);
        delete [] blockSize;
    }

    COMM_WORLD.Barrier();
    double end = Wtime();
    if (rank == 0) {
        std::cout << "Time: " << (end - start) * 1000 << "ms" << std::endl;
        imwrite(argv[2], outputImage);
    }
    Finalize();

    return EXIT_SUCCESS;
}
