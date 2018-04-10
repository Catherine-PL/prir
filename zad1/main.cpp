#include <omp.h>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
    if (argc < 4 ){
        std::cout << "Invalid number of arguments!" << std::endl;
        return EXIT_FAILURE;
    }

    int numberOfThreads = atoi(argv[1]);
    cv::Mat inputImage = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (inputImage.empty() ){
        std::cout << "File doesn't exist: " << argv[2] << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat outputImage = inputImage.clone();

    auto start = std::chrono::system_clock::now();

    //TODO: napisac wlasna funkcje rozmycia gaussa, bo w obecnej formie daje odwrotny rezultat - nie zrownolegla się
    #pragma omp parallel num_threads(numberOfThreads)
    {
        cv::GaussianBlur(inputImage, outputImage, cv::Size(5, 5), 0, 0);
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedMiliseconds = (end - start) * 1000;
    std::cout << "Elapsed time: " << elapsedMiliseconds.count()<< "ms" << std::endl;

    cv::imwrite(argv[3], outputImage);
    return EXIT_SUCCESS;
}
