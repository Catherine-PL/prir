#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

struct myNumber{
    long value;
    bool prime;
};

bool isPrime(long number);
void printResults(std::vector<myNumber> numbers);
std::vector<myNumber> readFile(std::ifstream &inputFile);

int main(int argc, char **argv) {
    if (argc < 3 ) {
        std::cout << "Invalid number of arguments!" << std::endl;
        return EXIT_FAILURE;
    }

        std::ifstream inputFile(argv[2]);
        if (!inputFile.is_open()){
            std::cout << "File doesn't exist: " << argv[2] << std::endl;
            return EXIT_FAILURE;
        }

            std::vector <myNumber> numbers = readFile(inputFile);

            inputFile.close();

            int numberOfThreds = atoi(argv[1]);

            auto startTime = std::chrono::system_clock::now();

            uint i;

            #pragma omp parallel for \
                num_threads(numberOfThreds) \
                default(none) shared(numbers) private(i) \
                schedule(dynamic, 1)
            for (i=0; i < numbers.size(); ++i){
                numbers[i].prime = isPrime(numbers[i].value);
            }

            auto endTime = std::chrono::system_clock::now();

            std::chrono::duration<double> elapsedMiliseconds = (endTime - startTime) * 1000;

            std::cout << "Time: " << elapsedMiliseconds.count()<< "ms" << std::endl;

            printResults(numbers);

            return EXIT_SUCCESS;
}

std::vector<myNumber> readFile(std::ifstream &inputFile){
    long number;
    std::vector <myNumber> numbers;

    while(inputFile >> number){
        myNumber n;
        n.value = number;
        n.prime = false;
        numbers.push_back(n);
    }

        return numbers;
}

/**
 * tests primality (naive approach  with some optimization)
 *
 * @param number number to be tested
 * @return true if the number is pirme, otherwise false
 */
bool isPrime(long number){
    if (number == 2 || number == 3){
        return true;
    } else if (number < 2 || number % 2 == 0 || number % 3 == 0){
        return false;
    }

        long step = 4;
        long max = sqrt(number);
        for (int i = 5; i <= max; i += step){
            if (number % i == 0){
                return false;
            }
                    step = 6 - step; //HACK: if (step == 2) {step = 4;} else {step = 2;}
        }

            return true;
}

void printResults(std::vector<myNumber> numbers){
    for(uint i = 0; i < numbers.size(); ++i) {
        if (numbers[i].prime){
            std::cout<< numbers[i].value << ": prime" << std::endl;
        } else {
            std::cout<< numbers[i].value << ": composite" << std::endl;
        }
    }
}
