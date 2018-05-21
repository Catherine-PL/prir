#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

struct myNumber {
    long value;
    bool prime;
};

__global__ void kernel(myNumber *numbers, int size);
__device__ bool isPrime(long number);
void printResults(vector<myNumber> numbers);
vector<myNumber> readNumbers(ifstream &inputFile);

int main(int argc, char **argv) {
    if (argc != 2 ) {
        cout << "Invalid number of arguments!" << endl;
        return EXIT_FAILURE;
    }

    ifstream inputFile(argv[1]);
    if (!inputFile.is_open()) {
        cout << "File doesn't exist: " << argv[1] << endl;
        return EXIT_FAILURE;
    }

    vector <myNumber> numbers = readNumbers(inputFile);
    inputFile.close();

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    myNumber *dev_numbers;
    // allocating memory on GPU
    cudaMalloc( (void**) &dev_numbers, numbers.size() * sizeof(struct myNumber) );

    // copying data to GPU
    cudaMemcpy( dev_numbers, numbers.data(),  numbers.size() * sizeof(struct myNumber), cudaMemcpyHostToDevice );

    cudaEventRecord( start, 0 );
    // doing calculation on GPU
    kernel<<< numbers.size(), 1>>>(dev_numbers, numbers.size());
    cudaEventRecord( stop, 0 );

    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );

    //copying results from GPU
    cudaMemcpy(numbers.data(), dev_numbers, numbers.size() * sizeof(struct myNumber), cudaMemcpyDeviceToHost );

    // freeing memory from GPU
    cudaFree(dev_numbers);

    cout << "Time: " << elapsedTime << "ms" << endl;
    printResults(numbers);

    return EXIT_SUCCESS;
}

__global__ void kernel(myNumber *numbers, int size) {
    int tid = blockIdx.x;
    if (tid < size) {
        numbers[tid].prime = isPrime(numbers[tid].value);
    }
}

/**
 * Reads numbers from the file.
 *
 * @param  inputFile file to be read from
 * @return list of numbers
 */
vector<myNumber> readNumbers(ifstream &inputFile) {
    long number;
    vector <myNumber> numbers;

    while(inputFile >> number) {
        myNumber n;
        n.value = number;
        numbers.push_back(n);
    }
    return numbers;
}

/**
 * Tests primality (naive approach  with some optimization)
 *
 * @param number number to be tested
 * @return true if the number is pirme, otherwise false
 */
__device__ bool isPrime(long number) {
    if (number == 2 || number == 3) {
        return true;
    } else if (number < 2 || number % 2 == 0 || number % 3 == 0) {
        return false;
    }

    int step = 4;
    for (int i = 5; i*i <= number; i += step) { //NOTICE: sqrt() is not allowed
        if (number % i == 0) {
            return false;
        }
        step = 6 - step; //HACK: if (step == 2) {step = 4;} else {step = 2;}
    }
    return true;
}

/**
 * Prints results of the primality tests.
 *
 * @param numbers list of numbers
 */
void printResults(vector<myNumber> numbers) {
    for(uint i = 0; i < numbers.size(); ++i) {
        if (numbers[i].prime) {
            cout<< numbers[i].value << ": prime" << endl;
        } else {
            cout<< numbers[i].value << ": composite" << endl;
        }
    }
}
