#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

// Tests primality of given number
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


__global__ void kernel(long *numbers, int size, bool *results) {
    int tid = blockIdx.x;
    if (tid < size ){
        results[tid] = isPrime(numbers[tid]);
    }
}

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

    vector <long> numbers;
    long number;
    while(inputFile >> number) {
        numbers.push_back(number);
    }
    inputFile.close();

    long *dev_numbers;
    bool *dev_results;

    auto startTime = chrono::system_clock::now();
    // allocating memory on GPU
    cudaMalloc( (void**) &dev_numbers, numbers.size() * sizeof(long) );
    cudaMalloc( (void**) &dev_results, numbers.size() * sizeof(bool) );

    // copying data to GPU
    cudaMemcpy( dev_numbers, numbers.data(),  numbers.size() * sizeof(long), cudaMemcpyHostToDevice );

    // doing calculation on GPU
    kernel<<< numbers.size(), 1>>>(dev_numbers, numbers.size(), dev_results);

    // copying results from GPU
    bool *results = new bool [numbers.size()];
    cudaMemcpy(results, dev_results, numbers.size() * sizeof(bool), cudaMemcpyDeviceToHost );

    // freeing memory from GPU
    cudaFree(dev_numbers);
    cudaFree(dev_results);

    auto endTime = chrono::system_clock::now();
    chrono::duration<double> elapsedMiliseconds = (endTime - startTime) * 1000;
    cout << "Time: " << elapsedMiliseconds.count() << "ms" << endl;

    for(uint i = 0; i < numbers.size(); ++i) {
        cout<< numbers[i] << (results[i] ? ": prime" : ": composite" ) << endl;
    }

    delete [] results;
    return EXIT_SUCCESS;
}
