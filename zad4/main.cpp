#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

using namespace std;

struct myNumber {
    long value;
    bool prime;
};

bool isPrime(long number);
void printResults(vector<myNumber> numbers);
vector<myNumber> readNumbers(ifstream &inputFile);

int main(int argc, char **argv) {
    if (argc != 3 ) {
        cout << "Invalid number of arguments!" << endl;
        return EXIT_FAILURE;
    }

    ifstream inputFile(argv[2]);
    if (!inputFile.is_open()) {
        cout << "File doesn't exist: " << argv[2] << endl;
        return EXIT_FAILURE;
    }

    vector <myNumber> numbers = readNumbers(inputFile);
    inputFile.close();

    uint i;
    int numberOfThreds = atoi(argv[1]);
    auto startTime = chrono::system_clock::now();

    #pragma omp parallel for num_threads(numberOfThreds) \
        shared(numbers) private(i) schedule(dynamic, 1)
    for (i=0; i < numbers.size(); ++i) {
        numbers[i].prime = isPrime(numbers[i].value);
    }

    auto endTime = chrono::system_clock::now();
    chrono::duration<double> elapsedMiliseconds = (endTime - startTime) * 1000;

    cout << "Time: " << elapsedMiliseconds.count()<< "ms" << endl;
    printResults(numbers);

    return EXIT_SUCCESS;
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
bool isPrime(long number) {
    if (number == 2 || number == 3) {
        return true;
    } else if (number < 2 || number % 2 == 0 || number % 3 == 0) {
        return false;
    }

    int step = 4;
    long max = sqrt(number);
    for (int i = 5; i <= max; i += step) {
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
