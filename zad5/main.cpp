#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <vector>

using namespace std;
using namespace MPI;

const long STOP_SIGNAL = -1;

struct myNumber {
    long value;
    bool prime;
};

bool isPrime(long number);
void printResults(vector<myNumber> numbers);
void update(vector<myNumber> &numbers, myNumber newNumber);
vector<myNumber> readNumbers(ifstream &inputFile);

int main(int argc, char **argv) {
    Init(argc, argv);
    if (argc != 2 ) {
        cout << "Invalid number of arguments!" << endl;
        Finalize();
        return EXIT_FAILURE;
    }

    ifstream inputFile(argv[1]);
    if (!inputFile.is_open()) {
        cout << "File doesn't exist: " << argv[1] << endl;
        Finalize();
        return EXIT_FAILURE;
    }

    uint numberOfThreads = COMM_WORLD.Get_size();
    int worldRank = COMM_WORLD.Get_rank();

    vector <myNumber> numbers;
    myNumber currentNumber;
    uint i;

    COMM_WORLD.Barrier();
    double startTime = Wtime();

    if (worldRank == 0) { //MASTER
        numbers = readNumbers(inputFile);
        inputFile.close();

        if (numberOfThreads == 1) {
            for (i=0; i < numbers.size(); ++i) {
                numbers[i].prime = isPrime(numbers[i].value);
            }
        } else {
            myNumber numberToTest;
            Status status;
            uint counter = 0;

            for (uint i = 1; i < numberOfThreads; ++i) {
                if ( (i-1) < numbers.size()) {
                    numberToTest = numbers[counter++];
                } else {
                    numberToTest.value = STOP_SIGNAL;
                }
                COMM_WORLD.Send(&numberToTest, sizeof(struct myNumber), CHAR, i, 0);
            }

            for (uint i = 0; i < numbers.size(); ++i) {
                COMM_WORLD.Recv(&currentNumber, sizeof(struct myNumber), CHAR, ANY_SOURCE, 0, status);
                update(numbers, currentNumber);
                if (counter < numbers.size()) {
                    numberToTest = numbers[counter++];
                } else {
                    numberToTest.value = STOP_SIGNAL;
                }
                COMM_WORLD.Send(&numberToTest, sizeof(struct myNumber), CHAR, status.Get_source(), 0);
            }
        }
    } else { //SLAVE
        while(true) {
            COMM_WORLD.Recv(&currentNumber, sizeof(struct myNumber), CHAR, 0, 0);
            if (currentNumber.value == STOP_SIGNAL) {
                break;
            }
            currentNumber.prime = isPrime(currentNumber.value);
            COMM_WORLD.Send(&currentNumber, sizeof(struct myNumber), CHAR, 0, 0);
        }
    }

    COMM_WORLD.Barrier();
    double endTime = Wtime();
    if (worldRank == 0) {
        cout << "Time: " << (endTime - startTime) * 1000 << "ms" << endl;
        printResults(numbers);
    }

    Finalize();
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

void update(vector<myNumber> &numbers, myNumber newNumber) {
    for (uint i = 0; i < numbers.size(); ++i) {
        if (numbers[i].value == newNumber.value) {
            numbers[i].prime = newNumber.prime;
            return;
        }
    }
}
