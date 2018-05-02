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
void manageProcesses(uint numberOfThreads, vector<myNumber> &numbers);
void printResults(vector<myNumber> numbers);
void processData();
vector<myNumber> readNumbers(ifstream &inputFile);
void update(vector<myNumber> &numbers, myNumber newNumber);

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
    int rank = COMM_WORLD.Get_rank();

    vector <myNumber> numbers;
    double startTime;

    COMM_WORLD.Barrier();

    if (rank == 0) {
        numbers = readNumbers(inputFile);
        inputFile.close();
        startTime = Wtime();

        if (numberOfThreads == 1) {
            for (uint i=0; i < numbers.size(); ++i) {
                numbers[i].prime = isPrime(numbers[i].value);
            }
        } else {
            manageProcesses(numberOfThreads, numbers);
        }
    } else { //rank != 0
        processData();
    }

    COMM_WORLD.Barrier();
    if (rank == 0) {
        double endTime = Wtime();
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


/*
 * Updates the list of numbers.
 *
 * @param numbers list of numbers
 * @param newNumber number to update
 */
void update(vector<myNumber> &numbers, myNumber newNumber) {
    for (uint i = 0; i < numbers.size(); ++i) {
        if (numbers[i].value == newNumber.value) {
            numbers[i].prime = newNumber.prime;
            return;
        }
    }
}

/**
 * Manages processes - sends and receives data.
 *
 * @param numberOfThreads number of threads
 * @param numbers list of numbers
 */
void manageProcesses(uint numberOfThreads, vector<myNumber> &numbers) {
    myNumber numberToTest, receivedNumber;
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
        COMM_WORLD.Recv(&receivedNumber, sizeof(struct myNumber), CHAR, ANY_SOURCE, 0, status);
        update(numbers, receivedNumber);
        if (counter < numbers.size()) {
            numberToTest = numbers[counter++];
        } else {
            numberToTest.value = STOP_SIGNAL;
        }
        COMM_WORLD.Send(&numberToTest, sizeof(struct myNumber), CHAR, status.Get_source(), 0);
    }
}

/**
 * Receives number, checks primality and sends the result back.
 */
void processData() {
    myNumber receivedNumber;
    while(true) {
        COMM_WORLD.Recv(&receivedNumber, sizeof(struct myNumber), CHAR, 0, 0);
        if (receivedNumber.value == STOP_SIGNAL) {
            break;
        }
        receivedNumber.prime = isPrime(receivedNumber.value);
        COMM_WORLD.Send(&receivedNumber, sizeof(struct myNumber), CHAR, 0, 0);
    }
}
