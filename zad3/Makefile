main:
	g++ -std=c++11 -c main.cpp -o main.o `pkg-config --cflags opencv --libs opencv`
	nvcc -c main.cu -o main_gpu.o
	g++ main.o main_gpu.o -o gauss_gpu