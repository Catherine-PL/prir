EXECS=main
MPICXX?=mpicxx

all: ${EXECS}

main: main.cpp
	${MPICXX} -o gauss_mpi main.cpp -std=c++11 `pkg-config --cflags opencv --libs opencv`

clean:
	rm ${EXECS}
