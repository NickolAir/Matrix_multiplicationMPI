#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DEFAULT 3

int main(int argc, char *argv[]) {
    double *MatrixA = NULL, *MatrixB = NULL, *MatrixRes = NULL;
    double start_time = 0.0, end_time;
    int numprocs, rank;
    int RestLines;
    int size = DEFAULT;
    if (argc >= 2){
        size = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("%d\n", size);

    MPI_Finalize();
    return 0;
}
