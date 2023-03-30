#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define DEFAULT 3

int numprocs = 0;
int rank = 0;
int gridSize;
int gridCoords[2];
MPI_Comm gridComm;
MPI_Comm colComm;
MPI_Comm rowComm;

double *create_matrix (int N, int M){
    double *A = (double*) malloc(N * M * sizeof(double));
    for (int i = 0; i < N * M; ++i) {
        A[i] = 1.0;
    }
    return A;
}

double *create_emptyMatrix (int N, int M){
    double *A = (double*) malloc(N * M * sizeof(double));
    for (int i = 0; i < N * M; ++i) {
        A[i] = 0.0;
    }
    return A;
}

void print_matrix(double *Matrix, int N, int M) {
    for (int i = 0; i < N * M; ++i) {
        printf("%f ", Matrix[i]);
        if (i != 0 && (i + 1) % N == 0) {
            printf("\n");
        }
    }
}

void matrix_multiplication(double *A, double *B, double *Res, int N1, int N2, int N3) {
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N3; j++) {
            int sum = 0;
            for (int k = 0; k < N2; k++) {
                sum += A[i * N2 + k] * B[k * N3 + j];
            }
            Res[i * N3 + j] = sum;
        }
    }
}

void FreeProcess(double* A, double* B, double* Res) {
    free(A);
    free(B);
    free(Res);
}

void create_gridComm() {
    int dims[2] = {gridSize, gridSize}, periods[2] = {1, 1}, subdims[2] = {0, 1};

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &gridComm);

    MPI_Cart_coords(gridComm, rank, 2, gridCoords);

    MPI_Cart_sub(gridComm, subdims, &rowComm);

    subdims[0] = 1;
    subdims[1] = 0;
    MPI_Cart_sub(gridComm, subdims, &colComm);
}

int main(int argc, char *argv[]) {
    double *MatrixA = NULL, *MatrixB = NULL, *MatrixRes = NULL;
    double start_time = 0.0, end_time;
    int N1 = DEFAULT, N2 = DEFAULT, N3 = DEFAULT;
    if (argc >= 2){
        N1 = atoi(argv[1]);
        N2 = atoi(argv[2]);
        N3 = atoi(argv[3]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    int gridSize = (int)sqrt((double)numprocs);
    if (numprocs != gridSize * gridSize) {
        if (rank == 0) {
            printf ("Number of processes must be a perfect square \n");
        }
    }
    else {
        if (rank == 0) {
            MatrixA = create_matrix(N1, N2);
            MatrixB = create_matrix(N2, N3);
            MatrixRes = create_emptyMatrix(N1, N3);

            create_gridComm();

            matrix_multiplication(MatrixA, MatrixB, MatrixRes, N1, N2, N3);
            print_matrix(MatrixRes, N1, N3);
        }
    }
    
    MPI_Finalize();
    FreeProcess(MatrixA, MatrixB, MatrixRes);
    return 0;
}