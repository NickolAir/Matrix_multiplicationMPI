#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DEFAULT 3

double *create_matrix (int N){
    double *A = (double*) malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0;
    }
    return A;
}

double *create_emptyMatrix (int N){
    double *A = (double*) malloc(N * N * sizeof(double));
    for (int i = 0; i < N * N; ++i) {
        A[i] = 0.0;
    }
    return A;
}

void print_matrix(double *Matrix, int N) {
    for (int i = 0; i < N * N; ++i) {
        printf("%f ", Matrix[i]);
        if (i != 0 && (i + 1) % N == 0) {
            printf("\n");
        }
    }
}

void matrix_multiplication(double *A, double *B, double *Res, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                Res[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

void FreeProcess(double* A, double* B, double* Res) {
    free(A);
    free(B);
    free(Res);
}

int main(int argc, char *argv[]) {
    double *MatrixA = NULL, *MatrixB = NULL, *MatrixRes = NULL;
    double start_time = 0.0, end_time;
    int numprocs, rank;
    int size = DEFAULT;
    if (argc >= 2){
        size = atoi(argv[1]);
    }
    MatrixA = create_matrix(size);
    MatrixB = create_matrix(size);
    MatrixRes = create_emptyMatrix(size);
    matrix_multiplication(MatrixA, MatrixB, MatrixRes, size);
    print_matrix(MatrixRes, size);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    
    MPI_Finalize();
    FreeProcess(MatrixA, MatrixB, MatrixRes);
    return 0;
}