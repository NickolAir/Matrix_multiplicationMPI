#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DEFAULT 3
#define Y 0
#define X 1

int *decompose(int n, int k) {
    if (n < k) {
        printf("Невозможно разложить %d на %d слагаемых\n", n, k);
        exit(-1);
    }
    int sum = k;
    if (k == 1) {
        int *summands = (int*) malloc(1 * sizeof(int));
        summands[0] = n;
        return summands;
    }
    int *summands = (int*) malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) {
        summands[i] = 1;
    }
    for (int i = 0; i <= (n / 2) + 1; ++i) {
        for (int j = 0; j < k; ++j) {
            if (sum < n) {
                summands[j]++;
                sum++;
            }
        }
    }
    return summands;
}

void transpose_matrix(double* matrix, int rows, int cols, double* result) {
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}

double *create_matrix (int N, int M){
    double value = 1.0;
    double *A = (double*) malloc(N * M * sizeof(double));
    for (int i = 0; i < N * M; ++i) {
        A[i] = value;
        value += 1.0;
    }
    return A;
}

double *matrixCreateEmpty (int N, int M){
    double *A = (double*) malloc(N * M * sizeof(double));
    for (int i = 0; i < N * M; ++i) {
        A[i] = 0.0;
    }
    return A;
}

void matrixPrint(double *Matrix, int N, int M) {
    for (int i = 0; i < N * M; ++i) {
        printf("%f ", Matrix[i]);
        if (i != 0 && (i + 1) % M == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

void matrix_multiplication(double *A, double *B, double *Res, int N1, int N2, int N3) {
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N3; j++) {
            int sum = 0;
            for (int k = 0; k < N2; k++) {
                sum += A[i * N2 + k] * B[j * N2 + k];
            }
            Res[i * N3 + j] = sum;
        }
    }
}

void freeProcess(double* A, double* B, double* Res, MPI_Comm *colComm, MPI_Comm *rowComm,
                 int *summandsA, int *summandsB, int rank) {
    if (rank == 0) {
        free(A);
        free(B);
        free(Res);
    }
    free(colComm);
    free(rowComm);
    free(summandsA);
    free(summandsB);
}

void create_gridComm(int *dims, int *periods, int *coords, int numprocs, MPI_Comm *gridComm, int rank) {
    MPI_Dims_create(numprocs, 2, dims);
    if (rank == 0)
        printf("Grid size %d x %d\n", dims[Y], dims[X]);
    //creating communicator of 2d grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, gridComm);
    //getting rank of process in grid
    MPI_Comm_rank(*gridComm, &rank);
    //get coordinates in 2d grid
    //MPI_Cart_get(gridComm, 2, dims, periods, coords);
    MPI_Cart_coords(*gridComm, rank, 2, coords);
    //printf("rank: %d coords: %d %d\n", rank, coords[Y], coords[X]);
}

void create_Comms(MPI_Comm *gridComm, MPI_Group *gridGroup, MPI_Comm *rowComms, MPI_Comm *colComms, int *dims) {
    MPI_Group buffer;
    int cur_coord[2] = {0, 0};
    int rank;
    int *rankArray = (int*) malloc(dims[Y] * sizeof(int));
    //communicator for column
    for (int i = 0; i < dims[X]; ++i) {
        for (int j = 0; j < dims[Y]; ++j) {
            cur_coord[X] = i;
            cur_coord[Y] = j;
            MPI_Cart_rank(*gridComm, cur_coord, &rank);
            rankArray[j] = rank;
        }
        MPI_Group_incl(*gridGroup, dims[Y], rankArray, &buffer);
        MPI_Comm_create(*gridComm, buffer, &colComms[i]);
    }
    //communicator for rows
    rankArray = (int*) realloc(rankArray, dims[X] * sizeof(int));
    for (int i = 0; i < dims[Y]; ++i) {
        for (int j = 0; j < dims[X]; ++j) {
            cur_coord[Y] = i;
            cur_coord[X] = j;
            MPI_Cart_rank(*gridComm, cur_coord, &rank);
            rankArray[j] = rank;
        }
        MPI_Group_incl(*gridGroup, dims[X], rankArray, &buffer);
        MPI_Comm_create(*gridComm, buffer, &rowComms[i]);
    }

    free(rankArray);
}

void matrix_partition(double *Matrix, int K, int *summands, double *subMatrix, int rankY, int rankX,
                      int dim, MPI_Comm *colComm) {
    if (rankX == 0) {
        int *sendNum = (int*) malloc(dim * sizeof(int));
        int *sendOffset = (int*) malloc(dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            sendNum[i] = summands[i] * K;
            if (i > 0) {
                sendOffset[i] = summands[i - 1] * K + sendOffset[i - 1];
            } else {
                sendOffset[i] = 0;
            }
        }
        MPI_Scatterv(Matrix, sendNum, sendOffset, MPI_DOUBLE, subMatrix,
                     sendNum[rankY], MPI_DOUBLE, 0, colComm[rankX]);
        free(sendNum);
        free(sendOffset);
    }
}

void data_distribution(double *subMatrixA, double *subMatrixB, int rankY, int rankX, int *summandsA,
                       int *summandsB, int *dims, int N2, MPI_Comm *rowComm, MPI_Comm *colComm) {
    for (int i = 0; i < dims[0]; ++i) {
        if (rankY == i){
            MPI_Bcast(subMatrixA, summandsA[rankY] * N2, MPI_DOUBLE, 0, rowComm[i]);
        }
    }
    for (int i = 0; i < dims[1]; ++i) {
        if (rankX == i){
            MPI_Bcast(subMatrixB, summandsB[rankX] * N2, MPI_DOUBLE, 0, colComm[i]);
        }
    }
}

void data_collection(int sizeSubmatrixA, int sizeSubmatrixB, int N3, int rank, int numprocs, double *matrixRes,
                     double *submatrixRes, int*dims, MPI_Comm gridComm) {
    MPI_Datatype send_submatrix, send_submatrix_resized;
    MPI_Type_vector(sizeSubmatrixA, sizeSubmatrixB, N3, MPI_DOUBLE, &send_submatrix);
    MPI_Type_commit(&send_submatrix);
    MPI_Type_create_resized(send_submatrix, 0, (int)(sizeSubmatrixB * sizeof(double)), &send_submatrix_resized);
    MPI_Type_commit(&send_submatrix_resized);

    int *sendOffset = NULL, *sendNum = NULL;
    if (rank == 0) {
        sendOffset = calloc(numprocs, sizeof(int));
        sendNum = calloc(numprocs, sizeof(int));
        sendOffset[0] = 0;
        for (int i = 0; i < dims[Y]; i++) {
            for (int j = 0; j < dims[X]; j++) {
                sendOffset[i + j * dims[Y]] = (j * sizeSubmatrixB + sizeSubmatrixA * i * N3) / sizeSubmatrixB;
                //printf("%d ",(j * sizeSubmatrixB + sizeSubmatrixA * i * N3) / sizeSubmatrixB);
            }
        }
        for (int i = 0; i < numprocs; ++i) {
            sendNum[i] = 1;
        }
    }

    MPI_Gatherv(submatrixRes, sizeSubmatrixA * sizeSubmatrixB, MPI_DOUBLE, matrixRes,
                sendNum, sendOffset, send_submatrix_resized, 0, gridComm);

    MPI_Type_free(&send_submatrix);
    MPI_Type_free(&send_submatrix_resized);
    if (rank == 0) {
        free(sendOffset);
        free(sendNum);
    }
}

int main(int argc, char *argv[]) {
    MPI_Comm gridComm;
    MPI_Group gridGroup;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2] = {0, 0};
    int numprocs, rank;
    double *MatrixA = NULL, *MatrixB = NULL, *MatrixRes = NULL;
    double start_time = 0.0, end_time;
    int N1 = DEFAULT, N2 = DEFAULT, N3 = DEFAULT;
    if (argc >= 2) {
        N1 = atoi(argv[1]);
        N2 = atoi(argv[2]);
        N3 = atoi(argv[3]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    create_gridComm(dims, periods, coords, numprocs, &gridComm, rank);
    int rankX = coords[X], rankY = coords[Y];
    MPI_Comm_group(gridComm, &gridGroup);
    MPI_Comm *colComm = (MPI_Comm*) malloc(dims[X] * sizeof(MPI_Comm));
    MPI_Comm *rowComm = (MPI_Comm*) malloc(dims[Y] * sizeof(MPI_Comm));
    create_Comms(&gridComm, &gridGroup, rowComm, colComm, dims);

    int *summandsA, *summandsB;
    summandsA = decompose(N1, dims[Y]);
    summandsB = decompose(N3, dims[X]);

    double *subMatrixA = (double*) malloc(summandsA[rankY] * N2 * sizeof(double));
    double *subMatrixB = (double*) malloc(summandsB[rankX] * N2 * sizeof(double));
    double *subMatrixRes = (double*) malloc(summandsA[rankY] * summandsB[rankX] * sizeof(double));

    if (rank == 0){
        double *MatrixBtmp = create_matrix(N2, N3);
        MatrixB = create_matrix(N3, N2);
        transpose_matrix(MatrixBtmp, N2, N3, MatrixB);
        free(MatrixBtmp);
        MatrixA = create_matrix(N1, N2);
        MatrixRes = matrixCreateEmpty(N1, N3);
    }

    start_time = MPI_Wtime();
    matrix_partition(MatrixA, N2, summandsA, subMatrixA, rankY, rankX, dims[Y], colComm);
    matrix_partition(MatrixB, N2, summandsB, subMatrixB, rankX, rankY, dims[X], rowComm);

    data_distribution(subMatrixA, subMatrixB, rankY, rankX, summandsA, summandsB, dims, N2, rowComm, colComm);

    matrix_multiplication(subMatrixA, subMatrixB, subMatrixRes, summandsA[rankY], N2, summandsB[rankX]);

    data_collection(summandsA[rankY], summandsB[rankX], N3, rank, numprocs, MatrixRes, subMatrixRes, dims, gridComm);
    end_time = MPI_Wtime();

    printf("Rank %d\n", rank);
    matrixPrint(subMatrixA, summandsA[rankY], summandsB[rankX]);
    matrixPrint(subMatrixB, summandsA[rankY], summandsB[rankX]);
//    printf("Rank %d\n", rank);
//    matrixPrint(subMatrixRes, summandsA[rankY], summandsB[rankX]);
    if (rank == 0) {
        matrixPrint(MatrixRes, N1, N3);
        printf("Time taken: %lf\n", end_time - start_time);
    }

    MPI_Finalize();
    freeProcess(MatrixA, MatrixB, MatrixRes, rowComm, colComm, summandsA,
                summandsB, rank);
    return 0;
}