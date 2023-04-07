#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define max(X, Y) (((X) > (Y)) ? (X) : (Y))
#define DEFAULT 3
#define Y 0
#define X 1

int isPrime(int x) {
    for (int i = 2; i <= sqrt(x); ++i) {
        if (x % i == 0)
            return i;
    }
    return x;
}

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
    for (int i = 0; i <= (k / 2) + 1; ++i) {
        for (int j = 0; j < k; ++j) {
            if (sum < n) {
                summands[j]++;
                sum++;
            }
        }
    }
    return summands;
}

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

void FreeProcess(double* A, double* B, double* Res, MPI_Comm *colComm, MPI_Comm *rowComm,
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
        printf("%d %d\n", dims[Y], dims[X]);
    //creating communicator of 2d grid
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, gridComm);
    //getting rank of process in grid
    MPI_Comm_rank(*gridComm, &rank);
    //get coordinates in 2d grid
    //MPI_Cart_get(gridComm, 2, dims, periods, coords);
    MPI_Cart_coords(*gridComm, rank, 2, coords);
    printf("rank: %d coords: %d %d\n", rank, coords[Y], coords[X]);
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

void partition(double *Matrix, int N, int K, int *summands, double *subMatrix, int rankY, int rankX,
               int dim, MPI_Comm *groupComms) {
    if (rankX == 0) {
        int *sendNum = (int*) malloc(dim * sizeof(int));
        int *sendOffset = (int*) malloc(dim * sizeof(int));
        for (int i = 0; i < dim; ++i) {
            sendNum[i] = summands[i] * N;
            sendOffset[i] = i * summands[i] * K;
        }
        MPI_Scatterv(Matrix, sendNum, sendOffset, MPI_DOUBLE, subMatrix,
                     sendNum[rankX], MPI_DOUBLE, 0, groupComms[rankX]);
        printf("SUCCESS %d %d\n", rankY, rankX);
        free(sendNum);
        free(sendOffset);
    }
}

void partitionCol(double *Matrix, int N2, int N3) {

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
    MPI_Status status;

    create_gridComm(dims, periods, coords, numprocs, &gridComm, rank);
    int rankX = coords[X], rankY = coords[Y];
    MPI_Comm_group(gridComm, &gridGroup);
    MPI_Comm *colComm = (MPI_Comm*) malloc(dims[X] * sizeof(MPI_Comm));
    MPI_Comm *rowComm = (MPI_Comm*) malloc(dims[Y] * sizeof(MPI_Comm));
    create_Comms(&gridComm, &gridGroup, rowComm, colComm, dims);

    int *summandsA, *summandsB;
    summandsA = decompose(N1, dims[Y]);
    summandsB = decompose(N3, dims[X]);

    double *subMatrixRow = (double*) malloc(summandsA[rankY] * N2 * sizeof(double));
    double *subMatrixCol = (double*) malloc(summandsB[rankX] * N2 * sizeof(double));

    if (rank == 0){
        MatrixA = create_matrix(N1, N2);
        MatrixB = create_matrix(N2, N3);
        MatrixRes = create_emptyMatrix(N1, N3);

        //matrix_multiplication(MatrixA, MatrixB, MatrixRes, N1, N2, N3);
        //print_matrix(MatrixRes, N1, N3);
    }

    partition(MatrixA, N1, N2, summandsA, subMatrixRow, rankY, rankX, dims[Y], colComm);

    MPI_Finalize();
    FreeProcess(MatrixA, MatrixB, MatrixRes, rowComm, colComm, summandsA,
                summandsB, rank);
    return 0;
}