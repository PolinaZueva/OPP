#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <float.h>

int N1, N2, N3;

void print_matrix(double* matrix, int rows, int cols, char* name, int rank) {
    printf("Process %d: Matrix %s (%dx%d):\n", rank, name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

void multiplyMatrix(double* localA, double* localB, double* localC, int localN1, int localN3) {
    for (int i = 0; i < localN1; i++) {
        for (int j = 0; j < localN3; j++) {
            localC[i * localN3 + j] = 0;
            for (int k = 0; k < N2; k++) {
                localC[i * localN3 + j] += localA[i * N2 + k] * localB[k * localN3 + j];
            }
        }
    }
}

int main(int args, char* argv[]) {
    MPI_Init(&args, &argv);

    MPI_Comm comm2d, commRow, commCol;
    int dims[2], periods[2] = {0, 0}, coords[2] = {0, 0};
    
    int size, rank, p1, p2;
    MPI_Comm_size(MPI_COMM_WORLD, &size);  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (args != 6) {
        if (rank == 0) {
            printf("Use: mpirun -np <число процессов> ./main <p1> <p2> <N1> <N2> <N3>\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char *endptr;
    long var1 = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || var1 <= 0) {
        if (rank == 0) {
            printf("Error: p1 %s is non-positive number\n", argv[1]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    p1 = (int)var1;

    long var2 = strtol(argv[2], &endptr, 10);
    if (*endptr != '\0' || var2 <= 0) {
        if (rank == 0) {
            printf("Error: p2 %s is non-positive number\n", argv[2]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    p2 = (int)var2;

    N1 = strtol(argv[3], &endptr, 10);
    if (*endptr != '\0' || N1 <= 0) { 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }    
    N2 = strtol(argv[4], &endptr, 10);
    if (*endptr != '\0' || N2 <= 0) { 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    N3 = strtol(argv[5], &endptr, 10);
    if (*endptr != '\0' || N3 <= 0) { 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (p1 * p2 != size) {
        if (rank == 0) {
            printf("Error: p1 * p2 = %d * %d != size = %d\n", p1, p2, size);
            fflush(stdout);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if ((N1 % p1 != 0) || (N3 % p2 != 0)) {
        if (rank == 0) {
            printf("Error: N1, N3 not кратны p1 and p2\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        printf("DIMS: %d %d\n", p1, p2);
    }

    dims[0] = p1;  //по x
    dims[1] = p2;  //по y
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);    

    MPI_Comm_rank(comm2d, &rank);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);

    int varyingCoords[2];
    varyingCoords[0] = 0;  //игнорируем изменения
    varyingCoords[1] = 1;  //сохраняем изменение в подгруппе
    MPI_Cart_sub(comm2d, varyingCoords, &commRow);
    varyingCoords[0] = 1;  
    varyingCoords[1] = 0;  
    MPI_Cart_sub(comm2d, varyingCoords, &commCol);
    
    MPI_Barrier(comm2d);
    for (int i = 0; i < size; i++) {
        MPI_Barrier(comm2d);
        if (rank == i) {
            printf("Process %d: координаты (%d, %d)\n", rank, coords[0], coords[1]);
            fflush(stdout);
        }
        MPI_Barrier(comm2d);
    }
    MPI_Barrier(comm2d);
            
    int localN1 = N1 / p1;
    int localN3 = N3 / p2;

    double *fullA = NULL, *fullB = NULL, *fullC = NULL;
    double *localA = (double*)calloc(localN1 * N2, sizeof(double));
    double *localB = (double*)calloc(N2 * localN3,  sizeof(double));
    double *localC = (double*)calloc(localN1 * localN3, sizeof(double));  

    if (rank == 0) {
        fullA = (double*)calloc(N1 * N2, sizeof(double));
        fullB = (double*)calloc(N2 * N3, sizeof(double));
        fullC = (double*)calloc(N1 * N3, sizeof(double));
        for (int i = 0; i < N1 * N2; i++) {
            fullA[i] = 1.0;
        }
        for (int i = 0; i < N2 * N3; i++) {
            fullB[i] = 1.0;
        }
    }

    double minMultiplyTime = DBL_MAX, minCommTime = DBL_MAX, minTotalTime = DBL_MAX;
    int numRuns = 5;
    for (int run = 0; run < numRuns; run++) {
        double totalStart = MPI_Wtime();

        MPI_Datatype horizontalA;
        MPI_Type_contiguous(localN1 * N2, MPI_DOUBLE, &horizontalA);
        MPI_Type_commit(&horizontalA);
        if (coords[1] == 0) {
            MPI_Scatter(fullA, 1, horizontalA, localA, localN1 * N2, MPI_DOUBLE, 0, commCol);
        }

        MPI_Datatype verticalB;
        MPI_Type_vector(N2, localN3, N3, MPI_DOUBLE, &verticalB);
        MPI_Type_commit(&verticalB);

        MPI_Datatype verticalBContiguous;
        MPI_Type_contiguous(N2 * localN3, MPI_DOUBLE, &verticalBContiguous);
        MPI_Type_commit(&verticalBContiguous);    

        if (coords[0] == 0 && coords[1] == 0) {  //главный процесс первой строки отправляет части fullB другим процессам первой строки
            for (int i = 0; i < N2; i++) {
                for (int j = 0; j < localN3; j++) {
                    localB[i * localN3 + j] = fullB[i * N3 + j];
                }
            }

            for (int sendRank = 1; sendRank < p2; sendRank++) {
                MPI_Send(fullB + sendRank * localN3, 1, verticalB, sendRank, 0, commRow);
            }        
        } else if (coords[0] == 0 && coords[1] != 0) {  //принимаем свою часть fullB
            MPI_Recv(localB, 1, verticalBContiguous, 0, 0, commRow, MPI_STATUS_IGNORE);
        }

        //print_matrix(localB, N2, localN3, "localB (before broadcast)", rank);

        MPI_Bcast(localA, localN1 * N2, MPI_DOUBLE, 0, commRow);
        MPI_Bcast(localB, N2 * localN3, MPI_DOUBLE, 0, commCol);

        //print_matrix(localB, N2, localN3, "localB (after broadcast)", rank);  

        double multiplyStart = MPI_Wtime();
        multiplyMatrix(localA, localB, localC, localN1, localN3);
        double multiplyEnd = MPI_Wtime();

        //print_matrix(localC, localN1, localN3, "localC", rank);

        if (rank == 0 && fullC == NULL) {
            fullC = (double*)malloc(N1 * N3 * sizeof(double));
        }

        MPI_Datatype submatrixType;
        MPI_Type_vector(localN1, localN3, N3, MPI_DOUBLE, &submatrixType);
        MPI_Type_commit(&submatrixType);    
        
        if (rank == 0) {  //собираем данные на 0 процессе
            for (int i = 0; i < localN1; i++) {
                for (int j = 0; j < localN3; j++) {
                    fullC[i * N3 + j] = localC[i * localN3 + j];
                }
            }

            double* temp = (double*)malloc(localN1 * localN3 * sizeof(double));
            for (int senderRank = 1; senderRank < size; senderRank++) {
                MPI_Recv(temp, localN1 * localN3, MPI_DOUBLE, senderRank, 0, comm2d, MPI_STATUS_IGNORE);
                int senderRankCoords[2];
                MPI_Cart_coords(comm2d, senderRank, 2, senderRankCoords);
                int displsI = senderRankCoords[0] * localN1;
                int displsJ = senderRankCoords[1] * localN3;

                for (int i = 0; i < localN1; i++) {
                    for (int j = 0; j < localN3; j++) {
                        fullC[(displsI + i) * N3 + (displsJ + j)] = temp[i * localN3 + j];
                    }
                }
            }
            free(temp);
        } else {
            MPI_Send(localC, localN1 * localN3, MPI_DOUBLE, 0, 0, comm2d);
        }
        MPI_Type_free(&submatrixType);
        MPI_Type_free(&horizontalA);
        MPI_Type_free(&verticalB);
        MPI_Type_free(&verticalBContiguous);

        double totalEnd = MPI_Wtime();
        double multiplyTime = multiplyEnd - multiplyStart;
        double commTime = (totalEnd - totalStart) - (multiplyEnd - multiplyStart);
        double totalTime = totalEnd - totalStart;

        if (multiplyTime < minMultiplyTime) {
            minMultiplyTime = multiplyTime;
        }
        if (commTime < minCommTime) {
            minCommTime = commTime;
        }
        if (totalTime < minTotalTime) {
            minTotalTime = totalTime;
        }
    }    

    if (rank == 0) {
        printf("Минимальное время вычислений: %lf секунд\n", minMultiplyTime);
        printf("Минимальное время коммуникаций: %lf секунд\n", minCommTime);
        printf("Минимальное общее время: %lf секунд\n", minTotalTime);
        //print_matrix(fullA, N1, N2, "A", rank);
        //print_matrix(fullB, N2, N3, "B", rank);
        //print_matrix(fullC, N1, N3, "Result C", rank);
    }

    free(localA); 
    free(localB); 
    free(localC);
    if (rank == 0) { 
        free(fullA); 
        free(fullB); 
        free(fullC); 
    }

    MPI_Comm_free(&comm2d);
    MPI_Comm_free(&commRow);
    MPI_Comm_free(&commCol);      

    MPI_Finalize();
    return 0;
}