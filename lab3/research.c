#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <float.h>
#include <time.h>
#include <math.h>

int N1, N2, N3;

int checkConvertArg(char* arg, char* name, int rank) {
    char *endptr;
    long var = strtol(arg, &endptr, 10);
    if (*endptr != '\0' || var <= 0) {
        if (rank == 0) {
            printf("Error: %s = %s is not a positive number\n", name, arg);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return (int)var;
}

void printMatrix(double* matrix, int rows, int cols, char* name, int rank) {
    printf("Process %d: Matrix %s (%dx%d):\n", rank, name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void multiplyMatrix(double* localA, double* localB, double* localC, int localN1, int localN3) {
	for (int i = 0; i < localN1; i++) {
		for (int j = 0; j < localN3; j++) {
			localC[i * localN3 + j] = 0.0f;
		}
    }

    for (int i = 0; i < localN1; i++) {
        for (int k = 0; k < N2; k++) {
            for (int j = 0; j < localN3; j++) {
                localC[i * localN3 + j] += localA[i * N2 + k] * localB[k * localN3 + j];
            }
        }
    }
}

void checkMatrices(double* A, double* B, double* C, double* AxB, int N1, int N3) {
    double epsilon = 1e-10;
    multiplyMatrix(A, B, AxB, N1, N3);
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N3; j++) {            
            if (fabs(AxB[i * N3 + j] - C[i * N3 + j]) > epsilon) {
                printf("!!! Матрицы A×B и C не равны\n");
                return;
            }
        }
    }
    printf("!!! Матрицы A×B и C равны\n");
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

    p1 = checkConvertArg(argv[1], "p1", rank);
    p2 = checkConvertArg(argv[2], "p2", rank);
    N1 = checkConvertArg(argv[3], "N1", rank);
    N2 = checkConvertArg(argv[4], "N2", rank);
    N3 = checkConvertArg(argv[5], "N3", rank);    

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

    dims[0] = p1;  //по x - строки
    dims[1] = p2;  //по y - столбцы
    int reorder = 1;  //может перераспределять ранги процессов (для оптимизации)
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &comm2d);  //создает 2D-решетку процессов - новая топология, новый коммуникатор   

    MPI_Comm_rank(comm2d, &rank);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);  //возвращает координаты текущего процесса в решетке

    int varyingCoords[2];  //1-меняем, 0-запрет
    varyingCoords[0] = 0;  //фиксируем строку
    varyingCoords[1] = 1;  //меняем столбцы
    MPI_Cart_sub(comm2d, varyingCoords, &commRow);  //создаем подкоммуникатор
    varyingCoords[0] = 1;  
    varyingCoords[1] = 0;  
    MPI_Cart_sub(comm2d, varyingCoords, &commCol);
    
    /*MPI_Barrier(comm2d);
    for (int i = 0; i < size; i++) {
        MPI_Barrier(comm2d);
        if (rank == i) {
            printf("Process %d: координаты (%d, %d)\n", rank, coords[0], coords[1]);
            fflush(stdout);
        }
        MPI_Barrier(comm2d);
    }
    MPI_Barrier(comm2d);*/
            
    int localN1 = N1 / p1;
    int localN3 = N3 / p2;

    double *fullA = NULL, *fullB = NULL, *fullC = NULL;
    double *localA = (double*)calloc(localN1 * N2, sizeof(double));
    double *localB = (double*)calloc(N2 * localN3,  sizeof(double));
    double *localC = (double*)calloc(localN1 * localN3, sizeof(double));  

    srand(time(NULL));   
    if (rank == 0) {
        fullA = (double*)calloc(N1 * N2, sizeof(double));
        fullB = (double*)calloc(N2 * N3, sizeof(double));
        fullC = (double*)calloc(N1 * N3, sizeof(double));     
        for (int i = 0; i < N1 * N2; i++) {
            fullA[i] = (double)(rand() % 100) / 10.0f;
        }

        for (int i = 0; i < N2 * N3; i++) {
            fullB[i] = (double)(rand() % 100) / 10.0f;
        }
    }

    double minMultiplyTime = DBL_MAX, minCommTime = DBL_MAX, minTotalTime = DBL_MAX;
    int numRuns = 5;
    for (int run = 0; run < numRuns; run++) {
        double totalStart = MPI_Wtime();

        //распределяем данные между процессами
        MPI_Datatype horizontalBlockTypeA;
        MPI_Type_contiguous(localN1 * N2, MPI_DOUBLE, &horizontalBlockTypeA);
        MPI_Type_commit(&horizontalBlockTypeA);  //регистрирует новый тип
        if (coords[1] == 0) {
            MPI_Scatter(fullA, 1, horizontalBlockTypeA, localA, localN1 * N2, MPI_DOUBLE, 0, commCol);
        }
        MPI_Bcast(localA, localN1 * N2, MPI_DOUBLE, 0, commRow);

        MPI_Datatype verticalBlockTypeB, resizedTypeB;
        MPI_Type_vector(N2, localN3, N3, MPI_DOUBLE, &verticalBlockTypeB);
        MPI_Type_create_resized(verticalBlockTypeB, 0, localN3 * sizeof(double), &resizedTypeB);  //изменяет шаг типа verticalBlockTypeB, чтобы p2 блоков можно было отправить как последовательные куски памяти в MPI_Scatter
        MPI_Type_commit(&verticalBlockTypeB);
        MPI_Type_commit(&resizedTypeB);
        if (coords[0] == 0) {
            MPI_Scatter(fullB, 1, resizedTypeB, localB, N2 * localN3, MPI_DOUBLE, 0, commRow);
            //printMatrix(localB, N2, localN3, "localB (after scatter)", rank);
        }   
        MPI_Bcast(localB, N2 * localN3, MPI_DOUBLE, 0, commCol);   
        if (rank == 0) {
            //printMatrix(localB, N2, localN3, "localB (after broadcast)", rank);
        }                 

        double multiplyStart = MPI_Wtime();
        multiplyMatrix(localA, localB, localC, localN1, localN3);
        double multiplyEnd = MPI_Wtime();

        //printMatrix(localC, localN1, localN3, "localC", rank);

        if (rank == 0 && fullC == NULL) {
            fullC = (double*)malloc(N1 * N3 * sizeof(double));
        }

        MPI_Datatype submatrixType;
        MPI_Type_vector(localN1, localN3, N3, MPI_DOUBLE, &submatrixType);
        MPI_Type_create_resized(submatrixType, 0, localN3 * sizeof(double), &submatrixType);
        MPI_Type_commit(&submatrixType);    
        
        int *recvcounts = (int*)malloc(size * sizeof(int));  //сколько данных принять от каждого процесса
        int *displs = (int*)malloc(size * sizeof(int));
        if (rank == 0) {
            for (int procRank = 0; procRank < size; procRank++) {
                recvcounts[procRank] = 1;  //от каждого процесса принимается ровно 1 блок типа submatrixType
                int procCoords[2];  //[0]-строки до p1−1,[1]-столбцы до p2−1
                MPI_Cart_coords(comm2d, procRank, 2, procCoords);
                displs[procRank] = procCoords[0] * localN1 * dims[1] + procCoords[1];
            }
            MPI_Gatherv(localC, localN1 * localN3, MPI_DOUBLE, fullC, recvcounts, displs, submatrixType, 0, comm2d); 
        } else {
            MPI_Gatherv(localC, localN1 * localN3, MPI_DOUBLE, NULL, recvcounts, displs, submatrixType, 0, comm2d);
        }     
        
        MPI_Type_free(&horizontalBlockTypeA);
        MPI_Type_free(&verticalBlockTypeB);
        MPI_Type_free(&resizedTypeB);
        MPI_Type_free(&submatrixType); 

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
        //printMatrix(fullA, N1, N2, "A", rank);
        //printMatrix(fullB, N2, N3, "B", rank);
        //printMatrix(fullC, N1, N3, "Result C", rank);
        double* AxB = (double*)calloc(N1 * N3, sizeof(double));
        checkMatrices(fullA, fullB, fullC, AxB, N1, N3); 
        free(AxB);
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