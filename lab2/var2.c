#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <mpi.h>

#define PI 3.14159265358979323846
#define E 1e-5
#define TAU 0.001
#define N 1580

void multiplyMatrixVector(double* localA, double* localX, double* localRes, int localRowsFromN, int* recvcounts, int* displs, int size, int rank, MPI_Comm comm) {
    int maxRows;
    MPI_Allreduce(&localRowsFromN, &maxRows, 1, MPI_INT, MPI_MAX, comm);

    if (maxRows <= 0) {
        MPI_Abort(comm, 1);
    }

    double* bufferX = (double*)calloc((size_t)maxRows, sizeof(double));
    if (bufferX == NULL) {
        MPI_Abort(comm, 1);
    }
    for (int i = 0; i < localRowsFromN; i++) {
        bufferX[i] = localX[i];  //копируем в буфер
    }

    double* tempRes = (double*)calloc((size_t)localRowsFromN, sizeof(double));
    if (tempRes == NULL) {
        free(bufferX);
        MPI_Abort(comm, 1);
    }

    for (int step = 0; step < size; step++) {
        int senderRank = (rank - step + size) % size;
        int senderDispls = displs[senderRank];
        int senderRecvcounts = recvcounts[senderRank];

        for (int i = 0; i < localRowsFromN; i++) {
            for (int j = 0; j < senderRecvcounts; j++) {
                tempRes[i] += localA[i * N + (senderDispls + j)] * bufferX[j];
            }
        }
        
        if (step < size - 1) {
            int sendTo = (rank + 1) % size;
            int recvFrom = (rank - 1 + size) % size;
            MPI_Sendrecv_replace(bufferX, maxRows, MPI_DOUBLE, sendTo, 0, recvFrom, 0, comm, MPI_STATUS_IGNORE);
        }
    }

    for (int i = 0; i < localRowsFromN; i++) {
        localRes[i] = tempRes[i];
    }

    free(bufferX);
    free(tempRes);
}

void subVectors(double* localAx, double* localB, double* localRes, int localRowsFromN) {
    for (int i = 0; i < localRowsFromN; i++){
        localRes[i] = localAx[i] - localB[i];
    }
}

double calculateNorm(double* localB, int localRowsFromN) {
    double localSum = 0.0;
    for (int i = 0; i < localRowsFromN; i++){
        localSum += localB[i] * localB[i]; 
    }
    double allSum;
    MPI_Allreduce(&localSum, &allSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(allSum);
}

int test(double* localTempVector, double bNorm, int localRowsFromN) {
    return (calculateNorm(localTempVector, localRowsFromN) / bNorm) < E;
}

void multiplyNumberVector(double* localTempVector, double* localRes, int localRowsFromN) {
    for (int i = 0; i < localRowsFromN; i++) {
        localRes[i] = TAU * localTempVector[i];
    }
}

int main(int argc, char* argv[]) {    
    MPI_Init(&argc,&argv);

    int size,rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);  //общее число процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //номер процесса

    int numberOfRowsForProcess = N / size;
    int ostatok = N % size;
    int localRowsFromN = numberOfRowsForProcess + (rank < ostatok ? 1 : 0);
    int startRowForEachProcess = numberOfRowsForProcess * rank + (rank < ostatok ? rank : ostatok);  //начало строки матрицы A для какого-го процесса

    double* localA = (double*)malloc(localRowsFromN * N * sizeof(double));
    double* localX = (double*)malloc(localRowsFromN * sizeof(double));
    double* localB = (double*)malloc(localRowsFromN * sizeof(double));
    double* localU = (double*)malloc(localRowsFromN * sizeof(double));
    double* localAx = (double*)malloc(localRowsFromN * sizeof(double));
    double* localTempVector = (double*)malloc(localRowsFromN * sizeof(double));        

    for (int i = 0; i < localRowsFromN; i++) { 
        localU[i] = sin(2.0 * PI * (startRowForEachProcess + i) / N);
        localX[i] = 0.0; 
    }
	
    for (int i = 0; i < localRowsFromN; i++) {
        for (int j = 0; j < N; j++) {
		    localA[i * N + j] = 1.0;
			if (startRowForEachProcess + i == j) {
				localA[i * N + j] = 2.0;
			}
		}
    }

    int* recvcounts = (int*)malloc(size * sizeof(int));  //количество элементов от каждого процесса
    int* displs = (int*)malloc(size * sizeof(int));;  //массив смещений
    for (int i = 0; i < size; i++){
        recvcounts[i] = numberOfRowsForProcess + (i < ostatok ? 1 : 0);
        displs[i] = numberOfRowsForProcess * i + (i < ostatok ? i : ostatok);
    }    

    multiplyMatrixVector(localA, localU, localB, localRowsFromN, recvcounts, displs, size, rank, MPI_COMM_WORLD);
    
    double minTime = DBL_MAX;

    for (int run = 0; run < 5; run++) {  
        for (int i = 0; i < localRowsFromN; i++) {
            localX[i] = 0.0;
        }

        double start = MPI_Wtime();
        double bNorm = calculateNorm(localB, localRowsFromN);

        while (1) {
            multiplyMatrixVector(localA, localX, localAx, localRowsFromN, recvcounts, displs, size, rank, MPI_COMM_WORLD);
            subVectors(localAx, localB, localTempVector, localRowsFromN);

            if (test(localTempVector, bNorm, localRowsFromN)) break;

            multiplyNumberVector(localTempVector, localTempVector, localRowsFromN);
            subVectors(localX, localTempVector, localX, localRowsFromN);
        }

        double end = MPI_Wtime();

        double difTime = end - start;
        if (rank == 0) {
            printf("Запуск %d: Время выполнения: %lf секунд\n", run + 1, difTime);
        }
        
        if (difTime < minTime) {
            minTime = difTime;
        }
    }

    if (rank == 0) {
        printf("Минимальное время выполнения: %lf секунд\n", minTime);
    }

    double* fullX = (double*)malloc(N * sizeof(double)); 
    double* fullU = (double*)malloc(N * sizeof(double));
    MPI_Allgatherv(localX, localRowsFromN, MPI_DOUBLE, fullX, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localU, localRowsFromN, MPI_DOUBLE, fullU, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0) {
        double maxRaznitsa = 0.0;
        for (int i = 0; i < N; i++) {
            double raznitsa = fabs(fullX[i] - fullU[i]);
            if (raznitsa > maxRaznitsa) {
                maxRaznitsa = raznitsa;
            }
        }
        printf("Модуль максимальной разницы: %lf\n", maxRaznitsa);    

        printf("Первые 5 элементов векторов x и u для сравнения:\n");
        for (int i = 0; i < 5 && i < N; i++) {
            printf("x[%d] = %lf, u[%d] = %lf, разница = %lf\n", i, fullX[i], i, fullU[i], fabs(fullX[i] - fullU[i]));
        }
    }  

    free(localA);
    free(localX);
    free(localB);
    free(localU);
    free(localAx);
    free(localTempVector);
    free(recvcounts);
    free(displs);
    free(fullX);
    free(fullU);

    MPI_Finalize(); 
    return 0;
}