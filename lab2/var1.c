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

void multiplyMatrixVector(double* localA, double* x, double* res, int localRowsFromN, int* recvcounts, int* displs, MPI_Comm comm) {
    double* localRes = (double*)malloc(localRowsFromN * sizeof(double));
    for (int i = 0; i < localRowsFromN; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += localA[i * N + j] * x[j];
        }
        localRes[i] = sum;        
    }
    MPI_Allgatherv(localRes, localRowsFromN, MPI_DOUBLE, res, recvcounts, displs, MPI_DOUBLE, comm);
    free(localRes);
}

void subVectors(double* Ax, double* b, double* res) {
    for (int i = 0; i < N; i++){
        res[i] = Ax[i] - b[i];
    }
}

double calculateNorm(double* b) {
    double sum = 0.0;
    for (int i = 0; i < N; i++){
        sum += b[i] * b[i]; 
    }
    return sqrt(sum);
}

int test(double* tempVector, double bNorm) {
    return (calculateNorm(tempVector) / bNorm) < E;
}

void multiplyNumberVector(double* tempVector, double* res) {
    for (int i = 0; i < N; i++) {
        res[i] = TAU * tempVector[i];
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
    double* x = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* u = (double*)malloc(N * sizeof(double));
    double* Ax = (double*)malloc(N * sizeof(double));
    double* tempVector = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        if (rank == 0) { u[i] = sin(2.0 * PI * i / N); }        
        x[i] = 0.0;
    }
    MPI_Bcast(u, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
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

    multiplyMatrixVector(localA, u, b, localRowsFromN, recvcounts, displs, MPI_COMM_WORLD);
    
    double minTime = DBL_MAX;

    for (int numRun = 0; numRun < 5; numRun++) {  
        for (int i = 0; i < N; i++) {
            x[i] = 0.0;
        }

        double start = MPI_Wtime();
        double bNorm = calculateNorm(b);

        while (1) {
            multiplyMatrixVector(localA, x, Ax, localRowsFromN, recvcounts, displs, MPI_COMM_WORLD);
            subVectors(Ax, b, tempVector);

            if (test(tempVector, bNorm)) break;

            multiplyNumberVector(tempVector, tempVector);
            subVectors(x, tempVector, x);
        }

        double end = MPI_Wtime();

        double difTime = end - start;
        if (rank == 0) {
            printf("Запуск %d: Время выполнения: %lf секунд\n", numRun + 1, difTime);
        }
        
        if (difTime < minTime) {
            minTime = difTime;
        }
    }

    if (rank == 0) {
        printf("Минимальное время выполнения: %lf секунд\n", minTime);

        double maxRaznitsa = 0.0;
        for (int i = 0; i < N; i++) {
            double raznitsa = fabs(x[i] - u[i]);
            if (raznitsa > maxRaznitsa) {
                maxRaznitsa = raznitsa;
            }
        }
        printf("Модуль максимальной разницы: %lf\n", maxRaznitsa);    

        printf("Первые 5 элементов векторов x и u для сравнения:\n");
        for (int i = 0; i < 5 && i < N; i++) {
            printf("x[%d] = %lf, u[%d] = %lf, разница = %lf\n", i, x[i], i, u[i], fabs(x[i] - u[i]));
        }
    }  

    free(localA);
    free(x);
    free(b);
    free(u);
    free(Ax);
    free(tempVector);
    free(recvcounts);
    free(displs);

    MPI_Finalize(); 
    return 0;
}