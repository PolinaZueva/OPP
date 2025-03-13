#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h> 

#define PI 3.14159265358979323846
#define E 1e-5
#define TAU 0.001
#define N 200
//1580
double calculateNorm(double* b){
    double sum = 0.0;
    for (int i = 0; i < N; i++){
        sum += b[i] * b[i]; 
    }
    return sqrt(sum); 
}

void subVectors(double* Ax, double* b, double* res) {
    for (int i = 0; i < N; i++){
        res[i] = Ax[i] - b[i];
    }
}

void multiplyMatrixVector(double* A, double* x, double* res) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        res[i] = sum;
    }
}

int test(double* tempVector, double bNorm) {
    return (calculateNorm(tempVector) / bNorm) < E;
}

void multiplyNumberVector(double* tempVector, double* res) {
    for (int i = 0; i < N; i++) {
        res[i] = TAU * tempVector[i];
    }
}

int main() {    
    double* A = (double*)malloc(N * N * sizeof(double));
    double* x = (double*)malloc(N * sizeof(double));
    double* b = (double*)malloc(N * sizeof(double));
    double* u = (double*)malloc(N * sizeof(double));
    double* Ax = (double*)malloc(N * sizeof(double));
    double* tempVector = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        u[i] = sin(2.0 * PI * i / N);
        x[i] = 0.0;
		for (int j = 0; j < N; j++) {
			A[i * N + j] = 1.0;
			if (i == j) {
				A[i * N + j] = 2.0;
			}
		}
	}    

    multiplyMatrixVector(A, u, b);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    double bNorm = calculateNorm(b);
    while (1) {
        multiplyMatrixVector(A, x, Ax);
        subVectors(Ax, b, tempVector);

        if (test(tempVector, bNorm)) break;

        multiplyNumberVector(tempVector, tempVector);
        subVectors(x, tempVector, x);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double difTime = (end.tv_sec - start.tv_sec) + 0.000000001 * (end.tv_nsec - start.tv_nsec);
    printf("Время выполнения: %lf секунд\n", difTime);

    double maxRaznitsa = 0.0;
    for (int i = 0; i < N; i++) {
        double Raznitsa = fabs(x[i] - u[i]);
        if (Raznitsa > maxRaznitsa) {
            maxRaznitsa = Raznitsa;
        }
    }
    printf("Модуль максимальной разницы: %e\n", maxRaznitsa);

    free(A);
    free(x);
    free(b);
    free(u);
    free(Ax);
    free(tempVector);
    return 0;
}