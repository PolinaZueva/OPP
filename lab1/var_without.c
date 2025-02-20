#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846
#define E 1e-5

double calculateNorm(double* b, int N){
    double sum = 0.0;
    for (int i = 0; i < N; i++){
        sum += b[i] * b[i]; 
    }
    return sqrt(sum); 
}

void subVectors(double* Ax, double* b, int N, double* res) {
    for (int i = 0; i < N; i++){
        res[i] = Ax[i] - b[i];
    }
}

void multiplyMatrixVector(double* A, double* x, int N, double* res) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        res[i] = sum;
    }
}

int test(double* tempVector, double* b, int N) {
    return (calculateNorm(tempVector, N) / calculateNorm(b, N)) < E;
}

void multiplyNumberVector(double tau, double* tempVector, int N, double* res) {
    for (int i = 0; i < N; i++) {
        res[i] = tau * tempVector[i];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
	    printf("Use: %s <N> <tau> \n", argv[0]);
	    return 0;
    }

    int N = atoi(argv[1]);
    double tau = atof(argv[2]);
    
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

    multiplyMatrixVector(A, u, N, b);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    int countIterations = 0;
    while (1) {
        multiplyMatrixVector(A, x, N, Ax);
        subVectors(Ax, b, N, tempVector);

        if (test(tempVector, b, N)) break;

        multiplyNumberVector(tau, tempVector, N, tempVector);
        subVectors(x, tempVector, N, x);

        if (++countIterations > N * 100) {
            printf("Метод не сошелся, попробуйте другое значение tau\n");
            break; 
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    double difTime = (end.tv_sec - start.tv_sec) + 0.000000001 * (end.tv_nsec - start.tv_nsec);
    printf("Время выполнения: %lf секунд\n", difTime);

    for (int i = 0; i < 5 && i < N; i++) {
	    printf("x[%d] = %f\n", i, x[i]);
    }

    free(A);
    free(x);
    free(b);
    free(u);
    free(Ax);
    free(tempVector);
    return 0;
}
