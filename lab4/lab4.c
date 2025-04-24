#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

#define a 1e+5 
#define epsilon 1e-8

#define x0 -1.0
#define y0 -1.0
#define z0 -1.0

#define Nx 360
#define Ny 360
#define Nz 360

#define Dx 2.0
#define Dy 2.0
#define Dz 2.0

#define hx (Dx / (Nx-1.0))
#define hy (Dy / (Ny-1.0))
#define hz (Dz / (Nz-1.0))

#define tempValue (1.0 / ((2 / (hx * hx)) + (2 / (hy * hy)) + (2 / (hz * hz)) + a))

double phiFunction(double x, double y, double z) {  //задаем граничные условия
    return x * x + y * y + z * z;
}

double rhoRight(double x, double y, double z) {  //вычисляем правую часть уравнения
    return 6 - a * phiFunction(x, y, z);
}

double methodJacobi(double* prevPhi, int x, int y, int z, int globalIndexStart, int localNx, double* leftRegion, double* rightRegion, int rank, int size) {
    double phiX, phiY, phiZ;

    if (x == 1 && rank != 0) {  //левая граница?
        phiX = (leftRegion[Nz * y + z] + prevPhi[Ny * Nz * (x + 1) + Nz * y + z]) / (hx * hx);
    } else if (x == localNx - 2 && rank != size - 1) {  //правая граница?
        phiX = (prevPhi[Ny * Nz * (x - 1) + Nz * y + z] + rightRegion[Nz * y + z]) / (hx * hx);
    } else {  //внутренние точки
        phiX = (prevPhi[Ny * Nz * (x + 1) + Nz * y + z] + prevPhi[Ny * Nz * (x - 1) + Nz * y + z]) / (hx * hx);
    }

    phiY = (prevPhi[Ny * Nz * x + Nz * (y + 1) + z] + prevPhi[Ny * Nz * x + Nz * (y - 1) + z]) / (hy * hy);
    phiZ = (prevPhi[Ny * Nz * x + Nz * y + (z + 1)] + prevPhi[Ny * Nz * x + Nz * y + (z - 1)]) / (hz * hz);

    return tempValue * (phiX + phiY + phiZ - rhoRight(x0 + (globalIndexStart + x - 1) * hx, y0 + y * hy, z0 + z * hz));
}

void calculatePhiInside(double* prevPhi, double* phi, int localNx, int globalIndexStart, double* maxDiff, double* leftRegion, double* rightRegion, int rank, int size) {
    for (int x = 2; x < localNx - 2; x++) {
        for (int y = 1; y < Ny - 1; y++) {
            for (int z = 1; z < Nz - 1; z++) {
                phi[Ny * Nz * x + Nz * y + z] = methodJacobi(prevPhi, x, y, z, globalIndexStart, localNx, leftRegion, rightRegion, rank, size);
                double diff = fabs(phi[Ny * Nz * x + Nz * y + z] - prevPhi[Ny * Nz * x + Nz * y + z]);
                if (diff > *maxDiff) {
                    *maxDiff = diff;
                }
            }
        }
    }
}

void calculatePhiOutside(double* prevPhi, double* phi, int localNx, int globalIndexStart, double* maxDiff, double* leftRegion, double* rightRegion, int rank, int size) {
    if (rank != 0) {
        for (int y = 1; y < Ny - 1; y++) {
            for (int z = 1; z < Nz - 1; z++) {
                phi[Ny * Nz * 1 + Nz * y + z] = methodJacobi(prevPhi, 1, y, z, globalIndexStart, localNx, leftRegion, rightRegion, rank, size);
                double diff = fabs(phi[Ny * Nz * 1 + Nz * y + z] - prevPhi[Ny * Nz * 1 + Nz * y + z]);
                if (diff > *maxDiff) {
                    *maxDiff = diff;
                }
            }
        }
    }

    if (rank != size - 1) {
        for (int y = 1; y < Ny - 1; y++) {
            for (int z = 1; z < Nz - 1; z++) {
                phi[Ny * Nz * (localNx - 2) + Nz * y + z] = methodJacobi(prevPhi, localNx - 2, y, z, globalIndexStart, localNx, leftRegion, rightRegion, rank, size);
                double diff = fabs(phi[Ny * Nz * (localNx - 2) + Nz * y + z] - prevPhi[Ny * Nz * (localNx - 2) + Nz * y + z]);
                if (diff > *maxDiff) {
                    *maxDiff = diff;
                }
            }
        }
    } 
}

void checkMaxDifference(double* phi, int globalIndexStart, int localNx, int rank) {
    double error = 0.0;
    for (int x = 1; x < localNx - 1; x++) {
        double coordX = x0 + (globalIndexStart + x - 1) * hx;
        for (int y = 1; y < Ny - 1; y++) {
            double coordY = y0 + y * hy;
            for (int z = 1; z < Nz - 1; z++) {
                double coordZ = z0 + z * hz;
                double diff = fabs(phi[Ny * Nz * x + Nz * y + z] - phiFunction(coordX, coordY, coordZ));
                if (diff > error) {
                    error = diff;
                }
            }
        }
    }

    double globalError;
    MPI_Allreduce(&error, &globalError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Максимальная разница: %e\n", globalError);
    } 
}

int main(int args, char* argv[]) {
    MPI_Init(&args, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int tempLocalNx = Nx / size + (rank < Nx % size ? 1 : 0);
    int localNx = tempLocalNx + 2;
    int globalIndexStart = 0;
    for (int i = 0; i < rank; i++) {
        globalIndexStart += (Nx / size) + (i < Nx % size ? 1 : 0);
    }

    double* phi = (double*)malloc(sizeof(double) * localNx * Ny * Nz);
    double* prevPhi = (double*)malloc(sizeof(double) * localNx * Ny * Nz);
    double* sendLeft = (double*)malloc(sizeof(double) * Ny * Nz);
    double* recvLeft = (double*)malloc(sizeof(double) * Ny * Nz);
    double* sendRight = (double*)malloc(sizeof(double) * Ny * Nz);
    double* recvRight = (double*)malloc(sizeof(double) * Ny * Nz);

    double minTime = DBL_MAX;
    int numRuns = 1;
    int lastIter = 0;
    for (int run = 0; run < numRuns; run++) {
        for (int x = 0; x < localNx; x++) {
            int globalIndex = globalIndexStart + x - 1;
            double coordX = x0 + globalIndex * hx;
            for (int y = 0; y < Ny; y++) {
                double coordY = y0 + y * hy;
                for (int z = 0; z < Nz; z++) {
                    double coordZ = z0 + z * hz;
                    phi[Ny * Nz * x + Nz * y + z] = 0.0;
                    prevPhi[Ny * Nz * x + Nz * y + z] = 0.0;
                    if ((globalIndex == 0 || globalIndex == Nx - 1 || y == 0 || y == Ny - 1 || z == 0 || z == Nz - 1) && (x > 0 && x < localNx - 1)) {
                        phi[Ny * Nz * x + Nz * y + z] = phiFunction(coordX, coordY, coordZ);
                        prevPhi[Ny * Nz * x + Nz * y + z] = phiFunction(coordX, coordY, coordZ);
                    }
                }
            }
        }

        int check = 0;
        int iter = 0;
        MPI_Request requests[4];  //т.к. четыре асинхронные операции может выполнять процесс
        double timeStart = MPI_Wtime();

        do {  //пока максимальная разница между последовательными итерациями не станет меньше epsilon
            double maxDiff = 0.0;
            double* tmp = prevPhi;
            prevPhi = phi;
            phi = tmp;

            int reqCount = 0;  //для отслеживания числа операций
            if (rank != 0) {
                for (int y = 0; y < Ny; ++y) {
                    for (int z = 0; z < Nz; ++z) {
                        sendLeft[Nz * y + z] = prevPhi[Ny * Nz * 1 + Nz * y + z];
                    }
                }
                MPI_Isend(sendLeft, Ny * Nz, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[reqCount++]);  //передаем граничные данные соседним процессам
                MPI_Irecv(recvLeft, Ny * Nz, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[reqCount++]);  //принимаем
            }
            if (rank != size - 1) {
                for (int y = 0; y < Ny; ++y) {
                    for (int z = 0; z < Nz; ++z) {
                        sendRight[Nz * y + z] = prevPhi[Ny * Nz * (localNx - 2) + Nz * y + z];
                    }
                }
                MPI_Isend(sendRight, Ny * Nz, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[reqCount++]);
                MPI_Irecv(recvRight, Ny * Nz, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[reqCount++]);
            }         

            calculatePhiInside(prevPhi, phi, localNx, globalIndexStart, &maxDiff, recvLeft, recvRight, rank, size);  //обновляем phi^(k+1) во внутренних точках
            
            for (int r = 0; r < reqCount; ++r) {  //блокируем процесс, пока все асинхронные операции не завершатся
                MPI_Wait(&requests[r], MPI_STATUS_IGNORE);  
            }

            calculatePhiOutside(prevPhi, phi, localNx, globalIndexStart, &maxDiff, recvLeft, recvRight, rank, size);  //обновляем phi^(k+1) на границах

            double globalMaxDiff;
            MPI_Allreduce(&maxDiff, &globalMaxDiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            check = (globalMaxDiff < epsilon);
            iter++;
        } while (!check);

        double timeEnd = MPI_Wtime();
        if (timeEnd - timeStart < minTime) {
            minTime = timeEnd - timeStart;
            lastIter = iter;
        }

        if (run == numRuns - 1) {
            checkMaxDifference(phi, globalIndexStart, localNx, rank);
        }        
    }

    if (rank == 0) {
        printf("Количество итераций: %d\n", lastIter);
        printf("Минимальное время за %d запусков: %f секунд\n", numRuns, minTime);
    }

    free(phi);
    free(prevPhi);
    free(sendLeft);
    free(recvLeft);
    free(sendRight);
    free(recvRight);

    MPI_Finalize();
    return 0;
}