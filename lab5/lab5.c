#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <mpi.h>
#include <unistd.h>
#include <time.h>
#include <stddef.h>

#define N_TASKS 200
#define ITERATIONS 10

typedef struct {
    int repeatNum;  //количество повторений вычислений для задачи
    int completed;  //0 - не выполнена, 1 - выполнена/отправлена, 2 - получена
    int taskId;     //уникальный айди задачи    
} Task;

typedef struct {
    int tasksProcessed;
    double idleTime;
    double workTime;
    int tasksReceived;
    int tasksSent;
    int initial_weight;
    int total_weight;
    int own_tasks;
} ProcessStats;

Task* taskList = NULL;
int taskListSize = 0;
int taskListCapacity = 0;
pthread_mutex_t taskMutex;
pthread_cond_t tasksAvailable;  //сигнал для workerThread, когда появляется новые задачи
double globalRes = 0.0;
int allTasksDone = 0;
int rank, size;
MPI_Datatype MPI_TASK_TYPE, MPI_STATS_TYPE;
ProcessStats stats = {0, 0.0, 0.0, 0, 0, 0, 0, 0};

int enableLoadBalancing = 1;  //флаг балансировки
int distributionCode = 0;

void* workerThread(void* arg);
void* requestThread(void* arg);
void* responseThread(void* arg);
void printFinalStats(void);
void parseArguments(int argc, char** argv);
void resizeTaskList(int newCapacity);
void uniformDistribution(int* numberOfTasksForProcesses, int size);
void hillDistribution(int* numberOfTasksForProcesses, int size);
void distributeTasks(int distributionCode, int* numberOfTasksForProcesses, int size);

void parseArguments(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-balance") == 0) {
            enableLoadBalancing = 0;
        } else if (strcmp(argv[i], "--balance") == 0) {
            enableLoadBalancing = 1;
        } else {
            distributionCode = atoi(argv[i]);
            if (distributionCode < 0 || distributionCode > 1) {
                if (rank == 0) {
                    printf("Invalid distribution code. Use 0 (uniform) or 1 (hill).\n");
                }
                MPI_Finalize();
                exit(1);
            }
        }
    }
}

void resizeTaskList(int newCapacity) {  //для увеличения емкости для хранения новых задач
    if (newCapacity <= taskListCapacity) return;
    Task* newList = (Task*)malloc(newCapacity * sizeof(Task));
    if (!newList) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (taskList) {
        memcpy(newList, taskList, taskListSize * sizeof(Task));  //из таска в нью 
        free(taskList);
    }
    taskList = newList;
    taskListCapacity = newCapacity;
}

void uniformDistribution(int* numberOfTasksForProcesses, int size) {
    int perProcess = N_TASKS / size;  //за процесс
    for (int i = 0; i < size; i++) {
        numberOfTasksForProcesses[i] = perProcess;
    }
    int remaining = N_TASKS - perProcess * size;
    for (int i = 0; remaining > 0; i = (i + 1) % size) {  //распределяю по одной задаче от остатка 
        numberOfTasksForProcesses[i]++;
        remaining--;
    }
}

void hillDistribution(int* numberOfTasksForProcesses, int size) {  //процессы с более высокими номерами получают больше задач
    int totalTasks = N_TASKS;
    int totalAssignedTasks = 0;
    for (int i = 0; i < size; i++) {
        numberOfTasksForProcesses[i] = (i + 1) * 10;  //начальное число задач
        totalAssignedTasks += numberOfTasksForProcesses[i];
    }
    double normalization = (double)totalTasks / totalAssignedTasks;
    for (int i = 0; i < size; i++) {
        numberOfTasksForProcesses[i] = (int)(numberOfTasksForProcesses[i] * normalization);
    }
    int remaining_tasks = totalTasks;
    for (int i = 0; i < size; i++) {
        remaining_tasks -= numberOfTasksForProcesses[i];
    }
    for (int i = 0; remaining_tasks > 0; i = (i + 1) % size) {  
        numberOfTasksForProcesses[i]++;
        remaining_tasks--;
    }
}

void distributeTasks(int distributionCode, int* numberOfTasksForProcesses, int size) {
    if (size == 1) {  //если 1
        numberOfTasksForProcesses[0] = N_TASKS;
        return;
    }
    switch (distributionCode) {
        case 0: uniformDistribution(numberOfTasksForProcesses, size); break;
        case 1: hillDistribution(numberOfTasksForProcesses, size); break;
        default: uniformDistribution(numberOfTasksForProcesses, size); break;
    }
}

void* workerThread(void* arg) {  //выполняет задачи из списка
    while (1) {
        double idleStart = MPI_Wtime();
        pthread_mutex_lock(&taskMutex);  //блокируем
        int hasPendingTasks = 0;  //флаг задач в ожидании
        for (int i = 0; i < taskListSize; i++) {
            if (taskList[i].completed == 0 || taskList[i].completed == 2) {
                hasPendingTasks = 1;
                break;
            }
        }
        while (!hasPendingTasks && !allTasksDone) {
            pthread_cond_wait(&tasksAvailable, &taskMutex);  //ждем сигнала, временно освобождая мьютекс
            hasPendingTasks = 0;
            for (int i = 0; i < taskListSize; i++) {
                if (taskList[i].completed == 0 || taskList[i].completed == 2) {
                    hasPendingTasks = 1;
                    break;
                }
            }
        }
        if (allTasksDone && !hasPendingTasks) {
            pthread_mutex_unlock(&taskMutex);
            break;
        }

        //нашли первую доступную задачу
        int taskIndex = -1;
        for (int i = 0; i < taskListSize; i++) {
            if (taskList[i].completed == 0 || taskList[i].completed == 2) {
                taskIndex = i;
                break;
            }
        }
        if (taskIndex == -1) {  
            pthread_mutex_unlock(&taskMutex);
            continue;  
        }

        Task task = taskList[taskIndex];
        int wasTransferred = taskList[taskIndex].completed == 2;
        taskList[taskIndex].completed = 1;
        pthread_mutex_unlock(&taskMutex);
        stats.idleTime += MPI_Wtime() - idleStart;

        double workStart = MPI_Wtime();
        double result = 0.0;
        for (int i = 0; i < task.repeatNum; i++) {
            volatile double tmp = sqrt(i + 1);
            result += tmp;
        }
        usleep(10000);

        pthread_mutex_lock(&taskMutex);
        globalRes += result;
        stats.tasksProcessed++;
        stats.total_weight += task.repeatNum;
        if (!wasTransferred) stats.own_tasks++;  //если задача не получена, увеличиваем число собственных
        pthread_mutex_unlock(&taskMutex);
        stats.workTime += MPI_Wtime() - workStart;
    }
    return NULL;
}

void* requestThread(void* arg) {  //запрашивает у других, если своих задач мало 
    if (!enableLoadBalancing) return NULL;
    int* processRanks = (int*)malloc(size * sizeof(int));
    if (!processRanks) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < size; i++) processRanks[i] = i;
    srand(time(NULL) ^ rank);

    while (!allTasksDone) {
        pthread_mutex_lock(&taskMutex);
        int needsTasks = 1;
        for (int i = 0; i < taskListSize; i++) {
            if (taskList[i].completed == 0 || taskList[i].completed == 2) {
                needsTasks = 0;
                break;
            }
        }
        needsTasks = needsTasks && !allTasksDone;
        pthread_mutex_unlock(&taskMutex);

        if (!needsTasks || allTasksDone) {
            usleep(20000);
            continue;
        }

        for (int i = size - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = processRanks[i];
            processRanks[i] = processRanks[j];
            processRanks[j] = temp;
        }

        for (int i = 0; i < size; i++) {
            int targetRank = processRanks[i];
            if (targetRank == rank) continue;

            pthread_mutex_lock(&taskMutex);
            if (allTasksDone) {
                pthread_mutex_unlock(&taskMutex);
                break;
            }
            pthread_mutex_unlock(&taskMutex);

            int requestCode = 1;
            MPI_Send(&requestCode, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD);

            MPI_Request request;
            int numTasks;
            MPI_Irecv(&numTasks, 1, MPI_INT, targetRank, 1, MPI_COMM_WORLD, &request);
            int flag = 0;
            double startTime = MPI_Wtime();
            while (MPI_Wtime() - startTime < 0.1) {
                MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
                if (flag) break;
                usleep(1000);
            }
            if (!flag) {
                MPI_Cancel(&request);
                MPI_Request_free(&request);
                continue;
            }
            MPI_Wait(&request, MPI_STATUS_IGNORE);

            if (numTasks > 0) {
                Task* receivedTasks = (Task*)malloc(numTasks * sizeof(Task));
                if (!receivedTasks) {
                    fprintf(stderr, "Memory allocation failed\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                MPI_Recv(receivedTasks, numTasks, MPI_TASK_TYPE, targetRank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pthread_mutex_lock(&taskMutex);
                int oldSize = taskListSize;
                taskListSize += numTasks;
                if (taskListSize > taskListCapacity) {
                    resizeTaskList(taskListSize * 2);
                }
                for (int j = 0; j < numTasks; j++) {
                    receivedTasks[j].completed = 2;
                    taskList[oldSize + j] = receivedTasks[j];
                }
                stats.tasksReceived += numTasks;
                pthread_cond_broadcast(&tasksAvailable);  //отправляет сигнал всем потокам "" чтобы они продолжили выполнение
                pthread_mutex_unlock(&taskMutex);
                free(receivedTasks);
            }
        }
    }
    free(processRanks);
    return NULL;
}

void* responseThread(void* arg) {  //отвечает на запросы от других процессов
    if (!enableLoadBalancing) return NULL;
    while (!allTasksDone) {
        pthread_mutex_lock(&taskMutex);
        if (allTasksDone) {
            pthread_mutex_unlock(&taskMutex);
            break;
        }
        pthread_mutex_unlock(&taskMutex);

        MPI_Status status;
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int sourceRank = status.MPI_SOURCE;
            int requestCode;
            MPI_Recv(&requestCode, 1, MPI_INT, sourceRank, 0, MPI_COMM_WORLD, &status);
            pthread_mutex_lock(&taskMutex);
            int availableTasks = 0;
            for (int i = 0; i < taskListSize; i++) {
                if (!taskList[i].completed) availableTasks++;
            }
            if (availableTasks > 10) {
                int numTasksToSend = (availableTasks / 3) < 5 ? (availableTasks / 3) : 5;
                Task* tasksToSend = (Task*)malloc(numTasksToSend * sizeof(Task));
                int sent = 0;
                for (int i = 0; i < taskListSize && sent < numTasksToSend; i++) {
                    if (!taskList[i].completed) {
                        tasksToSend[sent] = taskList[i];
                        taskList[i].completed = 1;
                        sent++;
                    }
                }
                MPI_Send(&numTasksToSend, 1, MPI_INT, sourceRank, 1, MPI_COMM_WORLD);
                MPI_Send(tasksToSend, numTasksToSend, MPI_TASK_TYPE, sourceRank, 2, MPI_COMM_WORLD);
                int newSize = 0;
                for (int i = 0; i < taskListSize; i++) {
                    if (taskList[i].completed != 1) {
                        taskList[newSize++] = taskList[i];
                    }
                }
                taskListSize = newSize;
                stats.tasksSent += numTasksToSend;
                free(tasksToSend);
            } else {
                int zero = 0;
                MPI_Send(&zero, 1, MPI_INT, sourceRank, 1, MPI_COMM_WORLD);
            }
            pthread_mutex_unlock(&taskMutex);
        }
        usleep(10000);
    }
    return NULL;
}

void printFinalStats(void) {
    ProcessStats* allStats = (ProcessStats*)malloc(size * sizeof(ProcessStats));
    if (!allStats) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Gather(&stats, 1, MPI_STATS_TYPE, allStats, 1, MPI_STATS_TYPE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== Final Statistics ===\n");
        printf("Mode: %s\n", enableLoadBalancing ? "With Load Balancing" : "Without Load Balancing");
        printf("Distribution: %s\n", distributionCode == 0 ? "Uniform" : "Hill");
        printf("Rank | Tasks | Initial Weight | Total Weight | Own Tasks | Work Time | Idle Time | Received | Sent\n");
        printf("---------------------------------------------------------------------------------------------\n");

        double maxProcessTime = 0.0, minProcessTime = 1e9;
        for (int i = 0; i < size; i++) {
            if (allStats[i].workTime > maxProcessTime) maxProcessTime = allStats[i].workTime;
            if (allStats[i].workTime < minProcessTime) minProcessTime = allStats[i].workTime;
            double ownTasksPercentage = allStats[i].tasksProcessed == 0 ? 0.0 :
                ((double)allStats[i].own_tasks / allStats[i].tasksProcessed * 100);
            printf("%4d | %7d | %14d | %12d | %9d | %12.3f | %12.3f | %8d | %8d\n",
                   i, allStats[i].tasksProcessed, allStats[i].initial_weight, allStats[i].total_weight,
                   allStats[i].own_tasks, allStats[i].workTime, allStats[i].idleTime,
                   allStats[i].tasksReceived, allStats[i].tasksSent);
            printf("     | Percentage of own tasks: %.2f%%\n", ownTasksPercentage);
        }
        double loadImbalance = maxProcessTime / (minProcessTime > 0 ? minProcessTime : 1.0);
        printf("\nLoad Imbalance: %.3f\n", loadImbalance);
        if (enableLoadBalancing) {
            int totalTransferred = 0;
            for (int i = 0; i < size; i++) {
                totalTransferred += allStats[i].tasksSent;
            }
            printf("Total Tasks Transferred: %d\n", totalTransferred);
        }
        int totalTasksProcessed = 0;
        for (int i = 0; i < size; i++) {
            totalTasksProcessed += allStats[i].tasksProcessed;
        }
        printf("Total tasks processed: %d (Expected: %d)\n", totalTasksProcessed, N_TASKS * ITERATIONS);
    }
    free(allStats);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI implementation doesn't support MPI_THREAD_MULTIPLE\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    parseArguments(argc, argv);

    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint offsets[3];
    offsets[0] = offsetof(Task, repeatNum);
    offsets[1] = offsetof(Task, completed);
    offsets[2] = offsetof(Task, taskId);
    MPI_Type_create_struct(3, blocklengths, offsets, types, &MPI_TASK_TYPE);
    MPI_Type_commit(&MPI_TASK_TYPE);

    MPI_Datatype statsTypes[8] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    int statsBlocklengths[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    MPI_Aint statsOffsets[8];
    statsOffsets[0] = offsetof(ProcessStats, tasksProcessed);
    statsOffsets[1] = offsetof(ProcessStats, workTime);
    statsOffsets[2] = offsetof(ProcessStats, idleTime);
    statsOffsets[3] = offsetof(ProcessStats, tasksReceived);
    statsOffsets[4] = offsetof(ProcessStats, tasksSent);
    statsOffsets[5] = offsetof(ProcessStats, initial_weight);
    statsOffsets[6] = offsetof(ProcessStats, total_weight);
    statsOffsets[7] = offsetof(ProcessStats, own_tasks);
    MPI_Type_create_struct(8, statsBlocklengths, statsOffsets, statsTypes, &MPI_STATS_TYPE);
    MPI_Type_commit(&MPI_STATS_TYPE);

    pthread_mutex_init(&taskMutex, NULL);
    pthread_cond_init(&tasksAvailable, NULL);

    pthread_t worker, requester, responder;
    pthread_attr_t attrs;  //атрибут, определяет поведение потока
    pthread_attr_init(&attrs);
    pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);  //потоки будут создаваться в состоянии "joinable", не завершаются автоматически, а ждут вызова pthread_join

    if (pthread_create(&worker, &attrs, workerThread, NULL) != 0) {  //указатель, атрибут потока, сама функция, аргуметы функции (у меня не передается)
        perror("Cannot create worker thread");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (enableLoadBalancing) {
        if (pthread_create(&requester, &attrs, requestThread, NULL) != 0) {  
            perror("Cannot create requester thread");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (pthread_create(&responder, &attrs, responseThread, NULL) != 0) {
            perror("Cannot create responder thread");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    pthread_attr_destroy(&attrs);

    double totalStart = MPI_Wtime();
    int* numberOfTasksForProcesses = (int*)malloc(size * sizeof(int));
    if (rank == 0) {
        distributeTasks(distributionCode, numberOfTasksForProcesses, size);
    }
    MPI_Bcast(numberOfTasksForProcesses, size, MPI_INT, 0, MPI_COMM_WORLD);

    for (int iterCounter = 1; iterCounter <= ITERATIONS; iterCounter++) {
        double iterStart = MPI_Wtime();

        pthread_mutex_lock(&taskMutex);  //захватываю мьютекс для защиты общих данных
        taskListSize = numberOfTasksForProcesses[rank];
        stats.initial_weight = 0;
        if (taskListSize > taskListCapacity) {
            resizeTaskList(taskListSize * 2);
        }
        for (int i = 0; i < taskListSize; i++) {
            int weight = 15000 + abs(rank - (iterCounter % size)) * 500;
            taskList[i].repeatNum = weight;
            taskList[i].completed = 0;
            taskList[i].taskId = rank * 1000000 + iterCounter * 1000 + i;
            stats.initial_weight += weight;
        }
        pthread_cond_broadcast(&tasksAvailable);
        pthread_mutex_unlock(&taskMutex);  //освобождаю мьютекс

        while (1) {
            pthread_mutex_lock(&taskMutex);
            int localProcessed = stats.tasksProcessed;
            pthread_mutex_unlock(&taskMutex);

            int globalProcessed;
            MPI_Allreduce(&localProcessed, &globalProcessed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (globalProcessed >= N_TASKS * iterCounter) break;

            usleep(50000);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double iterTime = MPI_Wtime() - iterStart;
        if (rank == 0) {
            printf("Iteration %d completed in %.3f seconds (Mode: %s, Distribution: %s)\n",
                   iterCounter, iterTime, enableLoadBalancing ? "Balancing" : "No Balancing",
                   distributionCode == 0 ? "Uniform" : "Hill");
        }
    }

    pthread_mutex_lock(&taskMutex);
    allTasksDone = 1;
    pthread_cond_broadcast(&tasksAvailable);
    pthread_mutex_unlock(&taskMutex);

    pthread_join(worker, NULL);  //ждем завершения потока, блокируя поток-main
    if (enableLoadBalancing) {
        pthread_join(requester, NULL);
        pthread_join(responder, NULL);
    }

    double totalTime = MPI_Wtime() - totalStart;
    if (rank == 0) {
        printf("Total time: %.3f seconds\n", totalTime);
    }
    printFinalStats();

    free(numberOfTasksForProcesses);
    pthread_mutex_destroy(&taskMutex);
    pthread_cond_destroy(&tasksAvailable);
    if (taskList) free(taskList);
    MPI_Type_free(&MPI_TASK_TYPE);
    MPI_Type_free(&MPI_STATS_TYPE);
    MPI_Finalize();
    return 0;
}