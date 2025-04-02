#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <time.h> 

#define NUM_SAMPLES 1000000000

int main(int argc, char *argv[]){
    int rank, size;
    long long local_count = 0, global_count = 0;
    long long samples_per_process;
    double x, y;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    samples_per_process = NUM_SAMPLES / size;

    unsigned int seed = rank + time(NULL);

    #pragma omp parallel num_threads(4) private(x, y) 
    {
        unsigned int local_seed = seed + omp_get_thread_num();
        long long thread_count = 0;

        #pragma omp for reduction(+:local_count) 
        for(int i = 0; i < samples_per_process; i++){
            x = (double) rand_r(&local_seed) / RAND_MAX * 2.0 - 1.0;
            y = (double) rand_r(&local_seed) / RAND_MAX * 2.0 - 1.0;

            if(x * x + y * y <= 1){
                thread_count++;
            }
        }

        #pragma omp atomic
        local_count += thread_count;
    }

    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        double pi = 4 * (double) global_count / NUM_SAMPLES;
        printf("Approximation of pi: %f\n", pi);
    }

    MPI_Finalize();
    return 0;
}