#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <chrono>
using namespace std;

__global__ void matrixMul(int *a, int *b, int *c, int N){
    // Calculate the global row and column for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check for our matrix
    if(row < N && col < N){
        // Accumulate a partial result
        int tmp = 0;
        for(int i = 0; i < N; i++){
            tmp += a[row * N + i] * b[i * N + col];
        }

        // Write back the result
        c[row * N + col] = tmp;
    }
}

// Initializes a square matrix with random numbers between 0-100
void init_matrix(int *m, int N){
    for(int i = 0; i < N * N; i++){
        m[i] = rand() % 100;
    }
}

// verify the result on the CPU
void verify_result(int *a, int *b, int *c, int N){
    int tmp;
    // For every row
    for(int i = 0; i < N; i++){
        // For every column
        for(int j = 0; j < N; j++){
            // For every element in the row-col
            tmp = 0;
            for(int k = 0; k < N; k++){
                tmp += a[i * N + k] * b[k * N + j];
            }

            // Check each result
            assert(tmp == c[i * N + j]);
        }
    }
}

int main(){
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(int);

    int *a;
    int *b;
    int *c;
    // Allocate memory for matrices
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // Initialize our matrices
    init_matrix(a, N);
    init_matrix(b, N);

    // GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Set our Cooperative Thread Array and Grid dimensions
    int threads = 16;
    int blocks = (N + threads - 1) / threads; // Padding trick

    // Set up our kernel launch parameters
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    matrixMul<<<BLOCKS, THREADS>>>(a, b, c, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // CPU timing
    auto cpu_start = chrono::high_resolution_clock::now();
    // Verify the result
    verify_result(a, b, c, N);
    auto cpu_stop = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> cpu_duration = cpu_stop - cpu_start;

    // Print the results
    cout << "GPU Time: " << gpu_time_ms << " ms" << endl;
    cout << "CPU Time: " << cpu_duration.count() << " ms" << endl;
    
    cout << "PROGRAM COMPLETED SUCCESSFULLY!" << endl;
}