#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda.h>

#define N 1024  // Matrix size N x N

__global__ void matrixTransposeKernel(float* input, float* output, int n) {
    __shared__ float tile[16][16];  // Shared memory for tiles

    // Global indices for reading from input
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (row < n && col < n) {
        tile[threadIdx.y][threadIdx.x] = input[row * n + col];
    }
    __syncthreads();

    // Global indices for writing to output (transposed)
    row = blockIdx.x * blockDim.y + threadIdx.y;  // Swap blockIdx.x and blockIdx.y
    col = blockIdx.y * blockDim.x + threadIdx.x;

    // Write transposed tile to output
    if (row < n && col < n) {
        output[row * n + col] = tile[threadIdx.x][threadIdx.y];  // Transpose within tile
    }
}

void cpuMatrixTranspose(float* input, float* output, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            output[j * n + i] = input[i * n + j];
}

int main() {
    size_t bytes = N * N * sizeof(float);

    // Allocate pinned memory for host matrices
    float* h_input;
    float* h_output_cpu;
    float* h_output_gpu;
    cudaMallocHost(&h_input, bytes);
    cudaMallocHost(&h_output_cpu, bytes);
    cudaMallocHost(&h_output_gpu, bytes);

    // Initialize input matrix
    for (int i = 0; i < N * N; ++i)
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;

    // CPU Transpose
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpuMatrixTranspose(h_input, h_output_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // GPU Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixTransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop);

    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // Validate correctness
    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (abs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    // Print results
    std::cout << "Matrix size: " << N << " x " << N << "\n";
    std::cout << "CPU time: " << cpu_time.count() << " ms\n";
    std::cout << "GPU time: " << gpu_time << " ms\n";
    std::cout << "Result: " << (correct ? "Success ✅" : "Mismatch ❌") << "\n";

    // Free memory
    cudaFreeHost(h_input);
    cudaFreeHost(h_output_cpu);
    cudaFreeHost(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}