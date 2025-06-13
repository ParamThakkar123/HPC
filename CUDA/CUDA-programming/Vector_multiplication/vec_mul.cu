#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void vectorMultiplyGPU(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

void vectorMultiplyCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

void initializeVector(float *vec, int n, float start) {
    for (int i = 0; i < n; i++) {
        vec[i] = start + i;
    }
}

bool verifyResults(float *cpu, float *gpu, int n) {
    for (int i = 0; i < n; i++) {
        if (abs(cpu[i] - gpu[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

int main() {
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    cudaMallocHost(&h_a, size);
    cudaMallocHost(&h_b, size);
    cudaMallocHost(&h_c_cpu, size);
    cudaMallocHost(&h_c_gpu, size);

    initializeVector(h_a, n, 1.0f);
    initializeVector(h_b, n, 2.0f);

    clock_t start_cpu = clock();
    vectorMultiplyCPU(h_a, h_b, h_c_cpu, n);
    clock_t end_cpu = clock();
    double cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorMultiplyGPU<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    bool correct = verifyResults(h_c_cpu, h_c_gpu, n);

    printf("Vector size: %d\n", n);
    printf("CPU execution time: %.6f seconds\n", cpu_time);
    printf("GPU execution time: %.3f ms (%.6f seconds)\n", gpu_time_ms / 1000);
    printf("Speedup: %.2fx\n", cpu_time / (gpu_time_ms / 1000));
    printf("Results %s\n", correct ? "match!" : "don't match!");

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c_cpu);
    cudaFreeHost(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}