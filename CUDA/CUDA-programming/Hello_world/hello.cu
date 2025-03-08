#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void helloFromGPU(){
    printf("Hello, World from GPU thread %d!\n", threadIdx.x);
}

int main(){
    cout << "Hello, World from CPU!" << endl;
    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}