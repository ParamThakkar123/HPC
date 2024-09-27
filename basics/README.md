## CUDA full form
Compute Unified Device Architecture

## CUDA architecture 
- expose GPU for general purpose computing
- Retain performance

## Heterogenous Computing
1. Terminology
   - Host : The CPU and its memory (host memory)
   - Device : The GPU and its memory (device memory)
  
   ![image](https://github.com/user-attachments/assets/0d0a9782-127e-4647-b3f1-de1185b18b5a)

## Simple Processing Flow
1. Copy input data from CPU memory to GPU memory
2. Load GPU code and execute it, caching data on chip for performance
3. Copy results from GPU memory to CPU memory

NVIDIA CUDA compile (nvcc) can be used to compile programs with no device code.

## Syntax
1. CUDA C / C++ keyword __global__ indicates a function that:
   - Runs on the device
   - is called from the host code
2. nvcc separates source code into host and device components
   - Device functions processed by NVIDIA compiler
   - Host functions processed by standard host compiler - gcc, cl.exe
  
mykernel <<<1, 1>>>
- triple angle brackets mark a call from host code to device code
- Also called "kernel launch"
- functions written using the __global__ keyword are known as kernels in CUDA

## Memory Management in CUDA
Host and device memory are separate entities 
- Device pointers point to GPU memory
  - Maybe passed to / from host code
  - May not be dereferenced in the host code
- Host pointers point to CPU memory
  - May be passed to / from device code
  - May not be dereferenced in device code
 
CUDA API for handling device memory
- cudaMalloc(), cudaFree(), cudaMemcpy()
- similar to C equivalents malloc(), free(), memcpy()

## Executing in parallel

### Terminology: 
1. Block - each parallel invocation of a kernel is referred to as a block
2. Grid - set of blocks is referred to as a grid.
3. Each invocation can refer to its block index using `blockIdx.x`

By using blockIdx.x to index into the array, each block handles a different element of the array
