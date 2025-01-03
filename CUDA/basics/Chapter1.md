
- General purpose parallel computing platorm and programming model that leverages the parallel compute engine in NVIDIA GPUs to solve many complex computational problems in a more efficient way.
- An extension of the standard ANSI C.
- A scalable programming model that enables programs to transparently scale their parallelism to GPUs with varying numbers of cores.
- Provides two API levels for managing the GPU device and organizing threads:
	- CUDA driver API
	- CUDA runtime API
- The driver API is a low level API and is relatively hard to program, but provides more control over how the GPU device is used
- The runtime API is a higher level API implemented on top of the driver API. Each function of the runtime API is broken down into more basic operations issued to the driver API.
- A CUDA program consists of a mixture of the following two parts: 
	- The host code runs on CPU
	- The device code runs on GPU
- CUDA nvcc compiler separates the device code from the host code during the compilation process.
- Device code is written using CUDA C extended with keywords for labeling data-parallel functions, called kernels.
- The device code is further compiled by nvcc
- nvcc compiler is based on the widely used LLVM open source compiler infrastructure

- `__global__` tells the compiler that the function will be called from the CPU and executed in the GPU
- Triple angle brackets `<<< >>>` mark a call from the host thread to the code on the device side.
- A kernel is executed by an array of threads and all threads run the same code.
- The parameters within the triple angle brackets are the execution configuration, which specifies how many threads will execute the kernel
- `cudaDeviceReset()` will explicitly destroy and cleanup all resources associated with the current device in the current process.

### CUDA Program Structure

- Consists of five main steps
	- Allocate GPU memories
	- Copy data from CPU memory to GPU memory
	- Invoke the CUDA kernel to perform program-specific computation
	- Copy data back from GPU memory to CPU memory
	- Destroy GPU memories

Locality refers to the reuse of data so as to reduce memory access latency.

There are two basic types of reference locality.
- Temporal locality: refers to reuse of data and/or resources within relatively small time durations
- Spatial locality: refers to the use of data elements within relatively close storage locations.

- CUDA abstracts away the hardware details and does not require applications to be mapped to traditional graphics APIs.
- At its core are three key abstractions: a hierarchy of thread groups, a hierarchy of memory groups, and barrier synchronization.