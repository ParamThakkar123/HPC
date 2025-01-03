
1. Programming models present an abstraction of computer architectures that act as a bridge between an application and its implementation on available hardware.
2. The communication abstraction is the boundary between program and the programming model implementation, which is realized through a compiler or libraries using privileged hardware primitives and the operating system.
![[Pasted image 20241222000806.png]]

3. The program, written for a programming model, dictates how components of the program share information and coordinate their activities.
4. The programming model provides a logical view of specific computing architectures.
5. It is embodied in a programming language or programming environment.
6. The CUDA programming model provides the following special features to harness the computing power of GPU architectures.
	1. A way to organize threads on the GPU through a hierarchy structure
	2. A way to access memory on the GPU through a hierarchy structure.
7. We can view parallel computation from different levels, such as :
	1. Domain level: How to decompose data and functions so as to solve the problem correctly and efficiently while running in a parallel environment.
	2. Logic level
	3. Hardware level
8. CUDA exposes a thread hierarchy abstraction to allow you to control thread behavior.

## CUDA Programming Structure

1. The CUDA programming model enables to execute applications on heterogeneous computing systems by simply annotating code with a small set of extensions to the C programming language.
2. A heterogeneous environment consists of CPUs complemented by GPUs, each with its own memory separated by a PCI-Express bus.
3. A key component of the CUDA programming model is the kernel - the code that runs on the GPU device.
4. CUDA manages scheduling programmer written kernels on GPU threads.

## Managing Memory

1. The CUDA programming model assumes a system composed of a host and a device, each with its own separate memory.
2. Kernels operate out of device memory.
3. The CUDA runtime provides functions to allocate device memory, release device memory, and transfer data between the host memory and device memory.
![[Pasted image 20241222002244.png]]

4. The function used to perform GPU memory allocation is cudaMalloc, and its function signature is :
	`cudaError_t cudaMalloc (void** devPtr, size_t size)`
5. This function allocates a linear range of device memory with the specified size in bytes.
6. The allocated memory is returned through devPtr.
7. The function used to transfer data between the host and the device is: cudaMemcpy, and its function signature is:
`cudaError_t cudaMemcpy (void* dst, const void* src, size_t count, cudaMemcpyKind kind)`

8. This function copies the specifies bytes from the source memory area, pointed to by src, to the destination memory area, pointed to by dst, with the direction specified by kind.
9. kind takes one of the following types:
	1. cudaMemcpyHostToHost
	2. cudaMemcpyHostToDevice
	3. cudaMemcpyDeviceToHost
	4. cudaMemcpyDeviceToDevice
10. This function exhibits synchronous behavior because the host application blocks until cudaMemcpy returns and the transfer is complete.
11. Every CUDA call, except kernel launches, returns an error code of an enumerated type cudaError_t.
12. You can convert an error code to a human-readable error message with the following CUDA run time function: char* cudaGetErrorString(cudaError_t error)
13. The cudaGetErrorString function is analogous to the Standard C strerror function.

In the GPU memory hierarchy, the two most important types of memory are global memory and shared memory.

Global memory is analogous to CPU system memory, while shared memory. Global memory is analogous to CPU system memory, while shared memory is similar to the CPU cache. However, GPU shared memory can be directly controlled from a CUDA C kernel.

## Organizing Threads

1. When a kernel function is launched from the host side, execution is moved to a device where a large number of threads are generated and each thread executes the statements specified by the kernel function.
2. All threads spawned by a single kernel launch are collectively called a grid. All threads in a grid share the same global memory space.
3. A grid is made up of many thread blocks. A thread block is a group of threads that can cooperate with each other using:
	1. Block local synchronization
	2. Block local shared memory
4. Thread from different blocks cannot cooperate
5. Threads rely on the following two unique coordinates to distinguish themselves from each other : 
	1. blockIdx (block index within grid)
	2. threadIdx (thread index within a block)
6. These variables appear as built-in, pre-initialized variables that can be accessed within kernel functions.
7. When a kernel function is executed, the coordinate variables blockIdx and threadIdx are assigned to each thread by the CUDA runtime. Based on the coordinates, you can assign portions of data to different threads.
8. The coordinate variable is of type uint3, a CUDA built-in vector type, derived from the basic inte ger type. It is a structure containing three unsigned integers, and the 1st, 2nd, and 3rd components are accessible through the fields x, y, and z respectively.
9. CUDA organizes grids and blocks in three dimensions.
10. The dimensions of a grid and a block are specifi ed by the following two built-in variables:
	1. blockDim (block dimension, measured in threads)
	2. gridDim (grid dimension, measured in blocks)
11. These variables are of type dim3, an integer vector type based on uint3 that is used to specify dimensions.
12. When defining a variable of type dim3, any component left unspecified is initialized to 1. Each component in a variable of type dim3 is accessible through its x, y, and z fields.
13. There are two distinct sets of grid and block variables in a CUDA program: manually-defined dim3 data type and pre-defined uint3 data type.
14. On the host side, you define the dimensions of a grid and block using a dim3 data type as part of a kernel invocation
15. When the kernel is executing, the CUDA runtime generates the corresponding built-in, pre-initialized grid, block, and thread variables, which are accessible within the kernel function and have type uint3.
16. The manually-defined grid and block variables for the dim3 data type are only visible on the host side, and the built-in, pre-initialized grid and block variables of the uint3 data type are only visible on the device side.
17. For a given data size, the general steps to determine the grid and block dimensions are:
	1. Decide the block size.
	2. Calculate the grid dimension based on the application data size and the block size.
18. To determine the block dimension, you usually need to consider:
	1. Performance characteristics of the kernel
	2. Limitations on GPU resources
19. A CUDA kernel call is a direct extension to the C function syntax that adds a kernel’s execution confi guration inside triple-angle-brackets:
`kernel_name <<<grid, block>>>(argument list)`