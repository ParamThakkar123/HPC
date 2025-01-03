
1. A kernel function is the code to be executed on the device side.
2. In a kernel function, you define the computation for a single thread, and the data access for that thread.
3. When the kernel is called, many different CUDA threads perform the same computation in parallel.
4. A kernel is defined using the `__global__` declaration specification :
	1. `__global__ void kernel_name(argument list);`
5. A kernel function must have a `void` type
6. Function type qualifiers specify whether a function executes on the host or on the device and whether it is callable from the host or from the device.

![[Pasted image 20241224005830.png]]

7. The `__device__` and `__host__`qualifiers can be used together, in which case the function is compiled for both the host and the device.
8. The following restrictions apply for all kernels:
	1. Access to device memory only
	2. Must have void return type
	3. No support for a variable number of arguments
	4. No support for static variables
	5. No support for function pointers
	6. Exhibit an asynchronous behavior
9. Besides many useful debugging tools, there are two very basic but useful means by which you can verify your kernel code. First, you can use printf in your kernel for Fermi and later generation devices. Second, you can set the execution confi guration to <<<1,1>>>, so you force the kernel to run with only one block and one thread. This emulates a sequential implementation. This is useful for debugging and verifying correct results. Also, this helps you verify that numeric results are bitwise exact from run-to-run if you encounter order of operations issues.

![[Pasted image 20241224010217.png]]

