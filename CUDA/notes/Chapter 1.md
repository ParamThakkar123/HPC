- High Performance Computing Pertains to the use of multiple processes or computer to accomplish a complex task concurrently with high throughput and efficiency.

### Parallel Computing

- The primary goal of parallel computing is to improve the speed of computation
- Parallel computing can be defined as a form of computation in which many calculations are carried out simultaneously, operating on the principle that large problems can often be divided into smaller ones, which are then solve concurrently.
- Parallel computing usually involves two distinct areas of computing technologies:
	- Computer architecture (hardware aspect)
	- Parallel Programming (software aspect)
- Computer architecture focuses on supporting parallelism at an architectural level, while parallel programming focuses on solving a problem concurrently by fully using the computational power of the computer architecture.
- Most modern processor implement the Harvard architecture:
	- Memory (instruction memory and data memory)
	- Central processing unit (control unit and arithmetic logic unit)
	- Input / Output interfaces
![[Pasted image 20241220235114.png]]
- The key component in high performance computing is the central processing unit, usually called the core.
- A processor with a single core on the chip is referred to as a uniprocessor
- When multiple cores are integrated on a single processor, it is usually termed as multicore.

### Sequential and Parallel Programming

- When solving a problem with a computer program, it is natural to divide the problem into a discrete series of calculations, each calculation performs a specified task, such a program is called a sequential program.
- When a computational problem is broken down into many small pieces of computation, each piece is called a task.
- In a task, individual instructions consume inputs, apply a function and produce outputs
- A data dependency occurs when an instruction consumes data produces by a preceding instruction.
### Parallelism

- There are two fundamental types of parallelism in applications :
	- Task Parallelism
	- Data Parallelism
- Task parallelism arises when there are many tasks or functions that can be operated independently and largely in parallel. Task parallelism focuses on distributing functions across multiple cores.
- Data Parallelism arises when there are many data items that can be operated on at the same time. Data parallelism focuses on distributing the data across multiple cores
- CUDA programming is especially well suited to address problems that can be expressed as data-parallel computations.
- Data parallel processing maps data elements to parallel threads.
- The first step in designing a data parallel program is to partition data across threads, with each thread working on a portion of the data.
- In general, there are two approaches to partitioning data:
	- Block Partitioning
	- Cyclic Partitioning
- In block partitioning, many consecutive elements of data are chunked together. Each chunk is assigned to a single thread in any order, and threads generally process only one chunk at a time.
- In chunk partitioning, fewer data elements are chunked together. Neighboring threads receive neighboring chunks, and each thread can handle more than one chunk. Selecting a new chunk for a thread to process implies jumping ahead as many chunks as there are threads
![[Pasted image 20241221000836.png]]

### Computer Architecture

1. Flynn's taxonomy classifies architectures into four different types according to how instructions and data flow through cores, including:
	1. Single Instruction Single Data (SISD): 
		- refers to the traditional computer: a serial architecture. There is only one core in the computer. 
		- At any time only one instruction stream is executed, and operations are performed on one data stream.
	1. Single Instruction Multiple Data (SIMD):
		- Parallel Architecture
		- There are multiple cores in the computer.
		- All cores execute the same instruction stream at any time, each operating on different data streams.
	1. Multiple Instruction Single Data (MISD):
		- Refers to an uncommon architecture where each core operates on the same data stream via separate instruction streams
	2. Multiple Instruction Multiple Data (MIMD):
		- Parallel architecture 
		- Multiple cores operate on multiple data streams, each executing independent instructions
		- Many MIMD architectures also include SIMD execution sub components
![[Pasted image 20241221001038.png]]

Latency: time it takes for an operation to start and complete, commonly expressed in microseconds

Bandwidth: The amount of data that can be processed per unit of time, commonly expressed as megabytes/sec or gigabytes/sec.

Throughput: The amount of operations that can be processed per unit of time, commonly expressed as gflops

Computer architecture can also be subdivided by their memory organization:
- Multi-node with distributed memory: 
	- large scale computational engines are constructed from many processors connected by a network
	- Each processor has its own local memory, and processors can communicate the contents of their local memory over the network
	- These systems are referred to as clusters
- Multiprocessor with shared memory


There are two important features that describe GPU capability
- Number of CUDA cores
- Memory size

There are two different metrics for describing GPU performance
- Peak computational performance
- Memory bandwidth

Peak computational performance is a measure of computational capability, usually defined as how many single-precision or double precision floating point calculations can be processed per second. Usually expressed in gflops

Memory bandwidth is a measure of the ratio at which data can be read from or stored to memory. Memory bandwidth is a measure of the ratio at which data can be read from or stored to memory. Usually expressed in gigabytes per second.

Homogeneous computing uses one or more processor of the same architecture to execute an application.

Heterogeneous computing instead uses a suite of processor architectures to execute an application, applying tasks to architectures to which they are well suited, yielding performance improvement as a result.

## Heterogeneous Architecture

1. Consists of two multicore CPU sockets and two or more many-core GPUs.
2. GPUs must operate in conjunction with a CPU based host through a PCI-Express bus.
3. The CPU is therefore called the host and the GPU is called the device.
4. A heterogenous application consists of two parts: Host code, Device code.
5. Host code runs on CPUs and device code runs on GPUs. 
