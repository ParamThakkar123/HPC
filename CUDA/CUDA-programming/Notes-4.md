## Memory Hierarchy

- CUDA threads may access data from multiple memory spaces during their execution
- Each thread has private local memory.
- Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block.
- Thread blocks in a thread block can perform read, write and atomics operations on each other's shared memory. All threads have access to the same global memory.

- There are also two additional read-only memory spaces accessible by all threads : the constant and texture memory spaces.
- The global, constant and texture memory spaces are optimized for different memory spaces.
- Texture memory also offers different addressing modes, as well as data filtering, for some specific data formats

- The global, constant and texture memory spaces are persistent across kernel launches by the same application

![Memory Hierarchy](image-3.png)

## Heterogeneous Programming

Assumptions of CUDA programming model : 

- CUDA threads execute on a physically separate device that operates as a coprocessor to the host running the C++ program.
- Both the host and the device maintain their own separate memory spaces in DRAM, referred to as host memory and device memory respectively.


- A program manages the global, constant, and texture memory spaces visible to kernels through calls to the CUDA runtime. This includes device memory allocation and deallocation as well as data transfer between host and device memory.

- Unified Memory provides managed memory to bridge the host and the device memory spaces.
- Managed memory is accessible from all CPUs and GPUs in the system as a single, coherent memory image with a common address space.
- This capability enables oversubscription of device memory and can greatly simplify the task of porting applications by eliminating the need to explicitlly mirror data on host and device

## Asynchronous SIMT Programming Model
- The CUDA programming model provides acceleration to memory operations via the asynchronous programming model.
- The asynchronous programming model defines the behavior of asynchronous operations with respect to CUDA threads.
- The model also explains and defines how cuda::memcpy_async can be used to move data asynchronously from global memory while computing in the GPU.

## Asynchronous Operations

- An asynchronous operation is defined as an operation that is initiated by a CUDA thread and is executed asynchronously as-if by another thread.

- In a well formed program one or more CUDA threads synchronize with the asynchronous operation. The CUDA thread that initiated the asynchronous operation is not required to be among the synchronizing threads.

- Such an asynchronous thread is always associated with the CUDA thread that initiated the asynchronous operation. 

- An asynchronous operation uses a synchronization object to synchronize the completion of the operation.

- Such a synchronization object can be explicitly managed by a user or implicitly managed within a library.

- A synchronization object could be a `cuda::barrier` or a `cuda::pipeline`

- These synchronization objects can be used at different thread scopes. A scope defines the set of threads that may use the synchronization object to synchronize with asynchronous operation.

### Thread Scopes in CUDA C++

- `cuda::thread_scope::thread_scope_thread` : Only the CUDA thread which initiated asynchronous operations synchronizes

- `cuda::thread_scope::thread_scope_block` : All or any CUDA threads within the same thread block as the initiating thread synchronizes.

- `cuda::thread_scope::thread_scope_device` : All or any CUDA threads in the same GPU device as the initiating thread synchronizes.

- `cuda::thread_scope::thread_scope_system` : All or any CUDA or CPU threads in the same system as the initiating thread synchronizes.

## Compute Capability

- The compute capability of a device is represented by a version number, also sometimes called its "SM version". 

- This version number identifies the features supported by the GPU hardware and is used by applications at runtime to determine which hardware features and/or instructions are available on the present GPU.

- The compute capability comprises a major revision number X and a minor revision number Y and is denoted by X.Y.

- Devices with the same major revision number are of the same core architecture,

- The minor revision number corresponds to an incremental improvement to the core architecture, possibly including new featurs.