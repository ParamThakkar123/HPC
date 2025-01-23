# Introduction to GPU Programming

In general, High Performance Computing pertains to the use of multiple processors or computers to accomplish a complex task concurrently with high throughput and efficiency.

## Parallel Computing
- The primary goal of parallel computing is to improve the speed of computation.

- Parallel computing can be defined as a form of computation in which many calculations are carried out simultaneously, operating on the principle that large problems can often be divided into smaller ones, which are then solve concurrently.

- Parallel computing involves two distinct areas of computing technologies
    - Computer Architecture (The hardware aspect)
    - Parallel Programming (The software aspect)

- Computer Architecture focuses on supporting parallelism at an architectural level, while parallel programming focuses on solving a problem concurrently by fully using the computational power of the computer architecture.

- Most modern processors implement the Harvard architecture which is comprised of three main components:
    - Memory (instruction memory and data memory)
    - Central Processing Unit (Control unit and arithmetic logic unit)
    - Input / Output interfaces

    ![alt text](image.png)

- The key component in high performance computing is the central processing unit, usually called the core.

- When there is only one core on the chip, the architecture is known as uniprocessor

- Programming can be viewed as the process of mapping the computation of a problem to available cores such that parallel execution is obtained.

## Sequential and Parallel Programming

- When solving a problem with a computer program, it is natural to divide the problem into discrete series of calculations, each calculation performs a specified task. Such a program is called a sequential program.

- There are two ways to classify the relationship between two pieces of computation:
    - Some are related by a precedence restraint and therefore must be calculated sequentially, others have no such restraints and can be calculated concurrently.


- When a computational problem is broken down into many small pieces of computation, each piece is called a task.

- A data dependency occurs when an instruction consumes data produced by a preceding instruction.

## Parallelism

- There are two fundamental types of parallelism in applications:
    - Task parallelism
    - Data parallelism

= Task parallelism arises when there are many tasks or functions that can be operated independently and largely in parallel.
- Task parallelism focuses on distributing functions across multiple cores.

- Data parallelism arises when there are many data items that can be operated on at the same time.
- Data parallelism focuses on distributing the data across multiple cores

- Data-parallel processing maps data elements to parallel threads.
- The first step in designing a data parallel program is to partition data across threads, with each thread working on a portion of the data.

- There are two approaches to partitioning data: block partitioning and cyclic partitioning.
- In block partitioning, many consecutive elements of data are chunked together. Each chunk is assigned to a single thread in any order, and threads generally process only one chunk at a time.
- In cyclic partitioning, fewer data elements are chunked together. Neighboring threads receive neighboring chunks, and each thread can handle more than one chunk. Selecting a new chunk for a thread to process implies jumping ahead as many chunks as there are threads.