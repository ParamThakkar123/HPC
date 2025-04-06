# Introduction to MPI

## What is MPI ?

- First message-passing interface standard
- MPI is library of function / subroutine calls
- MPI is not a language
- There is no such things as an MPI compiler

## Goals of MPI

- To provide source-code portability
- To allow efficient implementation

It also offers:
- A great deal of functionality
- Support for heterogeneous parallel architectures

## Header files for C

```
#include <mpi.h>
```

## MPI Function format
```
error = MPI_Xxxxxx(parameter, ....);
MPI_Xxxxxx(parameter, ....);
```

- MPI controls its own internal data structures
- MPI releases 'handles' to allow programmers to refer to these
- C handles are of defined `typedefs`
- Fortran handles are `INTEGERs`

## Initialising MPI

```
int MPI_Init(int *argc, char **argv)
```

- Must be the first MPI procedure called:
    - but multiple processes are already running before MPI_Init

## MPI_Init for various cases

```
int main(int argc, char *argv[]){
    ...
    MPI_Init(&argc, &argv);
    ...
}

int main(){
    ...
    MPI_Init(NULL, NULL);
    ...
}
```

## Communicators

1. MPI_Comm_World: a communicator provided by MPI, it contains all MPI processes.

### How do you identify different processes in a communicator ?

We identify different processes in a communicator by rank using MPI_Comm_rank

`MPI_Comm_rank(MPI_Comm comm, int *rank)`

`MPI_Comm_rank` is a function in the Message Passing Interface (MPI) used to determine the rank (ID) of a process within a given communicator.

The rank is not the physical processor number.
- numbering is always 0, 1, 2, ..., N - 1

```
int rank;
...
MPI_Comm_rank(MPI_Comm_World, &rank);
printf("Hello from rank %d\n", rank);
```

### How many processes are contained within a communicator ?

The size of the communicator can be found out using `MPI_Comm_size`.

`MPI_Comm_size(MPI_Comm comm, int *size)`

## Exiting MPI

`int MPI_Finalize()`

- Must be the last MPI procedure called.

## Aborting MPI

Aborting the execution from any processor

`int MPI_Abort(MPI_Comm comm, int errorcode)`

Behaviour:
- Will abort all processes even if only called by one process
- This is the only MPI routine that can have this effect
- Only use as a fast-resort "nuclear" option!