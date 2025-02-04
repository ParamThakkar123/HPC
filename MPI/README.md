# Running an MPI Program

To run an MPI (Message Passing Interface) program, follow these steps:

1. **Install MPI Library**:
    Ensure you have an MPI library installed, such as OpenMPI or MPICH.

    ```sh
    # For OpenMPI
    sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

    # For MPICH
    sudo apt-get install mpich
    ```

2. **Compile the MPI Program**:
    Use the `mpicc` compiler to compile your MPI program.

    ```sh
    mpicc -o my_mpi_program my_mpi_program.c
    ```

3. **Run the MPI Program**:
    Use the `mpirun` or `mpiexec` command to run your compiled MPI program.

    ```sh
    mpirun -np <number_of_processes> ./my_mpi_program
    ```

    Replace `<number_of_processes>` with the number of processes you want to run.

4. **Example**:
    If you have a program `hello_mpi.c`, compile and run it as follows:

    ```sh
    mpicc -o hello_mpi hello_mpi.c
    mpirun -np 4 ./hello_mpi
    ```

    This will run the `hello_mpi` program using 4 processes.

5. **Check MPI Installation**:
    Verify your MPI installation with:

    ```sh
    mpirun --version
    ```

    This should display the version of the MPI library installed.

6. **Common Options**:
    - `-np <number>`: Specifies the number of processes.
    - `-hostfile <file>`: Specifies a host file for distributed execution.
    - `-machinefile <file>`: Similar to `-hostfile`, specifies machines to use.

Make sure to consult the documentation of your specific MPI implementation for more details and advanced usage.
