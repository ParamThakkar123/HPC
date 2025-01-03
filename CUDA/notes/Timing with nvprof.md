
.1. nvprof is available to help collect timeline information from your application's CPU and GPU activity, including kernel execution, memory transfers, and CUDA API calls.

`nvprof [nvprof_args] <application> [application_args]`

In the output, the first half of the message contains output from the program, and the second half contains output from nvprof.

For HPC workloads, it is important to understand the compute to communication ratio in a program. If your applications spends more time computing than transferring data, then it may be possible to overlap these operations and completely hide the latency associated with transferring data, it is important to minimize the transfer between the host and the device.
