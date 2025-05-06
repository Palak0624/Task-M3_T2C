#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <CL/cl.h>

#define N 1024       // Matrix size (N x N)
#define LOCAL_SIZE 16  // OpenCL work-group size

void initialize_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (double)rand() / RAND_MAX;
        }
    }
}

void matrix_mult_sequential(double *A, double *B, double *C, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < cols; k++) {
                sum += A[i * cols + k] * B[k * cols + j];
            }
            C[i * cols + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (N % num_procs != 0) {
        if (rank == 0) {
            printf("Matrix size must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *kernelSource =
        "_kernel void matrix_mult(_global double *A, __global double *B, __global double *C, int width) {\n"
        "    int row = get_global_id(0);\n"
        "    int col = get_global_id(1);\n"
        "    double sum = 0.0;\n"
        "    for (int k = 0; k < width; k++) {\n"
        "        sum += A[row * width + k] * B[k * width + col];\n"
        "    }\n"
        "    C[row * width + col] = sum;\n"
        "}\n";

    int rows_per_process = N / num_procs;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = (double *)malloc(rows_per_process * N * sizeof(double));
    double *local_C = (double *)malloc(rows_per_process * N * sizeof(double));

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        B = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));

        srand(time(NULL));
        initialize_matrix(A, N);
        initialize_matrix(B, N);

        start_time = MPI_Wtime();
    } else {
        B = (double *)malloc(N * N * sizeof(double));
    }

    MPI_Scatter(A, rows_per_process * N, MPI_DOUBLE,
                local_A, rows_per_process * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem bufA = NULL, bufB = NULL, bufC = NULL;
    cl_int err;
    int use_opencl = 1;

    // Try to initialize OpenCL
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        if (rank == 0) {
            printf("started");
        }
        use_opencl = 0;
    }

    if (use_opencl) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            if (rank == 0) printf("success");
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
            if (err != CL_SUCCESS) {
                if (rank == 0) printf("No OpenCL devices found, falling back to sequential\n");
                use_opencl = 0;
            }
        }
    }

    if (use_opencl) {
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            if (rank == 0) printf("Error creating context, falling back to sequential\n");
            use_opencl = 0;
        }
    }

    if (use_opencl) {
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, 0, 0};
        queue = clCreateCommandQueueWithProperties(context, device, props, &err);
        if (err != CL_SUCCESS) {
            if (rank == 0) printf("Error creating command queue, falling back to sequential\n");
            use_opencl = 0;
        }
    }

    if (use_opencl) {
        program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
        if (err != CL_SUCCESS) {
            if (rank == 0) printf("Error creating program, falling back to sequential\n");
            use_opencl = 0;
        }
    }

    if (use_opencl) {
        err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            if (rank == 0) {
                printf("Error building program, falling back to sequential\n");
                size_t log_size;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
                char *log = (char *)malloc(log_size);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
                printf("Build log:\n%s\n", log);
                free(log);
            }
            use_opencl = 0;
        }
    }

    if (use_opencl) {
        kernel = clCreateKernel(program, "matrix_mult", &err);
        if (err != CL_SUCCESS) {
            if (rank == 0) printf("Error creating kernel, falling back to sequential\n");
            use_opencl = 0;
        }
    }

    if (use_opencl) {
        bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                rows_per_process * N * sizeof(double), local_A, &err);
        bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                N * N * sizeof(double), B, &err);
        bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                rows_per_process * N * sizeof(double), NULL, &err);
        if (err != CL_SUCCESS) {
            if (rank == 0) printf("Error creating buffers, falling back to sequential\n");
            use_opencl = 0;
        }
    }

    if (use_opencl) {
        int width = N;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
        err |= clSetKernelArg(kernel, 3, sizeof(int), &width);
        if (err != CL_SUCCESS) {
            if (rank == 0) printf("Error setting kernel args, falling back to sequential\n");
            use_opencl = 0;
        }
    }

    if (use_opencl) {
        size_t globalSize[2] = {rows_per_process, N};
        size_t localSize[2] = {LOCAL_SIZE, LOCAL_SIZE};
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            if (rank == 0) printf("Error executing kernel, falling back to sequential\n");
            use_opencl = 0;
        }
    }

    if (use_opencl) {
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                                        rows_per_process * N * sizeof(double), local_C, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            if (rank == 0) printf("Error reading results, falling back to sequential\n");
            use_opencl = 0;
        }
    }

    if (!use_opencl) {
        // Fall back to sequential multiplication
        matrix_mult_sequential(local_A, B, local_C, rows_per_process, N);
    }

    // Clean up OpenCL resources if they were created
    if (bufA) clReleaseMemObject(bufA);
    if (bufB) clReleaseMemObject(bufB);
    if (bufC) clReleaseMemObject(bufC);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);

    MPI_Gather(local_C, rows_per_process * N, MPI_DOUBLE,
                C, rows_per_process * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        printf("Matrix Multiplication Time: %.4f seconds\n", end_time - start_time);
        printf("Used %s\n", use_opencl ? "OpenCL" : "sequential fallback");
        free(A);
        free(C);
    }

    free(local_A);
    free(local_C);
    free(B);

    MPI_Finalize();
    return 0;
}
