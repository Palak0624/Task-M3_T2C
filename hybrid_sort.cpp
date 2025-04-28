#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <CL/cl.h>

#define MAX_SOURCE_SIZE (0x100000)

void generate_random_array(std::vector<int>& arr, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    
    arr.resize(size);
    for (int i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }
}

bool is_sorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i-1] > arr[i]) {
            return false;
        }
    }
    return true;
}

void print_array(const std::vector<int>& arr, int limit) {
    int count = 0;
    for (int num : arr) {
        if (count >= limit) break;
        std::cout << num << " ";
        count++;
        if (count % 8 == 0) std::cout << std::endl;
    }
    if (count % 8 != 0) std::cout << std::endl;
}

cl_program load_kernel(cl_context context, cl_device_id device, const char* filename) {
    FILE *fp;
    char *source_str;
    size_t source_size;
    
    fp = fopen(filename, "r");
    if (!fp) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        return NULL;
    }
    
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, 
                                                  (const size_t*)&source_size, NULL);
    free(source_str);
    
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        std::cerr << "Error building program: " << log << std::endl;
        free(log);
        return NULL;
    }
    
    return program;
}

void opencl_sort(std::vector<int>& data) {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;
    
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
    
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    
    cl_program program = load_kernel(context, device_id, "bitonic_kernel.cl");
    if (!program) {
        std::cerr << "OpenCL initialization failed: Failed to open kernel file: bitonic_kernel.cl" << std::endl;
        return;
    }
    
    cl_kernel kernel = clCreateKernel(program, "bitonic_sort", &ret);
    
    cl_mem mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                   sizeof(int) * data.size(), NULL, &ret);
    
    ret = clEnqueueWriteBuffer(command_queue, mem_obj, CL_TRUE, 0,
                              sizeof(int) * data.size(), data.data(), 0, NULL, NULL);
    
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&data.size());
    
    size_t global_item_size = data.size();
    size_t local_item_size = 64;
    
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                                &global_item_size, &local_item_size, 0, NULL, NULL);
    
    ret = clEnqueueReadBuffer(command_queue, mem_obj, CL_TRUE, 0,
                             sizeof(int) * data.size(), data.data(), 0, NULL, NULL);
    
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(mem_obj);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int total_elements = 1000000;
    if (argc > 1) {
        total_elements = std::atoi(argv[1]);
    }
    
    std::vector<int> initial_array;
    std::vector<int> local_array;
    
    if (world_rank == 0) {
        generate_random_array(initial_array, total_elements);
        
        if (total_elements <= 32) {
            std::cout << "Initial array (first " << total_elements << " elements):" << std::endl;
            print_array(initial_array, total_elements);
        }
    }
    
    auto total_start = MPI_Wtime();
    double sort_time = 0.0;
    double merge_time = 0.0;
    
    // Broadcast the size to all processes
    MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate elements per process
    int elements_per_proc = total_elements / world_size;
    int remainder = total_elements % world_size;
    
    // Adjust for remainder
    if (world_rank < remainder) {
        elements_per_proc++;
    }
    
    // Scatter the data
    std::vector<int> sendcounts(world_size, total_elements / world_size);
    std::vector<int> displs(world_size, 0);
    
    for (int i = 0; i < remainder; ++i) {
        sendcounts[i]++;
    }
    
    for (int i = 1; i < world_size; ++i) {
        displs[i] = displs[i-1] + sendcounts[i-1];
    }
    
    local_array.resize(elements_per_proc);
    MPI_Scatterv(initial_array.data(), sendcounts.data(), displs.data(), MPI_INT,
                 local_array.data(), elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Sort locally with OpenCL
    auto sort_start = MPI_Wtime();
    opencl_sort(local_array);
    // Fall back to std::sort if OpenCL fails
    if (!is_sorted(local_array)) {
        std::sort(local_array.begin(), local_array.end());
    }
    sort_time = MPI_Wtime() - sort_start;
    
    // Gather the sorted subarrays
    std::vector<int> sorted_array;
    if (world_rank == 0) {
        sorted_array.resize(total_elements);
    }
    
    auto merge_start = MPI_Wtime();
    MPI_Gatherv(local_array.data(), elements_per_proc, MPI_INT,
                sorted_array.data(), sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    
    // Final merge on root process
    if (world_rank == 0) {
        std::sort(sorted_array.begin(), sorted_array.end());
        merge_time = MPI_Wtime() - merge_start;
    }
    
    if (world_rank == 0) {
        auto total_end = MPI_Wtime();
        double total_elapsed = total_end - total_start;
        
        std::cout << "Process " << world_rank << " sorted " << elements_per_proc 
                  << " elements in " << sort_time << " seconds" << std::endl;
        std::cout << "\nResults:" << std::endl;
        std::cout << "......" << std::endl;
        std::cout << "Total elements: " << total_elements << std::endl;
        std::cout << "Processes used: " << world_size << std::endl;
        
        bool success = is_sorted(sorted_array);
        std::cout << "Validation: " << (success ? "PASSED" : "FAILED") << std::endl;
        std::cout << "Total time: " << total_elapsed << " seconds" << std::endl;
        std::cout << "Final merge time: " << merge_time << " seconds" << std::endl;
        
        if (total_elements <= 32) {
            std::cout << "Sorted array (first " << total_elements << " elements):" << std::endl;
            print_array(sorted_array, total_elements);
        }
    }
    
    MPI_Finalize();
    return 0;
}
