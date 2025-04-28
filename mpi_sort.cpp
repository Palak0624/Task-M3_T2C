#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int total_elements = 32;
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
    
    auto start_time = MPI_Wtime();
    
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
    
    // Sort locally
    std::sort(local_array.begin(), local_array.end());
    
    // Gather the sorted subarrays
    std::vector<int> sorted_array;
    if (world_rank == 0) {
        sorted_array.resize(total_elements);
    }
    
    MPI_Gatherv(local_array.data(), elements_per_proc, MPI_INT,
                sorted_array.data(), sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    
    // Final merge on root process
    if (world_rank == 0) {
        // Simple approach for small arrays - for larger arrays, a proper merge would be better
        std::sort(sorted_array.begin(), sorted_array.end());
        
        auto end_time = MPI_Wtime();
        double elapsed = end_time - start_time;
        
        if (total_elements <= 32) {
            std::cout << "Sorted array (first " << total_elements << " elements):" << std::endl;
            print_array(sorted_array, total_elements);
        }
        
        bool success = is_sorted(sorted_array);
        std::cout << "Sorting verification: " << (success ? "SUCCESS" : "FAILURE") << std::endl;
        std::cout << "Total elements: " << total_elements << std::endl;
        std::cout << "Time elapsed: " << elapsed << " seconds" << std::endl;
        std::cout << "Number of processes: " << world_size << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
