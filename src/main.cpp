
#include <iostream>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <random>
#include <chrono>

void benchmark_gemm(rocblas_handle, int, int);

// This is a benchmarking program for GEMM operations on MI210 using rocBLAS library
int main(int argc, const char** argv)
{
    // Initialize rocBLAS
    rocblas_handle handle;
    rocblas_status status = rocblas_create_handle(&handle);

    // Check for errors
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS initialization failed" << std::endl;
        return -1;
    }

    // Parse the matrix sizes to be used for benchmarking from the command line
    // The arguments are a list of matrix sizes separated by spaces, e.g. 1024,1024 2048,512 4096,4096
    std::vector<std::pair<int, int>> matrix_sizes;
    for (int i = 1; i < argc; i++)
    {
        int m = 0, n = 0;
        std::sscanf(argv[i], "%d,%d", &m, &n);
        matrix_sizes.push_back(std::make_pair(m, n));
    }

    // Benchmarking GEMM operations
    for (auto& matrix_size : matrix_sizes)
    {
        benchmark_hgemm(handle, matrix_size.first, matrix_size.second);
    }

    return 0;
}

// Benchmarking GEMM operations
void benchmark_hgemm(rocblas_handle handle, int rows, int columns)
{
    // Print the matrix size
    std::cout << "Benchmarking GEMM operation on " << rows << "x" << columns << " matrix" << std::endl;

    typdef rocblas_bfloat16 data_type;

    // Allocate the matrices on the host
    std::vector<data_type> A(rows * columns);
    std::vector<data_type> B(rows * columns);

    // Initialize the matrices with random values
    auto rng = std::default_random_engine(0);
    for (int i = 0; i < rows * columns; i++)
    {
        A[i] = rng() % 100;
        B[i] = rng() % 100;
    }

    // Allocate the matrices on the device
    data_type* dA;
    data_type* dB;
    hipError_t error = hipMalloc((void**)&dA, rows * columns * sizeof(data_type));
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS device memory allocation failed for A" << std::endl;
        return;
    }

    error = hipMalloc((void**)&dB, rows * columns * sizeof(data_type));
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS device memory allocation failed for B" << std::endl;
        return;
    }

    // Copy the matrices from the host to the device
    rocblas_status status = rocblas_set_matrix(rows, columns, sizeof(data_type), A.data(), rows, dA, rows);
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS copy from host to device failed for A" << std::endl;
        return;
    }

    status = rocblas_set_matrix(rows, columns, sizeof(data_type), B.data(), rows, dB, rows);
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS copy from host to device failed for B" << std::endl;
        return;
    }

    // Synchronize the device
    error = hipDeviceSynchronize();
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS device synchronization failed" << std::endl;
        return;
    }

    // Start a timer
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the GEMM operation on the device
    data_type alpha = 1.0f;
    data_type beta = 0.0f;
    status = rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, rows, columns, rows, &alpha, dA, rows, dB, rows, &beta, dB, rows);
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS GEMM operation failed" << std::endl;
        return;
    }

    // Synchronize the device
    error = hipDeviceSynchronize();
    if (error != hipSuccess)
    {
        std::cout << "rocblas gemm failed" << std::endl;
        return;
    }

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Print the time taken
    std::cout << "Time taken for GEMM operation on " << rows << "x" << columns << " matrix: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Print the TFLOP/s
    std::cout << "TFLOP/s: " << (2.0 * rows * columns * rows) / (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1e9) << std::endl;

    // Free the device memory
    error = hipFree(dA);
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS free failed" << std::endl;
        return;
    }
    error = hipFree(dB);
    if (error != hipSuccess)
    {
        std::cout << "rocBLAS free failed" << std::endl;
        return;
    }

}


