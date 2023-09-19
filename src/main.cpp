
#include <iostream>
#include <rocblas.h>
#include <vector>
#include <random>

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
        benchmark_gemm(handle, matrix_size.first, matrix_size.second);
    }

    return 0;
}

// Benchmarking GEMM operations
void benchmark_gemm(rocblas_handle handle, int rows, int columns)
{
    // Print the matrix size
    std::cout << "Benchmarking GEMM operation on " << rows << "x" << columns << " matrix" << std::endl;

    // Allocate the matrices on the host
    std::vector<float> A(rows * columns);
    std::vector<float> B(rows * columns);

    // Initialize the matrices with random values
    auto rng = std::default_random_engine(seed=0);
    for (int i = 0; i < rows * columns; i++)
    {
        A[i] = rng() % 100;
        B[i] = rng() % 100;
    }

    // Allocate the matrices on the device
    float* dA;
    float* dB;
    rocblas_status status = rocblas_malloc((void**)&dA, rows * columns * sizeof(float));
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS device memory allocation failed for A" << std::endl;
        return;
    }

    status = rocblas_malloc((void**)&dB, rows * columns * sizeof(float));
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS device memory allocation failed for B" << std::endl;
        return;
    }

    // Copy the matrices from the host to the device
    status = rocblas_set_matrix(rows, columns, sizeof(float), A.data(), rows, dA, rows);
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS copy from host to device failed for A" << std::endl;
        return;
    }

    status = rocblas_set_matrix(rows, columns, sizeof(float), B.data(), rows, dB, rows);
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS copy from host to device failed for B" << std::endl;
        return;
    }

    // Synchronize the device
    status = rocblas_synchronize(handle);
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS device synchronization failed" << std::endl;
        return;
    }

    // Start a timer
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the GEMM operation on the device
    float alpha = 1.0f;
    float beta = 0.0f;
    status = rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, rows, columns, rows, &alpha, dA, rows, dB, rows, &beta, dB, rows);
    if (status != rocblas_status_success)
    {
        std::cout << "rocBLAS GEMM operation failed" << std::endl;
        return;
    }

    // Synchronize the device
    status = rocblas_synchronize(handle);

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();

    // Print the time taken
    std::cout << "Time taken for GEMM operation on " << rows << "x" << columns << " matrix: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Print the TFLOP/s
    std::cout << "TFLOP/s: " << (2.0 * rows * columns * rows) / (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() * 1000000) << std::endl;

    // Free the device memory
    rocblas_free(dA);
    rocblas_free(dB);

}


