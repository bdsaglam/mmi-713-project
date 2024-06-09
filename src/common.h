#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


void randomInit(float* a, long n_rows, long n_cols) {
    for (long i = 0; i < n_rows; i++)
    {
        for (long j = 0; j < n_cols; j++)
        {
            a[i*n_cols + j] = rand() / (float)RAND_MAX;
        }
    }
}

template <typename T>
void printMatrix(const T* matrix, long n_rows, long n_cols, long rows_to_print, long cols_to_print) {
    for (long row = 0; row < std::min(n_rows, rows_to_print); ++row) {
        std::cout << "Row " << row << std::endl;
        for (long col = 0; col < std::min(n_cols, cols_to_print); ++col) {
            std::cout << matrix[row * n_cols + col] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void printResults(const T* matrix, long n_rows, long n_cols, long cols_to_print) {
    printf("================================================================\n");
    for (long row = 0; row < n_rows; ++row) {
        std::cout << "Query " << row << ": Top " <<  cols_to_print << " matching docs" << std::endl;
        for (long col = 0; col < std::min(n_cols, cols_to_print); ++col) {
            std::cout << matrix[row * n_cols + col] << " ";
        }
        std::cout << std::endl;
    }
    printf("================================================================\n");
}
