#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


void randomInit(float* a, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            a[i*n_cols + j] = rand() / (float)RAND_MAX;
        }
    }
}

template <typename T>
void printMatrix(const T* matrix, int n_rows, int n_cols, int rows_to_print, int cols_to_print) {
    for (int row = 0; row < std::min(n_rows, rows_to_print); ++row) {
        std::cout << "Row " << row << std::endl;
        for (int col = 0; col < std::min(n_cols, cols_to_print); ++col) {
            std::cout << matrix[row * n_cols + col] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void printResults(const T* matrix, int n_rows, int n_cols, int cols_to_print) {
    printf("================================================================\n");
    for (int row = 0; row < n_rows; ++row) {
        std::cout << "Query " << row << ": Top " <<  cols_to_print << " matching docs" << std::endl;
        for (int col = 0; col < std::min(n_cols, cols_to_print); ++col) {
            std::cout << matrix[row * n_cols + col] << " ";
        }
        std::cout << std::endl;
    }
    printf("================================================================\n");
}
