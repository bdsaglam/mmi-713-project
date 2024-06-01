#include <algorithm>
#include <iostream>
#include "sorting.h"
#include "common.h"

// Function to sort matrix rows based on argsort results
template <typename T>
void sortRowsByIndices(const T* matrix, const int* sorted_indices, T* sorted_matrix, int n_rows, int n_cols) {
    for (int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {
            sorted_matrix[row * n_cols + col] = matrix[row * n_cols + sorted_indices[row * n_cols + col]];
        }
    }
}

// Function to check if all rows are sorted
template <typename T>
bool areRowsSorted(const T* matrix, int n_rows, int n_cols) {
    for (int row = 0; row < n_rows; ++row) {
        for (int col = 1; col < n_cols; ++col) {
            if (matrix[row * n_cols + col] < matrix[row * n_cols + col - 1]) {
                std::cout << "Row " << row << " is not sorted." << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    float values[] = {
        173.377, 174.195, 170.147, 180.184, 177.743, 180.625, 172.44, 173.879, 174.055, 162.424,
        166.112, 171.351, 168.213, 182.188, 172.153, 166, 178.473, 164.797, 170.585, 162.037,
        170.239, 171.761, 168.462, 178.009, 171.772, 160.761, 163.037, 162.488, 163.554, 165.491
    };
    int n_rows = 3;
    int n_cols = 10;

    // Perform argsort on the matrix
    int* sorted_indices = argsort(values, n_rows, n_cols);
    float* sorted_matrix = new float[n_rows * n_cols];

    // Sort the matrix rows based on the argsort results
    sortRowsByIndices(values, sorted_indices, sorted_matrix, n_rows, n_cols);

    // Print the original matrix
    std::cout << "Original matrix:\n";
    printMatrix(values, n_rows, n_cols);

    // Print the sorted indices
    std::cout << "Sorted indices:\n";
    printMatrix(sorted_indices, n_rows, n_cols);

    // Print the sorted matrix
    std::cout << "Sorted matrix:\n";
    printMatrix(sorted_matrix, n_rows, n_cols);

    // Verify if the rows are sorted
    if (areRowsSorted(sorted_matrix, n_rows, n_cols)) {
        std::cout << "All rows are sorted." << std::endl;
    } else {
        std::cout << "Some rows are not sorted." << std::endl;
    }

    // Clean up dynamically allocated memory
    delete[] sorted_indices;
    delete[] sorted_matrix;

    return 0;
}
