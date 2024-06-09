#include <algorithm>
#include <iostream>
#include <random>
#include "sorting.h"
#include "common.h"

// Function to sort matrix rows based on argsort results
template <typename T>
void sortRowsByIndices(const T* matrix, const int * sorted_indices, T* sorted_matrix, int n_rows, int n_cols) {
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
    const int n_rows = 100;
    const int n_cols = 10;
    float values[n_rows * n_cols];

    // Seed for random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);

    // Generate random values for the matrix
    for (int i = 0; i < n_rows * n_cols; ++i) {
        values[i] = dis(gen);
    }
    
    // Perform argsort on the matrix
    int * sorted_indices = argsort(values, n_rows, n_cols);
    float* sorted_matrix = new float[n_rows * n_cols];

    // Sort the matrix rows based on the argsort results
    sortRowsByIndices(values, sorted_indices, sorted_matrix, n_rows, n_cols);

    // Verify if the rows are sorted
    if (areRowsSorted(sorted_matrix, n_rows, n_cols)) {
        std::cout << "All rows are sorted." << std::endl;
    } else {
        std::cout << "Some rows are not sorted!" << std::endl;
    }

    // Clean up dynamically allocated memory
    delete[] sorted_indices;
    delete[] sorted_matrix;

    return 0;
}
