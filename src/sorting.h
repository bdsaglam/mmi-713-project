#include <algorithm>

int* argsort(const float* values, int n_rows, int n_cols) {
    int* indices = new int[n_rows * n_cols];

    // Initialize indices with the values 0, 1, 2, ..., n_cols-1 for each row
    for (int row = 0; row < n_rows; ++row) {
        for (int col = 0; col < n_cols; ++col) {
            indices[row * n_cols + col] = col;
        }
    }

    // Sort indices based on values
    for (int row = 0; row < n_rows; ++row) {
        std::sort(indices + row * n_cols, indices + (row + 1) * n_cols,
                  [&values, row, n_cols](int i1, int i2) {
                      return values[row * n_cols + i1] < values[row * n_cols + i2];
                  });
    }

    return indices;
}