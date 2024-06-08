#include <algorithm>

long* argsort(const float* values, long n_rows, long n_cols) {
    long* indices = new long[n_rows * n_cols];

    // Initialize indices with the values 0, 1, 2, ..., n_cols-1 for each row
    for (long row = 0; row < n_rows; ++row) {
        for (long col = 0; col < n_cols; ++col) {
            indices[row * n_cols + col] = col;
        }
    }

    // Sort indices based on values
    for (long row = 0; row < n_rows; ++row) {
        std::sort(indices + row * n_cols, indices + (row + 1) * n_cols,
                  [&values, row, n_cols](long i1, long i2) {
                      return values[row * n_cols + i1] < values[row * n_cols + i2];
                  });
    }

    return indices;
}