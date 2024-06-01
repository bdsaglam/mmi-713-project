#include <algorithm>
#include <cmath> // For fabsf
#include <iostream>
#include "common.h"
#include "sorting.h"

// CPU

void computeL1DistanceCPU(float *documents, float *queries, float *output, int D, int N, int Q) {
    for (int q = 0; q < Q; ++q) {
        for (int n = 0; n < N; ++n) {
            for (int d = 0; d < D; ++d) {
                int docIndex = n * D + d;
                int queryIndex = q * D + d;
                int outputIndex = (q * N + n) * D + d;
                output[outputIndex] = fabsf(queries[queryIndex] - documents[docIndex]);
            }
        }
    }
}

// Function to sum over the last dimension
void sumOverLastDim(float *h_distances, float *h_output, int D, int N, int Q) {
    for (int q = 0; q < Q; ++q) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                int index = (q * N + n) * D + d;
                sum += h_distances[index];
            }
            int outputIndex = q * N + n;
            h_output[outputIndex] = sum;
        }
    }
}

int main() {
    // Example dimensions
    int D = 512;   // Dimensionality
    int N = 10;  // Number of documents
    int Q = 4;   // Number of queries

    // Allocate host memory
    float *h_documents = (float *)malloc(N * D * sizeof(float));
    float *h_queries = (float *)malloc(Q * D * sizeof(float));
    float *h_distances = (float *)malloc(Q * N * D * sizeof(float));
    float *h_results = (float *)malloc(Q * N * sizeof(float));

    // Initialize data with random values
    // srand(time(NULL));
    randomInit(h_documents, N, D);
    randomInit(h_queries, Q, D);

    // KNN algorithm
    computeL1DistanceCPU(h_documents, h_queries, h_distances, D, N, Q);
    sumOverLastDim(h_distances, h_results, D, N, Q);
    int* h_sorted_indices = argsort(h_results, Q, N);

    // Print a few results
    printf("Distances\n");
    printMatrix(h_results, Q, N);
    printf("Sorted indices\n");
    printMatrix(h_sorted_indices, Q, N);

    // Clean up memory
    free(h_documents);
    free(h_queries);
    free(h_results);
    free(h_distances);
    free(h_sorted_indices);

    return 0;
}
