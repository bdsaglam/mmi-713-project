#include <algorithm>
#include <cmath> // For fabsf
#include <iostream>
#include "common.h"
#include "sorting.h"
#include "knn.h"


int main() {
    // Example dimensions
    int D = 512;   // Dimensionality
    int N = 10;  // Number of documents
    int Q = 4;   // Number of queries

    // Allocate host memory
    float *h_documents = (float *)malloc(N * D * sizeof(float));
    float *h_queries = (float *)malloc(Q * D * sizeof(float));

    // Initialize data with random values
    // srand(time(NULL));
    randomInit(h_documents, N, D);
    randomInit(h_queries, Q, D);

    // KNN algorithm
    int* h_sorted_indices = knn(h_documents, h_queries, D, N, Q);

    // Print a few results
    printf("Sorted indices\n");
    printMatrix(h_sorted_indices, Q, N);

    // Clean up memory
    free(h_documents);
    free(h_queries);
    free(h_sorted_indices);

    return 0;
}
