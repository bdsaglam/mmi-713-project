#include <algorithm>
#include <cmath> // For fabsf
#include <iostream>
#include "cli.h"
#include "constants.h"
#include "common.h"
#include "sorting.h"
#include "knn.h"

int main(int argc, char *argv[]) {
    Params params;
    parseCommandLine(argc, argv, params);

    int N = params.N;
    int Q = params.Q;

    // Allocate host memory
    float *h_documents = (float *)malloc(N * D * sizeof(float));
    float *h_queries = (float *)malloc(Q * D * sizeof(float));

    // Initialize data with random values
    // srand(time(NULL));
    randomInit(h_documents, N, D);
    randomInit(h_queries, Q, D);

    // Mark start time
    clock_t start = clock();

    // KNN algorithm
    int* h_sorted_indices = knn(h_documents, h_queries, D, N, Q);

    // Measure elapsed time
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f ms\n", elapsed_time_ms);

    // Print results
    printResults(h_sorted_indices, Q, N, K);

    // Clean up memory
    free(h_documents);
    free(h_queries);
    free(h_sorted_indices);

    return 0;
}
