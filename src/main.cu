#include <cuda_runtime.h>
#include <algorithm>
#include <cmath> // For fabsf
#include <iostream>
#include <stdlib.h>
#include "cli.h"
#include "common.h"
#include "constants.h"
#include "sorting.h"
#include "knn.h"
#include "knn.cu"


int main(int argc, char *argv[]) {
    Params params;
    parseCommandLine(argc, argv, params);

    int N = params.N;
    int Q = params.Q;

    // Allocate host memory
    float *h_documents = (float *)malloc(N * D * sizeof(float));
    float *h_queries = (float *)malloc(Q * D * sizeof(float));
    int *h_indices = (int *)malloc(Q * N * sizeof(int )); // Indices array to store the output of kSelectKernel

    // Initialize data with random values
    // srand(time(NULL));
    randomInit(h_documents, N, D);
    randomInit(h_queries, Q, D);

    // Mark start time
    clock_t start = clock();

    // parallel kNN
    knnParallel(h_documents, h_queries, h_indices, D, N, Q, K);

    // Measure elapsed time
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f ms\n", elapsed_time_ms);
    
    // Print results
    printResults(h_indices, Q, N, K);

    free(h_documents);
    free(h_queries);
    free(h_indices);

    return 0;
}
