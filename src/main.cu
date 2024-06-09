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

    // Allocate device memory
    float *d_documents, *d_queries, *d_distances, *d_agg_distances;
    int *d_indices;
    cudaMalloc(&d_documents, N * D * sizeof(float));
    cudaMalloc(&d_queries, Q * D * sizeof(float));
    cudaMalloc(&d_distances, Q * N * D * sizeof(float));
    cudaMalloc(&d_agg_distances, Q * N * sizeof(float));
    cudaMalloc(&d_indices, Q * N * sizeof(int )); // Device memory for indices

    // Copy data from host to device
    cudaMemcpy(d_documents, h_documents, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, Q * D * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block sizes
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((Q + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   (D + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Compute L1 distances
    computeL1DistanceKernel<<<numBlocks, threadsPerBlock>>>(d_documents, d_queries, d_distances, D, N, Q);
    cudaError_t err_dist = cudaGetLastError();
    if (err_dist != cudaSuccess) {
        std::cerr << "Failed to launch computeL1DistanceKernel: " << cudaGetErrorString(err_dist) << std::endl;
        return -1;
    }

    // Sum over the last dim
    dim3 blockDim(D);
    dim3 gridDim(1, N, Q);
    size_t sharedMemSize = D * sizeof(float);
    sumOverLastDimKernel<<<gridDim, blockDim, sharedMemSize>>>(d_distances, d_agg_distances, D, N, Q);
    cudaError_t err_sum = cudaGetLastError();
    if (err_sum != cudaSuccess) {
        std::cerr << "Failed to launch sumOverLastDimKernel: " << cudaGetErrorString(err_sum) << std::endl;
        return -1;
    }
      
    // Select k smallest elements
    int kSelectThreadsPerBlock = 1024;
    int kSelectBlocksPerGrid = (Q + kSelectThreadsPerBlock - 1) / kSelectThreadsPerBlock;
    kSelectKernel<<<kSelectBlocksPerGrid, kSelectThreadsPerBlock>>>(d_agg_distances, d_indices, Q, N, K);
    
    // Copy the sorted indices back to the host
    cudaMemcpy(h_indices, d_indices, Q * N * sizeof(int ), cudaMemcpyDeviceToHost);

    // Measure elapsed time
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f ms\n", elapsed_time_ms);
    
    // Print results
    printResults(h_indices, Q, N, K);

    // Clean up memory
    cudaFree(d_documents);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_agg_distances);
    cudaFree(d_indices);

    free(h_documents);
    free(h_queries);
    free(h_indices);

    return 0;
}
