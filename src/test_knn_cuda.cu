#include <cuda_runtime.h>
#include <algorithm>
#include <cmath> // For fabsf
#include <iostream>
#include <stdlib.h>
#include "common.h"
#include "sorting.h"
#include "knn.h"
#include "knn.cu"


#define DEBUG 1

int main(int argc, char *argv[]) {
    // Constants
    int N = 1000; // Number of documents
    int D = 512; // Dimension of embedding vector
    int Q = 10;  // Number of queries
    int K = 10;  // Number of matches to return

    // Allocate host memory
    float *h_documents = (float *)malloc(N * D * sizeof(float));
    float *h_queries = (float *)malloc(Q * D * sizeof(float));
    float *h_distances = (float *)malloc(Q * N * D * sizeof(float));
    float *h_agg_distances = (float *)malloc(Q * N * sizeof(float));
    int *h_indices = (int *)malloc(Q * N * sizeof(int )); // Indices array to store the output of kSelectKernel

    // Initialize data with random values
    srand(time(NULL));
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
    // Copy the result back to host
    cudaMemcpy(h_distances, d_distances, Q * N * D * sizeof(float), cudaMemcpyDeviceToHost);

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
    cudaMemcpy(h_agg_distances, d_agg_distances, Q * N * sizeof(float), cudaMemcpyDeviceToHost);
      
    // Select k smallest elements
    int kSelectThreadsPerBlock = 256;
    int kSelectBlocksPerGrid = (Q + kSelectThreadsPerBlock - 1) / kSelectThreadsPerBlock;
    kSelectKernel<<<kSelectBlocksPerGrid, kSelectThreadsPerBlock>>>(d_agg_distances, d_indices, Q, N, K);
    
    // Copy the sorted indices back to the host
    cudaMemcpy(h_indices, d_indices, Q * N * sizeof(int ), cudaMemcpyDeviceToHost);

    // Copy the result back to host
    int * h_sorted_indices = argsort(h_agg_distances, Q, N);

    // Measure elapsed time
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f ms\n", elapsed_time_ms);
    
    // Print results
    printf("Results with k-select on GPU");
    printResults(h_indices, Q, N, K);
    
    // Print results
    printf("Results with sorting on CPU");
    printResults(h_sorted_indices, Q, N, K);

    int returnCode = 0;
    // Verification
#if DEBUG

    // Allocate memory
    float *h_distances_cpu = (float *)malloc(Q * N * D * sizeof(float));
    float *h_agg_distances_cpu = (float *)malloc(Q * N * sizeof(float));

    // Perform the same operations on the CPU
    computeL1Distance(h_documents, h_queries, h_distances_cpu, D, N, Q);
    sumOverLastDim(h_distances_cpu, h_agg_distances_cpu, D, N, Q);
    int * h_sorted_indices_cpu = argsort(h_agg_distances_cpu, Q, N);
    
    // Verify the distances by comparing the GPU and CPU results
    printf("\nVerifying distance computation...\n");
    for (int q = 0; q < Q; ++q) {
        float totalError = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < D; ++j) {
              int index = (q * N + i) * D + j;
              totalError += abs(h_distances[index] - h_distances_cpu[index]);
            }
        }
        float avgError = totalError / N;
        if (avgError > 1e-3) {
            printf("Avg error for query %d: %f\n", q, avgError);
            returnCode = 1;
        }
    }

    // Verify the distances by comparing the GPU and CPU results
    printf("\nVerifying aggregated distances...\n");
    for (int q = 0; q < Q; ++q) {
        float totalError = 0.0;
        for (int i = 0; i < N; ++i) {
            int index = q * N + i;
            totalError += abs(h_agg_distances[index] - h_agg_distances_cpu[index]);
        }
        float avgError = totalError / N;
        if (avgError > 1e-3) {
            printf("Avg error for query %d: %f\n", q, avgError);
            returnCode = 1;
        }
    }
    
    // Verify the sorting by comparing the GPU and CPU results
    printf("\nVerifying sorting...\n");
    for (int q = 0; q < Q; ++q) {
        int errorCount = 0;
        for (int i = 0; i < N; ++i) {
            int index = q * N + i;
            if ((h_sorted_indices[index] - h_sorted_indices_cpu[index]) != 0) errorCount++;
        }
        if (errorCount > 0) {
            printf("Error count for query %d: %d\n", q, errorCount);
            returnCode = 1;
        }
    }

    // Verify the k-select by comparing the GPU and CPU results
    printf("\nVerifying k-select...\n");
    for (int q = 0; q < Q; ++q) {
        int errorCount = 0;
        for (int i = 0; i < K; ++i) {
            int index = q * N + i;
            if ((h_indices[index] - h_sorted_indices_cpu[index]) != 0) errorCount++;
        }
        if (errorCount > 0) {
            printf("Error count for query %d: %d\n", q, errorCount);
            returnCode = 1;
        }
    }

    // Deallocate memory
    free(h_distances_cpu);
    free(h_agg_distances_cpu);
    free(h_sorted_indices_cpu);
    
#endif

    // Clean up memory
    cudaFree(d_documents);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_agg_distances);
    cudaFree(d_indices);

    free(h_documents);
    free(h_queries);
    free(h_agg_distances);
    free(h_indices);
    free(h_sorted_indices);

    return returnCode;
}
