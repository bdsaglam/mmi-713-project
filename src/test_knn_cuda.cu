#include <cuda_runtime.h>
#include <algorithm>
#include <cmath> // For fabsf
#include <iostream>
#include <stdlib.h>
#include "common.h"
#include "sorting.h"
#include "knn.h"

#define DEBUG 1

// Kernel for computing distances for each point in documents and queries. 
// Output dimensions: QxNxD
__global__ void computeL1DistanceKernel(float *documents, float *queries, float *output, int D, int N, int Q) {
    // Calculate the thread's unique ID
    int qIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int nIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int dIndex = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure the thread ID is within the bounds of the queries, documents, and dimensions
    if (qIndex < Q && nIndex < N && dIndex < D) {
        int docIndex = nIndex * D + dIndex;
        int queryIndex = qIndex * D + dIndex;
        int outputIndex = (qIndex * N + nIndex) * D + dIndex;
        output[outputIndex] = fabsf(queries[queryIndex] - documents[docIndex]);
    }
}

// Kernel for summing distance values over the last dimension of a 3D array (QxNxD) flattened in memory
// Output dimensions: QxN
__global__ void sumOverLastDimKernel(float *g_idata, float *g_odata, int D, int N, int Q) {
    extern __shared__ float sdata[];

    // Calculate the global and shared memory indices
    unsigned int tid = threadIdx.x;
    unsigned int q = blockIdx.z;
    unsigned int n = blockIdx.y;
    unsigned int i = q * (N * D) + n * D + tid;

    // Load data into shared memory
    if (tid < D) {
        sdata[tid] = g_idata[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        g_odata[q * N + n] = sdata[0];
    }
}

/**
  * Gathers k-th smallest distances for each row of the distance matrix in the top.
  *
  * @param distances   pointer for distances array
  * @param indices     pointer for indices array
  * @param n_rows      number of columns
  * @param n_cols      number of rows
  * @param k           number of smallest element to select
  */
__global__ void kSelectKernel(float *distances, long *indices, int n_rows, int n_cols, int k) {
    unsigned int yIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (yIndex >= n_rows || k > n_cols) return;

    // Pointer shift and initialization
    float *p_dist = distances + yIndex * n_cols;
    long *p_ind = indices + yIndex * n_cols;

    // Initialize indices for tracking positions
    for (int i = 0; i < n_cols; i++) {
        p_ind[i] = i;
    }

    // Initial sorting of the first k elements using selection sort
    for (int i = 0; i < k; i++) {
        int min_idx = i;
        for (int j = i + 1; j < k; j++) {
            if (p_dist[j] < p_dist[min_idx]) {
                min_idx = j;
            }
        }
        // Swap if needed
        if (min_idx != i) {
            float temp_dist = p_dist[i];
            p_dist[i] = p_dist[min_idx];
            p_dist[min_idx] = temp_dist;

            long temp_ind = p_ind[i];
            p_ind[i] = p_ind[min_idx];
            p_ind[min_idx] = temp_ind;
        }
    }

    // Set initial max_dist from the first k elements
    float max_dist = p_dist[k-1];

    // Process remaining elements
    for (int l = k; l < n_cols; l++) {
        float curr_dist = p_dist[l];
        if (curr_dist < max_dist) {
            // Find the correct position to insert the current distance
            int i;
            for (i = k - 1; i > 0 && p_dist[i-1] > curr_dist; i--) {
                p_dist[i] = p_dist[i-1];
                p_ind[i] = p_ind[i-1];
            }
            p_dist[i] = curr_dist;
            p_ind[i] = l;

            // Update max_dist
            max_dist = p_dist[k-1];
        }
    }
}




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
    long *h_indices = (long *)malloc(Q * N * sizeof(long)); // Indices array to store the output of kSelectKernel

    // Initialize data with random values
    // srand(time(NULL));
    randomInit(h_documents, N, D);
    randomInit(h_queries, Q, D);

    // Mark start time
    clock_t start = clock();

    // Allocate device memory
    float *d_documents, *d_queries, *d_distances, *d_agg_distances;
    long *d_indices;
    cudaMalloc(&d_documents, N * D * sizeof(float));
    cudaMalloc(&d_queries, Q * D * sizeof(float));
    cudaMalloc(&d_distances, Q * N * D * sizeof(float));
    cudaMalloc(&d_agg_distances, Q * N * sizeof(float));
    cudaMalloc(&d_indices, Q * N * sizeof(long)); // Device memory for indices

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
    cudaMemcpy(h_indices, d_indices, Q * N * sizeof(long), cudaMemcpyDeviceToHost);

    // Copy the result back to host
    long* h_sorted_indices = argsort(h_agg_distances, Q, N);

    // Measure elapsed time
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f ms\n", elapsed_time_ms);
    
    // Print results
    printResults(h_indices, Q, N, K);
    
    // Print results
    printResults(h_sorted_indices, Q, N, K);

    // Verification
#if DEBUG
    // Allocate memory
    float *h_distances_cpu = (float *)malloc(Q * N * D * sizeof(float));
    float *h_agg_distances_cpu = (float *)malloc(Q * N * sizeof(float));

    // Perform the same operations on the CPU
    computeL1Distance(h_documents, h_queries, h_distances_cpu, D, N, Q);
    sumOverLastDim(h_distances_cpu, h_agg_distances_cpu, D, N, Q);
    long* h_sorted_indices_cpu = argsort(h_agg_distances_cpu, Q, N);
    
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
        if (avgError > 1e-3)
            printf("Avg error for query %d: %f\n", q, avgError);
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
        if (avgError > 1e-3)
            printf("Avg error for query %d: %f\n", q, avgError);
    }
    
    // Verify the sorting by comparing the GPU and CPU results
    printf("\nVerifying sorting...\n");
    for (int q = 0; q < Q; ++q) {
        float totalError = 0.0;
        for (int i = 0; i < N; ++i) {
            int index = q * N + i;
            totalError += abs(h_sorted_indices[index] - h_sorted_indices_cpu[index]);
        }
        float avgError = totalError / N;
        if (avgError > 1e-3)
            printf("Avg error for query %d: %f\n", q, avgError);
    }

    // Verify the k-select by comparing the GPU and CPU results
    printf("\nVerifying k-select...\n");
    for (int q = 0; q < Q; ++q) {
        float totalError = 0.0;
        for (int i = 0; i < K; ++i) {
            int index = q * N + i;
            totalError += abs(h_indices[index] - h_sorted_indices_cpu[index]);
        }
        float avgError = totalError / N;
        if (avgError > 1e-3)
            printf("Avg error for query %d: %f\n", q, avgError);
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

    return 0;
}
