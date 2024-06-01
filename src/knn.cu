#include <cuda_runtime.h>
#include <algorithm>
#include <cmath> // For fabsf
#include <iostream>
#include <stdlib.h>
#include "common.h"
#include "sorting.h"

#define DEBUG 1

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

// GPU

__global__ void computeL1Distance(float *documents, float *queries, float *output, int D, int N, int Q) {
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

// GPU kernel for summing distance values over the last dimension of a 3D array (QxNxD) flattened in memory
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


int main() {
    // Example dimensions
    int D = 512;   // Dimensionality
    int N = 100;  // Number of documents
    int Q = 4;   // Number of queries
    int K = 10; // Number of matches to return

    // Allocate host memory
    float *h_documents = (float *)malloc(N * D * sizeof(float));
    float *h_queries = (float *)malloc(Q * D * sizeof(float));
    float *h_results = (float *)malloc(Q * N * sizeof(float));

    // Initialize data with random values
    // srand(time(NULL));
    randomInit(h_documents, N, D);
    randomInit(h_queries, Q, D);

    // Allocate device memory
    float *d_documents, *d_queries, *d_distances, *d_results;
    cudaMalloc(&d_documents, N * D * sizeof(float));
    cudaMalloc(&d_queries, Q * D * sizeof(float));
    cudaMalloc(&d_distances, Q * N * D * sizeof(float));
    cudaMalloc(&d_results, Q * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_documents, h_documents, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, Q * D * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block sizes
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((Q + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   (D + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Compute L1 distances
    computeL1Distance<<<numBlocks, threadsPerBlock>>>(d_documents, d_queries, d_distances, D, N, Q);
    cudaError_t err_dist = cudaGetLastError();
    if (err_dist != cudaSuccess) {
        std::cerr << "Failed to launch computeL1Distance kernel: " << cudaGetErrorString(err_dist) << std::endl;
        return -1;
    }

    // Sum over the last dim
    dim3 blockDim(D);  // Ensure blockDim does not exceed 512
    dim3 gridDim(1, N, Q);
    size_t sharedMemSize = D * sizeof(float);
    sumOverLastDimKernel<<<gridDim, blockDim, sharedMemSize>>>(d_distances, d_results, D, N, Q);
    cudaError_t err_sum = cudaGetLastError();
    if (err_sum != cudaSuccess) {
        std::cerr << "Failed to launch sumOverLastDimKernel kernel: " << cudaGetErrorString(err_sum) << std::endl;
        return -1;
    }

    // Copy the result back to host
    cudaMemcpy(h_results, d_results, Q * N * sizeof(float), cudaMemcpyDeviceToHost);
    int* h_sorted_indices = argsort(h_results, Q, N);
    
    // Verification
#if DEBUG
    // Allocate memory
    float *h_distances_cpu = (float *)malloc(Q * N * D * sizeof(float));
    float *h_results_cpu = (float *)malloc(Q * N * sizeof(float));

    // Perform the same operations on the CPU
    computeL1DistanceCPU(h_documents, h_queries, h_distances_cpu, D, N, Q);
    sumOverLastDim(h_distances_cpu, h_results_cpu, D, N, Q);
    int* h_sorted_indices_cpu = argsort(h_results_cpu, Q, N);

    // Verify the distances by comparing the GPU and CPU results
    printf("\nVerifying distance computation...\n");
    for (int q = 0; q < Q; ++q) {
        float totalError = 0.0;
        for (int i = 0; i < N; ++i) {
            int index = q * N + i;
            totalError += h_results[index] - h_results_cpu[index];
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
            totalError += h_sorted_indices[index] - h_sorted_indices_cpu[index];
        }
        float avgError = totalError / N;
        if (avgError > 1e-3)
            printf("Avg error for query %d: %f\n", q, avgError);
    }

    // Deallocate memory
    free(h_distances_cpu);
    free(h_results_cpu);
    free(h_sorted_indices_cpu);
    
#endif

    // Print a few results
    printf("\nTop documents for queries\n");
    printMatrix(h_sorted_indices, Q, N, Q, K);

    // Clean up memory
    cudaFree(d_documents);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_results);

    free(h_documents);
    free(h_queries);
    free(h_results);
    free(h_sorted_indices);

    return 0;
}
