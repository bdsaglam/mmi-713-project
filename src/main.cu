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
  if (yIndex >= n_rows) return;

  // Pointer shift, initialization, and max value
  float *p_dist = distances + yIndex * n_cols;
  long *p_ind = indices + yIndex * n_cols;

  float max_dist = p_dist[0];
  p_ind[0] = 0;

  int l, i;

  // Part 1 : sort k-th first elements
  float curr_dist;
  long  curr_col;
  for (l = 1; l < k; l++) {
    curr_col  = l;
    curr_dist = p_dist[curr_col];
    if (curr_dist < max_dist) {
      // new small element found
      // find insertion index
      i = l - 1;
      for (int a = 0; a < l - 1; a++) {
        if (p_dist[a] > curr_dist) {
          i = a;
          break;
        }
      }
      // shift all elements after insertion index to right
      for (int j = l; j > i; j--) {
        p_dist[j] = p_dist[j - 1];
        p_ind[j]  = p_ind[j - 1];
      }
      p_dist[i] = curr_dist;
      p_ind[i]  = l;
    } else {
      p_ind[l] = l;
    }
    max_dist = p_dist[curr_col];
  }

  // Part 2 : insert element in the k-th first lines
  long max_col = k - 1;
  for (l = k; l < n_cols; l++) {
    curr_dist = p_dist[l];
    if (curr_dist < max_dist) {
      i = k - 1;
      for (int a = 0; a < k - 1; a++) {
        if (p_dist[a] > curr_dist) {
          i = a;
          break;
        }
      }
      for (int j = k - 1; j > i; j--) {
        p_dist[j] = p_dist[j - 1];
        p_ind[j]  = p_ind[j - 1];
      }
      p_dist[i] = curr_dist;
      p_ind[i]  = l;
      max_dist  = p_dist[max_col];
    }
  }
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s number_of_docs\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]); // Number of documents

    // Constants
    int D = 512; // Dimension of embedding vector
    int Q = 10;  // Number of queries
    int K = 10;  // Number of matches to return

    // Allocate host memory
    float *h_documents = (float *)malloc(N * D * sizeof(float));
    float *h_queries = (float *)malloc(Q * D * sizeof(float));
    float *h_agg_distances = (float *)malloc(Q * N * sizeof(float));

    // Initialize data with random values
    // srand(time(NULL));
    randomInit(h_documents, N, D);
    randomInit(h_queries, Q, D);

    // Mark start time
    clock_t start = clock();

    // Allocate device memory
    float *d_documents, *d_queries, *d_distances, *d_agg_distances;
    cudaMalloc(&d_documents, N * D * sizeof(float));
    cudaMalloc(&d_queries, Q * D * sizeof(float));
    cudaMalloc(&d_distances, Q * N * D * sizeof(float));
    cudaMalloc(&d_agg_distances, Q * N * sizeof(float));

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
    dim3 blockDim(D);  // Ensure blockDim does not exceed 512
    dim3 gridDim(1, N, Q);
    size_t sharedMemSize = D * sizeof(float);
    sumOverLastDimKernel<<<gridDim, blockDim, sharedMemSize>>>(d_distances, d_agg_distances, D, N, Q);
    cudaError_t err_sum = cudaGetLastError();
    if (err_sum != cudaSuccess) {
        std::cerr << "Failed to launch sumOverLastDimKernel: " << cudaGetErrorString(err_sum) << std::endl;
        return -1;
    }

    // Copy the result back to host
    cudaMemcpy(h_agg_distances, d_agg_distances, Q * N * sizeof(float), cudaMemcpyDeviceToHost);
    long* h_sorted_indices = argsort(h_agg_distances, Q, N);

    // Measure elapsed time
    clock_t end = clock();
    double elapsed_time_ms = 1000 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f ms\n", elapsed_time_ms);
    
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
            int index = q * N + i;
            totalError += h_agg_distances[index] - h_agg_distances_cpu[index];
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
    free(h_agg_distances_cpu);
    free(h_sorted_indices_cpu);
    
#endif

    // Print results
    printResults(h_sorted_indices, Q, N, K);

    // Clean up memory
    cudaFree(d_documents);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_agg_distances);

    free(h_documents);
    free(h_queries);
    free(h_agg_distances);
    free(h_sorted_indices);

    return 0;
}
