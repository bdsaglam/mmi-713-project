
#include <cuda_runtime.h>
#include <cmath> // For fabsf

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
__global__ void kSelectKernel(float *distances, int *indices, int n_rows, int n_cols, int k) {
    unsigned int yIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (yIndex >= n_rows || k > n_cols) return;

    // Pointer shift and initialization
    float *p_dist = distances + yIndex * n_cols;
    int *p_ind = indices + yIndex * n_cols;

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

            int temp_ind = p_ind[i];
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
