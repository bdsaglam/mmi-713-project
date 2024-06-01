#include <cuda_runtime.h>
#include <iostream>
#include <cmath> // For fabsf
#include "common.h"

// Function to sum along the last dimension
void sumAlongLastDim(float *h_output, float *h_result, int Q, int N, int D) {
    for (int q = 0; q < Q; ++q) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                int outputIndex = (q * N + n) * D + d;
                sum += h_output[outputIndex];
            }
            int resultIndex = q * N + n;
            h_result[resultIndex] = sum;
        }
    }
}

__global__ void computeL1Distance(float *documents, float *queries, float *output, int N, int D, int Q) {
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

int main() {
    // Example dimensions
    int N = 1024;  // Number of documents
    int D = 128;   // Dimensionality
    int Q = 512;   // Number of queries

    // Allocate host memory
    float *h_documents = (float *)malloc(N * D * sizeof(float));
    float *h_queries = (float *)malloc(Q * D * sizeof(float));
    float *h_output = (float *)malloc(Q * N * D * sizeof(float));
    float *h_result = (float *)malloc(Q * N * sizeof(float));

    // Initialize data with random values
    randomInit(h_documents, N, D);
    randomInit(h_queries, Q, D);

    // Allocate device memory
    float *d_documents, *d_queries, *d_output;
    cudaMalloc(&d_documents, N * D * sizeof(float));
    cudaMalloc(&d_queries, Q * D * sizeof(float));
    cudaMalloc(&d_output, Q * N * D * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_documents, h_documents, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, Q * D * sizeof(float), cudaMemcpyHostToDevice);

    // Configure grid and block sizes
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((Q + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                   (D + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Launch the kernel
    computeL1Distance<<<numBlocks, threadsPerBlock>>>(d_documents, d_queries, d_output, N, D, Q);
    
    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch kernel: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy the result back to host
    cudaMemcpy(h_output, d_output, Q * N * D * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum along last dimension
    sumAlongLastDim(h_output, h_result, Q, N, D);
    for (int i = 0; i < 10; ++i) {
        std::cout << "Sum for query " << i << " and document " << i << ": " << h_result[i * N + i] << std::endl;
    }

    // Clean up memory
    cudaFree(d_documents);
    cudaFree(d_queries);
    cudaFree(d_output);
    free(h_documents);
    free(h_queries);
    free(h_output);
    free(h_result); // Free h_result as well

    return 0;
}
