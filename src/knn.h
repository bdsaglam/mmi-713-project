#include <cmath> // For fabsf

void computeL1Distance(float *documents, float *queries, float *output, int D, int N, int Q) {
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
void sumOverLastDim(float *distances, float *output, int D, int N, int Q) {
    for (int q = 0; q < Q; ++q) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                int index = (q * N + n) * D + d;
                sum += distances[index];
            }
            int outputIndex = q * N + n;
            output[outputIndex] = sum;
        }
    }
}

int* knn(float *documents, float *queries, int D, int N, int Q) {
    float *distances = (float *)malloc(Q * N * D * sizeof(float));
    float *results = (float *)malloc(Q * N * sizeof(float));

    computeL1Distance(documents, queries, distances, D, N, Q);
    sumOverLastDim(distances, results, D, N, Q);
    int* sorted_indices = argsort(results, Q, N);

    // Clean up memory
    free(distances);
    free(results);

    return sorted_indices;
}