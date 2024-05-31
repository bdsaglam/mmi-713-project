#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "common.h"

#define MAX_POINTS 1000
#define MAX_K 10

double distance(Point a, Point b, int dimensions) {
    double sum = 0.0;
    for (int i = 0; i < dimensions; i++) {
        sum += pow(a.coordinates[i] - b.coordinates[i], 2);
    }
    return sqrt(sum);
}

void sortDistances(double *distances, int *indices, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (distances[j] > distances[j + 1]) {
                double tempDist = distances[j];
                distances[j] = distances[j + 1];
                distances[j + 1] = tempDist;
                
                int tempIndex = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = tempIndex;
            }
        }
    }
}

void findTopKNeighbors(Point points[], int n, Point p, int k, int dimensions, int *topKIndices) {
    double distances[MAX_POINTS];
    int indices[MAX_POINTS];

    for (int i = 0; i < n; i++) {
        distances[i] = distance(points[i], p, dimensions);
        indices[i] = i;
    }

    sortDistances(distances, indices, n);

    for (int i = 0; i < k; i++) {
        topKIndices[i] = indices[i];
    }
}

int main() {
    int n_points = 10; // Number of points to generate
    int n_dim = 3; // Number of dimensions
    int k = 3; // Number of nearest neighbors to find

    Point points[MAX_POINTS];
    generateRandomPoints(points, n_points, n_dim);

    // printf("Generated Points:\n");
    // printPoints(points, n_points, n_dim);

    Point queryPoint = {9999, {50.0, 50.0, 50.0}}; // Example query point

    int topKIndices[MAX_K];
    findTopKNeighbors(points, n_points, queryPoint, k, n_dim, topKIndices);

    printf("\nTop-%d Nearest Neighbors:\n", k);
    printNeighbors(points, topKIndices, k, n_dim);

    return 0;
}
