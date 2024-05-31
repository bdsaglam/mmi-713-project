#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAX_POINTS 1000
#define MAX_DIMENSIONS 512
#define MAX_K 10

typedef struct {
    int id;
    double coordinates[MAX_DIMENSIONS];
} Point;

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

void generateRandomPoints(Point points[], int n_points, int n_dim) {
    srand(time(NULL));
    for (int i = 0; i < n_points; i++) {
        points[i].id = i + 1; // Assigning unique IDs starting from 1
        for (int j = 0; j < n_dim; j++) {
            points[i].coordinates[j] = (double)rand() / RAND_MAX * 100.0; // Random coordinate between 0 and 100
        }
    }
}

void printPoints(Point points[], int n_points, int n_dim) {
    for (int i = 0; i < n_points; i++) {
        printf("Point ID %d: (", points[i].id);
        for (int j = 0; j < n_dim; j++) {
            printf("%f", points[i].coordinates[j]);
            if (j < n_dim - 1) {
                printf(", ");
            }
        }
        printf(")\n");
    }
}

void printNeighbors(Point points[], int *neighbors, int k, int n_dim) {
    for (int i = 0; i < k; i++) {
        int index = neighbors[i];
        printf("Neighbor %d: ID %d (", i, points[index].id);
        for (int j = 0; j < n_dim; j++) {
            printf("%f", points[index].coordinates[j]);
            if (j < n_dim - 1) {
                printf(", ");
            }
        }
        printf(")\n");
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
