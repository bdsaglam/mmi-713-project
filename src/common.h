#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_DIMENSIONS 512

typedef struct {
    int id;
    double coordinates[MAX_DIMENSIONS];
} Point;

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


void generateRandomPoints(Point points[], int n_points, int n_dim) {
    for (int i = 0; i < n_points; i++) {
        points[i].id = i + 1; // Assigning unique IDs starting from 1
        for (int j = 0; j < n_dim; j++) {
            points[i].coordinates[j] = (double)rand() / RAND_MAX * 100.0; // Random coordinate between 0 and 100
        }
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


void randomInit(float* a, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            a[i*n_cols + j] = rand() / (float)RAND_MAX;
        }
    }
}