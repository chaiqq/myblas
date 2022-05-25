#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/matrix.h"
#include "include/vector.h"

#define DIMENSION 5
#define WEIGHT 1.7
#define MAX_NUM_STEPS 100

int main() {
  // Use current time as seed for random generator
  srand(time(0));

  const Dimension matrix_dim = {DIMENSION, DIMENSION};

  // System Ax = b
  double **A = allocate_matrix(matrix_dim);
  double *b = allocate_vector(DIMENSION);
  double *x = allocate_vector(DIMENSION);

  const MatrixType random_matrix = RANDOM;
  const VectorType random_vector = RANDOM;
  fill_matrix(A, matrix_dim, random_matrix);
  fill_vector(b, DIMENSION, random_vector);

  // SOR works for diagonal dominant matrix
  const ModificationType diagonal_dominant = DIAGONAL_DOMINANT;
  modify_matrix(A, matrix_dim, diagonal_dominant);

  print_matrix(A, matrix_dim);
  print_vector(b, DIMENSION);

  double sum;
  for (int k = 0; k < MAX_NUM_STEPS; k++) {
    for (int i = 0; i < matrix_dim.num_rows; i++) {
      sum = 0.0;
      for (int j = 0; j < matrix_dim.num_columns; j++) {
        if (j != i) sum += A[i][j] * x[j];
      }
      x[i] = x[i] + WEIGHT * ((b[i] - sum) / A[i][i] - x[i]);
    }
  }

  print_vector(x, DIMENSION);

  deallocate_matrix(A);
  free(b);
  free(x);
  return 0;
}
