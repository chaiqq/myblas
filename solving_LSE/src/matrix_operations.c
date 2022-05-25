#include "include/matrix_operations.h"

void matrix_vector_mult(double** matrix, const Dimension matrix_dim,
                        double vector[], double result[]) {
  for (int i = 0; i < matrix_dim.num_rows; i++) {
    result[i] = 0;
    for (int j = 0; j < matrix_dim.num_columns; j++) {
      result[i] += matrix[i][j] * vector[j];
    }
  }
}
