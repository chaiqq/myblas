#include "include/matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GET_INDEX(row, column, num_columns) ((column) + (row) * (num_columns))
#define MIN(val1, val2) ((val1) < (val2) ? val1 : val2)

void fill_matrix_random(double** matrix, const Dimension dimension);
void fill_matrix_indexed(double** matrix, const Dimension dimension);
void fill_matrix_zeroes(double** matrix, const Dimension dimension);
void fill_matrix_identity(double** matrix, const Dimension dimension);
void fill_matrix_sparse(double** matrix, const Dimension dimension);
void fill_matrix_spd(double** matrix, const Dimension dimension);

void remove_zeroes_diag(double** matrix, const Dimension dimension);
void make_matrix_diag_dominant(double** matrix, const Dimension dimension);

int random_int(const int min_value, const int max_value);
double random_double(const double min_value, const double max_value);

double** allocate_matrix(const Dimension dimension) {
  const int num_elements = dimension.num_columns * dimension.num_rows;
  double** matrix = (double**)malloc(dimension.num_rows * sizeof(double*));
  double* buffer = (double*)malloc(num_elements * sizeof(double));

  for (int i = 0; i < dimension.num_rows; i++) {
    matrix[i] = &buffer[i * dimension.num_columns];
  }

  return matrix;
}

void deallocate_matrix(double** matrix) {
  // The matrix has a contiguous memory, which is why only one call
  // to matrix[0] is required, instead of one free for each row (i.e.,
  // free(matrix[i]), i = 0...num_rows)
  free(matrix[0]);
  free(matrix);
}

void fill_matrix(double** matrix, const Dimension dimension,
                 const MatrixType type) {
  switch (type) {
    case RANDOM:
      fill_matrix_random(matrix, dimension);
      break;

    case INDEXED:
      fill_matrix_indexed(matrix, dimension);
      break;

    case ZEROES:
      fill_matrix_zeroes(matrix, dimension);
      break;

    case IDENTITY:
      fill_matrix_identity(matrix, dimension);
      break;

    case SPARSE:
      fill_matrix_sparse(matrix, dimension);
      break;

    case SPD:
      fill_matrix_spd(matrix, dimension);
      break;

    default:
      printf("[ERROR] fill_matrix() Unknown matrix type.\n");
  }
}

void print_matrix(double** matrix, const Dimension dimension) {
  printf("\n");
  for (int i = 0; i < dimension.num_rows; i++) {
    printf("[ ");
    for (int j = 0; j < dimension.num_columns; j++) {
      printf(" %lf", matrix[i][j]);
    }
    printf(" ]\n");
  }
  printf("\n");
}

void multiply_matrices(double** matrix_A, const Dimension dimension_A,
                       double** matrix_B, const Dimension dimension_B,
                       double** result) {
  if (dimension_A.num_columns == dimension_B.num_rows) {
    for (int i = 0; i < dimension_A.num_rows; i++) {
      for (int j = 0; j < dimension_B.num_columns; j++) {
        for (int k = 0; k < dimension_A.num_columns; k++) {
          result[i][j] += matrix_A[i][k] * matrix_B[k][j];
        }
      }
    }
  } else {
    printf(
        "[ERROR] multiply_matrices(). Matrix A (%dx%d) cannot be multiplied "
        "with matrix B (%dx%d)\n",
        dimension_A.num_rows, dimension_A.num_columns, dimension_B.num_rows,
        dimension_B.num_rows);
    result = NULL;
  }
}

void modify_matrix(double** matrix, const Dimension dimension,
                   const ModificationType modification) {
  switch (modification) {
    case REMOVE_ZEROES_DIAG:
      remove_zeroes_diag(matrix, dimension);
      break;

    case DIAGONAL_DOMINANT:
      make_matrix_diag_dominant(matrix, dimension);
      break;

    default:
      printf("[ERROR] modify_matrix() Unknown modification type\n");
  }
}

void fill_matrix_random(double** matrix, const Dimension dimension) {
  for (int i = 0; i < dimension.num_rows; i++) {
    for (int j = 0; j < dimension.num_columns; j++) {
      matrix[i][j] = random_double(0, 1);
    }
  }
}

void fill_matrix_indexed(double** matrix, const Dimension dimension) {
  for (int i = 0; i < dimension.num_rows; i++) {
    for (int j = 0; j < dimension.num_columns; j++) {
      matrix[i][j] = GET_INDEX(i, j, dimension.num_columns);
    }
  }
}

void fill_matrix_zeroes(double** matrix, const Dimension dimension) {
  for (int i = 0; i < dimension.num_rows; i++) {
    for (int j = 0; j < dimension.num_columns; j++) {
      matrix[i][j] = 0;
    }
  }
}

void fill_matrix_identity(double** matrix, const Dimension dimension) {
  for (int i = 0; i < dimension.num_rows; i++) {
    for (int j = 0; j < dimension.num_columns; j++) {
      if (i == j)
        matrix[i][j] = 1;
      else
        matrix[i][j] = 0;
    }
  }
}

void fill_matrix_sparse(double** matrix, const Dimension dimension) {
  const double percentage_non_zero = 0.3;
  const int num_elements = dimension.num_columns * dimension.num_rows;
  const int num_non_zero = percentage_non_zero * num_elements;
  const int num_non_zero_diagonal = num_non_zero / 3;
  const int num_elements_diagonal =
      MIN(dimension.num_rows, dimension.num_columns);

  for (int k = 0; k < num_non_zero_diagonal; k++) {
    const int i = random_int(0, num_elements_diagonal - 1);
    matrix[i][i] = random_double(0, 1.0);
  }

  for (int k = 0; k < num_non_zero - num_non_zero_diagonal; k++) {
    const int i = random_int(0, dimension.num_rows - 1);
    const int j = random_int(0, dimension.num_columns - 1);
    matrix[i][j] = random_double(0, 1.0);
  }
}

void remove_zeroes_diag(double** matrix, const Dimension dimension) {
  const int min_dimension = MIN(dimension.num_rows, dimension.num_columns);
  for (int i = 0; i < min_dimension; i++) {
    if (matrix[i][i] == 0) matrix[i][i] = random_double(0.1, 1);
  }
}

void fill_matrix_spd(double** matrix, const Dimension dimension) {
  if (dimension.num_rows == dimension.num_columns) {
    // If a matrix is strictly diagonal dominant and all its diagonal
    // elements are positive, then the real parts of its eigenvalues
    // are positive
    for (int i = 0; i < dimension.num_rows; i++) {
      for (int j = i; j < dimension.num_columns; j++) {
        const double rand_number = random_double(0, 1);
        matrix[i][j] = rand_number;
        matrix[j][i] = rand_number;
      }
    }

    const ModificationType diagonal_dominant = DIAGONAL_DOMINANT;
    modify_matrix(matrix, dimension, diagonal_dominant);

  } else {
    printf(
        "[ERROR] fill_matrix_spd() Non-square matrices cannot be symmetric\n");
  }
}

void make_matrix_diag_dominant(double** matrix, const Dimension dimension) {
  const int num_elements_diagonal =
      MIN(dimension.num_rows, dimension.num_columns);
  // If we assume that all matrix values are between 0.1 and 1, then it is
  // enough to add num_columns to each diagonal element to guarantee
  // diagonal dominance.
  const double factor = 2 * dimension.num_columns;
  for (int k = 0; k < num_elements_diagonal; k++) {
    matrix[k][k] += factor;
  }
}

double random_double(const double min_value, const double max_value) {
  return min_value +
         ((double)rand() / (double)RAND_MAX) * (max_value - min_value);
}

int random_int(const int min_value, const int max_value) {
  // We add 0.99999 to increase the prob. of having max_value
  return (int)random_double((double)min_value, (double)max_value + 0.999999);
}
