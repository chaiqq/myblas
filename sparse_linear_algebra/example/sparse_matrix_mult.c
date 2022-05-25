#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/coordinate_form.h"
#include "include/csr_extraction_format.h"
#include "include/csr_format.h"
#include "include/matrix.h"
#include "include/vector.h"

#define MIN(val1, val2) ((val1) < (val2) ? val1 : val2)

#define PRINT 1
#define DIMENSION 10

int get_num_non_zero_elements(double **matrix, const Dimension dimension);
int get_num_non_zero_elements_diag(double **matrix, const Dimension dimension);

// Result must be a vector filled with 0s
void matrix_vector_multiplication(double **matrix, double *vector,
                                  const Dimension matrix_dim, double *result);

int main() {
  clock_t time_start, time_end;

  // Use current time as seed for random generator
  srand(time(0));

  const Dimension matrix_dim = {DIMENSION, DIMENSION};

  double **matrix = allocate_matrix(matrix_dim);
  const MatrixType sparse_type = SPARSE;
  fill_matrix(matrix, matrix_dim, sparse_type);

  const int num_non_zero = get_num_non_zero_elements(matrix, matrix_dim);
  const int num_non_zero_diagonal =
      get_num_non_zero_elements_diag(matrix, matrix_dim);

  double *vector = allocate_vector(DIMENSION);
  const VectorType random_vec_type = RANDOM;
  fill_vector(vector, DIMENSION, random_vec_type);

  double *result = allocate_vector(DIMENSION);
  const VectorType zero_vec_type = ZEROES;
  fill_vector(result, DIMENSION, zero_vec_type);


  time_start = clock();
  matrix_vector_multiplication(matrix, vector, matrix_dim, result);
  time_end = clock();
  printf("Full matrix and vector multiplication: %ld clock ticks\n",
         time_end - time_start);

#if PRINT
  printf("A = \n");
  print_matrix(matrix, matrix_dim);
  printf("b = \n");
  print_vector(vector, DIMENSION);
  printf("Result full matrix - vector mult = \n");
  print_vector(result, DIMENSION);
#endif

  CoordinateForm *coordinate_form =
      get_coordinate_form(matrix, matrix_dim, num_non_zero);
  fill_vector(result, DIMENSION, zero_vec_type);

  time_start = clock();
  multiply_coordinate_form_with_vector(coordinate_form, vector, result);
  time_end = clock();
  printf("Coordinate form matrix and vector multiplication: %ld clock ticks\n",
         time_end - time_start);

#if PRINT
  printf("Coordiante Form = \n");
  print_coordinate_form(coordinate_form);
  printf("Result coordinate form matrix - vector mult = \n");
  print_vector(result, DIMENSION);
#endif

  CSRFormat *csr_format = get_csr_format(matrix, matrix_dim, num_non_zero);
  fill_vector(result, DIMENSION, zero_vec_type);

  time_start = clock();
  multiply_csr_format_with_vector(csr_format, vector, result);
  time_end = clock();
  printf("CSR format matrix and vector multiplication: %ld clock ticks\n",
         time_end - time_start);

#if PRINT
  printf("CSR Format = \n");
  print_csr_format(csr_format);
  printf("Result CSR format - vector mult = \n");
  print_vector(result, DIMENSION);
#endif

  CSRExtractionFormat *csr_extraction_format = get_csr_extraction_format(
      matrix, matrix_dim, num_non_zero, num_non_zero_diagonal);
  fill_vector(result, DIMENSION, zero_vec_type);

  time_start = clock();
  multiply_csr_extraction_format_with_vector(csr_extraction_format, vector,
                                             result);
  time_end = clock();
  printf(
      "CSR extraction format matrix and vector multiplication: %ld clock "
      "ticks\n",
      time_end - time_start);

#if PRINT
  printf("CSR Extraction Format = \n");
  print_csr_extraction_format(csr_extraction_format);
  printf("Result CSR extraction format - vector mult = \n");
  print_vector(result, DIMENSION);
#endif

  free_coordinate_form(coordinate_form);
  free_csr_format(csr_format);
  free_csr_extraction_format(csr_extraction_format);
  deallocate_matrix(matrix);
  free(vector);
  free(result);
  return 0;
}

void matrix_vector_multiplication(double **matrix, double *vector,
                                  const Dimension matrix_dim, double *result) {
  for (int c = 0; c < matrix_dim.num_columns; c++) {
    for (int r = 0; r < matrix_dim.num_rows; r++) {
      result[r] += matrix[r][c] * vector[c];
    }
  }
}

int get_num_non_zero_elements(double **matrix, const Dimension dimension) {
  int num_non_zero = 0;

  for (int i = 0; i < dimension.num_rows; i++) {
    for (int j = 0; j < dimension.num_columns; j++) {
      if (matrix[i][j] != 0) num_non_zero++;
    }
  }

  return num_non_zero;
}

int get_num_non_zero_elements_diag(double **matrix, const Dimension dimension) {
  int num_non_zero = 0;
  const int num_elements_diagonal =
      MIN(dimension.num_rows, dimension.num_columns);

  for (int k = 0; k < num_elements_diagonal; k++) {
    if (matrix[k][k] != 0) num_non_zero++;
  }

  return num_non_zero;
}
