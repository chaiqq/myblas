#include <stdio.h>
#include <stdlib.h>

#include "include/csr_extraction_format.h"

#define MIN(val1, val2) ((val1) < (val2) ? val1 : val2)

CSRExtractionFormat *make_csr_extraction_format(const int num_non_zero,
                                                const int num_non_zero_diagonal,
                                                const int length_diagonal) {
  CSRExtractionFormat *csr_extraction_format;
  const int num_non_zero_outside_diagonal =
      num_non_zero - num_non_zero_diagonal;
  const int total_length =
      (length_diagonal + 1) + num_non_zero_outside_diagonal;

  csr_extraction_format =
      (CSRExtractionFormat *)malloc(sizeof(CSRExtractionFormat));
  csr_extraction_format->length_diagonal = length_diagonal;
  csr_extraction_format->total_length = total_length;
  csr_extraction_format->values =
      (double *)malloc(total_length * sizeof(double));
  csr_extraction_format->indices = (int *)malloc(total_length * sizeof(int));

  return csr_extraction_format;
}

void free_csr_extraction_format(CSRExtractionFormat *csr_extraction_format) {
  free(csr_extraction_format->values);
  free(csr_extraction_format->indices);
  free(csr_extraction_format);
}

// TODO: Implement.
//    IN:
//      - matrix: 2D matrix.
//      - matrix_dim: matrix dimensions.
//      - num_non_zero
//      - num_non_zero_diagonal
//    OUT:
//      - A pointer to a CSRExtractionFormat representation of the matrix
CSRExtractionFormat *get_csr_extraction_format(
    double **matrix, const Dimension matrix_dim, const int num_non_zero,
    const int num_non_zero_diagonal) {

  const int length_diagonal = MIN(matrix_dim.num_columns, matrix_dim.num_rows);
  CSRExtractionFormat *csr_extraction_format = make_csr_extraction_format(
      num_non_zero, num_non_zero_diagonal, length_diagonal);
  double value;
  const int offset = length_diagonal + 1;
  int index_non_zero = 0;

  for (int i = 0; i < length_diagonal; i++) {
    csr_extraction_format->indices[i] = offset + index_non_zero;
    csr_extraction_format->values[i] = 0.0;
    for (int j = 0; j < matrix_dim.num_columns; j++) {
      if ((value = matrix[i][j]) != 0) {
        if (i == j) {
          csr_extraction_format->values[i] = value;
        } else {
          csr_extraction_format->values[offset + index_non_zero] = value;
          csr_extraction_format->indices[offset + index_non_zero] = j;
          index_non_zero++;
        }
      }
    }
  }
  csr_extraction_format->indices[matrix_dim.num_rows] = offset + index_non_zero;


  return csr_extraction_format;
}

// TODO: Implement.
//    IN:
//      - csr_extraction_format: pointer to a CSR Extraction Format representation
//      of a matrix.
//      - vector: vector to multiply with.
//    OUT:
//      - result: vector to store the result.
void multiply_csr_extraction_format_with_vector(
    CSRExtractionFormat *csr_extraction_format, double *vector,
    double *result) {

  for (int r = 0; r < csr_extraction_format->length_diagonal; r++) {
    result[r] = csr_extraction_format->values[r] * vector[r];
    for (int k = csr_extraction_format->indices[r];
         k < csr_extraction_format->indices[r + 1]; k++) {
      result[r] += csr_extraction_format->values[k] *
                   vector[csr_extraction_format->indices[k]];
    }
  }


}

void print_csr_extraction_format(CSRExtractionFormat *csr_extraction_format) {
  printf("| Values\t| indices\t|\n");
  const int length_diagonal = csr_extraction_format->length_diagonal;
  for (int i = 0; i < csr_extraction_format->length_diagonal; i++) {
    printf("| %lf\t| %d\t\t|\n", csr_extraction_format->values[i],
           csr_extraction_format->indices[i] + 1);
  }
  printf("| *\t\t| %d\t\t|\n", csr_extraction_format->indices[length_diagonal]);
  for (int k = length_diagonal + 1; k < csr_extraction_format->total_length;
       k++) {
    printf("| %lf\t| %d\t\t|\n", csr_extraction_format->values[k],
           csr_extraction_format->indices[k] + 1);
  }
  printf("\n");
}
