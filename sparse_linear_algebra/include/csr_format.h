#ifndef _CSR_FORMAT_H_
#define _CSR_FORMAT_H_

#include "matrix.h"

typedef struct CSRFormat {
  double *values;
  int *col_indices;
  int *non_zero_until_row;
  int num_non_zero;
  int num_rows;
} CSRFormat;

CSRFormat *make_csr_format(const int num_non_zero, const int num_rows);
void free_csr_format(CSRFormat *csr_format);
CSRFormat *get_csr_format(double **matrix, const Dimension matrix_dim,
                          const int num_non_zero);
// Result must be a vector filled with 0s
void multiply_csr_format_with_vector(CSRFormat *csr_format, double *vector,
                                     double *result);
void print_csr_format(CSRFormat *csr_format);

#endif
