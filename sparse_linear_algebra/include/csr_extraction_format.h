#ifndef _CSR_EXTRACTED_FORMAT_H
#define _CSR_EXTRACTED_FORMAT_H

#include "matrix.h"

typedef struct CSRExtractionFormat {
  double *values;
  int *indices;
  int length_diagonal;
  int total_length;
} CSRExtractionFormat;

CSRExtractionFormat *make_csr_extraction_format(const int num_non_zero,
                                                const int num_non_zero_diagonal,
                                                const int length_diagonal);
void free_csr_extraction_format(CSRExtractionFormat *csr_extraction_format);
CSRExtractionFormat *get_csr_extraction_format(double **matrix,
                                               const Dimension matrix_dim,
                                               const int num_non_zero,
                                               const int num_non_zero_diagonal);
void multiply_csr_extraction_format_with_vector(
    CSRExtractionFormat *csr_extraction_format, double *vector, double *result);
void print_csr_extraction_format(CSRExtractionFormat *csr_extraction_format);

#endif
