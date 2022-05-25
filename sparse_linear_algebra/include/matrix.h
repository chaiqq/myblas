#ifndef MATRIX_H
#define MATRIX_H

#include "buffer.h"

typedef BufferType MatrixType;

/*
 * We can modify our matrices as needed:
 *  - REMOVE_ZEROES_DIAG: Remove zeroes from diagonal and replace them
 *  with random values between 0 and 1.0.
 */
typedef enum ModificationType { REMOVE_ZEROES_DIAG } ModificationType;

typedef struct Dimension {
  int num_rows;
  int num_columns;
} Dimension;

/* Allocates memory for a matrix. The memory is contiguous. */
double** allocate_matrix(const Dimension dimension);

/* Frees the memory allocated for the matrix */
void deallocate_matrix(double** matrix);

void fill_matrix(double** matrix, const Dimension dimension,
                 const MatrixType type);

void print_matrix(double** matrix, const Dimension dimension);

/* The matrix result must be a pointer to a contiguous allocated block memory.
 * This method DOES NOT initialize result to 0s
 */
void multiply_matrices(double** matrix_A, const Dimension dimension_A,
                       double** matrix_B, const Dimension dimension_B,
                       double** result);

void modify_matrix(double** matrix, const Dimension dimension,
                   const ModificationType modification);

#endif
