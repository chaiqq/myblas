#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "matrix.h"

/* 
 * If matrix has size RxC, then vector[] must have C elements, and
 * result should be able to store R elements. 
 */
void matrix_vector_mult(double** matrix, const Dimension matrix_dim,
                        double vector[], double result[]);

#endif
