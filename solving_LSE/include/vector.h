#ifndef VECTOR_H
#define VECTOR_H

#include "buffer.h"

typedef BufferType VectorType;

double* allocate_vector(const int num_elements);

/* vector must already have allocated memory */
void fill_vector(double vector[], const int num_elements,
                 const VectorType type);

void sum_vectors(const double vector1[], const double vector2[],
                 const int num_elements, double result[]);

double dot_product(const double vector1[], const double vector2[],
                   const int num_elements);

void scalar_vector_mult(const double scalar, double vector[],
                        const int num_elements, double result[]);

/* It performs result = v1 + scalar * v2 */
void daxpy(const double scalar, double vector1[], double vector2[],
           const double num_elements, double result[]);

void print_vector(const double vector[], const int num_elements);


double norm_l2(const double vector1[], const double vector2[], const int num_elements);
#endif
