#ifndef _COORDINATE_FORM_H_
#define _COORDINATE_FORM_H_

#include "matrix.h"

typedef struct CoordinateForm {
  double *values;
  int *row_indices;
  int *col_indices;
  int length;
} CoordinateForm;

CoordinateForm *make_coordinate_form(const int num_non_zero);
void free_coordinate_form(CoordinateForm *coordinate_form);
CoordinateForm *get_coordinate_form(double **matrix, const Dimension matrix_dim,
                                    const int num_non_zero);
// Result must be a vector filled with 0s
void multiply_coordinate_form_with_vector(CoordinateForm *coordinate_form,
                                          double *vector, double *result);
void print_coordinate_form(CoordinateForm *coordinate_form);

#endif
