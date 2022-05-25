#include <stdio.h>
#include <stdlib.h>

#include "include/coordinate_form.h"

CoordinateForm *make_coordinate_form(const int num_non_zero) {
  CoordinateForm *coordinate_form;

  coordinate_form = (CoordinateForm *)malloc(sizeof(CoordinateForm));
  coordinate_form->length = num_non_zero;
  coordinate_form->values = (double *)malloc(num_non_zero * sizeof(double));
  coordinate_form->row_indices = (int *)malloc(num_non_zero * sizeof(int));
  coordinate_form->col_indices = (int *)malloc(num_non_zero * sizeof(int));

  return coordinate_form;
}

void free_coordinate_form(CoordinateForm *coordinate_form) {
  free(coordinate_form->values);
  free(coordinate_form->row_indices);
  free(coordinate_form->col_indices);
  free(coordinate_form);
}

// TODO: Implement.
//    IN:
//      - matrix: 2D matrix.
//      - matrix_dim: matrix dimensions.
//      - num_non_zero: num_non_zero elements
//    OUT:
//      - A pointer to a CoordinateForm representation of the matrix
CoordinateForm *get_coordinate_form(double **matrix, const Dimension matrix_dim,
                                    const int num_non_zero) {

  CoordinateForm *coordinate_form = make_coordinate_form(num_non_zero);
  double value;
  int index_non_zero = 0;

  for (int i = 0; i < matrix_dim.num_rows; i++) {
    for (int j = 0; j < matrix_dim.num_columns; j++) {
      if ((value = matrix[i][j]) != 0) {
        coordinate_form->values[index_non_zero] = value;
        coordinate_form->row_indices[index_non_zero] = i;
        coordinate_form->col_indices[index_non_zero] = j;
        index_non_zero++;
      }
    }
  }


  return coordinate_form;
}

// TODO: Implement.
//    IN:
//      - coordinate_form: pointer to a Coordinate form representation
//      of a matrix.
//      - vector: vector to multiply with.
//    OUT:
//      - result: vector to store the result.
void multiply_coordinate_form_with_vector(CoordinateForm *coordinate_form,
                                          double *vector, double *result) {

  for (int i = 0; i < coordinate_form->length; i++) {
    result[coordinate_form->row_indices[i]] +=
        coordinate_form->values[i] * vector[coordinate_form->col_indices[i]];
  }


}

void print_coordinate_form(CoordinateForm *coordinate_form) {
  printf("| Values\t| Row indices\t| Col indices\t|\n");
  for (int i = 0; i < coordinate_form->length; i++) {
    printf("| %lf\t| %d\t\t| %d\t\t|\n", coordinate_form->values[i],
           coordinate_form->row_indices[i] + 1,
           coordinate_form->col_indices[i] + 1);
  }
  printf("\n");
}
