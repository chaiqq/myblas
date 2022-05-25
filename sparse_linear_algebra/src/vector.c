#include "include/vector.h"

#include <stdio.h>
#include <stdlib.h>

void fill_vector_random(double vector[], const int num_elements);
void fill_vector_indexed(double vector[], const int num_elements);
void fill_vector_zeroes(double vector[], const int num_elements);

double *allocate_vector(const int num_elements) {
  return (double *)malloc(num_elements * sizeof(double));
}

void fill_vector(double vector[], const int num_elements,
                 const VectorType type) {
  switch (type) {
    case RANDOM:
      fill_vector_random(vector, num_elements);
      break;

    case INDEXED:
      fill_vector_indexed(vector, num_elements);
      break;

    case ZEROES:
      fill_vector_zeroes(vector, num_elements);
      break;

    default:
      printf("[ERROR] fill_vector() Type not supported for vectors.\n");
  }
}

void sum_vectors(const double vector1[], const double vector2[],
                 const int num_elements, double result[]) {
  for (int i = 0; i < num_elements; i++) {
    result[i] = vector1[i] + vector2[i];
  }
}

double dot_product(const double vector1[], const double vector2[],
                   const int num_elements) {
  double result = 0;
  for (int i = 0; i < num_elements; i++) {
    result += vector1[i] * vector2[i];
  }

  return result;
}

void print_vector(const double *const v, const int num_elements) {
  printf("[ ");
  for (int i = 0; i < num_elements; i++) {
    printf("%lf ", v[i]);
  }
  printf("]\n");
}

void fill_vector_random(double vector[], const int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    vector[i] = rand() / (double)RAND_MAX;
  }
}

void fill_vector_indexed(double vector[], const int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    vector[i] = i;
  }
}

void fill_vector_zeroes(double vector[], const int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    vector[i] = 0;
  }
}
