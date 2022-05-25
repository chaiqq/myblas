#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/matrix.h"
#include "include/matrix_operations.h"
#include "include/vector.h"

#define DIMENSION 3
#define MAX_NUM_STEPS 100
#define EPSILON 1.0e-20

int main() {
  // Use current time as seed for random generator
  srand(time(0));

  const Dimension matrix_dim = {DIMENSION, DIMENSION};

  // System Ax = b
  double **A = allocate_matrix(matrix_dim);
  double *b = allocate_vector(DIMENSION);
  double *x = allocate_vector(DIMENSION);

  double *p = allocate_vector(DIMENSION);   // search direction vector
  double *r = allocate_vector(DIMENSION);   // residual vector
  double *Ap = allocate_vector(DIMENSION);  // A*p

  double residual;  // Make sure starts bigger than EPSILON
  double alpha;
  double beta;

  const MatrixType spd_type = SPD;
  fill_matrix(A, matrix_dim, spd_type);
  const VectorType random_type = RANDOM;
  fill_vector(b, DIMENSION, random_type);

  print_matrix(A, matrix_dim);
  print_vector(b, DIMENSION);

  // First guess x_0 = {0, ..., 0}, therefore:
  //      p_0 = r_0 = b - A*x_0 = b
  for (int i = 0; i < DIMENSION; i++) {
    r[i] = b[i];
    p[i] = r[i];
  }

  int k = 0;
  double tmp = dot_product(r, r, DIMENSION);
  residual = tmp;
  while (residual > EPSILON && k < MAX_NUM_STEPS) {
    // TODO: Implement the algorithm described in the worksheet.
    // HINTS: There are already some methods implemented for you:
    //    * matrix_vector_mult
    //    * daxpy

    // Ap = A*p
    matrix_vector_mult(A, matrix_dim, p, Ap);
    // alpha = -tmp / <p,Ap> = <r,r> / <p,A*p>
    alpha = -tmp / dot_product(p, Ap, DIMENSION);
    // x_k+1 = x_k - alpha * p
    daxpy(-alpha, x, p, DIMENSION, x);
    // r_k+1 = r_k + alpha * A*p
    daxpy(alpha, r, Ap, DIMENSION, r);
    // res = ||r||^2 = <r_k+1, r_k+1>
    residual = dot_product(r, r, DIMENSION);
    // beta = <r_k+1, r_k+1> / <r,r>
    beta = residual / tmp;
    // temp = <r_k+1, r_k+1>
    tmp = residual;
    // p_k+1 = r_k+1 + beta * p_k
    daxpy(beta, r, p, DIMENSION, p);
    k++;
  }

  printf("Result took %d iterations\n", k);
  print_vector(x, DIMENSION);

  deallocate_matrix(A);
  free(b);
  free(x);
  free(p);
  free(r);
  free(Ap);
  return 0;
}
