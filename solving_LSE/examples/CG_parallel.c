#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpi.h>

#include "include/matrix.h"
#include "include/matrix_operations.h"
#include "include/vector.h"

#define DIMENSION 8
#define MAX_NUM_STEPS 100
#define EPSILON 1.0e-20

int main(int argc, char *argv[]) {
  MPI_Datatype MPI_submatrix, MPI_matrix;

  double **A_global, *b_global, *x_global;
  double **A_local, *x_local;

  double *p_local, *r_local, *Ap_global, *Ap_local;

  double residual_global, residual_local, alpha, beta;

  const Dimension matrix_dim_global = {DIMENSION, DIMENSION};

  int rank, num_proc;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  if (DIMENSION % num_proc != 0) {
    if (rank == 0) printf("ERROR: DIMENSION mod num_proc != 0\n");
    goto exit;
  }

  // Use current time as seed for random generator
  srand(time(0));

  const Dimension matrix_dim_local = {DIMENSION, DIMENSION / num_proc};
  const int vector_length_local = DIMENSION / num_proc;

  if (rank == 0) {
    A_global = allocate_matrix(matrix_dim_global);
    b_global = allocate_vector(DIMENSION);
    x_global = allocate_vector(DIMENSION);

    const MatrixType spd_type = SPD;
    fill_matrix(A_global, matrix_dim_global, spd_type);
    const VectorType random_type = RANDOM;
    fill_vector(b_global, DIMENSION, random_type);
  } else {
    // We allocate some dummy space to avoid segmentation faults when
    // MPI calls from other ranks use A_global as parameter.
    const Dimension dummy_dimension = {1, 1};
    A_global = allocate_matrix(dummy_dimension);
    b_global = allocate_vector(1);
    x_global = allocate_vector(1);
  }

  if (rank == 0) {
    print_matrix(A_global, matrix_dim_global);
    print_vector(b_global, DIMENSION);
  }

  Ap_global = allocate_vector(DIMENSION);

  A_local = allocate_matrix(matrix_dim_local);
  x_local = allocate_vector(vector_length_local);
  p_local = allocate_vector(vector_length_local);
  r_local = allocate_vector(vector_length_local);
  Ap_local = allocate_vector(DIMENSION);

  // Create submatrix datatype
  MPI_Type_vector(matrix_dim_local.num_rows, matrix_dim_local.num_columns,
                  matrix_dim_global.num_columns, MPI_DOUBLE, &MPI_submatrix);
  // We are doing a scatter with non-contiguous data. This is always tricky.
  // Basically, we make MPI think that the extent of the MPI_submatrix is
  // actually sizeof(double). With this we can use scatterv with proper
  // displacenments to scatter the matrix. (see:
  // https://stackoverflow.com/questions/5512245/mpi-scatter-sending-columns-of-2d-array)
  MPI_Type_create_resized(MPI_submatrix, 0, 1 * sizeof(double), &MPI_submatrix);
  MPI_Type_commit(&MPI_submatrix);

  // Create matrix datatype
  MPI_Type_contiguous(matrix_dim_local.num_rows * matrix_dim_local.num_columns,
                      MPI_DOUBLE, &MPI_matrix);
  MPI_Type_commit(&MPI_matrix);

  // Scatter A
  int *sendcounts = (int *)malloc(num_proc * sizeof(*sendcounts));
  int *displacenments = (int *)malloc(num_proc * sizeof(*displacenments));

  for (int p = 0; p < num_proc; p++) {
    sendcounts[p] = 1;
    displacenments[p] = p * vector_length_local;
  }

  MPI_Scatterv(A_global[0], sendcounts, displacenments, MPI_submatrix, A_local[0], 1,
               MPI_matrix, 0, MPI_COMM_WORLD);

  // NOTE: In this case we are assuming x_0 = {0, ..., 0}, and therefore we only
  // need a scatter to get the initial values on the ranks (p = r = b - Ax_0 = b
  // -> scatter b). If this were not the case, there would be two options:
  //      1. Rank 0 scatter also x_0, then each rank can calculate its own
  //      portion of p and r
  //      2. Rank 0 calculates global p and r, and scatter their values

  MPI_Scatter(b_global, vector_length_local, MPI_DOUBLE, r_local,
              vector_length_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for (int i = 0; i < vector_length_local; i++) {
    p_local[i] = r_local[i];
  }

  int k = 0;
  double tmp = dot_product(r_local, r_local, vector_length_local);
  double tmp_inner_r_global;                 // <r, r>
  double inner_pAp_local, inner_pAp_global;  // local and global <p, Ap>

  MPI_Allreduce(&tmp, &tmp_inner_r_global, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
  residual_global = tmp_inner_r_global;
  while (residual_global > EPSILON && k < MAX_NUM_STEPS) {
    // *********************************************************************
    // TODO: Implement the main computation loop. You can use the serial code
    // as base. Consider carefully which operations can be done in parallel, and
    // which information each rank needs to perform it.
    // Ap = A*p parallel
    matrix_vector_mult(A_local, matrix_dim_local, p_local, Ap_local);
    MPI_Allreduce(Ap_local, Ap_global, DIMENSION, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    // Alpha
    inner_pAp_local = dot_product(p_local, Ap_global + displacenments[rank],
                                  vector_length_local);
    MPI_Allreduce(&inner_pAp_local, &inner_pAp_global, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    alpha = -tmp_inner_r_global / inner_pAp_global;

    // x_k+1 = x_k - alpha * p
    daxpy(-alpha, x_local, p_local, vector_length_local, x_local);
    // r_k+1 = r_k + alpha * A*p
    daxpy(alpha, r_local, Ap_global + displacenments[rank], vector_length_local,
          r_local);

    // Residual
    // res = ||r||^2 = <r_k+1, r_k+1>
    residual_local = dot_product(r_local, r_local, vector_length_local);
    MPI_Allreduce(&residual_local, &residual_global, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    // beta = <r_k+1, r_k+1> / <r,r>
    beta = residual_global / tmp_inner_r_global;
    // temp = <r_k+1, r_k+1>
    tmp_inner_r_global = residual_global;
    // p_k+1 = r_k+1 + beta * p_k
    daxpy(beta, r_local, p_local, vector_length_local, p_local);

    // *********************************************************************
    k++;
  }

  MPI_Gather(x_local, vector_length_local, MPI_DOUBLE, x_global,
             vector_length_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("Result took %d iterations\n", k);
    print_vector(x_global, DIMENSION);
  }
  MPI_Type_free(&MPI_submatrix);
  MPI_Type_free(&MPI_matrix);
  deallocate_matrix(A_global);
  free(Ap_global);
  free(b_global);
  free(x_global);
  deallocate_matrix(A_local);
  free(x_local);
  free(p_local);
  free(r_local);
  free(Ap_local);
exit:
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
