/*
 * solve Ax = b using Jacobi method
 * local block-row distribution of A
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/matrix.h"
#include "include/vector.h"

#define DIMENSION 18
#define TOL 0.0001
#define MAX_ITER 10000

MPI_Datatype MPI_local_matrix;

Dimension get_local_dimensions(const Dimension global_dimension, const int num_proc);
/*
 * Checks if the number of rows can be divided evenly between the number of
 * processors.
 */
void check_number_of_processors(const int rank, const int num_proc,
    const int dimension);
void Parallel_Jacobi(double** local_A, double *local_x, double *local_b, 
                    int global_dim, int max_iter, int num_proc, int my_rank);

int main(int argc, char *argv[]) {
  int num_proc, rank;
  double **local_A, *local_x, *local_b;
  double **global_A, *global_x, *global_b;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  check_number_of_processors(rank, num_proc, DIMENSION);

  const Dimension dimension_global_matrix = {DIMENSION, DIMENSION};
  const Dimension dimension_local_matrix =
      get_local_dimensions(dimension_global_matrix, num_proc);

  MPI_Type_vector(dimension_local_matrix.num_rows, dimension_local_matrix.num_columns,
                dimension_local_matrix.num_columns, MPI_DOUBLE,
                &MPI_local_matrix);
  MPI_Type_commit(&MPI_local_matrix);

  

  global_x = allocate_vector(dimension_global_matrix.num_columns);
  global_b = allocate_vector(dimension_global_matrix.num_columns);
  local_x = allocate_vector(dimension_local_matrix.num_rows);
  local_b = allocate_vector(dimension_local_matrix.num_rows);
  local_A = allocate_matrix(dimension_local_matrix);

  if (rank == 0) {
    global_A = allocate_matrix(dimension_global_matrix);
    const MatrixType random_type = RANDOM;
    fill_matrix(global_A, dimension_global_matrix, random_type);

    // This algorithm only works if there are no 0 on the diagonal
    const ModificationType no_zeroes_diag = REMOVE_ZEROES_DIAG;
    modify_matrix(global_A, dimension_global_matrix, no_zeroes_diag);
    print_matrix(global_A, dimension_global_matrix);

    fill_vector_random(global_b, dimension_global_matrix.num_columns);
    fill_vector_random(global_x, dimension_global_matrix.num_columns);
  } else {
    // We have to initialize global matrix on all ranks, or we could ran
    // into a segmentation fault.
    const Dimension dummy_dimension = {1, 1};
    global_A = allocate_matrix(dummy_dimension);
  }
  // the whole vector b and x are stored in each processor

    MPI_Bcast(&global_x, dimension_global_matrix.num_columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_b, dimension_global_matrix.num_columns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(&global_x, dimension_local_matrix.num_rows, MPI_DOUBLE, &local_x, 
                dimension_local_matrix.num_rows,MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(&global_b, dimension_local_matrix.num_rows, MPI_DOUBLE, &local_b, 
                dimension_local_matrix.num_rows,MPI_DOUBLE, 0, MPI_COMM_WORLD);            
    MPI_Scatter(&global_A, 1, MPI_local_matrix, &local_A, 1, MPI_local_matrix, 0, MPI_COMM_WORLD);

    Parallel_Jacobi(local_A, local_x, local_b, dimension_global_matrix.num_columns, num_proc, rank);

    MPI_Allgather(&local_x, dimension_local_matrix.num_rows, MPI_DOUBLE, 
                 &global_x, dimension_global_matrix.num_columns,, MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank ==0){
        print_vector(global_x, dimension_global_matrix.num_columns);
    }
    MPI_Type_free(&MPI_local_matrix);

    deallocate_matrix(local_A);
    deallocate_matrix(local_x);
    deallocate_matrix(local_b);
    deallocate_matrix(global_A);
    deallocate_matrix(global_x);
    deallocate_matrix(global_b);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    return 0;
}

Dimension get_local_dimensions(const Dimension global_dimension, const int num_proc){
    return {global_dimension / num_proc, global_dimension};
}
void check_number_of_processors(const int rank, const int num_proc,
    const int dimension) {
  if (dimension % num_proc != 0) {
    if (rank == 0) {
      printf("%d rows cannot be divided evenly between %d processes\n",
          dimension, num_proc);
    }
    fflush(stdout);
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    exit(1);
  }
}

void Parallel_Jacobi(double** local_A, double *local_x, double *local_b, 
                    int global_dim,  int num_proc, int my_rank){
    int i_local, i_diag, j, iter_num, local_dim;
    double x_temp1[DIMENSION], x_temp2[DIMENSION];
    double* x_old;
    double* x_new;

    local_dim = global_dim / num_proc;
    // Initialize local temporary x
    MPI_Allgather(b_local, local_dim, MPI_DOUBLE, x_temp1, local_dim,
                    MPI_DOUBLE, MPI_COMM_WORLD);
    x_new = x_temp1;
    x_old = x_temp2;
    iter_num = 0;
    while(iter_num < MAX_ITER && norm_l2(x_new, x_old,global_dim) >= TOL){
        iter_num++;
        swap(x_old, x_new);
        // run through own local part of vector
        for(i_local = 0; i_local < local_dim; i_local++){
            i_diag = i_local + my_rank * local_dim;

            // copy b to x
            local_x[i_local] = local_b[i_local];
            // first sum from 0 : diag-1
            for(int j = 0; j < i_diag; j++){
                local_x[i_local] -= local_A[i_local][j] * x_old[j];
            }
            // second sum from diag+1 : n
            for(int j = i_diag+1; j < global_dim; j++){
                local_x[i_local] -= local_A[i_local][j] * x_old[j];
            }
            // divide by diagonal coefficient
            local_x[i_local] /= local_A[i_local][i_diag];
        }
        // collect new global vector x_new, must be available for norm
        MPI_Allgather(local_x, local_dim, MPI_DOUBLE, x_new, local_dim, MPI_DOUBLE, MPI_COMM_WORLD);
    }

}

