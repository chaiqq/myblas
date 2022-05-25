#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/matrix.h"

#define DIMENSION 6
#define COLUMN_TO_RANK(column, num_proc) ((column) % (num_proc))

MPI_Datatype MPI_global_column;
MPI_Datatype MPI_local_column;

Dimension get_local_dimensions(const Dimension global_dimension,
                               const int num_proc, const int rank);

void scatter_matrix(double **origin, const Dimension global_matrix_dim,
                    const int num_proc, double **destiny);

void gather_matrix(double **origin, const Dimension global_matrix_dim,
                   const int num_proc, double **destiny);

void solve_parallel(double **local_matrix,
                    const Dimension dimensions_global_matrix, const int rank,
                    const int num_proc);

/*
 * Returns 1 if the column was assigned to the rank, 0 otherwise.
 */
int has_column(const int column, const int rank, const int num_proc);

int main(int argc, char *argv[]) {
  int rank, num_proc;
  double **local_matrix;
  double **global_matrix;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const Dimension dimension_global_matrix = {DIMENSION, DIMENSION};
  const Dimension dimension_local_matrix =
      get_local_dimensions(dimension_global_matrix, num_proc, rank);

  local_matrix = allocate_matrix(dimension_local_matrix);

  // Create datatypes
  MPI_Type_vector(dimension_local_matrix.num_rows, 1,
                  dimension_global_matrix.num_columns, MPI_DOUBLE,
                  &MPI_global_column);
  /*
   * MPI_Type_vector(int repeat_times, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
   * assume, local_num_rows = 5, global_num_cols = 4
   * We can get
   * x . . .
   * x . . .
   * x . . .
   * x . . .
   * x . . .
   * x represents "filled", . represents "blank";
   * The column with x is what myrank will use
   */
  MPI_Type_commit(&MPI_global_column);

  MPI_Type_vector(dimension_local_matrix.num_rows, 1,
                  dimension_local_matrix.num_columns, MPI_DOUBLE,
                  &MPI_local_column);
  MPI_Type_commit(&MPI_local_column);

  if (rank == 0) {
    global_matrix = allocate_matrix(dimension_global_matrix);
    const MatrixType random_type = RANDOM;
    fill_matrix(global_matrix, dimension_global_matrix, random_type);

    // This algorithm only works if there are no 0 on the diagonal
    const ModificationType no_zeroes_diag = REMOVE_ZEROES_DIAG;
    modify_matrix(global_matrix, dimension_global_matrix, no_zeroes_diag);
    print_matrix(global_matrix, dimension_global_matrix);
  } else {
    // We have to initialize global matrix on all ranks, or we could ran
    // into a segmentation fault.
    const Dimension dummy_dimension = {1, 1};
    global_matrix = allocate_matrix(dummy_dimension);
  }

  scatter_matrix(global_matrix, dimension_global_matrix, num_proc,
                 local_matrix);
  solve_parallel(local_matrix, dimension_global_matrix, rank, num_proc);
  gather_matrix(local_matrix, dimension_global_matrix, num_proc, global_matrix);

  if (rank == 0) {
    print_matrix(global_matrix, dimension_global_matrix);
  }

  MPI_Type_free(&MPI_global_column);
  MPI_Type_free(&MPI_local_column);

  deallocate_matrix(local_matrix);
  deallocate_matrix(global_matrix);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}

Dimension get_local_dimensions(const Dimension global_dimension,
                               const int num_proc, const int rank) {
  Dimension local_dimension;
  local_dimension.num_rows = global_dimension.num_rows;
  // Assign remaining columns to lower ranks (i.e., some ranks could have one
  // more column than others)
  local_dimension.num_columns =
      global_dimension.num_columns / num_proc +
      (rank < global_dimension.num_columns % num_proc);
  return local_dimension;
}
// 比如, global 共17列, rank 0 的 local_dim = 5, rank 1 dim = 4, 相当于每个rank接管几列

void scatter_matrix(double **origin, const Dimension global_matrix_dim,
                    const int num_proc, double **destiny) {
  const int SEND_TAG = 0;

  int rank;
  MPI_Request request;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const Dimension local_matrix_dim =
      get_local_dimensions(global_matrix_dim, num_proc, rank);

  if (rank == 0) {
    for (int col = 0; col < global_matrix_dim.num_columns; col++)
      MPI_Isend(origin[0] + col, 1, MPI_global_column,
                COLUMN_TO_RANK(col, num_proc), SEND_TAG + col / num_proc,
                MPI_COMM_WORLD, &request);
  }

  for (int col = 0; col < local_matrix_dim.num_columns; col++)
    MPI_Recv(destiny[0] + col, 1, MPI_local_column, 0, SEND_TAG + col,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void gather_matrix(double **origin, const Dimension global_matrix_dim,
                   const int num_proc, double **destiny) {
  const int SEND_TAG = 0;
  int rank;
  MPI_Request request;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const Dimension local_matrix_dim =
      get_local_dimensions(global_matrix_dim, num_proc, rank);

  for (int col = 0; col < local_matrix_dim.num_columns; col++)
    MPI_Isend(origin[0] + col, 1, MPI_local_column, 0,
              SEND_TAG + (rank + num_proc * col), MPI_COMM_WORLD, &request);

  if (rank == 0) {
    for (int col = 0; col < global_matrix_dim.num_columns; col++)
      MPI_Recv(destiny[0] + col, 1, MPI_global_column,
               COLUMN_TO_RANK(col, num_proc), SEND_TAG + col, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  }
}

void solve_parallel(double **local_matrix, const Dimension global_matrix_dim,
                    const int rank, const int num_proc) {
  int local_c;
  double *l;

  const Dimension local_matrix_dim =
      get_local_dimensions(global_matrix_dim, num_proc, rank);
  l = (double *)malloc(global_matrix_dim.num_columns * sizeof(double));
  /* This parallel implementation is doing following things:
   * 1. We have (num_col - 1) rounds, in each round:
   *    the rank who has the column computes the l coefficients of this column,
   * 2. the rank broadcasts this l to all other ranks
   * 3. all ranks update all entries
   */

  // Looping over columns is equivalent to loop over k (worksheet)
  for (int c = 0; c < global_matrix_dim.num_columns - 1; c++) {
    // The rank with column c is in charge of calculating vector l
    if (has_column(c, rank, num_proc)) {
      local_c = c / num_proc;
      // Vector l is calculated for elements below main diagonal, hence the
      // loop starts at r = c+1
      for (int r = c + 1; r < global_matrix_dim.num_rows; r++) {
        // r - (c+1) is used to start index at 0
        l[r - (c + 1)] = local_matrix[r][local_c] / local_matrix[c][local_c];
      }
    }

    MPI_Bcast(l, global_matrix_dim.num_rows - (c + 1), MPI_DOUBLE,
              COLUMN_TO_RANK(c, num_proc), MPI_COMM_WORLD);

    for (int r = c + 1; r < global_matrix_dim.num_rows; r++) {
      int start_column = (c / num_proc) + (COLUMN_TO_RANK(c, num_proc) > rank);
      for (int lc = start_column; lc < local_matrix_dim.num_columns; lc++) {
        local_matrix[r][lc] -= l[r - (c + 1)] * local_matrix[c][lc];
      }
    }
  }

  free(l);
}

int has_column(const int column, const int rank, const int num_proc) {
  return rank == COLUMN_TO_RANK(column, num_proc);
}
