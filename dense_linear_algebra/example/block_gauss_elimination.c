/*
 * This is a block-wise Gaussian Elimination for tridiagobal_matrices
 * A x = d, where A is tridiagonal
 * A = ( b1  c1 
 *       a2  b2  c2
 *           a3  b3  c3
 *               ...
 *                        )
 * For tridiagonal matrices, the LU-decomposition as in"lu_decomp.cpp" in column cyclic fashion
 * is not working well due to the data dependency.
 * So a better way is to split the rows for A into blocks. 
 * Thus we have neighbour[UPPER] or neighbour[LOWER] in the code
 * For the details of this algorithm please refer: 
 * https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjBvM-p0vr3AhVStqQKHWz6AuEQFnoECA4QAQ&url=https%3A%2F%2Fwww5.in.tum.de%2Flehre%2Fvorlesungen%2Fparnum%2FWS16%2Fsolution06.pdf&usg=AOvVaw2KDVthJfeuaXUE_-NteiLs
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/vector.h"

#define UPPER 0
#define LOWER 1

#define DIMENSION 12

MPI_Datatype MPI_row;

/*
 * Checks if the number of rows can be divided evenly between the number of
 * processors.
 */
void check_number_of_processors(const int rank, const int num_proc,
    const int dimension);

/*
 * Returns either the rank of the upper/lower neighbour or MPI_PROC_NULL.
 */
int get_upper_rank(const int rank);
int get_lower_rank(const int rank, const int num_proc);

int main(int argc, char *argv[]) {
  int num_proc, rank;
  double *local_low_diag, *local_main_diag, *local_upper_diag,
         *local_rhs;   // rhs = Right Hand Side
  double *local_x;  // solution
  double *global_low_diag, *global_main_diag, *global_upper_diag, *global_rhs,
         *global_x;
  MPI_Request requests[3];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  check_number_of_processors(rank, num_proc, DIMENSION);

  const int local_dimension = DIMENSION / num_proc;
  const int global_dimension = DIMENSION;
  int rank_neighbours[2];

  rank_neighbours[UPPER] = get_upper_rank(rank);
  rank_neighbours[LOWER] = get_lower_rank(rank, num_proc);

  local_low_diag = allocate_vector(local_dimension);
  local_main_diag = allocate_vector(local_dimension);
  local_upper_diag = allocate_vector(local_dimension);
  local_rhs = allocate_vector(local_dimension);
  local_x = allocate_vector(local_dimension);

  if (rank == 0) {
    global_low_diag = allocate_vector(global_dimension);
    global_main_diag = allocate_vector(global_dimension);
    global_upper_diag = allocate_vector(global_dimension);
    global_rhs = allocate_vector(global_dimension);
    global_x = allocate_vector(global_dimension);

    for (int i = 0; i < global_dimension; i++) {
      global_low_diag[i] = 1;
      global_main_diag[i] = 4;
      global_upper_diag[i] = 5;
      global_rhs[i] = 3;
    }
    // The lower and upper diagonal have 1 less element than the main one. To
    // simplify the scatter we have made their size equal, and then assign 0 to
    // the remaining elements of each one (i.e., the first one of
    // global_low_diag, and the last one of global_upper_diag)
    global_low_diag[0] = 0;
    global_upper_diag[global_dimension - 1] = 0;
  } else {
    const int dummy_dimension = 1;
    global_low_diag = allocate_vector(dummy_dimension);
    global_main_diag = allocate_vector(dummy_dimension);
    global_upper_diag = allocate_vector(dummy_dimension);
    global_rhs = allocate_vector(dummy_dimension);
    global_x = allocate_vector(dummy_dimension);
  }

  MPI_Scatter(global_low_diag, local_dimension, MPI_DOUBLE, local_low_diag,
      local_dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(global_main_diag, local_dimension, MPI_DOUBLE, local_main_diag,
      local_dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(global_upper_diag, local_dimension, MPI_DOUBLE, local_upper_diag,
      local_dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(global_rhs, local_dimension, MPI_DOUBLE, local_rhs,
      local_dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // ****************************************************************************
  // TODO: STEP 1: Elimination subdiagonal elements. Addition of an appropriate multiple
  // of the row above to element
  // Keep in mind that we only need to keep track of the changes in the main
  // diagonal and in the RHS because the lower diagonal is now eliminated, but notice:
  // in the blocks that are not the 1st block, there will be fill-ins
  // This is done parallely by different processors, sequentially inside each processor
  double factor = 0.0;
  for (int i = 0; i < local_dimension - 1; i++) {
    factor = local_low_diag[i + 1] / local_main_diag[i];
    local_main_diag[i + 1] -= local_upper_diag[i] * factor;
    local_rhs[i + 1] -= local_rhs[i] * factor;
    // We will use local_low_diag to store the new non-zero elements created as
    // result. We should not modify these values on the first block, but since
    // they are assumed to be 0 anyways it is okay to do it.
    local_low_diag[i + 1] = -local_low_diag[i] * factor; //store the fill-ins

  }
  // ****************************************************************************

  // ****************************************************************************
  // TODO: STEP 2: Elimination of upper diagonal elements

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // 2.1 First, we need to update the value of local_low_diag that will influence
  // our upper neighbour, and do the operations that are independent from
  // communication
  for (int i = local_dimension - 2; i > 0; i--) { 
    // start elimination from the second last row of each block so that can do this in parallel
    factor = local_upper_diag[i - 1] / local_main_diag[i];
    local_low_diag[i - 1] -= local_low_diag[i] * factor; // fill-ins
    local_rhs[i - 1] -= local_rhs[i] * factor;
    // Again, we will use local_upper_diag to store the new non-zero values
    local_upper_diag[i - 1] = -local_upper_diag[i] * factor;

  }
  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // 2.2 Second, we need to comunicate our first row to our upper neighbour
  double row[3];  // Temporal variables to income communication
  double buff_d;

  // We do not risk deadlock because one upper_rank will be MPI_PROC_NULL
  MPI_Send(&local_low_diag[0], 1, MPI_DOUBLE, rank_neighbours[UPPER], 0,
      MPI_COMM_WORLD);
  MPI_Recv(&row[0], 1, MPI_DOUBLE, rank_neighbours[LOWER], 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  MPI_Send(&local_main_diag[0], 1, MPI_DOUBLE, rank_neighbours[UPPER], 0,
      MPI_COMM_WORLD);
  MPI_Recv(&row[1], 1, MPI_DOUBLE, rank_neighbours[LOWER], 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  MPI_Send(&local_upper_diag[0], 1, MPI_DOUBLE, rank_neighbours[UPPER], 0,
      MPI_COMM_WORLD);
  MPI_Recv(&row[2], 1, MPI_DOUBLE, rank_neighbours[LOWER], 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  MPI_Send(&local_rhs[0], 1, MPI_DOUBLE, rank_neighbours[UPPER], 0,
      MPI_COMM_WORLD);
  MPI_Recv(&buff_d, 1, MPI_DOUBLE, rank_neighbours[LOWER], 0, MPI_COMM_WORLD,
      MPI_STATUS_IGNORE);

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  // 2.3 Third, use our lower neighbour's first row to update our last one row (except
  // for the last rank)
  if (rank != num_proc - 1) {
    factor = local_upper_diag[local_dimension - 1] / row[1];
    local_main_diag[local_dimension - 1] -= row[0] * factor;
    local_rhs[local_dimension - 1] -= buff_d * factor;
    // Again, we use global_upper_diag to store the new non-zero element(the fill-in)
    local_upper_diag[local_dimension - 1] = -row[2] * factor;
  }

  // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  // ****************************************************************************

  // ****************************************************************************
  // TODO: STEP 3: Transform the matrix in upper diagonal

  if (rank == 0) {
    // Rank 0 sends it last row to its lower rank. It does nothing more.
    MPI_Send(&local_main_diag[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[LOWER], 0, MPI_COMM_WORLD);
    MPI_Send(&local_upper_diag[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[LOWER], 0, MPI_COMM_WORLD);
    MPI_Send(&local_rhs[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[LOWER], 0, MPI_COMM_WORLD);

  } else {
    // Each rank (but root) waits to get the last row of their upper neighbour,
    // then updates its last row inmediatly, such that it can be sent to its
    // respective lower rank. While the communication is taking place it can
    // calculate the new inner values.
    MPI_Recv(&row[1], 1, MPI_DOUBLE, rank_neighbours[UPPER], 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
    MPI_Recv(&row[2], 1, MPI_DOUBLE, rank_neighbours[UPPER], 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
    MPI_Recv(&buff_d, 1, MPI_DOUBLE, rank_neighbours[UPPER], 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
    row[0] = 0;  // Initialize just in case we use it by mistake (bug easier to
    // find this way)

    factor = local_low_diag[local_dimension - 1] / row[1];
    local_main_diag[local_dimension - 1] -= row[2] * factor;
    local_rhs[local_dimension - 1] -= buff_d * factor;

    MPI_Isend(&local_main_diag[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[LOWER], 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&local_upper_diag[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[LOWER], 0, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(&local_rhs[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[LOWER], 0, MPI_COMM_WORLD, &requests[2]);

    for (int i = 0; i < local_dimension - 1; i++) {
      factor = local_low_diag[i] / row[1];
      local_upper_diag[i] -= row[2] * factor;
      local_rhs[i] -= buff_d * factor;
    }

  }

  // ****************************************************************************

  // ****************************************************************************
  // TODO: STEP 4: Transform the matrix to diagonal form.

  // First we send our last row to our upper neighbour, so the neighbour can update their
  // last row (keep in mind that the  only "visible" effect will be on
  // local_rhs). This needs to be done strictly from lower to upper rank, such
  // that my upper rank gets the updated local_rhs. Next we use our last row to
  // put zeroes above the main diagonal. Again, the only "visible" efect will be
  // on local_rhs

  row[0] = 0;
  row[1] = 0;
  row[2] = 0;
  if (rank == num_proc - 1) {
    MPI_Isend(&local_main_diag[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[UPPER], 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&local_rhs[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[UPPER], 0, MPI_COMM_WORLD, &requests[1]);

  } else {
    MPI_Recv(&row[1], 1, MPI_DOUBLE, rank_neighbours[LOWER], 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);
    MPI_Recv(&buff_d, 1, MPI_DOUBLE, rank_neighbours[LOWER], 0, MPI_COMM_WORLD,
        MPI_STATUS_IGNORE);

    factor = local_upper_diag[local_dimension - 1] / row[1];
    local_rhs[local_dimension - 1] -= buff_d * factor;

    MPI_Isend(&local_main_diag[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[UPPER], 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(&local_rhs[local_dimension - 1], 1, MPI_DOUBLE,
        rank_neighbours[UPPER], 0, MPI_COMM_WORLD, &requests[1]);
  }

  for (int i = 0; i < local_dimension - 1; i++) {
    factor = local_upper_diag[i] / local_main_diag[local_dimension - 1];
    local_rhs[i] -= local_rhs[local_dimension - 1] * factor;
  }

  // ****************************************************************************

  // FINAL STEP: Calculate the local solution and group them in rank 0
  for (int i = 0; i < local_dimension; i++) {
    local_x[i] = local_rhs[i] / local_main_diag[i];
  }

  MPI_Gather(local_x, local_dimension, MPI_DOUBLE, global_x, local_dimension,
      MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    print_vector(global_x, global_dimension);
  }

  free(local_low_diag);
  free(local_main_diag);
  free(local_upper_diag);
  free(local_rhs);
  free(local_x);
  free(global_low_diag);
  free(global_main_diag);
  free(global_upper_diag);
  free(global_rhs);
  free(global_x);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
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

int get_upper_rank(const int rank) {
  return rank != 0 ? rank - 1 : MPI_PROC_NULL;
}

int get_lower_rank(const int rank, const int num_proc) {
  return rank < num_proc - 1 ? rank + 1 : MPI_PROC_NULL;
}
