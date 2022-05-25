#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/matrix.h"

#define DIMENSION 6
#define GET_INDEX(row, column, num_columns) ((column) + (row) * (num_columns))
#define COLUMN_TO_RANK(column, num_proc) ((column) % (num_proc))
#define SWITCH_VALUES(val1, val2, tmp) \
  {                                    \
    (tmp) = (val1);                    \
    (val1) = (val2);                   \
    (val2) = (tmp);                    \
  }

#define INDEX_SEND 0
#define INDEX_RECV 1

#define LEFT 0
#define RIGHT 1
#define UPPER 2
#define LOWER 3

MPI_Datatype MPI_submatrix, MPI_matrix;

typedef struct Coordinate {
  int row;
  int column;
} Coordinate;

/*
 * Checks if the number of rows can be divided evenly between the number of
 * processors.
 */
void check_number_of_processors(const int rank, const int num_proc,
                                const int dimension);

/*
 * Returns 0 if number is not a perfect square. 1 otherwise
 */
int is_perfect_square(const int number);

/* Given the index of an element in a 1D array, it returns the cartesian
 * coordinates if that buffer were representing a 2D matrix.
 */
Coordinate get_2D_coordinates(const int index_1D, const int num_rows,
                              const int num_columns);

/*
 * Given the coordinate (i,j) of an element in the submatrix that belongs to
 * the rank, it returns the global coordinate (I, J) with respect to the
 * global matrix
 */
Coordinate local_to_global_coordinates(const Coordinate local_coordinates,
                                       const Coordinate rank_coordinates,
                                       const Dimension dimension_local_matrix);

/*
 * Non-blocking shift. It sends data to rank_send and receives data from
 * rank_recv
 */
void Ishift_matrices(double **matrix_send, const int rank_send,
                     double **matrix_recv, const int rank_recv, const int tag,
                     MPI_Request *requests);

/*
 * It calculates the new coordinates if the point in origin is moved by the
 * values specified in displacenment
 */
Coordinate move_coordinates(const Coordinate origin,
                            const Coordinate displacenment,
                            const Dimension grid_dimensions);

Coordinate get_coordinates_left_neighbour(const Coordinate coordinates,
                                          const Dimension grid_dimensions);
Coordinate get_coordinates_right_neighbour(const Coordinate coordinates,
                                           const Dimension grid_dimensions);
Coordinate get_coordinates_upper_neighbour(const Coordinate coordinates,
                                           const Dimension grid_dimensions);
Coordinate get_coordinates_lower_neighbour(const Coordinate coordinates,
                                           const Dimension grid_dimensions);
void get_neighbours(const Coordinate rank_coordinates,
                    const Dimension grid_dimensions, int rank_neighbours[4]);

int main(int argc, char *argv[]) {
  int rank, num_proc;
  const int ROOT = 0;
  double **global_matrix_A, **global_matrix_B, **global_matrix_C;
  double **local_matrix_A, **local_matrix_B, **local_matrix_C;
  double **tmp_local_matrix_A, **tmp_local_matrix_B;
  double **tmp_pointer;
  const Dimension dimension_global_matrix = {DIMENSION, DIMENSION};
  int rank_neighbours[4];
  MPI_Request requests_shift_left[2];
  MPI_Request requests_shift_up[2];
  const int TAG_SHIFT_LEFT = 1;
  const int TAG_SHIFT_UP =
      TAG_SHIFT_LEFT + 1;  // Make sure that both tags are always different

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (!is_perfect_square(num_proc)) {
    printf("ERROR: %d num_proc is not a perfect square\n", num_proc);
    goto exit;
  }
  check_number_of_processors(rank, num_proc, dimension_global_matrix.num_rows);

  const int ranks_per_dimension = (int)sqrt(num_proc);
  const Dimension grid_dimensions = {ranks_per_dimension, ranks_per_dimension};
  const Dimension dimension_local_matrix = {
      dimension_global_matrix.num_rows / ranks_per_dimension,
      dimension_global_matrix.num_columns / ranks_per_dimension};
  const Coordinate rank_coordinates = get_2D_coordinates(
      rank, grid_dimensions.num_rows, grid_dimensions.num_columns);
  get_neighbours(rank_coordinates, grid_dimensions, rank_neighbours);

  local_matrix_A = allocate_matrix(dimension_local_matrix);
  local_matrix_B = allocate_matrix(dimension_local_matrix);
  local_matrix_C = allocate_matrix(dimension_local_matrix);
  tmp_local_matrix_A = allocate_matrix(dimension_local_matrix);
  tmp_local_matrix_B = allocate_matrix(dimension_local_matrix);

  // We need to make sure it starts with 0s, because the multiply_matrices
  // function does not do that for us
  const MatrixType type_zero_matrix = ZEROES;
  const MatrixType type_random_matrix = RANDOM;
  fill_matrix(local_matrix_C, dimension_local_matrix, type_zero_matrix);

  if (rank == ROOT) {
    global_matrix_A = allocate_matrix(dimension_global_matrix);
    global_matrix_B = allocate_matrix(dimension_global_matrix);
    global_matrix_C = allocate_matrix(dimension_global_matrix);

    fill_matrix(global_matrix_A, dimension_global_matrix, type_random_matrix);
    fill_matrix(global_matrix_B, dimension_global_matrix, type_random_matrix);

    printf("\nA =\n");
    print_matrix(global_matrix_A, dimension_global_matrix);
    printf("\nB =\n");
    print_matrix(global_matrix_B, dimension_global_matrix);
  } else {
    // If global matrix is not initialized on the other ranks, one will get a
    // segmentation fault when compiling the code with the optimization flags
    // -O1, -O2 and -O3
    const Dimension dummy_dimension = {1, 1};
    global_matrix_A = allocate_matrix(dummy_dimension);
    global_matrix_B = allocate_matrix(dummy_dimension);
    global_matrix_C = allocate_matrix(dummy_dimension);
  }

  // Create submatrix datatype
  MPI_Type_vector(
      dimension_local_matrix.num_rows, dimension_local_matrix.num_columns,
      dimension_global_matrix.num_columns, MPI_DOUBLE, &MPI_submatrix);
  // We are doing a scatter with non-contiguous data. This is always tricky.
  // Basically, we make MPI think that the extent of the MPI_submatrix is
  // actually sizeof(double). With this we can use scatterv with proper
  // displacenments to scatter the matrix. (see:
  // https://stackoverflow.com/questions/5512245/mpi-scatter-sending-columns-of-2d-array)
  MPI_Type_create_resized(MPI_submatrix, 0, 1 * sizeof(double), &MPI_submatrix);
  MPI_Type_commit(&MPI_submatrix);

  // *****************************************************************************
  // TODO: Create the MPI_matrix datatype. This will represent a matrix of
  // size dimension_local_matrix.num_rows * dimension_local_matrix.num_columns.
  // The data is contiguous in memory.
  MPI_Type_contiguous(
      dimension_local_matrix.num_rows * dimension_local_matrix.num_columns,
      MPI_DOUBLE, &MPI_matrix);
  MPI_Type_commit(&MPI_matrix);

  // *****************************************************************************

  // Scatter the matrices
  int *displacenments_A, *displacenments_B, *displacenments_C, *sendcounts;
  const Coordinate origin = {0, 0};
  Coordinate coordinate_first_element;
  Coordinate coordinate_rank_i;
  Coordinate coordinate_disp;

  // Cannon's algorithm requires an initial alignment. Instead of
  // communicating everything, and afterwards do shifts to get it, it is
  // more efficient to communicate the correct blocks from the beginning. That
  // is why we have 3 different displacenments vectors.
  displacenments_A = (int *)malloc(num_proc * sizeof(*displacenments_A));
  displacenments_B = (int *)malloc(num_proc * sizeof(*displacenments_B));
  displacenments_C = (int *)malloc(num_proc * sizeof(*displacenments_B)); //displacements_C?
  sendcounts = (int *)malloc(num_proc * sizeof(*sendcounts));
  for (int i = 0; i < num_proc; i++) {
    coordinate_rank_i =
        get_2D_coordinates(i, ranks_per_dimension, ranks_per_dimension);

    coordinate_disp.row = 0;
    coordinate_disp.column = coordinate_rank_i.row;
    coordinate_first_element = local_to_global_coordinates(
        origin,
        move_coordinates(coordinate_rank_i, coordinate_disp, grid_dimensions),
        dimension_local_matrix);
    displacenments_A[i] =
        GET_INDEX(coordinate_first_element.row, coordinate_first_element.column,
                  dimension_global_matrix.num_columns);

    coordinate_disp.column = 0;
    coordinate_disp.row = coordinate_rank_i.column;
    coordinate_first_element = local_to_global_coordinates(
        origin,
        move_coordinates(coordinate_rank_i, coordinate_disp, grid_dimensions),
        dimension_local_matrix);
    displacenments_B[i] =
        GET_INDEX(coordinate_first_element.row, coordinate_first_element.column,
                  dimension_global_matrix.num_columns);

    coordinate_first_element = local_to_global_coordinates(
        origin, coordinate_rank_i, dimension_local_matrix);
    displacenments_C[i] =
        GET_INDEX(coordinate_first_element.row, coordinate_first_element.column,
                  dimension_global_matrix.num_columns);

    sendcounts[i] = 1;
  }

  MPI_Scatterv(global_matrix_A[0], sendcounts, displacenments_A, MPI_submatrix,
               local_matrix_A[0], 1, MPI_matrix, 0, MPI_COMM_WORLD);
  MPI_Scatterv(global_matrix_B[0], sendcounts, displacenments_B, MPI_submatrix,
               local_matrix_B[0], 1, MPI_matrix, 0, MPI_COMM_WORLD);

  // Solve in parallel
  for (int num_blocks = 0; num_blocks < ranks_per_dimension - 1; num_blocks++) {
    // TODO: Implement the algorithm. You might use the Ishift_matrices() and
    // multiply_matrices functions.
    Ishift_matrices(local_matrix_A, rank_neighbours[LEFT], tmp_local_matrix_A,
                    rank_neighbours[RIGHT], TAG_SHIFT_LEFT,
                    requests_shift_left);
    Ishift_matrices(local_matrix_B, rank_neighbours[UPPER], tmp_local_matrix_B,
                    rank_neighbours[LOWER], TAG_SHIFT_UP, requests_shift_up);
    multiply_matrices(local_matrix_A, dimension_local_matrix, local_matrix_B,
                      dimension_local_matrix, local_matrix_C);
    MPI_Waitall(2, requests_shift_left, MPI_STATUS_IGNORE);
    MPI_Waitall(2, requests_shift_up, MPI_STATUS_IGNORE);
    SWITCH_VALUES(local_matrix_A, tmp_local_matrix_A, tmp_pointer);
    SWITCH_VALUES(local_matrix_B, tmp_local_matrix_B, tmp_pointer);

  }
  // We need to multiply the last block
  multiply_matrices(local_matrix_A, dimension_local_matrix, local_matrix_B,
                    dimension_local_matrix, local_matrix_C);

  int *recvcounts = sendcounts;
  MPI_Gatherv(local_matrix_C[0], 1, MPI_matrix, global_matrix_C[0], recvcounts,
              displacenments_C, MPI_submatrix, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("\n[%d] C =\n", rank);
    print_matrix(global_matrix_C, dimension_global_matrix);
  }

  MPI_Type_free(&MPI_submatrix);
  MPI_Type_free(&MPI_matrix);
  deallocate_matrix(local_matrix_A);
  deallocate_matrix(local_matrix_B);
  deallocate_matrix(local_matrix_C);
  deallocate_matrix(tmp_local_matrix_A);
  deallocate_matrix(tmp_local_matrix_B);
  deallocate_matrix(global_matrix_A);
  deallocate_matrix(global_matrix_B);
  deallocate_matrix(global_matrix_C);
  free(displacenments_A);
  free(displacenments_B);
  free(displacenments_C);
  free(sendcounts);
exit:
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}

void check_number_of_processors(const int rank, const int num_proc,
                                const int dimension) {
  if (dimension * dimension % num_proc != 0) {
    if (rank == 0) {
      printf(
          "%d entries cannot be divided evenly in square submatrices between "
          "%d processes\n",
          dimension * dimension, num_proc);
    }
    fflush(stdout);
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    exit(1);
  }
}

int is_perfect_square(const int number) {
  double sqrt_number = sqrt(number);
  int floor_sqrt = (int)sqrt_number;

  return (sqrt_number - floor_sqrt == 0);
}

Coordinate get_2D_coordinates(const int index_1D, const int num_rows,
                              const int num_columns) {
  Coordinate coordinate;
  if (index_1D < num_rows * num_columns) {
    coordinate.row =
        index_1D /
        num_columns;  // int division does not require floor operation
    coordinate.column = index_1D % num_rows;
  } else {
    printf("WARNING: Index %d is outside of grid of dimensions %dx%d\n",
           index_1D, num_rows, num_columns);
    coordinate.row = -1;
    coordinate.column = -1;
  }
  return coordinate;
}

Coordinate local_to_global_coordinates(const Coordinate local_coordinates,
                                       const Coordinate rank_coordinates,
                                       const Dimension dimension_local_matrix) {
  Coordinate global_coordinate;

  global_coordinate.column =
      local_coordinates.column +
      rank_coordinates.column * dimension_local_matrix.num_columns;
  global_coordinate.row =
      local_coordinates.row +
      rank_coordinates.row * dimension_local_matrix.num_rows;

  return global_coordinate;
}

void Ishift_matrices(double **matrix_send, const int rank_send,
                     double **matrix_recv, const int rank_recv, const int tag,
                     MPI_Request *requests) {
  MPI_Isend(matrix_send[0], 1, MPI_matrix, rank_send, tag, MPI_COMM_WORLD,
            &requests[INDEX_SEND]);
  MPI_Irecv(matrix_recv[0], 1, MPI_matrix, rank_recv, tag, MPI_COMM_WORLD,
            &requests[INDEX_RECV]);
}

Coordinate move_coordinates(const Coordinate origin,
                            const Coordinate displacenment,
                            const Dimension grid_dimensions) {
  Coordinate destiny;
  destiny.row = (origin.row + displacenment.row) % grid_dimensions.num_rows;
  destiny.column =
      (origin.column + displacenment.column) % grid_dimensions.num_columns;

  destiny.row =
      destiny.row < 0 ? grid_dimensions.num_rows + destiny.row : destiny.row;
  destiny.column = destiny.column < 0
                       ? grid_dimensions.num_columns + destiny.column
                       : destiny.column;

  return destiny;
}

Coordinate get_coordinates_left_neighbour(const Coordinate coordinates,
                                          const Dimension grid_dimensions) {
  const Coordinate displacenment = {0, -1};
  return move_coordinates(coordinates, displacenment, grid_dimensions);
}

Coordinate get_coordinates_right_neighbour(const Coordinate coordinates,
                                           const Dimension grid_dimensions) {
  const Coordinate displacenment = {0, 1};
  return move_coordinates(coordinates, displacenment, grid_dimensions);
}

Coordinate get_coordinates_upper_neighbour(const Coordinate coordinates,
                                           const Dimension grid_dimensions) {
  const Coordinate displacenment = {-1, 0};
  return move_coordinates(coordinates, displacenment, grid_dimensions);
}

Coordinate get_coordinates_lower_neighbour(const Coordinate coordinates,
                                           const Dimension grid_dimensions) {
  const Coordinate displacenment = {1, 0};
  return move_coordinates(coordinates, displacenment, grid_dimensions);
}

void get_neighbours(const Coordinate rank_coordinates,
                    const Dimension grid_dimensions, int rank_neighbours[4]) {
  Coordinate coordinates =
      get_coordinates_left_neighbour(rank_coordinates, grid_dimensions);
  rank_neighbours[LEFT] = GET_INDEX(coordinates.row, coordinates.column,
                                    grid_dimensions.num_columns);
  coordinates =
      get_coordinates_right_neighbour(rank_coordinates, grid_dimensions);
  rank_neighbours[RIGHT] = GET_INDEX(coordinates.row, coordinates.column,
                                     grid_dimensions.num_columns);
  coordinates =
      get_coordinates_upper_neighbour(rank_coordinates, grid_dimensions);
  rank_neighbours[UPPER] = GET_INDEX(coordinates.row, coordinates.column,
                                     grid_dimensions.num_columns);
  coordinates =
      get_coordinates_lower_neighbour(rank_coordinates, grid_dimensions);
  rank_neighbours[LOWER] = GET_INDEX(coordinates.row, coordinates.column,
                                     grid_dimensions.num_columns);
}
