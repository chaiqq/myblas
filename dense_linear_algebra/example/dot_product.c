#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/vector.h"

#define VECTOR_LENGTH 10

/*
 * Checks if the number of rows can be divided evenly between the number of
 * processors.
 */
void check_number_of_processors(const int num_proc, const int dimension);

int main(int argc, char *argv[]) {
  int rank, num_proc;
  double *global_a, *global_b;
  double global_z;
  double *local_a, *local_b;
  double local_z;
  int num_elements;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  check_number_of_processors(num_proc, VECTOR_LENGTH);
  num_elements = VECTOR_LENGTH / num_proc;

  local_a = allocate_vector(num_elements);
  local_b = allocate_vector(num_elements);

  if (rank == 0) {
    global_a = allocate_vector(VECTOR_LENGTH);
    global_b = allocate_vector(VECTOR_LENGTH);

    const VectorType vec_type = RANDOM;
    fill_vector(global_a, VECTOR_LENGTH, vec_type);
    fill_vector(global_b, VECTOR_LENGTH, vec_type);
    printf("\na = ");
    print_vector(global_a, VECTOR_LENGTH);
    printf("b = ");
    print_vector(global_b, VECTOR_LENGTH);
  }

  MPI_Scatter(global_a, num_elements, MPI_DOUBLE, local_a, num_elements,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(global_b, num_elements, MPI_DOUBLE, local_b, num_elements,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  local_z = dot_product(local_a, local_b, num_elements);
  MPI_Reduce(&local_z, &global_z, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("z = ");
    printf(" %lf\n", global_z);
  }

  free(local_a);
  free(local_b);
  if (rank == 0) {
    free(global_a);
    free(global_b);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}

void check_number_of_processors(const int num_proc, const int num_elements) {
  if (num_elements % num_proc != 0) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      printf("%d elements cannot be divided evenly between %d processes\n",
             num_elements, num_proc);
    }
    fflush(stdout);
    fflush(stderr);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    exit(1);
  }
}
