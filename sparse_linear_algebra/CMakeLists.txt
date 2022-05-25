cmake_minimum_required(VERSION 3.0)

project (sparse)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

find_package(MPI REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/include
)

add_library( LAutils SHARED 
	src/vector.c
	src/matrix.c
	src/coordinate_form.c
	src/csr_extraction_format.c
	src/csr_format.c
)

# change your project src here
add_executable(sparseMat example/sparse_matrix_mult.c)
#add_executable(dot_product.out example/dot_product.c)


target_link_libraries(sparseMat LAutils  ${MPI_C_INCLUDE_PATH} )


