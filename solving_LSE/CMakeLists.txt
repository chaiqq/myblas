cmake_minimum_required(VERSION 3.0)


project (solvingLSE)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

find_package(MPI REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/include
)

add_library( LSEutils SHARED 
	src/vector.c
	src/matrix.c
	src/matrix_operation.c
)

# change your project src here
add_executable(CG_parallel example/CG_parallel.c)
#add_executable(dot_product example/dot_product.c)


target_link_libraries( CG_parallel LSEutils  ${MPI_C_INCLUDE_PATH} )


