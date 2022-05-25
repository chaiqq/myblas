cmake_minimum_required(VERSION 3.0)


project (densLA)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

find_package(MPI REQUIRED)

include_directories(
	${PROJECT_SOURCE_DIR}/include
)

add_library( LAutils SHARED 
	src/vector.c
	src/matrix.c
	src/matrix_operation.c
)

# change your project src here
add_executable(lu_decomp example/lu_decomp.c)
#add_executable(dot_product example/dot_product.c)


target_link_libraries(lu_decomp LAutils ${MPI_C_INCLUDE_PATH} )


