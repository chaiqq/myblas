# Overview
This repository is an exercise collection and extension based on the course material _Parallel Numerics_ at _Technical University of Munich_. All codes are not guaranteed to be correct. Only for personal usage and practising purpose.

# Content
- Dense Linear Algebra:
  - matrix-matrix multiplication
  - LU decomposition
  - Block-wise Gaussian elimination for tridiagonal matrices
- Solving linear system equations(LSE)
  - Jacobi method
  - Red-black Gauss-Seidel method
  - Successive over-relaxation(SOR)
  - Conjugate gradient method
- Sparse Linear Algebra
  - Sparse matrix multiplication 

# Organization
In each folder, the codes are organized as follows:

 -CMakeLists.txt  
 -example  
 -include  
 -lib  
 -src 

You can find the main algorithms in the `example` folder. Most of the algorithms are implemented in parallel version using MPI. The rest are in sequential.

# Future Work
Make MPI implementation more universal eg: arbitrary dimensions, make it into libraries, make it into c++ form, combine CUDA and MPI, etc...

--By chaiqq, 2022.01.24