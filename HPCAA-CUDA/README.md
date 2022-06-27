# HPCAA-CUDA

## Overview
This repository is an exercise collection and extension of the course "High Performance Computing - Algorithms and Applications" at Technical University of Munich. CUDA is mainly used. See course page https://campus.tum.de/tumonline/wbLv.wbShowLVDetail?pStpSpNr=950490233&pSpracheNr=2 for more details. This repository is only for practising purpose and personal usage.

## Content
- Dense Linear Algebra-mainly matrix-matrix multiplication
  - usual
  - tiled
  - coalesced
  - overlapped
  - cuBLAS
- Sparse Linear Algebra
  - sparse matrix vector multiplication with CSR format
  - cuSPARSE, cuBLAS, ELLPACK(small exercises)

Notice: we use column-major to store matrices!

## Run
To run the code, make sure you have the following libraries installed:
- CMake at least v3.5
- Boost v1.65.1
- GTest and GMock v1.10.0
- libpython3-dev
- Pybind11 v2.5.0

Usually comes with any Linux distribution by default: 
- gcc at least 5.5.0
- python3 
- libjpeg 
- OpenMP (pthreads)

To run the test

`<build>/tests --gtest_filter=MatMult.GPU_GLOBAL`