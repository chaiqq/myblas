#include "kernels.h"
#include <iostream>
#include <map>
#include <sstream>

#define WARP_SIZE 32

namespace err {
std::string PrevFile{};
int PrevLine{0};


void checkErr(const std::string &file, int line) {
#ifndef NDEBUG
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess) {
        std::stringstream stream;
        stream << '\n'
               << file << ", line " << line << ": " << cudaGetErrorString(Error) << " (" << Error
               << ")\n";
        if (PrevLine > 0) {
            stream << "Previous CUDA call:" << '\n' << PrevFile << ", line " << PrevLine << '\n';
        }
        throw std::runtime_error(stream.str());
    }
    PrevFile = file;
    PrevLine = line;
#endif
}

void checkCublasStatus(cublasStatus_t status, const std::string &file, int line) {
    static std::map<cublasStatus_t, std::string> cublasErrorMap{
        {CUBLAS_STATUS_SUCCESS, "CUBLAS_STATUS_SUCCESS"},
        {CUBLAS_STATUS_NOT_INITIALIZED, "CUBLAS_STATUS_NOT_INITIALIZED"},
        {CUBLAS_STATUS_ALLOC_FAILED, "CUBLAS_STATUS_ALLOC_FAILED"},
        {CUBLAS_STATUS_INVALID_VALUE, "CUBLAS_STATUS_INVALID_VALUE"},
        {CUBLAS_STATUS_ARCH_MISMATCH, "CUBLAS_STATUS_ARCH_MISMATCH"},
        {CUBLAS_STATUS_MAPPING_ERROR, "CUBLAS_STATUS_MAPPING_ERROR"},
        {CUBLAS_STATUS_EXECUTION_FAILED, "CUBLAS_STATUS_EXECUTION_FAILED"},
        {CUBLAS_STATUS_INTERNAL_ERROR, "CUBLAS_STATUS_INTERNAL_ERROR"}};

    if (status == CUBLAS_STATUS_SUCCESS) {
        return;
    } else {
        std::stringstream stream;
        stream << file << ", line " << line << ": ";
        if (cublasErrorMap.find(status) != cublasErrorMap.end()) {
            stream << "cublas returned with error: " << cublasErrorMap[status];
        } else {
            stream << "cublas returned with unknown error";
        }
        throw std::runtime_error(stream.str());
    }
}

void checkCusparseStatus(cusparseStatus_t status, const std::string &file, int line) {
    static std::map<cusparseStatus_t, std::string> cusparseErrorMap{
        {CUSPARSE_STATUS_SUCCESS, "CUSPARSE_STATUS_SUCCESS"},
        {CUSPARSE_STATUS_NOT_INITIALIZED, "CUSPARSE_STATUS_NOT_INITIALIZED"},
        {CUSPARSE_STATUS_ALLOC_FAILED, "CUSPARSE_STATUS_ALLOC_FAILED"},
        {CUSPARSE_STATUS_INVALID_VALUE, "CUSPARSE_STATUS_INVALID_VALUE"},
        {CUSPARSE_STATUS_ARCH_MISMATCH, "CUSPARSE_STATUS_ARCH_MISMATCH"},
        {CUSPARSE_STATUS_MAPPING_ERROR, "CUSPARSE_STATUS_MAPPING_ERROR"},
        {CUSPARSE_STATUS_EXECUTION_FAILED, "CUSPARSE_STATUS_EXECUTION_FAILED"},
        {CUSPARSE_STATUS_INTERNAL_ERROR, "CUSPARSE_STATUS_INTERNAL_ERROR"},
        {CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"}};

    if (status == CUSPARSE_STATUS_SUCCESS) {
        return;
    } else {
        std::stringstream stream;
        stream << file << ", line " << line << ": ";
        if (cusparseErrorMap.find(status) != cusparseErrorMap.end()) {
            stream << "cusparse returned with error: " << cusparseErrorMap[status];
        } else {
            stream << "cusparse returned with unknown error";
        }
        throw std::runtime_error(stream.str());
    }
}
} // namespace err


std::string getDeviceName() {
    int deviceId{-1};
    cudaGetDevice(&deviceId);

    cudaDeviceProp devProp{};
    cudaGetDeviceProperties(&devProp, deviceId);
    std::stringstream stream;

    stream << devProp.name << ", Compute Capability: " << devProp.major << '.' << devProp.minor;
    return stream.str();
}

size_t get1DGrid(size_t blockSize, size_t size) {
    return (size + blockSize - 1) / blockSize;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_csrMatVecMult(float *y, const DevCsrMatrix matrix, const float *x) {
  // TODO: T3.2a implement mat-vec multiplication
  // 每个thread负责一行的点积
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < matrix.numRows){
    float dot = 0.0f;
    for(int i = matrix.start[row]; i < matrix.start[row+1]; i++){
        dot += matrix.values[i] * x[matrix.indices[i]];
    }
    y[row] = dot;
  }
}


template <int TILE_SIZE>
__global__ void kernel_csrMatVecMult_vectorized(float *y, const DevCsrMatrix matrix, const float *x) {
  // TODO: H3.1 implement mat-vec multiplication
  // each warp does a row * vector multiplication for one row of matrix
  // 1. assign one warp to a matrix row
  // 2. allocate a shared arry vals[] for the partial results of a block
  // 3. compute one row*vec product in a loop. This time, parallelize the loop over all 32 threads in the warp
  // take care that access to the arrays j and a is coalesced
  // 4. use reduction (binary fan-in) to add up the partial sum in vals[], and add the output to the result vector y

  __shared__ float values[TILE_SIZE];

    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadId / WARP_SIZE;
    int lane = threadId & (WARP_SIZE - 1); // 相当于mod
    int row = warpId;

    if (row < matrix.numRows) {
        int rowStart = matrix.start[row];
        int rowEnd = matrix.start[row + 1];

        // compute running sum per thread
        values[threadIdx.x] = 0.0f;

        for (int j = rowStart + lane; j < rowEnd; j += WARP_SIZE) { // 可能一次完不成,eg 51列,warp需做两遍
            values[threadIdx.x] += matrix.values[j] * x[matrix.indices[j]];
        }

#if CUDART_VERSION > 9000
        __syncwarp();
#else
        __syncthreads();
#endif

        // parallel reduction in shared memory
        for (int d = WARP_SIZE >> 1; d >= 1; d >>= 1) {
            if (lane < d)
                values[threadIdx.x] += values[threadIdx.x + d];
#if CUDART_VERSION > 9000
            __syncwarp();
#else
            __syncthreads();
#endif
        }

        // first thread in a warp writes the result
        if (lane == 0) {
            y[row] = values[threadIdx.x];
        }
    }
}


void launch_csrMatVecMult(float *y, const DevCsrMatrix matrix, const float *x,
                       const ExecutionMode mode) {
    constexpr int TILE_SIZE = 64;
    switch (mode) {
        case ExecutionMode::PAGERANK: {
            // #threads = #rows (= N)
            // TODO: T3.2a define grid/block size
            // #threads = #rows (= N)
            dim3 grid(get1DGrid(TILE_SIZE, matrix.numRows), 1, 1);
            dim3 block(TILE_SIZE, 1, 1);
            kernel_csrMatVecMult<<<grid, block>>>(y, matrix, x);
            break;
        }
        case ExecutionMode::PAGERANK_VECTORIZED: {
            // TODO: H3.1 define grid/block size
            // #threads = #rows * #threads per row (= N * WARP_SIZE), each row done by a warp
            dim3 grid(get1DGrid(TILE_SIZE, matrix.numRows * WARP_SIZE), 1, 1);
            dim3 block(TILE_SIZE, 1, 1);
            kernel_csrMatVecMult_vectorized<TILE_SIZE><<<grid, block>>>(y, matrix, x);
            break;
        }
        default: {
            std::stringstream stream;
            stream << "Unknown execution mode #(" << mode << ") for page rank solver";
            throw std::runtime_error(stream.str());
        }
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_ellMatVecMult(float *y, const DevEllMatrix matrix, const float *x) {
  // TODO: T4.1a
  // 一个thread负责一行的点积
    const int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < matrix.numRows) {
        float dot = 0.0f;
        for (int i = 0; i < matrix.numColsPerRow; ++i) {
            int column = matrix.indices[row + i * matrix.numRows];
            float value = matrix.values[row + i * matrix.numRows];
            if (value != 0.0f) {
                dot += value * x[column];
            }
        }
        y[row] = dot;
    }
}


void launch_ellMatVecMult(float *y, const DevEllMatrix matrix, const float *x) {
  // TODO: T4.1a
    constexpr int TILE_SIZE = 64;
    dim3 grid(get1DGrid(TILE_SIZE, matrix.numRows), 1, 1);
    dim3 block(TILE_SIZE, 1, 1);

    kernel_ellMatVecMult<<<grid, block>>>(y, matrix, x);
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_bandMatVecMult(float *y, const DevBandMatrix matrix, const float *x) {
  // TODO: H5.1
   const int row = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < matrix.numRows) {
        float dot = 0.0f;

        for (int k = 0; k < (2 * matrix.halfSize + 1); ++k) {
            float value = matrix.values[row + k * matrix.numRows];
            int column = row + k - matrix.halfSize;

            if ((column >= 0) && (column < matrix.numRows)) {
              if (value != 0) {
                  dot += value * x[column];
              }
            }
        }
        y[row] = dot;
    }
}


void launch_bandMatVecMult(float *y, const DevBandMatrix matrix, const float *x) {
    // TODO: H5.1
    constexpr int TILE_SIZE = 64;
    //#threads = #rows (= N)
    dim3 grid(get1DGrid(TILE_SIZE, matrix.numRows), 1, 1);
    dim3 block(TILE_SIZE, 1, 1);

    kernel_bandMatVecMult<<<grid, block>>>(y, matrix, x);
    CHECK_ERR;
}