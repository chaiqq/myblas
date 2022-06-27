#include "kernels.h"
#include "util.hpp"
#include <iostream>

namespace gpu {
size_t get1DGrid(size_t blockSize, size_t matrixSize) {
    // TODO: complete function
    // given matrixSize, n*n, compute how many blocks do we need
    // assume blocks are arranged in 1D
    // eg: get1DGrid(5, 16) = 4
    // ceiling(matrixSize / blockSize)'
    return (matrixSize + blockSize - 1) / blockSize;
    // -1 是避免 get1DGrid(4, 16)的情况, 多出一个block
    // size_t是个unsigned int
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_matrixMultGlobal(const float *devA, const float *devB, float *devC,
                                        const int size) {
    // 
    // TODO: complete function
    // A thread is in charge of 1 result element in devC
    // threads同时执行这个kernel
    // 目前我们的grid中只有一个block

    // 我是一个thread, 我在block中的相对坐标为(threadIdx.x, threadIdx.y), 我负责计算C矩阵的第row行,第col列
    // 我的绝对坐标需要结合我所在的block的坐标进行计算
    // 计算的过程是一个点乘
    // int row = threadIdx.y;
    // int col = threadIdx.x;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // NOTE: 注意索引的计算
    // 答案里的row和col怎么看怎么别扭,如果简单地直接映射topology到matrix上, 计算就简单了
    // 修改: 由于这里矩阵使用col-major, fastest-running的坐标是row, 所谓fastest,是相邻内存的内容. 因为要迎合cuBLAS的col major
    // A[i][j]的相邻内存是A[i+1][j], col-major不改变我们习惯的矩阵元素编号. 
    // fastest就是体现在i变动最快, 也就是row号先变动, 一列完成后,再变动col号
    // 在cuda中, thread编号是 (x,y),  (x+1,y)
    //                      (x,y+1),(x+1, y+1)
    // z这个编号应该fastest变的是x号, 所以要与矩阵的fastet对应
    // 当然row = ...y * ..y 那个也可以
    // 
    

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    // C(i, k) += A(i, j) * B(j, k), for j = 0 : size, i = row, k = col
    if((row < size) && (col < size)){
        float accumulator = 0.0f;

        for(int j = 0; j < size; ++j){
            accumulator += devA[j * size + row] * devB[col * size + j];
        }

        devC[col * size + row] += accumulator;
    }
}

void executeMatrixMultGlobal(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                             const Configuration &config) {
    for (int i = 0; i < config.numRepeats; ++i) {
        kernel_matrixMultGlobal<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void kernel_matrixMultTiled(const float *__restrict__ devA,
                                       const float *__restrict__ devB, float *__restrict__ devC,
                                       const size_t size) {
  // TODO: complete function
  __shared__ float shrA[TILE_SIZE][TILE_SIZE];
  __shared__ float shrB[TILE_SIZE][TILE_SIZE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE_SIZE + ty;
  const int col = blockIdx.x * TILE_SIZE + tx; // blockDim.x == TILE_SIZE 这里简化,设block和tile一样大
  

  // NOTE: 详细索引,循环设置的原理见ppt cuda_p3, 第8页, 第13页
  // 答案我感觉i和j的计算是错误的, 目前使用自己觉得舒服的计算
  // C22 = A21*B12 + A22*B22 + A23*B32; 这里的A21是一个tile, 这里有3个phase
  if((row < size) && (col < size)){
      float Celem = 0.0f;
      for(int phase = 0; phase < size / TILE_SIZE; ++phase) {// 假设整个矩阵大小正好整除tile size
        shrA[ty][tx] = devA[ (phase * TILE_SIZE + tx) * size + row ];
        shrB[ty][tx] = devB[ col * size + phase * TILE_SIZE + ty];
        __syncthreads();
        for(int k = 0; k < TILE_SIZE; k++){
            Celem += shrA[ty][k] * shrB[k][tx];
        }
        __syncthreads();
      }
      devC[col * size + row] = Celem;
  }

}
/* Tiled
Elapsed time : 0.00988173 s
operations: 2.68173e+08
Performance: 25.2745 GFlop/s
*/


void executeMatrixMultTiled(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                            const Configuration &config) {
    switch (config.tileSize) {
        case 4:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<4><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 8:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<8><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 16:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<16><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 32:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultTiled<32><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void kernel_matrixMultCoalesced(const float *__restrict__ devA,
                                           const float *__restrict__ devB, float *__restrict__ devC,
                                           const size_t size) {
  // TODO: complete function
  __shared__ float shrA[TILE_SIZE][TILE_SIZE];
  __shared__ float shrB[TILE_SIZE][TILE_SIZE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.x * TILE_SIZE + tx;
  const int col = blockIdx.y * TILE_SIZE + ty; // blockDim.x == TILE_SIZE 这里简化,设block和tile一样大
  // 这里的逻辑是, 在GPU中,tx是fastest running index, 要对应A的内存,colmajor的fastest running是沿着列
  // 这样子, 两个neighboring threads, 相差一个tx, 就能access 同样也是相邻的A的两个元素


  // 想要coalesced, 需要在同一个iter中时, 相邻的thread访问连续的一块内存, 而不是同一个thread在不同iter
  // 相邻的thread具有相邻的threadIdx.x, 所以要让x不能stride访问
  if((row < size) && (col < size)){
      float Celem = 0.0f;

      // 可以选择换一种mapping矩阵到block的方式, 想象成将block转置. 去顺应fastest running的那个维度的变化
      for(int phase = 0; phase < size / TILE_SIZE; ++phase) {// 假设整个矩阵大小正好整除tile size
        shrA[ty][tx] = devA[(phase * TILE_SIZE + ty) * size + row]; // access global的时候相邻的thread会合并访存
        shrB[ty][tx] = devB[col * size + phase * TILE_SIZE + tx];
        // 
        // load进来的shrA,shrB是原来的转置
        // 其实也可以shrA[tx][ty] = ... 这样load进来的就是原来形状的A 和B了,相应的celem +=也要改一下
        __syncthreads();
        // shared mem又是row major
        for(int k = 0; k < TILE_SIZE; k++){
            Celem += shrA[k][tx] * shrB[ty][k];
        }
        __syncthreads();
      }
      devC[col * size + row] = Celem;
  }
}
/*
Coalesced:
Elapsed time : 0.00659242 s
operations: 2.68173e+08
Performance: 37.8853 GFlop/s
*/

void executeMatrixMultCoalesced(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                                const Configuration &config) {
    switch (config.tileSize) {
        case 4:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<4><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 8:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<8><<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 16:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<16>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 32:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultCoalesced<32>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
    }
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
__global__ void kernel_matrixMultCoalescedDym(const float *__restrict__ devA,
                                              const float *__restrict__ devB,
                                              float *__restrict__ devC, const size_t size) {
  // TODO: complete function
  // shr mem size may be known only at run-time, so compute and pass it as the third parameter when launching the kernel
    const int TILE_SIZE = blockDim.x;
    extern __shared__ float shrA[];
    float *__restrict__ shrB = &shrA[TILE_SIZE * TILE_SIZE]; // shrB从shrA内存的一半处开始

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.x * TILE_SIZE + tx;
    const int column = blockIdx.y * TILE_SIZE + ty;
    if ((row < size) && (column < size)) {
        float Celem = 0.0f;

        for (int m = 0; m < size / TILE_SIZE; ++m) {
            // load tiles of A and B to the shared mem.
            shrA[tx + TILE_SIZE * ty] = devA[row + size * (ty + m * TILE_SIZE)]; //shrA[ty][tx]的一维
            shrB[tx + TILE_SIZE * ty] = devB[(tx + m * TILE_SIZE) + column * size];
            __syncthreads();

            for (int j = 0; j < TILE_SIZE; ++j)
                Celem += shrA[tx + TILE_SIZE * j] * shrB[j + TILE_SIZE * ty];
            __syncthreads();
        };

        devC[row + size * column] += Celem;
    }
}


void executeMatrixMultCoalescedDym(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                                   const Configuration &config) {
    const size_t shrMemSize = 2 * config.tileSize * config.tileSize * sizeof(float);
    for (int i = 0; i < config.numRepeats; ++i) {
        kernel_matrixMultCoalescedDym<<<dimGrid, dimBlock, shrMemSize>>>(Ad, Bd, Cd,
                                                                         config.matrixSize);
    }
    // NOTE: the 3rd parameter is shrMemSize
    CHECK_ERR;
}


//--------------------------------------------------------------------------------------------------
template <int TILE_SIZE>
__global__ void kernel_matrixMultOverlapped(const float *__restrict__ devA,
                                            const float *__restrict__ devB,
                                            float *__restrict__ devC, const size_t size) {
    // TODO: complete function, prefetch to register
    // load first tile into registers, copy registers to shared memory, barrier
    // load next tile into registers. compute current tile(shr mem), barrier
    // repeat above 2 steps
    // compute last tile
    __shared__ float shrA[TILE_SIZE][TILE_SIZE];
    __shared__ float shrB[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.x * TILE_SIZE + tx;
    const int column = blockIdx.y * TILE_SIZE + ty;
    if ((row < size) && (column < size)) {

        float Celem = 0.0f, Aelem = 0.0f, Belem = 0.0f;
        // load the first tile into registers
        Aelem = devA[row + size * ty];
        Belem = devB[tx + column * size];

        for (int m = 0; m < (size / TILE_SIZE) - 1; ++m) {
            // load tiles of A and B to the shared mem.
            shrA[ty][tx] = Aelem;
            shrB[ty][tx] = Belem;
            __syncthreads();

            // load the next tile to the registers
            Aelem = devA[row + size * (ty + (m + 1) * TILE_SIZE)];
            Belem = devB[(tx + (m + 1) * TILE_SIZE) + column * size];

            for (int j = 0; j < TILE_SIZE; ++j)
                Celem += shrA[j][tx] * shrB[ty][j];
            __syncthreads();
        };

        // compute the last tile
        const int m = (size / TILE_SIZE) - 1;
        shrA[ty][tx] = Aelem;
        shrB[ty][tx] = Belem;
        __syncthreads();
        for (int j = 0; j < TILE_SIZE; ++j)
          Celem += shrA[j][tx] * shrB[ty][j];

        devC[row + size * column] += Celem;
    }


}


void executeMatrixMultOverlapped(dim3 dimBlock, dim3 dimGrid, float *Ad, float *Bd, float *Cd,
                                 const Configuration &config) {
    switch (config.tileSize) {
        case 4:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<4>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 8:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<8>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 16:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<16>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
        case 32:
            for (int i = 0; i < config.numRepeats; ++i) {
                kernel_matrixMultOverlapped<32>
                    <<<dimGrid, dimBlock>>>(Ad, Bd, Cd, config.matrixSize);
            }
            break;
    }
    CHECK_ERR;
}

//--------------------------------------------------------------------------------------------------
void matrixMult(float *Ad, float *Bd, float *Cd, const Configuration &config) {
    // TODO: adjust dimBlock and dimGrid
    dim3 dimBlock(config.tileSize, config.tileSize); // tilesize = 8的话, 64 threads
    const size_t Grid1D = get1DGrid(dimBlock.x, config.matrixSize);
    dim3 dimGrid(Grid1D, Grid1D);// assume grid is square

    switch (config.kernelType) {
        case KernelType::KERNEL_GLOBAL:
            executeMatrixMultGlobal(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_TILED:
            executeMatrixMultTiled(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_COALESCED:
            executeMatrixMultCoalesced(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_COALESCED_DYM:
            executeMatrixMultCoalescedDym(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
        case KernelType::KERNEL_OVERLAPPED:
            executeMatrixMultOverlapped(dimBlock, dimGrid, Ad, Bd, Cd, config);
            break;
    }
    CHECK_ERR;
}
} // namespace gpu
