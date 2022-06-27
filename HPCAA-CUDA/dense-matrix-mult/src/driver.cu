#include "driver.h"
#include "kernels/kernels.h"
#include "util.hpp"
#include <chrono>
#include <cublas_v2.h>

float compute(std::vector<float> &C, const std::vector<float> &A, const std::vector<float> &B,
              const Configuration &config) { // 由于cudaMemcpy需要传入的src是一个const void *src, 所以传参时, A是一个const

    cudaDeviceReset();
    CHECK_ERR; // GPU error occurs but may terminate in another place: to locate the error
    // Check err every time after each GPU API call. save time for debugging
     // errors usually on GPU, CPU does not know
    // if not check, 有可能GPU在执行第三行时,第一行执行完了,但是问题出在第一行
    // 准确定位error的位置

    float *devA{nullptr}, *devB{nullptr}, *devC{nullptr};
    {
      // TODO: Allocate matrices A, B, C on device
      cudaMalloc(&devA, A.size() * sizeof(float)); CHECK_ERR;
      cudaMalloc(&devB, B.size() * sizeof(float)); CHECK_ERR;
      cudaMalloc(&devC, C.size() * sizeof(float)); CHECK_ERR;
    }

    {
      // TODO: Copy the data from host to the device
      // NOTE: You may copy C as well, as it is zeroed, or cudaMemset it to zero on the device
      // devA是一个float *的指针. 所以memcpy只要传一个指针即devA即可. 而cudaMalloc需要传**, 指向指针的指针, 因此需要传入&devA
      cudaMemcpy(devA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
      cudaMemcpy(devB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
      cudaMemcpy(devC, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice); CHECK_ERR;
      // cudaMemset(devC, 0, C.size() * sizeof(float));
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    float cpuTime{};
    cudaEvent_t startTimer{}, stopTimer{};
    cudaEventCreate(&startTimer);
    cudaEventCreate(&stopTimer);


    // Start computing
    cudaEventRecord(startTimer, 0);
    switch (config.kernelType) {
        case KernelType::KERNEL_CPU: {
            auto begin = std::chrono::high_resolution_clock::now();
            // NOTE: repeat loop is inside of cpu::matrixMult
            cpu::matrixMult(C, A, B, config);
            auto end = std::chrono::high_resolution_clock::now();
            cpuTime =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin).count();
            break;
        }
        case KernelType::KERNEL_CUBLAS: {
          // 调用cublas库计算
            float alpha = 1.0f, beta = 1.0f;
            for (int i = 0; i < config.numRepeats; ++i) {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, config.matrixSize, config.matrixSize,
                            config.matrixSize, &alpha, devA, config.matrixSize, devB,
                            config.matrixSize, &beta, devC, config.matrixSize);
                            // CUBLAS_OP_N: non transpose operation
                // NOTE: float是单精度,所以用sgemm
            }
            CHECK_ERR;
            break;
        }
        default: {
            // NOTE: repeat loop is inside of gpu::matrixMult
            gpu::matrixMult(devA, devB, devC, config); // use the self-written mat-mul
            break;
        }
    }

    cudaEventRecord(stopTimer, 0);
    cudaEventSynchronize(stopTimer);
    CHECK_ERR;
    // NOTE: cudaEventSynchronize(stopTimer) is implicit cudaDeviceSynchronize() in this context
    // The time measurement does not count memory movements, only focus on kernel performance
    float gpuTime{};
    cudaEventElapsedTime(&gpuTime, startTimer, stopTimer);

    // release resources
    cublasDestroy(handle);
    cudaEventDestroy(startTimer);
    cudaEventDestroy(stopTimer);

    {
      // TODO: transfer matrix C back, from device to the host
      // NOTE: const_cast<float *>(); generally, sizeof(float) returns 4, i.e. single precision
      cudaMemcpy(const_cast<float *> (C.data()), devC, C.size() * sizeof(float), cudaMemcpyDeviceToHost);
      CHECK_ERR;
      // NOTE: should be const_cast, otherwise breaks
    }

    {
      // TODO: clean gpu memory
      cudaFree(devA); CHECK_ERR;
      cudaFree(devB); CHECK_ERR;
      cudaFree(devC); CHECK_ERR;
    }
    return (config.kernelType == KernelType::KERNEL_CPU) ? cpuTime : gpuTime;
}
