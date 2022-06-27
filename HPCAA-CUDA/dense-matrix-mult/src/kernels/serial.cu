#include "kernels.h"

namespace cpu {
void matrixMult(std::vector<float> &C, const std::vector<float> &A, const std::vector<float> &B,
                const Configuration &config) {

    const int size = config.matrixSize;
    for (int r = 0; r < config.numRepeats; ++r) {

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {

                float tmp = B[i + size * j];
                for (int k = 0; k < size; ++k) {
                    C[k + size * j] += A[k + size * i] * tmp;
                }
            }
        }
    }
}
} // namespace cpu