#include "data_types.h"
#include "driver.h"
#include "kernels/kernels.h"
#include "util.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cstdlib>
#include <vector>

using namespace ::testing;

std::vector<Matcher<float>> getExpectedArray(const std::vector<float> &Answer) {
    std::vector<Matcher<float>> expectedArray;
    for (const auto item : Answer) {
        expectedArray.emplace_back(item);
    }
    return expectedArray;
}

class MatMult : public ::testing::Test {
  protected:
    void SetUp() override {
        // prepare a fresh configuration for each test
        config = Configuration();
        config.numRepeats = 1;
        config.printInfo = false;
        config.printMatrix = false;
    }
    Configuration config{};
};


TEST_F(MatMult, CPU) {
    constexpr size_t matrixSize = 7;
    constexpr size_t totalSize = matrixSize * matrixSize;
    std::vector<float> A = {91., 30., 20., 14., 39., 62., 88., 75., 9.,  75., 82., 63., 26.,
                            74., 37., 1.,  84., 74., 40., 22., 17., 6.,  38., 15., 68., 71.,
                            13., 31., 4.,  97., 99., 96., 49., 68., 11., 0.,  45., 73., 14.,
                            83., 4.,  3.,  20., 56., 18., 66., 73., 76., 50.};

    std::vector<float> B = {18., 3.,  36., 27., 25., 9.,  11., 74., 17., 34., 61., 21., 39.,
                            14., 43., 72., 65., 98., 27., 3.,  21., 79., 17., 8.,  65., 4.,
                            21., 62., 69., 39., 37., 72., 85., 8.,  97., 25., 97., 8.,  29.,
                            30., 65., 48., 59., 6.,  54., 47., 86., 22., 5.};

    std::vector<float> TestC(49, 0.0f);


    config.matrixSize = matrixSize;
    config.kernelType = KernelType::KERNEL_CPU;
    cpu::matrixMult(TestC, A, B, config);

    std::vector<float> ExpectedC = {
        3677.,  5075.,  7344.,  8250.,  7023.,  4909.,  4107.,  9997.,  9301.,  11704.,
        12580., 14936., 9219.,  11287., 12834., 9657.,  16460., 22000., 18876., 10686.,
        14611., 10406., 9806.,  7547.,  12282., 15552., 11429., 13568., 13285., 19231.,
        19238., 26472., 23650., 20226., 17628., 11100., 11256., 17461., 17826., 19834.,
        10573., 13338., 8543.,  13276., 17081., 17404., 14581., 11929., 9273.};

    auto expectedArray = getExpectedArray(ExpectedC);
    EXPECT_THAT(TestC, ElementsAreArray(expectedArray));
}


TEST(CudaGrids, Grid1D) {
    EXPECT_EQ(gpu::get1DGrid(4, 16), 4);
    EXPECT_EQ(gpu::get1DGrid(5, 16), 4);
    EXPECT_EQ(gpu::get1DGrid(8, 16), 2);
    EXPECT_EQ(gpu::get1DGrid(9, 16), 2);
    EXPECT_EQ(gpu::get1DGrid(16, 16), 1);
    EXPECT_EQ(gpu::get1DGrid(17, 16), 1);
}


struct TestMemory {
  public:
    TestMemory(size_t size) : matrixSize(size), totalSize(size * size) {
        TestC.resize(totalSize, 0.0);
        ExpectedC.resize(totalSize, 0.0);
        A.resize(totalSize, 0.0);
        B.resize(totalSize, 0.0);

        auto randomizer = [](float) { return static_cast<float>(rand() % 100); };
        std::transform(A.begin(), A.end(), A.begin(), randomizer);
        std::transform(B.begin(), B.end(), B.begin(), randomizer);
    }

    std::vector<float> A{};
    std::vector<float> B{};
    std::vector<float> TestC{};
    std::vector<float> ExpectedC{};

  private:
    size_t totalSize{};
    size_t matrixSize{};
};


namespace test {
  namespace local {
    size_t size = 64;
  }
}


TEST_F(MatMult, GPU_CUBLAS) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_CUBLAS;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_GLOBAL) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_GLOBAL;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_TILED) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_TILED;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_COALESCED) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_COALESCED;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_COALESCED_DYM) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_COALESCED_DYM;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}


TEST_F(MatMult, GPU_OVERLAPPED) {
    config.tileSize = 4;
    config.matrixSize = test::local::size;
    config.kernelType = KernelType::KERNEL_OVERLAPPED;
    TestMemory memory(config.matrixSize);

    // run on GPU
    compute(memory.TestC, memory.A, memory.B, config);

    // run on CPU
    cpu::matrixMult(memory.ExpectedC, memory.A, memory.B, config);

    auto expectedArray = getExpectedArray(memory.ExpectedC);

    EXPECT_THAT(memory.TestC, ElementsAreArray(expectedArray));
}