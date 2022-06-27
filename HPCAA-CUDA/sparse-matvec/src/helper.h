#ifndef HELPER_HPP
#define HELPER_HPP

#include "data_types.h"

void convertCooToCsr(CsrMatrix &matrix, const std::vector<int>& I);
EllMatrix get1DStencilEllMatrix(int numPoints);
CsrMatrix get1DStencilCsrMatrix(int numPoints);


#endif // HELPER_HPP
