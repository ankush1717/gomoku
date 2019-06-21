#ifndef KERNELS_MINIMAX_CUH
#define KERNELS_MINIMAX_CUH
#include "board.hpp"


void cudaCallNegamaxKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        Board *boards,
        int *scores,
        const unsigned int leaves,
        const unsigned int d_depth,
        const unsigned int shape);



#endif
