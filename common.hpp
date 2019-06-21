#ifndef __COMMON_H__
#define __COMMON_H__
#include <cstring>
#include <iostream>
#include <cstdlib>
using namespace std;

#define BOARD_SIZE 15

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#endif

enum Side {
    EX, OH
};


class Move {

public:
    int x, y;
    CUDA_HOSTDEV Move(int x, int y) {
        this->x = x;
        this->y = y;
    }
    CUDA_HOSTDEV Move() {
        this->x = -1;
        this->y = -1;
    }
    CUDA_HOSTDEV ~Move() {}

    CUDA_HOSTDEV int getX() { return x; }
    CUDA_HOSTDEV int getY() { return y; }

    CUDA_HOSTDEV void setX(int x) { this->x = x; }
    CUDA_HOSTDEV void setY(int y) { this->y = y; }
};

#endif
