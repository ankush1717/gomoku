#ifndef __BOARD_H__
#define __BOARD_H__
#include "common.hpp"


class Board {

private:
    int pieces[15][15] = {};

public:
    CUDA_HOSTDEV Board();
    CUDA_HOSTDEV ~Board();
    CUDA_HOSTDEV Board(const Board &obj);

    CUDA_HOST void print_board();
    CUDA_HOST Board* copy();

    CUDA_HOSTDEV bool isTie();
    CUDA_HOSTDEV bool isDone();
    CUDA_HOSTDEV bool checkDone(Side side);
    CUDA_HOSTDEV bool isLegal(int x, int y);
    CUDA_HOSTDEV bool checkMove(int x, int y);
    CUDA_HOSTDEV bool checkTaken(int x, int y);
    CUDA_HOSTDEV bool checkBlack(int x, int y);
    CUDA_HOSTDEV bool checkWhite(int x, int y);
    CUDA_HOSTDEV int atPosition(int x, int y);
    CUDA_HOSTDEV int doMove(Move *m, Side side);
    CUDA_HOST int undoMove();


    Move* last = NULL;

};

#endif
