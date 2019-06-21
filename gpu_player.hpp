#ifndef __GPU_H__
#define __GPU_H__
#include "common.hpp"
#include "board.hpp"
#include "kernels_minimax.cuh"
#include <vector>

class GpuAI {

public:
    GpuAI(Side side, int depth);
    ~GpuAI();

    Move* get_move(Move* last_move);
    void first_negamax(int cur_depth, Board* current_board, Side side, bool verbose);
    int second_negamax(int cur_depth, Board* current_board, int shape, Side side, bool verbose);
    int get_score(Board* current_board, bool verbose, bool next, int pov, Move* last_move);
    vector <Move *> generate_moves(Board* current_board);

    Side mySide;
    Side theirSide;
    int depth;
    Board* my_board;
    Move* best_move;

    int s_depth;
    int d_depth;
    int depth_correction;

};

#endif
