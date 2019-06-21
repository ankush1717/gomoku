#ifndef __CPU_H__
#define __CPU_H__
#include "common.hpp"
#include "board.hpp"
#include <vector>


class CpuAI {

public:
    CpuAI(Side side, int depth);
    ~CpuAI();

    Move* get_move(Move*last_move);
    int negamax(int cur_depth, Board* current_board, int shape, Side side, bool verbose, vector <Move *> moveOrder);
    int get_score(Board* current_board, bool verbose, bool next, int pov);
    vector <Move *> generate_moves(Board* current_board);

    Side mySide;
    Side theirSide;
    int depth;
    Board* my_board;
    Move* best_move;

};

#endif
