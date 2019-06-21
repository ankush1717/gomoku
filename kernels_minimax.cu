#include <cstdio>
#include <cuda_runtime.h>

#include "kernels_minimax.cuh"

// Same function as described in gpu_player.cpp to generate moves. Called on GPU
// on a thread level.
__device__
void generate_moves_gpu(Board current_board, Move* possibleMoves) {

    int num_found = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (current_board.checkMove(i,j)) {
                if (current_board.checkTaken(i + 1,j) || current_board.checkTaken(i - 1,j)
                || current_board.checkTaken(i,j - 1) || current_board.checkTaken(i,j + 1)
                || current_board.checkTaken(i + 1,j - 1) || current_board.checkTaken(i + 1,j + 1)
                || current_board.checkTaken(i - 1,j - 1) || current_board.checkTaken(i - 1,j + 1)) {
                    possibleMoves[num_found].setX(i);
                    possibleMoves[num_found].setY(j);
                    num_found += 1;
                }
            }
        }
    }
}

// This is the heuristic function to get the score of a board that is a leaf.
// This function will not be commented in detail because it is not relevant to
// the gpu portion of project and is also quite long. The general idea is
// that we look through the board and see if we can find four, three, or two
// in a row of a color in any direction, and we assign different weights to
// each of the possibilities.
__device__
int get_score_gpu(Board current_board, int pov)
{
    int score = 0;

    // Set of heuristics
    int open_four = 500;
    int half_four = 5;
    int wall_four = 0;

    int holed_four = 50;

    int open_three = 50;
    int half_three = 15;
    int wall_three = 0;

    int open_two = 8;
    int half_two = 5;
    int wall_two = 0;

    int next_open_four = 1000;
    int next_half_four = 1000;
    int next_wall_four = 1000;

    int next_holed_four = 200;

    int next_open_three = 200;

    int last = 0;

    // fours
    int moves [] = {1,-1,1,1,1,0,0,1};
    for (int goal = 1; goal <= 2; goal++) {
        int semi_score = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_board.atPosition(i,j) != goal) {
                    continue;
                }
                for (int k = 0; k < 8; k+=2) {
                    int direction_one = moves[k];
                    int direction_two = moves[k + 1];
                    int x = i, y = j, count = 0;

                    last = current_board.atPosition(x - direction_one, y - direction_two);
                    if (last == goal) {
                        continue;
                    }

                    for (int l = 0; l < 4; l++) {
                        if (!current_board.isLegal(x, y)) {
                            break;
                        }
                        if (current_board.atPosition(x,y) != goal) {
                            break;
                        }
                        x += direction_one;
                        y += direction_two;
                        count += 1;
                    }
                    if (count == 4) {
                        if (pov + 1 != goal) {
                            if (current_board.atPosition(x, y) == 0) {
                                if (last == 0) {
                                    semi_score += next_open_four;
                                }
                                else if (last == -1){
                                    semi_score += next_wall_four;
                                }
                                else {
                                    semi_score += next_half_four;
                                }
                            }
                            else if (current_board.atPosition(x, y) == -1) {
                                if (last == 0) {
                                    semi_score += next_wall_four;
                                }
                            }
                            else if (current_board.atPosition(x, y) != goal) {
                                if (last == 0) {
                                    semi_score += next_half_four;
                                }
                            }
                        }
                        else {
                            if (current_board.atPosition(x, y) == 0) {
                                if (last == 0) {
                                    semi_score += open_four;
                                }
                                else if (last == -1){
                                    semi_score += wall_four;
                                }
                                else {
                                    semi_score += half_four;
                                }
                            }
                            else if (current_board.atPosition(x, y) == -1) {
                                if (last == 0) {
                                    semi_score += wall_four;
                                }
                            }
                            else if (current_board.atPosition(x, y) != goal) {
                                if (last == 0) {
                                    semi_score += half_four;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (goal == 2) {
            score -= semi_score;
        }
        else {
            score += semi_score;
        }
    }

    // threes
    for (int goal = 1; goal <= 2; goal++) {
        int semi_score = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_board.atPosition(i,j) != goal) {
                    continue;
                }
                for (int k = 0; k < 8; k+=2) {
                    int direction_one = moves[k];
                    int direction_two = moves[k + 1];
                    int x = i, y = j, count = 0;

                    last = current_board.atPosition(x - direction_one, y - direction_two);
                    if (last == goal) {
                        continue;
                    }

                    for (int l = 0; l < 3; l++) {
                        if (!current_board.isLegal(x, y)) {
                            break;
                        }
                        if (current_board.atPosition(x,y) != goal) {
                            break;
                        }
                        x += direction_one;
                        y += direction_two;
                        count += 1;
                    }
                    if (count == 3) {
                        if (current_board.atPosition(x, y) == 0) {
                            if (last == 0) {
                                if (pov + 1 != goal) {
                                    semi_score += next_open_three;
                                }
                                else {
                                    semi_score += open_three;
                                }

                            }
                            else if (last == -1){
                                semi_score += wall_three;
                            }
                            else {
                                semi_score += half_three;
                            }
                        }
                        else if (current_board.atPosition(x, y) == -1) {
                            if (last == 0) {
                                semi_score += wall_three;
                            }
                        }
                        else if (current_board.atPosition(x, y) != goal) {
                            if (last == 0) {
                                semi_score += half_three;
                            }
                        }
                    }
                }
            }
        }
        if (goal == 2) {
            score -= semi_score;
        }
        else {
            score += semi_score;
        }
    }

    // twos
    for (int goal = 1; goal <= 2; goal++) {
        int semi_score = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_board.atPosition(i,j) != goal) {
                    continue;
                }
                for (int k = 0; k < 8; k+=2) {
                    int direction_one = moves[k];
                    int direction_two = moves[k + 1];
                    int x = i, y = j, count = 0;

                    last = current_board.atPosition(x - direction_one, y - direction_two);
                    if (last == goal) {
                        continue;
                    }

                    for (int l = 0; l < 2; l++) {
                        if (!current_board.isLegal(x, y)) {
                            break;
                        }
                        if (current_board.atPosition(x,y) != goal) {
                            break;
                        }
                        x += direction_one;
                        y += direction_two;
                        count += 1;
                    }
                    if (count == 2) {
                        if (current_board.atPosition(x, y) == 0 &&
                            current_board.atPosition(x + direction_one, y + direction_two) == goal &&
                            current_board.atPosition(x + 2*direction_one, y + 2*direction_two) == 0 &&
                            last == 0)
                        {
                            if (pov + 1 != goal) {
                                semi_score += next_holed_four;
                            }
                            else {
                                semi_score += holed_four;
                            }
                        }

                        if (current_board.atPosition(x, y) == 0 &&
                            current_board.atPosition(x - 4*direction_one, y - 4*direction_two) == goal &&
                            current_board.atPosition(x - 5*direction_one, y + 5*direction_two) == 0 &&
                            last == 0)
                        {
                            if (pov + 1 != goal) {
                                semi_score += next_holed_four;
                            }
                            else {
                                semi_score += holed_four;
                            }
                        }


                        if (current_board.atPosition(x, y) == 0) {
                            if (last == 0) {
                                semi_score += open_two;
                            }
                            else if (last == -1){
                                semi_score += wall_two;
                            }
                            else {
                                semi_score += half_two;
                            }
                        }
                        else if (current_board.atPosition(x, y) == -1) {
                            if (last == 0) {
                                semi_score += wall_two;
                            }
                        }
                        else if (current_board.atPosition(x, y) != goal) {
                            if (last == 0) {
                                semi_score += half_two;
                            }
                        }
                    }
                }
            }
        }
        if (goal == 2) {
            score -= semi_score;
        }
        else {
            score += semi_score;
        }
    }


    return score;
}

// This is the recursive negamax function that the kernel calls. It is
// very similar to the second_negamax function in cpu_player.cpp (and so I won't comment every line)
// The difference is that we don't know the size of the number of possible
// moves generated, so we add a flag to know when to stop (when the move is (-1,-1))
__device__
int negamax_gpu (int cur_depth, Board current_board, int shape, int thread_index){

    int sign;
    Side side;
    if (shape == 0) {
        sign = 1;
        side = EX;
    }
    else {
        sign = -1;
        side = OH;
    }

    if (current_board.checkDone(EX)) {
        return sign*5000;
    }
    if (current_board.checkDone(OH)) {
        return sign*-5000;
    }
    if (cur_depth == 0) {
        int score = sign*get_score_gpu(current_board, 1 - shape);
        return score;
    }
    int maxi = -10000;
    int best;

    Move possibleMoves[100];
    generate_moves_gpu(current_board, possibleMoves);

    int i = 0;
    while (true) {
        if (i > 99) {
            break;
        }
        int x = possibleMoves[i].getX();
        int y = possibleMoves[i].getY();
        if (x == -1 && y == -1) {
            break;
        }
        Board temp = current_board;
        temp.doMove(&possibleMoves[i], side);

        best = -negamax_gpu(cur_depth - 1, temp, 1 - shape, thread_index);

        if (best > maxi) {
            maxi = best;
        }
        i += 1;
    }

    return maxi;
}

// This is the actual kernel that is called. It assigns each thread to a board. When
// the negamax function returns, it assigns that index the score in the scores array.
__global__
void
cudaNegamaxKernel(Board* boards, int* scores, int leaves, int d_depth, int shape) {
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	while (thread_index < leaves) {
		Board current_board = boards[thread_index];
        scores[thread_index] = negamax_gpu(d_depth, current_board, shape, thread_index);
        thread_index += blockDim.x * gridDim.x;
    }
}

// This is API to the kernel that the gpu_player can call. The leaves argument
// specifies the number of boards, the d_depth argument specifies the parallel
// depth, and the shape specifies which side we are.
void cudaCallNegamaxKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        Board *boards,
        int *scores,
        const unsigned int leaves,
        const unsigned int d_depth,
        const unsigned int shape) {

    cudaNegamaxKernel<<<blocks, threadsPerBlock>>>
        (boards, scores, leaves, d_depth, shape);

}
