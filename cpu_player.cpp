#include "cpu_player.hpp"
#include <vector>

CpuAI::CpuAI(Side side, int depth) {
    mySide = side;
    if(side == EX)
    {
        theirSide = OH;
    }
    else
    {
        theirSide = EX;
    }
    this->depth = depth;

    my_board = new Board();

}

/*
 * Destructor for the player.
 */
CpuAI::~CpuAI() {
    delete my_board;
}

Move *CpuAI::get_move(Move* last_move) {
    int shape = (mySide == EX) ? 0 : 1;
    if (last_move->getX() == -1) {
        best_move = new Move(7, 7);
    }
    else {
        my_board->doMove(last_move, theirSide);
        Board* copy_board = my_board->copy();
        vector<Move *> moveOrder;
        negamax(depth, copy_board, shape, mySide, false, moveOrder);
        delete copy_board;
    }
    my_board->doMove(best_move, mySide);

    return best_move;
}

const int sign[2]={1,-1};

int CpuAI::negamax(int cur_depth, Board* current_board, int shape, Side side, bool verbose, vector <Move *> moveOrder) {

    Side otherSide = (side == EX) ? OH : EX;
    if (current_board->checkDone(EX)) {
        return sign[shape]*5000;
    }
    if (current_board->checkDone(OH)) {
        return sign[shape]*-5000;
    }
    if (cur_depth == 0) {
        int score = sign[shape]*get_score(current_board, verbose, false, 1 - shape);
        return score;
    }
    int maxi = -10000;
    int best;
    if (cur_depth == depth) {
        best_move = new Move(1, 1);
        vector<Move *> possibleMoves = generate_moves(current_board);
        for (unsigned int i = 0; i < possibleMoves.size(); i++) {
            Board* temp = current_board->copy();
            temp->doMove(possibleMoves[i], side);
            vector<Move *> moveOrder2;
            moveOrder2 = moveOrder;
            moveOrder2.push_back(possibleMoves[i]);
            best = -negamax(cur_depth - 1, temp, 1 - shape, otherSide, verbose, moveOrder2);
            if (best > maxi) {
                maxi = best;
                best_move->setX(possibleMoves[i]->getX());
                best_move->setY(possibleMoves[i]->getY());
            }
            delete temp;
        }
    }

    else {
        vector<Move *> possibleMoves = generate_moves(current_board);
        for (unsigned int i = 0; i < possibleMoves.size(); i++) {
            Board* temp = current_board->copy();
            temp->doMove(possibleMoves[i], side);
            vector<Move *> moveOrder2;
            moveOrder2 = moveOrder;
            moveOrder2.push_back(possibleMoves[i]);

            best = -negamax(cur_depth - 1, temp, 1 - shape, otherSide, verbose, moveOrder2);

            if (best > maxi) {
                maxi = best;
            }
            delete temp;
        }
    }

    return maxi;

}

vector <Move *> CpuAI::generate_moves(Board* current_board) {
    vector<Move *> possibleMoves;


    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (current_board->checkMove(i,j)) {
                if (current_board->checkTaken(i + 1,j) || current_board->checkTaken(i - 1,j)
                || current_board->checkTaken(i,j - 1) || current_board->checkTaken(i,j + 1)
                || current_board->checkTaken(i + 1,j - 1) || current_board->checkTaken(i + 1,j + 1)
                || current_board->checkTaken(i - 1,j - 1) || current_board->checkTaken(i - 1,j + 1)) {
                    Move *addmine = new Move(i, j);
                    possibleMoves.push_back(addmine);
                }
            }
        }
    }
    return possibleMoves;
}

int CpuAI::get_score(Board* current_board, bool verbose, bool next, int pov)
{
    int score = 0;

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
                if (current_board->atPosition(i,j) != goal) {
                    continue;
                }
                for (int k = 0; k < 8; k+=2) {
                    int direction_one = moves[k];
                    int direction_two = moves[k + 1];
                    int x = i, y = j, count = 0;

                    last = current_board->atPosition(x - direction_one, y - direction_two);
                    if (last == goal) {
                        continue;
                    }

                    for (int l = 0; l < 4; l++) {
                        if (!current_board->isLegal(x, y)) {
                            break;
                        }
                        if (current_board->atPosition(x,y) != goal) {
                            break;
                        }
                        x += direction_one;
                        y += direction_two;
                        count += 1;
                    }
                    if (count == 4) {
                        if ((next == true && pov + 1 == goal) || (next == false && pov + 1 != goal)) {
                            if (current_board->atPosition(x, y) == 0) {
                                if (last == 0) {
                                    semi_score += next_open_four;
                                    if (verbose) {
                                        cout << "next_open_four" << goal << endl;
                                    }
                                }
                                else if (last == -1){
                                    semi_score += next_wall_four;
                                    if (verbose) {
                                        cout << "next_wall_four" << goal << endl;
                                    }
                                }
                                else {
                                    semi_score += next_half_four;
                                    if (verbose) {
                                        cout << "next_half_four" << goal << endl;
                                    }
                                }
                            }
                            else if (current_board->atPosition(x, y) == -1) {
                                if (last == 0) {
                                    semi_score += next_wall_four;
                                    if (verbose) {
                                        cout << "next_wall_four" << goal << endl;
                                    }
                                }
                            }
                            else if (current_board->atPosition(x, y) != goal) {
                                if (last == 0) {
                                    semi_score += next_half_four;
                                    if (verbose) {
                                        cout << "next_half_four" << goal << endl;
                                    }
                                }
                            }
                        }
                        else {
                            if (current_board->atPosition(x, y) == 0) {
                                if (last == 0) {
                                    semi_score += open_four;
                                    if (verbose) {
                                        cout << "open_four" << goal << endl;
                                    }
                                }
                                else if (last == -1){
                                    semi_score += wall_four;
                                    if (verbose) {
                                        cout << "wall_four" << goal << endl;
                                    }
                                }
                                else {
                                    semi_score += half_four;
                                    if (verbose) {
                                        cout << "half_four" << goal << endl;
                                    }
                                }
                            }
                            else if (current_board->atPosition(x, y) == -1) {
                                if (last == 0) {
                                    semi_score += wall_four;
                                    if (verbose) {
                                        cout << "wall_four" << goal << endl;
                                    }
                                }
                            }
                            else if (current_board->atPosition(x, y) != goal) {
                                if (last == 0) {
                                    semi_score += half_four;
                                    if (verbose) {
                                        cout << "half_four" << goal << endl;
                                    }
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
                if (current_board->atPosition(i,j) != goal) {
                    continue;
                }
                for (int k = 0; k < 8; k+=2) {
                    int direction_one = moves[k];
                    int direction_two = moves[k + 1];
                    int x = i, y = j, count = 0;

                    last = current_board->atPosition(x - direction_one, y - direction_two);
                    if (last == goal) {
                        continue;
                    }

                    for (int l = 0; l < 3; l++) {
                        if (!current_board->isLegal(x, y)) {
                            break;
                        }
                        if (current_board->atPosition(x,y) != goal) {
                            break;
                        }
                        x += direction_one;
                        y += direction_two;
                        count += 1;
                    }
                    if (count == 3) {
                        if (current_board->atPosition(x, y) == 0) {
                            if (last == 0) {
                                if ((next == true && pov + 1 == goal) || (next == false && pov + 1 != goal)) {
                                    semi_score += next_open_three;
                                    if (verbose) {
                                        cout << "next_open_three" << goal << endl;
                                    }
                                }
                                else {
                                    semi_score += open_three;
                                    if (verbose) {
                                        cout << "open_three" << goal << endl;
                                    }
                                }

                            }
                            else if (last == -1){
                                semi_score += wall_three;
                                if (verbose) {
                                    cout << "wall_three" << goal << endl;
                                }
                            }
                            else {
                                semi_score += half_three;
                                if (verbose) {
                                    cout << "half_three" << goal << endl;
                                }
                            }
                        }
                        else if (current_board->atPosition(x, y) == -1) {
                            if (last == 0) {
                                semi_score += wall_three;
                                if (verbose) {
                                    cout << "wall_three" << goal << endl;
                                }
                            }
                        }
                        else if (current_board->atPosition(x, y) != goal) {
                            if (last == 0) {
                                semi_score += half_three;
                                if (verbose) {
                                    cout << "half_three" << goal << endl;
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

    // twos
    for (int goal = 1; goal <= 2; goal++) {
        int semi_score = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                if (current_board->atPosition(i,j) != goal) {
                    continue;
                }
                for (int k = 0; k < 8; k+=2) {
                    int direction_one = moves[k];
                    int direction_two = moves[k + 1];
                    int x = i, y = j, count = 0;

                    last = current_board->atPosition(x - direction_one, y - direction_two);
                    if (last == goal) {
                        continue;
                    }

                    for (int l = 0; l < 2; l++) {
                        if (!current_board->isLegal(x, y)) {
                            break;
                        }
                        if (current_board->atPosition(x,y) != goal) {
                            break;
                        }
                        x += direction_one;
                        y += direction_two;
                        count += 1;
                    }
                    if (count == 2) {
                        if (current_board->atPosition(x, y) == 0 &&
                            current_board->atPosition(x + direction_one, y + direction_two) == goal &&
                            current_board->atPosition(x + 2*direction_one, y + 2*direction_two) == 0 &&
                            last == 0)
                        {
                            if ((next == true && pov + 1 == goal) || (next == false && pov + 1 != goal)) {
                                semi_score += next_holed_four;
                                if (verbose) {
                                    cout << "next_holed_four" << goal << endl;
                                }
                            }
                            else {
                                semi_score += holed_four;
                                if (verbose) {
                                    cout << "holed_four" << goal << endl;
                                }
                            }
                        }

                        if (current_board->atPosition(x, y) == 0 &&
                            current_board->atPosition(x - 4*direction_one, y - 4*direction_two) == goal &&
                            current_board->atPosition(x - 5*direction_one, y + 5*direction_two) == 0 &&
                            last == 0)
                        {
                            if ((next == true && pov + 1 == goal) || (next == false && pov + 1 != goal)) {
                                semi_score += next_holed_four;
                                if (verbose) {
                                    cout << "next_holed_four" << goal << endl;
                                }
                            }
                            else {
                                semi_score += holed_four;
                                if (verbose) {
                                    cout << "holed_four" << goal << endl;
                                }
                            }
                        }


                        if (current_board->atPosition(x, y) == 0) {
                            if (last == 0) {
                                semi_score += open_two;
                                if (verbose) {
                                    cout << "open_two" << goal << endl;
                                }
                            }
                            else if (last == -1){
                                semi_score += wall_two;
                                if (verbose) {
                                    cout << "wall_two" << goal << endl;
                                }
                            }
                            else {
                                semi_score += half_two;
                                if (verbose) {
                                    cout << "half_two" << goal << endl;
                                }
                            }
                        }
                        else if (current_board->atPosition(x, y) == -1) {
                            if (last == 0) {
                                semi_score += wall_two;
                                if (verbose) {
                                    cout << "wall_two" << goal << endl;
                                }
                            }
                        }
                        else if (current_board->atPosition(x, y) != goal) {
                            if (last == 0) {
                                semi_score += half_two;
                                if (verbose) {
                                    cout << "half_two" << goal << endl;
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


    return score;
}
