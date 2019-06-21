#include "board.hpp"

CUDA_HOSTDEV
Board::Board() {
}

// This is a function to construct a copy of a board given another board.
CUDA_HOSTDEV
Board::Board(const Board &obj) {
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 15; j++) {
            pieces[i][j] = obj.pieces[i][j];
        }
    }
}

CUDA_HOSTDEV
/*
 * Destructor for the board.
 */
Board::~Board() {
}

CUDA_HOST
void Board::print_board() {
    cout << "      1   2   3   4   5   6   7   8   9  10   11  12  13  14  15" << endl;
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i < 9) {
            cout << i + 1 << ' ' << ' ';
        }
        else {
            cout << i + 1 << ' ';
        }
        for (int j = 0; j < BOARD_SIZE; j++) {
            char output;
            if (pieces[i][j] == 0)
                output = ' ';
            else if (pieces[i][j] == 1)
                output = 'x';
            else
                output = 'o';
            cout << " | " << output;
        }
        cout << " |" << endl;
    }
}

// This is the copy function on the host that creates a dynamic board.
CUDA_HOST
Board* Board::copy() {
    Board *newBoard = new Board();
    std::copy(&pieces[0][0], &pieces[0][0] + BOARD_SIZE*BOARD_SIZE, &newBoard->pieces[0][0]);
    // newBoard->pieces = pieces;
    return newBoard;
}

CUDA_HOSTDEV
bool Board::isTie() {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (pieces[i][j] != 0) {
                return false;
            }
        }
    }
    return true;
}


CUDA_HOSTDEV
bool Board::isDone() {
    if (checkDone(EX) || checkDone(OH)) {
        return true;
    }
    return false;
}

CUDA_HOSTDEV
bool Board::checkDone(Side side) {
    int goal;
    if (side == EX) {
        goal = 1;
    }
    else {
        goal = 2;
    }
    int moves [] = {1,-1,1,1,1,0,0,1};
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (pieces[i][j] != goal) {
                continue;
            }
            for (int k = 0; k < 8; k+=2) {
                int direction_one = moves[k];
                int direction_two = moves[k + 1];
                int x = i, y = j, count = 0;
                for (int l = 0; l < 5; l++) {
                    if (!isLegal(x, y)) {
                        break;
                    }
                    if (pieces[x][y] != goal) {
                        break;
                    }
                    x += direction_one;
                    y += direction_two;
                    count += 1;
                }
                if (count == 5) {
                    return true;
                }
            }
        }
    }
    return false;
}

CUDA_HOSTDEV
bool Board::isLegal(int x, int y) {

    if ((x > 14) || (x < 0) || (y > 14) || (y < 0)) {

        return false;
    }
    return true;
}

CUDA_HOSTDEV
bool Board::checkMove(int x, int y) {
    if (isLegal(x, y)) {
        if (pieces[x][y] == 0) {
            return true;
        }
    }
    return false;
}

CUDA_HOSTDEV
bool Board::checkTaken(int x, int y) {
    if (isLegal(x, y)) {
        if (pieces[x][y] != 0) {
            return true;
        }
    }
    return false;
}

CUDA_HOSTDEV
bool Board::checkBlack(int x, int y) {
    if (isLegal(x, y)) {
        if (pieces[x][y] == 1) {
            return true;
        }
    }

    return false;
}

CUDA_HOSTDEV
bool Board::checkWhite(int x, int y) {
    if (isLegal(x, y)) {
        if (pieces[x][y] == 2) {
            return true;
        }
    }

    return false;
}

CUDA_HOSTDEV
int Board::atPosition(int x, int y) {
    if (isLegal(x, y)) {
        return pieces[x][y];
    }

    return -1;
}

CUDA_HOSTDEV
int Board::doMove(Move *m, Side side) {

    int x = m->getX();
    int y = m->getY();

    // Ignore if move is invalid.
    if (!checkMove(x, y)) {
        return -1;
    }
    pieces[x][y] = (side == EX) ? 1 : 2;
    last = m;
    return 0;
}

CUDA_HOST
int Board::undoMove() {

    if (last == NULL) {
        cout << "Sorry cannot erase that far" << endl;
        return -1;
    }
    int x = last->getX();
    int y = last->getY();

    pieces[x][y] = 0;
    last = NULL;
    return 0;
}
