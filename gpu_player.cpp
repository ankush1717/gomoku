#include "gpu_player.hpp"
#include <cuda_runtime.h>

// Error checking function that wraps around every cuda call
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Initializer for GPU AI. Sets the side and depth of the GPU. The parallel
// depth is hard coded as 2 because of watchdog timeouts.
GpuAI::GpuAI(Side side, int depth) {
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
    this->d_depth = 2;
    this->s_depth = depth - 2;
    if (s_depth % 2 == 0) {
        this->depth_correction = -1;
    }
    else {
        this->depth_correction = 1;
    }

    my_board = new Board();

    // Changed stack size of GPU.
    cudaDeviceSetLimit(cudaLimitStackSize, 50000);
    size_t* stack_size = (size_t*) malloc(sizeof(size_t));
    cudaDeviceGetLimit(stack_size, cudaLimitStackSize);

}

/*
 * Destructor for the player.
 */
GpuAI::~GpuAI() {
    delete my_board;
}

// GLOBALS

// Where the leaves of the first_negamax are stored
vector <Board> s_leaves;

// Sign of score depending on if minimizing or maximizing player
const int sign[2]={1,-1};

// Array of board scores at depth s returned from GPU.
int *host_minimax_scores;

// Counter to index into scores during second negamax.
int counter = -1;

Move *GpuAI::get_move(Move* last_move) {

    // Erase leaves at the beginning of each move to reset.
    s_leaves.erase (s_leaves.begin(), s_leaves.end());

    // Shape is a variable to indicate which side we are on in negamax function.
    int shape = (mySide == EX) ? 0 : 1;
    int cur_shape = shape;
    if (s_depth % 2 == 1) {
        cur_shape = 1 - shape;
    }

    // If AI is first, then move to center of board.
    if (last_move->getX() == -1) {
        best_move = new Move(7, 7);
    }
    else {
        my_board->doMove(last_move, theirSide);
        Board* copy_board = my_board->copy();

        first_negamax(s_depth, copy_board, mySide, false);
        int num_leaves_start = s_leaves.size();

        // Allocate space for device arrays of boards and scores.
        Board *device_d_boards_start;
        int *device_minimax_scores;

        gpuErrchk(cudaMalloc((void **)
                &device_d_boards_start, sizeof(Board) * (num_leaves_start)));

        gpuErrchk(cudaMalloc((void **)
                &device_minimax_scores, sizeof(int) * (num_leaves_start)));

        // Access leaves from first_negamax function, and copy them to device.
        Board *host_d_boards_start = &s_leaves[0];
        host_minimax_scores = (int*) malloc(sizeof(int) * num_leaves_start);

        gpuErrchk(cudaMemcpy(device_d_boards_start, host_d_boards_start,
            sizeof(Board) * num_leaves_start, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemset((void *) (device_minimax_scores), 0,
                            (num_leaves_start) * sizeof(int)));

        cudaCallNegamaxKernel(128, 128, device_d_boards_start, device_minimax_scores, num_leaves_start, d_depth, cur_shape);

        // Copy scores from device to host after kernel call.
        gpuErrchk(cudaMemcpy(host_minimax_scores, device_minimax_scores,
            sizeof(int) * num_leaves_start, cudaMemcpyDeviceToHost));

        // Reset counter so we can index into scores array correctly.
        counter = -1;
        second_negamax(s_depth, copy_board, shape, mySide, false);

        // Delete dynamic memory allocated.
        delete copy_board;
        delete[] host_minimax_scores;
    }

    my_board->doMove(best_move, mySide);

    return best_move;
}

// First negamax call that generates leaves at depth s.
void GpuAI::first_negamax(int cur_depth, Board* current_board, Side side, bool verbose) {

    Side otherSide = (side == EX) ? OH : EX;

    // If board is done, then we don't need to generate it as a leaf.
    if (current_board->checkDone(EX) || current_board->checkDone(OH)) {
        return;
    }

    // If depth is s, then add it to the vector of leaves.
    if (cur_depth == 0) {
        s_leaves.push_back(*current_board);
        return;
    }
    vector<Move *> possibleMoves = generate_moves(current_board);

    // Recursively call the function for each possible move.
    for (unsigned int i = 0; i < possibleMoves.size(); i++) {
        Board* temp = current_board->copy();
        temp->doMove(possibleMoves[i], side);
        first_negamax(cur_depth - 1, temp, otherSide, verbose);
        delete temp;
    }
}

// Same function as the first except we keep track of the best move, and we
// have access to the scores of the boards at depth s now.
int GpuAI::second_negamax(int cur_depth, Board* current_board, int shape, Side side, bool verbose) {
    Side otherSide = (side == EX) ? OH : EX;
    if (current_board->checkDone(EX)) {
        return sign[shape]*5000;
    }
    if (current_board->checkDone(OH)) {
        return sign[shape]*-5000;
    }
    if (cur_depth == 0) {
        counter += 1;
        // We use counter to index into the array of minimax scores generated by GPU.
        int score = sign[shape]*host_minimax_scores[counter]*depth_correction;
        return score;
    }

    // We initally set maximum to a really low value that can never be reached.
    int maxi = -10000;
    int best;
    if (cur_depth == s_depth) {
        best_move = new Move(1, 1);
        vector<Move *> possibleMoves = generate_moves(current_board);
        for (unsigned int i = 0; i < possibleMoves.size(); i++) {
            Board* temp = current_board->copy();
            temp->doMove(possibleMoves[i], side);

            best = -second_negamax(cur_depth - 1, temp, 1 - shape, otherSide, verbose);

            // If new best, then save this move as the best move.
            if (best > maxi) {
                maxi = best;
                best_move->setX(possibleMoves[i]->getX());
                best_move->setY(possibleMoves[i]->getY());
            }
            delete temp;
        }
        // cout << "max score seen: " << maxi << endl;
    }

    else {
        vector<Move *> possibleMoves = generate_moves(current_board);
        for (unsigned int i = 0; i < possibleMoves.size(); i++) {
            Board* temp = current_board->copy();
            temp->doMove(possibleMoves[i], side);
            best = -second_negamax(cur_depth - 1, temp, 1 - shape, otherSide, verbose);

            if (best > maxi) {
                maxi = best;
            }
            delete temp;
        }
    }

    return maxi;

}

// Function to generate moves given a board state. Basically, we only consider moves
// adjacent to an already taken spot on the board.
vector <Move *> GpuAI::generate_moves(Board* current_board) {

    vector<Move *> possibleMoves;

    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (current_board->checkMove(i,j)) {
                if (current_board->checkTaken(i + 1,j) || current_board->checkTaken(i - 1,j)
                || current_board->checkTaken(i,j - 1) || current_board->checkTaken(i,j + 1)
                || current_board->checkTaken(i + 1,j - 1) || current_board->checkTaken(i + 1,j + 1)
                || current_board->checkTaken(i - 1,j - 1) || current_board->checkTaken(i - 1,j + 1)) {
                    // If spot is adjacent to an already taken spot, add it to list.
                    Move *addmine = new Move(i, j);
                    possibleMoves.push_back(addmine);
                }
            }
        }
    }
    return possibleMoves;
}
