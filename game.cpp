#include "game.hpp"

/**
 * @brief Removes trailing whitespaces from a string
 *
 * This function, given a string, strips it of any trailing whitespaces
 *
 * @param move string input
 */
void strip_move(string &move)
{
    int pos = move.find_last_not_of(" ");
    if ((size_t) pos != string::npos) // if user entered only spaces
    {
        move = move.substr(0, pos + 1);
    }
}

/**
 * @brief Allows user interaction
 *
 * This function, allows user input and handles user input by calling
 * appropriate methods
 */
void Game::Run(Board board, string input, int comp_start, int depth)
{
    if (input == "human") {
        int count = 0;
        while (true)
        {
            board.print_board();
            if (count % 2 == 0) {
                cout << "It is X turn " << endl;
            }
            else {
                cout << "It is O turn " << endl;
            }
            string move;
            cout << "Please enter a move: " << endl;
            getline(cin, move);
            strip_move(move);
            if (move[0] == 'q')
            {
                break;
            }
            else if (move[0] == 'u')
            {
                int ret = board.undoMove();
                if (ret == 0) {
                    count -= 1;
                }
            }
            else
            {
                string move2;
                getline(cin, move2);
                int row = stoi(move);
                int col = stoi(move2);
                if (row - 1 < 0 || row - 1 > BOARD_SIZE - 1)
                {
                    cout << "ERROR: invalid row input" << endl;
                    continue;
                }
                if (col - 1 < 0 || col - 1 > BOARD_SIZE - 1)
                {
                    cout << "ERROR: invalid column input" << endl;
                    continue;
                }
                Move next_move(row - 1, col - 1);
                int ret;
                if (count % 2 == 0) {
                    ret = board.doMove(&next_move, EX);
                }
                else {
                    ret = board.doMove(&next_move, OH);
                }
                if (board.checkDone(EX))
                {
                    cout << "X has won!" << endl;
                    break;
                }
                if (board.checkDone(OH))
                {
                    cout << "O has won!" << endl;
                    break;
                }
                if (ret == 0) {
                    count += 1;
                }
                else {
                    cout << "ERROR: move already taken" << endl;
                }
            }
        }
        board.print_board();
    }
    else if (input == "cpu" || input == "gpu") {
        Side side = (comp_start == 0) ? EX : OH;
        cout << "depth is: " << depth << endl;
        CpuAI* cpu = new CpuAI(side, depth);
        GpuAI* gpu = new GpuAI(side, depth);

        int count = 0;
        Move* next_move = new Move(1,1);
        Move* last_move = new Move(1,1);

        while (true)
        {
            board.print_board();
            if (count % 2 == 0) {
                cout << "It is X turn " << endl;
            }
            else {
                cout << "It is O turn " << endl;
            }

            int row, col;

            if (count % 2 != comp_start) {
                string move;
                cout << "Please enter a move: " << endl;
                getline(cin, move);
                strip_move(move);
                if (move[0] == 'q')
                {
                    break;
                }
                else if (move[0] == 'u')
                {
                    int ret = board.undoMove();
                    if (ret == 0) {
                        count -= 1;
                    }
                }
                else
                {
                    string move2;
                    getline(cin, move2);
                    row = stoi(move);
                    col = stoi(move2);
                    if (row - 1 < 0 || row - 1 > BOARD_SIZE - 1)
                    {
                        cout << "ERROR: invalid row input" << endl;
                        continue;
                    }
                    if (col - 1 < 0 || col - 1 > BOARD_SIZE - 1)
                    {
                        cout << "ERROR: invalid column input" << endl;
                        continue;
                    }
                    next_move->setX(row - 1);
                    next_move->setY(col - 1);

                    last_move->setX(next_move->getX());
                    last_move->setY(next_move->getY());
                }
            }
            else {
                cout << "It is AI turn" << endl;
                if (input == "cpu") {
                    if (count == 0) {
                        last_move->setX(-1);
                    }
                    clock_t begin = clock();

                    Move* temp = cpu->get_move(last_move);

                    clock_t end = clock();
                    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
                    cout << "Time to calculate move: " << elapsed_secs << endl;

                    next_move->setX(temp->getX());
                    next_move->setY(temp->getY());

                    cout << "CPU move: " << next_move->getX() + 1 << " " << next_move->getY() + 1<< endl;
                    delete temp;
                }
                else {
                    if (count == 0) {
                        last_move->setX(-1);
                    }
                    clock_t begin = clock();
                    Move* temp = gpu->get_move(last_move);
                    clock_t end = clock();
                    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
                    cout << "Time to calculate move: " << elapsed_secs << endl;

                    next_move->setX(temp->getX());
                    next_move->setY(temp->getY());

                    cout << "GPU move: " << next_move->getX() + 1 << " " << next_move->getY() + 1<< endl;
                    delete temp;
                }
            }

            int ret;
            if (count % 2 == 0) {
                ret = board.doMove(next_move, EX);
            }
            else {
                ret = board.doMove(next_move, OH);
            }
            if (board.checkDone(EX))
            {
                cout << "X has won!" << endl;
                break;
            }
            if (board.checkDone(OH))
            {
                cout << "O has won!" << endl;
                break;
            }
            if (ret == 0) {
                count += 1;
            }
        }
        board.print_board();
    }
}
