#include "game.hpp"

int main(int argc, char *argv[])
{
    if (argc < 2) {
        cout << "Playing with two humans" << endl;
        Game gomoku_game;
        Board board;
        gomoku_game.Run(board, "human", 0, 0);
    }
    else if (argc == 4) {
        int comp_start;
        if (strcmp(argv[1], "cpu") == 0) {
            cout << "Playing vs CPU" << endl;
            cout << "CPU is " << argv[2] << endl;
        }
        else {
            cout << "Playing vs GPU" << endl;
            cout << "GPU is " << argv[2] << endl;
        }
        if (strcmp(argv[2], "X") == 0) {
            comp_start = 0;
        }
        else {
            comp_start = 1;
        }
        Game gomoku_game;
        Board board;
        gomoku_game.Run(board, argv[1], comp_start, stoi(argv[3]));
    }
    else {
        cerr << "Incorrect number of arguments.\n";
        cerr << "Arguments: ./run opponent_type [ai_type ai_start depth] " << endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}
