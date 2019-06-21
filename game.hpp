#include "board.hpp"
#include "common.hpp"
#include "cpu_player.hpp"
#include "gpu_player.hpp"
#include <ctime>

/*
* A game class which stores the state of the game, allows user input and
* runs the game
*/
class Game
{
public:
    void Run(Board board, string input, int comp_start, int depth);
};
