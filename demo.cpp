#include "board.hpp"
#include "common.hpp"
#include "cpu_player.hpp"
#include "gpu_player.hpp"
#include <ctime>

int main(int argc, char *argv[])
{
    cout << "Demo script" << endl;
    cout << "This game is called Gomoku. The object of the game is to get five of your pieces in a row (like tic-tac-toe but 5 instead of 3)." << endl;
    cout << "We will play a demo game against both a cpu and gpu based AI with a minimax depth of 5" << endl;
    cout << "This demo script will ensure that both AI's return the same move, and then it will also compare the time it took to compute those moves" << endl;

    Board cpu_board;
    Board gpu_board;
    CpuAI* cpu = new CpuAI(OH, 5);
    GpuAI* gpu = new GpuAI(OH, 5);
    Move* last_move = new Move(7,7);

    cpu_board.print_board();
    cout << "This is an empty board. We are X and the AI's will be O." << endl;
    cout << "Our first move is (8,8)" << endl;

    cpu_board.doMove(last_move, EX);
    gpu_board.doMove(last_move, EX);
    cpu_board.print_board();

    clock_t begin = clock();
    Move* next_move_cpu = cpu->get_move(last_move);
    clock_t end = clock();

    cout << "CPU response: " << "(" << next_move_cpu->getX() + 1 << "," << next_move_cpu->getY() + 1 << ")" << endl;

    double cpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for CPU to calculate move: " << cpu_elapsed_secs << endl;

    cpu_board.doMove(next_move_cpu, OH);
    delete next_move_cpu;

    cpu_board.print_board();


    begin = clock();
    Move* next_move_gpu = gpu->get_move(last_move);
    end = clock();

    cout << "GPU response: " << "(" << next_move_gpu->getX() + 1 << "," << next_move_gpu->getY() + 1 << ")" << endl;

    double gpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for GPU to calculate move: " << gpu_elapsed_secs << endl;

    gpu_board.doMove(next_move_gpu, OH);
    delete next_move_gpu;

    gpu_board.print_board();

    cout << "The speedup of the gpu to the cpu was: " << cpu_elapsed_secs / gpu_elapsed_secs << endl << endl;

    cout << "Next, we will move to (9,7)" << endl;

    last_move->setX(8);
    last_move->setY(6);

    cpu_board.doMove(last_move, EX);
    gpu_board.doMove(last_move, EX);
    cpu_board.print_board();

    begin = clock();
    next_move_cpu = cpu->get_move(last_move);
    end = clock();

    cout << "CPU response: " << "(" << next_move_cpu->getX() + 1 << "," << next_move_cpu->getY() + 1 << ")" << endl;

    cpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for CPU to calculate move: " << cpu_elapsed_secs << endl;

    cpu_board.doMove(next_move_cpu, OH);
    delete next_move_cpu;

    cpu_board.print_board();


    begin = clock();
    next_move_gpu = gpu->get_move(last_move);
    end = clock();

    cout << "GPU response: " << "(" << next_move_gpu->getX() + 1 << "," << next_move_gpu->getY() + 1 << ")" << endl;

    gpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for GPU to calculate move: " << gpu_elapsed_secs << endl;

    gpu_board.doMove(next_move_gpu, OH);
    delete next_move_gpu;

    gpu_board.print_board();

    cout << "The speedup of the gpu to the cpu was: " << cpu_elapsed_secs / gpu_elapsed_secs << endl << endl;

    cout << "Next, we will move to (7,8)" << endl;

    last_move->setX(6);
    last_move->setY(7);

    cpu_board.doMove(last_move, EX);
    gpu_board.doMove(last_move, EX);
    cpu_board.print_board();

    begin = clock();
    next_move_cpu = cpu->get_move(last_move);
    end = clock();

    cout << "CPU response: " << "(" << next_move_cpu->getX() + 1 << "," << next_move_cpu->getY() + 1 << ")" << endl;

    cpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for CPU to calculate move: " << cpu_elapsed_secs << endl;

    cpu_board.doMove(next_move_cpu, OH);
    delete next_move_cpu;

    cpu_board.print_board();


    begin = clock();
    next_move_gpu = gpu->get_move(last_move);
    end = clock();

    cout << "GPU response: " << "(" << next_move_gpu->getX() + 1 << "," << next_move_gpu->getY() + 1 << ")" << endl;

    gpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for GPU to calculate move: " << gpu_elapsed_secs << endl;

    gpu_board.doMove(next_move_gpu, OH);
    delete next_move_gpu;

    gpu_board.print_board();

    cout << "The speedup of the gpu to the cpu was: " << cpu_elapsed_secs / gpu_elapsed_secs << endl << endl;

    cout << "Next, we will move to (8,6)" << endl;

    last_move->setX(7);
    last_move->setY(5);

    cpu_board.doMove(last_move, EX);
    gpu_board.doMove(last_move, EX);
    cpu_board.print_board();

    begin = clock();
    next_move_cpu = cpu->get_move(last_move);
    end = clock();

    cout << "CPU response: " << "(" << next_move_cpu->getX() + 1 << "," << next_move_cpu->getY() + 1 << ")" << endl;

    cpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for CPU to calculate move: " << cpu_elapsed_secs << endl;

    cpu_board.doMove(next_move_cpu, OH);
    delete next_move_cpu;

    cpu_board.print_board();


    begin = clock();
    next_move_gpu = gpu->get_move(last_move);
    end = clock();

    cout << "GPU response: " << "(" << next_move_gpu->getX() + 1 << "," << next_move_gpu->getY() + 1 << ")" << endl;

    gpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for GPU to calculate move: " << gpu_elapsed_secs << endl;

    gpu_board.doMove(next_move_gpu, OH);
    delete next_move_gpu;

    gpu_board.print_board();

    cout << "The speedup of the gpu to the cpu was: " << cpu_elapsed_secs / gpu_elapsed_secs << endl << endl;

    cout << "Next, we will move to (8,7)" << endl;

    last_move->setX(7);
    last_move->setY(6);

    cpu_board.doMove(last_move, EX);
    gpu_board.doMove(last_move, EX);
    cpu_board.print_board();

    begin = clock();
    next_move_cpu = cpu->get_move(last_move);
    end = clock();

    cout << "CPU response: " << "(" << next_move_cpu->getX() + 1 << "," << next_move_cpu->getY() + 1 << ")" << endl;

    cpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for CPU to calculate move: " << cpu_elapsed_secs << endl;

    cpu_board.doMove(next_move_cpu, OH);
    delete next_move_cpu;

    cpu_board.print_board();


    begin = clock();
    next_move_gpu = gpu->get_move(last_move);
    end = clock();

    cout << "GPU response: " << "(" << next_move_gpu->getX() + 1 << "," << next_move_gpu->getY() + 1 << ")" << endl;

    gpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for GPU to calculate move: " << gpu_elapsed_secs << endl;

    gpu_board.doMove(next_move_gpu, OH);
    delete next_move_gpu;

    gpu_board.print_board();

    cout << "The speedup of the gpu to the cpu was: " << cpu_elapsed_secs / gpu_elapsed_secs << endl << endl;

    cout << "Next, we will move to (8,10) to attempt to block the AI's four" << endl;

    last_move->setX(7);
    last_move->setY(9);

    cpu_board.doMove(last_move, EX);
    gpu_board.doMove(last_move, EX);
    cpu_board.print_board();

    begin = clock();
    next_move_cpu = cpu->get_move(last_move);
    end = clock();

    cout << "CPU response: " << "(" << next_move_cpu->getX() + 1 << "," << next_move_cpu->getY() + 1 << ")" << endl;

    cpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for CPU to calculate move: " << cpu_elapsed_secs << endl;

    cpu_board.doMove(next_move_cpu, OH);
    delete next_move_cpu;

    cpu_board.print_board();


    begin = clock();
    next_move_gpu = gpu->get_move(last_move);
    end = clock();

    cout << "GPU response: " << "(" << next_move_gpu->getX() + 1 << "," << next_move_gpu->getY() + 1 << ")" << endl;

    gpu_elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Time for GPU to calculate move: " << gpu_elapsed_secs << endl;

    gpu_board.doMove(next_move_gpu, OH);
    delete next_move_gpu;

    gpu_board.print_board();

    cout << "The speedup of the gpu to the cpu was: " << cpu_elapsed_secs / gpu_elapsed_secs << endl << endl;

    cout << "We see that both the CPU and GPU have won the game with the same exact moves (hence proving the correctness of our GPU kernel in performing the tree search), and we get a speedup in later parts of the game (when the game tree has a much larger branching factor due to the number of potential good moves) of roughly 20x" << endl;

    return 0;
}
