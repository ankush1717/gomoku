
# Product Names
CUDA_OBJ = cuda.o
# Input Names
CUDA_FILES = kernels_minimax.cu
CPP_FILES = board.cpp game.cpp cpu_player.cpp gpu_player.cpp

# ------------------------------------------------------------------------------

# CUDA Compiler and Flags
CUDA_PATH = /usr/local/cuda
CUDA_INC_PATH = $(CUDA_PATH)/include
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64

NVCC = $(CUDA_BIN_PATH)/nvcc

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
NVCC_FLAGS := -m32
else
NVCC_FLAGS := -m64
endif
NVCC_FLAGS += -dc -Wno-deprecated-gpu-targets --std=c++11 \
             --expt-relaxed-constexpr
NVCC_INCLUDE =
NVCC_LIBS =
NVCC_GENCODES = \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 \
		-gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_61,code=compute_61

# CUDA Object Files
CUDA_OBJ_FILES = $(notdir $(addsuffix .o, $(CUDA_FILES)))

# ------------------------------------------------------------------------------

# CUDA Linker and Flags
CUDA_LINK_FLAGS = -dlink -Wno-deprecated-gpu-targets

# ------------------------------------------------------------------------------

# C++ Compiler and Flags
GPP = g++
FLAGS = -O3 -g -std=c++11 -Werror -Wall
INCLUDE = -I$(CUDA_INC_PATH)
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcufft -lsndfile

# ------------------------------------------------------------------------------
# Make Rules
# ------------------------------------------------------------------------------

# C++ Object Files
OBJ = $(notdir $(addsuffix .o, $(CPP_FILES)))

# Top level rules
all: interactive demo

interactive: gomoku.cpp.o $(OBJ) $(CUDA_OBJ) $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o run $(INCLUDE) $^ $(LIBS)

demo: demo.cpp.o $(OBJ) $(CUDA_OBJ) $(CUDA_OBJ_FILES)
	$(GPP) $(FLAGS) -o demo $(INCLUDE) $^ $(LIBS)

game.cpp.o: game.cpp game.hpp board.hpp common.hpp cpu_player.hpp gpu_player.hpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

gomoku.cpp.o: gomoku.cpp game.hpp board.hpp common.hpp cpu_player.hpp gpu_player.hpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

demo.cpp.o: demo.cpp board.hpp common.hpp cpu_player.hpp gpu_player.hpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

board.cpp.o: board.cpp board.hpp common.hpp
	$(NVCC) $(NVCC_FLAGS) -x cu $(NVCC_GENCODES) -dc -c -o $@ $(NVCC_INCLUDE) $<

cpu_player.cpp.o: cpu_player.cpp cpu_player.hpp board.hpp common.hpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

gpu_player.cpp.o: gpu_player.cpp gpu_player.hpp board.hpp common.hpp
	$(GPP) $(FLAGS) -c -o $@ $(INCLUDE) $<

# Compile CUDA Source Files
kernels_minimax.cu.o: kernels_minimax.cu kernels_minimax.cuh board.hpp common.hpp
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODES) -c -o $@ $(NVCC_INCLUDE) $<

cuda: $(CUDA_OBJ_FILES) $(CUDA_OBJ)

# Make linked device code
$(CUDA_OBJ): $(CUDA_OBJ_FILES) board.cpp.o
	$(NVCC) $(CUDA_LINK_FLAGS) $(NVCC_GENCODES) -o $@ $(NVCC_INCLUDE) $^


# Clean everything including temporary Emacs files
clean:
	rm -f *.o *~

.PHONY: clean
