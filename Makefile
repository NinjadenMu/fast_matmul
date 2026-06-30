CC = clang

OMP_PREFIX := $(shell brew --prefix libomp)

IMPL_SRCS := $(wildcard src/*.c)
IMPL_OBJS := $(patsubst src/%.c, obj/src/%.o, $(IMPL_SRCS))

UTIL_SRCS := $(wildcard utils/*.c)
UTIL_OBJS := $(patsubst utils/%.c, obj/utils/%.o, $(UTIL_SRCS))

ALL_OBJS := $(IMPL_OBJS) $(UTIL_OBJS)

# replacing -march with -mcpu may work better depending on the machine (especially if it's older)
# BASE_FLAGS = -Wall -Wextra -Wpedantic -I./include -MMD -MP -ffast-math -mcpu=apple-m4
BASE_FLAGS = -O3 -Wall -Wextra -Wpedantic -I./include -MMD -MP -ffast-math -march=native
LDFLAGS = -L$(OMP_PREFIX)/lib -lomp -Wl,-rpath,$(OMP_PREFIX)/lib

obj/src/matmul_naive.o: IMPL_FLAGS = -mllvm -enable-loopinterchange=false -fno-vectorize -fno-slp-vectorize
obj/src/matmul_permuted.o: IMPL_FLAGS = -mllvm -enable-loopinterchange=false -fno-vectorize -fno-slp-vectorize
obj/src/matmul_tiled.o: IMPL_FLAGS = -fno-vectorize -fno-slp-vectorize
obj/src/matmul_micro_kernel.o: IMPL_FLAGS = -fno-vectorize -fno-slp-vectorize
# assumes OpenMP was installed on Mac using Homebrew
obj/src/matmul_parallel.o: IMPL_FLAGS = -Xpreprocessor -fopenmp -I"$(OMP_PREFIX)/include"

.PHONY: all
all: bench

bench: $(ALL_OBJS)
	$(CC) $(ALL_OBJS) -o $@ $(LDFLAGS)

obj/utils/%.o: utils/%.c | obj
	$(CC) $(BASE_FLAGS) -O0 -c $< -o $@ 

obj/src/%.o: src/%.c | obj
	$(CC) $(BASE_FLAGS) $(IMPL_FLAGS) -c $< -o $@

obj:
	mkdir -p obj/src obj/utils

.PHONY: format
format:
	clang-format -i $(wildcard utils/*.c utils/*.h include/*.c include/*.h src/*.c src/*.h)

.PHONY: clean
clean:
	rm -rf obj bench
