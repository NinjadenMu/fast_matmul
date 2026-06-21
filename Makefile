CC = clang

IMPL_SRCS := $(wildcard src/*.c)
IMPL_OBJS := $(patsubst src/%.c, obj/%.o, $(IMPL_SRCS))

BENCH_SRC = bench/bench.c
BENCH_OBJ = obj/bench.o

ALL_OBJS := $(IMPL_OBJS) $(BENCH_OBJ)

# replacing -march with -mcpu may be a bit cleaner and faster
# BASE_FLAGS = -Wall -Wextra -Wpedantic -I./include -MMD -MP -ffast-math -mcpu=apple-m4
BASE_FLAGS = -Wall -Wextra -Wpedantic -I./include -MMD -MP -ffast-math -march=native

obj/matmul_naive.o: IMPL_FLAGS = -O3 -mllvm -enable-loopinterchange=false -fno-vectorize -fno-slp-vectorize
obj/matmul_permuted.o: IMPL_FLAGS = -O3 -mllvm -enable-loopinterchange=false -fno-vectorize -fno-slp-vectorize
obj/matmul_tiled.o: IMPL_FLAGS = -O3 -fno-vectorize -fno-slp-vectorize
obj/matmul_micro_kernel.o: IMPL_FLAGS = -O3 -fno-vectorize -fno-slp-vectorize
obj/matmul_vectorized.o: IMPL_FLAGS = -O3

.PHONY: all
all: bench_exe

bench_exe: $(ALL_OBJS)
	$(CC) $(ALL_OBJS) -o $@ $(LDFLAGS)

obj/bench.o: bench/bench.c include/matmul.h | obj
	$(CC) $(BASE_FLAGS) -O0 -c $< -o $@ 

obj/%.o: src/%.c | obj
	$(CC) $(BASE_FLAGS) $(IMPL_FLAGS) -c $< -o $@

obj:
	mkdir obj

.PHONY: format
format:
	clang-format -i $(wildcard bench/*.c bench/*.h include/*.c include/*.h src/*.c src/*.h)

.PHONY: clean
clean:
	rm -rf obj bench_exe
