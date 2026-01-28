CC ?= clang
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I../runtime/cpu

all: bwpp_bench

bwpp_bench: bench_cpu.c ../runtime/cpu/bwpp_cpu_ref.c
	$(CC) $(CFLAGS) -o $@ bench_cpu.c ../runtime/cpu/bwpp_cpu_ref.c -lm

compare: bwpp_bench
	python3 bench_compare.py --iters 10 --m 256 --n 256 --k 256

clean:
	rm -f bwpp_bench
