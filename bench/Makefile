CC ?= clang
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror -I../runtime/cpu

all: bwpp_bench

bwpp_bench: bench_cpu.c ../runtime/cpu/bwpp_cpu_ref.c
	$(CC) $(CFLAGS) -o $@ bench_cpu.c ../runtime/cpu/bwpp_cpu_ref.c -lm

clean:
	rm -f bwpp_bench
