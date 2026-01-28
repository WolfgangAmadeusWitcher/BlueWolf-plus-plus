CC ?= clang
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror
BWPP_ROOT ?= ../..
BWPP_COMPILER ?= $(BWPP_ROOT)/compiler/bwppc
BWPP_EXAMPLES ?= $(BWPP_ROOT)/examples
BWPP_METAL_OUT ?= .metal_out

.PHONY: all clean cpu-metal-tests

all: bwpp_cpu_test bwpp_cpu_norm_test bwpp_cpu_metal_test bwpp_cpu_reduce_max_test

bwpp_cpu_test: bwpp_cpu_ref.c test_matmul.c
	$(CC) $(CFLAGS) -o $@ bwpp_cpu_ref.c test_matmul.c -lm

bwpp_cpu_norm_test: bwpp_cpu_ref.c test_norm.c
	$(CC) $(CFLAGS) -o $@ bwpp_cpu_ref.c test_norm.c -lm

bwpp_cpu_metal_test: bwpp_cpu_ref.c test_metal_parity.c
	$(CC) $(CFLAGS) -o $@ bwpp_cpu_ref.c test_metal_parity.c -lm

bwpp_cpu_reduce_max_test: bwpp_cpu_ref.c test_reduce_max.c
	$(CC) $(CFLAGS) -o $@ bwpp_cpu_ref.c test_reduce_max.c -lm

cpu-metal-tests: bwpp_cpu_metal_test
	$(MAKE) -C $(BWPP_ROOT)/compiler
	@mkdir -p $(BWPP_METAL_OUT)
	$(BWPP_COMPILER) $(BWPP_EXAMPLES)/matmul_add_silu.bwpp $(BWPP_METAL_OUT)/matmul_add_silu.metal
	$(BWPP_COMPILER) $(BWPP_EXAMPLES)/norms.bwpp $(BWPP_METAL_OUT)/norms.metal
	./bwpp_cpu_metal_test $(BWPP_METAL_OUT)/matmul_add_silu.metal
	./bwpp_cpu_metal_test $(BWPP_METAL_OUT)/norms.metal

clean:
	rm -f bwpp_cpu_test bwpp_cpu_norm_test bwpp_cpu_metal_test bwpp_cpu_reduce_max_test
