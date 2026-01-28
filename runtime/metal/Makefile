CC ?= clang
CFLAGS ?= -fobjc-arc -framework Metal -framework Foundation
BWPP_ROOT ?= ../..
BWPP_COMPILER ?= $(BWPP_ROOT)/compiler/bwppc
BWPP_EXAMPLES ?= $(BWPP_ROOT)/examples
BWPP_METAL_OUT ?= .metal_out

.PHONY: all clean metal-tests

all: bwpp_metal_matmul_test bwpp_metal_norm_test bwpp_metal_attention_test bwpp_metal_bench

bwpp_metal_matmul_test: test_matmul.m dispatch_stub.m dispatch_stub.h
	$(CC) $(CFLAGS) -o $@ test_matmul.m dispatch_stub.m

bwpp_metal_norm_test: test_norm.m dispatch_stub.m dispatch_stub.h
	$(CC) $(CFLAGS) -o $@ test_norm.m dispatch_stub.m

bwpp_metal_attention_test: test_attention.m dispatch_stub.m dispatch_stub.h
	$(CC) $(CFLAGS) -o $@ test_attention.m dispatch_stub.m

bwpp_metal_bench: bench_metal.m dispatch_stub.m dispatch_stub.h
	$(CC) $(CFLAGS) -o $@ bench_metal.m dispatch_stub.m

metal-tests: all
	$(MAKE) -C $(BWPP_ROOT)/compiler
	@mkdir -p $(BWPP_METAL_OUT)
	$(BWPP_COMPILER) $(BWPP_EXAMPLES)/matmul_add_silu.bwpp $(BWPP_METAL_OUT)/matmul_add_silu.metal
	$(BWPP_COMPILER) $(BWPP_EXAMPLES)/norms.bwpp $(BWPP_METAL_OUT)/norms.metal
	$(BWPP_COMPILER) $(BWPP_EXAMPLES)/attention.bwpp $(BWPP_METAL_OUT)/attention.metal
	$(BWPP_COMPILER) $(BWPP_EXAMPLES)/tiny_model.bwpp $(BWPP_METAL_OUT)/tiny_model.metal --entry tiny_model
	./bwpp_metal_matmul_test $(BWPP_METAL_OUT)/matmul_add_silu.metal
	./bwpp_metal_norm_test $(BWPP_METAL_OUT)/norms.metal
	./bwpp_metal_attention_test $(BWPP_METAL_OUT)/attention.metal

clean:
	rm -f bwpp_metal_matmul_test bwpp_metal_norm_test bwpp_metal_attention_test bwpp_metal_bench
