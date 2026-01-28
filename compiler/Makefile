CC ?= clang
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror
INCLUDES = -Iinclude

SRCS = \
  main.c \
  lexer.c \
  parser.c \
  ast.c \
  typecheck.c \
  ir.c \
  graph_ir.c \
  tile_ir.c \
  codegen_metal.c

OBJS = $(SRCS:.c=.o)

all: bwppc

bwppc: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) bwppc
