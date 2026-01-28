#ifndef BWPP_ARENA_H
#define BWPP_ARENA_H

#include <stddef.h>

typedef struct {
  unsigned char *buffer;
  size_t capacity;
  size_t offset;
} BwppArena;

int bwpp_arena_init(BwppArena *arena, size_t capacity);
void bwpp_arena_reset(BwppArena *arena);
void *bwpp_arena_alloc(BwppArena *arena, size_t size, size_t align);
void bwpp_arena_destroy(BwppArena *arena);

#endif
