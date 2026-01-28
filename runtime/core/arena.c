#include "arena.h"
#include <stdlib.h>

static size_t bwpp_align_up(size_t v, size_t align) {
  size_t mask = align - 1;
  return (v + mask) & ~mask;
}

int bwpp_arena_init(BwppArena *arena, size_t capacity) {
  arena->buffer = (unsigned char *)malloc(capacity);
  if (!arena->buffer) {
    return 0;
  }
  arena->capacity = capacity;
  arena->offset = 0;
  return 1;
}

void bwpp_arena_reset(BwppArena *arena) {
  arena->offset = 0;
}

void *bwpp_arena_alloc(BwppArena *arena, size_t size, size_t align) {
  size_t start = bwpp_align_up(arena->offset, align ? align : 1);
  if (start + size > arena->capacity) {
    return NULL;
  }
  void *ptr = arena->buffer + start;
  arena->offset = start + size;
  return ptr;
}

void bwpp_arena_destroy(BwppArena *arena) {
  free(arena->buffer);
  arena->buffer = NULL;
  arena->capacity = 0;
  arena->offset = 0;
}
