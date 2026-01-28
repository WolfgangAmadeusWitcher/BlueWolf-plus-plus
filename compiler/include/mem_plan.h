#ifndef BWPP_MEM_PLAN_H
#define BWPP_MEM_PLAN_H

#include "graph_ir.h"
#include <stdint.h>
#include <stdio.h>

typedef struct {
  BwppShape shape;
  BwppDType dtype;
  BwppLayout layout;
} BwppBufferDesc;

typedef struct {
  BwppBufferDesc *buffers;
  uint32_t buffer_count;
  uint32_t buffer_capacity;
  uint32_t *value_to_buffer;
  uint32_t value_count;
} BwppMemPlan;

BwppMemPlan *bwpp_mem_plan_build(const BwppGraph *graph);
void bwpp_mem_plan_dump(const BwppMemPlan *plan, FILE *out);
void bwpp_mem_plan_destroy(BwppMemPlan *plan);

#endif
