#include "mem_plan.h"
#include <stdlib.h>
#include <string.h>

static int bwpp_shape_equal(const BwppShape *a, const BwppShape *b) {
  if (a->rank != b->rank) {
    return 0;
  }
  for (uint32_t i = 0; i < a->rank; ++i) {
    if (a->dims[i].len != b->dims[i].len ||
        strncmp(a->dims[i].ptr, b->dims[i].ptr, a->dims[i].len) != 0) {
      return 0;
    }
  }
  return 1;
}

static int bwpp_buffer_match(const BwppBufferDesc *a, const BwppBufferDesc *b) {
  return a->dtype == b->dtype && a->layout == b->layout && bwpp_shape_equal(&a->shape, &b->shape);
}

static uint32_t bwpp_add_buffer(BwppMemPlan *plan, BwppBufferDesc desc) {
  if (plan->buffer_count == plan->buffer_capacity) {
    uint32_t new_cap = plan->buffer_capacity == 0 ? 8 : plan->buffer_capacity * 2;
    BwppBufferDesc *nb = (BwppBufferDesc *)realloc(plan->buffers, new_cap * sizeof(BwppBufferDesc));
    if (!nb) {
      return UINT32_MAX;
    }
    plan->buffers = nb;
    plan->buffer_capacity = new_cap;
  }
  plan->buffers[plan->buffer_count] = desc;
  return plan->buffer_count++;
}

static void bwpp_shape_copy(BwppShape *dst, const BwppShape *src) {
  dst->rank = src->rank;
  for (uint32_t i = 0; i < src->rank && i < BWPP_GRAPH_MAX_DIMS; ++i) {
    dst->dims[i] = src->dims[i];
  }
}

BwppMemPlan *bwpp_mem_plan_build(const BwppGraph *graph) {
  if (!graph) {
    return NULL;
  }
  BwppMemPlan *plan = (BwppMemPlan *)calloc(1, sizeof(BwppMemPlan));
  if (!plan) {
    return NULL;
  }
  plan->value_count = graph->value_count;
  plan->value_to_buffer = (uint32_t *)malloc(sizeof(uint32_t) * graph->value_count);
  if (!plan->value_to_buffer) {
    bwpp_mem_plan_destroy(plan);
    return NULL;
  }
  for (uint32_t i = 0; i < graph->value_count; ++i) {
    plan->value_to_buffer[i] = UINT32_MAX;
  }

  uint32_t *last_use = (uint32_t *)malloc(sizeof(uint32_t) * graph->value_count);
  if (!last_use) {
    bwpp_mem_plan_destroy(plan);
    return NULL;
  }
  for (uint32_t i = 0; i < graph->value_count; ++i) {
    last_use[i] = 0;
  }
  for (uint32_t i = 0; i < graph->node_count; ++i) {
    const BwppGraphNode *n = &graph->nodes[i];
    for (uint32_t j = 0; j < n->input_count; ++j) {
      uint32_t v = n->inputs[j];
      if (v < graph->value_count) {
        last_use[v] = i;
      }
    }
  }
  for (uint32_t i = 0; i < graph->output_count; ++i) {
    uint32_t v = graph->outputs[i];
    if (v < graph->value_count) {
      last_use[v] = graph->node_count;
    }
  }

  uint32_t *free_list = NULL;
  uint32_t free_count = 0;
  uint32_t free_capacity = 0;

  for (uint32_t i = 0; i < graph->node_count; ++i) {
    const BwppGraphNode *n = &graph->nodes[i];
    for (uint32_t j = 0; j < n->input_count; ++j) {
      uint32_t v = n->inputs[j];
      if (v >= graph->value_count) {
        continue;
      }
      if (last_use[v] == i && plan->value_to_buffer[v] != UINT32_MAX) {
        if (free_count == free_capacity) {
          uint32_t new_cap = free_capacity == 0 ? 8 : free_capacity * 2;
          uint32_t *nf = (uint32_t *)realloc(free_list, new_cap * sizeof(uint32_t));
          if (!nf) {
            free_list = NULL;
            free_capacity = 0;
            free_count = 0;
            break;
          }
          free_list = nf;
          free_capacity = new_cap;
        }
        free_list[free_count++] = plan->value_to_buffer[v];
      }
    }

    uint32_t out = n->output;
    if (out >= graph->value_count) {
      continue;
    }
    const BwppGraphValue *v = &graph->values[out];
    if (v->flags & (BWPP_GRAPH_VALUE_INPUT | BWPP_GRAPH_VALUE_CONST)) {
      continue;
    }

    BwppBufferDesc desc = {0};
    desc.dtype = v->dtype;
    desc.layout = v->layout;
    bwpp_shape_copy(&desc.shape, &v->shape);

    uint32_t chosen = UINT32_MAX;
    for (uint32_t f = 0; f < free_count; ++f) {
      uint32_t buf_id = free_list[f];
      if (buf_id < plan->buffer_count && bwpp_buffer_match(&plan->buffers[buf_id], &desc)) {
        chosen = buf_id;
        free_list[f] = free_list[free_count - 1];
        free_count--;
        break;
      }
    }
    if (chosen == UINT32_MAX) {
      chosen = bwpp_add_buffer(plan, desc);
    }
    plan->value_to_buffer[out] = chosen;
  }

  free(last_use);
  free(free_list);
  return plan;
}

static const char *bwpp_dtype_name(BwppDType dt) {
  switch (dt) {
    case BWPP_DTYPE_F16: return "f16";
    case BWPP_DTYPE_BF16: return "bf16";
    case BWPP_DTYPE_F32: return "f32";
    default: return "unknown";
  }
}

static const char *bwpp_layout_name(BwppLayout l) {
  switch (l) {
    case BWPP_LAYOUT_ROW_MAJOR: return "row_major";
    case BWPP_LAYOUT_COL_MAJOR: return "col_major";
    default: return "unknown";
  }
}

static void bwpp_print_shape(FILE *out, const BwppShape *shape) {
  fprintf(out, "[");
  for (uint32_t i = 0; i < shape->rank; ++i) {
    if (i) {
      fprintf(out, ",");
    }
    fprintf(out, "%.*s", (int)shape->dims[i].len, shape->dims[i].ptr);
  }
  fprintf(out, "]");
}

void bwpp_mem_plan_dump(const BwppMemPlan *plan, FILE *out) {
  if (!plan || !out) {
    return;
  }
  fprintf(out, "buffers=%u values=%u\n", plan->buffer_count, plan->value_count);
  for (uint32_t i = 0; i < plan->buffer_count; ++i) {
    const BwppBufferDesc *b = &plan->buffers[i];
    fprintf(out, "buffer%u %s ", i, bwpp_dtype_name(b->dtype));
    bwpp_print_shape(out, &b->shape);
    fprintf(out, " %s\n", bwpp_layout_name(b->layout));
  }
  for (uint32_t i = 0; i < plan->value_count; ++i) {
    if (plan->value_to_buffer[i] == UINT32_MAX) {
      continue;
    }
    fprintf(out, "v%u -> buffer%u\n", i, plan->value_to_buffer[i]);
  }
}

void bwpp_mem_plan_destroy(BwppMemPlan *plan) {
  if (!plan) {
    return;
  }
  free(plan->buffers);
  free(plan->value_to_buffer);
  free(plan);
}
