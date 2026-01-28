#ifndef BWPP_TENSOR_H
#define BWPP_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
  BWPP_DTYPE_F16 = 0,
  BWPP_DTYPE_BF16,
  BWPP_DTYPE_F32
} BwppDType;

typedef enum {
  BWPP_LAYOUT_ROW_MAJOR = 0,
  BWPP_LAYOUT_COL_MAJOR
} BwppLayout;

typedef struct {
  BwppDType dtype;
  BwppLayout layout;
  uint32_t rank;
  uint64_t shape[4];
  uint64_t stride[4];
  void *buffer;
} BwppTensor;

#endif
