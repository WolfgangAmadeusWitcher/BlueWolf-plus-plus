#ifndef BWPP_CODEGEN_METAL_H
#define BWPP_CODEGEN_METAL_H

#include "bwpp.h"
#include "ir.h"

BwppStatus bwpp_codegen_metal(const BwppIrModule *ir, const char *out_path);

#endif
