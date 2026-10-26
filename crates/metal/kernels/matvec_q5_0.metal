// The Q5_0 entries of the quantized mat-vec: instantiations of the row loop
// matvec_q4k.metal owns over the Q5_0 decode owner (quant_b32.metal), under
// the names the host launches. No entry is written yet; the file is in
// build.rs's list so the entries that land here edit this file alone.
//
// This file comes after matvec_q4k.metal (its row loop and entry templates)
// and quant_b32.metal.

#include <metal_stdlib>
