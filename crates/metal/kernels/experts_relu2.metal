// The routed experts of a model whose expert body is a squared ReLU with no
// gate: the up kernel and the folded down, over the 32-value block owners.
// No kernel is written yet; the file is in build.rs's list so the kernels
// that land here edit this file alone.
//
// This file comes after experts.metal (the fold it follows), relu2.metal and
// quant_b32.metal.

#include <metal_stdlib>
