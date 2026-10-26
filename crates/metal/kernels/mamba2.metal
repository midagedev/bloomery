// The Mamba-2 decode kernels: the causal conv, the state scan step and the
// grouped RMS norm over its output. No kernel is written yet; the file is in
// build.rs's list so the kernels that land here edit this file alone.
//
// This file comes after delta.metal, whose conv ring and state-lane scheme
// the step follows, and fault.metal.

#include <metal_stdlib>
