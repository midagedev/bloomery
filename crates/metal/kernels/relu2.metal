// The squared ReLU: max(x, 0)^2 with a NaN input kept a NaN, as an
// elementwise kernel and as the helper the squared-ReLU experts call. No
// kernel is written yet; the file is in build.rs's list so the kernel that
// lands here edits this file alone.
//
// This file comes after fault.metal and before experts_relu2.metal.

#include <metal_stdlib>
