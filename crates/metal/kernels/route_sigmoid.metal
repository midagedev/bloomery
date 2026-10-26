// The sigmoid router: the pick from the biased score, the weights from the
// unbiased one, renormalized and scaled. No kernel is written yet; the file
// is in build.rs's list so the kernel that lands here edits this file alone.
//
// This file comes after route.metal and router.metal (the selection and the
// ticket it follows).

#include <metal_stdlib>
