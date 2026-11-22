// The attention output gate of an Inkling 2 visit: y = kqv_out ⊙ sigmoid(g),
// one gate value a channel — the wqkv_gate projection's output over the
// attention's, both hidden wide. One thread a value, one dispatch.
//
// The gate's rounding points, the reference's elementwise order: the
// sigmoid is 1 / (1 + exp(-g)) rounded at each step, the product rounds
// last.
//
// No silent failure: an output that is not finite (an input that already
// was not, or a product past the range) raises the product site and the
// value written there is NaN, never a plausible gate.
//
// This file comes after fault.metal (raise).

#pragma METAL fp contract(off)

// The launch's argument block, bound by value.
struct OutGateArgs {
    uint n;      // values
    uint layer;
};

static_assert(sizeof(OutGateArgs) == 8, "OutGateArgs is 8 bytes: the host packs two words");

kernel void out_gate_proj(device const float *kqv [[buffer(0)]],
                          device const float *g [[buffer(1)]],
                          device float *y [[buffer(2)]],
                          constant OutGateArgs &args [[buffer(3)]],
                          device atomic_uint *faults [[buffer(4)]],
                          uint i [[thread_position_in_grid]])
{
    if (i >= args.n) {
        return;
    }
    const float gate = 1.0f / (1.0f + exp(-g[i]));
    const float out = kqv[i] * gate;
    if (!isfinite(out)) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    y[i] = isfinite(out) ? out : NAN;
}
