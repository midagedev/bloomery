// The situ activation of Inkling 2's experts, the combine of the pair:
//
//   h = gate_beta · tanh(g / gate_beta) · sigmoid(g) · value_beta · tanh(u / value_beta)
//
// both betas the launch's own words (the file's keys, 4.0 and 25.0 on the
// family's files) — the pair_swiglu of gemm_shared.metal is this backend's
// one activation epilogue today (silu(g)·u, its betas implicit); this file
// is the family's own, the combine over the two rows a pair's dots wrote.
//
// The dots of a pair are the segmented mat-vec's own dispatches (the gate
// rows and the up rows of the kept expert, norm-on-load as their launches
// feed them); this entry is the combine over their outputs, one thread a
// value: y[i] = situ(g[i], u[i]). The gate term rounds its product, its
// tanh and its sigmoid each once, the up term its quotient and its tanh,
// and the two halves' product rounds last — the reference's elementwise
// order.
//
// No silent failure: a beta that is not finite and positive, or an output
// that stops being finite (an input that already was not, or a product
// past the range), raises the product site and the value written there is
// NaN — never a plausible activation.
//
// This file comes after fault.metal (raise).

#pragma METAL fp contract(off)

// The activation itself, the reference's elementwise order.
inline float pair_situ(float gate, float up, float gate_beta, float value_beta) {
    const float g = gate_beta * tanh(gate / gate_beta) * (1.0f / (1.0f + exp(-gate)));
    const float u = value_beta * tanh(up / value_beta);
    return g * u;
}

// The launch's argument block, bound by value.
struct SituArgs {
    uint n;          // values a row
    uint layer;
    float gate_beta;
    float value_beta;
};

static_assert(sizeof(SituArgs) == 16, "SituArgs is 16 bytes: the host packs four words");

kernel void pair_situ_rows(device const float *gate [[buffer(0)]],
                           device const float *up [[buffer(1)]],
                           device float *y [[buffer(2)]],
                           constant SituArgs &args [[buffer(3)]],
                           device atomic_uint *faults [[buffer(4)]],
                           uint i [[thread_position_in_grid]])
{
    if (i >= args.n) {
        return;
    }
    const bool betas = isfinite(args.gate_beta) && args.gate_beta > 0.f &&
                       isfinite(args.value_beta) && args.value_beta > 0.f;
    const float h = betas ? pair_situ(gate[i], up[i], args.gate_beta, args.value_beta) : NAN;
    if (!betas || !isfinite(h)) {
        raise(faults, args.layer, BLOOMERY_FAULT_SITE_F32_PRODUCT);
    }
    y[i] = isfinite(h) ? h : NAN;
}
