// The fault word: word 0 is the first-layer word, word 1 + layer is that
// layer's site mask. A raise stores the layer above the site with a
// minimum, and ORs the site's bit into its own layer's mask. A layer past
// BLOOMERY_LAYER_NONE has no mask word: word 0 still names the fault, and
// no mask is written, so every access stays inside the word.

#include <metal_stdlib>

using namespace metal;

#pragma METAL fp contract(off)

static void raise(device atomic_uint *faults, uint layer, uint site)
{
    atomic_fetch_min_explicit(&faults[0],
                              BLOOMERY_FAULT_WORD(layer, site),
                              memory_order_relaxed);
    if (layer <= BLOOMERY_LAYER_NONE) {
        atomic_fetch_or_explicit(&faults[1 + layer],
                                 1u << site,
                                 memory_order_relaxed);
    }
}

kernel void raise_fault(device atomic_uint *faults [[buffer(0)]],
                        constant uint &layer [[buffer(1)]],
                        constant uint &site [[buffer(2)]])
{
    raise(faults, layer, site);
}

// One raise per thread, each with its own layer and site: the raises race.
kernel void raise_fault_lanes(device atomic_uint *faults [[buffer(0)]],
                              constant uint *layers [[buffer(1)]],
                              constant uint *sites [[buffer(2)]],
                              constant uint &count [[buffer(3)]],
                              uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        raise(faults, layers[gid], sites[gid]);
    }
}
