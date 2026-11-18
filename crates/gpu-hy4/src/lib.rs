//! bloomery-gpu-hy4 — the Hy4-flash program on one card: a whole-card body over `bloomery-gpu`'s
//! kernels, host code only (llama.cpp PR #32517, `src/models/hy-v4-flash.cpp`). The model
//! description is `model::arch::hy4`.
//!
//! One token a step, every layer on the card ([`body::Body`] behind `GpuModel`):
//!
//! 1. the embedding row, read and dequantized on the host, laid into the four residual streams;
//! 2. on an engram site, the delta of the rows the hash of the token's n-grams names (the
//!    rows gathered on the host from the file's table, normed, projected, gated per stream and
//!    added under the token's mask) — before the layer's attention mix reads the streams;
//! 3. the attention sub-layer: the identity mix's fold of the streams, `attn_norm`, the gated
//!    latent attention (the q and kv projections, the rope of the 64 rotated values, the
//!    576-wide cache row (every position on a sparse layer, slot `pos % W` of a ring on a window
//!    layer), the k-pool selector's list on a sparse layer, the absorbed query, the latent flash
//!    with a sink a head over the list or the ring, the value projection at 256 a head, the output
//!    gate, `attn_output`) and its sandwich norm, then the identity mix's post into the streams;
//! 4. the feed-forward sub-layer the same way around the dense SiLU block (the lead layer) or the
//!    routed block (the sigmoid router on the full-width row, the latent down, the card leg's
//!    experts under the gate-first clamp, the latent up, plus the shared SiLU expert);
//! 5. the head's fold of the four streams into the output head's input; the head (`output_norm`,
//!    the projection, the argmax) is the common one.
//!
//! A file whose routed experts the card cannot hold loads by a placement plan
//! ([`Body::open_placed`]): each routed layer keeps the plan's card share of its stacks and hands
//! the rest of its picks to the common host tier, whose sum the card adds before the latent up
//! ([`placed`]'s three parts around the host leg).
//!
//! What the file forces here, each named where it is used: the identity hyper-connections, the
//! sandwich norms, the per-layer head count and rope base, the window ring, the forced-tail
//! selector, the sink, the 256-wide values, the latent routed block and its clamp, the engram site.

mod alone;
pub mod batch;
pub mod body;
mod placed;
mod rooms;
mod step;
pub mod tier;

pub use alone::{MixAlone, Picks, Selection};
pub use batch::{
    Cfg, PASS_ROWS, PassShape, PlacedPass, Prompt, SHAPES, pass_bytes, prompt, prompt_placed,
};
pub use body::{Body, Seq, TAP_STAGES, TapRows};
pub use placed::Placed;
