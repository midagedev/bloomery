//! bloomery-gpu-glm5next — the GLM-5.3-Flash program: a body over the
//! runtime's layer walk (`runtime::sched`) whose layers are the programs
//! their descriptions name (`runtime::layer`). It has no device code of its
//! own: the KDA delta rule, the latent appends, the q8_0 gemvs and the head
//! are `bloomery-gpu`'s, the hyper-connections, the absorbed attention, the
//! GLM router, the shared expert, the host handoff and the card slots' sum
//! `bloomery-gpu-deepseek41`'s, the routed experts' `_sel` entries
//! `bloomery-gpu`'s (`kquant`, `q4k_sel`). The routed experts run on the card
//! where the plan puts them, on the host tier otherwise.
//!
//! - [`body`]: the load, the stores, the step's buffers and [`body::Body`]
//!   behind `GpuModel`;
//! - `program`: the step's walk and its host leg's port;
//! - [`body::prefill`]: the prompt fed in batches, bit for bit the steps;
//! - `kda`, `mla`: the two mixers; `ffn`: the dense block, and the routed
//!   block around its host leg with its card experts in the leg's shadow;
//! - `host`: the host tier's routed stacks;
//! - `tensors`: each layer's tensor names, made at load;
//! - [`forced`]: one layer alone on given streams, for the gates.
//!
//! Built with `cargo oxide`, as the device crates it links are.

pub mod body;
mod ffn;
pub mod forced;
mod host;
mod kda;
mod mla;
mod program;
mod tensors;

pub use body::prefill::{
    CHUNK, PrefillMode, StoreDigest, T_MAX, batches_of, call_batches, feed, prefill, prefill_mode,
    set_prefill, store_digests,
};
pub use body::{Body, CHECKPOINT_EVERY, Glm5nextModel, Plant, prompt, set_taps};
pub use host::GlmHost;
pub use program::{layer_launches, step_launches};
