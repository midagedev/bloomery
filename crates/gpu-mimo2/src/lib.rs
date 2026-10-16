//! bloomery-gpu-mimo2 — the MiMo-V2.6-Flash program: a body over the
//! runtime's layer walk (`runtime::sched`) whose layers are the programs
//! their descriptions name (`runtime::layer`). It has no device code of its
//! own: the q8_0 gemvs, the K192 rope-and-append, the K192 flash with its
//! sink merge, the host handoff and the head are `bloomery-gpu`'s, the
//! shared-expert gate·up and the 256-expert sigmoid router
//! `bloomery-gpu-deepseek41`'s. Every routed expert runs on the host tier.
//!
//! - [`body`]: the load, the stores, the step's buffers and [`body::Body`]
//!   behind `GpuModel`;
//! - `program`: the step's walk and its host leg's port;
//! - [`body::prefill`]: the prompt fed in batches, bit for bit the steps;
//! - `attn`: the attention sub-layer;
//! - `ffn`: the dense block and the routed block around its host leg.
//!
//! What each layer's launches take, and the step's node counts, are facts of
//! the file's description (`model::arch::mimo2::program`), where their tests
//! run; this crate has none of its own.
//!
//! Built with `cargo oxide`, as the device crates it links are.

mod attn;
pub mod body;
mod ffn;
mod program;

pub use body::prefill::{
    FLASH_ROWS, PrefillMode, PromptBytes, T_MAX, feed, prefill, prefill_group, prefill_mode,
    prompt_bytes, set_prefill, set_prefill_group,
};
pub use body::{Body, Mimo2Model, StepInput, set_taps};
