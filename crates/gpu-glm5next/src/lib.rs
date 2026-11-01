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
//! - [`body::prefill`]: the prompt fed in batches, bit for bit the steps up
//!   to a chunk a batch and the mixers' projections on the GEMM past it,
//!   with media spans' rows in place of their positions' embeddings
//!   (`body::media`);
//! - `body::pair`: the verify of two rows behind `Rows`, bit for bit two
//!   steps, and its commit over the KDA lanes;
//! - [`body::nextn`]: the next-token (MTP) layer beside the chain on a NextN
//!   load, and its walks;
//! - [`body::seq`]: a sequence's state on the host, saved and put back;
//! - `kda`, `mla`: the two mixers; `ffn`: the dense block, and the routed
//!   block around its host leg with its card experts in the leg's shadow;
//! - `host`: the host tier's routed stacks;
//! - [`swap`]: the model's side of adaptive expert residency;
//! - `tensors`: each layer's tensor names, made at load;
//! - `tier`: the expert tier card's side of the routed layers, GLM's
//!   computation on it and the stage card's handoff and card sum around it;
//! - [`forced`]: one layer alone on given streams, for the gates;
//! - [`gemm`]: the prompt batch's Q8_0 projections on the tensor-core GEMM.
//!
//! Built with `cargo oxide`, as the device crates it links are.

pub mod body;
mod ffn;
pub mod forced;
pub mod gemm;
mod host;
mod kda;
mod mla;
mod program;
mod swap;
mod tensors;
mod tier;
pub mod tp;

pub use body::nextn::{
    GlmArena, Nextn, WALK_ROWS, nextn_chain, nextn_hidden, nextn_logits, nextn_router, nextn_store,
    nextn_target_streams, nextn_walk, retain_prompt_row,
};
pub use body::prefill::{
    CHUNK, FfnTapRows, GEMM_FROM, GlmPromptSink, PrefillMode, PromptBytes, RouteTapRows,
    StoreDigest, StoreRows, T_MAX, batches_of, call_batches, feed, plant_prompt_routes, prefill,
    prefill_group, prefill_media, prefill_mode, prompt_bytes, prompt_ffn_taps, prompt_media_with,
    prompt_route_taps, prompt_with, set_prefill, set_prefill_group, set_prefill_media,
    set_prompt_ffn_taps, set_prompt_route_taps, set_prompt_stats, store_digests, store_rows,
    take_prompt_stats,
};
pub use body::seq::{
    GlmSeq, seq_bytes, seq_lane, seq_resume, seq_resume_prefix, seq_save, seq_save_async,
    seq_save_bytes,
};
pub use body::{
    Body, CHECKPOINT_EVERY, Glm5nextModel, LANES, LOAD_ROWS, NEXTN_ON_TIER, Plant,
    TIER_BEFORE_UPLOAD, checkpoint_budget, checkpoint_price, prompt, set_taps, trunk_blocks,
    trunk_mixer_extras,
};
pub use model::arch::glm5next::place::KdaLanes;
pub use program::{layer_launches, step_launches};
pub use swap::{DEADLINE, LIVE_DELAY};
pub use tier::NODE_FAMILY;
