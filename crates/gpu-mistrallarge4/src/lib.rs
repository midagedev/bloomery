//! bloomery-gpu-mistrallarge4 — the Mistral Large 4 program: a whole-card
//! body over the runtime's kernels, host code only. It has no device code of
//! its own: the norms, the LayerNorm, the rope, the SSMax pair, the fused key
//! append, the 576-wide latent flash and its selected row gather, the q8_0
//! gemvs and per-head absorption, the softmax router, the i-quant `_sel`
//! experts and the head are `bloomery-gpu`'s; the lightning indexer, its
//! input-projected key append and the selected 512-wide walk
//! `bloomery-gpu-deepseek41`'s. Built with `cargo oxide`, as the device
//! crates it links are.
//!
//! - [`body`]: the load, the stores, the step's buffers, [`body::Body`]
//!   behind `GpuModel` and the gate instruments (taps, spans, the routing,
//!   the lists, the scores);
//! - `tensors`: each layer's tensors, built from the file's own arrays.
//!
//! Not this crate's (later rounds): a placed load and the host expert tier,
//! batched prompt processing, DSpark, Pixtral.

pub mod body;
mod tensors;
