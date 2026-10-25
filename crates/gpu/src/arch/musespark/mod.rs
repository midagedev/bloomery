//! The `muse-spark` body: one repeated layer of grouped-query attention —
//! the sliding layers rope in NORM pairs at the sliding base inside their
//! window, the full layers are NoPE, every layer with a per-head q/k norm —
//! over a sigmoid-routed mixture of SiLU-gated experts in a latent, the
//! sublayers joined by per-channel α/β mixes, and a scale-then-softcap head.
//! Whole-card or placed (the routed experts the card cannot hold on the host
//! tier), one token a step; every launch a shared kernel
//! (docs/research/muse-spark-attach.md §2 is the numerical contract, and
//! llama.cpp's `src/models/muse-spark.cpp` the dialect's authority).

pub mod body;

pub use body::{Body, LayerNodes, LayerRoute, MuseSeq, Placed};
