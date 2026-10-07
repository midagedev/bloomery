//! `qwen35moe` — Qwen3.5/3.6 mixture-of-experts (Qwen3.6-35B-A3B), its dense
//! variant `qwen35` (Qwen3.5-9B, Cloudflare's Clef backbone: the same
//! layers with a dense SwiGLU FFN), and its variant `qwen4exp`
//! (Qwen3.8-Flash-Next). The file's keys (`hparams`),
//! every tensor's role (`roles`) and the typed description (`spec`), which
//! the coverage check lists what a program has to run from; for qwen4exp the
//! tensor names its program reads (`names`), the plan from the headers
//! (`place`) and the host tier's view of a routed layer (`host`).
//!
//! The trunk interleaves gated delta-rule (GDN) layers with gated GQA layers;
//! every qwen35moe layer routes to experts and runs a sigmoid-gated shared
//! expert, every qwen35 layer runs one dense SwiGLU block.
//! qwen4exp wraps every layer in gated-residual hyper-connections, selects an
//! attention layer's positions by a mean-pool top-k, and adds a per-layer
//! n-gram embedding (PLE) on one GDN layer. A qwen4exp MTP draft file is read
//! against its target by `mtp`. The line numbers cited are
//! llama.cpp's `src/models/qwen35moe.cpp`, `src/models/qwen35.cpp`,
//! `src/models/qwen4exp.cpp` and
//! `src/llama-hparams.cpp` unless another file is named.

pub mod fixture;
pub mod head_list;
pub mod host;
pub mod hparams;
pub mod mtp;
pub mod names;
pub mod place;
pub mod roles;
pub mod spec;

use crate::{ModelError, Tensor2};
use gguf::Gguf;

/// Rows `ids` of `output.weight`, dequantized: the output embedding of each
/// id, `[embd, ids.len()]` (id `i`'s row at `data[i·embd ..]`) — what Clef's
/// head reads beside the backbone's hidden states, scoring options by their
/// tokens' output vectors. The same row gather as the embedding lookup's
/// ([`embed_with`]); an id past the vocabulary is the lookup's named refusal.
///
/// [`embed_with`]: super::deepseek2::forward::embed_with
pub fn output_rows(gguf: &Gguf, ids: &[u32]) -> Result<Tensor2, ModelError> {
    let w = gguf
        .find("output.weight")
        .ok_or_else(|| ModelError::MissingTensor("output.weight".into()))?;
    super::deepseek2::forward::embed_with(gguf, w, ids)
}
