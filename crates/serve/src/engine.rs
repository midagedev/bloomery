//! The engine contract the server drives, and the sampler hook it calls.
//!
//! The server owns one `Engine` behind a mutex (one slot). A request is:
//! `reset`, `prefill(ids[..n-1])`, then `next(ids[n-1])` yields the first generated
//! token and every later `next(prev)` the one after it. `tokens_evaluated` and
//! `prompt_n` still count the whole prompt, `ids.len()`.

use std::sync::Arc;

/// A failure inside the engine (a device error, a context overflow it detected itself).
#[derive(Debug, thiserror::Error)]
#[error("engine: {0}")]
pub struct EngineError(pub String);

/// Streaming detokenizer: holds bytes of an incomplete UTF-8 sequence until the
/// token that completes it arrives.
pub trait Decoder: Send {
    /// Feeds one token; returns the text it completes, if any.
    fn push(&mut self, id: u32) -> Option<String>;
    /// Returns whatever is still held (lossily decoded) and clears the buffer.
    fn flush(&mut self) -> String;
}

/// What the server needs from a model plus its tokenizer.
///
/// `encode` must parse special-token strings the way llama.cpp does with
/// `parse_special = true`: a rendered chat prompt carries the BOS string, the role
/// markers and the think tags as text, and a plain BPE pass would split them.
pub trait Engine: Send {
    /// Text to token ids, special-token strings mapped to their ids. Adds no BOS.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// Token ids to text, special tokens rendered as their strings.
    fn decode(&self, ids: &[u32]) -> String;
    /// A fresh streaming decoder.
    fn decoder(&self) -> Box<dyn Decoder>;
    /// Evaluates `ids` from the current position without producing a token.
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError>;
    /// Evaluates `last`, writes the next position's logits into `logits_out`
    /// (length `n_vocab`) and returns their argmax.
    fn next(&mut self, last: u32, logits_out: &mut [f32]) -> Result<u32, EngineError>;
    /// Drops the whole cache; the next `prefill` starts at position 0.
    fn reset(&mut self);
    /// Beginning-of-sequence token id.
    fn bos(&self) -> u32;
    /// End-of-generation token id.
    fn eos(&self) -> u32;
    /// Whether a text prompt on `/completion` gets a BOS prepended (GGUF `add_bos_token`).
    fn add_bos(&self) -> bool;
    /// Positions the cache holds.
    fn ctx_max(&self) -> usize;
    /// Logit vector length.
    fn n_vocab(&self) -> usize;
}

/// The sampling knobs a request carries, llama-server names and defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    /// `<= 0` is greedy: the engine's argmax is used and no sampler is built.
    pub temperature: f32,
    /// `<= 0` keeps every candidate.
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    /// The effective seed (a request's `-1` is replaced by a clock-derived one).
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            seed: u64::from(u32::MAX),
        }
    }
}

/// One request's sampler: `(logits, tokens generated so far) -> id`.
pub type Sampler = Box<dyn FnMut(&[f32], &[u32]) -> u32 + Send>;

/// Builds a sampler per request. The real one comes from the sampler crate; the
/// server ships [`crate::sampling::reference_factory`] so it runs without it.
pub type SamplerFactory = Arc<dyn Fn(&SamplingParams) -> Sampler + Send + Sync>;
