//! The engine contract the server drives, and the sampler hook it calls.
//!
//! The server owns one `Engine` behind a mutex (one slot) and the engine's
//! [`Tokenizer`] outside it: `/tokenize`, `/detokenize` and prompt encoding never
//! wait for a generation. A request keeps the longest prefix `k` of its ids the
//! cache already holds and the engine can keep ([`Engine::keepable`]): `cut(k)`,
//! or `reset` when `k` is 0; then `prefill(ids[k..n-1])`, and `next(ids[n-1])`
//! yields the first generated token and every later `next(prev)` the one after
//! it. `timings.cache_n` and `usage.prompt_tokens_details.cached_tokens` are
//! `k`, `timings.prompt_n` is the `n − k` ids evaluated, and
//! `tokens_evaluated` and `usage.prompt_tokens` count the whole prompt, `n`, as
//! llama-server's do.
//!
//! Any `EngineError` is fatal to the server: the request that met it gets a 500,
//! `/health` answers 503, and [`crate::Server::run`] returns the error so the
//! process exits instead of serving an engine in an unknown state.
//!
//! Slot persistence (`POST /slots/0?action=save|restore|erase`): the server owns
//! the file, its header and the slot's ids, and hands the engine the stream
//! after them ([`Engine::save_state`], [`Engine::restore_state`]); the engine's
//! bytes run to the end of the file. Erase needs no engine call: it is
//! [`Engine::reset`] and the slot forgetting its ids. The defaults refuse with
//! [`StateError::Unsupported`], which is not fatal: the server answers 501 and
//! keeps serving.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::sync::Arc;

/// A failure inside the engine (a device error, a context overflow it detected
/// itself, a NaN in the logits).
#[derive(Debug, thiserror::Error)]
#[error("engine: {0}")]
pub struct EngineError(pub String);

/// Why a save or a restore of the slot's cache failed. Only
/// [`StateError::Engine`] is fatal to the server; the others are the request's.
#[derive(Debug, thiserror::Error)]
pub enum StateError {
    /// The engine cannot snapshot its cache; it read and wrote nothing.
    #[error("this engine does not support {0}")]
    Unsupported(&'static str),
    /// The file is not a state this server and engine can take: its tag, its
    /// version, a count or an id is out of what they accept.
    #[error("{0}")]
    Format(String),
    /// Reading or writing the file failed (a short file reads as `UnexpectedEof`).
    #[error("{0}")]
    Io(#[from] io::Error),
    /// The engine failed mid-call.
    #[error(transparent)]
    Engine(#[from] EngineError),
}

/// What a save wrote or a restore read of the engine's part of a slot file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SavedState {
    /// The positions the state covers: every position the cache holds.
    pub n_tokens: usize,
    /// The bytes the engine wrote or read.
    pub n_bytes: u64,
}

/// Streaming detokenizer: holds bytes of an incomplete UTF-8 sequence until the
/// token that completes it arrives.
pub trait Decoder: Send {
    /// Feeds one token; returns the text it completes, if any.
    fn push(&mut self, id: u32) -> Option<String>;
    /// Returns whatever is still held (lossily decoded) and clears the buffer.
    fn flush(&mut self) -> String;
}

/// The vocabulary side of a model. Shared by every connection thread, so it
/// takes `&self` only and is read without the engine lock.
///
/// `encode` must parse special-token strings the way llama.cpp does with
/// `parse_special = true`: a rendered chat prompt carries the BOS string, the role
/// markers and the think tags as text, and a plain BPE pass would split them.
pub trait Tokenizer: Send + Sync {
    /// Text to token ids, special-token strings mapped to their ids. Adds no BOS.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// Token ids to text, special tokens rendered as their strings.
    fn decode(&self, ids: &[u32]) -> String;
    /// A fresh streaming decoder.
    fn decoder(&self) -> Box<dyn Decoder>;
    /// Beginning-of-sequence token id.
    fn bos(&self) -> u32;
    /// End-of-generation token id.
    fn eos(&self) -> u32;
    /// Whether a text prompt on `/completion` gets a BOS prepended (GGUF `add_bos_token`).
    fn add_bos(&self) -> bool;
    /// Vocabulary size, which is also the logit vector length.
    fn n_vocab(&self) -> usize;
}

/// What the server needs from a model.
pub trait Engine: Send {
    /// The vocabulary, taken once when the server binds.
    fn tokenizer(&self) -> Arc<dyn Tokenizer>;
    /// Evaluates `ids` from the current position without producing a token.
    /// An empty slice is a no-op.
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError>;
    /// Evaluates `last` and returns the argmax of the next position's logits.
    /// With `Some(out)` (length `n_vocab`) it also writes those logits; a
    /// greedy request passes `None` and the engine may skip reading them.
    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError>;
    /// Drops the whole cache; the next `prefill` starts at position 0.
    fn reset(&mut self) -> Result<(), EngineError>;
    /// The longest prefix, at most `n` positions, of what the cache holds now
    /// that [`Engine::cut`] can keep. The default is 0: the caller resets
    /// instead, so an engine without `cut` never has it called.
    fn keepable(&self, n: usize) -> usize {
        let _ = n;
        0
    }
    /// Keeps positions `[0, n)` and drops the rest; the next `prefill`
    /// continues at `n`. Called only with an `n` [`Engine::keepable`] granted.
    fn cut(&mut self, n: usize) -> Result<(), EngineError> {
        let _ = n;
        Err(EngineError("cut is not supported".to_owned()))
    }
    /// Positions the cache holds.
    fn ctx_max(&self) -> usize;
    /// What a crash report names besides the error: the device, the position.
    fn describe(&self) -> String;
    /// What `/props` reports about this engine under `engine`, read once when
    /// the server binds. The default reports nothing: every key is left out.
    fn props_engine(&self) -> EngineProps {
        EngineProps::default()
    }
    /// Writes the whole cache to `out`, from position 0 to what it holds, in a
    /// form [`Engine::restore_state`] reads back; the cache is unchanged. The
    /// returned `n_bytes` is what went to `out`. The default writes nothing
    /// and refuses.
    fn save_state(&self, out: &mut dyn Write) -> Result<SavedState, StateError> {
        let _ = out;
        Err(StateError::Unsupported("slot save/restore"))
    }
    /// Replaces the cache with the state `input` carries, which runs to its end;
    /// the next `prefill` continues after the returned `n_tokens`. An engine
    /// that refuses a state it has started to read leaves its cache in no
    /// defined state: the server resets it. The default reads nothing and
    /// refuses, the cache untouched.
    fn restore_state(&mut self, input: &mut dyn Read) -> Result<SavedState, StateError> {
        let _ = input;
        Err(StateError::Unsupported("slot save/restore"))
    }
}

/// An engine's part of `/props`' `engine` object (toktape's shape), beside the
/// `name`, `version`, `args` and `server_pid` the server fills itself. A `None`
/// leaves its key out, which a reader shows as unknown: nothing is guessed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EngineProps {
    /// A word the server appends to its `version` for an engine that runs no
    /// model (the mock's `mock`), so a recording of it never reads as a model's.
    pub version_note: Option<String>,
    /// The model file.
    pub model: Option<ModelProps>,
    /// Where the weights live.
    pub placement: Option<PlacementProps>,
    /// The speculative drafter, when one runs.
    pub draft: Option<DraftProps>,
}

/// The model file (`engine.model`), from its header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelProps {
    /// The file format, `gguf`.
    pub format: Option<String>,
    /// The file's architecture name.
    pub arch: Option<String>,
    /// The type name that holds the most bytes of the weights a step reads.
    pub quant: Option<String>,
    /// Bytes on disk, every shard.
    pub bytes: Option<u64>,
    /// Shards.
    pub files: Option<u64>,
    /// Layers of the decode graph.
    pub n_layers: Option<u64>,
    /// Experts in each routed stack.
    pub n_experts: Option<u64>,
    /// Experts one token uses.
    pub n_experts_used: Option<u64>,
    /// The context the model was trained for.
    pub ctx_train: Option<u64>,
}

/// Where the weights live (`engine.placement`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlacementProps {
    /// The cards, then the host.
    pub devices: Vec<DeviceProps>,
    /// The KV cache's bytes on the cards.
    pub vram_kv_bytes: Option<u64>,
}

/// One device of a placement.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceProps {
    /// `GPU<n>` with `n` the nvidia-smi index, or `CPU`.
    pub device: String,
    /// Resident bytes by tensor class; the device's `bytes` is their sum.
    pub class_bytes: BTreeMap<String, u64>,
    /// The layers the device runs, as `first-last`.
    pub layers: Option<String>,
}

/// The speculative drafter (`engine.draft`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DraftProps {
    /// `lookup`, or the draft model's file name.
    pub model: String,
    /// The most tokens it drafts per step.
    pub n_max: Option<u64>,
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
