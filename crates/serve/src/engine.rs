//! The engine contract the server drives, and the sampler hook it calls.
//!
//! The server owns one `Engine` on an engine thread of its own and the
//! engine's [`Tokenizer`] outside it: `/tokenize`, `/detokenize` and prompt
//! encoding never wait for a generation. The engine serves
//! [`Engine::slots`] sequences at once, each with its own cache of
//! [`Engine::ctx_max`] positions; [`Engine::select_slot`] picks the one every
//! per-slot call below acts on, and [`Engine::step_slots`] steps several in
//! one call. An engine that declares one slot (the default) is never asked to
//! select and never steps more than one row.
//!
//! On the selected slot, a request keeps the longest prefix `k` of its ids the
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
//! Slot persistence (`POST /slots/{id}?action=save|restore|erase`): the server owns
//! the file, its header and the slot's ids, and hands the engine the stream
//! after them ([`Engine::save_state`], [`Engine::restore_state`]); the engine's
//! bytes run to the end of the file. Erase needs no engine call: it is
//! [`Engine::reset`] and the slot forgetting its ids. The defaults refuse with
//! [`StateError::Unsupported`], which is not fatal: the server answers 501 and
//! keeps serving.
//!
//! The host prompt cache (llama-server's `--cache-ram`), one for every slot:
//! before a request drops most of what its slot holds, the server takes the
//! slot's state as a
//! value ([`Engine::snapshot`]) into a host-RAM LRU of [`Engine::cache_ram`]
//! bytes, keyed by the ids it covers; a later request that one of those states
//! serves better than the slot does gets it back ([`Engine::resume`]). What
//! the cache did, and every prefix the engine keeps less of than a request
//! shares, reaches the engine's binary as a [`CacheNote`] ([`Engine::note`]).

use std::any::Any;
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::sync::Arc;

use crate::swap::Park;

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
    /// End-of-sequence token id: the text `/props` names as `eos_token`.
    fn eos(&self) -> u32;
    /// Every id that ends a generation (a vocabulary's end-of-generation
    /// set: a file can name an end of turn and an end of message besides its
    /// end of sequence). Never empty; the default is [`Tokenizer::eos`] alone.
    fn stops(&self) -> Vec<u32> {
        vec![self.eos()]
    }
    /// Whether a text prompt on `/completion` gets a BOS prepended (GGUF `add_bos_token`).
    fn add_bos(&self) -> bool;
    /// Vocabulary size, which is also the logit vector length.
    fn n_vocab(&self) -> usize;
    /// The ids that open a user (or tool) message in this vocabulary's chat
    /// format: where a prompt's messages start, which a prompt call is cut at
    /// ([`Engine::prefill_splits`]). Empty (the default) marks none.
    fn user_start(&self) -> Vec<u32> {
        Vec::new()
    }
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
    /// One greedy pass from `last`: `last` and whatever the engine's draft
    /// proposes after it, evaluated together, and the tokens the pass keeps
    /// appended to `out` in position order — each the target's own argmax,
    /// the first the one `next(last, None)` returns. The engine then stands
    /// past `last` and every appended token but the last, which the next call
    /// feeds. The default is one `next`: nothing drafted.
    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        out.push(self.next(last, None)?);
        Ok(Drafted::default())
    }
    /// The most positions one [`Engine::advance`] evaluates: 1 for an engine
    /// without a draft. Past 1 the engine drafts its greedy requests; a request
    /// that needs the logits row (sampling, a banned id) takes plain
    /// [`Engine::next`] steps instead, since a draft's pass keeps the target's
    /// argmax and reads no row. Under a draft, `next` with `Some(out)` writes
    /// the target's row of the step it ran.
    fn advance_rows(&self) -> usize {
        1
    }
    /// The sequences this engine serves at once, each with a cache of its own
    /// of [`Engine::ctx_max`] positions: the most `--parallel` it takes. The
    /// default is 1.
    fn slots(&self) -> usize {
        1
    }
    /// Makes `slot` (below [`Engine::slots`]) the one every per-slot call
    /// acts on until the next select: `prefill`, `next`, `advance`, `reset`,
    /// `will_reply`, `keepable`, `cut`, `keep_limit`, `save_state`,
    /// `restore_state`, `snapshot`, `resume` and `prefill_splits`. Slot 0 is selected from the
    /// start; the server selects before a per-slot call only when the slot
    /// changes. The default serves slot 0 alone and refuses any other by name.
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        if slot == 0 {
            Ok(())
        } else {
            Err(EngineError(format!(
                "slot {slot}: this engine serves slot 0 alone"
            )))
        }
    }
    /// One step of several slots in one call: each row evaluates its `last`
    /// on its own `slot` (distinct, each below [`Engine::slots`]) at that
    /// slot's own position, and its `next` is set to the argmax of the
    /// position after, as [`Engine::next`] gives it; a row with `logits` gets
    /// that position's logits written there too. Afterwards the selected slot
    /// is unspecified: the server selects before its next per-slot call. The
    /// server calls this only with two rows or more (one row is a `next`).
    /// The default is a select and a `next` a row, in order.
    fn step_slots(&mut self, rows: &mut [SlotRow<'_>]) -> Result<(), EngineError> {
        for row in rows {
            self.select_slot(row.slot)?;
            row.next = self.next(row.last, row.logits.as_deref_mut())?;
        }
        Ok(())
    }
    /// How the slots share the engine. `None` (the default): each slot holds
    /// a sequence of its own and several step in one call. `Some(park)`: the
    /// slots take the engine's one sequence in turns ([`crate::SwapEngine`]):
    /// the server steps one slot a call, so a draft keeps drafting, and keeps
    /// the state of a slot it leaves as `park` says.
    fn turns(&self) -> Option<Park> {
        None
    }
    /// Drops the whole cache; the next `prefill` starts at position 0.
    fn reset(&mut self) -> Result<(), EngineError>;
    /// The next request's reply, told before its cache is reused: at most
    /// `tokens` of its generated tokens come from [`Engine::advance`]
    /// (`Some(0)` for a request that takes no pass, `None` when the request
    /// bounds nothing). An engine whose [`Engine::keepable`] weighs a kept
    /// prefix against the reply reads it there; the default ignores it.
    fn will_reply(&mut self, tokens: Option<usize>) {
        let _ = tokens;
    }
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
    /// Positions the engine serves: its cache holds at least these, and it
    /// computes the model at each. The server refuses a prompt of `ctx_max`
    /// tokens or more and stops generation there, so an engine never sees a
    /// position past it.
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
    /// Why [`Engine::keepable`] of `n` keeps less than `n`: the rule that
    /// stopped it, which the server's reuse note prints. Asked only after
    /// `keepable(n)` granted less, before the cache changes. `None` names no
    /// rule; the default names none.
    fn keep_limit(&self, n: usize) -> Option<String> {
        let _ = n;
        None
    }
    /// The host RAM, in bytes, the server's prompt cache may hold this
    /// engine's saved states in. 0 (the default) turns the cache off.
    fn cache_ram(&self) -> u64 {
        0
    }
    /// Returns the engine's adaptive expert residency to the placement it
    /// loaded with, the sequence left as it stands: the one explicit call a
    /// runner makes before a timed request (`POST /residency/reset`); no
    /// request and no other call does it. `Ok(None)` (the default): the
    /// engine runs no residency, a 501.
    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, EngineError> {
        Ok(None)
    }
    /// The whole cache as a value [`Engine::resume`] takes back, the cache
    /// unchanged. The default is [`Engine::save_state`] into host memory.
    fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
        let mut bytes = Vec::new();
        let saved = self.save_state(&mut bytes)?;
        if saved.n_bytes != bytes.len() as u64 {
            return Err(StateError::Format(format!(
                "the engine saved {} bytes and reported {}",
                bytes.len(),
                saved.n_bytes
            )));
        }
        Ok(Arc::new(SavedBytes {
            n_tokens: saved.n_tokens,
            bytes,
        }))
    }
    /// Replaces the cache with `state`, which this engine's
    /// [`Engine::snapshot`] took; the next `prefill` continues after
    /// `state.n_tokens()`. A refusal leaves the cache in no defined state (the
    /// server resets it). The default reads the default snapshot's bytes with
    /// [`Engine::restore_state`].
    fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
        let Some(saved) = state.as_any().downcast_ref::<SavedBytes>() else {
            return Err(StateError::Format(
                "a saved state this engine did not take".to_owned(),
            ));
        };
        let mut input = saved.bytes.as_slice();
        let read = self.restore_state(&mut input)?;
        if !input.is_empty() || read.n_tokens != saved.n_tokens {
            return Err(StateError::Format(format!(
                "the engine restored {} positions and left {} bytes of a state of {} positions",
                read.n_tokens,
                input.len(),
                saved.n_tokens
            )));
        }
        Ok(())
    }
    /// Where to cut a prompt call of positions `first .. end` into calls so
    /// that the `marks` (ascending, each inside the call) a later request may
    /// be cut back to stay keepable: a subset of `marks`, ascending. The
    /// default cuts nowhere: an engine whose cache keeps every position needs
    /// no cut.
    fn prefill_splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        let _ = (first, end, marks);
        Vec::new()
    }
    /// What the prompt cache did, for this engine's binary to print. The
    /// default writes one plain line to stderr.
    fn note(&self, note: &CacheNote) {
        eprintln!("bloomery-serve: {note}");
    }
}

/// One row of [`Engine::step_slots`]: the slot, the id it evaluates, and the
/// logits buffer (length `n_vocab`) of a row that needs them. `next` is the
/// engine's answer; the server sets it past the vocabulary before the call,
/// and a row still past it afterwards is the engine's error.
#[derive(Debug)]
pub struct SlotRow<'a> {
    pub slot: usize,
    pub last: u32,
    pub logits: Option<&'a mut [f32]>,
    pub next: u32,
}

/// What one [`Engine::advance`] drafted: the ids its draft proposed (0 for a
/// pass that ran none) and how many of them the target kept, llama-server's
/// `draft_n` and `draft_n_accepted` of one pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Drafted {
    pub proposed: usize,
    pub accepted: usize,
}

/// A saved engine state held by the prompt cache ([`Engine::snapshot`]).
pub trait Saved: Send + Sync {
    /// The positions it holds.
    fn n_tokens(&self) -> usize;
    /// The host bytes it holds, which the cache's budget counts.
    fn n_bytes(&self) -> u64;
    /// What [`Engine::keepable`] would grant of `n` right after a resume of
    /// this state: the cache ranks its states by it.
    fn keepable(&self, n: usize) -> usize;
    /// The value, for the engine that took it to read back.
    fn as_any(&self) -> &dyn Any;
}

/// The default snapshot: [`Engine::save_state`]'s bytes. Its `keepable` is
/// every position it holds.
struct SavedBytes {
    n_tokens: usize,
    bytes: Vec<u8>,
}

impl Saved for SavedBytes {
    fn n_tokens(&self) -> usize {
        self.n_tokens
    }

    fn n_bytes(&self) -> u64 {
        self.bytes.len() as u64
    }

    fn keepable(&self, n: usize) -> usize {
        n.min(self.n_tokens)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// What the prompt cache did ([`Engine::note`]). `ms` is the server's wall
/// clock around the engine call, not an admissible measurement.
#[derive(Clone, Debug, PartialEq)]
pub enum CacheNote {
    /// A request shared `common` ids with the slot and asked to keep `ask` of
    /// them (all but its last id), and the engine kept `kept < ask`, for
    /// `reason` ([`Engine::keep_limit`]; `None` when it named none).
    Reuse {
        common: usize,
        ask: usize,
        kept: usize,
        held: usize,
        reason: Option<String>,
    },
    /// The slot's state of `positions` went into the cache.
    Save {
        positions: usize,
        bytes: u64,
        ms: f64,
        entries: usize,
        cache_bytes: u64,
    },
    /// A cached state of `positions` sharing `common` ids with the request
    /// replaced the slot's, which kept `slot_kept` of them; the engine then
    /// keeps `kept`.
    Load {
        positions: usize,
        common: usize,
        kept: usize,
        slot_kept: usize,
        bytes: u64,
        ms: f64,
    },
    /// A cached state left the cache: the budget needed its bytes, or a newer
    /// state keeps everything it did.
    Evict {
        positions: usize,
        bytes: u64,
        why: &'static str,
    },
    /// The slot's state of `positions` was not cached, for `why`.
    Skip { positions: usize, why: String },
    /// A prompt call from `first` to `end` ran as calls cut at `at`, so a
    /// later request keeps those positions.
    Split {
        first: usize,
        end: usize,
        at: Vec<usize>,
    },
}

impl std::fmt::Display for CacheNote {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CacheNote::Reuse {
                common,
                ask,
                kept,
                held,
                reason,
            } => write!(
                f,
                "reuse common={common} ask={ask} kept={kept} held={held} reason={}",
                reason.as_deref().unwrap_or("unstated")
            ),
            CacheNote::Save {
                positions,
                bytes,
                ms,
                entries,
                cache_bytes,
            } => write!(
                f,
                "cache save positions={positions} bytes={bytes} ms={ms:.3} entries={entries} \
                 cache_bytes={cache_bytes}"
            ),
            CacheNote::Load {
                positions,
                common,
                kept,
                slot_kept,
                bytes,
                ms,
            } => write!(
                f,
                "cache load positions={positions} common={common} kept={kept} \
                 slot_kept={slot_kept} bytes={bytes} ms={ms:.3}"
            ),
            CacheNote::Evict {
                positions,
                bytes,
                why,
            } => write!(
                f,
                "cache evict positions={positions} bytes={bytes} why={why}"
            ),
            CacheNote::Skip { positions, why } => {
                write!(f, "cache skip positions={positions} why={why}")
            }
            CacheNote::Split { first, end, at } => {
                write!(f, "prefill split first={first} end={end} at={at:?}")
            }
        }
    }
}

/// What a residency reset did ([`Engine::residency_reset`]): flips in flight
/// cancelled, experts copied back onto the card, entries of the live map
/// that still differ from the load's after it (0 when it worked), and host
/// bytes released.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyReset {
    pub cancelled: u64,
    pub copies: u64,
    pub diff: u64,
    pub dropped_bytes: u64,
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
    /// The deepest context, in positions, at which the engine's numbers are
    /// held to its reference engine's on this model. It bounds nothing: the
    /// server serves `n_ctx`.
    pub ctx_verified: Option<u64>,
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
    /// What drafts: `lookup`, or the draft model's kind (`dspark`).
    pub kind: Option<String>,
    /// The draft model's file, as `model_path` names the target's.
    pub path: Option<String>,
    /// The device the draft model runs on, as a placement device names it
    /// (`GPU<n>`); its resident bytes are that device's `draft` class.
    pub device: Option<String>,
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
