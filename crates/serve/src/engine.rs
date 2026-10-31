//! The engine contract the server drives, and the sampler hook it calls.
//!
//! The server owns one `Engine` on an engine thread of its own and the
//! engine's [`Tokenizer`] outside it: `/tokenize`, `/detokenize` and prompt
//! encoding never wait for a generation. The engine serves
//! [`Engine::slots`] sequences at once, each with its own cache of
//! [`Engine::ctx_max`] positions; [`Engine::select_slot`] picks the one every
//! per-slot call below acts on, [`Engine::step_slots`] steps several in one
//! call and [`Engine::advance_slots`] runs one drafted pass on each of
//! several, which an engine that keeps its draft's state per slot declares
//! ([`Engine::slot_drafts`]). An engine that declares one slot (the default)
//! is never asked to select and never steps more than one row.
//!
//! On the selected slot, a request keeps the longest prefix `k` of its ids the
//! cache already holds and the engine can keep ([`Engine::keepable`]): `cut(k)`,
//! or `reset` when `k` is 0; then `prefill(ids[k..n-1])`, and `next(ids[n-1])`
//! yields the first generated token and every later `next(prev)` the one after
//! it. The server runs each prompt call as [`Engine::prefill_marked`] (whose
//! default is that `prefill`), handing it the call's message starts, where a
//! later request may be cut back to, and cutting the call where
//! [`Engine::prefill_splits`] asks; the prompt's last call and the step of
//! `ids[n-1]` reach the engine as one [`Engine::prefill_next`], whose default
//! is those two calls. An engine that names a prompt quantum `q`
//! ([`Engine::prompt_quantum`]) may see that `prefill` arrive instead as calls
//! ending at `k + j·q`, the grid
//! starting again at each cut [`Engine::prefill_splits`] asks for, one a round
//! between the other busy slots' decode rounds; the quantum's contract makes
//! the request's bits those of the one call. `timings.cache_n` and `usage.prompt_tokens_details.cached_tokens` are
//! `k`, `timings.prompt_n` is the `n − k` ids evaluated, and
//! `tokens_evaluated` and `usage.prompt_tokens` count the whole prompt, `n`, as
//! llama-server's do. A chat's image is a span of positions that each carry the
//! model's image token ([`Engine::media_model`]): `k` never ends inside one,
//! and a prompt call that holds a span is [`Engine::prefill_media`] with the
//! image's feed.
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
//! before a request drops what its slot holds (on a server of one slot, most
//! of it) — unless it carries the slot's last prompt whole, and so drops only
//! that prompt's reply — the server takes the slot's state as a value
//! ([`Engine::snapshot`]) into a host-RAM LRU of [`Engine::cache_ram`] bytes,
//! keyed by the ids and the images it covers; a later request that one of
//! those states serves better than the slot does gets back the prefix of it
//! it keeps ([`Engine::resume_prefix`]). Slots that take the engine in turns
//! hand the cache the idle states they stop parking. What the cache did, and
//! every prefix the engine keeps less of than a request shares, reaches the
//! engine's binary as a [`CacheNote`] ([`Engine::note`]).
//!
//! An engine whose slots can hold one prefix in common on the device
//! ([`Engine::share_keepable`]) gives a request the longest prefix another
//! slot holds when it keeps more than the request's own slot and every
//! cached state ([`Engine::share_prefix`]): the slot takes those positions
//! from the other slot's cache instead of evaluating them, and the other
//! slot keeps serving its own request.
//!
//! An engine that copies a state on a stream of its own
//! ([`Engine::snapshots_async`]) is asked for it when a request ends, off the
//! request path ([`Engine::snapshot_begin`]): the state goes into the cache at
//! once as a [`Pending`], which the cache never evicts before its copy has
//! landed ([`Engine::snapshot_poll`], [`Engine::snapshot_wait`]), and the
//! slot's next request waits for the copy before it cuts, resets or replaces
//! the rows the copy reads. The copy running while a cut, a reset or a resume
//! reaches the same rows is [`SaveInFlight`], never an overwrite.

use std::any::Any;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read, Write};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use crate::media::{MediaFeed, SharedMediaModel};
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
    /// The engine's staging pool has no extent for a state of `need` bytes
    /// ([`Engine::snapshot_begin`]): it pins `held` bytes and has `lent` of
    /// them out. `held - lent >= need` tells a pool whose free bytes are in
    /// pieces ([`StateError::fallback`]) from one that is full. Nothing was
    /// begun and nothing changed.
    #[error(
        "the staging pool has no room for {need} bytes: {lent} of its {held} bytes are lent ({})",
        pool_state(*.need, *.held, *.lent)
    )]
    NoRoom { need: u64, held: u64, lent: u64 },
}

/// `full` or `fragmented`: what a pool of `held` bytes, `lent` of them out,
/// lacks for an extent of `need` ([`StateError::NoRoom`]).
fn pool_state(need: u64, held: u64, lent: u64) -> &'static str {
    if held.saturating_sub(lent) >= need {
        "fragmented"
    } else {
        "full"
    }
}

impl StateError {
    /// What a save that fell back to the request path names
    /// ([`CacheNote::Save`]): `fragmented` or `full` for a pool that had no
    /// room ([`StateError::NoRoom`]), `None` for any other error.
    #[must_use]
    pub fn fallback(&self) -> Option<&'static str> {
        match *self {
            StateError::NoRoom { need, held, lent } => Some(pool_state(need, held, lent)),
            _ => None,
        }
    }
}

/// A slot's rows were about to be cut, reset or replaced while the copy of
/// its state was still reading them ([`Engine::snapshot_begin`]): a defect of
/// the caller, which waits for the copy first ([`Engine::snapshot_wait`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "slot {slot}: its state is still being copied; its rows cannot be cut, reset or replaced \
     before the copy lands"
)]
pub struct SaveInFlight {
    pub slot: usize,
}

impl From<SaveInFlight> for EngineError {
    fn from(e: SaveInFlight) -> EngineError {
        EngineError(e.to_string())
    }
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
/// markers and the think tags as text, and a plain BPE pass would split them. The text a
/// client sent is in that string too, and its special-token spellings must stay text:
/// [`Tokenizer::guard`] marks them before the render and [`Tokenizer::encode_marked`] reads
/// the marked render.
pub trait Tokenizer: Send + Sync {
    /// Text to token ids, special-token strings mapped to their ids. Adds no BOS.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// `text` with the non-empty `mark` spliced in beside every spelling of a special token it
    /// carries (CONTROL and USER_DEFINED alike, and not a one-character token) and beside the
    /// beginning or the end of one that it ends or starts with; the spelling stays whole in the
    /// text, and no mark is at the start or the end of the text. The text with the marks cut out is
    /// `text`, and it comes back borrowed when there is nothing to mark. A chat template renders
    /// the marked text as it would the text, and [`Tokenizer::encode_marked`] reads the render.
    fn guard<'a>(&self, text: &'a str, mark: &str) -> Cow<'a, str>;
    /// [`Tokenizer::encode`] of a text that carries `mark`s: each is cut out of the text and no
    /// special token matches that holds the place of one or the byte on either side of it, so a
    /// text [`Tokenizer::guard`] marked is its characters' ids where it carried a spelling. The
    /// text between marks is read as one text, and a text with no mark gives the ids `encode` does.
    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32>;
    /// Token ids to text, special tokens rendered as their strings.
    fn decode(&self, ids: &[u32]) -> String;
    /// A fresh streaming decoder.
    fn decoder(&self) -> Box<dyn Decoder>;
    /// Beginning-of-sequence token id; `None` where the vocabulary names none, which a
    /// vocabulary that adds no BOS ([`Tokenizer::add_bos`]) may do.
    fn bos(&self) -> Option<u32>;
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
    /// The BOS a text prompt starts with: the vocabulary's where it adds one, none where it does
    /// not. A vocabulary that adds a BOS and names none is refused by name.
    fn bos_to_prepend(&self) -> Result<Option<u32>, String> {
        if !self.add_bos() {
            return Ok(None);
        }
        self.bos().map(Some).ok_or_else(|| {
            "the vocabulary adds a BOS (add_bos_token) and names no BOS token".to_owned()
        })
    }
    /// Whether text tokenized with the vocabulary's special tokens gets an EOS appended (GGUF
    /// `add_eos_token`), beside the BOS [`Tokenizer::add_bos`] puts first.
    fn add_eos(&self) -> bool {
        false
    }
    /// Vocabulary size, which is also the logit vector length.
    fn n_vocab(&self) -> usize;
    /// Whether `id` is one of the vocabulary's special tokens (CONTROL or
    /// USER_DEFINED): a marker the output parsers read only where this id
    /// wrote it, never where ordinary tokens spell its text. The default marks none,
    /// so every marker is read by its text.
    fn is_special(&self, _id: u32) -> bool {
        false
    }
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
    /// The model's image input, `None` for an engine that takes none: the
    /// server renders an image part as the model's placeholder only when it
    /// has one, and refuses it by name otherwise. Taken once when the server
    /// binds.
    fn media_model(&self) -> Option<SharedMediaModel> {
        None
    }
    /// Evaluates `ids` from the current position without producing a token.
    /// An empty slice is a no-op.
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError>;
    /// [`Engine::prefill`] where the slice carries images: each feed's `at` is
    /// its span's first position in `ids`, every span lies whole inside the
    /// slice, and every position of a span carries the model's image token.
    /// The default is [`Engine::prefill`] when the slice carries no image, and
    /// a named refusal when it does.
    fn prefill_media(&mut self, ids: &[u32], media: &[MediaFeed]) -> Result<(), EngineError> {
        if media.is_empty() {
            return self.prefill(ids);
        }
        Err(EngineError("this engine takes no image input".to_owned()))
    }
    /// Evaluates `last` and returns the argmax of the next position's logits.
    /// With `Some(out)` (length `n_vocab`) it also writes those logits; a
    /// greedy request passes `None` and the engine may skip reading them.
    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError>;
    /// Evaluates `ids` (at least one) from the current position and returns
    /// the argmax of the position after the last, writing its logits to
    /// `logits_out` as [`Engine::next`] does: a run of an answer's ids and
    /// the step that closes it as one call. The engine then stands past
    /// every id. Answer positions are decode positions: the call takes no
    /// prompt checkpoints, as the steps it replaces take none. The default is
    /// [`Engine::prefill`] of all but the last id and [`Engine::next`] of the
    /// last; an engine whose prefill already runs the head on its last row,
    /// or whose prefill takes checkpoints, overrides it.
    fn next_after(
        &mut self,
        ids: &[u32],
        logits_out: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let Some((&last, run)) = ids.split_last() else {
            return Err(EngineError("next_after of no ids".to_owned()));
        };
        if !run.is_empty() {
            self.prefill(run)?;
        }
        self.next(last, logits_out)
    }
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
    /// that needs the logits row (sampling, a banned id) takes
    /// [`Engine::advance_sampled`] passes on an engine that
    /// [`Engine::drafts_sampled`], and plain [`Engine::next`] steps on any
    /// other, since a greedy pass keeps the target's argmax and reads no row.
    /// Under a draft, `next` with `Some(out)` writes the target's row of the
    /// step it ran.
    fn advance_rows(&self) -> usize {
        1
    }
    /// Whether a cut into the ids the last pass kept is always granted as
    /// asked: right after [`Engine::advance`] or [`Engine::advance_sampled`],
    /// [`Engine::keepable`] grants every `n` from the pass's `last` on to
    /// where the engine stands, so [`Engine::cut`] takes it back with nothing
    /// fed again. A request that watches for a tool call takes passes only on
    /// such an engine: the call opener a pass ran past is cut back to. The
    /// default holds for an engine whose pass keeps one id, which leaves no
    /// position inside a pass; a drafting engine says true only where its
    /// caches' rule grants every such cut.
    fn cuts_into_passes(&self) -> bool {
        self.advance_rows() <= 1
    }
    /// One drafted pass for a request that draws (it samples, or bans an id:
    /// its sampler sets the banned logits to -inf first), by llama.cpp's
    /// `common_sampler_sample_and_accept_n`. `last` and the ids the draft
    /// proposes after it are evaluated together. Then, row by row in position
    /// order, `sampler` takes an id from that row's logits, given `history`
    /// (the ids its draws see: the prompt's tail then the generated ids so
    /// far) followed by the ids this pass already appended. Each taken id is
    /// appended to `out`. The pass stops after the
    /// first taken id that differs from the draft's id at its row, or after
    /// the row past the last drafted id. The engine then stands as after an
    /// [`Engine::advance`] that kept those ids, its draft anchored on the
    /// last taken id. `history` and `sampler` come back with the outcome. The
    /// sampler's draws are exactly one per appended id, as one
    /// [`Engine::next`] and one sample per token give them, so a drafted and
    /// a plain run of one seed take the same ids. Called only when
    /// [`Engine::drafts_sampled`]; the default refuses by name.
    fn advance_sampled(
        &mut self,
        last: u32,
        history: Vec<u32>,
        sampler: Sampler,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Vec<u32>, Sampler) {
        let _ = (last, &out);
        (
            Err(EngineError(
                "this engine does not draft a sampled request".to_string(),
            )),
            history,
            sampler,
        )
    }
    /// Whether [`Engine::advance_sampled`] runs: a drafting engine that reads
    /// its verify rows' logits. The default is false.
    fn drafts_sampled(&self) -> bool {
        false
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
    /// One drafted pass on each row's own slot (distinct, each below
    /// [`Engine::slots`]), in row order: each row's `last` and whatever the
    /// engine's draft proposes after it evaluated together, the tokens the
    /// pass keeps appended to its `out` in position order and what it drafted
    /// set in its `drafted`, as [`Engine::advance`] does for one, or for a
    /// row with a [`SlotDraw`] as [`Engine::advance_sampled`] does: its ids
    /// the ones its pass alone draws, so a sampled request of one seed takes
    /// the same ids beside other slots as alone. The server sends a row with
    /// a draw only to an engine that [`Engine::drafts_sampled`]. Afterwards
    /// the selected slot is unspecified: the server selects before its next
    /// per-slot call. The server calls this only with two rows or more (one
    /// row is an `advance`, which an engine may run another way than a round:
    /// a width chooser that cuts the window). The default is a select and an
    /// `advance` a row.
    fn advance_slots(&mut self, rows: &mut [SlotPass<'_>]) -> Result<(), EngineError> {
        for row in rows {
            self.select_slot(row.slot)?;
            row.drafted = advance_row(self, row)?;
        }
        Ok(())
    }
    /// Whether this engine keeps its draft's state (proposal history, draft
    /// cache, acceptance counters) per slot, so a slot's drafted passes are
    /// the passes it would run alone whatever the other slots run; the server
    /// serves several slots of such an engine past one. The default is false:
    /// a drafting engine that shares its draft's state across slots serves
    /// one slot, or several that take it in turns ([`Engine::turns`]).
    fn slot_drafts(&self) -> bool {
        false
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
    /// The host bytes a snapshot would hold, without copying its stores.
    /// `None` when the engine has no size query.
    fn snapshot_size(&self) -> Result<Option<u64>, StateError> {
        Ok(None)
    }
    /// Returns the engine's adaptive expert residency to the placement it
    /// loaded with, the sequence left as it stands: the one explicit call a
    /// runner makes before a timed request (`POST /residency/reset`); no
    /// request and no other call does it. `Ok(None)` (the default): the
    /// engine runs no residency, a 501.
    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, EngineError> {
        Ok(None)
    }
    /// The server's last call, made once its engine thread leaves its loop,
    /// which leaves the engine to the process exit: prints the records the
    /// engine still holds of its last requests, then ends the load on every
    /// other process that serves part of it (a tier node holding its
    /// experts) as the engine's Drop would; frees nothing in this process.
    /// The default holds no record and has no such process.
    fn close(&mut self) -> Result<(), EngineError> {
        Ok(())
    }
    /// The whole cache as a value [`Engine::resume`] takes back, the cache
    /// unchanged. The default is [`Engine::save_state`] into host memory.
    fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
        snapshot_bytes(self)
    }
    /// Replaces the cache with `state`, which this engine's
    /// [`Engine::snapshot`] took; the next `prefill` continues after
    /// `state.n_tokens()`. A refusal leaves the cache in no defined state (the
    /// server resets it). The default reads the default snapshot's bytes with
    /// [`Engine::restore_state`].
    fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
        resume_bytes(self, state)
    }
    /// Whether [`Engine::snapshot_begin`] copies the state on a stream of the
    /// engine's own, so a request's end can start it without stopping the
    /// slots: only then does the server save a state when its request ends
    /// instead of when the slot's rows are about to be cut. The default is
    /// false: the snapshot is taken on the request path, as before.
    fn snapshots_async(&self) -> bool {
        false
    }
    /// The selected slot's state as a value whose copy may still run
    /// ([`Pending`]): the cache holds it at once, and it is read only after
    /// [`Engine::snapshot_wait`] returned. The slot's rows are not cut, reset
    /// or replaced before then. The default is [`Engine::snapshot`], already
    /// landed.
    fn snapshot_begin(&self) -> Result<Pending, StateError> {
        Ok(Pending::landed(self.snapshot()?))
    }
    /// What `pending`'s copy cost once it has landed, `None` while it runs;
    /// never waits.
    fn snapshot_poll(&self, pending: &Pending) -> Result<Option<Landed>, StateError> {
        pending.flight().poll()
    }
    /// Waits until `pending`'s copy has landed and returns what it cost; the
    /// same answer on every call.
    fn snapshot_wait(&self, pending: &Pending) -> Result<Landed, StateError> {
        pending.flight().wait()
    }
    /// Replaces the cache with the first `n` positions of `state` (a state
    /// this engine's [`Engine::snapshot`] took, `n` a count
    /// [`Saved::keepable`] granted): only the rows and the stores of those
    /// positions go back to the engine. Returns the positions the cache holds
    /// after, `n` or fewer; the server cuts what it keeps of them. The
    /// default is [`Engine::resume`] of the whole state, which holds all of
    /// its positions.
    fn resume_prefix(&mut self, state: &Arc<dyn Saved>, n: usize) -> Result<usize, StateError> {
        let _ = n;
        self.resume(state)?;
        Ok(state.n_tokens())
    }
    /// The most of the first `n` positions slot `from` holds that the
    /// selected slot can take from it ([`Engine::share_prefix`]). The default
    /// is 0: the engine shares nothing between its slots.
    fn share_keepable(&self, from: usize, n: usize) -> usize {
        let _ = (from, n);
        0
    }
    /// Replaces the selected slot's cache with the first positions slot
    /// `from` holds, at most `n` (a count [`Engine::share_keepable`]
    /// granted), taken on the device from `from`'s cache: rows the two slots
    /// then hold in common stay one copy until either writes below them.
    /// `from` keeps its cache. Returns the positions the selected slot holds
    /// after, `n` or fewer, and the bytes it holds on `from`'s memory and
    /// copied. A refusal leaves the selected slot in no defined state (the
    /// server resets it). The default refuses with
    /// [`StateError::Unsupported`].
    fn share_prefix(&mut self, from: usize, n: usize) -> Result<SharedPrefix, StateError> {
        let _ = (from, n);
        Err(StateError::Unsupported("a prefix shared between slots"))
    }
    /// Where to cut a prompt call of positions `first .. end` into calls so
    /// that the `marks` (ascending, each inside the call) a later request may
    /// be cut back to stay keepable: a subset of `marks`, ascending. The
    /// default cuts nowhere: an engine whose cache keeps every position needs
    /// no cut, and one that keeps a mark inside a call takes it from
    /// [`Engine::prefill_marked`].
    fn prefill_splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        let _ = (first, end, marks);
        Vec::new()
    }
    /// The call the server runs for every part of a prompt call: `ids` from
    /// the current position, carrying `media` as [`Engine::prefill_media`]'s
    /// feeds, with the prompt call's `marks` that fall in this part (the
    /// positions past its first and up to its end, ascending), each a
    /// position a later request may be cut back to. A mark at the part's end
    /// is that end: a cut [`Engine::prefill_splits`] asked for, or a quantum's.
    /// The default is [`Engine::prefill`], or [`Engine::prefill_media`] when
    /// `media` is not empty, the marks unread: an engine whose cache keeps
    /// every position, or that cuts where a mark needs it, keeps them already.
    /// An engine that keeps a position inside a call only where it is told
    /// answers here instead and cuts nowhere: one call that keeps its marks
    /// in place of a call a mark.
    fn prefill_marked(
        &mut self,
        ids: &[u32],
        media: &[MediaFeed],
        marks: &[usize],
    ) -> Result<(), EngineError> {
        let _ = marks;
        if media.is_empty() {
            self.prefill(ids)
        } else {
            self.prefill_media(ids, media)
        }
    }
    /// A prompt's last call `ids` (at least one) from the current position
    /// and the step of its last id: [`Engine::prefill_marked`] of all but the
    /// last id with `media` and `marks` (the feeds and the marks of that
    /// part), then [`Engine::next`] of the last, writing its logits to
    /// `logits_out` as that call does. That is the default. An engine whose
    /// prompt call can run the last id as its last row overrides it; it then
    /// stands past every id with the checkpoints the two calls would leave,
    /// so a later request keeps of it what it keeps of theirs, and its logits
    /// are that row's.
    fn prefill_next(
        &mut self,
        ids: &[u32],
        media: &[MediaFeed],
        marks: &[usize],
        logits_out: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let Some((&last, run)) = ids.split_last() else {
            return Err(EngineError("prefill_next of no ids".to_owned()));
        };
        self.prefill_marked(run, media, marks)?;
        self.next(last, logits_out)
    }
    /// The call length `q` at which this engine's prompt work is bit-neutral
    /// to cut: the calls `ids[a..a + k·q]` then `ids[a + k·q..b]` leave the
    /// same bits as the one call `ids[a..b]` (every cell of the cache, every
    /// other state, the next logits, and what a later request keeps of them)
    /// for any `k`, from any position `a` the engine stands at. While another
    /// slot is busy generating, the server runs a prompt call of more than one
    /// quantum as such calls, one a round, a decode round of the busy slots
    /// between them, so a long prompt does not stall the streams already
    /// answering. The grid starts at each call the prompt call would run as
    /// otherwise: its own start, and every cut [`Engine::prefill_splits`]
    /// asks for, which still cuts. The cuts are then the same whatever the
    /// load, and each one this contract covers. A seat that answers `Some(q)`
    /// proves it with a gate of its own: the cut calls against the one call,
    /// those bits and every id equal. `None` (the default): every prompt call
    /// runs in one go.
    fn prompt_quantum(&self) -> Option<NonZeroUsize> {
        None
    }
    /// What the prompt cache did, for this engine's binary to print. The
    /// default writes one plain line to stderr.
    fn note(&self, note: &CacheNote) {
        eprintln!("bloomery-serve: {note}");
    }
    /// What the engine has waited for ([`Waits`]), each total only growing
    /// from a start the engine fixes (its open, or its process's start): the
    /// server counts from its first reading and reads it around every decode
    /// round. The default, for an engine that counts no waits, is zero.
    fn waits(&self) -> Waits {
        Waits::default()
    }
}

/// The time an engine's calls spent waiting, summed ([`Engine::waits`]):
/// `/metrics`' card and PLE wait counters, and each decode round's share by
/// its width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Waits {
    /// On the card's work to finish.
    pub card: Duration,
    /// On the rows of a PLE table (a per-layer embedding) read from storage
    /// for a call that needs them.
    pub ple: Duration,
}

impl Waits {
    /// What was waited since `before`, an earlier reading of the same
    /// engine. Both totals must only grow; equal readings give zero.
    /// Panics naming the counter if either total decreased.
    pub(crate) fn since(self, before: Waits) -> Waits {
        Waits {
            card: self
                .card
                .checked_sub(before.card)
                .expect("Engine::waits card total must only grow"),
            ple: self
                .ple
                .checked_sub(before.ple)
                .expect("Engine::waits PLE total must only grow"),
        }
    }
}

impl std::ops::AddAssign for Waits {
    fn add_assign(&mut self, w: Waits) {
        self.card += w.card;
        self.ple += w.ple;
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

/// One row of [`Engine::advance_slots`]: the slot, the id its pass runs, the
/// ids the pass keeps (appended), what the pass drafted, and the draw of a
/// request that draws. `drafted` is the engine's answer, set by the call;
/// the server checks the kept ids against the engine's rows as
/// [`Engine::advance`]'s caller does.
#[derive(Debug)]
pub struct SlotPass<'a> {
    pub slot: usize,
    pub last: u32,
    pub out: &'a mut Vec<u32>,
    pub drafted: Drafted,
    /// `None` for a greedy pass ([`Engine::advance`]'s); `Some` for a
    /// request that draws, whose pass is [`Engine::advance_sampled`]'s.
    pub draw: Option<SlotDraw<'a>>,
}

/// What a sampled row of [`Engine::advance_slots`] draws with: the ids its
/// draws see (the prompt's tail then the generated ids so far), and the
/// request's sampler, which the engine takes for the call and puts back,
/// its state moved one draw a kept id. The server refuses a row whose
/// sampler did not come back.
pub struct SlotDraw<'a> {
    pub history: &'a [u32],
    pub sampler: &'a mut Option<Sampler>,
}

impl fmt::Debug for SlotDraw<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SlotDraw")
            .field("history", &self.history.len())
            .field("sampler", &self.sampler.is_some())
            .finish()
    }
}

/// `row`'s pass on `engine`'s selected slot: [`Engine::advance`], or for a
/// row that draws [`Engine::advance_sampled`] with its history and its
/// sampler, which goes back into the row whatever the outcome. The pass a
/// row of the default [`Engine::advance_slots`] runs.
pub(crate) fn advance_row<E: Engine + ?Sized>(
    engine: &mut E,
    row: &mut SlotPass<'_>,
) -> Result<Drafted, EngineError> {
    let Some(draw) = row.draw.as_mut() else {
        return engine.advance(row.last, row.out);
    };
    let sampler = draw.sampler.take().ok_or_else(|| {
        EngineError(format!(
            "slot {}: a sampled row without its sampler",
            row.slot
        ))
    })?;
    let (d, back, sampler) =
        engine.advance_sampled(row.last, draw.history.to_vec(), sampler, row.out);
    *draw.sampler = Some(sampler);
    let d = d?;
    if back != draw.history {
        return Err(EngineError(format!(
            "slot {}: a sampled pass handed back another history",
            row.slot
        )));
    }
    Ok(d)
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

/// What a snapshot's copy cost on the engine's copy stream ([`Flight`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Landed {
    /// The copy's own time, from its first byte's start to its last byte's
    /// landing, as the copy stream timed it.
    pub copy_ms: f64,
    /// The copy calls the state took.
    pub calls: u64,
    /// The pinned staging this save allocated for itself, in ms: 0 when the
    /// staging pool lent it memory it already held.
    pub alloc_ms: f64,
}

/// The copy of a snapshot as it runs on the engine's copy stream.
pub trait Flight: Send + Sync {
    /// What the copy cost once it has landed; `None` while it runs. Never
    /// waits.
    fn poll(&self) -> Result<Option<Landed>, StateError>;
    /// Waits for the copy to land and returns what it cost; the same answer
    /// on every call.
    fn wait(&self) -> Result<Landed, StateError>;
}

/// A copy that has landed already (a snapshot the engine took on the request
/// path).
struct Landed0;

impl Flight for Landed0 {
    fn poll(&self) -> Result<Option<Landed>, StateError> {
        Ok(Some(Landed::default()))
    }

    fn wait(&self) -> Result<Landed, StateError> {
        Ok(Landed::default())
    }
}

/// What [`Engine::share_prefix`] put on the selected slot: the positions it
/// holds, the device bytes of its cache that map the other slot's memory,
/// and the bytes copied from it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedPrefix {
    pub positions: usize,
    pub mapped_bytes: u64,
    pub copied_bytes: u64,
}

/// A snapshot whose copy may still run ([`Engine::snapshot_begin`]): the
/// state it becomes, which the cache ranks and counts from the moment it is
/// begun, and the copy's [`Flight`]. The state's bytes are read only after
/// the flight has landed.
#[derive(Clone)]
pub struct Pending {
    state: Arc<dyn Saved>,
    flight: Arc<dyn Flight>,
}

impl Pending {
    /// `state`, whose copy `flight` runs.
    #[must_use]
    pub fn new(state: Arc<dyn Saved>, flight: Arc<dyn Flight>) -> Pending {
        Pending { state, flight }
    }

    /// `state` with nothing in flight: a snapshot taken on the request path.
    #[must_use]
    pub fn landed(state: Arc<dyn Saved>) -> Pending {
        Pending::new(state, Arc::new(Landed0))
    }

    /// The state the copy fills.
    #[must_use]
    pub fn state(&self) -> &Arc<dyn Saved> {
        &self.state
    }

    /// The copy's flight.
    #[must_use]
    pub fn flight(&self) -> &Arc<dyn Flight> {
        &self.flight
    }

    /// Whether `other` is this very snapshot.
    #[must_use]
    pub fn is(&self, other: &Pending) -> bool {
        Arc::ptr_eq(&self.flight, &other.flight)
    }
}

/// The default [`Engine::snapshot`]: [`Engine::save_state`]'s bytes, into
/// host memory.
pub(crate) fn snapshot_bytes<E: Engine + ?Sized>(e: &E) -> Result<Arc<dyn Saved>, StateError> {
    let mut bytes = Vec::new();
    let saved = e.save_state(&mut bytes)?;
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

/// The default [`Engine::resume`]: [`Engine::restore_state`] of the default
/// snapshot's bytes.
pub(crate) fn resume_bytes<E: Engine + ?Sized>(
    e: &mut E,
    state: &Arc<dyn Saved>,
) -> Result<(), StateError> {
    let Some(saved) = state.as_any().downcast_ref::<SavedBytes>() else {
        return Err(StateError::Format(
            "a saved state this engine did not take".to_owned(),
        ));
    };
    let mut input = saved.bytes.as_slice();
    let read = e.restore_state(&mut input)?;
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
    /// The slot's state of `positions` went into the cache. `copied`: the
    /// state is a snapshot taken for the save, or for a park that found no
    /// room; not a state the slot already had, handed over as it is. `ms` is
    /// the save's own wall clock (a park's snapshot is the switch's).
    /// `off_path`: the state is a [`Pending`] begun when its request ended,
    /// its copy running on the engine's copy stream; `ms` is then the call
    /// that began it and [`CacheNote::Landed`] follows. `fallback`: the
    /// engine's staging pool had no room for it even after every landed
    /// state left, so it was taken on the request path: `full` (the pool's
    /// free bytes cannot hold it) or `fragmented` (they could, in one piece;
    /// [`StateError::fallback`]).
    Save {
        positions: usize,
        bytes: u64,
        ms: f64,
        entries: usize,
        cache_bytes: u64,
        copied: bool,
        off_path: bool,
        fallback: Option<&'static str>,
    },
    /// The copy of a saved state of `positions` landed: its bytes, the copy
    /// calls it took, its time on the copy stream, the pinned staging it
    /// allocated (0 when the pool lent it), and `waited_ms`, how long the
    /// request path waited for it (0 when it had landed before a request
    /// needed it).
    Landed {
        positions: usize,
        bytes: u64,
        calls: u64,
        copy_ms: f64,
        alloc_ms: f64,
        waited_ms: f64,
    },
    /// A cached state of `positions` sharing `common` ids with the request
    /// replaced the slot's, which kept `slot_kept` of them; the engine then
    /// keeps `kept`, and `resumed` is the positions the engine put back of
    /// the state (all of them for an engine that resumes whole). `waited_ms`
    /// is how long the request path waited for the state's copy to land
    /// first.
    Load {
        positions: usize,
        common: usize,
        kept: usize,
        slot_kept: usize,
        bytes: u64,
        ms: f64,
        resumed: usize,
        waited_ms: f64,
    },
    /// The slot took the first `kept` positions of slot `from`'s cache on
    /// the device ([`Engine::share_prefix`]) in place of its own, which kept
    /// `slot_kept` of the `common` ids the request shares with `from`: of
    /// its cache, `mapped_bytes` map `from`'s memory and `copied_bytes`
    /// were copied, in `ms`.
    Share {
        from: usize,
        common: usize,
        kept: usize,
        slot_kept: usize,
        mapped_bytes: u64,
        copied_bytes: u64,
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
                copied,
                off_path,
                fallback,
            } => write!(
                f,
                "cache save positions={positions} bytes={bytes} ms={ms:.3} entries={entries} \
                 cache_bytes={cache_bytes} copied={copied} off_path={off_path} fallback={}",
                fallback.unwrap_or("none")
            ),
            CacheNote::Landed {
                positions,
                bytes,
                calls,
                copy_ms,
                alloc_ms,
                waited_ms,
            } => write!(
                f,
                "cache landed positions={positions} bytes={bytes} calls={calls} \
                 copy_ms={copy_ms:.3} alloc_ms={alloc_ms:.3} waited_ms={waited_ms:.3}"
            ),
            CacheNote::Load {
                positions,
                common,
                kept,
                slot_kept,
                bytes,
                ms,
                resumed,
                waited_ms,
            } => write!(
                f,
                "cache load positions={positions} common={common} kept={kept} \
                 slot_kept={slot_kept} bytes={bytes} ms={ms:.3} resumed={resumed} \
                 waited_ms={waited_ms:.3}"
            ),
            CacheNote::Share {
                from,
                common,
                kept,
                slot_kept,
                mapped_bytes,
                copied_bytes,
                ms,
            } => write!(
                f,
                "cache share from={from} common={common} kept={kept} slot_kept={slot_kept} \
                 mapped_bytes={mapped_bytes} copied_bytes={copied_bytes} ms={ms:.3}"
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
    /// Divides a positive logit and multiplies a non-positive one of each
    /// token in the window; 1 is off.
    pub repeat_penalty: f32,
    /// Subtracted from a token's logit once per occurrence in the window.
    pub frequency_penalty: f32,
    /// Subtracted from a token's logit once if it occurs in the window.
    pub presence_penalty: f32,
    /// The window: the last `repeat_last_n` ids of the prompt's tail and the
    /// generated ids, as llama-server's ring buffer over the prompt and the
    /// reply is; 0 is off.
    pub repeat_last_n: usize,
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
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: 64,
            seed: u64::from(u32::MAX),
        }
    }
}

impl SamplingParams {
    /// Whether the penalties change any logit: a non-empty window and one
    /// penalty off its neutral value (llama.cpp's `is_disabled`, negated).
    #[must_use]
    pub fn penalizes(&self) -> bool {
        self.repeat_last_n > 0
            && (self.repeat_penalty != 1.0
                || self.frequency_penalty != 0.0
                || self.presence_penalty != 0.0)
    }
}

/// One request's sampler: `(logits, the history a draw sees) -> id` — the
/// prompt's tail then the generated ids, oldest first.
pub type Sampler = Box<dyn FnMut(&[f32], &[u32]) -> u32 + Send>;

/// A sampler factory's refusal of a request's sampling parameters, in the
/// sampler's own words. The request fails with it, a 400, and is never run
/// with another sampler.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct SamplerRefused(pub String);

/// Builds a sampler per request, or refuses its parameters. The real one comes
/// from the sampler crate; the server ships [`crate::sampling::reference_factory`]
/// so it runs without it.
pub type SamplerFactory =
    Arc<dyn Fn(&SamplingParams) -> Result<Sampler, SamplerRefused> + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::Waits;
    use std::time::Duration;

    #[test]
    fn wait_deltas_require_monotonic_totals() {
        let before = Waits {
            card: Duration::from_millis(10),
            ple: Duration::from_millis(20),
        };
        assert_eq!(before.since(before), Waits::default());
        let growth = Waits {
            card: Duration::from_millis(3),
            ple: Duration::from_millis(7),
        };
        let mut after = before;
        after += growth;
        assert_eq!(after.since(before), growth);
        for (reading, name) in [
            (
                Waits {
                    card: Duration::from_millis(2),
                    ..after
                },
                "card",
            ),
            (
                Waits {
                    ple: Duration::from_millis(2),
                    ..after
                },
                "PLE",
            ),
        ] {
            let panic = std::panic::catch_unwind(|| reading.since(before))
                .expect_err("a decreasing wait total must panic");
            assert_eq!(
                crate::api::panic_text(&*panic),
                format!("Engine::waits {name} total must only grow")
            );
        }
    }
}
