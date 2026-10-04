//! The generation loop shared by `/completion` and `/v1/chat/completions`.
//!
//! `timings` are the server's wall clock around the engine calls, the way
//! llama-server reports them to bench clients. They are not admissible
//! measurements for this repository: those come only from the lease runners.
//! Like llama-server, the prompt phase ends when the first token's logits are
//! out, and the predicted phase runs from there to the end, so `predicted_ms`
//! spans `predicted_n - 1` decode steps.
//!
//! An engine that drafts ([`Engine::advance_rows`] past 1) takes every greedy
//! token after the first through [`Engine::advance`]: a pass keeps one token
//! or more, which the loop then takes one at a time as it takes a step's. A
//! pass that would run past the context is not run; the positions left take
//! one step each, so the ids end where a plain run's end. `timings` then carry
//! llama-server's `draft_n` and `draft_n_accepted`: the ids the passes proposed
//! and kept. A request that samples or bans an id reads the logits row every
//! token, which a pass does not give, so on the same engine it takes one
//! [`Engine::next`] a token and carries no draft counts.
//!
//! A request runs as a [`Gen`], one engine call at a time, so the engine
//! thread can make one call of several slots' steps: each running request's
//! next step a row of one [`Engine::step_slots`], its next drafted pass a row
//! of one [`Engine::advance_slots`]. A drafting engine serves one slot unless
//! it keeps its draft's state per slot ([`Engine::slot_drafts`]).

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde_json::{Value, json};

use crate::engine::{
    CacheNote, Decoder, Drafted, Engine, EngineError, Sampler, SamplerFactory, SamplingParams,
    Saved, SlotPass, SlotRow, StateError, Tokenizer,
};
use crate::promptcache::{self, PromptCache};
use crate::reasoning::{THINK_CLOSE, ThinkSplit};
use crate::sampling;
use crate::slotfile::{self, Counting};
use crate::stop::StopScan;

/// Request knobs after parsing, llama-server defaults filled in.
#[derive(Clone, Debug)]
pub(crate) struct GenParams {
    /// `-1` is unbounded (up to the context).
    pub n_predict: i64,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    pub stream: bool,
    pub timings_per_token: bool,
    pub return_progress: bool,
    pub include_usage: bool,
    /// Keep the longest cached prefix of the prompt (llama-server's default `true`).
    pub cache_prompt: bool,
    /// The think-span budget, in generated ids taken while the span is open:
    /// spending it force-feeds the span's close id, which the model's context
    /// then really carries (llama-server's `--reasoning-budget` per request).
    /// `Some` only when the prompt's text opened the span — a prompt that
    /// already closed it has no reasoning to cap and the budget is silently
    /// ignored, as llama-server ignores the flag when thinking is off.
    pub reasoning_budget: Option<usize>,
}

/// llama-server's `timings` object, as the server clocked it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Timings {
    /// Prompt tokens evaluated by this request: the prompt less what the
    /// cache kept (`cache_n`).
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub n_ctx: usize,
    pub n_past: usize,
    /// Prompt positions kept from the previous request instead of evaluated.
    pub cache_n: usize,
    /// The whole prompt, kept and evaluated (llama-server's `n_prompt_tokens`;
    /// not a `timings` field).
    pub n_prompt: usize,
    /// The ids the draft proposed and the ones the target kept, over every
    /// pass (llama-server's `draft_n`, `draft_n_accepted`).
    pub draft_n: usize,
    pub draft_n_accepted: usize,
    /// The passes that verified a proposal (llama-server's
    /// `n_draft_verif_steps`; not a `timings` field).
    pub draft_passes: usize,
}

impl Timings {
    /// The JSON llama-server emits. A zero count gives NaN per-token fields,
    /// which serialize as `null` exactly as nlohmann writes them. The draft's
    /// two counts are there once a pass proposed, as llama-server's are.
    pub(crate) fn to_json(&self) -> Value {
        let (pn, dn) = (self.prompt_n as f64, self.predicted_n as f64);
        let mut v = json!({
            "prompt_n": self.prompt_n,
            "prompt_ms": self.prompt_ms,
            "prompt_per_token_ms": self.prompt_ms / pn,
            "prompt_per_second": 1e3 / self.prompt_ms * pn,
            "predicted_n": self.predicted_n,
            "predicted_ms": self.predicted_ms,
            "predicted_per_token_ms": self.predicted_ms / dn,
            "predicted_per_second": 1e3 / self.predicted_ms * dn,
            "n_ctx": self.n_ctx,
            "n_past": self.n_past,
            "cache_n": self.cache_n,
        });
        if self.draft_n > 0 {
            v["draft_n"] = json!(self.draft_n);
            v["draft_n_accepted"] = json!(self.draft_n_accepted);
        }
        v
    }

    /// One pass's draft counts into the request's.
    fn book(&mut self, d: Drafted) {
        self.draft_n += d.proposed;
        self.draft_n_accepted += d.accepted;
        self.draft_passes += usize::from(d.proposed > 0);
    }
}

/// Why a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StopKind {
    Eos,
    Word,
    Limit,
}

impl StopKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            StopKind::Eos => "eos",
            StopKind::Word => "word",
            StopKind::Limit => "limit",
        }
    }

    /// OpenAI `finish_reason`.
    pub(crate) fn finish_reason(self) -> &'static str {
        match self {
            StopKind::Eos | StopKind::Word => "stop",
            StopKind::Limit => "length",
        }
    }
}

/// A finished generation.
pub(crate) struct Outcome {
    pub content: String,
    /// Every generated id, the end-of-generation one included.
    pub tokens: Vec<u32>,
    pub stop: StopKind,
    pub stopping_word: String,
    pub truncated: bool,
    pub timings: Timings,
}

/// What the loop hands its sink.
pub(crate) enum Event<'a> {
    /// The prompt is evaluated (llama-server's `prompt_progress`, sent once, complete).
    Prompt(&'a Timings),
    /// Text safe to stream.
    Text(&'a str, &'a Timings),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum GenError {
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// The sink failed: the client went away.
    #[error("client: {0}")]
    Client(#[from] io::Error),
    /// Under `ignore_eos` the sampler chose an end-of-generation id, whose
    /// logit the loop had set to -inf: a sampler defect, not the engine's.
    #[error("ignore_eos: the sampler chose end-of-generation id {0}, whose logit is -inf")]
    Banned(u32),
}

pub(crate) fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// The logits buffer the engine fills, or `None` on a greedy request (empty buffer).
fn out(logits: &mut [f32]) -> Option<&mut [f32]> {
    (!logits.is_empty()).then_some(logits)
}

/// The next id: the engine's argmax, or the sampler's draw. `banned` (the
/// stop ids under `ignore_eos`, as llama-server's `logit_bias_eog`) get a
/// -inf logit first, and a greedy request whose argmax is banned takes the
/// largest other logit (first of equals).
fn choose(
    sampler: &mut Option<Sampler>,
    greedy: u32,
    logits: &mut [f32],
    history: &[u32],
    banned: &[u32],
) -> u32 {
    for &b in banned {
        if let Some(l) = usize::try_from(b).ok().and_then(|i| logits.get_mut(i)) {
            *l = f32::NEG_INFINITY;
        }
    }
    match sampler {
        Some(s) => s(logits, history),
        None if banned.contains(&greedy) => sampling::argmax(logits),
        None => greedy,
    }
}

/// The engine and its slots: the ids each slot's cache holds, one per
/// position, and the host prompt cache of states the slots held before
/// ([`PromptCache`]), one for every slot. Every method but [`Slot::select`],
/// [`Slot::held_of`], [`Slot::step_slots`] and [`Slot::advance_slots`] acts
/// on the selected slot.
pub(crate) struct Slot {
    pub engine: Box<dyn Engine>,
    vocab: Arc<dyn Tokenizer>,
    /// Every id the selected slot has evaluated since its last reset, in
    /// order. The last generated id of a request is not in it: it is never
    /// fed back.
    held: Vec<u32>,
    /// The other slots' ids; the selected slot's entry is empty.
    parked: Vec<Vec<u32>>,
    /// The slot whose ids `held` is.
    cur: usize,
    /// The slot the engine has selected: `None` after a call of several.
    selected: Option<usize>,
    cache: PromptCache,
}

impl Slot {
    /// The engine with one slot.
    #[cfg(test)]
    pub(crate) fn new(engine: Box<dyn Engine>) -> Slot {
        Slot::with_slots(engine, 1)
    }

    /// The engine with `n` slots (at least 1, at most what it declares).
    pub(crate) fn with_slots(engine: Box<dyn Engine>, n: usize) -> Slot {
        Slot {
            vocab: engine.tokenizer(),
            cache: PromptCache::new(engine.cache_ram()),
            engine,
            held: Vec::new(),
            parked: vec![Vec::new(); n],
            cur: 0,
            selected: Some(0),
        }
    }

    /// Makes `slot` the selected one, on the engine too when it changes.
    pub(crate) fn select(&mut self, slot: usize) -> Result<(), EngineError> {
        if self.selected != Some(slot) {
            self.engine.select_slot(slot)?;
            self.selected = Some(slot);
        }
        if self.cur != slot {
            let parked = std::mem::take(&mut self.parked[slot]);
            self.parked[self.cur] = std::mem::replace(&mut self.held, parked);
            self.cur = slot;
        }
        Ok(())
    }

    /// The ids `slot`'s cache holds.
    pub(crate) fn held_of(&self, slot: usize) -> &[u32] {
        if slot == self.cur {
            &self.held
        } else {
            &self.parked[slot]
        }
    }

    fn held_mut(&mut self, slot: usize) -> &mut Vec<u32> {
        if slot == self.cur {
            &mut self.held
        } else {
            &mut self.parked[slot]
        }
    }

    /// One step of several slots ([`Engine::step_slots`]) that books what
    /// each fed. While the call is in flight the rows' ids are empty, so a
    /// failed call leaves no claim about their caches. A row the engine
    /// answers with no id of the vocabulary (its `next` left as it was) is the
    /// engine's error.
    pub(crate) fn step_slots(&mut self, rows: &mut [SlotRow<'_>]) -> Result<(), EngineError> {
        let held: Vec<Vec<u32>> = rows
            .iter()
            .map(|r| std::mem::take(self.held_mut(r.slot)))
            .collect();
        self.selected = None;
        for r in rows.iter_mut() {
            r.next = u32::MAX;
        }
        self.engine.step_slots(rows)?;
        let n_vocab = self.vocab.n_vocab();
        if let Some(r) = rows.iter().find(|r| r.next as usize >= n_vocab) {
            return Err(EngineError(format!(
                "a step of {} slots answered slot {} with id {}, past the vocabulary's {n_vocab}",
                rows.len(),
                r.slot,
                r.next
            )));
        }
        for (r, h) in rows.iter().zip(held) {
            let ids = self.held_mut(r.slot);
            *ids = h;
            ids.push(r.last);
        }
        Ok(())
    }

    /// One drafted pass of several slots ([`Engine::advance_slots`]) that
    /// books what each fed: `last` and every kept token but the last. While
    /// the call is in flight the rows' ids are empty, so a failed call leaves
    /// no claim about their caches. A row whose pass kept no token, more than
    /// the engine's rows, or other than one more than its draft's accepted
    /// ids is the engine's error.
    pub(crate) fn advance_slots(&mut self, rows: &mut [SlotPass<'_>]) -> Result<(), EngineError> {
        let held: Vec<Vec<u32>> = rows
            .iter()
            .map(|r| std::mem::take(self.held_mut(r.slot)))
            .collect();
        self.selected = None;
        for r in rows.iter_mut() {
            r.out.clear();
        }
        self.engine.advance_slots(rows)?;
        let rows_n = self.engine.advance_rows();
        for r in rows.iter() {
            if r.out.is_empty()
                || r.out.len() > rows_n
                || r.out.len() != r.drafted.accepted + 1
                || r.drafted.accepted > r.drafted.proposed
            {
                return Err(EngineError(format!(
                    "slot {}: a pass of at most {rows_n} rows kept {} tokens, its draft {} of {} \
                     proposed ids",
                    r.slot,
                    r.out.len(),
                    r.drafted.accepted,
                    r.drafted.proposed
                )));
            }
        }
        for (r, h) in rows.iter().zip(held) {
            let ids = self.held_mut(r.slot);
            *ids = h;
            ids.push(r.last);
            ids.extend_from_slice(&r.out[..r.out.len() - 1]);
        }
        Ok(())
    }

    /// Brings the cache to the longest prefix of `ids` it can keep, leaving at
    /// least the last id for `next`, and returns that length. `want` false
    /// (`cache_prompt: false`) always resets. While an engine call is in flight
    /// `held` is empty, so a failed call leaves no claim about the cache.
    ///
    /// With the prompt cache on: a cached state that keeps more of `ids` than
    /// the slot replaces the slot's state; the slot's state goes into the
    /// cache first whenever it is replaced or the request keeps less than half
    /// of it (llama-server's rule). A prefix the engine keeps less of than the
    /// request shares is noted with the engine's reason.
    fn reuse(&mut self, ids: &[u32], want: bool) -> Result<usize, EngineError> {
        let (mut common, mut ask, mut k) = self.keep_of(ids, want);
        // The pick is taken out before the slot's state is saved: making room
        // for that state may evict it, and the pick then lives on in `picked`.
        let picked = if want && self.cache.enabled() {
            self.cache.best(ids, k).map(|p| self.cache.take(p))
        } else {
            None
        };
        if self.cache.enabled()
            && !self.held.is_empty()
            && (picked.is_some() || 2 * k < self.held.len())
        {
            self.save_held()?;
        }
        if let Some((entry, state)) = picked {
            let slot_kept = k;
            if let Some((positions, bytes, ms)) = self.load(entry, &state)? {
                (common, ask, k) = self.keep_of(ids, want);
                self.engine.note(&CacheNote::Load {
                    positions,
                    common,
                    kept: k,
                    slot_kept,
                    bytes,
                    ms,
                });
            } else {
                (common, ask, k) = (0, 0, 0);
            }
        }
        if k < ask {
            self.engine.note(&CacheNote::Reuse {
                common,
                ask,
                kept: k,
                held: self.held.len(),
                reason: self.engine.keep_limit(ask),
            });
        }
        let held = std::mem::take(&mut self.held);
        if k == 0 {
            self.engine.reset()?;
        } else if k < held.len() {
            self.engine.cut(k)?;
        }
        self.held = held;
        self.held.truncate(k);
        Ok(k)
    }

    /// The ids `ids` shares with the slot, the most of them a request may
    /// keep (all but its last), and what the engine keeps of those.
    fn keep_of(&self, ids: &[u32], want: bool) -> (usize, usize, usize) {
        let common = if want {
            promptcache::common_prefix(&self.held, ids)
        } else {
            0
        };
        let ask = common.min(ids.len() - 1);
        (common, ask, self.engine.keepable(ask).min(ask))
    }

    /// The slot's state into the prompt cache, unless a cached state already
    /// keeps all of it. A snapshot the engine refuses is noted and not kept;
    /// an engine failure is the request's error.
    fn save_held(&mut self) -> Result<(), EngineError> {
        if self.cache.covers(&self.held) {
            return Ok(());
        }
        let positions = self.held.len();
        let t = Instant::now();
        let state = match self.engine.snapshot() {
            Ok(s) => s,
            Err(StateError::Engine(e)) => return Err(e),
            Err(e) => {
                self.engine.note(&CacheNote::Skip {
                    positions,
                    why: format!("the engine took no snapshot: {e}"),
                });
                return Ok(());
            }
        };
        if state.n_tokens() != positions {
            self.engine.note(&CacheNote::Skip {
                positions,
                why: format!(
                    "the engine's snapshot holds {} positions, the slot {positions}",
                    state.n_tokens()
                ),
            });
            return Ok(());
        }
        for note in self.cache.insert(self.held.clone(), state, ms_since(t)) {
            self.engine.note(&note);
        }
        Ok(())
    }

    /// A cached `state` of `ids` into the engine, the slot then holding its
    /// ids; returns its positions, bytes and the wall time of the resume. A
    /// state the engine refuses leaves the cache, the engine is reset and the
    /// slot holds nothing (`None`); an engine failure is the request's error.
    fn load(
        &mut self,
        ids: Vec<u32>,
        state: &Arc<dyn Saved>,
    ) -> Result<Option<(usize, u64, f64)>, EngineError> {
        self.held.clear();
        let t = Instant::now();
        match self.engine.resume(state) {
            Ok(()) => {
                let ms = ms_since(t);
                let got = (ids.len(), state.n_bytes(), ms);
                self.held = ids;
                Ok(Some(got))
            }
            Err(StateError::Engine(e)) => Err(e),
            Err(e) => {
                self.cache.remove(state);
                self.engine.reset()?;
                self.engine.note(&CacheNote::Skip {
                    positions: ids.len(),
                    why: format!("the engine refused to take the state back: {e}"),
                });
                Ok(None)
            }
        }
    }

    /// Feeds `ids[from..to]`, cut into calls where the engine asks
    /// ([`Engine::prefill_splits`]) at the messages the prompt opens
    /// ([`Tokenizer::user_start`]): its first and its last inside the range.
    fn prefill_marked(&mut self, ids: &[u32], from: usize, to: usize) -> Result<(), EngineError> {
        let marks = message_starts(&ids[..to], &self.vocab.user_start(), from);
        let at = if marks.is_empty() {
            Vec::new()
        } else {
            self.engine.prefill_splits(from, to, &marks)
        };
        if !at.iter().all(|u| marks.contains(u)) || !at.windows(2).all(|w| w[0] < w[1]) {
            return Err(EngineError(format!(
                "the engine cut the prompt call {from}..{to} at {at:?}, not among its marks \
                 {marks:?} in order"
            )));
        }
        let mut first = from;
        for &u in &at {
            self.prefill(&ids[first..u])?;
            first = u;
        }
        self.prefill(&ids[first..to])?;
        if !at.is_empty() {
            self.engine.note(&CacheNote::Split {
                first: from,
                end: to,
                at,
            });
        }
        Ok(())
    }

    /// `prefill` that books what it fed.
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        let held = std::mem::take(&mut self.held);
        self.engine.prefill(ids)?;
        self.held = held;
        self.held.extend_from_slice(ids);
        Ok(())
    }

    /// `next` that books what it fed.
    pub(crate) fn next(
        &mut self,
        last: u32,
        logits: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let held = std::mem::take(&mut self.held);
        let g = self.engine.next(last, logits)?;
        self.held = held;
        self.held.push(last);
        Ok(g)
    }

    /// `advance` that books what it fed: `last` and every kept token but the
    /// last. `out` is cleared first. A pass that kept no token, more than the
    /// engine's rows, or other than one more than its draft's accepted ids is
    /// the engine's error.
    pub(crate) fn advance(
        &mut self,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, EngineError> {
        let held = std::mem::take(&mut self.held);
        out.clear();
        let d = self.engine.advance(last, out)?;
        let rows = self.engine.advance_rows();
        if out.is_empty()
            || out.len() > rows
            || out.len() != d.accepted + 1
            || d.accepted > d.proposed
        {
            return Err(EngineError(format!(
                "a pass of at most {rows} rows kept {} tokens, its draft {} of {} proposed ids",
                out.len(),
                d.accepted,
                d.proposed
            )));
        }
        self.held = held;
        self.held.push(last);
        self.held.extend_from_slice(&out[..out.len() - 1]);
        Ok(d)
    }

    /// Writes the slot to `path` ([`slotfile`]'s layout): the file is written
    /// beside it under a temporary name and renamed over it only once complete,
    /// so a failed save leaves any earlier file at `path` as it was. The cache
    /// is unchanged. Returns the positions saved and the file's bytes.
    pub(crate) fn save(&self, path: &Path) -> Result<(usize, u64), StateError> {
        let partial = partial_path(path);
        let r = self.save_to(&partial).and_then(|bytes| {
            std::fs::rename(&partial, path)
                .map(|()| bytes)
                .map_err(StateError::from)
        });
        if r.is_err() {
            // The save's error is the answer; a partial file left behind is
            // named as one and never read.
            let _ = std::fs::remove_file(&partial);
        }
        r.map(|bytes| (self.held.len(), bytes))
    }

    fn save_to(&self, partial: &Path) -> Result<u64, StateError> {
        let mut w = Counting::new(BufWriter::new(File::create(partial)?));
        slotfile::write_header(&mut w, self.vocab.n_vocab(), &self.held)?;
        let head = w.bytes;
        let saved = self.engine.save_state(&mut w)?;
        if saved.n_tokens != self.held.len() || saved.n_bytes != w.bytes - head {
            return Err(StateError::Format(format!(
                "the engine saved {} positions in {} bytes; the slot holds {} positions and {} \
                 bytes reached the file",
                saved.n_tokens,
                saved.n_bytes,
                self.held.len(),
                w.bytes - head
            )));
        }
        w.flush()?;
        let file = w
            .inner
            .into_inner()
            .map_err(io::IntoInnerError::into_error)?;
        file.sync_all()?;
        Ok(w.bytes)
    }

    /// Replaces the cache with the slot saved in `path`. A file refused before
    /// the engine reads its part (unreadable, another tag, version or
    /// vocabulary, a count past the context, an id past the vocabulary) and an
    /// engine that does not restore leave the cache as it was; once the engine
    /// has read, any failure resets it and the slot holds nothing. Returns the
    /// positions restored and the file's bytes.
    pub(crate) fn restore(&mut self, path: &Path) -> Result<(usize, u64), StateError> {
        let mut r = Counting::new(BufReader::new(File::open(path)?));
        let ids = slotfile::read_header(&mut r, self.vocab.n_vocab(), self.engine.ctx_max())?;
        let head = r.bytes;
        let held = std::mem::take(&mut self.held);
        let restored = match self.engine.restore_state(&mut r) {
            Ok(s) => s,
            Err(StateError::Unsupported(what)) => {
                self.held = held;
                return Err(StateError::Unsupported(what));
            }
            Err(e @ StateError::Engine(_)) => return Err(e),
            Err(e) => {
                self.engine.reset()?;
                return Err(e);
            }
        };
        let read = r.bytes - head;
        let trailing = r.read(&mut [0u8; 1]);
        let mismatch = match trailing {
            Err(e) => Some(format!("reading past the engine's state: {e}")),
            Ok(n) if n > 0 => Some(format!(
                "the engine read {read} bytes and the file runs on past them"
            )),
            Ok(_) if restored.n_tokens != ids.len() || restored.n_bytes != read => Some(format!(
                "the engine restored {} positions from {} bytes; the file holds {} positions and \
                 the engine read {read} bytes",
                restored.n_tokens,
                restored.n_bytes,
                ids.len()
            )),
            Ok(_) => None,
        };
        if let Some(m) = mismatch {
            self.engine.reset()?;
            return Err(StateError::Format(m));
        }
        let n = ids.len();
        self.held = ids;
        Ok((n, r.bytes))
    }

    /// Drops the whole cache and the ids it held; returns how many it held.
    pub(crate) fn erase(&mut self) -> Result<usize, EngineError> {
        let held = std::mem::take(&mut self.held);
        self.engine.reset()?;
        Ok(held.len())
    }
}

/// The positions in `from + 1 .. ids.len()` where `marker` starts: the first
/// and the last of them (one when they are the same). An empty marker marks
/// nothing.
fn message_starts(ids: &[u32], marker: &[u32], from: usize) -> Vec<usize> {
    if marker.is_empty() {
        return Vec::new();
    }
    let mut at = (from + 1..ids.len()).filter(|&u| ids[u..].starts_with(marker));
    let first = at.next();
    let last = at.next_back();
    first.into_iter().chain(last).collect()
}

/// Where a save to `path` is written before it is renamed there: a name no
/// slot file can take (a slot file name has no `:`), and no other save's —
/// the process id and a count of the process's saves, so two servers of one
/// process saving into one directory never write one file.
fn partial_path(path: &Path) -> PathBuf {
    static SAVES: AtomicU64 = AtomicU64::new(0);
    let n = SAVES.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(".bloomery-slot-{}-{n}:partial", std::process::id()))
}

/// What a generation needs next ([`Gen::pump`]).
pub(crate) enum Need {
    /// One step from this id: [`Engine::next`], or a row of
    /// [`Engine::step_slots`], whose answer goes to [`Gen::stepped`].
    Step(u32),
    /// One drafted pass from this id ([`Engine::advance`] into
    /// [`Gen::kept_mut`]), whose count goes to [`Gen::advanced`].
    Advance(u32),
    /// Nothing more: [`Gen::finish`] ends it.
    Done,
}

/// The think-span budget's state in [`Gen`]: the budget left while the span is
/// open, the span's close detection over the generated text (the output's own
/// split is api-side; this instance only decides when the span closes), and the
/// ids of the close, `at` naming the next of them the loop force-feeds —
/// `close.len()` while the budget holds.
struct Think {
    /// Budget left; the id whose text closes the span does not spend.
    left: usize,
    split: ThinkSplit,
    close: Vec<u32>,
    at: usize,
}

impl Think {
    /// A budget of `left` generated ids over a span the prompt opened; at 0 the
    /// close is forced from the first taken id.
    fn new(left: usize, close: Vec<u32>) -> Think {
        let n = close.len();
        Think {
            left,
            split: ThinkSplit::inside(),
            close,
            at: if left == 0 { 0 } else { n },
        }
    }

    /// Whether a close id is queued to force.
    fn forcing(&self) -> bool {
        self.at < self.close.len()
    }

    /// The next close id to force.
    fn next_forced(&mut self) -> Option<u32> {
        self.forcing().then(|| {
            let id = self.close[self.at];
            self.at += 1;
            id
        })
    }
}

/// One request's generation on its slot, run a call at a time so the engine
/// thread can step several slots' generations in one engine call. `ids` is
/// non-empty and shorter than the context. [`Gen::timings`] are filled even
/// when the sink failed midway (the caller's counters still see the work
/// done). A greedy request never asks the engine for its logits, and takes its
/// tokens after the first through the engine's passes. `tick` sees the timings
/// after every generated token.
pub(crate) struct Gen {
    n: usize,
    ctx_max: usize,
    sampler: Option<Sampler>,
    stops: Vec<u32>,
    /// The stop ids under `ignore_eos`, else empty.
    banned: Vec<u32>,
    /// The engine's logits row, or empty on a greedy request.
    logits: Vec<f32>,
    tim: Timings,
    t1: Instant,
    budget: usize,
    scan: StopScan,
    dec: Box<dyn Decoder>,
    generated: Vec<u32>,
    stop: StopKind,
    stopping_word: String,
    truncated: bool,
    /// A pass's rows; 1 steps. Only a greedy request with no banned id passes.
    rows: usize,
    /// The last pass's kept tokens, and how many of them the loop has taken.
    kept: Vec<u32>,
    taken: usize,
    /// The token the loop takes next; `None` once there is none.
    tok: Option<u32>,
    /// The think-span budget, `None` on a request without one (or whose prompt
    /// never opened the span) and once the span closed.
    think: Option<Think>,
}

impl Gen {
    /// A generation of `p` for a prompt of `n` ids on `slot`'s engine; nothing
    /// runs yet.
    pub(crate) fn new(slot: &Slot, factory: &SamplerFactory, n: usize, p: &GenParams) -> Gen {
        let sampler = (p.sampling.temperature > 0.0).then(|| factory(&p.sampling));
        let stops = slot.vocab.stops();
        let banned = if p.ignore_eos {
            stops.clone()
        } else {
            Vec::new()
        };
        // A greedy request reads the logits only to step past a banned argmax.
        let logits = vec![
            0.0f32;
            if sampler.is_some() || !banned.is_empty() {
                slot.vocab.n_vocab()
            } else {
                0
            }
        ];
        let rows = if sampler.is_none() && banned.is_empty() {
            slot.engine.advance_rows()
        } else {
            1
        };
        Gen {
            n,
            ctx_max: slot.engine.ctx_max(),
            sampler,
            stops,
            banned,
            logits,
            tim: Timings::default(),
            t1: Instant::now(),
            budget: usize::try_from(p.n_predict).unwrap_or(usize::MAX),
            scan: StopScan::new(p.stop.clone()),
            dec: slot.vocab.decoder(),
            generated: Vec::new(),
            stop: StopKind::Limit,
            stopping_word: String::new(),
            truncated: false,
            rows,
            kept: Vec::with_capacity(rows),
            taken: 0,
            tok: None,
            think: p
                .reasoning_budget
                .map(|left| Think::new(left, slot.vocab.encode(THINK_CLOSE))),
        }
    }

    /// The timings so far.
    pub(crate) fn timings(&self) -> &Timings {
        &self.tim
    }

    /// The buffer a pass's kept tokens go to.
    pub(crate) fn kept_mut(&mut self) -> &mut Vec<u32> {
        &mut self.kept
    }

    /// The logits buffer a step writes, or `None` on a greedy request.
    pub(crate) fn logits_out(&mut self) -> Option<&mut [f32]> {
        out(&mut self.logits)
    }

    /// The id the loop takes for an engine answer `g`: the next forced close id
    /// while one is queued (the engine's answer is discarded — the loop never
    /// commits it, so nothing desyncs), else the sampled or greedy choice.
    fn answer(&mut self, g: u32) -> u32 {
        if let Some(id) = self.think.as_mut().and_then(Think::next_forced) {
            return id;
        }
        choose(
            &mut self.sampler,
            g,
            &mut self.logits,
            &self.generated,
            &self.banned,
        )
    }

    /// Whether a close id is queued to force: the budget spent with the span
    /// still open.
    fn forcing(&self) -> bool {
        self.think.as_ref().is_some_and(Think::forcing)
    }

    /// The think budget left, `usize::MAX` with none: a pass keeps at most
    /// `rows` tokens and each spends, so it starts only while they all fit.
    fn think_left(&self) -> usize {
        self.think.as_ref().map_or(usize::MAX, |t| t.left)
    }

    /// One taken id against the think budget: its text first — a piece that
    /// closes the span retires the tracker, the closing id spending nothing —
    /// then the count, which arms the forced close at zero.
    fn spend(&mut self, piece: Option<&str>) {
        let Some(t) = self.think.as_mut() else {
            return;
        };
        if let Some(text) = piece
            && t.split.push(text).closed
        {
            self.think = None;
            return;
        }
        if t.left > 0 {
            t.left -= 1;
            if t.left == 0 {
                t.at = 0;
            }
        }
    }

    /// The prompt on the selected slot: what the cache keeps of `ids`, the
    /// rest fed, and the first token's logits read.
    pub(crate) fn prompt(
        &mut self,
        slot: &mut Slot,
        ids: &[u32],
        p: &GenParams,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<(), GenError> {
        let n = self.n;
        // Only a greedy request with no banned id passes, and its first token is
        // the prompt's step: the passes make at most the rest.
        let passes = self.sampler.is_none() && self.banned.is_empty();
        slot.engine.will_reply(match usize::try_from(p.n_predict) {
            Ok(n) if passes => Some(n.saturating_sub(1)),
            Err(_) if passes => None,
            _ => Some(0),
        });
        let cache_n = slot.reuse(ids, p.cache_prompt)?;
        let t0 = Instant::now();
        slot.prefill_marked(ids, cache_n, n - 1)?;
        let greedy = slot.next(ids[n - 1], out(&mut self.logits))?;
        self.tim = Timings {
            prompt_n: n - cache_n,
            prompt_ms: ms_since(t0),
            n_ctx: self.ctx_max,
            n_past: n,
            cache_n,
            n_prompt: n,
            ..Timings::default()
        };
        self.t1 = Instant::now();
        if p.return_progress {
            sink(Event::Prompt(&self.tim))?;
        }
        if self.budget > 0 {
            self.tok = Some(self.answer(greedy));
        }
        Ok(())
    }

    /// Takes the tokens in hand, one at a time, until the generation stops or
    /// needs the engine.
    pub(crate) fn pump(
        &mut self,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
        tick: &mut dyn FnMut(&Timings),
    ) -> Result<Need, GenError> {
        while let Some(tok) = self.tok.take() {
            if self.banned.contains(&tok) {
                return Err(GenError::Banned(tok));
            }
            self.generated.push(tok);
            self.tim.predicted_n = self.generated.len();
            self.tim.predicted_ms = ms_since(self.t1);
            tick(&self.tim);
            if self.stops.contains(&tok) {
                self.stop = StopKind::Eos;
                break;
            }
            let piece = self.dec.push(tok);
            if let Some(piece) = &piece {
                let pushed = self.scan.push(piece);
                if !pushed.send.is_empty() {
                    sink(Event::Text(&pushed.send, &self.tim))?;
                }
                if let Some(w) = pushed.stopped {
                    self.stop = StopKind::Word;
                    self.stopping_word = w;
                    break;
                }
            }
            // The think budget: the text first, then the count.
            self.spend(piece.as_deref());
            if self.generated.len() >= self.budget {
                break;
            }
            // While a close id is queued, every taken id is stepped into the
            // engine, whose context then carries the close text: a pass cannot
            // run (one starts only while the budget holds `rows`, below, and
            // keeps at most `rows`, so the budget spends at the earliest with
            // the pass's last kept token taken — none remains for the fast
            // path below to drop), and the context bound stops the forced ids
            // like any token.
            if self.forcing() {
                let (n, len) = (self.n, self.generated.len());
                if n + len > self.ctx_max {
                    self.truncated = true;
                    break;
                }
                return Ok(Need::Step(tok));
            }
            // The last pass evaluated this token already.
            if let Some(&t) = self.kept.get(self.taken) {
                self.taken += 1;
                self.tok = Some(t);
                continue;
            }
            let (n, len) = (self.n, self.generated.len());
            // Feeding `tok` takes position n + len - 1; the cache holds ctx_max.
            if n + len > self.ctx_max {
                self.truncated = true;
                break;
            }
            // A pass takes `rows` positions from there, and every token it
            // keeps spends the think budget: it starts only while the budget
            // holds them all, so no kept token spends past the cap.
            if self.rows > 1
                && self.think_left() >= self.rows
                && n + len + self.rows - 1 <= self.ctx_max
            {
                return Ok(Need::Advance(tok));
            }
            return Ok(Need::Step(tok));
        }
        Ok(Need::Done)
    }

    /// The engine's answer `g` to a [`Need::Step`].
    pub(crate) fn stepped(&mut self, g: u32) {
        self.tim.n_past = self.n + self.generated.len();
        self.tok = Some(self.answer(g));
    }

    /// What a [`Need::Advance`] drafted; its tokens are in [`Gen::kept_mut`].
    /// The force is never armed here by [`Gen::pump`]'s pass gate: a pass
    /// starts only while the budget holds `rows` and keeps at most `rows`, so
    /// its first kept token is always there for the taking.
    pub(crate) fn advanced(&mut self, d: Drafted) {
        self.tim.book(d);
        self.tim.n_past = self.n + self.generated.len() + self.kept.len() - 1;
        self.tok = Some(self.kept[0]);
        self.taken = 1;
    }

    /// The text still held, and the outcome.
    pub(crate) fn finish(
        &mut self,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<Outcome, GenError> {
        if self.stop != StopKind::Word {
            let tail = self.dec.flush();
            let pushed = self.scan.push(&tail);
            let mut rest = pushed.send;
            if let Some(w) = pushed.stopped {
                self.stop = StopKind::Word;
                self.stopping_word = w;
            } else {
                rest.push_str(&self.scan.finish());
            }
            if !rest.is_empty() {
                sink(Event::Text(&rest, &self.tim))?;
            }
        }
        self.tim.predicted_ms = ms_since(self.t1);
        Ok(Outcome {
            content: self.scan.text().to_owned(),
            tokens: std::mem::take(&mut self.generated),
            stop: self.stop,
            stopping_word: std::mem::take(&mut self.stopping_word),
            truncated: self.truncated,
            timings: self.tim.clone(),
        })
    }

    /// The whole generation on `slot` alone, its calls one at a time.
    #[cfg(test)]
    fn run(
        &mut self,
        slot: &mut Slot,
        ids: &[u32],
        p: &GenParams,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
        tick: &mut dyn FnMut(&Timings),
    ) -> Result<Outcome, GenError> {
        self.prompt(slot, ids, p, sink)?;
        loop {
            match self.pump(sink, tick)? {
                Need::Done => return self.finish(sink),
                Need::Step(t) => {
                    let g = slot.next(t, out(&mut self.logits))?;
                    self.stepped(g);
                }
                Need::Advance(t) => {
                    let d = slot.advance(t, &mut self.kept)?;
                    self.advanced(d);
                }
            }
        }
    }
}

/// Runs one request on the slot alone ([`Gen`]); `timings_out` gets its
/// timings whether it ends or fails.
#[cfg(test)]
pub(crate) fn generate(
    slot: &mut Slot,
    factory: &SamplerFactory,
    ids: &[u32],
    p: &GenParams,
    sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    tick: &mut dyn FnMut(&Timings),
    timings_out: &mut Timings,
) -> Result<Outcome, GenError> {
    let mut g = Gen::new(slot, factory, ids.len(), p);
    let r = g.run(slot, ids, p, sink, tick);
    timings_out.clone_from(g.timings());
    r
}

#[cfg(test)]
mod tests {
    use super::partial_path;
    use crate::slotfile;
    use std::path::Path;

    /// Two saves to one path write two partial files, neither of which a
    /// slot file can be named.
    #[test]
    fn partial_paths_are_distinct_per_save() {
        let path = Path::new("/slots/a.bin");
        let (a, b) = (partial_path(path), partial_path(path));
        assert_ne!(a, b, "two saves share the partial file {}", a.display());
        for p in [&a, &b] {
            assert_eq!(p.parent(), path.parent());
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .expect("a UTF-8 name");
            assert!(
                !slotfile::valid_filename(name),
                "{name} is a valid slot name"
            );
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use std::any::Any;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};

    use super::{GenParams, Slot, Timings, generate};
    use crate::engine::{
        CacheNote, Decoder, Engine, EngineError, SamplingParams, Saved, StateError, Tokenizer,
    };
    use crate::mock::{MockEngine, MockTokenizer};
    use crate::sampling;

    /// The mock vocabulary, whose `<｜User｜>` (id 2) opens a message.
    struct MarkTok(MockTokenizer);

    impl Tokenizer for MarkTok {
        fn encode(&self, text: &str) -> Vec<u32> {
            self.0.encode(text)
        }
        fn decode(&self, ids: &[u32]) -> String {
            self.0.decode(ids)
        }
        fn decoder(&self) -> Box<dyn Decoder> {
            self.0.decoder()
        }
        fn bos(&self) -> u32 {
            self.0.bos()
        }
        fn eos(&self) -> u32 {
            self.0.eos()
        }
        fn add_bos(&self) -> bool {
            self.0.add_bos()
        }
        fn n_vocab(&self) -> usize {
            self.0.n_vocab()
        }
        fn user_start(&self) -> Vec<u32> {
            vec![2]
        }
    }

    /// What a [`Probe`] saw: its notes and the length of every prefill call.
    #[derive(Default)]
    struct Log {
        notes: Vec<CacheNote>,
        calls: Vec<usize>,
        /// What each request told [`Engine::will_reply`].
        replies: Vec<Option<usize>>,
    }

    /// The mock engine under a cut rule like V4.1's CED hole: a position
    /// strictly inside a prompt call is not kept, and a cut there falls to the
    /// call's start. It caches its states (`budget` bytes) and, with `split`,
    /// cuts a prompt call at every mark it is offered.
    struct Probe {
        inner: MockEngine,
        tok: Arc<MarkTok>,
        ctx: Vec<u32>,
        calls: Vec<Range<usize>>,
        budget: u64,
        split: bool,
        /// Cut one position past each mark instead: a defective engine.
        off_mark: bool,
        /// Refuse every state it is handed back.
        refuse: bool,
        log: Arc<Mutex<Log>>,
    }

    fn rule(n: usize, calls: &[Range<usize>]) -> (usize, Option<String>) {
        let mut k = n;
        let mut why = None;
        while let Some(c) = calls.iter().find(|c| c.start < k && k < c.end) {
            why = Some(format!("position {k} lies inside prompt call {c:?}"));
            k = c.start;
        }
        (k, why)
    }

    struct ProbeSaved {
        ctx: Vec<u32>,
        calls: Vec<Range<usize>>,
    }

    impl Saved for ProbeSaved {
        fn n_tokens(&self) -> usize {
            self.ctx.len()
        }
        fn n_bytes(&self) -> u64 {
            4 * self.ctx.len() as u64
        }
        fn keepable(&self, n: usize) -> usize {
            rule(n.min(self.ctx.len()), &self.calls).0
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl Probe {
        fn slot(budget: u64, split: bool) -> (Slot, Arc<Mutex<Log>>) {
            let log = Arc::new(Mutex::new(Log::default()));
            let probe = Probe {
                inner: MockEngine::new(4096),
                tok: Arc::new(MarkTok(MockTokenizer)),
                ctx: Vec::new(),
                calls: Vec::new(),
                budget,
                split,
                off_mark: false,
                refuse: false,
                log: Arc::clone(&log),
            };
            (Slot::new(Box::new(probe)), log)
        }
    }

    impl Engine for Probe {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.tok.clone()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.inner.prefill(ids)?;
            let at = self.ctx.len();
            self.calls.push(at..at + ids.len());
            self.ctx.extend_from_slice(ids);
            self.log.lock().expect("the log").calls.push(ids.len());
            Ok(())
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            let g = self.inner.next(last, out)?;
            self.ctx.push(last);
            Ok(g)
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.ctx.clear();
            self.calls.clear();
            self.inner.reset()
        }
        fn will_reply(&mut self, tokens: Option<usize>) {
            self.log.lock().expect("the log").replies.push(tokens);
        }
        fn keepable(&self, n: usize) -> usize {
            rule(n.min(self.ctx.len()), &self.calls).0
        }
        fn keep_limit(&self, n: usize) -> Option<String> {
            rule(n.min(self.ctx.len()), &self.calls).1
        }
        fn cut(&mut self, n: usize) -> Result<(), EngineError> {
            if self.keepable(n) != n {
                return Err(EngineError(format!("probe: {n} is not a kept point")));
            }
            self.inner.cut(n)?;
            self.ctx.truncate(n);
            self.calls.retain_mut(|c| {
                c.end = c.end.min(n);
                c.start < c.end
            });
            Ok(())
        }
        fn ctx_max(&self) -> usize {
            self.inner.ctx_max()
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
        fn cache_ram(&self) -> u64 {
            self.budget
        }
        fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
            Ok(Arc::new(ProbeSaved {
                ctx: self.ctx.clone(),
                calls: self.calls.clone(),
            }))
        }
        fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
            let s = state
                .as_any()
                .downcast_ref::<ProbeSaved>()
                .ok_or_else(|| StateError::Format("not a probe state".to_owned()))?;
            if self.refuse {
                return Err(StateError::Format("probe: refused".to_owned()));
            }
            self.inner.reset()?;
            self.inner.prefill(&s.ctx)?;
            self.ctx.clone_from(&s.ctx);
            self.calls.clone_from(&s.calls);
            Ok(())
        }
        fn prefill_splits(&self, _first: usize, _end: usize, marks: &[usize]) -> Vec<usize> {
            match (self.split, self.off_mark) {
                (true, false) => marks.to_vec(),
                (true, true) => marks.iter().map(|u| u + 1).collect(),
                (false, _) => Vec::new(),
            }
        }
        fn note(&self, note: &CacheNote) {
            self.log.lock().expect("the log").notes.push(note.clone());
        }
    }

    /// One greedy request of at most 8 tokens: what the cache kept, and
    /// what came out.
    fn run(slot: &mut Slot, ids: &[u32]) -> (usize, Vec<u32>) {
        let p = GenParams {
            n_predict: 8,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: false,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
        };
        let factory = sampling::reference_factory();
        let mut tim = Timings::default();
        let o = generate(
            slot,
            &factory,
            ids,
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        )
        .expect("a generation");
        (o.timings.cache_n, o.tokens)
    }

    fn enc(text: &str) -> Vec<u32> {
        MockTokenizer.encode(text)
    }

    /// A second conversation between two turns of the first costs the first
    /// nothing: its state goes to the cache and comes back, and the turn
    /// keeps all it held and generates what a fresh prefill does.
    #[test]
    fn an_interleaved_conversation_keeps_the_first_ones_prefix() {
        let (mut slot, log) = Probe::slot(1 << 20, false);
        let a1 = enc("<｜User｜>the cat sat on the mat and the ");
        let (_, t1) = run(&mut slot, &a1);
        run(&mut slot, &enc("<｜User｜>xyz uvw xyz "));
        let mut a2 = a1.clone();
        a2.extend(&t1);
        a2.extend(enc("<｜User｜>and then the "));
        let (kept, t2) = run(&mut slot, &a2);
        assert_eq!(
            kept,
            a1.len() + t1.len() - 1,
            "{:?}",
            log.lock().expect("log").notes
        );
        let (fresh, _) = Probe::slot(0, false);
        let mut fresh = fresh;
        assert_eq!(run(&mut fresh, &a2).1, t2, "the restored turn's ids");
        let notes = &log.lock().expect("log").notes;
        assert!(
            notes
                .iter()
                .any(|n| matches!(n, CacheNote::Load { positions, .. }
                if *positions == a1.len() + t1.len() - 1)),
            "{notes:?}"
        );
    }

    /// A prefix the engine keeps less of than the request shares is noted,
    /// with the engine's reason.
    #[test]
    fn a_prefix_the_engine_drops_is_noted_with_its_reason() {
        let (mut slot, log) = Probe::slot(0, false);
        let a = enc("<｜User｜>the cat sat on the mat and the ");
        run(&mut slot, &a);
        let mut b = a[..20].to_vec();
        b.extend(enc("dog ran"));
        let (kept, _) = run(&mut slot, &b);
        assert_eq!(kept, 0);
        let notes = &log.lock().expect("log").notes;
        assert!(
            notes.iter().any(
                |n| matches!(n, CacheNote::Reuse { common: 20, ask: 20, kept: 0,
                reason: Some(r), .. } if r.contains("inside prompt call"))
            ),
            "{notes:?}"
        );
    }

    /// A prompt call is cut at its first and last message starts where the
    /// engine asks, and the cut position is then kept by a request that
    /// shares the prompt up to it — with the same ids a fresh prefill gives.
    #[test]
    fn a_prompt_call_is_cut_at_its_messages() {
        let (mut slot, log) = Probe::slot(0, true);
        let system = enc("be brief and kind. ");
        let mut a = system.clone();
        a.extend(enc(
            "<｜User｜>the cat sat<｜Assistant｜>on the mat<｜User｜>and the ",
        ));
        run(&mut slot, &a);
        let u1 = system.len();
        let u2 = a.len() - enc("<｜User｜>and the ").len();
        assert_eq!(
            log.lock().expect("log").calls[..3],
            [u1, u2 - u1, a.len() - 1 - u2],
            "the prompt call's pieces"
        );
        let mut b = system.clone();
        b.extend(enc("<｜User｜>a dog ran and the "));
        let (kept, tb) = run(&mut slot, &b);
        assert_eq!(kept, u1, "the system prompt, where the second call starts");
        let (mut fresh, _) = Probe::slot(0, false);
        assert_eq!(run(&mut fresh, &b).1, tb);
    }

    /// An engine that cuts a prompt call anywhere but at the marks it was
    /// offered fails the request by name; nothing is fed.
    #[test]
    fn a_cut_off_the_marks_is_a_named_error() {
        let log = Arc::new(Mutex::new(Log::default()));
        let probe = Probe {
            inner: MockEngine::new(4096),
            tok: Arc::new(MarkTok(MockTokenizer)),
            ctx: Vec::new(),
            calls: Vec::new(),
            budget: 0,
            split: true,
            off_mark: true,
            refuse: false,
            log: Arc::clone(&log),
        };
        let mut slot = Slot::new(Box::new(probe));
        let a = enc("be brief and kind. <｜User｜>the cat sat on the mat and the ");
        let p = GenParams {
            n_predict: 4,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: false,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
        };
        let mut tim = Timings::default();
        let r = generate(
            &mut slot,
            &sampling::reference_factory(),
            &a,
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        );
        let e = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("not among its marks"), "{e}");
        assert!(log.lock().expect("log").calls.is_empty());
    }

    /// A full cache: saving the slot's state before the pick is taken back
    /// evicts the oldest state, and the pick is still the one taken back.
    #[test]
    fn the_pick_survives_the_save_that_makes_room() {
        let a1 = enc("<｜User｜>the cat sat on the mat and the ");
        let b = enc("<｜User｜>xyz uvw xyz ");
        let c = enc("<｜User｜>abc def abc def abc ");
        let (mut probe_slot, _) = Probe::slot(1 << 20, false);
        let (_, t1) = run(&mut probe_slot, &a1);
        let held_a = a1.len() + t1.len() - 1;
        let (_, tb) = run(&mut probe_slot, &b);
        let (_, tc) = run(&mut probe_slot, &c);
        let held_c = c.len() + tc.len() - 1;
        let held_b = b.len() + tb.len() - 1;
        assert!(held_b <= held_a, "the fixture needs b no longer than a");
        // Room for a's state and c's, not for b's beside them.
        let budget = 4 * (held_a + held_c) as u64;
        let (mut slot, log) = Probe::slot(budget, false);
        run(&mut slot, &a1);
        run(&mut slot, &b);
        run(&mut slot, &c);
        let mut a2 = a1.clone();
        a2.extend(&t1);
        a2.extend(enc("<｜User｜>and then the "));
        let (kept, t2) = run(&mut slot, &a2);
        assert_eq!(kept, held_a, "{:?}", log.lock().expect("log").notes);
        let (mut fresh, _) = Probe::slot(0, false);
        assert_eq!(run(&mut fresh, &a2).1, t2);
    }

    /// A state the engine refuses to take back leaves the cache, is noted,
    /// and the request runs from a reset with a fresh prefill's ids.
    #[test]
    fn a_refused_state_leaves_the_cache() {
        let log = Arc::new(Mutex::new(Log::default()));
        let probe = Probe {
            inner: MockEngine::new(4096),
            tok: Arc::new(MarkTok(MockTokenizer)),
            ctx: Vec::new(),
            calls: Vec::new(),
            budget: 1 << 20,
            split: false,
            off_mark: false,
            refuse: true,
            log: Arc::clone(&log),
        };
        let mut slot = Slot::new(Box::new(probe));
        let a1 = enc("<｜User｜>the cat sat on the mat and the ");
        let (_, t1) = run(&mut slot, &a1);
        run(&mut slot, &enc("<｜User｜>xyz uvw xyz "));
        let mut a2 = a1.clone();
        a2.extend(&t1);
        a2.extend(enc("<｜User｜>and then the "));
        let (kept, t2) = run(&mut slot, &a2);
        assert_eq!(kept, 0);
        let (mut fresh, _) = Probe::slot(0, false);
        assert_eq!(run(&mut fresh, &a2).1, t2);
        let notes = log.lock().expect("log").notes.clone();
        let refused = |n: &CacheNote| matches!(n, CacheNote::Skip { why, .. } if why.contains("refused to take the state back"));
        assert_eq!(notes.iter().filter(|n| refused(n)).count(), 1, "{notes:?}");
        assert!(!slot.cache.covers(&a1), "the refused state is still cached");
    }

    /// Each request tells the engine its reply before the cache is reused:
    /// a greedy one the tokens after its first (the passes make at most
    /// those), one that takes no pass — it bans the end of generation —
    /// none.
    #[test]
    fn a_request_tells_the_engine_its_reply_before_the_reuse() {
        let (mut slot, log) = Probe::slot(0, false);
        let ids = enc("<｜User｜>the cat sat on the mat and the ");
        run(&mut slot, &ids);
        let p = GenParams {
            n_predict: 8,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: true,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
        };
        let factory = sampling::reference_factory();
        let mut tim = Timings::default();
        generate(
            &mut slot,
            &factory,
            &ids,
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        )
        .expect("a generation");
        assert_eq!(log.lock().expect("log").replies, [Some(7), Some(0)]);
    }
}

#[cfg(test)]
mod slot_tests {
    use std::sync::Arc;

    use super::Slot;
    use crate::engine::{Engine, EngineError, SlotRow, Tokenizer};
    use crate::mock::MockEngine;

    /// The two-slot mock whose steps of several slots answer nothing.
    struct Mute(MockEngine);

    impl Engine for Mute {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.0.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.0.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.0.next(last, out)
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.0.reset()
        }
        fn ctx_max(&self) -> usize {
            self.0.ctx_max()
        }
        fn describe(&self) -> String {
            self.0.describe()
        }
        fn slots(&self) -> usize {
            self.0.slots()
        }
        fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
            self.0.select_slot(slot)
        }
        fn step_slots(&mut self, _rows: &mut [SlotRow<'_>]) -> Result<(), EngineError> {
            Ok(())
        }
    }

    fn rows() -> Vec<SlotRow<'static>> {
        (0..2)
            .map(|slot| SlotRow {
                slot,
                last: 10,
                logits: None,
                next: 0,
            })
            .collect()
    }

    /// A step of several slots books each row's id on its own slot, and a
    /// row the engine leaves unanswered is a named error, never id 0.
    #[test]
    fn a_step_of_several_slots_books_each_and_names_a_row_left_unanswered() {
        let mut slot = Slot::with_slots(Box::new(MockEngine::new(64).with_slots(2)), 2);
        let mut r = rows();
        slot.step_slots(&mut r).expect("the mock answers");
        assert_eq!((slot.held_of(0), slot.held_of(1)), (&[10][..], &[10][..]));
        let mut mute = Slot::with_slots(Box::new(Mute(MockEngine::new(64).with_slots(2))), 2);
        let e = mute
            .step_slots(&mut rows())
            .expect_err("a row left unanswered");
        assert!(e.0.contains("answered slot 0 with id 4294967295"), "{e}");
    }
}

#[cfg(test)]
mod think_tests {
    use std::sync::{Arc, Mutex};

    use super::{GenParams, Outcome, Slot, StopKind, Timings, generate};
    use crate::engine::{Drafted, Engine, EngineError, SamplingParams, Tokenizer};
    use crate::mock::{DraftMock, MockEngine, MockTokenizer};
    use crate::sampling;

    /// An engine behind a log of every id fed to it: `prefill`'s, every
    /// step's `last`, and a pass's `last` and kept tokens but its last.
    struct Fed<E> {
        inner: E,
        fed: Arc<Mutex<Vec<u32>>>,
    }

    impl<E: Engine> Engine for Fed<E> {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.inner.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.fed.lock().expect("fed").extend_from_slice(ids);
            self.inner.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.fed.lock().expect("fed").push(last);
            self.inner.next(last, out)
        }
        fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
            let d = self.inner.advance(last, out)?;
            let mut fed = self.fed.lock().expect("fed");
            fed.push(last);
            fed.extend(&out[..out.len() - 1]);
            Ok(d)
        }
        fn advance_rows(&self) -> usize {
            self.inner.advance_rows()
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.inner.reset()
        }
        fn ctx_max(&self) -> usize {
            self.inner.ctx_max()
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
    }

    /// One greedy generation of at most 8 tokens of `prompt` with `budget`,
    /// and every id the engine was fed. A prompt holding `<think>` twice makes
    /// the mock echo `a`, `<think>`, `a`, `<think>` forever after the span
    /// opens, so only the forced close ever closes the span; one holding
    /// `<think></think>` makes it close the span itself on its first token.
    fn run_on<E: Engine + 'static>(
        inner: E,
        prompt: &str,
        budget: Option<usize>,
    ) -> (Vec<u32>, Outcome) {
        let fed = Arc::new(Mutex::new(Vec::new()));
        let mut slot = Slot::new(Box::new(Fed {
            inner,
            fed: Arc::clone(&fed),
        }));
        let ids = MockTokenizer.encode(prompt);
        let p = GenParams {
            n_predict: 8,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: false,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: budget,
        };
        let mut tim = Timings::default();
        let o = generate(
            &mut slot,
            &sampling::reference_factory(),
            &ids,
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        )
        .expect("a generation");
        let fed = std::mem::take(&mut *fed.lock().expect("fed"));
        (fed, o)
    }

    fn run(prompt: &str, budget: Option<usize>) -> (Vec<u32>, Outcome) {
        run_on(MockEngine::new(4096), prompt, budget)
    }

    /// The mock ids these tests spell: `q`, `a`, the span's open and close.
    const Q: u32 = 6 + 113;
    const A: u32 = 6 + 97;
    const CLOSE: u32 = 5;

    /// A budget of 3 spends on the third generated id and the next id the
    /// engine is fed is the close id, which is taken, counted and fed itself.
    #[test]
    fn the_budget_spends_into_the_forced_close() {
        assert_eq!(MockTokenizer.encode("</think>"), vec![CLOSE]);
        let (fed, o) = run("q<think>aa<think>", Some(3));
        // `a`, `<think>`, `a` spent, the close forced, the eos the model
        // answers after it.
        assert_eq!(o.tokens, vec![A, 4, A, CLOSE, MockTokenizer.eos()]);
        assert_eq!(o.timings.predicted_n, 5, "the close counts like any token");
        assert_eq!(o.stop, StopKind::Eos);
        // The prompt's four prefill ids and its last, the three spent ids,
        // then the close.
        assert_eq!(fed, vec![Q, 4, A, A, 4, A, 4, A, CLOSE]);
    }

    /// A budget of 0 forces the close before any model token: the first taken
    /// id is the close id, and the engine is fed nothing the model answered.
    #[test]
    fn a_budget_of_zero_closes_before_any_model_token() {
        let (fed, o) = run("q<think>aa<think>", Some(0));
        assert_eq!(o.tokens, vec![CLOSE, MockTokenizer.eos()]);
        assert_eq!(fed, vec![Q, 4, A, A, 4, CLOSE]);
    }

    /// A close the model makes itself retires the tracker: the fed ids and
    /// the tokens are a no-budget run's exactly.
    #[test]
    fn a_natural_close_retires_the_tracker() {
        let prompt = "z<think></think>abab<think>";
        let (fed, with) = run(prompt, Some(4));
        let (fed_free, without) = run(prompt, None);
        assert_eq!(with.tokens, without.tokens);
        assert_eq!(fed, fed_free);
        assert!(
            with.tokens.len() > 4,
            "the budget is smaller than the run, or it pins nothing"
        );
    }

    /// A drafting engine stops passing near the limit: a pass of 2 rows needs
    /// the budget to hold 2, so with 4 the loop steps once at 1 left and the
    /// force fires at exactly 4 taken ids. The pass the span's close itself
    /// opens afterwards is the normal flow resumed, its eos ending the run.
    #[test]
    fn a_pass_waits_for_the_budget_to_hold_its_rows() {
        let (fed, o) = run_on(DraftMock::new(4096), "q<think>aa<think>", Some(4));
        assert_eq!(o.tokens, vec![A, 4, A, 4, CLOSE, MockTokenizer.eos()]);
        // The prompt's four prefill ids and its last, the pass's `a` and
        // `<think>`, then the gated step's `a` (a second pass here would feed
        // `a` and `<think>` again), the armed step's `<think>`, the close, and
        // the eos the pass the retired span opens kept.
        assert_eq!(
            fed,
            vec![Q, 4, A, A, 4, A, 4, A, 4, CLOSE, MockTokenizer.eos()]
        );
    }
}
