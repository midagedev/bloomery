//! The generation loop shared by `/completion` and `/v1/chat/completions`.
//!
//! `timings` are the server's wall clock around the engine calls, the way
//! llama-server reports them to bench clients. They are not admissible
//! measurements for this repository: those come only from the lease runners.
//! Like llama-server, the prompt phase ends when the first token's logits are
//! out, and the predicted phase runs from there to the end, so `predicted_ms`
//! spans `predicted_n - 1` decode steps. `cache_ms`, which llama-server does
//! not report, is the phase before the prompt's: the prompt cache's work on
//! the slot (its state saved, a cached state put back, the cut), which
//! `prompt_ms` does not count. Nor does it count, on a prompt the engine
//! thread runs a call a round ([`Prompt`]), the other slots' decode rounds
//! between its calls: `prompt_ms`, and so `prompt_per_second`, are the
//! prompt's own engine time alone.
//!
//! An engine that drafts ([`Engine::advance_rows`] past 1) takes every greedy
//! token after the first through [`Engine::advance`]: a pass keeps one token
//! or more, which the loop then takes one at a time as it takes a step's. A
//! pass that would run past the context is not run; the positions left take
//! one step each, so the ids end where a plain run's end. `timings` then carry
//! llama-server's `draft_n` and `draft_n_accepted`: the ids the passes proposed
//! and kept. A request that samples takes its passes through
//! [`Engine::advance_sampled`] on an engine that drafts sampled requests
//! ([`Engine::drafts_sampled`]), its sampler drawing each kept id from the
//! verified rows, and only while its slot runs alone in the round: beside
//! another busy slot, a pass's rows are not pinned to a step's, so it steps.
//! A request that bans an id, or samples on another drafting engine, reads the
//! logits row every token, which a pass does not give, so it takes one
//! [`Engine::next`] a token and carries no draft counts.
//!
//! A request that asks for probabilities ([`GenParams::logprobs`]) reads the
//! row every token too, so it also takes one [`Engine::next`] a token, greedy
//! or not: each row is read before the loop bans an id in it, and each taken
//! id's value goes to the request's log with the text it released
//! ([`crate::api::logprobs`]). Its stream sends one event a token, the text possibly
//! empty, and the end of generation's own; a request that asks for nothing
//! meets one check of its fixed `None` where a row is read and one where a
//! token's text goes out.
//!
//! A request runs as a [`Gen`], one engine call at a time, so the engine
//! thread can make one call of several slots' steps: each running request's
//! next step a row of one [`Engine::step_slots`], its next drafted pass a row
//! of one [`Engine::advance_slots`]. A drafting engine serves one slot unless
//! it keeps its draft's state per slot ([`Engine::slot_drafts`]). Its prompt
//! is one call, cut where the engine asks at its messages, or on an engine
//! that names a prompt quantum ([`Engine::prompt_quantum`]) calls the engine
//! thread runs one a round ([`Prompt`]).

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde_json::{Value, json};

use crate::api::logprobs::{Ask, Collector, RowError};
use crate::engine::{
    CacheNote, Decoder, Drafted, Engine, EngineError, Sampler, SamplerFactory, SamplerRefused,
    SamplingParams, Saved, SlotPass, SlotRow, StateError, Tokenizer,
};
use crate::media::{Held, MediaFeed, MediaSpan, common_prefix, keep_whole_spans};
use crate::promptcache::PromptCache;
use crate::reasoning::{THINK_CLOSE, ThinkEntry, ThinkSplit};
use crate::sampling;
use crate::slotfile::{self, Counting};
use crate::stop::{Pushed, StopScan};

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
    /// `Some` only where a span can open — a prompt that already closed it has
    /// no reasoning to cap and the budget is silently ignored, as llama-server
    /// ignores the flag when thinking is off.
    pub reasoning_budget: Option<usize>,
    /// Where the prompt left the think span, for the budget's tracker: the
    /// ids before a model-opened span spends nothing, and its close ids are
    /// not forced until the span opens.
    pub think_entry: ThinkEntry,
    /// The probabilities the request asked for and the log its generation
    /// writes them to; `None` asks for none. Only the routes that render them
    /// set it.
    pub logprobs: Option<Ask>,
}

/// llama-server's `timings` object, as the server clocked it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Timings {
    /// Prompt tokens evaluated by this request: the prompt less what the
    /// cache kept (`cache_n`).
    pub prompt_n: usize,
    /// The prompt's own engine time: its calls and its last id's step, not
    /// the other slots' decode rounds between the calls of a prompt run a
    /// call a round ([`Prompt`]).
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub n_ctx: usize,
    pub n_past: usize,
    /// Prompt positions kept from the previous request instead of evaluated.
    pub cache_n: usize,
    /// The prompt cache's work before the prompt phase: the slot's state
    /// saved, a cached state put back, the cut (ours; llama-server has no
    /// such field).
    pub cache_ms: f64,
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
    /// The JSON llama-server emits, and `cache_ms`. A zero count gives NaN
    /// per-token fields, which serialize as `null` exactly as nlohmann writes
    /// them. The draft's two counts are there once a pass proposed, as
    /// llama-server's are.
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
            "cache_ms": self.cache_ms,
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
    /// Text safe to stream. Empty between the calls of a prompt run a call a
    /// round ([`Prompt`]): the event a client gone fails on, as a token's
    /// text does, its timings the prompt's so far.
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
    /// The row a generated token was taken from has no distribution (a NaN,
    /// a +inf, no finite logit) for the probabilities the request asked for.
    /// It ends the request, wherever in the generation it is met.
    #[error("the probabilities of generated token {token}: {error}")]
    Logprobs { token: usize, error: RowError },
}

pub(crate) fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// The logits buffer the engine fills, or `None` on a greedy request (empty buffer).
fn out(logits: &mut [f32]) -> Option<&mut [f32]> {
    (!logits.is_empty()).then_some(logits)
}

/// The last `n` ids of `held` a draw's penalties see, in order: an image
/// span's positions are left out, as llama-server skips the null ids a media
/// chunk holds.
fn prompt_tail(held: &Held, n: usize) -> Vec<u32> {
    let mut spans = held.media.iter().rev().peekable();
    let mut tail: Vec<u32> = Vec::new();
    for (at, &id) in held.ids.iter().enumerate().rev() {
        if tail.len() == n {
            break;
        }
        while spans.next_if(|s| s.at > at).is_some() {}
        if spans.peek().is_some_and(|s| at < s.end()) {
            continue;
        }
        tail.push(id);
    }
    tail.reverse();
    tail
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

/// Whether a request takes its tokens after the first through the engine's
/// passes when the engine drafts: one that bans no id and asks for no
/// probabilities (`asks`), greedy ([`Engine::advance`]) or sampling on an
/// engine that drafts sampled requests ([`Engine::advance_sampled`]).
fn takes_passes(
    sampler: Option<&Sampler>,
    banned: &[u32],
    asks: bool,
    engine: &dyn Engine,
) -> bool {
    banned.is_empty() && !asks && (sampler.is_none() || engine.drafts_sampled())
}

/// The engine and its slots: what each slot's cache holds — the ids, one per
/// position, and the image spans among them ([`Held`]) — and the host prompt
/// cache of states the slots held before ([`PromptCache`]), one for every
/// slot. Every method but [`Slot::select`], [`Slot::held_of`],
/// [`Slot::step_slots`] and [`Slot::advance_slots`] acts on the selected slot.
pub(crate) struct Slot {
    pub engine: Box<dyn Engine>,
    vocab: Arc<dyn Tokenizer>,
    /// Every id the selected slot has evaluated since its last reset, in
    /// order, and the image spans among them. The last generated id of a
    /// request is not in it: it is never fed back.
    held: Held,
    /// The other slots' held sequences; the selected slot's entry is empty.
    parked: Vec<Held>,
    /// The slot whose held sequence `held` is.
    cur: usize,
    /// The slot the engine has selected: `None` after a call of several.
    selected: Option<usize>,
    /// Each slot's last request prompt, in ids, the one it fed; 0 when not
    /// known (nothing fed since the slot emptied). A state the slot took back
    /// whole (a cached state, a slot file) counts as all prompt.
    prompts: Vec<usize>,
    /// The selected slot's state while the engine holds another: the parked
    /// state handed back for the request about to start there
    /// ([`Slot::hand_back`]), which that request's [`Slot::reuse`] consumes.
    away: Option<Arc<dyn Saved>>,
    /// The selected slot's state is not on the engine, which holds another
    /// slot's ([`Slot::select_off`]): [`Slot::erase`] drops its ids alone.
    off: bool,
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
            held: Held::default(),
            parked: vec![Held::default(); n],
            cur: 0,
            selected: Some(0),
            prompts: vec![0; n],
            away: None,
            off: false,
        }
    }

    /// Makes `slot` the selected one, on the engine too when it changes.
    ///
    /// # Panics
    ///
    /// When another slot is selected while the selected one's handed-back
    /// state waits for its request ([`Slot::hand_back`]).
    pub(crate) fn select(&mut self, slot: usize) -> Result<(), EngineError> {
        assert!(
            self.away.is_none() || self.cur == slot,
            "slot {} was handed its parked state back and slot {slot} was selected before a \
             request took it",
            self.cur
        );
        self.off = false;
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

    /// [`Slot::select`] of a slot whose state the engine does not hold, for
    /// an action that drops it ([`Slot::erase`]); the next select ends it.
    pub(crate) fn select_off(&mut self, slot: usize) -> Result<(), EngineError> {
        self.select(slot)?;
        self.off = true;
        Ok(())
    }

    /// What `slot`'s cache holds: its ids and their image spans.
    pub(crate) fn held_of(&self, slot: usize) -> &Held {
        if slot == self.cur {
            &self.held
        } else {
            &self.parked[slot]
        }
    }

    /// `state`, the selected slot's parked state of the ids it holds, handed
    /// back for the request about to start on it while the engine holds
    /// another: that request's [`Slot::reuse`] puts it into the prompt cache
    /// as it is where the rule saves it (no copy), and back into the engine
    /// only when the request keeps some of it.
    ///
    /// # Panics
    ///
    /// When `state` holds other than the slot's positions: the park table and
    /// the slot disagree.
    pub(crate) fn hand_back(&mut self, state: Arc<dyn Saved>) {
        assert_eq!(
            state.n_tokens(),
            self.held.len(),
            "slot {}'s parked state holds {} positions, the slot {}",
            self.cur,
            state.n_tokens(),
            self.held.len()
        );
        self.away = Some(state);
    }

    fn held_mut(&mut self, slot: usize) -> &mut Held {
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
        let held: Vec<Held> = rows
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
            let held = self.held_mut(r.slot);
            *held = h;
            held.ids.push(r.last);
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
        let held: Vec<Held> = rows
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
            let held = self.held_mut(r.slot);
            *held = h;
            held.ids.push(r.last);
            held.ids.extend_from_slice(&r.out[..r.out.len() - 1]);
        }
        Ok(())
    }

    /// Brings the cache to the longest prefix of `req` it can keep, leaving at
    /// least the last id for `next`, and returns that length. `want` false
    /// (`cache_prompt: false`) always resets. While an engine call is in flight
    /// `held` is empty, so a failed call leaves no claim about the cache.
    ///
    /// The prefix is the longest the slot and the request share as sequences
    /// ([`common_prefix`]): the same ids and, inside the shared part, the same
    /// images, and it never ends inside an image span.
    ///
    /// With the prompt cache on: a cached state that keeps more of `req` than
    /// the slot replaces the slot's state; the slot's state goes into the
    /// cache first whenever it is replaced, or the request cuts it and does
    /// not carry the slot's last prompt whole. On a server of one slot the
    /// cut saves only when it keeps less than half of the slot, llama-server's
    /// rule (`f_keep < 0.5`); on one of several, the request sat on this slot
    /// for the prefix it shares where another slot may have been free, so any
    /// cut saves and the choice costs the slot's conversation nothing. The
    /// last-prompt condition is ours: a request that carries the last prompt
    /// whole drops only that prompt's reply, which a client that renders the
    /// reply again (a template that leaves the reasoning out) never sends
    /// back; the price is that a regenerated reply's old state is not kept. A
    /// prefix the engine keeps less of than the request shares is noted with
    /// the engine's reason.
    ///
    /// A slot handed its parked state back ([`Slot::hand_back`]) holds it off
    /// the engine: the cache takes that state as it is where the rule saves
    /// it, and the engine takes it back only when the request keeps some of
    /// it. One the request keeps none of leaves the slot holding nothing, so
    /// no prefix it shared is noted.
    fn reuse(&mut self, req: &Held, want: bool) -> Result<usize, EngineError> {
        let away = self.away.take();
        let (mut common, mut ask, mut k) = self.keep_of(req, want, away.as_ref());
        // The pick is taken out before the slot's state is saved: making room
        // for that state may evict it, and the pick then lives on in `picked`.
        let picked = if want && self.cache.enabled() {
            self.cache.best(req, k).map(|p| self.cache.take(p))
        } else {
            None
        };
        let last = self.prompts[self.cur];
        let continues = last > 0 && common >= last;
        let saves_cut = if self.parked.len() > 1 {
            k < self.held.len()
        } else {
            2 * k < self.held.len()
        };
        if self.cache.enabled()
            && !self.held.is_empty()
            && (picked.is_some() || (saves_cut && !continues))
        {
            self.save_held(away.as_ref())?;
        }
        if let Some((entry, state)) = picked {
            let slot_kept = k;
            if let Some((positions, bytes, ms)) = self.load(entry, &state)? {
                (common, ask, k) = self.keep_of(req, want, None);
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
        } else if let Some(state) = away
            && (k == 0 || !self.put_back(&state)?)
        {
            self.held = Held::default();
            (common, ask, k) = (0, 0, 0);
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

    /// The positions `req` shares with the slot, the most of them a request
    /// may keep (all but its last), and what the engine keeps of those: what
    /// it would keep of `away`, the slot's state off the engine, when given,
    /// never ending inside an image span ([`keep_whole_spans`]).
    fn keep_of(
        &self,
        req: &Held,
        want: bool,
        away: Option<&Arc<dyn Saved>>,
    ) -> (usize, usize, usize) {
        let common = if want {
            common_prefix(&self.held, req)
        } else {
            0
        };
        let ask = common.min(req.len() - 1);
        let kept = keep_whole_spans(req, ask, |n| {
            away.map_or_else(|| self.engine.keepable(n), |s| s.keepable(n))
        });
        (common, ask, kept)
    }

    /// The slot's state into the prompt cache, unless a cached state already
    /// keeps all of it: `away`, the slot's state off the engine, as it is,
    /// else the engine's snapshot. A snapshot the engine refuses is noted and
    /// not kept; an engine failure is the request's error.
    fn save_held(&mut self, away: Option<&Arc<dyn Saved>>) -> Result<(), EngineError> {
        if self.cache.covers(&self.held) {
            return Ok(());
        }
        let t = Instant::now();
        let (state, copied) = if let Some(s) = away {
            (Arc::clone(s), false)
        } else {
            match self.engine.snapshot() {
                Ok(s) => (s, true),
                Err(StateError::Engine(e)) => return Err(e),
                Err(e) => {
                    self.engine.note(&CacheNote::Skip {
                        positions: self.held.len(),
                        why: format!("the engine took no snapshot: {e}"),
                    });
                    return Ok(());
                }
            }
        };
        self.keep(self.cur, state, copied, t);
        Ok(())
    }

    /// `state`, `slot`'s state of what it holds, which the slots that take
    /// the engine in turns let go of, into the prompt cache as a save puts
    /// it there ([`Slot::reuse`]), unless a cached state already keeps all of
    /// it; the slot drops it after. `copied`: a snapshot was taken for it,
    /// not a parked state handed over as it is.
    pub(crate) fn offer(&mut self, slot: usize, state: Arc<dyn Saved>, copied: bool) {
        if self.cache.enabled() && !self.cache.covers(self.held_of(slot)) {
            self.keep(slot, state, copied, Instant::now());
        }
    }

    /// `state` of what `slot` holds into the prompt cache, timed from `t`.
    /// A state of other than those positions is noted and not kept.
    fn keep(&mut self, slot: usize, state: Arc<dyn Saved>, copied: bool, t: Instant) {
        let held = self.held_of(slot).clone();
        let positions = held.ids.len();
        if state.n_tokens() != positions {
            self.engine.note(&CacheNote::Skip {
                positions,
                why: format!(
                    "the state holds {} positions, slot {slot} holds {positions}",
                    state.n_tokens()
                ),
            });
            return;
        }
        for note in self.cache.insert(held, state, ms_since(t), copied) {
            self.engine.note(&note);
        }
    }

    /// A cached `state` of `held` into the engine, the slot then holding it;
    /// returns its positions, bytes and the wall time of the resume. A
    /// state the engine refuses leaves the cache, the engine is reset and the
    /// slot holds nothing (`None`); an engine failure is the request's error.
    fn load(
        &mut self,
        held: Held,
        state: &Arc<dyn Saved>,
    ) -> Result<Option<(usize, u64, f64)>, EngineError> {
        self.held = Held::default();
        let t = Instant::now();
        match self.engine.resume(state) {
            Ok(()) => {
                let ms = ms_since(t);
                let got = (held.ids.len(), state.n_bytes(), ms);
                self.prompts[self.cur] = held.ids.len();
                self.held = held;
                Ok(Some(got))
            }
            Err(StateError::Engine(e)) => Err(e),
            Err(e) => {
                self.cache.remove(state);
                self.engine.reset()?;
                self.engine.note(&CacheNote::Skip {
                    positions: held.ids.len(),
                    why: format!("the engine refused to take the state back: {e}"),
                });
                Ok(None)
            }
        }
    }

    /// The slot's own handed-back `state` into the engine; true once it is
    /// there. A state the engine refuses leaves the cache (where the rule
    /// saved it just now), the engine is reset and the slot holds nothing
    /// (false); an engine failure is the request's error.
    fn put_back(&mut self, state: &Arc<dyn Saved>) -> Result<bool, EngineError> {
        let held = std::mem::take(&mut self.held);
        match self.engine.resume(state) {
            Ok(()) => {
                self.held = held;
                Ok(true)
            }
            Err(StateError::Engine(e)) => Err(e),
            Err(e) => {
                self.cache.remove(state);
                self.engine.reset()?;
                self.engine.note(&CacheNote::Skip {
                    positions: held.len(),
                    why: format!("the engine refused to take the slot's parked state back: {e}"),
                });
                Ok(false)
            }
        }
    }

    /// Where the engine cuts the prompt call `ids[from..to]` into calls
    /// ([`Engine::prefill_splits`]) at the messages the prompt opens
    /// ([`Tokenizer::user_start`]): its first and its last inside the range.
    /// An answer not among those marks in order is the engine's error.
    fn marked_cuts(&self, ids: &[u32], from: usize, to: usize) -> Result<Vec<usize>, EngineError> {
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
        Ok(at)
    }

    /// Feeds `ids[from..to]`, cut into calls where the engine asks
    /// ([`Slot::marked_cuts`]). Each of the prompt's images (`media`) rides
    /// the call that holds its span.
    fn prefill_marked(
        &mut self,
        ids: &[u32],
        from: usize,
        to: usize,
        media: &[MediaFeed],
    ) -> Result<(), EngineError> {
        let at = self.marked_cuts(ids, from, to)?;
        let mut first = from;
        for &u in &at {
            self.prefill_range(&ids[first..u], first, media)?;
            first = u;
        }
        self.prefill_range(&ids[first..to], first, media)?;
        self.note_split(from, to, at);
        Ok(())
    }

    /// The note of a prompt call `first..end` the engine cut at its marks
    /// `at`; none when it cut nowhere.
    fn note_split(&self, first: usize, end: usize, at: Vec<usize>) {
        if !at.is_empty() {
            self.engine.note(&CacheNote::Split { first, end, at });
        }
    }

    /// `prefill` of the prompt's positions `at..at + ids.len()` that books
    /// what it fed; a call that holds an image span (of `media`, the prompt's
    /// feeds) is [`Engine::prefill_media`] with the feeds it holds, at its own
    /// positions, and books their spans. A call that would cut a span is the
    /// server's error: the cache and the prompt's last position never lie
    /// inside one.
    fn prefill_range(
        &mut self,
        ids: &[u32],
        at: usize,
        media: &[MediaFeed],
    ) -> Result<(), EngineError> {
        let end = at + ids.len();
        let mut feeds = Vec::new();
        for f in media {
            let s = f.span();
            if s.end() <= at || s.at >= end {
                continue;
            }
            if s.at < at || s.end() > end {
                return Err(EngineError(format!(
                    "the prompt call {at}..{end} cuts the image span {}..{}",
                    s.at,
                    s.end()
                )));
            }
            feeds.push(MediaFeed {
                at: s.at - at,
                ..f.clone()
            });
        }
        let held = std::mem::take(&mut self.held);
        if feeds.is_empty() {
            self.engine.prefill(ids)?;
        } else {
            self.engine.prefill_media(ids, &feeds)?;
        }
        self.held = held;
        self.held.ids.extend_from_slice(ids);
        self.held.media.extend(feeds.iter().map(|f| MediaSpan {
            at: at + f.at,
            ..f.span()
        }));
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
        self.check_pass(out, d)?;
        self.held = held;
        self.held.push(last);
        self.held.ids.extend_from_slice(&out[..out.len() - 1]);
        Ok(d)
    }

    /// [`Engine::advance_sampled`] that books what it fed and checks what it
    /// kept as [`Slot::advance`] does. The engine is lent a copy of
    /// `history` in `lent` (kept for its capacity), which must come back
    /// equal to it, else the engine's error; `sampler` comes back whatever
    /// the outcome.
    pub(crate) fn advance_sampled(
        &mut self,
        last: u32,
        history: &[u32],
        lent: &mut Vec<u32>,
        sampler: Sampler,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Sampler) {
        let held = std::mem::take(&mut self.held);
        out.clear();
        lent.clear();
        lent.extend_from_slice(history);
        let (d, back, sampler) =
            self.engine
                .advance_sampled(last, std::mem::take(lent), sampler, out);
        let d = d.and_then(|d| {
            self.check_pass(out, d)?;
            if back != history {
                return Err(EngineError(format!(
                    "a sampled pass was lent a history of {} ids and handed back another of {}",
                    history.len(),
                    back.len()
                )));
            }
            Ok(d)
        });
        *lent = back;
        if d.is_ok() {
            self.held = held;
            self.held.push(last);
            self.held.ids.extend_from_slice(&out[..out.len() - 1]);
        }
        (d, sampler)
    }

    /// A pass that kept no token, more than the engine's rows, or other than
    /// one more than its draft's accepted ids is the engine's error.
    fn check_pass(&self, out: &[u32], d: Drafted) -> Result<(), EngineError> {
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
        Ok(())
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
    /// vocabulary, a count past the context, an id past the vocabulary, a
    /// span outside the ids) and an engine that does not restore leave the
    /// cache as it was; once the engine has read, any failure resets it and
    /// the slot holds nothing. Returns the positions restored and the file's
    /// bytes.
    pub(crate) fn restore(&mut self, path: &Path) -> Result<(usize, u64), StateError> {
        let mut r = Counting::new(BufReader::new(File::open(path)?));
        let stored = slotfile::read_header(&mut r, self.vocab.n_vocab(), self.engine.ctx_max())?;
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
                self.prompts[self.cur] = 0;
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
            Ok(_) if restored.n_tokens != stored.len() || restored.n_bytes != read => {
                Some(format!(
                    "the engine restored {} positions from {} bytes; the file holds {} positions and \
                 the engine read {read} bytes",
                    restored.n_tokens,
                    restored.n_bytes,
                    stored.len()
                ))
            }
            Ok(_) => None,
        };
        if let Some(m) = mismatch {
            self.prompts[self.cur] = 0;
            self.engine.reset()?;
            return Err(StateError::Format(m));
        }
        let n = stored.len();
        self.held = stored;
        self.prompts[self.cur] = n;
        Ok((n, r.bytes))
    }

    /// Drops the whole cache and the ids it held; returns how many it held.
    pub(crate) fn erase(&mut self) -> Result<usize, EngineError> {
        let held = std::mem::take(&mut self.held);
        self.prompts[self.cur] = 0;
        if !self.off {
            self.engine.reset()?;
        }
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

/// What a prompt's opening kept ([`Gen::open`]): the positions the cache
/// keeps of it, where its prompt call starts, and the wall time of the cache's
/// work (`cache_ms`).
#[derive(Clone, Copy)]
pub(crate) struct Kept {
    pub cache_n: usize,
    cache_ms: f64,
}

/// A prompt the engine thread runs a call a round, between the other busy
/// slots' decode rounds ([`Engine::prompt_quantum`], [`Gen::plan`]): its
/// prompt call cut where the engine asks at its messages
/// ([`Slot::marked_cuts`]), and each of those calls cut again every quantum
/// from its own start. The cuts follow from the prompt and what the cache
/// kept, whatever the load, and each is one the quantum makes bit-neutral, so
/// the request's bits are those of the prompt run in one go.
pub(crate) struct Prompt {
    ids: Vec<u32>,
    /// The prompt's images, each riding the call that holds its span: a
    /// quantum's cut inside a span moves to the span's end.
    media: Vec<MediaFeed>,
    kept: Kept,
    /// The engine's cuts at the messages, which the prompt call's note names
    /// as an uninterrupted call's does: a quantum's cut keeps nothing more.
    marks: Vec<usize>,
    /// Every call's end, ascending; the last is the prompt call's, `n - 1`.
    ends: Vec<usize>,
    /// The calls run.
    ran: usize,
    progress: bool,
}

/// The ends of the calls the prompt call `from..to` runs as: cut at `marks`
/// (ascending, inside the range), and each of those calls every `q` ids from
/// its own start. A cut inside an image span (`media`, the prompt's feeds)
/// moves to the span's end, as a mark's end is one: a span never crosses
/// calls, and the call after it runs on from the span's end.
fn quantum_ends(
    from: usize,
    to: usize,
    marks: &[usize],
    media: &[MediaFeed],
    q: usize,
) -> Vec<usize> {
    // The spans do not overlap, so the end a snap returns lies in none.
    let snap = |u: usize| {
        media
            .iter()
            .map(MediaFeed::span)
            .find(|s| s.at < u && u < s.end())
            .map_or(u, |s| s.end())
    };
    let mut ends = Vec::new();
    let mut start = from;
    for &end in marks.iter().chain([&to]) {
        let mut u = start + q;
        while u < end {
            u = snap(u);
            if u >= end {
                break;
            }
            ends.push(u);
            start = u;
            u = start + q;
        }
        ends.push(end);
        start = end;
    }
    ends
}

/// What a generation needs next ([`Gen::pump`]).
pub(crate) enum Need {
    /// One step from this id: [`Engine::next`], or a row of
    /// [`Engine::step_slots`], whose answer goes to [`Gen::stepped`].
    Step(u32),
    /// One drafted pass from this id ([`Engine::advance`] into
    /// [`Gen::kept_mut`]), whose count goes to [`Gen::advanced`].
    Advance(u32),
    /// One drafted pass of a request that samples from this id
    /// ([`Gen::advance_sampled`]) when its slot is the round's only busy
    /// one; beside another, a [`Need::Step`] from it.
    Sampled(u32),
    /// Nothing more: [`Gen::finish`] ends it.
    Done,
}

/// The think-span budget's state in [`Gen`]: the budget left while the span is
/// open, the span's close detection over the generated text (the output's own
/// split is api-side; this instance only decides when the span closes), and the
/// ids of the close, `at` naming the next of them the loop force-feeds —
/// `close.len()` while the budget holds.
struct Think {
    /// Budget left; the id whose text opens a model-opened span and the id
    /// whose text closes the span do not spend.
    left: usize,
    split: ThinkSplit,
    close: Vec<u32>,
    at: usize,
}

impl Think {
    /// A budget of `left` generated ids over a span the prompt left at
    /// `entry`; at 0 the close is forced from the first taken id inside the
    /// span, which for a model-opened span is the first after its `<think>`.
    fn new(entry: ThinkEntry, left: usize, close: Vec<u32>) -> Think {
        let n = close.len();
        Think {
            left,
            split: ThinkSplit::for_entry(entry),
            close,
            at: if left == 0 { 0 } else { n },
        }
    }

    /// Whether a close id is queued to force: the budget spent with the span
    /// open — a span the model may still open forces nothing.
    fn forcing(&self) -> bool {
        self.split.in_span() && self.at < self.close.len()
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
/// done). A greedy request that asks for no probabilities never asks the
/// engine for its logits, and takes its tokens after the first through the
/// engine's passes. `tick` sees the timings after every generated token.
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
    /// The ids a draw sees, in order: the prompt's tail (its last
    /// `repeat_last_n` ids, as much of the prompt as there is), then the
    /// generated ids. llama-server feeds every prompt id into the sampler
    /// before the first draw; only the window the penalties read is kept, so
    /// a long prompt costs one bounded copy, not a copy a step. Only a
    /// request that draws (which has a sampler) keeps it.
    history: Vec<u32>,
    stop: StopKind,
    stopping_word: String,
    truncated: bool,
    /// A pass's rows; 1 steps. A request with no banned id passes when it is
    /// greedy, or samples on an engine that drafts sampled requests.
    rows: usize,
    /// The last pass's kept tokens, and how many of them the loop has taken.
    kept: Vec<u32>,
    taken: usize,
    /// The copy of the history a sampled pass lends the engine
    /// ([`Slot::advance_sampled`]), kept for its capacity.
    lent: Vec<u32>,
    /// The token the loop takes next; `None` once there is none.
    tok: Option<u32>,
    /// The think-span budget, `None` on a request without one (or whose prompt
    /// closed the span) and once the span closed.
    think: Option<Think>,
    /// What reads each row and writes the request's probabilities, `None` on
    /// a request that asks for none.
    probs: Option<Collector>,
}

/// The sampler a request of `p` draws with, `None` for a greedy one, or the
/// factory's refusal of its parameters. A penalized request at temperature 0
/// takes the sampler's argmax after the penalties, as llama-server's chain
/// does; only an unpenalized one is greedy.
pub(crate) fn sampler_of(
    factory: &SamplerFactory,
    p: &GenParams,
) -> Result<Option<Sampler>, SamplerRefused> {
    (p.sampling.temperature > 0.0 || p.sampling.penalizes())
        .then(|| factory(&p.sampling))
        .transpose()
}

impl Gen {
    /// A generation of `p` for a prompt of `n` ids on `slot`'s engine, drawing
    /// with `sampler` ([`sampler_of`]); nothing runs yet.
    pub(crate) fn new(slot: &Slot, sampler: Option<Sampler>, n: usize, p: &GenParams) -> Gen {
        let stops = slot.vocab.stops();
        let banned = if p.ignore_eos {
            stops.clone()
        } else {
            Vec::new()
        };
        let probs = p.logprobs.as_ref().map(Ask::collector);
        // A greedy request reads the logits only to step past a banned argmax
        // or for the probabilities it asked for.
        let logits = vec![
            0.0f32;
            if sampler.is_some() || !banned.is_empty() || probs.is_some() {
                slot.vocab.n_vocab()
            } else {
                0
            }
        ];
        let rows = if takes_passes(
            sampler.as_ref(),
            &banned,
            probs.is_some(),
            slot.engine.as_ref(),
        ) {
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
            history: Vec::new(),
            stop: StopKind::Limit,
            stopping_word: String::new(),
            truncated: false,
            rows,
            kept: Vec::with_capacity(rows),
            taken: 0,
            lent: Vec::new(),
            tok: None,
            think: p
                .reasoning_budget
                .map(|left| Think::new(p.think_entry, left, slot.vocab.encode(THINK_CLOSE))),
            probs,
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
    /// commits it, so nothing desyncs), else the sampled or greedy choice. A
    /// request that asks for probabilities reads the row first, as the engine
    /// wrote it, before a banned id's logit is set to -inf.
    fn answer(&mut self, g: u32) -> u32 {
        if let Some(c) = self.probs.as_mut() {
            c.read(&self.logits);
        }
        if let Some(id) = self.think.as_mut().and_then(Think::next_forced) {
            return id;
        }
        choose(
            &mut self.sampler,
            g,
            &mut self.logits,
            &self.history,
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
    /// closes the span retires the tracker, the closing id spending nothing,
    /// and a piece that opens a model-opened span spending nothing either —
    /// then the count, only for a span already open, which arms the forced
    /// close at zero.
    fn spend(&mut self, piece: Option<&str>) {
        let Some(t) = self.think.as_mut() else {
            return;
        };
        if let Some(text) = piece {
            let s = t.split.push(text);
            if s.closed {
                self.think = None;
                return;
            }
            if s.opened {
                return;
            }
        }
        if !t.split.in_span() {
            return;
        }
        if t.left > 0 {
            t.left -= 1;
            if t.left == 0 {
                t.at = 0;
            }
        }
    }

    /// The prompt on the selected slot: what the cache keeps of it (`held`),
    /// the rest fed — its images with it ([`Engine::prefill_media`]) — and
    /// the first token's logits read.
    #[cfg(test)]
    pub(crate) fn prompt(
        &mut self,
        slot: &mut Slot,
        held: &Held,
        media: &[MediaFeed],
        p: &GenParams,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<(), GenError> {
        let kept = self.open(slot, held, p)?;
        self.prompt_whole(slot, &held.ids, media, kept, p, sink)
    }

    /// The prompt's opening on the selected slot: the engine told the reply
    /// the request may make, the cache brought to the longest prefix of
    /// `held` it keeps, and a request that draws seeds its sampler's history
    /// with the prompt's tail. No id of the prompt is fed yet.
    pub(crate) fn open(
        &mut self,
        slot: &mut Slot,
        held: &Held,
        p: &GenParams,
    ) -> Result<Kept, GenError> {
        // The history the draws read opens on the prompt's tail, the whole
        // prompt's last `repeat_last_n` ids — the cached part included, as
        // llama-server feeds `prompt.tokens` whole into the sampler.
        if self.sampler.is_some() {
            self.history = prompt_tail(held, p.sampling.repeat_last_n);
        }
        // A request that passes has its first token from the prompt's step:
        // the passes make at most the rest.
        let passes = takes_passes(
            self.sampler.as_ref(),
            &self.banned,
            self.probs.is_some(),
            slot.engine.as_ref(),
        );
        slot.engine.will_reply(match usize::try_from(p.n_predict) {
            Ok(n) if passes => Some(n.saturating_sub(1)),
            Err(_) if passes => None,
            _ => Some(0),
        });
        let t = Instant::now();
        let cache_n = slot.reuse(held, p.cache_prompt)?;
        Ok(Kept {
            cache_n,
            cache_ms: ms_since(t),
        })
    }

    /// The prompt after its opening in one go: `ids[cache_n..n-1]` fed as one
    /// call, cut where the engine asks at its messages, and the last id
    /// stepped.
    pub(crate) fn prompt_whole(
        &mut self,
        slot: &mut Slot,
        ids: &[u32],
        media: &[MediaFeed],
        kept: Kept,
        p: &GenParams,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<(), GenError> {
        let t0 = Instant::now();
        slot.prefill_marked(ids, kept.cache_n, self.n - 1, media)?;
        self.close(slot, ids[self.n - 1], kept, t0, p.return_progress, sink)
    }

    /// The prompt after its opening as calls the engine thread runs one a
    /// round ([`Gen::prompt_call`]), its prompt call cut where the engine asks
    /// at its messages and each of those calls every `q` ids from its start
    /// ([`Prompt`]). No call runs here; from here the timings carry the prompt
    /// so far.
    pub(crate) fn plan(
        &mut self,
        slot: &Slot,
        ids: Vec<u32>,
        media: Vec<MediaFeed>,
        kept: Kept,
        q: NonZeroUsize,
        progress: bool,
    ) -> Result<Prompt, GenError> {
        let t = Instant::now();
        let to = self.n - 1;
        let marks = slot.marked_cuts(&ids, kept.cache_n, to)?;
        let ends = quantum_ends(kept.cache_n, to, &marks, &media, q.get());
        self.tim = Timings {
            prompt_ms: ms_since(t),
            n_ctx: self.ctx_max,
            n_past: kept.cache_n,
            cache_n: kept.cache_n,
            cache_ms: kept.cache_ms,
            n_prompt: self.n,
            ..Timings::default()
        };
        Ok(Prompt {
            ids,
            media,
            kept,
            marks,
            ends,
            ran: 0,
            progress,
        })
    }

    /// The next call of `p` on the selected slot; true while calls remain.
    /// Every call but the first follows an empty text event, which fails as a
    /// token's text does once the client is gone, so a request whose client
    /// left ends between its calls. The last call notes the engine's cuts at
    /// the messages, as the prompt run in one go does, and steps the prompt's
    /// last id.
    pub(crate) fn prompt_call(
        &mut self,
        slot: &mut Slot,
        p: &mut Prompt,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<bool, GenError> {
        if p.ran > 0 {
            sink(Event::Text("", &self.tim))?;
        }
        let from = p.ran.checked_sub(1).map_or(p.kept.cache_n, |i| p.ends[i]);
        let to = p.ends[p.ran];
        let t = Instant::now();
        slot.prefill_range(&p.ids[from..to], from, &p.media)?;
        p.ran += 1;
        self.tim.prompt_ms += ms_since(t);
        self.tim.prompt_n = to - p.kept.cache_n;
        self.tim.n_past = to;
        if p.ran < p.ends.len() {
            return Ok(true);
        }
        let t = Instant::now();
        slot.note_split(p.kept.cache_n, to, std::mem::take(&mut p.marks));
        self.close(slot, p.ids[self.n - 1], p.kept, t, p.progress, sink)?;
        Ok(false)
    }

    /// The prompt's end: its last id stepped and the first token's logits
    /// read, its timings, its event and the first token in hand. Its engine
    /// time is what the timings hold of it (nothing for a prompt run in one
    /// go) and the time since `t`.
    fn close(
        &mut self,
        slot: &mut Slot,
        last: u32,
        kept: Kept,
        t: Instant,
        progress: bool,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<(), GenError> {
        let n = self.n;
        let greedy = slot.next(last, out(&mut self.logits))?;
        slot.prompts[slot.cur] = n;
        self.tim = Timings {
            prompt_n: n - kept.cache_n,
            prompt_ms: self.tim.prompt_ms + ms_since(t),
            n_ctx: self.ctx_max,
            n_past: n,
            cache_n: kept.cache_n,
            cache_ms: kept.cache_ms,
            n_prompt: n,
            ..Timings::default()
        };
        self.t1 = Instant::now();
        if progress {
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
            if self.sampler.is_some() {
                self.history.push(tok);
            }
            self.tim.predicted_n = self.generated.len();
            self.tim.predicted_ms = ms_since(self.t1);
            tick(&self.tim);
            if self.stops.contains(&tok) {
                self.stop = StopKind::Eos;
                break;
            }
            let piece = self.dec.push(tok);
            // The token's text out. A request that asks for probabilities
            // sends an event a token, its value written first, the text
            // possibly empty; any other sends text alone, when there is some.
            if let Some(c) = self.probs.as_mut() {
                let pushed = match &piece {
                    Some(piece) => self.scan.push(piece),
                    None => Pushed {
                        send: String::new(),
                        stopped: None,
                    },
                };
                let token = self.generated.len() - 1;
                c.take(tok, &self.logits, &pushed.send)
                    .map_err(|error| GenError::Logprobs { token, error })?;
                sink(Event::Text(&pushed.send, &self.tim))?;
                if let Some(w) = pushed.stopped {
                    self.stop = StopKind::Word;
                    self.stopping_word = w;
                    break;
                }
            } else if let Some(piece) = &piece {
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
                return Ok(if self.sampler.is_some() {
                    Need::Sampled(tok)
                } else {
                    Need::Advance(tok)
                });
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

    /// A [`Need::Sampled`] pass from `last` on the selected slot
    /// ([`Slot::advance_sampled`]): the request's sampler goes to the engine
    /// and comes back, so the next step or pass draws on from where this one
    /// stopped, and the engine is lent the history a step's draw is given
    /// ([`Gen::answer`]): the prompt's tail then the generated ids. Its
    /// tokens are then in hand as an [`Engine::advance`] pass's, the loop
    /// adding each to the generated ids as it takes it.
    ///
    /// # Panics
    ///
    /// When the request does not sample: [`Gen::pump`] asks for this pass
    /// only of one that does.
    pub(crate) fn advance_sampled(
        &mut self,
        slot: &mut Slot,
        last: u32,
    ) -> Result<(), EngineError> {
        let sampler = self
            .sampler
            .take()
            .expect("a sampled pass of a request that samples");
        let (d, sampler) =
            slot.advance_sampled(last, &self.history, &mut self.lent, sampler, &mut self.kept);
        self.sampler = Some(sampler);
        self.advanced(d?);
        Ok(())
    }

    /// The text still held, and the outcome. On a request that asks for
    /// probabilities, a generation that ended on an end-of-generation id
    /// writes that id's value here, with the held text as what it released,
    /// and sends it in an event of its own.
    pub(crate) fn finish(
        &mut self,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<Outcome, GenError> {
        let mut rest = String::new();
        if self.stop != StopKind::Word {
            let tail = self.dec.flush();
            let pushed = self.scan.push(&tail);
            rest = pushed.send;
            if let Some(w) = pushed.stopped {
                self.stop = StopKind::Word;
                self.stopping_word = w;
            } else {
                rest.push_str(&self.scan.finish());
            }
        }
        // `pump` takes the end-of-generation id without its text out: the one
        // generated id that can still lack its value here.
        let valued = match self.probs.as_mut() {
            Some(c) => match self.generated.get(c.taken()) {
                Some(&eog) => {
                    let token = c.taken();
                    c.take(eog, &self.logits, &rest)
                        .map_err(|error| GenError::Logprobs { token, error })?;
                    true
                }
                None => false,
            },
            None => false,
        };
        if valued || !rest.is_empty() {
            sink(Event::Text(&rest, &self.tim))?;
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
        held: &Held,
        media: &[MediaFeed],
        p: &GenParams,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
        tick: &mut dyn FnMut(&Timings),
    ) -> Result<Outcome, GenError> {
        self.prompt(slot, held, media, p, sink)?;
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
                Need::Sampled(t) => self.advance_sampled(slot, t)?,
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
    let sampler = sampler_of(factory, p).expect("the test's factory takes the request");
    let mut g = Gen::new(slot, sampler, ids.len(), p);
    let r = g.run(slot, &Held::from(ids.to_vec()), &[], p, sink, tick);
    timings_out.clone_from(g.timings());
    r
}

#[cfg(test)]
mod tests {
    use super::{partial_path, prompt_tail};
    use crate::media::{Held, ImageKey, MediaSpan};
    use crate::slotfile;
    use std::path::Path;

    /// The penalties' prompt tail leaves an image span's positions out and
    /// keeps the last `n` of the rest, in order.
    #[test]
    fn prompt_tail_leaves_image_spans_out() {
        let span = |at, len| MediaSpan {
            at,
            len,
            key: ImageKey([0; 32]),
        };
        let held = Held {
            ids: vec![1, 2, 9, 9, 9, 3, 4, 9, 9, 5],
            media: vec![span(2, 3), span(7, 2)],
        };
        assert_eq!(prompt_tail(&held, 64), vec![1, 2, 3, 4, 5]);
        assert_eq!(prompt_tail(&held, 3), vec![3, 4, 5]);
        assert_eq!(prompt_tail(&held, 0), Vec::<u32>::new());
        let plain = Held::from(vec![7, 8, 9]);
        assert_eq!(prompt_tail(&plain, 2), vec![8, 9]);
    }

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
    use std::time::Duration;

    use super::{GenParams, Outcome, Slot, Timings, generate};
    use crate::api::logprobs::Ask;
    use crate::engine::{
        CacheNote, Decoder, Engine, EngineError, SamplingParams, Saved, StateError, Tokenizer,
    };
    use crate::media::Held;
    use crate::mock::{MockEngine, MockTokenizer};
    use crate::reasoning::ThinkEntry;
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

    /// What a [`Probe`] saw: its notes, the length of every prefill call,
    /// and the states it took and took back.
    #[derive(Default)]
    struct Log {
        notes: Vec<CacheNote>,
        calls: Vec<usize>,
        /// What each request told [`Engine::will_reply`].
        replies: Vec<Option<usize>>,
        snapshots: usize,
        resumes: usize,
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
        /// How long each snapshot sleeps.
        snapshot_takes: Duration,
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
        fn new(budget: u64, split: bool, log: &Arc<Mutex<Log>>) -> Probe {
            Probe {
                inner: MockEngine::new(4096),
                tok: Arc::new(MarkTok(MockTokenizer)),
                ctx: Vec::new(),
                calls: Vec::new(),
                budget,
                split,
                off_mark: false,
                refuse: false,
                snapshot_takes: Duration::ZERO,
                log: Arc::clone(log),
            }
        }

        fn slot(budget: u64, split: bool) -> (Slot, Arc<Mutex<Log>>) {
            let log = Arc::new(Mutex::new(Log::default()));
            (Slot::new(Box::new(Probe::new(budget, split, &log))), log)
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
            std::thread::sleep(self.snapshot_takes);
            self.log.lock().expect("the log").snapshots += 1;
            Ok(Arc::new(ProbeSaved {
                ctx: self.ctx.clone(),
                calls: self.calls.clone(),
            }))
        }
        fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
            self.log.lock().expect("the log").resumes += 1;
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
        let o = outcome(slot, ids);
        (o.timings.cache_n, o.tokens)
    }

    /// [`run`]'s request, its whole outcome.
    fn outcome(slot: &mut Slot, ids: &[u32]) -> Outcome {
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
            think_entry: ThinkEntry::Closed,
            logprobs: None,
        };
        let factory = sampling::reference_factory();
        let mut tim = Timings::default();
        generate(
            slot,
            &factory,
            ids,
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        )
        .expect("a generation")
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
            off_mark: true,
            ..Probe::new(0, true, &log)
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
            think_entry: ThinkEntry::Closed,
            logprobs: None,
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
            refuse: true,
            ..Probe::new(1 << 20, false, &log)
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
        assert!(
            !slot.cache.covers(&Held::from(a1.clone())),
            "the refused state is still cached"
        );
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
            think_entry: ThinkEntry::Closed,
            logprobs: None,
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

    /// A greedy request that asks for probabilities takes no pass (each of
    /// its rows is a step's), so it tells the engine none of its reply comes
    /// from one.
    #[test]
    fn an_asking_request_tells_the_engine_it_takes_no_pass() {
        let (mut slot, log) = Probe::slot(0, false);
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
            think_entry: ThinkEntry::Closed,
            logprobs: Some(Ask::new(2, MockTokenizer.n_vocab())),
        };
        let mut tim = Timings::default();
        generate(
            &mut slot,
            &sampling::reference_factory(),
            &enc("<｜User｜>the cat sat on the mat and the "),
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        )
        .expect("a generation");
        assert_eq!(log.lock().expect("log").replies, [Some(0)]);
    }

    /// One request ([`run`]) and what it cost the engine in states: the
    /// snapshots taken and the states taken back.
    fn costed(slot: &mut Slot, log: &Mutex<Log>, ids: &[u32]) -> ((usize, usize), usize, Vec<u32>) {
        let count = || {
            let l = log.lock().expect("log");
            (l.snapshots, l.resumes)
        };
        let before = count();
        let (kept, tokens) = run(slot, ids);
        let after = count();
        ((after.0 - before.0, after.1 - before.1), kept, tokens)
    }

    /// The slot's state goes into the cache when another conversation takes
    /// the slot or a cached state replaces it, and never for a request that
    /// carries the slot's last prompt whole. Each row's (snapshots, resumes):
    /// (i) a first turn, (ii) its next turn, which keeps all of it, (iii)
    /// another conversation, (iv) the first one back from the cache, (v-a) a
    /// short turn of a third, (v-b) its next turn with the reply rendered
    /// again — the reply's first id not the slot's — which keeps less than
    /// half the slot, where llama-server's half rule saves.
    #[test]
    fn a_switch_pays_one_snapshot_and_a_continuation_none() {
        let (mut slot, log) = Probe::slot(1 << 20, false);
        let a1 = enc("<｜User｜>the cat sat on the mat and the ");
        let (i, _, t1) = costed(&mut slot, &log, &a1);
        let mut a2 = a1.clone();
        a2.extend(&t1);
        a2.extend(enc("<｜User｜>and then the "));
        let (ii, _, t2) = costed(&mut slot, &log, &a2);
        let (iii, _, _) = costed(&mut slot, &log, &enc("<｜User｜>xyz uvw xyz "));
        let mut a3 = a2.clone();
        a3.extend(&t2);
        a3.extend(enc("<｜User｜>and so the "));
        let (iv, _, _) = costed(&mut slot, &log, &a3);
        let c1 = enc("aba");
        let (va, _, tc1) = costed(&mut slot, &log, &c1);
        let mut c2 = c1.clone();
        c2.extend(enc("<think></think>ba<｜User｜>ab"));
        assert_ne!(
            Some(&c2[c1.len()]),
            tc1.first(),
            "the fixture: (v-b) parts from the reply at its first id"
        );
        let (vb, kept, _) = costed(&mut slot, &log, &c2);
        let held = c1.len() + tc1.len() - 1;
        assert!(
            2 * kept < held,
            "the fixture: (v-b) keeps {kept} of the {held} positions the slot held, at least half"
        );
        assert_eq!(
            [i, ii, iii, iv, va, vb],
            [(0, 0), (0, 0), (1, 0), (1, 1), (1, 0), (0, 0)],
            "{:?}",
            log.lock().expect("log").notes
        );
    }

    /// The prompt cache's work is the request's `cache_ms`, before its
    /// `prompt_ms`: a request that saves the slot's state reports at least
    /// the time the probe's snapshot sleeps, in its `timings` too.
    #[test]
    fn a_save_is_timed_in_cache_ms() {
        const TAKES: Duration = Duration::from_millis(20);
        let log = Arc::new(Mutex::new(Log::default()));
        let probe = Probe {
            snapshot_takes: TAKES,
            ..Probe::new(1 << 20, false, &log)
        };
        let mut slot = Slot::new(Box::new(probe));
        outcome(&mut slot, &enc("<｜User｜>the cat sat on the mat and the "));
        let t = outcome(&mut slot, &enc("xyz uvw xyz ")).timings;
        assert_eq!(log.lock().expect("log").snapshots, 1, "the second saves");
        assert!(
            t.cache_ms >= TAKES.as_secs_f64() * 1e3,
            "cache_ms {} under the snapshot's {TAKES:?}",
            t.cache_ms
        );
        assert_eq!(t.to_json()["cache_ms"].as_f64(), Some(t.cache_ms));
    }
}

/// A prompt that carries an image, through [`Gen`] on the mock that takes
/// images: what the cache keeps of it and what the engine is fed. The mock's
/// context holds a span as its key's letters and the prompt `q\n` image `\n`
/// generates from the span, so a reply names the image the cache holds.
#[cfg(test)]
mod media_tests {
    use std::sync::{Arc, Mutex};

    use vision::{GridPlan, Patches};

    use super::{Gen, GenParams, Slot, sampler_of};
    use crate::engine::{Engine, EngineError, SamplingParams, Tokenizer};
    use crate::media::{Held, ImageKey, MediaFeed, MediaSpan, Prepared, Prompt};
    use crate::mock::{IMAGE_ID, MediaCall, MediaTokenizer, MockEngine};
    use crate::reasoning::ThinkEntry;
    use crate::sampling;

    /// `before`, an image of `len` positions and key `[key; 32]`, `after`.
    fn prompt(before: &str, len: usize, key: u8, after: &str) -> Prompt {
        let mut ids = MediaTokenizer.encode(before);
        let at = ids.len();
        ids.extend(std::iter::repeat_n(IMAGE_ID, len));
        ids.extend(MediaTokenizer.encode(after));
        let plan = GridPlan {
            n_llm_h: 1,
            n_llm_w: len,
            best_h: 1,
            best_w: len,
        };
        let feed = MediaFeed {
            at,
            key: ImageKey([key; 32]),
            prepared: Arc::new(Prepared {
                span_len: len,
                patches: Patches {
                    plan,
                    n_vit_h: 0,
                    n_vit_w: 0,
                    patch_len: 0,
                    bf16: Vec::new(),
                },
            }),
        };
        Prompt {
            held: Held {
                ids,
                media: vec![feed.span()],
            },
            feeds: vec![feed],
        }
    }

    /// One greedy request of four tokens: what the cache kept, and the tokens.
    fn run(slot: &mut Slot, p: &Prompt) -> Result<(usize, Vec<u32>), String> {
        let params = GenParams {
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
            think_entry: ThinkEntry::Closed,
            logprobs: None,
        };
        let factory = sampling::reference_factory();
        let sampler = sampler_of(&factory, &params).expect("the reference refuses none");
        let mut g = Gen::new(slot, sampler, p.held.len(), &params);
        let o = g
            .run(
                slot,
                &p.held,
                &p.feeds,
                &params,
                &mut |_| Ok(()),
                &mut |_| {},
            )
            .map_err(|e| format!("{e:?}"))?;
        Ok((o.timings.cache_n, o.tokens))
    }

    /// The media mock on a slot of its own, and its log of image calls.
    fn media_slot() -> (Slot, Arc<Mutex<Vec<MediaCall>>>) {
        let (mock, log) = MockEngine::new(4096).with_media();
        (Slot::new(Box::new(mock)), log)
    }

    /// What `p` generates on a fresh slot.
    fn fresh(p: &Prompt) -> Vec<u32> {
        run(&mut media_slot().0, p).expect("a fresh generation").1
    }

    fn calls(log: &Mutex<Vec<MediaCall>>) -> Vec<MediaCall> {
        std::mem::take(&mut *log.lock().expect("the media log"))
    }

    /// A second chat whose image is another of the same size keeps nothing of
    /// the first's span: the ids are equal (every span position carries the
    /// image token), the image is not, so the reuse stops at the span's start
    /// and the span is fed again, and the reply is the one a fresh slot gives,
    /// not the first image's.
    #[test]
    fn another_image_of_one_size_stops_the_reuse_at_the_span() {
        let (mut slot, log) = media_slot();
        let a = prompt("q\n", 3, 1, "\n");
        let b = prompt("q\n", 3, 2, "\n");
        assert_eq!(a.held.ids, b.held.ids, "one size: the same ids");
        let (kept, from_a) = run(&mut slot, &a).expect("a");
        assert_eq!(kept, 0);
        assert_eq!(
            calls(&log),
            [MediaCall {
                ids: a.held.ids[..5].to_vec(),
                spans: a.held.media.clone(),
            }]
        );
        let (kept, from_b) = run(&mut slot, &b).expect("b");
        assert_eq!(kept, 2, "the span's start");
        assert_eq!(
            calls(&log),
            [MediaCall {
                ids: vec![IMAGE_ID; 3],
                spans: vec![MediaSpan {
                    at: 0,
                    len: 3,
                    key: ImageKey([2; 32]),
                }],
            }],
            "the span fed again at the call's own positions"
        );
        assert_eq!(from_b, fresh(&b), "the reply of b alone");
        assert_ne!(from_b, from_a, "the replies name their images");
    }

    /// The same image again keeps the span whole: the reuse passes its end,
    /// no image is fed, and the reply is a fresh slot's.
    #[test]
    fn the_same_image_keeps_the_span() {
        let (mut slot, log) = media_slot();
        let a = prompt("q\n", 3, 1, "\n");
        run(&mut slot, &a).expect("a");
        calls(&log);
        let again = prompt("q\n", 3, 1, "\nq\n");
        let (kept, got) = run(&mut slot, &again).expect("again");
        assert_eq!(kept, 6, "all a shares with it: past the span 2..5");
        assert_eq!(calls(&log), [], "a span kept whole is not fed again");
        assert_eq!(got, fresh(&again));
    }

    /// The media mock that keeps only multiples of four positions.
    struct Fours(MockEngine);

    impl Engine for Fours {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.0.tokenizer()
        }
        fn media_model(&self) -> Option<crate::media::SharedMediaModel> {
            self.0.media_model()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.0.prefill(ids)
        }
        fn prefill_media(&mut self, ids: &[u32], m: &[MediaFeed]) -> Result<(), EngineError> {
            self.0.prefill_media(ids, m)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.0.next(last, out)
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.0.reset()
        }
        fn keepable(&self, n: usize) -> usize {
            self.0.keepable(n) / 4 * 4
        }
        fn cut(&mut self, n: usize) -> Result<(), EngineError> {
            self.0.cut(n)
        }
        fn ctx_max(&self) -> usize {
            self.0.ctx_max()
        }
        fn describe(&self) -> String {
            self.0.describe()
        }
    }

    /// An engine whose keep would end inside a span keeps none of the span:
    /// the keep goes back to the span's start and the engine is asked again,
    /// so the span is fed whole and the reply is a fresh slot's.
    #[test]
    fn a_keep_inside_a_span_goes_back_to_its_start() {
        let (mock, log) = MockEngine::new(4096).with_media();
        let mut slot = Slot::new(Box::new(Fours(mock)));
        let a = prompt("abcde", 5, 1, "\n");
        run(&mut slot, &a).expect("a");
        calls(&log);
        let again = prompt("abcde", 5, 1, "\nq\n");
        let (kept, got) = run(&mut slot, &again).expect("a keep inside the span");
        assert_eq!(
            kept, 4,
            "11 shared, 8 lies inside 5..10, then 5 rounds to 4"
        );
        assert_eq!(
            calls(&log),
            [MediaCall {
                ids: again.held.ids[4..again.held.len() - 1].to_vec(),
                spans: vec![MediaSpan {
                    at: 1,
                    len: 5,
                    key: ImageKey([1; 32]),
                }],
            }]
        );
        assert_eq!(got, fresh(&again));
    }
}

#[cfg(test)]
mod slot_tests {
    use std::sync::Arc;

    use super::Slot;
    use crate::engine::{Engine, EngineError, SlotRow, Tokenizer};
    use crate::media::Held;
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
        let want = Held::from(vec![10]);
        assert_eq!((slot.held_of(0), slot.held_of(1)), (&want, &want));
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
    use crate::reasoning::ThinkEntry;
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
            think_entry: ThinkEntry::of_prompt(prompt),
            logprobs: None,
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

    /// The mock ids the model-opened cases spell: `x` and `y`. A prompt
    /// `x<think>yx` ends in plain text, and the mock's first answer is the
    /// `<think>` its earlier occurrence taught (the prompt's `x` follows with
    /// it): the model opens the span itself.
    const X: u32 = 6 + 120;
    const Y: u32 = 6 + 121;

    /// A span the model opens itself: the opening id spends nothing, no close
    /// id is forced before it, and the budget counts the ids after it until
    /// the forced close retires the tracker.
    #[test]
    fn a_model_opened_span_spends_from_its_open_tag() {
        let (fed, o) = run("x<think>yx", Some(2));
        assert_eq!(o.tokens, vec![4, Y, X, CLOSE, MockTokenizer.eos()]);
        assert_eq!(fed, vec![X, 4, Y, X, 4, Y, X, CLOSE]);
    }

    /// A budget of 0 over a model-opened span forces the close from the first
    /// id inside it — the model's opening `<think>` passes, as the start tag
    /// is where the sampler begins, and the second taken id is the close.
    #[test]
    fn a_budget_of_zero_forces_after_the_model_opens() {
        let (fed, o) = run("x<think>yx", Some(0));
        assert_eq!(o.tokens, vec![4, CLOSE, MockTokenizer.eos()]);
        assert_eq!(fed, vec![X, 4, Y, X, 4, CLOSE]);
    }
}

#[cfg(test)]
mod sampled_tests {
    use std::sync::{Arc, Mutex};

    use super::{GenParams, Slot, Timings, generate};
    use crate::engine::{Drafted, Engine, EngineError, Sampler, SamplingParams, Tokenizer};
    use crate::mock::{DraftMock, MockTokenizer};
    use crate::reasoning::ThinkEntry;
    use crate::sampling;

    /// [`DraftMock`] that logs what each request tells [`Engine::will_reply`].
    struct Replies {
        inner: DraftMock,
        replies: Arc<Mutex<Vec<Option<usize>>>>,
    }

    impl Engine for Replies {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.inner.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.inner.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.inner.next(last, out)
        }
        fn advance_rows(&self) -> usize {
            self.inner.advance_rows()
        }
        fn advance_sampled(
            &mut self,
            last: u32,
            history: Vec<u32>,
            sampler: Sampler,
            out: &mut Vec<u32>,
        ) -> (Result<Drafted, EngineError>, Vec<u32>, Sampler) {
            self.inner.advance_sampled(last, history, sampler, out)
        }
        fn drafts_sampled(&self) -> bool {
            self.inner.drafts_sampled()
        }
        fn will_reply(&mut self, tokens: Option<usize>) {
            self.replies.lock().expect("the replies").push(tokens);
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.inner.reset()
        }
        fn keepable(&self, n: usize) -> usize {
            self.inner.keepable(n)
        }
        fn cut(&mut self, n: usize) -> Result<(), EngineError> {
            self.inner.cut(n)
        }
        fn ctx_max(&self) -> usize {
            self.inner.ctx_max()
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
    }

    /// On an engine that drafts sampled requests, a sampled request tells it
    /// the tokens after its first, as a greedy one does (the passes make at
    /// most those); the same request banning the end of generation, which
    /// steps, tells it none.
    #[test]
    fn a_sampled_request_tells_a_drafting_engine_its_passes() {
        let replies = Arc::new(Mutex::new(Vec::new()));
        let mut slot = Slot::new(Box::new(Replies {
            inner: DraftMock::new(4096),
            replies: Arc::clone(&replies),
        }));
        let ids = MockTokenizer.encode("abacadaeabacada");
        for ignore_eos in [false, true] {
            let p = GenParams {
                n_predict: 8,
                sampling: SamplingParams {
                    temperature: 0.8,
                    seed: 3,
                    ..SamplingParams::default()
                },
                stop: Vec::new(),
                ignore_eos,
                stream: false,
                timings_per_token: false,
                return_progress: false,
                include_usage: false,
                cache_prompt: true,
                reasoning_budget: None,
                think_entry: ThinkEntry::Closed,
                logprobs: None,
            };
            let mut tim = Timings::default();
            generate(
                &mut slot,
                &sampling::reference_factory(),
                &ids,
                &p,
                &mut |_| Ok(()),
                &mut |_| {},
                &mut tim,
            )
            .expect("a generation");
        }
        assert_eq!(*replies.lock().expect("the replies"), [Some(7), Some(0)]);
    }
}
