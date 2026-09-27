//! The generation loop shared by `/completion` and `/v1/chat/completions`.
//!
//! `timings` are the server's wall clock around the engine calls, the way
//! llama-server reports them to bench clients. They are not admissible
//! measurements for this repository: those come only from the lease runners.
//! Like llama-server, the prompt phase ends when the first token's logits are
//! out, and the predicted phase runs from there to the end, so `predicted_ms`
//! spans `predicted_n - 1` decode steps.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde_json::{Value, json};

use crate::engine::{
    CacheNote, Engine, EngineError, Sampler, SamplerFactory, SamplingParams, Saved, StateError,
    Tokenizer,
};
use crate::promptcache::{self, PromptCache};
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
}

impl Timings {
    /// The JSON llama-server emits. A zero count gives NaN per-token fields,
    /// which serialize as `null` exactly as nlohmann writes them.
    pub(crate) fn to_json(&self) -> Value {
        let (pn, dn) = (self.prompt_n as f64, self.predicted_n as f64);
        json!({
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
        })
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

/// The one slot: the engine, the ids its cache holds, one per position, and
/// the host prompt cache of states it held before ([`PromptCache`]).
pub(crate) struct Slot {
    pub engine: Box<dyn Engine>,
    vocab: Arc<dyn Tokenizer>,
    /// Every id the engine has evaluated since its last reset, in order. The
    /// last generated id of a request is not in it: it is never fed back.
    held: Vec<u32>,
    cache: PromptCache,
}

impl Slot {
    pub(crate) fn new(engine: Box<dyn Engine>) -> Slot {
        Slot {
            vocab: engine.tokenizer(),
            cache: PromptCache::new(engine.cache_ram()),
            engine,
            held: Vec::new(),
        }
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
    fn next(&mut self, last: u32, logits: Option<&mut [f32]>) -> Result<u32, EngineError> {
        let held = std::mem::take(&mut self.held);
        let g = self.engine.next(last, logits)?;
        self.held = held;
        self.held.push(last);
        Ok(g)
    }

    /// Positions the cache holds.
    pub(crate) fn held_len(&self) -> usize {
        self.held.len()
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

/// Runs one request on the slot. `ids` is non-empty and shorter than the context.
/// `timings` in the returned outcome are filled even when the sink failed midway
/// (the caller's counters still see the work done). A greedy request never asks the
/// engine for its logits. `tick` sees the timings after every generated token.
pub(crate) fn generate(
    slot: &mut Slot,
    factory: &SamplerFactory,
    ids: &[u32],
    p: &GenParams,
    sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    tick: &mut dyn FnMut(&Timings),
    timings_out: &mut Timings,
) -> Result<Outcome, GenError> {
    let n = ids.len();
    let ctx_max = slot.engine.ctx_max();
    let vocab = Arc::clone(&slot.vocab);
    let mut sampler = (p.sampling.temperature > 0.0).then(|| factory(&p.sampling));
    let stops = vocab.stops();
    let banned: &[u32] = if p.ignore_eos { &stops } else { &[] };
    // A greedy request reads the logits only to step past a banned argmax.
    let mut logits = vec![
        0.0f32;
        if sampler.is_some() || !banned.is_empty() {
            vocab.n_vocab()
        } else {
            0
        }
    ];
    let cache_n = slot.reuse(ids, p.cache_prompt)?;
    let t0 = Instant::now();
    slot.prefill_marked(ids, cache_n, n - 1)?;
    let greedy = slot.next(ids[n - 1], out(&mut logits))?;
    let tim = timings_out;
    *tim = Timings {
        prompt_n: n - cache_n,
        prompt_ms: ms_since(t0),
        n_ctx: ctx_max,
        n_past: n,
        cache_n,
        n_prompt: n,
        ..Timings::default()
    };
    let t1 = Instant::now();
    if p.return_progress {
        sink(Event::Prompt(tim))?;
    }
    let budget = usize::try_from(p.n_predict).unwrap_or(usize::MAX);
    let mut scan = StopScan::new(p.stop.clone());
    let mut dec = vocab.decoder();
    let mut generated: Vec<u32> = Vec::new();
    let mut stop = StopKind::Limit;
    let mut stopping_word = String::new();
    let mut truncated = false;
    if budget > 0 {
        let mut tok = choose(&mut sampler, greedy, &mut logits, &generated, banned);
        loop {
            if banned.contains(&tok) {
                return Err(GenError::Banned(tok));
            }
            generated.push(tok);
            tim.predicted_n = generated.len();
            tim.predicted_ms = ms_since(t1);
            tick(tim);
            if stops.contains(&tok) {
                stop = StopKind::Eos;
                break;
            }
            if let Some(piece) = dec.push(tok) {
                let pushed = scan.push(&piece);
                if !pushed.send.is_empty() {
                    sink(Event::Text(&pushed.send, tim))?;
                }
                if let Some(w) = pushed.stopped {
                    stop = StopKind::Word;
                    stopping_word = w;
                    break;
                }
            }
            if generated.len() >= budget {
                break;
            }
            // Feeding `tok` takes position n + len - 1; the cache holds ctx_max.
            if n + generated.len() > ctx_max {
                truncated = true;
                break;
            }
            let g = slot.next(tok, out(&mut logits))?;
            tim.n_past = n + generated.len();
            tok = choose(&mut sampler, g, &mut logits, &generated, banned);
        }
    }
    if stop != StopKind::Word {
        let tail = dec.flush();
        let pushed = scan.push(&tail);
        let mut rest = pushed.send;
        if let Some(w) = pushed.stopped {
            stop = StopKind::Word;
            stopping_word = w;
        } else {
            rest.push_str(&scan.finish());
        }
        if !rest.is_empty() {
            sink(Event::Text(&rest, tim))?;
        }
    }
    tim.predicted_ms = ms_since(t1);
    Ok(Outcome {
        content: scan.text().to_owned(),
        tokens: generated,
        stop,
        stopping_word,
        truncated,
        timings: tim.clone(),
    })
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
}
