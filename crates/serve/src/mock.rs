//! A model-free engine for the server's gates: a byte-level vocabulary plus the
//! special strings the V4.1 chat template emits, and a bigram echo for "inference".
//!
//! Vocabulary: ids `0..SPECIALS.len()` are the special strings, then one id per
//! byte. A byte id is `SPECIALS.len() + byte`, so any UTF-8 text round-trips and a
//! multi-byte character spans several tokens (the streaming decoder must hold it).
//!
//! Next token: look up the most recent earlier occurrence of the last token in the
//! whole context; the token that followed it is the prediction (logit 4), every
//! other token that ever followed the last token gets logit 2, the rest -8. A last
//! token never seen before predicts EOS. Deterministic, and seed-sensitive once a
//! sampler with temperature > 0 is in the loop. The prediction reads the whole
//! context, so a stale position a `cut` failed to drop changes what comes out.
//! `cut` keeps any prefix.
//!
//! [`MockEngine::failing_at`] makes the `k`-th `next` of the engine's life an
//! error, for the crash-path gate.
//!
//! [`MockEngine::with_slots`] serves several slots, each a context of its own
//! of `ctx_max` positions, so a slot's ids are what it would give alone; its
//! steps of several slots are [`Engine::step_slots`]'s default.
//! [`MockEngine::with_cache_ram`] gives the server a prompt cache of the
//! mock's states, [`MockEngine::with_shares`] lets a slot take another's
//! prefix, and [`MockEngine::with_prompt_quantum`] lets the server
//! interleave its prompts with the other slots' decode rounds.
//!
//! [`MockEngine::with_media`] takes images: [`MediaTokenizer`]'s vocabulary
//! (one more special, [`IMAGE_SPECIAL`] as [`IMAGE_ID`]), [`MockMedia`]'s
//! placeholder and span (one position a pixel), and a log of every
//! [`Engine::prefill_media`] call ([`MediaCall`]). A span's positions enter the
//! context as letters its image's key picks, so what follows a span depends on
//! the image in it; an image token without its image is the server's error.
//!
//! [`DraftMock`] is the same engine behind a draft of one id: each pass
//! verifies a proposal after `last`, the mock's own next token on two passes of
//! three and another id on the third, and keeps what the target agrees with.
//! [`DraftMock::with_slots`] serves several slots, the pass count — the state
//! its draft keeps — one a slot. It drafts sampled requests too
//! ([`Engine::advance_sampled`], unless [`DraftMock::without_sampled`]): a
//! pass proposes its ids ([`DraftMock::with_sampled_draft`]: how many, and the
//! share of them that are the target's argmax at their row, the rest another
//! id), evaluates `last` and every proposal, reads every row, lets the sampler
//! take ids row by row until one differs from the proposal, and takes the rest
//! back with a `cut`. Its rows read the whole context, so a wrong commit
//! changes every later row.
//!
//! Its saved state is its context: [`MOCK_STATE`], a u32 version, a u64 count
//! and the ids, little-endian.

use std::borrow::Cow;
use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use vision::{GridPlan, MediaModel, Patches, Prepared, Rgb8, VisionError};

use crate::engine::{
    Decoder, DraftProps, Drafted, Engine, EngineError, EngineProps, Flight, Landed, Pending,
    ResidencyReset, Sampler, Saved, SavedState, SharedPrefix, SlotPass, StateError, Tokenizer,
    resume_bytes, snapshot_bytes,
};
use crate::media::{MediaFeed, MediaSpan, SharedMediaModel};
use crate::slotfile;

/// Special strings, in id order. Longest-match wins on encode.
pub const SPECIALS: [&str; 6] = [
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<think>",
    "</think>",
];

const N_SPECIAL: u32 = SPECIALS.len() as u32;

/// The first bytes of the mock's state in a slot file.
const MOCK_STATE: [u8; 4] = *b"MOCK";
const MOCK_STATE_VERSION: u32 = 1;
/// The tag, the version and the count.
const MOCK_STATE_HEAD: u64 = 16;
const PREDICTED: f32 = 4.0;
const FOLLOWER: f32 = 2.0;
const FLOOR: f32 = -8.0;

/// The mock engine. `ctx_max` is chosen by the caller so the gate can hit the
/// context limit cheaply.
pub struct MockEngine {
    /// The selected slot's context.
    ctx: Vec<u32>,
    /// Every slot's context; the selected slot's entry is empty.
    parked: Vec<Vec<u32>>,
    cur: usize,
    ctx_max: usize,
    tok: Arc<dyn Tokenizer>,
    nexts: usize,
    fail_at: Option<usize>,
    /// Counts [`Engine::residency_reset`] calls when the mock has a
    /// residency ([`MockEngine::with_residency`]).
    resets: Option<Arc<AtomicUsize>>,
    /// [`Engine::cache_ram`]: 0, the prompt cache off, unless
    /// [`MockEngine::with_cache_ram`].
    cache_ram: u64,
    /// [`Engine::prompt_quantum`]: `None`, prompts never interleave, unless
    /// [`MockEngine::with_prompt_quantum`].
    prompt_quantum: Option<NonZeroUsize>,
    /// The log of its image calls, when it takes images
    /// ([`MockEngine::with_media`]).
    media: Option<Arc<Mutex<Vec<MediaCall>>>>,
    /// Its saves, when it copies its states off the request path and puts a
    /// prefix back ([`MockEngine::with_async_snapshots`]).
    saves: Option<AsyncSaves>,
    /// Whether a slot takes another slot's prefix ([`MockEngine::with_shares`]).
    shares: bool,
}

/// The mock's copy stream: how long a copy takes, what the engine saw, and
/// each slot's copy that may still run.
struct AsyncSaves {
    copy: Duration,
    stats: Arc<SaveStats>,
    /// Each slot's latest copy, begun through `&self`.
    flights: Mutex<Vec<Option<Arc<MockFlight>>>>,
    /// The pool the copies' extents come from, when it is bounded
    /// ([`MockEngine::with_staging`]).
    pool: Option<Arc<MockPool>>,
}

/// The mock's staging pool: cells of `cell` bytes, a state taking as many
/// next to each other as its bytes need, first fit, and giving them back when
/// it drops.
struct MockPool {
    cell: u64,
    used: Mutex<Vec<bool>>,
}

impl MockPool {
    /// An extent for `bytes`, or what the pool lacks for it.
    fn lend(self: &Arc<Self>, bytes: u64) -> Result<Extent, StateError> {
        let cells = usize::try_from(bytes.div_ceil(self.cell)).expect("cells fit usize");
        let mut used = self.used.lock().expect("the mock's pool");
        let at = (0..used.len().saturating_sub(cells.saturating_sub(1)))
            .find(|&i| used[i..i + cells].iter().all(|u| !u));
        let Some(at) = at else {
            let lent = used.iter().filter(|&&u| u).count() as u64;
            return Err(StateError::NoRoom {
                need: cells as u64 * self.cell,
                held: used.len() as u64 * self.cell,
                lent: lent * self.cell,
            });
        };
        used[at..at + cells].fill(true);
        Ok(Extent {
            pool: Arc::clone(self),
            at,
            cells,
        })
    }
}

/// A run of cells of the mock's pool, lent to one state.
struct Extent {
    pool: Arc<MockPool>,
    at: usize,
    cells: usize,
}

impl Drop for Extent {
    fn drop(&mut self) {
        let mut used = self.pool.used.lock().expect("the mock's pool");
        used[self.at..self.at + self.cells].fill(false);
    }
}

/// The mock's state under [`MockEngine::with_async_snapshots`]: its context,
/// and the extent of the pool it holds when its copy ran in one.
struct MockSaved {
    ctx: Vec<u32>,
    _extent: Option<Extent>,
}

impl Saved for MockSaved {
    fn n_tokens(&self) -> usize {
        self.ctx.len()
    }

    fn n_bytes(&self) -> u64 {
        MOCK_STATE_HEAD + 4 * self.ctx.len() as u64
    }

    fn keepable(&self, n: usize) -> usize {
        n.min(self.ctx.len())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// What a mock of [`MockEngine::with_async_snapshots`] saw.
#[derive(Debug, Default)]
pub struct SaveStats {
    /// Snapshots taken on the request path ([`Engine::snapshot`]).
    pub sync_snapshots: AtomicUsize,
    /// Copies begun ([`Engine::snapshot_begin`]).
    pub begun: AtomicUsize,
    /// Copies the staging pool had no room to begin
    /// ([`MockEngine::with_staging`]).
    pub no_room: AtomicUsize,
    /// Calls that cut, reset or replaced a slot's rows while its copy still
    /// ran: the copy would have read rows another call rewrote.
    pub hazards: AtomicUsize,
    /// Positions [`Engine::resume_prefix`] put back, in all.
    pub resumed: AtomicUsize,
    /// States put back whole ([`Engine::resume`]).
    pub whole_resumes: AtomicUsize,
}

/// A copy that lands `copy` after it begins.
struct MockFlight {
    done_at: Instant,
    copy: Duration,
}

impl MockFlight {
    fn landed(&self) -> Landed {
        Landed {
            copy_ms: self.copy.as_secs_f64() * 1e3,
            calls: 1,
            alloc_ms: 0.0,
        }
    }
}

impl Flight for MockFlight {
    fn poll(&self) -> Result<Option<Landed>, StateError> {
        Ok((Instant::now() >= self.done_at).then(|| self.landed()))
    }

    fn wait(&self) -> Result<Landed, StateError> {
        let left = self.done_at.saturating_duration_since(Instant::now());
        std::thread::sleep(left);
        Ok(self.landed())
    }
}

impl MockEngine {
    /// A mock with room for `ctx_max` positions.
    #[must_use]
    pub fn new(ctx_max: usize) -> Self {
        MockEngine {
            ctx: Vec::new(),
            parked: vec![Vec::new()],
            cur: 0,
            ctx_max,
            tok: Arc::new(MockTokenizer),
            nexts: 0,
            fail_at: None,
            resets: None,
            cache_ram: 0,
            prompt_quantum: None,
            media: None,
            saves: None,
            shares: false,
        }
    }

    /// A mock with a residency: each [`Engine::residency_reset`] adds one to
    /// `resets` and reports a reset that left nothing differing.
    #[must_use]
    pub fn with_residency(ctx_max: usize, resets: Arc<AtomicUsize>) -> Self {
        MockEngine {
            resets: Some(resets),
            ..MockEngine::new(ctx_max)
        }
    }

    /// A mock whose `k`-th call to `next` (counted from 1 over its whole life)
    /// fails with an `EngineError`.
    #[must_use]
    pub fn failing_at(ctx_max: usize, k: usize) -> Self {
        MockEngine {
            fail_at: Some(k),
            ..MockEngine::new(ctx_max)
        }
    }

    /// The same mock serving `n` slots (at least 1).
    ///
    /// # Panics
    ///
    /// When `n` is 0.
    #[must_use]
    pub fn with_slots(self, n: usize) -> Self {
        assert!(n > 0, "a mock of no slots");
        assert!(
            self.saves.is_none(),
            "a mock's slots are set before its async snapshots"
        );
        MockEngine {
            parked: vec![Vec::new(); n],
            ..self
        }
    }

    /// The same mock putting any prefix another slot holds on the selected
    /// slot ([`Engine::share_prefix`]): the ids copied, four bytes an id
    /// counted as mapped.
    #[must_use]
    pub fn with_shares(self) -> Self {
        MockEngine {
            shares: true,
            ..self
        }
    }

    /// The same mock with a prompt cache of `bytes` over its saved states.
    #[must_use]
    pub fn with_cache_ram(self, bytes: u64) -> Self {
        MockEngine {
            cache_ram: bytes,
            ..self
        }
    }

    /// The same mock naming `quantum` as the length its prompt calls are
    /// bit-neutral to cut at ([`Engine::prompt_quantum`]), so the server may
    /// run one of its prompts a call a round between the other busy slots'
    /// decode rounds. The mock's bits are the ids themselves, which any cut
    /// leaves unchanged: it lends the gates the cutting, not the neutrality —
    /// a seat proves its own quantum with a gate of its own.
    ///
    /// # Panics
    ///
    /// When `quantum` is 0: a quantum names a call length.
    #[must_use]
    pub fn with_prompt_quantum(self, quantum: usize) -> Self {
        assert!(quantum > 0, "a prompt quantum of 0 ids");
        MockEngine {
            prompt_quantum: NonZeroUsize::new(quantum),
            ..self
        }
    }

    /// The same mock copying its states on a stream of its own that takes
    /// `copy` a state ([`Engine::snapshots_async`]), and putting back only
    /// the prefix a request keeps ([`Engine::resume_prefix`]); what it saw
    /// is in the returned [`SaveStats`]. A call that cuts, resets or replaces
    /// a slot's rows while the slot's copy still runs counts a hazard.
    #[must_use]
    pub fn with_async_snapshots(self, copy: Duration) -> (Self, Arc<SaveStats>) {
        let stats = Arc::new(SaveStats::default());
        let flights = Mutex::new(vec![None; self.parked.len()]);
        let mock = MockEngine {
            saves: Some(AsyncSaves {
                copy,
                stats: Arc::clone(&stats),
                flights,
                pool: None,
            }),
            ..self
        };
        (mock, stats)
    }

    /// The same mock whose copies take their room from a pool of `cells`
    /// cells of `cell` bytes ([`MockPool`]): a copy with no free run of cells
    /// for its state is refused with [`StateError::NoRoom`], and the cells
    /// come back when the state drops. Only after
    /// [`MockEngine::with_async_snapshots`].
    #[must_use]
    pub fn with_staging(mut self, cell: u64, cells: usize) -> Self {
        let saves = self
            .saves
            .as_mut()
            .expect("a staging pool needs with_async_snapshots first");
        saves.pool = Some(Arc::new(MockPool {
            cell,
            used: Mutex::new(vec![false; cells]),
        }));
        self
    }

    /// Counts a hazard when the selected slot's copy still runs.
    fn rewrites(&self) {
        let Some(saves) = &self.saves else { return };
        let running = saves.flights.lock().expect("the mock's flights")[self.cur]
            .as_ref()
            .is_some_and(|f| Instant::now() < f.done_at);
        if running {
            saves.stats.hazards.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// The same mock taking images (see the module header), and the log of
    /// its [`Engine::prefill_media`] calls.
    #[must_use]
    pub fn with_media(self) -> (Self, Arc<Mutex<Vec<MediaCall>>>) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mock = MockEngine {
            tok: Arc::new(MediaTokenizer),
            media: Some(Arc::clone(&log)),
            ..self
        };
        (mock, log)
    }

    /// The same mock whose tokenizer asks for a BOS ([`BosTokenizer`]), as a
    /// GGUF with `add_bos_token` does.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_bos(self) -> Self {
        MockEngine {
            tok: Arc::new(BosTokenizer),
            ..self
        }
    }

    /// The same mock whose tokenizer names no BOS ([`NoBosTokenizer`]) and adds one or not as
    /// `add_bos` says.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_no_bos(self, add_bos: bool) -> Self {
        MockEngine {
            tok: Arc::new(NoBosTokenizer(add_bos)),
            ..self
        }
    }

    fn push(&mut self, id: u32) -> Result<(), EngineError> {
        if self.ctx.len() >= self.ctx_max {
            return Err(EngineError(format!(
                "mock: position {} is past ctx_max {}",
                self.ctx.len(),
                self.ctx_max
            )));
        }
        self.ctx.push(id);
        Ok(())
    }
}

fn token_bytes(id: u32) -> Vec<u8> {
    match usize::try_from(id).ok().and_then(|i| SPECIALS.get(i)) {
        Some(s) => s.as_bytes().to_vec(),
        None => id
            .checked_sub(N_SPECIAL)
            .and_then(|b| u8::try_from(b).ok())
            .map(|b| vec![b])
            .unwrap_or_default(),
    }
}

/// The ids of `text` read piece by piece between its `mark`s, which a special token cannot span:
/// the mocks encode a byte at a time, so the pieces' ids are the text's. The mocks mark nothing
/// ([`MockTokenizer`]), so a mark in a text is one a test wrote.
fn encode_pieces(tok: &dyn Tokenizer, text: &str, mark: &str) -> Vec<u32> {
    text.split(mark)
        .flat_map(|piece| tok.encode(piece))
        .collect()
}

/// The mock's vocabulary (see the module header).
pub struct MockTokenizer;

impl Tokenizer for MockTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            let special = SPECIALS
                .iter()
                .enumerate()
                .filter(|(_, s)| bytes[i..].starts_with(s.as_bytes()))
                .max_by_key(|(_, s)| s.len());
            match special {
                Some((id, s)) => {
                    out.push(u32::try_from(id).expect("SPECIALS has six entries"));
                    i += s.len();
                }
                None => {
                    out.push(N_SPECIAL + u32::from(bytes[i]));
                    i += 1;
                }
            }
        }
        out
    }

    /// Marks nothing: a spelling typed into a message is its token here, which is how a gate
    /// scripts the mock's greedy reply (the keep-the-prompt gates type `</think>` and the
    /// end-of-sequence string into a user message). A vocabulary that follows the chat path's
    /// rule keeps client text text ([`Tokenizer::guard`]); `api::ctl_tests` holds the rule on the
    /// real tokenizer.
    fn guard<'a>(&self, text: &'a str, _mark: &str) -> Cow<'a, str> {
        Cow::Borrowed(text)
    }

    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
        encode_pieces(self, text, mark)
    }

    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| token_bytes(id)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        Box::new(Utf8Decoder { held: Vec::new() })
    }

    fn bos(&self) -> Option<u32> {
        Some(0)
    }

    fn eos(&self) -> u32 {
        1
    }

    fn add_bos(&self) -> bool {
        false
    }

    fn n_vocab(&self) -> usize {
        SPECIALS.len() + 256
    }
}

impl Engine for MockEngine {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.tok.clone()
    }

    fn media_model(&self) -> Option<SharedMediaModel> {
        self.media.as_ref().map(|_| {
            let model: SharedMediaModel = Arc::new(MockMedia);
            model
        })
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        if self.media.is_some()
            && let Some(at) = ids.iter().position(|&id| id == IMAGE_ID)
        {
            return Err(EngineError(format!(
                "mock: the image token at {at} of a prefill of {} ids came without its image",
                ids.len()
            )));
        }
        ids.iter().try_for_each(|&id| self.push(id))
    }

    /// Each feed's span must lie whole inside `ids`, after the one before it,
    /// every position the image token; the call is logged, and each span's
    /// positions enter the context as its key's letters.
    fn prefill_media(&mut self, ids: &[u32], media: &[MediaFeed]) -> Result<(), EngineError> {
        let Some(log) = self.media.as_ref().map(Arc::clone) else {
            return Err(EngineError(
                "mock: this mock takes no image input".to_owned(),
            ));
        };
        let mut fed = ids.to_vec();
        let mut free = 0;
        for f in media {
            let s = f.span();
            if s.len == 0
                || s.at < free
                || s.end() > ids.len()
                || ids[s.at..s.end()].iter().any(|&id| id != IMAGE_ID)
            {
                return Err(EngineError(format!(
                    "mock: the image span {}..{} of a call of {} ids is empty, overlaps the one \
                     before, passes the call or holds other than the image token",
                    s.at,
                    s.end(),
                    ids.len()
                )));
            }
            for (j, id) in fed[s.at..s.end()].iter_mut().enumerate() {
                *id = N_SPECIAL + u32::from(b'a' + f.key.0[j % 32] % 26);
            }
            free = s.end();
        }
        log.lock().expect("the mock's media log").push(MediaCall {
            ids: ids.to_vec(),
            spans: media.iter().map(MediaFeed::span).collect(),
        });
        fed.iter().try_for_each(|&id| self.push(id))
    }

    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.nexts += 1;
        if self.fail_at == Some(self.nexts) {
            return Err(EngineError(format!(
                "mock: injected failure at next #{} (position {})",
                self.nexts,
                self.ctx.len()
            )));
        }
        self.push(last)?;
        let mut logits = logits_out;
        if let Some(l) = logits.as_deref_mut() {
            l.fill(FLOOR);
        }
        let mut raise = |id: u32, v: f32| {
            if let Some(slot) = logits
                .as_deref_mut()
                .and_then(|l| l.get_mut(usize::try_from(id).ok()?))
            {
                *slot = slot.max(v);
            }
        };
        let n = self.ctx.len();
        let mut predicted = MockTokenizer.eos();
        let mut found = false;
        for j in (0..n - 1).rev() {
            if self.ctx[j] != last {
                continue;
            }
            let follower = self.ctx[j + 1];
            raise(follower, FOLLOWER);
            if !found {
                predicted = follower;
                found = true;
            }
        }
        raise(predicted, PREDICTED);
        Ok(predicted)
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.rewrites();
        self.ctx.clear();
        Ok(())
    }

    fn keepable(&self, n: usize) -> usize {
        n.min(self.ctx.len())
    }

    fn cut(&mut self, n: usize) -> Result<(), EngineError> {
        self.rewrites();
        if n > self.ctx.len() {
            return Err(EngineError(format!(
                "mock: cut to {n} with {} positions held",
                self.ctx.len()
            )));
        }
        self.ctx.truncate(n);
        Ok(())
    }

    fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    fn cache_ram(&self) -> u64 {
        self.cache_ram
    }

    fn prompt_quantum(&self) -> Option<NonZeroUsize> {
        self.prompt_quantum
    }

    fn slots(&self) -> usize {
        self.parked.len()
    }

    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        if slot >= self.parked.len() {
            return Err(EngineError(format!(
                "mock: slot {slot} of {} slots",
                self.parked.len()
            )));
        }
        if slot != self.cur {
            let parked = std::mem::take(&mut self.parked[slot]);
            self.parked[self.cur] = std::mem::replace(&mut self.ctx, parked);
            self.cur = slot;
        }
        Ok(())
    }

    fn share_keepable(&self, from: usize, n: usize) -> usize {
        match self.parked.get(from) {
            Some(held) if self.shares && from != self.cur => n.min(held.len()),
            _ => 0,
        }
    }

    fn share_prefix(&mut self, from: usize, n: usize) -> Result<SharedPrefix, StateError> {
        let kept = self.share_keepable(from, n);
        if kept == 0 {
            return Err(StateError::Format(format!(
                "mock: no prefix of slot {from} to share onto slot {}",
                self.cur
            )));
        }
        self.rewrites();
        self.ctx = self.parked[from][..kept].to_vec();
        Ok(SharedPrefix {
            positions: kept,
            mapped_bytes: 4 * kept as u64,
            copied_bytes: 0,
        })
    }

    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, EngineError> {
        Ok(self.resets.as_ref().map(|r| {
            r.fetch_add(1, Ordering::SeqCst);
            ResidencyReset::default()
        }))
    }

    fn describe(&self) -> String {
        format!("mock position={}", self.ctx.len())
    }

    /// No model file, placement or draft: only the version's `mock` mark.
    fn props_engine(&self) -> EngineProps {
        EngineProps {
            version_note: Some("mock".to_owned()),
            ..EngineProps::default()
        }
    }

    fn snapshot_size(&self) -> Result<Option<u64>, StateError> {
        Ok(Some(MOCK_STATE_HEAD + 4 * self.ctx.len() as u64))
    }

    fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
        let Some(saves) = &self.saves else {
            return snapshot_bytes(self);
        };
        saves.stats.sync_snapshots.fetch_add(1, Ordering::SeqCst);
        // A snapshot on the request path costs the copy's time.
        std::thread::sleep(saves.copy);
        Ok(Arc::new(MockSaved {
            ctx: self.ctx.clone(),
            _extent: None,
        }))
    }

    fn snapshots_async(&self) -> bool {
        self.saves.is_some()
    }

    /// The state as [`Engine::snapshot`] takes it, whose copy lands `copy`
    /// later; until then a cut, a reset or a resume of its slot is a hazard.
    fn snapshot_begin(&self) -> Result<Pending, StateError> {
        let Some(saves) = &self.saves else {
            return Ok(Pending::landed(self.snapshot()?));
        };
        let extent = match &saves.pool {
            Some(pool) => match pool.lend(MOCK_STATE_HEAD + 4 * self.ctx.len() as u64) {
                Ok(extent) => Some(extent),
                Err(e) => {
                    saves.stats.no_room.fetch_add(1, Ordering::SeqCst);
                    return Err(e);
                }
            },
            None => None,
        };
        saves.stats.begun.fetch_add(1, Ordering::SeqCst);
        let flight = Arc::new(MockFlight {
            done_at: Instant::now() + saves.copy,
            copy: saves.copy,
        });
        saves.flights.lock().expect("the mock's flights")[self.cur] = Some(Arc::clone(&flight));
        let state: Arc<dyn Saved> = Arc::new(MockSaved {
            ctx: self.ctx.clone(),
            _extent: extent,
        });
        Ok(Pending::new(state, flight))
    }

    fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
        let Some(saves) = &self.saves else {
            return resume_bytes(self, state);
        };
        saves.stats.whole_resumes.fetch_add(1, Ordering::SeqCst);
        self.resume_prefix(state, usize::MAX).map(|_| ())
    }

    /// The first `n` positions of the state: only those are put back.
    fn resume_prefix(&mut self, state: &Arc<dyn Saved>, n: usize) -> Result<usize, StateError> {
        if self.saves.is_none() {
            self.resume(state)?;
            return Ok(state.n_tokens());
        }
        let Some(saved) = state.as_any().downcast_ref::<MockSaved>() else {
            return Err(StateError::Format(
                "a saved state this engine did not take".to_owned(),
            ));
        };
        self.rewrites();
        let kept = n.min(saved.ctx.len());
        self.ctx.clear();
        self.ctx.extend_from_slice(&saved.ctx[..kept]);
        if let Some(saves) = &self.saves {
            saves.stats.resumed.fetch_add(kept, Ordering::SeqCst);
        }
        Ok(kept)
    }

    fn save_state(&self, out: &mut dyn Write) -> Result<SavedState, StateError> {
        let n = u64::try_from(self.ctx.len()).expect("a context length fits u64");
        out.write_all(&MOCK_STATE)?;
        slotfile::write_u32(out, MOCK_STATE_VERSION)?;
        slotfile::write_u64(out, n)?;
        slotfile::write_ids(out, &self.ctx)?;
        Ok(SavedState {
            n_tokens: self.ctx.len(),
            n_bytes: MOCK_STATE_HEAD + 4 * n,
        })
    }

    /// Reads the whole state before it replaces the context: a refused state
    /// leaves the context as it was.
    fn restore_state(&mut self, input: &mut dyn Read) -> Result<SavedState, StateError> {
        let mut tag = [0u8; 4];
        input.read_exact(&mut tag)?;
        let version = slotfile::read_u32(input)?;
        if tag != MOCK_STATE || version != MOCK_STATE_VERSION {
            return Err(StateError::Format(format!(
                "mock state {tag:02x?} version {version}; the mock reads {MOCK_STATE:02x?} version \
                 {MOCK_STATE_VERSION}"
            )));
        }
        let n = slotfile::read_u64(input)?;
        let n = usize::try_from(n)
            .ok()
            .filter(|&n| n <= self.ctx_max)
            .ok_or_else(|| {
                StateError::Format(format!(
                    "mock state of {n} positions; ctx_max is {}",
                    self.ctx_max
                ))
            })?;
        self.ctx = slotfile::read_ids(input, n, self.tok.n_vocab())?;
        Ok(SavedState {
            n_tokens: n,
            n_bytes: MOCK_STATE_HEAD + 4 * n as u64,
        })
    }
}

/// [`MockTokenizer`] with `add_bos` set, for the paths that prepend a BOS.
#[cfg(test)]
pub(crate) struct BosTokenizer;

#[cfg(test)]
impl Tokenizer for BosTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        MockTokenizer.encode(text)
    }

    fn guard<'a>(&self, text: &'a str, mark: &str) -> Cow<'a, str> {
        MockTokenizer.guard(text, mark)
    }

    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
        MockTokenizer.encode_marked(text, mark)
    }

    fn decode(&self, ids: &[u32]) -> String {
        MockTokenizer.decode(ids)
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        MockTokenizer.decoder()
    }

    fn bos(&self) -> Option<u32> {
        MockTokenizer.bos()
    }

    fn eos(&self) -> u32 {
        MockTokenizer.eos()
    }

    fn add_bos(&self) -> bool {
        true
    }

    fn n_vocab(&self) -> usize {
        MockTokenizer.n_vocab()
    }
}

/// [`MockTokenizer`] with no BOS id: `add_bos` as the field says, so a vocabulary that adds none
/// and one that asks for a BOS it does not name can both be served.
#[cfg(test)]
pub(crate) struct NoBosTokenizer(pub(crate) bool);

#[cfg(test)]
impl Tokenizer for NoBosTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        MockTokenizer.encode(text)
    }

    fn guard<'a>(&self, text: &'a str, mark: &str) -> Cow<'a, str> {
        MockTokenizer.guard(text, mark)
    }

    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
        MockTokenizer.encode_marked(text, mark)
    }

    fn decode(&self, ids: &[u32]) -> String {
        MockTokenizer.decode(ids)
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        MockTokenizer.decoder()
    }

    fn bos(&self) -> Option<u32> {
        None
    }

    fn eos(&self) -> u32 {
        MockTokenizer.eos()
    }

    fn add_bos(&self) -> bool {
        self.0
    }

    fn n_vocab(&self) -> usize {
        MockTokenizer.n_vocab()
    }
}

/// [`MockTokenizer`] with a BOS and an EOS around text tokenized with its
/// special tokens, as an embedding model's vocabulary puts them.
#[cfg(test)]
struct EmbedTokenizer;

#[cfg(test)]
impl Tokenizer for EmbedTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        MockTokenizer.encode(text)
    }

    fn guard<'a>(&self, text: &'a str, mark: &str) -> Cow<'a, str> {
        MockTokenizer.guard(text, mark)
    }

    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
        MockTokenizer.encode_marked(text, mark)
    }

    fn decode(&self, ids: &[u32]) -> String {
        MockTokenizer.decode(ids)
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        MockTokenizer.decoder()
    }

    fn bos(&self) -> Option<u32> {
        MockTokenizer.bos()
    }

    fn eos(&self) -> u32 {
        MockTokenizer.eos()
    }

    fn add_bos(&self) -> bool {
        true
    }

    fn add_eos(&self) -> bool {
        true
    }

    fn n_vocab(&self) -> usize {
        MockTokenizer.n_vocab()
    }
}

/// How [`MockEmbed`] fails a pass.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum MockFail {
    Error,
    Panic,
    Short,
    Nan,
}

/// The ids of each pass a [`MockEmbed`] ran, in order.
#[cfg(test)]
pub(crate) type Passes = Arc<Mutex<Vec<Vec<u32>>>>;

/// An embedding model for the embeddings routes' tests: [`EmbedTokenizer`]'s
/// vocabulary and a four-value fixture (`[n_ids, 1, 0, -n_ids]`) cut to two
/// or four. Each pass's ids are logged in `calls`; `fail` returns a defect
/// on the pass whose ids hold its id.
#[cfg(test)]
pub(crate) struct MockEmbed {
    pub(crate) prompts: std::collections::BTreeMap<String, String>,
    pub(crate) ctx: usize,
    pub(crate) fail: Option<(u32, MockFail)>,
    pub(crate) calls: Passes,
}

#[cfg(test)]
impl MockEmbed {
    pub(crate) fn new(ctx: usize) -> MockEmbed {
        MockEmbed {
            prompts: std::collections::BTreeMap::default(),
            ctx,
            fail: None,
            calls: Arc::default(),
        }
    }
}

#[cfg(test)]
impl crate::embed::Embed for MockEmbed {
    fn prompts(&self) -> std::collections::BTreeMap<String, String> {
        self.prompts.clone()
    }

    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::new(EmbedTokenizer)
    }

    fn name(&self) -> String {
        "mock-embd".to_owned()
    }

    fn path(&self) -> String {
        "/models/mock-embd.gguf".to_owned()
    }

    fn ctx(&self) -> usize {
        self.ctx
    }

    fn dims(&self) -> Vec<usize> {
        vec![2, 4]
    }

    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, crate::embed::EmbedError> {
        self.calls.lock().expect("calls").push(ids.to_vec());
        match self.fail {
            Some((id, MockFail::Error)) if ids.contains(&id) => {
                return Err(crate::embed::EmbedError("the mock failed".into()));
            }
            Some((id, MockFail::Panic)) if ids.contains(&id) => panic!("the mock panicked"),
            Some((id, MockFail::Short)) if ids.contains(&id) => return Ok(vec![1.0; 3]),
            Some((id, MockFail::Nan)) if ids.contains(&id) => return Ok(vec![f32::NAN; 4]),
            _ => {}
        }
        let n = ids.len() as f32;
        Ok(vec![n, 1.0, 0.0, -n])
    }
}

/// The media mock's placeholder text: one more special of [`MediaTokenizer`].
pub const IMAGE_SPECIAL: &str = "<｜image｜>";

/// The id [`IMAGE_SPECIAL`] encodes to, the one past [`MockTokenizer`]'s.
pub const IMAGE_ID: u32 = N_SPECIAL + 256;

/// [`MockTokenizer`]'s vocabulary and [`IMAGE_SPECIAL`] as [`IMAGE_ID`]: the
/// media mock's placeholder is one id, as a real model's is.
pub struct MediaTokenizer;

impl Tokenizer for MediaTokenizer {
    /// No special contains the placeholder or overlaps it, so the text
    /// between placeholders encodes as [`MockTokenizer`] encodes it.
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len());
        for (i, piece) in text.split(IMAGE_SPECIAL).enumerate() {
            if i > 0 {
                out.push(IMAGE_ID);
            }
            out.extend(MockTokenizer.encode(piece));
        }
        out
    }

    fn guard<'a>(&self, text: &'a str, _mark: &str) -> Cow<'a, str> {
        Cow::Borrowed(text)
    }

    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
        encode_pieces(self, text, mark)
    }

    fn decode(&self, ids: &[u32]) -> String {
        let pieces: Vec<String> = ids
            .split(|&id| id == IMAGE_ID)
            .map(|run| MockTokenizer.decode(run))
            .collect();
        pieces.join(IMAGE_SPECIAL)
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        Box::new(MediaDecoder(MockTokenizer.decoder()))
    }

    fn bos(&self) -> Option<u32> {
        MockTokenizer.bos()
    }

    fn eos(&self) -> u32 {
        MockTokenizer.eos()
    }

    fn add_bos(&self) -> bool {
        MockTokenizer.add_bos()
    }

    fn n_vocab(&self) -> usize {
        MockTokenizer.n_vocab() + 1
    }
}

/// [`MockTokenizer`]'s streaming decoder, [`IMAGE_ID`] its placeholder text
/// after the bytes held before it.
struct MediaDecoder(Box<dyn Decoder>);

impl Decoder for MediaDecoder {
    fn push(&mut self, id: u32) -> Option<String> {
        if id != IMAGE_ID {
            return self.0.push(id);
        }
        let mut text = self.0.flush();
        text.push_str(IMAGE_SPECIAL);
        Some(text)
    }

    fn flush(&mut self) -> String {
        self.0.flush()
    }
}

/// The media mock's model: [`IMAGE_SPECIAL`] as [`IMAGE_ID`], parts joined by
/// the default separator, and one span position a pixel, so two images of one
/// size take spans of one length and differ only in their keys.
pub struct MockMedia;

impl MediaModel for MockMedia {
    fn image_placeholder(&self, _index: usize) -> Cow<'_, str> {
        Cow::Borrowed(IMAGE_SPECIAL)
    }

    fn image_marker(&self) -> &str {
        IMAGE_SPECIAL
    }

    fn image_token(&self) -> Option<u32> {
        Some(IMAGE_ID)
    }

    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
        let plan = GridPlan {
            n_llm_h: image.height,
            n_llm_w: image.width,
            best_h: image.height,
            best_w: image.width,
        };
        Ok(Prepared::of_grid(
            image.width * image.height,
            Patches {
                plan,
                n_vit_h: 0,
                n_vit_w: 0,
                patch_len: 0,
                bf16: Vec::new(),
            },
        ))
    }
}

/// One [`Engine::prefill_media`] call of the media mock: the ids it was handed
/// and its images' spans, at the call's own positions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaCall {
    pub ids: Vec<u32>,
    pub spans: Vec<MediaSpan>,
}

/// [`MockEngine`] with a draft of one id a pass (see the module header). Its
/// passes keep the target's argmax, so its greedy ids are the plain mock's.
/// [`DraftMock::with_slots`] serves several, the pass count that decides each
/// pass's proposal its own a slot ([`Engine::slot_drafts`]).
pub struct DraftMock {
    inner: MockEngine,
    /// One pass counter a slot, the selected one's read by each pass.
    passes: Vec<usize>,
    /// [`Engine::drafts_sampled`].
    sampled: bool,
    /// The ids a sampled pass proposes.
    n_draft: usize,
    /// Of every `of` proposals of a slot's sampled passes, the first `agree`
    /// are the target's argmax at their row, the rest another id.
    agree: usize,
    of: usize,
    /// One count of sampled proposals a slot.
    proposals: Vec<usize>,
    /// Every step and pass the server asked for, in order, when
    /// [`DraftMock::logged`].
    log: Option<Arc<Mutex<Vec<MockCall>>>>,
    /// Blocks every `prefill` until it opens, when [`DraftMock::holding`].
    hold: Option<Arc<Hold>>,
}

/// A hold on a mock's prompt calls: until it opens, every `prefill` blocks,
/// so a test queues requests behind the one the engine thread is in.
#[derive(Default)]
pub struct Hold {
    /// The prompt calls that reached it, and whether it is open.
    state: Mutex<(usize, bool)>,
    cv: Condvar,
}

impl Hold {
    fn enter(&self) {
        let mut g = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.0 += 1;
        self.cv.notify_all();
        while !g.1 {
            g = self
                .cv
                .wait(g)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// Whether `n` prompt calls reached the hold within `bound`.
    #[must_use]
    pub fn reached(&self, n: usize, bound: std::time::Duration) -> bool {
        let g = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (g, _) = self
            .cv
            .wait_timeout_while(g, bound, |s| s.0 < n)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.0 >= n
    }

    /// Lets every prompt call through, now and later.
    pub fn open(&self) {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .1 = true;
        self.cv.notify_all();
    }
}

/// A step or pass a [`DraftMock`] was asked for ([`DraftMock::logged`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MockCall {
    /// [`Engine::next`].
    Next,
    /// [`Engine::advance`].
    Advance,
    /// [`Engine::advance_sampled`].
    Sampled,
    /// [`Engine::advance_slots`], with its row count; each row also logs the
    /// [`MockCall::Advance`] it runs.
    AdvanceSlots(usize),
}

impl DraftMock {
    /// A drafting mock with room for `ctx_max` positions. It drafts sampled
    /// requests, one id a pass, two proposals of three the target's argmax.
    #[must_use]
    pub fn new(ctx_max: usize) -> Self {
        DraftMock {
            inner: MockEngine::new(ctx_max),
            passes: vec![0],
            sampled: true,
            n_draft: 1,
            agree: 2,
            of: 3,
            proposals: vec![0],
            log: None,
            hold: None,
        }
    }

    /// The same drafting mock serving `n` slots (at least 1), each a context
    /// and a pass count of its own.
    ///
    /// # Panics
    ///
    /// When `n` is 0.
    #[must_use]
    pub fn with_slots(self, n: usize) -> Self {
        assert!(n > 0, "a mock of no slots");
        DraftMock {
            inner: self.inner.with_slots(n),
            passes: vec![0; n],
            proposals: vec![0; n],
            ..self
        }
    }

    /// The same drafting mock declining sampled requests: they step.
    #[must_use]
    pub fn without_sampled(self) -> Self {
        DraftMock {
            sampled: false,
            ..self
        }
    }

    /// The same drafting mock whose sampled passes propose `n` ids, `agree`
    /// of every `of` of them the target's argmax at their row.
    ///
    /// # Panics
    ///
    /// When `n` or `of` is 0, or `agree` is past `of`.
    #[must_use]
    pub fn with_sampled_draft(self, n: usize, agree: usize, of: usize) -> Self {
        assert!(
            n > 0 && of > 0 && agree <= of,
            "a sampled draft of {n} ids agreeing {agree} of {of}"
        );
        DraftMock {
            n_draft: n,
            agree,
            of,
            ..self
        }
    }

    /// The same drafting mock writing every step and pass it is asked for
    /// into `log`.
    #[must_use]
    pub fn logged(self, log: Arc<Mutex<Vec<MockCall>>>) -> Self {
        DraftMock {
            log: Some(log),
            ..self
        }
    }

    /// The same drafting mock whose every `prefill` waits for `hold` to open.
    #[must_use]
    pub fn holding(self, hold: Arc<Hold>) -> Self {
        DraftMock {
            hold: Some(hold),
            ..self
        }
    }

    fn record(&self, call: MockCall) {
        if let Some(log) = &self.log {
            log.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(call);
        }
    }

    /// The sampled pass ([`Engine::advance_sampled`]'s rule): the proposals
    /// and the verify of `last` and each, every row read; the sampler's ids
    /// row by row, `history` holding this pass's ids too while it draws; the
    /// rows past the last kept id taken back.
    fn pass_sampled(
        &mut self,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut Sampler,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, EngineError> {
        if !self.sampled {
            return Err(EngineError(
                "mock: this draft does not draft a sampled request".to_owned(),
            ));
        }
        let slot = self.inner.cur;
        let at = self.inner.ctx.len();
        let n_vocab = MockTokenizer.n_vocab();
        let vocab = u32::try_from(n_vocab).expect("the mock's vocabulary fits u32");
        let mut rows = vec![0.0f32; (self.n_draft + 1) * n_vocab];
        let mut draft = Vec::with_capacity(self.n_draft);
        let mut fed = last;
        for (i, row) in rows.chunks_exact_mut(n_vocab).enumerate() {
            let argmax = self.inner.next(fed, Some(row))?;
            if i == self.n_draft {
                break;
            }
            let p = self.proposals[slot];
            self.proposals[slot] += 1;
            fed = if p % self.of < self.agree {
                argmax
            } else {
                (argmax + 1) % vocab
            };
            draft.push(fed);
        }
        let n = history.len();
        for (i, row) in rows.chunks_exact(n_vocab).enumerate() {
            let id = sampler(row, history);
            history.push(id);
            out.push(id);
            if draft.get(i) != Some(&id) {
                break;
            }
        }
        history.truncate(n);
        self.inner.cut(at + out.len())?;
        Ok(Drafted {
            proposed: self.n_draft,
            accepted: out.len() - 1,
        })
    }
}

impl Engine for DraftMock {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.inner.tokenizer()
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        if let Some(hold) = &self.hold {
            hold.enter();
        }
        self.inner.prefill(ids)
    }

    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.record(MockCall::Next);
        self.inner.next(last, logits_out)
    }

    /// Row 0 runs `last`; a proposal the target's argmax agrees with runs row
    /// 1 on it, and both rows' argmax are kept. The selected slot's pass
    /// counter decides the proposal, so a slot's passes are its own.
    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        self.record(MockCall::Advance);
        let slot = self.inner.cur;
        self.passes[slot] += 1;
        let first = self.inner.next(last, None)?;
        let n_vocab =
            u32::try_from(MockTokenizer.n_vocab()).expect("the mock's vocabulary fits u32");
        let proposal = if self.passes[slot].is_multiple_of(3) {
            (first + 1) % n_vocab
        } else {
            first
        };
        out.push(first);
        if proposal != first {
            return Ok(Drafted {
                proposed: 1,
                accepted: 0,
            });
        }
        out.push(self.inner.next(proposal, None)?);
        Ok(Drafted {
            proposed: 1,
            accepted: 1,
        })
    }

    /// The greedy pass's two rows, or a sampled pass's `last` and its
    /// proposals when they are more.
    fn advance_rows(&self) -> usize {
        2.max(self.n_draft + 1)
    }

    /// The inner mock keeps every position it holds, and a pass leaves it
    /// holding every id the pass kept but the last.
    fn cuts_into_passes(&self) -> bool {
        true
    }

    fn advance_sampled(
        &mut self,
        last: u32,
        mut history: Vec<u32>,
        mut sampler: Sampler,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Vec<u32>, Sampler) {
        self.record(MockCall::Sampled);
        let d = self.pass_sampled(last, &mut history, &mut sampler, out);
        (d, history, sampler)
    }

    fn drafts_sampled(&self) -> bool {
        self.sampled
    }

    fn slots(&self) -> usize {
        self.inner.slots()
    }

    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        self.inner.select_slot(slot)
    }

    fn slot_drafts(&self) -> bool {
        true
    }

    /// The default's select and pass a row (`advance`, or `advance_sampled`
    /// for a row that draws), logged as one call of its row count.
    fn advance_slots(&mut self, rows: &mut [SlotPass<'_>]) -> Result<(), EngineError> {
        self.record(MockCall::AdvanceSlots(rows.len()));
        for row in rows {
            self.select_slot(row.slot)?;
            row.drafted = crate::engine::advance_row(self, row)?;
        }
        Ok(())
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

    fn props_engine(&self) -> EngineProps {
        EngineProps {
            draft: Some(DraftProps {
                model: "mock".to_owned(),
                n_max: u64::try_from(self.n_draft).ok(),
                kind: Some("mock".to_owned()),
                ..DraftProps::default()
            }),
            ..self.inner.props_engine()
        }
    }
}

/// An engine that answers every prompt with the same text, then EOS: the model
/// output a parser gate scripts. Tokens are [`MockTokenizer`]'s, so the special
/// strings are single ids and everything else goes byte by byte;
/// [`ScriptedEngine::with_stops`] gives it a vocabulary with other stop ids.
///
/// The script is served as the ids the engine is fed, not as a count of its
/// rows: a prefill that continues the script (a jump arm feeding the fixed
/// ids of a program) advances it — its first id the one the last step
/// answered, fed for the first time — where the prompt, which spells none of
/// it, leaves it alone.
pub struct ScriptedEngine {
    script: Vec<u32>,
    /// The script's ids consumed so far: the ids fed since the generation
    /// began, the jump arm's calls among them.
    at: usize,
    pos: usize,
    /// Whether a generation's first row has been asked for: prefills before
    /// it are the prompt, ones after it a call the jump arm feeds.
    started: bool,
    /// The ids one pass keeps ([`ScriptedEngine::drafting`]); 1 without a
    /// draft.
    rows: usize,
    ctx_max: usize,
    tok: Arc<dyn Tokenizer>,
}

impl ScriptedEngine {
    /// An engine whose every generation is `text` followed by EOS.
    #[must_use]
    pub fn new(ctx_max: usize, text: &str) -> Self {
        ScriptedEngine::from_ids(ctx_max, MockTokenizer.encode(text))
    }

    /// An engine whose every generation is the ids `script` followed by EOS.
    #[must_use]
    pub fn from_ids(ctx_max: usize, script: Vec<u32>) -> Self {
        ScriptedEngine {
            script,
            at: 0,
            pos: 0,
            started: false,
            rows: 1,
            ctx_max,
            tok: Arc::new(MockTokenizer),
        }
    }

    /// The same engine drafting: a pass keeps the script's next `rows` ids,
    /// its draft always right, up to the end of generation. It grants no
    /// cut, so it does not cut into its passes.
    #[must_use]
    pub fn drafting(self, rows: usize) -> Self {
        ScriptedEngine { rows, ..self }
    }

    /// Consumes `ids` as the script's next ones, stopping at the first the
    /// script does not hold: what the engine is fed is what its script has
    /// written.
    fn consume(&mut self, ids: &[u32]) {
        for &id in ids {
            if self.script.get(self.at) != Some(&id) {
                break;
            }
            self.at += 1;
        }
    }

    /// The same engine over a vocabulary whose stop ids are `stops`, the
    /// first of them its EOS (see `StopsTokenizer`).
    ///
    /// # Panics
    ///
    /// When `stops` is empty.
    #[must_use]
    pub fn with_stops(self, stops: &[u32]) -> Self {
        ScriptedEngine {
            tok: Arc::new(StopsTokenizer::new(stops)),
            ..self
        }
    }
}

/// [`MockTokenizer`] with its own stop ids. Its vocabulary reaches past the
/// largest of them; an id past the mock's own decodes to nothing, as a
/// control token does.
pub(crate) struct StopsTokenizer {
    stops: Vec<u32>,
}

impl StopsTokenizer {
    /// # Panics
    ///
    /// When `stops` is empty: a vocabulary names at least its EOS.
    #[must_use]
    pub(crate) fn new(stops: &[u32]) -> Self {
        assert!(!stops.is_empty(), "StopsTokenizer: no stop id");
        StopsTokenizer {
            stops: stops.to_vec(),
        }
    }
}

impl Tokenizer for StopsTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        MockTokenizer.encode(text)
    }

    fn guard<'a>(&self, text: &'a str, mark: &str) -> Cow<'a, str> {
        MockTokenizer.guard(text, mark)
    }

    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
        MockTokenizer.encode_marked(text, mark)
    }

    fn decode(&self, ids: &[u32]) -> String {
        MockTokenizer.decode(ids)
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        MockTokenizer.decoder()
    }

    fn bos(&self) -> Option<u32> {
        MockTokenizer.bos()
    }

    fn eos(&self) -> u32 {
        self.stops[0]
    }

    fn stops(&self) -> Vec<u32> {
        self.stops.clone()
    }

    fn add_bos(&self) -> bool {
        false
    }

    fn n_vocab(&self) -> usize {
        let past = self.stops.iter().max().map_or(0, |&m| m as usize + 1);
        MockTokenizer.n_vocab().max(past)
    }
}

impl Engine for ScriptedEngine {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::clone(&self.tok)
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.pos += ids.len();
        if self.started {
            self.consume(ids);
        }
        Ok(())
    }

    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        if self.pos >= self.ctx_max {
            return Err(EngineError(format!(
                "scripted: position {} is past ctx_max {}",
                self.pos, self.ctx_max
            )));
        }
        self.pos += 1;
        if self.started {
            self.consume(std::slice::from_ref(&last));
        } else {
            self.started = true;
        }
        let id = self
            .script
            .get(self.at)
            .copied()
            .unwrap_or_else(|| self.tok.eos());
        if let Some(l) = logits_out {
            l.fill(FLOOR);
            if let Some(slot) = usize::try_from(id).ok().and_then(|i| l.get_mut(i)) {
                *slot = PREDICTED;
            }
        }
        Ok(id)
    }

    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        let mut id = self.next(last, None)?;
        out.push(id);
        let mut kept = 0;
        while kept + 1 < self.rows && id != self.tok.eos() {
            id = self.next(id, None)?;
            out.push(id);
            kept += 1;
        }
        Ok(Drafted {
            proposed: kept,
            accepted: kept,
        })
    }

    fn advance_rows(&self) -> usize {
        self.rows
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.at = 0;
        self.pos = 0;
        self.started = false;
        Ok(())
    }

    fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    fn describe(&self) -> String {
        format!("scripted position={}", self.pos)
    }
}

/// Byte-accumulating decoder: emits the longest valid UTF-8 prefix it holds.
struct Utf8Decoder {
    held: Vec<u8>,
}

impl Decoder for Utf8Decoder {
    fn push(&mut self, id: u32) -> Option<String> {
        self.held.extend(token_bytes(id));
        let valid = match std::str::from_utf8(&self.held) {
            Ok(_) => self.held.len(),
            Err(e) if e.error_len().is_some() => {
                // An invalid (not merely incomplete) sequence: give up on it.
                let s = String::from_utf8_lossy(&self.held).into_owned();
                self.held.clear();
                return Some(s);
            }
            Err(e) => e.valid_up_to(),
        };
        if valid == 0 {
            return None;
        }
        let rest = self.held.split_off(valid);
        let done = std::mem::replace(&mut self.held, rest);
        String::from_utf8(done).ok()
    }

    fn flush(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.held).into_owned();
        self.held.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip_with_specials_and_multibyte() {
        let m = MockTokenizer;
        let text = "<｜User｜>héllo</think>";
        let ids = m.encode(text);
        assert_eq!(ids[0], 2);
        assert_eq!(*ids.last().expect("non-empty"), 5);
        assert_eq!(m.decode(&ids), text);
        let mut d = m.decoder();
        let streamed: String = ids.iter().filter_map(|&id| d.push(id)).collect();
        assert_eq!(streamed, text);
    }

    /// A jump arm's call feeds the id the last step answered, then the run:
    /// each is the script's next id, a repeat of the one before it too, so
    /// the script "aab" ends at its end.
    #[test]
    fn a_scripted_call_consumes_a_repeated_id() {
        let t = MockTokenizer;
        let (a, b) = (t.encode("a")[0], t.encode("b")[0]);
        let mut e = ScriptedEngine::from_ids(64, vec![a, a, b]);
        let prompt = t.encode("hi");
        e.prefill(&prompt[..prompt.len() - 1]).expect("fits");
        assert_eq!(e.next(prompt[prompt.len() - 1], None).expect("fits"), a);
        assert_eq!(e.next(a, None).expect("fits"), a);
        assert_eq!(e.next_after(&[a, b], None).expect("fits"), t.eos());
    }

    #[test]
    fn bigram_echo_and_eos() {
        let mut m = MockEngine::new(64);
        let t = MockTokenizer;
        let ids = t.encode("abcab");
        let mut logits = vec![0.0; t.n_vocab()];
        m.prefill(&ids[..ids.len() - 1]).expect("fits");
        let next = m.next(ids[ids.len() - 1], Some(&mut logits)).expect("fits");
        assert_eq!(t.decode(&[next]), "c");
        assert_eq!(logits[next as usize], PREDICTED);
        m.reset().expect("mock reset");
        let ids = t.encode("xyz");
        m.prefill(&ids[..2]).expect("fits");
        assert_eq!(m.next(ids[2], None).expect("fits"), t.eos());
    }
}
