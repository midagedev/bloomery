//! Reader of ik's KL-divergence base file: what `llama-perplexity
//! --kl-divergence-base <file>` writes (examples/perplexity/perplexity.cpp,
//! `perplexity()` and the `process_logits` overload that takes a stream) and
//! `tools/ref/ik-ppl.sh --kld-base` keeps as `$BLOOMERY_DATA/ikppl/<tag>.kld`.
//! The file holds the ids ik evaluated and, at every position it scored, its
//! log-probabilities over the whole vocabulary: an engine can be scored against
//! ik position by position on the same ids, with no tokenizer of its own.
//!
//! Layout, little-endian, no padding between fields:
//!
//! ```text
//! "_logits_"                 8 bytes
//! n_ctx, n_vocab, n_chunk    i32 each
//! token ids                  i32 × n_chunk·n_ctx, chunk after chunk
//! records                    one per scored position, chunk after chunk
//! ```
//!
//! ik evaluates each chunk of `n_ctx` ids from an empty cache and scores
//! positions `n_ctx/2 ..= n_ctx-2`, each against the id after it. A record is
//! `2·⌈n_vocab/2⌉ + 4` u16: f32 `scale`, f32 `min_log_prob`, one level per
//! vocabulary entry, and one zero u16 when `n_vocab` is odd. Entry `i`'s
//! log-probability is `scale·level[i] + min_log_prob`. ik's quantizer (the
//! `log_softmax` overload that writes levels) maps the logits between the
//! maximum and `max(min, max − 24)` onto `0 ..= 65535`: the maximum lands on
//! [`TOP_LEVEL`], and an entry more than [`FLOOR_NATS`] below it is stored at
//! level 0, the floor.
//!
//! The ids are the text's own. When a model adds a BOS, ik evaluates BOS in
//! place of each chunk's first id and does not write that here; a runner must
//! do the same. The V4.1 file sets `tokenizer.ggml.add_bos_token` to false.
//!
//! Two files of one text are compared record by record through [`Pair`] and
//! [`compare`] (the `kld_diff` bin).

use crate::GateError;
use memmap2::Mmap;
use std::fs::File;
use std::path::{Path, PathBuf};

/// ik's floor, in nats below a record's maximum: a lower entry is stored at
/// level 0.
pub const FLOOR_NATS: f32 = 24.0;

/// The level ik gives a record's maximum.
pub const TOP_LEVEL: u16 = u16::MAX;

/// The widest step a record can have, in nats: the floor's range over the
/// 65,535 steps between level 0 and [`TOP_LEVEL`].
const MAX_STEP: f64 = FLOOR_NATS as f64 / TOP_LEVEL as f64;

/// How far a stored scale may exceed [`MAX_STEP`]: ik takes the range as
/// `max − (max − 24)` in f32, which overshoots 24 nats by half an ulp of
/// `max − 24` and one of 24 — under 1e-5 relative for any logit below 4,096.
const SCALE_ROUNDING: f64 = 1.0 + 1e-5;

/// f32 rounding between ik's printed NLL and a record's, per position, in
/// nats. Every quantity involved is below 64 nats (|`min_log_prob`| ≤ 24 +
/// ln n_vocab), where half an ulp is 2^-19: ik's NLL and `min_log_prob` take
/// two roundings each; a level carries four relative roundings before
/// `nearest_int` (≤ 0.016 level, 5.9e-6 nats at the widest step); the stored
/// scale times a level carries three relative roundings of at most 24 nats
/// (4.3e-6). That is 1.8e-5 in all; this is it rounded up.
pub const F32_SLACK: f64 = 2.5e-5;

const MAGIC: &[u8; 8] = b"_logits_";

/// The magic and three i32.
const HEADER_BYTES: usize = MAGIC.len() + 3 * 4;

/// A record's two f32 before its levels.
const RECORD_HEAD_BYTES: usize = 2 * 4;

/// One base file, mapped and checked against ik's layout by [`KldBase::open`].
pub struct KldBase {
    path: PathBuf,
    map: Mmap,
    n_ctx: usize,
    n_vocab: usize,
    n_chunk: usize,
    record_bytes: usize,
    tokens: Vec<u32>,
}

impl KldBase {
    /// Map `path` and check it against ik's layout: the magic; positive sizes
    /// whose `n_vocab` is `want_vocab`; a file length equal to the one they
    /// imply; every token id below `n_vocab`. A file that fails is an error
    /// naming what disagrees. The records are read when asked for, not here;
    /// each checks itself ([`Record::check`]).
    pub fn open(path: &Path, want_vocab: usize) -> Result<KldBase, GateError> {
        Self::map_checked(path, Some(want_vocab))
    }

    /// [`KldBase::open`] with the vocabulary the file's own header states:
    /// for comparing two base files with each other ([`Pair`]), where neither
    /// side is the model file.
    pub fn open_own_vocab(path: &Path) -> Result<KldBase, GateError> {
        Self::map_checked(path, None)
    }

    fn map_checked(path: &Path, want_vocab: Option<usize>) -> Result<KldBase, GateError> {
        let at = path.display();
        let file = File::open(path).map_err(|e| format!("kld: cannot open {at}: {e}"))?;
        let len = file
            .metadata()
            .map_err(|e| format!("kld: {at}: {e}"))?
            .len();
        if len < HEADER_BYTES as u64 {
            return Err(format!(
                "kld: {at} is {len} bytes, shorter than ik's {HEADER_BYTES}-byte header"
            )
            .into());
        }
        // SAFETY: the file is opened read-only and nothing here maps it
        // writable; a concurrent truncation would surface as SIGBUS, the
        // contract `gguf::Gguf::open_backed` accepts for the model file.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("kld: cannot map {at}: {e}"))?;
        if &map[..MAGIC.len()] != MAGIC {
            return Err(format!("kld: {at} does not open with ik's \"_logits_\" magic").into());
        }
        let size = |i: usize, what: &str| -> Result<usize, GateError> {
            let v = le_i32(&map, MAGIC.len() + 4 * i);
            match usize::try_from(v) {
                Ok(n) if n > 0 => Ok(n),
                _ => Err(format!("kld: {at} states {what} {v}").into()),
            }
        };
        let (n_ctx, n_vocab, n_chunk) =
            (size(0, "n_ctx")?, size(1, "n_vocab")?, size(2, "n_chunk")?);
        if let Some(want_vocab) = want_vocab.filter(|&w| w != n_vocab) {
            return Err(format!(
                "kld: {at} states n_vocab {n_vocab}, but the model's vocabulary is {want_vocab}"
            )
            .into());
        }
        let record_bytes = 2 * (n_vocab.next_multiple_of(2) + 4);
        let n_ids = n_chunk
            .checked_mul(n_ctx)
            .ok_or_else(|| format!("kld: {at}: n_chunk × n_ctx overflows"))?;
        let want = layout_bytes(n_ids, n_chunk * scored_per_chunk(n_ctx), record_bytes)
            .ok_or_else(|| format!("kld: {at}: the sizes its header states overflow"))?;
        if map.len() != want {
            return Err(format!(
                "kld: {at} is {} bytes; its header (n_ctx {n_ctx}, n_vocab {n_vocab}, n_chunk \
                 {n_chunk}) implies {want}",
                map.len()
            )
            .into());
        }
        let mut tokens = Vec::with_capacity(n_ids);
        for (i, b) in map[HEADER_BYTES..HEADER_BYTES + 4 * n_ids]
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
        {
            let id = i32::from_le_bytes(*b);
            match u32::try_from(id) {
                Ok(t) if (t as usize) < n_vocab => tokens.push(t),
                _ => {
                    return Err(format!(
                        "kld: {at}: id {i} (chunk {}, position {}) is {id}, outside 0..{n_vocab}",
                        i / n_ctx,
                        i % n_ctx
                    )
                    .into());
                }
            }
        }
        Ok(KldBase {
            path: path.to_path_buf(),
            map,
            n_ctx,
            n_vocab,
            n_chunk,
            record_bytes,
            tokens,
        })
    }

    /// The file this was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Ids per chunk.
    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Entries per record: the model's vocabulary.
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Chunks ik scored.
    pub fn n_chunk(&self) -> usize {
        self.n_chunk
    }

    /// The first position ik scores in a chunk, `n_ctx / 2`.
    pub fn first_scored(&self) -> usize {
        self.n_ctx / 2
    }

    /// Positions scored per chunk: `first_scored() ..= n_ctx - 2`.
    pub fn scored_per_chunk(&self) -> usize {
        scored_per_chunk(self.n_ctx)
    }

    /// Every id ik evaluated, `n_chunk · n_ctx`, chunk after chunk.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Chunk `chunk`'s `n_ctx` ids; `None` past the last chunk.
    pub fn chunk_tokens(&self, chunk: usize) -> Option<&[u32]> {
        (chunk < self.n_chunk).then(|| &self.tokens[chunk * self.n_ctx..(chunk + 1) * self.n_ctx])
    }

    /// The record at position `pos` of chunk `chunk`; `None` unless ik scored
    /// that position.
    pub fn record(&self, chunk: usize, pos: usize) -> Option<Record<'_>> {
        let k = pos.checked_sub(self.first_scored())?;
        let per = self.scored_per_chunk();
        (chunk < self.n_chunk && k < per).then(|| self.record_at(chunk * per + k))
    }

    /// Every record in file order: chunk by chunk, position by position. The
    /// mapping pages the file in as records are read; nothing is copied.
    pub fn records(&self) -> impl ExactSizeIterator<Item = Record<'_>> + '_ {
        (0..self.n_chunk * self.scored_per_chunk()).map(|i| self.record_at(i))
    }

    /// Record `i` in file order; `i` is below the record count.
    fn record_at(&self, i: usize) -> Record<'_> {
        let per = self.scored_per_chunk();
        let (chunk, pos) = (i / per, self.first_scored() + i % per);
        let at = HEADER_BYTES + 4 * self.tokens.len() + i * self.record_bytes;
        let rec = &self.map[at..at + self.record_bytes];
        Record {
            chunk,
            pos,
            next: self.tokens[chunk * self.n_ctx + pos + 1],
            scale: le_f32(rec, 0),
            min_log_prob: le_f32(rec, 4),
            levels: &rec[RECORD_HEAD_BYTES..RECORD_HEAD_BYTES + 2 * self.n_vocab],
        }
    }
}

/// One scored position: ik's log-probabilities for the id after `pos`.
#[derive(Clone, Copy)]
pub struct Record<'a> {
    /// The chunk, from 0.
    pub chunk: usize,
    /// The position within the chunk whose logits these are.
    pub pos: usize,
    /// The id at `pos + 1`: the one scored.
    pub next: u32,
    /// Nats per level.
    pub scale: f32,
    /// The log-probability of level 0.
    pub min_log_prob: f32,
    levels: &'a [u8],
}

impl Record<'_> {
    /// The levels, entry by entry.
    pub fn levels(&self) -> impl ExactSizeIterator<Item = u16> + '_ {
        self.levels
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
    }

    /// Entry `tok`'s level. Panics past the vocabulary.
    pub fn level(&self, tok: u32) -> u16 {
        let at = 2 * tok as usize;
        u16::from_le_bytes([self.levels[at], self.levels[at + 1]])
    }

    /// Entry `tok`'s log-probability, `scale · level + min_log_prob`, in f64
    /// from the stored f32.
    pub fn log_prob(&self, tok: u32) -> f64 {
        f64::from(self.scale) * f64::from(self.level(tok)) + f64::from(self.min_log_prob)
    }

    /// −log p(next).
    pub fn nll(&self) -> f64 {
        -self.log_prob(self.next)
    }

    /// `next` is stored at level 0. When the record's range reaches the floor,
    /// its true log-probability may be lower than stored, and [`Record::nll`]
    /// is then a lower bound, not a value within half a step.
    pub fn next_at_floor(&self) -> bool {
        self.level(self.next) == 0
    }

    /// The dequantized row, entry by entry, in f32 — the arithmetic of ik's
    /// own `--kl-divergence` mode.
    pub fn log_probs(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        let (s, m) = (self.scale, self.min_log_prob);
        self.levels().map(move |q| s * f32::from(q) + m)
    }

    /// The record is one ik's quantizer can write: finite, a scale no wider
    /// than the floor's step, its top level [`TOP_LEVEL`] (every level 0 when
    /// the scale is 0), and a mass `Σ exp(log p)` within its band of 1. Each
    /// entry above the floor is off by at most half a step plus the f32
    /// rounding slack in log space, which bounds their share of the mass by
    /// `expm1` of that; an entry at level 0 may stand for any smaller
    /// probability, so it adds its whole stored mass. An error names what
    /// disagrees.
    pub fn check(&self) -> Result<(), GateError> {
        let (scale, mlp) = (f64::from(self.scale), f64::from(self.min_log_prob));
        if !(scale.is_finite()
            && mlp.is_finite()
            && (0.0..=MAX_STEP * SCALE_ROUNDING).contains(&scale))
        {
            return Err(format!(
                "scale {} with min_log_prob {} is no record of ik's quantizer (scale above \
                 {MAX_STEP:.4e} or not finite)",
                self.scale, self.min_log_prob
            )
            .into());
        }
        let (mut top, mut zeros, mut mass) = (0u16, 0usize, 0.0f64);
        for q in self.levels() {
            top = top.max(q);
            if q == 0 {
                zeros += 1;
            } else {
                mass += (scale * f64::from(q) + mlp).exp();
            }
        }
        let floor_mass = zeros as f64 * mlp.exp();
        mass += floor_mass;
        let want_top = if scale > 0.0 { TOP_LEVEL } else { 0 };
        if top != want_top {
            return Err(format!("top level {top}, where ik's quantizer puts {want_top}").into());
        }
        let band = (scale / 2.0 + F32_SLACK).exp_m1() + floor_mass;
        if (mass - 1.0).abs() > band {
            return Err(format!("mass {mass:.6} is off 1 by more than its band {band:.3e}").into());
        }
        Ok(())
    }
}

/// ik's `--kl-divergence` sums only the base entries whose log-probability,
/// dequantized in f32, is above this many nats (`log_softmax`, the overload
/// that reads a base record).
pub const IK_KLD_CUT: f32 = -16.0;

/// Two base files of one text, compared position by position: P is taken as
/// the truth and Q is scored against it — the order of ik's
/// `--kl-divergence`, whose base file is P.
pub struct Pair<'a> {
    p: &'a KldBase,
    q: &'a KldBase,
    n_chunk: usize,
}

impl<'a> Pair<'a> {
    /// `p` and `q` over the chunks both hold: a file with more chunks is
    /// compared on its prefix. Refuses two files whose `n_ctx` or `n_vocab`
    /// differ, or whose ids differ anywhere in those chunks, with an error
    /// naming the field.
    pub fn new(p: &'a KldBase, q: &'a KldBase) -> Result<Pair<'a>, GateError> {
        let (pa, qa) = (p.path.display(), q.path.display());
        if p.n_ctx != q.n_ctx {
            return Err(format!(
                "kld: n_ctx differs: {pa} states {}, {qa} states {}",
                p.n_ctx, q.n_ctx
            )
            .into());
        }
        if p.n_vocab != q.n_vocab {
            return Err(format!(
                "kld: n_vocab differs: {pa} states {}, {qa} states {}",
                p.n_vocab, q.n_vocab
            )
            .into());
        }
        let n_chunk = p.n_chunk.min(q.n_chunk);
        let n_ids = n_chunk * p.n_ctx;
        if let Some(i) = (0..n_ids).find(|&i| p.tokens[i] != q.tokens[i]) {
            return Err(format!(
                "kld: ids differ: id {i} (chunk {}, position {}) is {} in {pa} and {} in {qa}",
                i / p.n_ctx,
                i % p.n_ctx,
                p.tokens[i],
                q.tokens[i]
            )
            .into());
        }
        Ok(Pair { p, q, n_chunk })
    }

    /// The file taken as the truth.
    pub fn p(&self) -> &'a KldBase {
        self.p
    }

    /// The file scored against it.
    pub fn q(&self) -> &'a KldBase {
        self.q
    }

    /// Chunks compared: the fewer of the two files'.
    pub fn n_chunk(&self) -> usize {
        self.n_chunk
    }

    /// P's and Q's records at every scored position of the compared chunks,
    /// in file order.
    pub fn records(&self) -> impl ExactSizeIterator<Item = (Record<'a>, Record<'a>)> + 'a {
        let (p, q) = (self.p, self.q);
        (0..self.n_chunk * p.scored_per_chunk()).map(move |i| (p.record_at(i), q.record_at(i)))
    }
}

/// One scored position of a [`Pair`], compared entry by entry ([`compare`]).
#[derive(Clone, Copy, Debug)]
pub struct PosDiff {
    /// The chunk, from 0.
    pub chunk: usize,
    /// The position within the chunk whose logits these are.
    pub pos: usize,
    /// The id scored: the one at `pos + 1`.
    pub next: u32,
    /// KL(P‖Q) = Σ p·(ln p − ln q) over the whole vocabulary, in nats, in f64
    /// from both records' stored levels; an entry at level 0 counts at the
    /// value stored for it.
    pub kld: f64,
    /// The same sum over the entries ik's own `--kl-divergence` sums: those
    /// P stores above [`IK_KLD_CUT`], tested in f32 as ik tests them.
    pub kld_ik_cut: f64,
    /// Entries in [`PosDiff::kld_ik_cut`] that Q stores at level 0, where Q's
    /// true log-probability may be lower than stored.
    pub cut_at_q_floor: u32,
    /// Σ p over P's record.
    pub p_mass: f64,
    /// P's most likely id: the first entry at its top level, the rule ik
    /// applies to its base record.
    pub top_p: u32,
    /// Q's most likely id by the same rule.
    pub top_q: u32,
    /// Entries at Q's top level. Above 1, the id Q's logits ranked first is
    /// one of them and the file does not say which.
    pub q_top_count: u32,
    /// −ln p(next).
    pub nll_p: f64,
    /// −ln q(next).
    pub nll_q: f64,
    /// Q stores `next` at level 0: `nll_q` is then a lower bound.
    pub q_next_at_floor: bool,
    /// Q's step, nats per level.
    pub q_scale: f32,
}

impl PosDiff {
    /// How far [`PosDiff::kld_ik_cut`] can sit from the KLD ik computes at
    /// this position from Q's fresh logits and P's record, in nats. ik takes
    /// Q's log-probabilities in f32 from the logits; Q's record holds them
    /// quantized, each off by at most half Q's step plus [`F32_SLACK`] (the
    /// roundings between ik's f32 log-probability and a record's). ik's own
    /// evaluation adds four f32 roundings per entry, two in `log p` and two in
    /// `log p − logit + max`, of values below 128 nats while every logit is
    /// below 64: 1.5e-5 at most, within one more [`F32_SLACK`]. Weighted by P's
    /// mass, that is the band. An entry the cut keeps at Q's floor
    /// ([`PosDiff::cut_at_q_floor`]) may stand for a lower log q, which only
    /// raises ik's sum: with one, the band holds on the side where this sum
    /// is the larger.
    pub fn ik_kld_band(&self) -> f64 {
        (f64::from(self.q_scale) / 2.0 + 2.0 * F32_SLACK) * self.p_mass
    }

    /// How far `nll_q` can sit from the NLL ik computes from Q's fresh
    /// logits, in nats: half Q's step plus [`F32_SLACK`], one-sided (the
    /// file's value can only be lower) when `next` is at Q's floor.
    pub fn ik_nll_band(&self) -> f64 {
        f64::from(self.q_scale) / 2.0 + F32_SLACK
    }
}

/// The records of one scored position in P and in Q, entry by entry over the
/// vocabulary: see [`PosDiff`]. The two are the same position of two files a
/// [`Pair`] matched.
pub fn compare(p: &Record<'_>, q: &Record<'_>) -> PosDiff {
    let (sp, mp) = (f64::from(p.scale), f64::from(p.min_log_prob));
    let (sq, mq) = (f64::from(q.scale), f64::from(q.min_log_prob));
    let p_floor = mp.exp();
    let (mut kld, mut cut, mut mass) = (0.0f64, 0.0f64, 0.0f64);
    let mut cut_at_q_floor = 0u32;
    let (mut top_p, mut best_p) = (0u32, 0u16);
    let (mut top_q, mut best_q, mut q_top_count) = (0u32, 0u16, 0u32);
    for (i, (a, b)) in (0u32..).zip(p.levels().zip(q.levels())) {
        let lp = sp * f64::from(a) + mp;
        let pa = if a == 0 { p_floor } else { lp.exp() };
        let term = pa * (lp - (sq * f64::from(b) + mq));
        kld += term;
        mass += pa;
        if p.scale * f32::from(a) + p.min_log_prob > IK_KLD_CUT {
            cut += term;
            cut_at_q_floor += u32::from(b == 0);
        }
        if a > best_p {
            (top_p, best_p) = (i, a);
        }
        if b > best_q || i == 0 {
            (top_q, best_q, q_top_count) = (i, b, 1);
        } else if b == best_q {
            q_top_count += 1;
        }
    }
    PosDiff {
        chunk: p.chunk,
        pos: p.pos,
        next: p.next,
        kld,
        kld_ik_cut: cut,
        cut_at_q_floor,
        p_mass: mass,
        top_p,
        top_q,
        q_top_count,
        nll_p: p.nll(),
        nll_q: q.nll(),
        q_next_at_floor: q.next_at_floor(),
        q_scale: q.scale,
    }
}

/// The model side of a perplexity run over a [`KldBase`], scored against one
/// ([`score_ppl`]) or written as one ([`write_base`]): back to position 0, one
/// id fed at the next position, and the logits that step left, one per
/// vocabulary entry.
pub trait PplModel {
    fn reset(&mut self) -> Result<(), GateError>;
    fn step(&mut self, token: u32) -> Result<(), GateError>;
    fn logits(&mut self) -> Result<Vec<f32>, GateError>;
}

/// One chunk from an empty model, the way ik evaluates it: the ids fed one
/// position at a time, and `visit(pos, logits)` called after each position from
/// `n_ctx / 2` through `n_ctx - 2` (the last id is scored, never fed). The
/// logits are one per vocabulary entry, or the run ends naming the count.
fn feed_chunk(
    m: &mut impl PplModel,
    toks: &[u32],
    n_vocab: usize,
    mut visit: impl FnMut(usize, &[f32]) -> Result<(), GateError>,
) -> Result<(), GateError> {
    let n_ctx = toks.len();
    m.reset()?;
    for (pos, &tok) in toks.iter().enumerate().take(n_ctx - 1) {
        m.step(tok)?;
        if pos < n_ctx / 2 {
            continue;
        }
        let logits = m.logits()?;
        if logits.len() != n_vocab {
            return Err(format!("{} logits, the vocabulary {n_vocab}", logits.len()).into());
        }
        visit(pos, &logits)?;
    }
    Ok(())
}

/// What [`score_ppl`] measured over every scored position of a base.
pub struct PplScore {
    /// Positions scored.
    pub n: usize,
    /// The engine's perplexity and ik's over those positions.
    pub ppl_ours: f64,
    pub ppl_ik: f64,
    /// `d = NLL_ours − NLL_ik` per position: its mean, the mean's standard
    /// error, and its standard deviation.
    pub mean_d: f64,
    pub se_d: f64,
    pub sd_d: f64,
    /// KLD(ik‖ours) per position: its mean and the mean's standard error.
    pub kld: f64,
    pub kld_se: f64,
    /// Positions whose top id is ik's.
    pub same_top: usize,
    /// The rms over positions of the engine's top1 − top2 logit margin minus
    /// ik's log-probability margin.
    pub sigma_rel: f64,
    /// Wall seconds of the run.
    pub secs: f64,
}

impl PplScore {
    /// The relative perplexity change, `exp(mean d) − 1`.
    pub fn dppl(&self) -> f64 {
        self.mean_d.exp() - 1.0
    }
}

/// The top two of `v` by (value desc, id asc): ids and values.
pub fn top2(v: &[f32]) -> (usize, f32, usize, f32) {
    let ahead = |a: usize, b: usize| v[a] > v[b] || (v[a] == v[b] && a < b);
    let (mut best, mut second) = (0usize, usize::MAX);
    for i in 1..v.len() {
        if ahead(i, best) {
            second = best;
            best = i;
        } else if second == usize::MAX || ahead(i, second) {
            second = i;
        }
    }
    (best, v[best], second, v[second])
}

/// Score `m` against `base`: every chunk from position 0, its ids fed one
/// position at a time as ik evaluated them, and at each position ik recorded:
/// the engine's NLL of the next id (log-softmax in f64), KLD(ik‖ours) over the
/// vocabulary, the top id and the top-two margin against the record's. Prints
/// a line per chunk. An engine whose logits are not one per vocabulary entry
/// of the base is an error.
pub fn score_ppl(base: &KldBase, m: &mut impl PplModel) -> Result<PplScore, GateError> {
    let n_vocab = base.n_vocab();
    let t = std::time::Instant::now();
    let (mut n, mut sd, mut sd2) = (0usize, 0.0f64, 0.0f64);
    let (mut nll_o, mut nll_i, mut kld, mut kld2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut same_top, mut dm2) = (0usize, 0.0f64);
    let mut lq = vec![0.0f64; n_vocab];
    for c in 0..base.n_chunk() {
        let toks = base
            .chunk_tokens(c)
            .ok_or_else(|| format!("chunk {c} has no ids"))?;
        feed_chunk(m, toks, n_vocab, |pos, logits| {
            let rec = base
                .record(c, pos)
                .ok_or_else(|| format!("chunk {c} position {pos} has no record"))?;
            let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            let lse = f64::from(mx)
                + logits
                    .iter()
                    .map(|&v| (f64::from(v) - f64::from(mx)).exp())
                    .sum::<f64>()
                    .ln();
            for (q, &v) in lq.iter_mut().zip(logits) {
                *q = f64::from(v) - lse;
            }
            let no = -lq[rec.next as usize];
            let ni = rec.nll();
            let d = no - ni;
            n += 1;
            sd += d;
            sd2 += d * d;
            nll_o += no;
            nll_i += ni;
            let mut k = 0.0f64;
            let (mut top_i, mut best_i, mut second_i) =
                (0usize, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for (i, lp) in rec.log_probs().enumerate() {
                let lp = f64::from(lp);
                k += lp.exp() * (lp - lq[i]);
                if lp > best_i {
                    second_i = best_i;
                    (top_i, best_i) = (i, lp);
                } else if lp > second_i {
                    second_i = lp;
                }
            }
            kld += k;
            kld2 += k * k;
            let (a, av, _, bv) = top2(logits);
            same_top += usize::from(a == top_i);
            let dm = f64::from(av - bv) - (best_i - second_i);
            dm2 += dm * dm;
            Ok(())
        })?;
        let nf = n as f64;
        println!(
            "ppl chunk {c}: {n} positions, mean d {:+.5}, ppl ours {:.4} ik {:.4}, {:.0} s",
            sd / nf,
            (nll_o / nf).exp(),
            (nll_i / nf).exp(),
            t.elapsed().as_secs_f64()
        );
    }
    let nf = n as f64;
    let mean = sd / nf;
    let var = (sd2 / nf - mean * mean).max(0.0) * nf / (nf - 1.0);
    let kmean = kld / nf;
    Ok(PplScore {
        n,
        ppl_ours: (nll_o / nf).exp(),
        ppl_ik: (nll_i / nf).exp(),
        mean_d: mean,
        se_d: (var / nf).sqrt(),
        sd_d: var.sqrt(),
        kld: kmean,
        kld_se: ((kld2 / nf - kmean * kmean).max(0.0) / (nf - 1.0)).sqrt(),
        same_top,
        sigma_rel: (dm2 / nf).sqrt(),
        secs: t.elapsed().as_secs_f64(),
    })
}

/// The first `need` ids of a corpus file, one decimal id a line
/// (`corpus-prose.ids`, the oracle prompts' `.ids`), each below `n_vocab`. A line that is no id, an id outside
/// the vocabulary and a file of fewer than `need` ids are errors that name the line or the count.
pub fn read_ids(path: &Path, need: usize, n_vocab: usize) -> Result<Vec<u32>, GateError> {
    let at = path.display();
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("kld: cannot read ids {at}: {e}"))?;
    let mut ids = Vec::with_capacity(need);
    for (n, line) in text.lines().enumerate() {
        if ids.len() == need {
            break;
        }
        let id: u32 = line
            .trim()
            .parse()
            .map_err(|_| format!("kld: {at}:{}: {line:?} is no id", n + 1))?;
        if id as usize >= n_vocab {
            return Err(format!("kld: {at}:{}: id {id} is outside 0..{n_vocab}", n + 1).into());
        }
        ids.push(id);
    }
    if ids.len() < need {
        return Err(format!("kld: {at} holds {} ids, {need} wanted", ids.len()).into());
    }
    Ok(ids)
}

/// Append ik's record for one position's `logits` to `out` — `log_softmax`, the overload that writes levels,
/// arithmetic for arithmetic: the f32 scale, the f32 `min_log_prob`, one u16 level per entry and, for an odd
/// vocabulary, one zero u16 — and return −log p(`next`) as ik computes it. A level is the nearest integer, ties
/// to even, as ik's `nearest_int`; where ik's f32 product could reach 65,536 and wrap to 0 the level is held
/// at [`TOP_LEVEL`]. A logit that is not finite is an error naming the entry: ik's record holds no such row.
pub fn encode_record(logits: &[f32], next: u32, out: &mut Vec<u8>) -> Result<f64, GateError> {
    let Some(&first) = logits.first() else {
        return Err("kld: no logits to encode".into());
    };
    let tok = next as usize;
    if tok >= logits.len() {
        return Err(format!("kld: id {next} is outside the {} logits", logits.len()).into());
    }
    if let Some(i) = logits.iter().position(|v| !v.is_finite()) {
        return Err(format!("kld: logit {i} is {}", logits[i]).into());
    }
    let (mut max, mut min) = (first, first);
    for &l in &logits[1..] {
        max = max.max(l);
        min = min.min(l);
    }
    let min = min.max(max - FLOOR_NATS);
    let sum: f64 = logits.iter().map(|&l| f64::from((l - max).exp())).sum();
    let lse = sum.ln() as f32;
    let scale = (max - min) / f32::from(TOP_LEVEL);
    out.extend_from_slice(&scale.to_le_bytes());
    out.extend_from_slice(&(min - max - lse).to_le_bytes());
    if scale != 0.0 {
        let inv = 1.0 / scale;
        for &l in logits {
            let q = if l > min {
                (inv * (l - min))
                    .round_ties_even()
                    .min(f32::from(TOP_LEVEL)) as u16
            } else {
                0
            };
            out.extend_from_slice(&q.to_le_bytes());
        }
    } else {
        out.resize(out.len() + 2 * logits.len(), 0);
    }
    if logits.len() % 2 == 1 {
        out.extend_from_slice(&[0, 0]);
    }
    Ok(f64::from(max + lse - logits[tok]))
}

/// A base file in ik's layout written record by record: the header and the ids on [`KldWriter::create`], one
/// record a [`KldWriter::push`] in file order, the length checked and the file renamed whole on
/// [`KldWriter::finish`]. The bytes go to `<path>.part`, so a run that dies leaves no file that reads as a
/// complete base (ik does not check its writes; `tools/ref/ik-ppl.sh` keeps the same rule). An existing `path` or
/// `.part` is refused, never overwritten.
pub struct KldWriter {
    path: PathBuf,
    part: PathBuf,
    out: std::io::BufWriter<File>,
    ids: Vec<u32>,
    n_ctx: usize,
    n_vocab: usize,
    n_records: usize,
    written: usize,
    row: Vec<u8>,
}

impl KldWriter {
    /// Open `<path>.part` for chunks of `n_ctx` over `ids` (a whole number of chunks, each id below `n_vocab`)
    /// and write the header and the ids. `n_ctx` is at least 3: ik scores `n_ctx/2 ..= n_ctx-2`, which is empty
    /// below it.
    pub fn create(
        path: &Path,
        n_ctx: usize,
        n_vocab: usize,
        ids: &[u32],
    ) -> Result<KldWriter, GateError> {
        use std::io::Write;
        let at = path.display();
        if n_ctx < 3 {
            return Err(format!("kld: {at}: n_ctx {n_ctx} scores no position (at least 3)").into());
        }
        if ids.is_empty() || !ids.len().is_multiple_of(n_ctx) {
            return Err(format!(
                "kld: {at}: {} ids are no whole number of chunks of {n_ctx}",
                ids.len()
            )
            .into());
        }
        if let Some(i) = ids.iter().position(|&t| t as usize >= n_vocab) {
            return Err(format!("kld: {at}: id {i} is {}, outside 0..{n_vocab}", ids[i]).into());
        }
        let n_chunk = ids.len() / n_ctx;
        let size = |v: usize, what: &str| {
            i32::try_from(v).map_err(|_| format!("kld: {at}: {what} {v} does not fit ik's i32"))
        };
        let head = [
            size(n_ctx, "n_ctx")?,
            size(n_vocab, "n_vocab")?,
            size(n_chunk, "n_chunk")?,
        ];
        if path.exists() {
            return Err(format!("kld: {at} exists, and a base is never overwritten").into());
        }
        let mut part = path.as_os_str().to_owned();
        part.push(".part");
        let part = PathBuf::from(part);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&part)
            .map_err(|e| format!("kld: cannot create {}: {e}", part.display()))?;
        let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
        let mut bytes = MAGIC.to_vec();
        for v in head {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &t in ids {
            bytes.extend_from_slice(&(t as i32).to_le_bytes());
        }
        out.write_all(&bytes)
            .map_err(|e| format!("kld: {}: {e}", part.display()))?;
        Ok(KldWriter {
            path: path.to_path_buf(),
            part,
            out,
            ids: ids.to_vec(),
            n_ctx,
            n_vocab,
            n_records: n_chunk * scored_per_chunk(n_ctx),
            written: 0,
            row: Vec::new(),
        })
    }

    /// The next record in file order, from the logits of the position it scores: chunk by chunk, position
    /// `n_ctx/2` on. Returns −log p of the id after it. More records than the layout holds, or logits of
    /// another length than the vocabulary, is an error.
    pub fn push(&mut self, logits: &[f32]) -> Result<f64, GateError> {
        use std::io::Write;
        if self.written == self.n_records {
            return Err(format!(
                "kld: {}: a record past the {} the layout holds",
                self.part.display(),
                self.n_records
            )
            .into());
        }
        if logits.len() != self.n_vocab {
            return Err(format!(
                "kld: {}: {} logits, the vocabulary {}",
                self.part.display(),
                logits.len(),
                self.n_vocab
            )
            .into());
        }
        let per = scored_per_chunk(self.n_ctx);
        let (chunk, pos) = (self.written / per, self.n_ctx / 2 + self.written % per);
        let next = self.ids[chunk * self.n_ctx + pos + 1];
        self.row.clear();
        let nll = encode_record(logits, next, &mut self.row)
            .map_err(|e| format!("{e} (chunk {chunk}, position {pos})"))?;
        self.out
            .write_all(&self.row)
            .map_err(|e| format!("kld: {}: {e}", self.part.display()))?;
        self.written += 1;
        Ok(nll)
    }

    /// Records written so far.
    pub fn written(&self) -> usize {
        self.written
    }

    /// Check that every record the layout holds is written and the file has the length its header implies, then
    /// rename `<path>.part` to `path`.
    pub fn finish(self) -> Result<PathBuf, GateError> {
        let at = self.part.display().to_string();
        if self.written != self.n_records {
            return Err(format!(
                "kld: {at}: {} records written, the layout holds {}",
                self.written, self.n_records
            )
            .into());
        }
        let file = self
            .out
            .into_inner()
            .map_err(|e| format!("kld: {at}: {e}"))?;
        file.sync_all().map_err(|e| format!("kld: {at}: {e}"))?;
        let len = file
            .metadata()
            .map_err(|e| format!("kld: {at}: {e}"))?
            .len();
        let want = layout_bytes(
            self.ids.len(),
            self.n_records,
            2 * (self.n_vocab.next_multiple_of(2) + 4),
        )
        .ok_or_else(|| format!("kld: {at}: the sizes overflow"))?;
        if len != want as u64 {
            return Err(format!("kld: {at} is {len} bytes, its header implies {want}").into());
        }
        std::fs::rename(&self.part, &self.path)
            .map_err(|e| format!("kld: cannot rename {at} to {}: {e}", self.path.display()))?;
        Ok(self.path)
    }
}

/// What [`write_base`] wrote.
pub struct Written {
    /// The finished file.
    pub path: PathBuf,
    /// Positions scored: the records.
    pub positions: usize,
    /// exp of the mean NLL the walk computed from the logits, before the records' quantization.
    pub ppl: f64,
    /// Wall seconds of the walk.
    pub secs: f64,
}

/// Positions between two progress lines of a walk.
const PROGRESS_EVERY: usize = 256;

/// Run `m` over `ids` in chunks of `n_ctx`, every chunk from an empty model as ik evaluates it, and write the
/// base file `out` (ik's layout: the ids as given, no BOS substituted, a record at every position from
/// `n_ctx/2` through `n_ctx - 2`) that [`score_ppl`], `kld_diff` and ik's own `--kl-divergence` read. Prints a
/// progress line every [`PROGRESS_EVERY`] positions and one a chunk.
pub fn write_base(
    m: &mut impl PplModel,
    ids: &[u32],
    n_ctx: usize,
    n_vocab: usize,
    out: &Path,
) -> Result<Written, GateError> {
    let t = std::time::Instant::now();
    let mut w = KldWriter::create(out, n_ctx, n_vocab, ids)?;
    let (mut n, mut nll) = (0usize, 0.0f64);
    for (c, toks) in ids.chunks(n_ctx).enumerate() {
        feed_chunk(m, toks, n_vocab, |pos, logits| {
            nll += w.push(logits)?;
            n += 1;
            if n % PROGRESS_EVERY == 0 {
                println!(
                    "kld chunk {c} position {pos}: {n} positions, ppl {:.4}, {:.0} s",
                    (nll / n as f64).exp(),
                    t.elapsed().as_secs_f64()
                );
            }
            Ok(())
        })?;
        println!(
            "kld chunk {c}: {n} positions, ppl {:.4}, {:.0} s",
            (nll / n as f64).exp(),
            t.elapsed().as_secs_f64()
        );
    }
    let path = w.finish()?;
    Ok(Written {
        path,
        positions: n,
        ppl: (nll / n as f64).exp(),
        secs: t.elapsed().as_secs_f64(),
    })
}

/// The model-free half of a loss tool's command line, one owner for every model's tool (`ppl_exaone`,
/// `ppl_ds41`): the tool reads its own flags and opens its model, and everything between the arguments and
/// [`write_base`] is here.
///
/// ```text
/// TOOL (--ids FILE | --ids-of BASE.kld) --run CTX[xCHUNKS]=OUT.kld ... [--plan-ctx C] [the tool's own flags]
/// ```
///
/// `--ids` is one decimal id a line (`corpus-prose.ids`); `--ids-of` is the ids an existing base file holds
/// ([`KldBase::tokens`], chunk after chunk), so a run reads the text another engine's base was made from
/// without a tokenizer. Each `--run` (repeatable, one load serves them all) takes the first `CTX × CHUNKS` ids as
/// `CHUNKS` chunks of `CTX` (default one chunk) and writes their base file `OUT`; two files share a base
/// file's meaning only when their ids, `CTX` and `CHUNKS` are the same. `--plan-ctx` is the positions the model is
/// planned for (default [`cli::PLAN_CTX`]); a `CTX` above it is refused, and an existing `OUT` before the load.
pub mod cli {
    use super::{KldBase, PplModel, read_ids, write_base};
    use crate::GateError;
    use std::path::PathBuf;

    /// Positions the model is planned for when `--plan-ctx` is not given: the serve seat's context.
    pub const PLAN_CTX: usize = 8192;

    /// The flags every loss tool reads, each with one value; a tool names its own in its [`Spec`].
    const COMMON: [&str; 4] = ["--ids", "--ids-of", "--run", "--plan-ctx"];

    /// What a tool reads beside [`COMMON`]: its name (the prefix of every refusal), its own flags with a
    /// value, its switches (no value), and the switch that ends the run before the load, when it has one: the
    /// ids and the runs are then optional.
    pub struct Spec {
        pub tool: &'static str,
        pub own: &'static [&'static str],
        pub switches: &'static [&'static str],
        pub plan_only: Option<&'static str>,
    }

    /// One `--run`: the first `ctx * chunks` ids as `chunks` chunks of `ctx`, written to `out`.
    pub struct Run {
        pub ctx: usize,
        pub chunks: usize,
        pub out: PathBuf,
    }

    impl Run {
        /// `CTX=OUT` or `CTXxCHUNKS=OUT`; a spec of another shape is refused by name.
        fn parse(tool: &str, spec: &str) -> Result<Self, GateError> {
            let bad = || {
                GateError::from(format!(
                    "{tool}: --run {spec:?} is not CTX[xCHUNKS]=OUT.kld (CTX at least 3, CHUNKS at least 1)"
                ))
            };
            let (shape, out) = spec.split_once('=').ok_or_else(bad)?;
            let (ctx, chunks) = shape.split_once('x').unwrap_or((shape, "1"));
            let ctx: usize = ctx.parse().map_err(|_| bad())?;
            let chunks: usize = chunks.parse().map_err(|_| bad())?;
            if ctx < 3 || chunks == 0 || out.is_empty() {
                return Err(bad());
            }
            Ok(Self {
                ctx,
                chunks,
                out: PathBuf::from(out),
            })
        }

        /// Ids the run walks.
        pub fn ids(&self) -> usize {
            self.ctx * self.chunks
        }
    }

    /// Where the ids come from: exactly one of `--ids` and `--ids-of`.
    enum IdsFrom {
        File(PathBuf),
        Base(PathBuf),
    }

    /// A parsed command line.
    pub struct Args {
        tool: &'static str,
        ids: Option<IdsFrom>,
        plan_ctx: usize,
        runs: Vec<Run>,
        own: Vec<(&'static str, String)>,
        given: Vec<&'static str>,
    }

    impl Args {
        /// Read `args` (the arguments after the program's name) by `spec`. Refused by name: a flag that is
        /// none of the tool's, a flag with no value or given twice (`--run` repeats), both or neither of
        /// `--ids` and `--ids-of`, a `--plan-ctx` that is no count, a `--run` of another shape or above
        /// `--plan-ctx`, an `OUT` that exists or is two runs'. The model is not read here.
        pub fn parse(spec: &Spec, args: &[String]) -> Result<Args, GateError> {
            let tool = spec.tool;
            let mut values: Vec<(&str, &str)> = Vec::new();
            let mut given = Vec::new();
            let mut at = 0;
            while at < args.len() {
                let flag = args[at].as_str();
                if let Some(&s) = spec.switches.iter().find(|&&s| s == flag) {
                    given.push(s);
                    at += 1;
                    continue;
                }
                if !COMMON.contains(&flag) && !spec.own.contains(&flag) {
                    let all: Vec<&str> = COMMON
                        .iter()
                        .chain(spec.own)
                        .chain(spec.switches)
                        .copied()
                        .collect();
                    return Err(
                        format!("{tool}: {flag} is no flag of this tool (one of {all:?})").into(),
                    );
                }
                let v = args
                    .get(at + 1)
                    .ok_or_else(|| GateError::from(format!("{tool}: {flag} needs a value")))?;
                values.push((flag, v));
                at += 2;
            }
            let one = |name: &str| -> Result<Option<&str>, GateError> {
                let mut found = values.iter().filter(|(f, _)| *f == name).map(|&(_, v)| v);
                let first = found.next();
                match found.next() {
                    Some(_) => Err(format!("{tool}: {name} is given once").into()),
                    None => Ok(first),
                }
            };
            let plan_only = spec.plan_only.is_some_and(|s| given.contains(&s));
            let ids = match (one("--ids")?, one("--ids-of")?) {
                (Some(_), Some(_)) => {
                    return Err(format!(
                        "{tool}: --ids and --ids-of are two ways to give the ids: give one"
                    )
                    .into());
                }
                (Some(p), None) => Some(IdsFrom::File(PathBuf::from(p))),
                (None, Some(p)) => Some(IdsFrom::Base(PathBuf::from(p))),
                (None, None) if plan_only => None,
                (None, None) => {
                    return Err(format!(
                        "{tool}: --ids FILE (one id a line) or --ids-of BASE.kld is required"
                    )
                    .into());
                }
            };
            let plan_ctx = one("--plan-ctx")?.map_or(Ok(PLAN_CTX), |s| {
                s.parse::<usize>()
                    .map_err(|_| GateError::from(format!("{tool}: --plan-ctx {s:?} is no count")))
            })?;
            let runs = values
                .iter()
                .filter(|(f, _)| *f == "--run")
                .map(|&(_, s)| Run::parse(tool, s))
                .collect::<Result<Vec<_>, _>>()?;
            if runs.is_empty() && !plan_only {
                return Err(
                    format!("{tool}: at least one --run CTX[xCHUNKS]=OUT.kld is required").into(),
                );
            }
            for (i, r) in runs.iter().enumerate() {
                if r.ctx > plan_ctx {
                    return Err(format!(
                        "{tool}: --run context {} is above --plan-ctx {plan_ctx}",
                        r.ctx
                    )
                    .into());
                }
                if r.out.exists() {
                    return Err(format!(
                        "{tool}: {} exists, and a base is never overwritten",
                        r.out.display()
                    )
                    .into());
                }
                if runs[..i].iter().any(|p| p.out == r.out) {
                    return Err(
                        format!("{tool}: {} is the output of two runs", r.out.display()).into(),
                    );
                }
            }
            let mut own = Vec::new();
            for &flag in spec.own {
                if let Some(v) = one(flag)? {
                    own.push((flag, v.to_owned()));
                }
            }
            Ok(Args {
                tool,
                ids,
                plan_ctx,
                runs,
                own,
                given,
            })
        }

        /// Positions the model is planned for.
        pub fn plan_ctx(&self) -> usize {
            self.plan_ctx
        }

        /// The value of the tool's own flag `name`, when given.
        pub fn value(&self, name: &str) -> Option<&str> {
            self.own
                .iter()
                .find(|(f, _)| *f == name)
                .map(|(_, v)| v.as_str())
        }

        /// Whether the tool's switch `name` was given.
        pub fn switch(&self, name: &str) -> bool {
            self.given.contains(&name)
        }

        /// The ids the runs need, read before the load: the count and the format of a `--ids` file (its
        /// vocabulary bound waits for the model, [`Ids::for_vocab`]), or the first ids of the `--ids-of` base.
        /// A file of fewer ids than the longest run takes is an error naming the count.
        pub fn ids(&self) -> Result<Ids, GateError> {
            let tool = self.tool;
            let need = self.runs.iter().map(Run::ids).max().unwrap_or(0);
            match self.ids.as_ref() {
                None => Err(format!("{tool}: no ids were given").into()),
                Some(IdsFrom::File(path)) => {
                    read_ids(path, need, usize::MAX)?;
                    Ok(Ids {
                        file: Some(path.clone()),
                        need,
                        ids: Vec::new(),
                        vocab: 0,
                    })
                }
                Some(IdsFrom::Base(path)) => {
                    let base = KldBase::open_own_vocab(path)?;
                    if base.tokens().len() < need {
                        return Err(format!(
                            "{tool}: --ids-of {} holds {} ids, {need} wanted",
                            path.display(),
                            base.tokens().len()
                        )
                        .into());
                    }
                    Ok(Ids {
                        file: None,
                        need,
                        ids: base.tokens()[..need].to_vec(),
                        vocab: base.n_vocab(),
                    })
                }
            }
        }

        /// Run `m` over `ids` ([`Ids::for_vocab`]) once for each `--run`, writing its base file
        /// ([`write_base`]) and printing its `kld_base` line: the path, `n_ctx`, `n_chunk`, `n_vocab`, the
        /// positions scored, the PPL of the walk and its wall seconds.
        pub fn walk(
            &self,
            m: &mut impl PplModel,
            ids: &[u32],
            vocab: usize,
        ) -> Result<(), GateError> {
            for r in &self.runs {
                let w = write_base(m, &ids[..r.ids()], r.ctx, vocab, &r.out)?;
                println!(
                    "kld_base path={} n_ctx={} n_chunk={} n_vocab={vocab} positions={} ppl={:.4} wall_s={:.1}",
                    w.path.display(),
                    r.ctx,
                    r.chunks,
                    w.positions,
                    w.ppl,
                    w.secs
                );
            }
            Ok(())
        }
    }

    /// The ids a command line names, read before the model: [`Ids::for_vocab`] holds them to the model's
    /// vocabulary once it is loaded.
    pub struct Ids {
        file: Option<PathBuf>,
        need: usize,
        ids: Vec<u32>,
        vocab: usize,
    }

    impl Ids {
        /// The ids for a model whose head is `vocab` wide: a `--ids` file read again with every id held below
        /// `vocab`, so a line that is outside it is named; a `--ids-of` base states the vocabulary it was run
        /// over, which must be the model's.
        pub fn for_vocab(self, vocab: usize) -> Result<Vec<u32>, GateError> {
            match self.file {
                Some(path) => read_ids(&path, self.need, vocab),
                None if self.vocab != vocab => Err(format!(
                    "kld: the --ids-of base states n_vocab {}, but the model's vocabulary is {vocab}",
                    self.vocab
                )
                .into()),
                None => Ok(self.ids),
            }
        }
    }
}

/// A run's result line: the last line of `tools/ref/ik-ppl.sh`'s log that
/// opens with `ppl tag=<tag> `, read field by field (`key=value`, space
/// separated).
pub struct ResultLine {
    log: PathBuf,
    line: String,
}

impl ResultLine {
    /// The result line of run `tag` in `log`; an error when the log has none.
    pub fn read(log: &Path, tag: &str) -> Result<ResultLine, GateError> {
        let raw = std::fs::read(log).map_err(|e| format!("cannot read {}: {e}", log.display()))?;
        // ik's model-load lines carry bytes that are not UTF-8; the result
        // line is ASCII.
        let text = String::from_utf8_lossy(&raw);
        let head = format!("ppl tag={tag} ");
        let line = text
            .lines()
            .rev()
            .find(|l| l.starts_with(&head))
            .ok_or_else(|| format!("{} has no result line for {tag}", log.display()))?;
        Ok(ResultLine {
            log: log.to_path_buf(),
            line: line.to_string(),
        })
    }

    /// The text of field `key`; an error when the line has none.
    pub fn text(&self, key: &str) -> Result<&str, GateError> {
        self.line
            .split_whitespace()
            .find_map(|t| t.strip_prefix(key)?.strip_prefix('='))
            .ok_or_else(|| format!("{}: the result line has no {key}=", self.log.display()).into())
    }

    /// Field `key` parsed; an error naming the log and the key when it is
    /// missing or does not parse.
    pub fn parse<T>(&self, key: &str) -> Result<T, GateError>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        self.text(key)?
            .parse()
            .map_err(|e| format!("{}: {key}=: {e}", self.log.display()).into())
    }
}

/// Positions ik scores in a chunk of `n_ctx`: `n_ctx/2 ..= n_ctx-2`.
fn scored_per_chunk(n_ctx: usize) -> usize {
    n_ctx - 1 - n_ctx / 2
}

/// The file length ik's layout gives `n_ids` ids and `n_records` records of
/// `record_bytes`; `None` on overflow.
fn layout_bytes(n_ids: usize, n_records: usize, record_bytes: usize) -> Option<usize> {
    HEADER_BYTES
        .checked_add(n_ids.checked_mul(4)?)?
        .checked_add(n_records.checked_mul(record_bytes)?)
}

fn le_i32(b: &[u8], at: usize) -> i32 {
    let mut w = [0u8; 4];
    w.copy_from_slice(&b[at..at + 4]);
    i32::from_le_bytes(w)
}

fn le_f32(b: &[u8], at: usize) -> f32 {
    let mut w = [0u8; 4];
    w.copy_from_slice(&b[at..at + 4]);
    f32::from_le_bytes(w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{open_model, ref_model_path, verdict};
    use model::arch::Arch;
    use refset::family::{Family, Identity};

    /// ik prints a base run's PPL with four decimals (`%.4lf`): within 5e-5 of
    /// exp(mean NLL), which moves ln PPL by at most 5e-5 / PPL ≤ 5e-5.
    const PRINT_HALF: f64 = 5e-5;

    /// The band on ln PPL between a file's mean NLL and the PPL its run
    /// printed. A position's NLL in the file is off ik's by at most half its
    /// step plus [`F32_SLACK`], so the mean is off by at most
    /// `MAX_STEP / 2 + F32_SLACK`, whatever the signs; the print adds
    /// [`PRINT_HALF`]. A target at level 0 is outside the half-step bound on
    /// one side only — the file can understate its NLL, never overstate it —
    /// so when a file has one, the gate checks the side it cannot move.
    const PPL_BAND_LN: f64 = MAX_STEP / 2.0 + F32_SLACK + PRINT_HALF;

    /// A base file is the run that wrote it (`just gate-ds41-kld`): for each
    /// base in place of the model file's KLD family ([`kld_family`]) — or the
    /// one file `$BLOOMERY_KLD_FILE` names, judged against the run its file
    /// stem names, whose log lies beside it — the run read the model file the
    /// tree runs and ran the family's ik tree (`refset::kld::check`), the
    /// header states the run's ctx and chunks, the file has the bytes the run
    /// recorded, the vocabulary is the model file's, every record is one ik's
    /// quantizer can write, and exp(mean NLL) over the file is the PPL the run
    /// printed, within [`PPL_BAND_LN`].
    #[test]
    #[ignore = "hw: needs the ik base files in $BLOOMERY_DATA/ikppl and the V4.1 model file on the box"]
    fn hw_ds41_kld_base_matches_its_run() {
        let (model, vocab) = model_vocab().unwrap_or_else(|e| panic!("hw_ds41_kld: {e}"));
        let family = kld_family().unwrap_or_else(|e| panic!("hw_ds41_kld: {e}"));
        let files: Vec<(PathBuf, String)> = match std::env::var_os("BLOOMERY_KLD_FILE") {
            Some(p) if !p.is_empty() => {
                let p = PathBuf::from(p);
                let tag = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_else(|| panic!("hw_ds41_kld: {} has no file stem", p.display()))
                    .to_string();
                vec![(p, tag)]
            }
            _ => family
                .in_place()
                .into_iter()
                .map(|run| {
                    let tag = run
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or_else(|| panic!("hw_ds41_kld: {} has no tag", run.display()))
                        .to_string();
                    (run.with_file_name(format!("{tag}.kld")), tag)
                })
                .collect(),
        };
        let mut pass = true;
        for (file, tag) in &files {
            match judge(file, tag, &model, vocab, family) {
                Ok(ok) => pass &= ok,
                Err(e) => {
                    println!("hw_ds41_kld: {} — {e} — FAIL", file.display());
                    pass = false;
                }
            }
        }
        assert!(
            pass,
            "hw_ds41_kld: a base file disagrees with its run (lines above)"
        );
    }

    /// One base file against its run: the run's log first, refused by name
    /// when the run read another model file or ran another ik tree
    /// (`refset::kld::check`), then its result line. Prints the file's
    /// verdict line; `Ok(false)` is a check that failed, `Err` a file that
    /// cannot be judged.
    fn judge(
        file: &Path,
        tag: &str,
        model: &Path,
        vocab: usize,
        family: &Family,
    ) -> Result<bool, GateError> {
        let log = refset::kld::check(file, tag, family)?.log;
        let run = RunLine::read(&log, tag)?;
        let model_name = model.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let base = KldBase::open(file, vocab)?;
        if (base.n_ctx(), base.n_chunk()) != (run.ctx, run.chunks) {
            return Err(format!(
                "the header states ctx {} over {} chunk(s), the run ctx {} over {}",
                base.n_ctx(),
                base.n_chunk(),
                run.ctx,
                run.chunks
            )
            .into());
        }
        let bytes = std::fs::metadata(file)
            .map_err(|e| format!("{}: {e}", file.display()))?
            .len();
        if bytes != run.kld_bytes {
            return Err(format!("{bytes} bytes, but the run recorded {}", run.kld_bytes).into());
        }

        let (mut sum, mut at_floor, mut failed) = (0.0f64, 0usize, Vec::new());
        for r in base.records() {
            sum += r.nll();
            at_floor += usize::from(r.next_at_floor());
            if let Err(e) = r.check() {
                failed.push(format!("chunk {} position {}: {e}", r.chunk, r.pos));
            }
        }
        let n = base.records().len();
        if n == 0 {
            return Err("no scored position".into());
        }
        let mean = sum / n as f64;
        let d = mean - run.ppl.ln();
        let band_ok = d <= PPL_BAND_LN && (at_floor > 0 || -d <= PPL_BAND_LN);
        for f in failed.iter().take(5) {
            println!("hw_ds41_kld: FAIL record {f}");
        }
        if failed.len() > 5 {
            println!("hw_ds41_kld: ... and {} more", failed.len() - 5);
        }
        let pass = band_ok && failed.is_empty();
        println!(
            "hw_ds41_kld: {} (run {tag}: ctx {}, chunks {}, printed ppl {}; vocab {vocab} from {}) \
             — {n} positions, next at the floor {at_floor} — ppl from the file {:.6}, Δln {d:+.3e} \
             (band ±{PPL_BAND_LN:.3e}{}) — records failed {} of {n} — {}",
            file.display(),
            run.ctx,
            run.chunks,
            run.ppl_text,
            model_name,
            mean.exp(),
            if at_floor > 0 {
                ", lower side unchecked"
            } else {
                ""
            },
            failed.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The KLD family of the model file the gate opens: the one family of its
    /// architecture whose sets are ik's KL-divergence base runs.
    fn kld_family() -> Result<&'static Family, GateError> {
        let arch = Arch::detect(&open_model()?)?;
        let runs: Vec<&'static Family> = refset::arch::families(arch.name())
            .iter()
            .copied()
            .filter(|f| f.identity == Identity::RunLog)
            .collect();
        match runs[..] {
            [f] => Ok(f),
            _ => Err(format!(
                "{} KLD families of {} in the reference-set table, want one",
                runs.len(),
                arch.name()
            )
            .into()),
        }
    }

    /// The gate's model file and its vocabulary, read from the file: the
    /// length of `tokenizer.ggml.tokens`, which is ik's `n_vocab`.
    fn model_vocab() -> Result<(PathBuf, usize), GateError> {
        let path = ref_model_path()?;
        let inv = gguf::inventory_of(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        match inv.value("tokenizer.ggml.tokens") {
            Some(gguf::Value::Array(v)) => Ok((path, v.len())),
            _ => Err(format!("{}: no tokenizer.ggml.tokens array", path.display()).into()),
        }
    }

    /// The fields the gate reads from a run's result line: the last line of
    /// its log that opens with `ppl tag=<tag> ` (`tools/ref/ik-ppl.sh`).
    struct RunLine {
        ppl: f64,
        ppl_text: String,
        ctx: usize,
        chunks: usize,
        kld_bytes: u64,
    }

    impl RunLine {
        fn read(log: &Path, tag: &str) -> Result<RunLine, GateError> {
            let line = ResultLine::read(log, tag)?;
            Ok(RunLine {
                ppl: line.parse("ppl")?,
                ppl_text: line.text("ppl")?.to_string(),
                ctx: line.parse("ctx")?,
                chunks: line.parse("chunks")?,
                kld_bytes: line.parse("kld_bytes")?,
            })
        }
    }

    /// A file of one chunk of four ids over a three-entry vocabulary — an odd
    /// one, so its record carries the pad — scored at position 2 against id 1.
    fn tiny_file(ids: [i32; 4], logits: &[f32; 3]) -> Vec<u8> {
        file_of(4, &ids, &[logits])
    }

    /// A base file in ik's layout: `ids` in chunks of `n_ctx`, then one record
    /// per scored position, ik's quantizer applied to `rows` in file order.
    fn file_of(n_ctx: usize, ids: &[i32], rows: &[&[f32]]) -> Vec<u8> {
        let (n_vocab, n_chunk) = (rows[0].len(), ids.len() / n_ctx);
        assert_eq!(rows.len(), n_chunk * scored_per_chunk(n_ctx));
        let mut b = MAGIC.to_vec();
        for v in [n_ctx, n_vocab, n_chunk] {
            let v = i32::try_from(v).expect("a test file's sizes fit in i32");
            b.extend_from_slice(&v.to_le_bytes());
        }
        for v in ids {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let per = scored_per_chunk(n_ctx);
        for (k, row) in rows.iter().enumerate() {
            let (chunk, pos) = (k / per, n_ctx / 2 + k % per);
            let next =
                u32::try_from(ids[chunk * n_ctx + pos + 1]).expect("a test id is not negative");
            encode_record(row, next, &mut b).expect("a test row encodes");
        }
        b
    }

    /// A row's log-probabilities in f64, from the logits.
    fn log_softmax(logits: &[f32]) -> Vec<f64> {
        let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let sum: f64 = logits.iter().map(|&l| (f64::from(l) - max).exp()).sum();
        logits
            .iter()
            .map(|&l| f64::from(l) - max - sum.ln())
            .collect()
    }

    /// Two files of one text compared position by position against a KLD
    /// known from the logits: within the band the two records' storage
    /// derives (every entry within 24 nats of its row's maximum, so each is off
    /// by at most half its step plus [`F32_SLACK`]); the entry P puts below
    /// [`IK_KLD_CUT`] is the whole difference between the two sums; a tie at
    /// Q's top level is counted and resolved to the first id; equal rows give
    /// exactly 0, and so does a file against itself. The longer file is read
    /// on the shorter one's chunks, and a file of another `n_ctx`, another
    /// vocabulary or other ids is refused by name.
    #[test]
    fn two_files_compare_to_a_known_kld() {
        let p_rows: [&[f32]; 6] = [
            &[2.0, 0.5, -1.0, -3.0, -18.0],
            &[0.5, 1.0, 0.0, -1.0, -2.0],
            &[0.3, -0.2, 1.1, 0.4, -0.7],
            &[0.0, 0.1, 0.2, 0.3, 0.4],
            &[-1.0, 2.0, 0.0, 0.5, -0.5],
            &[1.0, 0.0, -1.0, 0.0, 1.0],
        ];
        let q_rows: [&[f32]; 3] = [
            &[1.5, 0.7, -0.8, -3.5, -17.0],
            &[1.0, 1.0, 0.0, -1.0, -2.0],
            &[0.3, -0.2, 1.1, 0.4, -0.7],
        ];
        let ids0 = [1, 2, 3, 4, 0, 1, 2, 3];
        let ids = [ids0, [4, 3, 2, 1, 0, 4, 3, 2]].concat();
        let write = |name: &str, bytes: &[u8]| {
            let p = std::env::temp_dir().join(format!(
                "bloomery-kld-{}-pair-{name}.kld",
                std::process::id()
            ));
            std::fs::write(&p, bytes).expect("the temp directory is writable");
            let r = KldBase::open_own_vocab(&p);
            let _ = std::fs::remove_file(&p);
            r.unwrap_or_else(|e| panic!("{name}: {e}"))
        };
        let p = write("p", &file_of(8, &ids, &p_rows));
        let q = write("q", &file_of(8, &ids0, &q_rows));

        let pair = Pair::new(&p, &q).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(pair.n_chunk(), 1);
        let d: Vec<PosDiff> = pair.records().map(|(a, b)| compare(&a, &b)).collect();
        assert_eq!(d.len(), 3);
        let recs: Vec<(Record<'_>, Record<'_>)> = pair.records().collect();
        for (k, (pr, qr)) in recs.iter().enumerate() {
            let (lp, lq) = (log_softmax(p_rows[k]), log_softmax(q_rows[k]));
            let ep = f64::from(pr.scale) / 2.0 + F32_SLACK;
            let eq = f64::from(qr.scale) / 2.0 + F32_SLACK;
            let (mut exact, mut band) = (0.0, 0.0);
            for (a, b) in lp.iter().zip(&lq) {
                exact += a.exp() * (a - b);
                band += a.exp() * (ep.exp_m1() * (a - b).abs() + ep.exp() * (ep + eq));
            }
            assert!(
                (d[k].kld - exact).abs() <= band,
                "position {k}: kld {} vs exact {exact}, band {band:.3e}",
                d[k].kld
            );
            assert_eq!((d[k].chunk, d[k].pos), (0, 4 + k));
            assert_eq!(i64::from(d[k].next), i64::from(ids0[5 + k]));
            assert!((d[k].p_mass - 1.0).abs() <= ep.exp_m1());
        }
        let t4 = pr_term(&recs[0], 4);
        assert!(d[0].kld > 0.0 && d[0].kld_ik_cut != d[0].kld);
        assert!((d[0].kld - d[0].kld_ik_cut - t4).abs() <= 1e-15);
        assert_eq!(d[0].cut_at_q_floor, 0);
        assert_eq!((d[1].top_p, d[1].top_q, d[1].q_top_count), (1, 0, 2));
        assert_eq!((d[2].kld, d[2].kld_ik_cut), (0.0, 0.0));
        assert_eq!(d[2].nll_p, d[2].nll_q);

        let same = Pair::new(&p, &p).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(same.n_chunk(), 2);
        for (a, b) in same.records() {
            let s = compare(&a, &b);
            assert_eq!(
                (s.kld, s.kld_ik_cut, s.top_p, s.nll_p),
                (0.0, 0.0, s.top_q, s.nll_q)
            );
        }

        let refused = |other: &KldBase, word: &str| {
            let e = Pair::new(&p, other)
                .err()
                .unwrap_or_else(|| panic!("{word}: paired"))
                .to_string();
            assert!(e.contains(word), "{word}: {e}");
        };
        refused(
            &write("ctx", &tiny_file([2, 0, 1, 1], &[0.0, -1.0, -3.0])),
            "n_ctx",
        );
        let row: &[f32] = &[0.0, 1.0, 2.0];
        let small = [0, 1, 2, 0, 1, 2, 0, 1];
        refused(&write("vocab", &file_of(8, &small, &[row; 3])), "n_vocab");
        let mut moved = ids0;
        moved[7] = 0;
        refused(&write("ids", &file_of(8, &moved, &q_rows)), "ids differ");
    }

    /// Entry `i`'s term `p·(ln p − ln q)` from the stored values of a pair of
    /// records.
    fn pr_term((p, q): &(Record<'_>, Record<'_>), i: u32) -> f64 {
        p.log_prob(i).exp() * (p.log_prob(i) - q.log_prob(i))
    }

    /// The reader decodes ik's layout — the scored position, its id, an NLL
    /// within half a step of the exact one, a record that checks — and refuses
    /// a file that is not one: a wrong magic, a length its header does not
    /// imply, another vocabulary, an id outside it; a record whose top level
    /// lost a bit fails its check.
    #[test]
    fn a_base_file_reads_back_and_a_bad_one_is_refused() {
        let logits = [0.0f32, -1.0, -30.0];
        let good = tiny_file([2, 0, 1, 1], &logits);
        let path = |name: &str| {
            std::env::temp_dir().join(format!("bloomery-kld-{}-{name}.kld", std::process::id()))
        };
        let open = |name: &str, bytes: &[u8], vocab: usize| {
            let p = path(name);
            std::fs::write(&p, bytes).expect("the temp directory is writable");
            let r = KldBase::open(&p, vocab);
            let _ = std::fs::remove_file(&p);
            r
        };

        let base = open("good", &good, 3).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((base.n_ctx(), base.n_vocab(), base.n_chunk()), (4, 3, 1));
        assert_eq!(base.tokens(), &[2, 0, 1, 1]);
        let recs: Vec<Record<'_>> = base.records().collect();
        assert_eq!(recs.len(), 1);
        let r = recs[0];
        assert_eq!((r.chunk, r.pos, r.next), (0, 2, 1));
        assert!(base.record(0, 1).is_none() && base.record(0, 3).is_none());
        let lse = (1.0 + (-1.0f64).exp() + (-30.0f64).exp()).ln();
        let exact = lse + 1.0;
        assert!(
            (r.nll() - exact).abs() <= f64::from(r.scale) / 2.0 + F32_SLACK,
            "nll {} vs exact {exact}",
            r.nll()
        );
        assert!(r.level(2) == 0 && !r.next_at_floor());
        r.check().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(r.log_probs().len(), 3);

        let refused = |name: &str, bytes: &[u8], vocab: usize, word: &str| {
            let e = open(name, bytes, vocab)
                .err()
                .unwrap_or_else(|| panic!("{name}: opened"))
                .to_string();
            assert!(e.contains(word), "{name}: {e}");
        };
        let mut magic = good.clone();
        magic[0] = b'L';
        refused("magic", &magic, 3, "magic");
        refused("short", &good[..good.len() - 1], 3, "implies");
        refused("vocab", &good, 4, "n_vocab");
        let mut id = good.clone();
        id[HEADER_BYTES + 4..HEADER_BYTES + 8].copy_from_slice(&3i32.to_le_bytes());
        refused("id", &id, 3, "outside");

        let mut flipped = good;
        let top_high_byte = HEADER_BYTES + 4 * 4 + RECORD_HEAD_BYTES + 1;
        flipped[top_high_byte] ^= 0x80;
        let base = open("flipped", &flipped, 3).unwrap_or_else(|e| panic!("{e}"));
        let r = base.records().next().expect("one record");
        assert!(
            r.check().is_err(),
            "a record whose top level lost a bit checks"
        );
    }
    /// A model whose logits are a pure function of the ids fed since its reset, over a spread wide enough that
    /// some entries of every row lie past ik's floor: which id went in at which position, and which position
    /// each record scores, shows in every record.
    struct Stub {
        vocab: usize,
        hist: Vec<u32>,
        resets: usize,
        steps: usize,
    }

    /// The logits `Stub` leaves after the ids `hist`: entry `i` of a hash of the history, in [−15, 15).
    fn stub_row(vocab: usize, hist: &[u32]) -> Vec<f32> {
        let seed = hist.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &t| {
            (h ^ u64::from(t)).wrapping_mul(0x0100_0000_01b3)
        });
        (0..vocab)
            .map(|i| {
                let x = seed.wrapping_add(i as u64 * 0x9e37_79b9_7f4a_7c15);
                let x = (x ^ (x >> 29)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                (x >> 40) as f32 / (1u64 << 24) as f32 * 30.0 - 15.0
            })
            .collect()
    }

    impl PplModel for Stub {
        fn reset(&mut self) -> Result<(), GateError> {
            self.hist.clear();
            self.resets += 1;
            Ok(())
        }

        fn step(&mut self, token: u32) -> Result<(), GateError> {
            self.hist.push(token);
            self.steps += 1;
            Ok(())
        }

        fn logits(&mut self) -> Result<Vec<f32>, GateError> {
            Ok(stub_row(self.vocab, &self.hist))
        }
    }

    fn stub(vocab: usize) -> Stub {
        Stub {
            vocab,
            hist: Vec::new(),
            resets: 0,
            steps: 0,
        }
    }

    fn temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("bloomery-kld-{}-{name}.kld", std::process::id()))
    }

    /// The loss tools' command line (`cli`): `--ids-of` is the ids a base file holds, `--ids` a file of them, and
    /// exactly one is given; a tool's own flag and switch are read beside the common ones, and a switch can end the
    /// run before the ids and runs are needed. Refused by name: a flag of no tool's, a flag with no value or given
    /// twice, both or neither way of giving the ids, a count that is no count, a run of another shape or above
    /// `--plan-ctx` or with an `OUT` that exists or is two runs', and, before the load, too few ids. The
    /// `--ids-of` ids are the base's first ones, held to the model's vocabulary by the base's own header, and a
    /// run over them writes the base the walk would from the same ids given as a file.
    #[test]
    fn a_loss_tool_cli_reads_its_flags() {
        use cli::{Args, Spec};
        const SPEC: Spec = Spec {
            tool: "tool",
            own: &["--place"],
            switches: &["--plan"],
            plan_only: Some("--plan"),
        };
        let parse = |a: &[&str]| {
            Args::parse(
                &SPEC,
                &a.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>(),
            )
        };
        let refuse = |a: &[&str], word: &str| {
            let e = parse(a)
                .err()
                .unwrap_or_else(|| panic!("{a:?}: accepted"))
                .to_string();
            assert!(e.starts_with("tool: ") && e.contains(word), "{a:?}: {e}");
        };
        let (vocab, n_ctx) = (33usize, 10usize);
        let ids: Vec<u32> = (0..2 * n_ctx)
            .map(|i| ((3 * i + 1) % vocab) as u32)
            .collect();
        let (base, out, list) = (temp("cli-base"), temp("cli-out"), temp("cli-ids"));
        write_base(&mut stub(vocab), &ids, n_ctx, vocab, &base).unwrap_or_else(|e| panic!("{e}"));
        let (b, o, l) = (
            base.display().to_string(),
            out.display().to_string(),
            list.display().to_string(),
        );
        let run = format!("{n_ctx}x2={o}");

        refuse(&["--bogus", "1"], "no flag of this tool");
        refuse(&["--ids", &l, "--run"], "--run needs a value");
        refuse(
            &["--ids", &l, "--ids", &l, "--run", &run],
            "--ids is given once",
        );
        refuse(&["--ids", &l, "--ids-of", &b, "--run", &run], "two ways");
        refuse(&["--run", &run], "--ids FILE");
        refuse(&["--ids-of", &b], "at least one --run");
        refuse(
            &["--ids-of", &b, "--plan-ctx", "x", "--run", &run],
            "no count",
        );
        refuse(
            &[
                "--ids-of", &b, "--place", "a", "--place", "b", "--run", &run,
            ],
            "--place is given once",
        );
        for bad in ["2=x.kld", "8x0=x.kld", "8x=x.kld", "8=", "x8=x.kld", "8"] {
            refuse(&["--ids-of", &b, "--run", bad], "is not CTX");
        }
        refuse(
            &["--ids-of", &b, "--plan-ctx", "4", "--run", &run],
            "above --plan-ctx 4",
        );
        refuse(
            &["--ids-of", &b, "--run", &format!("{n_ctx}={b}")],
            "exists",
        );
        refuse(
            &[
                "--ids-of",
                &b,
                "--run",
                &format!("{n_ctx}={o}"),
                "--run",
                &format!("{n_ctx}x2={o}"),
            ],
            "two runs",
        );

        let a = parse(&[
            "--ids-of",
            &b,
            "--run",
            &run,
            "--place",
            "bp",
            "--plan",
            "--plan-ctx",
            "64",
        ])
        .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            (
                a.plan_ctx(),
                a.value("--place"),
                a.value("--arm"),
                a.switch("--plan")
            ),
            (64, Some("bp"), None, true)
        );
        let planned = parse(&["--plan", "--place", "a"]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            (planned.plan_ctx(), planned.switch("--plan")),
            (cli::PLAN_CTX, true)
        );
        assert!(planned.ids().is_err(), "a plan-only line names no ids");

        // `--ids-of`: the base's first ids, its vocabulary the model's.
        let a =
            parse(&["--ids-of", &b, "--run", &format!("6={o}")]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(a.plan_ctx(), cli::PLAN_CTX);
        let held = a
            .ids()
            .unwrap_or_else(|e| panic!("{e}"))
            .for_vocab(vocab)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(held, ids[..6]);
        let e = a
            .ids()
            .and_then(|i| i.for_vocab(vocab + 1))
            .expect_err("another vocabulary")
            .to_string();
        assert!(e.contains(&format!("n_vocab {vocab}")), "{e}");
        let too_many = parse(&["--ids-of", &b, "--run", &format!("{}x1={o}", 2 * n_ctx + 2)])
            .unwrap_or_else(|e| panic!("{e}"));
        let e = too_many
            .ids()
            .err()
            .expect("more ids than the base holds")
            .to_string();
        assert!(e.contains("holds 20 ids, 22 wanted"), "{e}");

        // `--ids`: a file, its count checked before the load and its vocabulary after.
        std::fs::write(
            &list,
            ids[..8]
                .iter()
                .map(|i| format!("{i}\n"))
                .collect::<String>(),
        )
        .expect("the temp directory is writable");
        let a = parse(&["--ids", &l, "--run", &format!("8={o}")]).unwrap_or_else(|e| panic!("{e}"));
        let from_file = a.ids().unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            from_file.for_vocab(vocab).unwrap_or_else(|e| panic!("{e}")),
            ids[..8]
        );
        let e = a
            .ids()
            .and_then(|i| i.for_vocab(5))
            .expect_err("an id past a small vocabulary")
            .to_string();
        assert!(e.contains("outside 0..5"), "{e}");
        let a = parse(&["--ids", &l, "--run", &format!("9={o}")]).unwrap_or_else(|e| panic!("{e}"));
        let e = a.ids().err().expect("a file of too few ids").to_string();
        assert!(e.contains("8 ids, 9 wanted"), "{e}");

        // The walk: the base the runs name, the ids the base held.
        let a = parse(&["--ids-of", &b, "--run", &format!("{n_ctx}x2={o}")])
            .unwrap_or_else(|e| panic!("{e}"));
        let held = a
            .ids()
            .and_then(|i| i.for_vocab(vocab))
            .unwrap_or_else(|e| panic!("{e}"));
        a.walk(&mut stub(vocab), &held, vocab)
            .unwrap_or_else(|e| panic!("{e}"));
        let (same, twin) = (std::fs::read(&base), std::fs::read(&out));
        for p in [&base, &out, &list] {
            let _ = std::fs::remove_file(p);
        }
        assert!(
            same.is_ok() && same.as_ref().ok() == twin.as_ref().ok(),
            "the same ids, context and model write the same base"
        );
    }

    /// The walk writes the base ik's layout names, from the engine's rows alone: each chunk from a reset, its ids
    /// fed in order, the record at position `p` the logits left by ids `0..=p`, scored against id `p + 1` — every
    /// entry above the floor within half a step plus [`F32_SLACK`] of the exact log-softmax, an entry at level 0
    /// no higher than the floor says, the scale the floor's 24 nats over 65,535 levels, the record one that
    /// checks, an odd vocabulary padded — and `score_ppl` of the same model against the file reads KLD within a
    /// step of 0, every top id its own, and the NLL the walk printed. The writer refuses by name: a path that
    /// exists, an `n_ctx` that scores nothing, ids that are no whole number of chunks or outside the vocabulary,
    /// a row of another length or a non-finite logit, a record past the layout, a file finished short; and the
    /// ids reader a line that is no id, an id outside the vocabulary and too few ids. Mutants: the walk feeds
    /// `toks[pos + 1]`, scores from position `n_ctx/2 + 1`, or resets once; a level rounded down; `min_log_prob`
    /// without `− lse`; the floor at 16 nats.
    #[test]
    fn a_walk_writes_a_base_ik_reads() {
        let (vocab, n_ctx, n_chunk) = (33usize, 10usize, 2usize);
        let ids: Vec<u32> = (0..n_ctx * n_chunk)
            .map(|i| ((3 * i + 1) % vocab) as u32)
            .collect();
        let path = temp("walk");
        let mut m = stub(vocab);
        let w = write_base(&mut m, &ids, n_ctx, vocab, &path).unwrap_or_else(|e| panic!("{e}"));
        let base = KldBase::open(&path, vocab);
        let _ = std::fs::remove_file(&path);
        let base = base.unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((m.resets, m.steps), (n_chunk, n_chunk * (n_ctx - 1)));
        assert_eq!(
            (base.n_ctx(), base.n_vocab(), base.n_chunk()),
            (n_ctx, vocab, n_chunk)
        );
        assert_eq!(base.tokens(), &ids[..]);
        assert_eq!(w.positions, n_chunk * scored_per_chunk(n_ctx));
        assert_eq!(base.records().len(), w.positions);
        let (mut nll, mut floored) = (0.0f64, 0usize);
        for r in base.records() {
            r.check()
                .unwrap_or_else(|e| panic!("chunk {} position {}: {e}", r.chunk, r.pos));
            let at = r.chunk * n_ctx;
            let exact = log_softmax(&stub_row(vocab, &ids[at..=at + r.pos]));
            assert_eq!(r.next, ids[at + r.pos + 1]);
            let half = f64::from(r.scale) / 2.0 + F32_SLACK;
            for (i, (q, e)) in r.levels().zip(&exact).enumerate() {
                let stored = r.log_prob(i as u32);
                if q == 0 {
                    floored += 1;
                    assert!(
                        *e <= stored + half,
                        "entry {i} at the floor: exact {e}, stored {stored}"
                    );
                } else {
                    assert!(
                        (stored - e).abs() <= half,
                        "entry {i}: exact {e}, stored {stored}"
                    );
                }
            }
            let row = stub_row(vocab, &ids[at..=at + r.pos]);
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = row.iter().copied().fold(f32::INFINITY, f32::min);
            let scale = (max - min.max(max - FLOOR_NATS)) / f32::from(TOP_LEVEL);
            assert!(
                (r.scale - scale).abs() <= 1e-9,
                "scale {}, ik's {scale}: the floor is not {FLOOR_NATS} nats",
                r.scale
            );
            nll -= exact[r.next as usize];
        }
        assert!(floored > 0, "the stub's rows reach past the floor");
        let n = w.positions as f64;
        assert!(
            (w.ppl.ln() - nll / n).abs() <= MAX_STEP / 2.0 + F32_SLACK,
            "ppl {} against the exact mean NLL {}",
            w.ppl,
            nll / n
        );

        let s = score_ppl(&base, &mut stub(vocab)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((s.n, s.same_top), (w.positions, w.positions));
        assert!(
            s.kld.abs() <= MAX_STEP,
            "kld {} against its own file",
            s.kld
        );
        // A next id at level 0 stands for any smaller probability: the file then understates its NLL and
        // d = NLL_ours − NLL_file is bounded on the one side only.
        let next_floored = base.records().filter(|r| r.next_at_floor()).count();
        let band = MAX_STEP / 2.0 + F32_SLACK;
        assert!(
            s.mean_d >= -band && (next_floored > 0 || s.mean_d <= band),
            "mean d {} ({next_floored} next ids at the floor)",
            s.mean_d
        );

        let refuse = |what: &str, r: Result<(), GateError>, word: &str| {
            let e = r
                .err()
                .unwrap_or_else(|| panic!("{what}: accepted"))
                .to_string();
            assert!(e.contains(word), "{what}: {e}");
        };
        let exists = temp("exists");
        std::fs::write(&exists, b"x").expect("the temp directory is writable");
        refuse(
            "an existing path",
            KldWriter::create(&exists, n_ctx, vocab, &ids).map(|_| ()),
            "exists",
        );
        let _ = std::fs::remove_file(&exists);
        refuse(
            "n_ctx 2",
            KldWriter::create(&temp("n2"), 2, vocab, &ids[..4]).map(|_| ()),
            "at least 3",
        );
        refuse(
            "a ragged chunk",
            KldWriter::create(&temp("rag"), 4, vocab, &ids[..9]).map(|_| ()),
            "whole number",
        );
        refuse(
            "an id past the vocabulary",
            KldWriter::create(&temp("big"), 4, 5, &ids[..8]).map(|_| ()),
            "outside",
        );
        let mut ok =
            KldWriter::create(&temp("rows"), 4, vocab, &ids[..4]).unwrap_or_else(|e| panic!("{e}"));
        refuse("a short row", ok.push(&[0.0; 3]).map(|_| ()), "3 logits");
        let mut nan = vec![0.0f32; vocab];
        nan[4] = f32::NAN;
        refuse("a NaN logit", ok.push(&nan).map(|_| ()), "logit 4");
        let row = stub_row(vocab, &ids[..3]);
        ok.push(&row).unwrap_or_else(|e| panic!("{e}"));
        refuse(
            "a record past the layout",
            ok.push(&row).map(|_| ()),
            "past the 1",
        );
        let done = ok.finish().unwrap_or_else(|e| panic!("{e}"));
        let _ = std::fs::remove_file(&done);
        let short_path = temp("short");
        let short =
            KldWriter::create(&short_path, 4, vocab, &ids[..4]).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(short.written(), 0);
        refuse(
            "a short file",
            short.finish().map(|_| ()),
            "0 records written",
        );
        let _ = std::fs::remove_file(format!("{}.part", short_path.display()));

        let list = temp("ids");
        let ids_file = |text: &str| {
            std::fs::write(&list, text).expect("the temp directory is writable");
            read_ids(&list, 3, 7)
        };
        assert_eq!(
            ids_file("1\n2\n3\n4\n").unwrap_or_else(|e| panic!("{e}")),
            [1, 2, 3]
        );
        refuse(
            "a line that is no id",
            ids_file("1\nx\n3\n").map(|_| ()),
            ":2:",
        );
        refuse(
            "an id past the vocabulary",
            ids_file("1\n2\n7\n").map(|_| ()),
            "outside 0..7",
        );
        refuse(
            "too few ids",
            ids_file("1\n2\n").map(|_| ()),
            "2 ids, 3 wanted",
        );
        let _ = std::fs::remove_file(&list);
    }
}
