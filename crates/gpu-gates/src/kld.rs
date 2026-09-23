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
    use crate::{data_dir, ref_model_path, verdict};

    /// The base files the gate reads by default, as `ik-ppl.sh --kld-base`
    /// tags: the c2048 × 4 run the PR's PPL is quoted at, and the c512 × 16 run
    /// the early comparison reads.
    const BASES: [&str; 2] = ["kldbase-c2048x4", "kldbase-c512x16"];

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
    /// of [`BASES`] — or the one file `$BLOOMERY_KLD_FILE` names, judged
    /// against the run its file stem names — the header states the run's ctx
    /// and chunks, the file has the bytes the run recorded, the vocabulary is
    /// the model file's, every record is one ik's quantizer can write, and
    /// exp(mean NLL) over the file is the PPL the run printed, within
    /// [`PPL_BAND_LN`].
    #[test]
    #[ignore = "hw: needs the ik base files in $BLOOMERY_DATA/ikppl and the V4.1 model file on the box"]
    fn hw_ds41_kld_base_matches_its_run() {
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
            _ => BASES
                .iter()
                .map(|t| {
                    (
                        data_dir().join("ikppl").join(format!("{t}.kld")),
                        t.to_string(),
                    )
                })
                .collect(),
        };
        let (model, vocab) = model_vocab().unwrap_or_else(|e| panic!("hw_ds41_kld: {e}"));
        let mut pass = true;
        for (file, tag) in &files {
            match judge(file, tag, &model, vocab) {
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

    /// One base file against its run's result line. Prints the file's verdict
    /// line; `Ok(false)` is a check that failed, `Err` a file that cannot be
    /// judged.
    fn judge(file: &Path, tag: &str, model: &Path, vocab: usize) -> Result<bool, GateError> {
        let log = data_dir().join("ikppl").join(format!("{tag}.log"));
        let run = RunLine::read(&log, tag)?;
        let model_name = model.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if run.model != model_name {
            return Err(format!(
                "the run read {}, but the gate's model file is {}",
                run.model,
                model.display()
            )
            .into());
        }
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
        model: String,
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
                model: line.text("model")?.to_string(),
            })
        }
    }

    /// ik's quantizer for one position (`log_softmax`, the overload that
    /// writes levels): the record's scale, `min_log_prob` and levels.
    fn quantize(logits: &[f32]) -> (f32, f32, Vec<u16>) {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = logits
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .max(max - FLOOR_NATS);
        let sum: f64 = logits.iter().map(|&l| f64::from((l - max).exp())).sum();
        let lse = sum.ln() as f32;
        let scale = (max - min) / f32::from(TOP_LEVEL);
        let inv = 1.0 / scale;
        let levels = logits
            .iter()
            .map(|&l| {
                if l > min {
                    (inv * (l - min)).round() as u16
                } else {
                    0
                }
            })
            .collect();
        (scale, min - max - lse, levels)
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
        for row in rows {
            let (scale, mlp, levels) = quantize(row);
            b.extend_from_slice(&scale.to_le_bytes());
            b.extend_from_slice(&mlp.to_le_bytes());
            for q in levels
                .into_iter()
                .chain(std::iter::repeat_n(0, n_vocab % 2))
            {
                b.extend_from_slice(&q.to_le_bytes());
            }
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
}
