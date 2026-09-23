//! What the DSpark draft's Markov head alone would accept on a token stream, offline, if the
//! target were the text itself.
//!
//! The head is the checkpoint's bigram correction: for the previous token `p`, `e = markov_w1[p]`
//! (rank values) and `delta[v] = markov_w2[v] · e` for every vocabulary entry `v` (ik's
//! `ggml_mul_mat(markov_w2, get_rows(markov_w1, previous))`). With no draft body the head's
//! draft for position `i` is `argmax_v delta(t[i-1])[v]`, lowest index on a tie (`ggml_argmax`).
//! Both tensors are `[rank, vocab]` bf16 in the file: one rank-long row per token. They are
//! widened to f32 once at load (`gguf::dequant_row`, `bits << 16`, exact), each row padded with
//! zeros to a multiple of eight.
//!
//! Two paths compute the same drafts:
//! - per position: for every position `i ≥ 1` of a stream's first `--tokens` ids, the full
//!   logits of `t[i-1]`, their argmax, and the rank of `t[i]` among them (the number of entries
//!   ahead of it: greater, or equal at a lower index). Rank 0 is an accept; rank < 2 and < 4 are
//!   the top-2 and top-4 hits a k-way tree draft would catch.
//! - the table: the argmax for every previous token of the vocabulary, once. Since `delta`
//!   depends only on the previous token, a serving engine's online draft is then a lookup. The
//!   per-position argmax must equal the table's entry at every position; the run prints the
//!   mismatch count and the table's FNV-1a 64 hash over its little-endian `u32` bytes, and
//!   `--table-out` writes those bytes.
//!
//! Every logit comes from one kernel ([`logits_lanes`]) with one sum order, so the two paths'
//! logits are the same bits; they differ in how the vocabulary is cut and how partial argmaxes
//! merge. The self-test pins the kernel to a scalar twin of its sum order bit for bit, and a
//! synthetic rank-4 head's table to a known answer.
//!
//! Beside the head, the stream-learned order-1 Markov draft of `tools/ref/draft-accept.py`
//! (`markov1`: the follower of the previous token seen most often in the prefix, ties to the
//! most recent; no proposal for a token never seen as a predecessor) is recomputed on the same
//! stream, whole and by 10,000-position segments, next to the head's.
//!
//! The target is the corpus text, human-written: the numbers are how predictable the domain is
//! to the head, not the engine's acceptance, whose target is the model's greedy continuation.

use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::Instant;

use gguf::{GgmlType, Split, dequant_row};
use model::arch::dspark::{DraftHparams, names};

type RunError = Box<dyn std::error::Error>;

/// Embeddings one kernel call scores against a block of rows.
const LANES: usize = 8;
/// Kernel calls per pool dispatch: a dispatch scores `LANES * GROUPS` previous tokens.
const GROUPS: usize = 4;
const BLOCK: usize = LANES * GROUPS;
/// Vocabulary rows one kernel call scores: their f32 rows stay in L2 across the groups.
const ROWS: usize = 128;
/// Stream tokens read when `--tokens` is not given (the E5 length).
const DEFAULT_TOKENS: usize = 50_000;
/// Positions per segment of the segment table.
const SEGMENT: usize = 10_000;
const CAVEAT: &str = "target = the corpus text itself (human text), so this is the domain's \
predictability by the draft, not the engine's acceptance (target = the model's greedy continuation)";

const USAGE: &str = "usage: markov-accept <draft.gguf> <ids>... [--tokens N] [--table-out PATH]
       markov-accept --self-test
  <ids>: one decimal token id per line ($BLOOMERY_DATA/engram/corpus-<name>.ids); the first N
  (default 50000) are read, a shorter file or a non-integer line is refused";

/// The Markov head, widened: `w1` and `w2` hold `vocab` rows of `stride` f32 each, the rank's
/// values then zeros.
struct Head {
    vocab: usize,
    rank: usize,
    stride: usize,
    w1: Vec<f32>,
    w2: Vec<f32>,
}

impl Head {
    fn e(&self, p: usize) -> &[f32] {
        &self.w1[p * self.stride..(p + 1) * self.stride]
    }

    fn w2_rows(&self, rows: Range<usize>) -> &[f32] {
        &self.w2[rows.start * self.stride..rows.end * self.stride]
    }

    /// Both tensors from the draft file, checked against the shape and type `dspark::tensors`
    /// states for them.
    fn load(path: &Path) -> Result<Head, RunError> {
        let split = Split::open(path)?;
        let hp = DraftHparams::read(&split)?;
        let (vocab, rank) = (hp.n_vocab, hp.markov_rank);
        let stride = rank.div_ceil(LANES) * LANES;
        let read = |name: String| -> Result<Vec<f32>, RunError> {
            let (shard, t) = split
                .find(&name)
                .ok_or_else(|| format!("{name} is not in {}", path.display()))?;
            let want = [rank as u64, vocab as u64];
            if t.ty != GgmlType::BF16 || t.dims != want {
                return Err(
                    format!("{name}: {:?} {:?}, expected BF16 {want:?}", t.ty, t.dims).into(),
                );
            }
            let bytes = split
                .shard(shard)
                .ok_or("a tensor names a shard the split does not have")?
                .data(t)?;
            let mut out = vec![0.0f32; vocab * stride];
            for (row, src) in out
                .chunks_exact_mut(stride)
                .zip(bytes.chunks_exact(2 * rank))
            {
                dequant_row(GgmlType::BF16, src, &mut row[..rank])?;
            }
            Ok(out)
        };
        let w1 = read(names::markov_w1())?;
        let w2 = read(names::markov_w2())?;
        Ok(Head {
            vocab,
            rank,
            stride,
            w1,
            w2,
        })
    }
}

/// Logits of a block of vocabulary rows against [`LANES`] embeddings:
/// `out[r * LANES + p] = w2[r] · e[p]`. Every dot has one sum order whatever block it is
/// computed in: one 8-lane accumulator, fused multiply-add per lane over the octets ascending,
/// then the lane tree `((a0+a4)+(a1+a5)) + ((a2+a6)+(a3+a7))`. [`scalar_dot`] is that order
/// in scalar code.
///
/// # Safety
/// The CPU must support AVX2 and FMA; `stride` is a multiple of 8, `e.len() == LANES * stride`,
/// `w2.len() == out.len() / LANES * stride`.
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn logits_lanes(w2: &[f32], e: &[f32], stride: usize, out: &mut [f32]) {
    // SAFETY: the fn contract above — ISA from the caller's detection; every load reads eight
    // floats at `row * stride + k` with `row` below the row count of its slice and
    // `k + 8 <= stride`, so it stays inside `w2` or `e`.
    unsafe {
        use std::arch::x86_64::*;
        let (wp, ep) = (w2.as_ptr(), e.as_ptr());
        for (r, o) in out.as_chunks_mut::<LANES>().0.iter_mut().enumerate() {
            let mut acc = [_mm256_setzero_ps(); LANES];
            for k in (0..stride).step_by(8) {
                let w = _mm256_loadu_ps(wp.add(r * stride + k));
                for (p, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(w, _mm256_loadu_ps(ep.add(p * stride + k)), *a);
                }
            }
            for (a, v) in acc.iter().zip(o.iter_mut()) {
                let c = _mm_add_ps(_mm256_castps256_ps128(*a), _mm256_extractf128_ps(*a, 1));
                let t = _mm_add_ps(c, _mm_movehdup_ps(c));
                *v = _mm_cvtss_f32(_mm_add_ps(t, _mm_movehl_ps(t, t)));
            }
        }
    }
}

/// [`logits_lanes`] behind its checks: the ISA and every length.
fn logits(w2: &[f32], e: &[f32], stride: usize, out: &mut [f32]) {
    assert!(
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"),
        "markov-accept needs AVX2 and FMA"
    );
    assert!(stride.is_multiple_of(8) && e.len() == LANES * stride);
    assert!(out.len().is_multiple_of(LANES) && w2.len() == out.len() / LANES * stride);
    // SAFETY: the ISA and the lengths the kernel's contract names were asserted just above.
    unsafe { logits_lanes(w2, e, stride, out) }
}

/// One dot in [`logits_lanes`]'s sum order, scalar: lane `l` accumulates elements `8i + l`
/// with `mul_add`, then the same lane tree.
fn scalar_dot(w: &[f32], e: &[f32]) -> f32 {
    let mut a = [0.0f32; 8];
    for (wc, ec) in w.as_chunks::<8>().0.iter().zip(e.as_chunks::<8>().0) {
        for l in 0..8 {
            a[l] = wc[l].mul_add(ec[l], a[l]);
        }
    }
    ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))
}

/// A previous token's draft over part of the vocabulary: the first maximum and, when a target
/// is scored, the entries ahead of the target.
#[derive(Clone, Copy, Debug)]
struct Partial {
    best: f32,
    arg: u32,
    ahead: u32,
}

impl Partial {
    const EMPTY: Partial = Partial {
        best: f32::NEG_INFINITY,
        arg: u32::MAX,
        ahead: 0,
    };

    /// `self` covers lower vocabulary indices than `later`.
    fn merge(&mut self, later: &Partial) {
        if later.best > self.best || self.arg == u32::MAX {
            self.best = later.best;
            self.arg = later.arg;
        }
        self.ahead += later.ahead;
    }
}

/// One previous token to score, and the target whose rank is counted, with the target's logit.
#[derive(Clone, Copy)]
struct Query {
    prev: u32,
    target: Option<(u32, f32)>,
}

/// Scores up to [`BLOCK`] queries against the whole vocabulary in one pool dispatch, cut by
/// vocabulary into one contiguous range per participant. Partials merge in range order, so a
/// tie goes to the lowest index as within a range.
fn score_block(head: &Head, queries: &[Query]) -> Vec<Partial> {
    assert!(!queries.is_empty() && queries.len() <= BLOCK);
    let stride = head.stride;
    let mut e = vec![0.0f32; BLOCK * stride];
    for (q, dst) in (0..BLOCK).zip(e.chunks_exact_mut(stride)) {
        let prev = queries[q.min(queries.len() - 1)].prev as usize;
        dst.copy_from_slice(head.e(prev));
    }
    let pool = threads::pool();
    let n = pool.threads();
    let slots: Vec<Mutex<[Partial; BLOCK]>> = (0..n)
        .map(|_| Mutex::new([Partial::EMPTY; BLOCK]))
        .collect();
    pool.for_each_chunk(n, |parts| {
        for part in parts {
            let (v0, v1) = threads::chunk_bounds(head.vocab, n, part);
            let mut local = [Partial::EMPTY; BLOCK];
            let mut out = vec![0.0f32; ROWS * LANES];
            let mut r0 = v0;
            while r0 < v1 {
                let r1 = (r0 + ROWS).min(v1);
                let out = &mut out[..(r1 - r0) * LANES];
                for (g, eg) in e.chunks_exact(LANES * stride).enumerate() {
                    logits(head.w2_rows(r0..r1), eg, stride, out);
                    for (r, row) in out.as_chunks::<LANES>().0.iter().enumerate() {
                        let v = (r0 + r) as u32;
                        for (p, &x) in row.iter().enumerate() {
                            let l = &mut local[g * LANES + p];
                            if x > l.best || l.arg == u32::MAX {
                                l.best = x;
                                l.arg = v;
                            }
                            if let Some(Query {
                                target: Some((t, lt)),
                                ..
                            }) = queries.get(g * LANES + p)
                                && (x > *lt || (x == *lt && v < *t))
                            {
                                l.ahead += 1;
                            }
                        }
                    }
                }
                r0 = r1;
            }
            *slots[part].lock().unwrap_or_else(|p| p.into_inner()) = local;
        }
    });
    let mut merged = [Partial::EMPTY; BLOCK];
    for slot in &slots {
        let s = slot.lock().unwrap_or_else(|p| p.into_inner());
        for (m, l) in merged.iter_mut().zip(s.iter()) {
            m.merge(l);
        }
    }
    merged[..queries.len()].to_vec()
}

/// The logit of one target under one previous token, by the kernel itself (so it is the bits
/// the dispatch compares against).
fn target_logit(head: &Head, prev: usize, target: usize) -> f32 {
    let stride = head.stride;
    let mut e = vec![0.0f32; LANES * stride];
    e[..stride].copy_from_slice(head.e(prev));
    let mut out = [0.0f32; LANES];
    logits(head.w2_rows(target..target + 1), &e, stride, &mut out);
    out[0]
}

/// The argmax for every previous token of the vocabulary.
fn argmax_table(head: &Head) -> Vec<u32> {
    let mut table = Vec::with_capacity(head.vocab);
    let mut queries = Vec::with_capacity(BLOCK);
    for start in (0..head.vocab).step_by(BLOCK) {
        queries.clear();
        queries.extend((start..(start + BLOCK).min(head.vocab)).map(|p| Query {
            prev: p as u32,
            target: None,
        }));
        table.extend(score_block(head, &queries).iter().map(|q| q.arg));
    }
    table
}

/// The per-position path over one stream: `(argmax, rank of the target)` for positions
/// `1..ids.len()`.
fn per_position(head: &Head, ids: &[u32]) -> Vec<(u32, u32)> {
    let mut out = Vec::with_capacity(ids.len().saturating_sub(1));
    let mut queries = Vec::with_capacity(BLOCK);
    for start in (1..ids.len()).step_by(BLOCK) {
        queries.clear();
        queries.extend((start..(start + BLOCK).min(ids.len())).map(|i| {
            let (prev, target) = (ids[i - 1], ids[i]);
            let lt = target_logit(head, prev as usize, target as usize);
            Query {
                prev,
                target: Some((target, lt)),
            }
        }));
        out.extend(score_block(head, &queries).iter().map(|q| (q.arg, q.ahead)));
    }
    out
}

/// `tools/ref/draft-accept.py`'s `markov1`, causally: `Some(accepted)` per position `1..n`, or
/// `None` where the previous token had never been a predecessor.
fn markov1(ids: &[u32]) -> Vec<Option<bool>> {
    // prev -> (follower -> (count, last position)), and prev -> (best follower, count).
    let mut counts: HashMap<u32, HashMap<u32, (u32, usize)>> = HashMap::new();
    let mut best: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut out = Vec::with_capacity(ids.len().saturating_sub(1));
    for i in 1..ids.len() {
        let (prev, next) = (ids[i - 1], ids[i]);
        out.push(best.get(&prev).map(|&(b, _)| b == next));
        let c = counts
            .entry(prev)
            .or_default()
            .entry(next)
            .or_insert((0, 0));
        c.0 += 1;
        c.1 = i;
        // The follower just seen is the most recent, so it wins every count tie.
        let b = best.entry(prev).or_insert((next, 0));
        if c.0 >= b.1 {
            *b = (next, c.0);
        }
    }
    out
}

fn read_ids(path: &Path, n: usize, vocab: usize) -> Result<Vec<u32>, RunError> {
    let text = std::fs::read_to_string(path)?;
    let mut ids = Vec::with_capacity(n);
    for (lineno, line) in text.lines().enumerate() {
        if ids.len() == n {
            break;
        }
        let s = line.trim();
        let id: u32 = s
            .parse()
            .ok()
            .filter(|_| s.bytes().all(|b| b.is_ascii_digit()))
            .ok_or_else(|| {
                format!(
                    "{}:{}: not a token id: {line:?}",
                    path.display(),
                    lineno + 1
                )
            })?;
        if id as usize >= vocab {
            return Err(format!(
                "{}:{}: id {id} is outside the vocabulary of {vocab}",
                path.display(),
                lineno + 1
            )
            .into());
        }
        ids.push(id);
    }
    if ids.len() < n {
        return Err(format!(
            "{} holds {} tokens, fewer than the {n} asked for",
            path.display(),
            ids.len()
        )
        .into());
    }
    Ok(ids)
}

/// FNV-1a 64 over the table's little-endian bytes.
fn fnv1a64(table: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in table.iter().flat_map(|v| v.to_le_bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn corpus_name(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
    stem.strip_prefix("corpus-").unwrap_or(stem).to_string()
}

/// One stream's counts.
struct Stream {
    name: String,
    positions: usize,
    acc: usize,
    top2: usize,
    top4: usize,
    table_mismatch: usize,
    m1_proposed: usize,
    m1_acc: usize,
    /// Per segment: (positions, head accepts, markov1 accepts).
    segments: Vec<(usize, usize, usize)>,
    wall_s: f64,
}

fn run_stream(head: &Head, table: &[u32], path: &Path, n: usize) -> Result<Stream, RunError> {
    let ids = read_ids(path, n, head.vocab)?;
    let t0 = Instant::now();
    let drafts = per_position(head, &ids);
    let wall_s = t0.elapsed().as_secs_f64();
    let m1 = markov1(&ids);
    let mut s = Stream {
        name: corpus_name(path),
        positions: drafts.len(),
        acc: 0,
        top2: 0,
        top4: 0,
        table_mismatch: 0,
        m1_proposed: 0,
        m1_acc: 0,
        segments: vec![(0, 0, 0); drafts.len().div_ceil(SEGMENT)],
        wall_s,
    };
    for (j, (&(arg, ahead), m)) in drafts.iter().zip(&m1).enumerate() {
        let i = j + 1;
        let hit = ahead == 0;
        if hit != (arg == ids[i]) {
            return Err(format!(
                "{}: position {i}: rank {ahead} but argmax {arg} vs target {}",
                s.name, ids[i]
            )
            .into());
        }
        s.acc += usize::from(hit);
        s.top2 += usize::from(ahead < 2);
        s.top4 += usize::from(ahead < 4);
        s.table_mismatch += usize::from(table[ids[i - 1] as usize] != arg);
        s.m1_proposed += usize::from(m.is_some());
        let m1_hit = *m == Some(true);
        s.m1_acc += usize::from(m1_hit);
        let seg = &mut s.segments[j / SEGMENT];
        seg.0 += 1;
        seg.1 += usize::from(hit);
        seg.2 += usize::from(m1_hit);
    }
    Ok(s)
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}

fn run(args: &[String]) -> Result<(), RunError> {
    let mut paths = Vec::new();
    let mut tokens = DEFAULT_TOKENS;
    let mut table_out = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tokens" => {
                tokens = it.next().ok_or(USAGE)?.parse()?;
            }
            "--table-out" => table_out = Some(it.next().ok_or(USAGE)?.clone()),
            s if s.starts_with("--") => return Err(format!("unknown flag {s}\n{USAGE}").into()),
            s => paths.push(s.to_string()),
        }
    }
    if paths.len() < 2 || tokens < 2 {
        return Err(USAGE.into());
    }
    let draft = Path::new(&paths[0]);
    let t0 = Instant::now();
    let head = Head::load(draft)?;
    println!(
        "markov-accept: {}: markov_w1/w2 rank {} x vocab {}, widened to f32 in {:.2} s; pool {} threads",
        draft.display(),
        head.rank,
        head.vocab,
        t0.elapsed().as_secs_f64(),
        threads::pool().threads()
    );

    let t1 = Instant::now();
    let table = argmax_table(&head);
    let table_s = t1.elapsed().as_secs_f64();
    let hash = fnv1a64(&table);
    let distinct = {
        let mut seen = vec![false; head.vocab];
        table.iter().for_each(|&a| seen[a as usize] = true);
        seen.iter().filter(|&&s| s).count()
    };
    println!(
        "markov-accept: argmax table: {} entries, {} bytes as u32, fnv1a64 {hash:016x}, {distinct} distinct drafts, {table_s:.2} s",
        table.len(),
        4 * table.len()
    );
    if let Some(p) = &table_out {
        let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes)?;
        println!("markov-accept: table written to {p}");
    }

    let mut streams = Vec::new();
    for p in &paths[1..] {
        streams.push(run_stream(&head, &table, Path::new(p), tokens)?);
    }

    println!(
        "\nFirst {tokens} tokens of each stream, one document. positions = tokens with a previous token. \
         The head always proposes, so proposed = positions.\n"
    );
    println!("### Markov head alone — {CAVEAT}\n");
    println!(
        "| corpus | positions | acc/positions | top-2 | top-4 | markov1 recomputed (acc/positions) | table mismatches | wall s |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for s in &streams {
        println!(
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {:.2} |",
            s.name,
            s.positions,
            ratio(s.acc, s.positions),
            ratio(s.top2, s.positions),
            ratio(s.top4, s.positions),
            ratio(s.m1_acc, s.positions),
            s.table_mismatch,
            s.wall_s
        );
    }
    println!("\n### By {SEGMENT}-position segment, acc/positions head / markov1 — {CAVEAT}\n");
    let nseg = streams.iter().map(|s| s.segments.len()).max().unwrap_or(0);
    let head_row: String = (0..nseg).map(|k| format!(" seg {k} |")).collect();
    println!("| corpus |{head_row}");
    println!("|---|{}", "---:|".repeat(nseg));
    for s in &streams {
        let cells: String = s
            .segments
            .iter()
            .map(|&(n, h, m)| format!(" {:.3} / {:.3} |", ratio(h, n), ratio(m, n)))
            .collect();
        println!("| {} |{cells}", s.name);
    }
    println!();
    for s in &streams {
        println!(
            "markov-accept: {}: accepted {}, top-2 {}, top-4 {}; markov1 proposed {} accepted {}; table mismatches {}",
            s.name, s.acc, s.top2, s.top4, s.m1_proposed, s.m1_acc, s.table_mismatch
        );
    }
    let mismatches: usize = streams.iter().map(|s| s.table_mismatch).sum();
    if mismatches != 0 {
        return Err(format!(
            "the table disagrees with the per-position path at {mismatches} positions"
        )
        .into());
    }
    println!("markov-accept: table = per-position argmax at every position — PASS");
    Ok(())
}

/// A deterministic generator for the self-test's data.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn check(ok: bool, what: &str) -> Result<(), RunError> {
    if ok {
        println!("markov-accept: self-test {what} — ok");
        Ok(())
    } else {
        Err(format!("self-test failed: {what}").into())
    }
}

fn self_test() -> Result<(), RunError> {
    // (i) The kernel against its scalar twin, bit for bit, at two strides and odd row counts.
    let mut g = SplitMix(1);
    for (stride, rows) in [(16usize, 13usize), (40, 5), (256, 3)] {
        let w2: Vec<f32> = (0..rows * stride).map(|_| g.unit()).collect();
        let e: Vec<f32> = (0..LANES * stride).map(|_| g.unit()).collect();
        let mut out = vec![0.0f32; rows * LANES];
        logits(&w2, &e, stride, &mut out);
        let same = (0..rows).all(|r| {
            (0..LANES).all(|p| {
                let s = scalar_dot(
                    &w2[r * stride..(r + 1) * stride],
                    &e[p * stride..(p + 1) * stride],
                );
                s.to_bits() == out[r * LANES + p].to_bits()
            })
        });
        check(
            same,
            &format!("kernel = scalar sum order, bit for bit (stride {stride}, {rows} rows)"),
        )?;
    }

    // (ii) A synthetic rank-4 head over a vocabulary of 144. Token t embeds at angle
    // 2π(t mod 16)/16; vocabulary rows 9r..9r+8 all sit at the angle of f⁻¹(r),
    // f(p) = (5p + 3) mod 16, so the dot peaks at nine equal rows and is at most cos(π/8)
    // elsewhere; dims 2 and 3 add a constant. The nine-row ties span the pool's vocabulary cut
    // at any thread count up to 36 and sit inside a range at every count, so both tie rules (in
    // a range and in the merge) must give the lowest index: 9 f(t mod 16).
    let (vocab, rank, dup) = (144usize, 4usize, 9usize);
    let f = |p: usize| (5 * p + 3) % 16;
    let finv = |v: usize| (0..16).find(|&p| f(p) == v).unwrap_or(0);
    let angle = |p: usize| 2.0 * std::f32::consts::PI * p as f32 / 16.0;
    let stride = rank.div_ceil(LANES) * LANES;
    let mut w1 = vec![0.0f32; vocab * stride];
    let mut w2 = vec![0.0f32; vocab * stride];
    for t in 0..vocab {
        let a = angle(t % 16);
        w1[t * stride..t * stride + rank].copy_from_slice(&[a.cos(), a.sin(), 0.5, -0.25]);
        let b = angle(finv(t / dup));
        w2[t * stride..t * stride + rank].copy_from_slice(&[b.cos(), b.sin(), 0.5, 1.0]);
    }
    let head = Head {
        vocab,
        rank,
        stride,
        w1,
        w2,
    };
    let known: Vec<u32> = (0..vocab).map(|p| (dup * f(p % 16)) as u32).collect();
    let table = argmax_table(&head);
    check(
        table == known,
        &format!(
            "synthetic table = known answer 9 f(p mod 16), {vocab} entries, ties to the lowest"
        ),
    )?;
    check(
        fnv1a64(&[]) == 0xcbf2_9ce4_8422_2325 && fnv1a64(&[0x6463_6261]) == 0xfc17_9f83_ee07_24dd,
        "fnv1a64 of \"\" and \"abcd\"",
    )?;

    // (iii) The per-position path on a stream: argmax against the table, the target's rank
    // against a brute-force count over the scalar logits (greater, or equal at a lower index).
    let ids: Vec<u32> = (0..300).map(|_| (g.next() % vocab as u64) as u32).collect();
    let drafts = per_position(&head, &ids);
    let mut rank_ok = true;
    for (j, &(arg, ahead)) in drafts.iter().enumerate() {
        let (p, t) = (ids[j] as usize, ids[j + 1] as usize);
        let l: Vec<f32> = (0..vocab)
            .map(|v| scalar_dot(&head.w2[v * stride..(v + 1) * stride], head.e(p)))
            .collect();
        let brute = (0..vocab)
            .filter(|&v| l[v] > l[t] || (l[v] == l[t] && v < t))
            .count();
        rank_ok &= brute == ahead as usize && arg == table[p];
    }
    check(
        rank_ok && drafts.len() == ids.len() - 1,
        "per-position argmax = table, rank = brute-force count, 299 positions",
    )?;

    // (iv) markov1 on [0 1 2 0 1 3] repeated: after 1 the followers 2 and 3 alternate, so every
    // proposal after 1 is the one seen last and misses; 3 is no predecessor before position 6.
    let pat = [0u32, 1, 2, 0, 1, 3];
    let s: Vec<u32> = pat.iter().cycle().take(18).copied().collect();
    let m = markov1(&s);
    let (t, f, n) = (Some(true), Some(false), None);
    let expect = vec![n, n, n, t, f, n, t, f, t, t, f, t, t, f, t, t, f];
    check(m == expect, "markov1 = hand-counted proposals")?;
    println!(
        "markov-accept: self-test ok (pool {} threads)",
        threads::pool().threads()
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = if args.first().map(String::as_str) == Some("--self-test") {
        self_test()
    } else {
        run(&args)
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("markov-accept: {e}");
            ExitCode::FAILURE
        }
    }
}
