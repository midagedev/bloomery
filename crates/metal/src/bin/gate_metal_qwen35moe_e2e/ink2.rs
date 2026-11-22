//! The `inkling2` row: the family's session (`bloomery_metal::arch::inkling2`)
//! on its fixture file, against the vendor tree's node dumps of that file
//! (refset family `fx-lcpp-inkling2`, set `a`: 35 prompt ids, every visit's
//! nodes at every position, and the greedy table that follows them).
//!
//! The clauses:
//!
//! - `k_taps`: the set's prompt replayed (`Session::step_replayed`): every
//!   visit and the head start from the vendor's own rows and run the vendor's
//!   routes, so each node's error is its own visit's, every node read back is
//!   held to the set's row of that position within its band ([`band_of`]),
//!   and the last position's logits to `result_output`; the router's own kept
//!   ids as a set, a part admitted only where the vendor's own eighth and
//!   ninth biased scores lie within [`TIE`] of each other. A free walk is no
//!   oracle here: the fixture's visits amplify each input's rounding in turn
//!   (`--chain` prints a walk from the vendor's embedded rows against the
//!   same walk from those rows rounded once), and one expert swapped at a
//!   near-tie moves a visit's FFN by about √(2/8) of it.
//! - `k_greedy`: each set's replayed prompt, its pick against the first id
//!   of `greedy.tsv`; a part where the vendor's top-two margin is
//!   [`MARGIN_FLOOR`] or more is red.
//! - `k_visits`: the session's walk is the vendor's: as many visits as the
//!   set has `l_out` slots, the block boundaries where the set has no
//!   `attn_out`, the loop's message visits where it has `loop_q_inp`.
//! - `k_room`: the open's device bytes are the plan's, to within a page a
//!   buffer, and the room line the plan prints.
//!
//! The set is the GGUF fixture's, the one file whose every expert the vendor
//! tree reads (the MLX cut holds 24 of the 128); the qwen rows' MLX-only rule
//! (`sets.rs`) is theirs.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use bloomery_device::DeviceError;
use bloomery_device::conformance::{Clause, Report, Verdict, why};
use bloomery_metal::arch::inkling2::check::VisitPlan;
use bloomery_metal::arch::inkling2::{Session, Tap};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use bloomery_metal::session::room::PAGE;
use model_hf::ModelFile;
use refset::arch::inkling2::fixture::{A, B, C, D, LCPP};
use refset::ik::{Layout, RefManifest, RowKind, load_ref_logical_in, ref_ints};

use super::{GATE, flag, mutants};

/// The row's clauses, in the order they run.
pub const NAMES: [&str; 4] = ["k_taps", "k_greedy", "k_visits", "k_room"];

/// The cache rows of the session: the vendor's own `-c`.
const CTX: usize = 4096;

/// The classification floor of a first part (`gpu-gates` `prompts::MARGIN_FLOOR`,
/// PIN(2026-09-21)): a part where the vendor's top-two margin is under it is a
/// near-tie re-lottery of two engines' roundings, not a fault.
const MARGIN_FLOOR: f32 = 0.5;

/// [derived] The biased-score gap under which the vendor's eighth and ninth
/// picks may trade places: the router reads the FFN norm, which a replayed
/// visit holds to [`ATTN_BAND`]; a 2,304-wide f32 dot at that relative size
/// moves a logit of a few units by ~1e-2 of it, and the sigmoid's slope of at
/// most 1/4 takes a score to under 1e-2.
const TIE: f32 = 1e-2;

/// [derived] A replayed visit's attention nodes against the vendor's (each
/// visit starts from the vendor's own rows, so the error is the visit's own):
/// the vendor rounds every quantized projection's input to 32-value blocks of
/// int8 (a step of absmax/127, an rms error of 2.1/127/√12 = 4.8e-3 of a
/// Gaussian block's rms) where the session rounds none. q, rel, gate, k and v
/// read one rounded input; the attention's output carries the v and gate
/// shares (√2 · 4.8e-3 = 6.8e-3) and `wo` rounds once more (8.3e-3); twice
/// that.
const ATTN_BAND: f64 = 2e-2;

/// [derived] A replayed visit's FFN nodes: the gate·up input and each
/// expert's down input rounded on the vendor side (4.8e-3 each), the gate and
/// up shares multiplied in the SITU (√2 each), over the attention's carried
/// 8.3e-3: √(2 · 8.3² + 2 · 4.8² + 4.8²)e-3 = 1.5e-2; four times that, the
/// card gate's leg band (`gate_inkling2_moe` OUT_BAND) on the same vendor.
const FFN_BAND: f64 = 6e-2;

/// [derived] The embedded row: one Q8_0 row dequantized exactly and an f64
/// norm against the vendor's f32 one, a few ulps.
const EMBED_BAND: f64 = 1e-5;

/// [derived] The head's logits from the vendor's last rows: the head's input
/// rounded once on the vendor side (4.8e-3) over the mix's f32 order; four
/// times that.
const LOGITS_BAND: f64 = 2e-2;

/// The band of `tap`.
fn band_of(tap: Tap) -> f64 {
    match tap {
        Tap::Embed => EMBED_BAND,
        Tap::Q | Tap::K | Tap::V | Tap::R | Tap::Kqv | Tap::AttnOut | Tap::FfnInp | Tap::LoopQ => {
            ATTN_BAND
        }
        Tap::FfnNorm | Tap::Probs | Tap::TopK | Tap::Weights | Tap::LoopFfn => ATTN_BAND,
        Tap::MoeOut | Tap::FfnOut | Tap::LOut => FFN_BAND,
    }
}

/// `‖a − b‖₂ / ‖b‖₂` in f64.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        num += (f64::from(x) - f64::from(y)).powi(2);
        den += f64::from(y).powi(2);
    }
    if !num.is_finite() {
        return f64::INFINITY;
    }
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// The set's node name of `tap` at visit `vi`, and its occurrence: the FFN
/// norm is the visit's second norm on a plain windowed visit and its fourth
/// on a windowless or message visit (three attention norms before it); the
/// embedded row is the graph's first `norm`.
fn node_of(tap: Tap, vi: usize, windowed: bool, message: bool) -> (String, u32) {
    let occ = match tap {
        Tap::FfnNorm if windowed && !message => 1,
        Tap::FfnNorm => 3,
        _ => 0,
    };
    match tap {
        Tap::Embed => ("norm".to_string(), 0),
        _ => (format!("{}-{vi}", tap.name()), occ),
    }
}

/// Set `name` of the family, opened through it.
fn set(name: &str) -> Result<RefManifest, String> {
    RefManifest::open(&LCPP.path(name), &LCPP).map_err(|e| format!("{name}: {e}"))
}

/// The set's prompt ids.
fn prompt(man: &RefManifest) -> Result<Vec<u32>, String> {
    let (_, tokens, _) = man.step().map_err(|e| e.to_string())?;
    Ok(tokens.to_vec())
}

/// The fixture file's path.
fn model_path() -> Result<PathBuf, String> {
    refset::fixture::first_shard("inkling2")
        .map(PathBuf::from)
        .map_err(|e| format!("the fixture file: {e}"))
}

/// The worst node of one kind: its relative L2, position and visit.
#[derive(Clone, Copy)]
struct Worst {
    rel: f64,
    pos: usize,
    vi: usize,
}

/// The vendor's rows of one node, every position's row after the last, read
/// once.
struct Rows<'m> {
    man: &'m RefManifest,
    cache: HashMap<(String, u32), Vec<f32>>,
}

impl Rows<'_> {
    fn row(&mut self, name: &str, occ: u32, pos: usize, len: usize) -> Result<&[f32], String> {
        let key = (name.to_string(), occ);
        if !self.cache.contains_key(&key) {
            let (_, vals) =
                load_ref_logical_in(self.man, name, occ).map_err(|e| format!("{name}: {e}"))?;
            self.cache.insert(key.clone(), vals);
        }
        let vals = &self.cache[&key];
        vals.get(pos * len..(pos + 1) * len).ok_or_else(|| {
            format!(
                "{name}/{occ} holds {} values, not a row of {len} at position {pos}",
                vals.len()
            )
        })
    }
}

/// The comparisons of one replay: every node's worst and the first past its
/// band.
#[derive(Default)]
struct Tally {
    /// Each node kind's worst, by the kind's name, with its band.
    worst: Vec<(&'static str, f64, Worst)>,
    first_red: Option<String>,
}

impl Tally {
    fn take(&mut self, tap: Tap, name: &str, at: (usize, usize), rel: f64, detail: usize) {
        let kind = match tap {
            Tap::Embed => "embedded norm",
            Tap::FfnNorm => "ffn norm",
            t => t.name(),
        };
        self.take_kind(kind, band_of(tap), name, at, rel, detail);
    }

    fn take_kind(
        &mut self,
        kind: &'static str,
        band: f64,
        name: &str,
        at: (usize, usize),
        rel: f64,
        detail: usize,
    ) {
        let (pos, vi) = at;
        if pos < detail {
            println!("  pos {pos} visit {vi:2} {name:>26}: rel {rel:.3e} (band {band:.0e})");
        }
        if rel > band && self.first_red.is_none() {
            self.first_red = Some(format!(
                "{name} at position {pos}, visit {vi}: rel {rel:.3e} > {band:.0e}"
            ));
        }
        match self.worst.iter_mut().find(|(k, ..)| *k == kind) {
            Some((.., w)) if w.rel >= rel => {}
            Some((.., w)) => *w = Worst { rel, pos, vi },
            None => self.worst.push((kind, band, Worst { rel, pos, vi })),
        }
    }
}

/// The vendor's chain as the walk reads it: the stream before each visit and
/// the blocks closed so far, re-derived from the visits' plan (a block closes
/// the stream a boundary visit starts from; a pass the loop runs again keeps
/// the prelude's blocks; a later pass starts from the prelude's output), and
/// the routes every visit took.
struct Chain {
    /// `l_out-<slot>` of every visit: every position's row.
    outs: Vec<Vec<f32>>,
    /// The embedded rows.
    embed: Vec<f32>,
    /// Every visit's kept ids and scaled weights at every position.
    routes: Vec<(Vec<i64>, Vec<f32>)>,
    n: usize,
}

impl Chain {
    fn open(man: &RefManifest, visits: usize, n: usize) -> Result<Chain, String> {
        let load = |name: &str| -> Result<Vec<f32>, String> {
            load_ref_logical_in(man, name, 0)
                .map(|(_, v)| v)
                .map_err(|e| format!("{name}: {e}"))
        };
        let mut outs = Vec::with_capacity(visits);
        let mut routes = Vec::with_capacity(visits);
        for vi in 0..visits {
            outs.push(load(&format!("l_out-{vi}"))?);
            let topk = format!("{}-{vi}", Tap::TopK.name());
            let kept = ref_ints(man, &topk, 0, RowKind::Tensor, Layout::Logical)
                .map_err(|e| format!("{topk}: {e}"))?;
            routes.push((kept, load(&format!("{}-{vi}", Tap::Weights.name()))?));
        }
        Ok(Chain {
            outs,
            embed: load("norm")?,
            routes,
            n,
        })
    }

    /// The route visit `vi` took at position `pos`.
    fn route(&self, vi: usize, pos: usize) -> Result<([u32; 8], [f32; 8]), String> {
        let (kept, w) = &self.routes[vi];
        let mut out = ([0u32; 8], [0.0f32; 8]);
        for s in 0..8 {
            let at = pos * 8 + s;
            let (Some(&e), Some(&x)) = (kept.get(at), w.get(at)) else {
                return Err(format!(
                    "the set holds no route of visit {vi} at position {pos}"
                ));
            };
            out.0[s] = u32::try_from(e).map_err(|_| format!("visit {vi} keeps expert {e}"))?;
            out.1[s] = x;
        }
        Ok(out)
    }

    /// The stream visit `vi` starts from at position `pos`, `core0` the
    /// first pass's first core visit.
    fn stream(&self, plans: &[VisitPlan], core0: usize, vi: usize, pos: usize) -> Vec<f32> {
        let n = self.n;
        let src = match vi {
            0 => &self.embed,
            _ if plans[vi].pass > 0 && plans[vi].core_index == Some(0) => &self.outs[core0 - 1],
            _ => &self.outs[vi - 1],
        };
        src[pos * n..(pos + 1) * n].to_vec()
    }

    /// The blocks then the stream visit `vi` (the head at the visit count)
    /// starts from, `pb` the prelude's blocks.
    fn rows(&self, plans: &[VisitPlan], pb: usize, vi: usize, pos: usize) -> Vec<f32> {
        let core0 = plans
            .iter()
            .position(|v| v.core_index == Some(0))
            .unwrap_or(0);
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        for u in 0..vi {
            if plans[u].boundary {
                blocks.push(self.stream(plans, core0, u, pos));
            }
            if plans[u].pass_end {
                blocks.truncate(pb);
            }
        }
        let stream = if vi == plans.len() {
            self.outs[vi - 1][pos * self.n..(pos + 1) * self.n].to_vec()
        } else {
            self.stream(plans, core0, vi, pos)
        };
        blocks.push(stream);
        blocks.concat()
    }
}

/// What a replay feeds the walk beside the vendor's routes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Feed {
    /// Every visit and the head start from the vendor's rows: each node's
    /// error is its own visit's.
    Visits,
    /// The first visit starts from the vendor's embedded row, rounded to
    /// 32-value blocks of int8 where `rounded`, and the walk runs on from its
    /// own rows: the `--chain` diagnostic.
    First { rounded: bool },
}

/// A row rounded as the vendor rounds a quantized projection's input: blocks
/// of 32 values, each to int8 steps of its largest magnitude over 127.
fn q8_rounded(row: &[f32]) -> Vec<f32> {
    row.chunks(32)
        .flat_map(|b| {
            let d = b.iter().fold(0.0f32, |m, x| m.max(x.abs())) / 127.0;
            b.iter()
                .map(move |&x| if d > 0.0 { (x / d).round() * d } else { 0.0 })
        })
        .collect()
}

/// One replay of a set's prompt: every position's nodes (where kept), the
/// pick after the last id and the last position's logits.
struct Replayed {
    taps: Vec<Vec<(usize, Tap, Vec<f32>)>>,
    pick: u32,
    logits: Vec<f32>,
}

/// The set's prompt stepped from position 0 over the vendor's routes, fed as
/// `feed` says.
fn replay(
    session: &mut Session,
    man: &RefManifest,
    feed: Feed,
    keep: bool,
) -> Result<Replayed, String> {
    let ids = prompt(man)?;
    session.reset().map_err(|e| e.to_string())?;
    let plans = session.checked().visits.clone();
    let pb = session.checked().prelude_blocks;
    let n = session.checked().hidden;
    let chain = Chain::open(man, plans.len(), n)?;
    let mut taps = Vec::with_capacity(ids.len());
    let mut pick = 0;
    for (pos, &id) in ids.iter().enumerate() {
        let mut got: Vec<(usize, Tap, Vec<f32>)> = Vec::new();
        let mut routes = |vi: usize| chain.route(vi, pos);
        let mut inputs = |vi: usize, banks: usize| -> Result<Option<Vec<f32>>, String> {
            let r = match feed {
                Feed::Visits => chain.rows(&plans, pb, vi, pos),
                Feed::First { .. } if vi > 0 => return Ok(None),
                Feed::First { rounded } => {
                    let row = chain.rows(&plans, pb, 0, pos);
                    if rounded { q8_rounded(&row) } else { row }
                }
            };
            if r.len() != (banks + 1) * n {
                return Err(format!(
                    "the vendor's chain has {} blocks before visit {vi}; the walk {banks}",
                    r.len() / n - 1
                ));
            }
            Ok(Some(r))
        };
        pick = session
            .step_replayed(
                id,
                &mut |vi, tap, vals| {
                    if keep {
                        got.push((vi, tap, vals.to_vec()));
                    }
                },
                &mut routes,
                Some(&mut inputs),
            )
            .map_err(|e| format!("position {pos}: {e}"))?;
        taps.push(got);
    }
    let logits = session.logits_row().map_err(|e| e.to_string())?.to_vec();
    Ok(Replayed { taps, pick, logits })
}

/// The vendor's last-position logits over the cut.
fn vendor_logits(man: &RefManifest, cut: usize) -> Result<Vec<f32>, String> {
    let (_, want) =
        load_ref_logical_in(man, "result_output", 0).map_err(|e| format!("result_output: {e}"))?;
    want.get(..cut)
        .map(<[f32]>::to_vec)
        .ok_or_else(|| format!("result_output holds {} logits, the cut {cut}", want.len()))
}

/// `k_taps`.
fn k_taps(session: &mut Session, detail: usize) -> Clause {
    let man = set(A)?;
    let run = replay(session, &man, Feed::Visits, true)?;
    let plans = session.checked().visits.clone();
    let experts = session.checked().experts;
    let mut rows = Rows {
        man: &man,
        cache: HashMap::new(),
    };
    let chain = Chain::open(&man, plans.len(), session.checked().hidden)?;
    let mut tally = Tally::default();
    let mut parted = 0usize;
    let mut picks = 0usize;
    for (pos, got) in run.taps.iter().enumerate() {
        let mut weights: HashMap<usize, &[f32]> = HashMap::new();
        let mut kept: Vec<(usize, Vec<u32>)> = Vec::new();
        for (vi, tap, vals) in got {
            let v = &plans[*vi];
            match tap {
                Tap::TopK => {
                    kept.push((*vi, vals.iter().map(|&x| x as u32).collect()));
                    continue;
                }
                Tap::Weights => {
                    weights.insert(*vi, vals);
                    continue;
                }
                _ => {}
            }
            let (name, occ) = node_of(*tap, *vi, v.windowed, v.message);
            let want = rows.row(&name, occ, pos, vals.len())?;
            let rel = rel_l2(vals, want);
            tally.take(*tap, &format!("{name}/{occ}"), (pos, *vi), rel, detail);
        }
        for (vi, ours) in &kept {
            picks += 1;
            let (theirs, theirs_w) = chain.route(*vi, pos)?;
            let mut a = ours.clone();
            let mut b = theirs.to_vec();
            a.sort_unstable();
            b.sort_unstable();
            if a == b {
                let ow = weights
                    .get(vi)
                    .ok_or_else(|| format!("visit {vi} handed its ids and not its weights"))?;
                let pair = |ids: &[u32], w: &[f32]| -> Vec<f32> {
                    let mut p: Vec<(u32, f32)> =
                        ids.iter().copied().zip(w.iter().copied()).collect();
                    p.sort_unstable_by_key(|x| x.0);
                    p.into_iter().map(|x| x.1).collect()
                };
                let rel = rel_l2(&pair(ours, ow), &pair(&theirs, &theirs_w));
                let name = format!("{}-{vi}", Tap::Weights.name());
                tally.take(Tap::Weights, &name, (pos, *vi), rel, detail);
                continue;
            }
            // A part: admitted only at a vendor near-tie of the eighth and ninth biased scores.
            let biased = format!("ffn_moe_probs_biased-{vi}");
            let mut sorted: Vec<f32> = rows.row(&biased, 0, pos, experts)?.to_vec();
            sorted.sort_unstable_by(|x, y| y.total_cmp(x));
            let k = ours.len();
            let gap = sorted[k - 1] - sorted[k];
            parted += 1;
            if pos < detail {
                println!(
                    "  pos {pos} visit {vi:2} kept parted: ours {a:?}, the vendor {b:?}, gap {gap:.3e}"
                );
            }
            if gap >= TIE && tally.first_red.is_none() {
                tally.first_red = Some(format!(
                    "ffn_moe_topk-{vi} at position {pos}: kept {a:?}, the vendor {b:?}, its \
                     eighth-ninth gap {gap:.3e} ≥ {TIE:.0e}"
                ));
            }
        }
    }
    // The head at the last position, from the vendor's last blocks and stream.
    let want = vendor_logits(&man, session.vocab_cut())?;
    let rel = rel_l2(&run.logits, &want);
    let last = run.taps.len() - 1;
    tally.take_kind(
        "result_output",
        LOGITS_BAND,
        "result_output",
        (last, plans.len()),
        rel,
        usize::MAX,
    );
    for (kind, band, w) in &tally.worst {
        println!(
            "  worst {kind:>22}: rel {:.3e} at position {} visit {} (band {band:.0e})",
            w.rel, w.pos, w.vi
        );
    }
    match tally.first_red {
        Some(red) => Err(red),
        None => Ok(Verdict::Noted(format!(
            "{} positions × {} visits replayed within their bands, the logits rel {rel:.2e}; \
             {parted} of {picks} kept sets parted at a vendor tie",
            run.taps.len(),
            plans.len()
        ))),
    }
}

/// The sets whose greedy tables `k_greedy` reads: every set but `e`, whose
/// 1,038 ids are thirty times set a's replay.
const GREEDY_SETS: [&str; 4] = [A, B, C, D];

/// `k_greedy`: each set's replayed prompt, its pick against the table's
/// first id. The fixture's walk amplifies the vendor's per-input rounding
/// visit by visit (`--chain`), so a free walk's picks part from the table at
/// its first step whatever the engine; only the replayed prompt keeps the
/// vendor's chain, and the table's later ids follow positions no set dumps.
fn k_greedy(session: &mut Session) -> Clause {
    let mut said = Vec::new();
    for name in GREEDY_SETS {
        let man = set(name)?;
        let table =
            refset::arch::prompt_tables::greedy(&man).map_err(|e| format!("{name}: {e}"))?;
        let (Some(&want), Some(&margin)) = (table.ids.first(), table.margins.first()) else {
            return Err(format!("{name}: an empty greedy table"));
        };
        let run = replay(session, &man, Feed::Visits, false)?;
        if run.pick == want {
            said.push(format!("{name} {want}"));
            continue;
        }
        let part = format!(
            "{name}: {}, the vendor {want} at margin {margin:.3}",
            run.pick
        );
        if margin >= MARGIN_FLOOR {
            return Err(format!("{part} ≥ the floor {MARGIN_FLOOR}"));
        }
        said.push(format!("{part} (a near-tie)"));
    }
    Ok(Verdict::Noted(format!(
        "the replayed prompts' picks: {}",
        said.join("; ")
    )))
}

/// `--chain`: set a's prompt over the vendor's routes from its embedded rows,
/// the walk running on from its own rows, and again from those rows rounded
/// as the vendor rounds a projection's input: each visit's stream at the
/// last position against the vendor's and between the two walks. A
/// diagnostic of the fixture's own sensitivity, no clause.
fn chain_print(session: &mut Session) -> Result<(), String> {
    let man = set(A)?;
    let exact = replay(session, &man, Feed::First { rounded: false }, true)?;
    let rounded = replay(session, &man, Feed::First { rounded: true }, true)?;
    let last = exact.taps.len() - 1;
    let n = session.checked().hidden;
    let chain = Chain::open(&man, session.checked().visits.len(), n)?;
    let outs = |r: &Replayed| -> Vec<Vec<f32>> {
        r.taps[last]
            .iter()
            .filter(|(_, t, _)| *t == Tap::LOut)
            .map(|(_, _, v)| v.clone())
            .collect()
    };
    let (a, b) = (outs(&exact), outs(&rounded));
    for (vi, (x, y)) in a.iter().zip(&b).enumerate() {
        let want = &chain.outs[vi][last * n..(last + 1) * n];
        println!(
            "  visit {vi:2} l_out at position {last}: rel {:.3e} to the vendor's, {:.3e} between \
             the walks",
            rel_l2(x, want),
            rel_l2(x, y)
        );
    }
    let want = vendor_logits(&man, session.vocab_cut())?;
    println!(
        "  logits: rel {:.3e} to the vendor's, {:.3e} between the walks; picks {} and {}, the \
         vendor's greedy table starts {}",
        rel_l2(&exact.logits, &want),
        rel_l2(&exact.logits, &rounded.logits),
        exact.pick,
        rounded.pick,
        refset::arch::prompt_tables::greedy(&man)
            .map_err(|e| e.to_string())?
            .ids
            .first()
            .copied()
            .unwrap_or(u32::MAX)
    );
    Ok(())
}

/// `k_visits`.
fn k_visits(session: &Session) -> Clause {
    let man = set(A)?;
    let has = |name: String| man.tensor(&name, 0).is_ok();
    let slots = (0..).take_while(|s| has(format!("l_out-{s}"))).count();
    let plans = &session.checked().visits;
    if plans.len() != slots {
        return Err(format!(
            "the session walks {} visits; the set has l_out slots 0..{slots}",
            plans.len()
        ));
    }
    for (vi, v) in plans.iter().enumerate() {
        let boundary = !has(format!("attn_out-{vi}"));
        let message = has(format!("loop_q_inp-{vi}"));
        if v.boundary != boundary || v.message != message {
            return Err(format!(
                "visit {vi} (layer {}): the session's boundary {} and message {}, the set's {boundary} \
                 and {message}",
                v.layer, v.boundary, v.message
            ));
        }
    }
    let stored = session.checked().layers.len();
    let boundaries = plans.iter().filter(|v| v.boundary).count();
    let messages = plans.iter().filter(|v| v.message).count();
    Ok(Verdict::Noted(format!(
        "visits {} over {stored} stored layers, {boundaries} boundaries, {messages} message \
         visits, as the set's",
        plans.len()
    )))
}

/// `k_room`: the device bytes the open took, `opened`, against the plan.
fn k_room(session: &Session, opened: u64) -> Clause {
    let held = session.bytes().held();
    let slack = held.buffers * PAGE;
    if opened.abs_diff(held.requested) > slack {
        return Err(format!(
            "the open took {opened} device bytes; the plan asks {} in {} buffers (± {slack})",
            held.requested, held.buffers
        ));
    }
    let free = bloomery_metal::plan::device_room()?.free;
    Ok(Verdict::Noted(format!(
        "the open took {opened} bytes, the plan {} ({} paged) in {} buffers; {}",
        held.requested,
        held.paged,
        held.buffers,
        bloomery_metal::session::room::room_words(&session.bytes(), free, None)
    )))
}

/// The row's run: the open, then the clauses `only` leaves.
pub fn run(
    args: &[String],
    mutant: Option<(&mutants::Mutant, mutants::Program)>,
    only: Option<&str>,
) -> ExitCode {
    if let Some(name) = only
        && !NAMES.contains(&name)
    {
        println!("{GATE}: row inkling2 has no clause named {name}; its clauses are {NAMES:?}");
        return ExitCode::from(64);
    }
    let detail = match flag(args, "--detail").map(str::parse::<usize>) {
        None => 0,
        Some(Ok(n)) => n,
        Some(Err(e)) => {
            println!("{GATE}: --detail takes a count of positions: {e}");
            return ExitCode::from(64);
        }
    };
    let path = match model_path() {
        Ok(p) => p,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let file = match ModelFile::open(&path) {
        Ok(f) => f,
        Err(e) => {
            println!("clause open: FAIL ({}: {e})", path.display());
            return ExitCode::FAILURE;
        }
    };
    let instance = match Session::instance_of(&file, CTX) {
        Ok(i) => i,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let device = match MetalDevice::open(instance) {
        Ok(d) => d,
        Err(e @ DeviceError::Unsupported { .. }) => {
            println!("clause open: REFUSED ({e})");
            return ExitCode::from(78);
        }
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let mutated = match mutant {
        Some((m, mutants::Program::Main)) => {
            match mutants::source_at(instance, m, mutants::Program::Main)
                .and_then(|text| Library::from_source_text(&device, &text).map_err(why))
            {
                Ok(lib) => Some(lib),
                Err(e) => {
                    println!("{GATE}: the mutant does not build: {e}");
                    return ExitCode::from(64);
                }
            }
        }
        Some((m, mutants::Program::Tensor)) => {
            println!(
                "{GATE}: mutant {} edits the tensor program, which no inkling2 launch runs",
                m.name
            );
            return ExitCode::from(64);
        }
        None => None,
    };
    let library = mutated.as_ref().unwrap_or_else(|| device.library());
    if let Some((m, _)) = mutant {
        println!("mutant: {} ({})", m.name, m.why);
    }
    let before = match device.allocated_bytes() {
        Ok(b) => b,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let t = Instant::now();
    let mut session = match Session::open(&device, library, &file, CTX) {
        Ok(s) => s,
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    let opened = match device.allocated_bytes() {
        Ok(after) => after.saturating_sub(before),
        Err(e) => {
            println!("clause open: FAIL ({e})");
            return ExitCode::FAILURE;
        }
    };
    drop(file);
    println!(
        "session: {} ({} stored layers, {} visits, ctx {CTX}) open in {:.1} s; row inkling2, \
         instance {}",
        path.display(),
        session.checked().layers.len(),
        session.checked().visits.len(),
        t.elapsed().as_secs_f64(),
        instance.name
    );
    if args.iter().any(|a| a == "--chain") {
        return match chain_print(&mut session) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                println!("{GATE} --chain: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let mut report = Report::new();
    for name in NAMES {
        if only.is_some_and(|o| o != name) {
            continue;
        }
        bloomery_metal::stream::breadcrumb_clause(name);
        let result = match name {
            "k_taps" => k_taps(&mut session, detail),
            "k_greedy" => k_greedy(&mut session),
            "k_visits" => k_visits(&session),
            "k_room" => k_room(&session, opened),
            other => Err(format!("no clause named {other}")),
        };
        report.clause(name, result);
    }
    let scope = match (mutant, only) {
        (Some((m, _)), _) => format!(" (row inkling2, mutant {})", m.name),
        (None, Some(name)) => format!(" (row inkling2, only {name})"),
        (None, None) => " (row inkling2)".to_string(),
    };
    report.finish(&format!("{GATE}{scope}"))
}
