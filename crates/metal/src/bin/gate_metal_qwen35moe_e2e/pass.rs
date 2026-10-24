//! The batched prompt pass (`Session::pass`) on the d1k set's prompt of about
//! a thousand ids, held to the stepped path it replaces and to ik's
//! reference set.
//!
//! - `q_short` (p′): a prompt of at most a step's window of ids is the step
//!   path: the batch session and the steps session leave the same pick, logits
//!   row bits, delta lanes, conv rings and key and value rows, and the batch
//!   session's rows arena was never opened.
//! - `q_state` (u): the prompt as one pass against the same ids stepped one at
//!   a time: layer 0's recurrent state and conv ring within [`U_L0_REL`] of
//!   the stepped run's; every delta layer's state distance printed.
//! - `q_chunk` (w): the pass size moves no bit: the same ids as passes of 512
//!   (a checkpoint every 512), of 1,000 and 24, of 100 ten times and 24, and
//!   as two calls cut at 700, leave the pick, the logits row and every store
//!   of the one-pass run bit for bit.
//! - `q_ref` (d): the step at the prompt's depth free-running from the pass
//!   run's state and from the stepped run's own: each layer's output distance
//!   from ik's `l_out` within [`FREE_BAND`], the pass run's over the stepped
//!   run's within [`LAYER_RATIO`], the logits' within [`LOGITS_RATIO`], and
//!   the head's pick by its tie rule.
//! - `q_fault`: an id past the vocabulary in a pass raises the token-id site
//!   at layer 0, the position does not move, the next pass and the next step
//!   are refused by poison, and after the reset a pass runs clean.
//! - `q_layers`: an unprofiled pass is one command buffer a layer and one
//!   host wait, and it leaves the profiled pass's pick, logits row and
//!   stores bit for bit, the profiled pass committing a buffer an op.
//! - `q_free`: the same N greedy tokens teacher-forced through the pass run
//!   and the stepped run: every layer's output of every token within
//!   [`FREE_BAND`] of the stepped run's, and the routers' picks differing in
//!   no more than [`FREE_FLIPS_MAX`] sites.
//! - `q_mark`: a pass's copies of its delta stores (`StateOut::after_rows`)
//!   move none of its bits (pick, logits row, every store); the copy after
//!   its last row, taken as a point and cut back to, is the pass's own end
//!   state bit for bit; a copy's ring slots of the positions before the pass
//!   are the ring the pass found; and a copy is the same bits whichever other
//!   copies the pass takes.
//! - `q_mark_cut`: a prompt marked at two positions inside a pass, cut back
//!   to each mark and continued to the end, twice: the same pick, logits row
//!   and stores both times, and the tokens after it within `q_free`'s band of
//!   the stepped run's.

use std::path::Path;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_metal::arch::qwen3moe::count::Layer;
use bloomery_metal::arch::qwen3moe::session::{
    Node, Op, PassOut, Session, SessionError, StateOut, StepMode,
};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::kernels::{CONV_TAPS, HEAD_COLS, HEAD_DIM, KV_HEADS, RING_ROWS, VALUE_HEADS};
use bloomery_metal::library::Library;
use runtime::prompt::PrefillMode;
use runtime::{Target, Want};

use super::chain;
use super::flip;
use super::free::{FREE_BAND, FREE_FLIPS_MAX};
use super::keep;
use super::says;
use super::sets::{self, Set};

/// PIN(2026-10-24): (u)'s band on layer 0's recurrent state and conv ring
/// against the stepped run's (`‖pass − stepped‖ / ‖stepped‖`). Layer 0's
/// input is the embedding rows on both paths, so only its projections differ:
/// the pass runs the f16 operand, which rounds each weight and each activation
/// once into half (a relative error uniform within ±2^-11, rms 2^-11/√3 =
/// 2.8e-4 each, independent), so a projection's element carries
/// Δ = √2 · 2.8e-4 = 4.0e-4 of its own rms over the random signs of 2,048
/// products [derived]. The ring holds the q·k·v projection's rows (Δ). The
/// state is f32 and each token's update is a product of its normed k, its v,
/// β and the decay, each carrying Δ, under a map that does not expand; over
/// 1,024 tokens the sum's relative error is a few Δ (the CUDA gate's own
/// reading of its Δ). The band is 3Δ, 1.2e-3. A wiring fault in layer 0 (β
/// read from α's weights, a token's position off, a ring slot not written)
/// reads an error of order one.
const U_L0_REL: f64 = 1.2e-3;

/// PIN(2026-10-24): (d)'s bound on a layer's store distance from ik's (a
/// recurrent state and the conv rows ik keeps, or the key and value rows, the
/// larger of the two), the pass run's over the stepped run's: the CUDA gate's
/// PROMPT_RATIO, with the derivation of [`LAYER_RATIO`] below.
const PROMPT_RATIO: f64 = 2.5;

/// PIN(2026-10-24): (d)'s bound on a layer's output distance from ik's, the
/// pass run's over the stepped run's. The pass run is another realization of
/// the stepped run's rule (the f16 operand's roundings, the GEMM's sum order),
/// so each distance is one excursion of the state it started from, not an
/// average, and the two runs' own distance up to 2.3 times the stepped run's
/// gives √(1 + 2.3²) = 2.5: the CUDA gate's PROMPT_RATIO and placed step
/// ratio, with the same derivation. A wiring fault adds an error of order one
/// to distances of 1e-2 to 0.3, a ratio above 3 on every layer it reaches.
const LAYER_RATIO: f64 = 2.5;

/// PIN(2026-10-24): (d)'s bound on the logits' distance from ik's, the pass
/// run's over the stepped run's: the one tensor all layers' excursions average
/// into, the CUDA gate's STEP_RATIO.
const LOGITS_RATIO: f64 = 1.5;

/// The conv rows ik's state holds: the taps less the current token's.
const CONV_ROWS: usize = CONV_TAPS - 1;

/// A row of the cache: one cell's key (or value) heads.
const KV_ROW: usize = KV_HEADS * HEAD_DIM;

/// The ids of `q_layers`'s pass: more than a step's window, so the pass runs
/// every rows arm.
const Q_LAYERS_IDS: usize = 100;

/// The greedy tokens `q_free` teacher-forces: with the fixture's four layers,
/// the sixteen router sites [`FREE_FLIPS_MAX`] is derived over.
const FREE_STEPS: usize = 4;

/// The ids `q_mark` passes, and the rows its copies follow: an inner row
/// before the ring's length, one past it, and the last.
const MARK_IDS: usize = 40;
const MARK_ROWS: [u32; 3] = [6, 25, 39];

/// The marks `q_mark_cut` puts inside its prompt's second pass (a point every
/// [`MARK_EVERY`]): one where the ring is all that pass's rows, one whose ring
/// the pass's last rows overwrite.
const MARK_CUTS: [u32; 2] = [700, 1017];
const MARK_EVERY: u32 = 512;

/// The ids of a prompt of `n` ids: the d1k set's first.
pub(super) fn prompt_ids(n: usize) -> Result<Vec<u32>, String> {
    let man = sets::open(Set::D1k)?;
    let (_, _, prompt) = man
        .step()
        .map_err(|e| format!("the d1k set's tokens: {e}"))?;
    prompt
        .get(..n)
        .map(<[u32]>::to_vec)
        .ok_or_else(|| format!("the d1k set holds {} prompt ids, not {n}", prompt.len()))
}

/// ‖a − b‖ / ‖b‖ in f64.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = f64::from(*x) - f64::from(*y);
        num += d * d;
        den += f64::from(*y) * f64::from(*y);
    }
    if num.is_nan() || den == 0.0 {
        return f64::INFINITY;
    }
    (num / den).sqrt()
}

/// One layer's store: a delta layer's state lane and conv ring, or an
/// attention layer's key and value rows.
enum Store {
    Rec { state: Vec<f32>, ring: Vec<f32> },
    Kv { k: Vec<f32>, v: Vec<f32> },
}

/// What a prompt left in the session.
struct Stores {
    pos: u32,
    pick: u32,
    logits: Vec<u32>,
    /// Every layer's store: the state lane the next token reads and the whole
    /// ring, or the key and value rows `0..pos`.
    layers: Vec<Store>,
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The prompt `ids` from zero stores through `Target::prompt`, and what it
/// left.
fn run_prompt(s: &mut Session, ids: &[u32]) -> Result<Stores, String> {
    s.reset().map_err(says)?;
    let (logits, pick) = keep::prompt_bits(s, ids)?;
    read_stores(s, pick, logits)
}

fn read_stores(s: &mut Session, pick: u32, logits: Vec<u32>) -> Result<Stores, String> {
    let pos = s.pos();
    let probe = s.probe().map_err(says)?;
    let mut layers = Vec::new();
    for (l, layer) in s.token_layers().iter().enumerate() {
        layers.push(match layer {
            Layer::Delta { .. } => Store::Rec {
                state: probe.read(Node::DeltaState(l)).map_err(says)?,
                ring: probe.read(Node::DeltaRing(l)).map_err(says)?,
            },
            Layer::Attention { .. } => {
                let (mut k, mut v) = (Vec::new(), Vec::new());
                for row in 0..pos {
                    k.extend(probe.read(Node::KeyRow(l, row)).map_err(says)?);
                    v.extend(probe.read(Node::ValueRow(l, row)).map_err(says)?);
                }
                Store::Kv { k, v }
            }
        });
    }
    Ok(Stores {
        pos,
        pick,
        logits,
        layers,
    })
}

/// The first way two runs' stores differ, by name.
fn differs(what: &str, a: &Stores, b: &Stores) -> Option<String> {
    if a.pos != b.pos {
        return Some(format!("{what}: the positions {} and {}", a.pos, b.pos));
    }
    if a.pick != b.pick {
        return Some(format!("{what}: the picks {} and {}", a.pick, b.pick));
    }
    if a.logits != b.logits {
        return Some(format!("{what}: the logits rows differ"));
    }
    for (l, (x, y)) in a.layers.iter().zip(&b.layers).enumerate() {
        let same = match (x, y) {
            (
                Store::Rec { state: s, ring: r },
                Store::Rec {
                    state: s2,
                    ring: r2,
                },
            ) => {
                if bits(s) != bits(s2) {
                    return Some(format!("{what}: delta layer {l}'s state differs"));
                }
                bits(r) == bits(r2)
            }
            (Store::Kv { k, v }, Store::Kv { k: k2, v: v2 }) => {
                bits(k) == bits(k2) && bits(v) == bits(v2)
            }
            _ => false,
        };
        if !same {
            return Some(format!("{what}: layer {l}'s store differs"));
        }
    }
    None
}

/// The rows of the three positions before `p` in a conv ring, a position's
/// row first: the rows ik's conv state holds.
fn kept_ring(ring: &[f32], p: usize) -> Vec<f32> {
    let ch = ring.len() / RING_ROWS;
    let mut out = Vec::with_capacity(CONV_ROWS * ch);
    for j in 0..CONV_ROWS {
        let slot = (p + j - CONV_ROWS) % RING_ROWS;
        out.extend_from_slice(&ring[slot * ch..(slot + 1) * ch]);
    }
    out
}

/// ik's stores after its prompt batch, from the d1k set's inputs: a delta
/// layer's state (ik's `[k][v]` per head transposed into ours) and the three
/// conv rows it keeps (its `[C][3]` window), an attention layer's key and
/// value rows `0..p` (rows of one cell's two heads, 512 values, the flash
/// layout the fixture's set has; any other shape is refused by name).
fn ik_stores(
    man: &refset::ik::RefManifest,
    kinds: &[Layer],
    p: usize,
) -> Result<Vec<Store>, String> {
    let mut out = Vec::new();
    for (l, kind) in kinds.iter().enumerate() {
        out.push(match kind {
            Layer::Delta { .. } => {
                let s = sets::input(man, &format!("cache_s_l{l}"))?;
                let ssm_len = VALUE_HEADS * HEAD_COLS * HEAD_COLS;
                let conv_len = s.len().checked_sub(ssm_len).unwrap_or(1);
                if conv_len == 0 || conv_len % CONV_ROWS != 0 {
                    return Err(format!("cache_s_l{l} holds {} values", s.len()));
                }
                let channels = conv_len / CONV_ROWS;
                let (conv, ssm) = s.split_at(conv_len);
                let mut state = vec![0.0f32; ssm.len()];
                for h in 0..VALUE_HEADS {
                    for k in 0..HEAD_COLS {
                        for v in 0..HEAD_COLS {
                            state[(h * HEAD_COLS + v) * HEAD_COLS + k] =
                                ssm[(h * HEAD_COLS + k) * HEAD_COLS + v];
                        }
                    }
                }
                let mut ring = vec![0.0f32; conv_len];
                for j in 0..CONV_ROWS {
                    for ch in 0..channels {
                        ring[j * channels + ch] = conv[ch * CONV_ROWS + j];
                    }
                }
                Store::Rec { state, ring }
            }
            Layer::Attention { .. } => {
                let plane = |name: String| -> Result<Vec<f32>, String> {
                    let (vals, ne) = sets::input_f16_rows(man, &name)?;
                    let rows_of_cell_heads =
                        ne[0] as usize == HEAD_DIM && (ne[1] as usize / KV_HEADS) >= p;
                    let flat = ne[1] == 1 && (ne[0] as usize / KV_ROW) >= p;
                    if !(rows_of_cell_heads || flat) || vals.len() < p * KV_ROW {
                        return Err(format!(
                            "{name} is {ne:?}, not rows of {KV_ROW} holding {p} positions"
                        ));
                    }
                    Ok(vals[..p * KV_ROW].to_vec())
                };
                Store::Kv {
                    k: plane(format!("cache_k_l{l}"))?,
                    v: plane(format!("cache_v_l{l}"))?,
                }
            }
        });
    }
    Ok(out)
}

/// A layer's distance from `base` (ik's): the larger of its two stores'
/// `‖ours − base‖ / ‖base‖`, a ring read at the three rows ik keeps.
fn store_dist(ours: &Store, base: &Store, p: usize) -> f64 {
    match (ours, base) {
        (Store::Rec { state, ring }, Store::Rec { state: s, ring: r }) => {
            flip::worse(flip::rel_to(state, s), flip::rel_to(&kept_ring(ring, p), r))
        }
        (Store::Kv { k, v }, Store::Kv { k: kb, v: vb }) => {
            flip::worse(flip::rel_to(k, kb), flip::rel_to(v, vb))
        }
        _ => f64::INFINITY,
    }
}

fn open(
    device: &MetalDevice,
    library: &Library,
    path: &Path,
    ctx: usize,
    every: u32,
    mode: PrefillMode,
) -> Result<Session, String> {
    let mut s = Session::open_with(device, library, path, ctx, every).map_err(says)?;
    s.set_prefill(mode);
    Ok(s)
}

/// (p′): a short prompt is the step path.
pub fn q_short(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let ctx = 64;
    let mut batch = open(device, library, path, ctx, 256, PrefillMode::Batch)?;
    let mut steps = open(device, library, path, ctx, 256, PrefillMode::Steps)?;
    let mut notes = Vec::new();
    for n in [1usize, 5, 8] {
        let ids = prompt_ids(n)?;
        let a = run_prompt(&mut batch, &ids)?;
        let b = run_prompt(&mut steps, &ids)?;
        if let Some(why) = differs(&format!("{n} ids"), &a, &b) {
            return Err(why);
        }
        notes.push(n.to_string());
    }
    if let Some(bytes) = batch.rows_arena_bytes() {
        return Err(format!(
            "a prompt of at most a step's window opened the rows arena ({bytes} bytes)"
        ));
    }
    Ok(Verdict::Noted(format!(
        "prompts of {} ids leave the steps run's pick, logits, states, rings and cache rows bit for bit; no rows arena opened",
        notes.join(", ")
    )))
}

/// (u): layer 0's recurrent state and conv ring, one pass against steps.
pub fn q_state(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let ids = prompt_ids(1024)?;
    let ctx = ids.len() + 1;
    let mut pass = open(device, library, path, ctx, 2048, PrefillMode::Batch)?;
    let a = run_prompt(&mut pass, &ids)?;
    if pass.rows_arena_bytes().is_none() {
        return Err("a prompt of 1,024 ids did not take the pass".to_string());
    }
    drop(pass);
    let mut steps = open(device, library, path, ctx, 2048, PrefillMode::Steps)?;
    let b = run_prompt(&mut steps, &ids)?;
    let (
        Some(Store::Rec {
            state: s0,
            ring: r0,
        }),
        Some(Store::Rec {
            state: t0,
            ring: u0,
        }),
    ) = (a.layers.first(), b.layers.first())
    else {
        return Err("layer 0 is not a delta layer".to_string());
    };
    let state0 = rel(s0, t0);
    let ring0 = rel(r0, u0);
    let others: Vec<String> = a
        .layers
        .iter()
        .zip(&b.layers)
        .enumerate()
        .skip(1)
        .map(|(l, pair)| match pair {
            (Store::Rec { state: x, .. }, Store::Rec { state: y, .. }) => {
                format!("delta layer {l} state {:.3e}", rel(x, y))
            }
            (Store::Kv { k: x, v: xv }, Store::Kv { k: y, v: yv }) => {
                format!(
                    "attention layer {l} k {:.3e} v {:.3e}",
                    rel(x, y),
                    rel(xv, yv)
                )
            }
            _ => format!("layer {l} of another kind"),
        })
        .collect();
    println!(
        "q_state: delta layer 0 state {state0:.3e} ring {ring0:.3e} (band {U_L0_REL:.1e}); {}; picks {} and {}",
        others.join(", "),
        a.pick,
        b.pick
    );
    if state0 <= U_L0_REL && ring0 <= U_L0_REL {
        Ok(Verdict::Noted(format!(
            "layer 0 after 1,024 ids: state {state0:.3e}, ring {ring0:.3e} of the stepped run's, within {U_L0_REL:.1e}"
        )))
    } else {
        Err(format!(
            "layer 0 after 1,024 ids: state {state0:.3e}, ring {ring0:.3e}, past {U_L0_REL:.1e}"
        ))
    }
}

/// (w): the pass size moves no bit.
pub fn q_chunk(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let ids = prompt_ids(1024)?;
    let ctx = ids.len() + 1;
    let whole = {
        let mut s = open(device, library, path, ctx, 2048, PrefillMode::Batch)?;
        run_prompt(&mut s, &ids)?
    };
    let mut notes = Vec::new();
    for (name, every, cut) in [
        ("512 x 2", 512u32, None),
        ("1,000 and 24", 1000, None),
        ("100 x 10 and 24", 100, None),
        ("two calls cut at 700", 2048, Some(700usize)),
    ] {
        let mut s = open(device, library, path, ctx, every, PrefillMode::Batch)?;
        let got = match cut {
            None => run_prompt(&mut s, &ids)?,
            Some(at) => {
                s.reset().map_err(says)?;
                keep::prompt_bits(&mut s, &ids[..at])?;
                let (logits, pick) = keep::prompt_bits(&mut s, &ids[at..])?;
                read_stores(&mut s, pick, logits)?
            }
        };
        if let Some(why) = differs(name, &got, &whole) {
            return Err(why);
        }
        notes.push(name);
    }
    Ok(Verdict::Noted(format!(
        "the ids as passes of {} leave the one pass's pick, logits row, states, rings and cache rows bit for bit",
        notes.join("; ")
    )))
}

/// One layer's output and the logits of the step token through the traced
/// step.
struct Traced {
    outs: Vec<Vec<f32>>,
    ids: Vec<Vec<u32>>,
    logits: Vec<f32>,
    top: u32,
}

fn traced_step(s: &mut Session, tok: u32) -> Result<Traced, String> {
    let n = s.layer_count();
    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); n];
    let mut ids: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut logits = Vec::new();
    let stepped = s
        .step_traced(tok, |op, probe| {
            match op {
                Op::Experts(l) => {
                    outs[l] = probe.read(Node::Residual)?;
                    ids[l] = probe.router_ids(l)?;
                }
                Op::Head => logits = probe.read(Node::Logits)?,
                _ => {}
            }
            Ok(())
        })
        .map_err(says)?;
    Ok(Traced {
        outs,
        ids,
        logits,
        top: stepped.id,
    })
}

/// (d): the step at the prompt's depth against ik, from the pass run's state
/// and from the stepped run's.
pub fn q_ref(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let man = sets::open(Set::D1k)?;
    let (pos, step, prompt) = man
        .step()
        .map_err(|e| format!("the d1k set's tokens: {e}"))?;
    let [tok] = step else {
        return Err(format!("the d1k set holds {} step tokens", step.len()));
    };
    let ctx = pos as usize + 1;
    let ids = prompt.to_vec();
    let mut pass = open(device, library, path, ctx, 2048, PrefillMode::Batch)?;
    let mut steps = open(device, library, path, ctx, 2048, PrefillMode::Steps)?;
    let n = pass.layer_count();
    let ik_out: Vec<Vec<f32>> = (0..n)
        .map(|l| sets::tap(&man, &format!("l_out-{l}")))
        .collect::<Result<_, _>>()?;
    let ik_logits = sets::tap(&man, "result_output")?;
    let vocab = pass.vocab();
    let ik_last = ik_logits
        .len()
        .checked_sub(vocab)
        .map(|at| &ik_logits[at..])
        .ok_or("result_output holds no row")?;
    let stores_pass = run_prompt(&mut pass, &ids)?;
    let stores_steps = run_prompt(&mut steps, &ids)?;
    let p = ids.len();
    let ik_layers = ik_stores(&man, pass.token_layers(), p)?;
    let mut worst_store = (0f64, 0usize);
    for (l, ik) in ik_layers.iter().enumerate() {
        let (dp, ds) = (
            store_dist(&stores_pass.layers[l], ik, p),
            store_dist(&stores_steps.layers[l], ik, p),
        );
        let ratio = flip::worse(0.0, dp / ds.max(f64::MIN_POSITIVE));
        println!(
            "q_ref store layer={l} pass_from_ik={dp:.3e} steps_from_ik={ds:.3e} ratio={ratio:.2}"
        );
        if ratio > worst_store.0 {
            worst_store = (ratio, l);
        }
    }
    if worst_store.0 > PROMPT_RATIO {
        return Err(format!(
            "layer {}'s store is {:.2} times as far from ik's after the pass as after the steps, past {PROMPT_RATIO}",
            worst_store.1, worst_store.0
        ));
    }
    let a = traced_step(&mut pass, *tok)?;
    let b = traced_step(&mut steps, *tok)?;
    let mut worst_ratio = 0f64;
    let (mut worst_a, mut worst_b) = (0f64, 0f64);
    for l in 0..n {
        let (ea, eb) = (
            flip::rel_to(&a.outs[l], &ik_out[l]),
            flip::rel_to(&b.outs[l], &ik_out[l]),
        );
        worst_a = flip::worse(worst_a, ea);
        worst_b = flip::worse(worst_b, eb);
        let ratio = ea / eb.max(f64::MIN_POSITIVE);
        println!(
            "q_ref layer={l} pass_l_out_rel={ea:.3e} steps_l_out_rel={eb:.3e} ratio={ratio:.2}"
        );
        if ea > FREE_BAND || !ea.is_finite() {
            return Err(format!(
                "layer {l}: the pass run's output is {ea:.3e} from ik's, past {FREE_BAND}"
            ));
        }
        if ratio > LAYER_RATIO {
            return Err(format!(
                "layer {l}: the pass run is {ea:.3e} from ik's and the stepped run {eb:.3e}, a ratio {ratio:.2} past {LAYER_RATIO}"
            ));
        }
        worst_ratio = worst_ratio.max(ratio);
    }
    let (la, lb) = (
        flip::rel_to(&a.logits, ik_last),
        flip::rel_to(&b.logits, ik_last),
    );
    let logits_ratio = la / lb.max(f64::MIN_POSITIVE);
    let head_in = flip::rel_to(&a.outs[n - 1], &ik_out[n - 1]);
    let tie = head_in.is_finite() && flip::head_tie(ik_last, a.top, head_in);
    let line = format!(
        "the d1k step after a pass prompt: worst l_out_rel {worst_a:.3e} (stepped {worst_b:.3e}, worst layer ratio {worst_ratio:.2}, band {LAYER_RATIO}); logits_rel {la:.3e} (stepped {lb:.3e}, ratio {logits_ratio:.2}, band {LOGITS_RATIO}); pick ours={} stepped={} ik={}",
        a.top,
        b.top,
        flip::argmax(ik_last)
    );
    if logits_ratio <= LOGITS_RATIO && tie {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}

/// `q_free`: the greedy tokens of the stepped run teacher-forced through the
/// pass run.
pub fn q_free(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let ids = prompt_ids(1024)?;
    let ctx = ids.len() + FREE_STEPS + 1;
    let mut pass = open(device, library, path, ctx, 2048, PrefillMode::Batch)?;
    let mut steps = open(device, library, path, ctx, 2048, PrefillMode::Steps)?;
    let (_, pick) = keep::prompt_bits(&mut steps, &ids)?;
    let (_, pass_pick) = keep::prompt_bits(&mut pass, &ids)?;
    let mut tok = pick;
    let (mut worst, mut flips, mut agree) = (0f64, 0usize, usize::from(pick == pass_pick));
    let layers = pass.layer_count();
    for t in 0..FREE_STEPS {
        let b = traced_step(&mut steps, tok)?;
        let a = traced_step(&mut pass, tok)?;
        for l in 0..layers {
            worst = flip::worse(worst, flip::rel_to(&a.outs[l], &b.outs[l]));
            flips += usize::from(a.ids[l][..a.ids[l].len() - 1] != b.ids[l][..b.ids[l].len() - 1]);
        }
        if a.top == b.top {
            agree += 1;
        }
        println!("q_free token {t}: picks pass={} steps={}", a.top, b.top);
        tok = b.top;
    }
    let line = format!(
        "{FREE_STEPS} teacher-forced tokens after 1,024 ids: worst layer output {worst:.3e} of the stepped run's (band {FREE_BAND}), {flips} router sites with other picks (cap {FREE_FLIPS_MAX}), {agree} of {} picks equal",
        FREE_STEPS + 1
    );
    if worst <= FREE_BAND && flips <= FREE_FLIPS_MAX {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}

/// `q_fault`: the pass reads its fault word once, at its end, and poisons.
pub fn q_fault(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let mut s = open(device, library, path, 64, 2048, PrefillMode::Batch)?;
    let mut ids = prompt_ids(12)?;
    let bad = u32::try_from(s.vocab()).map_err(|_| "the vocabulary does not fit u32")?;
    ids[7] = bad;
    let fault = match s.pass(&ids, PassOut::Last, StateOut::default()) {
        Err(SessionError::Fault(f)) => f,
        Err(e) => {
            return Err(format!(
                "a pass with id {bad}: another error than a fault: {e}"
            ));
        }
        Ok(p) => return Err(format!("a pass with id {bad} ran clean, picking {}", p.id)),
    };
    if fault.layer != 0 || fault.sites & (1u32 << (FaultSite::TokenId as u32)) == 0 {
        return Err(format!(
            "a pass with id {bad} raised {fault}; the token-id site at layer 0 is not among them"
        ));
    }
    if s.pos() != 0 {
        return Err(format!(
            "the faulted pass moved the position to {}",
            s.pos()
        ));
    }
    for (what, refused) in [
        (
            "pass",
            s.pass(&ids[..12], PassOut::Last, StateOut::default())
                .map(|_| ()),
        ),
        ("step", s.step(ids[0], StepMode::HostFed).map(|_| ())),
    ] {
        match refused {
            Err(SessionError::Poisoned(f)) if f == fault => {}
            Err(e) => {
                return Err(format!(
                    "the {what} after the fault: {e}; poison is expected"
                ));
            }
            Ok(()) => {
                return Err(format!(
                    "the {what} after the fault ran on a poisoned session"
                ));
            }
        }
    }
    s.reset().map_err(says)?;
    ids[7] = ids[6];
    let clean = s
        .pass(&ids, PassOut::Last, StateOut::default())
        .map_err(|e| format!("the pass after the reset: {}", says(e)))?;
    if clean.id as usize >= s.vocab() || s.pos() != 12 {
        return Err(format!(
            "the pass after the reset picked {} and stands at {}",
            clean.id,
            s.pos()
        ));
    }
    Ok(Verdict::Noted(format!(
        "a pass with id {bad} raised {fault} and left the position at 0; the next pass and step \
         were refused by poison; after the reset a pass of 12 ids ran clean"
    )))
}

/// A pass of [`Q_LAYERS_IDS`] ids as the prompt runs it and as the profile runs
/// it: what each left, and the buffers and waits each moved.
pub fn q_layers(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let both = prompt_ids(2 * Q_LAYERS_IDS)?;
    let (ids, other) = both.split_at(Q_LAYERS_IDS);
    let mut s = open(
        device,
        library,
        path,
        Q_LAYERS_IDS + 1,
        2048,
        PrefillMode::Batch,
    )?;
    // The rows arena opens at a first pass of other ids, outside the counted
    // one: a row an op reads before the op ahead of it wrote holds those.
    s.pass(other, PassOut::None, StateOut::default())
        .map_err(says)?;
    s.reset().map_err(says)?;
    let before = s.stats();
    let one = s
        .pass(ids, PassOut::Last, StateOut::default())
        .map_err(says)?;
    let moved = chain::moved(&s, before);
    let layers = s.layer_count() as u64;
    if moved.submits != layers || moved.waits != 1 {
        return Err(format!(
            "a pass of {Q_LAYERS_IDS} ids committed {} command buffers and waited {} times; a \
             buffer a layer ({layers}) and one wait are expected",
            moved.submits, moved.waits
        ));
    }
    let logits = bits(s.logits_row().map_err(says)?);
    let one_stores = read_stores(&mut s, one.id, logits)?;
    s.reset().map_err(says)?;
    let mut samples = Vec::new();
    let before = s.stats();
    let per_op = s
        .pass_profile(ids, PassOut::Last, &mut samples)
        .map_err(says)?;
    let moved = chain::moved(&s, before);
    if moved.submits != samples.len() as u64 {
        return Err(format!(
            "the profiled pass committed {} command buffers for {} ops; one an op is expected",
            moved.submits,
            samples.len()
        ));
    }
    let logits = bits(s.logits_row().map_err(says)?);
    let per_op_stores = read_stores(&mut s, per_op.id, logits)?;
    if let Some(why) = differs(
        "a buffer a layer against a buffer an op",
        &one_stores,
        &per_op_stores,
    ) {
        return Err(why);
    }
    if one.count != per_op.count {
        return Err(format!(
            "the layers' buffers encoded {:?}; the ops' buffers {:?}",
            one.count, per_op.count
        ));
    }
    Ok(Verdict::Noted(format!(
        "a pass of {Q_LAYERS_IDS} ids is {layers} command buffers (one a layer) and one wait, and \
         leaves the pick, logits row, states, rings and cache rows of the profiled pass's {} \
         buffers bit for bit",
        samples.len()
    )))
}

/// The pass of `ids` from zero stores taking copies after the rows `after`,
/// and what it left.
fn copied_pass(s: &mut Session, ids: &[u32], after: &[u32]) -> Result<Stores, String> {
    s.reset().map_err(says)?;
    let passed = s
        .pass(ids, PassOut::Last, StateOut { after_rows: after })
        .map_err(says)?;
    let logits = bits(s.logits_row().map_err(says)?);
    read_stores(s, passed.id, logits)
}

/// The delta layers' stores at `at`, a point the session holds: one step on
/// from the position, then the cut back to it. The attention rows read are
/// those before `at`, which neither the step nor the cut moves.
fn stores_at_point(s: &mut Session, at: u32, end: &Stores) -> Result<Stores, String> {
    Target::step(s, end.pick, Want::Argmax).map_err(says)?;
    Target::cut(s, at).map_err(says)?;
    read_stores(s, end.pick, end.logits.clone())
}

/// The first delta layer whose ring in `at`, a copy after row `r` of a pass
/// from 0, differs from `found` (the ring that pass found) in a slot of a
/// position before the pass: slots `r + 1..`, which the copy takes from the
/// ring as it was.
fn found_slots(at: &Stores, found: &Stores, r: usize) -> Option<usize> {
    at.layers
        .iter()
        .zip(&found.layers)
        .position(|(a, f)| match (a, f) {
            (Store::Rec { ring: a, .. }, Store::Rec { ring: f, .. }) => {
                let from = (r + 1) * (f.len() / RING_ROWS);
                bits(&a[from..]) != bits(&f[from..])
            }
            _ => false,
        })
}

/// `q_mark`: a pass's copies are its own state at their rows and move none of
/// its bits.
pub fn q_mark(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let ids = prompt_ids(MARK_IDS)?;
    let mut s = open(
        device,
        library,
        path,
        MARK_IDS + 2,
        2048,
        PrefillMode::Batch,
    )?;
    s.reset().map_err(says)?;
    let found = read_stores(&mut s, 0, Vec::new())?;
    let plain = copied_pass(&mut s, &ids, &[])?;
    let [inner, outer, last] = MARK_ROWS;
    let mut seen: Vec<(u32, Stores)> = Vec::new();
    for after in [[inner, last], [outer, last], [inner, outer]] {
        let what = format!("the pass copying after rows {after:?}");
        let got = copied_pass(&mut s, &ids, &after)?;
        if let Some(why) = differs(&what, &got, &plain) {
            return Err(format!("{why}, against the pass with no copies"));
        }
        for &r in &after {
            s.checkpoint_copy(r + 1).map_err(says)?;
        }
        // Descending, so each cut finds the point it names.
        for &r in after.iter().rev() {
            let at = stores_at_point(&mut s, r + 1, &got)?;
            let held = format!("{what}: the point at {} from its copy", r + 1);
            if r == last
                && let Some(why) = differs(&held, &at, &plain)
            {
                return Err(format!("{why}, against the pass's own end"));
            }
            if (r as usize) + 1 < RING_ROWS
                && let Some(l) = found_slots(&at, &found, r as usize)
            {
                return Err(format!(
                    "{held}: delta layer {l}'s ring slots before the pass are not the ring the \
                     pass found"
                ));
            }
            match seen.iter().find(|(row, _)| *row == r) {
                Some((_, first)) => {
                    if let Some(why) = differs(&held, &at, first) {
                        return Err(format!("{why}, against the same row's copy beside another"));
                    }
                }
                None => seen.push((r, at)),
            }
        }
    }
    Ok(Verdict::Noted(format!(
        "{MARK_IDS} ids as one pass copying after rows {MARK_ROWS:?} two at a time: the pick, logits \
         row and every store of the pass with none; the copy after the last row is the pass's end, \
         the copy after row {inner} keeps the found ring before the pass, and each copy the same \
         beside either other"
    )))
}

/// `q_mark_cut`: cut back to a mark inside a pass and continued, twice, and
/// the tokens after it against the stepped run's.
pub fn q_mark_cut(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let ids = prompt_ids(1024)?;
    let ctx = ids.len() + FREE_STEPS + 1;
    let mut steps = open(device, library, path, ctx, 2048, PrefillMode::Steps)?;
    let (_, pick) = keep::prompt_bits(&mut steps, &ids)?;
    let mut want = Vec::with_capacity(FREE_STEPS);
    let mut tok = pick;
    for _ in 0..FREE_STEPS {
        let t = traced_step(&mut steps, tok)?;
        tok = t.top;
        want.push(t);
    }
    let mut s = open(device, library, path, ctx, MARK_EVERY, PrefillMode::Batch)?;
    s.reset().map_err(says)?;
    Target::prompt_marked(&mut s, &ids, &MARK_CUTS, Want::Argmax).map_err(says)?;
    if s.rows_arena_bytes().is_none() {
        return Err("a prompt of 1,024 ids did not take the pass".to_string());
    }
    let layers = s.layer_count();
    let mut notes = Vec::new();
    // Descending: a cut drops the points past it.
    for &u in MARK_CUTS.iter().rev() {
        let rest = &ids[u as usize..];
        let mut first: Option<Stores> = None;
        for _ in 0..2 {
            Target::cut(&mut s, u).map_err(says)?;
            let (logits, pick) = keep::prompt_bits(&mut s, rest)?;
            let got = read_stores(&mut s, pick, logits)?;
            match &first {
                Some(a) => {
                    if let Some(why) = differs(&format!("continued from {u} again"), &got, a) {
                        return Err(why);
                    }
                }
                None => first = Some(got),
            }
        }
        let start = first.as_ref().map_or(pick, |a| a.pick);
        let (mut worst, mut flips) = (0f64, 0usize);
        let mut tok = pick;
        for b in &want {
            let a = traced_step(&mut s, tok)?;
            for l in 0..layers {
                worst = flip::worse(worst, flip::rel_to(&a.outs[l], &b.outs[l]));
                flips +=
                    usize::from(a.ids[l][..a.ids[l].len() - 1] != b.ids[l][..b.ids[l].len() - 1]);
            }
            tok = b.top;
        }
        let line = format!(
            "from {u}: {FREE_STEPS} teacher-forced tokens worst layer output {worst:.3e} of the \
             stepped run's (band {FREE_BAND}), {flips} router sites with other picks (cap \
             {FREE_FLIPS_MAX}), pick {start} against the stepped run's {pick}"
        );
        if worst > FREE_BAND || flips > FREE_FLIPS_MAX {
            return Err(line);
        }
        notes.push(line);
    }
    Ok(Verdict::Noted(format!(
        "1,024 ids in passes of {MARK_EVERY} marked at {MARK_CUTS:?}, cut back to each and \
         continued twice: the same pick, logits row and stores; {}",
        notes.join("; ")
    )))
}
