//! The batched prompt pass (`Session::pass`) on the row's prompt of about a
//! thousand ids (the d1k set's, or generated ids), held to the stepped path it
//! replaces.
//!
//! - `q_short` (p′): a prompt of at most a step's window of ids is the step
//!   path: the batch session and the steps session leave the same pick, logits
//!   row bits, delta lanes, conv rings and key and value rows, and the batch
//!   session ran no pass.
//! - `q_state` (u): the prompt as one pass against the same ids stepped one at
//!   a time: layer 0's recurrent state and conv ring within [`U_L0_REL`] of
//!   the stepped run's (a short conv layer 0: its ring); every recurrent
//!   layer's distance printed.
//! - `q_chunk` (w): the pass size moves no bit: the same ids as passes of 512
//!   (a checkpoint every 512), of 1,000 and 24, of 100 ten times and 24, and
//!   as two calls cut at 700, leave the pick, the logits row and every store
//!   of the one-pass run bit for bit.
//! - `q_fault`: an id past the vocabulary in a pass raises the token-id site
//!   at layer 0, the position does not move, the next pass and the next step
//!   are refused by poison, and after the reset a pass runs clean.
//! - `q_layers`: an unprofiled pass is one command buffer a layer and one
//!   host wait, and it leaves the profiled pass's pick, logits row and
//!   stores bit for bit, the profiled pass committing a buffer an op.
//! - `q_free`: the same N greedy tokens teacher-forced through the pass run
//!   and the stepped run: every layer's output of every token within
//!   [`FREE_BAND`] of the stepped run's ([`FREE_BAND_DENSE_LAYER`] over the
//!   layers where no layer routes), and the routers' picks differing in no more than
//!   [`FREE_FLIPS_MAX`] sites.
//! - `q_mark`: a pass's copies of its delta stores (`StateOut::after_rows`)
//!   move none of its bits (pick, logits row, every store); the copy after
//!   its last row, taken as a point and cut back to, is the pass's own end
//!   state bit for bit; a copy's ring slots of the positions before the pass
//!   are the ring the pass found; and a copy is the same bits whichever other
//!   copies the pass takes. A file whose recurrent store is a conv ring has no
//!   such copy: the pass refuses the request by name and the clause holds the
//!   refusal.
//! - `q_mark_cut`: a prompt stepped with a mark, cut back to the mark and
//!   continued, twice: the stores after each cut a fresh run's at the mark and
//!   the continued ids the fresh run's bits, on every file with a recurrent
//!   store; and, where a pass copies its stores, a prompt marked at two
//!   positions inside a pass, cut back to each mark and continued to the end,
//!   twice: the same pick, logits row and stores both times, and the tokens
//!   after it within `q_free`'s band of the stepped run's.
//! - `q_marked`: a prompt call with a mark inside it, longer than a step's
//!   window, runs as a pass and leaves the unmarked call's bits; the stepped
//!   run's pick and the teacher-forced tokens after it hold to `q_free`'s
//!   band. A file with no recurrent store drops the mark (it has nothing to
//!   copy).
//! - `q_tensor`: `q_free` with the pass's GEMMs on the tensor-op kernels
//!   (`Library::with_tensor_gemms`), forced on any device that builds them,
//!   held to the same band and cap: the tensor path's dot band is `q_free`'s
//!   f16 operand roundings plus an f32 sum of at most K − 1 additions a term
//!   (the kernels gate's `gemm_band`), under 2^-11 of a dot's magnitude at
//!   K = 2,048, so the pass is another realization of the same rule.

use std::path::Path;

use bloomery_device::conformance::{Clause, Verdict};
use bloomery_device::fault::FaultSite;
use bloomery_metal::arch::qwen3moe::count::{FfnKind, Layer};
use bloomery_metal::arch::qwen3moe::session::{
    Node, Op, PassOut, Session, SessionError, StateOut, StepMode,
};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::{GemmPath, Library};
use runtime::prompt::PrefillMode;
use runtime::{Target, Want};

use super::chain;
use super::flip;
use super::keep;
use super::says;
use super::sets;

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

/// PIN(2026-09-27): the band on a layer output's relative distance from the
/// stepped run's. The derivation is the CUDA gate's free-running bound:
/// per-layer errors of about 4e-2 of a layer's update, added in quadrature
/// over 40 layers, √40 · 4e-2 ≈ 0.25.
const FREE_BAND: f64 = 0.26;

/// PIN(2026-10-28): the band on a layer output's relative distance from the
/// stepped run's for a session none of whose layers route, which
/// [`FREE_BAND`]'s router-flip term does not reach. A dense FFN has no router,
/// so the two runs take the same slot and differ in operands alone: the
/// stepped mat-vec reads f32 activations and weights, and the pass's GEMMs run
/// the f16 operand (`PASS_OPERAND`), which rounds each weight and each
/// activation once into half. `U_L0_REL` reads that rounding as Δ = 4.0e-4 of
/// a projection element's own rms. A layer's update carries at most 3Δ: the
/// mixer's, by `U_L0_REL`'s count; the FFN's gate and up GEMMs at Δ each
/// through silu(g)·u, whose log-slope is at most 1.28 where g > 0 and the
/// sum's weight sits, 1.6Δ, then the down GEMM at Δ, 1.9Δ. Taking the update
/// as the layer's whole output (the residual only dilutes it), 3Δ = 1.2e-3 a
/// layer [derived]; [`free_band`] adds the session's layers in quadrature
/// (the fixture's four: 3Δ · √4 = 2.4e-3).
const FREE_BAND_DENSE_LAYER: f64 = 1.2e-3;

/// PIN(2026-10-13): the cap on router sites whose picks differ from the
/// stepped run's. The derivation is the CUDA gate's: a broken router flips
/// about every one of the sixteen sites, a correct one about eight; the cap
/// is three quarters.
const FREE_FLIPS_MAX: usize = 12;

/// The band a session's layer outputs are held to, with the words that name it
/// in a verdict line: [`FREE_BAND_DENSE_LAYER`] over the session's layers in
/// quadrature where no layer's FFN is routed (the session's own layer kinds),
/// [`FREE_BAND`] otherwise.
fn free_band(s: &Session) -> (f64, &'static str) {
    if s.token_layers().iter().any(|l| l.ffn() == FfnKind::Routed) {
        (FREE_BAND, "band")
    } else {
        let layers = s.layer_count() as f64;
        (FREE_BAND_DENSE_LAYER * layers.sqrt(), "dense band")
    }
}

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

/// The ids `q_marked` prompts, more than a step's window so the call is a
/// pass, and the mark inside it.
const MARKED_IDS: usize = 40;
const MARKED_AT: u32 = 20;

/// The marks `q_mark_cut` puts inside its prompt's second pass (a point every
/// [`MARK_EVERY`]): one where the ring is all that pass's rows, one whose ring
/// the pass's last rows overwrite.
const MARK_CUTS: [u32; 2] = [700, 1017];
const MARK_EVERY: u32 = 512;

/// The ids of a prompt of `n` ids: the row's own (`sets::prompt_ids`).
pub(super) fn prompt_ids(n: usize) -> Result<Vec<u32>, String> {
    sets::prompt_ids(n)
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

/// One layer's store: a delta layer's state lane and conv ring, a short conv
/// layer's ring, or an attention layer's key and value rows.
enum Store {
    Rec { state: Vec<f32>, ring: Vec<f32> },
    Conv { ring: Vec<f32> },
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
            Layer::ShortConv { .. } => Store::Conv {
                ring: probe.read(Node::ConvRing(l)).map_err(says)?,
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
            (Store::Conv { ring: r }, Store::Conv { ring: r2 }) => bits(r) == bits(r2),
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
    let passes = batch.stats().passes;
    if passes != 0 {
        return Err(format!(
            "prompts of at most a step's window ran {passes} passes; they are steps"
        ));
    }
    Ok(Verdict::Noted(format!(
        "prompts of {} ids leave the steps run's pick, logits, states, rings and cache rows bit for bit; no pass ran",
        notes.join(", ")
    )))
}

/// (u): layer 0's recurrent store, one pass against steps: a delta layer's
/// state and conv ring, or a short conv layer's ring. A short conv's ring holds
/// u = B ⊙ x, the product of two channel groups of the in_proj output, each
/// carrying the pass's operand rounding Δ (see [`U_L0_REL`]): at most 2Δ under
/// the band's 3Δ. Whether the conv ring is also bit equal is printed, not
/// required: the pass's GEMM runs the f16 operand.
///
/// Mutant: in `shortconv.metal`, the step's ring write
/// `slot_ring[sc_slot(p, args.ring_rows) * args.hidden + c]` with `p + 1u`
/// for `p`: the stepped ring's slots sit one position late against the pass's,
/// an error of order one.
pub fn q_state(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let ids = prompt_ids(1024)?;
    let ctx = ids.len() + 1;
    let mut pass = open(device, library, path, ctx, 2048, PrefillMode::Batch)?;
    let a = run_prompt(&mut pass, &ids)?;
    if pass.stats().passes == 0 {
        return Err("a prompt of 1,024 ids did not take the pass".to_string());
    }
    drop(pass);
    let mut steps = open(device, library, path, ctx, 2048, PrefillMode::Steps)?;
    let b = run_prompt(&mut steps, &ids)?;
    let (kind, state0, ring0, ring_bits) = match (a.layers.first(), b.layers.first()) {
        (
            Some(Store::Rec {
                state: s0,
                ring: r0,
            }),
            Some(Store::Rec {
                state: t0,
                ring: u0,
            }),
        ) => (
            "delta",
            Some(rel(s0, t0)),
            rel(r0, u0),
            bits(r0) == bits(u0),
        ),
        (Some(Store::Conv { ring: r0 }), Some(Store::Conv { ring: u0 })) => {
            ("conv", None, rel(r0, u0), bits(r0) == bits(u0))
        }
        _ => return Err("layer 0 keeps no recurrent store".to_string()),
    };
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
            (Store::Conv { ring: x }, Store::Conv { ring: y }) => {
                format!("conv layer {l} ring {:.3e}", rel(x, y))
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
    let shown = state0.map_or_else(
        || format!("{kind} layer 0 ring {ring0:.3e} (bit equal: {ring_bits})"),
        |s| format!("{kind} layer 0 state {s:.3e} ring {ring0:.3e}"),
    );
    println!(
        "q_state: {shown} (band {U_L0_REL:.1e}); {}; picks {} and {}",
        others.join(", "),
        a.pick,
        b.pick
    );
    if state0.is_none_or(|s| s <= U_L0_REL) && ring0 <= U_L0_REL {
        Ok(Verdict::Noted(format!(
            "{kind} layer 0 after 1,024 ids: {}ring {ring0:.3e} of the stepped run's, within \
             {U_L0_REL:.1e}",
            state0.map_or_else(String::new, |s| format!("state {s:.3e}, "))
        )))
    } else {
        Err(format!(
            "{kind} layer 0 after 1,024 ids: {}ring {ring0:.3e}, past {U_L0_REL:.1e}",
            state0.map_or_else(String::new, |s| format!("state {s:.3e}, "))
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

/// One layer's residual and the routers' picks of the step token through the
/// traced step.
struct Traced {
    outs: Vec<Vec<f32>>,
    ids: Vec<Vec<u32>>,
    top: u32,
}

fn traced_step(s: &mut Session, tok: u32) -> Result<Traced, String> {
    let n = s.layer_count();
    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); n];
    let mut ids: Vec<Vec<u32>> = vec![Vec::new(); n];
    let stepped = s
        .step_traced(tok, |op, probe| {
            if let Op::Experts(l) = op {
                outs[l] = probe.read(Node::Residual)?;
                ids[l] = probe.router_ids(l)?;
            }
            Ok(())
        })
        .map_err(says)?;
    Ok(Traced {
        outs,
        ids,
        top: stepped.id,
    })
}

// `q_ref` compared the step at the prompt's depth to ik's d1k set.
// PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified

/// `q_free`: the greedy tokens of the stepped run teacher-forced through the
/// pass run.
pub fn q_free(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    free_run(device, library, path, "q_free")
}

/// `q_tensor`: `q_free` with the pass's GEMMs from the tensor-op program
/// `tensor_text` (the shipped one, or a mutant's).
pub fn q_tensor(device: &MetalDevice, library: &Library, tensor_text: &str, path: &Path) -> Clause {
    let tensor = Library::tensor_from_source_text(device, tensor_text)
        .map_err(|e| format!("the tensor-op GEMMs do not build here: {e}"))?;
    let forced = library.with_tensor_gemms(tensor);
    if forced.gemm_path() != GemmPath::TensorOps {
        return Err(format!(
            "the forced library's GEMMs are {}, not the tensor-op ones",
            forced.gemm_path().name()
        ));
    }
    let line = free_run(device, &forced, path, "q_tensor")?;
    match line {
        Verdict::Noted(line) => Ok(Verdict::Noted(format!("tensor-op GEMMs: {line}"))),
        other => Ok(other),
    }
}

/// The greedy tokens of the stepped run teacher-forced through the pass run,
/// both sessions on `library`'s kernels.
fn free_run(device: &MetalDevice, library: &Library, path: &Path, clause: &str) -> Clause {
    let ids = prompt_ids(1024)?;
    let ctx = ids.len() + FREE_STEPS + 1;
    let mut pass = open(device, library, path, ctx, 2048, PrefillMode::Batch)?;
    let mut steps = open(device, library, path, ctx, 2048, PrefillMode::Steps)?;
    let (_, pick) = keep::prompt_bits(&mut steps, &ids)?;
    let (_, pass_pick) = keep::prompt_bits(&mut pass, &ids)?;
    teacher_forced(
        clause,
        &mut pass,
        &mut steps,
        (pick, pass_pick),
        "1,024 ids",
    )
}

/// The greedy tokens of `steps` teacher-forced through both sessions, which
/// hold the same prompt (`after` names it), `picks` the two runs' prompt
/// picks: every layer's output of every token within [`free_band`] of the
/// stepped run's and the routers' picks differing in no more than
/// [`FREE_FLIPS_MAX`] sites.
fn teacher_forced(
    clause: &str,
    pass: &mut Session,
    steps: &mut Session,
    picks: (u32, u32),
    after: &str,
) -> Clause {
    let (pick, pass_pick) = picks;
    let mut tok = pick;
    let (mut worst, mut flips, mut agree) = (0f64, 0usize, usize::from(pick == pass_pick));
    let layers = pass.layer_count();
    let routed = pass.geom().experts.routed_ids;
    let (band, band_name) = free_band(pass);
    for t in 0..FREE_STEPS {
        let b = traced_step(steps, tok)?;
        let a = traced_step(pass, tok)?;
        for l in 0..layers {
            let rel = flip::rel_to(&a.outs[l], &b.outs[l]);
            let flipped = a.ids[l][..routed] != b.ids[l][..routed];
            println!(
                "{clause} token {t} layer {l}: output {rel:.3e} of the stepped run's, router picks {}",
                if flipped { "differ" } else { "equal" }
            );
            worst = flip::worse(worst, rel);
            flips += usize::from(flipped);
        }
        if a.top == b.top {
            agree += 1;
        }
        println!("{clause} token {t}: picks pass={} steps={}", a.top, b.top);
        tok = b.top;
    }
    let line = format!(
        "{FREE_STEPS} teacher-forced tokens after {after}: worst layer output {worst:.3e} of the stepped run's ({band_name} {band}), {flips} router sites with other picks (cap {FREE_FLIPS_MAX}), {agree} of {} picks equal",
        FREE_STEPS + 1
    );
    if worst <= band && flips <= FREE_FLIPS_MAX {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}

/// `q_marked`: a prompt call with a mark inside it, longer than a step's
/// window, is a pass that moves none of the unmarked call's bits, and the
/// stepped run's pick and teacher-forced tokens hold to `q_free`'s band after
/// it. The mark is a checkpoint where the file has a recurrent store to copy
/// and nothing where it has none.
pub fn q_marked(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let ids = prompt_ids(MARKED_IDS)?;
    let ctx = ids.len() + FREE_STEPS + 1;
    let mut pass = open(device, library, path, ctx, 2048, PrefillMode::Batch)?;
    let plain = run_prompt(&mut pass, &ids)?;
    if pass.stats().passes == 0 {
        return Err(format!(
            "a prompt of {MARKED_IDS} ids did not take the pass"
        ));
    }
    pass.reset().map_err(says)?;
    let before = pass.stats();
    let (logits, pick) = keep::prompt_bits_marked(&mut pass, &ids, &[MARKED_AT])?;
    let moved = chain::moved(&pass, before);
    if moved.passes == 0 {
        return Err(format!(
            "the prompt of {MARKED_IDS} ids marked at {MARKED_AT} ran no pass"
        ));
    }
    let recurrent = super::row::current().recurrent;
    if pass.checkpoint_positions().contains(&MARKED_AT) != recurrent {
        return Err(format!(
            "the mark at {MARKED_AT} left the points {:?}; the file {} a recurrent store to copy",
            pass.checkpoint_positions(),
            if recurrent { "has" } else { "has no" }
        ));
    }
    let marked = read_stores(&mut pass, pick, logits)?;
    if let Some(why) = differs(&format!("a prompt marked at {MARKED_AT}"), &marked, &plain) {
        return Err(format!("{why}, against the unmarked call"));
    }
    let mut steps = open(device, library, path, ctx, 2048, PrefillMode::Steps)?;
    let (_, steps_pick) = keep::prompt_bits(&mut steps, &ids)?;
    let line = teacher_forced(
        "q_marked",
        &mut pass,
        &mut steps,
        (steps_pick, pick),
        &format!("{MARKED_IDS} ids marked at {MARKED_AT}"),
    )?;
    Ok(match line {
        Verdict::Noted(line) => Verdict::Noted(format!(
            "the prompt of {MARKED_IDS} ids marked at {MARKED_AT} ran a pass and left the unmarked \
             call's pick, logits row, states, rings and cache rows bit for bit, {} point; {line}",
            if recurrent { "the mark a" } else { "no" }
        )),
        other => other,
    })
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
    // A first pass of other ids, outside the counted one, leaves its rows in
    // the arena: a row an op reads before the op ahead of it wrote holds those.
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
fn found_slots(at: &Stores, found: &Stores, r: usize, ring_rows: usize) -> Option<usize> {
    at.layers
        .iter()
        .zip(&found.layers)
        .position(|(a, f)| match (a, f) {
            (Store::Rec { ring: a, .. }, Store::Rec { ring: f, .. }) => {
                let from = (r + 1) * (f.len() / ring_rows);
                bits(&a[from..]) != bits(&f[from..])
            }
            _ => false,
        })
}

/// `q_mark` on a file whose recurrent store is a short conv ring: a pass asked
/// for copies after rows is refused by name, never run without them, and after
/// a reset the session leaves the same bits as one never asked.
///
/// Mutant: the refusal deleted, so a conv layer's pass takes the request and
/// copies nothing of its ring: the pass runs and the clause names its pick.
fn copies_refused(s: &mut Session, ids: &[u32]) -> Clause {
    let plain = copied_pass(s, ids, &[])?;
    let [inner, _, last] = MARK_ROWS;
    s.reset().map_err(says)?;
    let after = [inner, last];
    let refused = match s.pass(ids, PassOut::Last, StateOut { after_rows: &after }) {
        Err(e @ SessionError::Shape { .. }) => e,
        Err(e) => {
            return Err(format!(
                "a pass with copies after rows {after:?}: another error than a refusal by name: {e}"
            ));
        }
        Ok(p) => {
            return Err(format!(
                "a pass with copies after rows {after:?} ran and picked {}: the copy of a conv \
                 ring is not run, and the pass must refuse the request by name",
                p.id
            ));
        }
    };
    let again = copied_pass(s, ids, &[])?;
    if let Some(why) = differs(
        "the pass after a refused request for copies",
        &again,
        &plain,
    ) {
        return Err(why);
    }
    Ok(Verdict::Noted(format!(
        "a pass of {MARK_IDS} ids copying after rows {after:?} is refused by name ({refused}); \
         after the reset the pass with no copies leaves the bits it left before the request"
    )))
}

/// `q_mark`: a pass's copies are its own state at their rows and move none of
/// its bits; on a file whose recurrent store is a conv ring the copies are
/// refused by name ([`copies_refused`]).
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
    if !keep::pass_copies(&s) {
        return copies_refused(&mut s, &ids);
    }
    let ring_rows = s.geom().delta.ring_rows;
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
            if (r as usize) + 1 < ring_rows
                && let Some(l) = found_slots(&at, &found, r as usize, ring_rows)
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

/// The ids of the cut-and-re-step case: more than a ring's rows, so a ring slot
/// has wrapped before the mark, and the mark inside them.
const CUT_IDS: usize = 40;
const CUT_AT: u32 = 25;

/// A prompt stepped with a mark, cut back to the mark and continued, twice: the
/// checkpoint holds every recurrent store. After the cut the stores are those
/// of a fresh run of the ids before the mark, and the ids continued from it
/// leave the bits of a fresh run of all of them. Whether a pass can copy its
/// stores is returned with the line.
///
/// Mutant: in `Session::cut`, the `None if n < self.pos => Ok(())` arm taken
/// for a file whose recurrent store is a conv ring (the checkpoints list of
/// `Session::open` left empty for it): the mark leaves no checkpoint, and a cut
/// returns `Ok` with the ring as the end of the run left it.
fn cut_restep(
    device: &MetalDevice,
    library: &Library,
    path: &Path,
) -> Result<(String, bool), String> {
    let ids = prompt_ids(CUT_IDS)?;
    let at = CUT_AT as usize;
    let mut s = open(
        device,
        library,
        path,
        ids.len() + 1,
        2048,
        PrefillMode::Steps,
    )?;
    let copies = keep::pass_copies(&s);
    let at_mark = run_prompt(&mut s, &ids[..at])?;
    let end = run_prompt(&mut s, &ids)?;
    s.reset().map_err(says)?;
    let (logits, pick) = keep::prompt_bits_marked(&mut s, &ids, &[CUT_AT])?;
    if !s.checkpoint_positions().contains(&CUT_AT) {
        return Err(format!(
            "the prompt marked at {CUT_AT} left the checkpoints {:?}: none at the mark",
            s.checkpoint_positions()
        ));
    }
    let marked = read_stores(&mut s, pick, logits)?;
    if let Some(why) = differs(&format!("a prompt marked at {CUT_AT}"), &marked, &end) {
        return Err(format!("{why}, against the unmarked prompt"));
    }
    for round in 0..2 {
        Target::cut(&mut s, CUT_AT).map_err(says)?;
        if s.pos() != CUT_AT {
            return Err(format!(
                "a cut to {CUT_AT} stands at {} (round {round})",
                s.pos()
            ));
        }
        let cut = read_stores(&mut s, at_mark.pick, at_mark.logits.clone())?;
        if let Some(why) = differs(&format!("the cut back to {CUT_AT}"), &cut, &at_mark) {
            return Err(format!("{why}, against a fresh run of the ids before it"));
        }
        let (logits, pick) = keep::prompt_bits(&mut s, &ids[at..])?;
        let again = read_stores(&mut s, pick, logits)?;
        if let Some(why) = differs(&format!("the ids continued from {CUT_AT}"), &again, &end) {
            return Err(format!("{why}, against a fresh run of all of them"));
        }
    }
    Ok((
        format!(
            "{CUT_IDS} ids stepped marked at {CUT_AT} and cut back to the mark twice: the stores \
             after each cut are a fresh run's at the mark, and the ids continued from it leave \
             the fresh run's pick, logits row and stores"
        ),
        copies,
    ))
}

/// `q_mark_cut`: cut back to a mark and continued ([`cut_restep`]), and on a
/// file whose passes copy their stores, a mark inside a pass ([`cut_marked_pass`]).
pub fn q_mark_cut(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
    let (line, copies) = cut_restep(device, library, path)?;
    if !copies {
        return Ok(Verdict::Noted(format!(
            "{line}; no mark inside a pass: a pass's copy of a conv ring is refused by name \
             (`q_mark`)"
        )));
    }
    match cut_marked_pass(device, library, path)? {
        Verdict::Noted(rest) => Ok(Verdict::Noted(format!("{line}; {rest}"))),
        other => Ok(other),
    }
}

/// A prompt marked at two positions inside a pass, cut back to each mark and
/// continued, twice, and the tokens after it against the stepped run's.
fn cut_marked_pass(device: &MetalDevice, library: &Library, path: &Path) -> Clause {
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
    if s.stats().passes == 0 {
        return Err("a prompt of 1,024 ids did not take the pass".to_string());
    }
    let layers = s.layer_count();
    let routed = s.geom().experts.routed_ids;
    let (band, band_name) = free_band(&s);
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
                flips += usize::from(a.ids[l][..routed] != b.ids[l][..routed]);
            }
            tok = b.top;
        }
        let line = format!(
            "from {u}: {FREE_STEPS} teacher-forced tokens worst layer output {worst:.3e} of the \
             stepped run's ({band_name} {band}), {flips} router sites with other picks (cap \
             {FREE_FLIPS_MAX}), pick {start} against the stepped run's {pick}"
        );
        if worst > band || flips > FREE_FLIPS_MAX {
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
