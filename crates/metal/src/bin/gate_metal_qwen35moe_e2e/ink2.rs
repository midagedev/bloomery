//! The `inkling2` row: the family's session (`bloomery_metal::arch::inkling2`)
//! on the checkpoint folder `inkling2-mlx` under the fixture root, a cut of
//! the Mac users' own `mlx-community/Inkling-2-Mini-4bit`
//! (`tools/mlx-fixture-cuts/inkling2-mlx.json`: 13 stored layers, every expert
//! of each, the core's six run three times, 25 visits), against mlx-lm's own
//! `inkling2` modules in f32 over the exact dequant of the same folder
//! (`pins/inkling2.txt` and `.f32`, `tools/mlx-logits.py dump`). The folder is
//! the MLX form a user opens: the affine 4-bit dense projections, embedding
//! and head, the NVFP4 experts with their global scale and the MXFP8 coda.
//!
//! The walk every clause reads: the pin's ids fed one step each, the cache
//! kept, the way the dump runs them, each visit's experts run at mlx-lm's kept
//! ids of that call (`pins/inkling2.routes`). The router keeps 8 of 128 by a
//! biased score whose 8th and 9th often lie within a rounding of each other,
//! and with the route scale of 8 one expert swapped moves the stream by a
//! whole expert's output, so a free walk that parted at such a tie would carry
//! the part through the cache. Where the session's own kept
//! set is mlx-lm's the walk runs the session's own ids and weights; where it
//! parts, mlx-lm's ids with the weights the session's own scores give them.
//!
//! The clauses:
//!
//! - `k_routes`: at every router call of the walk, the session's own kept ids
//!   (its top 8 of its own scores) against mlx-lm's; a part is admitted only
//!   where mlx-lm's 8th-9th biased gap is under [`TIE`].
//! - `k_logits`: at each step the pin holds a row for, the session's logits
//!   (after the epilogue, over the vocabulary cut) held to the row's band
//!   (`Row::logit_abs`).
//! - `k_greedy`: at each held step the head's own pick against the pin's
//!   argmax; a part where the pin's top-two margin is [`MARGIN_FLOOR`] or more
//!   is red.
//! - `k_visits`: the session's walk is mlx-lm's over the folder's config
//!   ([`VISIT_LAYERS`]).
//! - `k_room`: the open's device bytes are the plan's, to within a page a
//!   buffer, and the room line the plan prints.
//! - `k_pass`: the pin's ids as prompt passes (`Session::pass_routed`), each
//!   ending at a held step and run on the walk's routing rule, against the
//!   step walk's logits at the held steps, within twice the row's band (each
//!   walk is within one band of the pin); the passes' own kept sets part from
//!   mlx-lm's only under `k_routes`' floor.
//! - `k_pass_pin`: the same passes' logits against the pin, by `k_logits`'s
//!   measure and band.
//! - `k_carry`: a prompt of the pin's ids repeated to 1,600 positions as five
//!   passes ([`CARRY_ENDS`]: a pass boundary three rows past a full pass,
//!   rows a window and more past the first key, a windowed visit's ring
//!   wrapped), each visit's route recorded, against the steps of the same ids
//!   at the recorded routes, at each pass's last row: the conv rings, the K
//!   and V rows and the ring's window carried across the passes are the
//!   steps'.
//! - `k_serve`: the server's engine over the session (`serve.rs`), last, since
//!   the engine takes the session: a prompt of the pin's ids repeated to 600
//!   (two passes), a request extending it by the pin's first 20 ids, and one
//!   sharing only the prompt.

use std::cell::RefCell;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use bloomery_device::DeviceError;
use bloomery_device::conformance::{Clause, Report, Verdict, why};
use bloomery_metal::arch::inkling2::{Session, Tap};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::library::Library;
use bloomery_metal::session::room::PAGE;
use model_hf::ModelFile;

use super::mlx::{Pin, distance, parse_pin, pin_of};
use super::{GATE, flag, mlx_split, mutants, row, serve};

/// The row's clauses, in the order they run.
pub const NAMES: [&str; 9] = [
    "k_routes",
    "k_logits",
    "k_greedy",
    "k_visits",
    "k_room",
    "k_pass",
    "k_pass_pin",
    "k_carry",
    "k_serve",
];

/// The experts a router call keeps.
const TOP_K: usize = 8;

/// PIN(2026-11-23): the 8th-9th biased-score gap of mlx-lm's under which the
/// session's own kept set may part from it. A part needs the two scores'
/// errors to close the gap, and each moves by at most the rounding rule's
/// own: `tools/mlx-logits.py simulate --round all` over the pin's walk at the
/// pin's routes moves a router score by 5.53e-4 at most (step 32), so twice
/// that, 1.11e-3, rounded up. The pin's walk holds 107 of its 875 calls under
/// it.
const TIE: f32 = 1.2e-3;

/// The cache rows of the session.
const CTX: usize = 4096;

/// The classification floor of a pick (`gpu-gates` `prompts::MARGIN_FLOOR`,
/// PIN(2026-09-21)): a part where the pin's top-two margin is under it is a
/// near-tie re-lottery of two engines' roundings, not a fault.
const MARGIN_FLOOR: f32 = 0.5;

/// The stored layer of each visit mlx-lm walks over the folder's config, derived on paper:
/// `TextModel.layer_visits` is `layers[:core_start] + layers[core_start:core_end] * num_passes +
/// layers[core_end:]`, over 13 layers, the core 6..12 and three passes.
const VISIT_LAYERS: [usize; 25] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12,
];

/// The layers outside `local_layer_ids` (the folder's 0-4, 6-10 and 12): a windowless visit
/// reads the whole prefix.
const GLOBAL_LAYERS: [usize; 2] = [5, 11];

/// `n_layers_per_attn_res_block`: a stored layer whose id it divides closes a block onto the
/// bank (`DecoderLayer.is_block_boundary`), so the visits 0, 6, 12, 18 and 24 do.
const BLOCK: usize = 6;

/// The ends of `k_carry`'s passes: a whole prompt batch, three rows across its boundary, two
/// passes of a whole batch whose rows read keys a window back (the second's past the ring's first
/// wrap, at 1,536 rows at [`CTX`]) and a tail.
const CARRY_ENDS: [usize; 5] = [512, 515, 1027, 1539, 1600];

/// The first visit of the second pass: the core's visits from it on merge the messages of the
/// pass before (`_run_loop` hands a bank to every pass past the first), the visits 12 to 23.
const FIRST_MESSAGE: usize = 12;

/// The folder's path.
fn model_path() -> Result<PathBuf, String> {
    mlx_split::folder(row::current().arch)
}

/// The row's pin, refused by name when its rows are not the session's vocabulary cut.
fn pin(session: &Session) -> Result<Pin, String> {
    let family = row::current().arch;
    let (text, raw) = pin_of(None, family)?;
    let pin = parse_pin(family, &text, &raw)?;
    if pin.n != session.vocab_cut() {
        return Err(format!(
            "the pin's rows hold {} logits; the session's vocabulary cut is {}",
            pin.n,
            session.vocab_cut()
        ));
    }
    Ok(pin)
}

/// One router call of mlx-lm's walk: its kept ids ascending and its 8th-9th biased gap.
struct Route {
    ids: [u32; TOP_K],
    gap: f32,
}

/// The pin's routes (`pins/inkling2.routes`, `tools/mlx-logits.py dump`'s): one a router call,
/// in run order, refused by name unless it is the pin's steps of `visits` calls each.
fn routes(pin: &Pin, visits: usize) -> Result<Vec<Route>, String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/bin/gate_metal_qwen35moe_e2e/pins")
        .join(format!("{}.routes", row::current().arch));
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("the pin's routes {}: {e}", path.display()))?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.by_ref().take(4).collect();
    let want = [
        "version 1".to_string(),
        format!("family {}", row::current().arch),
        format!("steps {}", pin.ids.len()),
        format!("visits {visits}"),
    ];
    if head != want {
        return Err(format!("the pin's routes open {head:?}, not {want:?}"));
    }
    let mut out = Vec::with_capacity(pin.ids.len() * visits);
    for (k, line) in lines.enumerate() {
        let bad = || format!("the pin's routes line {}: {line:?}", k + 5);
        let f: Vec<&str> = line.split(' ').collect();
        let [step, visit, ids, gap] = f[..] else {
            return Err(bad());
        };
        if (step.parse(), visit.parse()) != (Ok(k / visits), Ok(k % visits)) {
            return Err(bad());
        }
        let ids: Vec<u32> = ids
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()
            .map_err(|_| bad())?;
        out.push(Route {
            ids: ids.try_into().map_err(|_| bad())?,
            gap: gap.parse().map_err(|_| bad())?,
        });
    }
    if out.len() != pin.ids.len() * visits {
        return Err(format!(
            "the pin's routes hold {} calls, not {} steps of {visits}",
            out.len(),
            pin.ids.len()
        ));
    }
    Ok(out)
}

/// One held step of the walk: the step, the session's logits and its pick.
struct Held {
    step: usize,
    logits: Vec<f32>,
    pick: u32,
}

/// A router call where the session's own kept set is not mlx-lm's.
struct Part {
    step: usize,
    visit: usize,
    ours: Vec<u32>,
    theirs: [u32; TOP_K],
    gap: f32,
}

/// The walk: each held step, every part, and the router calls compared.
struct Walked {
    held: Vec<Held>,
    parts: Vec<Part>,
    calls: usize,
}

/// What a visit's router handed on before its route is asked for.
#[derive(Default)]
struct Seen {
    probs: Vec<f32>,
    kept: Vec<u32>,
    weights: Vec<f32>,
}

/// Every node of the walk as it runs: the step, the visit, the kind and its values.
type StepTaps<'a> = &'a mut dyn FnMut(usize, usize, Tap, &[f32]);

/// One route of the walk: each kept id and its weight.
type Route8 = ([u32; TOP_K], [f32; TOP_K]);

/// One router call's route by the walk's rule: the session's own ids and weights where its kept
/// set is mlx-lm's `r`, else mlx-lm's ids with the weights the session's own scores `probs` give
/// them (route scale `scale`), and then the session's kept set, ascending, which parted.
fn route_of(
    r: &Route,
    probs: &[f32],
    kept: &[u32],
    weights: &[f32],
    scale: f32,
) -> Result<(Route8, Option<Vec<u32>>), String> {
    let mut ours = kept.to_vec();
    ours.sort_unstable();
    if ours == r.ids {
        let ids = kept.try_into().map_err(|_| "a route of another width")?;
        let w = weights.try_into().map_err(|_| "weights of another width")?;
        return Ok(((ids, w), None));
    }
    let at = |e: u32| {
        probs
            .get(e as usize)
            .copied()
            .ok_or("an expert past the scores")
    };
    let mut sum = 0.0f32;
    for &e in &r.ids {
        sum += at(e)?;
    }
    let mut w = [0.0f32; TOP_K];
    for (wi, &e) in w.iter_mut().zip(&r.ids) {
        *wi = at(e)? / (sum + 1e-20) * scale;
    }
    Ok(((r.ids, w), Some(ours)))
}

/// The pin's ids through the session from position 0, one step an id, each visit's experts at
/// mlx-lm's kept ids (the session's own route where its kept set is mlx-lm's): each held step's
/// logits and pick, and every call where the session's own kept set parts from mlx-lm's; every
/// node goes to `sink` as it runs.
fn walk(session: &mut Session, pin: &Pin, sink: StepTaps<'_>) -> Result<Walked, String> {
    let visits = session.checked().visits.len();
    let scale = session.checked().route_scale;
    if session.checked().top_k != TOP_K {
        return Err(format!(
            "the file keeps {} experts a call; the pin's routes keep {TOP_K}",
            session.checked().top_k
        ));
    }
    let routes = routes(pin, visits)?;
    session.reset().map_err(|e| e.to_string())?;
    let seen = RefCell::new(Seen::default());
    let parts = RefCell::new(Vec::new());
    let mut held = Vec::with_capacity(pin.positions.len());
    for (k, &id) in pin.ids.iter().enumerate() {
        let mut taps = |vi: usize, tap: Tap, vals: &[f32]| {
            sink(k, vi, tap, vals);
            let mut s = seen.borrow_mut();
            match tap {
                Tap::Probs => s.probs = vals.to_vec(),
                Tap::TopK => s.kept = vals.iter().map(|&x| x as u32).collect(),
                Tap::Weights => s.weights = vals.to_vec(),
                _ => {}
            }
        };
        let mut route = |vi: usize| -> Result<([u32; TOP_K], [f32; TOP_K]), String> {
            let r = &routes[k * visits + vi];
            let s = seen.borrow();
            let ((ids, w), parted) = route_of(r, &s.probs, &s.kept, &s.weights, scale)?;
            if let Some(ours) = parted {
                parts.borrow_mut().push(Part {
                    step: k,
                    visit: vi,
                    ours,
                    theirs: r.ids,
                    gap: r.gap,
                });
            }
            Ok((ids, w))
        };
        let pick = session
            .step_routed(id, &mut taps, &mut route)
            .map_err(|e| format!("step {k} (id {id}): {e}"))?;
        if pin.positions.get(held.len()) != Some(&k) {
            continue;
        }
        let logits = session.logits_row().map_err(|e| e.to_string())?.to_vec();
        held.push(Held {
            step: k,
            logits,
            pick,
        });
    }
    Ok(Walked {
        held,
        parts: parts.into_inner(),
        calls: routes.len(),
    })
}

/// `k_routes`.
fn k_routes(session: &mut Session) -> Clause {
    let pin = pin(session)?;
    let w = walk(session, &pin, &mut |_, _, _, _| {})?;
    for p in &w.parts {
        println!(
            "k_routes step {} visit {}: the session keeps {:?}, mlx-lm {:?} at gap {:.3e}",
            p.step, p.visit, p.ours, p.theirs, p.gap
        );
    }
    if let Some(p) = w.parts.iter().find(|p| p.gap >= TIE) {
        return Err(format!(
            "step {} visit {}: the session keeps {:?}, mlx-lm {:?}, whose 8th-9th gap {:.3e} ≥ \
             the floor {TIE:.1e}",
            p.step, p.visit, p.ours, p.theirs, p.gap
        ));
    }
    let largest = match w.parts.iter().map(|p| p.gap).reduce(f32::max) {
        Some(g) => format!("the largest gap among them {g:.3e}"),
        None => "none to measure".to_string(),
    };
    Ok(Verdict::Noted(format!(
        "{} of {} router calls kept mlx-lm's set; {} parts admitted under the near-tie floor \
         {TIE:.1e} ({largest})",
        w.calls - w.parts.len(),
        w.calls,
        w.parts.len()
    )))
}

/// `--nodes OUT`: the walk the clauses read, with each visit's attention output before its gate,
/// after its post-conv, the experts' sum and the FFN's output after its post-conv written in run
/// order as `OUT.f32`, one line a node in `OUT.tsv` (step, visit, kind, offset and length in
/// values). A diagnostic, no clause: `tools/mlx-logits.py nodes` writes mlx-lm's of the same walk.
fn nodes_print(session: &mut Session, out: &str) -> Result<(), String> {
    use std::fmt::Write as _;
    let pin = pin(session)?;
    let mut f32s: Vec<u8> = Vec::new();
    let mut tsv = String::new();
    let mut at = 0usize;
    walk(session, &pin, &mut |k, vi, tap, vals| {
        let kind = match tap {
            Tap::Kqv => "kqv",
            Tap::AttnOut => "attn_out",
            Tap::MoeOut => "moe_out",
            Tap::FfnOut => "ffn_out",
            _ => return,
        };
        let _ = writeln!(tsv, "{k}\t{vi}\t{kind}\t{at}\t{}", vals.len());
        at += vals.len();
        f32s.extend(vals.iter().flat_map(|x| x.to_le_bytes()));
    })?;
    std::fs::write(format!("{out}.f32"), f32s).map_err(|e| format!("{out}.f32: {e}"))?;
    std::fs::write(format!("{out}.tsv"), tsv).map_err(|e| format!("{out}.tsv: {e}"))?;
    println!("{GATE} --nodes: wrote {out}.f32 and {out}.tsv");
    Ok(())
}

/// `k_logits`.
fn k_logits(session: &mut Session) -> Clause {
    let pin = pin(session)?;
    let band = row::current().logit_abs;
    let mut worst: Option<(usize, f64)> = None;
    let mut per_step = Vec::new();
    let mut over = 0;
    for (h, got) in walk(session, &pin, &mut |_, _, _, _| {})?
        .held
        .iter()
        .enumerate()
    {
        let want = &pin.rows[h * pin.n..(h + 1) * pin.n];
        let d = distance(&got.logits, want, band)
            .map_err(|e| format!("step {} (id {}): {e}", got.step, pin.ids[got.step]))?;
        println!(
            "k_logits step {} (id {}): max |logit| difference {:.3e} at {} (session {:.6}, pin \
             {:.6}), {} values over the band, rms {:.3e}",
            got.step, pin.ids[got.step], d.max, d.at, d.session, d.stored, d.over, d.rms
        );
        per_step.push(format!("{}: {:.2e}", got.step, d.max));
        over += d.over;
        if worst.is_none_or(|(_, w)| d.max > w) {
            worst = Some((got.step, d.max));
        }
    }
    let (at, max) = worst.ok_or("the pin holds no row")?;
    let line = format!(
        "pin inkling2: max |logit| difference {max:.3e} at step {at}; per step [{}]; band \
         {band}, {over} values over it; {} rows of {} held of {} fed",
        per_step.join(", "),
        pin.positions.len(),
        pin.n,
        pin.ids.len()
    );
    if max <= band {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}

/// The pin row's top-two margin.
fn margin(row: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in row {
        if x > a {
            b = a;
            a = x;
        } else if x > b {
            b = x;
        }
    }
    a - b
}

/// `k_greedy`.
fn k_greedy(session: &mut Session) -> Clause {
    let pin = pin(session)?;
    let argmax = pin
        .argmax
        .clone()
        .ok_or("the pin names no argmax line, the picks this clause holds")?;
    let mut said = Vec::new();
    for (h, got) in walk(session, &pin, &mut |_, _, _, _| {})?
        .held
        .iter()
        .enumerate()
    {
        let want = argmax[h];
        if got.pick == want {
            said.push(format!("{} {want}", got.step));
            continue;
        }
        let m = margin(&pin.rows[h * pin.n..(h + 1) * pin.n]);
        let part = format!(
            "step {}: {}, the pin {want} at margin {m:.3}",
            got.step, got.pick
        );
        if m >= MARGIN_FLOOR {
            return Err(format!("{part} ≥ the floor {MARGIN_FLOOR}"));
        }
        said.push(format!("{part} (a near-tie)"));
    }
    Ok(Verdict::Noted(format!(
        "the walk's picks at the pinned steps: {}",
        said.join("; ")
    )))
}

/// The pin's ids through the session as prompt passes, each ending at a held step, every visit's
/// experts routed by [`route_of`] row by row: each held step's logits and the head's pick, and
/// every router call where the pass's own kept set parts from mlx-lm's.
fn pass_walk(session: &mut Session, pin: &Pin) -> Result<(Vec<Held>, Vec<Part>), String> {
    let visits = session.checked().visits.len();
    let scale = session.checked().route_scale;
    let experts = session.checked().experts;
    if session.checked().top_k != TOP_K {
        return Err(format!(
            "the file keeps {} experts a call; the pin's routes keep {TOP_K}",
            session.checked().top_k
        ));
    }
    let routes = routes(pin, visits)?;
    session.reset().map_err(|e| e.to_string())?;
    let mut held = Vec::with_capacity(pin.positions.len());
    let mut parts = Vec::new();
    let mut from = 0usize;
    for &end in &pin.positions {
        let rows = end + 1 - from;
        let mut hook = |vi: usize,
                        probs: &[f32],
                        kept: &[u32],
                        weights: &[f32]|
         -> Result<(Vec<u32>, Vec<f32>), String> {
            let mut ids = Vec::with_capacity(rows * TOP_K);
            let mut w = Vec::with_capacity(rows * TOP_K);
            for t in 0..rows {
                let r = &routes[(from + t) * visits + vi];
                let ((i, wt), parted) = route_of(
                    r,
                    &probs[t * experts..(t + 1) * experts],
                    &kept[t * TOP_K..(t + 1) * TOP_K],
                    &weights[t * TOP_K..(t + 1) * TOP_K],
                    scale,
                )?;
                if let Some(ours) = parted {
                    parts.push(Part {
                        step: from + t,
                        visit: vi,
                        ours,
                        theirs: r.ids,
                        gap: r.gap,
                    });
                }
                ids.extend(i);
                w.extend(wt);
            }
            Ok((ids, w))
        };
        let passed = session
            .pass_routed(&pin.ids[from..=end], &mut hook)
            .map_err(|e| format!("the pass of steps {from}..={end}: {e}"))?;
        let logits = session.logits_row().map_err(|e| e.to_string())?.to_vec();
        held.push(Held {
            step: end,
            logits,
            pick: passed.id,
        });
        from = end + 1;
    }
    Ok((held, parts))
}

/// `k_pass`.
fn k_pass(session: &mut Session) -> Clause {
    let pin = pin(session)?;
    let band = 2.0 * row::current().logit_abs;
    let steps = walk(session, &pin, &mut |_, _, _, _| {})?.held;
    let (passes, parts) = pass_walk(session, &pin)?;
    if let Some(p) = parts.iter().find(|p| p.gap >= TIE) {
        return Err(format!(
            "step {} visit {}: the pass keeps {:?}, mlx-lm {:?}, whose 8th-9th gap {:.3e} ≥ the \
             floor {TIE:.1e}",
            p.step, p.visit, p.ours, p.theirs, p.gap
        ));
    }
    let mut worst: Option<(usize, f64)> = None;
    let mut per_step = Vec::new();
    let mut picks = 0;
    for (s, p) in steps.iter().zip(&passes) {
        let d =
            distance(&p.logits, &s.logits, band).map_err(|e| format!("step {}: {e}", p.step))?;
        println!(
            "k_pass step {} (id {}): max |pass − steps| {:.3e} at {} (pass {:.6}, steps {:.6}), \
             rms {:.3e}; picks {} and {}",
            p.step, pin.ids[p.step], d.max, d.at, d.session, d.stored, d.rms, p.pick, s.pick
        );
        per_step.push(format!("{}: {:.2e}", p.step, d.max));
        picks += usize::from(p.pick == s.pick);
        if worst.is_none_or(|(_, w)| d.max > w) {
            worst = Some((p.step, d.max));
        }
    }
    let (at, max) = worst.ok_or("the pin holds no row")?;
    let line = format!(
        "pass = steps: max |logit| difference {max:.3e} at step {at}; per step [{}]; band {band} \
         (twice the row's: each walk within one of the pin); {picks} of {} picks equal; {} router \
         calls of the passes part from mlx-lm's, each under the near-tie floor; order: the \
         pass's projections and experts sum half products in f32 over K in the GEMM tiles' steps, \
         the step's mat-vecs f32 products a row",
        per_step.join(", "),
        passes.len(),
        parts.len()
    );
    if max <= band {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}

/// `k_pass_pin`.
fn k_pass_pin(session: &mut Session) -> Clause {
    let pin = pin(session)?;
    let band = row::current().logit_abs;
    let mut worst: Option<(usize, f64)> = None;
    let mut per_step = Vec::new();
    let mut over = 0;
    for (h, got) in pass_walk(session, &pin)?.0.iter().enumerate() {
        let want = &pin.rows[h * pin.n..(h + 1) * pin.n];
        let d = distance(&got.logits, want, band)
            .map_err(|e| format!("step {} (id {}): {e}", got.step, pin.ids[got.step]))?;
        println!(
            "k_pass_pin step {} (id {}): max |logit| difference {:.3e} at {} (pass {:.6}, pin \
             {:.6}), {} values over the band, rms {:.3e}",
            got.step, pin.ids[got.step], d.max, d.at, d.session, d.stored, d.over, d.rms
        );
        per_step.push(format!("{}: {:.2e}", got.step, d.max));
        over += d.over;
        if worst.is_none_or(|(_, w)| d.max > w) {
            worst = Some((got.step, d.max));
        }
    }
    let (at, max) = worst.ok_or("the pin holds no row")?;
    let line = format!(
        "passes against pin inkling2: max |logit| difference {max:.3e} at step {at}; per step \
         [{}]; band {band}, {over} values over it",
        per_step.join(", ")
    );
    if max <= band {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}

/// `k_carry`.
fn k_carry(session: &mut Session) -> Clause {
    let pin = pin(session)?;
    let band = 2.0 * row::current().logit_abs;
    let visits = session.checked().visits.len();
    let n = CARRY_ENDS[CARRY_ENDS.len() - 1];
    if session.ctx() < n {
        return Err(format!(
            "a cache of {} rows holds no prompt of {n}",
            session.ctx()
        ));
    }
    let ids: Vec<u32> = pin.ids.iter().copied().cycle().take(n).collect();
    session.reset().map_err(|e| e.to_string())?;
    let mut table: Vec<Route8> = vec![([0; TOP_K], [0.0; TOP_K]); n * visits];
    let mut passed: Vec<(usize, Vec<f32>, u32)> = Vec::with_capacity(CARRY_ENDS.len());
    let mut from = 0usize;
    for &end in &CARRY_ENDS {
        let mut hook = |vi: usize,
                        _probs: &[f32],
                        kept: &[u32],
                        weights: &[f32]|
         -> Result<(Vec<u32>, Vec<f32>), String> {
            for t in 0..end - from {
                let at = (from + t) * visits + vi;
                table[at]
                    .0
                    .copy_from_slice(&kept[t * TOP_K..(t + 1) * TOP_K]);
                table[at]
                    .1
                    .copy_from_slice(&weights[t * TOP_K..(t + 1) * TOP_K]);
            }
            Ok((kept.to_vec(), weights.to_vec()))
        };
        let p = session
            .pass_routed(&ids[from..end], &mut hook)
            .map_err(|e| format!("the pass of positions {from}..{end}: {e}"))?;
        let logits = session.logits_row().map_err(|e| e.to_string())?.to_vec();
        passed.push((end - 1, logits, p.id));
        from = end;
    }
    session.reset().map_err(|e| e.to_string())?;
    let mut worst: Option<(usize, f64)> = None;
    let mut per_pos = Vec::new();
    let mut picks = 0;
    let mut next = 0usize;
    for (k, &id) in ids.iter().enumerate() {
        let mut taps = |_: usize, _: Tap, _: &[f32]| {};
        let mut route = |vi: usize| -> Result<Route8, String> { Ok(table[k * visits + vi]) };
        let pick = session
            .step_routed(id, &mut taps, &mut route)
            .map_err(|e| format!("step {k} (id {id}): {e}"))?;
        if passed.get(next).is_none_or(|p| p.0 != k) {
            continue;
        }
        let (_, want, pass_pick) = &passed[next];
        let got = session.logits_row().map_err(|e| e.to_string())?;
        let d = distance(want, got, band).map_err(|e| format!("position {k}: {e}"))?;
        println!(
            "k_carry position {k}: max |pass − steps| {:.3e} at {} (pass {:.6}, steps {:.6}), rms \
             {:.3e}; picks {pass_pick} and {pick}",
            d.max, d.at, d.session, d.stored, d.rms
        );
        per_pos.push(format!("{k}: {:.2e}", d.max));
        picks += usize::from(*pass_pick == pick);
        if worst.is_none_or(|(_, w)| d.max > w) {
            worst = Some((k, d.max));
        }
        next += 1;
    }
    let (at, max) = worst.ok_or("no pass ended")?;
    let line = format!(
        "passes ending at {CARRY_ENDS:?} against the steps at their routes: max |logit| difference \
         {max:.3e} at position {at}; per position [{}]; band {band}; {picks} of {} picks equal",
        per_pos.join(", "),
        passed.len()
    );
    if max <= band {
        Ok(Verdict::Noted(line))
    } else {
        Err(line)
    }
}

/// `k_visits`.
fn k_visits(session: &Session) -> Clause {
    let plans = &session.checked().visits;
    if plans.len() != VISIT_LAYERS.len() {
        return Err(format!(
            "the session walks {} visits; mlx-lm's plan of the folder is {}",
            plans.len(),
            VISIT_LAYERS.len()
        ));
    }
    for (vi, (v, &layer)) in plans.iter().zip(&VISIT_LAYERS).enumerate() {
        let windowed = !GLOBAL_LAYERS.contains(&layer);
        let boundary = layer.is_multiple_of(BLOCK);
        let message = (FIRST_MESSAGE..VISIT_LAYERS.len() - 1).contains(&vi);
        let got = (v.layer, v.windowed, v.boundary, v.message);
        if got != (layer, windowed, boundary, message) {
            return Err(format!(
                "visit {vi}: the session's (layer, windowed, boundary, message) {got:?}, \
                 mlx-lm's {:?}",
                (layer, windowed, boundary, message)
            ));
        }
    }
    let stored = session.checked().layers.len();
    let boundaries = plans.iter().filter(|v| v.boundary).count();
    let messages = plans.iter().filter(|v| v.message).count();
    Ok(Verdict::Noted(format!(
        "visits {} over {stored} stored layers, {boundaries} boundaries, {messages} message \
         visits, as mlx-lm's",
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
    if let Some(out) = flag(args, "--nodes") {
        return match nodes_print(&mut session, out) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                println!("{GATE} --nodes: {e}");
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
            "k_routes" => k_routes(&mut session),
            "k_logits" => k_logits(&mut session),
            "k_greedy" => k_greedy(&mut session),
            "k_visits" => k_visits(&session),
            "k_room" => k_room(&session, opened),
            "k_pass" => k_pass(&mut session),
            "k_pass_pin" => k_pass_pin(&mut session),
            "k_carry" => k_carry(&mut session),
            "k_serve" => continue,
            other => Err(format!("no clause named {other}")),
        };
        report.clause(name, result);
    }
    if only.is_none_or(|o| o == "k_serve") {
        bloomery_metal::stream::breadcrumb_clause("k_serve");
        let result = match pin(&session) {
            Ok(p) => {
                let prompt: Vec<u32> = p.ids.iter().copied().cycle().take(600).collect();
                serve::k_serve(session, &path, &prompt, &p.ids[..20])
            }
            Err(e) => Err(e),
        };
        report.clause("k_serve", result);
    }
    let scope = match (mutant, only) {
        (Some((m, _)), _) => format!(" (row inkling2, mutant {})", m.name),
        (None, Some(name)) => format!(" (row inkling2, only {name})"),
        (None, None) => " (row inkling2)".to_string(),
    };
    report.finish(&format!("{GATE}{scope}"))
}
