//! The ERNIE 5.5 Flash Preview end-to-end gate: the whole program on one card (the embedding into
//! the four streams, the mHC pre and post around every sub-layer, the HCA and the MLA-DSA
//! attention, the dense SiTU lead and the latent routed blocks, the head's fold, the BF16 head and
//! its MultiMax epilogue) over the fixture cut (`fixture cut`: the released folder's layers 0, 1
//! and 7, an HCA layer with the dense lead, an HCA routed layer and an MLA-DSA routed layer),
//! against the numpy f32 forward's sets of that cut (`tools/ref/ernie55_ref.py`, refset family
//! `fx-np-ernie55`: `short`, 34 ids, and `long`, 2,328 ids, the one prompt past the indexer's
//! top-k). The model opens through `Body::open_placed` of `bloomery-gpu-ernie55` in two arms of
//! the one placement rule:
//!
//! - `card`: the plan of the card in view, which holds every expert of the cut: the prompt runs as
//!   the pass's units (`bloomery_gpu_ernie55::prompt`), the steps as the captured chain;
//! - `hosted`: the same card under a card-experts cap of half the routed experts, so every routed
//!   layer serves the rest of its picks from the common host tier (the card leg's front, shadow
//!   and back around the tier's go and wait): the prompt steps. It opens the cut's GGUF, the file
//!   `bloomery-quantize` writes of the folder, named by [`HOSTED_GGUF`]: the folder's MXFP4
//!   expert stacks are made at load, so no load from the folder splits a stack between the card
//!   and the host. Without the argument the arm is refused by one printed line
//!   ([`HOSTED_REFUSED`]), and the gate's verdict is the card arm's.
//!
//! For each set and arm, against the writer's rows at the prompt's last positions:
//!
//! - `layer_outputs_are_reference`: each layer's four streams after it (`l_out-L`, the last four
//!   rows, the last one on a stepped prompt) within [`L_OUT_BAND`] by relative L2.
//! - `stepped_layer_outputs_are_reference` (an arm whose prompt runs as the pass, the card arm):
//!   the prompt fed again one id a step from a reset, the last step's four streams after each
//!   layer within [`L_OUT_BAND`] of the writer's last row, so the whole-card step chain is held
//!   layer by layer and not by the teacher-forced argmax alone.
//! - `picks_are_reference`: each routed layer's picks at those rows are the writer's ids
//!   (`ffn_moe_topk-L`) as a set, exactly, and their weights by id within [`WEIGHTS_BAND`] of
//!   `ffn_moe_weights-L` (renormalized, times the learned scale).
//! - `selection_is_reference` (`long` alone): the MLA-DSA layer's selected positions at the last
//!   row against the writer's `indexer_top_k-L`: at most [`SEL_SWAPS`] positions of either side
//!   outside the other.
//! - `head_is_reference`: the logits before MultiMax (`result_output_raw`) and after it
//!   (`result_output`) within [`LOGITS_BAND`], and both argmaxes the writer's.
//! - `forced_top1_is_reference`: the writer's greedy ids fed back one a step (teacher-forced),
//!   our argmax at each of the greedy table's positions against its id: the disagreements at a
//!   writer margin of [`MARGIN_FLOOR`] or more at most the pin the Gaussian error model gives at
//!   the σ̂ of the arm's own last prompt row ([`forced_pin`]).
//!
//! and the clauses that need no reference:
//!
//! - `opens_the_fixture`: each arm opens and carries the cut's three layers of their kinds; the
//!   card arm hosts no layer, the hosted arm every routed one.
//! - `graph_is_eager`: `short`'s prompt and [`GRAPH_STEPS`] greedy steps on the card arm in graph
//!   mode (the chain captured once from the reset model, before any token, as a session's open
//!   captures it, and replayed at every position) give the eager chain's ids and last logits row
//!   bit for bit.
//!
//! Before the card loads, a set absent from the box, dumped from another cut or writer, or
//! unfinished ends the gate by name. A step that raises the fault word ends the gate by name.
//!
//! What the bands hold [derived]. The writer runs every formula in f32 over the dequantized
//! weights; the card stores the attention rows in f16 (a value's step 2⁻¹¹ ≈ 4.9e-4 relative)
//! and its quantized projections read their input in 8-bit blocks (a block's step `amax / 127`,
//! about 0.6 % rms of a row). A sub-layer chains up to five such sites, 1.3 % in quadrature, and
//! a layer's streams carry the residual beside it, so the third layer's streams sit within
//! 1.3 % and [`L_OUT_BAND`] is 1.5 times that; the head reads that row normed through one more
//! site ([`LOGITS_BAND`]). The router reads the same row in f32 ([`WEIGHTS_BAND`]). Two discrete
//! choices turn a rounding gap into another computation once a margin is narrower than it: a
//! routed row's tenth and eleventh scores, and the indexer's cut. The fixture's sets are fixed,
//! so a pick at such a tie is red on every run, never by chance; its line prints both sides.
//!
//! The real tier ends by name before anything loads: the released folder's sets
//! (`ref_ernie55_{short,long}`) have no family yet.

#[cfg(feature = "ernie55")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_ernie55_e2e", gate::run())
}

#[cfg(feature = "ernie55")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "ernie55")]
#[path = "shared/serve_seats/folder.rs"]
#[allow(
    dead_code,
    reason = "the shared folder open serves several seats; this gate takes open and model"
)]
mod folder;

#[cfg(feature = "ernie55")]
mod gate {
    use crate::{folder, gate_card};
    use bloomery_gpu::GpuModel;
    use bloomery_gpu::folder::Model;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_ernie55::{Body, PASS_ROWS, TapRows};
    use bloomery_gpu_gates::prompts::{
        GreedyRow, MARGIN_FLOOR, compare_forced, read_greedy, std_normal_cdf,
    };
    use bloomery_gpu_gates::rel::rel;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, Layout, RefManifest, RowKind, bits_equal, checks_failed, ref_ints,
        ref_tensor_logical_in, verdict,
    };
    use bloomery_levers::HostCfg;
    use gguf::Split;
    use model::arch::ernie55::place::PlanInputs;
    use model::placement::PlanLevers;
    use model::placement::workstation;
    use refset::arch::ernie55::fixture::{LONG, NP, SHORT};
    use std::path::{Path, PathBuf};

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_ernie55_e2e";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-ernie55-e2e";
    /// The fixture table's and the sets' name of the model.
    const ARCH: &str = "ernie5_5";
    /// The sets of the writer's family of the cut ([`NP`]), in the order they run; the second is
    /// the one past the indexer's top-k.
    const SETS: [&str; 2] = [SHORT, LONG];

    /// The cut's layers and their kinds, (latent MLA-DSA, dense lead), in layer order: the
    /// released folder's layers 0, 1 and 7.
    const KINDS: [(bool, bool); 3] = [(false, true), (false, false), (true, false)];

    /// Positions of every store: `long`'s 2,328 ids and its greedy ids fit.
    const CTX: usize = 4096;
    /// The rows the writer keeps of each layer: the prompt's last four.
    const TAP_ROWS: usize = 4;

    /// A layer's streams, relative L2 from the writer's `l_out-L` [derived: the module doc].
    const L_OUT_BAND: f64 = 2.0e-2;
    /// The logits before and after MultiMax, relative L2 [derived: the module doc].
    const LOGITS_BAND: f64 = 3.0e-2;
    /// A routed row's pick weights by id, relative L2 [derived: the module doc].
    const WEIGHTS_BAND: f64 = 2.0e-2;
    /// The MLA-DSA layer's selected positions either side holds that the other does not, of the
    /// writer's 2,048 [derived: positions whose score lies within the row's error of the cut's;
    /// the first run reads the count].
    const SEL_SWAPS: usize = 16;
    /// The floor of one logit's deviation in the forced pin, in logit units: an arm whose last
    /// prompt row matches the writer's more closely is pinned at this σ.
    const SIGMA_FLOOR: f64 = 1.0e-3;

    /// The greedy steps `graph_is_eager` takes after the prompt.
    const GRAPH_STEPS: usize = 8;

    /// The reference clauses of an arm: four a set, and `long`'s selection.
    const FX_ARM: usize = 4 * SETS.len() + 1;

    /// The reference clauses of the card arm: [`FX_ARM`] and the stepped layer outputs a set, the
    /// card arm's prompt running as the pass. The hosted arm, where it opens, adds [`FX_ARM`].
    const FX_CARD: usize = FX_ARM + SETS.len();

    /// The argument that names the hosted arm's file, the cut's GGUF.
    const HOSTED_GGUF: &str = "--hosted-gguf";

    /// The line a run without [`HOSTED_GGUF`] prints in place of the hosted arm.
    const HOSTED_REFUSED: &str = "opens_the_fixture arm=hosted REFUSED: the folder's MXFP4 expert \
        stacks are made at load, so no load from the folder splits them between the card and the \
        host; a hosted load needs the cut's GGUF bloomery-quantize writes (--hosted-gguf <file>)";

    /// The host tier's configuration the load opens with: no lever is read.
    const HOST_CFG: HostCfg = HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// One set's prompt, greedy table and the writer's rows at the prompt's last positions.
    struct Set {
        name: &'static str,
        prompt: Vec<u32>,
        greedy: GreedyRow,
        /// Per layer, the last rows' four streams, row-major `[rows, 4, hidden]`.
        l_out: Vec<Vec<f32>>,
        /// Per layer, the last rows' picks and weights, row-major `[rows, n_used]`; empty on the
        /// dense lead.
        picks: Vec<(Vec<u32>, Vec<f32>)>,
        /// Per layer, the selected positions at the last row, ascending; `None` where not dumped.
        sel: Vec<Option<Vec<u32>>>,
        raw: Vec<f32>,
        logits: Vec<f32>,
    }

    /// Integer row `name` of `man`, each value a `u32`.
    fn ids(man: &RefManifest, name: &str) -> Result<Vec<u32>, GateError> {
        ref_ints(man, name, 0, RowKind::Tensor, Layout::Flat)?
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|e| GateError::from(format!("{name}: {v}: {e}"))))
            .collect()
    }

    fn open_set(clause: &str, name: &'static str) -> Result<Set, GateError> {
        let path = tier::fixture_set(clause, &NP, name)?;
        let man = RefManifest::open(&path, &NP)?;
        let prompt = refset::arch::prompt_tables::tokens(&man)?.to_vec();
        let greedy = read_greedy(&man.dir.join("greedy.tsv"))?
            .into_iter()
            .next()
            .ok_or_else(|| format!("{name}: greedy.tsv holds no row"))?;
        if greedy.gen_ids.is_empty() || greedy.gen_margins.len() != greedy.gen_ids.len() {
            return Err(format!(
                "{name}: a greedy table of {} ids and {} margins",
                greedy.gen_ids.len(),
                greedy.gen_margins.len()
            )
            .into());
        }
        let f32s = |n: &str| -> Result<Vec<f32>, GateError> {
            Ok(ref_tensor_logical_in(&man.dir, man.tensor(n, 0)?)?)
        };
        let (mut l_out, mut picks, mut sel) = (Vec::new(), Vec::new(), Vec::new());
        for (l, &(_, dense)) in KINDS.iter().enumerate() {
            l_out.push(f32s(&format!("l_out-{l}"))?);
            picks.push(if dense {
                (Vec::new(), Vec::new())
            } else {
                (
                    ids(&man, &format!("ffn_moe_topk-{l}"))?,
                    f32s(&format!("ffn_moe_weights-{l}"))?,
                )
            });
            let top = format!("indexer_top_k-{l}");
            sel.push(match man.find(RowKind::Tensor, &top, 0) {
                Some(_) => Some(ids(&man, &top)?),
                None => None,
            });
        }
        let need = prompt.len() + greedy.gen_ids.len();
        if need > CTX {
            return Err(format!("{name}: {need} positions, the gate's stores hold {CTX}").into());
        }
        Ok(Set {
            name,
            prompt,
            greedy,
            l_out,
            picks,
            sel,
            raw: f32s("result_output_raw")?,
            logits: f32s("result_output")?,
        })
    }

    /// An arm: its name and the model it opened.
    struct Arm {
        name: &'static str,
        m: GpuModel<Body>,
    }

    /// The hosted arm's file: the value of [`HOSTED_GGUF`], the gate's one argument.
    fn hosted_gguf() -> Result<Option<PathBuf>, GateError> {
        let usage = format!("{NAME} [{HOSTED_GGUF} <the cut's GGUF>]");
        let mut path = None;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            if arg != HOSTED_GGUF {
                return Err(format!("unknown argument {arg:?}: {usage}").into());
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{HOSTED_GGUF} needs a value: {usage}"))?;
            path = Some(PathBuf::from(v));
        }
        Ok(path)
    }

    /// The fixture cut opened on the card in view, from `gguf` where given and else from the cut
    /// folder, under a card-experts cap of half the routed experts where `capped`: the plan,
    /// whether it hosts experts, and the model (with the pass where the card holds every expert).
    fn open_arm(
        name: &'static str,
        capped: bool,
        gguf: Option<&Path>,
    ) -> Result<(Arm, bool), GateError> {
        let model: Model = match gguf {
            Some(path) => {
                let split = Split::open(path)
                    .map_err(|e| format!("{NAME} {name}: open {}: {e}", path.display()))?;
                Tier::from_env()?.check_file(path, split.iter_kv().map(|(k, _)| k))?;
                split.into()
            }
            None => {
                let f = folder::open(&tier::cut(ARCH)?, ARCH)?;
                folder::model(f.header, f.source)?
            }
        };
        let inputs = PlanInputs::read(model.header())
            .map_err(|e| format!("{NAME} {name}: the cut's description: {e}"))?;
        let hp = &inputs.hp;
        let routed = (0..hp.n_layer).filter(|&l| !hp.dense(l)).count() as u64;
        let levers = PlanLevers {
            card_experts_cap: capped.then_some(routed * hp.experts.n_expert as u64 / 2),
            ..PlanLevers::default()
        };
        let machine = workstation::plan_on(gate_card::room()?, hp.n_layer);
        let plan = Body::plan(&inputs, &machine, CTX as u64, &levers)?;
        let hosts = Body::plan_hosts(&plan, &inputs);
        let rows = (!hosts).then_some(PASS_ROWS);
        let m = Body::open_placed(model, (&plan, &inputs), 0, HOST_CFG, (rows, Residency::Off))?;
        Ok((Arm { name, m }, hosts))
    }

    /// `ids` from the model's position: the pass where the load has one, else one id a step.
    fn feed(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<u32, GateError> {
        if m.body(NAME)?.pass_rows().is_some() {
            Ok(bloomery_gpu_ernie55::prompt(m, ids)?)
        } else {
            Ok(m.step(ids)?)
        }
    }

    /// The taps of the last walk.
    fn taps(m: &mut GpuModel<Body>) -> Result<TapRows, GateError> {
        let (gpu, _, body) = m.body_parts(NAME)?;
        Ok(body.taps(gpu.stream())?)
    }

    /// The last `n` of `all`'s rows of `width` values.
    fn last_rows(all: &[f32], width: usize, n: usize) -> &[f32] {
        &all[all.len().saturating_sub(n * width)..]
    }

    /// Clause `layer_outputs_are_reference` of the walk `t` left.
    fn layer_outputs(arm: &str, s: &Set, t: &TapRows) -> Result<bool, GateError> {
        let mut ok = true;
        for (l, theirs) in s.l_out.iter().enumerate() {
            let width = theirs.len() / TAP_ROWS;
            let r = rel(&t.l_out[l], last_rows(theirs, width, t.rows));
            let pass = r <= L_OUT_BAND;
            ok &= pass;
            println!(
                "layer_outputs_are_reference set={} arm={arm} layer={l} rows={} rel={r:.3e} \
                 band={L_OUT_BAND:.1e} {}",
                s.name,
                t.rows,
                verdict(pass)
            );
        }
        Ok(ok)
    }

    /// Clause `stepped_layer_outputs_are_reference` on an arm whose prompt runs as the pass: the
    /// prompt fed one id a step from a reset, the last step's layer outputs against the writer's
    /// last row ([`layer_outputs`]).
    fn stepped_layer_outputs(arm: &mut Arm, s: &Set) -> Result<bool, GateError> {
        arm.m.reset()?;
        arm.m.step(&s.prompt)?;
        let t = taps(&mut arm.m)?;
        layer_outputs(&format!("{}-stepped", arm.name), s, &t)
    }

    /// Picks as `(id, weight)` sorted by id.
    fn by_id(ids: &[u32], ws: &[f32]) -> Vec<(u32, f32)> {
        let mut v: Vec<(u32, f32)> = ids.iter().copied().zip(ws.iter().copied()).collect();
        v.sort_unstable_by_key(|p| p.0);
        v
    }

    /// Clause `picks_are_reference` of the walk `t` left: per routed layer and tapped row.
    fn picks(arm: &str, s: &Set, t: &TapRows, k: usize) -> Result<bool, GateError> {
        let mut ok = true;
        for (l, (ids, ws)) in s.picks.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let (their_ids, their_ws) = (last_rows_u(ids, k, t.rows), last_rows(ws, k, t.rows));
            for r in 0..t.rows {
                let row = r * k..(r + 1) * k;
                let ours = by_id(&t.ids[l][row.clone()], &t.weights[l][row.clone()]);
                let theirs = by_id(&their_ids[row.clone()], &their_ws[row]);
                let same = ours.iter().map(|p| p.0).eq(theirs.iter().map(|p| p.0));
                let wr = if same {
                    let a: Vec<f32> = ours.iter().map(|p| p.1).collect();
                    let b: Vec<f32> = theirs.iter().map(|p| p.1).collect();
                    rel(&a, &b)
                } else {
                    f64::INFINITY
                };
                let pass = same && wr <= WEIGHTS_BAND;
                ok &= pass;
                let only = |x: &[(u32, f32)], y: &[(u32, f32)]| -> Vec<(u32, f32)> {
                    x.iter()
                        .copied()
                        .filter(|p| !y.iter().any(|q| q.0 == p.0))
                        .collect()
                };
                println!(
                    "picks_are_reference set={} arm={arm} layer={l} row={r} same_ids={same} \
                     weights_rel={wr:.3e} band={WEIGHTS_BAND:.1e} ours_only={:?} \
                     reference_only={:?} {}",
                    s.name,
                    only(&ours, &theirs),
                    only(&theirs, &ours),
                    verdict(pass)
                );
            }
        }
        Ok(ok)
    }

    /// The last `n` of `all`'s rows of `width` ids.
    fn last_rows_u(all: &[u32], width: usize, n: usize) -> &[u32] {
        &all[all.len().saturating_sub(n * width)..]
    }

    /// Clause `selection_is_reference`: each layer the set dumps a selection of, at the last row.
    fn selection(arm: &str, s: &Set, m: &mut GpuModel<Body>) -> Result<bool, GateError> {
        let mut ok = true;
        for (l, theirs) in s.sel.iter().enumerate() {
            let Some(theirs) = theirs else { continue };
            let (gpu, _, body) = m.body_parts(NAME)?;
            let ours = body.selection(gpu.stream(), l)?;
            let (swaps, n) = match &ours {
                Some(v) => {
                    let a: std::collections::BTreeSet<u32> = v.iter().copied().collect();
                    let b: std::collections::BTreeSet<u32> = theirs.iter().copied().collect();
                    (
                        a.difference(&b).count().max(b.difference(&a).count()),
                        v.len(),
                    )
                }
                None => (usize::MAX, 0),
            };
            let pass = n == theirs.len() && swaps <= SEL_SWAPS;
            ok &= pass;
            println!(
                "selection_is_reference set={} arm={arm} layer={l} ours={n} reference={} \
                 swaps={} pin={SEL_SWAPS} {}",
                s.name,
                theirs.len(),
                if ours.is_some() {
                    swaps.to_string()
                } else {
                    "none".to_string()
                },
                verdict(pass)
            );
        }
        Ok(ok)
    }

    /// The root mean square of the logits' difference: one logit's deviation, σ̂.
    fn sigma_hat(ours: &[f32], theirs: &[f32]) -> f64 {
        let s: f64 = ours
            .iter()
            .zip(theirs)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        (s / ours.len().max(1) as f64).sqrt()
    }

    /// The index of the largest value, the first of equals.
    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0
    }

    /// Clause `head_is_reference` of the prompt's last row: σ̂ of the MultiMax logits for the
    /// forced pin, and whether the clause holds.
    fn head(arm: &str, s: &Set, t: &TapRows, logits: &[f32]) -> (f64, bool) {
        let (r_raw, r_mm) = (rel(&t.raw, &s.raw), rel(logits, &s.logits));
        let (a_raw, a_mm) = (argmax(&t.raw), argmax(logits));
        let want = s.greedy.argmax as usize;
        let pass =
            r_raw <= LOGITS_BAND && r_mm <= LOGITS_BAND && a_raw == argmax(&s.raw) && a_mm == want;
        let sigma = sigma_hat(logits, &s.logits);
        println!(
            "head_is_reference set={} arm={arm} raw_rel={r_raw:.3e} multimax_rel={r_mm:.3e} \
             band={LOGITS_BAND:.1e} argmax_raw={a_raw}/{} argmax={a_mm}/{want} sigma_hat={sigma:.3e} {}",
            s.name,
            argmax(&s.raw),
            verdict(pass)
        );
        (sigma, pass)
    }

    /// The disagreement count the Gaussian error model expects at the writer's greedy margins at
    /// one logit's deviation `sigma`, and the pin: that count plus three standard deviations of
    /// it, rounded up.
    ///
    /// [derived]: at the writer's margin `m` the argmax flips with probability Φ(−m / (√2 σ)).
    fn forced_pin(margins: &[f32], sigma: f64) -> (f64, usize) {
        let sigma_pair = sigma.max(SIGMA_FLOOR) * std::f64::consts::SQRT_2;
        let expected: f64 = margins
            .iter()
            .filter(|&&m| m >= MARGIN_FLOOR)
            .map(|&m| std_normal_cdf(-f64::from(m) / sigma_pair))
            .sum();
        (expected, (expected + 3.0 * expected.sqrt()).ceil() as usize)
    }

    /// Clause `forced_top1_is_reference` from a reset model: the prompt, then the writer's greedy
    /// ids fed back one a step.
    fn forced_top1(arm: &mut Arm, s: &Set, sigma: f64) -> Result<bool, GateError> {
        let m = &mut arm.m;
        m.reset()?;
        let mut table = vec![feed(m, &s.prompt)?];
        for &id in &s.greedy.gen_ids[..s.greedy.gen_ids.len() - 1] {
            table.push(m.step(&[id])?);
        }
        let report = compare_forced(&[table], std::slice::from_ref(&s.greedy), MARGIN_FLOOR);
        let (expected, pin) = forced_pin(&s.greedy.gen_margins, sigma);
        let pass = report.disagree_ge_floor <= pin;
        println!(
            "forced_top1_is_reference set={} arm={} positions={} disagree={} buckets={:?} \
             ge_floor={} expected {expected:.2} pin {pin} {}",
            s.name,
            arm.name,
            report.positions,
            report.disagree.len(),
            report.buckets,
            report.disagree_ge_floor,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Whether `tier` runs reference clause `clause` of set `s` and arm `arm` (counted either way).
    fn runs(clause: &str, s: &Set, arm: &str) -> Result<bool, GateError> {
        tier::run_clause(
            &format!("{clause} set={} arm={arm}", s.name),
            Tag::FixtureOracle,
        )
    }

    /// The reference clauses of set `s` on `arm`, in the order their state needs: the prompt from
    /// a reset and its taps, the stepped prompt from a reset on an arm with a pass, then the
    /// forced argmax from a reset.
    fn reference_clauses(arm: &mut Arm, s: &Set) -> Result<bool, GateError> {
        let a = arm.name;
        arm.m.reset()?;
        feed(&mut arm.m, &s.prompt)?;
        let t = taps(&mut arm.m)?;
        let logits = arm.m.logits()?;
        let k = arm.m.body(NAME)?.shape().n_used;
        let mut ok = true;
        if runs("layer_outputs_are_reference", s, a)? {
            ok &= layer_outputs(a, s, &t)?;
        }
        if runs("picks_are_reference", s, a)? {
            ok &= picks(a, s, &t, k)?;
        }
        if s.sel.iter().any(Option::is_some) && runs("selection_is_reference", s, a)? {
            ok &= selection(a, s, &mut arm.m)?;
        }
        let mut sigma = SIGMA_FLOOR;
        if runs("head_is_reference", s, a)? {
            let (sg, pass) = head(a, s, &t, &logits);
            sigma = sg;
            ok &= pass;
        }
        if arm.m.body(NAME)?.pass_rows().is_some()
            && runs("stepped_layer_outputs_are_reference", s, a)?
        {
            ok &= stepped_layer_outputs(arm, s)?;
        }
        if runs("forced_top1_is_reference", s, a)? {
            ok &= forced_top1(arm, s, sigma)?;
        }
        Ok(ok)
    }

    /// Clause `graph_is_eager`: `s`'s prompt and [`GRAPH_STEPS`] greedy steps in graph mode give
    /// the eager chain's ids and last logits row bit for bit, the chain captured from the reset
    /// model before any token, as a session's open captures it.
    fn graph_is_eager(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause("graph_is_eager", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let run =
            |m: &mut GpuModel<Body>, mode: StepMode| -> Result<(Vec<u32>, Vec<f32>), GateError> {
                m.set_mode(mode);
                m.reset()?;
                if mode == StepMode::Graph {
                    m.capture_step()?;
                }
                let mut ids = vec![feed(m, &s.prompt)?];
                while ids.len() <= GRAPH_STEPS {
                    let last = ids[ids.len() - 1];
                    ids.push(m.step(&[last])?);
                }
                Ok((ids, m.logits()?))
            };
        let eager = run(m, StepMode::Eager)?;
        let graph = run(m, StepMode::Graph)?;
        m.set_mode(StepMode::Eager);
        let same_bits = bits_equal(&eager.1, &graph.1);
        let pass = eager.0 == graph.0 && same_bits;
        println!(
            "graph_is_eager set={} eager={:?} graph={:?} logits_same_bits={same_bits} {}",
            s.name,
            eager.0,
            graph.0,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `opens_the_fixture` of one arm: its kinds, and its hosted layers those `want_hosts`
    /// asks (none, or every routed layer).
    fn opens(arm: &Arm, hosts: bool, want_hosts: bool) -> Result<bool, GateError> {
        let body = arm.m.body(NAME)?;
        let kinds = body.kinds();
        let hosted = body.hosted();
        let routed: Vec<usize> = (0..KINDS.len()).filter(|&l| !KINDS[l].1).collect();
        let want: Vec<usize> = if want_hosts { routed } else { Vec::new() };
        let pass = kinds == KINDS && hosts == want_hosts && hosted == want;
        println!(
            "opens_the_fixture arm={} kinds={kinds:?} plan_hosts={hosts} hosted={hosted:?} \
             pass_rows={:?} resident_bytes={} {}",
            arm.name,
            body.pass_rows(),
            arm.m.resident_bytes(),
            verdict(pass)
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        let gguf = hosted_gguf()?;
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the released folder's sets (ref_ernie55_short, ref_ernie55_long) have no \
                 refset family yet, so the gate runs the fixture tier: BLOOMERY_TIER=fixture just \
                 {RECIPE}"
            )
            .into());
        }
        let sets: Vec<Set> = SETS
            .iter()
            .map(|&n| open_set("the reference clauses", n))
            .collect::<Result<_, _>>()?;
        let opened = tier::run_clause("opens_the_fixture", Tag::SelfConsistency)?;
        let (mut card, card_hosts) = open_arm("card", false, None)?;
        let mut hosted = match &gguf {
            Some(path) => Some(open_arm("hosted", true, Some(path))?),
            None => {
                println!("{HOSTED_REFUSED}");
                None
            }
        };
        let mut ok = true;
        if opened {
            ok &= opens(&card, card_hosts, false)?;
            if let Some((arm, hosts)) = &hosted {
                ok &= opens(arm, *hosts, true)?;
            }
            if !ok {
                return Err(checks_failed());
            }
        }
        let mut arms = vec![&mut card];
        arms.extend(hosted.as_mut().map(|(arm, _)| arm));
        for arm in arms {
            let (gpu, _, body) = arm.m.body_parts(NAME)?;
            body.arm_taps(gpu, TAP_ROWS)?;
            for s in &sets {
                ok &= reference_clauses(arm, s)?;
            }
        }
        let arms = if hosted.is_some() {
            "card and hosted"
        } else {
            "card, hosted refused"
        };
        tier::expect_fixture_oracle(NAME, FX_CARD + hosted.as_ref().map_or(0, |_| FX_ARM))?;
        drop(hosted);
        ok &= graph_is_eager(&mut card.m, &sets[0])?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} on {SETS:?}, arms {arms}: every layer within \
             {L_OUT_BAND:.1e}, the writer's picks with weights within {WEIGHTS_BAND:.1e}, the \
             selection within {SEL_SWAPS} swaps, the logits before and after MultiMax within \
             {LOGITS_BAND:.1e}, the teacher-forced argmax within the pin; the graph is the eager \
             chain"
        );
        Ok(())
    }
}
