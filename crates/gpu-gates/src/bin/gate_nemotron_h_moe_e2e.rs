//! The nemotron_h_moe end-to-end gate: the whole decode chain of Nemotron
//! 3.5 Bolt — Mamba-2 layers with NVFP4 projections, grouped-query attention
//! with no rope at sixteen query heads a key head, latent MoE layers of their
//! own expert counts and widths, the head and the argmax — on one card. Its
//! tier is the fixture's (`/models/fixtures/nemotron-bolt`: a Mamba-2 layer,
//! a MoE layer of 240 experts × 1536, an attention layer, a MoE layer of 496
//! × 2560); the real tier adds the mainline clauses below against the real
//! file's set (`ref-nemotron-moe-logits`).
//!
//! The clauses are a list ([`CLAUSES`]); each prints lines ending in PASS or
//! FAIL, and a step that raises the card's fault word is a FAIL naming the
//! fault, never a pass. Each carries its tier tag (`tier::run_clause`):
//!
//! - `structure` (self-consistency): the captured step holds the entries the
//!   layer kinds derive — one Mamba-2 scan and two NVFP4 gemvs a Mamba-2
//!   layer; a group-16 attention layer's `2 · n_kv` group-8 segment passes,
//!   one merge and `2 · n_kv` appends; a MoE layer's router at its own count,
//!   four `iq2_xxs_gemv_sel` (the routed up and down, the shared expert's
//!   up and down), two squared ReLUs and one weighted sum; one Q8_0
//!   embedding row — and no group-5 entry, no host node.
//! - `eager_equals_replay`, `reset_rewinds`, `state_carries`,
//!   `fault_word_clean` (self-consistency): as `gate_nemotron_e2e`'s, over a
//!   16-token walk.
//! - `routing_is_mainlines` (fixture oracle): per prompt and MoE layer, the
//!   body's kept experts at the last position against the set's, as sets
//!   (ggml's top-k gives no order), with two checks of the card against its
//!   own read-back scores: the ids are the top 9 of the scores plus the
//!   file's bias, the weights the kept scores renormalized times the scale.
//!   A set differing from mainline's only in experts within their band of
//!   the body's 9th in its own selection values (the two engines' scores
//!   crossing at the cut: the fixture's near-equal sigmoid scores put the
//!   9th and 10th of 240 or 496 experts within the engines' difference) is
//!   counted and printed; any other set is a FAIL. A layer after one whose
//!   pick crossed runs on another expert's output and is not compared. The band is
//!   `eps · z_max · (s₉(1 − s₉) + s_m(1 − s_m))`: the sigmoid's slope at each
//!   of the two scores times the router-logit error, taken as `eps` (the
//!   body's max |Δ| against mainline relative to the peak logit, over the
//!   prompts that kept mainline's experts in every layer) of the layer's peak
//!   router logit.
//! - `logits_near_mainlines` (fixture oracle): the cosine of the
//!   last-position logits at least [`COSINE_FLOOR`] on every prompt whose
//!   MoE layers all kept mainline's set; a prompt where a pick crossed at the
//!   cut runs another expert from there on, and its cosine and max |Δ| are
//!   printed, not gated. At least one prompt is gated.
//! - `top1_is_mainlines` (fixture oracle): the body's argmax id is mainline's
//!   wherever mainline's own top-1 margin exceeds twice the prompt's max
//!   |Δ| (by the triangle inequality no element's error can then swap the
//!   two, so a differing id is the body's argmax, not the logits); below the
//!   band the two ids and the margin are printed, not gated. The fixture's
//!   logits are nearly flat, so a fixed floor would gate noise. On every
//!   prompt, band or not, the run's recorded id is the argmax of its own
//!   logits row ([`argmax`]) — a row whose id is not its own argmax is
//!   a FAIL.
//! - `routing_is_mainlines_real`, `logits_near_mainlines_real`,
//!   `top1_is_mainlines_real` (oracle): the same three contracts against the
//!   real file's set. The real file's quantized weights put near-ties at the
//!   9-expert cut — no prompt keeps mainline's experts in every layer — so
//!   the routing clause's eps comes from every prompt, a differing layer is
//!   judged at its own cut (a crossing does not cross its prompt off), and
//!   the logits clause gates every prompt; the tiers share one body a pair,
//!   the functions' `real` parameter carrying exactly these differences.
//!
//! - `prefill_near_mainlines` (fixture oracle) and `prefill_near_mainlines_real`
//!   (oracle): every prompt of the set through `GpuModel::prefill` (passes of
//!   many rows) from a reset: the last-position logits keep [`COSINE_FLOOR`]
//!   against mainline's, the step path's cosine printed beside. Not held to
//!   the step path's bits: past eight rows a pass runs its Q8_0 and NVFP4
//!   sites as GEMMs on the rows' q8 blocks of 32 (the NVFP4 ones over a Q8_0
//!   twin of the weight), as mainline's own prompt path quantizes, so the two
//!   paths differ by that rounding. The fixture tier gates a prompt only where
//!   both the step path and the pass kept mainline's experts in every MoE
//!   layer (the pass's routing read after it, its last row's), at least one;
//!   the real tier gates every prompt, as `logits_near_mainlines_real`.
//! - `prefill_then_step_near_mainlines` and `…_real`: per prompt of two ids or
//!   more, all but the last id through `prefill`, the last as one step, under
//!   the same gate: this holds the scan states, the conv rings and the K/V rows
//!   a pass leaves to the step after it. A one-id prompt is skipped by name.
//!
//! A set absent from the box, dumped from another fixture or build, or
//! unfinished ends the gate by name before the card loads.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nemotron_h_moe_e2e", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
mod gate {
    use crate::e2e::{Run, argmax, feed, finite, fresh, histogram, or_fault, same, top_ids, walk};
    use bloomery_gpu::arch::nemotron_h_moe::Routing;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, NemotronHMoeModel};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, verdict};
    use cuda_core::sys;
    use model::arch::Arch;
    use model::arch::nemotron_h_moe::hparams::{Hparams, Kind};
    use refset::arch::nemotron_h_moe::fixture::{LOGITS, LOGITS_SET};
    use refset::arch::nemotron_h_moe::{LOGITS as REAL_LOGITS, LOGITS_SET as REAL_SET};
    use refset::nemotron::LogitsSet;
    use std::collections::BTreeMap;

    type Model = NemotronHMoeModel;

    const NAME: &str = "gate_nemotron_h_moe_e2e";

    /// Rows of the K/V planes a key head holds beyond the longest prompt the
    /// gate runs.
    const CTX_SPARE: usize = 16;

    /// The cosine the last-position logits keep against mainline's.
    const COSINE_FLOOR: f64 = 0.999;

    /// Tokens of the walk every own-contract clause runs.
    const PROMPT_LEN: usize = 16;

    /// The fixture-oracle clauses the gate declares.
    const FIXTURE_ORACLE: usize = 5;

    /// What a clause is run over: the layer kinds and expert counts the
    /// file's header names, the walk, and in the fixture tier the set and
    /// the body's runs of its prompts.
    struct Case {
        hp: Hparams,
        walk: Vec<u32>,
        /// Each MoE layer's selection bias, from the file, in layer order.
        bias: Vec<Vec<f32>>,
        mainline: Option<Mainline>,
    }

    /// The set and, per prompt, the body's run of it.
    struct Mainline {
        set: LogitsSet,
        runs: Vec<Option<SetRun>>,
    }

    /// The body after a prompt of the set, from a reset: its last argmax id,
    /// its last-position logits and each MoE layer's kept ids and weights.
    struct SetRun {
        id: u32,
        logits: Vec<f32>,
        routes: Vec<Routing>,
    }

    /// One clause: its name, its tier tag and its check.
    type Clause = fn(&mut Model, &Case) -> Result<bool, GateError>;

    /// The clauses, in the order they run.
    const CLAUSES: &[(&str, Tag, Clause)] = &[
        ("structure", Tag::SelfConsistency, structure),
        (
            "eager_equals_replay",
            Tag::SelfConsistency,
            eager_equals_replay,
        ),
        ("reset_rewinds", Tag::SelfConsistency, reset_rewinds),
        ("state_carries", Tag::SelfConsistency, state_carries),
        (
            "routing_is_mainlines",
            Tag::FixtureOracle,
            routing_is_mainlines,
        ),
        (
            "logits_near_mainlines",
            Tag::FixtureOracle,
            logits_near_mainlines,
        ),
        ("top1_is_mainlines", Tag::FixtureOracle, top1_is_mainlines),
        // The same three against the real file's mainline set (ref-nemotron-moe-logits), in the
        // real tier, under its own contracts: the file's quantized weights put near-ties at the
        // 9-expert cut, so the routing clause gates crossings, not set equality, and the logits
        // clause holds every prompt.
        (
            "routing_is_mainlines_real",
            Tag::Oracle,
            routing_is_mainlines_real,
        ),
        (
            "logits_near_mainlines_real",
            Tag::Oracle,
            logits_near_mainlines_real,
        ),
        (
            "top1_is_mainlines_real",
            Tag::Oracle,
            top1_is_mainlines_real,
        ),
        (
            "prefill_near_mainlines",
            Tag::FixtureOracle,
            prefill_near_mainlines,
        ),
        (
            "prefill_then_step_near_mainlines",
            Tag::FixtureOracle,
            prefill_then_step_near_mainlines,
        ),
        (
            "prefill_near_mainlines_real",
            Tag::Oracle,
            prefill_near_mainlines_real,
        ),
        (
            "prefill_then_step_near_mainlines_real",
            Tag::Oracle,
            prefill_then_step_near_mainlines_real,
        ),
        ("fault_word_clean", Tag::SelfConsistency, fault_word_clean),
    ];

    fn structure(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        let hp = &case.hp;
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let list = m.step_graph_nodes()?;
        let hist = histogram(&list);
        let ([host], _) = count_kinds(&list, [sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST]);
        let kinds = |k: Kind| hp.kinds.iter().filter(|&&x| x == k).count();
        let (mamba, attn, moe) = (kinds(Kind::Mamba), kinds(Kind::Attention), kinds(Kind::Moe));
        let n_kv = hp.kv.iter().copied().max().unwrap_or(0);
        let got = |name: &str| hist.get(name).copied().unwrap_or(0);
        let mut want: Vec<(String, usize)> = vec![
            ("mamba2_step".into(), mamba),
            ("nvfp4_gemv".into(), 2 * mamba),
            ("norm_quant".into(), mamba + moe),
            ("gqa_flash_merge".into(), attn),
            ("kv_append_pos_buf".into(), 2 * n_kv * attn),
            ("iq2_xxs_gemv_sel".into(), 4 * moe),
            // PIN(2026-10-17): the up rows' quantize applies the squared
            // ReLU in its load, so the step graph holds the relu_sqr rule
            // inside `q3k_quantize_q8_1_relu_sqr` (2 * moe) and no standalone
            // `relu_sqr` entry (was 2 * moe).
            ("q3k_quantize_q8_1_relu_sqr".into(), 2 * moe),
            ("relu_sqr".into(), 0),
            ("weighted_sum".into(), moe),
            ("embed_rows_q8_0".into(), 1),
        ];
        let mut routers: BTreeMap<usize, usize> = BTreeMap::new();
        for x in hp.experts.iter().flatten() {
            *routers.entry(x.n_expert).or_insert(0) += 1;
        }
        for (n, count) in &routers {
            want.push((format!("nemotron_moe_router_{n}"), *count));
        }
        let mut pass = host == 0;
        for (name, n) in &want {
            pass &= got(name) == *n;
        }
        // The group-16 pass is group-8 segment entries over each key head's
        // planes, two a key head; no group-5 entry runs.
        let seg = |g5: bool| -> usize {
            hist.iter()
                .filter(|(k, _)| k.starts_with("gqa_flash_seg") && k.ends_with("_g5") == g5)
                .map(|(_, n)| n)
                .sum()
        };
        let (group8, group5) = (seg(false), seg(true));
        pass &= group8 == 2 * n_kv * attn && group5 == 0;
        let listed: Vec<String> = want
            .iter()
            .map(|(name, n)| format!("{name}={}/{n}", got(name)))
            .collect();
        println!(
            "structure: layers mamba={mamba} attn={attn} moe={moe}, graph_nodes={nodes} host={host}, \
             entries (got/want) {} group8_segments={group8}/{} group5_entries={group5} (want 0) {}",
            listed.join(" "),
            2 * n_kv * attn,
            verdict(pass)
        );
        println!(
            "structure: kernel entries {}",
            hist.iter()
                .map(|(k, n)| format!("{k}={n}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        Ok(pass)
    }

    // ------------------------------------------------ eager_equals_replay

    fn eager_equals_replay(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "eager_equals_replay";
        m.set_mode(StepMode::Eager);
        let eager = fresh(CLAUSE, "eager", m, &case.walk)?;
        m.set_mode(StepMode::Graph);
        let replay = fresh(CLAUSE, "replay", m, &case.walk)?;
        let (Some(eager), Some(replay)) = (eager, replay) else {
            return Ok(false);
        };
        let identical = same(&eager, &replay);
        let pass = identical && finite(&eager);
        println!(
            "{CLAUSE}: walk{PROMPT_LEN} ids {:?}, logits bit-identical {identical}, finite {} {}",
            &eager.ids[eager.ids.len().saturating_sub(4)..],
            finite(&eager),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------ reset_rewinds

    fn reset_rewinds(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "reset_rewinds";
        m.set_mode(StepMode::Graph);
        let first = fresh(CLAUSE, "the first run", m, &case.walk)?;
        let second = fresh(CLAUSE, "the second run", m, &case.walk)?;
        let (Some(first), Some(second)) = (first, second) else {
            return Ok(false);
        };
        let ids = first.ids == second.ids;
        let bits = same(&first, &second);
        let pass = ids && bits;
        println!(
            "{CLAUSE}: walk{PROMPT_LEN} ids equal {ids}, logits bit-identical {bits} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------ state_carries

    /// The first Mamba-2 layer's scan state: finite and not all zero.
    fn scan_state(m: &Model, case: &Case) -> Result<(bool, bool), GateError> {
        let layer = case
            .hp
            .kinds
            .iter()
            .position(|&k| k == Kind::Mamba)
            .ok_or("the file has no Mamba-2 layer")?;
        let state = m.body(NAME)?.scan_state(m.gpu().stream(), layer)?;
        Ok((
            state.iter().all(|v| v.is_finite()),
            state.iter().any(|&v| v != 0.0),
        ))
    }

    fn state_carries(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "state_carries";
        m.set_mode(StepMode::Graph);
        let Some(&last) = case.walk.last() else {
            return Ok(false);
        };
        if fresh(CLAUSE, "the first run", m, &case.walk)?.is_none() {
            return Ok(false);
        }
        let (finite_state, nonzero) = scan_state(m, case)?;
        let Some(after_a) = or_fault(CLAUSE, "the last token", feed(m, &[last]))? else {
            return Ok(false);
        };
        if fresh(CLAUSE, "the second run", m, &case.walk)?.is_none() {
            return Ok(false);
        }
        let Some(after_b) = or_fault(CLAUSE, "the last token", feed(m, &[last]))? else {
            return Ok(false);
        };
        let Some(alone) = fresh(CLAUSE, "the last token alone", m, &[last])? else {
            return Ok(false);
        };
        let twice = after_a.ids == after_b.ids && same(&after_a, &after_b);
        let carried = !same(&after_a, &alone);
        let pass = twice && finite_state && nonzero && carried;
        println!(
            "{CLAUSE}: last token {last} stepped twice from the same reset: ids {:?} vs {:?} equal \
             {twice}; scan state finite {finite_state} nonzero {nonzero}; logits after the walk \
             differ from the token alone {carried} {}",
            after_a.ids,
            after_b.ids,
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------- the mainline set

    /// The body after the set's prompt `p` from a reset, with the MoE
    /// layers' routing read back, or `None` after a line naming the fault.
    fn body_on(
        m: &mut Model,
        hp: &Hparams,
        set: &LogitsSet,
        p: &refset::nemotron::Prompt,
    ) -> Result<Option<SetRun>, GateError> {
        let ids = set.ids(p)?;
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut id = 0;
        for &t in &ids {
            match or_fault("mainline runs", &p.name, m.step(&[t]))? {
                Some(got) => id = got,
                None => return Ok(None),
            }
        }
        let logits = m.logits()?;
        let mut routes = Vec::new();
        for l in moe_layers(hp) {
            routes.push(m.body(NAME)?.routing(m.gpu().stream(), l)?);
        }
        Ok(Some(SetRun { id, logits, routes }))
    }

    /// The cosine and the max |Δ| of `got` against `want`.
    fn compare(got: &[f32], want: &[f32]) -> (f64, f64) {
        let max_delta = got
            .iter()
            .zip(want)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0, f64::max);
        let dot: f64 = got
            .iter()
            .zip(want)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        (dot / (norm(got) * norm(want)), max_delta)
    }

    /// Each prompt of the set with the body's run of it; a prompt whose run
    /// faulted has printed its line and is a FAIL of the clause.
    fn mainline(case: &Case) -> Result<&Mainline, GateError> {
        case.mainline
            .as_ref()
            .ok_or_else(|| "the fixture-oracle clauses ran with no set opened".into())
    }

    /// The MoE layers' indices, in order.
    fn moe_layers(hp: &Hparams) -> Vec<usize> {
        hp.kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == Kind::Moe)
            .map(|(l, _)| l)
            .collect()
    }

    /// The experts of one layer in the body's selection order: the score plus
    /// the bias, larger first, the larger id first on a tie (`route_core::take`
    /// at `LARGER_ID`, the device's rule), with each one's selection value.
    fn ranked(r: &Routing, bias: &[f32]) -> Vec<(u32, f32)> {
        let mut sel: Vec<(u32, f32)> = (0..)
            .zip(r.scores.iter().zip(bias))
            .map(|(e, (s, b))| (e, s + b))
            .collect();
        sel.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
        sel
    }

    /// How one layer's kept experts stand against mainline's.
    enum Routed {
        /// The same set.
        Same,
        /// Mainline keeps `d` experts the body does not, each within its band
        /// of the body's 9th in the body's own selection values: the two
        /// engines' scores crossed at the cut. `worst` is the largest
        /// gap / band.
        Crossed { d: usize, worst: f32 },
        /// A mainline pick past its band, with its rank in the body's order,
        /// its gap and its band.
        Differs {
            id: u32,
            rank: usize,
            gap: f32,
            band: f32,
        },
    }

    /// The band within which two experts' selection values may cross between
    /// the engines: each expert's score moves by `σ'(z) · Δz`, with `σ'(z) =
    /// s (1 − s)` at its own score `s` and `Δz` the body's router-logit
    /// error, taken as `eps` (the body's output error against mainline,
    /// relative to the peak logit, on the prompts that kept mainline's
    /// experts) times the layer's peak router logit `z_max`; the two errors
    /// add.
    fn cross_band(eps: f32, z_max: f32, a: f32, b: f32) -> f32 {
        eps * z_max * (a * (1.0 - a) + b * (1.0 - b))
    }

    /// Layer `r`'s kept set against mainline's `want`, under the body's
    /// selection order `rank`, crossings allowed within [`cross_band`] — a
    /// non-finite band (a saturated sigmoid's recovered logit) is a
    /// difference, never a crossing.
    fn routed(r: &Routing, want: &[u32], rank: &[(u32, f32)], eps: f32) -> Routed {
        let missing: Vec<u32> = want
            .iter()
            .filter(|i| !r.ids.contains(i))
            .copied()
            .collect();
        if missing.is_empty() {
            return Routed::Same;
        }
        let logit = |s: f32| (s / (1.0 - s)).ln();
        let z_max = r
            .scores
            .iter()
            .map(|&s| logit(s).abs())
            .fold(0.0f32, f32::max);
        let k = r.ids.len();
        let (last_id, last) = rank[k - 1];
        let s9 = r.scores[last_id as usize];
        let mut worst = 0.0f32;
        for &m in &missing {
            let Some(p) = rank.iter().position(|(e, _)| *e == m) else {
                return Routed::Differs {
                    id: m,
                    rank: usize::MAX,
                    gap: f32::INFINITY,
                    band: 0.0,
                };
            };
            let gap = last - rank[p].1;
            let band = cross_band(eps, z_max, s9, r.scores[m as usize]);
            if !band.is_finite() || gap.is_nan() || gap > band {
                return Routed::Differs {
                    id: m,
                    rank: p + 1,
                    gap,
                    band,
                };
            }
            worst = worst.max(gap / band);
        }
        Routed::Crossed {
            d: missing.len(),
            worst,
        }
    }

    /// Whether every MoE layer of prompt `p`'s run kept mainline's set.
    fn same_routes(
        ml: &Mainline,
        p: &refset::nemotron::Prompt,
        run: &SetRun,
    ) -> Result<bool, GateError> {
        let Some(want) = ml.set.routes(p)? else {
            return Ok(false);
        };
        Ok(run.routes.iter().zip(&want).all(|(r, w)| {
            let mut a = r.ids.clone();
            let mut b = w.clone();
            a.sort_unstable();
            b.sort_unstable();
            a == b
        }))
    }

    /// Both tiers' body: the cosine of the last-position logits at least
    /// [`COSINE_FLOOR`]. The fixture tier gates only the prompts whose every
    /// MoE layer kept mainline's set — a prompt where a pick crossed at the
    /// cut ([`Routed::Crossed`]) runs another expert from there on, and its
    /// cosine and max |Δ| are printed, not gated, and at least one prompt is
    /// gated. The real tier gates every prompt: on the real set a crossing
    /// stays inside the floor.
    /// PIN(2026-10-16): the fixture's floor 0.999 re-pinned for the real set — 0.999401 to
    /// 0.999998 over its 10 prompts with the crossings included.
    fn logits_near_mainlines_t(
        _m: &mut Model,
        case: &Case,
        clause: &'static str,
        real: bool,
    ) -> Result<bool, GateError> {
        let ml = mainline(case)?;
        let mut ok = true;
        let mut gated = 0;
        for (p, run) in ml.set.prompts.iter().zip(&ml.runs) {
            let Some(run) = run else {
                println!("{clause}: {} faulted FAIL", p.name);
                ok = false;
                continue;
            };
            let want = ml.set.logits(p)?;
            let got = &run.logits;
            if got.len() != want.len() || !got.iter().all(|v| v.is_finite()) {
                println!(
                    "{clause}: {} {} logits for the set's {}, finite {} FAIL",
                    p.name,
                    got.len(),
                    want.len(),
                    got.iter().all(|v| v.is_finite())
                );
                ok = false;
                continue;
            }
            let (cosine, max_delta) = compare(got, &want);
            let values = |ids: &[usize], v: &[f32]| -> Vec<String> {
                ids.iter().map(|&i| format!("{i}:{:.3}", v[i])).collect()
            };
            let line = format!(
                "{clause}: {} {} tokens, max |delta| {max_delta:.4}, cosine {cosine:.6} \
                 (floor {COSINE_FLOOR}), top-5 body {:?} mainline {:?}",
                p.name,
                p.n_ids,
                values(&top_ids(got, 5), got),
                values(&top_ids(&want, 5), &want),
            );
            if real || same_routes(ml, p, run)? {
                gated += 1;
                let pass = cosine >= COSINE_FLOOR;
                println!("{line} {}", verdict(pass));
                ok &= pass;
            } else {
                println!("{line}: a layer's pick crossed mainline's at the cut, not gated");
            }
        }
        let pass = ok && (real || gated > 0);
        if real {
            println!(
                "{clause}: {} prompts gated at the floor {}",
                ml.set.prompts.len(),
                verdict(pass)
            );
        } else {
            println!(
                "{clause}: {gated} of {} prompts gated (every MoE layer mainline's experts) {}",
                ml.set.prompts.len(),
                verdict(pass)
            );
        }
        Ok(pass)
    }

    fn logits_near_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        logits_near_mainlines_t(m, case, "logits_near_mainlines", false)
    }

    /// The same clause against the real file's set, every prompt gated.
    fn logits_near_mainlines_real(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        logits_near_mainlines_t(m, case, "logits_near_mainlines_real", true)
    }

    /// Both tiers' body: the body's argmax id is mainline's wherever
    /// mainline's own top-1 margin exceeds twice the prompt's max |Δ|; the
    /// tiers differ only in the clause name they print.
    fn top1_is_mainlines_t(
        _m: &mut Model,
        case: &Case,
        clause: &'static str,
    ) -> Result<bool, GateError> {
        let ml = mainline(case)?;
        let mut ok = true;
        let mut gated = 0;
        for (p, run) in ml.set.prompts.iter().zip(&ml.runs) {
            let Some(run) = run else {
                println!("{clause}: {} faulted FAIL", p.name);
                ok = false;
                continue;
            };
            let want = ml.set.logits(p)?;
            let (_, max_delta) = compare(&run.logits, &want);
            let top = top_ids(&want, 2);
            let margin = f64::from(want[top[0]]) - f64::from(want[top[1]]);
            let band = 2.0 * max_delta;
            let mainline = top[0] as u32;
            let own = argmax(&run.logits);
            if run.id != own {
                println!(
                    "{clause}: {} run id {} is not its logits' argmax {own} FAIL",
                    p.name, run.id
                );
                ok = false;
            }
            if margin > band {
                gated += 1;
                let pass = run.id == mainline;
                println!(
                    "{clause}: {} body argmax {} mainline {mainline}, margin {margin:.4} > band \
                     {band:.4} (2 x max |delta|) {}",
                    p.name,
                    run.id,
                    verdict(pass)
                );
                ok &= pass;
            } else {
                println!(
                    "{clause}: {} body argmax {} mainline {mainline}, margin {margin:.4} <= band \
                     {band:.4}: not gated",
                    p.name, run.id
                );
            }
        }
        println!(
            "{clause}: {gated} of {} prompts gated {}",
            ml.set.prompts.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    fn top1_is_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        top1_is_mainlines_t(m, case, "top1_is_mainlines")
    }

    /// The same clause against the real file's set.
    fn top1_is_mainlines_real(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        top1_is_mainlines_t(m, case, "top1_is_mainlines_real")
    }

    /// Per prompt and MoE layer: the body's ids are the top 9 of its own
    /// scores plus the file's bias and its weights their renormalized
    /// scores times the scale (the card's selection against its own read-back
    /// scores); and its set is mainline's, or differs only at the cut
    /// ([`Routed::Crossed`]), counted and printed with the gap. The fixture
    /// tier takes its band's eps from the prompts that kept mainline's
    /// experts in every layer and FAILs when there is none, and a layer
    /// after one whose pick crossed runs on another expert's output, so its
    /// set is not mainline's to compare. The real tier's states stay on
    /// mainline's (the logits clause's floor holds with the crossings in),
    /// so it takes the eps from every prompt, compares each differing layer
    /// at its own cut, and prints per prompt how many layers keep
    /// mainline's experts.
    /// PIN(2026-10-16): the real tier's eps comes from every prompt's own
    /// logits against mainline's — no prompt on the real set keeps mainline's
    /// experts in every layer, and the crossings' divergence is inside the
    /// deltas, so the eps is an upper bound: a crossing that dominates its
    /// prompt's max |delta| widens its own band, and the clause's teeth are
    /// the logits floor and top1's argmax, which so wide a crossing cannot
    /// pass.
    fn routing_is_mainlines_t(
        _m: &mut Model,
        case: &Case,
        clause: &'static str,
        real: bool,
    ) -> Result<bool, GateError> {
        let ml = mainline(case)?;
        let moe = moe_layers(&case.hp);
        let layers: Vec<usize> = ml
            .set
            .route_layers
            .iter()
            .map(|r| r.layer as usize)
            .collect();
        if layers != moe {
            println!(
                "{clause}: the set routes layers {layers:?}, the file's MoE layers are {moe:?} FAIL"
            );
            return Ok(false);
        }
        let mut eps = 0.0f32;
        for (p, run) in ml.set.prompts.iter().zip(&ml.runs) {
            if let Some(run) = run
                && (real || same_routes(ml, p, run)?)
            {
                let want = ml.set.logits(p)?;
                let (_, max_delta) = compare(&run.logits, &want);
                let peak = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                eps = eps.max((max_delta / f64::from(peak)) as f32);
            }
        }
        if eps == 0.0 && !real {
            println!("{clause}: no prompt kept mainline's experts in every layer, no band FAIL");
            return Ok(false);
        }
        let source = if real {
            "every prompt"
        } else {
            "the prompts that kept mainline's experts in every layer"
        };
        println!("{clause}: eps {eps:.4} (max |delta| / peak |logit| over {source})");
        if real {
            // Per prompt: how many MoE layers keep mainline's expert set, and the first that
            // does not.
            for (p, run) in ml.set.prompts.iter().zip(&ml.runs) {
                let (Some(run), Some(want)) = (run, ml.set.routes(p)?) else {
                    continue;
                };
                let eq: Vec<bool> = run
                    .routes
                    .iter()
                    .zip(&want)
                    .map(|(r, w)| {
                        let mut a = r.ids.clone();
                        let mut b = w.clone();
                        a.sort_unstable();
                        b.sort_unstable();
                        a == b
                    })
                    .collect();
                let first = eq.iter().position(|e| !e).map(|i| moe[i]);
                println!(
                    "{clause}: {} {} of {} MoE layers keep mainline's experts, first other layer \
                     {first:?}",
                    p.name,
                    eq.iter().filter(|e| **e).count(),
                    eq.len()
                );
            }
        }
        let scale = case.hp.weights_scale;
        let mut ok = true;
        let (mut same, mut boundary) = (0, 0);
        for (p, run) in ml.set.prompts.iter().zip(&ml.runs) {
            let Some(run) = run else {
                println!("{clause}: {} faulted FAIL", p.name);
                ok = false;
                continue;
            };
            let Some(want) = ml.set.routes(p)? else {
                println!("{clause}: {} carries no routes FAIL", p.name);
                ok = false;
                continue;
            };
            // A layer after one whose pick crossed runs on another expert's
            // output: its set is not mainline's to compare. The real tier
            // never crosses off (above), so its crossed_at stays None.
            let mut crossed_at = None;
            for (((l, r), want), bias) in moe.iter().zip(&run.routes).zip(&want).zip(&case.bias) {
                let rank = ranked(r, bias);
                let k = r.ids.len();
                let own: Vec<u32> = rank[..k].iter().map(|(e, _)| *e).collect();
                let picks = own == r.ids;
                let kept: Vec<f64> = r
                    .ids
                    .iter()
                    .map(|&e| f64::from(r.scores[e as usize]))
                    .collect();
                let sum = kept.iter().sum::<f64>();
                let weights = r
                    .weights
                    .iter()
                    .zip(&kept)
                    .all(|(w, s)| (f64::from(*w) - s / sum * f64::from(scale)).abs() <= 1e-5);
                let compared = match crossed_at {
                    Some(c) => Err(c),
                    None => Ok(routed(r, want, &rank, eps)),
                };
                let (verdict_of, kept_ok) = match compared {
                    Err(c) => (format!("after layer {c} crossed: not compared"), true),
                    Ok(Routed::Same) => {
                        same += 1;
                        ("mainline's set".to_string(), true)
                    }
                    Ok(Routed::Crossed { d, worst }) => {
                        boundary += 1;
                        if !real {
                            crossed_at = Some(*l);
                        }
                        let text = format!(
                            "crossed at the cut: {d} of mainline's within the band of the \
                             body's 9th, worst gap/band {worst:.3}"
                        );
                        (text, true)
                    }
                    Ok(Routed::Differs {
                        id,
                        rank,
                        gap,
                        band,
                    }) => (
                        format!(
                            "differs: mainline's {id} at the body's rank {rank}, gap {gap:.3e} \
                             past its band {band:.3e}"
                        ),
                        false,
                    ),
                };
                let pass = picks && weights && kept_ok;
                println!(
                    "{clause}: {} layer {l} body {:?} mainline {want:?}: {verdict_of}; the body's \
                     top 9 of its scores + bias {picks}, weights its renormalized scores x {scale} \
                     {weights} {}",
                    p.name,
                    r.ids,
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        println!(
            "{clause}: {same} layer runs mainline's set, {boundary} crossed at the cut {}",
            verdict(ok)
        );
        Ok(ok)
    }

    fn routing_is_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        routing_is_mainlines_t(m, case, "routing_is_mainlines", false)
    }

    /// The same clause against the real file's set, crossings compared at
    /// their own cuts.
    fn routing_is_mainlines_real(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        routing_is_mainlines_t(m, case, "routing_is_mainlines_real", true)
    }

    // --------------------------------------------------- fault_word_clean

    // ------------------------------------------------- the prompt path

    /// The body after the set's prompt `p` from a reset with every id but the
    /// last `tail` through `prefill` and those `tail` as steps, the MoE
    /// layers' routing read back, or `None` after a line naming the fault.
    fn prefill_on(
        clause: &str,
        m: &mut Model,
        hp: &Hparams,
        (set, p): (&LogitsSet, &refset::nemotron::Prompt),
        tail: usize,
    ) -> Result<Option<SetRun>, GateError> {
        let ids = set.ids(p)?;
        let (head, steps) = ids.split_at(ids.len() - tail);
        m.reset()?;
        let Some(mut id) = or_fault(clause, &p.name, m.prefill(head))? else {
            return Ok(None);
        };
        for &t in steps {
            match or_fault(clause, &p.name, m.step(&[t]))? {
                Some(got) => id = got,
                None => return Ok(None),
            }
        }
        let logits = m.logits()?;
        let mut routes = Vec::new();
        for l in moe_layers(hp) {
            routes.push(m.body(NAME)?.routing(m.gpu().stream(), l)?);
        }
        Ok(Some(SetRun { id, logits, routes }))
    }

    /// Both prompt-path clauses' body: each prompt through [`prefill_on`]
    /// with `tail` ids as steps, its logits against mainline's at
    /// [`COSINE_FLOOR`], gated as the module doc says per tier.
    fn prefill_near_t(
        m: &mut Model,
        case: &Case,
        (clause, real, tail): (&'static str, bool, usize),
    ) -> Result<bool, GateError> {
        let ml = mainline(case)?;
        let mut ok = true;
        let mut gated = 0;
        for (p, step) in ml.set.prompts.iter().zip(&ml.runs) {
            if (p.n_ids as usize) <= tail {
                println!(
                    "{clause}: {} {} id: no pass before its step, skipped",
                    p.name, p.n_ids
                );
                continue;
            }
            let Some(step) = step else {
                println!("{clause}: {} faulted on the step path FAIL", p.name);
                ok = false;
                continue;
            };
            let Some(run) = prefill_on(clause, m, &case.hp, (&ml.set, p), tail)? else {
                ok = false;
                continue;
            };
            let want = ml.set.logits(p)?;
            if run.logits.len() != want.len() || !run.logits.iter().all(|v| v.is_finite()) {
                println!(
                    "{clause}: {} {} logits for the set's {}, finite {} FAIL",
                    p.name,
                    run.logits.len(),
                    want.len(),
                    run.logits.iter().all(|v| v.is_finite())
                );
                ok = false;
                continue;
            }
            let (cosine, max_delta) = compare(&run.logits, &want);
            let (step_cos, step_delta) = compare(&step.logits, &want);
            let line = format!(
                "{clause}: {} {} tokens, argmax {} mainline {} step {}, cosine {cosine:.6} (floor \
                 {COSINE_FLOOR}), max |delta| {max_delta:.4}; the step path's cosine \
                 {step_cos:.6}, max |delta| {step_delta:.4}",
                p.name,
                p.n_ids,
                run.id,
                argmax(&want),
                step.id,
            );
            if real || (same_routes(ml, p, step)? && same_routes(ml, p, &run)?) {
                gated += 1;
                let pass = cosine >= COSINE_FLOOR;
                println!("{line} {}", verdict(pass));
                ok &= pass;
            } else {
                println!("{line}: a layer's pick crossed mainline's at the cut, not gated");
            }
        }
        let pass = ok && gated > 0;
        println!(
            "{clause}: {gated} of {} prompts gated {}",
            ml.set.prompts.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    fn prefill_near_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        prefill_near_t(m, case, ("prefill_near_mainlines", false, 0))
    }

    fn prefill_then_step_near_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        prefill_near_t(m, case, ("prefill_then_step_near_mainlines", false, 1))
    }

    fn prefill_near_mainlines_real(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        prefill_near_t(m, case, ("prefill_near_mainlines_real", true, 0))
    }

    fn prefill_then_step_near_mainlines_real(
        m: &mut Model,
        case: &Case,
    ) -> Result<bool, GateError> {
        prefill_near_t(m, case, ("prefill_then_step_near_mainlines_real", true, 1))
    }

    fn fault_word_clean(m: &mut Model, _case: &Case) -> Result<bool, GateError> {
        let fault = m.gpu().fault()?;
        let pass = fault.is_none() && m.poisoned().is_none();
        println!(
            "fault_word_clean: word {fault:?}, poisoned {:?} {}",
            m.poisoned(),
            verdict(pass)
        );
        Ok(pass)
    }

    // ---------------------------------------------------------------- run

    pub fn run() -> Result<(), GateError> {
        let file = open_split(Arch::NemotronHMoe, "gate-gpu-nemotron-h-moe-e2e")?;
        let hp = Hparams::read(&file).map_err(|e| format!("nemotron_h_moe header: {e}"))?;
        // The set opens before the card loads: one the tier runs, missing or
        // stale, ends the gate here.
        let tier_now = tier::Tier::from_env()?;
        let set = if tier_now.runs(Tag::FixtureOracle) {
            let path = tier::fixture_set("the mainline clauses", &LOGITS, LOGITS_SET)?;
            Some(LogitsSet::open(&path, &LOGITS)?)
        } else if tier_now.runs(Tag::Oracle) {
            Some(LogitsSet::open(&REAL_LOGITS.path(REAL_SET), &REAL_LOGITS)?)
        } else {
            None
        };
        let longest = set
            .iter()
            .flat_map(|s| s.prompts.iter().map(|p| p.n_ids as usize))
            .max()
            .unwrap_or(0)
            .max(PROMPT_LEN);
        let ctx = longest + CTX_SPARE;
        let walk = walk(PROMPT_LEN, hp.n_vocab);
        let bias = moe_layers(&hp)
            .into_iter()
            .map(|l| {
                bloomery_gpu_gates::split_f32(
                    &file,
                    &format!("blk.{l}.exp_probs_b.bias"),
                    hp.experts[l]
                        .expect("a MoE layer's experts are read")
                        .n_expert,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let t = std::time::Instant::now();
        let mut m = Model::open(Gpu::new()?, file, ctx)?;
        println!(
            "load resident_bytes={} ctx={ctx} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        let mainline = match set {
            Some(set) => {
                let mut runs = Vec::with_capacity(set.prompts.len());
                for p in &set.prompts {
                    runs.push(body_on(&mut m, &hp, &set, p)?);
                }
                Some(Mainline { set, runs })
            }
            None => None,
        };
        let case = Case {
            hp,
            walk,
            bias,
            mainline,
        };
        let mut ok = true;
        for (name, tag, clause) in CLAUSES {
            if !tier::run_clause(name, *tag)? {
                continue;
            }
            let pass = clause(&mut m, &case)?;
            if !pass {
                println!("clause {name} FAIL");
            }
            ok &= pass;
        }
        println!("{}", tier::tally_line());
        tier::expect_fixture_oracle(NAME, FIXTURE_ORACLE)?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the captured step holds the entries its layer kinds derive; eager and \
             replay give bit-identical logits; a reset rewinds the ids and the logits bits; the \
             scan state carries; the fault word stays clean; in the fixture tier the logits keep \
             the cosine floor, the argmax is mainline's above the band and every MoE layer keeps \
             mainline's experts on every prompt of the set; the prompt path, alone and before a \
             step, keeps the cosine floor"
        );
        Ok(())
    }
}
