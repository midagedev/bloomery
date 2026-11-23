//! The Inkling 2 end-to-end gate: the whole decode step of the card body
//! (`bloomery_gpu::arch::inkling2::Body`: the visits of the looped trunk, the message bank at the
//! pass edges, the head's mix) over the file the tier names, one prompt id a step, against the
//! patched llama.cpp PR #32695 tree's CPU node dumps of that file (refset families
//! `fx-lcpp-inkling2` on the fixture — thirteen stored layers, 25 visits — and `lcpp-inkling2` on
//! the Mini — 24 stored layers, 48 visits). The model opens through `GpuModel::<Body>::open`.
//!
//! For each prompt set (set `e`, 1,038 ids, runs the windowed visits' rings past their window),
//! against the vendor's nodes at the prompt's last position, the fixture tier's clauses on the
//! fixture's sets and the real tier's (`_real`) on the Mini's:
//!
//! - `layer_outputs_are_vendors` (the fixture): the teacher-forced chain. The prompt one id a
//!   step with every visit's routed picks and weights the vendor's at that position
//!   (`Body::force_picks`), each visit's output within its block's band ([`FORCED_BANDS`]) and
//!   every other stage the vendor names within [`FORCED_STAGE_BANDS`], by relative L2. The fixture's router, controller and clip tensors are uniform random: free, its
//!   close sigmoid scores flip a pick within a visit or two (the free chain leaves 1e-1 by the
//!   second visit); forced, the chain is the two engines' rounding, which the random attention
//!   mixes still grow about twofold across each block join, so a block's visits take that
//!   block's band and the visits past the last band, with the logits row, print only.
//! - `layer_outputs_are_vendors_real` (the Mini): the free chain. The first visit's output
//!   (`l_out-0`) within [`REAL_FREE_BAND`] and the logits row within [`REAL_LOGITS_BAND`].
//!
//!   Both print every visit's [`TAP_STAGES`] against the vendor's nodes, so a miss names the
//!   first stage that left the band.
//! - `forced_top1_is_vendors`: the vendor's own greedy ids fed back one a step (teacher-forced),
//!   our argmax at each of the greedy table's positions against its id: the disagreements at a
//!   vendor margin of [`MARGIN_FLOOR`] or more are at most the pin the Gaussian error model gives
//!   ([`forced_pin`]) at the tier's σ ([`SIGMA_FIXTURE`], [`SIGMA_REAL`]).
//!
//! and the clauses that need no vendor, in both tiers:
//!
//! - `visits_are_the_loop`: the body walks the tier's visits over its stored layers, the
//!   windowless ones at the tier's slots.
//! - `loop_proven`: set `a`'s logits with the messages off (`Body::set_messages`) differ from
//!   those with them on: the message path is live.
//! - `graph_nodes_are_the_walks`: the captured step holds the nodes the walk derives
//!   ([`derived_nodes`], term by term from the body's launches).
//! - `graph_is_eager`: set `a`'s prompt and [`GRAPH_STEPS`] greedy steps in graph mode (the chain
//!   captured once and replayed at every position) give the eager chain's ids and last logits row
//!   bit for bit.
//!
//! Before the card loads, a set absent from the box, dumped from another file or build, or
//! unfinished ends the gate by name. A step that raises the fault word ends the gate by name.
//!
//! What the free chain can hold. The two engines read the same weights and differ by how they
//! round: the vendor's CPU path quantizes each Q8_0 projection's input row in blocks of 32, ours
//! reads it in f32; a visit chains about ten such sites and 48 visits compound them, and the
//! router's 8 of 128 picks and the argmax turn a rounding gap into a different computation once
//! a margin is narrower than it. So the free chain holds the first visit and the logits row
//! loosely, and the teacher-forced argmax holds the chain at every step.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_e2e", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::inkling2::{Body, PAD_LOGIT, TAP_STAGES};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::prompts::{GreedyRow, MARGIN_FLOOR, compare_forced, std_normal_cdf};
    use bloomery_gpu_gates::rel::rel;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, RefManifest, RowKind, bits_equal, checks_failed, open_split,
        ref_tensor_logical_in, topk_ids_logical_within, verdict,
    };
    use model::arch::Arch;
    use model::arch::inkling2::hparams::Hparams;
    use refset::arch::inkling2;
    use refset::arch::prompt_tables::{self, Greedy, Top};
    use refset::family::Family;

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_inkling2_e2e";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-inkling2-e2e";

    /// Positions of every store: the longest prompt (set `e`, 1,038 ids) and its 32 greedy ids
    /// fit, and the windowed visits ring at their window of 1,024 below it.
    const CTX: usize = 2048;

    /// The visit's output among [`TAP_STAGES`].
    const L_OUT: usize = TAP_STAGES.len() - 1;

    /// The visits and stored layers of each tier's file, and its windowless visits' slots.
    const FIXTURE_WALK: (usize, usize, &[usize]) = (25, 13, &[5, 11, 17, 23]);
    const REAL_WALK: (usize, usize, &[usize]) = (48, 24, &[5, 11, 17, 23, 29, 35, 41, 47]);

    /// The forced chain's visit outputs on the fixture, relative L2 from the vendor's
    /// `l_out-<slot>`, by the visit's block (the block joins at or before it, less one).
    /// PIN(2026-11-23): 1.6 times the five sets' measured maxima by block, 4.14e-2, 1.27e-1 and
    /// 2.70e-1 (the next blocks reach 6.8e-1 and 7.1e-1: rows past correlation, printed only).
    const FORCED_BANDS: [f64; 3] = [6.6e-2, 2.0e-1, 4.3e-1];

    /// The forced chain's other [`TAP_STAGES`] on the fixture, each relative L2 from the vendor's
    /// node of that name and visit, by the visit's block as [`FORCED_BANDS`]: a stage the output
    /// does not show (a key row's scale the clip takes back) still answers to the vendor's.
    /// PIN(2026-11-23): 1.6 times the five sets' measured maxima over the stages by block, 7.27e-2
    /// and 2.18e-1 (the attention output of the windowless visits; the next block reaches 6.0e-1).
    const FORCED_STAGE_BANDS: [f64; 2] = [1.2e-1, 3.5e-1];

    /// The Mini's free chain, first visit output, relative L2 from the vendor's `l_out-0`.
    /// PIN(2026-11-23): 1.6 times the measured maximum 1.54e-2 (five sets, 5.5e-3 to 1.54e-2).
    const REAL_FREE_BAND: f64 = 2.5e-2;

    /// The Mini's free chain, logits row, relative L2 from the vendor's `result_output`.
    /// PIN(2026-11-23): 1.5 times the measured maximum 8.11e-2 (five sets, 3.13e-2 to 8.11e-2).
    const REAL_LOGITS_BAND: f64 = 1.2e-1;

    /// One implementation's logit deviation from the other's, in logit units: the σ of the
    /// teacher-forced argmax pin ([`forced_pin`]) on the Mini.
    /// PIN(2026-11-23): above the measured σ̂ of the five sets' last logits rows (0.127 to 0.304).
    const SIGMA_REAL: f64 = 0.5;

    /// [`SIGMA_REAL`] on the fixture, whose free chain flips picks.
    /// PIN(2026-11-23): above the measured σ̂ of the five sets' last logits rows (0.73 to 1.22).
    const SIGMA_FIXTURE: f64 = 1.5;

    /// The greedy steps `graph_is_eager` takes after the prompt.
    const GRAPH_STEPS: usize = 8;

    /// The vendor clauses a run declares: two a set, each a fixture-oracle clause.
    const FX_ORACLE: usize = 2 * inkling2::SETS.len();

    /// The tier's sets: their family and names.
    fn sets_of(tier: Tier) -> (&'static Family, Vec<&'static str>) {
        match tier {
            Tier::Real => (&inkling2::LCPP, inkling2::SETS.to_vec()),
            Tier::Fixture => (
                &inkling2::fixture::LCPP,
                vec![
                    inkling2::fixture::A,
                    inkling2::fixture::B,
                    inkling2::fixture::C,
                    inkling2::fixture::D,
                    inkling2::fixture::E,
                ],
            ),
        }
    }

    /// One set's prompt, its tables and the vendor's nodes at the prompt's last position.
    struct Set {
        label: &'static str,
        prompt: Vec<u32>,
        top: Vec<Top>,
        greedy: Greedy,
        /// Every visit's [`TAP_STAGES`] nodes (`<stage>-<slot>`), slot-major; `None` where the
        /// vendor names no such node (a block's first visit's attention output is its
        /// `ffn_inp`).
        stages: Vec<Vec<Option<Vec<f32>>>>,
        /// The logits row over the unpadded vocabulary (`result_output`).
        logits: Vec<f32>,
        /// Every visit's routed ids and weights at every prompt position (`ffn_moe_topk-<slot>`,
        /// `ffn_moe_weights_scaled-<slot>`), visit-major then position-major; read on the fixture.
        picks: Option<Picks>,
    }

    /// Every visit's routed ids and weights over the prompt, visit-major then position-major.
    type Picks = (Vec<Vec<u32>>, Vec<Vec<f32>>);

    /// The experts and the picks a token of the files.
    const N_EXPERT: u32 = 128;
    const N_USED: usize = 8;

    /// Every occurrence of `name` (a ubatch an occurrence) in order: the prompt's rows.
    fn all_rows(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        let mut out = Vec::new();
        let mut occ = 0u32;
        while let Some(row) = man.find(RowKind::Tensor, name, occ) {
            out.extend(ref_tensor_logical_in(&man.dir, row)?);
            occ += 1;
        }
        Ok(out)
    }

    /// Every visit's vendor picks over the prompt: the ids and weights of `visits` visits at
    /// `tokens` positions, each checked `tokens · N_USED` long.
    fn picks_of(man: &RefManifest, (visits, tokens): (usize, usize)) -> Result<Picks, GateError> {
        let mut ids = Vec::with_capacity(visits);
        let mut ws = Vec::with_capacity(visits);
        for v in 0..visits {
            let topk = format!("ffn_moe_topk-{v}");
            let mut got = Vec::new();
            let mut occ = 0u32;
            while let Some(row) = man.find(RowKind::Tensor, &topk, occ) {
                got.extend(
                    topk_ids_logical_within(man, row, N_EXPERT)?
                        .into_iter()
                        .map(|id| id as u32),
                );
                occ += 1;
            }
            let w = all_rows(man, &format!("ffn_moe_weights_scaled-{v}"))?;
            if got.len() != tokens * N_USED || w.len() != tokens * N_USED {
                return Err(format!(
                    "visit {v}: {} picks and {} weights for {tokens} positions",
                    got.len(),
                    w.len()
                )
                .into());
            }
            ids.push(got);
            ws.push(w);
        }
        Ok((ids, ws))
    }

    /// The last row of `n` values of node `name`'s last occurrence (a ubatch an occurrence; the
    /// token is the outermost dimension), `None` where the dump names no such node.
    fn last_row(man: &RefManifest, name: &str, n: usize) -> Result<Option<Vec<f32>>, GateError> {
        if man.tensor(name, 0).is_err() {
            return Ok(None);
        }
        let mut occ = 0u32;
        while man.tensor(name, occ + 1).is_ok() {
            occ += 1;
        }
        let row = man.tensor(name, occ)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        if all.len() < n || !all.len().is_multiple_of(n) {
            return Err(format!("{name} occurrence {occ} is {:?}, not rows of {n}", row.ne).into());
        }
        Ok(Some(all[all.len() - n..].to_vec()))
    }

    fn open_set(
        tier: Tier,
        family: &'static Family,
        name: &'static str,
        (visits, widths): (usize, &[usize; TAP_STAGES.len()]),
    ) -> Result<Set, GateError> {
        let path = match tier {
            Tier::Fixture => tier::fixture_set("the vendor clauses", family, name)?,
            Tier::Real => family.path(name),
        };
        let man = RefManifest::open(&path, family)?;
        let prompt: Vec<u32> = prompt_tables::tokens(&man)?.to_vec();
        let stages: Vec<Vec<Option<Vec<f32>>>> = (0..visits)
            .map(|v| {
                TAP_STAGES
                    .iter()
                    .zip(widths)
                    .map(|(stage, &w)| last_row(&man, &format!("{stage}-{v}"), w))
                    .collect::<Result<_, _>>()
            })
            .collect::<Result<_, _>>()?;
        if let Some(v) = stages.iter().position(|s| s[L_OUT].is_none()) {
            return Err(format!("{name}: no l_out-{v}").into());
        }
        let vocab = man.tensor("result_output", 0)?.ne[0] as usize;
        let picks = match tier {
            Tier::Fixture => Some(picks_of(&man, (visits, prompt.len()))?),
            Tier::Real => None,
        };
        let set = Set {
            label: name
                .trim_start_matches("fx_ref_inkling2_")
                .trim_start_matches("ref_inkling2_"),
            prompt,
            top: prompt_tables::topk(&man)?,
            greedy: prompt_tables::greedy(&man)?,
            picks,
            stages,
            logits: last_row(&man, "result_output", vocab)?
                .ok_or_else(|| format!("{name}: no result_output"))?,
        };
        if set.greedy.ids.is_empty() || set.greedy.margins.len() != set.greedy.ids.len() {
            return Err(format!(
                "{name}: a greedy table of {} ids and {} margins",
                set.greedy.ids.len(),
                set.greedy.margins.len()
            )
            .into());
        }
        let need = set.prompt.len() + set.greedy.ids.len();
        if need > CTX {
            return Err(format!("{name}: {need} positions, the gate's stores hold {CTX}").into());
        }
        Ok(set)
    }

    /// Whether the vendor clause `clause` of set `label` runs in this tier: the fixture tier's on
    /// the fixture's sets, the real tier's (`_real`) on the Mini's. Both are decided in both
    /// tiers, so each prints the other's line.
    fn vendor_runs(clause: &str, label: &str) -> Result<bool, GateError> {
        let fixture = tier::run_clause(&format!("{clause} set={label}"), Tag::FixtureOracle)?;
        let real = tier::run_clause(&format!("{clause}_real set={label}"), Tag::Oracle)?;
        Ok(fixture || real)
    }

    /// The σ the two logits rows show, over the vendor's row's scale-free difference: the RMS of
    /// their difference in logit units.
    fn sigma_hat(ours: &[f32], theirs: &[f32]) -> f64 {
        let n = theirs.len().max(1) as f64;
        (ours
            .iter()
            .zip(theirs)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    }

    /// The disagreements the Gaussian error of two implementations predicts at the clear
    /// positions of `margins` (those at or past [`MARGIN_FLOOR`]), and the pin that follows: the
    /// expected count plus three standard deviations of it, rounded up.
    ///
    /// [derived]: at the vendor's margin `m` the argmax flips with probability
    /// Φ(−m / (√2 σ)), σ the tier's ([`SIGMA_FIXTURE`], [`SIGMA_REAL`]).
    fn forced_pin(margins: &[f32], sigma: f64) -> (f64, usize) {
        let sigma_pair = sigma * std::f64::consts::SQRT_2;
        let expected: f64 = margins
            .iter()
            .filter(|&&m| m >= MARGIN_FLOOR)
            .map(|&m| std_normal_cdf(-f64::from(m) / sigma_pair))
            .sum();
        (expected, (expected + 3.0 * expected.sqrt()).ceil() as usize)
    }

    /// The vendor clauses of set `s` from a reset model, taps armed: the prompt one id a step,
    /// every visit's output and the logits row against the vendor's (`layer_outputs_are_vendors`),
    /// then the vendor's greedy ids fed back one a step (`forced_top1_is_vendors`).
    /// Each visit's block: the block joins at or before it, less one, a join being every visit
    /// of a stored layer `l` with `l % block == 0` (the reference's `ip % block == 0`).
    fn blocks_of(table: &[(u32, u32)], block: usize) -> Vec<usize> {
        let mut joins = 0usize;
        table
            .iter()
            .map(|&(l, _)| {
                if (l as usize).is_multiple_of(block) {
                    joins += 1;
                }
                joins.saturating_sub(1)
            })
            .collect()
    }

    fn vendor_clauses(
        tier: Tier,
        m: &mut GpuModel<Body>,
        s: &Set,
        block: usize,
    ) -> Result<bool, GateError> {
        let outs_run = vendor_runs("layer_outputs_are_vendors", s.label)?;
        let forced_run = vendor_runs("forced_top1_is_vendors", s.label)?;
        if !outs_run && !forced_run {
            return Ok(true);
        }
        let mut ok = true;
        if outs_run {
            m.reset()?;
            if let Some((ids, ws)) = &s.picks {
                // The fixture's chain with the vendor's picks at every position, one eager step
                // an id (the forced buffers are read at each step's launch).
                m.set_mode(StepMode::Eager);
                let at = |all: &[Vec<u32>], p: usize| -> Vec<u32> {
                    all.iter()
                        .flat_map(|v| v[p * N_USED..(p + 1) * N_USED].iter().copied())
                        .collect()
                };
                for (p, &id) in s.prompt.iter().enumerate() {
                    let w: Vec<f32> = ws
                        .iter()
                        .flat_map(|v| v[p * N_USED..(p + 1) * N_USED].iter().copied())
                        .collect();
                    {
                        let (gpu, _, body) = m.body_parts(NAME)?;
                        body.force_picks(gpu, (&at(ids, p), &w))?;
                    }
                    m.step(&[id])?;
                }
                m.body_parts(NAME)?.2.unforce_picks();
            } else {
                m.set_mode(StepMode::Graph);
                m.step(&s.prompt)?;
            }
            let (taps, widths, blocks) = {
                let (gpu, _, body) = m.body_parts(NAME)?;
                let blocks = blocks_of(&body.visit_table(), block);
                (body.taps(gpu.stream())?, body.tap_widths(), blocks)
            };
            let visits = s.stages.len();
            let row: usize = widths.iter().sum();
            if taps.len() != visits * row || blocks.len() != visits {
                return Err(format!(
                    "{NAME}: {} tapped values and {} walked visits for {visits} visits of {row}",
                    taps.len(),
                    blocks.len()
                )
                .into());
            }
            // Each visit's stages against the vendor's: `rels[v][stage]`, NaN where the vendor
            // names no such node.
            let rels: Vec<Vec<f64>> = (0..visits)
                .map(|v| {
                    let mut at = v * row;
                    widths
                        .iter()
                        .zip(&s.stages[v])
                        .map(|(&w, theirs)| {
                            let ours = &taps[at..at + w];
                            at += w;
                            theirs.as_ref().map_or(f64::NAN, |t| rel(ours, t))
                        })
                        .collect()
                })
                .collect();
            let per: Vec<f64> = rels.iter().map(|r| r[L_OUT]).collect();
            let logits = m.logits()?;
            let ours = &logits[..s.logits.len()];
            let lr = rel(ours, &s.logits);
            let pad_ok = logits[s.logits.len()..]
                .iter()
                .all(|v| v.to_bits() == PAD_LOGIT.to_bits());
            let worst = per.iter().copied().fold(0.0f64, f64::max);
            let (pass, chain) = match tier {
                Tier::Fixture => {
                    // The held values' worst share of their block's band: the output against
                    // FORCED_BANDS, every other stage the vendor names against FORCED_STAGE_BANDS.
                    let (mut held, mut worst) = (0usize, (0.0f64, 0usize, ""));
                    for (v, (r, &b)) in rels.iter().zip(&blocks).enumerate() {
                        for (k, (&x, stage)) in r.iter().zip(TAP_STAGES).enumerate() {
                            let bands: &[f64] = if k == L_OUT {
                                &FORCED_BANDS
                            } else {
                                &FORCED_STAGE_BANDS
                            };
                            if let (Some(band), false) = (bands.get(b), x.is_nan()) {
                                held += 1;
                                if x / band > worst.0 {
                                    worst = (x / band, v, stage);
                                }
                            }
                        }
                    }
                    let pass = worst.0 <= 1.0 && pad_ok;
                    (
                        pass,
                        format!(
                            "forced held={held} worst_share={:.2} at {}-{} (bands {FORCED_BANDS:?} \
                             for l_out, {FORCED_STAGE_BANDS:?} for the other stages, by block)",
                            worst.0, worst.2, worst.1
                        ),
                    )
                }
                Tier::Real => (
                    per[0] <= REAL_FREE_BAND && lr <= REAL_LOGITS_BAND && pad_ok,
                    format!(
                        "free (bands l_out-0 {REAL_FREE_BAND:.1e}, logits {REAL_LOGITS_BAND:.1e})"
                    ),
                ),
            };
            println!(
                "layer_outputs_are_vendors set={} chain={chain} ids={} l_out-0={:.2e} \
                 worst_l_out={worst:.2e} logits={lr:.3e} sigma_hat={:.3} pad_columns={} \
                 pad_ok={pad_ok} {}",
                s.label,
                s.prompt.len(),
                per[0],
                sigma_hat(ours, &s.logits),
                logits.len() - s.logits.len(),
                verdict(pass)
            );
            for (v, r) in rels.iter().enumerate() {
                println!(
                    "  visit {v}: {}",
                    TAP_STAGES
                        .iter()
                        .zip(r)
                        .map(|(stage, x)| format!("{stage}={x:.2e}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            ok &= pass;
        }
        if forced_run {
            m.set_mode(StepMode::Graph);
            m.reset()?;
            let mut table = vec![m.step(&s.prompt)?];
            for &id in &s.greedy.ids[..s.greedy.ids.len() - 1] {
                table.push(m.step(&[id])?);
            }
            let vendor = GreedyRow {
                id: 0,
                n_tokens: s.prompt.len(),
                argmax: s.top.first().map_or(u32::MAX, |t| t.id),
                top5: s.top.iter().take(5).map(|t| t.id).collect(),
                gen_ids: s.greedy.ids.clone(),
                gen_margins: s.greedy.margins.clone(),
            };
            let report = compare_forced(&[table], &[vendor], MARGIN_FLOOR);
            let sigma = match tier {
                Tier::Fixture => SIGMA_FIXTURE,
                Tier::Real => SIGMA_REAL,
            };
            let (expected, pin) = forced_pin(&s.greedy.margins, sigma);
            let pass = report.disagree_ge_floor <= pin;
            println!(
                "forced_top1_is_vendors set={} positions={} disagree={} buckets={:?} ge_floor={} \
                 expected {expected:.2} pin {pin} {}",
                s.label,
                report.positions,
                report.disagree.len(),
                report.buckets,
                report.disagree_ge_floor,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Clause `visits_are_the_loop`: the body's visit count, stored layers and windowless slots
    /// are the tier's file's.
    fn visits_are_the_loop(tier: Tier, body: &Body, hp: &Hparams) -> Result<bool, GateError> {
        if !tier::run_clause("visits_are_the_loop", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let (visits, stored, windowless) = match tier {
            Tier::Fixture => FIXTURE_WALK,
            Tier::Real => REAL_WALK,
        };
        let table = body.visit_table();
        let got: Vec<usize> = table
            .iter()
            .enumerate()
            .filter(|(_, (l, _))| !hp.windowed[*l as usize])
            .map(|(slot, _)| slot)
            .collect();
        let pass = table.len() == visits && body.stored_layers() == stored && got == windowless;
        println!(
            "visits_are_the_loop visits={} (want {visits}) stored={} (want {stored}) \
             windowless={got:?} (want {windowless:?}) store_bytes={} {}",
            table.len(),
            body.stored_layers(),
            body.store_bytes(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `loop_proven`: set `s`'s logits with the messages off differ from those with them
    /// on, each one's distance from the vendor's row printed.
    fn loop_proven(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause("loop_proven", Tag::SelfConsistency)? {
            return Ok(true);
        }
        // Eager: a captured chain is the mode it was captured in.
        m.set_mode(StepMode::Eager);
        let mut run = |on: bool| -> Result<Vec<f32>, GateError> {
            m.body_parts(NAME)?.2.set_messages(on);
            m.reset()?;
            m.step(&s.prompt)?;
            Ok(m.logits()?)
        };
        let off = run(false)?;
        let on = run(true)?;
        let n = s.logits.len();
        let pass = !bits_equal(&off, &on);
        println!(
            "loop_proven set={} messages_on={:.3e} messages_off={:.3e} (logits rel from the \
             vendor's) differ={pass} {}",
            s.label,
            rel(&on[..n], &s.logits),
            rel(&off[..n], &s.logits),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The nodes one captured step holds, derived from the walk (`arch/inkling2/body.rs`) term by
    /// term at one row, and the terms. A mix is the step mix's launch pair (scores, output). A
    /// visit: its attention mix (a layer past the first), the input norms (one on a windowed visit
    /// without messages, else three), six projections plus a windowless visit's two conv taps —
    /// one launch for them all on a layer `norm_proj` names (the body's `norm_proj`) — the
    /// k and v convs, the epilogue, the attention's two launches (segments, merge), the output
    /// gate, the output projection, its post conv and the join (a copy or an add); the
    /// feed-forward's mix, norm and router, the card leg's four launches and its sum, the post
    /// conv and the stream's add; a visit with messages runs the message mix in its attention
    /// mix's pair and adds two controller launch pairs. A pass edge copies the stream (pass 0) or
    /// the bank, its last row and the prelude back (a bank of no row copies nothing). The step
    /// opens with the embedding's lookup and norm and closes with the head's mix, norm,
    /// projection, softcap, padded-column copy and argmax.
    pub(super) fn derived_nodes(
        hp: &Hparams,
        table: &[(u32, u32)],
        norm_proj: &[bool],
    ) -> (usize, String) {
        const MIX: usize = 2;
        const VISIT: usize = 2 + 1 + 2 + 1 + 1 + 1 + 1 + MIX + 1 + 1 + 5 + 1 + 1;
        const MESSAGES: usize = 2 + 2;
        const EMBED: usize = 2;
        const HEAD: usize = MIX + 5;
        let core = hp.loop_.filter(|lp| lp.passes > 1);
        let block = hp.block;
        let (mut visit, mut edges, mut pushed, mut prelude) = (0usize, 0usize, 0usize, 0usize);
        for &(l, pass) in table {
            let l = l as usize;
            let windowed = hp.windowed[l];
            let msg = pass > 0 && core.is_some_and(|lp| (lp.core_start..lp.core_end).contains(&l));
            if let Some(lp) = core
                && l == lp.core_start
            {
                if pass == 0 {
                    prelude = pushed;
                    edges += 1;
                } else {
                    edges += 2 + usize::from(pushed > prelude + 1);
                    pushed = prelude;
                }
            }
            let front = if norm_proj[l] {
                1
            } else {
                6 + if windowed && !msg { 1 } else { 3 } + if windowed { 0 } else { 2 }
            };
            visit += VISIT + front + MIX * usize::from(l != 0) + if msg { MESSAGES } else { 0 };
            if l.is_multiple_of(block) {
                pushed += 1;
            }
        }
        let total = EMBED + visit + edges + HEAD;
        (
            total,
            format!("embed={EMBED} visits={visit} edges={edges} head={HEAD}"),
        )
    }

    /// Clause `graph_nodes_are_the_walks`: the captured step's node count is [`derived_nodes`].
    fn graph_nodes_are_the_walks(m: &mut GpuModel<Body>, hp: &Hparams) -> Result<bool, GateError> {
        if !tier::run_clause("graph_nodes_are_the_walks", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let body = m.body(NAME)?;
        let table = body.visit_table();
        let norm_proj: Vec<bool> = (0..body.stored_layers())
            .map(|l| body.norm_proj(l))
            .collect();
        let (want, terms) = derived_nodes(hp, &table, &norm_proj);
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let nodes = m.capture_step()?;
        let mut hist = std::collections::BTreeMap::<String, usize>::new();
        for n in m.step_graph_nodes()? {
            let key = n
                .kernel
                .map_or_else(|| format!("kind{}", n.kind), |k| k.name);
            *hist.entry(key).or_default() += 1;
        }
        m.set_mode(StepMode::Eager);
        let pass = nodes == want;
        println!(
            "graph_nodes_are_the_walks nodes={nodes} want={want} ({terms}) {}",
            verdict(pass)
        );
        println!(
            "  entries: {}",
            hist.iter()
                .map(|(k, n)| format!("{k}={n}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        Ok(pass)
    }

    /// Clause `graph_is_eager`: set `s`'s prompt and [`GRAPH_STEPS`] greedy steps in graph mode
    /// give the eager chain's ids and last logits row bit for bit.
    fn graph_is_eager(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause("graph_is_eager", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let run =
            |m: &mut GpuModel<Body>, mode: StepMode| -> Result<(Vec<u32>, Vec<f32>), GateError> {
                m.set_mode(mode);
                m.reset()?;
                let mut ids = vec![m.step(&s.prompt)?];
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
            s.label,
            eager.0,
            graph.0,
            verdict(pass)
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        let tier = Tier::from_env()?;
        let split = open_split(Arch::Inkling2, RECIPE)?;
        let hp = Hparams::read(&split).map_err(|e| format!("{NAME}: the file's header: {e}"))?;
        let (family, names) = sets_of(tier);
        let visits = match tier {
            Tier::Fixture => FIXTURE_WALK.0,
            Tier::Real => REAL_WALK.0,
        };
        let (q, kv, r) = (
            hp.n_head[0] * 128,
            hp.n_head_kv[0] * 128,
            hp.n_head[0] * hp.d_rel,
        );
        let n = hp.n_embd;
        let widths = [q, kv, kv, r, q, n, n, n, n, n];
        let sets: Vec<Set> = names
            .iter()
            .map(|&n| open_set(tier, family, n, (visits, &widths)))
            .collect::<Result<_, _>>()?;
        let t = std::time::Instant::now();
        let mut m = match GpuModel::<Body>::open(Gpu::new()?, split, CTX) {
            Ok(m) => m,
            Err(e) => {
                println!("opens {e} {}", verdict(false));
                return Err(e.into());
            }
        };
        if m.body(NAME)?.tap_widths() != widths {
            return Err(format!(
                "{NAME}: the body taps {:?} a visit, the file's widths are {widths:?}",
                m.body(NAME)?.tap_widths()
            )
            .into());
        }
        println!(
            "{NAME}: tier {tier:?}, device {}, resident_bytes={} ctx={CTX}, opened in {:.1} s",
            m.body_parts(NAME)?.0.device_name()?,
            m.resident_bytes(),
            t.elapsed().as_secs_f64()
        );
        let mut ok = visits_are_the_loop(tier, m.body(NAME)?, &hp)?;
        {
            let (gpu, _, body) = m.body_parts(NAME)?;
            body.arm_taps(gpu)?;
        }
        for s in &sets {
            ok &= vendor_clauses(tier, &mut m, s, hp.block)?;
        }
        tier::expect_fixture_oracle(NAME, FX_ORACLE)?;
        m.body_parts(NAME)?.2.disarm_taps();
        ok &= loop_proven(&mut m, &sets[0])?;
        ok &= graph_nodes_are_the_walks(&mut m, &hp)?;
        ok &= graph_is_eager(&mut m, &sets[0])?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} tier {tier:?}: the visits of the loop; on every set the chain's \
             visits and logits within their bands (the fixture's with the vendor's picks, the \
             Mini's free), the teacher-forced argmax disagreements within the pin; the message \
             path live; the captured step's nodes the walk's; the graph the eager chain"
        );
        Ok(())
    }
}
