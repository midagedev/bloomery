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
//!   ([`derived_nodes`], term by term from the body's launches), LoopCD-Hidden off and on.
//! - `graph_is_eager`: set `a`'s prompt and [`GRAPH_STEPS`] greedy steps in graph mode (the chain
//!   captured once and replayed at every position) give the eager chain's ids and last logits row
//!   bit for bit.
//! - `rollback_is_a_fresh_run`: the longest set's ids stepped to the position a windowed visit's
//!   ring is [`PASS_ROWS`] rows short of wrapping, then `k` = 1 to [`PASS_ROWS`] cycles of: roll
//!   back `k` positions, step those `k` ids again and [`ROLLBACK_ADVANCE`] new ones, each step's
//!   id and logits row bit for bit the row of a run that never rolled back. A rollback past the
//!   reach is refused by name and the model stands where it did; a rollback is as deep as its
//!   written end, not where the model stands (a rollback of 3 and another of 6 more is nine
//!   below the end, so it is refused, where a rollback of 5 more is taken and replays the fresh
//!   run). A prompt pass moves the written end as steps do: after one, a rollback of
//!   [`PASS_ROWS`] positions is taken and one more is refused.
//! - `verify_is_m_steps`: set `a`'s prompt, then for `m` = 2 to [`VERIFY_ROWS`] the next `m` ids
//!   (its greedy ids, cycled) as a verify of `m` rows (`GpuModel::step_rows`, the body's `Rows`)
//!   and as `m` steps in turn, each rolled back to the prompt's end: the verify's ids and every
//!   row's logits are the steps' bit for bit and the model stands `m` positions on, eager and
//!   replayed (graph mode captures the verify's graphs as a load serving a draft does,
//!   `GpuModel::capture_verify`). Beside the bits, the clause holds the verify graph's forms at
//!   every row count: the controllers and the mixes launch the step's pairs alone (none of the
//!   prompt pass's rows forms, a down for an out and a scores for an out), and prints each
//!   count's geometry (`verify_geometry`, launches and grids an entry).
//! - `verify_then_rollback`: the longest set's prompt (past a windowed visit's window), a verify
//!   of [`VERIFY_ROWS`] ids, a rollback to [`VERIFY_BACK`] rows past the prompt and the rest of
//!   the ids stepped from there: the verify's rows and the steps' are those of a run that only
//!   stepped, bit for bit; the verify wrote its rows' end, so a rollback below the prompt's end
//!   is refused and the model stands where it did.
//! - `loopcd_extrapolates_the_boundary`: LoopCD-Hidden (`BLOOMERY_INK2_LOOPCD`,
//!   `GpuModel::set_loopcd`) on set `a`, the on and the zero runs walking the off run's tokens. At
//!   ω 0 after an on run, the ids and the logits rows are the unset run's bit for bit. At ω 0.2
//!   with the reference pass 1, every visit's stages through the last core one are the off run's
//!   bits, the stream entering the coda's first visit is
//!   `h + ω (h − h_1)` (f64 over the run's own streams) within four f32 units of the last place
//!   a value, and the logits differ from the off run's; a verify of 2 and of 8 rows under the rule
//!   is its steps' bits; the prompt pass's logits move, and its rows through the boundary are the
//!   off pass's bits. It prints, beside, the step's wall off and on: development numbers.
//!   `graph_nodes_are_the_walks` derives the captured step's nodes with the rule off and on.
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
    use bloomery_gpu::model::{Rows, StepMode};
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
    use runtime::stores::PASS_ROWS;

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

    /// The new positions each cycle of `rollback_is_a_fresh_run` steps after replaying its
    /// rolled-back ids.
    const ROLLBACK_ADVANCE: usize = 2;

    /// The ids `rollback_is_a_fresh_run` feeds as a prompt pass before it takes the pass's end
    /// back.
    const PASS_PROMPT: usize = 64;

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
    /// epilogue (one launch with the k and v convs inside), the attention's two launches
    /// (segments, merge), the output gate, the output projection — one launch with the post conv
    /// and the join's add where the projection is Q8_0 (the body's `o_tail`), a block's first
    /// visit then adding nothing but copying the stream to its bank; else the projection, its
    /// post conv and the join (a copy or an add); the feed-forward's mix, norm and router, the
    /// card leg's four launches and its sum (one launch with the post conv and the stream's
    /// add); a visit with messages runs the message mix in its attention
    /// mix's pair and adds two controller launch pairs. A pass edge copies the stream (pass 0) or
    /// the bank, its last row and the prelude back (a bank of no row copies nothing). The step
    /// opens with the embedding's lookup and norm and closes with the head's mix, norm,
    /// projection, softcap, padded-column copy and argmax. LoopCD-Hidden on, with the reference
    /// pass `loopcd` (its index from 1): one copy more at that pass's edge and one launch before
    /// the coda's first visit.
    pub(super) fn derived_nodes(
        hp: &Hparams,
        table: &[(u32, u32)],
        (norm_proj, o_tail): (&[bool], &[bool]),
        loopcd: Option<u32>,
    ) -> (usize, String) {
        const MIX: usize = 2;
        // PIN(2026-11-27): the k and v convs run inside the epilogue's launch, so the visit's first
        // term is 1 where it was 2 + 1 (-2 a visit, -96 a step).
        // PIN(2026-11-27): the feed-forward's post conv and add run in the card sum's launch
        // (-2 a visit), and a Q8_0 output projection runs its post conv and the join's add in its
        // own launch (-2 a visit, -1 on a block's first visit, which still copies the stream):
        // the visit's terms below are the epilogue, the attention's two, the gate, the output
        // projection, the feed-forward's mix, norm and router and the card leg's five.
        const VISIT: usize = 1 + 2 + 1 + 1 + MIX + 1 + 1 + 5;
        const MESSAGES: usize = 2 + 2;
        const EMBED: usize = 2;
        const HEAD: usize = MIX + 5;
        let core = hp.loop_.filter(|lp| lp.passes > 1);
        let block = hp.block;
        let (mut visit, mut edges, mut pushed, mut prelude, mut boundary) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
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
                    edges +=
                        2 + usize::from(pushed > prelude + 1) + usize::from(loopcd == Some(pass));
                    pushed = prelude;
                }
            }
            if loopcd.is_some() && core.is_some_and(|lp| l == lp.core_end) {
                boundary += 1;
            }
            let front = if norm_proj[l] {
                1
            } else {
                6 + if windowed && !msg { 1 } else { 3 } + if windowed { 0 } else { 2 }
            };
            let banks = l.is_multiple_of(block);
            // The output projection's post conv and the join: nothing but the bank's copy in its
            // launch, else two launches.
            let join = if o_tail[l] { usize::from(banks) } else { 2 };
            visit +=
                VISIT + front + join + MIX * usize::from(l != 0) + if msg { MESSAGES } else { 0 };
            if banks {
                pushed += 1;
            }
        }
        let total = EMBED + visit + edges + boundary + HEAD;
        (
            total,
            format!("embed={EMBED} visits={visit} edges={edges} boundary={boundary} head={HEAD}"),
        )
    }

    /// The nodes of the captured step by entry, with the step captured in graph mode after a reset.
    fn captured_entries(m: &mut GpuModel<Body>) -> Result<(usize, String), GateError> {
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
        let entries = hist
            .iter()
            .map(|(k, n)| format!("{k}={n}"))
            .collect::<Vec<_>>()
            .join(" ");
        Ok((nodes, entries))
    }

    /// Clause `graph_nodes_are_the_walks`: the captured step's node count is [`derived_nodes`],
    /// LoopCD-Hidden off and on (on: the reference pass's copy and the boundary's launch alone).
    fn graph_nodes_are_the_walks(m: &mut GpuModel<Body>, hp: &Hparams) -> Result<bool, GateError> {
        if !tier::run_clause("graph_nodes_are_the_walks", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let (table, norm_proj, o_tail) = {
            let body = m.body(NAME)?;
            let per_layer = |f: &dyn Fn(usize) -> bool| -> Vec<bool> {
                (0..body.stored_layers()).map(f).collect()
            };
            (
                body.visit_table(),
                per_layer(&|l| body.norm_proj(l)),
                per_layer(&|l| body.o_tail(l)),
            )
        };
        let flags = (&norm_proj[..], &o_tail[..]);
        let (want, terms) = derived_nodes(hp, &table, flags, None);
        let (nodes, entries) = captured_entries(m)?;
        m.set_loopcd(LOOPCD_OMEGA, LOOPCD_REF)?;
        let (want_on, terms_on) = derived_nodes(hp, &table, flags, Some(LOOPCD_REF));
        let (nodes_on, entries_on) = captured_entries(m)?;
        m.set_loopcd(0.0, LOOPCD_REF)?;
        m.set_mode(StepMode::Eager);
        let pass = nodes == want && nodes_on == want_on && nodes_on == nodes + 2;
        println!(
            "graph_nodes_are_the_walks nodes={nodes} want={want} ({terms}) on_nodes={nodes_on} \
             on_want={want_on} ({terms_on}) {}",
            verdict(pass)
        );
        println!("  entries: {entries}");
        println!("  entries on: {entries_on}");
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

    /// Clause `rollback_is_a_fresh_run`: see the module header. The ids are set `s`'s prompt and
    /// its greedy ids, cycled, up to [`PASS_ROWS`] positions past the wrap of a windowed ring and
    /// the cycles' advance; the fresh run keeps the id and logits row of every position from
    /// [`PASS_ROWS`] below the first rollback on.
    fn rollback_is_a_fresh_run(
        m: &mut GpuModel<Body>,
        hp: &Hparams,
        s: &Set,
    ) -> Result<bool, GateError> {
        if !tier::run_clause("rollback_is_a_fresh_run", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let reach = PASS_ROWS;
        let advance = reach * ROLLBACK_ADVANCE;
        let ring = usize::try_from(runtime::stores::ring_rows(hp.swa as u64, CTX as u64))?;
        let n = ring.saturating_sub(reach).max(s.prompt.len());
        let ids: Vec<u32> = s
            .prompt
            .iter()
            .chain(s.greedy.ids.iter().cycle())
            .copied()
            .take(n + advance)
            .collect();
        if ids.len() != n + advance || ids.len() >= CTX {
            return Err(format!(
                "{NAME}: {} ids for a rollback run of {} positions in stores of {CTX}",
                ids.len(),
                n + advance
            )
            .into());
        }
        let first = n - reach;
        // The fresh run: every id a step, the rows of the positions the cycles replay kept.
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut fresh: Vec<(u32, Vec<f32>)> = Vec::with_capacity(reach + advance);
        for (p, &id) in ids.iter().enumerate() {
            let token = m.step(&[id])?;
            if p >= first {
                fresh.push((token, m.logits()?));
            }
        }
        // The rolled run, to the first cycle's start.
        m.reset()?;
        for &id in &ids[..n] {
            m.step(&[id])?;
        }
        let mut held = n;
        let mut miss: Option<String> = None;
        let mut replays = 0usize;
        // One step of the rolled run at position `p`, checked against the fresh run's row.
        let mut again = |m: &mut GpuModel<Body>, p: usize| -> Result<bool, GateError> {
            let token = m.step(&[ids[p]])?;
            let (want, row) = &fresh[p - first];
            replays += 1;
            Ok(token == *want && bits_equal(&m.logits()?, row))
        };
        'cycles: for k in 1..=reach {
            let to = held - k;
            if let Err(e) = m.rollback(u32::try_from(to)?) {
                miss = Some(format!("k={k}: a rollback to {to} refused: {e}"));
                break;
            }
            if m.pos() as usize != to {
                miss = Some(format!(
                    "k={k}: the model stands at {} after a rollback to {to}",
                    m.pos()
                ));
                break;
            }
            for p in to..held + ROLLBACK_ADVANCE {
                if !again(m, p)? {
                    miss = Some(format!("k={k}: position {p} differs from the fresh run"));
                    break 'cycles;
                }
            }
            held += ROLLBACK_ADVANCE;
        }
        // Past the reach: refused by name, the model where it stood.
        let at = u32::try_from(held)?;
        let (reach32, below) = (u32::try_from(reach)?, u32::try_from(reach + 1)?);
        let past = m.rollback(at - below);
        let past_text = past.as_ref().err().map(ToString::to_string);
        let past_named = past_text
            .as_deref()
            .is_some_and(|t| t.contains("inkling2::Body::rollback") && t.contains("conv ring"));
        let past_stands = m.pos() == at;
        // The depth is the written end's: 3 back, then 6 more is 9 below the end (refused, the
        // model standing at `at - 3`), then 5 more is 8 below it and replays the fresh run.
        let three = m.rollback(at - 3).is_ok() && m.pos() == at - 3;
        let deeper_refused = m.rollback(at - 9).is_err() && m.pos() == at - 3;
        let eight = m.rollback(at - reach32).is_ok() && m.pos() == at - reach32;
        let mut end_run = three && eight;
        if miss.is_none() && end_run {
            for p in held - reach..held {
                end_run &= again(m, p)?;
            }
        }
        m.set_mode(StepMode::Eager);
        // A pass writes the end as steps do: the rollback of the reach is taken, one more refused.
        m.reset()?;
        m.prompt_pass(&ids[..PASS_PROMPT])?;
        let end = u32::try_from(PASS_PROMPT)?;
        let pass_refused = m.rollback(end - below).is_err() && m.pos() == end;
        let pass_taken = m.rollback(end - reach32).is_ok() && m.pos() == end - reach32;
        let pass_end = pass_refused && pass_taken;
        let reaches = m.body(NAME)?.rollback_reach() as usize == reach;
        let pass = miss.is_none()
            && past_named
            && past_stands
            && deeper_refused
            && end_run
            && pass_end
            && reaches;
        println!(
            "rollback_is_a_fresh_run set={} positions={} first_cycle_at={n} reach={reach} \
             (body {}) rows_checked={replays} first_miss={miss:?} \
             past_reach_refused={past_text:?} named={past_named} stands={past_stands} \
             nine_below_the_end_refused={deeper_refused} \
             eight_below_the_end_replays={end_run} after_a_pass_end_held={pass_end} {}",
            s.label,
            ids.len(),
            m.body(NAME)?.rollback_reach(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The rows a verify walks at most, a draft block: the body's [`Rows::MAX_ROWS`].
    const VERIFY_ROWS: usize = 8;
    const _: () = assert!(<Body as Rows>::MAX_ROWS == VERIFY_ROWS);
    /// The rows of a verified block `verify_then_rollback` keeps.
    const VERIFY_BACK: usize = 3;

    /// The prompt pass's rows forms of the controllers and the mix: one block a module at a
    /// verify's rows. A verify launches none of them ([`verify_geometry`]): its controllers and
    /// its mixes run the step's own pair at the rows (`arch/inkling2/rows.rs`).
    const PASS_ROWS_FORMS: [&str; 4] = [
        "loop_control_down_rows",
        "loop_control_out_rows",
        "loop_control_out3_rows",
        "attn_res_mix_normed",
    ];

    /// The verify graph of `rows` rows' launch geometry: the controllers' and the mixes' entries
    /// (the step's and the pass's rows forms) and the input norms' and projections' rows entry
    /// with their launches and the grids they launch at,
    /// and whether the graph holds the step's pairs alone: none of [`PASS_ROWS_FORMS`], every
    /// pair's two launches there (a down for an out, a scores for an out), and a mix at least.
    fn verify_geometry(m: &GpuModel<Body>, rows: usize) -> Result<(Vec<String>, bool), GateError> {
        let nodes = match rows {
            2 => m.rows_graph_nodes::<2>()?,
            3 => m.rows_graph_nodes::<3>()?,
            4 => m.rows_graph_nodes::<4>()?,
            5 => m.rows_graph_nodes::<5>()?,
            6 => m.rows_graph_nodes::<6>()?,
            7 => m.rows_graph_nodes::<7>()?,
            8 => m.rows_graph_nodes::<8>()?,
            _ => {
                return Err(format!("{NAME}: a verify of {rows} rows (2 to {VERIFY_ROWS})").into());
            }
        };
        const ENTRIES: [&str; 10] = [
            "loop_control_down",
            "loop_control_out",
            "loop_control_out3",
            "loop_control_down_rows",
            "loop_control_out_rows",
            "loop_control_out3_rows",
            "attn_res_step_scores",
            "attn_res_step_out",
            "attn_res_mix_normed",
            "norm_proj_q8_f32_cols",
        ];
        let mut hist =
            std::collections::BTreeMap::<&str, (usize, std::collections::BTreeSet<u32>)>::new();
        for name in ENTRIES {
            hist.insert(name, (0, std::collections::BTreeSet::new()));
        }
        let mut pass_forms = 0usize;
        for n in &nodes {
            let Some(k) = &n.kernel else {
                continue;
            };
            if PASS_ROWS_FORMS.contains(&k.name.as_str()) {
                pass_forms += 1;
            }
            if let Some((count, grids)) = hist.get_mut(k.name.as_str()) {
                *count += 1;
                grids.insert(k.grid[0]);
            }
        }
        let launches = |name: &str| hist.get(name).map_or(0, |(count, _)| *count);
        let step_pairs = launches("loop_control_down")
            == launches("loop_control_out") + launches("loop_control_out3")
            && launches("attn_res_step_scores") == launches("attn_res_step_out")
            && launches("attn_res_step_scores") > 0;
        let entries = hist
            .iter()
            .map(|(name, (count, grids))| {
                let grids: Vec<String> = grids.iter().map(u32::to_string).collect();
                format!("{name}={count}x{{{}}}", grids.join(","))
            })
            .collect();
        Ok((entries, pass_forms == 0 && step_pairs))
    }

    /// What a verify gave: its ids, every row's logits, and its wall in ms (the enqueue or the
    /// replay, and the ids' read).
    struct Verified {
        ids: Vec<u32>,
        rows: Vec<Vec<f32>>,
        ms: f64,
    }

    /// A verify of the first `M` of `ids` from where the model stands.
    fn verify_of<const M: usize>(
        m: &mut GpuModel<Body>,
        ids: &[u32],
    ) -> Result<Verified, GateError> {
        let tokens: [u32; M] = std::array::from_fn(|r| ids[r]);
        let t = std::time::Instant::now();
        let out = m.step_rows::<M>(tokens)?;
        let ms = t.elapsed().as_secs_f64() * 1e3;
        Ok(Verified {
            ids: out.to_vec(),
            rows: m.rows_logits::<M>()?.into_iter().collect(),
            ms,
        })
    }

    /// [`verify_of`] at `rows` rows, 2 to [`VERIFY_ROWS`].
    fn verify(m: &mut GpuModel<Body>, ids: &[u32], rows: usize) -> Result<Verified, GateError> {
        if ids.len() < rows {
            return Err(format!("{NAME}: a verify of {rows} rows over {} ids", ids.len()).into());
        }
        match rows {
            2 => verify_of::<2>(m, ids),
            3 => verify_of::<3>(m, ids),
            4 => verify_of::<4>(m, ids),
            5 => verify_of::<5>(m, ids),
            6 => verify_of::<6>(m, ids),
            7 => verify_of::<7>(m, ids),
            8 => verify_of::<8>(m, ids),
            _ => Err(format!("{NAME}: a verify of {rows} rows (2 to {VERIFY_ROWS})").into()),
        }
    }

    /// Whether a verify's ids and logits rows are the steps' `(id, logits)` bit for bit.
    fn rows_are_steps(steps: &[(u32, Vec<f32>)], v: &Verified) -> bool {
        steps.len() == v.ids.len()
            && steps.len() == v.rows.len()
            && steps
                .iter()
                .zip(v.ids.iter().zip(&v.rows))
                .all(|((id, row), (vid, vrow))| id == vid && bits_equal(row, vrow))
    }

    /// Clause `verify_is_m_steps`: see the module header. Prints, beside, each verify's wall at its
    /// second replay (the first uploads the graph) and its steps' walls, at the prompt's depth:
    /// development numbers, not timed runs; and each row count's launch geometry
    /// ([`verify_geometry`]), whose step-pair forms the clause holds beside the bits.
    fn verify_is_m_steps(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause("verify_is_m_steps", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let at = u32::try_from(s.prompt.len())?;
        let ids: Vec<u32> = s
            .greedy
            .ids
            .iter()
            .cycle()
            .copied()
            .take(VERIFY_ROWS)
            .collect();
        let mut miss: Option<String> = None;
        let mut nodes = Vec::new();
        let mut walls = Vec::new();
        let mut geometry = Vec::new();
        let mut step_forms = true;
        for mode in [StepMode::Eager, StepMode::Graph] {
            m.set_mode(mode);
            m.reset()?;
            m.step(&s.prompt)?;
            if mode == StepMode::Graph {
                nodes = m.capture_verify()?.to_vec();
            }
            for rows in 2..=VERIFY_ROWS {
                let mut steps = Vec::with_capacity(rows);
                let mut step_ms = 0.0;
                for &id in &ids[..rows] {
                    let t = std::time::Instant::now();
                    let token = m.step(&[id])?;
                    step_ms += t.elapsed().as_secs_f64() * 1e3;
                    steps.push((token, m.logits()?));
                }
                m.rollback(at)?;
                let v = verify(m, &ids, rows)?;
                let on = at + u32::try_from(rows)?;
                let same = rows_are_steps(&steps, &v);
                if !(same && m.pos() == on) && miss.is_none() {
                    miss = Some(format!(
                        "{mode:?} m={rows}: ids {:?} against the steps' {:?}, rows bit for bit \
                         {same}, standing at {} (want {on})",
                        v.ids,
                        steps.iter().map(|s| s.0).collect::<Vec<_>>(),
                        m.pos()
                    ));
                }
                m.rollback(at)?;
                if mode == StepMode::Graph {
                    let again = verify(m, &ids, rows)?;
                    walls.push(format!("m{rows}={:.2}/{step_ms:.2}", again.ms));
                    m.rollback(at)?;
                    let (entries, pairs) = verify_geometry(m, rows)?;
                    step_forms &= pairs;
                    geometry.push(format!("m{rows}: {}", entries.join(" ")));
                }
            }
        }
        m.set_mode(StepMode::Eager);
        let captured = nodes.len() == VERIFY_ROWS - 1 && nodes.iter().all(|&n| n > 0);
        let pass = miss.is_none() && captured && step_forms;
        println!(
            "verify_is_m_steps set={} at={at} rows=2..={VERIFY_ROWS} modes=eager,graph \
             verify_graph_nodes={nodes:?} first_miss={miss:?} step_pairs={step_forms} {}",
            s.label,
            verdict(pass)
        );
        println!(
            "verify_walls_ms (development: the verify's second graph replay / its m graph \
             steps, depth {at}) {}",
            walls.join(" ")
        );
        println!(
            "verify_geometry (launches x {{grids}} of the controllers', the mixes' and the \
             projections' entries) {}",
            geometry.join(" | ")
        );
        Ok(pass)
    }

    /// Clause `verify_then_rollback`: see the module header.
    fn verify_then_rollback(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        if !tier::run_clause("verify_then_rollback", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let at = u32::try_from(s.prompt.len())?;
        let ids: Vec<u32> = s
            .greedy
            .ids
            .iter()
            .cycle()
            .copied()
            .take(VERIFY_ROWS)
            .collect();
        m.set_mode(StepMode::Graph);
        // The fresh run: the prompt, then every id a step.
        m.reset()?;
        m.step(&s.prompt)?;
        let mut fresh = Vec::with_capacity(VERIFY_ROWS);
        for &id in &ids {
            let token = m.step(&[id])?;
            fresh.push((token, m.logits()?));
        }
        // The verified run: the prompt, the block's verify, back to VERIFY_BACK rows past the
        // prompt, the rest stepped.
        m.reset()?;
        m.step(&s.prompt)?;
        let verified = rows_are_steps(&fresh, &verify(m, &ids, VERIFY_ROWS)?);
        let end = at + u32::try_from(VERIFY_ROWS)?;
        let past_refused = m.rollback(at - 1).is_err() && m.pos() == end;
        let back = at + u32::try_from(VERIFY_BACK)?;
        let rolled = m.rollback(back).is_ok() && m.pos() == back;
        let mut replayed = Vec::new();
        if rolled {
            for (j, &id) in ids.iter().enumerate().skip(VERIFY_BACK) {
                let token = m.step(&[id])?;
                let (want, row) = &fresh[j];
                replayed.push(token == *want && bits_equal(&m.logits()?, row));
            }
        }
        m.set_mode(StepMode::Eager);
        let replays = replayed.len() == VERIFY_ROWS - VERIFY_BACK && replayed.iter().all(|&b| b);
        let pass = verified && past_refused && rolled && replays;
        println!(
            "verify_then_rollback set={} at={at} rows={VERIFY_ROWS} verify_is_the_steps={verified} \
             below_the_prompt_refused={past_refused} back_to={back} taken={rolled} \
             steps_are_the_fresh_run={replayed:?} {}",
            s.label,
            verdict(pass)
        );
        Ok(pass)
    }

    /// LoopCD-Hidden's strength and reference pass the two clauses run the rule at.
    const LOOPCD_OMEGA: f32 = 0.2;
    const LOOPCD_REF: u32 = 1;
    /// The graph steps after the prompt one look times, for the step's wall.
    const LOOPCD_LOOK_STEPS: usize = 32;

    /// One run of set `s` in graph mode under one LoopCD-Hidden setting: the prompt's last logits
    /// row, the argmaxes and logits rows of the next [`GRAPH_STEPS`] positions, every visit's
    /// tapped stages, the stream entering the coda's first visit (the draft taps'), and the mean
    /// wall of [`LOOPCD_LOOK_STEPS`] graph steps after them, each with its argmax read back.
    struct LoopcdRun {
        logits: Vec<f32>,
        ids: Vec<u32>,
        rows: Vec<Vec<f32>>,
        taps: Vec<f32>,
        entering: Vec<f32>,
        ms: f64,
    }

    /// [`LoopcdRun`] of set `s` from a reset: the prompt, [`GRAPH_STEPS`] greedy steps with their
    /// rows, the taps, then [`LOOPCD_LOOK_STEPS`] steps timed. `forced` feeds each of those
    /// positions its id there instead of the step's own argmax, so a run under another setting
    /// walks the same tokens; it names one id a position, the prompt's argmax first.
    fn loopcd_run(
        m: &mut GpuModel<Body>,
        s: &Set,
        forced: Option<&[u32]>,
    ) -> Result<LoopcdRun, GateError> {
        let steps = GRAPH_STEPS + LOOPCD_LOOK_STEPS;
        if forced.is_some_and(|f| f.len() < steps + 1) {
            return Err(format!(
                "{NAME}: a forced run of {steps} steps over {} ids",
                forced.map_or(0, <[u32]>::len)
            )
            .into());
        }
        m.reset()?;
        let mut ids = vec![m.step(&s.prompt)?];
        let logits = m.logits()?;
        let mut rows = Vec::new();
        let next = |ids: &[u32]| forced.map_or(ids[ids.len() - 1], |f| f[ids.len() - 1]);
        for _ in 0..GRAPH_STEPS {
            ids.push(m.step(&[next(&ids)])?);
            rows.push(m.logits()?);
        }
        let taps = {
            let (gpu, _, body) = m.body_parts(NAME)?;
            body.taps(gpu.stream())?
        };
        let entering = m.draft_taps()?;
        let t = std::time::Instant::now();
        for _ in 0..LOOPCD_LOOK_STEPS {
            ids.push(m.step(&[next(&ids)])?);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / LOOPCD_LOOK_STEPS as f64;
        Ok(LoopcdRun {
            logits,
            ids,
            rows,
            taps,
            entering,
            ms,
        })
    }

    /// The prompt as a pass: its last logits row and every visit's stages at the last row.
    fn loopcd_pass(m: &mut GpuModel<Body>, s: &Set) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        m.reset()?;
        m.prompt_pass(&s.prompt)?;
        let logits = m.logits()?;
        let (gpu, _, body) = m.body_parts(NAME)?;
        Ok((logits, body.taps(gpu.stream())?))
    }

    /// Visit `v`'s output among its tapped stages (the last of [`TAP_STAGES`]): the last `n`
    /// values of its block of `row` values.
    fn visit_output(taps: &[f32], v: usize, (row, n): (usize, usize)) -> &[f32] {
        &taps[v * row + row - n..v * row + row]
    }

    /// The first of the first `visits` visits' stages at which two tapped runs' values differ,
    /// `None` when they are the same bits.
    fn first_tap_gap(
        a: &[f32],
        b: &[f32],
        widths: &[usize; TAP_STAGES.len()],
        visits: usize,
    ) -> Option<String> {
        let row: usize = widths.iter().sum();
        (0..visits).find_map(|v| {
            let mut at = v * row;
            widths.iter().zip(TAP_STAGES).find_map(|(&w, stage)| {
                let differs = !bits_equal(&a[at..at + w], &b[at..at + w]);
                at += w;
                differs.then(|| format!("visit {v} stage {stage}"))
            })
        })
    }

    /// Clause `loopcd_extrapolates_the_boundary`: see the module header. Mutants: the boundary
    /// step with its strength's sign flipped, the reference saved one pass later, the launch left
    /// out: each is red at the stream entering the coda, the line naming the lever and the value.
    fn loopcd_extrapolates_the_boundary(
        m: &mut GpuModel<Body>,
        hp: &Hparams,
        s: &Set,
    ) -> Result<bool, GateError> {
        if !tier::run_clause("loopcd_extrapolates_the_boundary", Tag::SelfConsistency)? {
            return Ok(true);
        }
        let core = hp
            .loop_
            .filter(|lp| lp.passes > 1)
            .ok_or("a looped trunk: LoopCD-Hidden has no boundary to hold")?;
        let (n, widths) = (hp.n_embd, m.body(NAME)?.tap_widths());
        let row: usize = widths.iter().sum();
        let table = m.body(NAME)?.visit_table();
        let coda = table
            .iter()
            .position(|&(l, _)| l as usize == core.core_end)
            .ok_or("no visit past the core")?;
        let last = coda
            .checked_sub(1)
            .ok_or("a coda with no core visit before it")?;
        // The reference pass's last core visit: a coda visit carries pass 0 too, so the core's
        // layers bound the search.
        let h_b = table
            .iter()
            .rposition(|&(l, pass)| {
                (core.core_start..core.core_end).contains(&(l as usize)) && pass == LOOPCD_REF - 1
            })
            .ok_or("no visit of the reference pass")?;
        let (omega, reference) = (LOOPCD_OMEGA, LOOPCD_REF);
        m.set_mode(StepMode::Graph);
        m.set_loopcd(0.0, reference)?;
        {
            let (gpu, _, body) = m.body_parts(NAME)?;
            body.arm_taps(gpu)?;
        }
        m.set_draft_taps(&[coda])?;
        let off = loopcd_run(m, s, None)?;
        m.set_loopcd(omega, reference)?;
        let on = loopcd_run(m, s, Some(&off.ids))?;
        m.set_loopcd(0.0, reference)?;
        let zero = loopcd_run(m, s, Some(&off.ids))?;
        // The rule touches the boundary alone: every stage of every visit through the last core
        // one is the off run's bits.
        let untouched = bits_equal(&on.taps[..(last + 1) * row], &off.taps[..(last + 1) * row]);
        let gap = first_tap_gap(&on.taps, &off.taps, &widths, last + 1);
        let zero_gap = first_tap_gap(&zero.taps, &off.taps, &widths, last + 1);
        // The draft tap reads the stream entering the visit: off, the last core visit's output.
        let read_ok = bits_equal(&off.entering, visit_output(&off.taps, last, (row, n)));
        // [derived] the stream entering the coda is h + ω (h − h_b) in f64 over the run's own h
        // and h_b, each within four f32 units of the last place (two roundings).
        let (h_last, h_ref) = (
            visit_output(&on.taps, last, (row, n)),
            visit_output(&on.taps, h_b, (row, n)),
        );
        let omega64 = f64::from(omega);
        let mut worst = (0.0f64, 0usize, 0.0f64, 0.0f64);
        let mut miss: Option<String> = None;
        for i in 0..n {
            let (h, h_b, got) = (
                f64::from(h_last[i]),
                f64::from(h_ref[i]),
                f64::from(on.entering[i]),
            );
            if !got.is_finite() {
                miss = Some(format!(
                    "BLOOMERY_INK2_LOOPCD={omega}: value {i} of the stream entering the coda is \
                     not finite"
                ));
                break;
            }
            let want = h + omega64 * (h - h_b);
            let err = (got - want).abs();
            let tol = 4.0 * f64::from(f32::EPSILON) * (want.abs() + omega64 * (h - h_b).abs());
            let ratio = if tol > 0.0 {
                err / tol
            } else if err > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            if ratio > worst.0 {
                worst = (ratio, i, err, tol);
            }
            if ratio > 1.0 && miss.is_none() {
                miss = Some(format!(
                    "BLOOMERY_INK2_LOOPCD={omega} with BLOOMERY_INK2_LOOPCD_REF={reference}: value \
                     {i} of the stream entering the coda is {err:.3e} from h + the strength times \
                     (h − h_b), past {tol:.3e} (four units of the last place)"
                ));
            }
        }
        let rule_ok = miss.is_none();
        let n_vocab = s.logits.len();
        let moved = !bits_equal(&on.logits[..n_vocab], &off.logits[..n_vocab]);
        let on_finite = on.logits.iter().all(|v| v.is_finite());
        let zero_same = zero.ids == off.ids
            && bits_equal(&zero.logits, &off.logits)
            && zero.rows.len() == off.rows.len()
            && zero
                .rows
                .iter()
                .zip(&off.rows)
                .all(|(a, b)| bits_equal(a, b));
        // A verify runs with no taps armed: both sets come off; the visit taps return for the
        // pass.
        m.body_parts(NAME)?.2.disarm_taps();
        m.set_draft_taps(&[])?;
        // A verify of 2 and of 8 rows under the rule is its steps' bits.
        m.set_loopcd(omega, reference)?;
        let at = u32::try_from(s.prompt.len())?;
        let ids: Vec<u32> = s
            .greedy
            .ids
            .iter()
            .cycle()
            .copied()
            .take(VERIFY_ROWS)
            .collect();
        let mut verify_same = true;
        let mut verify_miss: Option<String> = None;
        for rows in [2usize, VERIFY_ROWS] {
            m.reset()?;
            m.step(&s.prompt)?;
            if rows == 2 {
                m.capture_verify()?;
            }
            let mut steps = Vec::with_capacity(rows);
            for &id in &ids[..rows] {
                steps.push((m.step(&[id])?, m.logits()?));
            }
            m.rollback(at)?;
            let v = verify(m, &ids, rows)?;
            let same = rows_are_steps(&steps, &v) && m.pos() == at + u32::try_from(rows)?;
            verify_same &= same;
            if !same && verify_miss.is_none() {
                verify_miss = Some(format!(
                    "m={rows}: ids {:?} against the steps' {:?}, rows bit for bit {}",
                    v.ids,
                    steps.iter().map(|s| s.0).collect::<Vec<_>>(),
                    rows_are_steps(&steps, &v)
                ));
            }
        }
        // The prompt pass under the rule: its rows through the last core visit are the off
        // pass's bits, and its logits move.
        {
            let (gpu, _, body) = m.body_parts(NAME)?;
            body.arm_taps(gpu)?;
        }
        let (pass_on, pass_taps_on) = loopcd_pass(m, s)?;
        m.set_loopcd(0.0, reference)?;
        let (pass_off, pass_taps_off) = loopcd_pass(m, s)?;
        let pass_moved = !bits_equal(&pass_on[..n_vocab], &pass_off[..n_vocab]);
        let pass_untouched = bits_equal(
            &pass_taps_on[..(last + 1) * row],
            &pass_taps_off[..(last + 1) * row],
        );
        m.set_mode(StepMode::Eager);
        m.set_draft_taps(&[])?;
        m.body_parts(NAME)?.2.disarm_taps();
        let pass = untouched
            && read_ok
            && rule_ok
            && moved
            && on_finite
            && zero_same
            && verify_same
            && pass_moved
            && pass_untouched;
        println!(
            "loopcd_extrapolates_the_boundary set={} omega={omega} ref={reference} coda_visit={coda} \
             h_b_visit={h_b} before_boundary_same_bits={untouched} entering_is_l_out={read_ok} \
             boundary_worst_share={:.3} at value {} (err {:.3e}, 4 ulp {:.3e}) logits_moved={moved} \
             logits_finite={on_finite} logits_rel_from_off={:.3e} zero_is_unset={zero_same} \
             verify_is_m_steps={verify_same} pass_moved={pass_moved} \
             pass_before_boundary_same_bits={pass_untouched} first_gap={gap:?} \
             zero_first_gap={zero_gap:?} first_miss={miss:?} verify_miss={verify_miss:?} {}",
            s.label,
            worst.0,
            worst.1,
            worst.2,
            worst.3,
            rel(&on.logits[..n_vocab], &off.logits[..n_vocab]),
            verdict(pass)
        );
        println!(
            "loopcd_walls_ms (development: the mean of {LOOPCD_LOOK_STEPS} graph steps after {GRAPH_STEPS}, \
             the argmax read back, depth {}) off={:.3} on={:.3} apart={:+.2} %",
            s.prompt.len() + GRAPH_STEPS,
            off.ms,
            on.ms,
            (on.ms / off.ms - 1.0) * 100.0
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
        let longest = sets
            .iter()
            .max_by_key(|s| s.prompt.len())
            .ok_or("no prompt set")?;
        ok &= rollback_is_a_fresh_run(&mut m, &hp, longest)?;
        ok &= verify_is_m_steps(&mut m, &sets[0])?;
        ok &= verify_then_rollback(&mut m, longest)?;
        ok &= loopcd_extrapolates_the_boundary(&mut m, &hp, &sets[0])?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} tier {tier:?}: the visits of the loop; on every set the chain's \
             visits and logits within their bands (the fixture's with the vendor's picks, the \
             Mini's free), the teacher-forced argmax disagreements within the pin; the message \
             path live; the captured step's nodes the walk's, LoopCD-Hidden off and on; the graph \
             the eager chain; a rollback within the reach a fresh run; a verify of 2 to 8 rows \
             the steps, and a rollback after one a fresh run; LoopCD-Hidden's stream entering the \
             coda within ulps of the rule, zero the unset bits, live in a step, a verify and a \
             prompt pass"
        );
        Ok(())
    }
}
