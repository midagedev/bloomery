//! The Kimi K3 end-to-end gate: the whole decode step on one card (the embedding, the 93-layer
//! chain's delta and gated latent layers, the dense and routed `situ` blocks, the AttnRes mixes
//! over the banked block inputs, the output head and the argmax) over the fixture file (4 layers:
//! a dense delta layer, a routed delta layer, a routed latent layer, a routed delta layer; hidden
//! 512, an attn_res block of 2), one prompt id a step, against mainline llama.cpp's CPU node dumps
//! of that fixture (refset family `fx-lcpp-kimi-k3`, `BLOOMERY_TIER=fixture just dump-ref-kimi-k3`).
//! The model opens through `GpuModel<Body>::open`, the whole-card load `generate_kimi_k3` runs.
//!
//! The fixture tier only: the real file (594 GB) does not fit one card and the body has no placed
//! load (two stage cards, the host and the NVMe tier), so the real tier ends by name before
//! anything loads.
//!
//! For each prompt set of [`SETS`] (an English paragraph, a Korean one, a Rust function head and a
//! five-id question), against the vendor's node dumps:
//!
//! - `layer_outputs_are_vendors`: the free chain's layers 0 to 2, each layer's output at the
//!   prompt's last position (the stream after the layer's feed-forward block, `l_out-L`) within
//!   [`LAYER_REL_BAND`] of the vendor's by relative L2. Every stage of every layer (the vendor's
//!   `attn_norm`, `kda_*`, `prefix_sum_attn`, `ffn_norm`, `ffn_out`, `l_out` nodes) prints beside
//!   it, so a miss names the first stage that left the band.
//! - `forced_mixers_are_vendors`: each layer's mixer alone over the vendor's own normed rows of
//!   every position (from a fresh store; the rows divided by the layer's norm gain so the mixer's
//!   own norm gives them back), its output at the last position within [`FORCED_REL_BAND`] of the
//!   vendor's `kda_out`, or `prefix_sum_attn` on a banking layer.
//! - `forced_ffn_is_vendors`: each layer's feed-forward block alone over the vendor's own normed
//!   row at the last position: its update within [`FORCED_REL_BAND`] of `ffn_out`, and a routed
//!   block's picks the vendor's `ffn_moe_topk`.
//! - `forced_top1_is_vendors`: the vendor's own greedy ids fed back one a step (teacher-forced),
//!   our argmax at each of the greedy table's positions against its id: the disagreements at a
//!   vendor margin of [`MARGIN_FLOOR`] or more are at most the pin the Gaussian error model gives
//!   ([`forced_pin`], the common harness's form).
//! - `logits_are_vendors`: the last position's whole logits row within [`LOGITS_REL_BAND`]
//!   relative L2 of the vendor's `result_output`.
//!
//! and the clauses that need no vendor:
//!
//! - `opens_the_fixture`: the model opens and carries the file's four layers.
//! - `steps_are_deterministic`: the same prompt twice from a reset gives the logits row bit for bit,
//!   and a reset after another set's prompt gives it again (the state book, the delta layers'
//!   stores and the latent caches, is back where the load left it).
//! - `graph_is_eager`: the steps in graph mode (the chain captured once and replayed) give the
//!   eager chain's ids and last logits row bit for bit.
//! - `generate_matches_the_gate`: `generate_kimi_k3` (built beside this gate, its own process and
//!   load, graph mode) over set `d`'s prompt ids prints the ids this gate's own steps give from
//!   the same ids (the prompt one id a step, then each id fed back), 8 of them, and the line
//!   prints where they and the vendor's greedy table part.
//! - `mix_matches_host`: the AttnRes mix kernel against a host f64 mix of its definition at the
//!   fixture's width (1, 2 and 15 banks) and the real file's (7168, 8 banks), within [`MIX_BAND`].
//! - `mix_rows_are_steps`: a pass of 8 rows is 8 one-row mixes bit for bit. This is the mix kernel
//!   alone; the body's rows pass is `gate_kimi_k3_pp`'s.
//! - `mix_is_the_mix`: the stream alone in place of the mix (the mix dropped) lies past [`SEP`]
//!   bands of the kernel's output.
//! - `mix_refused`: a row or width of 0, more than 16 vectors, a stride under a pass's rows, an
//!   epsilon that is not positive and finite and a short output are refused by name.
//! - `mix_faults`: a non-finite score raises the card's fault word and the row is NaN.
//!
//! Before the card loads, a set absent from the box, dumped from another file or build, or
//! unfinished ends the gate by name. A step that raises the fault word ends the gate by name.
//!
//! What the free chain can hold. The two engines read the same weights and differ by how they
//! round: ours quantizes each projection's input row to q8_1 in blocks of 128 values, the vendor's
//! CPU path to q8_K in blocks of 256. A block's step is `amax / 127`, its rounding error uniform,
//! so an input element's relative error is `r / (127 √12)` with `r = amax / rms` (2.8 for a block of
//! 128 values, 3.0 for 256): 0.64 % and 0.68 %, 0.93 % between the two. A projection's output
//! inherits it and a branch (a mixer or a feed-forward block) chains two to three such sites
//! through a nonlinearity: 1.5 % predicted, 1.4 % to 2.8 % measured by the two forced clauses on
//! every layer and set, with the router's picks the vendor's on all twelve routed blocks. That is
//! rounding, not a defect. The free chain compounds it: a layer's input error enters each branch
//! with the branch's gain (1.4 to 2 at layers 1 and 2, measured) and adds the branch's own, so
//! layer 0 is `√(1.9² + 2.2²)` = 2.9 % (measured 2.6 % to 3.0 %), layer 1 4.5 % to 5.1 %, layer 2
//! 6.9 % to 13.5 %. Layer 3 is the final delta layer: on set `a`'s last token its gain is 10 (5 %
//! in, 54 % out, while the layer alone on the vendor's rows is 1.9 % off), so its output spans 9.6 %
//! to 73 % across the sets and the head reads it within a tenth (the logits row 9.7 % to 73 %).
//! The vendor's margins (0.04 to 0.37 on the greedy tables' first positions) sit under the logit
//! noise that follows (σ̂, the rms distance of our logit from the vendor's over its top 40 ids at
//! the prompt's end: 0.10 to 1.42), so the free chain's argmax is the vendor's on set `d` and not
//! on the other three: per-position top-1 identity is not a claim a 4-layer cut of this model with
//! an 8-expert router can hold at q8 activations, and the forced clauses are what hold the blocks.
//!
//! Bands, each pinned at about 1.5 times its measured maximum on the A6000 at this commit:
//!
//! - `FORCED_REL_BAND` 4e-2: the forced blocks' measured maximum is 2.8e-2.
//! - `LAYER_REL_BAND` 5e-2, 8e-2, 2e-1 for layers 0 to 2: measured maxima 3.0e-2, 5.1e-2, 1.35e-1.
//! - `LOGITS_REL_BAND` 1.0: the row is closer to the vendor's than the zero row is. It holds
//!   nothing finer than NaN or garbage; the finer holds are the layers' and the forced blocks'.
//! - `SIGMA_LOGIT` 1.0: between the three sets' σ̂ (0.10 to 0.25) and set `a`'s (1.42).
//! - `MIX_BAND`: the mix's terms. A score is a sum of 512 to 7168 f32 products, error `6e-8 √n`
//!   times the products' size, 8e-7 at the fixture's width for products of unit size; `rstd` and
//!   the softmax's `exp` at 2 ulps, 1.2e-7 each; the weighted sum of at most 16 vectors, 1e-7. The
//!   scores are scaled to order 1, so the probabilities move by about their own relative error and
//!   the output's relative L2 is 1e-6 predicted, 3.5e-7 measured; the band is ten times the
//!   prediction.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_kimi_k3_e2e", gate::run())
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
    use crate::e2e::{rel, tap, top_ids};
    use bloomery_gpu::arch::kimi_k3::{Body, TAP_STAGES};
    use bloomery_gpu::attn_res::{AttnResArgs, AttnResKernels, MAX_VECS};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::prompts::{GreedyRow, MARGIN_FLOOR, compare_forced, std_normal_cdf};
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, GateRng, RefManifest, beside, bits_equal, checks_failed, open_split,
        ref_tensor_logical_in, topk_ids_logical_within, verdict,
    };
    use cuda_core::DeviceBuffer;
    use model::arch::Arch;
    use refset::arch::kimi_k3::fixture::{self, A, B, C, D};
    use refset::arch::prompt_tables::{self, Greedy, Top};

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_kimi_k3_e2e";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-gpu-kimik3-e2e";

    /// The sets the step path is held to, in the order they run.
    const SETS: &[&str] = &[A, B, C, D];

    /// Rows of each latent layer's cache: the longest prompt and its continuation fit.
    const CTX: usize = 128;

    /// The experts of the fixture's routed layers.
    const N_EXPERT: u32 = 8;

    /// The layers of the fixture file.
    const LAYERS: usize = 4;

    /// The free chain's layers 0 to 2: each layer's output at the last position, relative L2
    /// from the vendor's `l_out-L`, the ceiling (module doc). Layer 3 is not held: it is the
    /// head's input within a tenth of the logits' own distance.
    /// PIN(2026-11-06): 1.5 to 1.7 times the measured maxima 3.0e-2, 5.1e-2, 1.35e-1.
    const LAYER_REL_BAND: [f64; 3] = [5.0e-2, 8.0e-2, 2.0e-1];

    /// One implementation's logit deviation from the other's, in logit units: the σ of the
    /// teacher-forced argmax pin ([`forced_pin`]).
    /// PIN(2026-11-06): between the measured σ̂ of three sets (0.10 to 0.25) and of the fourth
    /// (1.42).
    const SIGMA_LOGIT: f64 = 1.0;

    /// A block run alone on the vendor's own rows, relative L2 from the vendor's node: the
    /// ceiling (module doc).
    /// PIN(2026-11-06): 1.4 times the measured maximum 2.8e-2.
    const FORCED_REL_BAND: f64 = 4.0e-2;

    /// The last-position ids whose overlap is printed.
    const TOP_K: usize = 5;

    /// The vendor clauses a run declares: three a set.
    const FX_ORACLE: usize = 5 * SETS.len();

    /// The last-position logits row's relative distance from the vendor's (module doc).
    /// PIN(2026-11-06): the row is closer to the vendor's than the zero row is; measured 0.097 to
    /// 0.73.
    const LOGITS_REL_BAND: f64 = 1.0;

    /// The mix kernel against the host f64 mix, relative L2 of a row (module doc).
    /// PIN(2026-11-06): 1.0e-5, ten times the derived 1e-6.
    const MIX_BAND: f64 = 1.0e-5;

    /// The bands a mutated mix must lie past.
    const SEP: f64 = 10.0;

    /// `layer_norm_rms_epsilon` of the synthetic mixes: the file's.
    const EPS: f32 = 1e-5;

    /// Rows of the pass clause.
    const ROWS: usize = 8;

    /// One set's prompt and its tables.
    struct Set {
        name: &'static str,
        prompt: Vec<u32>,
        top: Vec<Top>,
        greedy: Greedy,
        /// The vendor's whole logits row at the prompt's last position (`result_output`).
        logits: Vec<f32>,
        /// The vendor's node of each stage of each layer ([`TAP_STAGES`]), every position's row,
        /// the last position's last; `None` where the layer's kind has no such node.
        stages: Vec<Vec<Option<Vec<f32>>>>,
        /// The vendor's routed picks (`ffn_moe_topk-L`), every position's ids; `None` for a dense
        /// layer.
        picks: Vec<Option<Vec<u32>>>,
    }

    /// The vendor's nodes of layer `l`, one a stage of [`TAP_STAGES`]: `mixer_out` is whichever of
    /// `kda_out` and `mla_out` the layer has (neither on a banking layer, whose mixer output is
    /// `prefix_sum_attn` itself). A node the set does not hold is `None`; one it holds is read in
    /// full.
    fn stage_nodes(man: &RefManifest, l: usize) -> Result<Vec<Option<Vec<f32>>>, GateError> {
        TAP_STAGES
            .iter()
            .map(|&stage| {
                let names: &[&str] = if stage == "mixer_out" {
                    &["kda_out", "mla_out"]
                } else {
                    &[stage]
                };
                names
                    .iter()
                    .find(|n| man.tensor(&format!("{n}-{l}"), 0).is_ok())
                    .map(|n| tap(man, &format!("{n}-{l}")))
                    .transpose()
            })
            .collect()
    }

    fn open_set(clause: &str, name: &'static str) -> Result<Set, GateError> {
        let path = tier::fixture_set(clause, &fixture::LCPP, name)?;
        let man = RefManifest::open(&path, &fixture::LCPP)?;
        let row = man.tensor("result_output", 0)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        let vocab = usize::try_from(row.ne[0])?;
        if vocab == 0 || all.len() % vocab != 0 {
            return Err(format!("{name}: result_output is {:?}", row.ne).into());
        }
        let set = Set {
            name,
            prompt: prompt_tables::tokens(&man)?.to_vec(),
            top: prompt_tables::topk(&man)?,
            greedy: prompt_tables::greedy(&man)?,
            logits: all[all.len() - vocab..].to_vec(),
            stages: (0..LAYERS)
                .map(|l| stage_nodes(&man, l))
                .collect::<Result<_, _>>()?,
            picks: (0..LAYERS)
                .map(|l| {
                    let Ok(row) = man.tensor(&format!("ffn_moe_topk-{l}"), 0) else {
                        return Ok(None);
                    };
                    let ids = topk_ids_logical_within(&man, row, N_EXPERT)?;
                    ids.into_iter()
                        .map(|i| u32::try_from(i).map_err(|e| GateError::from(e.to_string())))
                        .collect::<Result<Vec<_>, _>>()
                        .map(Some)
                })
                .collect::<Result<_, GateError>>()?,
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
            return Err(format!("{name}: {need} positions, the gate's caches hold {CTX}").into());
        }
        Ok(set)
    }

    /// Our argmax at each position of set `s`'s greedy table with the vendor's own ids fed back
    /// (teacher-forced): position 0 is the prompt's (`first`), position `i` the argmax after the
    /// vendor's `i` greedy ids.
    fn forced_table(m: &mut GpuModel<Body>, s: &Set, first: u32) -> Result<Vec<u32>, GateError> {
        let mut table = vec![first];
        for &id in &s.greedy.ids[..s.greedy.ids.len() - 1] {
            table.push(m.step(&[id])?);
        }
        Ok(table)
    }

    /// The disagreements the Gaussian error of two implementations predicts at the clear
    /// positions of `margins` (those at or past [`MARGIN_FLOOR`]), and the pin that follows: the
    /// expected count plus three standard deviations of it, rounded up.
    ///
    /// [derived]: at the vendor's margin `m` the argmax flips with probability
    /// Φ(−m / (√2 σ)), σ being [`SIGMA_LOGIT`], one implementation's logit deviation from the
    /// other's (`lcpp_e2e::forced_pin`, the common harness's form of it).
    fn forced_pin(margins: &[f32]) -> (f64, usize) {
        let sigma_pair = SIGMA_LOGIT * std::f64::consts::SQRT_2;
        let expected: f64 = margins
            .iter()
            .filter(|&&m| m >= MARGIN_FLOOR)
            .map(|&m| std_normal_cdf(-f64::from(m) / sigma_pair))
            .sum();
        (expected, (expected + 3.0 * expected.sqrt()).ceil() as usize)
    }

    /// Every stage of every layer at the last position of the model's last call against the
    /// vendor's node, relative L2: `None` where the vendor holds no such node or the layer's
    /// kind has no such stage.
    fn stage_rels(m: &GpuModel<Body>, s: &Set) -> Result<Vec<Vec<Option<f64>>>, GateError> {
        let (ours, stride) = m.body(NAME)?.taps(m.gpu().stream())?;
        let stages = TAP_STAGES.len();
        if ours.len() != LAYERS * stages * stride {
            return Err(format!(
                "{}: {} tap values for {LAYERS} layers of {stages} stages of {stride}",
                s.name,
                ours.len()
            )
            .into());
        }
        let tokens = s.prompt.len();
        s.stages
            .iter()
            .enumerate()
            .map(|(l, row)| {
                row.iter()
                    .enumerate()
                    .map(|(i, theirs)| {
                        let Some(theirs) = theirs else {
                            return Ok(None);
                        };
                        if theirs.len() % tokens != 0 {
                            return Err(format!(
                                "{}: {}-{l} holds {} values for {tokens} positions",
                                s.name,
                                TAP_STAGES[i],
                                theirs.len()
                            ));
                        }
                        let w = theirs.len() / tokens;
                        if w > stride {
                            return Err(format!(
                                "{}: {}-{l} is {w} wide, the taps' stride {stride}",
                                s.name, TAP_STAGES[i]
                            ));
                        }
                        let at = (l * stages + i) * stride;
                        Ok(Some(rel(&ours[at..at + w], &theirs[theirs.len() - w..])))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<_, _>>()
            .map_err(GateError::from)
    }

    /// The vendor's node `stage` of layer `l`: every position's row.
    fn node<'a>(s: &'a Set, l: usize, stage: &str) -> Option<&'a [f32]> {
        let i = TAP_STAGES.iter().position(|&t| t == stage)?;
        s.stages[l][i].as_deref()
    }

    /// The last of `tokens` rows of `all`.
    fn last_row(all: &[f32], tokens: usize) -> Result<&[f32], GateError> {
        if tokens == 0 || !all.len().is_multiple_of(tokens) {
            return Err(format!("{} values for {tokens} positions", all.len()).into());
        }
        Ok(&all[all.len() - all.len() / tokens..])
    }

    /// The clauses that hold a block alone to the vendor's own rows, so the chain's compounding
    /// does not enter: `forced_mixers_are_vendors` (each layer's mixer over the vendor's normed
    /// rows of every position, from a fresh store, against its output at the last position) and
    /// `forced_ffn_is_vendors` (each layer's block over the vendor's normed row at the last
    /// position, its update and its picks).
    fn forced_clauses(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        let tokens = s.prompt.len();
        let mut ok = true;
        if tier::run_clause(
            &format!("forced_mixers_are_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            let mut rels = Vec::new();
            for l in 0..LAYERS {
                let normed = node(s, l, "attn_norm").ok_or("a set without attn_norm")?;
                let n = normed.len() / tokens;
                let rows: Vec<Vec<f32>> = normed.chunks(n).map(<[f32]>::to_vec).collect();
                let theirs = node(s, l, "mixer_out")
                    .or_else(|| node(s, l, "prefix_sum_attn"))
                    .ok_or("a set without the mixer's output")?;
                let (gpu, w, body) = m.body_parts(NAME)?;
                let ours = body.mixer_alone(gpu, w, l, &rows)?;
                rels.push(rel(&ours, last_row(theirs, tokens)?));
            }
            m.reset()?;
            let worst = rels.iter().copied().fold(0.0f64, f64::max);
            let pass = worst <= FORCED_REL_BAND;
            println!(
                "forced_mixers_are_vendors set={} rel_by_layer={:?} band={FORCED_REL_BAND:.1e} {}",
                s.name,
                rels.iter().map(|r| format!("{r:.3e}")).collect::<Vec<_>>(),
                verdict(pass)
            );
            ok &= pass;
        }
        if tier::run_clause(
            &format!("forced_ffn_is_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            let mut rels = Vec::new();
            let mut same_picks = true;
            for l in 0..LAYERS {
                let xn = last_row(
                    node(s, l, "ffn_norm").ok_or("a set without ffn_norm")?,
                    tokens,
                )?;
                let theirs = last_row(
                    node(s, l, "ffn_out").ok_or("a set without ffn_out")?,
                    tokens,
                )?;
                let (gpu, w, body) = m.body_parts(NAME)?;
                let (out, picks) = body.ffn_alone(gpu, w, l, xn)?;
                rels.push(rel(&out, theirs));
                match (picks, &s.picks[l]) {
                    (Some(ours), Some(all)) => {
                        let k = ours.len();
                        let mut want = all[all.len() - k..].to_vec();
                        let mut have = ours;
                        want.sort_unstable();
                        have.sort_unstable();
                        let same = want == have;
                        println!(
                            "  picks set={} layer={l} ours={have:?} vendor={want:?}",
                            s.name
                        );
                        same_picks &= same;
                    }
                    (None, None) => {}
                    _ => {
                        return Err(format!(
                            "{}: layer {l}'s routed picks and the vendor's disagree in kind",
                            s.name
                        )
                        .into());
                    }
                }
            }
            let worst = rels.iter().copied().fold(0.0f64, f64::max);
            let pass = worst <= FORCED_REL_BAND && same_picks;
            println!(
                "forced_ffn_is_vendors set={} rel_by_layer={:?} picks_same={same_picks} band={FORCED_REL_BAND:.1e} {}",
                s.name,
                rels.iter().map(|r| format!("{r:.3e}")).collect::<Vec<_>>(),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// The vendor clauses of set `s` from a reset model: `top1_is_vendors`, `greedy_is_vendors`
    /// and `logits_are_vendors`.
    fn vendor_clauses(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        m.reset()?;
        let first = m.step(&s.prompt)?;
        let row = m.logits()?;
        let theirs = s
            .top
            .first()
            .ok_or_else(|| format!("{}: an empty top-k table", s.name))?;
        let ours5 = top_ids(&row, TOP_K);
        let d = s
            .top
            .iter()
            .map(|t| (row[t.id as usize] - t.logit).abs())
            .fold(0.0f32, f32::max);
        let logits_rel = rel(&row, &s.logits);
        let overlap = s
            .top
            .iter()
            .take(TOP_K)
            .filter(|t| ours5.contains(&(t.id as usize)))
            .count();
        let mut ok = true;
        let rels = stage_rels(m, s)?;
        if tier::run_clause(
            &format!("layer_outputs_are_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            for (l, row) in rels.iter().enumerate() {
                let cells = TAP_STAGES
                    .iter()
                    .zip(row)
                    .map(|(name, r)| match r {
                        Some(r) => format!("{name}={r:.2e}"),
                        None => format!("{name}=-"),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                println!("  stages set={} layer={l}: {cells}", s.name);
            }
            let l_out = TAP_STAGES.len() - 1;
            let outs: Vec<f64> = rels
                .iter()
                .map(|row| row[l_out].unwrap_or(f64::INFINITY))
                .collect();
            let pass = outs.iter().zip(&LAYER_REL_BAND).all(|(r, band)| r <= band);
            let by_layer = outs
                .iter()
                .map(|r| format!("{r:.3e}"))
                .collect::<Vec<_>>()
                .join(",");
            println!(
                "layer_outputs_are_vendors set={} l_out_rel_by_layer=[{by_layer}] bands={LAYER_REL_BAND:?} {}",
                s.name,
                verdict(pass)
            );
            ok &= pass;
        }
        if tier::run_clause(
            &format!("forced_top1_is_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            let table = forced_table(m, s, first)?;
            let sigma_hat = (s
                .top
                .iter()
                .map(|t| f64::from(row[t.id as usize] - t.logit).powi(2))
                .sum::<f64>()
                / s.top.len() as f64)
                .sqrt();
            let vendor = GreedyRow {
                id: 0,
                n_tokens: s.prompt.len(),
                argmax: theirs.id,
                top5: s.top.iter().take(TOP_K).map(|t| t.id).collect(),
                gen_ids: s.greedy.ids.clone(),
                gen_margins: s.greedy.margins.clone(),
            };
            let report = compare_forced(&[table], &[vendor], MARGIN_FLOOR);
            let (expected, pin) = forced_pin(&s.greedy.margins);
            let pass = report.disagree_ge_floor <= pin;
            println!(
                "forced_top1_is_vendors set={} positions={} disagree={} buckets={:?} ge_floor={} expected {expected:.2} pin {pin} \
                 prompt_top1_ours={first} vendor={} top{TOP_K}_overlap={overlap}/{TOP_K} d={d} sigma_hat={sigma_hat:.3} {}",
                s.name,
                report.positions,
                report.disagree.len(),
                report.buckets,
                report.disagree_ge_floor,
                theirs.id,
                verdict(pass)
            );
            ok &= pass;
        }
        if tier::run_clause(
            &format!("logits_are_vendors set={}", s.name),
            Tag::FixtureOracle,
        )? {
            let pass = logits_rel <= LOGITS_REL_BAND;
            println!(
                "logits_are_vendors set={} logits_rel={logits_rel:.4e} band={LOGITS_REL_BAND:.1e} {}",
                s.name,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// Clause `steps_are_deterministic`: set `a`'s prompt twice from a reset gives its logits row
    /// bit for bit, and so does a reset after set `b`'s prompt and its continuation.
    fn steps_are_deterministic(
        m: &mut GpuModel<Body>,
        (a, b): (&Set, &Set),
    ) -> Result<bool, GateError> {
        let run = |m: &mut GpuModel<Body>, s: &Set| -> Result<(u32, Vec<f32>), GateError> {
            m.reset()?;
            let id = m.step(&s.prompt)?;
            let row = m.logits()?;
            Ok((id, row))
        };
        let first = run(m, a)?;
        let again = run(m, a)?;
        let _ = run(m, b)?;
        let after = m.step(&[first.0])?;
        let _ = (after, m.logits()?);
        let third = run(m, a)?;
        let same = |x: &(u32, Vec<f32>), y: &(u32, Vec<f32>)| x.0 == y.0 && bits_equal(&x.1, &y.1);
        let pass = same(&first, &again) && same(&first, &third);
        println!(
            "steps_are_deterministic set={} ids={}/{}/{} again_same={} after_other_set_same={} {}",
            a.name,
            first.0,
            again.0,
            third.0,
            same(&first, &again),
            same(&first, &third),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `graph_is_eager`: set `s`'s prompt and 8 greedy steps in graph mode give the eager
    /// chain's ids and last logits row bit for bit.
    fn graph_is_eager(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        let run =
            |m: &mut GpuModel<Body>, mode: StepMode| -> Result<(Vec<u32>, Vec<f32>), GateError> {
                m.set_mode(mode);
                m.reset()?;
                let mut ids = vec![m.step(&s.prompt)?];
                while ids.len() <= 8 {
                    let last = ids[ids.len() - 1];
                    ids.push(m.step(&[last])?);
                }
                Ok((ids, m.logits()?))
            };
        let eager = run(m, StepMode::Eager)?;
        let graph = run(m, StepMode::Graph)?;
        m.set_mode(StepMode::Eager);
        let pass = eager.0 == graph.0 && bits_equal(&eager.1, &graph.1);
        println!(
            "graph_is_eager set={} eager={:?} graph={:?} logits_same_bits={} {}",
            s.name,
            eager.0,
            graph.0,
            bits_equal(&eager.1, &graph.1),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The ids after `ids` that [`generate_matches_the_gate`] asks `generate_kimi_k3` for.
    const GENERATE_IDS: usize = 8;

    /// The `ids [..]` line `generate_kimi_k3` prints, as ids.
    fn printed_ids(stdout: &str) -> Result<Vec<u32>, GateError> {
        let line = stdout
            .lines()
            .find_map(|l| l.strip_prefix("ids ["))
            .ok_or_else(|| format!("generate_kimi_k3 printed no ids line:\n{stdout}"))?;
        let body = line
            .strip_suffix(']')
            .ok_or_else(|| format!("generate_kimi_k3's ids line is not a list: {line}"))?;
        body.split(',')
            .map(|t| t.trim().parse::<u32>().map_err(GateError::from))
            .collect()
    }

    /// Clause `generate_matches_the_gate`: `generate_kimi_k3` over set `s`'s prompt ids gives the
    /// ids this gate's own steps give from the same ids in graph mode, [`GENERATE_IDS`] of them;
    /// the line prints the vendor's greedy table beside them and where the two part.
    fn generate_matches_the_gate(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut ours = vec![m.step(&s.prompt)?];
        while ours.len() < GENERATE_IDS {
            let last = ours[ours.len() - 1];
            ours.push(m.step(&[last])?);
        }
        m.set_mode(StepMode::Eager);
        let prompt = s
            .prompt
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let bin = beside("generate_kimi_k3")?;
        let out = std::process::Command::new(&bin)
            .args(["--tokens", &prompt])
            .args(["-n", &GENERATE_IDS.to_string()])
            .args(["--ctx", &CTX.to_string(), "--place", "a"])
            .output()
            .map_err(|e| format!("{}: {e}", bin.display()))?;
        if !out.status.success() {
            return Err(format!(
                "{} ended {}:\n{}",
                bin.display(),
                out.status,
                String::from_utf8_lossy(&out.stderr)
            )
            .into());
        }
        let printed = printed_ids(&String::from_utf8_lossy(&out.stdout))?;
        let vendor = &s.greedy.ids[..GENERATE_IDS.min(s.greedy.ids.len())];
        let parts_at = ours.iter().zip(vendor).position(|(a, b)| a != b);
        let pass = printed == ours;
        println!(
            "generate_matches_the_gate set={} generate={printed:?} gate={ours:?} vendor={vendor:?} \
             parts_from_vendor_at={parts_at:?} vendor_margin_there={:?} {}",
            s.name,
            parts_at.map(|i| s.greedy.margins[i]),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------------ the AttnRes mix

    /// One synthetic mix: `n_banks` banks and the stream, `m` rows of `n`.
    struct Mix {
        n: usize,
        n_banks: usize,
        m: usize,
        /// The values between two banks: `m · n` and `slack` more.
        stride: usize,
        banks: Vec<f32>,
        cur: Vec<f32>,
        score: Vec<f32>,
    }

    impl Mix {
        /// Vectors of different scales (0.5 to 4), scores scaled so a vector's score is of order 1
        /// (the softmax is not one-hot).
        fn new(n: usize, n_banks: usize, m: usize, seed: u64) -> Mix {
            let mut r = GateRng(seed);
            let stride = m * n + 3 * n;
            let scale = |i: usize| [0.5f32, 1.0, 2.0, 4.0][i % 4];
            let mut banks = vec![0.0f32; n_banks * stride];
            for i in 0..n_banks {
                for v in &mut banks[i * stride..i * stride + m * n] {
                    *v = r.uni(-1.0, 1.0) * scale(i);
                }
            }
            let cur = (0..m * n).map(|_| r.uni(-1.0, 1.0) * 1.5).collect();
            let lim = 2.6 / (n as f32).sqrt();
            let score = (0..n).map(|_| r.uni(-lim, lim)).collect();
            Mix {
                n,
                n_banks,
                m,
                stride,
                banks,
                cur,
                score,
            }
        }

        /// The definition in f64: each vector's score is its RMS-normed values dotted with the
        /// scores, the softmax over the banks and the stream, the weighted sum of the raw vectors.
        fn host(&self) -> Vec<f64> {
            let (n, nb) = (self.n, self.n_banks);
            let mut out = vec![0.0f64; self.m * n];
            for t in 0..self.m {
                let vec_of = |i: usize| -> &[f32] {
                    if i < nb {
                        &self.banks[i * self.stride + t * n..][..n]
                    } else {
                        &self.cur[t * n..][..n]
                    }
                };
                let s: Vec<f64> = (0..=nb)
                    .map(|i| {
                        let v = vec_of(i);
                        let ms = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / n as f64;
                        let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
                        let dot: f64 = v
                            .iter()
                            .zip(&self.score)
                            .map(|(&x, &w)| f64::from(x) * f64::from(w))
                            .sum();
                        inv * dot
                    })
                    .collect();
                let mx = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let total: f64 = e.iter().sum();
                for j in 0..n {
                    out[t * n + j] = (0..=nb)
                        .map(|i| e[i] / total * f64::from(vec_of(i)[j]))
                        .sum();
                }
            }
            out
        }
    }

    /// The card's mix of `x`, `rows` rows from row `at`: `y`, `m · n` values.
    fn card_mix(
        gpu: &Gpu,
        k: &AttnResKernels,
        x: &Mix,
        (at, rows): (usize, usize),
    ) -> Result<Vec<f32>, GateError> {
        let s = gpu.stream();
        let n = x.n;
        // The rows `at..at + rows` of every bank, repacked at the stride `rows · n` apart.
        let stride = rows * n;
        let mut banks = vec![0.0f32; (x.n_banks * stride).max(1)];
        for i in 0..x.n_banks {
            banks[i * stride..(i + 1) * stride]
                .copy_from_slice(&x.banks[i * x.stride + at * n..][..stride]);
        }
        let banks = DeviceBuffer::from_host(s, &banks)?;
        let cur = DeviceBuffer::from_host(s, &x.cur[at * n..(at + rows) * n])?;
        let score = DeviceBuffer::from_host(s, &x.score)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, rows * n)?;
        k.enqueue(
            s,
            AttnResArgs {
                banks: &banks,
                n_banks: x.n_banks,
                stride,
                cur: &cur,
                score: &score,
                eps: EPS,
                n,
                m: rows,
                fault: gpu.layer_sink(0)?,
            },
            &mut y,
        )?;
        Ok(y.to_host_vec(s)?)
    }

    /// The mix clauses on the model's card: the synthetic cases of the module doc.
    fn mix_clauses(gpu: &Gpu) -> Result<bool, GateError> {
        let k = AttnResKernels::load(gpu.context())?;
        let mut ok = true;
        // The fixture's width at 1, 2 and 15 banks, the real file's final mix (7168, 8 banks) and
        // a pass of 8 rows at 4 banks.
        let cases = [
            (512usize, 1usize, 1usize),
            (512, 2, 1),
            (512, 15, 1),
            (7168, 8, 1),
            (512, 4, ROWS),
        ];
        let mut worst = 0.0f64;
        let mut dropped = f64::INFINITY;
        for (seed, &(n, nb, m)) in cases.iter().enumerate() {
            let x = Mix::new(n, nb, m, 0x9e37_79b9 + seed as u64);
            let ours = card_mix(gpu, &k, &x, (0, m))?;
            let theirs = x.host();
            let r = rel_f64(&ours, &theirs);
            worst = worst.max(r);
            let off = rel_f64(&x.cur, &theirs);
            dropped = dropped.min(off);
            println!("  mix n={n} banks={nb} rows={m}: rel={r:.3e} stream_alone_rel={off:.3e}");
        }
        let pass = worst <= MIX_BAND;
        println!(
            "mix_matches_host worst_rel={worst:.3e} band={MIX_BAND:.1e} over {} cases {}",
            cases.len(),
            verdict(pass)
        );
        ok &= pass;
        let pass = dropped > SEP * MIX_BAND;
        println!(
            "mix_is_the_mix stream_alone_rel_min={dropped:.3e} past {SEP} bands ({:.1e}) {}",
            SEP * MIX_BAND,
            verdict(pass)
        );
        ok &= pass;
        // A pass of 8 rows is 8 one-row mixes.
        let x = Mix::new(512, 4, ROWS, 0x51ed);
        let whole = card_mix(gpu, &k, &x, (0, ROWS))?;
        let mut stepped = Vec::new();
        for t in 0..ROWS {
            stepped.extend(card_mix(gpu, &k, &x, (t, 1))?);
        }
        let pass = bits_equal(&whole, &stepped);
        println!("mix_rows_are_steps rows={ROWS} {}", verdict(pass));
        ok &= pass;
        ok &= mix_refused(gpu, &k)?;
        ok &= mix_faults(gpu, &k)?;
        Ok(ok)
    }

    /// Relative L2 of the card's rows from the host's, in f64.
    fn rel_f64(a: &[f32], b: &[f64]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - y).powi(2);
            den += y * y;
        }
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_nan() { f64::INFINITY } else { r }
    }

    /// Clause `mix_refused`: each shape the launch does not take is refused by name.
    fn mix_refused(gpu: &Gpu, k: &AttnResKernels) -> Result<bool, GateError> {
        let s = gpu.stream();
        let n = 512usize;
        let buf = |len: usize| DeviceBuffer::<f32>::zeroed(s, len);
        let (banks, cur, score) = (buf(20 * n)?, buf(2 * n)?, buf(n)?);
        let mut y = buf(2 * n)?;
        let mut short = buf(n)?;
        let fault = gpu.layer_sink(0)?;
        let args = |n_banks: usize, stride: usize, m: usize, eps: f32| AttnResArgs {
            banks: &banks,
            n_banks,
            stride,
            cur: &cur,
            score: &score,
            eps,
            n,
            m,
            fault,
        };
        let mut out = true;
        let mut check = |what: &str, got: Result<(), GpuError>, want: &str| {
            let text = got.err().map(|e| e.to_string());
            let pass = text.as_deref().is_some_and(|t| t.contains(want));
            println!(
                "  refused {what}: {text:?} (wants {want:?}) {}",
                verdict(pass)
            );
            out &= pass;
        };
        check(
            "no rows",
            k.enqueue(s, args(1, 2 * n, 0, EPS), &mut y),
            "0 rows",
        );
        check(
            "16 banks and the stream",
            k.enqueue(s, args(MAX_VECS, 2 * n, 2, EPS), &mut y),
            "at most 16",
        );
        check(
            "a stride under the pass's rows",
            k.enqueue(s, args(1, n, 2, EPS), &mut y),
            "stride of",
        );
        check(
            "epsilon 0",
            k.enqueue(s, args(1, 2 * n, 2, 0.0), &mut y),
            "epsilon",
        );
        check(
            "a short output",
            k.enqueue(s, args(1, 2 * n, 2, EPS), &mut short),
            "the output",
        );
        println!("mix_refused {}", verdict(out));
        Ok(out)
    }

    /// Clause `mix_faults`: a NaN score raises the fault word and the row is NaN. Clears the word.
    fn mix_faults(gpu: &Gpu, k: &AttnResKernels) -> Result<bool, GateError> {
        let mut x = Mix::new(512, 2, 1, 0xfa17);
        x.score[3] = f32::NAN;
        let y = card_mix(gpu, k, &x, (0, 1))?;
        let fault = gpu.fault()?;
        let nan = y.iter().all(|v| v.is_nan());
        let pass = fault.is_some() && nan;
        println!(
            "mix_faults fault={fault:?} row_all_nan={nan} {}",
            verdict(pass)
        );
        gpu.clear_fault()?;
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the real file (594 GB) does not fit one card and the body has no placed \
                 load, so the gate runs the fixture tier: BLOOMERY_TIER=fixture just {RECIPE}"
            )
            .into());
        }
        let sets: Vec<Set> = SETS
            .iter()
            .map(|&n| open_set("the vendor clauses", n))
            .collect::<Result<_, _>>()?;
        let file = open_split(Arch::KimiK3, RECIPE)?;
        let t = std::time::Instant::now();
        let opened = GpuModel::<Body>::open(Gpu::new()?, file, CTX);
        let mut m = match opened {
            Ok(m) => m,
            Err(e) => {
                println!("opens_the_fixture {e} {}", verdict(false));
                return Err(e.into());
            }
        };
        let layers_ok = m.layers().len() == LAYERS;
        println!(
            "opens_the_fixture resident_bytes={} store_bytes={} ctx={CTX} layers={} in {:.1} s {}",
            m.resident_bytes(),
            m.body(NAME)?.store_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64(),
            verdict(layers_ok)
        );
        let mut ok = layers_ok;
        if !layers_ok {
            return Err(checks_failed());
        }
        {
            let (gpu, _, body) = m.body_parts(NAME)?;
            body.arm_taps(gpu)?;
        }
        for s in &sets {
            ok &= vendor_clauses(&mut m, s)?;
            ok &= forced_clauses(&mut m, s)?;
        }
        tier::expect_fixture_oracle(NAME, FX_ORACLE)?;
        ok &= steps_are_deterministic(&mut m, (&sets[0], &sets[1]))?;
        ok &= graph_is_eager(&mut m, &sets[3])?;
        ok &= generate_matches_the_gate(&mut m, &sets[3])?;
        ok &= mix_clauses(m.gpu())?;
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} on every set of {SETS:?} the free chain's layers 0 to 2 stay within \
             {LAYER_REL_BAND:?} of the vendor's, each mixer and feed-forward block alone on the \
             vendor's rows within {FORCED_REL_BAND:.1e} with its picks, the teacher-forced argmax \
             disagreements within the pin, the logits row within {LOGITS_REL_BAND:.1e}; the steps \
             are deterministic and the graph is the eager chain; the AttnRes mix is its definition"
        );
        Ok(())
    }
}
