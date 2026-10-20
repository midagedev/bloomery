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
//!   layer; a group-16 attention layer's one group-8 segment pass (two packs a
//!   key head), one merge and `2 · n_kv` appends; a MoE layer's router at its own count,
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
//! - `prefill_pass_size_invariant` (self-consistency): one walk of the most
//!   ids that keep every pass past the gemv arm's eight rows at two pass
//!   sizes — the load's own rows and half of them — through
//!   `GpuModel::prefill_with` from a reset: the final logits and the greedy
//!   id bit-equal, the pass size moving no bit (the GEMMs accumulate each
//!   (slot, row) over K blocks in increasing k whatever tile it lands in,
//!   and the router, the quantizers, the flash split and the scan work per
//!   row; the scan state, the conv ring and the K/V rows carry across
//!   passes in memory). The sizes are the load's own choice
//!   (`NemotronHMoeModel::pass_rows`), whatever the card's room gave it.
//! - `watchdog_progress` (self-consistency): the walk's ids as one
//!   `GpuModel::step` call under the prompt call's busy mark
//!   (`GpuModel::prompt_busy`) move the engine watchdog's progress word
//!   (`GpuModel::engine_progress`) at least once a token. The mark's holder is
//!   the caller, so the call adds none of its own and the word moves by the
//!   tokens' passes alone; without the pass-done call of `GpuModel::step`'s
//!   token loop it does not move, and the watchdog would name a long prompt
//!   run as steps wedged.
//!
//! - `host_tier_near_whole_card` (self-consistency, after the clauses above,
//!   on loads of its own once the whole-card model is released): the routed
//!   experts where the planner's three arms put them (`Experts::Placed`
//!   under no budget — the card holds every expert, the placed load running
//!   no host leg; under a budget of half the experts' bytes — each MoE layer
//!   partly on the host; `Experts::Host` — every expert on the host; and in
//!   the real tier the 12 GiB room), the walk run as decode steps through
//!   `GpuModel::open_placed`. The card arm's logits are bit-identical to the
//!   whole-card load's; a host arm's keep [`COSINE_FLOOR`] against them and
//!   its argmax is the whole-card's wherever the whole-card's top-1 margin
//!   passes twice the run's max |Δ| (the host quantizes the latent with its
//!   own kernel and sums its slots in its own order, so the sums differ by
//!   rounding: the two ids can swap only inside that band). Each host arm's
//!   graph holds one `ds41_ffn_handoff_9` a hosted layer and one
//!   `ds41_ffn_card_acc_9` a hosted layer with card experts, its eager and
//!   replayed runs are bit-identical, and its fault word stays clean. It
//!   blocks a host share lost or counted twice in the latent sum. Each arm
//!   then runs the walk's first [`GEMV_ROWS`] ids and the whole walk as a
//!   prompt batch (`GpuModel::prefill`: the gemv arm's `_sel` launches on the
//!   places, the GEMM arm's grouped launches on the remapped route table):
//!   the batch service of the host tier runs one serve a hosted layer over
//!   the prompt's columns. The whole-card load's batch of the same ids is the
//!   reference (the same kernels, so only the experts' placement differs): the
//!   card arm's logits are bit-identical to it, a host arm's keep
//!   [`COSINE_FLOOR`] with its argmax wherever its top-1 margin passes twice
//!   the max |Δ|, and [`CONTINUE`] steps after the batch (fed the reference's
//!   greedy ids) keep the same band at each position against the reference's
//!   own, which holds the stores the batch leaves. The batch's last position
//!   also keeps the floor and the argmax band against the arm's own decode
//!   steps; the continuation does not run against the steps, because the
//!   batch and the steps sum in another order and the fixture's near-tied
//!   routers turn that into a different expert past the first position. A
//!   launch of the whole-card load's walks or batches, or of an arm's, that
//!   raises the fault word fails the clause, the whole-card load's too: the
//!   arms are then held against nothing, and the clause is red, not skipped.
//!   Each arm also recovers from a fault on every path: a step, a gemv-arm
//!   batch and a GEMM-arm batch whose first id is past the embedding table
//!   raise the fault word, and after a reset the valid call that ran before
//!   it runs clean with the same bits (a hosted layer's launches skip the
//!   slots the host serves, so a reset must clear the rows the faulted call
//!   left in them).
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
#[allow(
    dead_code,
    reason = "the card the gate plans on; the host-tier arms read its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
mod gate {
    use crate::e2e::{
        Run, argmax, cosine, feed, finite, fresh, histogram, max_delta, or_fault, same, same_bits,
        second, top_ids, walk,
    };
    use crate::gate_card;
    use bloomery_gpu::arch::nemotron_h_moe::Routing;
    use bloomery_gpu::host::served::TierBody;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, NemotronHMoeModel};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{GateError, checks_failed, open_split, verdict};
    use cuda_core::sys;
    use model::arch::Arch;
    use model::arch::nemotron_h_moe::hparams::{Hparams, Kind};
    use model::arch::nemotron_h_moe::place::{self as moe_place, Experts};
    use model::placement::{PlanLevers, workstation};
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

    /// The most rows of a prompt that run the gemv arm (the pass's `_sel`
    /// launches); a longer one runs the GEMM arm.
    const GEMV_ROWS: usize = 8;

    /// Steps after a prompt batch the host-tier clause compares with the
    /// whole-card load's.
    const CONTINUE: usize = 4;

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
        (
            "prefill_pass_size_invariant",
            Tag::SelfConsistency,
            prefill_pass_size_invariant,
        ),
        ("watchdog_progress", Tag::SelfConsistency, watchdog_progress),
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
        // The group-16 pass is one group-8 segment entry a layer over the key
        // heads' planes (`packs` 2); no group-5 entry runs.
        // PIN(2026-10-18): one segment node a layer (was `2 * n_kv`, one a key head and pack).
        let seg = |g5: bool| -> usize {
            hist.iter()
                .filter(|(k, _)| k.starts_with("gqa_flash_seg") && k.ends_with("_g5") == g5)
                .map(|(_, n)| n)
                .sum()
        };
        let (group8, group5) = (seg(false), seg(true));
        pass &= group8 == attn && group5 == 0;
        let listed: Vec<String> = want
            .iter()
            .map(|(name, n)| format!("{name}={}/{n}", got(name)))
            .collect();
        println!(
            "structure: layers mamba={mamba} attn={attn} moe={moe}, graph_nodes={nodes} host={host}, \
             entries (got/want) {} group8_segments={group8}/{} group5_entries={group5} (want 0) {}",
            listed.join(" "),
            attn,
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

    // ------------------------------------------- prefill_pass_size_invariant

    /// One walk through [`GpuModel::prefill_with`] at two pass sizes — the
    /// load's own rows (`pass_rows`) and half of them — over the most ids
    /// that keep every pass, tail included, past the gemv arm's eight rows
    /// at both sizes: the final logits and the greedy id bit-equal (module
    /// doc). The walk is the clause's own, so no set is read and both tiers
    /// run it; a load whose rows leave no such pair is skipped by name.
    fn prefill_pass_size_invariant(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "prefill_pass_size_invariant";
        // The multi-column gemv arm's most columns: a pass past it runs the
        // GEMMs, whose sums no pass size may move.
        const ARM: usize = 8;
        let cap = m
            .pass_rows()
            .map_err(|e| format!("{CLAUSE}: the pass rows: {e}"))?;
        let small = cap / 2;
        if small <= ARM {
            println!("{CLAUSE}: the load's {cap} rows leave no second size past {ARM}, skipped");
            return Ok(true);
        }
        let ctx = m.ctx_max();
        let Some(n) = (ARM + 1..=ctx).rev().find(|&n| {
            let whole = |rows: usize| n % rows == 0 || n % rows > ARM;
            n > small && whole(small) && whole(cap)
        }) else {
            println!(
                "{CLAUSE}: no walk of the {ctx}-row context keeps every pass past {ARM} at \
                 {small} and {cap} rows, skipped"
            );
            return Ok(true);
        };
        let ids = walk(n, case.hp.n_vocab);
        let mut runs = Vec::with_capacity(2);
        for rows in [small, cap] {
            m.reset()?;
            let Some(id) = or_fault(CLAUSE, "the walk", m.prefill_with(&ids, rows))? else {
                return Ok(false);
            };
            let logits = m.logits()?;
            if !logits.iter().all(|v| v.is_finite()) {
                println!("{CLAUSE}: rows {rows}: a non-finite logit FAIL");
                return Ok(false);
            }
            runs.push((id, logits));
        }
        let [(id_a, logits_a), (id_b, logits_b)] = &runs[..] else {
            return Err(format!("{CLAUSE}: two runs").into());
        };
        let bits = same_bits(logits_a, logits_b);
        let pass = id_a == id_b && bits;
        println!(
            "{CLAUSE}: walk{n} at rows {small} ({} passes) and {cap} ({} passes): ids {id_a} \
             {id_b}, logits bit-identical {bits} {}",
            n.div_ceil(small),
            n.div_ceil(cap),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------- host_tier_near_whole_card

    /// The host-tier clause's name.
    const HOST_CLAUSE: &str = "host_tier_near_whole_card";

    /// The 12 GiB room of the 3060 host, in bytes.
    const ROOM_12G: u64 = 12 << 30;

    /// One arm of the host split: its name, the levers it plans under and
    /// where the planner puts the experts.
    struct Arm {
        name: &'static str,
        levers: PlanLevers,
        experts: Experts,
    }

    /// The whole-card load's run of the walk the host-tier arms are held
    /// against: the ids and the last position's logits, kept before the
    /// model is released.
    struct Whole {
        run: Run,
        batch: Vec<BatchRun>,
        ctx: usize,
    }

    /// The whole-card load's walk and batches, or `None` after a line naming
    /// the fault a launch raised.
    fn whole_card_run(m: &mut Model, case: &Case) -> Result<Option<Whole>, GateError> {
        m.set_mode(StepMode::Graph);
        let Some(run) = fresh(
            "host_tier_near_whole_card",
            "the whole-card walk",
            m,
            &case.walk,
        )?
        else {
            return Ok(None);
        };
        let Some(batch) = whole_batches(m, case)? else {
            return Ok(None);
        };
        Ok(Some(Whole {
            run,
            batch,
            ctx: m.ctx_max(),
        }))
    }

    /// The host-tier clause (module doc): the whole-card load has been
    /// released, each arm opens the file as a placed load of its plan.
    fn host_tier_near_whole_card(case: &Case, whole: &Whole) -> Result<bool, GateError> {
        const CLAUSE: &str = HOST_CLAUSE;
        let file = open_split(Arch::NemotronHMoe, "gate-gpu-nemotron-h-moe-e2e")?;
        let inputs = moe_place::read(&file)?;
        let spec = gate_card::card()?;
        let machine = workstation::placed_on(spec, case.hp.n_layer, workstation::SCRATCH);
        let ctx = u64::try_from(whole.ctx)?;
        // No lever is read: the clause plans and loads by its arms alone, so a
        // lever left in the runner's environment cannot move it.
        let host = bloomery_levers::HostCfg {
            populate: false,
            lock: false,
            card_dontneed: false,
            r8: false,
            route_trace: false,
        };
        let all_card = inputs.plan_with(&machine, ctx, &PlanLevers::default(), Experts::Placed)?;
        let c = &all_card.cards[0];
        let half = c.dense_bytes
            + c.rounding_bytes
            + c.kv_bytes
            + c.stage_rows_bytes
            + c.expert_bytes / 2
            + machine.cards[0]
                .set_aside_bytes()
                .ok_or("the card's set-aside is unbounded")?
            + machine.cards[0].margin_bytes;
        let mut arms = vec![
            Arm {
                name: "card",
                levers: PlanLevers::default(),
                experts: Experts::Placed,
            },
            Arm {
                name: "mid",
                levers: PlanLevers {
                    card_budget_bytes: Some(half),
                },
                experts: Experts::Placed,
            },
            Arm {
                name: "host",
                levers: PlanLevers::default(),
                experts: Experts::Host,
            },
        ];
        if tier::Tier::from_env()? == tier::Tier::Real {
            arms.push(Arm {
                name: "room12g",
                levers: PlanLevers {
                    card_budget_bytes: Some(ROOM_12G),
                },
                experts: Experts::Placed,
            });
        }
        let (Some(want_ids), Some(want)) = (whole.run.ids.last(), whole.run.logits.last()) else {
            return Err("the whole-card walk is empty".into());
        };
        let mut pass = true;
        for arm in &arms {
            let plan = inputs.plan_with(&machine, ctx, &arm.levers, arm.experts)?;
            // A layer is hosted when the card holds fewer than the model's
            // experts of it; a layer with no routed stacks is neither.
            let split: Vec<(usize, u64, u64)> = (0..case.hp.n_layer)
                .filter_map(|l| {
                    let x = case.hp.experts[l]?;
                    Some((l, plan.n_l[l], x.n_expert as u64))
                })
                .collect();
            let hosted: Vec<usize> = split
                .iter()
                .filter(|&&(_, on, of)| on < of)
                .map(|&(l, _, _)| l)
                .collect();
            let with_card = split
                .iter()
                .filter(|&&(_, on, of)| 0 < on && on < of)
                .count();
            println!(
                "{CLAUSE}: arm {} plan: {} experts on the host, card experts by layer \
                 (layer, on card, of) {split:?}, hosted layers {hosted:?}",
                arm.name, plan.host.experts
            );
            let mut m = Model::open_placed(
                open_split(Arch::NemotronHMoe, "gate-gpu-nemotron-h-moe-e2e")?,
                &plan,
                0,
                host,
            )?;
            m.set_mode(StepMode::Eager);
            let eager = fresh(CLAUSE, "eager", &mut m, &case.walk)?;
            m.set_mode(StepMode::Graph);
            let nodes = m.capture_step()?;
            let hist = histogram(&m.step_graph_nodes()?);
            let replay = fresh(CLAUSE, "replay", &mut m, &case.walk)?;
            let (Some(eager), Some(replay)) = (eager, replay) else {
                pass = false;
                continue;
            };
            let got = |name: &str| hist.get(name).copied().unwrap_or(0);
            let moe_whole = case
                .hp
                .kinds
                .iter()
                .enumerate()
                .filter(|&(l, &k)| k == Kind::Moe && !hosted.contains(&l))
                .count();
            let (handoff, acc, wsum) = (
                got("ds41_ffn_handoff_9"),
                got("ds41_ffn_card_acc_9"),
                got("weighted_sum"),
            );
            let structure = handoff == hosted.len() && acc == with_card && wsum == moe_whole;
            let (Some(got_ids), Some(logits)) = (replay.ids.last(), replay.logits.last()) else {
                return Err("the placed walk is empty".into());
            };
            let (host_near, cos, dmax, _) = in_band((*want_ids, want), (*got_ids, logits));
            let clean = m.gpu().fault()?.is_none() && m.poisoned().is_none();
            let fine = finite(&replay) && same(&eager, &replay);
            let near = if arm.name == "card" {
                same(&whole.run, &replay)
            } else {
                host_near
            };
            let batch = batch_near_whole(&mut m, case, whole, (arm.name, &hosted))?;
            let recovers = fault_recovery(&mut m, case, arm.name)?;
            let arm_pass = structure
                && fine
                && clean
                && near
                && batch
                && recovers
                && (arm.name != "card" || hosted.is_empty());
            let resident = m.resident_bytes();
            println!(
                "{CLAUSE}: arm {}: graph_nodes={nodes} resident_bytes={resident} (runtime value) \
                 handoff_9={handoff}/{} card_acc_9={acc}/{} \
                 weighted_sum={wsum}/{moe_whole}; eager == replay {} finite {} fault clean {clean}; \
                 last id {got_ids} vs whole-card {want_ids}, cosine {cos:.6}, max|Δ| {dmax:.3e}; \
                 host band {host_near} {}",
                arm.name,
                hosted.len(),
                with_card,
                same(&eager, &replay),
                finite(&replay),
                verdict(arm_pass)
            );
            pass &= arm_pass;
        }
        Ok(pass)
    }

    /// Whether `got` keeps [`COSINE_FLOOR`] against `want`, and its id is
    /// `want`'s or `want`'s top-1 margin is within twice their max |Δ|: the
    /// host arms' band. The cosine, the max |Δ| and whether the ids are equal
    /// come back for the line.
    fn in_band(want: (u32, &[f32]), got: (u32, &[f32])) -> (bool, f64, f64, bool) {
        let ((want_id, want), (got_id, got)) = (want, got);
        let (dmax, cos) = (max_delta(want, got), cosine(want, got));
        let margin = f64::from(want[argmax(want) as usize])
            - f64::from(want[second(want, argmax(want)) as usize]);
        let same_id = want_id == got_id;
        (
            cos >= COSINE_FLOOR && (same_id || margin <= 2.0 * dmax),
            cos,
            dmax,
            same_id,
        )
    }

    /// A prompt run as one `GpuModel::prefill` batch from a reset: the id and
    /// the logits after the prompt, then [`CONTINUE`] decode steps (each
    /// step's id and logits).
    struct BatchRun {
        id: u32,
        logits: Vec<f32>,
        after: Vec<(u32, Vec<f32>)>,
    }

    /// `prompt` as a batch, then [`CONTINUE`] steps fed `forced`'s greedy ids
    /// (its own where there is none). `None` when a launch raised the fault
    /// word.
    fn batch_run(
        m: &mut Model,
        prompt: &[u32],
        forced: Option<&BatchRun>,
    ) -> Result<Option<BatchRun>, GateError> {
        const CLAUSE: &str = HOST_CLAUSE;
        m.reset()?;
        let Some(id) = or_fault(CLAUSE, "the prompt as a batch", m.prefill(prompt))? else {
            return Ok(None);
        };
        let logits = m.logits()?;
        let mut after = Vec::with_capacity(CONTINUE);
        let mut tok = forced.map_or(id, |f| f.id);
        for i in 0..CONTINUE {
            let Some(next) = or_fault(CLAUSE, "a step after the batch", m.step(&[tok]))? else {
                return Ok(None);
            };
            after.push((next, m.logits()?));
            tok = forced.map_or(next, |f| f.after[i].0);
        }
        Ok(Some(BatchRun { id, logits, after }))
    }

    /// The whole-card load's batch of the walk's first [`GEMV_ROWS`] ids and
    /// of the whole walk: what each arm's batches are held against.
    fn whole_batches(m: &mut Model, case: &Case) -> Result<Option<Vec<BatchRun>>, GateError> {
        let mut out = Vec::new();
        for len in [GEMV_ROWS, case.walk.len()] {
            let Some(run) = batch_run(m, &case.walk[..len], None)? else {
                return Ok(None);
            };
            out.push(run);
        }
        Ok(Some(out))
    }

    /// One arm's prompt batches (module doc): the walk's first [`GEMV_ROWS`]
    /// ids, then the whole walk, each as a batch against the whole-card
    /// load's batch of the same ids (the same kernels: only the experts'
    /// placement differs, so the card arm's bits are the whole-card's and a
    /// host arm's differ by the host's rounding), and its last position
    /// against the arm's own decode steps (the batch's reductions sit in
    /// another order, so only the floor and the argmax band hold there).
    fn batch_near_whole(
        m: &mut Model,
        case: &Case,
        whole: &Whole,
        (arm, hosted): (&str, &[usize]),
    ) -> Result<bool, GateError> {
        const CLAUSE: &str = HOST_CLAUSE;
        let stats = |m: &Model| -> Result<_, GateError> {
            Ok(m.body(NAME)?
                .placed()
                .ok_or("a placed load's host tier")?
                .hybrid()
                .stats())
        };
        let rows = m.pass_rows()?;
        let mut pass = true;
        for (len, want) in [GEMV_ROWS, case.walk.len()].into_iter().zip(&whole.batch) {
            let prompt = &case.walk[..len];
            let Some(steps) = fresh(CLAUSE, "the prompt as steps", m, prompt)? else {
                pass = false;
                continue;
            };
            let (Some(&steps_id), Some(steps_logits)) = (steps.ids.last(), steps.logits.last())
            else {
                return Err("the steps' prompt is empty".into());
            };
            let before = stats(m)?;
            let Some(got) = batch_run(m, prompt, Some(want))? else {
                pass = false;
                continue;
            };
            let after = stats(m)?;
            let passes = len.div_ceil(rows) as u64;
            let served = after.batch_served - before.batch_served;
            let cols = after.batch_cols - before.batch_cols;
            let structure = served == hosted.len() as u64 * passes
                && cols == (hosted.len() * len) as u64
                && got.logits.iter().all(|v| v.is_finite());
            let (near, cos, dmax, same_id) =
                in_band((want.id, &want.logits), (got.id, &got.logits));
            let (vs_steps, steps_cos, ..) =
                in_band((steps_id, steps_logits), (got.id, &got.logits));
            let mut held = 0;
            let mut worst = 1.0f64;
            let mut exact = same_bits(&want.logits, &got.logits);
            for ((next_id, next), (id, logits)) in want.after.iter().zip(&got.after) {
                let (ok, c, ..) = in_band((*next_id, next), (*id, logits));
                held += usize::from(ok);
                worst = worst.min(c);
                exact &= same_bits(next, logits);
            }
            let arm_pass =
                structure && near && vs_steps && held == CONTINUE && (arm != "card" || exact);
            println!(
                "{CLAUSE}: arm {arm} batch of {len} ids ({passes} pass of up to {rows} rows, {}): \
                 batch_served {served}/{}, batch_cols {cols}/{}; last id {} vs the whole-card \
                 batch's {} (equal {same_id}), cosine {cos:.6}, max|Δ| {dmax:.3e}; {held}/{CONTINUE} \
                 steps after it in the band, worst cosine {worst:.6}, bit-identical {exact}; \
                 against the arm's steps: id {steps_id}, cosine {steps_cos:.6} {}",
                if len <= GEMV_ROWS {
                    "gemv arm"
                } else {
                    "GEMM arm"
                },
                hosted.len() as u64 * passes,
                hosted.len() * len,
                got.id,
                want.id,
                verdict(arm_pass)
            );
            pass &= arm_pass;
        }
        Ok(pass)
    }

    /// Whether `a` and `b` are the same batch run to the bit: the id and the
    /// logits after the prompt and after each step.
    fn same_batch(a: &BatchRun, b: &BatchRun) -> bool {
        a.id == b.id
            && same_bits(&a.logits, &b.logits)
            && a.after.len() == b.after.len()
            && a.after
                .iter()
                .zip(&b.after)
                .all(|((x, lx), (y, ly))| x == y && same_bits(lx, ly))
    }

    /// One arm's fault recovery (module doc): on each path (the walk as
    /// steps, the gemv arm's batch, the GEMM arm's batch) a call whose first
    /// id is past the embedding table raises the fault word, and after a
    /// reset the valid call that ran before it runs again clean with the same
    /// bits. A hosted layer's launches skip the slots the host serves and the
    /// quantize after them reads every slot's row, so a reset that left the
    /// faulted call's non-finite rows would fault the valid call after it.
    fn fault_recovery(m: &mut Model, case: &Case, arm: &str) -> Result<bool, GateError> {
        const CLAUSE: &str = HOST_CLAUSE;
        const BAD: u32 = u32::MAX;
        fn raised<T>(r: Result<T, GpuError>) -> bool {
            matches!(r, Err(GpuError::Fault { .. }))
        }
        // Per path: its name, whether the faulted call raised the fault word,
        // and whether the valid call after the reset is the one before it.
        let mut paths: Vec<(String, bool, bool)> = Vec::new();
        let before = fresh(CLAUSE, "the walk before a fault", m, &case.walk)?;
        m.reset()?;
        let hit = raised(m.step(&[BAD]));
        let after = fresh(CLAUSE, "the walk after a fault", m, &case.walk)?;
        let bits = matches!((&before, &after), (Some(b), Some(a)) if same(b, a));
        paths.push(("steps".into(), hit, bits));
        for len in [GEMV_ROWS, case.walk.len()] {
            let prompt = &case.walk[..len];
            let before = batch_run(m, prompt, None)?;
            let mut bad = prompt.to_vec();
            bad[0] = BAD;
            m.reset()?;
            let hit = raised(m.prefill(&bad));
            let after = batch_run(m, prompt, None)?;
            let bits = matches!((&before, &after), (Some(b), Some(a)) if same_batch(b, a));
            let name = if len <= GEMV_ROWS {
                format!("gemv batch of {len}")
            } else {
                format!("GEMM batch of {len}")
            };
            paths.push((name, hit, bits));
        }
        let pass = paths.iter().all(|&(_, hit, bits)| hit && bits);
        let listed: Vec<String> = paths
            .iter()
            .map(|(name, hit, bits)| format!("{name}: raised {hit}, then bit-identical {bits}"))
            .collect();
        println!(
            "{CLAUSE}: arm {arm} fault recovery (id {BAD} first, reset, the valid call again): {} {}",
            listed.join("; "),
            verdict(pass)
        );
        Ok(pass)
    }

    // --------------------------------------------------- watchdog_progress

    /// The walk's ids as one `GpuModel::step` call under the prompt call's
    /// busy mark (a prompt run as steps) move the engine watchdog's progress
    /// word at least once a token, so a long prompt run that way is not read
    /// as wedged. The mark's holder is the caller, so the step call adds no
    /// mark of its own and the word moves by the tokens' passes alone.
    fn watchdog_progress(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "watchdog_progress";
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let Some(_busy) = m.prompt_busy() else {
            return Err("the busy mark is held by a call of the model".into());
        };
        let before = m.engine_progress();
        if or_fault(CLAUSE, "the walk as steps", m.step(&case.walk))?.is_none() {
            return Ok(false);
        }
        let moved = m.engine_progress() - before;
        let pass = moved >= case.walk.len() as u64;
        println!(
            "{CLAUSE}: {} ids as one step call under the prompt busy mark moved the progress \
             word {moved} times, at least one a token {}",
            case.walk.len(),
            verdict(pass)
        );
        Ok(pass)
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
            "load resident_bytes={} ctx={ctx} pass_rows={} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.pass_rows()?,
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
        // The host-tier arms open loads of their own beside nothing: the
        // whole-card model is released first, its walk kept. A walk that
        // raised the fault word leaves the arms nothing to be held against,
        // and the clause fails.
        let host_clause = tier::run_clause(HOST_CLAUSE, Tag::SelfConsistency)?;
        let held = if host_clause {
            whole_card_run(&mut m, &case)?
        } else {
            None
        };
        drop(m);
        if host_clause {
            let pass = match &held {
                Some(whole) => host_tier_near_whole_card(&case, whole)?,
                None => false,
            };
            if !pass {
                println!("clause {HOST_CLAUSE} FAIL");
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
