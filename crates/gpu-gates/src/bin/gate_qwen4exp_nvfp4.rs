//! The Qwen3.8-Flash-Next NVFP4 gate: the split `prepare` writes from `nvidia/Qwen3.8-Flash-Next-NVFP4`
//! (routed experts NVFP4 with a tensor scale an expert, dense projections BF16 as the checkpoint
//! stores them or, under `BLOOMERY_PREPARE_Q8=1`, rewritten to Q8_0, gains F32, the PLE table in
//! its F8 rows) against the checkpoint's own forward (refset
//! `hf-qwen4exp-nvfp4`, `tools/ref/qwen4exp-nvfp4-oracle.py`: the `_bf16` sets for the stored
//! projections, the others for the Q8_0 ones; the head's type says which), by the e2e gate's bands
//! (`shared/q38_oracle.rs`). The oracle keeps indexer projections in BF16; the model loader
//! refuses Q8_0 indexer projections.
//! The plan is `--place a`'s on the one visible card, a DGX Spark's GB10,
//! whose budget holds every routed expert, so the plan leaves none on the host.
//!
//! What is asserted:
//! - (l) the load: the plan puts every routed expert of every layer on the card and none on the
//!   host, so each layer's step runs without the host leg, and the load opens.
//! - (c) free: the batch set's five ids by eager steps with the taps armed. Every route tap is the
//!   top ten of its own logits; each layer output is within `FREE_BAND` of the oracle's off every
//!   flip's path; every flip's pair error is within `FLIP_ERR_CAP` router-logit units; the last
//!   argmax is the oracle's.
//! - (p) the batch set as one pass (`Prompt38::Pass`: the batch port, no layer served on the host)
//!   and (g) as one ubatch walk (`Prompt38::Gemm`: the routed experts through the NVFP4 GEMM): each
//!   last argmax is the oracle's, the logits' distance printed.
//! - (t) D3K's step at position 3,000 after its prefill by ubatches: the argmax is the oracle's or a
//!   named tie (ours the oracle's runner-up, the oracle's margin at most twice our distance at the
//!   two ids, the logits within `FREE_BAND`). The layer outputs are printed: a flip in any of the
//!   3,000 positions before it may lie on their path.
//! - (s) the split: the plan of (l) under a card budget `SPLIT_HOST` short of the bytes (l) puts
//!   on the card, so each layer keeps an id prefix of its experts on the card and the rest on the
//!   host, where qdot's NVFP4 dot reads the split's blocks or the folder's rows where they lie.
//!   The load opens, and the batch set's eager steps hold (c)'s band, flips and last argmax
//!   against the oracle; the last logits' distance to the all-card run is printed. Mutant: the plan's refusal of an NVFP4 stack split between a card
//!   and the host, which refuses the load by name.
//! - (rs) the split under `mid` residency, each pinned count the fewest card experts a layer
//!   leaves past a spare and one that moves (a small churn pool), every layer's card set reseeded
//!   to its highest ids: the machine stages those host experts onto the card, an NVFP4 stack's
//!   codes, block scales and tensor scale each through its own transform, and every part of each
//!   is the bytes a static load uploads; the batch set's eager steps after it hold (c)'s band.
//!   Mutant: the codes of a folder's rows staged as stored, not in the card's block order.
//! - (rt) the load made for a route trace (`HostCfg::route_trace`): it opens, every layer keeps its
//!   host leg, which writes the trace, and the host run holds no view (no layer holds a host
//!   expert); the batch set's ids by eager steps as one prompt call fill a complete trace, every
//!   slot on the card, and each step's logits are the plain load's bit for bit.
//! - Recorded, not judged: the batch run and the D3K step against the other pair of sets, the Q8_0
//!   rounding's share of the distance.
//!
//! `--placement` runs only the saved-state and reader-sizing contract on a
//! checkpoint folder or split: ubatch 512, checkpoints enabled, no draft,
//! one short prompt and 16,384 ids. The measured save bytes must equal the
//! actual arrays and fit the planned bound; the row reserve must equal the
//! reader's columns, including both NVFP4 planes.
//!
//! The real tier loads the whole file on the card. `BLOOMERY_REF_MODEL` selects a split outside the
//! profile's model path, or the checkpoint folder itself, which the load reads as stored
//! (`qwen38_folder::open`, a build with the `checkpoint` feature).
//!
//! The fixture tier runs the same clauses on the `qwen4exp-nvfp4` profile's cut (four layers of the
//! checkpoint in their stored bits, `tools/fixture-cuts/qwen38nvfp4.json`) against the oracle's
//! sets of that cut (refset `fx-hf-qwen4exp-nvfp4`), each oracle clause the fixture-oracle clause
//! of its name `on the cut`. Every plan is made under the card budget the cut records
//! (`BLOOMERY_CARD_BUDGET`), which holds every routed expert, so the plan is the same on any card
//! of that budget; (s) falls short of (l)'s card bytes by an eighth of them instead of
//! `SPLIT_HOST`. Then the `qwen4exp-fp4l` profile's cut (the same layers of
//! `nvidia/Qwen3.8-Flash-Next-FP4-Local`, whose routed stacks keep their `csf` block scales and
//! whose dense projections are MXFP8) is loaded under the same budget: no oracle set holds its
//! forward, so (l), the route taps' self-consistency of (c) and (rt) run on it.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen4exp_nvfp4", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/e2e.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the compare, layer-table, flip-accounting and step-set helpers; the rest serve the other gates"
)]
mod e2e;

#[cfg(feature = "gpu")]
#[path = "shared/q38_oracle.rs"]
mod q38_oracle;

#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/folder.rs"]
#[allow(
    dead_code,
    reason = "the gate opens a Qwen3.8 folder alone: the checked open is what it reads"
)]
mod folder;

#[cfg(feature = "gpu")]
#[path = "shared/serve_seats/qwen38_folder.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the folder's parts, not its tokenizer"
)]
mod qwen38_folder;

#[cfg(feature = "gpu")]
#[path = "shared/qwen38_open.rs"]
#[allow(
    dead_code,
    reason = "the e2e gate prints and reads the plan's whole facts; this gate reads the counts"
)]
mod qwen38_open;

#[cfg(feature = "gpu")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "gpu")]
#[path = "shared/residency_clauses.rs"]
#[allow(
    dead_code,
    reason = "the residency clauses serve several gates; this one takes the seed lists and the transform check"
)]
mod residency_clauses;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::Path;
    use std::time::Instant;

    use crate::e2e::{
        argmax, flips_report, ik_last, layer_rels, layer_table, print_layers, rel, tie_numbers,
        worst_off_path,
    };
    use crate::q38_oracle::{FLIP_ERR_CAP, FREE_BAND, IkRoute, route_flips};
    use crate::qwen38_folder;
    use crate::qwen38_open::{Open38, open_split};
    use crate::residency_clauses;

    use bloomery_gpu::arch::qwen3moe::{
        Body38, Model38, Prompt38, Qwen38Model, RouteTap, TargetRows,
    };
    use bloomery_gpu::draft::{DraftFeed, DraftHead, Feed, Hidden, WalkMode};
    use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader, TraceSet};
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::kv_format::{KvChoice, KvQ8};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::flip;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, ref_model_path, verdict,
    };
    use bloomery_levers::{CardBudget, HostCfg};
    use gguf::GgmlType;
    use model::arch::models::headpick::HeadSet;
    use model::arch::qwen35moe::names;
    use model::arch::qwen35moe::place::{Experts, PlanInputs};
    use model::placement::{ModelTensors, PlanLevers, Role};
    use refset::arch::qwen4exp::nvfp4::{BATCH, BATCH_BF16, D3K, D3K_BF16, FX, HF, PROFILE};
    use runtime::Advance;

    /// Cache rows: D3K's step at position 3,000, with room.
    const CTX: usize = 3072;

    /// The gate's name: its first line and the `what` of its refusals.
    const GATE: &str = "gate_qwen4exp_nvfp4";

    /// The card bytes the real tier's (s) budget falls short of (l)'s plan by: a few of each
    /// layer's routed experts.
    const SPLIT_HOST: u64 = 8 << 30;

    /// The clauses judged against the oracle's sets: the real file's in the real tier, the cut's
    /// in the fixture tier ([`oracle_clause`]).
    const ORACLE: [&str; 6] = ["(c)", "(p)", "(g)", "(t)", "(s)", "(rs)"];

    /// The model profile of the FP4-Local cut (`tools/fixture-dirs.tsv`).
    const FP4L_PROFILE: &str = "qwen4exp-fp4l";

    /// What the clauses read of the file's header: the trunk's layers, a layer output's values
    /// (every stream of the hidden width), the routed experts and the slots a token takes, and
    /// the sets its dense form is judged by.
    struct Dims {
        n_layer: usize,
        row: usize,
        n_expert: usize,
        n_used: usize,
        dense: Dense,
    }

    /// The form of a split's dense matrices ([`dense_of`]) and the oracle's sets it is judged by,
    /// each beside the set of the other form, recorded.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Dense {
        /// The head's type: BF16 or Q8_0.
        ty: GgmlType,
        /// At least one indexer projection is Q8_0; the oracle's are all BF16.
        q8_indexer: bool,
        /// The batch set judged, then the one recorded.
        batch: (&'static str, &'static str),
        /// The D3K set judged, then the one recorded.
        d3k: (&'static str, &'static str),
    }

    /// Steps' last logits, and with the taps armed each step's layer outputs and route taps.
    struct Run {
        logits: Vec<Vec<f32>>,
        taps: Vec<Vec<f32>>,
        routes: Vec<Vec<RouteTap>>,
    }

    /// The set `name` of [`HF`] in the real tier, its twin on the cut (`fx_<name>` of [`FX`]) in
    /// the fixture tier, checked against its family.
    fn set(name: &str) -> Result<RefManifest, GateError> {
        let (name, family, path) = match Tier::from_env()? {
            Tier::Real => (name.to_string(), &HF, HF.path_in(&data_dir(), name)),
            Tier::Fixture => {
                let fx = format!("fx_{name}");
                let path = tier::fixture_set(&fx, &FX, &fx)?;
                (fx, &FX, path)
            }
        };
        let man = RefManifest::open(&path, family)?;
        refset::arch::qwen4exp::check_inputs(&man, &name)?;
        Ok(man)
    }

    /// The fixture-oracle name of the oracle clause `clause`.
    fn on_cut(clause: &str) -> String {
        format!("{clause} on the cut")
    }

    /// Whether this tier runs the oracle clause `clause`: in the real tier against the real
    /// file's sets, in the fixture tier as the fixture-oracle clause [`on_cut`].
    fn oracle_clause(clause: &str) -> Result<bool, GateError> {
        match Tier::from_env()? {
            Tier::Real => tier::run_clause(clause, Tag::Oracle),
            Tier::Fixture => tier::run_clause(&on_cut(clause), Tag::FixtureOracle),
        }
    }

    /// The levers every plan is made under: none in the real tier; in the fixture tier the card
    /// budget the cut records, which the tier exports as `BLOOMERY_CARD_BUDGET`.
    fn plan_levers(levers: &bloomery_levers::Levers) -> Result<PlanLevers, GateError> {
        match (Tier::from_env()?, levers.card_budget()) {
            (Tier::Real, _) => Ok(PlanLevers::default()),
            (Tier::Fixture, CardBudget::Every(b)) => Ok(PlanLevers {
                card_budget_bytes: CardBudget::Every(b),
            }),
            (Tier::Fixture, other) => Err(format!(
                "{GATE}: BLOOMERY_CARD_BUDGET is {other:?}; the fixture tier sets it to the one \
                 byte count the cut records"
            )
            .into()),
        }
    }

    /// The model at `path`: a split of architecture `qwen4exp` ([`open_split`]), or the checkpoint
    /// folder when `path` is a directory ([`qwen38_folder::open`]).
    fn model_of(path: &Path) -> Result<Model38, GateError> {
        if path.is_dir() {
            let f = qwen38_folder::open(path)?;
            return Ok((f.parts)(f.header).into());
        }
        open_split(path).map(Model38::from)
    }

    /// (l): the plan on the one visible card under `plan_levers`, every routed expert on it, and
    /// the load under `host`.
    fn open(
        path: &Path,
        d: &Dims,
        (levers, plan_levers): (&bloomery_levers::Levers, &PlanLevers),
        host: HostCfg,
    ) -> Result<(Qwen38Model, bool), GateError> {
        let file = model_of(path)?;
        let specs = Place::A.on_host()?.card_specs()?;
        let [card] = specs[..] else {
            return Err(format!("--place a resolved to {} cards, not one", specs.len()).into());
        };
        let t = Instant::now();
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX)?;
        if levers.draft() == Some("mtp") {
            let inputs = PlanInputs::describe(file.header())?;
            let draft = qwen38_folder::draft(path)?;
            let mtp = model::arch::qwen35moe::place::MtpInputs::read(
                &draft,
                file.header(),
                &inputs,
                HeadSet::Full,
            )?;
            let machine = model::arch::qwen35moe::place::machine_for_experts(
                card,
                inputs.stack(),
                ub as u64,
                Experts::Card,
            );
            let plan = inputs.plan_mtp_with_slots(
                &machine,
                (CTX as u64, u64::try_from(ub)?),
                plan_levers,
                &mtp,
                Experts::Card,
                1,
            )?;
            let ok = plan.plan.host.experts == 0
                && plan.plan.n_l.iter().all(|&n| n == d.n_expert as u64);
            println!(
                "(l) folder MTP draft_resident_bytes={} arena_bytes={} {}",
                plan.draft_resident_bytes(),
                plan.arena_bytes,
                verdict(ok)
            );
            let mut model =
                Body38::open_placed_mtp_slots(file, &plan, &inputs, 0, host, ub, &draft, &mtp, 1)?;
            model.set_mode(StepMode::Graph);
            return Ok((model, ok));
        }
        let planned = Open38 {
            card,
            ctx: CTX,
            ub,
            experts: Experts::Card,
            slots: 1,
            plan_levers,
            host,
            room: None,
            residency: Residency::Off,
            kv: KvQ8::F16,
        }
        .plan(file)?;
        let p = planned.facts();
        let short: Vec<usize> = (0..d.n_layer)
            .filter(|&l| p.n_l.get(l) != Some(&(d.n_expert as u64)))
            .collect();
        let ok = p.host_experts == 0 && short.is_empty();
        println!(
            "(l) plan card={} ctx_max={} ubatch={ub} host_experts={} card_experts={} layers short of \
             {} card experts: {short:?} {}",
            card.name,
            p.ctx_max,
            p.host_experts,
            p.card_experts,
            d.n_expert,
            verdict(ok)
        );
        let o = planned.open()?;
        println!(
            "(l) load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            o.model.resident_bytes(),
            o.model.layers().len(),
            t.elapsed().as_secs_f64()
        );
        Ok((o.model, ok))
    }

    /// `toks` by eager steps from where the model stands, the taps armed.
    fn run_steps(m: &mut Qwen38Model, toks: &[u32]) -> Result<Run, GateError> {
        m.set_mode(StepMode::Eager);
        m.set_layer_taps(true)?;
        let mut r = Run {
            logits: Vec::new(),
            taps: Vec::new(),
            routes: Vec::new(),
        };
        for &t in toks {
            m.step(&[t])?;
            r.logits.push(m.logits()?);
            let (gpu, _, b) = m.body_parts("run_steps")?;
            r.taps.push(b.taps(gpu)?);
            r.routes.push(b.route_taps(gpu)?);
        }
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        Ok(r)
    }

    /// The route taps of `r` that are not the top `n_used` of their own `n_expert` logits.
    fn inconsistent_routes(r: &Run, d: &Dims) -> usize {
        r.routes
            .iter()
            .flatten()
            .filter(|t| {
                t.logits.len() != d.n_expert || !flip::picks_its_top(&t.ids, &t.logits, d.n_used)
            })
            .count()
    }

    /// (c) on `man`: the route taps' self-consistency, the layer outputs and flips against the
    /// oracle's, and the last argmax; the verdict, or `true` with the lines printed only when
    /// `judged` is false.
    fn free(
        man: &RefManifest,
        r: &Run,
        d: &Dims,
        vocab: usize,
        judged: bool,
    ) -> Result<bool, GateError> {
        let arm = man.dir.display().to_string();
        let inconsistent = inconsistent_routes(r, d);
        let table = layer_table(man, &r.taps, d.row, d.n_layer)?;
        let routes = (0..d.n_layer)
            .map(|l| IkRoute::read(man, l, d.n_expert, d.n_used))
            .collect::<Result<Vec<_>, _>>()?;
        let flips = route_flips(&r.routes, &routes)?;
        for (l, row) in table.iter().enumerate() {
            let cells: Vec<String> = row
                .iter()
                .map(|e| e.map_or("-".to_string(), |e| format!("{e:.3e}")))
                .collect();
            println!(
                "free {arm} layer={l} l_out_rel by position {}",
                cells.join(" ")
            );
        }
        let flips_ok = flips_report(&flips, "free", FLIP_ERR_CAP);
        let (held, hl, ht, exempt) = worst_off_path(&table, &flips, false);
        let oracle = ik_last(man, vocab)?;
        let ours = r.logits.last().ok_or("no logits")?;
        let (top, oracle_top) = (argmax(ours), argmax(&oracle));
        let ok = held <= FREE_BAND && flips_ok && top == oracle_top && inconsistent == 0;
        println!(
            "free {arm}: {} tokens, {inconsistent} route taps not the top {} of their own logits, {} \
             flips ({} allowed), {exempt} outputs on a flip's path; worst l_out_rel off every flip's \
             path {held:.3e} at layer {hl} position {ht} (band {FREE_BAND:.2}); last argmax \
             ours={top} oracle={oracle_top} logits_rel={:.3e} (printed) {}",
            r.logits.len(),
            d.n_used,
            flips.len(),
            flips.iter().filter(|f| f.allowed(FLIP_ERR_CAP)).count(),
            rel(ours, &oracle),
            if judged {
                verdict(ok)
            } else {
                "(recorded, not judged)"
            }
        );
        Ok(ok || !judged)
    }

    fn logits_kl(oracle: &[f32], ours: &[f32]) -> Result<f64, GateError> {
        if oracle.is_empty() || oracle.len() != ours.len() {
            return Err("NVFP4 KL: empty or unequal logits rows".into());
        }
        for row in [oracle, ours] {
            if row.iter().any(|x| x.is_nan() || *x == f32::INFINITY)
                || !row.iter().any(|x| x.is_finite())
            {
                return Err("NVFP4 KL: invalid logits distribution".into());
            }
        }
        let p = bloomery_gpu_gates::prompts::log_softmax(oracle);
        let q = bloomery_gpu_gates::prompts::log_softmax(ours);
        Ok(p.iter()
            .zip(q)
            .filter(|(lp, _)| lp.is_finite())
            .map(|(lp, lq)| lp.exp() * (lp - lq))
            .sum())
    }

    /// (p) or (g), `tag`: the batch set by `path` (one pass or one ubatch walk, `how`) from a
    /// reset; its last argmax against the oracle's.
    fn batch_by(
        m: &mut Qwen38Model,
        toks: &[u32],
        vocab: usize,
        (tag, path, how): (&str, Prompt38, &str),
        man: &RefManifest,
    ) -> Result<bool, GateError> {
        m.reset()?;
        let top = m.prompt38(toks, path)?;
        let ours = m.logits()?;
        let oracle = ik_last(man, vocab)?;
        let ok = top == argmax(&oracle);
        println!(
            "({tag}) the batch set by {how}: last argmax ours={top} oracle={} logits_rel={:.3e} \
             kld={:.9} (printed) {}",
            argmax(&oracle),
            rel(&ours, &oracle),
            logits_kl(&oracle, &ours)?,
            verdict(ok)
        );
        Ok(ok)
    }

    /// (t): D3K's prefill by ubatches from a reset, then its step by an eager step with the taps
    /// armed, against the judged set `man` and, recorded, the other one.
    fn d3k_step(
        m: &mut Qwen38Model,
        d: &Dims,
        vocab: usize,
        (name, man): (&str, &RefManifest),
        (other, recorded): (&str, &RefManifest),
    ) -> Result<bool, GateError> {
        let (pos, step, prefill) = man.step()?;
        let tok = step[0];
        m.reset()?;
        let t = Instant::now();
        m.prompt38(prefill, Prompt38::Gemm)?;
        let fed = t.elapsed().as_secs_f64();
        let r = run_steps(m, &[tok])?;
        let ours = r.logits.last().ok_or("no logits")?;
        let mut ok = true;
        for (name, man, judged) in [(name, man, true), (other, recorded, false)] {
            let oracle = ik_last(man, vocab)?;
            let (top, oracle_top, oracle_2, margin, dist, logits_rel) = tie_numbers(ours, &oracle);
            let tie = top != oracle_top
                && top == oracle_2
                && margin <= 2.0 * dist
                && logits_rel <= FREE_BAND;
            let rels = layer_rels(man, &r.taps, d.row, d.n_layer)?;
            print_layers(name, &rels, 0, FREE_BAND);
            let step_ok = top == oracle_top || tie;
            let kl = logits_kl(&oracle, ours)?;
            println!(
                "(t) {name}: position {pos} after {} ids by ubatches ({fed:.1} s, runtime value); \
                 argmax ours={top} oracle={oracle_top} (its runner-up {oracle_2}, margin {margin:.4}, \
                 our distance at the two {dist:.4}{}); logits_rel {logits_rel:.3e}; kld={kl:.9}; worst \
                 l_out_rel {:.3e} (printed) {}",
                prefill.len(),
                if tie { ", a named tie" } else { "" },
                rels.iter().map(|r| r.0).fold(0.0, f64::max),
                if judged {
                    verdict(step_ok)
                } else {
                    "(recorded, not judged)"
                }
            );
            ok &= step_ok || !judged;
        }
        Ok(ok)
    }

    /// The dense form of the loaded dense projections of `model`: the head's
    /// type, BF16 as the checkpoint stores it (what `prepare` writes by default, judged by the
    /// `_bf16` sets) or Q8_0 (what `BLOOMERY_PREPARE_Q8=1` writes, or another converter, judged by
    /// the others). Every other BF16 or Q8_0 tensor but the indexer's projections and the router
    /// with its shared gate, which `prepare` keeps in BF16 beside either form, must be of the
    /// head's type: the sets hold one form or the other. This picks the reference sets only; the
    /// model loader refuses Q8_0 indexer projections.
    /// A head of another type or none, and a mixed split, are refused by name.
    fn dense_of(model: &ModelTensors) -> Result<Dense, String> {
        let tensors: Vec<(&str, GgmlType)> = model
            .tensors
            .iter()
            .filter(|t| t.role != Role::Unused)
            .filter(|t| (model.bf16_words)(t))
            .map(|t| (t.name.as_str(), t.ty))
            .collect();
        let head = names::output();
        let ty = match tensors.iter().find(|(n, _)| *n == head) {
            Some(&(_, ty @ (GgmlType::BF16 | GgmlType::Q8_0))) => ty,
            Some((_, ty)) => return Err(format!("{head} is {ty:?}, not BF16 or Q8_0")),
            None => return Err(format!("the file has no {head}")),
        };
        let indexer: Vec<String> = (0..model.layers)
            .flat_map(|l| [names::indexer_q_proj(l), names::indexer_k_proj(l)])
            .collect();
        let router: Vec<String> = (0..model.layers)
            .flat_map(|l| [names::ffn_gate_inp(l), names::ffn_gate_inp_shexp(l)])
            .collect();
        if let Some((name, other)) = tensors.iter().find(|(n, t)| {
            matches!(t, GgmlType::BF16 | GgmlType::Q8_0)
                && *t != ty
                && !indexer.iter().chain(&router).any(|i| i == n)
        }) {
            return Err(format!(
                "{name} is {other:?} beside a {ty:?} {head}, a dense form no pair of sets holds"
            ));
        }
        let q8_indexer = tensors
            .iter()
            .any(|(n, t)| *t == GgmlType::Q8_0 && indexer.iter().any(|i| i == n));
        Ok(if ty == GgmlType::BF16 {
            Dense {
                ty,
                q8_indexer,
                batch: (BATCH_BF16, BATCH),
                d3k: (D3K_BF16, D3K),
            }
        } else {
            Dense {
                ty,
                q8_indexer,
                batch: (BATCH, BATCH_BF16),
                d3k: (D3K, D3K_BF16),
            }
        })
    }

    /// The header's dims, the file's routed gate type, which must be NVFP4 (a split's blocks, or
    /// the rows a checkpoint folder stores), and its dense form ([`dense_of`]).
    fn dims(path: &Path) -> Result<Dims, GateError> {
        let model = model_of(path)?;
        let file = model.header();
        let inputs = PlanInputs::describe(file)?;
        let spec = &inputs.spec;
        let gate = file
            .find(&names::ffn_gate_exps(0))
            .map(|(_, t)| t.ty)
            .ok_or("the file has no routed gate stack on layer 0")?;
        if !matches!(gate, GgmlType::NVFP4 | GgmlType::NVFP4_ROWS) {
            return Err(format!(
                "{}: layer 0's routed gate is {gate:?}, not NVFP4",
                path.display()
            )
            .into());
        }
        let dense = dense_of(&inputs.model).map_err(|e| format!("{}: {e}", path.display()))?;
        let streams = spec.hc.map_or(1, |hc| hc.streams as usize);
        let routed = spec
            .routed()
            .ok_or("the description holds no routed layer")?;
        Ok(Dims {
            n_layer: spec.layers.len(),
            row: streams * spec.hidden as usize,
            n_expert: routed.experts as usize,
            n_used: routed.top_k as usize,
            dense,
        })
    }

    /// (rt): the load of `path` made for a route trace, the trace attached and `toks` run from a
    /// reset as one prompt call ([`run_steps`]), against `want`, the plain load's logits of the
    /// same steps. A load that fails is a named FAIL.
    fn traced(
        path: &Path,
        d: &Dims,
        levers: (&bloomery_levers::Levers, &PlanLevers),
        (toks, want): (&[u32], &[Vec<f32>]),
    ) -> Result<bool, GateError> {
        let host = HostCfg {
            route_trace: true,
            ..levers.0.host()
        };
        let mut m = match open(path, d, levers, host) {
            Ok((m, _)) => m,
            Err(e) => {
                println!(
                    "(rt) the load for a route trace FAILED: {e} {}",
                    verdict(false)
                );
                return Ok(false);
            }
        };
        let dir =
            std::env::temp_dir().join(format!("gate_qwen4exp_nvfp4-trace-{}", std::process::id()));
        m.reset()?;
        let (_, _, body) = m.body_parts("rt")?;
        let tier = body.hybrid_mut();
        let views = tier.host_mut().len();
        let layers = tier.slots().layers();
        let trace = RouteTrace::create(
            &dir,
            TraceHeader {
                model: path.to_path_buf(),
                arch: "qwen4exp".to_owned(),
                build: "gate_qwen4exp_nvfp4".to_owned(),
                n_expert: d.n_expert,
                n_used: d.n_used,
                first_layer: layers.start,
                n_layer: layers.len(),
                extra: Vec::new(),
            },
        )?;
        tier.attach_route_trace(trace)?;
        tier.route_prompt(0, toks.len())?;
        let same = run_steps(&mut m, toks)?
            .logits
            .iter()
            .flatten()
            .map(|v| v.to_bits())
            .eq(want.iter().flatten().map(|v| v.to_bits()));
        m.body_parts("rt")?
            .2
            .hybrid_mut()
            .take_route_trace()
            .ok_or("(rt): the route trace left the tier")?
            .finish()?;
        let set = TraceSet::read(&dir)?;
        std::fs::remove_dir_all(&dir)?;
        let on_card = set.kinds.iter().flatten().all(|&k| k == b'C');
        let ok = views == 0 && set.complete == Some(toks.len() as u64) && on_card && same;
        println!(
            "(rt) the load for a route trace: host views {views} (want 0), trace positions {:?} \
             (want {}), every slot on the card {on_card}, logits the plain load's bit for bit \
             {same} {}",
            set.complete,
            toks.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// (s): `toks`, the batch set `batch`'s ids, by eager steps on the split load, judged as (c)
    /// judges the all-card run, whose logits are `card`, then (rs) on the same split
    /// ([`reseeded`]). A load that fails is a named FAIL.
    fn split(
        path: &Path,
        d: &Dims,
        (levers, whole): (&bloomery_levers::Levers, &PlanLevers),
        (batch, toks, card): (&RefManifest, &[u32], &[Vec<f32>]),
        vocab: usize,
    ) -> Result<bool, GateError> {
        let specs = Place::A.on_host()?.card_specs()?;
        let [spec] = specs[..] else {
            return Err(format!("--place a resolved to {} cards, not one", specs.len()).into());
        };
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX)?;
        let mut open = Open38 {
            card: spec,
            ctx: CTX,
            ub,
            experts: Experts::Card,
            slots: 1,
            plan_levers: whole,
            host: levers.host(),
            room: None,
            residency: Residency::Off,
            kv: KvQ8::F16,
        };
        let card_bytes = open.plan(model_of(path)?)?.facts().card_bytes;
        let short = match Tier::from_env()? {
            Tier::Real => SPLIT_HOST,
            Tier::Fixture => card_bytes / 8,
        };
        let budget_bytes = (card_bytes + spec.margin()).saturating_sub(short);
        let budget = PlanLevers {
            card_budget_bytes: Some(budget_bytes).into(),
        };
        open.plan_levers = &budget;
        let planned = open.plan(model_of(path)?)?;
        let p = planned.facts();
        let split_layers = (0..d.n_layer)
            .filter(|&l| {
                p.n_l
                    .get(l)
                    .is_some_and(|&n| n > 0 && n < d.n_expert as u64)
            })
            .count();
        println!(
            "(s) plan card_budget={budget_bytes} host_experts={} card_experts={} layers split \
             between the card and the host: {split_layers}",
            p.host_experts, p.card_experts
        );
        let split_plan = p.host_experts > 0 && split_layers > 0;
        let n_l = p.n_l.clone();
        let mut m = match planned.open() {
            Ok(o) => o.model,
            Err(e) => {
                println!("(s) the split load FAILED: {e} {}", verdict(false));
                return Ok(false);
            }
        };
        m.reset()?;
        let r = run_steps(&mut m, toks)?;
        let held = free(batch, &r, d, vocab, true)?;
        let (ours, all) = (
            r.logits.last().ok_or("no split logits")?,
            card.last().ok_or("no all-card logits")?,
        );
        let ok = split_plan && held;
        println!(
            "(s) the split load: the plan splits {split_plan}, the oracle's band held {held}; last \
             argmax {} the all-card run's {}, logits_rel to it {:.3e} (printed) {}",
            argmax(ours),
            argmax(all),
            rel(ours, all),
            verdict(ok)
        );
        drop(m);
        let staged =
            !oracle_clause("(rs)")? || reseeded(open, path, &n_l, d, (batch, toks), vocab)?;
        Ok(ok && staged)
    }

    /// (rs): `open`, (s)'s split, under `mid` residency pinned at the fewest card experts a layer
    /// of `n_l` less a spare and one that moves, every layer's card set reseeded to its highest
    /// ids, then `toks` by eager steps judged as (s) judges them. A load that fails is a named
    /// FAIL.
    fn reseeded(
        open: Open38<'_>,
        path: &Path,
        n_l: &[u64],
        d: &Dims,
        (batch, toks): (&RefManifest, &[u32]),
        vocab: usize,
    ) -> Result<bool, GateError> {
        let fewest = n_l
            .iter()
            .copied()
            .filter(|&n| n > 0)
            .min()
            .ok_or("(rs) the split holds no card expert")?;
        let pinned = usize::try_from(fewest)?
            .checked_sub(2)
            .filter(|&p| p > 0)
            .ok_or("(rs) a card layer of fewer than three experts")?;
        let residency = Residency::Mid { pinned, spares: 1 };
        let mut m = match (Open38 { residency, ..open }).plan(model_of(path)?)?.open() {
            Ok(o) => o.model,
            Err(e) => {
                println!(
                    "(rs) the split load under mid-p{pinned}-s1 FAILED: {e} {}",
                    verdict(false)
                );
                return Ok(false);
            }
        };
        let start = {
            let b = m.body("reseeded")?;
            let machine = b.hybrid().swap().ok_or("(rs) the load runs no machine")?;
            residency_clauses::layer_seeds(machine, b.hybrid().slots().layers())?
        };
        let highest: Vec<u32> = (0..u32::try_from(d.n_expert)?).rev().collect();
        let rule = m.residency_rule().ok_or("(rs) the load runs no machine")?;
        let order = vec![highest.as_slice(); rule.layers()];
        m.residency_reseed(&order)?
            .ok_or("(rs) the load reseeded no machine")?;
        let (checked, bad) = {
            let b = m.body("reseeded")?;
            let machine = b.hybrid().swap().ok_or("(rs) the load runs no machine")?;
            let source = b
                .residency_source()
                .ok_or("(rs) the load has no residency source")?;
            residency_clauses::transform_check(m.gpu(), machine, source, &start)?
        };
        let staged = residency_clauses::transform_verdict(checked, &bad);
        m.reset()?;
        let r = run_steps(&mut m, toks)?;
        let held = free(batch, &r, d, vocab, true)?;
        let ok = staged && held;
        println!(
            "(rs) mid-p{pinned}-s1, each layer reseeded to its highest ids: {checked} host experts \
             staged onto the card, the oracle's band held {held} {}",
            verdict(ok)
        );
        Ok(ok)
    }

    fn mtp(mut m: Qwen38Model, prompt: &[u32]) -> Result<bool, GateError> {
        m.reset()?;
        m.prompt38(prompt, Prompt38::Pass)?;
        let hidden = m.target_streams(TargetRows::Pass, prompt.len())?;
        let width = hidden.len() / prompt.len();
        let mut ok = true;
        for rows in 1..=4 {
            let mut eager = None;
            for mode in [WalkMode::Eager, WalkMode::Graph, WalkMode::Graph] {
                m.poison_mtp_scratch()?;
                let out = m.mtp_draft(
                    DraftFeed::Rows(Feed {
                        tokens: &prompt[..rows],
                        pos0: 0,
                        hidden: Hidden::Host(&hidden[..rows * width]),
                        media: None,
                    }),
                    DraftHead::Full,
                    mode,
                )?;
                let state = m
                    .mtp_l_out()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>();
                let got = (
                    out.tokens,
                    out.p.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    state,
                );
                match &eager {
                    None => eager = Some(got),
                    Some(want) => ok &= *want == got,
                }
            }
            println!(
                "(mtp-g) rows={rows} poisoned scratch: eager = graph = replay {}",
                verdict(ok)
            );
        }
        m.reset()?;
        let mut plain = vec![m.prompt38(prompt, Prompt38::Auto)?];
        while plain.len() < 32 {
            plain.push(m.step(&[*plain.last().ok_or("no plain token")?])?);
        }
        m.reset()?;
        let mut session = app::Session::from_model(m, CTX as u32);
        let draft = app::mtp::MtpDraft::open(session.model(), Prompt38::Auto, StepMode::Graph)?;
        let mut spec =
            session.with_draft::<app::mtp::MtpDraft<Body38>, 4>(draft, &mut crate::quiet::Quiet)?;
        let first = spec.prompt(&mut session, prompt)?;
        spec.begin(&session, prompt, first)?;
        let mut drafted = vec![first];
        while drafted.len() < plain.len() {
            spec.pass(
                &mut session,
                *drafted.last().ok_or("no draft token")?,
                &mut drafted,
            )?;
        }
        let same = drafted[..plain.len()] == plain;
        println!(
            "(mtp-v) greedy target oracle: {} tokens plain = verified {}",
            plain.len(),
            verdict(same)
        );
        Ok(ok && same)
    }

    fn placement(path: &Path, levers: &bloomery_levers::Levers) -> Result<(), GateError> {
        use bloomery_gpu::arch::qwen3moe::{checkpoint_bytes, seq_positional_bytes};
        use model::placement::{RowTableSource, row_room};
        const LONG: usize = 16_384;
        const UB: usize = 512;
        let file = model_of(path)?;
        let kv = KvChoice::choose(None, None)?.kv;
        let mut inputs = PlanInputs::describe(file.header())?;
        inputs.kv = inputs.kv.in_q8_if(kv == KvQ8::Q8);
        let gdn = inputs
            .spec
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, model::arch::models::Mixer::DeltaRule(_)))
            .count();
        let qsa = inputs.spec.layers.len() - gdn;
        let short_bound = inputs.seq_state_bytes(1, None);
        let long_bound = inputs.seq_state_bytes(LONG as u64, None);
        let specs = Place::A.on_host()?.card_specs()?;
        let [card] = specs[..] else {
            return Err("placement needs one visible card".into());
        };
        let plan_levers = PlanLevers::default();
        let planned = Open38 {
            card,
            ctx: LONG + 1,
            ub: UB,
            experts: Experts::Card,
            slots: 1,
            plan_levers: &plan_levers,
            host: levers.host(),
            room: None,
            residency: Residency::Off,
            kv,
        }
        .plan(file)?;
        let page = os::page_bytes()?;
        let want = if planned.facts().ple_tier == Some(model::placement::Device::Nvme) {
            row_room(
                &inputs.model,
                runtime::stores::reader_columns(UB as u64),
                page,
                inputs.row_source,
            )?
        } else {
            0
        };
        let reserve = planned.facts().row_reserve_bytes;
        let parts = inputs.row_source == RowTableSource::Parts;
        println!(
            "(placement) ubatch={UB} reader_columns={} parts={parts} q8={} row_reserve_bytes={reserve} expected={want} {}",
            runtime::stores::reader_columns(UB as u64),
            kv == KvQ8::Q8,
            verdict(reserve == want)
        );
        let mut ok = reserve == want;
        let mut m = planned.open()?.model;
        m.body_parts("placement")?.2.set_checkpoints(true);
        for n in [1, LONG] {
            m.reset()?;
            m.prompt38(&vec![1; n], Prompt38::Gemm)?;
            let bytes = m.seq_save_bytes()?;
            let saved = m.seq_save()?;
            let copies = if n == 1 { 1 } else { 4 };
            let expected = seq_positional_bytes(qsa, n, kv) + copies * checkpoint_bytes(gdn, qsa);
            let bound = if n == 1 { short_bound } else { long_bound };
            let good = bytes == expected && saved.bytes() as u64 == bytes && bytes <= bound;
            println!(
                "(placement) positions={n} checkpoint_copies={copies} seq_save_bytes={bytes} arrays_bytes={} planned_bound={bound} expected={expected} {}",
                saved.bytes(),
                verdict(good)
            );
            ok &= good;
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    pub fn run() -> Result<(), GateError> {
        let tier = Tier::from_env()?;
        let levers = bloomery_levers::at_main(&tier::acts_in(
            tier,
            &[
                bloomery_levers::DRAFT,
                bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
            ],
        ))?;
        if let Some(word) = levers.draft()
            && !matches!(word, "off" | "mtp")
        {
            return Err(
                format!("BLOOMERY_DRAFT={word:?}: this gate supports only off or mtp").into(),
            );
        }
        let path = match tier {
            Tier::Real => ref_model_path()?,
            Tier::Fixture => tier::cut(PROFILE)?,
        };
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args == ["--placement"] {
            return placement(&path, &levers);
        }
        if !args.is_empty() {
            return Err("expected no arguments or --placement".into());
        }
        let plan = plan_levers(&levers)?;
        println!("model {}", path.display());
        let d = dims(&path)?;
        if levers.draft() == Some("mtp") {
            let batch = set(d.dense.batch.0)?;
            let (_, toks, _) = batch.step()?;
            let (m, mut ok) = open(&path, &d, (&levers, &plan), levers.host())?;
            ok &= mtp(m, toks)?;
            return if ok { Ok(()) } else { Err(checks_failed()) };
        }
        let Dense {
            ty,
            q8_indexer,
            batch: (batch_name, other_batch),
            d3k: (d3k_name, other_d3k),
        } = d.dense;
        println!(
            "dense projections {}: judged by {batch_name} and {d3k_name}; indexer {}, the oracle's BF16",
            if ty == GgmlType::BF16 {
                "BF16 as stored"
            } else {
                "Q8_0"
            },
            if q8_indexer { "includes Q8_0" } else { "BF16" }
        );
        let batch = set(batch_name)?;
        let batch_other = set(other_batch)?;
        let d3k = set(d3k_name)?;
        let d3k_other = set(other_d3k)?;
        for (judged, other) in [(&batch, &batch_other), (&d3k, &d3k_other)] {
            if judged.header.tokens != other.header.tokens {
                return Err(format!(
                    "{}: oracle token ids differ from {}",
                    other.dir.display(),
                    judged.dir.display()
                )
                .into());
            }
        }
        let (_, toks, _) = batch.step()?;
        tier::sc("(l)")?;
        let (mut m, mut ok) = open(&path, &d, (&levers, &plan), levers.host())?;
        let vocab = m.body("run")?.vocab();
        m.reset()?;
        let r = run_steps(&mut m, toks)?;
        if oracle_clause("(c)")? {
            ok &= free(&batch, &r, &d, vocab, true)?;
            ok &= free(&batch_other, &r, &d, vocab, false)?;
        }
        if oracle_clause("(p)")? {
            ok &= batch_by(
                &mut m,
                toks,
                vocab,
                ("p", Prompt38::Pass, "one pass"),
                &batch,
            )?;
        }
        if oracle_clause("(g)")? {
            ok &= batch_by(
                &mut m,
                toks,
                vocab,
                ("g", Prompt38::Gemm, "one ubatch"),
                &batch,
            )?;
        }
        if oracle_clause("(t)")? {
            ok &= d3k_step(&mut m, &d, vocab, (d3k_name, &d3k), (other_d3k, &d3k_other))?;
        }
        drop(m);
        if oracle_clause("(s)")? {
            ok &= split(
                &path,
                &d,
                (&levers, &plan),
                (&batch, toks, &r.logits),
                vocab,
            )?;
        }
        tier::sc("(rt)")?;
        ok &= traced(&path, &d, (&levers, &plan), (toks, &r.logits))?;
        match tier {
            Tier::Real => {
                for clause in ORACLE {
                    tier::run_clause(&on_cut(clause), Tag::FixtureOracle)?;
                }
            }
            Tier::Fixture => ok &= fp4_local((&levers, &plan), toks)?,
        }
        tier::expect_fixture_oracle(GATE, ORACLE.len())?;
        println!("{}", tier::tally_line());
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    /// The FP4-Local cut under `levers`' plan, which holds every routed expert on the card (the
    /// host does not read a `csf` block scale): (l), `toks` by eager steps with every route tap
    /// the top of its own logits and every logit finite, and (rt) against those steps.
    fn fp4_local(
        levers: (&bloomery_levers::Levers, &PlanLevers),
        toks: &[u32],
    ) -> Result<bool, GateError> {
        let path = tier::cut(FP4L_PROFILE)?;
        println!("FP4-Local cut {}", path.display());
        let d = dims(&path)?;
        tier::sc("(l) FP4-Local")?;
        let (mut m, mut ok) = open(&path, &d, levers, levers.0.host())?;
        m.reset()?;
        let r = run_steps(&mut m, toks)?;
        drop(m);
        tier::sc("(c) FP4-Local route taps")?;
        let inconsistent = inconsistent_routes(&r, &d);
        let finite = r.logits.iter().flatten().all(|v| v.is_finite());
        let held = inconsistent == 0 && finite;
        println!(
            "(c) FP4-Local: {} steps, route taps not the top {} of their own logits {inconsistent}, \
             every logit finite {finite} {}",
            toks.len(),
            d.n_used,
            verdict(held)
        );
        ok &= held;
        tier::sc("(rt) FP4-Local")?;
        ok &= traced(&path, &d, levers, (toks, &r.logits))?;
        Ok(ok)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn logits_kl_compares_distributions_and_refuses_invalid_rows() {
            let p = [0.0, 1.0, f32::NEG_INFINITY];
            assert!(logits_kl(&p, &[4.0, 5.0, f32::NEG_INFINITY]).unwrap().abs() < 1e-12);
            assert!(logits_kl(&p, &[1.0, 0.0, f32::NEG_INFINITY]).unwrap() > 0.4);
            for q in [
                &[][..],
                &[0.0][..],
                &[0.0, f32::NAN, 0.0][..],
                &[f32::NEG_INFINITY; 3][..],
            ] {
                assert!(logits_kl(&p, q).is_err());
            }
        }

        /// The sets follow the split's dense form: `prepare`'s default split (every dense matrix
        /// BF16) the `_bf16` pair; its `BLOOMERY_PREPARE_Q8=1` split (Q8_0, the indexer's
        /// projections BF16) and a Q8_0 split another converter wrote (the indexer Q8_0 too) the
        /// Q8_0 pair, the router BF16 beside either; a projection of the other form, a head of a
        /// third type and no head are refused by name. Mutants: the form read as one type
        /// whatever the head, the pairs swapped, the mixed check dropped, the indexer or the
        /// router not exempt.
        #[test]
        fn a_split_is_judged_by_the_sets_of_its_dense_form() {
            let split = |dense, indexer| -> Vec<(String, GgmlType)> {
                vec![
                    (names::output(), dense),
                    (names::token_embd(), dense),
                    (names::ssm_out(0), dense),
                    (names::ffn_down_shexp(0), dense),
                    (names::ffn_gate_exps(0), GgmlType::NVFP4),
                    (names::attn_q_norm(3), GgmlType::F32),
                    (names::attn_output(3), dense),
                    (names::indexer_q_proj(3), indexer),
                    (names::indexer_k_proj(3), indexer),
                ]
            };
            let of = |t: &[(String, GgmlType)]| {
                use model::placement::ModelTensor;
                dense_of(&ModelTensors {
                    tensors: t
                        .iter()
                        .map(|(n, ty)| ModelTensor {
                            name: n.clone(),
                            shard: 0,
                            layer: None,
                            role: if n.starts_with("blk.4.") {
                                Role::Unused
                            } else {
                                Role::Head
                            },
                            ty: *ty,
                            dims: vec![32, 32],
                            file_bytes: 0,
                            gathered_rows: None,
                        })
                        .collect(),
                    layers: 4,
                    experts: 1,
                    experts_used: 1,
                    bf16_words: |_| true,
                })
            };
            let (bf16, q8) = (GgmlType::BF16, GgmlType::Q8_0);
            let stored = Dense {
                ty: bf16,
                q8_indexer: false,
                batch: (BATCH_BF16, BATCH),
                d3k: (D3K_BF16, D3K),
            };
            let rewritten = Dense {
                ty: q8,
                q8_indexer: false,
                batch: (BATCH, BATCH_BF16),
                d3k: (D3K, D3K_BF16),
            };
            let mut unused = split(q8, bf16);
            unused.push((names::indexer_q_proj(4), bf16));
            unused.push((names::attn_output(4), bf16));
            assert_eq!(of(&unused), Ok(rewritten));
            let mut values = split(q8, bf16);
            values.push((names::ffn_gate_inp(0), bf16));
            values.push((names::ffn_gate_inp_shexp(0), bf16));
            assert_eq!(of(&values), Ok(rewritten));
            assert_eq!(of(&split(bf16, bf16)), Ok(stored));
            assert_eq!(of(&split(q8, bf16)), Ok(rewritten));
            for (ty, expected) in [(bf16, stored), (q8, rewritten)] {
                assert_eq!(
                    of(&split(ty, q8)),
                    Ok(Dense {
                        q8_indexer: true,
                        ..expected
                    })
                );
                let mut mixed = split(ty, bf16);
                mixed[7].1 = q8;
                assert_eq!(
                    of(&mixed),
                    Ok(Dense {
                        q8_indexer: true,
                        ..expected
                    })
                );
            }
            for (at, ty, named) in [
                (3, bf16, names::ffn_down_shexp(0)),
                (6, q8, names::attn_output(3)),
                (0, GgmlType::Q6_K, format!("{} is Q6_K", names::output())),
            ] {
                let mut t = split(if ty == q8 { bf16 } else { q8 }, bf16);
                t[at].1 = ty;
                let e = of(&t).unwrap_err();
                assert!(e.contains(&named), "{e}");
            }
            let e = of(&split(q8, bf16)[1..]).unwrap_err();
            assert!(e.contains(&names::output()), "{e}");
        }
    }
}
