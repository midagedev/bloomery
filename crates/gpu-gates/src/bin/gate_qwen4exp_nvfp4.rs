//! The Qwen3.8-Flash-Next NVFP4 gate: the split `prepare` writes from `nvidia/Qwen3.8-Flash-Next-NVFP4`
//! (routed experts NVFP4 with a tensor scale an expert, dense projections BF16 as the checkpoint
//! stores them or, under `BLOOMERY_PREPARE_Q8=1`, rewritten to Q8_0, gains F32, the PLE table in
//! its F8 rows) against the checkpoint's own forward (refset
//! `hf-qwen4exp-nvfp4`, `tools/ref/qwen4exp-nvfp4-oracle.py`: the `_bf16` sets for the stored
//! projections, the others for the Q8_0 ones; the head's type says which), by the e2e gate's bands
//! (`shared/q38_oracle.rs`). The oracle keeps indexer projections in BF16; the model loader
//! refuses Q8_0 indexer projections.
//! The plan is `--place a`'s on the one visible card, a DGX Spark's GB10,
//! whose budget holds every routed expert: no CPU kernel reads an NVFP4 stack.
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
//! - (rt) the load made for a route trace (`HostCfg::route_trace`): it opens, every layer keeps its
//!   host leg, which writes the trace, and the host run holds no view (no CPU kernel reads an NVFP4
//!   stack); the batch set's ids by eager steps as one prompt call fill a complete trace, every
//!   slot on the card, and each step's logits are the plain load's bit for bit.
//! - Recorded, not judged: the batch run and the D3K step against the other pair of sets, the Q8_0
//!   rounding's share of the distance.
//!
//! Real-only: the fixture holds no NVFP4 stack. Loads the whole file on the card.
//! `BLOOMERY_REF_MODEL` selects a split outside the recipe's model folder, or the checkpoint folder
//! itself, which the load reads as stored (`qwen38_folder::open`, a build with the `checkpoint` feature).

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

    use bloomery_gpu::arch::qwen3moe::{
        Body38, Model38, Prompt38, Qwen38Model, RouteTap, TargetRows,
    };
    use bloomery_gpu::draft::{DraftFeed, DraftHead, Feed, Hidden, WalkMode};
    use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader, TraceSet};
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::flip;
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, ref_model_path, verdict,
    };
    use bloomery_levers::HostCfg;
    use gguf::GgmlType;
    use model::arch::models::headpick::HeadSet;
    use model::arch::qwen35moe::names;
    use model::arch::qwen35moe::place::{Experts, PlanInputs};
    use model::placement::{ModelTensors, PlanLevers, Role};
    use refset::arch::qwen4exp::nvfp4::{BATCH, BATCH_BF16, D3K, D3K_BF16, HF};
    use runtime::Advance;

    /// Cache rows: D3K's step at position 3,000, with room.
    const CTX: usize = 3072;

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

    /// The set `name` of the family, checked against it.
    fn set(name: &str) -> Result<RefManifest, GateError> {
        let man = RefManifest::open(&HF.path_in(&data_dir(), name), &HF)?;
        refset::arch::qwen4exp::check_inputs(&man, name)?;
        Ok(man)
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

    /// (l): the plan on the one visible card, every routed expert on it, and the load under
    /// `host`.
    fn open(
        path: &Path,
        d: &Dims,
        levers: &bloomery_levers::Levers,
        host: HostCfg,
    ) -> Result<(Qwen38Model, bool), GateError> {
        let file = model_of(path)?;
        let specs = Place::A.on_host()?.card_specs()?;
        let [card] = specs[..] else {
            return Err(format!("--place a resolved to {} cards, not one", specs.len()).into());
        };
        let t = Instant::now();
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX)?;
        let plan_levers = PlanLevers::default();
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
                inputs.spec.layers.len(),
                ub as u64,
                Experts::Card,
            );
            let plan = inputs.plan_mtp_with_slots(
                &machine,
                CTX as u64,
                &plan_levers,
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
            plan_levers: &plan_levers,
            host,
            room: None,
            residency: Residency::Off,
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
        let inconsistent = r
            .routes
            .iter()
            .flatten()
            .filter(|t| {
                t.logits.len() != d.n_expert || !flip::picks_its_top(&t.ids, &t.logits, d.n_used)
            })
            .count();
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

    /// The header's dims, the file's routed gate type, which must be NVFP4, and its dense form
    /// ([`dense_of`]).
    fn dims(path: &Path) -> Result<Dims, GateError> {
        let model = model_of(path)?;
        let file = model.header();
        let inputs = PlanInputs::describe(file)?;
        let spec = &inputs.spec;
        let gate = file
            .find(&names::ffn_gate_exps(0))
            .map(|(_, t)| t.ty)
            .ok_or("the file has no routed gate stack on layer 0")?;
        if gate != GgmlType::NVFP4 {
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
        levers: &bloomery_levers::Levers,
        (toks, want): (&[u32], &[Vec<f32>]),
    ) -> Result<bool, GateError> {
        let host = HostCfg {
            route_trace: true,
            ..levers.host()
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

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            bloomery_levers::DRAFT,
            bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
        ])?;
        if let Some(word) = levers.draft()
            && !matches!(word, "off" | "mtp")
        {
            return Err(
                format!("BLOOMERY_DRAFT={word:?}: this gate supports only off or mtp").into(),
            );
        }
        let path = ref_model_path()?;
        println!("model {}", path.display());
        let d = dims(&path)?;
        if levers.draft() == Some("mtp") {
            let batch = set(d.dense.batch.0)?;
            let (_, toks, _) = batch.step()?;
            let (m, mut ok) = open(&path, &d, &levers, levers.host())?;
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
        let (mut m, mut ok) = open(&path, &d, &levers, levers.host())?;
        let vocab = m.body("run")?.vocab();
        m.reset()?;
        let r = run_steps(&mut m, toks)?;
        ok &= free(&batch, &r, &d, vocab, true)?;
        ok &= free(&batch_other, &r, &d, vocab, false)?;
        ok &= batch_by(
            &mut m,
            toks,
            vocab,
            ("p", Prompt38::Pass, "one pass"),
            &batch,
        )?;
        ok &= batch_by(
            &mut m,
            toks,
            vocab,
            ("g", Prompt38::Gemm, "one ubatch"),
            &batch,
        )?;
        ok &= d3k_step(&mut m, &d, vocab, (d3k_name, &d3k), (other_d3k, &d3k_other))?;
        drop(m);
        ok &= traced(&path, &d, &levers, (toks, &r.logits))?;
        if ok { Ok(()) } else { Err(checks_failed()) }
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
