//! The nemotron_h end-to-end gate: the whole decode chain — Mamba-2 layers,
//! grouped-query attention with no rope, squared-ReLU MLP layers, the head
//! and the argmax — on one card, over the Nemotron 3 Nano 4B file.
//!
//! The clauses are a list ([`CLAUSES`]); each prints one line ending in PASS
//! or FAIL, and a step that raises the card's fault word is a FAIL naming the
//! fault, never a pass:
//!
//! - `structure`: the captured step holds the entries the layer kinds derive
//!   — one Mamba-2 scan a Mamba layer, one squared ReLU an MLP layer, the
//!   group-5 flash pass and its merge an attention layer, two appends a key
//!   head an attention layer, one Q5_0 embedding row — and no entry of the
//!   group-8 flash pass; no host node.
//! - `eager_equals_replay`: eager and graph replay give bit-identical logits
//!   over the 16-token prompt, all finite.
//! - `reset_rewinds`: after `reset` the same prompt gives the same ids and the
//!   same logits bits.
//! - `state_carries`: after the prompt the last token stepped from the same
//!   reset twice gives the same ids; layer 0's scan state is finite and not
//!   all zero; and the last token's logits after the prompt differ from the
//!   same token's alone from a reset (a body that never carried state would
//!   give equal ones).
//! - `fault_word_clean`: the fault word is clean after every run.
//!
//! - `top1_is_mainlines`: for each prompt of [`STEP_PROMPTS`] in the
//!   `ref-nemotron-logits` set, the body's argmax at the last position is the
//!   set's.
//! - `logits_near_mainlines`: per prompt of [`STEP_PROMPTS`] the max |Δ| and
//!   the cosine of the last-position logits against the set's are printed.
//!   The floor is cosine at least [`COSINE_FLOOR`] and the same set of top-5
//!   ids; a prompt below
//!   it is a FAIL, never a pin. The order inside the top-5 is printed with
//!   both sides' logits and not gated: two logits closer than the quantised
//!   file's own noise swap places, and the order of rank 1 is the other
//!   clause.
//!
//! The prompt path (`GpuModel::prefill`, passes of many rows) is held to
//! mainline, the reference independent of both paths. It is not held to the
//! step path's bits: at a Q8_0 site the step's gemv reads the f32 rows and a
//! pass's GEMM reads them quantized to q8 blocks of 32 (as mainline's own
//! prompt path does), so the two paths differ by that rounding, and the step
//! path's distance is printed beside the gated lines.
//!
//! - `prefill_is_mainlines`: every prompt of the set through `prefill`: the
//!   argmax is mainline's and the cosine at least [`COSINE_FLOOR`]. The top-5
//!   set is printed, not gated: its rank-5 boundary is a near tie on prose64
//!   and prose257, and the step clauses keep their top-5 gate on
//!   [`STEP_PROMPTS`].
//! - `prefill_then_step_is_mainlines`: per prompt of two ids or more, all but
//!   the last id through `prefill`, then the last as one step: its logits are
//!   mainline's last-position logits under the same gate (the argmax and the
//!   cosine floor). This holds the scan states, the conv rings and the K/V
//!   rows a pass leaves to the step after it. prose1 (one id) has no pass and
//!   is skipped by name. Then [`STEPS_AFTER`] greedy steps from it and from
//!   the step path are printed with their largest |Δ|, ungated.
//! - `prefill_edges`: passes of [`EDGE_ROWS`] rows over prose255, prose256,
//!   prose257 and prose512 (a pass short of, at and past its rows, and two
//!   whole passes): each within `B` of the step path, `B` the step path's own
//!   max |Δ| against mainline's logits for that prompt, the same argmax past a
//!   near tie (a step-path margin at most `2B`).
//! - `prefill_at_a_small_context`: a model loaded at [`SMALL_CTX`] rows, under
//!   [`PASS_ROWS`], takes `prefill` of [`SMALL_IDS`] ids in passes its prompt
//!   buffers hold, and its logits are the gate's model's bits on the same ids.
//!
//! The own-contract clauses run over a fixed walk of token ids; the mainline
//! clauses run over the set's prompts, read through `crates/refset`. A set
//! absent from the box, dumped from another file or build, or unfinished ends
//! the gate by name before the card loads.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nemotron_e2e", gate::run())
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
    use crate::e2e::{Run, feed, finite, fresh, histogram, or_fault, same, top_ids, walk};
    use bloomery_gpu::arch::nemotron_h::PASS_ROWS;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, NemotronHModel, NodeInfo};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, open_split, verdict};
    use cuda_core::sys;
    use model::arch::Arch;
    use model::arch::nemotron_h::hparams::{Hparams, Kind};
    use refset::arch::nemotron_h::LOGITS;
    use refset::nemotron::LogitsSet;
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    type Model = NemotronHModel;

    /// Rows of the K/V planes a key head holds: the longest prompt of the
    /// reference set (code4096) with room for the clauses' extra steps.
    const CTX: usize = 4_160;

    /// Greedy steps `prefill_then_step_is_mainlines` prints after the gated
    /// step.
    const STEPS_AFTER: usize = 16;

    /// The rows of `prefill_edges`' passes: the set's prose255, prose256 and
    /// prose257 fall short of, on and past one pass's end.
    const EDGE_ROWS: usize = 256;

    /// `prefill_at_a_small_context`'s context, under [`PASS_ROWS`], and the
    /// prompt it reads: the first [`SMALL_IDS`] ids of prose255.
    const SMALL_CTX: usize = 256;
    const SMALL_IDS: usize = 200;

    /// The cosine the last-position logits keep against mainline's.
    const COSINE_FLOOR: f64 = 0.999;

    /// How many of the highest logits must be the same ids.
    const TOP_K: usize = 5;

    /// The set's prompts the step path's mainline clauses read
    /// (`top1_is_mainlines`, `logits_near_mainlines`); the prompt path's
    /// clauses read every prompt. The others hold the step path to floors no
    /// derived rounding band backs yet (prose1 and prose257 near ties,
    /// prose256's cosine), an open finding of the plan.
    const STEP_PROMPTS: &[&str] = &["code5", "code29", "prose64", "prose512"];

    /// The set's prompts `STEP_PROMPTS` names, each present.
    fn step_prompts(case: &Case) -> Result<Vec<&refset::nemotron::Prompt>, GateError> {
        STEP_PROMPTS
            .iter()
            .map(|n| {
                case.set
                    .prompts
                    .iter()
                    .find(|p| p.name == *n)
                    .ok_or_else(|| format!("the set holds no prompt {n}").into())
            })
            .collect()
    }

    /// Tokens of the prompt every clause runs.
    const PROMPT_LEN: usize = 16;

    /// What a clause is run over: the layer kinds the file's header names and
    /// the prompts, in order.
    struct Case {
        hp: Hparams,
        prompts: Vec<Prompt>,
        set: LogitsSet,
        /// The step path's argmax and last-position logits per set prompt,
        /// run once and read by every clause after the first.
        steps: RefCell<BTreeMap<String, Last>>,
    }

    struct Prompt {
        name: &'static str,
        ids: Vec<u32>,
    }

    /// One clause: its name and its check.
    type Clause = fn(&mut Model, &Case) -> Result<bool, GateError>;

    /// The clauses, in the order they run.
    const CLAUSES: &[(&str, Clause)] = &[
        ("structure", structure),
        ("eager_equals_replay", eager_equals_replay),
        ("reset_rewinds", reset_rewinds),
        ("state_carries", state_carries),
        ("top1_is_mainlines", top1_is_mainlines),
        ("logits_near_mainlines", logits_near_mainlines),
        ("prefill_is_mainlines", prefill_is_mainlines),
        (
            "prefill_then_step_is_mainlines",
            prefill_then_step_is_mainlines,
        ),
        ("prefill_edges", prefill_edges),
        ("prefill_at_a_small_context", prefill_at_a_small_context),
        ("fault_word_clean", fault_word_clean),
    ];

    /// A run's argmax after the prompt and its last-position logits.
    type Last = (u32, Vec<f32>);

    fn structure(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        let nodes = m.capture_step()?;
        let list = m.step_graph_nodes()?;
        let hist = histogram(&list);
        let ([host], _) = count_kinds(&list, [sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST]);
        let kinds = |k: Kind| case.hp.kinds.iter().filter(|&&x| x == k).count();
        let (mamba, attn, mlp) = (kinds(Kind::Mamba), kinds(Kind::Attention), kinds(Kind::Mlp));
        let n_kv = case.hp.kv.iter().copied().max().unwrap_or(0);
        let got = |name: &str| hist.get(name).copied().unwrap_or(0);
        let want = [
            ("mamba2_step", mamba),
            ("relu_sqr", mlp),
            ("gqa_flash_seg_g5", attn),
            ("gqa_flash_merge", attn),
            ("kv_append_pos_buf", 2 * n_kv * attn),
            ("embed_rows_q5_0", 1),
        ];
        let mut pass = host == 0;
        for (name, n) in want {
            pass &= got(name) == n;
        }
        // The group-8 pass runs the other segment entries.
        let group8: usize = hist
            .iter()
            .filter(|(k, _)| k.starts_with("gqa_flash_seg") && !k.ends_with("_g5"))
            .map(|(_, n)| n)
            .sum();
        pass &= group8 == 0;
        let listed: Vec<String> = want
            .iter()
            .map(|(name, n)| format!("{name}={}/{n}", got(name)))
            .collect();
        println!(
            "structure: layers mamba={mamba} attn={attn} mlp={mlp}, graph_nodes={nodes} host={host}, \
             entries (got/want) {} group8_entries={group8} (want 0) {}",
            listed.join(" "),
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
        let mut ok = true;
        for p in &case.prompts {
            m.set_mode(StepMode::Eager);
            let eager = fresh(CLAUSE, "eager", m, &p.ids)?;
            m.set_mode(StepMode::Graph);
            let replay = fresh(CLAUSE, "replay", m, &p.ids)?;
            let (Some(eager), Some(replay)) = (eager, replay) else {
                ok = false;
                continue;
            };
            let identical = same(&eager, &replay);
            let pass = identical && finite(&eager);
            println!(
                "{CLAUSE}: {} {} tokens, ids {:?}, logits bit-identical {identical}, finite {} {}",
                p.name,
                p.ids.len(),
                &eager.ids[eager.ids.len().saturating_sub(4)..],
                finite(&eager),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------------------ reset_rewinds

    fn reset_rewinds(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "reset_rewinds";
        m.set_mode(StepMode::Graph);
        let mut ok = true;
        for p in &case.prompts {
            let first = fresh(CLAUSE, "the first run", m, &p.ids)?;
            let second = fresh(CLAUSE, "the second run", m, &p.ids)?;
            let (Some(first), Some(second)) = (first, second) else {
                ok = false;
                continue;
            };
            let ids = first.ids == second.ids;
            let bits = same(&first, &second);
            let pass = ids && bits;
            println!(
                "{CLAUSE}: {} ids equal {ids}, logits bit-identical {bits} {}",
                p.name,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------------------ state_carries

    /// Layer 0's scan state: finite and not all zero.
    fn layer0_state(m: &Model) -> Result<(bool, bool), GateError> {
        let state = m
            .body("gate_nemotron_e2e")?
            .scan_state(m.gpu().stream(), 0)?;
        Ok((
            state.iter().all(|v| v.is_finite()),
            state.iter().any(|&v| v != 0.0),
        ))
    }

    fn state_carries(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "state_carries";
        m.set_mode(StepMode::Graph);
        let mut ok = true;
        for p in &case.prompts {
            let Some(&last) = p.ids.last() else {
                continue;
            };
            let Some(_) = fresh(CLAUSE, "the first run", m, &p.ids)? else {
                ok = false;
                continue;
            };
            let (finite_state, nonzero) = layer0_state(m)?;
            let Some(after_a) = or_fault(CLAUSE, "the last token", feed(m, &[last]))? else {
                ok = false;
                continue;
            };
            let Some(_) = fresh(CLAUSE, "the second run", m, &p.ids)? else {
                ok = false;
                continue;
            };
            let Some(after_b) = or_fault(CLAUSE, "the last token", feed(m, &[last]))? else {
                ok = false;
                continue;
            };
            let Some(alone) = fresh(CLAUSE, "the last token alone", m, &[last])? else {
                ok = false;
                continue;
            };
            let twice = after_a.ids == after_b.ids && same(&after_a, &after_b);
            let carried = !same(&after_a, &alone);
            let pass = twice && finite_state && nonzero && carried;
            println!(
                "{CLAUSE}: {} last token {last} stepped twice from the same reset: ids {:?} vs {:?} \
                 equal {twice}; layer 0 scan state finite {finite_state} nonzero {nonzero}; \
                 logits after the prompt differ from the token alone {carried} {}",
                p.name,
                after_a.ids,
                after_b.ids,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------------- the mainline set

    /// The step path's argmax and last-position logits after the set's
    /// prompt `p`, from a reset, or `None` after a line naming the fault; run
    /// once a prompt ([`Case::steps`]).
    fn body_on(
        clause: &str,
        m: &mut Model,
        case: &Case,
        p: &refset::nemotron::Prompt,
    ) -> Result<Option<Last>, GateError> {
        if let Some(hit) = case.steps.borrow().get(&p.name) {
            return Ok(Some(hit.clone()));
        }
        let ids = case.set.ids(p)?;
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut last = 0;
        for &t in &ids {
            match m.step(&[t]) {
                Ok(id) => last = id,
                Err(GpuError::Fault { fault, .. }) => {
                    println!("{clause}: {} raised the fault word: {fault} FAIL", p.name);
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            }
        }
        let run = (last, m.logits()?);
        case.steps.borrow_mut().insert(p.name.clone(), run.clone());
        Ok(Some(run))
    }

    fn top1_is_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "top1_is_mainlines";
        let mut ok = true;
        for p in step_prompts(case)? {
            let want = crate::e2e::argmax(&case.set.logits(p)?);
            let Some((got, logits)) = body_on(CLAUSE, m, case, p)? else {
                ok = false;
                continue;
            };
            let pass = got == want && crate::e2e::argmax(&logits) == want;
            println!(
                "{CLAUSE}: {} {} tokens, body argmax {got} (logits' {}), mainline {want} {}",
                p.name,
                p.n_ids,
                crate::e2e::argmax(&logits),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    fn logits_near_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "logits_near_mainlines";
        let mut ok = true;
        for p in step_prompts(case)? {
            let want = case.set.logits(p)?;
            let Some((_, got)) = body_on(CLAUSE, m, case, p)? else {
                ok = false;
                continue;
            };
            if got.len() != want.len() || !got.iter().all(|v| v.is_finite()) {
                println!(
                    "{CLAUSE}: {} {} logits for the set's {}, finite {} FAIL",
                    p.name,
                    got.len(),
                    want.len(),
                    got.iter().all(|v| v.is_finite())
                );
                ok = false;
                continue;
            }
            let max_delta = max_delta(&got, &want);
            let cosine = cosine(&got, &want);
            let (top_got, top_want) = (top_ids(&got, TOP_K), top_ids(&want, TOP_K));
            let (mut set_got, mut set_want) = (top_got.clone(), top_want.clone());
            set_got.sort_unstable();
            set_want.sort_unstable();
            let pass = cosine >= COSINE_FLOOR && set_got == set_want;
            let values = |ids: &[usize], v: &[f32]| -> Vec<String> {
                ids.iter().map(|&i| format!("{i}:{:.3}", v[i])).collect()
            };
            println!(
                "{CLAUSE}: {} {} tokens, max |delta| {max_delta:.4}, cosine {cosine:.6} \
                 (floor {COSINE_FLOOR}), top-{TOP_K} same ids {} (body {:?}, mainline {:?}) {}",
                p.name,
                p.n_ids,
                set_got == set_want,
                values(&top_got, &got),
                values(&top_want, &want),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------------- the prompt path

    /// The largest |a − b| over two logits rows.
    fn max_delta(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
            .fold(0.0, f64::max)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        dot / (norm(a) * norm(b))
    }

    /// `v`'s largest value less its second.
    fn margin(v: &[f32]) -> f64 {
        let top = top_ids(v, 2);
        f64::from(v[top[0]]) - f64::from(v[top[1]])
    }

    /// The prompt path's argmax and last-position logits after `ids` in
    /// passes of `rows`, from a reset, or `None` after a line naming the
    /// fault.
    fn prefill_on(
        clause: &str,
        m: &mut Model,
        name: &str,
        (ids, rows): (&[u32], usize),
    ) -> Result<Option<Last>, GateError> {
        m.reset()?;
        match m.prefill_with(ids, rows) {
            Ok(id) => Ok(Some((id, m.logits()?))),
            Err(GpuError::Fault { fault, .. }) => {
                println!("{clause}: {name} raised the fault word: {fault} FAIL");
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Prompt `p`'s band `B` (the step path's max |Δ| against mainline's
    /// logits, module doc) and the step path's run, or `None` after a fault
    /// line.
    fn band(
        clause: &str,
        m: &mut Model,
        case: &Case,
        p: &refset::nemotron::Prompt,
    ) -> Result<Option<(f64, Last)>, GateError> {
        let want = case.set.logits(p)?;
        let Some(step) = body_on(clause, m, case, p)? else {
            return Ok(None);
        };
        Ok(Some((max_delta(&step.1, &want), step)))
    }

    fn prefill_is_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "prefill_is_mainlines";
        let mut ok = true;
        for p in &case.set.prompts {
            let want = case.set.logits(p)?;
            let ids = case.set.ids(p)?;
            let Some((b, _)) = band(CLAUSE, m, case, p)? else {
                ok = false;
                continue;
            };
            let Some((id, got)) = prefill_on(CLAUSE, m, &p.name, (&ids, PASS_ROWS))? else {
                ok = false;
                continue;
            };
            let top1 = crate::e2e::argmax(&want);
            let cos = cosine(&got, &want);
            let (mut set_got, mut set_want) = (top_ids(&got, TOP_K), top_ids(&want, TOP_K));
            set_got.sort_unstable();
            set_want.sort_unstable();
            let pass = id == top1 && cos >= COSINE_FLOOR;
            println!(
                "{CLAUSE}: {} {} tokens, argmax {id} mainline {top1}, cosine {cos:.6} (floor \
                 {COSINE_FLOOR}), top-{TOP_K} same ids {} (printed), max |delta| {:.4} (step \
                 path's {b:.4}) {}",
                p.name,
                p.n_ids,
                set_got == set_want,
                max_delta(&got, &want),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    fn prefill_then_step_is_mainlines(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "prefill_then_step_is_mainlines";
        let mut ok = true;
        for p in &case.set.prompts {
            let ids = case.set.ids(p)?;
            let Some((&last, head)) = ids.split_last().filter(|(_, h)| !h.is_empty()) else {
                println!(
                    "{CLAUSE}: {} {} id: no pass before its step, skipped",
                    p.name, p.n_ids
                );
                continue;
            };
            let want = case.set.logits(p)?;
            if prefill_on(CLAUSE, m, &p.name, (head, PASS_ROWS))?.is_none() {
                ok = false;
                continue;
            }
            let then = |m: &mut Model, first: u32| -> Result<Run, GpuError> {
                let mut run = Run {
                    ids: vec![m.step(&[first])?],
                    logits: vec![m.logits()?],
                };
                for _ in 0..STEPS_AFTER {
                    let prev = *run.ids.last().unwrap_or(&first);
                    run.ids.push(m.step(&[prev])?);
                    run.logits.push(m.logits()?);
                }
                Ok(run)
            };
            let Some(pre) = or_fault(CLAUSE, &p.name, then(m, last))? else {
                ok = false;
                continue;
            };
            let (id, got) = (pre.ids[0], &pre.logits[0]);
            let top1 = crate::e2e::argmax(&want);
            let cos = cosine(got, &want);
            let pass = id == top1 && cos >= COSINE_FLOOR;
            m.reset()?;
            let stepped = feed(m, head).and_then(|_| then(m, last));
            let Some(stp) = or_fault(CLAUSE, &p.name, stepped)? else {
                ok = false;
                continue;
            };
            let shared = pre
                .ids
                .iter()
                .zip(&stp.ids)
                .position(|(a, b)| a != b)
                .unwrap_or(pre.ids.len());
            let worst = (0..shared)
                .map(|t| max_delta(&pre.logits[t], &stp.logits[t]))
                .fold(0.0, f64::max);
            println!(
                "{CLAUSE}: {} {} ids as a pass of {} and a step, argmax {id} mainline {top1}, \
                 cosine {cos:.6} (floor {COSINE_FLOOR}); then {STEPS_AFTER} greedy steps: ids \
                 equal to the step path's for {} of {STEPS_AFTER}, largest |delta| {worst:.4} \
                 (printed) {}",
                p.name,
                p.n_ids,
                head.len(),
                shared.saturating_sub(1),
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    fn prefill_edges(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "prefill_edges";
        let mut ok = true;
        for name in ["prose255", "prose256", "prose257", "prose512"] {
            let Some(p) = case.set.prompts.iter().find(|p| p.name == name) else {
                println!("{CLAUSE}: the set holds no prompt {name} FAIL");
                ok = false;
                continue;
            };
            let ids = case.set.ids(p)?;
            let Some((b, step)) = band(CLAUSE, m, case, p)? else {
                ok = false;
                continue;
            };
            let Some((id, got)) = prefill_on(CLAUSE, m, name, (&ids, EDGE_ROWS))? else {
                ok = false;
                continue;
            };
            let d = max_delta(&got, &step.1);
            let pass = d <= b && (id == step.0 || margin(&step.1) <= 2.0 * b);
            println!(
                "{CLAUSE}: {name} {} tokens in passes of {EDGE_ROWS}, argmax {id} step path's {}, \
                 max |delta| {d:.4} band B {b:.4} {}",
                p.n_ids,
                step.0,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// A model loaded at [`SMALL_CTX`] (prompt buffers of that many rows)
    /// takes `prefill` of [`SMALL_IDS`] ids in one pass, and its logits are the
    /// gate's model's bits on the same ids: the same pass, cut by the context.
    fn prefill_at_a_small_context(m: &mut Model, case: &Case) -> Result<bool, GateError> {
        const CLAUSE: &str = "prefill_at_a_small_context";
        let Some(p) = case.set.prompts.iter().find(|p| p.name == "prose255") else {
            println!("{CLAUSE}: the set holds no prompt prose255 FAIL");
            return Ok(false);
        };
        let ids = case.set.ids(p)?;
        let ids = &ids[..SMALL_IDS.min(ids.len())];
        let Some((want_id, want)) = prefill_on(CLAUSE, m, "the gate's model", (ids, PASS_ROWS))?
        else {
            return Ok(false);
        };
        let file = open_split(Arch::NemotronH, "gate-gpu-nemotron-e2e")?;
        let mut small = Model::open(Gpu::new()?, file, SMALL_CTX)?;
        let got = match small.prefill(ids) {
            Ok(id) => Some((id, small.logits()?)),
            Err(GpuError::Fault { fault, .. }) => {
                println!("{CLAUSE}: the small model raised the fault word: {fault} FAIL");
                None
            }
            Err(e) => {
                println!("{CLAUSE}: the small model's prefill: {e} FAIL");
                None
            }
        };
        let Some((got_id, got)) = got else {
            return Ok(false);
        };
        let pass = got_id == want_id && bits_equal(&got, &want);
        println!(
            "{CLAUSE}: prose255's first {} ids at ctx {SMALL_CTX}: argmax {got_id} (the gate's \
             model's {want_id}), logits bit-identical {} {}",
            ids.len(),
            bits_equal(&got, &want),
            verdict(pass)
        );
        Ok(pass)
    }

    // --------------------------------------------------- fault_word_clean

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
        let file = open_split(Arch::NemotronH, "gate-gpu-nemotron-e2e")?;
        let hp = Hparams::read(&file).map_err(|e| format!("nemotron_h header: {e}"))?;
        let set = LogitsSet::open(&LOGITS.path(refset::arch::nemotron_h::LOGITS_SET), &LOGITS)?;
        let longest = set.prompts.iter().map(|p| p.n_ids as usize).max();
        if longest.is_none_or(|n| n + STEPS_AFTER + 1 > CTX) {
            return Err(format!(
                "the set's longest prompt is {longest:?} tokens, the gate's context holds {CTX}"
            )
            .into());
        }
        let case = Case {
            set,
            prompts: vec![Prompt {
                name: "walk16",
                ids: walk(PROMPT_LEN, hp.n_vocab),
            }],
            hp,
            steps: RefCell::new(BTreeMap::new()),
        };
        let t = std::time::Instant::now();
        let mut m = Model::open(Gpu::new()?, file, CTX)?;
        println!(
            "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        let mut ok = true;
        for (name, clause) in CLAUSES {
            let pass = clause(&mut m, &case)?;
            if !pass {
                println!("clause {name} FAIL");
            }
            ok &= pass;
        }
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_nemotron_e2e the captured step holds the entries its layer kinds derive \
             and no group-8 flash entry; eager and replay give bit-identical logits; a reset \
             rewinds the ids and the logits bits; the scan state carries; the body's argmax is \
             mainline's and its logits keep the cosine floor and the top-{TOP_K} ids on every \
             prompt of the set; the prompt path keeps mainline's argmax and cosine floor, also on \
             the step after a pass, and stays within the step path's own band across pass edges \
             of {EDGE_ROWS} rows and at a context under one pass; the fault word stays clean"
        );
        Ok(())
    }
}
