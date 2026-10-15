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
//! The prompt is a fixed walk of token ids: with no reference set the clauses
//! are the body's own contracts. The mainline comparison (`top1_is_mainlines`,
//! `logits_near_mainlines`) joins the list when the reference set lands.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_nemotron_e2e", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, NemotronHModel, NodeInfo};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, open_split, verdict};
    use cuda_core::sys;
    use model::arch::Arch;
    use model::arch::nemotron_h::hparams::{Hparams, Kind};
    use std::collections::BTreeMap;

    type Model = NemotronHModel;

    /// Rows of the K/V planes a key head holds.
    const CTX: usize = 256;

    /// Tokens of the prompt every clause runs.
    const PROMPT_LEN: usize = 16;

    /// What a clause is run over: the layer kinds the file's header names and
    /// the prompts, in order.
    struct Case {
        hp: Hparams,
        prompts: Vec<Prompt>,
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
        ("fault_word_clean", fault_word_clean),
    ];

    /// A step run: the argmax id and the logits after each token.
    struct Run {
        ids: Vec<u32>,
        logits: Vec<Vec<f32>>,
    }

    /// Feed `ids` one token at a time from wherever `m` stands.
    fn feed(m: &mut Model, ids: &[u32]) -> Result<Run, GpuError> {
        let mut out = Run {
            ids: Vec::new(),
            logits: Vec::new(),
        };
        for &t in ids {
            out.ids.push(m.step(&[t])?);
            out.logits.push(m.logits()?);
        }
        Ok(out)
    }

    /// `r`'s run, or — when a step raised the fault word — a line naming the
    /// fault and `None`: the clause fails there and the gate goes on to the
    /// next one. Any other error ends the gate.
    fn or_fault(
        clause: &str,
        what: &str,
        r: Result<Run, GpuError>,
    ) -> Result<Option<Run>, GateError> {
        match r {
            Ok(run) => Ok(Some(run)),
            Err(GpuError::Fault { fault, .. }) => {
                println!("{clause}: {what} raised the fault word: {fault} FAIL");
                Ok(None)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Reset, then [`feed`] `ids`, as [`or_fault`] reads it.
    fn fresh(
        clause: &str,
        what: &str,
        m: &mut Model,
        ids: &[u32],
    ) -> Result<Option<Run>, GateError> {
        m.reset()?;
        or_fault(clause, what, feed(m, ids))
    }

    fn finite(r: &Run) -> bool {
        r.logits.iter().flatten().all(|v| v.is_finite())
    }

    fn same(a: &Run, b: &Run) -> bool {
        a.ids == b.ids
            && a.logits.len() == b.logits.len()
            && a.logits
                .iter()
                .zip(&b.logits)
                .all(|(x, y)| bits_equal(x, y))
    }

    // ---------------------------------------------------------- structure

    /// The kernel entries of `nodes` with their counts.
    fn histogram(nodes: &[NodeInfo]) -> BTreeMap<String, usize> {
        let mut h = BTreeMap::new();
        for k in nodes.iter().filter_map(|n| n.kernel.as_ref()) {
            *h.entry(k.name.clone()).or_insert(0) += 1;
        }
        h
    }

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

    /// The fixed prompt: a multiplicative walk through ids the vocabulary
    /// holds.
    fn walk(n: usize, vocab: usize) -> Vec<u32> {
        (0..n)
            .map(|i| u32::try_from(1_000 + (i * 7_919 + 13) % (vocab - 1_000)).unwrap_or(1))
            .collect()
    }

    pub fn run() -> Result<(), GateError> {
        let file = open_split(Arch::NemotronH, "gate-gpu-nemotron-e2e")?;
        let hp = Hparams::read(&file).map_err(|e| format!("nemotron_h header: {e}"))?;
        let case = Case {
            prompts: vec![Prompt {
                name: "walk16",
                ids: walk(PROMPT_LEN, hp.n_vocab),
            }],
            hp,
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
             rewinds the ids and the logits bits; the scan state carries; the fault word stays \
             clean"
        );
        Ok(())
    }
}
