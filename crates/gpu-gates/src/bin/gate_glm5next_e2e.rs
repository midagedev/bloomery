//! The GLM-5.3-Flash end-to-end gate: the whole program — 34 KDA layers and
//! 11 latent-attention layers, every block in the four hyper-connection
//! streams, three dense blocks and 42 routed blocks whose experts all run on
//! the host tier, the streams' mean, the q8_0 head and the argmax — loaded
//! once by its placement on the gate card (`workstation::plan_gate`), against
//! ik's CPU oracle sets (`refset::arch::glm5next`: the 5-token batch set, the
//! step after a fused 4-token prefill, the same after a prefill run node by
//! node, the step after a fused 1,024-token prefill of the prose), every set
//! read through its family.
//!
//! What is asserted:
//! - (s) structure: the captured decode step holds [`NODES_DECODE`] nodes,
//!   [`MEMOPS`] of them stream memory-operation batches (each routed layer's
//!   go and wait) and the rest kernels, and the program's own count
//!   (`step_launches` over the body's layer programs) is the same; the
//!   latent layers are the ones the file's description names, the dense
//!   blocks the first three; the stores' bytes equal their derivation from
//!   the header ([`store_bytes`]).
//! - (p) one chain: the batch set's five tokens as five graph steps and as
//!   five eager steps (the per-layer taps armed), each from a reset, leave
//!   the same per-position tokens and logits bit for bit.
//! - (c) free-running on the batch set: each position's every layer output
//!   (its four streams) against ik's `l_out-L` within [`FREE_BAND`], the
//!   first layer past it named; the last position's argmax equal to ik's
//!   `result_output` argmax.
//! - (t) each step set: its prefill fed by our own steps from a reset, then
//!   the step; its argmax equal to ik's, or — named and counted, never
//!   silently — our argmax ik's runner-up, ik's own margin between the two
//!   inside twice the distance between our logits and ik's at those ids,
//!   and our whole logits row within [`FREE_BAND`] of ik's (the head is a
//!   linear map of the last layer's streams, so it carries their bound). The step's layer
//!   outputs against ik's `l_out-L` are printed, and held to [`FREE_BAND`]
//!   on the two 4-token sets only: ik's prefill is its own fused graph, not
//!   our steps, and after 1,024 positions the two states have drifted by an
//!   amount no band here derives.
//!
//! Named differences, not banded away: ik clamps each KDA state to ±1e6
//! after every token, ours raises its fault site where the state stops being
//! finite and clamps nothing; ik renormalizes the router's eight weights by
//! their bare sum, ours adds `1e-20` (under half an ulp of every sum from
//! 2^-42 up); ik's CPU head quantizes the normed row to q8_0 per 32 values
//! for its q8_0 lm_head, ours multiplies the f32 row.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_e2e: built without the `glm5next` feature; see `just gate-gpu-glm5next-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_e2e", gate::run())
}

#[cfg(feature = "glm5next")]
mod gate {
    use std::time::Instant;

    use app::arch::glm5next::GlmCfg;
    use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, ref_tensor_logical_in, verdict,
    };
    use bloomery_gpu_glm5next::{Body, Glm5nextModel, step_launches};
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::glm5next::place::PlanInputs;
    use model::placement::{Machine, Plan, PlanLevers, workstation};
    use refset::arch::glm5next::{BATCH, D1K, IK, MODEL, STEP4, STEP4_EVERY_NODE};
    use runtime::layer::{FfnKind, MixerKind};

    /// Cache rows: the 1,024-token set's step at position 1,024, with room.
    const CTX: usize = 1088;

    /// The file's shape, as the header states it (glmops-design §1): what
    /// the derivations below are written against.
    const HIDDEN: usize = 4096;
    const STREAMS: usize = 4;
    const N_LAYER: usize = 45;
    const N_LATENT: usize = 11;
    const N_KDA: usize = 34;
    const N_DENSE: usize = 3;
    const N_ROUTED: usize = 42;
    const N_VOCAB: usize = 154_880;

    /// PIN(2026-09-27): the captured decode step's node count, derived before
    /// the chain was built: every layer's two sub-layers take three launches
    /// each for the streams (`hc_pre_q8_0`, the fold, `hc_post`); a mixer 11
    /// (KDA: the norm, five q8_0 projections of the normed row and two of
    /// the low-rank halves, the conv and prep, the delta step, the gated
    /// norm, the output projection — the q·k·v counted once, joined; latent:
    /// the norm, the joined projection, the q_a norm, two appends, q_b,
    /// k_b, the attention's two launches, v_b, the output projection); a
    /// dense block 3 (norm, gate·up, down); a routed block 8 (norm, router,
    /// handoff, the go, the shared gate·up and down, the wait, the sum); the
    /// head 4 (the mean, the norm, the q8_0 gemv, the argmax):
    /// 45·6 + 45·11 + 3·3 + 42·8 + 4.
    const NODES_DECODE: usize = 1114;

    /// PIN(2026-09-27): each routed layer's go and wait.
    const MEMOPS: usize = 2 * N_ROUTED;

    const _: () = assert!(
        NODES_DECODE == N_LAYER * 6 + N_LAYER * 11 + N_DENSE * 3 + N_ROUTED * 8 + 4
            && N_KDA + N_LATENT == N_LAYER
            && N_DENSE + N_ROUTED == N_LAYER
    );

    /// PIN(2026-09-27): the free-running bound on a layer output's relative
    /// distance from ik's. Derivation, the qwen3moe and qwen35moe gates':
    /// the forced arm's per-layer errors, up to about 4e-2 of a layer's
    /// update, whose norm is of the residual's order, added in quadrature
    /// over 45 layers, independent: √45 · 4e-2 ≈ 0.27, rounded up.
    const FREE_BAND: f64 = 0.28;

    /// The stores' bytes at [`CTX`] rows, derived from the header: each KDA
    /// layer's state, 64 heads of 128 × 128 f32, and conv ring, 11 rows of
    /// 3 · 64 · 128 f32; each latent layer's latent and index rows, 512 +
    /// 256 f16 a position.
    fn store_bytes() -> usize {
        N_KDA * (64 * 128 * 128 + 11 * 3 * 64 * 128) * 4 + N_LATENT * CTX * (512 + 256) * 2
    }

    /// `‖a − b‖ / ‖b‖` in f64; infinite on a NaN or a length mismatch, so
    /// neither passes a band.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += f64::from(y).powi(2);
        }
        let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_nan() { f64::INFINITY } else { r }
    }

    /// The first index of the largest value, as the head's argmax breaks ties.
    fn argmax(v: &[f32]) -> u32 {
        let best = v
            .iter()
            .enumerate()
            .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b });
        best as u32
    }

    /// The runner-up's index: the largest value other than at `top`.
    fn second(v: &[f32], top: u32) -> u32 {
        let best = v.iter().enumerate().fold(None::<usize>, |b, (i, &x)| {
            if i == top as usize {
                b
            } else {
                match b {
                    Some(j) if v[j] >= x => Some(j),
                    _ => Some(i),
                }
            }
        });
        best.unwrap_or(0) as u32
    }

    /// A set's tap `name` in its logical order.
    fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
    }

    /// The open's records, as the gate prints them.
    struct Log {
        t: Instant,
        nodes: Option<usize>,
    }

    impl OpenLog<Body> for Log {
        fn plan(
            &mut self,
            place: &'static str,
            _inputs: &PlanInputs,
            _machine: &Machine,
            plan: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            println!(
                "plan place={place} ctx_max={} host_experts={} card_experts={}",
                plan.ctx_max, plan.host.experts, plan.cards[0].experts
            );
            Ok(true)
        }

        fn load(&mut self, m: &Glm5nextModel) -> Result<(), SessionError> {
            println!(
                "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
                m.resident_bytes(),
                m.layers().len(),
                self.t.elapsed().as_secs_f64()
            );
            Ok(())
        }

        fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
            self.nodes = Some(nodes);
            Ok(())
        }

        fn prompt_buffers(&mut self, _m: &Glm5nextModel) -> Result<(), SessionError> {
            Ok(())
        }
    }

    fn open() -> Result<(Session<Body>, usize), GateError> {
        let levers = bloomery_levers::at_main(&[])?;
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let cfg = GlmCfg {
            place: PlanLevers::from_levers(&levers)?,
            host: levers.host(),
        };
        let mut log = Log {
            t: Instant::now(),
            nodes: None,
        };
        let args = OpenArgs {
            place: "gate",
            machine: workstation::plan_gate,
            ctx: CTX,
            mode: StepMode::Graph,
            cfg,
        };
        let loaded =
            Loaded::<Body>::open(file, args, &mut log)?.ok_or("the open stopped at its plan")?;
        let s = loaded.ready(&mut log)?;
        let nodes = log.nodes.ok_or("graph mode captured no step")?;
        Ok((s, nodes))
    }

    // ------------------------------------------------------ (s) structure

    fn structure(m: &Glm5nextModel, nodes: usize) -> Result<bool, GateError> {
        let body = m.body("structure")?;
        let kinds = body.kinds();
        let at = |f: &dyn Fn(usize) -> bool| -> Vec<usize> {
            (0..kinds.len()).filter(|&l| f(l)).collect()
        };
        let latent = at(&|l| kinds[l].mixer == MixerKind::Latent);
        let dense = at(&|l| kinds[l].ffn == FfnKind::Dense);
        let want_latent: Vec<usize> = (0..N_LAYER).filter(|l| l % 4 == 3).collect();
        let (stores, want_stores) = (body.store_bytes(), store_bytes());
        let mut ok = kinds.len() == N_LAYER
            && latent == want_latent
            && dense == (0..N_DENSE).collect::<Vec<_>>()
            && body.host_run() == (N_DENSE..N_LAYER)
            && stores == want_stores;
        println!(
            "structure layers={} latent at {latent:?} dense at {dense:?} host run {:?}; store bytes \
             {stores} (want {want_stores}, derived) {}",
            kinds.len(),
            body.host_run(),
            verdict(ok)
        );
        let mixers: Vec<MixerKind> = kinds.iter().map(|k| k.mixer).collect();
        let ffns: Vec<FfnKind> = kinds.iter().map(|k| k.ffn).collect();
        let counted = step_launches(&mixers, &ffns);
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        let pass = nodes == NODES_DECODE
            && counted == NODES_DECODE
            && b == MEMOPS
            && k == NODES_DECODE - MEMOPS
            && other == 0;
        println!(
            "structure decode graph_nodes={nodes} (want {NODES_DECODE}; the program counts \
             {counted}) kernel={k} batch_mem_op={b} (want {MEMOPS}) other={other} {}",
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
    }

    // ------------------------------------------ (p) one chain, (c) free

    /// One run of `toks` from a reset: each position's argmax and logits,
    /// and, with the taps armed, each position's layer outputs.
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        taps: Vec<Vec<f32>>,
    }

    fn run_steps(m: &mut Glm5nextModel, toks: &[u32], mode: StepMode) -> Result<Run, GateError> {
        m.reset()?;
        m.set_mode(mode);
        let taps_on = mode == StepMode::Eager;
        {
            let (gpu, _, b) = m.body_parts("run_steps")?;
            b.set_taps(gpu, taps_on)?;
        }
        let mut r = Run {
            tokens: Vec::new(),
            logits: Vec::new(),
            taps: Vec::new(),
        };
        for &t in toks {
            r.tokens.push(m.step(&[t])?);
            r.logits.push(m.logits()?);
            if taps_on {
                let (gpu, _, b) = m.body_parts("run_steps")?;
                r.taps.push(b.taps(gpu)?);
            }
        }
        {
            let (gpu, _, b) = m.body_parts("run_steps")?;
            b.set_taps(gpu, false)?;
        }
        m.set_mode(StepMode::Graph);
        Ok(r)
    }

    /// Bit equality of two logits rows.
    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn one_chain(graph: &Run, eager: &Run) -> bool {
        let tokens = graph.tokens == eager.tokens;
        let logits = graph.logits.len() == eager.logits.len()
            && graph
                .logits
                .iter()
                .zip(&eager.logits)
                .all(|(a, b)| same_bits(a, b));
        let ok = tokens && logits;
        println!(
            "one chain: graph tokens {:?}, eager tokens {:?}; logits bit for bit: {logits} {}",
            graph.tokens,
            eager.tokens,
            verdict(ok)
        );
        ok
    }

    /// Each layer's worst relative distance over the positions `taps` holds
    /// against ik's `l_out-L`, whose last rows are those positions' (ik
    /// keeps only the output token's rows past the last attention), and the
    /// index into `taps` it was met at.
    fn layer_rels(man: &RefManifest, taps: &[Vec<f32>]) -> Result<Vec<(f64, usize)>, GateError> {
        let row = STREAMS * HIDDEN;
        let n = taps.len();
        let mut worst = vec![(0.0f64, 0usize); N_LAYER];
        for (l, w) in worst.iter_mut().enumerate() {
            let ik = tap(man, &format!("l_out-{l}"))?;
            let kept = ik.len() / row;
            for (t, ours) in taps.iter().enumerate() {
                let Some(i) = (t + kept).checked_sub(n) else {
                    continue;
                };
                let got = &ours[l * row..(l + 1) * row];
                let e = rel(got, &ik[i * row..(i + 1) * row]);
                if e > w.0 {
                    *w = (e, t);
                }
            }
        }
        Ok(worst)
    }

    /// The worst layer and the first past the band, printed; whether every
    /// layer is inside it.
    fn print_layers(what: &str, rels: &[(f64, usize)]) -> bool {
        for (l, &(e, t)) in rels.iter().enumerate() {
            if l % 8 == 0 || l == rels.len() - 1 || e > FREE_BAND {
                println!("{what} layer={l} l_out_rel={e:.3e} at tap {t}");
            }
        }
        match rels.iter().position(|&(e, _)| e > FREE_BAND) {
            Some(l) => {
                println!("{what}: first layer past the band {FREE_BAND:.2}: {l}");
                false
            }
            None => true,
        }
    }

    fn free(man: &RefManifest, eager: &Run) -> Result<bool, GateError> {
        let rels = layer_rels(man, &eager.taps)?;
        let inside = print_layers("free", &rels);
        let ik = tap(man, "result_output")?;
        let ik_last = &ik[ik
            .len()
            .checked_sub(N_VOCAB)
            .ok_or("result_output holds no row")?..];
        let ours = eager.logits.last().ok_or("no logits")?;
        let (top, ik_top) = (argmax(ours), argmax(ik_last));
        let ok = inside && top == ik_top;
        println!(
            "free: {} tokens, worst l_out_rel={:.3e} (band {FREE_BAND:.2}); last position argmax \
             ours={top} ik={ik_top} logits_rel={:.3e} (printed) {}",
            eager.tokens.len(),
            rels.iter().map(|r| r.0).fold(0.0, f64::max),
            rel(ours, ik_last),
            verdict(ok)
        );
        Ok(ok)
    }

    // --------------------------------------------------- (t) step sets

    /// The step of set `name` after its prefill fed by our steps; its
    /// argmax against ik's, a tie named and counted; its layer outputs
    /// held to the band when `banded`.
    fn step_set(
        m: &mut Glm5nextModel,
        name: &str,
        banded: bool,
        ties: &mut usize,
    ) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(name), &IK)?;
        let (pos, step, prefill) = man.step()?;
        let (pos, step, prefill) = (pos, step.to_vec(), prefill.to_vec());
        let [tok] = step[..] else {
            return Err(format!("{name}: a step of {} tokens, not one", step.len()).into());
        };
        m.reset()?;
        let t = Instant::now();
        if !prefill.is_empty() {
            m.step(&prefill)?;
        }
        let r = run_last(m, tok)?;
        let ik = tap(&man, "result_output")?;
        let ik_last = &ik[ik
            .len()
            .checked_sub(N_VOCAB)
            .ok_or("result_output holds no row")?..];
        let (top, ik_top) = (argmax(&r.1), argmax(ik_last));
        let ik_2 = second(ik_last, ik_top);
        let margin = f64::from(ik_last[ik_top as usize]) - f64::from(ik_last[ik_2 as usize]);
        let dist = [ik_top, ik_2]
            .iter()
            .map(|&i| (f64::from(r.1[i as usize]) - f64::from(ik_last[i as usize])).abs())
            .fold(0.0, f64::max);
        let logits_rel = rel(&r.1, ik_last);
        let tie = top != ik_top && top == ik_2 && margin <= 2.0 * dist && logits_rel <= FREE_BAND;
        *ties += usize::from(tie);
        let rels = layer_rels(&man, std::slice::from_ref(&r.0))?;
        let inside = print_layers(name, &rels);
        let ok = (top == ik_top || tie) && (inside || !banded);
        println!(
            "step {name}: position {pos} after {} fed ({:.1} s, runtime value); argmax ours={top} \
             ik={ik_top} (ik's runner-up {ik_2}, margin {margin:.4}, our distance at the two \
             {dist:.4}, logits_rel {logits_rel:.3e}{}); worst l_out_rel={:.3e} ({}) {}",
            prefill.len(),
            t.elapsed().as_secs_f64(),
            if tie { ", a named tie" } else { "" },
            rels.iter().map(|r| r.0).fold(0.0, f64::max),
            if banded { "band" } else { "printed" },
            verdict(ok)
        );
        Ok(ok)
    }

    /// The last token eagerly with the taps armed: its layer outputs and
    /// its logits.
    fn run_last(m: &mut Glm5nextModel, tok: u32) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        m.set_mode(StepMode::Eager);
        {
            let (gpu, _, b) = m.body_parts("run_last")?;
            b.set_taps(gpu, true)?;
        }
        m.step(&[tok])?;
        let logits = m.logits()?;
        let taps = {
            let (gpu, _, b) = m.body_parts("run_last")?;
            let t = b.taps(gpu)?;
            b.set_taps(gpu, false)?;
            t
        };
        m.set_mode(StepMode::Graph);
        Ok((taps, logits))
    }

    pub fn run() -> Result<(), GateError> {
        let (mut s, nodes) = open()?;
        let m = s.model_mut();
        let mut ok = structure(m, nodes)?;
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, toks, _) = man.step()?;
        let toks = toks.to_vec();
        let graph = run_steps(m, &toks, StepMode::Graph)?;
        let eager = run_steps(m, &toks, StepMode::Eager)?;
        ok &= one_chain(&graph, &eager);
        ok &= free(&man, &eager)?;
        let mut ties = 0usize;
        for (name, banded) in [(STEP4, true), (STEP4_EVERY_NODE, true), (D1K, false)] {
            ok &= step_set(m, name, banded, &mut ties)?;
        }
        println!("step sets: {ties} named tie(s)");
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
