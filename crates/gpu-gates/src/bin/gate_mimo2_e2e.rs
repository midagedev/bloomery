//! The MiMo-V2.6-Flash end-to-end gate: the whole program — 48 GQA attention
//! layers over two head widths (key 192, value 128; nine full layers and 39
//! window layers with sinks), the dense layer 0 and 47 routed blocks whose
//! experts all run on the host tier (the plan keeps none on a card), the q8_0
//! head and the argmax — loaded once by its placement on the gate card
//! (`crate::gate_card::plan_gate`), against ik's CPU oracle sets
//! (`refset::arch::mimo2`: the 5-token batch set, the step after a fused
//! 4-token prefill, the same after a prefill run node by node, and the steps
//! after 1,024 and 4,096 positions of the prose), every set read through its
//! family.
//!
//! What is asserted:
//! - (s) structure: the captured decode step holds [`NODES_DECODE`] nodes,
//!   [`MEMOPS`] of them stream memory-operation batches (each routed layer's
//!   go and wait) and the rest kernels, and the program's own count
//!   (`step_launches` over the body's layer programs, a fact of the file's
//!   description) is the same; the dense layer is layer 0 and the host run
//!   is layers `1..48`; the plan puts no routed expert on a card (the plan's
//!   `n_l` and the slot map's both); the stores hold
//!   [`KV_BYTES_PER_POSITION`] a position at [`CTX`] positions;
//!   each layer's arguments to its attention launches (kv heads, window,
//!   sinks, the rope table's base, the value multiplier) are its
//!   description's and the file's pins ([`FULL`], [`WINDOW`], [`THETA_FULL`],
//!   [`THETA_WINDOW`], [`V_SCALE`]).
//! - (p) one chain: the batch set's five tokens as five graph steps and as
//!   five eager steps (the per-layer taps armed), each from a reset, leave
//!   the same per-position tokens and logits bit for bit.
//! - (c) free-running on the batch set: its five tokens as eager steps from a
//!   reset, the last position's argmax equal to ik's `result_output` argmax,
//!   or — named and counted, never silently — ours ik's runner-up with ik's
//!   own margin between the two inside twice the distance between our logits
//!   and ik's at those ids ([`tie_allowed`]). Each layer's output against
//!   ik's `l_out-L` is printed by position; there is no band yet (the forced
//!   arm that derives one is a later round's).
//! - (t) each step set: its prefill fed by our own steps from a reset, then
//!   the step; its argmax equal to ik's or a named tie as in (c). The step's
//!   layer outputs against ik's `l_out-L` are printed, and each set's logits
//!   print their FNV-1a digest. A set whose prefill is past the 128-position
//!   window ([`D1K`](refset::arch::mimo2::D1K)) reads a window of real keys
//!   in every window layer; the 4,096-position one has the nine full layers
//!   read the whole context.
//!
//! `--only s|p|c|t` runs one clause on the load; `--step-sets short` takes
//! (t)'s two 4-token sets only, `--step-sets long` the 1,024- and
//! 4,096-position sets only, `--step-sets all` (the default) all four; it
//! goes with `--only t` or no `--only`.
//!
//! Every clause prints one elapsed line when it ends: `clause (c) in 3.2 s
//! (runtime value)`.
//!
//! Named differences, not banded away: ik multiplies a q8_0 projection's
//! input quantized to 32-value blocks (q8_2), ours the f32 input; ik divides
//! the router's eight weights by their bare sum, ours adds `1e-20`; ik's CPU
//! head quantizes the normed row to q8_0 for its q8_0 lm_head, ours
//! multiplies the f32 row. The batch set's last layer keeps one row in ik
//! (`inp_out_ids`), so layer 47's output is compared at the last position
//! alone.

#[cfg(not(feature = "mimo2"))]
fn main() {
    eprintln!(
        "gate_mimo2_e2e: built without the `mimo2` feature; see `just weekly-gpu-mimo2-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "mimo2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mimo2_e2e", gate::run())
}

#[cfg(feature = "mimo2")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "mimo2")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "mimo2")]
mod gate {
    use crate::e2e::{
        elapsed, ik_last, layer_rels, layer_table, same_bits, set_open, tie_numbers, word_after,
    };
    use std::time::Instant;

    use app::arch::mimo2::Mimo2Cfg;
    use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::flip::tie_allowed;
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::{Fnv1a64, GateError, RefManifest, checks_failed, data_dir, verdict};
    use bloomery_gpu_mimo2::{Body, Mimo2Model, set_taps};
    use bloomery_levers::CARD_BUDGET;
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::mimo2::place::PlanInputs;
    use model::arch::mimo2::program::{AttnArgs, step_launches};
    use model::arch::models::Mixer;
    use model::placement::{Machine, Plan, PlanLevers};
    use refset::arch::mimo2::{BATCH, D1K, D4096, IK, MODEL, STEP4, STEP4_EVERY_NODE};
    use refset::family::Family;
    use runtime::layer::FfnKind;

    /// Cache rows: the 4,096-position set's step at position 4,096, with a
    /// margin of whole 64 positions.
    const CTX: usize = 4160;

    /// The model's width, vocabulary and layer count.
    const HIDDEN: usize = 4096;
    const N_VOCAB: usize = 152_576;
    const N_LAYER: usize = 48;

    /// PIN(2026-10-08): the captured decode step's node count, derived before
    /// the chain was built: every layer's attention takes 7 launches (the
    /// norm, the fused q·k·v gemv, the rope-and-append, the flash's segment
    /// pass and its merge, the output gemv, the add); the dense block 4
    /// (norm, gate·up, down, add); a routed block 6 (norm, router, handoff,
    /// the go, the wait, the add); the head 3 (norm, q8_0 gemv, argmax):
    /// 48·7 + 4 + 47·6 + 3.
    const NODES_DECODE: usize = 625;

    /// PIN(2026-10-08): each routed layer's go and wait: 47 routed layers.
    const MEMOPS: usize = 94;

    /// PIN(2026-10-08): the stores' bytes a position, derived from the
    /// header: the 39 window layers' 8 key/value heads and the 9 full layers'
    /// 4, each head 192 + 128 f16 values: (39·8 + 9·4)·320·2.
    const KV_BYTES_PER_POSITION: usize = 222_720;

    /// PIN(2026-10-08): the full layers (`attention.sliding_window_pattern`
    /// 0), by index; the other 39 are window layers.
    const FULL: [usize; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];

    /// PIN(2026-10-08): the window layers' positions, the rope bases of the
    /// two layer kinds and the multiplier every layer's value rows take
    /// (`attention.sliding_window`, `rope.freq_base`, `rope.freq_base_swa`,
    /// `attention.value_scale` of the header).
    const WINDOW: usize = 128;
    const THETA_FULL: f32 = 1.0e7;
    const THETA_WINDOW: f32 = 1.0e4;
    const V_SCALE: f32 = 0.707;

    /// No band yet on a named tie's logits row against the head input's
    /// distance (the forced arm that measures one is a later round's): the
    /// tie is ik's runner-up inside twice our distance at the two ids, and the
    /// row's and the last layer's distances are printed.
    const HEAD_BAND: f64 = f64::INFINITY;

    /// What the open decided besides the session: the plan's card experts a
    /// layer.
    struct Opened {
        n_l: Vec<u64>,
    }

    /// The open's records, as the gate prints them.
    struct Log {
        t: Instant,
        n_l: Vec<u64>,
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
            self.n_l = plan.n_l.clone();
            Ok(true)
        }

        fn load(&mut self, m: &Mimo2Model) -> Result<(), SessionError> {
            println!(
                "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
                m.resident_bytes(),
                m.layers().len(),
                self.t.elapsed().as_secs_f64()
            );
            Ok(())
        }

        fn capture(&mut self, _nodes: usize) -> Result<(), SessionError> {
            Ok(())
        }

        fn prompt_buffers(&mut self, _m: &Mimo2Model) -> Result<(), SessionError> {
            Ok(())
        }
    }

    /// The session at [`CTX`] positions on the gate placement, its prompts
    /// fed one step an id.
    fn open(levers: &bloomery_levers::Levers) -> Result<(Session<Body>, Opened), GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let cfg = Mimo2Cfg {
            place: PlanLevers::from_levers(levers)?,
            host: levers.host(),
        };
        let mut log = Log {
            t: Instant::now(),
            n_l: Vec::new(),
        };
        let args = OpenArgs {
            place: "gate",
            machine: crate::gate_card::plan_gate,
            ctx: CTX,
            mode: StepMode::Graph,
            cfg,
        };
        let s = Loaded::<Body>::open(file, args, &mut log)?
            .ok_or("the open stopped at its plan")?
            .ready(&mut log)?;
        Ok((s, Opened { n_l: log.n_l }))
    }

    // ------------------------------------------------------ (s) structure

    /// Layer `l`'s attention arguments as the file's pins say them.
    fn pinned(l: usize) -> AttnArgs {
        if FULL.contains(&l) {
            AttnArgs {
                kv_heads: 4,
                window: 0,
                sinks: false,
                theta: THETA_FULL,
                v_scale: V_SCALE,
            }
        } else {
            AttnArgs {
                kv_heads: 8,
                window: WINDOW,
                sinks: true,
                theta: THETA_WINDOW,
                v_scale: V_SCALE,
            }
        }
    }

    /// Layer `l`'s attention arguments as the description says them: the
    /// header's own facts, read apart from the program's mapping.
    fn described(inputs: &PlanInputs, l: usize) -> Result<AttnArgs, GateError> {
        let Mixer::Gqa(g) = &inputs.spec.layers[l].mixer else {
            return Err(format!("layer {l}: a mixer that is not GQA").into());
        };
        Ok(AttnArgs {
            kv_heads: g.kv_heads as usize,
            window: g.window.map_or(0, |w| w as usize),
            sinks: g.sinks,
            theta: g.rope.base,
            v_scale: g.value_scale.unwrap_or(1.0),
        })
    }

    fn structure(m: &Mimo2Model, o: &Opened) -> Result<bool, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let inputs = PlanInputs::read(&file)?;
        drop(file);
        let body = m.body("structure")?;
        let kinds = body.kinds();
        let dense: Vec<usize> = (0..kinds.len())
            .filter(|&l| kinds[l].ffn == FfnKind::Dense)
            .collect();
        let (stores, want_stores) = (body.store_bytes(), KV_BYTES_PER_POSITION * CTX);
        let ok = kinds.len() == N_LAYER
            && dense == [0]
            && body.host_run() == (1..N_LAYER)
            && body.ctx() == CTX
            && stores == want_stores;
        println!(
            "structure layers={} dense at {dense:?} host run {:?}; ctx {} (want {CTX}); store \
             bytes {stores} (want {want_stores}, derived) {}",
            kinds.len(),
            body.host_run(),
            body.ctx(),
            verdict(ok)
        );
        // No routed expert on a card: the plan's count and the slot map's.
        let slots = body.hybrid().slots();
        let mut on_card = 0usize;
        for l in body.host_run() {
            on_card += slots.on_card(l)? + slots.on_tier(l)?;
        }
        let planned: u64 = o.n_l.iter().sum();
        let card_ok = on_card == 0 && planned == 0;
        println!(
            "structure card experts: the plan's n_l sums to {planned}, the slot map holds \
             {on_card} on a card or tier (want 0 and 0: every routed expert on the host) {}",
            verdict(card_ok)
        );
        // Each layer's attention arguments: the program's against the
        // description's and against the file's pins.
        let args = body.attn_args();
        let mut args_ok = args.len() == N_LAYER;
        let mut first_off = None;
        for (l, got) in args.iter().enumerate() {
            let want = described(&inputs, l)?;
            if *got != want || *got != pinned(l) {
                args_ok = false;
                first_off.get_or_insert((l, *got, want, pinned(l)));
            }
        }
        println!(
            "structure attention arguments over {} layers: {} full (kv 4, window 0, no sinks, \
             theta {THETA_FULL}), {} window (kv 8, window {WINDOW}, sinks, theta {THETA_WINDOW}), \
             value multiplier {V_SCALE}; the program's equal the description's and the pins' \
             {}{}",
            args.len(),
            FULL.len(),
            N_LAYER - FULL.len(),
            verdict(args_ok),
            first_off.map_or(String::new(), |(l, got, want, pin)| format!(
                "; first off at layer {l}: program {got:?}, description {want:?}, pin {pin:?}"
            ))
        );
        let ffns: Vec<FfnKind> = kinds.iter().map(|k| k.ffn).collect();
        let counted = step_launches(&ffns);
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        let nodes = k + b + other;
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
        Ok(ok && card_ok && args_ok && pass)
    }

    // ------------------------------------------------ (p) one chain, (c) free

    /// One run of `toks` from a reset: each position's argmax and logits,
    /// and, with the taps armed, each position's layer outputs.
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        taps: Vec<Vec<f32>>,
    }

    fn run_steps(m: &mut Mimo2Model, toks: &[u32], mode: StepMode) -> Result<Run, GateError> {
        m.reset()?;
        let taps_on = mode == StepMode::Eager;
        set_taps(m, taps_on)?;
        m.set_mode(mode);
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
        set_taps(m, false)?;
        m.set_mode(StepMode::Graph);
        Ok(r)
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

    /// The set holds every tensor the gate compares with ik's: each layer's
    /// output (`l_out-L`) and the logits (`result_output`), the names the
    /// shared helpers read them by, so a set short of one fails here and not
    /// after its steps.
    fn taps_in_set(man: &RefManifest) -> Result<(), GateError> {
        for l in 0..N_LAYER {
            man.tensor(&format!("l_out-{l}"), 0)?;
        }
        man.tensor("result_output", 0)?;
        Ok(())
    }

    /// The last position's argmax against ik's: equal, or a named tie
    /// ([`tie_allowed`], no band on the logits row). Prints the numbers; the
    /// verdict and whether it was a tie.
    fn last_argmax(what: &str, (ours, ik): (&[f32], &[f32]), input_rel: f64) -> (bool, bool) {
        let (top, ik_top, ik_2, margin, dist, logits_rel) = tie_numbers(ours, ik);
        let tie = tie_allowed(
            (top, ik_top, ik_2),
            (margin, dist),
            (logits_rel, input_rel),
            HEAD_BAND,
        );
        println!(
            "{what}: argmax ours={top} ik={ik_top} (ik's runner-up {ik_2}, margin {margin:.4}, \
             our distance at the two {dist:.4}, logits_rel {logits_rel:.3e} against the last \
             layer's {input_rel:.3e}{})",
            if tie { ", a named tie" } else { "" }
        );
        (top == ik_top || tie, tie)
    }

    /// (c): the batch set's tokens as eager steps, the last position's argmax
    /// against ik's, each layer's output by position printed.
    fn free(m: &mut Mimo2Model, man: &RefManifest, toks: &[u32]) -> Result<bool, GateError> {
        taps_in_set(man)?;
        let eager = run_steps(m, toks, StepMode::Eager)?;
        let table = layer_table(man, &eager.taps, HIDDEN, N_LAYER)?;
        for (l, row) in table.iter().enumerate() {
            let cells: Vec<String> = row
                .iter()
                .map(|e| e.map_or("-".to_string(), |e| format!("{e:.3e}")))
                .collect();
            println!("free table layer={l} l_out_rel by tap {}", cells.join(" "));
        }
        let rels = layer_rels(man, &eager.taps, HIDDEN, N_LAYER)?;
        let input_rel = rels.last().map_or(f64::INFINITY, |r| r.0);
        let worst =
            rels.iter().enumerate().fold(
                (0.0f64, 0usize),
                |w, (l, r)| {
                    if r.0 > w.0 { (r.0, l) } else { w }
                },
            );
        let ik = ik_last(man, N_VOCAB)?;
        let ours = eager.logits.last().ok_or("no logits")?;
        let (ok, _) = last_argmax("free", (ours, &ik), input_rel);
        println!(
            "free: {} tokens, worst l_out_rel={:.3e} at layer {} (printed, no band); last \
             position {}",
            eager.tokens.len(),
            worst.0,
            worst.1,
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------------------ (t) sets

    /// The step of set `name` after its prefill fed by our steps; its argmax
    /// against ik's, a tie named and counted; its layer outputs printed.
    fn step_set(
        m: &mut Mimo2Model,
        (name, family): (&str, &Family),
        ties: &mut usize,
    ) -> Result<bool, GateError> {
        let (man, pos, tok, prefill, _) = set_open((name, family), None)?;
        taps_in_set(&man)?;
        m.reset()?;
        let t = Instant::now();
        if !prefill.is_empty() {
            m.step(&prefill)?;
        }
        let (taps, logits) = run_last(m, tok)?;
        let ik = ik_last(&man, N_VOCAB)?;
        let rels = layer_rels(&man, std::slice::from_ref(&taps), HIDDEN, N_LAYER)?;
        let input_rel = rels.last().map_or(f64::INFINITY, |r| r.0);
        for (l, &(e, _)) in rels.iter().enumerate() {
            println!("{name} layer={l} l_out_rel={e:.3e}");
        }
        println!(
            "step {name}: logits digest {:016x} (FNV-1a over the f32 bits)",
            Fnv1a64::default().f32s(&logits).value()
        );
        let (ok, tie) = last_argmax(&format!("step {name}"), (&logits, &ik), input_rel);
        *ties += usize::from(tie);
        println!(
            "step {name}: position {pos} after {} fed ({:.1} s, runtime value); worst \
             l_out_rel={:.3e} (printed, no band) {}",
            prefill.len(),
            t.elapsed().as_secs_f64(),
            rels.iter().map(|r| r.0).fold(0.0, f64::max),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The last token eagerly with the taps armed: its layer outputs and its
    /// logits.
    fn run_last(m: &mut Mimo2Model, tok: u32) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        set_taps(m, true)?;
        m.step(&[tok])?;
        let logits = m.logits()?;
        let taps = {
            let (gpu, _, b) = m.body_parts("run_last")?;
            b.taps(gpu)?
        };
        set_taps(m, false)?;
        m.set_mode(StepMode::Graph);
        Ok((taps, logits))
    }

    // ------------------------------------------------------------ selectors

    /// Which clause a run takes.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Only {
        All,
        S,
        P,
        C,
        T,
    }

    /// Which step sets (t) takes.
    #[derive(Clone, Copy)]
    enum StepSets {
        All,
        /// [`STEP4`] and [`STEP4_EVERY_NODE`].
        Short,
        /// [`D1K`] and [`D4096`].
        Long,
    }

    impl StepSets {
        fn name(self) -> &'static str {
            match self {
                StepSets::All => "all",
                StepSets::Short => "short",
                StepSets::Long => "long",
            }
        }

        /// Whether the set is taken: the 4-token sets are the short ones.
        fn takes(self, name: &str) -> bool {
            let short = name == STEP4 || name == STEP4_EVERY_NODE;
            match self {
                StepSets::All => true,
                StepSets::Short => short,
                StepSets::Long => !short,
            }
        }
    }

    /// `--only s|p|c|t`, or every clause.
    fn only() -> Result<Only, GateError> {
        match word_after("--only") {
            None => Ok(Only::All),
            Some(w) => match w.as_deref() {
                Some("s") => Ok(Only::S),
                Some("p") => Ok(Only::P),
                Some("c") => Ok(Only::C),
                Some("t") => Ok(Only::T),
                other => Err(format!("--only is s, p, c or t, not {other:?}").into()),
            },
        }
    }

    /// `--step-sets short|long|all`, `all` when absent; refused beside an
    /// `--only` that runs no (t).
    fn step_sets(only: Only) -> Result<StepSets, GateError> {
        let sets = match word_after("--step-sets") {
            None => return Ok(StepSets::All),
            Some(w) => match w.as_deref() {
                Some("all") => StepSets::All,
                Some("short") => StepSets::Short,
                Some("long") => StepSets::Long,
                other => {
                    return Err(format!("--step-sets is short, long or all, not {other:?}").into());
                }
            },
        };
        if !matches!(only, Only::All | Only::T) {
            return Err(
                "--step-sets names the step sets of (t): it goes with --only t or no --only".into(),
            );
        }
        Ok(sets)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[CARD_BUDGET])?;
        crate::gate_card::init()?;
        let only = only()?;
        let sets = step_sets(only)?;
        let (mut s, opened) = open(&levers)?;
        let m = s.model_mut();
        let mut ok = true;
        if matches!(only, Only::All | Only::S) {
            let t = Instant::now();
            ok &= structure(m, &opened)?;
            elapsed("(s)", &t);
        }
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, toks, _) = man.step()?;
        let toks = toks.to_vec();
        if matches!(only, Only::All | Only::P) {
            let t = Instant::now();
            let graph = run_steps(m, &toks, StepMode::Graph)?;
            let eager = run_steps(m, &toks, StepMode::Eager)?;
            ok &= one_chain(&graph, &eager);
            elapsed("(p)", &t);
        }
        if matches!(only, Only::All | Only::C) {
            let t = Instant::now();
            ok &= free(m, &man, &toks)?;
            elapsed("(c)", &t);
        }
        if matches!(only, Only::All | Only::T) {
            let mut ties = 0usize;
            let mut ran = Vec::new();
            for set in [
                (STEP4, &IK),
                (STEP4_EVERY_NODE, &IK),
                (D1K, &IK),
                (D4096, &IK),
            ] {
                if sets.takes(set.0) {
                    let t = Instant::now();
                    ok &= step_set(m, set, &mut ties)?;
                    elapsed(&format!("(t) {}", set.0), &t);
                    ran.push(set.0);
                }
            }
            println!(
                "step sets ({}): {} ran, {ties} named tie(s)",
                sets.name(),
                ran.join(" ")
            );
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
