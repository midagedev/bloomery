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
//! - (b) the batch feed: each long step set ([`D1K`](refset::arch::mimo2::D1K),
//!   [`D4096`](refset::arch::mimo2::D4096)) fed by batches
//!   ([`bloomery_gpu_mimo2::prefill`]: up to a union's columns a batch, the
//!   window layers' flash over a whole batch and the full layers' in chunks),
//!   from a reset, then the step: the prefill's argmax, the step's logits and
//!   the step's layer outputs equal those of the set fed by our steps bit for
//!   bit — (t)'s run, or its own when (b) runs without (t). The 1,024-position
//!   prefill is two batches and the 4,096-position one eight, so the cut at a
//!   batch's end and a flash over a window of real keys are both on the path.
//!
//! - (slots) resident sequence slots, on loads of their own at [`SLOT_CTX`]
//!   positions after the main load is dropped: the slot harness's contracts
//!   (`slots_gate`: H1 interleave — two streams, each run solo and then
//!   together on their own slots with a select between every token, every
//!   slot's ids, last logits, store digest and position its solo run's —, H3
//!   bytes, H4 reset, H5 poison, H6 refusals, H7 captures), through
//!   [`MimoSlots`]; and (sp): a sequence past the plan's count refused by
//!   name, the plan's kv class of two slots twice the stores of one slot's
//!   (and the descriptor's bytes), and a load of two sequences by a plan of
//!   one refused by name. FAIL-first: a `swap_seq` that exchanges nothing
//!   leaves both slots on the live stores and H1 is red; a `seq_bytes` of 0
//!   is red at H3.
//!
//! `--only s|p|c|t|b|slots` runs one clause on the load (`slots` on its own
//! loads); `--step-sets short` takes
//! (t)'s two 4-token sets only, `--step-sets long` the 1,024- and
//! 4,096-position sets only, `--step-sets d1k` the 1,024-position one,
//! `--step-sets all` (the default) all four; it goes with `--only t`, `--only
//! b` or no `--only`, and (b) takes the long sets among those it names.
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
#[path = "shared/mimo2_open.rs"]
mod mimo2_open;

#[cfg(feature = "mimo2")]
mod gate {
    use crate::e2e::{elapsed, ik_last, layer_rels, layer_table, same_bits, set_open, word_after};
    use crate::mimo2_open::{N_VOCAB, Opened, last_argmax, open};
    use std::num::NonZeroUsize;
    use std::path::Path;
    use std::time::Instant;

    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::slots_gate::{self, Derived, SlotsAdapter};
    use bloomery_gpu_gates::{Fnv1a64, GateError, RefManifest, checks_failed, data_dir, verdict};
    use bloomery_gpu_mimo2::{Body, Mimo2Model, PrefillMode, feed, set_prefill, set_taps};
    use bloomery_levers::CARD_BUDGET;
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::mimo2::place::PlanInputs;
    use model::arch::mimo2::program::{AttnArgs, step_launches};
    use model::arch::models::Mixer;
    use model::placement::PlanLevers;
    use refset::arch::mimo2::{BATCH, D1K, D4096, IK, MODEL, STEP4, STEP4_EVERY_NODE};
    use refset::family::Family;
    use runtime::layer::FfnKind;

    /// Cache rows: the 4,096-position set's step at position 4,096, with a
    /// margin of whole 64 positions.
    const CTX: usize = 4160;

    /// The model's width and layer count.
    const HIDDEN: usize = 4096;
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

    /// A set's prefill fed one way and its step after it: the prefill's
    /// argmax, the step's layer outputs and its logits.
    struct Fed {
        prefill_argmax: u32,
        taps: Vec<f32>,
        logits: Vec<f32>,
    }

    /// `prefill` fed by our steps from a reset, then the step of `tok`.
    fn fed_by_steps(m: &mut Mimo2Model, prefill: &[u32], tok: u32) -> Result<Fed, GateError> {
        m.reset()?;
        let prefill_argmax = if prefill.is_empty() {
            0
        } else {
            m.step(prefill)?
        };
        let (taps, logits) = run_last(m, tok)?;
        Ok(Fed {
            prefill_argmax,
            taps,
            logits,
        })
    }

    /// The step of set `name` after its prefill fed by our steps; its argmax
    /// against ik's, a tie named and counted; its layer outputs printed. The
    /// run, which (b) holds the batch feed to.
    fn step_set(
        m: &mut Mimo2Model,
        (name, family): (&str, &Family),
        ties: &mut usize,
    ) -> Result<(bool, Fed), GateError> {
        let (man, pos, tok, prefill, _) = set_open((name, family), None)?;
        taps_in_set(&man)?;
        let t = Instant::now();
        let fed = fed_by_steps(m, &prefill, tok)?;
        let (taps, logits) = (&fed.taps, &fed.logits);
        let ik = ik_last(&man, N_VOCAB)?;
        let rels = layer_rels(&man, std::slice::from_ref(taps), HIDDEN, N_LAYER)?;
        let input_rel = rels.last().map_or(f64::INFINITY, |r| r.0);
        for (l, &(e, _)) in rels.iter().enumerate() {
            println!("{name} layer={l} l_out_rel={e:.3e}");
        }
        println!(
            "step {name}: logits digest {:016x} (FNV-1a over the f32 bits)",
            Fnv1a64::default().f32s(logits).value()
        );
        let (ok, tie) = last_argmax(&format!("step {name}"), (logits, &ik), input_rel);
        *ties += usize::from(tie);
        println!(
            "step {name}: position {pos} after {} fed ({:.1} s, runtime value); worst \
             l_out_rel={:.3e} (printed, no band) {}",
            prefill.len(),
            t.elapsed().as_secs_f64(),
            rels.iter().map(|r| r.0).fold(0.0, f64::max),
            verdict(ok)
        );
        Ok((ok, fed))
    }

    // -------------------------------------------------------- (b) batch feed

    /// The step of set `name` after its prefill fed by batches from a reset,
    /// held bit for bit to `steps`, the same set fed by our steps: the
    /// prefill's argmax, the step's logits and its layer outputs.
    fn batch_set(
        m: &mut Mimo2Model,
        (name, family): (&str, &Family),
        steps: Option<Fed>,
    ) -> Result<bool, GateError> {
        let (_, pos, tok, prefill, _) = set_open((name, family), None)?;
        let steps = match steps {
            Some(f) => f,
            None => {
                let t = Instant::now();
                let f = fed_by_steps(m, &prefill, tok)?;
                println!(
                    "batch {name}: the steps' run, {:.1} s (runtime value)",
                    t.elapsed().as_secs_f64()
                );
                f
            }
        };
        m.reset()?;
        let t = Instant::now();
        let prefill_argmax = bloomery_gpu_mimo2::prefill(m, &prefill)?;
        let fed_s = t.elapsed().as_secs_f64();
        if let Some(b) = bloomery_gpu_mimo2::prompt_bytes(m)? {
            println!(
                "batch {name}: the batch's buffers take {} B over {} unit(s), and the card has {} B \
                 free after them (runtime value)",
                b.total, b.units, b.free
            );
        }
        let (taps, logits) = run_last(m, tok)?;
        let argmax_ok = prefill_argmax == steps.prefill_argmax;
        let logits_ok = same_bits(&logits, &steps.logits);
        let taps_ok = same_bits(&taps, &steps.taps);
        let ok = argmax_ok && logits_ok && taps_ok;
        println!(
            "batch {name}: position {pos} after {} fed by batches ({fed_s:.1} s, runtime value); \
             the prefill's argmax {prefill_argmax} (steps {}); logits bit for bit {logits_ok}; \
             layer outputs bit for bit {taps_ok} {}",
            prefill.len(),
            steps.prefill_argmax,
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

    // ------------------------------------------------------- (slots) slots

    /// Cache rows of the slots' load: a slot's positions, over the longest
    /// stream (a prompt, [`SLOT_STEPS`] steps and the harness's tails).
    const SLOT_CTX: usize = 256;

    /// Greedy steps and continuation steps of a slot's stream
    /// ([`SlotsAdapter::STEPS`], [`SlotsAdapter::TAIL`]).
    const SLOT_STEPS: usize = 12;
    const SLOT_TAIL: usize = 6;

    /// The (slots) body for the slot harness ([`slots_gate`]): the load at
    /// [`SLOT_CTX`] positions on the gate placement, its plan counting the
    /// slots it serves ([`PlanInputs::plan_with_slots`]), stream `i`'s prompt
    /// `prompts[i]` fed as the server feeds it ([`server_start`]).
    struct MimoSlots<'a> {
        levers: &'a bloomery_levers::Levers,
        inputs: PlanInputs,
        prompts: [Vec<u32>; slots_gate::STREAMS],
    }

    impl MimoSlots<'_> {
        /// The plan of `slots` sequences on `machine` the load runs by.
        fn plan<'p>(
            &'p self,
            machine: &'p model::placement::Machine,
            slots: NonZeroUsize,
        ) -> Result<model::placement::Plan<'p>, GateError> {
            let place = PlanLevers::from_levers(self.levers)?;
            Ok(self
                .inputs
                .plan_with_slots(machine, SLOT_CTX as u64, &place, slots)?)
        }
    }

    impl SlotsAdapter for MimoSlots<'_> {
        type Body = Body;

        const STEPS: usize = SLOT_STEPS;
        const TAIL: usize = SLOT_TAIL;

        fn open(&self, slots: usize) -> Result<Mimo2Model, GateError> {
            let n = NonZeroUsize::new(slots).ok_or("(slots) a load of no slot")?;
            let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
            let machine = crate::gate_card::plan_gate(self.inputs.model.layers);
            let plan = self.plan(&machine, n)?;
            let t = Instant::now();
            let mut m =
                Body::open_placed_slots(file, &plan, &self.inputs, 0, self.levers.host(), n)?;
            set_prefill(&mut m, PrefillMode::Batch)?;
            println!(
                "slots load resident_bytes={} ctx={SLOT_CTX} slots={slots} in {:.1} s (runtime \
                 value)",
                m.resident_bytes(),
                t.elapsed().as_secs_f64()
            );
            Ok(m)
        }

        /// The reset, and every store zeroed: a reset keeps the rows past
        /// the position (a position is written before a later one reads
        /// it), which [`SlotsAdapter::state_hash`] reads whole, so no other
        /// stream's rows may stand there.
        fn rewind(&self, m: &mut Mimo2Model) -> Result<(), GateError> {
            m.reset()?;
            let (gpu, _, body) = m.body_parts("gate_mimo2_e2e slots")?;
            body.clear_stores(gpu)?;
            gpu.stream().synchronize()?;
            Ok(())
        }

        fn prompt(&self, m: &mut Mimo2Model, stream: usize) -> Result<u32, GateError> {
            let p = self
                .prompts
                .get(stream)
                .ok_or_else(|| format!("(slots) runs streams 0 and 1, not {stream}"))?;
            server_start(m, p)
        }

        /// Every layer's key and value planes of the selected slot, every
        /// row of them.
        fn state_hash(&self, m: &mut Mimo2Model) -> Result<u64, GateError> {
            let (gpu, _, body) = m.body_parts("gate_mimo2_e2e slots")?;
            let mut h = Fnv1a64::default();
            for plane in body.store_planes(gpu)? {
                let bytes: Vec<u8> = plane.iter().flat_map(|v| v.to_le_bytes()).collect();
                h = h.bytes(&bytes);
            }
            Ok(h.value())
        }

        /// The stores alone: the sequence holds nothing else, so it is the
        /// header's bytes a position ([`KV_BYTES_PER_POSITION`]) at
        /// [`SLOT_CTX`] positions.
        fn seq_bytes_derived(&self, _m: &Mimo2Model) -> Result<Derived, GateError> {
            let bytes = KV_BYTES_PER_POSITION * SLOT_CTX;
            Ok(Derived {
                bytes,
                terms: format!(
                    "the stores {SLOT_CTX} positions of {KV_BYTES_PER_POSITION} B (derived from \
                     the header), nothing beside them"
                ),
            })
        }

        /// The plan's sequence descriptor ([`PlanInputs::seq_terms`]).
        fn seq_terms_bytes(&self, _m: &Mimo2Model) -> Result<Option<usize>, GateError> {
            Ok(Some(usize::try_from(
                self.inputs.seq_terms().bytes(SLOT_CTX as u64, 1),
            )?))
        }

        /// H5's planter ([`SlotsAdapter::plant_refusal`]): the tier through
        /// the body's own mut path.
        fn plant_refusal(&self, m: &mut Mimo2Model) -> Result<bool, GateError> {
            m.body_parts("gate_mimo2_e2e slots")?
                .2
                .hybrid_mut()
                .plant_refusal("a planted refusal (the slots harness's seam)");
            Ok(true)
        }

        /// H5's window: the tier's own refusal, read through the body's
        /// tier.
        fn tier_poisoned(&self, m: &mut Mimo2Model) -> Result<bool, GateError> {
            Ok(m.body_parts("gate_mimo2_e2e slots")?
                .2
                .hybrid()
                .refuse_if_poisoned("slots H5")
                .is_err())
        }
    }

    /// The prompt as the server feeds it: every id but the last in one call
    /// by the load's feed ([`feed`]), then the last one step; the step's
    /// argmax.
    fn server_start(m: &mut Mimo2Model, p: &[u32]) -> Result<u32, GateError> {
        let (&last, head) = p.split_last().ok_or("an empty prompt")?;
        feed(m, head)?;
        Ok(m.step(&[last])?)
    }

    /// (sp) the plan's count: a sequence past it refused by name, the plan
    /// of [`slots_gate::STREAMS`] counting [`slots_gate::STREAMS`] times the
    /// stores of the plan of one (and the descriptor's bytes), and a load
    /// of a plan that counts another number refused by name. Moves no slot;
    /// the harness's model is the caller's to have dropped before the last.
    fn slots_plan(a: &MimoSlots<'_>, m: &mut Mimo2Model) -> Result<bool, GateError> {
        const N: usize = slots_gate::STREAMS;
        let past = m.add_slots(N + 1);
        let named = matches!(&past, Err(e) if e.to_string().contains(&format!(
            "of a plan that counts {N} resident sequences"
        )));
        let machine = crate::gate_card::plan_gate(a.inputs.model.layers);
        let one = NonZeroUsize::MIN;
        let many = NonZeroUsize::new(N).ok_or("no slots")?;
        let kv = |n| -> Result<u64, GateError> { Ok(a.plan(&machine, n)?.cards[0].kv_bytes) };
        let (kv_one, kv_many) = (kv(one)?, kv(many)?);
        let seq = a.inputs.seq_terms().bytes(SLOT_CTX as u64, 1);
        let counted = kv_one == seq && kv_many == N as u64 * seq;
        println!(
            "(sp) a sequence past the plan's {N} refused by name {named}; the plan's kv class \
             {kv_one} B for one slot and {kv_many} B for {N} (one sequence's descriptor {seq} B): \
             {counted} {}",
            verdict(named && counted)
        );
        Ok(named && counted)
    }

    /// (sp)'s load of a plan that counts one sequence as a load of two: the
    /// plan's kv class is another count's, refused by name at the load.
    fn slots_other_plan(a: &MimoSlots<'_>) -> Result<bool, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let machine = crate::gate_card::plan_gate(a.inputs.model.layers);
        let plan = a.plan(&machine, NonZeroUsize::MIN)?;
        let two = NonZeroUsize::new(slots_gate::STREAMS).ok_or("no slots")?;
        let refused = Body::open_placed_slots(file, &plan, &a.inputs, 0, a.levers.host(), two);
        let named = matches!(&refused, Err(e) if e.to_string().contains("the plan counts"));
        println!(
            "(sp) a load of {} sequences by a plan of one is refused by name {named} {}",
            slots_gate::STREAMS,
            verdict(named)
        );
        Ok(named)
    }

    /// The (slots) clauses: the harness's contracts and (sp), each on a load
    /// at [`SLOT_CTX`] positions of its own, the main load dropped first.
    fn slots(levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let inputs = PlanInputs::read(&file)?;
        drop(file);
        // Two streams of distinct ids and lengths: the batch set's five
        // tokens, and the head of the 1,024-position set's prompt.
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, a, _) = man.step()?;
        let (_, _, _, long, _) = set_open((D1K, &IK), None)?;
        let b = long.get(..21).ok_or("the 1,024-position prompt is short")?;
        let body = MimoSlots {
            levers,
            inputs,
            prompts: [a.to_vec(), b.to_vec()],
        };
        let t = Instant::now();
        let mut s = slots_gate::interleave(&body)?;
        elapsed("(slots) harness open (H1, H3, H6, H7)", &t);
        let mut ok = slots_plan(&body, s.model())?;
        let t = Instant::now();
        ok &= s.finish()?;
        elapsed("(slots) harness finish (H4, H5)", &t);
        ok &= slots_other_plan(&body)?;
        Ok(ok)
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
        B,
        /// The resident slots' clauses ([`slots`]), on loads of their own.
        Slots,
    }

    /// Which step sets (t) and (b) take.
    #[derive(Clone, Copy)]
    enum StepSets {
        All,
        /// [`STEP4`] and [`STEP4_EVERY_NODE`].
        Short,
        /// [`D1K`] and [`D4096`].
        Long,
        /// [`D1K`] alone.
        D1k,
    }

    impl StepSets {
        fn name(self) -> &'static str {
            match self {
                StepSets::All => "all",
                StepSets::Short => "short",
                StepSets::Long => "long",
                StepSets::D1k => "d1k",
            }
        }

        /// Whether the set is taken: the 4-token sets are the short ones.
        fn takes(self, name: &str) -> bool {
            let short = name == STEP4 || name == STEP4_EVERY_NODE;
            match self {
                StepSets::All => true,
                StepSets::Short => short,
                StepSets::Long => !short,
                StepSets::D1k => name == D1K,
            }
        }
    }

    /// `--only s|p|c|t|b|slots`, or every clause.
    fn only() -> Result<Only, GateError> {
        match word_after("--only") {
            None => Ok(Only::All),
            Some(w) => match w.as_deref() {
                Some("s") => Ok(Only::S),
                Some("p") => Ok(Only::P),
                Some("c") => Ok(Only::C),
                Some("t") => Ok(Only::T),
                Some("b") => Ok(Only::B),
                Some("slots") => Ok(Only::Slots),
                other => Err(format!("--only is s, p, c, t, b or slots, not {other:?}").into()),
            },
        }
    }

    /// `--step-sets short|long|d1k|all`, `all` when absent; refused beside an
    /// `--only` that runs neither (t) nor (b).
    fn step_sets(only: Only) -> Result<StepSets, GateError> {
        let sets = match word_after("--step-sets") {
            None => return Ok(StepSets::All),
            Some(w) => match w.as_deref() {
                Some("all") => StepSets::All,
                Some("short") => StepSets::Short,
                Some("long") => StepSets::Long,
                Some("d1k") => StepSets::D1k,
                other => {
                    return Err(
                        format!("--step-sets is short, long, d1k or all, not {other:?}").into(),
                    );
                }
            },
        };
        if !matches!(only, Only::All | Only::T | Only::B) {
            return Err(
                "--step-sets names the step sets of (t) and (b): it goes with --only t, --only b \
                 or no --only"
                    .into(),
            );
        }
        Ok(sets)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[CARD_BUDGET])?;
        crate::gate_card::init()?;
        let only = only()?;
        let sets = step_sets(only)?;
        if only == Only::Slots {
            let t = Instant::now();
            let ok = slots(&levers)?;
            elapsed("(slots)", &t);
            return if ok { Ok(()) } else { Err(checks_failed()) };
        }
        let (mut s, opened) = open(Path::new(MODEL), &levers, CTX)?;
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
        let mut by_steps: Vec<(&str, Fed)> = Vec::new();
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
                    let (set_ok, fed) = step_set(m, set, &mut ties)?;
                    ok &= set_ok;
                    by_steps.push((set.0, fed));
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
        if matches!(only, Only::All | Only::B) {
            let mut ran = Vec::new();
            for set in [(D1K, &IK), (D4096, &IK)] {
                if sets.takes(set.0) {
                    let t = Instant::now();
                    let steps = by_steps
                        .iter()
                        .position(|(n, _)| *n == set.0)
                        .map(|i| by_steps.swap_remove(i).1);
                    ok &= batch_set(m, set, steps)?;
                    elapsed(&format!("(b) {}", set.0), &t);
                    ran.push(set.0);
                }
            }
            if ran.is_empty() && only == Only::B {
                return Err(format!(
                    "--only b takes the long step sets, and --step-sets {} names none",
                    sets.name()
                )
                .into());
            }
            println!("batch feed ({}): {} ran", sets.name(), ran.join(" "));
        }
        if only == Only::All {
            // The slots' loads take the card with the main one dropped.
            drop(s);
            let t = Instant::now();
            ok &= slots(&levers)?;
            elapsed("(slots)", &t);
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
