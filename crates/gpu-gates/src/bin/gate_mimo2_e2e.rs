//! The MiMo-V2.6-Flash end-to-end gate: the whole program — 48 GQA attention
//! layers over two head widths (key 192, value 128; nine full layers and 39
//! window layers with sinks), the dense layer 0 and 47 routed blocks whose
//! experts run on the card where the plan puts them (the common MXFP4 pair, in
//! the host leg's shadow) and on the host tier otherwise, the q8_0 head and
//! the argmax — loaded once by its placement on the gate card
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
//!   (`step_launches` over the body's layer programs and the layers the plan
//!   puts card experts on, a fact of the file's description and the plan) is
//!   the same — [`NODES_HOST`] plus [`CARD_LAUNCHES`] for each such layer; the
//!   dense layer is layer 0 and the host run is layers `1..48`; the plan puts
//!   at least one routed expert on the card, as many as the slot map holds
//!   there and none on a tier (the plan's `n_l` and the slot map's both), on
//!   every routed layer; the stores hold
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
//!   ik's `l_out-L` is printed by position, and the worst one is held to
//!   [`FREE_WORST`], a ceiling that sees a dropped card sum; there is no
//!   finer band yet (the forced arm that derives one is a later round's).
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
//!   The open runs the prompt group of the lever's default
//!   (`Levers::prefill_group`, 2: the seat's), held to it here, and each set is
//!   fed at that group (the 1,024-position set one group of two batches, the
//!   4,096-position one four) and again at groups of one, each the steps' bits;
//!   the batch's units hold the group (`group_sets`). A call of three batches
//!   (`JOIN_IDS` ids of the 4,096-position set) is fed at groups of two, the
//!   lone last batch joined to the group before it, and at groups of one: the
//!   argmax, the step's logits and its layer outputs bit for bit the same.
//!   FAIL-first: an open that pins the group to 1 prints `the open's prompt
//!   group is 1, the lever's default 2` red; a `group_sets` that holds `g`
//!   units for a group of `g` refuses the call of three batches by name.
//!
//! - (slots) resident sequence slots, on loads of their own at [`SLOT_CTX`]
//!   positions after the main load is dropped: the slot harness's contracts
//!   (`slots_gate`: H1 interleave — two streams, each run solo and then
//!   together on their own slots with a select between every token, every
//!   slot's ids, last logits, store digest and position its solo run's —, H2
//!   the pass of several slots' rows against the slots stepped one at a time
//!   (the launches it adds a slot are [`SEQ_BOUND`] a layer, the body's own
//!   counts held), H3 bytes, H4 reset, H5 poison, H6 refusals, H7 captures),
//!   through [`MimoSlots`]; the captured nodes of a pass of one row (the
//!   step's), of two rows of one slot and of one row on each of two slots
//!   (`pass_nodes`: the step's nodes plus a slot's extra launches plus, for
//!   the second row, one router launch and one handoff in each routed layer);
//!   and (sp): a sequence past the plan's count refused by
//!   name, the plan's kv class of two slots twice the stores of one slot's
//!   (and the descriptor's bytes), and a load of two sequences by a plan of
//!   one refused by name. FAIL-first: a `swap_seq` that exchanges nothing
//!   leaves both slots on the live stores and H1 is red; a `seq_bytes` of 0
//!   is red at H3; a pass that binds its attention to the live stores for
//!   every slot is red at H2.
//!
//! - (r) adaptive expert residency, the common machine on MiMo's slot map
//!   (`bloomery_gpu::host::swap`, the stack table `bloomery_gpu_mimo2::swap`):
//!   off, the load of the clauses above runs no machine (`residency_source`
//!   and the tier's machine unset: its slot map is the load's for its life);
//!   unset, the serving default on this plan is the machine
//!   (`bloomery_levers::residency_unset`, `residency::at_plan`: the seat's own
//!   two calls; its `residency lever` record printed), and off beside `--place
//!   gate` and the steps feed; on, a load of its own at the word the unset
//!   resolved to, fed in batches: every routed layer's three parts are an
//!   MXFP4 expert's bytes ([`PART_BYTES`], read back through the source), a
//!   history of [`R_PROMPT`] prompt ids and [`R_STEPS`] greedy steps run twice
//!   from a clear gives the same tokens and logits, its boundaries end the
//!   prompt (0 rows kept) and each step (1 kept), and flips land
//!   (`residency_clauses`: c1, passes); the batch set's free-running tokens on
//!   the map use has built are (c)'s, in its band; and the reset brings the
//!   live sets back to the seed (c7) and the slot map to where a clear left
//!   it. FAIL-first: `Body::residency_parts` of `None` leaves the machine
//!   unstarted (the structure line, c1 and c7 red); a history that counts no
//!   flip is red at c1; a reset that does nothing is red at the map's return.
//! - (rp) the pass of two slots under the machine (`residency_pass`): a load
//!   of two sequences at the serving word runs [`R_STEPS`] rounds through
//!   `step_slots` and the same rounds a slot at a time; the logits and ids are
//!   equal bit for bit until the first flip lands, after it the ids are equal
//!   until a round where the pass's id trails the turns' by less than
//!   [`MARGIN_FLOOR`] in the turns' logits (a near tie: the two arms' maps
//!   hold other experts on the card) and the logits are inside
//!   [`FREE_WORST`] through that round, a second pass repeats the first, and
//!   flips land. FAIL-first: the load's tally rows left at 1 refuse the pass
//!   by name at the first round (`Tally::note`) and the clause is red; the
//!   pass's id at the first round past the bit-for-bit ones replaced by the
//!   turns' least likely id is a first difference at a confident gap and the
//!   clause is red.
//!
//! `--only s|p|c|t|b|slots|r` runs one clause on the load (`slots` and `r` on
//! their own loads); `--step-sets short` takes
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
#[allow(
    dead_code,
    reason = "the residency clauses serve several gates; each takes the helpers it reads"
)]
#[path = "shared/residency_clauses.rs"]
mod residency_clauses;

#[cfg(feature = "mimo2")]
mod gate {
    use crate::e2e::{elapsed, ik_last, layer_rels, layer_table, same_bits, set_open, word_after};
    use crate::mimo2_open::{N_VOCAB, Opened, last_argmax, open};
    use std::num::NonZeroUsize;
    use std::path::Path;
    use std::time::Instant;

    use app::Session;
    use bloomery_gpu::GpuError;
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::swap::{Residency, SwapSource};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::prompts::MARGIN_FLOOR;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::residency;
    use bloomery_gpu_gates::slots_gate::{self, Derived, Launches, PassAdapter, SlotsAdapter};
    use bloomery_gpu_gates::{Fnv1a64, GateError, RefManifest, checks_failed, data_dir, verdict};
    use bloomery_gpu_mimo2::{
        Body, Mimo2Model, PrefillMode, T_MAX, feed, prefill_group, prompt_bytes, set_prefill,
        set_prefill_group, set_taps,
    };
    use bloomery_levers::{CARD_BUDGET, ResidencyAt, ResidencyWhy};
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::mimo2::place::PlanInputs;
    use model::arch::mimo2::program::{AttnArgs, step_launches};
    use model::arch::models::Mixer;
    use model::placement::PlanLevers;
    use model::placement::workstation::{HostRead, PLAN_ROOM};
    use refset::arch::mimo2::{BATCH, D1K, D4096, IK, MODEL, STEP4, STEP4_EVERY_NODE};
    use refset::family::Family;
    use runtime::layer::FfnKind;
    use runtime::prompt::{call_batches, group_sets, groups};
    use runtime::{Out, Target as _, Want};

    use crate::mimo2_open::open_with;
    use crate::residency_clauses::{c1_clause_with, c7_clause, passes_clause, seeds};

    /// Cache rows: the 4,096-position set's step at position 4,096, with a
    /// margin of whole 64 positions.
    const CTX: usize = 4160;

    /// The model's width and layer count.
    const HIDDEN: usize = 4096;
    const N_LAYER: usize = 48;

    /// PIN(2026-10-08): the captured decode step's node count with every
    /// routed expert on the host, derived before the chain was built: every
    /// layer's attention takes 7 launches (the norm, the fused q·k·v gemv, the
    /// rope-and-append, the flash's segment pass and its merge, the output
    /// gemv, the add); the dense block 4 (norm, gate·up, down, add); a routed
    /// block 6 (norm, router, handoff, the go, the wait, the add); the head 3
    /// (norm, q8_0 gemv, argmax): 48·7 + 4 + 47·6 + 3.
    const NODES_HOST: usize = 625;

    /// PIN(2026-10-10): a routed layer with card experts adds its shadow's six
    /// launches (the norm's q8_1, the gate·up, the q8_1 of the card columns,
    /// the down, the card sum, the add of `x1`) and no memory operation.
    const CARD_LAUNCHES: usize = 6;

    /// PIN(2026-10-10): the captured decode step's node count at the gate
    /// plan, which puts card experts on every routed layer (its prefixes
    /// spread evenly over the 47, the card's budget holding far more than one
    /// each): [`NODES_HOST`] + [`CARD_LAUNCHES`]·47 = 625 + 6·47.
    const NODES_DECODE: usize = 907;

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

    /// [`PlanInputs::read`] of `file` at this workstation's plan room with
    /// nothing loaded ([`PLAN_ROOM`], read [`HostRead::Given`]): no reading of
    /// the box's available bytes enters a plan this gate holds to a pin, and
    /// the plan is the unpaged one the pins were taken at.
    fn read_inputs(file: &Split) -> Result<PlanInputs, GateError> {
        let mut inputs = PlanInputs::read(file)?;
        inputs.room = (PLAN_ROOM, HostRead::Given);
        Ok(inputs)
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
        let inputs = read_inputs(&file)?;
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
        // Card experts: the plan's count and the slot map's, on every routed
        // layer, none on a tier.
        let slots = body.hybrid().slots();
        let (mut on_card, mut on_tier, mut bare) = (0usize, 0usize, Vec::new());
        for l in body.host_run() {
            let here = slots.on_card(l)?;
            on_card += here;
            on_tier += slots.on_tier(l)?;
            if here == 0 {
                bare.push(l);
            }
        }
        let planned: u64 = o.n_l.iter().sum();
        let card_ok = on_card > 0
            && on_tier == 0
            && planned == on_card as u64
            && bare.is_empty()
            && o.n_l.iter().skip(1).all(|&n| n > 0);
        println!(
            "structure card experts: the plan's n_l sums to {planned}, the slot map holds \
             {on_card} on the card and {on_tier} on a tier (want at least 1 on the card, the \
             plan's count, 0 on a tier), routed layers with none on the card {bare:?} (want \
             none) {}",
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
        let blocks = body.blocks();
        let card_layers = blocks.iter().filter(|(_, card)| *card).count();
        // The pins are the q8_0 dense block's; a K-quant block's extra launches are the plan's.
        let block_extra: usize = body.block_plans().iter().map(|p| p.extra_launches()).sum();
        let want = NODES_DECODE + block_extra;
        let counted = step_launches(&blocks) + block_extra;
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        let nodes = k + b + other;
        let derived = NODES_HOST + block_extra + CARD_LAUNCHES * card_layers;
        let pass = nodes == want
            && derived == want
            && counted == want
            && b == MEMOPS
            && k == want - MEMOPS
            && other == 0;
        println!(
            "structure decode graph_nodes={nodes} (want {want} = {NODES_HOST} + {block_extra} \
             block launches + {CARD_LAUNCHES} · {card_layers} layers with card experts; the \
             program counts {counted}) kernel={k} batch_mem_op={b} (want {MEMOPS}) \
             other={other} {}",
            verdict(pass)
        );
        Ok(ok && card_ok && args_ok && pass)
    }

    /// PIN(2026-10-10): the ceiling of the batch set's worst layer-output
    /// distance from ik's (`l_out_rel`, a distance over the output's own
    /// size): 1.0, past which an output is no nearer ik's than zero is. The
    /// distance reads 1.669e-1 (layer 42) with the card experts on; with the
    /// shadow's `acc + x1` replaced by `x1` (the card slots' sum dropped) it
    /// reads 1.340e1 while the argmax stays ik's runner-up inside the tie
    /// rule, so this ceiling is what sees the card sum.
    const FREE_WORST: f64 = 1.0;

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
        let (argmax_ok, _) = last_argmax("free", (ours, &ik), input_rel);
        let band_ok = worst.0 <= FREE_WORST;
        println!(
            "free: {} tokens, worst l_out_rel={:.3e} at layer {} (want at most {FREE_WORST}); last \
             position {}",
            eager.tokens.len(),
            worst.0,
            worst.1,
            verdict(argmax_ok && band_ok)
        );
        Ok(argmax_ok && band_ok)
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

    /// `prefill` fed by batches from a reset, then the step of `tok`: the
    /// prefill's argmax, the step's layer outputs and its logits, and the
    /// wall of the prefill.
    fn fed_by_batches(
        m: &mut Mimo2Model,
        prefill: &[u32],
        tok: u32,
    ) -> Result<(Fed, f64), GateError> {
        m.reset()?;
        let t = Instant::now();
        let prefill_argmax = bloomery_gpu_mimo2::prefill(m, prefill)?;
        let fed_s = t.elapsed().as_secs_f64();
        let (taps, logits) = run_last(m, tok)?;
        Ok((
            Fed {
                prefill_argmax,
                taps,
                logits,
            },
            fed_s,
        ))
    }

    /// The step of set `name` after its prefill fed by batches from a reset,
    /// held bit for bit to `steps`, the same set fed by our steps: the
    /// prefill's argmax, the step's logits and its layer outputs. The open's
    /// prompt group is the lever's default, which the seat runs (held to it
    /// here); the set is fed at that group and again at groups of one, each
    /// the steps' bits, so the 1,024-position set is one group of two
    /// batches and the 4,096-position one four.
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
        let default = usize::try_from(bloomery_levers::PREFILL_GROUP_DEFAULT)?;
        let held = prefill_group(m)?;
        let mut ok = held == default;
        println!(
            "batch {name}: the open's prompt group is {held}, the lever's default {default} {}",
            verdict(ok)
        );
        for g in [default, 1] {
            set_prefill_group(m, g)?;
            let (fed, fed_s) = fed_by_batches(m, &prefill, tok)?;
            let units = prompt_bytes(m)?.map(|b| {
                println!(
                    "batch {name}: the batch's buffers take {} B over {} unit(s), and the card has \
                     {} B free after them (runtime value)",
                    b.total, b.units, b.free
                );
                b.units
            });
            let argmax_ok = fed.prefill_argmax == steps.prefill_argmax;
            let logits_ok = same_bits(&fed.logits, &steps.logits);
            let taps_ok = same_bits(&fed.taps, &steps.taps);
            let units_ok = units.is_some_and(|u| u >= group_sets(g));
            let arm_ok = argmax_ok && logits_ok && taps_ok && units_ok;
            println!(
                "batch {name}: position {pos} after {} fed by batches at groups of {g} ({fed_s:.1} \
                 s, runtime value); the prefill's argmax {} (steps {}); logits bit for bit \
                 {logits_ok}; layer outputs bit for bit {taps_ok}; units {units:?} hold the group \
                 ({} at least) {units_ok} {}",
                prefill.len(),
                fed.prefill_argmax,
                steps.prefill_argmax,
                group_sets(g),
                verdict(arm_ok)
            );
            ok &= arm_ok;
        }
        set_prefill_group(m, held)?;
        Ok(ok)
    }

    /// Positions of (b)'s call of three batches: a group of three at groups
    /// of two, where the call's lone last batch joins the group before it
    /// (`runtime::prompt::groups`) — a call of 1,025 to 1,536 positions, cut
    /// by `runtime::prompt::call_batches` at [`T_MAX`], on the path of every
    /// prompt of an odd count of batches.
    const JOIN_IDS: usize = 1300;

    /// A call of [`JOIN_IDS`] ids of the 4,096-position set fed at groups of
    /// two (one group of three batches) and at groups of one from a reset,
    /// each followed by the step of the next id: the prefill's argmax, the
    /// step's logits and its layer outputs bit for bit the same.
    fn join_group(m: &mut Mimo2Model) -> Result<bool, GateError> {
        let (_, _, _, ids, _) = set_open((D4096, &IK), None)?;
        let end = u32::try_from(JOIN_IDS)?;
        let cut = call_batches(0, end, &[end], T_MAX);
        let walks = groups(cut.len(), 2);
        let premise = cut.len() == 3 && walks.len() == 1 && walks[0].len() == 3;
        println!(
            "join: {JOIN_IDS} ids cut into {} batches, walked at groups of two as {} group(s) of \
             {:?} batches {}",
            cut.len(),
            walks.len(),
            walks.iter().map(|r| r.len()).collect::<Vec<_>>(),
            verdict(premise)
        );
        let (prefix, tok) = (&ids[..JOIN_IDS], ids[JOIN_IDS]);
        let held = prefill_group(m)?;
        set_prefill_group(m, 1)?;
        let (one, one_s) = fed_by_batches(m, prefix, tok)?;
        set_prefill_group(m, 2)?;
        let (two, two_s) = fed_by_batches(m, prefix, tok)?;
        set_prefill_group(m, held)?;
        let argmax_ok = one.prefill_argmax == two.prefill_argmax;
        let logits_ok = same_bits(&one.logits, &two.logits);
        let taps_ok = same_bits(&one.taps, &two.taps);
        let ok = premise && argmax_ok && logits_ok && taps_ok;
        println!(
            "join: {JOIN_IDS} ids at groups of one ({one_s:.1} s) and of two ({two_s:.1} s, \
             runtime value); the prefill's argmax {} ({}); logits bit for bit {logits_ok}; layer \
             outputs bit for bit {taps_ok} {}",
            two.prefill_argmax,
            one.prefill_argmax,
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
        /// The residency the load runs: off for the harness's contracts, the
        /// serving word for (rp).
        residency: Residency,
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
            let mut m = Body::open_placed_slots(
                file,
                &plan,
                &self.inputs,
                0,
                self.levers.host(),
                self.residency,
                n,
            )?;
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

        /// H5's round of several slots: the body's one pass
        /// ([`GpuModel::step_slots`], this body's `SlotRows`).
        fn step_all(&self, m: &mut Mimo2Model, last: &[u32]) -> Result<Vec<u32>, GpuError> {
            let ones: Vec<[u32; 1]> = last.iter().map(|&t| [t]).collect();
            let rows: Vec<(usize, &[u32])> = ones.iter().map(|t| &t[..]).enumerate().collect();
            Ok(m.step_slots(&rows)?.ids)
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

    /// The launches an attention layer binds to one sequence's stores,
    /// whatever the rows of its slot: the rope-and-append, the flash's segment
    /// pass and its merge. Spelled here from the layer program's launch list,
    /// apart from the body's own count.
    const SEQ_BOUND: usize = 3;

    impl PassAdapter for MimoSlots<'_> {
        fn added_slot_launches(&self, m: &Mimo2Model) -> Result<Launches, GateError> {
            let layers = m.body("slots")?.kinds().len();
            Ok(Launches {
                n: SEQ_BOUND * layers,
                terms: format!(
                    "the rope-and-append 1 + the flash's segment pass 1 + its merge 1 on each of \
                     {layers} layers"
                ),
            })
        }

        fn slots_launches(
            &self,
            m: &Mimo2Model,
            key: &[(usize, usize)],
        ) -> Result<usize, GateError> {
            Ok(m.body("slots")?.slots_launches(key))
        }
    }

    /// The captured pass's node counts against the header's counts: a pass of
    /// one row is the decode step's [`NODES_DECODE`] whichever slot runs it; a
    /// pass of `m` rows over `s` busy slots is the step's nodes, plus
    /// [`SEQ_BOUND`] a layer for each busy slot past the first, plus on each
    /// routed layer the router's second launch and a handoff a column past
    /// the first (`m` in all: the router over the rows is two launches, a
    /// handoff a column one), and the dense block's columns past the step's
    /// own launches: none on a q8_0 block, whose columns share its launches,
    /// a whole block a column on a K-quant one.
    fn pass_nodes(m: &mut Mimo2Model) -> Result<bool, GateError> {
        let body = m.body("slots nodes")?;
        let blocks = body.blocks();
        let plans = body.block_plans();
        let block_extra: usize = plans.iter().map(|p| p.extra_launches()).sum();
        let card = blocks.iter().filter(|b| b.1).count();
        let routed = blocks.iter().filter(|b| b.0 == FfnKind::Moe).count();
        let step_nodes = NODES_DECODE + block_extra;
        let step = NODES_HOST + block_extra + CARD_LAUNCHES * card;
        let derived = |key: &[(usize, usize)]| -> usize {
            let rows: usize = key.iter().map(|&(_, r)| r).sum();
            if rows == 1 {
                step
            } else {
                let dense: usize = plans
                    .iter()
                    .map(|p| p.column_launches(rows) - 2 - p.extra_launches())
                    .sum();
                step + N_LAYER * SEQ_BOUND * (key.len() - 1) + rows * routed + dense
            }
        };
        let mut ok = step == step_nodes;
        let mut line = String::new();
        for key in [&[(0, 1)][..], &[(1, 1)], &[(0, 2)], &[(0, 1), (1, 1)]] {
            let got = m.capture_slots(key)?;
            let own = m.body("slots nodes")?.slots_launches(key);
            let want = derived(key);
            ok &= got == want && own == want;
            line.push_str(&format!(
                " {key:?} {got} nodes (derived {want}, body {own});"
            ));
        }
        println!(
            "check slots nodes:{line} the step {step_nodes}, {card} card layers and {routed} \
             routed {}",
            verdict(ok)
        );
        Ok(ok)
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
        let refused = Body::open_placed_slots(
            file,
            &plan,
            &a.inputs,
            0,
            a.levers.host(),
            Residency::Off,
            two,
        );
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
        let inputs = read_inputs(&file)?;
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
            residency: Residency::Off,
        };
        let t = Instant::now();
        let mut s = slots_gate::interleave(&body)?;
        elapsed("(slots) harness open (H1, H3, H6, H7)", &t);
        s.one_pass();
        let mut ok = slots_plan(&body, s.model())?;
        ok &= pass_nodes(s.model())?;
        let t = Instant::now();
        ok &= s.finish()?;
        elapsed("(slots) harness finish (H4, H5)", &t);
        ok &= slots_other_plan(&body)?;
        Ok(ok)
    }

    // ------------------------------------------------- (r) residency

    /// Prompt ids of (r)'s history (the 1,024-position set's head) and the
    /// greedy steps after it.
    const R_PROMPT: usize = 24;
    const R_STEPS: usize = 32;

    /// One routed expert's part, each of the three: a gate or up part is
    /// `EXPERT_FF` rows of `HIDDEN / 32` 17-byte `block_mxfp4`s, the down
    /// `HIDDEN` rows of `EXPERT_FF / 32`: the same bytes.
    const EXPERT_FF: usize = 2048;
    const PART_BYTES: usize = EXPERT_FF * (HIDDEN / 32) * 17;

    /// What a history saw: the prompt's argmax, every step's argmax and
    /// logits FNV, each boundary's ended pass (its kind and kept rows) and the
    /// flips that landed there, and the host map's stage view after its clear,
    /// where those flips started from.
    struct History {
        first: u32,
        tokens: Vec<u32>,
        fnvs: Vec<u64>,
        passes: Vec<(PassKind, usize)>,
        landings: Vec<usize>,
        start: Vec<u32>,
    }

    impl History {
        fn landed(&self) -> usize {
            self.landings.iter().sum()
        }

        /// The runs' contents, not where each began.
        fn same(&self, o: &History) -> bool {
            (
                &self.first,
                &self.tokens,
                &self.fnvs,
                &self.passes,
                &self.landings,
            ) == (&o.first, &o.tokens, &o.fnvs, &o.passes, &o.landings)
        }
    }

    /// The history from a clear ([`Session::clear`], the residency back to its
    /// seed): the prompt call, then [`R_STEPS`] greedy steps.
    fn history(s: &mut Session<Body>, ids: &[u32]) -> Result<History, GateError> {
        s.clear()?;
        s.model_mut()
            .body_parts("(r) history")?
            .2
            .take_residency_passes();
        let start = s.model().body("(r) history")?.hybrid().slots().stage_view();
        let mut next = s.prompt(ids, Want::Argmax)?.argmax();
        let mut h = History {
            first: next,
            tokens: Vec::with_capacity(R_STEPS),
            fnvs: Vec::with_capacity(R_STEPS),
            passes: Vec::new(),
            landings: Vec::new(),
            start,
        };
        for _ in 0..R_STEPS {
            let out = s.step(next, Want::Logits)?;
            if let Out::Logits { row, .. } = out {
                h.fnvs.push(Fnv1a64::default().f32s(row).value());
            }
            next = out.argmax();
            h.tokens.push(next);
        }
        let passes = s
            .model_mut()
            .body_parts("(r) history")?
            .2
            .take_residency_passes();
        h.landings = passes.iter().map(|(_, r)| r.landed).collect();
        h.passes = passes.iter().map(|&(k, r)| (k, r.kept)).collect();
        println!(
            "history: prompt {R_PROMPT} + {R_STEPS} steps, {} boundaries, {} flips landed",
            h.passes.len(),
            h.landed()
        );
        Ok(h)
    }

    /// (r) off: the load of the clauses above runs no machine, its slot map the
    /// load's for its life.
    fn residency_off(m: &Mimo2Model) -> Result<bool, GateError> {
        let body = m.body("(r) off")?;
        let (source, machine) = (
            body.residency_source().is_some(),
            body.hybrid().swap().is_some(),
        );
        let ok = !source && !machine;
        println!(
            "(r) off: source {source}, machine {machine} (want neither) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// (r) unset on the plan of the gate placement, the seat's own two calls
    /// ([`residency::at_plan`] over `levers.residency_at`): the serving word
    /// is the machine, and the gate placement and the steps feed are off.
    /// Returns the word the load runs and whether the clause held.
    fn residency_unset(
        levers: &bloomery_levers::Levers,
        inputs: &PlanInputs,
    ) -> Result<(Residency, bool), GateError> {
        let machine = crate::gate_card::plan_gate(inputs.model.layers);
        let plan = inputs
            .plan(&machine, CTX as u64, &PlanLevers::from_levers(levers)?)
            .map_err(|e| format!("(r) the plan: {e}"))?;
        let at = |serving_place, prefill_steps| {
            levers.residency_at(ResidencyAt {
                serving_place,
                check_finite: false,
                route_trace: false,
                prefill_steps,
            })
        };
        let serving = residency::at_plan(
            <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
            &plan,
            0,
            at(true, false),
        )?;
        record::residency_lever(serving).print();
        if let Some(d) = serving.why.detail() {
            println!("{d}");
        }
        let word = Residency::parse(serving.word)?;
        let on = matches!(word, Residency::Mid { .. })
            && matches!(
                serving.why,
                ResidencyWhy::Place | ResidencyWhy::Shrunk { .. }
            );
        let fixed = residency::at_plan(
            <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
            &plan,
            0,
            at(false, false),
        )?;
        let steps = residency::at_plan(
            <Body as bloomery_gpu::model::ChainBody>::UNHOSTED_POOL,
            &plan,
            0,
            at(true, true),
        )?;
        let off = (fixed.word, fixed.why) == ("off", ResidencyWhy::FixedPlace)
            && (steps.word, steps.why) == ("off", ResidencyWhy::PrefillSteps);
        println!(
            "(r) unset: the serving word {} ({}), beside the gate placement {} and the steps feed \
             {} {}",
            serving.word,
            serving.why.name(),
            fixed.word,
            steps.word,
            verdict(on && off)
        );
        Ok((word, on && off))
    }

    /// (r) on: a load at `residency` fed in batches (module doc).
    fn residency_on(
        levers: &bloomery_levers::Levers,
        residency: Residency,
        man: &RefManifest,
        toks: &[u32],
    ) -> Result<bool, GateError> {
        let (mut s, _) = open_with(Path::new(MODEL), levers, CTX, PrefillMode::Batch, residency)?;
        s.model_mut().body_parts("(r) on")?.2.log_residency(0);
        let mut ok = true;
        {
            let body = s.model().body("(r) on")?;
            let started = body.hybrid().swap().is_some();
            let source = body
                .residency_source()
                .ok_or("(r) on: the load runs no source")?;
            let mut card_layers = 0usize;
            let mut parts_ok = true;
            for l in body.host_run() {
                let parts = source.part_bytes(l);
                if parts.is_empty() {
                    continue;
                }
                card_layers += 1;
                parts_ok &= parts == [PART_BYTES; 3]
                    && (0..3).all(|p| {
                        source
                            .card_bytes(l, 0, p)
                            .is_ok_and(|b| b.len() == PART_BYTES)
                    });
            }
            let structure = started && card_layers > 0 && parts_ok && !source.unpacks();
            println!(
                "(r) on: machine started {started}, {card_layers} layers hold card parts of \
                 {PART_BYTES} B each, all so {parts_ok} {}",
                verdict(structure)
            );
            ok &= structure;
        }
        let (_, _, _, long, _) = set_open((D1K, &IK), None)?;
        let ids = long
            .get(..R_PROMPT)
            .ok_or("the 1,024-position prompt is short")?;
        let first = history(&mut s, ids)?;
        ok &= passes_clause(&first.passes, R_STEPS);
        let again = history(&mut s, ids)?;
        ok &= c1_clause_with(first.tokens.len(), first.landed(), first.same(&again), true);
        ok &= free(s.model_mut(), man, toks)?;
        let r = s
            .residency_reset()?
            .ok_or("(r) on: the load runs no residency machine")?;
        let body = s.model().body("(r) on")?;
        let machine = body
            .hybrid()
            .swap()
            .ok_or("(r) on: the load runs no machine")?;
        let seed = seeds(machine, body.hybrid().slots().layers())?;
        ok &= c7_clause(&r, machine, &seed);
        let back = body.hybrid().slots().stage_view() == first.start;
        println!("c7: the reset's slot map is the clear's {}", verdict(back));
        Ok(ok && back)
    }

    /// One arm of (rp): per round, each slot's greedy id and logits, and the
    /// flips that landed at the round's boundaries.
    struct PassArm {
        ids: Vec<Vec<u32>>,
        logits: Vec<Vec<Vec<f32>>>,
        landed: Vec<usize>,
    }

    impl PassArm {
        fn fnvs(&self, round: usize) -> Vec<u64> {
            self.logits[round]
                .iter()
                .map(|row| Fnv1a64::default().f32s(row).value())
                .collect()
        }

        /// The first round after whose boundary a flip had landed.
        fn first_flip(&self) -> usize {
            let mut seen = 0;
            self.landed
                .iter()
                .position(|&n| {
                    seen += n;
                    seen > 0
                })
                .unwrap_or(R_STEPS)
        }
    }

    /// The turns' logit of their own id less their logit of the pass's id:
    /// how far the turns preferred their id. NaN when an id is past the row
    /// or a logit is NaN, which no margin floor takes.
    fn gap(row: &[f32], own: u32, other: u32) -> f32 {
        match (row.get(own as usize), row.get(other as usize)) {
            (Some(&own), Some(&other)) => own - other,
            _ => f32::NAN,
        }
    }

    /// (rp) one arm on `m`, a load of [`slots_gate::STREAMS`] sequences under
    /// the machine: from the residency's seed, both slots prompted, then
    /// [`R_STEPS`] rounds, as one pass of both rows (`one_pass`) or as a
    /// select and a step a slot.
    fn pass_arm(
        body: &MimoSlots<'_>,
        m: &mut Mimo2Model,
        one_pass: bool,
    ) -> Result<PassArm, GateError> {
        const N: usize = slots_gate::STREAMS;
        m.residency_reset()?;
        let mut last = [0u32; N];
        for (s, id) in last.iter_mut().enumerate() {
            m.select_slot(s)?;
            body.rewind(m)?;
            *id = body.prompt(m, s)?;
        }
        m.body_parts("(rp)")?.2.take_residency_passes();
        let mut arm = PassArm {
            ids: Vec::with_capacity(R_STEPS),
            logits: Vec::with_capacity(R_STEPS),
            landed: Vec::with_capacity(R_STEPS),
        };
        for _ in 0..R_STEPS {
            let (ids, logits) = if one_pass {
                let ones: Vec<[u32; 1]> = last.iter().map(|&t| [t]).collect();
                let rows: Vec<(usize, &[u32])> = ones.iter().map(|t| &t[..]).enumerate().collect();
                let ids = m.step_slots(&rows)?.ids;
                (ids, m.slots_logits()?)
            } else {
                let (mut ids, mut logits) = (Vec::new(), Vec::new());
                for (s, &t) in last.iter().enumerate() {
                    m.select_slot(s)?;
                    ids.push(m.step(&[t])?);
                    logits.push(m.logits()?);
                }
                (ids, logits)
            };
            last.copy_from_slice(&ids);
            let passes = m.body_parts("(rp)")?.2.take_residency_passes();
            arm.landed.push(passes.iter().map(|(_, r)| r.landed).sum());
            arm.ids.push(ids);
            arm.logits.push(logits);
        }
        Ok(arm)
    }

    /// (rp) a pass of two slots under the machine, on a load of two
    /// sequences at `word`: [`R_STEPS`] rounds as one pass of both slots'
    /// rows against the same rounds as a select and a step a slot, each from
    /// the residency's seed with both prompts fed. Held: the pass's logits
    /// are the turns' bit for bit in every round up to the one after whose
    /// boundary a flip first landed in either arm (the slot map is the
    /// seed's until then), so are its ids; past it the two arms' maps move at
    /// other boundaries (the pass plans its tally under both rows, the turns
    /// under one at a time) and hold other experts on the card, so the
    /// pass's ids are the turns' until a round where the pass's id trails the
    /// turns' by less than [`MARGIN_FLOOR`] in the turns' logits (a near
    /// tie: the e2e gates' rule on the reference's top two, which a swap of
    /// the top two meets), and its logits within [`FREE_WORST`] of the turns'
    /// relative to their largest logit through that round — the arms feed
    /// other ids after it; the pass run again from the seed
    /// repeats itself bit for bit; and a flip lands in the pass arm, so the
    /// machine planned under the rows of two slots. A call that fails is the
    /// clause's red with its error printed (a machine whose tally holds one
    /// row refuses the first pass of two rows by name).
    fn residency_pass(
        levers: &bloomery_levers::Levers,
        word: Residency,
    ) -> Result<bool, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let inputs = read_inputs(&file)?;
        drop(file);
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, a, _) = man.step()?;
        let (_, _, _, long, _) = set_open((D1K, &IK), None)?;
        let b = long
            .get(..R_PROMPT)
            .ok_or("the 1,024-position prompt is short")?;
        let body = MimoSlots {
            levers,
            inputs,
            prompts: [a.to_vec(), b.to_vec()],
            residency: word,
        };
        let mut m = body.open(slots_gate::STREAMS)?;
        m.add_slots(slots_gate::STREAMS)?;
        m.body_parts("(rp)")?.2.log_residency(0);
        let arms = (|| -> Result<(PassArm, PassArm, PassArm), GateError> {
            let pass = pass_arm(&body, &mut m, true)?;
            let turns = pass_arm(&body, &mut m, false)?;
            let again = pass_arm(&body, &mut m, true)?;
            Ok((pass, turns, again))
        })();
        let (pass, turns, again) = match arms {
            Ok(arms) => arms,
            Err(e) => {
                println!("(rp) a call failed: {e} {}", verdict(false));
                return Ok(false);
            }
        };
        let exact_to = pass.first_flip().min(turns.first_flip()).min(R_STEPS - 1);
        let exact_ok = (0..=exact_to).all(|r| pass.fnvs(r) == turns.fnvs(r));
        // The first round and slot whose id is not the turns', and how far the
        // turns' logits preferred their id there. Past the bit-for-bit rounds
        // the two arms' maps hold other experts on the card, so an id may
        // differ only at a near tie; from the round after it the arms feed
        // other ids and their logits are no longer compared.
        let first_diff = (0..R_STEPS).find_map(|r| {
            (0..slots_gate::STREAMS)
                .find(|&s| pass.ids[r][s] != turns.ids[r][s])
                .map(|s| {
                    (
                        r,
                        s,
                        gap(&turns.logits[r][s], turns.ids[r][s], pass.ids[r][s]),
                    )
                })
        });
        let ids_ok = first_diff.is_none_or(|(r, _, gap)| r > exact_to && gap < MARGIN_FLOOR);
        let fed_same_to = first_diff.map_or(R_STEPS - 1, |(r, _, _)| r);
        let mut rel = 0.0f64;
        for r in exact_to + 1..=fed_same_to {
            for (p, t) in pass.logits[r].iter().zip(&turns.logits[r]) {
                let top = t.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
                let worst = p.iter().zip(t).fold(0.0f64, |m, (&p, &t)| {
                    m.max((f64::from(p) - f64::from(t)).abs())
                });
                rel = rel.max(worst / top.max(f64::MIN_POSITIVE));
            }
        }
        let band_ok = rel <= FREE_WORST;
        let repeat_ok = pass.ids == again.ids
            && (0..R_STEPS).all(|r| pass.fnvs(r) == again.fnvs(r))
            && pass.landed == again.landed;
        let flips = pass.landed.iter().sum::<usize>();
        let ok = ids_ok && exact_ok && band_ok && repeat_ok && flips > 0;
        let diff = first_diff.map_or("none".to_owned(), |(r, s, gap)| {
            format!("round {r} slot {s}, {gap:.4} under the turns' id")
        });
        println!(
            "(rp) a pass of {} slots under {word:?}, {R_STEPS} rounds: the first id not the \
             turns' {diff} (want none, or past round {exact_to} under {MARGIN_FLOOR}) {ids_ok}; \
             logits bit for bit through round {exact_to} {exact_ok}, after it through round \
             {fed_same_to} worst rel {rel:.3e} (want at most {FREE_WORST}) {band_ok}; the pass \
             run again repeats {repeat_ok}; {flips} flips landed in the pass arm, {} in the \
             turns' (want some in the pass arm) {}",
            slots_gate::STREAMS,
            turns.landed.iter().sum::<usize>(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// (r) on its own loads: the unset word on the gate placement's plan, then
    /// a load at it ([`residency_on`]).
    fn residency_arm(levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let inputs = read_inputs(&file)?;
        drop(file);
        let (word, mut ok) = residency_unset(levers, &inputs)?;
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, toks, _) = man.step()?;
        let toks = toks.to_vec();
        if matches!(word, Residency::Mid { .. }) {
            ok &= residency_on(levers, word, &man, &toks)?;
            ok &= residency_pass(levers, word)?;
        } else {
            println!(
                "(r) on: the unset word is off, no machine to load {}",
                verdict(false)
            );
            ok = false;
        }
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
        /// The residency machine's clauses ([`residency_arm`]), on a load of
        /// its own.
        Residency,
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

    /// `--only s|p|c|t|b|slots|r`, or every clause.
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
                Some("r") => Ok(Only::Residency),
                other => Err(format!("--only is s, p, c, t, b, slots or r, not {other:?}").into()),
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
        if only == Only::Residency {
            let t = Instant::now();
            let ok = residency_arm(&levers)?;
            elapsed("(r)", &t);
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
            let t = Instant::now();
            ok &= join_group(m)?;
            elapsed("(b) join", &t);
        }
        if only == Only::All {
            ok &= residency_off(s.model())?;
            // The slots' and the residency's loads take the card with the main
            // one dropped.
            drop(s);
            let t = Instant::now();
            ok &= slots(&levers)?;
            elapsed("(slots)", &t);
            let t = Instant::now();
            ok &= residency_arm(&levers)?;
            elapsed("(r)", &t);
        }
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
