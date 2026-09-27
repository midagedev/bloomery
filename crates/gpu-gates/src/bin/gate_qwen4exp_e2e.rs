//! The Qwen3.8-Flash-Next (`qwen4exp`) end-to-end gate: the whole program —
//! 36 sigmoid-gated delta-rule layers and 12 selecting attention layers, every
//! block in the four gated-residual streams, the PLE site on layer 1, 48
//! MoE blocks whose 512 routed experts all run on the host tier beside a
//! sigmoid-gated shared expert on the card, the head's mix, the q8_0 head and
//! the argmax — loaded once by its placement on the gate card, against ik's
//! CPU oracle sets (`refset::arch::qwen4exp`: the 5-token batch set, the step
//! after a fused 4-token prefill, the same after a prefill run node by node,
//! the steps at positions 1,024 and 3,000 of the prose), every set read
//! through its family.
//!
//! The file opens through `PlanInputs::describe` and `Body38::open_placed`,
//! which refuses by name every coverage item past the two the program
//! allows (`bloomery_gpu::arch::qwen3moe::ALLOWED`, the chat surface's; the
//! plain `PlanInputs::read` refuses those two as well). The coverage check
//! holds qwen4exp to `Body38`'s own rows (`crates/model/src/arch/coverage.rs`,
//! not this gate's to change).
//!
//! What is asserted:
//! - (s) structure: the captured decode step holds [`NODES_DECODE`] nodes,
//!   [`MEMOPS`] of them stream memory-operation batches (each layer's go and
//!   wait) and the rest kernels, and the program's own count
//!   (`Body38::step_launches`) is the same; the selecting layers are every
//!   fourth ([`qsa_layers`]); the stores' bytes equal their derivation from
//!   the header ([`store_bytes`]).
//! - (p) one program: the batch set's five tokens as five graph steps, as
//!   five eager steps (the taps armed) and as one eager pass of five rows
//!   through the batch port (`Prompt38::Pass`), each from a reset with every
//!   selecting store's planes set to [`PLANE_FILL`] (`reset` leaves them, so
//!   a row a path reads without writing reads the pattern, not the previous
//!   run's value), leave the same last token, the same last logits and every
//!   store — the delta states and conv rings, the K/V planes, the raw and
//!   pooled indexer keys, the PLE ring — bit for bit; the graph and eager
//!   steps also every position's token and logits. Then five graph steps
//!   from `reset` after the pass's state leave the first run's everything:
//!   `reset` clears the recurrent state.
//! - (c) free-running on the batch set, each layer's picks read from the
//!   eager run's route taps (our own chain's routing, which is what a
//!   layer-by-layer run would read, so this gate has no layered clause),
//!   each tap's ids the top ten of its own logits ([`picks_its_top`]): a
//!   token whose chosen set differs from ik's `ffn_moe_topk-L` is a flip,
//!   allowed only when every exchanged pair's gap in ik's logits
//!   (`ffn_moe_logits`, whose softmax order is the pick's) lies within our
//!   two logits' error there and that error within [`FLIP_ERR_CAP`], a
//!   measured frontier ([`Flip::allowed`]), named and counted. A flip
//!   at layer `L'` and position `t'` lies on the path of every layer output
//!   from `L'` on at `t'` and at every later position; each layer output off
//!   every flip's path against ik's `l_out-L` within [`FREE_BAND`], the
//!   outputs past a flip printed and counted; the last position's argmax
//!   equal to ik's `result_output` argmax.
//! - (t) each step set: its prefill fed by our own steps from a reset, then
//!   the step; its argmax equal to ik's, or — named and counted — our argmax
//!   ik's runner-up, ik's own margin between the two inside twice the
//!   distance between our logits and ik's at those ids, and our whole
//!   logits row within [`FREE_BAND`] of ik's. The logits row is printed and
//!   bounds only a tie, as in the GLM gate: [`FREE_BAND`] bounds a layer
//!   output off every flip's path, and every step set's last position lies
//!   on a flip's path from its first layers (the router's top-10 margins
//!   over 512 sit under ik's q8_2 spacing). The step's layer outputs
//!   against ik's `l_out-L` are printed, and held to [`FREE_BAND`] on the two
//!   4-token sets only, below the first layer a flip lies on the path of the
//!   batch set's last position as (c) names them. D1K's step reads every
//!   pool (1,024 positions, 256 pools, under the 512 the selector keeps), so
//!   its selection list is the identity; D3K's reads a selection (751 pools
//!   past 512).
//! - (q) the pass in the selector region: D3K's prefill as eager passes of
//!   up to eight positions (`Prompt38::Pass`) from a reset with the planes
//!   filled, then the same
//!   step, leaves the step-fed run's step token, logits, layer outputs and
//!   every store bit for bit.
//! - (v) the verify: after a prefix of two steps, a verify of the first T
//!   of four rows ([`verify_rows`]; T = 2, 3, 4, `step_rows`, captured) — one
//!   row completing pool 0 — returns every row's argmax and logits bit for
//!   bit the steps' of those rows; for every k in 1..=T the commit of k rows
//!   (`rollback` to the first position not kept) leaves the lane word at
//!   `(c + k − 1) mod 4`, computed here from the lane before the verify, and
//!   the live stores bit for bit the k steps' — each delta layer's committed
//!   lane, the conv ring's slots of the eight positions before the count,
//!   the K/V planes and raw keys below it, the pooled plane's pools complete
//!   at the count, the PLE ring's slots of the fourteen positions before the
//!   count (a rejected row wrote position-indexed rows past the count —
//!   among them the pool it completed — which nothing reads until the row
//!   at that position rewrites them first) — and the step after the commit
//!   its token and logits; a second verify from a moved lane (4 rows after a
//!   commit of 3, lanes 2, 3, 0, 1) the same against its steps; an eager
//!   verify bit for bit the replay; and a deep verify in the selector
//!   region, after [`DEEP`] positions of D3K's prefill by passes, its row
//!   completing pool 513 rejected, the same against its steps — with every
//!   pooled row past the count set to NaN after the commit ([`POOL_POISON`]),
//!   the step after and the one after it (which completes pool 513 again)
//!   bit for bit the steps', raising nothing. Structure: each verify's graph holds
//!   [`NODES_VERIFY`] nodes, [`MEMOPS`] of them batch mem ops, one argmax,
//!   one `ds41_ffn_handoff_10_cols` a layer and no one-column handoff; a
//!   replay's host tier serves each layer once with one call into the host
//!   experts (48 services, 48 calls, 48 `Cols` services). Refusals: a step
//!   while a verify waits for its commit, and a commit with no verify
//!   waiting, by name; the lane word planted on a lane no call wrote makes
//!   the next step's first delta layer raise `delta_stamp` there, alone.
//! - (o) one owner of the position: after the prefix, a graph step, an eager
//!   pass of three rows and a graph verify of two rows (kept whole), each
//!   with a failure planted before its launch (`Body38::plant_before_launch`)
//!   fail by name with the model at the prefix and not poisoned, and the
//!   same call again gives the clean call's token, logits, PLE rows
//!   (`Body38::ple_rows`) and every store bit for bit; each with a failure
//!   planted after its launch (`Body38::plant_after_launch`) fails by name
//!   with the model at the prefix, `Body38::kept` keeps less than the
//!   prefix (no delta state holds it any more), a step there is refused by
//!   name (the recurrent stores hold it already), and after `reset` the
//!   prefix and the same call give the clean bits.
//! - (r) refusals: `Prompt38::parse` refuses `gemm` and `auto` by name,
//!   naming `gemm_q8_0`; the image placeholder [`IMAGE_TOKEN`] as a step and
//!   as a pass is refused by name with the position kept and the model not
//!   poisoned; a prompt past the stores is refused by name before any
//!   launch, on either path, the position kept.
//!
//! Named differences, not banded away: ik combines the block as
//! `routed + σ(g)·shared`, ours as `hsum + shared·w` with the sigmoid weight
//! the router's eleventh slot — one f32 rounding of the update; ik's CPU
//! projections quantize their input to q8_2 per 32 values, ours read the f32
//! row; at D3K's position (3,001 keys, 751 pools, the last one of one key)
//! ik cuts the selection by cells and keeps up to three keys of the 513th
//! pool that ours, which keeps 512 whole pools and the tail, does not (the
//! difference q38sel measured), its logits printed.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen4exp_e2e: built without the `gpu` feature; see `just gate-gpu-qwen4exp-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen4exp_e2e", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::time::Instant;

    use bloomery_gpu::arch::qwen3moe::{
        Body38, LayerKind38, Prompt38, Qwen38Model, RouteTap, Store38Host,
    };
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Fault, FaultSite, GpuError};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, data_dir, ref_tensor_logical_in,
        topk_ids_logical_within, verdict,
    };
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::qwen35moe::place::{PlanInputs, machine};
    use model::placement::PlanLevers;
    use model::placement::workstation::RTX_3090;
    use refset::arch::qwen4exp::{BATCH, D1K, D3K, IK, MODEL, STEP4, STEP4_EVERY_NODE};

    /// Cache rows: D3K's step at position 3,000, with room.
    const CTX: usize = 3072;

    /// The file's shape, as the header states it (qwen4arch-design §1): what
    /// the derivations below are written against.
    const HIDDEN: usize = 2560;
    const STREAMS: usize = 4;
    const N_LAYER: usize = 48;
    const N_QSA: usize = 12;
    const N_GDN: usize = 36;
    const N_EXPERT: usize = 512;
    const N_USED: usize = 10;
    /// Delta layers: 48 value heads and 16 key heads of 128, a conv of 4
    /// taps; selecting layers: 2 K/V heads of 256, indexer keys of 128, pools
    /// of 4; the PLE conv: 4 taps at dilation 3 over the four streams.
    const V_HEADS: usize = 48;
    const K_HEADS: usize = 16;
    const HEAD_V: usize = 128;
    const CONV: usize = 4;
    const N_KV: usize = 2;
    const HEAD: usize = 256;
    const IDX_DIM: usize = 128;
    const POOL: usize = 4;
    const PLE_TAPS: usize = 4;
    const PLE_DILATION: usize = 3;
    /// The most positions an eager pass or a ring's tail takes.
    const PASS_ROWS: usize = 8;
    /// The image placeholder the file's PLE hash refuses as an input.
    const IMAGE_TOKEN: u32 = 248_056;

    /// The f16 pattern every selecting store's planes hold before a run
    /// (100.0): `reset` leaves those rows as they were, and a path that reads
    /// a row it did not write must read this, not a previous run's value — a
    /// finite value, so a correct path's masked reads stay finite, and far
    /// from any key, value or indexer key the model writes.
    const PLANE_FILL: u16 = 0x5640;

    /// PIN(2026-09-27): the captured decode step's node count, derived before
    /// the chain was built: the embedding row; each layer's two mixes, three
    /// launches each (the grouped norm with the down projection, the up, the
    /// gated mean — the combine folded into the next mix); a delta layer's
    /// 7 mixer launches (q·k·v, z, β·α, the conv, the delta step, the gated
    /// norm, the output projection) and a selecting layer's 14 (q, k, v, the
    /// indexer key's projection, its append and the pool, the indexer query,
    /// the selection's two, the q/k norm, turn and append, the selected
    /// flash's two, the out gate, the output projection); each block's 9 (the
    /// router, the handoff, the go, the shared expert's four, the wait, the
    /// gated sum); the PLE site's 5 on layer 1 (the combine, key and value,
    /// gate, conv); the head's 5 (its mix's three, the q8_0 gemv, the argmax):
    /// 1 + 36·22 + 12·29 + 5 + 5.
    const NODES_DECODE: usize = 1151;

    /// PIN(2026-09-27): each layer's go and wait.
    const MEMOPS: usize = 2 * N_LAYER;

    const _: () =
        assert!(NODES_DECODE == 1 + N_GDN * 22 + N_QSA * 29 + 5 + 5 && N_GDN + N_QSA == N_LAYER);

    /// PIN(2026-09-28): the captured verify's node count at 2, 3 and 4 rows,
    /// derived before the chain was built: the step's [`NODES_DECODE`], plus
    /// at more than one row each delta layer's two token-major copies (β and
    /// α out of the joined projection) and each selecting layer's one (the
    /// indexer queries); the handoff (one launch writing every row's image),
    /// the go, the wait and the head (its mix's three, one q8_0 gemv over
    /// every row, one argmax) are one each whatever the rows:
    /// 1151 + 36·2 + 12·1.
    const NODES_VERIFY: usize = 1235;

    const _: () = assert!(NODES_VERIFY == NODES_DECODE + N_GDN * 2 + N_QSA);

    /// Lanes of a delta layer's state, a verify's rows at most.
    const LANES: usize = 4;

    /// Rows of the conv ring and of the PLE ring: the reach of the conv and
    /// a pass.
    const CONV_RING: usize = CONV - 1 + PASS_ROWS;
    const PLE_RING: usize = (PLE_TAPS - 1) * PLE_DILATION + PASS_ROWS;

    /// Positions of the verify's prefix: the verify starts at position 2,
    /// so its row 1 (position 3, count 4) completes pool 0.
    const PREFIX: usize = 2;

    /// Positions of the deep verify's prefix: its rows (counts 2,053 to
    /// 2,056) select, each seeing more complete pools than the 512 a select
    /// keeps, and its row 3 completes pool 513; kept to [`DEEP_KEPT`] rows,
    /// the step after (count 2,055) selects among pools 0 to 512 only, and
    /// the step after that (count 2,056) completes pool 513 again before its
    /// select reads it.
    const DEEP: usize = 2052;
    const DEEP_KEPT: usize = 2;

    /// An f16 NaN: the deep verify's pooled rows past the count after its
    /// commit, so a select that scores one raises `pool_select` by name.
    const POOL_POISON: u16 = 0x7e00;

    /// PIN(2026-09-27): the free-running bound on a layer output's relative
    /// distance from ik's, off every flip's path. Borrowed, not measured on
    /// this model: GLM-5.3's gate's derivation (√45 · 1.415e-2, its forced
    /// arm's worst layer error on the streams on ik's inputs) over this
    /// model's 48 layers, √48 · 1.415e-2 ≈ 0.098, rounded up. GLM is the
    /// analog and Qwen3.6 (0.26) is not: at every q8_0 projection our side
    /// reads the f32 row, so the gap is ik's q8_2 activation alone, as in
    /// GLM, where Qwen3.6's inputs are quantized on both sides; the host
    /// experts run the q8_2_x4 rule on both. This gate has no forced arm to
    /// re-derive it on Qwen3.8; (c) prints every layer's distance, the input
    /// a forced arm would take.
    const FREE_BAND: f64 = 0.10;

    /// PIN(2026-09-28): [잠정 — 백로그] the most error, in router logit units, our
    /// two logits may carry at an excused flip — a measured frontier, not a derivation.
    /// In the batch set's free run the clean chain's largest pair error
    /// was 0.942 (gap 0.483); the PLE site skipped (m06) reached 4.69, the other broken
    /// chains 20.15 and 20.66; 2.0 sits 2.1x above clean and 2.3x below m06. The route tap
    /// reading the next layer's logits (m14, 6.99) is left out: that error is the tap's,
    /// not the router's. A flip-aware bound from a Qwen3.8 forced arm replaces it.
    const FLIP_ERR_CAP: f64 = 2.0;

    /// The selecting layers: every fourth, as the header's interval states.
    fn qsa_layers() -> Vec<usize> {
        (0..N_LAYER).filter(|l| l % 4 == 3).collect()
    }

    /// The stores' bytes at [`CTX`] rows, derived from the header: each
    /// delta layer's state, 48 heads of 128 × 128 f32, and conv ring,
    /// `CONV − 1 + PASS_ROWS` rows of the conv's `2·16·128 + 48·128`
    /// channels in f32; each selecting layer's K and V rows, `2 · 2 · 256`
    /// f16 a position, its raw indexer key, 128 f16 a position, and its
    /// pooled key, 128 f16 a pool of 4 (the last pool counted whole); the
    /// PLE ring, `(taps − 1)·dilation + PASS_ROWS` rows of the four streams
    /// in f32. PIN(2026-09-28): a delta layer's state is [`LANES`] lanes, a
    /// verify's rows each keeping its own, with a u32 stamp a lane.
    fn store_bytes() -> usize {
        let conv_ch = 2 * K_HEADS * HEAD_V + V_HEADS * HEAD_V;
        let rec = LANES * (V_HEADS * HEAD_V * HEAD_V * 4 + 4) + CONV_RING * conv_ch * 4;
        let sel = CTX * (2 * N_KV * HEAD * 2 + IDX_DIM * 2) + CTX.div_ceil(POOL) * IDX_DIM * 2;
        let ple = ((PLE_TAPS - 1) * PLE_DILATION + PASS_ROWS) * STREAMS * HIDDEN * 4;
        N_GDN * rec + N_QSA * sel + ple
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

    /// Bit equality of two f32 rows.
    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// A set's tap `name` in its logical order.
    fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
    }

    /// ik's last logits row of a set.
    fn ik_last(man: &RefManifest, vocab: usize) -> Result<Vec<f32>, GateError> {
        let ik = tap(man, "result_output")?;
        let at = ik
            .len()
            .checked_sub(vocab)
            .ok_or("result_output holds no row")?;
        Ok(ik[at..].to_vec())
    }

    fn open() -> Result<Qwen38Model, GateError> {
        let levers = bloomery_levers::at_main(&[])?;
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        if file.architecture() != Some("qwen4exp") {
            return Err(format!("{MODEL} is {:?}, not qwen4exp", file.architecture()).into());
        }
        let t = Instant::now();
        // `describe`, not `read`: `read` refuses the chat surface's two
        // items too; `open_placed` refuses what `ALLOWED` does not name.
        let inputs = PlanInputs::describe(&file)?;
        let machine = machine(RTX_3090, inputs.spec.layers.len());
        let plan = inputs.plan(&machine, CTX as u64, &PlanLevers::from_levers(&levers)?)?;
        println!(
            "plan card={} ctx_max={} host_experts={} card_experts={}",
            RTX_3090.name, plan.ctx_max, plan.host.experts, plan.cards[0].experts
        );
        let mut m = Body38::open_placed(file, &plan, &inputs, 0, levers.host())?;
        m.set_mode(StepMode::Graph);
        println!(
            "load resident_bytes={} ctx={CTX} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    // ------------------------------------------------------ (s) structure

    fn structure(m: &mut Qwen38Model) -> Result<bool, GateError> {
        let nodes = m.capture_step()?;
        let body = m.body("structure")?;
        let kinds = body.kinds();
        let qsa: Vec<usize> = (0..kinds.len())
            .filter(|&l| kinds[l] == LayerKind38::Qsa)
            .collect();
        let (stores, want_stores) = (body.store_bytes(), store_bytes());
        let mut ok = kinds.len() == N_LAYER && qsa == qsa_layers() && stores == want_stores;
        println!(
            "structure layers={} selecting at {qsa:?}; store bytes {stores} (want {want_stores}, \
             derived) {}",
            kinds.len(),
            verdict(ok)
        );
        let (counted, memops) = body.step_launches();
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        let pass = nodes == NODES_DECODE
            && counted == NODES_DECODE
            && memops == MEMOPS
            && b == MEMOPS
            && k == NODES_DECODE - MEMOPS
            && other == 0;
        println!(
            "structure decode graph_nodes={nodes} (want {NODES_DECODE}; the program counts \
             {counted}, {memops} of them batch_mem_op) kernel={k} batch_mem_op={b} (want \
             {MEMOPS}) other={other} {}",
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
    }

    // ------------------------------------------ (p) one program, (c) free

    /// What a run of tokens left: each position's argmax and logits (only
    /// the last of each for a pass), with the taps armed each position's
    /// layer outputs and route, and every store with the PLE ring.
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        taps: Vec<Vec<f32>>,
        routes: Vec<Vec<RouteTap>>,
        stores: Vec<Store38Host>,
        ple_ring: Vec<f32>,
    }

    /// Arm or disarm the layer and route taps.
    fn set_taps(m: &mut Qwen38Model, on: bool) -> Result<(), GateError> {
        m.set_layer_taps(on)?;
        Ok(())
    }

    /// `reset`, then every selecting store's planes set to [`PLANE_FILL`].
    fn fresh(m: &mut Qwen38Model) -> Result<(), GateError> {
        m.reset()?;
        let (gpu, _, b) = m.body_parts("fresh")?;
        b.fill_planes(gpu, PLANE_FILL)?;
        Ok(())
    }

    /// Every store and the PLE ring, read back.
    fn stores(m: &mut Qwen38Model) -> Result<(Vec<Store38Host>, Vec<f32>), GateError> {
        let (gpu, _, b) = m.body_parts("stores")?;
        Ok(b.stores_host(gpu)?)
    }

    /// `toks` as steps from [`fresh`] (without `reset`: from where the
    /// model stands), eager steps with the taps armed.
    fn run_steps(
        m: &mut Qwen38Model,
        toks: &[u32],
        mode: StepMode,
        reset: bool,
    ) -> Result<Run, GateError> {
        if reset {
            fresh(m)?;
        }
        m.set_mode(mode);
        let taps_on = mode == StepMode::Eager;
        set_taps(m, taps_on)?;
        let mut r = Run {
            tokens: Vec::new(),
            logits: Vec::new(),
            taps: Vec::new(),
            routes: Vec::new(),
            stores: Vec::new(),
            ple_ring: Vec::new(),
        };
        for &t in toks {
            r.tokens.push(m.step(&[t])?);
            r.logits.push(m.logits()?);
            if taps_on {
                let (gpu, _, b) = m.body_parts("run_steps")?;
                r.taps.push(b.taps(gpu)?);
                r.routes.push(b.route_taps(gpu)?);
            }
        }
        set_taps(m, false)?;
        m.set_mode(StepMode::Graph);
        (r.stores, r.ple_ring) = stores(m)?;
        Ok(r)
    }

    /// `toks` as one prompt call by eager passes from [`fresh`]: the last
    /// token and logits, and every store.
    fn run_pass(m: &mut Qwen38Model, toks: &[u32]) -> Result<Run, GateError> {
        fresh(m)?;
        let last = m.prompt38(toks, Prompt38::Pass)?;
        let logits = m.logits()?;
        let (stores, ple_ring) = stores(m)?;
        Ok(Run {
            tokens: vec![last],
            logits: vec![logits],
            taps: Vec::new(),
            routes: Vec::new(),
            stores,
            ple_ring,
        })
    }

    /// The layers whose stores differ, and whether the PLE rings do.
    fn store_diff(a: &Run, b: &Run) -> (Vec<usize>, bool) {
        let n = a.stores.len().max(b.stores.len());
        let layers = (0..n)
            .filter(|&l| match (a.stores.get(l), b.stores.get(l)) {
                (Some(x), Some(y)) => !x.same_bits(y),
                _ => true,
            })
            .collect();
        (layers, !same_bits(&a.ple_ring, &b.ple_ring))
    }

    /// `got` against `want`: the tokens and logits (every position, or the
    /// last only) and every store, bit for bit.
    fn same_run(label: &str, got: &Run, want: &Run, last_only: bool) -> bool {
        let (tokens, logits) = if last_only {
            (
                got.tokens.last() == want.tokens.last() && !got.tokens.is_empty(),
                matches!((got.logits.last(), want.logits.last()), (Some(a), Some(b)) if same_bits(a, b)),
            )
        } else {
            (
                got.tokens == want.tokens,
                got.logits.len() == want.logits.len()
                    && got
                        .logits
                        .iter()
                        .zip(&want.logits)
                        .all(|(a, b)| same_bits(a, b)),
            )
        };
        let (layers, ring) = store_diff(got, want);
        let ok = tokens && logits && layers.is_empty() && !ring;
        println!(
            "paths {label}: tokens {:?} vs {:?}; {} logits bit for bit: {logits}; stores \
             differing at layers {layers:?}, PLE ring differing: {ring} {}",
            got.tokens,
            want.tokens,
            if last_only {
                "the last"
            } else {
                "every position's"
            },
            verdict(ok)
        );
        ok
    }

    /// (p): the graph steps, the eager steps (returned, taps armed), the
    /// pass, and the steps again from `reset` alone.
    fn paths(m: &mut Qwen38Model, toks: &[u32]) -> Result<(bool, Run), GateError> {
        let graph = run_steps(m, toks, StepMode::Graph, true)?;
        let eager = run_steps(m, toks, StepMode::Eager, true)?;
        let mut ok = same_run(
            "five eager steps vs five graph replays",
            &eager,
            &graph,
            false,
        );
        let pass = run_pass(m, toks)?;
        ok &= same_run(
            "one pass of five rows vs five graph steps",
            &pass,
            &graph,
            true,
        );
        // After the pass's state: `reset` must clear it.
        let again = run_steps(m, toks, StepMode::Graph, true)?;
        ok &= same_run(
            "after the pass's state and a reset, the same five steps",
            &again,
            &graph,
            false,
        );
        Ok((ok, eager))
    }

    /// Each layer's relative distance at every position `taps` holds
    /// against ik's `l_out-L` (`None` where ik kept no row for it).
    fn layer_table(
        man: &RefManifest,
        taps: &[Vec<f32>],
    ) -> Result<Vec<Vec<Option<f64>>>, GateError> {
        let row = STREAMS * HIDDEN;
        let n = taps.len();
        (0..N_LAYER)
            .map(|l| {
                let ik = tap(man, &format!("l_out-{l}"))?;
                let kept = ik.len() / row;
                Ok(taps
                    .iter()
                    .enumerate()
                    .map(|(t, ours)| {
                        let i = (t + kept).checked_sub(n)?;
                        Some(rel(
                            ours.get(l * row..(l + 1) * row)?,
                            &ik[i * row..(i + 1) * row],
                        ))
                    })
                    .collect())
            })
            .collect()
    }

    /// Each row's worst entry of a [`layer_table`] and the tap it was met at.
    fn worst_of(table: &[Vec<Option<f64>>]) -> Vec<(f64, usize)> {
        table
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter_map(|(t, e)| e.map(|e| (e, t)))
                    .fold((0.0f64, 0usize), |w, x| if x.0 > w.0 { x } else { w })
            })
            .collect()
    }

    /// The worst layer and the first of layers `0..held` past the band,
    /// printed; whether every one of those is inside it.
    fn print_layers(what: &str, rels: &[(f64, usize)], held: usize) -> bool {
        for (l, &(e, t)) in rels.iter().enumerate() {
            if l % 8 == 0 || l == rels.len() - 1 || (l < held && e > FREE_BAND) {
                println!("{what} layer={l} l_out_rel={e:.3e} at tap {t}");
            }
        }
        match rels.iter().take(held).position(|&(e, _)| e > FREE_BAND) {
            Some(l) => {
                println!("{what}: first layer past the band {FREE_BAND:.2}: {l}");
                false
            }
            None => true,
        }
    }

    /// ik's routing of one layer over the set's tokens: its logits
    /// ([`N_EXPERT`] a token; the pick ranks their softmax, the same order)
    /// and its chosen ids ([`N_USED`] a token).
    struct IkRoute {
        logits: Vec<f32>,
        ids: Vec<i32>,
    }

    impl IkRoute {
        fn read(man: &RefManifest, l: usize) -> Result<IkRoute, GateError> {
            let logits = tap(man, &format!("ffn_moe_logits-{l}"))?;
            let row = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
            let ids = topk_ids_logical_within(man, row, N_EXPERT as u32)?;
            if logits.len() % N_EXPERT != 0 || ids.len() * N_EXPERT != logits.len() * N_USED {
                return Err(format!(
                    "layer {l}: {} logits and {} ids; a token takes {N_EXPERT} and {N_USED}",
                    logits.len(),
                    ids.len()
                )
                .into());
            }
            Ok(IkRoute { logits, ids })
        }

        fn tokens(&self) -> usize {
            self.ids.len() / N_USED
        }

        /// Token `t`'s logits and ids.
        fn at(&self, t: usize) -> (&[f32], &[i32]) {
            (
                &self.logits[t * N_EXPERT..(t + 1) * N_EXPERT],
                &self.ids[t * N_USED..(t + 1) * N_USED],
            )
        }

        /// ik's own margin at token `t`: its tenth pick's logit less the
        /// best it left.
        fn margin(&self, t: usize) -> f64 {
            let (v, ids) = self.at(t);
            let min_in = ids
                .iter()
                .map(|&e| f64::from(v[e as usize]))
                .fold(f64::INFINITY, f64::min);
            let max_out = (0..N_EXPERT)
                .filter(|&e| !ids.contains(&(e as i32)))
                .map(|e| f64::from(v[e]))
                .fold(f64::NEG_INFINITY, f64::max);
            min_in - max_out
        }
    }

    /// A token whose chosen set differs from ik's: every exchanged pair —
    /// `a` ours only, `b` ik's only — with ik's gap `ik[b] − ik[a]` and our
    /// two logits' error `|ours[a] − ik[a]| + |ours[b] − ik[b]|`.
    struct Flip {
        layer: usize,
        token: usize,
        margin: f64,
        pairs: Vec<(u32, u32, f64, f64)>,
    }

    impl Flip {
        /// Allowed only when every pair's gap lies within its error and the
        /// error within [`FLIP_ERR_CAP`]: our ranking then differs from ik's
        /// by no more than our logits' distance from ik's, a distance a
        /// correct chain has been measured to stay under. Past either the
        /// pick is wrong, named.
        fn allowed(&self) -> bool {
            !self.pairs.is_empty()
                && self
                    .pairs
                    .iter()
                    .all(|&(_, _, gap, err)| gap <= err && err <= FLIP_ERR_CAP)
        }

        fn line(&self) -> String {
            let pairs: Vec<String> = self
                .pairs
                .iter()
                .map(|(a, b, gap, err)| format!("{a}<-{b} gap {gap:.3e} err {err:.3e}"))
                .collect();
            let verdict = if self.allowed() {
                "allowed (counted)"
            } else if self
                .pairs
                .iter()
                .any(|&(_, _, _, err)| err > FLIP_ERR_CAP || err.is_nan())
            {
                "FAIL: our router error past the pinned frontier"
            } else {
                "FAIL: a pair's gap past our error"
            };
            format!(
                "free flip layer={} token={}: ik margin {:.3e}; {} (cap {FLIP_ERR_CAP:.1}): \
                 {verdict}",
                self.layer,
                self.token,
                self.margin,
                pairs.join(", "),
            )
        }
    }

    /// The flip at layer `l`, token `t`, if our chosen set is not ik's.
    fn flip_at(l: usize, t: usize, ours: &RouteTap, ik: &IkRoute) -> Option<Flip> {
        let (iv, ids) = ik.at(t);
        let (ov, oids) = (&ours.logits, &ours.ids);
        let only_ours: Vec<u32> = oids
            .iter()
            .copied()
            .filter(|&e| !ids.contains(&(e as i32)))
            .collect();
        let only_ik: Vec<u32> = ids
            .iter()
            .map(|&e| e as u32)
            .filter(|e| !oids.contains(e))
            .collect();
        if only_ours.is_empty() && only_ik.is_empty() {
            return None;
        }
        let v = |x: &[f32], e: u32| x.get(e as usize).map_or(f64::NAN, |&y| f64::from(y));
        let mut pairs = Vec::new();
        for &a in &only_ours {
            for &b in &only_ik {
                let gap = v(iv, b) - v(iv, a);
                let err = (v(ov, a) - v(iv, a)).abs() + (v(ov, b) - v(iv, b)).abs();
                pairs.push((a, b, gap, err));
            }
        }
        Some(Flip {
            layer: l,
            token: t,
            margin: ik.margin(t),
            pairs,
        })
    }

    /// Whether a route tap's ids are [`N_USED`] distinct experts under
    /// [`N_EXPERT`] and the top of its own logits: none it left above the
    /// lowest it kept (ties either way). A tap of another layer's logits
    /// beside this layer's ids does not hold it.
    fn picks_its_top(r: &RouteTap) -> bool {
        let mut ids = r.ids.clone();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != N_USED
            || r.logits.len() != N_EXPERT
            || ids.iter().any(|&e| e as usize >= N_EXPERT)
        {
            return false;
        }
        let kept = |e: usize| ids.binary_search(&(e as u32)).is_ok();
        let min_in = ids
            .iter()
            .map(|&e| r.logits[e as usize])
            .fold(f32::INFINITY, f32::min);
        (0..N_EXPERT)
            .filter(|&e| !kept(e))
            .all(|e| r.logits[e] <= min_in)
    }

    /// Whether a flip at `(l', t')` lies on the path of layer `l`'s output
    /// at position `t`: every layer from `l'` on reads it at `t'`, and every
    /// later position through the mixers' stores.
    fn on_path(flips: &[Flip], l: usize, t: usize) -> bool {
        flips.iter().any(|f| f.layer <= l && f.token <= t)
    }

    /// The free clause's verdict, and by position the first layer a flip
    /// lies on the path of ([`N_LAYER`] where none does).
    fn free(man: &RefManifest, eager: &Run, vocab: usize) -> Result<(bool, Vec<usize>), GateError> {
        let table = layer_table(man, &eager.taps)?;
        for (l, row) in table.iter().enumerate() {
            let cells: Vec<String> = row
                .iter()
                .map(|e| e.map_or("-".to_string(), |e| format!("{e:.3e}")))
                .collect();
            println!("free table layer={l} l_out_rel by tap {}", cells.join(" "));
        }
        let mut flips = Vec::new();
        let mut inconsistent = Vec::new();
        for (t, routes) in eager.routes.iter().enumerate() {
            for (l, r) in routes.iter().enumerate() {
                if !picks_its_top(r) {
                    inconsistent.push((l, t));
                }
            }
        }
        println!(
            "free: route taps whose ids are not the top {N_USED} of their own logits (distinct, \
             under {N_EXPERT}): {} {:?} {}",
            inconsistent.len(),
            &inconsistent[..inconsistent.len().min(8)],
            verdict(inconsistent.is_empty())
        );
        for l in 0..N_LAYER {
            let ik = IkRoute::read(man, l)?;
            if ik.tokens() != eager.routes.len() {
                return Err(format!(
                    "layer {l}: ik routes {} tokens, the run {}",
                    ik.tokens(),
                    eager.routes.len()
                )
                .into());
            }
            let margins: Vec<String> = (0..ik.tokens())
                .map(|t| format!("{:.2e}", ik.margin(t)))
                .collect();
            println!("ik margin layer={l} by token {}", margins.join(" "));
            for (t, routes) in eager.routes.iter().enumerate() {
                let ours = routes
                    .get(l)
                    .ok_or_else(|| format!("position {t}: no route tap for layer {l}"))?;
                flips.extend(flip_at(l, t, ours, &ik));
            }
        }
        for f in &flips {
            println!("{}", f.line());
        }
        let flips_ok = flips.iter().all(Flip::allowed);
        let (mut held, mut hl, mut ht, mut exempt) = (0.0f64, 0usize, 0usize, 0usize);
        for (l, row) in table.iter().enumerate() {
            for (t, e) in row.iter().enumerate() {
                let Some(e) = *e else { continue };
                if on_path(&flips, l, t) {
                    exempt += 1;
                } else if e > held {
                    (held, hl, ht) = (e, l, t);
                }
            }
        }
        let firsts: Vec<usize> = (0..eager.taps.len())
            .map(|t| {
                (0..N_LAYER)
                    .find(|&l| on_path(&flips, l, t))
                    .unwrap_or(N_LAYER)
            })
            .collect();
        println!(
            "free: first layer a flip lies on the path of, by position {}; {exempt} (layer, \
             position) outputs past a flip, printed and counted",
            firsts
                .iter()
                .map(|l| l.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        let ik = ik_last(man, vocab)?;
        let ours = eager.logits.last().ok_or("no logits")?;
        let (top, ik_top) = (argmax(ours), argmax(&ik));
        let ok = held <= FREE_BAND && flips_ok && top == ik_top && inconsistent.is_empty();
        println!(
            "free: {} tokens, {} flips ({} allowed), worst l_out_rel off every flip's path \
             {held:.3e} at layer {hl} position {ht} (band {FREE_BAND:.2}); last position argmax \
             ours={top} ik={ik_top} logits_rel={:.3e} (printed) {}",
            eager.tokens.len(),
            flips.len(),
            flips.iter().filter(|f| f.allowed()).count(),
            rel(ours, &ik),
            verdict(ok)
        );
        Ok((ok, firsts))
    }

    // --------------------------------------------------- (t) step sets

    /// The last token eagerly with the taps armed from where the model
    /// stands: its layer outputs, its logits and argmax, then every store.
    fn run_last(m: &mut Qwen38Model, tok: u32) -> Result<Run, GateError> {
        run_steps(m, &[tok], StepMode::Eager, false)
    }

    /// The step of set `name` after its prefill fed by `path`; its argmax
    /// against ik's, a tie named and counted; with `band` — the batch set's
    /// tokens and the first layer a flip lies on the path of its last
    /// position — its layer outputs below that layer held to the band.
    fn step_set(
        m: &mut Qwen38Model,
        name: &str,
        path: Prompt38,
        band: Option<(&[u32], usize)>,
        ties: &mut usize,
    ) -> Result<(bool, Run), GateError> {
        let man = RefManifest::open(&data_dir().join(name), &IK)?;
        let (pos, step, prefill) = man.step()?;
        let (pos, step, prefill) = (pos, step.to_vec(), prefill.to_vec());
        let [tok] = step[..] else {
            return Err(format!("{name}: a step of {} tokens, not one", step.len()).into());
        };
        let held = match band {
            Some((toks, first)) => {
                if toks.split_last() != Some((&tok, &prefill[..])) {
                    return Err(format!(
                        "{name}: prefill {prefill:?} and step {tok} are not the batch set's {toks:?}"
                    )
                    .into());
                }
                first
            }
            None => 0,
        };
        fresh(m)?;
        let t = Instant::now();
        if !prefill.is_empty() {
            m.prompt38(&prefill, path)?;
        }
        let r = run_last(m, tok)?;
        let vocab = m.body("step_set")?.vocab();
        let ik = ik_last(&man, vocab)?;
        let ours = r.logits.last().ok_or("no logits")?;
        let (top, ik_top) = (argmax(ours), argmax(&ik));
        let ik_2 = second(&ik, ik_top);
        let margin = f64::from(ik[ik_top as usize]) - f64::from(ik[ik_2 as usize]);
        let dist = [ik_top, ik_2]
            .iter()
            .map(|&i| (f64::from(ours[i as usize]) - f64::from(ik[i as usize])).abs())
            .fold(0.0, f64::max);
        let logits_rel = rel(ours, &ik);
        let tie = top != ik_top && top == ik_2 && margin <= 2.0 * dist && logits_rel <= FREE_BAND;
        *ties += usize::from(tie);
        let rels = worst_of(&layer_table(&man, &r.taps)?);
        let inside = print_layers(name, &rels, held);
        let ok = (top == ik_top || tie) && inside;
        println!(
            "step {name}: position {pos} after {} fed by {} ({:.1} s, runtime value); argmax \
             ours={top} ik={ik_top} (ik's runner-up {ik_2}, margin {margin:.4}, our distance at \
             the two {dist:.4}{}); logits_rel {logits_rel:.3e} (printed; bounds a tie at \
             {FREE_BAND:.2}); worst \
             l_out_rel={:.3e} (band on layers 0..{held}, off every flip's path; the rest \
             printed) {}",
            prefill.len(),
            path.name(),
            t.elapsed().as_secs_f64(),
            if tie { ", a named tie" } else { "" },
            rels.iter().map(|r| r.0).fold(0.0, f64::max),
            verdict(ok)
        );
        Ok((ok, r))
    }

    // ------------------------------------ (q) the pass past the dense region

    /// D3K's prefill by passes against `steps`, its step-fed run: the step's
    /// token, logits, layer outputs and every store bit for bit.
    fn pass_selects(m: &mut Qwen38Model, steps: &Run) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(D3K), &IK)?;
        let (_, step, prefill) = man.step()?;
        let (step, prefill) = (step.to_vec(), prefill.to_vec());
        let [tok] = step[..] else {
            return Err(format!("{D3K}: a step of {} tokens, not one", step.len()).into());
        };
        fresh(m)?;
        let t = Instant::now();
        m.prompt38(&prefill, Prompt38::Pass)?;
        let fed = t.elapsed().as_secs_f64();
        let r = run_last(m, tok)?;
        let taps = r.taps.len() == steps.taps.len()
            && r.taps.iter().zip(&steps.taps).all(|(a, b)| same_bits(a, b));
        let mut ok = taps;
        println!(
            "pass {D3K}: {} ids by passes of up to {} ({fed:.1} s, runtime value), then the \
             step: layer outputs bit for bit the step-fed run's: {taps}",
            prefill.len(),
            Prompt38::PASS_ROWS
        );
        ok &= same_run(
            "D3K fed by passes vs fed by steps, the step after",
            &r,
            steps,
            false,
        );
        Ok(ok)
    }

    // ------------------------------------------------------- (v) verify

    /// The rows the verifies feed, the first T of them a verify's, and the
    /// token of the step after a commit: the batch set's tokens, rotated
    /// past the prefix.
    fn verify_rows(toks: &[u32]) -> Result<([u32; LANES], u32), GateError> {
        let t = |i: usize| -> Result<u32, GateError> {
            Ok(*toks
                .get(i % toks.len())
                .ok_or("the batch set holds no token")?)
        };
        Ok((
            [t(PREFIX)?, t(PREFIX + 1)?, t(PREFIX + 2)?, t(PREFIX + 3)?],
            t(PREFIX + 4)?,
        ))
    }

    /// A step's token and logits, or the error it failed with.
    type Stepped = Result<(u32, Vec<f32>), String>;

    /// What a run left after its last kept row: each row's token and logits
    /// (a verify's every row, a step run's every step), the stores and the
    /// PLE ring, the lane word, and the step after it.
    struct VRun {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        stores: Vec<Store38Host>,
        ple_ring: Vec<f32>,
        lane: u32,
        next: Stepped,
    }

    /// The step of `next` from where the model stands: its token and logits,
    /// or the error it failed with (the model reset behind it).
    fn next_step(m: &mut Qwen38Model, next: u32) -> Stepped {
        let r = m.step(&[next]).and_then(|t| Ok((t, m.logits()?)));
        r.map_err(|e| {
            let text = e.to_string();
            let _ = m.reset();
            text
        })
    }

    /// Where a (v) run stands before its rows: [`fresh`], then the prefix
    /// as graph steps or as one prompt by passes.
    #[derive(Clone, Copy)]
    enum Prefix<'a> {
        Steps(&'a [u32]),
        Pass(&'a [u32]),
    }

    impl Prefix<'_> {
        fn feed(self, m: &mut Qwen38Model) -> Result<(), GateError> {
            fresh(m)?;
            m.set_mode(StepMode::Graph);
            match self {
                Prefix::Steps(t) => m.step(t)?,
                Prefix::Pass(t) => m.prompt38(t, Prompt38::Pass)?,
            };
            Ok(())
        }
    }

    /// Steps of `feeds` (each a run of rows), every row a step, after the
    /// prefix: the rows' tokens and logits, the stores, then the step of
    /// `next`.
    fn ref_run(
        m: &mut Qwen38Model,
        prefix: Prefix<'_>,
        feeds: &[&[u32]],
        next: u32,
    ) -> Result<VRun, GateError> {
        prefix.feed(m)?;
        let (mut tokens, mut logits) = (Vec::new(), Vec::new());
        for &t in feeds.iter().flat_map(|f| f.iter()) {
            tokens.push(m.step(&[t])?);
            logits.push(m.logits()?);
        }
        let (stores, ple_ring) = stores(m)?;
        let lane = m.body("ref_run")?.lane();
        Ok(VRun {
            tokens,
            logits,
            stores,
            ple_ring,
            lane,
            next: next_step(m, next),
        })
    }

    /// A verify of `rows` in `mode` from where the model stands, kept to its
    /// first `k` rows: its rows' tokens and logits, the host tier's services,
    /// host calls and `Cols` services during it, the stores after the
    /// commit, the lane word, then the step of `next` (when given).
    fn verify<const T: usize>(
        m: &mut Qwen38Model,
        rows: [u32; T],
        k: usize,
        mode: StepMode,
        next: Option<u32>,
    ) -> Result<(VRun, [u64; 3]), GateError> {
        m.set_mode(mode);
        let before = m.body("verify")?.hybrid().stats();
        let pos0 = m.pos();
        let tokens = m.step_rows::<T>(rows)?.to_vec();
        let logits = m.rows_logits::<T>()?.to_vec();
        let after = m.body("verify")?.hybrid().stats();
        let served = [
            after.served - before.served,
            after.host_calls - before.host_calls,
            after.cols_served - before.cols_served,
        ];
        m.rollback(pos0 + k as u32)?;
        m.set_mode(StepMode::Graph);
        let (stores, ple_ring) = stores(m)?;
        let lane = m.body("verify")?.lane();
        let next = match next {
            Some(t) => next_step(m, t),
            None => Err("no step after".to_string()),
        };
        Ok((
            VRun {
                tokens,
                logits,
                stores,
                ple_ring,
                lane,
                next,
            },
            served,
        ))
    }

    /// The layers whose live stores differ at count `pos`, and whether the
    /// PLE rings' live slots do ((v)'s rule: the committed lane, the conv
    /// ring's eight positions before the count, the K/V planes' and raw
    /// keys' rows below it, the pools complete at the count, the PLE ring's
    /// fourteen positions before the count).
    fn live_diff(a: &VRun, b: &VRun, pos: usize) -> (Vec<usize>, bool) {
        let conv_ch = 2 * K_HEADS * HEAD_V + V_HEADS * HEAD_V;
        let slots =
            |ring: usize, back: usize| (pos.saturating_sub(back)..pos).map(move |p| p % ring);
        let ring_same = |x: &[f32], y: &[f32], ring: usize, width: usize, back: usize| {
            x.len() == ring * width
                && y.len() == ring * width
                && slots(ring, back).all(|s| {
                    same_bits(
                        &x[s * width..(s + 1) * width],
                        &y[s * width..(s + 1) * width],
                    )
                })
        };
        // A plane of `heads` heads of `ctx` rows of `width` (`ctx` read off
        // its length), compared on the rows below the count.
        let rows_same = |x: &[u16], y: &[u16], heads: usize, width: usize| {
            let ctx = x.len() / (heads * width);
            x.len() == y.len()
                && x.len() == heads * ctx * width
                && pos <= ctx
                && (0..heads).all(|h| {
                    let at = h * ctx * width;
                    x[at..at + pos * width] == y[at..at + pos * width]
                })
        };
        let n = a.stores.len().max(b.stores.len());
        let layers = (0..n)
            .filter(|&l| match (a.stores.get(l), b.stores.get(l)) {
                (
                    Some(Store38Host::Rec { state: s, ring: r }),
                    Some(Store38Host::Rec {
                        state: s2,
                        ring: r2,
                    }),
                ) => !(same_bits(s, s2) && ring_same(r, r2, CONV_RING, conv_ch, 8)),
                (
                    Some(Store38Host::Qsa { k, v, raw, pooled }),
                    Some(Store38Host::Qsa {
                        k: k2,
                        v: v2,
                        raw: raw2,
                        pooled: pooled2,
                    }),
                ) => {
                    // A pool past the count is read only at a count that
                    // completes it, and that row writes it before its select.
                    let live = pos / POOL * IDX_DIM;
                    !(rows_same(k, k2, N_KV, HEAD)
                        && rows_same(v, v2, N_KV, HEAD)
                        && rows_same(raw, raw2, 1, IDX_DIM)
                        && pooled.len() == pooled2.len()
                        && live <= pooled.len()
                        && pooled[..live] == pooled2[..live])
                }
                _ => true,
            })
            .collect();
        let ple = !ring_same(&a.ple_ring, &b.ple_ring, PLE_RING, STREAMS * HIDDEN, 14);
        (layers, ple)
    }

    /// `got` — a verify kept to `k` rows — against `want`, the steps of those
    /// rows: every row's token and logits `got` ran that `want` holds, the
    /// live stores, the lane word the gate computes (`want_lane`), and the
    /// step after.
    fn same_vrun(label: &str, got: &VRun, want: &VRun, pos: usize, want_lane: u32) -> bool {
        let n = got.tokens.len().min(want.tokens.len());
        let tokens = got.tokens[..n] == want.tokens[..n];
        let logits = (0..n).all(|r| same_bits(&got.logits[r], &want.logits[r]));
        let (layers, ple) = live_diff(got, want, pos);
        let lane = got.lane == want_lane;
        let next = match (&got.next, &want.next) {
            (Ok((t, l)), Ok((t2, l2))) => t == t2 && same_bits(l, l2),
            _ => false,
        };
        let ok = tokens && logits && layers.is_empty() && !ple && lane && next;
        println!(
            "verify {label}: rows {:?} vs steps {:?} ({n} compared), logits bit for bit: \
             {logits}; live stores differing at layers {layers:?}, PLE ring: {ple}; lane {} \
             (want {want_lane}, computed here); the step after: {} {}",
            got.tokens,
            want.tokens,
            got.lane,
            match (&got.next, &want.next) {
                (Ok((t, _)), Ok((t2, _))) => format!("{t} vs {t2}, logits bit for bit {next}"),
                (Err(e), _) => format!("FAILED: {e}"),
                (_, Err(e)) => format!("the reference FAILED: {e}"),
            },
            verdict(ok)
        );
        ok
    }

    /// The structure of the captured verify of `T` rows: the node count, its
    /// batch mem ops, its argmax and handoff launches.
    fn verify_structure<const T: usize>(m: &mut Qwen38Model) -> Result<bool, GateError> {
        let nodes = m.capture_rows::<T>()?;
        let counted = m.body("verify_structure")?.verify_launches(T);
        let list = m.rows_graph_nodes::<T>()?;
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([_, b], other) = count_kinds(&list, [kernel, memop]);
        let named = |f: &dyn Fn(&str) -> bool| {
            list.iter()
                .filter(|n| n.kernel.as_ref().is_some_and(|k| f(&k.name)))
                .count()
        };
        let argmax = named(&|n| n.contains("argmax"));
        let cols = named(&|n| n == "ds41_ffn_handoff_10_cols");
        let one = named(&|n| n == "ds41_ffn_handoff_10");
        let ok = nodes == NODES_VERIFY
            && counted == NODES_VERIFY
            && b == MEMOPS
            && other == 0
            && argmax == 1
            && cols == N_LAYER
            && one == 0;
        println!(
            "verify structure T={T}: graph_nodes={nodes} (want {NODES_VERIFY}; the program counts \
             {counted}) batch_mem_op={b} (want {MEMOPS}) other={other}; argmax launches {argmax} \
             (want 1: one head of {T} rows); ds41_ffn_handoff_10_cols {cols} (want {N_LAYER}), \
             ds41_ffn_handoff_10 {one} (want 0) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// One verify of `T` rows kept to `k`, against `refs[k − 1]`: the case,
    /// its structure of services, and a failed call named, not aborting.
    fn verify_case<const T: usize>(
        m: &mut Qwen38Model,
        toks: &[u32],
        rows: [u32; LANES],
        next: u32,
        k: usize,
        refs: &[VRun],
    ) -> Result<bool, GateError> {
        let label = format!("T={T} k={k}");
        let run = (|| -> Result<(VRun, [u64; 3], u32), GateError> {
            Prefix::Steps(&toks[..PREFIX]).feed(m)?;
            let c = m.body("verify_case")?.lane();
            let r: [u32; T] = std::array::from_fn(|i| rows[i]);
            let (v, served) = verify::<T>(m, r, k, StepMode::Graph, Some(next))?;
            Ok((v, served, c))
        })();
        let (v, served, c) = match run {
            Ok(x) => x,
            Err(e) => {
                println!("verify {label}: FAILED: {e} {}", verdict(false));
                m.reset()?;
                return Ok(false);
            }
        };
        let want = refs.get(k - 1).ok_or("a reference for every kept count")?;
        let mut ok = same_vrun(
            &label,
            &v,
            want,
            PREFIX + k,
            (c + k as u32 - 1) % LANES as u32,
        );
        // Rows past k: the verify's own, against the steps of all four.
        let all = refs.last().ok_or("the reference of four rows")?;
        let rows_ok = v.tokens[..] == all.tokens[..T]
            && (0..T).all(|r| same_bits(&v.logits[r], &all.logits[r]));
        let services = served == [N_LAYER as u64; 3];
        ok &= rows_ok && services;
        println!(
            "verify {label}: every row's token and logits bit for bit the steps' {rows_ok}; the \
             replay's services, host calls and Cols services {served:?} (want {N_LAYER} each: \
             one go, one union call and one wait a layer) {}",
            verdict(rows_ok && services)
        );
        Ok(ok)
    }

    fn verify_clause(m: &mut Qwen38Model, toks: &[u32]) -> Result<bool, GateError> {
        let (rows, next) = verify_rows(toks)?;
        let mut ok = true;
        ok &= verify_structure::<2>(m)?;
        ok &= verify_structure::<3>(m)?;
        ok &= verify_structure::<4>(m)?;
        let refs = (1..=LANES)
            .map(|k| ref_run(m, Prefix::Steps(&toks[..PREFIX]), &[&rows[..k]], next))
            .collect::<Result<Vec<_>, _>>()?;
        for k in 1..=2 {
            ok &= verify_case::<2>(m, toks, rows, next, k, &refs)?;
        }
        for k in 1..=3 {
            ok &= verify_case::<3>(m, toks, rows, next, k, &refs)?;
        }
        for k in 1..=4 {
            ok &= verify_case::<4>(m, toks, rows, next, k, &refs)?;
        }
        // A second verify from a moved lane: four rows kept to three (lane 2),
        // then four more kept whole (lanes 2, 3, 0, 1: the word to 1).
        let want = ref_run(
            m,
            Prefix::Steps(&toks[..PREFIX]),
            &[&rows[..3], &rows[..]],
            next,
        )?;
        let chained = (|| -> Result<VRun, GateError> {
            Prefix::Steps(&toks[..PREFIX]).feed(m)?;
            verify::<4>(m, rows, 3, StepMode::Graph, None)?;
            Ok(verify::<4>(m, rows, 4, StepMode::Graph, Some(next))?.0)
        })();
        match chained {
            Ok(v) => {
                // The second verify's rows are the reference's last four.
                let (t, l) = (want.tokens[3..].to_vec(), want.logits[3..].to_vec());
                let tail = VRun {
                    tokens: t,
                    logits: l,
                    stores: want.stores,
                    ple_ring: want.ple_ring,
                    lane: want.lane,
                    next: want.next,
                };
                ok &= same_vrun(
                    "after a commit of 3, T=4 k=4 from lane 2",
                    &v,
                    &tail,
                    PREFIX + 7,
                    (2 + 4 - 1) % LANES as u32,
                );
            }
            Err(e) => {
                println!("verify chained: FAILED: {e} {}", verdict(false));
                m.reset()?;
                ok = false;
            }
        }
        // Eager verify against the replay.
        let eager = (|| -> Result<(VRun, VRun), GateError> {
            fresh(m)?;
            m.step(&toks[..PREFIX])?;
            let g = verify::<4>(m, rows, 4, StepMode::Graph, Some(next))?.0;
            fresh(m)?;
            m.step(&toks[..PREFIX])?;
            let e = verify::<4>(m, rows, 4, StepMode::Eager, Some(next))?.0;
            Ok((e, g))
        })();
        match eager {
            Ok((e, g)) => {
                ok &= same_vrun("eager T=4 k=4 vs its replay", &e, &g, PREFIX + 4, g.lane);
            }
            Err(e) => {
                println!("verify eager: FAILED: {e} {}", verdict(false));
                m.reset()?;
                ok = false;
            }
        }
        ok &= verify_deep(m)?;
        ok &= verify_refusals(m, toks, rows)?;
        Ok(ok)
    }

    /// The deep verify: D3K's first [`DEEP`] prefill ids by passes, then a
    /// verify of the next four kept to [`DEEP_KEPT`], against the steps of
    /// those rows; the pooled rows past the count poisoned, then the two
    /// steps after, the prefill's next ids.
    fn verify_deep(m: &mut Qwen38Model) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(D3K), &IK)?;
        let (_, _, prefill) = man.step()?;
        let ids = prefill
            .get(..DEEP + LANES)
            .ok_or_else(|| format!("{D3K}: a prefill of {} ids", prefill.len()))?
            .to_vec();
        let (prefix, rows) = (Prefix::Pass(&ids[..DEEP]), &ids[DEEP..]);
        let rows: [u32; LANES] = std::array::from_fn(|i| rows[i]);
        let (next, then) = (rows[DEEP_KEPT], rows[DEEP_KEPT + 1]);
        let want = ref_run(m, prefix, &[&rows[..DEEP_KEPT]], next)?;
        let want_then = next_step(m, then);
        let run = (|| -> Result<(VRun, u32, Stepped), GateError> {
            prefix.feed(m)?;
            let c = m.body("verify_deep")?.lane();
            let mut v = verify::<4>(m, rows, DEEP_KEPT, StepMode::Graph, None)?.0;
            {
                let (gpu, _, b) = m.body_parts("verify_deep")?;
                b.poison_pools_from(gpu, (DEEP + DEEP_KEPT) / POOL, POOL_POISON)?;
            }
            v.next = next_step(m, next);
            let got_then = next_step(m, then);
            Ok((v, c, got_then))
        })();
        match run {
            Ok((v, c, got_then)) => {
                let mut ok = same_vrun(
                    &format!(
                        "deep at {DEEP}, T=4 k={DEEP_KEPT}, pool 513's row rejected, the pools \
                         past the count poisoned"
                    ),
                    &v,
                    &want,
                    DEEP + DEEP_KEPT,
                    (c + DEEP_KEPT as u32 - 1) % LANES as u32,
                );
                let then_ok = matches!(
                    (&got_then, &want_then),
                    (Ok((t, l)), Ok((t2, l2))) if t == t2 && same_bits(l, l2)
                );
                ok &= then_ok;
                println!(
                    "verify deep: the step after that (count {}, completing pool 513 again \
                     before its select): {} {}",
                    DEEP + DEEP_KEPT + 2,
                    match (&got_then, &want_then) {
                        (Ok((t, _)), Ok((t2, _))) => {
                            format!("{t} vs {t2}, logits bit for bit {then_ok}")
                        }
                        (Err(e), _) => format!("FAILED: {e}"),
                        (_, Err(e)) => format!("the reference FAILED: {e}"),
                    },
                    verdict(then_ok)
                );
                Ok(ok)
            }
            Err(e) => {
                println!("verify deep: FAILED: {e} {}", verdict(false));
                m.reset()?;
                Ok(false)
            }
        }
    }

    /// (v)'s refusals: a step while a verify waits, a commit with no verify,
    /// and the planted lane's fault.
    fn verify_refusals(
        m: &mut Qwen38Model,
        toks: &[u32],
        rows: [u32; LANES],
    ) -> Result<bool, GateError> {
        let mut ok = true;
        fresh(m)?;
        m.step(&toks[..PREFIX])?;
        m.step_rows::<2>([rows[0], rows[1]])?;
        let got = m.step(&[rows[2]]);
        let named = matches!(&got, Err(e) if e.to_string().contains("waits for its commit"));
        ok &= named;
        println!(
            "verify refusal: a step while the verify waits -> {} {}",
            match &got {
                Ok(t) => format!("accepted, next {t}"),
                Err(e) => e.to_string(),
            },
            verdict(named)
        );
        m.rollback(PREFIX as u32 + 2)?;
        let got = m.rollback(PREFIX as u32);
        let named = matches!(&got, Err(e) if e.to_string().contains("with no verify waiting"));
        ok &= named;
        println!(
            "verify refusal: a commit back to {PREFIX} with no verify waiting -> {} {}",
            match &got {
                Ok(()) => "accepted".to_string(),
                Err(e) => e.to_string(),
            },
            verdict(named)
        );
        fresh(m)?;
        m.step(&toks[..PREFIX])?;
        {
            let (gpu, _, b) = m.body_parts("plant")?;
            b.plant_lane(gpu, 2)?;
        }
        let got = m.step(&[rows[0]]);
        let want = Fault::at(0, FaultSite::DeltaStamp);
        let raised = matches!(&got, Err(GpuError::Fault { fault, .. }) if *fault == want);
        ok &= raised;
        println!(
            "verify refusal: the lane word planted on lane 2, never written -> {} (want {want}) {}",
            match &got {
                Ok(t) => format!("accepted, next {t}"),
                Err(e) => e.to_string(),
            },
            verdict(raised)
        );
        m.reset()?;
        Ok(ok)
    }

    // ------------------------------------------- (o) one owner of the position

    /// The calls (o) plants its failures on, each from position [`PREFIX`].
    #[derive(Clone, Copy)]
    enum Call {
        Step,
        Pass,
        Verify,
    }

    impl Call {
        fn name(self) -> &'static str {
            match self {
                Call::Step => "a graph step",
                Call::Pass => "an eager pass of 3 rows",
                Call::Verify => "a graph verify of 2 rows",
            }
        }

        /// The call at [`PREFIX`]: its last row's token.
        fn run(self, m: &mut Qwen38Model, toks: &[u32], rows: [u32; 2]) -> Result<u32, GpuError> {
            match self {
                Call::Step => m.step(&toks[PREFIX..PREFIX + 1]),
                Call::Pass => m.prompt38(&toks[PREFIX..PREFIX + 3], Prompt38::Pass),
                Call::Verify => m.step_rows::<2>(rows).map(|t| t[1]),
            }
        }
    }

    /// What a clean call left: its last row's token and logits, the PLE
    /// rows its fill named, and every store with the PLE ring after it (a
    /// verify's after its commit of both rows).
    struct Owned {
        token: u32,
        logits: Vec<f32>,
        rows: Vec<u32>,
        stores: Vec<Store38Host>,
        ple_ring: Vec<f32>,
    }

    /// The call `c` that just returned `token`, read back.
    fn owned(m: &mut Qwen38Model, c: Call, token: u32) -> Result<Owned, GateError> {
        let rows = m.body("owned")?.ple_rows().to_vec();
        let logits = match c {
            Call::Verify => {
                let [_, last] = m.rows_logits::<2>()?;
                m.rollback(PREFIX as u32 + 2)?;
                last
            }
            Call::Step | Call::Pass => m.logits()?,
        };
        let (stores, ple_ring) = stores(m)?;
        Ok(Owned {
            token,
            logits,
            rows,
            stores,
            ple_ring,
        })
    }

    /// [`fresh`], the prefix as graph steps, then the call `c`.
    fn clean_call(
        m: &mut Qwen38Model,
        c: Call,
        toks: &[u32],
        rows: [u32; 2],
    ) -> Result<Owned, GateError> {
        Prefix::Steps(&toks[..PREFIX]).feed(m)?;
        let token = c.run(m, toks, rows)?;
        owned(m, c, token)
    }

    /// `got` against the clean call `want`, bit for bit: a line's tail.
    fn same_owned(got: &Owned, want: &Owned) -> (bool, String) {
        let logits = same_bits(&got.logits, &want.logits);
        let rows = got.rows == want.rows && !got.rows.is_empty();
        let layers: Vec<usize> = (0..got.stores.len().max(want.stores.len()))
            .filter(|&l| match (got.stores.get(l), want.stores.get(l)) {
                (Some(x), Some(y)) => !x.same_bits(y),
                _ => true,
            })
            .collect();
        let ring = same_bits(&got.ple_ring, &want.ple_ring);
        let ok = got.token == want.token && logits && rows && layers.is_empty() && ring;
        (
            ok,
            format!(
                "token {} (clean {}), logits bit for bit: {logits}, PLE rows equal: {rows}, \
                 stores differing at layers {layers:?}, PLE ring equal: {ring}",
                got.token, want.token
            ),
        )
    }

    fn text<T: std::fmt::Display>(r: &Result<T, GpuError>) -> String {
        match r {
            Ok(t) => format!("accepted ({t})"),
            Err(e) => format!("error \"{e}\""),
        }
    }

    /// (o) for the call `c`: a failure before its launch, then one after.
    fn owner_call(
        m: &mut Qwen38Model,
        c: Call,
        toks: &[u32],
        rows: [u32; 2],
    ) -> Result<bool, GateError> {
        let clean = clean_call(m, c, toks, rows)?;
        let at = PREFIX as u32;

        Prefix::Steps(&toks[..PREFIX]).feed(m)?;
        let lane = m.body("owner")?.lane();
        m.body_parts("owner")?.2.plant_before_launch();
        let first = c.run(m, toks, rows);
        let (pos, lane_after) = (m.pos(), m.body("owner")?.lane());
        let failed = matches!(&first, Err(e)
            if e.to_string().contains("the planted failure before the launch"));
        let again = c.run(m, toks, rows);
        let (rerun_ok, rerun) = match &again {
            Ok(t) => same_owned(&owned(m, c, *t)?, &clean),
            Err(_) => (false, String::new()),
        };
        let before_ok = failed
            && pos == at
            && lane_after == lane
            && m.poisoned().is_none()
            && again.is_ok()
            && rerun_ok;
        println!(
            "position owner, {}: a failure before the launch at {at}: {} at position {pos} (want \
             {at}), lane {lane_after} (want {lane}); the call again: {} {rerun} {}",
            c.name(),
            text(&first),
            text(&again),
            verdict(before_ok)
        );

        Prefix::Steps(&toks[..PREFIX]).feed(m)?;
        m.body_parts("owner")?.2.plant_after_launch();
        let first = c.run(m, toks, rows);
        let pos = m.pos();
        let failed = matches!(&first, Err(e)
            if e.to_string().contains("the planted failure after the launch"));
        let kept = m.body("owner")?.kept(at, pos);
        let step = m.step(&toks[PREFIX..PREFIX + 1]);
        let refused = matches!(&step, Err(e)
            if e.to_string().contains("failed after its chain was launched"));
        let (pos_step, poisoned) = (m.pos(), m.poisoned());
        let back = clean_call(m, c, toks, rows)?;
        let (back_ok, back_line) = same_owned(&back, &clean);
        let after_ok = failed
            && pos == at
            && kept.at < at
            && refused
            && pos_step == at
            && poisoned.is_none()
            && back_ok;
        println!(
            "position owner, {}: a failure after the launch at {at}: {} at position {pos} (want \
             {at}); a cut keeps {} ({kept}, want under {at}); a step at {at}: {} (want refused \
             by name), position {pos_step}; after reset the call again: {back_line} {}",
            c.name(),
            text(&first),
            kept.at,
            text(&step),
            verdict(after_ok)
        );
        m.reset()?;
        Ok(before_ok && after_ok)
    }

    /// (o): each call's failures, before and after its launch.
    fn position_owner(m: &mut Qwen38Model, toks: &[u32]) -> Result<bool, GateError> {
        let (rows, _) = verify_rows(toks)?;
        let rows = [rows[0], rows[1]];
        let mut ok = true;
        for c in [Call::Step, Call::Pass, Call::Verify] {
            ok &= owner_call(m, c, toks, rows)?;
        }
        Ok(ok)
    }

    // ---------------------------------------------------- (r) refusals

    fn refusals(m: &mut Qwen38Model) -> Result<bool, GateError> {
        let mut ok = true;
        for s in ["gemm", "auto"] {
            let got = Prompt38::parse(s);
            let named = matches!(&got, Err(e) if e.to_string().contains("gemm_q8_0"));
            ok &= named;
            println!(
                "refusal: --prefill {s} -> {} {}",
                match &got {
                    Ok(p) => format!("accepted as {}", p.name()),
                    Err(e) => e.to_string(),
                },
                verdict(named)
            );
        }
        m.reset()?;
        m.step(&[0])?;
        // A step refuses its one token; a pass refuses the whole pass before
        // any row of it runs.
        for (path, ids) in [
            (Prompt38::Step, &[IMAGE_TOKEN][..]),
            (Prompt38::Pass, &[1, IMAGE_TOKEN][..]),
        ] {
            let before = m.pos();
            let got = m.prompt38(ids, path);
            let named =
                matches!(&got, Err(e) if e.to_string().contains("is the image placeholder"));
            let kept = m.pos() == before && m.poisoned().is_none();
            ok &= named && kept;
            println!(
                "refusal: the image placeholder {IMAGE_TOKEN} by {} at position {before} -> {}; \
                 position {} (kept: {kept}) {}",
                path.name(),
                match &got {
                    Ok(t) => format!("accepted, next {t}"),
                    Err(e) => e.to_string(),
                },
                m.pos(),
                verdict(named && kept)
            );
        }
        let before = m.pos();
        let past = vec![0u32; CTX + 1 - before as usize];
        for path in [Prompt38::Step, Prompt38::Pass] {
            let got = m.prompt38(&past, path);
            let named = matches!(&got, Err(e) if e.to_string().contains("passes the stores"));
            let kept = m.pos() == before;
            ok &= named && kept;
            println!(
                "refusal: {} ids by {} from position {before} past {CTX} -> {}; position {} \
                 (kept: {kept}) {}",
                past.len(),
                path.name(),
                match &got {
                    Ok(t) => format!("accepted, next {t}"),
                    Err(e) => e.to_string(),
                },
                m.pos(),
                verdict(named && kept)
            );
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let mut m = open()?;
        let mut ok = structure(&mut m)?;
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let (_, toks, _) = man.step()?;
        let toks = toks.to_vec();
        let (paths_ok, eager) = paths(&mut m, &toks)?;
        ok &= paths_ok;
        let vocab = m.body("run")?.vocab();
        let (free_ok, firsts) = free(&man, &eager, vocab)?;
        ok &= free_ok;
        drop(eager);
        let last = *firsts.last().ok_or("no positions")?;
        let mut ties = 0usize;
        let band = Some((&toks[..], last));
        for (name, band) in [(STEP4, band), (STEP4_EVERY_NODE, band), (D1K, None)] {
            ok &= step_set(&mut m, name, Prompt38::Step, band, &mut ties)?.0;
        }
        let (d3k_ok, d3k) = step_set(&mut m, D3K, Prompt38::Step, None, &mut ties)?;
        ok &= d3k_ok;
        println!(
            "{D3K}: ik cuts its selection by cells and keeps up to three keys of the 513th pool \
             ours does not read (a named difference, its logits printed)"
        );
        println!("step sets: {ties} named tie(s)");
        ok &= pass_selects(&mut m, &d3k)?;
        drop(d3k);
        ok &= verify_clause(&mut m, &toks)?;
        ok &= position_owner(&mut m, &toks)?;
        ok &= refusals(&mut m)?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
