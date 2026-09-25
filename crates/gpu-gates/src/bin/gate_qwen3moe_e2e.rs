//! The qwen3moe end-to-end gate: the whole chain — 48 layers, the head and
//! the argmax — on one card, against ik's CPU oracle and its greedy answers.
//!
//! Every prompt runs from position 0 (`GpuModel::reset`), its tokens fed one
//! position at a time through the decode path — the greedy reference was
//! written the same way (`argmax_ref --step-prefill`) — and then again
//! through the prompt prefill (`Qwen3moeModel::prefill`): on the pass path,
//! which must leave the same answer bit for bit, and on the GEMM ubatches,
//! which must stay inside a derived band.
//!
//! What is asserted:
//! - (s) structure: the captured step holds [`NODES_CHAIN`] nodes, none of
//!   them a memcpy (the combine writes the next layer's input in place) or
//!   a host node; the captured prefill pass holds [`NODES_PASS_1`] nodes at
//!   one token and [`NODES_PASS_M`] at every count from two to `MAX_TOKENS`,
//!   all of them kernels.
//! - (f) teacher-forced, per layer: each layer run alone on ik's own input
//!   row (`inp_embd` for layer 0, `l_out-(L−1)` after) at each of the
//!   oracle's five positions. The attention half against ik's `attn_out-L`
//!   (`‖(ffn_inp − x_in) − attn_out‖ / ‖attn_out‖`), and the FFN half run
//!   alone on ik's own FFN input `x_in + attn_out` against ik's
//!   `routed_out-L`, measured on the magnitude of the routed sum's terms
//!   (`Σ_s |w_s · down_s|`, which the eight terms' cancellation cannot
//!   shrink). Each half's error over the gap between the two sides' 8-bit
//!   inputs to it must stay within [`RATIO_BAND`]. A site where the router
//!   picks an expert ik does not is counted and printed, not judged.
//! - (c) free-running: the five tokens through the whole chain, every
//!   layer's `l_out` against ik's, `‖ours − ik‖ / ‖ik‖` within
//!   [`FREE_BAND`]; the last position's logits against ik's
//!   `result_output`: the same argmax, the relative distance printed.
//! - (g) greedy: prompts `0..PROMPTS` of `tools/ref/prompts.tsv` under this
//!   model's tokenizer (`just ik-greedy-qwen3moe` writes them and ik's
//!   continuations), [`GEN`] tokens each in graph mode: no prompt diverges
//!   from ik at a position where ik's own top1-top2 margin is at or above
//!   [`MARGIN_FLOOR`] — `gpu_gates::prompts::compare_greedy`'s classes, the
//!   V2-Lite gate's criterion; later divergences and near ties are printed.
//! - (r) graph replay equals the eager body: the same prompts in eager mode
//!   give the same tokens, and the last step's logits are bit-identical.
//! - (p) prefill equals the one-token path: the same prompts prefilled
//!   (each fits one pass, so the default path takes the pass), then the same
//!   greedy steps, give the same tokens and bit-identical last logits; so do
//!   the prompts' concatenation on the pass path (several passes) against its
//!   own one-token run, whole and in two calls split at [`SPLIT`], the second
//!   starting at a position that is not a pass boundary. These run through
//!   the captured passes.
//! - (q) a replayed prefill pass equals the eager one: for every pass size
//!   `m`, the concatenation's first `MAX_TOKENS + m` ids (a full pass, then
//!   one of `m`) prefilled on the pass path in graph mode and in eager mode,
//!   each from a reset over cache rows seeded with a pattern (`seed_depth`),
//!   leave the same K/V rows bit for bit, none of them still the pattern, and
//!   the same last logits bit for bit.
//! - (u) the GEMM prefill against the one-token path, on the first
//!   [`LONG`] ids of the prose file `corpus-prose.ids` (its digest checked),
//!   at the load's ubatch size (the default, 4096, clipped to the cache: each
//!   prompt one ubatch). Layer 0's K and V rows within [`GEMM_L0_REL`] of the
//!   one-token path's; each later layer's distance within
//!   [`GEMM_SPREAD_RATIO`] times the one-token path's own distance between
//!   its two flash arithmetics (the same prompt stepped eagerly on the other
//!   flash pass, `set_flash_mma`); the last logits within the same ratio of
//!   that run's; the greedy continuation after the prefill judged as (g)
//!   against the one-token path's own continuation and margins. And the 1,300 ids
//!   prefilled in two calls cut at [`LONG_SPLIT`] (every unit a ubatch, the
//!   second call's first position off every ubatch boundary) leave the whole
//!   prefill's K/V rows and last logits bit for bit: a token's GEMM-path
//!   values do not depend on the ubatch it lands in.
//! - (w) the ubatch size moves no bit: a model opened with [`CTX_W`] cache
//!   rows prefills the prose's first [`W_LONG`] ids at each size of
//!   [`W_SIZES`] (ubatches of 512 x 8; 1,000 x 4 and 96; one of 4,096 —
//!   32,768 routed slots in one table) and every run leaves the K/V rows,
//!   the last logits and the token of the 4,096 run bit for bit; so do one
//!   id more at 512 and at 4,096 (each ending in a one-id pass), and the
//!   4,096 ids at 4,096 prefilled in two calls cut at [`W_SPLIT`]. Sizes
//!   outside `1..=UBATCH` are refused with the size kept.
//! - (x) a fault names its layer: layer [`FAULT_LAYER`]'s FFN half run alone
//!   on a finite row with one NaN raises inside that layer's launches, and
//!   the next step returns `GpuError::Fault` with that layer (the site is the
//!   smallest code among the layer's raises, printed), the model poisoned;
//!   `reset` leaves the word clean and the next step a token.
//!
//! `--gemm-only` runs the load, (u) and (w), `--ubatch-only` (w) alone,
//! `--fault-only` the load and (x).
//!
//! The flash pass is read once per process (`BLOOMERY_GQA_MMA`), so each
//! pass is its own run; the `load` line names it.
//!
//! `--dump DIR` also writes each greedy prompt's tokens (u32 LE) and last
//! logits (f32 LE) of the graph, the eager and the prefilled pass as raw
//! files under DIR (`p{i}-{graph,eager,prefill}.{tokens,logits}`), for a
//! byte comparison across builds (`md5sum DIR/*`).
//!
//! `--ppl TAG` instead scores the chain against ik's KL-divergence base file
//! `$BLOOMERY_DATA/ikppl/TAG.kld` (`tools/ref/ik-ppl.sh --kld-base`): chunk by
//! chunk from a reset, one step per id, at every scored position our NLL,
//! ik's, KL(ik‖ours) and the top-1 agreement. Printed, not judged.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_e2e: built without the `gpu` feature; see `just gate-gpu-qwen3moe-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_e2e", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::PrefillPath;
    use bloomery_gpu::arch::qwen3moe::router::MAX_TOKENS;
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
    use bloomery_gpu::flash_gqa::HEAD;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{GpuError, Qwen3moeModel};
    use bloomery_gpu_gates::kld::{KldBase, PplModel, score_ppl};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::oracle::{self, Set};
    use bloomery_gpu_gates::prompts::{
        GreedyClass, GreedyRow, PromptRow, compare_greedy, read_greedy, read_prompts,
    };
    use bloomery_gpu_gates::{
        GateError, Layout, RefManifest, RowKind, bits_equal, checks_failed, data_dir, ik_q8_2,
        open_split, q8_1_dequant, ref_ints, ref_tensor_logical_in, topk_ids_logical_within,
        verdict,
    };
    use cuda_core::sys;
    use model::arch::Arch;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    /// Generated tokens per prompt: the greedy files' width.
    const GEN: usize = 32;

    /// Where the split prefill of the concatenated prompts cuts them: inside
    /// the first pass, so the second call starts off a pass boundary.
    const SPLIT: usize = 5;

    /// Prompts of the greedy arm: rows `0..PROMPTS` of `tools/ref/prompts.tsv`.
    const PROMPTS: usize = 8;

    /// Cache rows: the longest of [`LONG`] plus `GEN`, with room (the
    /// oracle's five tokens and the prompts' concatenation fit far below).
    const CTX: usize = 1344;

    /// The GEMM clause's prompt lengths: three ubatch-sized units each, the
    /// last a one-id pass and a 276-id ubatch.
    const LONG: [usize; 2] = [1025, 1300];

    /// Where the split GEMM prefill of the longer prompt cuts it: off every
    /// ubatch boundary, so both calls end in a partial ubatch.
    const LONG_SPLIT: usize = 700;

    /// The prose the GEMM clause reads, and its digest as
    /// `tools/ref/models/qwen3moe.sh` pins it.
    const PROSE: &str = "corpus-prose.ids";
    const PROSE_SHA256: &str = "9444bc5b4e2a7c5f4caac4f1aa7b6ef8e945b114fdecac52aca36a1de4e76cf6";

    /// PIN(2026-09-25): the GEMM prefill's band on layer 0's K and V rows
    /// against the one-token path's (`‖ours − one-token‖ / ‖one-token‖`).
    /// Layer 0's input is the same bits on both paths (the embedding rows;
    /// `rms_norm` then the quantizer, the bytes `norm_quant` writes), so only
    /// the k and v projections differ, each within its own sum's rounding of
    /// the exact dot of the same codes: the GEMM's `γ(nb + 4) · Σ_b (|D_b| +
    /// |M_b|)` (gate_gemm), the gemv's lane partials and warp tree the same
    /// order, about 1.2e-6 of `Σ_b |terms|` each at K = 2048, which is about
    /// √16 = 4 times a value's size for blocks of either sign: Δ ≤ 1e-5 of the
    /// row. The head norm and the rope carry that relative distance; the f16
    /// rounding then moves a value by one ulp (at most 2^-10 of it) with
    /// probability Δ/ulp, so the rows' distance is at most √(Δ · ulp) ≈
    /// √(1e-5 · 1e-3) = 1e-4. A value much smaller than its row can move by
    /// several of its own ulps under the same absolute Δ, so the count of
    /// moved values and their largest move in ulp are printed, not judged.
    const GEMM_L0_REL: f64 = 1e-4;

    /// PIN(2026-09-25): the GEMM prefill's K/V distance from the one-token
    /// path at every layer past 0, over the one-token path's own distance
    /// between its two gated flash arithmetics on the same prompt (the
    /// tensor-core pass's f16 scores against the scalar pass's f32), each
    /// layer's `max(K, V)` over `max(K, V)`; and the same ratio of the last
    /// logits' relative distances. Derivation: past layer 0 the two
    /// paths' first difference (the products' sum order, about 1e-6 of the
    /// terms, [`GEMM_L0_REL`]'s note) grows at every q8_1 re-quantization by
    /// code flips — a value crosses a rounding boundary with probability
    /// δ/d8 and then moves by d8, so δ' ≈ √(δ · d8/σ) with d8/σ ≈ 2.3e-2 —
    /// until, within two or three layers, the two paths round like two
    /// independent roundings of one rule. From there the distance is the
    /// model's own amplification of rounding-sized differences, which no
    /// isotropic model gives (an isotropic `√(l + 1) · 4e-2`, the forced
    /// arm's per-layer term composed as [`FREE_BAND`] is, undercounts layers
    /// 34-37, whose V moves 0.2-0.3 under either perturbation). So the band
    /// is a ratio against a same-class perturbation measured in the same
    /// process, the teacher-forced arm's method: the flash arithmetics'
    /// distance starts later (the scores are first rounded in layer 0's
    /// attention) and smaller per layer. The GEMM path's attention is the
    /// prefill flash in either arm (`flash_gqa_prefill`: the tensor-core
    /// scores, f16 weights), so from layer 0's attention on its difference
    /// also carries that arithmetic's against the arm's decode flash. A
    /// correct GEMM path's K/V rows read below 1.9 on these prompts under
    /// either flash pass, highest on layers 34-37, and its logits ratio below
    /// 2.8 (highest in the scalar arm, where the prefill flash's f16
    /// arithmetic stands against the scalar decode flash); a wiring fault — a table built from another layer's or token's
    /// ids, weights one slot off, positions off by one — reads an error of
    /// order one against a spread of 1e-3 to 2.5e-1, a K/V ratio above 3.7
    /// on every layer it reaches. The logits ratio alone does not separate
    /// every fault (positions off by one leave it near 1); the K/V rows do.
    const GEMM_SPREAD_RATIO: f64 = 3.0;

    /// PIN(2026-09-25): the captured step's node count, derived before the
    /// chain was built: the embedding row, 12 nodes per layer (attention
    /// norm+quant, q·k·v, QK-norm+rope+append, flash segment pass, flash
    /// merge, q8_1 of the attention rows, attn_output with the residual, the
    /// FFN norm with the router gemv and the routing, gate·up·SwiGLU, q8_1 of
    /// the SwiGLU rows, down, combine into the next layer's input), one more
    /// on each of the 24 layers whose value projection is Q6_K (its own
    /// gemv), and the head's three (norm, q8_1, the Q6_K gemv with the argmax
    /// folded in): 1 + 24·12 + 24·13 + 3. Was 508: the two q8_1 launches
    /// were folded into attn_output and gate·up, and decode ran slower —
    /// every attn_output block re-quantized the attention row, every gate·up
    /// block fenced and drew a ticket (rig-log `#qwen3fuse-regression-nsys`).
    const NODES_CHAIN: usize = 604;

    /// Memcpy nodes in the captured step: the residual crosses no layer
    /// boundary as a copy.
    const MEMCPY_CHAIN: usize = 0;

    /// PIN(2026-09-25): the captured prefill pass's node count at one token,
    /// derived from the pass's launches before the graphs were built: the
    /// decode step's chain without its head, 1 + 24·12 + 24·13. Was 505,
    /// with the two q8_1 launches of [`NODES_CHAIN`]'s note folded.
    const NODES_PASS_1: usize = 601;

    /// PIN(2026-09-25): the captured prefill pass's node count at every `m`
    /// from two to `MAX_TOKENS`, derived the same way: the embedding rows,
    /// then per layer the attention half's 7 (norm+quant, q·k·v,
    /// QK-norm+rope+append, flash segment pass, flash merge, the attention
    /// rows' q8_1, attn_output with the residual) plus 2 on each of the 24
    /// layers whose value projection is Q6_K (its gemv, the token-major
    /// copy), and the FFN half's 6 (norm+quant, the m-token router,
    /// gate·up·SwiGLU, one q8_1 over every token's slots, one down `_sel`
    /// over every token's slots, the combine): 1 + 24·13 + 24·15. Every
    /// launch covers all of the pass's tokens, so the count does not grow
    /// with m. The head after the last pass runs eager.
    const NODES_PASS_M: usize = 673;

    /// PIN(2026-09-24): the teacher-forced bound on each half's error ratio
    /// — its relative error over the relative distance between the two
    /// sides' 8-bit inputs to that half (`quant_gap`: ours q8_1 per 128
    /// values against ik's q8_2 per 32, at the attention's q/k/v and
    /// attn_output inputs, the FFN's gate·up and down inputs, in
    /// quadrature). To first order a linear map carries its input's relative
    /// perturbation to its output unchanged, so a correct chain reads about
    /// 1; measured on this set: median about 1, worst 2.8 in attention (the
    /// softmax sharpens a score error) and 6.1 in the FFN (a layer whose
    /// input has one dominant channel, where the isotropic prediction
    /// undercounts). The kernel gates bound each kernel exactly; this band
    /// only has to separate that class from a wiring fault, which reads an
    /// error of order one over a gap of about 2e-2.
    const RATIO_BAND: f64 = 10.0;

    /// PIN(2026-09-24): the free-running bound on a layer output's relative
    /// distance. Derivation: the forced arm's per-layer errors (up to about
    /// 4e-2 of a layer's update, whose norm is of the residual's order)
    /// added in quadrature over 48 layers, independent: √48 · 4e-2 ≈ 0.28.
    const FREE_BAND: f64 = 0.28;

    /// PIN(2026-09-21): the V2-Lite gate's floor — a first difference where
    /// the reference's own top1-top2 margin is below this is a near-tie
    /// re-lottery, not a fault.
    const MARGIN_FLOOR: f32 = 0.5;

    fn rel(a: &[f32], b: &[f32], base: impl Fn(usize) -> f64) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += base(i).powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// ik's `name` rows, `k` values each.
    fn rows(man: &RefManifest, name: &str, k: usize) -> Result<Vec<Vec<f32>>, GateError> {
        let row = man.tensor(name, 0)?;
        let v = ref_tensor_logical_in(&man.dir, row)?;
        if row.ne[0] as usize != k || v.len() % k != 0 {
            return Err(format!("{name} is {:?}, want rows of {k}", row.ne).into());
        }
        Ok(v.chunks(k).map(<[f32]>::to_vec).collect())
    }

    fn open(ctx: usize, mode: StepMode) -> Result<Qwen3moeModel, GateError> {
        let file = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-e2e")?;
        let t = Instant::now();
        let mut m = Qwen3moeModel::load_full(file, ctx)?;
        m.set_mode(mode);
        let mma = m.body("gate_qwen3moe_e2e")?.flash_mma();
        println!(
            "load resident_bytes={} ctx={ctx} layers={} flash_mma={mma} ubatch={} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            m.stages()[0].layers().len(),
            m.ubatch()?,
            t.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    pub fn run() -> Result<(), GateError> {
        let args: Vec<String> = std::env::args().collect();
        if let Some(i) = args.iter().position(|a| a == "--ppl") {
            let tag = args.get(i + 1).ok_or("--ppl needs a tag")?;
            return ppl(tag);
        }
        let dump = match args.iter().position(|a| a == "--dump") {
            Some(i) => Some(PathBuf::from(
                args.get(i + 1).ok_or("--dump needs a directory")?,
            )),
            None => None,
        };
        if args.iter().any(|a| a == "--ubatch-only") {
            let ok = ubatch_sizes()?;
            println!("gate_qwen3moe_e2e --ubatch-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        let mut m = open(CTX, StepMode::Graph)?;
        if args.iter().any(|a| a == "--fault-only") {
            let ok = fault_layer(&mut m)?;
            println!("gate_qwen3moe_e2e --fault-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        if args.iter().any(|a| a == "--gemm-only") {
            let mut ok = gemm_prefill(&mut m)?;
            drop(m);
            ok &= ubatch_sizes()?;
            println!("gate_qwen3moe_e2e --gemm-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        let mut ok = true;
        ok &= structure(&mut m)?;
        let o = oracle::for_arch(Arch::Qwen3moe)?;
        let man = o.open(Set::Cpu)?;
        ok &= forced(&mut m, &man)?;
        ok &= free(&mut m, &man)?;
        ok &= greedy(&mut m, dump.as_deref())?;
        ok &= fault_layer(&mut m)?;
        drop(m);
        ok &= ubatch_sizes()?;
        println!("gate_qwen3moe_e2e: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }

    // ------------------------------------------------- (x) a fault's layer

    /// The layer clause (x) plants its fault in.
    const FAULT_LAYER: usize = 13;

    /// (x) (module doc).
    fn fault_layer(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        m.reset()?;
        let before = m.stages()[0].gpu().fault()?;
        let hidden = m.body("gate_qwen3moe_e2e")?.hparams().n_embd;
        let mut x = vec![0.25f32; hidden];
        x[5] = f32::NAN;
        m.step_ffn(FAULT_LAYER, &x)?;
        let raised = m.stages()[0].gpu().fault()?;
        let step = m.step(&[1]);
        let poisoned = m.poisoned();
        m.reset()?;
        let after = m.stages()[0].gpu().fault()?;
        let clean_step = m.step(&[1]).is_ok();
        m.reset()?;
        let layer = u32::try_from(FAULT_LAYER)?;
        let named = match &step {
            Err(GpuError::Fault { fault, .. }) => {
                Some(*fault) == raised && fault.layer == layer && fault.site().is_some()
            }
            _ => false,
        };
        let pass = before.is_none() && named && poisoned == raised && after.is_none() && clean_step;
        println!(
            "fault layer={FAULT_LAYER} ffn input with a NaN: word before {before:?}, the next step {} (want a \
             fault at layer {FAULT_LAYER}), poisoned {poisoned:?}, after reset {after:?} and a clean step \
             {clean_step} {}",
            match &step {
                Ok(t) => format!("returned token {t}"),
                Err(e) => format!("returned \"{e}\""),
            },
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------- (s) structure

    fn structure(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        let nodes = m.capture_step()?;
        let ([kernel, memcpy, memset, host], other) = count_kinds(
            &m.step_graph_nodes()?,
            [
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST,
            ],
        );
        let mut pass = nodes == NODES_CHAIN && memcpy == MEMCPY_CHAIN && host == 0;
        println!(
            "structure graph_nodes={nodes} (want {NODES_CHAIN}) kernel={kernel} memcpy={memcpy} \
             (want {MEMCPY_CHAIN}) memset={memset} host={host} (want 0) other={other} {}",
            verdict(pass)
        );
        let t = Instant::now();
        let counts = m.capture_prefill()?;
        println!(
            "structure prefill: {} passes captured in {:.1} ms (runtime value)",
            counts.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
        for (i, &nodes) in counts.iter().enumerate() {
            let rows = i + 1;
            let want = if rows == 1 {
                NODES_PASS_1
            } else {
                NODES_PASS_M
            };
            let ([kernel], other) = count_kinds(
                &m.prefill_graph_nodes(rows)?,
                [sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL],
            );
            let ok = nodes == want && kernel == nodes;
            println!(
                "structure prefill m={rows} graph_nodes={nodes} (want {want}) kernel={kernel} \
                 other={other} (want 0) {}",
                verdict(ok)
            );
            pass &= ok;
        }
        Ok(pass)
    }

    // ------------------------------------------------------ (f) forced

    /// ik's rows of `name` at every layer, and the offset of its first row
    /// in the prefill's positions (the last layer keeps only the output
    /// token's rows past its attention).
    fn per_layer(
        man: &RefManifest,
        stem: &str,
        n_layer: usize,
        k: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>, GateError> {
        (0..n_layer)
            .map(|l| rows(man, &format!("{stem}-{l}"), k))
            .collect()
    }

    /// The row of `v` (a layer's rows) for prefill position `t` of `t_n`, if
    /// the layer kept it.
    fn at(v: &[Vec<f32>], t: usize, t_n: usize) -> Option<&[f32]> {
        t.checked_sub(t_n - v.len()).map(|i| v[i].as_slice())
    }

    /// `‖x̂_ours − x̂_ik‖ / ‖x‖` over the `k`-value rows of `x`: how far apart
    /// the two sides' 8-bit activations of the same input sit (ours q8_1 per
    /// 128 values, ik q8_2 per 32).
    fn quant_gap(x: &[f32], k: usize) -> f64 {
        let o = q8_1_dequant(x, k, x.len() / k);
        let i = ik_q8_2::reconstruct(x);
        rel(&o, &i, |j| f64::from(x[j]))
    }

    fn forced(m: &mut Qwen3moeModel, man: &RefManifest) -> Result<bool, GateError> {
        let hp = m.body("forced")?.hparams().clone();
        let (h, n_layer, n_exp) = (hp.n_embd, hp.n_layer, hp.experts.n_expert as u32);
        let embd = rows(man, "inp_embd", h)?;
        let t_n = embd.len();
        let ik_out = per_layer(man, "l_out", n_layer, h)?;
        let ik_attn = per_layer(man, "attn_out", n_layer, h)?;
        let ik_routed = per_layer(man, "routed_out", n_layer, h)?;
        let ik_anorm = per_layer(man, "attn_norm", n_layer, h)?;
        let ik_fnorm = per_layer(man, "ffn_inp_normed", n_layer, h)?;
        let q_len = hp.n_head * hp.head_dim;
        let ik_fa: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|l| rows(man, &format!("fa-{l} (reshaped)"), q_len))
            .collect::<Result<_, _>>()?;
        let ff = hp.experts.ff;
        let ik_par: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|l| {
                let slots = rows(man, &format!("ffn_moe_gate_par-{l}"), ff)?;
                Ok(slots
                    .chunks(hp.experts.n_used)
                    .map(<[Vec<f32>]>::concat)
                    .collect())
            })
            .collect::<Result<_, GateError>>()?;
        // The routed sum's terms: each slot's down output times its weight,
        // summed in magnitude — the scale the FFN's error is measured on,
        // since the eight terms can cancel in `routed_out`.
        let n_used = hp.experts.n_used;
        let ik_mag: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|l| {
                let down = rows(man, &format!("ffn_moe_down-{l}"), h)?;
                let w = ref_tensor_logical_in(
                    &man.dir,
                    man.tensor(&format!("ffn_moe_weights_norm-{l}"), 0)?,
                )?;
                Ok(down
                    .chunks(n_used)
                    .zip(w.chunks(n_used))
                    .map(|(d, w)| {
                        (0..h)
                            .map(|i| (0..n_used).map(|s| (w[s] * d[s][i]).abs()).sum::<f32>())
                            .collect()
                    })
                    .collect())
            })
            .collect::<Result<_, GateError>>()?;
        let mut ik_ids = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let trow = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
            let ids = topk_ids_logical_within(man, trow, n_exp)?;
            let n_used = hp.experts.n_used;
            ik_ids.push(ids.chunks(n_used).map(<[i32]>::to_vec).collect::<Vec<_>>());
        }
        m.reset()?;
        let mut ok = true;
        let (mut w_up, mut w_attn, mut w_ffn) = (0.0f64, 0.0f64, 0.0f64);
        let (mut w_attn_r, mut w_ffn_r) = (0.0f64, 0.0f64);
        let (mut flips, mut sites) = (0usize, 0usize);
        for l in 0..n_layer {
            let (mut l_up, mut l_attn, mut l_ffn, mut l_flip) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            let (mut l_attn_r, mut l_attn_p, mut l_ffn_r, mut l_ffn_p) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for t in 0..t_n {
                let x_in: &[f32] = if l == 0 {
                    &embd[t]
                } else {
                    at(&ik_out[l - 1], t, t_n).ok_or("a layer input row is missing")?
                };
                let run = m.step_layer(l, x_in, u32::try_from(t)?)?;
                if let Some(a) = at(&ik_attn[l], t, t_n) {
                    let got: Vec<f32> =
                        run.ffn_inp.iter().zip(x_in).map(|(&f, &x)| f - x).collect();
                    let e = rel(&got, a, |i| f64::from(a[i]));
                    let g_in = at(&ik_anorm[l], t, t_n).map_or(0.0, |x| quant_gap(x, h));
                    let g_fa = at(&ik_fa[l], t, t_n).map_or(0.0, |x| quant_gap(x, q_len));
                    let pred = g_in.hypot(g_fa);
                    l_attn = l_attn.max(e);
                    l_attn_r = l_attn_r.max(e / pred.max(f64::MIN_POSITIVE));
                    l_attn_p = l_attn_p.max(pred);
                }
                let (Some(want), Some(a), Some(r)) = (
                    at(&ik_out[l], t, t_n),
                    at(&ik_attn[l], t, t_n),
                    at(&ik_routed[l], t, t_n),
                ) else {
                    continue;
                };
                l_up = l_up.max(rel(&run.l_out, want, |i| {
                    f64::from(want[i]) - f64::from(x_in[i])
                }));
                // The FFN half alone, on ik's own FFN input.
                let ik_inp: Vec<f32> = a.iter().zip(x_in).map(|(&a, &x)| a + x).collect();
                let f = m.step_ffn(l, &ik_inp)?;
                let got: Vec<f32> = f.l_out.iter().zip(&ik_inp).map(|(&o, &i)| o - i).collect();
                let theirs = &ik_ids[l][ik_ids[l].len() - (t_n - t)];
                let flipped = f
                    .ids
                    .iter()
                    .filter(|&&e| !theirs.contains(&(e as i32)))
                    .count();
                sites += 1;
                let mag = at(&ik_mag[l], t, t_n).ok_or("a routed magnitude row is missing")?;
                let e = rel(&got, r, |i| f64::from(mag[i]));
                let g_in = at(&ik_fnorm[l], t, t_n).map_or(0.0, |x| quant_gap(x, h));
                let g_par = at(&ik_par[l], t, t_n).map_or(0.0, |x| quant_gap(x, ff));
                let pred = g_in.hypot(g_par);
                l_ffn_p = l_ffn_p.max(pred);
                if flipped == 0 {
                    l_ffn_r = l_ffn_r.max(e / pred.max(f64::MIN_POSITIVE));
                }
                if flipped > 0 {
                    flips += 1;
                    l_flip += 1;
                    println!(
                        "forced layer={l} pos={t}: {flipped} router id(s) differ from ik's, ffn_rel={e:.3e} (printed)"
                    );
                } else {
                    l_ffn = l_ffn.max(e);
                }
            }
            println!(
                "forced layer={l} attn_rel={l_attn:.3e} (quant gap {l_attn_p:.3e}, worst ratio {l_attn_r:.2}) \
                 ffn_rel={l_ffn:.3e} (quant gap {l_ffn_p:.3e}, worst ratio {l_ffn_r:.2}) \
                 update_rel={l_up:.3e} flip_sites={l_flip}"
            );
            w_up = w_up.max(l_up);
            w_attn = w_attn.max(l_attn);
            w_ffn = w_ffn.max(l_ffn);
            w_attn_r = w_attn_r.max(l_attn_r);
            w_ffn_r = w_ffn_r.max(l_ffn_r);
            if l_attn_r > RATIO_BAND || l_ffn_r > RATIO_BAND {
                ok = false;
                println!("forced layer={l}: an error ratio passes {RATIO_BAND} FAIL");
            }
        }
        println!(
            "forced: {n_layer} layers x {t_n} positions; attention half worst rel {w_attn:.3e}, ratio \
             {w_attn_r:.2}; FFN half on ik's input worst rel {w_ffn:.3e}, ratio {w_ffn_r:.2} over {} \
             sites without a routing difference (band ratio {RATIO_BAND}); {flips} sites route an \
             expert ik does not (printed); whole-layer update worst {w_up:.3e} (printed) {}",
            sites - flips,
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------- (c) free-running

    fn free(m: &mut Qwen3moeModel, man: &RefManifest) -> Result<bool, GateError> {
        let hp = m.body("free")?.hparams().clone();
        let (h, n_layer) = (hp.n_embd, hp.n_layer);
        let toks: Vec<u32> = ref_ints(man, "inp_tokens", 0, RowKind::Input, Layout::Flat)?
            .iter()
            .map(|&i| u32::try_from(i))
            .collect::<Result<_, _>>()?;
        let mut ik_out = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            ik_out.push(rows(man, &format!("l_out-{l}"), h)?);
        }
        let ik_logits = rows(man, "result_output", hp.n_vocab)?;
        m.set_layer_taps(true)?;
        m.reset()?;
        let t_n = toks.len();
        let mut per_layer = vec![0.0f64; n_layer];
        let mut last = 0u32;
        for (t, &tok) in toks.iter().enumerate() {
            last = m.step(&[tok])?;
            let taps = m.layer_taps()?;
            for l in 0..n_layer {
                let off = t_n - ik_out[l].len();
                let Some(t_ik) = t.checked_sub(off) else {
                    continue;
                };
                let want = &ik_out[l][t_ik];
                per_layer[l] = per_layer[l].max(rel(&taps[l], want, |i| f64::from(want[i])));
            }
        }
        let logits = m.logits()?;
        m.set_layer_taps(false)?;
        let want = ik_logits.last().ok_or("result_output has no row")?;
        let ik_top = argmax(want);
        let lrel = rel(&logits, want, |i| f64::from(want[i]));
        let worst = per_layer.iter().copied().fold(0.0f64, f64::max);
        let ok = worst <= FREE_BAND && last == ik_top;
        for (l, &e) in per_layer.iter().enumerate() {
            if l % 8 == 0 || l == n_layer - 1 || e > FREE_BAND {
                println!("free layer={l} l_out_rel={e:.3e}");
            }
        }
        println!(
            "free: {t_n} tokens {toks:?}, worst l_out_rel={worst:.3e} (band {FREE_BAND:.0e}); last \
             position argmax ours={last} ik={ik_top} logits_rel={lrel:.3e} (printed) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    fn argmax(v: &[f32]) -> u32 {
        let mut best = 0usize;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best as u32
    }

    // ------------------------------------------------ (g) greedy, (r) replay

    fn greedy_dir() -> PathBuf {
        data_dir().join("qwen3moe").join("greedy")
    }

    /// Our greedy continuation of `ids`: `GEN` tokens, the first the
    /// prompt's own next token, and the last step's logits.
    fn continue_greedy(
        m: &mut Qwen3moeModel,
        ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        m.reset()?;
        let mut out = Vec::with_capacity(GEN);
        let mut next = m.step(ids)?;
        out.push(next);
        for _ in 1..GEN {
            next = m.step(&[next])?;
            out.push(next);
        }
        Ok((out, m.logits()?))
    }

    /// One prompt's greedy tokens and last logits as raw little-endian files
    /// under `dir` (`--dump`).
    fn dump(
        dir: &Path,
        tag: &str,
        p: usize,
        toks: &[u32],
        logits: &[f32],
    ) -> Result<(), GateError> {
        std::fs::create_dir_all(dir)?;
        let t: Vec<u8> = toks.iter().flat_map(|v| v.to_le_bytes()).collect();
        let l: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("p{p}-{tag}.tokens")), t)?;
        std::fs::write(dir.join(format!("p{p}-{tag}.logits")), l)?;
        Ok(())
    }

    fn greedy(m: &mut Qwen3moeModel, dump_dir: Option<&Path>) -> Result<bool, GateError> {
        let dir = greedy_dir();
        let mut prompts = Vec::new();
        let mut reference = Vec::new();
        for p in 0..PROMPTS {
            let pr = read_prompts(&dir.join(format!("prompt{p}.tsv")))?;
            let gr = read_greedy(&dir.join(format!("greedy-ik-cpu-{GEN}-p{p}.tsv")))?;
            let (Some(pr), Some(gr)) = (pr.into_iter().next(), gr.into_iter().next()) else {
                return Err(format!("{}: prompt {p} has no row", dir.display()).into());
            };
            if gr.n_tokens != pr.tokens.len() {
                return Err(format!(
                    "prompt {p}: {} ids, the greedy file's row says {}",
                    pr.tokens.len(),
                    gr.n_tokens
                )
                .into());
            }
            prompts.push(pr);
            reference.push(gr);
        }
        let mut graph = Vec::with_capacity(PROMPTS);
        let mut graph_logits = Vec::with_capacity(PROMPTS);
        m.set_mode(StepMode::Graph);
        for (i, p) in prompts.iter().enumerate() {
            let (toks, logits) = continue_greedy(m, &p.tokens)?;
            if let Some(dir) = dump_dir {
                dump(dir, "graph", i, &toks, &logits)?;
            }
            graph.push(toks);
            graph_logits.push(logits);
        }
        m.set_mode(StepMode::Eager);
        let mut replay_ok = true;
        for (i, p) in prompts.iter().enumerate() {
            let (toks, logits) = continue_greedy(m, &p.tokens)?;
            if let Some(dir) = dump_dir {
                dump(dir, "eager", i, &toks, &logits)?;
            }
            let same_t = toks == graph[i];
            let same_l = logits
                .iter()
                .zip(&graph_logits[i])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            if !(same_t && same_l) {
                replay_ok = false;
                println!(
                    "replay prompt {}: tokens_equal={same_t} logits_bit_equal={same_l} FAIL",
                    p.id
                );
            }
        }
        m.set_mode(StepMode::Graph);
        println!(
            "replay: {PROMPTS} prompts x {GEN} tokens, graph = eager tokens and last logits bit for bit {}",
            verdict(replay_ok)
        );
        let rep = compare_greedy(&graph, &reference, MARGIN_FLOOR);
        for (r, p) in rep.rows.iter().zip(&prompts) {
            println!(
                "greedy prompt {} ({} ids): {:?} first_diff={:?} ik_margin={:?} ours={}/{} tokens",
                r.id,
                p.tokens.len(),
                r.class,
                r.first_diff,
                r.ref_margin,
                r.n_ours,
                r.n_ref
            );
        }
        let greedy_ok = rep.rows.iter().all(|r| r.class != GreedyClass::Diverged);
        println!(
            "greedy: identical={} near_tie={} diverged={} (want 0, margin floor {MARGIN_FLOOR}) {}",
            rep.n_identical,
            rep.n_near_tie,
            rep.n_diverged,
            verdict(greedy_ok)
        );
        let prefill_ok = prefilled(m, &prompts, &graph, &graph_logits, dump_dir)?;
        let passes_ok = prefill_replay(m, &prompts)?;
        let gemm_ok = gemm_prefill(m)?;
        Ok(replay_ok && greedy_ok && prefill_ok && passes_ok && gemm_ok)
    }

    // ------------------------------------------------------------ (p) prefill

    /// Our greedy continuation of `ids` prefilled by `path` — whole, or in
    /// two calls cut at `cut` — then stepped: `GEN` tokens and the last
    /// step's logits.
    fn continue_prefilled(
        m: &mut Qwen3moeModel,
        ids: &[u32],
        cut: Option<usize>,
        path: PrefillPath,
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        m.reset()?;
        let mut out = Vec::with_capacity(GEN);
        let mut next = match cut {
            Some(c) => {
                m.prefill_with(&ids[..c], path)?;
                m.prefill_with(&ids[c..], path)?
            }
            None => m.prefill_with(ids, path)?,
        };
        out.push(next);
        for _ in 1..GEN {
            next = m.step(&[next])?;
            out.push(next);
        }
        Ok((out, m.logits()?))
    }

    fn prefilled(
        m: &mut Qwen3moeModel,
        prompts: &[PromptRow],
        graph: &[Vec<u32>],
        graph_logits: &[Vec<f32>],
        dump_dir: Option<&Path>,
    ) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        let mut ok = true;
        let mut passes = 0usize;
        let same = |toks: &[u32], logits: &[f32], i: usize| {
            toks == graph[i]
                && logits.len() == graph_logits[i].len()
                && logits
                    .iter()
                    .zip(&graph_logits[i])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        };
        for (i, p) in prompts.iter().enumerate() {
            let plan = m.prefill_plan(p.tokens.len(), PrefillPath::Auto)?;
            if plan.kind() != "prefill" {
                return Err(format!(
                    "prompt {} ({} ids) plans {plan}; this clause is the pass path's",
                    p.id,
                    p.tokens.len()
                )
                .into());
            }
            let (toks, logits) = continue_prefilled(m, &p.tokens, None, PrefillPath::Auto)?;
            if let Some(dir) = dump_dir {
                dump(dir, "prefill", i, &toks, &logits)?;
            }
            passes += Qwen3moeModel::prefill_passes(p.tokens.len());
            if !same(&toks, &logits, i) {
                ok = false;
                println!(
                    "prefill prompt {} ({} ids): tokens or last logits differ from the one-token path FAIL",
                    p.id,
                    p.tokens.len()
                );
            }
        }
        // Every prompt fits one pass, so the multi-pass walk runs on their
        // concatenation, against its own one-token run.
        let long: Vec<u32> = prompts
            .iter()
            .flat_map(|p| p.tokens.iter().copied())
            .collect();
        let (step_toks, step_logits) = continue_greedy(m, &long)?;
        let same_long = |(toks, logits): &(Vec<u32>, Vec<f32>)| {
            *toks == step_toks
                && logits.len() == step_logits.len()
                && logits
                    .iter()
                    .zip(&step_logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        };
        let whole_ok = same_long(&continue_prefilled(m, &long, None, PrefillPath::Pass)?);
        let split_ok = same_long(&continue_prefilled(
            m,
            &long,
            Some(SPLIT),
            PrefillPath::Pass,
        )?);
        println!(
            "prefill: {PROMPTS} prompts in {passes} passes of up to {MAX_TOKENS} positions, then \
             {GEN}-token greedy steps: tokens and last logits = the one-token path's bit for bit \
             {}; their {}-id concatenation in {} passes: {}; prefilled as {SPLIT} + {} ids: {}",
            verdict(ok),
            long.len(),
            Qwen3moeModel::prefill_passes(long.len()),
            verdict(whole_ok),
            long.len() - SPLIT,
            verdict(split_ok)
        );
        Ok(ok && whole_ok && split_ok)
    }

    // ------------------------------------------- (q) prefill replay = eager

    /// What one prefill leaves: every layer's K/V rows over its positions,
    /// the last logits and the token.
    struct PrefillRun {
        kv: Vec<Vec<u16>>,
        logits: Vec<f32>,
        token: u32,
    }

    /// One prefill of `ids` in `mode`, from a reset over cache rows seeded
    /// with the pattern.
    fn prefill_run(
        m: &mut Qwen3moeModel,
        ids: &[u32],
        mode: StepMode,
    ) -> Result<PrefillRun, GateError> {
        m.set_mode(mode);
        m.seed_depth(ids.len())?;
        m.reset()?;
        let token = m.prefill_with(ids, PrefillPath::Pass)?;
        Ok(PrefillRun {
            kv: m.kv_rows(ids.len())?,
            logits: m.logits()?,
            token,
        })
    }

    /// Rows of `a` (per layer, [`HEAD`]-value rows) that differ from `b`'s.
    fn rows_differing(a: &[Vec<u16>], b: &[Vec<u16>]) -> usize {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                x.chunks(HEAD)
                    .zip(y.chunks(HEAD))
                    .filter(|(r, s)| r != s)
                    .count()
            })
            .sum()
    }

    /// Rows of `a` (per layer, [`HEAD`]-value rows) equal to `b`'s.
    fn rows_same(a: &[Vec<u16>], b: &[Vec<u16>]) -> usize {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                x.chunks(HEAD)
                    .zip(y.chunks(HEAD))
                    .filter(|(r, s)| r == s)
                    .count()
            })
            .sum()
    }

    fn prefill_replay(m: &mut Qwen3moeModel, prompts: &[PromptRow]) -> Result<bool, GateError> {
        let long: Vec<u32> = prompts
            .iter()
            .flat_map(|p| p.tokens.iter().copied())
            .collect();
        if long.len() < 2 * MAX_TOKENS {
            return Err(format!(
                "the prompts' concatenation has {} ids; the replay clause takes {}",
                long.len(),
                2 * MAX_TOKENS
            )
            .into());
        }
        let mut ok = true;
        for rows in 1..=MAX_TOKENS {
            let ids = &long[..MAX_TOKENS + rows];
            m.seed_depth(ids.len())?;
            let seeded = m.kv_rows(ids.len())?;
            let g = prefill_run(m, ids, StepMode::Graph)?;
            let e = prefill_run(m, ids, StepMode::Eager)?;
            let kv_diff = rows_differing(&g.kv, &e.kv);
            let stale = rows_same(&g.kv, &seeded);
            let logits_same = g.logits.len() == e.logits.len()
                && g.logits
                    .iter()
                    .zip(&e.logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            let pass = kv_diff == 0 && stale == 0 && logits_same && g.token == e.token;
            println!(
                "prefill replay m={rows}: {} ids in passes of {MAX_TOKENS} + {rows}: K/V rows \
                 differing graph vs eager {kv_diff} (want 0), still the seeded pattern {stale} \
                 (want 0), last logits bit-equal {logits_same}, token graph={} eager={} {}",
                ids.len(),
                g.token,
                e.token,
                verdict(pass)
            );
            ok &= pass;
        }
        m.set_mode(StepMode::Graph);
        println!(
            "prefill replay: every pass size's replay = its eager twin (K/V rows and last logits bit \
             for bit) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // --------------------------------------------- (u) GEMM prefill band

    /// The first `n` ids of the prose file, after its digest is checked.
    fn prose(n: usize) -> Result<Vec<u32>, GateError> {
        let path = data_dir().join("qwen3moe").join(PROSE);
        let out = std::process::Command::new("sha256sum")
            .arg(&path)
            .output()?;
        let digest = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() || digest.split_whitespace().next() != Some(PROSE_SHA256) {
            return Err(format!(
                "{}: sha256 {:?}, want {PROSE_SHA256}",
                path.display(),
                digest.trim()
            )
            .into());
        }
        let ids: Vec<u32> = std::fs::read_to_string(&path)?
            .lines()
            .take(n)
            .map(|l| l.trim().parse::<u32>())
            .collect::<Result<_, _>>()?;
        if ids.len() < n {
            return Err(format!("{} holds {} ids, want {n}", path.display(), ids.len()).into());
        }
        Ok(ids)
    }

    /// Top-1 minus top-2 of `v`.
    fn margin(v: &[f32]) -> f32 {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &x in v {
            if x > a {
                b = a;
                a = x;
            } else if x > b {
                b = x;
            }
        }
        a - b
    }

    /// What one prefill of a long prompt leaves: every layer's K/V rows, the
    /// last logits, and the greedy continuation after it with each
    /// generated position's top1-top2 margin.
    struct LongRun {
        kv: Vec<Vec<u16>>,
        logits: Vec<f32>,
        tokens: Vec<u32>,
        margins: Vec<f32>,
    }

    /// `ids` from a reset, fed one step per token (`None`) or prefilled by
    /// the path (`Some`; cut into two calls at `cut`), then `GEN − 1`
    /// greedy steps.
    fn long_run(
        m: &mut Qwen3moeModel,
        ids: &[u32],
        path: Option<PrefillPath>,
        cut: Option<usize>,
    ) -> Result<LongRun, GateError> {
        m.reset()?;
        let first = match (path, cut) {
            (None, _) => m.step(ids)?,
            (Some(p), Some(c)) => {
                m.prefill_with(&ids[..c], p)?;
                m.prefill_with(&ids[c..], p)?
            }
            (Some(p), None) => m.prefill_with(ids, p)?,
        };
        let kv = m.kv_rows(ids.len())?;
        let logits = m.logits()?;
        let mut tokens = vec![first];
        let mut margins = vec![margin(&logits)];
        for _ in 1..GEN {
            let t = m.step(&[*tokens.last().ok_or("no token")?])?;
            tokens.push(t);
            margins.push(margin(&m.logits()?));
        }
        Ok(LongRun {
            kv,
            logits,
            tokens,
            margins,
        })
    }

    /// An f16's bits as a monotone integer, so two values' distance in ulp
    /// is the difference (both zeros at 0).
    fn f16_ord(b: u16) -> i32 {
        let mag = i32::from(b & 0x7fff);
        if b & 0x8000 != 0 { -mag } else { mag }
    }

    /// `‖a − b‖ / ‖b‖` over f16 bits.
    fn rel16(a: &[u16], b: &[u16]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (
                f64::from(gguf::quant::half_to_f32(x)),
                f64::from(gguf::quant::half_to_f32(y)),
            );
            num += (x - y).powi(2);
            den += y.powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    fn gemm_prefill(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        let longest = LONG.iter().copied().max().ok_or("no long prompt")?;
        let prose = prose(longest)?;
        let mut ok = true;
        let mut whole_1300 = None;
        for &n in &LONG {
            let ids = &prose[..n];
            let plan = m.prefill_plan(n, PrefillPath::Auto)?;
            let one = long_run(m, ids, None, None)?;
            let t0 = Instant::now();
            let gemm = long_run(m, ids, Some(PrefillPath::Auto), None)?;
            let wall = t0.elapsed().as_secs_f64();
            // The ruler: the same prompt on the other flash pass, eager.
            let mma = m.body("gemm_prefill")?.flash_mma();
            m.set_mode(StepMode::Eager);
            m.set_flash_mma(!mma)?;
            let alt = long_run(m, ids, None, None)?;
            m.set_flash_mma(mma)?;
            m.set_mode(StepMode::Graph);
            // Layer 0's moves in ulp, printed.
            let (mut l0_max, mut l0_moved) = (0u32, 0usize);
            for (&x, &y) in gemm.kv[0].iter().zip(&one.kv[0]) {
                let d = (f16_ord(x) - f16_ord(y)).unsigned_abs();
                l0_max = l0_max.max(d);
                l0_moved += usize::from(d != 0);
            }
            println!(
                "gemm n={n} plan={plan} layer=0 kv_values={} moved={l0_moved} max_ulp={l0_max} \
                 (printed)",
                one.kv[0].len()
            );
            let mut kv_ok = true;
            // (layer, distance over its band) of the layer closest to its band.
            let mut worst = (0usize, 0.0f64);
            for (l, ((g, o), a)) in gemm.kv.iter().zip(&one.kv).zip(&alt.kv).enumerate() {
                let half = o.len() / 2;
                let (rk, rv) = (rel16(&g[..half], &o[..half]), rel16(&g[half..], &o[half..]));
                let (sk, sv) = (rel16(&a[..half], &o[..half]), rel16(&a[half..], &o[half..]));
                let (d, band) = if l == 0 {
                    (rk.max(rv), GEMM_L0_REL)
                } else {
                    (
                        rk.max(rv) / sk.max(sv).max(f64::MIN_POSITIVE),
                        GEMM_SPREAD_RATIO,
                    )
                };
                let pass = d <= band;
                if d / band > worst.1 {
                    worst = (l, d / band);
                }
                if l % 8 == 0 || l == gemm.kv.len() - 1 || (32..40).contains(&l) || !pass {
                    let judged = if l == 0 { "rel" } else { "ratio" };
                    println!(
                        "gemm n={n} layer={l} k_rel={rk:.3e} v_rel={rv:.3e} spread k={sk:.3e} \
                         v={sv:.3e} {judged}={d:.3e} band={band:.3e} {}",
                        verdict(pass)
                    );
                }
                kv_ok &= pass;
            }
            let lrel = rel(&gemm.logits, &one.logits, |i| f64::from(one.logits[i]));
            let lspread = rel(&alt.logits, &one.logits, |i| f64::from(one.logits[i]));
            let lratio = lrel / lspread.max(f64::MIN_POSITIVE);
            let logits_ok = lratio <= GEMM_SPREAD_RATIO;
            let reference = GreedyRow {
                id: n,
                n_tokens: n,
                argmax: one.tokens[0],
                top5: Vec::new(),
                gen_ids: one.tokens.clone(),
                gen_margins: one.margins.clone(),
            };
            let rep = compare_greedy(
                std::slice::from_ref(&gemm.tokens),
                &[reference],
                MARGIN_FLOOR,
            );
            let r = &rep.rows[0];
            let greedy_ok = r.class != GreedyClass::Diverged;
            println!(
                "gemm n={n}: K/V rows within their bands {} (closest: layer {} at {:.2} of its \
                 band); last logits rel {lrel:.3e} against the flash arithmetics' own \
                 {lspread:.3e}, ratio {lratio:.3} (band {GEMM_SPREAD_RATIO}) {}; greedy {:?} \
                 first_diff={:?} one_token_margin={:?} {}; the GEMM prefill and its {GEN} tokens \
                 {wall:.2} s (runtime value)",
                verdict(kv_ok),
                worst.0,
                worst.1,
                verdict(logits_ok),
                r.class,
                r.first_diff,
                r.ref_margin,
                verdict(greedy_ok)
            );
            ok &= kv_ok && logits_ok && greedy_ok;
            if n == LONG[1] {
                whole_1300 = Some(gemm);
            }
        }
        let whole = whole_1300.ok_or("the 1,300-id run is missing")?;
        let n = LONG[1];
        let ids = &prose[..n];
        let cut_plans = (
            m.prefill_plan(LONG_SPLIT, PrefillPath::Auto)?,
            m.prefill_plan(n - LONG_SPLIT, PrefillPath::Auto)?,
        );
        let split = long_run(m, ids, Some(PrefillPath::Auto), Some(LONG_SPLIT))?;
        let kv_diff = rows_differing(&split.kv, &whole.kv);
        let logits_same = bits_equal(&split.logits, &whole.logits);
        let pass = kv_diff == 0 && logits_same && split.tokens == whole.tokens;
        println!(
            "gemm split n={n} as {LONG_SPLIT} ({}) + {} ({}): K/V rows differing from the whole \
             prefill {kv_diff} (want 0), last logits bit-equal {logits_same}, continuation equal {} {}",
            cut_plans.0,
            n - LONG_SPLIT,
            cut_plans.1,
            split.tokens == whole.tokens,
            verdict(pass)
        );
        ok &= pass;
        println!(
            "gemm prefill: {LONG:?} prose ids against the one-token path, K/V in band (layer 0 \
             {GEMM_L0_REL:e}, later layers and the logits {GEMM_SPREAD_RATIO} x the flash \
             arithmetics' spread), greedy not diverged, split = whole {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------ (w) ubatch-size invariance

    /// Cache rows of the model (w) opens: its longest prompt, with room
    /// past it (`seed_depth` keeps a row free for a step).
    const CTX_W: usize = W_LONG + 64;

    /// (w)'s prompt: a whole number of the largest ubatch.
    const W_LONG: usize = UBATCH;

    /// The ubatch sizes (w) compares: the size before the load-time lever,
    /// an odd one that cuts the prompt with a ragged last ubatch, the largest.
    const W_SIZES: [usize; 3] = [512, 1000, UBATCH];

    /// Where (w)'s split prefill at the largest size cuts the prompt: off
    /// every boundary of [`W_SIZES`].
    const W_SPLIT: usize = 2500;

    /// `ids` prefilled by the default path from a reset over cache rows
    /// seeded with the pattern (in two calls cut at `cut`), at the model's
    /// ubatch size: the K/V rows, the last logits, the token.
    fn ub_run(
        m: &mut Qwen3moeModel,
        ids: &[u32],
        cut: Option<usize>,
    ) -> Result<PrefillRun, GateError> {
        m.seed_depth(ids.len())?;
        m.reset()?;
        let token = match cut {
            Some(c) => {
                m.prefill_with(&ids[..c], PrefillPath::Auto)?;
                m.prefill_with(&ids[c..], PrefillPath::Auto)?
            }
            None => m.prefill_with(ids, PrefillPath::Auto)?,
        };
        Ok(PrefillRun {
            kv: m.kv_rows(ids.len())?,
            logits: m.logits()?,
            token,
        })
    }

    /// Whether `run` left what `want` did, bit for bit; one line under `label`.
    fn same_run(label: &str, run: &PrefillRun, want: &PrefillRun) -> bool {
        let kv_diff = rows_differing(&run.kv, &want.kv);
        let logits_same = bits_equal(&run.logits, &want.logits);
        let pass = kv_diff == 0 && logits_same && run.token == want.token;
        println!(
            "ubatch {label}: K/V rows differing from the {UBATCH}-token ubatch {kv_diff} (want 0), \
             last logits bit-equal {logits_same}, token {} (want {}) {}",
            run.token,
            want.token,
            verdict(pass)
        );
        pass
    }

    /// (w) (module doc), on its own model: the caller drops any other first,
    /// so the two never share the card.
    fn ubatch_sizes() -> Result<bool, GateError> {
        let mut m = open(CTX_W, StepMode::Graph)?;
        let prose = prose(W_LONG + 1)?;
        let (ids, ids1) = (&prose[..W_LONG], &prose[..=W_LONG]);
        let mut ok = true;
        m.set_ubatch(UBATCH)?;
        let plan = |m: &Qwen3moeModel, n: usize| m.prefill_plan(n, PrefillPath::Auto);
        // The references, each checked to have written every row it covers
        // (none still the seeded pattern), so an equal run is not two runs
        // that wrote nothing.
        let mut refs = Vec::with_capacity(2);
        for x in [ids, ids1] {
            m.seed_depth(x.len())?;
            let seeded = m.kv_rows(x.len())?;
            let run = ub_run(&mut m, x, None)?;
            let stale = rows_same(&run.kv, &seeded);
            let pass = stale == 0;
            println!(
                "ubatch size={UBATCH} n={} plan={} resident_bytes={} (a reference): K/V rows still \
                 the seeded pattern {stale} (want 0) {}",
                x.len(),
                plan(&m, x.len())?,
                m.resident_bytes(),
                verdict(pass)
            );
            ok &= pass;
            refs.push(run);
        }
        let (whole, whole1) = (&refs[0], &refs[1]);
        let split = ub_run(&mut m, ids, Some(W_SPLIT))?;
        ok &= same_run(
            &format!(
                "size={UBATCH} n={W_LONG} split as {W_SPLIT} ({}) + {} ({})",
                plan(&m, W_SPLIT)?,
                W_LONG - W_SPLIT,
                plan(&m, W_LONG - W_SPLIT)?
            ),
            &split,
            whole,
        );
        drop(split);
        for &u in &W_SIZES[..W_SIZES.len() - 1] {
            m.set_ubatch(u)?;
            let run = ub_run(&mut m, ids, None)?;
            ok &= same_run(
                &format!(
                    "size={u} n={W_LONG} plan={} resident_bytes={}",
                    plan(&m, W_LONG)?,
                    m.resident_bytes()
                ),
                &run,
                whole,
            );
            drop(run);
            if u == W_SIZES[0] {
                let run = ub_run(&mut m, ids1, None)?;
                ok &= same_run(
                    &format!("size={u} n={} plan={}", W_LONG + 1, plan(&m, W_LONG + 1)?),
                    &run,
                    whole1,
                );
            }
        }
        // Sizes past the range: refused, the size kept.
        let kept = m.ubatch()?;
        let refused = [0, UBATCH + 1].iter().all(|&u| m.set_ubatch(u).is_err());
        let still = m.ubatch()? == kept;
        let pass = refused && still;
        println!(
            "ubatch refusals: sizes 0 and {} refused {refused}, size kept at {kept} {still} {}",
            UBATCH + 1,
            verdict(pass)
        );
        ok &= pass;
        println!(
            "ubatch: the prompt's K/V rows, last logits and token at sizes {W_SIZES:?} and split = \
             whole at {UBATCH}, bit for bit {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------------------- ppl

    /// The chain as the PPL scorer's model.
    struct Scored<'a>(&'a mut Qwen3moeModel);

    impl PplModel for Scored<'_> {
        fn reset(&mut self) -> Result<(), GateError> {
            Ok(self.0.reset()?)
        }

        fn step(&mut self, token: u32) -> Result<(), GateError> {
            self.0.step(&[token])?;
            Ok(())
        }

        fn logits(&mut self) -> Result<Vec<f32>, GateError> {
            Ok(self.0.logits()?)
        }
    }

    fn ppl(tag: &str) -> Result<(), GateError> {
        let path: PathBuf = data_dir().join("ikppl").join(format!("{tag}.kld"));
        let base = KldBase::open_own_vocab(&path)?;
        println!(
            "ppl base {}: ctx {} chunks {} scored per chunk {} from {}",
            path.display(),
            base.n_ctx(),
            base.n_chunk(),
            base.scored_per_chunk(),
            base.first_scored()
        );
        let mut m = open(base.n_ctx(), StepMode::Graph)?;
        let n_vocab = m.body("ppl")?.hparams().n_vocab;
        if base.n_vocab() != n_vocab {
            return Err(format!(
                "{} holds {} vocabulary entries per record, the model {n_vocab}",
                path.display(),
                base.n_vocab()
            )
            .into());
        }
        let s = score_ppl(&base, &mut Scored(&mut m))?;
        let nf = s.n as f64;
        println!(
            "ppl: {} positions; PPL ours {:.4} ik {:.4}; d = NLL_ours − NLL_ik mean {:+.5} ± \
             {:.5} (SE), sd(d) {:.4}; Δ_PPL {:+.3} %; KLD(ik‖ours) {:.5} ± {:.5}; same top \
             {:.2} %; σ_rel (rms of margin ours − ik) {:.4}; {:.0} s (printed, not judged)",
            s.n,
            s.ppl_ours,
            s.ppl_ik,
            s.mean_d,
            s.se_d,
            s.sd_d,
            100.0 * s.dppl(),
            s.kld,
            s.kld_se,
            100.0 * s.same_top as f64 / nf,
            s.sigma_rel,
            s.secs
        );
        Ok(())
    }
}
