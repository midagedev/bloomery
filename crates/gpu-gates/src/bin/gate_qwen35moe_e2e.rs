//! The Qwen3.6-35B-A3B end-to-end gate: the whole chain — 30 gated-delta-rule
//! layers, 10 gated GQA layers at head 256, 40 MoE blocks with the shared
//! expert as a ninth slot, the head and the argmax — on one card, against
//! ik's CPU oracle sets (`refset::arch::qwen35moe`: the 5-token batch set,
//! the step after a 4-token prefill, the step after a 1,024-token prefill),
//! all read in one process over one load.
//!
//! What is asserted:
//! - (s) structure: the captured decode step holds [`NODES_DECODE`] nodes,
//!   all kernels; a captured pass of `m` rows holds [`NODES_PASS`] `+ 4m`
//!   (`m` memcpys of a row into its head, the rest kernels), at `m` = 5 and
//!   8; each count also equals the body's own launch count
//!   (`Body35::pass_launches`) plus its heads. The layer stores' bytes equal
//!   their derivation from the header ([`store_bytes`]).
//! - (p) one layer body: the batch set's five tokens as five decode steps
//!   in graph mode, as one pass of five rows (`step_rows`) and as five eager
//!   steps, each from zero stores written by the gate, then as five graph
//!   steps from `reset` alone after the eager run's state, leave the same
//!   per-position logits and tokens and every layer's store (K/V planes,
//!   recurrent state, conv ring) bit for bit. No reduction differs between
//!   the decode step and the pass: every launch of a pass computes each row
//!   the way its one-row launch does (the FFN norm inside the router at one
//!   row is `norm_quant` + the router bit for bit, gate_qwen35moe_moe's norm
//!   clause; the flash and rope rows are their one-row launches',
//!   gate_qwen35moe_attn; the q4_K projections run the one-column body;
//!   `q6k_gemv` keeps one accumulator chain per column whatever the column
//!   count). Every other clause starts from the gate's zero stores, so a
//!   `reset` that keeps state reddens the last line of this clause alone.
//! - (c) free-running on the batch set: the five decode steps' every layer
//!   output against ik's `l_out-L` within [`FREE_BAND`], and the last
//!   position's argmax equal to ik's `result_output` argmax.
//! - (f) teacher-forced, per layer, on the batch set as one pass of five
//!   rows from a reset: each layer run alone on ik's own input rows
//!   (`inp_embd` for layer 0, `l_out-(L−1)` after), its mixer's taps against
//!   ik's, and its FFN half run alone on ik's own FFN input against ik's MoE
//!   taps; every tap's relative error over the relative distance between
//!   the two sides' 8-bit activations of the input that reaches it
//!   ([`quant_gap`]) within [`RATIO_BAND`]; the routed ids equal as sets
//!   except where ik's own margin between its eighth pick and the best
//!   other expert lies inside our logits' measured error (counted,
//!   printed); the decay's underflow sites (exp(g) = 0) equal.
//! - (t) the decode step after each step set's prefill: the prefill's state
//!   loaded from the set's inputs (`cache_s_lL` — conv then ssm —,
//!   `cache_k_lL`, `cache_v_lL`) into every layer's store, the model stood at
//!   the set's position, then (t1) the step free-running: every layer
//!   output within [`FREE_BAND`], the argmax equal to ik's, every delta
//!   layer's new state within [`RATIO_BAND`] of the input gap; and (t2) each
//!   layer teacher-forced at one row on its reloaded store, its taps as (f).
//!
//! Tap maps (ik → ours). Delta layer: `qkv_mixed` → the q·k·v projection;
//! `z` → the gate projection; `beta_in`, `alpha` → β's and α's raw
//! projections; `g_in` → `ln` of our decay (`exp(g)`); the v channels of
//! `conv_output_silu` → the conv's v channels; `q_fused`, `k_fused` (the
//! L2 norm, a PERMUTE's src0 where the set names the permute) → the conv's
//! normed q (over `Q_SCALE`) and k; `attn_output` → the delta output;
//! `new_state` (`[k][v]` per head) → our state (`[v][k]`), transposed;
//! `new_conv_states_cont` (`[C][3]`) → the ring's last three positions;
//! `attn_out_norm` → the gated norm's output; `linear_attn_out` → the
//! residual update. Attention layer: `Qaux` → the `[q | gate]` rows;
//! `Qcur_normed` and `Qcur_roped` → our queries past and inside the 64
//! turned values; `Kcur_*` likewise; `Vcur`; `fa` → the flash output;
//! `qkv_gated` → our flash output times the sigmoid of our gate (a host
//! product: the kernel folds it into the quantizer); `attn_out` → the
//! residual update. MoE: `ffn_moe_logits` → the router's first 256 logits;
//! `ffn_moe_topk` → the first eight slots' ids; `ffn_moe_weights_norm` →
//! their weights, matched by id; `shared_expert_gate_sigmoid` → the ninth
//! slot's weight; `l_out` → the layer output.
//!
//! Known differences, named, not banded away: ik clamps nothing on these
//! sets (q35oracle Q1); layer 37's decay underflows to 0 at one site in 960,
//! and ours must underflow at the same sites; ik combines `(routed + resid)
//! + shexp_gated`, ours `((Σ8 w·d) + w8·d_sh) + resid` — one or two f32
//! roundings of the update, some 1e-7 of it, five orders under the band of
//! `l_out`, which is the only tap past that add.
//!
//! The chain runs the tensor-core flash pass, the engine's.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen35moe_e2e: built without the `gpu` feature; see `just gate-gpu-qwen35moe-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35moe_e2e", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::{
        Delta35Run, Gqa35Run, LayerKind35, Mixer35Run, Qwen35moeModel, StoreHost,
    };
    use bloomery_gpu::linear::{Q_SCALE, RING_ROWS, expf_ik};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::route_core::sigmoid;
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::{
        GateError, Layout, RefManifest, RowKind, bits_equal, checks_failed, data_dir, ik_q8_2,
        q8_1_dequant, ref_ints, ref_tensor_logical_in, split_f32, topk_ids_logical_within, verdict,
        widened_f16_rows_in,
    };
    use cuda_core::sys;
    use gguf::Split;
    use refset::arch::qwen35moe::{BATCH, D1K, IK, MODEL, STEP4};
    use std::time::Instant;

    /// Cache rows: the 1,024-token set's step at position 1,024, with room.
    const CTX: usize = 1088;

    /// The file's shape, as the header states it (q35design §1): what the
    /// derivations below are written against.
    const HIDDEN: usize = 2048;
    const N_LAYER: usize = 40;
    const N_ATTN: usize = 10;
    const N_DELTA: usize = 30;
    /// Delta layers whose `attn_qkv` is Q6_K, and attention layers whose
    /// `attn_v` is (3, 7, 19, 31, 35, 39).
    const Q6_QKV: usize = 14;
    const Q6_V: usize = 6;
    const N_EXPERT: usize = 256;
    const N_USED: usize = 8;
    const SLOTS: usize = N_USED + 1;
    const C: usize = 8192;
    const N_V: usize = 32;
    const HEAD_V: usize = 128;
    const Q_ROWS: usize = 8192;
    const KV_ROW: usize = 512;
    const N_HEAD: usize = 16;
    const N_KV: usize = 2;
    const HEAD: usize = 256;
    const ROT: usize = 64;
    const EPS: f64 = 1e-6;

    /// PIN(2026-09-27): the captured decode step's node count, derived before
    /// the chain was built: the embedding row; each delta layer's 8 mixer
    /// launches (norm+quant, the two projection launches — a Q6_K `attn_qkv`
    /// in its own gemv, `attn_gate`·β·α in one — conv, delta, gated norm,
    /// q8_1, `ssm_out` with the residual) and 5 FFN launches (the norm with
    /// the gated router, gate·up over the joined stacks, q8_1 of the nine
    /// slots, the down `_sel`, the combine); each attention layer's 7 mixer
    /// launches (norm+quant, q·k·v, rope-256, flash segment pass and merge,
    /// the gated q8_1, the output projection with the residual), one more
    /// on the 6 layers whose `attn_v` is Q6_K, and 5 FFN; then the head's
    /// three: 1 + 30·13 + 10·12 + 6 + 3.
    const NODES_DECODE: usize = 520;

    /// PIN(2026-09-27): a captured pass of `m >= 2` rows without its heads,
    /// derived the same way at more than one row: each delta layer 8 + 6
    /// (the router's norm its own launch), one more on the 14 whose
    /// `attn_qkv` is Q6_K (its token-major copy); each attention layer
    /// 7 + 6, two more on the 6 Q6_K `attn_v` layers (gemv and copy):
    /// 1 + 30·14 + 14 + 10·13 + 12. Each row's head adds a copy of the
    /// row's residual and three launches.
    const NODES_PASS: usize = 577;

    const _: () = assert!(
        NODES_DECODE == 1 + N_DELTA * 13 + N_ATTN * 12 + Q6_V + 3
            && NODES_PASS == 1 + N_DELTA * 14 + Q6_QKV + N_ATTN * 13 + 2 * Q6_V
            && N_DELTA + N_ATTN == N_LAYER
    );

    /// PIN(2026-09-27): the teacher-forced bound on a tap's error ratio — its
    /// relative error over the relative distance between the two sides'
    /// 8-bit activations of the input that reaches it (ours q8_1 per 128
    /// values, ik's q8_2 per 32: [`quant_gap`]; two inputs in quadrature).
    /// Derivation, the qwen3moe gate's: to first order a linear map carries
    /// its input's relative perturbation unchanged, so a projection reads
    /// about 1; the kernels are held to their host rules bit for bit or
    /// within γ(n) of them (gate_linear, gate_qwen35moe_attn,
    /// gate_qwen35moe_moe), terms some 1e-6 of the ~1e-2 gap, so the gap is
    /// the whole prediction. Predicted per class: the projections, β/α, z,
    /// the conv's v channels and the normed q/k about 1 (median), at most 3;
    /// the delta output and the state, a sum over the call's tokens of
    /// per-token errors the decay and β (both at most 1) do not amplify,
    /// at most √5 ≈ 2.2 times a token's, so at most 5; the flash output at
    /// most 3 (the softmax sharpens a score error, qwen3moe measured 2.8);
    /// the FFN at most 6 (qwen3moe measured 6.1 on a layer with one dominant
    /// channel). A wiring fault — another layer's weight, a head or a token
    /// off by one, the state transposed — reads an error of order one over a
    /// gap of about 2e-2, a ratio above 40.
    const RATIO_BAND: f64 = 10.0;

    /// PIN(2026-09-27): the free-running bound on a layer output's relative
    /// distance from ik's. Derivation, the qwen3moe gate's: the forced
    /// arm's per-layer errors, up to about 4e-2 of a layer's update, whose
    /// norm is of the residual's order, added in quadrature over 40 layers,
    /// independent: √40 · 4e-2 ≈ 0.25. The step sets' one step starts from
    /// ik's own state, so the same composition bounds it.
    const FREE_BAND: f64 = 0.26;

    /// The layer stores' bytes at [`CTX`] rows, derived from the header:
    /// each attention layer's K and V planes, `2 · n_kv · ctx · 256` f16;
    /// each delta layer's state, `32 · 128 · 128` f32 (one lane), and conv
    /// ring, `RING_ROWS · 8192` f32.
    fn store_bytes() -> usize {
        N_ATTN * 2 * N_KV * CTX * HEAD * 2 + N_DELTA * (N_V * HEAD_V * HEAD_V + RING_ROWS * C) * 4
    }

    /// `‖a − b‖ / ‖base‖` over the values, in f64.
    fn rel(a: &[f32], b: &[f32], base: impl Fn(usize) -> f64) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += base(i).powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// `rel` of our last `ik.len()` values against ik's, on ik's own norm:
    /// a tap ik keeps only the output token's rows of (the last layer past
    /// its attention) meets our last rows. Infinite when ik holds more
    /// values than ours, and a NaN reads infinite, so neither passes a band.
    fn rel_to(ours: &[f32], ik: &[f32]) -> f64 {
        let Some(tail) = ours.len().checked_sub(ik.len()).map(|s| &ours[s..]) else {
            return f64::INFINITY;
        };
        worse(0.0, rel(tail, ik, |i| f64::from(ik[i])))
    }

    /// The larger of two errors, a NaN counting as infinite: `f64::max`
    /// would drop it.
    fn worse(a: f64, b: f64) -> f64 {
        if a.is_nan() || b.is_nan() {
            f64::INFINITY
        } else {
            a.max(b)
        }
    }

    /// `‖x̂_ours − x̂_ik‖ / ‖x‖` over the `k`-value rows of `x`: how far apart
    /// the two sides' 8-bit activations of the same input sit (ours q8_1 per
    /// 128 values, ik q8_2 per 32).
    fn quant_gap(x: &[f32], k: usize) -> f64 {
        let o = q8_1_dequant(x, k, x.len() / k);
        let i = ik_q8_2::reconstruct(x);
        rel(&o, &i, |j| f64::from(x[j]))
    }

    /// The RMS-normed rows of `x` (`k` values each) times `gain`, in f64
    /// then rounded: the input a norm+quant launch quantizes, near enough
    /// for its gap.
    fn normed(x: &[f32], gain: &[f32], k: usize) -> Vec<f32> {
        x.chunks(k)
            .flat_map(|r| {
                let ms = r.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / k as f64;
                let s = 1.0 / (ms + EPS).sqrt();
                r.iter()
                    .zip(gain)
                    .map(move |(&v, &g)| (f64::from(v) * s * f64::from(g)) as f32)
            })
            .collect()
    }

    /// The first index of the largest value, as the head's argmax breaks ties.
    fn argmax(v: &[f32]) -> u32 {
        let best = v
            .iter()
            .enumerate()
            .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b });
        best as u32
    }

    /// A set's tap `name` in its logical order.
    fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
    }

    /// The node a set dumps as `name`, or when that row only relabels
    /// another (a permute, reshape, view or copy), the node it relabels,
    /// through its `src0`, with that row's `ne`: the step sets name the
    /// permute of the L2 norm's output and leave the output itself unnamed,
    /// the batch set names the output.
    fn tap_or_src(man: &RefManifest, name: &str) -> Result<(Vec<f32>, [u64; 4]), GateError> {
        const RELABEL: [&str; 5] = ["PERMUTE", "RESHAPE", "VIEW", "CONT", "TRANSPOSE"];
        let (mut at, mut row) = man.tensor_at(name, 0)?;
        for _ in 0..4 {
            if !RELABEL.contains(&row.op.as_str()) {
                return Ok((ref_tensor_logical_in(&man.dir, row)?, row.ne));
            }
            (at, row) = man.last_before(at, row.src0.as_deref())?;
        }
        Err(format!("{name}: four relabelling rows and no node under them").into())
    }

    /// The last `n` rows of `k` values of `v`, the rows a tap kept (the
    /// last layer keeps only the output token's rows past its attention):
    /// `None` when `v` holds fewer.
    fn last_rows(v: &[f32], k: usize, n: usize) -> Option<&[f32]> {
        v.len().checked_sub(n * k).map(|s| &v[s..])
    }

    /// Worst ratio and its tap, over a layer's taps.
    #[derive(Default)]
    struct Worst {
        ratio: f64,
        tap: String,
        lines: Vec<String>,
    }

    impl Worst {
        fn add(&mut self, tap: &str, e: f64, gap: f64) {
            let r = worse(0.0, e / gap.max(f64::MIN_POSITIVE));
            self.lines.push(format!("{tap} rel={e:.3e} ratio={r:.2}"));
            if self.tap.is_empty() || r > self.ratio {
                self.ratio = r;
                self.tap = tap.to_string();
            }
        }
    }

    fn open(ctx: usize) -> Result<Qwen35moeModel, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        if file.architecture() != Some("qwen35moe") {
            return Err(format!("{MODEL} is {:?}, not qwen35moe", file.architecture()).into());
        }
        let t = Instant::now();
        let m = Qwen35moeModel::open(Gpu::new()?, file, ctx, true)?;
        println!(
            "load resident_bytes={} ctx={ctx} layers={} flash_mma={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            m.body("gate_qwen35moe_e2e")?.flash_mma(),
            t.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    // ------------------------------------------------------ (s) structure

    fn structure(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        let body = m.body("structure")?;
        let kinds = body.kinds();
        let n_attn = kinds
            .iter()
            .filter(|k| **k == LayerKind35::Attention)
            .count();
        let (launch_1, launch_m) = (body.pass_launches(1), body.pass_launches(2));
        let (stores, want_stores) = (body.store_bytes(), store_bytes());
        let mut ok = n_attn == N_ATTN && kinds.len() == N_LAYER && stores == want_stores;
        let attn_at: Vec<usize> = (0..kinds.len())
            .filter(|&l| kinds[l] == LayerKind35::Attention)
            .collect();
        println!(
            "structure layers={} attention at {attn_at:?} ({n_attn}, want {N_ATTN}); store bytes \
             {stores} (want {want_stores}, derived) {}",
            kinds.len(),
            verdict(ok)
        );
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memcpy = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY;
        let nodes = m.capture_step()?;
        let ([k, c], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memcpy]);
        let pass = nodes == NODES_DECODE && k == nodes && launch_1 + 3 == nodes;
        println!(
            "structure decode graph_nodes={nodes} (want {NODES_DECODE}; the body counts {} + 3) \
             kernel={k} memcpy={c} other={other} {}",
            launch_1,
            verdict(pass)
        );
        ok &= pass;
        for (rows, nodes, list) in [
            (5usize, m.capture_rows::<5>()?, m.rows_graph_nodes::<5>()?),
            (8, m.capture_rows::<8>()?, m.rows_graph_nodes::<8>()?),
        ] {
            let ([k, c], other) = count_kinds(&list, [kernel, memcpy]);
            let want = NODES_PASS + 4 * rows;
            let pass =
                nodes == want && c == rows && k == nodes - rows && launch_m + 4 * rows == nodes;
            println!(
                "structure pass m={rows} graph_nodes={nodes} (want {want}; the body counts {} + 4·{rows}) \
                 kernel={k} memcpy={c} (want {rows}) other={other} {}",
                launch_m,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------------- (p) one layer body

    /// Every layer's store, read back.
    fn stores(m: &mut Qwen35moeModel) -> Result<Vec<StoreHost>, GateError> {
        let n = m.layers().len();
        let (gpu, _, body) = m.body_parts("stores")?;
        Ok((0..n)
            .map(|l| body.store(gpu, l))
            .collect::<Result<_, GpuError>>()?)
    }

    fn same_store(a: &StoreHost, b: &StoreHost) -> bool {
        match (a, b) {
            (StoreHost::Kv { k: ka, v: va }, StoreHost::Kv { k: kb, v: vb }) => {
                ka == kb && va == vb
            }
            (
                StoreHost::Rec {
                    state: sa,
                    ring: ra,
                },
                StoreHost::Rec {
                    state: sb,
                    ring: rb,
                },
            ) => bits_equal(sa, sb) && bits_equal(ra, rb),
            _ => false,
        }
    }

    /// What a run of the batch tokens left: each position's logits and
    /// token, and every layer's store.
    struct PathRun {
        logits: Vec<Vec<f32>>,
        tokens: Vec<u32>,
        stores: Vec<StoreHost>,
    }

    /// `reset`, then every layer's store written to zeros through
    /// `set_store`: a start from zero state that does not rest on `reset`'s
    /// own clearing, so only the clause that tests `reset` reads it.
    fn fresh(m: &mut Qwen35moeModel) -> Result<(), GateError> {
        m.reset()?;
        let n = m.layers().len();
        let (gpu, _, body) = m.body_parts("fresh")?;
        for l in 0..n {
            let zero = match body.store(gpu, l)? {
                StoreHost::Kv { k, v } => StoreHost::Kv {
                    k: vec![0; k.len()],
                    v: vec![0; v.len()],
                },
                StoreHost::Rec { state, ring } => StoreHost::Rec {
                    state: vec![0.0; state.len()],
                    ring: vec![0.0; ring.len()],
                },
            };
            body.set_store(gpu, l, &zero)?;
        }
        Ok(())
    }

    /// The batch tokens as decode steps from position 0: from [`fresh`]'s
    /// zero stores, or (`by_reset`) from `reset` alone.
    fn decode_run(
        m: &mut Qwen35moeModel,
        toks: &[u32],
        by_reset: bool,
    ) -> Result<PathRun, GateError> {
        if by_reset {
            m.reset()?;
        } else {
            fresh(m)?;
        }
        let (mut logits, mut tokens) = (Vec::new(), Vec::new());
        for &t in toks {
            tokens.push(m.step(&[t])?);
            logits.push(m.logits()?);
        }
        Ok(PathRun {
            logits,
            tokens,
            stores: stores(m)?,
        })
    }

    fn same_run(label: &str, got: &PathRun, want: &PathRun) -> bool {
        let logits = got.logits.len() == want.logits.len()
            && got
                .logits
                .iter()
                .zip(&want.logits)
                .all(|(a, b)| bits_equal(a, b));
        let stores_same = got.stores.len() == want.stores.len()
            && got
                .stores
                .iter()
                .zip(&want.stores)
                .all(|(a, b)| same_store(a, b));
        let differ: Vec<usize> = (0..got.stores.len().min(want.stores.len()))
            .filter(|&l| !same_store(&got.stores[l], &want.stores[l]))
            .collect();
        let ok = logits && stores_same && got.tokens == want.tokens;
        println!(
            "paths {label}: tokens {:?} vs {:?}, every position's logits bit-identical={logits}, \
             every store bit-identical={stores_same} (layers differing {differ:?}) {}",
            got.tokens,
            want.tokens,
            verdict(ok)
        );
        ok
    }

    fn paths(m: &mut Qwen35moeModel, toks: &[u32]) -> Result<(bool, PathRun), GateError> {
        let toks5: [u32; 5] = toks
            .try_into()
            .map_err(|_| format!("the batch set holds {} tokens, want 5", toks.len()))?;
        m.set_mode(StepMode::Graph);
        let decode = decode_run(m, toks, false)?;
        // One pass of five rows.
        fresh(m)?;
        let tokens = m.step_rows::<5>(toks5)?.to_vec();
        let pass = PathRun {
            logits: m.rows_logits::<5>()?.to_vec(),
            tokens,
            stores: stores(m)?,
        };
        let mut ok = same_run("pass m=5 vs five decode steps (graph)", &pass, &decode);
        m.set_mode(StepMode::Eager);
        let eager = decode_run(m, toks, false)?;
        ok &= same_run("five eager steps vs five graph replays", &eager, &decode);
        m.set_mode(StepMode::Graph);
        // The only run that starts from `reset` alone, after the eager run
        // left its state: `reset` must clear it.
        let again = decode_run(m, toks, true)?;
        ok &= same_run("after reset alone, the same five steps", &again, &decode);
        Ok((ok, decode))
    }

    // --------------------------------------------------- (c) free-running

    fn free(
        m: &mut Qwen35moeModel,
        man: &RefManifest,
        toks: &[u32],
        decode: &PathRun,
    ) -> Result<bool, GateError> {
        let n = m.layers().len();
        let ik_out: Vec<Vec<f32>> = (0..n)
            .map(|l| tap(man, &format!("l_out-{l}")))
            .collect::<Result<_, _>>()?;
        let ik_logits = tap(man, "result_output")?;
        let vocab = m.body("free")?.vocab();
        let ik_last = last_rows(&ik_logits, vocab, 1).ok_or("result_output holds no row")?;
        m.set_layer_taps(true)?;
        fresh(m)?;
        let t_n = toks.len();
        let mut per_layer = vec![0.0f64; n];
        for (t, &tok) in toks.iter().enumerate() {
            m.step(&[tok])?;
            let taps = m.layer_taps()?;
            for ((worst, ik), got) in per_layer.iter_mut().zip(&ik_out).zip(&taps) {
                let kept = ik.len() / HIDDEN;
                let Some(i) = (t + kept).checked_sub(t_n) else {
                    continue;
                };
                *worst = worse(*worst, rel_to(got, &ik[i * HIDDEN..(i + 1) * HIDDEN]));
            }
        }
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        let ours = decode.logits.last().ok_or("no decode logits")?;
        let (top, ik_top) = (argmax(ours), argmax(ik_last));
        let worst = per_layer.iter().copied().fold(0.0f64, worse);
        for (l, &e) in per_layer.iter().enumerate() {
            if l % 8 == 0 || l == n - 1 || e > FREE_BAND {
                println!("free layer={l} l_out_rel={e:.3e}");
            }
        }
        let ok = worst <= FREE_BAND && top == ik_top;
        println!(
            "free: {t_n} tokens {toks:?}, worst l_out_rel={worst:.3e} (band {FREE_BAND:.2}); last \
             position argmax ours={top} ik={ik_top} logits_rel={:.3e} (printed) {}",
            rel_to(ours, ik_last),
            verdict(ok)
        );
        Ok(ok)
    }

    // ---------------------------------------------- (f)/(t2) forced taps

    /// The file's gains a gap is computed through.
    struct Gains {
        attn_norm: Vec<Vec<f32>>,
        ffn_norm: Vec<Vec<f32>>,
    }

    impl Gains {
        fn attn(&self, l: usize) -> Result<&[f32], GateError> {
            Ok(self
                .attn_norm
                .get(l)
                .map(Vec::as_slice)
                .ok_or("a layer past the file's gains")?)
        }

        fn read(n: usize) -> Result<Gains, GateError> {
            let split = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
            let each = |stem: &str| -> Result<Vec<Vec<f32>>, GateError> {
                (0..n)
                    .map(|l| split_f32(&split, &format!("blk.{l}.{stem}"), HIDDEN))
                    .collect()
            };
            Ok(Gains {
                attn_norm: each("attn_norm.weight")?,
                ffn_norm: each("post_attention_norm.weight")?,
            })
        }
    }

    /// Our `[t][h][d]` values of `h_n` heads of `d_n` against ik's `ik`
    /// laid out by `ne`: `[t][h][d]` (`ne = [d, h, t]`) or `[h][t][d]` (`ne
    /// = [d, t, h]`, a permuted norm's output), each ik value times `scale`.
    fn heads_rel(
        ours: &[f32],
        ik: &[f32],
        ne: [u64; 4],
        (t_n, h_n, d_n): (usize, usize, usize),
        scale: f32,
    ) -> Result<f64, GateError> {
        let ne = [ne[0] as usize, ne[1] as usize, ne[2] as usize];
        let head_major = ne == [d_n, t_n, h_n] && t_n > 1;
        if !(head_major || ne == [d_n, h_n, t_n] || ik.len() == t_n * h_n * d_n) {
            return Err(format!("a head tap of ne {ne:?} for {t_n} x {h_n} x {d_n}").into());
        }
        let mut want = vec![0.0f32; t_n * h_n * d_n];
        for t in 0..t_n {
            for h in 0..h_n {
                for d in 0..d_n {
                    let src = if head_major {
                        (h * t_n + t) * d_n + d
                    } else {
                        (t * h_n + h) * d_n + d
                    };
                    want[(t * h_n + h) * d_n + d] = ik[src] * scale;
                }
            }
        }
        Ok(rel_to(ours, &want))
    }

    /// Values `lo..hi` of every `width`-value head of `v`.
    fn head_part(v: &[f32], width: usize, lo: usize, hi: usize) -> Vec<f32> {
        v.chunks(width).flat_map(|h| h[lo..hi].to_vec()).collect()
    }

    /// A delta layer's taps against ik's: the worst ratio; the decay's
    /// underflow sites must be ik's.
    #[allow(
        clippy::too_many_arguments,
        reason = "a layer's run, its inputs' gaps, its store and ik's set, as the tap map reads them"
    )]
    fn delta_taps(
        man: &RefManifest,
        l: usize,
        r: &Delta35Run,
        store: &StoreHost,
        (t_n, pos_end): (usize, usize),
        gap_in: f64,
        upd: (&[f32], &[f32]),
        w: &mut Worst,
    ) -> Result<bool, GateError> {
        let gap_y = quant_gap(&r.y, N_V * HEAD_V);
        w.add(
            "qkv_mixed",
            rel_to(&r.x, &tap(man, &format!("qkv_mixed-{l}"))?),
            gap_in,
        );
        w.add("z", rel_to(&r.z, &tap(man, &format!("z-{l}"))?), gap_in);
        w.add(
            "beta_in",
            rel_to(&r.b, &tap(man, &format!("beta_in-{l}"))?),
            gap_in,
        );
        w.add(
            "alpha",
            rel_to(&r.a, &tap(man, &format!("alpha-{l}"))?),
            gap_in,
        );
        // ik dumps g, not exp(g) (the delta op takes g): our g is the log of
        // our decay where it did not underflow, and the underflow sites are
        // those where ik's exp — `expf_ik`, the same algorithm — gives 0.
        let g = tap(man, &format!("g_in-{l}"))?;
        let ik_decay: Vec<f32> = g.iter().map(|&x| expf_ik(x)).collect();
        let under = |v: &[f32]| -> Vec<usize> { (0..v.len()).filter(|&i| v[i] == 0.0).collect() };
        let (ours_u, ik_u) = (under(&r.decay), under(&ik_decay));
        let live: Vec<usize> = (0..g.len().min(r.decay.len()))
            .filter(|&i| r.decay[i] > 0.0)
            .collect();
        let g_ours: Vec<f32> = live
            .iter()
            .map(|&i| f64::from(r.decay[i]).ln() as f32)
            .collect();
        let g_ik: Vec<f32> = live.iter().map(|&i| g[i]).collect();
        w.add(
            "g_in (the log of our decay)",
            rel_to(&g_ours, &g_ik),
            gap_in,
        );
        let silu = tap(man, &format!("conv_output_silu-{l}"))?;
        let v_of =
            |x: &[f32]| -> Vec<f32> { x.chunks(C).flat_map(|t| t[C / 2..].to_vec()).collect() };
        w.add(
            "conv_output_silu (v)",
            rel_to(&v_of(&r.conv), &v_of(&silu)),
            gap_in,
        );
        let (q, q_ne) = tap_or_src(man, &format!("q_fused-{l}"))?;
        let (k, k_ne) = tap_or_src(man, &format!("k_fused-{l}"))?;
        let q_ours: Vec<f32> = r.conv.chunks(C).flat_map(|t| t[..C / 4].to_vec()).collect();
        let k_ours: Vec<f32> = r
            .conv
            .chunks(C)
            .flat_map(|t| t[C / 4..C / 2].to_vec())
            .collect();
        w.add(
            "q_fused",
            heads_rel(&q_ours, &q, q_ne, (t_n, 16, HEAD_V), Q_SCALE)?,
            gap_in,
        );
        w.add(
            "k_fused",
            heads_rel(&k_ours, &k, k_ne, (t_n, 16, HEAD_V), 1.0)?,
            gap_in,
        );
        w.add(
            "attn_output",
            rel_to(&r.o, &tap(man, &format!("attn_output-{l}"))?),
            gap_in,
        );
        w.add(
            "attn_out_norm",
            rel_to(&r.y, &tap(man, &format!("attn_out_norm-{l}"))?),
            gap_in,
        );
        let (ours_upd, x_in) = upd;
        let ik_upd = tap(man, &format!("linear_attn_out-{l}"))?;
        let got: Vec<f32> = ours_upd.iter().zip(x_in).map(|(&f, &x)| f - x).collect();
        w.add(
            "linear_attn_out",
            rel_to(&got, &ik_upd),
            gap_in.hypot(gap_y),
        );
        let StoreHost::Rec { state, ring } = store else {
            return Err(format!("layer {l} is a delta layer with another store").into());
        };
        let ik_state = tap(man, &format!("new_state-{l}"))?;
        let mut t = vec![0.0f32; state.len()];
        for h in 0..N_V {
            for kk in 0..HEAD_V {
                for v in 0..HEAD_V {
                    t[(h * HEAD_V + v) * HEAD_V + kk] = ik_state[(h * HEAD_V + kk) * HEAD_V + v];
                }
            }
        }
        w.add("new_state (transposed)", rel_to(state, &t), gap_in);
        let ik_conv = tap(man, &format!("new_conv_states_cont-{l}"))?;
        let mut ours_conv = vec![0.0f32; 3 * C];
        for j in 0..3 {
            let p = pos_end - 3 + j;
            let slot = p % RING_ROWS;
            for ch in 0..C {
                ours_conv[ch * 3 + j] = ring[slot * C + ch];
            }
        }
        w.add("new_conv_states_cont", rel_to(&ours_conv, &ik_conv), gap_in);
        let ok = ours_u == ik_u;
        if !ours_u.is_empty() || !ik_u.is_empty() {
            println!(
                "forced layer={l} decay underflow sites ours {ours_u:?} ik {ik_u:?} {}",
                verdict(ok)
            );
        }
        Ok(ok)
    }

    /// An attention layer's taps against ik's.
    fn attn_taps(
        man: &RefManifest,
        l: usize,
        r: &Gqa35Run,
        t_n: usize,
        gap_in: f64,
        upd: (&[f32], &[f32]),
        w: &mut Worst,
    ) -> Result<(), GateError> {
        w.add(
            "Qaux",
            rel_to(&r.qg, &tap(man, &format!("Qaux-{l}"))?),
            gap_in,
        );
        let q_normed = tap(man, &format!("Qcur_normed-{l}"))?;
        let q_roped = tap(man, &format!("Qcur_roped-{l}"))?;
        let k_normed = tap(man, &format!("Kcur_normed-{l}"))?;
        let k_roped = tap(man, &format!("Kcur_roped-{l}"))?;
        for (name, ours, ik, lo, hi) in [
            ("Qcur_normed (past the turn)", &r.q, &q_normed, ROT, HEAD),
            ("Qcur_roped (the turn)", &r.q, &q_roped, 0, ROT),
            ("Kcur_normed (past the turn)", &r.k, &k_normed, ROT, HEAD),
            ("Kcur_roped (the turn)", &r.k, &k_roped, 0, ROT),
        ] {
            w.add(
                name,
                rel_to(&head_part(ours, HEAD, lo, hi), &head_part(ik, HEAD, lo, hi)),
                gap_in,
            );
        }
        w.add(
            "Vcur",
            rel_to(&r.v, &tap(man, &format!("Vcur-{l}"))?),
            gap_in,
        );
        w.add("fa", rel_to(&r.fa, &tap(man, &format!("fa-{l}"))?), gap_in);
        let gated: Vec<f32> = (0..t_n * N_HEAD * HEAD)
            .map(|i| {
                let (t, h, d) = (i / (N_HEAD * HEAD), (i / HEAD) % N_HEAD, i % HEAD);
                r.fa[i] * sigmoid(r.qg[t * Q_ROWS + h * 2 * HEAD + HEAD + d])
            })
            .collect();
        let gap_g = quant_gap(&gated, N_HEAD * HEAD);
        w.add(
            "qkv_gated (host product)",
            rel_to(&gated, &tap(man, &format!("qkv_gated-{l}"))?),
            gap_in,
        );
        let (ours_upd, x_in) = upd;
        let got: Vec<f32> = ours_upd.iter().zip(x_in).map(|(&f, &x)| f - x).collect();
        let ik_upd = tap(man, &format!("attn_out-{l}"))?;
        w.add("attn_out", rel_to(&got, &ik_upd), gap_in.hypot(gap_g));
        Ok(())
    }

    /// The FFN half on ik's own FFN input against ik's MoE taps, over the
    /// rows ik kept (`ffn_in` may hold more: the last layer keeps only the
    /// output token's rows past its attention): the router's 257 logits
    /// (`ffn_moe_logits` and `shared_expert_gate`), the ids as sets — a
    /// token whose sets differ is a flip, allowed only where ik's own margin
    /// between its eighth pick and the best expert it left lies inside
    /// twice our logits' error at that token, counted and printed — the
    /// weights matched by id with the shared expert's (ik's sigmoid tap, or
    /// the sigmoid of ik's logit where a one-token graph fused it away), and
    /// the layer output's update on the tokens without a flip.
    fn moe_taps(
        m: &mut Qwen35moeModel,
        man: &RefManifest,
        l: usize,
        ffn_in: &[f32],
        gains: &Gains,
        w: &mut Worst,
    ) -> Result<(bool, usize), GateError> {
        let logits = tap(man, &format!("ffn_moe_logits-{l}"))?;
        let gate = tap(man, &format!("shared_expert_gate-{l}"))?;
        let t_n = gate.len();
        if t_n == 0 || logits.len() != t_n * N_EXPERT {
            return Err(format!(
                "layer {l}: {} router logits for {t_n} gate logits",
                logits.len()
            )
            .into());
        }
        let ffn_in = last_rows(ffn_in, HIDDEN, t_n)
            .ok_or_else(|| format!("layer {l}: ik routes {t_n} rows, the input holds fewer"))?;
        let f = m.ffn_rows(l, ffn_in)?;
        let gap = quant_gap(&normed(ffn_in, &gains.ffn_norm[l], HIDDEN), HIDDEN);
        let ik_logits: Vec<f32> = (0..t_n)
            .flat_map(|t| {
                let mut row = logits[t * N_EXPERT..(t + 1) * N_EXPERT].to_vec();
                row.push(gate[t]);
                row
            })
            .collect();
        w.add(
            "ffn_moe_logits + shared_expert_gate",
            rel_to(&f.logits, &ik_logits),
            gap,
        );
        let trow = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
        let ids = topk_ids_logical_within(man, trow, N_EXPERT as u32)?;
        let wn = tap(man, &format!("ffn_moe_weights_norm-{l}"))?;
        if ids.len() != t_n * N_USED || wn.len() != t_n * N_USED {
            return Err(format!(
                "layer {l}: {} ids and {} weights for {t_n} tokens",
                ids.len(),
                wn.len()
            )
            .into());
        }
        let ik_sg: Vec<f32> = match man.tensor(&format!("shared_expert_gate_sigmoid-{l}"), 0) {
            Ok(row) => ref_tensor_logical_in(&man.dir, row)?,
            Err(_) => gate.iter().map(|&g| sigmoid(g)).collect(),
        };
        if ik_sg.len() != t_n {
            return Err(format!("layer {l}: {} shared gates for {t_n} tokens", ik_sg.len()).into());
        }
        let (mut ok, mut flipped) = (true, Vec::new());
        let (mut w_ours, mut w_ik) = (Vec::new(), Vec::new());
        let row = N_EXPERT + 1;
        for (t, &sg) in ik_sg.iter().enumerate() {
            let ours = &f.ids[t * SLOTS..t * SLOTS + N_USED];
            let theirs = &ids[t * N_USED..(t + 1) * N_USED];
            if !ours.iter().all(|&e| theirs.contains(&(e as i32))) {
                let lg = &ik_logits[t * row..t * row + N_EXPERT];
                let min_in = theirs
                    .iter()
                    .map(|&e| lg[e as usize])
                    .fold(f32::INFINITY, f32::min);
                let max_out = (0..N_EXPERT)
                    .filter(|&e| !theirs.contains(&(e as i32)))
                    .map(|e| lg[e])
                    .fold(f32::NEG_INFINITY, f32::max);
                let err = f.logits[t * row..t * row + N_EXPERT]
                    .iter()
                    .zip(lg)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let tie = min_in - max_out <= 2.0 * err;
                ok &= tie;
                flipped.push(t);
                println!(
                    "forced layer={l} token={t}: ids ours {ours:?} ik {theirs:?}, ik margin {:.3e}, \
                     our logits' error {err:.3e}: {}",
                    min_in - max_out,
                    if tie { "a near tie (counted)" } else { "FAIL" }
                );
                continue;
            }
            for (s, &e) in ours.iter().enumerate() {
                let j = theirs
                    .iter()
                    .position(|&x| x == e as i32)
                    .ok_or("an id the set test found")?;
                w_ours.push(f.weights[t * SLOTS + s]);
                w_ik.push(wn[t * N_USED + j]);
            }
            w_ours.push(f.weights[t * SLOTS + N_USED]);
            w_ik.push(sg);
        }
        if !w_ik.is_empty() {
            w.add(
                "ffn_moe_weights_norm + shared_expert_gate_sigmoid",
                rel_to(&w_ours, &w_ik),
                gap,
            );
        }
        let out = tap(man, &format!("l_out-{l}"))?;
        let ik_out = last_rows(&out, HIDDEN, t_n).ok_or("l_out rows")?;
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for t in (0..t_n).filter(|t| !flipped.contains(t)) {
            let r = t * HIDDEN..(t + 1) * HIDDEN;
            got.extend(
                f.l_out[r.clone()]
                    .iter()
                    .zip(&ffn_in[r.clone()])
                    .map(|(&o, &i)| o - i),
            );
            want.extend(
                ik_out[r.clone()]
                    .iter()
                    .zip(&ffn_in[r])
                    .map(|(&o, &i)| o - i),
            );
        }
        if !want.is_empty() {
            w.add("l_out (update)", rel_to(&got, &want), gap);
        }
        Ok((ok, flipped.len()))
    }

    /// Teacher-forced taps on every layer of `man` at `t_n` rows from
    /// position `pos`, each layer's store first set by `load` (a reset's
    /// zero store, or the set's prefill state).
    fn forced(
        m: &mut Qwen35moeModel,
        man: &RefManifest,
        label: &str,
        (pos, t_n): (u32, usize),
        gains: &Gains,
        load: &dyn Fn(&mut Qwen35moeModel, usize) -> Result<(), GateError>,
    ) -> Result<bool, GateError> {
        let n = m.layers().len();
        let kinds = m.body("forced")?.kinds();
        let embd = tap(man, "inp_embd")?;
        let mut ok = true;
        let (mut worst, mut flips_all) = (0.0f64, 0usize);
        for (l, &kind) in kinds.iter().enumerate() {
            let x_all = if l == 0 {
                embd.clone()
            } else {
                tap(man, &format!("l_out-{}", l - 1))?
            };
            let x_in = last_rows(&x_all, HIDDEN, t_n).ok_or("a layer input row is missing")?;
            load(m, l)?;
            let run = m.layer_rows(l, x_in, pos)?;
            let gap_in = quant_gap(&normed(x_in, gains.attn(l)?, HIDDEN), HIDDEN);
            let mut w = Worst::default();
            let body_ok = match (&run.mixer, kind) {
                (Mixer35Run::Delta(r), LayerKind35::Delta) => {
                    let store = {
                        let (gpu, _, body) = m.body_parts("forced")?;
                        body.store(gpu, l)?
                    };
                    delta_taps(
                        man,
                        l,
                        r,
                        &store,
                        (t_n, pos as usize + t_n),
                        gap_in,
                        (&run.ffn_inp, x_in),
                        &mut w,
                    )?
                }
                (Mixer35Run::Attention(r), LayerKind35::Attention) => {
                    attn_taps(man, l, r, t_n, gap_in, (&run.ffn_inp, x_in), &mut w)?;
                    true
                }
                _ => return Err(format!("layer {l}: a run of another kind than the layer").into()),
            };
            // The FFN half alone on ik's FFN input: the layer input plus
            // ik's mixer update, over the rows ik kept.
            let upd_name = match kind {
                LayerKind35::Delta => format!("linear_attn_out-{l}"),
                LayerKind35::Attention => format!("attn_out-{l}"),
            };
            let ik_upd = tap(man, &upd_name)?;
            let kept = ik_upd.len() / HIDDEN;
            let x_kept = last_rows(x_in, HIDDEN, kept).ok_or("rows")?;
            let ffn_in: Vec<f32> = ik_upd.iter().zip(x_kept).map(|(&a, &x)| a + x).collect();
            let (moe_ok, flips) = moe_taps(m, man, l, &ffn_in, gains, &mut w)?;
            let pass = body_ok && moe_ok && w.ratio <= RATIO_BAND;
            println!(
                "forced {label} layer={l} {:?}: worst ratio {:.2} at {} (band {RATIO_BAND}); \
                 flip sites {flips} {}",
                kind,
                w.ratio,
                w.tap,
                verdict(pass)
            );
            if !pass || l % 8 == 0 {
                for line in &w.lines {
                    println!("  forced {label} layer={l} {line}");
                }
            }
            worst = worse(worst, w.ratio);
            flips_all += flips;
            ok &= pass;
        }
        println!(
            "forced {label}: {n} layers x {t_n} rows from position {pos}; worst ratio {worst:.2} \
             (band {RATIO_BAND}); flip sites {flips_all} (near ties, counted) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------- (t) the step sets

    /// Layer `l`'s store as a step set's prefill left it: a delta layer's
    /// state (ik's `[k][v]` per head, transposed into ours) and conv ring
    /// (ik's `[C][3]` window of the three positions before `p`, into slots
    /// `(p − 3 + j) mod RING_ROWS`), or an attention layer's K and V rows
    /// `0..p` (ik's rows of 512, or its transposed V, by the row's `ne`).
    fn prefill_store(
        man: &RefManifest,
        kind: LayerKind35,
        l: usize,
        p: usize,
    ) -> Result<StoreHost, GateError> {
        match kind {
            LayerKind35::Delta => {
                let s = ref_tensor_logical_in(&man.dir, man.input(&format!("cache_s_l{l}"), 0)?)?;
                let conv_len = 3 * C;
                if s.len() != conv_len + N_V * HEAD_V * HEAD_V {
                    return Err(format!("cache_s_l{l} holds {} values", s.len()).into());
                }
                let (conv, ssm) = s.split_at(conv_len);
                let mut state = vec![0.0f32; ssm.len()];
                for h in 0..N_V {
                    for k in 0..HEAD_V {
                        for v in 0..HEAD_V {
                            state[(h * HEAD_V + v) * HEAD_V + k] =
                                ssm[(h * HEAD_V + k) * HEAD_V + v];
                        }
                    }
                }
                let mut ring = vec![0.0f32; RING_ROWS * C];
                for j in 0..3 {
                    let Some(pos) = (p + j).checked_sub(3) else {
                        continue;
                    };
                    let slot = pos % RING_ROWS;
                    for ch in 0..C {
                        ring[slot * C + ch] = conv[ch * 3 + j];
                    }
                }
                Ok(StoreHost::Rec { state, ring })
            }
            LayerKind35::Attention => {
                let plane = |name: String| -> Result<Vec<u16>, GateError> {
                    let row = man.input(&name, 0)?;
                    let bits = widened_f16_rows_in(&man.dir, row)?;
                    let (ne0, ne1) = (row.ne[0] as usize, row.ne[1] as usize);
                    let mut out = vec![0u16; N_KV * CTX * HEAD];
                    for r in 0..p {
                        for h in 0..N_KV {
                            for d in 0..HEAD {
                                let c = h * HEAD + d;
                                let src = if ne0 == KV_ROW {
                                    r * KV_ROW + c
                                } else if ne1 == KV_ROW {
                                    c * ne0 + r
                                } else {
                                    return Err(format!(
                                        "{name} is {:?}, not rows of {KV_ROW}",
                                        row.ne
                                    )
                                    .into());
                                };
                                out[(h * CTX + r) * HEAD + d] = bits[src];
                            }
                        }
                    }
                    Ok(out)
                };
                Ok(StoreHost::Kv {
                    k: plane(format!("cache_k_l{l}"))?,
                    v: plane(format!("cache_v_l{l}"))?,
                })
            }
        }
    }

    fn step_set(m: &mut Qwen35moeModel, name: &str, gains: &Gains) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(name), &IK)?;
        let (p, tail, before) = man.step()?;
        let token = *tail.first().ok_or("the step set holds no token")?;
        let pu = p as usize;
        let kinds = m.body("step_set")?.kinds();
        let loaded: Vec<StoreHost> = kinds
            .iter()
            .enumerate()
            .map(|(l, &k)| prefill_store(&man, k, l, pu))
            .collect::<Result<_, _>>()?;
        println!(
            "step {name}: position {p}, token {token} after {} tokens; every layer's store loaded \
             from the set's inputs",
            before.len()
        );
        let load_all = |m: &mut Qwen35moeModel| -> Result<(), GateError> {
            fresh(m)?;
            m.seed_depth(pu)?;
            let (gpu, _, body) = m.body_parts("step_set")?;
            for (l, s) in loaded.iter().enumerate() {
                body.set_store(gpu, l, s)?;
            }
            Ok(())
        };
        // (t1) the step, free-running from the loaded state.
        load_all(m)?;
        m.set_layer_taps(true)?;
        m.set_mode(StepMode::Eager);
        let top = m.step(&[token])?;
        let taps = m.layer_taps()?;
        let logits = m.logits()?;
        let after = stores(m)?;
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        let ik_logits = tap(&man, "result_output")?;
        let vocab = m.body("step_set")?.vocab();
        let ik_last = last_rows(&ik_logits, vocab, 1).ok_or("result_output")?;
        let ik_top = argmax(ik_last);
        let (mut worst, mut worst_state) = (0.0f64, 0.0f64);
        for (l, got) in taps.iter().enumerate() {
            let want = tap(&man, &format!("l_out-{l}"))?;
            let want = last_rows(&want, HIDDEN, 1).ok_or("l_out")?;
            let e = rel_to(got, want);
            worst = worse(worst, e);
            if let (LayerKind35::Delta, StoreHost::Rec { state, .. }) = (kinds[l], &after[l]) {
                let ik_state = tap(&man, &format!("new_state-{l}"))?;
                let mut t = vec![0.0f32; state.len()];
                for h in 0..N_V {
                    for k in 0..HEAD_V {
                        for v in 0..HEAD_V {
                            t[(h * HEAD_V + v) * HEAD_V + k] =
                                ik_state[(h * HEAD_V + k) * HEAD_V + v];
                        }
                    }
                }
                let x_in = if l == 0 {
                    tap(&man, "inp_embd")?
                } else {
                    tap(&man, &format!("l_out-{}", l - 1))?
                };
                let x_in = last_rows(&x_in, HIDDEN, 1).ok_or("input")?;
                let gap = quant_gap(&normed(x_in, &gains.attn_norm[l], HIDDEN), HIDDEN);
                worst_state = worse(worst_state, rel_to(state, &t) / gap.max(f64::MIN_POSITIVE));
            }
            if l % 8 == 0 || l == taps.len() - 1 || e > FREE_BAND {
                println!("step {name} free layer={l} l_out_rel={e:.3e}");
            }
        }
        let mut ok = worst <= FREE_BAND && top == ik_top && worst_state <= RATIO_BAND;
        println!(
            "step {name} free: worst l_out_rel={worst:.3e} (band {FREE_BAND:.2}); worst new_state \
             ratio {worst_state:.2} (band {RATIO_BAND}); argmax ours={top} ik={ik_top} \
             logits_rel={:.3e} (printed) {}",
            rel_to(&logits, ik_last),
            verdict(ok)
        );
        // (t2) each layer teacher-forced at one row on its reloaded store.
        let reload = |m: &mut Qwen35moeModel, l: usize| -> Result<(), GateError> {
            let (gpu, _, body) = m.body_parts("step_set")?;
            body.set_store(gpu, l, &loaded[l])?;
            Ok(())
        };
        load_all(m)?;
        m.set_mode(StepMode::Eager);
        ok &= forced(m, &man, name, (p, 1), gains, &reload)?;
        m.set_mode(StepMode::Graph);
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        bloomery_levers::at_main(&[])?;
        let mut m = open(CTX)?;
        let mut ok = structure(&mut m)?;
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let toks: Vec<u32> = ref_ints(&man, "inp_tokens", 0, RowKind::Input, Layout::Flat)?
            .iter()
            .map(|&i| u32::try_from(i))
            .collect::<Result<_, _>>()?;
        let (p_ok, decode) = paths(&mut m, &toks)?;
        ok &= p_ok;
        ok &= free(&mut m, &man, &toks, &decode)?;
        let gains = Gains::read(m.layers().len())?;
        m.set_mode(StepMode::Eager);
        fresh(&mut m)?;
        let zero = |_: &mut Qwen35moeModel, _: usize| -> Result<(), GateError> { Ok(()) };
        ok &= forced(&mut m, &man, BATCH, (0, toks.len()), &gains, &zero)?;
        m.set_mode(StepMode::Graph);
        for set in [STEP4, D1K] {
            ok &= step_set(&mut m, set, &gains)?;
        }
        println!("gate_qwen35moe_e2e: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
