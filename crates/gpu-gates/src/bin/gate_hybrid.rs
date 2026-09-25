//! The hybrid MoE boundary gate (`bloomery_gpu::hybrid`): DeepSeek-V2-Lite
//! with experts `[0, n_l)` of every routed layer on the card and the rest on
//! the host, against the all-card model, for `n_l` in {32, 0}.
//!
//! What is asserted, per `n_l`:
//! - structure: the captured step's node count and node kinds are the
//!   all-card step's plus the boundary's per routed layer — the handoff copy
//!   (memcpy), the go and the wait (two batch memops), the join (one kernel),
//!   and, with experts on the card, the zeroing of their outputs (memset) and
//!   the card's slot list (one kernel: a host slot written as `HOST`, which
//!   the `_sel` kernels skip without a fault); with none on the card the two
//!   `_sel` launches drop out.
//! - eager = replay: the same tokens stepped eagerly and by graph replay give
//!   bit-identical logits at every step of two prompts plus `GEN_STEPS`
//!   generated tokens. A stale host sum or a missing wait breaks it.
//! - per layer, same input: every routed layer run on its own at position 0
//!   with the all-card chain's own input, for the first `LAYER_TOKENS` prompts'
//!   last tokens. The routing and every card slot's expert output are
//!   bit-identical to the all-card layer's, every host slot's is zero, the
//!   layer output is `(Σ_card w·down + (shexp + hsum)) + resid` within the
//!   rounding of that sum, and the host sum differs from the all-card
//!   model's weighted sum of the same experts by no more than
//!   `BAND_FACTOR` times the error model's prediction.
//! - end to end: on the teacher-forced path of the e2e set (the reference's
//!   own tokens fed back), the hybrid argmax equals the all-card argmax
//!   wherever the all-card margin is above the derived flip band; every
//!   disagreement is printed with its margin.
//!
//! And once: `n_l` = 32 with the overlap lever off captures the same node
//! count and kinds and gives the same logits, bit for bit, as with it on.
//!
//! And the host tier's refusals, on a boundary of its own over host experts
//! that scale each column by its listed weights (no model file): a handoff
//! the card should already have refused — a routed id the slot map does not
//! know, a non-finite activation — runs no host expert, its sum is NaN and
//! the tier records it once. The caller sees the card's fault when the card
//! raised one at or before the layer, or unlabelled (planted by a
//! `norm_quant` launch over a NaN column: `Fault::at(layer, NormQuant)`), and
//! the host's own error — "host saw undefined input the card did not refuse"
//! — when the word is clean or names a later layer. The step service (eager,
//! one row, unwatched) fails with a named host error, never the host
//! experts' own, and the word read after the stream drains names it through
//! `hybrid::name_refusal`, as `GpuModel::name_host_refusal` does; the batch
//! service (three columns: the refused column NaN, the others the clean
//! run's bit for bit) names it itself when it watches the word and returns
//! when it does not. The word is clean before and after a clean service.
//! Last, on the engine itself (`n_l` = 32): a NaN in a hybrid layer's input
//! residual through `step_layer_hybrid` returns the card's fault at that
//! layer, with one refusal recorded by the host tier.
//!
//! Printed and not judged: the host tier's counters (services, the host's
//! share of the routed weight², the host leg, pool parks inside a service,
//! services whose go was already there) and the allocations per steady
//! step. Nothing here is a timing: the box's 3090 runs gates, and the lead
//! times on the A6000.
//!
//! The error model (derived before the first run). Per routed expert the
//! host and the card quantize the activations by different rules — the host
//! by the CPU engine's (q8_K 256-value blocks for the Q3_K gate/up, q8_2
//! 32-value blocks with a bf16 scale for the Q5_0 down), the card by q8_1
//! 128-value blocks and 32-value blocks with an f32 scale. With
//! `e_rule(v) = rms(q_rule(v) − v) / rms(v)`, one expert's output differs by
//! a relative RMS of about `r² = 2·(e_256(x)² + e_128(x)²) + e_q82(h)² +
//! e_q32(h)²`: the two gate/up branches each carry the difference of the two
//! rules' noise on `x`, and the down carries the two rules' independent
//! noise on its input `h` (the scale formats differ and `h` itself already
//! differs, so neither rounding cancels). Over the host slots of a layer,
//! `rms(ΔF)² ≈ Σ_s w_s² · rms(down_s)² · r_s²`. The band is `BAND_FACTOR`
//! times that, because the weight rows are not random. Margins: `σ(margin
//! difference) ≤ √(3h) · SIGMA_OURS` with `h` the host's measured share of
//! the routed weight², and the flip band is `TAIL_FACTOR` times that bound.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_hybrid: built without the `gpu` feature; see `just gate-gpu-hybrid`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hybrid", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::fused::FusedKernels;
    use bloomery_gpu::hybrid::{
        Boundary, BoundaryShape, HostExperts, Hybrid, HybridConfig, SlotMap, levers, name_refusal,
    };
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{
        Deepseek2Model, Fault, FaultSite, Gpu, GpuError, LAYER_NONE, NodeInfo, Q8Act,
    };
    use bloomery_gpu_gates::nodes::{count_kinds, kind_name};
    use bloomery_gpu_gates::prompts::{GreedyRow, PromptRow, read_greedy, read_prompts};
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, data_dir, open_model, q8_1_dequant, ref_model_path,
        verdict,
    };
    use cuda_core::DeviceBuffer;
    use cuda_core::sys;
    use gguf::{GgmlType, Gguf, Split};
    use model::Tensor2;
    use model::arch::deepseek2::derived::Derived;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Counts allocator calls, for the per-step allocation diagnostic.
    struct Counting;

    static ALLOCS: AtomicU64 = AtomicU64::new(0);

    // SAFETY: every call forwards to `System` with the caller's own
    // arguments; the counter is a relaxed atomic that allocates nothing.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the caller's layout, passed through unchanged.
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: `ptr` came from `alloc` above with this layout.
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the caller's layout, passed through unchanged.
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: `ptr` came from this allocator with `layout`; the new
            // size is the caller's.
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL: Counting = Counting;

    /// Cache rows: the e2e set's longest prompt and its 32 forced steps fit,
    /// as in `gate_e2e`, whose node count this model then shares.
    const CTX_MAX: usize = 256;
    /// The two card prefixes the round names.
    const N_LS: [usize; 2] = [32, 0];
    /// Generated steps after each eager = replay prompt.
    const GEN_STEPS: usize = 8;
    /// Prompts the eager = replay arm runs.
    const REPLAY_PROMPTS: usize = 2;
    /// Prompts whose last token the per-layer arm runs through every routed
    /// layer.
    const LAYER_TOKENS: usize = 4;
    /// Steady graph steps the allocation and park diagnostics average over.
    const STEADY_STEPS: usize = 16;
    /// The per-layer band over the error model's prediction.
    const BAND_FACTOR: f64 = 3.0;
    /// σ of the default path's margins against the exact truth: the error
    /// model's scale for the margin difference (`docs/plan.md` 「오차 모델」).
    const SIGMA_OURS: f64 = 0.367;
    /// The flip band over the σ bound: measured tails of two of our own
    /// rounding variants reached about nine RMS.
    const TAIL_FACTOR: f64 = 9.0;
    /// The forced path.
    const FORCED_FILE: &str = "greedy-ik-cuda-32.tsv";

    /// `CUgraphNodeType` values the boundary adds, and the kernel kind.
    const KINDS: [sys::CUgraphNodeType; 4] = [
        sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
        sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY,
        sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET,
        sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
    ];

    /// Node counts by kind, in `KINDS` order, and every other kind together.
    type Kinds = ([i64; 4], i64);

    /// What one `n_l` leaves behind: its verdict, and the replay logits,
    /// node count and kinds the overlap arm compares against.
    struct NlOutcome {
        ok: bool,
        replay: LogitsRun,
        nodes: usize,
        kinds: Kinds,
    }

    fn kinds(nodes: &[NodeInfo]) -> Kinds {
        let (c, other) = count_kinds(nodes, KINDS);
        (c.map(|n| n as i64), other as i64)
    }

    fn fmt_kinds(c: &[i64; 4], other: i64) -> String {
        let mut s = String::new();
        for (&kind, n) in KINDS.iter().zip(c) {
            s.push_str(&format!("{}={n} ", kind_name(kind)));
        }
        s.push_str(&format!("other={other}"));
        s
    }

    /// The all-card model's forced run: its argmax and top1-top2 margin at
    /// every forced position of every row.
    struct Forced {
        top1: Vec<Vec<u32>>,
        margin: Vec<Vec<f32>>,
    }

    /// `logits[top1]` minus the largest other logit.
    fn margin_at(logits: &[f32], top1: usize) -> f32 {
        let second = logits
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != top1)
            .map(|(_, &v)| v)
            .fold(f32::NEG_INFINITY, f32::max);
        logits[top1] - second
    }

    /// What a forced run hands each position: `(row, step, token, logits)`.
    type EachPosition<'a> = dyn FnMut(usize, usize, u32, &[f32]) + 'a;

    /// Logits per step, in step order.
    type LogitsRun = Vec<Vec<f32>>;

    /// Step `model` along every row's forced path (the prompt, then the
    /// reference's own tokens) and call `each(row, step, token, logits)`.
    fn forced_run(
        model: &mut Deepseek2Model,
        prompts: &[PromptRow],
        reference: &[GreedyRow],
        each: &mut EachPosition<'_>,
    ) -> Result<(), GateError> {
        for (r, (row, p)) in reference.iter().zip(prompts).enumerate() {
            if row.id != p.id {
                return Err(format!(
                    "forced_run: prompt {} against reference row {}",
                    p.id, row.id
                )
                .into());
            }
            model.reset()?;
            for s in 0..row.gen_ids.len() {
                let token = if s == 0 {
                    model.step(&p.tokens)?
                } else {
                    model.step(&[row.gen_ids[s - 1]])?
                };
                let logits = model.logits()?;
                each(r, s, token, &logits);
            }
        }
        Ok(())
    }

    /// `rms(q(v) − v) / rms(v)`.
    fn rel_noise(v: &[f32], q: &[f32]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&a, &b) in v.iter().zip(q) {
            num += f64::from(b - a) * f64::from(b - a);
            den += f64::from(a) * f64::from(a);
        }
        if den > 0.0 { (num / den).sqrt() } else { 0.0 }
    }

    fn rms(v: &[f32]) -> f64 {
        (v.iter().map(|&a| f64::from(a) * f64::from(a)).sum::<f64>() / v.len().max(1) as f64).sqrt()
    }

    /// The card's 32-value activation rule: per block, `d = amax / 127`,
    /// codes rounded to nearest, dequantized.
    fn q32_card(v: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; v.len()];
        for (blk, o) in v.chunks(32).zip(out.chunks_mut(32)) {
            let amax = blk.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            for (x, y) in blk.iter().zip(o) {
                *y = (x / d).round().clamp(-127.0, 127.0) * d;
            }
        }
        out
    }

    /// The CPU engine's activation rule for weights of type `ty`, as a round
    /// trip.
    fn q_host(ty: GgmlType, v: &[f32]) -> Result<Vec<f32>, GateError> {
        let mut out = vec![0.0f32; v.len()];
        gguf::quant::quantize_activations(ty, v, &mut out)?;
        Ok(out)
    }

    /// Expert `e`'s SwiGLU input to its down projection at `x`, by the CPU
    /// engine's matmuls: the `h` whose quantization noise the error model
    /// needs.
    fn expert_h(
        gguf: &Gguf,
        derived: &Derived,
        l: usize,
        e: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, GateError> {
        let plan = derived.block_plan(l)?.moe()?;
        let xt = Tensor2::from_vec(x.len(), 1, x.to_vec());
        let g = model::ops::matmul_q(gguf, &plan.gate_views[e], &xt)?;
        let u = model::ops::matmul_q(gguf, &plan.up_views[e], &xt)?;
        Ok(g.data
            .iter()
            .zip(&u.data)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect())
    }

    /// The all-card chain's per-layer snapshots for one token at position 0:
    /// each routed layer's input and taps.
    struct Chain {
        /// `(layer, input residual, taps)` per routed layer.
        layers: Vec<(usize, Vec<f32>, bloomery_gpu::arch::deepseek2::LayerTaps)>,
    }

    fn all_card_chain(
        a: &mut Deepseek2Model,
        token: u32,
        n_layers: usize,
    ) -> Result<Chain, GateError> {
        a.reset()?;
        let b0 = a.step_block0_taps(token, 0)?;
        let mut x = b0.l_out;
        let mut layers = Vec::new();
        for l in 1..n_layers {
            let taps = a.step_layer_taps(l, &x, 0)?;
            let next = taps.l_out.clone();
            if !taps.moe_ids.is_empty() {
                layers.push((l, x, taps));
            }
            x = next;
        }
        Ok(Chain { layers })
    }

    /// The per-layer arm's accumulated verdicts.
    #[derive(Default)]
    struct LayerArm {
        samples: usize,
        empty: usize,
        same_input: bool,
        same_routing: bool,
        card_slots_equal: bool,
        host_slots_zero: bool,
        recon_ok: bool,
        recon_worst: f64,
        band_worst: f64,
        written_worst: f64,
        ratios: Vec<f64>,
        written: Vec<f64>,
    }

    /// Run every routed layer of `chain` on `h` and fold the checks into `arm`.
    #[allow(clippy::too_many_arguments, reason = "one gate arm's inputs")]
    fn layer_arm(
        h: &mut Deepseek2Model,
        chain: &Chain,
        n_l: usize,
        gguf: &Gguf,
        derived: &Derived,
        arm: &mut LayerArm,
    ) -> Result<(), GateError> {
        h.reset()?;
        for (l, x_in, t) in &chain.layers {
            let hy = h.step_layer_hybrid(*l, x_in, 0)?;
            let hidden = t.l_out.len();
            let n_used = t.moe_ids.len();
            arm.samples += 1;
            arm.same_input &= bits_equal(&hy.ffn_inp, &t.ffn_inp);
            arm.same_routing &= hy.ids == t.moe_ids && bits_equal(&hy.weights, &t.moe_weights);
            let slot = |d: &[f32], s: usize| d[s * hidden..(s + 1) * hidden].to_vec();
            let mut host: Vec<usize> = Vec::new();
            for s in 0..n_used {
                let (hd, ad) = (slot(&hy.down, s), slot(&t.expert_down, s));
                if (t.moe_ids[s] as usize) < n_l {
                    arm.card_slots_equal &= bits_equal(&hd, &ad);
                } else {
                    arm.host_slots_zero &= hd.iter().all(|&v| v.to_bits() == 0);
                    host.push(s);
                }
            }
            // The combine's own sum, in its order, from the hybrid layer's pieces.
            for j in 0..hidden {
                let mut acc = 0.0f32;
                let mut mag = 0.0f64;
                for s in 0..n_used {
                    let p = hy.weights[s] * hy.down[s * hidden + j];
                    acc += p;
                    mag += f64::from(p.abs());
                }
                let joined = hy.shexp[j] + hy.hsum[j];
                let want = (acc + joined) + hy.ffn_inp[j];
                mag += f64::from(hy.shexp[j].abs())
                    + f64::from(hy.hsum[j].abs())
                    + f64::from(hy.ffn_inp[j].abs());
                let tol = 4.0 * f64::from(f32::EPSILON) * mag;
                let d = f64::from((hy.l_out[j] - want).abs());
                if d > tol {
                    arm.recon_ok = false;
                }
                if mag > 0.0 {
                    arm.recon_worst = arm.recon_worst.max(d / (f64::from(f32::EPSILON) * mag));
                }
            }
            if host.is_empty() {
                arm.empty += 1;
                continue;
            }
            // The band: the host sum against the card's sum of the same experts.
            let mut card = vec![0.0f32; hidden];
            let x = &t.ffn_norm;
            let (ex_host, ex_card) = (
                rel_noise(x, &q_host(GgmlType::Q3_K, x)?),
                rel_noise(x, &q8_1_dequant(x, hidden, 1)),
            );
            let (mut pred2, mut written2) = (0.0f64, 0.0f64);
            for &s in &host {
                let w = hy.weights[s];
                let ad = slot(&t.expert_down, s);
                for (c, &v) in card.iter_mut().zip(&ad) {
                    *c += w * v;
                }
                let hs = expert_h(gguf, derived, *l, t.moe_ids[s] as usize, x)?;
                let eh = rel_noise(&hs, &q_host(GgmlType::Q5_0, &hs)?).powi(2)
                    + rel_noise(&hs, &q32_card(&hs)).powi(2);
                let gu = 2.0 * (ex_host * ex_host + ex_card * ex_card);
                let base = f64::from(w * w) * rms(&ad).powi(2);
                pred2 += base * (gu + eh);
                written2 += base * gu;
            }
            let diff: Vec<f32> = hy.hsum.iter().zip(&card).map(|(a, b)| a - b).collect();
            let got = rms(&diff);
            let ratio = got / pred2.sqrt().max(f64::MIN_POSITIVE);
            let wratio = got / written2.sqrt().max(f64::MIN_POSITIVE);
            arm.band_worst = arm.band_worst.max(ratio);
            arm.written_worst = arm.written_worst.max(wratio);
            arm.ratios.push(ratio);
            arm.written.push(wratio);
        }
        Ok(())
    }

    fn median(v: &mut [f64]) -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// Eager then graph over the same tokens: the eager argmax picks the
    /// tokens, the graph run is fed them. Returns each run's logits per step.
    fn eager_replay(
        m: &mut Deepseek2Model,
        prompts: &[PromptRow],
    ) -> Result<(LogitsRun, LogitsRun), GateError> {
        let (mut eager, mut graph) = (Vec::new(), Vec::new());
        for p in prompts.iter().take(REPLAY_PROMPTS) {
            m.set_mode(StepMode::Eager);
            m.reset()?;
            let mut toks = Vec::with_capacity(GEN_STEPS);
            let mut next = m.step(&p.tokens)?;
            eager.push(m.logits()?);
            for _ in 0..GEN_STEPS {
                toks.push(next);
                next = m.step(&[next])?;
                eager.push(m.logits()?);
            }
            m.set_mode(StepMode::Graph);
            m.reset()?;
            m.step(&p.tokens)?;
            graph.push(m.logits()?);
            for &t in &toks {
                m.step(&[t])?;
                graph.push(m.logits()?);
            }
        }
        Ok((eager, graph))
    }

    fn logits_equal(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| bits_equal(x, y))
    }

    /// Allocator calls per steady graph step: `STEADY_STEPS` single-token
    /// steps after a warm one, logits never read.
    fn steady(m: &mut Deepseek2Model, token: u32) -> Result<f64, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        m.step(&[token])?;
        let a0 = ALLOCS.load(Ordering::Relaxed);
        let mut t = token;
        for _ in 0..STEADY_STEPS {
            t = m.step(&[t])?;
        }
        let a1 = ALLOCS.load(Ordering::Relaxed);
        Ok((a1 - a0) as f64 / STEADY_STEPS as f64)
    }

    /// An arm that could not run: its error printed as a failing verdict.
    fn arm_err(name: &str, e: &GateError) {
        println!("{name} error={e} FAIL");
    }

    /// Everything one `n_l` asserts. Returns whether it all held, and the
    /// eager = replay logits for the overlap comparison.
    #[allow(clippy::too_many_arguments, reason = "one n_l's inputs")]
    fn run_n_l(
        n_l: usize,
        overlap: bool,
        a: &Deepseek2Model,
        a_kinds: Kinds,
        routed: usize,
        chains: &[Chain],
        forced: &Forced,
        prompts: &[PromptRow],
        reference: &[GreedyRow],
        gguf: &Gguf,
        derived: &Derived,
    ) -> Result<NlOutcome, GateError> {
        let tag = format!("n_l={n_l} overlap={}", u8::from(overlap));
        let mut ok = true;
        let mut h = Deepseek2Model::load_hybrid(
            Split::open(ref_model_path()?)?,
            CTX_MAX,
            HybridConfig { n_l, overlap },
        )?;
        println!(
            "{tag} load resident_bytes={} all_card_resident_bytes={}",
            h.resident_bytes(),
            a.resident_bytes()
        );

        // Structure.
        let nodes = h.capture_step()?;
        let (hk, ho) = kinds(&h.step_graph_nodes()?);
        let r = routed as i64;
        // With experts on the card each routed layer adds two kernels (the
        // join and the card's slot list, host slots as HOST).
        let want = if n_l > 0 {
            [
                a_kinds.0[0] + 2 * r,
                a_kinds.0[1] + r,
                a_kinds.0[2] + r,
                a_kinds.0[3] + 2 * r,
            ]
        } else {
            [
                a_kinds.0[0] - r,
                a_kinds.0[1] + r,
                a_kinds.0[2],
                a_kinds.0[3] + 2 * r,
            ]
        };
        let want_nodes = (want.iter().sum::<i64>() + a_kinds.1) as usize;
        let s_ok = hk == want && ho == a_kinds.1 && nodes == want_nodes;
        ok &= s_ok;
        println!(
            "{tag} structure graph_nodes={nodes} want={want_nodes} kinds[{}] want[{}] {}",
            fmt_kinds(&hk, ho),
            fmt_kinds(&want, a_kinds.1),
            verdict(s_ok)
        );

        // Eager = replay.
        let replay = match eager_replay(&mut h, prompts) {
            Ok((e, g)) => {
                let same = logits_equal(&e, &g);
                ok &= same;
                println!(
                    "{tag} eager_vs_replay steps={} logits_bit_identical={same} {}",
                    e.len(),
                    verdict(same)
                );
                g
            }
            Err(e) => {
                arm_err(&format!("{tag} eager_vs_replay"), &e);
                ok = false;
                Vec::new()
            }
        };

        // Per layer.
        let mut arm = LayerArm {
            same_input: true,
            same_routing: true,
            card_slots_equal: true,
            host_slots_zero: true,
            recon_ok: true,
            ..LayerArm::default()
        };
        let mut layer_res = Ok(());
        for c in chains {
            layer_res = layer_arm(&mut h, c, n_l, gguf, derived, &mut arm);
            if layer_res.is_err() {
                break;
            }
        }
        match layer_res {
            Ok(()) => {
                let (mut rs, mut ws) = (arm.ratios.clone(), arm.written.clone());
                let band_ok = arm.band_worst <= BAND_FACTOR;
                let l_ok = arm.same_input
                    && arm.same_routing
                    && arm.card_slots_equal
                    && arm.host_slots_zero
                    && arm.recon_ok
                    && band_ok;
                ok &= l_ok;
                println!(
                    "{tag} per_layer samples={} no_host_slot={} same_input={} same_routing={} card_slots_bit_equal={} host_slots_zero={} recon_within_4ulp={} (worst {:.2} ulp·Σ|terms|) band ratio worst={:.3} median={:.3} (<= {BAND_FACTOR}) as_written_ratio worst={:.3} median={:.3} {}",
                    arm.samples,
                    arm.empty,
                    arm.same_input,
                    arm.same_routing,
                    arm.card_slots_equal,
                    arm.host_slots_zero,
                    arm.recon_ok,
                    arm.recon_worst,
                    arm.band_worst,
                    median(&mut rs),
                    arm.written_worst,
                    median(&mut ws),
                    verdict(l_ok)
                );
            }
            Err(e) => {
                arm_err(&format!("{tag} per_layer"), &e);
                ok = false;
            }
        }

        // End to end, teacher-forced.
        let stats0 = h.hybrid_stats().unwrap_or_default();
        h.set_mode(StepMode::Graph);
        let mut dis: Vec<(usize, usize, u32, u32, f32)> = Vec::new();
        let (mut sq, mut n) = (0.0f64, 0usize);
        let forced_res = forced_run(&mut h, prompts, reference, &mut |r, s, tok, logits| {
            let at = forced.top1[r][s] as usize;
            let m_h = margin_at(logits, at);
            let d = f64::from(m_h - forced.margin[r][s]);
            sq += d * d;
            n += 1;
            if tok as usize != at {
                dis.push((
                    reference[r].id,
                    s,
                    forced.top1[r][s],
                    tok,
                    forced.margin[r][s],
                ));
            }
        });
        let stats1 = h.hybrid_stats().unwrap_or_default();
        let served = stats1.served - stats0.served;
        let h_share = if served > 0 {
            (stats1.host_w2 - stats0.host_w2) / served as f64
        } else {
            0.0
        };
        let bound = (3.0 * h_share).sqrt() * SIGMA_OURS;
        let flip = TAIL_FACTOR * bound;
        if let Err(e) = &forced_res {
            arm_err(&format!("{tag} e2e"), e);
            ok = false;
        } else {
            let sigma = (sq / n.max(1) as f64).sqrt();
            let outside: Vec<_> = dis.iter().filter(|d| f64::from(d.4) > flip).collect();
            let e_ok = outside.is_empty();
            ok &= e_ok;
            println!(
                "{tag} e2e positions={n} disagree={} outside_flip_band={} host_share_w2={h_share:.3} sigma_margin_diff={sigma:.4} bound={bound:.4} flip_band={flip:.3} {}",
                dis.len(),
                outside.len(),
                verdict(e_ok)
            );
            for (id, s, a_t, h_t, m) in &dis {
                println!(
                    "  {tag} disagree prompt={id} step={s} all_card={a_t} hybrid={h_t} all_card_margin={m:.3}{}",
                    if f64::from(*m) > flip { " OUTSIDE" } else { "" }
                );
            }
        }

        // Diagnostics.
        let st = h.hybrid_stats().unwrap_or_default();
        if st.served > 0 {
            println!(
                "{tag} host_tier served={} go_early={} of_which_first_of_replay={} host_slots_per_service={:.2} host_share_w2={:.3} leg_us_mean={:.1} parks_in_service={} straggle_us_mean={:.2} straggle_us_max={:.1} (3090 gate load, not a timing)",
                st.served,
                st.go_early,
                st.go_early_first,
                st.host_slots as f64 / st.served as f64,
                st.host_w2 / st.served as f64,
                st.leg_ns as f64 / st.served as f64 / 1e3,
                st.parks_in_service,
                st.straggle_ns as f64 / (st.served - st.go_early).max(1) as f64 / 1e3,
                st.straggle_max_ns as f64 / 1e3
            );
        }
        match steady(&mut h, prompts[0].tokens[0]) {
            Ok(allocs) => {
                println!("{tag} steady allocs_per_step={allocs:.2} (hybrid step, graph mode)")
            }
            Err(e) => {
                arm_err(&format!("{tag} steady"), &e);
                ok = false;
            }
        }
        println!("{tag} verdict {}", verdict(ok));
        Ok(NlOutcome {
            ok,
            replay,
            nodes,
            kinds: (hk, ho),
        })
    }

    /// The refusal arm's boundary: its width, slots, layers, experts and the
    /// card's prefix of them.
    const R_HIDDEN: usize = 256;
    const R_USED: usize = 6;
    const R_LAYERS: usize = 4;
    const R_EXPERTS: usize = 16;
    const R_CARD: usize = 8;
    /// The layer the refusal arm serves.
    const R_LAYER: usize = 2;

    /// Host experts for the refusal arm: a column's sum is its activation
    /// times the sum of its listed weights, so it depends on its own input
    /// and list alone; `experts` counts the experts run. An id past the
    /// file is refused as the engine's host tier refuses it, by name.
    #[derive(Default)]
    struct Stub {
        experts: usize,
    }

    impl HostExperts for Stub {
        fn experts_into(
            &mut self,
            _layer: usize,
            x: &Tensor2,
            experts: &[(u32, f32)],
            out: &mut [f32],
        ) -> Result<(), GpuError> {
            if let Some(&(id, _)) = experts.iter().find(|e| e.0 as usize >= R_EXPERTS) {
                return Err(GpuError::Shape {
                    what: "refusal arm host",
                    detail: format!("expert {id} is not in the file"),
                });
            }
            self.experts += experts.len();
            let w: f32 = experts.iter().map(|e| e.1).sum();
            for (o, &v) in out.iter_mut().zip(&x.data) {
                *o = v * w;
            }
            Ok(())
        }

        fn experts_union_into(
            &mut self,
            layer: usize,
            x: &Tensor2,
            lists: &[&[(u32, f32)]],
            out: &mut [f32],
        ) -> Result<(), GpuError> {
            for (j, list) in lists.iter().enumerate() {
                let col = Tensor2::from_vec(x.ne0, 1, x.col(j).to_vec());
                self.experts_into(layer, &col, list, &mut out[j * x.ne0..][..x.ne0])?;
            }
            Ok(())
        }
    }

    /// A fresh tier over its own boundary, watching `gpu`'s fault word when
    /// `watch`.
    fn refusal_tier(gpu: &Gpu, watch: bool) -> Result<Hybrid<Stub>, GateError> {
        let shape = BoundaryShape {
            hidden: R_HIDDEN,
            n_used: R_USED,
        };
        let slots = SlotMap::prefix(0..R_LAYERS, R_EXPERTS, R_CARD)?;
        let b = Boundary::new(gpu.context(), gpu.stream(), shape, slots, true)?;
        let mut h = Hybrid::new(b, Stub::default(), R_LAYERS)?;
        if watch {
            h.watch_fault(gpu.fault_word())?;
        }
        gpu.stream().synchronize()?;
        Ok(h)
    }

    /// One eager step service of layer [`R_LAYER`] over the handoff (`ids`,
    /// `w`, `x`) written into row 0's image, then the go and the wait: the
    /// tier's answer, and the host sum as the page holds it after the stream
    /// drained.
    fn serve_step(
        gpu: &Gpu,
        h: &mut Hybrid<Stub>,
        ids: &[u32],
        w: &[f32],
        x: &[f32],
    ) -> Result<(Result<(), GpuError>, Vec<f32>), GateError> {
        let stream = gpu.stream();
        let seq = u32::try_from(h.stats().served)?;
        h.begin_chain(stream)?;
        {
            let t = h.boundary_mut().handoff_target();
            let l = t.layout;
            let mut img = vec![0u32; l.x + l.hidden];
            img[l.seq] = seq;
            img[l.ids..l.ids + l.n_used].copy_from_slice(ids);
            for (d, v) in img[l.weights..l.weights + l.n_used].iter_mut().zip(w) {
                *d = v.to_bits();
            }
            for (d, v) in img[l.x..].iter_mut().zip(x) {
                *d = v.to_bits();
            }
            t.image.copy_from_host(stream, &img)?;
        }
        h.boundary().enqueue_go(stream, R_LAYER)?;
        h.boundary().enqueue_back(stream)?;
        let r = h.layer_enqueued(R_LAYER);
        stream.synchronize()?;
        Ok((r, h.hsum_copy()?))
    }

    /// A `norm_quant` launch over a NaN column raising into layer `layer`
    /// ([`LAYER_NONE`]: the unlabelled sink), then a synchronize: the card
    /// refusing an input at that layer, finished before the host looks.
    fn plant(gpu: &Gpu, fused: &FusedKernels, layer: u32) -> Result<(), GateError> {
        let stream = gpu.stream();
        let mut act = Q8Act::with_k(stream, 1, R_HIDDEN)?;
        let x = DeviceBuffer::from_host(stream, &[f32::NAN; R_HIDDEN])?;
        let g = DeviceBuffer::from_host(stream, &[1.0f32; R_HIDDEN])?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, R_HIDDEN)?;
        let sink = if layer == LAYER_NONE {
            gpu.unlabelled_sink()
        } else {
            gpu.layer_sink(usize::try_from(layer)?)?
        };
        fused.enqueue_norm_quant(stream, &x, &g, 1e-6, &mut act, &mut y, sink)?;
        stream.synchronize()?;
        Ok(())
    }

    /// What a refused service's answer must be: the card's fault, or the
    /// host's own error (a `Protocol` saying so, and naming a later fault
    /// when the card holds one), or — an unwatched batch — a return.
    #[derive(Clone, Copy)]
    enum Want {
        Card(Fault),
        Host { later: bool },
        Returns,
    }

    /// One refused handoff: its name, the routing and activation handed
    /// over, the layer the card refused at beforehand, whether the tier
    /// watches the word, and the answer the caller must see.
    struct Case<'a> {
        name: &'a str,
        ids: &'a [u32],
        x: &'a [f32],
        planted: Option<u32>,
        watch: bool,
        want: Want,
    }

    impl<'a> Case<'a> {
        fn new(
            name: &'a str,
            ids: &'a [u32],
            x: &'a [f32],
            planted: Option<u32>,
            watch: bool,
            want: Want,
        ) -> Case<'a> {
            Case {
                name,
                ids,
                x,
                planted,
                watch,
                want,
            }
        }
    }

    fn answer_ok(r: &Result<(), GpuError>, want: Want) -> bool {
        match (r, want) {
            (Err(GpuError::Fault { fault, .. }), Want::Card(f)) => *fault == f,
            (Err(e @ GpuError::Protocol { .. }), Want::Host { later }) => {
                let t = e.to_string();
                t.contains("host saw undefined input the card did not refuse")
                    && t.contains(if later {
                        "later than this layer"
                    } else {
                        "is clean"
                    })
            }
            (Ok(()), Want::Returns) => true,
            _ => false,
        }
    }

    fn show(r: &Result<(), GpuError>) -> String {
        match r {
            Ok(()) => "Ok".to_string(),
            Err(e) => e.to_string(),
        }
    }

    /// The tier recorded exactly one refusal, at [`R_LAYER`], whose detail
    /// starts with `at`.
    fn recorded_once(h: &Hybrid<Stub>, at: &str) -> bool {
        h.stats().refusals == 1
            && h.refusal()
                .is_some_and(|r| r.layer == R_LAYER && r.detail.starts_with(at))
    }

    /// The host tier's refusals (module doc): whether every case held.
    fn refusal_arm() -> Result<bool, GateError> {
        let gpu = Gpu::new()?;
        let fused = FusedKernels::load(gpu.context())?;
        let stream = gpu.stream();
        let x = bloomery_gpu_gates::activations(R_HIDDEN, 3, 91);
        let w = [0.30f32, 0.20, 0.15, 0.15, 0.10, 0.10];
        // Slots 2, 3 and 5 go to the host.
        let ids = [0u32, 1, 9, 10, 3, 12];
        let host_w = w[2] + w[3] + w[5];
        let at = |l: u32| Fault::at(l, FaultSite::NormQuant);
        let layer = u32::try_from(R_LAYER)?;
        let mut ok = true;
        gpu.clear_fault()?;

        // The step service: the tier fails by name and releases the stream;
        // the step's caller (`GpuModel::name_host_refusal`) reads the word on
        // the engine stream once the step has drained and names the refusal.
        let x0 = &x[..R_HIDDEN];
        let before = gpu.fault()?;
        let mut h = refusal_tier(&gpu, false)?;
        let (r, sum) = serve_step(&gpu, &mut h, &ids, &w, x0)?;
        let want_sum: Vec<f32> = x0.iter().map(|&v| v * host_w).collect();
        let clean = r.is_ok()
            && bits_equal(&sum, &want_sum)
            && h.host_mut().experts == 3
            && h.stats().refusals == 0
            && h.take_step_refusal().is_none();
        let pass = clean && before.is_none() && gpu.fault()?.is_none();
        println!(
            "refusal step clean: answer {} host sum = the stub's and nothing recorded {clean} word \
             clean before {} and after {} {}",
            show(&r),
            before.is_none(),
            gpu.fault()?.is_none(),
            verdict(pass)
        );
        ok &= pass;
        let mut unknown = ids;
        unknown[4] = 99;
        let mut nan_x = x0.to_vec();
        nan_x[5] = f32::NAN;
        let cases = [
            Case::new(
                "unknown id, word clean",
                &unknown,
                x0,
                None,
                false,
                Want::Host { later: false },
            ),
            Case::new(
                "unknown id, card raised at the layer",
                &unknown,
                x0,
                Some(layer),
                false,
                Want::Card(at(layer)),
            ),
            Case::new(
                "non-finite x, card raised at the layer",
                &ids,
                &nan_x,
                Some(layer),
                false,
                Want::Card(at(layer)),
            ),
            Case::new(
                "non-finite x, card raised earlier",
                &ids,
                &nan_x,
                Some(layer - 1),
                false,
                Want::Card(at(layer - 1)),
            ),
            Case::new(
                "non-finite x, card raised unlabelled",
                &ids,
                &nan_x,
                Some(LAYER_NONE),
                false,
                Want::Card(at(LAYER_NONE)),
            ),
            Case::new(
                "non-finite x, card raised later",
                &ids,
                &nan_x,
                Some(layer + 1),
                false,
                Want::Host { later: true },
            ),
        ];
        for c in cases {
            let mut h = refusal_tier(&gpu, c.watch)?;
            if let Some(l) = c.planted {
                plant(&gpu, &fused, l)?;
            }
            // `serve_step` synchronizes the stream: the drain.
            let (r, sum) = serve_step(&gpu, &mut h, c.ids, &w, c.x)?;
            let raw = match &r {
                Err(e @ GpuError::Protocol { .. }) => {
                    let t = e.to_string();
                    t.contains("host saw undefined input: layer 2, row 0")
                }
                _ => false,
            };
            let recorded = recorded_once(&h, "row 0: ");
            // What `name_host_refusal` hands the caller; with no refusal
            // taken, the service's own answer passes through.
            let named = match h.take_step_refusal() {
                Some(refusal) if r.is_err() => Some(Err(name_refusal(&refusal, gpu.fault()?))),
                _ => None,
            };
            let answer = answer_ok(named.as_ref().unwrap_or(&r), c.want);
            let once = h.take_step_refusal().is_none();
            let sum_nan = sum.iter().all(|v| v.is_nan());
            let no_expert = h.host_mut().experts == 0;
            let word = gpu.take_fault()?;
            let word_ok = word == c.planted.map(at);
            let pass = raw && recorded && answer && once && sum_nan && no_expert && word_ok;
            println!(
                "refusal step {}: service \"{}\" a named host error {raw}, recorded once {recorded}; \
                 named \"{}\" as wanted {answer}, taken once {once}, host sum NaN {sum_nan}, no host \
                 expert ran {no_expert}, word as planted {word_ok} {}",
                c.name,
                show(&r),
                show(named.as_ref().unwrap_or(&r)),
                verdict(pass)
            );
            ok &= pass;
        }

        // The batch service: three columns, column 1 or 2 refused. It runs
        // after the card's work has finished (`plant` synchronizes, as the
        // engine's caller waits on the event its handoffs came down at), so
        // a watched tier reads the word itself.
        let cols = 3usize;
        let bids: Vec<u32> = (0..cols).flat_map(|_| ids).collect();
        let bw: Vec<f32> = (0..cols).flat_map(|_| w).collect();
        let batch = |h: &mut Hybrid<Stub>, bx: &[f32], bi: &[u32]| {
            let mut out = vec![0.0f32; cols * R_HIDDEN];
            let t = Tensor2::from_vec(R_HIDDEN, cols, bx.to_vec());
            let r = h.serve_batch(R_LAYER, &t, bi, &bw, &mut out);
            (r, out)
        };
        let mut h = refusal_tier(&gpu, true)?;
        let before = gpu.fault()?;
        let (r, clean_out) = batch(&mut h, &x, &bids);
        let want_out: Vec<f32> = x.iter().map(|&v| v * host_w).collect();
        let clean = r.is_ok() && bits_equal(&clean_out, &want_out) && h.stats().refusals == 0;
        let pass = clean && before.is_none() && gpu.fault()?.is_none();
        println!(
            "refusal batch clean: answer {} sums = the stub's and nothing recorded {clean} word clean \
             before {} and after {} {}",
            show(&r),
            before.is_none(),
            gpu.fault()?.is_none(),
            verdict(pass)
        );
        ok &= pass;
        let mut bunk = bids.clone();
        bunk[R_USED + 4] = 99;
        let mut bnan = x.clone();
        bnan[2 * R_HIDDEN + 5] = f32::NAN;
        let bcases = [
            (
                1,
                Case::new(
                    "column 1 unknown id, word clean",
                    &bunk,
                    &x,
                    None,
                    true,
                    Want::Host { later: false },
                ),
            ),
            (
                2,
                Case::new(
                    "column 2 non-finite, card raised at the layer",
                    &bids,
                    &bnan,
                    Some(layer),
                    true,
                    Want::Card(at(layer)),
                ),
            ),
            (
                2,
                Case::new(
                    "column 2 non-finite, card raised later",
                    &bids,
                    &bnan,
                    Some(layer + 1),
                    true,
                    Want::Host { later: true },
                ),
            ),
            (
                2,
                Case::new(
                    "column 2 non-finite, unwatched",
                    &bids,
                    &bnan,
                    Some(layer),
                    false,
                    Want::Returns,
                ),
            ),
        ];
        for (bad, c) in bcases {
            let mut h = refusal_tier(&gpu, c.watch)?;
            if let Some(l) = c.planted {
                plant(&gpu, &fused, l)?;
            }
            let (r, out) = batch(&mut h, c.x, c.ids);
            let answer = answer_ok(&r, c.want);
            let recorded = recorded_once(&h, &format!("column {bad} of {cols}: "));
            let col = |v: &[f32], j: usize| v[j * R_HIDDEN..(j + 1) * R_HIDDEN].to_vec();
            let bad_nan = col(&out, bad).iter().all(|v| v.is_nan());
            let others = (0..cols)
                .filter(|&j| j != bad)
                .all(|j| bits_equal(&col(&out, j), &col(&clean_out, j)));
            let word = gpu.take_fault()?;
            let word_ok = word == c.planted.map(at);
            let pass = answer && recorded && bad_nan && others && word_ok;
            println!(
                "refusal batch {}: answer \"{}\" as wanted {answer}, recorded once {recorded}, that \
                 column NaN {bad_nan}, other columns bit-identical {others}, word as planted \
                 {word_ok} {}",
                c.name,
                show(&r),
                verdict(pass)
            );
            ok &= pass;
        }
        stream.synchronize()?;
        println!("refusal verdict {}", verdict(ok));
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let refusals = refusal_arm()?;
        let lv = levers()?;
        if lv.n_l.is_some() {
            return Err("gate_hybrid: BLOOMERY_HYBRID_NL is set — the all-card reference must load all-card; unset it".into());
        }
        let prompts_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/prompts.tsv");
        let prompts = read_prompts(&prompts_path)?;
        let reference = read_greedy(&data_dir().join(FORCED_FILE))?;
        if reference.len() != prompts.len() {
            return Err(format!(
                "gate_hybrid: {} prompts, {} reference rows",
                prompts.len(),
                reference.len()
            )
            .into());
        }
        let gguf = open_model()?;
        let derived = Derived::new(&gguf)?;
        let n_layers = usize::try_from(gguf.block_count().ok_or("gate_hybrid: no block_count")?)?;
        let routed = (0..n_layers)
            .filter(|&l| derived.block_plan(l).is_ok_and(|b| b.routed))
            .count();

        let mut a = Deepseek2Model::load_full(Split::open(ref_model_path()?)?, CTX_MAX)?;
        println!(
            "all_card resident_bytes={} layers={n_layers} routed={routed} ctx_max={CTX_MAX}",
            a.resident_bytes()
        );

        // The all-card reference: its forced run, its step graph, its chains.
        a.set_mode(StepMode::Graph);
        let mut forced = Forced {
            top1: vec![Vec::new(); reference.len()],
            margin: vec![Vec::new(); reference.len()],
        };
        forced_run(&mut a, &prompts, &reference, &mut |r, _, tok, logits| {
            forced.top1[r].push(tok);
            forced.margin[r].push(margin_at(logits, tok as usize));
        })?;
        let a_kinds = kinds(&a.step_graph_nodes()?);
        println!(
            "all_card structure graph_nodes={} kinds[{}]",
            a_kinds.0.iter().sum::<i64>() + a_kinds.1,
            fmt_kinds(&a_kinds.0, a_kinds.1)
        );
        let mut chains = Vec::new();
        for p in prompts.iter().take(LAYER_TOKENS) {
            let token = *p.tokens.last().ok_or("gate_hybrid: empty prompt")?;
            chains.push(all_card_chain(&mut a, token, n_layers)?);
        }
        let mut ok = refusals;
        match steady(&mut a, prompts[0].tokens[0]) {
            Ok(allocs) => println!("all_card steady allocs_per_step={allocs:.2} (graph mode)"),
            Err(e) => {
                arm_err("all_card steady", &e);
                ok = false;
            }
        }

        let mut on32: Option<NlOutcome> = None;
        for n_l in N_LS {
            let out = run_n_l(
                n_l, true, &a, a_kinds, routed, &chains, &forced, &prompts, &reference, &gguf,
                &derived,
            )?;
            ok &= out.ok;
            if n_l == 32 {
                on32 = Some(out);
            }
        }
        let on32 = on32.ok_or("gate_hybrid: no n_l = 32 run to compare the overlap lever with")?;

        // The overlap lever: same nodes, same logits.
        {
            let mut h = Deepseek2Model::load_hybrid(
                Split::open(ref_model_path()?)?,
                CTX_MAX,
                HybridConfig {
                    n_l: 32,
                    overlap: false,
                },
            )?;
            let nodes = h.capture_step()?;
            let k = kinds(&h.step_graph_nodes()?);
            let s_ok = nodes == on32.nodes && k == on32.kinds;
            let same = match eager_replay(&mut h, &prompts) {
                Ok((e, g)) => logits_equal(&e, &g) && logits_equal(&g, &on32.replay),
                Err(e) => {
                    arm_err("n_l=32 overlap=0 eager_vs_replay", &e);
                    false
                }
            };
            ok &= s_ok && same;
            println!(
                "n_l=32 overlap=0 graph_nodes={nodes} kinds[{}] same_as_overlap_on={s_ok} logits_bit_identical_to_overlap_on={same} {}",
                fmt_kinds(&k.0, k.1),
                verdict(s_ok && same)
            );

            // The engine's own naming on a real layer: a NaN in a hybrid
            // layer's input residual is refused by the card's norm at that
            // layer and met by the host service in its handoff, and the
            // caller sees the card's fault (`GpuModel::name_host_refusal`),
            // never the host's error. It poisons the tier, so it runs last.
            let (l, x_in, _) = chains
                .first()
                .and_then(|c| c.layers.first())
                .ok_or("gate_hybrid: no routed layer")?;
            let mut x_nan = x_in.clone();
            x_nan[0] = f32::NAN;
            let r = h.step_layer_hybrid(*l, &x_nan, 0);
            let refused = h.hybrid_stats().is_some_and(|s| s.refusals == 1);
            let named = match &r {
                Err(GpuError::Fault { fault, .. }) => {
                    usize::try_from(fault.layer).is_ok_and(|fl| fl == *l)
                        && fault.sites & (1 << FaultSite::NormQuant as u32) != 0
                }
                _ => false,
            };
            let pass = named && refused;
            ok &= pass;
            println!(
                "n_l=32 engine refusal layer={l} input NaN at value 0: answer \"{}\" names the card's \
                 fault at the layer {named}, the host recorded one refusal {refused} {}",
                match &r {
                    Ok(_) => "Ok".to_string(),
                    Err(e) => e.to_string(),
                },
                verdict(pass)
            );
        }

        if ok {
            println!(
                "gate_hybrid: PASS — at n_l 32 and 0 the captured step carries exactly the boundary's nodes, \
                 replays its eager body bit for bit, runs every routed layer on the all-card routing with the \
                 card's slots unchanged, the host's slots zero and the host sum inside the error model's band, \
                 and leaves the all-card argmax only inside the flip band; the overlap lever changes the order \
                 and nothing else; a handoff the card should have refused names the card's fault, or says the \
                 card raised nothing."
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
