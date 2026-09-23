//! The hybrid MoE boundary gate (`bloomery_gpu::hybrid`): DeepSeek-V2-Lite
//! with experts `[0, n_l)` of every routed layer on the card and the rest on
//! the host, against the all-card model, for `n_l` in {32, 0}.
//!
//! What is asserted, per `n_l`:
//! - structure: the captured step's node count and node kinds are the
//!   all-card step's plus the boundary's per routed layer — the handoff copy
//!   (memcpy), the go and the wait (two batch memops), the join (one kernel),
//!   and, with experts on the card, the zeroing of their outputs (memset);
//!   with none on the card the two `_sel` launches drop out.
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
    use bloomery_gpu::hybrid::{HybridConfig, levers};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Deepseek2Model, NodeInfo};
    use bloomery_gpu_gates::nodes::{count_kinds, kind_name};
    use bloomery_gpu_gates::prompts::{GreedyRow, PromptRow, read_greedy, read_prompts};
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, data_dir, open_model, q8_1_dequant, ref_model_path,
        verdict,
    };
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
        let want = if n_l > 0 {
            [
                a_kinds.0[0] + r,
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

    pub fn run() -> Result<(), GateError> {
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
        let mut ok = true;
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
        }

        if ok {
            println!(
                "gate_hybrid: PASS — at n_l 32 and 0 the captured step carries exactly the boundary's nodes, \
                 replays its eager body bit for bit, runs every routed layer on the all-card routing with the \
                 card's slots unchanged, the host's slots zero and the host sum inside the error model's band, \
                 and leaves the all-card argmax only inside the flip band; the overlap lever changes the order \
                 and nothing else."
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
