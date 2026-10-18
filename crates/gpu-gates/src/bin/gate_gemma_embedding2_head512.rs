//! GPU gate for the card kernels of EmbeddingGemma 2's global attention
//! layers, model-less at the model's shapes: 4 query heads over one
//! key/value head of 512 values, a NEOX rope over all 512 values at θ 1e6,
//! RMS eps 1e-6 and an attention scale of 1 (llama.cpp's
//! `f_attention_scale` for the architecture: the query norm stands in for
//! the scale). Inputs are seeded ([`activations`]); nothing is read from a
//! model file.
//!
//! 1. Rope (`rope_neox::head_norm_neox_append_512`), against the shared host
//!    transcription of its rule: the norm (each thread's two squares added in
//!    f64, the warp butterfly, the eight warp sums in warp order, `(sum / 512)
//!    as f32`), `(scale · gain) · x`, the turn of pairs `(i, i + 256)` by the
//!    table's row at the token's position, the query read from its own rows
//!    at head stride 512, the key turned in place, both planes' rows the f16
//!    (nearest even) of the turned key and of the value: all bit for bit, the
//!    planes' other rows untouched, and a rerun bit for bit. Runs use 4/1
//!    and 8/2 query/key heads and end at position `ctx - 1`. A position at the
//!    cache's height raises `cache_pos` with the launch's layer, leaves that
//!    token's heads NaN and appends nothing; every other token is the clean
//!    run's.
//! 2. Prefill flash (`flash_gqa_prefill_512::enqueue_p4`): each output within
//!    its first-order bound of the exact attention computed here in f64 on the
//!    same f16 keys and values (`flash_band`'s exact attention and prefill
//!    bound at a head of 512 and a key tile of 32). The launches: every row
//!    over the whole input (the encoder's counts, `T` ∈ [`ENC_T`]), an
//!    unpadded encoder at `T = ctx = 33`, counts
//!    `p0 + t + 1` ([`SEED_CAUSAL`], crossing the 32-key tile and the
//!    8-position block edges), counts scattered within one block
//!    ([`SCATTER_T`]), one encoder launch at scale [`SCALE_ALT`], and one
//!    causal launch of [`GROUP`] heads (two key heads, two packs of four
//!    query heads each), whose blocks map to every key head and pack; the
//!    output starts NaN, so a value no block writes is out of band. Every row
//!    is bit for bit the same row launched alone, NaN in the cache rows past
//!    the launch's largest count changes no bit, a rerun is bit for bit. Then
//!    a count of zero and one past the cache (`key_count`, NaN in those rows,
//!    the rest the clean run's), the captured launch (one node) the eager
//!    bits, groups that are not a multiple of four query heads a key head
//!    (3 over 1, 4 over 2) refused by the group's error, and windows at
//!    aligned offsets accepted while `q` at 4 bytes past 8 and `kc`/`vc` at 8
//!    bytes past 16 are refused by name.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemma_embedding2_head512", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::fault::{Fault, FaultSink, FaultSite, LAYER_NONE};
    use bloomery_gpu::flash_gqa_prefill::GqaPrefillArgs;
    use bloomery_gpu::flash_gqa_prefill_512::{FlashGqaPrefill512, KEY_TILE_512};
    use bloomery_gpu::rope_neox::{HEAD_512 as HEAD, RopeNeoxKernels};
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::flash_band::judge_prefill;
    use bloomery_gpu_gates::prefill_gate::{WindowCase, Windows, cache, window_refusals};
    use bloomery_gpu_gates::rope_gate::{self, RopeIn, RopeOut, RopeShape};
    use bloomery_gpu_gates::{GateError, activations, bits_equal, checks_failed, verdict};
    use cuda_core::{CudaStream, DeviceBuffer};

    /// The model's global attention shape.
    const N_HEAD: usize = 4;
    const N_KV: usize = 1;
    /// The grouped launch's heads `(n_head, n_kv)`: two key heads, two packs
    /// of four query heads each.
    const GROUP: (usize, usize) = (16, 2);
    const THETA: f32 = 1.0e6;
    const EPS: f32 = 1.0e-6;
    const SCALE: f32 = 1.0;
    /// The scale of one more encoder launch: a kernel that ignores its
    /// `scale` argument passes at 1.
    const SCALE_ALT: f32 = 0.0625;
    const ALT_T: usize = 40;

    /// The rope check's tokens, first position and cache height.
    const ROPE_M: usize = 9;
    const ROPE_P0: usize = ROPE_CTX - ROPE_M;
    const ROPE_CTX: usize = 40;

    /// The encoder launches: `T` rows, each over all `T` keys.
    const ENC_T: [usize; 6] = [1, 8, 31, 33, 100, 512];
    /// The causal launches `(p0, T)`: row `t` over `p0 + t + 1` keys.
    const SEED_CAUSAL: [(usize, usize); 5] = [(0, 17), (31, 9), (32, 40), (1000, 9), (2047, 5)];
    /// The scattered launch's rows: counts `(37·t mod 97) + 1`.
    const SCATTER_T: usize = 24;
    /// Rows past the largest count in each cache, NaN in the padded copy.
    const PAD: usize = 40;
    /// The seeded query's scale: scores of a few tens, as a normed head's.
    const SEED_Q_SCALE: f32 = 2.0;
    /// The layer the fault checks label their launches with.
    const LAYER: usize = 11;

    // ------------------------------------------------------------ rope

    fn rope_run(
        kern: &RopeNeoxKernels,
        stream: &CudaStream,
        inp: &RopeIn,
        pos: &[u32],
        fault: FaultSink,
        (n_head, n_kv): (usize, usize),
    ) -> Result<RopeOut, GateError> {
        rope_gate::run(
            stream,
            inp,
            pos,
            RopeShape {
                head: HEAD,
                n_head,
                n_kv,
                ctx: ROPE_CTX,
                eps: EPS,
            },
            fault,
            |s, args| {
                // SAFETY: rope_gate::run checks that the positions are distinct.
                unsafe { kern.enqueue_head_norm_neox_append_512(s, args) }
            },
        )
    }

    fn rope_inputs((n_head, n_kv): (usize, usize)) -> Result<(RopeIn, Vec<u32>), GateError> {
        let table_rt = RopeTable::new(&RopeSpec::window(THETA, HEAD))?;
        let mut table = Vec::with_capacity(ROPE_CTX * HEAD);
        for p in 0..ROPE_CTX {
            table_rt.push(u32::try_from(p)?, Direction::Forward, &mut table);
        }
        let gain = |seed: u32, base: f32| -> Vec<f32> {
            activations(HEAD, 1, seed)
                .iter()
                .map(|v| base + 0.5 * v)
                .collect()
        };
        let inp = RopeIn {
            qg: activations(HEAD, ROPE_M * n_head, 3),
            k: activations(HEAD, ROPE_M * n_kv, 6)
                .iter()
                .map(|v| 4.0 * v)
                .collect(),
            v: activations(HEAD, ROPE_M * n_kv, 7),
            gq: gain(4, 1.25),
            gk: gain(5, 1.3),
            table,
        };
        let pos: Vec<u32> = (0..ROPE_M)
            .map(|t| u32::try_from(ROPE_P0 + t))
            .collect::<Result<_, _>>()?;
        Ok((inp, pos))
    }

    fn rope_check(
        gpu: &Gpu,
        kern: &RopeNeoxKernels,
        heads: (usize, usize),
    ) -> Result<bool, GateError> {
        let (inp, pos) = rope_inputs(heads)?;
        let shape = RopeShape {
            head: HEAD,
            n_head: heads.0,
            n_kv: heads.1,
            ctx: ROPE_CTX,
            eps: EPS,
        };
        let exact = rope_gate::check::<HEAD, HEAD, HEAD>(gpu, &inp, &pos, shape, LAYER, |p, f| {
            rope_run(kern, gpu.stream(), &inp, p, f, heads)
        })?;
        let overflow =
            rope_gate::check_overflow::<HEAD>(gpu, &inp, &pos, shape, LAYER, |i, p, f| {
                rope_run(kern, gpu.stream(), i, p, f, heads)
            })?;
        Ok(exact && overflow)
    }

    // --------------------------------------------------------- prefill

    const WIDTH: usize = N_HEAD * HEAD;
    const WHAT: &str = "flash_gqa_prefill_512::enqueue_p4";

    /// One launch over `n_keys.len()` rows of `heads = (n_head, n_kv)` into
    /// an output that starts NaN, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's inputs, each a value the callers vary"
    )]
    fn run_pref(
        kp: &FlashGqaPrefill512,
        gpu: &Gpu,
        q: &[f32],
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
        scale: f32,
        (n_head, n_kv): (usize, usize),
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let qd = DeviceBuffer::from_host(stream, q)?;
        let t = n_keys.len();
        let nk = DeviceBuffer::from_host(stream, n_keys)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * n_head * HEAD])?;
        kp.enqueue_p4(
            stream,
            GqaPrefillArgs {
                q: &qd,
                kc,
                vc,
                n_keys: &nk,
                scale,
                n_head,
                n_kv,
                ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// One seeded launch: its name, its rows' counts, its scale and its
    /// heads `(n_head, n_kv)` and cache padding.
    type Seeded = (String, Vec<usize>, f32, (usize, usize), usize);

    /// The seeded launches: the encoder's counts, the causal and the
    /// scattered at the model's scale and shape, one encoder launch at
    /// [`SCALE_ALT`], and one causal launch of [`GROUP`] heads.
    fn seeded() -> Vec<Seeded> {
        let model = (N_HEAD, N_KV);
        let enc = ENC_T
            .iter()
            .map(move |&t| (format!("encoder T={t}"), vec![t; t], SCALE, model, PAD));
        let causal = SEED_CAUSAL.iter().map(move |&(p0, t)| {
            (
                format!("causal T={t} p0={p0}"),
                (0..t).map(|i| p0 + i + 1).collect(),
                SCALE,
                model,
                PAD,
            )
        });
        let scatter = std::iter::once((
            format!("scattered T={SCATTER_T}"),
            (0..SCATTER_T).map(|t| (37 * t) % 97 + 1).collect(),
            SCALE,
            model,
            PAD,
        ));
        let alt = std::iter::once((
            format!("encoder T={ALT_T} scale {SCALE_ALT}"),
            vec![ALT_T; ALT_T],
            SCALE_ALT,
            model,
            PAD,
        ));
        let (gh, gkv) = GROUP;
        let group = std::iter::once((
            format!("causal T=17 p0=40 heads {gh}/{gkv}"),
            (0..17).map(|i| 40 + i + 1).collect(),
            SCALE,
            GROUP,
            PAD,
        ));
        let tight = std::iter::once((
            "encoder T=33 unpadded".to_string(),
            vec![33; 33],
            SCALE,
            model,
            0,
        ));
        enc.chain(causal)
            .chain(scatter)
            .chain(alt)
            .chain(group)
            .chain(tight)
            .collect()
    }

    /// The prefill clause (module doc, 2) up to the window refusals. Returns
    /// the verdict and what [`misaligned`] reads, which the caller runs last.
    fn prefill_check(gpu: &Gpu, kp: &FlashGqaPrefill512) -> Result<(bool, Windows), GateError> {
        let stream = gpu.stream();
        let model = (N_HEAD, N_KV);
        let mut ok = true;
        let mut seed = 700u32;
        for (name, cu, scale, heads, pad) in seeded() {
            seed += 7;
            let (n_head, n_kv) = heads;
            let w = n_head * HEAD;
            let live = cu.iter().copied().max().unwrap_or(0);
            let ctx = live + pad;
            let t = cu.len();
            let c = cache(stream, HEAD, n_kv, ctx, live, seed)?;
            let q: Vec<f32> = activations(HEAD, t * n_head, seed + 3)
                .iter()
                .map(|v| v * SEED_Q_SCALE)
                .collect();
            let counts: Vec<u32> = cu
                .iter()
                .map(|&n| u32::try_from(n))
                .collect::<Result<_, _>>()?;
            let y = run_pref(kp, gpu, &q, &counts, (&c.kc, &c.vc), ctx, scale, heads)?;
            let y2 = run_pref(kp, gpu, &q, &counts, (&c.kc, &c.vc), ctx, scale, heads)?;
            let yn = run_pref(kp, gpu, &q, &counts, (&c.kn, &c.vn), ctx, scale, heads)?;
            let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
            let (band, worst) =
                judge_prefill::<HEAD, KEY_TILE_512>(&q, &cu, &c.host, &y, scale, heads, 0);
            let mut differ = 0usize;
            for (r, &n) in counts.iter().enumerate() {
                let one = run_pref(
                    kp,
                    gpu,
                    &q[r * w..(r + 1) * w],
                    &[n],
                    (&c.kc, &c.vc),
                    ctx,
                    scale,
                    heads,
                )?;
                differ += usize::from(!bits_equal(&y[r * w..(r + 1) * w], &one));
            }
            let pass = rerun && nan_same && band && differ == 0;
            println!(
                "prefill {name} ctx={ctx}: measured/bound {worst:.3e} band={band} \
                 rows_alone_differing={differ} rerun={rerun} nan_padding_same={nan_same} {}",
                verdict(pass)
            );
            ok &= pass;
        }

        // A count of zero and one past the cache, on a 17-row launch at 63.
        let (t, p0) = (17usize, 63usize);
        let ctx = p0 + t + PAD;
        let w = WIDTH;
        let c = cache(stream, HEAD, N_KV, ctx, p0 + t, 91)?;
        let q = activations(HEAD, t * N_HEAD, 94);
        let clean: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i + 1))
            .collect::<Result<_, _>>()?;
        let (bad_hi, bad_zero) = (3usize, 10usize);
        let mut bad = clean.clone();
        bad[bad_hi] = u32::try_from(ctx + 1)?;
        bad[bad_zero] = 0;
        let before = gpu.fault()?;
        let y = run_pref(kp, gpu, &q, &clean, (&c.kc, &c.vc), ctx, SCALE, model)?;
        let after_clean = gpu.fault()?;
        let yb = run_pref(kp, gpu, &q, &bad, (&c.kc, &c.vc), ctx, SCALE, model)?;
        let raised = gpu.take_fault()?;
        let mut others = true;
        let mut nan = true;
        for r in 0..t {
            let (a, b) = (&y[r * w..(r + 1) * w], &yb[r * w..(r + 1) * w]);
            if r == bad_hi || r == bad_zero {
                nan &= b.iter().all(|v| v.is_nan());
            } else {
                others &= bits_equal(a, b);
            }
        }
        let want = Fault::at(LAYER_NONE, FaultSite::KeyCount);
        let fault_ok =
            before.is_none() && after_clean.is_none() && raised == Some(want) && nan && others;
        println!(
            "prefill fault: counts {} (past ctx {ctx}) and 0 at rows {bad_hi}, {bad_zero}: word \
             {raised:?} (want {want:?}), those rows NaN {nan}, other rows bit-identical {others} {}",
            ctx + 1,
            verdict(fault_ok)
        );
        ok &= fault_ok;

        // The captured launch: one node, the eager bits.
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let nk = DeviceBuffer::from_host(stream, &clean)?;
        let mut yg = DeviceBuffer::from_host(stream, &vec![f32::NAN; t * w])?;
        let graph = gpu.capture(|s| {
            kp.enqueue_p4(
                s,
                GqaPrefillArgs {
                    q: &qd,
                    kc: &c.kc,
                    vc: &c.vc,
                    n_keys: &nk,
                    scale: SCALE,
                    n_head: N_HEAD,
                    n_kv: N_KV,
                    ctx,
                    t,
                    fault: gpu.unlabelled_sink(),
                    y: &mut yg,
                },
            )
        })?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let same = bits_equal(&yg.to_host_vec(stream)?, &y);
        let nodes = graph.node_count();
        let graph_ok = same && nodes == 1;
        println!(
            "prefill graph T={t} p0={p0}: eager_vs_graph_bit_identical={same} graph_nodes={nodes} \
             {}",
            verdict(graph_ok)
        );
        ok &= graph_ok;

        // Groups that are not a multiple of four query heads a key head, over
        // planes that hold every key head, so only the group refuses them.
        let planes2 = vec![0u16; 2 * ctx * HEAD];
        let (k2, v2) = (
            DeviceBuffer::from_host(stream, &planes2)?,
            DeviceBuffer::from_host(stream, &planes2)?,
        );
        let mut yr = DeviceBuffer::<f32>::zeroed(stream, t * w)?;
        for (n_head, n_kv) in [(N_HEAD - 1, N_KV), (4, 2)] {
            let refused = kp.enqueue_p4(
                stream,
                GqaPrefillArgs {
                    q: &qd,
                    kc: &k2,
                    vc: &v2,
                    n_keys: &nk,
                    scale: SCALE,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault: gpu.unlabelled_sink(),
                    y: &mut yr,
                },
            );
            let refuse_ok = matches!(
                refused,
                Err(GpuError::Shape {
                    what,
                    ref detail,
                }) if what == WHAT && detail.starts_with("the kernel packs")
            );
            println!(
                "prefill refusal n_head={n_head} over n_kv={n_kv}: {} {}",
                refused
                    .err()
                    .map_or("accepted".to_string(), |e| e.to_string()),
                verdict(refuse_ok)
            );
            ok &= refuse_ok;
        }
        let win = Windows {
            q,
            kb: c.kb,
            vb: c.vb,
            clean,
            y,
            ctx,
        };
        Ok((ok, win))
    }

    fn misaligned(kp: &FlashGqaPrefill512, gpu: &Gpu, win: &Windows) -> Result<bool, GateError> {
        window_refusals(
            gpu,
            WHAT,
            "prefill",
            WindowCase {
                q: &win.q,
                planes: (&win.kb, &win.vb),
                counts: &win.clean,
                y: &win.y,
                heads: (N_HEAD, N_KV),
                ctx: win.ctx,
                scale: SCALE,
            },
            |s, args| kp.enqueue_p4(s, args),
        )
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        let rope = RopeNeoxKernels::load(ctx)?;
        let kp = FlashGqaPrefill512::load(ctx)?;
        println!(
            "gate_gemma_embedding2_head512: device {}: {N_HEAD}/{N_KV} heads of {HEAD}, rope over \
             {HEAD} at θ {THETA:e}, scale {SCALE:e}, key tile {KEY_TILE_512}",
            gpu.device_name()?
        );
        let mut ok = true;
        for heads in [(N_HEAD, N_KV), (8, 2)] {
            let r = rope_check(&gpu, &rope, heads)?;
            println!("rope-512 heads {}/{} {}", heads.0, heads.1, verdict(r));
            ok &= r;
        }
        let (p, win) = prefill_check(&gpu, &kp)?;
        println!("prefill flash 512 p4 {}", verdict(p));
        ok &= p;
        // Last: the window refusals, whose failure would be a sticky error.
        let m = misaligned(&kp, &gpu, &win)?;
        println!("prefill windows {}", verdict(m));
        ok &= m;
        println!("gate_gemma_embedding2_head512: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
