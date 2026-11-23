//! GPU gate for Inkling 2's prompt-pass kernels — the rows forms the batched prompt runs where the
//! step runs one token: the rel-bias attention over a pass's rows (`rel_bias_prefill`, the tensor
//! core walk at the group of three and of six) and the looped core's message controller over a
//! pass's rows (`loop_control::enqueue_rows`, `enqueue_attn_rows`), each against the step entry
//! the decode runs on the same inputs. Seeded at the Mini's shapes (24 query and 8 key heads of
//! 128 values, window 1024, relative extent 1024; hidden 2,304, gate rank 64), and the prompt MoE
//! over the model file's own expert stacks (clauses 8 and 9).
//!
//! 1. `prefill_attn`: a pass of rows over a cached prefix at the group of three, against
//!    `rel_bias::enqueue` at the same rows — windowed rows that cross the window's turnover (live
//!    counts 990 to 1066, a ragged last block) and rows at the sequence head (counts 1 to 40),
//!    global rows whose deep keys sit past the extent (counts 1061 to 1100), the sinks ranging
//!    from `−inf` to 30 above every logit — each within [`PP_BAND`] of the step, which is under
//!    the step gate's own band of the step against its f64 re-dot (`gate_inkling2_attn`'s
//!    `ATTN_BAND`, 2e-4), so the prompt rows meet that bar with room; a rerun bit-identical.
//! 2. `prefill_ring`: the same windowed pass over a ring of the window plus the pass's rows
//!    holding the keys the rows read: the plane's bits.
//! 3. `prefill_group6`: the group of six over the eight key heads equals, bit for bit, the group
//!    of three over the same key planes each listed twice — a head's arithmetic is its own warp's
//!    whatever the group, so the one body serves both files.
//! 4. `prefill_faults` / `prefill_refused`: a live count past `ctx` raises `KeyCount` with NaN
//!    rows; a group of four, a ring that cannot hold the window and the pass, a zero extent and a
//!    short buffer are each refused by name.
//! 5. `control_rows`: the controller's rows form equals the step pair bit for bit — the scratch
//!    and the output — at 512 rows of the Mini's (2,304, 64) and at ragged and off-group shapes.
//! 6. `control_attn_rows`: the three attention modules' rows form equals `enqueue_attn`'s bits.
//! 7. `control_rows_nan`: a NaN row stays NaN with the fault raised in both forms, every other
//!    row the step's bits.
//! 8. `moe_rows`: the prompt route block (route table, the 32-value GEMMs over the file's NVFP4
//!    planes with their per-expert scales or its Q8_0 blocks, SiTU between gate·up and down)
//!    over 512 seeded rows against the step's leg on the same rows, an NVFP4 layer and the Q8_0
//!    coda, within [`MOE_BAND`] (derived at the constant).
//! 9. `moe_leg_rows`: the leg over the 512 rows equals the leg over one row, bit for bit.
//!
//! Clauses 8 and 9 read the tier's model file (the fixture's on the fixture tier, the Mini's on
//! the real tier): its expert stacks, nothing else.
//!
//! [derived] [`PP_BAND`]: the two kernels compute the same f32 rule in two orders — the step's
//! lane dot and segment merge against the tensor core's k16 steps and the tile walk — with the
//! query and the weights carried as two f16 halves to `2⁻²²` of each value. The terms at the
//! seeded magnitudes (logits up to ~12): the dot's f32 sums `√128 · 2⁻²⁴` of the logit
//! (~1e-6), `ex2.approx` 2⁻²² a weight in either walk, the value sums over at most 1,100 keys
//! `√1100 · 2⁻²⁴` (~2e-6 of the largest output); their sum is under 5e-6, and 5e-5 is ten times
//! it. The query's single f16 rounding alone (2⁻¹² of each value, the low half dropped) moves the
//! logits by ~3e-4 and lies outside it.
//!
//! What each clause holds red: a bias distance one past the step's, the query's low half dropped
//! (plain f16 queries miss [`PP_BAND`] on every pass), a window cut one key short or the sink left
//! out turn `prefill_attn`; any change to the controller's order of rounding turns `control_rows`;
//! SwiGLU in SiTU's place in the route block turns `moe_rows`.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_pp", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, RouteIo};
    use bloomery_gpu::flash_gqa::window_segments;
    use bloomery_gpu::gemm::{Gemm32Kernels, GemmAct32, GemmKernels, GemmRoute};
    use bloomery_gpu::kquant::Act;
    use bloomery_gpu::loop_control::{
        LoopControlArgs, LoopControlKernels, MAX_MODULES, MAX_N, MAX_RANK, OUT_COLS, scratch_len,
    };
    use bloomery_gpu::rel_bias::{REL, RelBiasArgs, RelBiasKernels};
    use bloomery_gpu::rel_bias_prefill::{RelBiasPrefillKernels, RelPrefillArgs};
    use bloomery_gpu::weights::Weights;
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::{GateError, GateRng, bits_equal, checks_failed, max_rel_err, verdict};
    use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
    use gguf::Split;
    use gguf::quant::{GgmlType, f32_to_f16_bits};
    use model::arch::inkling2::hparams::Hparams;
    use model::placement::CardFormat;
    use model::placement::accepts::LegArms;
    use model::placement::kernels::DownAct;

    /// The Mini's attention shape: heads of 128 values, three query heads a key head.
    const HEAD: usize = 128;
    const N_HEAD: usize = 24;
    const N_KV: usize = 8;
    /// The windowed visits' window and the bias table's extent.
    const WINDOW: usize = 1024;
    const EXTENT: usize = 1024;
    /// [derived] A prompt row against the step's row (module doc).
    const PP_BAND: f32 = 5e-5;
    /// A sink this far above every logit dominates the denominator.
    const SINK_OVER: f32 = 30.0;
    /// The seed of every seeded draw.
    const SEED: u64 = 0x496e_6b32_7070_6174;

    fn scale() -> f32 {
        1.0 / (HEAD as f32).sqrt()
    }

    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{} {name}: {detail}", verdict(pass));
        pass
    }

    fn banded(name: &str, band: f32, y: &[f32], want: &[f32]) -> bool {
        match max_rel_err(y, want) {
            Ok(rel) => clause(
                name,
                rel <= band,
                &format!("max_rel {rel:.3e} (band {band:.0e})"),
            ),
            Err(e) => clause(name, false, &e.to_string()),
        }
    }

    fn refused(name: &str, r: Result<(), GpuError>) -> bool {
        clause(
            &format!("refusal {name}"),
            matches!(r, Err(GpuError::Shape { .. })),
            &format!("{r:?}"),
        )
    }

    /// The card, its stream and the modules.
    struct Dev {
        gpu: Gpu,
        rel: RelBiasKernels,
        pp: RelBiasPrefillKernels,
        control: LoopControlKernels,
    }

    impl Dev {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        fn up<T: DeviceCopy>(&self, v: &[T]) -> Result<DeviceBuffer<T>, GateError> {
            Ok(DeviceBuffer::from_host(self.stream(), v)?)
        }

        fn read<T: DeviceCopy>(&self, b: &DeviceBuffer<T>) -> Result<Vec<T>, GateError> {
            self.stream().synchronize()?;
            Ok(b.to_host_vec(self.stream())?)
        }

        fn no_fault(&self, name: &str) -> Result<bool, GateError> {
            let f = self.gpu.fault()?;
            Ok(clause(
                &format!("{name} no_fault"),
                f.is_none(),
                &format!("{f:?}"),
            ))
        }
    }

    // -------------------------------------------------- the attention's inputs

    /// One pass's seeded inputs at `n_head` heads: the query rows, `r`, the sinks and the live
    /// counts.
    struct Pass {
        q: Vec<f32>,
        r: Vec<f32>,
        sinks: Vec<f32>,
        n_keys: Vec<u32>,
    }

    impl Pass {
        fn m(&self) -> usize {
            self.n_keys.len()
        }
    }

    /// The bias table: seeded columns, the last column far larger (a distance read one column
    /// off moves the score), the column count the extent.
    fn seeded_bias(rng: &mut GateRng) -> Vec<f32> {
        let mut bias = vec![0.0f32; REL * EXTENT];
        for d in 0..REL {
            for i in 0..EXTENT {
                bias[d * EXTENT + i] = rng.uni(-1.0, 1.0) * (1.0 + d as f32) / 4.0;
            }
            bias[d * EXTENT + EXTENT - 1] = 12.0 * (1.0 + d as f32) / REL as f32;
        }
        bias
    }

    /// A pass of rows at the live counts `n_keys`: a fourth of the heads an all-zero query (the
    /// score is the bias alone), the rest at three magnitudes; the sinks cycling `−inf`, 0, 3 and
    /// [`SINK_OVER`].
    fn seeded_pass(rng: &mut GateRng, n_keys: Vec<u32>, n_head: usize) -> Pass {
        let m = n_keys.len();
        let head_scales = [0.0f32, 0.6, 1.5, 3.0];
        let sink_kinds = [f32::NEG_INFINITY, 0.0, 3.0, SINK_OVER];
        let mut q = vec![0.0f32; m * n_head * HEAD];
        for (i, row) in q.chunks_mut(HEAD).enumerate() {
            let sc = head_scales[i % 4];
            for x in row.iter_mut() {
                *x = rng.uni(-1.0, 1.0) * sc;
            }
        }
        let r = (0..m * n_head * REL).map(|_| rng.uni(-1.0, 1.0)).collect();
        let sinks = (0..m * n_head).map(|i| sink_kinds[(i / 3) % 4]).collect();
        Pass {
            q,
            r,
            sinks,
            n_keys,
        }
    }

    /// The step's launch over the plane: `rel_bias::enqueue` at `m` rows.
    fn step_launch(
        dev: &Dev,
        p: &Pass,
        cache: &DeviceBuffer<u16>,
        bias: &DeviceBuffer<f32>,
        window: usize,
        ctx: usize,
    ) -> Result<Vec<f32>, GateError> {
        let s = dev.stream();
        let m = p.m();
        let segs = window_segments(window);
        let (q, nk, r, sinks) = (
            dev.up(&p.q)?,
            dev.up(&p.n_keys)?,
            dev.up(&p.r)?,
            dev.up(&p.sinks)?,
        );
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, m * N_HEAD * segs * HEAD)?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, m * N_HEAD * segs * 2)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, m * N_HEAD * HEAD)?;
        dev.rel.enqueue(
            s,
            RelBiasArgs {
                q: &q,
                kc: cache,
                vc: cache,
                n_keys: &nk,
                r: &r,
                bias,
                extent: EXTENT,
                scale: scale(),
                n_kv: N_KV,
                ctx,
                ring_rows: None,
                m,
                window,
                sinks: &sinks,
                part_v: &mut part_v,
                part_ms: &mut part_ms,
                fault: dev.gpu.unlabelled_sink(),
                y: &mut y,
            },
            N_HEAD,
        )?;
        dev.read(&y)
    }

    /// The prompt launch: `rel_bias_prefill::enqueue` at `n_head` heads over `n_kv` key heads.
    #[allow(clippy::too_many_arguments, reason = "one launch helper's flat setup")]
    fn pp_launch(
        dev: &Dev,
        p: &Pass,
        (cache, n_kv, n_head): (&DeviceBuffer<u16>, usize, usize),
        bias: &DeviceBuffer<f32>,
        (window, extent): (usize, usize),
        ctx: usize,
        ring_rows: Option<usize>,
    ) -> Result<Vec<f32>, GateError> {
        let s = dev.stream();
        let m = p.m();
        let (q, nk, r, sinks) = (
            dev.up(&p.q)?,
            dev.up(&p.n_keys)?,
            dev.up(&p.r)?,
            dev.up(&p.sinks)?,
        );
        let mut y = DeviceBuffer::<f32>::zeroed(s, m * n_head * HEAD)?;
        dev.pp.enqueue(
            s,
            RelPrefillArgs {
                q: &q,
                kc: cache,
                vc: cache,
                n_keys: &nk,
                r: &r,
                bias,
                extent,
                scale: scale(),
                n_kv,
                ctx,
                ring_rows,
                m,
                window,
                sinks: &sinks,
                fault: dev.gpu.unlabelled_sink(),
                y: &mut y,
            },
            n_head,
        )?;
        dev.read(&y)
    }

    fn seeded_cache(rng: &mut GateRng, n_kv: usize, rows: usize) -> Vec<u16> {
        (0..n_kv * rows * HEAD)
            .map(|_| f32_to_f16_bits(rng.uni(-1.0, 1.0)))
            .collect()
    }

    fn attention_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        const CTX: usize = 2200;
        let host_cache = seeded_cache(rng, N_KV, CTX);
        let cache = dev.up(&host_cache)?;
        let bias = dev.up(&seeded_bias(rng))?;
        let mut ok = Vec::new();
        let passes: [(&str, usize, Vec<u32>); 4] = [
            ("windowed deep", WINDOW, (1990..=2066).collect()),
            ("windowed turnover", WINDOW, (990..=1066).collect()),
            ("windowed head", WINDOW, (1..=40).collect()),
            ("global past the extent", 0, (1061..=1100).collect()),
        ];
        let mut deep = None;
        for (what, window, counts) in passes {
            let p = seeded_pass(rng, counts, N_HEAD);
            let want = step_launch(dev, &p, &cache, &bias, window, CTX)?;
            let got = pp_launch(
                dev,
                &p,
                (&cache, N_KV, N_HEAD),
                &bias,
                (window, EXTENT),
                CTX,
                None,
            )?;
            ok.push(banded(
                &format!("prefill_attn {what} vs the step"),
                PP_BAND,
                &got,
                &want,
            ));
            let again = pp_launch(
                dev,
                &p,
                (&cache, N_KV, N_HEAD),
                &bias,
                (window, EXTENT),
                CTX,
                None,
            )?;
            ok.push(clause(
                &format!("prefill_attn {what} rerun"),
                bits_equal(&got, &again),
                "bit-identical",
            ));
            ok.push(dev.no_fault(&format!("prefill_attn {what}"))?);
            if deep.is_none() {
                deep = Some((p, got));
            }
        }

        // The ring: the window plus the pass's rows and three slots more, each key the rows read
        // at its slot and the three spare slots NaN.
        let (p, plane_bits) = deep.ok_or("the deep pass ran first")?;
        let ring_rows = WINDOW + p.m() - 1 + 3;
        let hi = *p.n_keys.iter().max().ok_or("a pass has rows")? as usize;
        let lo = *p.n_keys.iter().min().ok_or("a pass has rows")? as usize - WINDOW;
        let mut ring = vec![0x7e00u16; N_KV * ring_rows * HEAD];
        for kh in 0..N_KV {
            for key in lo..hi {
                let src = (kh * CTX + key) * HEAD;
                let dst = (kh * ring_rows + key % ring_rows) * HEAD;
                ring[dst..dst + HEAD].copy_from_slice(&host_cache[src..src + HEAD]);
            }
        }
        let ring = dev.up(&ring)?;
        let got = pp_launch(
            dev,
            &p,
            (&ring, N_KV, N_HEAD),
            &bias,
            (WINDOW, EXTENT),
            CTX,
            Some(ring_rows),
        )?;
        ok.push(clause(
            "prefill_ring",
            bits_equal(&got, &plane_bits),
            &format!("a ring of {ring_rows} rows (NaN outside the read keys): the plane's bits"),
        ));

        // The group of six over the eight planes = the group of three over each plane twice.
        let p6 = seeded_pass(rng, (1000..=1040).collect(), 2 * N_HEAD);
        let g6 = pp_launch(
            dev,
            &p6,
            (&cache, N_KV, 2 * N_HEAD),
            &bias,
            (WINDOW, EXTENT),
            CTX,
            None,
        )?;
        let mut twice = Vec::with_capacity(2 * host_cache.len());
        for kh in 0..N_KV {
            let plane = &host_cache[kh * CTX * HEAD..(kh + 1) * CTX * HEAD];
            twice.extend_from_slice(plane);
            twice.extend_from_slice(plane);
        }
        let twice = dev.up(&twice)?;
        let g3 = pp_launch(
            dev,
            &p6,
            (&twice, 2 * N_KV, 2 * N_HEAD),
            &bias,
            (WINDOW, EXTENT),
            CTX,
            None,
        )?;
        ok.push(clause(
            "prefill_group6",
            bits_equal(&g6, &g3),
            "48 heads over 8 = the group of three over the 8 planes each twice, bit for bit",
        ));
        ok.push(dev.no_fault("prefill_group6")?);

        // A live count past ctx: named, NaN rows.
        let bad = seeded_pass(rng, vec![CTX as u32 + 1, 5], N_HEAD);
        let y = pp_launch(
            dev,
            &bad,
            (&cache, N_KV, N_HEAD),
            &bias,
            (WINDOW, EXTENT),
            CTX,
            None,
        )?;
        let f = dev.gpu.fault()?;
        let first_nan = y[..N_HEAD * HEAD].iter().all(|v| v.is_nan());
        let second_fin = y[N_HEAD * HEAD..].iter().all(|v| v.is_finite());
        ok.push(clause(
            "prefill_faults key count",
            f.is_some() && first_nan && second_fin,
            &format!("{f:?}, the refused row NaN, the other finite"),
        ));
        dev.gpu.clear_fault()?;

        // Refusals by name, nothing enqueued.
        let p = seeded_pass(rng, vec![10, 11], N_HEAD);
        let s = dev.stream();
        let (q, nk, r, sinks) = (
            dev.up(&p.q)?,
            dev.up(&p.n_keys)?,
            dev.up(&p.r)?,
            dev.up(&p.sinks)?,
        );
        let mut y = DeviceBuffer::<f32>::zeroed(s, 2 * N_HEAD * HEAD)?;
        let args = |n_kv: usize,
                    extent: usize,
                    ring_rows: Option<usize>,
                    m: usize,
                    y: &mut DeviceBuffer<f32>|
         -> Result<(), GpuError> {
            dev.pp.enqueue(
                s,
                RelPrefillArgs {
                    q: &q,
                    kc: &cache,
                    vc: &cache,
                    n_keys: &nk,
                    r: &r,
                    bias: &bias,
                    extent,
                    scale: scale(),
                    n_kv,
                    ctx: CTX,
                    ring_rows,
                    m,
                    window: WINDOW,
                    sinks: &sinks,
                    fault: dev.gpu.unlabelled_sink(),
                    y,
                },
                N_HEAD,
            )
        };
        ok.push(refused("group of four", args(6, EXTENT, None, 2, &mut y)));
        ok.push(refused(
            "ring under the window and the pass",
            args(N_KV, EXTENT, Some(WINDOW), 2, &mut y),
        ));
        ok.push(refused("zero extent", args(N_KV, 0, None, 2, &mut y)));
        let mut short = DeviceBuffer::<f32>::zeroed(s, N_HEAD * HEAD)?;
        ok.push(refused(
            "short output",
            args(N_KV, EXTENT, None, 2, &mut short),
        ));
        Ok(ok)
    }

    // -------------------------------------------------- the loop controller over rows

    /// The Mini's controller width and rank, and its norm's epsilon.
    const CTRL_N: usize = 2304;
    const CTRL_RANK: usize = 64;
    const CTRL_EPS: f32 = 1e-6;

    /// The three modules' seeded inputs at `(n, rank, m)`: rows at five magnitudes, so the two
    /// norms differ row to row; with `nan_at`, module 0's row there carries a NaN.
    struct Control {
        cur: Vec<DeviceBuffer<f32>>,
        msg: Vec<DeviceBuffer<f32>>,
        down: Vec<DeviceBuffer<f32>>,
        gain: Vec<DeviceBuffer<f32>>,
        forget: Vec<DeviceBuffer<f32>>,
        shape: (usize, usize, usize),
    }

    fn seeded_control(
        dev: &Dev,
        rng: &mut GateRng,
        (n, rank, m): (usize, usize, usize),
        nan_at: Option<usize>,
    ) -> Result<Control, GateError> {
        let mut c = Control {
            cur: Vec::new(),
            msg: Vec::new(),
            down: Vec::new(),
            gain: Vec::new(),
            forget: Vec::new(),
            shape: (n, rank, m),
        };
        for mi in 0..MAX_MODULES {
            let rows = |rng: &mut GateRng| -> Vec<f32> {
                (0..m * n)
                    .map(|i| rng.uni(-1.0, 1.0) * (0.25 + ((i / n) % 5) as f32))
                    .collect()
            };
            let mut cur = rows(rng);
            if let (0, Some(t)) = (mi, nan_at) {
                cur[t * n + n / 2] = f32::NAN;
            }
            let msg = rows(rng);
            let mut w =
                |len: usize, s: f32| -> Vec<f32> { (0..len).map(|_| rng.uni(-s, s)).collect() };
            let down = w(rank * 2 * n, 1.0 / (n as f32).sqrt());
            let gain = w(n * rank, 1.0 / (rank as f32).sqrt());
            let forget = w(n * rank, 1.0 / (rank as f32).sqrt());
            c.cur.push(dev.up(&cur)?);
            c.msg.push(dev.up(&msg)?);
            c.down.push(dev.up(&down)?);
            c.gain.push(dev.up(&gain)?);
            c.forget.push(dev.up(&forget)?);
        }
        Ok(c)
    }

    fn control_args<'a>(dev: &Dev, c: &'a Control, mi: usize) -> LoopControlArgs<'a> {
        let (n, rank, m) = c.shape;
        LoopControlArgs {
            cur: &c.cur[mi],
            msg: &c.msg[mi],
            down: &c.down[mi],
            gain: &c.gain[mi],
            forget: &c.forget[mi],
            n,
            rank,
            m,
            eps: CTRL_EPS,
            fault: dev.gpu.unlabelled_sink(),
        }
    }

    /// One module's step pair and rows pair on `c`: the scratch and the output of each.
    fn control_one(dev: &Dev, c: &Control) -> Result<[Vec<f32>; 4], GateError> {
        let s = dev.stream();
        let (n, rank, m) = c.shape;
        let mut out = Vec::new();
        for rows in [false, true] {
            let mut h = DeviceBuffer::<f32>::zeroed(s, scratch_len(1, m, rank))?;
            let mut y = DeviceBuffer::<f32>::zeroed(s, m * n)?;
            if rows {
                dev.control
                    .enqueue_rows(s, control_args(dev, c, 0), &mut h, &mut y)?;
            } else {
                dev.control
                    .enqueue(s, control_args(dev, c, 0), &mut h, &mut y)?;
            }
            out.push(dev.read(&h)?);
            out.push(dev.read(&y)?);
        }
        let [h0, y0, h1, y1]: [Vec<f32>; 4] = out.try_into().map_err(|_| "four buffers read")?;
        Ok([h0, y0, h1, y1])
    }

    /// The three attention modules' step and rows launches on `c`: the scratch and the three
    /// outputs, step first.
    fn control_three(dev: &Dev, c: &Control) -> Result<[Vec<Vec<f32>>; 2], GateError> {
        let s = dev.stream();
        let (n, rank, m) = c.shape;
        let mut out = Vec::new();
        for rows in [false, true] {
            let mut h = DeviceBuffer::<f32>::zeroed(s, scratch_len(MAX_MODULES, m, rank))?;
            let (mut y0, mut y1, mut y2) = (
                DeviceBuffer::<f32>::zeroed(s, m * n)?,
                DeviceBuffer::<f32>::zeroed(s, m * n)?,
                DeviceBuffer::<f32>::zeroed(s, m * n)?,
            );
            let a = [0, 1, 2].map(|mi| control_args(dev, c, mi));
            if rows {
                dev.control
                    .enqueue_attn_rows(s, a, &mut h, [&mut y0, &mut y1, &mut y2])?;
            } else {
                dev.control
                    .enqueue_attn(s, a, &mut h, [&mut y0, &mut y1, &mut y2])?;
            }
            out.push(vec![
                dev.read(&h)?,
                dev.read(&y0)?,
                dev.read(&y1)?,
                dev.read(&y2)?,
            ]);
        }
        let [step, rows]: [Vec<Vec<f32>>; 2] = out.try_into().map_err(|_| "two arms read")?;
        Ok([step, rows])
    }

    /// 5. `control_rows`: the rows form (`enqueue_rows`) equals the step pair (`enqueue`) bit
    ///    for bit — the scratch and the output — at a prompt ubatch of 512 rows at the Mini's
    ///    shape, a ragged pass of 37, a width off the output group and a rank off the warp's
    ///    eight, one output at rank 1 over 33 rows, the widest staging over 9.
    /// 6. `control_attn_rows`: the three attention modules' rows form (`enqueue_attn_rows`)
    ///    equals `enqueue_attn` bit for bit at 100 rows.
    /// 7. `control_rows_nan`: a NaN in one row: that row's output NaN in both forms, the fault
    ///    raised by both, every other row the step's bits.
    fn control_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        let mut ok = Vec::new();
        for shape in [
            (CTRL_N, CTRL_RANK, 512usize),
            (CTRL_N, CTRL_RANK, 37),
            (200, 7, 3),
            (OUT_COLS + 1, 1, 33),
            (MAX_N, MAX_RANK, 9),
            (CTRL_N, 63, 40),
        ] {
            let c = seeded_control(dev, rng, shape, None)?;
            let [h0, y0, h1, y1] = control_one(dev, &c)?;
            let (n, rank, m) = shape;
            ok.push(clause(
                &format!("control_rows {n} wide at rank {rank}, {m} rows"),
                bits_equal(&h0, &h1) && bits_equal(&y0, &y1),
                "the scratch and the output = the step's bits",
            ));
            ok.push(dev.no_fault(&format!("control_rows {n}x{rank}x{m}"))?);
        }
        let c = seeded_control(dev, rng, (CTRL_N, CTRL_RANK, 100), None)?;
        let [step, rows] = control_three(dev, &c)?;
        let same = step.iter().zip(&rows).all(|(a, b)| bits_equal(a, b));
        ok.push(clause(
            "control_attn_rows three modules, 100 rows",
            same,
            "the scratch and the three outputs = enqueue_attn's bits",
        ));
        ok.push(dev.no_fault("control_attn_rows")?);

        let nan_row = 5;
        let c = seeded_control(dev, rng, (CTRL_N, CTRL_RANK, 12), Some(nan_row))?;
        let s = dev.stream();
        let (n, rank, m) = c.shape;
        let mut faults = Vec::new();
        let mut ys = Vec::new();
        for rows in [false, true] {
            let mut h = DeviceBuffer::<f32>::zeroed(s, scratch_len(1, m, rank))?;
            let mut y = DeviceBuffer::<f32>::zeroed(s, m * n)?;
            if rows {
                dev.control
                    .enqueue_rows(s, control_args(dev, &c, 0), &mut h, &mut y)?;
            } else {
                dev.control
                    .enqueue(s, control_args(dev, &c, 0), &mut h, &mut y)?;
            }
            ys.push(dev.read(&y)?);
            faults.push(dev.gpu.fault()?);
            dev.gpu.clear_fault()?;
        }
        let row = nan_row * n..(nan_row + 1) * n;
        let nan_both = ys.iter().all(|y| y[row.clone()].iter().all(|v| v.is_nan()));
        let rest_same = bits_equal(&ys[0][..row.start], &ys[1][..row.start])
            && bits_equal(&ys[0][row.end..], &ys[1][row.end..]);
        ok.push(clause(
            "control_rows_nan",
            nan_both && rest_same && faults.iter().all(Option::is_some),
            &format!(
                "row {nan_row} NaN in both forms, the others the step's bits; faults {faults:?}"
            ),
        ));
        Ok(ok)
    }

    // -------------------------------------------------- the prompt MoE over rows

    /// Rows of the MoE pass: a prompt ubatch.
    const MOE_ROWS: usize = 512;
    /// [derived] The prompt route block's down rows against the step leg's, rel-L2 over every
    /// slot (module doc, clause 8).
    const MOE_BAND: f64 = 3e-2;
    const RECIPE: &str = "gate-inkling2-pp";

    /// The leg's gate·up and down arms: the family's NVFP4 planes and Q8_0 blocks.
    fn moe_arm(ty: GgmlType) -> bool {
        matches!(ty, GgmlType::NVFP4 | GgmlType::Q8_0)
    }
    const GATE_UP_ROLE: &str = "a resident fused-w13 half of NVFP4 planes or Q8_0 blocks";
    const DOWN_ROLE: &str = "a resident down stack of NVFP4 planes or Q8_0 blocks";

    fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&g, &w) in got.iter().zip(want) {
            num += (f64::from(g) - f64::from(w)).powi(2);
            den += f64::from(w).powi(2);
        }
        if got.len() != want.len() || !num.is_finite() || den == 0.0 {
            return f64::INFINITY;
        }
        (num / den).sqrt()
    }

    /// The first stored layer whose fused w13 stack is NVFP4 and the first whose is Q8_0.
    fn moe_layers(split: &Split, hp: &Hparams) -> Result<[usize; 2], GateError> {
        let (mut nvfp4, mut q8) = (None, None);
        for l in 0..hp.n_layer {
            let name = format!("blk.{l}.ffn_gate_up_exps.weight");
            let (_, tt) = split
                .find(&name)
                .ok_or_else(|| format!("{name} is not in the model file"))?;
            match tt.ty {
                GgmlType::NVFP4 if nvfp4.is_none() => nvfp4 = Some(l),
                GgmlType::Q8_0 if q8.is_none() => q8 = Some(l),
                _ => {}
            }
        }
        match (nvfp4, q8) {
            (Some(a), Some(b)) => Ok([a, b]),
            _ => Err("the model file holds no NVFP4 and Q8_0 pair of expert sets".into()),
        }
    }

    /// 8. `moe_rows`: the prompt route block (`card_leg::RouteBlock::enqueue_rule`: the route
    ///    table, the 32-value gate and up GEMMs over the NVFP4 planes with their per-expert
    ///    scales or the Q8_0 blocks, SiTU at the file's betas into the down's q8 blocks, the
    ///    down GEMM) over 512 seeded rows and seeded picks, against the step's leg (`CardLeg`:
    ///    the `_sel` launches) on the same rows: every slot's down row within [`MOE_BAND`] by
    ///    rel-L2, on an NVFP4 layer and the Q8_0 coda alike, the file's real expert stacks.
    /// 9. `moe_leg_rows`: the leg over the 512 rows equals the leg over one row at the first
    ///    and the last row, bit for bit: the step's leg is the rows' own reference.
    ///
    /// [derived] [`MOE_BAND`]: the two paths read the same weight blocks but quantize the input
    /// rows onto two grids — the leg's gate·up reads q8_1 at one scale a 128 values (`Q8Act`,
    /// `enqueue_quantize_q8_1_cols`), the route block's 32-value GEMMs one a 32 (`GemmAct32`) —
    /// so the gate·up sums carry two independent roundings of the same rows: each
    /// `(amax/rms)/(127·√12)` ≈ 1.7/440 ≈ 3.9e-3 of a seeded uniform row, together √2 · 3.9e-3
    /// ≈ 5.5e-3 of each sum, and the SiTU rows (the gate's and the up's errors independent,
    /// the tanh·sigmoid damping the gate's) 5.5e-3 to 7.8e-3 (the detail prints this stage).
    /// The down's inputs are both q8 blocks of 32, but of rows already apart by that much, so
    /// their roundings add from nothing to √2 · 6.8e-3 (SiTU rows' amax/rms ≈ 3): the down rows
    /// 7.8e-3 to 1.24e-2 by rel-L2. 3e-2 is 2.5 times the top. SiTU equals SwiGLU where both caps
    /// are loose (`|g| ≪ beta`, `|u| ≪ linear_beta`), so SwiGLU in its place moves the rows only
    /// where the gate sums reach the cap: past the band at these rows, not by order 1; the wrong
    /// expert's rows move them by order 1. The f32 order of the block sums (the gemv's lane fold
    /// against the GEMM's k walk, ~√72 · 2⁻²⁴) is three orders under it.
    fn moe_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        let path = bloomery_gpu_gates::ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        bloomery_gpu_gates::expect_arch(&split, model::arch::Arch::Inkling2, RECIPE)?;
        let hp = Hparams::read(&split).map_err(|e| format!("{}: {e}", path.display()))?;
        let s = dev.stream();
        let gpu = &dev.gpu;
        let leg = CardLeg::load(gpu)?;
        let gemm = GemmKernels::load(gpu.context())?;
        let g32 = Gemm32Kernels::load(gpu.context())?;
        let rule = Act::Situ {
            beta: hp.situ.gate_beta,
            linear_beta: hp.situ.value_beta,
        };
        let (embd, n_expert, n_used) = (hp.n_embd, hp.n_expert, hp.n_used[0]);
        let t = MOE_ROWS;
        let slots = t * n_used;
        let x_host: Vec<f32> = (0..t * embd)
            .map(|i| rng.uni(-1.0, 1.0) * (0.5 + ((i / embd) % 3) as f32))
            .collect();
        let mut ids_host = Vec::with_capacity(slots);
        for _ in 0..t {
            let mut pool: Vec<u32> = (0..n_expert as u32).collect();
            for k in 0..n_used {
                let j = k + (rng.unit() * (n_expert - k) as f32) as usize % (n_expert - k);
                pool.swap(k, j);
                ids_host.push(pool[k]);
            }
        }
        let (x, ids) = (dev.up(&x_host)?, dev.up(&ids_host)?);
        let mut ok = Vec::new();
        for l in moe_layers(&split, &hp)? {
            let ff = hp.expert_ff[l];
            let (fused, down_name) = (
                format!("blk.{l}.ffn_gate_up_exps.weight"),
                format!("blk.{l}.ffn_down_exps.weight"),
            );
            let scaled = matches!(
                split.find(&fused).map(|(_, tt)| tt.ty),
                Some(GgmlType::NVFP4)
            );
            let mut w = Weights::load(s, &split, l..l + 1, false)?;
            if !scaled {
                // The Q8_0 entries of both paths read the file's blocks.
                w.relayout(s, &split, &fused, CardFormat::KQuant)?;
                w.relayout(s, &split, &down_name, CardFormat::KQuant)?;
            }
            let (gate, up) = w.split_fused_w13(s, &fused, ff)?;
            let stacks = LegStacks::of_arms(
                &w,
                [gate.as_str(), up.as_str(), down_name.as_str()],
                LegArms {
                    gate_up: moe_arm,
                    down: moe_arm,
                },
                (GATE_UP_ROLE, DOWN_ROLE),
            )?;
            let fault = gpu.layer_sink(l)?;

            // The step's leg over the rows, and over one row at the first and the last.
            let leg_rows = |m: usize, x: &DeviceBuffer<f32>, sel: &DeviceBuffer<u32>| {
                let shape = LegShape {
                    cap: m,
                    slots_per_col: n_used,
                    n: embd,
                    ff,
                };
                let mut bufs = LegBufs::for_down(s, shape, DownAct::Q8Blocks32)?;
                let mut down = DeviceBuffer::<f32>::zeroed(s, m * n_used * embd)?;
                leg.enqueue_rows(
                    gpu,
                    &mut bufs,
                    &LegRun {
                        layer: l,
                        stacks: &stacks,
                        normed: x,
                        m,
                        sel,
                        n_card: n_expert,
                        rule,
                    },
                    &mut down,
                )?;
                Ok::<_, GateError>((dev.read(bufs.act_rows())?, dev.read(&down)?))
            };
            let (leg_situ, leg_all) = leg_rows(t, &x, &ids)?;
            let mut one_same = true;
            for row in [0, t - 1] {
                let x1 = dev.up(&x_host[row * embd..(row + 1) * embd])?;
                let ids1 = dev.up(&ids_host[row * n_used..(row + 1) * n_used])?;
                let (_, one) = leg_rows(1, &x1, &ids1)?;
                let span = row * n_used * embd..(row + 1) * n_used * embd;
                one_same &= bits_equal(&one, &leg_all[span]);
            }

            // The prompt route block over the same rows.
            let block = stacks.block(ff, embd)?;
            let mut route = GemmRoute::new(s, slots, n_expert)?;
            gemm.enqueue_route(s, &ids, slots, &mut route, fault)?;
            let mut x32 = GemmAct32::new(s, t, embd)?;
            g32.enqueue_quantize_gemm32(s, &x, t, &mut x32, fault)?;
            let mut g = DeviceBuffer::<f32>::zeroed(s, slots * ff)?;
            let mut u = DeviceBuffer::<f32>::zeroed(s, slots * ff)?;
            let mut h32 = GemmAct32::new(s, slots, ff)?;
            let mut y = DeviceBuffer::<f32>::zeroed(s, slots * embd)?;
            block.enqueue_rule(
                gpu,
                (None, &g32),
                (&route, n_used),
                slots,
                (
                    RouteIo {
                        x128: None,
                        x32: Some(&x32),
                        g: &mut g,
                        u: &mut u,
                        h128: None,
                        h32: Some(&mut h32),
                        y: &mut y,
                    },
                    rule,
                ),
                fault,
            )?;
            let got = dev.read(&y)?;
            let rel = rel_l2(&got, &leg_all);
            // The stage split: the route block's gate and up sums through the host SiTU rule
            // against the leg's SiTU rows (the gate·up stage alone).
            let (gh, uh) = (dev.read(&g)?, dev.read(&u)?);
            let Act::Situ { beta, linear_beta } = rule else {
                return Err("the family's rule is SiTU".into());
            };
            let situ_host: Vec<f32> = gh
                .iter()
                .zip(&uh)
                .map(|(&a, &b)| bloomery_gpu::kquant::act::situ(a, b, beta, linear_beta))
                .collect();
            let situ_rel = rel_l2(&situ_host, &leg_situ);
            let worst = (0..slots)
                .map(|sl| {
                    let span = sl * embd..(sl + 1) * embd;
                    rel_l2(&got[span.clone()], &leg_all[span])
                })
                .fold(0.0f64, f64::max);
            let quant = if scaled { "nvfp4" } else { "q8_0" };
            ok.push(clause(
                &format!("moe_rows l={l} ({quant}) route block vs the step leg"),
                rel <= MOE_BAND,
                &format!(
                    "{t} rows x {n_used} slots, ff {ff}: rel-L2 {rel:.3e} (band {MOE_BAND:.0e}), \
                     the worst slot {worst:.3e}; the gate·up stage (SiTU rows) {situ_rel:.3e}"
                ),
            ));
            ok.push(clause(
                &format!("moe_leg_rows l={l} ({quant})"),
                one_same,
                "the leg over 512 rows = the leg over one row at rows 0 and 511, bit for bit",
            ));
            ok.push(dev.no_fault(&format!("moe l={l}"))?);
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        let dev = Dev {
            rel: RelBiasKernels::load(ctx)?,
            pp: RelBiasPrefillKernels::load(ctx)?,
            control: LoopControlKernels::load(ctx)?,
            gpu,
        };
        println!(
            "gate_inkling2_pp: device {} — heads {N_HEAD}/{N_KV} x {HEAD}, window {WINDOW}, \
             extent {EXTENT}",
            dev.gpu.device_name()?
        );
        let mut rng = GateRng(SEED);
        let mut results = Vec::new();
        results.extend(attention_clauses(&dev, &mut rng)?);
        results.extend(control_clauses(&dev, &mut rng)?);
        results.extend(moe_clauses(&dev, &mut rng)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "gate_inkling2_pp: {} — {} clauses, {failed} failed",
            verdict(failed == 0),
            results.len()
        );
        if failed == 0 {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
