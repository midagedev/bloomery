//! GPU gate for Hy4-flash's latent attention on the card (design row 11, the kernel N1) and its
//! output gate at 256 values a head (N6), seeded inputs at the released file's attention shapes;
//! nothing is read from a model file. A DSA layer runs 32 heads over the rows its indexer keeps (all
//! of them below 2,048 keys), an SWA layer 64 heads over a 512-row window ring; every head has a
//! learned sink. The cache row is `[rope 64 | latent 512]` f16, the key, its latent the value; the
//! scale is `1/16` (`1/√(192 + 64)`).
//!
//! The reference is llama.cpp's `build_attention` in `src/models/hy-v4-flash.cpp` at `0a054d0b`:
//! a softmax over the token's keys and the head's sink logit (ggml's soft-max with sinks: the sink
//! unscaled, no value), in f64 over the same f16 query rows the tensor-core pass stages, so [`BAND`]
//! holds only the card's f32 sums and `ex2.approx`. Rows the launch must not read hold the f16 NaN
//! pattern: a stray read fails the finiteness scan of `max_rel_err`.
//!
//! 1. `dense`: `enqueue_flash_latent_sink` at 32 heads, one token at counts 1, 31, 32, 33, 64, 65,
//!    1000 and 2048 and four tokens at 1000, over a 2,112-row cache, within [`BAND`] of f64; a
//!    rerun bit-identical; the selected path over the ascending list `0 .. count` (stride 2,048)
//!    the dense path's bits, so either serves a DSA layer below 2,048 keys.
//! 2. `selected`: `enqueue_flash_latent_sel_sink` at 32 heads, three tokens over lists of 2,048
//!    distinct rows of a 4,096-row cache in a seeded order, counts 2048, 1 and 1000, and at the
//!    fixture's list of 64 (counts 64, 1 and 33): within [`BAND`] of f64, no fault raised.
//! 3. `window`: `enqueue_flash_window` at 64 heads over a 512-row ring at positions 0, 1, 63, 64,
//!    65, 511, 512, 513, 1000 and 1535, and over the fixture's 64-row ring at 0, 1, 63, 64, 65,
//!    100 and 200, the ring holding each slot's latest position: within [`BAND`] of f64 over
//!    positions `max(0, p − W + 1) ..= p`; one graph captured at the first position and replayed
//!    at every position, the position word rewritten, each replay the eager launch's bits.
//! 4. `batch`: the staged window batch at `(b, T)` = (0, 7), (505, 20) and (1000, 520) over 512
//!    rows and (0, 7), (60, 10) and (100, 70) over 64: each token's rows the one-token launch's at
//!    its position bit for bit, a sample of tokens within [`BAND`] of f64, and after
//!    `enqueue_kv_ring_commit` the ring equal to the ring at `b + T − 1`.
//! 5. `sink_exact`: with every sink `−inf` the sink merges' rows are the plain merges' bits — the
//!    dense launch at counts 1, 2 and 1000, the selected launch over the identity list, and the
//!    window at positions 0, 100 and 511 equal to the dense launch over the ring's rows (the same
//!    64-key cut) — so the fold adds no ε; sinks 30 above every logit leave the rows within
//!    [`BAND`] of f64.
//! 6. `out_gate`: `enqueue_out_gate_head` at 256 values a head under `Proj` over 3 tokens of 64
//!    heads, the host rule's bits; `ProjHead` at 256 refused; `MlaDims` at Hy4's widths passes its
//!    check and a value head of 192 is refused.
//! 7. `refusals`: a 512-wide ring, a ring of no row, fewer sinks than heads, a commit of no
//!    token and 24 heads, each refused by its shape check before anything is enqueued.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_hy4_attn", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::flash::{
        FlashGeom, FlashInputs, FlashKernels, FlashMergeArgs, FlashMergeSelArgs, FlashSegArgs,
        FlashSelArgs, FlashSplitArgs, FlashWindowArgs, RingCommitArgs, partials_ms_len,
        partials_v_len, window_partials_lens,
    };
    use bloomery_gpu::latent_gated::MlaDims;
    use bloomery_gpu::linear::out_gate::{OutGateArgs, OutGateKernels, out_gate_host};
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
    use bloomery_gpu_gates::{
        GateError, GateRng, bits_equal, checks_failed, max_rel_err, par_map, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use model::arch::models::OutGate;

    /// The rope lanes in front of the latent.
    const ROPE: usize = 64;
    /// The latent: a key's tail and its value.
    const LATENT: usize = 512;
    /// A cache row.
    const ROW: usize = ROPE + LATENT;
    /// `1/√(nope 192 + rope 64)`.
    const SCALE: f32 = 1.0 / 16.0;
    /// A DSA layer's heads and an SWA layer's.
    const DSA_HEADS: usize = 32;
    const SWA_HEADS: usize = 64;
    /// The window ring's rows (`attention.sliding_window`).
    const WINDOW: usize = 512;
    /// The selection list's width (`attention.indexer.top_k`).
    const TOP_K: usize = 2048;
    /// The fixture file's window and list (both launch arguments, not compile-time widths).
    const FIXTURE_WINDOW: usize = 64;
    const FIXTURE_TOP_K: usize = 64;
    /// The f16 NaN pattern a row the launch must not read holds.
    const NAN16: u16 = 0x7e00;
    /// [derived] Against an f64 reference over the card's own f16 query rows the error left is the
    /// f32 logit sums (≲ 1e-5 of a logit), `ex2.approx` (2^-22 a weight) and the f32 value sums
    /// over at most 2,048 keys (√2048 · 2^-24 of the summed magnitude, ~1e-6 of a row's largest
    /// value); 2e-4 is ten times the sum of the three at the largest output a launch writes.
    const BAND: f32 = 2e-4;
    /// A sink this far above every logit dominates the denominator.
    const SINK_OVER: f32 = 30.0;

    /// One launch's query rows (`[tokens][heads][ROW]` f32) and the heads' sinks.
    struct Queries {
        q: Vec<f32>,
        sinks: Vec<f32>,
        heads: usize,
    }

    impl Queries {
        /// Seeded rows: head `h` scaled by 0.6, 1.5 or 3 (flat to sharp softmaxes), its sink
        /// `−inf`, −2, 3 or 9 by `h % 4`.
        fn new(rng: &mut GateRng, tokens: usize, heads: usize) -> Queries {
            const SCALES: [f32; 3] = [0.6, 1.5, 3.0];
            const SINKS: [f32; 4] = [f32::NEG_INFINITY, -2.0, 3.0, 9.0];
            let q = (0..tokens * heads * ROW)
                .map(|i| rng.uni(-1.0, 1.0) * SCALES[(i / ROW) % heads % 3])
                .collect();
            let sinks = (0..heads).map(|h| SINKS[h % 4]).collect();
            Queries { q, sinks, heads }
        }

        fn tokens(&self) -> usize {
            self.q.len() / (self.heads * ROW)
        }
    }

    /// A seeded cache row of f16 values in `[-1, 1)`.
    fn row(rng: &mut GateRng) -> Vec<u16> {
        (0..ROW)
            .map(|_| f32_to_f16_bits(rng.uni(-1.0, 1.0)))
            .collect()
    }

    /// A cache value as f64.
    fn val(cache: &[u16], r: usize, d: usize) -> f64 {
        f64::from(half_to_f32(cache[r * ROW + d]))
    }

    /// One head's logits over `keys` (row indices of `cache`): `SCALE · q·k` in f64, the query
    /// row rounded to f16 as the tensor-core pass stages it.
    fn logits(q: &[f32], cache: &[u16], keys: &[usize]) -> Vec<f64> {
        let q16: Vec<f64> = q
            .iter()
            .map(|&v| f64::from(half_to_f32(f32_to_f16_bits(v))))
            .collect();
        keys.iter()
            .map(|&r| f64::from(SCALE) * (0..ROW).map(|d| q16[d] * val(cache, r, d)).sum::<f64>())
            .collect()
    }

    /// One head's reference rows: softmax over the [`logits`] and the sink's logit, applied to
    /// the keys' latent tails, in f64.
    fn ref_head(q: &[f32], cache: &[u16], keys: &[usize], sink: f32) -> Vec<f64> {
        let logits = logits(q, cache, keys);
        let mx = logits.iter().copied().fold(f64::from(sink), f64::max);
        let w: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
        let den = w.iter().sum::<f64>() + (f64::from(sink) - mx).exp();
        (0..LATENT)
            .map(|d| {
                keys.iter()
                    .zip(&w)
                    .map(|(&r, wj)| wj * val(cache, r, ROPE + d))
                    .sum::<f64>()
                    / den
            })
            .collect()
    }

    /// The reference rows of a launch: token `t`, head `h` over `keys(t)`.
    fn reference(
        qs: &Queries,
        cache: &[u16],
        keys: &(dyn Fn(usize) -> Vec<usize> + Sync),
    ) -> Vec<f32> {
        let h = qs.heads;
        par_map(qs.tokens() * h, |i| {
            let q = &qs.q[i * ROW..(i + 1) * ROW];
            ref_head(q, cache, &keys(i / h), qs.sinks[i % h])
        })
        .into_iter()
        .flatten()
        .map(|v| v as f32)
        .collect()
    }

    /// Print one clause's line and return whether it passed.
    fn clause(name: &str, pass: bool, detail: &str) -> bool {
        println!("{} {name}: {detail}", verdict(pass));
        pass
    }

    /// The band check of a launch's rows against their reference. A row that is not finite (a
    /// NaN row read) fails this clause by name, and the later clauses still run.
    fn banded(name: &str, y: &[f32], want: &[f32]) -> bool {
        match max_rel_err(y, want) {
            Ok(rel) => clause(
                name,
                rel <= BAND,
                &format!("max_rel {rel:.3e} (band {BAND:.0e})"),
            ),
            Err(e) => clause(name, false, &e.to_string()),
        }
    }

    /// The card, its stream and the flash module.
    struct Dev {
        gpu: Gpu,
        flash: FlashKernels,
    }

    impl Dev {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        fn up<T: DeviceCopy>(&self, v: &[T]) -> Result<DeviceBuffer<T>, GateError> {
            Ok(DeviceBuffer::from_host(self.stream(), v)?)
        }

        fn rows(&self, v: &[u16], rows: usize) -> Result<DeviceTensor<u16>, GateError> {
            Ok(DeviceTensor::upload(self.stream(), v, rows, ROW)?)
        }

        fn read(&self, b: &DeviceBuffer<f32>) -> Result<Vec<f32>, GateError> {
            self.stream().synchronize()?;
            Ok(b.to_host_vec(self.stream())?)
        }

        /// No launch raised its fault word.
        fn no_fault(&self, name: &str) -> Result<bool, GateError> {
            let f = self.gpu.fault()?;
            Ok(clause(
                &format!("{name} no_fault"),
                f.is_none(),
                &format!("{f:?}"),
            ))
        }
    }

    fn geom(tokens: usize, heads: usize) -> FlashGeom {
        FlashGeom {
            tokens,
            heads,
            rope_dims: ROPE,
            latent_dims: LATENT,
        }
    }

    /// The dense launch (`sinks: Some`) or the plain split launch (`None`) of `qs` over `cache`
    /// at live count `n_keys`, token `t` of `T` attending keys `0 .. n_keys − T + t + 1`.
    fn dense(
        dev: &Dev,
        qs: &Queries,
        cache: &DeviceTensor<u16>,
        n_keys: usize,
        sinks: Option<&[f32]>,
    ) -> Result<Vec<f32>, GateError> {
        let s = dev.stream();
        let q_rows = qs.tokens() * qs.heads;
        let q = dev.up(&qs.q)?;
        let nk = dev.up(&[n_keys as u32])?;
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, partials_v_len(q_rows, cache.rows()))?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, partials_ms_len(q_rows, cache.rows()))?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, q_rows * LATENT)?;
        let inputs = FlashInputs {
            q: &q,
            kv: cache,
            n_keys_buf: &nk,
            kq_scale: SCALE,
            geom: geom(qs.tokens(), qs.heads),
        };
        match sinks {
            Some(sk) => {
                let sk = dev.up(sk)?;
                dev.flash.enqueue_flash_latent_sink(
                    s,
                    FlashSplitArgs {
                        inputs,
                        part_v: &mut part_v,
                        part_ms: &mut part_ms,
                        y: &mut y,
                    },
                    &sk,
                )?;
                dev.read(&y)
            }
            None => {
                dev.flash.enqueue_flash_latent_mma(
                    s,
                    FlashSegArgs {
                        inputs,
                        part_v: &mut part_v,
                        part_ms: &mut part_ms,
                    },
                )?;
                dev.flash.enqueue_flash_merge(
                    s,
                    FlashMergeArgs {
                        n_keys_buf: &nk,
                        cache_rows: cache.rows(),
                        tokens: qs.tokens(),
                        heads: qs.heads,
                        latent_dims: LATENT,
                        part_v: &part_v,
                        part_ms: &part_ms,
                        y: &mut y,
                    },
                )?;
                dev.read(&y)
            }
        }
    }

    /// The selected launch with sinks (`Some`) or the plain one, token `t` over
    /// `list[t·stride ..][.. counts[t]]`.
    fn selected(
        dev: &Dev,
        qs: &Queries,
        cache: &DeviceTensor<u16>,
        list: &[u32],
        counts: &[u32],
        sinks: Option<&[f32]>,
    ) -> Result<Vec<f32>, GateError> {
        let s = dev.stream();
        let tokens = qs.tokens();
        let q_rows = tokens * qs.heads;
        let stride = list.len() / tokens;
        let (q, list, counts) = (dev.up(&qs.q)?, dev.up(list)?, dev.up(counts)?);
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, partials_v_len(q_rows, stride))?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, partials_ms_len(q_rows, stride))?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, q_rows * LATENT)?;
        let args = FlashSelArgs {
            q: &q,
            kv: cache,
            list: &list,
            counts: &counts,
            stride,
            kq_scale: SCALE,
            geom: geom(tokens, qs.heads),
            part_v: &mut part_v,
            part_ms: &mut part_ms,
            fault: dev.gpu.unlabelled_sink(),
        };
        match sinks {
            Some(sk) => {
                let sk = dev.up(sk)?;
                dev.flash
                    .enqueue_flash_latent_sel_sink(s, args, &sk, &mut y)?;
            }
            None => {
                dev.flash.enqueue_flash_latent_mma_sel(s, args)?;
                dev.flash.enqueue_flash_merge_sel(
                    s,
                    FlashMergeSelArgs {
                        counts: &counts,
                        stride,
                        tokens,
                        heads: qs.heads,
                        latent_dims: LATENT,
                        part_v: &part_v,
                        part_ms: &part_ms,
                        y: &mut y,
                    },
                )?;
            }
        }
        dev.read(&y)
    }

    /// A window launch's device buffers, kept so a graph can be captured over them.
    struct Win {
        q: DeviceBuffer<f32>,
        ring: DeviceTensor<u16>,
        stage: Option<DeviceTensor<u16>>,
        pos: DeviceBuffer<u32>,
        sinks: DeviceBuffer<f32>,
        tokens: usize,
        heads: usize,
        part_v: DeviceBuffer<f32>,
        part_ms: DeviceBuffer<f32>,
        y: DeviceBuffer<f32>,
    }

    impl Win {
        fn new(
            dev: &Dev,
            qs: &Queries,
            ring: &[u16],
            stage: Option<&[u16]>,
            b: usize,
        ) -> Result<Win, GateError> {
            let s = dev.stream();
            let tokens = qs.tokens();
            let ring_rows = ring.len() / ROW;
            let (lv, lms) = window_partials_lens(tokens * qs.heads, ring_rows);
            Ok(Win {
                q: dev.up(&qs.q)?,
                ring: dev.rows(ring, ring_rows)?,
                stage: stage.map(|st| dev.rows(st, st.len() / ROW)).transpose()?,
                pos: dev.up(&[b as u32])?,
                sinks: dev.up(&qs.sinks)?,
                tokens,
                heads: qs.heads,
                part_v: DeviceBuffer::<f32>::zeroed(s, lv)?,
                part_ms: DeviceBuffer::<f32>::zeroed(s, lms)?,
                y: DeviceBuffer::<f32>::zeroed(s, tokens * qs.heads * LATENT)?,
            })
        }

        fn enqueue(&mut self, flash: &FlashKernels, s: &CudaStream) -> Result<(), GpuError> {
            flash.enqueue_flash_window(
                s,
                FlashWindowArgs {
                    q: &self.q,
                    ring: &self.ring,
                    staged: self.stage.as_ref(),
                    pos: &self.pos,
                    sinks: &self.sinks,
                    kq_scale: SCALE,
                    geom: geom(self.tokens, self.heads),
                    part_v: &mut self.part_v,
                    part_ms: &mut self.part_ms,
                    y: &mut self.y,
                },
            )
        }
    }

    /// The positions' rows and the ring each position leaves.
    struct Positions {
        rows: Vec<Vec<u16>>,
    }

    impl Positions {
        fn new(rng: &mut GateRng, n: usize) -> Positions {
            Positions {
                rows: (0..n).map(|_| row(rng)).collect(),
            }
        }

        /// The `w`-row ring after position `last` (`None`: before position 0): slot `s` holds the
        /// latest position `≤ last` at `s mod w`, a slot no position reached the NaN pattern.
        fn ring(&self, last: Option<usize>, w: usize) -> Vec<u16> {
            let mut r = vec![NAN16; w * ROW];
            if let Some(last) = last {
                for p in last.saturating_sub(w - 1)..=last {
                    let slot = p % w;
                    r[slot * ROW..(slot + 1) * ROW].copy_from_slice(&self.rows[p]);
                }
            }
            r
        }

        /// The rows `lo .. hi` back to back.
        fn span(&self, lo: usize, hi: usize) -> Vec<u16> {
            self.rows[lo..hi].concat()
        }
    }

    /// The window keys of position `p` as rows of the `w`-row ring after `p`.
    fn window_keys(p: usize, w: usize) -> Vec<usize> {
        (p.saturating_sub(w - 1)..=p).map(|q| q % w).collect()
    }

    fn dense_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        const ROWS: usize = TOP_K + 64;
        let mut ok = Vec::new();
        let host: Vec<u16> = (0..ROWS).flat_map(|_| row(rng)).collect();
        for (counts, tokens) in [
            (&[1usize, 31, 32, 33, 64, 65, 1000, 2048][..], 1usize),
            (&[1000][..], 4),
        ] {
            for &n in counts {
                // Rows past the count hold NaN: the launch must not read them.
                let mut c = host.clone();
                c[n * ROW..].fill(NAN16);
                let cache = dev.rows(&c, ROWS)?;
                let qs = Queries::new(rng, tokens, DSA_HEADS);
                let y = dense(dev, &qs, &cache, n, Some(&qs.sinks))?;
                let want = reference(&qs, &c, &|t| (0..n - tokens + t + 1).collect());
                ok.push(banded(&format!("dense n={n} tokens={tokens}"), &y, &want));
                let again = dense(dev, &qs, &cache, n, Some(&qs.sinks))?;
                ok.push(clause(
                    &format!("dense n={n} tokens={tokens} rerun"),
                    bits_equal(&y, &again),
                    "bit-identical",
                ));
                if tokens == 1 {
                    let list: Vec<u32> = (0..TOP_K as u32).collect();
                    let sel = selected(dev, &qs, &cache, &list, &[n as u32], Some(&qs.sinks))?;
                    ok.push(clause(
                        &format!("dense n={n} is_identity_list"),
                        bits_equal(&y, &sel),
                        "the selected launch over 0..count, stride 2048: the dense bits",
                    ));
                }
            }
        }
        ok.push(dev.no_fault("dense")?);
        Ok(ok)
    }

    /// Lists of `top_k` distinct rows of a `2 · top_k`-row cache, one a token, token `t` live
    /// over its first `counts[t]`.
    fn selected_clauses(
        dev: &Dev,
        rng: &mut GateRng,
        top_k: usize,
        counts: &[u32],
    ) -> Result<Vec<bool>, GateError> {
        let rows = 2 * top_k;
        let mut c: Vec<u16> = vec![NAN16; rows * ROW];
        let mut list = Vec::with_capacity(counts.len() * top_k);
        for _ in counts {
            // A seeded draw of top_k distinct rows, in the draw's order.
            let mut pool: Vec<u32> = (0..rows as u32).collect();
            for i in 0..top_k {
                let j = i + rng.below(rows - i);
                pool.swap(i, j);
            }
            list.extend_from_slice(&pool[..top_k]);
        }
        // Only rows a token's live list names get values; the rest stay NaN.
        for (t, &n) in counts.iter().enumerate() {
            for &r in &list[t * top_k..t * top_k + n as usize] {
                let r = r as usize;
                if c[r * ROW] == NAN16 {
                    c[r * ROW..(r + 1) * ROW].copy_from_slice(&row(rng));
                }
            }
        }
        let cache = dev.rows(&c, rows)?;
        let qs = Queries::new(rng, counts.len(), DSA_HEADS);
        let y = selected(dev, &qs, &cache, &list, counts, Some(&qs.sinks))?;
        let want = reference(&qs, &c, &|t| {
            list[t * top_k..t * top_k + counts[t] as usize]
                .iter()
                .map(|&r| r as usize)
                .collect()
        });
        Ok(vec![
            banded(&format!("selected k={top_k} counts={counts:?}"), &y, &want),
            dev.no_fault(&format!("selected k={top_k}"))?,
        ])
    }

    /// The `ring_rows`-row window at each of `positions` (ascending, the first 0), eager and
    /// replayed from one graph.
    fn window_clauses(
        dev: &Dev,
        rng: &mut GateRng,
        ring_rows: usize,
        positions: &[usize],
    ) -> Result<Vec<bool>, GateError> {
        let st = Positions::new(rng, positions[positions.len() - 1] + 1);
        let mut ok = Vec::new();
        let qs0 = Queries::new(rng, 1, SWA_HEADS);
        let ring0 = st.ring(Some(positions[0]), ring_rows);
        let mut w = Win::new(dev, &qs0, &ring0, None, positions[0])?;
        let s = dev.stream();
        let graph = {
            let (flash, w) = (&dev.flash, &mut w);
            dev.gpu.capture(|cs| w.enqueue(flash, cs))?
        };
        for &p in positions {
            let qs = Queries::new(rng, 1, SWA_HEADS);
            let ring = st.ring(Some(p), ring_rows);
            w.q.copy_from_host(s, &qs.q)?;
            w.ring.buf_mut().copy_from_host(s, &ring)?;
            w.pos.copy_from_host(s, &[p as u32])?;
            w.enqueue(&dev.flash, s)?;
            let y = dev.read(&w.y)?;
            let want = reference(&qs, &ring, &|_| window_keys(p, ring_rows));
            ok.push(banded(&format!("window w={ring_rows} pos={p}"), &y, &want));
            graph.launch(s)?;
            let yg = dev.read(&w.y)?;
            ok.push(clause(
                &format!("window w={ring_rows} pos={p} replay"),
                bits_equal(&y, &yg),
                "the graph captured at position 0: the eager bits",
            ));
        }
        ok.push(dev.no_fault(&format!("window w={ring_rows}"))?);
        Ok(ok)
    }

    /// The staged batch over a `w`-row ring at each `(b, T)` of `cases`.
    fn batch_clauses(
        dev: &Dev,
        rng: &mut GateRng,
        w_rows: usize,
        cases: &[(usize, usize)],
    ) -> Result<Vec<bool>, GateError> {
        let mut ok = Vec::new();
        let s = dev.stream();
        for &(b, tokens) in cases {
            let st = Positions::new(rng, b + tokens);
            let qs = Queries::new(rng, tokens, SWA_HEADS);
            let before = st.ring(b.checked_sub(1), w_rows);
            let stage = st.span(b, b + tokens);
            let mut w = Win::new(dev, &qs, &before, Some(&stage), b)?;
            w.enqueue(&dev.flash, s)?;
            let yb = dev.read(&w.y)?;
            let per = SWA_HEADS * LATENT;
            // Each token's one-token launch over the ring after its own position.
            let one = Queries {
                q: qs.q[..SWA_HEADS * ROW].to_vec(),
                sinks: qs.sinks.clone(),
                heads: SWA_HEADS,
            };
            let mut d = Win::new(dev, &one, &before, None, b)?;
            let mut differing = Vec::new();
            let sample: Vec<usize> = (0..tokens)
                .filter(|t| t % 64 == 0 || *t == tokens - 1)
                .collect();
            let mut want = Vec::new();
            let mut got = Vec::new();
            for t in 0..tokens {
                let p = b + t;
                let ring = st.ring(Some(p), w_rows);
                d.q.copy_from_host(s, &qs.q[t * SWA_HEADS * ROW..(t + 1) * SWA_HEADS * ROW])?;
                d.ring.buf_mut().copy_from_host(s, &ring)?;
                d.pos.copy_from_host(s, &[p as u32])?;
                d.enqueue(&dev.flash, s)?;
                let y1 = dev.read(&d.y)?;
                if !bits_equal(&y1, &yb[t * per..(t + 1) * per]) {
                    differing.push(t);
                }
                if sample.contains(&t) {
                    let qt = Queries {
                        q: qs.q[t * SWA_HEADS * ROW..(t + 1) * SWA_HEADS * ROW].to_vec(),
                        sinks: qs.sinks.clone(),
                        heads: SWA_HEADS,
                    };
                    want.extend(reference(&qt, &ring, &|_| window_keys(p, w_rows)));
                    got.extend_from_slice(&yb[t * per..(t + 1) * per]);
                }
            }
            ok.push(clause(
                &format!("batch w={w_rows} b={b} T={tokens} is_steps"),
                differing.is_empty(),
                &format!(
                    "{} of {tokens} tokens differ from their one-token launch, first {:?}",
                    differing.len(),
                    &differing[..differing.len().min(8)]
                ),
            ));
            ok.push(banded(
                &format!("batch w={w_rows} b={b} T={tokens} sample={}", sample.len()),
                &got,
                &want,
            ));
            dev.flash.enqueue_kv_ring_commit(
                s,
                RingCommitArgs {
                    stage: w.stage.as_ref().ok_or("the batch has staging")?,
                    base: &w.pos,
                    tokens,
                    ring: &mut w.ring,
                },
            )?;
            s.synchronize()?;
            let after = w.ring.buf().to_host_vec(s)?;
            let want_ring = st.ring(Some(b + tokens - 1), w_rows);
            let slots = (0..w_rows)
                .filter(|&k| after[k * ROW..(k + 1) * ROW] != want_ring[k * ROW..(k + 1) * ROW])
                .count();
            ok.push(clause(
                &format!("batch w={w_rows} b={b} T={tokens} commit"),
                slots == 0,
                &format!(
                    "{slots} ring slots differ from the ring at position {}",
                    b + tokens - 1
                ),
            ));
        }
        ok.push(dev.no_fault(&format!("batch w={w_rows}"))?);
        Ok(ok)
    }

    fn sink_exact_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        const ROWS: usize = TOP_K + 64;
        let mut ok = Vec::new();
        let host: Vec<u16> = (0..ROWS).flat_map(|_| row(rng)).collect();
        let cache = dev.rows(&host, ROWS)?;
        let dead = vec![f32::NEG_INFINITY; SWA_HEADS];
        for n in [1usize, 2, 1000] {
            let qs = Queries::new(rng, 1, DSA_HEADS);
            let with = dense(dev, &qs, &cache, n, Some(&dead[..DSA_HEADS]))?;
            let plain = dense(dev, &qs, &cache, n, None)?;
            ok.push(clause(
                &format!("sink_exact dense n={n}"),
                bits_equal(&with, &plain),
                "every sink -inf: the plain merge's bits",
            ));
            let list: Vec<u32> = (0..TOP_K as u32).collect();
            let with = selected(
                dev,
                &qs,
                &cache,
                &list,
                &[n as u32],
                Some(&dead[..DSA_HEADS]),
            )?;
            let plain = selected(dev, &qs, &cache, &list, &[n as u32], None)?;
            ok.push(clause(
                &format!("sink_exact selected n={n}"),
                bits_equal(&with, &plain),
                "every sink -inf: the plain merge's bits",
            ));
            // Each head's sink SINK_OVER above its largest logit: the rows shrink by e^-30 and
            // stay in band.
            let mut big = Queries::new(rng, 1, DSA_HEADS);
            let keys: Vec<usize> = (0..n).collect();
            big.sinks = (0..DSA_HEADS)
                .map(|h| {
                    let l = logits(&big.q[h * ROW..(h + 1) * ROW], &host, &keys);
                    l.into_iter().fold(f64::NEG_INFINITY, f64::max) as f32 + SINK_OVER
                })
                .collect();
            let y = dense(dev, &big, &cache, n, Some(&big.sinks))?;
            let want = reference(&big, &host, &|_| (0..n).collect());
            ok.push(banded(&format!("sink_exact dominant n={n}"), &y, &want));
        }
        let st = Positions::new(rng, WINDOW);
        for p in [0usize, 100, 511] {
            let mut qs = Queries::new(rng, 1, SWA_HEADS);
            qs.sinks = dead.clone();
            let ring = st.ring(Some(p), WINDOW);
            let mut w = Win::new(dev, &qs, &ring, None, p)?;
            w.enqueue(&dev.flash, dev.stream())?;
            let yw = dev.read(&w.y)?;
            let plain = dense(dev, &qs, &w.ring, p + 1, None)?;
            ok.push(clause(
                &format!("sink_exact window pos={p}"),
                bits_equal(&yw, &plain),
                "every sink -inf: the dense plain launch over the ring's rows 0..=pos",
            ));
        }
        ok.push(dev.no_fault("sink_exact")?);
        Ok(ok)
    }

    fn out_gate_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        const V_HEAD: usize = 256;
        const TOKENS: usize = 3;
        let s = dev.stream();
        let og = OutGateKernels::load(dev.gpu.context())?;
        let n = TOKENS * SWA_HEADS * V_HEAD;
        let attn: Vec<f32> = (0..n).map(|_| rng.uni(-3.0, 3.0)).collect();
        let gate: Vec<f32> = (0..n).map(|_| rng.uni(-9.0, 9.0)).collect();
        let (a, g) = (dev.up(&attn)?, dev.up(&gate)?);
        let mut y = DeviceBuffer::<f32>::zeroed(s, n)?;
        let mut y_head = DeviceBuffer::<f32>::zeroed(s, n)?;
        let args = |kind: OutGate, y: &mut DeviceBuffer<f32>| -> Result<(), GpuError> {
            og.enqueue_out_gate_head(
                s,
                OutGateArgs {
                    kind,
                    attn: &a,
                    gate: &g,
                    n_head: SWA_HEADS,
                    m: TOKENS,
                    fault: dev.gpu.unlabelled_sink(),
                    y,
                },
                V_HEAD,
            )
        };
        args(OutGate::Proj, &mut y)?;
        let got = dev.read(&y)?;
        let head = args(OutGate::ProjHead, &mut y_head);
        let dims = |v_head| MlaDims {
            embd: 3072,
            heads: SWA_HEADS,
            q_lora: 1024,
            nope: 192,
            extra: ROPE,
            v_head,
            rms_eps: 1e-6,
            latent_eps: 1e-6,
        };
        Ok(vec![
            clause(
                "out_gate v_head=256 proj",
                bits_equal(&got, &out_gate_host(&attn, &gate)),
                "the host rule's bits over 3 tokens of 64 heads",
            ),
            clause(
                "out_gate v_head=256 proj_head",
                matches!(head, Err(GpuError::Shape { .. })),
                &format!("{head:?}"),
            ),
            clause(
                "out_gate mla_dims",
                dims(V_HEAD).check().is_ok() && dims(192).check().is_err(),
                &format!("{:?} / {:?}", dims(V_HEAD).check(), dims(192).check()),
            ),
        ])
    }

    /// A refusal clause: the launch refused by its shape check, before anything is enqueued (a
    /// launch contract that fires later, after the window pass ran, does not count).
    fn refused(name: &str, r: Result<(), GpuError>) -> bool {
        clause(
            &format!("refusal {name}"),
            matches!(r, Err(GpuError::Shape { .. })),
            &format!("{r:?}"),
        )
    }

    fn refusal_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        let s = dev.stream();
        let mut ok = Vec::new();
        let ring = vec![0u16; WINDOW * ROW];
        let mut try_window = |name: &str,
                              heads: usize,
                              ring_rows: usize,
                              cols: usize,
                              sinks: usize|
         -> Result<bool, GateError> {
            let qs = Queries::new(rng, 1, heads);
            let mut w = Win::new(dev, &qs, &ring, None, 0)?;
            w.ring = DeviceTensor::<u16>::zeroed(s, ring_rows, cols)?;
            w.sinks = DeviceBuffer::<f32>::zeroed(s, sinks)?;
            Ok(refused(name, w.enqueue(&dev.flash, s)))
        };
        ok.push(try_window(
            "ring 512 wide",
            SWA_HEADS,
            WINDOW,
            LATENT,
            SWA_HEADS,
        )?);
        ok.push(try_window("ring of no row", SWA_HEADS, 0, ROW, SWA_HEADS)?);
        ok.push(try_window(
            "sinks 63 for 64 heads",
            SWA_HEADS,
            WINDOW,
            ROW,
            SWA_HEADS - 1,
        )?);
        let stage = dev.rows(&ring[..ROW], 1)?;
        let base = dev.up(&[0u32])?;
        let mut rt = DeviceTensor::<u16>::zeroed(s, WINDOW, ROW)?;
        let r = dev.flash.enqueue_kv_ring_commit(
            s,
            RingCommitArgs {
                stage: &stage,
                base: &base,
                tokens: 0,
                ring: &mut rt,
            },
        );
        ok.push(refused("commit of no token", r));
        // Last: a launch past this check would read a second, partial group of query rows.
        ok.push(try_window("heads=24", 24, WINDOW, ROW, 24)?);
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let flash = FlashKernels::load(gpu.context())?;
        println!(
            "gate_hy4_attn: device {} — row {ROW} (rope {ROPE} + latent {LATENT}), heads \
             {DSA_HEADS}/{SWA_HEADS}, window {WINDOW}, list {TOP_K}, scale 1/16, synthetic",
            gpu.device_name()?
        );
        let dev = Dev { gpu, flash };
        let mut rng = GateRng(0x4879_3461_7474_6e00);
        let mut results = Vec::new();
        results.extend(dense_clauses(&dev, &mut rng)?);
        results.extend(selected_clauses(&dev, &mut rng, TOP_K, &[2048, 1, 1000])?);
        results.extend(selected_clauses(
            &dev,
            &mut rng,
            FIXTURE_TOP_K,
            &[64, 1, 33],
        )?);
        let positions = [0, 1, 63, 64, 65, 511, 512, 513, 1000, 1535];
        results.extend(window_clauses(&dev, &mut rng, WINDOW, &positions)?);
        let positions = [0, 1, 63, 64, 65, 100, 200];
        results.extend(window_clauses(&dev, &mut rng, FIXTURE_WINDOW, &positions)?);
        let cases = [(0, 7), (505, 20), (1000, 520)];
        results.extend(batch_clauses(&dev, &mut rng, WINDOW, &cases)?);
        let cases = [(0, 7), (60, 10), (100, 70)];
        results.extend(batch_clauses(&dev, &mut rng, FIXTURE_WINDOW, &cases)?);
        results.extend(sink_exact_clauses(&dev, &mut rng)?);
        results.extend(out_gate_clauses(&dev, &mut rng)?);
        results.extend(refusal_clauses(&dev, &mut rng)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "gate_hy4_attn: {} — {} clauses, {failed} failed",
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
