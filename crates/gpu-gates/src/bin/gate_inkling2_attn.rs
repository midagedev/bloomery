//! GPU gate for Inkling 2's attention path on the card: the projection epilogues of a visit and
//! the rel-bias attention at the group of three and of six, seeded at the Mini's shapes (24 query
//! and 8 key heads of 128 values, window 1024, relative extent 1024, the qk clip 22.627417, the
//! rel clip 8.0) and at the Small's heads (48 over 8), then the fixture's node dumps replayed as inputs (refset family `fx-lcpp-inkling2`).
//! Nothing opens a model file.
//!
//! The reference is llama.cpp's `src/models/inkling2.cpp` of pull request #32695's head (`clip()`
//! and the attention walk there): the epilogue scales a query row by its per-channel table, clips
//! the row to an L2 norm of at most the threshold through a root biased by 1e-6 and a factor
//! clamped to `[0, 1]`, RMS-norms a value row gainless, clips a relative row at 8.0, multiplies
//! the query and relative rows of a global visit by the log-scaled position factor, and appends
//! the key and value rows to the cache as f16; the attention scores carry the bias
//! `Σ_d rel_bias[d, dist]·r_d` gathered by key distance (0 at a distance at or past the extent),
//! and a per-query sink logit joins the softmax as one more denominator term.
//!
//! 1. `epilogue`: `qk_clip_append` over two tokens — a row at each clip's exact boundary (128
//!    values of 2.0, whose L2 root is the qk threshold's own f32; 16 values of 2.0, whose root is
//!    8.0), rows above and below the thresholds, and the second token at a position past the log
//!    scale's floor with its tau — bit for bit the host rule, which is the kernel's own order
//!    (squares in f32, the lane sum ascending, the warp butterfly in f64, the f32 root, the
//!    biased divide, the clamp, the multiplies); the appends' f16 bits the rule's; a rerun
//!    bit-identical; the untouched cache cells untouched. A launch with a value whose square
//!    overflows raises `CacheValue` and writes NaN rows; a position past `ctx` raises `CachePos`,
//!    writes NaN and appends nothing.
//!    `file_clip`: the fixture file's own `attn_q_clip` and `attn_rel_clip` (one scale row a head)
//!    through the same launch of layers 0 and 5, against the host rule with each head's own row.
//! 2. `attention`: the segment and merge launches of `rel_bias` over a plane — windowed rows at
//!    live counts 1, 511, 512, 1023, 1024 and 1025 (the last with the oldest key dropped), global
//!    rows at 1, 1024 and 1100 (the last with 76 keys past the extent, where the bias reads 0) —
//!    against an f64 re-dot within [`ATTN_BAND`], a rerun bit-identical, the ring form over a
//!    ring of the full context the plane's bits, a live count past `ctx` raising `KeyCount` with
//!    NaN rows, and the refusals by name.
//! 3. `sink`: a launch whose every sink is 30 above every logit leaves its rows at zero weight
//!    within [`ATTN_BAND`] — the sink column the softmax's, not an afterthought.
//! 4. `fx_a`: prompt a's windowed slot 1 — every (token, head) row — `Qcur` as the query rows,
//!    `Kcur` and `Vcur` as the f16 cache, and the bias table rebuilt from `kq_b` so the kernel's
//!    scores are the vendor's own, the sink the algebra's split of `kq_b`'s sink; each row within
//!    [`FX_F64_BAND`] of an f64 re-dot and within [`FX_VENDOR_BAND`] of the dumped `kqv_out`.
//!    The windowed slots' `Qcur`, `Kcur` and `Rcur` rows also carry the clips' invariants (an L2
//!    norm at most the thresholds', a value row's RMS at 1) — the reference clipped them too.
//! 5. `fx_e`: prompt e's last occurrence — slot 1's rows at the window's turnover (the last 14
//!    positions attend 1,024 keys with the oldest dropped) and slot 5's global rows at 1,025 to
//!    1,038 keys, where the vendor's `kq_b` columns past the extent hold the negated sink and
//!    the kernel's pad rule reads 0: the deep columns' constancy, then the same two bands a row.
//!
//! The bands hold the card's f32 sums, `ex2.approx` (2^-22 a weight) and the f32 value sums over
//! at most 1,102 keys; [`FX_VENDOR_BAND`] also holds the dump's CPU tree's f16 roundings of q and
//! of the softmax weights against our f32 operands. Bit identity with the reference is not
//! claimed.
//!
//! The FAIL-first mutants this gate holds red, one a contract: (a) a distance map that swaps
//! `dist` and `extent − 1 − dist` in the bias staging — the window-edge rows of `attention` and
//! `fx_e`'s turnover rows; (b) a merge that drops the sink column — `sink`'s rows; (c) a clip
//! threshold one ulp off — `epilogue`'s bit rule on the rows above the threshold; (d) a tau of 1
//! on a global row — `epilogue`'s second token; (e) a pad rule that reads the table's last column
//! — `attention`'s global 1,100 row and `fx_e`'s deep rows.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_attn", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::flash_gqa::window_segments;
    use bloomery_gpu::qk_clip::{HEAD, QkClipArgs, QkClipKernels, REL};
    use bloomery_gpu::rel_bias::{RelBiasArgs, RelBiasKernels};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{
        GateError, GateRng, RefManifest, bits_equal, checks_failed, load_ref_in, max_rel_err,
        par_map, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use refset::arch::inkling2::fixture::{self, A, E};

    /// The fixture's and the Mini's attention shape: query and key heads of 128 values, the
    /// query heads three a key head.
    const N_HEAD: usize = 24;
    const N_KV: usize = 8;
    /// The Small's query heads over the same eight key heads: the group of six.
    const N_HEAD_SMALL: usize = 48;
    /// The windowed visits' window and the bias table's extent, the one distance the bias reads
    /// 0 past (the Mini's and the fixture's own values).
    const WINDOW: usize = 1024;
    const EXTENT: usize = 1024;
    /// The qk rows' clip threshold (`f_qk_clip`) and the relative rows' (`f_relative_clip`), the
    /// value rows' RMS epsilon.
    const QK_CLIP: f32 = 22.627417;
    const REL_CLIP: f32 = 8.0;
    const V_EPS: f32 = 1e-6;
    /// The log scale's floor and slope, the global visits' tau.
    const TAU_FLOOR: f64 = 8192.0;
    const TAU_ALPHA: f64 = 0.1;

    /// `1/√128`, the score scale of a 128-value head.
    fn scale() -> f32 {
        1.0 / (HEAD as f32).sqrt()
    }

    /// The log-scaled position factor of the global visits: 1 + 0.1·ln(max(1, (pos+1)/8192)).
    fn tau_of(pos: u32) -> f32 {
        let ratio = (f64::from(pos + 1) / TAU_FLOOR).max(1.0);
        (1.0 + TAU_ALPHA * ratio.ln()) as f32
    }

    /// A sink this far above every logit dominates the denominator: the row's weights all but
    /// vanish.
    const SINK_OVER: f32 = 30.0;
    /// [derived] A seeded attention launch against its f64 re-dot: the card's f32 logit sums
    /// (≲ 1e-5 of a logit), `ex2.approx` (2^-22 a weight) and the f32 value sums over at most
    /// 1,102 keys (√1102 · 2^-24 of a row's largest value, ~2e-6) — 2e-4 is forty times the
    /// value-sum term at the largest output a launch writes.
    const ATTN_BAND: f32 = 2e-4;
    /// [derived] A replayed row against its f64 re-dot: the same three terms at the dumped
    /// rows' magnitudes.
    const FX_F64_BAND: f32 = 2e-4;
    /// [derived] A replayed row against the vendor's `kqv_out`: the f64 band's terms plus the
    /// dump's CPU tree's two f16 roundings against our f32 operands — its f16-cache `mul_mat`
    /// takes q as f16 (2^-11 of an element) and the softmax weights as f16 for the value
    /// product (2^-11 a weight); the two added whole, 2 · 2^-11 ≈ 1e-3.
    /// PIN(2026-11-22): from 5e-2 to the derived bound; the fixture's sets a and e hold 2.04e-4.
    const FX_VENDOR_BAND: f32 = 1e-3;
    /// The bound of a clipped row's L2 root as the dump holds it: the threshold plus the f32
    /// rounding of a row the clip scaled to it (worst case its ulp error over 128 values,
    /// ~3e-6 at the qk threshold); a threshold wrong in a printed digit fails this by far more.
    const CLIP_SLACK: f32 = 1e-5;
    /// The seed of every seeded draw.
    const SEED: u64 = 0x496e_6b32_6174_746e;

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

    /// Exact equality of two u16 rows (the cache planes' bits).
    fn u16_equal(a: &[u16], b: &[u16]) -> bool {
        a == b
    }

    fn refused(name: &str, r: Result<(), GpuError>) -> bool {
        clause(
            &format!("refusal {name}"),
            matches!(r, Err(GpuError::Shape { .. })),
            &format!("{r:?}"),
        )
    }

    /// The card, its stream and the two modules.
    struct Dev {
        gpu: Gpu,
        rel: RelBiasKernels,
        clip: QkClipKernels,
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

    // -------------------------------------------------- the epilogue's host rule

    /// The warp's f64 butterfly over 32 lane sums: the five shuffles, halves down, each step
    /// reading the lanes' values before it.
    #[allow(
        clippy::needless_range_loop,
        reason = "the butterfly reads the lanes before the step at `l` and `l ^ off`"
    )]
    fn butterfly(mut lanes: [f64; 32]) -> f64 {
        for off in [16usize, 8, 4, 2, 1] {
            let old = lanes;
            for l in 0..32 {
                lanes[l] = old[l] + old[l ^ off];
            }
        }
        lanes[0]
    }

    /// One 128-wide row's clip, the kernel's own order: the channel scale, each value squared
    /// in f32, the lane sums ascending, the warp butterfly in f64, the sum cast to f32 and
    /// rooted, the 1e-6 bias, the divide, the clamp to `[0, 1]`, the two multiplies back.
    #[allow(
        clippy::needless_range_loop,
        reason = "the lane layout is the kernel's own: lane l owns values l, l+32, l+64, l+96"
    )]
    fn host_clip128(x: &[f32], g: &[f32], thresh: f32, tau: f32) -> Vec<f32> {
        let mut lanes = [0.0f64; 32];
        for l in 0..32 {
            let mut acc = 0.0f64;
            for i in 0..4 {
                let c = l + 32 * i;
                let y = if g.is_empty() { x[c] } else { x[c] * g[c] };
                acc += f64::from(y * y);
            }
            lanes[l] = acc;
        }
        let root = (butterfly(lanes) as f32).sqrt();
        let f = (thresh / (root + 1e-6)).clamp(0.0, 1.0);
        (0..HEAD)
            .map(|c| {
                let y = if g.is_empty() { x[c] } else { x[c] * g[c] };
                y * f * tau
            })
            .collect()
    }

    /// One relative row's clip at `REL_CLIP`: the same order over the row's first `REL` values,
    /// the channel scale `gr`, `tau` carried.
    #[allow(
        clippy::needless_range_loop,
        reason = "the lane layout is the kernel's own: lane l owns values l, l+32, l+64, l+96"
    )]
    fn host_clip_rel(x: &[f32], gr: &[f32], tau: f32) -> Vec<f32> {
        let mut lanes = [0.0f64; 32];
        for l in 0..32 {
            let y = if l < REL { x[l] * gr[l] } else { 0.0 };
            lanes[l] = if l < REL { f64::from(y * y) } else { 0.0 };
        }
        let root = (butterfly(lanes) as f32).sqrt();
        let f = (REL_CLIP / (root + 1e-6)).clamp(0.0, 1.0);
        (0..REL).map(|d| x[d] * gr[d] * f * tau).collect()
    }

    /// One value row's gainless RMS, `qk_rms`'s order: the same square sum, the mean, the
    /// scale `1/√(mean + eps)` (NaN outside the finite positive range), one multiply.
    #[allow(
        clippy::needless_range_loop,
        reason = "the lane layout is the kernel's own: lane l owns values l, l+32, l+64, l+96"
    )]
    fn host_rms128(x: &[f32], eps: f32) -> Vec<f32> {
        let mut lanes = [0.0f64; 32];
        for l in 0..32 {
            let mut acc = 0.0f64;
            for i in 0..4 {
                let y = x[l + 32 * i];
                acc += f64::from(y * y);
            }
            lanes[l] = acc;
        }
        let mean = (butterfly(lanes) / HEAD as f64) as f32;
        let s = 1.0 / (mean + eps).sqrt();
        let scale = if s > 0.0 && s < f32::INFINITY {
            s
        } else {
            f32::NAN
        };
        (0..HEAD).map(|c| scale * x[c]).collect()
    }

    // -------------------------------------------------- the attention's f64 rule

    /// One row's f64 attention: the live keys of `n_keys` under `window`, the scores
    /// `scale·q·k + Σ_d bias[d, dist]·r_d` (the bias 0 at a distance at or past `EXTENT`), the
    /// sink one more denominator logit, the values the cache's f16 rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "the f64 rule of one row's attention, the kernel's own flat state"
    )]
    fn ref_rel_row(
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys: u32,
        window: usize,
        bias: &[f32],
        r_row: &[f32],
        sink: f32,
    ) -> Vec<f64> {
        let limit = n_keys as usize;
        let first = if window == 0 || limit < window {
            0
        } else {
            limit - window
        };
        let mut s = Vec::with_capacity(limit - first);
        for j in first..limit {
            let kb = j * HEAD;
            let mut dot = 0.0f64;
            for c in 0..HEAD {
                dot += f64::from(q[c]) * f64::from(half_to_f32(kc[kb + c]));
            }
            let dist = limit - 1 - j;
            let mut b = 0.0f64;
            if dist < EXTENT {
                for d in 0..REL {
                    b += f64::from(bias[d * EXTENT + dist]) * f64::from(r_row[d]);
                }
            }
            s.push(f64::from(scale()) * dot + b);
        }
        let m = s.iter().copied().fold(f64::from(sink), f64::max);
        let mut den = (f64::from(sink) - m).exp();
        let w: Vec<f64> = s
            .iter()
            .map(|x| {
                let e = (x - m).exp();
                den += e;
                e
            })
            .collect();
        (0..HEAD)
            .map(|c| {
                let mut acc = 0.0f64;
                for (i, wj) in w.iter().enumerate() {
                    acc += wj * f64::from(half_to_f32(vc[(first + i) * HEAD + c]));
                }
                acc / den
            })
            .collect()
    }

    // -------------------------------------------------- the seeded launches

    /// One rel-bias launch over device buffers the helper uploads; `n_head` heads a key head at
    /// three.
    #[allow(clippy::too_many_arguments, reason = "one launch helper's flat setup")]
    fn rel_launch(
        dev: &Dev,
        q: &[f32],
        kc: &[u16],
        vc: &[u16],
        n_keys: &[u32],
        r: &[f32],
        bias: &[f32],
        window: usize,
        sinks: &[f32],
        ring_rows: Option<usize>,
        (n_head, m, ctx): (usize, usize, usize),
    ) -> Result<Vec<f32>, GateError> {
        let s = dev.stream();
        let segs = window_segments(window);
        let (q, kc, vc, nk, r, bias, sinks) = (
            dev.up(q)?,
            dev.up(kc)?,
            dev.up(vc)?,
            dev.up(n_keys)?,
            dev.up(r)?,
            dev.up(bias)?,
            dev.up(sinks)?,
        );
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, m * n_head * segs * HEAD)?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, m * n_head * segs * 2)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, m * n_head * HEAD)?;
        dev.rel.enqueue(
            s,
            RelBiasArgs {
                q: &q,
                kc: &kc,
                vc: &vc,
                n_keys: &nk,
                r: &r,
                bias: &bias,
                extent: EXTENT,
                scale: scale(),
                n_kv: N_KV,
                ctx,
                ring_rows,
                m,
                window,
                sinks: &sinks,
                part_v: &mut part_v,
                part_ms: &mut part_ms,
                fault: dev.gpu.unlabelled_sink(),
                y: &mut y,
            },
            n_head,
        )?;
        dev.read(&y)
    }

    /// One epilogue launch; the buffers come back as `(q, k, v, r, cache_k, cache_v)`.
    #[allow(
        clippy::too_many_arguments,
        clippy::type_complexity,
        reason = "one launch helper's flat setup and its six read-backs (rust-quality R8)"
    )]
    fn clip_launch(
        dev: &Dev,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        r: &[f32],
        gq: &[f32],
        gr: &[f32],
        tau: &[f32],
        pos: &[u32],
        ctx: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u16>, Vec<u16>), GateError> {
        let s = dev.stream();
        let m = pos.len();
        let (mut q, mut k, mut v, mut r) = (dev.up(q)?, dev.up(k)?, dev.up(v)?, dev.up(r)?);
        let (gq, gr, tau, pos) = (dev.up(gq)?, dev.up(gr)?, dev.up(tau)?, dev.up(pos)?);
        let mut cache_k = DeviceBuffer::<u16>::zeroed(s, N_KV * ctx * HEAD)?;
        let mut cache_v = DeviceBuffer::<u16>::zeroed(s, N_KV * ctx * HEAD)?;
        cache_k.copy_from_host(s, &vec![0xa5a5; N_KV * ctx * HEAD])?;
        cache_v.copy_from_host(s, &vec![0xa5a5; N_KV * ctx * HEAD])?;
        dev.clip.enqueue(
            s,
            QkClipArgs {
                q: &mut q,
                k: &mut k,
                v: &mut v,
                r: &mut r,
                gq: &gq,
                gr: &gr,
                tau: &tau,
                pos: &pos,
                v_eps: V_EPS,
                qk_thresh: QK_CLIP,
                r_thresh: REL_CLIP,
                n_head: N_HEAD,
                n_kv: N_KV,
                ctx,
                m,
                fault: dev.gpu.unlabelled_sink(),
                cache_k: &mut cache_k,
                cache_v: &mut cache_v,
            },
        )?;
        Ok((
            dev.read(&q)?,
            dev.read(&k)?,
            dev.read(&v)?,
            dev.read(&r)?,
            dev.read(&cache_k)?,
            dev.read(&cache_v)?,
        ))
    }

    fn epilogue_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        const CTX: usize = 64;
        const P0: u32 = 17;
        const P1: u32 = 37;
        // The second token's tau is the one a global visit at pos 24575
        // carries (1 + 0.1*ln(3)); the append position is its own argument
        // and stays inside ctx.
        let m = 2usize;
        let gq: Vec<f32> = (0..N_HEAD * HEAD).map(|_| rng.uni(0.5, 1.5)).collect();
        let gr: Vec<f32> = (0..N_HEAD * REL).map(|_| rng.uni(0.5, 1.5)).collect();
        let tau = vec![1.0f32, tau_of(3 * 8192 - 1)];
        let pos = [P0, P1];
        // Head 0 of each kind sits at its clip's exact boundary: the qk rows 128 values of 2.0
        // (the root the threshold's own f32), the rel row 16 values of 2.0 (root 8.0). The
        // rows around it land above and below the thresholds.
        let mut q = vec![0.0f32; m * N_HEAD * HEAD];
        for x in q.iter_mut() {
            *x = rng.uni(-1.0, 1.0) * 4.0;
        }
        for i in 0..HEAD {
            q[i] = 2.0;
            q[N_HEAD * HEAD + i] = 0.25;
        }
        let mut k = vec![0.0f32; m * N_KV * HEAD];
        for x in k.iter_mut() {
            *x = rng.uni(-1.0, 1.0) * 4.0;
        }
        k[..HEAD].fill(2.0);
        let v: Vec<f32> = (0..m * N_KV * HEAD).map(|_| rng.uni(-1.0, 1.0)).collect();
        let mut r = vec![0.0f32; m * N_HEAD * REL];
        for x in r.iter_mut() {
            *x = rng.uni(-1.0, 1.0) * 4.0;
        }
        r[..REL].fill(2.0);
        let (q0, k0, v0, r0, ck, cv) = clip_launch(dev, &q, &k, &v, &r, &gq, &gr, &tau, &pos, CTX)?;
        // The host rule over every row, the appends at each token's position, the filler
        // everywhere else.
        let mut want_q = Vec::with_capacity(q.len());
        let mut want_k = Vec::new();
        let mut want_v = Vec::new();
        let mut want_r = Vec::new();
        for (t, &tau_t) in tau.iter().enumerate() {
            for h in 0..N_HEAD {
                let b = (t * N_HEAD + h) * HEAD;
                let g = &gq[h * HEAD..(h + 1) * HEAD];
                want_q.extend(host_clip128(&q[b..b + HEAD], g, QK_CLIP, tau_t));
                let rb = (t * N_HEAD + h) * REL;
                let g = &gr[h * REL..(h + 1) * REL];
                want_r.extend(host_clip_rel(&r[rb..rb + REL], g, tau_t));
            }
            for h in 0..N_KV {
                let b = (t * N_KV + h) * HEAD;
                want_k.extend(host_clip128(&k[b..b + HEAD], &[], QK_CLIP, 1.0));
                want_v.extend(host_rms128(&v[b..b + HEAD], V_EPS));
            }
        }
        let mut want_ck = vec![0xa5a5u16; N_KV * CTX * HEAD];
        let mut want_cv = vec![0xa5a5u16; N_KV * CTX * HEAD];
        for (t, &p) in pos.iter().enumerate() {
            for h in 0..N_KV {
                let b = (t * N_KV + h) * HEAD;
                let at = (h * CTX + p as usize) * HEAD;
                for c in 0..HEAD {
                    want_ck[at + c] = f32_to_f16_bits(want_k[b + c]);
                    want_cv[at + c] = f32_to_f16_bits(want_v[b + c]);
                }
            }
        }
        let ok = vec![
            clause(
                "epilogue q bits",
                bits_equal(&q0, &want_q),
                "the host rule's bits",
            ),
            clause(
                "epilogue k bits",
                bits_equal(&k0, &want_k),
                "the host rule's bits",
            ),
            clause(
                "epilogue v bits",
                bits_equal(&v0, &want_v),
                "the host rule's bits",
            ),
            clause(
                "epilogue r bits",
                bits_equal(&r0, &want_r),
                "the host rule's bits",
            ),
            clause(
                "epilogue cache bits",
                u16_equal(&ck, &want_ck) && u16_equal(&cv, &want_cv),
                "the rule's f16 at the positions, the filler elsewhere",
            ),
        ];
        let again = clip_launch(dev, &q, &k, &v, &r, &gq, &gr, &tau, &pos, CTX)?;
        let ok2 = vec![
            clause(
                "epilogue rerun",
                bits_equal(&q0, &again.0) && u16_equal(&ck, &again.4),
                "bit-identical",
            ),
            dev.no_fault("epilogue")?,
        ];
        // A value whose square overflows f32: the root is not finite, the rows NaN, the fault
        // named.
        let mut bad = q.clone();
        bad[7 * HEAD] = 1e30;
        let (qb, _, _, _, _, _) = clip_launch(dev, &bad, &k, &v, &r, &gq, &gr, &tau, &pos, CTX)?;
        let f = dev.gpu.fault()?;
        let nan_row = qb[7 * HEAD..8 * HEAD].iter().all(|v| v.is_nan());
        let ok3 = clause(
            "epilogue overflow fault",
            f.is_some() && nan_row,
            &format!("{f:?}, the row NaN: {nan_row}"),
        );
        dev.gpu.clear_fault()?;
        // A position past ctx: named, NaN rows, nothing appended.
        let (_, kb, _, _, ckb, _) = clip_launch(
            dev,
            &q,
            &k,
            &v,
            &r,
            &gq,
            &gr,
            &[1.0, 1.0],
            &[CTX as u32 + 1, CTX as u32 + 2],
            CTX,
        )?;
        let f = dev.gpu.fault()?;
        let nan_rows = kb[..N_KV * HEAD].iter().all(|v| v.is_nan());
        let clean = ckb.iter().all(|&x| x == 0xa5a5);
        let ok4 = clause(
            "epilogue cache-pos fault",
            f.is_some() && nan_rows && clean,
            &format!("{f:?}, NaN rows {nan_rows}, cache untouched {clean}"),
        );
        dev.gpu.clear_fault()?;
        Ok([ok, ok2, vec![ok3], vec![ok4]].concat())
    }

    /// The file's own clip tensors, `attn_q_clip` `[n_head][128]` and `attn_rel_clip`
    /// `[n_head][16]`, through the epilogue: layers 0 (windowed) and 5 (windowless) of the
    /// fixture, seeded rows against the host rule with each head's own scale row. The rows of
    /// the scale tensors differ across heads (the premise a shared scale would hide), and
    /// the output of a head that sits at the file's scale of head 0 would not be this head's.
    fn file_clip_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        const CTX: usize = 8;
        let path = bloomery_gpu_gates::ref_model_path()?;
        let split =
            gguf::Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut ok = Vec::new();
        for l in [0usize, 5] {
            let gq = bloomery_gpu_gates::split_f32(
                &split,
                &format!("blk.{l}.attn_q_clip.weight"),
                N_HEAD * HEAD,
            )?;
            let gr = bloomery_gpu_gates::split_f32(
                &split,
                &format!("blk.{l}.attn_rel_clip.weight"),
                N_HEAD * REL,
            )?;
            let differ = (1..N_HEAD).all(|h| {
                gq[h * HEAD..(h + 1) * HEAD] != gq[..HEAD]
                    && gr[h * REL..(h + 1) * REL] != gr[..REL]
            });
            ok.push(clause(
                &format!("file clip rows differ by head, layer {l}"),
                differ,
                "every head's q and rel scale row is not head 0's",
            ));
            let m = 2usize;
            let tau = vec![1.0f32, 1.0];
            let pos = [3u32, 5];
            let q: Vec<f32> = (0..m * N_HEAD * HEAD)
                .map(|_| rng.uni(-1.0, 1.0) * 4.0)
                .collect();
            let k: Vec<f32> = (0..m * N_KV * HEAD)
                .map(|_| rng.uni(-1.0, 1.0) * 4.0)
                .collect();
            let v: Vec<f32> = (0..m * N_KV * HEAD).map(|_| rng.uni(-1.0, 1.0)).collect();
            let r: Vec<f32> = (0..m * N_HEAD * REL)
                .map(|_| rng.uni(-1.0, 1.0) * 4.0)
                .collect();
            let (q0, _, _, r0, _, _) = clip_launch(dev, &q, &k, &v, &r, &gq, &gr, &tau, &pos, CTX)?;
            let mut want_q = Vec::with_capacity(q.len());
            let mut want_r = Vec::with_capacity(r.len());
            for t in 0..m {
                for h in 0..N_HEAD {
                    let b = (t * N_HEAD + h) * HEAD;
                    want_q.extend(host_clip128(
                        &q[b..b + HEAD],
                        &gq[h * HEAD..(h + 1) * HEAD],
                        QK_CLIP,
                        1.0,
                    ));
                    let rb = (t * N_HEAD + h) * REL;
                    want_r.extend(host_clip_rel(
                        &r[rb..rb + REL],
                        &gr[h * REL..(h + 1) * REL],
                        1.0,
                    ));
                }
            }
            ok.push(clause(
                &format!("file clip q bits, layer {l}"),
                bits_equal(&q0, &want_q),
                "each head's own scale row, the host rule's bits",
            ));
            ok.push(clause(
                &format!("file clip r bits, layer {l}"),
                bits_equal(&r0, &want_r),
                "each head's own scale row, the host rule's bits",
            ));
            ok.push(dev.no_fault("file clip")?);
        }
        Ok(ok)
    }

    /// A seeded `[REL][EXTENT]` table with strong distance variation and a large last column
    /// (the column a pad rule must not read).
    fn seeded_bias(rng: &mut GateRng) -> Vec<f32> {
        let mut bias = vec![0.0f32; REL * EXTENT];
        for d in 0..REL {
            for i in 0..EXTENT - 1 {
                bias[d * EXTENT + i] = rng.uni(-1.0, 1.0) * (1.0 + d as f32);
            }
            bias[d * EXTENT + EXTENT - 1] = 50.0 * (1.0 + d as f32) / REL as f32;
        }
        bias
    }

    /// The windowed and global seeded launches against the f64 rule, the ring form, the fault
    /// row.
    fn attention_clauses(
        dev: &Dev,
        rng: &mut GateRng,
        n_head: usize,
    ) -> Result<Vec<bool>, GateError> {
        let group = n_head / N_KV;
        const CTX: usize = 1100;
        let mut cache = vec![0u16; N_KV * CTX * HEAD];
        for x in cache.iter_mut() {
            *x = f32_to_f16_bits(rng.uni(-1.0, 1.0));
        }
        let bias = seeded_bias(rng);
        // A fourth of the heads carry an all-zero query row — the row's whole score is the
        // bias, the distance map alone — and a fourth a flat softmax's 3x spread.
        let head_scales = [0.0f32, 0.6, 1.5, 3.0];
        let sink_kinds = [f32::NEG_INFINITY, 0.0, 3.0, SINK_OVER];
        let mut ok = Vec::new();
        for window in [WINDOW, 0] {
            let counts = if window == 0 {
                vec![1u32, 1024, 1100]
            } else {
                vec![1, 511, 512, 1023, 1024, 1025]
            };
            let m = counts.len();
            let mut q = vec![0.0f32; m * n_head * HEAD];
            for x in q.iter_mut() {
                *x = rng.uni(-1.0, 1.0) * 3.0;
            }
            for i in 0..m * n_head {
                let sc = head_scales[i % 4];
                let b = i * HEAD;
                if sc == 0.0 {
                    q[b..b + HEAD].fill(0.0);
                } else {
                    for c in 0..HEAD {
                        q[b + c] *= sc / 3.0;
                    }
                }
            }
            let r: Vec<f32> = (0..m * n_head * REL).map(|_| rng.uni(-1.0, 1.0)).collect();
            let sinks: Vec<f32> = (0..m * n_head).map(|i| sink_kinds[i % 4]).collect();
            let n_keys = counts.clone();
            let y = rel_launch(
                dev,
                &q,
                &cache,
                &cache,
                &n_keys,
                &r,
                &bias,
                window,
                &sinks,
                None,
                (n_head, m, CTX),
            )?;
            let want: Vec<f32> = par_map(m * n_head, |i| {
                let (t, h) = (i / n_head, i % n_head);
                let kh = h / group;
                let plane = &cache[kh * CTX * HEAD..(kh + 1) * CTX * HEAD];
                ref_rel_row(
                    &q[i * HEAD..(i + 1) * HEAD],
                    plane,
                    plane,
                    n_keys[t],
                    window,
                    &bias,
                    &r[i * REL..(i + 1) * REL],
                    sinks[i],
                )
            })
            .into_iter()
            .flatten()
            .map(|v| v as f32)
            .collect();
            let what = if window == 0 { "global" } else { "windowed" };
            ok.push(banded(
                &format!("attention {n_head} heads {what} counts {counts:?}"),
                ATTN_BAND,
                &y,
                &want,
            ));
            let again = rel_launch(
                dev,
                &q,
                &cache,
                &cache,
                &n_keys,
                &r,
                &bias,
                window,
                &sinks,
                None,
                (n_head, m, CTX),
            )?;
            ok.push(clause(
                &format!("attention {n_head} heads {what} rerun"),
                bits_equal(&y, &again),
                "bit-identical",
            ));
            let ring = rel_launch(
                dev,
                &q,
                &cache,
                &cache,
                &n_keys,
                &r,
                &bias,
                window,
                &sinks,
                Some(CTX),
                (n_head, m, CTX),
            )?;
            ok.push(clause(
                &format!("attention {n_head} heads {what} ring"),
                bits_equal(&y, &ring),
                "the ring of the full context over the same rows: the plane's bits",
            ));
            ok.push(dev.no_fault(&format!("attention {n_head} heads {what}"))?);
        }
        // A live count past ctx: named, NaN rows.
        let y = rel_launch(
            dev,
            &vec![0.0; n_head * HEAD],
            &cache,
            &cache,
            &[CTX as u32 + 1],
            &vec![0.0; n_head * REL],
            &bias,
            WINDOW,
            &vec![0.0; n_head],
            None,
            (n_head, 1, CTX),
        )?;
        let f = dev.gpu.fault()?;
        ok.push(clause(
            "attention key-count fault",
            f.is_some() && y.iter().all(|v| v.is_nan()),
            &format!("{f:?}, rows NaN"),
        ));
        dev.gpu.clear_fault()?;
        Ok(ok)
    }

    /// A launch whose every sink is 30 above every logit: the rows' weights vanish.
    fn sink_clauses(dev: &Dev, rng: &mut GateRng, n_head: usize) -> Result<Vec<bool>, GateError> {
        let group = n_head / N_KV;
        const CTX: usize = 64;
        let mut cache = vec![0u16; N_KV * CTX * HEAD];
        for x in cache.iter_mut() {
            *x = f32_to_f16_bits(rng.uni(-1.0, 1.0));
        }
        // Every logit of this clause sits far under the sink: a small table,
        // small query and relative rows, so the sink's 30 is 28 above every
        // score and the rows' weights are e^-28 of the denominator.
        let mut bias = vec![0.0f32; REL * EXTENT];
        for x in bias.iter_mut() {
            *x = rng.uni(-0.05, 0.05);
        }
        let q: Vec<f32> = (0..n_head * HEAD)
            .map(|_| rng.uni(-1.0, 1.0) * 0.2)
            .collect();
        let r: Vec<f32> = (0..n_head * REL)
            .map(|_| rng.uni(-1.0, 1.0) * 0.2)
            .collect();
        let sinks = vec![SINK_OVER; n_head];
        let y = rel_launch(
            dev,
            &q,
            &cache,
            &cache,
            &[32],
            &r,
            &bias,
            WINDOW,
            &sinks,
            None,
            (n_head, 1, CTX),
        )?;
        let want: Vec<f32> = (0..n_head)
            .flat_map(|h| {
                let plane = &cache[(h / group) * CTX * HEAD..(h / group + 1) * CTX * HEAD];
                ref_rel_row(
                    &q[h * HEAD..(h + 1) * HEAD],
                    plane,
                    plane,
                    32,
                    WINDOW,
                    &bias,
                    &r[h * REL..(h + 1) * REL],
                    SINK_OVER,
                )
            })
            .map(|v| v as f32)
            .collect();
        let mut ok = vec![banded(
            &format!("sink 30 over every logit at {n_head} heads"),
            ATTN_BAND,
            &y,
            &want,
        )];
        let mag = y.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let want_mag = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        ok.push(clause(
            &format!("sink 30 rows vanish at {n_head} heads"),
            mag < 1e-9 && want_mag < 1e-9,
            &format!("max |y| {mag:.3e} (reference {want_mag:.3e})"),
        ));
        ok.push(dev.no_fault("sink")?);
        Ok(ok)
    }

    // -------------------------------------------------- the refusals

    fn refusal_clauses(dev: &Dev, rng: &mut GateRng) -> Result<Vec<bool>, GateError> {
        let s = dev.stream();
        let ctx = 1300usize;
        let cache = vec![0u16; N_KV * ctx * HEAD];
        let (q, nk, r, bias, sinks) = (
            vec![0.5f32; N_HEAD * HEAD],
            vec![4u32],
            vec![0.5f32; N_HEAD * REL],
            seeded_bias(rng),
            vec![0.0f32; N_HEAD],
        );
        let (q, kc, vc, nk, r, bias, sinks) = (
            dev.up(&q)?,
            dev.up(&cache)?,
            dev.up(&cache)?,
            dev.up(&nk)?,
            dev.up(&r)?,
            dev.up(&bias)?,
            dev.up(&sinks)?,
        );
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, N_HEAD * 16 * HEAD)?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, N_HEAD * 16 * 2)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, N_HEAD * HEAD)?;
        let mut rel = |n_head: usize, extent: usize, scale: f32, sinks_len: usize| {
            let sinks = dev.up(&vec![0.0f32; sinks_len]).unwrap();
            dev.rel.enqueue(
                s,
                RelBiasArgs {
                    q: &q,
                    kc: &kc,
                    vc: &vc,
                    n_keys: &nk,
                    r: &r,
                    bias: &bias,
                    extent,
                    scale,
                    n_kv: N_KV,
                    ctx,
                    ring_rows: None,
                    m: 1,
                    window: WINDOW,
                    sinks: &sinks,
                    part_v: &mut part_v,
                    part_ms: &mut part_ms,
                    fault: dev.gpu.unlabelled_sink(),
                    y: &mut y,
                },
                n_head,
            )
        };
        let mut ok = vec![
            refused("rel heads 26 over 8", rel(26, EXTENT, scale(), N_HEAD)),
            refused("rel heads 21 over 8", rel(21, EXTENT, scale(), N_HEAD)),
            refused("rel heads 32 over 8", rel(32, EXTENT, scale(), N_HEAD)),
            refused("rel heads 72 over 8", rel(72, EXTENT, scale(), N_HEAD)),
            refused("rel extent 0", rel(N_HEAD, 0, scale(), N_HEAD)),
            refused("rel scale inf", rel(N_HEAD, EXTENT, f32::INFINITY, N_HEAD)),
            refused("rel sinks short", rel(N_HEAD, EXTENT, scale(), N_HEAD - 1)),
        ];
        // A short bias and a short query, by name, through a whole second launch.
        let short = dev.up(&vec![0.0f32; REL * EXTENT - 1])?;
        let qshort = dev.up(&vec![0.0f32; N_HEAD * HEAD - 1])?;
        for (name, q, bias) in [
            ("rel bias short", &q, &short),
            ("rel q short", &qshort, &bias),
        ] {
            let mut pv = DeviceBuffer::<f32>::zeroed(s, N_HEAD * 16 * HEAD)?;
            let mut pm = DeviceBuffer::<f32>::zeroed(s, N_HEAD * 16 * 2)?;
            let mut yy = DeviceBuffer::<f32>::zeroed(s, N_HEAD * HEAD)?;
            let r2 = dev.rel.enqueue(
                s,
                RelBiasArgs {
                    q,
                    kc: &kc,
                    vc: &vc,
                    n_keys: &nk,
                    r: &r,
                    bias,
                    extent: EXTENT,
                    scale: scale(),
                    n_kv: N_KV,
                    ctx,
                    ring_rows: None,
                    m: 1,
                    window: WINDOW,
                    sinks: &sinks,
                    part_v: &mut pv,
                    part_ms: &mut pm,
                    fault: dev.gpu.unlabelled_sink(),
                    y: &mut yy,
                },
                N_HEAD,
            );
            ok.push(refused(name, r2));
        }
        // The epilogue's refusals.
        let (mut q, mut k, mut v, mut r) = (
            dev.up(&vec![0.5f32; N_HEAD * HEAD])?,
            dev.up(&vec![0.5f32; N_KV * HEAD])?,
            dev.up(&vec![0.5f32; N_KV * HEAD])?,
            dev.up(&vec![0.5f32; N_HEAD * REL])?,
        );
        let (gq, gr, tau, pos) = (
            dev.up(&vec![1.0f32; N_HEAD * HEAD])?,
            dev.up(&vec![1.0f32; N_HEAD * REL])?,
            dev.up(&[1.0f32])?,
            dev.up(&[0u32])?,
        );
        let mut cache_k = DeviceBuffer::<u16>::zeroed(s, N_KV * 4 * HEAD)?;
        let mut cache_v = DeviceBuffer::<u16>::zeroed(s, N_KV * 4 * HEAD)?;
        // One head's scale row each: the shared form the kernel no longer reads.
        let (gq_one, gr_one) = (dev.up(&[1.0f32; HEAD])?, dev.up(&[1.0f32; REL])?);
        let mut clip = |m: usize, ctx: usize, qk: f32, re: f32, eps: f32, shared: bool| {
            dev.clip.enqueue(
                s,
                QkClipArgs {
                    q: &mut q,
                    k: &mut k,
                    v: &mut v,
                    r: &mut r,
                    gq: if shared { &gq_one } else { &gq },
                    gr: if shared { &gr_one } else { &gr },
                    tau: &tau,
                    pos: &pos,
                    v_eps: eps,
                    qk_thresh: qk,
                    r_thresh: re,
                    n_head: N_HEAD,
                    n_kv: N_KV,
                    ctx,
                    m,
                    fault: dev.gpu.unlabelled_sink(),
                    cache_k: &mut cache_k,
                    cache_v: &mut cache_v,
                },
            )
        };
        ok.push(refused(
            "clip m 0",
            clip(0, 4, QK_CLIP, REL_CLIP, V_EPS, false),
        ));
        ok.push(refused(
            "clip m over ctx",
            clip(2, 1, QK_CLIP, REL_CLIP, V_EPS, false),
        ));
        ok.push(refused(
            "clip qk 0",
            clip(1, 4, 0.0, REL_CLIP, V_EPS, false),
        ));
        ok.push(refused(
            "clip rel NaN",
            clip(1, 4, QK_CLIP, f32::NAN, V_EPS, false),
        ));
        ok.push(refused(
            "clip eps -1",
            clip(1, 4, QK_CLIP, REL_CLIP, -1.0, false),
        ));
        ok.push(refused(
            "clip scales of one head for n_head heads",
            clip(1, 4, QK_CLIP, REL_CLIP, V_EPS, true),
        ));
        Ok(ok)
    }

    // -------------------------------------------------- the fixture replay

    /// One slot's taps of one occurrence: the query rows, the gathered bias and the attention
    /// output, with the bias's key count, the occurrence's tokens and the tokens before it.
    struct Slot {
        q: Vec<f32>,
        kq_b: Vec<f32>,
        out: Vec<f32>,
        kq_keys: usize,
        tokens: usize,
        base: usize,
    }

    /// The occurrence's token count of `Qcur-{slot}`, and the tokens before it.
    fn occ_extent(man: &RefManifest, slot: u32, occ: u32) -> Result<(usize, usize), GateError> {
        let name = format!("Qcur-{slot}");
        let mut before = 0usize;
        for o in 0..occ {
            before += man.tensor(&name, o)?.ne[2] as usize;
        }
        Ok((man.tensor(&name, occ)?.ne[2] as usize, before))
    }

    /// Slot `slot`, occurrence `occ` of the set at `man`: `Qcur`'s and `kq_b`'s and `kqv_out`'s
    /// values, the occurrence's token count and its base position.
    fn read_slot(man: &RefManifest, slot: u32, occ: u32) -> Result<Slot, GateError> {
        let (row, q) = load_ref_in(man, &format!("Qcur-{slot}"), occ)?;
        let (kb, kq_b) = load_ref_in(man, &format!("kq_b-{slot}"), occ)?;
        let (_, out) = load_ref_in(man, &format!("kqv_out-{slot}"), occ)?;
        if row.ne[0] != HEAD as u64 || row.ne[1] != N_HEAD as u64 {
            return Err(
                format!("Qcur-{slot}: ne {:?}, want {HEAD} x {N_HEAD}", &row.ne[..2]).into(),
            );
        }
        if kb.ne[2] != N_HEAD as u64 {
            return Err(format!("kq_b-{slot}: ne2 {}, want {N_HEAD}", kb.ne[2]).into());
        }
        Ok(Slot {
            q,
            kq_b,
            out,
            kq_keys: kb.ne[0] as usize,
            tokens: row.ne[2] as usize,
            base: occ_extent(man, slot, occ)?.1,
        })
    }

    /// The stitched f32 planes of every occurrence of a slot's `Kcur` and `Vcur`, head-major
    /// `[n_kv][total][HEAD]` — the kernels' own layout, so a key head's rows are one contiguous
    /// slice and `kh·total·HEAD` picks them. `total` positions whole.
    fn stitch_kv(
        man: &RefManifest,
        slot: u32,
        total: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        let mut k = vec![0.0f32; total * N_KV * HEAD];
        let mut v = vec![0.0f32; total * N_KV * HEAD];
        let mut base = 0usize;
        for occ in 0u32.. {
            let Ok((row, kk)) = load_ref_in(man, &format!("Kcur-{slot}"), occ) else {
                break;
            };
            let tokens = row.ne[2] as usize;
            let (_, vv) = load_ref_in(man, &format!("Vcur-{slot}"), occ)?;
            for t in 0..tokens {
                for h in 0..N_KV {
                    let src = (t * N_KV + h) * HEAD;
                    let dst = (h * total + base + t) * HEAD;
                    k[dst..dst + HEAD].copy_from_slice(&kk[src..src + HEAD]);
                    v[dst..dst + HEAD].copy_from_slice(&vv[src..src + HEAD]);
                }
            }
            base += tokens;
        }
        if base != total {
            return Err(format!("Kcur-{slot}: {base} stitched tokens, want {total}").into());
        }
        Ok((k, v))
    }

    /// One (token, head) row's replay: the bias table that makes the kernel's scores the
    /// vendor's (`kq_b`'s column plus the row's sink), the sink itself, the row's live count,
    /// its query row, its f64 row and the vendor's.
    struct Replay<'a> {
        table: Vec<f32>,
        sink: f32,
        limit: usize,
        kh: usize,
        q: &'a [f32],
        want: Vec<f64>,
        vendor: &'a [f32],
    }

    /// Build one row's replay: `windowed` rows attend their window, global rows every key; a
    /// row with a key past the extent recovers the sink the vendor folded into `kq_b` (its
    /// columns past the extent hold the negated sink, one value, checked), a row with none
    /// splits at 0 — the same distribution either way.
    fn replay_row<'a>(
        slot: &'a Slot,
        planes: &(Vec<u16>, Vec<u16>),
        total: usize,
        t: usize,
        h: usize,
        window: Option<usize>,
    ) -> Result<Replay<'a>, GateError> {
        let pos = slot.base + t;
        let limit = pos + 1;
        let first = match window {
            Some(w) if limit >= w => limit - w,
            _ => 0,
        };
        let kq_at = |j: usize| slot.kq_b[j + slot.kq_keys * t + slot.kq_keys * slot.tokens * h];
        let mut sink = 0.0f32;
        if limit > EXTENT {
            // The vendor's deep columns hold the pad column minus the sink, one value: the
            // negated sink. The sink is its negation, and every deep column the same value.
            let deep = kq_at(0);
            sink = -deep;
            for j in 0..=limit - 1 - EXTENT {
                if kq_at(j) != deep {
                    return Err(format!(
                        "kq_b: column {j} of the deep range holds {}, want the negated sink \
                         {deep}",
                        kq_at(j)
                    )
                    .into());
                }
            }
        }
        let mut table = vec![0.0f32; REL * EXTENT];
        for j in first..limit {
            let dist = limit - 1 - j;
            if dist < EXTENT {
                table[dist] = kq_at(j) + sink;
            }
        }
        let q = &slot.q[(t * N_HEAD + h) * HEAD..][..HEAD];
        let kh = h / 3;
        let kc = &planes.0[kh * total * HEAD..(kh + 1) * total * HEAD];
        let vc = &planes.1[kh * total * HEAD..(kh + 1) * total * HEAD];
        let mut s = Vec::with_capacity(limit - first);
        for j in first..limit {
            let mut dot = 0.0f64;
            for c in 0..HEAD {
                dot += f64::from(q[c]) * f64::from(half_to_f32(kc[j * HEAD + c]));
            }
            let dist = limit - 1 - j;
            let b = if dist < EXTENT {
                f64::from(table[dist])
            } else {
                0.0
            };
            s.push(f64::from(scale()) * dot + b);
        }
        let m = s.iter().copied().fold(f64::from(sink), f64::max);
        let mut den = (f64::from(sink) - m).exp();
        let w: Vec<f64> = s
            .iter()
            .map(|x| {
                let e = (x - m).exp();
                den += e;
                e
            })
            .collect();
        let want: Vec<f64> = (0..HEAD)
            .map(|c| {
                let mut acc = 0.0f64;
                for (i, wj) in w.iter().enumerate() {
                    acc += wj * f64::from(half_to_f32(vc[(first + i) * HEAD + c]));
                }
                acc / den
            })
            .collect();
        let vendor = &slot.out[(t * N_HEAD + h) * HEAD..][..HEAD];
        Ok(Replay {
            table,
            sink,
            limit,
            kh,
            q,
            want,
            vendor,
        })
    }

    /// Replay the rows of one slot, one launch a row (each row's bias is its own), against the
    /// f64 rule and the vendor.
    fn replay_rows(
        dev: &Dev,
        slot: &Slot,
        planes: &(Vec<u16>, Vec<u16>),
        total: usize,
        rows: &[(usize, usize)],
        window: Option<usize>,
        name: &str,
    ) -> Result<Vec<bool>, GateError> {
        let s = dev.stream();
        let segs = window_segments(window.unwrap_or(0));
        let mut got = Vec::with_capacity(rows.len() * HEAD);
        let mut want = Vec::with_capacity(rows.len() * HEAD);
        let mut vendor = Vec::with_capacity(rows.len() * HEAD);
        for &(t, h) in rows {
            let rp = replay_row(slot, planes, total, t, h, window)?;
            let kc = dev.up(&planes.0[rp.kh * total * HEAD..(rp.kh + 1) * total * HEAD])?;
            let vc = dev.up(&planes.1[rp.kh * total * HEAD..(rp.kh + 1) * total * HEAD])?;
            let mut e0 = vec![0.0f32; 3 * REL];
            for i in 0..3 {
                e0[i * REL] = 1.0;
            }
            let q3: Vec<f32> =
                rp.q.iter()
                    .copied()
                    .chain(rp.q.iter().copied())
                    .chain(rp.q.iter().copied())
                    .collect();
            let (q3, r3, sinks, nk, bias) = (
                dev.up(&q3)?,
                dev.up(&e0)?,
                dev.up(&[rp.sink, 0.0, 0.0])?,
                dev.up(&[rp.limit as u32])?,
                dev.up(&rp.table)?,
            );
            let mut part_v = DeviceBuffer::<f32>::zeroed(s, 3 * segs * HEAD)?;
            let mut part_ms = DeviceBuffer::<f32>::zeroed(s, 3 * segs * 2)?;
            let mut y = DeviceBuffer::<f32>::zeroed(s, 3 * HEAD)?;
            dev.rel.enqueue(
                s,
                RelBiasArgs {
                    q: &q3,
                    kc: &kc,
                    vc: &vc,
                    n_keys: &nk,
                    r: &r3,
                    bias: &bias,
                    extent: EXTENT,
                    scale: scale(),
                    n_kv: 1,
                    ctx: total,
                    ring_rows: None,
                    m: 1,
                    window: window.unwrap_or(0),
                    sinks: &sinks,
                    part_v: &mut part_v,
                    part_ms: &mut part_ms,
                    fault: dev.gpu.unlabelled_sink(),
                    y: &mut y,
                },
                3,
            )?;
            let y = dev.read(&y)?;
            got.extend_from_slice(&y[..HEAD]);
            want.extend(rp.want.iter().map(|&v| v as f32));
            vendor.extend_from_slice(rp.vendor);
        }
        Ok(vec![
            banded(&format!("{name} vs f64"), FX_F64_BAND, &got, &want),
            banded(&format!("{name} vs vendor"), FX_VENDOR_BAND, &got, &vendor),
            dev.no_fault(name)?,
        ])
    }

    /// A row's L2 norm in f64.
    fn l2(x: &[f32]) -> f32 {
        x.iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt() as f32
    }

    /// A row's root mean square in f64.
    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64).sqrt()
            as f32
    }

    /// The f16 planes of the stitched rows.
    fn f16_planes(k: &[f32], v: &[f32]) -> (Vec<u16>, Vec<u16>) {
        (
            k.iter().map(|&x| f32_to_f16_bits(x)).collect(),
            v.iter().map(|&x| f32_to_f16_bits(x)).collect(),
        )
    }

    /// The windowed slots' clips' invariants over prompt a's dumped rows: the reference clipped
    /// them too, at the thresholds this gate launches with.
    fn fx_clip_invariants(man: &RefManifest) -> Result<bool, GateError> {
        let mut worst_q = 0.0f32;
        let mut worst_k = 0.0f32;
        let mut worst_r = 0.0f32;
        let mut worst_v = 0.0f32;
        for slot in 0..25u32 {
            if matches!(slot, 5 | 11 | 17 | 23) {
                continue; // the global slots carry a tau past the clip
            }
            let (_, q) = load_ref_in(man, &format!("Qcur-{slot}"), 0)?;
            let (_, k) = load_ref_in(man, &format!("Kcur-{slot}"), 0)?;
            let (_, v) = load_ref_in(man, &format!("Vcur-{slot}"), 0)?;
            let (_, r) = load_ref_in(man, &format!("Rcur-{slot}"), 0)?;
            for row in q.chunks(HEAD) {
                worst_q = worst_q.max(l2(row));
            }
            for row in k.chunks(HEAD) {
                worst_k = worst_k.max(l2(row));
            }
            for row in r.chunks(REL) {
                worst_r = worst_r.max(l2(row));
            }
            for row in v.chunks(HEAD) {
                worst_v = worst_v.max(rms(row));
            }
        }
        Ok(clause(
            "fx_a the dumped rows clipped",
            worst_q <= QK_CLIP + CLIP_SLACK
                && worst_k <= QK_CLIP + CLIP_SLACK
                && worst_r <= REL_CLIP + CLIP_SLACK
                && (worst_v - 1.0).abs() < 1e-3,
            &format!(
                "max L2 q {worst_q:.6} k {worst_k:.6} r {worst_r:.6} (thresholds {QK_CLIP}, \
                 {REL_CLIP}), v rms {worst_v:.6}"
            ),
        ))
    }

    /// Prompt a: every (token, head) row of the windowed slot 1, and the clips' invariants.
    fn fx_a_clauses(dev: &Dev) -> Result<Vec<bool>, GateError> {
        let path = tier::fixture_set("the replay of prompt a", &fixture::LCPP, A)?;
        let man = RefManifest::open(&path, &fixture::LCPP)?;
        let total = occ_extent(&man, 1, 0)?.0;
        let (all_k, all_v) = stitch_kv(&man, 1, total)?;
        let planes = f16_planes(&all_k, &all_v);
        let slot = read_slot(&man, 1, 0)?;
        let rows: Vec<(usize, usize)> = (0..slot.tokens * N_HEAD)
            .map(|i| (i / N_HEAD, i % N_HEAD))
            .collect();
        let mut ok = replay_rows(
            dev,
            &slot,
            &planes,
            total,
            &rows,
            Some(WINDOW),
            "fx_a slot 1",
        )?;
        ok.push(fx_clip_invariants(&man)?);
        Ok(ok)
    }

    /// Prompt e's last occurrence: the turnover rows of the windowed slot 1 and the deep rows
    /// of the global slot 5.
    fn fx_e_clauses(dev: &Dev) -> Result<Vec<bool>, GateError> {
        let path = tier::fixture_set("the replay of prompt e", &fixture::LCPP, E)?;
        let man = RefManifest::open(&path, &fixture::LCPP)?;
        let last = {
            let mut occ = 0u32;
            while man.tensor("Qcur-1", occ + 1).is_ok() {
                occ += 1;
            }
            occ
        };
        let (tokens, base) = occ_extent(&man, 1, last)?;
        let total = base + tokens;
        let mut ok = Vec::new();
        let (all_k, all_v) = stitch_kv(&man, 1, total)?;
        let planes = f16_planes(&all_k, &all_v);
        let slot1 = read_slot(&man, 1, last)?;
        let rows1: Vec<(usize, usize)> = (0..slot1.tokens * N_HEAD)
            .map(|i| (i / N_HEAD, i % N_HEAD))
            .collect();
        ok.extend(replay_rows(
            dev,
            &slot1,
            &planes,
            total,
            &rows1,
            Some(WINDOW),
            "fx_e slot 1",
        )?);
        let (gk, gv) = stitch_kv(&man, 5, total)?;
        let gplanes = f16_planes(&gk, &gv);
        let slot5 = read_slot(&man, 5, last)?;
        let rows5: Vec<(usize, usize)> = (0..slot5.tokens * N_HEAD)
            .map(|i| (i / N_HEAD, i % N_HEAD))
            .collect();
        ok.extend(replay_rows(
            dev,
            &slot5,
            &gplanes,
            total,
            &rows5,
            None,
            "fx_e slot 5",
        )?);
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        let dev = Dev {
            rel: RelBiasKernels::load(ctx)?,
            clip: QkClipKernels::load(ctx)?,
            gpu,
        };
        println!(
            "gate_inkling2_attn: device {} — heads {N_HEAD}/{N_KV} x {HEAD}, window {WINDOW}, \
             extent {EXTENT}, qk clip {QK_CLIP}, rel clip {REL_CLIP}",
            dev.gpu.device_name()?
        );
        let mut rng = GateRng(SEED);
        let mut results = Vec::new();
        results.extend(epilogue_clauses(&dev, &mut rng)?);
        results.extend(file_clip_clauses(&dev, &mut rng)?);
        for n_head in [N_HEAD, N_HEAD_SMALL] {
            results.extend(attention_clauses(&dev, &mut rng, n_head)?);
            results.extend(sink_clauses(&dev, &mut rng, n_head)?);
        }
        results.extend(refusal_clauses(&dev, &mut rng)?);
        results.extend(fx_a_clauses(&dev)?);
        results.extend(fx_e_clauses(&dev)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "gate_inkling2_attn: {} — {} clauses, {failed} failed",
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
