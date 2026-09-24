//! GPU gate for the DSpark draft's routed MoE (`bloomery_gpu_deepseek41::
//! experts_mxfp4`) on the 3090: block 0's three MXFP4 stacks and its router.
//!
//! 1. **Experts, host rule.** For slots `[0, 64, 127]` and m = 1, 3, 8 token
//!    columns (every token routed to all three slots, weights seeded), then
//!    for three tokens routed by the router itself (one slot per token and
//!    rank, `concat_route`): the q8_1 codes the quantizer wrote equal the
//!    host's, the SwiGLU rows `h` and the combined down rows equal the host
//!    transcription of the module's rule (`bloomery_gpu::mxfp4`'s host
//!    functions: integer block sums, no dequant) bit for bit, and a second
//!    run equals the first.
//! 2. **Experts, f32 reference.** The same plans against `dequant_mxfp4`
//!    rows times the f32 activations in f64: every row's error within the
//!    q8_1 bound (the module doc of this file derives it), and the largest
//!    `max|ours − ref| / max|ref|` of the gate·up dots and of the combined
//!    down rows under their pins. The gate·up dots are the rule's on the
//!    host (the kernel keeps only `h`; item 1 holds it to that rule), the
//!    down rows the kernel's own output.
//! 3. **Router.** 16 seeded vectors against the file's router: the logits
//!    equal the host `f32_gemv` transcription bit for bit, the ids and
//!    weights equal the host rule on the kernel's scores bit for bit (the
//!    scores themselves are CUDA's `expf`/`logf` against the host libm and
//!    are printed in ulps), the tickets return to zero. Then planted ties,
//!    where the larger id must win, and a row whose every score is 0, whose
//!    weights must still be finite.
//! 4. **Oracle diagnostics, no pin**: the dsref set `code64_n32_w3`, block 0
//!    layer 0 — our router and experts on ik's FFN input against ik's
//!    logits, top-3, weights, SwiGLU rows, per-slot down rows and combine.
//!
//! The bound of item 2. A value `x` of a 128-value block with scale `d =
//! amax/127` is stored as `q·d`, `|q·d − x| ≤ (d/2)(1 + 256u)` (the rounding
//! of `x/d` adds at most `127u` to the half step). A dot `Σ w_i x_i` with
//! exact weights then errs by at most `Σ_B (d_B/2)(1 + 256u) Σ_{i∈B} |w_i|`
//! from the activations, plus `γ(n_b + 5) Σ |w_i q_i d_i|` from our f32
//! arithmetic — `n_b` fused multiply-adds on a lane, five butterfly adds;
//! the block sums are exact integers and `d_w · d_x` an exact power-of-two
//! scaling. The combine adds one fused multiply-add per slot: `Σ_u |w_u|
//! (B_u + γ(3)|dot_u|)`. That bound is the worst case, every error aligned.
//! The expected size is the random-rounding one: `q·d − x` roughly uniform
//! on ±d/2, rms `d/√12`, so a dot errs by `≈ (d/√12)·‖w‖` against a value of
//! size `≈ rms(x)·‖w‖` — for x uniform on ±2 (`amax ≈ 2`, `d ≈ 0.0157`,
//! `rms(x) ≈ 1.15`) about 0.4 % of a typical dot, whatever the 5120 terms
//! cancel to. `max|rel|` compares the largest error with the largest value
//! over the rows, both near the tail of the same spread, so it reads about
//! that ratio too.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_dspark_experts: built without the `deepseek41` feature; see `just gate-gpu-dspark-experts`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_dspark_experts", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};

    use bloomery_gpu::mxfp4::{BLOCK_BYTES, BLOCK_VALUES, lane_partial_host};
    use bloomery_gpu::route_core::renorm_divisor;
    use bloomery_gpu::{DeviceTensor, Gpu};
    use bloomery_gpu_deepseek41::experts::swiglu_clamp;
    use bloomery_gpu_deepseek41::experts_mxfp4::{
        DownArgs, DraftExpertKernels, DraftRouterOut, GateUpArgs, MxAct, MxStack, N_EXPERT, N_USED,
        RouterArgs, concat_route, sqrt_softplus,
    };
    use bloomery_gpu_gates::rounding::{U, butterfly, gamma};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, data_dir, dump_stem, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row};
    use model::arch::dspark::{DraftHparams, names};

    const NAME: &str = "gate_dspark_experts";
    /// The dsref set this gate reads its diagnostics from.
    const DSREF_SET: &str = "code64_n32_w3";
    /// The ik tree every V4.1 oracle set must name in its `# build` line.
    const IK_BUILD: &str = "db517b69";
    /// The expert ids of the host-rule checks: both ends and the middle.
    const SLOTS: [u32; 3] = [0, 64, 127];
    /// Token counts of the host-rule checks.
    const MS: [usize; 3] = [1, 3, 8];
    /// Seeded vectors of the router check.
    const ROUTER_VECTORS: usize = 16;
    /// A slot index no plan uses: a route entry that names nothing.
    const NO_SLOT: u32 = u32::MAX;

    /// PIN(2026-09-24): the largest gate·up `max|ours − ref| / max|ref|` of
    /// item 2 was 5.358e-3 (plan all3, m = 3; 4.24e-3 to 5.36e-3 over the
    /// four plans, the derivation's 0.4 %), rounded up.
    const GATE_UP_REL_PIN: f64 = 5.4e-3;
    /// PIN(2026-09-24): the largest combined-down `max|rel|` was 1.051e-2
    /// (plan all3, m = 8; 8.21e-3 to 1.051e-2 over the four plans), rounded
    /// up. SwiGLU rows are heavier-tailed than the uniform inputs: a larger
    /// block amax per rms, a larger q8_1 step.
    const DOWN_REL_PIN: f64 = 1.1e-2;

    // ------------------------------------------------------------ the file

    fn tensor_bytes<'a>(
        split: &'a Split,
        name: &str,
        ty: GgmlType,
        dims: &[u64],
    ) -> Result<&'a [u8], GateError> {
        let (shard, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the draft"))?;
        if t.ty != ty || t.dims != dims {
            return Err(
                format!("{name}: {:?} {:?}, expected {ty:?} {dims:?}", t.ty, t.dims).into(),
            );
        }
        Ok(split
            .shard(shard)
            .ok_or("a tensor names a shard the split does not have")?
            .data(t)?)
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// One MXFP4 stack: the file bytes (the host rule reads them) and the
    /// card copy.
    struct Stack<'a> {
        bytes: &'a [u8],
        dev: MxStack,
        rows_per_expert: usize,
        k: usize,
    }

    impl Stack<'_> {
        /// Row `r` of expert `id`, file bytes.
        fn row(&self, id: u32, r: usize) -> &[u8] {
            let rb = self.k / BLOCK_VALUES * BLOCK_BYTES;
            let at = (id as usize * self.rows_per_expert + r) * rb;
            &self.bytes[at..at + rb]
        }

        /// Row `r` of expert `id`, dequantized by ggml's rule.
        fn row_f32(&self, id: u32, r: usize) -> Vec<f32> {
            let mut out = vec![0.0f32; self.k];
            dequant_row(GgmlType::MXFP4, self.row(id, r), &mut out)
                .expect("gguf dequantizes MXFP4 rows");
            out
        }
    }

    /// Block 0's stack `name`, `[k, rows, N_EXPERT]`, read and uploaded.
    fn load_stack<'a>(
        split: &'a Split,
        s: &cuda_core::CudaStream,
        name: &str,
        k: usize,
        rows: usize,
    ) -> Result<Stack<'a>, GateError> {
        let dims = [k as u64, rows as u64, N_EXPERT as u64];
        let bytes = tensor_bytes(split, name, GgmlType::MXFP4, &dims)?;
        Ok(Stack {
            bytes,
            dev: MxStack::upload(s, bytes, N_EXPERT, rows, k)?,
            rows_per_expert: rows,
            k,
        })
    }

    // ------------------------------------------------- host rules, helpers

    /// `f(i)` for `i < n` on every core, in order.
    fn par<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
        let threads = std::thread::available_parallelism().map_or(8, usize::from);
        let chunk = n.div_ceil(threads).max(1);
        let f = &f;
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..n)
                .step_by(chunk)
                .map(|s| sc.spawn(move || (s..(s + chunk).min(n)).map(f).collect::<Vec<T>>()))
                .collect();
            hs.into_iter()
                .flat_map(|h| h.join().expect("a host worker panicked"))
                .collect()
        })
    }

    /// Our q8_1 rule of one column on the host: per 128-value block `d =
    /// amax/127` (1 for an all-zero block), `q = round(x/d)` half away from
    /// zero, clamped to ±127. The codes in value order and the block scales.
    fn q8_1_host(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let (mut q, mut d8) = (Vec::with_capacity(x.len()), Vec::new());
        for b in x.chunks(128) {
            let amax = b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            d8.push(d);
            q.extend(
                b.iter()
                    .map(|&v| (v / d).round().clamp(-127.0, 127.0) as i8),
            );
        }
        (q, d8)
    }

    /// The q8_1 quantizer's Q4_K slot of value-order word `v` (the
    /// `bloomery_gpu::mxfp4` module doc).
    fn q4_slot(v: usize) -> usize {
        256 * (v >> 8) + 32 * (v & 7) + 8 * ((v >> 6) & 3) + ((v >> 3) & 7)
    }

    /// Whether the card's q8_1 of `cols` columns equals the host rule's.
    fn q8_equal(
        cx: &Ctx<'_>,
        act: &MxAct,
        host: &[(Vec<i8>, Vec<f32>)],
    ) -> Result<bool, GateError> {
        let s = cx.gpu.stream();
        let (q4, d8) = (act.q4().to_host_vec(s)?, act.d8().to_host_vec(s)?);
        let k = act.k();
        let col_words = q4.len() / act.n_cols();
        let mut ok = true;
        for (c, (xq, d)) in host.iter().enumerate() {
            ok &= bits_equal(&d8[c * k / 128..(c + 1) * k / 128], d);
            for v in 0..k / 4 {
                let w = u32::from_le_bytes(std::array::from_fn(|i| xq[4 * v + i] as u8));
                ok &= q4[c * col_words + q4_slot(v)] == w;
            }
        }
        Ok(ok)
    }

    /// One row's dot by the host rule: the 32 lane partials, then the
    /// butterfly.
    fn dot_host(row: &[u8], q: &(Vec<i8>, Vec<f32>)) -> f32 {
        butterfly(std::array::from_fn(|lane| {
            lane_partial_host(row, &q.0, &q.1, lane)
        }))
    }

    /// The exact dot of a dequantized row with f32 activations, the bound
    /// of its q8_1 error (the module doc), and `Σ|w q d|`.
    fn dot_ref(w: &[f32], x: &[f32], q: &(Vec<i8>, Vec<f32>)) -> (f64, f64, f64) {
        let (mut exact, mut quant, mut mag) = (0.0f64, 0.0f64, 0.0f64);
        for (b, wb) in w.chunks(128).enumerate() {
            let d = f64::from(q.1[b]);
            let mut sw = 0.0f64;
            for (i, &wi) in wb.iter().enumerate() {
                let at = 128 * b + i;
                exact += f64::from(wi) * f64::from(x[at]);
                sw += f64::from(wi).abs();
                mag += (f64::from(wi) * f64::from(q.0[at]) * d).abs();
            }
            quant += d / 2.0 * (1.0 + 256.0 * U) * sw;
        }
        (exact, quant, mag)
    }

    /// Token `c`'s weight for slot `j`, the kernel's rule.
    fn slot_weight(route: &[u32], wts: &[f32], j: u32, c: usize) -> Option<f32> {
        let mut w = None;
        for u in 0..N_USED {
            if route[c * N_USED + u] == j {
                w = Some(wts[c * N_USED + u]);
            }
        }
        w
    }

    /// Values seeded by an LCG (Numerical Recipes' constants), uniform on
    /// `lo .. lo + span`.
    fn seeded(n: usize, seed: u32, lo: f32, span: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                lo + ((s >> 8) as f32 / (1u32 << 24) as f32) * span
            })
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
            .fold(0.0, f64::max)
    }

    fn max_abs(a: &[f32]) -> f64 {
        a.iter().map(|x| f64::from(x.abs())).fold(0.0, f64::max)
    }

    /// `max|ours − ref| / max|ref|` and the largest per-row error / bound.
    fn rel_and_ratio(ours: &[f32], exact: &[f64], bound: &[f64]) -> (f64, f64) {
        let (mut num, mut den, mut ratio) = (0.0f64, 0.0f64, 0.0f64);
        for ((&o, &e), &b) in ours.iter().zip(exact).zip(bound) {
            let err = (f64::from(o) - e).abs();
            num = num.max(err);
            den = den.max(e.abs());
            ratio = ratio.max(err / b);
        }
        (num / den, ratio)
    }

    // ------------------------------------------------------------- checks

    struct Ctx<'a> {
        gpu: &'a Gpu,
        dk: &'a DraftExpertKernels,
        gate: Stack<'a>,
        up: Stack<'a>,
        down: Stack<'a>,
        limit: f32,
    }

    /// One plan of the expert checks: slots, route and weights for `m` tokens.
    struct Plan {
        sel: Vec<u32>,
        route: Vec<u32>,
        wts: Vec<f32>,
        m: usize,
    }

    impl Plan {
        fn mask(&self, j: u32) -> u32 {
            (0..self.m * N_USED)
                .filter(|&i| self.route[i] == j)
                .fold(0, |m, i| m | 1 << (i / N_USED))
        }
    }

    /// What the card computes for a plan: `h`, the down input's q8_1 codes
    /// equal to the host's, the combined rows.
    struct Run {
        h: Vec<f32>,
        out: Vec<f32>,
        q8_in: bool,
        q8_h: bool,
    }

    fn run_plan(cx: &Ctx<'_>, p: &Plan, x: &[f32]) -> Result<Run, GateError> {
        let s = cx.gpu.stream();
        let (k, ff, n) = (cx.gate.k, cx.gate.rows_per_expert, cx.down.rows_per_expert);
        let n_slots = p.sel.len();
        let xd = DeviceBuffer::from_host(s, x)?;
        let mut act_x = MxAct::new(s, p.m, k)?;
        cx.dk.enqueue_quantize(s, &xd, &mut act_x)?;
        let sel = DeviceBuffer::from_host(s, &p.sel)?;
        let route = DeviceBuffer::from_host(s, &p.route)?;
        let wts = DeviceBuffer::from_host(s, &p.wts)?;
        let mut h = DeviceBuffer::<f32>::zeroed(s, n_slots * p.m * ff)?;
        let a = GateUpArgs {
            gate: &cx.gate.dev,
            up: &cx.up.dev,
            act: &act_x,
            sel: &sel,
            route: &route,
            n_slots,
            limit: cx.limit,
        };
        cx.dk.enqueue_gate_up(s, &a, &mut h)?;
        let mut act_h = MxAct::new(s, n_slots * p.m, ff)?;
        cx.dk.enqueue_quantize(s, &h, &mut act_h)?;
        let mut out = DeviceBuffer::<f32>::zeroed(s, p.m * n)?;
        let d = DownArgs {
            down: &cx.down.dev,
            act: &act_h,
            sel: &sel,
            route: &route,
            wts: &wts,
            n_slots,
            m: p.m,
        };
        cx.dk.enqueue_down(s, &d, &mut out)?;
        s.synchronize()?;
        let hv = h.to_host_vec(s)?;
        let qx: Vec<_> = (0..p.m)
            .map(|c| q8_1_host(&x[c * k..(c + 1) * k]))
            .collect();
        let qh: Vec<_> = (0..n_slots * p.m)
            .map(|c| q8_1_host(&hv[c * ff..(c + 1) * ff]))
            .collect();
        Ok(Run {
            h: hv,
            out: out.to_host_vec(s)?,
            q8_in: q8_equal(cx, &act_x, &qx)?,
            q8_h: q8_equal(cx, &act_h, &qh)?,
        })
    }

    /// The q8_1 of a given `h` and the down-and-combine launch alone.
    fn run_down(cx: &Ctx<'_>, p: &Plan, h: &[f32]) -> Result<Vec<f32>, GateError> {
        let s = cx.gpu.stream();
        let (ff, n) = (cx.gate.rows_per_expert, cx.down.rows_per_expert);
        let n_slots = p.sel.len();
        let hd = DeviceBuffer::from_host(s, h)?;
        let mut act_h = MxAct::new(s, n_slots * p.m, ff)?;
        cx.dk.enqueue_quantize(s, &hd, &mut act_h)?;
        let sel = DeviceBuffer::from_host(s, &p.sel)?;
        let route = DeviceBuffer::from_host(s, &p.route)?;
        let wts = DeviceBuffer::from_host(s, &p.wts)?;
        let mut out = DeviceBuffer::<f32>::zeroed(s, p.m * n)?;
        let d = DownArgs {
            down: &cx.down.dev,
            act: &act_h,
            sel: &sel,
            route: &route,
            wts: &wts,
            n_slots,
            m: p.m,
        };
        cx.dk.enqueue_down(s, &d, &mut out)?;
        s.synchronize()?;
        Ok(out.to_host_vec(s)?)
    }

    /// The host rule for a plan: `h` and the combined rows, from the f32
    /// inputs alone.
    fn host_plan(cx: &Ctx<'_>, p: &Plan, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (k, ff, n) = (cx.gate.k, cx.gate.rows_per_expert, cx.down.rows_per_expert);
        let n_slots = p.sel.len();
        let qx: Vec<_> = (0..p.m)
            .map(|c| q8_1_host(&x[c * k..(c + 1) * k]))
            .collect();
        let cols: Vec<(usize, usize)> = (0..n_slots)
            .flat_map(|j| (0..p.m).map(move |c| (j, c)))
            .collect();
        let hcols = par(cols.len(), |i| {
            let (j, c) = cols[i];
            let routed = p.mask(j as u32) >> c & 1 != 0;
            (0..ff)
                .map(|r| {
                    let (g, u) = if routed {
                        (
                            dot_host(cx.gate.row(p.sel[j], r), &qx[c]),
                            dot_host(cx.up.row(p.sel[j], r), &qx[c]),
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    swiglu_clamp(g, u, cx.limit)
                })
                .collect::<Vec<f32>>()
        });
        let h: Vec<f32> = hcols.concat();
        let qh: Vec<_> = (0..n_slots * p.m)
            .map(|c| q8_1_host(&h[c * ff..(c + 1) * ff]))
            .collect();
        let rows = par(n, |d| {
            let mut acc = vec![0.0f32; p.m];
            for (j, &id) in p.sel.iter().enumerate() {
                for (c, a) in acc.iter_mut().enumerate() {
                    if let Some(w) = slot_weight(&p.route, &p.wts, j as u32, c) {
                        let dot = dot_host(cx.down.row(id, d), &qh[j * p.m + c]);
                        *a = dot.mul_add(w, *a);
                    }
                }
            }
            acc
        });
        let mut out = vec![0.0f32; p.m * n];
        for (d, acc) in rows.iter().enumerate() {
            for (c, &v) in acc.iter().enumerate() {
                out[c * n + d] = v;
            }
        }
        (h, out)
    }

    /// Item 2 for a plan: (gate·up max|rel|, down max|rel|, the largest
    /// error / bound ratio over every dot and combined row).
    fn reference(cx: &Ctx<'_>, p: &Plan, x: &[f32], h: &[f32], out: &[f32]) -> (f64, f64, f64) {
        let (k, ff, n) = (cx.gate.k, cx.gate.rows_per_expert, cx.down.rows_per_expert);
        let qx: Vec<_> = (0..p.m)
            .map(|c| q8_1_host(&x[c * k..(c + 1) * k]))
            .collect();
        let (mut gu_rel, mut ratio) = (0.0f64, 0.0f64);
        let nb_k = (k / BLOCK_VALUES).div_ceil(32);
        let nb_ff = (ff / BLOCK_VALUES).div_ceil(32);
        for (j, &id) in p.sel.iter().enumerate() {
            let rows: Vec<[Vec<f32>; 2]> =
                par(ff, |r| [cx.gate.row_f32(id, r), cx.up.row_f32(id, r)]);
            for c in (0..p.m).filter(|&c| p.mask(j as u32) >> c & 1 != 0) {
                let xc = &x[c * k..(c + 1) * k];
                for (si, stack) in [&cx.gate, &cx.up].into_iter().enumerate() {
                    let per: Vec<(f32, f64, f64)> = par(ff, |r| {
                        let w = &rows[r][si];
                        let ours = dot_host(stack.row(id, r), &qx[c]);
                        let (e, qb, mag) = dot_ref(w, xc, &qx[c]);
                        (ours, e, qb + gamma(nb_k + 5) * mag)
                    });
                    let ours: Vec<f32> = per.iter().map(|t| t.0).collect();
                    let exact: Vec<f64> = per.iter().map(|t| t.1).collect();
                    let bound: Vec<f64> = per.iter().map(|t| t.2).collect();
                    let (rel, rt) = rel_and_ratio(&ours, &exact, &bound);
                    gu_rel = gu_rel.max(rel);
                    ratio = ratio.max(rt);
                }
            }
        }
        let n_slots = p.sel.len();
        let qh: Vec<_> = (0..n_slots * p.m)
            .map(|c| q8_1_host(&h[c * ff..(c + 1) * ff]))
            .collect();
        let per: Vec<Vec<(f64, f64)>> = par(n, |d| {
            let mut acc = vec![(0.0f64, 0.0f64); p.m];
            for (j, &id) in p.sel.iter().enumerate() {
                let w = cx.down.row_f32(id, d);
                for (c, a) in acc.iter_mut().enumerate() {
                    if let Some(wt) = slot_weight(&p.route, &p.wts, j as u32, c) {
                        let col = j * p.m + c;
                        let (e, qb, mag) = dot_ref(&w, &h[col * ff..(col + 1) * ff], &qh[col]);
                        let b = qb + gamma(nb_ff + 5) * mag + gamma(3) * e.abs();
                        a.0 += f64::from(wt) * e;
                        a.1 += f64::from(wt).abs() * b;
                    }
                }
            }
            acc
        });
        let mut down_rel = 0.0f64;
        for c in 0..p.m {
            let ours: Vec<f32> = (0..n).map(|d| out[c * n + d]).collect();
            let exact: Vec<f64> = per.iter().map(|a| a[c].0).collect();
            let bound: Vec<f64> = per.iter().map(|a| a[c].1).collect();
            let (rel, rt) = rel_and_ratio(&ours, &exact, &bound);
            down_rel = down_rel.max(rel);
            ratio = ratio.max(rt);
        }
        (gu_rel, down_rel, ratio)
    }

    /// Items 1 and 2 for one plan; the verdict.
    fn check_plan(cx: &Ctx<'_>, name: &str, p: &Plan, x: &[f32]) -> Result<bool, GateError> {
        let r1 = run_plan(cx, p, x)?;
        let r2 = run_plan(cx, p, x)?;
        let (hh, ho) = host_plan(cx, p, x);
        let h_same = bits_equal(&r1.h, &hh);
        let out_same = bits_equal(&r1.out, &ho);
        let rerun = bits_equal(&r1.h, &r2.h) && bits_equal(&r1.out, &r2.out);
        let pass1 = r1.q8_in && r1.q8_h && h_same && out_same && rerun;
        println!(
            "experts plan={name} m={} sel={:?} q8_1_codes_equal_host(x {} h {}) h_bit_identical={h_same} \
             (max|diff| {:e}) out_bit_identical={out_same} (max|diff| {:e}, max|out| {:e}) \
             rerun_bit_identical={rerun} {}",
            p.m,
            p.sel,
            r1.q8_in,
            r1.q8_h,
            max_abs_diff(&r1.h, &hh),
            max_abs_diff(&r1.out, &ho),
            max_abs(&r1.out),
            verdict(pass1)
        );
        let (gu, dn, ratio) = reference(cx, p, x, &r1.h, &r1.out);
        let pass2 = ratio <= 1.0 && gu <= GATE_UP_REL_PIN && dn <= DOWN_REL_PIN;
        println!(
            "band plan={name} m={} gate_up_max|rel|={gu:.3e} (pin {GATE_UP_REL_PIN:e}) \
             down_max|rel|={dn:.3e} (pin {DOWN_REL_PIN:e}) worst_error/q8_1_bound={ratio:.3e} (<= 1) {}",
            p.m,
            verdict(pass2)
        );
        Ok(pass1 && pass2)
    }

    // ------------------------------------------------------------- router

    struct Router {
        w: Vec<f32>,
        dev: DeviceTensor<f32>,
        bias: Vec<f32>,
        bias_dev: DeviceBuffer<f32>,
        k: usize,
        scale: f32,
        norm: bool,
    }

    /// One lane's `f32_gemv` partial: `w[32·it + lane] · x[32·it + lane]` by
    /// `mul_add` from 0, `it` ascending.
    fn gemv_lane(w: &[f32], x: &[f32], lane: usize) -> f32 {
        let mut f = 0.0f32;
        let mut i = lane;
        while i < x.len() {
            f = w[i].mul_add(x[i], f);
            i += 32;
        }
        f
    }

    /// The top [`N_USED`] of `key`, descending, an equal key to the larger id.
    fn top3<T: Copy>(key: &[T], cmp: impl Fn(&T, &T) -> std::cmp::Ordering) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..key.len() as u32).collect();
        idx.sort_by(|&a, &b| cmp(&key[b as usize], &key[a as usize]).then(b.cmp(&a)));
        idx.truncate(N_USED);
        idx
    }

    /// The weights rule on scores `p` at `ids`: summed in f64 in slot order
    /// and narrowed, each divided by the sum's [`renorm_divisor`] when
    /// `norm`, then scaled.
    fn weights_rule(p: &[f32], ids: &[u32], scale: f32, norm: bool) -> Vec<f32> {
        let g: Vec<f32> = ids.iter().map(|&i| p[i as usize]).collect();
        let div = renorm_divisor(g.iter().fold(0.0f64, |s, &v| s + f64::from(v)) as f32);
        g.iter()
            .map(|&v| (if norm { v / div } else { v }) * scale)
            .collect()
    }

    /// Distance in f32 ulps (finite values of one sign).
    fn ulps(a: f32, b: f32) -> u32 {
        a.to_bits().abs_diff(b.to_bits())
    }

    /// The router on `n_tok` tokens of `x`: logits, scores, ids, weights,
    /// tickets after.
    #[allow(clippy::type_complexity, reason = "one gate helper's four readbacks")]
    fn run_router(
        cx: &Ctx<'_>,
        r: &Router,
        w: &DeviceTensor<f32>,
        bias: &DeviceBuffer<f32>,
        x: &[f32],
        n_tok: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>, u32), GateError> {
        let s = cx.gpu.stream();
        let xd = DeviceBuffer::from_host(s, x)?;
        let mut out = DraftRouterOut::new(s, n_tok)?;
        for tok in 0..n_tok {
            let a = RouterArgs {
                w,
                x: &xd,
                bias,
                scale: r.scale,
                norm: r.norm,
                tok,
            };
            cx.dk.enqueue_router(s, &a, &mut out)?;
        }
        s.synchronize()?;
        Ok((
            out.logits.to_host_vec(s)?,
            out.probs.to_host_vec(s)?,
            out.ids.to_host_vec(s)?,
            out.weights.to_host_vec(s)?,
            out.tickets(s)?,
        ))
    }

    fn check_router(cx: &Ctx<'_>, r: &Router) -> Result<bool, GateError> {
        let k = r.k;
        let x = seeded(k * ROUTER_VECTORS, 101, -2.0, 4.0);
        let (lk, pk, idk, wk, tickets) =
            run_router(cx, r, &r.dev, &r.bias_dev, &x, ROUTER_VECTORS)?;
        let (lk2, pk2, idk2, wk2, _) = run_router(cx, r, &r.dev, &r.bias_dev, &x, ROUTER_VECTORS)?;
        let rerun =
            bits_equal(&lk, &lk2) && bits_equal(&pk, &pk2) && idk == idk2 && bits_equal(&wk, &wk2);
        let (mut logits_ok, mut rule_ok, mut libm_ids, mut max_ulps) =
            (0usize, 0usize, 0usize, 0u32);
        for t in 0..ROUTER_VECTORS {
            let xt = &x[t * k..(t + 1) * k];
            let lh: Vec<f32> = (0..N_EXPERT)
                .map(|e| {
                    butterfly(std::array::from_fn(|lane| {
                        gemv_lane(&r.w[e * k..(e + 1) * k], xt, lane)
                    }))
                })
                .collect();
            let (lt, pt) = (
                &lk[t * N_EXPERT..(t + 1) * N_EXPERT],
                &pk[t * N_EXPERT..(t + 1) * N_EXPERT],
            );
            let (it, wt) = (
                &idk[t * N_USED..(t + 1) * N_USED],
                &wk[t * N_USED..(t + 1) * N_USED],
            );
            logits_ok += usize::from(bits_equal(lt, &lh));
            let biased: Vec<f32> = pt.iter().zip(&r.bias).map(|(&p, &b)| p + b).collect();
            let ids = top3(&biased, f32::total_cmp);
            rule_ok +=
                usize::from(ids == it && bits_equal(wt, &weights_rule(pt, it, r.scale, r.norm)));
            let ph: Vec<f32> = lt.iter().map(|&l| sqrt_softplus(l)).collect();
            max_ulps = max_ulps.max(
                pt.iter()
                    .zip(&ph)
                    .map(|(&a, &b)| ulps(a, b))
                    .max()
                    .unwrap_or(0),
            );
            let bh: Vec<f32> = ph.iter().zip(&r.bias).map(|(&p, &b)| p + b).collect();
            libm_ids += usize::from(top3(&bh, f32::total_cmp) == it);
        }
        let pass =
            logits_ok == ROUTER_VECTORS && rule_ok == ROUTER_VECTORS && tickets == 0 && rerun;
        println!(
            "router vectors={ROUTER_VECTORS} K={k} logits_bit_identical_f32_gemv={logits_ok}/{ROUTER_VECTORS} \
             ids+weights_host_rule_on_kernel_scores={rule_ok}/{ROUTER_VECTORS} | scores vs host libm: \
             max {max_ulps} ulp, ids equal on host scores {libm_ids}/{ROUTER_VECTORS} (printed) | \
             tickets={tickets} rerun_bit_identical={rerun} scale={} norm={} {}",
            r.scale,
            r.norm,
            verdict(pass)
        );
        Ok(pass && check_ties(cx, r)?)
    }

    /// `ln(1 + e^x)` in f64 (the exact key of a planted logit).
    fn softplus64(x: f32) -> f64 {
        let x = f64::from(x);
        if x > 20.0 { x } else { x.exp().ln_1p() }
    }

    /// Planted ties: logit `v_e` in column 0 of a 32-wide weight, `x = e_0`,
    /// so every logit is exact; equal logits must go to the larger id, and
    /// the weights are finite in every case, `all_underflow` too, whose every
    /// score is 0 (a logit of −30 is below the −16.6 where `1 + e^x` rounds
    /// to 1).
    fn check_ties(cx: &Ctx<'_>, r: &Router) -> Result<bool, GateError> {
        const K: usize = 32;
        let s = cx.gpu.stream();
        let base: Vec<f32> = (0..N_EXPERT).map(|e| -3.0 + 0.01 * e as f32).collect();
        let planted = |pairs: &[(usize, f32)]| {
            let mut v = base.clone();
            for &(i, val) in pairs {
                v[i] = val;
            }
            v
        };
        let zero = vec![0.0f32; N_EXPERT];
        let mut bias_decides = zero.clone();
        bias_decides[7] = 0.5;
        bias_decides[77] = 0.5;
        let cases: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
            (
                "one_lane_in_top3",
                planted(&[(10, 10.0), (42, 10.0)]),
                zero.clone(),
            ),
            (
                "boundary_3_4",
                planted(&[(100, 6.0), (101, 6.1), (50, 5.0), (70, 5.0)]),
                zero.clone(),
            ),
            (
                "adjacent_lanes",
                planted(&[(64, 7.0), (65, 7.0)]),
                zero.clone(),
            ),
            ("ends", planted(&[(0, 7.0), (127, 7.0)]), zero.clone()),
            (
                "three_way",
                planted(&[(5, 7.0), (37, 7.0), (100, 7.0), (99, 7.0)]),
                zero.clone(),
            ),
            ("all_equal", vec![1.0; N_EXPERT], zero.clone()),
            ("bias_decides", vec![1.0; N_EXPERT], bias_decides),
            ("all_underflow", vec![-30.0; N_EXPERT], zero.clone()),
        ];
        let mut x = vec![0.0f32; K];
        x[0] = 1.0;
        let mut all = true;
        for (name, logits, bias) in cases {
            let mut w = vec![0.0f32; N_EXPERT * K];
            for (e, &v) in logits.iter().enumerate() {
                w[e * K] = v;
            }
            let wd = DeviceTensor::upload(s, &w, N_EXPERT, K)?;
            let bd = DeviceBuffer::from_host(s, &bias)?;
            let (lk, pk, ids, wk, tickets) = run_router(cx, r, &wd, &bd, &x, 1)?;
            let key: Vec<f64> = logits
                .iter()
                .zip(&bias)
                .map(|(&l, &b)| softplus64(l) + f64::from(b))
                .collect();
            let stated = top3(&key, f64::total_cmp);
            let biased: Vec<f32> = pk.iter().zip(&bias).map(|(&p, &b)| p + b).collect();
            let host = top3(&biased, f32::total_cmp);
            let w_ok = bits_equal(&wk, &weights_rule(&pk, &ids, r.scale, r.norm));
            let finite = wk.iter().all(|w| w.is_finite());
            let pass = ids == stated
                && ids == host
                && w_ok
                && finite
                && bits_equal(&lk, &logits)
                && tickets == 0;
            println!(
                "ties case={name} ids={ids:?} stated={stated:?} host={host:?} weights_exact={w_ok} \
                 weights_finite={finite} tickets={tickets} {}",
                verdict(pass)
            );
            all &= pass;
        }
        Ok(all)
    }

    // ------------------------------------------------- oracle diagnostics

    /// A block-0 block-graph row of the draft set's manifest: `tensor`,
    /// `input` and `int` rows (`tools/ref/dump_draft.cpp`).
    struct DRow {
        kind: String,
        name: String,
        occ: u32,
        ne: [usize; 4],
        logical: bool,
        file: String,
    }

    struct DSet {
        dir: PathBuf,
        rows: Vec<DRow>,
    }

    fn read_set(dir: &Path) -> Result<DSet, GateError> {
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv"))
            .map_err(|e| format!("{}/MANIFEST.tsv: {e}", dir.display()))?;
        let build = text
            .lines()
            .find_map(|l| l.strip_prefix("# build\t"))
            .unwrap_or("");
        if !build.contains(IK_BUILD) {
            return Err(format!(
                "{}: # build {build:?} does not name {IK_BUILD} — stale set",
                dir.display()
            )
            .into());
        }
        let mut rows = Vec::new();
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = l.split('\t').collect();
            let u = |i: usize| f[i].parse::<usize>().unwrap_or(0);
            match f[0] {
                "tensor" | "input" if f.len() == 19 && f[15] == "0" && f[18] == "block" => {
                    rows.push(DRow {
                        kind: f[0].to_string(),
                        name: f[1].to_string(),
                        occ: f[2].parse()?,
                        ne: [u(4), u(5), u(6), u(7)],
                        logical: f[12] == "1",
                        file: String::new(),
                    });
                }
                "int" if f.len() == 16 && f[12] == "0" && f[15] == "block" => rows.push(DRow {
                    kind: "int".to_string(),
                    name: f[1].to_string(),
                    occ: f[2].parse()?,
                    ne: [u(7), 1, 1, 1],
                    logical: f[6] == "logical",
                    file: f[11].to_string(),
                }),
                _ => {}
            }
        }
        Ok(DSet {
            dir: dir.to_path_buf(),
            rows,
        })
    }

    impl DSet {
        fn f32_row(&self, name: &str) -> Result<(Vec<f32>, [usize; 4]), GateError> {
            let r = self
                .rows
                .iter()
                .find(|r| r.kind == "tensor" && r.name == name && r.occ == 0)
                .ok_or_else(|| format!("the set has no tensor row {name}"))?;
            let logical = if r.logical { ".logical" } else { "" };
            let path = self
                .dir
                .join(format!("b0.{}.{}{logical}.f32", dump_stem(&r.name), r.occ));
            let v = f32s(&std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?);
            if v.len() != r.ne.iter().product::<usize>() {
                return Err(format!(
                    "{}: {} values, manifest ne {:?}",
                    path.display(),
                    v.len(),
                    r.ne
                )
                .into());
            }
            Ok((v, r.ne))
        }

        fn i32_logical(&self, name: &str) -> Result<Vec<u32>, GateError> {
            let r = self
                .rows
                .iter()
                .find(|r| r.kind == "int" && r.name == name && r.occ == 0 && r.logical)
                .ok_or_else(|| format!("the set has no logical int row {name}"))?;
            let path = self.dir.join(&r.file);
            let b = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let v: Vec<u32> = b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c) as u32)
                .collect();
            if v.len() != r.ne[0] {
                return Err(
                    format!("{}: {} ids, manifest {}", path.display(), v.len(), r.ne[0]).into(),
                );
            }
            Ok(v)
        }
    }

    fn oracle(cx: &Ctx<'_>, r: &Router) -> Result<(), GateError> {
        let dir = data_dir().join("ref-draft").join(DSREF_SET);
        let set = read_set(&dir)?;
        let (x, ne) = set.f32_row("node_44")?;
        let t = ne[1];
        println!(
            "oracle set {} block 0 layer 0: {} block-graph rows; FFN input node_44 {}x{t}; the set holds \
             ffn_moe_logits/probs/topk/weights_scaled, ffn_moe_gate_par (h), ffn_moe_down (per slot), \
             ffn_moe_weighted (combine) — diagnostic, not pinned",
            dir.display(),
            set.rows.len(),
            ne[0]
        );
        let (ik_l, _) = set.f32_row("ffn_moe_logits-0")?;
        let (ik_p, _) = set.f32_row("ffn_moe_probs-0")?;
        let ik_ids = set.i32_logical("ffn_moe_topk-0")?;
        let (ik_w, _) = set.f32_row("ffn_moe_weights_scaled-0")?;
        let (ik_h, _) = set.f32_row("ffn_moe_gate_par-0")?;
        let (ik_d, _) = set.f32_row("ffn_moe_down-0")?;
        let (ik_y, _) = set.f32_row("ffn_moe_weighted-0")?;

        let (lk, pk, idk, wk, _) = run_router(cx, r, &r.dev, &r.bias_dev, &x, t)?;
        println!(
            "oracle router T={t}: logits max|diff| {:e} (ik max|logit| {:e}); scores max|diff| {:e}; \
             top-3 ours {idk:?} ik {ik_ids:?} agree={}; weights ours {wk:?} ik {ik_w:?} max|diff| {:e}",
            max_abs_diff(&lk, &ik_l),
            max_abs(&ik_l),
            max_abs_diff(&pk, &ik_p),
            idk == ik_ids,
            max_abs_diff(&wk, &ik_w)
        );

        let (ff, n) = (cx.gate.rows_per_expert, cx.down.rows_per_expert);
        let plan = |sel: &[u32], wts: &[f32]| Plan {
            sel: sel.to_vec(),
            route: concat_route(t),
            wts: wts.to_vec(),
            m: t,
        };
        let ik_plan = plan(&ik_ids, &ik_w);
        let run = run_plan(cx, &ik_plan, &x)?;
        // Ours in ik's [token][slot][row] order: slot 3t + u, column t.
        let ours_h: Vec<f32> = (0..t * N_USED)
            .flat_map(|j| {
                let c = j / N_USED;
                run.h[(j * t + c) * ff..(j * t + c + 1) * ff].to_vec()
            })
            .collect();
        println!(
            "oracle experts at ik's ids and weights: h max|diff| {:e} (ik max|h| {:e}); combine max|diff| {:e} \
             (ik max|y| {:e})",
            max_abs_diff(&ours_h, &ik_h),
            max_abs(&ik_h),
            max_abs_diff(&run.out, &ik_y),
            max_abs(&ik_y)
        );
        // Per slot: route only rank u, weight 1, so the combine is the dot.
        let mut per_slot = vec![0.0f32; t * N_USED * n];
        for u in 0..N_USED {
            let mut route = vec![NO_SLOT; t * N_USED];
            for c in 0..t {
                route[c * N_USED + u] = (c * N_USED + u) as u32;
            }
            let p = Plan {
                sel: ik_ids.clone(),
                route,
                wts: vec![1.0; t * N_USED],
                m: t,
            };
            let rs = run_plan(cx, &p, &x)?;
            for c in 0..t {
                per_slot[(c * N_USED + u) * n..(c * N_USED + u + 1) * n]
                    .copy_from_slice(&rs.out[c * n..(c + 1) * n]);
            }
        }
        println!(
            "oracle down per slot at ik's ids: max|diff| {:e} (ik max|down| {:e})",
            max_abs_diff(&per_slot, &ik_d),
            max_abs(&ik_d)
        );
        // Our down on ik's own h (in our layout: slot 3t + u, column t), per slot.
        let mut h_ik = vec![0.0f32; t * N_USED * t * ff];
        for j in 0..t * N_USED {
            let c = j / N_USED;
            h_ik[(j * t + c) * ff..(j * t + c + 1) * ff]
                .copy_from_slice(&ik_h[j * ff..(j + 1) * ff]);
        }
        let mut per_slot_ik_h = vec![0.0f32; t * N_USED * n];
        for u in 0..N_USED {
            let mut route = vec![NO_SLOT; t * N_USED];
            for c in 0..t {
                route[c * N_USED + u] = (c * N_USED + u) as u32;
            }
            let p = Plan {
                sel: ik_ids.clone(),
                route,
                wts: vec![1.0; t * N_USED],
                m: t,
            };
            let y = run_down(cx, &p, &h_ik)?;
            for c in 0..t {
                per_slot_ik_h[(c * N_USED + u) * n..(c * N_USED + u + 1) * n]
                    .copy_from_slice(&y[c * n..(c + 1) * n]);
            }
        }
        println!(
            "oracle down per slot on ik's h at ik's ids: max|diff| {:e} (ik max|down| {:e})",
            max_abs_diff(&per_slot_ik_h, &ik_d),
            max_abs(&ik_d)
        );
        // Both per-slot downs against the exact product on ik's h.
        let exact: Vec<f32> = par(t * N_USED * n, |i| {
            let (j, d) = (i / n, i % n);
            let w = cx.down.row_f32(ik_ids[j], d);
            let hv = &ik_h[j * ff..(j + 1) * ff];
            w.iter()
                .zip(hv)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>() as f32
        });
        println!(
            "oracle down per slot on ik's h vs the exact f64 product: ours max|diff| {:e}, ik max|diff| {:e}",
            max_abs_diff(&per_slot_ik_h, &exact),
            max_abs_diff(&ik_d, &exact)
        );
        let own = run_plan(cx, &plan(&idk, &wk), &x)?;
        println!(
            "oracle chain, our router's ids and weights: combine max|diff| {:e} vs ik's ffn_moe_weighted — \
             diagnostic, not pinned",
            max_abs_diff(&own.out, &ik_y)
        );
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let path = std::env::var_os("BLOOMERY_DSPARK_MODEL")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .ok_or(
                "BLOOMERY_DSPARK_MODEL unset — run through `just gate-gpu-dspark-experts`, which exports \
                 it from the V4.1 profile's DSPARK_MODEL",
            )?;
        let split = Split::open(&path)?;
        let hp = DraftHparams::read(&split)?;
        let ex = hp.experts;
        if ex.n_expert != N_EXPERT || ex.n_used != N_USED {
            return Err(format!(
                "{} experts, {} used: the kernels take {N_EXPERT} / {N_USED}",
                ex.n_expert, ex.n_used
            )
            .into());
        }
        let (k, ff) = (hp.n_embd, ex.ff);
        println!(
            "draft {}: n_embd {k} ff {ff} experts {} used {} scale {} norm {} swiglu_limit[0] {}",
            path.display(),
            ex.n_expert,
            ex.n_used,
            ex.routed_scale,
            ex.weights_norm,
            hp.swiglu_limit[0]
        );
        let gpu = Gpu::new()?;
        let dk = DraftExpertKernels::load(gpu.context())?;
        let s = gpu.stream();
        let e = N_EXPERT as u64;
        let cx = Ctx {
            gpu: &gpu,
            dk: &dk,
            gate: load_stack(&split, s, &names::ffn_gate_exps(0), k, ff)?,
            up: load_stack(&split, s, &names::ffn_up_exps(0), k, ff)?,
            down: load_stack(&split, s, &names::ffn_down_exps(0), ff, k)?,
            limit: hp.swiglu_limit[0],
        };
        let wb = tensor_bytes(
            &split,
            &names::ffn_gate_inp(0),
            GgmlType::BF16,
            &[k as u64, e],
        )?;
        let w: Vec<f32> = wb
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
            .collect();
        let bias = f32s(tensor_bytes(
            &split,
            &names::exp_probs_b(0),
            GgmlType::F32,
            &[e],
        )?);
        let router = Router {
            dev: DeviceTensor::upload(s, &w, N_EXPERT, k)?,
            bias_dev: DeviceBuffer::from_host(s, &bias)?,
            w,
            bias,
            k,
            scale: ex.routed_scale,
            norm: ex.weights_norm,
        };

        let mut ok = true;
        for (i, &m) in MS.iter().enumerate() {
            let x = seeded(k * m, 11 + i as u32, -2.0, 4.0);
            let p = Plan {
                sel: SLOTS.to_vec(),
                route: (0..m).flat_map(|_| 0..N_USED as u32).collect(),
                wts: seeded(N_USED * m, 21 + i as u32, 0.2, 1.3),
                m,
            };
            ok &= check_plan(&cx, "all3", &p, &x)?;
        }
        // Three tokens routed by the router itself: sel = its ids, one slot each.
        let x = seeded(k * 3, 31, -2.0, 4.0);
        let (_, _, ids, wts, _) = run_router(&cx, &router, &router.dev, &router.bias_dev, &x, 3)?;
        let p = Plan {
            sel: ids,
            route: concat_route(3),
            wts,
            m: 3,
        };
        ok &= check_plan(&cx, "concat", &p, &x)?;
        ok &= check_router(&cx, &router)?;

        oracle(&cx, &router)?;

        if ok {
            println!(
                "PASSED: {NAME} — dflash_expert_gate_up and dflash_expert_down (slots 0/64/127 at m 1/3/8, \
                 a routed m=3 plan) bit-identical to the host rule and inside the q8_1 bound and pins; \
                 dflash_router (16 vectors, 7 tie cases) bit-identical to the host rule; oracle rows printed, not pinned"
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
