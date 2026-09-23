//! GPU gate for DeepSeek-V4.1's attention output projection — B4 op block I
//! (`docs/research/v41-b4-plan-report.md` §1-I) — against ik's CPU dump.
//! `attn_output_a` is block-diagonal: group g multiplies weight rows
//! `g·rank ..` with its slice of the attention output, heads
//! `g·(n_head/groups) ..` (`build_deepseek4.cpp:1414-1428`). The engine runs
//! all groups as ONE launch of `q8_0_gemv_heads`, a head per group: x window
//! `g·group_k`, output span `g·rank`. That output order is the order
//! `attn_output_b` reads, so wo_b is one plain `q8_0_gemv` on it.
//!
//! Sites: `attn_wo_a-L` and `attn_out-L` at every layer of every set — the
//! 5-token prefill (`Set::Cpu`, T = 5, where ik copies the permuted groups
//! with `ggml_cont_2d`) and the table's decode-step sets (`step_sets`, T = 1,
//! where it only reshapes them, `build_deepseek4.cpp:1430-1434`). The engine's decode
//! shape is m = 1, so a T-token site is T launches of that shape. Every
//! input is the dump's own: `attn-L`, the ROPE_BACK output, for wo_a; the
//! row `attn_out-L` reads for wo_b.
//!
//! Per site, the B4 gate form:
//! 1. the kernel against this binary's transcription of our rule — f32
//!    activations; lane L fuses a multiply-add per value of code words L,
//!    L+32, … in increasing order, each word in byte order; lane 0 of the
//!    xor butterfly (`bloomery_gpu::q8f32` module doc) — bit-identical, and
//!    a rerun bit-identical;
//! 2. ik's rule (`bloomery_gpu_gates::ik_q8_2`) against the dump,
//!    bit-identical: the activations quantized to q8_2 blocks, then the AVX2
//!    Q8_0 × Q8_2 dot. With the two views proven against the dump's
//!    own view rows bit for bit — wo_a's input is the permuted groups of
//!    `attn-L`, wo_b's the permuted `attn_wo_a-L` — this proves which rows
//!    meet which slice of the attention output;
//! 3. the kernel against the dump: `ik_rel` printed, and every value inside
//!    its own band ([`row_ref`]), the two rules' distance bounded through
//!    ik's reconstruction of the activations, which layer 2 proves.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_woa: built without the `deepseek41` feature; see `just gate-gpu-ds41-woa`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_woa", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::model::{Q8_0GemvHeadsArgs, StepKernels};
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{DeviceTensor, Gpu};
    use bloomery_gpu_gates::ik_q8_2::{self, QK, folded, half_sum};
    use bloomery_gpu_gates::oracle::{self, Set};
    use bloomery_gpu_gates::{
        GateError, RefManifest, RefRow, bits_equal, checks_failed, max_rel_err, ref_model_path,
        ref_tensor_logical_in, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{GgmlType, Q8Block, half_to_f32};
    use model::arch::Arch;

    /// f32's unit roundoff, 2^-24.
    const U: f64 = f32::EPSILON as f64 / 2.0;

    /// Host threads for the two rules' transcriptions.
    const HOST_THREADS: usize = 16;

    /// `γ(n) = n·u / (1 − n·u)`: the relative bound on a term that went
    /// through `n` roundings of its partial sums.
    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    /// A Q8_0 weight twice: the file's blocks for the host rules, and the
    /// q8f32 device planes the kernels read.
    struct Q8 {
        /// Row-major, `k/32` per row.
        blocks: Vec<Q8Block>,
        /// Values per row.
        k: usize,
        rows: usize,
        qs: DeviceTensor<u32>,
        d: DeviceTensor<u16>,
    }

    /// Q8_0 tensor `name` of the model file, `[k, rows]` in ggml order, and
    /// its q8f32 planes on the device, packed by the loader's own
    /// `q8_0_planes`.
    fn load_q8(split: &Split, stream: &CudaStream, name: &str) -> Result<Q8, GateError> {
        let (s, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the model file"))?;
        let [k, rows] = match t.dims.as_slice() {
            &[k, rows] if t.ty == GgmlType::Q8_0 && k.is_multiple_of(QK as u64) => {
                [usize::try_from(k)?, usize::try_from(rows)?]
            }
            _ => return Err(format!("{name} is {:?} {:?}, want a 2-D Q8_0", t.ty, t.dims).into()),
        };
        let bytes = split.shard(s).ok_or("shard index out of range")?.data(t)?;
        let blocks: Vec<Q8Block> = bytes
            .as_chunks::<34>()
            .0
            .iter()
            .map(Q8Block::from_bytes)
            .collect();
        if blocks.len() != rows * k / QK {
            return Err(format!("{name}: {} blocks, want {}", blocks.len(), rows * k / QK).into());
        }
        let (qs, d) = q8_0_planes(&blocks);
        Ok(Q8 {
            qs: DeviceTensor::upload(stream, &qs, rows, k / 4)?,
            d: DeviceTensor::upload(stream, &d, rows, k / QK)?,
            blocks,
            k,
            rows,
        })
    }

    // -------------------------------------------------- our rule, transcribed

    /// Our rule on one row: lane L of 32 multiply-adds `code·scale` (exact in
    /// f32) by the activation, one fused multiply-add per value, over code
    /// words L, L+32, … in increasing order, each word's four values in byte
    /// order; lane 0 of the xor butterfly is the output.
    fn ours(blocks: &[Q8Block], x: &[f32]) -> f32 {
        let words = x.len() / 4;
        let mut lanes = [0.0f32; 32];
        for (lane, f) in lanes.iter_mut().enumerate() {
            for w in (lane..words).step_by(32) {
                let b = &blocks[w / 8];
                let d = half_to_f32(b.d);
                let c0 = 4 * w % QK;
                for (&c, &xv) in b.q[c0..c0 + 4].iter().zip(&x[4 * w..4 * w + 4]) {
                    *f = (f32::from(c) * d).mul_add(xv, *f);
                }
            }
        }
        butterfly(lanes)
    }

    /// The xor butterfly `warp::reduce_sum_f32` runs (16, 8, 4, 2, 1; each
    /// lane adds its partner to itself): lane 0's result.
    fn butterfly(mut v: [f32; 32]) -> f32 {
        for off in [16, 8, 4, 2, 1] {
            let p = v;
            for (l, s) in v.iter_mut().enumerate() {
                *s = p[l] + p[l ^ off];
            }
        }
        v[0]
    }

    // ------------------------------------------------------ rows and the band

    /// One output row: both rules in f32, the gap between their exact
    /// values, and the band.
    #[derive(Clone, Copy, Default)]
    struct RowRef {
        ours: f32,
        ik: f32,
        /// `|ours − ik|` in exact arithmetic (f64, whose sums round each term
        /// by at most 2^-53 of it — below every printed digit).
        gap: f64,
        band: f64,
        /// Terms where ik's sign fold wraps: activation code −128 under a
        /// negative weight code.
        wraps: u32,
    }

    /// Both rules on one row, and its band: how far our kernel may sit from
    /// ik's value. Per value v, ik multiplies the weight `w_v` by its
    /// reconstruction `x̂_v = sign(w_v)·folded_v·d_x` of the activation (the
    /// sign fold's wrap included — `ik_q8_2::folded`) where ours multiplies by
    /// `x_v`, so the exact sums differ by `|Σ w_v (x_v − x̂_v)| <= Σ |w_v|·|x_v
    /// − x̂_v|`. Each side's f32 accumulation adds its roundings: ours at most
    /// `γ(k/32 + 5)` of `Σ|w_v x_v|` — k/32 multiply-adds along a lane, five
    /// butterfly levels — and ik's `γ(k/128 + 3)` of `Σ|d_w d_x p|` over its
    /// partials — k/128 multiply-adds along a lane, three levels of
    /// `hsum_float_8`. The codes' products with their scales, the partials
    /// and the scale products are exact, so nothing else rounds. `x̂` comes
    /// from the q8_2 codes layer 2 proves, never from our own rule.
    fn row_ref(blocks: &[Q8Block], x: &[f32], xq: &[i8], xd: &[f32]) -> RowRef {
        let k = x.len();
        let (mut ours_exact, mut ik_exact) = (0.0f64, 0.0f64);
        let (mut abs_ours, mut abs_ik, mut act) = (0.0f64, 0.0f64, 0.0f64);
        let mut wraps = 0u32;
        for (b, blk) in blocks.iter().enumerate() {
            let (dw, dx) = (half_to_f32(blk.d), xd[b]);
            let a = &xq[b * QK..(b + 1) * QK];
            for h in 0..2 {
                let t = f64::from(dw * dx) * f64::from(half_sum(blk, a, h));
                ik_exact += t;
                abs_ik += t.abs();
            }
            for (j, &c) in blk.q.iter().enumerate() {
                let wv = f64::from(f32::from(c) * dw);
                let xv = f64::from(x[b * QK + j]);
                ours_exact += wv * xv;
                abs_ours += (wv * xv).abs();
                let xh = f64::from(c.signum()) * f64::from(folded(c, a[j])) * f64::from(dx);
                act += wv.abs() * (xv - xh).abs();
                wraps += u32::from(c < 0 && a[j] == i8::MIN);
            }
        }
        RowRef {
            ours: ours(blocks, x),
            ik: ik_q8_2::dot(blocks, xq, xd),
            gap: (ours_exact - ik_exact).abs(),
            band: act + gamma(k / QK + 5) * abs_ours + gamma(k / (4 * QK) + 3) * abs_ik,
            wraps,
        }
    }

    /// [`row_ref`] for every row of `w` against one token, row r reading the
    /// window `r / rows_per_window` of `x` (`w.k` values) and its q8_2 blocks.
    fn host_rows(w: &Q8, rows_per_window: usize, x: &[f32]) -> Vec<RowRef> {
        let (k, nb) = (w.k, w.k / QK);
        let (xq, xd) = ik_q8_2::quantize(x);
        let (xq, xd) = (&xq[..], &xd[..]);
        let mut out = vec![RowRef::default(); w.rows];
        let chunk = w.rows.div_ceil(HOST_THREADS);
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(chunk).enumerate() {
                s.spawn(move || {
                    for (i, o) in part.iter_mut().enumerate() {
                        let r = c * chunk + i;
                        let x0 = r / rows_per_window * k;
                        *o = row_ref(
                            &w.blocks[r * nb..(r + 1) * nb],
                            &x[x0..x0 + k],
                            &xq[x0..x0 + k],
                            &xd[x0 / QK..x0 / QK + nb],
                        );
                    }
                });
            }
        });
        out
    }

    // --------------------------------------------------------------- the sets

    /// A reader's src column is `want`.
    fn reads(row: &RefRow, src: Option<&str>, want: &str) -> Result<(), GateError> {
        if src != Some(want) {
            return Err(format!("{} reads {src:?}, want {want:?}", row.name).into());
        }
        Ok(())
    }

    fn n(v: usize) -> u64 {
        v as u64
    }

    /// The shapes of one layer's projection (`build_deepseek4.cpp:1414-1420`):
    /// `group_k` is wo_a's row width, `groups` the attention output's width
    /// over it, `rank` wo_b's row width over the groups (wo_a's rows per
    /// group), `embd` wo_b's rows.
    #[derive(Clone, Copy, Debug)]
    struct Geom {
        /// Values of one token's attention output.
        width: usize,
        group_k: usize,
        groups: usize,
        rank: usize,
        embd: usize,
    }

    /// One layer's inputs and dumped outputs in one set, every view proven:
    /// token-major, in the kernels' layouts.
    struct SiteData {
        t: usize,
        /// `attn-L`: T rows of `width`.
        attn: Vec<f32>,
        /// `attn_wo_a-L` in the heads kernel's order: T rows of
        /// `groups·rank`, group g's outputs at `g·rank`.
        wo_a: Vec<f32>,
        /// The row `attn_out-L` reads: T rows of `groups·rank`.
        wo_b_in: Vec<f32>,
        /// `attn_out-L`: T rows of `embd`.
        wo_b: Vec<f32>,
        /// wo_a's input, gathered from `attn`, equals the dump's permuted
        /// view bit for bit.
        view_a: bool,
        /// wo_b's input equals `wo_a` bit for bit: the heads kernel's output
        /// order is the order wo_b reads.
        view_b: bool,
        /// wo_b's input row's op: RESHAPE at T = 1, CONT above.
        branch: String,
    }

    /// Read layer `l` of `man`: the chain `attn-L` → reshape → permute →
    /// `attn_wo_a-L` → permute → cont (T > 1) or reshape (T = 1) →
    /// `attn_out-L`, each row's op, shape and src checked.
    fn site_data(
        man: &RefManifest,
        l: usize,
        a: &Q8,
        b: &Q8,
    ) -> Result<(Geom, SiteData), GateError> {
        let dir = &man.dir;
        let name_a = format!("attn_wo_a-{l}");
        let (at_a, row_a) = man.tensor_at(&name_a, 0)?;
        reads(
            row_a,
            row_a.src0.as_deref(),
            &format!("blk.{l}.attn_output_a.weight (reshaped)"),
        )?;
        let (at_p, perm) = man.last_before(at_a, row_a.src1.as_deref())?;
        let (at_r, resh) = man.last_before(at_p, perm.src0.as_deref())?;
        let (_, attn_row) = man.last_before(at_r, resh.src0.as_deref())?;
        let [hd, nh, t, one] = attn_row.ne.map(|v| v as usize);
        if one != 1 || attn_row.op != "ROPE_BACK" || attn_row.name != format!("attn-{l}") {
            return Err(format!(
                "{name_a} reads {:?} ({} {:?}), want attn-{l} ROPE_BACK [head, heads, tokens, 1]",
                attn_row.name, attn_row.op, attn_row.ne
            )
            .into());
        }
        let width = hd * nh;
        if a.k == 0
            || !width.is_multiple_of(a.k)
            || !a.rows.is_multiple_of(width / a.k)
            || b.k != a.rows
        {
            return Err(format!(
                "layer {l}: attention width {width}, wo_a [{}, {}], wo_b [{}, {}] do not group",
                a.k, a.rows, b.k, b.rows
            )
            .into());
        }
        let g = Geom {
            width,
            group_k: a.k,
            groups: width / a.k,
            rank: a.rows / (width / a.k),
            embd: b.rows,
        };
        resh.expect(
            &name_a,
            "f32",
            [n(g.group_k), n(g.groups), n(t), 1],
            "RESHAPE",
        )?;
        perm.expect(
            &name_a,
            "f32",
            [n(g.group_k), n(t), n(g.groups), 1],
            "PERMUTE",
        )?;
        row_a.expect(&name_a, "f32", [n(g.rank), n(t), n(g.groups), 1], "MUL_MAT")?;

        let name_b = format!("attn_out-{l}");
        let (at_b, row_b) = man.tensor_at(&name_b, 0)?;
        reads(
            row_b,
            row_b.src0.as_deref(),
            &format!("blk.{l}.attn_output_b.weight"),
        )?;
        row_b.expect(&name_b, "f32", [n(g.embd), n(t), 1, 1], "MUL_MAT")?;
        let (at_in, b_in) = man.last_before(at_b, row_b.src1.as_deref())?;
        let branch = if t == 1 { "RESHAPE" } else { "CONT" };
        b_in.expect(&name_b, "f32", [n(g.groups * g.rank), n(t), 1, 1], branch)?;
        let (_, b_perm) = man.last_before(at_in, b_in.src0.as_deref())?;
        b_perm.expect(&name_b, "f32", [n(g.rank), n(g.groups), n(t), 1], "PERMUTE")?;
        reads(b_perm, b_perm.src0.as_deref(), &name_a)?;

        let attn = ref_tensor_logical_in(dir, attn_row)?;
        let perm_v = ref_tensor_logical_in(dir, perm)?;
        let wo_a_dump = ref_tensor_logical_in(dir, row_a)?;
        let wo_b_in = ref_tensor_logical_in(dir, b_in)?;
        let wo_b = ref_tensor_logical_in(dir, row_b)?;

        // wo_a's input: element (i, tok, grp) of the permuted view is value
        // grp·group_k + i of token tok's attention output. wo_a's output:
        // element (j, tok, grp) of the dump is output grp·rank + j of token
        // tok in the heads kernel's order.
        let mut gathered = vec![0.0f32; perm_v.len()];
        let mut wo_a = vec![0.0f32; wo_a_dump.len()];
        for tok in 0..t {
            for grp in 0..g.groups {
                let (dst, src) = (g.group_k * (tok + t * grp), tok * g.width + grp * g.group_k);
                gathered[dst..dst + g.group_k].copy_from_slice(&attn[src..src + g.group_k]);
                let (dst, src) = ((tok * g.groups + grp) * g.rank, g.rank * (tok + t * grp));
                wo_a[dst..dst + g.rank].copy_from_slice(&wo_a_dump[src..src + g.rank]);
            }
        }
        let (view_a, view_b) = (bits_equal(&gathered, &perm_v), bits_equal(&wo_a, &wo_b_in));
        Ok((
            g,
            SiteData {
                t,
                attn,
                wo_a,
                wo_b_in,
                wo_b,
                view_a,
                view_b,
                branch: b_in.op.clone(),
            },
        ))
    }

    // -------------------------------------------------------------- one site

    /// What one site compared, over all its tokens.
    struct Tally {
        values: usize,
        kernel_ours: usize,
        rerun: bool,
        ik_dump: usize,
        over_band: usize,
        /// Largest `|kernel − dump| / band`.
        dev_band: f64,
        max_dump: f64,
        max_gap: f64,
        max_band: f64,
        /// ik's sign-fold wraps over the site ([`RowRef::wraps`]).
        wraps: u32,
        kernel: Vec<f32>,
        dump: Vec<f32>,
    }

    /// One token's run of a kernel: its input in, its outputs out.
    type Launch<'a> = dyn FnMut(&[f32]) -> Result<Vec<f32>, GateError> + 'a;

    /// The three checks of one op over T tokens: `launch` runs the kernel on
    /// one token's input, `rows_per_window` places each weight row's window
    /// in it, and `dump` holds ik's outputs in the kernel's order.
    fn compare(
        w: &Q8,
        rows_per_window: usize,
        inputs: &[f32],
        dump: &[f32],
        launch: &mut Launch<'_>,
    ) -> Result<Tally, GateError> {
        let width = inputs.len() / (dump.len() / w.rows);
        let mut tl = Tally {
            values: 0,
            kernel_ours: 0,
            rerun: true,
            ik_dump: 0,
            over_band: 0,
            dev_band: 0.0,
            max_dump: 0.0,
            max_gap: 0.0,
            max_band: 0.0,
            wraps: 0,
            kernel: Vec::with_capacity(dump.len()),
            dump: dump.to_vec(),
        };
        for (x, want) in inputs.chunks_exact(width).zip(dump.chunks_exact(w.rows)) {
            let y = launch(x)?;
            tl.rerun &= bits_equal(&y, &launch(x)?);
            let refs = host_rows(w, rows_per_window, x);
            for ((&got, &ik_val), r) in y.iter().zip(want).zip(&refs) {
                tl.values += 1;
                tl.kernel_ours += usize::from(got.to_bits() == r.ours.to_bits());
                tl.ik_dump += usize::from(ik_val.to_bits() == r.ik.to_bits());
                let dev = (f64::from(got) - f64::from(ik_val)).abs();
                if dev.is_nan() || dev > r.band {
                    tl.over_band += 1;
                }
                tl.dev_band = tl.dev_band.max(dev / r.band);
                tl.max_dump = tl.max_dump.max(f64::from(ik_val).abs());
                tl.max_gap = tl.max_gap.max(r.gap);
                tl.max_band = tl.max_band.max(r.band);
                tl.wraps += r.wraps;
            }
            tl.kernel.extend(y);
        }
        Ok(tl)
    }

    /// Print one site's verdict line; true when it passes. A kernel output
    /// that is not finite (a slot left at the NaN fill) prints as the error
    /// `max_rel_err` names and fails the site.
    fn report(label: &str, l: usize, op: &str, t: usize, view: bool, tl: &Tally) -> bool {
        let rel = max_rel_err(&tl.kernel, &tl.dump);
        let ik_rel = rel
            .as_ref()
            .map_or_else(|e| format!("[{e}]"), |e| format!("{e:.3e}"));
        let pass = rel.is_ok()
            && tl.kernel_ours == tl.values
            && tl.rerun
            && tl.ik_dump == tl.values
            && view
            && tl.over_band == 0;
        println!(
            "site set={label} L={l} op={op} T={t} values={} kernel_vs_ours_bits={}/{} rerun_bit_identical={} \
             ik_sim_vs_dump_bits={}/{} ik_sign_fold_wraps={} input_view_bit_identical={view} ik_rel={ik_rel} \
             gap_rel={:.3e} band_rel={:.3e} over_band={} max_dev_over_band={:.3} {}",
            tl.values,
            tl.kernel_ours,
            tl.values,
            tl.rerun,
            tl.ik_dump,
            tl.values,
            tl.wraps,
            tl.max_gap / tl.max_dump,
            tl.max_band / tl.max_dump,
            tl.over_band,
            tl.dev_band,
            verdict(pass)
        );
        pass
    }

    /// What every site reads besides its set.
    struct Cx<'a> {
        gpu: &'a Gpu,
        step: &'a StepKernels,
        stream: &'a CudaStream,
    }

    /// wo_a of token `x` through `q8_0_gemv_heads`: a head per group. The
    /// output starts as NaN, so a slot the kernel skips fails the bit check.
    fn run_wo_a(cx: &Cx, w: &Q8, g: Geom, x: &[f32]) -> Result<Vec<f32>, GateError> {
        let x_dev = DeviceBuffer::from_host(cx.stream, x)?;
        let mut y = DeviceBuffer::from_host(cx.stream, &vec![f32::NAN; g.groups * g.rank])?;
        cx.step.enqueue_q8_0_gemv_heads(
            cx.stream,
            Q8_0GemvHeadsArgs {
                qs: &w.qs,
                d: &w.d,
                x: &x_dev,
                rows_per_head: g.rank,
                x_head_stride: g.group_k,
                y_head_stride: g.rank,
                y_off: 0,
                y: &mut y,
            },
        )?;
        cx.stream.synchronize()?;
        Ok(y.to_host_vec(cx.stream)?)
    }

    /// wo_b of token `x` through `q8_0_gemv` at m = 1, NaN-filled as above.
    fn run_wo_b(cx: &Cx, w: &Q8, x: &[f32]) -> Result<Vec<f32>, GateError> {
        let x_dev = DeviceBuffer::from_host(cx.stream, x)?;
        let mut y = DeviceBuffer::from_host(cx.stream, &vec![f32::NAN; w.rows])?;
        cx.gpu
            .q8f32()
            .enqueue_q8_0_gemv(cx.stream, &w.qs, &w.d, &x_dev, 1, &mut y)?;
        cx.stream.synchronize()?;
        Ok(y.to_host_vec(cx.stream)?)
    }

    /// Both ops of layer `l` in one set; returns how many of the two failed.
    fn layer_sites(
        cx: &Cx,
        label: &str,
        man: &RefManifest,
        l: usize,
        a: &Q8,
        b: &Q8,
    ) -> Result<u32, GateError> {
        let (g, d) = site_data(man, l, a, b)?;
        let tl_a = compare(a, g.rank, &d.attn, &d.wo_a, &mut |x| run_wo_a(cx, a, g, x))?;
        let pass_a = report(label, l, "wo_a", d.t, d.view_a, &tl_a);
        let tl_b = compare(b, b.rows, &d.wo_b_in, &d.wo_b, &mut |x| run_wo_b(cx, b, x))?;
        let pass_b = report(
            label,
            l,
            &format!("wo_b({})", d.branch),
            d.t,
            d.view_b,
            &tl_b,
        );
        Ok(u32::from(!pass_a) + u32::from(!pass_b))
    }

    pub fn run() -> Result<(), GateError> {
        let split = Split::open(ref_model_path()?)?;
        let want = Arch::Deepseek41.name();
        if split.architecture() != Some(want) {
            return Err(format!(
                "the model file is {:?}, want {want} — run through `just gate-gpu-ds41-woa`, \
                 which picks the deepseek41 profile",
                split.architecture()
            )
            .into());
        }
        let layers = usize::try_from(
            split
                .arch_get_u64("block_count")
                .ok_or("metadata block_count missing")?,
        )?;
        let gpu = Gpu::new()?;
        let step = StepKernels::load(gpu.context())?;
        let cx = Cx {
            gpu: &gpu,
            step: &step,
            stream: gpu.stream(),
        };
        let cpu = oracle::for_arch(Arch::Deepseek41)?;
        let mut sets = vec![(cpu.set_name(Set::Cpu)?, cpu.open(Set::Cpu)?)];
        for &name in cpu.step_sets {
            sets.push((name, cpu.open_named(name)?));
        }
        // Every wo_a and wo_b node of every set is a site: the layers below
        // must be all of them.
        for (label, man) in &sets {
            for p in ["attn_wo_a-", "attn_out-"] {
                let count = man
                    .tensors
                    .iter()
                    .filter(|r| {
                        r.op == "MUL_MAT"
                            && r.name
                                .strip_prefix(p)
                                .is_some_and(|s| s.parse::<usize>().is_ok())
                    })
                    .count();
                if count != layers {
                    return Err(format!(
                        "{label}: {count} {p}L nodes, the file has {layers} layers"
                    )
                    .into());
                }
            }
            println!(
                "set {label}: {} (build {})",
                man.dir.display(),
                man.build.as_deref().unwrap_or("-")
            );
        }
        println!(
            "gate_deepseek41_woa: device {} — {layers} layers x {} sets, wo_a through q8_0_gemv_heads, \
             wo_b through q8_0_gemv, m = 1 per token",
            gpu.device_name()?,
            sets.len()
        );

        let (mut sites, mut failed) = (0u32, 0u32);
        for l in 0..layers {
            let a = load_q8(&split, cx.stream, &format!("blk.{l}.attn_output_a.weight"))?;
            let b = load_q8(&split, cx.stream, &format!("blk.{l}.attn_output_b.weight"))?;
            if !(a.k / QK).is_multiple_of(4) || !(b.k / QK).is_multiple_of(4) {
                return Err(format!(
                    "layer {l}: ik's x4 path needs a block count divisible by 4, rows are {} and {} values",
                    a.k, b.k
                )
                .into());
            }
            for (label, man) in &sets {
                sites += 2;
                failed += layer_sites(&cx, label, man, l, &a, &b)?;
            }
        }
        let pass = failed == 0;
        println!(
            "gate_deepseek41_woa: {sites} sites (wo_a and wo_b at {layers} layers of {} sets), {failed} failed — {}",
            sets.len(),
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }
}
