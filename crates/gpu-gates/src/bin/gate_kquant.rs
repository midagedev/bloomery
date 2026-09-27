//! GPU gate for the model-free K-quant expert family (`bloomery_gpu::kquant`):
//! the Q5_K down `_sel`, the Q5_K gate·up with its activation rule as a launch
//! argument, and Walk A under both. Synthetic stacks only, no model file: every
//! super-block is random words with a positive normal `d` and `dmin`.
//!
//! 1. band: `q5k_gemv_sel` against the f64 dot of `gguf::quant::dequant_row`'s
//!    rows with the q8_1-dequantized columns (`ref_gemv`), `max_rel_err` within
//!    `KERNEL_BAND`, at the stack shapes (rows × K) 2048 × 512, 512 × 2048,
//!    4096 × 2048, 2048 × 4096 and at K = 256, 768, 1280, 2304 (the partial last
//!    iteration). `dequant_row(Q5_K)` is ggml's `to_float` bit for bit
//!    (`hw_q5k_dequant_matches_ggml` in `crates/qdot/tests/qdot.rs`). With it, the
//!    row probe on one row of each shape: its decode (codes, `cda`, `cdb`) equals
//!    the host's reading of the same bytes bit for bit and its sum is the
//!    launch's value for that row. A shape out of band names its first
//!    sub-block out of band, the probe's term beside the host's.
//! 2. sel: slot `s` is the same launch on an upload of expert `sel[s]` alone
//!    against column `s` alone, bit for bit, with two slots on one expert that
//!    must differ; a rerun is bit-identical; a captured launch replays as one
//!    node and follows ids overwritten between replays; the gate·up over two
//!    tokens is each token's launch alone; the launchers refuse a mismatched
//!    activation, a non-dividing expert size and a mismatched up stack.
//! 3. mcol: the gate binary's m-column entry (`walk::row_dot`) at m = 1..8 is the
//!    down `_sel` of the same m columns (each slot one column, the one-column
//!    walk), bit for bit, at K = 256, 512, 768, 1280, 2048, 2304, 4096, 8192. A split
//!    term over a nonzero sum needs a second pair of iterations (K >= 4096); K = 8192
//!    has three.
//! 4. fault: an id past the stack raises `ExpertId` (unlabelled); the down leaves
//!    its slot as it was, the gate·up writes NaN into its rows; `HOST` raises
//!    nothing and leaves its slot as it was in both; every other slot is the clean
//!    launch's. A NaN in an activation column raises `QuantColumn` from the
//!    quantizer, and every row that reads that column is NaN in both entries.
//! 5. act: the gate·up's rows are the rule on the down `_sel`'s sums of the same
//!    rows and column, bit for bit: `silu_mul` against the engine's elementwise
//!    swiglu (the same core), `swiglu_clamp` at two limits against
//!    `kquant::act::swiglu_clamp` run on the host. The clamped rows are counted.
//! 6. q4k_sibling: the gate binary's Walk A instance over `Q4k` is
//!    `q4k_gemv_sel` bit for bit, `HOST` slot included, at K = 256, 768, 1280,
//!    2048, 2304, 4096.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_kquant: built without the `gpu` feature; see `just gate-gpu-kquant`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_kquant", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::kquant::sel::{ROWS_PER_BLOCK, THREADS, gemv_sel_body};
    use bloomery_gpu::kquant::walk::{iter_term, row_dot, row_dot_1col};
    use bloomery_gpu::kquant::{
        Act, GateUpAct, KquantKernels, Q4k, Q5k, SbDecode, SelDown, act, walk_a_planes,
    };
    use bloomery_gpu::{
        DeviceTensor, Fault, FaultSink, FaultSite, Gpu, GpuError, LAYER_NONE, Q8Act, col_sums,
    };
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, activations, bits_equal, max_rel_err, q8_1_dequant, ref_gemv,
        verdict,
    };
    use cuda_core::{DeviceBuffer, LaunchConfig1D};
    use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
    use cuda_host::cuda_module;
    use gguf::quant::{GgmlType, dequant_row, half_to_f32};

    /// The probe's words a sub-block: eight code words, `cda`, `cdb`, the term.
    const PROBE_SUB: usize = 11;
    /// The probe's words before the sub-blocks: 32 lane partials, the sum.
    const PROBE_HEAD: usize = 33;

    #[cuda_module]
    mod gate_kernels {
        use super::*;

        /// Walk A over Q4_K through the family's `_sel` body: `q4k_gemv_sel`'s
        /// launch contract, block and bounds, so the two are the same code.
        #[allow(
            clippy::too_many_arguments,
            reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
        )]
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                w.len() >= n_experts * rows_per_expert * 36 * n_sb,
                q.len() >= n_slots * 256 * iters,
                s8.len() >= n_slots * 8 * n_sb,
                d8.len() >= n_slots * 2 * n_sb,
                sel.len() >= n_slots,
                y.len() >= n_slots * rows_per_expert
            )
        )]
        pub fn kq_gemv_sel_q4k(
            w: &[u32],
            q: &[u32],
            s8: &[i32],
            d8: &[f32],
            sel: &[u32],
            n_experts: u32,
            rows_per_expert: u32,
            n_slots: u32,
            n_sb: u32,
            iters: u32,
            fault: FaultSink,
            mut y: DisjointSlice<f32>,
        ) {
            let t = thread::index_1d().get() % THREADS as usize;
            let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
            if row >= n_slots as usize * rows_per_expert as usize {
                return;
            }
            // SAFETY: the launch contract is the body's (36 = Q4k::WORDS); the
            // host passes iters = ceil(n_sb/4); the return above is
            // warp-uniform and keeps row in range.
            unsafe {
                gemv_sel_body::<Q4k>(
                    w,
                    q,
                    s8,
                    d8,
                    sel,
                    n_experts,
                    rows_per_expert,
                    n_sb,
                    iters,
                    fault,
                    row,
                    &mut y,
                );
            }
        }

        /// `rows` Q5_K rows against `m_cols` (1..=8) columns through the
        /// m-column walk: `y[c · rows + r]` is row `r` against column `c`.
        #[allow(
            clippy::too_many_arguments,
            reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
        )]
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                w.len() >= rows * 44 * n_sb,
                q.len() >= m_cols * 256 * iters,
                s8.len() >= m_cols * 8 * n_sb,
                d8.len() >= m_cols * 2 * n_sb,
                y.len() >= m_cols * rows,
                m_cols >= 1,
                m_cols <= 8
            )
        )]
        pub fn kq_gemv_mcol_q5k(
            w: &[u32],
            q: &[u32],
            s8: &[i32],
            d8: &[f32],
            rows: u32,
            m_cols: u32,
            n_sb: u32,
            iters: u32,
            mut y: DisjointSlice<f32>,
        ) {
            let t = thread::index_1d().get() % THREADS as usize;
            let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
            if row >= rows as usize {
                return;
            }
            let lane = warp::lane_id() as usize;
            let m = m_cols as usize;
            // SAFETY: row < rows of `w`, columns 0..m_cols of the planes, iters =
            // ceil(n_sb/4) from the host, m_cols in 1..=8 and launch-uniform,
            // and the warp's 32 lanes are here (the return is warp-uniform).
            let f = unsafe { row_dot::<Q5k>(w, q, s8, d8, n_sb as usize, iters, row, 0, m, lane) };
            let v = col_sums(f, m);
            if lane == 0 {
                let r = rows as usize;
                // SAFETY: c·rows + row < m_cols·rows <= y.len() for every c <
                // m_cols; lane 0 of the row's warp is each value's only writer.
                unsafe {
                    *y.get_unchecked_mut(row) = v[0];
                    if m > 1 {
                        *y.get_unchecked_mut(r + row) = v[1];
                    }
                    if m > 2 {
                        *y.get_unchecked_mut(2 * r + row) = v[2];
                    }
                    if m > 3 {
                        *y.get_unchecked_mut(3 * r + row) = v[3];
                    }
                    if m > 4 {
                        *y.get_unchecked_mut(4 * r + row) = v[4];
                    }
                    if m > 5 {
                        *y.get_unchecked_mut(5 * r + row) = v[5];
                    }
                    if m > 6 {
                        *y.get_unchecked_mut(6 * r + row) = v[6];
                    }
                    if m > 7 {
                        *y.get_unchecked_mut(7 * r + row) = v[7];
                    }
                }
            }
        }

        /// The row probe: one warp walks Q5_K row `row_abs` against column
        /// `col` as the `_sel` entries do and writes, as bits, its 32 lane
        /// partials at `out[0..32]`, their warp sum at `out[32]`, and per
        /// sub-block `(sb, s)` at `out[33 + 11·(8·sb + s)..]` the decoder's
        /// eight code words, `cda`, `cdb` and the walk's term for it.
        #[allow(
            clippy::too_many_arguments,
            reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
        )]
        #[kernel]
        #[launch_bounds(32)]
        #[launch_contract(
            domain = 1,
            block = (32, 1, 1),
            requires = (
                w.len() >= (row_abs + 1) * 44 * n_sb,
                q.len() >= (col + 1) * 256 * iters,
                s8.len() >= (col + 1) * 8 * n_sb,
                d8.len() >= (col + 1) * 2 * n_sb,
                out.len() >= 33 + 88 * n_sb
            )
        )]
        pub fn kq_probe_q5k(
            w: &[u32],
            q: &[u32],
            s8: &[i32],
            d8: &[f32],
            row_abs: u32,
            col: u32,
            n_sb: u32,
            iters: u32,
            mut out: DisjointSlice<u32>,
        ) {
            if thread::index_1d().get() >= 32 {
                return;
            }
            let lane = warp::lane_id() as usize;
            let (n_sb, row, col) = (n_sb as usize, row_abs as usize, col as usize);
            // SAFETY: the launch contract bounds row `row_abs` of `w` and column
            // `col` of the planes; iters = ceil(n_sb/4) from the host; the one
            // block is the one warp.
            let f = unsafe { row_dot_1col::<Q5k>(w, q, s8, d8, n_sb, iters, row, col, lane) };
            let sum = warp::reduce_sum_f32(f);
            // SAFETY: lane < 32 < out.len(); each lane writes its own word, lane
            // 0 also word 32.
            unsafe {
                *out.get_unchecked_mut(lane) = f.to_bits();
                if lane == 0 {
                    *out.get_unchecked_mut(32) = sum.to_bits();
                }
            }
            let (s, grp) = (lane & 7, lane >> 3);
            let mut it = 0u32;
            while it < iters {
                let sbp = 4 * it as usize + grp;
                if sbp < n_sb {
                    // SAFETY: sbp < n_sb keeps the super-block inside row
                    // `row_abs` and the term's plane reads inside column `col`.
                    let ((vi, cda, cdb), term) = unsafe {
                        (
                            Q5k::decode(w, row * 44 * n_sb + 44 * sbp, s),
                            iter_term::<Q5k>(
                                w,
                                q,
                                s8,
                                d8,
                                n_sb,
                                it,
                                row,
                                col * 256 * iters as usize,
                                col * 8 * n_sb,
                                col * 2 * n_sb,
                                s,
                                grp,
                                lane,
                            ),
                        )
                    };
                    let b = 33 + 11 * (8 * sbp + s);
                    // SAFETY: b + 10 < 33 + 88·n_sb <= out.len() because sbp <
                    // n_sb and s < 8; (sbp, s) is this lane's alone.
                    unsafe {
                        *out.get_unchecked_mut(b) = vi[0];
                        *out.get_unchecked_mut(b + 1) = vi[1];
                        *out.get_unchecked_mut(b + 2) = vi[2];
                        *out.get_unchecked_mut(b + 3) = vi[3];
                        *out.get_unchecked_mut(b + 4) = vi[4];
                        *out.get_unchecked_mut(b + 5) = vi[5];
                        *out.get_unchecked_mut(b + 6) = vi[6];
                        *out.get_unchecked_mut(b + 7) = vi[7];
                        *out.get_unchecked_mut(b + 8) = cda.to_bits();
                        *out.get_unchecked_mut(b + 9) = cdb.to_bits();
                        *out.get_unchecked_mut(b + 10) = term.to_bits();
                    }
                }
                it += 1;
            }
        }
    }

    // The gate module spells the two formats' super-block words.
    const _: () = assert!(Q4k::WORDS == 36 && Q5k::WORDS == 44);

    /// Experts of every stack here.
    const E: usize = 4;
    /// What an output buffer holds before a launch: a slot left alone reads
    /// back as these bits.
    const SENT: f32 = 1.0e30;

    /// Everything the clauses share.
    struct Ctx {
        gpu: Gpu,
        kq: KquantKernels,
        gm: gate_kernels::LoadedModule,
    }

    /// One synthetic stack of `E` experts of `rpe` rows of `k` values.
    struct Stack {
        tag: &'static str,
        ty: GgmlType,
        k: usize,
        rpe: usize,
        words: Vec<u32>,
        w: DeviceTensor<u32>,
    }

    impl Stack {
        fn new(
            c: &Ctx,
            tag: &'static str,
            ty: GgmlType,
            rpe: usize,
            k: usize,
            seed: u64,
        ) -> Result<Stack, GateError> {
            let n_sb = k / 256;
            let words = synthetic(ty, E * rpe, n_sb, seed)?;
            let wpr = words.len() / (E * rpe);
            let w = DeviceTensor::upload(c.gpu.stream(), &words, E * rpe, wpr)?;
            Ok(Stack {
                tag,
                ty,
                k,
                rpe,
                words,
                w,
            })
        }

        fn n_sb(&self) -> usize {
            self.k / 256
        }

        /// Expert `e`'s rows as bytes.
        fn expert_bytes(&self, e: usize) -> Vec<u8> {
            let wpr = self.w.cols();
            words_bytes(&self.words[e * self.rpe * wpr..(e + 1) * self.rpe * wpr])
        }
    }

    /// `rows` synthetic super-block rows of `ty` (Q4_K or Q5_K), `n_sb`
    /// super-blocks each, as words from a fixed-seed xorshift64: every word
    /// random except the first of each super-block, whose halves are `d` and
    /// `dmin` — positive normal f16 (exponent field 1..=9), so no NaN or
    /// infinity enters. Every other word pattern is a valid super-block.
    fn synthetic(ty: GgmlType, rows: usize, n_sb: usize, seed: u64) -> Result<Vec<u32>, GateError> {
        let words = match ty {
            GgmlType::Q4_K => Q4k::WORDS,
            GgmlType::Q5_K => Q5k::WORDS,
            other => return Err(format!("gate_kquant: no synthetic {other:?}").into()),
        };
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let half = |r: u64| -> u32 { ((1 + (r % 9) as u32) << 10) | ((r >> 32) as u32 & 0x3ff) };
        let mut out = Vec::with_capacity(rows * n_sb * words);
        for _ in 0..rows * n_sb {
            let (d, dmin) = (half(next()), half(next()));
            out.push(d | (dmin << 16));
            for _ in 1..words {
                out.push((next() >> 32) as u32);
            }
        }
        Ok(out)
    }

    fn words_bytes(w: &[u32]) -> Vec<u8> {
        w.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// `cols` columns of `k` values quantized to q8_1 by the engine.
    fn quantize(c: &Ctx, x: &[f32], cols: usize, k: usize) -> Result<Q8Act, GateError> {
        let stream = c.gpu.stream();
        let xd = DeviceBuffer::from_host(stream, &x[..cols * k])?;
        let mut act = Q8Act::with_slots(stream, cols, k)?;
        c.gpu.enqueue_quantize_q8_1(&xd, &mut act)?;
        stream.synchronize()?;
        Ok(act)
    }

    /// The down `_sel` of stack `st` for `sel` (one id a column of `act`) into a
    /// `SENT`-filled output, synchronized; the output and the fault word.
    fn down(
        c: &Ctx,
        st: &Stack,
        act: &Q8Act,
        sel: &[u32],
    ) -> Result<(Vec<f32>, Option<Fault>), GateError> {
        let stream = c.gpu.stream();
        let sel_dev = DeviceBuffer::from_host(stream, sel)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; sel.len() * st.rpe])?;
        let a = SelDown {
            w: &st.w,
            act,
            sel: &sel_dev,
            n_slots: sel.len(),
            rows_per_expert: st.rpe,
        };
        c.kq.enqueue_gemv_q5k_sel(stream, &a, c.gpu.unlabelled_sink(), &mut y)?;
        stream.synchronize()?;
        Ok((y.to_host_vec(stream)?, c.gpu.take_fault()?))
    }

    /// The gate·up of stacks `g`/`u` for `sel` at `spc` slots a column of
    /// `act` under `rule`, into a `SENT`-filled output; the output and the
    /// fault word.
    fn gate_up(
        c: &Ctx,
        g: &Stack,
        u: &Stack,
        act: &Q8Act,
        sel: &[u32],
        spc: usize,
        rule: Act,
    ) -> Result<(Vec<f32>, Option<Fault>), GateError> {
        let stream = c.gpu.stream();
        let sel_dev = DeviceBuffer::from_host(stream, sel)?;
        let mut h = DeviceBuffer::from_host(stream, &vec![SENT; sel.len() * g.rpe])?;
        let a = GateUpAct {
            wg: &g.w,
            wu: &u.w,
            act,
            sel: &sel_dev,
            n_slots: sel.len(),
            rows_per_expert: g.rpe,
            slots_per_col: spc,
            rule,
        };
        c.kq.enqueue_gate_up_q5k(stream, &a, c.gpu.unlabelled_sink(), &mut h)?;
        stream.synchronize()?;
        Ok((h.to_host_vec(stream)?, c.gpu.take_fault()?))
    }

    /// The probe of Q5_K row `row_abs` of `st` against column `col` of `act`.
    fn probe(
        c: &Ctx,
        st: &Stack,
        act: &Q8Act,
        row_abs: usize,
        col: usize,
    ) -> Result<Vec<u32>, GateError> {
        let stream = c.gpu.stream();
        let n_sb = st.n_sb();
        let mut out = DeviceBuffer::<u32>::zeroed(stream, PROBE_HEAD + 8 * PROBE_SUB * n_sb)?;
        let (q, s8, d8) = walk_a_planes(act);
        let prep = c.gm.prepare_kq_probe_q5k(LaunchConfig1D::new(1, 32, 0))?;
        c.gm.kq_probe_q5k(
            stream,
            &prep,
            st.w.buf(),
            q,
            s8,
            d8,
            u32::try_from(row_abs)?,
            u32::try_from(col)?,
            u32::try_from(n_sb)?,
            u32::try_from(n_sb.div_ceil(4))?,
            &mut out,
        )?;
        stream.synchronize()?;
        Ok(out.to_host_vec(stream)?)
    }

    /// ggml's `get_scale_min_k4`: sub-block `j`'s 6-bit scale and minimum.
    fn scale_min(j: usize, sc: &[u8]) -> (u8, u8) {
        if j < 4 {
            (sc[j] & 63, sc[j + 4] & 63)
        } else {
            (
                (sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4),
                (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
            )
        }
    }

    /// The host's reading of sub-block `s` of a Q5_K super-block: its eight
    /// code words (bytes `nib | 16·h`, values `32·s + 4·i ..` in word `i`),
    /// `d·sc` and `−dmin·m`.
    fn host_decode(sb: &[u8], s: usize) -> ([u32; 8], f32, f32) {
        let d = half_to_f32(u16::from_le_bytes([sb[0], sb[1]]));
        let dmin = half_to_f32(u16::from_le_bytes([sb[2], sb[3]]));
        let (sc, m) = scale_min(s, &sb[4..16]);
        let (qh, qs) = (&sb[16..48], &sb[48..176]);
        let mut words = [0u32; 8];
        for (i, w) in words.iter_mut().enumerate() {
            for b in 0..4 {
                let l = 4 * i + b;
                let nib = (qs[32 * (s >> 1) + l] >> (4 * (s & 1))) & 0x0f;
                let h = (qh[l] >> s) & 1;
                *w |= u32::from(nib | (h << 4)) << (8 * b);
            }
        }
        (words, d * f32::from(sc), -(dmin * f32::from(m)))
    }

    /// The probe's decode of row `row_abs` of `st` against the host's, word for
    /// word; the first sub-block that differs, or `None`.
    fn probe_decode_diff(st: &Stack, row_abs: usize, out: &[u32]) -> Option<String> {
        let n_sb = st.n_sb();
        let row = &words_bytes(&st.words[row_abs * 44 * n_sb..(row_abs + 1) * 44 * n_sb]);
        for sb in 0..n_sb {
            for s in 0..8 {
                let (codes, cda, cdb) = host_decode(&row[176 * sb..176 * (sb + 1)], s);
                let b = PROBE_HEAD + PROBE_SUB * (8 * sb + s);
                let dev = &out[b..b + 10];
                let host: Vec<u32> = codes
                    .iter()
                    .copied()
                    .chain([cda.to_bits(), cdb.to_bits()])
                    .collect();
                if dev != host.as_slice() {
                    return Some(format!(
                        "sb {sb} sub {s} device {dev:08x?} host {host:08x?}"
                    ));
                }
            }
        }
        None
    }

    /// ` first_out_of_band=…`: the first sub-block of row `row_abs` whose probe
    /// term is further than `tol` from the host's f64 value of that sub-block
    /// (column `xq`, the q8_1-dequantized values), both printed.
    fn first_sub_out(st: &Stack, row_abs: usize, xq: &[f32], out: &[u32], tol: f64) -> String {
        let k = st.k;
        let wpr = st.w.cols();
        let bytes = words_bytes(&st.words[row_abs * wpr..(row_abs + 1) * wpr]);
        let mut vals = vec![0.0f32; k];
        if dequant_row(st.ty, &bytes, &mut vals).is_err() {
            return " first_out_of_band=dequant_row refused the row".into();
        }
        for sb in 0..st.n_sb() {
            for s in 0..8 {
                let lo = 256 * sb + 32 * s;
                let host: f64 = (lo..lo + 32)
                    .map(|i| f64::from(vals[i]) * f64::from(xq[i]))
                    .sum();
                let dev = f32::from_bits(out[PROBE_HEAD + PROBE_SUB * (8 * sb + s) + 10]);
                if (f64::from(dev) - host).abs() > tol {
                    return format!(
                        " first_out_of_band=row {row_abs} sb {sb} sub {s} device {dev:e} host {host:e}"
                    );
                }
            }
        }
        " first_out_of_band=none at sub-block grain".into()
    }

    /// ` <label>=slot S row R got 0x… want 0x…` at the first index where the
    /// bits differ, `rpe` rows a slot; empty when they agree.
    fn mismatch(label: &str, got: &[f32], want: &[f32], rpe: usize) -> String {
        if got.len() != want.len() {
            return format!(" {label}=length got {} want {}", got.len(), want.len());
        }
        match got
            .iter()
            .zip(want)
            .position(|(g, w)| g.to_bits() != w.to_bits())
        {
            Some(i) => format!(
                " {label}=slot {} row {} got {:#010x} want {:#010x}",
                i / rpe,
                i % rpe,
                got[i].to_bits(),
                want[i].to_bits()
            ),
            None => String::new(),
        }
    }

    fn shown(f: Option<Fault>) -> String {
        f.map_or_else(|| "none".to_owned(), |f| f.to_string())
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let kq = KquantKernels::load(gpu.context(), gpu.fault_word())?;
        // SAFETY: this binary owns the embedded bundle the module above produced;
        // every launch here passes the launch contract's check.
        let gm = unsafe { gate_kernels::load(gpu.context())? };
        let c = Ctx { gpu, kq, gm };
        if let Some(f) = c.gpu.take_fault()? {
            return Err(format!("gate_kquant: the fault word held {f} before any launch").into());
        }
        let mut ok = true;
        ok &= check_band(&c)?;
        ok &= check_sel(&c)?;
        ok &= check_mcol(&c)?;
        ok &= check_fault(&c)?;
        ok &= check_act(&c)?;
        ok &= check_q4k_sibling(&c)?;
        if !ok {
            return Err(bloomery_gpu_gates::checks_failed());
        }
        println!(
            "PASSED: gate_kquant q5k_gemv_sel within {KERNEL_BAND:e} of the f64 dequant_row \
             reference at 8 shapes, its probe decoding as the host; slots, duplicate ids, rerun, \
             graph replay and two-token gate·up bit for bit; the m-column walk is the one-column \
             walk at m = 1..8 on 8 K; ids past the stack raise expert_id, HOST raises nothing, a \
             NaN column raises quant_column and NaNs its rows; the gate·up is silu_mul and \
             swiglu_clamp on the down's sums bit for bit; Walk A over Q4_K is q4k_gemv_sel bit \
             for bit on 6 K"
        );
        Ok(())
    }

    /// Clause 1 (module doc).
    fn check_band(c: &Ctx) -> Result<bool, GateError> {
        // (tag, rows an expert, K)
        const SHAPES: [(&str, usize, usize); 8] = [
            ("down_2048x512", 2048, 512),
            ("gate_up_512x2048", 512, 2048),
            ("down_4096x2048", 4096, 2048),
            ("gate_up_2048x4096", 2048, 4096),
            ("k256", 512, 256),
            ("k768", 512, 768),
            ("k1280", 512, 1280),
            ("k2304", 512, 2304),
        ];
        const SEL: [u32; 4] = [3, 0, 2, 1];
        let mut ok = true;
        for (i, &(tag, rpe, k)) in SHAPES.iter().enumerate() {
            let st = Stack::new(c, tag, GgmlType::Q5_K, rpe, k, 0x51a7_0001 + i as u64)?;
            let x = activations(k, SEL.len(), 7001 + i as u32);
            let act = quantize(c, &x, SEL.len(), k)?;
            let (y, fault) = down(c, &st, &act, &SEL)?;
            let xq = q8_1_dequant(&x, k, SEL.len());
            let mut want = Vec::with_capacity(y.len());
            for (s, &e) in SEL.iter().enumerate() {
                want.extend(ref_gemv(
                    GgmlType::Q5_K,
                    &st.expert_bytes(e as usize),
                    k,
                    rpe,
                    &xq[s * k..(s + 1) * k],
                    1,
                )?);
            }
            let rel = max_rel_err(&y, &want)?;
            let in_band = rel <= KERNEL_BAND;
            // The probe on a row off the block and warp boundaries, in slot 1.
            let (slot, r) = (1usize, rpe / 2 + 3);
            let row_abs = SEL[slot] as usize * rpe + r;
            let out = probe(c, &st, &act, row_abs, slot)?;
            let decode_diff = probe_decode_diff(&st, row_abs, &out);
            let sum_same = out[32] == y[slot * rpe + r].to_bits();
            let pass = fault.is_none() && in_band && decode_diff.is_none() && sum_same;
            // Out of band: the worst row's first sub-block out of band.
            let detail = if in_band {
                String::new()
            } else {
                let worst = (0..y.len())
                    .max_by(|&a, &b| (y[a] - want[a]).abs().total_cmp(&(y[b] - want[b]).abs()))
                    .unwrap_or(0);
                let (ws, wr) = (worst / rpe, worst % rpe);
                let w_abs = SEL[ws] as usize * rpe + wr;
                let wout = probe(c, &st, &act, w_abs, ws)?;
                let denom = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                format!(
                    " worst=slot {ws} row {wr} got {:e} want {:e}{}",
                    y[worst],
                    want[worst],
                    first_sub_out(
                        &st,
                        w_abs,
                        &xq[ws * k..(ws + 1) * k],
                        &wout,
                        f64::from(KERNEL_BAND * denom)
                    )
                )
            };
            ok &= pass;
            println!(
                "band[{}] rows={rpe} K={k} n_sb={} max_rel={rel:.3e} band={KERNEL_BAND:e} \
                 fault=\"{}\" probe_decode_as_host={} probe_sum_is_launch={sum_same} {}{}{}",
                st.tag,
                st.n_sb(),
                shown(fault),
                decode_diff.is_none(),
                verdict(pass),
                decode_diff.map_or_else(String::new, |d| format!(" first_decode_diff={d}")),
                detail,
            );
        }
        Ok(ok)
    }

    /// Clause 2 (module doc).
    fn check_sel(c: &Ctx) -> Result<bool, GateError> {
        const SEL_A: [u32; 6] = [2, 0, 3, 3, 1, 2];
        const SEL_B: [u32; 6] = [1, 3, 3, 0, 2, 0];
        let stream = c.gpu.stream();
        let mut ok = true;
        let mut graph_stack = None;
        for (i, &(tag, rpe, k)) in [("k2304", 256usize, 2304usize), ("k512", 512, 512)]
            .iter()
            .enumerate()
        {
            let st = Stack::new(c, tag, GgmlType::Q5_K, rpe, k, 0x5e1_0001 + i as u64)?;
            let x = activations(k, SEL_A.len(), 7101 + i as u32);
            let act = quantize(c, &x, SEL_A.len(), k)?;
            for (name, sel) in [("a", SEL_A), ("b", SEL_B)] {
                let (y1, f1) = down(c, &st, &act, &sel)?;
                let (y2, f2) = down(c, &st, &act, &sel)?;
                // Slot s alone: expert sel[s]'s rows uploaded alone, column s alone.
                let mut want = Vec::with_capacity(y1.len());
                for (s, &e) in sel.iter().enumerate() {
                    let wpr = st.w.cols();
                    let lo = e as usize * rpe * wpr;
                    let one = Stack {
                        tag: st.tag,
                        ty: st.ty,
                        k,
                        rpe,
                        words: st.words[lo..lo + rpe * wpr].to_vec(),
                        w: DeviceTensor::upload(stream, &st.words[lo..lo + rpe * wpr], rpe, wpr)?,
                    };
                    let a1 = quantize(c, &x[s * k..(s + 1) * k], 1, k)?;
                    want.extend(down(c, &one, &a1, &[0])?.0);
                }
                let (da, db) = (2, 3);
                let dup_differ =
                    !bits_equal(&y1[da * rpe..(da + 1) * rpe], &y1[db * rpe..(db + 1) * rpe]);
                let slot_same = bits_equal(&y1, &want);
                let rerun = bits_equal(&y1, &y2);
                let pass = slot_same && rerun && dup_differ && f1.is_none() && f2.is_none();
                ok &= pass;
                println!(
                    "sel[{tag}:{name}] sel={sel:?} slot_is_expert_alone={slot_same} \
                     rerun_bit_identical={rerun} dup_slots_differ={dup_differ} faults=\"{}\",\"{}\" {}{}",
                    shown(f1),
                    shown(f2),
                    verdict(pass),
                    mismatch("first_mismatch", &y1, &want, rpe),
                );
            }
            if i == 0 {
                graph_stack = Some((st, act));
            }
        }
        // The graph: one launch captured with SEL_A, replayed, the ids
        // overwritten with SEL_B outside it, replayed.
        let (st, act) = graph_stack.ok_or("gate_kquant: no stack for the graph case")?;
        let eager_a = down(c, &st, &act, &SEL_A)?.0;
        let eager_b = down(c, &st, &act, &SEL_B)?.0;
        let mut sel_dev = DeviceBuffer::from_host(stream, &SEL_A)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; SEL_A.len() * st.rpe])?;
        let graph = c.gpu.capture(|s| {
            let a = SelDown {
                w: &st.w,
                act: &act,
                sel: &sel_dev,
                n_slots: SEL_A.len(),
                rows_per_expert: st.rpe,
            };
            c.kq.enqueue_gemv_q5k_sel(s, &a, c.gpu.unlabelled_sink(), &mut y)
        })?;
        let nodes = graph.node_count();
        graph.launch(stream)?;
        stream.synchronize()?;
        let ya = y.to_host_vec(stream)?;
        sel_dev.copy_from_host(stream, &SEL_B)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let yb = y.to_host_vec(stream)?;
        drop(graph);
        let (a_same, b_same) = (bits_equal(&ya, &eager_a), bits_equal(&yb, &eager_b));
        let pass = a_same && b_same && nodes == 1 && c.gpu.take_fault()?.is_none();
        ok &= pass;
        println!(
            "sel_graph[{}] replay_a_bit_identical={a_same} replay_b_bit_identical={b_same} \
             graph_nodes={nodes} {}{}{}",
            st.tag,
            verdict(pass),
            mismatch("first_mismatch_a", &ya, &eager_a, st.rpe),
            mismatch("first_mismatch_b", &yb, &eager_b, st.rpe),
        );
        // The gate·up over two tokens of four slots against each token alone.
        let (rpe, k) = (512, 2048);
        let g = Stack::new(c, "gate", GgmlType::Q5_K, rpe, k, 0x9a7e_0001)?;
        let u = Stack::new(c, "up", GgmlType::Q5_K, rpe, k, 0x9a7e_0002)?;
        let sel2: [u32; 8] = [0, 3, 1, 2, 2, 1, 3, 3];
        let x = activations(k, 2, 7201);
        let act2 = quantize(c, &x, 2, k)?;
        let (h2, f2) = gate_up(c, &g, &u, &act2, &sel2, 4, Act::SiluMul)?;
        let mut want = Vec::with_capacity(h2.len());
        for t in 0..2 {
            let a1 = quantize(c, &x[t * k..(t + 1) * k], 1, k)?;
            want.extend(gate_up(c, &g, &u, &a1, &sel2[4 * t..4 * t + 4], 4, Act::SiluMul)?.0);
        }
        let tokens_same = bits_equal(&h2, &want);
        let pass = tokens_same && f2.is_none();
        ok &= pass;
        println!(
            "sel_gate_up_tokens[{}x{}] tokens=2 slots_per_col=4 sel={sel2:?} \
             each_token_alone_bit_identical={tokens_same} fault=\"{}\" {}{}",
            rpe,
            k,
            shown(f2),
            verdict(pass),
            mismatch("first_mismatch", &h2, &want, rpe),
        );
        ok &= check_host_contract(c, &g, &act2)?;
        Ok(ok)
    }

    /// Clause 2's refusals: each a `Shape` error of its launcher.
    fn check_host_contract(c: &Ctx, g: &Stack, act2: &Q8Act) -> Result<bool, GateError> {
        let stream = c.gpu.stream();
        let sel = DeviceBuffer::from_host(stream, &[0u32; 8])?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; 8 * g.rpe])?;
        let sink = c.gpu.unlabelled_sink();
        let small =
            DeviceTensor::upload(stream, &g.words[..E * 64 * g.w.cols()], E * 64, g.w.cols())?;
        let cases: [(&str, &str, Result<(), GpuError>); 3] = [
            (
                "enqueue_gemv_q5k_sel",
                "act_2_columns_for_8_slots",
                c.kq.enqueue_gemv_q5k_sel(
                    stream,
                    &SelDown {
                        w: &g.w,
                        act: act2,
                        sel: &sel,
                        n_slots: 8,
                        rows_per_expert: g.rpe,
                    },
                    sink,
                    &mut y,
                ),
            ),
            (
                "enqueue_gemv_q5k_sel",
                "rows_per_expert_300_of_2048",
                c.kq.enqueue_gemv_q5k_sel(
                    stream,
                    &SelDown {
                        w: &g.w,
                        act: act2,
                        sel: &sel,
                        n_slots: 2,
                        rows_per_expert: 300,
                    },
                    sink,
                    &mut y,
                ),
            ),
            (
                "enqueue_gate_up_q5k",
                "up_stack_of_other_rows",
                c.kq.enqueue_gate_up_q5k(
                    stream,
                    &GateUpAct {
                        wg: &g.w,
                        wu: &small,
                        act: act2,
                        sel: &sel,
                        n_slots: 8,
                        rows_per_expert: g.rpe,
                        slots_per_col: 4,
                        rule: Act::SiluMul,
                    },
                    sink,
                    &mut y,
                ),
            ),
        ];
        let mut ok = true;
        for (what, case, r) in cases {
            let pass = matches!(&r, Err(GpuError::Shape { what: w, .. }) if *w == what);
            let seen = match &r {
                Ok(()) => "Ok (accepted)".to_string(),
                Err(e) => format!("Err: {e}"),
            };
            println!(
                "sel_host[{case}] want=Err(Shape {what}) got={seen} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        stream.synchronize()?;
        Ok(ok && c.gpu.take_fault()?.is_none())
    }

    /// Clause 3 (module doc).
    fn check_mcol(c: &Ctx) -> Result<bool, GateError> {
        const ROWS: usize = 64;
        let stream = c.gpu.stream();
        let mut ok = true;
        for (i, k) in [256usize, 512, 768, 1280, 2048, 2304, 4096, 8192]
            .into_iter()
            .enumerate()
        {
            let n_sb = k / 256;
            let words = synthetic(GgmlType::Q5_K, ROWS, n_sb, 0x3c01_0001 + i as u64)?;
            let w = DeviceTensor::upload(stream, &words, ROWS, 44 * n_sb)?;
            let x = activations(k, 8, 7301 + i as u32);
            let mut same = [false; 8];
            let mut first = String::new();
            for m in 1..=8 {
                let act = quantize(c, &x, m, k)?;
                let (q, s8, d8) = walk_a_planes(&act);
                let mut y = DeviceBuffer::from_host(stream, &vec![SENT; m * ROWS])?;
                let prep = c.gm.prepare_kq_gemv_mcol_q5k(LaunchConfig1D::new(
                    u32::try_from(ROWS.div_ceil(ROWS_PER_BLOCK))?,
                    THREADS,
                    0,
                ))?;
                c.gm.kq_gemv_mcol_q5k(
                    stream,
                    &prep,
                    w.buf(),
                    q,
                    s8,
                    d8,
                    ROWS as u32,
                    m as u32,
                    n_sb as u32,
                    n_sb.div_ceil(4) as u32,
                    &mut y,
                )?;
                stream.synchronize()?;
                let got = y.to_host_vec(stream)?;
                let sel_dev = DeviceBuffer::from_host(stream, &vec![0u32; m])?;
                let mut yr = DeviceBuffer::from_host(stream, &vec![SENT; m * ROWS])?;
                let a = SelDown {
                    w: &w,
                    act: &act,
                    sel: &sel_dev,
                    n_slots: m,
                    rows_per_expert: ROWS,
                };
                c.kq.enqueue_gemv_q5k_sel(stream, &a, c.gpu.unlabelled_sink(), &mut yr)?;
                stream.synchronize()?;
                let want = yr.to_host_vec(stream)?;
                same[m - 1] = bits_equal(&got, &want);
                if !same[m - 1] && first.is_empty() {
                    first = format!(" m={m}{}", mismatch("first_mismatch", &got, &want, ROWS));
                }
            }
            let pass = same.iter().all(|&s| s) && c.gpu.take_fault()?.is_none();
            ok &= pass;
            println!(
                "mcol[K={k}] n_sb={n_sb} rows={ROWS} m=1..8 column_is_one_column_walk={same:?} {}{first}",
                verdict(pass)
            );
        }
        Ok(ok)
    }

    /// Clause 4 (module doc).
    fn check_fault(c: &Ctx) -> Result<bool, GateError> {
        const CLEAN: [u32; 4] = [1, 3, 0, 2];
        let past = E as u32 + 3;
        let (rpe, k) = (256, 2304);
        let st = Stack::new(c, "down", GgmlType::Q5_K, rpe, k, 0xfa17_0001)?;
        let g = Stack::new(c, "gate", GgmlType::Q5_K, rpe, k, 0xfa17_0002)?;
        let u = Stack::new(c, "up", GgmlType::Q5_K, rpe, k, 0xfa17_0003)?;
        let x = activations(k, 4, 7401);
        let act = quantize(c, &x, 4, k)?;
        let act1 = quantize(c, &x, 1, k)?;
        let expert_id = Some(Fault::at(LAYER_NONE, FaultSite::ExpertId));
        let (clean_y, f0) = down(c, &st, &act, &CLEAN)?;
        let (clean_h, f1) = gate_up(c, &g, &u, &act1, &CLEAN, 4, Act::SiluMul)?;
        let mut ok = f0.is_none() && f1.is_none();
        // (entry, case, the id in slot 1, the fault, slot 1's rows: SENT or NaN)
        let cases = [
            ("down", "past_stack", past, expert_id, SENT),
            ("down", "host", HOST, None, SENT),
            ("gate_up", "past_stack", past, expert_id, f32::NAN),
            ("gate_up", "host", HOST, None, SENT),
        ];
        for (entry, case, id, want_fault, fill) in cases {
            let mut sel = CLEAN;
            sel[1] = id;
            let (got, fault, clean) = if entry == "down" {
                let (y, f) = down(c, &st, &act, &sel)?;
                (y, f, &clean_y)
            } else {
                let (h, f) = gate_up(c, &g, &u, &act1, &sel, 4, Act::SiluMul)?;
                (h, f, &clean_h)
            };
            let mut want = clean.clone();
            want[rpe..2 * rpe].fill(fill);
            let values_ok = bits_equal(&got, &want);
            let pass = values_ok && fault == want_fault;
            ok &= pass;
            println!(
                "fault[{entry}:{case}] sel={sel:?} fault=\"{}\" want=\"{}\" slot1_{}_others_clean={values_ok} {}{}",
                shown(fault),
                shown(want_fault),
                if fill.is_nan() { "nan" } else { "untouched" },
                verdict(pass),
                mismatch("first_mismatch", &got, &want, rpe),
            );
        }
        // A NaN in column 2 (the down) and in the gate·up's one column.
        let quant_column = Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
        let mut xn = x.clone();
        xn[2 * k + 77] = f32::NAN;
        let actn = quantize(c, &xn, 4, k)?;
        let qf = c.gpu.take_fault()?;
        let (y, fd) = down(c, &st, &actn, &CLEAN)?;
        let slot2_nan = y[2 * rpe..3 * rpe].iter().all(|v| v.is_nan());
        let others = bits_equal(&y[..2 * rpe], &clean_y[..2 * rpe])
            && bits_equal(&y[3 * rpe..], &clean_y[3 * rpe..]);
        let pass = qf == quant_column && fd.is_none() && slot2_nan && others;
        ok &= pass;
        println!(
            "fault[down:nan_column] quantizer_fault=\"{}\" want=\"{}\" down_fault=\"{}\" \
             column2_slot_all_nan={slot2_nan} other_slots_clean={others} {}",
            shown(qf),
            shown(quant_column),
            shown(fd),
            verdict(pass)
        );
        let mut x1 = x[..k].to_vec();
        x1[1000] = f32::NAN;
        let act1n = quantize(c, &x1, 1, k)?;
        let qf = c.gpu.take_fault()?;
        let (h, fg) = gate_up(c, &g, &u, &act1n, &CLEAN, 4, Act::SiluMul)?;
        let (hc, fc) = gate_up(
            c,
            &g,
            &u,
            &act1n,
            &CLEAN,
            4,
            Act::SwigluClamp { limit: 10.0 },
        )?;
        let all_nan = h.iter().chain(&hc).all(|v| v.is_nan());
        let pass = qf == quant_column && fg.is_none() && fc.is_none() && all_nan;
        ok &= pass;
        println!(
            "fault[gate_up:nan_column] quantizer_fault=\"{}\" want=\"{}\" gate_up_faults=\"{}\",\"{}\" \
             every_row_nan_both_rules={all_nan} {}",
            shown(qf),
            shown(quant_column),
            shown(fg),
            shown(fc),
            verdict(pass)
        );
        Ok(ok)
    }

    /// Clause 5 (module doc).
    fn check_act(c: &Ctx) -> Result<bool, GateError> {
        const SEL: [u32; 4] = [3, 0, 2, 1];
        let stream = c.gpu.stream();
        let (rpe, k) = (512, 2048);
        let g = Stack::new(c, "gate", GgmlType::Q5_K, rpe, k, 0xac70_0001)?;
        let u = Stack::new(c, "up", GgmlType::Q5_K, rpe, k, 0xac70_0002)?;
        let x = activations(k, 1, 7501);
        let act1 = quantize(c, &x, 1, k)?;
        // The down `_sel` of the same rows: every slot on the token's column.
        let x4: Vec<f32> = (0..SEL.len()).flat_map(|_| x.iter().copied()).collect();
        let act4 = quantize(c, &x4, SEL.len(), k)?;
        let (gs, fg) = down(c, &g, &act4, &SEL)?;
        let (us, fu) = down(c, &u, &act4, &SEL)?;
        let n = gs.len();
        let mut ok = fg.is_none() && fu.is_none();
        // silu_mul: the engine's elementwise swiglu on the same sums.
        let (gd, ud) = (
            DeviceBuffer::from_host(stream, &gs)?,
            DeviceBuffer::from_host(stream, &us)?,
        );
        let mut yd = DeviceBuffer::<f32>::zeroed(stream, n)?;
        c.gpu.elem().enqueue_swiglu(stream, &gd, &ud, n, &mut yd)?;
        stream.synchronize()?;
        let want_silu = yd.to_host_vec(stream)?;
        let rules = [
            ("silu_mul", Act::SiluMul),
            ("swiglu_clamp_10", Act::SwigluClamp { limit: 10.0 }),
            ("swiglu_clamp_0.5", Act::SwigluClamp { limit: 0.5 }),
        ];
        for (name, rule) in rules {
            let (h, f) = gate_up(c, &g, &u, &act1, &SEL, 4, rule)?;
            let (want, clamped) = match rule {
                Act::SiluMul => (want_silu.clone(), 0),
                Act::SwigluClamp { limit } => {
                    let want: Vec<f32> = gs
                        .iter()
                        .zip(&us)
                        .map(|(&g, &u)| act::swiglu_clamp(g, u, limit))
                        .collect();
                    let clamped = gs
                        .iter()
                        .zip(&us)
                        .filter(|&(&g, &u)| act::silu_ik(g) > limit || u.abs() > limit)
                        .count();
                    (want, clamped)
                }
            };
            let same = bits_equal(&h, &want);
            // A clamp rule must bite on some rows and not on others here.
            let spread = matches!(rule, Act::SiluMul) || (clamped > 0 && clamped < n);
            let pass = same && spread && f.is_none();
            ok &= pass;
            println!(
                "act[{name}] rows={n} rule_on_down_sums_bit_identical={same} clamped_rows={clamped} \
                 fault=\"{}\" {}{}",
                shown(f),
                verdict(pass),
                mismatch("first_mismatch", &h, &want, rpe),
            );
        }
        Ok(ok)
    }

    /// Clause 6 (module doc).
    fn check_q4k_sibling(c: &Ctx) -> Result<bool, GateError> {
        const SEL: [u32; 6] = [2, 0, 3, 3, HOST, 1];
        const RPE: usize = 128;
        let stream = c.gpu.stream();
        let mut ok = true;
        for (i, k) in [256usize, 768, 1280, 2048, 2304, 4096]
            .into_iter()
            .enumerate()
        {
            let st = Stack::new(c, "q4k", GgmlType::Q4_K, RPE, k, 0x4a4b_0001 + i as u64)?;
            let n_sb = st.n_sb();
            let x = activations(k, SEL.len(), 7601 + i as u32);
            let act = quantize(c, &x, SEL.len(), k)?;
            let sel_dev = DeviceBuffer::from_host(stream, &SEL)?;
            let mut yw = DeviceBuffer::from_host(stream, &vec![SENT; SEL.len() * RPE])?;
            c.gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                &st.w,
                &act,
                &sel_dev,
                SEL.len(),
                RPE,
                &mut yw,
            )?;
            let mut yg = DeviceBuffer::from_host(stream, &vec![SENT; SEL.len() * RPE])?;
            let (q, s8, d8) = walk_a_planes(&act);
            let prep = c.gm.prepare_kq_gemv_sel_q4k(LaunchConfig1D::new(
                u32::try_from((SEL.len() * RPE).div_ceil(ROWS_PER_BLOCK))?,
                THREADS,
                0,
            ))?;
            c.gm.kq_gemv_sel_q4k(
                stream,
                &prep,
                st.w.buf(),
                q,
                s8,
                d8,
                &sel_dev,
                E as u32,
                RPE as u32,
                SEL.len() as u32,
                n_sb as u32,
                n_sb.div_ceil(4) as u32,
                c.gpu.unlabelled_sink(),
                &mut yg,
            )?;
            stream.synchronize()?;
            let (want, got) = (yw.to_host_vec(stream)?, yg.to_host_vec(stream)?);
            let same = bits_equal(&got, &want);
            let host_untouched = got[4 * RPE..5 * RPE]
                .iter()
                .all(|v| v.to_bits() == SENT.to_bits());
            let pass = same && host_untouched && c.gpu.take_fault()?.is_none();
            ok &= pass;
            println!(
                "q4k_sibling[K={k}] n_sb={n_sb} sel={SEL:?} walk_a_q4k_is_q4k_gemv_sel={same} \
                 host_slot_untouched={host_untouched} {}{}",
                verdict(pass),
                mismatch("first_mismatch", &got, &want, RPE),
            );
        }
        Ok(ok)
    }
}
