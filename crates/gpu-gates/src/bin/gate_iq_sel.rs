//! GPU gate for the i-quant expert-select entries over raw blocks
//! (`bloomery_gpu::iq_sel`) on one card: the IQ3_XXS and IQ4_XS gate·ups
//! and the IQ4_NL down `_sel32`, Qwen3.8 UD-Q3_K_XL's card-leg formats.
//!
//! Weights. The gate·up formats take the synthetic rows `dequant_ref
//! --synthetic` writes (`$BLOOMERY_DATA/ref-synth/<type>.blocks`: 64 rows
//! ggml-quantized then 16 rows of random codes, 4096 values each) cut to ten
//! super-blocks (K = 2,560, Qwen3.8's gate width) and to nine (K = 2,304, a
//! partial last lane iteration), cycled into stacks of 4 experts × 80 rows
//! with a per-expert row phase (so two experts never share a row's bytes)
//! and one case of 2 experts × 640 rows. IQ4_NL has no synthetic set
//! (`tools/ref/dequant_ref.cpp` writes none), so its stacks are seeded
//! random blocks (random nibbles, finite f16 `d` in ±[2^-10, 2^-2]) at K =
//! 640 and K = 608 (19 blocks, an odd count that puts every other row at 2
//! mod 4), 3 experts × 2,560 rows. Every stack is uploaded as the KQuant
//! word stream (`runtime::words::stream_words`), the layout the entries
//! read. Activations are seeded columns uniform on ±2, quantized on the
//! device by the engine's own quantizers (`Gpu::enqueue_quantize_q8_1_cols`
//! into a `Q8Act` for the gate·ups, the plain 32-value quantizer into a
//! `Q8Blocks32` for the down) and read back for the host rules.
//!
//! Clauses, each with at least two assertions:
//! 1. **Host rule, bits.** Every `h` and `y` output equals the host
//!    transcription bit for bit — `lane_partial_host` per lane plus the
//!    butterfly, then `act::apply` with `Act::SwigluClamp { limit: 0.0 }`
//!    (which rounds the same on host and device); the down
//!    `iq4_nl_row_dot32_host` per lane plus the butterfly — at m ∈ {1, 2, 4,
//!    8} columns × 10 slots a column, ids repeating an expert inside a
//!    column and across columns, `HOST` included. `HOST` rows stay at the
//!    sentinel; a rerun is bit-identical.
//! 2. **Cross-layout, bits.** Under the same rule at m = 8, `h` equals
//!    `act::apply` of `iq3_xxs_rows` / `iq4_xs_rows` over `IqFormat::repack`
//!    of the same gate and up rows (the plane kernels, one launch an expert
//!    over all columns), bit for bit.
//! 3. **f64 band.** Against `gguf::dequant_row` × the f32 activations,
//!    summed in f64, within `gate_iq.rs`'s bound derivation (below),
//!    composed through the clamp rule for `h`: `1.1·Eg·|u| + |silu(g64)|·Eu
//!    + 16u·max(|g64·u64|, |h|)` (silu' ≤ 1.1; the gate·ups over every row,
//!    the down over every 128th row of its stacks, its 2,560-row experts
//!    beside the dots the m-sweep reads). A `PIN(2026-10-06):` per format
//!    and K ratchets `max|ours − ref| / max|ref|`.
//! 4. **`Act::SiluMul`**, Qwen3.8's rule: within `1e-6` relative of the
//!    host `silu(g)·u` in f64 over the clause-1 `g`, `u` (the host rule's
//!    bits): the device `silu_mul` is `g/(1 + expf(−g))·u` in f32, its
//!    `expf` at most ~2 ulp from the f64 exponential, and the `1 +`, the
//!    division and the product round once each — under `4·u + 3·u ≈ 4e-7`
//!    of the result. A rerun is bit-identical.
//! 5. **Ids.** `HOST` rows stay at the sentinel and raise nothing; an id
//!    past the stack raises `ExpertId`, read back as `GpuError::Fault`
//!    naming the site, with the gate·up's rows NaN and the down's at the
//!    sentinel; every other slot is the clean launch's. The next clean run
//!    after `Gpu::take_fault` raises nothing.
//! 6. **NaN `d`.** A NaN `d` block and an inf-`d` row (two super-blocks of
//!    one row with identical bytes and opposite `d` signs: the row's inf
//!    terms cancel into NaN) make exactly their rows NaN in both entries,
//!    every other row bit-equal to the clean stack's, no fault.
//! 7. **Capture.** The three entries captured in one graph replay as three
//!    nodes, bit-identical to the eager launches; a second eager launch
//!    equals the first.
//! 8. **Refusals by name.** A wrong type, short words, `k` off the block,
//!    m outside 1..=8, short `sel` or `y`: each an `Err(Shape)` of its
//!    launcher naming the case.
//! 9. **Route vs host dot, a real file** (`--model <path>`): the Qwen3.8
//!    UD-Q3_K_XL first shard's own stacks, four experts a layer of layers 0
//!    (IQ3_XXS/IQ4_NL), 4 (IQ3_XXS/Q8_0) and 2 (IQ4_XS/Q8_0), read from the
//!    split and uploaded as the plan's word streams. The card entries
//!    (`iq_sel`'s, and `q8_0_gemv_sel32` for the Q8_0 downs) and
//!    `qdot::dot_row` on the same rows, each side through its own activation
//!    quantizer (the device's q8_1 of 128 values the gate·ups and 32 the
//!    downs, qdot's own column for the host), rule-composed for the gate·ups
//!    — per output `|card − host| ≤ bound_card + bound_host`, each side
//!    within its own bound of the f64 reference. `bound_card` is clause 3's
//!    derivation; `bound_host` the same activation term at qdot's block size
//!    (256 the K-quants, 32 the others) with the host kernel's arithmetic
//!    band `γ_n·Σ|leaves|`, `n` and the leaf bound `qwen4exp_host`'s
//!    over-count (`4·k/256 + 12` and `k/32 + 12`, a leaf at most
//!    `127·d_B·Σ_{i∈B}|w_i|`), and for IQ4_XS beside those the kernel's
//!    `maddubs` saturation term ([`iq4xs_saturation`]): its pairs of
//!    `(kv + 128)·q` products saturate the i16 lane on real weights, a
//!    deviation from the dequant reference the term counts exactly — qdot's
//!    iq kernels are ik's rule, which saturates, and the card entries
//!    ggml's, which does not.
//!
//! The bound of clause 3 (gate_iq.rs's derivation). A value `x` of a
//! 128-value q8_1 block with scale `d = amax/127` is stored as `q·d` with
//! `|q·d − x| ≤ (d/2)(1 + 256u)`; with exact weights `w` a dot errs by at
//! most `Σ_B (d_B/2)(1 + 256u) Σ_{i∈B} |w_i|` from the activations (32-value
//! blocks for the down, summed the same way). The rule's integer sums are
//! exact, so the arithmetic adds one rounding per scale product and one per
//! fused multiply-add on a lane (two roundings a block for the down's
//! `(A·d)·e` and its add) and five butterfly adds: at most `γ(2·n_it +
//! 6)·Σ_k |t_k|` for the gate·ups and `γ(3·n_it + 6)·Σ_k |t_k|` for the
//! down, `t_k` the lane terms and `n_it` a lane's sub-blocks or blocks. That
//! is the worst case; the expected error is the random-rounding one, `≈
//! (d/√12)·‖w‖` against a dot of size `≈ rms(x)·‖w‖` — for x uniform on ±2
//! about 0.4 % of a typical dot whatever the format, since the weights
//! enter both sides alike.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_iq_sel: built without the `gpu` feature; see `just gate-gpu-iq-sel`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_iq_sel", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::GpuError;
    use bloomery_gpu::fault::Fault;
    use bloomery_gpu::iq::{
        IqFormat, IqKernels, IqRows, SUB_VALUES, lane_partial_host, sub_sums_host,
    };
    use bloomery_gpu::iq_sel::{
        IQ4_NL_BLOCK_BYTES, IqDown, IqGateUp, IqSelKernels, iq4_nl_row_dot32_host,
    };
    use bloomery_gpu::kquant::act::{self, Act};
    use bloomery_gpu::kquant::walk_a_planes;
    use bloomery_gpu::q5::Q8Blocks32;
    use bloomery_gpu::q8_0_sel32::{Q80SelDown, Q80SelKernels};
    use bloomery_gpu::{DeviceTensor, Q8Act};
    use bloomery_gpu_gates::rounding::{U, butterfly, gamma};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, data_dir, verdict};
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::iq_tables::KVALUES_IQ4NL;
    use gguf::quant::{GgmlType, dequant_row, half_to_f32};
    use runtime::words::stream_words;

    /// Slots a column: a token's run (Qwen3.8 routes ten).
    const SLOTS: usize = 10;
    /// The m sweep of clauses 1 and 3.
    const MS: [usize; 4] = [1, 2, 4, 8];
    /// Values per full synthetic row.
    const K_FULL: usize = 4096;
    /// The cut geometries: ten and nine super-blocks.
    const KS: [usize; 2] = [2560, 2304];
    /// What `h` and `y` hold before a launch, so an output a kernel skips
    /// reads back as these bits.
    const SENT: f32 = 1.0e30;
    /// The bit rule of clauses 1, 2 and 5.
    const CLAMP0: Act = Act::SwigluClamp { limit: 0.0 };
    /// The down's clause-3 row sampling (every 128th row of its stacks).
    const NL_BAND_STRIDE: usize = 128;

    /// PIN(2026-10-06): per format and K, the larger `max|ours − f64 ref| /
    /// max|ref|` of the m-sweep, rounded up. The pins sit above the
    /// derivation's 0.4 % dot class because the reference divides by
    /// `silu(g64)·u64`, small on negative-g rows while the error enters
    /// through `1.1·Eg·|u|`; every row is inside that composed bound (the
    /// `f64_band` column), which is the assertion, the pin the ratchet.
    const REL_PIN_GU: [(IqFormat, usize, f64); 4] = [
        (IqFormat::Iq3Xxs, 2560, 1.2e-2),
        (IqFormat::Iq3Xxs, 2304, 2.0e-2),
        (IqFormat::Iq4Xs, 2560, 8.0e-3),
        (IqFormat::Iq4Xs, 2304, 1.1e-2),
    ];
    /// PIN(2026-10-06): the down's, both K, the dot-level 0.4 % class
    /// widened by the sampled rows' spread.
    const REL_PIN_NL: [(usize, f64); 2] = [(640, 8.0e-3), (608, 6.5e-3)];

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

    /// A byte stream as the card holds it: little-endian words, zero-padded
    /// to the whole words `stream_words` gives.
    fn stream_words_of(bytes: &[u8], rows: usize) -> Result<Vec<u32>, GateError> {
        let total = stream_words(bytes.len() as u64, rows as u64)
            .ok_or("gate_iq_sel: a stack of no rows")? as usize;
        let mut w = vec![0u32; total];
        for (i, c) in bytes.chunks(4).enumerate() {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            w[i] = u32::from_le_bytes(b);
        }
        Ok(w)
    }

    /// One gate·up geometry's stacks: the format's synthetic rows cut to `k`
    /// and cycled into `experts · rpe` rows with a per-expert phase.
    struct GuStack {
        fmt: IqFormat,
        g: Vec<u8>,
        u: Vec<u8>,
        experts: usize,
        rpe: usize,
        k: usize,
    }

    impl GuStack {
        fn row_bytes(&self) -> usize {
            self.k / 256 * self.fmt.block_bytes()
        }

        /// Absolute row `ra`'s file bytes of one stack.
        fn row(&self, up: bool, ra: usize) -> &[u8] {
            let rb = self.row_bytes();
            let bytes = if up { &self.u } else { &self.g };
            &bytes[ra * rb..(ra + 1) * rb]
        }

        /// One stack's words, uploaded.
        fn upload(&self, s: &CudaStream, up: bool) -> Result<DeviceTensor<u32>, GateError> {
            let bytes = if up { &self.u } else { &self.g };
            let rows = self.experts * self.rpe;
            let w = stream_words_of(bytes, rows)?;
            let cols = w.len() / rows;
            Ok(DeviceTensor::upload(s, &w, rows, cols)?)
        }
    }

    /// The synthetic rows of `fmt` cut to `n_sb` super-blocks, or the error.
    fn synth_rows(fmt: IqFormat, n_sb: usize) -> Result<Vec<Vec<u8>>, GateError> {
        let name = fmt.ggml().name().ok_or("an IqFormat names its type")?;
        let path = data_dir().join("ref-synth").join(format!("{name}.blocks"));
        let all = std::fs::read(&path)
            .map_err(|e| format!("{}: {e} (run `just gate-1-1` first)", path.display()))?;
        let bb = fmt.block_bytes();
        let row_bytes = K_FULL / 256 * bb;
        if all.is_empty() || !all.len().is_multiple_of(row_bytes) {
            return Err(format!(
                "{}: {} bytes is not whole rows of {row_bytes}",
                path.display(),
                all.len()
            )
            .into());
        }
        let keep = n_sb * bb;
        Ok(all.chunks(row_bytes).map(|r| r[..keep].to_vec()).collect())
    }

    /// A gate·up stack of `experts · rpe` rows over the format's synthetic
    /// rows, expert `e`'s row `i` the synth row `(17·e + i) % 80` (so two
    /// experts never share a row's bytes), the up stack a phase apart (so a
    /// gate/up mixup shows in the bits, not as a pass).
    fn gu_stack(fmt: IqFormat, k: usize, experts: usize, rpe: usize) -> Result<GuStack, GateError> {
        let rows = synth_rows(fmt, k / 256)?;
        let n = rows.len();
        let mut g = Vec::with_capacity(experts * rpe * k / 256 * fmt.block_bytes());
        let mut u = Vec::new();
        for e in 0..experts {
            for i in 0..rpe {
                g.extend_from_slice(&rows[(17 * e + i) % n]);
                u.extend_from_slice(&rows[(17 * e + i + 29) % n]);
            }
        }
        Ok(GuStack {
            fmt,
            g,
            u,
            experts,
            rpe,
            k,
        })
    }

    /// One IQ4_NL stack: seeded random blocks, finite f16 `d` in ±[2^-10,
    /// 2^-2] (exponent field 5..=13), random nibbles.
    struct NlStack {
        bytes: Vec<u8>,
        experts: usize,
        rpe: usize,
        k_blocks: usize,
    }

    impl NlStack {
        fn k(&self) -> usize {
            32 * self.k_blocks
        }

        fn upload(&self, s: &CudaStream) -> Result<DeviceTensor<u32>, GateError> {
            let rows = self.experts * self.rpe;
            let w = stream_words_of(&self.bytes, rows)?;
            let cols = w.len() / rows;
            Ok(DeviceTensor::upload(s, &w, rows, cols)?)
        }
    }

    fn nl_stack(k_blocks: usize, experts: usize, rpe: usize, seed: u64) -> NlStack {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut bytes = Vec::with_capacity(experts * rpe * k_blocks * IQ4_NL_BLOCK_BYTES);
        for _ in 0..experts * rpe * k_blocks {
            let e = 5 + (next() % 9) as u16;
            let d = ((next() as u16 & 0x8000) | (e << 10) | (next() as u16 & 0x3ff)).to_le_bytes();
            bytes.extend_from_slice(&d);
            for _ in 0..16 {
                bytes.push((next() >> 32) as u8);
            }
        }
        NlStack {
            bytes,
            experts,
            rpe,
            k_blocks,
        }
    }

    /// The q8_1 quantizer's Q4_K slot of value-order word `v` (gate_iq's).
    fn q4_slot(v: usize) -> usize {
        256 * (v >> 8) + 32 * (v & 7) + 8 * ((v >> 6) & 3) + ((v >> 3) & 7)
    }

    /// One activation column: value-order int8 codes and block scales.
    type Col = (Vec<i8>, Vec<f32>);

    /// A `Q8Act`'s device-quantized columns read back through
    /// `walk_a_planes`: value-order codes and block scales a column.
    fn act_cols(act: &Q8Act, s: &CudaStream) -> Result<Vec<Col>, GateError> {
        let (q4, _s8, d8) = walk_a_planes(act);
        let (q4, d8) = (q4.to_host_vec(s)?, d8.to_host_vec(s)?);
        let (m, k) = (act.m(), 256 * act.n_sb());
        let col_words = 256 * (k / 256).div_ceil(4);
        let mut cols = Vec::with_capacity(m);
        for c in 0..m {
            let mut xq = vec![0i8; k];
            for v in 0..k / 4 {
                let b = q4[c * col_words + q4_slot(v)].to_le_bytes();
                for i in 0..4 {
                    xq[4 * v + i] = b[i] as i8;
                }
            }
            cols.push((xq, d8[c * k / 128..(c + 1) * k / 128].to_vec()));
        }
        Ok(cols)
    }

    /// `cols` columns of `k` values quantized by the engine's plain 32-value
    /// q8_1 quantizer and read back: the activation and its value-order
    /// codes and block scales a column.
    fn blocks32_cols(
        gpu: &Gpu,
        x: &[f32],
        k: usize,
        cols: usize,
    ) -> Result<(Q8Blocks32, Vec<Col>), GateError> {
        let s = gpu.stream();
        let xd = DeviceBuffer::from_host(s, &x[..cols * k])?;
        let mut act = Q8Blocks32::with_slots(s, k, cols)?;
        gpu.q5()
            .enqueue_quantize_q8(s, &xd, &mut act, gpu.unlabelled_sink())?;
        s.synchronize()?;
        let h = act.readback(s)?;
        let kb = k / 32;
        let q_stride = h.q.len() / cols;
        let mut out = Vec::with_capacity(cols);
        for c in 0..cols {
            let mut xq = vec![0i8; k];
            for b in 0..kb {
                for i in 0..8 {
                    let b8 = h.q[c * q_stride + 256 * (b >> 5) + 32 * i + (b & 31)].to_le_bytes();
                    for v in 0..4 {
                        xq[32 * b + 4 * i + v] = b8[v] as i8;
                    }
                }
            }
            out.push((xq, h.d8[c * kb..(c + 1) * kb].to_vec()));
        }
        Ok((act, out))
    }

    /// The gate·up entry's output for `sel` into a SENT-filled buffer,
    /// synchronized: the rows and the fault word.
    fn gate_up(
        c: &Ctx,
        st: &GuStack,
        wg: &DeviceTensor<u32>,
        wu: &DeviceTensor<u32>,
        act: &Q8Act,
        sel: &[u32],
        rule: Act,
    ) -> Result<(Vec<f32>, Option<Fault>), GateError> {
        let stream = c.gpu.stream();
        let sel_dev = DeviceBuffer::from_host(stream, sel)?;
        let mut h = DeviceBuffer::from_host(stream, &vec![SENT; sel.len() * st.rpe])?;
        c.iqsel.enqueue_gate_up(
            stream,
            &IqGateUp {
                ty: st.fmt.ggml(),
                wg,
                wu,
                act,
                sel: &sel_dev,
                n_slots: sel.len(),
                rows_per_expert: st.rpe,
                slots_per_col: SLOTS,
                rule,
            },
            c.gpu.unlabelled_sink(),
            &mut h,
        )?;
        stream.synchronize()?;
        Ok((h.to_host_vec(stream)?, c.gpu.take_fault()?))
    }

    /// The down entry's output for `sel` into a SENT-filled buffer,
    /// synchronized: the rows and the fault word.
    fn down(
        c: &Ctx,
        st: &NlStack,
        w: &DeviceTensor<u32>,
        act: &Q8Blocks32,
        sel: &[u32],
    ) -> Result<(Vec<f32>, Option<Fault>), GateError> {
        let stream = c.gpu.stream();
        let sel_dev = DeviceBuffer::from_host(stream, sel)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; sel.len() * st.rpe])?;
        c.iqsel.enqueue_down(
            stream,
            &IqDown {
                w,
                act,
                sel: &sel_dev,
                n_slots: sel.len(),
                rows_per_expert: st.rpe,
            },
            c.gpu.unlabelled_sink(),
            &mut y,
        )?;
        stream.synchronize()?;
        Ok((y.to_host_vec(stream)?, c.gpu.take_fault()?))
    }

    /// The clause-1 ids of m columns: repeats inside a column and across
    /// columns (the last slot cycles with the column), `HOST` at slot 8 of
    /// every column.
    fn sel_of(m: usize, experts: usize) -> Vec<u32> {
        let fold = |v: u32| {
            if v == u32::MAX || experts >= 4 {
                v
            } else {
                v % experts as u32
            }
        };
        (0..m)
            .flat_map(|c| {
                [0u32, 1, 2, 3, 2, 1, 0, u32::MAX, (c % experts) as u32, 2]
                    .iter()
                    .map(|&v| fold(v))
                    .collect::<Vec<u32>>()
            })
            .collect()
    }

    /// One row's dot by the host rule: the 32 lane partials, then the
    /// butterfly.
    fn dot_host(fmt: IqFormat, row: &[u8], col: &(Vec<i8>, Vec<f32>)) -> f32 {
        butterfly(std::array::from_fn(|lane| {
            lane_partial_host(fmt, row, &col.0, &col.1, lane)
        }))
    }

    /// The gate·up's host rule over `sel`: `SENT` for `HOST` slots and ids
    /// past the stack, `act::apply` of the two host-rule dots elsewhere.
    fn gu_host(st: &GuStack, cols: &[Col], sel: &[u32], rule: Act) -> Vec<f32> {
        let (act, limit) = rule.code();
        let mut h = vec![SENT; sel.len() * st.rpe];
        for (s, &id) in sel.iter().enumerate() {
            if id == u32::MAX || id as usize >= st.experts {
                continue;
            }
            let col = &cols[s / SLOTS];
            for r in 0..st.rpe {
                let ra = id as usize * st.rpe + r;
                let g = dot_host(st.fmt, st.row(false, ra), col);
                let u = dot_host(st.fmt, st.row(true, ra), col);
                h[s * st.rpe + r] = act::apply(act, limit, g, u);
            }
        }
        h
    }

    /// The down's host rule over `sel`: `SENT` for `HOST` slots and ids
    /// past the stack.
    fn down_host(st: &NlStack, w: &[u32], cols: &[Col], sel: &[u32]) -> Vec<f32> {
        let mut y = vec![SENT; sel.len() * st.rpe];
        for (s, &id) in sel.iter().enumerate() {
            if id == u32::MAX || id as usize >= st.experts {
                continue;
            }
            let col = &cols[s];
            for r in 0..st.rpe {
                let ra = id as usize * st.rpe + r;
                y[s * st.rpe + r] = butterfly(std::array::from_fn(|lane| {
                    iq4_nl_row_dot32_host(w, &col.0, &col.1, st.k_blocks, ra, lane)
                }));
            }
        }
        y
    }

    /// `Σ_k |t_k|` of the rule's lane terms for one row and column, and the
    /// largest lane iteration count `n_it` (gate_iq's term_mag).
    fn term_mag(fmt: IqFormat, row: &[u8], col: &(Vec<i8>, Vec<f32>)) -> (f64, usize) {
        let bb = fmt.block_bytes();
        let codes = col.0.as_chunks::<SUB_VALUES>().0;
        let n_sub = 8 * row.len() / bb;
        let mut mag = 0.0f64;
        for (b, xq) in codes.iter().enumerate().take(n_sub) {
            let blk = &row[bb * (b / 8)..bb * (b / 8 + 1)];
            let (i, _) = sub_sums_host(fmt, blk, b % 8, xq);
            let dx = f64::from(col.1[b / 4]);
            let h = f64::from(half_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
            mag += match fmt {
                IqFormat::Iq3Xxs => (h / 4.0 * dx * f64::from(i)).abs(),
                _ => (h * dx * f64::from(i)).abs(),
            };
        }
        (mag, n_sub.div_ceil(32))
    }

    /// The exact dot of a dequantized row with f32 activations, and the
    /// module doc's bound on our error (gate_iq's dot_ref over 128-value
    /// blocks).
    fn dot_ref(
        fmt: IqFormat,
        row: &[u8],
        w: &[f32],
        x: &[f32],
        col: &(Vec<i8>, Vec<f32>),
    ) -> (f64, f64) {
        let (mut exact, mut quant) = (0.0f64, 0.0f64);
        for (b, (wb, xb)) in w.chunks(128).zip(x.chunks(128)).enumerate() {
            let d = f64::from(col.1[b]);
            let mut sw = 0.0f64;
            for (&wi, &xi) in wb.iter().zip(xb) {
                exact += f64::from(wi) * f64::from(xi);
                sw += f64::from(wi).abs();
            }
            quant += d / 2.0 * (1.0 + 256.0 * U) * sw;
        }
        let (mag, n_it) = term_mag(fmt, row, col);
        (exact, quant + gamma(2 * n_it + 6) * mag)
    }

    /// `silu` in f64 (the clause-3 and clause-4 references).
    fn silu64(g: f64) -> f64 {
        g / (1.0 + (-g).exp())
    }

    /// The first index two f32 vectors differ at, printed for a failed bits
    /// check.
    fn mismatch(what: &str, a: &[f32], b: &[f32]) -> String {
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            if x.to_bits() != y.to_bits() {
                return format!(
                    " {what}[{i}]: {x:e} ({:#010x}) vs {y:e} ({:#010x})",
                    x.to_bits(),
                    y.to_bits()
                );
            }
        }
        String::new()
    }

    /// A fault as the gate prints it.
    fn shown(f: &Option<Fault>) -> String {
        match f {
            None => "none".to_string(),
            Some(f) => format!("{f:#}"),
        }
    }

    /// `true` when every `HOST` slot's rows still hold the sentinel.
    fn host_slots_sent(sel: &[u32], out: &[f32], rpe: usize) -> bool {
        sel.iter()
            .enumerate()
            .filter(|&(_, &id)| id == u32::MAX)
            .all(|(si, _)| {
                out[si * rpe..(si + 1) * rpe]
                    .iter()
                    .all(|&v| v.to_bits() == SENT.to_bits())
            })
    }

    struct Ctx {
        gpu: Gpu,
        iqsel: IqSelKernels,
        iq: IqKernels,
        q80: Q80SelKernels,
    }

    /// One side's dot band of a dequantized row `w` with an activation
    /// column `x` quantized in `block`-value blocks of scale `amax/127`: the
    /// activation term `Σ_B (d_B/2)(1+256u)·Σ_{i∈B}|w_i|` plus the kernel's
    /// arithmetic band `γ_n·Σ|leaves|`, `n` over-counted as
    /// `qwen4exp_host`'s (`4·k/256 + 12` at 256, `k/32 + 12` at 32). The
    /// iq and K-quant kernels sum each block's `ŵ·x̂` in exact integers
    /// under one scale product a term pair (the value sum and the min
    /// term's block sums, `dot_iq4xs_q8k_emul`), so a leaf's magnitude is
    /// the block's real value: at most `max|x̂|·Σ_{i∈B}|w_i|`, and
    /// `|x̂| ≤ 127·d_B` (clause 9).
    fn host_dot_band(w: &[f32], x: &[f32], block: usize) -> f64 {
        let n = if block == 256 {
            4 * w.len() / 256 + 12
        } else {
            w.len() / 32 + 12
        };
        let (mut act_t, mut leaf_t) = (0.0f64, 0.0f64);
        for (wb, xb) in w.chunks(block).zip(x.chunks(block)) {
            let amax = xb.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let d = f64::from(amax / 127.0);
            let sw: f64 = wb.iter().map(|&v| f64::from(v.abs())).sum();
            act_t += d / 2.0 * (1.0 + 256.0 * U) * sw;
            leaf_t += 127.0 * d * sw;
        }
        act_t + gamma(n) * leaf_t
    }

    /// The IQ4_XS host dot's saturation term (clause 9): qdot's kernel sums
    /// each pair of `(kv + 128)·q` products in one i16 `maddubs` lane, which
    /// saturates at 32,767 — `kv`'s largest entry is 113, a code at most
    /// 127, a pair at most `2·241·127 = 61,214` — and a pair that passes it
    /// loses its excess to the clamp, `dot_iq4xs_q8k_emul`'s own mirror of
    /// the instruction. The excess is an exact integer over the row's codes
    /// and the column's codes (`quantize_col_scalar`, the bit-identical
    /// twin the crate documents); each pair's weight in the dot is
    /// `d·dy·sc`, so this returns the lost excess times that scale, block
    /// by block. IQ3_XXS cannot saturate (`|kv + 64| ≤ 71`, a pair at most
    /// `2·71·127 = 18,034`), and the 32-value cells fold the weight's sign
    /// into the activation with `|w| ≤ 128` (`2·128·127 = 32,512`), so the
    /// downs need no term.
    fn iq4xs_saturation(row: &[u8], x: &[f32]) -> f64 {
        let kb = x.len() / 256;
        let mut cb = vec![0u8; qdot::col_bytes(GgmlType::IQ4_XS, x.len())];
        qdot::quantize_col_scalar(GgmlType::IQ4_XS, x, &mut cb);
        const KV: [i32; 16] = [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        let mut out = 0.0f64;
        for i in 0..kb {
            let blk = &row[136 * i..136 * (i + 1)];
            let d = f64::from(half_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
            let dy = f64::from(f32::from_le_bytes(
                cb[296 * i..296 * i + 4]
                    .try_into()
                    .expect("the block's scale"),
            ));
            let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
            let scales_l = &blk[4..8];
            let sc = |b: usize| {
                let lo = (scales_l[b / 2] >> (4 * (b % 2))) & 0x0f;
                let hi = ((scales_h >> (2 * b)) & 3) as u8;
                i32::from(lo | (hi << 4)) - 32
            };
            for b in 0..8 {
                let qs = &blk[8 + 16 * b..8 + 16 * b + 16];
                let qc = |n: usize| i32::from(cb[296 * i + 8 + 32 * b + n] as i8);
                for p in 0..16 {
                    let mut v = 0i64;
                    for j in [2 * p, 2 * p + 1] {
                        let kv = if j < 16 {
                            KV[usize::from(qs[j] & 0x0f)]
                        } else {
                            KV[usize::from(qs[j - 16] >> 4)]
                        };
                        v += i64::from(kv + 128) * i64::from(qc(j));
                    }
                    let excess = v.unsigned_abs().saturating_sub(32_767);
                    out += f64::from(u32::try_from(excess).unwrap_or(0))
                        * (d * dy * f64::from(sc(b))).abs();
                }
            }
        }
        out
    }

    /// One stack of the real file over four experts, as the plan uploads it:
    /// their file bytes in id order, the word stream, and the per-row byte
    /// width of its type.
    struct FileStack {
        ty: GgmlType,
        k: usize,
        rows_per: usize,
        words: Vec<u32>,
        bytes: Vec<u8>,
        per: usize,
        row_bytes: usize,
    }

    /// Layer `l`'s stack `name` of `ty` over `ids`, read from `split`.
    fn file_stack(
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        split: &Split,
        name: &str,
        ty: GgmlType,
        ids: &[u32],
    ) -> Result<FileStack, GateError> {
        let t = inputs
            .model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("the file holds no {name}"))?;
        if t.ty != ty {
            return Err(format!("{name} is {}, the clause reads a {ty} stack", t.ty).into());
        }
        let experts = inputs.model.experts;
        let per = usize::try_from(t.file_bytes / experts).unwrap_or(0);
        let (s, info) = split
            .find(name)
            .ok_or_else(|| format!("the split holds no {name}"))?;
        let whole = split
            .shard(s)
            .ok_or_else(|| format!("shard {s}"))?
            .data(info)
            .map_err(|e| format!("{name}: {e}"))?;
        let mut bytes = Vec::with_capacity(per * ids.len());
        for &id in ids {
            let at = id as usize * per;
            bytes.extend_from_slice(
                whole
                    .get(at..at + per)
                    .ok_or_else(|| format!("{name}: expert {id} past its bytes"))?,
            );
        }
        let (k, rows_per) = (
            usize::try_from(t.dims[0]).unwrap_or(0),
            usize::try_from(t.dims[1]).unwrap_or(0),
        );
        let row_bytes =
            (k as u64 / ty.blck_size().unwrap_or(1) * ty.type_size().unwrap_or(1)) as usize;
        let rows = rows_per * ids.len();
        let words = stream_words_of(&bytes, rows)?;
        Ok(FileStack {
            ty,
            k,
            rows_per,
            words,
            bytes,
            per,
            row_bytes,
        })
    }

    impl FileStack {
        /// Expert `e`'s row `r`, its file bytes and its dequantized values.
        fn row(&self, e: usize, r: usize) -> &[u8] {
            &self.bytes[self.per * e + r * self.row_bytes..][..self.row_bytes]
        }

        fn dequant(&self, e: usize, r: usize, out: &mut [f32]) -> Result<(), GateError> {
            dequant_row(self.ty, self.row(e, r), out).map_err(|e| e.to_string().into())
        }

        /// The device upload of the four experts' rows.
        fn upload(&self, s: &CudaStream) -> Result<DeviceTensor<u32>, GateError> {
            let rows = self.rows_per * 4;
            Ok(DeviceTensor::upload(
                s,
                &self.words,
                rows,
                self.words.len() / rows,
            )?)
        }
    }

    /// The rule-composed band of one side's `h` against the f64 reference
    /// (clause 3's composition, `silu' ≤ 1.1`): `1.1·Eg·|u| + |silu(g)|·Eu`
    /// over that side's own two dot bands.
    fn silu_band(eg: f64, g64: f64, eu: f64, u64v: f64) -> f64 {
        1.1 * eg * u64v.abs() + silu64(g64).abs() * eu
    }

    /// Clause 9 (module doc): the route-vs-host dots over the real file's
    /// own stacks. Returns whether every assertion held.
    fn check_real_file(c: &Ctx, path: &str) -> Result<bool, GateError> {
        let split = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let inputs = model::arch::qwen35moe::place::PlanInputs::describe(&split)
            .map_err(|e| format!("describe {path}: {e}"))?;
        let ids: [u32; 4] = [0, 1, 17, 299];
        // The layer, its gate·up type and its down type, as the design
        // table reads the file.
        let layers: [(usize, GgmlType, GgmlType); 3] = [
            (0, GgmlType::IQ3_XXS, GgmlType::IQ4_NL),
            (4, GgmlType::IQ3_XXS, GgmlType::Q8_0),
            (2, GgmlType::IQ4_XS, GgmlType::Q8_0),
        ];
        let s = c.gpu.stream();
        let mut ok = true;
        for (l, gu_ty, down_ty) in layers {
            let [gn, un, dn] = [
                model::arch::qwen35moe::names::ffn_gate_exps(l),
                model::arch::qwen35moe::names::ffn_up_exps(l),
                model::arch::qwen35moe::names::ffn_down_exps(l),
            ];
            let (g, u, d) = (
                file_stack(&inputs, &split, &gn, gu_ty, &ids)?,
                file_stack(&inputs, &split, &un, gu_ty, &ids)?,
                file_stack(&inputs, &split, &dn, down_ty, &ids)?,
            );
            let (mut card_max, mut host_max, mut cross_max, mut ref_max) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let (mut card_ok, mut host_ok, mut cross_ok) = (true, true, true);
            let (mut worst_si, mut worst_r, mut worst) = (0usize, 0usize, f64::NEG_INFINITY);
            let (mut worst_part, mut worst_vals) = (String::new(), String::new());

            // The gate·ups: four columns of ten slots, every slot one of the
            // four experts, the card leg's own rule composing the dots.
            let m = 4usize;
            let fmt = match gu_ty {
                GgmlType::IQ3_XXS => IqFormat::Iq3Xxs,
                _ => IqFormat::Iq4Xs,
            };
            let x = seeded(m * g.k, 5101 + l as u32, -2.0, 4.0);
            let xd = DeviceBuffer::from_host(s, &x)?;
            let mut act = Q8Act::with_k(s, m, g.k)?;
            c.gpu.enqueue_quantize_q8_1_cols(&xd, &mut act, m, 0)?;
            s.synchronize()?;
            let cols = act_cols(&act, s)?;
            let sel: Vec<u32> = (0..m)
                .flat_map(|_| [0u32, 1, 2, 3, 0, 1, 2, 3, 0, 1])
                .collect();
            let sel_dev = DeviceBuffer::from_host(s, &sel)?;
            let (wg, wu) = (g.upload(s)?, u.upload(s)?);
            let mut h = DeviceBuffer::from_host(s, &vec![SENT; sel.len() * g.rows_per])?;
            c.iqsel.enqueue_gate_up(
                s,
                &IqGateUp {
                    ty: gu_ty,
                    wg: &wg,
                    wu: &wu,
                    act: &act,
                    sel: &sel_dev,
                    n_slots: sel.len(),
                    rows_per_expert: g.rows_per,
                    slots_per_col: SLOTS,
                    rule: Act::SiluMul,
                },
                c.gpu.unlabelled_sink(),
                &mut h,
            )?;
            s.synchronize()?;
            let h_card = h.to_host_vec(s)?;
            // The host's own columns, qdot's quantizer a token.
            let host_cols: Vec<Vec<u8>> = (0..m)
                .map(|col| {
                    let mut cb = vec![0u8; qdot::col_bytes(gu_ty, g.k)];
                    qdot::quantize_col(gu_ty, &x[col * g.k..][..g.k], &mut cb);
                    cb
                })
                .collect();
            let mut w_f32 = vec![0.0f32; g.k];
            let mut w_up = vec![0.0f32; u.k];
            for (si, &e) in sel.iter().enumerate() {
                let col = si / SLOTS;
                for r in 0..g.rows_per {
                    // The f64 reference and the card's band: clause 3's
                    // machinery over the same rows.
                    g.dequant(e as usize, r, &mut w_f32)?;
                    let (g64, eg) = dot_ref(
                        fmt,
                        g.row(e as usize, r),
                        &w_f32,
                        &x[col * g.k..],
                        &cols[col],
                    );
                    u.dequant(e as usize, r, &mut w_up)?;
                    let (u64v, eu) = dot_ref(
                        fmt,
                        u.row(e as usize, r),
                        &w_up,
                        &x[col * g.k..],
                        &cols[col],
                    );
                    let hv = f64::from(h_card[si * g.rows_per + r]);
                    let reference = silu64(g64) * u64v;
                    let bound_card = 1.1 * eg * u64v.abs()
                        + silu64(g64).abs() * eu
                        + 16.0 * U * (g64 * u64v).abs().max(hv.abs());
                    // The host's dots and their band over its own columns.
                    let (gh, uh) = (
                        f64::from(
                            qdot::dot_row(g.ty, g.row(e as usize, r), &host_cols[col], g.k)
                                .expect("a fused dot"),
                        ),
                        f64::from(
                            qdot::dot_row(u.ty, u.row(e as usize, r), &host_cols[col], u.k)
                                .expect("a fused dot"),
                        ),
                    );
                    let (ehg, ehu) = (
                        host_dot_band(&w_f32, &x[col * g.k..], 256)
                            + if gu_ty == GgmlType::IQ4_XS {
                                iq4xs_saturation(
                                    g.row(e as usize, r),
                                    &x[col * g.k..(col + 1) * g.k],
                                )
                            } else {
                                0.0
                            },
                        host_dot_band(&w_up, &x[col * u.k..], 256)
                            + if gu_ty == GgmlType::IQ4_XS {
                                iq4xs_saturation(
                                    u.row(e as usize, r),
                                    &x[col * u.k..(col + 1) * u.k],
                                )
                            } else {
                                0.0
                            },
                    );
                    let host_h = silu64(gh) * uh;
                    let bound_host = silu_band(ehg, gh, ehu, uh);
                    let host_err = (host_h - reference).abs();
                    if host_err - bound_host > worst {
                        (worst_si, worst_r, worst) = (si, r, host_err - bound_host);
                        worst_part = "gu".to_string();
                        worst_vals = format!(
                            "g64 {g64:.4} u64 {u64v:.4} gh {gh:.4} eg {eg:.3e} uh {uh:.4} eu \
                             {eu:.3e} host {host_h:.4} ref {reference:.4} band {bound_host:.3e} \
                             row_u {row_u:02x?} x {xv:?} w_up {wu:?} w_tail {wt:?} x_tail {xt:?}",
                            row_u = &u.row(e as usize, r)[..16],
                            xv = &x[col * g.k..][..4],
                            wu = &w_up[..2],
                            wt = &w_up[1000..1004],
                            xt = &x[col * g.k + 1000..][..4],
                        );
                    }
                    card_ok &= (hv - reference).abs() <= bound_card;
                    host_ok &= host_err <= bound_host;
                    cross_ok &= (hv - host_h).abs() <= bound_card + bound_host;
                    card_max = card_max.max((hv - reference).abs());
                    host_max = host_max.max(host_err);
                    cross_max = cross_max.max((hv - host_h).abs());
                    ref_max = ref_max.max(reference.abs());
                }
            }

            // The downs: one column a slot, the IQ4_NL entry or the Q8_0
            // one against qdot's own column.
            let n_slots = sel.len();
            let x = seeded(n_slots * d.k, 5201 + l as u32, -2.0, 4.0);
            let (act_d, cols_d) = blocks32_cols(&c.gpu, &x, d.k, n_slots)?;
            let wd = d.upload(s)?;
            let mut y = DeviceBuffer::from_host(s, &vec![SENT; n_slots * d.rows_per])?;
            match down_ty {
                GgmlType::IQ4_NL => {
                    c.iqsel.enqueue_down(
                        s,
                        &IqDown {
                            w: &wd,
                            act: &act_d,
                            sel: &sel_dev,
                            n_slots,
                            rows_per_expert: d.rows_per,
                        },
                        c.gpu.unlabelled_sink(),
                        &mut y,
                    )?;
                }
                GgmlType::Q8_0 => {
                    c.q80.enqueue_gemv_q8_0_sel32(
                        s,
                        &Q80SelDown {
                            w: &wd,
                            act: &act_d,
                            sel: &sel_dev,
                            n_slots,
                            rows_per_expert: d.rows_per,
                        },
                        c.gpu.unlabelled_sink(),
                        &mut y,
                    )?;
                }
                other => panic!("the down clause reads IQ4_NL or Q8_0, not {other}"),
            }
            s.synchronize()?;
            let y_card = y.to_host_vec(s)?;
            let host_cols: Vec<Vec<u8>> = (0..n_slots)
                .map(|si| {
                    let mut cb = vec![0u8; qdot::col_bytes(down_ty, d.k)];
                    qdot::quantize_col(down_ty, &x[si * d.k..][..d.k], &mut cb);
                    cb
                })
                .collect();
            let mut w_d = vec![0.0f32; d.k];
            for (si, &e) in sel.iter().enumerate() {
                for r in 0..d.rows_per {
                    d.dequant(e as usize, r, &mut w_d)?;
                    let mut reference = 0.0f64;
                    for (wi, xi) in w_d.iter().zip(&x[si * d.k..][..d.k]) {
                        reference += f64::from(*wi) * f64::from(*xi);
                    }
                    let yv = f64::from(y_card[si * d.rows_per + r]);
                    let yh = f64::from(
                        qdot::dot_row(d.ty, d.row((e as usize + 1) % 4, r), &host_cols[si], d.k)
                            .expect("a fused dot"),
                    );
                    // The card's band over the read-back column: the
                    // activation term of its 32-value blocks and its
                    // `(A·d_w)·e` terms bounded by `127·|e_B|·Σ|w|`, the
                    // arithmetic `γ(3·n_it + 6)` (check_nl's derivation).
                    let kb = d.k / 32;
                    let (mut act_t, mut mag) = (0.0f64, 0.0f64);
                    for b in 0..kb {
                        let e_b = f64::from(cols_d[si].1[b]);
                        let sw: f64 = w_d[32 * b..32 * b + 32]
                            .iter()
                            .map(|&v| f64::from(v.abs()))
                            .sum();
                        act_t += e_b / 2.0 * (1.0 + 256.0 * U) * sw;
                        mag += 127.0 * e_b * sw;
                    }
                    let n_it = kb.div_ceil(32);
                    let bound_card = act_t + gamma(3 * n_it + 6) * mag;
                    let bound_host =
                        host_dot_band(&w_d, &x[si * d.k..][..d.k], qdot::k_granularity(down_ty));
                    let host_err = (yh - reference).abs();
                    if host_err - bound_host > worst {
                        (worst_si, worst_r, worst) = (si, r, host_err - bound_host);
                        worst_part = "down".to_string();
                        worst_vals = format!(
                            "yh {yh:.4} ref {reference:.4} band {bound_host:.3e} card {yv:.4}"
                        );
                    }
                    card_ok &= (yv - reference).abs() <= bound_card;
                    host_ok &= host_err <= bound_host;
                    cross_ok &= (yv - yh).abs() <= bound_card + bound_host;
                    card_max = card_max.max((yv - reference).abs());
                    host_max = host_max.max(host_err);
                    cross_max = cross_max.max((yv - yh).abs());
                    ref_max = ref_max.max(reference.abs());
                }
            }
            let no_fault = c.gpu.take_fault()?.is_none();
            ok &= card_ok && host_ok && cross_ok && no_fault;
            if !host_ok {
                println!(
                    "real[l={l}] worst host offender ({worst_part}): slot {worst_si} row \
                     {worst_r}, over its band by {worst:.3e}: {worst_vals}",
                );
            }
            println!(
                "real[l={l} gu={gu_ty} down={down_ty}] card_band={} host_band={} cross_band={}                  fault=none {}",
                verdict(card_ok),
                verdict(host_ok),
                verdict(cross_ok),
                verdict(card_ok && host_ok && cross_ok && no_fault),
            );
            println!(
                "real[l={l}] max errs card {card_max:.3e} host {host_max:.3e} cross                  {cross_max:.3e} against |ref| {ref_max:.3e}",
            );
        }
        Ok(ok)
    }

    /// for the 640-row case, whose rows the 4 × 80 geometries already
    /// cover), and the geometry's largest relative error for clause 3's pin.
    fn check_gu(
        c: &Ctx,
        st: &GuStack,
        wg: &DeviceTensor<u32>,
        wu: &DeviceTensor<u32>,
        cross: bool,
        band: bool,
    ) -> Result<(bool, f64), GateError> {
        let s = c.gpu.stream();
        let (mut ok, mut rel) = (true, 0.0f64);
        for &m in MS.iter() {
            let x = seeded(m * st.k, 7001 + st.k as u32 + m as u32, -2.0, 4.0);
            let xd = DeviceBuffer::from_host(s, &x)?;
            let mut act = Q8Act::with_k(s, m, st.k)?;
            c.gpu.enqueue_quantize_q8_1_cols(&xd, &mut act, m, 0)?;
            s.synchronize()?;
            let cols = act_cols(&act, s)?;
            let sel = sel_of(m, st.experts);
            let (h, fault) = gate_up(c, st, wg, wu, &act, &sel, CLAMP0)?;
            let (again, fault2) = gate_up(c, st, wg, wu, &act, &sel, CLAMP0)?;
            let host = gu_host(st, &cols, &sel, CLAMP0);
            let bits = bits_equal(&h, &host);
            let rerun = bits_equal(&h, &again);
            let sentinel = host_slots_sent(&sel, &h, st.rpe);
            let no_fault = fault.is_none() && fault2.is_none();
            ok &= bits && rerun && sentinel && no_fault;

            // Clause 2: the plane kernels over repack of the same rows, m
            // columns, one launch an expert; the _sel h must equal apply of
            // their dots.
            let mut cross_bits = true;
            let mut cross_want = None;
            if cross && m == MS[3] {
                let (q4, _s8, d8) = walk_a_planes(&act);
                let (qd, d8d) = (
                    DeviceBuffer::from_host(s, &q4.to_host_vec(s)?)?,
                    DeviceBuffer::from_host(s, &d8.to_host_vec(s)?)?,
                );
                let mut expert_y = Vec::with_capacity(st.experts);
                let rb = st.row_bytes();
                for e in 0..st.experts {
                    let rg = IqRows::upload(
                        s,
                        st.fmt,
                        &st.g[e * st.rpe * rb..][..st.rpe * rb],
                        st.rpe,
                        st.k,
                    )?;
                    let ru = IqRows::upload(
                        s,
                        st.fmt,
                        &st.u[e * st.rpe * rb..][..st.rpe * rb],
                        st.rpe,
                        st.k,
                    )?;
                    let mut yg = DeviceBuffer::from_host(s, &vec![0.0f32; m * st.rpe])?;
                    let mut yu = DeviceBuffer::from_host(s, &vec![0.0f32; m * st.rpe])?;
                    c.iq.enqueue_rows(s, &rg, &qd, &d8d, m, &mut yg)?;
                    c.iq.enqueue_rows(s, &ru, &qd, &d8d, m, &mut yu)?;
                    expert_y.push((yg.to_host_vec(s)?, yu.to_host_vec(s)?));
                }
                let mut want = vec![SENT; h.len()];
                for (si, &id) in sel.iter().enumerate() {
                    if id == u32::MAX {
                        continue;
                    }
                    let (yg, yu) = &expert_y[id as usize];
                    let col = si / SLOTS;
                    for r in 0..st.rpe {
                        want[si * st.rpe + r] = act::apply(
                            CLAMP0.code().0,
                            0.0,
                            yg[col * st.rpe + r],
                            yu[col * st.rpe + r],
                        );
                    }
                }
                cross_bits = bits_equal(&h, &want);
                cross_want = Some(want);
                ok &= cross_bits;
            }

            // Clause 3: the f64 band over every slot's row and its column.
            let (mut err_max, mut ref_max, mut in_bound) = (0.0f64, 0.0f64, true);
            if band {
                let mut w_f32 = vec![0.0f32; st.k];
                for (si, &id) in sel.iter().enumerate() {
                    if id == u32::MAX {
                        continue;
                    }
                    let col = si / SLOTS;
                    for r in 0..st.rpe {
                        let ra = id as usize * st.rpe + r;
                        let (gbytes, ubytes) = (st.row(false, ra), st.row(true, ra));
                        dequant_row(st.fmt.ggml(), gbytes, &mut w_f32)?;
                        let (g64, eg) =
                            dot_ref(st.fmt, gbytes, &w_f32, &x[col * st.k..], &cols[col]);
                        dequant_row(st.fmt.ggml(), ubytes, &mut w_f32)?;
                        let (u64v, eu) =
                            dot_ref(st.fmt, ubytes, &w_f32, &x[col * st.k..], &cols[col]);
                        let hv = f64::from(h[si * st.rpe + r]);
                        let reference = silu64(g64) * u64v;
                        let bound = 1.1 * eg * u64v.abs()
                            + silu64(g64).abs() * eu
                            + 16.0 * U * (g64 * u64v).abs().max(hv.abs());
                        let err = (hv - reference).abs();
                        in_bound &= err <= bound;
                        err_max = err_max.max(err);
                        ref_max = ref_max.max(reference.abs());
                    }
                }
                ok &= in_bound;
                rel = (err_max / ref_max).max(rel);
            }
            let name = st.fmt.ggml().name().unwrap_or("?");
            println!(
                "gu[{name} k={} experts={} rpe={} m={m}] host_rule_bits={} rerun={} \
                 host_slots_sentinel={} cross_layout={} f64_band={} fault=\"{}\" {}{}",
                st.k,
                st.experts,
                st.rpe,
                verdict(bits),
                verdict(rerun),
                verdict(sentinel),
                verdict(cross_bits),
                verdict(in_bound),
                shown(&fault),
                mismatch("host_rule", &h, &host),
                cross_want
                    .as_ref()
                    .map_or(String::new(), |w| mismatch("cross", &h, w)),
            );
        }
        Ok((ok, rel))
    }

    /// Clause 4: `Act::SiluMul` within its band of the host `silu(g)·u`.
    fn check_silu(
        c: &Ctx,
        st: &GuStack,
        wg: &DeviceTensor<u32>,
        wu: &DeviceTensor<u32>,
    ) -> Result<bool, GateError> {
        let s = c.gpu.stream();
        let m = 4usize;
        let x = seeded(m * st.k, 8801 + st.k as u32, -2.0, 4.0);
        let xd = DeviceBuffer::from_host(s, &x)?;
        let mut act = Q8Act::with_k(s, m, st.k)?;
        c.gpu.enqueue_quantize_q8_1_cols(&xd, &mut act, m, 0)?;
        s.synchronize()?;
        let cols = act_cols(&act, s)?;
        let sel = sel_of(m, st.experts);
        let (h, fault) = gate_up(c, st, wg, wu, &act, &sel, Act::SiluMul)?;
        let (again, _) = gate_up(c, st, wg, wu, &act, &sel, Act::SiluMul)?;
        let (mut err_max, mut ref_max) = (0.0f64, 0.0f64);
        for (si, &id) in sel.iter().enumerate() {
            if id == u32::MAX {
                continue;
            }
            let col = &cols[si / SLOTS];
            for r in 0..st.rpe {
                let ra = id as usize * st.rpe + r;
                let g = f64::from(dot_host(st.fmt, st.row(false, ra), col));
                let u = f64::from(dot_host(st.fmt, st.row(true, ra), col));
                let reference = silu64(g) * u;
                err_max = err_max.max((f64::from(h[si * st.rpe + r]) - reference).abs());
                ref_max = ref_max.max(reference.abs());
            }
        }
        let rel = err_max / ref_max;
        let in_band = rel <= 1.0e-6;
        let rerun = bits_equal(&h, &again);
        let name = st.fmt.ggml().name().unwrap_or("?");
        let pass = in_band && rerun && fault.is_none();
        println!(
            "silu[{name} k={} m={m}] rel={rel:.3e} band=1e-6 {} rerun={} fault=\"{}\" {}",
            st.k,
            verdict(in_band),
            verdict(rerun),
            shown(&fault),
            verdict(pass),
        );
        Ok(pass)
    }

    /// Clauses 1 and 3 over one IQ4_NL geometry (the band on every
    /// `NL_BAND_STRIDE`-th row, module doc).
    fn check_nl(c: &Ctx, st: &NlStack, w: &DeviceTensor<u32>) -> Result<(bool, f64), GateError> {
        let w_host = stream_words_of(&st.bytes, st.experts * st.rpe)?;
        let (mut ok, mut rel) = (true, 0.0f64);
        for &m in MS.iter() {
            let n_slots = SLOTS * m;
            let x = seeded(n_slots * st.k(), 9101 + st.k() as u32 + m as u32, -2.0, 4.0);
            let (act, cols) = blocks32_cols(&c.gpu, &x, st.k(), n_slots)?;
            let sel = sel_of(m, st.experts);
            let (y, fault) = down(c, st, w, &act, &sel)?;
            let (again, fault2) = down(c, st, w, &act, &sel)?;
            let host = down_host(st, &w_host, &cols, &sel);
            let bits = bits_equal(&y, &host);
            let rerun = bits_equal(&y, &again);
            let sentinel = host_slots_sent(&sel, &y, st.rpe);
            ok &= bits && rerun && sentinel && fault.is_none() && fault2.is_none();

            // Clause 3: the f64 dot of dequant_row's row with the f32
            // column, the module doc's bound, over the sampled rows.
            let kv = |c: u8| i64::from(KVALUES_IQ4NL[usize::from(c)]);
            let (mut err_max, mut ref_max, mut in_bound) = (0.0f64, 0.0f64, true);
            let mut w_f32 = vec![0.0f32; st.k()];
            let rb = st.k_blocks * IQ4_NL_BLOCK_BYTES;
            for (si, &id) in sel.iter().enumerate() {
                if id == u32::MAX {
                    continue;
                }
                let col = &cols[si];
                let xc = &x[si * st.k()..][..st.k()];
                for r in (0..st.rpe).step_by(NL_BAND_STRIDE) {
                    let ra = id as usize * st.rpe + r;
                    let bytes = &st.bytes[ra * rb..][..rb];
                    dequant_row(GgmlType::IQ4_NL, bytes, &mut w_f32)?;
                    let (mut exact, mut quant, mut mag) = (0.0f64, 0.0f64, 0.0f64);
                    for b in 0..st.k_blocks {
                        let e = f64::from(col.1[b]);
                        let dw = f64::from(half_to_f32(u16::from_le_bytes([
                            bytes[b * IQ4_NL_BLOCK_BYTES],
                            bytes[b * IQ4_NL_BLOCK_BYTES + 1],
                        ])));
                        let mut sw = 0.0f64;
                        let mut a = 0i64;
                        for j in 0..16 {
                            let byte = bytes[b * IQ4_NL_BLOCK_BYTES + 2 + j];
                            let lo = f64::from(w_f32[32 * b + j]);
                            let hi = f64::from(w_f32[32 * b + 16 + j]);
                            exact += lo * f64::from(xc[32 * b + j]);
                            exact += hi * f64::from(xc[32 * b + 16 + j]);
                            sw += lo.abs() + hi.abs();
                            a += kv(byte & 15) * i64::from(col.0[32 * b + j]);
                            a += kv(byte >> 4) * i64::from(col.0[32 * b + 16 + j]);
                        }
                        quant += e / 2.0 * (1.0 + 256.0 * U) * sw;
                        mag += (a as f64 * dw).abs() * e;
                    }
                    let n_it = st.k_blocks.div_ceil(32);
                    let bound = quant + gamma(3 * n_it + 6) * mag;
                    let err = (f64::from(y[si * st.rpe + r]) - exact).abs();
                    in_bound &= err <= bound;
                    err_max = err_max.max(err);
                    ref_max = ref_max.max(exact.abs());
                }
            }
            ok &= in_bound;
            rel = (err_max / ref_max).max(rel);
            println!(
                "nl[k={} experts={} rpe={} slots={n_slots}] host_rule_bits={} rerun={} \
                 host_slots_sentinel={} f64_band={} fault=\"{}\" {}{}",
                st.k(),
                st.experts,
                st.rpe,
                verdict(bits),
                verdict(rerun),
                verdict(sentinel),
                verdict(in_bound),
                shown(&fault),
                mismatch("host_rule", &y, &host),
                mismatch("rerun", &y, &again),
            );
        }
        Ok((ok, rel))
    }

    /// Clauses 5 and 6 (module doc).
    fn check_ids_and_nan(c: &Ctx) -> Result<bool, GateError> {
        let s = c.gpu.stream();
        let mut ok = true;
        for fmt in [IqFormat::Iq3Xxs, IqFormat::Iq4Xs] {
            // Clause 5, gate·up: an id past the stack and HOST slots.
            let st = gu_stack(fmt, KS[0], 4, 80)?;
            let (wg, wu) = (st.upload(s, false)?, st.upload(s, true)?);
            let m = 2usize;
            let xd = DeviceBuffer::from_host(s, &seeded(m * st.k, 6601, -2.0, 4.0))?;
            let mut act = Q8Act::with_k(s, m, st.k)?;
            c.gpu.enqueue_quantize_q8_1_cols(&xd, &mut act, m, 0)?;
            s.synchronize()?;
            let mut sel = sel_of(m, st.experts);
            sel[3] = u32::MAX - 1;
            let (h, fault) = gate_up(c, &st, &wg, &wu, &act, &sel, CLAMP0)?;
            let bad_nan = h[3 * st.rpe..4 * st.rpe].iter().all(|&v| v.is_nan());
            let sentinel = host_slots_sent(&sel, &h, st.rpe);
            let named = match &fault {
                Some(f) => GpuError::fault("gate_iq_sel", *f)
                    .to_string()
                    .contains("expert_id"),
                None => false,
            };
            let mut clean_sel = sel.clone();
            clean_sel[3] = 0;
            let (clean, _) = gate_up(c, &st, &wg, &wu, &act, &clean_sel, CLAMP0)?;
            let others_same = clean
                .iter()
                .zip(&h)
                .enumerate()
                .all(|(i, (a, b))| i / st.rpe == 3 || a.to_bits() == b.to_bits());
            let (_, clean_fault) = gate_up(c, &st, &wg, &wu, &act, &clean_sel, CLAMP0)?;
            let pass = bad_nan && sentinel && named && others_same && clean_fault.is_none();
            ok &= pass;
            println!(
                "ids_gu[{}] past_stack_nan={} fault_names_expert_id={} host_slots_sentinel={} \
                 others_are_clean={} next_clean_clean={} fault=\"{}\" {}",
                fmt.ggml().name().unwrap_or("?"),
                verdict(bad_nan),
                verdict(named),
                verdict(sentinel),
                verdict(others_same),
                verdict(clean_fault.is_none()),
                shown(&fault),
                verdict(pass),
            );
        }
        // Clause 5, down: an id past the stack leaves its rows at the
        // sentinel.
        {
            let st = nl_stack(20, 3, 64, 0x51ce_0001);
            let w = st.upload(s)?;
            let n_slots = SLOTS;
            let x = seeded(n_slots * st.k(), 6701, -2.0, 4.0);
            let (act, cols) = blocks32_cols(&c.gpu, &x, st.k(), n_slots)?;
            let mut sel = sel_of(1, st.experts);
            sel[5] = 99;
            let (y, fault) = down(c, &st, &w, &act, &sel)?;
            let bad_sent = y[5 * st.rpe..6 * st.rpe]
                .iter()
                .all(|&v| v.to_bits() == SENT.to_bits());
            let sentinel = host_slots_sent(&sel, &y, st.rpe);
            let named = match &fault {
                Some(f) => GpuError::fault("gate_iq_sel", *f)
                    .to_string()
                    .contains("expert_id"),
                None => false,
            };
            let w_host = stream_words_of(&st.bytes, st.experts * st.rpe)?;
            let host = down_host(&st, &w_host, &cols, &sel);
            let others_same = host
                .iter()
                .zip(&y)
                .enumerate()
                .all(|(i, (a, b))| i / st.rpe == 5 || a.to_bits() == b.to_bits());
            let pass = bad_sent && sentinel && named && others_same;
            ok &= pass;
            println!(
                "ids_nl[k={}] past_stack_sentinel={} fault_names_expert_id={} \
                 host_slots_sentinel={} others_are_host_rule={} fault=\"{}\" {}",
                st.k(),
                verdict(bad_sent),
                verdict(named),
                verdict(sentinel),
                verdict(others_same),
                shown(&fault),
                verdict(pass),
            );
        }
        // Clause 6: a NaN `d` block and an inf-`d` row (two super-blocks,
        // one row, identical bytes, opposite `d`) in both entries.
        for fmt in [IqFormat::Iq3Xxs, IqFormat::Iq4Xs] {
            let st = gu_stack(fmt, KS[0], 4, 80)?;
            let (mut gb, mut ub) = (st.g.clone(), st.u.clone());
            let (rb, bb) = (st.row_bytes(), fmt.block_bytes());
            // Expert 1, row 3, super-block 2 of the gate stack: a NaN d.
            gb[(80 + 3) * rb + 2 * bb..][..2].copy_from_slice(&0x7e00u16.to_le_bytes());
            // Expert 2, row 5, super-blocks 1 and 4 of the up stack: the
            // same bytes with +inf and −inf d.
            let src = ub[(2 * 80 + 5) * rb + bb..][..bb].to_vec();
            ub[(2 * 80 + 5) * rb + 4 * bb..][..bb].copy_from_slice(&src);
            ub[(2 * 80 + 5) * rb + bb..][..2].copy_from_slice(&0x7c00u16.to_le_bytes());
            ub[(2 * 80 + 5) * rb + 4 * bb..][..2].copy_from_slice(&0xfc00u16.to_le_bytes());
            let bad = GuStack {
                fmt,
                g: gb,
                u: ub,
                experts: st.experts,
                rpe: st.rpe,
                k: st.k,
            };
            let (wg, wu) = (bad.upload(s, false)?, bad.upload(s, true)?);
            let (wg0, wu0) = (st.upload(s, false)?, st.upload(s, true)?);
            let m = 2usize;
            let xd = DeviceBuffer::from_host(s, &seeded(m * st.k, 6801, -2.0, 4.0))?;
            let mut act = Q8Act::with_k(s, m, st.k)?;
            c.gpu.enqueue_quantize_q8_1_cols(&xd, &mut act, m, 0)?;
            s.synchronize()?;
            let sel = sel_of(m, st.experts);
            let (h, fault) = gate_up(c, &bad, &wg, &wu, &act, &sel, CLAMP0)?;
            let (clean, _) = gate_up(c, &st, &wg0, &wu0, &act, &sel, CLAMP0)?;
            let poisoned =
                |id: u32, r: usize| id != u32::MAX && ((id == 1 && r == 3) || (id == 2 && r == 5));
            let nan_ok = sel.iter().enumerate().all(|(si, &id)| {
                (0..st.rpe).all(|r| h[si * st.rpe + r].is_nan() == poisoned(id, r))
            });
            let others_same = clean.iter().zip(&h).enumerate().all(|(i, (a, b))| {
                let (si, r) = (i / st.rpe, i % st.rpe);
                poisoned(sel[si], r) || a.to_bits() == b.to_bits()
            });
            let pass = nan_ok && others_same && fault.is_none();
            ok &= pass;
            println!(
                "nan_gu[{}] exactly_poisoned_rows_nan={} others_are_clean={} fault=\"{}\" {}",
                fmt.ggml().name().unwrap_or("?"),
                verdict(nan_ok),
                verdict(others_same),
                shown(&fault),
                verdict(pass),
            );
        }
        {
            let st = nl_stack(20, 3, 64, 0x51ce_0002);
            let mut bytes = st.bytes.clone();
            let rb = 20 * IQ4_NL_BLOCK_BYTES;
            // Expert 1, row 7, block 3: a NaN d.
            bytes[(64 + 7) * rb + 3 * IQ4_NL_BLOCK_BYTES..][..2]
                .copy_from_slice(&0x7e00u16.to_le_bytes());
            // Expert 2, row 9, blocks 5 and 11: identical bytes, the d signs
            // picked against the one slot's column that reads expert 2 so
            // the row's two inf terms oppose whatever the data says.
            let x = seeded(SLOTS * st.k(), 6901, -2.0, 4.0);
            let (act, cols) = blocks32_cols(&c.gpu, &x, st.k(), SLOTS)?;
            let sel = [0u32, 1, 2, 0, 1, 0, 1, u32::MAX, 0, 1];
            let col = &cols[2];
            let a_of = |b: usize, blk: &[u8]| -> i64 {
                (0..16)
                    .map(|j| {
                        let byte = blk[2 + j];
                        i64::from(KVALUES_IQ4NL[usize::from(byte & 15)])
                            * i64::from(col.0[32 * b + j])
                            + i64::from(KVALUES_IQ4NL[usize::from(byte >> 4)])
                                * i64::from(col.0[32 * b + 16 + j])
                    })
                    .sum()
            };
            let src =
                bytes[(2 * 64 + 9) * rb + 5 * IQ4_NL_BLOCK_BYTES..][..IQ4_NL_BLOCK_BYTES].to_vec();
            let a5 = a_of(
                5,
                &bytes[(2 * 64 + 9) * rb + 5 * IQ4_NL_BLOCK_BYTES..][..IQ4_NL_BLOCK_BYTES],
            );
            bytes[(2 * 64 + 9) * rb + 11 * IQ4_NL_BLOCK_BYTES..][..IQ4_NL_BLOCK_BYTES]
                .copy_from_slice(&src);
            // A zero dot makes its term NaN at either sign; else the signs
            // oppose: the first block's term +inf, the second's −inf.
            let a11 = a_of(11, &src);
            let d5 = if a5 < 0 { 0xfc00u16 } else { 0x7c00 };
            let d11 = if a11 < 0 { 0x7c00u16 } else { 0xfc00 };
            bytes[(2 * 64 + 9) * rb + 5 * IQ4_NL_BLOCK_BYTES..][..2]
                .copy_from_slice(&d5.to_le_bytes());
            bytes[(2 * 64 + 9) * rb + 11 * IQ4_NL_BLOCK_BYTES..][..2]
                .copy_from_slice(&d11.to_le_bytes());
            let bad = NlStack {
                bytes,
                experts: st.experts,
                rpe: st.rpe,
                k_blocks: st.k_blocks,
            };
            let (w, w0) = (bad.upload(s)?, st.upload(s)?);
            let (y, fault) = down(c, &bad, &w, &act, &sel)?;
            let (clean, _) = down(c, &st, &w0, &act, &sel)?;
            let poisoned =
                |id: u32, r: usize| id != u32::MAX && ((id == 1 && r == 7) || (id == 2 && r == 9));
            let nan_ok = sel.iter().enumerate().all(|(si, &id)| {
                (0..st.rpe).all(|r| y[si * st.rpe + r].is_nan() == poisoned(id, r))
            });
            let others_same = clean.iter().zip(&y).enumerate().all(|(i, (a, b))| {
                let (si, r) = (i / st.rpe, i % st.rpe);
                poisoned(sel[si], r) || a.to_bits() == b.to_bits()
            });
            let pass = nan_ok && others_same && fault.is_none();
            ok &= pass;
            println!(
                "nan_nl[k={}] exactly_poisoned_rows_nan={} others_are_clean={} fault=\"{}\" {}",
                st.k(),
                verdict(nan_ok),
                verdict(others_same),
                shown(&fault),
                verdict(pass),
            );
        }
        Ok(ok)
    }

    /// Clause 7: the three entries in one graph (module doc).
    fn check_capture(c: &Ctx) -> Result<bool, GateError> {
        let s = c.gpu.stream();
        let st3 = gu_stack(IqFormat::Iq3Xxs, KS[0], 4, 80)?;
        let st4 = gu_stack(IqFormat::Iq4Xs, KS[0], 4, 80)?;
        let (wg3, wu3) = (st3.upload(s, false)?, st3.upload(s, true)?);
        let (wg4, wu4) = (st4.upload(s, false)?, st4.upload(s, true)?);
        let m = 2usize;
        let xd = DeviceBuffer::from_host(s, &seeded(m * st3.k, 7501, -2.0, 4.0))?;
        let mut act = Q8Act::with_k(s, m, st3.k)?;
        c.gpu.enqueue_quantize_q8_1_cols(&xd, &mut act, m, 0)?;
        s.synchronize()?;
        let sel = sel_of(m, 4);
        let sel_dev = DeviceBuffer::from_host(s, &sel)?;
        let nl = nl_stack(20, 3, 64, 0x51ce_0003);
        let wn = nl.upload(s)?;
        let (act_nl, _) = blocks32_cols(
            &c.gpu,
            &seeded(SLOTS * nl.k(), 7601, -2.0, 4.0),
            nl.k(),
            SLOTS,
        )?;
        let sel_nl = sel_of(1, 3);
        let sel_nl_dev = DeviceBuffer::from_host(s, &sel_nl)?;
        let sink = c.gpu.unlabelled_sink();
        let eager = |st: &CudaStream,
                     h3: &mut DeviceBuffer<f32>,
                     h4: &mut DeviceBuffer<f32>,
                     y: &mut DeviceBuffer<f32>|
         -> Result<(), GpuError> {
            c.iqsel.enqueue_gate_up(
                st,
                &IqGateUp {
                    ty: GgmlType::IQ3_XXS,
                    wg: &wg3,
                    wu: &wu3,
                    act: &act,
                    sel: &sel_dev,
                    n_slots: sel.len(),
                    rows_per_expert: st3.rpe,
                    slots_per_col: SLOTS,
                    rule: CLAMP0,
                },
                sink,
                h3,
            )?;
            c.iqsel.enqueue_gate_up(
                st,
                &IqGateUp {
                    ty: GgmlType::IQ4_XS,
                    wg: &wg4,
                    wu: &wu4,
                    act: &act,
                    sel: &sel_dev,
                    n_slots: sel.len(),
                    rows_per_expert: st4.rpe,
                    slots_per_col: SLOTS,
                    rule: CLAMP0,
                },
                sink,
                h4,
            )?;
            c.iqsel.enqueue_down(
                st,
                &IqDown {
                    w: &wn,
                    act: &act_nl,
                    sel: &sel_nl_dev,
                    n_slots: sel_nl.len(),
                    rows_per_expert: nl.rpe,
                },
                sink,
                y,
            )
        };
        let bufs = |s: &CudaStream| {
            Ok::<_, GateError>((
                DeviceBuffer::from_host(s, &vec![SENT; sel.len() * st3.rpe])?,
                DeviceBuffer::from_host(s, &vec![SENT; sel.len() * st4.rpe])?,
                DeviceBuffer::from_host(s, &vec![SENT; sel_nl.len() * nl.rpe])?,
            ))
        };
        // Two eager launches, then the capture's replay, all bit-compared.
        let (mut e3, mut e4, mut ey) = bufs(s)?;
        eager(s, &mut e3, &mut e4, &mut ey)?;
        s.synchronize()?;
        let (e3v, e4v, eyv) = (e3.to_host_vec(s)?, e4.to_host_vec(s)?, ey.to_host_vec(s)?);
        let (mut f3, mut f4, mut fy) = bufs(s)?;
        eager(s, &mut f3, &mut f4, &mut fy)?;
        s.synchronize()?;
        let second = (
            bits_equal(&e3v, &f3.to_host_vec(s)?),
            bits_equal(&e4v, &f4.to_host_vec(s)?),
            bits_equal(&eyv, &fy.to_host_vec(s)?),
        );
        let (mut h3, mut h4, mut y) = bufs(s)?;
        let graph = c.gpu.capture(|cap| eager(cap, &mut h3, &mut h4, &mut y))?;
        let nodes = graph.node_count();
        graph.launch(s)?;
        s.synchronize()?;
        let replay = (
            bits_equal(&h3.to_host_vec(s)?, &e3v),
            bits_equal(&h4.to_host_vec(s)?, &e4v),
            bits_equal(&y.to_host_vec(s)?, &eyv),
        );
        drop(graph);
        let no_fault = c.gpu.take_fault()?.is_none();
        let pass = nodes == 3
            && replay.0
            && replay.1
            && replay.2
            && second.0
            && second.1
            && second.2
            && no_fault;
        println!(
            "capture graph_nodes={nodes} (want 3) replay_bits={},{},{} second_eager_bits={},{},{} \
             fault=none {}",
            verdict(replay.0),
            verdict(replay.1),
            verdict(replay.2),
            verdict(second.0),
            verdict(second.1),
            verdict(second.2),
            verdict(pass),
        );
        Ok(pass)
    }

    /// Clause 8: the launchers' refusals, each an `Err(Shape)` of its own
    /// (module doc).
    fn check_refusals(c: &Ctx) -> Result<bool, GateError> {
        let s = c.gpu.stream();
        let st = gu_stack(IqFormat::Iq3Xxs, KS[0], 4, 80)?;
        let (wg, wu) = (st.upload(s, false)?, st.upload(s, true)?);
        let mut act = Q8Act::with_k(s, 1, st.k)?;
        let xd = DeviceBuffer::from_host(s, &seeded(st.k, 1, -2.0, 4.0))?;
        c.gpu.enqueue_quantize_q8_1_cols(&xd, &mut act, 1, 0)?;
        let act9 = Q8Act::with_slots(s, 9, st.k)?;
        let act_k = Q8Act::with_k(s, 1, KS[1])?;
        let sel = DeviceBuffer::from_host(s, &[0u32; SLOTS])?;
        let sel_short = DeviceBuffer::from_host(s, &[0u32; 3])?;
        let mut h = DeviceBuffer::from_host(s, &vec![SENT; SLOTS * st.rpe])?;
        let sink = c.gpu.unlabelled_sink();
        // A stack one word a row short.
        let rows = st.experts * st.rpe;
        let words = stream_words_of(&st.g, rows)?;
        let short = DeviceTensor::upload(
            s,
            &words[..words.len() - rows],
            rows,
            words.len() / rows - 1,
        )?;
        let launch = |ty: GgmlType,
                      wg: &DeviceTensor<u32>,
                      act: &Q8Act,
                      sel: &DeviceBuffer<u32>,
                      h: &mut DeviceBuffer<f32>|
         -> Result<(), GpuError> {
            c.iqsel.enqueue_gate_up(
                s,
                &IqGateUp {
                    ty,
                    wg,
                    wu: &wu,
                    act,
                    sel,
                    n_slots: SLOTS,
                    rows_per_expert: st.rpe,
                    slots_per_col: SLOTS,
                    rule: CLAMP0,
                },
                sink,
                h,
            )
        };
        let mut h_short = DeviceBuffer::from_host(s, &[SENT; 8])?;
        let cases: [(&str, GpuError); 6] = [
            (
                "wrong_type_q4_k",
                launch(GgmlType::Q4_K, &wg, &act, &sel, &mut h)
                    .err()
                    .ok_or("accepted a Q4_K stack")?,
            ),
            (
                "short_words",
                launch(GgmlType::IQ3_XXS, &short, &act, &sel, &mut h)
                    .err()
                    .ok_or("accepted a short stack")?,
            ),
            (
                "k_off_the_block",
                launch(GgmlType::IQ3_XXS, &wg, &act_k, &sel, &mut h)
                    .err()
                    .ok_or("accepted k = 2304 on a k = 2560 stack")?,
            ),
            (
                "m_nine_columns",
                launch(GgmlType::IQ3_XXS, &wg, &act9, &sel, &mut h)
                    .err()
                    .ok_or("accepted nine columns")?,
            ),
            (
                "short_sel",
                launch(GgmlType::IQ3_XXS, &wg, &act, &sel_short, &mut h)
                    .err()
                    .ok_or("accepted three ids for ten slots")?,
            ),
            (
                "short_h",
                launch(GgmlType::IQ3_XXS, &wg, &act, &sel, &mut h_short)
                    .err()
                    .ok_or("accepted an eight-value h")?,
            ),
        ];
        let mut ok = true;
        for (case, e) in cases {
            let pass = matches!(
                &e,
                GpuError::Shape {
                    what: "IqSelKernels::enqueue_gate_up",
                    ..
                }
            );
            println!(
                "refuse_gu[{case}] want=Err(Shape enqueue_gate_up) got=\"{e}\" {}",
                verdict(pass)
            );
            ok &= pass;
        }
        // The down's: a Q8_0-width stack (34 B a block), a non-dividing
        // expert size, a column count other than the slots, a short y.
        let nl = nl_stack(20, 3, 64, 0x51ce_0004);
        let w = nl.upload(s)?;
        let q80_rows = 3 * 64;
        let q80_words = stream_words_of(&vec![0u8; 34 * 20 * q80_rows], q80_rows)?;
        let wq80 = DeviceTensor::upload(s, &q80_words, q80_rows, q80_words.len() / q80_rows)?;
        let (act_nl, _) = blocks32_cols(&c.gpu, &seeded(2 * nl.k(), 2, -2.0, 4.0), nl.k(), 2)?;
        let sel_nl = DeviceBuffer::from_host(s, &[0u32; SLOTS])?;
        let mut y = DeviceBuffer::from_host(s, &vec![SENT; SLOTS * nl.rpe])?;
        let launch_nl = |w: &DeviceTensor<u32>,
                         act: &Q8Blocks32,
                         rpe: usize,
                         y: &mut DeviceBuffer<f32>|
         -> Result<(), GpuError> {
            c.iqsel.enqueue_down(
                s,
                &IqDown {
                    w,
                    act,
                    sel: &sel_nl,
                    n_slots: SLOTS,
                    rows_per_expert: rpe,
                },
                sink,
                y,
            )
        };
        let mut y_short = DeviceBuffer::from_host(s, &[SENT; 9])?;
        let dcases: [(&str, GpuError); 4] = [
            (
                "q8_0_row_width",
                launch_nl(&wq80, &act_nl, nl.rpe, &mut y)
                    .err()
                    .ok_or("accepted a 34-byte-block stack")?,
            ),
            (
                "rows_per_expert_50_of_192",
                launch_nl(&w, &act_nl, 50, &mut y)
                    .err()
                    .ok_or("accepted 50 of 192 rows")?,
            ),
            (
                "two_columns_for_ten_slots",
                launch_nl(&w, &act_nl, nl.rpe, &mut y)
                    .err()
                    .ok_or("accepted two columns for ten slots")?,
            ),
            (
                "short_y",
                launch_nl(&w, &act_nl, nl.rpe, &mut y_short)
                    .err()
                    .ok_or("accepted a nine-value y")?,
            ),
        ];
        for (case, e) in dcases {
            let pass = matches!(
                &e,
                GpuError::Shape {
                    what: "IqSelKernels::enqueue_down",
                    ..
                }
            );
            println!(
                "refuse_nl[{case}] want=Err(Shape enqueue_down) got=\"{e}\" {}",
                verdict(pass)
            );
            ok &= pass;
        }
        s.synchronize()?;
        Ok(ok && c.gpu.take_fault()?.is_none())
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let c = Ctx {
            iqsel: IqSelKernels::load(gpu.context(), gpu.fault_word())?,
            iq: IqKernels::load(gpu.context())?,
            q80: Q80SelKernels::load(gpu.context(), gpu.fault_word())?,
            gpu,
        };
        let s = c.gpu.stream();
        let mut ok = true;
        let mut rels: Vec<(IqFormat, usize, f64)> = Vec::new();
        for fmt in [IqFormat::Iq3Xxs, IqFormat::Iq4Xs] {
            for &k in KS.iter() {
                let st = gu_stack(fmt, k, 4, 80)?;
                let (wg, wu) = (st.upload(s, false)?, st.upload(s, true)?);
                let (pass, rel) = check_gu(&c, &st, &wg, &wu, true, true)?;
                ok &= pass;
                rels.push((fmt, k, rel));
            }
            // The 2 experts × 640 rows case, cycled rows: clause 1.
            let st = gu_stack(fmt, KS[0], 2, 640)?;
            let (wg, wu) = (st.upload(s, false)?, st.upload(s, true)?);
            let (pass, _) = check_gu(&c, &st, &wg, &wu, false, false)?;
            ok &= pass;
            // Clause 4 on the first geometry.
            let st4 = gu_stack(fmt, KS[0], 4, 80)?;
            let (wg4, wu4) = (st4.upload(s, false)?, st4.upload(s, true)?);
            ok &= check_silu(&c, &st4, &wg4, &wu4)?;
        }
        for (fmt, k, rel) in &rels {
            let pin = REL_PIN_GU
                .iter()
                .find(|(f, kp, _)| f == fmt && kp == k)
                .map(|&(_, _, p)| p)
                .ok_or("every gate·up geometry has a pin")?;
            let under = *rel <= pin;
            ok &= under;
            println!(
                "{:8} k={k} max|rel|, the larger of the m-sweep: {rel:.4e} (pin {pin:.1e}) {}",
                fmt.ggml().name().unwrap_or("?"),
                verdict(under),
            );
        }
        let mut nl_rels = Vec::new();
        for (kb, seed) in [(20usize, 0x51ce_0011u64), (19, 0x51ce_0012)] {
            let st = nl_stack(kb, 3, 2560, seed);
            let w = st.upload(s)?;
            let (pass, rel) = check_nl(&c, &st, &w)?;
            ok &= pass;
            nl_rels.push((st.k(), rel));
        }
        for (k, rel) in &nl_rels {
            let pin = REL_PIN_NL
                .iter()
                .find(|(kp, _)| kp == k)
                .map(|&(_, p)| p)
                .ok_or("every down geometry has a pin")?;
            let under = *rel <= pin;
            ok &= under;
            println!(
                "IQ4_NL   k={k} max|rel|, the larger of the m-sweep: {rel:.4e} (pin {pin:.1e}) {}",
                verdict(under),
            );
        }
        ok &= check_ids_and_nan(&c)?;
        ok &= check_capture(&c)?;
        ok &= check_refusals(&c)?;
        let args: Vec<String> = std::env::args().collect();
        if let Some(p) = args
            .iter()
            .position(|a| a == "--model")
            .and_then(|i| args.get(i + 1))
        {
            ok &= check_real_file(&c, p)?;
        }
        println!("gate_iq_sel: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
