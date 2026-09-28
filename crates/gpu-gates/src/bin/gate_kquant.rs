//! GPU gate for the model-free K-quant expert family (`bloomery_gpu::kquant`):
//! the Q5_K and Q8_0 down `_sel`s, their gate·ups with the activation rule as a
//! launch argument, and Walk A under them. Synthetic stacks only, no model file:
//! every K-quant super-block is random words with a positive normal `d` and
//! `dmin`, every Q8_0 block random codes with a positive normal `d`.
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
//!    swiglu (the same core), `swiglu_clamp` at two limits against the CPU
//!    tier's ik-verified rule `qdot::swiglu_clamp`. The clamped rows are counted.
//! 6. q4k_sibling: the gate binary's Walk A instance over `Q4k` is
//!    `q4k_gemv_sel` bit for bit, `HOST` slot included, at K = 256, 768, 1280,
//!    2048, 2304, 4096.
//! 7. q5_1: the Q5_1 down `_sel` (`bloomery_gpu::q5_1_sel`, the file's
//!    `block_q5_1` bytes resident) over seven experts, ten slots (Qwen3.8's
//!    `n_used`, columns from `Q8Blocks32::with_slots`) with ids first (0), last
//!    (6), middle (3, twice), `HOST` and the rest, at the stack shapes (rows × K)
//!    2560 × 640 (Qwen3.8's routed down), 100 × 1280 (a partial last grid block
//!    and a second lane pass), 64 × 2080 (three q windows, lane 0 alone in its
//!    third pass) and 64 × 32 (one block): within `KERNEL_BAND` of the f64 dot of
//!    `dequant_row(Q5_1)` rows with the activation dequantized from the
//!    quantizer's own readback; `q5_1_gemv` over the `pack_q5_1` copy of each
//!    slot's expert against the slot's column, bit for bit; the `HOST` slot
//!    untouched; a rerun bit-identical. On 2560 × 640: a captured launch replays
//!    as one node and follows overwritten ids; an id past the stack raises
//!    `ExpertId` and leaves its slot as it was; a NaN in a column raises
//!    `Q5Quant` from the quantizer and the down adds no fault; the launcher
//!    refuses an activation of another K, a column count other than the slots
//!    and a non-dividing expert size.
//! 8. q8_0: the Q8_0 entries (`q8_0_gemv_sel`, `kq_gate_up_act_q8_0`) over rows
//!    in the file's 34-byte block layout, synthetic codes over all 256 byte
//!    values. Clause 1 against `dequant_row(Q8_0)` at K = 256, 768, 1280, 2304
//!    on four experts and at GLM-5.3-Flash's non-routed shapes as one-expert
//!    stacks with every id 0 (shared gate·up 2048 × 4096 and down 4096 × 2048,
//!    dense gate·up 12288 × 4096 and down 4096 × 12288), the probe's decode
//!    (codes, `d`, 0) as the host's reading of the block bytes; clauses 5 and 4
//!    on the Q8_0 entries; and each launcher refusing a stack of the other
//!    format's row width.
//! 9. q4k_gate_up: the Q4_K gate·up (`kq_gate_up_act_q4k`, GLM-5.3-Flash's
//!    routed gate and up) under clauses 5 and 4, its down sums from clause 6's
//!    Walk A instance over `Q4k` (so `q4k_gemv_sel`'s bit for bit): the rows
//!    are `silu_mul` and `swiglu_clamp` at two limits on the sums of the same
//!    rows, an id past the stack raises `ExpertId` and NaNs its rows, `HOST`
//!    leaves them, a NaN column NaNs every row.
//! 10. q38_card: Qwen3.8's card leg over two tokens of ten slots, seven card
//!     experts, three slots the host serves (their gate·up rows poisoned NaN):
//!     the Q4_K gate·up `_sel` (640 × 2560, a column a token), the card
//!     columns' 32-value q8_1 (`q5_quantize_q8_sel`), the Q5_1 down `_sel`
//!     (2560 × 640), the card sum (`q38_card_acc`) and the combine with the
//!     host's sum and the gated shared expert (`q38_card_shared_add`). The
//!     quantizer's card columns are the plain quantizer's bytes of the same
//!     columns, its host columns keep a pattern written before it, and it
//!     raises nothing on their NaNs; the sum is `runtime::combine::card_sum`
//!     of the ten slots at the router's pitch of eleven, each token its own
//!     weights, and the combine `(hsum + sum) + sh · w[10]` (host, card,
//!     shared: `runtime::combine::combine`), bit for bit, no fault. The five
//!     launches captured replay as five nodes and follow places overwritten
//!     between replays, the host slots moved. A NaN in a card column raises
//!     `Q5Quant`, a NaN card sum `F32Product` from the combine; the launchers
//!     refuse a card of no expert, weights at a pitch of ten, columns past the
//!     scratch and a slot past the weights.

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
        Act, GateUpAct, KquantKernels, Q4k, Q5k, Q8_0, SbDecode, SelDown, act, walk_a_planes,
    };
    use bloomery_gpu::q4k_sel::QuantSel;
    use bloomery_gpu::q5::{Q8Blocks32, pack_q5_1};
    use bloomery_gpu::q5_1_sel::{BLOCK_WORDS, Q51SelDown, Q51SelKernels};
    use bloomery_gpu::q38::{CardAccArgs, CardSharedAddArgs, Q38Kernels, SLOTS, W_PITCH};
    use bloomery_gpu::{
        DeviceTensor, Fault, FaultSink, FaultSite, Gpu, GpuError, LAYER_NONE, Q8Act, col_sums,
    };
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, activations, bits_equal, max_rel_err, max_ulps, q8_1_dequant,
        ref_gemv, verdict,
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

        /// The row probe over Q5_K ([`probe_walk`]).
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
            // SAFETY: the launch contract is the walk's (44 = Q5k::WORDS);
            // iters = ceil(n_sb/4) from the host; the one block is the one warp.
            unsafe { probe_walk::<Q5k>(w, q, s8, d8, row_abs, col, n_sb, iters, &mut out) };
        }

        /// The row probe over Q8_0 ([`probe_walk`]).
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
                w.len() >= (row_abs + 1) * 68 * n_sb,
                q.len() >= (col + 1) * 256 * iters,
                s8.len() >= (col + 1) * 8 * n_sb,
                d8.len() >= (col + 1) * 2 * n_sb,
                out.len() >= 33 + 88 * n_sb
            )
        )]
        pub fn kq_probe_q8_0(
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
            // SAFETY: the launch contract is the walk's (68 = Q8_0::WORDS);
            // iters = ceil(n_sb/4) from the host; the one block is the one warp.
            unsafe { probe_walk::<Q8_0>(w, q, s8, d8, row_abs, col, n_sb, iters, &mut out) };
        }
    }

    /// The row probe's body: one warp walks row `row_abs` of format `D`
    /// against column `col` as the `_sel` entries do and writes, as bits, its
    /// 32 lane partials at `out[0..32]`, their warp sum at `out[32]`, and per
    /// sub-block `(sb, s)` at `out[33 + 11·(8·sb + s)..]` the decoder's eight
    /// code words, `cda`, `cdb` and the walk's term for it.
    ///
    /// # Safety
    ///
    /// `w.len() >= (row_abs + 1) · D::WORDS · n_sb`, column `col` inside the
    /// three planes, `iters = ceil(n_sb / 4)`, `out.len() >= 33 + 88 · n_sb`,
    /// and the 32 lanes of one warp call it.
    #[allow(
        clippy::too_many_arguments,
        reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
    )]
    #[inline(always)]
    unsafe fn probe_walk<D: SbDecode>(
        w: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        row_abs: u32,
        col: u32,
        n_sb: u32,
        iters: u32,
        out: &mut DisjointSlice<u32>,
    ) {
        let lane = warp::lane_id() as usize;
        let (n_sb, row, col) = (n_sb as usize, row_abs as usize, col as usize);
        // SAFETY: this fn's contract bounds row `row_abs` of `w` and column
        // `col` of the planes, iters = ceil(n_sb/4), one warp.
        let f = unsafe { row_dot_1col::<D>(w, q, s8, d8, n_sb, iters, row, col, lane) };
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
                        D::decode(w, row * D::WORDS * n_sb + D::WORDS * sbp, s),
                        iter_term::<D>(
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

    // The gate module spells the three formats' super-block words.
    const _: () = assert!(Q4k::WORDS == 36 && Q5k::WORDS == 44 && Q8_0::WORDS == 68);

    /// Experts of every stack here.
    const E: usize = 4;
    /// What an output buffer holds before a launch: a slot left alone reads
    /// back as these bits.
    const SENT: f32 = 1.0e30;

    /// Everything the clauses share.
    struct Ctx {
        gpu: Gpu,
        kq: KquantKernels,
        q51: Q51SelKernels,
        gm: gate_kernels::LoadedModule,
    }

    /// One synthetic stack of experts (`E` unless built one-expert) of `rpe`
    /// rows of `k` values.
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
            Stack::with_experts(c, tag, ty, E, rpe, k, seed)
        }

        /// A stack of `experts` experts: one is a non-routed FFN's matrix.
        fn with_experts(
            c: &Ctx,
            tag: &'static str,
            ty: GgmlType,
            experts: usize,
            rpe: usize,
            k: usize,
            seed: u64,
        ) -> Result<Stack, GateError> {
            let n_sb = k / 256;
            let words = synthetic(ty, experts * rpe, n_sb, seed)?;
            let wpr = words.len() / (experts * rpe);
            let w = DeviceTensor::upload(c.gpu.stream(), &words, experts * rpe, wpr)?;
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

    /// `rows` synthetic super-block rows of `ty` (Q4_K, Q5_K or Q8_0), `n_sb`
    /// super-blocks each, as words from a fixed-seed xorshift64: every word
    /// random except the first of each K-quant super-block, whose halves are
    /// `d` and `dmin`, and each Q8_0 block's first two bytes, its `d` — each a
    /// positive normal f16 (exponent field 1..=9), so no NaN or infinity
    /// enters. Every other pattern is a valid block, Q8_0's code −128
    /// included.
    fn synthetic(ty: GgmlType, rows: usize, n_sb: usize, seed: u64) -> Result<Vec<u32>, GateError> {
        let words = match ty {
            GgmlType::Q4_K => Q4k::WORDS,
            GgmlType::Q5_K => Q5k::WORDS,
            GgmlType::Q8_0 => Q8_0::WORDS,
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
        if ty == GgmlType::Q8_0 {
            // Eight 34-byte blocks a super-block: `d`, then 32 codes.
            let mut bytes = Vec::with_capacity(rows * n_sb * 4 * words);
            for _ in 0..rows * n_sb * 8 {
                bytes.extend_from_slice(&(half(next()) as u16).to_le_bytes());
                for _ in 0..8 {
                    bytes.extend_from_slice(&((next() >> 32) as u32).to_le_bytes());
                }
            }
            return Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect());
        }
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
    /// `SENT`-filled output, synchronized; the output and the fault word. The
    /// stack's type picks the entry: Q8_0's, Q4_K's (this binary's Walk A
    /// instance, clause 6's), else Q5_K's.
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
        match st.ty {
            GgmlType::Q8_0 => {
                c.kq.enqueue_gemv_q8_0_sel(stream, &a, c.gpu.unlabelled_sink(), &mut y)?;
            }
            GgmlType::Q4_K => {
                let n_sb = st.n_sb();
                let (q, s8, d8) = walk_a_planes(act);
                let prep = c.gm.prepare_kq_gemv_sel_q4k(LaunchConfig1D::new(
                    u32::try_from((sel.len() * st.rpe).div_ceil(ROWS_PER_BLOCK))?,
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
                    u32::try_from(st.w.rows() / st.rpe)?,
                    u32::try_from(st.rpe)?,
                    u32::try_from(sel.len())?,
                    u32::try_from(n_sb)?,
                    u32::try_from(n_sb.div_ceil(4))?,
                    c.gpu.unlabelled_sink(),
                    &mut y,
                )?;
            }
            _ => {
                c.kq.enqueue_gemv_q5k_sel(stream, &a, c.gpu.unlabelled_sink(), &mut y)?
            }
        }
        stream.synchronize()?;
        Ok((y.to_host_vec(stream)?, c.gpu.take_fault()?))
    }

    /// The gate·up of stacks `g`/`u` for `sel` at `spc` slots a column of
    /// `act` under `rule`, into a `SENT`-filled output; the output and the
    /// fault word. The gate stack's type picks the entry, as [`down`].
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
        match g.ty {
            GgmlType::Q8_0 => {
                c.kq.enqueue_gate_up_q8_0(stream, &a, c.gpu.unlabelled_sink(), &mut h)?
            }
            GgmlType::Q4_K => {
                c.kq.enqueue_gate_up_q4k(stream, &a, c.gpu.unlabelled_sink(), &mut h)?
            }
            _ => {
                c.kq.enqueue_gate_up_q5k(stream, &a, c.gpu.unlabelled_sink(), &mut h)?
            }
        }
        stream.synchronize()?;
        Ok((h.to_host_vec(stream)?, c.gpu.take_fault()?))
    }

    /// The probe of row `row_abs` of `st` (Q5_K or Q8_0) against column `col`
    /// of `act`.
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
        let cfg = LaunchConfig1D::new(1, 32, 0);
        let (row_abs, col) = (u32::try_from(row_abs)?, u32::try_from(col)?);
        let (n_sb, iters) = (u32::try_from(n_sb)?, u32::try_from(n_sb.div_ceil(4))?);
        if st.ty == GgmlType::Q8_0 {
            let prep = c.gm.prepare_kq_probe_q8_0(cfg)?;
            c.gm.kq_probe_q8_0(
                stream,
                &prep,
                st.w.buf(),
                q,
                s8,
                d8,
                row_abs,
                col,
                n_sb,
                iters,
                &mut out,
            )?;
        } else {
            let prep = c.gm.prepare_kq_probe_q5k(cfg)?;
            c.gm.kq_probe_q5k(
                stream,
                &prep,
                st.w.buf(),
                q,
                s8,
                d8,
                row_abs,
                col,
                n_sb,
                iters,
                &mut out,
            )?;
        }
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

    /// The host's reading of sub-block `s` of a Q8_0 super-block (eight
    /// 34-byte blocks): block `s`'s 32 codes as eight words (values `4·i ..`
    /// in word `i`, byte order), `d` and 0.
    fn host_decode_q8_0(sb: &[u8], s: usize) -> ([u32; 8], f32, f32) {
        let blk = &sb[34 * s..34 * (s + 1)];
        let mut words = [0u32; 8];
        for (i, w) in words.iter_mut().enumerate() {
            *w = u32::from_le_bytes([
                blk[2 + 4 * i],
                blk[3 + 4 * i],
                blk[4 + 4 * i],
                blk[5 + 4 * i],
            ]);
        }
        (
            words,
            half_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
            0.0,
        )
    }

    /// The probe's decode of row `row_abs` of `st` against the host's, word for
    /// word; the first sub-block that differs, or `None`.
    fn probe_decode_diff(st: &Stack, row_abs: usize, out: &[u32]) -> Option<String> {
        let n_sb = st.n_sb();
        let wpr = st.w.cols();
        let sbb = 4 * wpr / n_sb; // bytes a super-block
        let row = &words_bytes(&st.words[row_abs * wpr..(row_abs + 1) * wpr]);
        for sb in 0..n_sb {
            for s in 0..8 {
                let bytes = &row[sbb * sb..sbb * (sb + 1)];
                let (codes, cda, cdb) = if st.ty == GgmlType::Q8_0 {
                    host_decode_q8_0(bytes, s)
                } else {
                    host_decode(bytes, s)
                };
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
        let q51 = Q51SelKernels::load(gpu.context(), gpu.fault_word())?;
        let c = Ctx { gpu, kq, q51, gm };
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
        ok &= check_q5_1(&c)?;
        ok &= check_q8_0(&c)?;
        ok &= check_q4k_gate_up(&c)?;
        ok &= check_q38_card(&c)?;
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
             for bit on 6 K; q5_1_gemv_sel within {KERNEL_BAND:e} of the f64 dequant_row \
             reference and q5_1_gemv of the packed expert bit for bit at 4 shapes, HOST \
             untouched, graph replay, expert_id past the stack, q5_quant on a NaN column; \
             the Q8_0 entries within the band at 8 shapes (GLM's dense and shared \
             FFN among them) with their probe decoding as the host, their gate·up the rule on \
             their down's sums, their faults as Q5_K's, and each launcher refusing the other \
             format's rows; the Q4_K gate·up the rule on Walk A's Q4_K sums, its faults as \
             Q5_K's; Qwen3.8's card leg over two tokens: the card columns' q8_1 the plain \
             quantizer's, the host columns untouched, the card sum and the combine the \
             runtime's rule bit for bit, five nodes replaying moved places, q5_quant and \
             f32_product named, the launchers' refusals"
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
            ok &= band_case(c, "band", &st, &SEL, 7001 + i as u32)?;
        }
        Ok(ok)
    }

    /// One band case (clause 1's, clause 8's): the down `_sel` of `st` for
    /// `sel` on columns from `xseed` against the f64 reference, the probe on a
    /// row of slot 1, one line under `label`.
    fn band_case(
        c: &Ctx,
        label: &str,
        st: &Stack,
        sel: &[u32],
        xseed: u32,
    ) -> Result<bool, GateError> {
        let (rpe, k, n) = (st.rpe, st.k, sel.len());
        let x = activations(k, n, xseed);
        let act = quantize(c, &x, n, k)?;
        let (y, fault) = down(c, st, &act, sel)?;
        let xq = q8_1_dequant(&x, k, n);
        let mut want = Vec::with_capacity(y.len());
        for (s, &e) in sel.iter().enumerate() {
            want.extend(ref_gemv(
                st.ty,
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
        let row_abs = sel[slot] as usize * rpe + r;
        let out = probe(c, st, &act, row_abs, slot)?;
        let decode_diff = probe_decode_diff(st, row_abs, &out);
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
            let w_abs = sel[ws] as usize * rpe + wr;
            let wout = probe(c, st, &act, w_abs, ws)?;
            let denom = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            format!(
                " worst=slot {ws} row {wr} got {:e} want {:e}{}",
                y[worst],
                want[worst],
                first_sub_out(
                    st,
                    w_abs,
                    &xq[ws * k..(ws + 1) * k],
                    &wout,
                    f64::from(KERNEL_BAND * denom)
                )
            )
        };
        println!(
            "{label}[{}] rows={rpe} K={k} n_sb={} max_rel={rel:.3e} band={KERNEL_BAND:e} \
             fault=\"{}\" probe_decode_as_host={} probe_sum_is_launch={sum_same} {}{}{}",
            st.tag,
            st.n_sb(),
            shown(fault),
            decode_diff.is_none(),
            verdict(pass),
            decode_diff.map_or_else(String::new, |d| format!(" first_decode_diff={d}")),
            detail,
        );
        Ok(pass)
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
        fault_cases(c, GgmlType::Q5_K, "fault")
    }

    /// Clause 4's cases on the entries of `ty`, each line under `label`.
    fn fault_cases(c: &Ctx, ty: GgmlType, label: &str) -> Result<bool, GateError> {
        const CLEAN: [u32; 4] = [1, 3, 0, 2];
        let past = E as u32 + 3;
        let (rpe, k) = (256, 2304);
        let st = Stack::new(c, "down", ty, rpe, k, 0xfa17_0001)?;
        let g = Stack::new(c, "gate", ty, rpe, k, 0xfa17_0002)?;
        let u = Stack::new(c, "up", ty, rpe, k, 0xfa17_0003)?;
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
                "{label}[{entry}:{case}] sel={sel:?} fault=\"{}\" want=\"{}\" slot1_{}_others_clean={values_ok} {}{}",
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
            "{label}[down:nan_column] quantizer_fault=\"{}\" want=\"{}\" down_fault=\"{}\" \
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
            "{label}[gate_up:nan_column] quantizer_fault=\"{}\" want=\"{}\" gate_up_faults=\"{}\",\"{}\" \
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
        act_cases(c, GgmlType::Q5_K, "act")
    }

    /// Clause 5's cases on the entries of `ty`, each line under `label`.
    fn act_cases(c: &Ctx, ty: GgmlType, label: &str) -> Result<bool, GateError> {
        const SEL: [u32; 4] = [3, 0, 2, 1];
        let stream = c.gpu.stream();
        let (rpe, k) = (512, 2048);
        let g = Stack::new(c, "gate", ty, rpe, k, 0xac70_0001)?;
        let u = Stack::new(c, "up", ty, rpe, k, 0xac70_0002)?;
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
                    let mut want = vec![0.0f32; n];
                    qdot::swiglu_clamp(&gs, &us, limit, &mut want);
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
                "{label}[{name}] rows={n} rule_on_down_sums_bit_identical={same} clamped_rows={clamped} \
                 fault=\"{}\" {}{}",
                shown(f),
                verdict(pass),
                mismatch("first_mismatch", &h, &want, rpe),
            );
        }
        Ok(ok)
    }

    /// Clause 9 (module doc).
    fn check_q4k_gate_up(c: &Ctx) -> Result<bool, GateError> {
        let act = act_cases(c, GgmlType::Q4_K, "q4k_gate_up_act")?;
        let fault = fault_cases(c, GgmlType::Q4_K, "q4k_gate_up_fault")?;
        Ok(act && fault)
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

    /// Experts of every Q5_1 stack: ids 0, 3 and 6 are the first, a middle
    /// and the last.
    const E51: usize = 7;
    /// Clause 7's slots, Qwen3.8's ten a token: first, last, middle twice, a
    /// host slot, then the others and the last again.
    const SEL51: [u32; 10] = [6, 0, 3, 3, HOST, 1, 6, 2, 5, 4];
    /// The slot of [`SEL51`] the host tier serves.
    const HOST_SLOT: usize = 4;

    /// One synthetic Q5_1 stack of [`E51`] experts of `rpe` rows of `k`
    /// values, as the file's `block_q5_1` words.
    struct Stack51 {
        k: usize,
        rpe: usize,
        words: Vec<u32>,
        w: DeviceTensor<u32>,
    }

    impl Stack51 {
        fn new(c: &Ctx, rpe: usize, k: usize, seed: u64) -> Result<Stack51, GateError> {
            let words = synthetic_q5_1(E51 * rpe * (k / 32), seed);
            let w = DeviceTensor::upload(c.gpu.stream(), &words, E51 * rpe, BLOCK_WORDS * k / 32)?;
            Ok(Stack51 { k, rpe, words, w })
        }

        /// Expert `e`'s rows as bytes.
        fn expert_bytes(&self, e: usize) -> Vec<u8> {
            let wpe = self.rpe * self.w.cols();
            words_bytes(&self.words[e * wpe..(e + 1) * wpe])
        }
    }

    /// `blocks` synthetic `block_q5_1`s as words from a fixed-seed xorshift64:
    /// `d` a positive normal f16 and `m` a normal f16 of either sign, both of
    /// exponent field 1..=9 (so `m·s` is of `A·d`'s order in many blocks and
    /// no NaN or infinity enters), `qh` and `qs` random.
    fn synthetic_q5_1(blocks: usize, seed: u64) -> Vec<u32> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let half = |r: u64| -> u32 { ((1 + (r % 9) as u32) << 10) | ((r >> 32) as u32 & 0x3ff) };
        let mut out = Vec::with_capacity(blocks * BLOCK_WORDS);
        for _ in 0..blocks {
            let d = half(next());
            let sign = if next() & 1 == 1 { 0x8000 } else { 0 };
            let m = half(next()) | sign;
            out.push(d | (m << 16));
            for _ in 1..BLOCK_WORDS {
                out.push((next() >> 32) as u32);
            }
        }
        out
    }

    /// `cols` columns of `k` values quantized by the engine's 32-value q8_1
    /// quantizer, and the values its readback dequantizes to (`d8[b]` times
    /// each code byte, from the transposed window order).
    fn quantize51(
        c: &Ctx,
        x: &[f32],
        cols: usize,
        k: usize,
    ) -> Result<(Q8Blocks32, Vec<f32>), GateError> {
        let stream = c.gpu.stream();
        let xd = DeviceBuffer::from_host(stream, &x[..cols * k])?;
        let mut act = Q8Blocks32::with_slots(stream, k, cols)?;
        c.gpu
            .q5()
            .enqueue_quantize_q8(stream, &xd, &mut act, c.gpu.unlabelled_sink())?;
        stream.synchronize()?;
        let h = act.readback(stream)?;
        let kb = k / 32;
        let q_stride = h.q.len() / cols;
        let mut xq = vec![0.0f32; cols * k];
        for col in 0..cols {
            for b in 0..kb {
                let e = h.d8[col * kb + b];
                for i in 0..8 {
                    let w = h.q[col * q_stride + 256 * (b >> 5) + 32 * i + (b & 31)];
                    for v in 0..4 {
                        let code = f32::from((w >> (8 * v)) as u8 as i8);
                        xq[col * k + 32 * b + 4 * i + v] = e * code;
                    }
                }
            }
        }
        Ok((act, xq))
    }

    /// The Q5_1 down `_sel` of `st` for `sel` (one id a column of `act`) into a
    /// `SENT`-filled output, synchronized; the output and the fault word.
    fn down51(
        c: &Ctx,
        st: &Stack51,
        act: &Q8Blocks32,
        sel: &[u32],
    ) -> Result<(Vec<f32>, Option<Fault>), GateError> {
        let stream = c.gpu.stream();
        let sel_dev = DeviceBuffer::from_host(stream, sel)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; sel.len() * st.rpe])?;
        let a = Q51SelDown {
            w: &st.w,
            act,
            sel: &sel_dev,
            n_slots: sel.len(),
            rows_per_expert: st.rpe,
        };
        c.q51
            .enqueue_gemv_q5_1_sel(stream, &a, c.gpu.unlabelled_sink(), &mut y)?;
        stream.synchronize()?;
        Ok((y.to_host_vec(stream)?, c.gpu.take_fault()?))
    }

    /// `q5_1_gemv` over the `pack_q5_1` copy of `st`: slot `s` of `sel` (not
    /// [`HOST`]) as expert `sel[s]`'s rows against column `s`, one launch a
    /// slot, into a `SENT`-filled output.
    fn packed51(
        c: &Ctx,
        st: &Stack51,
        act: &Q8Blocks32,
        sel: &[u32],
    ) -> Result<Vec<f32>, GateError> {
        let stream = c.gpu.stream();
        let rows = E51 * st.rpe;
        let packed = pack_q5_1(&words_bytes(&st.words), st.k, rows)?;
        let wp = DeviceTensor::upload(stream, &packed, rows, packed.len() / rows)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; sel.len() * st.rpe])?;
        for (s, &e) in sel.iter().enumerate() {
            if e == HOST {
                continue;
            }
            c.gpu.q5().enqueue_gemv_q5_1(
                stream,
                &wp,
                act,
                e as usize * st.rpe,
                st.rpe,
                s,
                1,
                &mut y,
                s * st.rpe,
            )?;
        }
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// Clause 7 (module doc).
    fn check_q5_1(c: &Ctx) -> Result<bool, GateError> {
        // (tag, rows an expert, K)
        const SHAPES: [(&str, usize, usize); 4] = [
            ("qwen38_down_2560x640", 2560, 640),
            ("grid_tail_lane_pass2_100x1280", 100, 1280),
            ("windows3_64x2080", 64, 2080),
            ("one_block_64x32", 64, 32),
        ];
        let stream = c.gpu.stream();
        let mut ok = true;
        let mut first = None;
        for (i, &(tag, rpe, k)) in SHAPES.iter().enumerate() {
            let st = Stack51::new(c, rpe, k, 0x51b1_0001 + i as u64)?;
            let x = activations(k, SEL51.len(), 7701 + i as u32);
            let (act, xq) = quantize51(c, &x, SEL51.len(), k)?;
            let (y, fault) = down51(c, &st, &act, &SEL51)?;
            let (y2, fault2) = down51(c, &st, &act, &SEL51)?;
            // The band over the slots the card serves; the host slot holds SENT.
            let (mut got, mut want) = (Vec::new(), Vec::new());
            for (s, &e) in SEL51.iter().enumerate() {
                if e == HOST {
                    continue;
                }
                got.extend_from_slice(&y[s * rpe..(s + 1) * rpe]);
                want.extend(ref_gemv(
                    GgmlType::Q5_1,
                    &st.expert_bytes(e as usize),
                    k,
                    rpe,
                    &xq[s * k..(s + 1) * k],
                    1,
                )?);
            }
            let rel = max_rel_err(&got, &want)?;
            let packed = packed51(c, &st, &act, &SEL51)?;
            let packed_same = bits_equal(&y, &packed);
            let host_untouched = y[HOST_SLOT * rpe..(HOST_SLOT + 1) * rpe]
                .iter()
                .all(|v| v.to_bits() == SENT.to_bits());
            let rerun = bits_equal(&y, &y2);
            let pass = fault.is_none()
                && fault2.is_none()
                && rel <= KERNEL_BAND
                && packed_same
                && host_untouched
                && rerun;
            ok &= pass;
            println!(
                "q5_1[{tag}] rows={rpe} K={k} k_blocks={} sel={SEL51:?} max_rel={rel:.3e} \
                 band={KERNEL_BAND:e} is_packed_q5_1_gemv={packed_same} max_ulps_vs_packed={} \
                 host_slot_untouched={host_untouched} rerun_bit_identical={rerun} \
                 faults=\"{}\",\"{}\" {}{}",
                k / 32,
                max_ulps(&y, &packed),
                shown(fault),
                shown(fault2),
                verdict(pass),
                mismatch("first_mismatch_vs_packed", &y, &packed, rpe),
            );
            if i == 0 {
                first = Some((st, x, act));
            }
        }
        let (st, x, act) = first.ok_or("gate_kquant: no Q5_1 stack for the graph case")?;
        let (rpe, k) = (st.rpe, st.k);
        let (clean, _) = down51(c, &st, &act, &SEL51)?;
        // The graph: captured with SEL51, replayed, the ids overwritten outside it.
        let sel_b: [u32; 10] = [1, 6, HOST, 0, 3, 5, 2, 2, 4, 0];
        let eager_b = down51(c, &st, &act, &sel_b)?.0;
        let mut sel_dev = DeviceBuffer::from_host(stream, &SEL51)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; SEL51.len() * rpe])?;
        let graph = c.gpu.capture(|s| {
            let a = Q51SelDown {
                w: &st.w,
                act: &act,
                sel: &sel_dev,
                n_slots: SEL51.len(),
                rows_per_expert: rpe,
            };
            c.q51
                .enqueue_gemv_q5_1_sel(s, &a, c.gpu.unlabelled_sink(), &mut y)
        })?;
        let nodes = graph.node_count();
        graph.launch(stream)?;
        stream.synchronize()?;
        let ya = y.to_host_vec(stream)?;
        sel_dev.copy_from_host(stream, &sel_b)?;
        // The host slot moved: the replay leaves the old bytes there, as eager
        // leaves SENT, so compare it against a SENT-filled output.
        y.copy_from_host(stream, &vec![SENT; SEL51.len() * rpe])?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let yb = y.to_host_vec(stream)?;
        drop(graph);
        let (a_same, b_same) = (bits_equal(&ya, &clean), bits_equal(&yb, &eager_b));
        let pass = a_same && b_same && nodes == 1 && c.gpu.take_fault()?.is_none();
        ok &= pass;
        println!(
            "q5_1_graph[{rpe}x{k}] replay_a_bit_identical={a_same} replay_b_bit_identical={b_same} \
             graph_nodes={nodes} {}{}{}",
            verdict(pass),
            mismatch("first_mismatch_a", &ya, &clean, rpe),
            mismatch("first_mismatch_b", &yb, &eager_b, rpe),
        );
        // An id past the stack in slot 1.
        let expert_id = Some(Fault::at(LAYER_NONE, FaultSite::ExpertId));
        let mut sel = SEL51;
        sel[1] = E51 as u32 + 3;
        let (got, fault) = down51(c, &st, &act, &sel)?;
        let mut want = clean.clone();
        want[rpe..2 * rpe].fill(SENT);
        let values_ok = bits_equal(&got, &want);
        let pass = values_ok && fault == expert_id;
        ok &= pass;
        println!(
            "q5_1_fault[past_stack] sel={sel:?} fault=\"{}\" want=\"{}\" \
             slot1_untouched_others_clean={values_ok} {}{}",
            shown(fault),
            shown(expert_id),
            verdict(pass),
            mismatch("first_mismatch", &got, &want, rpe),
        );
        // A NaN in column 2: the quantizer's fault; the down adds none.
        let q5_quant = Some(Fault::at(LAYER_NONE, FaultSite::Q5Quant));
        let mut xn = x.clone();
        xn[2 * k + 77] = f32::NAN;
        let (actn, _) = quantize51(c, &xn, SEL51.len(), k)?;
        let qf = c.gpu.take_fault()?;
        let (yn, fd) = down51(c, &st, &actn, &SEL51)?;
        let nan_rows = yn[2 * rpe..3 * rpe].iter().filter(|v| v.is_nan()).count();
        let pass = qf == q5_quant && fd.is_none();
        ok &= pass;
        println!(
            "q5_1_fault[nan_column] quantizer_fault=\"{}\" want=\"{}\" down_fault=\"{}\" \
             column2_nan_rows={nan_rows}/{rpe} {}",
            shown(qf),
            shown(q5_quant),
            shown(fd),
            verdict(pass)
        );
        ok &= check_q5_1_host_contract(c, &st, &act)?;
        Ok(ok)
    }

    /// Clause 7's refusals: each a `Shape` error of the launcher.
    fn check_q5_1_host_contract(
        c: &Ctx,
        st: &Stack51,
        act: &Q8Blocks32,
    ) -> Result<bool, GateError> {
        let stream = c.gpu.stream();
        let what = "enqueue_gemv_q5_1_sel";
        let sel = DeviceBuffer::from_host(stream, &SEL51)?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; SEL51.len() * st.rpe])?;
        let sink = c.gpu.unlabelled_sink();
        let other_k = Q8Blocks32::with_slots(stream, 2 * st.k, SEL51.len())?;
        let few = Q8Blocks32::new(stream, st.k, 2)?;
        let down = |act: &Q8Blocks32, rpe: usize, y: &mut DeviceBuffer<f32>| {
            c.q51.enqueue_gemv_q5_1_sel(
                stream,
                &Q51SelDown {
                    w: &st.w,
                    act,
                    sel: &sel,
                    n_slots: SEL51.len(),
                    rows_per_expert: rpe,
                },
                sink,
                y,
            )
        };
        let cases: [(&str, Result<(), GpuError>); 3] = [
            ("act_of_twice_k", down(&other_k, st.rpe, &mut y)),
            ("act_2_columns_for_10_slots", down(&few, st.rpe, &mut y)),
            ("rows_per_expert_300", down(act, 300, &mut y)),
        ];
        let mut ok = true;
        for (case, r) in cases {
            let pass = matches!(&r, Err(GpuError::Shape { what: w, .. }) if *w == what);
            let seen = match &r {
                Ok(()) => "Ok (accepted)".to_string(),
                Err(e) => format!("Err: {e}"),
            };
            println!(
                "q5_1_host[{case}] want=Err(Shape {what}) got={seen} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        stream.synchronize()?;
        Ok(ok && c.gpu.take_fault()?.is_none())
    }

    /// Clause 8 (module doc).
    fn check_q8_0(c: &Ctx) -> Result<bool, GateError> {
        // (tag, experts, rows an expert, K); one expert is the non-routed form.
        const SHAPES: [(&str, usize, usize, usize); 8] = [
            ("k256", E, 512, 256),
            ("k768", E, 512, 768),
            ("k1280", E, 512, 1280),
            ("k2304", E, 512, 2304),
            ("shexp_gate_up_2048x4096", 1, 2048, 4096),
            ("shexp_down_4096x2048", 1, 4096, 2048),
            ("dense_gate_up_12288x4096", 1, 12288, 4096),
            ("dense_down_4096x12288", 1, 4096, 12288),
        ];
        const SEL: [u32; 4] = [3, 0, 2, 1];
        let mut ok = true;
        for (i, &(tag, experts, rpe, k)) in SHAPES.iter().enumerate() {
            let st = Stack::with_experts(
                c,
                tag,
                GgmlType::Q8_0,
                experts,
                rpe,
                k,
                0x8a00_0001 + i as u64,
            )?;
            let sel = if experts == 1 { [0; 4] } else { SEL };
            ok &= band_case(c, "q8_0_band", &st, &sel, 7701 + i as u32)?;
        }
        ok &= act_cases(c, GgmlType::Q8_0, "q8_0_act")?;
        ok &= fault_cases(c, GgmlType::Q8_0, "q8_0_fault")?;
        ok &= check_q8_0_host(c)?;
        Ok(ok)
    }

    /// Clause 8's refusals: each launcher refuses a stack of the other
    /// format's row width, a `Shape` error of its own.
    fn check_q8_0_host(c: &Ctx) -> Result<bool, GateError> {
        let stream = c.gpu.stream();
        let (rpe, k) = (64, 512);
        let q5 = Stack::new(c, "q5k", GgmlType::Q5_K, rpe, k, 0x8a0f_0001)?;
        let q8 = Stack::new(c, "q8_0", GgmlType::Q8_0, rpe, k, 0x8a0f_0002)?;
        let act2 = quantize(c, &activations(k, 2, 7801), 2, k)?;
        let sel = DeviceBuffer::from_host(stream, &[0u32; 2])?;
        let mut y = DeviceBuffer::from_host(stream, &vec![SENT; 2 * rpe])?;
        let sink = c.gpu.unlabelled_sink();
        let (down_q5, down_q8) = (
            SelDown {
                w: &q5.w,
                act: &act2,
                sel: &sel,
                n_slots: 2,
                rows_per_expert: rpe,
            },
            SelDown {
                w: &q8.w,
                act: &act2,
                sel: &sel,
                n_slots: 2,
                rows_per_expert: rpe,
            },
        );
        let gate_up_q5 = GateUpAct {
            wg: &q5.w,
            wu: &q5.w,
            act: &act2,
            sel: &sel,
            n_slots: 2,
            rows_per_expert: rpe,
            slots_per_col: 1,
            rule: Act::SwigluClamp { limit: 10.0 },
        };
        let cases: [(&str, &str, Result<(), GpuError>); 3] = [
            (
                "enqueue_gemv_q8_0_sel",
                "q5k_rows",
                c.kq.enqueue_gemv_q8_0_sel(stream, &down_q5, sink, &mut y),
            ),
            (
                "enqueue_gate_up_q8_0",
                "q5k_rows",
                c.kq.enqueue_gate_up_q8_0(stream, &gate_up_q5, sink, &mut y),
            ),
            (
                "enqueue_gemv_q5k_sel",
                "q8_0_rows",
                c.kq.enqueue_gemv_q5k_sel(stream, &down_q8, sink, &mut y),
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
                "q8_0_host[{what}:{case}] want=Err(Shape {what}) got={seen} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        stream.synchronize()?;
        Ok(ok && c.gpu.take_fault()?.is_none())
    }

    /// Clause 10's card experts: Qwen3.8's shapes, [`E51`] experts a stack.
    const E38: usize = E51;
    /// Clause 10's tokens.
    const M38: usize = 2;
    /// Clause 10's hidden width and routed expert width.
    const N38: usize = 2560;
    const FF38: usize = 640;
    /// Clause 10's places, ten a token: token 0 [`SEL51`], token 1 two host
    /// slots and a repeat.
    const SEL38: [u32; 20] = [
        6, 0, 3, 3, HOST, 1, 6, 2, 5, 4, //
        1, HOST, 5, 0, HOST, 2, 4, 6, 3, 3,
    ];
    /// The places the graph replays after its capture: the host slots moved.
    const SEL38_B: [u32; 20] = [
        HOST, 0, 3, 5, 1, 1, 6, HOST, 5, 4, //
        1, 2, 5, 0, 4, 2, HOST, 6, 3, HOST,
    ];

    /// The router's weights of clause 10's tokens: each token's own ten, then
    /// its shared expert's gate, none two alike across the tokens.
    fn weights38() -> Vec<f32> {
        (0..M38 * W_PITCH)
            .map(|i| {
                let (t, j) = (i / W_PITCH, i % W_PITCH);
                if j == SLOTS {
                    [0.71, 0.23][t]
                } else {
                    0.013 * (j + 1) as f32 + 0.37 * t as f32 + 0.001
                }
            })
            .collect()
    }

    /// Clause 10's buffers: the stacks, the tokens' q8_1 columns, the places
    /// and weights, and every launch's output.
    struct Leg38 {
        q38: Q38Kernels,
        gate: Stack,
        up: Stack,
        down: Stack51,
        act_x: Q8Act,
        sel: DeviceBuffer<u32>,
        w: DeviceBuffer<f32>,
        hsum: DeviceBuffer<f32>,
        sh: DeviceBuffer<f32>,
        h: DeviceBuffer<f32>,
        act_h: Q8Blocks32,
        y_down: DeviceBuffer<f32>,
        acc: DeviceBuffer<f32>,
        y: DeviceBuffer<f32>,
    }

    impl Leg38 {
        fn new(c: &Ctx) -> Result<Leg38, GateError> {
            let stream = c.gpu.stream();
            let slots = SEL38.len();
            let x = activations(N38, M38, 9101);
            Ok(Leg38 {
                q38: Q38Kernels::load(c.gpu.context())?,
                gate: Stack::with_experts(c, "q38_gate", GgmlType::Q4_K, E38, FF38, N38, 0x38a1)?,
                up: Stack::with_experts(c, "q38_up", GgmlType::Q4_K, E38, FF38, N38, 0x38a2)?,
                down: Stack51::new(c, N38, FF38, 0x38a3)?,
                act_x: quantize(c, &x, M38, N38)?,
                sel: DeviceBuffer::from_host(stream, &SEL38)?,
                w: DeviceBuffer::from_host(stream, &weights38())?,
                hsum: DeviceBuffer::from_host(stream, &activations(N38, M38, 9102))?,
                sh: DeviceBuffer::from_host(stream, &activations(N38, M38, 9103))?,
                h: DeviceBuffer::from_host(stream, &vec![f32::NAN; slots * FF38])?,
                act_h: Q8Blocks32::with_slots(stream, FF38, slots)?,
                y_down: DeviceBuffer::from_host(stream, &vec![f32::NAN; slots * N38])?,
                acc: DeviceBuffer::from_host(stream, &vec![SENT; M38 * N38])?,
                y: DeviceBuffer::from_host(stream, &vec![SENT; M38 * N38])?,
            })
        }

        /// The leg's five launches on `s`, in the step's order.
        fn enqueue(&mut self, c: &Ctx, s: &cuda_core::CudaStream) -> Result<(), GpuError> {
            let slots = SEL38.len();
            let sink = c.gpu.unlabelled_sink();
            let a = GateUpAct {
                wg: &self.gate.w,
                wu: &self.up.w,
                act: &self.act_x,
                sel: &self.sel,
                n_slots: slots,
                rows_per_expert: FF38,
                slots_per_col: SLOTS,
                rule: Act::SiluMul,
            };
            c.kq.enqueue_gate_up_q4k(s, &a, sink, &mut self.h)?;
            let q = QuantSel {
                x: &self.h,
                cols: 0..slots,
                sel: &self.sel,
                n_card: E38,
            };
            c.gpu
                .q5()
                .enqueue_quantize_q8_sel(s, &q, &mut self.act_h, sink)?;
            let d = Q51SelDown {
                w: &self.down.w,
                act: &self.act_h,
                sel: &self.sel,
                n_slots: slots,
                rows_per_expert: N38,
            };
            c.q51.enqueue_gemv_q5_1_sel(s, &d, sink, &mut self.y_down)?;
            self.q38.enqueue_card_acc(
                s,
                CardAccArgs {
                    down: &self.y_down,
                    w: &self.w,
                    sel: &self.sel,
                    n: N38,
                    m: M38,
                    n_card: E38,
                    acc: &mut self.acc,
                },
            )?;
            self.q38.enqueue_card_shared_add(
                s,
                CardSharedAddArgs {
                    hsum: &self.hsum,
                    acc: &self.acc,
                    sh: &self.sh,
                    w: &self.w,
                    slot: SLOTS,
                    slots: W_PITCH,
                    n: N38,
                    m: M38,
                    fault: sink,
                    y: &mut self.y,
                },
            )
        }

        /// The runtime's rule on the leg's own down outputs for places `sel`:
        /// each value's card sum and its combine. A host slot's down value is
        /// NaN here, so reading it would show.
        fn host_rule(&self, c: &Ctx, sel: &[u32; 20]) -> Result<(Vec<f32>, Vec<f32>), GateError> {
            let stream = c.gpu.stream();
            let down = self.y_down.to_host_vec(stream)?;
            let w = weights38();
            let (hsum, sh) = (self.hsum.to_host_vec(stream)?, self.sh.to_host_vec(stream)?);
            let (mut acc, mut y) = (vec![0.0f32; M38 * N38], vec![0.0f32; M38 * N38]);
            for t in 0..M38 {
                let card: [bool; SLOTS] =
                    std::array::from_fn(|j| (sel[t * SLOTS + j] as usize) < E38);
                let wv: [f32; SLOTS] = std::array::from_fn(|j| w[t * W_PITCH + j]);
                for d in 0..N38 {
                    let dv: [f32; SLOTS] = std::array::from_fn(|j| {
                        if card[j] {
                            down[(t * SLOTS + j) * N38 + d]
                        } else {
                            f32::NAN
                        }
                    });
                    let i = t * N38 + d;
                    acc[i] = runtime::combine::card_sum(dv, wv, card);
                    let shexp = sh[i] * w[t * W_PITCH + SLOTS];
                    y[i] = runtime::combine::combine(dv, wv, card, hsum[i], shexp);
                }
            }
            Ok((acc, y))
        }
    }

    /// One column of a q8_1 scratch read back: its code words, block sums and
    /// scales' bits.
    fn column38(x: &bloomery_gpu::q5::Q8Blocks32Host, s: usize) -> (Vec<u32>, Vec<i32>, Vec<u32>) {
        let (kb, slots) = (FF38 / 32, SEL38.len());
        let qs = x.q.len() / slots;
        (
            x.q[s * qs..(s + 1) * qs].to_vec(),
            x.s8[s * kb..(s + 1) * kb].to_vec(),
            x.d8[s * kb..(s + 1) * kb]
                .iter()
                .map(|v| v.to_bits())
                .collect(),
        )
    }

    /// Clause 10 (module doc).
    fn check_q38_card(c: &Ctx) -> Result<bool, GateError> {
        let stream = c.gpu.stream();
        let sink = c.gpu.unlabelled_sink();
        let slots = SEL38.len();
        let mut leg = Leg38::new(c)?;
        let mut ok = true;
        // The pattern: every column of the q8_1 scratch the plain quantizer's
        // bytes of 3.0, which no column of the gate·up's output is.
        let pattern = DeviceBuffer::from_host(stream, &vec![3.0f32; slots * FF38])?;
        c.gpu
            .q5()
            .enqueue_quantize_q8(stream, &pattern, &mut leg.act_h, sink)?;
        stream.synchronize()?;
        let before = leg.act_h.readback(stream)?;
        leg.enqueue(c, stream)?;
        stream.synchronize()?;
        let fault = c.gpu.take_fault()?;
        let after = leg.act_h.readback(stream)?;
        // The plain quantizer over the same columns, the host ones zeroed.
        let mut h = leg.h.to_host_vec(stream)?;
        let host_slots: Vec<usize> = (0..slots).filter(|&s| SEL38[s] == HOST).collect();
        let host_nan = host_slots
            .iter()
            .all(|&s| h[s * FF38..(s + 1) * FF38].iter().all(|v| v.is_nan()));
        for &s in &host_slots {
            h[s * FF38..(s + 1) * FF38].fill(0.0);
        }
        let (plain, _) = quantize51(c, &h, slots, FF38)?;
        let want = plain.readback(stream)?;
        let card_cols = (0..slots)
            .filter(|s| !host_slots.contains(s))
            .all(|s| column38(&after, s) == column38(&want, s));
        let host_cols = host_slots
            .iter()
            .all(|&s| column38(&after, s) == column38(&before, s));
        let (acc_ref, y_ref) = leg.host_rule(c, &SEL38)?;
        let acc = leg.acc.to_host_vec(stream)?;
        let y = leg.y.to_host_vec(stream)?;
        let (acc_same, y_same) = (bits_equal(&acc, &acc_ref), bits_equal(&y, &y_ref));
        let pass = fault.is_none() && host_nan && card_cols && host_cols && acc_same && y_same;
        ok &= pass;
        println!(
            "q38_card[{M38} tokens x {SLOTS} slots, {E38} card experts, host slots \
             {host_slots:?}] host_rows_poisoned={host_nan} \
             quantize_sel_card_columns_are_plain={card_cols} host_columns_untouched={host_cols} \
             card_sum_is_runtime_card_sum={acc_same} combine_is_runtime_combine={y_same} \
             max_ulps_sum={} max_ulps_combine={} fault=\"{}\" {}{}{}",
            max_ulps(&acc, &acc_ref),
            max_ulps(&y, &y_ref),
            shown(fault),
            verdict(pass),
            mismatch("first_mismatch_sum", &acc, &acc_ref, N38),
            mismatch("first_mismatch_combine", &y, &y_ref, N38),
        );
        // The graph: the five launches captured with SEL38, replayed, then the
        // places overwritten outside it and replayed again.
        let graph = c.gpu.capture(|s| leg.enqueue(c, s))?;
        let nodes = graph.node_count();
        graph.launch(stream)?;
        stream.synchronize()?;
        let ya = leg.y.to_host_vec(stream)?;
        leg.sel.copy_from_host(stream, &SEL38_B)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let yb = leg.y.to_host_vec(stream)?;
        let (_, yb_ref) = leg.host_rule(c, &SEL38_B)?;
        drop(graph);
        let (a_same, b_same) = (bits_equal(&ya, &y_ref), bits_equal(&yb, &yb_ref));
        let pass = a_same && b_same && nodes == 5 && c.gpu.take_fault()?.is_none();
        ok &= pass;
        println!(
            "q38_card_graph replay_a_bit_identical={a_same} \
             replay_b_moved_places_is_the_rule={b_same} graph_nodes={nodes} (want 5) {}{}{}",
            verdict(pass),
            mismatch("first_mismatch_a", &ya, &y_ref, N38),
            mismatch("first_mismatch_b", &yb, &yb_ref, N38),
        );
        leg.sel.copy_from_host(stream, &SEL38)?;
        // A NaN in card column 0 (slot 0 is expert 6): the `_sel` quantizer raises.
        let q5_quant = Some(Fault::at(LAYER_NONE, FaultSite::Q5Quant));
        let mut hn = h.clone();
        hn[77] = f32::NAN;
        let hn = DeviceBuffer::from_host(stream, &hn)?;
        let q = QuantSel {
            x: &hn,
            cols: 0..slots,
            sel: &leg.sel,
            n_card: E38,
        };
        c.gpu
            .q5()
            .enqueue_quantize_q8_sel(stream, &q, &mut leg.act_h, sink)?;
        stream.synchronize()?;
        let qf = c.gpu.take_fault()?;
        // A NaN card sum: the combine raises.
        let f32_product = Some(Fault::at(LAYER_NONE, FaultSite::F32Product));
        let mut an = acc.clone();
        an[N38 + 5] = f32::NAN;
        let an = DeviceBuffer::from_host(stream, &an)?;
        leg.q38.enqueue_card_shared_add(
            stream,
            CardSharedAddArgs {
                hsum: &leg.hsum,
                acc: &an,
                sh: &leg.sh,
                w: &leg.w,
                slot: SLOTS,
                slots: W_PITCH,
                n: N38,
                m: M38,
                fault: sink,
                y: &mut leg.y,
            },
        )?;
        stream.synchronize()?;
        let cf = c.gpu.take_fault()?;
        let pass = qf == q5_quant && cf == f32_product;
        ok &= pass;
        println!(
            "q38_card_fault quantize_sel_nan_card_column=\"{}\" (want \"{}\") \
             combine_nan_sum=\"{}\" (want \"{}\") {}",
            shown(qf),
            shown(q5_quant),
            shown(cf),
            shown(f32_product),
            verdict(pass)
        );
        ok &= check_q38_card_host_contract(c, &mut leg)?;
        Ok(ok)
    }

    /// Clause 10's refusals: each a `Shape` error of its launcher.
    fn check_q38_card_host_contract(c: &Ctx, leg: &mut Leg38) -> Result<bool, GateError> {
        let stream = c.gpu.stream();
        let sink = c.gpu.unlabelled_sink();
        let slots = SEL38.len();
        let w10 = DeviceBuffer::from_host(stream, &[0.5f32; M38 * SLOTS])?;
        let mut r_acc = Vec::with_capacity(2);
        for (w, n_card) in [(&leg.w, 0), (&w10, E38)] {
            r_acc.push(leg.q38.enqueue_card_acc(
                stream,
                CardAccArgs {
                    down: &leg.y_down,
                    w,
                    sel: &leg.sel,
                    n: N38,
                    m: M38,
                    n_card,
                    acc: &mut leg.acc,
                },
            ));
        }
        let mut r_quant = Vec::with_capacity(2);
        for (cols, n_card) in [(0..slots, 0), (1..slots + 1, E38)] {
            let q = QuantSel {
                x: &leg.h,
                cols,
                sel: &leg.sel,
                n_card,
            };
            r_quant.push(
                c.gpu
                    .q5()
                    .enqueue_quantize_q8_sel(stream, &q, &mut leg.act_h, sink),
            );
        }
        let r_slot = leg.q38.enqueue_card_shared_add(
            stream,
            CardSharedAddArgs {
                hsum: &leg.hsum,
                acc: &leg.acc,
                sh: &leg.sh,
                w: &leg.w,
                slot: W_PITCH,
                slots: W_PITCH,
                n: N38,
                m: M38,
                fault: sink,
                y: &mut leg.y,
            },
        );
        let [r_card0, r_pitch10]: [Result<(), GpuError>; 2] = r_acc
            .try_into()
            .map_err(|_| "gate_kquant: two card-sum cases")?;
        let [r_quant_card0, r_quant_past]: [Result<(), GpuError>; 2] = r_quant
            .try_into()
            .map_err(|_| "gate_kquant: two quantizer cases")?;
        let cases: [(&str, &str, Result<(), GpuError>); 5] = [
            ("q38::enqueue_card_acc", "a_card_of_no_expert", r_card0),
            (
                "q38::enqueue_card_acc",
                "weights_at_a_pitch_of_10",
                r_pitch10,
            ),
            (
                "enqueue_quantize_q8_sel",
                "a_card_of_no_expert",
                r_quant_card0,
            ),
            (
                "enqueue_quantize_q8_sel",
                "columns_past_the_scratch",
                r_quant_past,
            ),
            ("q38::enqueue_card_shared_add", "slot_11_of_11", r_slot),
        ];
        let mut ok = true;
        for (what, case, r) in cases {
            let pass = matches!(&r, Err(GpuError::Shape { what: w, .. }) if *w == what);
            let seen = match &r {
                Ok(()) => "Ok (accepted)".to_string(),
                Err(e) => format!("Err: {e}"),
            };
            println!(
                "q38_card_host[{case}] want=Err(Shape {what}) got={seen} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        stream.synchronize()?;
        Ok(ok && c.gpu.take_fault()?.is_none())
    }
}
