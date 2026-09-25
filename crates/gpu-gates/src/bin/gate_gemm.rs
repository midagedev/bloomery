//! GPU gate for the grouped int8 tensor-core GEMM (`bloomery_gpu::gemm`):
//! K-quant expert stacks times q8_1 activations for a batch of routed slots,
//! the route table built on the card from the ids.
//!
//! Cases: the real Qwen3-30B-A3B stacks (the model file of the qwen3moe
//! profile) — a routed gate stack (Q4_K, 2048 → 768, 128 experts, top-8,
//! every slot reading its token's column), a Q4_K and a Q6_K down stack
//! (768 → 2048, each slot its own column; 630-byte Q6_K rows, every other
//! one at 2 mod 4) — and one routed gate stack of DeepSeek-V4.1-Flash (Q3_K,
//! 5120 → 2304, 384 experts, top-6; `BLOOMERY_V41_MODEL`); then synthetic
//! stacks from a fixed seed at the same shapes for Q4_K, Q6_K and Q3_K, a
//! Q5_K stack at the Qwen3 gate shape, and a Q4_K stack whose rows per expert
//! (400) end in a partial 128-row slab. Each case runs T ∈ {1, 15, 16, 17,
//! 64, 511, 512, 4096} tokens (4096 the largest ubatch) under four routings:
//! uniform, every token on the same top-k experts, a quarter of the experts
//! only, and slots dealt to experts round-robin (one slot per expert while
//! there are fewer slots than experts).
//!
//! Checks per run:
//! 1. Every output of every slot written (`y` starts at a sentinel).
//! 2. Every checked output within its derived band of the f64 reference:
//!    the dot of the exactly decoded weights with the q8_1-reconstructed
//!    activation, summed per 128-value block as the kernel groups it. The
//!    kernel's per-block integers are exact; each block adds at most four
//!    f32 roundings (the i32 → f32 conversion, `d·isum`, the min fma, the
//!    accumulating fma) and the accumulator sees `nb` blocks, so
//!    `|y − ref| <= γ(nb + 4) · Σ_b (|D_b| + |M_b|)` with `D_b =
//!    d8·d·isum`, `M_b = d8·dmin·imin` (`M_b = 0` for Q6_K and Q3_K). The
//!    decode is checked against ggml's own `dequant_row` on every decoded
//!    row.
//! 3. Every checked output bit for bit the host transcription of the
//!    module's numeric contract: per 128-value block the exact i32 `isum`
//!    and `imin`, then `acc = fma(d8, fma(−dmin, f32(imin), d·f32(isum)),
//!    acc)` (Q6_K, Q3_K: `acc = fma(d8, d·f32(isum), acc)`) from `acc = 0`,
//!    blocks in increasing k. The band of 2 admits any order or rounding
//!    inside it; this check admits the contract's alone.
//! 4. Diagnostic, not a pin: the max relative difference to today's `_sel`
//!    gemv over the same q8_1 codes (`q4k_gemv_sel`, `q6k_gemv_sel`,
//!    `q3k_gemv_sel`; Q5_K has none), and an FNV-1a digest of every output
//!    of the run (`y_fnv`), which two builds' logs compare line by line.
//!
//! Checked outputs: every row of every slot up to 136 slots; above, every
//! slot's rows `r` with `(r + s) % 4 == 0`, so each row is checked on a
//! quarter of its expert's slots.
//!
//! `--case <text>` runs only the cases whose name contains `text` (the
//! dense case is `dense`, the faults and refusals `fault`, the ubatch router
//! `router`, the route table `route_table`).
//!
//! The route table (`GemmKernels::enqueue_route`, case `route_table`) for T
//! ∈ {1, 17, 512, 1000, 4095, 4096} tokens under the four routings, on a
//! Qwen3 stack's shape (128 experts, top-8: up to 32,768 slots) and a V4.1
//! one's (384 experts, top-6), and the dense table (`enqueue_route_dense`)
//! at the same counts: the slot list and the tiles bit for bit the host's
//! stable grouping — experts ascending, each expert's slots ascending, runs
//! cut into tiles of at most `GEMM_BN`. Then at 32,768 slots: three ids past
//! the stack in three different chunks end as the named fault with the
//! table the host builds without them, and a captured route's replay writes
//! the eager table.
//!
//! The ubatch router the GEMM prefill routes with
//! (`RouterKernels::enqueue_ubatch`: the logits launch, then the routing
//! launch) runs on the file's layer-0 router weight for T ∈ {1, 7, 8, 9, 15,
//! 16, 17, 31, 33, 63, 512} tokens: every logit, probability, id and weight
//! bit for bit what the fused router (`enqueue_fused`, eight tokens a launch)
//! writes for the same columns, and every logit within `γ(k/32 + 5) · Σ
//! |w·x|` of the f64 dot (each lane's sequential sum of `k/32` products, then
//! the five-step butterfly), T = 4096 included. Then its refusals (tokens
//! past the buffers, none, a short input, a weight of the wrong shape, a
//! ubatch past `UBATCH` tokens).
//!
//! The SwiGLU quantizer between gate·up and down
//! (`GemmKernels::enqueue_swiglu_quant`, case `swiglu`) at K ∈ {768, 2048}
//! for 1, 8, 64 and 4096 columns: its five planes bit for bit what
//! `ElemKernels::enqueue_swiglu` then `Gpu::enqueue_quantize_gemm` write;
//! the scales and code sums the host transcription of the quantizer gives
//! from those SwiGLU rows; every SwiGLU value within four f32 ulps (and
//! 2^-126) of the f64 `g / (1 + e^-g) · u`. A NaN in a gate row ends as the
//! named `QuantColumn` fault; its refusals (no columns, more than the
//! activation holds, a short input).
//!
//! And once per case: a rerun bit-identical, the route and the GEMM
//! captured into a graph whose replay equals the eager launch. Then the
//! faults: a NaN activation and an out-of-range id each end as the named
//! error `GpuError::Fault`, the refused slot's outputs untouched; the same
//! for the three `_sel` gemvs, where an id past the stack raises
//! `FaultSite::ExpertId` and a host-served slot (`hybrid::HOST`) raises
//! nothing; and the host API's refusals (type, K, slot count, row count,
//! unfilled table, token shape, activation capacity). Last, the fault's site
//! mask: two sites raised in one layer after a third in a later layer are
//! all the first layer's mask holds, on the card and in the argmax's copy.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("gate_gemm: built without the `gpu` feature; see `just gate-gpu-gemm`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_gemm", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::router::{
        MAX_TOKENS, N_EXPERT, N_USED, RouterKernels, RouterOut,
    };
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
    use bloomery_gpu::gemm::{
        GEMM_BN, GEMM_MAX_SLOTS, GemmAct, GemmInput, GemmKernels, GemmRoute, GemmTile, GemmWeight,
    };
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::q6k_sel::Q6kSelKernels;
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE, Q8Act};
    use bloomery_gpu_gates::rounding::gamma;
    use bloomery_gpu_gates::{
        GateError, activations, bits_equal, bytes_to_words, checks_failed, open_model, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::{GgmlType, dequant_row, half_to_f32};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// What `y` holds before a launch, so an output the kernel leaves alone
    /// reads back as these bits.
    const SENT: f32 = 1.0e30;
    /// Token counts per case.
    const TOKENS: [usize; 8] = [1, 15, 16, 17, 64, 511, 512, UBATCH];
    /// Slot counts up to which every row of every slot is checked.
    const ALL_ROWS_UP_TO: usize = 136;

    #[derive(Clone, Copy, Debug)]
    enum Routing {
        Uniform,
        SameTopk,
        Quarter,
        RoundRobin,
    }

    const ROUTINGS: [Routing; 4] = [
        Routing::Uniform,
        Routing::SameTopk,
        Routing::Quarter,
        Routing::RoundRobin,
    ];

    /// A 64-bit LCG, the gate's one source of synthetic values.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 11
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// The ids of `n_tok` tokens × `top_k` distinct experts of `n_exp` under
    /// routing `r`, slot `t·top_k + k`.
    fn route_ids(r: Routing, n_tok: usize, top_k: usize, n_exp: usize, seed: u64) -> Vec<u32> {
        let mut g = Lcg(seed);
        let mut ids = Vec::with_capacity(n_tok * top_k);
        for t in 0..n_tok {
            let pool = match r {
                Routing::Quarter => (n_exp / 4).max(top_k),
                _ => n_exp,
            };
            let mut pick: Vec<u32> = Vec::with_capacity(top_k);
            for k in 0..top_k {
                let e = match r {
                    Routing::SameTopk => (7 * k + 3) % n_exp,
                    Routing::RoundRobin => (t * top_k + k) % n_exp,
                    Routing::Uniform | Routing::Quarter => loop {
                        let c = g.below(pool);
                        if !pick.contains(&(c as u32)) {
                            break c;
                        }
                    },
                };
                pick.push(e as u32);
            }
            ids.extend(pick);
        }
        ids
    }

    /// One stack under test: its type, bytes (the file's stream), geometry.
    struct Stack<'a> {
        name: String,
        ty: GemmWeight,
        bytes: std::borrow::Cow<'a, [u8]>,
        n_exp: usize,
        rows: usize,
        k: usize,
        input: Input,
    }

    /// The case's slot-to-column rule, with the top-k the routing uses.
    #[derive(Clone, Copy)]
    enum Input {
        Shared(usize),
        PerSlot(usize),
    }

    impl Input {
        fn top_k(self) -> usize {
            match self {
                Input::Shared(k) | Input::PerSlot(k) => k,
            }
        }
        fn gemm(self) -> GemmInput {
            match self {
                Input::Shared(top_k) => GemmInput::Shared { top_k },
                Input::PerSlot(_) => GemmInput::PerSlot,
            }
        }
        /// Activation columns `n_slots` slots read.
        fn cols(self, n_slots: usize) -> usize {
            match self {
                Input::Shared(k) => n_slots / k,
                Input::PerSlot(_) => n_slots,
            }
        }
        fn col_of(self, slot: usize) -> usize {
            match self {
                Input::Shared(k) => slot / k,
                Input::PerSlot(_) => slot,
            }
        }
    }

    /// One row decoded to the integers the kernel multiplies: per
    /// super-block `d`, `dmin`; per scale group of `grp` values its scale;
    /// per 32 values its min (Q4_K, Q5_K); per value its code.
    struct Dec {
        grp: usize,
        d: Vec<f64>,
        dmin: Vec<f64>,
        sc: Vec<i32>,
        mn: Vec<i32>,
        q: Vec<i8>,
    }

    fn f16(b: &[u8]) -> f64 {
        f64::from(half_to_f32(u16::from_le_bytes([b[0], b[1]])))
    }

    /// ggml's `get_scale_min_k4`.
    fn scale_min_k4(j: usize, q: &[u8]) -> (i32, i32) {
        if j < 4 {
            (i32::from(q[j] & 63), i32::from(q[j + 4] & 63))
        } else {
            (
                i32::from((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4)),
                i32::from((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
            )
        }
    }

    /// Decode one row of `k` values of type `ty` (ggml's `dequantize_row_*`
    /// read as integers).
    fn decode(ty: GemmWeight, row: &[u8], k: usize) -> Dec {
        let n_sb = k / 256;
        let bpb = ty.block_bytes();
        let grp = match ty {
            GemmWeight::Q4K | GemmWeight::Q5K => 32,
            GemmWeight::Q6K | GemmWeight::Q3K => 16,
        };
        let mut dec = Dec {
            grp,
            d: Vec::with_capacity(n_sb),
            dmin: Vec::with_capacity(n_sb),
            sc: Vec::with_capacity(k / grp),
            mn: Vec::with_capacity(k / 32),
            q: Vec::with_capacity(k),
        };
        for sb in 0..n_sb {
            let b = &row[sb * bpb..(sb + 1) * bpb];
            match ty {
                GemmWeight::Q4K | GemmWeight::Q5K => {
                    dec.d.push(f16(&b[0..2]));
                    dec.dmin.push(f16(&b[2..4]));
                    for j in 0..8 {
                        let (s, m) = scale_min_k4(j, &b[4..16]);
                        dec.sc.push(s);
                        dec.mn.push(m);
                    }
                    for v in 0..256 {
                        let (p, h, l) = (v / 64, (v % 64) / 32, v % 32);
                        let q = if ty == GemmWeight::Q4K {
                            (b[16 + 32 * p + l] >> (4 * h)) & 0xf
                        } else {
                            ((b[48 + 32 * p + l] >> (4 * h)) & 0xf)
                                | (((b[16 + l] >> (2 * p + h)) & 1) << 4)
                        };
                        dec.q.push(q as i8);
                    }
                }
                GemmWeight::Q6K => {
                    dec.d.push(f16(&b[208..210]));
                    dec.dmin.push(0.0);
                    for g in 0..16 {
                        dec.sc.push(i32::from(b[192 + g] as i8));
                    }
                    for v in 0..256 {
                        let (n, c, l) = (v / 128, (v % 128) / 32, v % 32);
                        let ql = b[64 * n + l + 32 * (c & 1)];
                        let qh = b[128 + 32 * n + l];
                        let q = ((ql >> (4 * (c >> 1))) & 0xf) | (((qh >> (2 * c)) & 3) << 4);
                        dec.q.push(q as i8 - 32);
                    }
                }
                GemmWeight::Q3K => {
                    dec.d.push(f16(&b[108..110]));
                    dec.dmin.push(0.0);
                    let w = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
                    let (a0, a1, a2) = (w(96), w(100), w(104));
                    let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
                    let aux = [
                        (a0 & k2) | ((a2 & k1) << 4),
                        (a1 & k2) | (((a2 >> 2) & k1) << 4),
                        ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4),
                        ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4),
                    ];
                    for g in 0..16 {
                        dec.sc
                            .push(((aux[g / 4] >> (8 * (g % 4))) & 0xff) as i32 - 32);
                    }
                    for v in 0..256 {
                        let (n, j, l) = (v / 128, (v % 128) / 32, v % 32);
                        let low = ((b[32 + 32 * n + l] >> (2 * j)) & 3) as i8;
                        let high = b[l] & (1 << (4 * n + j)) != 0;
                        dec.q.push(low - if high { 0 } else { 4 });
                    }
                }
            }
        }
        dec
    }

    /// Whether `dec` reproduces ggml's `dequant_row` of the same row: every
    /// value within four f32 roundings of its two terms' magnitude (ggml
    /// forms `d·sc` and `dmin·m` in f32 and then the value).
    fn decode_matches(ty: GgmlType, row: &[u8], dec: &Dec) -> Result<bool, GateError> {
        let k = dec.q.len();
        let mut want = vec![0.0f32; k];
        dequant_row(ty, row, &mut want)?;
        Ok((0..k).all(|v| {
            let sb = v / 256;
            let a = dec.d[sb] * f64::from(dec.sc[v / dec.grp]) * f64::from(dec.q[v]);
            let m = if dec.mn.is_empty() {
                0.0
            } else {
                dec.dmin[sb] * f64::from(dec.mn[v / 32])
            };
            ((a - m) - f64::from(want[v])).abs() <= gamma(4) * (a.abs() + m.abs()) + 1e-38
        }))
    }

    /// The host transcription of the q8_1 quantizer for one column: per 128
    /// values `d8 = amax/127` (1 for an all-zero block) and codes
    /// `round(x/d8)` clamped to ±127.
    fn q8_column(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
        let mut codes = Vec::with_capacity(x.len());
        let mut d8 = Vec::with_capacity(x.len() / 128);
        for blk in x.as_chunks::<128>().0 {
            let amax = blk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            d8.push(d);
            codes.extend(
                blk.iter()
                    .map(|&v| (v / d).round().clamp(-127.0, 127.0) as i8),
            );
        }
        (codes, d8)
    }

    /// One output three ways, summed per 128-value block as the kernel groups
    /// it: the f64 reference, its band's magnitude `Σ_b (|D_b| + |M_b|)`, and
    /// the module's numeric contract transcribed — per block the exact i32
    /// `isum` and `imin`, then `acc = fma(d8, fma(−dmin, f32(imin),
    /// d·f32(isum)), acc)` (`acc = fma(d8, d·f32(isum), acc)` for a type
    /// without mins) from `acc = 0`, blocks in increasing k. `d` and `dmin`
    /// are f16 values, exact in f32.
    fn dot_ref(dec: &Dec, a: &[i8], d8: &[f32]) -> (f64, f64, f32) {
        let (mut y, mut mag) = (0.0f64, 0.0f64);
        let mut acc = 0.0f32;
        for (b, &db) in d8.iter().enumerate() {
            let sb = b / 2;
            let mut isum = 0i32;
            for g in (128 * b..128 * b + 128).step_by(dec.grp) {
                let dot: i32 = (g..g + dec.grp)
                    .map(|v| i32::from(dec.q[v]) * i32::from(a[v]))
                    .sum();
                isum += dec.sc[g / dec.grp] * dot;
            }
            let mut imin = 0i32;
            if !dec.mn.is_empty() {
                for j in (128 * b..128 * b + 128).step_by(32) {
                    let s: i32 = a[j..j + 32].iter().map(|&v| i32::from(v)).sum();
                    imin += dec.mn[j / 32] * s;
                }
            }
            let (d, dmin) = (dec.d[sb] as f32, dec.dmin[sb] as f32);
            let t = d * isum as f32;
            let t = if dec.mn.is_empty() {
                t
            } else {
                (-dmin).mul_add(imin as f32, t)
            };
            acc = db.mul_add(t, acc);
            let dd = f64::from(db) * dec.d[sb] * f64::from(isum);
            let mm = f64::from(db) * dec.dmin[sb] * f64::from(imin);
            y += dd - mm;
            mag += dd.abs() + mm.abs();
        }
        (y, mag, acc)
    }

    /// Whether row `r` of slot `s` of an `n_slots`-slot run is checked
    /// against the reference: every row up to [`ALL_ROWS_UP_TO`] slots, a
    /// quarter of them above, rotating with the slot.
    fn row_checked(n_slots: usize, s: usize, r: usize) -> bool {
        n_slots <= ALL_ROWS_UP_TO || (r + s).is_multiple_of(4)
    }

    /// The reference check of one run: the first output outside its band,
    /// the largest error-to-band ratio, the count checked; the outputs whose
    /// bits differ from the contract's transcription and the first of them;
    /// and whether every decoded row matched ggml.
    struct RefOutcome {
        fail: Option<String>,
        worst_ratio: f64,
        checked: usize,
        bits_differ: usize,
        bits_fail: Option<String>,
        decode_ok: bool,
    }

    impl RefOutcome {
        fn new() -> RefOutcome {
            RefOutcome {
                fail: None,
                worst_ratio: 0.0,
                checked: 0,
                bits_differ: 0,
                bits_fail: None,
                decode_ok: true,
            }
        }
    }

    /// Check `y` (slot-major, `rows` per slot) against the f64 reference for
    /// every slot and the checked rows, in parallel over experts.
    fn check_ref(
        st: &Stack<'_>,
        ids: &[u32],
        codes: &[Vec<i8>],
        d8: &[Vec<f32>],
        y: &[f32],
    ) -> Result<RefOutcome, GateError> {
        let n_slots = ids.len();
        let mut by_exp: Vec<Vec<usize>> = vec![Vec::new(); st.n_exp];
        for (s, &e) in ids.iter().enumerate() {
            by_exp[e as usize].push(s);
        }
        // Work items are (expert, 64-row block), so a routing that puts every
        // slot on a few experts still spreads over every thread.
        let work: Vec<(usize, usize)> = (0..st.n_exp)
            .filter(|&e| !by_exp[e].is_empty())
            .flat_map(|e| (0..st.rows.div_ceil(64)).map(move |b| (e, b)))
            .collect();
        let next = AtomicUsize::new(0);
        let rb = st.ty.block_bytes() * st.k / 256;
        let ggml_ty = match st.ty {
            GemmWeight::Q3K => GgmlType::Q3_K,
            GemmWeight::Q4K => GgmlType::Q4_K,
            GemmWeight::Q5K => GgmlType::Q5_K,
            GemmWeight::Q6K => GgmlType::Q6_K,
        };
        let nb = st.k / 128;
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(32));
        let results: Vec<Result<RefOutcome, String>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut out = RefOutcome::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            let Some(&(e, blk)) = work.get(i) else { break };
                            for r in 64 * blk..(64 * blk + 64).min(st.rows) {
                                let off = (e * st.rows + r) * rb;
                                let row = &st.bytes[off..off + rb];
                                let dec = decode(st.ty, row, st.k);
                                out.decode_ok &=
                                    decode_matches(ggml_ty, row, &dec).map_err(|x| x.to_string())?;
                                for &s in by_exp[e].iter().filter(|&&s| row_checked(n_slots, s, r)) {
                                    let c = st.input.col_of(s);
                                    let (want, mag, contract) = dot_ref(&dec, &codes[c], &d8[c]);
                                    let got = y[s * st.rows + r];
                                    let band = gamma(nb + 4) * mag;
                                    let err = (f64::from(got) - want).abs();
                                    let ratio = if band > 0.0 { err / band } else { err };
                                    out.checked += 1;
                                    let within = err <= band;
                                    if !within && out.fail.is_none() {
                                        out.fail = Some(format!(
                                            "slot {s} row {r} expert {e}: got {got:e} want {want:e} \
                                             err {err:.3e} band {band:.3e}"
                                        ));
                                    }
                                    out.worst_ratio = out.worst_ratio.max(ratio);
                                    if got.to_bits() != contract.to_bits() {
                                        out.bits_differ += 1;
                                        if out.bits_fail.is_none() {
                                            out.bits_fail = Some(format!(
                                                "slot {s} row {r} expert {e}: got {got:e} ({:#010x}) \
                                                 contract {contract:e} ({:#010x})",
                                                got.to_bits(),
                                                contract.to_bits()
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        Ok(out)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("reference thread panicked".into()))
                })
                .collect()
        });
        let mut all = RefOutcome::new();
        for r in results {
            let r = r?;
            all.fail = all.fail.or(r.fail);
            all.worst_ratio = all.worst_ratio.max(r.worst_ratio);
            all.checked += r.checked;
            all.bits_differ += r.bits_differ;
            all.bits_fail = all.bits_fail.or(r.bits_fail);
            all.decode_ok &= r.decode_ok;
        }
        Ok(all)
    }

    /// The device side of one run: activations quantized, the route table,
    /// the GEMM; `y` read back.
    struct Dev<'g> {
        gpu: &'g Gpu,
        gk: GemmKernels,
        q6s: Q6kSelKernels,
    }

    /// The case's resident stack and scratch.
    struct Resident {
        w: DeviceTensor<u32>,
        act: GemmAct,
        route: GemmRoute,
        ids: DeviceBuffer<u32>,
        y: DeviceBuffer<f32>,
        /// `Q8Act`s of 1..=8 columns for the gemv diagnostic.
        acts: Vec<Q8Act>,
    }

    impl Dev<'_> {
        fn resident(&self, st: &Stack<'_>) -> Result<Resident, GateError> {
            let stream = self.gpu.stream();
            let n_rows = st.n_exp * st.rows;
            let mut words = bytes_to_words(&st.bytes);
            words.resize(words.len().div_ceil(n_rows) * n_rows, 0);
            let cols = words.len() / n_rows;
            let max_slots = TOKENS[TOKENS.len() - 1] * st.input.top_k();
            let acts = (1..=8)
                .map(|m| Q8Act::with_k(stream, m, st.k))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Resident {
                w: DeviceTensor::upload(stream, &words, n_rows, cols)?,
                act: GemmAct::new(stream, st.input.cols(max_slots), st.k)?,
                route: GemmRoute::new(stream, max_slots, st.n_exp)?,
                ids: DeviceBuffer::zeroed(stream, max_slots)?,
                y: DeviceBuffer::zeroed(stream, max_slots * st.rows)?,
                acts,
            })
        }

        /// Quantize `x` (`n_cols` columns), route `ids`, run the GEMM; `y`
        /// starts at [`SENT`]. Returns `y`'s first `ids.len()·rows` values.
        fn run(
            &self,
            st: &Stack<'_>,
            res: &mut Resident,
            x: &DeviceBuffer<f32>,
            ids: &[u32],
        ) -> Result<Vec<f32>, GateError> {
            let stream = self.gpu.stream();
            let n_slots = ids.len();
            res.ids.copy_from_host(stream, &pad(ids, res.ids.len()))?;
            res.y.copy_from_host(stream, &vec![SENT; res.y.len()])?;
            self.gpu.enqueue_quantize_gemm(
                x,
                st.input.cols(n_slots),
                &mut res.act,
                self.gpu.unlabelled_sink(),
            )?;
            self.gk.enqueue_route(
                stream,
                &res.ids,
                n_slots,
                &mut res.route,
                self.gpu.unlabelled_sink(),
            )?;
            self.gk.enqueue_gemm(
                stream,
                st.ty,
                &res.w,
                st.rows,
                &res.act,
                &res.route,
                st.input.gemm(),
                &mut res.y,
            )?;
            stream.synchronize()?;
            let mut y = res.y.to_host_vec(stream)?;
            y.truncate(n_slots * st.rows);
            Ok(y)
        }

        /// Today's `_sel` gemv over the same codes, slot-major like `y`: up to
        /// eight slots a launch, each slot's column quantized into its own
        /// `Q8Act` column. `None` for a type with no `_sel` kernel.
        fn gemv_ref(
            &self,
            st: &Stack<'_>,
            res: &mut Resident,
            x: &[f32],
            ids: &[u32],
        ) -> Result<Option<Vec<f32>>, GateError> {
            let stream = self.gpu.stream();
            let n_slots = ids.len();
            let mut out = vec![0.0f32; n_slots * st.rows];
            let mut ybuf = DeviceBuffer::<f32>::zeroed(stream, 8 * st.rows)?;
            match (st.ty, st.input) {
                (GemmWeight::Q5K, _) | (GemmWeight::Q3K, Input::PerSlot(_)) => return Ok(None),
                (GemmWeight::Q3K, Input::Shared(top_k)) => {
                    // One launch per token: every slot of it reads the one column.
                    for t in 0..n_slots / top_k {
                        let xc = DeviceBuffer::from_host(stream, &x[t * st.k..(t + 1) * st.k])?;
                        let act = &mut res.acts[0];
                        self.gpu.enqueue_quantize_q8_1(&xc, act)?;
                        let sel =
                            DeviceBuffer::from_host(stream, &ids[t * top_k..(t + 1) * top_k])?;
                        self.gpu
                            .enqueue_gemv_q3k_sel(&res.w, act, &sel, top_k, st.rows, &mut ybuf)?;
                        let got = ybuf.to_host_vec(stream)?;
                        out[t * top_k * st.rows..(t + 1) * top_k * st.rows]
                            .copy_from_slice(&got[..top_k * st.rows]);
                    }
                }
                (GemmWeight::Q4K | GemmWeight::Q6K, _) => {
                    for s0 in (0..n_slots).step_by(8) {
                        let m = (n_slots - s0).min(8);
                        let mut xs = Vec::with_capacity(m * st.k);
                        for s in s0..s0 + m {
                            let c = st.input.col_of(s);
                            xs.extend_from_slice(&x[c * st.k..(c + 1) * st.k]);
                        }
                        let xc = DeviceBuffer::from_host(stream, &xs)?;
                        let act = &mut res.acts[m - 1];
                        self.gpu.enqueue_quantize_q8_1(&xc, act)?;
                        let sel = DeviceBuffer::from_host(stream, &ids[s0..s0 + m])?;
                        if st.ty == GemmWeight::Q4K {
                            self.gpu.q4k_sel().enqueue_gemv_q4k_sel(
                                stream, &res.w, act, &sel, m, st.rows, &mut ybuf,
                            )?;
                        } else {
                            self.q6s.enqueue_gemv_q6k_sel(
                                stream, &res.w, act, &sel, m, st.rows, &mut ybuf,
                            )?;
                        }
                        let got = ybuf.to_host_vec(stream)?;
                        out[s0 * st.rows..(s0 + m) * st.rows].copy_from_slice(&got[..m * st.rows]);
                    }
                }
            }
            Ok(Some(out))
        }
    }

    impl Dev<'_> {
        /// One `_sel` gemv launch of `ids` over `res.w` into a [`SENT`]-filled
        /// `y`: Q3_K's slots share one column of `x`, Q4_K's and Q6_K's slot
        /// `s` reads column `s`. Returns `y`.
        fn sel_once(
            &self,
            st: &Stack<'_>,
            res: &mut Resident,
            x: &[f32],
            ids: &[u32],
        ) -> Result<Vec<f32>, GateError> {
            let stream = self.gpu.stream();
            let m = ids.len();
            let sel = DeviceBuffer::from_host(stream, ids)?;
            let mut y = DeviceBuffer::from_host(stream, &vec![SENT; m * st.rows])?;
            let cols = if st.ty == GemmWeight::Q3K { 1 } else { m };
            let xc = DeviceBuffer::from_host(stream, &x[..cols * st.k])?;
            let act = &mut res.acts[cols - 1];
            self.gpu.enqueue_quantize_q8_1(&xc, act)?;
            match st.ty {
                GemmWeight::Q3K => self
                    .gpu
                    .enqueue_gemv_q3k_sel(&res.w, act, &sel, m, st.rows, &mut y)?,
                GemmWeight::Q4K => self
                    .gpu
                    .q4k_sel()
                    .enqueue_gemv_q4k_sel(stream, &res.w, act, &sel, m, st.rows, &mut y)?,
                GemmWeight::Q6K => self
                    .q6s
                    .enqueue_gemv_q6k_sel(stream, &res.w, act, &sel, m, st.rows, &mut y)?,
                GemmWeight::Q5K => return Err("Q5_K has no _sel gemv".into()),
            }
            stream.synchronize()?;
            Ok(y.to_host_vec(stream)?)
        }
    }

    /// The `_sel` gemvs' out-of-range ids, per kernel (Q3_K, Q4_K, Q6_K) on
    /// a synthetic 16-expert stack: slot 1 at three past the stack raises
    /// [`FaultSite::ExpertId`] as an unlabelled launch, slot 1 at [`HOST`]
    /// (a slot the host tier serves) raises nothing; both leave slot 1 at
    /// [`SENT`], and every other slot bit for bit what the in-range launch
    /// writes.
    fn sel_faults(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("fault") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let (k, rows, n_exp) = (2048usize, 256usize, 16usize);
        let want_id = Fault::at(LAYER_NONE, FaultSite::ExpertId);
        let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        let mut ok = true;
        for (ty, input, seed) in [
            (GemmWeight::Q3K, Input::Shared(4), 0x5e13),
            (GemmWeight::Q4K, Input::PerSlot(4), 0x5e14),
            (GemmWeight::Q6K, Input::PerSlot(4), 0x5e16),
        ] {
            let st = Stack {
                name: format!("sel_fault_{ty:?}"),
                ty,
                bytes: synthetic(ty, n_exp * rows * k / 256, seed).into(),
                n_exp,
                rows,
                k,
                input,
            };
            let mut res = dev.resident(&st)?;
            let x = activations(k, 4, u32::try_from(seed)?);
            let clean = gpu.take_fault()?;
            let good = dev.sel_once(&st, &mut res, &x, &[3, 5, 0, 12])?;
            let good_fault = gpu.take_fault()?;
            let mut pass = clean.is_none() && good_fault.is_none();
            for (case, bad, want) in [
                ("past_stack", n_exp as u32 + 3, Some(want_id)),
                ("host", HOST, None),
            ] {
                let y = dev.sel_once(&st, &mut res, &x, &[3, bad, 0, 12])?;
                let fault = gpu.take_fault()?;
                let untouched = y[rows..2 * rows]
                    .iter()
                    .all(|v| v.to_bits() == SENT.to_bits());
                let others = [0usize, 2, 3].iter().all(|&s| {
                    bits_equal(
                        &y[s * rows..(s + 1) * rows],
                        &good[s * rows..(s + 1) * rows],
                    )
                });
                let case_ok = fault == want && untouched && others;
                println!(
                    "sel fault={case} ty={ty:?} sel=[3, {bad}, 0, 12] fault=\"{}\" want=\"{}\" \
                     slot1_untouched={untouched} other_slots_bit_identical={others} {}",
                    shown(fault),
                    shown(want),
                    verdict(case_ok)
                );
                pass &= case_ok;
            }
            if clean.is_some() || good_fault.is_some() {
                println!(
                    "sel fault=clean ty={ty:?} word_before={} after_in_range={} (want none) {}",
                    shown(clean),
                    shown(good_fault),
                    verdict(false)
                );
            }
            ok &= pass;
        }
        Ok(ok)
    }

    /// The fault's site mask: three launches raise, the first in time on
    /// layer 9 (`norm_quant` over a NaN column: [`FaultSite::NormQuant`]),
    /// then two on layer 3 (the GEMM quantizer over a NaN column,
    /// [`FaultSite::QuantColumn`], and the route over an id past the stack,
    /// [`FaultSite::ExpertId`]). The fault names layer 3, the smaller code
    /// there and a mask of exactly those two sites — layer 9's site is not in
    /// it — both read from the card's words and through the argmax copy the
    /// head's readback makes (token, word, mask).
    fn site_mask(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("fault") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let (k, rows, n_exp, top_k) = (2048usize, 256usize, 16usize, 4usize);
        let st = Stack {
            name: "site_mask".into(),
            ty: GemmWeight::Q4K,
            bytes: synthetic(GemmWeight::Q4K, n_exp * rows * k / 256, 0x0fa3).into(),
            n_exp,
            rows,
            k,
            input: Input::Shared(top_k),
        };
        let mut res = dev.resident(&st)?;
        let clean = gpu.take_fault()?;
        // Layer 9 first: a later layer raised earlier in time.
        let mut xn = activations(k, 1, 997);
        xn[11] = f32::NAN;
        let xn = DeviceBuffer::from_host(stream, &xn)?;
        let gain = DeviceBuffer::from_host(stream, &vec![1.0f32; k])?;
        let mut normed = DeviceBuffer::<f32>::zeroed(stream, k)?;
        let fused = bloomery_gpu::fused::FusedKernels::load(gpu.context())?;
        fused.enqueue_norm_quant(
            stream,
            &xn,
            &gain,
            1e-6,
            &mut res.acts[0],
            &mut normed,
            gpu.layer_sink(9)?,
        )?;
        // Then layer 3, twice.
        let n_tok = 2;
        let mut ids = route_ids(Routing::Uniform, n_tok, top_k, n_exp, 29);
        ids[2] = n_exp as u32 + 1;
        let mut x = activations(k, n_tok, 999);
        x[k + 5] = f32::NAN;
        let xd = DeviceBuffer::from_host(stream, &x)?;
        res.ids.copy_from_host(stream, &pad(&ids, res.ids.len()))?;
        let l3 = gpu.layer_sink(3)?;
        gpu.enqueue_quantize_gemm(&xd, st.input.cols(ids.len()), &mut res.act, l3)?;
        dev.gk
            .enqueue_route(stream, &res.ids, ids.len(), &mut res.route, l3)?;
        // The head's copy of the pair.
        let logits = DeviceBuffer::from_host(stream, &activations(64, 1, 1001))?;
        let mut out = DeviceBuffer::from_host(stream, &[7u32; 3])?;
        gpu.elem()
            .enqueue_argmax_fault(stream, &logits, 64, gpu.unlabelled_sink(), &mut out)?;
        let out = out.to_host_vec(stream)?;
        let copied = Fault::from_words(out[1], out[2]);
        let fault = gpu.take_fault()?;
        let want = Fault::of_sites(3, &[FaultSite::ExpertId, FaultSite::QuantColumn]);
        let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
        let masks = |f: Option<Fault>| f.map_or(0, |f| f.sites);
        let pass = clean.is_none() && fault == Some(want) && copied == Some(want);
        println!(
            "gemm fault=site_mask layer9=[norm_quant] then layer3=[quant_column, expert_id]: \
             word=\"{}\" mask={:#06x} copy=\"{}\" mask={:#06x} want=\"{want}\" mask={:#06x} \
             word_before={} {}",
            shown(fault),
            masks(fault),
            shown(copied),
            masks(copied),
            want.sites,
            shown(clean),
            verdict(pass)
        );
        Ok(pass)
    }

    /// `v` zero-padded to `n` entries.
    fn pad(v: &[u32], n: usize) -> Vec<u32> {
        let mut p = v.to_vec();
        p.resize(n, 0);
        p
    }

    /// FNV-1a over the bits of `y`: one run's every output, for comparing
    /// two builds' logs.
    fn fnv(y: &[f32]) -> u64 {
        y.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
            v.to_bits()
                .to_le_bytes()
                .iter()
                .fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
        })
    }

    /// `max|a − b| / max|b|`.
    fn rel_diff(a: &[f32], b: &[f32]) -> f64 {
        let num = a.iter().zip(b).fold(0.0f64, |m, (&x, &y)| {
            m.max((f64::from(x) - f64::from(y)).abs())
        });
        let den = b.iter().fold(0.0f64, |m, &y| m.max(f64::from(y).abs()));
        if den > 0.0 { num / den } else { num }
    }

    /// The `--case` filter: `None` runs every case.
    fn case_filter() -> Option<String> {
        let mut args = std::env::args().skip_while(|a| a != "--case");
        args.next().and(args.next())
    }

    /// Whether the case named `name` runs under the filter.
    fn wanted(name: &str) -> bool {
        case_filter().is_none_or(|f| name.contains(&f))
    }

    /// Every case: T × routing runs, then the rerun and graph checks.
    fn run_case(dev: &Dev<'_>, st: &Stack<'_>, seed: u64) -> Result<bool, GateError> {
        if !wanted(&st.name) {
            return Ok(true);
        }
        let t0 = std::time::Instant::now();
        let stream = dev.gpu.stream();
        let mut res = dev.resident(st)?;
        let top_k = st.input.top_k();
        let mut ok = true;
        for &n_tok in &TOKENS {
            for (ri, &r) in ROUTINGS.iter().enumerate() {
                let n_slots = n_tok * top_k;
                let ids = route_ids(
                    r,
                    n_tok,
                    top_k,
                    st.n_exp,
                    seed ^ (n_tok as u64) << 8 ^ ri as u64,
                );
                let n_cols = st.input.cols(n_slots);
                let x = activations(
                    st.k,
                    n_cols,
                    (seed as u32) ^ (n_tok as u32 * 131 + ri as u32),
                );
                let xd = DeviceBuffer::from_host(stream, &x)?;
                let y = dev.run(st, &mut res, &xd, &ids)?;
                let unwritten = y.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                // The device's scales and sums against the host transcription:
                // the codes the reference uses are the kernel's.
                let (codes, d8): (Vec<Vec<i8>>, Vec<Vec<f32>>) = (0..n_cols)
                    .map(|c| q8_column(&x[c * st.k..(c + 1) * st.k]))
                    .unzip();
                let dev_d8 = res.act.d8().to_host_vec(stream)?;
                let dev_s8 = res.act.s8().to_host_vec(stream)?;
                let nb = st.k / 128;
                let act_ok = (0..n_cols).all(|c| {
                    let d8_ok = (0..nb).all(|b| dev_d8[c * nb + b].to_bits() == d8[c][b].to_bits());
                    let s8_ok = (0..st.k / 32).all(|j| {
                        let s: i32 = codes[c][32 * j..32 * j + 32]
                            .iter()
                            .map(|&v| i32::from(v))
                            .sum();
                        dev_s8[c * (st.k / 32) + j] == s
                    });
                    d8_ok && s8_ok
                });
                let refc = check_ref(st, &ids, &codes, &d8, &y)?;
                let gemv = dev.gemv_ref(st, &mut res, &x, &ids)?;
                let diag = gemv.map_or_else(
                    || "none".to_string(),
                    |g| format!("{:.3e}", rel_diff(&y, &g)),
                );
                let pass = unwritten == 0
                    && act_ok
                    && refc.decode_ok
                    && refc.fail.is_none()
                    && refc.bits_differ == 0;
                println!(
                    "gemm case={} T={n_tok} routing={r:?} slots={n_slots} unwritten={unwritten} act_codes={} \
                     decode_vs_ggml={} checked={} worst_err_over_band={:.3} contract_bits_differ={} \
                     gemv_max_rel_diff={diag} y_fnv={:016x} {}",
                    st.name,
                    if act_ok { "same" } else { "DIFFER" },
                    if refc.decode_ok { "same" } else { "DIFFER" },
                    refc.checked,
                    refc.worst_ratio,
                    refc.bits_differ,
                    fnv(&y),
                    verdict(pass)
                );
                if let Some(f) = &refc.fail {
                    println!(
                        "gemm case={} T={n_tok} routing={r:?} first_failure: {f}",
                        st.name
                    );
                }
                if let Some(f) = &refc.bits_fail {
                    println!(
                        "gemm case={} T={n_tok} routing={r:?} first_contract_bits_failure: {f}",
                        st.name
                    );
                }
                ok &= pass;
            }
        }

        // Rerun and graph replay, 64 tokens, uniform routing.
        let n_tok = 64;
        let ids = route_ids(Routing::Uniform, n_tok, top_k, st.n_exp, seed ^ 0xabcd);
        let n_slots = ids.len();
        let x = activations(st.k, st.input.cols(n_slots), seed as u32 ^ 0x5151);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y1 = dev.run(st, &mut res, &xd, &ids)?;
        let y2 = dev.run(st, &mut res, &xd, &ids)?;
        let rerun = bits_equal(&y1, &y2);
        res.y.copy_from_host(stream, &vec![SENT; res.y.len()])?;
        let (gk, w, act, route, ids_d, y) = (
            &dev.gk,
            &res.w,
            &res.act,
            &mut res.route,
            &res.ids,
            &mut res.y,
        );
        let (ty, rows, input) = (st.ty, st.rows, st.input.gemm());
        let sink = dev.gpu.unlabelled_sink();
        let graph = dev.gpu.capture(|s| {
            gk.enqueue_route(s, ids_d, n_slots, route, sink)?;
            gk.enqueue_gemm(s, ty, w, rows, act, route, input, y)
        })?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let mut yg = res.y.to_host_vec(stream)?;
        yg.truncate(n_slots * st.rows);
        let replay = bits_equal(&yg, &y1);
        let nodes = graph.node_count();
        let pass = rerun && replay && nodes == 2;
        println!(
            "gemm case={} rerun_bits={rerun} graph_nodes={nodes} replay_eq_eager={replay} wall_s={:.1} {}",
            st.name,
            t0.elapsed().as_secs_f64(),
            verdict(pass)
        );
        Ok(ok && pass)
    }

    /// Synthetic super-blocks of `ty` with finite f16 scales: every other
    /// byte from the LCG, `d`/`dmin` of exponent 3..=9 (2^-12 .. 2^-5).
    fn synthetic(ty: GemmWeight, n_sb_total: usize, seed: u64) -> Vec<u8> {
        let mut g = Lcg(seed);
        let bpb = ty.block_bytes();
        let mut v = vec![0u8; n_sb_total * bpb];
        let f16 = |g: &mut Lcg| -> [u8; 2] {
            let e = 3 + g.below(7) as u16;
            let bits = (e << 10) | (g.next() as u16 & 0x3ff);
            bits.to_le_bytes()
        };
        for sb in v.chunks_exact_mut(bpb) {
            for b in sb.iter_mut() {
                *b = g.next() as u8;
            }
            let (a, c) = (f16(&mut g), f16(&mut g));
            match ty {
                GemmWeight::Q4K | GemmWeight::Q5K => {
                    sb[0..2].copy_from_slice(&a);
                    sb[2..4].copy_from_slice(&c);
                }
                GemmWeight::Q6K => sb[208..210].copy_from_slice(&a),
                GemmWeight::Q3K => sb[108..110].copy_from_slice(&a),
            }
        }
        v
    }

    /// The first layer's tensor named by `stem` (`blk.{l}.<stem>`) of type
    /// `ty`, from a table lookup `find`.
    fn first_layer<'a, F>(find: F, stem: &str, ty: GgmlType) -> Option<(usize, &'a [u8], Vec<u64>)>
    where
        F: Fn(&str) -> Option<(GgmlType, &'a [u8], Vec<u64>)>,
    {
        (0..64).find_map(|l| {
            let (t, b, dims) = find(&format!("blk.{l}.{stem}"))?;
            (t == ty).then_some((l, b, dims))
        })
    }

    /// Faults and host refusals (module doc, last paragraph).
    fn faults(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("fault") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let (k, rows, n_exp, top_k) = (2048usize, 256usize, 16usize, 4usize);
        let st = Stack {
            name: "fault".into(),
            ty: GemmWeight::Q4K,
            bytes: synthetic(GemmWeight::Q4K, n_exp * rows * k / 256, 0x0fa1).into(),
            n_exp,
            rows,
            k,
            input: Input::Shared(top_k),
        };
        let mut res = dev.resident(&st)?;
        let mut ok = true;
        gpu.clear_fault()?;

        // A NaN in token 3's column: the quantizer raises, the named error.
        let n_tok = 8;
        let ids = route_ids(Routing::Uniform, n_tok, top_k, n_exp, 17);
        let mut x = activations(k, n_tok, 991);
        x[3 * k + 77] = f32::NAN;
        let xd = DeviceBuffer::from_host(stream, &x)?;
        dev.run(&st, &mut res, &xd, &ids)?;
        let err = gpu.take_fault()?.map(|fault| GpuError::Fault {
            what: "gate_gemm",
            fault,
        });
        let nan_ok = matches!(&err, Some(GpuError::Fault { fault, .. })
            if fault.site() == Some(FaultSite::QuantColumn));
        println!(
            "gemm fault=nan_activation error=\"{}\" {}",
            err.as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string),
            verdict(nan_ok)
        );
        ok &= nan_ok;

        // Ids past the stack: the route raises and leaves the slots out.
        let mut ids = route_ids(Routing::Uniform, n_tok, top_k, n_exp, 23);
        ids[5] = n_exp as u32;
        ids[9] = u32::MAX;
        let x = activations(k, n_tok, 993);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y = dev.run(&st, &mut res, &xd, &ids)?;
        let err = gpu.take_fault()?.map(|fault| GpuError::Fault {
            what: "gate_gemm",
            fault,
        });
        let site_ok = matches!(&err, Some(GpuError::Fault { fault, .. })
            if fault.site() == Some(FaultSite::ExpertId));
        let refused_untouched = [5usize, 9].iter().all(|&s| {
            y[s * rows..(s + 1) * rows]
                .iter()
                .all(|v| v.to_bits() == SENT.to_bits())
        });
        let others_written = (0..ids.len()).filter(|s| *s != 5 && *s != 9).all(|s| {
            y[s * rows..(s + 1) * rows]
                .iter()
                .all(|v| v.to_bits() != SENT.to_bits())
        });
        let (cols, tiles) = res.route.read_back(stream)?;
        let listed: usize = tiles.iter().map(|t| t.len as usize).sum();
        let table_ok =
            listed == ids.len() - 2 && !cols[..listed].contains(&5) && !cols[..listed].contains(&9);
        let pass = site_ok && refused_untouched && others_written && table_ok;
        println!(
            "gemm fault=expert_id_out_of_range error=\"{}\" refused_slots_untouched={refused_untouched} \
             other_slots_written={others_written} table_lists={listed}_of_{} {}",
            err.as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string),
            ids.len(),
            verdict(pass)
        );
        ok &= pass;

        // Host refusals: each call must be a named error.
        let unfilled = GemmRoute::new(stream, 64, n_exp)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, 64 * rows)?;
        let mut refusals: Vec<(&str, bool)> = vec![
            ("type_q8_0", GemmWeight::from_ggml(GgmlType::Q8_0).is_err()),
            ("k_not_256", GemmAct::new(stream, 8, 1000).is_err()),
            (
                "slots_over_limit",
                GemmRoute::new(stream, GEMM_MAX_SLOTS + 1, n_exp).is_err(),
            ),
            (
                "act_over_limit",
                GemmAct::new(stream, GEMM_MAX_SLOTS + 1, k).is_err(),
            ),
            (
                "unfilled_route",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        st.ty,
                        &res.w,
                        rows,
                        &res.act,
                        &unfilled,
                        st.input.gemm(),
                        &mut y,
                    )
                    .is_err(),
            ),
            (
                "rows_not_16",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        st.ty,
                        &res.w,
                        250,
                        &res.act,
                        &res.route,
                        st.input.gemm(),
                        &mut y,
                    )
                    .is_err(),
            ),
            (
                "wrong_type_for_stack",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        GemmWeight::Q6K,
                        &res.w,
                        rows,
                        &res.act,
                        &res.route,
                        st.input.gemm(),
                        &mut y,
                    )
                    .is_err(),
            ),
            (
                "top_k_not_dividing",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        st.ty,
                        &res.w,
                        rows,
                        &res.act,
                        &res.route,
                        GemmInput::Shared { top_k: 3 },
                        &mut y,
                    )
                    .is_err(),
            ),
        ];
        let small = GemmAct::new(stream, 1, k)?;
        refusals.push((
            "act_too_small",
            dev.gk
                .enqueue_gemm(
                    stream,
                    st.ty,
                    &res.w,
                    rows,
                    &small,
                    &res.route,
                    st.input.gemm(),
                    &mut y,
                )
                .is_err(),
        ));
        let mut dense = GemmRoute::new(stream, 64, 1)?;
        refusals.push((
            "dense_over_capacity",
            dev.gk
                .enqueue_route_dense(stream, 65, &mut dense, gpu.unlabelled_sink())
                .is_err(),
        ));
        refusals.push((
            "dense_on_moe_table",
            dev.gk
                .enqueue_route_dense(stream, 8, &mut res.route, gpu.unlabelled_sink())
                .is_err(),
        ));
        let all = refusals.iter().all(|(_, r)| *r);
        println!(
            "gemm refusals {} {}",
            refusals
                .iter()
                .map(|(n, r)| format!("{n}={}", if *r { "refused" } else { "ACCEPTED" }))
                .collect::<Vec<_>>()
                .join(" "),
            verdict(all)
        );
        ok &= all;
        Ok(ok)
    }

    /// The dense special case: a one-expert Q4_K stack at 2048 × 2048 run
    /// through `enqueue_route_dense` for T ∈ TOKENS, against the reference.
    fn dense_case(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("dense") {
            return Ok(true);
        }
        let stream = dev.gpu.stream();
        let (k, rows) = (2048usize, 2048usize);
        let st = Stack {
            name: "dense_q4k_2048x2048".into(),
            ty: GemmWeight::Q4K,
            bytes: synthetic(GemmWeight::Q4K, rows * k / 256, 0xde45e).into(),
            n_exp: 1,
            rows,
            k,
            input: Input::PerSlot(1),
        };
        let n_rows = rows;
        let words = bytes_to_words(&st.bytes);
        let w = DeviceTensor::upload(stream, &words, n_rows, words.len() / n_rows)?;
        let max = TOKENS[TOKENS.len() - 1];
        let mut act = GemmAct::new(stream, max, k)?;
        let mut route = GemmRoute::new(stream, max, 1)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, max * rows)?;
        let mut ok = true;
        for &t in &TOKENS {
            let x = activations(k, t, 7001 + t as u32);
            let xd = DeviceBuffer::from_host(stream, &x)?;
            y.copy_from_host(stream, &vec![SENT; y.len()])?;
            dev.gpu
                .enqueue_quantize_gemm(&xd, t, &mut act, dev.gpu.unlabelled_sink())?;
            dev.gk
                .enqueue_route_dense(stream, t, &mut route, dev.gpu.unlabelled_sink())?;
            dev.gk.enqueue_gemm(
                stream,
                st.ty,
                &w,
                rows,
                &act,
                &route,
                GemmInput::PerSlot,
                &mut y,
            )?;
            stream.synchronize()?;
            let mut got = y.to_host_vec(stream)?;
            got.truncate(t * rows);
            let unwritten = got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
            let (codes, d8): (Vec<Vec<i8>>, Vec<Vec<f32>>) =
                (0..t).map(|c| q8_column(&x[c * k..(c + 1) * k])).unzip();
            let ids = vec![0u32; t];
            let refc = check_ref(&st, &ids, &codes, &d8, &got)?;
            let (cols, tiles) = route.read_back(stream)?;
            let table_ok = cols.iter().enumerate().all(|(i, &s)| s as usize == i)
                && tiles.len() == t.div_ceil(64);
            let pass = unwritten == 0
                && refc.decode_ok
                && refc.fail.is_none()
                && refc.bits_differ == 0
                && table_ok;
            println!(
                "gemm case={} T={t} tiles={} identity_table={table_ok} unwritten={unwritten} \
                 checked={} worst_err_over_band={:.3} contract_bits_differ={} y_fnv={:016x} {}",
                st.name,
                tiles.len(),
                refc.checked,
                refc.worst_ratio,
                refc.bits_differ,
                fnv(&got),
                verdict(pass)
            );
            if let Some(f) = &refc.fail {
                println!("gemm case={} T={t} first_failure: {f}", st.name);
            }
            if let Some(f) = &refc.bits_fail {
                println!(
                    "gemm case={} T={t} first_contract_bits_failure: {f}",
                    st.name
                );
            }
            ok &= pass;
        }
        Ok(ok)
    }

    /// Token counts of the route table case: one token, a ragged pass, the
    /// old ubatch, an odd size, the largest ubatch and one token below it.
    const ROUTE_TOKENS: [usize; 6] = [1, 17, 512, 1000, UBATCH - 1, UBATCH];

    /// The route table the host builds: the slots with an id below `n_exp`
    /// grouped by expert, experts ascending, each expert's slots ascending
    /// (the stable grouping), and each expert's run cut into tiles of at
    /// most `GEMM_BN` slots.
    fn route_ref(ids: &[u32], n_exp: usize) -> (Vec<u32>, Vec<GemmTile>) {
        let mut by_exp: Vec<Vec<u32>> = vec![Vec::new(); n_exp];
        for (s, &e) in ids.iter().enumerate() {
            if (e as usize) < n_exp {
                by_exp[e as usize].push(s as u32);
            }
        }
        let mut cols = Vec::with_capacity(ids.len());
        let mut tiles = Vec::new();
        for (e, slots) in by_exp.iter().enumerate() {
            for run in slots.chunks(GEMM_BN) {
                tiles.push(GemmTile {
                    expert: e as u32,
                    start: cols.len() as u32,
                    len: run.len() as u32,
                });
                cols.extend_from_slice(run);
            }
        }
        (cols, tiles)
    }

    /// Whether the table `route` holds is the host's for `ids`: the tiles
    /// equal, and the slot list's first entries (as many as the tiles list)
    /// equal. Prints one line under `label`.
    fn route_matches(
        dev: &Dev<'_>,
        route: &GemmRoute,
        ids: &[u32],
        n_exp: usize,
        label: &str,
    ) -> Result<bool, GateError> {
        let (want_cols, want_tiles) = route_ref(ids, n_exp);
        let (cols, tiles) = route.read_back(dev.gpu.stream())?;
        let tiles_eq = tiles == want_tiles;
        let cols_eq = cols.len() >= want_cols.len() && cols[..want_cols.len()] == want_cols[..];
        let pass = tiles_eq && cols_eq;
        println!(
            "gemm case=route_table {label} slots={} listed={} tiles={} (host {}) \
             tiles_eq_host={tiles_eq} cols_eq_host={cols_eq} {}",
            ids.len(),
            want_cols.len(),
            tiles.len(),
            want_tiles.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The route table against the host's (module doc), its faults at the
    /// largest slot count and its graph replay.
    fn route_table_case(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("route_table") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let sink = gpu.unlabelled_sink();
        let max_tok = ROUTE_TOKENS[ROUTE_TOKENS.len() - 1];
        let mut ok = true;
        gpu.clear_fault()?;
        for (stack, n_exp, top_k) in [("qwen3", N_EXPERT, N_USED), ("v41", 384usize, 6usize)] {
            let max_slots = max_tok * top_k;
            let mut route = GemmRoute::new(stream, max_slots, n_exp)?;
            let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
            for &t in &ROUTE_TOKENS {
                for (ri, &r) in ROUTINGS.iter().enumerate() {
                    let ids = route_ids(r, t, top_k, n_exp, 0x7a61e ^ (t as u64) << 8 ^ ri as u64);
                    ids_d.copy_from_host(stream, &pad(&ids, max_slots))?;
                    dev.gk
                        .enqueue_route(stream, &ids_d, ids.len(), &mut route, sink)?;
                    let label = format!("stack={stack} T={t} routing={r:?}");
                    ok &= route_matches(dev, &route, &ids, n_exp, &label)?;
                }
            }
            if gpu.take_fault()?.is_some() {
                println!("gemm case=route_table stack={stack}: a fault on valid ids FAIL");
                ok = false;
            }
            if stack != "qwen3" {
                continue;
            }
            // Ids past the stack in three chunks of the largest table: the
            // named fault, and the table without their slots.
            let mut ids = route_ids(Routing::Uniform, max_tok, top_k, n_exp, 0xbad1d);
            let bad = [0usize, max_slots / 2 + 1, max_slots - 1];
            ids[bad[0]] = n_exp as u32;
            ids[bad[1]] = u32::MAX;
            ids[bad[2]] = n_exp as u32 + 7;
            ids_d.copy_from_host(stream, &ids)?;
            dev.gk
                .enqueue_route(stream, &ids_d, ids.len(), &mut route, sink)?;
            let fault = gpu.take_fault()?;
            let site_ok = fault.is_some_and(|f| f.site() == Some(FaultSite::ExpertId));
            println!(
                "gemm case=route_table fault=ids_past_stack slots={max_slots} at {bad:?}: fault=\"{}\" \
                 (want ExpertId) {}",
                fault.map_or_else(|| "none".to_string(), |f| f.to_string()),
                verdict(site_ok)
            );
            ok &= site_ok;
            ok &= route_matches(dev, &route, &ids, n_exp, "fault=ids_past_stack")?;
            // A captured route's replay writes the eager table.
            let ids = route_ids(Routing::Uniform, max_tok, top_k, n_exp, 0x9a9f);
            ids_d.copy_from_host(stream, &ids)?;
            let n_slots = ids.len();
            let (gk, route_m, ids_ref) = (&dev.gk, &mut route, &ids_d);
            let graph = gpu.capture(|s| gk.enqueue_route(s, ids_ref, n_slots, route_m, sink))?;
            graph.launch(stream)?;
            stream.synchronize()?;
            ok &= route_matches(dev, &route, &ids, n_exp, "graph_replay")?;
        }
        // The dense table: every slot on expert 0, the identity list.
        let mut dense = GemmRoute::new(stream, max_tok, 1)?;
        for &t in &ROUTE_TOKENS {
            dev.gk.enqueue_route_dense(stream, t, &mut dense, sink)?;
            ok &= route_matches(
                dev,
                &dense,
                &vec![0u32; t],
                1,
                &format!("stack=dense T={t}"),
            )?;
        }
        if gpu.take_fault()?.is_some() {
            println!("gemm case=route_table stack=dense: a fault on the dense table FAIL");
            ok = false;
        }
        Ok(ok)
    }

    /// Column counts of the SwiGLU quantizer case.
    const SWIGLU_COLS: [usize; 4] = [1, 8, 64, 4096];

    /// The SwiGLU quantizer against the two-launch composition, the host
    /// transcription and the f64 SwiGLU (module doc), its fault and its
    /// refusals.
    fn swiglu_case(dev: &Dev<'_>) -> Result<bool, GateError> {
        if !wanted("swiglu") {
            return Ok(true);
        }
        let gpu = dev.gpu;
        let stream = gpu.stream();
        let mut ok = true;
        for k in [768usize, 2048] {
            let max = SWIGLU_COLS[SWIGLU_COLS.len() - 1];
            let mut fused = GemmAct::new(stream, max, k)?;
            let mut split = GemmAct::new(stream, max, k)?;
            let mut h = DeviceBuffer::<f32>::zeroed(stream, max * k)?;
            for &n in &SWIGLU_COLS {
                let gh = activations(k, n, 4400 + n as u32 + k as u32);
                let uh = activations(k, n, 5500 + n as u32 + k as u32);
                let (g, u) = (
                    DeviceBuffer::from_host(stream, &gh)?,
                    DeviceBuffer::from_host(stream, &uh)?,
                );
                dev.gk.enqueue_swiglu_quant(
                    stream,
                    &g,
                    &u,
                    n,
                    &mut fused,
                    gpu.unlabelled_sink(),
                )?;
                gpu.elem().enqueue_swiglu(stream, &g, &u, n * k, &mut h)?;
                gpu.enqueue_quantize_gemm(&h, n, &mut split, gpu.unlabelled_sink())?;
                stream.synchronize()?;
                let planes_same = fused.q3().to_host_vec(stream)?
                    == split.q3().to_host_vec(stream)?
                    && fused.q4().to_host_vec(stream)? == split.q4().to_host_vec(stream)?
                    && fused.q6().to_host_vec(stream)? == split.q6().to_host_vec(stream)?
                    && fused.s8().to_host_vec(stream)? == split.s8().to_host_vec(stream)?
                    && bits_equal(
                        &fused.d8().to_host_vec(stream)?,
                        &split.d8().to_host_vec(stream)?,
                    );
                let hd = h.to_host_vec(stream)?;
                let (dev_d8, dev_s8) = (
                    fused.d8().to_host_vec(stream)?,
                    fused.s8().to_host_vec(stream)?,
                );
                let nb = k / 128;
                let host_ok = (0..n).all(|c| {
                    let (codes, d8) = q8_column(&hd[c * k..(c + 1) * k]);
                    (0..nb).all(|b| dev_d8[c * nb + b].to_bits() == d8[b].to_bits())
                        && (0..k / 32).all(|j| {
                            let s: i32 = codes[32 * j..32 * j + 32]
                                .iter()
                                .map(|&v| i32::from(v))
                                .sum();
                            dev_s8[c * (k / 32) + j] == s
                        })
                });
                let mut worst = 0.0f64;
                let mut silu_ok = true;
                for i in 0..n * k {
                    let (gv, uv) = (f64::from(gh[i]), f64::from(uh[i]));
                    let want = gv / (1.0 + (-gv).exp()) * uv;
                    let tol =
                        4.0 * f64::from(f32::EPSILON) * want.abs() + f64::from(f32::MIN_POSITIVE);
                    let err = (f64::from(hd[i]) - want).abs();
                    silu_ok &= err <= tol;
                    worst = worst.max(err / tol);
                }
                let pass = planes_same && host_ok && silu_ok;
                println!(
                    "gemm case=swiglu K={k} cols={n} planes_eq_split_bits={planes_same} \
                     host_scales_and_sums={host_ok} silu_within_4ulp={silu_ok} \
                     worst_err_over_tol={worst:.3} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        // A NaN in one gate row: the named fault.
        let k = 768;
        let mut act = GemmAct::new(stream, 8, k)?;
        let mut gh = activations(k, 8, 77);
        gh[5 * k + 300] = f32::NAN;
        let uh = activations(k, 8, 78);
        let (g, u) = (
            DeviceBuffer::from_host(stream, &gh)?,
            DeviceBuffer::from_host(stream, &uh)?,
        );
        gpu.clear_fault()?;
        dev.gk
            .enqueue_swiglu_quant(stream, &g, &u, 8, &mut act, gpu.unlabelled_sink())?;
        let err = gpu.take_fault()?.map(|fault| GpuError::Fault {
            what: "gate_gemm",
            fault,
        });
        let nan_ok = matches!(&err, Some(GpuError::Fault { fault, .. })
            if fault.site() == Some(FaultSite::QuantColumn));
        println!(
            "gemm case=swiglu fault=nan_gate error=\"{}\" {}",
            err.as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string),
            verdict(nan_ok)
        );
        let short = DeviceBuffer::<f32>::zeroed(stream, k)?;
        let sink = gpu.unlabelled_sink();
        let refusals = [
            (
                "no_columns",
                dev.gk
                    .enqueue_swiglu_quant(stream, &g, &u, 0, &mut act, sink)
                    .is_err(),
            ),
            (
                "past_act",
                dev.gk
                    .enqueue_swiglu_quant(stream, &g, &u, 9, &mut act, sink)
                    .is_err(),
            ),
            (
                "short_input",
                dev.gk
                    .enqueue_swiglu_quant(stream, &short, &u, 2, &mut act, sink)
                    .is_err(),
            ),
        ];
        let all = refusals.iter().all(|(_, r)| *r);
        println!(
            "gemm case=swiglu refusals {} {}",
            refusals
                .iter()
                .map(|(n, r)| format!("{n}={}", if *r { "refused" } else { "ACCEPTED" }))
                .collect::<Vec<_>>()
                .join(" "),
            verdict(all)
        );
        Ok(ok && nan_ok && all)
    }

    /// Token counts of the ubatch router case: every edge of the logits
    /// launch's 32-token block and of its warps' 8-token tiles.
    const ROUTER_TOKENS: [usize; 12] = [1, 7, 8, 9, 15, 16, 17, 31, 33, 63, 512, UBATCH];

    /// The ubatch router against the fused router and the f64 dot (module
    /// doc), then its refusals. `w` is the router weight, [`N_EXPERT`] rows
    /// of `k` f32.
    fn router_case(dev: &Dev<'_>, w: &[f32], k: usize) -> Result<bool, GateError> {
        if !wanted("router") {
            return Ok(true);
        }
        let stream = dev.gpu.stream();
        let rk = RouterKernels::load(dev.gpu.context())?;
        let sink = dev.gpu.unlabelled_sink();
        let wd = DeviceTensor::upload(stream, w, N_EXPERT, k)?;
        let max = ROUTER_TOKENS[ROUTER_TOKENS.len() - 1];
        let mut ub = RouterOut::for_ubatch(stream, max)?;
        let mut fused = RouterOut::with_tokens(stream, MAX_TOKENS)?;
        let mut ok = true;
        for &t in &ROUTER_TOKENS {
            let x = activations(k, t, 9100 + t as u32);
            let xd = DeviceBuffer::from_host(stream, &x)?;
            rk.enqueue_ubatch(stream, &wd, &xd, t, sink, &mut ub)?;
            stream.synchronize()?;
            let (lg, pr, id, wt) = (
                ub.logits.to_host_vec(stream)?,
                ub.probs.to_host_vec(stream)?,
                ub.ids.to_host_vec(stream)?,
                ub.weights.to_host_vec(stream)?,
            );
            let mut same = true;
            for c0 in (0..t).step_by(MAX_TOKENS) {
                let m = (t - c0).min(MAX_TOKENS);
                let xc = DeviceBuffer::from_host(stream, &x[c0 * k..(c0 + m) * k])?;
                rk.enqueue_fused(stream, &wd, &xc, m, sink, &mut fused)?;
                stream.synchronize()?;
                let e = N_EXPERT;
                same &= bits_equal(
                    &fused.logits.to_host_vec(stream)?[..m * e],
                    &lg[c0 * e..(c0 + m) * e],
                );
                same &= bits_equal(
                    &fused.probs.to_host_vec(stream)?[..m * e],
                    &pr[c0 * e..(c0 + m) * e],
                );
                same &= fused.ids.to_host_vec(stream)?[..m * N_USED]
                    == id[c0 * N_USED..(c0 + m) * N_USED];
                same &= bits_equal(
                    &fused.weights.to_host_vec(stream)?[..m * N_USED],
                    &wt[c0 * N_USED..(c0 + m) * N_USED],
                );
            }
            let band_n = k / 32 + 5;
            let mut worst = 0.0f64;
            let mut in_band = true;
            for tok in 0..t {
                for e in 0..N_EXPERT {
                    let (mut dot, mut mag) = (0.0f64, 0.0f64);
                    for i in 0..k {
                        let p = f64::from(w[e * k + i]) * f64::from(x[tok * k + i]);
                        dot += p;
                        mag += p.abs();
                    }
                    let band = gamma(band_n) * mag;
                    let err = (f64::from(lg[tok * N_EXPERT + e]) - dot).abs();
                    in_band &= err <= band;
                    if band > 0.0 {
                        worst = worst.max(err / band);
                    }
                }
            }
            let pass = same && in_band;
            println!(
                "gemm case=router T={t} ubatch_eq_fused_bits={same} logits_in_band={in_band} \
                 worst_err_over_band={worst:.3} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        let x = DeviceBuffer::<f32>::zeroed(stream, max * k)?;
        let short = DeviceBuffer::<f32>::zeroed(stream, k)?;
        let wrong = DeviceTensor::upload(stream, &w[..64 * k], 64, k)?;
        let refusals = [
            (
                "tokens_past_buffers",
                rk.enqueue_ubatch(stream, &wd, &x, max + 1, sink, &mut ub)
                    .is_err(),
            ),
            (
                "no_tokens",
                rk.enqueue_ubatch(stream, &wd, &x, 0, sink, &mut ub)
                    .is_err(),
            ),
            (
                "short_input",
                rk.enqueue_ubatch(stream, &wd, &short, 2, sink, &mut ub)
                    .is_err(),
            ),
            (
                "weight_rows",
                rk.enqueue_ubatch(stream, &wrong, &x, 8, sink, &mut ub)
                    .is_err(),
            ),
            (
                "ubatch_over_limit",
                RouterOut::for_ubatch(stream, UBATCH + 1).is_err(),
            ),
            ("ubatch_empty", RouterOut::for_ubatch(stream, 0).is_err()),
        ];
        let all = refusals.iter().all(|(_, r)| *r);
        println!(
            "gemm case=router refusals {} {}",
            refusals
                .iter()
                .map(|(n, r)| format!("{n}={}", if *r { "refused" } else { "ACCEPTED" }))
                .collect::<Vec<_>>()
                .join(" "),
            verdict(all)
        );
        Ok(ok && all)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        println!("gemm device={}", gpu.device_name()?);
        let dev = Dev {
            gpu: &gpu,
            gk: GemmKernels::load(gpu.context())?,
            q6s: Q6kSelKernels::load(gpu.context(), gpu.fault_word())?,
        };
        let mut ok = true;

        // Qwen3: the model file of this profile.
        let qwen = open_model()?;
        let find_q = |name: &str| {
            let t = qwen.find(name)?;
            Some((t.ty, qwen.data(t).ok()?, t.dims.clone()))
        };
        let (lg, gate_b, gdims) = first_layer(find_q, "ffn_gate_exps.weight", GgmlType::Q4_K)
            .ok_or("no Q4_K ffn_gate_exps in the qwen3moe file")?;
        let (ld4, down4_b, d4dims) = first_layer(find_q, "ffn_down_exps.weight", GgmlType::Q4_K)
            .ok_or("no Q4_K ffn_down_exps in the qwen3moe file")?;
        let (ld6, down6_b, d6dims) = first_layer(find_q, "ffn_down_exps.weight", GgmlType::Q6_K)
            .ok_or("no Q6_K ffn_down_exps in the qwen3moe file")?;
        println!(
            "gemm qwen3 gate=blk.{lg} {gdims:?} down_q4k=blk.{ld4} {d4dims:?} down_q6k=blk.{ld6} {d6dims:?}"
        );
        let dims = |d: &[u64]| (d[0] as usize, d[1] as usize, d[2] as usize);
        let (gk_, gr, ge) = dims(&gdims);
        let (dk, dr, de) = dims(&d4dims);
        let cases_q = [
            (
                "qwen3_gate_q4k",
                GemmWeight::Q4K,
                gate_b,
                gk_,
                gr,
                ge,
                Input::Shared(8),
            ),
            (
                "qwen3_down_q4k",
                GemmWeight::Q4K,
                down4_b,
                dk,
                dr,
                de,
                Input::PerSlot(8),
            ),
            (
                "qwen3_down_q6k",
                GemmWeight::Q6K,
                down6_b,
                dk,
                dr,
                de,
                Input::PerSlot(8),
            ),
        ];
        for (i, (name, ty, b, k, rows, n_exp, input)) in cases_q.into_iter().enumerate() {
            let st = Stack {
                name: format!("{name}_real"),
                ty,
                bytes: b.into(),
                n_exp,
                rows,
                k,
                input,
            };
            ok &= run_case(&dev, &st, 0x51 + i as u64)?;
            if !wanted(&format!("{name}_synth")) {
                continue;
            }
            let syn = synthetic(ty, n_exp * rows * k / 256, 0x5e0 + i as u64);
            let st = Stack {
                name: format!("{name}_synth"),
                ty,
                bytes: syn.into(),
                n_exp,
                rows,
                k,
                input,
            };
            ok &= run_case(&dev, &st, 0x61 + i as u64)?;
        }
        if wanted("qwen3_gate_shape_q5k_synth") {
            let syn = synthetic(GemmWeight::Q5K, ge * gr * gk_ / 256, 0x5e5);
            let st = Stack {
                name: "qwen3_gate_shape_q5k_synth".into(),
                ty: GemmWeight::Q5K,
                bytes: syn.into(),
                n_exp: ge,
                rows: gr,
                k: gk_,
                input: Input::Shared(8),
            };
            ok &= run_case(&dev, &st, 0x71)?;
        }
        if wanted("q4k_ragged_rows_synth") {
            // 400 rows an expert: three full 128-row slabs and one of 16, whose
            // block has one warp over live rows and seven past them.
            let (n_exp, rows, k) = (16usize, 400usize, 2048usize);
            let syn = synthetic(GemmWeight::Q4K, n_exp * rows * k / 256, 0x5e6);
            let st = Stack {
                name: "q4k_ragged_rows_synth".into(),
                ty: GemmWeight::Q4K,
                bytes: syn.into(),
                n_exp,
                rows,
                k,
                input: Input::Shared(4),
            };
            ok &= run_case(&dev, &st, 0x72)?;
        }
        if wanted("router") {
            let t = qwen
                .find("blk.0.ffn_gate_inp.weight")
                .ok_or("no blk.0.ffn_gate_inp.weight in the qwen3moe file")?;
            if t.ty != GgmlType::F32 || t.dims.len() != 2 || t.dims[1] as usize != N_EXPERT {
                return Err(format!("blk.0.ffn_gate_inp.weight is {:?} {:?}", t.ty, t.dims).into());
            }
            let k = t.dims[0] as usize;
            let w: Vec<f32> = qwen
                .data(t)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            ok &= router_case(&dev, &w, k)?;
        }
        drop(qwen);

        if wanted("v41_gate_q3k") {
            ok &= v41_cases(&dev, dims)?;
        }
        ok &= dense_case(&dev)?;
        ok &= route_table_case(&dev)?;
        ok &= swiglu_case(&dev)?;
        ok &= faults(&dev)?;
        ok &= sel_faults(&dev)?;
        ok &= site_mask(&dev)?;

        if !ok {
            return Err(checks_failed());
        }
        if let Some(f) = case_filter() {
            println!("gemm filter --case {f}: only the matching cases ran");
        }
        println!(
            "PASSED: gate_gemm — every run inside its derived band of the f64 reference and bit for bit \
             the contract's transcription, every slot written, rerun and graph replay bit-identical, \
             faults named; the route table the host's stable grouping bit for bit up to 32,768 slots; \
             the ubatch router and the SwiGLU quantizer bit for bit their compositions"
        );
        Ok(())
    }

    /// The V4.1 cases: one routed gate stack of the file, and a synthetic
    /// stack of its shape.
    fn v41_cases(
        dev: &Dev<'_>,
        dims: impl Fn(&[u64]) -> (usize, usize, usize),
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let v41 = gguf::Split::open(gguf::v41::model())?;
        let find_v = |name: &str| {
            let (sh, t) = v41.find(name)?;
            Some((t.ty, v41.shard(sh)?.data(t).ok()?, t.dims.clone()))
        };
        let (lv, vb, vdims) = first_layer(find_v, "ffn_gate_exps.weight", GgmlType::Q3_K)
            .ok_or("no Q3_K ffn_gate_exps in the V4.1 file")?;
        println!("gemm v41 gate=blk.{lv} {vdims:?}");
        let (vk, vr, ve) = dims(&vdims);
        let st = Stack {
            name: "v41_gate_q3k_real".into(),
            ty: GemmWeight::Q3K,
            bytes: vb.into(),
            n_exp: ve,
            rows: vr,
            k: vk,
            input: Input::Shared(6),
        };
        ok &= run_case(dev, &st, 0x81)?;
        if wanted("v41_gate_q3k_synth") {
            let syn = synthetic(GemmWeight::Q3K, ve * vr * vk / 256, 0x5e3);
            let st = Stack {
                name: "v41_gate_q3k_synth".into(),
                ty: GemmWeight::Q3K,
                bytes: syn.into(),
                n_exp: ve,
                rows: vr,
                k: vk,
                input: Input::Shared(6),
            };
            ok &= run_case(dev, &st, 0x91)?;
        }
        Ok(ok)
    }
}
