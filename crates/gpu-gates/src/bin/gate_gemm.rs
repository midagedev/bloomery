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
//! The Qwen3.8 i-quant cases (`q38_*`, the card route `wide38.rs` wires):
//! synthetic IQ3_XXS and IQ4_XS stacks of the routed gate shape (64 experts
//! of 640 rows, K 2560, top-10, every slot its token's column; the blocks
//! cycled from the ref-synth dump) and an IQ3_XXS stack of 400 rows an
//! expert, each under the four routings — at T ∈ {1, 15, 16, 17, 64, 511,
//! 512, 3686} for the top-10 cases, ten slots a token capping the route
//! table at 36,864 slots — with the same checks, the T = 1 outputs against
//! the i-quant row cores (`iq3_xxs_rows`, `iq4_xs_rows`) over the repacked
//! stack as a diagnostic, a NaN `d` super-block's outputs NaN, and
//! `from_ggml`'s acceptance of the two types beside its refusal of IQ2_XS.
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
//! dense case is `dense`, the faults and refusals `fault`, the route table
//! `route_table`).
//!
//! `--bench-kernels` runs no case: it prices the GEMM's launches instead
//! (`bench` below, lead-only through `tools/ref/time-gate.sh`).
//!
//! The route table (`GemmKernels::enqueue_route`, case `route_table`) for T
//! ∈ {1, 17, 512, 1000, 4095, 4096} tokens under the four routings, on a
//! Qwen3 stack's shape (128 experts, top-8: up to 32,768 slots), a Qwen3.6
//! one's (257 experts — the shared expert joined as the last — and nine
//! slots a token: up to 36,864 slots, the route's cap, in 31 chunks) and a
//! V4.1 one's (384 experts, top-6), and the dense table (`enqueue_route_dense`)
//! at the same counts: the slot list and the tiles bit for bit the host's
//! stable grouping — experts ascending, each expert's slots ascending, runs
//! cut into tiles of at most `GEMM_BN`. Then at 32,768 slots: three ids past
//! the stack in three different chunks end as the named fault with the
//! table the host builds without them and their slots listed as refused, in
//! slot order, and a captured route's replay writes the eager table.
//!
//! The SwiGLU quantizer between gate·up and down
//! (`GemmKernels::enqueue_swiglu_quant`, case `swiglu`) at K ∈ {768, 2048}
//! for 1, 8, 64 and 4096 columns: its five planes bit for bit what
//! `ElemKernels::enqueue_swiglu` then `Gpu::enqueue_quantize_gemm` write;
//! the scales and code sums the host transcription of the quantizer gives
//! from those SwiGLU rows; every SwiGLU value within four f32 ulps (and
//! 2^-126) of the f64 `g / (1 + e^-g) · u`. A NaN in a gate row ends as the
//! named `QuantColumn` fault with its 128-value block refused: a NaN scale,
//! and every other byte of the five planes what the same input with that
//! block zeroed leaves; its refusals (no columns, more than the activation
//! holds, a short input).
//!
//! And once per case: a rerun bit-identical, the route and the GEMM
//! captured into a graph whose replay equals the eager launch. Then the
//! faults: a NaN activation ends as the named error `GpuError::Fault`, its
//! block refused as in the SwiGLU case, every slot reading its column NaN
//! and every other slot the zeroed block's run bit for bit; ids past the
//! stack end as the same error with their slots listed as refused at the
//! table's end and every output row of theirs NaN; the same fault for the
//! three `_sel` gemvs, where an id past the stack raises
//! `FaultSite::ExpertId` and a host-served slot (`hybrid::HOST`) raises
//! nothing; and the host API's refusals (type, K, slot count, row count,
//! unfilled table, token shape, activation capacity, a Q4_K stack that does
//! not start 16-byte aligned). Last, the fault's site
//! mask: two sites raised in one layer after a third in a later layer are
//! all the first layer's mask holds, on the card and in the argmax's copy.
//!
//! The 32-value family (`Gemm32Kernels`, cases `g32_*`), on synthetic stacks
//! whose decode is checked against ggml's `dequant_row`:
//! - `g32_quant`: `quantize_gemm32`'s three planes bit for bit the host
//!   quantizer (per 32 values `d = amax/127`, codes, their sum; the padding
//!   untouched) at K ∈ {96, 320, 352, 640, 2560, 6144, 10240} for 1, 9 and
//!   4096 columns; a NaN and an infinity each refuse their block (NaN
//!   scale, zero codes and sum) and raise `QuantColumn`; `swiglu_quant32`
//!   bit for bit `ElemKernels::enqueue_swiglu` then `quantize_gemm32`, its
//!   NaN refused the same way; `swiglu_quant32_sel` over a places row — the
//!   card columns the plain launch's bits, the host columns a first plain
//!   launch's, untouched, a NaN in a host column raising nothing and a
//!   place in `[n_card, HOST)` `ExpertId` with nothing written.
//! - `g32_swiglu_act`: `swiglu_act_quant32` under ik's clamped SwiGLU at
//!   limits 10 (GLM-5.3's) and 0.5, K ∈ {96, 2048, 12288} for 1, 9 and 512
//!   columns, bit for bit the CPU tier's ik-verified `qdot::swiglu_clamp`
//!   then the host quantizer, each side of the clamp met (silu(g) past the
//!   limit, u past either side, neither); under the plain rule
//!   `swiglu_quant32`'s bytes; a NaN or infinite `g` or `u` refusing its
//!   block and raising `QuantColumn` where the clamp alone would make it
//!   finite.
//! - `g32_dense`: at every K above and m ∈ {1, 8, 9, 512, 4096}, one
//!   quantization and the three entries over the dense table, 272 rows (two
//!   full slabs and one of 16): every output inside its band of the f64
//!   reference (per block the exact integer dot, at most three f32 roundings
//!   and the accumulation, `|y − ref| <= γ(K/32 + 3) · Σ_b (|D_b| + |M_b|)`)
//!   and bit for bit the contract's transcription (`gemm32.rs`); the file's
//!   Q8_0 words and the planes made from the same bytes give the same bits;
//!   a captured table and GEMM replay the eager run.
//! - `g32_routed`: routed stacks under uniform and same-expert routings — a
//!   Q8_0 file and a Q5_1 stack of 16 × 272 rows at K 640, each slot its own
//!   column; a Q8_0 plane stack of 8 experts at K 2560 whose two slots a
//!   token share its column; a Q5_1 stack of 8 × 2560 rows at K 640 (the
//!   routed down's shape) up to 4096 slots.
//! - `g38_iq4nl`: the IQ4_NL down of the Qwen3.8 card route (`gemm_iq4nl`)
//!   — a synthetic 64 × 2560-row stack at K 640 (random nibble codes, finite
//!   f16 `d`), each slot its own column, under the four routings of the
//!   K-quant cases' token counts; its decode against ggml's `dequant_row`,
//!   a NaN `d` block's outputs NaN, a rerun bit-identical, and the host
//!   API's refusal of the stack at another K.
//! - `g32_remap`: `enqueue_route_remap` over 48 experts, every third on a
//!   16-expert card stack in reversed slots and the rest `HOST`: the listed
//!   slots and tiles the host grouping of the mapped ids less the host slots,
//!   no slot refused, every host slot's rows untouched (still the sentinel)
//!   and every card slot's rows the contract's bits; `gemm_q4k` over the same
//!   table leaves the host slots untouched and gives every card slot the bits
//!   it gives over `gemm_route`'s table of the mapped ids; an identity map
//!   gives `gemm_route`'s table; an id past the map and a map value past the stack
//!   each raise `ExpertId`, their slots refused and NaN, the rest the clean
//!   run's bits.
//! - `g32_f32tile`: `enqueue_f32_tile` bit for bit `enqueue_f32_gemv` over
//!   chunks of up to eight columns at 96, 100, 128 and 512 rows, up to 4095
//!   columns, nothing written past its output.
//! - `g32_refuse`: the host API's refusals, each an error holding its own
//!   message (another check refusing first does not pass): the scratch's
//!   K and columns, the quantizer's input and columns, an unfilled table,
//!   rows not a multiple of 16, a layout or planes of another shape, an
//!   activation of fewer columns than the slots or quantized for fewer, a
//!   short `y`, a clamp limit that is not finite and a short input for the
//!   rule-taking SwiGLU, an empty map and slots past the table for the remap, and the
//!   F32 tile's K, short `x` and an `x` off a 16-byte boundary.

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
#[path = "shared/bench.rs"]
mod bench_arm;

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::router::RouterDims;
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
    use bloomery_gpu::gemm::{
        GEMM_BN, GEMM_MAX_SLOTS, GemmAct, GemmArgs, GemmInput, GemmKernels, GemmRoute, GemmTile,
        GemmWeight,
    };
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::iq::{IqFormat, IqKernels, IqRows};
    use bloomery_gpu::q6k_sel::Q6kSelKernels;
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE, Q8Act};
    use bloomery_gpu_gates::rounding::gamma;
    use bloomery_gpu_gates::{
        GateError, activations, bits_equal, bytes_to_words, checks_failed, data_dir, open_model,
        verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::iq_tables::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS, KVALUES_IQ4NL};
    use gguf::quant::{GgmlType, dequant_row, half_to_f32};
    use model::arch::models::shape::{MoeShape, rules};
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
            GemmWeight::Q4K | GemmWeight::Q5K | GemmWeight::Iq3Xxs | GemmWeight::Iq4Xs => 32,
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
                GemmWeight::Iq3Xxs => {
                    // The kernel's epilogue scales by d/4 and keeps `2s + 1`
                    // whole in the integer scale (iq.rs's rule); d/4 is exact
                    // in f64, so both the band and the contract read it.
                    dec.d.push(f16(&b[0..2]) / 4.0);
                    dec.dmin.push(0.0);
                    for c in 0..8 {
                        let aux = u32::from_le_bytes([
                            b[66 + 4 * c],
                            b[67 + 4 * c],
                            b[68 + 4 * c],
                            b[69 + 4 * c],
                        ]);
                        dec.sc.push(2 * (aux >> 28) as i32 + 1);
                        let idx = |j: usize| usize::from(b[2 + 8 * c + j]);
                        let signs = |l: usize| KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        for v in 0..32 {
                            let (l, j) = (v / 8, v % 8);
                            let (ib, m) = if j < 4 {
                                (idx(2 * l), j)
                            } else {
                                (idx(2 * l + 1), j - 4)
                            };
                            let g = i32::from(IQ3XXS_GRID[ib].to_le_bytes()[m]);
                            let sgn = if signs(l) & KMASK_IQ2XS[j] != 0 {
                                -1
                            } else {
                                1
                            };
                            dec.q.push((g * sgn) as i8);
                        }
                    }
                }
                GemmWeight::Iq4Xs => {
                    dec.d.push(f16(&b[0..2]));
                    dec.dmin.push(0.0);
                    let sh = u16::from_le_bytes([b[2], b[3]]);
                    for c in 0..8 {
                        let ls = (i32::from((b[4 + c / 2] >> (4 * (c % 2))) & 0x0f)
                            | (i32::from((sh >> (2 * c)) & 3) << 4))
                            - 32;
                        dec.sc.push(ls);
                        for j in 0..16 {
                            dec.q
                                .push(KVALUES_IQ4NL[usize::from(b[8 + 16 * c + j] & 0x0f)]);
                        }
                        for j in 0..16 {
                            dec.q
                                .push(KVALUES_IQ4NL[usize::from(b[8 + 16 * c + j] >> 4)]);
                        }
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
            GemmWeight::Iq3Xxs => GgmlType::IQ3_XXS,
            GemmWeight::Iq4Xs => GgmlType::IQ4_XS,
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
        iqk: IqKernels,
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
            self.resident_at(st, TOKENS[TOKENS.len() - 1] * st.input.top_k())
        }

        /// [`Self::resident`] for a case whose largest run is `max_slots`
        /// slots (the Qwen3.8 top-10 cases stop below TOKENS' last count).
        fn resident_at(&self, st: &Stack<'_>, max_slots: usize) -> Result<Resident, GateError> {
            let stream = self.gpu.stream();
            let n_rows = st.n_exp * st.rows;
            let mut words = bytes_to_words(&st.bytes);
            words.resize(words.len().div_ceil(n_rows) * n_rows, 0);
            let cols = words.len() / n_rows;
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
                GemmArgs {
                    ty: st.ty,
                    w: &res.w,
                    rows_per_expert: st.rows,
                    act: &res.act,
                    route: &res.route,
                    input: st.input.gemm(),
                    y: &mut res.y,
                },
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
                // The i-quant types' diagnostic is `iq_rows_diag`, not a
                // `_sel` gemv.
                (GemmWeight::Iq3Xxs, _) | (GemmWeight::Iq4Xs, _) => return Ok(None),
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
                GemmWeight::Iq3Xxs | GemmWeight::Iq4Xs => {
                    return Err("an i-quant stack has no _sel gemv".into());
                }
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
        run_case_at(dev, st, seed, &TOKENS)
    }

    /// [`run_case`] at a case's own token counts: ten slots a token (the
    /// Qwen3.8 gate) would take TOKENS' last count past the route table's
    /// 36,864-slot cap, so those cases stop at 3686 tokens.
    fn run_case_at(
        dev: &Dev<'_>,
        st: &Stack<'_>,
        seed: u64,
        tokens: &[usize],
    ) -> Result<bool, GateError> {
        if !wanted(&st.name) {
            return Ok(true);
        }
        let t0 = std::time::Instant::now();
        let stream = dev.gpu.stream();
        let mut res = dev.resident_at(st, tokens[tokens.len() - 1] * st.input.top_k())?;
        let top_k = st.input.top_k();
        let mut ok = true;
        for &n_tok in tokens {
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
            gk.enqueue_gemm(
                s,
                GemmArgs {
                    ty,
                    w,
                    rows_per_expert: rows,
                    act,
                    route,
                    input,
                    y,
                },
            )
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
                GemmWeight::Iq3Xxs | GemmWeight::Iq4Xs => sb[0..2].copy_from_slice(&a),
            }
        }
        v
    }

    /// The super-blocks of an i-quant stack: `$BLOOMERY_DATA/ref-synth/
    /// <type>.blocks` (`dequant_ref --synthetic`'s dump, `just gate-1-1`)
    /// cycled to `n_sb_total`, or seeded random super-blocks of [`synthetic`]
    /// when the dump is not there.
    fn synth_blocks(ty: GemmWeight, n_sb_total: usize, seed: u64) -> Vec<u8> {
        let name = match ty {
            GemmWeight::Iq3Xxs => "iq3_xxs",
            GemmWeight::Iq4Xs => "iq4_xs",
            _ => return synthetic(ty, n_sb_total, seed),
        };
        let path = data_dir().join("ref-synth").join(format!("{name}.blocks"));
        if let Ok(all) = std::fs::read(&path) {
            let bb = ty.block_bytes();
            if !all.is_empty() && all.len().is_multiple_of(bb) {
                let n = all.len() / bb;
                return (0..n_sb_total)
                    .flat_map(|i| all[bb * (i % n)..][..bb].iter().copied())
                    .collect();
            }
        }
        synthetic(ty, n_sb_total, seed)
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

    /// A [`GemmAct`]'s five planes, read back: q3, q4, q6, s8, d8.
    type Planes = (Vec<u64>, Vec<u32>, Vec<u32>, Vec<i32>, Vec<f32>);

    fn planes(act: &GemmAct, stream: &cuda_core::CudaStream) -> Result<Planes, GateError> {
        Ok((
            act.q3().to_host_vec(stream)?,
            act.q4().to_host_vec(stream)?,
            act.q6().to_host_vec(stream)?,
            act.s8().to_host_vec(stream)?,
            act.d8().to_host_vec(stream)?,
        ))
    }

    /// Whether `got` holds block `blk` (its index among the 128-value block
    /// scales) refused: `zero` is what the same input with that block's
    /// values zeroed quantizes to — scale 1.0, zero codes and sums — so a
    /// refused block leaves every plane equal to it but for its scale, NaN.
    fn refused_block(got: &Planes, zero: &Planes, blk: usize) -> bool {
        got.0 == zero.0
            && got.1 == zero.1
            && got.2 == zero.2
            && got.3 == zero.3
            && got.4.len() == zero.4.len()
            && got.4.iter().zip(&zero.4).enumerate().all(|(i, (g, z))| {
                if i == blk {
                    g.is_nan() && *z == 1.0
                } else {
                    g.to_bits() == z.to_bits()
                }
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

        // A NaN in token 3's column, in its first 128-value block: the
        // quantizer refuses the block and raises, the named error, and every
        // slot reading the column (its `top_k` slots) is NaN. The same input
        // with that block zeroed is the yardstick: its planes are the refused
        // ones but for the block's scale (1.0 there), and every other slot's
        // outputs are its bits.
        let n_tok = 8;
        let ids = route_ids(Routing::Uniform, n_tok, top_k, n_exp, 17);
        let mut x = activations(k, n_tok, 991);
        x[3 * k..3 * k + 128].fill(0.0);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y_zero = dev.run(&st, &mut res, &xd, &ids)?;
        let zero = planes(&res.act, stream)?;
        let zero_clean = gpu.take_fault()?.is_none();
        x[3 * k + 77] = f32::NAN;
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y = dev.run(&st, &mut res, &xd, &ids)?;
        let fault = gpu.take_fault()?;
        let err = fault.map(|fault| GpuError::fault("gate_gemm", fault));
        let site_ok = fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
        let block_refused = refused_block(&planes(&res.act, stream)?, &zero, 3 * (k / 128));
        let reads_col3 = |s: usize| s / top_k == 3;
        let col3_nan = (0..ids.len())
            .filter(|&s| reads_col3(s))
            .all(|s| y[s * rows..(s + 1) * rows].iter().all(|v| v.is_nan()));
        let others_same = (0..ids.len()).filter(|&s| !reads_col3(s)).all(|s| {
            bits_equal(
                &y[s * rows..(s + 1) * rows],
                &y_zero[s * rows..(s + 1) * rows],
            )
        });
        let nan_ok = zero_clean && site_ok && block_refused && col3_nan && others_same;
        println!(
            "gemm fault=nan_activation error=\"{}\" zeroed_block_run_clean={zero_clean} \
             block_refused(d8 NaN, codes and sums zero, other bytes as zeroed)={block_refused} \
             column_3_slots_nan={col3_nan} other_slots_bit_identical={others_same} {}",
            err.as_ref()
                .map_or_else(|| "none".to_string(), ToString::to_string),
            verdict(nan_ok)
        );
        ok &= nan_ok;

        // Ids past the stack: the route raises, leaves the slots out of the
        // tiles and lists them at the table's end, and every GEMM writes
        // their output rows NaN.
        let mut ids = route_ids(Routing::Uniform, n_tok, top_k, n_exp, 23);
        ids[5] = n_exp as u32;
        ids[9] = u32::MAX;
        let x = activations(k, n_tok, 993);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y = dev.run(&st, &mut res, &xd, &ids)?;
        let err = gpu
            .take_fault()?
            .map(|fault| GpuError::fault("gate_gemm", fault));
        let site_ok = matches!(&err, Some(GpuError::Fault { fault, .. })
            if fault.site() == Some(FaultSite::ExpertId));
        let refused_nan = [5usize, 9]
            .iter()
            .all(|&s| y[s * rows..(s + 1) * rows].iter().all(|v| v.is_nan()));
        let others_written = (0..ids.len()).filter(|s| *s != 5 && *s != 9).all(|s| {
            y[s * rows..(s + 1) * rows]
                .iter()
                .all(|v| v.to_bits() != SENT.to_bits() && !v.is_nan())
        });
        let (cols, tiles) = res.route.read_back(stream)?;
        let listed: usize = tiles.iter().map(|t| t.len as usize).sum();
        let refused = res.route.refused_back(stream)?;
        let table_ok = listed == ids.len() - 2
            && !cols[..listed].contains(&5)
            && !cols[..listed].contains(&9)
            && refused == [5, 9];
        let pass = site_ok && refused_nan && others_written && table_ok;
        println!(
            "gemm fault=expert_id_out_of_range error=\"{}\" refused_slots_nan={refused_nan} \
             other_slots_written={others_written} table_lists={listed}_of_{} refused_listed={refused:?} {}",
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
            (
                "type_iq2_xs",
                GemmWeight::from_ggml(GgmlType::IQ2_XS).is_err(),
            ),
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
                        GemmArgs {
                            ty: st.ty,
                            w: &res.w,
                            rows_per_expert: rows,
                            act: &res.act,
                            route: &unfilled,
                            input: st.input.gemm(),
                            y: &mut y,
                        },
                    )
                    .is_err(),
            ),
            (
                "rows_not_16",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty: st.ty,
                            w: &res.w,
                            rows_per_expert: 250,
                            act: &res.act,
                            route: &res.route,
                            input: st.input.gemm(),
                            y: &mut y,
                        },
                    )
                    .is_err(),
            ),
            (
                "wrong_type_for_stack",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty: GemmWeight::Q6K,
                            w: &res.w,
                            rows_per_expert: rows,
                            act: &res.act,
                            route: &res.route,
                            input: st.input.gemm(),
                            y: &mut y,
                        },
                    )
                    .is_err(),
            ),
            (
                "top_k_not_dividing",
                dev.gk
                    .enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty: st.ty,
                            w: &res.w,
                            rows_per_expert: rows,
                            act: &res.act,
                            route: &res.route,
                            input: GemmInput::Shared { top_k: 3 },
                            y: &mut y,
                        },
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
                    GemmArgs {
                        ty: st.ty,
                        w: &res.w,
                        rows_per_expert: rows,
                        act: &small,
                        route: &res.route,
                        input: st.input.gemm(),
                        y: &mut y,
                    },
                )
                .is_err(),
        ));
        // The Q4_K staging copies the stack in 16-byte pieces: a window one
        // word into an allocation is aligned for u32 and not for that. The
        // refusal comes before any launch, so the window's bytes are never
        // read; the error must name the alignment, not another shape check.
        let pad = DeviceBuffer::<u32>::zeroed(stream, res.w.buf().len() + 4)?;
        let w_off = DeviceTensor::<u32>::window_of(&pad, 4, res.w.rows(), res.w.cols())?;
        let misaligned = dev.gk.enqueue_gemm(
            stream,
            GemmArgs {
                ty: GemmWeight::Q4K,
                w: &w_off,
                rows_per_expert: rows,
                act: &res.act,
                route: &res.route,
                input: st.input.gemm(),
                y: &mut y,
            },
        );
        refusals.push((
            "q4k_stack_misaligned",
            misaligned.is_err_and(|e| e.to_string().contains("16-byte aligned")),
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
                GemmArgs {
                    ty: st.ty,
                    w: &w,
                    rows_per_expert: rows,
                    act: &act,
                    route: &route,
                    input: GemmInput::PerSlot,
                    y: &mut y,
                },
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
        // Qwen3's and Qwen3.6's routers, each at its model's shape through
        // the instance table: the stacks' experts and a token's slots.
        let router = |rule, experts, top_k| {
            RouterDims::of(MoeShape {
                rule,
                experts,
                top_k,
            })
        };
        let (q3, q35) = (
            router(rules::SOFTMAX_NORM, 128, 8)?,
            router(rules::SOFTMAX_NORM_GATED, 256, 8)?,
        );
        for (stack, n_exp, top_k) in [
            ("qwen3", q3.experts(), q3.slots()),
            ("qwen35", q35.logits(), q35.slots()),
            ("v41", 384usize, 6usize),
        ] {
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
            // named fault, the table without their slots, and their slots
            // listed as refused, in slot order.
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
            let refused = route.refused_back(stream)?;
            let listed_ok = refused.iter().map(|&s| s as usize).eq(bad);
            println!(
                "gemm case=route_table fault=ids_past_stack slots={max_slots} at {bad:?}: fault=\"{}\" \
                 (want ExpertId) refused_listed={refused:?} {}",
                fault.map_or_else(|| "none".to_string(), |f| f.to_string()),
                verdict(site_ok && listed_ok)
            );
            ok &= site_ok && listed_ok;
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
        // A NaN in one gate row, in block 2 of column 5: the named fault and
        // that block refused, against the same rows with the block's gate
        // and up values zeroed (a SwiGLU of 0).
        let k = 768;
        let mut act = GemmAct::new(stream, 8, k)?;
        let mut gh = activations(k, 8, 77);
        let mut uh = activations(k, 8, 78);
        gh[5 * k + 256..5 * k + 384].fill(0.0);
        uh[5 * k + 256..5 * k + 384].fill(0.0);
        gpu.clear_fault()?;
        let (g, u) = (
            DeviceBuffer::from_host(stream, &gh)?,
            DeviceBuffer::from_host(stream, &uh)?,
        );
        dev.gk
            .enqueue_swiglu_quant(stream, &g, &u, 8, &mut act, gpu.unlabelled_sink())?;
        let zero = planes(&act, stream)?;
        let zero_clean = gpu.take_fault()?.is_none();
        gh[5 * k + 300] = f32::NAN;
        let (g, u) = (
            DeviceBuffer::from_host(stream, &gh)?,
            DeviceBuffer::from_host(stream, &uh)?,
        );
        dev.gk
            .enqueue_swiglu_quant(stream, &g, &u, 8, &mut act, gpu.unlabelled_sink())?;
        let fault = gpu.take_fault()?;
        let err = fault.map(|fault| GpuError::fault("gate_gemm", fault));
        let block_refused = refused_block(&planes(&act, stream)?, &zero, 5 * (k / 128) + 2);
        let nan_ok = zero_clean
            && fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn))
            && block_refused;
        println!(
            "gemm case=swiglu fault=nan_gate error=\"{}\" zeroed_block_run_clean={zero_clean} \
             block_refused(d8 NaN, codes and sums zero, other bytes as zeroed)={block_refused} {}",
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

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        if std::env::args().any(|a| a == "--bench-kernels") {
            return bench::run(&gpu);
        }
        println!("gemm device={}", gpu.device_name()?);
        let dev = Dev {
            gpu: &gpu,
            gk: GemmKernels::load(gpu.context())?,
            q6s: Q6kSelKernels::load(gpu.context(), gpu.fault_word())?,
            iqk: IqKernels::load(gpu.context())?,
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
        drop(qwen);

        if wanted("v41_gate_q3k") {
            ok &= v41_cases(&dev, dims)?;
        }
        ok &= q38_cases(&dev)?;
        ok &= dense_case(&dev)?;
        ok &= route_table_case(&dev)?;
        ok &= swiglu_case(&dev)?;
        ok &= faults(&dev)?;
        ok &= sel_faults(&dev)?;
        ok &= site_mask(&dev)?;
        ok &= g32::run(&gpu, &dev.gk)?;

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
             the SwiGLU quantizer bit for bit its composition; the 32-value family (Q8_0 planes and \
             file, Q5_1) bit for bit its contract, its quantizers the host's, the remapped route's \
             host slots untouched, the wide F32 product f32_gemv's bits"
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

    /// Token counts of the Qwen3.8 i-quant cases: ten slots a token cap the
    /// route table at 36,864 slots, so the largest count is 3686 tokens.
    const Q38_TOKENS: [usize; 8] = [1, 15, 16, 17, 64, 511, 512, 3686];

    /// The q8_1 quantizer's Q4_K slot of value-order word `v` (gate_iq's
    /// helper): the i-quant row kernels read their columns in that
    /// permutation.
    fn q4_slot(v: usize) -> usize {
        256 * (v >> 8) + 32 * (v & 7) + 8 * ((v >> 6) & 3) + ((v >> 3) & 7)
    }

    /// The max relative difference of the grouped GEMM's T = 1 outputs to the
    /// i-quant row cores (`iq3_xxs_rows`, `iq4_xs_rows`) over the same host
    /// q8_1 codes: the row kernels run the whole repacked stack
    /// (`IqFormat::repack`) against the run's columns, each slot read at its
    /// expert's rows of the launch's `y`. A diagnostic, not a pin.
    fn iq_rows_diag(
        dev: &Dev<'_>,
        st: &Stack<'_>,
        res: &mut Resident,
        seed: u64,
    ) -> Result<f64, GateError> {
        let stream = dev.gpu.stream();
        let fmt = match st.ty {
            GemmWeight::Iq3Xxs => IqFormat::Iq3Xxs,
            GemmWeight::Iq4Xs => IqFormat::Iq4Xs,
            _ => return Err("iq_rows_diag: not an i-quant stack".into()),
        };
        let ids = route_ids(
            Routing::Uniform,
            1,
            st.input.top_k(),
            st.n_exp,
            seed ^ 0xd1a6,
        );
        let n_slots = ids.len();
        let x = activations(st.k, st.input.cols(n_slots), (seed as u32) ^ 0xd1a7);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y = dev.run(st, res, &xd, &ids)?;
        let rows = IqRows::upload(stream, fmt, &st.bytes, st.n_exp * st.rows, st.k)?;
        let total = st.n_exp * st.rows;
        let (codes, d8) = q8_column(&x[..st.k]);
        let col_words = 256 * (st.k / 256).div_ceil(4);
        let mut out = vec![0.0f32; n_slots * st.rows];
        let mut ybuf = DeviceBuffer::<f32>::zeroed(stream, 8 * total)?;
        for s0 in (0..n_slots).step_by(8) {
            let m = (n_slots - s0).min(8);
            // Every slot of the one token reads its one column.
            let mut q = vec![0u32; m * col_words];
            for c in 0..m {
                for v in 0..st.k / 4 {
                    q[c * col_words + q4_slot(v)] =
                        u32::from_le_bytes(std::array::from_fn(|i| codes[4 * v + i] as u8));
                }
            }
            let (qd, dd) = (
                DeviceBuffer::from_host(stream, &q)?,
                DeviceBuffer::from_host(stream, &d8.repeat(m))?,
            );
            dev.iqk
                .enqueue_rows(stream, &rows, &qd, &dd, m, &mut ybuf)?;
            let got = ybuf.to_host_vec(stream)?;
            for s in s0..s0 + m {
                let at = (s - s0) * total + ids[s] as usize * st.rows;
                out[s * st.rows..(s + 1) * st.rows].copy_from_slice(&got[at..at + st.rows]);
            }
        }
        Ok(rel_diff(&y, &out))
    }

    /// A NaN f16 `d` in the first super-block of every row of expert 0 makes
    /// every output of the slots that read it NaN — no silent failure: a
    /// T = 1 run with every slot on expert 0.
    fn iq_nan_d(dev: &Dev<'_>, st: &Stack<'_>, seed: u64) -> Result<bool, GateError> {
        let stream = dev.gpu.stream();
        let top_k = st.input.top_k();
        let bb = st.ty.block_bytes() * st.k / 256;
        let mut bytes = st.bytes.to_vec();
        for r in 0..st.rows {
            bytes[r * bb..r * bb + 2].copy_from_slice(&0x7e00u16.to_le_bytes());
        }
        let st = Stack {
            name: format!("{}_nan_d", st.name),
            ty: st.ty,
            bytes: bytes.into(),
            n_exp: st.n_exp,
            rows: st.rows,
            k: st.k,
            input: st.input,
        };
        let mut res = dev.resident_at(&st, top_k)?;
        let ids = vec![0u32; top_k];
        let x = activations(st.k, 1, (seed as u32) ^ 0xd1a9);
        let xd = DeviceBuffer::from_host(stream, &x)?;
        let y = dev.run(&st, &mut res, &xd, &ids)?;
        let nan = y.iter().all(|v| v.is_nan());
        println!(
            "gemm case={}_nan_d ty={:?} every_output_nan={nan} {}",
            st.name,
            st.ty,
            verdict(nan)
        );
        Ok(nan)
    }

    /// The Qwen3.8 i-quant cases (the card route `arch/qwen3moe/wide38.rs`
    /// wires): synthetic IQ3_XXS and IQ4_XS stacks of the routed gate shape —
    /// 64 experts of 640 rows, K 2560, top-10, every slot its token's column,
    /// the blocks cycled from the ref-synth dump — and an IQ3_XXS stack of
    /// 400 rows an expert (a partial 128-row slab). Each runs the four
    /// routings at [`Q38_TOKENS`] through [`run_case_at`], the T = 1 outputs
    /// against the i-quant row cores, a NaN `d` super-block's outputs NaN,
    /// `from_ggml`'s acceptance of the two types, and the host API's refusal
    /// of the other i-quant type's stack.
    fn q38_cases(dev: &Dev<'_>) -> Result<bool, GateError> {
        if ![
            "q38_iq3xxs_gate_synth",
            "q38_iq4xs_gate_synth",
            "q38_iq3xxs_ragged_rows_synth",
        ]
        .iter()
        .any(|&n| wanted(n))
        {
            return Ok(true);
        }
        let accept = GemmWeight::from_ggml(GgmlType::IQ3_XXS)
            .is_ok_and(|t| t == GemmWeight::Iq3Xxs)
            && GemmWeight::from_ggml(GgmlType::IQ4_XS).is_ok_and(|t| t == GemmWeight::Iq4Xs);
        println!(
            "gemm case=q38_types from_ggml(iq3_xxs, iq4_xs)={} {}",
            if accept { "accepted" } else { "REFUSED" },
            verdict(accept)
        );
        let mut ok = accept;
        for (name, ty, seed) in [
            ("q38_iq3xxs_gate_synth", GemmWeight::Iq3Xxs, 0x5e7u64),
            ("q38_iq4xs_gate_synth", GemmWeight::Iq4Xs, 0x5e9),
        ] {
            if !wanted(name) {
                continue;
            }
            let (n_exp, rows, k, top_k) = (64usize, 640usize, 2560usize, 10usize);
            let st = Stack {
                name: name.into(),
                ty,
                bytes: synth_blocks(ty, n_exp * rows * k / 256, seed).into(),
                n_exp,
                rows,
                k,
                input: Input::Shared(top_k),
            };
            let mut res = dev.resident_at(&st, Q38_TOKENS[Q38_TOKENS.len() - 1] * top_k)?;
            ok &= run_case_at(dev, &st, seed ^ 0x61, &Q38_TOKENS)?;
            let diag = iq_rows_diag(dev, &st, &mut res, seed)?;
            println!("gemm case={name} rows_diag_max_rel_diff_vs_iq_rows={diag:.3e} (diagnostic)");
            ok &= iq_nan_d(dev, &st, seed)?;
            // The host API refuses the other i-quant type on this stack.
            let stream = dev.gpu.stream();
            let mut y = DeviceBuffer::<f32>::zeroed(stream, top_k * rows)?;
            let other = if ty == GemmWeight::Iq3Xxs {
                GemmWeight::Iq4Xs
            } else {
                GemmWeight::Iq3Xxs
            };
            let refused = dev.gk.enqueue_gemm(
                stream,
                GemmArgs {
                    ty: other,
                    w: &res.w,
                    rows_per_expert: rows,
                    act: &res.act,
                    route: &res.route,
                    input: GemmInput::Shared { top_k },
                    y: &mut y,
                },
            );
            let pass = refused.is_err();
            println!(
                "gemm case={name} other_iq_type_on_stack_refused={pass} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        if wanted("q38_iq3xxs_ragged_rows_synth") {
            let (n_exp, rows, k) = (16usize, 400usize, 2560usize);
            let st = Stack {
                name: "q38_iq3xxs_ragged_rows_synth".into(),
                ty: GemmWeight::Iq3Xxs,
                bytes: synthetic(GemmWeight::Iq3Xxs, n_exp * rows * k / 256, 0x5ea).into(),
                n_exp,
                rows,
                k,
                input: Input::Shared(4),
            };
            ok &= run_case_at(dev, &st, 0x73, &TOKENS)?;
        }
        Ok(ok)
    }

    /// `--bench-kernels`: the grouped int8 GEMM's launch cost, the eager
    /// burst and the graph replay of each arm (`bench op=` rows), timed by
    /// `tools/ref/time-gate.sh` under the machine lease. Weight and activation
    /// bytes come from [`fill_pattern`](crate::bench_arm::fill_pattern): a
    /// K-quant core's f16 scale decode takes its cheapest arm on a zero, so a
    /// zeroed buffer would read faster than the launch it prices. Nothing
    /// here asserts on the results.
    ///
    /// `--bench-arm <op>` runs only the arm whose row is `bench op=<op>`
    /// (`gemm_q4k_moe_t4096`, say) and none of the others, so every `gemm_q4k`
    /// launch of the process is that arm's — the shape a counter run (`just
    /// ncu-gpu-gemm`) filters by kernel name. A name no arm carries is a named
    /// error.
    mod bench {
        use crate::bench_arm::{
            GREPS, N, ROUNDS, burst, fill_pattern, fill_pattern_f32, print_arm, replay,
        };
        use bloomery_gpu::DeviceTensor;
        use bloomery_gpu_gates::GateError;
        use cuda_core::{CudaStream, DeviceBuffer};

        pub fn run(gpu: &bloomery_gpu::Gpu) -> Result<(), GateError> {
            println!("bench n_per_burst={N} rounds={ROUNDS} graph_launches_per_round={GREPS}");
            // `--bench-arm <op>`: that one grouped-GEMM arm and nothing else.
            let only =
                {
                    let mut args = std::env::args().skip_while(|a| a != "--bench-arm");
                    match args.next() {
                        None => None,
                        Some(_) => Some(args.next().ok_or(
                            "gate_gemm: --bench-arm wants an arm name (a `bench op=` value)",
                        )?),
                    }
                };
            if let Some(arm) = only.as_deref() {
                return if gemm_arms(gpu, Some(arm))? || iq_gemm_arms(gpu, Some(arm))? {
                    Ok(())
                } else {
                    Err(format!("gate_gemm: --bench-arm {arm} names no grouped-GEMM arm").into())
                };
            }
            gemm_arms(gpu, None)?;
            iq_gemm_arms(gpu, None)?;
            Ok(())
        }

        // The grouped int8 GEMM (`bloomery_gpu::gemm`): Qwen3-30B-A3B's routed
        // gate shape — 128 experts of 768 x 2048 Q4_K, top-8, every slot reading
        // its token's column — and the dense 2048 x 2048 Q4_K case, each at T
        // tokens up to the largest ubatch (4096: 32,768 routed slots). The route
        // table is built once per T and its launch priced on its own; the GEMM
        // arm is the GEMM alone. Beside the usual row: the
        // arithmetic rate `2 * slots * rows * K` over the graph minimum, and that
        // rate against the card's int8 dense tensor peak. `only` keeps the one
        // arm of that name; the return says whether any arm ran.
        fn gemm_arms(gpu: &bloomery_gpu::Gpu, only: Option<&str>) -> Result<bool, GateError> {
            use bloomery_gpu::gemm::{
                GemmAct, GemmArgs, GemmInput, GemmKernels, GemmRoute, GemmWeight,
            };
            let stream = gpu.stream();
            let mut ran = false;
            let gk = GemmKernels::load(gpu.context())?;
            let name = gpu.device_name()?;
            let peak = int8_peak_tops(&name);
            println!(
                "bench gemm device={name:?} int8_dense_peak_tops={}",
                peak.map_or_else(|| "?".to_string(), |p| format!("{p}"))
            );
            let sink = gpu.unlabelled_sink();
            for (label, n_exp, top_k, rows, k) in [
                ("moe", 128usize, 8usize, 768usize, 2048usize),
                ("dense", 1, 1, 2048, 2048),
            ] {
                if only.is_some_and(|o| !o.starts_with(&format!("gemm_q4k_{label}_t"))) {
                    continue;
                }
                let n_rows = n_exp * rows;
                let cols = 36 * k / 256;
                let w = DeviceTensor::<u32>::upload(
                    stream,
                    &fill_pattern(n_rows * cols),
                    n_rows,
                    cols,
                )?;
                for t in [16usize, 64, 256, 512, 1024, 2048, 4096] {
                    let op = format!("gemm_q4k_{label}_t{t}");
                    if only.is_some_and(|o| o != op) {
                        continue;
                    }
                    ran = true;
                    let n_slots = t * top_k;
                    let xs = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(t * k))?;
                    let mut act = GemmAct::new(stream, t, k)?;
                    gpu.enqueue_quantize_gemm(&xs, t, &mut act, sink)?;
                    let ids = gemm_ids(t, top_k, n_exp);
                    let ids_d = DeviceBuffer::<u32>::from_host(stream, &ids)?;
                    let mut route = GemmRoute::new(stream, n_slots, n_exp)?;
                    let input = if n_exp == 1 {
                        gk.enqueue_route_dense(stream, n_slots, &mut route, sink)?;
                        GemmInput::PerSlot
                    } else {
                        let mut enq =
                            |s: &CudaStream| gk.enqueue_route(s, &ids_d, n_slots, &mut route, sink);
                        let e = burst(stream, &mut enq)?;
                        let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                        let r = replay(stream, &g)?;
                        print_arm(
                            &format!("gemm_route_t{t}"),
                            n_slots,
                            4 * n_slots as u64,
                            g.node_count(),
                            e,
                            r,
                        );
                        GemmInput::Shared { top_k }
                    };
                    let mut y = DeviceBuffer::<f32>::zeroed(stream, n_slots * rows)?;
                    let mut enq = |s: &CudaStream| {
                        gk.enqueue_gemm(
                            s,
                            GemmArgs {
                                ty: GemmWeight::Q4K,
                                w: &w,
                                rows_per_expert: rows,
                                act: &act,
                                route: &route,
                                input,
                                y: &mut y,
                            },
                        )
                    };
                    let e = burst(stream, &mut enq)?;
                    let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                    let r = replay(stream, &g)?;
                    let mut hit = vec![false; n_exp];
                    for &i in &ids {
                        hit[i as usize] = true;
                    }
                    let experts = hit.iter().filter(|&&h| h).count();
                    let n_sb = k / 256;
                    // The weights of the experts a slot picked, the activation
                    // buffers the launch reads, and the outputs it writes.
                    let bytes = (experts * rows * 144 * n_sb
                        + t * (4 * 128 * n_sb.div_ceil(2) + 4 * 8 * n_sb + 4 * 2 * n_sb)
                        + 4 * n_slots * rows) as u64;
                    print_arm(&op, rows, bytes, g.node_count(), e, r);
                    let ops = 2.0 * (n_slots * rows * k) as f64;
                    let tops = ops / r.0 / 1e6;
                    println!(
                        "bench op={op} slots={n_slots} experts={experts} ops={ops:.0} tops_graph_min={tops:.2} \
                         pct_int8_peak={}",
                        peak.map_or_else(
                            || "?".to_string(),
                            |p| format!("{:.1}", 100.0 * tops / p)
                        )
                    );
                }
            }
            Ok(ran)
        }

        /// The i-quant arms of the two GEMM families at the Qwen3.8 card
        /// route's shapes (`gate_gemm`'s q38 cases): the routed gate — 64
        /// experts of 640 × 2560, top-10's shape — for IQ3_XXS and IQ4_XS, and
        /// the routed down — 64 experts of 2560 rows × K 640, each slot its
        /// own column — for IQ4_NL. Ten slots a token pass the route table's
        /// 36,864-slot cap at 4096 tokens, so every arm runs 4096 tokens of
        /// nine slots: the table's cap, the heaviest launch of the shape.
        /// `only` keeps the one arm of that name; the return says whether any
        /// arm ran.
        fn iq_gemm_arms(gpu: &bloomery_gpu::Gpu, only: Option<&str>) -> Result<bool, GateError> {
            use bloomery_gpu::gemm::{
                Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput,
                GemmKernels, GemmRoute, GemmWeight,
            };
            let stream = gpu.stream();
            let mut ran = false;
            let gk = GemmKernels::load(gpu.context())?;
            let sink = gpu.unlabelled_sink();
            for (ty, op, words, bb) in [
                (
                    GemmWeight::Iq3Xxs,
                    "gemm_iq3xxs_moe_t4096",
                    245usize,
                    98usize,
                ),
                (GemmWeight::Iq4Xs, "gemm_iq4xs_moe_t4096", 340, 136),
            ] {
                if only.is_some_and(|o| o != op) {
                    continue;
                }
                ran = true;
                let (n_exp, top_k, rows, k, t) = (64usize, 9usize, 640usize, 2560usize, 4096usize);
                let n_rows = n_exp * rows;
                let w = DeviceTensor::<u32>::upload(
                    stream,
                    &fill_pattern(n_rows * words),
                    n_rows,
                    words,
                )?;
                let n_slots = t * top_k;
                let xs = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(t * k))?;
                let mut act = GemmAct::new(stream, t, k)?;
                gpu.enqueue_quantize_gemm(&xs, t, &mut act, sink)?;
                let ids = gemm_ids(t, top_k, n_exp);
                let ids_d = DeviceBuffer::<u32>::from_host(stream, &ids)?;
                let mut route = GemmRoute::new(stream, n_slots, n_exp)?;
                let mut rt =
                    |s: &CudaStream| gk.enqueue_route(s, &ids_d, n_slots, &mut route, sink);
                rt(stream)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, n_slots * rows)?;
                let mut enq = |s: &CudaStream| {
                    gk.enqueue_gemm(
                        s,
                        GemmArgs {
                            ty,
                            w: &w,
                            rows_per_expert: rows,
                            act: &act,
                            route: &route,
                            input: GemmInput::Shared { top_k },
                            y: &mut y,
                        },
                    )
                };
                let e = burst(stream, &mut enq)?;
                let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                let r = replay(stream, &g)?;
                let mut hit = vec![false; n_exp];
                for &i in &ids {
                    hit[i as usize] = true;
                }
                let experts = hit.iter().filter(|&&h| h).count();
                let n_sb = k / 256;
                // The weights of the experts a slot picked, the activation
                // buffers the launch reads, and the outputs it writes.
                let bytes = (experts * rows * bb * n_sb
                    + t * (4 * 128 * n_sb.div_ceil(2) + 4 * 8 * n_sb + 4 * 2 * n_sb)
                    + 4 * n_slots * rows) as u64;
                print_arm(op, rows, bytes, g.node_count(), e, r);
            }
            {
                let op = "gemm_iq4nl_down_t4096";
                if only.is_some_and(|o| o != op) {
                    return Ok(ran);
                }
                ran = true;
                let g32 = Gemm32Kernels::load(gpu.context())?;
                let (n_exp, top_k, rows, k, t) = (64usize, 9usize, 2560usize, 640usize, 4096usize);
                let n_rows = n_exp * rows;
                // 18 bytes a block, 20 blocks: 360 bytes = 90 words a row.
                let w =
                    DeviceTensor::<u32>::upload(stream, &fill_pattern(n_rows * 90), n_rows, 90)?;
                let n_slots = t * top_k;
                let xs = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(n_slots * k))?;
                let mut act = GemmAct32::new(stream, n_slots, k)?;
                g32.enqueue_quantize_gemm32(stream, &xs, n_slots, &mut act, sink)?;
                let ids = gemm_ids(t, top_k, n_exp);
                let ids_d = DeviceBuffer::<u32>::from_host(stream, &ids)?;
                let mut route = GemmRoute::new(stream, n_slots, n_exp)?;
                gk.enqueue_route(stream, &ids_d, n_slots, &mut route, sink)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, n_slots * rows)?;
                let mut enq = |s: &CudaStream| {
                    g32.enqueue_gemm32(
                        s,
                        Gemm32Args {
                            w: Gemm32Weight::Iq4NlFile(&w),
                            rows_per_expert: rows,
                            act: &act,
                            route: &route,
                            input: GemmInput::PerSlot,
                            y: &mut y,
                        },
                    )
                };
                let e = burst(stream, &mut enq)?;
                let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
                let r = replay(stream, &g)?;
                let mut hit = vec![false; n_exp];
                for &i in &ids {
                    hit[i as usize] = true;
                }
                let experts = hit.iter().filter(|&&h| h).count();
                let bytes = (experts * rows * 18 * (k / 32)
                    + n_slots * (4 * 16 * act.steps() + 4 * 2 * act.steps())
                    + 4 * n_slots * rows) as u64;
                print_arm(op, rows, bytes, g.node_count(), e, r);
            }
            Ok(ran)
        }

        /// The int8 dense tensor peak of the card named `name`, in TOPS — GA102
        /// whitepaper figures, the denominator the prefill literature report uses —
        /// or `None` for a card not listed.
        fn int8_peak_tops(name: &str) -> Option<f64> {
            if name.contains("A6000") {
                Some(309.7)
            } else if name.contains("3090") {
                Some(284.0)
            } else {
                None
            }
        }

        /// Expert ids for `t` tokens of `top_k` distinct experts out of `n_exp`,
        /// drawn from a fixed LCG: the bench's uniform routing.
        fn gemm_ids(t: usize, top_k: usize, n_exp: usize) -> Vec<u32> {
            let mut s = 0x9e37_79b9_7f4a_7c15u64;
            let mut ids = Vec::with_capacity(t * top_k);
            for _ in 0..t {
                let mut pick: Vec<u32> = Vec::with_capacity(top_k);
                while pick.len() < top_k {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let e = ((s >> 33) % n_exp as u64) as u32;
                    if !pick.contains(&e) {
                        pick.push(e);
                    }
                }
                ids.extend(pick);
            }
            ids
        }
    }

    /// The 32-value family (`Gemm32Kernels`): module doc, last part.
    mod g32 {
        use super::{
            Lcg, ROUTINGS, Routing, SENT, TOKENS, fnv, pad, route_ids, route_ref, row_checked,
            synthetic, wanted,
        };
        use bloomery_gpu::gemm::{
            Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput,
            GemmKernels, GemmRoute, GemmWeight,
        };
        use bloomery_gpu::hybrid::HOST;
        use bloomery_gpu::kquant::{Act, act::silu_ik};
        use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, LAYER_NONE};
        use bloomery_gpu_gates::gemm32::{HostAct, Planes32, dot32, host_act};
        use bloomery_gpu_gates::rounding::gamma;
        use bloomery_gpu_gates::{GateError, activations, bits_equal, bytes_to_words, verdict};
        use cuda_core::DeviceBuffer;
        use gguf::iq_tables::KVALUES_IQ4NL;
        use gguf::quant::{GgmlType, dequant_row, half_to_f32};
        use std::sync::atomic::{AtomicUsize, Ordering};

        /// K per dense case: 96 and 352 end in a one-block step (K/32 odd),
        /// 320 and 640 are Qwen3.8's HC up and shared down, not multiples of
        /// 256; 2560, 6144 and 10240 its hidden, GDN value and HC widths.
        const KS: [usize; 7] = [96, 320, 352, 640, 2560, 6144, 10240];
        /// Columns per dense case.
        const MS: [usize; 5] = [1, 8, 9, 512, 4096];
        /// Rows of a dense case: two full 128-row slabs and one of 16.
        const ROWS: usize = 272;

        /// The four layouts under test.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Lay {
            Q8Plane,
            Q8File,
            Iq4Nl,
            Q51,
        }

        /// A synthetic stack of `n_rows` rows of `k` values, decoded to what
        /// the kernels multiply: per block `d`, `m` (Q5_1) and per value the
        /// code; `bytes` the file's block stream.
        struct Wts {
            k: usize,
            bytes: Vec<u8>,
            d: Vec<f32>,
            m: Vec<f32>,
            q: Vec<i8>,
        }

        /// A finite f16 of exponent 3..=9 (2^-12 .. 2^-5), sign from `neg`.
        fn f16_bits(g: &mut Lcg, neg: bool) -> u16 {
            let e = 3 + g.below(7) as u16;
            let bits = (e << 10) | (g.next() as u16 & 0x3ff);
            if neg { bits | 0x8000 } else { bits }
        }

        /// `n_rows` rows of Q8_0 (`q51` false) or Q5_1 blocks from `seed`,
        /// decoded; the decode is checked against ggml's `dequant_row` (Q8_0
        /// bit for bit, Q5_1 within one rounding of its fma).
        fn weights(
            q51: bool,
            n_rows: usize,
            k: usize,
            seed: u64,
        ) -> Result<(Wts, bool), GateError> {
            let mut g = Lcg(seed);
            let kb = k / 32;
            let bpb = if q51 { 24 } else { 34 };
            let mut bytes = vec![0u8; n_rows * kb * bpb];
            let (mut d, mut m, mut q) = (
                Vec::with_capacity(n_rows * kb),
                Vec::with_capacity(n_rows * kb),
                Vec::with_capacity(n_rows * k),
            );
            for blk in bytes.chunks_exact_mut(bpb) {
                for b in blk.iter_mut() {
                    *b = g.next() as u8;
                }
                let db = f16_bits(&mut g, false);
                blk[0..2].copy_from_slice(&db.to_le_bytes());
                d.push(half_to_f32(db));
                if q51 {
                    let neg = g.next() & 1 == 1;
                    let mb = f16_bits(&mut g, neg);
                    blk[2..4].copy_from_slice(&mb.to_le_bytes());
                    m.push(half_to_f32(mb));
                    let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                    let qs = &blk[8..24];
                    let mut v = [0i8; 32];
                    for (j, &b) in qs.iter().enumerate() {
                        v[j] = ((b & 0x0f) | ((((qh >> j) & 1) as u8) << 4)) as i8;
                        v[16 + j] = ((b >> 4) | ((((qh >> (16 + j)) & 1) as u8) << 4)) as i8;
                    }
                    q.extend_from_slice(&v);
                } else {
                    m.push(0.0);
                    q.extend(blk[2..34].iter().map(|&b| b as i8));
                }
            }
            let ty = if q51 { GgmlType::Q5_1 } else { GgmlType::Q8_0 };
            let mut want = vec![0.0f32; k];
            let mut decode_ok = true;
            for r in 0..n_rows {
                dequant_row(ty, &bytes[r * kb * bpb..(r + 1) * kb * bpb], &mut want)?;
                for (v, &w) in want.iter().enumerate() {
                    let b = r * kb + v / 32;
                    let mine = f64::from(d[b]) * f64::from(q[r * k + v]) + f64::from(m[b]);
                    decode_ok &= if q51 {
                        (mine - f64::from(w)).abs() <= gamma(1) * mine.abs() + 1e-38
                    } else {
                        mine as f32 == w
                    };
                }
            }
            Ok((Wts { k, bytes, d, m, q }, decode_ok))
        }

        /// `n_rows` rows of IQ4_NL blocks from `seed`, decoded: per block a
        /// finite f16 `d` and 16 random code bytes, their low nibbles values
        /// `0..16` and their high nibbles `16..32`, each a `kvalues_iq4nl`
        /// entry; the decode is checked against ggml's `dequant_row` bit for
        /// bit (d·kv is one f32 product on both sides).
        fn weights_iq4nl(n_rows: usize, k: usize, seed: u64) -> Result<(Wts, bool), GateError> {
            let mut g = Lcg(seed);
            let kb = k / 32;
            let mut bytes = vec![0u8; n_rows * kb * 18];
            let (mut d, mut q) = (
                Vec::with_capacity(n_rows * kb),
                Vec::with_capacity(n_rows * k),
            );
            for blk in bytes.as_chunks_mut::<18>().0 {
                for b in blk.iter_mut() {
                    *b = g.next() as u8;
                }
                let db = f16_bits(&mut g, false);
                blk[0..2].copy_from_slice(&db.to_le_bytes());
                d.push(half_to_f32(db));
                for j in 0..16 {
                    q.push(KVALUES_IQ4NL[usize::from(blk[2 + j] & 0x0f)]);
                }
                for j in 0..16 {
                    q.push(KVALUES_IQ4NL[usize::from(blk[2 + j] >> 4)]);
                }
            }
            let mut want = vec![0.0f32; k];
            let mut decode_ok = true;
            for r in 0..n_rows {
                dequant_row(
                    GgmlType::IQ4_NL,
                    &bytes[r * kb * 18..(r + 1) * kb * 18],
                    &mut want,
                )?;
                for (v, &w) in want.iter().enumerate() {
                    let b = r * kb + v / 32;
                    let mine = f64::from(d[b]) * f64::from(q[r * k + v]);
                    decode_ok &= mine as f32 == w;
                }
            }
            Ok((
                Wts {
                    k,
                    bytes,
                    d,
                    m: vec![0.0; n_rows * kb],
                    q,
                },
                decode_ok,
            ))
        }

        /// The device planes of the first `n` columns equal the host's bit
        /// for bit, padding included.
        fn planes_equal(
            act: &GemmAct32,
            n: usize,
            want: &Planes32,
            stream: &cuda_core::CudaStream,
        ) -> Result<bool, GateError> {
            let st = act.steps();
            let q = act.q().to_host_vec(stream)?;
            let d = act.d().to_host_vec(stream)?;
            let s = act.s().to_host_vec(stream)?;
            Ok(q[..n * 16 * st] == want.0[..]
                && scales_equal(&d[..n * 2 * st], &want.1)
                && s[..n * 2 * st] == want.2[..])
        }

        /// Two scale planes equal: each pair the same bits, or both NaN (a
        /// refused block's scale, whatever its payload).
        fn scales_equal(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
        }

        /// One output of stack row `row` against column `col` three ways
        /// ([`dot32`]): the contract's bits, the f64 value and its band's
        /// magnitude.
        fn row_dot(w: &Wts, row: usize, a: &HostAct, col: usize, mins: bool) -> (f32, f64, f64) {
            let (k, kb) = (w.k, w.k / 32);
            dot32(
                &w.q[row * k..(row + 1) * k],
                &w.d[row * kb..(row + 1) * kb],
                &w.m[row * kb..(row + 1) * kb],
                a,
                col,
                mins,
            )
        }

        /// One run's reference check: outputs checked, outside the band,
        /// off the contract's bits, host slots written; the worst error over
        /// its band and the first failure.
        struct Checked {
            checked: usize,
            band_off: usize,
            bits_off: usize,
            host_bad: usize,
            worst: f64,
            first: Option<String>,
        }

        /// The reference check of one run over `slots` slots: slot `s` on
        /// weight rows `e(s)·rows ..` (`None`: a host slot, whose outputs must
        /// still read [`SENT`]), column `col(s)`.
        #[allow(clippy::too_many_arguments, reason = "a gate helper's operands, named")]
        fn check(
            w: &Wts,
            mins: bool,
            rows: usize,
            a: &HostAct,
            expert: &(dyn Fn(usize) -> Option<usize> + Sync),
            col: &(dyn Fn(usize) -> usize + Sync),
            y: &[f32],
            slots: usize,
        ) -> Checked {
            let next = AtomicUsize::new(0);
            let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(32));
            let kb = w.k / 32;
            let parts: Vec<_> = std::thread::scope(|sc| {
                let hs: Vec<_> = (0..threads)
                    .map(|_| {
                        sc.spawn(|| {
                            let (mut checked, mut band_off, mut bits_off, mut host_bad) = (0, 0, 0, 0);
                            let mut worst = 0.0f64;
                            let mut first: Option<String> = None;
                            loop {
                                let s = next.fetch_add(1, Ordering::Relaxed);
                                if s >= slots {
                                    break;
                                }
                                let out = &y[s * rows..(s + 1) * rows];
                                let Some(e) = expert(s) else {
                                    if out.iter().any(|v| v.to_bits() != SENT.to_bits()) {
                                        host_bad += 1;
                                        first.get_or_insert_with(|| format!("host slot {s} written"));
                                    }
                                    continue;
                                };
                                for r in (0..rows).filter(|&r| row_checked(slots, s, r)) {
                                    let (want, y64, mag) = row_dot(w, e * rows + r, a, col(s), mins);
                                    let got = out[r];
                                    let band = gamma(kb + 3) * mag;
                                    let err = (f64::from(got) - y64).abs();
                                    checked += 1;
                                    worst = worst.max(if band > 0.0 { err / band } else { err });
                                    let within = err <= band;
                                    if !within {
                                        band_off += 1;
                                        first.get_or_insert_with(|| {
                                            format!("slot {s} row {r}: got {got:e} f64 {y64:e} band {band:.3e}")
                                        });
                                    }
                                    if got.to_bits() != want.to_bits() {
                                        bits_off += 1;
                                        first.get_or_insert_with(|| {
                                            format!("slot {s} row {r}: got {got:e} contract {want:e}")
                                        });
                                    }
                                }
                            }
                            Checked { checked, band_off, bits_off, host_bad, worst, first }
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join()).collect()
            });
            let mut all = Checked {
                checked: 0,
                band_off: 0,
                bits_off: 0,
                host_bad: 0,
                worst: 0.0,
                first: None,
            };
            for p in parts {
                let p = p.unwrap_or(Checked {
                    checked: 0,
                    band_off: 0,
                    bits_off: 0,
                    host_bad: 1,
                    worst: 0.0,
                    first: Some("reference thread panicked".into()),
                });
                all.checked += p.checked;
                all.band_off += p.band_off;
                all.bits_off += p.bits_off;
                all.host_bad += p.host_bad;
                all.worst = all.worst.max(p.worst);
                all.first = all.first.or(p.first);
            }
            all
        }

        /// The device side: both kernel families' modules and the engine.
        struct Dev<'g> {
            gpu: &'g Gpu,
            gk: &'g GemmKernels,
            g32: Gemm32Kernels,
        }

        /// A stack resident in every layout it has: a Q8_0 stack as the
        /// planes and as the file's words, a Q5_1 stack as the file's words.
        struct Res {
            plane: Option<(DeviceTensor<u32>, DeviceTensor<u16>)>,
            file: DeviceTensor<u32>,
        }

        impl Res {
            /// `no_plane` for the layouts without q8f32 planes (Q5_1,
            /// IQ4_NL): the stack goes up as the file's words alone.
            fn new(
                dev: &Dev<'_>,
                w: &Wts,
                no_plane: bool,
                n_rows: usize,
            ) -> Result<Res, GateError> {
                let stream = dev.gpu.stream();
                let kb = w.k / 32;
                let plane = if no_plane {
                    None
                } else {
                    let qs: Vec<u32> =
                        w.q.as_chunks::<4>()
                            .0
                            .iter()
                            .map(|c| {
                                u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])
                            })
                            .collect();
                    let d: Vec<u16> = w
                        .bytes
                        .as_chunks::<34>()
                        .0
                        .iter()
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        .collect();
                    Some((
                        DeviceTensor::upload(stream, &qs, n_rows, w.k / 4)?,
                        DeviceTensor::upload(stream, &d, n_rows, kb)?,
                    ))
                };
                let mut words = bytes_to_words(&w.bytes);
                words.resize(words.len().div_ceil(n_rows) * n_rows, 0);
                let cols = words.len() / n_rows;
                Ok(Res {
                    plane,
                    file: DeviceTensor::upload(stream, &words, n_rows, cols)?,
                })
            }

            fn weight(&self, lay: Lay) -> Result<Gemm32Weight<'_>, GateError> {
                Ok(match lay {
                    Lay::Q8Plane => {
                        let (qs, d) = self.plane.as_ref().ok_or("a Q5_1 stack has no planes")?;
                        Gemm32Weight::Q8_0Plane { qs, d }
                    }
                    Lay::Q8File => Gemm32Weight::Q8_0File(&self.file),
                    Lay::Iq4Nl => Gemm32Weight::Iq4NlFile(&self.file),
                    Lay::Q51 => Gemm32Weight::Q5_1File(&self.file),
                })
            }
        }

        /// One GEMM over the table `route` last filled with `n_slots` slots,
        /// `y` set to [`SENT`] first; `y`'s first `n_slots·rows` values back.
        #[allow(clippy::too_many_arguments, reason = "a gate helper's operands, named")]
        fn gemm(
            dev: &Dev<'_>,
            res: &Res,
            lay: Lay,
            rows: usize,
            act: &GemmAct32,
            route: &GemmRoute,
            input: GemmInput,
            y: &mut DeviceBuffer<f32>,
            n_slots: usize,
        ) -> Result<Vec<f32>, GateError> {
            let stream = dev.gpu.stream();
            y.copy_from_host(stream, &vec![SENT; y.len()])?;
            dev.g32.enqueue_gemm32(
                stream,
                Gemm32Args {
                    w: res.weight(lay)?,
                    rows_per_expert: rows,
                    act,
                    route,
                    input,
                    y,
                },
            )?;
            stream.synchronize()?;
            let mut out = y.to_host_vec(stream)?;
            out.truncate(n_slots * rows);
            Ok(out)
        }

        /// The quantizer against the host transcription, its NaN refusal,
        /// and the SwiGLU quantizer against its two-launch composition.
        fn quant_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_quant") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let mut ok = true;
            gpu.clear_fault()?;
            for k in KS {
                let mut act = GemmAct32::new(stream, 4096, k)?;
                for n in [1usize, 9, 4096] {
                    let x = activations(k, n, 9100 + (k + n) as u32);
                    let xd = DeviceBuffer::from_host(stream, &x)?;
                    dev.g32.enqueue_quantize_gemm32(
                        stream,
                        &xd,
                        n,
                        &mut act,
                        gpu.unlabelled_sink(),
                    )?;
                    stream.synchronize()?;
                    let (_, want) = host_act(&x, k, n);
                    let same = planes_equal(&act, n, &want, stream)?;
                    let fault = gpu.take_fault()?;
                    let pass = same && fault.is_none();
                    println!(
                        "gemm32 case=g32_quant K={k} cols={n} planes_eq_host={same} fault={} {}",
                        fault.map_or_else(|| "none".into(), |f| f.to_string()),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            // A NaN and an infinity: each block refused, the named site raised.
            for (what, bad) in [("nan", f32::NAN), ("inf", f32::INFINITY)] {
                let (k, n) = (320usize, 9usize);
                let mut act = GemmAct32::new(stream, n, k)?;
                let mut x = activations(k, n, 9200);
                x[k + 2 * 32 + 5] = bad;
                let xd = DeviceBuffer::from_host(stream, &x)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, n, &mut act, gpu.unlabelled_sink())?;
                stream.synchronize()?;
                let (_, want) = host_act(&x, k, n);
                let same = planes_equal(&act, n, &want, stream)?;
                let d = act.d().to_host_vec(stream)?;
                let refused = d[act.steps() * 2 + 2].is_nan();
                let fault = gpu.take_fault()?;
                let site = fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
                let pass = same && refused && site;
                println!(
                    "gemm32 case=g32_quant fault={what} K={k} planes_eq_host(block refused: NaN d, \
                     zero codes and sum)={same} d_nan={refused} site_quant_column={site} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            // SwiGLU: the fused launch against `enqueue_swiglu` then the
            // quantizer, bit for bit, and its fault.
            for k in [96usize, 640] {
                let max = 4096;
                let mut fused = GemmAct32::new(stream, max, k)?;
                let mut split = GemmAct32::new(stream, max, k)?;
                let mut h = DeviceBuffer::<f32>::zeroed(stream, max * k)?;
                for (n, nan) in [(1usize, false), (9, false), (4096, false), (9, true)] {
                    let gh = activations(k, n, 9300 + (k + n) as u32);
                    let mut gh = gh.into_iter().map(|v| 4.0 * v).collect::<Vec<_>>();
                    if nan {
                        gh[3 * k + 33] = f32::NAN;
                    }
                    let uh = activations(k, n, 9400 + (k + n) as u32);
                    let (g, u) = (
                        DeviceBuffer::from_host(stream, &gh)?,
                        DeviceBuffer::from_host(stream, &uh)?,
                    );
                    dev.g32.enqueue_swiglu_quant32(
                        stream,
                        &g,
                        &u,
                        n,
                        &mut fused,
                        gpu.unlabelled_sink(),
                    )?;
                    stream.synchronize()?;
                    let f_fault = gpu.take_fault()?;
                    gpu.elem().enqueue_swiglu(stream, &g, &u, n * k, &mut h)?;
                    dev.g32.enqueue_quantize_gemm32(
                        stream,
                        &h,
                        n,
                        &mut split,
                        gpu.unlabelled_sink(),
                    )?;
                    stream.synchronize()?;
                    let s_fault = gpu.take_fault()?;
                    let st = fused.steps();
                    let same = fused.q().to_host_vec(stream)?[..n * 16 * st]
                        == split.q().to_host_vec(stream)?[..n * 16 * st]
                        && scales_equal(
                            &fused.d().to_host_vec(stream)?[..n * 2 * st],
                            &split.d().to_host_vec(stream)?[..n * 2 * st],
                        )
                        && fused.s().to_host_vec(stream)?[..n * 2 * st]
                            == split.s().to_host_vec(stream)?[..n * 2 * st];
                    let hh = h.to_host_vec(stream)?;
                    let (_, host) = host_act(&hh[..n * k], k, n);
                    let host_ok = planes_equal(&fused, n, &host, stream)?;
                    let want = nan.then(|| Fault::at(LAYER_NONE, FaultSite::QuantColumn));
                    let pass = same && host_ok && f_fault == want && s_fault == want;
                    println!(
                        "gemm32 case=g32_quant swiglu K={k} cols={n} nan={nan} planes_eq_composition={same} \
                         planes_eq_host_of_swiglu_rows={host_ok} fault={} {}",
                        f_fault.map_or_else(|| "none".into(), |f| f.to_string()),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            // The card-slots-only SwiGLU (`swiglu_quant32_sel`) over a
            // remapped route's places: a first plain launch seeds every
            // column's bytes, then the `_sel` launch over other rows — the
            // card columns the plain launch's bits over those rows, the host
            // columns the seed's, untouched; a NaN in a host column raising
            // nothing; a place in [n_card, HOST) `ExpertId` with nothing
            // written.
            {
                let (k, n, n_card) = (640usize, 33usize, 20usize);
                let place = |c: usize| {
                    if c.is_multiple_of(3) {
                        HOST
                    } else {
                        ((c * 7) % n_card) as u32
                    }
                };
                let sel: Vec<u32> = (0..n).map(place).collect();
                let sel_d = DeviceBuffer::from_host(stream, &sel)?;
                let mut plain = GemmAct32::new(stream, n, k)?;
                let mut sela = GemmAct32::new(stream, n, k)?;
                let seed = |s: u32| -> Vec<f32> {
                    activations(k, n, s).into_iter().map(|v| 4.0 * v).collect()
                };
                let (g0, u0, g1, u1) = (
                    DeviceBuffer::from_host(stream, &seed(9600))?,
                    DeviceBuffer::from_host(stream, &activations(k, n, 9601))?,
                    DeviceBuffer::from_host(stream, &seed(9602))?,
                    DeviceBuffer::from_host(stream, &activations(k, n, 9603))?,
                );
                dev.g32.enqueue_swiglu_quant32(
                    stream,
                    &g0,
                    &u0,
                    n,
                    &mut sela,
                    gpu.unlabelled_sink(),
                )?;
                stream.synchronize()?;
                let steps = sela.steps();
                let row = |a: &GemmAct32| -> Result<Planes32, GateError> {
                    Ok((
                        a.q().to_host_vec(stream)?,
                        a.d().to_host_vec(stream)?,
                        a.s().to_host_vec(stream)?,
                    ))
                };
                let before = row(&sela)?;
                dev.g32.enqueue_swiglu_quant32_sel(
                    stream,
                    &g1,
                    &u1,
                    &sel_d,
                    n_card,
                    n,
                    &mut sela,
                    gpu.unlabelled_sink(),
                )?;
                dev.g32.enqueue_swiglu_quant32(
                    stream,
                    &g1,
                    &u1,
                    n,
                    &mut plain,
                    gpu.unlabelled_sink(),
                )?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let (got, want) = (row(&sela)?, row(&plain)?);
                let eq = |a: &[u32], b: &[u32], c: usize| {
                    a[c * 16 * steps..][..16 * steps] == b[c * 16 * steps..][..16 * steps]
                };
                let mut card = 0usize;
                let mut host_ok = true;
                let mut card_ok = true;
                for c in 0..n {
                    let same = eq(&got.0, &want.0, c)
                        && got.1[c * 2 * steps..][..2 * steps]
                            == want.1[c * 2 * steps..][..2 * steps]
                        && got.2[c * 2 * steps..][..2 * steps]
                            == want.2[c * 2 * steps..][..2 * steps];
                    if sel[c] == HOST {
                        let kept = eq(&got.0, &before.0, c)
                            && got.1[c * 2 * steps..][..2 * steps]
                                == before.1[c * 2 * steps..][..2 * steps]
                            && got.2[c * 2 * steps..][..2 * steps]
                                == before.2[c * 2 * steps..][..2 * steps];
                        host_ok &= kept;
                    } else {
                        card += 1;
                        card_ok &= same;
                    }
                }
                let pass = card_ok && host_ok && fault.is_none();
                println!(
                    "gemm32 case=g32_quant swiglu_sel K={k} cols={n} card_cols_eq_plain={card_ok} \
                     ({card} of them) host_cols_untouched={host_ok} fault={} {}",
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
                // A NaN in a host column is never read: no fault, the host
                // column still the seed's, the card columns still written.
                let mut gn = seed(9604);
                let host_col = (0..n).find(|&c| sel[c] == HOST).unwrap_or(0);
                gn[host_col * k + 5] = f32::NAN;
                let gn = DeviceBuffer::from_host(stream, &gn)?;
                dev.g32.enqueue_swiglu_quant32_sel(
                    stream,
                    &gn,
                    &u1,
                    &sel_d,
                    n_card,
                    n,
                    &mut sela,
                    gpu.unlabelled_sink(),
                )?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let got = row(&sela)?;
                let host_kept = eq(&got.0, &before.0, host_col);
                let pass = fault.is_none() && host_kept;
                println!(
                    "gemm32 case=g32_quant swiglu_sel_nan_host K={k} col={host_col} \
                     host_col_untouched={host_kept} fault={} {}",
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
                // A place in [n_card, HOST): `ExpertId`, its column unwritten.
                let mut stray = sel.clone();
                stray[host_col] = n_card as u32;
                let stray_d = DeviceBuffer::from_host(stream, &stray)?;
                dev.g32.enqueue_swiglu_quant32_sel(
                    stream,
                    &g1,
                    &u1,
                    &stray_d,
                    n_card,
                    n,
                    &mut sela,
                    gpu.unlabelled_sink(),
                )?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let got = row(&sela)?;
                let unwritten = eq(&got.0, &before.0, host_col);
                let pass = fault == Some(Fault::at(LAYER_NONE, FaultSite::ExpertId)) && unwritten;
                println!(
                    "gemm32 case=g32_quant swiglu_sel_stray K={k} col={host_col} \
                     col_unwritten={unwritten} fault={} {}",
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The rule-taking SwiGLU quantizer (`swiglu_act_quant32`): under ik's
        /// clamp, bit for bit the CPU tier's ik-verified `qdot::swiglu_clamp`
        /// then the host quantizer, each side of the clamp met; under the
        /// plain rule, `swiglu_quant32`'s bytes; a non-finite input refused.
        fn swiglu_act_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_swiglu_act") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let mut ok = true;
            gpu.clear_fault()?;
            // GLM-5.3's dense lead and shared expert clamp at the file's
            // `swiglu_clamp_shexp` (`limit_shexp` of
            // `crates/model/src/arch/glm5next/hparams.rs`), its routed experts
            // at `swiglu_clamp_exp`; 10 stands for them here, 0.5 clamps most
            // values.
            const LIMITS: [f32; 2] = [10.0, 0.5];
            let max = 512;
            // Per limit: values whose silu(g) passes it, whose u passes +limit
            // or -limit, and that neither clamp touches.
            let mut sides = [[0usize; 4]; 2];
            // K 96 ends in a one-block step; 2048 and 12288 are the down's K of
            // GLM's shared expert and dense lead.
            for k in [96usize, 2048, 12288] {
                let mut act = GemmAct32::new(stream, max, k)?;
                let mut plain = GemmAct32::new(stream, max, k)?;
                for n in [1usize, 9, max] {
                    let scaled = |seed: u32| -> Vec<f32> {
                        activations(k, n, seed)
                            .into_iter()
                            .map(|v| 12.0 * v)
                            .collect()
                    };
                    let (gh, uh) = (scaled(9700 + (k + n) as u32), scaled(9800 + (k + n) as u32));
                    let (g, u) = (
                        DeviceBuffer::from_host(stream, &gh)?,
                        DeviceBuffer::from_host(stream, &uh)?,
                    );
                    for (li, limit) in LIMITS.into_iter().enumerate() {
                        dev.g32.enqueue_swiglu_act_quant32(
                            stream,
                            &g,
                            &u,
                            Act::SwigluClamp { limit },
                            n,
                            &mut act,
                            sink,
                        )?;
                        stream.synchronize()?;
                        let fault = gpu.take_fault()?;
                        let mut h = vec![0.0f32; n * k];
                        qdot::swiglu_clamp(&gh, &uh, limit, &mut h);
                        let (_, want) = host_act(&h, k, n);
                        let same = planes_equal(&act, n, &want, stream)?;
                        for (&gv, &uv) in gh.iter().zip(&uh) {
                            let over = silu_ik(gv) > limit;
                            let c = &mut sides[li];
                            c[0] += usize::from(over);
                            c[1] += usize::from(uv > limit);
                            c[2] += usize::from(uv < -limit);
                            c[3] += usize::from(!over && uv.abs() <= limit);
                        }
                        let pass = same && fault.is_none();
                        println!(
                            "gemm32 case=g32_swiglu_act clamp K={k} cols={n} limit={limit} \
                             planes_eq_host_of_ik_clamp={same} fault={} {}",
                            fault.map_or_else(|| "none".into(), |f| f.to_string()),
                            verdict(pass)
                        );
                        ok &= pass;
                    }
                    // The plain rule: the bytes of `swiglu_quant32`.
                    dev.g32.enqueue_swiglu_act_quant32(
                        stream,
                        &g,
                        &u,
                        Act::SiluMul,
                        n,
                        &mut act,
                        sink,
                    )?;
                    dev.g32
                        .enqueue_swiglu_quant32(stream, &g, &u, n, &mut plain, sink)?;
                    stream.synchronize()?;
                    let fault = gpu.take_fault()?;
                    let st = plain.steps();
                    let want: Planes32 = (
                        plain.q().to_host_vec(stream)?[..n * 16 * st].to_vec(),
                        plain.d().to_host_vec(stream)?[..n * 2 * st].to_vec(),
                        plain.s().to_host_vec(stream)?[..n * 2 * st].to_vec(),
                    );
                    let same = planes_equal(&act, n, &want, stream)?;
                    let pass = same && fault.is_none();
                    println!(
                        "gemm32 case=g32_swiglu_act silu_mul K={k} cols={n} \
                         planes_eq_swiglu_quant32={same} fault={} {}",
                        fault.map_or_else(|| "none".into(), |f| f.to_string()),
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            for (li, limit) in LIMITS.into_iter().enumerate() {
                let [sg, up, dn, free] = sides[li];
                let pass = sg > 0 && up > 0 && dn > 0 && free > 0;
                println!(
                    "gemm32 case=g32_swiglu_act sides limit={limit} silu_over={sg} up_over={up} \
                     up_under={dn} unclamped={free} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            // A non-finite input refuses its block under the clamp, where the
            // rule alone would carry a NaN or infinite `u`, or an infinite `g`,
            // to a finite value: the host side is the clamp's rows with that
            // value NaN.
            let (k, n) = (2048usize, 9usize);
            let mut act = GemmAct32::new(stream, n, k)?;
            let at = 3 * k + 33;
            let blk = 3 * act.steps() * 2 + 1;
            let limit = LIMITS[0];
            for (what, on_g, bad) in [
                ("g_nan", true, f32::NAN),
                ("g_inf", true, f32::INFINITY),
                ("u_nan", false, f32::NAN),
                ("u_inf", false, f32::NEG_INFINITY),
            ] {
                let mut gh: Vec<f32> = activations(k, n, 9900)
                    .into_iter()
                    .map(|v| 12.0 * v)
                    .collect();
                let mut uh: Vec<f32> = activations(k, n, 9901)
                    .into_iter()
                    .map(|v| 12.0 * v)
                    .collect();
                if on_g {
                    gh[at] = bad;
                } else {
                    uh[at] = bad;
                }
                let (g, u) = (
                    DeviceBuffer::from_host(stream, &gh)?,
                    DeviceBuffer::from_host(stream, &uh)?,
                );
                dev.g32.enqueue_swiglu_act_quant32(
                    stream,
                    &g,
                    &u,
                    Act::SwigluClamp { limit },
                    n,
                    &mut act,
                    sink,
                )?;
                stream.synchronize()?;
                let fault = gpu.take_fault()?;
                let mut h = vec![0.0f32; n * k];
                qdot::swiglu_clamp(&gh, &uh, limit, &mut h);
                h[at] = f32::NAN;
                let (_, want) = host_act(&h, k, n);
                let same = planes_equal(&act, n, &want, stream)?;
                let refused = act.d().to_host_vec(stream)?[blk].is_nan();
                let site = fault == Some(Fault::at(LAYER_NONE, FaultSite::QuantColumn));
                let pass = same && refused && site;
                println!(
                    "gemm32 case=g32_swiglu_act fault={what} K={k} limit={limit} \
                     planes_eq_host(block refused)={same} d_nan={refused} \
                     site_quant_column={site} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The dense cases: per K a Q8_0 stack (planes and file words from
        /// the same bytes) and a Q5_1 stack of [`ROWS`] rows, per column
        /// count one quantization and the three entries through the dense
        /// table; each against the reference, the two Q8_0 layouts equal bit
        /// for bit; then one captured route and GEMM replayed.
        fn dense_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_dense") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let max = MS[MS.len() - 1];
            let mut route = GemmRoute::new(stream, max, 1)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, max * ROWS)?;
            let mut ok = true;
            for (ki, k) in KS.into_iter().enumerate() {
                let (w8, dec8) = weights(false, ROWS, k, 0x3280 + ki as u64)?;
                let (w5, dec5) = weights(true, ROWS, k, 0x3250 + ki as u64)?;
                let r8 = Res::new(dev, &w8, false, ROWS)?;
                let r5 = Res::new(dev, &w5, true, ROWS)?;
                let mut act = GemmAct32::new(stream, max, k)?;
                for m in MS {
                    let x = activations(k, m, 9500 + (k * 7 + m) as u32);
                    let xd = DeviceBuffer::from_host(stream, &x)?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &xd, m, &mut act, sink)?;
                    dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
                    stream.synchronize()?;
                    let (ha, want) = host_act(&x, k, m);
                    let act_ok = planes_equal(&act, m, &want, stream)?;
                    let mut y8 = Vec::new();
                    for (lay, w, res, dec) in [
                        (Lay::Q8Plane, &w8, &r8, dec8),
                        (Lay::Q8File, &w8, &r8, dec8),
                        (Lay::Q51, &w5, &r5, dec5),
                    ] {
                        let got = gemm(
                            dev,
                            res,
                            lay,
                            ROWS,
                            &act,
                            &route,
                            GemmInput::PerSlot,
                            &mut y,
                            m,
                        )?;
                        let unwritten =
                            got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                        let Checked {
                            checked,
                            band_off,
                            bits_off,
                            worst,
                            first,
                            ..
                        } = check(w, lay == Lay::Q51, ROWS, &ha, &|_| Some(0), &|s| s, &got, m);
                        let layouts = match lay {
                            Lay::Q8Plane => {
                                y8 = got.clone();
                                "-".to_string()
                            }
                            Lay::Q8File => bits_equal(&got, &y8).to_string(),
                            Lay::Iq4Nl | Lay::Q51 => "-".to_string(),
                        };
                        let fault = gpu.take_fault()?;
                        let pass = act_ok
                            && dec
                            && unwritten == 0
                            && band_off == 0
                            && bits_off == 0
                            && layouts != "false"
                            && fault.is_none();
                        println!(
                            "gemm32 case=g32_dense lay={lay:?} K={k} m={m} rows={ROWS} act_eq_host={act_ok} \
                             decode_vs_ggml={dec} unwritten={unwritten} checked={checked} \
                             worst_err_over_band={worst:.3} band_off={band_off} contract_bits_differ={bits_off} \
                             file_eq_plane={layouts} y_fnv={:016x} {}",
                            fnv(&got),
                            verdict(pass)
                        );
                        if let Some(f) = first {
                            println!(
                                "gemm32 case=g32_dense lay={lay:?} K={k} m={m} first_failure: {f}"
                            );
                        }
                        ok &= pass;
                    }
                }
            }
            // Graph: the dense table and a plane GEMM captured, replayed.
            let (k, m) = (2560usize, 512usize);
            let (w8, _) = weights(false, ROWS, k, 0x32a0)?;
            let r8 = Res::new(dev, &w8, false, ROWS)?;
            let mut act = GemmAct32::new(stream, m, k)?;
            let xd = DeviceBuffer::from_host(stream, &activations(k, m, 9600))?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &xd, m, &mut act, sink)?;
            dev.gk.enqueue_route_dense(stream, m, &mut route, sink)?;
            let eager = gemm(
                dev,
                &r8,
                Lay::Q8Plane,
                ROWS,
                &act,
                &route,
                GemmInput::PerSlot,
                &mut y,
                m,
            )?;
            y.copy_from_host(stream, &vec![SENT; y.len()])?;
            let w = r8.weight(Lay::Q8Plane)?;
            let (gk, g32, route_r, act_r, y_r) = (dev.gk, &dev.g32, &mut route, &act, &mut y);
            let graph = gpu.capture(|s| {
                gk.enqueue_route_dense(s, m, route_r, sink)?;
                g32.enqueue_gemm32(
                    s,
                    Gemm32Args {
                        w,
                        rows_per_expert: ROWS,
                        act: act_r,
                        route: route_r,
                        input: GemmInput::PerSlot,
                        y: y_r,
                    },
                )
            })?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let mut replay = y.to_host_vec(stream)?;
            replay.truncate(m * ROWS);
            let (same, nodes) = (bits_equal(&replay, &eager), graph.node_count());
            let pass = same && nodes == 2;
            println!(
                "gemm32 case=g32_dense graph K={k} m={m} graph_nodes={nodes} replay_eq_eager={same} {}",
                verdict(pass)
            );
            Ok(ok && pass)
        }

        /// Routed stacks: a Q8_0 file stack and a Q5_1 one of 16 experts ×
        /// [`ROWS`] rows at K 640 (the routed down's K), each slot its own
        /// column, and one Q8_0 plane stack of 8 experts at K 2560 whose two
        /// slots a token share the token's column; plus a Q5_1 stack at the
        /// routed down's full shape, 2560 rows × K 640.
        fn routed_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_routed") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let mut ok = true;
            for (name, lay, n_exp, rows, k, top_k, shared, tokens) in [
                (
                    "q8_0f_16x272x640",
                    Lay::Q8File,
                    16usize,
                    ROWS,
                    640usize,
                    1usize,
                    false,
                    &MS[..],
                ),
                (
                    "q5_1_16x272x640",
                    Lay::Q51,
                    16,
                    ROWS,
                    640,
                    1,
                    false,
                    &MS[..],
                ),
                (
                    "q8_0p_8x272x2560_top2",
                    Lay::Q8Plane,
                    8,
                    ROWS,
                    2560,
                    2,
                    true,
                    &MS[..],
                ),
                (
                    "q5_1_8x2560x640",
                    Lay::Q51,
                    8,
                    2560,
                    640,
                    1,
                    false,
                    &[9usize, 512, 4096][..],
                ),
            ] {
                let q51 = lay == Lay::Q51;
                let n_rows = n_exp * rows;
                let (w, dec) = weights(q51, n_rows, k, 0x3300 + k as u64 + n_exp as u64)?;
                let res = Res::new(dev, &w, q51, n_rows)?;
                let max_t = tokens[tokens.len() - 1];
                let max_slots = max_t * top_k;
                let mut route = GemmRoute::new(stream, max_slots, n_exp)?;
                let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
                let mut act = GemmAct32::new(stream, max_slots, k)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
                for &t in tokens {
                    for r in [Routing::Uniform, Routing::SameTopk] {
                        let ids = route_ids(r, t, top_k, n_exp, 0x77 ^ (t as u64) << 4);
                        let n_slots = ids.len();
                        let n_cols = if shared { t } else { n_slots };
                        let x = activations(k, n_cols, 9700 + (t + k) as u32);
                        let xd = DeviceBuffer::from_host(stream, &x)?;
                        dev.g32
                            .enqueue_quantize_gemm32(stream, &xd, n_cols, &mut act, sink)?;
                        ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                        dev.gk
                            .enqueue_route(stream, &ids_d, n_slots, &mut route, sink)?;
                        let input = if shared {
                            GemmInput::Shared { top_k }
                        } else {
                            GemmInput::PerSlot
                        };
                        let got = gemm(dev, &res, lay, rows, &act, &route, input, &mut y, n_slots)?;
                        let (ha, _) = host_act(&x, k, n_cols);
                        let unwritten =
                            got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                        let Checked {
                            checked,
                            band_off,
                            bits_off,
                            worst,
                            first,
                            ..
                        } = check(
                            &w,
                            q51,
                            rows,
                            &ha,
                            &|s| Some(ids[s] as usize),
                            &|s| s / top_k,
                            &got,
                            n_slots,
                        );
                        let fault = gpu.take_fault()?;
                        let pass = dec
                            && unwritten == 0
                            && band_off == 0
                            && bits_off == 0
                            && fault.is_none();
                        println!(
                            "gemm32 case=g32_routed {name} T={t} routing={r:?} slots={n_slots} \
                             decode_vs_ggml={dec} unwritten={unwritten} checked={checked} \
                             worst_err_over_band={worst:.3} band_off={band_off} contract_bits_differ={bits_off} \
                             y_fnv={:016x} {}",
                            fnv(&got),
                            verdict(pass)
                        );
                        if let Some(f) = first {
                            println!("gemm32 case=g32_routed {name} T={t} first_failure: {f}");
                        }
                        ok &= pass;
                    }
                }
            }
            Ok(ok)
        }

        /// The IQ4_NL down of the Qwen3.8 card route (`gemm_iq4nl`): a
        /// synthetic stack of 64 experts × 2560 rows at K 640 (random nibble
        /// codes, finite f16 `d`), each slot its own column, the four routings
        /// at TOKENS; the decode against ggml's `dequant_row`, every output
        /// in its band and bit for bit the contract with `y_fnv`, a rerun
        /// bit-identical, a NaN `d` block's outputs NaN, and the host API's
        /// refusal of the stack at another K.
        fn iq4nl_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g38_iq4nl") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let (n_exp, rows, k) = (64usize, 2560usize, 640usize);
            let (w, dec) = weights_iq4nl(n_exp * rows, k, 0x3600)?;
            let res = Res::new(dev, &w, true, n_exp * rows)?;
            let max_slots = TOKENS[TOKENS.len() - 1];
            let mut route = GemmRoute::new(stream, max_slots, n_exp)?;
            let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
            let mut act = GemmAct32::new(stream, max_slots, k)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
            let mut ok = true;
            gpu.clear_fault()?;
            for &t in &TOKENS {
                for (ri, &r) in ROUTINGS.iter().enumerate() {
                    let ids = route_ids(r, t, 1, n_exp, 0x39 ^ (t as u64) << 6 ^ ri as u64);
                    let n_slots = ids.len();
                    let x = activations(k, n_slots, 9900 + (t + ri) as u32);
                    let xd = DeviceBuffer::from_host(stream, &x)?;
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &xd, n_slots, &mut act, sink)?;
                    ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                    dev.gk
                        .enqueue_route(stream, &ids_d, n_slots, &mut route, sink)?;
                    let got = gemm(
                        dev,
                        &res,
                        Lay::Iq4Nl,
                        rows,
                        &act,
                        &route,
                        GemmInput::PerSlot,
                        &mut y,
                        n_slots,
                    )?;
                    let (ha, _) = host_act(&x, k, n_slots);
                    let unwritten = got.iter().filter(|v| v.to_bits() == SENT.to_bits()).count();
                    let Checked {
                        checked,
                        band_off,
                        bits_off,
                        worst,
                        first,
                        ..
                    } = check(
                        &w,
                        false,
                        rows,
                        &ha,
                        &|s| Some(ids[s] as usize),
                        &|s| s,
                        &got,
                        n_slots,
                    );
                    let fault = gpu.take_fault()?;
                    let pass =
                        dec && unwritten == 0 && band_off == 0 && bits_off == 0 && fault.is_none();
                    println!(
                        "gemm32 case=g38_iq4nl down T={t} routing={r:?} slots={n_slots} \
                         decode_vs_ggml={dec} unwritten={unwritten} checked={checked} \
                         worst_err_over_band={worst:.3} band_off={band_off} \
                         contract_bits_differ={bits_off} y_fnv={:016x} {}",
                        fnv(&got),
                        verdict(pass)
                    );
                    if let Some(f) = first {
                        println!("gemm32 case=g38_iq4nl down T={t} first_failure: {f}");
                    }
                    ok &= pass;
                }
            }
            // A rerun at 64 tokens is bit for bit the first run.
            {
                let t = 64;
                let ids = route_ids(Routing::Uniform, t, 1, n_exp, 0x3a);
                let x = activations(k, ids.len(), 9950);
                let xd = DeviceBuffer::from_host(stream, &x)?;
                ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                dev.gk
                    .enqueue_route(stream, &ids_d, ids.len(), &mut route, sink)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, ids.len(), &mut act, sink)?;
                let a = gemm(
                    dev,
                    &res,
                    Lay::Iq4Nl,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    ids.len(),
                )?;
                let b = gemm(
                    dev,
                    &res,
                    Lay::Iq4Nl,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    ids.len(),
                )?;
                let same = bits_equal(&a, &b);
                println!("gemm32 case=g38_iq4nl rerun_bits={same} {}", verdict(same));
                ok &= same;
            }
            // A NaN f16 `d` in block 0 of expert 0's row 0: the outputs of
            // the slots on that expert's row 0 are NaN, every other output
            // finite — no silent failure.
            {
                let mut bytes = w.bytes.clone();
                bytes[0..2].copy_from_slice(&0x7e00u16.to_le_bytes());
                let wn = Wts {
                    k,
                    bytes,
                    d: w.d.clone(),
                    m: w.m.clone(),
                    q: w.q.clone(),
                };
                let resn = Res::new(dev, &wn, true, n_exp * rows)?;
                let ids = [0u32; 8];
                let x = activations(k, ids.len(), 9960);
                let xd = DeviceBuffer::from_host(stream, &x)?;
                ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                dev.gk
                    .enqueue_route(stream, &ids_d, ids.len(), &mut route, sink)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, ids.len(), &mut act, sink)?;
                let got = gemm(
                    dev,
                    &resn,
                    Lay::Iq4Nl,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    ids.len(),
                )?;
                let row0_nan = (0..ids.len()).all(|s| got[s * rows].is_nan());
                let rest_finite = (0..ids.len()).all(|s| {
                    got[s * rows + 1..(s + 1) * rows]
                        .iter()
                        .all(|v| v.is_finite())
                });
                let pass = row0_nan && rest_finite;
                println!(
                    "gemm32 case=g38_iq4nl fault=nan_d row0_nan={row0_nan} other_rows_finite={rest_finite} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            // The host API refuses the stack at another K's words.
            {
                let mut other = GemmAct32::new(stream, 8, 672)?;
                let xd = DeviceBuffer::from_host(stream, &activations(672, 8, 9970))?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, 8, &mut other, sink)?;
                let wgt = res.weight(Lay::Iq4Nl)?;
                let mut y8 = DeviceBuffer::<f32>::zeroed(stream, 8 * rows)?;
                let err = dev.g32.enqueue_gemm32(
                    stream,
                    Gemm32Args {
                        w: wgt,
                        rows_per_expert: rows,
                        act: &other,
                        route: &route,
                        input: GemmInput::PerSlot,
                        y: &mut y8,
                    },
                );
                let pass = err.is_err_and(|e| e.to_string().contains("IQ4_NL rows at K = 672"));
                println!(
                    "gemm32 case=g38_iq4nl refusal_other_k_refused={pass} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The remapped route: a map of 48 experts, every third on the card
        /// (slot `15 − e/3`, so the map reverses them) and the rest [`HOST`],
        /// under a Q5_1 stack of the 16 card experts. Per token count: the
        /// table's listed slots and tiles the host's grouping of the mapped
        /// ids with the host slots taken out, nothing refused, every host
        /// slot's rows still [`SENT`] and every card slot's rows the
        /// contract's bits. An identity map gives `gemm_route`'s table bit
        /// for bit. Then the faults: an id past the map and a map value past
        /// the stack each raise `ExpertId`, their slots listed as refused and
        /// NaN, every other output as the clean run's.
        fn remap_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_remap") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let (n_map, n_card, rows, k, top_k) = (48usize, 16usize, ROWS, 640usize, 4usize);
            let map: Vec<u32> = (0..n_map)
                .map(|e| {
                    if e % 3 == 0 {
                        (n_card - 1 - e / 3) as u32
                    } else {
                        HOST
                    }
                })
                .collect();
            let (w, dec) = weights(true, n_card * rows, k, 0x3400)?;
            let res = Res::new(dev, &w, true, n_card * rows)?;
            let tokens = [1usize, 17, 512, 2048];
            let max_slots = tokens[tokens.len() - 1] * top_k;
            let mut route = GemmRoute::new(stream, max_slots, n_card)?;
            let mut ids_d = DeviceBuffer::<u32>::zeroed(stream, max_slots)?;
            let map_d = DeviceBuffer::from_host(stream, &map)?;
            let mut act = GemmAct32::new(stream, max_slots, k)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, max_slots * rows)?;
            let mut ok = true;
            gpu.clear_fault()?;
            let mut last = None;
            for t in tokens {
                let ids = route_ids(Routing::Uniform, t, top_k, n_map, 0x88 ^ t as u64);
                let n_slots = ids.len();
                let x = activations(k, n_slots, 9800 + t as u32);
                let xd = DeviceBuffer::from_host(stream, &x)?;
                dev.g32
                    .enqueue_quantize_gemm32(stream, &xd, n_slots, &mut act, sink)?;
                ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                dev.g32
                    .enqueue_route_remap(stream, &ids_d, &map_d, n_slots, &mut route, sink)?;
                let got = gemm(
                    dev,
                    &res,
                    Lay::Q51,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    n_slots,
                )?;
                let mapped: Vec<u32> = ids.iter().map(|&e| map[e as usize]).collect();
                let (want_cols, want_tiles) = route_ref(&mapped, n_card);
                let (cols, tiles) = route.read_back(stream)?;
                let refused = route.refused_back(stream)?;
                let table_ok = tiles == want_tiles
                    && cols[..want_cols.len()] == want_cols[..]
                    && refused.is_empty();
                let (ha, _) = host_act(&x, k, n_slots);
                let host_slots = mapped.iter().filter(|&&v| v == HOST).count();
                let Checked {
                    checked,
                    band_off,
                    bits_off,
                    host_bad,
                    worst,
                    first,
                } = check(
                    &w,
                    true,
                    rows,
                    &ha,
                    &|s| (mapped[s] != HOST).then_some(mapped[s] as usize),
                    &|s| s,
                    &got,
                    n_slots,
                );
                let fault = gpu.take_fault()?;
                let pass = dec
                    && table_ok
                    && host_bad == 0
                    && band_off == 0
                    && bits_off == 0
                    && fault.is_none();
                println!(
                    "gemm32 case=g32_remap T={t} slots={n_slots} host_slots={host_slots} listed={} tiles={} \
                     table_eq_host={table_ok} host_untouched={} checked={checked} worst_err_over_band={worst:.3} \
                     band_off={band_off} contract_bits_differ={bits_off} {}",
                    want_cols.len(),
                    tiles.len(),
                    host_bad == 0,
                    verdict(pass)
                );
                if let Some(f) = first {
                    println!("gemm32 case=g32_remap T={t} first_failure: {f}");
                }
                ok &= pass;
                last = Some((ids, got));
            }
            let (ids, clean) = last.ok_or("no remap run")?;

            // The K-quant family over the same remapped table: a Q4_K stack of
            // the 16 card experts (the routed gate·up's type). Host slots keep
            // the sentinel; every card slot is bit for bit the same GEMM over
            // `gemm_route`'s table of the mapped ids, each host id replaced by
            // card slot 0 — an output's bits do not depend on its tile.
            {
                let (kq, rq) = (2048usize, 256usize);
                let n_slots = ids.len();
                let bytes = synthetic(GemmWeight::Q4K, n_card * rq * kq / 256, 0x3410);
                let words = bytes_to_words(&bytes);
                let wq =
                    DeviceTensor::upload(stream, &words, n_card * rq, words.len() / (n_card * rq))?;
                let mut aq = GemmAct::new(stream, n_slots, kq)?;
                let xq = DeviceBuffer::from_host(stream, &activations(kq, n_slots, 9870))?;
                gpu.enqueue_quantize_gemm(&xq, n_slots, &mut aq, sink)?;
                let mut yq = DeviceBuffer::<f32>::zeroed(stream, n_slots * rq)?;
                let mut run_q4k = |route: &GemmRoute| -> Result<Vec<f32>, GateError> {
                    yq.copy_from_host(stream, &vec![SENT; yq.len()])?;
                    dev.gk.enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty: GemmWeight::Q4K,
                            w: &wq,
                            rows_per_expert: rq,
                            act: &aq,
                            route,
                            input: GemmInput::PerSlot,
                            y: &mut yq,
                        },
                    )?;
                    stream.synchronize()?;
                    Ok(yq.to_host_vec(stream)?)
                };
                ids_d.copy_from_host(stream, &pad(&ids, ids_d.len()))?;
                dev.g32
                    .enqueue_route_remap(stream, &ids_d, &map_d, n_slots, &mut route, sink)?;
                let y_remap = run_q4k(&route)?;
                let mapped: Vec<u32> = ids.iter().map(|&e| map[e as usize]).collect();
                let plain_ids: Vec<u32> = mapped
                    .iter()
                    .map(|&v| if v == HOST { 0 } else { v })
                    .collect();
                let mut plain = GemmRoute::new(stream, max_slots, n_card)?;
                ids_d.copy_from_host(stream, &pad(&plain_ids, ids_d.len()))?;
                dev.gk
                    .enqueue_route(stream, &ids_d, n_slots, &mut plain, sink)?;
                let y_plain = run_q4k(&plain)?;
                let (mut host_ok, mut card_ok) = (true, true);
                for s in 0..n_slots {
                    let (a, b) = (
                        &y_remap[s * rq..(s + 1) * rq],
                        &y_plain[s * rq..(s + 1) * rq],
                    );
                    if mapped[s] == HOST {
                        host_ok &= a.iter().all(|v| v.to_bits() == SENT.to_bits());
                    } else {
                        card_ok &=
                            bits_equal(a, b) && a.iter().all(|v| v.to_bits() != SENT.to_bits());
                    }
                }
                let fault = gpu.take_fault()?;
                let pass = host_ok && card_ok && fault.is_none();
                println!(
                    "gemm32 case=g32_remap gemm_q4k slots={n_slots} host_untouched={host_ok} \
                     card_eq_gemm_route_table={card_ok} fault={} {}",
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
            }
            let n_slots = ids.len();
            let x = activations(k, n_slots, 9800 + 2048);
            let xd = DeviceBuffer::from_host(stream, &x)?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &xd, n_slots, &mut act, sink)?;

            // An identity map is `gemm_route`'s table.
            {
                let ident: Vec<u32> = (0..n_card as u32).collect();
                let ident_d = DeviceBuffer::from_host(stream, &ident)?;
                let ids16 = route_ids(Routing::Uniform, 1000, top_k, n_card, 0x99);
                ids_d.copy_from_host(stream, &pad(&ids16, ids_d.len()))?;
                dev.g32.enqueue_route_remap(
                    stream,
                    &ids_d,
                    &ident_d,
                    ids16.len(),
                    &mut route,
                    sink,
                )?;
                let a = route.read_back(stream)?;
                let mut plain = GemmRoute::new(stream, max_slots, n_card)?;
                dev.gk
                    .enqueue_route(stream, &ids_d, ids16.len(), &mut plain, sink)?;
                let b = plain.read_back(stream)?;
                let pass = a == b && gpu.take_fault()?.is_none();
                println!(
                    "gemm32 case=g32_remap identity_map slots={} table_eq_gemm_route={} {}",
                    ids16.len(),
                    a == b,
                    verdict(pass)
                );
                ok &= pass;
            }

            // Faults: slot 5's id past the map; then every slot of card expert
            // 3 (map value 16, past the stack).
            let want = Some(Fault::at(LAYER_NONE, FaultSite::ExpertId));
            for case in ["id_past_map", "value_past_stack"] {
                let (ids_f, map_f) = if case == "id_past_map" {
                    let mut v = ids.clone();
                    v[5] = n_map as u32 + 2;
                    (v, map.clone())
                } else {
                    let mut mp = map.clone();
                    mp[3] = n_card as u32;
                    (ids.clone(), mp)
                };
                let bad: Vec<usize> = (0..n_slots)
                    .filter(|&s| {
                        ids_f[s] as usize >= n_map || map_f[ids_f[s] as usize] == n_card as u32
                    })
                    .collect();
                let map_fd = DeviceBuffer::from_host(stream, &map_f)?;
                ids_d.copy_from_host(stream, &pad(&ids_f, ids_d.len()))?;
                dev.g32
                    .enqueue_route_remap(stream, &ids_d, &map_fd, n_slots, &mut route, sink)?;
                let got = gemm(
                    dev,
                    &res,
                    Lay::Q51,
                    rows,
                    &act,
                    &route,
                    GemmInput::PerSlot,
                    &mut y,
                    n_slots,
                )?;
                let fault = gpu.take_fault()?;
                let refused = route.refused_back(stream)?;
                let listed_ok = refused.iter().map(|&s| s as usize).collect::<Vec<_>>() == bad;
                let nan_ok = bad
                    .iter()
                    .all(|&s| got[s * rows..(s + 1) * rows].iter().all(|v| v.is_nan()));
                let others = (0..n_slots).filter(|s| !bad.contains(s)).all(|s| {
                    bits_equal(
                        &got[s * rows..(s + 1) * rows],
                        &clean[s * rows..(s + 1) * rows],
                    )
                });
                let pass = fault == want && listed_ok && nan_ok && others && !bad.is_empty();
                println!(
                    "gemm32 case=g32_remap fault={case} bad_slots={} fault=\"{}\" refused_listed={listed_ok} \
                     refused_nan={nan_ok} others_eq_clean={others} {}",
                    bad.len(),
                    fault.map_or_else(|| "none".into(), |f| f.to_string()),
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        }

        /// The wide F32 product against `f32_gemv` launches of up to eight
        /// columns, bit for bit, at the Qwen3.8 row counts (96, 128, 512) and
        /// one that is not a multiple of 32 (100); a column count one short of
        /// a ubatch (4095) ragged at both edges.
        fn f32tile_case(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_f32tile") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let k = 2560usize;
            let mut ok = true;
            for (rows, ns) in [
                (96usize, &[1usize, 8, 9, 512, 4095][..]),
                (100, &[1, 9, 512][..]),
                (128, &[1, 8, 9, 512, 4095][..]),
                (512, &[1, 9, 512][..]),
            ] {
                let wh = activations(k, rows, 9900 + rows as u32);
                let w = DeviceTensor::upload(stream, &wh, rows, k)?;
                for &n in ns {
                    let xh = activations(k, n, 9950 + n as u32);
                    let xd = DeviceBuffer::from_host(stream, &xh)?;
                    let mut y = DeviceBuffer::from_host(stream, &vec![SENT; rows * n + 64])?;
                    dev.g32.enqueue_f32_tile(stream, &w, &xd, n, &mut y)?;
                    stream.synchronize()?;
                    let got = y.to_host_vec(stream)?;
                    let tail_ok = got[rows * n..]
                        .iter()
                        .all(|v| v.to_bits() == SENT.to_bits());
                    let mut differ = 0usize;
                    let mut yg = DeviceBuffer::<f32>::zeroed(stream, rows * 8)?;
                    for c0 in (0..n).step_by(8) {
                        let m = (n - c0).min(8);
                        let xc = DeviceBuffer::from_host(stream, &xh[c0 * k..(c0 + m) * k])?;
                        gpu.q8f32().enqueue_f32_gemv(stream, &w, &xc, m, &mut yg)?;
                        let g = yg.to_host_vec(stream)?;
                        for c in 0..m {
                            for r in 0..rows {
                                if got[(c0 + c) * rows + r].to_bits() != g[r * m + c].to_bits() {
                                    differ += 1;
                                }
                            }
                        }
                    }
                    let pass = differ == 0 && tail_ok;
                    println!(
                        "gemm32 case=g32_f32tile rows={rows} K={k} n={n} bits_differ_vs_f32_gemv={differ} \
                         past_end_untouched={tail_ok} {}",
                        verdict(pass)
                    );
                    ok &= pass;
                }
            }
            Ok(ok)
        }

        /// The host API's refusals, each a named error: a case passes only
        /// when its call is an error whose text holds the case's own fragment
        /// (another check refusing first, or a launch error, does not).
        fn refusals(dev: &Dev<'_>) -> Result<bool, GateError> {
            if !wanted("g32_refuse") {
                return Ok(true);
            }
            let gpu = dev.gpu;
            let stream = gpu.stream();
            let sink = gpu.unlabelled_sink();
            let (k, rows) = (320usize, 32usize);
            let (w8, _) = weights(false, rows, k, 0x3500)?;
            let r8 = Res::new(dev, &w8, false, rows)?;
            let x = DeviceBuffer::from_host(stream, &activations(k, 8, 2))?;
            let short = DeviceBuffer::from_host(stream, &activations(k, 1, 3))?;
            // `act` quantized for all 8 columns; `act4` for 4 of its 8;
            // `narrow` holds 4; `act_q` takes the quantizer's refusals.
            let mut act = GemmAct32::new(stream, 8, k)?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &x, 8, &mut act, sink)?;
            let mut act4 = GemmAct32::new(stream, 8, k)?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &x, 4, &mut act4, sink)?;
            let mut narrow = GemmAct32::new(stream, 4, k)?;
            dev.g32
                .enqueue_quantize_gemm32(stream, &x, 4, &mut narrow, sink)?;
            let mut act_q = GemmAct32::new(stream, 8, k)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, 8 * rows)?;
            let mut y_short = DeviceBuffer::<f32>::zeroed(stream, 8 * rows - 1)?;
            let unfilled = GemmRoute::new(stream, 8, 1)?;
            let mut route = GemmRoute::new(stream, 8, 1)?;
            dev.gk.enqueue_route_dense(stream, 8, &mut route, sink)?;
            let mut remap_route = GemmRoute::new(stream, 8, 1)?;
            let w96 = DeviceTensor::upload(stream, &activations(96, 4, 1), 4, 96)?;
            let w320 = DeviceTensor::upload(stream, &activations(k, 4, 4), 4, k)?;
            let empty = DeviceBuffer::<u32>::zeroed(stream, 0)?;
            let map = DeviceBuffer::from_host(stream, &[0u32])?;
            let ids = DeviceBuffer::<u32>::zeroed(stream, 9)?;
            let plane = r8.weight(Lay::Q8Plane)?;
            let (qs, _) = r8.plane.as_ref().ok_or("a Q8_0 stack with no planes")?;
            // The scales of a stack one block wider than the codes.
            let d_wide =
                DeviceTensor::upload(stream, &vec![0u16; rows * (k / 32 + 1)], rows, k / 32 + 1)?;
            // `x` from its second value: 4 bytes past a 16-byte boundary.
            // SAFETY: values 1 .. 1 + 4·k of `x` (8·k values) lie inside it,
            // f32-aligned; `x` stays in place while the window lives (the
            // refused call reads nothing).
            let x_off =
                unsafe { bloomery_gpu::window::<f32>(x.cu_deviceptr() + 4, 4 * k, x.context()) };
            let gemm = |w: Gemm32Weight<'_>,
                        rows_per_expert: usize,
                        act: &GemmAct32,
                        route: &GemmRoute,
                        y: &mut DeviceBuffer<f32>| {
                dev.g32.enqueue_gemm32(
                    stream,
                    Gemm32Args {
                        w,
                        rows_per_expert,
                        act,
                        route,
                        input: GemmInput::PerSlot,
                        y,
                    },
                )
            };
            let unit = |r: Result<GemmAct32, bloomery_gpu::GpuError>| r.map(|_| ());
            let cases: Vec<(&str, Result<(), bloomery_gpu::GpuError>, &str)> = vec![
                (
                    "act_k_48",
                    unit(GemmAct32::new(stream, 8, 48)),
                    "k must be a multiple of 32",
                ),
                (
                    "act_cols_0",
                    unit(GemmAct32::new(stream, 0, k)),
                    "1 <= cols <=",
                ),
                (
                    "quant_short_input",
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &short, 8, &mut act_q, sink),
                    "< n_cols*k",
                ),
                (
                    "quant_cols_past_act",
                    dev.g32
                        .enqueue_quantize_gemm32(stream, &x, 9, &mut act_q, sink),
                    "1 <= n_cols <= act.cols()",
                ),
                (
                    "swiglu_act_limit_nan",
                    dev.g32.enqueue_swiglu_act_quant32(
                        stream,
                        &x,
                        &x,
                        Act::SwigluClamp { limit: f32::NAN },
                        8,
                        &mut act_q,
                        sink,
                    ),
                    "the clamp limit must be finite",
                ),
                (
                    "swiglu_act_short_input",
                    dev.g32.enqueue_swiglu_act_quant32(
                        stream,
                        &short,
                        &x,
                        Act::SiluMul,
                        8,
                        &mut act_q,
                        sink,
                    ),
                    "need n_cols*k",
                ),
                (
                    "unfilled_route",
                    gemm(plane, rows, &act, &unfilled, &mut y),
                    "a filled route table",
                ),
                (
                    "rows_not_16",
                    gemm(plane, 24, &act, &route, &mut y),
                    "rows_per_expert must be a positive multiple of 16",
                ),
                (
                    "q5_1_layout_at_q8_0_words",
                    gemm(Gemm32Weight::Q5_1File(&r8.file), rows, &act, &route, &mut y),
                    "Q5_1 rows at K =",
                ),
                (
                    "iq4nl_layout_at_q8_0_words",
                    gemm(
                        Gemm32Weight::Iq4NlFile(&r8.file),
                        rows,
                        &act,
                        &route,
                        &mut y,
                    ),
                    "IQ4_NL rows at K =",
                ),
                (
                    "plane_shape",
                    gemm(
                        Gemm32Weight::Q8_0Plane { qs, d: &d_wide },
                        rows,
                        &act,
                        &route,
                        &mut y,
                    ),
                    "Q8_0 planes at K =",
                ),
                (
                    "act_cols_short",
                    gemm(plane, rows, &narrow, &route, &mut y),
                    "activation columns, act holds 4",
                ),
                (
                    "act_quantized_short",
                    gemm(plane, rows, &act4, &route, &mut y),
                    "the last quantizer launch wrote 4",
                ),
                (
                    "y_short",
                    gemm(plane, rows, &act, &route, &mut y_short),
                    "< n_slots*rows_per_expert",
                ),
                (
                    "remap_empty_map",
                    dev.g32
                        .enqueue_route_remap(stream, &ids, &empty, 8, &mut remap_route, sink),
                    "the map holds 1 to",
                ),
                (
                    "remap_slots_past_table",
                    dev.g32
                        .enqueue_route_remap(stream, &ids, &map, 9, &mut remap_route, sink),
                    "1 <= n_slots <= the table's 8 slots",
                ),
                (
                    "f32_tile_k_96",
                    dev.g32.enqueue_f32_tile(stream, &w96, &x, 1, &mut y),
                    "k must be a positive multiple of 64",
                ),
                (
                    "f32_tile_short_x",
                    dev.g32.enqueue_f32_tile(stream, &w320, &short, 2, &mut y),
                    "x.len() ",
                ),
                (
                    "f32_tile_x_unaligned",
                    dev.g32.enqueue_f32_tile(stream, &w320, &x_off, 1, &mut y),
                    "must be 16-byte aligned",
                ),
            ];
            let all = cases
                .iter()
                .all(|(_, got, want)| matches!(got, Err(e) if e.to_string().contains(want)));
            println!(
                "gemm32 case=g32_refuse {} {}",
                cases
                    .iter()
                    .map(|(n, got, want)| match got {
                        Err(e) if e.to_string().contains(want) => format!("{n}=refused"),
                        Err(e) => format!("{n}=REFUSED-OTHERWISE({e})"),
                        Ok(()) => format!("{n}=ACCEPTED"),
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                verdict(all)
            );
            Ok(all)
        }

        /// Every 32-value case.
        pub(super) fn run(gpu: &Gpu, gk: &GemmKernels) -> Result<bool, GateError> {
            let dev = Dev {
                gpu,
                gk,
                g32: Gemm32Kernels::load(gpu.context())?,
            };
            let t0 = std::time::Instant::now();
            let mut ok = quant_case(&dev)?;
            ok &= swiglu_act_case(&dev)?;
            ok &= dense_case(&dev)?;
            ok &= routed_case(&dev)?;
            ok &= iq4nl_case(&dev)?;
            ok &= remap_case(&dev)?;
            ok &= f32tile_case(&dev)?;
            ok &= refusals(&dev)?;
            println!("gemm32 wall_s={:.1}", t0.elapsed().as_secs_f64());
            Ok(ok)
        }
    }
}
