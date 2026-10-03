//! GPU gate for the card kernels of Qwen3.6-35B-A3B's full-attention layers,
//! model-less at the model's shapes: 16 query heads over 2 key/value heads
//! (group 8) of 256 values, a NEOX rope over the first 64 values at θ 1e7,
//! RMS eps 1e-6, and a sigmoid output gate carried beside each query head in
//! the q projection's rows (`[q 256 | gate 256]` a head). Inputs are seeded
//! ([`activations`]); nothing is read from a model file.
//!
//! 1. Rope (`rope_neox::head_norm_neox_append_256`), against this binary's
//!    transcription of its rule: the norm (each thread's two squares added in
//!    f64, the warp butterfly, the four warp sums in warp order, `(sum / 256)
//!    as f32`), the turn of pairs `(i, i + 32)` for `i < 32` by the table's row
//!    at the token's position, the other 192 values `(scale · gain) · x`, the
//!    query read from the q+gate rows at head stride 512 and written to its own
//!    rows, the key turned in place, both planes' rows the f16 (nearest even)
//!    of the turned key and of the value — bit for bit, the planes' other rows
//!    untouched, and a rerun bit for bit. A position at the cache's height
//!    raises `cache_pos` with the launch's layer, leaves that token's heads NaN
//!    and appends nothing; every other token is the clean run's. Its q8_0
//!    append (`head_norm_neox_append_256_q8`) runs the same inputs: the heads
//!    the f16 entry's bit for bit, every appended row the two planes of
//!    `quantize_q8_0` (the engine's one Q8_0 quantizer) over the same f32
//!    values, packed by `weights::q8_0_planes`, a rerun bit for bit; and a
//!    value holding a NaN refuses its block — `kv_quant`, a NaN scale and zero
//!    codes, every other bit the clean run's.
//! 2. Decode flash (`flash_gqa::enqueue_pass_256`), both segment passes: each
//!    output within its first-order bound of the exact attention computed here
//!    in f64 on the same f16 keys and values (the model of
//!    `gate_qwen3moe_flash` at a head of 256: the scalar pass's score dot
//!    `γ(67)`, 64 fused multiply-adds per rotating partial; the tensor-core
//!    pass's `2^-11 + 2·γ(258)`), a rerun bit for bit, NaN in the cache rows
//!    past the count changing no bit, eight rows in one launch each bit for bit
//!    its one-row launch, a count of zero or past the cache raising `key_count`
//!    with NaN in exactly those rows, the captured launch (two nodes) the eager
//!    bits.
//! 3. Score order: a query whose two 128-value halves are equal, over keys `a`
//!    and `b` whose halves are swapped (`b = [a_hi | a_lo]`), with values that
//!    are multiples of 1/256. Summing each half's k16 chain on its own and then
//!    adding the halves — the order both the decode tensor-core pass and the
//!    prefill flash use — gives `a` and `b` the same score bit for bit, so
//!    both weigh 1 and each output is `(v_a + v_b)/2` exactly, in both kernels;
//!    any other order (one chain over the 256 dims, one half alone) splits the
//!    tie. The scores themselves are not observable; this is the clause that
//!    holds the two kernels to one score order.
//! 4. Prefill flash (`flash_gqa_prefill::enqueue_256`): the prefill band of
//!    `gate_qwen3moe_flash` at a head of 256 on seeded launches, `T` ∈
//!    [`SEED_T`] × first position `p0` ∈ [`SEED_P0`] (counts `p0 + t + 1`,
//!    crossing the 64-key tile edges): every row within its bound, every row
//!    bit for bit the same row launched alone, NaN in the cache rows past the
//!    launch's largest count changing no bit, a rerun bit for bit; then a
//!    count of zero and one past the cache (`key_count`, NaN in those rows,
//!    the rest the clean run's), the captured launch (one node) the eager
//!    bits, a head count the kernel is not built for refused, and windows at
//!    aligned offsets accepted while `q` at 4 bytes past 8 and `kc`/`vc` at 8
//!    bytes past 16 are refused by name.
//! 5. Gated quantizer (`gated_quant`): at gates of 0 and ±100, where the
//!    device's and the host's `sigmoid` are both exactly 1/2, 1 and 0, the
//!    gated launch's q8_1 bytes (codes and scales) bit for bit the plain
//!    quantizer's on the host-computed `attn · sigmoid(g)`, through both the
//!    decode activation (`Q8Act`, 1, 5 and 8 columns) and the GEMM's
//!    (`GemmAct`, [`GEMM_COLS`] columns); elsewhere (gates in [−6, 6]) every
//!    code within one of the plain quantizer's and every scale within
//!    `γ(10)` relative (the two `exp`s, device and host, at most three ulp
//!    apart, then the same three roundings), the count that differs printed;
//!    a NaN gate and an infinite gate each refuse their block (NaN scale, zero
//!    codes) and raise `quant_column` with the launch's layer, every other
//!    block the clean run's.
//! 6. ik's taps, on layer 3 of the Q4_K_M file (the qwen35moe family's batch
//!    set, five tokens, read through `refset`; the gains from the file): the
//!    rope launch on ik's own `Qaux-3` (the q+gate rows), `Kcur-3` and
//!    `Vcur-3` at the set's positions, each turned value within [`ROPE_BAND`]
//!    of its head's largest against `Qcur_roped-3`/`Kcur_roped-3` and each
//!    passed-through value against `Qcur_normed-3`/`Kcur_normed-3` — the norm
//!    is the only term that can move (the turn and the table are ik's bit for
//!    bit on the same normed values), the count of equal values printed; and
//!    the gated quantizer on ik's `fa-3` with the gates of `Qaux-3` against
//!    the plain quantizer on ik's `qkv_gated-3`, codes within one and scales
//!    within `γ(10)` (ik's sigmoid is its CPU `expf`, ours the card's).
//! 7. Qwen3.8-Flash-Next's layout (24 query heads over 2, group 12, heads of
//!    256) through the pack-of-four entries (`flash_gqa::enqueue_pass_256_p4`,
//!    `flash_gqa_prefill::enqueue_256_p4`), head `h` reading key head `h /
//!    12`: clause 2 at the counts [`DEC_KEYS_Q38`] (one key, and 2,051 — 33
//!    segments, the last of three keys) with a group of 13 refused by name,
//!    and clause 4 with [`SEED_Q38`] added (a 14-row tail tile, its last row
//!    slice two positions short; counts 2,049–2,051), the lines tagged `p4
//!    24/2`.
//! 8. Cross geometry: Qwen3.6's layout through the pack-of-four entries (two
//!    packs a key head) bit for bit the eight-head entries' output — the
//!    decode passes on one-row launches and the eight-row launch, the prefill
//!    on [`SEED_CROSS`]. A row's arithmetic does not depend on the heads that
//!    share its block or tensor-core tile, so the two geometries agree to the
//!    bit; the clause pins the new block and row maps to the gated kernel.
//!    The window refusals of clause 4 (both layouts) run last.
//! 9. Qwen3.8's token-pool selector (`qsa`), model-less at its shape (4
//!    indexer heads of 128, pools of 4, 2,048 tokens kept: 512 pools and a
//!    tail of at most 3, lists of at most 2,051): the pool pass over counts
//!    1..=8,193 bit for bit this binary's transcription of its rule (the
//!    four rows' mean, the RMS gain norm, the turn of the first 64 values at
//!    the pool's first position, f16), within its band of the references'
//!    rule in f64, the incomplete pool untouched, a rerun and launches of
//!    seven rows the same plane; the same on a crafted input whose rows
//!    cancel across 24 binades, where the transcription's pairwise and
//!    reversed sums move every key, so the bits see the mean's order; a NaN
//!    in one raw row raises `pool_select` with that pool alone not finite.
//! 10. Selection (score and top-k): the query heads bit for bit the
//!     transcription, each scored pool's score within its band of the f64
//!     score on the launch's own heads and keys, each list and length
//!     `runtime::qsa`'s rule on the launch's scores (the lower pool on a tie),
//!     a rerun, each row of a
//!     launch its one-row launch — at counts 2,051 (every token), 2,052 (the
//!     first dropped pool), 4,097 and 8,193 and on the eight rows of a verify
//!     at 8,186..=8,193; a tie of +0 scores across the cut, the lowest pools
//!     taken; counts 0 and past the cache an empty list;
//!     a NaN query raising `pool_select` with a defined list; nine rows
//!     refused by name. The rule is the modeling code's (exllamav3
//!     `qsa_indexer.py`, transformers): exactly 512 pools and the tail. ik and
//!     mainline cut 2,051 cells instead, so past position 2,050, when `(p + 1)
//!     % 4 != 3`, they also take `3 − t` cells of the 513th pool in ggml's tie
//!     order (`t` the tail's cells); these kernels do not (`runtime::qsa`'s
//!     test of the two cuts).
//! 11. The selected flash (`flash_gqa::enqueue_pass_256_p4_sel`) at counts up
//!     to 2,051, over the lists the selection wrote (every token), bit for bit
//!     the dense pack-of-four flash at those counts — both passes, one-row and
//!     eight-row launches. Two bodies are compared: the `_p4_sel` entries run
//!     their own copies of the `_p4` segment passes (`seg_scalar_ps`,
//!     `seg_mma_ps`), which differ only in the count bound and the staging
//!     load, so the clause holds the copies to the dense bodies' bits.
//! 12. Past 2,051: each row within the dense clause's band of the exact
//!     attention over its listed keys, NaN in every cache row no list names
//!     changing no bit, a rerun, each verify row its one-row launch; the
//!     captured chain (pool, score, top-k, segment pass, merge: five nodes)
//!     the eager bits.
//! 13. A list entry at the cache's height raising `pool_select`, a length of
//!     zero or past the width raising `key_count`: that row NaN, the other bit
//!     for bit clean.
//! 14. The five new entries compile with no local depot.
//! 15. Deep: the selector and the selected flash at a cache of 262,144 rows
//!     (Qwen3.8's `context_length`, the most a load serves), synthetic keys,
//!     no model: one pool launch over every count bit for bit the
//!     transcription and within its band, pool 65,535 the last; the
//!     selection at counts 65,537, 131,074, 262,143 and 262,144 in one
//!     launch and on the eight rows of a verify at 262,137..=262,144 — the
//!     heads bit for bit, the scores within their band, each list
//!     `runtime::qsa`'s rule over the full span of 65,536 pools, every list
//!     holding rows past 65,535 and a tail ending at its position; the selected
//!     flash, both passes, within the band of the exact attention over the
//!     listed rows, with NaN in every unlisted cache row changing no bit.
//!     The counts reach the kernels as their `u32` arguments, the list
//!     entries as `u32` cache rows: a count or a row cut to 16 bits, or the
//!     context clamped at 65,535, reads other rows or refuses the count.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen35moe_attn: built without the `gpu` feature; see `just gate-gpu-qwen35moe-attn`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35moe_attn", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::fault::{Fault, FaultSink, FaultSite, LAYER_NONE};
    use bloomery_gpu::flash_gqa::{
        FlashGqaKernels, GROUP, GqaArgs, GqaSelArgs, HEAD_256 as HEAD, KEY_TILE, PACK_4, SEG_KEYS,
        partials_ms_len, partials_v_len_256, segments_for,
    };
    use bloomery_gpu::flash_gqa_prefill::{FlashGqaPrefill, GqaPrefillArgs, KEY_TILE as PREF_TILE};
    use bloomery_gpu::gated_quant::{GateLayout, GatedQuantKernels};
    use bloomery_gpu::gemm::GemmAct;
    use bloomery_gpu::qsa::{
        DIM as IDX_DIM, HEADS as IDX_HEADS, MAX_ROWS, POOL, PoolArgs, QsaKernels, QsaScratch,
        ROT as IDX_ROT, SelectArgs, list_width, pools_for,
    };
    use bloomery_gpu::rope_neox::{
        PartialNeoxArgs, ROT_256 as ROT, RopeNeoxKernels, owned_pair, q8_plane_lens,
    };
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu::route_core::sigmoid;
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{Gpu, GpuError, Q8Act, window};
    use bloomery_gpu_gates::qwen3moe::dev::{SENTINEL_Q8_CODE, SENTINEL_Q8_SCALE};
    use bloomery_gpu_gates::rounding::{U, U_F32, butterfly, gamma};
    use bloomery_gpu_gates::{
        GateError, Layout, RefManifest, RowKind, activations, bits_equal, checks_failed, data_dir,
        no_local_depot, ref_ints, ref_tensor_logical_in, split_f32, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{Q8Block, f32_to_f16_bits, half_to_f32};
    use model::arch::deepseek2::attn::quantize_q8_0;
    use refset::arch::qwen35moe::{BATCH, IK, MODEL};
    use runtime::qsa::{Qsa, select as qsa_select};
    use std::mem::ManuallyDrop;

    /// The model's attention shape.
    const N_HEAD: usize = 16;
    const N_KV: usize = 2;
    const THETA: f32 = 1.0e7;
    const EPS: f32 = 1.0e-6;
    /// The width of a token's q+gate row, and of one head's pair in it.
    const QG_HEAD: usize = 2 * HEAD;
    const QG_ROW: usize = N_HEAD * QG_HEAD;

    /// An f16 NaN the padded rows are overwritten with; the plane sentinel
    /// the rope check starts from.
    const NAN16: u16 = 0x7e00;
    const SENTINEL: u16 = 0x5a5a;

    /// The rope check's tokens and their first position, and the cache height.
    const ROPE_M: usize = 7;
    const ROPE_P0: usize = 3;
    const ROPE_CTX: usize = 40;

    /// Qwen3.8-Flash-Next's full-attention layers: 24 query heads over the
    /// same two key heads (group 12), heads of 256, served by the pack-of-four
    /// entries.
    const N_HEAD_Q38: usize = 24;

    /// Decode: the live counts of the one-row launches, and the rows of the
    /// multi-row launch. Qwen3.8's add one key (the first position) and 2,051,
    /// the first program's last: 33 segments, the last holding three keys.
    const DEC_KEYS: [usize; 4] = [5, 64, 1025, 4097];
    const DEC_KEYS_Q38: [usize; 5] = [1, 5, 512, 2051, 4097];
    const ROWS: usize = 8;
    /// Cache rows past a launch's largest count.
    const PAD: usize = 40;

    /// The prefill flash's seeded launches: row counts and first positions.
    const SEED_T: [usize; 5] = [1, 17, 64, 65, 512];
    const SEED_P0: [usize; 4] = [0, 63, 64, 1000];
    /// Qwen3.8's further `(p0, T)`: a tail tile of 14 of the pack-of-four
    /// kernel's 16 positions (its last row slice two positions short), and the
    /// first program's last three positions (counts 2,049–2,051).
    const SEED_Q38: [(usize, usize); 2] = [(0, 30), (2048, 3)];
    /// The cross-geometry clause's prefill launches, `(p0, T)`.
    const SEED_CROSS: [(usize, usize); 3] = [(0, 17), (63, 65), (1000, 30)];

    /// A head layout a flash clause runs at, and the entries that serve it:
    /// `n_head` query heads over [`N_KV`] key heads, through the eight-head
    /// entries (`enqueue_pass_256`, `enqueue_256`) or the pack-of-four ones
    /// (`enqueue_pass_256_p4`, `enqueue_256_p4`).
    #[derive(Clone, Copy)]
    struct Shape {
        n_head: usize,
        pack4: bool,
    }

    impl Shape {
        /// Query heads per key head: head `h` reads key head `h / group`.
        fn group(self) -> usize {
            self.n_head / N_KV
        }

        /// f32 of one row's query (or output) heads.
        fn width(self) -> usize {
            self.n_head * HEAD
        }

        /// The tag a clause's lines carry: none for Qwen3.6 through its own
        /// entries.
        fn tag(self) -> String {
            if self.pack4 {
                format!(" p4 {}/{N_KV}", self.n_head)
            } else {
                String::new()
            }
        }

        /// The prefill enqueue's name, as its refusals carry it.
        fn pref_what(self) -> &'static str {
            if self.pack4 {
                "flash_gqa_prefill::enqueue_256_p4"
            } else {
                "flash_gqa_prefill::enqueue_256"
            }
        }
    }

    /// Qwen3.6 through the eight-head entries.
    const Q36: Shape = Shape {
        n_head: N_HEAD,
        pack4: false,
    };
    /// Qwen3.8 (group 12) through the pack-of-four entries.
    const Q38: Shape = Shape {
        n_head: N_HEAD_Q38,
        pack4: true,
    };
    /// Qwen3.6's layout through the pack-of-four entries (two packs a key
    /// head): the cross-geometry clause's second arm.
    const Q36_P4: Shape = Shape {
        n_head: N_HEAD,
        pack4: true,
    };
    const _: () = assert!(GROUP == 8 && N_HEAD == N_KV * GROUP);
    const _: () = assert!(PACK_4 == 4 && N_HEAD_Q38 == N_KV * 3 * PACK_4);

    /// One decode pass through `sh`'s entry.
    fn enqueue_dec(
        k: &FlashGqaKernels,
        stream: &CudaStream,
        sh: Shape,
        args: GqaArgs<'_>,
        mma: bool,
    ) -> Result<(), GpuError> {
        if sh.pack4 {
            k.enqueue_pass_256_p4(stream, args, sh.n_head, mma)
        } else {
            k.enqueue_pass_256(stream, args, mma)
        }
    }

    /// One prefill launch through `sh`'s entry.
    fn enqueue_pref(
        kp: &FlashGqaPrefill,
        stream: &CudaStream,
        sh: Shape,
        args: GqaPrefillArgs<'_>,
    ) -> Result<(), GpuError> {
        if sh.pack4 {
            kp.enqueue_256_p4(stream, args)
        } else {
            kp.enqueue_256(stream, args)
        }
    }

    /// The seeded query's scale over [`activations`]' [-1, 1): scores of a
    /// few units, so the weights spread over several orders of magnitude.
    const SEED_Q_SCALE: f32 = 3.0;

    /// The GEMM activation's columns in the gated quantizer check.
    const GEMM_COLS: usize = 37;
    /// Values a column of the gated quantizer's input: the heads' outputs.
    const QK: usize = N_HEAD * HEAD;
    /// The layer the fault checks label their launches with.
    const LAYER: usize = 13;
    /// The full-attention layer the model clauses read.
    const MODEL_LAYER: usize = 3;
    /// `gate_qwen3moe_rope`'s band on a turned value against ik's, of its
    /// head's largest value: the norm's scale may round one ulp apart (`2u`
    /// of the scale, `4u` with the gain), the inner products round on both
    /// sides, the fused add once: under `20u` of the head's largest.
    const ROPE_BAND: f32 = 20.0 * U_F32;

    /// The attention scale: one over the square root of the head width.
    fn scale() -> f32 {
        1.0f32 / (HEAD as f32).sqrt()
    }

    fn to16(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| f32_to_f16_bits(x)).collect()
    }

    fn from16(v: &[u16]) -> Vec<f32> {
        v.iter().map(|&h| half_to_f32(h)).collect()
    }

    // ------------------------------------------------------------ rope

    /// One head's norm and partial turn, the kernel's rule: `x` the head's 256
    /// values, `cs` its position's 64 table values.
    fn head_rule(x: &[f32], gain: &[f32], cs: &[f32]) -> Vec<f32> {
        let lanes: Vec<f64> = (0..HEAD / 2)
            .map(|t| {
                let (i0, i1) = owned_pair(HEAD, ROT, t);
                f64::from(x[i0] * x[i0]) + f64::from(x[i1] * x[i1])
            })
            .collect();
        let warp = |w: usize| -> f64 {
            let v: [f64; 32] = lanes[32 * w..32 * w + 32]
                .try_into()
                .expect("a warp is 32 lanes");
            butterfly(v)
        };
        let sum = ((warp(0) + warp(1)) + warp(2)) + warp(3);
        let mean = (sum / HEAD as f64) as f32;
        let scale = 1.0 / (mean + EPS).sqrt();
        let mut y: Vec<f32> = x.iter().zip(gain).map(|(&v, &g)| (scale * g) * v).collect();
        for i in 0..ROT / 2 {
            let (c, s) = (cs[2 * i], cs[2 * i + 1]);
            let (x0, x1) = (y[i], y[i + ROT / 2]);
            y[i] = x0.mul_add(c, -(x1 * s));
            y[i + ROT / 2] = x0.mul_add(s, x1 * c);
        }
        y
    }

    /// The rope launch's outputs read back: the query rows, the key rows (in
    /// place) and the two planes.
    struct RopeOut {
        q: Vec<f32>,
        k: Vec<f32>,
        ck: Vec<u16>,
        cv: Vec<u16>,
    }

    /// The rope check's inputs on the host.
    #[derive(Clone)]
    struct RopeIn {
        qg: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        gq: Vec<f32>,
        gk: Vec<f32>,
        table: Vec<f32>,
    }

    fn rope_run(
        kern: &RopeNeoxKernels,
        stream: &CudaStream,
        inp: &RopeIn,
        pos: &[u32],
        fault: FaultSink,
    ) -> Result<RopeOut, GateError> {
        let m = pos.len();
        let qg = DeviceBuffer::from_host(stream, &inp.qg)?;
        let mut q = DeviceBuffer::from_host(stream, &vec![0.0f32; m * N_HEAD * HEAD])?;
        let mut k = DeviceBuffer::from_host(stream, &inp.k)?;
        let v = DeviceBuffer::from_host(stream, &inp.v)?;
        let gq = DeviceBuffer::from_host(stream, &inp.gq)?;
        let gk = DeviceBuffer::from_host(stream, &inp.gk)?;
        let table = DeviceBuffer::from_host(stream, &inp.table)?;
        let posd = DeviceBuffer::from_host(stream, pos)?;
        let plane = vec![SENTINEL; N_KV * ROPE_CTX * HEAD];
        let mut ck = DeviceBuffer::from_host(stream, &plane)?;
        let mut cv = DeviceBuffer::from_host(stream, &plane)?;
        kern.enqueue_head_norm_neox_append_256(
            stream,
            PartialNeoxArgs {
                qg: &qg,
                q: &mut q,
                k: &mut k,
                v: &v,
                gq: &gq,
                gk: &gk,
                table: &table,
                pos: &posd,
                eps: EPS,
                n_head: N_HEAD,
                n_kv: N_KV,
                ctx: ROPE_CTX,
                m,
                fault,
                cache_k: &mut ck,
                cache_v: &mut cv,
            },
        )?;
        stream.synchronize()?;
        Ok(RopeOut {
            q: q.to_host_vec(stream)?,
            k: k.to_host_vec(stream)?,
            ck: ck.to_host_vec(stream)?,
            cv: cv.to_host_vec(stream)?,
        })
    }

    /// The host rule's outputs for `pos`: every token's heads, and the planes
    /// from the sentinel with each token's rows written.
    fn rope_host(inp: &RopeIn, pos: &[u32]) -> RopeOut {
        let m = pos.len();
        let mut q = vec![0.0f32; m * N_HEAD * HEAD];
        let mut k = vec![0.0f32; m * N_KV * HEAD];
        let mut ck = vec![SENTINEL; N_KV * ROPE_CTX * HEAD];
        let mut cv = ck.clone();
        for (t, &p) in pos.iter().enumerate() {
            let p = p as usize;
            let cs = &inp.table[p * ROT..(p + 1) * ROT];
            for h in 0..N_HEAD {
                let src = &inp.qg[t * QG_ROW + h * QG_HEAD..][..HEAD];
                q[(t * N_HEAD + h) * HEAD..][..HEAD].copy_from_slice(&head_rule(src, &inp.gq, cs));
            }
            for j in 0..N_KV {
                let at = (t * N_KV + j) * HEAD;
                let y = head_rule(&inp.k[at..at + HEAD], &inp.gk, cs);
                k[at..at + HEAD].copy_from_slice(&y);
                let row = (j * ROPE_CTX + p) * HEAD;
                ck[row..row + HEAD].copy_from_slice(&to16(&y));
                cv[row..row + HEAD].copy_from_slice(&to16(&inp.v[at..at + HEAD]));
            }
        }
        RopeOut { q, k, ck, cv }
    }

    /// The rope check's inputs ([`rope_check`]'s construction, shared with
    /// its q8_0 clause): the table, the rows, the gains and the positions.
    fn rope_inputs() -> Result<(RopeIn, Vec<u32>), GateError> {
        let table_rt = RopeTable::new(&RopeSpec::window(THETA, ROT))?;
        let mut table = Vec::with_capacity(ROPE_CTX * ROT);
        for p in 0..ROPE_CTX {
            table_rt.push(u32::try_from(p)?, Direction::Forward, &mut table);
        }
        // The gates are 1e6 in the q+gate rows: a query read at the wrong
        // stride meets them.
        let mut qg = activations(QG_HEAD, ROPE_M * N_HEAD, 3);
        for (i, v) in qg.iter_mut().enumerate() {
            if i % QG_HEAD >= HEAD {
                *v = 1.0e6;
            }
        }
        let gq: Vec<f32> = activations(HEAD, 1, 4)
            .iter()
            .map(|v| 1.25 + 0.5 * v)
            .collect();
        let gk: Vec<f32> = activations(HEAD, 1, 5)
            .iter()
            .map(|v| 1.3 + 0.5 * v)
            .collect();
        let inp = RopeIn {
            qg,
            k: activations(HEAD, ROPE_M * N_KV, 6)
                .iter()
                .map(|v| 4.0 * v)
                .collect(),
            v: activations(HEAD, ROPE_M * N_KV, 7),
            gq,
            gk,
            table,
        };
        let pos: Vec<u32> = (0..ROPE_M)
            .map(|t| u32::try_from(ROPE_P0 + t))
            .collect::<Result<_, _>>()?;
        Ok((inp, pos))
    }

    fn rope_check(gpu: &Gpu, kern: &RopeNeoxKernels) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (inp, pos) = rope_inputs()?;
        let unl = gpu.unlabelled_sink();
        let a = rope_run(kern, stream, &inp, &pos, unl)?;
        let b = rope_run(kern, stream, &inp, &pos, unl)?;
        let want = rope_host(&inp, &pos);
        let q_ok = bits_equal(&a.q, &want.q);
        let k_ok = bits_equal(&a.k, &want.k);
        let planes_ok = a.ck == want.ck && a.cv == want.cv;
        let rerun =
            bits_equal(&a.q, &b.q) && bits_equal(&a.k, &b.k) && a.ck == b.ck && a.cv == b.cv;
        let pass = q_ok && k_ok && planes_ok && rerun;
        println!(
            "rope m={ROPE_M} positions {ROPE_P0}.. ctx={ROPE_CTX} head {HEAD} rot {ROT}: q (read at \
             head stride {QG_HEAD}) bit_exact_host={q_ok} k in place bit_exact_host={k_ok} \
             planes_exact={planes_ok} rerun={rerun} {}",
            verdict(pass)
        );

        // The last token's position at the cache's height.
        let bad_t = ROPE_M - 1;
        let mut bad = pos.clone();
        bad[bad_t] = u32::try_from(ROPE_CTX)?;
        let before = gpu.fault()?;
        let sink = gpu.layer_sink(LAYER)?;
        let r = rope_run(kern, stream, &inp, &bad, sink)?;
        let raised = gpu.take_fault()?;
        let want_fault = Some(Fault::at(u32::try_from(LAYER)?, FaultSite::CachePos));
        let (qw, kw) = (N_HEAD * HEAD, N_KV * HEAD);
        let others = bits_equal(&r.q[..bad_t * qw], &a.q[..bad_t * qw])
            && bits_equal(&r.k[..bad_t * kw], &a.k[..bad_t * kw]);
        let nan = r.q[bad_t * qw..].iter().all(|v| v.is_nan())
            && r.k[bad_t * kw..].iter().all(|v| v.is_nan());
        // The clean run minus the bad token's rows: those keep the sentinel.
        let mut ck_want = a.ck.clone();
        let mut cv_want = a.cv.clone();
        for j in 0..N_KV {
            let row = (j * ROPE_CTX + pos[bad_t] as usize) * HEAD;
            ck_want[row..row + HEAD].fill(SENTINEL);
            cv_want[row..row + HEAD].fill(SENTINEL);
        }
        let not_appended = r.ck == ck_want && r.cv == cv_want;
        let again = rope_run(kern, stream, &inp, &pos, unl)?;
        let clean_after = gpu.fault()?.is_none() && bits_equal(&again.q, &a.q);
        let fault_ok = before.is_none()
            && raised == want_fault
            && others
            && nan
            && not_appended
            && clean_after;
        println!(
            "rope fault: token {bad_t} at position {ROPE_CTX} (the cache's height), layer {LAYER}: \
             word {raised:?} (want {want_fault:?}), its heads NaN {nan}, nothing appended \
             {not_appended}, other tokens bit-identical {others}, clean rerun and word clean \
             {clean_after} {}",
            verdict(fault_ok)
        );
        Ok(pass && fault_ok)
    }

    /// The q8_0 append's launch (`head_norm_neox_append_256_q8`) read back:
    /// [`RopeOut`]'s four planes in place of its two.
    struct RopeQ8Out {
        q: Vec<f32>,
        k: Vec<f32>,
        kq: Vec<u32>,
        kd: Vec<u16>,
        vq: Vec<u32>,
        vd: Vec<u16>,
    }

    fn rope_q8_run(
        kern: &RopeNeoxKernels,
        stream: &CudaStream,
        inp: &RopeIn,
        pos: &[u32],
        fault: FaultSink,
    ) -> Result<RopeQ8Out, GateError> {
        use bloomery_gpu::rope_neox::PartialNeoxQ8Args;
        let m = pos.len();
        let qg = DeviceBuffer::from_host(stream, &inp.qg)?;
        let mut q = DeviceBuffer::from_host(stream, &vec![0.0f32; m * N_HEAD * HEAD])?;
        let mut k = DeviceBuffer::from_host(stream, &inp.k)?;
        let v = DeviceBuffer::from_host(stream, &inp.v)?;
        let gq = DeviceBuffer::from_host(stream, &inp.gq)?;
        let gk = DeviceBuffer::from_host(stream, &inp.gk)?;
        let table = DeviceBuffer::from_host(stream, &inp.table)?;
        let posd = DeviceBuffer::from_host(stream, pos)?;
        let (words, scales) = q8_plane_lens(HEAD, N_KV, ROPE_CTX);
        let codes = vec![SENTINEL_Q8_CODE; words];
        let sc = vec![SENTINEL_Q8_SCALE; scales];
        let mut kq = DeviceBuffer::from_host(stream, &codes)?;
        let mut kd = DeviceBuffer::from_host(stream, &sc)?;
        let mut vq = DeviceBuffer::from_host(stream, &codes)?;
        let mut vd = DeviceBuffer::from_host(stream, &sc)?;
        kern.enqueue_head_norm_neox_append_256_q8(
            stream,
            PartialNeoxQ8Args {
                qg: &qg,
                q: &mut q,
                k: &mut k,
                v: &v,
                gq: &gq,
                gk: &gk,
                table: &table,
                pos: &posd,
                eps: EPS,
                n_head: N_HEAD,
                n_kv: N_KV,
                ctx: ROPE_CTX,
                m,
                fault,
                kq: &mut kq,
                kd: &mut kd,
                vq: &mut vq,
                vd: &mut vd,
            },
        )?;
        stream.synchronize()?;
        Ok(RopeQ8Out {
            q: q.to_host_vec(stream)?,
            k: k.to_host_vec(stream)?,
            kq: kq.to_host_vec(stream)?,
            kd: kd.to_host_vec(stream)?,
            vq: vq.to_host_vec(stream)?,
            vd: vd.to_host_vec(stream)?,
        })
    }

    /// The q8_0 append's expected planes ([`rope_host`]'s rows through
    /// `quantize_q8_0` and `q8_0_planes`, every other slot the sentinel).
    fn rope_q8_host(inp: &RopeIn, pos: &[u32]) -> (Vec<u32>, Vec<u16>, Vec<u32>, Vec<u16>) {
        let (words, scales) = q8_plane_lens(HEAD, N_KV, ROPE_CTX);
        let mut kq = vec![SENTINEL_Q8_CODE; words];
        let mut kd = vec![SENTINEL_Q8_SCALE; scales];
        let mut vq = kq.clone();
        let mut vd = kd.clone();
        let pack = |vals: &[f32], q: &mut [u32], d: &mut [u16], row: usize| {
            let blocks: Vec<Q8Block> = vals.chunks(32).map(quantize_q8_0).collect();
            let (qs, ds) = q8_0_planes(&blocks);
            q[row * (HEAD / 4)..row * (HEAD / 4) + qs.len()].copy_from_slice(&qs);
            d[row * (HEAD / 32)..row * (HEAD / 32) + ds.len()].copy_from_slice(&ds);
        };
        for (t, &p) in pos.iter().enumerate() {
            let p = p as usize;
            let cs = &inp.table[p * ROT..(p + 1) * ROT];
            for j in 0..N_KV {
                let at = (t * N_KV + j) * HEAD;
                let y = head_rule(&inp.k[at..at + HEAD], &inp.gk, cs);
                let row = j * ROPE_CTX + p;
                pack(&y, &mut kq, &mut kd, row);
                pack(&inp.v[at..at + HEAD], &mut vq, &mut vd, row);
            }
        }
        (kq, kd, vq, vd)
    }

    /// The q8_0 append clause (module doc): whether it held.
    fn rope_q8_check(gpu: &Gpu, kern: &RopeNeoxKernels) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (inp, pos) = rope_inputs()?;
        let unl = gpu.unlabelled_sink();
        let f16 = rope_run(kern, stream, &inp, &pos, unl)?;
        let a = rope_q8_run(kern, stream, &inp, &pos, unl)?;
        let b = rope_q8_run(kern, stream, &inp, &pos, unl)?;
        // The prefix the two entries share: the heads bit for bit.
        let heads = bits_equal(&a.q, &f16.q) && bits_equal(&a.k, &f16.k);
        let (wkq, wkd, wvq, wvd) = rope_q8_host(&inp, &pos);
        let planes = a.kq == wkq && a.kd == wkd && a.vq == wvq && a.vd == wvd;
        let rerun = bits_equal(&a.q, &b.q)
            && bits_equal(&a.k, &b.k)
            && a.kq == b.kq
            && a.kd == b.kd
            && a.vq == b.vq
            && a.vd == b.vd;
        let pass = heads && planes && rerun;
        println!(
            "rope q8 m={ROPE_M} positions {ROPE_P0}.. ctx={ROPE_CTX} head {HEAD} rot {ROT}: q/k \
             the f16 append's bits {heads} planes_exact={planes} rerun={rerun} {}",
            verdict(pass)
        );

        // A value row holding a NaN: the block that holds it refused.
        let (bad_t, bad_j, bad_val) = (2usize, 1usize, 200usize);
        let bad_block = bad_val / 32;
        let mut bad_inp = inp.clone();
        bad_inp.v[(bad_t * N_KV + bad_j) * HEAD + bad_val] = f32::NAN;
        let before = gpu.fault()?;
        let sink = gpu.layer_sink(LAYER)?;
        let bad = rope_q8_run(kern, stream, &bad_inp, &pos, sink)?;
        let raised = gpu.take_fault()?;
        let want_fault = Some(Fault::at(u32::try_from(LAYER)?, FaultSite::KvQuant));
        let row = bad_j * ROPE_CTX + pos[bad_t] as usize;
        let (mut wvq, mut wvd) = (a.vq.clone(), a.vd.clone());
        wvq[row * (HEAD / 4) + 8 * bad_block..row * (HEAD / 4) + 8 * (bad_block + 1)].fill(0);
        wvd[row * (HEAD / 32) + bad_block] = f32_to_f16_bits(f32::NAN);
        let others = bad.kq == a.kq
            && bad.kd == a.kd
            && bits_equal(&bad.q, &a.q)
            && bits_equal(&bad.k, &a.k);
        let refused = bad.vq == wvq && bad.vd == wvd;
        let again = rope_q8_run(kern, stream, &inp, &pos, unl)?;
        let clean_after = gpu.fault()?.is_none() && bits_equal(&again.q, &a.q) && again.vd == a.vd;
        let fault_ok = before.is_none() && raised == want_fault && others && refused && clean_after;
        println!(
            "rope q8 fault: token {bad_t} head {bad_j} value {bad_val} NaN, layer {LAYER}: word \
             {raised:?} (want {want_fault:?}), the block a NaN scale and zero codes {refused}, \
             every other bit the clean run's {others}, clean rerun and word clean {clean_after} {}",
            verdict(fault_ok)
        );
        Ok(pass && fault_ok)
    }

    // ------------------------------------------------------- the band

    /// The exact attention of one head in f64, and each kernel's bound.
    struct Exact {
        o: Vec<f64>,
        bound_scalar: Vec<f64>,
        bound_mma: Vec<f64>,
        bound_pref: Vec<f64>,
    }

    /// `gate_qwen3moe_flash`'s model at a head of [`HEAD`] (module doc):
    /// `q` one head's query, `kh`/`vh` its key head's first `n` rows.
    fn exact(q: &[f32], kh: &[f32], vh: &[f32], n: usize, scale: f32) -> Exact {
        let sc = f64::from(scale);
        let dot = |j: usize, abs: bool| -> f64 {
            let k = &kh[j * HEAD..(j + 1) * HEAD];
            q.iter()
                .zip(k)
                .map(|(&a, &b)| {
                    let p = f64::from(a) * f64::from(b);
                    if abs { p.abs() } else { p }
                })
                .sum::<f64>()
        };
        let s: Vec<f64> = (0..n).map(|j| sc * dot(j, false)).collect();
        let a: Vec<f64> = (0..n).map(|j| sc.abs() * dot(j, true)).collect();
        // The tensor-core scores also carry the query's f16 rounding: 2^-25
        // absolute below f16's normal range.
        let a16: Vec<f64> = (0..n)
            .map(|j| {
                let k = &kh[j * HEAD..(j + 1) * HEAD];
                a[j] + sc.abs()
                    * 2f64.powi(-14)
                    * k.iter().map(|&b| f64::from(b).abs()).sum::<f64>()
            })
            .collect();
        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let p: Vec<f64> = s.iter().map(|&v| (v - m).exp()).collect();
        let z: f64 = p.iter().sum();
        let pb: Vec<f64> = p.iter().map(|&v| v / z).collect();
        let x: Vec<f64> = s.iter().map(|&v| m - v).collect();
        let xmax = x.iter().copied().fold(0.0f64, f64::max);
        let segs = n.div_ceil(SEG_KEYS);
        let r_ours = (n.div_ceil(KEY_TILE) + segs) as f64;
        // The scalar dot: 64 fused multiply-adds per rotating partial, two
        // combine levels, the scale.
        let es_o = gamma(HEAD / 4 + 3);
        // Each half's tensor-core chain, their sum, and the query's f16.
        let es_m = 2f64.powi(-11) + 2.0 * gamma(HEAD + 2);
        let acc_o = gamma(SEG_KEYS + segs + 8);
        let r_pref = n.div_ceil(PREF_TILE) as f64;
        let w16: Vec<f64> = pb
            .iter()
            .zip(&p)
            .map(|(&q, &pm)| {
                let rel = 2f64.powi(-11) * q;
                if pm >= 2f64.powi(-14) {
                    rel
                } else {
                    rel.max(2f64.powi(-25) / z)
                }
            })
            .collect();
        let dbar: f64 = w16.iter().sum();
        let acc_pv = 2.0 * gamma(n.div_ceil(16) + 16);
        let acc_l = gamma(n.div_ceil(4) + n.div_ceil(PREF_TILE) + 2);
        let mut o = vec![0.0f64; HEAD];
        for (j, &w) in pb.iter().enumerate() {
            for (d, od) in o.iter_mut().enumerate() {
                *od += w * f64::from(vh[j * HEAD + d]);
            }
        }
        let mut bo = vec![0.0f64; HEAD];
        let mut bm = vec![0.0f64; HEAD];
        let mut bp = vec![0.0f64; HEAD];
        for d in 0..HEAD {
            let (mut t_o, mut t_m, mut t_p, mut pv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for j in 0..n {
                let v = f64::from(vh[j * HEAD + d]);
                let dev = (v - o[d]).abs();
                let ee = 4.0 * U + 2.0 * U * x[j];
                t_o += pb[j] * (es_o * a[j] + ee + 4.0 * U * r_ours + 2.0 * U * xmax) * dev;
                t_m += pb[j] * (es_m * a16[j] + ee + 4.0 * U * r_ours + 2.0 * U * xmax) * dev;
                t_p += (pb[j] * (es_m * a16[j] + ee + 4.0 * U * r_pref + 2.0 * U * xmax)
                    + w16[j]
                    + pb[j] * dbar)
                    * dev;
                pv += pb[j] * v.abs();
            }
            bo[d] = t_o + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bm[d] = t_m + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bp[d] = t_p + acc_pv * pv + acc_l * o[d].abs() + 2.0 * U * o[d].abs();
        }
        Exact {
            o,
            bound_scalar: bo,
            bound_mma: bm,
            bound_pref: bp,
        }
    }

    /// Which bound a band check reads.
    #[derive(Clone, Copy)]
    enum Pass {
        Scalar,
        Mma,
        Prefill,
    }

    impl Pass {
        fn name(self) -> &'static str {
            match self {
                Pass::Scalar => "scalar",
                Pass::Mma => "mma",
                Pass::Prefill => "prefill",
            }
        }
    }

    /// The f32 values of one cache: two planes of `N_KV · ctx` rows.
    struct HostCache {
        kf: Vec<f32>,
        vf: Vec<f32>,
        ctx: usize,
    }

    /// Every (row, head) of `y` against its exact value under `pass`'s bound:
    /// `q` holds the rows' `sh.n_head` query heads token-major, head `h`
    /// reading key head `h / sh.group()`, `counts[t]` row `t`'s keys. Returns
    /// whether every value is within its bound and the largest measured over
    /// bound. Rows are shared among worker threads.
    fn band_rows(
        sh: Shape,
        q: &[f32],
        counts: &[usize],
        cache: &HostCache,
        y: &[f32],
        pass: Pass,
    ) -> (bool, f64) {
        let workers = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        let chunk = counts.len().div_ceil(workers).max(1);
        let parts: Vec<(bool, f64)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..counts.len())
                .step_by(chunk)
                .map(|r0| {
                    sc.spawn(move || {
                        let (mut ok, mut worst) = (true, 0.0f64);
                        for (t, &n) in counts.iter().enumerate().skip(r0).take(chunk) {
                            for h in 0..sh.n_head {
                                let plane = (h / sh.group()) * cache.ctx * HEAD;
                                let row = (t * sh.n_head + h) * HEAD;
                                let ex = exact(
                                    &q[row..row + HEAD],
                                    &cache.kf[plane..plane + n * HEAD],
                                    &cache.vf[plane..plane + n * HEAD],
                                    n,
                                    scale(),
                                );
                                let bound = match pass {
                                    Pass::Scalar => &ex.bound_scalar,
                                    Pass::Mma => &ex.bound_mma,
                                    Pass::Prefill => &ex.bound_pref,
                                };
                                for d in 0..HEAD {
                                    let e = (f64::from(y[row + d]) - ex.o[d]).abs();
                                    worst = worst.max(e / bound[d]);
                                    ok &= e <= bound[d];
                                }
                            }
                        }
                        (ok, worst)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or((false, f64::INFINITY)))
                .collect()
        });
        parts
            .iter()
            .fold((true, 0.0f64), |(o, w), &(ok, wr)| (o && ok, w.max(wr)))
    }

    /// A seeded cache of `ctx` rows a key head, NaN (f16) in the rows at or
    /// past `live` in the padded copy: the planes, their padded copies, and
    /// the f32 values.
    struct Cache {
        kc: DeviceBuffer<u16>,
        vc: DeviceBuffer<u16>,
        kn: DeviceBuffer<u16>,
        vn: DeviceBuffer<u16>,
        host: HostCache,
        kb: Vec<u16>,
        vb: Vec<u16>,
    }

    fn cache(stream: &CudaStream, ctx: usize, live: usize, seed: u32) -> Result<Cache, GateError> {
        let kb = to16(&activations(HEAD, N_KV * ctx, seed));
        let vb = to16(&activations(HEAD, N_KV * ctx, seed + 1));
        let pad = |b: &[u16]| -> Vec<u16> {
            b.iter()
                .enumerate()
                .map(|(i, &h)| if (i / HEAD) % ctx >= live { NAN16 } else { h })
                .collect()
        };
        Ok(Cache {
            kc: DeviceBuffer::from_host(stream, &kb)?,
            vc: DeviceBuffer::from_host(stream, &vb)?,
            kn: DeviceBuffer::from_host(stream, &pad(&kb))?,
            vn: DeviceBuffer::from_host(stream, &pad(&vb))?,
            host: HostCache {
                kf: from16(&kb),
                vf: from16(&vb),
                ctx,
            },
            kb,
            vb,
        })
    }

    // ---------------------------------------------------------- decode

    /// One decode launch of `n_keys.len()` rows through `sh`'s entry, into
    /// fresh scratch, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernels, the stream and sink, the layout, the rows, the cache and its height, the pass"
    )]
    fn run_dec(
        k: &FlashGqaKernels,
        stream: &CudaStream,
        fault: FaultSink,
        sh: Shape,
        q: &[f32],
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
        mma: bool,
    ) -> Result<Vec<f32>, GateError> {
        let m = n_keys.len();
        let qd = DeviceBuffer::from_host(stream, q)?;
        let nk = DeviceBuffer::from_host(stream, n_keys)?;
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len_256(m, sh.n_head, ctx))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, sh.n_head, ctx))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * sh.width())?;
        enqueue_dec(
            k,
            stream,
            sh,
            GqaArgs {
                q: &qd,
                kc,
                vc,
                n_keys: &nk,
                scale: scale(),
                n_kv: N_KV,
                ctx,
                m,
                part_v: &mut pv,
                part_ms: &mut pms,
                fault,
                y: &mut y,
            },
            mma,
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// The decode clause (module doc, 2) at `sh`: `keys` the one-row
    /// launches' counts.
    fn decode_check(
        gpu: &Gpu,
        k: &FlashGqaKernels,
        sh: Shape,
        keys: &[usize],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let w = sh.width();
        let tag = sh.tag();
        let mut ok = true;
        for (i, &live) in keys.iter().enumerate() {
            let ctx = live + PAD;
            let seed = 100 + 10 * u32::try_from(i)?;
            let c = cache(stream, ctx, live, seed)?;
            let q: Vec<f32> = activations(HEAD, sh.n_head, seed + 5)
                .iter()
                .map(|v| v * SEED_Q_SCALE)
                .collect();
            let nk = [u32::try_from(live)?];
            for (pass, mma) in [(Pass::Scalar, false), (Pass::Mma, true)] {
                let y = run_dec(k, stream, unl, sh, &q, &nk, (&c.kc, &c.vc), ctx, mma)?;
                let y2 = run_dec(k, stream, unl, sh, &q, &nk, (&c.kc, &c.vc), ctx, mma)?;
                let yn = run_dec(k, stream, unl, sh, &q, &nk, (&c.kn, &c.vn), ctx, mma)?;
                let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
                let (band, worst) = band_rows(sh, &q, &[live], &c.host, &y, pass);
                let pass_ok = band && rerun && nan_same;
                println!(
                    "decode{tag} pass={} keys={live} ctx={ctx} segments={}: measured/bound \
                     {worst:.3e} band={band} rerun={rerun} nan_padding_same={nan_same} {}",
                    pass.name(),
                    segments_for(ctx),
                    verdict(pass_ok)
                );
                ok &= pass_ok;
            }
        }

        // Eight rows in one launch at counts spread over a 1025-key cache,
        // row t the query with its heads rotated by t.
        let live = 1025usize;
        let ctx = live + PAD;
        let c = cache(stream, ctx, live, 300)?;
        let q1 = activations(HEAD, sh.n_head, 305);
        let rows: Vec<f32> = (0..ROWS)
            .flat_map(|t| {
                let q1 = &q1;
                (0..sh.n_head).flat_map(move |h| {
                    q1[((h + t) % sh.n_head) * HEAD..][..HEAD]
                        .iter()
                        .map(|v| v * SEED_Q_SCALE)
                })
            })
            .collect();
        let counts: Vec<u32> = (0..ROWS)
            .map(|t| u32::try_from(1 + t * (live - 1) / (ROWS - 1)))
            .collect::<Result<_, _>>()?;
        for mma in [false, true] {
            let name = if mma { "mma" } else { "scalar" };
            let all = run_dec(k, stream, unl, sh, &rows, &counts, (&c.kc, &c.vc), ctx, mma)?;
            let mut alone = true;
            for t in 0..ROWS {
                let one = run_dec(
                    k,
                    stream,
                    unl,
                    sh,
                    &rows[t * w..(t + 1) * w],
                    &counts[t..=t],
                    (&c.kc, &c.vc),
                    ctx,
                    mma,
                )?;
                alone &= bits_equal(&all[t * w..(t + 1) * w], &one);
            }
            println!(
                "decode{tag} rows pass={name} m={ROWS} keys={counts:?} ctx={ctx}: each row = its \
                 one-row launch bit for bit {}",
                verdict(alone)
            );
            ok &= alone;

            // Row 3's count past the cache and row 5's zero, labelled.
            let (bad_hi, bad_zero) = (3usize, 5usize);
            let mut bad = counts.clone();
            bad[bad_hi] = u32::try_from(ctx + 1)?;
            bad[bad_zero] = 0;
            let before = gpu.fault()?;
            let yb = run_dec(
                k,
                stream,
                gpu.layer_sink(LAYER)?,
                sh,
                &rows,
                &bad,
                (&c.kc, &c.vc),
                ctx,
                mma,
            )?;
            let raised = gpu.take_fault()?;
            let want = Some(Fault::at(u32::try_from(LAYER)?, FaultSite::KeyCount));
            let mut others = true;
            let mut nan = true;
            for r in 0..ROWS {
                let (a, b) = (&all[r * w..(r + 1) * w], &yb[r * w..(r + 1) * w]);
                if r == bad_hi || r == bad_zero {
                    nan &= b.iter().all(|v| v.is_nan());
                } else {
                    others &= bits_equal(a, b);
                }
            }
            let fault_ok = before.is_none() && raised == want && nan && others;
            println!(
                "decode{tag} fault pass={name}: counts {} and 0 at rows {bad_hi}, {bad_zero}: word \
                 {raised:?} (want {want:?}), those rows NaN {nan}, other rows bit-identical {others} {}",
                ctx + 1,
                verdict(fault_ok)
            );
            ok &= fault_ok;

            // The captured launch: two nodes, the eager bits.
            let qd = DeviceBuffer::from_host(stream, &rows)?;
            let nk = DeviceBuffer::from_host(stream, &counts)?;
            let mut pv =
                DeviceBuffer::<f32>::zeroed(stream, partials_v_len_256(ROWS, sh.n_head, ctx))?;
            let mut pms =
                DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(ROWS, sh.n_head, ctx))?;
            let mut yg = DeviceBuffer::<f32>::zeroed(stream, ROWS * w)?;
            let graph = gpu.capture(|s| {
                enqueue_dec(
                    k,
                    s,
                    sh,
                    GqaArgs {
                        q: &qd,
                        kc: &c.kc,
                        vc: &c.vc,
                        n_keys: &nk,
                        scale: scale(),
                        n_kv: N_KV,
                        ctx,
                        m: ROWS,
                        part_v: &mut pv,
                        part_ms: &mut pms,
                        fault: gpu.unlabelled_sink(),
                        y: &mut yg,
                    },
                    mma,
                )
            })?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let same = bits_equal(&yg.to_host_vec(stream)?, &all);
            let nodes = graph.node_count();
            let graph_ok = same && nodes == 2;
            println!(
                "decode{tag} graph pass={name} m={ROWS}: eager_vs_graph_bit_identical={same} \
                 graph_nodes={nodes} {}",
                verdict(graph_ok)
            );
            ok &= graph_ok;

            // A group the pack-of-four entry is not built for: 26 heads over
            // two key heads, refused by name before any launch.
            if sh.pack4 {
                let n_head = sh.n_head + 2;
                let r = k.enqueue_pass_256_p4(
                    stream,
                    GqaArgs {
                        q: &qd,
                        kc: &c.kc,
                        vc: &c.vc,
                        n_keys: &nk,
                        scale: scale(),
                        n_kv: N_KV,
                        ctx,
                        m: ROWS,
                        part_v: &mut pv,
                        part_ms: &mut pms,
                        fault: gpu.unlabelled_sink(),
                        y: &mut yg,
                    },
                    n_head,
                    mma,
                );
                let named = matches!(
                    r,
                    Err(GpuError::Shape {
                        what: "flash_gqa::enqueue_256_p4",
                        ..
                    })
                );
                println!(
                    "decode{tag} refusal pass={name} n_head={n_head} over n_kv={N_KV}: {} {}",
                    r.err().map_or("accepted".to_string(), |e| e.to_string()),
                    verdict(named)
                );
                ok &= named;
            }
        }
        Ok(ok)
    }

    // ----------------------------------------------------- score order

    /// The tie (module doc, 3): every head's query has equal halves, each key
    /// head's keys 0 and 1 have their halves swapped, the values are
    /// multiples of 1/256 below 8, and both rows see two keys. Returns whether
    /// the decode tensor-core pass and the prefill flash each give `(v_0 +
    /// v_1)/2` in every value.
    fn tie_check(gpu: &Gpu, k: &FlashGqaKernels, kp: &FlashGqaPrefill) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let ctx = 2 + PAD;
        let half = HEAD / 2;
        let q: Vec<f32> = activations(half, N_HEAD, 401)
            .chunks(half)
            .flat_map(|h| h.iter().chain(h.iter()).map(|v| v * SEED_Q_SCALE))
            .collect();
        let mut kb = to16(&activations(HEAD, N_KV * ctx, 402));
        let mut vb: Vec<u16> = activations(HEAD, N_KV * ctx, 403)
            .iter()
            .map(|v| f32_to_f16_bits((v * 2047.0).round() / 256.0))
            .collect();
        for j in 0..N_KV {
            let a = (j * ctx) * HEAD;
            let b = a + HEAD;
            let key_a: Vec<u16> = kb[a..a + HEAD].to_vec();
            kb[b..b + half].copy_from_slice(&key_a[half..]);
            kb[b + half..b + HEAD].copy_from_slice(&key_a[..half]);
            // Rows at or past the count are NaN: they must not be read.
            for r in 2..ctx {
                let at = (j * ctx + r) * HEAD;
                kb[at..at + HEAD].fill(NAN16);
                vb[at..at + HEAD].fill(NAN16);
            }
        }
        let want: Vec<f32> = (0..N_HEAD)
            .flat_map(|h| {
                let a = (h / GROUP) * ctx * HEAD;
                let vb = &vb;
                (0..HEAD)
                    .map(move |d| (half_to_f32(vb[a + d]) + half_to_f32(vb[a + HEAD + d])) * 0.5)
            })
            .collect();
        let kc = DeviceBuffer::from_host(stream, &kb)?;
        let vc = DeviceBuffer::from_host(stream, &vb)?;
        let unl = gpu.unlabelled_sink();
        let dec = run_dec(k, stream, unl, Q36, &q, &[2], (&kc, &vc), ctx, true)?;
        let pre = run_pref(kp, gpu, Q36, &q, &[2], (&kc, &vc), ctx)?;
        let (dec_ok, pre_ok) = (bits_equal(&dec, &want), bits_equal(&pre, &want));
        let pass = dec_ok && pre_ok;
        println!(
            "score order: {N_HEAD} heads, query halves equal, keys 0 and 1 with swapped halves, two \
             live keys: decode mma = (v0 + v1)/2 bit for bit {dec_ok}, prefill = (v0 + v1)/2 bit \
             for bit {pre_ok} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    // --------------------------------------------------------- prefill

    /// One prefill launch of `n_keys.len()` rows through `sh`'s entry, into
    /// fresh output, read back. The fault word is the caller's to read.
    fn run_pref(
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        sh: Shape,
        q: &[f32],
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let qd = DeviceBuffer::from_host(stream, q)?;
        run_pref_dev(kp, gpu, sh, &qd, n_keys, (kc, vc), ctx)
    }

    fn run_pref_dev(
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        sh: Shape,
        q: &DeviceBuffer<f32>,
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let t = n_keys.len();
        let nk = DeviceBuffer::from_host(stream, n_keys)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, t * sh.width())?;
        enqueue_pref(
            kp,
            stream,
            sh,
            GqaPrefillArgs {
                q,
                kc,
                vc,
                n_keys: &nk,
                scale: scale(),
                n_head: sh.n_head,
                n_kv: N_KV,
                ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// Each row of `all` against the same row launched alone. Returns the
    /// rows that differ in any bit.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernels and card, the layout, the rows, the cache and its height, the launch"
    )]
    fn rows_alone(
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        sh: Shape,
        q: &[f32],
        counts: &[u32],
        cache: (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
        all: &[f32],
    ) -> Result<usize, GateError> {
        let w = sh.width();
        let mut differ = 0usize;
        for (t, &c) in counts.iter().enumerate() {
            let one = run_pref(kp, gpu, sh, &q[t * w..(t + 1) * w], &[c], cache, ctx)?;
            differ += usize::from(!bits_equal(&all[t * w..(t + 1) * w], &one));
        }
        Ok(differ)
    }

    /// What the window check of one layout reads: the fault launch's query
    /// and cache bytes, its clean counts and output, and the cache height.
    struct Windows {
        q: Vec<f32>,
        kb: Vec<u16>,
        vb: Vec<u16>,
        clean: Vec<u32>,
        y: Vec<f32>,
        ctx: usize,
    }

    /// The prefill clause (module doc, 4) at `sh`: the seeded launches
    /// `SEED_P0 × SEED_T` and then `extra` (`(p0, T)` pairs), the fault, graph
    /// and head-count refusal checks. Returns the verdict and what
    /// [`misaligned`] reads, which the caller runs last.
    fn prefill_check(
        gpu: &Gpu,
        kp: &FlashGqaPrefill,
        sh: Shape,
        extra: &[(usize, usize)],
    ) -> Result<(bool, Windows), GateError> {
        let stream = gpu.stream();
        let w = sh.width();
        let tag = sh.tag();
        let mut ok = true;
        let mut seed = 500u32;
        let seeded = SEED_P0
            .iter()
            .flat_map(|&p0| SEED_T.iter().map(move |&t| (p0, t)))
            .chain(extra.iter().copied());
        for (p0, t) in seeded {
            seed += 7;
            let ctx = p0 + t + PAD;
            let live = p0 + t;
            let c = cache(stream, ctx, live, seed)?;
            let q: Vec<f32> = activations(HEAD, t * sh.n_head, seed + 3)
                .iter()
                .map(|v| v * SEED_Q_SCALE)
                .collect();
            let counts: Vec<u32> = (0..t)
                .map(|i| u32::try_from(p0 + i + 1))
                .collect::<Result<_, _>>()?;
            let y = run_pref(kp, gpu, sh, &q, &counts, (&c.kc, &c.vc), ctx)?;
            let y2 = run_pref(kp, gpu, sh, &q, &counts, (&c.kc, &c.vc), ctx)?;
            let yn = run_pref(kp, gpu, sh, &q, &counts, (&c.kn, &c.vn), ctx)?;
            let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
            let cu: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
            let (band, worst) = band_rows(sh, &q, &cu, &c.host, &y, Pass::Prefill);
            let differ = rows_alone(kp, gpu, sh, &q, &counts, (&c.kc, &c.vc), ctx, &y)?;
            let pass = rerun && nan_same && band && differ == 0;
            println!(
                "prefill{tag} seeded T={t} p0={p0} ctx={ctx}: measured/bound {worst:.3e} \
                 band={band} rows_alone_differing={differ} rerun={rerun} \
                 nan_padding_same={nan_same} {}",
                verdict(pass)
            );
            ok &= pass;
        }

        // A count of zero and one past the cache, on a 17-row launch at 63.
        let (t, p0) = (17usize, 63usize);
        let ctx = p0 + t + PAD;
        let c = cache(stream, ctx, p0 + t, 91)?;
        let q = activations(HEAD, t * sh.n_head, 94);
        let clean: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i + 1))
            .collect::<Result<_, _>>()?;
        let (bad_hi, bad_zero) = (3usize, 10usize);
        let mut bad = clean.clone();
        bad[bad_hi] = u32::try_from(ctx + 1)?;
        bad[bad_zero] = 0;
        let before = gpu.fault()?;
        let y = run_pref(kp, gpu, sh, &q, &clean, (&c.kc, &c.vc), ctx)?;
        let after_clean = gpu.fault()?;
        let yb = run_pref(kp, gpu, sh, &q, &bad, (&c.kc, &c.vc), ctx)?;
        let raised = gpu.take_fault()?;
        let mut others = true;
        let mut nan = true;
        for r in 0..t {
            let (a, b) = (&y[r * w..(r + 1) * w], &yb[r * w..(r + 1) * w]);
            if r == bad_hi || r == bad_zero {
                nan &= b.iter().all(|v| v.is_nan());
            } else {
                others &= bits_equal(a, b);
            }
        }
        let want = Fault::at(LAYER_NONE, FaultSite::KeyCount);
        let fault_ok =
            before.is_none() && after_clean.is_none() && raised == Some(want) && nan && others;
        println!(
            "prefill{tag} fault: counts {} (past ctx {ctx}) and 0 at rows {bad_hi}, {bad_zero}: \
             word {raised:?} (want {want:?}), those rows NaN {nan}, other rows bit-identical \
             {others} {}",
            ctx + 1,
            verdict(fault_ok)
        );
        ok &= fault_ok;

        // The captured launch: one node, the eager bits.
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let nk = DeviceBuffer::from_host(stream, &clean)?;
        let mut yg = DeviceBuffer::<f32>::zeroed(stream, t * w)?;
        let graph = gpu.capture(|s| {
            enqueue_pref(
                kp,
                s,
                sh,
                GqaPrefillArgs {
                    q: &qd,
                    kc: &c.kc,
                    vc: &c.vc,
                    n_keys: &nk,
                    scale: scale(),
                    n_head: sh.n_head,
                    n_kv: N_KV,
                    ctx,
                    t,
                    fault: gpu.unlabelled_sink(),
                    y: &mut yg,
                },
            )
        })?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let same = bits_equal(&yg.to_host_vec(stream)?, &y);
        let nodes = graph.node_count();
        let graph_ok = same && nodes == 1;
        println!(
            "prefill{tag} graph T={t} p0={p0}: eager_vs_graph_bit_identical={same} \
             graph_nodes={nodes} {}",
            verdict(graph_ok)
        );
        ok &= graph_ok;

        // A head count the kernel is not built for.
        let mut yr = DeviceBuffer::<f32>::zeroed(stream, t * w)?;
        let refused = enqueue_pref(
            kp,
            stream,
            sh,
            GqaPrefillArgs {
                q: &qd,
                kc: &c.kc,
                vc: &c.vc,
                n_keys: &nk,
                scale: scale(),
                n_head: sh.n_head - 1,
                n_kv: N_KV,
                ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut yr,
            },
        );
        let refuse_ok =
            matches!(refused, Err(GpuError::Shape { what, .. }) if what == sh.pref_what());
        println!(
            "prefill{tag} refusal n_head={} over n_kv={N_KV}: {} {}",
            sh.n_head - 1,
            refused
                .err()
                .map_or("accepted".to_string(), |e| e.to_string()),
            verdict(refuse_ok)
        );
        ok &= refuse_ok;
        let win = Windows {
            q,
            kb: c.kb,
            vb: c.vb,
            clean,
            y,
            ctx,
        };
        Ok((ok, win))
    }

    /// The pack-of-four entries against the eight-head ones on Qwen3.6's
    /// layout (two packs a key head): a row's arithmetic does not depend on
    /// the rows or heads that share its block or its tensor-core tile, so both
    /// geometries give every output bit for bit — the decode passes on one-row
    /// launches and the eight-row launch, the prefill on [`SEED_CROSS`].
    fn cross_check(
        gpu: &Gpu,
        k: &FlashGqaKernels,
        kp: &FlashGqaPrefill,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let mut ok = true;
        for (i, &live) in [5usize, 64, 1025].iter().enumerate() {
            let ctx = live + PAD;
            let seed = 700 + 10 * u32::try_from(i)?;
            let c = cache(stream, ctx, live, seed)?;
            let q: Vec<f32> = activations(HEAD, N_HEAD, seed + 5)
                .iter()
                .map(|v| v * SEED_Q_SCALE)
                .collect();
            let counts: Vec<u32> = if live == 1025 {
                (0..ROWS)
                    .map(|t| u32::try_from(1 + t * (live - 1) / (ROWS - 1)))
                    .collect::<Result<_, _>>()?
            } else {
                vec![u32::try_from(live)?]
            };
            let q: Vec<f32> = q
                .iter()
                .cycle()
                .take(counts.len() * q.len())
                .copied()
                .collect();
            for mma in [false, true] {
                let a = run_dec(k, stream, unl, Q36, &q, &counts, (&c.kc, &c.vc), ctx, mma)?;
                let b = run_dec(
                    k,
                    stream,
                    unl,
                    Q36_P4,
                    &q,
                    &counts,
                    (&c.kc, &c.vc),
                    ctx,
                    mma,
                )?;
                let same = bits_equal(&a, &b);
                println!(
                    "cross decode pass={} m={} keys={live}: p4 = group-8 entry bit for bit {same} {}",
                    if mma { "mma" } else { "scalar" },
                    counts.len(),
                    verdict(same)
                );
                ok &= same;
            }
        }
        for (i, &(p0, t)) in SEED_CROSS.iter().enumerate() {
            let ctx = p0 + t + PAD;
            let seed = 800 + 10 * u32::try_from(i)?;
            let c = cache(stream, ctx, p0 + t, seed)?;
            let q: Vec<f32> = activations(HEAD, t * N_HEAD, seed + 3)
                .iter()
                .map(|v| v * SEED_Q_SCALE)
                .collect();
            let counts: Vec<u32> = (0..t)
                .map(|i| u32::try_from(p0 + i + 1))
                .collect::<Result<_, _>>()?;
            let a = run_pref(kp, gpu, Q36, &q, &counts, (&c.kc, &c.vc), ctx)?;
            let b = run_pref(kp, gpu, Q36_P4, &q, &counts, (&c.kc, &c.vc), ctx)?;
            let same = bits_equal(&a, &b);
            println!(
                "cross prefill T={t} p0={p0}: p4 = group-8 entry bit for bit {same} {}",
                verdict(same)
            );
            ok &= same;
        }
        Ok(ok)
    }

    /// `gate_qwen3moe_flash`'s window check at a head of 256, through `sh`'s
    /// entry: aligned windows accepted and bit for bit the plain launch, `q`
    /// at 4 bytes past 8 and `kc`/`vc` at 8 bytes past 16 refused by name
    /// before any launch. The gate's last device checks: a launch through a
    /// misaligned window is a sticky error that ends the context.
    fn misaligned(
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        sh: Shape,
        win: &Windows,
    ) -> Result<bool, GateError> {
        let Windows {
            q,
            kb,
            vb,
            clean,
            y,
            ctx,
        } = win;
        let (q, kb, vb, clean, y, ctx) = (&q[..], &kb[..], &vb[..], &clean[..], &y[..], *ctx);
        let tag = sh.tag();
        let stream = gpu.stream();
        let t = clean.len();
        let qd = DeviceBuffer::from_host(stream, q)?;
        let (kc, vc) = (
            DeviceBuffer::from_host(stream, kb)?,
            DeviceBuffer::from_host(stream, vb)?,
        );
        let nk = DeviceBuffer::from_host(stream, clean)?;
        let q_pad = DeviceBuffer::from_host(stream, &[&[0.0f32; 2][..], q].concat())?;
        let k_pad = DeviceBuffer::from_host(stream, &[&[0u16; 8][..], kb].concat())?;
        let v_pad = DeviceBuffer::from_host(stream, &[&[0u16; 8][..], vb].concat())?;
        let cx = gpu.context();
        // SAFETY: each window is `q.len()` f32 starting one or two f32 into
        // `q_pad`, or `kb.len()` (= `vb.len()`) f16 starting four or eight f16
        // into `k_pad` or `v_pad` — inside its own live allocation, which holds
        // two f32 or eight f16 more than the span. The allocations outlive
        // every call below, and the windows are given back after them.
        let (q_al, q_mis, k_al, k_mis, v_al, v_mis) = unsafe {
            (
                window::<f32>(q_pad.cu_deviceptr() + 8, q.len(), cx),
                window::<f32>(q_pad.cu_deviceptr() + 4, q.len(), cx),
                window::<u16>(k_pad.cu_deviceptr() + 16, kb.len(), cx),
                window::<u16>(k_pad.cu_deviceptr() + 8, kb.len(), cx),
                window::<u16>(v_pad.cu_deviceptr() + 16, vb.len(), cx),
                window::<u16>(v_pad.cu_deviceptr() + 8, vb.len(), cx),
            )
        };
        let ya = run_pref_dev(kp, gpu, sh, &q_al, clean, (&*k_al, &*v_al), ctx)?;
        let aligned_ok = bits_equal(&ya, y);
        println!(
            "prefill{tag} windows at aligned offsets: accepted, bit-identical to the plain \
             launch {aligned_ok} {}",
            verdict(aligned_ok)
        );
        let mut ok = aligned_ok;
        let mut ym = DeviceBuffer::<f32>::zeroed(stream, t * sh.width())?;
        let cases = [
            ("q", "4 bytes past an 8-byte boundary", &*q_mis, &kc, &vc),
            ("kc", "8 bytes past a 16-byte boundary", &qd, &*k_mis, &vc),
            ("vc", "8 bytes past a 16-byte boundary", &qd, &kc, &*v_mis),
        ];
        for (name, off, q_in, kc_in, vc_in) in cases {
            let r = enqueue_pref(
                kp,
                stream,
                sh,
                GqaPrefillArgs {
                    q: q_in,
                    kc: kc_in,
                    vc: vc_in,
                    n_keys: &nk,
                    scale: scale(),
                    n_head: sh.n_head,
                    n_kv: N_KV,
                    ctx,
                    t,
                    fault: gpu.unlabelled_sink(),
                    y: &mut ym,
                },
            );
            let named = matches!(
                r,
                Err(GpuError::Shape {
                    what,
                    ref detail,
                }) if what == sh.pref_what() && detail.starts_with(&format!("{name} at "))
            );
            let seen = match &r {
                _ if named => "refused by name".to_string(),
                Ok(()) => format!(
                    "accepted; the stream then reads {:?}",
                    stream.synchronize().err().map(|e| e.to_string())
                ),
                Err(e) => format!("refused without the name: {e}"),
            };
            println!(
                "prefill{tag} refusal {name} window {off}: {seen} {}",
                verdict(named)
            );
            ok &= named;
        }
        for w in [q_al, q_mis] {
            drop(ManuallyDrop::into_inner(w).into_raw_parts());
        }
        for w in [k_al, k_mis, v_al, v_mis] {
            drop(ManuallyDrop::into_inner(w).into_raw_parts());
        }
        Ok(ok)
    }

    // ------------------------------------------------- gated quantizer

    /// Qwen3.6's q+gate rows: each head's gate the 256 values after its query.
    fn layout() -> GateLayout {
        GateLayout {
            head: HEAD,
            head_stride: QG_HEAD,
            offset: HEAD,
            col_stride: QG_ROW,
        }
    }

    /// `m` columns' q+gate rows whose gates are drawn by `gate(i)` (the
    /// value's index in the column's attention output) and whose query
    /// halves are 1e6 (the quantizer must not read them).
    fn gate_rows(m: usize, gate: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut g = vec![1.0e6f32; m * QG_ROW];
        for c in 0..m {
            for i in 0..QK {
                g[c * QG_ROW + (i / HEAD) * QG_HEAD + HEAD + i % HEAD] = gate(c, i);
            }
        }
        g
    }

    /// The products on the host: `attn · sigmoid(g)` per value.
    fn products(attn: &[f32], g: &[f32], m: usize) -> Vec<f32> {
        (0..m * QK)
            .map(|x| {
                let (c, i) = (x / QK, x % QK);
                attn[x] * sigmoid(g[c * QG_ROW + (i / HEAD) * QG_HEAD + HEAD + i % HEAD])
            })
            .collect()
    }

    /// A launch's q3 codes (as bytes) and scales.
    struct Q8 {
        codes: Vec<i8>,
        d: Vec<f32>,
    }

    fn q8_of(
        stream: &CudaStream,
        q3: &DeviceBuffer<u64>,
        d8: &DeviceBuffer<f32>,
    ) -> Result<Q8, GateError> {
        let codes = q3
            .to_host_vec(stream)?
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .map(|b| b as i8)
            .collect();
        Ok(Q8 {
            codes,
            d: d8.to_host_vec(stream)?,
        })
    }

    /// The gated launch into a fresh `Q8Act` of `m` columns, raising on
    /// `fault`.
    fn gated_q8act(
        gq: &GatedQuantKernels,
        stream: &CudaStream,
        attn: &[f32],
        g: &[f32],
        m: usize,
        fault: FaultSink,
    ) -> Result<Q8, GateError> {
        let x = DeviceBuffer::from_host(stream, attn)?;
        let gd = DeviceBuffer::from_host(stream, g)?;
        let mut act = Q8Act::with_k(stream, m, QK)?;
        gq.enqueue_q8act(stream, (&x, &gd), layout(), &mut act, m, fault)?;
        stream.synchronize()?;
        q8_of(stream, act.q3(), act.d8())
    }

    /// The plain quantizer on `x` into a fresh `Q8Act` of `m` columns.
    fn plain_q8act(gpu: &Gpu, x: &[f32], m: usize) -> Result<Q8, GateError> {
        let stream = gpu.stream();
        let xd = DeviceBuffer::from_host(stream, x)?;
        let mut act = Q8Act::with_k(stream, m, QK)?;
        gpu.enqueue_quantize_q8_1(&xd, &mut act)?;
        stream.synchronize()?;
        q8_of(stream, act.q3(), act.d8())
    }

    /// Codes and scales equal bit for bit.
    fn q8_same(a: &Q8, b: &Q8) -> bool {
        a.codes == b.codes && bits_equal(&a.d, &b.d)
    }

    fn gated_check(gpu: &Gpu, gq: &GatedQuantKernels) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let mut ok = true;
        let exact_g = |c: usize, i: usize| -> f32 {
            match (i * 7 + c * 3) % 5 {
                0 | 3 => 0.0,
                1 => 100.0,
                2 => -100.0,
                _ => 0.0,
            }
        };
        let spread_g = |c: usize, i: usize| -> f32 {
            let s = ((i * 2_654_435_761usize + c * 40_503) >> 7) & 0xffff;
            (s as f32 / 32_768.0 - 1.0) * 6.0
        };
        for m in [1usize, 5, 8] {
            let attn = activations(QK, m, 700 + u32::try_from(m)?);
            let g = gate_rows(m, exact_g);
            let got = gated_q8act(gq, stream, &attn, &g, m, unl)?;
            let want = plain_q8act(gpu, &products(&attn, &g, m), m)?;
            let same = q8_same(&got, &want);
            println!(
                "gated quant q8act m={m} gates in {{0, 100, -100}}: codes and scales = the plain \
                 quantizer's on attn·sigmoid(g) bit for bit {same} {}",
                verdict(same)
            );
            ok &= same;

            let g = gate_rows(m, spread_g);
            let got = gated_q8act(gq, stream, &attn, &g, m, unl)?;
            let want = plain_q8act(gpu, &products(&attn, &g, m), m)?;
            let codes_differ = got
                .codes
                .iter()
                .zip(&want.codes)
                .filter(|(a, b)| a != b)
                .count();
            let codes_ok = got.codes.len() == want.codes.len()
                && got
                    .codes
                    .iter()
                    .zip(&want.codes)
                    .all(|(&a, &b)| (i16::from(a) - i16::from(b)).abs() <= 1);
            let d_worst = got
                .d
                .iter()
                .zip(&want.d)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs() / f64::from(b).abs())
                .fold(0.0f64, f64::max);
            let band = gamma(10);
            let pass = codes_ok && d_worst <= band;
            println!(
                "gated quant q8act m={m} gates in [-6, 6]: codes within 1 {codes_ok} \
                 ({codes_differ} of {} differ), scales rel {d_worst:.3e} <= {band:.3e} {}",
                got.codes.len(),
                verdict(pass)
            );
            ok &= pass;
        }

        // The GEMM activation: the same bytes as the plain GEMM quantizer
        // on the products, at the exact gates.
        let m = GEMM_COLS;
        let attn = activations(QK, m, 750);
        let g = gate_rows(m, exact_g);
        let (x, gd) = (
            DeviceBuffer::from_host(stream, &attn)?,
            DeviceBuffer::from_host(stream, &g)?,
        );
        let mut act = GemmAct::new(stream, m, QK)?;
        gq.enqueue_gemm(stream, (&x, &gd), layout(), &mut act, m, unl)?;
        let xp = DeviceBuffer::from_host(stream, &products(&attn, &g, m))?;
        let mut plain = GemmAct::new(stream, m, QK)?;
        gpu.enqueue_quantize_gemm(&xp, m, &mut plain, unl)?;
        stream.synchronize()?;
        let same = act.q3().to_host_vec(stream)? == plain.q3().to_host_vec(stream)?
            && act.q4().to_host_vec(stream)? == plain.q4().to_host_vec(stream)?
            && act.q6().to_host_vec(stream)? == plain.q6().to_host_vec(stream)?
            && act.s8().to_host_vec(stream)? == plain.s8().to_host_vec(stream)?
            && bits_equal(
                &act.d8().to_host_vec(stream)?,
                &plain.d8().to_host_vec(stream)?,
            );
        println!(
            "gated quant gemm cols={m} gates in {{0, 100, -100}}: the five planes = the plain GEMM \
             quantizer's on attn·sigmoid(g) bit for bit {same} {}",
            verdict(same)
        );
        ok &= same;

        // A NaN gate in column 0's block 3 and an infinite gate in column
        // 4's block 20: those blocks refused, the fault raised, the rest the
        // clean run's.
        let m = 5usize;
        let attn = activations(QK, m, 780);
        let clean_g = gate_rows(m, spread_g);
        let clean = gated_q8act(gq, stream, &attn, &clean_g, m, unl)?;
        let mut bad_g = clean_g.clone();
        let (c0, b0, c1, b1) = (0usize, 3usize, 4usize, 20usize);
        let at = |c: usize, b: usize| {
            let i = 128 * b + 37;
            c * QG_ROW + (i / HEAD) * QG_HEAD + HEAD + i % HEAD
        };
        bad_g[at(c0, b0)] = f32::NAN;
        bad_g[at(c1, b1)] = f32::INFINITY;
        let before = gpu.fault()?;
        let got = gated_q8act(gq, stream, &attn, &bad_g, m, gpu.layer_sink(LAYER)?)?;
        let raised = gpu.take_fault()?;
        let want = Some(Fault::at(u32::try_from(LAYER)?, FaultSite::QuantColumn));
        let n_blocks = QK / 128;
        let mut refused = true;
        let mut others = true;
        for c in 0..m {
            for b in 0..n_blocks {
                let di = c * n_blocks + b;
                let bad = (c, b) == (c0, b0) || (c, b) == (c1, b1);
                if bad {
                    refused &= got.d[di].is_nan();
                } else {
                    others &= got.d[di].to_bits() == clean.d[di].to_bits();
                }
            }
        }
        // The q3 codes of a column are permuted within groups of four blocks
        // (512 bytes): a group without a refused block holds the clean run's
        // bytes, and a group with one holds at least that block's 128 zero
        // codes.
        let group = 512usize;
        let mut codes_ok = got.codes.len() == clean.codes.len();
        for (gi, (a, b)) in got
            .codes
            .chunks(group)
            .zip(clean.codes.chunks(group))
            .enumerate()
        {
            let (c, g) = (gi / (n_blocks / 4), gi % (n_blocks / 4));
            let bad = (c == c0 && g == b0 / 4) || (c == c1 && g == b1 / 4);
            codes_ok &= if bad {
                a.iter().filter(|&&x| x == 0).count() >= 128
            } else {
                a == b
            };
        }
        let fault_ok = before.is_none() && raised == want && refused && others && codes_ok;
        println!(
            "gated quant fault: a NaN gate in column {c0} block {b0}, +inf in column {c1} block \
             {b1}, layer {LAYER}: word {raised:?} (want {want:?}), those blocks' scales NaN \
             {refused}, other blocks' scales bit-identical {others}, codes of the other groups \
             bit-identical and the refused blocks' zero {codes_ok} {}",
            verdict(fault_ok)
        );
        ok &= fault_ok;
        Ok(ok)
    }

    // ----------------------------------------------------------- ik

    /// The largest `|ours − ik| / max|ik head|` over heads of [`HEAD`] values
    /// (`ours` and `ik` token-major, head after head), and how many values
    /// are bit-identical. `dims` picks the values compared in each head.
    fn head_rel(ours: &[f32], ik: &[f32], dims: std::ops::Range<usize>) -> (f32, usize, usize) {
        let (mut worst, mut same, mut n) = (0.0f32, 0usize, 0usize);
        for (a, b) in ours.chunks(HEAD).zip(ik.chunks(HEAD)) {
            let m = b.iter().fold(0.0f32, |x, v| x.max(v.abs()));
            for d in dims.clone() {
                worst = worst.max((a[d] - b[d]).abs() / m);
                same += usize::from(a[d].to_bits() == b[d].to_bits());
                n += 1;
            }
        }
        (worst, same, n)
    }

    /// Clause 6 (module doc).
    fn model_check(
        gpu: &Gpu,
        rope: &RopeNeoxKernels,
        gq: &GatedQuantKernels,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let split = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        if split.architecture() != Some("qwen35moe") {
            return Err(format!("{MODEL} is {:?}, not qwen35moe", split.architecture()).into());
        }
        let l = MODEL_LAYER;
        let tensor = |name: String| -> Result<Vec<f32>, GateError> {
            Ok(ref_tensor_logical_in(&man.dir, man.tensor(&name, 0)?)?)
        };
        let qaux = tensor(format!("Qaux-{l}"))?;
        let m = qaux.len() / QG_ROW;
        if qaux.len() != m * QG_ROW || m == 0 {
            return Err(format!(
                "Qaux-{l} holds {} values, not whole rows of {QG_ROW}",
                qaux.len()
            )
            .into());
        }
        // ggml's multi-section positions: the first stream is the text position.
        let pos: Vec<u32> = ref_ints(&man, "inp_pos", 0, RowKind::Input, Layout::Flat)?[..m]
            .iter()
            .map(|&p| u32::try_from(p))
            .collect::<Result<_, _>>()?;
        let ctx = ROPE_CTX;
        let table_rt = RopeTable::new(&RopeSpec::window(THETA, ROT))?;
        let mut table = Vec::with_capacity(ctx * ROT);
        for p in 0..ctx {
            table_rt.push(u32::try_from(p)?, Direction::Forward, &mut table);
        }
        let inp = RopeIn {
            qg: qaux.clone(),
            k: tensor(format!("Kcur-{l}"))?,
            v: tensor(format!("Vcur-{l}"))?,
            gq: split_f32(&split, &format!("blk.{l}.attn_q_norm.weight"), HEAD)?,
            gk: split_f32(&split, &format!("blk.{l}.attn_k_norm.weight"), HEAD)?,
            table,
        };
        let out = rope_run(rope, stream, &inp, &pos, gpu.unlabelled_sink())?;
        let (q_roped, q_normed) = (
            tensor(format!("Qcur_roped-{l}"))?,
            tensor(format!("Qcur_normed-{l}"))?,
        );
        let (k_roped, k_normed) = (
            tensor(format!("Kcur_roped-{l}"))?,
            tensor(format!("Kcur_normed-{l}"))?,
        );
        let parts = [
            ("Qcur_roped", head_rel(&out.q, &q_roped, 0..ROT)),
            ("Qcur_normed", head_rel(&out.q, &q_normed, ROT..HEAD)),
            ("Kcur_roped", head_rel(&out.k, &k_roped, 0..ROT)),
            ("Kcur_normed", head_rel(&out.k, &k_normed, ROT..HEAD)),
        ];
        let mut ok = true;
        for (tap, (rel, same, n)) in parts {
            let pass = rel <= ROPE_BAND;
            println!(
                "ik layer={l} m={m} pos={pos:?} rope vs {tap}-{l}: rel {rel:.3e} (band \
                 {ROPE_BAND:.3e}) same={same}/{n} {}",
                verdict(pass)
            );
            ok &= pass;
        }

        // The gated quantizer on ik's attention output and gates against the
        // plain quantizer on ik's gated product.
        let fa = tensor(format!("fa-{l}"))?;
        let gated = tensor(format!("qkv_gated-{l}"))?;
        if fa.len() != m * QK || gated.len() != m * QK {
            return Err(format!("fa-{l} / qkv_gated-{l} are not {m} rows of {QK}").into());
        }
        let host = products(&fa, &qaux, m);
        let prod_rel = host
            .iter()
            .zip(&gated)
            .map(|(&a, &b)| {
                if b == 0.0 {
                    (a - b).abs()
                } else {
                    ((a - b) / b).abs()
                }
            })
            .fold(0.0f32, f32::max);
        let got = gated_q8act(gq, stream, &fa, &qaux, m, gpu.unlabelled_sink())?;
        let want = plain_q8act(gpu, &gated, m)?;
        let codes_differ = got
            .codes
            .iter()
            .zip(&want.codes)
            .filter(|(a, b)| a != b)
            .count();
        let codes_ok = got.codes.len() == want.codes.len()
            && got
                .codes
                .iter()
                .zip(&want.codes)
                .all(|(&a, &b)| (i16::from(a) - i16::from(b)).abs() <= 1);
        let d_worst = got
            .d
            .iter()
            .zip(&want.d)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs() / f64::from(b).abs())
            .fold(0.0f64, f64::max);
        let band = gamma(10);
        let pass = codes_ok && d_worst <= band;
        println!(
            "ik layer={l} m={m} gated quant on fa-{l} and Qaux-{l}'s gates vs the plain quantizer on \
             qkv_gated-{l}: codes within 1 {codes_ok} ({codes_differ} of {} differ), scales rel \
             {d_worst:.3e} <= {band:.3e}; host attn·sigmoid(g) vs qkv_gated rel {prod_rel:.3e} \
             (printed) {}",
            got.codes.len(),
            verdict(pass)
        );
        ok &= pass;
        Ok(ok)
    }

    // ------------------------------------- the token-pool selector (QSA)

    /// Qwen3.8's selector: `attention.indexer.top_k` 2,048 tokens in pools of
    /// four — 512 pools kept, lists of at most 2,051 tokens.
    const IDX_TOP_K: u32 = 2048;
    const KEPT: usize = IDX_TOP_K as usize / POOL;
    const WIDTH: usize = list_width(KEPT);
    const _: () = assert!(KEPT == 512 && WIDTH == 2051);
    /// The selection clauses' cache height and the largest count they run.
    const SEL_CTX: usize = 8200;
    const SEL_N: usize = 8193;
    /// One-row counts of the selection clauses: the last dense count, the
    /// first that drops a pool, two deep ones.
    const SEL_COUNTS: [usize; 4] = [2051, 2052, 4097, 8193];
    /// The eight rows of a draft's verify at the deepest count.
    const SEL_ROWS0: usize = SEL_N - 7;
    /// One-row counts of the dense-equality clause: one key, a tile edge, and
    /// the last dense count.
    const SEL_DENSE: [usize; 4] = [1, 5, 512, 2051];
    /// The raw keys' and the indexer queries' scale over [`activations`].
    const IDX_SCALE: f32 = 2.0;
    /// The pools whose raw rows the tie clause zeroes: 600 pools, so at count
    /// 4,097 (1,024 pools) the 512 kept end inside their shared score +0.
    const TIE_POOLS: std::ops::Range<usize> = 100..700;
    const TIE_COUNT: usize = 4097;
    /// The cache height of the pool clause's crafted input: 128 pools, every
    /// one complete.
    const ORDER_CTX: usize = 512;

    fn qsa_shape() -> Qsa {
        Qsa::new(IDX_TOP_K, u32::try_from(POOL).unwrap_or(0)).expect("Qwen3.8's selector shape")
    }

    /// The rope table at the indexer's turned width (the main attention's):
    /// `ctx` rows of [`IDX_ROT`].
    fn idx_table(ctx: usize) -> Result<Vec<f32>, GateError> {
        let rt = RopeTable::new(&RopeSpec::window(THETA, IDX_ROT))?;
        let mut t = Vec::with_capacity(ctx * IDX_ROT);
        for p in 0..ctx {
            rt.push(u32::try_from(p)?, Direction::Forward, &mut t);
        }
        Ok(t)
    }

    /// One indexer head's norm and turn, the kernels' rule (`qsa` module
    /// doc): lane `l`'s four squares (`l, l + 32` then `l + 64, l + 96`)
    /// summed in f64 as two pairs, the butterfly, `(sum / 128) as f32`, the
    /// scale, `(scale · gain) · x`, and the NEOX turn of pairs `(i, i + 32)`
    /// by `cs`, the position's table row.
    fn idx_rule(x: &[f32], gain: &[f32], cs: &[f32]) -> Vec<f32> {
        let lanes: [f64; 32] = std::array::from_fn(|l| {
            let sq = |d: usize| f64::from(x[d] * x[d]);
            (sq(l) + sq(l + 32)) + (sq(l + 64) + sq(l + 96))
        });
        let mean = (butterfly(lanes) / IDX_DIM as f64) as f32;
        let scale = 1.0 / (mean + EPS).sqrt();
        let mut y: Vec<f32> = x.iter().zip(gain).map(|(&v, &g)| (scale * g) * v).collect();
        for i in 0..IDX_ROT / 2 {
            let (c, s) = (cs[2 * i], cs[2 * i + 1]);
            let (x0, x1) = (y[i], y[i + IDX_ROT / 2]);
            y[i] = x0.mul_add(c, -(x1 * s));
            y[i + IDX_ROT / 2] = x0.mul_add(s, x1 * c);
        }
        y
    }

    /// A pool's four raw values summed in the kernel's order, left to right.
    fn in_order(r: [f32; 4]) -> f32 {
        ((r[0] + r[1]) + r[2]) + r[3]
    }

    /// Two other orders, which [`order_inputs`] must tell from [`in_order`].
    fn pairwise(r: [f32; 4]) -> f32 {
        (r[0] + r[1]) + (r[2] + r[3])
    }

    fn reversed(r: [f32; 4]) -> f32 {
        ((r[3] + r[2]) + r[1]) + r[0]
    }

    /// Pool `j`'s key, the kernel's rule when `sum` is [`in_order`]: the sum
    /// of the four raw rows times 0.25 in f32, [`idx_rule`] at the pool's
    /// first position, rounded to f16.
    fn pool_rule(
        raw: &[u16],
        j: usize,
        gain: &[f32],
        table: &[f32],
        sum: fn([f32; 4]) -> f32,
    ) -> Vec<u16> {
        let r = |row: usize, d: usize| half_to_f32(raw[(POOL * j + row) * IDX_DIM + d]);
        let x: Vec<f32> = (0..IDX_DIM)
            .map(|d| sum([r(0, d), r(1, d), r(2, d), r(3, d)]) * 0.25)
            .collect();
        let p = POOL * j;
        to16(&idx_rule(&x, gain, &table[p * IDX_ROT..(p + 1) * IDX_ROT]))
    }

    /// Pool `j`'s key in f64 from the references' rule (ex `pool_keys_ref`,
    /// ik `qwen4exp_qsa_mask`): the mean, the RMS gain norm, the NEOX turn of
    /// the first 64 values at the pool's first position; and the norm's scale.
    fn pool_f64(raw: &[u16], j: usize, gain: &[f32], table: &[f32]) -> (Vec<f64>, f64) {
        let x: Vec<f64> = (0..IDX_DIM)
            .map(|d| {
                (0..POOL)
                    .map(|row| f64::from(half_to_f32(raw[(POOL * j + row) * IDX_DIM + d])))
                    .sum::<f64>()
                    / POOL as f64
            })
            .collect();
        let ms = x.iter().map(|v| v * v).sum::<f64>() / IDX_DIM as f64;
        let scale = 1.0 / (ms + f64::from(EPS)).sqrt();
        let mut y: Vec<f64> = x
            .iter()
            .zip(gain)
            .map(|(&v, &g)| scale * f64::from(g) * v)
            .collect();
        let cs = &table[POOL * j * IDX_ROT..][..IDX_ROT];
        for i in 0..IDX_ROT / 2 {
            let (c, s) = (f64::from(cs[2 * i]), f64::from(cs[2 * i + 1]));
            let (x0, x1) = (y[i], y[i + IDX_ROT / 2]);
            y[i] = x0 * c - x1 * s;
            y[i + IDX_ROT / 2] = x0 * s + x1 * c;
        }
        (y, scale)
    }

    /// The selector's inputs: raw indexer keys (`ctx` rows of 128 f16, the
    /// rows of [`TIE_POOLS`] zero when `tie`), the key and query gains, the
    /// rope table; on the host and on the device.
    struct Idx {
        raw: Vec<u16>,
        gk: Vec<f32>,
        gq: Vec<f32>,
        table: Vec<f32>,
        ctx: usize,
        raw_d: DeviceBuffer<u16>,
        gk_d: DeviceBuffer<f32>,
        gq_d: DeviceBuffer<f32>,
        table_d: DeviceBuffer<f32>,
    }

    fn idx_inputs(stream: &CudaStream, ctx: usize, seed: u32, tie: bool) -> Result<Idx, GateError> {
        let mut raw = to16(
            &activations(IDX_DIM, ctx, seed)
                .iter()
                .map(|v| v * IDX_SCALE)
                .collect::<Vec<_>>(),
        );
        if tie {
            raw[TIE_POOLS.start * POOL * IDX_DIM..TIE_POOLS.end * POOL * IDX_DIM].fill(0);
        }
        let gain = |s: u32| -> Vec<f32> {
            activations(IDX_DIM, 1, s)
                .iter()
                .map(|v| 1.2 + 0.4 * v)
                .collect()
        };
        let (gk, gq) = (gain(seed + 1), gain(seed + 2));
        let table = idx_table(ctx)?;
        Ok(Idx {
            raw_d: DeviceBuffer::from_host(stream, &raw)?,
            gk_d: DeviceBuffer::from_host(stream, &gk)?,
            gq_d: DeviceBuffer::from_host(stream, &gq)?,
            table_d: DeviceBuffer::from_host(stream, &table)?,
            raw,
            gk,
            gq,
            table,
            ctx,
        })
    }

    /// The pool clause's crafted input: [`idx_inputs`]' at [`ORDER_CTX`] with,
    /// in every pool, each even unturned value holding `(s·2^k, s·2^(k−24),
    /// s·2^(k−24), −s·2^k)` over the four rows, `k` cycling 10..=15 and `s`
    /// alternating. [`idx_inputs`]' four-row sums are exact in f32 in any
    /// order; these are not: left to right the mean is 0, pairwise
    /// `s·2^(k−26)`, reversed `s·2^(k−25)`.
    fn order_inputs(stream: &CudaStream) -> Result<Idx, GateError> {
        let mut idx = idx_inputs(stream, ORDER_CTX, 501, false)?;
        for j in 0..ORDER_CTX / POOL {
            let dims = (IDX_ROT..IDX_DIM).step_by(2).zip((10..16).cycle());
            for (i, (d, k)) in dims.enumerate() {
                let s = if (i + j) % 2 == 0 { 1.0f32 } else { -1.0 };
                let (a, e) = (s * 2f32.powi(k), s * 2f32.powi(k - 24));
                for (row, v) in [a, e, e, -a].into_iter().enumerate() {
                    idx.raw[(POOL * j + row) * IDX_DIM + d] = f32_to_f16_bits(v);
                }
            }
        }
        idx.raw_d = DeviceBuffer::from_host(stream, &idx.raw)?;
        Ok(idx)
    }

    /// The pooled plane after the pool pass over counts `1 ..= n` in
    /// launches of `chunk` rows, from a plane of [`SENTINEL`]: the device
    /// plane and its host copy.
    fn pool_plane(
        qk: &QsaKernels,
        stream: &CudaStream,
        idx: &Idx,
        n: usize,
        chunk: usize,
        fault: FaultSink,
    ) -> Result<(DeviceBuffer<u16>, Vec<u16>), GateError> {
        let mut pooled =
            DeviceBuffer::from_host(stream, &vec![SENTINEL; pools_for(idx.ctx) * IDX_DIM])?;
        let mut c0 = 1usize;
        while c0 <= n {
            let counts: Vec<u32> = (c0..=n.min(c0 + chunk - 1))
                .map(u32::try_from)
                .collect::<Result<_, _>>()?;
            let nk = DeviceBuffer::from_host(stream, &counts)?;
            qk.enqueue_pool(
                stream,
                PoolArgs {
                    raw: &idx.raw_d,
                    gain: &idx.gk_d,
                    table: &idx.table_d,
                    n_keys: &nk,
                    eps: EPS,
                    ctx: idx.ctx,
                    m: counts.len(),
                    fault,
                    pooled: &mut pooled,
                },
            )?;
            stream.synchronize()?;
            c0 += chunk;
        }
        let host = pooled.to_host_vec(stream)?;
        Ok((pooled, host))
    }

    /// What one selection launch pair leaves, read back for its `m` rows.
    struct Sel {
        q_out: Vec<f32>,
        scores: Vec<f32>,
        list: Vec<u32>,
        n_sel: Vec<u32>,
    }

    /// One selection (score and top-k) of the rows `counts` with raw queries
    /// `q` (`[m][4][128]`) into `scratch`, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernels, the stream and sink, the inputs, the plane, the rows and the scratch"
    )]
    fn run_select(
        qk: &QsaKernels,
        stream: &CudaStream,
        fault: FaultSink,
        idx: &Idx,
        pooled: &DeviceBuffer<u16>,
        q: &[f32],
        counts: &[u32],
        scratch: &mut QsaScratch,
    ) -> Result<Sel, GateError> {
        let m = counts.len();
        let qd = DeviceBuffer::from_host(stream, q)?;
        let nk = DeviceBuffer::from_host(stream, counts)?;
        qk.enqueue_select(
            stream,
            SelectArgs {
                q: &qd,
                gain: &idx.gq_d,
                table: &idx.table_d,
                n_keys: &nk,
                pooled,
                eps: EPS,
                ctx: idx.ctx,
                kept: KEPT,
                m,
                fault,
                scratch,
            },
        )?;
        stream.synchronize()?;
        let pools = pools_for(idx.ctx);
        let take = |v: Vec<f32>, n: usize| v[..n].to_vec();
        Ok(Sel {
            q_out: take(scratch.q_out.to_host_vec(stream)?, m * IDX_HEADS * IDX_DIM),
            scores: take(scratch.scores.to_host_vec(stream)?, m * pools),
            list: scratch.list.to_host_vec(stream)?[..m * WIDTH].to_vec(),
            n_sel: scratch.n_sel.to_host_vec(stream)?[..m].to_vec(),
        })
    }

    /// Row `t`'s list as the launch wrote it.
    fn row_list(s: &Sel, t: usize) -> &[u32] {
        &s.list[t * WIDTH..t * WIDTH + (s.n_sel[t] as usize).min(WIDTH)]
    }

    /// Row `t`'s list by `runtime::qsa`'s rule on the launch's own scores: the
    /// identity below the dense edge, else the kept pools (the lower on a tie)
    /// and the tail; empty for a count of zero or past the cache.
    fn want_list(s: &Sel, t: usize, count: usize, ctx: usize) -> Vec<u32> {
        if count == 0 || count > ctx {
            return Vec::new();
        }
        let pools = pools_for(ctx);
        let nb = count / POOL;
        qsa_select(qsa_shape(), count, &s.scores[t * pools..t * pools + nb])
    }

    /// The seeded raw indexer queries of `m` rows.
    fn idx_queries(m: usize, seed: u32) -> Vec<f32> {
        activations(IDX_DIM, m * IDX_HEADS, seed)
            .iter()
            .map(|v| v * IDX_SCALE)
            .collect()
    }

    /// Every row's selection checks against the host: the query heads bit for
    /// bit [`idx_rule`]'s (NaN for a refused count), each scored pool's score
    /// within its bound of the f64 score on the launch's own heads and pooled
    /// keys, and the list and its length the rule's on the launch's scores.
    /// Returns `(heads, scores, lists, largest measured/bound)`.
    fn select_rows_ok(
        idx: &Idx,
        pooled: &[u16],
        q: &[f32],
        counts: &[usize],
        s: &Sel,
    ) -> (bool, bool, bool, f64) {
        let pools = pools_for(idx.ctx);
        let (mut heads, mut sc, mut lists) = (true, true, true);
        let mut worst = 0.0f64;
        for (t, &c) in counts.iter().enumerate() {
            for h in 0..IDX_HEADS {
                let at = (t * IDX_HEADS + h) * IDX_DIM;
                let got = &s.q_out[at..at + IDX_DIM];
                if c == 0 || c > idx.ctx {
                    heads &= got.iter().all(|v| v.is_nan());
                } else {
                    let p = c - 1;
                    let want = idx_rule(
                        &q[at..at + IDX_DIM],
                        &idx.gq,
                        &idx.table[p * IDX_ROT..(p + 1) * IDX_ROT],
                    );
                    heads &= bits_equal(got, &want);
                }
            }
            if qsa_shape().scored(c, idx.ctx) {
                // PIN(2026-09-27): a head's dot is four rotating chains of 32 fused multiply-adds, then two adds: γ(34) of Σ|q·k|; relu moves no error; the heads' three adds γ(3).
                for j in 0..c / POOL {
                    let key = &pooled[j * IDX_DIM..(j + 1) * IDX_DIM];
                    let (mut s64, mut bound) = (0.0f64, 0.0f64);
                    for h in 0..IDX_HEADS {
                        let qh = &s.q_out[(t * IDX_HEADS + h) * IDX_DIM..][..IDX_DIM];
                        let (mut dot, mut abs) = (0.0f64, 0.0f64);
                        for (a, &b) in qh.iter().zip(key) {
                            let p = f64::from(*a) * f64::from(half_to_f32(b));
                            dot += p;
                            abs += p.abs();
                        }
                        s64 += dot.max(0.0);
                        bound += gamma(34) * abs;
                    }
                    bound += gamma(3) * (s64 + bound);
                    let e = (f64::from(s.scores[t * pools + j]) - s64).abs();
                    worst = worst.max(e / bound.max(f64::MIN_POSITIVE));
                    sc &= e <= bound;
                }
            }
            let want = want_list(s, t, c, idx.ctx);
            lists &= s.n_sel[t] as usize == want.len() && row_list(s, t) == want.as_slice();
        }
        (heads, sc, lists, worst)
    }

    /// Pools `0..complete` of the plane `plane` against [`pool_rule`] bit for
    /// bit and within their band of [`pool_f64`]: `(bit exact, largest
    /// measured/bound)`.
    fn pool_keys_ok(idx: &Idx, plane: &[u16], complete: usize) -> (bool, f64) {
        let mut exact = true;
        let mut worst = 0.0f64;
        for j in 0..complete {
            let got = &plane[j * IDX_DIM..(j + 1) * IDX_DIM];
            exact &= got == pool_rule(&idx.raw, j, &idx.gk, &idx.table, in_order).as_slice();
            let (y, scale64) = pool_f64(&idx.raw, j, &idx.gk, &idx.table);
            let ymax = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            // The mean's three adds err by γ(3) of the rows' absolute sum, which
            // the norm scales up where the rows cancel.
            let rsum = (0..IDX_DIM)
                .map(|d| {
                    (0..POOL)
                        .map(|r| {
                            f64::from(half_to_f32(idx.raw[(POOL * j + r) * IDX_DIM + d]).abs())
                        })
                        .sum::<f64>()
                        / POOL as f64
                })
                .fold(0.0f64, f64::max);
            let gmax = idx
                .gk
                .iter()
                .fold(0.0f64, |m, &g| m.max(f64::from(g).abs()));
            let mean_term = 2.0 * gamma(3) * scale64 * gmax * rsum;
            for (d, &g) in got.iter().enumerate() {
                // PIN(2026-09-27): the f16 rounding (2^-11 relative, 2^-25 below f16's normal range); the mean's adds (γ(3) of the rows' absolute sum, scaled by the norm, both values of a turned pair); the rest of the f32 path — the norm's scale, two products, the turn's two roundings — under 16u of the head's largest.
                let bound =
                    2f64.powi(-11) * y[d].abs() + 2f64.powi(-25) + mean_term + 16.0 * U * ymax;
                let e = (f64::from(half_to_f32(g)) - y[d]).abs();
                worst = worst.max(e / bound);
            }
        }
        (exact, worst)
    }

    /// The pool clause (module doc, 9).
    fn pool_check(gpu: &Gpu, qk: &QsaKernels) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let idx = idx_inputs(stream, SEL_CTX, 501, false)?;
        let complete = SEL_N / POOL;
        let (_, a) = pool_plane(qk, stream, &idx, SEL_N, SEL_N, unl)?;
        let (_, b) = pool_plane(qk, stream, &idx, SEL_N, SEL_N, unl)?;
        let (exact, worst) = pool_keys_ok(&idx, &a, complete);
        let band = worst <= 1.0;
        let untouched = a[complete * IDX_DIM..].iter().all(|&h| h == SENTINEL);
        let rerun = a == b;
        // Launches of seven rows: every pool is written by the one row that
        // completes it, whatever the launch it sits in.
        let small = 701usize;
        let (_, c) = pool_plane(qk, stream, &idx, small, 7, unl)?;
        let chunked = c[..small / POOL * IDX_DIM] == a[..small / POOL * IDX_DIM]
            && c[small / POOL * IDX_DIM..].iter().all(|&h| h == SENTINEL);
        // The crafted input, where the order of the mean's adds moves bits.
        let ord = order_inputs(stream)?;
        let crafted = ORDER_CTX / POOL;
        let (_, o) = pool_plane(qk, stream, &ord, ORDER_CTX, ORDER_CTX, unl)?;
        let (o_exact, o_worst) = pool_keys_ok(&ord, &o, crafted);
        let o_band = o_worst <= 1.0;
        let key =
            |j: usize, sum: fn([f32; 4]) -> f32| pool_rule(&ord.raw, j, &ord.gk, &ord.table, sum);
        let seen = (0..crafted)
            .filter(|&j| {
                let k = key(j, in_order);
                k != key(j, pairwise) && k != key(j, reversed)
            })
            .count();
        let pass =
            exact && band && untouched && rerun && chunked && o_exact && o_band && seen == crafted;
        println!(
            "qsa pool counts 1..={SEL_N} ctx={SEL_CTX}: {complete} pools bit_exact_host={exact} \
             f64 measured/bound {worst:.3e} band={band} incomplete pools untouched={untouched} \
             rerun={rerun} seven-row launches the same plane={chunked}; crafted \
             ctx={ORDER_CTX}: {crafted} pools bit_exact_host={o_exact} f64 measured/bound \
             {o_worst:.3e} band={o_band}, pairwise and reversed sums move {seen} of {crafted} \
             keys {}",
            verdict(pass)
        );

        // A NaN in one raw row of pool 5.
        let mut bad = idx_inputs(stream, SEL_CTX, 501, false)?;
        bad.raw[(POOL * 5 + 2) * IDX_DIM + 17] = NAN16;
        bad.raw_d = DeviceBuffer::from_host(stream, &bad.raw)?;
        let before = gpu.fault()?;
        let (_, r) = pool_plane(qk, stream, &bad, SEL_N, SEL_N, gpu.layer_sink(LAYER)?)?;
        let raised = gpu.take_fault()?;
        let want = Some(Fault::at(u32::try_from(LAYER)?, FaultSite::PoolSelect));
        let nan = r[5 * IDX_DIM..6 * IDX_DIM]
            .iter()
            .any(|&h| !half_to_f32(h).is_finite());
        let others = r[..5 * IDX_DIM] == a[..5 * IDX_DIM] && r[6 * IDX_DIM..] == a[6 * IDX_DIM..];
        let fault_ok = before.is_none() && raised == want && nan && others;
        println!(
            "qsa pool fault: NaN in a raw row of pool 5, layer {LAYER}: word {raised:?} (want \
             {want:?}), pool 5 not finite {nan}, other pools bit-identical {others} {}",
            verdict(fault_ok)
        );
        Ok(pass && fault_ok)
    }

    /// The selection clause (module doc, 10).
    fn select_check(gpu: &Gpu, qk: &QsaKernels) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let idx = idx_inputs(stream, SEL_CTX, 501, false)?;
        let (pooled, ph) = pool_plane(qk, stream, &idx, SEL_N, SEL_N, unl)?;
        let mut scratch = QsaScratch::new(stream, MAX_ROWS, SEL_CTX, KEPT)?;
        let mut ok = true;

        let to_u32 = |c: &[usize]| -> Result<Vec<u32>, GateError> {
            Ok(c.iter()
                .map(|&v| u32::try_from(v))
                .collect::<Result<_, _>>()?)
        };
        let verify: Vec<usize> = (SEL_ROWS0..=SEL_N).collect();
        for (name, counts, seed) in [
            ("rows", SEL_COUNTS.to_vec(), 511u32),
            ("verify", verify, 512u32),
        ] {
            let q = idx_queries(counts.len(), seed);
            let cu = to_u32(&counts)?;
            let s = run_select(qk, stream, unl, &idx, &pooled, &q, &cu, &mut scratch)?;
            let s2 = run_select(qk, stream, unl, &idx, &pooled, &q, &cu, &mut scratch)?;
            let (heads, sc, lists, worst) = select_rows_ok(&idx, &ph, &q, &counts, &s);
            let rerun =
                s.list == s2.list && s.n_sel == s2.n_sel && bits_equal(&s.scores, &s2.scores);
            let mut alone = true;
            for t in 0..counts.len() {
                let w = IDX_HEADS * IDX_DIM;
                let one = run_select(
                    qk,
                    stream,
                    unl,
                    &idx,
                    &pooled,
                    &q[t * w..(t + 1) * w],
                    &cu[t..=t],
                    &mut scratch,
                )?;
                alone &= row_list(&one, 0) == row_list(&s, t);
            }
            let lens: Vec<u32> = s.n_sel.clone();
            let pass = heads && sc && lists && rerun && alone;
            println!(
                "qsa select {name} counts={counts:?}: heads bit_exact_host={heads} scores \
                 measured/bound {worst:.3e} band={sc} lists = rule on the scores (lower pool on a \
                 tie)={lists} lengths={lens:?} rerun={rerun} each row \
                 = its one-row launch={alone} {}",
                verdict(pass)
            );
            ok &= pass;
        }

        // A tie across the cut: pools 100..700 have zero raw rows, so each
        // scores +0 exactly; at count 4,097 fewer than 512 pools score above
        // it, and the rest of the 512 are the lowest of the tied pools.
        let tie = idx_inputs(stream, SEL_CTX, 521, true)?;
        let (tp, tph) = pool_plane(qk, stream, &tie, TIE_COUNT, TIE_COUNT, unl)?;
        let q = idx_queries(1, 522);
        let s = run_select(
            qk,
            stream,
            unl,
            &tie,
            &tp,
            &q,
            &[u32::try_from(TIE_COUNT)?],
            &mut scratch,
        )?;
        let (_, _, lists, _) = select_rows_ok(&tie, &tph, &q, &[TIE_COUNT], &s);
        let nb = TIE_COUNT / POOL;
        let zeros: Vec<usize> = (0..nb).filter(|&j| s.scores[j] == 0.0).collect();
        let kept_zero = row_list(&s, 0)
            .chunks(POOL)
            .filter(|c| c.len() == POOL && s.scores[c[0] as usize / POOL] == 0.0)
            .count();
        let straddles = kept_zero > 0 && kept_zero < zeros.len();
        let tie_ok = lists && straddles;
        println!(
            "qsa select tie count={TIE_COUNT}: {} pools score +0, the cut keeps {kept_zero} of \
             them (the lowest wanted) lists = rule={lists} tie straddles the cut={straddles} {}",
            zeros.len(),
            verdict(tie_ok)
        );
        ok &= tie_ok;

        // Refused counts: zero and past the cache get empty lists; the row
        // beside them is its clean list.
        let counts = [0usize, SEL_CTX + 1, TIE_COUNT];
        let q = idx_queries(3, 531);
        let cu = to_u32(&counts)?;
        let s = run_select(qk, stream, unl, &idx, &pooled, &q, &cu, &mut scratch)?;
        let (heads, _, lists, _) = select_rows_ok(&idx, &ph, &q, &counts, &s);
        let refused_ok = heads && lists && s.n_sel[0] == 0 && s.n_sel[1] == 0;
        println!(
            "qsa select refused counts={counts:?}: lengths {:?}, refused heads NaN and the rest \
             bit_exact={heads} lists={lists} {}",
            s.n_sel,
            verdict(refused_ok)
        );
        ok &= refused_ok;

        // A NaN in row 1's head 2: the fault, a defined list for that row,
        // row 0's the clean list.
        let counts = [4097usize, SEL_N];
        let cu = to_u32(&counts)?;
        let q = idx_queries(2, 541);
        let clean = run_select(qk, stream, unl, &idx, &pooled, &q, &cu, &mut scratch)?;
        let mut qn = q.clone();
        qn[(IDX_HEADS + 2) * IDX_DIM + 40] = f32::NAN;
        let before = gpu.fault()?;
        let s = run_select(
            qk,
            stream,
            gpu.layer_sink(LAYER)?,
            &idx,
            &pooled,
            &qn,
            &cu,
            &mut scratch,
        )?;
        let raised = gpu.take_fault()?;
        let want = Some(Fault::at(u32::try_from(LAYER)?, FaultSite::PoolSelect));
        let row1 = row_list(&s, 1);
        let defined = s.n_sel[1] as usize == qsa_shape().list_len(SEL_N)
            && row1.windows(2).all(|w| w[0] < w[1])
            && row1.iter().all(|&v| (v as usize) < SEL_N);
        let row0 = row_list(&s, 0) == row_list(&clean, 0);
        let nan_ok = before.is_none() && raised == want && defined && row0;
        println!(
            "qsa select fault: NaN in row 1's head 2, layer {LAYER}: word {raised:?} (want \
             {want:?}), row 1's list defined {defined}, row 0 its clean list {row0} {}",
            verdict(nan_ok)
        );
        ok &= nan_ok;

        // Nine rows are past the scratch: refused by name.
        let q9 = idx_queries(9, 551);
        let c9 = vec![TIE_COUNT as u32; 9];
        let (qd, nk) = (
            DeviceBuffer::from_host(stream, &q9)?,
            DeviceBuffer::from_host(stream, &c9)?,
        );
        let r = qk.enqueue_select(
            stream,
            SelectArgs {
                q: &qd,
                gain: &idx.gq_d,
                table: &idx.table_d,
                n_keys: &nk,
                pooled: &pooled,
                eps: EPS,
                ctx: SEL_CTX,
                kept: KEPT,
                m: 9,
                fault: unl,
                scratch: &mut scratch,
            },
        );
        let named = matches!(
            r,
            Err(GpuError::Shape {
                what: "qsa::enqueue_select",
                ..
            })
        );
        println!(
            "qsa select refusal m=9 over a scratch of {MAX_ROWS}: {} {}",
            r.err().map_or("accepted".to_string(), |e| e.to_string()),
            verdict(named)
        );
        Ok(ok && named)
    }

    /// One selected-flash launch of `m` rows over the lists and lengths on
    /// the device, into fresh scratch, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernels, the stream and sink, the rows, the lists, the cache and its height, the pass"
    )]
    fn run_sel(
        k: &FlashGqaKernels,
        stream: &CudaStream,
        fault: FaultSink,
        q: &[f32],
        (list, n_sel): (&DeviceBuffer<u32>, &DeviceBuffer<u32>),
        m: usize,
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
        mma: bool,
    ) -> Result<Vec<f32>, GateError> {
        let qd = DeviceBuffer::from_host(stream, q)?;
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len_256(m, N_HEAD_Q38, WIDTH))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, N_HEAD_Q38, WIDTH))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * Q38.width())?;
        k.enqueue_pass_256_p4_sel(
            stream,
            GqaSelArgs {
                q: &qd,
                kc,
                vc,
                list,
                n_sel,
                width: WIDTH,
                scale: scale(),
                n_kv: N_KV,
                ctx,
                m,
                part_v: &mut pv,
                part_ms: &mut pms,
                fault,
                y: &mut y,
            },
            N_HEAD_Q38,
            mma,
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// Every (row, head) of `y` against the exact attention over the row's
    /// listed keys, under `pass`'s bound: [`exact`] on the gathered rows.
    fn band_listed(
        q: &[f32],
        lists: &[Vec<u32>],
        cache: &HostCache,
        y: &[f32],
        pass: Pass,
    ) -> (bool, f64) {
        let (mut ok, mut worst) = (true, 0.0f64);
        for (t, list) in lists.iter().enumerate() {
            for h in 0..N_HEAD_Q38 {
                let plane = (h / Q38.group()) * cache.ctx * HEAD;
                let gather = |src: &[f32]| -> Vec<f32> {
                    list.iter()
                        .flat_map(|&r| src[plane + r as usize * HEAD..][..HEAD].iter().copied())
                        .collect()
                };
                let (kh, vh) = (gather(&cache.kf), gather(&cache.vf));
                let row = (t * N_HEAD_Q38 + h) * HEAD;
                let ex = exact(&q[row..row + HEAD], &kh, &vh, list.len(), scale());
                let bound = match pass {
                    Pass::Scalar => &ex.bound_scalar,
                    Pass::Mma => &ex.bound_mma,
                    Pass::Prefill => &ex.bound_pref,
                };
                for d in 0..HEAD {
                    let e = (f64::from(y[row + d]) - ex.o[d]).abs();
                    worst = worst.max(e / bound[d]);
                    ok &= e <= bound[d];
                }
            }
        }
        (ok, worst)
    }

    /// The selected flash's clauses (module doc, 11–13).
    fn sel_flash_check(gpu: &Gpu, k: &FlashGqaKernels, qk: &QsaKernels) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let idx = idx_inputs(stream, SEL_CTX, 601, false)?;
        let (pooled, _) = pool_plane(qk, stream, &idx, SEL_N, SEL_N, unl)?;
        let mut scratch = QsaScratch::new(stream, MAX_ROWS, SEL_CTX, KEPT)?;
        let c = cache(stream, SEL_CTX, SEL_N, 610)?;
        let w = Q38.width();
        let mut ok = true;

        // 11. At counts up to 2,051 the list is every token; the selected
        // flash's own body must give the dense pack-of-four body's bits.
        let dense_rows: Vec<usize> = (WIDTH - 7..=WIDTH).collect();
        for (name, counts) in [("one-row", SEL_DENSE.to_vec()), ("rows", dense_rows)] {
            let one_row = name == "one-row";
            let launches: Vec<Vec<usize>> = if one_row {
                counts.iter().map(|&n| vec![n]).collect()
            } else {
                vec![counts.clone()]
            };
            for cs in launches {
                let m = cs.len();
                let cu: Vec<u32> = cs
                    .iter()
                    .map(|&v| u32::try_from(v))
                    .collect::<Result<_, _>>()?;
                run_select(
                    qk,
                    stream,
                    unl,
                    &idx,
                    &pooled,
                    &idx_queries(m, 611),
                    &cu,
                    &mut scratch,
                )?;
                let q: Vec<f32> = activations(HEAD, m * N_HEAD_Q38, 612)
                    .iter()
                    .map(|v| v * SEED_Q_SCALE)
                    .collect();
                for mma in [false, true] {
                    let ys = run_sel(
                        k,
                        stream,
                        unl,
                        &q,
                        (&scratch.list, &scratch.n_sel),
                        m,
                        (&c.kc, &c.vc),
                        SEL_CTX,
                        mma,
                    )?;
                    let yd = run_dec(k, stream, unl, Q38, &q, &cu, (&c.kc, &c.vc), SEL_CTX, mma)?;
                    let same = bits_equal(&ys, &yd);
                    println!(
                        "qsa flash dense edge pass={} counts={cs:?}: selected = dense p4 bit for \
                         bit {same} {}",
                        if mma { "mma" } else { "scalar" },
                        verdict(same)
                    );
                    ok &= same;
                }
            }
        }

        // 12. Past 2,051: each row within the band of the exact attention
        // over its listed keys; NaN in every cache row no list names changes
        // no bit; the eight rows each their one-row launch; a rerun.
        // PIN(2026-09-27): the band is `exact`'s at n = the list's length — the segments and tiles are cut over list positions as the dense flash cuts them over the cache, so its model holds term for term.
        let verify: Vec<usize> = (SEL_ROWS0..=SEL_N).collect();
        for (name, counts) in [("one-row", SEL_COUNTS[1..].to_vec()), ("verify", verify)] {
            let launches: Vec<Vec<usize>> = if name == "one-row" {
                counts.iter().map(|&n| vec![n]).collect()
            } else {
                vec![counts.clone()]
            };
            for cs in launches {
                let m = cs.len();
                let cu: Vec<u32> = cs
                    .iter()
                    .map(|&v| u32::try_from(v))
                    .collect::<Result<_, _>>()?;
                let qi = idx_queries(m, 621);
                let s = run_select(qk, stream, unl, &idx, &pooled, &qi, &cu, &mut scratch)?;
                let lists: Vec<Vec<u32>> = (0..m).map(|t| row_list(&s, t).to_vec()).collect();
                let q: Vec<f32> = activations(HEAD, m * N_HEAD_Q38, 622)
                    .iter()
                    .map(|v| v * SEED_Q_SCALE)
                    .collect();
                // The planes with every row outside the lists' union NaN.
                let mut named = vec![false; SEL_CTX];
                for l in &lists {
                    for &r in l {
                        named[r as usize] = true;
                    }
                }
                let nanify = |b: &[u16]| -> Vec<u16> {
                    b.iter()
                        .enumerate()
                        .map(|(i, &h)| {
                            if named[(i / HEAD) % SEL_CTX] {
                                h
                            } else {
                                NAN16
                            }
                        })
                        .collect()
                };
                let kn = DeviceBuffer::from_host(stream, &nanify(&c.kb))?;
                let vn = DeviceBuffer::from_host(stream, &nanify(&c.vb))?;
                for (pass, mma) in [(Pass::Scalar, false), (Pass::Mma, true)] {
                    let lsel = (&scratch.list, &scratch.n_sel);
                    let y = run_sel(k, stream, unl, &q, lsel, m, (&c.kc, &c.vc), SEL_CTX, mma)?;
                    let y2 = run_sel(k, stream, unl, &q, lsel, m, (&c.kc, &c.vc), SEL_CTX, mma)?;
                    let yn = run_sel(k, stream, unl, &q, lsel, m, (&kn, &vn), SEL_CTX, mma)?;
                    let (band, worst) = band_listed(&q, &lists, &c.host, &y, pass);
                    let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
                    let mut alone = true;
                    if m > 1 {
                        let qw = IDX_HEADS * IDX_DIM;
                        for t in 0..m {
                            run_select(
                                qk,
                                stream,
                                unl,
                                &idx,
                                &pooled,
                                &qi[t * qw..(t + 1) * qw],
                                &cu[t..=t],
                                &mut scratch,
                            )?;
                            let yo = run_sel(
                                k,
                                stream,
                                unl,
                                &q[t * w..(t + 1) * w],
                                (&scratch.list, &scratch.n_sel),
                                1,
                                (&c.kc, &c.vc),
                                SEL_CTX,
                                mma,
                            )?;
                            alone &= bits_equal(&yo, &y[t * w..(t + 1) * w]);
                        }
                        // The m-row selection again, for the next pass.
                        run_select(qk, stream, unl, &idx, &pooled, &qi, &cu, &mut scratch)?;
                    }
                    let pass_ok = band && rerun && nan_same && alone;
                    println!(
                        "qsa flash {name} pass={} counts={cs:?} lengths={:?}: measured/bound \
                         {worst:.3e} band={band} rerun={rerun} nan_in_unlisted_rows_same={nan_same} \
                         rows = one-row launches={alone} {}",
                        pass.name(),
                        s.n_sel,
                        verdict(pass_ok)
                    );
                    ok &= pass_ok;
                }
            }
        }

        // The captured chain — pool, score, top-k, segment pass, merge — on
        // the eight verify rows: five nodes, the eager bits.
        let cs: Vec<u32> = (SEL_ROWS0..=SEL_N)
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
        let m = cs.len();
        let qi = DeviceBuffer::from_host(stream, &idx_queries(m, 631))?;
        let qa = DeviceBuffer::from_host(
            stream,
            &activations(HEAD, m * N_HEAD_Q38, 632)
                .iter()
                .map(|v| v * SEED_Q_SCALE)
                .collect::<Vec<_>>(),
        )?;
        let nk = DeviceBuffer::from_host(stream, &cs)?;
        let mut pooled_g = DeviceBuffer::from_host(stream, &pooled.to_host_vec(stream)?)?;
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len_256(m, N_HEAD_Q38, WIDTH))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, N_HEAD_Q38, WIDTH))?;
        let mut ya = DeviceBuffer::<f32>::zeroed(stream, m * w)?;
        let mut yg = DeviceBuffer::<f32>::zeroed(stream, m * w)?;
        let chain = |s: &CudaStream,
                     pooled: &mut DeviceBuffer<u16>,
                     scratch: &mut QsaScratch,
                     pv: &mut DeviceBuffer<f32>,
                     pms: &mut DeviceBuffer<f32>,
                     y: &mut DeviceBuffer<f32>|
         -> Result<(), GpuError> {
            qk.enqueue_pool(
                s,
                PoolArgs {
                    raw: &idx.raw_d,
                    gain: &idx.gk_d,
                    table: &idx.table_d,
                    n_keys: &nk,
                    eps: EPS,
                    ctx: SEL_CTX,
                    m,
                    fault: unl,
                    pooled: &mut *pooled,
                },
            )?;
            qk.enqueue_select(
                s,
                SelectArgs {
                    q: &qi,
                    gain: &idx.gq_d,
                    table: &idx.table_d,
                    n_keys: &nk,
                    pooled: &*pooled,
                    eps: EPS,
                    ctx: SEL_CTX,
                    kept: KEPT,
                    m,
                    fault: unl,
                    scratch: &mut *scratch,
                },
            )?;
            k.enqueue_pass_256_p4_sel(
                s,
                GqaSelArgs {
                    q: &qa,
                    kc: &c.kc,
                    vc: &c.vc,
                    list: &scratch.list,
                    n_sel: &scratch.n_sel,
                    width: WIDTH,
                    scale: scale(),
                    n_kv: N_KV,
                    ctx: SEL_CTX,
                    m,
                    part_v: pv,
                    part_ms: pms,
                    fault: unl,
                    y,
                },
                N_HEAD_Q38,
                true,
            )
        };
        chain(
            stream,
            &mut pooled_g,
            &mut scratch,
            &mut pv,
            &mut pms,
            &mut ya,
        )?;
        stream.synchronize()?;
        let eager = ya.to_host_vec(stream)?;
        let graph =
            gpu.capture(|s| chain(s, &mut pooled_g, &mut scratch, &mut pv, &mut pms, &mut yg))?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let same = bits_equal(&yg.to_host_vec(stream)?, &eager);
        let nodes = graph.node_count();
        let graph_ok = same && nodes == 5;
        println!(
            "qsa chain graph m={m}: eager_vs_graph_bit_identical={same} graph_nodes={nodes} {}",
            verdict(graph_ok)
        );
        ok &= graph_ok;

        // 13. Refusals: row 0's list naming the cache's height raises
        // pool_select, row 1's length zero and past the width raise key_count;
        // those rows NaN, the other row bit for bit its clean output.
        let cs = [u32::try_from(TIE_COUNT)?, u32::try_from(SEL_N)?];
        let s = run_select(
            qk,
            stream,
            unl,
            &idx,
            &pooled,
            &idx_queries(2, 641),
            &cs,
            &mut scratch,
        )?;
        let q: Vec<f32> = activations(HEAD, 2 * N_HEAD_Q38, 642)
            .iter()
            .map(|v| v * SEED_Q_SCALE)
            .collect();
        let clean_l = DeviceBuffer::from_host(stream, &s.list)?;
        let clean_n = DeviceBuffer::from_host(stream, &s.n_sel)?;
        let clean = run_sel(
            k,
            stream,
            unl,
            &q,
            (&clean_l, &clean_n),
            2,
            (&c.kc, &c.vc),
            SEL_CTX,
            true,
        )?;
        let mut bad_list = s.list.clone();
        bad_list[7] = u32::try_from(SEL_CTX)?;
        let cases = [
            (
                "list entry at the cache's height",
                bad_list,
                s.n_sel.clone(),
                0,
                FaultSite::PoolSelect,
            ),
            (
                "length 0",
                s.list.clone(),
                vec![s.n_sel[0], 0],
                1,
                FaultSite::KeyCount,
            ),
            (
                "length past the width",
                s.list.clone(),
                vec![s.n_sel[0], u32::try_from(WIDTH + 1)?],
                1,
                FaultSite::KeyCount,
            ),
        ];
        for (what, l, n, bad_row, site) in cases {
            let ld = DeviceBuffer::from_host(stream, &l)?;
            let nd = DeviceBuffer::from_host(stream, &n)?;
            for mma in [false, true] {
                let clean_p = if mma {
                    clean.clone()
                } else {
                    run_sel(
                        k,
                        stream,
                        unl,
                        &q,
                        (&clean_l, &clean_n),
                        2,
                        (&c.kc, &c.vc),
                        SEL_CTX,
                        false,
                    )?
                };
                let before = gpu.fault()?;
                let y = run_sel(
                    k,
                    stream,
                    gpu.layer_sink(LAYER)?,
                    &q,
                    (&ld, &nd),
                    2,
                    (&c.kc, &c.vc),
                    SEL_CTX,
                    mma,
                )?;
                let raised = gpu.take_fault()?;
                let want = Some(Fault::at(u32::try_from(LAYER)?, site));
                let other = 1 - bad_row;
                let nan = y[bad_row * w..(bad_row + 1) * w].iter().all(|v| v.is_nan());
                let others = bits_equal(
                    &y[other * w..(other + 1) * w],
                    &clean_p[other * w..(other + 1) * w],
                );
                let f_ok = before.is_none() && raised == want && nan && others;
                println!(
                    "qsa flash fault pass={}: {what} at row {bad_row}, layer {LAYER}: word \
                     {raised:?} (want {want:?}), that row NaN {nan}, the other row bit-identical \
                     {others} {}",
                    if mma { "mma" } else { "scalar" },
                    verdict(f_ok)
                );
                ok &= f_ok;
            }
        }
        Ok(ok)
    }

    /// The deep clause's cache height and its largest count: Qwen3.8's
    /// `context_length`, the most a load serves (`place::serve_ctx`).
    const DEEP_CTX: usize = 262_144;
    /// One-row counts of the deep clause: the first past the u16 range, one
    /// past twice it with a tail of two, a tail of three, the cap.
    const DEEP_COUNTS: [usize; 4] = [65_537, 131_074, 262_143, 262_144];
    /// The first count of the eight rows of a verify at the cap.
    const DEEP_ROWS0: usize = DEEP_CTX - 7;

    /// The deep clause (module doc, 15): the selector and the selected flash
    /// at [`DEEP_CTX`] on synthetic keys, no model loaded.
    fn deep_check(gpu: &Gpu, k: &FlashGqaKernels, qk: &QsaKernels) -> Result<bool, GateError> {
        let t0 = std::time::Instant::now();
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let idx = idx_inputs(stream, DEEP_CTX, 701, false)?;
        // One pool launch of every count: the row of count 262,144 completes
        // pool 65,535, the plane's last.
        let (pooled, ph) = pool_plane(qk, stream, &idx, DEEP_CTX, DEEP_CTX, unl)?;
        let complete = DEEP_CTX / POOL;
        let (exact, worst) = pool_keys_ok(&idx, &ph, complete);
        let pool_ok = exact && worst <= 1.0;
        println!(
            "qsa deep pool counts 1..={DEEP_CTX} ctx={DEEP_CTX}: {complete} pools \
             bit_exact_host={exact} f64 measured/bound {worst:.3e} {}",
            verdict(pool_ok)
        );
        let mut ok = pool_ok;
        let mut scratch = QsaScratch::new(stream, MAX_ROWS, DEEP_CTX, KEPT)?;
        let kb = to16(&activations(HEAD, N_KV * DEEP_CTX, 720));
        let vb = to16(&activations(HEAD, N_KV * DEEP_CTX, 721));
        let (kc, vc) = (
            DeviceBuffer::from_host(stream, &kb)?,
            DeviceBuffer::from_host(stream, &vb)?,
        );
        let host = HostCache {
            kf: from16(&kb),
            vf: from16(&vb),
            ctx: DEEP_CTX,
        };
        let verify: Vec<usize> = (DEEP_ROWS0..=DEEP_CTX).collect();
        for (name, counts, seed) in [
            ("one launch", DEEP_COUNTS.to_vec(), 711u32),
            ("verify", verify, 712u32),
        ] {
            let m = counts.len();
            let cu: Vec<u32> = counts
                .iter()
                .map(|&v| u32::try_from(v))
                .collect::<Result<_, _>>()?;
            let qi = idx_queries(m, seed);
            let sel = run_select(qk, stream, unl, &idx, &pooled, &qi, &cu, &mut scratch)?;
            let (heads, sc, lists, worst) = select_rows_ok(&idx, &ph, &qi, &counts, &sel);
            let rows: Vec<Vec<u32>> = (0..m).map(|t| row_list(&sel, t).to_vec()).collect();
            // Every list reaches past the u16 range, and a row with a tail
            // ends at its own position.
            let high = rows
                .iter()
                .all(|l| l.iter().any(|&r| r > u32::from(u16::MAX)));
            let last = counts.iter().zip(&rows).all(|(&c, l)| {
                c % POOL == 0 || l.last() == Some(&u32::try_from(c - 1).unwrap_or(u32::MAX))
            });
            let sel_ok = heads && sc && lists && high && last;
            println!(
                "qsa deep select {name} counts={counts:?}: heads bit_exact_host={heads} scores \
                 measured/bound {worst:.3e} band={sc} lists = rule on the scores={lists} \
                 lengths={:?} rows past 65,535 in every list={high} each tail ends at its \
                 position={last} {}",
                sel.n_sel,
                verdict(sel_ok)
            );
            ok &= sel_ok;
            // The planes with every row outside the lists' union NaN.
            let mut named = vec![false; DEEP_CTX];
            for &r in rows.iter().flatten() {
                named[r as usize] = true;
            }
            let nanify = |b: &[u16]| -> Vec<u16> {
                b.iter()
                    .enumerate()
                    .map(|(i, &h)| {
                        if named[(i / HEAD) % DEEP_CTX] {
                            h
                        } else {
                            NAN16
                        }
                    })
                    .collect()
            };
            let kn = DeviceBuffer::from_host(stream, &nanify(&kb))?;
            let vn = DeviceBuffer::from_host(stream, &nanify(&vb))?;
            let q: Vec<f32> = activations(HEAD, m * N_HEAD_Q38, seed + 10)
                .iter()
                .map(|v| v * SEED_Q_SCALE)
                .collect();
            for (pass, mma) in [(Pass::Scalar, false), (Pass::Mma, true)] {
                let lsel = (&scratch.list, &scratch.n_sel);
                let y = run_sel(k, stream, unl, &q, lsel, m, (&kc, &vc), DEEP_CTX, mma)?;
                let yn = run_sel(k, stream, unl, &q, lsel, m, (&kn, &vn), DEEP_CTX, mma)?;
                let (band, worst) = band_listed(&q, &rows, &host, &y, pass);
                let nan_same = bits_equal(&y, &yn);
                let pass_ok = band && nan_same;
                println!(
                    "qsa deep flash {name} pass={} counts={counts:?}: measured/bound \
                     {worst:.3e} band={band} nan_in_unlisted_rows_same={nan_same} {}",
                    pass.name(),
                    verdict(pass_ok)
                );
                ok &= pass_ok;
            }
        }
        println!(
            "qsa deep ctx={DEEP_CTX}: {:.1} s (runtime value)",
            t0.elapsed().as_secs_f64()
        );
        Ok(ok)
    }

    /// The new entries compile with no local depot (module doc, 14).
    fn sel_shapes() -> Result<bool, GateError> {
        no_local_depot(&[
            "qsa_pool",
            "qsa_score",
            "qsa_topk",
            "gqa_flash_seg_256_p4_sel",
            "gqa_flash_seg_mma_256_p4_sel",
        ])
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        let rope = RopeNeoxKernels::load(ctx)?;
        let k = FlashGqaKernels::load(ctx)?;
        let kp = FlashGqaPrefill::load(ctx)?;
        let gq = GatedQuantKernels::load(ctx)?;
        let qk = QsaKernels::load(ctx)?;
        println!(
            "gate_qwen35moe_attn: device {} — {N_HEAD}/{N_KV} heads of {HEAD}, rope over {ROT} at \
             θ {THETA:e}, scale {:e}",
            gpu.device_name()?,
            scale()
        );
        let mut ok = true;
        let r = rope_check(&gpu, &rope)?;
        println!("rope-256 {}", verdict(r));
        ok &= r;
        let r = rope_q8_check(&gpu, &rope)?;
        println!("rope-256 q8 {}", verdict(r));
        ok &= r;
        let d = decode_check(&gpu, &k, Q36, &DEC_KEYS)?;
        println!("decode flash 256 {}", verdict(d));
        ok &= d;
        let d = decode_check(&gpu, &k, Q38, &DEC_KEYS_Q38)?;
        println!("decode flash 256 p4 {}", verdict(d));
        ok &= d;
        let x = cross_check(&gpu, &k, &kp)?;
        println!("cross geometry {}", verdict(x));
        ok &= x;
        let t = tie_check(&gpu, &k, &kp)?;
        println!("score order {}", verdict(t));
        ok &= t;
        let g = gated_check(&gpu, &gq)?;
        println!("gated quantizer {}", verdict(g));
        ok &= g;
        let i = model_check(&gpu, &rope, &gq)?;
        println!("ik taps, layer {MODEL_LAYER} {}", verdict(i));
        ok &= i;
        let s = sel_shapes()?;
        println!("qsa shapes {}", verdict(s));
        ok &= s;
        let s = pool_check(&gpu, &qk)?;
        println!("qsa pool {}", verdict(s));
        ok &= s;
        let s = select_check(&gpu, &qk)?;
        println!("qsa select {}", verdict(s));
        ok &= s;
        let s = sel_flash_check(&gpu, &k, &qk)?;
        println!("qsa selected flash {}", verdict(s));
        ok &= s;
        let s = deep_check(&gpu, &k, &qk)?;
        println!("qsa deep {}", verdict(s));
        ok &= s;
        let (p, win36) = prefill_check(&gpu, &kp, Q36, &[])?;
        println!("prefill flash 256 {}", verdict(p));
        ok &= p;
        let (p, win38) = prefill_check(&gpu, &kp, Q38, &SEED_Q38)?;
        println!("prefill flash 256 p4 {}", verdict(p));
        ok &= p;
        // Last: the window refusals, whose failure would be a sticky error.
        for (sh, win) in [(Q36, &win36), (Q38, &win38)] {
            let m = misaligned(&kp, &gpu, sh, win)?;
            println!("prefill windows{} {}", sh.tag(), verdict(m));
            ok &= m;
        }
        println!("gate_qwen35moe_attn: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
