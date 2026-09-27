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
//!    and appends nothing; every other token is the clean run's.
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
        FlashGqaKernels, GROUP, GqaArgs, HEAD_256 as HEAD, KEY_TILE, SEG_KEYS, partials_ms_len,
        partials_v_len_256, segments_for,
    };
    use bloomery_gpu::flash_gqa_prefill::{FlashGqaPrefill, GqaPrefillArgs, KEY_TILE as PREF_TILE};
    use bloomery_gpu::gated_quant::{GateLayout, GatedQuantKernels};
    use bloomery_gpu::gemm::GemmAct;
    use bloomery_gpu::rope_neox::{PartialNeoxArgs, ROT_256 as ROT, RopeNeoxKernels, owned_pair};
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu::route_core::sigmoid;
    use bloomery_gpu::{Gpu, GpuError, Q8Act, window};
    use bloomery_gpu_gates::rounding::{U, U_F32, butterfly, gamma};
    use bloomery_gpu_gates::{
        GateError, Layout, RefManifest, RowKind, activations, bits_equal, checks_failed, data_dir,
        ref_ints, ref_tensor_logical_in, split_f32, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use refset::arch::qwen35moe::{BATCH, IK, MODEL};
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

    /// Decode: the live counts of the one-row launches, and the rows of the
    /// multi-row launch.
    const DEC_KEYS: [usize; 4] = [5, 64, 1025, 4097];
    const ROWS: usize = 8;
    /// Cache rows past a launch's largest count.
    const PAD: usize = 40;

    /// The prefill flash's seeded launches: row counts and first positions.
    const SEED_T: [usize; 5] = [1, 17, 64, 65, 512];
    const SEED_P0: [usize; 4] = [0, 63, 64, 1000];

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

    fn rope_check(gpu: &Gpu, kern: &RopeNeoxKernels) -> Result<bool, GateError> {
        let stream = gpu.stream();
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
    /// `q` holds the rows' query heads token-major, `counts[t]` row `t`'s keys.
    /// Returns whether every value is within its bound and the largest
    /// measured over bound. Rows are shared among worker threads.
    fn band_rows(
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
                            for h in 0..N_HEAD {
                                let plane = (h / GROUP) * cache.ctx * HEAD;
                                let row = (t * N_HEAD + h) * HEAD;
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

    /// One decode launch of `n_keys.len()` rows, into fresh scratch, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernels, the stream and sink, the rows, the cache and its height, the pass"
    )]
    fn run_dec(
        k: &FlashGqaKernels,
        stream: &CudaStream,
        fault: FaultSink,
        q: &[f32],
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
        mma: bool,
    ) -> Result<Vec<f32>, GateError> {
        let m = n_keys.len();
        let qd = DeviceBuffer::from_host(stream, q)?;
        let nk = DeviceBuffer::from_host(stream, n_keys)?;
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len_256(m, N_HEAD, ctx))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, N_HEAD, ctx))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * N_HEAD * HEAD)?;
        k.enqueue_pass_256(
            stream,
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

    fn decode_check(gpu: &Gpu, k: &FlashGqaKernels) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let w = N_HEAD * HEAD;
        let mut ok = true;
        for (i, &live) in DEC_KEYS.iter().enumerate() {
            let ctx = live + PAD;
            let seed = 100 + 10 * u32::try_from(i)?;
            let c = cache(stream, ctx, live, seed)?;
            let q: Vec<f32> = activations(HEAD, N_HEAD, seed + 5)
                .iter()
                .map(|v| v * SEED_Q_SCALE)
                .collect();
            let nk = [u32::try_from(live)?];
            for (pass, mma) in [(Pass::Scalar, false), (Pass::Mma, true)] {
                let y = run_dec(k, stream, unl, &q, &nk, (&c.kc, &c.vc), ctx, mma)?;
                let y2 = run_dec(k, stream, unl, &q, &nk, (&c.kc, &c.vc), ctx, mma)?;
                let yn = run_dec(k, stream, unl, &q, &nk, (&c.kn, &c.vn), ctx, mma)?;
                let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
                let (band, worst) = band_rows(&q, &[live], &c.host, &y, pass);
                let pass_ok = band && rerun && nan_same;
                println!(
                    "decode pass={} keys={live} ctx={ctx} segments={}: measured/bound {worst:.3e} \
                     band={band} rerun={rerun} nan_padding_same={nan_same} {}",
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
        let q1 = activations(HEAD, N_HEAD, 305);
        let rows: Vec<f32> = (0..ROWS)
            .flat_map(|t| {
                let q1 = &q1;
                (0..N_HEAD).flat_map(move |h| {
                    q1[((h + t) % N_HEAD) * HEAD..][..HEAD]
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
            let all = run_dec(k, stream, unl, &rows, &counts, (&c.kc, &c.vc), ctx, mma)?;
            let mut alone = true;
            for t in 0..ROWS {
                let one = run_dec(
                    k,
                    stream,
                    unl,
                    &rows[t * w..(t + 1) * w],
                    &counts[t..=t],
                    (&c.kc, &c.vc),
                    ctx,
                    mma,
                )?;
                alone &= bits_equal(&all[t * w..(t + 1) * w], &one);
            }
            println!(
                "decode rows pass={name} m={ROWS} keys={counts:?} ctx={ctx}: each row = its one-row \
                 launch bit for bit {}",
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
                "decode fault pass={name}: counts {} and 0 at rows {bad_hi}, {bad_zero}: word \
                 {raised:?} (want {want:?}), those rows NaN {nan}, other rows bit-identical {others} {}",
                ctx + 1,
                verdict(fault_ok)
            );
            ok &= fault_ok;

            // The captured launch: two nodes, the eager bits.
            let qd = DeviceBuffer::from_host(stream, &rows)?;
            let nk = DeviceBuffer::from_host(stream, &counts)?;
            let mut pv =
                DeviceBuffer::<f32>::zeroed(stream, partials_v_len_256(ROWS, N_HEAD, ctx))?;
            let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(ROWS, N_HEAD, ctx))?;
            let mut yg = DeviceBuffer::<f32>::zeroed(stream, ROWS * w)?;
            let graph = gpu.capture(|s| {
                k.enqueue_pass_256(
                    s,
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
                "decode graph pass={name} m={ROWS}: eager_vs_graph_bit_identical={same} \
                 graph_nodes={nodes} {}",
                verdict(graph_ok)
            );
            ok &= graph_ok;
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
        let dec = run_dec(k, stream, unl, &q, &[2], (&kc, &vc), ctx, true)?;
        let pre = run_pref(kp, gpu, &q, &[2], (&kc, &vc), ctx)?;
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

    /// One prefill launch of `n_keys.len()` rows, into fresh output, read
    /// back. The fault word is the caller's to read.
    fn run_pref(
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &[f32],
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let qd = DeviceBuffer::from_host(stream, q)?;
        run_pref_dev(kp, gpu, &qd, n_keys, (kc, vc), ctx)
    }

    fn run_pref_dev(
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &DeviceBuffer<f32>,
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let t = n_keys.len();
        let nk = DeviceBuffer::from_host(stream, n_keys)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, t * N_HEAD * HEAD)?;
        kp.enqueue_256(
            stream,
            GqaPrefillArgs {
                q,
                kc,
                vc,
                n_keys: &nk,
                scale: scale(),
                n_head: N_HEAD,
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
    fn rows_alone(
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &[f32],
        counts: &[u32],
        cache: (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        ctx: usize,
        all: &[f32],
    ) -> Result<usize, GateError> {
        let w = N_HEAD * HEAD;
        let mut differ = 0usize;
        for (t, &c) in counts.iter().enumerate() {
            let one = run_pref(kp, gpu, &q[t * w..(t + 1) * w], &[c], cache, ctx)?;
            differ += usize::from(!bits_equal(&all[t * w..(t + 1) * w], &one));
        }
        Ok(differ)
    }

    fn prefill_check(gpu: &Gpu, kp: &FlashGqaPrefill) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let w = N_HEAD * HEAD;
        let mut ok = true;
        let mut seed = 500u32;
        for &p0 in &SEED_P0 {
            for &t in &SEED_T {
                seed += 7;
                let ctx = p0 + t + PAD;
                let live = p0 + t;
                let c = cache(stream, ctx, live, seed)?;
                let q: Vec<f32> = activations(HEAD, t * N_HEAD, seed + 3)
                    .iter()
                    .map(|v| v * SEED_Q_SCALE)
                    .collect();
                let counts: Vec<u32> = (0..t)
                    .map(|i| u32::try_from(p0 + i + 1))
                    .collect::<Result<_, _>>()?;
                let y = run_pref(kp, gpu, &q, &counts, (&c.kc, &c.vc), ctx)?;
                let y2 = run_pref(kp, gpu, &q, &counts, (&c.kc, &c.vc), ctx)?;
                let yn = run_pref(kp, gpu, &q, &counts, (&c.kn, &c.vn), ctx)?;
                let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
                let cu: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
                let (band, worst) = band_rows(&q, &cu, &c.host, &y, Pass::Prefill);
                let differ = rows_alone(kp, gpu, &q, &counts, (&c.kc, &c.vc), ctx, &y)?;
                let pass = rerun && nan_same && band && differ == 0;
                println!(
                    "prefill seeded T={t} p0={p0} ctx={ctx}: measured/bound {worst:.3e} band={band} \
                     rows_alone_differing={differ} rerun={rerun} nan_padding_same={nan_same} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
        }

        // A count of zero and one past the cache, on a 17-row launch at 63.
        let (t, p0) = (17usize, 63usize);
        let ctx = p0 + t + PAD;
        let c = cache(stream, ctx, p0 + t, 91)?;
        let q = activations(HEAD, t * N_HEAD, 94);
        let clean: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i + 1))
            .collect::<Result<_, _>>()?;
        let (bad_hi, bad_zero) = (3usize, 10usize);
        let mut bad = clean.clone();
        bad[bad_hi] = u32::try_from(ctx + 1)?;
        bad[bad_zero] = 0;
        let before = gpu.fault()?;
        let y = run_pref(kp, gpu, &q, &clean, (&c.kc, &c.vc), ctx)?;
        let after_clean = gpu.fault()?;
        let yb = run_pref(kp, gpu, &q, &bad, (&c.kc, &c.vc), ctx)?;
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
            "prefill fault: counts {} (past ctx {ctx}) and 0 at rows {bad_hi}, {bad_zero}: word \
             {raised:?} (want {want:?}), those rows NaN {nan}, other rows bit-identical {others} {}",
            ctx + 1,
            verdict(fault_ok)
        );
        ok &= fault_ok;

        // The captured launch: one node, the eager bits.
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let nk = DeviceBuffer::from_host(stream, &clean)?;
        let mut yg = DeviceBuffer::<f32>::zeroed(stream, t * w)?;
        let graph = gpu.capture(|s| {
            kp.enqueue_256(
                s,
                GqaPrefillArgs {
                    q: &qd,
                    kc: &c.kc,
                    vc: &c.vc,
                    n_keys: &nk,
                    scale: scale(),
                    n_head: N_HEAD,
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
            "prefill graph T={t} p0={p0}: eager_vs_graph_bit_identical={same} graph_nodes={nodes} {}",
            verdict(graph_ok)
        );
        ok &= graph_ok;

        // A head count the kernel is not built for.
        let mut yr = DeviceBuffer::<f32>::zeroed(stream, t * w)?;
        let refused = kp.enqueue_256(
            stream,
            GqaPrefillArgs {
                q: &qd,
                kc: &c.kc,
                vc: &c.vc,
                n_keys: &nk,
                scale: scale(),
                n_head: N_HEAD - 1,
                n_kv: N_KV,
                ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut yr,
            },
        );
        let refuse_ok = refused.is_err();
        println!(
            "prefill refusal n_head={} over n_kv={N_KV}: {} {}",
            N_HEAD - 1,
            refused
                .err()
                .map_or("accepted".to_string(), |e| e.to_string()),
            verdict(refuse_ok)
        );
        ok &= refuse_ok;
        ok &= misaligned(kp, gpu, &q, (&c.kb, &c.vb), (&clean, &y), ctx)?;
        Ok(ok)
    }

    /// `gate_qwen3moe_flash`'s window check at a head of 256: aligned windows
    /// accepted and bit for bit the plain launch, `q` at 4 bytes past 8 and
    /// `kc`/`vc` at 8 bytes past 16 refused by name before any launch. The
    /// last device check of its section: a launch through a misaligned window
    /// is a sticky error that ends the context.
    fn misaligned(
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &[f32],
        (kb, vb): (&[u16], &[u16]),
        (clean, y): (&[u32], &[f32]),
        ctx: usize,
    ) -> Result<bool, GateError> {
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
        let ya = run_pref_dev(kp, gpu, &q_al, clean, (&*k_al, &*v_al), ctx)?;
        let aligned_ok = bits_equal(&ya, y);
        println!(
            "prefill windows at aligned offsets: accepted, bit-identical to the plain launch \
             {aligned_ok} {}",
            verdict(aligned_ok)
        );
        let mut ok = aligned_ok;
        let mut ym = DeviceBuffer::<f32>::zeroed(stream, t * N_HEAD * HEAD)?;
        let cases = [
            ("q", "4 bytes past an 8-byte boundary", &*q_mis, &kc, &vc),
            ("kc", "8 bytes past a 16-byte boundary", &qd, &*k_mis, &vc),
            ("vc", "8 bytes past a 16-byte boundary", &qd, &kc, &*v_mis),
        ];
        for (name, off, q_in, kc_in, vc_in) in cases {
            let r = kp.enqueue_256(
                stream,
                GqaPrefillArgs {
                    q: q_in,
                    kc: kc_in,
                    vc: vc_in,
                    n_keys: &nk,
                    scale: scale(),
                    n_head: N_HEAD,
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
                    what: "flash_gqa_prefill::enqueue_256",
                    ref detail,
                }) if detail.starts_with(&format!("{name} at "))
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
                "prefill refusal {name} window {off}: {seen} {}",
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

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        let rope = RopeNeoxKernels::load(ctx)?;
        let k = FlashGqaKernels::load(ctx)?;
        let kp = FlashGqaPrefill::load(ctx)?;
        let gq = GatedQuantKernels::load(ctx)?;
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
        let d = decode_check(&gpu, &k)?;
        println!("decode flash 256 {}", verdict(d));
        ok &= d;
        let t = tie_check(&gpu, &k, &kp)?;
        println!("score order {}", verdict(t));
        ok &= t;
        let g = gated_check(&gpu, &gq)?;
        println!("gated quantizer {}", verdict(g));
        ok &= g;
        let i = model_check(&gpu, &rope, &gq)?;
        println!("ik taps, layer {MODEL_LAYER} {}", verdict(i));
        ok &= i;
        // Last: its window refusals end with the check whose failure would
        // be a sticky error.
        let p = prefill_check(&gpu, &kp)?;
        println!("prefill flash 256 {}", verdict(p));
        ok &= p;
        println!("gate_qwen35moe_attn: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
