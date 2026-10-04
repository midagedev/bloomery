//! GPU gate for the grouped-query flash decode (`flash_gqa`) against ik's
//! `fa-L` (FLASH_ATTN_EXT) on its own inputs: the query rows `q-L`, the
//! cache views `k-L`/`v-L` (f16, `[n_kv][keys][128]` in logical order —
//! the planes' own layout) and the mask's live prefix, at every layer of
//! the decode-step sets: 5, 1,025 and 4,097 visible keys.
//!
//! The kernel is not bit-identical to ik by construction (another
//! exponential, other sum orders), so the comparison is a derived band. Per
//! output value `o_d` of head `h`, with the exact attention computed here in
//! f64 on the same f16 keys and values and f32 query — normalized weights
//! `p̄_j`, `A_j = scale·Σ_d |q_d·k_jd|`, `X_j = M − s_j` — each side's
//! first-order distance to the exact value is at most
//!
//! `Σ_j p̄_j·(ε_s·A_j + ε_e(X_j) + ε_r)·|v_jd − o_d| + γ(n_acc)·(Σ_j p̄_j·|v_jd| + |o_d|) + 2u·|o_d|`
//!
//! - `ε_s`: the score dot's roundings — ours `γ(35)` (32 fused
//!   multiply-adds per partial, two combine levels, the scale), ik's
//!   `γ(129)` (any order over 128 products, and the scale);
//! - `ε_e(X) = 4u + 2u·X`: the exponential's own error and the rounding of
//!   its argument (`s − M`, then ours `· log2 e`);
//! - `ε_r = 4u·R + 2u·max X`: the running-max rescales, `R` of them at most
//!   (ours one per tile and per segment, ik's one per 32-key block and per
//!   thread chunk);
//! - `γ(n_acc)`: the value and weight sums — ours a segment's keys, the
//!   segments and the butterfly, ik's every key serially.
//!
//! The tensor-core pass (`gqa_flash_seg_mma`) rounds the query to f16 for
//! its scores, so its `ε_s` is `2^-11 + 2·γ(130)` (the f16 rounding, and
//! the tensor core's f32 accumulation, whose order is undocumented) on
//! magnitudes that include `2^-14·scale·Σ|k|` for query values below f16's
//! normal range; everything after the scores is the scalar pass's.
//!
//! The f16 rounding of K and V costs nothing here: both sides read the same
//! f16 bits. Asserted: ours within its bound of the exact value, ik within
//! its bound (the check that this model of ik is right), ours within the sum
//! of both of ik's `fa-L`; a rerun bit-identical; padded cache rows set to
//! NaN change no bit; the captured graph (two nodes) replays bit-identical.
//!
//! And the prefill shape, on each set's layer 0 in both passes: one launch of
//! [`ROWS`] query rows — row `t` the step's query with its heads rotated by
//! `t`, attending over its own live count, spread from one key to all of
//! them — bit for bit the [`ROWS`] one-row launches of those rows and
//! counts. The same launch with row 3's count one past the cache and row
//! 5's zero, labelled with layer 13: the `key_count` fault with that layer,
//! NaN in exactly rows 3 and 5, every other row bit for bit the clean run's,
//! the word clean before and after a clean run.
//!
//! The prefill flash (`flash_gqa_prefill`, the GEMM ubatch's attention) has
//! its own arithmetic and band. Its scores are the tensor-core pass's (the
//! same f16 query, products and k16 order), so `ε_s` and the magnitudes are
//! the tensor-core pass's; the rest differs:
//!
//! - `ε_r = 4u·R + 2u·max X` with `R = ⌈n/64⌉`, one rescale per 64-key tile;
//! - each weight `p = exp(s − m)` is rounded to f16 (`m` the running max when
//!   its tile ran, so `p ≥ p_M = exp(s − M)`): relative `2^-11` for a normal
//!   f16, absolute `2^-25` below `2^-14`. The normalizer is the sum of the
//!   rounded weights, so the output moves by `Σ_j (w_j + p̄_j·δ̄)·|v_jd − o_d|`
//!   with `w_j = 2^-11·p̄_j` (or `max(2^-11·p̄_j, 2^-25/Z)` when `p_M <
//!   2^-14`, `Z = Σ p_M`) and `δ̄ = Σ_j w_j` — the deviation of the values,
//!   not their size;
//! - the value product on the tensor core: `⌈n/16⌉` k16 steps into one f32
//!   accumulator, each step's sixteen exact products summed in an
//!   undocumented order that may truncate, `2·γ(⌈n/16⌉ + 16)` on
//!   `Σ_j p̄_j·|v_jd|`;
//! - the normalizer: each lane sums its sixteen weights of a tile serially
//!   over the tiles, then two butterfly levels, `γ(⌈n/4⌉ + ⌈n/64⌉ + 2)` on
//!   `|o_d|`; `o · (1/l)` adds `2u·|o_d|`.
//!
//! Asserted for the prefill flash: on each set, every layer's one-row launch
//! (`T = 1` at the step's count) within its bound of the exact value and
//! within the sum of both bounds of ik's `fa-L`, a rerun and NaN padding
//! bit-identical; on layer 0, [`ROWS_PREF`] rows at the counts ending at the
//! step's (heads rotated as above) each within its bound and each bit for bit
//! its one-row launch. On seeded inputs, `T` ∈ [`SEED_T`] × first position
//! `p0` ∈ [`SEED_P0`] (counts `p0 + t + 1`, crossing the tile edges 63/64/65,
//! 127/128/129): every row within its bound, every row bit for bit the same
//! row launched alone (`T = 1` at `p0 + t`), NaN in the cache rows past the
//! launch's largest count changing no bit, a rerun bit-identical. A count of
//! zero or past the cache raises the `key_count` fault, writes NaN in that
//! row and changes no other row's bit; the captured launch (one node) replays
//! bit-identical; a head count the kernel is not built for is refused. Last,
//! windows into padded allocations: at aligned offsets accepted and bit for
//! bit the plain launch; `q` at an odd f32 offset (4 bytes past an 8-byte
//! boundary — the query staging reads u64 words) and `kc` or `vc` 8 bytes past
//! a 16-byte boundary (the tiles copy 16-byte pieces) each refused by name
//! before any launch — the last check, because a launch through such a
//! window is a sticky error that ends the context.
//!
//! The q8_0 read path (`enqueue_pass_q8` and `enqueue_q8`, no engine caller
//! yet) runs the same walks over the cache's Q8_0 form, host-built over each
//! layer's own f32 K and V values: `quantize_q8_0` a 32-value block, the two
//! planes `q8_0_planes` packs. The oracle is the dequantized cache — the
//! q8_0 reads' own bits are not the f16 cache's, and what the format's
//! algebra pins is which rounding each pass takes, so the comparison is
//! split by the pass:
//! - the scalar pass's score products are exact over the dequantized values
//!   (`code·d` never rounds in f32: a code's 7 significant bits against the
//!   f16 scale's 11 leave 6 of f32's 24), so its band is the scalar twin's
//!   over the dequantized keys and the f16-rounded dequantized values (the
//!   V tile's own format) — the f64 exact attention recomputed on those
//!   values, `bound_ours` unchanged;
//! - the tensor-core pass and the prefill stage the dequantized values
//!   rounded to f16 — the twin tiles' own bits — so each is held bit for bit
//!   to its f16 twin run on the synthesized f16 cache of the dequantized
//!   values (`f32_to_f16_bits` of each): the tiles hold the same bits, and
//!   the walk after them is the twin's verbatim, so the two launches must
//!   agree to the bit (a band would only re-derive the twin's own gate).
//!
//! Bit identity against a host oracle is not claimed for either: the
//! exponential is the device's own (`dev_exp`), and the host cannot run it.
//!
//! Asserted for the q8 entries: on each set, every layer's scalar and mma
//! decode passes and the prefill launch — a rerun bit-identical, the cache
//! rows at or past the count holding the q8 sentinel bits (codes
//! `0xa5a5a5a5`, scales `0x5a5a`) changing no bit, the scalar pass within its
//! band, the mma pass and the prefill bit for bit their f16 twins on the
//! synthesized cache; on layer 0, the [`ROWS`] rotated rows of the prefill
//! shape each bit for bit their one-row launch (both decode passes), the
//! `key_count` fault with NaN in exactly the bad rows and every other row
//! the clean run's, the captured decode launch (two nodes) and prefill
//! launch (one node) replaying the eager bits, and a `kq` one word short
//! refused by name. Last (after the f16 window checks), `kq` at 8 bytes past
//! a 16-byte boundary through the q8 prefill refused by name before any
//! launch.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_flash: built without the `gpu` feature; see `just gate-gpu-qwen3moe-flash`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_flash", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::fault::{Fault, FaultSink, FaultSite, LAYER_NONE};
    use bloomery_gpu::flash_gqa::{
        FlashGqaKernels, GROUP, GqaArgs, GqaQ8Args, HEAD, KEY_TILE, SEG_KEYS, partials_ms_len,
        partials_v_len, segments_for,
    };
    use bloomery_gpu::flash_gqa_prefill::{
        FlashGqaPrefill, GqaPrefillArgs, GqaPrefillQ8Args, KEY_TILE as PREF_TILE,
    };
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{Gpu, GpuError, window};
    use bloomery_gpu_gates::qwen3moe::dev::{SENTINEL_Q8_CODE, SENTINEL_Q8_SCALE};
    use bloomery_gpu_gates::qwen3moe::{f16_logical_bits, step_sets};
    use bloomery_gpu_gates::rounding::{U, gamma};
    use bloomery_gpu_gates::{
        GateError, activations, bits_equal, checks_failed, mask_bits_in, open_split,
        ref_tensor_logical_in, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::{Q8Block, f32_to_f16_bits, half_to_f32};
    use model::arch::Arch;
    use model::arch::deepseek2::attn::quantize_q8_0;
    use model::arch::qwen3moe::hparams::Hparams;
    use std::mem::ManuallyDrop;

    /// An f16 NaN the padded rows are overwritten with.
    const NAN16: u16 = 0x7e00;

    /// Query rows of the prefill-shape check: the prefill's pass width.
    const ROWS: usize = 8;

    /// Rows of the prefill flash's multi-row check on each set's layer 0:
    /// two query tiles and one row of a third.
    const ROWS_PREF: usize = 17;

    /// The prefill flash's seeded launches: row counts and first positions.
    const SEED_T: [usize; 5] = [1, 17, 64, 65, 512];
    const SEED_P0: [usize; 4] = [0, 63, 64, 1000];

    /// Cache rows past the seeded launch's largest count, NaN in the padded
    /// run.
    const SEED_PAD: usize = 40;

    /// The seeded query's scale over [`activations`]' [-1, 1): scores of a
    /// few units, so the weights spread over several orders of magnitude.
    const SEED_Q_SCALE: f32 = 3.0;

    /// The key heads of the seeded cache: the model's.
    const SEED_KV: usize = 4;

    /// One layer's inputs on the card.
    struct Inputs {
        q: DeviceBuffer<f32>,
        kc: DeviceBuffer<u16>,
        vc: DeviceBuffer<u16>,
        n_keys: DeviceBuffer<u32>,
    }

    /// The launch geometry every run of a layer shares.
    struct Geom {
        scale: f32,
        n_kv: usize,
        ctx: usize,
    }

    /// Rows of the prefill-shape check (module doc): row `t` is the step's
    /// query `q` with its `n_head` heads rotated by `t`.
    fn rotated_rows(q: &[f32], n_head: usize) -> Vec<Vec<f32>> {
        (0..ROWS)
            .map(|t| {
                (0..n_head)
                    .flat_map(|h| q[((h + t) % n_head) * HEAD..][..HEAD].iter().copied())
                    .collect()
            })
            .collect()
    }

    /// Their live counts, spread from one key to `live`.
    fn spread_counts(live: usize) -> Result<Vec<u32>, GateError> {
        (0..ROWS)
            .map(|t| u32::try_from(1 + t * (live - 1) / (ROWS - 1)).map_err(Into::into))
            .collect()
    }

    /// One launch of `m` query rows (`q` and `n_keys` hold `m` each) over
    /// the cache planes, into fresh scratch, raising on `fault`, read back.
    fn run_once(
        k: &FlashGqaKernels,
        (stream, fault): (&CudaStream, FaultSink),
        q: &DeviceBuffer<f32>,
        n_keys: &DeviceBuffer<u32>,
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        g: &Geom,
        mma: bool,
    ) -> Result<Vec<f32>, GateError> {
        let (n_head, m) = (g.n_kv * GROUP, n_keys.len());
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(m, n_head, g.ctx))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, n_head, g.ctx))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * n_head * HEAD)?;
        k.enqueue_pass(
            stream,
            GqaArgs {
                q,
                kc,
                vc,
                n_keys,
                scale: g.scale,
                n_kv: g.n_kv,
                ctx: g.ctx,
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

    /// The prefill shape (module doc): one launch of [`ROWS`] rows against
    /// each row's one-row launch. Returns whether every row matched.
    fn rows_check(
        k: &FlashGqaKernels,
        gpu: &Gpu,
        q: &[f32],
        inp: &Inputs,
        live: usize,
        g: &Geom,
        mma: bool,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let run = (stream, gpu.unlabelled_sink());
        let n_head = g.n_kv * GROUP;
        let rows = rotated_rows(q, n_head);
        let limits = spread_counts(live)?;
        let cache = (&inp.kc, &inp.vc);
        let all = run_once(
            k,
            run,
            &DeviceBuffer::from_host(stream, &rows.concat())?,
            &DeviceBuffer::from_host(stream, &limits)?,
            cache,
            g,
            mma,
        )?;
        let mut same = true;
        for (t, row) in rows.iter().enumerate() {
            let one = run_once(
                k,
                run,
                &DeviceBuffer::from_host(stream, row)?,
                &DeviceBuffer::from_host(stream, &limits[t..=t])?,
                cache,
                g,
                mma,
            )?;
            same &= bits_equal(&all[t * n_head * HEAD..(t + 1) * n_head * HEAD], &one);
        }
        println!(
            "rows pass={} m={ROWS} keys={limits:?} ctx={}: each row = its one-row launch bit for bit {}",
            if mma { "mma" } else { "scalar" },
            g.ctx,
            verdict(same)
        );
        Ok(same)
    }

    /// The decode flash's refusal (module doc): the prefill-shape launch with
    /// row 3's count past the cache and row 5's zero, labelled with layer 13,
    /// against the same launch clean. Returns whether every check held.
    fn decode_fault(
        k: &FlashGqaKernels,
        gpu: &Gpu,
        q: &[f32],
        inp: &Inputs,
        live: usize,
        g: &Geom,
        mma: bool,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let layer = 13usize;
        let run = (stream, gpu.layer_sink(layer)?);
        let want = Some(Fault::at(u32::try_from(layer)?, FaultSite::KeyCount));
        let n_head = g.n_kv * GROUP;
        let w = n_head * HEAD;
        let qd = DeviceBuffer::from_host(stream, &rotated_rows(q, n_head).concat())?;
        let clean = spread_counts(live)?;
        let (bad_hi, bad_zero) = (3usize, 5usize);
        let mut bad = clean.clone();
        bad[bad_hi] = u32::try_from(g.ctx + 1)?;
        bad[bad_zero] = 0;
        let cache = (&inp.kc, &inp.vc);
        let before = gpu.fault()?;
        let y = run_once(
            k,
            run,
            &qd,
            &DeviceBuffer::from_host(stream, &clean)?,
            cache,
            g,
            mma,
        )?;
        let after_clean = gpu.fault()?;
        let yb = run_once(
            k,
            run,
            &qd,
            &DeviceBuffer::from_host(stream, &bad)?,
            cache,
            g,
            mma,
        )?;
        let raised = gpu.take_fault()?;
        let mut others_same = true;
        let mut bad_nan = true;
        for r in 0..ROWS {
            let (a, b) = (&y[r * w..(r + 1) * w], &yb[r * w..(r + 1) * w]);
            if r == bad_hi || r == bad_zero {
                bad_nan &= b.iter().all(|v| v.is_nan());
            } else {
                others_same &= bits_equal(a, b);
            }
        }
        let y2 = run_once(
            k,
            run,
            &qd,
            &DeviceBuffer::from_host(stream, &clean)?,
            cache,
            g,
            mma,
        )?;
        let clean_again = bits_equal(&y, &y2) && gpu.fault()?.is_none();
        let pass = before.is_none()
            && after_clean.is_none()
            && raised == want
            && bad_nan
            && others_same
            && clean_again;
        println!(
            "decode fault pass={} m={ROWS} counts {} (past ctx {}) and 0 at rows {bad_hi}, {bad_zero}: word \
             \"{}\" (want \"{}\", clean before {} and after the clean run {}), those rows NaN {bad_nan}, \
             other rows bit-identical {others_same}, clean rerun bits and word clean {clean_again} {}",
            if mma { "mma" } else { "scalar" },
            g.ctx + 1,
            g.ctx,
            raised.map_or_else(|| "none".to_owned(), |f| f.to_string()),
            want.map_or_else(String::new, |f| f.to_string()),
            before.is_none(),
            after_clean.is_none(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The prefill flash's multi-row check on a set (module doc): [`ROWS_PREF`]
    /// rows (at most the step's count) at the counts ending at `live`, row `t`
    /// the step's query with its heads rotated by `t`: every row within its
    /// bound, and bit for bit the row launched alone. Returns whether both
    /// held.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernel, the card, the step's query and cache on both sides, the geometry"
    )]
    fn prefill_rows(
        k: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &[f32],
        inp: &Inputs,
        kf: &[f32],
        vf: &[f32],
        live: usize,
        g: &Geom,
    ) -> Result<bool, GateError> {
        let n_head = g.n_kv * GROUP;
        let t = ROWS_PREF.min(live);
        let rows: Vec<f32> = (0..t)
            .flat_map(|r| {
                (0..n_head)
                    .flat_map(move |h| q[((h + r) % n_head) * HEAD..][..HEAD].iter().copied())
            })
            .collect();
        let counts: Vec<u32> = (0..t)
            .map(|r| u32::try_from(live - t + 1 + r))
            .collect::<Result<_, _>>()?;
        let cache = (&inp.kc, &inp.vc);
        let qd = DeviceBuffer::from_host(gpu.stream(), &rows)?;
        let all = run_pref(k, gpu, &qd, &counts, cache, g)?;
        let hc = HostCache {
            kf: kf.to_vec(),
            vf: vf.to_vec(),
        };
        let cu: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
        let (band, worst) = band_rows(&rows, &cu, &hc, g, &all);
        let differ = rows_alone(k, gpu, &rows, &counts, cache, g, &all)?;
        let pass = band && differ == 0;
        println!(
            "prefill rows T={t} keys={:?}..={live} ctx={}: measured/bound {worst:.3e} band={band} \
             rows_alone_differing={differ} {}",
            counts.first(),
            g.ctx,
            verdict(pass)
        );
        Ok(pass)
    }

    /// The exact attention of one head in f64, and each side's bound.
    struct Exact {
        o: Vec<f64>,
        bound_ours: Vec<f64>,
        bound_mma: Vec<f64>,
        bound_ik: Vec<f64>,
        bound_pref: Vec<f64>,
    }

    fn exact(q: &[f32], kh: &[f32], vh: &[f32], n: usize, scale: f32) -> Exact {
        let sc = f64::from(scale);
        let s: Vec<f64> = (0..n)
            .map(|j| {
                let k = &kh[j * HEAD..(j + 1) * HEAD];
                sc * q
                    .iter()
                    .zip(k)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
            })
            .collect();
        let a: Vec<f64> = (0..n)
            .map(|j| {
                let k = &kh[j * HEAD..(j + 1) * HEAD];
                sc.abs()
                    * q.iter()
                        .zip(k)
                        .map(|(&a, &b)| (f64::from(a) * f64::from(b)).abs())
                        .sum::<f64>()
            })
            .collect();
        // The tensor-core pass's scores also carry the query's f16 rounding:
        // 2^-11 of a normal value, 2^-25 absolute below f16's normal range
        // (2^-14 of the key magnitudes then, relative to the 2^-11 term).
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
        let r_ik = (n.div_ceil(32) + 33) as f64;
        let (es_o, es_i) = (gamma(35), gamma(129));
        // The tensor core's f32 accumulation order is not documented and may
        // truncate: twice γ(130) for the dot of exact f16 products.
        let es_m = 2f64.powi(-11) + 2.0 * gamma(130);
        let (acc_o, acc_i) = (gamma(SEG_KEYS + segs + 8), gamma(n + 40));
        // The prefill flash (module doc): one rescale per 64-key tile, the
        // f16 weights normalized by their own sum, the tensor core's value
        // accumulation and the lanes' weight sums.
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
            for d in 0..HEAD {
                o[d] += w * f64::from(vh[j * HEAD + d]);
            }
        }
        let mut bo = vec![0.0f64; HEAD];
        let mut bm = vec![0.0f64; HEAD];
        let mut bi = vec![0.0f64; HEAD];
        let mut bp = vec![0.0f64; HEAD];
        for d in 0..HEAD {
            let (mut t_o, mut t_m, mut t_i, mut t_p, mut pv) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for j in 0..n {
                let v = f64::from(vh[j * HEAD + d]);
                let dev = (v - o[d]).abs();
                let ee = 4.0 * U + 2.0 * U * x[j];
                t_o += pb[j] * (es_o * a[j] + ee + 4.0 * U * r_ours + 2.0 * U * xmax) * dev;
                t_m += pb[j] * (es_m * a16[j] + ee + 4.0 * U * r_ours + 2.0 * U * xmax) * dev;
                t_i += pb[j] * (es_i * a[j] + ee + 4.0 * U * r_ik + 2.0 * U * xmax) * dev;
                t_p += (pb[j] * (es_m * a16[j] + ee + 4.0 * U * r_pref + 2.0 * U * xmax)
                    + w16[j]
                    + pb[j] * dbar)
                    * dev;
                pv += pb[j] * v.abs();
            }
            bo[d] = t_o + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bm[d] = t_m + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bi[d] = t_i + acc_i * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bp[d] = t_p + acc_pv * pv + acc_l * o[d].abs() + 2.0 * U * o[d].abs();
        }
        Exact {
            o,
            bound_ours: bo,
            bound_mma: bm,
            bound_ik: bi,
            bound_pref: bp,
        }
    }

    /// One prefill launch of `n_keys.len()` rows (`q` holds that many) over
    /// the cache planes, into fresh output, read back. The fault word is the
    /// caller's to read.
    fn run_pref(
        k: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &DeviceBuffer<f32>,
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        g: &Geom,
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let (n_head, t) = (g.n_kv * GROUP, n_keys.len());
        let nk = DeviceBuffer::from_host(stream, n_keys)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, t * n_head * HEAD)?;
        k.enqueue(
            stream,
            GqaPrefillArgs {
                q,
                kc,
                vc,
                n_keys: &nk,
                scale: g.scale,
                n_head,
                n_kv: g.n_kv,
                ctx: g.ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// The host side of one cache: f32 values of its f16 planes.
    struct HostCache {
        kf: Vec<f32>,
        vf: Vec<f32>,
    }

    /// Every (row, head) of `y` against its exact value: `q` holds the rows'
    /// query heads token-major, `counts[t]` row `t`'s keys. Returns whether
    /// every value is within its prefill bound, and the largest measured
    /// over bound. Rows are shared among worker threads; each (row, head) is
    /// its own f64 computation.
    fn band_rows(
        q: &[f32],
        counts: &[usize],
        cache: &HostCache,
        g: &Geom,
        y: &[f32],
    ) -> (bool, f64) {
        let n_head = g.n_kv * GROUP;
        let workers = std::thread::available_parallelism().map_or(8, |n| n.get().min(16));
        let chunk = counts.len().div_ceil(workers).max(1);
        let parts: Vec<(bool, f64)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..counts.len())
                .step_by(chunk)
                .map(|r0| {
                    sc.spawn(move || {
                        let (mut ok, mut worst) = (true, 0.0f64);
                        for (t, &n) in counts.iter().enumerate().skip(r0).take(chunk) {
                            for h in 0..n_head {
                                let plane = (h / GROUP) * g.ctx * HEAD;
                                let row = (t * n_head + h) * HEAD;
                                let ex = exact(
                                    &q[row..row + HEAD],
                                    &cache.kf[plane..plane + n * HEAD],
                                    &cache.vf[plane..plane + n * HEAD],
                                    n,
                                    g.scale,
                                );
                                for d in 0..HEAD {
                                    let e = (f64::from(y[row + d]) - ex.o[d]).abs();
                                    worst = worst.max(e / ex.bound_pref[d]);
                                    ok &= e <= ex.bound_pref[d];
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

    /// Each row of `all` (one launch of `q`'s rows at `counts`) against the
    /// same row launched alone. Returns the rows that differ in any bit.
    fn rows_alone(
        k: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &[f32],
        counts: &[u32],
        cache: (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        g: &Geom,
        all: &[f32],
    ) -> Result<usize, GateError> {
        let w = g.n_kv * GROUP * HEAD;
        let mut differ = 0usize;
        for (t, &c) in counts.iter().enumerate() {
            let qt = DeviceBuffer::from_host(gpu.stream(), &q[t * w..(t + 1) * w])?;
            let one = run_pref(k, gpu, &qt, &[c], cache, g)?;
            differ += usize::from(!bits_equal(&all[t * w..(t + 1) * w], &one));
        }
        Ok(differ)
    }

    // ------------------------------------------------ the q8_0 read path

    /// One side's Q8_0 planes of a cache — `quantize_q8_0` over its f32
    /// values a row, `q8_0_planes`' packing — with the two views the reads
    /// hold: the dequantized values `code·d` (exact in f32) and their f16
    /// rounding (the tiles' own format). `sentil`'s padded twin writes the
    /// sentinel bits into every row at or past `live`, which no live key
    /// reads.
    struct Q8Side {
        codes: Vec<u32>,
        scales: Vec<u16>,
        vals: Vec<f32>,
        f16: Vec<u16>,
    }

    impl Q8Side {
        fn build(vals: &[f32], head: usize) -> Q8Side {
            let n = vals.len() / head;
            let mut codes = Vec::with_capacity(n * head / 4);
            let mut scales = Vec::with_capacity(n * head / 32);
            let mut deq = Vec::with_capacity(vals.len());
            for row in 0..n {
                let blocks: Vec<Q8Block> = vals[row * head..(row + 1) * head]
                    .chunks(32)
                    .map(quantize_q8_0)
                    .collect();
                let (qs, ds) = q8_0_planes(&blocks);
                codes.extend_from_slice(&qs);
                scales.extend_from_slice(&ds);
                for b in &blocks {
                    let d = half_to_f32(b.d);
                    for &c in &b.q {
                        // The one product the format takes, exact in f32.
                        deq.push(f32::from(c) * d);
                    }
                }
            }
            Q8Side {
                f16: deq.iter().map(|&v| f32_to_f16_bits(v)).collect(),
                codes,
                scales,
                vals: deq,
            }
        }

        /// The planes with every row at or past `live` the sentinel bits.
        fn sentinel(&self, head: usize, ctx: usize, live: usize) -> (Vec<u32>, Vec<u16>) {
            let (mut q, mut d) = (self.codes.clone(), self.scales.clone());
            for row in live..ctx {
                let (qw, sw) = (row * head / 4, row * head / 32);
                q[qw..qw + head / 4].fill(SENTINEL_Q8_CODE);
                d[sw..sw + head / 32].fill(SENTINEL_Q8_SCALE);
            }
            (q, d)
        }
    }

    /// The q8 cache of one layer: the four planes (and their sentinel twins)
    /// on the card, and the host's two views of the values.
    struct Q8Cache {
        kq: DeviceBuffer<u32>,
        kd: DeviceBuffer<u16>,
        vq: DeviceBuffer<u32>,
        vd: DeviceBuffer<u16>,
        knq: DeviceBuffer<u32>,
        knd: DeviceBuffer<u16>,
        vnq: DeviceBuffer<u32>,
        vnd: DeviceBuffer<u16>,
        /// The synthesized f16 cache of the dequantized values (the mma and
        /// prefill bit anchors' twin launch).
        k16: DeviceBuffer<u16>,
        v16: DeviceBuffer<u16>,
        kf: Vec<f32>,
        vf16: Vec<f32>,
    }

    impl Q8Cache {
        fn build(
            stream: &CudaStream,
            kf: &[f32],
            vf: &[f32],
            g: &Geom,
            live: usize,
        ) -> Result<Q8Cache, GateError> {
            let (ks, vs) = (Q8Side::build(kf, HEAD), Q8Side::build(vf, HEAD));
            let (knq, knd) = ks.sentinel(HEAD, g.ctx, live);
            let (vnq, vnd) = vs.sentinel(HEAD, g.ctx, live);
            Ok(Q8Cache {
                kq: DeviceBuffer::from_host(stream, &ks.codes)?,
                kd: DeviceBuffer::from_host(stream, &ks.scales)?,
                vq: DeviceBuffer::from_host(stream, &vs.codes)?,
                vd: DeviceBuffer::from_host(stream, &vs.scales)?,
                knq: DeviceBuffer::from_host(stream, &knq)?,
                knd: DeviceBuffer::from_host(stream, &knd)?,
                vnq: DeviceBuffer::from_host(stream, &vnq)?,
                vnd: DeviceBuffer::from_host(stream, &vnd)?,
                k16: DeviceBuffer::from_host(stream, &ks.f16)?,
                v16: DeviceBuffer::from_host(stream, &vs.f16)?,
                kf: ks.vals,
                vf16: from16(&vs.f16),
            })
        }

        fn planes(
            &self,
        ) -> (
            &DeviceBuffer<u32>,
            &DeviceBuffer<u16>,
            &DeviceBuffer<u32>,
            &DeviceBuffer<u16>,
        ) {
            (&self.kq, &self.kd, &self.vq, &self.vd)
        }

        fn sentinel_planes(
            &self,
        ) -> (
            &DeviceBuffer<u32>,
            &DeviceBuffer<u16>,
            &DeviceBuffer<u32>,
            &DeviceBuffer<u16>,
        ) {
            (&self.knq, &self.knd, &self.vnq, &self.vnd)
        }
    }

    /// `from16` is in the qwen35 gate; a local twin keeps this gate's helpers
    /// one place.
    fn from16(v: &[u16]) -> Vec<f32> {
        v.iter().map(|&h| half_to_f32(h)).collect()
    }

    /// One launch of `m` query rows of the q8 decode pass over the four
    /// planes, into fresh scratch, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernel, the stream and sink, the query and counts, the planes, the geometry, the pass"
    )]
    fn run_once_q8(
        k: &FlashGqaKernels,
        (stream, fault): (&CudaStream, FaultSink),
        q: &DeviceBuffer<f32>,
        n_keys: &DeviceBuffer<u32>,
        c: &Q8Cache,
        g: &Geom,
        mma: bool,
        use_sentinel: bool,
    ) -> Result<Vec<f32>, GateError> {
        let (n_head, m) = (g.n_kv * GROUP, n_keys.len());
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(m, n_head, g.ctx))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, n_head, g.ctx))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * n_head * HEAD)?;
        let (kq, kd, vq, vd) = if use_sentinel {
            c.sentinel_planes()
        } else {
            c.planes()
        };
        k.enqueue_pass_q8(
            stream,
            GqaQ8Args {
                q,
                kq,
                kd,
                vq,
                vd,
                n_keys,
                scale: g.scale,
                n_kv: g.n_kv,
                ctx: g.ctx,
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

    /// One q8 prefill launch of `n_keys.len()` rows over the four planes
    /// (`use_sentinel`: their sentinel twins, for a launch whose largest
    /// count is `live`), into fresh output, read back. The fault word is the
    /// caller's to read.
    fn run_pref_q8(
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &DeviceBuffer<f32>,
        n_keys: &[u32],
        c: &Q8Cache,
        g: &Geom,
        use_sentinel: bool,
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let (n_head, t) = (g.n_kv * GROUP, n_keys.len());
        let nk = DeviceBuffer::from_host(stream, n_keys)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, t * n_head * HEAD)?;
        let (kq, kd, vq, vd) = if use_sentinel {
            c.sentinel_planes()
        } else {
            c.planes()
        };
        kp.enqueue_q8(
            stream,
            GqaPrefillQ8Args {
                q,
                kq,
                kd,
                vq,
                vd,
                n_keys: &nk,
                scale: g.scale,
                n_head,
                n_kv: g.n_kv,
                ctx: g.ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// The q8_0 read path's clauses on one layer of a set (module doc): both
    /// decode passes and the prefill launch over the cache's Q8_0 form —
    /// rerun, sentinel padding, the scalar pass's band, the mma pass's and
    /// the prefill's bit identity with their f16 twins on the synthesized
    /// cache; `layer0` adds the prefill-shape rows, the fault and the graphs.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernels, the card, the layer's query and cache values, the geometry, the flags"
    )]
    fn flash_q8_layer(
        k: &FlashGqaKernels,
        kp: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &[f32],
        kf: &[f32],
        vf: &[f32],
        live: usize,
        g: &Geom,
        label: &str,
        l: usize,
        layer0: bool,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let run = (stream, unl);
        let n_head = g.n_kv * GROUP;
        let c = Q8Cache::build(stream, kf, vf, g, live)?;
        let qd = DeviceBuffer::from_host(stream, q)?;
        let nk = DeviceBuffer::from_host(stream, &[u32::try_from(live)?])?;
        let mut ok = true;
        for mma in [false, true] {
            let name = if mma { "mma" } else { "scalar" };
            let y = run_once_q8(k, run, &qd, &nk, &c, g, mma, false)?;
            let y2 = run_once_q8(k, run, &qd, &nk, &c, g, mma, false)?;
            let ys = run_once_q8(k, run, &qd, &nk, &c, g, mma, true)?;
            let (rerun, sentinel) = (bits_equal(&y, &y2), bits_equal(&y, &ys));
            let mut pass_ok = rerun && sentinel;
            if mma {
                // The twin on the synthesized f16 cache: the same bits.
                let tw = run_once(k, run, &qd, &nk, (&c.k16, &c.v16), g, mma)?;
                let twin_same = bits_equal(&y, &tw);
                pass_ok &= twin_same;
                println!(
                    "flash q8 set={label} layer={l} pass={name} keys={live}: = the f16 mma twin \
                     on the synthesized f16 cache bit for bit {twin_same} {}",
                    verdict(pass_ok)
                );
            } else {
                // The exact attention of the scalar pass's oracle: the
                // dequantized keys, the f16-rounded dequantized values.
                let mut worst = 0.0f64;
                let mut band = true;
                for h in 0..n_head {
                    let plane = (h / GROUP) * g.ctx * HEAD;
                    let ex = exact(
                        &q[h * HEAD..(h + 1) * HEAD],
                        &c.kf[plane..plane + live * HEAD],
                        &c.vf16[plane..plane + live * HEAD],
                        live,
                        g.scale,
                    );
                    for d in 0..HEAD {
                        let e = (f64::from(y[h * HEAD + d]) - ex.o[d]).abs();
                        worst = worst.max(e / ex.bound_ours[d]);
                        band &= e <= ex.bound_ours[d];
                    }
                }
                pass_ok &= band;
                println!(
                    "flash q8 set={label} layer={l} pass={name} keys={live} ctx={}: measured/bound \
                     {worst:.3e} band={band} rerun={rerun} sentinel_same={sentinel} {}",
                    g.ctx,
                    verdict(pass_ok)
                );
            }
            ok &= pass_ok;
        }

        // The prefill launch: bit for bit the twin on the synthesized cache,
        // a rerun and the sentinel padding the same bits.
        let count = [u32::try_from(live)?];
        let yp = run_pref_q8(kp, gpu, &qd, &count, &c, g, false)?;
        let yp2 = run_pref_q8(kp, gpu, &qd, &count, &c, g, false)?;
        let yps = run_pref_q8(kp, gpu, &qd, &count, &c, g, true)?;
        let tw = run_pref(kp, gpu, &qd, &count, (&c.k16, &c.v16), g)?;
        let (rerun, sentinel) = (bits_equal(&yp, &yp2), bits_equal(&yp, &yps));
        let twin_same = bits_equal(&yp, &tw);
        let pref_ok = rerun && sentinel && twin_same;
        println!(
            "prefill q8 set={label} layer={l} keys={live}: = the f16 prefill twin on the \
             synthesized f16 cache bit for bit {twin_same} rerun={rerun} sentinel_same={sentinel} {}",
            verdict(pref_ok)
        );
        ok &= pref_ok;

        if layer0 {
            // The prefill shape: one launch of ROWS rotated rows, each row bit
            // for bit its one-row launch, both passes.
            let rows = rotated_rows(q, n_head);
            let limits = spread_counts(live)?;
            let rd = DeviceBuffer::from_host(stream, &rows.concat())?;
            let ld = DeviceBuffer::from_host(stream, &limits)?;
            for mma in [false, true] {
                let name = if mma { "mma" } else { "scalar" };
                let all = run_once_q8(k, run, &rd, &ld, &c, g, mma, false)?;
                let mut same = true;
                for (t, row) in rows.iter().enumerate() {
                    let one = run_once_q8(
                        k,
                        run,
                        &DeviceBuffer::from_host(stream, row)?,
                        &DeviceBuffer::from_host(stream, &limits[t..=t])?,
                        &c,
                        g,
                        mma,
                        false,
                    )?;
                    same &= bits_equal(&all[t * n_head * HEAD..(t + 1) * n_head * HEAD], &one);
                }
                println!(
                    "flash q8 rows pass={name} m={ROWS} keys={limits:?} ctx={}: each row = its \
                     one-row launch bit for bit {same} {}",
                    g.ctx,
                    verdict(same)
                );
                ok &= same;
            }

            // The refusal path: rows 3 and 5 of the eight-row launch.
            let (bad_hi, bad_zero) = (3usize, 5usize);
            let mut bad = limits.clone();
            bad[bad_hi] = u32::try_from(g.ctx + 1)?;
            bad[bad_zero] = 0;
            let bd = DeviceBuffer::from_host(stream, &bad)?;
            let clean = run_once_q8(k, run, &rd, &ld, &c, g, false, false)?;
            let before = gpu.fault()?;
            let yb = run_once_q8(
                k,
                (stream, gpu.layer_sink(13)?),
                &rd,
                &bd,
                &c,
                g,
                false,
                false,
            )?;
            let raised = gpu.take_fault()?;
            let want = Some(Fault::at(13, FaultSite::KeyCount));
            let w = n_head * HEAD;
            let (mut others, mut nan) = (true, true);
            for r in 0..ROWS {
                let (a, b) = (&clean[r * w..(r + 1) * w], &yb[r * w..(r + 1) * w]);
                if r == bad_hi || r == bad_zero {
                    nan &= b.iter().all(|v| v.is_nan());
                } else {
                    others &= bits_equal(a, b);
                }
            }
            let fault_ok = before.is_none() && raised == want && nan && others;
            println!(
                "flash q8 fault: counts {} (past ctx {}) and 0 at rows {bad_hi}, {bad_zero}: word \
                 {raised:?} (want {want:?}), those rows NaN {nan}, other rows bit-identical {others} {}",
                g.ctx + 1,
                g.ctx,
                verdict(fault_ok)
            );
            ok &= fault_ok;

            // The captured decode launch (two nodes) and prefill launch (one
            // node), replaying the eager bits.
            for mma in [false, true] {
                let name = if mma { "mma" } else { "scalar" };
                let (mut pv, mut pms, mut yg) = (
                    DeviceBuffer::<f32>::zeroed(stream, partials_v_len(ROWS, n_head, g.ctx))?,
                    DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(ROWS, n_head, g.ctx))?,
                    DeviceBuffer::<f32>::zeroed(stream, ROWS * n_head * HEAD)?,
                );
                let (kq, kd, vq, vd) = c.planes();
                let graph = gpu.capture(|s| {
                    k.enqueue_pass_q8(
                        s,
                        GqaQ8Args {
                            q: &rd,
                            kq,
                            kd,
                            vq,
                            vd,
                            n_keys: &ld,
                            scale: g.scale,
                            n_kv: g.n_kv,
                            ctx: g.ctx,
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
                let eager = run_once_q8(k, run, &rd, &ld, &c, g, mma, false)?;
                let same = bits_equal(&yg.to_host_vec(stream)?, &eager);
                let nodes = graph.node_count();
                let graph_ok = same && nodes == 2;
                println!(
                    "flash q8 graph pass={name} m={ROWS}: eager_vs_graph_bit_identical={same} \
                     graph_nodes={nodes} {}",
                    verdict(graph_ok)
                );
                ok &= graph_ok;
            }
            let mut yg = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
            let (kq, kd, vq, vd) = c.planes();
            let graph = gpu.capture(|s| {
                kp.enqueue_q8(
                    s,
                    GqaPrefillQ8Args {
                        q: &qd,
                        kq,
                        kd,
                        vq,
                        vd,
                        n_keys: &nk,
                        scale: g.scale,
                        n_head,
                        n_kv: g.n_kv,
                        ctx: g.ctx,
                        t: 1,
                        fault: gpu.unlabelled_sink(),
                        y: &mut yg,
                    },
                )
            })?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let same = bits_equal(&yg.to_host_vec(stream)?, &yp);
            let nodes = graph.node_count();
            let graph_ok = same && nodes == 1;
            println!(
                "prefill q8 graph T=1: eager_vs_graph_bit_identical={same} graph_nodes={nodes} {}",
                verdict(graph_ok)
            );
            ok &= graph_ok;

            // A `kq` one word short: refused by name.
            let short = DeviceBuffer::<u32>::zeroed(stream, c.kq.len().saturating_sub(1).max(1))?;
            let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(1, n_head, g.ctx))?;
            let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head, g.ctx))?;
            let mut yr = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
            let r = k.enqueue_pass_q8(
                stream,
                GqaQ8Args {
                    q: &qd,
                    kq: &short,
                    kd: &c.kd,
                    vq: &c.vq,
                    vd: &c.vd,
                    n_keys: &nk,
                    scale: g.scale,
                    n_kv: g.n_kv,
                    ctx: g.ctx,
                    m: 1,
                    part_v: &mut pv,
                    part_ms: &mut pms,
                    fault: unl,
                    y: &mut yr,
                },
                false,
            );
            let named = matches!(
                &r,
                Err(GpuError::Shape {
                    what: "flash_gqa::enqueue_q8",
                    detail,
                }) if detail.starts_with("kq.len()")
            );
            println!(
                "flash q8 refusal kq.len() {} < {}: {} {}",
                short.len(),
                c.kq.len(),
                r.err().map_or("accepted".to_string(), |e| e.to_string()),
                verdict(named)
            );
            ok &= named;
        }
        Ok(ok)
    }

    /// The q8 prefill's `kq` window check (module doc): the plane at 8 bytes
    /// past a 16-byte boundary (the code chunks are 16-byte copies) refused
    /// by name before any launch. The gate's last check: a launch through a
    /// misaligned window is a sticky error that ends the context.
    fn q8_window_refusal(kp: &FlashGqaPrefill, gpu: &Gpu, scale: f32) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (n_kv, t) = (SEED_KV, 17usize);
        let ctx = t + SEED_PAD;
        let g = Geom { scale, n_kv, ctx };
        let n_head = n_kv * GROUP;
        let kf: Vec<f32> = activations(HEAD, n_kv * ctx, 171)
            .into_iter()
            .map(f32_to_f16_bits)
            .map(half_to_f32)
            .collect();
        let c = Q8Cache::build(stream, &kf, &kf, &g, ctx)?;
        let q = activations(HEAD, n_head, 172);
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let nk = DeviceBuffer::from_host(stream, &[u32::try_from(ctx)?])?;
        let pad = DeviceBuffer::from_host(
            stream,
            &[&[0u32; 2][..], &c.kq.to_host_vec(stream)?].concat(),
        )?;
        let cu = gpu.context();
        // SAFETY: the window is `kq.len()` u32 starting 8 bytes into `pad`,
        // inside its own live allocation, which holds two u32 more than the
        // span; the window is given back after the refusal it checks.
        let k_mis = unsafe { window::<u32>(pad.cu_deviceptr() + 8, c.kq.len(), cu) };
        let mut y = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
        let r = kp.enqueue_q8(
            stream,
            GqaPrefillQ8Args {
                q: &qd,
                kq: &k_mis,
                kd: &c.kd,
                vq: &c.vq,
                vd: &c.vd,
                n_keys: &nk,
                scale,
                n_head,
                n_kv,
                ctx,
                t: 1,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
        );
        let named = matches!(
            &r,
            Err(GpuError::Shape {
                what: "flash_gqa_prefill::enqueue_q8",
                detail,
            }) if detail.starts_with("kq at ")
        );
        println!(
            "prefill q8 refusal kq window 8 bytes past a 16-byte boundary: {} {}",
            r.err().map_or("accepted".to_string(), |e| e.to_string()),
            verdict(named)
        );
        drop(ManuallyDrop::into_inner(k_mis).into_raw_parts());
        Ok(named)
    }

    /// The seeded launches (module doc), each against the exact value, its
    /// rows alone, NaN padding and a rerun; then the fault, the captured
    /// launch and a refusal. Returns whether every check passed.
    fn prefill_seeded(k: &FlashGqaPrefill, gpu: &Gpu, scale: f32) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let n_kv = SEED_KV;
        let n_head = n_kv * GROUP;
        let mut ok = true;
        let mut seed = 7u32;
        for &p0 in &SEED_P0 {
            for &t in &SEED_T {
                seed += 3;
                let ctx = p0 + t + SEED_PAD;
                let g = Geom { scale, n_kv, ctx };
                let q: Vec<f32> = activations(HEAD, t * n_head, seed)
                    .into_iter()
                    .map(|v| v * SEED_Q_SCALE)
                    .collect();
                let to16 =
                    |v: Vec<f32>| -> Vec<u16> { v.into_iter().map(f32_to_f16_bits).collect() };
                let kb = to16(activations(HEAD, n_kv * ctx, seed + 1));
                let vb = to16(activations(HEAD, n_kv * ctx, seed + 2));
                let hc = HostCache {
                    kf: kb.iter().map(|&h| half_to_f32(h)).collect(),
                    vf: vb.iter().map(|&h| half_to_f32(h)).collect(),
                };
                let live = p0 + t;
                let pad = |b: &[u16]| -> Vec<u16> {
                    b.iter()
                        .enumerate()
                        .map(|(i, &h)| if (i / HEAD) % ctx >= live { NAN16 } else { h })
                        .collect()
                };
                let (kc, vc) = (
                    DeviceBuffer::from_host(stream, &kb)?,
                    DeviceBuffer::from_host(stream, &vb)?,
                );
                let (kn, vn) = (
                    DeviceBuffer::from_host(stream, &pad(&kb))?,
                    DeviceBuffer::from_host(stream, &pad(&vb))?,
                );
                let qd = DeviceBuffer::from_host(stream, &q)?;
                let counts: Vec<u32> = (0..t)
                    .map(|i| u32::try_from(p0 + i + 1))
                    .collect::<Result<_, _>>()?;
                let y = run_pref(k, gpu, &qd, &counts, (&kc, &vc), &g)?;
                let y2 = run_pref(k, gpu, &qd, &counts, (&kc, &vc), &g)?;
                let y_nan = run_pref(k, gpu, &qd, &counts, (&kn, &vn), &g)?;
                let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &y_nan));
                let cu: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
                let (band, worst) = band_rows(&q, &cu, &hc, &g, &y);
                let differ = rows_alone(k, gpu, &q, &counts, (&kc, &vc), &g, &y)?;
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
        let ctx = p0 + t + SEED_PAD;
        let g = Geom { scale, n_kv, ctx };
        let q = activations(HEAD, t * n_head, 91);
        let kb: Vec<u16> = activations(HEAD, n_kv * ctx, 92)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        let vb: Vec<u16> = activations(HEAD, n_kv * ctx, 93)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        let (kc, vc) = (
            DeviceBuffer::from_host(stream, &kb)?,
            DeviceBuffer::from_host(stream, &vb)?,
        );
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let clean: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i + 1))
            .collect::<Result<_, _>>()?;
        let (bad_hi, bad_zero) = (3usize, 10usize);
        let mut bad = clean.clone();
        bad[bad_hi] = u32::try_from(ctx + 1)?;
        bad[bad_zero] = 0;
        let before = gpu.fault()?;
        let y = run_pref(k, gpu, &qd, &clean, (&kc, &vc), &g)?;
        let after_clean = gpu.fault()?;
        let yb = run_pref(k, gpu, &qd, &bad, (&kc, &vc), &g)?;
        let raised = gpu.fault()?;
        gpu.clear_fault()?;
        let w = n_head * HEAD;
        let mut others_same = true;
        let mut bad_nan = true;
        for r in 0..t {
            let (a, b) = (&y[r * w..(r + 1) * w], &yb[r * w..(r + 1) * w]);
            if r == bad_hi || r == bad_zero {
                bad_nan &= b.iter().all(|v| v.is_nan());
            } else {
                others_same &= bits_equal(a, b);
            }
        }
        let want = Fault::at(LAYER_NONE, FaultSite::KeyCount);
        let fault_ok = before.is_none()
            && after_clean.is_none()
            && raised == Some(want)
            && bad_nan
            && others_same;
        println!(
            "prefill fault: counts {} (past ctx {ctx}) and 0 at rows {bad_hi}, {bad_zero}: word \
             {raised:?} (want {want:?}, clean before {} and after the clean run {}), those rows NaN \
             {bad_nan}, other rows bit-identical {others_same} {}",
            ctx + 1,
            before.is_none(),
            after_clean.is_none(),
            verdict(fault_ok)
        );
        ok &= fault_ok;

        // The captured launch: one node, the eager bits.
        let nk = DeviceBuffer::from_host(stream, &clean)?;
        let mut yg = DeviceBuffer::<f32>::zeroed(stream, t * w)?;
        let graph = gpu.capture(|s| {
            k.enqueue(
                s,
                GqaPrefillArgs {
                    q: &qd,
                    kc: &kc,
                    vc: &vc,
                    n_keys: &nk,
                    scale,
                    n_head,
                    n_kv,
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
        let refused = k.enqueue(
            stream,
            GqaPrefillArgs {
                q: &qd,
                kc: &kc,
                vc: &vc,
                n_keys: &nk,
                scale,
                n_head: n_head - 1,
                n_kv,
                ctx,
                t,
                fault: gpu.unlabelled_sink(),
                y: &mut yr,
            },
        );
        let refuse_ok = refused.is_err();
        println!(
            "prefill refusal n_head={} over n_kv={n_kv}: {} {}",
            n_head - 1,
            refused
                .err()
                .map_or("accepted".to_string(), |e| e.to_string()),
            verdict(refuse_ok)
        );
        ok &= refuse_ok;
        ok &= misaligned_refusals(k, gpu, &q, (&kb, &vb), (&clean, &y), &g)?;
        Ok(ok)
    }

    /// Windows the kernel's loads cannot take (module doc), on the fault
    /// check's inputs `q`, `kb`/`vb` and counts `clean`, whose plain launch
    /// wrote `y`: the same windows at aligned offsets first — `q` two f32
    /// into its allocation, the planes eight f16 into theirs — accepted and
    /// bit for bit `y`; then `q` one f32 in (the query staging reads u64
    /// words) and `kc` or `vc` four f16 in (the tiles copy 16-byte pieces),
    /// each refused by name before a launch. The last check of the gate: a
    /// launch through a misaligned window is a sticky error that ends the
    /// context. Returns whether every check held.
    fn misaligned_refusals(
        k: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &[f32],
        (kb, vb): (&[u16], &[u16]),
        (clean, y): (&[u32], &[f32]),
        g: &Geom,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (n_head, t) = (g.n_kv * GROUP, clean.len());
        let qd = DeviceBuffer::from_host(stream, q)?;
        let (kc, vc) = (
            DeviceBuffer::from_host(stream, kb)?,
            DeviceBuffer::from_host(stream, vb)?,
        );
        let nk = DeviceBuffer::from_host(stream, clean)?;
        let q_pad = DeviceBuffer::from_host(stream, &[&[0.0f32; 2][..], q].concat())?;
        let k_pad = DeviceBuffer::from_host(stream, &[&[0u16; 8][..], kb].concat())?;
        let v_pad = DeviceBuffer::from_host(stream, &[&[0u16; 8][..], vb].concat())?;
        let ctx = gpu.context();
        // SAFETY: each window is `q.len()` f32 starting one or two f32 into
        // `q_pad`, or `kb.len()` (= `vb.len()`) f16 starting four or eight
        // f16 into `k_pad` or `v_pad` — inside its own live allocation, which
        // holds two f32 or eight f16 more than the span. The allocations
        // outlive every call below, and the windows are given back after them.
        let (q_al, q_mis, k_al, k_mis, v_al, v_mis) = unsafe {
            (
                window::<f32>(q_pad.cu_deviceptr() + 8, q.len(), ctx),
                window::<f32>(q_pad.cu_deviceptr() + 4, q.len(), ctx),
                window::<u16>(k_pad.cu_deviceptr() + 16, kb.len(), ctx),
                window::<u16>(k_pad.cu_deviceptr() + 8, kb.len(), ctx),
                window::<u16>(v_pad.cu_deviceptr() + 16, vb.len(), ctx),
                window::<u16>(v_pad.cu_deviceptr() + 8, vb.len(), ctx),
            )
        };
        let ya = run_pref(k, gpu, &q_al, clean, (&*k_al, &*v_al), g)?;
        let aligned_ok = bits_equal(&ya, y);
        println!(
            "prefill windows at aligned offsets (q 8 bytes, kc and vc 16 bytes into their \
             allocations): accepted, bit-identical to the plain launch {aligned_ok} {}",
            verdict(aligned_ok)
        );
        let mut ok = aligned_ok;
        let mut ym = DeviceBuffer::<f32>::zeroed(stream, t * n_head * HEAD)?;
        let cases = [
            ("q", "4 bytes past an 8-byte boundary", &*q_mis, &kc, &vc),
            ("kc", "8 bytes past a 16-byte boundary", &qd, &*k_mis, &vc),
            ("vc", "8 bytes past a 16-byte boundary", &qd, &kc, &*v_mis),
        ];
        for (name, off, q_in, kc_in, vc_in) in cases {
            let r = k.enqueue(
                stream,
                GqaPrefillArgs {
                    q: q_in,
                    kc: kc_in,
                    vc: vc_in,
                    n_keys: &nk,
                    scale: g.scale,
                    n_head,
                    n_kv: g.n_kv,
                    ctx: g.ctx,
                    t,
                    fault: gpu.unlabelled_sink(),
                    y: &mut ym,
                },
            );
            let named = matches!(
                r,
                Err(GpuError::Shape {
                    what: "flash_gqa_prefill::enqueue",
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

    /// One pass's measured-over-bound ratios over a set, and the plain
    /// relative distance to ik.
    #[derive(Default)]
    struct Ratios {
        ours: f64,
        ik: f64,
        pair: f64,
        plain: f64,
    }

    pub fn run() -> Result<(), GateError> {
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        bloomery_levers::at_main(&[])?;
        let split = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-flash")?;
        let hp = Hparams::read(&split)?;
        if hp.head_dim != HEAD {
            return Err(format!(
                "the kernel attends over {HEAD}-value heads; the file's heads are {}",
                hp.head_dim
            )
            .into());
        }
        // ik's `kq_scale`: one over the square root of the file's head width.
        let scale = 1.0f32 / (hp.head_dim as f32).sqrt();
        let gpu = Gpu::new()?;
        let k = FlashGqaKernels::load(gpu.context())?;
        let kp = FlashGqaPrefill::load(gpu.context())?;
        let stream = gpu.stream();
        println!(
            "gate_qwen3moe_flash: device {} — head {HEAD}, group {GROUP}, {SEG_KEYS}-key segments, scale {scale:e}",
            gpu.device_name()?
        );
        let mut ok = true;
        let mut graph_done = false;
        for (label, man) in step_sets()? {
            let mut rs = [Ratios::default(), Ratios::default()];
            let mut rp = Ratios::default();
            let (mut n_live, mut ctx_seen, mut layers) = (0usize, 0usize, 0usize);
            let mut set_ok = true;
            let mrow = man.input("KQ_mask", 0)?;
            let mask = mask_bits_in(&man.dir, mrow)?;
            let width = mrow.ne[0] as usize;
            let row0 = &mask[..width];
            let live = row0.iter().take_while(|&&b| b == 0).count();
            if live == 0 || row0[live..].iter().any(|&b| b != 0xfc00) {
                return Err(format!("{label}: KQ_mask row 0 is not a visible prefix").into());
            }
            while let Ok(fa) = man.tensor(&format!("fa-{layers}"), 0) {
                let l = layers;
                let qrow = man.tensor(&format!("q-{l}"), 0)?;
                let krow = man.tensor(&format!("k-{l}"), 0)?;
                let vrow = man.tensor(&format!("v-{l}"), 0)?;
                let (ctx, n_kv) = (krow.ne[1] as usize, krow.ne[2] as usize);
                let n_head = n_kv * GROUP;
                if krow.ne[0] as usize != HEAD
                    || vrow.ne != krow.ne
                    || qrow.ne != [HEAD as u64, 1, n_head as u64, 1]
                    || fa.ne != [HEAD as u64, n_head as u64, 1, 1]
                    || ctx != width
                {
                    return Err(format!(
                        "{label} layer {l}: q {:?} k {:?} v {:?} fa {:?} mask width {width}",
                        qrow.ne, krow.ne, vrow.ne, fa.ne
                    )
                    .into());
                }
                let q = ref_tensor_logical_in(&man.dir, qrow)?;
                let kb = f16_logical_bits(&man.dir, krow)?;
                let vb = f16_logical_bits(&man.dir, vrow)?;
                let want = ref_tensor_logical_in(&man.dir, fa)?;
                let pad = |b: &[u16]| -> Vec<u16> {
                    b.iter()
                        .enumerate()
                        .map(|(i, &h)| if (i / HEAD) % ctx >= live { NAN16 } else { h })
                        .collect()
                };
                let up = |kc: &[u16], vc: &[u16]| -> Result<Inputs, GateError> {
                    Ok(Inputs {
                        q: DeviceBuffer::from_host(stream, &q)?,
                        kc: DeviceBuffer::from_host(stream, kc)?,
                        vc: DeviceBuffer::from_host(stream, vc)?,
                        n_keys: DeviceBuffer::from_host(stream, &[live as u32])?,
                    })
                };
                let inp = up(&kb, &vb)?;
                let inp_nan = up(&pad(&kb), &pad(&vb))?;
                let kf: Vec<f32> = kb.iter().map(|&h| half_to_f32(h)).collect();
                let vf: Vec<f32> = vb.iter().map(|&h| half_to_f32(h)).collect();
                let exacts: Vec<Exact> = (0..n_head)
                    .map(|h| {
                        let plane = (h / GROUP) * ctx * HEAD;
                        exact(
                            &q[h * HEAD..(h + 1) * HEAD],
                            &kf[plane..plane + live * HEAD],
                            &vf[plane..plane + live * HEAD],
                            live,
                            scale,
                        )
                    })
                    .collect();
                let mx_ik = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let g = Geom { scale, n_kv, ctx };
                let mut ys = Vec::new();
                for (pass, mma) in [false, true].into_iter().enumerate() {
                    let run = |i: &Inputs| {
                        run_once(
                            &k,
                            (stream, gpu.unlabelled_sink()),
                            &i.q,
                            &i.n_keys,
                            (&i.kc, &i.vc),
                            &g,
                            mma,
                        )
                    };
                    let y = run(&inp)?;
                    let y2 = run(&inp)?;
                    let y_nan = run(&inp_nan)?;
                    let rerun = bits_equal(&y, &y2);
                    let nan_same = bits_equal(&y, &y_nan);
                    let mut layer_ok = rerun && nan_same;
                    let r = &mut rs[pass];
                    for (h, ex) in exacts.iter().enumerate() {
                        for d in 0..HEAD {
                            let (o, iv) =
                                (f64::from(y[h * HEAD + d]), f64::from(want[h * HEAD + d]));
                            let e = ex.o[d];
                            let bo = if mma {
                                ex.bound_mma[d]
                            } else {
                                ex.bound_ours[d]
                            };
                            let bi = ex.bound_ik[d];
                            let (d_o, d_i, d_p) = ((o - e).abs(), (iv - e).abs(), (o - iv).abs());
                            r.ours = r.ours.max(d_o / bo);
                            r.ik = r.ik.max(d_i / bi);
                            r.pair = r.pair.max(d_p / (bo + bi));
                            r.plain = r.plain.max(d_p / f64::from(mx_ik));
                            layer_ok &= d_o <= bo && d_i <= bi && d_p <= bo + bi;
                        }
                    }
                    if !layer_ok {
                        println!(
                            "flash set={label} layer={l} pass={} keys={live} ctx={ctx} rerun={rerun} \
                             nan_padding_same={nan_same} FAIL",
                            if mma { "mma" } else { "scalar" }
                        );
                    }
                    set_ok &= layer_ok;
                    if l == 0 {
                        set_ok &= rows_check(&k, &gpu, &q, &inp, live, &g, mma)?;
                        set_ok &= decode_fault(&k, &gpu, &q, &inp, live, &g, mma)?;
                    }
                    ys.push(y);
                }

                // The prefill flash: the step's row alone, at its count.
                let run_p =
                    |i: &Inputs| run_pref(&kp, &gpu, &i.q, &[live as u32], (&i.kc, &i.vc), &g);
                let yp = run_p(&inp)?;
                let (rerun_p, nan_p) = (
                    bits_equal(&yp, &run_p(&inp)?),
                    bits_equal(&yp, &run_p(&inp_nan)?),
                );
                let mut pref_ok = rerun_p && nan_p;
                for (h, ex) in exacts.iter().enumerate() {
                    for d in 0..HEAD {
                        let (o, iv) = (f64::from(yp[h * HEAD + d]), f64::from(want[h * HEAD + d]));
                        let (bp, bi) = (ex.bound_pref[d], ex.bound_ik[d]);
                        let (d_o, d_i, d_p) =
                            ((o - ex.o[d]).abs(), (iv - ex.o[d]).abs(), (o - iv).abs());
                        rp.ours = rp.ours.max(d_o / bp);
                        rp.ik = rp.ik.max(d_i / bi);
                        rp.pair = rp.pair.max(d_p / (bp + bi));
                        rp.plain = rp.plain.max(d_p / f64::from(mx_ik));
                        pref_ok &= d_o <= bp && d_p <= bp + bi;
                    }
                }
                if !pref_ok {
                    println!(
                        "flash set={label} layer={l} pass=prefill keys={live} ctx={ctx} rerun={rerun_p} \
                         nan_padding_same={nan_p} FAIL"
                    );
                }
                set_ok &= pref_ok;
                if l == 0 {
                    set_ok &= prefill_rows(&kp, &gpu, &q, &inp, &kf, &vf, live, &g)?;
                }

                // The q8_0 read path (module doc): every layer's three
                // entries over the cache's Q8_0 form, layer 0's extras once a
                // set.
                set_ok &= flash_q8_layer(&k, &kp, &gpu, &q, &kf, &vf, live, &g, label, l, l == 0)?;

                if !graph_done {
                    for (pass, mma) in [false, true].into_iter().enumerate() {
                        let mut pv =
                            DeviceBuffer::<f32>::zeroed(stream, partials_v_len(1, n_head, ctx))?;
                        let mut pms =
                            DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head, ctx))?;
                        let mut yg = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
                        let graph = gpu.capture(|s| {
                            k.enqueue_pass(
                                s,
                                GqaArgs {
                                    q: &inp.q,
                                    kc: &inp.kc,
                                    vc: &inp.vc,
                                    n_keys: &inp.n_keys,
                                    scale,
                                    n_kv,
                                    ctx,
                                    m: 1,
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
                        let same = bits_equal(&yg.to_host_vec(stream)?, &ys[pass]);
                        let nodes = graph.node_count();
                        let pass_ok = same && nodes == 2;
                        println!(
                            "graph op={}+gqa_flash_merge set={label} layer={l} segments={} \
                             eager_vs_graph_bit_identical={same} graph_nodes={nodes} {}",
                            if mma {
                                "gqa_flash_seg_mma"
                            } else {
                                "gqa_flash_seg"
                            },
                            segments_for(ctx),
                            verdict(pass_ok)
                        );
                        ok &= pass_ok;
                    }
                    graph_done = true;
                }
                n_live = live;
                ctx_seen = ctx;
                layers += 1;
            }
            if layers == 0 {
                return Err(format!("{label}: no fa-0 row").into());
            }
            for (pass, r) in rs.iter().enumerate() {
                println!(
                    "flash set={label} pass={} layers={layers} keys={n_live} ctx={ctx_seen} segments={} — \
                     measured / bound: ours-exact {:.3e}, ik-exact {:.3e}, ours-ik {:.3e}; ours-ik plain \
                     {:.3e} of max|ik| (printed)",
                    if pass == 1 { "mma" } else { "scalar" },
                    segments_for(ctx_seen),
                    r.ours,
                    r.ik,
                    r.pair,
                    r.plain
                );
            }
            println!(
                "flash set={label} pass=prefill layers={layers} keys={n_live} ctx={ctx_seen} tiles={} — \
                 measured / bound: ours-exact {:.3e}, ik-exact {:.3e}, ours-ik {:.3e}; ours-ik plain \
                 {:.3e} of max|ik| (printed)",
                n_live.div_ceil(PREF_TILE),
                rp.ours,
                rp.ik,
                rp.pair,
                rp.plain
            );
            println!(
                "flash set={label}: bounds, reruns, NaN padding {}",
                verdict(set_ok)
            );
            ok &= set_ok;
        }
        let seeded_ok = prefill_seeded(&kp, &gpu, scale)?;
        println!(
            "prefill flash seeded, fault, graph, refusal {}",
            verdict(seeded_ok)
        );
        ok &= seeded_ok;
        ok &= q8_window_refusal(&kp, &gpu, scale)?;
        println!("gate_qwen3moe_flash: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
