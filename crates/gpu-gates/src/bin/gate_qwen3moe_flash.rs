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
//!   segments and the butterfly, ik's every key serially. A segment holds
//!   [`seg_span`]'s keys: [`SEG_KEYS`] for every count up to [`SEGMENTS`]
//!   segments of them (5,120 keys), so at the sets' 5, 1,025 and 4,097 keys
//!   the cut, the segment count and the bound are the 64-key ones, value for
//!   value; past 5,120 the span grows and the segments stay at most
//!   [`SEGMENTS`].
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
//! NaN change no bit; the captured graph (two nodes) replays bit-identical
//! (the grid clause below, at the first set's own height).
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
//! The q8_0 read path (`enqueue_pass_q8`, the tensor-core segment pass, and
//! `enqueue_q8`) runs the same walks over the cache's Q8_0 form, host-built
//! over each layer's own f32 K and V values: `quantize_q8_0` a 32-value
//! block, the two planes `q8_0_planes` packs. Both stage the dequantized
//! values (`code·d`, exact in f32: a code's 7 significant bits against the
//! f16 scale's 11 leave 6 of f32's 24) rounded to f16 — the twin tiles' own
//! bits — so each is held bit for bit to its f16 twin run on the synthesized
//! f16 cache of the dequantized values (`f32_to_f16_bits` of each): the tiles
//! hold the same bits, and the walk after them is the twin's verbatim, so the
//! two launches must agree to the bit (a band would only re-derive the twin's
//! own gate). Bit identity against a host oracle is not claimed: the
//! exponential is the device's own (`dev_exp`), and the host cannot run it.
//!
//! Asserted for the q8 entries: on each set, every layer's decode pass and
//! prefill launch — a rerun bit-identical, every key head's cache rows at or
//! past the count holding the q8 sentinel bits (codes `0xa5a5a5a5`, scales
//! `0x5a5a`) changing no bit, each bit for bit its f16 twin on the
//! synthesized cache; on layer 0, the [`ROWS`] rotated rows of the prefill
//! shape each bit for bit their one-row launch, the `key_count` fault with
//! NaN in exactly the bad rows and every other row the clean run's, the
//! captured decode launch (two nodes) and prefill launch (one node) replaying
//! the eager bits, and a `kq` one word short refused by name. Last (after the
//! f16 window checks), `kq` at 8 bytes past a 16-byte boundary through the q8
//! prefill refused by name before any launch.
//!
//! The segment grid ([`SEGMENTS`] segments a row's key head) does not depend
//! on the cache height, and two clauses pin it. Grid at two heights: on the first set's layer 0, the
//! scalar pass, the tensor-core pass and the q8 tensor-core twin captured at
//! the set's own height and on a tall copy at [`GRID_TALL_CTX`] (each kv
//! plane's live rows its first rows, NaN in the rest, the same query, counts
//! and partial lengths) — two kernel nodes each, the segment node's grid
//! `n_kv · SEGMENTS` blocks and the merge's `n_head` at both heights, the
//! two outputs bit-identical to each other and to the eager launch. Deep
//! counts, the only clause that reaches the changed arithmetic past 5,120
//! keys: a synthetic f16 cache of [`DEEP_CTX`] rows over [`DEEP_KV`] key
//! heads and a synthetic query, one-row launches at counts 5,120 (the last
//! the 64-key cut covers), 5,121 (the first past it), 8,193, 65,537, 261,120 and
//! 262,144 — both passes within the band above read through [`seg_span`]
//! (the exact attention of two key heads), a rerun bit-identical, NaN in the
//! rows at or past the count changing no bit, the q8 tensor-core twin bit
//! for bit its f16 twin on the synthesized dequantized cache at 5,121 and
//! 65,537, and one launch of [`ROWS`] rows at counts straddling the
//! threshold each row bit for bit its one-row launch.
//!
//! Group 16 (`enqueue_pass` at [`GROUP_16`], two group-of-eight packs a key
//! head in one segment launch), at Nemotron 3.5 Bolt's 32 query heads over 2 key
//! heads of 128, on a seeded cache NaN past the count, both passes, at 6,
//! 512 and 4,096 keys: every head within the band above of the exact
//! attention over its key head `h / 16`; bit for bit the group-of-eight
//! launch over planes holding each key head twice (the eight-head entry's
//! own arithmetic, which the pack grid must not move); a rerun bit-identical.
//! At 512 keys the prefill shape and the `key_count` fault clause above at
//! group 16; at 4,096 the captured launch, one segment node of `2 · n_kv ·`
//! [`SEGMENTS`] blocks and one merge of `n_head` blocks, replaying the eager
//! bits.
//!
//! The split entry ([`FlashGqaPrefill::enqueue_split`], the host-side form
//! of the prefill flash for a group of any positive multiple of eight query
//! heads a key head) at Bolt's 32/2 of 128, on seeded inputs as the prefill
//! flash's own seeded clause (queries scaled by [`SEED_Q_SCALE`], counts
//! `p0 + t + 1`, cache [`SEED_PAD`] rows taller) at `T` ∈ [`SPLIT_T`] ×
//! `p0` ∈ [`SPLIT_P0`]: every row within its band of the exact attention
//! over its key head `h / 16`; bit for bit the plain group-of-eight launch
//! of the same queries over planes holding each key head twice, the twin
//! built as the group-16 clause's; a rerun bit-identical and NaN cache rows
//! past the largest count changing no bit; once, at `T` 65 and `p0` 63, the
//! captured launch replaying the eager bits in `3 · group/8` nodes, a launch
//! and two copies a sub-group; a group that is not a multiple of eight and a
//! scratch shorter than the launch refused by name, nothing enqueued.
//! The referee reads the full cache planes from key zero (`first = 0`);
//! `p0` is the query's prefix length, not a cut of those planes.
//!
//! The head-64 entries (`enqueue_pass_64`, Muse Spark's: [`GROUP`] query
//! heads a key head of 64 values, a layer's window 2048 or none) on a
//! synthetic f16 cache of 2 key heads × 4,096 rows and a seeded query, at
//! the module band with the 64-value head's own counts — 16 fused
//! multiply-adds a score partial (γ(19)), the tensor core's four k16 steps
//! (2·γ(66) beside the query's f16 rounding). Asserted, window off, at
//! counts 1, the tile edges 63/64/65, 2,047, 2,048, 4,095 and the cache's
//! full 4,096: every count of both passes within the band of the exact
//! attention of every query head, a rerun bit-identical, the rows at or
//! past the count NaN changing no bit. Windowed at 2048, at 1 (below the
//! window), 2,047/2,048/2,049 (its edge) and 3,072/4,096 (past it): within
//! the band of the exact attention over the window's own keys, and bit for
//! bit the windowless launch over the window's keys re-based at row 0 — the
//! same rows, the same walk, no window arithmetic of its own to hide
//! behind. The prefill shape ([`ROWS`] rotated rows at counts spread over
//! the cache) each bit for bit its one-row launch, at both windows and both
//! passes; the `key_count` fault rows of the decode clause (row 3 one past
//! the cache, row 5 zero) at both windows; the captured launch of both
//! passes at both windows two nodes — the segment node `n_kv ·` the window's
//! segments, the merge `n_head` — replaying the eager bits; a `part_v` one
//! short refused by name.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_flash", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::fault::{Fault, FaultSink, FaultSite, LAYER_NONE};
    use bloomery_gpu::flash_gqa::{
        FlashGqaKernels, GROUP, GROUP_5, GROUP_16, Gqa64Args, GqaArgs, GqaQ8Args, HEAD, HEAD_64,
        KEY_TILE, SEG_KEYS, SEGMENTS, partials_ms_len, partials_v_len, partials_v_len_64, seg_span,
        window_cut, window_segments,
    };
    use bloomery_gpu::flash_gqa_prefill::{
        FlashGqaPrefill, GqaPrefillArgs, GqaPrefillQ8Args, KEY_TILE as PREF_TILE, SplitScratch,
    };
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{Gpu, GpuError, Window};
    use bloomery_gpu_gates::flash_band::{ExactAttn, exact_attn, judge_prefill, prefill_bound};
    use bloomery_gpu_gates::prefill_gate::{HostCache, WindowCase, cache, window_refusals};
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

    /// One launch of `m` query rows (`q` and `n_keys` hold `m` each) at `G`
    /// query heads a key head over the cache planes, into fresh scratch,
    /// raising on `fault`, read back.
    fn run_once<const G: usize>(
        k: &FlashGqaKernels,
        (stream, fault): (&CudaStream, FaultSink),
        q: &DeviceBuffer<f32>,
        n_keys: &DeviceBuffer<u32>,
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        g: &Geom,
        mma: bool,
    ) -> Result<Vec<f32>, GateError> {
        let (n_head, m) = (g.n_kv * G, n_keys.len());
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(m, n_head))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, n_head))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * n_head * HEAD)?;
        k.enqueue_pass::<G>(
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

    /// The prefill shape (module doc): one launch of [`ROWS`] rows at `G`
    /// query heads a key head against each row's one-row launch. Returns
    /// whether every row matched.
    fn rows_check<const G: usize>(
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
        let n_head = g.n_kv * G;
        let rows = rotated_rows(q, n_head);
        let limits = spread_counts(live)?;
        let cache = (&inp.kc, &inp.vc);
        let all = run_once::<G>(
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
            let one = run_once::<G>(
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
            "rows group={G} pass={} m={ROWS} keys={limits:?} ctx={}: each row = its one-row launch bit for bit {}",
            if mma { "mma" } else { "scalar" },
            g.ctx,
            verdict(same)
        );
        Ok(same)
    }

    /// The decode flash's refusal (module doc): the prefill-shape launch with
    /// row 3's count past the cache and row 5's zero, labelled with layer 13,
    /// against the same launch clean, at `G` query heads a key head. Returns
    /// whether every check held.
    fn decode_fault<const G: usize>(
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
        let n_head = g.n_kv * G;
        let w = n_head * HEAD;
        let qd = DeviceBuffer::from_host(stream, &rotated_rows(q, n_head).concat())?;
        let clean = spread_counts(live)?;
        let (bad_hi, bad_zero) = (3usize, 5usize);
        let mut bad = clean.clone();
        bad[bad_hi] = u32::try_from(g.ctx + 1)?;
        bad[bad_zero] = 0;
        let cache = (&inp.kc, &inp.vc);
        let before = gpu.fault()?;
        let y = run_once::<G>(
            k,
            run,
            &qd,
            &DeviceBuffer::from_host(stream, &clean)?,
            cache,
            g,
            mma,
        )?;
        let after_clean = gpu.fault()?;
        let yb = run_once::<G>(
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
        let y2 = run_once::<G>(
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
            "decode fault group={G} pass={} m={ROWS} counts {} (past ctx {}) and 0 at rows {bad_hi}, {bad_zero}: word \
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

    /// The tall height of the grid clause (module doc): past every count the
    /// sets hold, under the 65,537-key count the deep clause walks.
    const GRID_TALL_CTX: usize = 65_536;

    /// `planes` — `n_kv` planes of `ctx` rows of `per_row` values — as
    /// planes of `tall` rows: each plane's first `live` rows its first rows,
    /// `fill` in the rest.
    fn tall_planes<T: Copy>(
        planes: &[T],
        n_kv: usize,
        per_row: usize,
        (ctx, live, tall): (usize, usize, usize),
        fill: T,
    ) -> Vec<T> {
        let mut out = vec![fill; n_kv * tall * per_row];
        for kh in 0..n_kv {
            let (from, to) = (kh * ctx * per_row, kh * tall * per_row);
            out[to..to + live * per_row].copy_from_slice(&planes[from..from + live * per_row]);
        }
        out
    }

    /// The grid clause (module doc): the decode launch captured at the set's
    /// own height and at [`GRID_TALL_CTX`] — the segment node's grid
    /// `n_kv · SEGMENTS` blocks and the merge's `n_head` at both heights, the
    /// outputs bit-identical to each other and to the eager launch (`ys`'s
    /// f16 bits, the q8 twin's own eager run). The scalar pass, the
    /// tensor-core pass and the q8 tensor-core twin; each tall copy holds the
    /// live rows in its first rows and NaN (the q8 sentinel bits) in the
    /// rest.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernels, the card, the set's inputs and host values, the eager bits, the geometry"
    )]
    fn grid_heights(
        k: &FlashGqaKernels,
        gpu: &Gpu,
        inp: &Inputs,
        (kf, vf): (&[f32], &[f32]),
        (kb, vb): (&[u16], &[u16]),
        ys: &[Vec<f32>],
        live: usize,
        g: &Geom,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let n_head = g.n_kv * GROUP;
        let rows = (g.ctx, live, GRID_TALL_CTX);
        let tall_k = DeviceBuffer::from_host(stream, &tall_planes(kb, g.n_kv, HEAD, rows, NAN16))?;
        let tall_v = DeviceBuffer::from_host(stream, &tall_planes(vb, g.n_kv, HEAD, rows, NAN16))?;
        // The tall q8 planes, the same rows of the set's Q8_0 form (the
        // codes and scales `Q8Cache::build` uploads) with the sentinel bits
        // in the rest.
        let c8 = Q8Cache::build(stream, kf, vf, g, live)?;
        let (ks, vs) = (Q8Side::build(kf, HEAD), Q8Side::build(vf, HEAD));
        let codes = |w: &[u32]| tall_planes(w, g.n_kv, HEAD / 4, rows, SENTINEL_Q8_CODE);
        let scales = |w: &[u16]| tall_planes(w, g.n_kv, HEAD / 32, rows, SENTINEL_Q8_SCALE);
        let (tall_kq, tall_kd, tall_vq, tall_vd) = (
            DeviceBuffer::from_host(stream, &codes(&ks.codes))?,
            DeviceBuffer::from_host(stream, &scales(&ks.scales))?,
            DeviceBuffer::from_host(stream, &codes(&vs.codes))?,
            DeviceBuffer::from_host(stream, &scales(&vs.scales))?,
        );
        let want_seg = [u32::try_from(g.n_kv * SEGMENTS)?, 1, 1];
        let want_merge = [u32::try_from(n_head)?, 1, 1];
        let mut ok = true;
        for (name, pass, q8) in [
            ("scalar", 0usize, false),
            ("mma", 1, false),
            ("q8-mma", 1, true),
        ] {
            let mut grids = Vec::with_capacity(2);
            let mut outs = Vec::with_capacity(2);
            let heights = [(g.ctx, &inp.kc, &inp.vc), (GRID_TALL_CTX, &tall_k, &tall_v)];
            for (ctx, kc, vc) in heights {
                let (mut pv, mut pms, mut y) = (
                    DeviceBuffer::<f32>::zeroed(stream, partials_v_len(1, n_head))?,
                    DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head))?,
                    DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?,
                );
                let graph = gpu.capture(|s| {
                    if q8 {
                        k.enqueue_pass_q8(
                            s,
                            GqaQ8Args {
                                q: &inp.q,
                                kq: if ctx == g.ctx { &c8.kq } else { &tall_kq },
                                kd: if ctx == g.ctx { &c8.kd } else { &tall_kd },
                                vq: if ctx == g.ctx { &c8.vq } else { &tall_vq },
                                vd: if ctx == g.ctx { &c8.vd } else { &tall_vd },
                                n_keys: &inp.n_keys,
                                scale: g.scale,
                                n_kv: g.n_kv,
                                ctx,
                                m: 1,
                                part_v: &mut pv,
                                part_ms: &mut pms,
                                fault: gpu.unlabelled_sink(),
                                y: &mut y,
                            },
                        )
                    } else {
                        k.enqueue_pass::<GROUP>(
                            s,
                            GqaArgs {
                                q: &inp.q,
                                kc,
                                vc,
                                n_keys: &inp.n_keys,
                                scale: g.scale,
                                n_kv: g.n_kv,
                                ctx,
                                m: 1,
                                part_v: &mut pv,
                                part_ms: &mut pms,
                                fault: gpu.unlabelled_sink(),
                                y: &mut y,
                            },
                            pass == 1,
                        )
                    }
                })?;
                graph.launch(stream)?;
                stream.synchronize()?;
                let nodes = graph.nodes()?;
                let grid = |prefix: &str| {
                    nodes.iter().find_map(|n| {
                        n.kernel
                            .as_ref()
                            .filter(|kn| kn.name.starts_with(prefix))
                            .map(|kn| kn.grid)
                    })
                };
                let two = nodes.len() == 2 && nodes.iter().all(|n| n.kernel.is_some());
                grids.push((two, grid("gqa_flash_seg"), grid("gqa_flash_merge")));
                outs.push(y.to_host_vec(stream)?);
            }
            let eager = if q8 {
                run_once_q8(
                    k,
                    (stream, gpu.unlabelled_sink()),
                    &inp.q,
                    &inp.n_keys,
                    &c8,
                    g,
                    false,
                )?
            } else {
                ys[pass].clone()
            };
            let same = bits_equal(&outs[0], &outs[1]) && bits_equal(&outs[0], &eager);
            let pass_ok = same
                && grids.iter().all(|&(two, seg, merge)| {
                    two && seg == Some(want_seg) && merge == Some(want_merge)
                });
            println!(
                "flash grid pass={name} ctx={},{} seg_grid={:?},{:?} (want {want_seg:?}) \
                 merge_grid={:?},{:?} (want {want_merge:?}) outputs_bit_identical={same} {}",
                g.ctx,
                GRID_TALL_CTX,
                grids[0].1,
                grids[1].1,
                grids[0].2,
                grids[1].2,
                verdict(pass_ok)
            );
            ok &= pass_ok;
        }
        Ok(ok)
    }

    /// The deep clause's cache height and key heads (module doc): the most a
    /// load serves, over the model's four.
    const DEEP_CTX: usize = 262_144;
    const DEEP_KV: usize = 4;

    /// The deep-count clause (module doc): the counts past the fixed cut's
    /// 5,120-key floor, where the segment boundaries move. One-row launches
    /// of both passes at 5,120 (the last 64-key cut), 5,121 (the first past
    /// it), 8,193, 65,537, 261,120 and 262,144 keys over a synthetic f16
    /// cache, each within the band of the exact attention of the query heads
    /// of two key heads; a rerun and the NaN rows past the count (at 5,121
    /// and 65,537) bit-identical; the q8 tensor-core twin bit for bit its
    /// f16 twin on the synthesized dequantized cache at 5,121 and 65,537; one
    /// launch of [`ROWS`] rows at counts straddling the threshold each row
    /// bit for bit its one-row launch.
    fn deep_counts(k: &FlashGqaKernels, gpu: &Gpu, scale: f32) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let (n_kv, ctx) = (DEEP_KV, DEEP_CTX);
        let n_head = n_kv * GROUP;
        let to16 = |v: Vec<f32>| -> Vec<u16> { v.iter().map(|&x| f32_to_f16_bits(x)).collect() };
        let kb = to16(activations(HEAD, n_kv * ctx, 730));
        let vb = to16(activations(HEAD, n_kv * ctx, 731));
        // The oracle's values are the planes' f16 bits, as the set clauses'
        // are (module doc: both sides read the same f16 bits).
        let (kf, vf): (Vec<f32>, Vec<f32>) = (
            kb.iter().map(|&h| half_to_f32(h)).collect(),
            vb.iter().map(|&h| half_to_f32(h)).collect(),
        );
        let (kc, vc) = (
            DeviceBuffer::from_host(stream, &kb)?,
            DeviceBuffer::from_host(stream, &vb)?,
        );
        let q: Vec<f32> = activations(HEAD, n_head, 732)
            .iter()
            .map(|v| v * SEED_Q_SCALE)
            .collect();
        // One one-row launch of `count` keys through `mma` over `ck`/`cv`.
        let launch = |q: &[f32],
                      count: usize,
                      mma: bool,
                      ck: &DeviceBuffer<u16>,
                      cv: &DeviceBuffer<u16>|
         -> Result<Vec<f32>, GateError> {
            let (qd, nk) = (
                DeviceBuffer::from_host(stream, q)?,
                DeviceBuffer::from_host(stream, &[u32::try_from(count)?])?,
            );
            let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(1, n_head))?;
            let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head))?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
            k.enqueue_pass::<GROUP>(
                stream,
                GqaArgs {
                    q: &qd,
                    kc: ck,
                    vc: cv,
                    n_keys: &nk,
                    scale,
                    n_kv,
                    ctx,
                    m: 1,
                    part_v: &mut pv,
                    part_ms: &mut pms,
                    fault: unl,
                    y: &mut y,
                },
                mma,
            )?;
            stream.synchronize()?;
            Ok(y.to_host_vec(stream)?)
        };
        // The NaN twins: every row at or past the count NaN.
        let nanify = |b: &[u16], past: usize| -> Vec<u16> {
            b.iter()
                .enumerate()
                .map(|(i, &h)| if (i / HEAD) % ctx >= past { NAN16 } else { h })
                .collect()
        };
        let counts = [5_120, 5_121, 8_193, 65_537, 261_120, 262_144];
        let mut ok = true;
        for &count in &counts {
            let span = seg_span(count, SEGMENTS, SEG_KEYS);
            let live_segs = count.div_ceil(span);
            // The exact attention of the query heads of the first two key
            // heads: the bound of both passes.
            let exacts: Vec<Exact> = (0..2 * GROUP)
                .map(|h| {
                    let plane = (h / GROUP) * ctx * HEAD;
                    exact(
                        &q[h * HEAD..(h + 1) * HEAD],
                        &kf[plane..plane + count * HEAD],
                        &vf[plane..plane + count * HEAD],
                        count,
                        scale,
                    )
                })
                .collect();
            let (nan_k, nan_v) = if count == 5_121 || count == 65_537 {
                (
                    &DeviceBuffer::from_host(stream, &nanify(&kb, count))?,
                    &DeviceBuffer::from_host(stream, &nanify(&vb, count))?,
                )
            } else {
                (&kc, &vc)
            };
            for mma in [false, true] {
                let name = if mma { "mma" } else { "scalar" };
                let y = launch(&q, count, mma, &kc, &vc)?;
                let y2 = launch(&q, count, mma, &kc, &vc)?;
                let rerun = bits_equal(&y, &y2);
                let mut worst = 0.0f64;
                let mut band = true;
                for (h, ex) in exacts.iter().enumerate() {
                    let bound = if mma { &ex.bound_mma } else { &ex.bound_ours };
                    for d in 0..HEAD {
                        let e = (f64::from(y[h * HEAD + d]) - ex.o[d]).abs() / bound[d];
                        worst = worst.max(e);
                        band &= e <= 1.0;
                    }
                }
                let nan_same = if count == 5_121 || count == 65_537 {
                    bits_equal(&y, &launch(&q, count, mma, nan_k, nan_v)?)
                } else {
                    true
                };
                let pass_ok = band && rerun && nan_same;
                println!(
                    "flash deep count={count} span={span} segments={live_segs} pass={name} \
                     measured/bound {worst:.3e} band={band} rerun={rerun} \
                     nan_past_count_same={nan_same} {}",
                    verdict(pass_ok)
                );
                ok &= pass_ok;
            }
        }
        // The q8 tensor-core twin and its f16 twin (the dequantized values'
        // own f16 cache) at the boundary counts.
        let (ks, vs) = (Q8Side::build(&kf, HEAD), Q8Side::build(&vf, HEAD));
        let (kq, kd, vq, vd) = (
            DeviceBuffer::from_host(stream, &ks.codes)?,
            DeviceBuffer::from_host(stream, &ks.scales)?,
            DeviceBuffer::from_host(stream, &vs.codes)?,
            DeviceBuffer::from_host(stream, &vs.scales)?,
        );
        let (k16, v16) = (
            DeviceBuffer::from_host(stream, &ks.f16)?,
            DeviceBuffer::from_host(stream, &vs.f16)?,
        );
        let qd = DeviceBuffer::from_host(stream, &q)?;
        for &count in &[5_121, 65_537] {
            let nk = DeviceBuffer::from_host(stream, &[u32::try_from(count)?])?;
            let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(1, n_head))?;
            let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head))?;
            let mut y8 = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
            k.enqueue_pass_q8(
                stream,
                GqaQ8Args {
                    q: &qd,
                    kq: &kq,
                    kd: &kd,
                    vq: &vq,
                    vd: &vd,
                    n_keys: &nk,
                    scale,
                    n_kv,
                    ctx,
                    m: 1,
                    part_v: &mut pv,
                    part_ms: &mut pms,
                    fault: unl,
                    y: &mut y8,
                },
            )?;
            stream.synchronize()?;
            let twin = launch(&q, count, true, &k16, &v16)?;
            let same = bits_equal(&y8.to_host_vec(stream)?, &twin);
            println!(
                "flash deep q8 twin count={count} = the f16 twin on the dequantized cache \
                 bit for bit {same} {}",
                verdict(same)
            );
            ok &= same;
        }
        // The straddling rows: one launch of ROWS rows at counts around the
        // 5,120-key floor and the tall counts, each row its one-row launch.
        let straddle = [
            5_119, 5_120, 5_121, 5_122, 65_537, 261_119, 261_120, 262_144,
        ];
        let rows = rotated_rows(&q, n_head);
        let rows_q = DeviceBuffer::from_host(stream, &rows.concat())?;
        let counts_d = DeviceBuffer::from_host(
            stream,
            &straddle
                .iter()
                .map(|&c| u32::try_from(c))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        for mma in [false, true] {
            let name = if mma { "mma" } else { "scalar" };
            let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(ROWS, n_head))?;
            let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(ROWS, n_head))?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, ROWS * n_head * HEAD)?;
            k.enqueue_pass::<GROUP>(
                stream,
                GqaArgs {
                    q: &rows_q,
                    kc: &kc,
                    vc: &vc,
                    n_keys: &counts_d,
                    scale,
                    n_kv,
                    ctx,
                    m: ROWS,
                    part_v: &mut pv,
                    part_ms: &mut pms,
                    fault: unl,
                    y: &mut y,
                },
                mma,
            )?;
            stream.synchronize()?;
            let all = y.to_host_vec(stream)?;
            let w = n_head * HEAD;
            let mut same = true;
            for (t, row) in rows.iter().enumerate() {
                let one = launch(row, straddle[t], mma, &kc, &vc)?;
                same &= bits_equal(&all[t * w..(t + 1) * w], &one);
            }
            println!(
                "flash deep rows m={ROWS} counts={straddle:?} pass={name} \
                 each_row_its_one_row_launch={same} {}",
                verdict(same)
            );
            ok &= same;
        }
        Ok(ok)
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
        let all = run_pref(k, gpu, &qd, &counts, cache, g, None)?;
        let hc = HostCache {
            kf: kf.to_vec(),
            vf: vf.to_vec(),
            ctx: g.ctx,
        };
        let cu: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
        let (band, worst) = judge_prefill::<HEAD, PREF_TILE>(
            &rows,
            &cu,
            &hc,
            &all,
            g.scale,
            (g.n_kv * GROUP, g.n_kv),
            0,
        );
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
        let ex = exact_attn(q, kh, vh, HEAD, n, scale);
        // The prefill flash (module doc): one rescale per 64-key tile, the
        // f16 weights normalized by their own sum, the tensor core's value
        // accumulation and the lanes' weight sums.
        let bound_pref = prefill_bound(&ex, vh, HEAD, PREF_TILE, 0);
        let ExactAttn {
            o,
            a,
            a16,
            pb,
            x,
            xmax,
            ..
        } = ex;
        // The cut read through `seg_span` (module doc): at or below 5,120
        // keys span = SEG_KEYS and segs = `n.div_ceil(SEG_KEYS)`.
        let span = seg_span(n, SEGMENTS, SEG_KEYS);
        let segs = n.div_ceil(span);
        let r_ours = (n.div_ceil(KEY_TILE) + segs) as f64;
        let r_ik = (n.div_ceil(32) + 33) as f64;
        let (es_o, es_i) = (gamma(35), gamma(129));
        // The tensor core's f32 accumulation order is not documented and may
        // truncate: twice γ(130) for the dot of exact f16 products.
        let es_m = 2f64.powi(-11) + 2.0 * gamma(130);
        let (acc_o, acc_i) = (gamma(span + segs + 8), gamma(n + 40));
        let mut bo = vec![0.0f64; HEAD];
        let mut bm = vec![0.0f64; HEAD];
        let mut bi = vec![0.0f64; HEAD];
        for d in 0..HEAD {
            let (mut t_o, mut t_m, mut t_i, mut pv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for j in 0..n {
                let v = f64::from(vh[j * HEAD + d]);
                let dev = (v - o[d]).abs();
                let ee = 4.0 * U + 2.0 * U * x[j];
                t_o += pb[j] * (es_o * a[j] + ee + 4.0 * U * r_ours + 2.0 * U * xmax) * dev;
                t_m += pb[j] * (es_m * a16[j] + ee + 4.0 * U * r_ours + 2.0 * U * xmax) * dev;
                t_i += pb[j] * (es_i * a[j] + ee + 4.0 * U * r_ik + 2.0 * U * xmax) * dev;
                pv += pb[j] * v.abs();
            }
            bo[d] = t_o + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bm[d] = t_m + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bi[d] = t_i + acc_i * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
        }
        Exact {
            o,
            bound_ours: bo,
            bound_mma: bm,
            bound_ik: bi,
            bound_pref,
        }
    }

    /// One prefill launch of `n_keys.len()` rows (`q` holds that many) over
    /// the cache planes, into fresh output, read back: the plain
    /// group-of-eight entry, or with `split` the split entry at
    /// [`SPLIT_GROUP`] query heads a key head through `scratch`. The fault
    /// word is the caller's to read.
    fn run_pref(
        k: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &DeviceBuffer<f32>,
        n_keys: &[u32],
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        g: &Geom,
        split: Option<&mut SplitScratch>,
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let group = if split.is_some() { SPLIT_GROUP } else { GROUP };
        let (n_head, t) = (g.n_kv * group, n_keys.len());
        let nk = DeviceBuffer::from_host(stream, n_keys)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, t * n_head * HEAD)?;
        let args = GqaPrefillArgs {
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
        };
        match split {
            Some(scratch) => k.enqueue_split(stream, args, scratch)?,
            None => k.enqueue(stream, args)?,
        }
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    /// A cache's planes with each key head's plane twice: twin key head `j`
    /// of the doubled cache is `j / 2`.
    fn twice(b: &[u16], ctx: usize) -> Vec<u16> {
        b.chunks(ctx * HEAD)
            .flat_map(|p| p.iter().chain(p).copied())
            .collect()
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
            let one = run_pref(k, gpu, &qt, &[c], cache, g, None)?;
            differ += usize::from(!bits_equal(&all[t * w..(t + 1) * w], &one));
        }
        Ok(differ)
    }

    // ------------------------------------------------ the q8_0 read path

    /// One side's Q8_0 planes of a cache — `quantize_q8_0` over its f32
    /// values a row, `q8_0_planes`' packing — with the f16 rounding of the
    /// dequantized values `code·d` (exact in f32), the tiles' own format.
    /// `sentinel`'s padded twin writes the sentinel bits into every row at or
    /// past `live`, which no live key reads.
    struct Q8Side {
        codes: Vec<u32>,
        scales: Vec<u16>,
        f16: Vec<u16>,
    }

    impl Q8Side {
        fn build(vals: &[f32], head: usize) -> Q8Side {
            let n = vals.len() / head;
            let mut codes = Vec::with_capacity(n * head / 4);
            let mut scales = Vec::with_capacity(n * head / 32);
            let mut f16 = Vec::with_capacity(vals.len());
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
                        f16.push(f32_to_f16_bits(f32::from(c) * d));
                    }
                }
            }
            Q8Side { codes, scales, f16 }
        }

        /// The planes with every row at or past `live` the sentinel bits, in
        /// every key head: the planes are `[n_kv][ctx][head]`, so a key
        /// head's dead rows are `live..ctx` of its own plane.
        fn sentinel(&self, head: usize, ctx: usize, live: usize) -> (Vec<u32>, Vec<u16>) {
            let (mut q, mut d) = (self.codes.clone(), self.scales.clone());
            let n_kv = q.len() / (ctx * head / 4);
            for kh in 0..n_kv {
                for row in kh * ctx + live..(kh + 1) * ctx {
                    let (qw, sw) = (row * head / 4, row * head / 32);
                    q[qw..qw + head / 4].fill(SENTINEL_Q8_CODE);
                    d[sw..sw + head / 32].fill(SENTINEL_Q8_SCALE);
                }
            }
            (q, d)
        }
    }

    /// The q8 cache of one layer: the four planes and their sentinel twins
    /// on the card, and the synthesized f16 cache of the dequantized values
    /// (the bit anchors' twin launches).
    struct Q8Cache {
        kq: DeviceBuffer<u32>,
        kd: DeviceBuffer<u16>,
        vq: DeviceBuffer<u32>,
        vd: DeviceBuffer<u16>,
        knq: DeviceBuffer<u32>,
        knd: DeviceBuffer<u16>,
        vnq: DeviceBuffer<u32>,
        vnd: DeviceBuffer<u16>,
        k16: DeviceBuffer<u16>,
        v16: DeviceBuffer<u16>,
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

    /// One launch of `m` query rows of the q8 decode pass over the four
    /// planes (`use_sentinel`: their sentinel twins), into fresh scratch,
    /// read back.
    fn run_once_q8(
        k: &FlashGqaKernels,
        (stream, fault): (&CudaStream, FaultSink),
        q: &DeviceBuffer<f32>,
        n_keys: &DeviceBuffer<u32>,
        c: &Q8Cache,
        g: &Geom,
        use_sentinel: bool,
    ) -> Result<Vec<f32>, GateError> {
        let (n_head, m) = (g.n_kv * GROUP, n_keys.len());
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(m, n_head))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, n_head))?;
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

    /// The q8_0 read path's clauses on one layer of a set (module doc): the
    /// decode pass and the prefill launch over the cache's Q8_0 form — a
    /// rerun, the sentinel padding and bit identity with their f16 twins on
    /// the synthesized cache; `layer0` adds the prefill-shape rows, the fault,
    /// the graphs and the refusal.
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

        // The decode pass: bit for bit the twin on the synthesized f16 cache,
        // a rerun and the sentinel padding the same bits.
        let y = run_once_q8(k, run, &qd, &nk, &c, g, false)?;
        let y2 = run_once_q8(k, run, &qd, &nk, &c, g, false)?;
        let ys = run_once_q8(k, run, &qd, &nk, &c, g, true)?;
        let tw = run_once::<GROUP>(k, run, &qd, &nk, (&c.k16, &c.v16), g, true)?;
        let (rerun, sentinel) = (bits_equal(&y, &y2), bits_equal(&y, &ys));
        let twin_same = bits_equal(&y, &tw);
        let dec_ok = rerun && sentinel && twin_same;
        println!(
            "flash q8 set={label} layer={l} pass=mma keys={live} ctx={}: = the f16 mma twin on the \
             synthesized f16 cache bit for bit {twin_same} rerun={rerun} sentinel_same={sentinel} {}",
            g.ctx,
            verdict(dec_ok)
        );
        ok &= dec_ok;

        // The prefill launch: bit for bit the twin on the synthesized cache,
        // a rerun and the sentinel padding the same bits.
        let count = [u32::try_from(live)?];
        let yp = run_pref_q8(kp, gpu, &qd, &count, &c, g, false)?;
        let yp2 = run_pref_q8(kp, gpu, &qd, &count, &c, g, false)?;
        let yps = run_pref_q8(kp, gpu, &qd, &count, &c, g, true)?;
        let tw = run_pref(kp, gpu, &qd, &count, (&c.k16, &c.v16), g, None)?;
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
            // for bit its one-row launch.
            let rows = rotated_rows(q, n_head);
            let limits = spread_counts(live)?;
            let rd = DeviceBuffer::from_host(stream, &rows.concat())?;
            let ld = DeviceBuffer::from_host(stream, &limits)?;
            let all = run_once_q8(k, run, &rd, &ld, &c, g, false)?;
            let mut same = true;
            for (t, row) in rows.iter().enumerate() {
                let one = run_once_q8(
                    k,
                    run,
                    &DeviceBuffer::from_host(stream, row)?,
                    &DeviceBuffer::from_host(stream, &limits[t..=t])?,
                    &c,
                    g,
                    false,
                )?;
                same &= bits_equal(&all[t * n_head * HEAD..(t + 1) * n_head * HEAD], &one);
            }
            println!(
                "flash q8 rows pass=mma m={ROWS} keys={limits:?} ctx={}: each row = its one-row \
                 launch bit for bit {same} {}",
                g.ctx,
                verdict(same)
            );
            ok &= same;

            // The refusal path: rows 3 and 5 of the eight-row launch.
            let (bad_hi, bad_zero) = (3usize, 5usize);
            let mut bad = limits;
            bad[bad_hi] = u32::try_from(g.ctx + 1)?;
            bad[bad_zero] = 0;
            let bd = DeviceBuffer::from_host(stream, &bad)?;
            let before = gpu.fault()?;
            let yb = run_once_q8(k, (stream, gpu.layer_sink(13)?), &rd, &bd, &c, g, false)?;
            let raised = gpu.take_fault()?;
            let want = Some(Fault::at(13, FaultSite::KeyCount));
            let w = n_head * HEAD;
            let (mut others, mut nan) = (true, true);
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
                "flash q8 fault pass=mma: counts {} (past ctx {}) and 0 at rows {bad_hi}, \
                 {bad_zero}: word {raised:?} (want {want:?}), those rows NaN {nan}, other rows \
                 bit-identical {others} {}",
                g.ctx + 1,
                g.ctx,
                verdict(fault_ok)
            );
            ok &= fault_ok;

            // The captured decode launch (two nodes) and prefill launch (one
            // node), replaying the eager bits.
            let (mut pv, mut pms, mut yg) = (
                DeviceBuffer::<f32>::zeroed(stream, partials_v_len(ROWS, n_head))?,
                DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(ROWS, n_head))?,
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
                )
            })?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let same = bits_equal(&yg.to_host_vec(stream)?, &all);
            let nodes = graph.node_count();
            let graph_ok = same && nodes == 2;
            println!(
                "flash q8 graph pass=mma m={ROWS}: eager_vs_graph_bit_identical={same} \
                 graph_nodes={nodes} {}",
                verdict(graph_ok)
            );
            ok &= graph_ok;
            let mut yg = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
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
            let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(1, n_head))?;
            let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head))?;
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
        let k_mis = Window::<u32>::of(&pad, 8, c.kq.len())?;
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
                let c = cache(stream, HEAD, n_kv, ctx, p0 + t, seed + 1)?;
                let qd = DeviceBuffer::from_host(stream, &q)?;
                let counts: Vec<u32> = (0..t)
                    .map(|i| u32::try_from(p0 + i + 1))
                    .collect::<Result<_, _>>()?;
                let y = run_pref(k, gpu, &qd, &counts, (&c.kc, &c.vc), &g, None)?;
                let y2 = run_pref(k, gpu, &qd, &counts, (&c.kc, &c.vc), &g, None)?;
                let y_nan = run_pref(k, gpu, &qd, &counts, (&c.kn, &c.vn), &g, None)?;
                let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &y_nan));
                let cu: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
                let (band, worst) = judge_prefill::<HEAD, PREF_TILE>(
                    &q,
                    &cu,
                    &c.host,
                    &y,
                    g.scale,
                    (g.n_kv * GROUP, g.n_kv),
                    0,
                );
                let differ = rows_alone(k, gpu, &q, &counts, (&c.kc, &c.vc), &g, &y)?;
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
        let y = run_pref(k, gpu, &qd, &clean, (&kc, &vc), &g, None)?;
        let after_clean = gpu.fault()?;
        let yb = run_pref(k, gpu, &qd, &bad, (&kc, &vc), &g, None)?;
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

    fn misaligned_refusals(
        k: &FlashGqaPrefill,
        gpu: &Gpu,
        q: &[f32],
        (kb, vb): (&[u16], &[u16]),
        (clean, y): (&[u32], &[f32]),
        g: &Geom,
    ) -> Result<bool, GateError> {
        window_refusals(
            gpu,
            "flash_gqa_prefill::enqueue",
            "prefill",
            WindowCase {
                q,
                planes: (kb, vb),
                counts: clean,
                y,
                heads: (g.n_kv * GROUP, g.n_kv),
                ctx: g.ctx,
                scale: g.scale,
            },
            |s, args| k.enqueue(s, args),
        )
    }

    /// The split clause's key heads and group (Nemotron 3.5 Bolt's 32/2),
    /// and its seeded row counts and first positions.
    const SPLIT_KV: usize = 2;
    const SPLIT_GROUP: usize = 16;
    const SPLIT_T: [usize; 4] = [1, 17, 65, 448];
    const SPLIT_P0: [usize; 3] = [0, 63, 1000];

    /// The split entry (module doc), at Bolt's shape on seeded inputs.
    /// Returns whether every check passed.
    fn prefill_split(k: &FlashGqaPrefill, gpu: &Gpu, scale: f32) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (n_kv, group) = (SPLIT_KV, SPLIT_GROUP);
        let n_head = n_kv * group;
        // The launch's largest row count of the clause.
        let mut scratch = SplitScratch::new(stream, SPLIT_T[3], n_kv)?;
        let mut ok = true;
        let mut seed = 401u32;
        for &p0 in &SPLIT_P0 {
            for &t in &SPLIT_T {
                seed += 3;
                let ctx = p0 + t + SEED_PAD;
                let g = Geom { scale, n_kv, ctx };
                let q: Vec<f32> = activations(HEAD, t * n_head, seed)
                    .into_iter()
                    .map(|v| v * SEED_Q_SCALE)
                    .collect();
                let c = cache(stream, HEAD, n_kv, ctx, p0 + t, seed + 1)?;
                let qd = DeviceBuffer::from_host(stream, &q)?;
                let counts: Vec<u32> = (0..t)
                    .map(|i| u32::try_from(p0 + i + 1))
                    .collect::<Result<_, _>>()?;
                let y = run_pref(k, gpu, &qd, &counts, (&c.kc, &c.vc), &g, Some(&mut scratch))?;
                let y2 = run_pref(k, gpu, &qd, &counts, (&c.kc, &c.vc), &g, Some(&mut scratch))?;
                let y_nan = run_pref(k, gpu, &qd, &counts, (&c.kn, &c.vn), &g, Some(&mut scratch))?;
                let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &y_nan));
                let cu: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
                let (band, worst) = judge_prefill::<HEAD, PREF_TILE>(
                    &q,
                    &cu,
                    &c.host,
                    &y,
                    g.scale,
                    (n_head, n_kv),
                    0,
                );
                // Each key head's plane twice: twin key head `j` is `j / 2`,
                // so head `h` of the twin's group-of-eight launch reads the
                // real key head `h / 16`.
                let (tk, tv) = (
                    DeviceBuffer::from_host(stream, &twice(&c.kb, ctx))?,
                    DeviceBuffer::from_host(stream, &twice(&c.vb, ctx))?,
                );
                let g_twin = Geom {
                    scale,
                    n_kv: n_head / GROUP,
                    ctx,
                };
                let twin = bits_equal(
                    &y,
                    &run_pref(k, gpu, &qd, &counts, (&tk, &tv), &g_twin, None)?,
                );
                let pass = band && twin && rerun && nan_same;
                println!(
                    "prefill split T={t} p0={p0} ctx={ctx}: measured/bound {worst:.3e} band={band} \
                     group8_twin_bits={twin} rerun={rerun} nan_padding_same={nan_same} {}",
                    verdict(pass)
                );
                ok &= pass;
                if (t, p0) == (65, 63) {
                    let nk = DeviceBuffer::from_host(stream, &counts)?;
                    let mut yg = DeviceBuffer::<f32>::zeroed(stream, t * n_head * HEAD)?;
                    let graph = gpu.capture(|s| {
                        k.enqueue_split(
                            s,
                            GqaPrefillArgs {
                                q: &qd,
                                kc: &c.kc,
                                vc: &c.vc,
                                n_keys: &nk,
                                scale,
                                n_head,
                                n_kv,
                                ctx,
                                t,
                                fault: gpu.unlabelled_sink(),
                                y: &mut yg,
                            },
                            &mut scratch,
                        )
                    })?;
                    graph.launch(stream)?;
                    stream.synchronize()?;
                    let same = bits_equal(&yg.to_host_vec(stream)?, &y);
                    let nodes = graph.node_count();
                    // A sub-group is a launch, a query copy in and an output
                    // copy out.
                    let graph_ok = same && nodes == 3 * (group / GROUP);
                    println!(
                        "prefill split graph T={t} p0={p0}: eager_vs_graph_bit_identical={same} \
                         graph_nodes={nodes} {}",
                        verdict(graph_ok)
                    );
                    ok &= graph_ok;
                }
            }
        }

        // The refusals (module doc): a group that is not a multiple of
        // eight, and a scratch one row short of the launch.
        let (t, p0) = (17usize, 63usize);
        let ctx = p0 + t + SEED_PAD;
        let q = activations(HEAD, t * n_head, 451);
        let kb: Vec<u16> = activations(HEAD, n_kv * ctx, 452)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        let vb: Vec<u16> = activations(HEAD, n_kv * ctx, 453)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        let (kc, vc) = (
            DeviceBuffer::from_host(stream, &kb)?,
            DeviceBuffer::from_host(stream, &vb)?,
        );
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let counts: Vec<u32> = (0..t)
            .map(|i| u32::try_from(p0 + i + 1))
            .collect::<Result<_, _>>()?;
        let nk = DeviceBuffer::from_host(stream, &counts)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, t * n_head * HEAD)?;
        let mut short = SplitScratch::new(stream, t - 1, n_kv)?;
        for (name, heads, needle, scr) in [
            (
                "a group of 12 over 2 key heads",
                n_kv * 12,
                "multiple of",
                &mut scratch,
            ),
            ("a scratch one row short", n_head, "scratch", &mut short),
        ] {
            let r = k.enqueue_split(
                stream,
                GqaPrefillArgs {
                    q: &qd,
                    kc: &kc,
                    vc: &vc,
                    n_keys: &nk,
                    scale,
                    n_head: heads,
                    n_kv,
                    ctx,
                    t,
                    fault: gpu.unlabelled_sink(),
                    y: &mut y,
                },
                scr,
            );
            let named = matches!(
                &r,
                Err(GpuError::Shape {
                    what: "flash_gqa_prefill::enqueue_split",
                    detail,
                }) if detail.contains(needle)
            );
            println!(
                "prefill split refusal {name}: {} {}",
                r.err().map_or("accepted".to_string(), |e| e.to_string()),
                verdict(named)
            );
            ok &= named;
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

    /// Head 128, [`GROUP_5`] query heads a key head, the scalar pass,
    /// against [`exact`]'s head-128 band. Two key heads: a KV index of
    /// `h / 8` still reads the first plane for the second group. 100 keys
    /// leaves four live keys in the second segment's last tile, so the
    /// live-key mask is read.
    fn group5(k: &FlashGqaKernels, gpu: &Gpu, scale: f32) -> Result<bool, GateError> {
        const N_KV: usize = 2;
        const KEYS: usize = 100;
        const CTX: usize = 128;
        let n_head = N_KV * GROUP_5;
        let stream = gpu.stream();
        let q: Vec<f32> = activations(HEAD, n_head, 5);
        let kb: Vec<u16> = activations(HEAD, N_KV * CTX, 6)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        let vb: Vec<u16> = activations(HEAD, N_KV * CTX, 7)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        let kf: Vec<f32> = kb.iter().copied().map(half_to_f32).collect();
        let vf: Vec<f32> = vb.iter().copied().map(half_to_f32).collect();
        let q_d = DeviceBuffer::from_host(stream, &q)?;
        let kc = DeviceBuffer::from_host(stream, &kb)?;
        let vc = DeviceBuffer::from_host(stream, &vb)?;
        let n_keys = DeviceBuffer::from_host(stream, &[KEYS as u32])?;
        let mut part_v = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(1, n_head))?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
        k.enqueue_pass::<GROUP_5>(
            stream,
            GqaArgs {
                q: &q_d,
                kc: &kc,
                vc: &vc,
                n_keys: &n_keys,
                scale,
                n_kv: N_KV,
                ctx: CTX,
                m: 1,
                part_v: &mut part_v,
                part_ms: &mut part_ms,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
            false,
        )?;
        stream.synchronize()?;
        let got = y.to_host_vec(stream)?;
        let mut worst = 0.0f64;
        let mut ok = true;
        for h in 0..n_head {
            let plane = (h / GROUP_5) * CTX * HEAD;
            let ex = exact(
                &q[h * HEAD..(h + 1) * HEAD],
                &kf[plane..plane + KEYS * HEAD],
                &vf[plane..plane + KEYS * HEAD],
                KEYS,
                scale,
            );
            for d in 0..HEAD {
                let err = (f64::from(got[h * HEAD + d]) - ex.o[d]).abs();
                let bound = ex.bound_ours[d];
                if bound > 0.0 {
                    worst = worst.max(err / bound);
                }
                ok &= err <= bound;
            }
        }
        println!(
            "flash group5 head={HEAD} group={GROUP_5} kv={N_KV} keys={KEYS}: measured/bound {worst:.3e} {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// The group-16 clause's key heads (Nemotron 3.5 Bolt's 32/2), its counts,
    /// and its cache height: the largest count and a tile of NaN rows past it.
    const G16_KV: usize = 2;
    const G16_KEYS: [usize; 3] = [6, 512, 4_096];
    const G16_CTX: usize = 4_096 + KEY_TILE;

    /// Head 128, [`GROUP_16`] query heads a key head (module doc of
    /// `flash_gqa`: two group-of-eight packs a key head in one launch), both passes, on a
    /// seeded query and cache whose rows at or past the count are NaN. At
    /// each count of [`G16_KEYS`]: every head within its band of the exact
    /// attention over key head `h / 16`; bit for bit the group-of-eight
    /// launch over planes holding each key head twice (`2 · n_kv` key heads,
    /// so head `h` reads key head `h / 8` of them, the real `h / 16`); a
    /// rerun bit-identical. At 512 keys the prefill shape and the fault
    /// clause at group 16; at 4,096 the captured launch: one segment node of
    /// `2 · n_kv ·` [`SEGMENTS`] blocks and one merge of `n_head`, replaying
    /// the eager bits.
    fn group16(k: &FlashGqaKernels, gpu: &Gpu, scale: f32) -> Result<bool, GateError> {
        let (n_kv, ctx) = (G16_KV, G16_CTX);
        let n_head = n_kv * GROUP_16;
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let to16 = |v: Vec<f32>| -> Vec<u16> { v.into_iter().map(f32_to_f16_bits).collect() };
        let q: Vec<f32> = activations(HEAD, n_head, 1601)
            .iter()
            .map(|v| v * SEED_Q_SCALE)
            .collect();
        let kb = to16(activations(HEAD, n_kv * ctx, 1602));
        let vb = to16(activations(HEAD, n_kv * ctx, 1603));
        let (kf, vf): (Vec<f32>, Vec<f32>) = (
            kb.iter().map(|&h| half_to_f32(h)).collect(),
            vb.iter().map(|&h| half_to_f32(h)).collect(),
        );
        let past = |b: &[u16], live: usize| -> Vec<u16> {
            b.iter()
                .enumerate()
                .map(|(i, &h)| if (i / HEAD) % ctx >= live { NAN16 } else { h })
                .collect()
        };
        let g = Geom { scale, n_kv, ctx };
        let g_twin = Geom {
            scale,
            n_kv: n_kv * GROUP_16 / GROUP,
            ctx,
        };
        let mut ok = true;
        for keys in G16_KEYS {
            let (kp, vp) = (past(&kb, keys), past(&vb, keys));
            let inp = Inputs {
                q: DeviceBuffer::from_host(stream, &q)?,
                kc: DeviceBuffer::from_host(stream, &kp)?,
                vc: DeviceBuffer::from_host(stream, &vp)?,
                n_keys: DeviceBuffer::from_host(stream, &[u32::try_from(keys)?])?,
            };
            let (tk, tv) = (
                DeviceBuffer::from_host(stream, &twice(&kp, ctx))?,
                DeviceBuffer::from_host(stream, &twice(&vp, ctx))?,
            );
            let exacts: Vec<Exact> = (0..n_head)
                .map(|h| {
                    let plane = (h / GROUP_16) * ctx * HEAD;
                    exact(
                        &q[h * HEAD..(h + 1) * HEAD],
                        &kf[plane..plane + keys * HEAD],
                        &vf[plane..plane + keys * HEAD],
                        keys,
                        scale,
                    )
                })
                .collect();
            for mma in [false, true] {
                let name = if mma { "mma" } else { "scalar" };
                let run = (stream, unl);
                let cache = (&inp.kc, &inp.vc);
                let y = run_once::<GROUP_16>(k, run, &inp.q, &inp.n_keys, cache, &g, mma)?;
                let rerun = bits_equal(
                    &y,
                    &run_once::<GROUP_16>(k, run, &inp.q, &inp.n_keys, cache, &g, mma)?,
                );
                let yt = run_once::<GROUP>(k, run, &inp.q, &inp.n_keys, (&tk, &tv), &g_twin, mma)?;
                let twin = bits_equal(&y, &yt);
                let mut worst = 0.0f64;
                let mut band = true;
                for (h, ex) in exacts.iter().enumerate() {
                    let bound = if mma { &ex.bound_mma } else { &ex.bound_ours };
                    for d in 0..HEAD {
                        let e = (f64::from(y[h * HEAD + d]) - ex.o[d]).abs() / bound[d];
                        worst = worst.max(e);
                        band &= e <= 1.0;
                    }
                }
                let pass = band && twin && rerun;
                println!(
                    "flash group16 head={HEAD} heads={n_head}/{n_kv} keys={keys} ctx={ctx} pass={name}: \
                     measured/bound {worst:.3e} band={band} group8_twin_bits={twin} rerun={rerun} {}",
                    verdict(pass)
                );
                ok &= pass;
                if keys == 512 {
                    ok &= rows_check::<GROUP_16>(k, gpu, &q, &inp, keys, &g, mma)?;
                    ok &= decode_fault::<GROUP_16>(k, gpu, &q, &inp, keys, &g, mma)?;
                }
                if keys == 4_096 {
                    ok &= group16_graph(k, gpu, &inp, &g, mma, &y)?;
                }
            }
        }
        Ok(ok)
    }

    /// [`group16`]'s captured launch: one row, its nodes and their grids, and
    /// the replay's bits against the eager `want`.
    fn group16_graph(
        k: &FlashGqaKernels,
        gpu: &Gpu,
        inp: &Inputs,
        g: &Geom,
        mma: bool,
        want: &[f32],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let n_head = g.n_kv * GROUP_16;
        let (mut pv, mut pms, mut y) = (
            DeviceBuffer::<f32>::zeroed(stream, partials_v_len(1, n_head))?,
            DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head))?,
            DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?,
        );
        let graph = gpu.capture(|s| {
            k.enqueue_pass::<GROUP_16>(
                s,
                GqaArgs {
                    q: &inp.q,
                    kc: &inp.kc,
                    vc: &inp.vc,
                    n_keys: &inp.n_keys,
                    scale: g.scale,
                    n_kv: g.n_kv,
                    ctx: g.ctx,
                    m: 1,
                    part_v: &mut pv,
                    part_ms: &mut pms,
                    fault: gpu.unlabelled_sink(),
                    y: &mut y,
                },
                mma,
            )
        })?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let nodes = graph.nodes()?;
        let grids = |prefix: &str| -> Vec<[u32; 3]> {
            nodes
                .iter()
                .filter_map(|n| n.kernel.as_ref())
                .filter(|kn| kn.name.starts_with(prefix))
                .map(|kn| kn.grid)
                .collect()
        };
        let (seg, merge) = (grids("gqa_flash_seg"), grids("gqa_flash_merge"));
        // PIN(2026-10-18): one segment node of every pack's blocks (was `2 * n_kv` nodes of `SEGMENTS`).
        let want_seg = vec![[u32::try_from(g.n_kv * GROUP_16 / GROUP * SEGMENTS)?, 1, 1]];
        let want_merge = vec![[u32::try_from(n_head)?, 1, 1]];
        let shape = nodes.len() == want_seg.len() + 1
            && nodes.iter().all(|n| n.kernel.is_some())
            && seg == want_seg
            && merge == want_merge;
        let same = bits_equal(&y.to_host_vec(stream)?, want);
        let pass = shape && same;
        println!(
            "flash group16 graph pass={} nodes={} seg_grids={seg:?} (want {want_seg:?}) \
             merge_grids={merge:?} (want {want_merge:?}) replay_bits_eager={same} {}",
            if mma { "mma" } else { "scalar" },
            nodes.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The head-64 clauses' cache (module doc): Muse Spark's geometry —
    /// [`GROUP`] query heads a key head of 64 values over [`H64_CTX`] rows,
    /// a layer's window [`H64_WINDOW`] or none.
    const H64_KV: usize = 2;
    const H64_CTX: usize = 4_096;
    const H64_WINDOW: usize = 2_048;
    /// One-row launch counts at no window: the tile edges 63/64/65, the
    /// window's own boundary and the cache's full height.
    const H64_COUNTS: [usize; 8] = [1, 63, 64, 65, 2_047, 2_048, 4_095, 4_096];
    /// The windowed clause's counts: below the window, at its edge, past it
    /// and the cache's full height.
    const H64_WIN_COUNTS: [usize; 6] = [1, 2_047, 2_048, 2_049, 3_072, 4_096];

    /// The exact attention of one head-64 query head in f64 and each pass's
    /// bound: the module band with the 64-value head's own counts (module
    /// doc).
    struct Exact64 {
        o: Vec<f64>,
        bound_ours: Vec<f64>,
        bound_mma: Vec<f64>,
    }

    /// [`Exact64`] of query head `q` (64 values) over `n` of the key and
    /// value rows at `scale`, a launch of `segs` segments a row's head.
    fn exact64(q: &[f32], kh: &[f32], vh: &[f32], n: usize, scale: f32, segs: usize) -> Exact64 {
        let ex = exact_attn(q, kh, vh, HEAD_64, n, scale);
        let ExactAttn {
            o,
            a,
            a16,
            pb,
            x,
            xmax,
            ..
        } = ex;
        // The cut read through `seg_span`, as the 128-value head's.
        let span = seg_span(n, segs, SEG_KEYS);
        let live = n.div_ceil(span);
        let r = (n.div_ceil(KEY_TILE) + live) as f64;
        // 16 fused multiply-adds a score partial, two combine levels and the
        // scale; the tensor core's four k16 steps, its order undocumented.
        let (es_o, es_m) = (gamma(19), 2f64.powi(-11) + 2.0 * gamma(66));
        let acc_o = gamma(span + live + 8);
        let (mut bo, mut bm) = (vec![0.0f64; HEAD_64], vec![0.0f64; HEAD_64]);
        for d in 0..HEAD_64 {
            let (mut t_o, mut t_m, mut pv) = (0.0f64, 0.0f64, 0.0f64);
            for j in 0..n {
                let v = f64::from(vh[j * HEAD_64 + d]);
                let dev = (v - o[d]).abs();
                let ee = 4.0 * U + 2.0 * U * x[j];
                t_o += pb[j] * (es_o * a[j] + ee + 4.0 * U * r + 2.0 * U * xmax) * dev;
                t_m += pb[j] * (es_m * a16[j] + ee + 4.0 * U * r + 2.0 * U * xmax) * dev;
                pv += pb[j] * v.abs();
            }
            bo[d] = t_o + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bm[d] = t_m + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
        }
        Exact64 {
            o,
            bound_ours: bo,
            bound_mma: bm,
        }
    }

    /// One launch of `m` head-64 query rows at `window` over the cache
    /// planes, into fresh scratch, raising on `fault`, read back — the
    /// head-64 `run_once`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the kernel, the card, the launch's own geometry and pass"
    )]
    fn run_once_64(
        k: &FlashGqaKernels,
        (stream, fault): (&CudaStream, FaultSink),
        q: &DeviceBuffer<f32>,
        n_keys: &DeviceBuffer<u32>,
        (kc, vc): (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        scale: f32,
        n_kv: usize,
        ctx: usize,
        window: usize,
        mma: bool,
    ) -> Result<Vec<f32>, GateError> {
        let (n_head, m) = (n_kv * GROUP, n_keys.len());
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len_64(m, n_head))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, n_head))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * n_head * HEAD_64)?;
        k.enqueue_pass_64(
            stream,
            Gqa64Args {
                q,
                kc,
                vc,
                n_keys,
                scale,
                n_kv,
                ctx,
                m,
                window,
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

    /// The head-64 entries (module doc): both passes over a seeded cache,
    /// window off and on. Returns whether every clause held.
    fn head64(k: &FlashGqaKernels, gpu: &Gpu, scale: f32) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let unl = gpu.unlabelled_sink();
        let run = (stream, unl);
        let (n_kv, ctx) = (H64_KV, H64_CTX);
        let n_head = n_kv * GROUP;
        let to16 = |v: Vec<f32>| -> Vec<u16> { v.iter().map(|&x| f32_to_f16_bits(x)).collect() };
        let kb = to16(activations(HEAD_64, n_kv * ctx, 740));
        let vb = to16(activations(HEAD_64, n_kv * ctx, 741));
        // The oracle's values are the planes' f16 bits, as the deep clause's.
        let (kf, vf): (Vec<f32>, Vec<f32>) = (
            kb.iter().map(|&h| half_to_f32(h)).collect(),
            vb.iter().map(|&h| half_to_f32(h)).collect(),
        );
        let (kc, vc) = (
            DeviceBuffer::from_host(stream, &kb)?,
            DeviceBuffer::from_host(stream, &vb)?,
        );
        let q: Vec<f32> = activations(HEAD_64, n_head, 742)
            .iter()
            .map(|v| v * SEED_Q_SCALE)
            .collect();
        let qd = DeviceBuffer::from_host(stream, &q)?;
        // One one-row launch of `count` keys at `window` through `mma` over
        // `ck`/`cv`.
        let launch = |q: &[f32],
                      count: usize,
                      window: usize,
                      mma: bool,
                      ck: &DeviceBuffer<u16>,
                      cv: &DeviceBuffer<u16>|
         -> Result<Vec<f32>, GateError> {
            let nk = DeviceBuffer::from_host(stream, &[u32::try_from(count)?])?;
            run_once_64(
                k,
                run,
                &DeviceBuffer::from_host(stream, q)?,
                &nk,
                (ck, cv),
                scale,
                n_kv,
                ctx,
                window,
                mma,
            )
        };
        // The NaN twins: every row at or past the count NaN.
        let nanify = |b: &[u16], past: usize| -> Vec<u16> {
            b.iter()
                .enumerate()
                .map(|(i, &h)| {
                    if (i / HEAD_64) % ctx >= past {
                        NAN16
                    } else {
                        h
                    }
                })
                .collect()
        };
        // A cache's planes with each key head's window re-based at row 0, the
        // rows past it NaN so the twin's ctx stride equals the cache's.
        let rebase = |b: &[u16], first: usize, rows: usize| -> Vec<u16> {
            b.chunks(ctx * HEAD_64)
                .flat_map(|p| {
                    p[first * HEAD_64..(first + rows) * HEAD_64]
                        .iter()
                        .copied()
                        .chain(std::iter::repeat_n(NAN16, (ctx - rows) * HEAD_64))
                })
                .collect()
        };
        let mut ok = true;

        // Window off (module doc): the band, a rerun, the NaN rows past the
        // count.
        for &count in &H64_COUNTS {
            let exacts: Vec<Exact64> = (0..n_head)
                .map(|h| {
                    let plane = (h / GROUP) * ctx * HEAD_64;
                    exact64(
                        &q[h * HEAD_64..(h + 1) * HEAD_64],
                        &kf[plane..plane + count * HEAD_64],
                        &vf[plane..plane + count * HEAD_64],
                        count,
                        scale,
                        SEGMENTS,
                    )
                })
                .collect();
            let (nan_k, nan_v) = if count == 65 || count == H64_CTX {
                (
                    &DeviceBuffer::from_host(stream, &nanify(&kb, count))?,
                    &DeviceBuffer::from_host(stream, &nanify(&vb, count))?,
                )
            } else {
                (&kc, &vc)
            };
            for mma in [false, true] {
                let name = if mma { "mma" } else { "scalar" };
                let y = launch(&q, count, 0, mma, &kc, &vc)?;
                let rerun = bits_equal(&y, &launch(&q, count, 0, mma, &kc, &vc)?);
                let mut worst = 0.0f64;
                let mut band = true;
                for (h, ex) in exacts.iter().enumerate() {
                    let bound = if mma { &ex.bound_mma } else { &ex.bound_ours };
                    for d in 0..HEAD_64 {
                        let e = (f64::from(y[h * HEAD_64 + d]) - ex.o[d]).abs() / bound[d];
                        worst = worst.max(e);
                        band &= e <= 1.0;
                    }
                }
                let nan_same = if count == 65 || count == H64_CTX {
                    bits_equal(&y, &launch(&q, count, 0, mma, nan_k, nan_v)?)
                } else {
                    true
                };
                let pass_ok = band && rerun && nan_same;
                println!(
                    "flash h64 window=0 count={count} pass={name} measured/bound {worst:.3e} \
                     band={band} rerun={rerun} nan_past_count_same={nan_same} {}",
                    verdict(pass_ok)
                );
                ok &= pass_ok;
            }
        }

        // Window on (module doc): the band over the window's own keys, and
        // the re-based twin bit for bit.
        for &count in &H64_WIN_COUNTS {
            let (first, n_w) = window_cut(count, H64_WINDOW);
            let segs = window_segments(H64_WINDOW);
            let exacts: Vec<Exact64> = (0..n_head)
                .map(|h| {
                    let plane = (h / GROUP) * ctx * HEAD_64;
                    exact64(
                        &q[h * HEAD_64..(h + 1) * HEAD_64],
                        &kf[plane + first * HEAD_64..plane + count * HEAD_64],
                        &vf[plane + first * HEAD_64..plane + count * HEAD_64],
                        n_w,
                        scale,
                        segs,
                    )
                })
                .collect();
            let (rk, rv) = (
                DeviceBuffer::from_host(stream, &rebase(&kb, first, n_w))?,
                DeviceBuffer::from_host(stream, &rebase(&vb, first, n_w))?,
            );
            for mma in [false, true] {
                let name = if mma { "mma" } else { "scalar" };
                let y = launch(&q, count, H64_WINDOW, mma, &kc, &vc)?;
                let mut worst = 0.0f64;
                let mut band = true;
                for (h, ex) in exacts.iter().enumerate() {
                    let bound = if mma { &ex.bound_mma } else { &ex.bound_ours };
                    for d in 0..HEAD_64 {
                        let e = (f64::from(y[h * HEAD_64 + d]) - ex.o[d]).abs() / bound[d];
                        worst = worst.max(e);
                        band &= e <= 1.0;
                    }
                }
                let twin = launch(&q, n_w, 0, mma, &rk, &rv)?;
                let twin_same = bits_equal(&y, &twin);
                let pass_ok = band && twin_same;
                println!(
                    "flash h64 window={H64_WINDOW} count={count} first={first} pass={name} \
                     measured/bound {worst:.3e} band={band} rebased_twin_bit_same={twin_same} {}",
                    verdict(pass_ok)
                );
                ok &= pass_ok;
            }
        }

        // The prefill shape (module doc): one launch of ROWS rotated rows at
        // spread counts, each bit for bit its one-row launch.
        let rows: Vec<Vec<f32>> = (0..ROWS)
            .map(|t| {
                (0..n_head)
                    .flat_map(|h| q[((h + t) % n_head) * HEAD_64..][..HEAD_64].iter().copied())
                    .collect()
            })
            .collect();
        let limits = spread_counts(ctx)?;
        let rq = DeviceBuffer::from_host(stream, &rows.concat())?;
        let lk = DeviceBuffer::from_host(stream, &limits)?;
        for window in [0, H64_WINDOW] {
            for mma in [false, true] {
                let name = if mma { "mma" } else { "scalar" };
                let all = run_once_64(k, run, &rq, &lk, (&kc, &vc), scale, n_kv, ctx, window, mma)?;
                let mut same = true;
                for (t, row) in rows.iter().enumerate() {
                    let one = run_once_64(
                        k,
                        run,
                        &DeviceBuffer::from_host(stream, row)?,
                        &DeviceBuffer::from_host(stream, &limits[t..=t])?,
                        (&kc, &vc),
                        scale,
                        n_kv,
                        ctx,
                        window,
                        mma,
                    )?;
                    same &=
                        bits_equal(&all[t * n_head * HEAD_64..(t + 1) * n_head * HEAD_64], &one);
                }
                println!(
                    "flash h64 rows m={ROWS} window={window} pass={name} keys={limits:?} ctx={ctx}: \
                     each row = its one-row launch bit for bit {}",
                    verdict(same)
                );
                ok &= same;
            }
        }

        // The fault rows (module doc): row 3 one past the cache, row 5 zero,
        // at both windows, labelled with layer 13.
        for window in [0, H64_WINDOW] {
            let layer = 13usize;
            let faulted = (stream, gpu.layer_sink(layer)?);
            let want = Some(Fault::at(u32::try_from(layer)?, FaultSite::KeyCount));
            let (bad_hi, bad_zero) = (3usize, 5usize);
            let mut bad = limits.clone();
            bad[bad_hi] = u32::try_from(ctx + 1)?;
            bad[bad_zero] = 0;
            let bad_d = DeviceBuffer::from_host(stream, &bad)?;
            let before = gpu.fault()?;
            let y = run_once_64(
                k,
                faulted,
                &rq,
                &lk,
                (&kc, &vc),
                scale,
                n_kv,
                ctx,
                window,
                false,
            )?;
            let after_clean = gpu.fault()?;
            let yb = run_once_64(
                k,
                faulted,
                &rq,
                &bad_d,
                (&kc, &vc),
                scale,
                n_kv,
                ctx,
                window,
                false,
            )?;
            let raised = gpu.take_fault()?;
            let mut others_same = true;
            let mut bad_nan = true;
            let w = n_head * HEAD_64;
            for r in 0..ROWS {
                let (a, b) = (&y[r * w..(r + 1) * w], &yb[r * w..(r + 1) * w]);
                if r == bad_hi || r == bad_zero {
                    bad_nan &= b.iter().all(|v| v.is_nan());
                } else {
                    others_same &= bits_equal(a, b);
                }
            }
            let y2 = run_once_64(
                k,
                faulted,
                &rq,
                &lk,
                (&kc, &vc),
                scale,
                n_kv,
                ctx,
                window,
                false,
            )?;
            let clean_again = bits_equal(&y, &y2) && gpu.fault()?.is_none();
            let pass = before.is_none()
                && after_clean.is_none()
                && raised == want
                && bad_nan
                && others_same
                && clean_again;
            println!(
                "flash h64 fault window={window} m={ROWS} counts {} (past ctx {}) and 0 at rows \
                 {bad_hi}, {bad_zero}: word \"{}\" (want \"{}\", clean before {} and after the \
                 clean run {}), those rows NaN {bad_nan}, other rows bit-identical {others_same}, \
                 clean rerun bits and word clean {clean_again} {}",
                ctx + 1,
                ctx,
                raised.map_or_else(|| "none".to_owned(), |f| f.to_string()),
                want.map_or_else(String::new, |f| f.to_string()),
                before.is_none(),
                after_clean.is_none(),
                verdict(pass)
            );
            ok &= pass;
        }

        // The captured launch (module doc): two nodes replaying the eager
        // bits, the segment grid the window's own.
        let nk1 = DeviceBuffer::from_host(stream, &[u32::try_from(ctx)?])?;
        for window in [0, H64_WINDOW] {
            let segs = window_segments(window);
            for mma in [false, true] {
                let name = if mma { "mma" } else { "scalar" };
                let want = launch(&q, ctx, window, mma, &kc, &vc)?;
                let (mut pv, mut pms, mut y) = (
                    DeviceBuffer::<f32>::zeroed(stream, partials_v_len_64(1, n_head))?,
                    DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head))?,
                    DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD_64)?,
                );
                let graph = gpu.capture(|s| {
                    k.enqueue_pass_64(
                        s,
                        Gqa64Args {
                            q: &qd,
                            kc: &kc,
                            vc: &vc,
                            n_keys: &nk1,
                            scale,
                            n_kv,
                            ctx,
                            m: 1,
                            window,
                            part_v: &mut pv,
                            part_ms: &mut pms,
                            fault: gpu.unlabelled_sink(),
                            y: &mut y,
                        },
                        mma,
                    )
                })?;
                graph.launch(stream)?;
                stream.synchronize()?;
                let nodes = graph.nodes()?;
                let grids = |prefix: &str| -> Vec<[u32; 3]> {
                    nodes
                        .iter()
                        .filter_map(|n| n.kernel.as_ref())
                        .filter(|kn| kn.name.starts_with(prefix))
                        .map(|kn| kn.grid)
                        .collect()
                };
                let (seg, merge) = (grids("gqa_flash_seg"), grids("gqa_flash_merge"));
                let want_seg = vec![[u32::try_from(n_kv * segs)?, 1, 1]];
                let want_merge = vec![[u32::try_from(n_head)?, 1, 1]];
                let shape = nodes.len() == 2
                    && nodes.iter().all(|n| n.kernel.is_some())
                    && seg == want_seg
                    && merge == want_merge;
                let same = bits_equal(&y.to_host_vec(stream)?, &want);
                let pass = shape && same;
                println!(
                    "flash h64 graph window={window} pass={name} nodes={} seg_grids={seg:?} (want \
                     {want_seg:?}) merge_grids={merge:?} (want {want_merge:?}) \
                     replay_bits_eager={same} {}",
                    nodes.len(),
                    verdict(pass)
                );
                ok &= pass;
            }
        }

        // A `part_v` one short: refused by name.
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len_64(1, n_head) - 1)?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(1, n_head))?;
        let mut yr = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD_64)?;
        let r = k.enqueue_pass_64(
            stream,
            Gqa64Args {
                q: &qd,
                kc: &kc,
                vc: &vc,
                n_keys: &nk1,
                scale,
                n_kv,
                ctx,
                m: 1,
                window: 0,
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
                what: "flash_gqa::enqueue_64",
                detail,
            }) if detail.contains("part_v.len()")
        );
        println!(
            "flash h64 part_v one short refused by name {named} {}",
            verdict(named)
        );
        ok &= named;
        Ok(ok)
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
            "gate_qwen3moe_flash: device {} — head {HEAD}, group {GROUP}, {SEGMENTS} segments of at \
             least {SEG_KEYS} keys, scale {scale:e}",
            gpu.device_name()?
        );
        let mut ok = true;
        let mut grid_done = false;
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
                        run_once::<GROUP>(
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
                        set_ok &= rows_check::<GROUP>(&k, &gpu, &q, &inp, live, &g, mma)?;
                        set_ok &= decode_fault::<GROUP>(&k, &gpu, &q, &inp, live, &g, mma)?;
                    }
                    ys.push(y);
                }

                // The prefill flash: the step's row alone, at its count.
                let run_p = |i: &Inputs| {
                    run_pref(&kp, &gpu, &i.q, &[live as u32], (&i.kc, &i.vc), &g, None)
                };
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

                // The q8_0 read path (module doc): every layer's two entries
                // over the cache's Q8_0 form, layer 0's extras once a set.
                set_ok &= flash_q8_layer(&k, &kp, &gpu, &q, &kf, &vf, live, &g, label, l, l == 0)?;

                // The grid clause, once: its capture at the set's own height
                // is also the captured launch's clause (two nodes, the eager
                // bits).
                if !grid_done {
                    ok &= grid_heights(&k, &gpu, &inp, (&kf, &vf), (&kb, &vb), &ys, live, &g)?;
                    grid_done = true;
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
                    "flash set={label} pass={} layers={layers} keys={n_live} ctx={ctx_seen} segments={} \
                     span={} live_segments={} — \
                     measured / bound: ours-exact {:.3e}, ik-exact {:.3e}, ours-ik {:.3e}; ours-ik plain \
                     {:.3e} of max|ik| (printed)",
                    if pass == 1 { "mma" } else { "scalar" },
                    SEGMENTS,
                    seg_span(n_live, SEGMENTS, SEG_KEYS),
                    n_live.div_ceil(seg_span(n_live, SEGMENTS, SEG_KEYS)),
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
        let deep_ok = deep_counts(&k, &gpu, scale)?;
        println!("flash deep counts {}", verdict(deep_ok));
        ok &= deep_ok;
        let seeded_ok = prefill_seeded(&kp, &gpu, scale)?;
        println!(
            "prefill flash seeded, fault, graph, refusal {}",
            verdict(seeded_ok)
        );
        ok &= seeded_ok;
        let split_ok = prefill_split(&kp, &gpu, scale)?;
        println!("prefill flash split {}", verdict(split_ok));
        ok &= split_ok;
        ok &= q8_window_refusal(&kp, &gpu, scale)?;
        ok &= group5(&k, &gpu, scale)?;
        ok &= group16(&k, &gpu, scale)?;
        ok &= head64(&k, &gpu, 1.0f32 / (HEAD_64 as f32).sqrt())?;
        println!("gate_qwen3moe_flash: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
