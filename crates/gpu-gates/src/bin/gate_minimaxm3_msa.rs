//! GPU gate for MiniMax-M3's sparse attention (MSA), model-less at the model's shapes: 4 index heads
//! of 128 values over one index key per position, blocks of 128 positions, the 16 best kept with the
//! block of the query forced, then 64 query heads over 4 key heads of 128 values attending the
//! kept blocks. Inputs are seeded ([`activations`]); nothing is read from a model file. The rule is
//! `runtime::msa`'s; the kernels are `bloomery_gpu_minimaxm3::attn`'s and the selected flash
//! `flash_gqa::enqueue_pass_p16_blocks`.
//!
//! 0. The shape check (`MsaShape::check`): the model's facts are accepted, a different head count,
//!    head width, rope width, index shape, `top_k` or `local` is refused naming its key.
//! 1. The index heads (`attn::enqueue_index`, the common head kernel at 4 heads over 1 key head):
//!    all bit for bit against the shared host transcription of the head kernel (`rope_gate`), the
//!    key plane's other rows untouched, a rerun bit for bit, and the cache-position and overflow
//!    faults of the rope gate.
//! 2. The selection (`attn::MsaKernels`) against its host twin at live counts below one block, at
//!    one, at 16 blocks (2,048: every block, no score computed), at 17 (2,049: one dropped) and up to
//!    32,768: each index head's block scores equal the twin's bit for bit (f16-rounded query, the
//!    lane dot over four rotating fused-multiply-add partials combined `(a0 + a1) + (a2 + a3)`, the
//!    largest of the block's visible positions by `order_key`), the kept blocks equal
//!    `runtime::msa`'s rule on the twin's scores and on the launch's own scores, the key count is the
//!    rule's, a row that does not select writes no score and lists every block, a count of 0 or past
//!    the cache gets length 0 and writes nothing, and an eight-row launch gives each row its one-row
//!    launch's bits.
//! 3. The tie rule: every full block holds the same keys, so every block scores alike; the 15 lowest
//!    blocks and the query's own are kept.
//! 4. The forced block: the query's own block scores lowest of all and is kept.
//! 5. No silent failure: a NaN key raises `pool_select` with a defined list (distinct blocks in range,
//!    ascending, the own block among them, the rule's count).
//! 6. The selected flash at the counts of 2: with every block kept (up to 2,048 keys) it is the dense
//!    group-of-16 tensor-core pass's bits; past it every (row, head) is within the band of the exact
//!    attention over its key head's kept keys (f64 on the cache's f16 values; the decode tensor-core
//!    pass's model, `gate_qwen35moe_attn`'s `exact`, derived in [`bound`]), a rerun gives the bits,
//!    NaN in every cache row no list names changes no bit, and each row of an eight-row launch is its
//!    one-row launch's bits.
//! 7. The flash's faults: a block past the cache raises `pool_select` and NaNs its key head's rows
//!    of that row; a length of 0 raises `key_count` and NaNs the row; every other (row, head) is the
//!    clean run's bits.

#[cfg(feature = "minimaxm3")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_minimaxm3_msa", gate::run())
}

#[cfg(feature = "minimaxm3")]
mod gate {
    use bloomery_gpu::fault::{Fault, FaultSink, FaultSite};
    use bloomery_gpu::flash_gqa::{
        FlashGqaKernels, GROUP_16, GqaArgs, GqaBlocksArgs, KEY_TILE, SEG_KEYS,
        listed_partials_ms_len, listed_partials_v_len, listed_segments, partials_ms_len,
        partials_v_len,
    };
    use bloomery_gpu::rope_neox::{HEAD, PartialNeoxArgs, ROT_PARTIAL, RopeNeoxKernels};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::flash_band::{ExactAttn, exact_attn};
    use bloomery_gpu_gates::prefill_gate::{HostCache, cache};
    use bloomery_gpu_gates::rope_gate::{self, RopeIn, RopeOut, RopeShape};
    use bloomery_gpu_gates::rounding::{U, gamma};
    use bloomery_gpu_gates::{GateError, activations, bits_equal, checks_failed, par_map, verdict};
    use bloomery_gpu_minimaxm3::attn::{
        AttnArgs, BLOCK, DIM, HEADS, IndexArgs, MsaKernels, MsaScratch, MsaShape, N_HEAD, PickArgs,
        ScoreArgs, attn_scale, blocks_for, enqueue_attention, enqueue_index, order_value,
        partials_lens,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use model::arch::minimaxm3::hparams::Msa;
    use runtime::qsa::order_key;

    /// The cache's height: positions `0..CTX`, 256 blocks.
    const CTX: usize = 32768;
    /// The layer the fault checks label their launches with.
    const LAYER: usize = 17;
    /// An f16 NaN.
    const NAN16: u16 = 0x7e00;
    /// The scores' sentinel: an f32 NaN pattern no kernel writes, and the lists' (no block).
    const SENT: u32 = 0x7fc0_dead;
    const NO_ID: u32 = u32::MAX;
    /// The scale of the seeded query rows.
    const Q_SCALE: f32 = 3.0;
    const THETA: f32 = 1.0e7;
    const EPS: f32 = 1.0e-6;
    /// The rope check's tokens, last position and cache height.
    const ROPE_M: usize = 9;
    const ROPE_CTX: usize = 40;

    /// Live counts that do not select: below one block, one block, two, and 16 blocks.
    const NONSEL: [u32; 8] = [1, 64, 127, 128, 129, 1000, 2047, 2048];
    /// Live counts that do: 17 blocks (one key in the 17th) up to 32k.
    const SEL: [u32; 9] = [2049, 2050, 2176, 2177, 4096, 5000, 8193, 16000, 32768];

    struct Rig<'a> {
        gpu: &'a Gpu,
        msa: MsaKernels,
        flash: FlashGqaKernels,
        shape: MsaShape,
    }

    // ------------------------------------------------------------- the twin

    fn f16r(v: f32) -> f32 {
        half_to_f32(f32_to_f16_bits(v))
    }

    /// The score kernel's dot of the f16-rounded query `q` with the key row `k` (f32 of f16 values).
    fn dot_twin(q: &[f32], k: &[f32]) -> f32 {
        let mut a = [0.0f32; 4];
        let mut wd = 0;
        while wd < DIM / 2 {
            for (p, ap) in a.iter_mut().enumerate() {
                let w = wd + p;
                *ap = q[2 * w].mul_add(k[2 * w], *ap);
                *ap = q[2 * w + 1].mul_add(k[2 * w + 1], *ap);
            }
            wd += 4;
        }
        (a[0] + a[1]) + (a[2] + a[3])
    }

    /// Each index head's block scores of a row of live count `c` as order keys: the largest of the
    /// block's visible positions' dots. `iq` the row's `HEADS · DIM` query values, `ikf` the f32 of
    /// the f16 index keys.
    fn pooled_twin(iq: &[f32], ikf: &[f32], c: usize) -> Vec<Vec<u32>> {
        let visible = c.div_ceil(BLOCK);
        par_map(HEADS, |h| {
            let q: Vec<f32> = iq[h * DIM..(h + 1) * DIM]
                .iter()
                .map(|&v| f16r(v))
                .collect();
            (0..visible)
                .map(|b| {
                    let n = (c - b * BLOCK).min(BLOCK);
                    (0..n)
                        .map(|i| order_key(dot_twin(&q, &ikf[(b * BLOCK + i) * DIM..][..DIM])))
                        .max()
                        .expect("a visible block holds a key")
                })
                .collect()
        })
    }

    fn f16_plane(f: &[f32]) -> Vec<u16> {
        f.iter().map(|&v| f32_to_f16_bits(v)).collect()
    }

    fn widen(b: &[u16]) -> Vec<f32> {
        b.iter().map(|&v| half_to_f32(v)).collect()
    }

    // --------------------------------------------------------- the launches

    /// One selection launch's outputs.
    struct Out {
        scores: Vec<f32>,
        ids: Vec<u32>,
        n_sel: Vec<u32>,
        fault: Option<Fault>,
        nblk: usize,
    }

    /// One launch of the chain over `counts`: score, pick and (when `q` is given) the selected flash
    /// over the cache planes, into scratch that starts as sentinels, read back.
    #[allow(
        clippy::too_many_arguments,
        reason = "the rig, the rows, the index plane, the query and cache, the chain's reach"
    )]
    fn run_chain(
        rig: &Rig,
        iq: &[f32],
        counts: &[u32],
        ik: &DeviceBuffer<u16>,
        attn: Option<(&[f32], &DeviceBuffer<u16>, &DeviceBuffer<u16>)>,
    ) -> Result<(Out, Vec<f32>), GateError> {
        let stream = rig.gpu.stream();
        let m = counts.len();
        let k = rig.shape.top_k();
        let nblk = blocks_for(CTX);
        let mut scratch = MsaScratch::new(stream, &rig.shape, m, CTX)?;
        scratch.scores =
            DeviceBuffer::from_host(stream, &vec![f32::from_bits(SENT); m * HEADS * nblk])?;
        scratch.ids = DeviceBuffer::from_host(stream, &vec![NO_ID; m * HEADS * k])?;
        scratch.n_sel = DeviceBuffer::from_host(stream, &vec![NO_ID; m])?;
        let iqd = DeviceBuffer::from_host(stream, iq)?;
        let nk = DeviceBuffer::from_host(stream, counts)?;
        let blocks = counts
            .iter()
            .map(|&c| (c as usize).div_ceil(BLOCK))
            .max()
            .unwrap_or(1)
            .clamp(1, nblk);
        let fault = rig.gpu.layer_sink(LAYER)?;
        rig.msa.enqueue_score(
            stream,
            &rig.shape,
            ScoreArgs {
                iq: &iqd,
                ik,
                n_keys: &nk,
                ctx: CTX,
                m,
                blocks,
                fault,
            },
            &mut scratch,
        )?;
        rig.msa.enqueue_pick(
            stream,
            &rig.shape,
            PickArgs {
                n_keys: &nk,
                ctx: CTX,
                m,
                fault,
            },
            &mut scratch,
        )?;
        let mut y = Vec::new();
        if let Some((q, kc, vc)) = attn {
            let qd = DeviceBuffer::from_host(stream, q)?;
            let (lv, lm) = partials_lens(&rig.shape, m);
            let mut pv = DeviceBuffer::<f32>::zeroed(stream, lv)?;
            let mut pms = DeviceBuffer::<f32>::zeroed(stream, lm)?;
            let mut yd = DeviceBuffer::<f32>::zeroed(stream, m * N_HEAD * DIM)?;
            enqueue_attention(
                &rig.flash,
                stream,
                &rig.shape,
                &scratch,
                AttnArgs {
                    q: &qd,
                    kc,
                    vc,
                    ctx: CTX,
                    m,
                    part_v: &mut pv,
                    part_ms: &mut pms,
                    fault,
                    y: &mut yd,
                },
            )?;
            stream.synchronize()?;
            y = yd.to_host_vec(stream)?;
        }
        stream.synchronize()?;
        let raised = rig.gpu.take_fault()?;
        Ok((
            Out {
                scores: scratch.scores.to_host_vec(stream)?,
                ids: scratch.ids.to_host_vec(stream)?,
                n_sel: scratch.n_sel.to_host_vec(stream)?,
                fault: raised,
                nblk,
            },
            y,
        ))
    }

    /// The selection of `out` against the twin and the rule (module doc, 2): whether it held.
    fn check_select(
        label: &str,
        rig: &Rig,
        out: &Out,
        iq: &[f32],
        counts: &[u32],
        ikf: &[f32],
    ) -> bool {
        let rule = rig.shape.rule();
        let k = rig.shape.top_k();
        let nblk = out.nblk;
        let (mut scored, mut bad_scores, mut bad_ids, mut bad_len, mut bad_untouched) =
            (0usize, 0usize, 0usize, 0usize, 0usize);
        for (t, &cu) in counts.iter().enumerate() {
            let c = cu as usize;
            let row_scores = |h: usize| &out.scores[(t * HEADS + h) * nblk..][..nblk];
            let row_ids = |h: usize| &out.ids[(t * HEADS + h) * k..][..k];
            let untouched_scores =
                |h: usize, from: usize| row_scores(h)[from..].iter().all(|v| v.to_bits() == SENT);
            if !(1..=CTX).contains(&c) {
                bad_len += usize::from(out.n_sel[t] != 0);
                for h in 0..HEADS {
                    bad_untouched += usize::from(
                        !untouched_scores(h, 0) || row_ids(h).iter().any(|&i| i != NO_ID),
                    );
                }
                continue;
            }
            let visible = c.div_ceil(BLOCK);
            bad_len += usize::from(out.n_sel[t] as usize != rule.key_count(c));
            if !rule.scored(c, CTX) {
                for h in 0..HEADS {
                    let want: Vec<u32> = (0..visible as u32).collect();
                    bad_ids += usize::from(row_ids(h)[..visible] != want[..]);
                    bad_untouched += usize::from(
                        !untouched_scores(h, 0)
                            || row_ids(h)[visible..].iter().any(|&i| i != NO_ID),
                    );
                }
                continue;
            }
            scored += 1;
            let twin = pooled_twin(&iq[t * HEADS * DIM..][..HEADS * DIM], ikf, c);
            for (h, twin_h) in twin.iter().enumerate() {
                let dev = row_scores(h);
                bad_scores += dev[..visible]
                    .iter()
                    .zip(twin_h)
                    .filter(|&(d, &key)| d.to_bits() != order_value(key).to_bits())
                    .count();
                bad_untouched += usize::from(!untouched_scores(h, visible));
                let tw: Vec<f32> = twin_h.iter().map(|&key| order_value(key)).collect();
                let kept = rule.kept(c);
                let got = &row_ids(h)[..kept];
                bad_ids += usize::from(
                    got != rule.select(c, &tw).as_slice()
                        || got != rule.select(c, &dev[..visible]).as_slice(),
                );
                bad_untouched += usize::from(row_ids(h)[kept..].iter().any(|&i| i != NO_ID));
            }
        }
        let clean = out.fault.is_none();
        let ok = bad_scores == 0 && bad_ids == 0 && bad_len == 0 && bad_untouched == 0 && clean;
        println!(
            "select {label} counts={counts:?}: {scored} scored rows, block scores bit_exact_twin \
             {} ids = rule(twin) = rule(device) {} lengths = rule {} nothing else written {} fault \
             word clean {clean} {}",
            bad_scores == 0,
            bad_ids == 0,
            bad_len == 0,
            bad_untouched == 0,
            verdict(ok)
        );
        ok
    }

    fn iq_rows(m: usize, seed: u32) -> Vec<f32> {
        activations(DIM, m * HEADS, seed)
    }

    // ----------------------------------------------------- 1. the index heads

    /// The index heads' rows: the shared seeded head inputs at one key head, with no value row (the
    /// index cache holds keys only).
    fn index_inputs() -> Result<(RopeIn, Vec<u32>), GateError> {
        let (mut inp, pos) =
            rope_gate::seeded_inputs(HEAD, ROT_PARTIAL, THETA, (ROPE_CTX, ROPE_M), (HEADS, 1))?;
        inp.v = vec![0.0; ROPE_M * HEAD];
        Ok((inp, pos))
    }

    fn index_run(
        rope: &RopeNeoxKernels,
        stream: &cuda_core::CudaStream,
        inp: &RopeIn,
        pos: &[u32],
        fault: FaultSink,
    ) -> Result<RopeOut, GateError> {
        rope_gate::run(
            stream,
            inp,
            pos,
            RopeShape {
                head: HEAD,
                n_head: HEADS,
                n_kv: 1,
                ctx: ROPE_CTX,
                eps: EPS,
            },
            fault,
            |s, a: PartialNeoxArgs<'_>| {
                // SAFETY: rope_gate::run checks that the positions are distinct.
                unsafe {
                    enqueue_index(
                        rope,
                        s,
                        IndexArgs {
                            proj_q: a.qg,
                            iq: a.q,
                            proj_k: a.k,
                            zeros: a.v,
                            gain_q: a.gq,
                            gain_k: a.gk,
                            table: a.table,
                            pos: a.pos,
                            eps: a.eps,
                            ctx: a.ctx,
                            m: a.m,
                            fault: a.fault,
                            keys: a.cache_k,
                            sink: a.cache_v,
                        },
                    )
                }
            },
        )
    }

    fn index_check(gpu: &Gpu, rope: &RopeNeoxKernels) -> Result<bool, GateError> {
        let (inp, pos) = index_inputs()?;
        let shape = RopeShape {
            head: HEAD,
            n_head: HEADS,
            n_kv: 1,
            ctx: ROPE_CTX,
            eps: EPS,
        };
        let exact =
            rope_gate::check::<HEAD, ROT_PARTIAL, HEAD>(gpu, &inp, &pos, shape, LAYER, |p, f| {
                index_run(rope, gpu.stream(), &inp, p, f)
            })?;
        let overflow =
            rope_gate::check_overflow::<HEAD>(gpu, &inp, &pos, shape, LAYER, |i, p, f| {
                index_run(rope, gpu.stream(), i, p, f)
            })?;
        println!(
            "index heads {HEADS}/1 of {HEAD}, rope over the first {ROT_PARTIAL}: head kernel's \
             host transcription bit_exact={exact} faults={overflow} {}",
            verdict(exact && overflow)
        );
        Ok(exact && overflow)
    }

    // ------------------------------------------------------ 6. the flash band

    /// One (row, head)'s exact attention over its key head's kept keys and the decode tensor-core
    /// pass's bound on each value (`gate_qwen35moe_attn`'s `exact`, `bound_mma`, at head 128): the
    /// scores' f16 query rounding and tensor-core chains (`2^-11 + 2γ(HEAD + 2)` on `a16`), one
    /// rescale per key tile and segment (`r`), and the f32 value accumulation over a segment's keys,
    /// the segments and the merge (`γ(span + segs + 8)`). The kept keys are cut over list
    /// positions in 64-key segments (`seg_span` at the list's width), the span the floor.
    fn bound(ex: &ExactAttn, vh: &[f32], n: usize) -> Vec<f64> {
        let width = 16 * BLOCK;
        let segs_total = listed_segments(width);
        let span = bloomery_gpu::flash_gqa::seg_span(n, segs_total, SEG_KEYS);
        let segs = n.div_ceil(span);
        let r = (n.div_ceil(KEY_TILE) + segs) as f64;
        let es = 2f64.powi(-11) + 2.0 * gamma(HEAD + 2);
        let acc = gamma(span + segs + 8);
        (0..HEAD)
            .map(|d| {
                let (mut t_m, mut pv) = (0.0f64, 0.0f64);
                for j in 0..n {
                    let v = f64::from(vh[j * HEAD + d]);
                    let dev = (v - ex.o[d]).abs();
                    let ee = 4.0 * U + 2.0 * U * ex.x[j];
                    t_m += ex.pb[j] * (es * ex.a16[j] + ee + 4.0 * U * r + 2.0 * U * ex.xmax) * dev;
                    pv += ex.pb[j] * v.abs();
                }
                t_m + acc * (pv + ex.o[d].abs()) + 2.0 * U * ex.o[d].abs()
            })
            .collect()
    }

    /// The cache positions a (row, key head)'s kept blocks stand for: block `ids[i]`'s positions
    /// below the row's key count `n`, in list order.
    fn listed_positions(ids: &[u32], n: usize) -> Vec<usize> {
        (0..n)
            .map(|j| ids[j / BLOCK] as usize * BLOCK + j % BLOCK)
            .collect()
    }

    /// Every value of `y` against the exact attention of its (row, head) over the kept keys, within
    /// [`bound`]. Returns whether every value is in band and the largest measured over bound.
    fn judge(q: &[f32], out: &Out, k: usize, host: &HostCache, y: &[f32]) -> (bool, f64) {
        let m = out.n_sel.len();
        let res = par_map(m * N_HEAD, |i| {
            let (t, h) = (i / N_HEAD, i % N_HEAD);
            let kh = h / (N_HEAD / HEADS);
            let n = out.n_sel[t] as usize;
            let ids = &out.ids[(t * HEADS + kh) * k..][..k];
            let pos = listed_positions(ids, n);
            let gather = |src: &[f32]| -> Vec<f32> {
                pos.iter()
                    .flat_map(|&p| src[(kh * host.ctx + p) * HEAD..][..HEAD].iter().copied())
                    .collect()
            };
            let (kg, vg) = (gather(&host.kf), gather(&host.vf));
            let ex = exact_attn(&q[i * HEAD..(i + 1) * HEAD], &kg, &vg, HEAD, n, scale());
            let bd = bound(&ex, &vg, n);
            let (mut ok, mut worst) = (true, 0.0f64);
            for d in 0..HEAD {
                let e = (f64::from(y[i * HEAD + d]) - ex.o[d]).abs();
                ok &= e <= bd[d];
                worst = worst.max(if e.is_nan() { f64::INFINITY } else { e / bd[d] });
            }
            (ok, worst)
        });
        res.into_iter()
            .fold((true, 0.0f64), |(ok, w), (o, x)| (ok && o, w.max(x)))
    }

    /// The attention's score scale as the reference forms it, `1.0f/sqrtf(float(128))` (the gate's
    /// own, so a kernel run at another scale is judged against this one).
    fn scale() -> f32 {
        1.0 / 128f32.sqrt()
    }

    fn q_rows(m: usize, seed: u32) -> Vec<f32> {
        activations(HEAD, m * N_HEAD, seed)
            .iter()
            .map(|v| v * Q_SCALE)
            .collect()
    }

    // --------------------------------------------------------------- the run

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let rope = RopeNeoxKernels::load(gpu.context())?;
        let rig = Rig {
            gpu: &gpu,
            msa: MsaKernels::load(gpu.context())?,
            flash: FlashGqaKernels::load(gpu.context())?,
            shape: MsaShape::new(16, 1)?,
        };
        println!(
            "gate_minimaxm3_msa: device {}: {HEADS} index heads of {DIM}, blocks of {BLOCK}, top {} \
             kept (local 1), {N_HEAD}/{HEADS} heads of {HEAD}, ctx {CTX}",
            gpu.device_name()?,
            rig.shape.top_k()
        );
        let mut ok = true;

        // 0. The shape check accepts the model's facts and refuses each other value by name.
        {
            let model = Msa {
                heads: 4,
                d: 128,
                top_k: 16,
                block: 128,
                local: 1,
            };
            let with = |f: fn(&mut Msa)| {
                let mut m = model;
                f(&mut m);
                m
            };
            let cases: [(&str, Result<MsaShape, GpuError>); 11] = [
                (
                    "attention.head_count",
                    MsaShape::check(32, 4, 128, 64, model),
                ),
                (
                    "attention.head_count_kv",
                    MsaShape::check(64, 8, 128, 64, model),
                ),
                (
                    "attention.key_length",
                    MsaShape::check(64, 4, 64, 64, model),
                ),
                (
                    "rope.dimension_count",
                    MsaShape::check(64, 4, 128, 128, model),
                ),
                (
                    "attention.indexer.head_count",
                    MsaShape::check(64, 4, 128, 64, with(|m| m.heads = 8)),
                ),
                (
                    "attention.indexer.key_length",
                    MsaShape::check(64, 4, 128, 64, with(|m| m.d = 64)),
                ),
                (
                    "attention.indexer.block_size",
                    MsaShape::check(64, 4, 128, 64, with(|m| m.block = 64)),
                ),
                (
                    "top_k",
                    MsaShape::check(64, 4, 128, 64, with(|m| m.top_k = 33)),
                ),
                (
                    "top_k",
                    MsaShape::check(64, 4, 128, 64, with(|m| m.top_k = 0)),
                ),
                (
                    "local",
                    MsaShape::check(64, 4, 128, 64, with(|m| m.local = 0)),
                ),
                (
                    "local",
                    MsaShape::check(64, 4, 128, 64, with(|m| m.local = 17)),
                ),
            ];
            let refused = cases
                .iter()
                .all(|(key, r)| matches!(r, Err(e) if e.to_string().contains(key)));
            let accepted =
                matches!(MsaShape::check(64, 4, 128, 64, model), Ok(s) if s == rig.shape);
            println!(
                "shape: the model's facts accepted {accepted}, {} other values each refused naming \
                 their key {refused} {}",
                cases.len(),
                verdict(accepted && refused)
            );
            ok &= accepted && refused;
        }

        // 1. The index heads.
        ok &= index_check(&gpu, &rope)?;

        // The seeded index keys: random rows, and the planes the tie and forced clauses use.
        let ik_f = activations(DIM, CTX, 5);
        let ik_bits = f16_plane(&ik_f);
        let ikf = widen(&ik_bits);
        let ik = DeviceBuffer::from_host(stream, &ik_bits)?;

        // 2. The selection.
        let iq1 = |seed| iq_rows(1, seed);
        for (name, counts) in [
            ("below the cut", NONSEL.to_vec()),
            ("selecting", SEL.to_vec()),
        ] {
            // Each count alone, then all of them in one launch: each row its one-row bits.
            let mut alone = true;
            let iq = iq_rows(counts.len(), 21);
            let (all, _) = run_chain(&rig, &iq, &counts, &ik, None)?;
            for (t, &c) in counts.iter().enumerate() {
                let rowq = &iq[t * HEADS * DIM..(t + 1) * HEADS * DIM];
                let (one, _) = run_chain(&rig, rowq, &[c], &ik, None)?;
                let r = check_select(name, &rig, &one, rowq, &[c], &ikf);
                ok &= r;
                let k = rig.shape.top_k();
                alone &= one.ids[..HEADS * k] == all.ids[t * HEADS * k..(t + 1) * HEADS * k]
                    && one.n_sel[0] == all.n_sel[t]
                    && bits_equal(
                        &one.scores[..HEADS * one.nblk],
                        &all.scores[t * HEADS * all.nblk..(t + 1) * HEADS * all.nblk],
                    );
            }
            println!(
                "select {name} rows: each of {counts:?} in one launch = its one-row launch bit for \
                 bit {alone} {}",
                verdict(alone)
            );
            ok &= alone;
        }
        {
            // A mix with the refusals: counts 0 and past the cache.
            let counts = [0u32, 2049, u32::try_from(CTX + 1)?, 5000, 129, 32768];
            let iq = iq_rows(counts.len(), 22);
            let (out, _) = run_chain(&rig, &iq, &counts, &ik, None)?;
            ok &= check_select("refusals", &rig, &out, &iq, &counts, &ikf);
        }

        // 3. The tie rule: every full block holds the same 128 keys.
        let tie_bits: Vec<u16> = (0..CTX)
            .flat_map(|p| ik_bits[(p % BLOCK) * DIM..][..DIM].iter().copied())
            .collect();
        let tie_f = widen(&tie_bits);
        let tie = DeviceBuffer::from_host(stream, &tie_bits)?;
        {
            let counts = [32768u32, 16000, 8193];
            let iq = iq_rows(counts.len(), 31);
            let (out, _) = run_chain(&rig, &iq, &counts, &tie, None)?;
            let mut r = check_select("ties", &rig, &out, &iq, &counts, &tie_f);
            let k = rig.shape.top_k();
            for (t, &c) in counts.iter().enumerate() {
                let own = (c as usize).div_ceil(BLOCK) as u32 - 1;
                let mut want: Vec<u32> = (0..15).collect();
                want.push(own);
                for h in 0..HEADS {
                    r &= out.ids[(t * HEADS + h) * k..][..k] == want[..];
                }
            }
            println!(
                "select ties: the 15 lowest blocks and the own block at counts {counts:?} {}",
                verdict(r)
            );
            ok &= r;
        }

        // 4. The forced block: the own block of count 32768 holds zero keys, so it scores 0 and
        // every other block's largest score is above it.
        {
            let mut forced = ik_bits.clone();
            forced[(CTX - BLOCK) * DIM..].fill(0);
            let forced_f = widen(&forced);
            let forced_d = DeviceBuffer::from_host(stream, &forced)?;
            let counts = [32768u32];
            let iq = iq1(41);
            let twin = pooled_twin(&iq, &forced_f, 32768);
            let lowest = (0..HEADS).all(|h| {
                twin[h][255] == order_key(0.0)
                    && twin[h][..255].iter().all(|&key| key > order_key(0.0))
            });
            let (out, _) = run_chain(&rig, &iq, &counts, &forced_d, None)?;
            let mut r = check_select("forced", &rig, &out, &iq, &counts, &forced_f) && lowest;
            let k = rig.shape.top_k();
            for h in 0..HEADS {
                r &= out.ids[h * k..][..k].last() == Some(&255);
            }
            println!(
                "select forced: the own block scores lowest of 256 ({lowest}) and is kept {}",
                verdict(r)
            );
            ok &= r;
        }

        // 5. A NaN key: pool_select raised, the list still defined.
        {
            let mut bad = ik_bits;
            bad[(3 * BLOCK + 5) * DIM + 9] = NAN16;
            let bad_d = DeviceBuffer::from_host(stream, &bad)?;
            let counts = [32768u32];
            let iq = iq1(51);
            let (out, _) = run_chain(&rig, &iq, &counts, &bad_d, None)?;
            let k = rig.shape.top_k();
            let want = Some(Fault::at(u32::try_from(LAYER)?, FaultSite::PoolSelect));
            let mut defined = true;
            for h in 0..HEADS {
                let l = &out.ids[h * k..][..k];
                defined &=
                    l.windows(2).all(|w| w[0] < w[1]) && l.iter().all(|&b| (b as usize) < 256);
                defined &= l.last() == Some(&255);
            }
            defined &= out.n_sel[0] as usize == rig.shape.rule().key_count(32768);
            let r = out.fault == want && defined;
            println!(
                "select NaN key: block 3 position 5, layer {LAYER}: word {:?} (want {want:?}), lists \
                 distinct, ascending, in range, own block last, length the rule's {defined} {}",
                out.fault,
                verdict(r)
            );
            ok &= r;
        }

        // 6. The selected flash.
        let c = cache(stream, HEAD, HEADS, CTX, CTX, 600)?;
        {
            let same = attn_scale().to_bits() == scale().to_bits();
            println!(
                "flash scale: attn_scale() {:e} = 1/sqrt(128) as the reference rounds it {:e} bit for \
                 bit {same} {}",
                attn_scale(),
                scale(),
                verdict(same)
            );
            ok &= same;
        }
        {
            // 6a. Every block kept: the dense group-of-16 pass's bits.
            let counts = NONSEL.to_vec();
            let m = counts.len();
            let iq = iq_rows(m, 61);
            let q = q_rows(m, 62);
            let (_, y) = run_chain(&rig, &iq, &counts, &ik, Some((&q, &c.kc, &c.vc)))?;
            let qd = DeviceBuffer::from_host(stream, &q)?;
            let nk = DeviceBuffer::from_host(stream, &counts)?;
            let mut pv = DeviceBuffer::<f32>::zeroed(stream, partials_v_len(m, N_HEAD))?;
            let mut pms = DeviceBuffer::<f32>::zeroed(stream, partials_ms_len(m, N_HEAD))?;
            let mut yd = DeviceBuffer::<f32>::zeroed(stream, m * N_HEAD * DIM)?;
            rig.flash.enqueue_pass::<GROUP_16>(
                stream,
                GqaArgs {
                    q: &qd,
                    kc: &c.kc,
                    vc: &c.vc,
                    n_keys: &nk,
                    scale: scale(),
                    n_kv: HEADS,
                    ctx: CTX,
                    m,
                    part_v: &mut pv,
                    part_ms: &mut pms,
                    fault: gpu.unlabelled_sink(),
                    y: &mut yd,
                },
                true,
            )?;
            stream.synchronize()?;
            let dense = yd.to_host_vec(stream)?;
            let same = bits_equal(&y, &dense);
            println!(
                "flash every block kept counts={counts:?}: selected = dense group-16 pass bit for \
                 bit {same} {}",
                verdict(same)
            );
            ok &= same;
        }
        {
            // 6b. Past 16 blocks: the band, the rerun, NaN in unlisted rows, rows = one-row launches.
            let counts: Vec<u32> = vec![2049, 4096, 8193, 32768, 129, 2048, 16000, 5000];
            let m = counts.len();
            let iq = iq_rows(m, 71);
            let q = q_rows(m, 72);
            let k = rig.shape.top_k();
            let (out, y) = run_chain(&rig, &iq, &counts, &ik, Some((&q, &c.kc, &c.vc)))?;
            let (_, y2) = run_chain(&rig, &iq, &counts, &ik, Some((&q, &c.kc, &c.vc)))?;
            // The planes with every cache row no (row, key head) lists NaN. A length other than the
            // rule's is a failed check, not a list to walk.
            let mut named = vec![false; HEADS * CTX];
            let mut lengths = true;
            for (t, &cnt) in counts.iter().enumerate() {
                let n = out.n_sel[t] as usize;
                lengths &= n == rig.shape.rule().key_count(cnt as usize);
                if !lengths {
                    continue;
                }
                for kh in 0..HEADS {
                    for p in listed_positions(&out.ids[(t * HEADS + kh) * k..][..k], n) {
                        named[kh * CTX + p] = true;
                    }
                }
            }
            if !lengths {
                println!(
                    "flash selected counts={counts:?} lengths={:?}: not the rule's FAIL",
                    out.n_sel
                );
                return Err(checks_failed());
            }
            let nanify = |b: &[u16]| -> Vec<u16> {
                b.iter()
                    .enumerate()
                    .map(|(i, &v)| if named[i / HEAD] { v } else { NAN16 })
                    .collect()
            };
            let kn = DeviceBuffer::from_host(stream, &nanify(&c.kb))?;
            let vn = DeviceBuffer::from_host(stream, &nanify(&c.vb))?;
            let (_, yn) = run_chain(&rig, &iq, &counts, &ik, Some((&q, &kn, &vn)))?;
            let (band, worst) = judge(&q, &out, k, &c.host, &y);
            let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &yn));
            let w = N_HEAD * DIM;
            let mut alone = true;
            for t in 0..m {
                let (_, yo) = run_chain(
                    &rig,
                    &iq[t * HEADS * DIM..(t + 1) * HEADS * DIM],
                    &counts[t..=t],
                    &ik,
                    Some((&q[t * w..(t + 1) * w], &c.kc, &c.vc)),
                )?;
                alone &= bits_equal(&yo, &y[t * w..(t + 1) * w]);
            }
            let pass = band && rerun && nan_same && alone && out.fault.is_none();
            println!(
                "flash selected counts={counts:?} lengths={:?}: measured/bound {worst:.3e} \
                 band={band} rerun={rerun} nan_in_unlisted_rows_same={nan_same} rows = one-row \
                 launches={alone} {}",
                out.n_sel,
                verdict(pass)
            );
            ok &= pass;
        }

        // 7. The flash's faults, on the lists of a real selection.
        {
            let counts: Vec<u32> = vec![4096, 8193, 32768];
            let m = counts.len();
            let iq = iq_rows(m, 81);
            let q = q_rows(m, 82);
            let k = rig.shape.top_k();
            let (out, clean) = run_chain(&rig, &iq, &counts, &ik, Some((&q, &c.kc, &c.vc)))?;
            let w = N_HEAD * DIM;
            let direct = |ids: &[u32],
                          n_sel: &[u32]|
             -> Result<(Vec<f32>, Option<Fault>), GateError> {
                let qd = DeviceBuffer::from_host(stream, &q)?;
                let idd = DeviceBuffer::from_host(stream, ids)?;
                let nk = DeviceBuffer::from_host(stream, n_sel)?;
                let width = rig.shape.rule().width();
                let mut pv =
                    DeviceBuffer::<f32>::zeroed(stream, listed_partials_v_len(m, N_HEAD, width))?;
                let mut pms =
                    DeviceBuffer::<f32>::zeroed(stream, listed_partials_ms_len(m, N_HEAD, width))?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, m * w)?;
                rig.flash.enqueue_pass_p16_blocks(
                    stream,
                    GqaBlocksArgs {
                        q: &qd,
                        kc: &c.kc,
                        vc: &c.vc,
                        ids: &idd,
                        n_keys: &nk,
                        blocks: k,
                        scale: scale(),
                        n_kv: HEADS,
                        ctx: CTX,
                        m,
                        part_v: &mut pv,
                        part_ms: &mut pms,
                        fault: gpu.layer_sink(LAYER)?,
                        y: &mut y,
                    },
                    N_HEAD,
                )?;
                stream.synchronize()?;
                Ok((y.to_host_vec(stream)?, gpu.take_fault()?))
            };
            // A block past the cache in row 1's key head 2.
            let mut ids = out.ids.clone();
            ids[(HEADS + 2) * k] = u32::try_from(CTX / BLOCK)?;
            let (yb, raised) = direct(&ids, &out.n_sel)?;
            let want = Some(Fault::at(u32::try_from(LAYER)?, FaultSite::PoolSelect));
            let (mut nan, mut others) = (true, true);
            for t in 0..m {
                for h in 0..N_HEAD {
                    let (a, b) = (
                        &clean[t * w + h * DIM..][..DIM],
                        &yb[t * w + h * DIM..][..DIM],
                    );
                    if t == 1 && h / (N_HEAD / HEADS) == 2 {
                        nan &= b.iter().all(|v| v.is_nan());
                    } else {
                        others &= bits_equal(a, b);
                    }
                }
            }
            let r1 = raised == want && nan && others;
            println!(
                "flash fault: block {} in row 1's key head 2: word {raised:?} (want {want:?}), \
                 those heads NaN {nan}, every other head bit-identical {others} {}",
                CTX / BLOCK,
                verdict(r1)
            );
            // A length of zero in row 2.
            let mut n_sel = out.n_sel.clone();
            n_sel[2] = 0;
            let (yz, raised) = direct(&out.ids, &n_sel)?;
            let want = Some(Fault::at(u32::try_from(LAYER)?, FaultSite::KeyCount));
            let nan = yz[2 * w..].iter().all(|v| v.is_nan());
            let others = bits_equal(&yz[..2 * w], &clean[..2 * w]);
            let r2 = raised == want && nan && others;
            println!(
                "flash fault: length 0 in row 2: word {raised:?} (want {want:?}), the row NaN {nan}, \
                 the other rows bit-identical {others} {}",
                verdict(r2)
            );
            ok &= r1 && r2;
        }

        println!("gate_minimaxm3_msa: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
