//! GPU gate for Inkling 2's visit pipeline — the block-attention residual's normed mix
//! (`attn_res::enqueue_normed`) and its step form (`attn_res_step`), the attention input's norms
//! and projections in one launch (`norm_proj`), the looped core's message controller (`loop_control::enqueue`, `enqueue_attn`)
//! and the residual short convolutions (`linear::conv_res::enqueue`) — against host f64 rules of
//! llama.cpp PR #32695's `src/models/inkling2.cpp` (`mix` :212-225, `control` :228-237, `convolve`
//! :241-283) at the files' shapes (hidden 2,304, gate rank 64, taps 4, the k/v convs 1,024 channels
//! of eight heads, the post convs 2,304), and against the fixture's own node dumps (refset family
//! `fx-lcpp-inkling2`, set `a`) through the whole forced chain every stage of the pipeline feeds:
//! the mixes and controllers of the visit's own tapped inputs against the vendor's `norm-<slot>`,
//! `loop_q_inp`, `loop_ffn_inp` and `result_norm` rows.
//!
//! Fixture-only (the recipe's `tools/ref/fixture-only.sh`): the stages this gate replays need the
//! fixture's taps and weights; the real file's are the e2e gate's, which lands with the body. The
//! conv's own inputs (the `wk`/`wv` projections' rows) are not tapped, so the conv's clauses are
//! the seeded ones — its contract is the rule, not a vendor row.
//!
//! The seeded clauses (nothing read from a model file):
//!
//! 1. `mix_seeded`: the normed mix at every mix point's shape — one query over two vectors (a
//!    windowed attention's), three queries over three (a windowless attention's), four over four
//!    (the two message tables staged as one launch), one over nine and over sixteen (the head's
//!    many-state mix) — each within [`MIX_BAND`] of the f64 rule at `m` rows, a rerun bit for bit,
//!    and a pass of rows each row's one-row launch's bits.
//! 2. `mix_separation`: the rule summing the RAW vectors (Kimi's) and the rule with the `1/√2304`
//!    scale dropped each lie past [`SEP`] bands of the card — the raw one at every mix point, the
//!    scale one at the many-state ones, where the softmax flattens enough to move the output.
//! 3. `mix_refused` / `mix_faults`: no query row, five query rows, seventeen vectors, a stride
//!    under a pass's rows, a zero epsilon, a NaN scale and a short output each refused by name; a
//!    NaN bank value raises `f32_product` and every query row is NaN.
//! 4. `control_seeded`: the controller at (2,304, 64) one row and eight, at the entry's bounds
//!    (4,608 wide, a narrow 512×16) within [`CONTROL_BAND`] of the f64 rule.
//! 5. `control_separation`: the rule with gain and forget swapped, the rule over the un-normed
//!    row, and the rule with no tanh each lie past [`SEP`] bands.
//! 6. `control_refused` / `control_faults`: no row, width or rank, a width over the staging bound,
//!    a rank over the bound, a negative epsilon each refused by name (a short weight is the shared
//!    length check); a NaN message raises `f32_product` and the row is NaN.
//! 7. `conv_seeded`: the residual conv at the k path's 1,024 channels (static, and dynamic at
//!    eight heads — the windowless layers' own taps) and at the attention-post conv's 2,304
//!    channels inside a 6,656-wide ring at its real offset, over a seeded walk that starts at the
//!    sequence head, wraps the ring and ends mid-call: each token within [`CONV_BAND`] of the f64
//!    rule, and one call of eight tokens the eight one-token calls' bits.
//! 8. `conv_separation`: the rule with the dynamic taps' order reversed lies past [`SEP`] bands of
//!    the dynamic conv and inside the band of the static ones (the static order is its own); the
//!    rule with the static taps reversed separates the static conv.
//! 9. `conv_refused` / `conv_faults`: no channel or token, seventeen taps, three heads over 1,024
//!    channels, a channel range outside the ring's rows and a short ring each refused by name; a
//!    NaN input raises `linear_conv`, its own value of the token NaN and every other value of
//!    the token finite (the conv is depthwise: one channel's NaN reaches no other channel).
//!
//! 9b. `conv_clip_is_the_pair`: the decode step's one launch for the k and v convs and the
//!     epilogue (`qk_clip::enqueue_conv`) against the three launches it replaces (the k and v
//!     `conv_res` calls, then `qk_clip::enqueue`), per token bit for bit — the query, relative,
//!     key and value rows, the f16 planes and the conv ring — over a walk that starts at the
//!     sequence head and wraps the ring, static and dynamic convs, a NaN input and an append
//!     past the planes' end alike (the same fault word).
//! 9c. `conv_tail_is_the_launches`: the output projection with its post conv and the join's add in
//!     one launch (`conv_tail::enqueue_gemv_conv`, a Q8_0 projection at the files' 2,304 rows of
//!     3,072 values) against `q8_0_gemv`, `conv_res` and the add it replaces, and the card
//!     slots' eight-slot sum with the feed-forward's post conv and the stream's add in one launch
//!     (`conv_tail::enqueue_card_acc_conv`) against `ds41_ffn_card_acc_8`, `conv_res` and the add —
//!     per token bit for bit (the conv's output, the join, the sum) and the ring at the end, over a
//!     walk that starts at the sequence head and wraps the ring; the projection with and without
//!     the join (a block's first visit joins nothing) and with a NaN activation, the sum over
//!     slots on the card, with host slots and with a place past the card (the same fault word).
//! 10. `mix_step_is_the_mix`: the step mix (`attn_res_step`) at one row of every mix point's
//!     shape — alone, run twice (the tickets back at zero between calls), and paired as the
//!     second mix of a launch beside the four-query message mix — gives `enqueue_normed`'s bits;
//!     a NaN bank value raises `f32_product` and every query row is NaN.
//! 11. `norm_proj_is_the_sites`: the fused input norms and projections (`norm_proj`) at the
//!     files' site shapes, with three inputs and with one, each with and without the conv taps,
//!     and at ragged rows (a Q8_0 site's last block of eight, a site of one block of eight,
//!     F32 sites of one to three blocks), give `rms_norm` and each site's own gemv's bits
//!     (`q8_0_gemv`, `f32_gemv`), the taps of a launch without them unwritten; a NaN input value
//!     raises the norm's fault from the fused launch, every site of that input NaN and the other
//!     inputs' sites their bits.
//!
//! The oracle clauses (`fx_ref_inkling2_a`, its positions' count read from the set, the fixture's
//! 25-visit schedule — the stem 0-5, the core's three passes 6-11, 12-17, 18-23, the coda 24; the
//! block boundaries 0, 6, 12, 18, 24; the windowless slots 5, 11, 17, 23; every input a tapped
//! node and every weight an f32 tensor of the fixture file, so the chain carries no quantization):
//!
//! 12. `mix_pass0`: the windowed attention mix of slot 8 (banks the embedding norm and `l_out-5`,
//!     the stream `l_out-7`) through its input norm within [`ORACLE_BAND`] of `norm-8` occurrence
//!     0; the windowless slot 11's three-query mix through the three input norms of `norm-11`
//!     occurrences 0 to 2 and its feed-forward mix through occurrence 3; the head's mix over the
//!     three blocks the walk ends with (the prelude's, pass 2's boundary push and the coda's —
//!     every pass resets the blocks to the prelude's, so pass 0's pushes are gone) and the
//!     stream, against the last position's `result_norm` row.
//! 13. `mix_pass1`: slot 13 (a pass-1 core visit; the fixture's message bank is empty — its pass 0
//!     pushed one block — and its `bank_last` is `l_out-11`): the two message tables' rows staged
//!     as one four-query launch over `l_out-11`, the attention mix of the prelude's two banks
//!     and `l_out-12`, the feed-forward mix of `ffn_inp-13`, each through its controller within
//!     [`ORACLE_BAND`] of `loop_q_inp-13` and `loop_ffn_inp-13`, and the controllers' outputs
//!     through their norms of `norm-13`: a looped windowed visit norms its three controller
//!     outputs — the dump numbers them in its own eval order, the k, v and q inputs as
//!     occurrences 0 to 2, each with the attention norm's gain — and its feed-forward mix
//!     through the ffn norm (occurrence 3).
//!
//! [derived] The bands: the card's f32 sums against the host's f64 — a 2,304-value dot's error
//! `6e-8·√2304 ≈ 3e-6` of its magnitude, the softmax and the norms a few ulps more, the
//! controller's two chained dots and the tanh under ten of those — so [`MIX_BAND`] 1e-5, and
//! [`CONTROL_BAND`], [`CONV_BAND`] and [`ORACLE_BAND`] 1e-4 are ten times the derived sum (the
//! oracle chain adds the vendor's own f32 rounding, the same size); [`SEP`] 10 bands. The first
//! run on the box re-pins them against the measured maxima.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_step", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::attn_res::{AttnResKernels, AttnResNormedArgs, MAX_NQ, MAX_VECS};
    use bloomery_gpu::attn_res_step::{AttnResStepKernels, MixStepScratch};
    use bloomery_gpu::card_slots::{CardAcc, CardSlotKernels};
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::linear::conv_res::{ConvResArgs, ConvResKernels};
    use bloomery_gpu::linear::conv_tail::{
        CardAccConvArgs, ConvTailKernels, GemvConvArgs, TailConvArgs,
    };
    use bloomery_gpu::loop_control::{
        LoopControlArgs, LoopControlKernels, MAX_MODULES, MAX_N, MAX_RANK, OUT_COLS, scratch_len,
    };
    use bloomery_gpu::norm_proj::{F32Site, NormProjArgs, NormProjKernels, Q8Site};
    use bloomery_gpu::qk_clip::{QkClipArgs, QkClipConv, QkClipKernels};
    use bloomery_gpu::{DeviceTensor, Fault, Gpu, GpuError};
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{
        GateError, GateRng, RefManifest, bits_equal, checks_failed, ref_tensor_logical_in, verdict,
    };
    use chain::linear::Ring;
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::GgmlType;
    use refset::arch::inkling2::fixture::{self, A};

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_inkling2_step";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-inkling2-step";

    /// The files' hidden width and the looped core's gate rank.
    const N: usize = 2304;
    const RANK: usize = 64;
    /// The fixture's core start, in its own layer coordinates (the looped fixture's core is its
    /// layers [6, 12)).
    const CORE_START: usize = 6;
    /// The convolutions' taps (`attention.shortconv_l_cache`) and the visit's whole conv ring
    /// width: k and v at 0 and 1,024, the attention output's post conv at 2,048, the
    /// feed-forward's at 4,352.
    const TAPS: usize = 4;
    const RING_CH: usize = 6656;
    const K_BASE: usize = 0;
    const POST_BASE: usize = 2048;
    /// The k and v paths' channels and their heads.
    const K_CH: usize = 1024;
    const KV_HEADS: usize = 8;
    /// The post convolutions' channels (the hidden width).
    const POST_CH: usize = 2304;
    /// The one epsilon every inkling2 norm and mix takes (the file's `rms_eps` and the mixes'
    /// fixed one agree).
    const EPS: f32 = 1e-6;
    /// The mixes' score scale, the reference's `1/sqrtf(n_embd)`: 2,304 is 48², so the square
    /// root is exact and the quotient is the correctly rounded value.
    const SCALE: f32 = 1.0 / 48.0;
    /// The convolutions' scales: `1/sqrtf(taps + 1)` static, `1/sqrtf(2·taps + 1)` dynamic
    /// (`inkling2.cpp:282`), each computed as the reference computes it.
    fn scale_static() -> f32 {
        1.0 / 5.0f32.sqrt()
    }

    fn scale_dynamic() -> f32 {
        1.0 / 9.0f32.sqrt()
    }
    /// [derived] The seeded mix against its f64 rule: ten times the derived 1e-6.
    const MIX_BAND: f64 = 1.0e-5;
    /// [derived] The controller and the convs against their f64 rules, and every oracle chain
    /// against the vendor's own rows: ten times the derived sum of the two engines' roundings.
    const CONTROL_BAND: f64 = 1.0e-4;
    const CONV_BAND: f64 = 1.0e-4;
    const ORACLE_BAND: f64 = 1.0e-4;
    /// The bands a mutated rule must lie past.
    const SEP: f64 = 10.0;

    /// One row's f64 rule values: a query row's mix output, a controller row, a conv row.
    type RuleRows = Vec<Vec<f64>>;

    /// `‖a − b‖₂ / ‖b‖₂` in f64.
    fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - y).powi(2);
            den += y * y;
        }
        if a.len() != b.len() || !num.is_finite() {
            return f64::INFINITY;
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// The f64 maximum of `f` over every row of `y` (`rows` of `width`).
    fn worst(y: &[f32], rows: usize, width: usize, f: impl Fn(usize, &[f32]) -> f64) -> f64 {
        (0..rows)
            .map(|t| f(t, &y[t * width..(t + 1) * width]))
            .fold(0.0f64, f64::max)
    }

    fn fill(r: &mut GateRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| r.uni(lo, hi)).collect()
    }

    fn f64s(v: &[f32]) -> Vec<f64> {
        v.iter().map(|x| f64::from(*x)).collect()
    }

    fn f32s(v: &[f64]) -> Vec<f32> {
        v.iter().map(|x| *x as f32).collect()
    }

    /// A gainless RMS scale of a row in f64 (the mean in f64).
    fn rms_of(v: &[f64]) -> f64 {
        let ms = v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64;
        1.0 / (ms + f64::from(EPS)).sqrt()
    }

    // ------------------------------------------------------------------ the f64 rules

    /// What a seeded mix's rule leaves out.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum MixDrop {
        None,
        /// The output over the raw vectors (Kimi's rule).
        Raw,
        /// The score scale at 1.
        NoScale,
    }

    /// The mix's rule of `inkling2.cpp:212-225` in f64 over one row's `banks` and `cur`, `nq`
    /// query rows: `normed_i` per vector (gainless, at a fixed epsilon), the scores
    /// `rstd · dot · scale` softmaxed over the vectors, the output over the normed (or raw)
    /// vectors.
    fn mix_rule(
        banks: &[Vec<f64>],
        cur: &[f64],
        queries: &[Vec<f64>],
        scale: f64,
        drop: MixDrop,
    ) -> Vec<Vec<f64>> {
        let vecs: Vec<&[f64]> = banks
            .iter()
            .map(Vec::as_slice)
            .chain(std::iter::once(cur))
            .collect();
        // The scores read the normed vectors (the reference's materialized states): one RMS
        // scale a vector, inside the normed values.
        let scales: Vec<f64> = vecs.iter().map(|v| rms_of(v)).collect();
        let normed: Vec<Vec<f64>> = vecs
            .iter()
            .zip(&scales)
            .map(|(v, s)| v.iter().map(|x| x * s).collect())
            .collect();
        // The sum's vectors: the normed ones (the reference's materialized states) or the raw
        // ones (Kimi's rule).
        let summed: Vec<Vec<f64>> = if drop == MixDrop::Raw {
            vecs.iter().map(|v| v.to_vec()).collect()
        } else {
            normed.clone()
        };
        queries
            .iter()
            .map(|q| {
                let s: Vec<f64> = normed
                    .iter()
                    .map(|v| {
                        let dot = v.iter().zip(q).map(|(a, b)| a * b).sum::<f64>();
                        dot * match drop {
                            MixDrop::NoScale => 1.0,
                            _ => scale,
                        }
                    })
                    .collect();
                let mx = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let z: f64 = e.iter().sum();
                (0..cur.len())
                    .map(|j| {
                        e.iter()
                            .zip(&summed)
                            .map(|(w, v)| w / z * v[j])
                            .sum::<f64>()
                    })
                    .collect()
            })
            .collect()
    }

    /// What a seeded controller's rule leaves out.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ControlDrop {
        None,
        /// Gain and forget swapped.
        Swapped,
        /// `h` over the raw row, no norm.
        Unnormed,
        /// No tanh on the gates.
        NoTanh,
    }

    /// The controller's rule of `inkling2.cpp:228-237` in f64 over one row: the two halves normed
    /// on their own, the down projection, the gain and forget rows, the tanh and the combine.
    #[allow(
        clippy::too_many_arguments,
        reason = "the rule's whole state, one argument a tensor"
    )]
    fn control_rule(
        cur: &[f64],
        msg: &[f64],
        down: &[f64],
        gain: &[f64],
        forget: &[f64],
        n: usize,
        rank: usize,
        drop: ControlDrop,
    ) -> Vec<f64> {
        let (rc, rm) = if drop == ControlDrop::Unnormed {
            (1.0, 1.0)
        } else {
            (rms_of(cur), rms_of(msg))
        };
        let h: Vec<f64> = (0..rank)
            .map(|r| {
                (0..2 * n)
                    .map(|k| {
                        let hv = if k < n { cur[k] * rc } else { msg[k - n] * rm };
                        down[r * 2 * n + k] * hv
                    })
                    .sum()
            })
            .collect();
        (0..n)
            .map(|j| {
                let (gw, fw) = if drop == ControlDrop::Swapped {
                    (
                        &forget[j * rank..(j + 1) * rank],
                        &gain[j * rank..(j + 1) * rank],
                    )
                } else {
                    (
                        &gain[j * rank..(j + 1) * rank],
                        &forget[j * rank..(j + 1) * rank],
                    )
                };
                let g: f64 = (0..rank).map(|d| gw[d] * h[d]).sum();
                let f: f64 = (0..rank).map(|d| fw[d] * h[d]).sum();
                let (g, f) = if drop == ControlDrop::NoTanh {
                    (g, f)
                } else {
                    (g.tanh(), f.tanh())
                };
                (1.0 - f) * cur[j] + g * msg[j]
            })
            .collect()
    }

    /// The residual conv's rule of `inkling2.cpp:241-283` in f64 at position `p` of a history of
    /// inputs (`hist[i]` position `i`'s row, zero before the sequence head), the static taps
    /// oldest first, the dynamic ones current first (`reversed` flips their order — the mutant's
    /// rule).
    #[allow(
        clippy::too_many_arguments,
        reason = "the rule's whole state, one argument a tensor"
    )]
    fn conv_rule(
        hist: &[Vec<f64>],
        w: &[f64],
        dynamic: &[Vec<f64>],
        p: usize,
        d: usize,
        heads: usize,
        scale: f64,
        reversed: bool,
    ) -> Vec<f64> {
        let per_head = d / heads.max(1);
        let at = |back: usize, c: usize| -> f64 { if p < back { 0.0 } else { hist[p - back][c] } };
        (0..d)
            .map(|c| {
                let mut acc = w[c * TAPS] * at(TAPS - 1, c);
                for k in 1..TAPS {
                    acc += w[c * TAPS + k] * at(TAPS - 1 - k, c);
                }
                if heads > 0 {
                    let head = c / per_head;
                    for lag in 0..TAPS {
                        let l = if reversed { TAPS - 1 - lag } else { lag };
                        acc += dynamic[p][head * TAPS + l] * at(lag, c);
                    }
                }
                (hist[p][c] + acc) * scale
            })
            .collect()
    }

    // ------------------------------------------------------------------ the card

    /// The card's normed mix of one case: `m` rows over the seeded banks, `cur` and `nq` queries,
    /// beside the host's f64 rule of the same values.
    struct MixCase {
        banks: Vec<Vec<f64>>,
        cur: Vec<Vec<f64>>,
        queries: Vec<Vec<f64>>,
    }

    impl MixCase {
        /// Seeded rows: the vectors' scales spread `0.6·1.25ⁱ` (diverse RMS scales, so the normed
        /// and raw sums differ), the queries small enough that the scaled scores span a softmax.
        /// Every bank carries the pass's whole `m` rows (`m · N` values, the launch's layout at
        /// `stride = m · N`), like the stream.
        fn new(seed: u64, n_banks: usize, nq: usize, m: usize) -> MixCase {
            let mut r = GateRng(seed);
            let row = |r: &mut GateRng, s: f32| -> Vec<f32> {
                (0..N).map(|_| r.uni(-1.0, 1.0) * s).collect()
            };
            let banks = (0..n_banks)
                .map(|i| {
                    let s = 0.6f32 * 1.25f32.powi(i as i32);
                    (0..m)
                        .flat_map(|_| f64s(&row(&mut r, s)))
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<_>>();
            let cur = (0..m).map(|_| f64s(&row(&mut r, 1.7))).collect::<Vec<_>>();
            let queries = (0..nq)
                .map(|_| f64s(&row(&mut r, 0.05)))
                .collect::<Vec<_>>();
            MixCase {
                banks,
                cur,
                queries,
            }
        }

        /// The case's banks (one value when there is none), stream and query rows on the card.
        fn upload(&self, gpu: &Gpu) -> Result<[DeviceBuffer<f32>; 3], GateError> {
            let s = gpu.stream();
            let mut bank_flat: Vec<f32> = self.banks.iter().flatten().map(|x| *x as f32).collect();
            if bank_flat.is_empty() {
                bank_flat.push(0.0);
            }
            Ok([
                DeviceBuffer::from_host(s, &bank_flat)?,
                DeviceBuffer::from_host(s, &f32s(&self.cur.concat()))?,
                DeviceBuffer::from_host(s, &f32s(&self.queries.concat()))?,
            ])
        }

        /// The case's one row as the step mix's mix (its buffers from [`MixCase::upload`]).
        fn step_args<'a>(&self, gpu: &Gpu, b: &'a [DeviceBuffer<f32>; 3]) -> AttnResNormedArgs<'a> {
            AttnResNormedArgs {
                banks: &b[0],
                n_banks: self.banks.len(),
                stride: N,
                cur: &b[1],
                queries: &b[2],
                nq: self.queries.len(),
                scale: SCALE,
                eps: EPS,
                n: N,
                m: 1,
                fault: gpu.unlabelled_sink(),
            }
        }

        /// The card's output `[m][nq · N]` and the f64 rule's rows (`m` of `nq` rows of `N`),
        /// every row's rule computed on its own rows.
        fn run(
            &self,
            gpu: &Gpu,
            k: &AttnResKernels,
            drop: MixDrop,
        ) -> Result<(Vec<f32>, Vec<RuleRows>), GateError> {
            let s = gpu.stream();
            let (m, nq) = (self.cur.len(), self.queries.len());
            let stride = m * N;
            let mut bank_flat = Vec::with_capacity(self.banks.len() * stride);
            for b in &self.banks {
                bank_flat.extend(b.iter().map(|x| *x as f32));
            }
            // An empty bank set still uploads one value: the launch reads no bank row then, and a
            // zero-length allocation is not one the card makes.
            if bank_flat.is_empty() {
                bank_flat.push(0.0);
            }
            let banks = DeviceBuffer::from_host(s, &bank_flat)?;
            let cur = DeviceBuffer::from_host(s, &f32s(&self.cur.concat()))?;
            let queries = DeviceBuffer::from_host(s, &f32s(&self.queries.concat()))?;
            let mut y = DeviceBuffer::zeroed(s, m * nq * N)?;
            k.enqueue_normed(
                s,
                AttnResNormedArgs {
                    banks: &banks,
                    n_banks: self.banks.len(),
                    stride,
                    cur: &cur,
                    queries: &queries,
                    nq,
                    scale: SCALE,
                    eps: EPS,
                    n: N,
                    m,
                    fault: gpu.unlabelled_sink(),
                },
                &mut y,
            )?;
            s.synchronize()?;
            let card = y.to_host_vec(s)?;
            let scale = match drop {
                MixDrop::NoScale => 1.0,
                _ => f64::from(SCALE),
            };
            let rule = (0..m)
                .map(|t| {
                    let banks: Vec<Vec<f64>> = self
                        .banks
                        .iter()
                        .map(|b| b[t * N..(t + 1) * N].to_vec())
                        .collect();
                    mix_rule(&banks, &self.cur[t], &self.queries, scale, drop)
                })
                .collect();
            Ok((card, rule))
        }
    }

    /// The card's controller of one seeded case, `m` rows, beside the f64 rule.
    struct ControlCase {
        cur: Vec<Vec<f64>>,
        msg: Vec<Vec<f64>>,
        down: Vec<f64>,
        gain: Vec<f64>,
        forget: Vec<f64>,
        n: usize,
        rank: usize,
    }

    impl ControlCase {
        fn new(seed: u64, n: usize, rank: usize, m: usize) -> ControlCase {
            let mut r = GateRng(seed);
            ControlCase {
                cur: (0..m).map(|_| f64s(&fill(&mut r, n, -1.0, 1.0))).collect(),
                msg: (0..m).map(|_| f64s(&fill(&mut r, n, -1.0, 1.0))).collect(),
                down: f64s(&fill(&mut r, rank * 2 * n, -0.05, 0.05)),
                gain: f64s(&fill(&mut r, n * rank, -0.05, 0.05)),
                forget: f64s(&fill(&mut r, n * rank, -0.05, 0.05)),
                n,
                rank,
            }
        }

        fn run(
            &self,
            gpu: &Gpu,
            k: &LoopControlKernels,
            drop: ControlDrop,
        ) -> Result<(Vec<f32>, RuleRows), GateError> {
            let s = gpu.stream();
            let m = self.cur.len();
            let cur = DeviceBuffer::from_host(s, &f32s(&self.cur.concat()))?;
            let msg = DeviceBuffer::from_host(s, &f32s(&self.msg.concat()))?;
            let down = DeviceBuffer::from_host(s, &f32s(&self.down))?;
            let gain = DeviceBuffer::from_host(s, &f32s(&self.gain))?;
            let forget = DeviceBuffer::from_host(s, &f32s(&self.forget))?;
            let mut y = DeviceBuffer::zeroed(s, m * self.n)?;
            let mut h = DeviceBuffer::zeroed(s, scratch_len(1, m, self.rank))?;
            k.enqueue(
                s,
                LoopControlArgs {
                    cur: &cur,
                    msg: &msg,
                    down: &down,
                    gain: &gain,
                    forget: &forget,
                    n: self.n,
                    rank: self.rank,
                    m,
                    eps: EPS,
                    fault: gpu.unlabelled_sink(),
                },
                &mut h,
                &mut y,
            )?;
            s.synchronize()?;
            let card = y.to_host_vec(s)?;
            let rule = (0..m)
                .map(|t| {
                    control_rule(
                        &self.cur[t],
                        &self.msg[t],
                        &self.down,
                        &self.gain,
                        &self.forget,
                        self.n,
                        self.rank,
                        drop,
                    )
                })
                .collect();
            Ok((card, rule))
        }

        fn upload(&self, gpu: &Gpu) -> Result<[DeviceBuffer<f32>; 5], GateError> {
            let s = gpu.stream();
            Ok([
                DeviceBuffer::from_host(s, &f32s(&self.cur.concat()))?,
                DeviceBuffer::from_host(s, &f32s(&self.msg.concat()))?,
                DeviceBuffer::from_host(s, &f32s(&self.down))?,
                DeviceBuffer::from_host(s, &f32s(&self.gain))?,
                DeviceBuffer::from_host(s, &f32s(&self.forget))?,
            ])
        }

        fn args<'a>(&self, gpu: &Gpu, b: &'a [DeviceBuffer<f32>; 5]) -> LoopControlArgs<'a> {
            LoopControlArgs {
                cur: &b[0],
                msg: &b[1],
                down: &b[2],
                gain: &b[3],
                forget: &b[4],
                n: self.n,
                rank: self.rank,
                m: self.cur.len(),
                eps: EPS,
                fault: gpu.unlabelled_sink(),
            }
        }
    }

    /// One module's rows from the split launches and from the one-block reference.
    type BitPair = (Vec<f32>, Vec<f32>);

    /// The split launches' output beside the one-fused-launch reference's, for `cases` (one
    /// case the single-module pair, three the attention launch), each module's own pair of
    /// output rows.
    fn control_pairs(
        gpu: &Gpu,
        k: &LoopControlKernels,
        cases: &[&ControlCase],
    ) -> Result<Vec<BitPair>, GateError> {
        let s = gpu.stream();
        let bufs = cases
            .iter()
            .map(|c| c.upload(gpu))
            .collect::<Result<Vec<_>, _>>()?;
        let (n, rank, m) = (cases[0].n, cases[0].rank, cases[0].cur.len());
        let mut old = Vec::new();
        for (c, b) in cases.iter().zip(&bufs) {
            let mut y = DeviceBuffer::zeroed(s, m * n)?;
            k.enqueue_single_block(s, c.args(gpu, b), &mut y)?;
            s.synchronize()?;
            old.push(y.to_host_vec(s)?);
        }
        let mut new = Vec::new();
        if cases.len() == 1 {
            let mut y = DeviceBuffer::zeroed(s, m * n)?;
            let mut h = DeviceBuffer::zeroed(s, scratch_len(1, m, rank))?;
            k.enqueue(s, cases[0].args(gpu, &bufs[0]), &mut h, &mut y)?;
            s.synchronize()?;
            new.push(y.to_host_vec(s)?);
        } else {
            assert_eq!(cases.len(), MAX_MODULES);
            let mut y0 = DeviceBuffer::zeroed(s, m * n)?;
            let mut y1 = DeviceBuffer::zeroed(s, m * n)?;
            let mut y2 = DeviceBuffer::zeroed(s, m * n)?;
            let mut h = DeviceBuffer::zeroed(s, scratch_len(MAX_MODULES, m, rank))?;
            k.enqueue_attn(
                s,
                [
                    cases[0].args(gpu, &bufs[0]),
                    cases[1].args(gpu, &bufs[1]),
                    cases[2].args(gpu, &bufs[2]),
                ],
                &mut h,
                [&mut y0, &mut y1, &mut y2],
            )?;
            s.synchronize()?;
            for y in [&y0, &y1, &y2] {
                new.push(y.to_host_vec(s)?);
            }
        }
        Ok(new.into_iter().zip(old).collect())
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The card's residual conv over a seeded walk: `hist` one row a position with the dynamic
    /// taps' rows beside them (empty a static conv), the ring walked one call a position.
    struct ConvWalk {
        hist: Vec<Vec<f64>>,
        dynamic: Vec<Vec<f64>>,
        w: Vec<f64>,
        d: usize,
        heads: usize,
        base: usize,
        ch: usize,
        scale: f32,
    }

    impl Clone for ConvWalk {
        fn clone(&self) -> ConvWalk {
            ConvWalk {
                hist: self.hist.clone(),
                dynamic: self.dynamic.clone(),
                w: self.w.clone(),
                d: self.d,
                heads: self.heads,
                base: self.base,
                ch: self.ch,
                scale: self.scale,
            }
        }
    }

    impl ConvWalk {
        fn new(seed: u64, d: usize, heads: usize, base: usize, ch: usize, len: usize) -> ConvWalk {
            let mut r = GateRng(seed);
            ConvWalk {
                hist: (0..len)
                    .map(|_| f64s(&fill(&mut r, d, -1.0, 1.0)))
                    .collect(),
                dynamic: (0..len)
                    .map(|_| f64s(&fill(&mut r, heads * TAPS, -0.5, 0.5)))
                    .collect(),
                w: f64s(&fill(&mut r, d * TAPS, -1.0, 1.0)),
                d,
                heads,
                base,
                ch,
                scale: if heads > 0 {
                    scale_dynamic()
                } else {
                    scale_static()
                },
            }
        }

        /// The walk's card outputs and the f64 rule's, one row a position, the calls of `calls`
        /// tokens each (`reversed` runs the rule only — the mutant's reference).
        fn run(
            &self,
            gpu: &Gpu,
            k: &ConvResKernels,
            calls: &[usize],
            reversed: bool,
        ) -> Result<(Vec<f32>, RuleRows), GateError> {
            let s = gpu.stream();
            let ring = Ring::new(TAPS);
            let rows = ring.rows();
            let w_dev = DeviceBuffer::from_host(s, &f32s(&self.w))?;
            // A static conv's dynamic buffer stays one value: the launch reads no tap row then,
            // and a zero-length allocation is not one the card makes.
            let none = DeviceBuffer::from_host(s, &[0.0f32])?;
            let mut state = DeviceBuffer::zeroed(s, rows * self.ch)?;
            let mut out = Vec::with_capacity(self.hist.len() * self.d);
            let mut at = 0usize;
            for &m in calls {
                let pos: Vec<u32> = (0..m).map(|t| (at + t) as u32).collect();
                let pos_dev = DeviceBuffer::from_host(s, &pos)?;
                let rows_dev = DeviceBuffer::from_host(s, &f32s(&self.hist[at..at + m].concat()))?;
                let rows_dyn = if self.heads > 0 {
                    Some(DeviceBuffer::from_host(
                        s,
                        &f32s(&self.dynamic[at..at + m].concat()),
                    )?)
                } else {
                    None
                };
                let mut y = DeviceBuffer::zeroed(s, m * self.d)?;
                k.enqueue(
                    s,
                    ConvResArgs {
                        ring,
                        d: self.d,
                        base: self.base,
                        ch: self.ch,
                        heads: self.heads,
                        m,
                        x: &rows_dev,
                        weights: &w_dev,
                        dynamic: rows_dyn.as_ref().unwrap_or(&none),
                        pos: &pos_dev,
                        fault: gpu.unlabelled_sink(),
                        scale: self.scale,
                        y: &mut y,
                        state: &mut state,
                    },
                )?;
                s.synchronize()?;
                out.extend(y.to_host_vec(s)?);
                at += m;
            }
            let rule = (0..self.hist.len())
                .map(|p| {
                    conv_rule(
                        &self.hist,
                        &self.w,
                        &self.dynamic,
                        p,
                        self.d,
                        self.heads,
                        f64::from(self.scale),
                        reversed,
                    )
                })
                .collect();
            Ok((out, rule))
        }
    }

    // ------------------------------------------------------------------ the seeded clauses

    /// Every mix point's shape: (banks, nq, rows).
    const MIX_CASES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (1, 3, 1),
        (2, 3, 8),
        (3, 4, 1),
        (8, 1, 1),
        (15, 1, 8),
    ];

    fn mix_clauses(gpu: &Gpu, k: &AttnResKernels) -> Result<Vec<bool>, GateError> {
        let mut out = Vec::new();
        let mut raw_least = f64::INFINITY;
        let mut noscale_least = f64::INFINITY;
        for (i, &(banks, nq, m)) in MIX_CASES.iter().enumerate() {
            let case = MixCase::new(0x1e2a_0000 + i as u64, banks, nq, m);
            let (card, rule) = case.run(gpu, k, MixDrop::None)?;
            let (again, _) = case.run(gpu, k, MixDrop::None)?;
            let width = nq * N;
            let per_row = |card: &[f32], rule: &[Vec<Vec<f64>>]| {
                worst(card, m, width, |t, row| {
                    rule[t]
                        .iter()
                        .enumerate()
                        .map(|(q, r)| rel_l2(&row[q * N..(q + 1) * N], r))
                        .fold(0.0f64, f64::max)
                })
            };
            let err = per_row(&card, &rule);
            let bits = bits_equal(&card, &again);
            // A pass of rows is each row's one-row launch's bits.
            let mut rows_bits = true;
            for t in 0..m {
                let one = MixCase {
                    banks: case
                        .banks
                        .iter()
                        .map(|b| b[t * N..(t + 1) * N].to_vec())
                        .collect(),
                    cur: vec![case.cur[t].clone()],
                    queries: case.queries.clone(),
                };
                let (y1, _) = one.run(gpu, k, MixDrop::None)?;
                rows_bits &= bits_equal(&y1[..width], &card[t * width..(t + 1) * width]);
            }
            let (_, raw) = case.run(gpu, k, MixDrop::Raw)?;
            raw_least = raw_least.min(per_row(&card, &raw));
            let (_, noscale) = case.run(gpu, k, MixDrop::NoScale)?;
            let noscale_here = per_row(&card, &noscale);
            if banks + 1 >= 9 {
                noscale_least = noscale_least.min(noscale_here);
            }
            let pass = err <= MIX_BAND && bits && rows_bits && gpu.fault()?.is_none();
            println!(
                "  mix_seeded {} vectors, {nq} queries, {m} rows: rel-L2 {err:.3e} (band \
                 {MIX_BAND:.0e}), rerun and rows bit-identical: {}: {}",
                banks + 1,
                if bits && rows_bits { "yes" } else { "no" },
                verdict(pass)
            );
            out.push(pass);
        }
        let pass = raw_least > SEP * MIX_BAND;
        println!(
            "  mix_separation against the raw-vector sum (Kimi's rule): least rel-L2 \
             {raw_least:.3e}, past {SEP}x band: {}",
            verdict(pass)
        );
        out.push(pass);
        let pass = noscale_least > SEP * MIX_BAND;
        println!(
            "  mix_separation against the scale dropped, the many-state mixes (>= 9 vectors): \
             least rel-L2 {noscale_least:.3e}, past {SEP}x band: {}",
            verdict(pass)
        );
        out.push(pass);
        Ok(out)
    }

    /// The step mix's outputs of case `a` and, paired, case `b` (one row each).
    fn step_run(
        gpu: &Gpu,
        step: &AttnResStepKernels,
        scratch: &mut MixStepScratch,
        a: &MixCase,
        b: Option<&MixCase>,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), GateError> {
        let s = gpu.stream();
        let ba = a.upload(gpu)?;
        let mut ya = DeviceBuffer::zeroed(s, a.queries.len() * N)?;
        let bb = b.map(|c| c.upload(gpu)).transpose()?;
        let mut yb = b
            .map(|c| DeviceBuffer::zeroed(s, c.queries.len() * N))
            .transpose()?;
        let second = match (b, &bb, yb.as_mut()) {
            (Some(c), Some(buf), Some(y)) => Some((c.step_args(gpu, buf), y)),
            _ => None,
        };
        step.enqueue(s, (a.step_args(gpu, &ba), &mut ya), second, scratch)?;
        s.synchronize()?;
        let out_b = yb.map(|y| y.to_host_vec(s)).transpose()?;
        Ok((ya.to_host_vec(s)?, out_b))
    }

    /// The `norm_proj` sites' rows at the files' shapes: the Q8_0 `q`, `k`, `v`, `rel` and `gate`
    /// (24 heads of 128, eight kv heads of 128, 16 relative rows a head), then the F32 `sink`,
    /// `kt` and `vt` (a sink a head, four conv taps a kv head).
    const PROJ_ROWS: [usize; 8] = [3072, 1024, 1024, 384, 3072, 24, 32, 32];
    /// Rows no Q8_0 block of sixteen divides: sites whose last block walks eight (24, 40, 56), a
    /// site of one block of eight (8) and one of exactly one block (16), and F32 sites of one to
    /// three blocks of eight.
    const PROJ_RAGGED: [usize; 8] = [24, 40, 8, 16, 56, 8, 16, 24];
    /// The input each site reads, at three inputs (`ik` 1, `iv` 2).
    const PROJ_INPUT: [usize; 8] = [0, 1, 2, 0, 0, 0, 1, 2];
    const PROJ_SITES: [&str; 8] = ["q", "k", "v", "rel", "gate", "sink", "kt", "vt"];
    /// What every output holds before a launch: a site the launch does not run keeps it.
    const UNWRITTEN: f32 = -7.25;

    /// One seeded `norm_proj` case: three input rows and their gains, the Q8_0 sites' words and
    /// f16 scales, the F32 sites' rows.
    struct ProjCase {
        rows: [usize; 8],
        x: [Vec<f32>; 3],
        gain: [Vec<f32>; 3],
        q8: [(Vec<u32>, Vec<u16>); 5],
        f32w: [Vec<f32>; 3],
    }

    impl ProjCase {
        fn new(seed: u64, rows: [usize; 8]) -> ProjCase {
            let mut r = GateRng(seed);
            let x = std::array::from_fn(|_| fill(&mut r, N, -2.0, 2.0));
            let gain = std::array::from_fn(|_| fill(&mut r, N, 0.5, 1.5));
            let q8 = std::array::from_fn(|i| {
                let rows = rows[i];
                let words = (0..rows * N / 4)
                    .map(|_| r.below(1 << 16) as u32 | ((r.below(1 << 16) as u32) << 16))
                    .collect();
                // Normal f16 scales of either sign in [2^-7, 2^-2).
                let scales = (0..rows * N / 32)
                    .map(|_| {
                        ((r.below(2) << 15) | ((8 + r.below(5)) << 10) | r.below(1 << 10)) as u16
                    })
                    .collect();
                (words, scales)
            });
            let f32w = std::array::from_fn(|i| fill(&mut r, rows[5 + i] * N, -0.1, 0.1));
            ProjCase {
                rows,
                x,
                gain,
                q8,
                f32w,
            }
        }
    }

    /// Eight outputs at the sites' rows, every value [`UNWRITTEN`].
    fn unwritten(gpu: &Gpu, rows: &[usize; 8]) -> Result<[DeviceBuffer<f32>; 8], GateError> {
        let s = gpu.stream();
        let y = |i: usize| -> Result<DeviceBuffer<f32>, GateError> {
            Ok(DeviceBuffer::from_host(s, &vec![UNWRITTEN; rows[i]])?)
        };
        Ok([y(0)?, y(1)?, y(2)?, y(3)?, y(4)?, y(5)?, y(6)?, y(7)?])
    }

    /// The oracle's outputs, the fused launch's, and the fault word the fused launch left.
    type ProjOut = (Vec<Vec<f32>>, Vec<Vec<f32>>, Option<Fault>);

    /// `case` at `n_in` inputs (one: every site reads input 0) with or without the taps: the
    /// oracle's outputs (`rms_norm` of each input, then each site's own gemv), the fused launch's,
    /// and the fault word the fused launch alone left.
    fn proj_run(
        gpu: &Gpu,
        k: &NormProjKernels,
        case: &ProjCase,
        n_in: usize,
        taps: bool,
    ) -> Result<ProjOut, GateError> {
        let s = gpu.stream();
        let (ik, iv) = if n_in == 1 { (0, 0) } else { (1, 2) };
        let at = |i: usize| if i < n_in { i } else { 0 };
        let mut xs = Vec::new();
        let mut gs = Vec::new();
        for i in 0..3 {
            xs.push(DeviceBuffer::from_host(s, &case.x[at(i)])?);
            gs.push(DeviceBuffer::from_host(s, &case.gain[at(i)])?);
        }
        let mut q8 = Vec::new();
        for (i, (words, scales)) in case.q8.iter().enumerate() {
            q8.push((
                DeviceTensor::upload(s, words, case.rows[i], N / 4)?,
                DeviceTensor::upload(s, scales, case.rows[i], N / 32)?,
            ));
        }
        let mut f32w = Vec::new();
        for (i, w) in case.f32w.iter().enumerate() {
            f32w.push(DeviceTensor::upload(s, w, case.rows[5 + i], N)?);
        }
        let mut want = unwritten(gpu, &case.rows)?;
        let mut card = unwritten(gpu, &case.rows)?;
        let mut xn = Vec::new();
        for (x, g) in xs.iter().zip(&gs) {
            let mut y = DeviceBuffer::zeroed(s, N)?;
            gpu.elem().enqueue_rms_norm(s, x, g, EPS, N, 1, &mut y)?;
            xn.push(y);
        }
        let input = PROJ_INPUT.map(at);
        let q = gpu.q8f32();
        for (j, y) in want.iter_mut().enumerate() {
            let x = &xn[input[j]];
            match j {
                0..5 => q.enqueue_q8_0_gemv(s, &q8[j].0, &q8[j].1, x, 1, y)?,
                5 => q.enqueue_f32_gemv(s, &f32w[0], x, 1, y)?,
                _ if taps => q.enqueue_f32_gemv(s, &f32w[j - 5], x, 1, y)?,
                _ => {}
            }
        }
        s.synchronize()?;
        gpu.clear_fault()?;
        let [cq, ck, cv, cr, cg, cs, ckt, cvt] = &mut card;
        let sink = &f32w[0];
        let site = |i: usize, y| Q8Site {
            qs: &q8[i].0,
            d: &q8[i].1,
            y,
        };
        k.enqueue(
            s,
            NormProjArgs {
                x: [&xs[0], &xs[1], &xs[2]],
                gain: [&gs[0], &gs[1], &gs[2]],
                ik,
                iv,
                eps: EPS,
                k: N,
                fault: gpu.unlabelled_sink(),
                q8: [
                    site(0, cq),
                    site(1, ck),
                    site(2, cv),
                    site(3, cr),
                    site(4, cg),
                ],
                sink: F32Site { w: sink, y: cs },
                kt: F32Site {
                    w: if taps { &f32w[1] } else { sink },
                    y: ckt,
                },
                vt: F32Site {
                    w: if taps { &f32w[2] } else { sink },
                    y: cvt,
                },
                taps,
            },
        )?;
        s.synchronize()?;
        let fault = gpu.fault()?;
        let host = |b: &[DeviceBuffer<f32>; 8]| -> Result<Vec<Vec<f32>>, GateError> {
            b.iter().map(|y| Ok(y.to_host_vec(s)?)).collect()
        };
        Ok((host(&want)?, host(&card)?, fault))
    }

    /// Clause `norm_proj_is_the_sites` (module doc, clause 11).
    fn norm_proj_clauses(gpu: &Gpu) -> Result<Vec<bool>, GateError> {
        let k = NormProjKernels::load(gpu.context())?;
        let case = ProjCase::new(0x1e2a_0200, PROJ_ROWS);
        let ragged = ProjCase::new(0x1e2a_0201, PROJ_RAGGED);
        let mut out = Vec::new();
        for (shape, c, combos) in [
            (
                "the sites' shapes",
                &case,
                &[(3, true), (1, false), (3, false), (1, true)][..],
            ),
            ("ragged rows", &ragged, &[(3, true), (1, false)][..]),
        ] {
            for &(n_in, taps) in combos {
                let (want, card, fault) = proj_run(gpu, &k, c, n_in, taps)?;
                let off: Vec<&str> = PROJ_SITES
                    .iter()
                    .zip(want.iter().zip(&card))
                    .filter(|(_, (w, c))| !bits_equal(w, c))
                    .map(|(name, _)| *name)
                    .collect();
                let kept = taps
                    || card[6..]
                        .iter()
                        .all(|y| y.iter().all(|v| v.to_bits() == UNWRITTEN.to_bits()));
                let pass = off.is_empty() && kept && fault.is_none();
                println!(
                    "  norm_proj_is_the_sites {shape}, {n_in} input(s), taps {taps}: sites off \
                     rms_norm and their gemvs' bits {off:?}, unrun taps unwritten {kept}, \
                     fault={fault:?}: {}",
                    verdict(pass)
                );
                out.push(pass);
            }
        }
        let mut bad = case;
        bad.x[0][7] = f32::NAN;
        let (want, card, fault) = proj_run(gpu, &k, &bad, 3, true)?;
        let nan = (0..8)
            .filter(|&j| PROJ_INPUT[j] == 0)
            .all(|j| card[j].iter().all(|v| v.is_nan()));
        let others = (0..8)
            .filter(|&j| PROJ_INPUT[j] != 0)
            .all(|j| bits_equal(&card[j], &want[j]));
        let pass = fault.is_some() && nan && others;
        println!(
            "  norm_proj_is_the_sites a NaN value of input 0: fault={fault:?} its sites all NaN \
             {nan}, the other inputs' sites their bits {others}: {}",
            verdict(pass)
        );
        gpu.clear_fault()?;
        out.push(pass);
        Ok(out)
    }

    /// Clause `mix_step_is_the_mix`: the step mix (`attn_res_step`) at one row of every mix
    /// point's shape — alone, run twice (the tickets back at zero between calls), and paired as
    /// the second mix of a launch beside the four-query message mix — gives `enqueue_normed`'s
    /// bits; a NaN bank value raises `f32_product` and every query row is NaN.
    fn mix_step_clauses(gpu: &Gpu, k: &AttnResKernels) -> Result<Vec<bool>, GateError> {
        let step = AttnResStepKernels::load(gpu.context())?;
        let mut scratch = MixStepScratch::new(gpu.stream(), N)?;
        let msg = MixCase::new(0x1e2a_0100, 3, 4, 1);
        let (msg_ref, _) = msg.run(gpu, k, MixDrop::None)?;
        let mut out = Vec::new();
        for (i, &(banks, nq, _)) in MIX_CASES.iter().enumerate() {
            let case = MixCase::new(0x1e2a_0000 + i as u64, banks, nq, 1);
            let (want, _) = case.run(gpu, k, MixDrop::None)?;
            let (alone, _) = step_run(gpu, &step, &mut scratch, &case, None)?;
            let (again, _) = step_run(gpu, &step, &mut scratch, &case, None)?;
            let (pm, pc) = step_run(gpu, &step, &mut scratch, &msg, Some(&case))?;
            let paired = bits_equal(&pm, &msg_ref) && pc.is_some_and(|c| bits_equal(&c, &want));
            let (a1, a2) = (bits_equal(&alone, &want), bits_equal(&again, &want));
            let pass = a1 && a2 && paired && gpu.fault()?.is_none();
            println!(
                "  mix_step_is_the_mix {} vectors, {nq} queries: alone {a1}, again {a2}, paired \
                 with the message mix {paired} (enqueue_normed's bits): {}",
                banks + 1,
                verdict(pass)
            );
            out.push(pass);
        }
        let mut bad = MixCase::new(0x1e2a_01ff, 2, 2, 1);
        for v in bad.banks.iter_mut() {
            v[7] = f64::NAN;
        }
        let (card, _) = step_run(gpu, &step, &mut scratch, &bad, None)?;
        let fault = gpu.fault()?;
        let nan = card.iter().all(|v| v.is_nan());
        let pass = fault.is_some() && nan;
        println!(
            "  mix_step_is_the_mix a NaN bank value: fault={fault:?} row_all_nan={nan} {}",
            verdict(pass)
        );
        gpu.clear_fault()?;
        out.push(pass);
        Ok(out)
    }

    fn mix_faults(gpu: &Gpu, k: &AttnResKernels) -> Result<Vec<bool>, GateError> {
        let mut case = MixCase::new(0x1e2a_00ff, 2, 2, 1);
        for v in case.banks.iter_mut() {
            v[7] = f64::NAN;
        }
        let (card, _) = case.run(gpu, k, MixDrop::None)?;
        let fault = gpu.fault()?;
        let nan = card.iter().all(|v| v.is_nan());
        let pass = fault.is_some() && nan;
        println!(
            "  mix_faults a NaN bank value: fault={fault:?} row_all_nan={nan} {}",
            verdict(pass)
        );
        gpu.clear_fault()?;
        Ok(vec![pass])
    }

    fn control_clauses(gpu: &Gpu, k: &LoopControlKernels) -> Result<Vec<bool>, GateError> {
        let mut out = Vec::new();
        for (i, &(n, rank, m)) in [
            (N, RANK, 1usize),
            (N, RANK, 8),
            (MAX_N, RANK, 1),
            (512, 16, 8),
        ]
        .iter()
        .enumerate()
        {
            let case = ControlCase::new(0x1c1c_0000 + i as u64, n, rank, m);
            let (card, rule) = case.run(gpu, k, ControlDrop::None)?;
            let err = worst(&card, m, n, |t, row| rel_l2(row, &rule[t]));
            let pass = err <= CONTROL_BAND && gpu.fault()?.is_none();
            println!(
                "  control_seeded {n} wide at rank {rank}, {m} rows: rel-L2 {err:.3e} (band \
                 {CONTROL_BAND:.0e}): {}",
                verdict(pass)
            );
            out.push(pass);
        }
        let case = ControlCase::new(0x1c1c_00ff, N, RANK, 1);
        let (card, _) = case.run(gpu, k, ControlDrop::None)?;
        for (drop, what) in [
            (ControlDrop::Swapped, "gain and forget swapped"),
            (ControlDrop::Unnormed, "the un-normed row"),
            (ControlDrop::NoTanh, "no tanh on the gates"),
        ] {
            let (_, rule) = case.run(gpu, k, drop)?;
            let sep = rel_l2(&card, &rule[0]);
            let pass = sep > SEP * CONTROL_BAND;
            println!(
                "  control_separation against {what}: rel-L2 {sep:.3e}, past {SEP}x band: {}",
                verdict(pass)
            );
            out.push(pass);
        }
        // The split launches against the one-fused-launch reference: the same bits, at a decode
        // row, the verify's eight, a width off the output group and a rank off the down group's
        // pair and the rank unroll, and the widest staging.
        for (i, &(n, rank, m)) in [
            (N, RANK, 1usize),
            (N, RANK, 8),
            (200, 7, 3),
            (OUT_COLS + 1, 1, 2),
            (MAX_N, RANK, 1),
        ]
        .iter()
        .enumerate()
        {
            let case = ControlCase::new(0x1c1c_1000 + i as u64, n, rank, m);
            let pair = control_pairs(gpu, k, &[&case])?;
            let pass = same_bits(&pair[0].0, &pair[0].1) && gpu.fault()?.is_none();
            println!(
                "  control_bitwise {n} wide at rank {rank}, {m} rows: split launches equal the                  one-block launch bit for bit: {}",
                verdict(pass)
            );
            out.push(pass);
        }
        // The attention launch: each of the three modules (their own weights and rows) equals
        // its own single launch bit for bit, and the three differ from one another.
        for (i, &(n, rank, m)) in [(N, RANK, 1usize), (N, RANK, 8), (200, 7, 3)]
            .iter()
            .enumerate()
        {
            let cases: Vec<ControlCase> = (0..3)
                .map(|j| ControlCase::new(0x1c1c_2000 + (i * 3 + j) as u64, n, rank, m))
                .collect();
            let pair = control_pairs(gpu, k, &[&cases[0], &cases[1], &cases[2]])?;
            let own = pair.iter().all(|(new, old)| same_bits(new, old));
            let distinct = !same_bits(&pair[0].1, &pair[1].1) && !same_bits(&pair[1].1, &pair[2].1);
            let pass = own && distinct && gpu.fault()?.is_none();
            println!(
                "  control_attn {n} wide at rank {rank}, {m} rows: three modules in one launch pair,                  each its own single launch's bits (own={own}, distinct={distinct}): {}",
                verdict(pass)
            );
            out.push(pass);
        }
        let mut bad = ControlCase::new(0x1c1c_ff00, N, RANK, 1);
        bad.msg[0][11] = f64::NAN;
        let (card, _) = bad.run(gpu, k, ControlDrop::None)?;
        let fault = gpu.fault()?;
        let nan = card.iter().all(|v| v.is_nan());
        let pass = fault.is_some() && nan;
        println!(
            "  control_faults a NaN message: fault={fault:?} row_all_nan={nan} {}",
            verdict(pass)
        );
        gpu.clear_fault()?;
        out.push(pass);
        Ok(out)
    }

    fn conv_clauses(gpu: &Gpu, k: &ConvResKernels) -> Result<Vec<bool>, GateError> {
        let mut out = Vec::new();
        let rows = Ring::new(TAPS).rows();
        let len = rows + 6;
        let cases: &[(&str, usize, usize, usize, usize)] = &[
            ("the k conv of a windowed visit", K_CH, 0, K_BASE, K_CH),
            (
                "the k conv of a windowless visit, dynamic",
                K_CH,
                KV_HEADS,
                K_BASE,
                K_CH,
            ),
            (
                "the attention output's post conv in the visit's ring",
                POST_CH,
                0,
                POST_BASE,
                RING_CH,
            ),
        ];
        for (i, &(what, d, heads, base, ch)) in cases.iter().enumerate() {
            let walk = ConvWalk::new(0xc0a0_0000 + i as u64, d, heads, base, ch, len);
            let calls = vec![1usize; len];
            let (card, rule) = walk.run(gpu, k, &calls, false)?;
            let err = worst(&card, len, d, |t, row| rel_l2(row, &rule[t]));
            let pass = err <= CONV_BAND && gpu.fault()?.is_none();
            println!(
                "  conv_seeded {what} ({d} channels, {} dynamic heads, a {len}-token walk over a \
                 ring of {rows} rows): rel-L2 {err:.3e} (band {CONV_BAND:.0e}): {}",
                if heads > 0 {
                    heads.to_string()
                } else {
                    "no".to_string()
                },
                verdict(pass)
            );
            out.push(pass);
            if heads > 0 {
                let (_, reversed) = walk.run(gpu, k, &calls, true)?;
                let sep = worst(&card, len, d, |t, row| rel_l2(row, &reversed[t]));
                let pass = sep > SEP * CONV_BAND;
                println!(
                    "  conv_separation {what}: the dynamic order reversed: rel-L2 {sep:.3e}, \
                     past {SEP}x band: {}",
                    verdict(pass)
                );
                out.push(pass);
            }
        }
        // The static order is its own: reversed static taps separate the static conv, and the
        // dynamic order reversed does not move one (the windowed case's separation clause is the
        // dynamic conv's; here the static rule with its taps reversed is the mutant's).
        let stat = ConvWalk::new(0xc0a0_00ee, K_CH, 0, K_BASE, K_CH, len);
        let calls = vec![1usize; len];
        let (card, _) = stat.run(gpu, k, &calls, false)?;
        let mut rev = stat;
        for c in 0..K_CH {
            rev.w[c * TAPS..(c + 1) * TAPS].reverse();
        }
        let sep = worst(&card, len, K_CH, |t, row| {
            let want = conv_rule(
                &rev.hist,
                &rev.w,
                &rev.dynamic,
                t,
                rev.d,
                rev.heads,
                f64::from(rev.scale),
                false,
            );
            rel_l2(row, &want)
        });
        let pass = sep > SEP * CONV_BAND;
        println!(
            "  conv_separation the k conv: the static order reversed: rel-L2 {sep:.3e}, past \
             {SEP}x band: {}",
            verdict(pass)
        );
        out.push(pass);
        // One call of eight tokens is the eight one-token calls' bits.
        let batch = ConvWalk::new(0xc0a0_00dd, K_CH, KV_HEADS, K_BASE, K_CH, 8);
        let (one, _) = batch.run(gpu, k, &[1; 8], false)?;
        let (many, _) = batch.run(gpu, k, &[8], false)?;
        let bits = bits_equal(&one, &many);
        let pass = bits && gpu.fault()?.is_none();
        println!(
            "  conv_seeded a call of eight tokens against eight one-token calls: bits {}: {}",
            if bits { "equal" } else { "differ" },
            verdict(pass)
        );
        out.push(pass);
        // A NaN input raises and NaNs exactly its own value: the conv is depthwise, so channel 5
        // of the token is NaN (its residual reads the NaN input) and every other channel of the
        // token is finite — the contract writes NaN where the rule is not finite, never a
        // plausible value and never a spurious NaN beside it.
        let mut bad = ConvWalk::new(0xc0a0_00cc, K_CH, 0, K_BASE, K_CH, 4);
        bad.hist[2][5] = f64::NAN;
        let (card, _) = bad.run(gpu, k, &[4], false)?;
        let fault = gpu.fault()?;
        let token = &card[2 * K_CH..3 * K_CH];
        let (nan, finite) = (
            token[5].is_nan(),
            token
                .iter()
                .enumerate()
                .all(|(c, v)| c == 5 || v.is_finite()),
        );
        let pass = fault.is_some() && nan && finite;
        println!(
            "  conv_faults a NaN input at the third token: fault={fault:?} its_channel_nan={nan} \
             others_finite={finite} {}",
            verdict(pass)
        );
        gpu.clear_fault()?;
        out.push(pass);
        Ok(out)
    }

    /// The epilogue's attention shape: heads and key-value heads of [`HEAD`] values, the
    /// relative rows of [`REL`].
    const N_HEAD: usize = 24;
    const HEAD: usize = 128;
    const REL: usize = 16;

    /// One token's seeded inputs of the fused launch's case: the epilogue's rows and the convs'
    /// inputs and dynamic taps.
    struct ClipToken {
        q: Vec<f32>,
        r: Vec<f32>,
        kx: Vec<f32>,
        vx: Vec<f32>,
        kd: Vec<f32>,
        vd: Vec<f32>,
        tau: f32,
    }

    /// One chain's device state: the rows the epilogue edits in place, the f16 planes and the
    /// conv ring.
    struct ClipChain {
        k: DeviceBuffer<f32>,
        v: DeviceBuffer<f32>,
        cache_k: DeviceBuffer<u16>,
        cache_v: DeviceBuffer<u16>,
        ring: DeviceBuffer<f32>,
    }

    impl ClipChain {
        fn new(gpu: &Gpu, ctx: usize, ch: usize) -> Result<ClipChain, GateError> {
            let s = gpu.stream();
            let kv = KV_HEADS * HEAD;
            Ok(ClipChain {
                k: DeviceBuffer::zeroed(s, kv)?,
                v: DeviceBuffer::zeroed(s, kv)?,
                cache_k: DeviceBuffer::zeroed(s, KV_HEADS * ctx * HEAD)?,
                cache_v: DeviceBuffer::zeroed(s, KV_HEADS * ctx * HEAD)?,
                ring: DeviceBuffer::zeroed(s, Ring::new(TAPS).rows() * ch)?,
            })
        }
    }

    /// The case's shared weights and positions: the convs' filters, the clip's channel scales.
    struct ClipCase {
        heads: usize,
        ctx: usize,
        wk: DeviceBuffer<f32>,
        wv: DeviceBuffer<f32>,
        gq: DeviceBuffer<f32>,
        gr: DeviceBuffer<f32>,
        tokens: Vec<ClipToken>,
    }

    impl ClipCase {
        /// `len` tokens, dynamic convs at `heads` (0 static), the planes `ctx` rows. The key
        /// rows of head `h` carry the scale `0.3 + 0.5 h`, so the clip bites on some heads and
        /// not on others; `nan_at` puts a NaN into that token's key input.
        fn new(
            gpu: &Gpu,
            seed: u64,
            (heads, len, ctx): (usize, usize, usize),
            nan_at: Option<usize>,
        ) -> Result<ClipCase, GateError> {
            let s = gpu.stream();
            let mut r = GateRng(seed);
            let kv = KV_HEADS * HEAD;
            let tokens = (0..len)
                .map(|t| {
                    let rows = |r: &mut GateRng, n: usize| -> Vec<f32> {
                        (0..n)
                            .map(|i| {
                                r.uni(-1.0, 1.0) * (0.3 + 0.5 * ((i / HEAD) % KV_HEADS) as f32)
                            })
                            .collect()
                    };
                    let mut kx = rows(&mut r, kv);
                    if nan_at == Some(t) {
                        kx[5] = f32::NAN;
                    }
                    ClipToken {
                        q: fill(&mut r, N_HEAD * HEAD, -3.0, 3.0),
                        r: fill(&mut r, N_HEAD * REL, -2.0, 2.0),
                        kx,
                        vx: rows(&mut r, kv),
                        kd: fill(&mut r, heads * TAPS, -0.5, 0.5),
                        vd: fill(&mut r, heads * TAPS, -0.5, 0.5),
                        tau: if heads > 0 {
                            1.0 + 0.01 * t as f32
                        } else {
                            1.0
                        },
                    }
                })
                .collect();
            Ok(ClipCase {
                heads,
                ctx,
                wk: DeviceBuffer::from_host(s, &fill(&mut r, kv * TAPS, -1.0, 1.0))?,
                wv: DeviceBuffer::from_host(s, &fill(&mut r, kv * TAPS, -1.0, 1.0))?,
                gq: DeviceBuffer::from_host(s, &fill(&mut r, N_HEAD * HEAD, 0.5, 2.0))?,
                gr: DeviceBuffer::from_host(s, &fill(&mut r, N_HEAD * REL, 0.5, 1.5))?,
                tokens,
            })
        }

        /// The walk once on the pair of launches (`fused` false) or on the one launch: each
        /// token's query, relative, key and value rows, every `every` tokens and at the end the
        /// planes and the ring, the fault word at the end.
        fn run(
            &self,
            gpu: &Gpu,
            (conv, clip): (&ConvResKernels, &QkClipKernels),
            fused: bool,
            every: usize,
        ) -> Result<(Vec<Vec<f32>>, String), GateError> {
            let s = gpu.stream();
            let ring = Ring::new(TAPS);
            let kv = KV_HEADS * HEAD;
            let mut c = ClipChain::new(gpu, self.ctx, RING_CH)?;
            let mut seen = Vec::new();
            gpu.clear_fault()?;
            for (t, tok) in self.tokens.iter().enumerate() {
                let kx = DeviceBuffer::from_host(s, &tok.kx)?;
                let vx = DeviceBuffer::from_host(s, &tok.vx)?;
                // A static conv's taps stay one value: a zero-length allocation is not one the
                // card makes, and the launch reads no tap then.
                let taps_of = |v: &Vec<f32>| -> Result<DeviceBuffer<f32>, GateError> {
                    Ok(DeviceBuffer::from_host(
                        s,
                        if v.is_empty() { &[0.0] } else { &v[..] },
                    )?)
                };
                let (kd, vd) = (taps_of(&tok.kd)?, taps_of(&tok.vd)?);
                let mut q = DeviceBuffer::from_host(s, &tok.q)?;
                let mut r = DeviceBuffer::from_host(s, &tok.r)?;
                let tau = DeviceBuffer::from_host(s, &[tok.tau])?;
                let pos = DeviceBuffer::from_host(s, &[t as u32])?;
                let scale = if self.heads > 0 {
                    scale_dynamic()
                } else {
                    scale_static()
                };
                macro_rules! clip_args {
                    () => {
                        QkClipArgs {
                            q: &mut q,
                            k: &mut c.k,
                            v: &mut c.v,
                            r: &mut r,
                            gq: &self.gq,
                            gr: &self.gr,
                            tau: &tau,
                            pos: &pos,
                            v_eps: EPS,
                            qk_thresh: 22.627_417,
                            r_thresh: 8.0,
                            n_head: N_HEAD,
                            n_kv: KV_HEADS,
                            ctx: self.ctx,
                            m: 1,
                            fault: gpu.unlabelled_sink(),
                            cache_k: &mut c.cache_k,
                            cache_v: &mut c.cache_v,
                        }
                    };
                }
                if fused {
                    clip.enqueue_conv(
                        s,
                        clip_args!(),
                        QkClipConv {
                            ring,
                            ch: RING_CH,
                            base: [K_BASE, K_BASE + kv],
                            heads: self.heads,
                            scale,
                            x: [&kx, &vx],
                            weights: [&self.wk, &self.wv],
                            dynamic: [&kd, &vd],
                            pos: &pos,
                            state: &mut c.ring,
                        },
                    )?;
                } else {
                    for (j, (x, w, dynamic, y)) in [
                        (&kx, &self.wk, &kd, &mut c.k),
                        (&vx, &self.wv, &vd, &mut c.v),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        conv.enqueue(
                            s,
                            ConvResArgs {
                                ring,
                                d: kv,
                                base: K_BASE + j * kv,
                                ch: RING_CH,
                                heads: self.heads,
                                m: 1,
                                x,
                                weights: w,
                                dynamic,
                                pos: &pos,
                                fault: gpu.unlabelled_sink(),
                                scale,
                                y,
                                state: &mut c.ring,
                            },
                        )?;
                    }
                    clip.enqueue(s, clip_args!())?;
                }
                s.synchronize()?;
                let mut row = [
                    q.to_host_vec(s)?,
                    r.to_host_vec(s)?,
                    c.k.to_host_vec(s)?,
                    c.v.to_host_vec(s)?,
                ]
                .concat();
                if (t + 1) % every == 0 || t + 1 == self.tokens.len() {
                    row.extend(c.ring.to_host_vec(s)?);
                    row.extend(c.cache_k.to_host_vec(s)?.iter().map(|b| f32::from(*b)));
                    row.extend(c.cache_v.to_host_vec(s)?.iter().map(|b| f32::from(*b)));
                }
                seen.push(row);
            }
            let fault = format!("{:?}", gpu.fault()?);
            gpu.clear_fault()?;
            Ok((seen, fault))
        }
    }

    fn conv_clip_clauses(
        gpu: &Gpu,
        conv: &ConvResKernels,
        clip: &QkClipKernels,
    ) -> Result<Vec<bool>, GateError> {
        let mut out = Vec::new();
        let len = Ring::new(TAPS).rows() + 6;
        // (what, dynamic heads, tokens, plane rows, NaN at)
        let cases: &[(&str, usize, usize, usize, Option<usize>)] = &[
            ("a windowed visit, static convs", 0, len, len, None),
            (
                "a windowless visit, dynamic convs",
                KV_HEADS,
                len,
                len,
                None,
            ),
            ("a NaN key input at the fourth token", 0, 8, 8, Some(3)),
            ("an append past the planes' end", KV_HEADS, 8, 5, None),
        ];
        for (i, &(what, heads, n, ctx, nan_at)) in cases.iter().enumerate() {
            let case = ClipCase::new(gpu, 0xc11b_0000 + i as u64, (heads, n, ctx), nan_at)?;
            let (pair, pair_fault) = case.run(gpu, (conv, clip), false, 64)?;
            let (fused, fused_fault) = case.run(gpu, (conv, clip), true, 64)?;
            let bits = pair.len() == fused.len()
                && pair.iter().zip(&fused).all(|(a, b)| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                });
            // The fault word is the pair's: none on a clean walk, the same site on a faulty one.
            let faults_expected = nan_at.is_some() || ctx < n;
            let pass =
                bits && pair_fault == fused_fault && (pair_fault != "None") == faults_expected;
            println!(
                "  conv_clip_is_the_pair {what} ({n} tokens, planes of {ctx} rows): bits {} fault \
                 pair={pair_fault} fused={fused_fault}: {}",
                if bits { "equal" } else { "differ" },
                verdict(pass)
            );
            out.push(pass);
        }
        Ok(out)
    }

    /// The tail clauses' shapes: the output projection's 3,072 values a row, the feed-forward's
    /// post conv at its offset in the ring, the card's slots a token and the places the card
    /// holds.
    const O_K: usize = 3072;
    const FFN_BASE: usize = 4352;
    const CARD_SLOTS: usize = 8;
    const N_CARD: usize = 128;
    /// What the join's output holds before a launch that does not write it.
    const JOIN_UNWRITTEN: f32 = -5.5;

    /// One tail walk's token rows and filters: the stream's rows and the conv's filters of a
    /// seeded walk of `len` tokens, positions `0..len` at the sequence head.
    struct TailWalk {
        len: usize,
        w: Vec<f32>,
        stream: Vec<Vec<f32>>,
    }

    impl TailWalk {
        fn new(r: &mut GateRng, len: usize) -> TailWalk {
            TailWalk {
                len,
                w: fill(r, N * TAPS, -1.0, 1.0),
                stream: (0..len).map(|_| fill(r, N, -2.0, 2.0)).collect(),
            }
        }
    }

    /// Each token's values the walk leaves, the ring after the last one, and the fault word.
    type TailOut = (Vec<Vec<f32>>, String);

    /// The projection's case: Q8_0 planes at the output projection's shape and a token's
    /// activations a position.
    struct GemvCase {
        walk: TailWalk,
        words: Vec<u32>,
        scales: Vec<u16>,
        x: Vec<Vec<f32>>,
    }

    impl GemvCase {
        /// `nan_at` puts a NaN into that token's activations.
        fn new(seed: u64, len: usize, nan_at: Option<usize>) -> GemvCase {
            let mut r = GateRng(seed);
            let walk = TailWalk::new(&mut r, len);
            let words = (0..N * O_K / 4)
                .map(|_| r.below(1 << 16) as u32 | ((r.below(1 << 16) as u32) << 16))
                .collect();
            // Normal f16 scales of either sign in [2^-7, 2^-2).
            let scales = (0..N * O_K / 32)
                .map(|_| ((r.below(2) << 15) | ((8 + r.below(5)) << 10) | r.below(1 << 10)) as u16)
                .collect();
            let x = (0..len)
                .map(|t| {
                    let mut x = fill(&mut r, O_K, -1.0, 1.0);
                    if nan_at == Some(t) {
                        x[7] = f32::NAN;
                    }
                    x
                })
                .collect();
            GemvCase {
                walk,
                words,
                scales,
                x,
            }
        }

        /// The walk on the three launches (`fused` false) or the one, `join` as a block's later
        /// visits have it: per token the conv's output and the join's row, then the ring.
        fn run(
            &self,
            gpu: &Gpu,
            (conv, tail): (&ConvResKernels, &ConvTailKernels),
            fused: bool,
            join: bool,
        ) -> Result<TailOut, GateError> {
            let s = gpu.stream();
            let ring = Ring::new(TAPS);
            let qs = DeviceTensor::upload(s, &self.words, N, O_K / 4)?;
            let d = DeviceTensor::upload(s, &self.scales, N, O_K / 32)?;
            let w = DeviceBuffer::from_host(s, &self.walk.w)?;
            let none = DeviceBuffer::from_host(s, &[0.0f32])?;
            let mut state = DeviceBuffer::zeroed(s, ring.rows() * RING_CH)?;
            let mut seen = Vec::new();
            gpu.clear_fault()?;
            for t in 0..self.walk.len {
                let x = DeviceBuffer::from_host(s, &self.x[t])?;
                let stream = DeviceBuffer::from_host(s, &self.walk.stream[t])?;
                let pos = DeviceBuffer::from_host(s, &[t as u32])?;
                let mut out = DeviceBuffer::zeroed(s, N)?;
                let mut joined = DeviceBuffer::from_host(s, &vec![JOIN_UNWRITTEN; N])?;
                if fused {
                    tail.enqueue_gemv_conv(
                        s,
                        GemvConvArgs {
                            qs: &qs,
                            d: &d,
                            x: &x,
                            conv: TailConvArgs {
                                ring,
                                ch: RING_CH,
                                base: POST_BASE,
                                scale: scale_static(),
                                weights: &w,
                                pos: &pos,
                                state: &mut state,
                            },
                            join,
                            stream: &stream,
                            fault: gpu.unlabelled_sink(),
                            out: &mut out,
                            joined: &mut joined,
                        },
                    )?;
                } else {
                    let mut o = DeviceBuffer::zeroed(s, N)?;
                    gpu.q8f32().enqueue_q8_0_gemv(s, &qs, &d, &x, 1, &mut o)?;
                    conv.enqueue(
                        s,
                        ConvResArgs {
                            ring,
                            d: N,
                            base: POST_BASE,
                            ch: RING_CH,
                            heads: 0,
                            m: 1,
                            x: &o,
                            weights: &w,
                            dynamic: &none,
                            pos: &pos,
                            fault: gpu.unlabelled_sink(),
                            scale: scale_static(),
                            y: &mut out,
                            state: &mut state,
                        },
                    )?;
                    if join {
                        gpu.elem().enqueue_add(s, &stream, &out, N, &mut joined)?;
                    }
                }
                s.synchronize()?;
                let mut row = out.to_host_vec(s)?;
                row.extend(joined.to_host_vec(s)?);
                seen.push(row);
            }
            seen.push(state.to_host_vec(s)?);
            let fault = format!("{:?}", gpu.fault()?);
            gpu.clear_fault()?;
            Ok((seen, fault))
        }
    }

    /// The card sum's case: a token's slots' down rows, routing weights and places a position.
    struct SumCase {
        walk: TailWalk,
        down: Vec<Vec<f32>>,
        w: Vec<Vec<f32>>,
        sel: Vec<Vec<u32>>,
    }

    impl SumCase {
        /// Every token's places drawn on the card; `host` slots of each are the host's and
        /// `past` puts the first token's slot 3 at a place that is neither.
        fn new(seed: u64, len: usize, (host, past): (&[usize], bool)) -> SumCase {
            let mut r = GateRng(seed);
            let walk = TailWalk::new(&mut r, len);
            let down = (0..len)
                .map(|_| fill(&mut r, CARD_SLOTS * N, -1.0, 1.0))
                .collect();
            let w = (0..len)
                .map(|_| fill(&mut r, CARD_SLOTS, 0.05, 0.5))
                .collect();
            let sel = (0..len)
                .map(|t| {
                    let mut sel: Vec<u32> =
                        (0..CARD_SLOTS).map(|_| r.below(N_CARD) as u32).collect();
                    for &j in host {
                        sel[j] = HOST;
                    }
                    if past && t == 0 {
                        sel[3] = N_CARD as u32 + 7;
                    }
                    sel
                })
                .collect();
            SumCase { walk, down, w, sel }
        }

        /// The walk on the three launches or the one: per token the sum, the conv's output and
        /// the join's row, then the ring.
        fn run(
            &self,
            gpu: &Gpu,
            (conv, slots, tail): (&ConvResKernels, &CardSlotKernels, &ConvTailKernels),
            fused: bool,
        ) -> Result<TailOut, GateError> {
            let s = gpu.stream();
            let ring = Ring::new(TAPS);
            let cw = DeviceBuffer::from_host(s, &self.walk.w)?;
            let none = DeviceBuffer::from_host(s, &[0.0f32])?;
            let mut state = DeviceBuffer::zeroed(s, ring.rows() * RING_CH)?;
            let mut seen = Vec::new();
            gpu.clear_fault()?;
            for t in 0..self.walk.len {
                let down = DeviceBuffer::from_host(s, &self.down[t])?;
                let w = DeviceBuffer::from_host(s, &self.w[t])?;
                let sel = DeviceBuffer::from_host(s, &self.sel[t])?;
                let inp = DeviceBuffer::from_host(s, &self.walk.stream[t])?;
                let pos = DeviceBuffer::from_host(s, &[t as u32])?;
                let mut acc = DeviceBuffer::zeroed(s, N)?;
                let mut out = DeviceBuffer::zeroed(s, N)?;
                let mut y = DeviceBuffer::zeroed(s, N)?;
                if fused {
                    tail.enqueue_card_acc_conv(
                        s,
                        CardAccConvArgs {
                            down: &down,
                            w: &w,
                            sel: &sel,
                            n_card: N_CARD,
                            n: N,
                            conv: TailConvArgs {
                                ring,
                                ch: RING_CH,
                                base: FFN_BASE,
                                scale: scale_static(),
                                weights: &cw,
                                pos: &pos,
                                state: &mut state,
                            },
                            inp: &inp,
                            fault: gpu.unlabelled_sink(),
                            acc: &mut acc,
                            conv_out: &mut out,
                            y: &mut y,
                        },
                    )?;
                } else {
                    slots.enqueue_card_acc(
                        s,
                        &CardAcc {
                            down: &down,
                            w: &w,
                            sel: &sel,
                            n: N,
                            m: 1,
                            n_card: N_CARD,
                            n_used: CARD_SLOTS,
                            w_pitch: CARD_SLOTS,
                            fault: gpu.unlabelled_sink(),
                        },
                        &mut acc,
                    )?;
                    conv.enqueue(
                        s,
                        ConvResArgs {
                            ring,
                            d: N,
                            base: FFN_BASE,
                            ch: RING_CH,
                            heads: 0,
                            m: 1,
                            x: &acc,
                            weights: &cw,
                            dynamic: &none,
                            pos: &pos,
                            fault: gpu.unlabelled_sink(),
                            scale: scale_static(),
                            y: &mut out,
                            state: &mut state,
                        },
                    )?;
                    gpu.elem().enqueue_add(s, &inp, &out, N, &mut y)?;
                }
                s.synchronize()?;
                let mut row = acc.to_host_vec(s)?;
                row.extend(out.to_host_vec(s)?);
                row.extend(y.to_host_vec(s)?);
                seen.push(row);
            }
            seen.push(state.to_host_vec(s)?);
            let fault = format!("{:?}", gpu.fault()?);
            gpu.clear_fault()?;
            Ok((seen, fault))
        }
    }

    /// Whether a fused walk and the launches' walk agree: every value's bits and the fault word,
    /// which is clean exactly where `faults` says the case raises.
    fn tail_agrees(pair: &TailOut, fused: &TailOut, faults: bool) -> (bool, bool) {
        let bits = pair.0.len() == fused.0.len()
            && pair.0.iter().zip(&fused.0).all(|(a, b)| same_bits(a, b));
        (
            bits,
            bits && pair.1 == fused.1 && (pair.1 != "None") == faults,
        )
    }

    fn conv_tail_clauses(
        gpu: &Gpu,
        (conv, slots, tail): (&ConvResKernels, &CardSlotKernels, &ConvTailKernels),
    ) -> Result<Vec<bool>, GateError> {
        let mut out = Vec::new();
        let len = Ring::new(TAPS).rows() + 6;
        // (what, join, NaN at)
        let gemv: &[(&str, bool, Option<usize>)] = &[
            ("a later visit of a block, joined", true, None),
            ("a block's first visit, nothing joined", false, None),
            ("a NaN activation at the fourth token", true, Some(3)),
        ];
        for (i, &(what, join, nan_at)) in gemv.iter().enumerate() {
            let case = GemvCase::new(0x7a11_0000 + i as u64, len, nan_at);
            let pair = case.run(gpu, (conv, tail), false, join)?;
            let fused = case.run(gpu, (conv, tail), true, join)?;
            let (bits, pass) = tail_agrees(&pair, &fused, nan_at.is_some());
            println!(
                "  conv_tail_is_the_launches the output projection, {what} ({len} tokens): bits \
                 {} fault pair={} fused={}: {}",
                if bits { "equal" } else { "differ" },
                pair.1,
                fused.1,
                verdict(pass)
            );
            out.push(pass);
        }
        // (what, host slots, a place past the card)
        let sums: &[(&str, &[usize], bool)] = &[
            ("every slot on the card", &[], false),
            ("two slots on the host", &[2, 5], false),
            ("a place past the card at the first token", &[], true),
        ];
        for (i, &(what, host, past)) in sums.iter().enumerate() {
            let case = SumCase::new(0x7a22_0000 + i as u64, len, (host, past));
            let pair = case.run(gpu, (conv, slots, tail), false)?;
            let fused = case.run(gpu, (conv, slots, tail), true)?;
            let (bits, pass) = tail_agrees(&pair, &fused, past);
            println!(
                "  conv_tail_is_the_launches the card sum, {what} ({len} tokens): bits {} fault \
                 pair={} fused={}: {}",
                if bits { "equal" } else { "differ" },
                pair.1,
                fused.1,
                verdict(pass)
            );
            out.push(pass);
        }
        Ok(out)
    }

    // ------------------------------------------------------------------ the refusals

    /// The text of the refusal `r` is, or `None` when it ran.
    fn refused<T>(r: Result<T, GpuError>) -> Option<String> {
        r.err().map(|e| e.to_string())
    }

    fn refusal(what: &str, got: Option<String>, names: &str) -> bool {
        let pass = got.as_deref().is_some_and(|m| m.contains(names));
        println!(
            "  refusal {what}: {}: {}",
            got.as_deref().unwrap_or("ran without a refusal"),
            verdict(pass)
        );
        pass
    }

    fn refusal_clauses(
        gpu: &Gpu,
        mix: &AttnResKernels,
        control: &LoopControlKernels,
        conv: &ConvResKernels,
    ) -> Result<Vec<bool>, GateError> {
        let s = gpu.stream();
        let mut out = Vec::new();
        let banks = DeviceBuffer::from_host(s, &vec![0.0f32; 2 * N])?;
        let cur = DeviceBuffer::from_host(s, &vec![0.0f32; N])?;
        let q1 = DeviceBuffer::from_host(s, &vec![0.0f32; N])?;
        let mut y = DeviceBuffer::zeroed(s, N)?;
        let mut mix_at = |nq: usize, nb: usize, st: usize, eps: f32, scale: f32| {
            mix.enqueue_normed(
                s,
                AttnResNormedArgs {
                    banks: &banks,
                    n_banks: nb,
                    stride: st,
                    cur: &cur,
                    queries: &q1,
                    nq,
                    scale,
                    eps,
                    n: N,
                    m: 1,
                    fault: gpu.unlabelled_sink(),
                },
                &mut y,
            )
        };
        out.push(refusal(
            "no query row",
            refused(mix_at(0, 2, N, EPS, SCALE)),
            "0 query",
        ));
        out.push(refusal(
            "five query rows",
            refused(mix_at(MAX_NQ + 1, 2, N, EPS, SCALE)),
            &format!("at most {MAX_NQ}"),
        ));
        out.push(refusal(
            "seventeen vectors",
            refused(mix_at(1, MAX_VECS, N, EPS, SCALE)),
            "at most",
        ));
        out.push(refusal(
            "a stride under a pass's rows",
            refused(mix_at(1, 2, N - 1, EPS, SCALE)),
            "stride",
        ));
        out.push(refusal(
            "a zero epsilon",
            refused(mix_at(1, 2, N, 0.0, SCALE)),
            "epsilon",
        ));
        out.push(refusal(
            "a NaN scale",
            refused(mix_at(1, 2, N, EPS, f32::NAN)),
            "scale",
        ));
        let mut short = DeviceBuffer::zeroed(s, N - 1)?;
        out.push(refusal(
            "a short output",
            refused(mix.enqueue_normed(
                s,
                AttnResNormedArgs {
                    banks: &banks,
                    n_banks: 2,
                    stride: N,
                    cur: &cur,
                    queries: &q1,
                    nq: 1,
                    scale: SCALE,
                    eps: EPS,
                    n: N,
                    m: 1,
                    fault: gpu.unlabelled_sink(),
                },
                &mut short,
            )),
            "the output",
        ));

        let cc = ControlCase::new(0x1c1c_0f0f, 512, 16, 1);
        let ccur = DeviceBuffer::from_host(s, &f32s(&cc.cur.concat()))?;
        let cmsg = DeviceBuffer::from_host(s, &f32s(&cc.msg.concat()))?;
        let cdown = DeviceBuffer::from_host(s, &f32s(&cc.down))?;
        let cgain = DeviceBuffer::from_host(s, &f32s(&cc.gain))?;
        let cforget = DeviceBuffer::from_host(s, &f32s(&cc.forget))?;
        let mut cy = DeviceBuffer::zeroed(s, 512)?;
        let mut ch = DeviceBuffer::zeroed(s, scratch_len(1, 1, 16))?;
        let mut short_h = DeviceBuffer::zeroed(s, scratch_len(1, 1, 16) - 1)?;
        let mut control_at = |n: usize, rank: usize, m: usize, eps: f32| {
            control.enqueue(
                s,
                LoopControlArgs {
                    cur: &ccur,
                    msg: &cmsg,
                    down: &cdown,
                    gain: &cgain,
                    forget: &cforget,
                    n,
                    rank,
                    m,
                    eps,
                    fault: gpu.unlabelled_sink(),
                },
                &mut ch,
                &mut cy,
            )
        };
        out.push(refusal(
            "no width",
            refused(control_at(0, 16, 1, EPS)),
            "0 values",
        ));
        out.push(refusal(
            "no rank",
            refused(control_at(512, 0, 1, EPS)),
            "rank 0",
        ));
        out.push(refusal(
            "a width over the staging bound",
            refused(control_at(MAX_N + 1, 16, 1, EPS)),
            "at most",
        ));
        out.push(refusal(
            "a rank over the bound",
            refused(control_at(512, MAX_RANK + 1, 1, EPS)),
            "at most",
        ));
        out.push(refusal(
            "a negative epsilon",
            refused(control_at(512, 16, 1, -1.0)),
            "epsilon",
        ));
        out.push(refusal(
            "a short weight",
            refused(control_at(512, 16, 2, EPS)),
            "cur",
        ));
        out.push(refusal(
            "a short scratch",
            refused(control.enqueue(
                s,
                LoopControlArgs {
                    cur: &ccur,
                    msg: &cmsg,
                    down: &cdown,
                    gain: &cgain,
                    forget: &cforget,
                    n: 512,
                    rank: 16,
                    m: 1,
                    eps: EPS,
                    fault: gpu.unlabelled_sink(),
                },
                &mut short_h,
                &mut cy,
            )),
            "h",
        ));
        let (mut cy1, mut cy2) = (
            DeviceBuffer::<f32>::zeroed(s, 512)?,
            DeviceBuffer::<f32>::zeroed(s, 512)?,
        );
        let mut ch3 = DeviceBuffer::zeroed(s, scratch_len(MAX_MODULES, 1, 16))?;
        let module = |n: usize| LoopControlArgs {
            cur: &ccur,
            msg: &cmsg,
            down: &cdown,
            gain: &cgain,
            forget: &cforget,
            n,
            rank: 16,
            m: 1,
            eps: EPS,
            fault: gpu.unlabelled_sink(),
        };
        out.push(refusal(
            "an attention module of another width",
            refused(control.enqueue_attn(
                s,
                [module(512), module(256), module(512)],
                &mut ch3,
                [&mut cy, &mut cy1, &mut cy2],
            )),
            "share one shape",
        ));

        let ring = Ring::new(TAPS);
        let walk = ConvWalk::new(0xc0a0_0f0f, K_CH, 0, K_BASE, K_CH, 2);
        let xs = DeviceBuffer::from_host(s, &f32s(&walk.hist.concat()))?;
        let ws = DeviceBuffer::from_host(s, &f32s(&walk.w))?;
        let none = DeviceBuffer::from_host(s, &[0.0f32])?;
        let ps = DeviceBuffer::from_host(s, &[0u32, 1])?;
        let mut ys = DeviceBuffer::zeroed(s, 2 * K_CH)?;
        let mut st = DeviceBuffer::zeroed(s, ring.rows() * K_CH)?;
        let mut conv_at =
            |d: usize, base: usize, ch: usize, heads: usize, m: usize, taps: usize| {
                conv.enqueue(
                    s,
                    ConvResArgs {
                        ring: Ring::new(taps),
                        d,
                        base,
                        ch,
                        heads,
                        m,
                        x: &xs,
                        weights: &ws,
                        dynamic: &none,
                        pos: &ps,
                        fault: gpu.unlabelled_sink(),
                        scale: scale_static(),
                        y: &mut ys,
                        state: &mut st,
                    },
                )
            };
        out.push(refusal(
            "no channel",
            refused(conv_at(0, 0, K_CH, 0, 2, TAPS)),
            "d=0",
        ));
        out.push(refusal(
            "no token",
            refused(conv_at(K_CH, 0, K_CH, 0, 0, TAPS)),
            "m=0",
        ));
        out.push(refusal(
            "seventeen taps",
            refused(conv_at(K_CH, 0, K_CH, 0, 2, 17)),
            "at most",
        ));
        out.push(refusal(
            "three heads over 1,024 channels",
            refused(conv_at(K_CH, 0, K_CH, 3, 2, TAPS)),
            "divides",
        ));
        out.push(refusal(
            "a channel range outside the ring's rows",
            refused(conv_at(K_CH, RING_CH, RING_CH, 0, 2, TAPS)),
            "inside it",
        ));
        out.push(refusal(
            "a short ring",
            refused({
                let mut little = DeviceBuffer::zeroed(s, ring.rows() * K_CH - 1)?;
                conv.enqueue(
                    s,
                    ConvResArgs {
                        ring,
                        d: K_CH,
                        base: 0,
                        ch: K_CH,
                        heads: 0,
                        m: 2,
                        x: &xs,
                        weights: &ws,
                        dynamic: &none,
                        pos: &ps,
                        fault: gpu.unlabelled_sink(),
                        scale: scale_static(),
                        y: &mut ys,
                        state: &mut little,
                    },
                )
            }),
            "state",
        ));
        Ok(out)
    }

    // ------------------------------------------------------------------ the oracle clauses

    /// The fixture's tapped rows and f32 weights: one reader a fact.
    struct Oracle {
        man: RefManifest,
        split: Split,
        n_pos: usize,
    }

    impl Oracle {
        fn open() -> Result<Oracle, GateError> {
            let path = tier::fixture_set("the vendor clauses", &fixture::LCPP, A)?;
            let man = RefManifest::open(&path, &fixture::LCPP)?;
            let file = refset::fixture::first_shard("inkling2")
                .map_err(|e| format!("the fixture file: {e}"))?;
            let split = Split::open(&file).map_err(|e| format!("open {file}: {e}"))?;
            if split.architecture() != Some("inkling2") {
                return Err(format!(
                    "the fixture file is {:?}, want inkling2",
                    split.architecture()
                )
                .into());
            }
            let n_pos = man.tensor("l_out-5", 0)?.ne[1] as usize;
            Ok(Oracle { man, split, n_pos })
        }

        /// A tapped node's rows, `[n_pos][N]` (the dump's logical order: a position's row
        /// contiguous), its shape checked.
        fn tap(&self, name: &str, occ: u32) -> Result<Vec<f32>, GateError> {
            let row = self.man.tensor(name, occ)?;
            if row.ne[0] != N as u64 || row.ne[1] != self.n_pos as u64 {
                return Err(format!(
                    "{name} occurrence {occ} is {:?}, want [{N}, {}]",
                    row.ne, self.n_pos
                )
                .into());
            }
            Ok(ref_tensor_logical_in(&self.man.dir, row)?)
        }

        /// A tapped node's single row (the head's, which the reference's out-ids keep at the
        /// last position only), its shape checked `[N, 1]`.
        fn tap_one(&self, name: &str) -> Result<Vec<f32>, GateError> {
            let row = self.man.tensor(name, 0)?;
            if row.ne[0] != N as u64 || row.ne[1] != 1 {
                return Err(format!("{name} is {:?}, want [{N}, 1]", row.ne).into());
            }
            Ok(ref_tensor_logical_in(&self.man.dir, row)?)
        }

        /// An f32 tensor of the fixture file, its dims checked.
        fn w(&self, name: &str, want: &[u64]) -> Result<Vec<f32>, GateError> {
            let (shard, info) = self
                .split
                .find(name)
                .ok_or_else(|| format!("tensor {name} is not in the fixture file"))?;
            if info.ty != GgmlType::F32 || info.dims != want {
                return Err(format!(
                    "tensor {name} is {} {:?}, want F32 {want:?}",
                    info.ty, info.dims
                )
                .into());
            }
            let g = self
                .split
                .shard(shard)
                .ok_or_else(|| format!("tensor {name}: shard {shard} missing"))?;
            let b = g.data(info)?;
            Ok(b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        }

        /// The card's normed mix of tapped rows: `banks` and `cur` `[n_pos][N]`, the queries the
        /// weight rows, the output `[n_pos][nq · N]`.
        fn mix(
            &self,
            gpu: &Gpu,
            k: &AttnResKernels,
            banks: &[Vec<f32>],
            cur: &[f32],
            queries: &[f32],
        ) -> Result<Vec<f32>, GateError> {
            let s = gpu.stream();
            let m = self.n_pos;
            let stride = m * N;
            let nb = banks.len();
            let mut flat = Vec::with_capacity(nb * stride);
            for b in banks {
                flat.extend_from_slice(b);
            }
            // An empty bank set still uploads one value: the launch reads no bank row then.
            if flat.is_empty() {
                flat.push(0.0);
            }
            let banks = DeviceBuffer::from_host(s, &flat)?;
            let cur = DeviceBuffer::from_host(s, cur)?;
            let queries = DeviceBuffer::from_host(s, queries)?;
            let nq = queries.len() / N;
            let mut y = DeviceBuffer::zeroed(s, m * nq * N)?;
            k.enqueue_normed(
                s,
                AttnResNormedArgs {
                    banks: &banks,
                    n_banks: nb,
                    stride,
                    cur: &cur,
                    queries: &queries,
                    nq,
                    scale: SCALE,
                    eps: EPS,
                    n: N,
                    m,
                    fault: gpu.unlabelled_sink(),
                },
                &mut y,
            )?;
            s.synchronize()?;
            Ok(y.to_host_vec(s)?)
        }

        /// The card's controller over tapped rows, the output `[n_pos][N]`.
        #[allow(
            clippy::too_many_arguments,
            reason = "the chain's inputs, one argument a tensor"
        )]
        fn control(
            &self,
            gpu: &Gpu,
            k: &LoopControlKernels,
            cur: &[f32],
            msg: &[f32],
            down: &[f32],
            gain: &[f32],
            forget: &[f32],
        ) -> Result<Vec<f32>, GateError> {
            let s = gpu.stream();
            let m = self.n_pos;
            let cur = DeviceBuffer::from_host(s, cur)?;
            let msg = DeviceBuffer::from_host(s, msg)?;
            let down = DeviceBuffer::from_host(s, down)?;
            let gain = DeviceBuffer::from_host(s, gain)?;
            let forget = DeviceBuffer::from_host(s, forget)?;
            let mut y = DeviceBuffer::zeroed(s, m * N)?;
            let mut h = DeviceBuffer::zeroed(s, scratch_len(1, m, RANK))?;
            k.enqueue(
                s,
                LoopControlArgs {
                    cur: &cur,
                    msg: &msg,
                    down: &down,
                    gain: &gain,
                    forget: &forget,
                    n: N,
                    rank: RANK,
                    m,
                    eps: EPS,
                    fault: gpu.unlabelled_sink(),
                },
                &mut h,
                &mut y,
            )?;
            s.synchronize()?;
            Ok(y.to_host_vec(s)?)
        }
    }

    /// The vendor's RMS norm of one computed row in f64 (the mean in f64, the gain in f32).
    fn norm_gain(row: &[f32], gain: &[f32]) -> Vec<f32> {
        let ms = row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / N as f64;
        let s = 1.0 / (ms + f64::from(EPS)).sqrt();
        row.iter()
            .zip(gain)
            .map(|(&x, &g)| ((f64::from(x) * s) as f32) * g)
            .collect()
    }

    fn oracle_clauses(
        gpu: &Gpu,
        mix: &AttnResKernels,
        control: &LoopControlKernels,
    ) -> Result<Vec<bool>, GateError> {
        let o = Oracle::open()?;
        let m = o.n_pos;
        let emb = o.tap("norm", 0)?;
        let l5 = o.tap("l_out-5", 0)?;
        let l7 = o.tap("l_out-7", 0)?;
        let l10 = o.tap("l_out-10", 0)?;
        let l11 = o.tap("l_out-11", 0)?;
        let l12 = o.tap("l_out-12", 0)?;
        let l23 = o.tap("l_out-23", 0)?;
        let l24 = o.tap("l_out-24", 0)?;
        let ffn11 = o.tap("ffn_inp-11", 0)?;
        let ffn13 = o.tap("ffn_inp-13", 0)?;
        let mut out = Vec::new();

        // A computed row block against a tapped one, position by position.
        let compare =
            |what: &str, got: &[f32], tap_name: &str, occ: u32| -> Result<bool, GateError> {
                let tap = o.tap(tap_name, occ)?;
                let mut err = 0.0f64;
                for t in 0..m {
                    err = err.max(rel_l2(
                        &got[t * N..(t + 1) * N],
                        &f64s(&tap[t * N..(t + 1) * N]),
                    ));
                }
                let pass = err <= ORACLE_BAND;
                println!(
                    "  {what} against {tap_name} occurrence {occ}: rel-L2 {err:.3e} (band \
                 {ORACLE_BAND:.0e}): {}",
                    verdict(pass)
                );
                Ok(pass)
            };

        // Slot 8: a windowed mid-block visit of pass 0 — the attention mix and its input norm.
        let q8 = o.w("blk.8.attn_res_query.weight", &[N as u64])?;
        let y8 = o.mix(gpu, mix, &[emb.clone(), l5.clone()], &l7, &q8)?;
        let g8 = o.w("blk.8.attn_norm.weight", &[N as u64])?;
        let normed8: Vec<f32> = (0..m)
            .flat_map(|t| norm_gain(&y8[t * N..(t + 1) * N], &g8))
            .collect();
        out.push(compare(
            "mix_pass0 the slot-8 attention mix through its input norm",
            &normed8,
            "norm-8",
            0,
        )?);

        // Slot 11: a windowless visit of pass 0 — three queries, three input norms, then the ffn.
        let q11 = o.w("blk.11.attn_res_query.weight", &[N as u64, 3])?;
        let y11 = o.mix(gpu, mix, &[emb.clone(), l5.clone()], &l10, &q11)?;
        for (j, part) in ["q", "k", "v"].iter().enumerate() {
            let g = o.w(&format!("blk.11.attn_{part}_inp_norm.weight"), &[N as u64])?;
            let normed: Vec<f32> = (0..m)
                .flat_map(|t| norm_gain(&y11[t * 3 * N + j * N..t * 3 * N + (j + 1) * N], &g))
                .collect();
            out.push(compare(
                &format!("mix_pass0 the slot-11 {part} input through its input norm"),
                &normed,
                "norm-11",
                j as u32,
            )?);
        }
        let f11 = o.w("blk.11.ffn_res_query.weight", &[N as u64])?;
        let yf11 = o.mix(gpu, mix, &[emb.clone(), l5.clone()], &ffn11, &f11)?;
        let gf11 = o.w("blk.11.ffn_norm.weight", &[N as u64])?;
        let normedf11: Vec<f32> = (0..m)
            .flat_map(|t| norm_gain(&yf11[t * N..(t + 1) * N], &gf11))
            .collect();
        out.push(compare(
            "mix_pass0 the slot-11 feed-forward mix through the ffn norm",
            &normedf11,
            "norm-11",
            3,
        )?);

        // Slot 13: a pass-1 core visit — the message mixes, the attention mix, the controllers.
        // The fixture's message bank is empty (its pass 0 pushed one block) and its bank_last is
        // pass 0's last output; the two message tables' rows at 2·(ip − core_start) stage as one
        // four-query launch (rows 0 and 1 the attention table's, rows 2 and 3 the kv one's).
        let lr = o.w("loop_res_query.weight", &[N as u64, 12])?;
        let lk = o.w("loop_kv_res_query.weight", &[N as u64, 12])?;
        let row2 = 2 * (7 - CORE_START);
        let mut staged: Vec<f32> = Vec::with_capacity(4 * N);
        for (tab, r) in [(&lr, row2), (&lr, row2 + 1), (&lk, row2), (&lk, row2 + 1)] {
            staged.extend_from_slice(&tab[r * N..(r + 1) * N]);
        }
        let msg = o.mix(gpu, mix, &[], &l11, &staged)?;
        let msg_row = |q: usize| -> Vec<f32> {
            (0..m)
                .flat_map(|t| msg[t * 4 * N + q * N..t * 4 * N + (q + 1) * N].to_vec())
                .collect()
        };
        let q7 = o.w("blk.7.attn_res_query.weight", &[N as u64])?;
        let y13 = o.mix(gpu, mix, &[emb.clone(), l5.clone()], &l12, &q7)?;
        let (d0, g0, f0) = (
            o.w("blk.7.loop_attn_down.weight", &[2 * N as u64, RANK as u64])?,
            o.w("blk.7.loop_attn_gain.weight", &[RANK as u64, N as u64])?,
            o.w("blk.7.loop_attn_forget.weight", &[RANK as u64, N as u64])?,
        );
        let q_inp = o.control(gpu, control, &y13, &msg_row(0), &d0, &g0, &f0)?;
        out.push(compare(
            "mix_pass1 the slot-13 attention input through its controller",
            &q_inp,
            "loop_q_inp-13",
            0,
        )?);
        // A looped windowed visit norms its three controller outputs (the k, v and q inputs —
        // the dump numbers the three in its own eval order, not the build loop's q, k, v — so
        // occurrences 0 to 2 are the k, v and q inputs, each with the one attention norm's gain)
        // before the feed-forward norm (occurrence 3). The three controllers read the same
        // windowed mix row (one query) and the message tables' own rows.
        let g7 = o.w("blk.7.attn_norm.weight", &[N as u64])?;
        let (d1, g1w, f1w) = (
            o.w("blk.7.loop_k_down.weight", &[2 * N as u64, RANK as u64])?,
            o.w("blk.7.loop_k_gain.weight", &[RANK as u64, N as u64])?,
            o.w("blk.7.loop_k_forget.weight", &[RANK as u64, N as u64])?,
        );
        let k_inp = o.control(gpu, control, &y13, &msg_row(2), &d1, &g1w, &f1w)?;
        let k_normed: Vec<f32> = (0..m)
            .flat_map(|t| norm_gain(&k_inp[t * N..(t + 1) * N], &g7))
            .collect();
        out.push(compare(
            "mix_pass1 the slot-13 k input through its controller and the attention norm",
            &k_normed,
            "norm-13",
            0,
        )?);
        let (d2, g2w, f2w) = (
            o.w("blk.7.loop_v_down.weight", &[2 * N as u64, RANK as u64])?,
            o.w("blk.7.loop_v_gain.weight", &[RANK as u64, N as u64])?,
            o.w("blk.7.loop_v_forget.weight", &[RANK as u64, N as u64])?,
        );
        let v_inp = o.control(gpu, control, &y13, &msg_row(3), &d2, &g2w, &f2w)?;
        let v_normed: Vec<f32> = (0..m)
            .flat_map(|t| norm_gain(&v_inp[t * N..(t + 1) * N], &g7))
            .collect();
        out.push(compare(
            "mix_pass1 the slot-13 v input through its controller and the attention norm",
            &v_normed,
            "norm-13",
            1,
        )?);
        let q_normed: Vec<f32> = (0..m)
            .flat_map(|t| norm_gain(&q_inp[t * N..(t + 1) * N], &g7))
            .collect();
        out.push(compare(
            "mix_pass1 the slot-13 attention input through its norm",
            &q_normed,
            "norm-13",
            2,
        )?);
        let f7 = o.w("blk.7.ffn_res_query.weight", &[N as u64])?;
        let yf13 = o.mix(gpu, mix, &[emb.clone(), l5.clone()], &ffn13, &f7)?;
        let (d3, g3, f3) = (
            o.w("blk.7.loop_mlp_down.weight", &[2 * N as u64, RANK as u64])?,
            o.w("blk.7.loop_mlp_gain.weight", &[RANK as u64, N as u64])?,
            o.w("blk.7.loop_mlp_forget.weight", &[RANK as u64, N as u64])?,
        );
        let ffn_inp = o.control(gpu, control, &yf13, &msg_row(1), &d3, &g3, &f3)?;
        out.push(compare(
            "mix_pass1 the slot-13 feed-forward mix through its controller",
            &ffn_inp,
            "loop_ffn_inp-13",
            0,
        )?);
        let gf7 = o.w("blk.7.ffn_norm.weight", &[N as u64])?;
        let ffn_normed: Vec<f32> = (0..m)
            .flat_map(|t| norm_gain(&ffn_inp[t * N..(t + 1) * N], &gf7))
            .collect();
        out.push(compare(
            "mix_pass1 the slot-13 feed-forward mix through its norm",
            &ffn_normed,
            "norm-13",
            3,
        )?);

        // The head: the blocks the walk ends with — the prelude's, pass 2's boundary push (each
        // pass restarts from the prelude stream, so its own push is `l_out-5` again) and the
        // coda's — and the stream. `result_norm` holds the last position's row only (the head's
        // out-ids keep it alone).
        let qh = o.w("output_res_query.weight", &[N as u64])?;
        let head_banks = vec![emb, l5, l23];
        let yh = o.mix(gpu, mix, &head_banks, &l24, &qh)?;
        let gh = o.w("output_norm.weight", &[N as u64])?;
        let last = norm_gain(&yh[(m - 1) * N..], &gh);
        let head_ref = o.tap_one("result_norm")?;
        let err = rel_l2(&last, &f64s(&head_ref));
        let pass = err <= ORACLE_BAND;
        println!(
            "  mix_pass0 the head's mix over three blocks and the stream against result_norm: \
             rel-L2 {err:.3e} (band {ORACLE_BAND:.0e}): {}",
            verdict(pass)
        );
        out.push(pass);
        Ok(out)
    }

    // ------------------------------------------------------------------ run

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Fixture {
            return Err(format!(
                "{NAME}: the stages this gate replays read the fixture's taps and weights, so the \
                 gate runs the fixture tier: BLOOMERY_TIER=fixture just {RECIPE}"
            )
            .into());
        }
        let gpu = Gpu::new()?;
        println!(
            "gate_inkling2_step: device {} — the visit pipeline at {N} wide, rank {RANK}, {TAPS} \
             taps, seeded and the fixture's own nodes",
            gpu.device_name()?
        );
        let ctx = gpu.context();
        let mix = AttnResKernels::load(ctx)?;
        let control = LoopControlKernels::load(ctx)?;
        let conv = ConvResKernels::load(ctx)?;
        let clip = QkClipKernels::load(ctx)?;
        let tail = ConvTailKernels::load(ctx)?;
        let slots = CardSlotKernels::load(ctx)?;
        let mut results = Vec::new();
        results.extend(mix_clauses(&gpu, &mix)?);
        results.extend(mix_faults(&gpu, &mix)?);
        results.extend(mix_step_clauses(&gpu, &mix)?);
        results.extend(norm_proj_clauses(&gpu)?);
        results.extend(control_clauses(&gpu, &control)?);
        results.extend(conv_clauses(&gpu, &conv)?);
        results.extend(conv_clip_clauses(&gpu, &conv, &clip)?);
        results.extend(conv_tail_clauses(&gpu, (&conv, &slots, &tail))?);
        results.extend(refusal_clauses(&gpu, &mix, &control, &conv)?);
        results.extend(oracle_clauses(&gpu, &mix, &control)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "gate_inkling2_step: {} — {} clauses, {failed} failed",
            verdict(failed == 0),
            results.len()
        );
        if failed == 0 {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
