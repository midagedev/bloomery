//! GPU gate for DeepSeek-V4.1's MoE sub-layer as one chain piece
//! (`bloomery_gpu_deepseek41::chain::ffn::FfnPiece`, B5 G1) on the file's own
//! weights with seeded input: two seeds a layer, T = 1, every layer.
//!
//! PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
//!
//! Per layer the gate makes the layer's dense tensors resident with
//! `Weights::load_where` and uploads the card's routed stacks compacted in slot
//! order; the host tier is served in-process — a `Hybrid` over the boundary
//! with the V4.1 host computation (`HostRun`, the file's `HostLayer`s). Per
//! seed it injects the streams the sub-layer reads and its folded input
//! (`activations` of the model's width, four columns and one: the piece takes
//! any finite pair, its routed ids being the router's on the file's own
//! weights), runs the piece eagerly (the host serves the layer as the piece
//! hands it over) and checks one thing:
//!
//! **The piece is the op path, bit for bit.** The gate runs the same op
//! kernels itself from its own buffers — the norm, the router, the card's
//! routed experts on places it translates on the host from the slot map, the
//! shared expert, HC_PRE — computes the host share with its own `HostLayer`
//! call on its own activation, and the combine ([`combine_elem`]), HC_POST
//! and the fold on the host. Every buffer the piece leaves must equal it:
//! the activation in the handoff, the routing, the places, the card's slots
//! of h and down, the shared expert, the host's partial sum in the mapped
//! page, the combine, HC_PRE, the streams and the fold. This is the
//! composition's contract — the op kernels' arithmetic is their own gates'.
//!
//! The slot map is synthetic, not the gate placement's prefix: the 3090's prefix
//! of a few dozen of 384 experts would put one routed id of a token on the card
//! about one layer in six, so most layers would run only one side of the join.
//! Here each layer whose down stack is q4_K puts on the card half of the
//! experts the two seeds' tokens use — every token has ids on both sides — in
//! slots numbered against the id order, so the translation is not the
//! identity. The ids come from a pass over every layer before the map is
//! built: the op path's own norm and router on each seeded input. A layer of
//! another down (q5_K, layers 0 and 1 of the real file) is all host in this
//! gate's map: its uploads and its op-path composition are the q4_K triple's,
//! and the q5_K down's own dispatch is the lib gate's clause (`chain::ffn`'s
//! `hw_` test).
//!
//! The shared expert runs in the file's format, as the piece runs it: q8_0
//! gate·up on the f32 activation and a q8_0 down, or Q3_K gate·up on the
//! norm's q8_1 form and a Q4_K down on the q8_1 of its output (a Q5_K down
//! on the f32 output).
//!
//! Structure, per layer: the piece's launches captured and replayed for each
//! seed with the host serving each replay, every buffer bit-identical to the
//! eager run; the captured nodes by kind (kernels, copies — none: the handoff
//! launch writes the page —, the two memory-operation batches) equal to the
//! piece's own count.
//!
//! The join's instances, before the layers, on synthetic inputs of eight
//! slots (`FfnKernels::enqueue_post`, which picks the entry from the slot
//! count): `ds41_ffn_post_8` and `ds41_ffn_post_streams_8` against the host
//! rule `runtime::combine::combine` at eight slots with HC_POST and the fold on
//! the host, bit for bit, over places that put slot 7 on the card and NaN in
//! every slot off it; `ds41_ffn_post` on the first six slots against the
//! same rule at six and V4.1's `combine_elem`; 5, 7 and 9 slots refused by
//! name with nothing launched. The `_8` entries, the handoff's among them,
//! and the handoff's `_10` compile with no local depot.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_chain_ffn", gate::run())
}

#[cfg(feature = "deepseek41")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::sync::Arc;

    use bloomery_gpu::fused::FusedKernels;
    use bloomery_gpu::hc::{HC_MIX, HC_STREAMS, HcKernels, HcParams, HcPreArgs, HcPreScratch};
    use bloomery_gpu::host::run::HostRun;
    use bloomery_gpu::hybrid::{Boundary, BoundaryShape, HOST, Hybrid, SlotMap};
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError, NodeInfo, Q8Act};
    use bloomery_gpu_deepseek41::chain::ffn::{
        CardStacks, FfnIo, FfnKernels, FfnPiece, Post, combine_elem, host_run,
    };
    use bloomery_gpu_deepseek41::dense::{Dense, DenseKernels};
    use bloomery_gpu_deepseek41::experts::{ExpertGateUp, ExpertKernels};
    use bloomery_gpu_deepseek41::router::{N_EXPERT, N_USED, RouterKernels, RouterOut};
    use bloomery_gpu_gates::{
        GateError, activations, bits_equal, bytes_to_words, checks_failed, no_local_depot,
        ref_model_path, row_bytes, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer, sys};
    use gguf::Split;
    use gguf::quant::GgmlType;
    use model::Tensor2;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::{host, names};
    use model::moe::{HostLayer, HostScratch};
    use model::r8file::R8Source;

    /// The two seeds of a layer's inputs, T = 1 each (`activations`, offset
    /// by the layer).
    const SEEDS: [u32; 2] = [0xC4A1_0004, 0xC4A1_012D];

    /// The folded input's seed, offset from the streams'.
    const FOLD_SALT: u32 = 0x00F0_1D00;

    // ---------------------------------------------------------- the input

    /// What one layer of one seed holds for the piece: the streams the
    /// sub-layer reads and its folded input, the norm's.
    struct Seeded {
        streams: Vec<f32>,
        fold_in: Vec<f32>,
    }

    /// Layer `l`'s input of `seed`, `n` the model's width: four stream
    /// columns and one folded column.
    fn seeded(n: usize, seed: u32, l: usize) -> Result<Seeded, GateError> {
        let at = seed.wrapping_add(u32::try_from(l)?);
        Ok(Seeded {
            streams: activations(n, HC_STREAMS, at),
            fold_in: activations(n, 1, at.wrapping_add(FOLD_SALT)),
        })
    }

    // ---------------------------------------------------------- the file

    /// A tensor's file bytes, its type and dims checked.
    fn tensor<'a>(
        split: &'a Split,
        name: &str,
        ty: GgmlType,
        dims: &[u64],
    ) -> Result<&'a [u8], GateError> {
        let (shard, info) = split
            .find(name)
            .ok_or_else(|| format!("tensor {name} is not in the model"))?;
        if info.ty != ty || info.dims != dims {
            return Err(format!(
                "tensor {name} is {} {:?}, want {ty} {dims:?}",
                info.ty, info.dims
            )
            .into());
        }
        let g = split
            .shard(shard)
            .ok_or_else(|| format!("tensor {name}: shard {shard} missing"))?;
        Ok(g.data(info)?)
    }

    fn tensor_type(split: &Split, name: &str) -> Result<GgmlType, GateError> {
        Ok(split
            .find(name)
            .map(|(_, t)| t.ty)
            .ok_or_else(|| format!("tensor {name} is not in the model"))?)
    }

    // ------------------------------------------------------ the slot map

    /// The gate's slot map and, per layer, the experts its card stacks hold in
    /// slot order (module doc).
    fn synthetic_map(
        split: &Split,
        hp: &Hparams,
        ids: &[Vec<[u32; N_USED]>],
    ) -> Result<(SlotMap, Vec<Vec<u32>>), GateError> {
        let n_expert = hp.experts.n_expert;
        let mut rows = vec![HOST; hp.n_layer * n_expert];
        let mut cards = Vec::with_capacity(hp.n_layer);
        for (l, per_set) in ids.iter().enumerate() {
            if tensor_type(split, &names::ffn_down_exps(l))? != GgmlType::Q4_K {
                cards.push(Vec::new());
                continue;
            }
            // Per expert: Some(true) card, Some(false) host.
            let mut side: Vec<Option<bool>> = vec![None; n_expert];
            for tok in per_set {
                for (rank, &e) in tok.iter().enumerate() {
                    let s = &mut side[e as usize];
                    if s.is_none() {
                        *s = Some(rank % 2 == 0);
                    }
                }
            }
            for tok in per_set {
                if !has_side(&side, tok, false) {
                    side[tok[N_USED - 1] as usize] = Some(false);
                }
                if !has_side(&side, tok, true) {
                    side[tok[0] as usize] = Some(true);
                }
            }
            for tok in per_set {
                if !(has_side(&side, tok, true) && has_side(&side, tok, false)) {
                    return Err(format!(
                        "layer {l}: no split of the sets' ids {per_set:?} puts ids of every token \
                         on both sides"
                    )
                    .into());
                }
            }
            // Slots against the id order: the largest id first.
            let mut card: Vec<u32> = (0..n_expert as u32)
                .filter(|&e| side[e as usize] == Some(true))
                .collect();
            card.reverse();
            for (slot, &e) in card.iter().enumerate() {
                rows[l * n_expert + e as usize] = u32::try_from(slot)?;
            }
            cards.push(card);
        }
        Ok((SlotMap::from_rows(0..hp.n_layer, n_expert, rows)?, cards))
    }

    /// Whether one of `tok`'s ids is on the side `card` names.
    fn has_side(side: &[Option<bool>], tok: &[u32; N_USED], card: bool) -> bool {
        tok.iter().any(|&e| side[e as usize] == Some(card))
    }

    /// The routed stacks of `experts` (slot order), compacted and uploaded.
    struct Stacks {
        gate: DeviceTensor<u32>,
        up: DeviceTensor<u32>,
        down: DeviceTensor<u32>,
    }

    fn upload_stacks(
        split: &Split,
        stream: &CudaStream,
        hp: &Hparams,
        l: usize,
        experts: &[u32],
    ) -> Result<Stacks, GateError> {
        let (n, ff, ne) = (hp.n_embd, hp.experts.ff, hp.experts.n_expert as u64);
        let gu = |name: String| tensor(split, &name, GgmlType::Q3_K, &[n as u64, ff as u64, ne]);
        let (g, u) = (gu(names::ffn_gate_exps(l))?, gu(names::ffn_up_exps(l))?);
        let d = tensor(
            split,
            &names::ffn_down_exps(l),
            GgmlType::Q4_K,
            &[ff as u64, n as u64, ne],
        )?;
        let (rb_gu, rb_d) = (
            row_bytes(GgmlType::Q3_K, n)?,
            row_bytes(GgmlType::Q4_K, ff)?,
        );
        let pack = |b: &[u8], rows: usize, rb: usize| -> Vec<u32> {
            let mut out = Vec::with_capacity(experts.len() * rows * rb);
            for &e in experts {
                let e = e as usize;
                out.extend_from_slice(&b[e * rows * rb..(e + 1) * rows * rb]);
            }
            bytes_to_words(&out)
        };
        let k = experts.len();
        Ok(Stacks {
            gate: DeviceTensor::upload(stream, &pack(g, ff, rb_gu), k * ff, rb_gu / 4)?,
            up: DeviceTensor::upload(stream, &pack(u, ff, rb_gu), k * ff, rb_gu / 4)?,
            down: DeviceTensor::upload(stream, &pack(d, n, rb_d), k * n, rb_d / 4)?,
        })
    }

    /// Layer `l`'s dense tensors the piece reads, made resident from the file.
    fn layer_weights(stream: &CudaStream, split: &Split, l: usize) -> Result<Weights, GateError> {
        let keep = [
            names::ffn_norm(l),
            names::ffn_gate_inp(l),
            names::exp_probs_b(l),
            names::ffn_gate_shexp(l),
            names::ffn_up_shexp(l),
            names::ffn_down_shexp(l),
            names::hc_ffn_fn(l),
            names::hc_ffn_scale(l),
            names::hc_ffn_base(l),
        ];
        let w = Weights::load_where(stream, split, |name| keep.iter().any(|k| k == name))?;
        let missing: Vec<&String> = keep.iter().filter(|k| w.get(k).is_none()).collect();
        if !missing.is_empty() {
            return Err(format!("layer {l}: not in the file: {missing:?}").into());
        }
        Ok(w)
    }

    fn f32_buf<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceBuffer<f32>, GateError> {
        match w.get(name) {
            Some(DevWeight::F32 { w, .. }) => Ok(w.buf()),
            _ => Err(format!("{name} is not a resident f32 tensor").into()),
        }
    }

    fn f32_ten<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<f32>, GateError> {
        match w.get(name) {
            Some(DevWeight::F32 { w, .. }) => Ok(w),
            _ => Err(format!("{name} is not a resident f32 tensor").into()),
        }
    }

    // ---------------------------------------------------- the op path

    /// The gate's own kernels and buffers: the op path the piece must equal.
    struct Op {
        fused: FusedKernels,
        router: RouterKernels,
        experts: ExpertKernels,
        dense: DenseKernels,
        hc: HcKernels,
        x: DeviceBuffer<f32>,
        act_x: Q8Act,
        rout: RouterOut,
        sel: DeviceBuffer<u32>,
        h: DeviceBuffer<f32>,
        act_h: Q8Act,
        down: DeviceBuffer<f32>,
        sh_h: DeviceBuffer<f32>,
        /// `sh_h` in q8_1, for a K-quant shared down.
        act_sh: Q8Act,
        sh_y: DeviceBuffer<f32>,
        mixes: DeviceBuffer<f32>,
        hc_out: DeviceBuffer<f32>,
        hc_scratch: HcPreScratch,
        host: HostScratch,
    }

    impl Op {
        fn new(gpu: &Gpu, hp: &Hparams) -> Result<Op, GateError> {
            let (ctx, stream) = (gpu.context(), gpu.stream());
            let (n, ff) = (hp.n_embd, hp.experts.ff);
            let z = |k: usize| DeviceBuffer::<f32>::zeroed(stream, k);
            Ok(Op {
                fused: FusedKernels::load(ctx)?,
                router: RouterKernels::load(ctx)?,
                experts: ExpertKernels::load(ctx)?,
                dense: DenseKernels::load(ctx)?,
                hc: HcKernels::load(ctx)?,
                x: z(n)?,
                act_x: Q8Act::with_k(stream, 1, n)?,
                rout: RouterOut::new(stream)?,
                sel: DeviceBuffer::zeroed(stream, N_USED)?,
                h: z(N_USED * ff)?,
                act_h: Q8Act::with_k(stream, N_USED, ff)?,
                down: z(N_USED * n)?,
                sh_h: z(ff)?,
                act_sh: Q8Act::with_k(stream, 1, ff)?,
                sh_y: z(n)?,
                mixes: z(HC_MIX)?,
                hc_out: z(HC_MIX)?,
                hc_scratch: HcPreScratch::new(stream, HC_STREAMS * n)?,
                host: HostScratch::new(n, ff, N_USED)?,
            })
        }
    }

    /// What one run leaves: every buffer the pins read, on the host.
    #[derive(Clone)]
    struct Out {
        x: Vec<f32>,
        logits: Vec<f32>,
        probs: Vec<f32>,
        ids: Vec<u32>,
        w: Vec<f32>,
        sel: Vec<u32>,
        /// The card's slots of h and down, in slot order (others zero).
        h: Vec<f32>,
        down: Vec<f32>,
        sh_h: Vec<f32>,
        sh: Vec<f32>,
        hsum: Vec<f32>,
        y: Vec<f32>,
        hc: Vec<f32>,
        streams: Vec<f32>,
        fold: Option<Vec<f32>>,
    }

    /// Each field of `a` and `b`, in order, and whether its bits agree.
    fn same(a: &Out, b: &Out) -> Vec<(&'static str, bool)> {
        let u = |x: &[u32], y: &[u32]| x == y;
        vec![
            ("x", bits_equal(&a.x, &b.x)),
            ("logits", bits_equal(&a.logits, &b.logits)),
            ("probs", bits_equal(&a.probs, &b.probs)),
            ("ids", u(&a.ids, &b.ids)),
            ("weights", bits_equal(&a.w, &b.w)),
            ("sel", u(&a.sel, &b.sel)),
            ("h", bits_equal(&a.h, &b.h)),
            ("down", bits_equal(&a.down, &b.down)),
            ("shexp_h", bits_equal(&a.sh_h, &b.sh_h)),
            ("shexp", bits_equal(&a.sh, &b.sh)),
            ("hsum", bits_equal(&a.hsum, &b.hsum)),
            ("y", bits_equal(&a.y, &b.y)),
            ("hc", bits_equal(&a.hc, &b.hc)),
            ("streams", bits_equal(&a.streams, &b.streams)),
            (
                "fold",
                match (&a.fold, &b.fold) {
                    (Some(x), Some(y)) => bits_equal(x, y),
                    (None, None) => true,
                    _ => false,
                },
            ),
        ]
    }

    /// The first field that differs, `-` when none does.
    fn first_diff(a: &Out, b: &Out) -> &'static str {
        same(a, b)
            .into_iter()
            .find(|(_, ok)| !ok)
            .map_or("-", |(name, _)| name)
    }

    /// Keep the card's slots of a slot-major buffer (`k` per slot), zero the
    /// rest: a host slot's rows are whatever an earlier launch left.
    fn card_slots(v: &[f32], sel: &[u32], n_card: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; v.len()];
        for (j, &p) in sel.iter().enumerate() {
            if (p as usize) < n_card {
                out[j * k..(j + 1) * k].copy_from_slice(&v[j * k..(j + 1) * k]);
            }
        }
        out
    }

    /// The HC_POST of one value (the HC gate's transcription): `hc` in the
    /// kernel layout.
    fn post_elem(x: f32, r: [f32; 4], hc: &[f32]) -> [f32; 4] {
        let mut o = [0.0f32; 4];
        for (i, oi) in o.iter_mut().enumerate() {
            let mut s = x.mul_add(hc[4 + i], hc[8 + i] * r[0]);
            for (j, &rj) in r.iter().enumerate().skip(1) {
                s = hc[8 + 4 * j + i].mul_add(rj, s);
            }
            *oi = s;
        }
        o
    }

    /// The fold of four stream values by `pre`.
    fn fold_elem(o: [f32; 4], pre: &[f32]) -> f32 {
        let mut y = o[0] * pre[0];
        for (&oj, &pj) in o.iter().zip(pre).skip(1) {
            y = oj.mul_add(pj, y);
        }
        y
    }

    fn streams_at(s: &[f32], n: usize, d: usize) -> [f32; 4] {
        [s[d], s[d + n], s[d + 2 * n], s[d + 3 * n]]
    }

    /// Everything the op path needs for one layer.
    struct LayerCx<'a> {
        l: usize,
        w: &'a Weights,
        stacks: Option<&'a Stacks>,
        /// The layer's row of the stage card's copy: each expert's card slot or HOST.
        row: &'a [u32],
        n_card: usize,
        host: Option<&'a HostLayer>,
        fold: bool,
    }

    /// The op path's routing of the folded input `fold_in` at layer `l`: the
    /// norm, then the router, into `op.act_x`, `op.x` and `op.rout`.
    fn route(
        gpu: &Gpu,
        op: &mut Op,
        hp: &Hparams,
        w: &Weights,
        l: usize,
        fold_in: &DeviceBuffer<f32>,
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        op.fused.enqueue_norm_quant(
            stream,
            fold_in,
            f32_buf(w, &names::ffn_norm(l))?,
            hp.rms_eps,
            &mut op.act_x,
            &mut op.x,
            gpu.layer_sink(l)?,
        )?;
        op.router.enqueue_router(
            stream,
            f32_ten(w, &names::ffn_gate_inp(l))?,
            &op.x,
            f32_buf(w, &names::exp_probs_b(l))?,
            hp.experts.routed_scale,
            &mut op.rout,
            gpu.layer_sink(l)?,
        )?;
        Ok(())
    }

    /// The op path for the injected inputs already in `streams`/`fold_in`.
    fn op_path(
        gpu: &Gpu,
        op: &mut Op,
        split: &Split,
        hp: &Hparams,
        lc: &LayerCx<'_>,
        streams: &DeviceBuffer<f32>,
        fold_in: &DeviceBuffer<f32>,
    ) -> Result<Out, GateError> {
        let stream = gpu.stream();
        let (n, ff, l, w) = (hp.n_embd, hp.experts.ff, lc.l, lc.w);
        let kind = &hp.layers[l];
        route(gpu, op, hp, w, l, fold_in)?;
        let ids = op.rout.ids.to_host_vec(stream)?;
        let weights = op.rout.weights.to_host_vec(stream)?;
        let sel = ids
            .iter()
            .map(|&id| {
                lc.row.get(id as usize).copied().ok_or_else(|| {
                    format!(
                        "layer {l}: the router picked expert {id}, past the slot map's {} entries",
                        lc.row.len()
                    )
                })
            })
            .collect::<Result<Vec<u32>, String>>()?;
        op.sel.copy_from_host(stream, &sel)?;
        if let Some(s) = lc.stacks {
            let args = ExpertGateUp {
                wg: &s.gate,
                wu: &s.up,
                act: &op.act_x,
                sel: &op.sel,
                n_slots: N_USED,
                rows_per_expert: ff,
                limit: kind.swiglu_limit,
                fault: gpu.unlabelled_sink(),
            };
            op.experts
                .enqueue_expert_gate_up(stream, &args, &mut op.h)?;
            gpu.enqueue_quantize_q8_1(&op.h, &mut op.act_h)?;
            gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                &s.down,
                &op.act_h,
                &op.sel,
                N_USED,
                n,
                &mut op.down,
            )?;
        }
        let (Some(sg), Some(su)) = (
            w.get(&names::ffn_gate_shexp(l)),
            w.get(&names::ffn_up_shexp(l)),
        ) else {
            return Err(format!("layer {l}: the shared gate and up are not resident").into());
        };
        let limit = kind.swiglu_limit_shared;
        match (sg, su) {
            (
                DevWeight::KQuant {
                    ty: GgmlType::Q3_K,
                    w: g,
                    ..
                },
                DevWeight::KQuant {
                    ty: GgmlType::Q3_K,
                    w: u,
                    ..
                },
            ) => op.dense.enqueue_shexp_gate_up_q3k(
                stream,
                g,
                u,
                &op.act_x,
                limit,
                gpu.unlabelled_sink(),
                &mut op.sh_h,
            )?,
            (g, u) => op.experts.enqueue_shexp_gate_up(
                stream,
                g,
                u,
                &op.x,
                limit,
                gpu.unlabelled_sink(),
                &mut op.sh_h,
            )?,
        }
        let sh_down = Dense::of(w, &names::ffn_down_shexp(l), ff, n)?;
        let act = if sh_down.reads_q8_1() {
            gpu.enqueue_quantize_q8_1(&op.sh_h, &mut op.act_sh)?;
            Some(&op.act_sh)
        } else {
            None
        };
        op.dense
            .enqueue(gpu, sh_down, &op.sh_h, act, &mut op.sh_y)?;
        let Some(DevWeight::KQuant { w: hc_w, .. }) = w.get(&names::hc_ffn_fn(l)) else {
            return Err(format!("layer {l}: hc_ffn_fn is not resident as a K-quant").into());
        };
        let params = HcParams {
            w: hc_w,
            scale: f32_buf(w, &names::hc_ffn_scale(l))?,
            base: f32_buf(w, &names::hc_ffn_base(l))?,
            eps: hp.hc.eps,
            iters: u32::try_from(hp.hc.sinkhorn_iters)?,
        };
        let pre = HcPreArgs {
            params: &params,
            x: streams,
            tokens: 1,
            rms_eps: hp.rms_eps,
            fault: gpu.layer_sink(l)?,
        };
        op.hc.enqueue_pre(
            stream,
            &pre,
            &mut op.hc_scratch,
            &mut op.mixes,
            &mut op.hc_out,
        )?;
        stream.synchronize()?;

        let x = op.x.to_host_vec(stream)?;
        let h = card_slots(&op.h.to_host_vec(stream)?, &sel, lc.n_card, ff);
        let down = card_slots(&op.down.to_host_vec(stream)?, &sel, lc.n_card, n);
        let sh_h = op.sh_h.to_host_vec(stream)?;
        let sh = op.sh_y.to_host_vec(stream)?;
        let hc = op.hc_out.to_host_vec(stream)?;
        let s_in = streams.to_host_vec(stream)?;

        // The host's share: the slots the map sends to the host, in slot
        // order, with the op path's own activation.
        let list: Vec<(u32, f32)> = (0..N_USED)
            .filter(|&j| sel[j] as usize >= lc.n_card)
            .map(|j| (ids[j], weights[j]))
            .collect();
        let mut hsum = vec![0.0f32; n];
        let hl = lc
            .host
            .ok_or_else(|| format!("layer {l}: no host view of a routed layer"))?;
        let xt = Tensor2::from_vec(n, 1, x.clone());
        hl.experts_into(R8Source::rows(split), &xt, &list, &mut hsum, &mut op.host)?;

        let card: [bool; N_USED] = std::array::from_fn(|j| (sel[j] as usize) < lc.n_card);
        let wv: [f32; N_USED] = std::array::from_fn(|j| if card[j] { weights[j] } else { 0.0 });
        let y: Vec<f32> = (0..n)
            .map(|dd| {
                let dv: [f32; N_USED] =
                    std::array::from_fn(|j| if card[j] { down[j * n + dd] } else { 0.0 });
                combine_elem(dv, wv, card, hsum[dd], sh[dd])
            })
            .collect();
        let mut st = vec![0.0f32; HC_STREAMS * n];
        let mut fold = lc.fold.then(|| vec![0.0f32; n]);
        for dd in 0..n {
            let o = post_elem(y[dd], streams_at(&s_in, n, dd), &hc);
            for (i, &oi) in o.iter().enumerate() {
                st[i * n + dd] = oi;
            }
            if let Some(f) = fold.as_mut() {
                f[dd] = fold_elem(o, &hc[..4]);
            }
        }
        let out = Out {
            x,
            logits: op.rout.logits.to_host_vec(stream)?,
            probs: op.rout.probs.to_host_vec(stream)?,
            ids,
            w: weights,
            sel,
            h,
            down,
            sh_h,
            sh,
            hsum,
            y,
            hc,
            streams: st,
            fold,
        };
        Ok(out)
    }

    // ------------------------------------------------------ the piece

    /// The device buffers the piece shares with the step, the gate's own.
    struct Io {
        streams: DeviceBuffer<f32>,
        fold_in: DeviceBuffer<f32>,
        streams_out: DeviceBuffer<f32>,
        fold_out: DeviceBuffer<f32>,
        slots: DeviceTensor<u32>,
    }

    /// One eager run of the piece at layer `lc.l`, its buffers read back.
    fn run_piece(
        gpu: &Gpu,
        piece: &mut FfnPiece,
        hybrid: &mut Hybrid<HostRun>,
        io: &mut Io,
        lc: &LayerCx<'_>,
    ) -> Result<Out, GateError> {
        let stream = gpu.stream();
        hybrid.begin_chain(stream)?;
        enqueue(gpu, piece, hybrid, io, lc)?;
        stream.synchronize()?;
        read_piece(gpu, piece, hybrid, io, lc)
    }

    fn enqueue(
        gpu: &Gpu,
        piece: &mut FfnPiece,
        hybrid: &mut Hybrid<HostRun>,
        io: &mut Io,
        lc: &LayerCx<'_>,
    ) -> Result<(), bloomery_gpu::GpuError> {
        let card = lc
            .stacks
            .map(|s| CardStacks::fused_over(&s.gate, &s.up, &s.down, GgmlType::Q4_K));
        let fio = FfnIo {
            streams: &io.streams,
            fold_in: &io.fold_in,
            streams_out: &mut io.streams_out,
            fold_out: lc.fold.then_some(&mut io.fold_out),
            slots: &io.slots,
        };
        piece.enqueue(gpu, lc.w, card, fio, hybrid, lc.l)
    }

    fn read_piece(
        gpu: &Gpu,
        piece: &FfnPiece,
        hybrid: &Hybrid<HostRun>,
        io: &Io,
        lc: &LayerCx<'_>,
    ) -> Result<Out, GateError> {
        let stream = gpu.stream();
        let t = piece.taps();
        let sel = t.sel.to_host_vec(stream)?;
        let (n, ff) = (io.fold_in.len(), t.shexp_h.len());
        Ok(Out {
            x: hybrid.boundary().normed().to_host_vec(stream)?,
            logits: t.router.logits.to_host_vec(stream)?,
            probs: t.router.probs.to_host_vec(stream)?,
            ids: t.router.ids.to_host_vec(stream)?,
            w: t.router.weights.to_host_vec(stream)?,
            h: card_slots(&t.h.to_host_vec(stream)?, &sel, lc.n_card, ff),
            down: card_slots(&t.down.to_host_vec(stream)?, &sel, lc.n_card, n),
            sel,
            sh_h: t.shexp_h.to_host_vec(stream)?,
            sh: t.shexp.to_host_vec(stream)?,
            hsum: hybrid.hsum_copy()?,
            y: t.y.to_host_vec(stream)?,
            hc: t.hc.to_host_vec(stream)?,
            streams: io.streams_out.to_host_vec(stream)?,
            fold: if lc.fold {
                Some(io.fold_out.to_host_vec(stream)?)
            } else {
                None
            },
        })
    }

    // -------------------------------------------------------- structure

    /// Node counts by kind: kernels, memcpy, batch memory operations, other.
    fn kinds(nodes: &[NodeInfo]) -> [usize; 4] {
        let mut c = [0usize; 4];
        for node in nodes {
            let i = match node.kind {
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL => 0,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY => 1,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP => 2,
                _ => 3,
            };
            c[i] += 1;
        }
        c
    }

    fn inject(stream: &CudaStream, io: &mut Io, s: &Seeded) -> Result<(), GateError> {
        io.streams.copy_from_host(stream, &s.streams)?;
        io.fold_in.copy_from_host(stream, &s.fold_in)?;
        Ok(())
    }

    // ------------------------------------------------ the 8-slot instances

    /// Slots a token of the join's second instance (`runtime::combine::Slots`).
    const SLOTS8: usize = 8;
    /// Values of the synthetic join: two blocks, the second partial.
    const ROWS8: usize = 300;
    /// What an output holds before a launch, so a value the launch leaves
    /// alone reads back as these bits.
    const SENT: f32 = 1.0e30;
    /// The synthetic join's places against [`N_CARD8`]: the card's at slots
    /// 0, 2, 3, 5 and 7 (slot 7 past a six-slot walk), place 9 (not below
    /// the card count) at slot 4, [`HOST`] at slots 1 and 6.
    const PLACES8: [u32; SLOTS8] = [0, HOST, 2, 1, 9, 5, HOST, 3];
    const N_CARD8: usize = 6;

    /// The synthetic join's inputs, [`SLOTS8`] slots of [`ROWS8`] values in
    /// the layout `FfnKernels::enqueue_post` reads. A slot off the card holds
    /// NaN in its down rows and its weight, so a read of it poisons the value;
    /// slot `j`'s down rows scale by `4^(j mod 4)`, so the slots' sum rounds
    /// differently in another order; every third value's host and shared
    /// sums are `1e8` and `−1e8`, so another association of the join moves
    /// its bits.
    struct Join8 {
        down: Vec<f32>,
        w: Vec<f32>,
        hsum: Vec<f32>,
        shexp: Vec<f32>,
        res: Vec<f32>,
        hc: Vec<f32>,
    }

    fn join8_inputs() -> Join8 {
        let on = |j: usize| (PLACES8[j] as usize) < N_CARD8;
        let raw = activations(ROWS8, SLOTS8, 801);
        let down = (0..SLOTS8 * ROWS8)
            .map(|i| {
                let j = i / ROWS8;
                if on(j) {
                    raw[i] * (1u32 << (2 * (j % 4))) as f32
                } else {
                    f32::NAN
                }
            })
            .collect();
        let wr = activations(SLOTS8, 1, 802);
        let w = (0..SLOTS8)
            .map(|j| if on(j) { wr[j] } else { f32::NAN })
            .collect();
        let (hr, sr) = (activations(ROWS8, 1, 803), activations(ROWS8, 1, 804));
        let hsum = (0..ROWS8)
            .map(|d| if d % 3 == 0 { 1.0e8 } else { hr[d] })
            .collect();
        let shexp = (0..ROWS8)
            .map(|d| if d % 3 == 0 { -1.0e8 } else { sr[d] })
            .collect();
        Join8 {
            down,
            w,
            hsum,
            shexp,
            res: activations(ROWS8, HC_STREAMS, 805),
            hc: activations(HC_MIX, 1, 806),
        }
    }

    /// Value `d`'s combine over the first `N` slots of `j`, by the host rule
    /// `runtime::combine::combine`.
    fn join8_y<const N: usize>(j: &Join8, d: usize) -> f32 {
        let card: [bool; N] = std::array::from_fn(|s| (PLACES8[s] as usize) < N_CARD8);
        let dv: [f32; N] = std::array::from_fn(|s| j.down[s * ROWS8 + d]);
        let wv: [f32; N] = std::array::from_fn(|s| j.w[s]);
        runtime::combine::combine(dv, wv, card, j.hsum[d], j.shexp[d])
    }

    /// `FfnKernels::enqueue_post` over the synthetic join: the `_8` entries
    /// (with the fold and without) against `runtime::combine::combine::<8>`,
    /// HC_POST and the fold on the host, bit for bit, and the form without
    /// the fold leaving its fold buffer alone; the six-slot entry on the
    /// first six slots against the same rule at six and V4.1's
    /// `combine_elem`; 5, 7 and 9 slots refused by name before any launch.
    fn join8_cases(gpu: &Gpu) -> Result<bool, GateError> {
        let kernels = FfnKernels::load(gpu.context())?;
        let stream = gpu.stream();
        let j = join8_inputs();
        let sel = DeviceBuffer::from_host(stream, &PLACES8)?;
        let dev = |v: &[f32]| DeviceBuffer::from_host(stream, v);
        let (down, w, hsum, shexp, res, hc) = (
            dev(&j.down)?,
            dev(&j.w)?,
            dev(&j.hsum)?,
            dev(&j.shexp)?,
            dev(&j.res)?,
            dev(&j.hc)?,
        );
        let post = |n_used: usize| Post {
            down: &down,
            w: &w,
            sel: &sel,
            hsum: &hsum,
            shexp: &shexp,
            res: &res,
            hc: &hc,
            rows: ROWS8,
            n_card: N_CARD8,
            n_used,
        };
        let mut ok = true;
        for (n_used, folds) in [(SLOTS8, true), (SLOTS8, false), (N_USED, true)] {
            let mut y = dev(&[SENT; ROWS8])?;
            let mut out = dev(&vec![SENT; HC_STREAMS * ROWS8])?;
            let mut fold = dev(&[SENT; ROWS8])?;
            kernels.enqueue_post(
                stream,
                &post(n_used),
                &mut y,
                &mut out,
                folds.then_some(&mut fold),
            )?;
            let (y, out, fold) = (
                y.to_host_vec(stream)?,
                out.to_host_vec(stream)?,
                fold.to_host_vec(stream)?,
            );
            stream.synchronize()?;
            let want_y: Vec<f32> = (0..ROWS8)
                .map(|d| {
                    if n_used == SLOTS8 {
                        join8_y::<SLOTS8>(&j, d)
                    } else {
                        join8_y::<N_USED>(&j, d)
                    }
                })
                .collect();
            // At six slots the host rule is V4.1's `combine_elem` too.
            let v41_ok = n_used == SLOTS8
                || (0..ROWS8).all(|d| {
                    let card: [bool; N_USED] =
                        std::array::from_fn(|s| (PLACES8[s] as usize) < N_CARD8);
                    let dv: [f32; N_USED] = std::array::from_fn(|s| j.down[s * ROWS8 + d]);
                    let wv: [f32; N_USED] = std::array::from_fn(|s| j.w[s]);
                    combine_elem(dv, wv, card, j.hsum[d], j.shexp[d]).to_bits()
                        == want_y[d].to_bits()
                });
            let mut want_out = vec![0.0f32; HC_STREAMS * ROWS8];
            let mut want_fold = vec![SENT; ROWS8];
            for d in 0..ROWS8 {
                let o = post_elem(want_y[d], streams_at(&j.res, ROWS8, d), &j.hc);
                for (i, &oi) in o.iter().enumerate() {
                    want_out[i * ROWS8 + d] = oi;
                }
                if folds {
                    want_fold[d] = fold_elem(o, &j.hc[..4]);
                }
            }
            let (y_ok, out_ok, fold_ok) = (
                bits_equal(&y, &want_y),
                bits_equal(&out, &want_out),
                bits_equal(&fold, &want_fold),
            );
            let bad_y = (0..ROWS8)
                .filter(|&d| y[d].to_bits() != want_y[d].to_bits())
                .count();
            let pass = y_ok && out_ok && fold_ok && v41_ok;
            ok &= pass;
            println!(
                "join{n_used} {}: places {PLACES8:?} n_card {N_CARD8}, {ROWS8} values: y {} ({bad_y} \
                 differ) streams {} fold {} v41_rule {}: {}",
                if folds { "post+fold" } else { "post" },
                verdict(y_ok),
                verdict(out_ok),
                if folds {
                    verdict(fold_ok)
                } else if fold_ok {
                    "untouched"
                } else {
                    "WRITTEN"
                },
                verdict(v41_ok),
                verdict(pass)
            );
        }
        for n_used in [5, 7, 9] {
            let mut y = dev(&[SENT; ROWS8])?;
            let mut out = dev(&vec![SENT; HC_STREAMS * ROWS8])?;
            let got = kernels.enqueue_post(stream, &post(n_used), &mut y, &mut out, None);
            let y = y.to_host_vec(stream)?;
            stream.synchronize()?;
            let refused = matches!(
                &got,
                Err(GpuError::Shape { what: "ds41_ffn_post", detail })
                    if detail.starts_with(&format!("{n_used} routed slots"))
            );
            let untouched = y.iter().all(|v| v.to_bits() == SENT.to_bits());
            let pass = refused && untouched;
            ok &= pass;
            println!(
                "join{n_used}: {}; y untouched {untouched}: {}",
                got.err()
                    .map_or_else(|| "launched".to_string(), |e| format!("refused ({e})")),
                verdict(pass)
            );
        }
        Ok(ok)
    }

    // ---------------------------------------------------------- driver

    #[derive(Default)]
    struct Tally {
        checks: u32,
        failed: u32,
    }

    impl Tally {
        fn add(&mut self, pass: bool) {
            self.checks += 1;
            if !pass {
                self.failed += 1;
            }
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[bloomery_levers::R8])?;
        let r8 = levers.host().r8;
        let path = ref_model_path()?;
        let split = Split::open(&path)?;
        // Check 1 is the one real-model clause that holds the r8 path to the
        // source bit for bit: with BLOOMERY_R8 on, no sidecar refuses the run
        // by name. Held here, so the tiers below read this open of it.
        let _r8 = model::r8file::HostR8::at_gate(&split, r8)?;
        let hp = Hparams::read(&split)?;
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let (n, ff) = (hp.n_embd, hp.experts.ff);
        println!(
            "gate_deepseek41_chain_ffn: device {} — layers {} n_embd {n} ff {ff} experts {} used {} \
             scale {} streams {}",
            gpu.device_name()?,
            hp.n_layer,
            hp.experts.n_expert,
            hp.experts.n_used,
            hp.experts.routed_scale,
            hp.hc.streams
        );

        // Every layer's routed ids on the seeded inputs first, from the op
        // path's own norm and router: the map is built from them.
        let mut op = Op::new(&gpu, &hp)?;
        let mut ids_per_layer: Vec<Vec<[u32; N_USED]>> = Vec::with_capacity(hp.n_layer);
        for l in 0..hp.n_layer {
            let w = layer_weights(stream, &split, l)?;
            let mut per = Vec::with_capacity(SEEDS.len());
            for seed in SEEDS {
                let fold_in = DeviceBuffer::from_host(stream, &seeded(n, seed, l)?.fold_in)?;
                route(&gpu, &mut op, &hp, &w, l, &fold_in)?;
                let ids = op.rout.ids.to_host_vec(stream)?;
                if let Some(bad) = ids.iter().find(|&&id| id as usize >= N_EXPERT) {
                    return Err(format!(
                        "seed {seed:#x}: layer {l}: the router picked expert {bad}, past the file's {N_EXPERT}"
                    )
                    .into());
                }
                per.push(<[u32; N_USED]>::try_from(ids).map_err(|v| {
                    format!(
                        "seed {seed:#x}: layer {l}: the router wrote {} ids",
                        v.len()
                    )
                })?);
            }
            ids_per_layer.push(per);
        }
        let (map, cards) = synthetic_map(&split, &hp, &ids_per_layer)?;
        let host_layers: Vec<Option<HostLayer>> = (0..hp.n_layer)
            .map(|l| host::layer(R8Source::rows(&split), &hp, l))
            .collect::<Result<_, _>>()?;

        let shape = BoundaryShape {
            hidden: n,
            n_used: N_USED,
        };
        let boundary = Boundary::new(gpu.context(), stream, shape)?;
        let tier = host_run(Arc::new(Split::open(&path)?), &hp, 0..hp.n_layer, r8)?;
        let mut hybrid = Hybrid::new(boundary, map.clone(), tier, hp.n_layer)?;
        let mut piece = FfnPiece::new(&gpu, &hp, &map)?;
        let mut io = Io {
            streams: DeviceBuffer::zeroed(stream, HC_STREAMS * n)?,
            fold_in: DeviceBuffer::zeroed(stream, n)?,
            streams_out: DeviceBuffer::zeroed(stream, HC_STREAMS * n)?,
            fold_out: DeviceBuffer::zeroed(stream, n)?,
            slots: DeviceTensor::upload(
                stream,
                &map.stage_view(),
                hp.n_layer,
                hp.experts.n_expert,
            )?,
        };
        println!(
            "piece: layers {:?} scratch_bytes={} map: {} layers with card experts, {} all-host",
            piece.layers(),
            piece.device_bytes(),
            cards.iter().filter(|c| !c.is_empty()).count(),
            cards.iter().filter(|c| c.is_empty()).count()
        );

        let mut tally = Tally::default();
        tally.add(no_local_depot(&[
            "ds41_ffn_handoff",
            "ds41_ffn_post",
            "ds41_ffn_post_streams",
            "ds41_ffn_handoff_8",
            "ds41_ffn_post_8",
            "ds41_ffn_post_streams_8",
            "ds41_ffn_handoff_10",
        ])?);
        tally.add(join8_cases(&gpu)?);
        let stage = map.stage_view();
        for l in 0..hp.n_layer {
            let w = layer_weights(stream, &split, l)?;
            let stacks = if cards[l].is_empty() {
                None
            } else {
                Some(upload_stacks(&split, stream, &hp, l, &cards[l])?)
            };
            let fold = piece
                .folds(l)
                .ok_or_else(|| format!("layer {l} is outside the piece"))?;
            let row = map
                .row_offset(l)
                .and_then(|at| stage.get(at..at + hp.experts.n_expert))
                .ok_or_else(|| format!("layer {l} has no map row"))?;
            let lc = LayerCx {
                l,
                w: &w,
                stacks: stacks.as_ref(),
                row,
                n_card: map.on_card(l)?,
                host: host_layers[l].as_ref(),
                fold,
            };
            let inputs: Vec<Seeded> = SEEDS
                .iter()
                .map(|&s| seeded(n, s, l))
                .collect::<Result<_, _>>()?;
            let mut eager = Vec::with_capacity(inputs.len());
            for (seed, input) in SEEDS.iter().zip(&inputs) {
                inject(stream, &mut io, input)?;
                let got = run_piece(&gpu, &mut piece, &mut hybrid, &mut io, &lc)?;
                let want = op_path(&gpu, &mut op, &split, &hp, &lc, &io.streams, &io.fold_in)?;
                let diff = first_diff(&got, &want);
                let pass = diff == "-";
                let kind = if lc.n_card > 0 {
                    format!("card={}", lc.n_card)
                } else {
                    "host-only".to_string()
                };
                let places: Vec<String> = got
                    .sel
                    .iter()
                    .map(|&p| {
                        if p == HOST {
                            "H".to_string()
                        } else {
                            p.to_string()
                        }
                    })
                    .collect();
                // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
                println!(
                    "ffn seed={seed:#x} L={l} {kind} form={} places={} ids={:?} op_path_bit={pass} \
                     first_diff={diff} {}",
                    if fold { "post+fold" } else { "post" },
                    places.join(","),
                    got.ids,
                    verdict(pass)
                );
                tally.add(pass);
                eager.push(got);
            }

            // Structure: capture, replay per seed with the host serving.
            let g = gpu.capture(|_| {
                hybrid.begin_chain(stream)?;
                enqueue(&gpu, &mut piece, &mut hybrid, &mut io, &lc)
            })?;
            let k = kinds(&g.nodes()?);
            let want_kernels = piece
                .launches(l)
                .ok_or_else(|| format!("layer {l} is outside the piece"))?;
            // The piece copies nothing: its handoff launch writes the page.
            let want = [want_kernels, 0, 2, 0];
            let mut replay_eq = true;
            for si in (0..inputs.len()).rev() {
                inject(stream, &mut io, &inputs[si])?;
                g.launch(stream)?;
                hybrid.serve_captured()?;
                stream.synchronize()?;
                let r = read_piece(&gpu, &piece, &hybrid, &io, &lc)?;
                replay_eq &= first_diff(&r, &eager[si]) == "-";
            }
            drop(g);
            let nodes_ok = k == want;
            let line = format!(
                "graph L={l} nodes={} kernels={} memcpy={} memops={} other={} (want {}/{}/{}/{}) \
                 replays_eq_eager={replay_eq}",
                k.iter().sum::<usize>(),
                k[0],
                k[1],
                k[2],
                k[3],
                want[0],
                want[1],
                want[2],
                want[3]
            );
            let pass = nodes_ok && replay_eq;
            println!("{line} {}", verdict(pass));
            tally.add(pass);
        }
        let stats = hybrid.stats();
        let pass = tally.failed == 0;
        println!(
            "gate_deepseek41_chain_ffn: {} checks across {} seeds and {} layers, {} failed — \
             host served {} layers, {} slots — {}",
            tally.checks,
            SEEDS.len(),
            hp.n_layer,
            tally.failed,
            stats.served,
            stats.host_slots,
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }
}
