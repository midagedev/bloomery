//! One pass of the encoder over an input's ids (the crate doc's order), its
//! launches all on the [`Gpu`]'s stream, and the rows it reads back.

use std::collections::BTreeMap;

use crate::config::{DenseHead, Hparams};
use bloomery_gpu::flash_gqa_prefill::GqaPrefillArgs;
use bloomery_gpu::gemm::{GemmAct32, GemmInput, GemmRoute};
use bloomery_gpu::rope_neox::{HEAD_256, PartialNeoxArgs};
use bloomery_gpu::rope_table::RopeRows;
use bloomery_gpu::site::{self, SiteTy, SiteWeight, WideIn, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, FaultSink, Gpu, GpuError, Window, to_host};
use cuda_core::DeviceBuffer;
use model::arch::gemma_embedding2::hparams::key_bounds;
use model::arch::gemma_embedding2::names;

use crate::{Attn, Consts, Encoder, Kernels, Scratch, decode_rows, reaches};

/// The refusal `what` of a pass.
const PASS: &str = "Encoder::pass";

/// One pass read back: the mean of every position's `output` projection
/// (`pooled`, `n_out` f32), and the taps when asked.
pub struct Pass {
    pub pooled: Vec<f32>,
    pub taps: Option<Taps>,
}

/// A pass's intermediate rows, each token-major in the reference's order.
pub struct Taps {
    /// The token rows times `√hidden`, `n · hidden`.
    pub inp_scaled: Vec<f32>,
    /// Every layer's normed input slice, `[per_layer, n_layer, n]`: position
    /// `t`'s layer `l` at `(t·n_layer + l)·per_layer`.
    pub inp_per_layer: Vec<f32>,
    /// Each layer's output, `n · hidden`.
    pub l_out: Vec<Vec<f32>>,
    /// The output norm's rows, `n · hidden`.
    pub result_norm: Vec<f32>,
    /// The `output` projection of every position, `n · n_out`.
    pub result_embd: Vec<f32>,
}

/// What a projection reads besides its input: the card, the kernels, the
/// weights and the one-expert table filled for the pass's columns.
struct Cx<'a> {
    gpu: &'a Gpu,
    k: &'a Kernels,
    w: &'a Weights,
    route: &'a GemmRoute,
}

impl Cx<'_> {
    /// `x`'s `n` columns of `width` values quantized into `acts`' scratch of
    /// that width, which the next [`Cx::proj`]s at `width` read.
    fn quantize(
        &self,
        acts: &mut BTreeMap<usize, GemmAct32>,
        x: &DeviceBuffer<f32>,
        width: usize,
        n: usize,
        sink: FaultSink,
    ) -> Result<(), GpuError> {
        let act = acts.get_mut(&width).ok_or(GpuError::State {
            what: PASS,
            missing: "a quantizer scratch of the projection's width",
        })?;
        self.k
            .g32
            .enqueue_quantize_gemm32(self.gpu.stream(), x, n, act, sink)
    }

    /// `y = W·x` for the q8_0 weight `name` of `rows` rows over the `n`
    /// columns [`Cx::quantize`] last wrote at `width`, token-major.
    fn proj(
        &self,
        acts: &BTreeMap<usize, GemmAct32>,
        name: &str,
        (width, rows): (usize, usize),
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let act = acts.get(&width).ok_or(GpuError::State {
            what: PASS,
            missing: "a quantizer scratch of the projection's width",
        })?;
        site::gemm(
            self.gpu,
            &WideKernels {
                gemm: &self.k.gemm,
                g32: Some(&self.k.g32),
            },
            (SiteTy::Q8_0, self.w, SiteWeight::Resident(name)),
            rows,
            &WideIn {
                q128: None,
                q32: Some(act),
                f32: None,
            },
            (self.route, GemmInput::PerSlot),
            n,
            y,
        )
    }

    fn dense(
        &self,
        acts: &mut BTreeMap<usize, GemmAct32>,
        scratch: &mut DeviceBuffer<f32>,
        (e, head, out): (usize, DenseHead, usize),
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.proj(acts, head.up, (e, head.width), n, scratch)?;
        self.quantize(acts, scratch, head.width, n, self.gpu.unlabelled_sink())?;
        self.proj(acts, head.down, (head.width, out), n, y)
    }
}

/// A reach's per-row key bounds on the card: row `t` attends keys
/// `[first[t], count[t])` ([`key_bounds`]).
struct Keys {
    reach: Option<usize>,
    first: DeviceBuffer<u32>,
    count: DeviceBuffer<u32>,
}

impl Keys {
    fn new(gpu: &Gpu, reach: Option<usize>, n: usize) -> Result<Keys, GpuError> {
        let narrow = |v: usize| {
            u32::try_from(v).map_err(|_| GpuError::Shape {
                what: PASS,
                detail: format!("a key bound of {v} over {n} positions does not fit u32"),
            })
        };
        let mut first = Vec::with_capacity(n);
        let mut count = Vec::with_capacity(n);
        for t in 0..n {
            let (f, c) = key_bounds(reach, t, n);
            first.push(narrow(f)?);
            count.push(narrow(c)?);
        }
        Ok(Keys {
            reach,
            first: DeviceBuffer::from_host(gpu.stream(), &first)?,
            count: DeviceBuffer::from_host(gpu.stream(), &count)?,
        })
    }
}

/// What every step of a pass reads unchanged: the card, the kernels, the
/// weights, the header, the constant rows, and the pass's `n` positions in
/// a load of `cap`.
struct Step<'a> {
    gpu: &'a Gpu,
    k: &'a Kernels,
    w: &'a Weights,
    hp: &'a Hparams,
    consts: &'a Consts,
    n: usize,
    cap: usize,
}

impl Step<'_> {
    fn gain(&self, name: &str) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.w.f32_buf(PASS, name)
    }

    fn cx<'b>(&'b self, route: &'b GemmRoute) -> Cx<'b> {
        Cx {
            gpu: self.gpu,
            k: self.k,
            w: self.w,
            route,
        }
    }

    /// Layer `l`'s attention over `emb` at layer 0, else over `s.x`: `s.a`
    /// is the input plus the post-normed attention of its norm, the turn by
    /// `turn`'s rows and the flash over `keys`' bounds.
    fn attention(
        &self,
        s: &mut Scratch,
        (l, a): (usize, &Attn),
        emb: Option<&DeviceBuffer<f32>>,
        turn: &RopeRows,
        keys: &Keys,
    ) -> Result<(), GpuError> {
        let (stream, elem, sink) = (self.gpu.stream(), self.gpu.elem(), self.gpu.layer_sink(l)?);
        let (e, n, n_head, eps) = (self.hp.n_embd, self.n, self.hp.n_head, self.hp.rms_eps);
        let (head, kv, qw) = (a.head, a.kv, n_head * a.head);
        let blk = |stem: &str| format!("blk.{l}.{stem}");
        let inp = match emb {
            Some(emb) => emb,
            None => &s.x,
        };
        let cx = self.cx(&s.route);
        let acts = &mut s.acts;
        elem.enqueue_rms_norm(
            stream,
            inp,
            self.gain(&blk(names::ATTN_NORM))?,
            eps,
            e,
            n,
            &mut s.xn,
        )?;
        cx.quantize(acts, &s.xn, e, n, sink)?;
        cx.proj(acts, &blk(names::ATTN_Q), (e, qw), n, &mut s.qs)?;
        cx.proj(acts, &blk(names::ATTN_K), (e, kv * head), n, &mut s.kk)?;
        cx.proj(acts, &blk(names::ATTN_V), (e, kv * head), n, &mut s.vv)?;
        let value = if self.hp.per_layer != 0 {
            elem.enqueue_rms_norm(
                stream,
                &s.vv,
                &self.consts.ones,
                eps,
                head,
                n * kv,
                &mut s.vn,
            )?;
            &s.vn
        } else {
            &s.vv
        };
        let rope = PartialNeoxArgs {
            qg: &s.qs,
            q: &mut s.q,
            k: &mut s.kk,
            v: value,
            gq: self.gain(&blk(names::ATTN_Q_NORM))?,
            gk: self.gain(&blk(names::ATTN_K_NORM))?,
            table: &turn.table,
            pos: &self.consts.pos,
            eps,
            n_head,
            n_kv: kv,
            ctx: self.cap,
            m: n,
            fault: sink,
            cache_k: &mut s.cache_k,
            cache_v: &mut s.cache_v,
        };
        if head == HEAD_256 {
            // SAFETY: `pos` is `0..ctx` (`Consts::pos`) and a pass's `n` is at
            // most `ctx`, so the `n` tokens hold distinct positions.
            unsafe {
                self.k
                    .rope
                    .enqueue_head_norm_neox_append_256_whole(stream, rope)?;
            }
        } else {
            // SAFETY: `pos` is `0..ctx` (`Consts::pos`) and a pass's `n` is at
            // most `ctx`, so the `n` tokens hold distinct positions.
            unsafe {
                self.k
                    .rope
                    .enqueue_head_norm_neox_append_512(stream, rope)?;
            }
        }
        if self.hp.dense.is_some() {
            // Scale Q before the flash stages it as f16, as the reference does.
            elem.enqueue_weighted_sum(stream, &s.q, &self.consts.inv, qw, 1, n, &mut s.qs)?;
        }
        let args = GqaPrefillArgs {
            q: if self.hp.dense.is_some() { &s.qs } else { &s.q },
            kc: &s.cache_k,
            vc: &s.cache_v,
            n_keys: &keys.count,
            scale: 1.0,
            n_head,
            n_kv: kv,
            ctx: self.cap,
            t: n,
            fault: sink,
            y: &mut s.att,
        };
        if head == HEAD_256 {
            if (n_head / kv).is_multiple_of(2) {
                self.k
                    .flash
                    .enqueue_256_p2_lower(stream, args, &keys.first)?;
            } else {
                self.k
                    .flash
                    .enqueue_256_p1_lower(stream, args, &keys.first)?;
            }
        } else {
            self.k.flash512.enqueue_p4(stream, args)?;
        }
        cx.quantize(acts, &s.att, qw, n, sink)?;
        cx.proj(acts, &blk(names::ATTN_OUTPUT), (qw, e), n, &mut s.o)?;
        elem.enqueue_rms_norm(
            stream,
            &s.o,
            self.gain(&blk(names::POST_ATTENTION_NORM))?,
            eps,
            e,
            n,
            &mut s.on,
        )?;
        elem.enqueue_add(stream, &s.on, inp, e * n, &mut s.a)?;
        Ok(())
    }

    /// Layer `l`'s GELU GLU: `s.c` is `s.a` plus the post-normed
    /// feed-forward of its norm.
    fn glu(&self, s: &mut Scratch, l: usize) -> Result<(), GpuError> {
        let (stream, elem, sink) = (self.gpu.stream(), self.gpu.elem(), self.gpu.layer_sink(l)?);
        let (e, n, ff, eps) = (self.hp.n_embd, self.n, self.hp.n_ff, self.hp.rms_eps);
        let blk = |stem: &str| format!("blk.{l}.{stem}");
        let cx = self.cx(&s.route);
        let acts = &mut s.acts;
        elem.enqueue_rms_norm(
            stream,
            &s.a,
            self.gain(&blk(names::FFN_NORM))?,
            eps,
            e,
            n,
            &mut s.xn,
        )?;
        cx.quantize(acts, &s.xn, e, n, sink)?;
        cx.proj(acts, &blk(names::FFN_GATE), (e, ff), n, &mut s.gate)?;
        cx.proj(acts, &blk(names::FFN_UP), (e, ff), n, &mut s.up)?;
        elem.enqueue_geglu(stream, &s.gate, &s.up, ff * n, &mut s.gu)?;
        cx.quantize(acts, &s.gu, ff, n, sink)?;
        cx.proj(acts, &blk(names::FFN_DOWN), (ff, e), n, &mut s.o)?;
        elem.enqueue_rms_norm(
            stream,
            &s.o,
            self.gain(&blk(names::POST_FFW_NORM))?,
            eps,
            e,
            n,
            &mut s.on,
        )?;
        elem.enqueue_add(stream, &s.on, &s.a, e * n, &mut s.c)?;
        Ok(())
    }

    /// Layer `l`'s per-layer input and output scale: `s.x` is the layer's
    /// scale times `s.c` plus the post-normed `proj` of `gelu(inp_gate ·
    /// s.c) · ple_l`, where `ple_l` is `ple · emb` times `1/√hidden`,
    /// RMS-normed, and copied into `taps` when asked.
    fn per_layer(
        &self,
        s: &mut Scratch,
        l: usize,
        ple: &DeviceTensor<f32>,
        emb: &DeviceBuffer<f32>,
        taps: Option<&mut Taps>,
    ) -> Result<(), GpuError> {
        let (stream, elem, sink) = (self.gpu.stream(), self.gpu.elem(), self.gpu.layer_sink(l)?);
        let (e, n, pl, eps) = (self.hp.n_embd, self.n, self.hp.per_layer, self.hp.rms_eps);
        let blk = |stem: &str| format!("blk.{l}.{stem}");
        let cx = self.cx(&s.route);
        let acts = &mut s.acts;
        cx.quantize(acts, &s.c, e, n, sink)?;
        cx.proj(acts, &blk(names::INP_GATE), (e, pl), n, &mut s.p1)?;
        self.k
            .g32
            .enqueue_f32_tile(stream, ple, emb, n, &mut s.p2)?;
        elem.enqueue_weighted_sum(stream, &s.p2, &self.consts.inv, pl, 1, n, &mut s.p3)?;
        elem.enqueue_rms_norm(
            stream,
            &s.p3,
            self.gain(names::PER_LAYER_PROJ_NORM)?,
            eps,
            pl,
            n,
            &mut s.p2,
        )?;
        if let Some(t) = taps {
            let rows = s.p2.to_host_vec(stream)?;
            for (tok, row) in rows[..n * pl].chunks_exact(pl).enumerate() {
                let at = (tok * self.hp.n_layer + l) * pl;
                t.inp_per_layer[at..at + pl].copy_from_slice(row);
            }
        }
        elem.enqueue_geglu(stream, &s.p1, &s.p2, pl * n, &mut s.p3)?;
        cx.quantize(acts, &s.p3, pl, n, sink)?;
        cx.proj(acts, &blk(names::PROJ), (pl, e), n, &mut s.o)?;
        elem.enqueue_rms_norm(
            stream,
            &s.o,
            self.gain(&blk(names::POST_NORM))?,
            eps,
            e,
            n,
            &mut s.on,
        )?;
        elem.enqueue_add(stream, &s.c, &s.on, e * n, &mut s.a)?;
        elem.enqueue_weighted_sum(stream, &s.a, &self.consts.scale[l], e, 1, n, &mut s.x)?;
        Ok(())
    }
}

impl Encoder {
    /// The token rows of `ids` from the file's table, each times `√hidden`
    /// in f32 (`inp_scaled`); an id past the vocabulary is refused by name.
    fn token_rows(&self, ids: &[u32]) -> Result<Vec<f32>, GpuError> {
        let e = self.hp.n_embd;
        if let Some(&bad) = ids.iter().find(|&&id| id as usize >= self.hp.n_vocab) {
            return Err(GpuError::Shape {
                what: PASS,
                detail: format!("id {bad} is past the vocabulary of {}", self.hp.n_vocab),
            });
        }
        let mut rows = Vec::with_capacity(ids.len() * e);
        decode_rows(
            &self.split,
            &names::token_embd(),
            e,
            ids.iter().map(|&id| id as usize),
            &mut rows,
        )?;
        let scale = (e as f32).sqrt();
        rows.iter_mut().for_each(|v| *v *= scale);
        Ok(rows)
    }

    /// One pass over `ids` (1 to [`Encoder::ctx`] of them, else refused by
    /// name): the mean of every position's `output` row, and with `taps` the
    /// rows the reference names on the way. A fault any launch raised is the
    /// error, and so is an output value that is not finite, by its position.
    pub fn pass(&mut self, ids: &[u32], taps: bool) -> Result<Pass, GpuError> {
        self.pass_media(ids, &[], taps)
    }

    /// One bidirectional pass with image rows already scaled by the tower,
    /// replacing their spans after the text rows receive `sqrt(hidden)`.
    pub fn pass_media(
        &mut self,
        ids: &[u32],
        spans: &[bloomery_gpu::media::MediaSpan<'_>],
        taps: bool,
    ) -> Result<Pass, GpuError> {
        self.pass_inner(ids, spans, taps, None, bloomery_gpu::media::widen_bf16)
    }

    /// Replay each layer from its supplied token-major residual (`n · hidden`
    /// values per layer), returning diagnostic taps rather than an end-to-end pass.
    pub fn replay_layers(&mut self, ids: &[u32], inputs: &[Vec<f32>]) -> Result<Taps, GpuError> {
        self.pass_inner(
            ids,
            &[],
            true,
            Some(inputs),
            bloomery_gpu::media::widen_bf16,
        )?
        .taps
        .ok_or(GpuError::State {
            what: PASS,
            missing: "the replay's requested taps",
        })
    }

    /// One pass with already-scaled f32 media rows, without narrowing audio to bf16.
    pub fn pass_media_f32(
        &mut self,
        ids: &[u32],
        spans: &[bloomery_gpu::media::MediaSpan<'_, f32>],
        taps: bool,
    ) -> Result<Pass, GpuError> {
        self.pass_inner(ids, spans, taps, None, |rows, dst| {
            dst.copy_from_slice(rows)
        })
    }

    fn pass_inner<T>(
        &mut self,
        ids: &[u32],
        spans: &[bloomery_gpu::media::MediaSpan<'_, T>],
        taps: bool,
        layer_inputs: Option<&[Vec<f32>]>,
        copy: fn(&[T], &mut [f32]),
    ) -> Result<Pass, GpuError> {
        let n = ids.len();
        if n == 0 || n > self.ctx {
            return Err(GpuError::Shape {
                what: PASS,
                detail: format!("{n} positions; a pass takes 1 to {}", self.ctx),
            });
        }
        if let Some(inputs) = layer_inputs {
            if inputs.len() != self.hp.n_layer {
                return Err(GpuError::shape(PASS, "replay needs one input per layer"));
            }
            for (l, input) in inputs.iter().enumerate() {
                if input.len() != n * self.hp.n_embd || input.iter().any(|v| !v.is_finite()) {
                    return Err(GpuError::shape(
                        PASS,
                        format!(
                            "replay layer {l} needs {n} finite rows of {}",
                            self.hp.n_embd
                        ),
                    ));
                }
            }
        }
        bloomery_gpu::media::check_spans(PASS, spans, n, self.hp.n_embd)?;
        let mut inp_scaled = self.token_rows(ids)?;
        for span in spans {
            let dst = &mut inp_scaled[span.at.start * self.hp.n_embd..span.at.end * self.hp.n_embd];
            copy(span.rows, dst);
            if dst.iter().any(|v| !v.is_finite()) {
                return Err(GpuError::shape(
                    PASS,
                    "media row contains a non-finite value",
                ));
            }
        }
        let Encoder {
            gpu,
            hp,
            layers,
            w,
            ple,
            k,
            rope_swa,
            rope_full,
            consts,
            s,
            ctx,
            readback,
            ..
        } = self;
        let stream = gpu.stream();
        let (e, n_out) = (hp.n_embd, hp.n_out);
        let emb = DeviceBuffer::from_host(stream, &inp_scaled)?;
        let keys = reaches(layers)
            .into_iter()
            .map(|r| Keys::new(gpu, r, n))
            .collect::<Result<Vec<_>, _>>()?;
        k.gemm
            .enqueue_route_dense(stream, n, &mut s.route, gpu.unlabelled_sink())?;
        let st = Step {
            gpu,
            k,
            w,
            hp,
            consts,
            n,
            cap: *ctx,
        };
        let mut tapped = taps.then(|| Taps {
            inp_scaled: inp_scaled.clone(),
            inp_per_layer: vec![0.0; n * hp.n_layer * hp.per_layer],
            l_out: Vec::new(),
            result_norm: Vec::new(),
            result_embd: Vec::new(),
        });
        let mut replay_input = Vec::new();
        if layer_inputs.is_some() {
            replay_input.resize(s.x.len(), 0.0);
        }
        for (l, a) in layers.iter().enumerate() {
            if let Some(inputs) = layer_inputs {
                replay_input[..n * e].copy_from_slice(&inputs[l]);
                s.x.copy_from_host(stream, &replay_input)?;
            }
            let turn = if a.reach.is_some() {
                &*rope_swa
            } else {
                &*rope_full
            };
            let bounds = keys
                .iter()
                .find(|r| r.reach == a.reach)
                .ok_or(GpuError::State {
                    what: PASS,
                    missing: "the key bounds of the layer's reach",
                })?;
            st.attention(
                s,
                (l, a),
                (l == 0 && layer_inputs.is_none()).then_some(&emb),
                turn,
                bounds,
            )?;
            st.glu(s, l)?;
            if hp.per_layer != 0 {
                st.per_layer(s, l, &ple[l], &emb, tapped.as_mut())?;
            } else {
                std::mem::swap(&mut s.x, &mut s.c);
            }
            if let Some(t) = &mut tapped {
                let x = Window::<f32>::of(&s.x, 0, n * e)?;
                t.l_out.push(to_host(stream, &x)?);
            }
        }
        let sink = gpu.unlabelled_sink();
        gpu.elem().enqueue_rms_norm(
            stream,
            &s.x,
            st.gain(&names::output_norm())?,
            hp.rms_eps,
            e,
            n,
            &mut s.xn,
        )?;
        let cx = st.cx(&s.route);
        cx.quantize(&mut s.acts, &s.xn, e, n, sink)?;
        let mut out = DeviceBuffer::<f32>::zeroed(stream, n * n_out)?;
        if let Some(head) = hp.dense {
            cx.dense(&mut s.acts, &mut s.dense, (e, head, n_out), n, &mut out)?;
        } else {
            cx.proj(&s.acts, &names::output(), (e, n_out), n, &mut out)?;
        }
        if readback.len() < out.len() {
            *readback = bloomery_gpu::host_filled(out.len());
        }
        let rows = &mut readback[..out.len()];
        out.copy_to_host(stream, rows)?;
        let pooled = if let Some(head) = hp.dense {
            let weights = DeviceBuffer::from_host(stream, &vec![1.0 / n as f32; n])?;
            let count = u32::try_from(n)
                .map_err(|_| GpuError::shape(PASS, "pooling count does not fit u32"))?;
            gpu.elem()
                .enqueue_weighted_sum(stream, &s.xn, &weights, e, count, 1, &mut s.a)?;
            k.gemm.enqueue_route_dense(stream, 1, &mut s.route, sink)?;
            let cx = st.cx(&s.route);
            cx.quantize(&mut s.acts, &s.a, e, 1, sink)?;
            cx.dense(&mut s.acts, &mut s.dense, (e, head, n_out), 1, &mut out)?;
            let pooled = Window::<f32>::of(&out, 0, n_out)?;
            to_host(stream, &pooled)?
        } else {
            embed::mean(rows, n_out)
        };
        if let Some(t) = &mut tapped {
            let xn = Window::<f32>::of(&s.xn, 0, n * e)?;
            t.result_norm = to_host(stream, &xn)?;
            t.result_embd = rows.to_vec();
        }
        if let Some(f) = gpu.take_fault()? {
            return Err(GpuError::fault(PASS, f));
        }
        if let Some(i) = rows.iter().position(|v| !v.is_finite()) {
            return Err(GpuError::Shape {
                what: PASS,
                detail: format!(
                    "position {}'s output value {} is {}, not finite",
                    i / n_out,
                    i % n_out,
                    rows[i]
                ),
            });
        }
        if pooled.iter().any(|v| !v.is_finite()) {
            return Err(GpuError::shape(
                PASS,
                "pooled output contains a non-finite value",
            ));
        }
        Ok(Pass {
            pooled,
            taps: tapped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_a_readback_keeps_its_memory_and_returns_each_read_whole() {
        crate::tests::with_model(64, 2048, |split, hp| -> Result<(), GpuError> {
            let mut enc = Encoder::open(split, hp.n_ctx, Gpu::new()?)?;
            assert!(enc.readback.is_empty());
            enc.pass(&[0, 1], true)?;
            assert_eq!(enc.readback.len(), 2 * hp.n_out);
            let at = enc.readback.as_ptr();
            enc.readback.fill(f32::NAN);
            let pass = enc.pass(&[1], true)?;
            let rows = pass.taps.unwrap().result_embd;
            assert_eq!(rows.len(), hp.n_out);
            assert_eq!(rows, enc.readback[..hp.n_out]);
            assert_eq!(
                enc.readback.as_ptr(),
                at,
                "the shorter pass reuses its memory"
            );
            enc.pass(&[1, 0], false)?;
            assert_eq!(
                enc.readback.as_ptr(),
                at,
                "the equal pass reuses its memory"
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_pass_inputs_are_checked_before_the_pass() {
        use bloomery_gpu::media::MediaSpan;

        crate::tests::with_model(64, 2048, |split, hp| -> Result<(), GpuError> {
            let mut enc = Encoder::open(split, hp.n_ctx, Gpu::new()?)?;
            let rows = vec![0.12345679_f32; hp.n_embd];
            let span = MediaSpan {
                at: 0..1,
                rows: rows.as_slice(),
            };
            let pass = enc.pass_media_f32(&[0], &[span], true)?;
            assert_eq!(pass.taps.unwrap().inp_scaled, rows, "f32 media precision");
            for values in [
                vec![0.0; hp.n_embd - 1],
                vec![f32::NAN; hp.n_embd],
                vec![f32::INFINITY; hp.n_embd],
            ] {
                let error = enc
                    .pass_media_f32(
                        &[0],
                        &[MediaSpan {
                            at: 0..1,
                            rows: &values,
                        }],
                        false,
                    )
                    .err()
                    .expect("invalid f32 rows accepted");
                assert!(matches!(error, GpuError::Shape { .. }), "{error}");
            }
            let rows = vec![0; hp.n_embd];
            for spans in [
                vec![MediaSpan {
                    at: 0..0,
                    rows: &[],
                }],
                vec![MediaSpan {
                    at: 1..2,
                    rows: &rows,
                }],
                vec![MediaSpan {
                    at: 0..1,
                    rows: &[],
                }],
                vec![
                    MediaSpan {
                        at: 0..1,
                        rows: &rows,
                    },
                    MediaSpan {
                        at: 0..1,
                        rows: &rows,
                    },
                ],
            ] {
                assert!(matches!(
                    enc.pass_media(&[0], &spans, false),
                    Err(GpuError::Shape { .. })
                ));
            }
            assert!(
                enc.pass_media(
                    &[0],
                    &[MediaSpan {
                        at: 0..1,
                        rows: &rows
                    }],
                    false
                )
                .is_ok()
            );
            for scale in &mut enc.consts.scale {
                scale.copy_from_host(enc.gpu.stream(), &vec![1.0; scale.len()])?;
            }
            for n in [1, hp.n_ctx] {
                let ids = vec![0; n];
                let taps = enc.pass(&ids, true)?.taps.unwrap();
                let mut inputs = vec![taps.inp_scaled];
                inputs.extend(taps.l_out[..hp.n_layer - 1].iter().cloned());
                assert_eq!(enc.replay_layers(&ids, &inputs)?.l_out, taps.l_out);
                for l in 0..hp.n_layer {
                    for row in 0..n {
                        let at = row * hp.n_embd;
                        let original = inputs[l][at];
                        inputs[l][at] += (l + row + 1) as f32;
                        let replay = enc.replay_layers(&ids, &inputs)?;
                        assert_ne!(replay.l_out[l], taps.l_out[l], "layer {l}, row {row}");
                        for other in 0..hp.n_layer {
                            if other != l {
                                assert_eq!(replay.l_out[other], taps.l_out[other]);
                            }
                        }
                        inputs[l][at] = original;
                    }
                    for len in [n * hp.n_embd - 1, n * hp.n_embd + 1] {
                        let mut bad = inputs.clone();
                        bad[l].resize(len, 0.0);
                        assert!(matches!(
                            enc.replay_layers(&ids, &bad),
                            Err(GpuError::Shape { what: PASS, detail })
                                if detail == format!("replay layer {l} needs {n} finite rows of {}", hp.n_embd)
                        ));
                    }
                    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let mut bad = inputs.clone();
                        bad[l][n * hp.n_embd - 1] = value;
                        assert!(matches!(
                            enc.replay_layers(&ids, &bad),
                            Err(GpuError::Shape { what: PASS, detail })
                                if detail == format!("replay layer {l} needs {n} finite rows of {}", hp.n_embd)
                        ));
                    }
                }
                for len in [0, hp.n_layer - 1, hp.n_layer + 1] {
                    let mut bad = inputs.clone();
                    bad.resize(len, inputs[0].clone());
                    assert!(matches!(
                        enc.replay_layers(&ids, &bad),
                        Err(GpuError::Shape { what: PASS, detail })
                            if detail == "replay needs one input per layer"
                    ));
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_per_layer_projection_scale_near_epsilon() {
        let (hp, weights, gain, pass) =
            crate::tests::with_model(64, 2048, |split, hp| -> Result<_, GpuError> {
                let mut weights = Vec::new();
                decode_rows(
                    &split,
                    names::PER_LAYER_MODEL_PROJ,
                    hp.n_embd,
                    0..hp.n_layer * hp.per_layer,
                    &mut weights,
                )?;
                let mut enc = Encoder::open(split, hp.n_ctx, Gpu::new()?)?;
                let gain = enc
                    .w
                    .f32_buf(PASS, names::PER_LAYER_PROJ_NORM)?
                    .to_host_vec(enc.gpu.stream())?;
                Ok((hp.clone(), weights, gain, enc.pass(&[0, 1], true)?))
            })
            .unwrap();
        let taps = pass.taps.unwrap();
        assert_eq!(taps.inp_scaled.len(), 2 * hp.n_embd);
        assert_eq!(taps.inp_per_layer.len(), 2 * hp.n_layer * hp.per_layer);
        let eps = f64::from(hp.rms_eps);
        for (t, input) in taps.inp_scaled.chunks_exact(hp.n_embd).enumerate() {
            for (l, layer) in weights.chunks_exact(hp.per_layer * hp.n_embd).enumerate() {
                let projected: Vec<f64> = layer
                    .chunks_exact(hp.n_embd)
                    .map(|row| {
                        row.iter()
                            .zip(input)
                            .map(|(&w, &x)| f64::from(w) * f64::from(x))
                            .sum::<f64>()
                            / (hp.n_embd as f64).sqrt()
                    })
                    .collect();
                let mean_square =
                    projected.iter().map(|x| x * x).sum::<f64>() / hp.per_layer as f64;
                assert!(
                    (eps / 2.0..eps * 2.0).contains(&mean_square),
                    "{mean_square:e}"
                );
                let at = (t * hp.n_layer + l) * hp.per_layer;
                for (i, (&x, &g)) in projected.iter().zip(&gain).enumerate() {
                    let expected = x / (mean_square + eps).sqrt() * f64::from(g);
                    let actual = f64::from(taps.inp_per_layer[at + i]);
                    assert!(
                        (actual - expected).abs() < 1.0e-5,
                        "token {t} layer {l} row {i}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}
