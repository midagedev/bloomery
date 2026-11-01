//! The rows arms of the ops: a pass of more rows than a step's window, every
//! op a GEMM or a row kernel over the rows arena.
//!
//! The groups of a layer, in the order the walk names them: the norm and the
//! P1 GEMMs; the delta step over the rows, the short conv over the rows and
//! the ring's commit, or the attention's append and flash, then the out
//! projection, a GEMM adding into the residual (on a sandwich a GEMM storing
//! into the pass's normed-rows buffer, and the join that norms those rows
//! into the stream); the FFN norm, the router's GEMM, the selection and the
//! route table; the grouped experts and the combine. Each op closes its group
//! with a barrier, as the step's do.
//! A layer's slots are the routed picks and, when the instance's router has a
//! shared row, the shared expert's: its gate GEMM, id, group and tensors are
//! there only then. A layer of a dense FFN has no router, selection or table:
//! its FFN is the norm, the gate and up GEMMs, the SwiGLU over their rows and
//! the down GEMM adding into the residual ([`MetalOps::rows_dense_ffn`]).
//!
//! The pass's operand is [`PASS_OPERAND`], one place for every GEMM of the
//! pass but the router's and the f32 projections', which are f32. The pass's step record slot, where
//! its faults land and its head writes its pick, is its last row's: the pass
//! is the token at its last position as far as the record goes.

use bloomery_device::{Buffer, Elems, WindowMut};
use chain::ChainError;
use chain::arch::qwen3moe::ops::LayerCtx;
use chain::arch::qwen3moe::plan::ResidualPlan;

use super::super::arena::{Arena, MixerBufs, lane_of, layer_word, slot_of};
use super::super::rows::FfnRows;
use super::super::session::SessionError;
use super::{
    Half, MetalOps, WHAT, all, all_mut, conv_bufs, gains, head_params, post_gain, refuse, rope_of,
};
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::kernels::{
    AttentionPrefillParams, AttnCache, AttnLayer, CombineParams, DeltaRowStrides, DeltaRowsParams,
    DeltaSnaps, EmbedParams, F32GemmParams, GemmMode, GemmParams, GemmSegParams, GemmXParams,
    HeadNormRowsParams, NormAddRowsParams, NormRowsParams, Operand, Quant, RouteSelectParams,
    RouteTableParams, RoutedKind, RoutedParams, Segment, ShortConvRowsParams, SwigluRowsParams,
    route_table_words, route_tiles,
};

/// The operand type the pass's GEMMs compute in.
pub(in super::super) const PASS_OPERAND: Operand = Operand::F16;

/// Where the out GEMM of one half of a pass writes, and how: a plain residual
/// adds into the stream in the residual epilogue; a sandwich stores into the
/// pass's normed rows, which the projections' GEMMs have read by then, and
/// [`MetalOps::rows_join`] norms them into the stream.
fn half_out<'b>(
    l: LayerCtx<'_>,
    x: &'b mut MetalBuffer<f32>,
    xn: &'b mut MetalBuffer<f32>,
    len: usize,
) -> Result<(WindowMut<'b, f32, MetalDevice>, GemmMode), SessionError> {
    match &l.plan.residual {
        ResidualPlan::Plain => Ok((x.window_mut(Elems(0), Elems(len))?, GemmMode::Residual)),
        ResidualPlan::Sandwich(_) => Ok((xn.window_mut(Elems(0), Elems(len))?, GemmMode::Store)),
    }
}

/// The weight bytes and the output columns `segments` of one dispatch of
/// `quant` span: the extents a segmented launch holds its windows to.
fn extents(quant: Quant, segments: &[Segment]) -> (usize, usize) {
    let (mut weights, mut out) = (0, 0);
    for s in segments {
        let bytes = s.rows * (s.cols / quant.values_per_block()) * quant.block_bytes();
        weights = weights.max(s.weight_blocks * quant.block_bytes() + bytes);
        out = out.max(s.out_offset + s.rows);
    }
    (weights, out)
}

/// The elements `rows` rows of `width` values, `stride` apart, span: the last
/// row ends where its values do, and the stride's gap after it is no row's. A
/// GEMM holds its activation and output windows to exactly this length.
fn strided_len(rows: usize, stride: usize, width: usize) -> usize {
    (rows - 1) * stride + width
}

impl MetalOps<'_, '_> {
    /// The pass's step record slot: its last row's.
    fn pass_slot(&self, m: usize) -> usize {
        slot_of(self.pos + u32::try_from(m).expect("a pass's rows fit u32") - 1)
    }

    fn no_rows<T>() -> Result<T, SessionError> {
        Err(ChainError::shape(WHAT, "a rows arm of ops built with no rows arena").into())
    }

    /// The refusal of a per-head gate over a pass arena with no gate rows.
    fn no_gate_rows(l: usize) -> SessionError {
        refuse(format!(
            "layer {l}: a per-head gate over a pass arena with no gate rows"
        ))
    }

    /// The refusal of a routed arm over a rows arena of a dense FFN.
    fn not_routed<T>(l: usize) -> Result<T, SessionError> {
        Err(ChainError::shape(
            WHAT,
            format!("layer {l}: a routed rows arm over the buffers of a dense FFN"),
        )
        .into())
    }

    /// The embedding of the pass's ids, which the host wrote to the rows
    /// arena before the walk.
    pub(super) fn rows_embed(&mut self, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            embed,
            embed_quant,
            vocab,
            hidden,
            faults,
            ..
        } = &mut *self.arena;
        self.kernels.embed(
            self.enc,
            EmbedParams {
                quant: *embed_quant,
                table: all(embed)?,
                vocab: *vocab,
                cols: *hidden,
                ids: rows.ids.window(Elems(0), Elems(m))?,
                out: rows.x.window_mut(Elems(0), Elems(m * *hidden))?,
                fault: &faults[slot],
                layer: layer_word(0),
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s P1: the norm of the rows, then one GEMM a weight type. An
    /// attention layer's per-head gate is not a column of the P1 rows: the
    /// flash reads a row's gate contiguous, so the gate's segment leaves its
    /// dispatch and has a GEMM of its own into the pass's gate rows.
    pub(super) fn rows_p1(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            faults,
            eps,
            hidden,
            geom,
            ..
        } = &mut *self.arena;
        let (p1, gate) = match &layers[l].mixer {
            MixerBufs::Delta(d) => (&d.p1, None),
            MixerBufs::Attention(a) => (&a.p1, a.gate.as_ref()),
            MixerBufs::ShortConv(c) => (&c.p1, None),
        };
        if let Some(g) = gate
            && !p1
                .parts
                .iter()
                .any(|part| part.segments.iter().any(|s| s.out_offset == g.start))
        {
            return Err(refuse(format!(
                "layer {l}: the per-head gate is in no dispatch of the layer's P1"
            )));
        }
        self.kernels.norm_rows(
            self.enc,
            NormRowsParams {
                rows: m,
                cols: *hidden,
                x: rows.x.window(Elems(0), Elems(m * *hidden))?,
                x_stride: *hidden,
                gain: all(&p1.norm)?,
                y: rows.xn.window_mut(Elems(0), Elems(m * *hidden))?,
                y_stride: *hidden,
                eps: *eps,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        // The quantized projections' GEMMs read the normed rows cast once.
        // A P1 of f32 projections only reads the f32 rows and casts nothing:
        // a finite value past the half range is no fault there.
        let x = if p1.parts.iter().any(|part| part.quant != Quant::F32) {
            self.kernels.gemm_x(
                self.enc,
                GemmXParams {
                    x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                    x_stride: *hidden,
                    cols: *hidden,
                    acts: m,
                    operand: PASS_OPERAND,
                    half: &mut rows.xh,
                    fault: &faults[slot],
                    layer: layer_word(l),
                },
            )?
        } else {
            crate::kernels::GemmX::F32 {
                x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                stride: *hidden,
            }
        };
        let stride = rows.proj_stride;
        let heads = geom.attn.query_heads;
        for part in &p1.parts {
            // The gate's segment is the last of its dispatch, so the
            // dispatch's others are a prefix of its list.
            let at_gate =
                gate.and_then(|g| part.segments.iter().position(|s| s.out_offset == g.start));
            let segments = match at_gate {
                None => &part.segments[..],
                Some(i) if i + 1 == part.segments.len() => &part.segments[..i],
                Some(_) => {
                    return Err(refuse(format!(
                        "layer {l}: the per-head gate is not the last segment of its dispatch"
                    )));
                }
            };
            if let Some(i) = at_gate {
                let seg = part.segments[i];
                if seg.rows != heads {
                    return Err(refuse(format!(
                        "layer {l}: a per-head gate of {} rows, the instance has {heads} query heads",
                        seg.rows
                    )));
                }
            }
            // The f32 projections run as the router's f32 GEMM, a segment a
            // dispatch, both operands f32.
            if part.quant == Quant::F32 {
                for seg in segments.iter().chain(at_gate.map(|i| &part.segments[i])) {
                    let at = seg.weight_blocks * part.quant.block_bytes();
                    let len = seg.rows * seg.cols * size_of::<f32>();
                    let (y, y_stride) = if gate.is_some_and(|g| g.start == seg.out_offset) {
                        let gate_rows = rows.gate.as_mut().ok_or_else(|| Self::no_gate_rows(l))?;
                        (gate_rows.window_mut(Elems(0), Elems(m * heads))?, heads)
                    } else {
                        (
                            rows.proj.window_mut(
                                Elems(seg.out_offset),
                                Elems(strided_len(m, stride, seg.rows)),
                            )?,
                            stride,
                        )
                    };
                    self.kernels.gemm_f32(
                        self.enc,
                        F32GemmParams {
                            rows: seg.rows,
                            cols: seg.cols,
                            acts: m,
                            weights: all(&part.weights)?
                                .sub(Elems(at), Elems(len))?
                                .cast::<f32>()?,
                            x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                            x_stride: *hidden,
                            y,
                            y_stride,
                            fault: &faults[slot],
                            layer: layer_word(l),
                        },
                    )?;
                }
                continue;
            }
            if !segments.is_empty() {
                let (weight_end, out_end) = extents(part.quant, segments);
                self.kernels.gemm_seg(
                    self.enc,
                    GemmSegParams {
                        quant: part.quant,
                        segments,
                        weights: all(&part.weights)?.sub(Elems(0), Elems(weight_end))?,
                        x,
                        acts: m,
                        y: rows
                            .proj
                            .window_mut(Elems(0), Elems(strided_len(m, stride, out_end)))?,
                        y_stride: stride,
                        mode: GemmMode::Store,
                        operand: PASS_OPERAND,
                        fault: &faults[slot],
                        layer: layer_word(l),
                    },
                )?;
            }
            if let Some(i) = at_gate {
                let seg = part.segments[i];
                let at = seg.weight_blocks * part.quant.block_bytes();
                let len = seg.rows
                    * (seg.cols / part.quant.values_per_block())
                    * part.quant.block_bytes();
                let gate_rows = rows.gate.as_mut().ok_or_else(|| Self::no_gate_rows(l))?;
                self.kernels.gemm(
                    self.enc,
                    GemmParams {
                        quant: part.quant,
                        rows: seg.rows,
                        cols: seg.cols,
                        acts: m,
                        weights: all(&part.weights)?.sub(Elems(at), Elems(len))?,
                        x,
                        y: gate_rows.window_mut(Elems(0), Elems(m * heads))?,
                        y_stride: heads,
                        mode: GemmMode::Store,
                        operand: PASS_OPERAND,
                        fault: &faults[slot],
                        layer: layer_word(l),
                    },
                )?;
            }
        }
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s attention over the pass: the prefill flash appends the
    /// chunk's keys and values and runs every row against the cache, over the
    /// layer's window of keys and with its per-head gate where it has them.
    pub(super) fn rows_attn_core(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let from = self.pos;
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            ropes,
            faults,
            eps,
            ctx: cache_rows,
            geom,
            ..
        } = &mut *self.arena;
        let MixerBufs::Attention(a) = &mut layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: an attention op on a layer of another mixer"),
            )
            .into());
        };
        let rope = rope_of(ropes, l, a.rope)?;
        let head_gate = if a.gate.is_some() {
            let gate_rows = rows.gate.as_ref().ok_or_else(|| Self::no_gate_rows(l))?;
            Some(gate_rows.window(Elems(0), Elems(m * geom.attn.query_heads))?)
        } else {
            None
        };
        let layer = AttnLayer {
            window: a.window,
            head_gate,
        };
        let stride = rows.proj_stride;
        let at = |range_start: usize| -> (usize, usize) { (range_start, m * stride - range_start) };
        let (q_at, q_len) = at(a.q.start);
        let (k_at, k_len) = at(a.k.start);
        let (v_at, v_len) = at(a.v.start);
        self.kernels.attention_prefill_layer(
            self.enc,
            AttentionPrefillParams {
                head_dim: geom.attn.head,
                query_heads: geom.attn.query_heads,
                kv_heads: geom.attn.kv_heads,
                ctx: *cache_rows,
                eps: *eps,
                layer: layer_word(l),
                from,
                rows: m,
                q: rows.proj.window(Elems(q_at), Elems(q_len))?,
                q_stride: stride,
                k: rows.proj.window(Elems(k_at), Elems(k_len))?,
                k_stride: stride,
                v: rows.proj.window(Elems(v_at), Elems(v_len))?,
                v_stride: stride,
                q_gain: gains(&a.q_norm)?,
                k_gain: gains(&a.k_norm)?,
                rope: all(rope)?,
                cache: AttnCache::F16 {
                    k: all_mut(&mut a.k_cache)?,
                    v: all_mut(&mut a.v_cache)?,
                },
                out: rows.mix.window_mut(Elems(0), Elems(m * rows.mix_stride))?,
                out_stride: rows.mix_stride,
                fault: &faults[slot],
            },
            layer,
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s attention output projection, added into the residual, or
    /// stored for the join on a sandwich.
    pub(super) fn rows_attn_out(&mut self, l: LayerCtx<'_>, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            faults,
            hidden,
            geom,
            ..
        } = &mut *self.arena;
        let MixerBufs::Attention(a) = &layers[l.layer].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!(
                    "layer {}: an attention op on a layer of another mixer",
                    l.layer
                ),
            )
            .into());
        };
        let out_cols = geom
            .attn
            .out_len()
            .ok_or_else(|| ChainError::shape(WHAT, "the attention output does not fit"))?;
        let x = self.kernels.gemm_x(
            self.enc,
            GemmXParams {
                x: rows
                    .mix
                    .window(Elems(0), Elems(strided_len(m, rows.mix_stride, out_cols)))?,
                x_stride: rows.mix_stride,
                cols: out_cols,
                acts: m,
                operand: PASS_OPERAND,
                half: &mut rows.xh,
                fault: &faults[slot],
                layer: layer_word(l.layer),
            },
        )?;
        let (y, mode) = half_out(l, &mut rows.x, &mut rows.xn, m * *hidden)?;
        self.kernels.gemm(
            self.enc,
            GemmParams {
                quant: a.out_quant,
                rows: *hidden,
                cols: out_cols,
                acts: m,
                weights: all(&a.out)?,
                x,
                y,
                y_stride: *hidden,
                mode,
                operand: PASS_OPERAND,
                fault: &faults[slot],
                layer: layer_word(l.layer),
            },
        )?;
        self.enc.memory_barrier();
        self.rows_join(l, Half::Mixer, m)
    }

    /// Layer `l`'s short conv over the pass: `C ⊙ conv(B ⊙ x)` for every row
    /// into the pass's mixer rows, over the pre-pass ring and the rows'
    /// predecessors, and the ring's commit after them (one chunk: the
    /// dispatch, a barrier, the commit). The ring's copy after a row inside
    /// the pass is not run: a pass that asks for one is refused by name.
    pub(super) fn rows_conv_core(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let from = self.pos;
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        if !ctx.after.is_empty() {
            return Err(refuse(format!(
                "layer {l}: a copy of the conv ring after rows {:?} of a pass: the ring's copy is not run",
                ctx.after
            )));
        }
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            faults,
            hidden,
            ..
        } = &mut *self.arena;
        let c = conv_bufs(layers, l)?;
        let stride = rows.proj_stride;
        self.kernels.shortconv_rows(
            self.enc,
            ShortConvRowsParams {
                rows: m,
                bcx: rows
                    .proj
                    .window(Elems(c.bcx.start), Elems(m * stride - c.bcx.start))?,
                bcx_stride: stride,
                conv_w: all(&c.conv)?,
                ring: all_mut(&mut c.ring)?,
                y: rows.mix.window_mut(Elems(0), Elems(m * rows.mix_stride))?,
                y_stride: rows.mix_stride,
                faults: &faults[slot],
                hidden: *hidden,
                taps: c.taps,
                ring_rows: c.ring_rows,
                layer: layer_word(l),
                pos: from,
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s short conv output projection over the pass's mixer rows,
    /// added into the residual, or stored for the join on a sandwich.
    pub(super) fn rows_conv_out(&mut self, l: LayerCtx<'_>, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            faults,
            hidden,
            ..
        } = &mut *self.arena;
        let c = conv_bufs(layers, l.layer)?;
        let x = self.kernels.gemm_x(
            self.enc,
            GemmXParams {
                x: rows
                    .mix
                    .window(Elems(0), Elems(strided_len(m, rows.mix_stride, *hidden)))?,
                x_stride: rows.mix_stride,
                cols: *hidden,
                acts: m,
                operand: PASS_OPERAND,
                half: &mut rows.xh,
                fault: &faults[slot],
                layer: layer_word(l.layer),
            },
        )?;
        let (y, mode) = half_out(l, &mut rows.x, &mut rows.xn, m * *hidden)?;
        self.kernels.gemm(
            self.enc,
            GemmParams {
                quant: c.out_quant,
                rows: *hidden,
                cols: *hidden,
                acts: m,
                weights: all(&c.out)?,
                x,
                y,
                y_stride: *hidden,
                mode,
                operand: PASS_OPERAND,
                fault: &faults[slot],
                layer: layer_word(l.layer),
            },
        )?;
        self.enc.memory_barrier();
        self.rows_join(l, Half::Mixer, m)
    }

    /// Layer `l`'s join over the pass's rows on a sandwich: the half's output
    /// rows in the pass's normed-rows buffer, normed by the half's gain and
    /// added into the stream at the half's scale. A plain residual has no
    /// join: its out and down GEMMs already added into the stream.
    fn rows_join(&mut self, l: LayerCtx<'_>, half: Half, m: usize) -> Result<(), SessionError> {
        let ResidualPlan::Sandwich(s) = &l.plan.residual else {
            return Ok(());
        };
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            faults,
            hidden,
            ..
        } = &mut *self.arena;
        self.kernels.norm_add_rows(
            self.enc,
            NormAddRowsParams {
                norm: NormRowsParams {
                    rows: m,
                    cols: *hidden,
                    x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                    x_stride: *hidden,
                    gain: all(post_gain(layers, l.layer, half)?)?,
                    y: rows.x.window_mut(Elems(0), Elems(m * *hidden))?,
                    y_stride: *hidden,
                    eps: s.eps,
                    fault: &faults[slot],
                    layer: layer_word(l.layer),
                },
                add: s.add[half.index()],
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s delta step over the rows, and the ring's commit after it;
    /// with the pass's copies, the lane and the ring after each row they
    /// name, into the layer's copies.
    pub(super) fn rows_delta_core(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let from = self.pos;
        let to = from + u32::try_from(m).expect("a pass's rows fit u32");
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let after = ctx.after;
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            lane,
            faults,
            eps,
            geom,
            ..
        } = &mut *self.arena;
        let MixerBufs::Delta(d) = &mut layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: a delta op on a layer of another mixer"),
            )
            .into());
        };
        let dense = DeltaRowStrides::of(&geom.delta, d.n_k).ok_or_else(|| SessionError::Shape {
            what: WHAT,
            detail: format!("{} key heads do not fit", d.n_k),
        })?;
        let stride = rows.proj_stride;
        let strides = DeltaRowStrides {
            x: stride,
            z: stride,
            b: stride,
            a: stride,
            ..dense
        };
        let tail = |start: usize| -> Elems { Elems(m * stride - start) };
        let snaps = if after.is_empty() {
            None
        } else {
            let copies = rows.snaps.as_mut();
            let at = copies.as_ref().and_then(|c| c.at.get(l).copied().flatten());
            let (Some(c), Some(at), Some(lane_len)) = (copies, at, geom.delta.state_elems(1))
            else {
                return Err(ChainError::shape(
                    WHAT,
                    format!("layer {l}: a pass's copies with none open for the layer"),
                )
                .into());
            };
            let n = after.len();
            Some(DeltaSnaps {
                after,
                lane: c.lane.window_mut(Elems(at.lane), Elems(n * lane_len))?,
                ring: c.ring.window_mut(Elems(at.ring), Elems(n * at.ring_len))?,
            })
        };
        self.kernels.delta_rows(
            self.enc,
            DeltaRowsParams {
                x: rows.proj.window(Elems(d.qkv.start), tail(d.qkv.start))?,
                conv_w: all(&d.conv)?,
                b_raw: rows.proj.window(Elems(d.beta.start), tail(d.beta.start))?,
                a_raw: rows
                    .proj
                    .window(Elems(d.alpha.start), tail(d.alpha.start))?,
                dt_bias: all(&d.dt_bias)?,
                ssm_a: all(&d.ssm_a)?,
                z: rows.proj.window(Elems(d.z.start), tail(d.z.start))?,
                norm_w: all(&d.ssm_norm)?,
                lane: all(lane)?,
                state: all_mut(&mut d.state)?,
                ring: all_mut(&mut d.ring)?,
                partials: rows
                    .delta_partials
                    .window_mut(Elems(0), Elems(m * strides.partials))?,
                g: rows.mix.window_mut(Elems(0), Elems(m * strides.g))?,
                prep: rows
                    .delta_prep
                    .window_mut(Elems(0), Elems(m * rows.delta_prep_stride))?,
                faults: &faults[slot],
                n_v: geom.delta.value_heads,
                head: geom.delta.head_cols,
                taps: geom.delta.conv_taps,
                n_k: d.n_k,
                map: d.map,
                lanes: super::super::arena::LANES,
                lane_at: lane_of(from),
                lane_to: lane_of(to),
                layer: layer_word(l),
                rows: m,
                pos: from,
                strides,
                eps: *eps,
                snaps,
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s delta out projection: the head norm's scale applied to
    /// the numerator, then the GEMM added into the residual.
    pub(super) fn rows_delta_out(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            faults,
            eps,
            hidden,
            geom,
            ..
        } = &mut *self.arena;
        let MixerBufs::Delta(d) = &layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: a delta op on a layer of another mixer"),
            )
            .into());
        };
        let dense = DeltaRowStrides::of(&geom.delta, d.n_k).ok_or_else(|| SessionError::Shape {
            what: WHAT,
            detail: format!("{} key heads do not fit", d.n_k),
        })?;
        self.kernels.headnorm_rows(
            self.enc,
            HeadNormRowsParams {
                rows: m,
                g: rows.mix.window_mut(Elems(0), Elems(m * dense.g))?,
                g_stride: dense.g,
                partials: rows
                    .delta_partials
                    .window(Elems(0), Elems(m * dense.partials))?,
                partials_stride: dense.partials,
                eps: *eps,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        let x = self.kernels.gemm_x(
            self.enc,
            GemmXParams {
                x: rows.mix.window(Elems(0), Elems(m * dense.g))?,
                x_stride: dense.g,
                cols: dense.g,
                acts: m,
                operand: PASS_OPERAND,
                half: &mut rows.xh,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.kernels.gemm(
            self.enc,
            GemmParams {
                quant: d.out_quant,
                rows: *hidden,
                cols: dense.g,
                acts: m,
                weights: all(&d.out)?,
                x,
                y: rows.x.window_mut(Elems(0), Elems(m * *hidden))?,
                y_stride: *hidden,
                mode: GemmMode::Residual,
                operand: PASS_OPERAND,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s router over the rows: the FFN norm, the router's logits
    /// (the routed rows, then the shared expert's gate when the layer has
    /// one), every row's picks and the route table.
    pub(super) fn rows_route(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let FfnRows::Routed(routed) = &mut rows.ffn else {
            return Self::not_routed(l);
        };
        let Arena {
            layers,
            faults,
            eps,
            hidden,
            top_k,
            n_experts,
            geom,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        let Some(router) = &f.router else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: a router launch on a layer with none"),
            )
            .into());
        };
        let slots = m * geom.experts.slots;
        let logit_stride = geom.router.logit_stride;
        self.kernels.norm_rows(
            self.enc,
            NormRowsParams {
                rows: m,
                cols: *hidden,
                x: rows.x.window(Elems(0), Elems(m * *hidden))?,
                x_stride: *hidden,
                gain: all(&f.norm)?,
                y: rows.xn.window_mut(Elems(0), Elems(m * *hidden))?,
                y_stride: *hidden,
                eps: *eps,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        let routed_cols = (m - 1) * logit_stride;
        self.kernels.gemm_f32(
            self.enc,
            F32GemmParams {
                rows: *n_experts,
                cols: *hidden,
                acts: m,
                weights: all(&router.gate_inp)?,
                x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                x_stride: *hidden,
                y: routed
                    .logits
                    .window_mut(Elems(0), Elems(routed_cols + *n_experts))?,
                y_stride: logit_stride,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        if let Some(shared) = &f.shared {
            self.kernels.gemm_f32(
                self.enc,
                F32GemmParams {
                    rows: 1,
                    cols: *hidden,
                    acts: m,
                    weights: all(&shared.gate_inp)?,
                    x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                    x_stride: *hidden,
                    y: routed
                        .logits
                        .window_mut(Elems(*n_experts), Elems(routed_cols + 1))?,
                    y_stride: logit_stride,
                    fault: &faults[slot],
                    layer: layer_word(l),
                },
            )?;
        }
        self.enc.memory_barrier();
        self.kernels.route_rows_select(
            self.enc,
            RouteSelectParams {
                rows: m,
                top_k: *top_k,
                logits: routed.logits.window(Elems(0), Elems(m * logit_stride))?,
                logit_stride,
                ids: routed.pick_ids.window_mut(Elems(0), Elems(slots))?,
                weights: routed.pick_weights.window_mut(Elems(0), Elems(slots))?,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        self.kernels.route_table(
            self.enc,
            RouteTableParams {
                slots,
                per_row: geom.experts.slots,
                experts: *n_experts,
                ids: routed.pick_ids.window(Elems(0), Elems(slots))?,
                sorted: routed.sorted.window_mut(Elems(0), Elems(slots))?,
                inv: routed.inv.window_mut(Elems(0), Elems(slots))?,
                tiles: routed
                    .tiles
                    .window_mut(Elems(0), Elems(route_table_words(slots, geom.router.rows)))?,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s experts over the rows: the pair and the down over the
    /// route table's tiles, and the combine into the residual.
    pub(super) fn rows_experts(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let FfnRows::Routed(routed) = &mut rows.ffn else {
            return Self::not_routed(l);
        };
        let Arena {
            layers,
            faults,
            hidden,
            n_experts,
            geom,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        let (slots, expert_k) = (m * geom.experts.slots, geom.experts.k);
        let max_tiles = route_tiles(slots, geom.router.rows);
        let table_words = route_table_words(slots, geom.router.rows);
        self.kernels.gemm_routed(
            self.enc,
            RoutedParams {
                quant: f.gate_up_quant,
                shared_quant: f
                    .shared
                    .as_ref()
                    .map_or(f.gate_up_quant, |s| s.gate_up_quant),
                kind: RoutedKind::Pair,
                operand: PASS_OPERAND,
                cols: *hidden,
                rows: expert_k,
                experts: *n_experts,
                stack: all(&f.gate)?,
                stack_up: Some(all(&f.up)?),
                shared: f.shared.as_ref().map(|s| all(&s.gate)).transpose()?,
                shared_up: f.shared.as_ref().map(|s| all(&s.up)).transpose()?,
                x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                x_rows: m,
                x_stride: *hidden,
                gather: true,
                y: rows.h.window_mut(Elems(0), Elems(slots * expert_k))?,
                y_stride: expert_k,
                slots,
                max_tiles,
                sorted: routed.sorted.window(Elems(0), Elems(slots))?,
                route_w: routed.pick_weights.window(Elems(0), Elems(slots))?,
                tiles: routed.tiles.window(Elems(0), Elems(table_words))?,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        self.kernels.gemm_routed(
            self.enc,
            RoutedParams {
                quant: f.down_quant,
                shared_quant: f.shared.as_ref().map_or(f.down_quant, |s| s.down_quant),
                kind: RoutedKind::Plain,
                operand: PASS_OPERAND,
                cols: expert_k,
                rows: *hidden,
                experts: *n_experts,
                stack: all(&f.down)?,
                stack_up: None,
                shared: f.shared.as_ref().map(|s| all(&s.down)).transpose()?,
                shared_up: None,
                x: rows.h.window(Elems(0), Elems(slots * expert_k))?,
                x_rows: slots,
                x_stride: expert_k,
                gather: false,
                y: routed.y.window_mut(Elems(0), Elems(slots * *hidden))?,
                y_stride: *hidden,
                slots,
                max_tiles,
                sorted: routed.sorted.window(Elems(0), Elems(slots))?,
                route_w: routed.pick_weights.window(Elems(0), Elems(slots))?,
                tiles: routed.tiles.window(Elems(0), Elems(table_words))?,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        self.kernels.combine_rows(
            self.enc,
            CombineParams {
                rows: m,
                per_row: geom.experts.slots,
                cols: *hidden,
                y: routed.y.window(Elems(0), Elems(slots * *hidden))?,
                y_rows: slots,
                y_stride: *hidden,
                inv: routed.inv.window(Elems(0), Elems(slots))?,
                x: rows.x.window_mut(Elems(0), Elems(m * *hidden))?,
                x_stride: *hidden,
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s dense FFN over the rows: the FFN norm, the gate and the up
    /// GEMMs over the normed rows, the SwiGLU over their two outputs, and the
    /// down GEMM added into the residual (on a sandwich stored into the
    /// normed rows, which the join then norms into the residual). The GEMMs
    /// are the dense GEMM of the mixers' projections; no row is routed, so no
    /// table, gather or combine stands between the norm and the residual.
    pub(super) fn rows_dense_ffn(
        &mut self,
        cx: LayerCtx<'_>,
        m: usize,
    ) -> Result<(), SessionError> {
        let l = cx.layer;
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let FfnRows::Dense { up } = &mut rows.ffn else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: a dense FFN over the buffers of a routed one"),
            )
            .into());
        };
        let Arena {
            layers,
            faults,
            eps,
            hidden,
            geom,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        let width = geom.experts.k;
        self.kernels.norm_rows(
            self.enc,
            NormRowsParams {
                rows: m,
                cols: *hidden,
                x: rows.x.window(Elems(0), Elems(m * *hidden))?,
                x_stride: *hidden,
                gain: all(&f.norm)?,
                y: rows.xn.window_mut(Elems(0), Elems(m * *hidden))?,
                y_stride: *hidden,
                eps: *eps,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        // The gate and the up GEMMs read the normed rows cast once.
        let x = self.kernels.gemm_x(
            self.enc,
            GemmXParams {
                x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                x_stride: *hidden,
                cols: *hidden,
                acts: m,
                operand: PASS_OPERAND,
                half: &mut rows.xh,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        for (weights, out) in [(&f.gate, &mut rows.h), (&f.up, &mut *up)] {
            self.kernels.gemm(
                self.enc,
                GemmParams {
                    quant: f.gate_up_quant,
                    rows: width,
                    cols: *hidden,
                    acts: m,
                    weights: all(weights)?,
                    x,
                    y: out.window_mut(Elems(0), Elems(m * width))?,
                    y_stride: width,
                    mode: GemmMode::Store,
                    operand: PASS_OPERAND,
                    fault: &faults[slot],
                    layer: layer_word(l),
                },
            )?;
        }
        self.enc.memory_barrier();
        self.kernels.swiglu_rows(
            self.enc,
            SwigluRowsParams {
                rows: m,
                cols: width,
                gate: rows.h.window_mut(Elems(0), Elems(m * width))?,
                up: up.window(Elems(0), Elems(m * width))?,
                stride: width,
            },
        )?;
        self.enc.memory_barrier();
        let x = self.kernels.gemm_x(
            self.enc,
            GemmXParams {
                x: rows.h.window(Elems(0), Elems(m * width))?,
                x_stride: width,
                cols: width,
                acts: m,
                operand: PASS_OPERAND,
                half: &mut rows.xh,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        let (y, mode) = half_out(cx, &mut rows.x, &mut rows.xn, m * *hidden)?;
        self.kernels.gemm(
            self.enc,
            GemmParams {
                quant: f.down_quant,
                rows: *hidden,
                cols: width,
                acts: m,
                weights: all(&f.down)?,
                x,
                y,
                y_stride: *hidden,
                mode,
                operand: PASS_OPERAND,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        self.rows_join(cx, Half::Ffn, m)
    }

    /// The head over the pass's last row: the logits, a barrier that closes
    /// their group, and the pick.
    pub(super) fn rows_head(&mut self, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let hidden = self.arena.hidden;
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let last = ctx.arena.x.window(Elems((m - 1) * hidden), Elems(hidden))?;
        let p = head_params(&mut *self.arena, Some(last), slot)?;
        self.kernels.head(self.enc, p)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{extents, strided_len};
    use crate::geom::Geometry;
    use crate::instance::REGISTERED;
    use crate::kernels::{Quant, Segment};

    /// The activation window of an out GEMM over rows `stride` apart ends where
    /// its last row's values do: a stride wider than the row (the mixer rows
    /// of a file with no delta layer are as wide as the delta numerator) leaves
    /// a gap after the last row that is no row's, and a row as wide as its
    /// stride leaves none.
    ///
    /// Mutant: `strided_len` is `rows * stride`.
    #[test]
    fn a_strided_window_ends_where_its_last_row_does() {
        assert_eq!(strided_len(1024, 4096, 2048), 1023 * 4096 + 2048);
        assert_eq!(strided_len(12, 4096, 2048), 47104);
        assert_eq!(strided_len(9, 2048, 2048), 9 * 2048);
    }

    /// A P1 dispatch of q, k, v and the per-head gate, laid out one segment
    /// after another as the loader lays them: the extents of the dispatch
    /// without its gate end exactly where the gate begins, in weight bytes and
    /// in output columns, and those of the whole dispatch end where it ends.
    /// The segmented launch holds its windows to its segments' extents, so a
    /// dispatch with the gate left out takes a prefix of both windows.
    ///
    /// Mutant: `extents` counts a segment's weight bytes without its
    /// `weight_blocks` offset, or its columns without `out_offset`.
    #[test]
    fn the_extents_of_a_p1_without_its_gate_end_where_the_gate_begins() {
        let quant = Quant::A4;
        let cols = 2048;
        let nblocks = cols / quant.values_per_block();
        let mut segments = Vec::new();
        let (mut bytes, mut out) = (0, 0);
        for rows in [2048, 512, 512, 16] {
            segments.push(Segment {
                cols,
                rows,
                weight_blocks: bytes / quant.block_bytes(),
                out_offset: out,
            });
            bytes += rows * nblocks * quant.block_bytes();
            out += rows;
        }
        let gate = segments[3];
        assert_eq!(
            extents(quant, &segments[..3]),
            (gate.weight_blocks * quant.block_bytes(), gate.out_offset)
        );
        assert_eq!(extents(quant, &segments), (bytes, out));
    }

    /// The logit rows of every registered instance hold the routed experts and
    /// any shared gate, at a stride the GEMM's output rows are aligned to.
    const _: () = {
        let mut i = 0;
        while i < REGISTERED.len() {
            let r = Geometry::of(&REGISTERED[i]).router;
            assert!(
                r.experts + r.shared == r.rows
                    && r.logit_stride >= r.rows
                    && r.logit_stride.is_multiple_of(4)
            );
            i += 1;
        }
    };
}
