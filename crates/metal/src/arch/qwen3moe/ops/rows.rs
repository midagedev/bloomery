//! The rows arms of the ops: a pass of more rows than a step's window, every
//! op a GEMM or a row kernel over the rows arena.
//!
//! The groups of a layer, in the order the walk names them: the norm and the
//! P1 GEMMs; the delta step over the rows (or the attention's append and
//! flash, one launch for each run of rows that share a segment count), then
//! the out projection, a GEMM adding into the residual; the FFN norm, the
//! router's GEMM, the selection and the route table; the grouped experts and
//! the combine. Each op closes its group with a barrier, as the step's do.
//!
//! The pass's operand is [`PASS_OPERAND`], one place for every GEMM of the
//! pass but the router's, which is f32. The pass's step record slot, where
//! its faults land and its head writes its pick, is its last row's: the pass
//! is the token at its last position as far as the record goes.

use bloomery_device::fault::LAYER_HEAD;
use bloomery_device::{Buffer, Elems};
use chain::ChainError;

use super::super::arena::{Arena, MixerBufs, lane_of, layer_word, slot_of};
use super::super::rows::MIX_STRIDE;
use super::super::session::SessionError;
use super::{MetalOps, WHAT, all, all_mut};
use crate::kernels::{
    AttentionPrefillParams, AttnCache, CONV_TAPS, CombineParams, DeltaRowStrides, DeltaRowsParams,
    EXPERT_K, EXPERT_SLOTS, EmbedParams, F32GemmParams, GemmMode, GemmParams, GemmSegParams,
    HEAD_COLS, HEAD_DIM, HEAD_SIMD_GROUPS, HeadNormRowsParams, HeadParams, KV_HEADS, LOGIT_STRIDE,
    NormRowsParams, Operand, QUERY_HEADS, RouteSelectParams, RouteTableParams, RoutedKind,
    RoutedParams, VALUE_HEADS, route_table_words, route_tiles,
};

/// The operand type the pass's GEMMs compute in.
pub(in super::super) const PASS_OPERAND: Operand = Operand::F16;

impl MetalOps<'_, '_> {
    /// The pass's step record slot: its last row's.
    fn pass_slot(&self, m: usize) -> usize {
        slot_of(self.pos + u32::try_from(m).expect("a pass's rows fit u32") - 1)
    }

    fn no_rows<T>() -> Result<T, SessionError> {
        Err(ChainError::shape(WHAT, "a rows arm of ops built with no rows arena").into())
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

    /// Layer `l`'s P1: the norm of the rows, then one GEMM a weight type.
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
            ..
        } = &mut *self.arena;
        let p1 = match &layers[l].mixer {
            MixerBufs::Delta(d) => &d.p1,
            MixerBufs::Attention(a) => &a.p1,
        };
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
        let stride = rows.proj_stride;
        for part in &p1.parts {
            self.kernels.gemm_seg(
                self.enc,
                GemmSegParams {
                    quant: part.quant,
                    segments: &part.segments,
                    weights: all(&part.weights)?,
                    x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                    x_stride: *hidden,
                    acts: m,
                    y: rows
                        .proj
                        .window_mut(Elems(0), Elems((m - 1) * stride + part.out_end))?,
                    y_stride: stride,
                    mode: GemmMode::Store,
                    operand: PASS_OPERAND,
                    fault: &faults[slot],
                    layer: layer_word(l),
                },
            )?;
        }
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s attention over the pass: the prefill flash appends the
    /// chunk's keys and values and runs every row against the cache.
    pub(super) fn rows_attn_core(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let from = self.pos;
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            rope,
            faults,
            eps,
            ctx: cache_rows,
            ..
        } = &mut *self.arena;
        let MixerBufs::Attention(a) = &mut layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: an attention op on a delta layer"),
            )
            .into());
        };
        let stride = rows.proj_stride;
        let at = |range_start: usize| -> (usize, usize) { (range_start, m * stride - range_start) };
        let (q_at, q_len) = at(a.q.start);
        let (k_at, k_len) = at(a.k.start);
        let (v_at, v_len) = at(a.v.start);
        self.kernels.attention_prefill(
            self.enc,
            AttentionPrefillParams {
                head_dim: HEAD_DIM,
                query_heads: QUERY_HEADS,
                kv_heads: KV_HEADS,
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
                q_gain: all(&a.q_norm)?,
                k_gain: all(&a.k_norm)?,
                rope: all(rope)?,
                cache: AttnCache::F16 {
                    k: all_mut(&mut a.k_cache)?,
                    v: all_mut(&mut a.v_cache)?,
                },
                out: rows.mix.window_mut(Elems(0), Elems(m * MIX_STRIDE))?,
                out_stride: MIX_STRIDE,
                fault: &faults[slot],
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s attention output projection, added into the residual.
    pub(super) fn rows_attn_out(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
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
        let MixerBufs::Attention(a) = &layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: an attention op on a delta layer"),
            )
            .into());
        };
        self.kernels.gemm(
            self.enc,
            GemmParams {
                quant: a.out_quant,
                rows: *hidden,
                cols: MIX_STRIDE,
                acts: m,
                weights: all(&a.out)?,
                x: rows.mix.window(Elems(0), Elems(m * MIX_STRIDE))?,
                x_stride: MIX_STRIDE,
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

    /// Layer `l`'s delta step over the rows, and the ring's commit after it.
    pub(super) fn rows_delta_core(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let from = self.pos;
        let to = from + u32::try_from(m).expect("a pass's rows fit u32");
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            lane,
            faults,
            eps,
            ..
        } = &mut *self.arena;
        let MixerBufs::Delta(d) = &mut layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: a delta op on an attention layer"),
            )
            .into());
        };
        let dense = DeltaRowStrides::dense(d.n_k).ok_or_else(|| SessionError::Shape {
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
                n_v: VALUE_HEADS,
                head: HEAD_COLS,
                taps: CONV_TAPS,
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
            ..
        } = &mut *self.arena;
        let MixerBufs::Delta(d) = &layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: a delta op on an attention layer"),
            )
            .into());
        };
        let dense = DeltaRowStrides::dense(d.n_k).ok_or_else(|| SessionError::Shape {
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
        self.kernels.gemm(
            self.enc,
            GemmParams {
                quant: d.out_quant,
                rows: *hidden,
                cols: dense.g,
                acts: m,
                weights: all(&d.out)?,
                x: rows.mix.window(Elems(0), Elems(m * dense.g))?,
                x_stride: dense.g,
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
    /// (the routed rows, then the shared expert's gate), every row's picks
    /// and the route table.
    pub(super) fn rows_route(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
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
            top_k,
            n_experts,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        let slots = m * EXPERT_SLOTS;
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
        let routed_cols = (m - 1) * LOGIT_STRIDE;
        self.kernels.gemm_f32(
            self.enc,
            F32GemmParams {
                rows: *n_experts,
                cols: *hidden,
                acts: m,
                weights: all(&f.gate_inp)?,
                x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                x_stride: *hidden,
                y: rows
                    .logits
                    .window_mut(Elems(0), Elems(routed_cols + *n_experts))?,
                y_stride: LOGIT_STRIDE,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.kernels.gemm_f32(
            self.enc,
            F32GemmParams {
                rows: 1,
                cols: *hidden,
                acts: m,
                weights: all(&f.gate_inp_shexp)?,
                x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                x_stride: *hidden,
                y: rows
                    .logits
                    .window_mut(Elems(*n_experts), Elems(routed_cols + 1))?,
                y_stride: LOGIT_STRIDE,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        self.kernels.route_rows_select(
            self.enc,
            RouteSelectParams {
                rows: m,
                top_k: *top_k,
                logits: rows.logits.window(Elems(0), Elems(m * LOGIT_STRIDE))?,
                logit_stride: LOGIT_STRIDE,
                ids: rows.pick_ids.window_mut(Elems(0), Elems(slots))?,
                weights: rows.pick_weights.window_mut(Elems(0), Elems(slots))?,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        self.kernels.route_table(
            self.enc,
            RouteTableParams {
                slots,
                per_row: EXPERT_SLOTS,
                experts: *n_experts,
                ids: rows.pick_ids.window(Elems(0), Elems(slots))?,
                sorted: rows.sorted.window_mut(Elems(0), Elems(slots))?,
                inv: rows.inv.window_mut(Elems(0), Elems(slots))?,
                tiles: rows
                    .tiles
                    .window_mut(Elems(0), Elems(route_table_words(slots, *n_experts)))?,
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
        let Arena {
            layers,
            faults,
            hidden,
            n_experts,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        let slots = m * EXPERT_SLOTS;
        let max_tiles = route_tiles(slots, *n_experts);
        let table_words = route_table_words(slots, *n_experts);
        self.kernels.gemm_routed(
            self.enc,
            RoutedParams {
                quant: f.gate_up_quant,
                kind: RoutedKind::Pair,
                operand: PASS_OPERAND,
                cols: *hidden,
                rows: EXPERT_K,
                experts: *n_experts,
                stack: all(&f.gate)?,
                stack_up: Some(all(&f.up)?),
                shared: all(&f.shexp_gate)?,
                shared_up: Some(all(&f.shexp_up)?),
                x: rows.xn.window(Elems(0), Elems(m * *hidden))?,
                x_rows: m,
                x_stride: *hidden,
                gather: true,
                y: rows.h.window_mut(Elems(0), Elems(slots * EXPERT_K))?,
                y_stride: EXPERT_K,
                slots,
                max_tiles,
                sorted: rows.sorted.window(Elems(0), Elems(slots))?,
                route_w: rows.pick_weights.window(Elems(0), Elems(slots))?,
                tiles: rows.tiles.window(Elems(0), Elems(table_words))?,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        self.kernels.gemm_routed(
            self.enc,
            RoutedParams {
                quant: f.down_quant,
                kind: RoutedKind::Plain,
                operand: PASS_OPERAND,
                cols: EXPERT_K,
                rows: *hidden,
                experts: *n_experts,
                stack: all(&f.down)?,
                stack_up: None,
                shared: all(&f.shexp_down)?,
                shared_up: None,
                x: rows.h.window(Elems(0), Elems(slots * EXPERT_K))?,
                x_rows: slots,
                x_stride: EXPERT_K,
                gather: false,
                y: rows.y.window_mut(Elems(0), Elems(slots * *hidden))?,
                y_stride: *hidden,
                slots,
                max_tiles,
                sorted: rows.sorted.window(Elems(0), Elems(slots))?,
                route_w: rows.pick_weights.window(Elems(0), Elems(slots))?,
                tiles: rows.tiles.window(Elems(0), Elems(table_words))?,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        self.kernels.combine_rows(
            self.enc,
            CombineParams {
                rows: m,
                per_row: EXPERT_SLOTS,
                cols: *hidden,
                y: rows.y.window(Elems(0), Elems(slots * *hidden))?,
                y_rows: slots,
                y_stride: *hidden,
                inv: rows.inv.window(Elems(0), Elems(slots))?,
                x: rows.x.window_mut(Elems(0), Elems(m * *hidden))?,
                x_stride: *hidden,
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// The head over the pass's last row.
    pub(super) fn rows_head(&mut self, m: usize) -> Result<(), SessionError> {
        let slot = self.pass_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return Self::no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            head_w,
            head_quant,
            head_norm,
            logits,
            candidates,
            head_ticket,
            step_id,
            faults,
            eps,
            hidden,
            vocab,
            ..
        } = &mut *self.arena;
        let last = rows.x.window(Elems((m - 1) * *hidden), Elems(*hidden))?;
        self.kernels.head(
            self.enc,
            HeadParams {
                quant: *head_quant,
                rows: *vocab,
                k: *hidden,
                simdgroups: HEAD_SIMD_GROUPS,
                weights: all(head_w)?,
                x: last,
                gain: all(head_norm)?,
                logits: all_mut(logits)?,
                candidates: all_mut(candidates)?,
                ticket: all_mut(head_ticket)?,
                step_id: step_id.window_mut(Elems(slot), Elems(1))?,
                eps: *eps,
                fault: &faults[slot],
                layer: LAYER_HEAD,
            },
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::router;

    /// The logit rows hold the routed experts and the shared gate, at a stride
    /// the GEMM's output rows are aligned to.
    const _: () = assert!(
        router::EXPERTS + 1 == router::ROWS
            && LOGIT_STRIDE >= router::ROWS
            && LOGIT_STRIDE.is_multiple_of(4)
    );
}
