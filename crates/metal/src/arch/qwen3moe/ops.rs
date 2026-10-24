//! The qwen35moe chain's ops on Metal: each op encodes its kernels on the
//! token's one encoder, in the token DAG's order, and closes its group with a
//! barrier.
//!
//! The DAG, one group per op past the embed: `E | P1 | mixer | out | R | G |
//! W` for every layer, then `H`. P1 is one dispatch per weight type of the
//! layer's projections in one group. An attention mixer past one segment of
//! keys is two groups: the flash writes its partials, the barrier, then the
//! merge folds them into the output. The FFN norm has no group of its own:
//! the router (R) and the gate-up pair (G) each norm the residual as they
//! load it. The head is the token's last group and closes no barrier; a
//! token whose pick nobody reads encodes none of it. Every op only encodes:
//! no wait, no allocation past the launches' own argument lists, no host
//! round trip.
//!
//! The token at `pos` binds the step record slot of its position: the head
//! writes its id there and every kernel raises that slot's fault word. A
//! host-fed token embeds the host's id slot; a token fed on the device embeds
//! the id the token before it wrote, in that token's slot: its head's pick,
//! or the id the host wrote there for a token ahead. The delta step reads the
//! lane of `pos` and writes the lane of `pos + 1`.

use bloomery_device::fault::LAYER_HEAD;
use bloomery_device::{Buffer, DeviceError, Elem, Elems, Window, WindowMut};
use chain::ChainError;
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::plan::{DeltaPlan, FfnPlan, GqaPlan};

use super::arena::{Arena, LANES, MixerBufs, lane_of, layer_word, slot_before, slot_of};
use super::session::SessionError;
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::ComputeEncoder;
use crate::kernels::router::RouterParams;
use crate::kernels::{
    AttentionParams, AttnCache, CONV_TAPS, DeltaParams, EXPERT_K, EXPERT_SLOTS, EmbedParams,
    FoldParams, HEAD_COLS, HEAD_DIM, HEAD_SIMD_GROUPS, HeadNormParams, HeadParams, KV_HEADS,
    Kernels, MatvecParams, Mode, NormOnLoad, PairParams, QUERY_HEADS, ROUTED_IDS, SegParams,
    SharedPlan, VALUE_HEADS, partials_len, router, segments_for_keys,
};

const WHAT: &str = "qwen35moe ops";

/// The whole of `b`.
fn all<T: Elem>(b: &MetalBuffer<T>) -> Result<Window<'_, T, MetalDevice>, DeviceError> {
    b.window(Elems(0), b.len())
}

/// The whole of `b`, written.
fn all_mut<T: Elem>(b: &mut MetalBuffer<T>) -> Result<WindowMut<'_, T, MetalDevice>, DeviceError> {
    let len = b.len();
    b.window_mut(Elems(0), len)
}

/// Values `r` of `b`.
fn part<'b, T: Elem>(
    b: &'b MetalBuffer<T>,
    r: &core::ops::Range<usize>,
) -> Result<Window<'b, T, MetalDevice>, DeviceError> {
    b.window(Elems(r.start), Elems(r.len()))
}

/// Where a token's embed finds its id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Feed {
    /// The host's id slot, which the host wrote before the commit.
    Host,
    /// The step record slot of the token before it: that token's pick, or
    /// the id the host wrote there for a token ahead whose pick nobody
    /// reads.
    Device,
}

/// The ops of one token over the session's arena, encoding on `enc`. `pos`
/// is the token's position: its cache row, its conv ring slot, its step
/// record slot and its delta lanes. `head` names whether the token encodes
/// the head: a step whose pick nobody reads skips it.
pub(super) struct MetalOps<'a, 'e> {
    enc: &'a mut ComputeEncoder<'e>,
    arena: &'a mut Arena,
    kernels: &'a Kernels,
    pos: u32,
    feed: Feed,
    head: bool,
}

impl<'a, 'e> MetalOps<'a, 'e> {
    pub(super) fn new(
        enc: &'a mut ComputeEncoder<'e>,
        arena: &'a mut Arena,
        kernels: &'a Kernels,
        pos: u32,
        feed: Feed,
        head: bool,
    ) -> Self {
        Self {
            enc,
            arena,
            kernels,
            pos,
            feed,
            head,
        }
    }

    /// The group closes: the next op's dispatches see what this one wrote.
    fn close(&mut self) {
        self.enc.memory_barrier();
    }

    /// One row a step: the only unit this backend's ops encode.
    fn one_row(m: usize) -> Result<(), SessionError> {
        if m == 1 {
            Ok(())
        } else {
            Err(ChainError::shape(WHAT, format!("a unit of {m} rows; a step runs one")).into())
        }
    }

    /// Layer `l`'s P1: the norm on load, one dispatch per weight type, one
    /// group.
    fn p1(&mut self, l: usize) -> Result<(), SessionError> {
        let slot = slot_of(self.pos);
        let Arena {
            x,
            proj,
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
        for part in &p1.parts {
            self.kernels.matvec_seg(
                self.enc,
                SegParams {
                    quant: part.quant,
                    segments: &part.segments,
                    weights: all(&part.weights)?,
                    x: x.window(Elems(0), Elems(*hidden))?,
                    y: proj.window_mut(Elems(0), Elems(part.out_end))?,
                    norm: NormOnLoad {
                        gain: all(&p1.norm)?,
                        eps: *eps,
                        fault: &faults[slot],
                        layer: layer_word(l),
                    },
                },
            )?;
        }
        self.close();
        Ok(())
    }
}

/// The layer's delta buffers, or a named refusal for an attention layer.
macro_rules! delta_of {
    ($layers:expr, $l:expr) => {
        match &mut $layers[$l].mixer {
            MixerBufs::Delta(d) => d,
            MixerBufs::Attention(_) => {
                return Err(ChainError::shape(
                    WHAT,
                    format!("layer {}: a delta op on an attention layer", $l),
                )
                .into());
            }
        }
    };
}

/// The layer's attention buffers, or a named refusal for a delta layer.
macro_rules! attention_of {
    ($layers:expr, $l:expr) => {
        match &mut $layers[$l].mixer {
            MixerBufs::Attention(a) => a,
            MixerBufs::Delta(_) => {
                return Err(ChainError::shape(
                    WHAT,
                    format!("layer {}: an attention op on a delta layer", $l),
                )
                .into());
            }
        }
    };
}

impl MetalOps<'_, '_> {
    /// The attention launch of layer `l` at `pos`: the one shape both its
    /// dispatches read, built per dispatch (the windows it holds are `Copy`,
    /// the parameters are not).
    fn attn_params(
        arena: &mut Arena,
        l: usize,
        pos: u32,
        n_keys: u32,
        segments: u32,
    ) -> Result<AttentionParams<'_>, SessionError> {
        let slot = slot_of(pos);
        let Arena {
            proj,
            layers,
            rope,
            attn,
            attn_partials,
            attn_ticket,
            faults,
            eps,
            ctx,
            ..
        } = &mut *arena;
        let a = attention_of!(layers, l);
        // The arena holds the partials of the cache's whole segment count;
        // the launch reads and writes the prefix this token's keys take.
        let parts = partials_len(segments as usize).ok_or_else(|| SessionError::Shape {
            what: WHAT,
            detail: format!("the partials of {segments} segments do not fit"),
        })?;
        Ok(AttentionParams {
            head_dim: HEAD_DIM,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            segments: segments as usize,
            ctx: *ctx,
            eps: *eps,
            layer: layer_word(l),
            pos,
            n_keys,
            q: part(proj, &a.q)?,
            k: part(proj, &a.k)?,
            v: part(proj, &a.v)?,
            q_gain: all(&a.q_norm)?,
            k_gain: all(&a.k_norm)?,
            rope: all(rope)?,
            cache: AttnCache::F16 {
                k: all_mut(&mut a.k_cache)?,
                v: all_mut(&mut a.v_cache)?,
            },
            partials: attn_partials.window_mut(Elems(0), Elems(parts))?,
            out: all_mut(attn)?,
            fault: &faults[slot],
            ticket: attn_ticket.window_mut(Elems(0), Elems(1))?,
        })
    }
}

impl MetalOps<'_, '_> {
    /// Layer `l`'s flash group: the attention over its `n_keys` live keys in
    /// `segments` segments, this position's keys stored first. The group a
    /// merge follows past one segment.
    pub(super) fn attn_flash(
        &mut self,
        l: usize,
        n_keys: u32,
        segments: u32,
    ) -> Result<(), SessionError> {
        let p = Self::attn_params(&mut *self.arena, l, self.pos, n_keys, segments)?;
        self.kernels.attention_flash(self.enc, p)?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s gate-up group: the routed picks' and the shared expert's
    /// gate and up, the residual normed as it is loaded.
    pub(super) fn experts_pair(&mut self, l: usize) -> Result<(), SessionError> {
        let slot = slot_of(self.pos);
        let Arena {
            layers,
            x,
            h,
            faults,
            hidden,
            eps,
            n_experts,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        let ids = f.ids.window(Elems(0), Elems(ROUTED_IDS))?.cast::<i32>()?;
        self.kernels.pair(
            self.enc,
            PairParams {
                quant: f.gate_up_quant,
                slots: EXPERT_SLOTS,
                k: EXPERT_K,
                cols: *hidden,
                n_experts: *n_experts,
                ids,
                gate: all(&f.gate)?,
                up: all(&f.up)?,
                shared_gate: all(&f.shexp_gate)?,
                shared_up: all(&f.shexp_up)?,
                x: x.window(Elems(0), Elems(*hidden))?,
                gain: all(&f.norm)?,
                eps: *eps,
                y: all_mut(h)?,
                out_stride: EXPERT_K,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s down group: the experts' down projections, the routed
    /// picks' and the shared expert's, folded into the residual.
    pub(super) fn experts_fold(&mut self, l: usize) -> Result<(), SessionError> {
        let slot = slot_of(self.pos);
        let Arena {
            layers,
            x,
            h,
            faults,
            hidden,
            n_experts,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        let ids = f.ids.window(Elems(0), Elems(ROUTED_IDS))?.cast::<i32>()?;
        let y = x.window_mut(Elems(0), Elems(*hidden))?;
        self.kernels.fold(
            self.enc,
            FoldParams {
                quant: f.down_quant,
                shared: SharedPlan::Same(all(&f.shexp_down)?),
                rows: *hidden,
                slots: EXPERT_SLOTS,
                k: EXPERT_K,
                n_experts: *n_experts,
                ids,
                down: all(&f.down)?,
                h: all(h)?,
                w: all(&f.weights)?,
                y: &y,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.close();
        Ok(())
    }
}

impl ChainOps for MetalOps<'_, '_> {
    type Error = SessionError;

    fn embed(&mut self, m: usize) -> Result<(), SessionError> {
        Self::one_row(m)?;
        let slot = slot_of(self.pos);
        let Arena {
            embed,
            embed_quant,
            vocab,
            hidden,
            id_in,
            step_id,
            x,
            faults,
            ..
        } = &mut *self.arena;
        let ids = match self.feed {
            Feed::Host => all(id_in)?,
            Feed::Device => step_id.window(Elems(slot_before(self.pos)), Elems(1))?,
        };
        self.kernels.embed(
            self.enc,
            EmbedParams {
                quant: *embed_quant,
                table: all(embed)?,
                vocab: *vocab,
                cols: *hidden,
                ids,
                out: x.window_mut(Elems(0), Elems(*hidden))?,
                fault: &faults[slot],
                layer: layer_word(0),
            },
        )?;
        self.close();
        Ok(())
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        Self::one_row(l.m)?;
        self.p1(l.layer)
    }

    fn attn_core(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        let n_keys = self.pos + 1;
        let segments = segments_for_keys(n_keys);
        // Live keys past one segment fold inside the flash: its last
        // threadgroup merges the partials and writes the gated output.
        self.attn_flash(l.layer, n_keys, segments)
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        let Arena {
            layers,
            attn,
            x,
            hidden,
            ..
        } = &mut *self.arena;
        let a = attention_of!(layers, l.layer);
        self.kernels.matvec(
            self.enc,
            MatvecParams {
                quant: a.out_quant,
                rows: *hidden,
                cols: attn.len().get(),
                weights: all(&a.out)?,
                x: all(attn)?,
                y: x.window_mut(Elems(0), Elems(*hidden))?,
                mode: Mode::Residual,
            },
        )?;
        self.close();
        Ok(())
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        Self::one_row(l.m)?;
        self.p1(l.layer)
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        let pos = [self.pos];
        let slot = slot_of(self.pos);
        let Arena {
            proj,
            layers,
            g,
            delta_partials,
            lane,
            faults,
            eps,
            ..
        } = &mut *self.arena;
        let d = delta_of!(layers, l.layer);
        self.kernels.delta(
            self.enc,
            DeltaParams {
                x: part(proj, &d.qkv)?,
                conv_w: all(&d.conv)?,
                b_raw: part(proj, &d.beta)?,
                a_raw: part(proj, &d.alpha)?,
                dt_bias: all(&d.dt_bias)?,
                ssm_a: all(&d.ssm_a)?,
                z: part(proj, &d.z)?,
                norm_w: all(&d.ssm_norm)?,
                lane: all(lane)?,
                state: all_mut(&mut d.state)?,
                ring: all_mut(&mut d.ring)?,
                partials: all_mut(delta_partials)?,
                g: all_mut(g)?,
                faults: &faults[slot],
                n_v: VALUE_HEADS,
                head: HEAD_COLS,
                taps: CONV_TAPS,
                n_k: d.n_k,
                map: d.map,
                lanes: LANES,
                lane_at: lane_of(self.pos),
                lane_to: lane_of(self.pos + 1),
                layer: layer_word(l.layer),
                pos: &pos,
                eps: *eps,
            },
        )?;
        self.close();
        Ok(())
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        let slot = slot_of(self.pos);
        let Arena {
            layers,
            g,
            delta_partials,
            x,
            faults,
            eps,
            hidden,
            ..
        } = &mut *self.arena;
        let d = delta_of!(layers, l.layer);
        self.kernels.headnorm_resid(
            self.enc,
            HeadNormParams {
                quant: d.out_quant,
                rows: *hidden,
                cols: g.len().get(),
                weights: all(&d.out)?,
                g: all(g)?,
                partials: all(delta_partials)?,
                y: x.window_mut(Elems(0), Elems(*hidden))?,
                eps: *eps,
                fault: &faults[slot],
                layer: layer_word(l.layer),
            },
        )?;
        self.close();
        Ok(())
    }

    fn route(&mut self, l: LayerCtx<'_>, _: &FfnPlan) -> Result<(), SessionError> {
        let slot = slot_of(self.pos);
        let Arena {
            layers,
            x,
            faults,
            eps,
            hidden,
            top_k,
            ..
        } = &mut *self.arena;
        let f = &mut layers[l.layer].ffn;
        self.kernels.router(
            self.enc,
            RouterParams {
                rows: router::ROWS,
                k: *hidden,
                top_k: *top_k,
                routed: all(&f.gate_inp)?,
                shared: all(&f.gate_inp_shexp)?,
                x: x.window(Elems(0), Elems(*hidden))?,
                gain: all(&f.norm)?,
                partials: all_mut(&mut f.partials)?,
                ticket: all_mut(&mut f.ticket)?,
                logits: all_mut(&mut f.logits)?,
                ids: all_mut(&mut f.ids)?,
                weights: all_mut(&mut f.weights)?,
                eps: *eps,
                fault: &faults[slot],
                layer: layer_word(l.layer),
            },
        )?;
        self.close();
        Ok(())
    }

    fn experts(&mut self, l: LayerCtx<'_>, _: &FfnPlan) -> Result<(), SessionError> {
        self.experts_pair(l.layer)?;
        self.experts_fold(l.layer)
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        Self::one_row(m)?;
        if !self.head {
            // The call said nobody reads this token's pick: the head's
            // dispatch is the whole of what is skipped, and no later op
            // depends on it — the head closes no barrier and writes nothing
            // but its own outputs.
            return Ok(());
        }
        let slot = slot_of(self.pos);
        let Arena {
            head_w,
            head_quant,
            head_norm,
            x,
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
        self.kernels.head(
            self.enc,
            HeadParams {
                quant: *head_quant,
                rows: *vocab,
                k: *hidden,
                simdgroups: HEAD_SIMD_GROUPS,
                weights: all(head_w)?,
                x: x.window(Elems(0), Elems(*hidden))?,
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
