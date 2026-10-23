//! The qwen35moe chain's ops on Metal: each op encodes its kernels on the
//! token's one encoder, in the token DAG's order, and closes its group with a
//! barrier.
//!
//! The DAG, one group per op past the embed: `E | P1 | mixer | out | R | G |
//! W` for every layer, then `H`. P1 is one dispatch per weight type of the
//! layer's projections in one group. The FFN norm has no group of its own:
//! the router (R) and the gate-up pair (G) each norm the residual as they
//! load it. The head is the token's last group and closes no barrier. Every op only encodes: no wait, no allocation past the
//! launches' own argument lists, no host round trip.

use bloomery_device::fault::LAYER_HEAD;
use bloomery_device::{Buffer, DeviceError, Elem, Elems, Window, WindowMut};
use chain::ChainError;
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::plan::{DeltaPlan, FfnPlan, GqaPlan};

use super::arena::{Arena, LANES, MixerBufs, layer_word};
use super::session::SessionError;
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::ComputeEncoder;
use crate::kernels::router::RouterParams;
use crate::kernels::{
    AttentionParams, AttnCache, CONV_TAPS, DeltaParams, EXPERT_K, EXPERT_SLOTS, EmbedParams,
    FoldParams, HEAD_COLS, HEAD_DIM, HEAD_SIMD_GROUPS, HeadNormParams, HeadParams, KV_HEADS,
    Kernels, MatvecParams, Mode, NormOnLoad, PairParams, QUERY_HEADS, ROUTED_IDS, SegParams,
    SharedPlan, VALUE_HEADS, attention_dispatches, router, segments_for_keys,
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

/// The ops of one token over the session's arena, encoding on `enc`. `pos`
/// is the token's position: its cache row and its conv ring slot.
pub(super) struct MetalOps<'a, 'e> {
    enc: &'a mut ComputeEncoder<'e>,
    arena: &'a mut Arena,
    kernels: &'a Kernels,
    pos: u32,
}

impl<'a, 'e> MetalOps<'a, 'e> {
    pub(super) fn new(
        enc: &'a mut ComputeEncoder<'e>,
        arena: &'a mut Arena,
        kernels: &'a Kernels,
        pos: u32,
    ) -> Self {
        Self {
            enc,
            arena,
            kernels,
            pos,
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
        let Arena {
            x,
            proj,
            layers,
            fault,
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
                        fault,
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

impl ChainOps for MetalOps<'_, '_> {
    type Error = SessionError;

    fn embed(&mut self, m: usize) -> Result<(), SessionError> {
        Self::one_row(m)?;
        let Arena {
            embed,
            vocab,
            hidden,
            id_in,
            x,
            fault,
            ..
        } = &mut *self.arena;
        self.kernels.embed(
            self.enc,
            EmbedParams {
                table: all(embed)?,
                vocab: *vocab,
                cols: *hidden,
                ids: all(id_in)?,
                out: x.window_mut(Elems(0), Elems(*hidden))?,
                fault,
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
        let pos = self.pos;
        let n_keys = pos + 1;
        let segments = segments_for_keys(n_keys);
        if attention_dispatches(segments) != 1 {
            return Err(SessionError::Shape {
                what: WHAT,
                detail: format!(
                    "{n_keys} live keys take the merge, which this session does not encode"
                ),
            });
        }
        let Arena {
            proj,
            layers,
            rope,
            attn,
            attn_partials,
            fault,
            eps,
            ctx,
            ..
        } = &mut *self.arena;
        let a = attention_of!(layers, l.layer);
        self.kernels.attention_flash(
            self.enc,
            AttentionParams {
                head_dim: HEAD_DIM,
                query_heads: QUERY_HEADS,
                kv_heads: KV_HEADS,
                segments: segments as usize,
                ctx: *ctx,
                eps: *eps,
                layer: layer_word(l.layer),
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
                partials: all_mut(attn_partials)?,
                out: all_mut(attn)?,
                fault,
            },
        )?;
        self.close();
        Ok(())
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
        let Arena {
            proj,
            layers,
            g,
            delta_partials,
            lane,
            fault,
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
                faults: fault,
                n_v: VALUE_HEADS,
                head: HEAD_COLS,
                taps: CONV_TAPS,
                n_k: d.n_k,
                map: d.map,
                lanes: LANES,
                lane_at: 0,
                layer: layer_word(l.layer),
                pos: &pos,
                eps: *eps,
            },
        )?;
        self.close();
        Ok(())
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        let Arena {
            layers,
            g,
            delta_partials,
            x,
            fault,
            eps,
            hidden,
            ..
        } = &mut *self.arena;
        let d = delta_of!(layers, l.layer);
        self.kernels.headnorm_resid(
            self.enc,
            HeadNormParams {
                rows: *hidden,
                cols: g.len().get(),
                weights: all(&d.out)?,
                g: all(g)?,
                partials: all(delta_partials)?,
                y: x.window_mut(Elems(0), Elems(*hidden))?,
                eps: *eps,
                fault,
                layer: layer_word(l.layer),
            },
        )?;
        self.close();
        Ok(())
    }

    fn route(&mut self, l: LayerCtx<'_>, _: &FfnPlan) -> Result<(), SessionError> {
        let Arena {
            layers,
            x,
            fault,
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
                fault,
                layer: layer_word(l.layer),
            },
        )?;
        self.close();
        Ok(())
    }

    fn experts(&mut self, l: LayerCtx<'_>, _: &FfnPlan) -> Result<(), SessionError> {
        let Arena {
            layers,
            x,
            h,
            fault,
            hidden,
            eps,
            n_experts,
            ..
        } = &mut *self.arena;
        let f = &layers[l.layer].ffn;
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
                fault,
                layer: layer_word(l.layer),
            },
        )?;
        self.enc.memory_barrier();
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
                fault,
                layer: layer_word(l.layer),
            },
        )?;
        self.close();
        Ok(())
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        Self::one_row(m)?;
        let Arena {
            head_w,
            head_norm,
            x,
            logits,
            candidates,
            head_ticket,
            step_id,
            fault,
            eps,
            hidden,
            vocab,
            ..
        } = &mut *self.arena;
        self.kernels.head(
            self.enc,
            HeadParams {
                rows: *vocab,
                k: *hidden,
                simdgroups: HEAD_SIMD_GROUPS,
                weights: all(head_w)?,
                x: x.window(Elems(0), Elems(*hidden))?,
                gain: all(head_norm)?,
                logits: all_mut(logits)?,
                candidates: all_mut(candidates)?,
                ticket: all_mut(head_ticket)?,
                step_id: all_mut(step_id)?,
                eps: *eps,
                fault,
                layer: LAYER_HEAD,
            },
        )?;
        Ok(())
    }
}
