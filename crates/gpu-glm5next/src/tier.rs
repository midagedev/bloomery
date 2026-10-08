//! The GLM-5.3-Flash expert tier ([`bloomery_gpu::host::tier`]): a card
//! beside the stage card that holds some of each routed layer's experts and
//! computes the routed slots the slot map sends it, in the host leg's shadow.
//!
//! The stage card's side is [`TierSide`]: the slot map's tier view, which the
//! tier handoff (`ds41_ffn_handoff_8_tier`) reads each slot's tier place
//! from, and each row's tier places, which the tier card sum
//! (`ds41_ffn_card_acc_8_tier`) reads. The row's tier image carries the
//! normed activation in f32 ([`TierAct::F32`]): GLM's `_sel` entries read
//! the q8_1 planes of Walk A, which the stage card makes in the shadow,
//! after the go; the tier quantizes the same values with the same kernel.
//!
//! The tier's own layer ([`GlmTier`], the architecture's [`TierExperts`]) is
//! the stage card's card-slot launches over the tier's stacks: the q8_1 of
//! the activation, the gate·up `_sel` with the routed clamped SwiGLU, the
//! q8_1 of the tier slots' columns and the down `_sel` into the tier's rows;
//! over a prompt batch's block, the same launches chunk by chunk of
//! [`CHUNK`] tokens, as the stage card's `card_rows` runs them. Each slot's
//! down output is the one the stage card would write for it, so the card sum
//! over both cards' slots in slot order is the one-card sum over their union.

use std::ops::Range;

use bloomery_gpu::host::tier::{
    TierAct, TierBlock, TierCard, TierExperts, TierInput, TierIo, TierOpen, TierSet, TierShape,
};
use bloomery_gpu::hybrid::SlotMap;
use bloomery_gpu::kquant::{Act, GateUpAct, KquantKernels, SelDown};
use bloomery_gpu::q4k_sel::QuantSel;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act};
use bloomery_gpu_deepseek41::router::glm5next::{N_EXPERT, N_USED};
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::DeviceBuffer;
use gguf::{GgmlType, Split};
use model::placement::Plan;
use models::{Ffn, LayerSpec};

use crate::body::prefill::CHUNK;
use crate::ffn::{CardStacks, stack_names, unrun};

/// What the tier's errors name.
const WHAT: &str = "glm5next tier";

// The plan reserves the tier's chunk scratch at the body's chunk.
const _: () = assert!(model::arch::glm5next::place::TIER_CHUNK as usize == CHUNK);

/// The tier the stage card's tier entries serve: they hand one tier its
/// image and join one tier's rows, so a load holds one tier card
/// ([`bloomery_gpu::host::refuse_tier_count`]).
pub(crate) const STAGE_TIER: usize = 0;

/// The stage card's side of the tier layers: the slot map's tier view (a row
/// of places per layer at the card copy's row offsets), each row's tier
/// places of its routed slots, and per layer of the description the experts
/// the tier holds. Built once at load by a body with a tier card.
pub(crate) struct TierSide {
    places: DeviceTensor<u32>,
    tsel: Vec<DeviceBuffer<u32>>,
    k: Vec<usize>,
}

impl TierSide {
    /// The side for `map`'s tier [`STAGE_TIER`] over `layers` layers of the
    /// description (a layer outside the map holds no tier expert), `rows`
    /// rows. Load-time only.
    pub(crate) fn new(
        gpu: &Gpu,
        map: &SlotMap,
        layers: usize,
        rows: usize,
    ) -> Result<TierSide, GpuError> {
        let stream = gpu.stream();
        let run = map.layers();
        let k = (0..layers)
            .map(|l| {
                if run.contains(&l) {
                    map.on_tier_of(STAGE_TIER, l)
                } else {
                    Ok(0)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TierSide {
            places: DeviceTensor::upload(
                stream,
                &map.tier_view(STAGE_TIER)?,
                run.len(),
                map.n_expert(),
            )?,
            tsel: (0..rows)
                .map(|_| DeviceBuffer::zeroed(stream, N_USED))
                .collect::<Result<Vec<_>, _>>()?,
            k,
        })
    }

    /// The tier's experts of layer `l`: 0 off the tier.
    pub(crate) fn k(&self, l: usize) -> usize {
        self.k.get(l).copied().unwrap_or(0)
    }

    /// The card copy of the map's tier view.
    pub(crate) fn places(&self) -> &DeviceBuffer<u32> {
        self.places.buf()
    }

    /// Row `row`'s tier places; a row past the load's is refused by name.
    pub(crate) fn tsel(&self, row: usize) -> Result<&DeviceBuffer<u32>, GpuError> {
        let n = self.tsel.len();
        self.tsel.get(row).ok_or_else(|| past_rows(n, row))
    }

    /// What row `row`'s tier handoff reads and writes: the tier view and
    /// the row's tier places; a row past the load's is refused by name.
    pub(crate) fn handoff_parts(
        &mut self,
        row: usize,
    ) -> Result<(&DeviceBuffer<u32>, &mut DeviceBuffer<u32>), GpuError> {
        let n = self.tsel.len();
        let tsel = self.tsel.get_mut(row).ok_or_else(|| past_rows(n, row))?;
        Ok((self.places.buf(), tsel))
    }

    /// Device bytes the side holds.
    pub(crate) fn bytes(&self) -> usize {
        self.places.buf().num_bytes() + self.tsel.iter().map(DeviceBuffer::num_bytes).sum::<usize>()
    }
}

fn past_rows(n: usize, row: usize) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: format!("row {row} of the tier side's {n} rows"),
    }
}

/// A tier layer's experts and its routed SwiGLU limit.
#[derive(Clone, Copy, Debug)]
struct TierLayer {
    k: usize,
    limit: f32,
}

/// The GLM tier card's computation ([`TierExperts`]): per tier layer the
/// stage card's card-slot launches over the tier's stacks into the tier's
/// rows; over a prompt batch's block, the same chunk by chunk. Its scratch —
/// the step's activation in q8_1, the gate·up rows and their q8_1, and per
/// chunk the same for [`CHUNK`] tokens and each chunk width — is made at
/// load, as the stage card's is.
pub(crate) struct GlmTier {
    kq: KquantKernels,
    layers: Range<usize>,
    /// Per layer of `layers`, `None` off the tier.
    cfg: Vec<Option<TierLayer>>,
    n_embd: usize,
    ff: usize,
    act_x: Q8Act,
    h: DeviceBuffer<f32>,
    act_h: Q8Act,
    block_x: Q8Act,
    block_h: DeviceBuffer<f32>,
    block_act_h: Vec<Q8Act>,
}

impl GlmTier {
    /// The tier's computation on `gpu` over `set`, whose stacks `w` holds,
    /// for the description `layers`, `n_embd` wide with routed experts `ff`
    /// wide. A tier layer whose three stacks are absent, of a type the card
    /// experts do not read, or of other rows than its experts', and a layer
    /// off the tier with resident stacks, are refused by name. Load-time
    /// only.
    pub(crate) fn new(
        gpu: &Gpu,
        layers: &[LayerSpec],
        set: &TierSet,
        w: &Weights,
        n_embd: usize,
        ff: usize,
    ) -> Result<GlmTier, GpuError> {
        let run = set.layers();
        let mut cfg = Vec::with_capacity(run.len());
        for l in run.clone() {
            let k = set.on_tier(l)?;
            if k == 0 {
                if let Some(n) = stack_names(l).iter().find(|n| w.get(n).is_some()) {
                    return Err(GpuError::Tensor {
                        what: WHAT,
                        name: n.clone(),
                        need: "no resident rows: the tier holds no expert of its layer",
                    });
                }
                cfg.push(None);
                continue;
            }
            let Some(Ffn::Moe(m)) = layers.get(l).map(|s| &s.ffn) else {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!("layer {l}: tier experts off a routed block"),
                });
            };
            let st = CardStacks::of(w, l)?;
            for (s, rpe) in [(&st.gate, ff), (&st.up, ff), (&st.down, n_embd)] {
                if s.w.rows() != k * rpe {
                    return Err(GpuError::Tensor {
                        what: WHAT,
                        name: s.name.clone(),
                        need: "the rows of the tier's experts of its layer",
                    });
                }
            }
            cfg.push(Some(TierLayer {
                k,
                limit: m.act.swiglu_limit(),
            }));
        }
        let stream = gpu.stream();
        Ok(GlmTier {
            kq: KquantKernels::load(gpu.context(), gpu.fault_word())?,
            layers: run,
            cfg,
            n_embd,
            ff,
            act_x: Q8Act::with_k(stream, 1, n_embd)?,
            h: DeviceBuffer::zeroed(stream, N_USED * ff)?,
            act_h: Q8Act::with_slots(stream, N_USED, ff)?,
            block_x: Q8Act::with_k(stream, CHUNK, n_embd)?,
            block_h: DeviceBuffer::zeroed(stream, CHUNK * N_USED * ff)?,
            block_act_h: (1..=CHUNK)
                .map(|c| Q8Act::with_slots(stream, c * N_USED, ff))
                .collect::<Result<_, _>>()?,
        })
    }

    /// Layer `layer`'s tier experts and limit; a layer the tier holds none
    /// of is refused by name.
    fn layer(&self, layer: usize, what: &'static str) -> Result<TierLayer, GpuError> {
        layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.cfg.get(i))
            .copied()
            .flatten()
            .ok_or_else(|| GpuError::Shape {
                what,
                detail: format!("layer {layer} holds no tier expert"),
            })
    }
}

impl TierExperts for GlmTier {
    /// The chunk scratch: the q8_1 of [`CHUNK`] tokens, their gate·up rows
    /// and, per chunk width, the q8_1 of its slots' columns.
    fn block_bytes(&self) -> usize {
        self.block_x.device_bytes()
            + self.block_h.num_bytes()
            + self
                .block_act_h
                .iter()
                .map(Q8Act::device_bytes)
                .sum::<usize>()
    }

    /// Per chunk of up to [`CHUNK`] tokens, the stage card's `card_rows`
    /// launches over the tier's stacks but its card sum: the chunk's q8_1,
    /// the gate·up `_sel`, the q8_1 of its tier slots' columns and the down
    /// `_sel` into the chunk's slots of `io.down`.
    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierBlock<'_>,
    ) -> Result<(), GpuError> {
        const W: &str = "GlmTier::enqueue_block";
        let tl = self.layer(layer, W)?;
        let st = CardStacks::of(weights, layer)?;
        let stream = gpu.stream();
        let fault = gpu.layer_sink(layer)?;
        let n = self.n_embd;
        for c0 in (0..io.cols).step_by(CHUNK) {
            let cn = CHUNK.min(io.cols - c0);
            let slots = cn * N_USED;
            let x = span(W, io.x, c0 * n, cn * n)?;
            let sel = span(W, io.sel, c0 * N_USED, slots)?;
            gpu.enqueue_quantize_q8_1_cols(&x, &mut self.block_x, cn, layer)?;
            let a = GateUpAct {
                wg: st.gate.w,
                wu: st.up.w,
                act: &self.block_x,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: self.ff,
                slots_per_col: N_USED,
                rule: Act::SwigluClamp { limit: tl.limit },
            };
            match st.gate.ty {
                GgmlType::Q4_K => {
                    self.kq
                        .enqueue_gate_up_q4k(stream, &a, fault, &mut self.block_h)?
                }
                GgmlType::Q5_K => {
                    self.kq
                        .enqueue_gate_up_q5k(stream, &a, fault, &mut self.block_h)?
                }
                _ => return Err(unrun(&st.gate)),
            }
            let act_h = self.block_act_h.get_mut(cn - 1).ok_or(GpuError::State {
                what: W,
                missing: "the q8_1 columns of a chunk that wide",
            })?;
            let q = QuantSel {
                x: &self.block_h,
                cols: 0..slots,
                sel: &sel,
                n_card: tl.k,
            };
            gpu.q4k_sel()
                .enqueue_quantize_sel(stream, &q, fault, act_h)?;
            let mut down = span_mut(W, &mut *io.down, c0 * N_USED * n, slots * n)?;
            match st.down.ty {
                GgmlType::Q5_K => {
                    let a = SelDown {
                        w: st.down.w,
                        act: act_h,
                        sel: &sel,
                        n_slots: slots,
                        rows_per_expert: n,
                    };
                    self.kq.enqueue_gemv_q5k_sel(stream, &a, fault, &mut down)?;
                }
                GgmlType::Q4_K => gpu
                    .q4k_sel()
                    .enqueue_gemv_q4k_sel(stream, st.down.w, act_h, &sel, slots, n, &mut down)?,
                _ => return Err(unrun(&st.down)),
            }
        }
        Ok(())
    }

    /// The stage card's card-slot launches of one token over the tier's
    /// stacks but its card sum: the staged activation's q8_1, the gate·up
    /// `_sel`, the q8_1 of the tier slots' columns and the down `_sel` into
    /// the row's routed rows.
    fn enqueue_layer(
        &mut self,
        gpu: &Gpu,
        weights: &Weights,
        layer: usize,
        io: TierIo<'_>,
    ) -> Result<(), GpuError> {
        const W: &str = "GlmTier::enqueue_layer";
        let tl = self.layer(layer, W)?;
        let TierInput::F32(x) = io.act else {
            return Err(GpuError::Shape {
                what: W,
                detail: "a q8_1 tier image: GLM's tier quantizes the f32 activation itself"
                    .to_string(),
            });
        };
        let st = CardStacks::of(weights, layer)?;
        let stream = gpu.stream();
        let fault = gpu.layer_sink(layer)?;
        let n = self.n_embd;
        gpu.enqueue_quantize_q8_1_layer(x, &mut self.act_x, layer)?;
        let a = GateUpAct {
            wg: st.gate.w,
            wu: st.up.w,
            act: &self.act_x,
            sel: io.sel,
            n_slots: N_USED,
            rows_per_expert: self.ff,
            slots_per_col: N_USED,
            rule: Act::SwigluClamp { limit: tl.limit },
        };
        match st.gate.ty {
            GgmlType::Q4_K => self
                .kq
                .enqueue_gate_up_q4k(stream, &a, fault, &mut self.h)?,
            GgmlType::Q5_K => self
                .kq
                .enqueue_gate_up_q5k(stream, &a, fault, &mut self.h)?,
            _ => return Err(unrun(&st.gate)),
        }
        let q = QuantSel {
            x: &self.h,
            cols: 0..N_USED,
            sel: io.sel,
            n_card: tl.k,
        };
        gpu.q4k_sel()
            .enqueue_quantize_sel(stream, &q, fault, &mut self.act_h)?;
        match st.down.ty {
            GgmlType::Q5_K => {
                let a = SelDown {
                    w: st.down.w,
                    act: &self.act_h,
                    sel: io.sel,
                    n_slots: N_USED,
                    rows_per_expert: n,
                };
                self.kq.enqueue_gemv_q5k_sel(stream, &a, fault, io.rows)
            }
            GgmlType::Q4_K => gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                st.down.w,
                &self.act_h,
                io.sel,
                N_USED,
                n,
                io.rows,
            ),
            _ => Err(unrun(&st.down)),
        }
    }
}

/// The tier card `t` of `plan`, tier `tier` of `map`, for the stage card
/// `stage`'s load: its card found by name, its routed segments uploaded, its
/// set the map's rows of that tier, GLM's tier computation over them (the
/// description `layers`, `n_embd` wide, experts `ff` wide), its page rows
/// `rows`; `card_dontneed` as for the stage card's segments. The stage
/// card's context is current again on return. Load-time only.
#[allow(
    clippy::too_many_arguments,
    reason = "the stage card, the file, the plan, the tier and its index, the description, the map, the widths, the rows and the page lever (rust-quality R8)"
)]
pub(crate) fn open_tier(
    stage: &Gpu,
    file: &Split,
    plan: &Plan<'_>,
    t: &TierOpen,
    tier: usize,
    layers: &[LayerSpec],
    map: &SlotMap,
    [n_embd, ff, rows]: [usize; 3],
    card_dontneed: bool,
) -> Result<TierCard, GpuError> {
    if map.n_expert() != N_EXPERT {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "a map of {} experts; the tier is built for {N_EXPERT}",
                map.n_expert()
            ),
        });
    }
    let gpu = Gpu::open_card(&t.name, t.device)?;
    let w = Weights::load_placed(gpu.stream(), file, plan, t.card, card_dontneed)?;
    let set = TierSet::of_map(map, tier)?;
    let experts = GlmTier::new(&gpu, layers, &set, &w, n_embd, ff)?;
    let card = TierCard::open(
        gpu,
        t.name.clone(),
        w,
        set,
        Box::new(experts),
        TierShape {
            hidden: n_embd,
            n_used: N_USED,
            rows,
            act: TierAct::F32,
        },
    )?;
    stage.context().bind_to_thread()?;
    Ok(card)
}
