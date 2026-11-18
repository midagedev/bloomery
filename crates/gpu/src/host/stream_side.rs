//! A placed body's side of the common host streaming: its prompt calls' stream ([`PromptStream`],
//! the `BLOOMERY_XSTREAM` lever) and the buffers the card's ring pass reads and writes once a
//! `split` armed them. A body whose card route runs its routed slots by the stack's places and the
//! stream's ring beside them holds one of these and hands it its own facts as data: its experts'
//! shape and forms ([`RingShape`]), the card bytes the ring may take ([`StreamSide::set_ring_room`]),
//! and per layer its card stacks, its card count and the rows the ring pass reads.
//!
//! The ring pass of a layer ([`RingPass::enqueue`]) runs the streamed experts' slots as the common
//! route GEMM over the half's stacks ([`RingGemm`]), batch by batch behind each landing batch's
//! event, their down rows written beside the card slots'; the union's places it writes beside them
//! (the stage's below `n_card`, `n_card + i` for the `i`-th streamed expert) are what the layer's
//! card sum reads ([`StreamSide::union_places`]), so a streamed expert's row joins the sum where a
//! host expert's would.

use cuda_core::DeviceBuffer;
use model::arch::place_hybrid::RingForms;

use super::HostExperts;
use super::HostTier;
use super::prompt_stream::PromptStream;
use super::xstream::{RingLayer, XCfg, XMode};
use crate::card_leg::{LegStacks, with_ring};
use crate::card_slots::{CardSlotKernels, ColPlaces};
use crate::gemm::{Gemm32Kernels, GemmKernels};
use crate::kquant::Act;
use crate::ring_gemm::RingGemm;
use crate::{FaultSink, Gpu, GpuError};

const WHAT: &str = "host::stream_side";

/// The ring pass's buffers' shape: the rows of a prompt unit, the routed slots a row, the experts'
/// input width and an expert's gate and up width, in the forms the routed stacks read.
#[derive(Clone, Copy, Debug)]
pub struct RingShape {
    pub rows: usize,
    pub top_k: usize,
    pub n: usize,
    pub ff: usize,
    pub forms: RingForms,
}

/// A body's prompt streaming and its ring pass's buffers.
#[derive(Default)]
pub struct StreamSide {
    /// The calls' host streaming: off until [`StreamSide::set_xstream`] sets it.
    pub stream: PromptStream,
    /// The ring pass's places and route GEMM buffers, from the first `split` on.
    ring: Option<RingPlaces>,
    /// The card bytes the ring may take past the stream's keep-free: the plan's ring term less the
    /// pass's buffers, so a ring never takes the card's margin or the pass's room; set at the load.
    ring_room: Option<u64>,
}

/// The ring pass's buffers: the union's places the layer's card sum reads, a unit's slots, and the
/// route GEMM's buffers over a half of the ring.
struct RingPlaces {
    union: DeviceBuffer<u32>,
    gemm: RingGemm,
}

/// One layer's ring pass: the stream's view of the layer, the places' kernels and the buffers the
/// pass writes ([`StreamSide::ring_pass`]).
pub struct RingPass<'a> {
    pub r: RingLayer,
    acc_k: &'a CardSlotKernels,
    union: &'a mut DeviceBuffer<u32>,
    gemm: &'a mut RingGemm,
}

/// The rows a layer's ring pass reads: `m` rows of the experts' input `x`, their `top_k` routed ids
/// a row over `n_expert` experts.
#[derive(Clone, Copy)]
pub struct RingRows<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub ids: &'a DeviceBuffer<u32>,
    pub m: usize,
    pub top_k: usize,
    pub n_expert: usize,
}

impl StreamSide {
    /// The ring's room cap, the plan's ring reserve. Load-time only.
    pub fn set_ring_room(&mut self, bytes: u64) {
        self.ring_room = Some(bytes);
    }

    /// The ring's room cap the load set; `None` takes what the card's margin leaves free.
    #[must_use]
    pub fn ring_room(&self) -> Option<u64> {
        self.ring_room
    }

    /// What the next prompt calls move: `Off` on a load with no host tier (`tier` `None`) and
    /// refused by name otherwise there. `Split` starts the expert stream at its first setting
    /// ([`PromptStream::start_split`], `ring`'s stream configuration: the ring sized from the
    /// card's free bytes past the stream's keep-free and the ring's room cap) and arms the ring
    /// pass's buffers at `ring`'s shape; the stream then stays for the load's life, idle under
    /// `admit` and `off`. Refused by name as `what`: `admit` or `split` on a load with no host
    /// tier, `ring`'s, and [`PromptStream::check_mode`]'s and the stream's start's refusals.
    pub fn set_xstream<H: HostExperts>(
        &mut self,
        tier: Option<&mut HostTier<H>>,
        (gpu, what): (&Gpu, &'static str),
        mode: XMode,
        ring: impl FnOnce(Option<u64>) -> Result<(XCfg, RingShape), GpuError>,
    ) -> Result<(), GpuError> {
        let Some(tier) = tier else {
            if mode == XMode::Off {
                self.stream.set_mode(mode);
                return Ok(());
            }
            return Err(GpuError::state(
                what,
                "a placed load (the host tier the stream serves)",
            ));
        };
        self.stream.check_mode(tier, mode)?;
        if mode == XMode::Split {
            let (cfg, shape) = ring(self.ring_room)?;
            if let Some(half) = PromptStream::start_split(tier, gpu, || cfg)? {
                self.arm(gpu, shape, half)?;
            }
        }
        self.stream.set_mode(mode);
        Ok(())
    }

    /// The ring pass's buffers at `shape` over a half of `half_slots` slots. Load-time only (the
    /// first `split`); a second arming is refused by name.
    fn arm(&mut self, gpu: &Gpu, shape: RingShape, half_slots: usize) -> Result<(), GpuError> {
        if self.ring.is_some() {
            return Err(GpuError::state(
                WHAT,
                "a stream side not yet armed for the ring",
            ));
        }
        let RingShape {
            rows,
            top_k,
            n,
            ff,
            forms,
        } = shape;
        self.ring = Some(RingPlaces {
            union: DeviceBuffer::zeroed(gpu.stream(), rows * top_k)?,
            gemm: RingGemm::new(gpu, (rows, top_k, n, ff), half_slots, forms)?,
        });
        Ok(())
    }

    /// Layer `r`'s ring pass with `acc_k`'s places: the armed buffers lent to it. Refused by name:
    /// a stream with no armed ring.
    pub fn ring_pass<'a>(
        &'a mut self,
        r: RingLayer,
        acc_k: &'a CardSlotKernels,
    ) -> Result<RingPass<'a>, GpuError> {
        let p = self.ring.as_mut().ok_or(GpuError::state(
            WHAT,
            "the ring pass's places (the first split arms them)",
        ))?;
        Ok(RingPass {
            r,
            acc_k,
            union: &mut p.union,
            gemm: &mut p.gemm,
        })
    }

    /// The union's places the last ring pass wrote, which its layer's card sum reads. Refused by
    /// name: a stream with no armed ring.
    pub fn union_places(&self) -> Result<&DeviceBuffer<u32>, GpuError> {
        self.ring.as_ref().map(|p| &p.union).ok_or(GpuError::state(
            WHAT,
            "the ring pass's places (the first split arms them)",
        ))
    }

    /// The device bytes of the ring pass's buffers; 0 before a `split` armed them.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.ring
            .as_ref()
            .map_or(0, |p| p.union.num_bytes() + p.gemm.bytes())
    }
}

impl RingPass<'_> {
    /// The layer's streamed slots beside the card's: `rows`' input in the route GEMM's form, the
    /// union's places of every slot (the stage's below `n_card`, `n_card + i` for the `i`-th
    /// streamed expert) for the card sum, then per landing batch, behind its own landed event, the
    /// route GEMM over the half's stacks — in the forms of the layer's card stacks `card` of
    /// `n_card` experts, `ff` rows a gate and up expert and `n` a down expert — of the batch's
    /// slots under `rule`, each slot's down row into its row of `y` beside the card slots' (a slot
    /// outside the batch neither read nor written). The streamed experts the pass joined, which
    /// the card sum adds to `n_card`. Asynchronous, allocation-free.
    pub fn enqueue(
        self,
        gpu: &Gpu,
        kernels: (Option<&GemmKernels>, &Gemm32Kernels),
        (card, n_card, ff, n): (&LegStacks<'_>, usize, usize, usize),
        rows: RingRows<'_>,
        (y, rule): (&mut DeviceBuffer<f32>, Act),
        fault: FaultSink,
    ) -> Result<usize, GpuError> {
        let RingPass {
            r,
            acc_k,
            union,
            gemm,
        } = self;
        let ctx = gpu.context();
        let stream = gpu.stream();
        let RingRows {
            x,
            ids,
            m,
            top_k,
            n_expert,
        } = rows;
        gemm.enqueue_rows(gpu, kernels.1, x, m, fault)?;
        r.with_union(ctx, n_expert, |map| {
            acc_k.enqueue_places_cols(
                stream,
                &ColPlaces {
                    ids,
                    ids_pitch: top_k,
                    used: top_k,
                    sel_pitch: top_k,
                    m,
                    map,
                    row_off: 0,
                    n_expert,
                },
                fault,
                union,
            )
        })?;
        with_ring(card, n_card, &r, ctx, (ff, n), |blk| {
            for batch in &r.batches {
                gemm.enqueue_batch(
                    gpu,
                    kernels,
                    blk,
                    batch,
                    (ids, m * top_k, top_k, n_expert),
                    (&mut *y, rule),
                    fault,
                )?;
            }
            Ok(())
        })?;
        Ok(r.n)
    }
}
