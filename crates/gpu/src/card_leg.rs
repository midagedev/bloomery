//! The card leg of a routed layer's experts, common to every program that runs
//! experts on the card by the kernel table (`model::placement::kernels`):
//! the four launches from a layer's normed rows to its slots' down rows.
//!
//! One run covers `m` tokens of `slots_per_col` routed slots each:
//! - the normed rows in q8_1 (`m` columns, `Gpu::enqueue_quantize_q8_1_cols`);
//! - the gate·up `_sel` of the entry [`kernels::gate_up`] names for the
//!   stacks' type, the activation rule a launch argument, slot `s` against
//!   column `s / slots_per_col`, writing `h` (`ff` rows a slot);
//! - the q8_1 of the card slots' `h` columns in the form the down entry reads
//!   ([`DownAct`]: `Q4kSelKernels::enqueue_quantize_sel` for 128-value
//!   blocks, `Q5Kernels::enqueue_quantize_q8_sel` for 32-value blocks; a slot
//!   the run places off the card is left alone);
//! - the down `_sel` of the entry [`kernels::down`] names, writing `down`
//!   (`n` rows a slot).
//!
//! - optionally the card slots' weighted sum ([`LegSum`], `crate::card_slots`):
//!   a token's slots in slot order, written to a caller's `acc`.
//!
//! A slot's rows are what a one-token run writes for it, whatever `m`: the
//! launches are slot-local and read their columns by slot, so a unit of a
//! prompt batch and a decode step leave the same bits for the same slot.
//!
//! The relu² form ([`Relu2Leg`]) is the same four launches for an expert with
//! no gate, `h = relu(up · x)²`, over the types [`kernels::card_relu2_runs`]
//! admits alone, the 32-value-block types and NVFP4 in its planes:
//! - the token's normed row in 32-value q8_1, once in the column of each of
//!   its slots (`Q5Kernels::enqueue_quantize_q8_rep`), so the up is a
//!   down-shaped `_sel` that reads column `s` for slot `s` ([`kernels::up`])
//!   and no `_sel` entry has a column rule of its own;
//! - the up `_sel`, writing `h` (`ff` rows a slot);
//! - the q8_1 of the card slots' `max(h, 0)²` in the form the down reads
//!   (`Q5Kernels::enqueue_quantize_q8_sel_relu_sqr`; a slot the run places
//!   off the card is left alone);
//! - the down `_sel` and, optionally, the card slots' weighted sum.
//!
//! The widths need not be multiples of 256: `n` and `ff` are multiples of 32.
//!
//! What stays with the caller, because it is the family's: the slots' places
//! `sel` (the handoff's, or a batch's `CardSlotKernels::enqueue_places`), what
//! the sum is combined with (a shared expert, the residual, the host tier's
//! rows) or its own sum where the common one is not its rule (the expert
//! tier's rows beside the card's, another slot count or weight pitch), and
//! the routing weights. A caller whose down rows live in a buffer of its own
//! (the expert tier's staged rows) runs [`CardLeg::enqueue_rows`].
//!
//! The run launches the entries [`kernels::card_gate_up_runs`] and
//! [`kernels::card_down_runs`] admit, each stack by its role, and a Q8_0 gate
//! and up ([`GateUpEntry::Q8_0`]) for a program whose arms name it
//! ([`LegStacks::of_arms`]; the other families' constructors read the first two
//! tables and keep a Q8_0 gate·up on the host). The launchers read an
//! activation of more columns than a run uses (a [`LegBufs`] sized for `cap`
//! tokens serves a unit of fewer in one launch).
//! The buffers hold the down's activation in one form ([`LegBufs::for_down`])
//! or in both, for a caller whose layers' downs read either
//! ([`LegBufs::for_downs`]); a stack whose down entry reads a form they do not
//! hold is refused by name before the first launch.
//! A stack of any other type is refused by name when the stacks are read
//! ([`LegStacks::of`]), and an entry the table names but the run has no arm
//! for is refused by name at the launch. A Q6_K down is in
//! [`kernels::card_down_runs`] and runs over the 128-value columns
//! ([`DownAct::Q8Act`]), the form [`LegBufs::new`] holds.
//!
//! A prompt walk that runs a unit's card slots through a route table reads
//! each resident stack by the GEMM entry [`kernels::gemm`] names for its type
//! ([`StackW::route`], [`LegStacks::block`]); the table and the SwiGLU between
//! the launches stay with the walk, which runs the block as [`RouteBlock`]:
//! the gate and up GEMMs, the SwiGLU into the form the down GEMM reads, and the
//! down GEMM. The expert stream's ring holds a half of host experts in the
//! card stacks' forms, and every family's ring pass reads it through
//! [`with_ring`] as the same block.

use crate::card_slots::{CardAcc, CardSlotKernels, CardSlots};
use crate::expert_input::{Protection, Slots};
use crate::fault::FaultSink;
use crate::gemm::{
    Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput, GemmKernels,
    GemmRoute, GemmWeight,
};
use crate::host::xstream::RingLayer;
use crate::iq_sel::{IqDown, IqGateUp};
use crate::kquant::{Act, GateUpAct, SelDown};
use crate::mxfp4_sel::{Mxfp4SelKernels, Nvfp4Down, Nvfp4GateUp, Nvfp4SelKernels, Nvfp4Stack};
use crate::q4k_sel::QuantSel;
use crate::q5::{Q5Kernels, Q8Blocks32};
use crate::q5_1_sel::{Q51SelDown, Q51SelKernels};
use crate::q6k_sel::Q6kSelKernels;
use crate::q8_0_sel32::{Q80SelDown, Q80SelKernels};
use crate::quant_sel::QuantSelKernels;
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError, Q8Act};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use gguf::GgmlType;
pub use model::ops::ExpertInput;
use model::placement::accepts::{self, LegArms};
use model::placement::kernels::{self, DownAct, DownEntry, DownForms, GateUpEntry, GemmEntry};
use std::sync::Arc;

/// What the leg's errors name.
const WHAT: &str = "CardLeg";

/// What a [`LegBufs`] is sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegShape {
    /// Tokens (columns of the normed rows) one run covers at most.
    pub cap: usize,
    /// Routed slots a token carries: the gate·up's `slots_per_col`.
    pub slots_per_col: usize,
    /// The model's width: a normed row's values and a down slot's rows.
    pub n: usize,
    /// The routed experts' width: a gate·up slot's rows.
    pub ff: usize,
}

impl LegShape {
    /// Slots of a run of `cap` tokens.
    const fn slots(self) -> usize {
        self.cap * self.slots_per_col
    }
}

/// The card slots' q8_1 columns in each form the buffers hold.
struct ActForms {
    q8_act: Option<Q8Act>,
    blocks32: Option<Q8Blocks32>,
}

impl ActForms {
    /// The columns in the form `act`, `None` when the buffers hold none.
    fn of(&mut self, act: DownAct) -> Option<ActH<'_>> {
        match act {
            DownAct::Q8Act => self.q8_act.as_mut().map(ActH::Q8Act),
            DownAct::Q8Blocks32 => self.blocks32.as_mut().map(ActH::Blocks32),
        }
    }

    fn bytes(&self) -> usize {
        self.q8_act.as_ref().map_or(0, Q8Act::device_bytes)
            + self
                .blocks32
                .as_ref()
                .map_or(0, |a| a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes())
    }
}

/// The card slots' q8_1 columns in the form a run's down entry reads.
enum ActH<'b> {
    Q8Act(&'b mut Q8Act),
    Blocks32(&'b mut Q8Blocks32),
}

/// One run's buffers for up to `shape.cap` tokens: the normed rows' q8_1 form,
/// the gate·up's rows, the card slots' q8_1 columns in each form the buffers
/// hold and the down's rows.
pub struct LegBufs {
    shape: LegShape,
    act_x: Q8Act,
    h: DeviceBuffer<f32>,
    act_h: ActForms,
    down: DeviceBuffer<f32>,
    protection: Option<Protection>,
}

impl LegBufs {
    /// The buffers for runs of `shape` whose down entries read [`DownAct::
    /// Q8Act`] (the K-quant, MXFP4 and IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M,
    /// IQ3_XXS, IQ3_S and IQ4_XS downs).
    /// Load-time or first-prompt only.
    pub fn new(stream: &CudaStream, shape: LegShape) -> Result<LegBufs, GpuError> {
        LegBufs::for_down(stream, shape, DownAct::Q8Act)
    }

    /// The buffers for runs of `shape` whose down entries read `act`, the form
    /// [`kernels::down`] names for the stacks' type. Load-time or first-prompt
    /// only.
    pub fn for_down(
        stream: &CudaStream,
        shape: LegShape,
        act: DownAct,
    ) -> Result<LegBufs, GpuError> {
        LegBufs::for_downs(stream, shape, DownForms::of(act))
    }

    /// The buffers for runs of `shape` over layers whose down entries read
    /// either form of `downs`: the slots' columns in each, the rest shared, so
    /// one run's rows land in [`LegBufs::down`] whatever its down's form.
    /// Load-time or first-prompt only.
    pub fn for_downs(
        stream: &CudaStream,
        shape: LegShape,
        downs: DownForms,
    ) -> Result<LegBufs, GpuError> {
        let slots = shape.slots();
        let act_h = ActForms {
            q8_act: downs
                .holds(DownAct::Q8Act)
                .then(|| Q8Act::with_slots(stream, slots, shape.ff))
                .transpose()?,
            blocks32: downs
                .holds(DownAct::Q8Blocks32)
                .then(|| Q8Blocks32::with_slots(stream, shape.ff, slots))
                .transpose()?,
        };
        Ok(LegBufs {
            shape,
            act_x: Q8Act::with_tier_cols(stream, shape.cap, shape.n)?,
            h: DeviceBuffer::zeroed(stream, slots * shape.ff)?,
            act_h,
            down: DeviceBuffer::zeroed(stream, slots * shape.n)?,
            protection: None,
        })
    }

    /// Select the loaded down-input policy before capture or execution.
    pub fn set_expert_input(&mut self, gpu: &Gpu, policy: ExpertInput) -> Result<(), GpuError> {
        match policy {
            ExpertInput::Direct => self.protection = None,
            ExpertInput::AmaxF16 if self.protection.is_none() => {
                self.protection = Some(Protection::new(gpu, self.shape.slots())?);
            }
            ExpertInput::AmaxF16 => {}
        }
        Ok(())
    }

    /// The shape the buffers were made for.
    #[must_use]
    pub fn shape(&self) -> LegShape {
        self.shape
    }

    /// Device bytes of the buffers, as allocated: [`kernels::leg_bytes`] of
    /// their shape and forms.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.act_x.device_bytes()
            + self.h.num_bytes()
            + self.act_h.bytes()
            + self.down.num_bytes()
            + self.protection.as_ref().map_or(0, Protection::bytes)
    }

    /// The down rows of the last run: `n` values a slot, slot-major, the
    /// slots the run placed on the card (the rest are as an earlier run left
    /// them).
    #[must_use]
    pub fn down(&self) -> &DeviceBuffer<f32> {
        &self.down
    }

    /// The down rows for a route GEMM over the same slots to write its
    /// slots' rows into ([`RouteBlock::enqueue_rule`]), slot-major as a run
    /// leaves them, so the card sum reads both alike.
    pub(crate) fn down_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.down
    }
}

/// Device bytes a [`LegBufs::new`] of `shape` allocates, counted before it
/// does: [`leg_bytes_for`] of the [`DownAct::Q8Act`] form.
#[must_use]
pub fn leg_bytes(shape: LegShape) -> usize {
    leg_bytes_for(shape, DownAct::Q8Act)
}

/// Device bytes a [`LegBufs::for_down`] of `shape` and `act` allocates,
/// counted before it does: [`kernels::leg_bytes`] of the one form, the one
/// formula of them (the normed rows' q8_1 form, `cap` columns of `n`; the
/// gate·up's rows; the card slots' q8_1 columns, `cap · slots_per_col` of
/// `ff`; the down's rows).
#[must_use]
pub fn leg_bytes_for(shape: LegShape, act: DownAct) -> usize {
    let b = kernels::leg_bytes(
        shape.cap as u64,
        shape.slots_per_col as u64,
        shape.n as u64,
        shape.ff as u64,
        DownForms::of(act),
    );
    usize::try_from(b).expect("the leg's bytes fit a usize")
}

/// A resident routed stack's device form: the file's blocks as words
/// (`DevWeight::KQuant`), NVFP4's planes and its experts' scales
/// (`DevWeight::Nvfp4Stack`), or a Q5_0 stack in the `pack_q5_0` layout
/// (`DevWeight::Q5_0`), which `q5_0_gemv_sel` reads.
#[derive(Clone, Copy)]
pub(crate) enum StackW<'a> {
    Words(&'a DeviceTensor<u32>),
    Nvfp4(Nvfp4Stack<'a>),
    Q5_0(&'a DeviceTensor<u32>),
}

impl<'a> StackW<'a> {
    /// The resident routed stack `name` of `w` and its file type, for a walk
    /// whose route GEMM reads it ([`StackW::route`]): the file's blocks as
    /// words, Q5_0 packed, or NVFP4's planes (type [`GgmlType::NVFP4`]). Any
    /// other weight is refused by name as `what`'s.
    pub(crate) fn resident(
        what: &'static str,
        w: &'a Weights,
        name: &str,
    ) -> Result<(GgmlType, StackW<'a>), GpuError> {
        match w.get(name) {
            Some(DevWeight::KQuant { ty, w, .. }) => Ok((*ty, StackW::Words(w))),
            Some(DevWeight::Q5_0 { w, .. }) => Ok((GgmlType::Q5_0, StackW::Q5_0(w))),
            Some(DevWeight::Nvfp4Stack { .. }) => {
                Ok((GgmlType::NVFP4, StackW::Nvfp4(w.nvfp4_stack(what, name)?)))
            }
            _ => Err(GpuError::tensor(
                what,
                name,
                "a resident routed stack (the file's blocks as words, Q5_0 packed, or NVFP4 planes)",
            )),
        }
    }

    /// Rows the stack holds, every expert's.
    pub(crate) fn rows(&self) -> usize {
        match self {
            StackW::Words(w) | StackW::Q5_0(w) => w.rows(),
            StackW::Nvfp4(s) => s.qs.rows(),
        }
    }

    /// The stack, of file type `ty`, as the route GEMM the kernel table names
    /// for that type reads it ([`kernels::gemm`]): a K-quant's or an
    /// i-quant's words through the grouped GEMM, the file's Q8_0, Q5_1 or
    /// IQ4_NL blocks, a Q5_0 stack's packed words and NVFP4's planes through
    /// the 32-value GEMM. `None` for a type the table names no GEMM for, or a
    /// form that GEMM does not read.
    pub(crate) fn route(self, ty: GgmlType) -> Option<RouteW<'a>> {
        Some(match (kernels::gemm(ty)?, self) {
            (GemmEntry::Q8_0, StackW::Words(w)) => RouteW::Blocks32(Gemm32Weight::Q8_0File(w)),
            (GemmEntry::Q5_0, StackW::Q5_0(w)) => RouteW::Blocks32(Gemm32Weight::Q5_0Packed(w)),
            (GemmEntry::Q5_1, StackW::Words(w)) => RouteW::Blocks32(Gemm32Weight::Q5_1File(w)),
            (GemmEntry::Iq4nl, StackW::Words(w)) => RouteW::Blocks32(Gemm32Weight::Iq4NlFile(w)),
            (GemmEntry::Nvfp4, StackW::Nvfp4(s)) => RouteW::Blocks32(Gemm32Weight::Nvfp4Planes(s)),
            (entry, StackW::Words(w)) => RouteW::Grouped(GemmWeight::of_entry(entry)?, w),
            (_, StackW::Nvfp4(_) | StackW::Q5_0(_)) => return None,
        })
    }
}

/// A resident routed stack as a prompt walk's route GEMM reads it
/// ([`StackW::route`]).
#[derive(Clone, Copy)]
pub(crate) enum RouteW<'a> {
    /// The grouped GEMM's type and the stack's words, over the unit's q8_1
    /// rows of 128 values (`GemmAct`).
    Grouped(GemmWeight, &'a DeviceTensor<u32>),
    /// The 32-value GEMM's weight, over the unit's q8 rows of 32 values
    /// (`GemmAct32`).
    Blocks32(Gemm32Weight<'a>),
}

/// A routed block over a route table, the wide arm of a prompt walk that runs
/// a layer's card experts: the gate and up GEMMs, the SwiGLU into the form the
/// down GEMM reads, and the down GEMM, each as the kernel table's GEMM entry
/// of its stack's type reads it ([`StackW::route`]). The gate and up are one
/// type, so one input form; the down's form is its own.
pub(crate) struct RouteBlock<'a> {
    pub(crate) gate: RouteW<'a>,
    pub(crate) up: RouteW<'a>,
    pub(crate) down: RouteW<'a>,
    /// Rows an expert of the gate and up: the SwiGLU's values a column.
    pub(crate) ff: usize,
    /// Rows an expert of the down.
    pub(crate) hidden: usize,
}

/// The rows a [`RouteBlock`] reads and writes: the unit's normed rows in the
/// form the gate and up read (the other absent), the gate and up outputs, the
/// SwiGLU rows in the form the down reads (the other absent), and the down's
/// output.
pub(crate) struct RouteIo<'a> {
    pub(crate) x128: Option<&'a GemmAct>,
    pub(crate) x32: Option<&'a GemmAct32>,
    pub(crate) g: &'a mut DeviceBuffer<f32>,
    pub(crate) u: &'a mut DeviceBuffer<f32>,
    pub(crate) h128: Option<&'a mut GemmAct>,
    pub(crate) h32: Option<&'a mut GemmAct32>,
    pub(crate) y: &'a mut DeviceBuffer<f32>,
}

impl RouteBlock<'_> {
    /// Enqueue the block over `n_slots` slot columns of `route`'s table, the
    /// gate and up reading each slot's token column (`top_k` slots a token) and
    /// the down its own slot's: six launches. A form a stack reads that `io`
    /// lacks, a gate and up the table reads by two GEMMs, and a block whose
    /// gate and up are of two types are refused by name before the first
    /// launch. Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the two GEMM kernel sets, the table and its column rule, the slots, the rows and the fault sink (rust-quality R8)"
    )]
    pub(crate) fn enqueue(
        &self,
        gpu: &Gpu,
        (gemm, g32): (&GemmKernels, &Gemm32Kernels),
        (route, top_k): (&GemmRoute, usize),
        n_slots: usize,
        io: RouteIo<'_>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        self.enqueue_rule(
            gpu,
            (Some(gemm), g32),
            (route, top_k),
            n_slots,
            (io, Act::SiluMul),
            fault,
        )
    }

    /// [`RouteBlock::enqueue`] under the gate·up rule `rule` (`kquant::act`:
    /// `silu(g) · u`, ik's clamped SwiGLU or the OAI SwiGLU) between the
    /// gate and up GEMMs and the down's form, the same six launches
    /// ([`Act::SiluMul`] launches `enqueue`'s SwiGLU entries). A caller with
    /// no grouped GEMM kernels passes `None`: a block the grouped GEMM reads
    /// is then refused by name before the first launch.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the two GEMM kernel sets, the table and its column rule, the slots, the rows and their rule, and the fault sink (rust-quality R8)"
    )]
    pub(crate) fn enqueue_rule(
        &self,
        gpu: &Gpu,
        (gemm, g32): (Option<&GemmKernels>, &Gemm32Kernels),
        (route, top_k): (&GemmRoute, usize),
        n_slots: usize,
        (io, rule): (RouteIo<'_>, Act),
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        self.enqueue_input(
            gpu,
            (gemm, g32),
            (route, top_k),
            n_slots,
            (io, rule),
            None,
            fault,
        )
    }

    /// The routed block under its loaded expert-input policy.
    #[allow(clippy::too_many_arguments, reason = "route execution contract")]
    pub(crate) fn enqueue_input(
        &self,
        gpu: &Gpu,
        (gemm, g32): (Option<&GemmKernels>, &Gemm32Kernels),
        (route, top_k): (&GemmRoute, usize),
        n_slots: usize,
        (io, rule): (RouteIo<'_>, Act),
        mut protection: Option<&mut Protection>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        const WHAT_B: &str = "card_leg::RouteBlock";
        let stream = gpu.stream();
        let RouteIo {
            x128,
            x32,
            g,
            u,
            h128,
            h32,
            y,
        } = io;
        let lacks = |form: &'static str| {
            GpuError::shape(
                WHAT_B,
                format!("a block that reads {form}, which the rows lack"),
            )
        };
        let grouped_k = || {
            gemm.ok_or_else(|| {
                GpuError::shape(
                    WHAT_B,
                    "a block the grouped GEMM reads, beside no grouped GEMM kernels",
                )
            })
        };
        if [self.gate, self.up, self.down]
            .iter()
            .any(|w| matches!(w, RouteW::Grouped(..)))
        {
            grouped_k()?;
        }
        let input = GemmInput::Shared { top_k };
        match (self.gate, self.up) {
            (RouteW::Grouped(ty, wg), RouteW::Grouped(tu, wu)) if ty == tu => {
                let act = x128.ok_or_else(|| lacks("q8_1 blocks of 128 (the gate and up)"))?;
                let gemm = grouped_k()?;
                for (w, y) in [(wg, &mut *g), (wu, &mut *u)] {
                    gemm.enqueue_gemm(
                        stream,
                        GemmArgs {
                            ty,
                            w,
                            rows_per_expert: self.ff,
                            act,
                            route,
                            input,
                            y,
                        },
                    )?;
                }
            }
            (RouteW::Blocks32(wg), RouteW::Blocks32(wu)) => {
                let act = x32.ok_or_else(|| lacks("q8 blocks of 32 (the gate and up)"))?;
                for (w, y) in [(wg, &mut *g), (wu, &mut *u)] {
                    g32.enqueue_gemm32(
                        stream,
                        Gemm32Args {
                            w,
                            rows_per_expert: self.ff,
                            act,
                            route,
                            input,
                            y,
                        },
                    )?;
                }
            }
            _ => {
                return Err(GpuError::shape(
                    WHAT_B,
                    "a gate and an up the table does not read by one GEMM of one type",
                ));
            }
        }
        let (ids, counts) = route.expert_slots();
        let selected = Slots {
            ids,
            counts,
            routed: true,
            n_card: 0,
            len: n_slots,
        };
        if let Some(p) = protection.as_deref_mut() {
            p.kernels.normalize(
                gpu,
                (g, u, &mut p.scales),
                &selected,
                self.ff,
                Some(rule),
                fault,
            )?;
        }
        let per_slot = GemmInput::PerSlot;
        match self.down {
            RouteW::Grouped(ty, w) => {
                let act = h128.ok_or_else(|| lacks("q8_1 blocks of 128 (the down)"))?;
                let gemm = grouped_k()?;
                if protection.is_some() {
                    gpu.enqueue_quantize_gemm(g, n_slots, act, fault)?;
                } else {
                    gemm.enqueue_swiglu_act_quant(stream, g, u, rule, n_slots, act, fault)?;
                }
                gemm.enqueue_gemm(
                    stream,
                    GemmArgs {
                        ty,
                        w,
                        rows_per_expert: self.hidden,
                        act,
                        route,
                        input: per_slot,
                        y,
                    },
                )?;
            }
            RouteW::Blocks32(w) => {
                let act = h32.ok_or_else(|| lacks("q8 blocks of 32 (the down)"))?;
                if protection.is_some() {
                    g32.enqueue_quantize_gemm32(stream, g, n_slots, act, fault)?;
                } else {
                    match rule {
                        Act::SiluMul => {
                            g32.enqueue_swiglu_quant32(stream, g, u, n_slots, act, fault)?
                        }
                        _ => {
                            g32.enqueue_swiglu_act_quant32(stream, g, u, rule, n_slots, act, fault)?
                        }
                    }
                }
                g32.enqueue_gemm32(
                    stream,
                    Gemm32Args {
                        w,
                        rows_per_expert: self.hidden,
                        act,
                        route,
                        input: per_slot,
                        y,
                    },
                )?;
            }
        }
        if let Some(p) = protection {
            p.kernels
                .restore(gpu, y, &p.scales, &selected, self.hidden)?;
        }
        Ok(())
    }
}

/// One resident routed stack: its name, file type, device form and row width.
#[derive(Clone, Copy)]
struct LegStack<'a> {
    name: &'a str,
    ty: GgmlType,
    w: StackW<'a>,
    k: usize,
}

impl<'a> LegStack<'a> {
    /// The resident stack `name` of a type `runs` admits in its role,
    /// including plane-backed NVFP4 when admitted; `role` names a refusal.
    fn of(
        w: &'a Weights,
        name: &'a str,
        runs: fn(GgmlType) -> bool,
        role: &'static str,
    ) -> Result<LegStack<'a>, GpuError> {
        match w.get(name) {
            Some(DevWeight::KQuant { ty, w, k }) if runs(*ty) => Ok(LegStack {
                name,
                ty: *ty,
                w: StackW::Words(w),
                k: *k,
            }),
            Some(DevWeight::Q5_0 { w, k }) if runs(GgmlType::Q5_0) => Ok(LegStack {
                name,
                ty: GgmlType::Q5_0,
                w: StackW::Q5_0(w),
                k: *k,
            }),
            Some(DevWeight::Nvfp4Stack { k, .. }) if runs(GgmlType::NVFP4) => Ok(LegStack {
                name,
                ty: GgmlType::NVFP4,
                w: StackW::Nvfp4(w.nvfp4_stack(WHAT, name)?),
                k: *k,
            }),
            _ => Err(GpuError::Tensor {
                what: WHAT,
                name: name.to_owned(),
                need: role,
            }),
        }
    }

    /// Rows the stack holds, every expert's.
    fn rows(&self) -> usize {
        self.w.rows()
    }

    /// The stack in the form `w` of its type and row width: a ring window's
    /// ([`with_ring`]).
    fn with_w<'b>(&self, w: StackW<'b>) -> LegStack<'b>
    where
        'a: 'b,
    {
        LegStack {
            name: self.name,
            ty: self.ty,
            w,
            k: self.k,
        }
    }

    /// The stack's words, for an entry over the file's blocks.
    fn words(&self) -> Result<&'a DeviceTensor<u32>, GpuError> {
        match self.w {
            StackW::Words(w) => Ok(w),
            StackW::Nvfp4(_) | StackW::Q5_0(_) => Err(self.unrun()),
        }
    }

    /// The stack's `pack_q5_0` words, for the Q5_0 entry.
    fn q5_0(&self) -> Result<&'a DeviceTensor<u32>, GpuError> {
        match self.w {
            StackW::Q5_0(w) => Ok(w),
            StackW::Words(_) | StackW::Nvfp4(_) => Err(self.unrun()),
        }
    }

    /// The stack's planes and scales, for an NVFP4 entry.
    fn planes(&self) -> Result<Nvfp4Stack<'a>, GpuError> {
        match self.w {
            StackW::Nvfp4(s) => Ok(s),
            StackW::Words(_) | StackW::Q5_0(_) => Err(self.unrun()),
        }
    }

    /// The stack's type has an entry in the table that this leg has no arm
    /// for.
    fn unrun(&self) -> GpuError {
        GpuError::Tensor {
            what: WHAT,
            name: self.name.to_owned(),
            need: "a type whose gate·up and down entries the card leg launches",
        }
    }
}

/// What a refusal of a gate or up stack the common card leg does not run says.
const GATE_UP: &str =
    "a resident routed gate or up stack of a type the card leg runs (`kernels::card_gate_up_runs`)";

/// The selected program admits the same resident stack roles on card and tier.
pub(crate) const STACK_ROLES: (&str, &str) = (
    "a resident routed gate or up stack the selected program's gate·up arm runs",
    "a resident routed down stack the selected program's down arm runs",
);

/// A layer's three resident routed stacks, the gate and up of one type.
#[derive(Clone, Copy)]
pub struct LegStacks<'a> {
    gate: LegStack<'a>,
    up: LegStack<'a>,
    down: LegStack<'a>,
}

impl<'a> LegStacks<'a> {
    /// The stacks named `[gate, up, down]` in `w`, each a resident
    /// `DevWeight::KQuant` or an NVFP4 stack in its planes: the gate and up of
    /// one type [`kernels::card_gate_up_runs`] or [`kernels::card_planes_runs`]
    /// admits, the down of a type [`kernels::card_down_runs`] or
    /// [`kernels::card_planes_runs`] admits (not necessarily the same one).
    /// Allocation-free on success.
    pub fn of(w: &'a Weights, names: [&'a str; 3]) -> Result<LegStacks<'a>, GpuError> {
        const DOWN: &str =
            "a resident routed down stack of a type the card leg runs (`kernels::card_down_runs`)";
        LegStacks::of_arms(
            w,
            names,
            LegArms {
                gate_up: accepts::planes_leg_gate_up,
                down: accepts::planes_leg_down,
            },
            (GATE_UP, DOWN),
        )
    }

    /// [`LegStacks::of`] for a body that loads every tensor in the format
    /// [`gguf::card::CardFormat::of`] names (`Weights::load`), whose downs may
    /// also be Q5_0 in the `pack_q5_0` layout ([`accepts::resident_down_runs`]).
    /// Allocation-free on success.
    pub fn of_resident(w: &'a Weights, names: [&'a str; 3]) -> Result<LegStacks<'a>, GpuError> {
        const DOWN: &str = "a resident routed down stack of a type a whole-file body's card leg \
                            runs (`kernels::card_down_runs` or Q5_0)";
        LegStacks::of_arms(
            w,
            names,
            LegArms {
                gate_up: accepts::planes_leg_gate_up,
                down: |ty| accepts::resident_down_runs(ty) || kernels::card_planes_runs(ty),
            },
            (GATE_UP, DOWN),
        )
    }

    /// [`LegStacks::of`] for a program whose card leg runs the types `arms`
    /// name ([`accepts::leg_arms`]): the one constructor behind the others,
    /// each a program's arms over the launches [`CardLeg`] has arms for. A
    /// stack of a type no arm of `arms` admits is refused by name (`roles`:
    /// the gate or up refusal's words, then the down's). Allocation-free on
    /// success.
    pub fn of_arms(
        w: &'a Weights,
        names: [&'a str; 3],
        arms: LegArms,
        roles: (&'static str, &'static str),
    ) -> Result<LegStacks<'a>, GpuError> {
        let [gate, up, down] = names;
        let (gate, up, down) = (
            LegStack::of(w, gate, arms.gate_up, roles.0)?,
            LegStack::of(w, up, arms.gate_up, roles.0)?,
            LegStack::of(w, down, arms.down, roles.1)?,
        );
        if up.ty != gate.ty {
            return Err(GpuError::Tensor {
                what: WHAT,
                name: up.name.to_owned(),
                need: "the gate stack's type",
            });
        }
        Ok(LegStacks { gate, up, down })
    }

    /// The gate's and the up's type.
    #[must_use]
    pub fn gate_up_ty(&self) -> GgmlType {
        self.gate.ty
    }

    /// The down's type.
    #[must_use]
    pub fn down_ty(&self) -> GgmlType {
        self.down.ty
    }

    /// The gate's, up's and down's words, for kernels over the file's blocks
    /// (the grouped tiles); a stack in another layout is refused by name.
    pub fn words(&self) -> Result<[&'a DeviceTensor<u32>; 3], GpuError> {
        Ok([self.gate.words()?, self.up.words()?, self.down.words()?])
    }

    /// The three stacks as a prompt walk's route GEMMs read them
    /// ([`StackW::route`]): each stack's own type through the kernel table's
    /// GEMM entry, as one [`RouteBlock`] of `ff` rows a gate and up expert
    /// and `hidden` a down expert. A stack no route GEMM reads is refused by
    /// name.
    pub(crate) fn block(&self, ff: usize, hidden: usize) -> Result<RouteBlock<'a>, GpuError> {
        let one = |st: &LegStack<'a>| {
            st.w.route(st.ty).ok_or_else(|| GpuError::Tensor {
                what: WHAT,
                name: st.name.to_owned(),
                need: "a type some route GEMM reads (`kernels::gemm`)",
            })
        };
        Ok(RouteBlock {
            gate: one(&self.gate)?,
            up: one(&self.up)?,
            down: one(&self.down)?,
            ff,
            hidden,
        })
    }

    /// The stacks hold `n_card` experts of `shape` each: the gate's and up's
    /// `ff` rows of `n` values, the down's `n` rows of `ff`.
    pub fn check(&self, n_card: usize, shape: LegShape) -> Result<(), GpuError> {
        let LegShape { n, ff, .. } = shape;
        for (st, rows, k) in [(&self.gate, ff, n), (&self.up, ff, n), (&self.down, n, ff)] {
            if st.rows() != n_card * rows || st.k != k {
                return Err(GpuError::Tensor {
                    what: WHAT,
                    name: st.name.to_owned(),
                    need: "the rows of the layer's card experts, of the shape's width",
                });
            }
        }
        Ok(())
    }
}

/// `f` over a layer's three stacks on a half of the expert stream's ring `r`
/// as the route GEMMs read them ([`LegStacks::block`], `ff` rows a gate and
/// up expert, `hidden` a down expert), in the forms of the layer's card
/// stacks `card` (`n_card` experts each, which give each expert's rows and
/// each plane's row width): the half holds `r.half_slots` experts from each
/// part's first slot ([`RingLayer::parts`]), a stack of words one part a
/// matrix, an NVFP4 stack three (its codes, its block scales and its tensor
/// scale). The one ring pass's view of the half for every family, whatever
/// the card stacks' form; the windows are given back after `f`. Refused by
/// name: a half whose parts are not the card stacks' form, a Q5_0 stack (no
/// ring holds one), card stacks whose rows are not whole experts, and a stack
/// no route GEMM reads.
pub(crate) fn with_ring<T>(
    card: &LegStacks<'_>,
    n_card: usize,
    r: &RingLayer,
    ctx: &Arc<CudaContext>,
    (ff, hidden): (usize, usize),
    f: impl FnOnce(&RouteBlock<'_>) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    let whole = |rows: usize| n_card > 0 && rows.is_multiple_of(n_card);
    let stacks = [&card.gate, &card.up, &card.down];
    if let Some(st) = stacks.iter().find(|st| !whole(st.rows())) {
        return Err(GpuError::Tensor {
            what: WHAT,
            name: st.name.to_owned(),
            need: "whole experts of the layer's card count, beside the expert stream's ring",
        });
    }
    let half = r.half_slots;
    let rows = |t: &DeviceTensor<u32>| half * (t.rows() / n_card);
    match (r.planes, card.gate.w, card.up.w, card.down.w) {
        (1, StackW::Words(g), StackW::Words(u), StackW::Words(d)) => {
            // SAFETY: the half holds `half_slots` experts in each part from
            // the part's first slot at the layer's part bytes a slot, an
            // expert's rows of the card stack's row width
            // (`XStream::ring_layer`, the source's part bytes); the ring stays
            // allocated while the stream lives, past `f`'s launches, and the
            // windows are given back below.
            let ws = unsafe {
                [
                    DeviceTensor::<u32>::window(r.parts[0], rows(g), g.cols(), ctx),
                    DeviceTensor::<u32>::window(r.parts[1], rows(u), u.cols(), ctx),
                    DeviceTensor::<u32>::window(r.parts[2], rows(d), d.cols(), ctx),
                ]
            };
            let half = LegStacks {
                gate: card.gate.with_w(StackW::Words(&ws[0])),
                up: card.up.with_w(StackW::Words(&ws[1])),
                down: card.down.with_w(StackW::Words(&ws[2])),
            };
            let ran = half.block(ff, hidden).and_then(|blk| f(&blk));
            for w in ws {
                DeviceTensor::release(w);
            }
            ran
        }
        (3, StackW::Nvfp4(g), StackW::Nvfp4(u), StackW::Nvfp4(d)) => {
            // SAFETY: as the words' arm, each matrix in three parts: its codes
            // and its block scales of the card planes' row widths, and one
            // f32 tensor scale a slot.
            let (qs, scales) = unsafe {
                (
                    [
                        DeviceTensor::<u32>::window(r.parts[0], rows(g.qs), g.qs.cols(), ctx),
                        DeviceTensor::<u32>::window(r.parts[1], rows(g.e), g.e.cols(), ctx),
                        DeviceTensor::<u32>::window(r.parts[3], rows(u.qs), u.qs.cols(), ctx),
                        DeviceTensor::<u32>::window(r.parts[4], rows(u.e), u.e.cols(), ctx),
                        DeviceTensor::<u32>::window(r.parts[6], rows(d.qs), d.qs.cols(), ctx),
                        DeviceTensor::<u32>::window(r.parts[7], rows(d.e), d.e.cols(), ctx),
                    ],
                    [
                        DeviceTensor::<f32>::window(r.parts[2], half, 1, ctx),
                        DeviceTensor::<f32>::window(r.parts[5], half, 1, ctx),
                        DeviceTensor::<f32>::window(r.parts[8], half, 1, ctx),
                    ],
                )
            };
            let planes = |i: usize| {
                StackW::Nvfp4(Nvfp4Stack {
                    qs: &qs[2 * i],
                    e: &qs[2 * i + 1],
                    scales: &scales[i],
                })
            };
            let half = LegStacks {
                gate: card.gate.with_w(planes(0)),
                up: card.up.with_w(planes(1)),
                down: card.down.with_w(planes(2)),
            };
            let ran = half.block(ff, hidden).and_then(|blk| f(&blk));
            for w in qs {
                DeviceTensor::release(w);
            }
            for s in scales {
                DeviceTensor::release(s);
            }
            ran
        }
        _ => Err(GpuError::shape(
            WHAT,
            format!(
                "the expert stream's ring of {} parts a matrix beside card stacks of {} and {}: \
                 a ring holds a stack of words in one part a matrix and an NVFP4 stack in three",
                r.planes, card.gate.ty, card.down.ty
            ),
        )),
    }
}

/// What [`CardLeg::enqueue`] reads: layer `layer`'s `stacks`, `m` tokens'
/// normed rows `normed`, the slots' places `sel` (`m · slots_per_col` ids; an
/// id below `n_card` is a card expert, the rest are left alone) and the
/// activation rule.
pub struct LegRun<'a> {
    pub layer: usize,
    pub stacks: &'a LegStacks<'a>,
    pub normed: &'a DeviceBuffer<f32>,
    pub m: usize,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_card: usize,
    pub rule: Act,
}

impl LegRun<'_> {
    /// The run's fields but its normed rows.
    fn common(&self) -> Common<'_> {
        Common {
            layer: self.layer,
            stacks: self.stacks,
            m: self.m,
            sel: self.sel,
            n_card: self.n_card,
            rule: self.rule,
        }
    }
}

/// What [`CardLeg::enqueue_rows_quantized`] reads: [`LegRun`]'s, the `m`
/// tokens' normed rows being the first `m` columns of `act`, already in q8_1.
pub struct QuantRun<'a> {
    pub layer: usize,
    pub stacks: &'a LegStacks<'a>,
    pub act: &'a Q8Act,
    pub m: usize,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_card: usize,
    pub rule: Act,
}

impl QuantRun<'_> {
    fn common(&self) -> Common<'_> {
        Common {
            layer: self.layer,
            stacks: self.stacks,
            m: self.m,
            sel: self.sel,
            n_card: self.n_card,
            rule: self.rule,
        }
    }
}

/// A run's fields besides where its normed rows come from.
struct Common<'a> {
    layer: usize,
    stacks: &'a LegStacks<'a>,
    m: usize,
    sel: &'a DeviceBuffer<u32>,
    n_card: usize,
    rule: Act,
}

/// Where a run's q8_1 normed rows come from: quantized from f32 rows in the
/// run's first launch, or already in q8_1.
enum Source<'a> {
    Normed(&'a DeviceBuffer<f32>),
    Quantized(&'a Q8Act),
}

/// The card slots' sum a run ends with ([`CardLeg::enqueue`]): `w` the
/// slots' routing weights, `slots_per_col` a token and token-major, and `acc`
/// the sum, `m` rows of `n` values. A token's slots are summed in slot order
/// over the slots whose place is below `n_card`; `slots_per_col` is a count
/// the sum has an entry for ([`CardSlots::of`]), any other is refused by name
/// (the card sum's one refusal of a count) before the run launches anything.
pub struct LegSum<'a> {
    pub w: &'a DeviceBuffer<f32>,
    pub acc: &'a mut DeviceBuffer<f32>,
}

/// The modules the leg's launches run on.
pub struct CardLeg {
    quant: QuantSelKernels,
    q6k: Q6kSelKernels,
    mxfp4: Mxfp4SelKernels,
    nvfp4: Nvfp4SelKernels,
    q51: Q51SelKernels,
    q80: Q80SelKernels,
    slots: CardSlotKernels,
}

/// The buffers of a [`LegBufs`] a run writes besides its down rows.
struct Scratch<'a> {
    shape: LegShape,
    act_x: &'a mut Q8Act,
    h: &'a mut DeviceBuffer<f32>,
    act_h: &'a mut ActForms,
    protection: &'a mut Option<Protection>,
}

impl CardLeg {
    /// The modules on `gpu`. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<CardLeg, GpuError> {
        let (ctx, word) = (gpu.context(), gpu.fault_word());
        Ok(CardLeg {
            quant: QuantSelKernels::load(ctx, word)?,
            q6k: Q6kSelKernels::load(ctx, word)?,
            mxfp4: Mxfp4SelKernels::load(ctx, word)?,
            nvfp4: Nvfp4SelKernels::load(ctx, word)?,
            q51: Q51SelKernels::load(ctx, word)?,
            q80: Q80SelKernels::load(ctx, word)?,
            slots: CardSlotKernels::load(ctx)?,
        })
    }

    /// The modules the 32-value-block `_sel`s run on, the q5 module `gpu`'s.
    fn sel32<'a>(&'a self, gpu: &'a Gpu) -> Sel32<'a> {
        Sel32 {
            q5: gpu.q5(),
            q51: &self.q51,
            q80: &self.q80,
            iq: &self.quant,
            nvfp4: &self.nvfp4,
        }
    }

    /// Enqueue on `gpu`'s stream the four launches (module doc) of `run` into
    /// `bufs`, and with `sum` the card slots' sum after them; `run.m` outside
    /// `1..=bufs.shape().cap`, a down entry that reads the other activation
    /// form than `bufs` holds, or a sum over a slot count the sum has no entry
    /// for, is refused by name before the first launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        gpu: &Gpu,
        bufs: &mut LegBufs,
        run: &LegRun<'_>,
        sum: Option<LegSum<'_>>,
    ) -> Result<(), GpuError> {
        let LegBufs {
            shape,
            act_x,
            h,
            act_h,
            down,
            protection,
        } = bufs;
        let shape = *shape;
        if sum.is_some() {
            check_sum(shape)?;
        }
        let scratch = Scratch {
            shape,
            act_x,
            h,
            act_h,
            protection,
        };
        self.launch(
            gpu,
            scratch,
            Source::Normed(run.normed),
            &run.common(),
            down,
        )?;
        let Some(sum) = sum else {
            return Ok(());
        };
        enqueue_sum(
            &self.slots,
            gpu,
            shape,
            (down, run.sel),
            (run.layer, run.m, run.n_card),
            sum,
        )
    }

    /// [`CardLeg::enqueue`]'s four launches, the down rows written to `down`
    /// (`shape.n` values a slot, slot-major, at least `m · slots_per_col`
    /// slots) instead of `bufs`'s own: for a caller whose rows live in a
    /// buffer of its own, and which sums them itself. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_rows(
        &self,
        gpu: &Gpu,
        bufs: &mut LegBufs,
        run: &LegRun<'_>,
        down: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let LegBufs {
            shape,
            act_x,
            h,
            act_h,
            protection,
            ..
        } = bufs;
        let scratch = Scratch {
            shape: *shape,
            act_x,
            h,
            act_h,
            protection,
        };
        self.launch(
            gpu,
            scratch,
            Source::Normed(run.normed),
            &run.common(),
            down,
        )
    }

    /// [`CardLeg::enqueue_rows`] over normed rows already in q8_1: the
    /// launches of the module comment but the first (the quantize of the
    /// normed rows), reading `run.act` in its place. For a caller whose
    /// norm writes the q8_1 form itself (the fused norm's, or the staged
    /// codes of an expert tier's block), the bits those columns hold being
    /// what the quantize launch writes. `bufs` holds the gate·up's rows and
    /// the card slots' columns; its own normed-rows planes are not used.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_rows_quantized(
        &self,
        gpu: &Gpu,
        bufs: &mut LegBufs,
        run: &QuantRun<'_>,
        down: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let LegBufs {
            shape,
            act_x,
            h,
            act_h,
            protection,
            ..
        } = bufs;
        let scratch = Scratch {
            shape: *shape,
            act_x,
            h,
            act_h,
            protection,
        };
        self.launch(
            gpu,
            scratch,
            Source::Quantized(run.act),
            &run.common(),
            down,
        )
    }

    /// The card slots' sum ([`LegSum`]) of the down rows the runs over `bufs`
    /// left, each run without a sum of its own: for a caller whose slots ran
    /// in more than one run over different stacks. `sel` places every slot
    /// those runs wrote below `n_card`; a token's slots are summed in slot
    /// order. `m` outside `1..=bufs.shape().cap` and a slot count the sum has
    /// no entry for are refused by name. Asynchronous, allocation-free,
    /// capturable.
    pub fn sum_rows(
        &self,
        gpu: &Gpu,
        bufs: &LegBufs,
        sel: &DeviceBuffer<u32>,
        (layer, m, n_card): (usize, usize, usize),
        sum: LegSum<'_>,
    ) -> Result<(), GpuError> {
        let shape = bufs.shape;
        if m == 0 || m > shape.cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a sum of {m} tokens in buffers of {}", shape.cap),
            });
        }
        check_sum(shape)?;
        enqueue_sum(
            &self.slots,
            gpu,
            shape,
            (&bufs.down, sel),
            (layer, m, n_card),
            sum,
        )
    }

    /// The four launches of `run` into `scratch` and `down`, the first one
    /// but for a `src` already in q8_1.
    fn launch(
        &self,
        gpu: &Gpu,
        scratch: Scratch<'_>,
        src: Source<'_>,
        run: &Common<'_>,
        down: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Scratch {
            shape:
                LegShape {
                    cap,
                    slots_per_col,
                    n,
                    ff,
                },
            act_x,
            h,
            act_h,
            protection,
        } = scratch;
        if run.m == 0 || run.m > cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a run of {} tokens in buffers of {cap}", run.m),
            });
        }
        let st = run.stacks;
        let Some((_, form)) = kernels::down(st.down.ty) else {
            return Err(st.down.unrun());
        };
        let mut act_h = act_h.of(form).ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!(
                "the down stack {} of {} reads {form:?} columns, which the buffers do not hold",
                st.down.name, st.down.ty
            ),
        })?;
        let stream = gpu.stream();
        let fault = gpu.layer_sink(run.layer)?;
        let slots = run.m * slots_per_col;
        let act_x: &Q8Act = match src {
            Source::Normed(normed) => {
                gpu.enqueue_quantize_q8_1_cols(normed, act_x, run.m, run.layer)?;
                act_x
            }
            Source::Quantized(act) => act,
        };
        let words = || -> Result<GateUpAct<'_>, GpuError> {
            Ok(GateUpAct {
                wg: st.gate.words()?,
                wu: st.up.words()?,
                act: act_x,
                sel: run.sel,
                n_slots: slots,
                rows_per_expert: ff,
                slots_per_col,
                rule: run.rule,
            })
        };
        match kernels::gate_up(st.gate.ty) {
            Some(GateUpEntry::Q4k) => {
                self.quant
                    .enqueue_gate_up_q4k(stream, &words()?, fault, h)?
            }
            Some(GateUpEntry::Q5k) => {
                self.quant
                    .enqueue_gate_up_q5k(stream, &words()?, fault, h)?
            }
            Some(GateUpEntry::Q8_0) => {
                self.quant
                    .enqueue_gate_up_q8_0(stream, &words()?, fault, h)?
            }
            Some(GateUpEntry::Q6k) => self.q6k.enqueue_gate_up(stream, &words()?, fault, h)?,
            Some(GateUpEntry::Mxfp4) => self.mxfp4.enqueue_gate_up(stream, &words()?, fault, h)?,
            Some(GateUpEntry::Nvfp4) => {
                let a = Nvfp4GateUp {
                    gate: st.gate.planes()?,
                    up: st.up.planes()?,
                    act: act_x,
                    sel: run.sel,
                    n_slots: slots,
                    rows_per_expert: ff,
                    slots_per_col,
                    rule: run.rule,
                };
                self.nvfp4.enqueue_gate_up(stream, &a, fault, h)?;
            }
            Some(
                GateUpEntry::Iq2xxs
                | GateUpEntry::Iq2xs
                | GateUpEntry::Iq2s
                | GateUpEntry::Iq1s
                | GateUpEntry::Iq1m
                | GateUpEntry::Iq3xxs
                | GateUpEntry::Iq3s
                | GateUpEntry::Iq4xs,
            ) => {
                let a = words()?;
                let iq = IqGateUp {
                    ty: st.gate.ty,
                    wg: a.wg,
                    wu: a.wu,
                    act: a.act,
                    sel: a.sel,
                    n_slots: a.n_slots,
                    rows_per_expert: a.rows_per_expert,
                    slots_per_col: a.slots_per_col,
                    rule: a.rule,
                };
                self.quant.enqueue_gate_up(stream, &iq, fault, h)?;
            }
            _ => return Err(st.gate.unrun()),
        }
        let selected = Slots {
            ids: run.sel,
            counts: run.sel,
            routed: false,
            n_card: run.n_card,
            len: slots,
        };
        if let Some(p) = protection {
            p.kernels.normalize(
                gpu,
                // Gate/up already activated h; None leaves the second operand unread.
                (h, &*down, &mut p.scales),
                &selected,
                ff,
                None,
                fault,
            )?;
        }
        let q = QuantSel {
            x: h,
            cols: 0..slots,
            sel: run.sel,
            n_card: run.n_card,
        };
        match &mut act_h {
            ActH::Q8Act(act) => gpu.q4k_sel().enqueue_quantize_sel(stream, &q, fault, act)?,
            ActH::Blocks32(act) => gpu.q5().enqueue_quantize_q8_sel(stream, &q, act, fault)?,
        }
        let (sel, rows_per_expert) = (run.sel, n);
        match (kernels::down(st.down.ty), &act_h) {
            (Some((DownEntry::Q4k, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                gpu.q4k_sel().enqueue_gemv_q4k_sel(
                    stream,
                    st.down.words()?,
                    act,
                    sel,
                    slots,
                    rows_per_expert,
                    down,
                )?;
            }
            (Some((DownEntry::Q5k, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                let d = SelDown {
                    w: st.down.words()?,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.quant.enqueue_gemv_q5k_sel(stream, &d, fault, down)?;
            }
            (Some((DownEntry::Q6k, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                self.q6k.enqueue_gemv_q6k_sel(
                    stream,
                    st.down.words()?,
                    act,
                    sel,
                    slots,
                    rows_per_expert,
                    down,
                )?;
            }
            (Some((DownEntry::Mxfp4, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                let d = SelDown {
                    w: st.down.words()?,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.mxfp4.enqueue_down(stream, &d, fault, down)?;
            }
            (Some((DownEntry::Nvfp4, DownAct::Q8Blocks32)), ActH::Blocks32(act)) => {
                let d = Nvfp4Down {
                    w: st.down.planes()?,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                    slots_per_col,
                };
                self.nvfp4.enqueue_sel32(stream, &d, fault, down)?;
            }
            (Some((entry, DownAct::Q8Blocks32)), ActH::Blocks32(act)) => {
                let run = Sel32Run {
                    entry,
                    stack: &st.down,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                    slots_per_col,
                };
                self.sel32(gpu).enqueue(stream, &run, fault, down)?;
            }
            (
                Some((
                    DownEntry::Iq2xs
                    | DownEntry::Iq2s
                    | DownEntry::Iq1s
                    | DownEntry::Iq1m
                    | DownEntry::Iq3xxs
                    | DownEntry::Iq3s
                    | DownEntry::Iq4xs,
                    DownAct::Q8Act,
                )),
                ActH::Q8Act(act),
            ) => {
                let d = SelDown {
                    w: st.down.words()?,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.quant
                    .enqueue_gemv_sel(stream, st.down.ty, &d, fault, down)?;
            }
            (Some((DownEntry::Iq2xxs, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                // The shared-table down reads a column a run of
                // `slots_per_col` slots; a down is one column a slot.
                let d = SelDown {
                    w: st.down.words()?,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.quant
                    .enqueue_gemv_sel_cols(stream, st.down.ty, &d, 1, fault, down)?;
            }
            _ => return Err(st.down.unrun()),
        }
        if let Some(p) = protection {
            p.kernels.restore(gpu, down, &p.scales, &selected, n)?;
        }
        Ok(())
    }
}

/// The sum a run ends with is over a slot count the sum has an entry for:
/// refused by name before the run launches anything.
fn check_sum(shape: LegShape) -> Result<(), GpuError> {
    match CardSlots::of(shape.slots_per_col) {
        Some(_) => Ok(()),
        None => Err(GpuError::Shape {
            what: WHAT,
            detail: format!("the card sum: {}", CardSlots::refusal(shape.slots_per_col)),
        }),
    }
}

/// The card slots' weighted sum of `down` (the run's rows, `m` tokens of
/// `shape.slots_per_col` slots) under the places `sel`, after the run's
/// launches: [`LegSum`] says what it reads and writes. `(layer, m, n_card)` are
/// the run's.
fn enqueue_sum(
    slots: &CardSlotKernels,
    gpu: &Gpu,
    shape: LegShape,
    (down, sel): (&DeviceBuffer<f32>, &DeviceBuffer<u32>),
    (layer, m, n_card): (usize, usize, usize),
    sum: LegSum<'_>,
) -> Result<(), GpuError> {
    let LegSum { w, acc } = sum;
    let a = CardAcc {
        down,
        w,
        sel,
        n: shape.n,
        m,
        n_card,
        n_used: shape.slots_per_col,
        w_pitch: shape.slots_per_col,
        fault: gpu.layer_sink(layer)?,
    };
    slots.enqueue_card_acc(gpu.stream(), &a, acc)
}

/// The modules a 32-value-block `_sel` runs on, borrowed from a leg.
struct Sel32<'a> {
    q5: &'a Q5Kernels,
    q51: &'a Q51SelKernels,
    q80: &'a Q80SelKernels,
    iq: &'a QuantSelKernels,
    nvfp4: &'a Nvfp4SelKernels,
}

/// One 32-value-block `_sel` launch: slot `s` writes the `rows_per_expert`
/// rows of expert `sel[s]` of `stack` dotted with column `s` of `act`, to
/// `y[s · rows_per_expert ..]`. `entry` is the table's row for the stack's
/// type, [`kernels::down`]'s for a down and [`kernels::up`]'s for an up (one
/// entry: an up is a down-shaped `_sel`). `slots_per_col` is the slots of a
/// token, which the NVFP4 entry reads to serve a run of tokens whose slots
/// can name one expert.
#[derive(Clone, Copy)]
struct Sel32Run<'a> {
    entry: DownEntry,
    stack: &'a LegStack<'a>,
    act: &'a Q8Blocks32,
    sel: &'a DeviceBuffer<u32>,
    n_slots: usize,
    rows_per_expert: usize,
    slots_per_col: usize,
}

impl Sel32<'_> {
    /// Enqueue the launch of `run`; an entry the table names that has no arm
    /// here is refused by name.
    fn enqueue(
        &self,
        stream: &CudaStream,
        run: &Sel32Run<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Sel32Run {
            entry,
            stack,
            act,
            sel,
            n_slots,
            rows_per_expert,
            slots_per_col,
        } = *run;
        match entry {
            DownEntry::Q5_0 => self.q5.enqueue_gemv_q5_0_sel_at(
                stream,
                (stack.q5_0()?, act),
                (sel, n_slots),
                rows_per_expert,
                fault,
                y,
            ),
            DownEntry::Q5_1 => {
                let d = Q51SelDown {
                    w: stack.words()?,
                    act,
                    sel,
                    n_slots,
                    rows_per_expert,
                };
                self.q51.enqueue_gemv_q5_1_sel(stream, &d, fault, y)
            }
            DownEntry::Q8_0 => {
                let d = Q80SelDown {
                    w: stack.words()?,
                    act,
                    sel,
                    n_slots,
                    rows_per_expert,
                };
                self.q80.enqueue_gemv_q8_0_sel32(stream, &d, fault, y)
            }
            DownEntry::Iq4nl => {
                let d = IqDown {
                    w: stack.words()?,
                    act,
                    sel,
                    n_slots,
                    rows_per_expert,
                };
                self.iq.enqueue_down(stream, &d, fault, y)
            }
            DownEntry::Nvfp4 => {
                let d = Nvfp4Down {
                    w: stack.planes()?,
                    act,
                    sel,
                    n_slots,
                    rows_per_expert,
                    slots_per_col,
                };
                self.nvfp4.enqueue_sel32(stream, &d, fault, y)
            }
            _ => Err(stack.unrun()),
        }
    }
}

/// A layer's two resident routed stacks of an expert with no gate: the up and
/// the down, each of a type [`kernels::card_relu2_runs`] admits (not
/// necessarily the same one).
pub struct Relu2Stacks<'a> {
    up: LegStack<'a>,
    down: LegStack<'a>,
}

impl<'a> Relu2Stacks<'a> {
    /// The stacks named `[up, down]` in `w`, each a resident
    /// `DevWeight::KQuant` of a 32-value-block type the relu² form runs, for
    /// Q5_0 a `DevWeight::Q5_0` in the packed layout its entry reads, or for
    /// NVFP4 a `DevWeight::Nvfp4Stack` in its planes.
    /// Allocation-free on success.
    pub fn of(w: &'a Weights, names: [&'a str; 2]) -> Result<Relu2Stacks<'a>, GpuError> {
        const UP: &str = "a resident routed up stack of a type the relu² card leg runs (`kernels::card_relu2_runs`)";
        const DOWN: &str = "a resident routed down stack of a type the relu² card leg runs (`kernels::card_relu2_runs`)";
        let [up, down] = names;
        Ok(Relu2Stacks {
            up: LegStack::of(w, up, kernels::card_relu2_runs, UP)?,
            down: LegStack::of(w, down, kernels::card_relu2_runs, DOWN)?,
        })
    }

    /// The stacks hold `n_card` experts of `shape` each: the up's `ff` rows of
    /// `n` values, the down's `n` rows of `ff`.
    pub fn check(&self, n_card: usize, shape: LegShape) -> Result<(), GpuError> {
        let LegShape { n, ff, .. } = shape;
        for (st, rows, k) in [(&self.up, ff, n), (&self.down, n, ff)] {
            if st.rows() != n_card * rows || st.k != k {
                return Err(GpuError::Tensor {
                    what: WHAT,
                    name: st.name.to_owned(),
                    need: "the rows of the layer's card experts, of the shape's width",
                });
            }
        }
        Ok(())
    }
}

/// One relu² run's buffers for up to `shape.cap` tokens: the normed row's
/// 32-value q8_1 form once a slot, the up's rows, the card slots' q8_1 columns
/// of `max(h, 0)²` and the down's rows. `shape.n` and `shape.ff` are
/// multiples of 32.
pub struct Relu2Bufs {
    shape: LegShape,
    act_x: Q8Blocks32,
    h: DeviceBuffer<f32>,
    act_h: Q8Blocks32,
    down: DeviceBuffer<f32>,
}

impl Relu2Bufs {
    /// The buffers for runs of `shape`. Load-time or first-prompt only.
    pub fn new(stream: &CudaStream, shape: LegShape) -> Result<Relu2Bufs, GpuError> {
        let slots = shape.slots();
        Ok(Relu2Bufs {
            shape,
            act_x: Q8Blocks32::with_slots(stream, shape.n, slots)?,
            h: DeviceBuffer::zeroed(stream, slots * shape.ff)?,
            act_h: Q8Blocks32::with_slots(stream, shape.ff, slots)?,
            down: DeviceBuffer::zeroed(stream, slots * shape.n)?,
        })
    }

    /// The shape the buffers were made for.
    #[must_use]
    pub fn shape(&self) -> LegShape {
        self.shape
    }

    /// Device bytes of the buffers: [`relu2_leg_bytes`] of their shape, as
    /// allocated.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let q8 = |a: &Q8Blocks32| a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes();
        q8(&self.act_x) + self.h.num_bytes() + q8(&self.act_h) + self.down.num_bytes()
    }

    /// The down rows of the last run: `n` values a slot, slot-major, the slots
    /// the run placed on the card (the rest are as an earlier run left them).
    #[must_use]
    pub fn down(&self) -> &DeviceBuffer<f32> {
        &self.down
    }
}

/// Device bytes a [`Relu2Bufs::new`] of `shape` allocates, counted before it
/// does: the normed row's q8_1 form and the card slots' (`cap · slots_per_col`
/// columns of `n` and of `ff`), the up's rows and the down's rows.
#[must_use]
pub fn relu2_leg_bytes(shape: LegShape) -> usize {
    let slots = shape.slots();
    let f = size_of::<f32>();
    let b32 = |k: usize| {
        usize::try_from(kernels::q8_blocks32_bytes(slots as u64, k as u64))
            .expect("the 32-value activation's bytes fit a usize")
    };
    b32(shape.n) + f * slots * shape.ff + b32(shape.ff) + f * slots * shape.n
}

/// What [`Relu2Leg::enqueue`] reads: layer `layer`'s `stacks`, `m` tokens'
/// normed rows `normed`, and the slots' places `sel` (`m · slots_per_col` ids;
/// an id below `n_card` is a card expert, the rest are left alone).
pub struct Relu2Run<'a> {
    pub layer: usize,
    pub stacks: &'a Relu2Stacks<'a>,
    pub normed: &'a DeviceBuffer<f32>,
    pub m: usize,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_card: usize,
}

/// The modules the relu² form's launches run on, beside the quantizers of
/// `Gpu::q5`.
pub struct Relu2Leg {
    q51: Q51SelKernels,
    q80: Q80SelKernels,
    iq: QuantSelKernels,
    nvfp4: Nvfp4SelKernels,
    slots: CardSlotKernels,
}

impl Relu2Leg {
    /// The modules on `gpu`. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<Relu2Leg, GpuError> {
        let (ctx, word) = (gpu.context(), gpu.fault_word());
        Ok(Relu2Leg {
            q51: Q51SelKernels::load(ctx, word)?,
            q80: Q80SelKernels::load(ctx, word)?,
            iq: QuantSelKernels::load(ctx, word)?,
            nvfp4: Nvfp4SelKernels::load(ctx, word)?,
            slots: CardSlotKernels::load(ctx)?,
        })
    }

    /// The modules the 32-value `_sel`s run on, the q5 module `gpu`'s.
    fn sel32<'a>(&'a self, gpu: &'a Gpu) -> Sel32<'a> {
        Sel32 {
            q5: gpu.q5(),
            q51: &self.q51,
            q80: &self.q80,
            iq: &self.iq,
            nvfp4: &self.nvfp4,
        }
    }

    /// Enqueue on `gpu`'s stream the four launches (module doc) of `run` into
    /// `bufs`, and with `sum` the card slots' sum after them; `run.m` outside
    /// `1..=bufs.shape().cap`, or a sum over a slot count the sum has no entry
    /// for, is refused by name before the first launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        gpu: &Gpu,
        bufs: &mut Relu2Bufs,
        run: &Relu2Run<'_>,
        sum: Option<LegSum<'_>>,
    ) -> Result<(), GpuError> {
        let Relu2Bufs {
            shape,
            act_x,
            h,
            act_h,
            down,
        } = bufs;
        let LegShape {
            cap,
            slots_per_col,
            n,
            ff,
        } = *shape;
        if run.m == 0 || run.m > cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a run of {} tokens in buffers of {cap}", run.m),
            });
        }
        if sum.is_some() {
            check_sum(*shape)?;
        }
        let st = run.stacks;
        let entry = |stack: &LegStack<'_>| kernels::up(stack.ty).ok_or_else(|| stack.unrun());
        let (up_entry, down_entry) = (entry(&st.up)?, entry(&st.down)?);
        let stream = gpu.stream();
        let fault = gpu.layer_sink(run.layer)?;
        let slots = run.m * slots_per_col;
        let sel32 = self.sel32(gpu);
        let q = QuantSel {
            x: run.normed,
            cols: 0..slots,
            sel: run.sel,
            n_card: run.n_card,
        };
        gpu.q5()
            .enqueue_quantize_q8_rep(stream, &q, slots_per_col, act_x, fault)?;
        let up = Sel32Run {
            entry: up_entry,
            stack: &st.up,
            act: act_x,
            sel: run.sel,
            n_slots: slots,
            rows_per_expert: ff,
            slots_per_col,
        };
        sel32.enqueue(stream, &up, fault, h)?;
        let q = QuantSel {
            x: h,
            cols: 0..slots,
            sel: run.sel,
            n_card: run.n_card,
        };
        gpu.q5()
            .enqueue_quantize_q8_sel_relu_sqr(stream, &q, act_h, fault)?;
        let dn = Sel32Run {
            entry: down_entry,
            stack: &st.down,
            act: act_h,
            sel: run.sel,
            n_slots: slots,
            rows_per_expert: n,
            slots_per_col,
        };
        sel32.enqueue(stream, &dn, fault, down)?;
        let Some(sum) = sum else {
            return Ok(());
        };
        enqueue_sum(
            &self.slots,
            gpu,
            *shape,
            (down, run.sel),
            (run.layer, run.m, run.n_card),
            sum,
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::hybrid::HOST;
    use crate::q4k_sel::QuantSel;
    use crate::q5::pack_q5_0;
    use gguf::Split;
    use gguf::write::{Layout, TensorDecl, Writer};
    use model::placement::CardFormat;
    use runtime::words::stream_words;

    /// The normed width: the gate·up stacks' K (eight super-blocks).
    const N: usize = 2048;
    /// The routed width: a gate·up slot's rows and the down stack's K (four
    /// super-blocks).
    const FF: usize = 1024;
    /// Experts the stacks hold.
    const N_CARD: usize = 4;
    /// Slots a token carries.
    const SPC: usize = 8;
    /// Tokens the leg's buffers are sized for.
    const CAP: usize = 3;
    /// What a slot the run leaves alone still holds.
    const SENT: f32 = 1.0e30;
    /// Buffers of both down forms.
    const BOTH: DownForms = DownForms::of(DownAct::Q8Act).with(DownAct::Q8Blocks32);
    /// Three tokens' places: duplicates, every card expert, and slots the
    /// card does not serve.
    const SEL: [u32; CAP * SPC] = [
        0, 1, 2, 3, 0, 1, HOST, 2, 3, 3, 1, 0, HOST, 2, 1, 0, 2, HOST, 0, 1, 3, 2, 1, 0,
    ];
    /// The shape of the small tests' buffers.
    const SHAPE: LegShape = LegShape {
        cap: CAP,
        slots_per_col: SPC,
        n: N,
        ff: FF,
    };

    /// Next value of a xorshift64 stream.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Write the f16 `d` (`bits`) of one block of `ty`: IQ1_M has no `d` field, its f16 is spread
    /// over the top nibbles of the four scale halfwords that close the block; every other type
    /// opens its block with it.
    fn put_d(ty: GgmlType, block: &mut [u8], bits: u16) {
        if ty == GgmlType::IQ1_M {
            for j in 0..4 {
                let top = u8::try_from((bits >> (4 * j)) & 0xf).expect("a nibble");
                block[49 + 2 * j] = (block[49 + 2 * j] & 0x0f) | (top << 4);
            }
        } else {
            block[..2].copy_from_slice(&bits.to_le_bytes());
        }
    }

    /// `rows` rows of `k` values of `ty` in the file's block bytes: every
    /// byte random but each block's scales, which are finite and small, so no
    /// dot is NaN or infinite — the f16 `d` (and `dmin` of Q4_K and Q5_K, `m`
    /// of Q5_1; Q6_K's last in its block), IQ1_M's, which [`put_d`] spreads over its scale
    /// halfwords, or MXFP4's E8M0 byte. Every other byte of these types is a valid code, so any
    /// draw is a valid block.
    pub(crate) fn stack_bytes(ty: GgmlType, rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let block = usize::try_from(ty.type_size().expect("a block size")).expect("a size");
        let width = usize::try_from(ty.blck_size().expect("a block width")).expect("a width");
        let f16s = if matches!(ty, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q5_1) {
            2
        } else {
            1
        };
        let mut state = seed;
        let mut out = vec![0u8; rows * (k / width) * block];
        for b in out.chunks_exact_mut(block) {
            for chunk in b.chunks_mut(8) {
                let v = next(&mut state).to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
            if ty == GgmlType::MXFP4 {
                b[0] = 118 + u8::try_from(next(&mut state) % 4).expect("a byte");
                continue;
            }
            if ty == GgmlType::IQ1_M {
                let (exp, mant) = (5 + next(&mut state) % 4, next(&mut state) & 0x3ff);
                put_d(ty, b, u16::try_from(exp << 10 | mant).expect("an f16"));
                continue;
            }
            // Q6_K's `d` closes its block; every other type's scales open it.
            let at = if ty == GgmlType::Q6_K { block - 2 } else { 0 };
            for scale in b[at..at + 2 * f16s].chunks_mut(2) {
                let (exp, mant) = (5 + next(&mut state) % 4, next(&mut state) & 0x3ff);
                let half = u16::try_from(exp << 10 | mant).expect("an f16");
                scale.copy_from_slice(&half.to_le_bytes());
            }
        }
        out
    }

    /// The little-endian words of a byte stream of `rows` rows as the card
    /// holds it (`runtime::words::stream_words`: zero-padded to whole words a
    /// row).
    pub(crate) fn words_of(bytes: &[u8], rows: usize) -> Vec<u32> {
        let total = stream_words(bytes.len() as u64, rows as u64).expect("a stack of rows");
        let mut w = vec![0u32; usize::try_from(total).expect("a word count")];
        for (word, c) in w.iter_mut().zip(bytes.chunks(4)) {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            *word = u32::from_le_bytes(b);
        }
        w
    }

    /// The loader keeps a tensor of `ty` as the file's rows.
    fn loads_as_file_rows(ty: GgmlType) -> bool {
        CardFormat::of(ty) == Some(CardFormat::KQuant)
    }

    /// The resident name of the `role` stack of `ty` that [`weights`] files:
    /// the file's own tensor for the types the loader keeps as the file's
    /// rows, a derived weight for the rest.
    fn stack_name(role: &str, ty: GgmlType) -> String {
        if loads_as_file_rows(ty) {
            format!("{role}_{ty}")
        } else {
            format!("derived.{role}_{ty}")
        }
    }

    /// The stacks `<role>_<ty>` of each type in `tys`, one for each role the
    /// leg runs the type in — gate and up `N_CARD · ff` rows of `n`, the down
    /// `N_CARD · n` rows of `ff` — resident in a [`Weights`] as
    /// `DevWeight::KQuant`, named by [`stack_name`]. The types the loader
    /// keeps as the file's rows (Q4_K, Q5_K) come through a file written for
    /// them, so `tys` holds one; the other types' rows are the same word
    /// stream, filed as derived weights (Q5_0's in the loader's packed
    /// layout, `DevWeight::Q5_0`).
    fn weights(gpu: &Gpu, shape: LegShape, tys: &[GgmlType]) -> Weights {
        weights_for(gpu, shape, tys, |role, ty| match role {
            "down" => kernels::card_down_runs(ty),
            _ => kernels::card_gate_up_runs(ty),
        })
    }

    /// [`weights`] with the roles each type is filed in chosen by `admits`
    /// (`"gate"`, `"up"` or `"down"`, and the type).
    fn weights_for(
        gpu: &Gpu,
        shape: LegShape,
        tys: &[GgmlType],
        admits: impl Fn(&str, GgmlType) -> bool,
    ) -> Weights {
        let LegShape { n, ff, .. } = shape;
        let mut decls = Vec::new();
        let mut data = Vec::new();
        let mut derived = Vec::new();
        for (i, &ty) in tys.iter().enumerate() {
            for (role, rows, k) in [
                ("gate", N_CARD * ff, n),
                ("up", N_CARD * ff, n),
                ("down", N_CARD * n, ff),
            ] {
                if !admits(role, ty) {
                    continue;
                }
                let seed = 0x9e37_79b9_7f4a_7c15 ^ ((i as u64) << 8 | role.len() as u64);
                let bytes = stack_bytes(ty, rows, k, seed);
                let name = stack_name(role, ty);
                if !loads_as_file_rows(ty) {
                    derived.push((name, ty, rows, k, bytes));
                    continue;
                }
                decls.push(TensorDecl {
                    name,
                    dims: vec![k as u64, rows as u64],
                    type_id: ty.as_u32(),
                    nbytes: bytes.len() as u64,
                });
                data.push(bytes);
            }
        }
        // One file per call: the tests run on parallel threads, and a second
        // File::create of a path the first test still maps is a SIGBUS.
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("bloomery-card-leg-{}-{call}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a temp dir");
        let path = dir.join("stacks.gguf");
        let layout = Layout::new(&[], decls.clone()).expect("a layout");
        let file = std::io::BufWriter::new(std::fs::File::create(&path).expect("the file"));
        let mut w = Writer::new(file, layout).expect("a writer");
        for (d, bytes) in decls.iter().zip(&data) {
            w.tensor(&d.name, bytes).expect("a tensor");
        }
        w.finish().expect("a finished file");
        let split = Split::open(&path).expect("the file opens");
        let mut weights = Weights::load_where(gpu.stream(), &split, |_| true).expect("resident");
        let _ = std::fs::remove_dir_all(&dir);
        for (name, ty, rows, k, bytes) in derived {
            // The loader's layout for the type: Q5_0 packed, else the file's words.
            let dev = if CardFormat::of(ty) == Some(CardFormat::Q5_0) {
                let packed = pack_q5_0(&bytes, k, rows).expect("a packed stack");
                let w = DeviceTensor::upload(gpu.stream(), &packed, rows, packed.len() / rows)
                    .expect("the stack's packed words");
                DevWeight::Q5_0 { w, k }
            } else {
                let words = words_of(&bytes, rows);
                let w = DeviceTensor::upload(gpu.stream(), &words, rows, words.len() / rows)
                    .expect("the stack's words");
                DevWeight::KQuant { ty, w, k }
            };
            weights
                .insert_derived(name, dev)
                .expect("a free derived name");
        }
        weights
    }

    /// The resident words of the stack `name`: the file's blocks, or Q5_0's
    /// packed words.
    fn words<'a>(w: &'a Weights, name: &str) -> &'a DeviceTensor<u32> {
        match w.get(name) {
            Some(DevWeight::KQuant { w, .. } | DevWeight::Q5_0 { w, .. }) => w,
            _ => panic!("{name} is not a resident K-quant or packed Q5_0"),
        }
    }

    /// The leg's K-quant arms against the launches the table names, run by
    /// hand on buffers of exactly the run's slots: for each type, a run of
    /// every token its buffers hold and of fewer. The down of a run of fewer
    /// reads an activation of more columns than the run has slots. A slot the
    /// card serves must hold a finite value the run wrote, a slot it does not
    /// must still hold what it held, and the rows are the hand launches' bit
    /// for bit.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_runs_the_kquant_arms_as_their_sel_launches() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let kq = QuantSelKernels::load(gpu.context(), gpu.fault_word()).expect("the kquant module");
        let tys = [GgmlType::Q4_K, GgmlType::Q5_K];
        let w = weights(&gpu, SHAPE, &tys);
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let shape = SHAPE;
        let rule = Act::SiluMul;
        for ty in tys {
            let names = [
                format!("gate_{ty}"),
                format!("up_{ty}"),
                format!("down_{ty}"),
            ];
            let stacks = LegStacks::of(&w, [&names[0], &names[1], &names[2]]).expect("the stacks");
            stacks.check(N_CARD, shape).expect("the stacks' shape");
            let mut bufs = LegBufs::new(stream, shape).expect("the leg's buffers");
            for m in [CAP, CAP - 1] {
                let case = format!("{ty}, {m} tokens");
                let slots = m * SPC;
                bufs.down
                    .copy_from_host(stream, &vec![SENT; CAP * SPC * N])
                    .expect("the sentinel");
                let run = LegRun {
                    layer: 0,
                    stacks: &stacks,
                    normed: &normed,
                    m,
                    sel: &sel,
                    n_card: N_CARD,
                    rule,
                };
                leg.enqueue(&gpu, &mut bufs, &run, None)
                    .unwrap_or_else(|e| panic!("{case}: the leg: {e}"));

                let mut act_x = Q8Act::with_tier_cols(stream, m, N).expect("act_x");
                let mut h = DeviceBuffer::<f32>::zeroed(stream, slots * FF).expect("h");
                let mut act_h = Q8Act::with_slots(stream, slots, FF).expect("act_h");
                let mut want =
                    DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("the rows");
                let sink = gpu.layer_sink(0).expect("a sink");
                gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, m, 0)
                    .expect("q8_1");
                let a = GateUpAct {
                    wg: words(&w, &names[0]),
                    wu: words(&w, &names[1]),
                    act: &act_x,
                    sel: &sel,
                    n_slots: slots,
                    rows_per_expert: FF,
                    slots_per_col: SPC,
                    rule,
                };
                match ty {
                    GgmlType::Q4_K => kq.enqueue_gate_up_q4k(stream, &a, sink, &mut h),
                    _ => kq.enqueue_gate_up_q5k(stream, &a, sink, &mut h),
                }
                .unwrap_or_else(|e| panic!("{case}: the gate·up: {e}"));
                let q = QuantSel {
                    x: &h,
                    cols: 0..slots,
                    sel: &sel,
                    n_card: N_CARD,
                };
                gpu.q4k_sel()
                    .enqueue_quantize_sel(stream, &q, sink, &mut act_h)
                    .unwrap_or_else(|e| panic!("{case}: the quantizer: {e}"));
                let d = SelDown {
                    w: words(&w, &names[2]),
                    act: &act_h,
                    sel: &sel,
                    n_slots: slots,
                    rows_per_expert: N,
                };
                match ty {
                    GgmlType::Q4_K => gpu
                        .q4k_sel()
                        .enqueue_gemv_q4k_sel(stream, d.w, d.act, d.sel, slots, N, &mut want),
                    _ => kq.enqueue_gemv_q5k_sel(stream, &d, sink, &mut want),
                }
                .unwrap_or_else(|e| panic!("{case}: the down: {e}"));

                let want = want.to_host_vec(stream).expect("the hand rows");
                let got = bufs.down.to_host_vec(stream).expect("the leg's rows");
                assert!(
                    gpu.take_fault().expect("the fault word").is_none(),
                    "{case}"
                );
                assert_hand_rows(&case, &got, &want);
            }
        }
    }

    /// Prequantized inputs preserve the loaded protection policy and changed-input replay.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_prequantized_leg_preserves_protection_and_replay() {
        let gpu = Gpu::new().expect("a GPU");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let ty = GgmlType::Q5_K;
        let weights = weights(&gpu, SHAPE, &[ty]);
        let names = [
            stack_name("gate", ty),
            stack_name("up", ty),
            stack_name("down", ty),
        ];
        let stacks =
            LegStacks::of(&weights, [&names[0], &names[1], &names[2]]).expect("the stacks");
        let mut ordinary = LegBufs::new(stream, SHAPE).expect("ordinary buffers");
        let mut quantized = LegBufs::new(stream, SHAPE).expect("quantized buffers");
        let mut host = vec![0.0f32; CAP * N];
        let mut normed = DeviceBuffer::zeroed(stream, host.len()).expect("normed rows");
        let mut act = Q8Act::with_tier_cols(stream, CAP, N).expect("quantized input");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("places");
        let sentinel = vec![SENT; CAP * SPC * N];
        let mut expected = DeviceBuffer::from_host(stream, &sentinel).expect("ordinary output");
        let mut actual = DeviceBuffer::from_host(stream, &sentinel).expect("quantized output");
        for policy in [ExpertInput::Direct, ExpertInput::AmaxF16] {
            ordinary
                .set_expert_input(&gpu, policy)
                .expect("ordinary policy");
            quantized
                .set_expert_input(&gpu, policy)
                .expect("quantized policy");
            for m in [CAP, CAP - 1] {
                let graph = gpu
                    .capture(|_| {
                        gpu.enqueue_quantize_q8_1_cols(&normed, &mut act, m, 0)?;
                        leg.enqueue_rows_quantized(
                            &gpu,
                            &mut quantized,
                            &QuantRun {
                                layer: 0,
                                stacks: &stacks,
                                act: &act,
                                m,
                                sel: &sel,
                                n_card: N_CARD,
                                rule: Act::SiluMul,
                            },
                            &mut actual,
                        )
                    })
                    .expect("capture quantized input and protected leg");
                let amplitudes = match policy {
                    ExpertInput::Direct => [1.0f32, 2.0],
                    ExpertInput::AmaxF16 => [1.0f32, 8192.0],
                };
                for amplitude in amplitudes {
                    for (i, value) in host.iter_mut().enumerate() {
                        *value = (((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0) * amplitude;
                    }
                    normed.copy_from_host(stream, &host).expect("changed input");
                    expected
                        .copy_from_host(stream, &sentinel)
                        .expect("reference canaries");
                    actual
                        .copy_from_host(stream, &sentinel)
                        .expect("replay canaries");
                    leg.enqueue_rows(
                        &gpu,
                        &mut ordinary,
                        &LegRun {
                            layer: 0,
                            stacks: &stacks,
                            normed: &normed,
                            m,
                            sel: &sel,
                            n_card: N_CARD,
                            rule: Act::SiluMul,
                        },
                        &mut expected,
                    )
                    .expect("ordinary input with the same policy");
                    graph.launch(stream).expect("changed-input replay");
                    let want = expected.to_host_vec(stream).expect("ordinary rows");
                    let got = actual.to_host_vec(stream).expect("quantized rows");
                    assert!(gpu.take_fault().expect("fault word").is_none());
                    let active = m * SPC * N;
                    let case = format!("{policy:?}, {m} tokens, amplitude {amplitude}");
                    assert_hand_rows(&case, &got[..active], &want[..active]);
                    assert!(
                        got[active..].iter().all(|&value| value == SENT),
                        "{case}: tail"
                    );
                }
            }
        }
    }

    /// A gate and up of `ty` beside a Q4_K down, a combination the qwen35moe
    /// program's arms admit ([`accepts::leg_arms`]): the leg's rows are the
    /// hand launches' bit for bit (the table's gate·up entry, the quantizer
    /// and the Q4_K down), for a run of every token its buffers hold and of
    /// fewer, and the constructors of the families whose plan reads
    /// [`kernels::card_gate_up_runs`] refuse the stacks by name.
    fn leg_runs_gate_up_as_its_launch(ty: GgmlType) {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let kq = QuantSelKernels::load(gpu.context(), gpu.fault_word()).expect("the kquant module");
        let q6 = Q6kSelKernels::load(gpu.context(), gpu.fault_word()).expect("the q6k module");
        let tys = [ty, GgmlType::Q4_K];
        let w = weights_for(&gpu, SHAPE, &tys, |role, t| match role {
            "down" => t == GgmlType::Q4_K,
            _ => t == ty,
        });
        let names = [
            stack_name("gate", ty),
            stack_name("up", ty),
            stack_name("down", GgmlType::Q4_K),
        ];
        let names = [names[0].as_str(), names[1].as_str(), names[2].as_str()];
        assert!(
            matches!(
                LegStacks::of(&w, names),
                Err(GpuError::Tensor { what: WHAT, .. })
            ),
            "the common constructor keeps a {ty} gate and up off the card"
        );
        let arms = accepts::leg_arms(accepts::Program::Qwen35Body).expect("the body's arms");
        let stacks = LegStacks::of_arms(&w, names, arms, (GATE_UP, "a down the body's arms run"))
            .expect("the stacks");
        stacks.check(N_CARD, SHAPE).expect("the stacks' shape");
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let mut bufs = LegBufs::new(stream, SHAPE).expect("the leg's buffers");
        let rule = Act::SiluMul;
        for m in [CAP, CAP - 1] {
            let case = format!("{ty} gate·up, {m} tokens");
            let slots = m * SPC;
            let got = leg_rows(&gpu, &leg, &mut bufs, &stacks, &normed, &sel, m);

            let mut act_x = Q8Act::with_tier_cols(stream, m, N).expect("act_x");
            let mut h = DeviceBuffer::<f32>::zeroed(stream, slots * FF).expect("h");
            let mut act_h = Q8Act::with_slots(stream, slots, FF).expect("act_h");
            let mut want =
                DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("the rows");
            let sink = gpu.layer_sink(0).expect("a sink");
            gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, m, 0)
                .expect("q8_1");
            let a = GateUpAct {
                wg: words(&w, names[0]),
                wu: words(&w, names[1]),
                act: &act_x,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: FF,
                slots_per_col: SPC,
                rule,
            };
            match ty {
                GgmlType::Q8_0 => kq.enqueue_gate_up_q8_0(stream, &a, sink, &mut h),
                _ => q6.enqueue_gate_up(stream, &a, sink, &mut h),
            }
            .unwrap_or_else(|e| panic!("{case}: the gate·up: {e}"));
            let q = QuantSel {
                x: &h,
                cols: 0..slots,
                sel: &sel,
                n_card: N_CARD,
            };
            gpu.q4k_sel()
                .enqueue_quantize_sel(stream, &q, sink, &mut act_h)
                .unwrap_or_else(|e| panic!("{case}: the quantizer: {e}"));
            gpu.q4k_sel()
                .enqueue_gemv_q4k_sel(
                    stream,
                    words(&w, names[2]),
                    &act_h,
                    &sel,
                    slots,
                    N,
                    &mut want,
                )
                .unwrap_or_else(|e| panic!("{case}: the down: {e}"));
            let want = want.to_host_vec(stream).expect("the hand rows");
            assert!(
                gpu.take_fault().expect("the fault word").is_none(),
                "{case}"
            );
            assert_hand_rows(&case, &got, &want);
        }
    }

    /// A Q8_0 gate and up through the leg. Mutant: the arm launching the Q5_K
    /// gate·up for a Q8_0 stack — the rows part from the hand launch.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_runs_a_q8_0_gate_up_as_its_sel_launch() {
        leg_runs_gate_up_as_its_launch(GgmlType::Q8_0);
    }

    /// A Q6_K gate and up through the leg. Mutant: the arm launching the Q8_0
    /// gate·up for a Q6_K stack — the stacks are refused by name or the rows
    /// part from the hand launch.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_runs_a_q6k_gate_up_as_its_sel_launch() {
        leg_runs_gate_up_as_its_launch(GgmlType::Q6_K);
    }

    /// A Q4_K gate and up with a Q6_K down, the common leg's own stacks
    /// ([`LegStacks::of`]): the rows are the hand launches' bit for bit (the
    /// Q4_K gate·up, the quantizer, and `enqueue_gemv_q6k_sel` of the same
    /// weights), for a run of every token the buffers hold and of fewer.
    /// Mutant: the down arm launching the Q5_K `_sel` for a Q6_K stack — the
    /// rows part from the hand launch — or [`LegStacks::of`] refusing the stacks.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_runs_a_q6k_down_as_its_sel_launch() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let kq = QuantSelKernels::load(gpu.context(), gpu.fault_word()).expect("the kquant module");
        let q6 = Q6kSelKernels::load(gpu.context(), gpu.fault_word()).expect("the q6k module");
        let (gate, down) = (GgmlType::Q4_K, GgmlType::Q6_K);
        let w = weights_for(&gpu, SHAPE, &[gate, down], |role, t| match role {
            "down" => t == down,
            _ => t == gate,
        });
        let names = [
            stack_name("gate", gate),
            stack_name("up", gate),
            stack_name("down", down),
        ];
        let names = [names[0].as_str(), names[1].as_str(), names[2].as_str()];
        let stacks = LegStacks::of(&w, names).expect("a Q4_K gate and up with a Q6_K down");
        stacks.check(N_CARD, SHAPE).expect("the stacks' shape");
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let mut bufs = LegBufs::new(stream, SHAPE).expect("the leg's buffers");
        let rule = Act::SiluMul;
        for m in [CAP, CAP - 1] {
            let case = format!("Q6_K down, {m} tokens");
            let slots = m * SPC;
            let got = leg_rows(&gpu, &leg, &mut bufs, &stacks, &normed, &sel, m);

            let mut act_x = Q8Act::with_tier_cols(stream, m, N).expect("act_x");
            let mut h = DeviceBuffer::<f32>::zeroed(stream, slots * FF).expect("h");
            let mut act_h = Q8Act::with_slots(stream, slots, FF).expect("act_h");
            let mut want =
                DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("the rows");
            let sink = gpu.layer_sink(0).expect("a sink");
            gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, m, 0)
                .expect("q8_1");
            let a = GateUpAct {
                wg: words(&w, names[0]),
                wu: words(&w, names[1]),
                act: &act_x,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: FF,
                slots_per_col: SPC,
                rule,
            };
            kq.enqueue_gate_up_q4k(stream, &a, sink, &mut h)
                .unwrap_or_else(|e| panic!("{case}: the gate·up: {e}"));
            let q = QuantSel {
                x: &h,
                cols: 0..slots,
                sel: &sel,
                n_card: N_CARD,
            };
            gpu.q4k_sel()
                .enqueue_quantize_sel(stream, &q, sink, &mut act_h)
                .unwrap_or_else(|e| panic!("{case}: the quantizer: {e}"));
            q6.enqueue_gemv_q6k_sel(
                stream,
                words(&w, names[2]),
                &act_h,
                &sel,
                slots,
                N,
                &mut want,
            )
            .unwrap_or_else(|e| panic!("{case}: the down: {e}"));
            let want = want.to_host_vec(stream).expect("the hand rows");
            assert!(
                gpu.take_fault().expect("the fault word").is_none(),
                "{case}"
            );
            assert_hand_rows(&case, &got, &want);
        }
    }

    /// `q6k_gate_up_sel` against its parts: the gate and up rows of a slot are
    /// the two dots `q6k_gemv_sel` writes for the slot's column (the same row
    /// walk, bit for bit, one activation column a slot), and `h` is the rule
    /// over them — within the device's fused multiply-adds of the host's
    /// transcription ([`crate::kquant::act::apply`]), for a run of every token the
    /// activation holds and of fewer; a slot the card does not serve keeps
    /// its sentinel, and an id past the stack's experts that is not the host's
    /// raises [`FaultSite::ExpertId`] and writes NaN into its rows. Mutant: the
    /// up sum read from the gate stack — `h` is `silu(g) · g`.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_q6k_gate_up_is_its_two_dots_and_the_rule() {
        use crate::fault::{FaultSite, LAYER_NONE};
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let q6 = Q6kSelKernels::load(gpu.context(), gpu.fault_word()).expect("the q6k module");
        let w = weights_for(&gpu, SHAPE, &[GgmlType::Q6_K], |role, _| role != "down");
        let (wg, wu) = (
            words(&w, &stack_name("gate", GgmlType::Q6_K)),
            words(&w, &stack_name("up", GgmlType::Q6_K)),
        );
        let rows_of = |t: usize| row_of(t, N);
        let (code, limit) = Act::SiluMul.code("test").expect("the rule's code");
        for m in [CAP, CAP - 1] {
            let slots = m * SPC;
            let case = format!("q6_K gate·up, {m} tokens");
            let normed_host: Vec<f32> = (0..m).flat_map(rows_of).collect();
            let normed = DeviceBuffer::from_host(stream, &normed_host).expect("the normed rows");
            // One column a slot for the hand dots: the slot's token's row.
            let by_slot: Vec<f32> = (0..slots).flat_map(|s| rows_of(s / SPC)).collect();
            let by_slot = DeviceBuffer::from_host(stream, &by_slot).expect("the rows a slot");
            let mut act_x = Q8Act::with_tier_cols(stream, m, N).expect("act_x");
            let mut act_s = Q8Act::with_slots(stream, slots, N).expect("act_s");
            gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, m, 0)
                .expect("q8_1");
            gpu.enqueue_quantize_q8_1_cols(&by_slot, &mut act_s, slots, 0)
                .expect("q8_1 a slot");
            let sel = DeviceBuffer::from_host(stream, &SEL[..slots]).expect("the places");
            let sink = gpu.layer_sink(0).expect("a sink");
            let mut h = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("h");
            let a = GateUpAct {
                wg,
                wu,
                act: &act_x,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: FF,
                slots_per_col: SPC,
                rule: Act::SiluMul,
            };
            q6.enqueue_gate_up(stream, &a, sink, &mut h)
                .unwrap_or_else(|e| panic!("{case}: the gate·up: {e}"));
            let mut g = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("g");
            let mut u = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("u");
            q6.enqueue_gemv_q6k_sel(stream, wg, &act_s, &sel, slots, FF, &mut g)
                .unwrap_or_else(|e| panic!("{case}: the gate dots: {e}"));
            q6.enqueue_gemv_q6k_sel(stream, wu, &act_s, &sel, slots, FF, &mut u)
                .unwrap_or_else(|e| panic!("{case}: the up dots: {e}"));
            let (h, g, u) = (
                h.to_host_vec(stream).expect("h"),
                g.to_host_vec(stream).expect("g"),
                u.to_host_vec(stream).expect("u"),
            );
            assert!(
                gpu.take_fault().expect("the fault word").is_none(),
                "{case}"
            );
            for (s, &place) in SEL[..slots].iter().enumerate() {
                for r in s * FF..(s + 1) * FF {
                    if place == HOST {
                        assert_eq!(h[r].to_bits(), SENT.to_bits(), "{case}: host slot {s}");
                        continue;
                    }
                    let want = crate::kquant::act::apply(code, limit, g[r], u[r]);
                    assert!(
                        want.is_finite() && (h[r] - want).abs() <= 2e-6 * (1.0 + want.abs()),
                        "{case}: slot {s} row {}: kernel {} against the rule over the two dots {want}",
                        r - s * FF,
                        h[r]
                    );
                }
            }
        }
        // An id past the stack: fault and NaN, the other slot untouched.
        let bad = DeviceBuffer::from_host(stream, &[N_CARD as u32 + 5, 1]).expect("the places");
        let normed_host: Vec<f32> = rows_of(0);
        let normed = DeviceBuffer::from_host(stream, &normed_host).expect("the normed row");
        let mut act_x = Q8Act::with_tier_cols(stream, 1, N).expect("act_x");
        gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, 1, 0)
            .expect("q8_1");
        let mut h = DeviceBuffer::from_host(stream, &vec![SENT; 2 * FF]).expect("h");
        let a = GateUpAct {
            wg,
            wu,
            act: &act_x,
            sel: &bad,
            n_slots: 2,
            rows_per_expert: FF,
            slots_per_col: 2,
            rule: Act::SiluMul,
        };
        q6.enqueue_gate_up(stream, &a, gpu.unlabelled_sink(), &mut h)
            .expect("the launch");
        let h = h.to_host_vec(stream).expect("h");
        let fault = gpu.take_fault().expect("the fault word");
        assert!(
            fault.is_some_and(|f| f.code == FaultSite::ExpertId as u32 && f.layer == LAYER_NONE),
            "an id past the stack raises ExpertId, got {fault:?}"
        );
        assert!(
            h[..FF].iter().all(|v| v.is_nan()),
            "the bad slot's rows are NaN"
        );
        assert!(
            h[FF..].iter().all(|v| v.is_finite() && *v != SENT),
            "the good slot is written"
        );
    }

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_gate_up_sums_fault_before_activation() {
        use crate::fault::{Fault, FaultSite, LAYER_NONE};
        use crate::iq_sel::IqGateUp;
        use crate::kquant::{
            EntryPlanes, GateUpAt, KquantTileKernels, TileGather, TileTable, TiledGateUpAct,
        };
        const K: usize = 256;
        const ROWS: usize = 8;
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let sink = gpu.unlabelled_sink();
        let kernels = QuantSelKernels::load(gpu.context(), gpu.fault_word()).expect("kernels");
        let tiles = KquantTileKernels::load(gpu.context(), gpu.fault_word()).expect("tiles");
        let x = DeviceBuffer::from_host(stream, &[1e38f32; K]).expect("x");
        let mut act = Q8Act::with_k(stream, 1, K).expect("act");
        gpu.enqueue_quantize_q8_1(&x, &mut act).expect("quantize");
        let sel = DeviceBuffer::from_host(stream, &[0u32]).expect("sel");
        let start = DeviceBuffer::from_host(stream, &[0u32, 1]).expect("start");
        let mut table_tiles =
            DeviceBuffer::zeroed(stream, crate::q4k_sel::tile_cap(1, 1) + 1).expect("tiles");
        gpu.q4k_sel()
            .enqueue_grouped_tiles(stream, &start, 1, 1, sink, &mut table_tiles)
            .expect("tile table");
        let table = || TileTable {
            order: &sel,
            start: &start,
            tiles: &table_tiles,
            n_experts: 1,
            n_slots: 1,
        };
        let mut planes = EntryPlanes::new(stream, 1, K).expect("planes");
        tiles
            .enqueue_card_gather(
                stream,
                &TileGather {
                    x: &act,
                    cols: 1,
                    slots_per_col: 1,
                    table: table(),
                },
                sink,
                &mut planes,
            )
            .expect("gather");
        assert_eq!(gpu.take_fault().expect("input fault"), None);
        let mut failures = Vec::new();
        for ty in [
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q8_0,
            GgmlType::IQ2_XS,
            GgmlType::IQ2_S,
            GgmlType::IQ1_S,
            GgmlType::IQ1_M,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
        ] {
            let mut bytes = stack_bytes(ty, ROWS, K, 42);
            let bb = ty.type_size().expect("block bytes") as usize;
            for block in bytes.chunks_exact_mut(bb) {
                put_d(ty, block, 0x7bff);
            }
            let up = words_of(&bytes, ROWS);
            let gate = vec![0u32; up.len()];
            let wg = DeviceTensor::upload(stream, &gate, ROWS, gate.len() / ROWS).expect("gate");
            let wu = DeviceTensor::upload(stream, &up, ROWS, up.len() / ROWS).expect("up");
            let arena = DeviceBuffer::from_host(stream, &[gate, up].concat()).expect("arena");
            let offs = DeviceBuffer::from_host(stream, &[0u64, (bytes.len() / 4) as u64, 0])
                .expect("offsets");
            let modes: &[&str] = match ty {
                GgmlType::Q4_K => &["sel", "sel_gate", "arena", "tiles"],
                GgmlType::Q5_K => &["sel", "arena", "tiles"],
                _ => &["sel"],
            };
            let rules = [
                Act::SwigluClamp { limit: 7.0 },
                Act::SwigluOai {
                    alpha: crate::kquant::act::SWIGLU_OAI_ALPHA,
                    limit: 7.0,
                },
            ];
            for (&mode, rule) in modes.iter().flat_map(|m| rules.map(|r| (m, r))) {
                let (gate, up) = if mode == "sel_gate" {
                    (&wu, &wg)
                } else {
                    (&wg, &wu)
                };
                let mut h = DeviceBuffer::zeroed(stream, ROWS).expect("h");
                let a = GateUpAct {
                    wg: gate,
                    wu: up,
                    act: &act,
                    sel: &sel,
                    n_slots: 1,
                    rows_per_expert: ROWS,
                    slots_per_col: 1,
                    rule,
                };
                match mode {
                    "arena" => {
                        let a = GateUpAt {
                            arena: &arena,
                            offs: &offs,
                            act: &act,
                            n_slots: 1,
                            rows_per_expert: ROWS,
                            slots_per_col: 1,
                            rule,
                        };
                        match ty {
                            GgmlType::Q4_K => {
                                kernels.enqueue_gate_up_q4k_at(stream, &a, sink, &mut h)
                            }
                            _ => kernels.enqueue_gate_up_q5k_at(stream, &a, sink, &mut h),
                        }
                    }
                    "tiles" => tiles.enqueue_gate_up_tiles(
                        stream,
                        ty,
                        &TiledGateUpAct {
                            wg: &wg,
                            wu: &wu,
                            act: &planes,
                            table: table(),
                            rows_per_expert: ROWS,
                            rule,
                        },
                        sink,
                        &mut h,
                    ),
                    _ => match ty {
                        GgmlType::Q4_K => kernels.enqueue_gate_up_q4k(stream, &a, sink, &mut h),
                        GgmlType::Q5_K => kernels.enqueue_gate_up_q5k(stream, &a, sink, &mut h),
                        GgmlType::Q8_0 => kernels.enqueue_gate_up_q8_0(stream, &a, sink, &mut h),
                        _ => kernels.enqueue_gate_up(
                            stream,
                            &IqGateUp {
                                ty,
                                wg: &wg,
                                wu: &wu,
                                act: &act,
                                sel: &sel,
                                n_slots: 1,
                                rows_per_expert: ROWS,
                                slots_per_col: 1,
                                rule,
                            },
                            sink,
                            &mut h,
                        ),
                    },
                }
                .expect("gate up");
                let fault = gpu.take_fault().expect("fault");
                let rows = h.to_host_vec(stream).expect("rows");
                let ok = fault == Some(Fault::at(LAYER_NONE, FaultSite::F32Product))
                    && rows.iter().all(|v| v.is_nan());
                println!(
                    "{ty:?} {mode} {rule:?}: fault={fault:?} nan={} {ok}",
                    rows.iter().filter(|v| v.is_nan()).count()
                );
                if !ok {
                    failures.push(format!("{ty:?} {mode} {rule:?}: {fault:?} {rows:?}"));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// An NVFP4 stack of `N_CARD` experts of `rows` rows of `k` values in its
    /// planes, random codes under small UE4M3 scale bytes, and a scale an
    /// expert that differs between experts, filed as derived weight `name`.
    fn nvfp4_stack(gpu: &Gpu, w: &mut Weights, name: &str, rows: usize, k: usize, seed: u64) {
        let mut state = seed;
        let mut bytes = vec![0u8; N_CARD * rows * k / 64 * 36];
        for b in bytes.as_chunks_mut::<36>().0 {
            for (i, v) in b.iter_mut().enumerate() {
                let r = u8::try_from(next(&mut state) & 0xff).expect("a byte");
                *v = if i < 4 { 0x30 | (r & 0x0f) } else { r };
            }
        }
        let p = crate::mxfp4::repack::<crate::mxfp4::Nvfp4>(&bytes, N_CARD * rows, k)
            .expect("the planes");
        let scales: Vec<f32> = (0..N_CARD).map(|e| 0.75 + 0.125 * e as f32).collect();
        let stream = gpu.stream();
        let stack = DevWeight::Nvfp4Stack {
            qs: DeviceTensor::upload(stream, &p.qs, N_CARD * rows, k / 8).expect("qs"),
            e: DeviceTensor::upload(stream, &p.e, N_CARD * rows, k / 64).expect("e"),
            scales: DeviceTensor::upload(stream, &scales, N_CARD, 1).expect("the scales"),
            k,
        };
        w.insert_derived(name.to_owned(), stack)
            .expect("a free derived name");
    }

    /// The leg's NVFP4 arms against the launches the table names, run by hand
    /// on buffers of exactly the run's slots: the gate·up over a `Q8Act`, the
    /// down over the 32-value form of buffers that hold both forms (a load
    /// whose layers' downs read either), for a run of every token the buffers
    /// hold and of fewer. A slot the card serves holds a
    /// finite value the run wrote, a host slot what it held, and the rows are
    /// the hand launches' bit for bit; the up and down stacks the leg holds in
    /// planes are the relu² form's as well, of the shape's rows and widths.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_runs_the_nvfp4_arms_as_their_sel_launches() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let nv = Nvfp4SelKernels::load(gpu.context(), gpu.fault_word()).expect("the nvfp4 module");
        let mut w = weights(&gpu, SHAPE, &[]);
        let names = ["derived.gate_nv", "derived.up_nv", "derived.down_nv"];
        nvfp4_stack(&gpu, &mut w, names[0], FF, N, 0x11);
        nvfp4_stack(&gpu, &mut w, names[1], FF, N, 0x22);
        nvfp4_stack(&gpu, &mut w, names[2], N, FF, 0x33);
        let stacks = LegStacks::of(&w, names).expect("the stacks");
        stacks.check(N_CARD, SHAPE).expect("the stacks' shape");
        Relu2Stacks::of(&w, [names[1], names[2]])
            .expect("the relu² form's stacks")
            .check(N_CARD, SHAPE)
            .expect("the relu² stacks' shape");
        // A prompt walk's route GEMM reads a resident NVFP4 stack as its planes.
        let (ty, stack) = StackW::resident(WHAT, &w, names[1]).expect("a resident NVFP4 stack");
        assert_eq!(ty, GgmlType::NVFP4);
        assert!(
            matches!(
                stack.route(ty),
                Some(RouteW::Blocks32(Gemm32Weight::Nvfp4Planes(_)))
            ),
            "the 32-value GEMM reads an NVFP4 stack as its planes"
        );
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let rule = Act::SwigluClamp { limit: 7.0 };
        let mut bufs = LegBufs::for_downs(stream, SHAPE, BOTH).expect("the leg's buffers");
        let plane = |name: &str| w.nvfp4_stack(WHAT, name).expect("a resident stack");
        for m in [CAP, CAP - 1] {
            let case = format!("NVFP4, {m} tokens");
            let slots = m * SPC;
            bufs.down
                .copy_from_host(stream, &vec![SENT; CAP * SPC * N])
                .expect("the sentinel");
            let run = LegRun {
                layer: 0,
                stacks: &stacks,
                normed: &normed,
                m,
                sel: &sel,
                n_card: N_CARD,
                rule,
            };
            leg.enqueue(&gpu, &mut bufs, &run, None)
                .unwrap_or_else(|e| panic!("{case}: the leg: {e}"));

            let mut act_x = Q8Act::with_tier_cols(stream, m, N).expect("act_x");
            let mut h = DeviceBuffer::<f32>::zeroed(stream, slots * FF).expect("h");
            let mut act_h = Q8Blocks32::with_slots(stream, FF, slots).expect("act_h");
            let mut want =
                DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("the rows");
            let sink = gpu.layer_sink(0).expect("a sink");
            gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, m, 0)
                .expect("q8_1");
            let a = Nvfp4GateUp {
                gate: plane(names[0]),
                up: plane(names[1]),
                act: &act_x,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: FF,
                slots_per_col: SPC,
                rule,
            };
            nv.enqueue_gate_up(stream, &a, sink, &mut h)
                .unwrap_or_else(|e| panic!("{case}: the gate·up: {e}"));
            let q = QuantSel {
                x: &h,
                cols: 0..slots,
                sel: &sel,
                n_card: N_CARD,
            };
            gpu.q5()
                .enqueue_quantize_q8_sel(stream, &q, &mut act_h, sink)
                .unwrap_or_else(|e| panic!("{case}: the quantizer: {e}"));
            let d = Nvfp4Down {
                w: plane(names[2]),
                act: &act_h,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: N,
                slots_per_col: SPC,
            };
            nv.enqueue_sel32(stream, &d, sink, &mut want)
                .unwrap_or_else(|e| panic!("{case}: the down: {e}"));

            let want = want.to_host_vec(stream).expect("the hand rows");
            let got = bufs.down.to_host_vec(stream).expect("the leg's rows");
            assert!(
                gpu.take_fault().expect("the fault word").is_none(),
                "{case}"
            );
            assert_hand_rows(&case, &got, &want);
        }
    }

    /// Tokens the wide test's buffers hold: more than a warp's 32 columns.
    const WIDE_CAP: usize = 40;
    /// The wide test's normed width: ten super-blocks, so the last group of
    /// four in each column's q8_1 planes is partial.
    const WIDE_N: usize = 2560;
    /// The wide test's routed width: five super-blocks and forty 32-value
    /// blocks, a partial last group in both forms of the down's activation.
    const WIDE_FF: usize = 1280;
    /// The shape of the wide test's buffers.
    const WIDE_SHAPE: LegShape = LegShape {
        cap: WIDE_CAP,
        slots_per_col: SPC,
        n: WIDE_N,
        ff: WIDE_FF,
    };
    /// The tokens of the wide test's runs: one, one past a group of eight
    /// columns, and the buffers' cap.
    const WIDE_RUNS: [usize; 3] = [1, 9, WIDE_CAP];
    /// The wide test's cases, `(gate·up type, down type)`: the eleven types of
    /// `kernels::card_gate_up_runs` each as a gate·up beside a down of its own
    /// type, then the three 32-value downs and the Q6_K down
    /// (`kernels::card_down_runs`) beside a gate·up that ran before.
    // PIN(2026-10-22): Q6_K joined `card_down_runs`, so the cases gain a Q5_K
    // gate·up over a Q6_K down. The set was the ten rows without that pair.
    // PIN(2026-10-23): IQ2_XXS joins both sets with its gate·up (`iq2_xxs_gate_up_sel`)
    // and down (`iq2_xxs_gemv_sel`); the set was the eleven rows without its
    // own-type pair.
    // PIN(2026-10-25): IQ2_S, IQ1_S and IQ1_M join both sets with their `_sel` gate·up and
    // down; the set was the twelve rows without their own-type pairs.
    const WIDE_CASES: [(GgmlType, GgmlType); 15] = [
        (GgmlType::Q4_K, GgmlType::Q4_K),
        (GgmlType::Q5_K, GgmlType::Q5_K),
        (GgmlType::MXFP4, GgmlType::MXFP4),
        (GgmlType::IQ2_XXS, GgmlType::IQ2_XXS),
        (GgmlType::IQ2_XS, GgmlType::IQ2_XS),
        (GgmlType::IQ2_S, GgmlType::IQ2_S),
        (GgmlType::IQ1_S, GgmlType::IQ1_S),
        (GgmlType::IQ1_M, GgmlType::IQ1_M),
        (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS),
        (GgmlType::IQ3_S, GgmlType::IQ3_S),
        (GgmlType::IQ4_XS, GgmlType::IQ4_XS),
        (GgmlType::Q4_K, GgmlType::Q5_1),
        (GgmlType::Q5_K, GgmlType::Q8_0),
        (GgmlType::IQ4_XS, GgmlType::IQ4_NL),
        (GgmlType::Q5_K, GgmlType::Q6_K),
    ];
    /// The wide test's case of a stack in its planes (`kernels::card_planes_runs`):
    /// NVFP4, whose runs of more than one token run each expert's rows on the
    /// warp of the first slot that names it, for every slot that names it.
    const WIDE_PLANES: [(GgmlType, GgmlType); 1] = [(GgmlType::NVFP4, GgmlType::NVFP4)];

    /// Token `t`'s normed row: a function of `t` alone, so a column holds the
    /// same values at every `m`.
    fn wide_row(t: usize) -> Vec<f32> {
        row_of(t, WIDE_N)
    }

    /// Token `t`'s normed row of `n` values, a function of `t` alone.
    fn row_of(t: usize, n: usize) -> Vec<f32> {
        let mut state = (t as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (0..n)
            .map(|_| ((next(&mut state) >> 33) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    /// Token `t`'s places, a function of `t` alone: slot `j` serves expert
    /// `(j + t) % N_CARD`, but the slot `(3t + 1) % SPC`, which the card does
    /// not serve. A run of any `m` therefore mixes duplicates (seven expert
    /// slots of four experts a token), every card expert and host slots.
    fn wide_places(t: usize) -> [u32; SPC] {
        places_of(t)
    }

    /// [`wide_places`] for tokens of `S` slots.
    fn places_of<const S: usize>(t: usize) -> [u32; S] {
        std::array::from_fn(|j| {
            if j == (3 * t + 1) % S {
                HOST
            } else {
                ((j + t) % N_CARD) as u32
            }
        })
    }

    /// `sel` repeats a card expert, names every card expert, and holds a slot
    /// the card does not serve.
    fn mixes(sel: &[u32]) -> bool {
        let mut seen = [0usize; N_CARD];
        for &place in sel {
            if let Some(count) = seen.get_mut(place as usize) {
                *count += 1;
            }
        }
        seen.iter().all(|&c| c > 0) && seen.iter().any(|&c| c > 1) && sel.contains(&HOST)
    }

    /// What a run's rows of one slot must hold: a host slot's are the
    /// sentinel, a card slot's are finite, not the sentinel and not all zero.
    fn slot_state(rows: &[f32], place: u32) -> Result<(), &'static str> {
        if place == HOST {
            if rows.iter().all(|v| *v == SENT) {
                return Ok(());
            }
            return Err("a host slot was written");
        }
        if !rows.iter().all(|v| v.is_finite() && *v != SENT) {
            return Err("a card slot holds no finite row the run wrote");
        }
        if rows.iter().all(|v| *v == 0.0) {
            return Err("a card slot's rows are all zero");
        }
        Ok(())
    }

    fn assert_hand_rows(case: &str, got: &[f32], want: &[f32]) {
        for (s, &place) in SEL[..want.len() / N].iter().enumerate() {
            let rows = s * N..(s + 1) * N;
            slot_state(&got[rows.clone()], place)
                .unwrap_or_else(|why| panic!("{case}: slot {s} (place {place}): {why}"));
            if let Some(r) = rows
                .clone()
                .find(|&r| got[r].to_bits() != want[r].to_bits())
            {
                panic!(
                    "{case}: slot {s} row {}: leg {:#010x}, hand {:#010x}",
                    r - rows.start,
                    got[r].to_bits(),
                    want[r].to_bits()
                );
            }
        }
        assert!(
            got[want.len()..].iter().all(|v| *v == SENT),
            "{case}: the leg wrote past its run's slots"
        );
    }

    /// The down rows of a run of the first `m` tokens of `normed` against
    /// `sel` in `bufs`, filled with the sentinel before it, synchronized and
    /// fault-free.
    fn leg_rows(
        gpu: &Gpu,
        leg: &CardLeg,
        bufs: &mut LegBufs,
        stacks: &LegStacks<'_>,
        normed: &DeviceBuffer<f32>,
        sel: &DeviceBuffer<u32>,
        m: usize,
    ) -> Vec<f32> {
        let stream = gpu.stream();
        let LegShape {
            cap,
            slots_per_col,
            n,
            ..
        } = bufs.shape();
        bufs.down
            .copy_from_host(stream, &vec![SENT; cap * slots_per_col * n])
            .expect("the sentinel");
        let run = LegRun {
            layer: 0,
            stacks,
            normed,
            m,
            sel,
            n_card: N_CARD,
            rule: Act::SiluMul,
        };
        leg.enqueue(gpu, bufs, &run, None)
            .unwrap_or_else(|e| panic!("a run of {m} tokens: the leg: {e}"));
        let rows = bufs.down.to_host_vec(stream).expect("the leg's rows");
        assert!(
            gpu.take_fault().expect("the fault word").is_none(),
            "a run of {m} tokens raised a fault"
        );
        rows
    }

    /// A slot's rows are what a one-token run writes for it, whatever `m`
    /// (the module doc), for every type [`kernels::card_gate_up_runs`] and
    /// [`kernels::card_down_runs`] admit, in fifteen cases (`WIDE_CASES`), and for
    /// NVFP4's planes (`WIDE_PLANES`). Runs of
    /// `m` in `WIDE_RUNS` tokens of eight slots go through fresh buffers of
    /// `WIDE_CAP` tokens, so the run's columns beyond the eighth, and beyond a
    /// warp's 32, are read through the leg's own quantize steps and buffers;
    /// the places and rows are functions of the token, so the same token holds
    /// the same slots in every run. The oracle is the leg's own run of that one
    /// token in buffers of one token and its down's one form; the runs' buffers
    /// hold both forms ([`LegBufs::for_downs`]), so each down reads the columns
    /// of its own form beside the other's. Each card slot's rows must equal its
    /// oracle's bit for bit, a host slot must keep its sentinel, and nothing
    /// past the run's slots may be written. Every case and `m` is judged, and
    /// the failures are listed together, each naming the slot and its column.
    ///
    /// The K-quant test above holds the leg's Q4_K and Q5_K arms to the launches
    /// the table names at one `m`; this one holds each arm's slots to the
    /// one-token run across `m`.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_writes_each_slots_one_token_rows_at_any_m() {
        let ids = |tys: Vec<GgmlType>| {
            let mut ids: Vec<u32> = tys.into_iter().map(GgmlType::as_u32).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        let admitted = |runs: fn(GgmlType) -> bool| {
            ids((0..64)
                .map(GgmlType::from_u32)
                .filter(|&ty| runs(ty))
                .collect())
        };
        assert_eq!(
            ids(WIDE_CASES.iter().map(|c| c.0).collect()),
            admitted(kernels::card_gate_up_runs),
            "the cases' gate·up types are not the types the leg admits"
        );
        assert_eq!(
            ids(WIDE_CASES.iter().map(|c| c.1).collect()),
            admitted(kernels::card_down_runs),
            "the cases' down types are not the types the leg admits"
        );
        let sel_all: Vec<u32> = (0..WIDE_CAP).flat_map(wide_places).collect();
        for m in WIDE_RUNS {
            assert!(
                mixes(&sel_all[..m * SPC]),
                "the places of {m} tokens must mix duplicates, every card expert and host slots"
            );
        }

        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let mut tys: Vec<GgmlType> = WIDE_CASES.iter().flat_map(|&(g, d)| [g, d]).collect();
        tys.sort_by_key(|ty| ty.as_u32());
        tys.dedup();
        let mut w = weights(&gpu, WIDE_SHAPE, &tys);
        for (gu, _) in WIDE_PLANES {
            assert!(kernels::card_planes_runs(gu), "{gu} is not a planes type");
            for (role, rows, k, seed) in [
                ("gate", WIDE_FF, WIDE_N, 0x41),
                ("up", WIDE_FF, WIDE_N, 0x42),
                ("down", WIDE_N, WIDE_FF, 0x43),
            ] {
                nvfp4_stack(&gpu, &mut w, &stack_name(role, gu), rows, k, seed);
            }
        }
        let host_normed: Vec<f32> = (0..WIDE_CAP).flat_map(wide_row).collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &sel_all).expect("the places");
        let one = LegShape {
            cap: 1,
            ..WIDE_SHAPE
        };

        let mut failures = Vec::new();
        for (gu, dn) in WIDE_CASES.into_iter().chain(WIDE_PLANES) {
            let names = [
                stack_name("gate", gu),
                stack_name("up", gu),
                stack_name("down", dn),
            ];
            let stacks = LegStacks::of(&w, [&names[0], &names[1], &names[2]]).expect("the stacks");
            stacks.check(N_CARD, WIDE_SHAPE).expect("the stacks' shape");
            let form = kernels::down(dn).expect("the down's entry").1;
            let mut one_bufs = LegBufs::for_down(stream, one, form).expect("one token's buffers");
            let oracle: Vec<Vec<f32>> = (0..WIDE_CAP)
                .map(|t| {
                    let row = DeviceBuffer::from_host(stream, &wide_row(t)).expect("a normed row");
                    let places = wide_places(t);
                    let sel_t = DeviceBuffer::from_host(stream, &places).expect("a token's places");
                    let rows = leg_rows(&gpu, &leg, &mut one_bufs, &stacks, &row, &sel_t, 1);
                    for (j, &place) in places.iter().enumerate() {
                        if let Err(why) = slot_state(&rows[j * WIDE_N..(j + 1) * WIDE_N], place) {
                            panic!("{gu}/{dn}: the one-token run of token {t}, slot {j}: {why}");
                        }
                    }
                    rows
                })
                .collect();

            for m in WIDE_RUNS {
                let mut bufs = LegBufs::for_downs(stream, WIDE_SHAPE, BOTH).expect("the buffers");
                let got = leg_rows(&gpu, &leg, &mut bufs, &stacks, &normed, &sel, m);
                let slots = m * SPC;
                let mut bad = Vec::new();
                for (s, &place) in sel_all[..slots].iter().enumerate() {
                    let (token, j) = (s / SPC, s % SPC);
                    let got_rows = &got[s * WIDE_N..(s + 1) * WIDE_N];
                    let want_rows = &oracle[token][j * WIDE_N..(j + 1) * WIDE_N];
                    let first_diff = (0..WIDE_N)
                        .find(|&r| got_rows[r].to_bits() != want_rows[r].to_bits())
                        .map(|r| {
                            format!(
                                "row {r}: {:#010x}, the one-token run's {:#010x}",
                                got_rows[r].to_bits(),
                                want_rows[r].to_bits()
                            )
                        });
                    let why = match slot_state(got_rows, place) {
                        Err(why) => Some(why.to_string()),
                        Ok(()) => first_diff,
                    };
                    if let Some(why) = why {
                        let at = match place {
                            HOST => "host".to_string(),
                            expert => format!("expert {expert}"),
                        };
                        bad.push(format!("slot {s} (column {token}, {at}): {why}"));
                    }
                }
                if got[slots * WIDE_N..].iter().any(|v| *v != SENT) {
                    bad.push("the run wrote past its slots".to_string());
                }
                let verdict = if bad.is_empty() { "ok" } else { "FAIL" };
                println!("leg_wide[gate·up {gu}, down {dn}, m = {m}] {verdict}");
                if let Some(first) = bad.first() {
                    failures.push(format!(
                        "gate·up {gu}, down {dn}, m = {m}: {} of {slots} slots; first: {first}",
                        bad.len()
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    /// The buffers of each activation form hold the bytes [`leg_bytes_for`]
    /// counts for their shape, and buffers of both forms the bytes
    /// [`kernels::leg_bytes`] counts for the pair: each form's columns beside
    /// one set of the rest.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_buffers_hold_the_counted_bytes_of_each_form() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let shape = SHAPE;
        for form in [DownAct::Q8Act, DownAct::Q8Blocks32] {
            let bufs = LegBufs::for_down(gpu.stream(), shape, form).expect("the buffers");
            assert_eq!(bufs.bytes(), leg_bytes_for(shape, form), "{form:?}");
        }
        let bufs = LegBufs::new(gpu.stream(), shape).expect("the buffers");
        assert_eq!(bufs.bytes(), leg_bytes(shape));
        let both = LegBufs::for_downs(gpu.stream(), shape, BOTH).expect("the buffers");
        let b32_cols = kernels::q8_blocks32_bytes((CAP * SPC) as u64, FF as u64);
        assert_eq!(both.bytes() as u64, leg_bytes(shape) as u64 + b32_cols);
        assert_eq!(
            both.bytes() as u64,
            kernels::leg_bytes(CAP as u64, SPC as u64, N as u64, FF as u64, BOTH)
        );
    }

    /// A stack routes by the GEMM entry the kernel table names for its type
    /// (`kernels::gemm`): a K-quant's or an i-quant's words through the
    /// grouped GEMM in the type that GEMM decodes (`GemmWeight::from_ggml`),
    /// the file's Q8_0, Q5_1 and IQ4_NL blocks, a Q5_0 stack's packed words and
    /// NVFP4's planes through the 32-value GEMM in their own weights; a stack
    /// in the other form, and a type the table names no GEMM for, has no route.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_stack_routes_by_the_tables_gemm_entry() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let qs = DeviceTensor::upload(stream, &[0u32; 4], 1, 4).expect("the words");
        let e = DeviceTensor::upload(stream, &[0u32], 1, 1).expect("the exponents");
        let scales = DeviceTensor::upload(stream, &[1.0f32], 1, 1).expect("the scales");
        let planes = StackW::Nvfp4(Nvfp4Stack {
            qs: &qs,
            e: &e,
            scales: &scales,
        });
        let words = StackW::Words(&qs);
        // PIN(2026-10-21): IQ2_XXS's words join the grouped routes with the
        // table's GEMM row for the type (`gemm_iq2xxs`).
        // PIN(2026-10-22): IQ3_S's words join them with its row (`gemm_iq3s`).
        for ty in [
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::IQ2_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
        ] {
            let want = GemmWeight::from_ggml(ty).expect("a type the grouped GEMM decodes");
            assert!(
                matches!(words.route(ty), Some(RouteW::Grouped(t, _)) if t == want),
                "{ty}"
            );
            assert!(planes.route(ty).is_none(), "{ty} in planes");
        }
        fn blocks32(st: StackW<'_>, ty: GgmlType) -> Option<Gemm32Weight<'_>> {
            match st.route(ty) {
                Some(RouteW::Blocks32(w)) => Some(w),
                Some(RouteW::Grouped(..)) | None => None,
            }
        }
        assert!(matches!(
            blocks32(words, GgmlType::Q8_0),
            Some(Gemm32Weight::Q8_0File(_))
        ));
        assert!(matches!(
            blocks32(words, GgmlType::Q5_1),
            Some(Gemm32Weight::Q5_1File(_))
        ));
        // PIN(2026-10-22): a Q5_0 stack's packed words join the 32-value
        // routes with the table's GEMM row for the type (`gemm_q5_0p`); the
        // file's blocks as words have no route.
        assert!(matches!(
            blocks32(StackW::Q5_0(&qs), GgmlType::Q5_0),
            Some(Gemm32Weight::Q5_0Packed(_))
        ));
        assert!(words.route(GgmlType::Q5_0).is_none(), "Q5_0 as words");
        assert!(matches!(
            blocks32(words, GgmlType::IQ4_NL),
            Some(Gemm32Weight::Iq4NlFile(_))
        ));
        assert!(matches!(
            blocks32(planes, GgmlType::NVFP4),
            Some(Gemm32Weight::Nvfp4Planes(_))
        ));
        for ty in [GgmlType::Q8_0, GgmlType::Q5_1, GgmlType::IQ4_NL] {
            assert!(planes.route(ty).is_none(), "{ty} in planes");
        }
        assert!(words.route(GgmlType::NVFP4).is_none(), "NVFP4 as words");
        for ty in [GgmlType::MXFP4, GgmlType::F32] {
            assert!(kernels::gemm(ty).is_none(), "{ty} has a GEMM row");
            assert!(words.route(ty).is_none(), "{ty}");
        }
    }

    /// A run whose down entry reads the other activation form than the
    /// buffers hold is refused by name before the first launch (the down's
    /// rows stay as they were), and [`CardLeg::enqueue_rows`] writes the rows
    /// [`CardLeg::enqueue`] writes into the caller's buffer, leaving the
    /// buffers' own as they were.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_refuses_the_other_form_and_writes_the_callers_rows() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let ty = GgmlType::Q4_K;
        let w = weights(&gpu, SHAPE, &[ty, GgmlType::Q5_K]);
        let names = [
            format!("gate_{ty}"),
            format!("up_{ty}"),
            format!("down_{ty}"),
        ];
        let stacks = LegStacks::of(&w, [&names[0], &names[1], &names[2]]).expect("the stacks");
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let shape = SHAPE;
        let run = LegRun {
            layer: 0,
            stacks: &stacks,
            normed: &normed,
            m: CAP,
            sel: &sel,
            n_card: N_CARD,
            rule: Act::SiluMul,
        };
        let sentinel = vec![SENT; CAP * SPC * N];

        let mut other = LegBufs::for_down(stream, shape, DownAct::Q8Blocks32).expect("buffers");
        other
            .down
            .copy_from_host(stream, &sentinel)
            .expect("sentinel");
        let refused = leg.enqueue(&gpu, &mut other, &run, None);
        assert!(
            matches!(&refused, Err(GpuError::Shape { what: WHAT, .. })),
            "a Q8Blocks32 buffer for a Q4_K down: {refused:?}"
        );
        let rows = other.down.to_host_vec(stream).expect("the rows");
        assert!(rows.iter().all(|v| *v == SENT), "a refused run wrote rows");

        let mut own = LegBufs::new(stream, shape).expect("buffers");
        own.down
            .copy_from_host(stream, &sentinel)
            .expect("sentinel");
        leg.enqueue(&gpu, &mut own, &run, None).expect("the run");
        let want = own.down.to_host_vec(stream).expect("the rows");
        let mut elsewhere = LegBufs::new(stream, shape).expect("buffers");
        elsewhere
            .down
            .copy_from_host(stream, &sentinel)
            .expect("sentinel");
        let mut mine = DeviceBuffer::from_host(stream, &sentinel).expect("the caller's rows");
        leg.enqueue_rows(&gpu, &mut elsewhere, &run, &mut mine)
            .expect("the run into the caller's rows");
        let got = mine.to_host_vec(stream).expect("the rows");
        assert!(
            gpu.take_fault().expect("the fault word").is_none(),
            "a fault"
        );
        assert!(
            got.iter()
                .zip(&want)
                .all(|(g, w)| g.to_bits() == w.to_bits()),
            "enqueue_rows wrote other rows than enqueue"
        );
        let own_rows = elsewhere.down.to_host_vec(stream).expect("the rows");
        assert!(
            own_rows.iter().all(|v| *v == SENT),
            "enqueue_rows wrote the buffers' own rows"
        );
    }

    /// The relu² tests' normed width and routed width: neither a multiple of
    /// 256 (84 and 58 blocks of 32), the second with a partial last group of
    /// four blocks in the quantizers.
    const R2_N: usize = 2688;
    /// See [`R2_N`].
    const R2_FF: usize = 1856;
    /// Slots a token carries in the relu² tests.
    const R2_SPC: usize = 6;
    /// Tokens the relu² buffers hold: 54 slot columns, more than a warp's 32.
    const R2_CAP: usize = 9;
    /// The shape of the relu² tests' buffers.
    const R2_SHAPE: LegShape = LegShape {
        cap: R2_CAP,
        slots_per_col: R2_SPC,
        n: R2_N,
        ff: R2_FF,
    };
    /// The tokens of the relu² runs: one, a few, and the buffers' cap.
    const R2_RUNS: [usize; 3] = [1, 4, R2_CAP];
    /// The relu² cases, `(up type, down type)`: the four 32-value types
    /// `kernels::card_sel32_runs` admits, each in both roles, with the mixes a
    /// real file holds (an IQ4_NL up under an IQ4_NL, a Q8_0 and a Q5_1 down;
    /// a Q5_0 up under a Q5_1 and a Q8_0 down).
    const R2_CASES: [(GgmlType, GgmlType); 8] = [
        (GgmlType::IQ4_NL, GgmlType::IQ4_NL),
        (GgmlType::IQ4_NL, GgmlType::Q8_0),
        (GgmlType::IQ4_NL, GgmlType::Q5_1),
        (GgmlType::Q8_0, GgmlType::Q8_0),
        (GgmlType::Q5_1, GgmlType::Q5_1),
        (GgmlType::Q5_0, GgmlType::Q5_0),
        (GgmlType::Q5_0, GgmlType::Q5_1),
        (GgmlType::Q5_0, GgmlType::Q8_0),
    ];

    /// The launches the relu² form is made of, run by hand on buffers of
    /// exactly the run's slots: the plain quantizer over the rows repeated once
    /// a slot, the up and down `_sel` entries of the type, `elem`'s `relu_sqr`
    /// and the select quantizer.
    struct Hand {
        q51: Q51SelKernels,
        q80: Q80SelKernels,
        iq: QuantSelKernels,
    }

    impl Hand {
        fn load(gpu: &Gpu) -> Hand {
            let (ctx, word) = (gpu.context(), gpu.fault_word());
            Hand {
                q51: Q51SelKernels::load(ctx, word).expect("the q5_1 module"),
                q80: Q80SelKernels::load(ctx, word).expect("the q8_0 module"),
                iq: QuantSelKernels::load(ctx, word).expect("the iq module"),
            }
        }

        /// The `_sel` of the 32-value type `ty` over `act`.
        fn sel32(
            &self,
            gpu: &Gpu,
            (ty, w): (GgmlType, &DeviceTensor<u32>),
            (act, sel): (&Q8Blocks32, &DeviceBuffer<u32>),
            (n_slots, rows_per_expert): (usize, usize),
            y: &mut DeviceBuffer<f32>,
        ) {
            let (stream, sink) = (gpu.stream(), gpu.layer_sink(0).expect("a sink"));
            match ty {
                GgmlType::Q5_0 => {
                    gpu.q5()
                        .enqueue_gemv_q5_0_sel(stream, w, act, sel, n_slots, rows_per_expert, y)
                }
                GgmlType::Q5_1 => {
                    let d = Q51SelDown {
                        w,
                        act,
                        sel,
                        n_slots,
                        rows_per_expert,
                    };
                    self.q51.enqueue_gemv_q5_1_sel(stream, &d, sink, y)
                }
                GgmlType::Q8_0 => {
                    let d = Q80SelDown {
                        w,
                        act,
                        sel,
                        n_slots,
                        rows_per_expert,
                    };
                    self.q80.enqueue_gemv_q8_0_sel32(stream, &d, sink, y)
                }
                GgmlType::IQ4_NL => {
                    let d = IqDown {
                        w,
                        act,
                        sel,
                        n_slots,
                        rows_per_expert,
                    };
                    self.iq.enqueue_down(stream, &d, sink, y)
                }
                other => panic!("no hand launch for {other}"),
            }
            .unwrap_or_else(|e| panic!("the hand {ty} launch: {e}"));
        }

        /// The first `m` tokens of `normed` (host values, `R2_N` a token)
        /// through the whole form: `(act_x, act_h, rows)`, the rows of the
        /// card slots written and the sentinel elsewhere.
        fn run(
            &self,
            gpu: &Gpu,
            w: &Weights,
            names: [&str; 2],
            (up_ty, dn_ty): (GgmlType, GgmlType),
            (normed, sel): (&[f32], &DeviceBuffer<u32>),
            m: usize,
        ) -> (Q8Blocks32, Q8Blocks32, Vec<f32>) {
            let (stream, sink) = (gpu.stream(), gpu.layer_sink(0).expect("a sink"));
            let slots = m * R2_SPC;
            let rep: Vec<f32> = (0..slots)
                .flat_map(|s| normed[(s / R2_SPC) * R2_N..][..R2_N].to_vec())
                .collect();
            let x = DeviceBuffer::from_host(stream, &rep).expect("the repeated rows");
            let mut act_x = Q8Blocks32::with_slots(stream, R2_N, slots).expect("act_x");
            gpu.q5()
                .enqueue_quantize_q8(stream, &x, &mut act_x, sink)
                .expect("the plain quantizer");
            let mut h = DeviceBuffer::<f32>::zeroed(stream, slots * R2_FF).expect("h");
            self.sel32(
                gpu,
                (up_ty, words(w, names[0])),
                (&act_x, sel),
                (slots, R2_FF),
                &mut h,
            );
            let mut sq = DeviceBuffer::<f32>::zeroed(stream, slots * R2_FF).expect("relu²");
            gpu.elem()
                .enqueue_relu_sqr(stream, &h, slots * R2_FF, &mut sq)
                .expect("relu_sqr");
            let mut act_h = Q8Blocks32::with_slots(stream, R2_FF, slots).expect("act_h");
            let q = QuantSel {
                x: &sq,
                cols: 0..slots,
                sel,
                n_card: N_CARD,
            };
            gpu.q5()
                .enqueue_quantize_q8_sel(stream, &q, &mut act_h, sink)
                .expect("the select quantizer");
            let mut down =
                DeviceBuffer::from_host(stream, &vec![SENT; slots * R2_N]).expect("the rows");
            self.sel32(
                gpu,
                (dn_ty, words(w, names[1])),
                (&act_h, sel),
                (slots, R2_N),
                &mut down,
            );
            let rows = down.to_host_vec(stream).expect("the hand rows");
            (act_x, act_h, rows)
        }
    }

    /// The first difference between two quantized activations over the card
    /// slots' columns, named by plane and column, or `None`.
    fn blocks32_diff(
        gpu: &Gpu,
        (got, want): (&Q8Blocks32, &Q8Blocks32),
        places: &[u32],
    ) -> Option<String> {
        let (g, w) = (
            got.readback(gpu.stream()).expect("readback"),
            want.readback(gpu.stream()).expect("readback"),
        );
        let (q_stride, k_blocks) = (got.q_stride(), got.k() / 32);
        for (c, &place) in places.iter().enumerate() {
            if place == HOST {
                continue;
            }
            let q = c * q_stride..(c + 1) * q_stride;
            let b = c * k_blocks..(c + 1) * k_blocks;
            if g.q[q.clone()] != w.q[q] {
                return Some(format!("column {c}: the codes differ"));
            }
            if g.s8[b.clone()] != w.s8[b.clone()] {
                return Some(format!("column {c}: the block sums differ"));
            }
            if let Some(i) = b.clone().find(|&i| g.d8[i].to_bits() != w.d8[i].to_bits()) {
                return Some(format!("column {c}: block {} scale differs", i - b.start));
            }
        }
        None
    }

    /// The relu² form against the launches it is made of, run by hand on
    /// buffers of exactly the run's slots, for every 32-value-block type of both
    /// roles (`R2_CASES`: [`kernels::card_sel32_runs`]'s; the NVFP4 planes' arms
    /// are `hw_the_leg_runs_the_nvfp4_arms_as_their_sel_launches`'s) at
    /// Lightning's widths and `R2_RUNS` tokens of six slots
    /// through buffers of `R2_CAP`: the q8_1 columns after the repeating
    /// quantizer and after the squared-ReLU select quantizer are the plain
    /// quantizer's of the repeated rows and the select quantizer's of
    /// `elem`'s `relu_sqr`, bit for bit; the down rows are the hand launches'
    /// bit for bit and, token by token, the rows a one-token run writes; a
    /// host slot keeps its sentinel; the card sum is the sum entry's over the
    /// leg's own rows; nothing past the run's slots is written.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_relu2_leg_runs_its_launches_as_the_hand_launches() {
        let ids = |tys: Vec<GgmlType>| {
            let mut ids: Vec<u32> = tys.into_iter().map(GgmlType::as_u32).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        let admitted = ids((0..64)
            .map(GgmlType::from_u32)
            .filter(|&ty| kernels::card_sel32_runs(ty))
            .collect());
        assert_eq!(
            ids(R2_CASES.iter().map(|c| c.0).collect()),
            admitted,
            "the cases' up types are not the types the relu² form admits"
        );
        assert_eq!(
            ids(R2_CASES.iter().map(|c| c.1).collect()),
            admitted,
            "the cases' down types are not the types the relu² form admits"
        );
        let sel_all: Vec<u32> = (0..R2_CAP).flat_map(places_of::<R2_SPC>).collect();
        for m in R2_RUNS {
            assert!(
                mixes(&sel_all[..m * R2_SPC]),
                "the places of {m} tokens must mix duplicates, every card expert and host slots"
            );
        }

        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (stream, sink) = (gpu.stream(), gpu.layer_sink(0).expect("a sink"));
        let leg = Relu2Leg::load(&gpu).expect("the leg");
        let hand = Hand::load(&gpu);
        let slots_kernels = CardSlotKernels::load(gpu.context()).expect("the sum entries");
        let mut tys: Vec<GgmlType> = R2_CASES.iter().flat_map(|&(u, d)| [u, d]).collect();
        tys.sort_by_key(|ty| ty.as_u32());
        tys.dedup();
        let w = weights_for(&gpu, R2_SHAPE, &tys, |role, ty| {
            role != "gate" && kernels::card_sel32_runs(ty)
        });
        let host_normed: Vec<f32> = (0..R2_CAP).flat_map(|t| row_of(t, R2_N)).collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &sel_all).expect("the places");
        let route: Vec<f32> = (0..R2_CAP * R2_SPC)
            .map(|i| 0.0625 + (i % 7) as f32 / 16.0)
            .collect();
        let route = DeviceBuffer::from_host(stream, &route).expect("the routing weights");
        let one = LegShape { cap: 1, ..R2_SHAPE };

        let mut failures = Vec::new();
        for (up_ty, dn_ty) in R2_CASES {
            let names = [stack_name("up", up_ty), stack_name("down", dn_ty)];
            let stacks = Relu2Stacks::of(&w, [&names[0], &names[1]]).expect("the stacks");
            stacks.check(N_CARD, R2_SHAPE).expect("the stacks' shape");
            let mut one_bufs = Relu2Bufs::new(stream, one).expect("one token's buffers");
            let oracle: Vec<Vec<f32>> = (0..R2_CAP)
                .map(|t| {
                    let row = DeviceBuffer::from_host(stream, &row_of(t, R2_N)).expect("a row");
                    let sel_t = DeviceBuffer::from_host(stream, &places_of::<R2_SPC>(t))
                        .expect("a token's places");
                    one_bufs
                        .down
                        .copy_from_host(stream, &vec![SENT; R2_SPC * R2_N])
                        .expect("the sentinel");
                    let run = Relu2Run {
                        layer: 0,
                        stacks: &stacks,
                        normed: &row,
                        m: 1,
                        sel: &sel_t,
                        n_card: N_CARD,
                    };
                    leg.enqueue(&gpu, &mut one_bufs, &run, None)
                        .unwrap_or_else(|e| panic!("{up_ty}/{dn_ty}: token {t}: {e}"));
                    one_bufs.down.to_host_vec(stream).expect("the rows")
                })
                .collect();

            for m in R2_RUNS {
                let slots = m * R2_SPC;
                let mut bufs = Relu2Bufs::new(stream, R2_SHAPE).expect("the buffers");
                bufs.down
                    .copy_from_host(stream, &vec![SENT; R2_CAP * R2_SPC * R2_N])
                    .expect("the sentinel");
                let mut acc = DeviceBuffer::from_host(stream, &vec![SENT; R2_CAP * R2_N])
                    .expect("the sum's rows");
                let run = Relu2Run {
                    layer: 0,
                    stacks: &stacks,
                    normed: &normed,
                    m,
                    sel: &sel,
                    n_card: N_CARD,
                };
                let sum = LegSum {
                    w: &route,
                    acc: &mut acc,
                };
                leg.enqueue(&gpu, &mut bufs, &run, Some(sum))
                    .unwrap_or_else(|e| panic!("{up_ty}/{dn_ty}, m = {m}: the leg: {e}"));
                let got = bufs.down.to_host_vec(stream).expect("the leg's rows");
                let got_acc = acc.to_host_vec(stream).expect("the leg's sum");
                let (hx, hh, want) = hand.run(
                    &gpu,
                    &w,
                    [&names[0], &names[1]],
                    (up_ty, dn_ty),
                    (&host_normed, &sel),
                    m,
                );
                let mut want_acc = DeviceBuffer::from_host(stream, &vec![SENT; R2_CAP * R2_N])
                    .expect("the sum's rows");
                let a = CardAcc {
                    down: &bufs.down,
                    w: &route,
                    sel: &sel,
                    n: R2_N,
                    m,
                    n_card: N_CARD,
                    n_used: R2_SPC,
                    w_pitch: R2_SPC,
                    fault: sink,
                };
                slots_kernels
                    .enqueue_card_acc(stream, &a, &mut want_acc)
                    .expect("the hand sum");
                let want_acc = want_acc.to_host_vec(stream).expect("the hand sum");
                let faulted = gpu.take_fault().expect("the fault word");

                let mut bad = Vec::new();
                if let Some(f) = faulted {
                    bad.push(format!("the run raised {f:?}"));
                }
                let places = &sel_all[..slots];
                if let Some(d) = blocks32_diff(&gpu, (&bufs.act_x, &hx), places) {
                    bad.push(format!("the repeated q8_1 columns: {d}"));
                }
                if let Some(d) = blocks32_diff(&gpu, (&bufs.act_h, &hh), places) {
                    bad.push(format!("the relu² q8_1 columns: {d}"));
                }
                for (s, &place) in places.iter().enumerate() {
                    let (token, j) = (s / R2_SPC, s % R2_SPC);
                    let rows = s * R2_N..(s + 1) * R2_N;
                    if let Err(why) = slot_state(&got[rows.clone()], place) {
                        bad.push(format!("slot {s} (token {token}): {why}"));
                        continue;
                    }
                    let one_token = &oracle[token][j * R2_N..(j + 1) * R2_N];
                    for (what, other) in [
                        ("the hand launches'", &want[rows.clone()]),
                        ("the one-token run's", one_token),
                    ] {
                        if let Some(r) = (0..R2_N).find(|&r| {
                            place != HOST && got[rows.start + r].to_bits() != other[r].to_bits()
                        }) {
                            bad.push(format!(
                                "slot {s} (token {token}) row {r}: {:#010x}, {what} {:#010x}",
                                got[rows.start + r].to_bits(),
                                other[r].to_bits()
                            ));
                        }
                    }
                }
                if got[slots * R2_N..].iter().any(|v| *v != SENT) {
                    bad.push("the run wrote past its slots".to_string());
                }
                let sum_rows = m * R2_N;
                if let Some(i) =
                    (0..sum_rows).find(|&i| got_acc[i].to_bits() != want_acc[i].to_bits())
                {
                    bad.push(format!(
                        "card sum value {i}: {:#010x}, the sum entry's {:#010x}",
                        got_acc[i].to_bits(),
                        want_acc[i].to_bits()
                    ));
                }
                if got_acc[..sum_rows]
                    .iter()
                    .any(|v| !v.is_finite() || *v == SENT)
                {
                    bad.push("a card sum value is no finite value the run wrote".to_string());
                }
                if got_acc[sum_rows..].iter().any(|v| *v != SENT) {
                    bad.push("the sum wrote past its tokens".to_string());
                }
                let verdict = if bad.is_empty() { "ok" } else { "FAIL" };
                println!("leg_relu2[up {up_ty}, down {dn_ty}, m = {m}] {verdict}");
                if let Some(first) = bad.first() {
                    failures.push(format!(
                        "up {up_ty}, down {dn_ty}, m = {m}: {} findings; first: {first}",
                        bad.len()
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    /// The relu² form's buffers hold the bytes [`relu2_leg_bytes`] counts, and
    /// its refusals are by name before the first launch: a stack of a type the
    /// form has no launch for, stacks of another shape, a run of no tokens or
    /// more than the buffers hold, a sum over seven slots, widths off the
    /// 32-value block. A NaN in a card slot's normed row raises `Q5Quant` from
    /// the repeating quantizer.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_relu2_leg_counts_its_bytes_and_refuses_by_name() {
        use crate::fault::FaultSite;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = Relu2Leg::load(&gpu).expect("the leg");
        for shape in [
            R2_SHAPE,
            LegShape {
                cap: 2,
                slots_per_col: 7,
                n: 96,
                ff: 32,
            },
        ] {
            let bufs = Relu2Bufs::new(stream, shape).expect("the buffers");
            assert_eq!(bufs.bytes(), relu2_leg_bytes(shape), "{shape:?}");
        }
        for (n, ff) in [(R2_N - 16, R2_FF), (R2_N, R2_FF + 8)] {
            let shape = LegShape { n, ff, ..R2_SHAPE };
            let refused = Relu2Bufs::new(stream, shape);
            assert!(
                matches!(&refused, Err(GpuError::Shape { .. })),
                "n = {n}, ff = {ff} are not multiples of 32: {:?}",
                refused.map(|_| ())
            );
        }

        // Q4_K is no 32-value type: refused when the stacks are read, in
        // either role; the 32-value types come through.
        let mixed = weights_for(
            &gpu,
            SHAPE,
            &[GgmlType::Q4_K, GgmlType::IQ4_NL],
            |role, ty| role != "gate" && (kernels::card_sel32_runs(ty) || ty == GgmlType::Q4_K),
        );
        let (k_up, k_down) = (
            stack_name("up", GgmlType::Q4_K),
            stack_name("down", GgmlType::Q4_K),
        );
        let (nl_up, nl_down) = (
            stack_name("up", GgmlType::IQ4_NL),
            stack_name("down", GgmlType::IQ4_NL),
        );
        for (role, names) in [("up", [&k_up, &nl_down]), ("down", [&nl_up, &k_down])] {
            let refused = Relu2Stacks::of(&mixed, [names[0], names[1]]);
            assert!(
                matches!(&refused, Err(GpuError::Tensor { what: WHAT, .. })),
                "a Q4_K {role} stack: {:?}",
                refused.map(|_| ())
            );
        }
        Relu2Stacks::of(&mixed, [&nl_up, &nl_down]).expect("IQ4_NL stacks");

        let tys = [GgmlType::IQ4_NL, GgmlType::Q8_0];
        let w = weights_for(&gpu, R2_SHAPE, &tys, |role, ty| {
            role != "gate" && kernels::card_sel32_runs(ty)
        });
        let names = [
            stack_name("up", GgmlType::IQ4_NL),
            stack_name("down", GgmlType::Q8_0),
        ];
        let stacks = Relu2Stacks::of(&w, [&names[0], &names[1]]).expect("the stacks");
        stacks.check(N_CARD, R2_SHAPE).expect("the stacks' shape");
        for (n_card, shape) in [
            (N_CARD + 1, R2_SHAPE),
            (
                N_CARD,
                LegShape {
                    ff: R2_FF - 32,
                    ..R2_SHAPE
                },
            ),
        ] {
            let refused = stacks.check(n_card, shape);
            assert!(
                matches!(&refused, Err(GpuError::Tensor { what: WHAT, .. })),
                "{n_card} experts of {shape:?}: {refused:?}"
            );
        }

        let host_normed: Vec<f32> = (0..R2_CAP).flat_map(|t| row_of(t, R2_N)).collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel_all: Vec<u32> = (0..R2_CAP).flat_map(places_of::<R2_SPC>).collect();
        let sel = DeviceBuffer::from_host(stream, &sel_all).expect("the places");
        let sentinel = vec![SENT; R2_CAP * R2_SPC * R2_N];
        let run = |m: usize| Relu2Run {
            layer: 0,
            stacks: &stacks,
            normed: &normed,
            m,
            sel: &sel,
            n_card: N_CARD,
        };
        let mut bufs = Relu2Bufs::new(stream, R2_SHAPE).expect("the buffers");
        bufs.down
            .copy_from_host(stream, &sentinel)
            .expect("the sentinel");
        for m in [0, R2_CAP + 1] {
            let refused = leg.enqueue(&gpu, &mut bufs, &run(m), None);
            assert!(
                matches!(&refused, Err(GpuError::Shape { what: WHAT, .. })),
                "a run of {m} tokens: {refused:?}"
            );
        }
        let route = DeviceBuffer::from_host(stream, &[0.5f32; 2 * 7]).expect("the weights");
        let mut acc = DeviceBuffer::<f32>::zeroed(stream, 2 * 96).expect("the sum");
        let seven = LegShape {
            cap: 2,
            slots_per_col: 7,
            n: 96,
            ff: 32,
        };
        let mut small = Relu2Bufs::new(stream, seven).expect("the buffers");
        let refused = leg.enqueue(
            &gpu,
            &mut small,
            &Relu2Run { m: 1, ..run(1) },
            Some(LegSum {
                w: &route,
                acc: &mut acc,
            }),
        );
        assert!(
            matches!(&refused, Err(GpuError::Shape { what: WHAT, .. })),
            "a sum over seven slots: {refused:?}"
        );
        assert!(
            bufs.down
                .to_host_vec(stream)
                .expect("the rows")
                .iter()
                .all(|v| *v == SENT),
            "a refused run wrote rows"
        );
        assert!(
            gpu.take_fault().expect("the fault word").is_none(),
            "a refusal raised a fault"
        );

        // A NaN in token 1's row: the repeating quantizer refuses the blocks
        // of its card slots' columns and raises Q5Quant.
        let mut nan_normed = host_normed;
        nan_normed[R2_N + 5] = f32::NAN;
        let nan_normed = DeviceBuffer::from_host(stream, &nan_normed).expect("the rows");
        leg.enqueue(
            &gpu,
            &mut bufs,
            &Relu2Run {
                normed: &nan_normed,
                ..run(2)
            },
            None,
        )
        .expect("the run");
        let fault = gpu.take_fault().expect("the fault word");
        assert!(
            fault
                .as_ref()
                .is_some_and(|f| f.sites & (1 << FaultSite::Q5Quant as u32) != 0 && f.layer == 0),
            "a NaN normed row raised {fault:?}"
        );
    }

    /// The arena entries' cases, `(gate·up type, down type)`: each arena
    /// entry runs in one, in GLM's pairs.
    const AT_CASES: [(GgmlType, GgmlType); 2] = [
        (GgmlType::Q4_K, GgmlType::Q5_K),
        (GgmlType::Q5_K, GgmlType::Q6_K),
    ];

    /// The arena entries against the stack `_sel` launches of their format:
    /// each expert's three parts copied as their file bytes into pinned host
    /// memory the card maps, out of stack order and each at its own word
    /// skew, and a table naming them by slot. The gate·up rows and, from the
    /// stack run's q8_1 columns, the down rows are the stack launches' bit
    /// for bit; a skipped slot keeps what it held. A part one word past the
    /// arena raises `ExpertId`: the gate·up writes NaN, the down leaves the
    /// slot alone.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_an_arena_part_computes_as_its_stack_rows() {
        use crate::fault::FaultSite;
        use crate::graph::MappedHost;
        use crate::kquant::{
            ARENA_PARTS, ARENA_SKIP, GateUpAt, PART_DOWN, PART_GATE, PART_UP, SelDownAt,
        };
        use crate::q6k_sel::Q6kSelKernels;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (stream, ctx) = (gpu.stream(), gpu.context());
        let kq = QuantSelKernels::load(ctx, gpu.fault_word()).expect("the kquant module");
        let q6 = Q6kSelKernels::load(ctx, gpu.fault_word()).expect("the q6k module");
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let slots = CAP * SPC;
        let rule = Act::SwigluClamp { limit: 0.25 };
        let sink = gpu.layer_sink(0).expect("a sink");
        let mut act_x = Q8Act::with_tier_cols(stream, CAP, N).expect("act_x");
        gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, CAP, 0)
            .expect("q8_1");
        for (gty, dty) in AT_CASES {
            let case = format!("{gty} gate·up, {dty} down");
            // Per part: the stack's file bytes, rows a part, and the rows' k.
            let roles = [(gty, FF, N, 1u64), (gty, FF, N, 2), (dty, N, FF, 3)];
            let bytes: Vec<Vec<u8>> = roles
                .iter()
                .map(|&(ty, rows, k, seed)| stack_bytes(ty, N_CARD * rows, k, seed << 32 | 0x51))
                .collect();
            let stacks: Vec<DeviceTensor<u32>> = roles
                .iter()
                .zip(&bytes)
                .map(|(&(_, rows, _, _), b)| {
                    let words = words_of(b, N_CARD * rows);
                    let n = words.len();
                    DeviceTensor::upload(stream, &words, N_CARD * rows, n / (N_CARD * rows))
                        .expect("a stack")
                })
                .collect();
            let part_len: Vec<usize> = bytes.iter().map(|b| b.len() / N_CARD).collect();
            // Every expert's parts, down first, each 1..=11 words past the
            // previous one's end.
            let mut offs_of = [[0u64; ARENA_PARTS]; N_CARD];
            let mut cursor = 3usize;
            for (i, e) in [2usize, 0, 3, 1].into_iter().enumerate() {
                for p in [PART_DOWN, PART_GATE, PART_UP] {
                    offs_of[e][p] = cursor as u64;
                    cursor += part_len[p].div_ceil(4) + 1 + (5 * i + 3 * p) % 11;
                }
            }
            let arena_words = cursor;
            let widest = part_len.iter().max().expect("parts").div_ceil(4);
            let page =
                MappedHost::read_in_place(ctx, 4 * (arena_words + widest), "the test's arena")
                    .expect("pinned host memory the card maps");
            for (e, offs) in offs_of.iter().enumerate() {
                for p in 0..ARENA_PARTS {
                    let src = &bytes[p][e * part_len[p]..(e + 1) * part_len[p]];
                    // SAFETY: the part's bytes lie inside the allocation (the
                    // cursor kept every part below arena_words words), and no
                    // launch reads the page yet.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr(),
                            page.host_at(4 * offs[p] as usize),
                            src.len(),
                        );
                    }
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            // SAFETY: arena_words words of the live allocation the card maps,
            // which outlives the window.
            let arena = unsafe { crate::tensor::window::<u32>(page.dev_at(0), arena_words, ctx) };
            let table = |past: Option<usize>| -> Vec<u64> {
                let mut t = Vec::with_capacity(slots * ARENA_PARTS);
                for (s, &id) in SEL.iter().enumerate() {
                    if id == HOST {
                        t.extend([ARENA_SKIP; ARENA_PARTS]);
                        continue;
                    }
                    let mut o = offs_of[id as usize];
                    if past == Some(s) {
                        for p in 0..ARENA_PARTS {
                            o[p] = (arena_words + 1 - part_len[p].div_ceil(4)) as u64;
                        }
                    }
                    t.extend(o);
                }
                t
            };
            let offs = DeviceBuffer::from_host(stream, &table(None)).expect("the table");

            let mut h_ref = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("h");
            let mut h_at = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("h");
            let a = GateUpAct {
                wg: &stacks[PART_GATE],
                wu: &stacks[PART_UP],
                act: &act_x,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: FF,
                slots_per_col: SPC,
                rule,
            };
            let at = GateUpAt {
                arena: &arena,
                offs: &offs,
                act: &act_x,
                n_slots: slots,
                rows_per_expert: FF,
                slots_per_col: SPC,
                rule,
            };
            match gty {
                GgmlType::Q4_K => kq
                    .enqueue_gate_up_q4k(stream, &a, sink, &mut h_ref)
                    .and_then(|()| kq.enqueue_gate_up_q4k_at(stream, &at, sink, &mut h_at)),
                _ => kq
                    .enqueue_gate_up_q5k(stream, &a, sink, &mut h_ref)
                    .and_then(|()| kq.enqueue_gate_up_q5k_at(stream, &at, sink, &mut h_at)),
            }
            .unwrap_or_else(|e| panic!("{case}: the gate·up: {e}"));
            let mut act_h = Q8Act::with_slots(stream, slots, FF).expect("act_h");
            let q = QuantSel {
                x: &h_ref,
                cols: 0..slots,
                sel: &sel,
                n_card: N_CARD,
            };
            gpu.q4k_sel()
                .enqueue_quantize_sel(stream, &q, sink, &mut act_h)
                .unwrap_or_else(|e| panic!("{case}: the quantizer: {e}"));
            let mut y_ref = DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("y");
            let mut y_at = DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("y");
            let d_at = SelDownAt {
                arena: &arena,
                offs: &offs,
                act: &act_h,
                n_slots: slots,
                rows_per_expert: N,
            };
            let down_at = |d: &SelDownAt<'_>, y: &mut DeviceBuffer<f32>| match dty {
                GgmlType::Q5_K => kq.enqueue_gemv_q5k_sel_at(stream, d, sink, y),
                _ => q6.enqueue_gemv_q6k_sel_at(stream, d, sink, y),
            };
            match dty {
                GgmlType::Q5_K => {
                    let d = SelDown {
                        w: &stacks[PART_DOWN],
                        act: &act_h,
                        sel: &sel,
                        n_slots: slots,
                        rows_per_expert: N,
                    };
                    kq.enqueue_gemv_q5k_sel(stream, &d, sink, &mut y_ref)
                }
                _ => q6.enqueue_gemv_q6k_sel(
                    stream,
                    &stacks[PART_DOWN],
                    &act_h,
                    &sel,
                    slots,
                    N,
                    &mut y_ref,
                ),
            }
            .and_then(|()| down_at(&d_at, &mut y_at))
            .unwrap_or_else(|e| panic!("{case}: the down: {e}"));
            let (h_ref, h_got) = (
                h_ref.to_host_vec(stream).expect("h"),
                h_at.to_host_vec(stream).expect("h"),
            );
            let (y_ref, y_got) = (
                y_ref.to_host_vec(stream).expect("y"),
                y_at.to_host_vec(stream).expect("y"),
            );
            assert!(
                gpu.take_fault().expect("the fault word").is_none(),
                "{case}: a fault"
            );
            for (what, rpe, want, got) in [("h", FF, &h_ref, &h_got), ("down", N, &y_ref, &y_got)] {
                for (s, &id) in SEL.iter().enumerate() {
                    let rows = s * rpe..(s + 1) * rpe;
                    if id == HOST {
                        assert!(
                            got[rows].iter().all(|v| *v == SENT),
                            "{case}: {what} of slot {s} (skipped) was written"
                        );
                        continue;
                    }
                    assert!(
                        got[rows.clone()]
                            .iter()
                            .all(|v| v.is_finite() && *v != SENT)
                            && got[rows.clone()].iter().any(|v| *v != 0.0),
                        "{case}: {what} of slot {s} (expert {id}) holds no finite rows the \
                         entry wrote"
                    );
                    if let Some(r) = rows
                        .clone()
                        .find(|&r| got[r].to_bits() != want[r].to_bits())
                    {
                        panic!(
                            "{case}: {what} of slot {s} (expert {id}) row {}: arena {:#010x}, \
                             stack {:#010x}",
                            r - rows.start,
                            got[r].to_bits(),
                            want[r].to_bits()
                        );
                    }
                }
            }

            let past = DeviceBuffer::from_host(stream, &table(Some(0))).expect("the table");
            let mut h = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("h");
            let at = GateUpAt { offs: &past, ..at };
            match gty {
                GgmlType::Q4_K => kq.enqueue_gate_up_q4k_at(stream, &at, sink, &mut h),
                _ => kq.enqueue_gate_up_q5k_at(stream, &at, sink, &mut h),
            }
            .unwrap_or_else(|e| panic!("{case}: the gate·up past the arena: {e}"));
            let h = h.to_host_vec(stream).expect("h");
            let fault = gpu.take_fault().expect("the fault word");
            assert!(
                fault
                    .as_ref()
                    .is_some_and(|f| f.code == FaultSite::ExpertId as u32 && f.layer == 0),
                "{case}: a gate·up part past the arena raised {fault:?}"
            );
            assert!(
                h[..FF].iter().all(|v| v.is_nan()),
                "{case}: a gate·up part past the arena left rows that are not NaN"
            );
            let mut y = DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("y");
            down_at(
                &SelDownAt {
                    offs: &past,
                    ..d_at
                },
                &mut y,
            )
            .unwrap_or_else(|e| panic!("{case}: the down past the arena: {e}"));
            let y = y.to_host_vec(stream).expect("y");
            let fault = gpu.take_fault().expect("the fault word");
            assert!(
                fault
                    .as_ref()
                    .is_some_and(|f| f.code == FaultSite::ExpertId as u32 && f.layer == 0),
                "{case}: a down part past the arena raised {fault:?}"
            );
            assert!(
                y[..N].iter().all(|v| *v == SENT),
                "{case}: a down part past the arena wrote its slot"
            );
            // The window frees nothing; the page goes after every launch
            // that read it drained (the reads above).
            drop(std::mem::ManuallyDrop::into_inner(arena).into_raw_parts());
            drop(page);
        }
    }
}
