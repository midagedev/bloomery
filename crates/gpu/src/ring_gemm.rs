//! The ring pass's route GEMMs: the host experts the expert stream landed in a half of its ring
//! ([`RingLayer`]) run batch by batch as one [`RouteBlock`] over the half's stacks, each batch
//! behind its own landed event, the down rows written into the caller's slot rows beside its card
//! slots'. One owner for every family whose card route runs the ring: the buffers are made in the
//! forms the stacks' route GEMMs read ([`RingForms`], from their file types through the kernel
//! table), and their bytes are the plan's formula ([`ring_pass_bytes_in`]).

use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::place_hybrid::{RingForms, ring_pass_bytes_in};

use crate::card_leg::ExpertInput;
use crate::card_leg::{RouteBlock, RouteIo};
use crate::expert_input::Protection;
use crate::gemm::{Gemm32Kernels, GemmAct, GemmAct32, GemmKernels, GemmRoute};
use crate::host::xstream::LandBatch;
use crate::kquant::Act;
use crate::{FaultSink, Gpu, GpuError};

const WHAT: &str = "ring_gemm::RingGemm";

/// One form's rows: q8 blocks of 32 or the grouped GEMM's q8_1 blocks of 128, by the stack's type.
pub(crate) enum Rows {
    Blocks32(Box<GemmAct32>),
    Q128(Box<GemmAct>),
}

impl Rows {
    /// `cols` rows of `k` values in the 32-block form when `blocks32`, else the 128-block one.
    pub(crate) fn new(
        stream: &CudaStream,
        cols: usize,
        k: usize,
        blocks32: bool,
    ) -> Result<Rows, GpuError> {
        Ok(if blocks32 {
            Rows::Blocks32(Box::new(GemmAct32::new(stream, cols, k)?))
        } else {
            Rows::Q128(Box::new(GemmAct::new(stream, cols, k)?))
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        match self {
            Rows::Blocks32(a) => a.bytes(),
            Rows::Q128(a) => a.bytes(),
        }
    }

    /// The rows as the route block reads them: the 128-block form or the 32-block one.
    pub(crate) fn parts(&self) -> (Option<&GemmAct>, Option<&GemmAct32>) {
        match self {
            Rows::Q128(a) => (Some(&**a), None),
            Rows::Blocks32(a) => (None, Some(&**a)),
        }
    }

    /// The rows as the route block writes them.
    pub(crate) fn parts_mut(&mut self) -> (Option<&mut GemmAct>, Option<&mut GemmAct32>) {
        match self {
            Rows::Q128(a) => (Some(&mut **a), None),
            Rows::Blocks32(a) => (None, Some(&mut **a)),
        }
    }

    /// The `m` rows of `x` quantized into this form. Asynchronous on the caller's stream.
    pub(crate) fn enqueue_quantize(
        &mut self,
        gpu: &Gpu,
        g32: &Gemm32Kernels,
        x: &DeviceBuffer<f32>,
        m: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        match self {
            Rows::Blocks32(a) => g32.enqueue_quantize_gemm32(gpu.stream(), x, m, a, fault),
            Rows::Q128(a) => gpu.enqueue_quantize_gemm(x, m, a, fault),
        }
    }
}

/// A prompt batch's ring pass buffers of up to `cap` rows over a half of the ring: the batch's
/// normed rows in the gate's and up's form, every slot's gate and up rows, their SwiGLU in the
/// down's form and the route table of the slots over the half.
pub(crate) struct RingGemm {
    x: Rows,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    h: Rows,
    route: GemmRoute,
    cap: usize,
    slots: usize,
    protection: Option<Protection>,
}

impl RingGemm {
    /// The buffers for batches of up to `cap` rows of `n` values, `n_used` slots a row and `ff`
    /// values a slot's gate and up row, over a half of `half_slots` slots, in `forms`. Refused by
    /// name: buffers whose made bytes are not [`ring_pass_bytes_in`] of the same widths.
    /// Load-time or first-prompt only.
    pub(crate) fn new(
        gpu: &Gpu,
        (cap, n_used, n, ff): (usize, usize, usize, usize),
        half_slots: usize,
        forms: RingForms,
    ) -> Result<RingGemm, GpuError> {
        let stream = gpu.stream();
        let slots = cap * n_used;
        let ring = RingGemm {
            x: Rows::new(stream, cap, n, forms.gate_up32)?,
            g: DeviceBuffer::zeroed(stream, slots * ff)?,
            u: DeviceBuffer::zeroed(stream, slots * ff)?,
            h: Rows::new(stream, slots, ff, forms.down32)?,
            route: GemmRoute::new(stream, slots, half_slots)?,
            cap,
            slots,
            protection: None,
        };
        let want = ring_pass_bytes_in(cap, n_used, n, ff, half_slots, forms);
        if ring.bytes() != want {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the ring pass's buffers made {} bytes; the plan reserves {want} \
                     (place_hybrid::ring_pass_bytes_in)",
                    ring.bytes()
                ),
            ));
        }
        Ok(ring)
    }

    pub(crate) fn set_expert_input(
        &mut self,
        gpu: &Gpu,
        policy: ExpertInput,
    ) -> Result<(), GpuError> {
        match policy {
            ExpertInput::Direct => self.protection = None,
            ExpertInput::AmaxF16 if self.protection.is_none() => {
                self.protection = Some(Protection::new(gpu, self.slots)?);
            }
            ExpertInput::AmaxF16 => {}
        }
        Ok(())
    }

    /// Device bytes of the buffers.
    pub(crate) fn bytes(&self) -> usize {
        self.x.bytes()
            + self.g.num_bytes()
            + self.u.num_bytes()
            + self.h.bytes()
            + self.route.bytes()
            + self.protection.as_ref().map_or(0, Protection::bytes)
    }

    /// The slots of the half the buffers were made for.
    pub(crate) fn half_slots(&self) -> usize {
        self.route.n_experts()
    }

    /// The batch's `m` normed rows `x` in the gate's and up's form, once a layer before its
    /// batches. Refused by name: more rows than the buffers hold.
    pub(crate) fn enqueue_rows(
        &mut self,
        gpu: &Gpu,
        g32: &Gemm32Kernels,
        x: &DeviceBuffer<f32>,
        m: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        if m == 0 || m > self.cap {
            return Err(GpuError::shape(
                WHAT,
                format!("a batch of {m} rows in buffers of {}", self.cap),
            ));
        }
        self.x.enqueue_quantize(gpu, g32, x, m, fault)
    }

    /// Landing batch `batch` of a half: its route table from the pass's routed `ids` (`n_slots`
    /// of them, `top_k` a row) through the batch's map row over `n_expert` experts — a slot whose
    /// expert the batch does not hold is in no tile, its rows neither read nor written — then,
    /// behind the batch's landed event, `blk` over that table under the gate·up rule `rule`, each
    /// listed slot's down row into its row of `y`. `gemm` is `None` for a caller with no grouped
    /// GEMM kernels: a block the grouped GEMM reads is then refused by name
    /// ([`RouteBlock::enqueue_rule`]), as is one whose stacks read a form the buffers were not
    /// made in. Asynchronous, allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the two GEMM kernel sets, the block, the batch, the routed slots, the \
                  rows and their rule, and the fault sink (rust-quality R8)"
    )]
    pub(crate) fn enqueue_batch(
        &mut self,
        gpu: &Gpu,
        (gemm, g32): (Option<&GemmKernels>, &Gemm32Kernels),
        blk: &RouteBlock<'_>,
        batch: &LandBatch,
        (ids, n_slots, top_k, n_expert): (&DeviceBuffer<u32>, usize, usize, usize),
        (y, rule): (&mut DeviceBuffer<f32>, Act),
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        batch.with_map(gpu.context(), n_expert, |row| {
            g32.enqueue_route_remap(stream, ids, row, n_slots, &mut self.route, fault)
        })?;
        if let Some(e) = batch.event.as_deref() {
            stream.wait(e)?;
        }
        let RingGemm {
            x,
            g,
            u,
            h,
            route,
            protection,
            ..
        } = self;
        run_block(
            (x, g, u, h, protection.as_mut()),
            gpu,
            (gemm, g32),
            blk,
            (&*route, top_k),
            n_slots,
            (y, rule),
            fault,
        )
    }

    /// `blk` over a route table `table` the caller filled (a stack of its own: the layer's card
    /// stacks, whose table is over their `n_card` experts, not the half's) through the buffers'
    /// rows: the same gate, up, SwiGLU and down launches as a landing batch's, each listed slot's
    /// down row into its row of `y`, under the buffers' expert-input protection. The batch's rows
    /// ([`RingGemm::enqueue_rows`]) are quantized first. Refused by name as
    /// [`RingGemm::enqueue_batch`] refuses a block the buffers were not made for, and a table over
    /// more slots than the buffers hold. Asynchronous, allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, the two GEMM kernel sets, the block, the filled table and its slots, the \
                  rows and their rule, and the fault sink (rust-quality R8)"
    )]
    pub(crate) fn enqueue_over(
        &mut self,
        gpu: &Gpu,
        (gemm, g32): (Option<&GemmKernels>, &Gemm32Kernels),
        blk: &RouteBlock<'_>,
        (table, n_slots, top_k): (&GemmRoute, usize, usize),
        (y, rule): (&mut DeviceBuffer<f32>, Act),
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        if n_slots == 0 || n_slots > self.slots {
            return Err(GpuError::shape(
                WHAT,
                format!("{n_slots} slots in buffers of {}", self.slots),
            ));
        }
        let RingGemm {
            x,
            g,
            u,
            h,
            protection,
            ..
        } = self;
        run_block(
            (x, g, u, h, protection.as_mut()),
            gpu,
            (gemm, g32),
            blk,
            (table, top_k),
            n_slots,
            (y, rule),
            fault,
        )
    }
}

/// The routed block `blk` over `table` through the buffers' rows `(x, g, u, h)`, the down rows into
/// `y`: the one launch sequence of a landing batch and of a caller's own table.
#[allow(
    clippy::too_many_arguments,
    reason = "the buffers' rows, the card, the two GEMM kernel sets, the block, the table, the \
              slots, the rows and their rule, and the fault sink (rust-quality R8)"
)]
fn run_block(
    (x, g, u, h, protection): (
        &Rows,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut Rows,
        Option<&mut Protection>,
    ),
    gpu: &Gpu,
    (gemm, g32): (Option<&GemmKernels>, &Gemm32Kernels),
    blk: &RouteBlock<'_>,
    (table, top_k): (&GemmRoute, usize),
    n_slots: usize,
    (y, rule): (&mut DeviceBuffer<f32>, Act),
    fault: FaultSink,
) -> Result<(), GpuError> {
    let (x128, x32) = x.parts();
    let (h128, h32) = h.parts_mut();
    blk.enqueue_input(
        gpu,
        (gemm, g32),
        (table, top_k),
        n_slots,
        (
            RouteIo {
                x128,
                x32,
                g,
                u,
                h128,
                h32,
                y,
            },
            rule,
        ),
        protection,
        fault,
    )
}
