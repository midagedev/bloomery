//! The stage card's join of a tier layer, one owner for every family whose card sum over both
//! cards runs on `ds41_ffn_card_acc_tier` ([`FfnBatchKernels::enqueue_card_acc_tier`]): the
//! shared-routed families ([`crate::shared_routed`]), GLM-5.3 and MiMo-V2. A tier layer's join is
//! one weighted sum over the stage card's and the expert tier's slots in slot order, then the
//! layer's base added (the family's: `x1`, `x1` plus the shared expert's output, or the shared
//! expert's output alone): the card sum plus the base of a one-card layer over the union of the
//! two cards' experts, bit for bit.
//!
//! - **Step** ([`enqueue_step_join`]): the wait for the host and the tier of the page's row, the
//!   join over the tier's rows of that row, then the host's sum added.
//! - **Prompt batch** ([`enqueue_join`]): the join of a unit's rows once the batch port hands the
//!   tier's rows over (inside [`bloomery_gpu::host::BatchLeg::join_tiered`], which uploads the
//!   host sums after it). The next unit's front routes over the shared weights and tier places
//!   before this unit's join, so the batch keeps copies of them ([`TierBatch::keep`]).
//!
//! What stays with the family is where its slots' rows, places and weights live
//! ([`TierJoin`], [`TierSlots`]) and what its base is.

use bloomery_gpu::DeviceBufferRaw;
use std::mem::size_of;

use bloomery_gpu::card_experts::TierJoin;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};

use crate::chain::ffn::{CardAccTier, FfnBatchKernels};
use crate::span::{span, span_mut};

/// What the join's errors name.
const WHAT: &str = "tier join";

/// The routing side of a tier layer's join, `n_used` slots a token, `n` wide: the slots' routing
/// weights `w` (`n_used` a token) and their tier places `tsel` (the tier's below `n_tier`), and
/// the layer's experts on the stage card (`n_card`, the places of [`TierJoin::sel`] below it).
pub struct TierSlots<'a> {
    pub w: &'a DeviceBuffer<f32>,
    pub tsel: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub n_card: usize,
    pub n_tier: usize,
    pub n_used: usize,
}

/// A tier layer's join over `m` tokens once the tier's rows `trows` have landed: the card sum over
/// both cards' slots in slot order (`join`'s stage card rows at its places, `slots`' weights and
/// tier places) into `join.acc`, then `join.acc + base` into `join.pre` (`m` rows of `slots.n`).
/// Two launches on `stream`; a slot count the sum has no entry for is refused by name before the
/// first. Asynchronous, allocation-free, capturable.
pub fn enqueue_join(
    gpu: &Gpu,
    (stream, kernels): (&CudaStream, &FfnBatchKernels),
    (join, slots): (&mut TierJoin<'_>, &TierSlots<'_>),
    (trows, m): (&DeviceBuffer<f32>, usize),
    base: &DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    enqueue_acc((stream, kernels), (&mut *join, slots), (trows, m))?;
    gpu.elem()
        .enqueue_add(stream, join.acc, base, m * slots.n, join.pre)
}

/// Sum both cards' routed slots without adding a base. A caller whose shared output and residual
/// join after the complete routed sum uses this result directly.
pub fn enqueue_acc(
    (stream, kernels): (&CudaStream, &FfnBatchKernels),
    (join, slots): (&mut TierJoin<'_>, &TierSlots<'_>),
    (trows, m): (&DeviceBuffer<f32>, usize),
) -> Result<(), GpuError> {
    let a = CardAccTier {
        down: join.down,
        trows,
        w: slots.w,
        sel: join.sel,
        tsel: slots.tsel,
        n: slots.n,
        m,
        n_card: slots.n_card,
        n_tier: slots.n_tier,
        n_used: slots.n_used,
    };
    kernels.enqueue_card_acc_tier(stream, &a, join.acc)
}

/// Layer `l`'s back of the step on a tier layer at the page's row `row`: the wait for the host and
/// the tier ([`Hybrid::enqueue_tier_back`]), the row's join ([`enqueue_join`] over the tier's rows
/// of `row`), then the host's sum of `row` plus `join.pre` into `dest`. Three launches after the
/// wait, on `gpu`'s stream. Asynchronous, allocation-free, capturable.
pub fn enqueue_step_join(
    gpu: &Gpu,
    kernels: &FfnBatchKernels,
    (hybrid, l, row): (&Hybrid<HostRun>, usize, usize),
    (mut join, slots): (TierJoin<'_>, &TierSlots<'_>),
    (base, dest): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    hybrid.enqueue_tier_back(stream, l, row)?;
    let trows = hybrid.tier_rows(row)?;
    enqueue_join(gpu, (stream, kernels), (&mut join, slots), (trows, 1), base)?;
    gpu.elem().enqueue_add(
        stream,
        hybrid.boundary().hsum_of(row)?,
        join.pre,
        slots.n,
        dest,
    )
}

/// A prompt batch's tier buffers for up to `cap` tokens of `n_used` slots: each slot's tier place
/// as the route's download carries it ([`TierBatch::places_mut`]), and the copies of the slots'
/// weights and tier places the join reads ([`TierBatch::keep`]).
pub struct TierBatch {
    places: DeviceBuffer<u32>,
    join_places: DeviceBuffer<u32>,
    join_weights: DeviceBuffer<f32>,
}

impl TierBatch {
    /// The buffers for batches of up to `cap` tokens of `n_used` slots. Load-time or first-prompt
    /// only.
    pub fn new(stream: &CudaStream, cap: usize, n_used: usize) -> Result<TierBatch, GpuError> {
        Ok(TierBatch {
            places: DeviceBuffer::zeroed(stream, cap * n_used)?,
            join_places: DeviceBuffer::zeroed(stream, cap * n_used)?,
            join_weights: DeviceBuffer::zeroed(stream, cap * n_used)?,
        })
    }

    /// Device bytes [`TierBatch::new`] makes for `cap` tokens of `n_used` slots: three words a
    /// slot.
    #[must_use]
    pub const fn bytes_of(cap: usize, n_used: usize) -> usize {
        3 * size_of::<u32>() * cap * n_used
    }

    /// Device bytes the buffers hold.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.places.num_bytes() + self.join_places.num_bytes() + self.join_weights.num_bytes()
    }

    /// Each slot's tier place, which the route writes and the download carries.
    #[must_use]
    pub fn places(&self) -> &DeviceBuffer<u32> {
        &self.places
    }

    /// [`TierBatch::places`], to write.
    pub fn places_mut(&mut self) -> &mut DeviceBuffer<u32> {
        &mut self.places
    }

    /// The copies of a unit's first `slots` slots' `weights` and tier places the join reads,
    /// after the unit's shadow and before the next unit's front routes over the shared ones. Two
    /// copies on `stream`; a run past either buffer is refused by name. Asynchronous.
    pub fn keep(
        &mut self,
        stream: &CudaStream,
        slots: usize,
        weights: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        span_mut(WHAT, &mut self.join_weights, 0, slots)?
            .copy_from_device_raw(&*span(WHAT, weights, 0, slots)?, stream)?;
        span_mut(WHAT, &mut self.join_places, 0, slots)?
            .copy_from_device_raw(&*span(WHAT, &self.places, 0, slots)?, stream)?;
        Ok(())
    }

    /// A unit's routing side of the join: the weights and tier places [`TierBatch::keep`] kept,
    /// `n` wide, `n_card` and `n_tier` the layer's experts on each card, `n_used` slots a token.
    #[must_use]
    pub fn slots(
        &self,
        n: usize,
        (n_card, n_tier, n_used): (usize, usize, usize),
    ) -> TierSlots<'_> {
        TierSlots {
            w: &self.join_weights,
            tsel: &self.join_places,
            n,
            n_card,
            n_tier,
            n_used,
        }
    }
}
