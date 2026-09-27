//! The family's layer program ([`Program`]): the chain and the pass of both
//! bodies as one walk of [`runtime::sched::walk`] over one unit of `m`
//! columns, `(1, m, Step)` — the decode step at `m = 1`, a pass at its `m`.
//! A layer has no host leg, so its front is the whole layer
//! ([`dispatch::layer`]) and its shadow and back are empty; its port
//! ([`NoLeg`]) exchanges nothing and refuses by name a point the arena
//! cannot hold. What follows the last layer is the walk's [`Tail`].

use super::body::{Kernels, TapRows};
use super::dispatch::{self, Ctx, PassCtx};
use super::head_argmax::HeadArgmaxState;
use super::scratch::{Arena, Io, KvPlanes, LayerStore, StoreMut, f32_view};
use crate::head::Head;
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use runtime::sched::{self, At, LayerProgram, Overlap, Port, PortKind, Refused};

const WHAT: &str = "qwen3moe::Program";

/// A body's layer stores, one per layer.
pub(super) trait Stores {
    /// Layer `l`'s store as its enqueue borrows it; `None` past the last.
    fn store(&mut self, l: usize) -> Option<StoreMut<'_>>;
}

impl Stores for [KvPlanes] {
    fn store(&mut self, l: usize) -> Option<StoreMut<'_>> {
        self.get_mut(l).map(StoreMut::Kv)
    }
}

impl Stores for [LayerStore] {
    fn store(&mut self, l: usize) -> Option<StoreMut<'_>> {
        self.get_mut(l).map(LayerStore::as_mut)
    }
}

/// What a walk enqueues after its last layer.
pub(super) enum Tail<'a> {
    /// The decode step: the last layer writes the head's input, each
    /// layer's output is copied into its tap row when taps are on, then the
    /// head.
    Step {
        head: &'a mut Head,
        state: &'a mut HeadArgmaxState,
        taps: Option<&'a mut TapRows>,
    },
    /// A pass whose rows each end in a head: row `r`'s residual into
    /// `heads[r]`'s input and that head, row by row.
    Rows {
        heads: &'a mut [Head],
        state: &'a mut HeadArgmaxState,
    },
    /// A prefill pass: the last layer's output stays in the arena's `x`.
    Pass,
    /// A prompt's last unit: its last row's residual, row `m − 1` of the
    /// arena's `x`, into `head`'s input and that head ([`enqueue_last`]).
    Last {
        head: &'a mut Head,
        state: &'a mut HeadArgmaxState,
    },
}

/// One walk of a qwen3moe chain: every layer at `m` rows over arena `s`
/// from input `io`, the stores `stores`, then the [`Tail`].
pub(super) struct Program<'a, S: Stores + ?Sized> {
    pub(super) c: &'a PassCtx<'a>,
    pub(super) stores: &'a mut S,
    pub(super) s: &'a mut Arena,
    pub(super) io: &'a Io<'a>,
    pub(super) m: usize,
    pub(super) tail: Tail<'a>,
}

impl<S: Stores + ?Sized> Program<'_, S> {
    /// Enqueue the walk `(1, m, Step)` over every layer of the plans; a
    /// chain of no layer is refused by name.
    pub(super) fn walk(mut self) -> Result<(), GpuError> {
        if self.c.plans.is_empty() {
            return Err(GpuError::shape(WHAT, "a chain of no layer"));
        }
        let o = Overlap {
            units: 1,
            cols: self.m,
            port: PortKind::Step,
        };
        let mut port = NoLeg { rows: self.s.rows };
        sched::walk(o, self.c.plans.len(), &mut port, &mut self)
    }
}

impl<S: Stores + ?Sized> LayerProgram for Program<'_, S> {
    type Port = NoLeg;

    /// The whole layer at `m` rows, the embedding in front of layer 0; on
    /// the decode step the last layer into the head's input, and the
    /// layer's output into its tap row.
    fn front(&mut self, _: &mut NoLeg, at: At) -> Result<(), GpuError> {
        let Program {
            c,
            stores,
            s,
            io,
            m,
            tail,
        } = self;
        let l = at.layer;
        let (p, st) = c
            .plans
            .get(l)
            .zip(stores.store(l))
            .ok_or(GpuError::state(WHAT, "a plan and a store for every layer"))?;
        let lc = Ctx::new(c.gpu, c.w, (p, l), c.k, c.mma, c.eps, c.table)?;
        let last = l + 1 == c.plans.len();
        let out = match tail {
            Tail::Step { head, .. } if last => Some(head.input_mut()),
            _ => None,
        };
        dispatch::layer(&lc, st, s, io, *m, l == 0, out)?;
        if let Tail::Step {
            head,
            taps: Some(t),
            ..
        } = tail
        {
            let src: &DeviceBuffer<f32> = if last { head.input_mut() } else { &s.x };
            t.rows[l].copy_from_device_async(src, c.gpu.stream())?;
        }
        Ok(())
    }

    /// The walk's [`Tail`].
    fn end(&mut self, _: usize) -> Result<(), GpuError> {
        let Program { c, s, m, tail, .. } = self;
        let (gpu, w, k) = (c.gpu, c.w, c.k);
        match tail {
            Tail::Step { head, state, .. } => dispatch::enqueue_head(gpu, w, k, state, head),
            Tail::Rows { heads, state } => {
                for (row, head) in s.x_rows.iter().zip(heads.iter_mut()) {
                    head.input_mut().copy_from_device_async(row, gpu.stream())?;
                    dispatch::enqueue_head(gpu, w, k, state, head)?;
                }
                Ok(())
            }
            Tail::Pass => Ok(()),
            Tail::Last { head, state } => enqueue_last(gpu, w, k, state, s, *m, head),
        }
    }
}

/// Enqueue the head after a unit of `m` rows over arena `s`: row `m − 1` of
/// `x`, the unit's last residual, into the head's input, then the one-row
/// head. A unit of no row, or of more rows than the arena holds, is refused
/// by name.
pub(super) fn enqueue_last(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    state: &mut HeadArgmaxState,
    s: &Arena,
    m: usize,
    head: &mut Head,
) -> Result<(), GpuError> {
    let h = s.dims.hidden;
    if m == 0 || m > s.rows {
        return Err(GpuError::shape(
            "qwen3moe::enqueue_last",
            format!("a unit of {m} rows on a {}-row arena", s.rows),
        ));
    }
    // SAFETY: row m − 1 < rows spans `hidden` values inside `x` (rows ·
    // hidden), which stays in place while the window lives (one copy).
    let row = unsafe { f32_view(&s.x, (m - 1) * h, h) };
    head.input_mut()
        .copy_from_device_async(&row, gpu.stream())?;
    dispatch::enqueue_head(gpu, w, k, state, head)
}

/// The port of a chain with no host leg: it exchanges nothing, and opens
/// one unit of `1..=rows` columns, `rows` the arena's.
pub(super) struct NoLeg {
    rows: usize,
}

impl Port for NoLeg {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Step;

    /// Refuses by name more than one unit (no leg to overlap across units)
    /// and more columns than the arena holds rows.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        if o.units != 1 || !(1..=self.rows).contains(&o.cols) {
            return Err(GpuError::shape(
                "qwen3moe::NoLeg::open",
                format!(
                    "{} units of {} columns; a chain with no host leg walks one unit of \
                     1..={} columns (the arena's rows)",
                    o.units, o.cols, self.rows
                ),
            ));
        }
        Ok(())
    }

    fn refused(why: Refused) -> GpuError {
        GpuError::shape(WHAT, why.to_string())
    }
}
