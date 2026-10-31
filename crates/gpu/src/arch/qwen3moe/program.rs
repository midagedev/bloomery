//! The family's layer program ([`Program`]): the ops ([`ChainOps`]) the chain
//! crate's walk ([`Walk`]) orders over one unit of `m` columns, `(1, m,
//! Step)` — the decode step at `m = 1`, a pass at its `m`. Each op is one of
//! the launch groups of [`dispatch`] and [`delta`], which picks its gemv or
//! wide arm from `m`. What follows the last layer is the program's [`Tail`]:
//! a head, nothing, the final norm of every row (a prompt call that returns
//! hidden states), or a caller's buffer the last layer's output goes into.
//! The FFN half's ops end each layer as the tail asks: the last layer's
//! output into the head's input or the caller's buffer, and on a decode step
//! with the gate's taps on, each layer's output copied into its tap row.

use super::body::{Kernels, TapRows};
use super::delta;
use super::dispatch::{self, Ctx, PassCtx};
use super::draft_taps::DraftTaps;
use super::head_argmax::HeadArgmaxState;
use super::plan::{DeltaPlan, FfnPlan, GqaPlan};
use super::scratch::{Arena, Io, KvPlanes, LayerStore, StoreMut, cells_view};
use crate::head::Head;
use crate::model::lookup::f32_gain;
use crate::tensor::Window;
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::walk::Walk;
use cuda_core::DeviceBuffer;

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

/// The planes of one layer run on its own outside a body's chain (the MTP
/// layer's, `index` its index), stored for that index alone.
pub(super) struct OneLayer<'a> {
    pub(super) index: usize,
    pub(super) kv: &'a mut KvPlanes,
}

impl Stores for OneLayer<'_> {
    fn store(&mut self, l: usize) -> Option<StoreMut<'_>> {
        (l == self.index).then_some(StoreMut::Kv(&mut *self.kv))
    }
}

/// What a walk enqueues after its last layer, and where the last layer's
/// output goes ([`Tail::last_out`]).
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
    /// A pass whose every row ends in one head of them all: rows `0..m` of
    /// the arena's `x` — the pass's rows, which its ranges tile from row 0 —
    /// into `head`'s input in one copy, then the head of the `m` rows.
    One {
        head: &'a mut Head,
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
    /// A prompt unit whose every row's final-norm hidden state is the output:
    /// `output_norm` over the `m` rows of the arena's `x` into its `normed`
    /// ([`enqueue_hidden`]); no head, no logits.
    Hidden,
    /// A layer run on its own whose output goes into a buffer of the
    /// caller's (the MTP layer's head input); nothing follows.
    Into(&'a mut DeviceBuffer<f32>),
}

impl Tail<'_> {
    /// The buffer the last layer's combine writes: the head's input on a
    /// decode step, the caller's buffer for [`Tail::Into`]; `None`, the
    /// arena's `x`, for every other tail.
    fn last_out(&mut self) -> Option<&mut DeviceBuffer<f32>> {
        match self {
            Tail::Step { head, .. } => Some(head.input_mut()),
            Tail::Into(buf) => Some(&mut **buf),
            _ => None,
        }
    }

    /// A decode step's gate instrument, at the end of layer `l`'s FFN half:
    /// the layer's output rows into its tap row, from the head's input after
    /// the last layer and from the arena's `x` before it.
    fn tap(&mut self, gpu: &Gpu, x: &DeviceBuffer<f32>, l: LayerCtx<'_>) -> Result<(), GpuError> {
        if let Tail::Step {
            head,
            taps: Some(t),
            ..
        } = self
        {
            let src: &DeviceBuffer<f32> = if l.last { head.input_mut() } else { x };
            t.rows[l.layer].copy_from_device_async(src, gpu.stream())?;
        }
        Ok(())
    }
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
    /// Enqueue the walk `(1, m, Step)` over every layer of the plans
    /// ([`Walk::run`]); a chain of no layer is refused by name.
    pub(super) fn walk(self) -> Result<(), GpuError> {
        self.walk_with(None)
    }

    /// [`Program::walk`], and with `draft` the walk also lays the rows
    /// entering the layers it taps into its buffer ([`DraftTaps`]): one
    /// device copy a tapped layer between its mixer and its FFN half
    /// ([`Walk::run_between`]), where the arena's `x` holds the rows entering
    /// the layer, layer 0's embedding rows included.
    pub(super) fn walk_with(mut self, draft: Option<&mut DraftTaps>) -> Result<(), GpuError> {
        let (plans, m, rows) = (self.c.plans, self.m, self.s.rows);
        let walk = Walk::new(&mut self, plans, m);
        match draft {
            None => walk.run(rows),
            Some(taps) => walk.run_between(rows, |p, cx| match taps.slot(cx.layer) {
                Some(slot) => taps.enter(p.c.gpu.stream(), slot, &p.s.x, p.m),
                None => Ok(()),
            }),
        }
    }

    /// Run `part` of layer `layer` of the plans on its own at `m` rows — the
    /// whole layer (`walk::layer`) or its FFN half (`walk::ffn`) — from the
    /// rows its caller laid in.
    pub(super) fn alone<P>(mut self, layer: usize, part: P) -> Result<(), GpuError>
    where
        P: FnOnce(&mut Self, LayerCtx<'_>) -> Result<(), GpuError>,
    {
        let cx = LayerCtx::of(self.c.plans, layer, self.m)?;
        part(&mut self, cx)
    }
}

/// Layer `l`'s launch context: what every launch of the layer reads besides
/// the arena and the store.
pub(super) fn layer_ctx<'x>(c: &'x PassCtx<'x>, l: LayerCtx<'x>) -> Result<Ctx<'x>, GpuError> {
    Ctx::new(c.gpu, c.w, (l.plan, l.layer), c.k, c.mma, c.eps, c.table)
}

/// Layer `layer`'s store, or a named refusal for a chain with fewer stores
/// than layers.
fn store_of<S: Stores + ?Sized>(stores: &mut S, layer: usize) -> Result<StoreMut<'_>, GpuError> {
    stores.store(layer).ok_or_else(|| {
        GpuError::shape(
            WHAT,
            format!("no store for layer {layer}: every layer of the plans has one"),
        )
    })
}

impl<S: Stores + ?Sized> ChainOps for Program<'_, S> {
    type Error = GpuError;

    fn embed(&mut self, m: usize) -> Result<(), GpuError> {
        let c = self.c;
        let first = c
            .plans
            .first()
            .ok_or_else(|| GpuError::shape(WHAT, "a chain of no layer"))?;
        let lc = Ctx::new(c.gpu, c.w, (first, 0), c.k, c.mma, c.eps, c.table)?;
        dispatch::unit_front(&lc, self.s, self.io, m)
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), GpuError> {
        let lc = layer_ctx(self.c, l)?;
        dispatch::attn_in(&lc, g, self.s, l.m)
    }

    fn attn_core(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), GpuError> {
        let lc = layer_ctx(self.c, l)?;
        match store_of(self.stores, l.layer)? {
            StoreMut::Kv(kv) => dispatch::attn_core(&lc, g, kv, self.s, l.m),
            StoreMut::Rec(_) => Err(dispatch::two_kinds(l.layer)),
        }
    }

    fn attn_append(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), GpuError> {
        let lc = layer_ctx(self.c, l)?;
        match store_of(self.stores, l.layer)? {
            StoreMut::Kv(kv) => dispatch::attn_append(&lc, g, kv, self.s, l.m),
            StoreMut::Rec(_) => Err(dispatch::two_kinds(l.layer)),
        }
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), GpuError> {
        l.plain_only("qwen3moe::Program::attn_out")?;
        let lc = layer_ctx(self.c, l)?;
        dispatch::attn_out(&lc, g, self.s, l.m)
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), GpuError> {
        let lc = layer_ctx(self.c, l)?;
        delta::delta_in(&lc, d, self.s, l.m)
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), GpuError> {
        let lc = layer_ctx(self.c, l)?;
        match store_of(self.stores, l.layer)? {
            StoreMut::Rec(r) => delta::delta_core(&lc, d, r, self.s, self.io, l.m),
            StoreMut::Kv(_) => Err(dispatch::two_kinds(l.layer)),
        }
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, d: &DeltaPlan) -> Result<(), GpuError> {
        l.plain_only("qwen3moe::Program::delta_out")?;
        let lc = layer_ctx(self.c, l)?;
        delta::delta_out(&lc, d, self.s, l.m)
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), GpuError> {
        let lc = layer_ctx(self.c, l)?;
        dispatch::route(&lc, f, self.s, l.m)
    }

    /// The FFN half's rest, the last layer's output into the buffer the
    /// [`Tail`] names ([`Tail::last_out`]), and the decode step's tap of
    /// the layer's output ([`Tail::tap`]).
    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), GpuError> {
        l.plain_only("qwen3moe::Program::experts")?;
        let lc = layer_ctx(self.c, l)?;
        let out = if l.last { self.tail.last_out() } else { None };
        dispatch::experts(&lc, f, self.s, l.m, out)?;
        self.tail.tap(self.c.gpu, &self.s.x, l)
    }

    /// The walk's [`Tail`].
    fn head(&mut self, m: usize) -> Result<(), GpuError> {
        let Program { c, s, tail, .. } = self;
        enqueue_tail(c, s, m, tail)
    }
}

/// Enqueue what follows the last layer of a walk of `m` rows over arena `s`:
/// the walk's [`Tail`] (a step's taps are its layers', copied as each layer
/// ends; a [`Tail::Into`]'s buffer the last layer wrote).
pub(super) fn enqueue_tail(
    c: &PassCtx<'_>,
    s: &mut Arena,
    m: usize,
    tail: &mut Tail<'_>,
) -> Result<(), GpuError> {
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
        Tail::One { head, state } => enqueue_one(gpu, w, k, state, s, m, head),
        Tail::Pass => Ok(()),
        Tail::Last { head, state } => enqueue_last(gpu, w, k, state, s, m, head),
        Tail::Hidden => enqueue_hidden(gpu, w, c.eps, s, m),
        Tail::Into(_) => Ok(()),
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
    let row = unsafe { cells_view(&s.x, (m - 1) * h, h) };
    head.input_mut()
        .copy_from_device_async(&row, gpu.stream())?;
    dispatch::enqueue_head(gpu, w, k, state, head)
}

/// Enqueue the head after a pass of `m` rows over arena `s`: rows `0..m` of
/// `x` — the pass's rows, which its ranges tile from row 0 — into the head's
/// input in one copy, then the head of the `m` rows (the fused arm at one
/// row, the shared arm past it). A unit of no row, or of more rows than the
/// arena holds, is refused by name.
pub(super) fn enqueue_one(
    gpu: &Gpu,
    w: &Weights,
    k: &Kernels,
    state: &mut HeadArgmaxState,
    s: &Arena,
    m: usize,
    head: &mut Head,
) -> Result<(), GpuError> {
    if m == 0 || m > s.rows {
        return Err(GpuError::shape(
            "qwen3moe::enqueue_one",
            format!("a unit of {m} rows on a {}-row arena", s.rows),
        ));
    }
    let rows = Window::of(&s.x, 0, m * s.dims.hidden)?;
    head.input_mut()
        .copy_from_device_async(&rows, gpu.stream())?;
    dispatch::enqueue_head(gpu, w, k, state, head)
}

/// Enqueue the final norm of a unit of `m` rows over arena `s`: rows `0..m`
/// of `x`, the last layer's output, RMS-normed by `output_norm.weight` with
/// epsilon `eps` into rows `0..m` of `normed` — llama.cpp's `result_norm`
/// for each row. A unit of no row, or of more rows than the arena holds, is
/// refused by name.
pub(super) fn enqueue_hidden(
    gpu: &Gpu,
    w: &Weights,
    eps: f32,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    if m == 0 || m > s.rows {
        return Err(GpuError::shape(
            "qwen3moe::enqueue_hidden",
            format!("a unit of {m} rows on a {}-row arena", s.rows),
        ));
    }
    gpu.elem().enqueue_rms_norm(
        gpu.stream(),
        &s.x,
        f32_gain(w, "output_norm.weight")?,
        eps,
        s.dims.hidden,
        m,
        &mut s.normed,
    )
}
