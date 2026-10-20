//! The routed block around the host leg, common to every program whose
//! routed layers run a sigmoid router (a selection bias, the top picks, the
//! weights from the unbiased scores normalized and scaled), hand the picks to
//! the host tier and compute the experts the slot map puts on the card in the
//! host leg's shadow. A family's file keeps what its architecture forces (the
//! shared expert, the base the card sum is added to, the layer program that
//! calls these parts) and takes the rest from here.
//!
//! A routed layer runs three parts around the host leg:
//! - the front, [`RouteBlock::front`]: `ffn_norm` RMS of the residual into the
//!   boundary's handoff region, the router, the handoff into the host-mapped
//!   image (each slot's place in the card's stacks, or the host mark, from the
//!   layer's slot-map row) and the go ([`handoff_go`]). Three kernels and the
//!   go;
//! - the shadow, [`RouteBlock::shadow`]: the card experts of the layer
//!   ([`crate::card_experts`]) over the normed row, with the base the card
//!   sum is added to; none on a layer with no card experts;
//! - the back, [`RouteBlock::back`]: the wait, then the add of the host's
//!   routed sum and the shadow's rows ([`wait_add`]).
//!
//! A prompt batch runs the front over a unit's rows ([`RouteBlock::route_rows`]:
//! the norm over the rows and the router's two launches, what that many
//! one-row fronts write) and the shadow over them
//! ([`RouteBlock::card_rows`]).
//!
//! What differs between families is data: the router instance
//! ([`RouterEntry`], one per `sigmoid_router_instance!`), each layer's tensor
//! names and slot-map row ([`RouteLayer`]), the dims ([`RouteDims`]), and the
//! base a family adds the card sum to.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use crate::card_experts::{CardBatch, CardExperts, Io};
use crate::gemm::Gemm32Kernels;
use crate::host::handoff::{Handoff, HandoffKernels};
use crate::host::run::HostRun;
use crate::hybrid::{Boundary, Hybrid};
use crate::kquant::Act;
use crate::weights::Weights;
use crate::{DeviceTensor, FaultSink, Gpu, GpuError};

/// What a router launch reads besides its activations: the layer's router
/// weight and selection bias, the file's `expert_weights_scale` and the
/// layer's fault sink.
pub struct RouteIn<'a> {
    pub w: &'a DeviceTensor<f32>,
    pub bias: &'a DeviceBuffer<f32>,
    pub scale: f32,
    pub fault: FaultSink,
}

/// Where the router leaves a unit's rows: the scores ([`RouterEntry::N_EXPERT`]
/// a row), the picks' ids and their weights ([`RouterEntry::N_USED`] a row).
pub struct RoutedRows<'a> {
    pub probs: &'a mut DeviceBuffer<f32>,
    pub ids: &'a mut DeviceBuffer<u32>,
    pub weights: &'a mut DeviceBuffer<f32>,
}

/// One instance of the sigmoid router (`sigmoid_router_instance!`): its
/// counts, its kernels and the launches the routed block makes of them.
pub trait RouterEntry: Sized {
    /// Experts the router scores.
    const N_EXPERT: usize;
    /// Experts each token routes to.
    const N_USED: usize;
    /// Where one token's launch leaves its results.
    type Out;

    /// Load the instance's kernels into `ctx`. Load-time only.
    fn load(ctx: &Arc<CudaContext>) -> Result<Self, GpuError>;
    /// The buffers of one token's results. Load-time only.
    fn new_out(stream: &CudaStream) -> Result<Self::Out, GpuError>;
    /// The picks' expert ids in `out`.
    fn out_ids(out: &Self::Out) -> &DeviceBuffer<u32>;
    /// The picks' weights in `out`.
    fn out_weights(out: &Self::Out) -> &DeviceBuffer<f32>;
    /// Device bytes of `out`.
    fn out_bytes(out: &Self::Out) -> usize;
    /// The router for the one token `x` into `out`.
    fn route(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        r: &RouteIn<'_>,
        out: &mut Self::Out,
    ) -> Result<(), GpuError>;
    /// The router for `t` tokens of `x` in two launches, into `rows`: what
    /// `t` calls of [`RouterEntry::route`] write.
    fn route_rows(
        &self,
        launch: (&CudaStream, &Gpu, &Gemm32Kernels),
        x: &DeviceBuffer<f32>,
        r: &RouteIn<'_>,
        t: usize,
        rows: RoutedRows<'_>,
    ) -> Result<(), GpuError>;
}

/// One routed layer's tensors and place in the slot map.
pub struct RouteLayer {
    /// `ffn_norm`'s gain.
    pub norm: String,
    /// The router weight.
    pub router: String,
    /// The selection bias; `None` routes with a zero bias.
    pub bias: Option<String>,
    /// The layer's row in the slot map's card copy, as a word offset
    /// (`SlotMap::row_offset`).
    pub row_off: usize,
}

/// What the file gives the launches of the routed block.
#[derive(Clone, Copy, Debug)]
pub struct RouteDims {
    /// The model's width.
    pub embd: usize,
    pub rms_eps: f32,
    /// `expert_weights_scale`.
    pub scale: f32,
}

/// The routed block of a program: the router instance's kernels, the
/// handoff's, the layers' names and the slot map's card copy. Read-only
/// after the load; the buffers a step writes are a [`RouteBufs`].
pub struct RouteBlock<R: RouterEntry> {
    what: &'static str,
    dims: RouteDims,
    /// Per layer of the description, `None` on a layer with no routed block.
    layers: Vec<Option<RouteLayer>>,
    slots: Arc<DeviceTensor<u32>>,
    router: R,
    handoff: HandoffKernels,
    /// The wide F32 tile GEMM the rows router's scores run on.
    g32: Gemm32Kernels,
    /// The zero selection bias of a layer the file gives none.
    no_bias: Option<DeviceBuffer<f32>>,
}

/// The step's buffers of a [`RouteBlock`]: the router's results for the
/// token in flight and the places of the slots, `rows` tokens' worth.
pub struct RouteBufs<R: RouterEntry> {
    out: R::Out,
    sel: DeviceBuffer<u32>,
}

impl<R: RouterEntry> RouteBufs<R> {
    /// The buffers for passes of up to `rows` tokens (one for a step). Load-time
    /// only.
    pub fn new(stream: &CudaStream, rows: usize) -> Result<Self, GpuError> {
        Ok(RouteBufs {
            out: R::new_out(stream)?,
            sel: DeviceBuffer::zeroed(stream, rows * R::N_USED)?,
        })
    }

    /// Each slot's place in the card's stacks, or the host mark, as the last
    /// handoff wrote them: [`RouterEntry::N_USED`] a token.
    #[must_use]
    pub fn sel(&self) -> &DeviceBuffer<u32> {
        &self.sel
    }

    /// [`RouteBufs::sel`], for a handoff that writes a token's slots at an
    /// offset.
    pub fn sel_mut(&mut self) -> &mut DeviceBuffer<u32> {
        &mut self.sel
    }

    /// The weights of the last token routed.
    #[must_use]
    pub fn weights(&self) -> &DeviceBuffer<f32> {
        R::out_weights(&self.out)
    }

    /// Device bytes of the buffers.
    #[must_use]
    pub fn bytes(&self) -> usize {
        R::out_bytes(&self.out) + self.sel.num_bytes()
    }
}

/// The handoff of one layer's picks `h` into row `row`'s host image with
/// each slot's place into `sel`, then the go that releases the host leg. One
/// kernel and the go.
pub fn handoff_go(
    gpu: &Gpu,
    kernels: &HandoffKernels,
    h: &Handoff<'_>,
    hybrid: &mut Hybrid<HostRun>,
    (l, row): (usize, usize),
    sel: &mut DeviceBuffer<u32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let target = hybrid.boundary_mut().handoff_target_of(row)?;
    kernels.enqueue_handoff(stream, h, target, gpu.layer_sink(l)?, sel)?;
    hybrid.boundary().enqueue_go_of(stream, l, row)
}

/// The wait for row `row`'s host leg of layer `l`, then the add of the
/// host's routed sum and `shadowed`, `n` values, into `dest`.
pub fn wait_add(
    gpu: &Gpu,
    hybrid: &Hybrid<HostRun>,
    (l, row): (usize, usize),
    (shadowed, n): (&DeviceBuffer<f32>, usize),
    dest: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let boundary = hybrid.boundary();
    boundary.enqueue_back_of(stream, row, l)?;
    gpu.elem()
        .enqueue_add(stream, boundary.hsum_of(row)?, shadowed, n, dest)
}

impl<R: RouterEntry> RouteBlock<R> {
    /// The routed block of `layers` (`None`: no routed block), refusing by
    /// name (`what`) a layer whose tensors the weights do not hold. `slots`
    /// is the slot map's card copy. Load-time only.
    pub fn new(
        gpu: &Gpu,
        w: &Weights,
        what: &'static str,
        (layers, slots): (Vec<Option<RouteLayer>>, &Arc<DeviceTensor<u32>>),
        dims: RouteDims,
    ) -> Result<Self, GpuError> {
        let mut zero_bias = false;
        for layer in layers.iter().flatten() {
            w.f32_buf(what, &layer.norm)?;
            w.f32_tensor(what, &layer.router)?;
            match &layer.bias {
                Some(name) => {
                    w.f32_buf(what, name)?;
                }
                None => zero_bias = true,
            }
        }
        let ctx = gpu.context();
        Ok(RouteBlock {
            what,
            dims,
            layers,
            slots: Arc::clone(slots),
            router: R::load(ctx)?,
            handoff: HandoffKernels::load(ctx)?,
            g32: Gemm32Kernels::load(ctx)?,
            no_bias: zero_bias
                .then(|| DeviceBuffer::zeroed(gpu.stream(), R::N_EXPERT))
                .transpose()?,
        })
    }

    /// Device bytes of the block's own buffers.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.no_bias.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }

    /// Layer `l`'s routed block, or a named refusal.
    fn layer(&self, l: usize) -> Result<&RouteLayer, GpuError> {
        self.layers.get(l).and_then(Option::as_ref).ok_or_else(|| {
            GpuError::shape(
                self.what,
                format!(
                    "layer {l}: no routed block (a program of {} layers)",
                    self.layers.len()
                ),
            )
        })
    }

    /// Layer `l`'s router weight and bias, the router's `scale` and the layer's
    /// fault sink.
    fn route_in<'a>(
        &'a self,
        w: &'a Weights,
        l: usize,
        gpu: &Gpu,
    ) -> Result<RouteIn<'a>, GpuError> {
        let layer = self.layer(l)?;
        let bias = match (&layer.bias, &self.no_bias) {
            (Some(name), _) => w.f32_buf(self.what, name)?,
            (None, Some(zero)) => zero,
            (None, None) => {
                return Err(GpuError::State {
                    what: self.what,
                    missing: "the zero selection bias of a layer with no bias",
                });
            }
        };
        Ok(RouteIn {
            w: w.f32_tensor(self.what, &layer.router)?,
            bias,
            scale: self.dims.scale,
            fault: gpu.layer_sink(l)?,
        })
    }

    /// Enqueue layer `l`'s front of the step (module doc): the norm of `x1`
    /// into the boundary's handoff region, the router, the handoff and the
    /// go, for row 0 of the boundary.
    pub fn front(
        &self,
        (gpu, w): (&Gpu, &Weights),
        bufs: &mut RouteBufs<R>,
        x1: &DeviceBuffer<f32>,
        (hybrid, l): (&mut Hybrid<HostRun>, usize),
    ) -> Result<(), GpuError> {
        let layer = self.layer(l)?;
        let r = self.route_in(w, l, gpu)?;
        let stream = gpu.stream();
        gpu.elem().enqueue_rms_norm(
            stream,
            x1,
            w.f32_buf(self.what, &layer.norm)?,
            self.dims.rms_eps,
            self.dims.embd,
            1,
            hybrid.boundary_mut().normed_mut(),
        )?;
        self.router
            .route(stream, hybrid.boundary().normed(), &r, &mut bufs.out)?;
        let h = Handoff {
            ids: R::out_ids(&bufs.out),
            weights: R::out_weights(&bufs.out),
            map: self.slots.buf(),
            row_off: layer.row_off,
            n_expert: R::N_EXPERT,
        };
        handoff_go(gpu, &self.handoff, &h, hybrid, (l, 0), &mut bufs.sel)
    }

    /// Layer `l`'s `ffn_norm` of the `m` rows of `x1` into `normed` and the
    /// router over them into `rows`.
    pub fn route_rows(
        &self,
        (gpu, w): (&Gpu, &Weights),
        (l, m): (usize, usize),
        x1: &DeviceBuffer<f32>,
        (normed, rows): (&mut DeviceBuffer<f32>, RoutedRows<'_>),
    ) -> Result<(), GpuError> {
        let layer = self.layer(l)?;
        let r = self.route_in(w, l, gpu)?;
        let stream = gpu.stream();
        gpu.elem().enqueue_rms_norm(
            stream,
            x1,
            w.f32_buf(self.what, &layer.norm)?,
            self.dims.rms_eps,
            self.dims.embd,
            m,
            normed,
        )?;
        self.router
            .route_rows((stream, gpu, &self.g32), normed, &r, m, rows)
    }

    /// Layer `l`'s slot map: the card copy, the layer's row in it and the
    /// handoff's kernels, for a program that hands the routing of several
    /// columns over one at a time.
    pub fn handoff_of(&self, l: usize) -> Result<HandoffSide<'_>, GpuError> {
        Ok(HandoffSide {
            kernels: &self.handoff,
            map: self.slots.buf(),
            row_off: self.layer(l)?.row_off,
            n_expert: R::N_EXPERT,
        })
    }

    /// Enqueue layer `l`'s card experts of the step in its host leg's shadow,
    /// after its [`RouteBlock::front`]: the card sum plus `base` goes to the
    /// card experts' `pre`. None on a layer with no card experts.
    pub fn shadow(
        &self,
        (gpu, w): (&Gpu, &Weights),
        card: &mut CardExperts,
        bufs: &RouteBufs<R>,
        (boundary, l): (&Boundary, usize),
        (base, rule): (&DeviceBuffer<f32>, Act),
    ) -> Result<(), GpuError> {
        if !card.has(l) {
            return Ok(());
        }
        let io = Io {
            normed: boundary.normed(),
            weights: bufs.weights(),
            x1: base,
            m: 1,
        };
        card.enqueue_step(gpu, w, l, rule, &io, &bufs.sel)
    }

    /// Enqueue layer `l`'s card experts over a prompt batch of `m` rows in the
    /// host leg's shadow: `normed`, `weights` and `ids` as
    /// [`RouteBlock::route_rows`] wrote them, the card sum plus `base` into
    /// `cb`'s `pre`. None on a layer with no card experts; a layer with them
    /// and no `cb` is refused by name.
    pub fn card_rows(
        &self,
        (gpu, w): (&Gpu, &Weights),
        (card, cb): (&CardExperts, Option<&mut CardBatch>),
        (l, m): (usize, usize),
        (base, rule): (&DeviceBuffer<f32>, Act),
        (normed, weights, ids): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<u32>),
    ) -> Result<(), GpuError> {
        if !card.has(l) {
            return Ok(());
        }
        let cb = cb.ok_or(GpuError::State {
            what: self.what,
            missing: "the batch's card experts' buffers",
        })?;
        let io = Io {
            normed,
            weights,
            x1: base,
            m,
        };
        let row_off = self.layer(l)?.row_off;
        card.enqueue_rows(gpu, w, cb, l, rule, &io, (ids, &self.slots, row_off))
    }

    /// Enqueue layer `l`'s back of the step: the wait, then the add of the
    /// host's routed sum and the shadow's rows into `dest` — the card sum
    /// plus the base on a layer with card experts, `base` alone on one
    /// without.
    pub fn back(
        &self,
        gpu: &Gpu,
        hybrid: &Hybrid<HostRun>,
        card: &CardExperts,
        l: usize,
        (base, dest): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let shadowed = if card.has(l) { card.pre() } else { base };
        wait_add(gpu, hybrid, (l, 0), (shadowed, self.dims.embd), dest)
    }
}

/// One layer's side of the handoff ([`RouteBlock::handoff_of`]).
pub struct HandoffSide<'a> {
    pub kernels: &'a HandoffKernels,
    /// The slot map's card copy.
    pub map: &'a DeviceBuffer<u32>,
    pub row_off: usize,
    pub n_expert: usize,
}
