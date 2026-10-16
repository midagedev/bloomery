//! Bolt's routers: the sigmoid router with a selection bias
//! ([`crate::sigmoid_router_instance!`], the generic bodies of
//! [`crate::route_body`] GLM-5.3-Flash's and MiMo's routers are instances
//! of) at each expert count a Bolt MoE layer has — 240, 304, 368, 432 and
//! 496, the top 9 of each (`models::shape::ROUTERS`' Bolt rows). Each
//! expert scores `sigmoid(logit)` of its F32 router row on the layer's normed
//! row; the bias is added for the pick only; the top 9 by the biased score
//! are kept, their unbiased scores summed in f64 in slot order, the sum
//! narrowed to f32 and guarded ([`crate::route_core::renorm_divisor`]), each
//! score divided by it and multiplied by `expert_weights_scale`, all in f32
//! (llama-graph.cpp `build_moe_ffn`:2167-2280 at cd0a5b6a1: `ggml_sigmoid`,
//! `probs + exp_probs_b` for the selection, `ggml_argsort_top_k`,
//! `ggml_get_rows` of the unbiased probs, the clamped sum's division,
//! `ggml_scale`). A selection value or a weight that is not finite raises
//! [`crate::FaultSite::Router`].
//!
//! [`Routers`] holds the five instances' modules, [`RouteOut`] one layer's
//! results at its own count; a count no instance is built for is refused by
//! name at load.

use crate::{DeviceTensor, FaultSink, Gpu, GpuError};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use std::sync::Arc;

/// Experts every Bolt MoE layer keeps a token.
pub(crate) const TOP_K: usize = 9;

crate::sigmoid_router_instance! {
    /// Bolt's router at 240 experts, the top 9.
    pub mod e240 {
        experts: 240,
        used: 9,
        kernels: nemotron_moe_router_240_kernels,
        entries: nemotron_moe_router_240, nemotron_moe_router_240_pick,
        prepare: prepare_nemotron_moe_router_240, prepare_nemotron_moe_router_240_pick,
    }
}

crate::sigmoid_router_instance! {
    /// Bolt's router at 304 experts, the top 9.
    pub mod e304 {
        experts: 304,
        used: 9,
        kernels: nemotron_moe_router_304_kernels,
        entries: nemotron_moe_router_304, nemotron_moe_router_304_pick,
        prepare: prepare_nemotron_moe_router_304, prepare_nemotron_moe_router_304_pick,
    }
}

crate::sigmoid_router_instance! {
    /// Bolt's router at 368 experts, the top 9.
    pub mod e368 {
        experts: 368,
        used: 9,
        kernels: nemotron_moe_router_368_kernels,
        entries: nemotron_moe_router_368, nemotron_moe_router_368_pick,
        prepare: prepare_nemotron_moe_router_368, prepare_nemotron_moe_router_368_pick,
    }
}

crate::sigmoid_router_instance! {
    /// Bolt's router at 432 experts, the top 9.
    pub mod e432 {
        experts: 432,
        used: 9,
        kernels: nemotron_moe_router_432_kernels,
        entries: nemotron_moe_router_432, nemotron_moe_router_432_pick,
        prepare: prepare_nemotron_moe_router_432, prepare_nemotron_moe_router_432_pick,
    }
}

crate::sigmoid_router_instance! {
    /// Bolt's router at 496 experts, the top 9.
    pub mod e496 {
        experts: 496,
        used: 9,
        kernels: nemotron_moe_router_496_kernels,
        entries: nemotron_moe_router_496, nemotron_moe_router_496_pick,
        prepare: prepare_nemotron_moe_router_496, prepare_nemotron_moe_router_496_pick,
    }
}

const _: () = assert!(
    e240::N_USED == TOP_K
        && e304::N_USED == TOP_K
        && e368::N_USED == TOP_K
        && e432::N_USED == TOP_K
        && e496::N_USED == TOP_K
);

/// The expert counts an instance is built for.
pub(crate) const COUNTS: [usize; 5] = [
    e240::N_EXPERT,
    e304::N_EXPERT,
    e368::N_EXPERT,
    e432::N_EXPERT,
    e496::N_EXPERT,
];

/// The five instances' loaded modules.
pub(crate) struct Routers {
    e240: e240::RouterKernels,
    e304: e304::RouterKernels,
    e368: e368::RouterKernels,
    e432: e432::RouterKernels,
    e496: e496::RouterKernels,
}

/// One layer's router results at its own expert count.
pub(crate) enum RouteOut {
    E240(e240::RouterOut),
    E304(e304::RouterOut),
    E368(e368::RouterOut),
    E432(e432::RouterOut),
    E496(e496::RouterOut),
}

impl RouteOut {
    /// The buffers of a layer of `n_expert` experts; a count no instance is
    /// built for is refused by name as `what`'s. Load-time only.
    pub(crate) fn new(
        what: &'static str,
        stream: &CudaStream,
        n_expert: usize,
    ) -> Result<RouteOut, GpuError> {
        Ok(match n_expert {
            e240::N_EXPERT => RouteOut::E240(e240::RouterOut::new(stream)?),
            e304::N_EXPERT => RouteOut::E304(e304::RouterOut::new(stream)?),
            e368::N_EXPERT => RouteOut::E368(e368::RouterOut::new(stream)?),
            e432::N_EXPERT => RouteOut::E432(e432::RouterOut::new(stream)?),
            e496::N_EXPERT => RouteOut::E496(e496::RouterOut::new(stream)?),
            other => {
                return Err(GpuError::shape(
                    what,
                    format!("a router of {other} experts: the instances route over {COUNTS:?}"),
                ));
            }
        })
    }

    /// The count of experts the layer routes over.
    pub(crate) fn n_expert(&self) -> usize {
        match self {
            RouteOut::E240(_) => e240::N_EXPERT,
            RouteOut::E304(_) => e304::N_EXPERT,
            RouteOut::E368(_) => e368::N_EXPERT,
            RouteOut::E432(_) => e432::N_EXPERT,
            RouteOut::E496(_) => e496::N_EXPERT,
        }
    }

    /// Enqueue the copy of row `row` of a pass's router results (`probs` of
    /// [`RouteOut::n_expert`] per row, `ids` and `weights` of [`TOP_K`]) into
    /// these buffers, so the layer's results read as that row's.
    pub(crate) fn enqueue_row_from(
        &mut self,
        stream: &CudaStream,
        (probs, ids, weights): (&DeviceBuffer<f32>, &DeviceBuffer<u32>, &DeviceBuffer<f32>),
        row: usize,
    ) -> Result<(), GpuError> {
        let n = self.n_expert();
        let row_of = |width: usize| crate::tensor::Pitched {
            width,
            rows: 1,
            dev_pitch: width,
            host_pitch: width,
        };
        let (f, u) = (size_of::<f32>(), size_of::<u32>());
        let (p, i, w) = match self {
            RouteOut::E240(o) => (&mut o.probs, &mut o.ids, &mut o.weights),
            RouteOut::E304(o) => (&mut o.probs, &mut o.ids, &mut o.weights),
            RouteOut::E368(o) => (&mut o.probs, &mut o.ids, &mut o.weights),
            RouteOut::E432(o) => (&mut o.probs, &mut o.ids, &mut o.weights),
            RouteOut::E496(o) => (&mut o.probs, &mut o.ids, &mut o.weights),
        };
        crate::tensor::dtod_2d(stream, (probs, row * n * f), (p, 0), row_of(n * f))?;
        crate::tensor::dtod_2d(stream, (ids, row * TOP_K * u), (i, 0), row_of(TOP_K * u))?;
        crate::tensor::dtod_2d(
            stream,
            (weights, row * TOP_K * f),
            (w, 0),
            row_of(TOP_K * f),
        )
    }

    /// The slots' expert ids, in rank order ([`TOP_K`]).
    pub(crate) fn ids(&self) -> &DeviceBuffer<u32> {
        match self {
            RouteOut::E240(o) => &o.ids,
            RouteOut::E304(o) => &o.ids,
            RouteOut::E368(o) => &o.ids,
            RouteOut::E432(o) => &o.ids,
            RouteOut::E496(o) => &o.ids,
        }
    }

    /// Every expert's unbiased score, the selection's input before the bias.
    pub(crate) fn probs(&self) -> &DeviceBuffer<f32> {
        match self {
            RouteOut::E240(o) => &o.probs,
            RouteOut::E304(o) => &o.probs,
            RouteOut::E368(o) => &o.probs,
            RouteOut::E432(o) => &o.probs,
            RouteOut::E496(o) => &o.probs,
        }
    }

    /// The slots' weights, in the ids' order.
    pub(crate) fn weights(&self) -> &DeviceBuffer<f32> {
        match self {
            RouteOut::E240(o) => &o.weights,
            RouteOut::E304(o) => &o.weights,
            RouteOut::E368(o) => &o.weights,
            RouteOut::E432(o) => &o.weights,
            RouteOut::E496(o) => &o.weights,
        }
    }

    /// The device bytes the results hold.
    pub(crate) fn bytes(&self) -> usize {
        let b = |l: &DeviceBuffer<f32>, p: &DeviceBuffer<f32>| l.num_bytes() + p.num_bytes();
        let (lp, slots) = match self {
            RouteOut::E240(o) => (b(&o.logits, &o.probs), o.ids.num_bytes()),
            RouteOut::E304(o) => (b(&o.logits, &o.probs), o.ids.num_bytes()),
            RouteOut::E368(o) => (b(&o.logits, &o.probs), o.ids.num_bytes()),
            RouteOut::E432(o) => (b(&o.logits, &o.probs), o.ids.num_bytes()),
            RouteOut::E496(o) => (b(&o.logits, &o.probs), o.ids.num_bytes()),
        };
        lp + 2 * slots + size_of::<u32>()
    }
}

impl Routers {
    /// Load the five modules into `ctx`. Load-time only.
    pub(crate) fn load(ctx: &Arc<CudaContext>) -> Result<Routers, GpuError> {
        Ok(Routers {
            e240: e240::RouterKernels::load(ctx)?,
            e304: e304::RouterKernels::load(ctx)?,
            e368: e368::RouterKernels::load(ctx)?,
            e432: e432::RouterKernels::load(ctx)?,
            e496: e496::RouterKernels::load(ctx)?,
        })
    }

    /// Enqueue the router of `out`'s count for one token: `w` the router rows
    /// as f32, `x` the token's normed row, `bias` the selection bias, `scale`
    /// the file's `expert_weights_scale`. Asynchronous, allocation-free,
    /// capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the instances' launcher's inputs and the layer's results (rust-quality R8)"
    )]
    pub(crate) fn enqueue(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        scale: f32,
        out: &mut RouteOut,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        match out {
            RouteOut::E240(o) => self
                .e240
                .enqueue_router(stream, w, x, bias, scale, o, fault),
            RouteOut::E304(o) => self
                .e304
                .enqueue_router(stream, w, x, bias, scale, o, fault),
            RouteOut::E368(o) => self
                .e368
                .enqueue_router(stream, w, x, bias, scale, o, fault),
            RouteOut::E432(o) => self
                .e432
                .enqueue_router(stream, w, x, bias, scale, o, fault),
            RouteOut::E496(o) => self
                .e496
                .enqueue_router(stream, w, x, bias, scale, o, fault),
        }
    }

    /// Enqueue the router of `n_expert` experts for `t` tokens: `x` the
    /// tokens' normed rows, row-major; the results `probs` ([`RouteOut::n_expert`]
    /// per row), `ids` and `weights` ([`TOP_K`] per row). A count no instance
    /// is built for is refused by name. Asynchronous, allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "the instances' launcher's inputs and the pass's results (rust-quality R8)"
    )]
    pub(crate) fn enqueue_rows(
        &self,
        stream: &CudaStream,
        gpu: &Gpu,
        wide: &crate::gemm::Gemm32Kernels,
        n_expert: usize,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        scale: f32,
        t: usize,
        (probs, ids, weights): (
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<u32>,
            &mut DeviceBuffer<f32>,
        ),
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        match n_expert {
            e240::N_EXPERT => self.e240.enqueue_router_rows(
                stream, gpu, wide, w, x, bias, scale, t, probs, ids, weights, fault,
            ),
            e304::N_EXPERT => self.e304.enqueue_router_rows(
                stream, gpu, wide, w, x, bias, scale, t, probs, ids, weights, fault,
            ),
            e368::N_EXPERT => self.e368.enqueue_router_rows(
                stream, gpu, wide, w, x, bias, scale, t, probs, ids, weights, fault,
            ),
            e432::N_EXPERT => self.e432.enqueue_router_rows(
                stream, gpu, wide, w, x, bias, scale, t, probs, ids, weights, fault,
            ),
            e496::N_EXPERT => self.e496.enqueue_router_rows(
                stream, gpu, wide, w, x, bias, scale, t, probs, ids, weights, fault,
            ),
            other => Err(GpuError::shape(
                "Routers::enqueue_rows",
                format!("a router of {other} experts: the instances route over {COUNTS:?}"),
            )),
        }
    }
}
