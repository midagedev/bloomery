//! The router instance of an Inkling 2 file: the biased sigmoid router at 128 experts (the Mini
//! and its fixture) or 384 (the Small), the top 8 either way. The launches are the common
//! instances' ([`sigmoid_128_8`], [`sigmoid_384_8`]); this file is the one place the body picks
//! between them, so every other site reads the file's expert count and the shared pick count.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};

use crate::gemm::Gemm32Kernels;
use crate::route_body::{sigmoid_128_8, sigmoid_384_8};
use crate::routed_block::{RouteIn, RoutedRows, RouterEntry};
use crate::{Gpu, GpuError};

const WHAT: &str = "inkling2 router";

/// The experts each token routes to: the top 8, both instances'.
pub(super) const N_USED: usize = 8;
const _: () = assert!(
    <sigmoid_128_8::RouterKernels as RouterEntry>::N_USED == N_USED
        && <sigmoid_384_8::RouterKernels as RouterEntry>::N_USED == N_USED
);

/// The router's kernels, one instance a load.
pub(super) enum Route {
    E128(sigmoid_128_8::RouterKernels),
    E384(sigmoid_384_8::RouterKernels),
}

/// One token's results of a [`Route`] launch, of the instance that made them.
pub(super) enum RouteOut {
    E128(sigmoid_128_8::RouterOut),
    E384(sigmoid_384_8::RouterOut),
}

impl Route {
    /// The instance for `n_expert` experts. Refused by name: a count no instance is built for.
    /// Load-time only.
    pub(super) fn load(ctx: &Arc<CudaContext>, n_expert: usize) -> Result<Route, GpuError> {
        match n_expert {
            n if n == <sigmoid_128_8::RouterKernels as RouterEntry>::N_EXPERT => Ok(Route::E128(
                <sigmoid_128_8::RouterKernels as RouterEntry>::load(ctx)?,
            )),
            n if n == <sigmoid_384_8::RouterKernels as RouterEntry>::N_EXPERT => Ok(Route::E384(
                <sigmoid_384_8::RouterKernels as RouterEntry>::load(ctx)?,
            )),
            n => Err(GpuError::shape(
                WHAT,
                format!(
                    "{n} experts, the top {N_USED}: the body's routers are the 128/{N_USED} and \
                     384/{N_USED} instances"
                ),
            )),
        }
    }

    /// The buffers of one token's results of this instance. Load-time only.
    pub(super) fn new_out(&self, stream: &CudaStream) -> Result<RouteOut, GpuError> {
        Ok(match self {
            Route::E128(_) => RouteOut::E128(sigmoid_128_8::RouterOut::new(stream)?),
            Route::E384(_) => RouteOut::E384(sigmoid_384_8::RouterOut::new(stream)?),
        })
    }

    /// The router for the one token `x` into `out`. Refused by name: results of the other
    /// instance.
    pub(super) fn route(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        r: &RouteIn<'_>,
        out: &mut RouteOut,
    ) -> Result<(), GpuError> {
        match (self, out) {
            (Route::E128(k), RouteOut::E128(o)) => k.route(stream, x, r, o),
            (Route::E384(k), RouteOut::E384(o)) => k.route(stream, x, r, o),
            _ => Err(GpuError::state(
                WHAT,
                "results of the kernels' own instance",
            )),
        }
    }

    /// The router for `t` tokens of `x` into `rows` (what `t` calls of [`Route::route`] write).
    pub(super) fn route_rows(
        &self,
        launch: (&CudaStream, &Gpu, &Gemm32Kernels),
        x: &DeviceBuffer<f32>,
        r: &RouteIn<'_>,
        t: usize,
        rows: RoutedRows<'_>,
    ) -> Result<(), GpuError> {
        match self {
            Route::E128(k) => k.route_rows(launch, x, r, t, rows),
            Route::E384(k) => k.route_rows(launch, x, r, t, rows),
        }
    }
}

impl RouteOut {
    /// The picks' expert ids.
    pub(super) fn ids(&self) -> &DeviceBuffer<u32> {
        match self {
            RouteOut::E128(o) => <sigmoid_128_8::RouterKernels as RouterEntry>::out_ids(o),
            RouteOut::E384(o) => <sigmoid_384_8::RouterKernels as RouterEntry>::out_ids(o),
        }
    }

    /// The picks' weights.
    pub(super) fn weights(&self) -> &DeviceBuffer<f32> {
        match self {
            RouteOut::E128(o) => <sigmoid_128_8::RouterKernels as RouterEntry>::out_weights(o),
            RouteOut::E384(o) => <sigmoid_384_8::RouterKernels as RouterEntry>::out_weights(o),
        }
    }

    /// Device bytes of the results.
    pub(super) fn bytes(&self) -> usize {
        match self {
            RouteOut::E128(o) => <sigmoid_128_8::RouterKernels as RouterEntry>::out_bytes(o),
            RouteOut::E384(o) => <sigmoid_384_8::RouterKernels as RouterEntry>::out_bytes(o),
        }
    }
}
