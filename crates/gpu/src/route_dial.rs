//! The router dials of a routed decode step ([`RouteDials`]): the resident bonus
//! (`BLOOMERY_ROUTE_RESIDENT_BONUS`) and the top-k cut (`BLOOMERY_ROUTE_TOPK`), their two launches
//! and the load's refusals. The rules are `runtime::routedial`'s, which a gate's host side holds
//! the device to bit for bit.
//!
//! - **Bonus** ([`RouteDialKernels::enqueue_resident_bias`]): `out[e] = base[e] + λ` for an expert
//!   the layer's stage-card slot map holds in a card slot, `base[e]` for any other, over the
//!   layer's row of the map. The result is the selection bias of the layer's one router launch
//!   in place of the file's: the router adds it to the unbiased score for the selection only and
//!   weighs the picks by the unbiased scores. The map is read on the device when the launch runs,
//!   so a residency flip is seen by the next step, in a captured graph too. One launch of
//!   `n_expert` f32 written, `n_expert` f32 and words read, a layer a step.
//! - **Cut** ([`RouteDialKernels::enqueue_cut`]): the router's `k` picks stay in rank order; the
//!   first `cut` slots are renormalized as the router's rule does for `cut` slots and the later
//!   slots weigh 0. One launch of one thread, `k` f32 written, `cut + k` f32 read, a layer a
//!   step. The host leg skips the zero-weight slots (`HostTier::set_route_cut`); the card's sum
//!   adds `fma(down, 0, acc)` for them, which is `acc` for a finite down row.
//!
//! Neither launch exists when its dial is unset: the router launch reads the file's bias buffer,
//! and no kernel of the step changed.

use std::sync::Arc;

use bloomery_levers::Levers;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::add_rn_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use runtime::routedial::{RouteDialError, cut_of};

use crate::hybrid::HOST;
use crate::route_core::{floor_divisor, renorm_divisor};
use crate::{FaultSink, FaultSite, GpuError, launch_u32};

/// Threads per block of the bias launch.
const THREADS: u32 = 256;
// The entry's launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// The dials of a load, parsed once ([`RouteDials::from_levers`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteDials {
    /// λ's `f32` bits; `None` off.
    bonus: Option<u32>,
    /// k'; `None` the model's own.
    topk: Option<usize>,
}

impl RouteDials {
    /// No dial: today's routing.
    pub const OFF: RouteDials = RouteDials {
        bonus: None,
        topk: None,
    };

    /// The dials of a binary's one parse (`bloomery_levers::at_main`).
    #[must_use]
    pub fn from_levers(levers: &Levers) -> RouteDials {
        RouteDials {
            bonus: levers.route_resident_bonus().map(f32::to_bits),
            topk: levers.route_topk(),
        }
    }

    /// The dials a gate states directly: λ (`None` or 0 off) and k'.
    #[must_use]
    pub fn of(bonus: Option<f32>, topk: Option<usize>) -> RouteDials {
        RouteDials {
            bonus: bonus.filter(|&b| b != 0.0).map(f32::to_bits),
            topk,
        }
    }

    /// λ, when the bonus is on.
    #[must_use]
    pub fn bonus(&self) -> Option<f32> {
        self.bonus.map(f32::from_bits)
    }

    /// The value of `BLOOMERY_ROUTE_TOPK` as set.
    #[must_use]
    pub fn topk(&self) -> Option<usize> {
        self.topk
    }

    /// Whether any dial is on (a count equal to the model's `k` is a dial a load drops, so this
    /// reads the values as set).
    #[must_use]
    pub fn any(&self) -> bool {
        self.bonus.is_some() || self.topk.is_some()
    }

    /// The load's check of the dials against the model's own `k` and what it runs beside:
    /// `draft` a draft or a verify pass (rows that pick without the dials), `tier` an expert tier
    /// card (its images carry the slot weights the cut zeroes on the stage card only), `softmax`
    /// a router that selects by probability alone (no selection bias to carry the bonus). Returns
    /// the cut as a slot count, `None` for the model's own `k`.
    pub fn check(&self, k: usize, beside: Beside) -> Result<Option<usize>, GpuError> {
        const WHAT: &str = "route dials";
        let refuse = |e: RouteDialError| GpuError::shape(WHAT, e.to_string());
        let cut = cut_of(self.topk, k).map_err(refuse)?;
        if self.any() && beside.draft {
            return Err(refuse(RouteDialError::Beside {
                dial: "BLOOMERY_ROUTE_RESIDENT_BONUS / BLOOMERY_ROUTE_TOPK",
                other: "BLOOMERY_DRAFT",
            }));
        }
        if cut.is_some() && beside.tier {
            return Err(refuse(RouteDialError::Beside {
                dial: "BLOOMERY_ROUTE_TOPK",
                other: "an expert tier card",
            }));
        }
        if self.bonus.is_some() && beside.softmax {
            return Err(GpuError::shape(
                WHAT,
                "BLOOMERY_ROUTE_RESIDENT_BONUS on a router that selects by probability alone: it \
                 has no selection bias to carry the bonus",
            ));
        }
        Ok(cut)
    }
}

/// What a load runs the dials beside ([`RouteDials::check`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct Beside {
    pub draft: bool,
    pub tier: bool,
    pub softmax: bool,
}

#[cuda_module]
mod route_dial_kernels {
    use super::*;

    /// The layer's selection bias with the resident bonus: thread `e < n` writes `out[e] =
    /// base[e] + lambda` when `map[row_off + e]` is not [`HOST`] (a stage card slot: the stage
    /// view of the map holds a tier or host expert as [`HOST`]), `base[e]` otherwise.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            base.len() >= n,
            map.len() >= row_off + n,
            out.len() >= n
        )
    )]
    pub fn route_resident_bias(
        base: &[f32],
        map: &[u32],
        row_off: u32,
        n: u32,
        lambda: f32,
        mut out: DisjointSlice<f32>,
    ) {
        let e = thread::index_1d().get();
        if e >= n as usize {
            return;
        }
        // SAFETY: e < n; base.len() >= n and map.len() >= row_off + n by the launch contract.
        let (b, place) = unsafe {
            (
                *base.get_unchecked(e),
                *map.get_unchecked(row_off as usize + e),
            )
        };
        let v = if place != HOST {
            add_rn_f32(b, lambda)
        } else {
            b
        };
        // SAFETY: e < n <= out.len(); thread e is out[e]'s only writer.
        unsafe { *out.get_unchecked_mut(e) = v };
    }

    /// The cut weights of one token: thread 0 sums the unbiased scores of the first `cut` picked
    /// slots in f64 in slot order, narrows the sum to f32, takes the router's divisor guard
    /// (`floor != 0`: [`floor_divisor`], else [`renorm_divisor`]) and writes `weights[j] =
    /// score_j / divisor · scale` for `j < cut`, 0 for `cut <= j < n_used`. An id not below
    /// `n_expert` raises [`FaultSite::ExpertId`] and writes NaN weights; a weight that is not
    /// finite raises [`FaultSite::Router`], as the router's own weight stage does.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            ids.len() >= n_used,
            probs.len() >= n_expert,
            cut >= 1,
            cut <= n_used,
            weights.len() >= n_used
        )
    )]
    pub fn route_cut_weights(
        ids: &[u32],
        probs: &[f32],
        n_expert: u32,
        n_used: u32,
        cut: u32,
        scale: f32,
        floor: u32,
        fault: FaultSink,
        mut weights: DisjointSlice<f32>,
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mut sum = 0.0f64;
        let mut j = 0usize;
        while j < cut as usize {
            // SAFETY: j < cut <= n_used <= ids.len() by the launch contract.
            let id = unsafe { *ids.get_unchecked(j) };
            if id >= n_expert {
                fault.raise(FaultSite::ExpertId);
                let mut s = 0usize;
                while s < n_used as usize {
                    // SAFETY: s < n_used <= weights.len(); thread 0 alone writes.
                    unsafe { *weights.get_unchecked_mut(s) = f32::NAN };
                    s += 1;
                }
                return;
            }
            // SAFETY: id < n_expert <= probs.len() by the check above and the launch contract.
            sum += f64::from(unsafe { *probs.get_unchecked(id as usize) });
            j += 1;
        }
        let sum = sum as f32;
        let div = if floor != 0 {
            floor_divisor(sum)
        } else {
            renorm_divisor(sum)
        };
        let mut j = 0usize;
        while j < n_used as usize {
            let wt = if j < cut as usize {
                // SAFETY: j < cut <= ids.len(), and the id was below n_expert in the first walk.
                let id = unsafe { *ids.get_unchecked(j) };
                // SAFETY: id < n_expert <= probs.len().
                let p = unsafe { *probs.get_unchecked(id as usize) };
                p / div * scale
            } else {
                0.0
            };
            if !wt.is_finite() {
                fault.raise(FaultSite::Router);
            }
            // SAFETY: j < n_used <= weights.len(); thread 0 alone writes.
            unsafe { *weights.get_unchecked_mut(j) = wt };
            j += 1;
        }
    }
}

/// The dials' two entries, loaded once ([`RouteDialKernels::load`]) by a load whose dials are on.
pub struct RouteDialKernels {
    module: route_dial_kernels::LoadedModule,
}

impl RouteDialKernels {
    /// Load the entries' device bundle into `ctx`. Load-time only, and only for a load whose
    /// dials are on.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RouteDialKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above;
        // each launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(route_dial_kernels, ctx)? };
        Ok(RouteDialKernels { module })
    }

    /// Enqueue the layer's biased selection bias: `out[e] = base[e] + lambda` for the `n` experts
    /// of the map's row at `row_off` that hold a stage card slot, `base[e]` for the others. Buffers
    /// short of `n` are refused by name. One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_resident_bias(
        &self,
        stream: &CudaStream,
        (base, map, row_off): (&DeviceBuffer<f32>, &DeviceBuffer<u32>, usize),
        lambda: f32,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "route_resident_bias";
        let n = base.len().min(out.len());
        if n == 0 || map.len() < row_off + n {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "base {} and out {} f32, a map of {} words read from {row_off}: a row of at \
                     least one expert in each",
                    base.len(),
                    out.len(),
                    map.len()
                ),
            ));
        }
        let grid = launch_u32(WHAT, "grid", n.div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_route_resident_bias(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.route_resident_bias(
            stream,
            &prep,
            base,
            map,
            launch_u32(WHAT, "row_off", row_off)?,
            launch_u32(WHAT, "n", n)?,
            lambda,
            out,
        )?;
        Ok(())
    }

    /// Enqueue the cut of one token's `n_used` picked slots to the first `cut`
    /// (`route_cut_weights`): `picks` the router's ids and its unbiased scores over `n_expert`
    /// experts, the new weights into `weights` (the buffer the router wrote). `cut` is 1 to
    /// `n_used`; `floor` the router instance's divisor guard. One launch of one thread.
    /// Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's picks, shape, rule, sink and output (rust-quality R8)"
    )]
    pub fn enqueue_cut(
        &self,
        stream: &CudaStream,
        picks: (&DeviceBuffer<u32>, &DeviceBuffer<f32>),
        (n_expert, n_used, cut): (usize, usize, usize),
        (scale, floor): (f32, bool),
        fault: FaultSink,
        weights: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "route_cut_weights";
        let (ids, probs) = picks;
        if cut == 0
            || cut > n_used
            || ids.len() < n_used
            || probs.len() < n_expert
            || weights.len() < n_used
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "cut {cut} of {n_used} slots, ids {} and weights {} words, scores {} of {n_expert} \
                     experts: a cut of 1 to the slot count over buffers that long",
                    ids.len(),
                    weights.len(),
                    probs.len()
                ),
            ));
        }
        let prep = self
            .module
            .prepare_route_cut_weights(LaunchConfig1D::new(1, 32, 0))?;
        self.module.route_cut_weights(
            stream,
            &prep,
            ids,
            probs,
            launch_u32(WHAT, "n_expert", n_expert)?,
            launch_u32(WHAT, "n_used", n_used)?,
            launch_u32(WHAT, "cut", cut)?,
            scale,
            u32::from(floor),
            fault,
            weights,
        )?;
        Ok(())
    }
}
