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
//! - **Mask** ([`MaskedBias`]): a banned expert's selection bias is the file's plus
//!   [`runtime::routedial::BAN`], a buffer per masked layer made at the load. It is the base of the
//!   bonus launch and the router's bias otherwise, so it needs no launch and no kernel entry.
//! - **Census** ([`RouteCensus`], `route_census_add`): per (layer, expert) the picks that ran and
//!   their gate weights, two device counters, one launch after a router launch.
//!
//! Neither launch exists when its dial is unset: the router launch reads the file's bias buffer,
//! and no kernel of the step changed.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bloomery_levers::Levers;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32, DeviceAtomicU32};
use cuda_device::float::add_rn_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use runtime::routedial::{CensusTable, MaskShape, RouteDialError, RouteMask, cut_of};

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

/// What a load takes of the router levers: the dials, the banned-experts sidecar's path and the
/// census file's ([`RouteSetup::from_levers`]). A family reads the mask against its own shape
/// ([`RouteSetup::read_mask`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RouteSetup {
    pub dials: RouteDials,
    pub mask: Option<PathBuf>,
    pub census: Option<PathBuf>,
}

impl RouteSetup {
    /// Every lever unset: today's routing.
    pub const OFF: RouteSetup = RouteSetup {
        dials: RouteDials::OFF,
        mask: None,
        census: None,
    };

    /// The setup of a binary's one parse (`bloomery_levers::at_main`).
    #[must_use]
    pub fn from_levers(levers: &Levers) -> RouteSetup {
        RouteSetup {
            dials: RouteDials::from_levers(levers),
            mask: levers.route_mask().map(Path::to_path_buf),
            census: levers.route_census().map(Path::to_path_buf),
        }
    }

    /// The sidecar and census paths of the process environment beside `dials`: for a load whose
    /// dials arrive typed (`BodyLevers`) and whose files are read where the model opens.
    pub fn with_env_files(dials: RouteDials) -> Result<RouteSetup, GpuError> {
        let levers =
            Levers::from_env().map_err(|e| GpuError::shape("route dials", e.to_string()))?;
        Ok(RouteSetup {
            dials,
            mask: levers.route_mask().map(Path::to_path_buf),
            census: levers.route_census().map(Path::to_path_buf),
        })
    }

    /// Whether any lever is on.
    #[must_use]
    pub fn any(&self) -> bool {
        self.dials.any() || self.mask.is_some() || self.census.is_some()
    }

    /// The load's check: [`RouteDials::check`], and the mask beside a draft refused the same way
    /// (a draft's verify rows pick without it). Returns the cut.
    pub fn check(&self, k: usize, beside: Beside) -> Result<Option<usize>, GpuError> {
        if (self.mask.is_some() || self.census.is_some()) && beside.draft {
            return Err(GpuError::shape(
                "route dials",
                RouteDialError::Beside {
                    dial: "BLOOMERY_ROUTE_MASK / BLOOMERY_ROUTE_CENSUS",
                    other: "BLOOMERY_DRAFT",
                }
                .to_string(),
            ));
        }
        self.dials.check(k, beside)
    }

    /// The mask file read against `shape`; `None` unset. Refused by name: a file that cannot be
    /// read and every line [`RouteMask::parse`] refuses.
    pub fn read_mask(&self, shape: MaskShape) -> Result<Option<RouteMask>, GpuError> {
        let Some(path) = &self.mask else {
            return Ok(None);
        };
        let text = std::fs::read_to_string(path).map_err(|e| {
            GpuError::shape(
                "route mask",
                format!("BLOOMERY_ROUTE_MASK {}: {e}", path.display()),
            )
        })?;
        RouteMask::parse(&text, shape)
            .map(Some)
            .map_err(|e| GpuError::shape("route mask", e.to_string()))
    }
}

/// The masked selection biases of a load: per layer with a banned expert, the file's bias plus
/// [`runtime::routedial::BAN`] on those experts, on the card ([`MaskedBias::upload`]).
pub struct MaskedBias {
    layers: Vec<Option<Arc<DeviceBuffer<f32>>>>,
}

impl MaskedBias {
    /// Upload the masked bias of every layer `mask` names; `base(layer)` reads that layer's file
    /// bias on the host. A model of `n_layer` layers. Load-time only.
    pub fn upload(
        stream: &CudaStream,
        mask: &RouteMask,
        n_layer: usize,
        mut base: impl FnMut(usize) -> Result<Vec<f32>, GpuError>,
    ) -> Result<MaskedBias, GpuError> {
        let mut layers: Vec<Option<Arc<DeviceBuffer<f32>>>> = (0..n_layer).map(|_| None).collect();
        for l in mask.layers() {
            let host = mask.masked_bias(l, &base(l)?);
            layers[l] = Some(Arc::new(DeviceBuffer::from_host(stream, &host)?));
        }
        Ok(MaskedBias { layers })
    }

    /// Layer `layer`'s masked bias; `None` for a layer the mask leaves alone.
    #[must_use]
    pub fn get(&self, layer: usize) -> Option<&DeviceBuffer<f32>> {
        self.layers.get(layer).and_then(Option::as_deref)
    }

    /// Layer `layer`'s masked bias as a shared handle, for a caller that holds it past a borrow
    /// of the set.
    #[must_use]
    pub fn handle(&self, layer: usize) -> Option<Arc<DeviceBuffer<f32>>> {
        self.layers
            .get(layer)
            .and_then(Option::as_ref)
            .map(Arc::clone)
    }

    /// Device bytes of the buffers.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.layers.iter().flatten().map(|b| b.num_bytes()).sum()
    }
}

/// The routing census of a load: per (layer, expert) the picks that ran and their gate weights'
/// sum, in two device counters that a launch after each router launch adds to
/// ([`RouteCensus::enqueue`]) and a flush reads back and writes ([`RouteCensus::flush`]).
pub struct RouteCensus {
    stream: Arc<CudaStream>,
    path: PathBuf,
    shape: (usize, usize, usize),
    picks: DeviceBuffer<u32>,
    weight: DeviceBuffer<f32>,
}

impl RouteCensus {
    /// Zeroed counters for `(n_layer, n_expert, n_used)`, written to `path` by [`flush`] and when
    /// the census is dropped (best effort: an error goes to stderr). The file's directory is
    /// checked now, so a path that cannot be written is refused at the load.
    ///
    /// [`flush`]: RouteCensus::flush
    pub fn new(
        stream: Arc<CudaStream>,
        path: PathBuf,
        shape: (usize, usize, usize),
    ) -> Result<RouteCensus, GpuError> {
        let (n_layer, n_expert, _) = shape;
        let dir = path.parent().filter(|d| !d.as_os_str().is_empty());
        if let Some(dir) = dir
            && !dir.is_dir()
        {
            return Err(GpuError::shape(
                "route census",
                format!(
                    "BLOOMERY_ROUTE_CENSUS {}: no directory {}",
                    path.display(),
                    dir.display()
                ),
            ));
        }
        let n = (n_layer * n_expert).max(1);
        Ok(RouteCensus {
            picks: DeviceBuffer::zeroed(&stream, n)?,
            weight: DeviceBuffer::zeroed(&stream, n)?,
            stream,
            path,
            shape,
        })
    }

    /// Enqueue the add of `rows` rows of layer `layer`'s picks (`ids` and `weights`, `n_used` a
    /// row, from slot `first` of the buffers), the first `live` slots of each: the slots a cut
    /// left unrun are not counted. One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue(
        &mut self,
        kernels: &RouteDialKernels,
        stream: &CudaStream,
        layer: usize,
        (ids, weights): (&DeviceBuffer<u32>, &DeviceBuffer<f32>),
        (first, rows, live): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        const WHAT: &str = "route_census_add";
        let (n_layer, n_expert, n_used) = self.shape;
        if layer >= n_layer
            || rows == 0
            || live == 0
            || live > n_used
            || ids.len() < first + rows * n_used
            || weights.len() < first + rows * n_used
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer} of {n_layer}, {rows} rows of {live} live slots of {n_used}, ids {} \
                     and weights {} words",
                    ids.len(),
                    weights.len()
                ),
            ));
        }
        let n = rows * n_used;
        let grid = launch_u32(WHAT, "grid", n.div_ceil(CENSUS_THREADS as usize))?;
        let prep = kernels
            .module
            .prepare_route_census_add(LaunchConfig1D::new(grid, CENSUS_THREADS, 0))?;
        kernels.module.route_census_add(
            stream,
            &prep,
            ids,
            weights,
            launch_u32(WHAT, "n_expert", n_expert)?,
            launch_u32(WHAT, "n_used", n_used)?,
            launch_u32(WHAT, "live", live)?,
            launch_u32(WHAT, "rows", rows)?,
            launch_u32(WHAT, "first", first)?,
            launch_u32(WHAT, "layer_off", layer * n_expert)?,
            &mut self.picks,
            &mut self.weight,
        )?;
        Ok(())
    }

    /// Read the counters back (synchronizes the engine stream) and write the table: a new file
    /// renamed over the old, so a reader never sees half of one.
    pub fn flush(&self) -> Result<(), GpuError> {
        const WHAT: &str = "route census";
        let picks = self.picks.to_host_vec(&self.stream)?;
        let weight = self.weight.to_host_vec(&self.stream)?;
        let table = CensusTable::of_device(self.shape, &picks, &weight)
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let mut tmp = self.path.clone().into_os_string();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        let io = |e: std::io::Error| GpuError::shape(WHAT, format!("{}: {e}", self.path.display()));
        std::fs::write(&tmp, table.render()).map_err(io)?;
        std::fs::rename(&tmp, &self.path).map_err(io)
    }

    /// Device bytes of the two counters.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.picks.num_bytes() + self.weight.num_bytes()
    }
}

impl Drop for RouteCensus {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            eprintln!("route census: not written at drop: {e}");
        }
    }
}

/// Threads per block of the census launch.
const CENSUS_THREADS: u32 = 128;
// The entry's launch attributes spell the block as a literal.
const _: () = assert!(CENSUS_THREADS == 128);

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
    /// score_j / divisor · scale` for `j < cut`, 0 for `cut <= j < n_used`; with `has_escale`
    /// each of those is then multiplied by `escale[id_j]`, the learned per-expert scale a
    /// router's weight stage applies to the quotient (with it off `escale` is an ABI borrow of
    /// any buffer of `n_expert` values and is never read). An id not below `n_expert` raises [`FaultSite::ExpertId`] and writes NaN weights; a weight that is not
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
            escale.len() >= n_expert,
            weights.len() >= n_used
        )
    )]
    pub fn route_cut_weights(
        ids: &[u32],
        probs: &[f32],
        escale: &[f32],
        n_expert: u32,
        n_used: u32,
        cut: u32,
        scale: f32,
        floor: u32,
        has_escale: u32,
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
                let w = p / div * scale;
                if has_escale != 0 {
                    // SAFETY: id < n_expert <= escale.len() by the launch contract.
                    w * unsafe { *escale.get_unchecked(id as usize) }
                } else {
                    w
                }
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

    /// The census add of one layer's picks: thread `i < rows * n_used` is slot `i % n_used` of
    /// row `i / n_used`, read at `ids[first + i]` and `weights[first + i]`; a slot below `live` whose id is below `n_expert` adds one to
    /// `picks[layer_off + id]` and its weight to `weight[layer_off + id]` (relaxed atomic adds:
    /// the f32 sum's order is the launch's, a diagnostic's). An id not below `n_expert` adds
    /// nothing; the router's own fault word names it.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            ids.len() >= first + rows * n_used,
            weights.len() >= first + rows * n_used,
            live <= n_used,
            picks.len() >= layer_off + n_expert,
            wsum.len() >= layer_off + n_expert
        )
    )]
    pub fn route_census_add(
        ids: &[u32],
        weights: &[f32],
        n_expert: u32,
        n_used: u32,
        live: u32,
        rows: u32,
        first: u32,
        layer_off: u32,
        mut picks: DisjointSlice<u32>,
        mut wsum: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= rows as usize * n_used as usize || i % n_used as usize >= live as usize {
            return;
        }
        let j = first as usize + i;
        // SAFETY: j < first + rows * n_used <= ids.len() and weights.len() by the launch
        // contract.
        let (id, w) = unsafe { (*ids.get_unchecked(j), *weights.get_unchecked(j)) };
        if id >= n_expert {
            return;
        }
        let at = layer_off as usize + id as usize;
        // SAFETY: at < layer_off + n_expert <= picks.len() and wsum.len() by the launch
        // contract; every access to the two counters is these relaxed atomic adds, and the host
        // reads them only after the stream is synchronized.
        unsafe {
            DeviceAtomicU32::from_ptr(picks.as_mut_ptr().add(at))
                .fetch_add(1, AtomicOrdering::Relaxed);
            DeviceAtomicF32::from_ptr(wsum.as_mut_ptr().add(at))
                .fetch_add(w, AtomicOrdering::Relaxed);
        }
    }
}

/// The dials' three entries (the bias, the cut and the census add), loaded once ([`RouteDialKernels::load`]) by a load whose dials are on.
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
    /// `n_used`; `floor` the router instance's divisor guard; `escale` the router's learned
    /// per-expert scale where its weight stage has one (ERNIE 5.5), applied to the quotient.
    /// One launch of one thread.
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
        escale: Option<&DeviceBuffer<f32>>,
        fault: FaultSink,
        weights: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "route_cut_weights";
        let (ids, probs) = picks;
        if cut == 0
            || cut > n_used
            || ids.len() < n_used
            || probs.len() < n_expert
            || escale.is_some_and(|e| e.len() < n_expert)
            || weights.len() < n_used
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "cut {cut} of {n_used} slots, ids {} and weights {} words, scores {} (and any \
                     scale) of {n_expert} experts: a cut of 1 to the slot count over buffers that \
                     long",
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
            escale.unwrap_or(probs),
            launch_u32(WHAT, "n_expert", n_expert)?,
            launch_u32(WHAT, "n_used", n_used)?,
            launch_u32(WHAT, "cut", cut)?,
            scale,
            u32::from(floor),
            u32::from(escale.is_some()),
            fault,
            weights,
        )?;
        Ok(())
    }
}
