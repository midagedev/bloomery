//! The two launches a placed load of the family's chain (`placed`) runs
//! around its host leg that no other module owns: each routed slot's place
//! in the card's stacks ([`HostLegKernels::enqueue_places`]), and the
//! combine of the card's slots with the host tier's routed sum
//! ([`HostLegKernels::enqueue_combine`]).
//!
//! The places, one thread per slot of `m` tokens at `pitch` slots a token
//! (the router's layout: the routed `used`, then a gated router's shared
//! slot): slot `j < used` of token `t` is `sel[t·pitch + j] = map[row_off +
//! id]`, the id's place in the card's stacks from the slot map's card copy
//! — or [`HOST`] for an expert the host serves — and every slot `j >= used`
//! is [`HOST`], so no routed launch reads a row for the shared slot, which
//! runs on its own stacks. An id not below `n_expert` has no place: it
//! raises [`FaultSite::ExpertId`] and its place is [`HOST`].
//!
//! The combine, one thread per output value `d` of token `t`: `y[t·rows +
//! d] = (card + hsum[t·rows + d]) + resid[t·rows + d]`, `card` the
//! slot-ascending sum of `w[s] · down[s·rows + d]` over the token's slots
//! whose place is below `n_card` — the card's — and, with the shared
//! expert, `+ w[t·pitch + used] · sh[t·rows + d]` last: the reference's
//! `(routed + ffn_inp) + shexp`, the routed sum split into the card's and
//! the host's. A slot at [`HOST`] is the host's and adds nothing here; a
//! place in `[n_card, HOST)` is no expert either side serves: it raises
//! [`FaultSite::ExpertId`] and the token's value is NaN.

use crate::fault::{FaultSink, FaultSite};
use crate::hybrid::HOST;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block of both entries.
const THREADS: u32 = 256;

/// The card slots' weighted sum of value `d` of token `t` (module doc), in
/// slot order: NaN, with [`FaultSite::ExpertId`] raised, when a place is in
/// `[n_card, HOST)`.
///
/// # Safety
///
/// `t < m`, `d < rows`, `sel.len()` and `w.len() >= (t + 1)·pitch` and
/// `down.len() >= (t + 1)·pitch·rows`.
#[inline(always)]
unsafe fn card_sum(
    down: &[f32],
    w: &[f32],
    sel: &[u32],
    (rows, pitch, n_card): (usize, usize, u32),
    t: usize,
    d: usize,
    fault: FaultSink,
) -> f32 {
    let mut acc = 0.0f32;
    let mut j = 0usize;
    while j < pitch {
        let s = t * pitch + j;
        // SAFETY: s < (t + 1)·pitch, inside sel and w by this fn's contract.
        let place = unsafe { *sel.get_unchecked(s) };
        if place < n_card {
            // SAFETY: as above, and s·rows + d < (t + 1)·pitch·rows <=
            // down.len().
            acc += unsafe { *w.get_unchecked(s) * *down.get_unchecked(s * rows + d) };
        } else if place != HOST {
            fault.raise(FaultSite::ExpertId);
            acc = f32::NAN;
        }
        j += 1;
    }
    acc
}

#[cuda_module]
mod qwen3moe_hostleg_kernels {
    use super::*;

    /// Each slot's place (module doc): thread `k < m·pitch` writes
    /// `sel[k]`.
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
            used <= pitch,
            ids.len() >= m * pitch,
            map.len() >= row_off + n_expert,
            sel.len() >= m * pitch
        )
    )]
    pub fn qwen3moe_places(
        ids: &[u32],
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        pitch: u32,
        used: u32,
        m: u32,
        fault: FaultSink,
        mut sel: DisjointSlice<u32>,
    ) {
        let k = thread::index_1d().get();
        if k >= m as usize * pitch as usize {
            return;
        }
        let place = if k % pitch as usize >= used as usize {
            HOST
        } else {
            // SAFETY: k < m·pitch <= ids.len() by the launch contract.
            let id = unsafe { *ids.get_unchecked(k) };
            if id < n_expert {
                // SAFETY: id < n_expert, so row_off + id < map.len() by the
                // launch contract.
                unsafe { *map.get_unchecked(row_off as usize + id as usize) }
            } else {
                fault.raise(FaultSite::ExpertId);
                HOST
            }
        };
        // SAFETY: k < m·pitch <= sel.len() by the launch contract; thread k
        // is sel[k]'s only writer.
        unsafe { *sel.get_unchecked_mut(k) = place };
    }

    /// The combine without a shared expert (module doc): thread `i <
    /// rows·m` writes `y[i]`.
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
            down.len() >= rows * pitch * m,
            w.len() >= pitch * m,
            sel.len() >= pitch * m,
            hsum.len() >= rows * m,
            resid.len() >= rows * m,
            y.len() >= rows * m
        )
    )]
    pub fn qwen3moe_combine_host(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        hsum: &[f32],
        resid: &[f32],
        rows: u32,
        pitch: u32,
        n_card: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let rows_u = rows as usize;
        if i >= rows_u * m as usize {
            return;
        }
        let t = i / rows_u;
        // SAFETY: i < rows·m gives t < m and i − t·rows < rows; the launch
        // contract bounds sel, w and down for every token below m.
        let card = unsafe {
            card_sum(
                down,
                w,
                sel,
                (rows_u, pitch as usize, n_card),
                t,
                i - t * rows_u,
                fault,
            )
        };
        // SAFETY: i < rows·m bounds the hsum and resid reads and the y store
        // by the launch contract; thread i is y[i]'s only writer.
        unsafe {
            let v = (card + *hsum.get_unchecked(i)) + *resid.get_unchecked(i);
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// The combine with the shared expert's output `sh` and its weight at
    /// slot `used` of each token (module doc): thread `i < rows·m` writes
    /// `y[i]`.
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
            used < pitch,
            down.len() >= rows * pitch * m,
            w.len() >= pitch * m,
            sel.len() >= pitch * m,
            hsum.len() >= rows * m,
            resid.len() >= rows * m,
            sh.len() >= rows * m,
            y.len() >= rows * m
        )
    )]
    pub fn qwen3moe_combine_host_sh(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        hsum: &[f32],
        resid: &[f32],
        sh: &[f32],
        rows: u32,
        pitch: u32,
        used: u32,
        n_card: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let rows_u = rows as usize;
        if i >= rows_u * m as usize {
            return;
        }
        let t = i / rows_u;
        let pitch_u = pitch as usize;
        // SAFETY: as qwen3moe_combine_host's.
        let card = unsafe {
            card_sum(
                down,
                w,
                sel,
                (rows_u, pitch_u, n_card),
                t,
                i - t * rows_u,
                fault,
            )
        };
        // SAFETY: i < rows·m bounds hsum, resid, sh and y; t·pitch + used <
        // (t + 1)·pitch <= w.len() since used < pitch; thread i is y[i]'s
        // only writer.
        unsafe {
            let w_sh = *w.get_unchecked(t * pitch_u + used as usize);
            let v = ((card + *hsum.get_unchecked(i)) + *resid.get_unchecked(i))
                + w_sh * *sh.get_unchecked(i);
            *y.get_unchecked_mut(i) = v;
        }
    }
}

/// [`HostLegKernels::enqueue_places`]'s arguments: the router's ids
/// (`pitch` a token, the routed `used` first), the slot map's card copy with
/// the layer's row at `row_off` (`n_expert` places a row), and `m` tokens.
pub(super) struct PlacesArgs<'a> {
    pub(super) ids: &'a DeviceBuffer<u32>,
    pub(super) map: &'a DeviceBuffer<u32>,
    pub(super) row_off: usize,
    pub(super) n_expert: usize,
    pub(super) pitch: usize,
    pub(super) used: usize,
    pub(super) m: usize,
}

/// [`HostLegKernels::enqueue_combine`]'s arguments: the down outputs (per
/// token slot-major, `pitch · rows` a token), the router's weights and the
/// slots' places (`pitch` a token), the host tier's routed sums, the
/// residual and the output (token-major, `rows` a token), the card's
/// experts of the layer, and with the shared expert its output (token-major)
/// at weight slot `used`.
pub(super) struct HostCombineArgs<'a> {
    pub(super) down: &'a DeviceBuffer<f32>,
    pub(super) w: &'a DeviceBuffer<f32>,
    pub(super) sel: &'a DeviceBuffer<u32>,
    pub(super) hsum: &'a DeviceBuffer<f32>,
    pub(super) resid: &'a DeviceBuffer<f32>,
    pub(super) shared: Option<(&'a DeviceBuffer<f32>, usize)>,
    pub(super) rows: usize,
    pub(super) pitch: usize,
    pub(super) n_card: usize,
    pub(super) m: usize,
    pub(super) fault: FaultSink,
    pub(super) y: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub(super) struct HostLegKernels {
    module: qwen3moe_hostleg_kernels::LoadedModule,
}

impl HostLegKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub(super) fn load(ctx: &Arc<CudaContext>) -> Result<HostLegKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(qwen3moe_hostleg_kernels, ctx)? };
        Ok(HostLegKernels { module })
    }

    /// Enqueue the places of `a.m` tokens' slots into `sel` (module doc). A
    /// shape whose buffers are short, no token, or more routed slots than
    /// the pitch is refused by name. Asynchronous, allocation-free,
    /// capturable.
    pub(super) fn enqueue_places(
        &self,
        stream: &CudaStream,
        a: &PlacesArgs<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "qwen3moe_places";
        let n = a.m * a.pitch;
        if a.m == 0
            || a.used == 0
            || a.used > a.pitch
            || a.ids.len() < n
            || sel.len() < n
            || a.map.len() < a.row_off + a.n_expert
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} tokens of {} routed slots at a pitch of {} over {} ids into {} places, a \
                     map of {} words read from {} for {} experts: a token or more, the routed \
                     slots within the pitch, and every buffer that long",
                    a.m,
                    a.used,
                    a.pitch,
                    a.ids.len(),
                    sel.len(),
                    a.map.len(),
                    a.row_off,
                    a.n_expert
                ),
            ));
        }
        let grid = launch_u32(WHAT, "grid", n.div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_qwen3moe_places(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.qwen3moe_places(
            stream,
            &prep,
            a.ids,
            a.map,
            launch_u32(WHAT, "row_off", a.row_off)?,
            launch_u32(WHAT, "n_expert", a.n_expert)?,
            launch_u32(WHAT, "pitch", a.pitch)?,
            launch_u32(WHAT, "used", a.used)?,
            launch_u32(WHAT, "m", a.m)?,
            fault,
            sel,
        )?;
        Ok(())
    }

    /// Enqueue the combine of `a.m` tokens (module doc): the card's slots,
    /// the host's sum, the residual and, when `a.shared` is set, the shared
    /// expert. A shape whose buffers are short, or a shared slot outside
    /// the pitch, is refused by name. Asynchronous, allocation-free,
    /// capturable.
    pub(super) fn enqueue_combine(
        &self,
        stream: &CudaStream,
        a: HostCombineArgs<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "qwen3moe_combine_host";
        let HostCombineArgs {
            down,
            w,
            sel,
            hsum,
            resid,
            shared,
            rows,
            pitch,
            n_card,
            m,
            fault,
            y,
        } = a;
        let short = rows == 0
            || pitch == 0
            || m == 0
            || down.len() < rows * pitch * m
            || w.len() < pitch * m
            || sel.len() < pitch * m
            || hsum.len() < rows * m
            || resid.len() < rows * m
            || y.len() < rows * m
            || shared.is_some_and(|(sh, slot)| slot >= pitch || sh.len() < rows * m);
        if short {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "rows {rows}, pitch {pitch}, m {m}, shared slot {:?}: down.len() {} w.len() {} \
                     sel.len() {} hsum.len() {} resid.len() {} y.len() {} shared.len() {:?}",
                    shared.map(|(_, s)| s),
                    down.len(),
                    w.len(),
                    sel.len(),
                    hsum.len(),
                    resid.len(),
                    y.len(),
                    shared.map(|(sh, _)| sh.len())
                ),
            ));
        }
        let cfg = LaunchConfig1D::new(
            launch_u32(WHAT, "grid", (rows * m).div_ceil(THREADS as usize))?,
            THREADS,
            0,
        );
        let (rows, pitch, n_card, m) = (
            launch_u32(WHAT, "rows", rows)?,
            launch_u32(WHAT, "pitch", pitch)?,
            launch_u32(WHAT, "n_card", n_card)?,
            launch_u32(WHAT, "m", m)?,
        );
        match shared {
            None => {
                let prep = self.module.prepare_qwen3moe_combine_host(cfg)?;
                self.module.qwen3moe_combine_host(
                    stream, &prep, down, w, sel, hsum, resid, rows, pitch, n_card, m, fault, y,
                )?;
            }
            Some((sh, slot)) => {
                let prep = self.module.prepare_qwen3moe_combine_host_sh(cfg)?;
                self.module.qwen3moe_combine_host_sh(
                    stream,
                    &prep,
                    down,
                    w,
                    sel,
                    hsum,
                    resid,
                    sh,
                    rows,
                    pitch,
                    launch_u32(WHAT, "used", slot)?,
                    n_card,
                    m,
                    fault,
                    y,
                )?;
            }
        }
        Ok(())
    }
}
