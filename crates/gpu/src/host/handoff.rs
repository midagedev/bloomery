//! The handoff of a token's routed slots to the host tier, on the card
//! ([`HandoffKernels::enqueue_handoff`]): one launch writes the handoff
//! straight into its image in the host-mapped page ([`super::page`]) — the
//! region's sequence word, the router's ids and weights, the activation — so
//! no copy node carries it, and writes each slot's place in the card's routed
//! stacks from the slot map's card copy ([`super::slots`]): the layer's row at
//! the id, [`HOST`] for an expert the card does not hold. The card's routed
//! launches read those places, so which experts the card computes comes from
//! the map alone. An id past the expert count has no place: it raises
//! [`FaultSite::ExpertId`] there.
//!
//! The entries are built per routed slot count a token — `ds41_ffn_handoff`
//! for 6 (DeepSeek-V4.1), `_8` (GLM-5.3-Flash) and `_10` (Qwen3.8), the last
//! two over one body ([`handoff_at`]) — and the launcher picks the entry from
//! the page's slot count ([`HandoffSlots::of`]), refusing any other by name.
//! `ds41_ffn_handoff_10_cols` writes a row of several columns of ten slots in
//! one launch ([`HandoffKernels::enqueue_handoff_cols`]);
//! `ds41_ffn_places_10_cols` writes the places alone of such a row, for a walk
//! whose routing reaches the host by a download
//! ([`HandoffKernels::enqueue_places_cols`]).

use std::fmt;
use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

use super::slots::HOST;
use super::step::HandoffTarget;
use crate::{FaultSink, FaultSite, GpuError, launch_u32};

/// Threads per block of the handoff.
const HANDOFF_THREADS: u32 = 256;

/// Slots a token of the first entry, `ds41_ffn_handoff`, whose contract
/// spells it as a literal.
const N_USED: usize = 6;
/// Slots a token of the `_8` and `_10` entries, whose contracts spell them
/// as literals.
const SLOTS_8: usize = 8;
const SLOTS_10: usize = 10;
// The instance table's counts are the entries' own.
const _: () = assert!(
    HandoffSlots::Six.n() == N_USED
        && HandoffSlots::Eight.n() == SLOTS_8
        && HandoffSlots::Ten.n() == SLOTS_10
);

/// The routed slot counts a token the handoff's entries are built for: one
/// entry per count, picked at launch, and every other count refused by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffSlots {
    /// `ds41_ffn_handoff`.
    Six,
    /// `ds41_ffn_handoff_8`.
    Eight,
    /// `ds41_ffn_handoff_10`.
    Ten,
}

/// A slot count no handoff entry serves ([`HandoffSlots::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandoffRefused {
    pub n_used: usize,
}

impl fmt::Display for HandoffRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} routed slots a token: the handoff's entries are built for 6, 8 and 10",
            self.n_used
        )
    }
}

impl std::error::Error for HandoffRefused {}

impl HandoffSlots {
    /// The entry for `n_used` slots a token, or the count refused.
    pub fn of(n_used: usize) -> Result<HandoffSlots, HandoffRefused> {
        match n_used {
            6 => Ok(HandoffSlots::Six),
            8 => Ok(HandoffSlots::Eight),
            10 => Ok(HandoffSlots::Ten),
            _ => Err(HandoffRefused { n_used }),
        }
    }

    /// The entry's slot count.
    #[must_use]
    pub const fn n(self) -> usize {
        match self {
            HandoffSlots::Six => 6,
            HandoffSlots::Eight => 8,
            HandoffSlots::Ten => 10,
        }
    }
}

#[cuda_module]
mod handoff_kernels {
    use super::*;

    /// The handoff, written straight into its image in the host-mapped page,
    /// and the card's places — one thread per activation value `d < n`,
    /// which copies `x[d]` to the image's word `x_at + d`. Threads `s < 6`
    /// also copy slot `s`'s id and weight to words `ids_at + s` and `wts_at +
    /// s` and write `sel[s] = map[row_off + id]` — the id's slot in the card's
    /// routed stacks, or [`HOST`]. An id not below `n_expert` has no place:
    /// it raises [`FaultSite::ExpertId`] on `fault` and its `sel` is
    /// [`HOST`], so no card kernel reads a row for it; the id itself goes to
    /// the image as it came. Thread 0 copies the region's sequence word to
    /// `seq_at`.
    /// The go that follows orders every write here before its generation.
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
            ids_in.len() >= 6,
            w_in.len() >= 6,
            map.len() >= row_off + n_expert,
            x.len() >= n,
            seq.len() >= 1,
            n >= 6,
            ids_at >= seq_at + 1,
            wts_at >= ids_at + 6,
            x_at >= wts_at + 6,
            image.len() >= x_at + n,
            sel.len() >= 6
        )
    )]
    pub fn ds41_ffn_handoff(
        ids_in: &[u32],
        w_in: &[f32],
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        x: &[f32],
        seq: &[u32],
        n: u32,
        seq_at: u32,
        ids_at: u32,
        wts_at: u32,
        x_at: u32,
        fault: FaultSink,
        mut image: DisjointSlice<u32>,
        mut sel: DisjointSlice<u32>,
    ) {
        let d = thread::index_1d().get();
        if d >= n as usize {
            return;
        }
        // SAFETY: d < n <= x.len(), and x_at + d < x_at + n <= image.len(), by
        // the launch contract; the image's words past x_at are the activation's
        // alone, and thread d is word x_at + d's only writer.
        unsafe {
            *image.get_unchecked_mut(x_at as usize + d) = (*x.get_unchecked(d)).to_bits();
        }
        if d < N_USED {
            // SAFETY: d < 6 <= ids_in.len() and w_in.len() by the launch
            // contract.
            let (id, w) = unsafe { (*ids_in.get_unchecked(d), *w_in.get_unchecked(d)) };
            let place = if id < n_expert {
                // SAFETY: id < n_expert, so row_off + id < map.len() by the
                // launch contract.
                unsafe { *map.get_unchecked(row_off as usize + id as usize) }
            } else {
                fault.raise(FaultSite::ExpertId);
                HOST
            };
            // SAFETY: d < 6 <= sel.len(); ids_at + d and wts_at + d lie in the
            // routing's two spans, which the contract keeps apart from each
            // other, from seq_at and from the activation, all inside the image;
            // thread d is each of those words' only writer.
            unsafe {
                *image.get_unchecked_mut(ids_at as usize + d) = id;
                *image.get_unchecked_mut(wts_at as usize + d) = w.to_bits();
                *sel.get_unchecked_mut(d) = place;
            }
        }
        if d == 0 {
            // SAFETY: seq.len() >= 1, and seq_at < ids_at is inside the image
            // and no other span's word, by the launch contract; thread 0 alone
            // writes it.
            unsafe { *image.get_unchecked_mut(seq_at as usize) = *seq.get_unchecked(0) };
        }
    }

    /// [`ds41_ffn_handoff`] of eight slots a token ([`handoff_at`]): threads
    /// `s < 8` copy slot `s`'s id and weight and write its place.
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
            ids_in.len() >= 8,
            w_in.len() >= 8,
            map.len() >= row_off + n_expert,
            x.len() >= n,
            seq.len() >= 1,
            n >= 8,
            ids_at >= seq_at + 1,
            wts_at >= ids_at + 8,
            x_at >= wts_at + 8,
            image.len() >= x_at + n,
            sel.len() >= 8
        )
    )]
    pub fn ds41_ffn_handoff_8(
        ids_in: &[u32],
        w_in: &[f32],
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        x: &[f32],
        seq: &[u32],
        n: u32,
        seq_at: u32,
        ids_at: u32,
        wts_at: u32,
        x_at: u32,
        fault: FaultSink,
        mut image: DisjointSlice<u32>,
        mut sel: DisjointSlice<u32>,
    ) {
        let d = thread::index_1d().get();
        if d >= n as usize {
            return;
        }
        let a = HandoffIn {
            ids_in,
            w_in,
            map,
            row_off,
            n_expert,
            x,
            seq,
            seq_at,
            ids_at,
            wts_at,
            x_at,
            fault,
        };
        // SAFETY: d < n, and the launch contract is `handoff_at`'s at
        // SLOTS_8 slots; thread d is the launch's only thread at d.
        unsafe { handoff_at::<SLOTS_8>(&a, d, &mut image, &mut sel) };
    }

    /// [`ds41_ffn_handoff`] of ten slots a token ([`handoff_at`]): threads
    /// `s < 10` copy slot `s`'s id and weight and write its place.
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
            ids_in.len() >= 10,
            w_in.len() >= 10,
            map.len() >= row_off + n_expert,
            x.len() >= n,
            seq.len() >= 1,
            n >= 10,
            ids_at >= seq_at + 1,
            wts_at >= ids_at + 10,
            x_at >= wts_at + 10,
            image.len() >= x_at + n,
            sel.len() >= 10
        )
    )]
    pub fn ds41_ffn_handoff_10(
        ids_in: &[u32],
        w_in: &[f32],
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        x: &[f32],
        seq: &[u32],
        n: u32,
        seq_at: u32,
        ids_at: u32,
        wts_at: u32,
        x_at: u32,
        fault: FaultSink,
        mut image: DisjointSlice<u32>,
        mut sel: DisjointSlice<u32>,
    ) {
        let d = thread::index_1d().get();
        if d >= n as usize {
            return;
        }
        let a = HandoffIn {
            ids_in,
            w_in,
            map,
            row_off,
            n_expert,
            x,
            seq,
            seq_at,
            ids_at,
            wts_at,
            x_at,
            fault,
        };
        // SAFETY: d < n, and the launch contract is `handoff_at`'s at
        // SLOTS_10 slots; thread d is the launch's only thread at d.
        unsafe { handoff_at::<SLOTS_10>(&a, d, &mut image, &mut sel) };
    }

    /// The handoff of `m` columns of ten slots in one launch: one thread per
    /// value `d < m·n` of the columns' activations, which copies `x[d]` to
    /// the image's word `x_at + d`; thread `d` of column `c = d / n` whose
    /// value `e = d % n` is below ten also copies column `c`'s slot `e` —
    /// the router's word `c·pitch + e` — to words `ids_at + 10·c + e` and
    /// `wts_at + 10·c + e` and writes `sel[10·c + e]`, the id's place or
    /// [`HOST`] with [`FaultSite::ExpertId`] raised for an id not below
    /// `n_expert`. Thread 0 copies the region's sequence word to `seq_at`.
    /// Each column's words are what [`ds41_ffn_handoff_10`] writes for it
    /// alone at those offsets. The go that follows orders every write here
    /// before its generation.
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
            m >= 1,
            pitch >= 10,
            ids_in.len() >= m * pitch,
            w_in.len() >= m * pitch,
            map.len() >= row_off + n_expert,
            x.len() >= m * n,
            seq.len() >= 1,
            n >= 10,
            ids_at >= seq_at + 1,
            wts_at >= ids_at + m * 10,
            x_at >= wts_at + m * 10,
            image.len() >= x_at + m * n,
            sel.len() >= m * 10
        )
    )]
    pub fn ds41_ffn_handoff_10_cols(
        ids_in: &[u32],
        w_in: &[f32],
        pitch: u32,
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        x: &[f32],
        seq: &[u32],
        n: u32,
        m: u32,
        seq_at: u32,
        ids_at: u32,
        wts_at: u32,
        x_at: u32,
        fault: FaultSink,
        mut image: DisjointSlice<u32>,
        mut sel: DisjointSlice<u32>,
    ) {
        let d = thread::index_1d().get();
        let (n, m) = (n as usize, m as usize);
        if d >= m * n {
            return;
        }
        // SAFETY: d < m·n <= x.len(), and x_at + d < x_at + m·n <=
        // image.len(), by the launch contract; the image's words past x_at
        // are the activations' alone, and thread d is word x_at + d's only
        // writer.
        unsafe {
            *image.get_unchecked_mut(x_at as usize + d) = (*x.get_unchecked(d)).to_bits();
        }
        let (c, e) = (d / n, d % n);
        if e < SLOTS_10 {
            let at = c * pitch as usize + e;
            // SAFETY: c < m and e < 10 <= pitch, so at < m·pitch <=
            // ids_in.len() and w_in.len() by the launch contract.
            let (id, w) = unsafe { (*ids_in.get_unchecked(at), *w_in.get_unchecked(at)) };
            let place = if id < n_expert {
                // SAFETY: id < n_expert, so row_off + id < map.len() by the
                // launch contract.
                unsafe { *map.get_unchecked(row_off as usize + id as usize) }
            } else {
                fault.raise(FaultSite::ExpertId);
                HOST
            };
            let k = SLOTS_10 * c + e;
            // SAFETY: k < 10·m <= sel.len(); ids_at + k and wts_at + k lie
            // in the routing's two spans of 10·m words, which the contract
            // keeps apart from each other, from seq_at and from the
            // activations, all inside the image; thread d is the only one
            // at (c, e), so each word's only writer.
            unsafe {
                *image.get_unchecked_mut(ids_at as usize + k) = id;
                *image.get_unchecked_mut(wts_at as usize + k) = w.to_bits();
                *sel.get_unchecked_mut(k) = place;
            }
        }
        if d == 0 {
            // SAFETY: seq.len() >= 1, and seq_at < ids_at is inside the image
            // and no other span's word, by the launch contract; thread 0 alone
            // writes it.
            unsafe { *image.get_unchecked_mut(seq_at as usize) = *seq.get_unchecked(0) };
        }
    }

    /// The places alone of `m` columns of ten slots, for a walk whose
    /// routing reaches the host by a download instead of the image: one
    /// thread per slot `k < 10·m`, column `c = k / 10`, slot `e = k % 10`,
    /// reads the router's word `c·pitch + e` and writes `sel[k]` — the id's
    /// place, or [`HOST`] with [`FaultSite::ExpertId`] raised for an id not
    /// below `n_expert` ([`ds41_ffn_handoff_10_cols`]'s rule). A word of a
    /// column past its ten (the shared expert's, at a pitch of eleven) is
    /// never read.
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
            m >= 1,
            pitch >= 10,
            ids_in.len() >= m * pitch,
            map.len() >= row_off + n_expert,
            sel.len() >= m * 10
        )
    )]
    pub fn ds41_ffn_places_10_cols(
        ids_in: &[u32],
        pitch: u32,
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        m: u32,
        fault: FaultSink,
        mut sel: DisjointSlice<u32>,
    ) {
        let k = thread::index_1d().get();
        if k >= m as usize * SLOTS_10 {
            return;
        }
        let (c, e) = (k / SLOTS_10, k % SLOTS_10);
        // SAFETY: c < m and e < 10 <= pitch, so c·pitch + e < m·pitch <=
        // ids_in.len() by the launch contract.
        let id = unsafe { *ids_in.get_unchecked(c * pitch as usize + e) };
        let place = if id < n_expert {
            // SAFETY: id < n_expert, so row_off + id < map.len() by the launch
            // contract.
            unsafe { *map.get_unchecked(row_off as usize + id as usize) }
        } else {
            fault.raise(FaultSite::ExpertId);
            HOST
        };
        // SAFETY: k < 10·m <= sel.len() by the launch contract; thread k is
        // sel[k]'s only writer.
        unsafe { *sel.get_unchecked_mut(k) = place };
    }
}

/// What a handoff entry of `N` slots reads ([`handoff_at`]): its arguments
/// but the value count, as [`handoff_kernels::ds41_ffn_handoff`] names them.
struct HandoffIn<'a> {
    ids_in: &'a [u32],
    w_in: &'a [f32],
    map: &'a [u32],
    row_off: u32,
    n_expert: u32,
    x: &'a [f32],
    seq: &'a [u32],
    seq_at: u32,
    ids_at: u32,
    wts_at: u32,
    x_at: u32,
    fault: FaultSink,
}

/// Thread `d`'s part of the handoff of `N` slots a token:
/// `ds41_ffn_handoff`'s rule with `N` in place of six — thread `d` copies
/// `x[d]` to the image's word `x_at + d`; threads `s < N` copy slot `s`'s id
/// and weight to words `ids_at + s` and `wts_at + s` and write `sel[s]`, the
/// id's place or [`HOST`] with [`FaultSite::ExpertId`] raised for an id not
/// below `n_expert`; thread 0 copies the sequence word.
///
/// SAFETY: `d < n`, and no other thread of the launch runs `d`;
/// `ids_in.len()`, `w_in.len()` and `sel.len() >= N`, `n >= N`,
/// `map.len() >= row_off + n_expert`, `x.len() >= n`, `seq.len() >= 1`,
/// `ids_at >= seq_at + 1`, `wts_at >= ids_at + N`, `x_at >= wts_at + N` and
/// `image.len() >= x_at + n`.
#[inline(always)]
unsafe fn handoff_at<const N: usize>(
    a: &HandoffIn<'_>,
    d: usize,
    image: &mut DisjointSlice<u32>,
    sel: &mut DisjointSlice<u32>,
) {
    // SAFETY: d < n <= x.len(), and x_at + d < x_at + n <= image.len(), by
    // this fn's contract; the image's words past x_at are the activation's
    // alone, and thread d is word x_at + d's only writer.
    unsafe {
        *image.get_unchecked_mut(a.x_at as usize + d) = (*a.x.get_unchecked(d)).to_bits();
    }
    if d < N {
        // SAFETY: d < N <= ids_in.len() and w_in.len() by this fn's contract.
        let (id, w) = unsafe { (*a.ids_in.get_unchecked(d), *a.w_in.get_unchecked(d)) };
        let place = if id < a.n_expert {
            // SAFETY: id < n_expert, so row_off + id < map.len() by this fn's
            // contract.
            unsafe { *a.map.get_unchecked(a.row_off as usize + id as usize) }
        } else {
            a.fault.raise(FaultSite::ExpertId);
            HOST
        };
        // SAFETY: d < N <= sel.len(); ids_at + d and wts_at + d lie in the
        // routing's two spans, which the contract keeps apart from each
        // other, from seq_at and from the activation, all inside the image;
        // thread d is each of those words' only writer.
        unsafe {
            *image.get_unchecked_mut(a.ids_at as usize + d) = id;
            *image.get_unchecked_mut(a.wts_at as usize + d) = w.to_bits();
            *sel.get_unchecked_mut(d) = place;
        }
    }
    if d == 0 {
        // SAFETY: seq.len() >= 1, and seq_at < ids_at is inside the image and
        // no other span's word, by this fn's contract; thread 0 alone writes
        // it.
        unsafe { *image.get_unchecked_mut(a.seq_at as usize) = *a.seq.get_unchecked(0) };
    }
}

/// What [`HandoffKernels::enqueue_handoff`] reads: a token's routed ids and
/// weights (the target layout's `n_used` each), and the slot map's card copy
/// with the layer's row at `row_off` (`n_expert` places a row).
pub struct Handoff<'a> {
    pub ids: &'a DeviceBuffer<u32>,
    pub weights: &'a DeviceBuffer<f32>,
    pub map: &'a DeviceBuffer<u32>,
    pub row_off: usize,
    pub n_expert: usize,
}

/// The handoff's kernels, loaded once by a program whose layers hand their
/// routed slots to the host tier, or alone by a gate.
pub struct HandoffKernels {
    module: handoff_kernels::LoadedModule,
}

impl HandoffKernels {
    /// Load the handoff's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<HandoffKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { handoff_kernels::load(ctx)? };
        Ok(HandoffKernels { module })
    }

    /// Enqueue the handoff of `target.layout.n_used` slots a token
    /// (`ds41_ffn_handoff` for 6, `ds41_ffn_handoff_8` for 8,
    /// `ds41_ffn_handoff_10` for 10; any other count is refused by name,
    /// [`HandoffSlots::of`]): `h`'s routing and `target`'s activation into
    /// `target`'s image, with the region's sequence word, and each slot's
    /// place into `sel`. An id past the stack raises [`FaultSite::ExpertId`]
    /// on `fault` and its place is [`HOST`]. One launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_handoff(
        &self,
        stream: &CudaStream,
        h: &Handoff<'_>,
        target: HandoffTarget<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let lay = target.layout;
        let slots = HandoffSlots::of(lay.n_used).map_err(|e| GpuError::Shape {
            what: "ds41_ffn_handoff",
            detail: e.to_string(),
        })?;
        let what = match slots {
            HandoffSlots::Six => "ds41_ffn_handoff",
            HandoffSlots::Eight => "ds41_ffn_handoff_8",
            HandoffSlots::Ten => "ds41_ffn_handoff_10",
        };
        let n = lay.hidden;
        let grid = launch_u32(what, "grid", n.div_ceil(HANDOFF_THREADS as usize))?;
        let cfg = LaunchConfig1D::new(grid, HANDOFF_THREADS, 0);
        let (row_off, n_expert) = (
            launch_u32(what, "row_off", h.row_off)?,
            launch_u32(what, "n_expert", h.n_expert)?,
        );
        let at = [
            launch_u32(what, "n", n)?,
            launch_u32(what, "seq_at", lay.seq)?,
            launch_u32(what, "ids_at", lay.ids)?,
            launch_u32(what, "wts_at", lay.weights)?,
            launch_u32(what, "x_at", lay.x)?,
        ];
        match slots {
            HandoffSlots::Six => {
                let prep = self.module.prepare_ds41_ffn_handoff(cfg)?;
                self.module.ds41_ffn_handoff(
                    stream,
                    &prep,
                    h.ids,
                    h.weights,
                    h.map,
                    row_off,
                    n_expert,
                    target.x,
                    target.seq,
                    at[0],
                    at[1],
                    at[2],
                    at[3],
                    at[4],
                    fault,
                    target.image,
                    sel,
                )?;
            }
            HandoffSlots::Eight => {
                let prep = self.module.prepare_ds41_ffn_handoff_8(cfg)?;
                self.module.ds41_ffn_handoff_8(
                    stream,
                    &prep,
                    h.ids,
                    h.weights,
                    h.map,
                    row_off,
                    n_expert,
                    target.x,
                    target.seq,
                    at[0],
                    at[1],
                    at[2],
                    at[3],
                    at[4],
                    fault,
                    target.image,
                    sel,
                )?;
            }
            HandoffSlots::Ten => {
                let prep = self.module.prepare_ds41_ffn_handoff_10(cfg)?;
                self.module.ds41_ffn_handoff_10(
                    stream,
                    &prep,
                    h.ids,
                    h.weights,
                    h.map,
                    row_off,
                    n_expert,
                    target.x,
                    target.seq,
                    at[0],
                    at[1],
                    at[2],
                    at[3],
                    at[4],
                    fault,
                    target.image,
                    sel,
                )?;
            }
        }
        Ok(())
    }
}

impl HandoffKernels {
    /// Enqueue the handoff of `m` columns of ten slots in one launch
    /// (`ds41_ffn_handoff_10_cols`): column `c`'s routing — `h`'s ids and
    /// weights from word `c·pitch`, `pitch` words a column — and its
    /// activation, `target.x`'s values `c·hidden ..`, into `target`'s image
    /// at the layout's offsets plus `c·n_used` and `c·hidden`, the region's
    /// sequence word, and each slot's place into `sel` (`10·m`). A layout of
    /// another slot count, `m` of 0, and a pitch under ten are refused by
    /// name. Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "one launch's routing, its pitch and width, the target, the sink and the places"
    )]
    pub fn enqueue_handoff_cols(
        &self,
        stream: &CudaStream,
        h: &Handoff<'_>,
        pitch: usize,
        m: usize,
        target: HandoffTarget<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "ds41_ffn_handoff_10_cols";
        let lay = target.layout;
        if HandoffSlots::of(lay.n_used) != Ok(HandoffSlots::Ten) || m == 0 || pitch < SLOTS_10 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{} slots a column, {m} columns, a pitch of {pitch}: the entry is built for \
                     ten slots, one column or more, a pitch of at least ten",
                    lay.n_used
                ),
            });
        }
        let n = lay.hidden;
        let grid = launch_u32(WHAT, "grid", (m * n).div_ceil(HANDOFF_THREADS as usize))?;
        let cfg = LaunchConfig1D::new(grid, HANDOFF_THREADS, 0);
        let prep = self.module.prepare_ds41_ffn_handoff_10_cols(cfg)?;
        self.module.ds41_ffn_handoff_10_cols(
            stream,
            &prep,
            h.ids,
            h.weights,
            launch_u32(WHAT, "pitch", pitch)?,
            h.map,
            launch_u32(WHAT, "row_off", h.row_off)?,
            launch_u32(WHAT, "n_expert", h.n_expert)?,
            target.x,
            target.seq,
            launch_u32(WHAT, "n", n)?,
            launch_u32(WHAT, "m", m)?,
            launch_u32(WHAT, "seq_at", lay.seq)?,
            launch_u32(WHAT, "ids_at", lay.ids)?,
            launch_u32(WHAT, "wts_at", lay.weights)?,
            launch_u32(WHAT, "x_at", lay.x)?,
            fault,
            target.image,
            sel,
        )?;
        Ok(())
    }

    /// Enqueue the places alone of `m` columns of ten slots
    /// (`ds41_ffn_places_10_cols`): column `c`'s slot `e` — `p.ids`'s word
    /// `c·pitch + e` — into `sel[10·c + e]`, its place in `p.map`'s row at
    /// `p.row_off`, or [`HOST`] with [`FaultSite::ExpertId`] raised on
    /// `fault` for an id not below `p.n_expert`; no image, no sequence word.
    /// `m` of 0, a pitch under ten, and buffers short of `m` columns are
    /// refused by name. One launch. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_places_cols(
        &self,
        stream: &CudaStream,
        p: &Places<'_>,
        pitch: usize,
        m: usize,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "ds41_ffn_places_10_cols";
        if m == 0
            || pitch < SLOTS_10
            || p.ids.len() < m * pitch
            || sel.len() < m * SLOTS_10
            || p.map.len() < p.row_off + p.n_expert
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{m} columns at a pitch of {pitch} over {} ids into {} places, a map of {} \
                     words read from {} for {} experts: one column or more, a pitch of at least \
                     ten, and every buffer that long",
                    p.ids.len(),
                    sel.len(),
                    p.map.len(),
                    p.row_off,
                    p.n_expert
                ),
            });
        }
        let grid = launch_u32(
            WHAT,
            "grid",
            (m * SLOTS_10).div_ceil(HANDOFF_THREADS as usize),
        )?;
        let prep = self
            .module
            .prepare_ds41_ffn_places_10_cols(LaunchConfig1D::new(grid, HANDOFF_THREADS, 0))?;
        self.module.ds41_ffn_places_10_cols(
            stream,
            &prep,
            p.ids,
            launch_u32(WHAT, "pitch", pitch)?,
            p.map,
            launch_u32(WHAT, "row_off", p.row_off)?,
            launch_u32(WHAT, "n_expert", p.n_expert)?,
            launch_u32(WHAT, "m", m)?,
            fault,
            sel,
        )?;
        Ok(())
    }
}

/// What [`HandoffKernels::enqueue_places_cols`] reads: the router's ids and
/// the slot map's card copy with the layer's row at `row_off` (`n_expert`
/// places a row).
pub struct Places<'a> {
    pub ids: &'a DeviceBuffer<u32>,
    pub map: &'a DeviceBuffer<u32>,
    pub row_off: usize,
    pub n_expert: usize,
}

#[cfg(test)]
mod tests {
    use super::{HandoffRefused, HandoffSlots};

    #[test]
    fn slots_of_takes_six_eight_and_ten_only() {
        for n in 0..=16 {
            match HandoffSlots::of(n) {
                Ok(s) => {
                    assert!(n == 6 || n == 8 || n == 10, "{n} slots picked an entry");
                    assert_eq!(s.n(), n);
                }
                Err(e) => {
                    assert!(n != 6 && n != 8 && n != 10, "{n} slots refused");
                    assert_eq!(e, HandoffRefused { n_used: n });
                    assert!(e.to_string().starts_with(&format!("{n} routed slots")));
                }
            }
        }
    }
}
