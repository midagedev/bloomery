//! The batch port: an eager layer-batch's host leg, outside the go/wait
//! protocol. The card copies a block's activations and routing to the host
//! behind an event ([`BatchPort::download`]); the calling thread waits on
//! that event and serves the block's host experts in one union call
//! ([`BatchPort::serve`], through [`BatchService`]); one copy sends the sums
//! back ([`BatchPort::upload`]). The port holds two sets, taken in turn, so a
//! layer-batch's route copies into one set while the union still reads the
//! other.

use super::slots::{HOST, SlotMap};
use super::{Health, HostExperts, Refusal, name_refusal, non_finite, unknown_id};
use crate::GpuError;
use crate::graph::cu;
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer, sys};
use model::{Tensor2, Tensor2View};
use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

/// The name every batch service's errors and refusals carry.
pub(super) const SERVE_BATCH: &str = "Hybrid::serve_batch";

/// What the port's refusals name.
const PORT: &str = "BatchPort";

/// Which layer-batch a set holds ([`BatchPort::download`],
/// [`BatchPort::serve`], [`BatchPort::upload`]): model layer `layer` of the
/// group's batch `set`, the batch's tokens `at .. u` of its block. Two
/// batches' blocks can cover the same tokens of their batches; their keys
/// still differ, so a serve or an upload out of the route order is refused by
/// name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchKey {
    pub layer: usize,
    pub set: usize,
    pub at: usize,
    pub u: usize,
}

/// [`BatchPort::serve`]'s host time outside the union call, in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServeTimes {
    /// Waiting on the route's copies to the host.
    pub wait_ns: u64,
    /// Copying the activations into the union's view.
    pub copy_ns: u64,
}

/// Where a set stands: free, holding a layer-batch's route copies, or
/// holding its host sums before their upload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Free,
    Routed(BatchKey),
    Served(BatchKey),
}

/// One set: page-locked copies of a block's activations, routing and host
/// sums, the event its route's copies complete at, and its stage.
struct Set {
    x: PinnedHostBuffer<f32>,
    ids: PinnedHostBuffer<u32>,
    w: PinnedHostBuffer<f32>,
    sum: PinnedHostBuffer<f32>,
    routed: CudaEvent,
    stage: Stage,
}

/// The two sets, and the set the next download, serve and upload take: each
/// of the three goes through the sets in turn, so a serve takes the set of
/// the oldest download not served yet.
///
/// Each set owns the event its download records, and a serve waits on the
/// event of the set whose buffers it reads: a serve that waited on a shared
/// event would wait for the next batch's route too — the same bits with the
/// overlap gone. Nothing outside this port names a set's event.
pub struct BatchPort {
    sets: [Set; 2],
    n_embd: usize,
    n_used: usize,
    down: usize,
    serve: usize,
    up: usize,
}

impl BatchPort {
    /// Two sets for blocks of up to `cap` tokens of rows of `n_embd`, each
    /// routed to `n_used` experts. Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        n_embd: usize,
        n_used: usize,
        cap: usize,
    ) -> Result<BatchPort, GpuError> {
        let pinned = |what: &'static str, n: usize| {
            PinnedHostBuffer::<f32>::zeroed(ctx, n).map_err(|source| GpuError::Driver {
                op: Some(what),
                source,
            })
        };
        let set = || -> Result<Set, GpuError> {
            Ok(Set {
                x: pinned("cuMemAllocHost (the batch's activations)", cap * n_embd)?,
                ids: PinnedHostBuffer::<u32>::zeroed(ctx, cap * n_used).map_err(|source| {
                    GpuError::Driver {
                        op: Some("cuMemAllocHost (the batch's routing)"),
                        source,
                    }
                })?,
                w: pinned("cuMemAllocHost (the batch's routing)", cap * n_used)?,
                sum: pinned("cuMemAllocHost (the batch's host sums)", cap * n_embd)?,
                routed: ctx.new_event(None)?,
                stage: Stage::Free,
            })
        };
        Ok(BatchPort {
            sets: [set()?, set()?],
            n_embd,
            n_used,
            down: 0,
            serve: 0,
            up: 0,
        })
    }

    /// Both sets free, the next download into the first: at a group's
    /// start, when the stream holds no copy of an earlier group (the group's
    /// prologue waits for the stream).
    pub fn begin(&mut self) {
        for s in &mut self.sets {
            s.stage = Stage::Free;
        }
        (self.down, self.serve, self.up) = (0, 0, 0);
    }

    /// Enqueue the copies of `key`'s tokens `at .. u` of `x` (`n_embd` a
    /// token), `ids` and `w` (`n_used` a token) into the next set, and its
    /// event. Refused while that set holds a layer not uploaded yet.
    pub fn download(
        &mut self,
        stream: &CudaStream,
        [x, w]: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        let (n, s) = (self.n_embd, self.n_used);
        let BatchKey { at, u, .. } = key;
        let set = &mut self.sets[self.down];
        if set.stage != Stage::Free {
            return Err(GpuError::Shape {
                what: PORT,
                detail: format!(
                    "a route of {key:?} into an exchange set that holds {:?}: both sets hold a \
                     layer not uploaded yet",
                    set.stage
                ),
            });
        }
        // SAFETY: each copy reads the first values of a device buffer and
        // writes as many into this set's page-locked buffers (`dtoh` checks
        // both lengths). The set is free: its last serve returned, so the
        // union no longer reads them, and its upload is enqueued before these
        // copies. The host reads them again only in this set's next serve,
        // after its wait on the event recorded below.
        unsafe {
            dtoh(stream, &mut set.x, x, at * n..u * n)?;
            dtoh(stream, &mut set.ids, ids, at * s..u * s)?;
            dtoh(stream, &mut set.w, w, at * s..u * s)?;
        }
        set.routed.record(stream)?;
        set.stage = Stage::Routed(key);
        self.down ^= 1;
        Ok(())
    }

    /// Wait for the oldest unserved set's copies, which must be `key`'s,
    /// then have `serve` compute its layer's host sums for its tokens: the
    /// layer, the activations as a view, the routing, and the set's sums the
    /// upload sends.
    pub fn serve(
        &mut self,
        key: BatchKey,
        serve: impl FnOnce(usize, Tensor2View<'_>, &[u32], &[f32], &mut [f32]) -> Result<(), GpuError>,
    ) -> Result<ServeTimes, GpuError> {
        let (n, s) = (self.n_embd, self.n_used);
        let BatchKey { layer, at, u, .. } = key;
        let set = &mut self.sets[self.serve];
        if set.stage != Stage::Routed(key) {
            return Err(GpuError::Shape {
                what: PORT,
                detail: format!(
                    "the serve of {key:?}; the oldest unserved exchange set holds {:?}",
                    set.stage
                ),
            });
        }
        let t0 = Instant::now();
        set.routed.synchronize()?;
        let t1 = Instant::now();
        let x = Tensor2View::new(&set.x[at * n..u * n], n, u - at)?;
        let times = ServeTimes {
            wait_ns: nanos(t1 - t0),
            copy_ns: nanos(t1.elapsed()),
        };
        serve(
            layer,
            x,
            &set.ids[at * s..u * s],
            &set.w[at * s..u * s],
            &mut set.sum[at * n..u * n],
        )?;
        set.stage = Stage::Served(key);
        self.serve ^= 1;
        Ok(times)
    }

    /// Enqueue the copy of the oldest served set's sums, which must be
    /// `key`'s, to `hsum`; the set is free again.
    pub fn upload(
        &mut self,
        stream: &CudaStream,
        hsum: &mut DeviceBuffer<f32>,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        let n = self.n_embd;
        let BatchKey { at, u, .. } = key;
        let set = &mut self.sets[self.up];
        if set.stage != Stage::Served(key) {
            return Err(GpuError::Shape {
                what: PORT,
                detail: format!(
                    "the upload of {key:?}; the oldest served exchange set holds {:?}",
                    set.stage
                ),
            });
        }
        // SAFETY: the copy writes values at·n .. u·n of `hsum` from this
        // set's page-locked sums, which the host writes again only in this
        // set's next serve, after a wait on the event of its next download —
        // enqueued after this copy, since the set is free from here on.
        unsafe { htod(stream, hsum, &set.sum, at * n..u * n)? };
        set.stage = Stage::Free;
        self.up ^= 1;
        Ok(())
    }
}

/// `d` in whole nanoseconds, saturating.
fn nanos(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// Enqueue the copy of values `at` of `src` to the same values of `dst`.
///
/// SAFETY: `dst` is not read or freed until the copy completes.
unsafe fn dtoh<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    dst: &mut PinnedHostBuffer<T>,
    src: &DeviceBuffer<T>,
    at: Range<usize>,
) -> Result<(), GpuError> {
    if at.is_empty() || at.end > src.len() || at.end > dst.len() {
        return Err(GpuError::Shape {
            what: PORT,
            detail: format!("values {at:?} from {} into {}", src.len(), dst.len()),
        });
    }
    // SAFETY: both buffers hold the values of `at` (checked above); `dst`
    // stays untouched until the copy completes by this fn's contract.
    let rc = unsafe {
        sys::cuMemcpyDtoHAsync_v2(
            dst.as_mut_ptr().add(at.start).cast(),
            src.cu_deviceptr() + (at.start * size_of::<T>()) as u64,
            at.len() * size_of::<T>(),
            stream.cu_stream(),
        )
    };
    cu(rc, "cuMemcpyDtoHAsync_v2 (the batch's handoffs)")
}

/// Enqueue the copy of values `at` of `src` to the same values of `dst`.
///
/// SAFETY: `src` is not written or freed until the copy completes.
unsafe fn htod<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    dst: &mut DeviceBuffer<T>,
    src: &PinnedHostBuffer<T>,
    at: Range<usize>,
) -> Result<(), GpuError> {
    if at.is_empty() || at.end > src.len() || at.end > dst.len() {
        return Err(GpuError::Shape {
            what: PORT,
            detail: format!("values {at:?} from {} into {}", src.len(), dst.len()),
        });
    }
    // SAFETY: both buffers hold the values of `at` (checked above); `src`
    // stays unwritten until the copy completes by this fn's contract.
    let rc = unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            dst.cu_deviceptr() + (at.start * size_of::<T>()) as u64,
            src.as_ptr().add(at.start).cast(),
            at.len() * size_of::<T>(),
            stream.cu_stream(),
        )
    };
    cu(rc, "cuMemcpyHtoDAsync_v2 (the batch's host sums)")
}

// ----------------------------------------------------------------- service

/// What batch services have done since load, summed. The fields are
/// [`super::HybridStats`]'s of the same names.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BatchStats {
    pub(super) batch_served: u64,
    pub(super) batch_cols: u64,
    pub(super) batch_host_slots: u64,
    pub(super) batch_ns: u64,
    pub(super) batch_excluded_slots: u64,
}

/// A batch service's scratch and counters: the host lists, the boundary's
/// `n_used` entries a column, each column's length and the refused columns,
/// grown to the widest batch served, once.
#[derive(Default)]
pub(super) struct BatchService {
    lists: Vec<(u32, f32)>,
    lens: Vec<usize>,
    /// The refused columns, ascending: the loop that refuses a column
    /// records it here, and the zero and NaN fills read only this record.
    refused: Vec<usize>,
    /// The storage of the per-column list slices, empty between services
    /// ([`reuse_slices`]).
    slices: Vec<&'static [(u32, f32)]>,
    pub(super) stats: BatchStats,
}

/// What one batch service reads besides its columns: the tier's parts.
pub(super) struct Tier<'a, H> {
    pub(super) experts: &'a mut H,
    pub(super) health: &'a mut Health,
    pub(super) slots: &'a SlotMap,
    pub(super) fault: Option<&'a Arc<DeviceBuffer<u32>>>,
    pub(super) hidden: usize,
    pub(super) n_used: usize,
}

impl BatchService {
    /// Serve layer `layer` for `x.ne1()` columns in one union call, as
    /// [`super::HostTier::serve_batch`] describes, on a tier its caller
    /// found sound.
    #[allow(
        clippy::too_many_arguments,
        reason = "the service's columns, their routing, the set to leave out and the sums"
    )]
    fn serve_one<H: HostExperts>(
        &mut self,
        t: Tier<'_, H>,
        layer: usize,
        x: Tensor2View<'_>,
        ids: &[u32],
        weights: &[f32],
        exclude: &[u32],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        const WHAT: &str = SERVE_BATCH;
        let (hidden, n_used) = (t.hidden, t.n_used);
        let cols = x.ne1();
        if x.ne0() != hidden
            || cols == 0
            || ids.len() != cols * n_used
            || weights.len() != cols * n_used
            || out.len() != cols * hidden
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{cols} columns of {} values with {} ids, {} weights and {} sums; the \
                     boundary carries {n_used} slots of {hidden} values a column",
                    x.ne0(),
                    ids.len(),
                    weights.len(),
                    out.len()
                ),
            ));
        }
        let t0 = Instant::now();
        let map_row = t.slots.row(layer).ok_or(GpuError::state(
            WHAT,
            "a hybrid layer without a slot map row",
        ))?;
        check_exclude(exclude, map_row, layer)?;
        if self.lens.len() < cols {
            self.lens.resize(cols, 0);
            self.lists.resize(cols * n_used, (0, 0.0f32));
            self.refused.reserve(cols);
        }
        self.refused.clear();
        let (mut host_slots, mut excluded) = (0u64, 0u64);
        // One pool pass over x finds the first column whose activation is not
        // finite; the columns before it are, and only the ones from it on are
        // scanned here, one by one, for what the host saw.
        let first_non_finite = model::ops::first_non_finite_col(x);
        // The first refused column and what the host saw in it.
        let mut refused: Option<(usize, String)> = None;
        for j in 0..cols {
            let list = &mut self.lists[j * n_used..][..n_used];
            let (mut n, mut out_of_set) = (0usize, 0u64);
            let mut unknown = None;
            for s in 0..n_used {
                let (id, w) = (ids[j * n_used + s], weights[j * n_used + s]);
                match usize::try_from(id).ok().and_then(|id| map_row.get(id)) {
                    Some(&HOST) if exclude.binary_search(&id).is_ok() => out_of_set += 1,
                    Some(&HOST) => {
                        list[n] = (id, w);
                        n += 1;
                    }
                    Some(_) => {}
                    None => unknown = unknown.or(Some((s, id))),
                }
            }
            let saw = match unknown {
                Some((s, id)) => Some(unknown_id(s, id, map_row.len())),
                None if first_non_finite.is_some_and(|f| j >= f) => non_finite(x.col(j)),
                None => None,
            };
            if let Some(saw) = saw {
                (n, out_of_set) = (0, 0);
                refused = refused.or(Some((j, saw)));
                self.refused.push(j);
            }
            self.lens[j] = n;
            host_slots += n as u64;
            excluded += out_of_set;
        }
        let mut lists = reuse_slices(std::mem::take(&mut self.slices));
        lists.extend(
            self.lens[..cols]
                .iter()
                .enumerate()
                .map(|(j, &n)| &self.lists[j * n_used..][..n]),
        );
        let Some((first, saw)) = refused else {
            let r = t.experts.experts_union_into(layer, x, &lists, out);
            self.slices = reuse_slices(lists);
            r?;
            self.count(cols, host_slots, excluded, t0);
            return Ok(());
        };
        // The refused columns reach the host experts as zeros with an empty
        // list, so no refused value is quantized, and come back as NaN.
        let mut clean = Tensor2::from_vec(hidden, cols, x.data().to_vec());
        for &j in &self.refused {
            clean.col_mut(j).fill(0.0);
        }
        let r = t
            .experts
            .experts_union_into(layer, clean.view(), &lists, out);
        self.slices = reuse_slices(lists);
        r?;
        for &j in &self.refused {
            out[j * hidden..][..hidden].fill(f32::NAN);
        }
        self.count(cols, host_slots, excluded, t0);
        let r = Refusal {
            what: WHAT,
            layer,
            detail: format!("column {first} of {cols}: {saw}"),
        };
        let Some(word) = t.fault else {
            t.health.record_refusal(r, false);
            return Ok(());
        };
        t.health.record_refusal(r.clone(), true);
        Err(name_refusal(&r, crate::fault::read(word)?))
    }

    /// [`BatchService::serve_one`] on a sound tier: refused on a poisoned
    /// one; a failure, or a panic inside the host experts, poisons it.
    #[allow(
        clippy::too_many_arguments,
        reason = "the service's columns, their routing, the set to leave out and the sums"
    )]
    pub(super) fn serve_guarded<H: HostExperts>(
        &mut self,
        t: Tier<'_, H>,
        layer: usize,
        x: Tensor2View<'_>,
        ids: &[u32],
        weights: &[f32],
        exclude: &[u32],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let Tier {
            experts,
            health,
            slots,
            fault,
            hidden,
            n_used,
        } = t;
        health.refuse_if_poisoned(SERVE_BATCH)?;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let t = Tier {
                experts: &mut *experts,
                health: &mut *health,
                slots,
                fault,
                hidden,
                n_used,
            };
            self.serve_one(t, layer, x, ids, weights, exclude, out)
        }));
        match r {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                health.set_poison(SERVE_BATCH, layer, Some(&e), None);
                Err(e)
            }
            Err(p) => {
                health.set_poison(SERVE_BATCH, layer, None, Some(&*p));
                std::panic::resume_unwind(p)
            }
        }
    }

    /// A service's counters: `cols` columns, `host_slots` host slots computed
    /// and `excluded` left out by the call's set, the union call's wall time
    /// since `t0`.
    fn count(&mut self, cols: usize, host_slots: u64, excluded: u64, t0: Instant) {
        let s = &mut self.stats;
        s.batch_served += 1;
        s.batch_cols += cols as u64;
        s.batch_host_slots += host_slots;
        s.batch_excluded_slots += excluded;
        s.batch_ns += nanos(t0.elapsed());
    }
}

/// `v`'s storage as an empty `Vec` of slices of another lifetime: the
/// in-place collect of an empty iterator keeps the allocation, so a buffer of
/// borrowed lists outlives the borrow it held without a copy.
fn reuse_slices<'a, 'b, T>(mut v: Vec<&'a [T]>) -> Vec<&'b [T]> {
    v.clear();
    v.into_iter()
        .map(|_| -> &'b [T] { unreachable!("the vector was cleared") })
        .collect()
}

/// A batch service's exclusion set against layer `layer`'s slot map row
/// `map_row`: strictly ascending, so no id twice, and every id one of the
/// layer's experts that the map sends to the host. A set that breaks any of
/// these is a caller's bug — an expert it names would be computed twice or
/// never — and is refused by name.
fn check_exclude(exclude: &[u32], map_row: &[u32], layer: usize) -> Result<(), GpuError> {
    let what = SERVE_BATCH;
    if let Some(p) = exclude.windows(2).find(|p| p[0] >= p[1]) {
        let why = if p[0] == p[1] {
            format!("expert {} twice", p[0])
        } else {
            format!("expert {} after {}: the set must ascend", p[1], p[0])
        };
        return Err(GpuError::shape(
            what,
            format!("exclusion set of layer {layer}: {why}"),
        ));
    }
    for &e in exclude {
        match usize::try_from(e).ok().and_then(|i| map_row.get(i)) {
            Some(&HOST) => {}
            Some(_) => {
                return Err(GpuError::shape(
                    what,
                    format!("exclusion set of layer {layer}: expert {e} is on the card"),
                ));
            }
            None => {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "exclusion set of layer {layer}: expert {e} past the layer's {} experts",
                        map_row.len()
                    ),
                ));
            }
        }
    }
    Ok(())
}
