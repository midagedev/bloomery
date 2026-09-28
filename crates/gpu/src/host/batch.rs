//! The batch port: an eager layer-batch's host leg, outside the go/wait
//! protocol. The card copies a block's activations and routing to the host
//! behind an event ([`BatchPort::download`]); the calling thread waits on
//! that event and serves the block's host experts in one union call
//! ([`BatchPort::serve`], through [`BatchService`]); one copy sends the sums
//! back ([`BatchPort::upload`]). The port holds two sets, taken in turn, so a
//! layer-batch's route copies into one set while the union still reads the
//! other.
//!
//! With an expert tier ([`super::tier::TierCard`]) a set also carries the
//! tier's leg of a tiered layer ([`BatchPort::attach_tier`]): the route's
//! download adds the slots' tier places; once the host has waited for the
//! set's copies it enqueues the tier's service on the tier's stream before
//! the union — the block's f32 activations and places copied from the set to
//! the tier, quantized there, the tier's experts by the same tile path the
//! stage card runs, their rows copied back into the set's host-mapped rows
//! and an event recorded — and after the union the host waits for that event
//! under the go deadline ([`HostTier::tier_rows_of`]) before the stage card's
//! card sum reads the rows in place. No launch of either card waits for the
//! other: the host orders them through the set's two events.

use super::slots::{HOST, Slot, SlotMap};
use super::step::GO_DEADLINE;
use super::tier::{TierBlock, TierCard};
use super::{Health, HostExperts, HostTier, Refusal, name_refusal, non_finite, unknown_id};
use crate::fault::Fault;
use crate::graph::{MappedHost, cu};
use crate::tensor::window;
use crate::{Gpu, GpuError, Q8Act};
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer, sys};
use model::placement::Machine;
use model::placement::workstation::{TIER_BATCH_HOST_RESERVE, TIER_BATCH_RESERVE};
use model::{Tensor2, Tensor2View};
use std::mem::{ManuallyDrop, size_of};
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

/// Where a set's tier leg stands: no tiered layer, a tiered layer routed
/// (its places downloaded), the tier's service enqueued, or its rows waited
/// for — which the set's upload requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    Idle,
    Routed(BatchKey),
    Issued(BatchKey),
    Settled(BatchKey),
}

/// A set's tier leg: the slots' tier places the route downloads, the rows
/// the tier's down writes back (host-mapped, slot-major from the block's
/// first slot) as the stage card's card sum reads them, the event the
/// tier's service completes at (the tier's context), and its stage.
///
/// Field order is drop order: the window before the allocation it views.
struct TierLeg {
    rows_stage: ManuallyDrop<DeviceBuffer<f32>>,
    rows: MappedHost,
    tsel: PinnedHostBuffer<u32>,
    done: CudaEvent,
    leg: Leg,
}

impl Drop for TierLeg {
    fn drop(&mut self) {
        // SAFETY: the window is taken once, here, and never read again; its
        // raw parts are dropped and nothing is freed — `rows` frees the
        // allocation after it.
        unsafe { drop(ManuallyDrop::take(&mut self.rows_stage).into_raw_parts()) };
    }
}

/// The tier's side of the port: its two legs, one a set, and its staging on
/// the tier card for a block — the activations and places copied in, their
/// q8_1 form, the down outputs by slot.
struct TierPort {
    legs: [TierLeg; 2],
    x: DeviceBuffer<f32>,
    tsel: DeviceBuffer<u32>,
    act: Q8Act,
    y: DeviceBuffer<f32>,
    stats: TierBatchStats,
}

/// What the tier's batch services have done since load: services enqueued,
/// the host's waits for their rows, those the tier had already finished, and
/// the waits' wall time in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierBatchStats {
    pub served: u64,
    pub settles: u64,
    pub settle_early: u64,
    pub settle_ns: u64,
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
    tier: Option<TierPort>,
    n_embd: usize,
    n_used: usize,
    cap: usize,
    down: usize,
    serve: usize,
    up: usize,
}

/// What a tiered set hands the tier's service
/// ([`HostTier::serve_port`]): `key`'s tokens' activations and tier places
/// in the set, the tier's staging, the set's rows and the event to record.
struct TierServe<'a> {
    key: BatchKey,
    n_embd: usize,
    n_used: usize,
    x: &'a PinnedHostBuffer<f32>,
    tsel: &'a PinnedHostBuffer<u32>,
    stage_x: &'a mut DeviceBuffer<f32>,
    stage_tsel: &'a mut DeviceBuffer<u32>,
    act: &'a mut Q8Act,
    y: &'a mut DeviceBuffer<f32>,
    rows: &'a MappedHost,
    rows_len: usize,
    done: &'a CudaEvent,
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
            tier: None,
            n_embd,
            n_used,
            cap,
            down: 0,
            serve: 0,
            up: 0,
        })
    }

    /// The tier's side of the port, for an expert tier on `tier` under the
    /// stage card of `stage`: per set the places' copy, the rows (host-mapped,
    /// seen from both cards) and the tier's event; on the tier card the
    /// staging for a block of up to the port's tokens. Load-time only; a
    /// second call is refused by name.
    pub fn attach_tier(&mut self, stage: &Arc<CudaContext>, tier: &Gpu) -> Result<(), GpuError> {
        const WHAT: &str = "BatchPort::attach_tier";
        if self.tier.is_some() {
            return Err(GpuError::state(WHAT, "a port without a tier leg"));
        }
        let (n, s, cap) = (self.n_embd, self.n_used, self.cap);
        let slots = cap * s;
        let row_values = slots * n;
        let leg = || -> Result<TierLeg, GpuError> {
            let rows = MappedHost::new(stage, 4 * row_values, "cuMemHostAlloc (the tier's rows)")?;
            // SAFETY: the window spans the `row_values` f32 of `rows`'s
            // allocation, which the leg keeps and frees only after the window
            // (field order).
            let rows_stage = unsafe { window::<f32>(rows.dev_at(0), row_values, stage) };
            let tsel = PinnedHostBuffer::<u32>::zeroed(stage, slots).map_err(|source| {
                GpuError::Driver {
                    op: Some("cuMemAllocHost (the batch's tier places)"),
                    source,
                }
            })?;
            Ok(TierLeg {
                rows_stage,
                rows,
                tsel,
                done: tier.context().new_event(None)?,
                leg: Leg::Idle,
            })
        };
        let legs = [leg()?, leg()?];
        tier.context().bind_to_thread()?;
        let ts = tier.stream();
        let port = TierPort {
            legs,
            x: DeviceBuffer::zeroed(ts, cap * n)?,
            tsel: DeviceBuffer::zeroed(ts, slots)?,
            act: Q8Act::with_slots(ts, cap, n)?,
            y: DeviceBuffer::zeroed(ts, row_values)?,
            stats: TierBatchStats::default(),
        };
        stage.bind_to_thread()?;
        self.tier = Some(port);
        Ok(())
    }

    /// Tokens a block of this port holds at most.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Host bytes the tier's legs hold: per set its rows and places.
    #[must_use]
    pub fn tier_host_bytes(&self) -> usize {
        self.tier.as_ref().map_or(0, |t| {
            t.legs
                .iter()
                .map(|l| 4 * l.rows_stage.len() + l.tsel.len() * size_of::<u32>())
                .sum()
        })
    }

    /// Device bytes the tier's staging holds on the tier card.
    #[must_use]
    pub fn tier_device_bytes(&self) -> usize {
        self.tier.as_ref().map_or(0, |t| {
            t.x.num_bytes() + t.tsel.num_bytes() + t.act.device_bytes() + t.y.num_bytes()
        })
    }

    /// Both sets free, the next download into the first: at a group's
    /// start. Host bookkeeping only; the sets are next written by the group's
    /// downloads, when neither card's stream holds a copy of an earlier group
    /// (the group's prologue waits for the stage's stream; a group that
    /// failed drained the tier's, or lost the tier, after which the host
    /// tier serves nothing more).
    pub fn begin(&mut self) {
        for s in &mut self.sets {
            s.stage = Stage::Free;
        }
        if let Some(t) = self.tier.as_mut() {
            for l in &mut t.legs {
                l.leg = Leg::Idle;
            }
        }
        (self.down, self.serve, self.up) = (0, 0, 0);
    }

    /// Enqueue the copies of `key`'s tokens `at .. u` of `x` (`n_embd` a
    /// token), `ids` and `w` (`n_used` a token) into the next set, and its
    /// event: [`BatchPort::download_pitched`] at a pitch of `n_used`.
    pub fn download(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        self.download_pitched(stream, xw, ids, self.n_used, key)
    }

    /// Enqueue the copies of `key`'s tokens `at .. u` of `x` (`n_embd` a
    /// token) and of the routing into the next set, and its event. The
    /// routing buffers hold `pitch >= n_used` entries a token whose first
    /// `n_used` are the routed slots (a router that writes a shared expert's
    /// slot after them): token `t`'s ids and weights are entries `t·pitch ..
    /// t·pitch + n_used`, copied into the set's `n_used` a token — one
    /// contiguous copy each at a pitch of `n_used`, one strided copy each
    /// past it. Refused while that set holds a layer not uploaded yet, and
    /// for a pitch below `n_used`.
    pub fn download_pitched(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        pitch: usize,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        self.download_with(stream, xw, ids, pitch, None, key)
    }

    /// [`BatchPort::download`] of a tiered layer: the slots' tier places
    /// `tsel` (`n_used` a token) copied into the set's tier leg before the
    /// event too, so the tier's service, enqueued once the host has waited
    /// for it, reads them. Refused on a port without a tier leg.
    pub fn download_tiered(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        tsel: &DeviceBuffer<u32>,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        self.download_with(stream, xw, ids, self.n_used, Some(tsel), key)
    }

    fn download_with(
        &mut self,
        stream: &CudaStream,
        [x, w]: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        pitch: usize,
        tsel: Option<&DeviceBuffer<u32>>,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        let (n, s) = (self.n_embd, self.n_used);
        let BatchKey { at, u, .. } = key;
        if pitch < s {
            return Err(GpuError::Shape {
                what: PORT,
                detail: format!("routing of {pitch} entries a token for {s} routed slots"),
            });
        }
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
        let mut leg = match (tsel, self.tier.as_mut()) {
            (None, _) => None,
            (Some(t), Some(port)) => Some((t, &mut port.legs[self.down])),
            (Some(_), None) => {
                return Err(GpuError::state(
                    PORT,
                    "a tier leg for a tiered layer's route (BatchPort::attach_tier)",
                ));
            }
        };
        if let Some((_, l)) = &leg
            && l.leg != Leg::Idle
        {
            return Err(GpuError::Shape {
                what: PORT,
                detail: format!(
                    "a tiered route of {key:?} into a tier leg that holds {:?}",
                    l.leg
                ),
            });
        }
        // SAFETY: each copy writes this set's page-locked buffers (the helpers
        // check both extents). The set is free: its last serve returned, so the
        // union no longer reads them, and its upload is enqueued before these
        // copies — and a tiered set's upload came only after the host saw the
        // tier's service, which read its places, complete. The host and the
        // tier read them again only in this set's next serve, after the host's
        // wait on the event recorded below.
        unsafe {
            dtoh(stream, &mut set.x, x, at * n..u * n)?;
            if pitch == s {
                dtoh(stream, &mut set.ids, ids, at * s..u * s)?;
                dtoh(stream, &mut set.w, w, at * s..u * s)?;
            } else {
                dtoh_pitched(stream, &mut set.ids, ids, (at..u, s, pitch))?;
                dtoh_pitched(stream, &mut set.w, w, (at..u, s, pitch))?;
            }
            if let Some((t, l)) = leg.as_mut() {
                dtoh(stream, &mut l.tsel, *t, at * s..u * s)?;
            }
        }
        if let Some((_, l)) = leg {
            l.leg = Leg::Routed(key);
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
        self.serve_tiered(key, |_| Ok(()), serve)
    }

    /// [`BatchPort::serve`], and for a tiered set, once its copies have
    /// landed and before the union, `tier` enqueues the tier's service of
    /// the set's leg; the leg is then issued, and the set's upload waits for
    /// [`BatchPort::settle_tier`]. `tier` is not called for a set without one.
    fn serve_tiered(
        &mut self,
        key: BatchKey,
        tier: impl FnOnce(TierServe<'_>) -> Result<(), GpuError>,
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
        if let Some(port) = self.tier.as_mut() {
            let TierPort {
                legs,
                x,
                tsel,
                act,
                y,
                stats,
            } = port;
            let l = &mut legs[self.serve];
            match l.leg {
                Leg::Idle => {}
                Leg::Routed(k) if k == key => {
                    tier(TierServe {
                        key,
                        n_embd: n,
                        n_used: s,
                        x: &set.x,
                        tsel: &l.tsel,
                        stage_x: x,
                        stage_tsel: tsel,
                        act,
                        y,
                        rows: &l.rows,
                        rows_len: l.rows_stage.len(),
                        done: &l.done,
                    })?;
                    l.leg = Leg::Issued(key);
                    stats.served += 1;
                }
                other => {
                    return Err(GpuError::Shape {
                        what: PORT,
                        detail: format!("the serve of {key:?}; the set's tier leg holds {other:?}"),
                    });
                }
            }
        }
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

    /// Wait, until `deadline`, for the tier's service of the oldest served
    /// set — `key`'s, and tiered, else refused by name — and hand back its
    /// rows as the stage card reads them: the tier's down outputs of the
    /// block's slots, slot-major from its first slot. `Ok(None)` when the
    /// tier's event has not completed by `deadline`: the caller names the
    /// tier lost. A second call on a settled leg returns the rows at once.
    pub fn settle_tier(
        &mut self,
        key: BatchKey,
        deadline: Instant,
    ) -> Result<Option<&DeviceBuffer<f32>>, GpuError> {
        const WHAT: &str = "BatchPort::settle_tier";
        let set = &self.sets[self.up];
        let port = self
            .tier
            .as_mut()
            .ok_or(GpuError::state(WHAT, "a tier leg (BatchPort::attach_tier)"))?;
        let l = &mut port.legs[self.up];
        if set.stage != Stage::Served(key) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the tier's rows of {key:?}; the oldest served exchange set holds {:?}",
                    set.stage
                ),
            });
        }
        let settled = match l.leg {
            Leg::Settled(k) if k == key => true,
            Leg::Issued(k) if k == key => false,
            other => {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "the tier's rows of {key:?}; the set's tier leg holds {other:?}"
                    ),
                });
            }
        };
        if !settled {
            let t0 = Instant::now();
            let mut early = true;
            while !l.done.query()? {
                early = false;
                if Instant::now() > deadline {
                    return Ok(None);
                }
                std::thread::yield_now();
            }
            let st = &mut port.stats;
            st.settles += 1;
            st.settle_early += u64::from(early);
            st.settle_ns += nanos(t0.elapsed());
            l.leg = Leg::Settled(key);
        }
        Ok(Some(&*l.rows_stage))
    }

    /// What the tier's batch services have done since load.
    #[must_use]
    pub fn tier_stats(&self) -> TierBatchStats {
        self.tier.as_ref().map(|t| t.stats).unwrap_or_default()
    }

    /// Enqueue the copy of the oldest served set's sums, which must be
    /// `key`'s, to `hsum`; the set is free again. A tiered set's tier rows
    /// must have been waited for ([`BatchPort::settle_tier`]) — a join
    /// without them is refused by name.
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
        if let Some(port) = self.tier.as_mut() {
            let l = &mut port.legs[self.up];
            match l.leg {
                Leg::Idle => {}
                Leg::Settled(k) if k == key => l.leg = Leg::Idle,
                other => {
                    return Err(GpuError::Shape {
                        what: PORT,
                        detail: format!(
                            "the upload of {key:?} before its tier rows were joined; the set's \
                             tier leg holds {other:?}"
                        ),
                    });
                }
            }
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

/// Enqueue the copy of tokens `toks` of `src` — `width` values a token at
/// a pitch of `pitch` values — to the same tokens of `dst` at a pitch of
/// `width`: one strided copy.
///
/// SAFETY: `dst` is not read or freed until the copy completes.
unsafe fn dtoh_pitched<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    dst: &mut PinnedHostBuffer<T>,
    src: &DeviceBuffer<T>,
    (toks, width, pitch): (Range<usize>, usize, usize),
) -> Result<(), GpuError> {
    let src_end = (toks.end.saturating_sub(1)) * pitch + width;
    if toks.is_empty()
        || width == 0
        || pitch < width
        || src_end > src.len()
        || toks.end * width > dst.len()
    {
        return Err(GpuError::Shape {
            what: PORT,
            detail: format!(
                "tokens {toks:?} of {width} values at a pitch of {pitch} from {} into {}",
                src.len(),
                dst.len()
            ),
        });
    }
    let z = size_of::<T>();
    let copy = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(),
        srcDevice: src.cu_deviceptr() + (toks.start * pitch * z) as u64,
        srcArray: std::ptr::null_mut(),
        srcPitch: pitch * z,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: sys::CUmemorytype_enum_CU_MEMORYTYPE_HOST,
        // SAFETY: token `toks.start`'s first value lies inside `dst` (checked
        // above).
        dstHost: unsafe { dst.as_mut_ptr().add(toks.start * width) }.cast(),
        dstDevice: 0,
        dstArray: std::ptr::null_mut(),
        dstPitch: width * z,
        WidthInBytes: width * z,
        Height: toks.len(),
    };
    // SAFETY: the copy reads `toks.len()` rows of `width` values at `pitch`
    // from token `toks.start` of `src`, the last ending at `src_end <=
    // src.len()`, and writes as many rows back to back into `dst` from token
    // `toks.start`, ending at `toks.end · width <= dst.len()` (checked
    // above); the descriptor is read at the call; `dst` stays untouched until
    // the copy completes by this fn's contract.
    let rc = unsafe { sys::cuMemcpy2DAsync_v2(&raw const copy, stream.cu_stream()) };
    cu(rc, "cuMemcpy2DAsync_v2 (the batch's routing)")
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

/// Enqueue the copy of values `at` of `src` to `dst` from its start.
///
/// SAFETY: `src` is not written or freed until the copy completes.
unsafe fn htod_front<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    dst: &mut DeviceBuffer<T>,
    src: &PinnedHostBuffer<T>,
    at: Range<usize>,
) -> Result<(), GpuError> {
    if at.is_empty() || at.end > src.len() || at.len() > dst.len() {
        return Err(GpuError::Shape {
            what: PORT,
            detail: format!("values {at:?} from {} into {}", src.len(), dst.len()),
        });
    }
    // SAFETY: `src` holds the values of `at` and `dst` their count (checked
    // above); `src` stays unwritten until the copy completes by this fn's
    // contract.
    let rc = unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            dst.cu_deviceptr(),
            src.as_ptr().add(at.start).cast(),
            at.len() * size_of::<T>(),
            stream.cu_stream(),
        )
    };
    cu(rc, "cuMemcpyHtoDAsync_v2 (the tier's block)")
}

/// Enqueue the copy of the first `len` f32 of `src` to the start of `dst`,
/// a host-mapped allocation of `dst_len` f32.
///
/// SAFETY: `dst` holds at least `dst_len` f32; nothing reads its first `len`
/// until the copy completes.
unsafe fn dtoh_mapped(
    stream: &CudaStream,
    (dst, dst_len): (&MappedHost, usize),
    src: &DeviceBuffer<f32>,
    len: usize,
) -> Result<(), GpuError> {
    if len == 0 || len > src.len() || len > dst_len {
        return Err(GpuError::Shape {
            what: PORT,
            detail: format!("{len} values from {} into {dst_len}", src.len()),
        });
    }
    // SAFETY: both spans hold `len` f32 (checked above, `dst_len` by this
    // fn's contract); nothing reads the host span until the copy completes.
    let rc = unsafe {
        sys::cuMemcpyDtoHAsync_v2(
            dst.host_at(0).cast(),
            src.cu_deviceptr(),
            4 * len,
            stream.cu_stream(),
        )
    };
    cu(rc, "cuMemcpyDtoHAsync_v2 (the tier's rows)")
}

// ------------------------------------------------------------------- tier

/// Enqueue the tier's service of `io`'s block on `tier`'s stream
/// ([`enqueue_tier_leg`]); the stage card's context `stage` is current again
/// on return, whatever the result.
fn enqueue_tier_service(
    tier: &mut TierCard,
    io: TierServe<'_>,
    stage: &Arc<CudaContext>,
) -> Result<(), GpuError> {
    let r = enqueue_tier_leg(tier, io);
    let back = stage.bind_to_thread();
    r?;
    back?;
    Ok(())
}

/// The tier's service of `io`'s block, on the tier's stream, its tier
/// slots counted as the tier's hits ([`TierCard::hit`]): the block's
/// f32 activations and tier places copied from the set to the staging from
/// column 0; their q8_1 form, the bytes the stage card's fused norm wrote
/// for the same columns (its q8_1 output is the quantizer's over its f32
/// output); the tier's experts over the block by the tile path into the
/// down outputs by slot; those copied into the set's rows; the set's event.
/// Each copy's source was complete when the host enqueued it: the host has
/// waited for the set's route copies.
fn enqueue_tier_leg(tier: &mut TierCard, io: TierServe<'_>) -> Result<(), GpuError> {
    let TierServe {
        key,
        n_embd: n,
        n_used: s,
        x,
        tsel,
        stage_x,
        stage_tsel,
        act,
        y,
        rows,
        rows_len,
        done,
    } = io;
    let BatchKey { layer, at, u, .. } = key;
    let cols = u - at;
    let places = tsel.get(at * s..u * s).ok_or_else(|| GpuError::Shape {
        what: PORT,
        detail: format!("the tier places of {key:?} from a set of {}", tsel.len()),
    })?;
    tier.hit(layer, places.iter().filter(|&&p| p != HOST).count() as u64);
    {
        let g = tier.gpu();
        g.context().bind_to_thread()?;
        // SAFETY: the set's activations and places are not written again
        // before its upload, which waits for this service's event, recorded
        // after these copies.
        unsafe {
            htod_front(g.stream(), stage_x, x, at * n..u * n)?;
            htod_front(g.stream(), stage_tsel, tsel, at * s..u * s)?;
        }
        g.enqueue_quantize_q8_1_cols(stage_x, act, cols, layer)?;
    }
    tier.enqueue_block(
        layer,
        TierBlock {
            act: &*act,
            sel: &*stage_tsel,
            cols,
            down: &mut *y,
        },
    )?;
    let stream = tier.gpu().stream();
    // SAFETY: `rows_len` is the rows' allocation's length (their window spans
    // all of it); the stage card reads them only after the host has seen the
    // event recorded below complete.
    unsafe { dtoh_mapped(stream, (rows, rows_len), y, cols * s * n)? };
    done.record(stream)?;
    Ok(())
}

impl<H: HostExperts> HostTier<H> {
    /// The batch port's tier leg, once the host tier holds both the port
    /// ([`HostTier::prepare_batch`]) and an expert tier: per set the places,
    /// the rows and the tier's event, and the tier card's staging
    /// ([`BatchPort::attach_tier`]). Nothing without a tier; made once.
    /// Load-time or first-prompt only; what it holds is checked against the
    /// plan by [`HostTier::check_tier_reserves`].
    pub fn prepare_tier_batch(&mut self) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::prepare_tier_batch";
        let stage = Arc::clone(self.step.boundary.region.context());
        let (Some(port), Some(tier)) = (self.port.as_mut(), self.tier.as_ref()) else {
            return Ok(());
        };
        if port.tier.is_some() {
            return Ok(());
        }
        let r = port.attach_tier(&stage, tier.gpu());
        let back = stage.bind_to_thread();
        r?;
        back.map_err(|e| GpuError::shape(WHAT, format!("rebinding the stage context: {e}")))?;
        Ok(())
    }

    /// The expert tier's prompt-batch bytes against the reserves `machine`'s
    /// plan carries for them: on the tier card, index `tier` of
    /// [`Machine::all_cards`], the batch staging and the architecture's block
    /// scratch ([`TierCard::block_bytes`]) against its [`TIER_BATCH_RESERVE`]
    /// row; on the host, the tier leg's rows and places against the host's
    /// [`TIER_BATCH_HOST_RESERVE`] row. A missing row, a row named twice, or
    /// a difference is refused by name, naming the reserve. Load-time only,
    /// once [`HostTier::prepare_batch`] has made the leg.
    pub fn check_tier_reserves(&self, machine: &Machine, tier: usize) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::check_tier_reserves";
        let t = self.tier_ref(WHAT)?;
        let port = self.port.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch port's sets (HostTier::prepare_batch)",
        })?;
        let card = machine.card(tier).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!(
                    "the placement has no card {tier} for the expert tier {}",
                    t.name()
                ),
            )
        })?;
        let rows = [
            (
                card.name.as_str(),
                &card.reserves,
                TIER_BATCH_RESERVE,
                port.tier_device_bytes() + t.block_bytes(),
            ),
            (
                "the host",
                &machine.host.reserves,
                TIER_BATCH_HOST_RESERVE,
                port.tier_host_bytes(),
            ),
        ];
        for (on, reserves, name, got) in rows {
            let want = match reserves
                .iter()
                .filter(|(n, _)| n == name)
                .collect::<Vec<_>>()
                .as_slice()
            {
                [(_, b)] => *b,
                named => {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "the plan names the reserve {name:?} on {on} {} times, not once",
                            named.len()
                        ),
                    ));
                }
            };
            if got as u64 != want {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "the expert tier's prompt batch holds {got} B on {on}; the plan's reserve \
                         {name:?} there is {want} B"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// [`HostTier::enqueue_download`] of a tiered layer: the slots' tier
    /// places `tsel` (`n_used` a token) too, which the tier's service reads
    /// ([`BatchPort::download_tiered`]). Refused by name for a layer whose
    /// tier holds no expert, and without a tier or its leg.
    pub fn enqueue_download_tiered(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        tsel: &DeviceBuffer<u32>,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::enqueue_download_tiered";
        if self.tier.is_none() || self.slots.on_tier(key.layer)? == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {}: a tiered route of a layer the tier holds no expert of",
                    key.layer
                ),
            ));
        }
        self.port_mut(WHAT)?
            .download_tiered(stream, xw, ids, tsel, key)
    }

    /// The oldest download not served yet — `key`'s, else refused by name —
    /// waited for; a tiered set's tier service enqueued on the tier's stream
    /// ([`enqueue_tier_leg`]); then its layer's host experts served for its
    /// tokens in one union call, the sums into the set the upload sends. A
    /// failed tier enqueue poisons the host tier and releases both cards'
    /// waits; a poisoned host tier is refused by name before either card is
    /// given work. Returns the host time outside the union call.
    pub fn serve_port(&mut self, key: BatchKey) -> Result<ServeTimes, GpuError> {
        const WHAT: &str = "HostTier::serve_key";
        self.health.refuse_if_poisoned(SERVE_BATCH)?;
        let h = self.step.boundary.layout.handoff();
        let stage = Arc::clone(self.step.boundary.region.context());
        let Some(port) = self.port.as_mut() else {
            return Err(GpuError::State {
                what: WHAT,
                missing: "the batch port's sets (HostTier::prepare_batch)",
            });
        };
        let (batch, experts, health, tier) = (
            &mut self.batch,
            &mut self.experts,
            &mut self.health,
            &mut self.tier,
        );
        let (slots, fault) = (&self.slots, self.fault.as_ref());
        let mut tier_failed = false;
        let r = port.serve_tiered(
            key,
            |io| {
                let r = match tier.as_mut() {
                    Some(t) => enqueue_tier_service(t, io, &stage),
                    None => Err(GpuError::state(WHAT, "an expert tier for a tiered set")),
                };
                tier_failed = r.is_err();
                r
            },
            |layer, x, ids, w, out| {
                let t = Tier {
                    experts,
                    health,
                    slots,
                    fault,
                    hidden: h.hidden,
                    n_used: h.n_used,
                };
                batch.serve_guarded(t, layer, x, ids, w, &[], out)
            },
        );
        if let Err(e) = &r
            && tier_failed
        {
            self.health
                .set_poison(SERVE_BATCH, key.layer, Some(e), None);
            self.release_all();
        }
        r
    }

    /// The tier's rows of `key`'s set as the stage card's card sum reads
    /// them ([`BatchPort::settle_tier`]), once the host has seen the tier's
    /// service complete, under the go deadline: a tier that has not finished
    /// by then is lost — the host tier is poisoned as a lost card, both
    /// cards' waits released, and the error names the card.
    pub fn tier_rows_of(&mut self, key: BatchKey) -> Result<&DeviceBuffer<f32>, GpuError> {
        const WHAT: &str = "HostTier::tier_rows_of";
        let deadline = Instant::now() + GO_DEADLINE;
        let landed = self.port_mut(WHAT)?.settle_tier(key, deadline)?.is_some();
        if !landed {
            return Err(self.lose_tier_as(SERVE_BATCH, key.layer, |t| {
                format!(
                    "the expert tier on {} did not finish its service of the prompt block's \
                     columns {}..{} in {GO_DEADLINE:?}: the card is lost",
                    t.name(),
                    key.at,
                    key.u
                )
            }));
        }
        self.port_mut(WHAT)?
            .settle_tier(key, deadline)?
            .ok_or(GpuError::state(WHAT, "the tier's rows just settled"))
    }

    /// The expert tier's fault word once its stream has drained; `None`
    /// without a tier or while it is clean, and for a tier lost as a card
    /// ([`super::PoisonKind::CardLost`]), whose stream may never drain — the
    /// loss is the error, no reset lifts it, and every later call is refused
    /// at its group's start ([`HostTier::begin_group`]) before it touches a
    /// set that stream could still touch. The stage card's context is current
    /// again on return. A prompt call reads it at a group's end and when a
    /// group fails, beside the stage card's.
    pub fn tier_fault(&mut self) -> Result<Option<Fault>, GpuError> {
        let Some(t) = self.tier.as_ref() else {
            return Ok(None);
        };
        if self
            .health
            .poison
            .as_ref()
            .is_some_and(|p| p.mark().kind == super::PoisonKind::CardLost)
        {
            return Ok(None);
        }
        let stage = Arc::clone(self.step.boundary.region.context());
        let r = t.gpu().fault();
        let back = stage.bind_to_thread();
        let f = r?;
        back?;
        Ok(f)
    }

    /// Device bytes the tier's batch staging holds on the tier card
    /// ([`BatchPort::tier_device_bytes`]); 0 before it is made.
    #[must_use]
    pub fn tier_batch_bytes(&self) -> usize {
        self.port.as_ref().map_or(0, BatchPort::tier_device_bytes)
    }

    /// What the tier's batch services have done since load.
    #[must_use]
    pub fn tier_batch_stats(&self) -> TierBatchStats {
        self.port
            .as_ref()
            .map(BatchPort::tier_stats)
            .unwrap_or_default()
    }
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
        if t.slots.row_offset(layer).is_none() {
            return Err(GpuError::state(
                WHAT,
                "a hybrid layer without a slot map row",
            ));
        }
        check_exclude(exclude, t.slots, layer)?;
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
                match t.slots.slot(layer, id) {
                    Some(Slot::Host) if exclude.binary_search(&id).is_ok() => out_of_set += 1,
                    Some(Slot::Host) => {
                        list[n] = (id, w);
                        n += 1;
                    }
                    Some(Slot::Card(_) | Slot::Tier(_)) => {}
                    None => unknown = unknown.or(Some((s, id))),
                }
            }
            let saw = match unknown {
                Some((s, id)) => Some(unknown_id(s, id, t.slots.n_expert())),
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

/// A batch service's exclusion set against layer `layer`'s row of `slots`:
/// strictly ascending, so no id twice, and every id one of the layer's
/// experts that the map sends to the host. A set that breaks any of these is
/// a caller's bug — an expert it names would be computed twice or never — and
/// is refused by name.
fn check_exclude(exclude: &[u32], slots: &SlotMap, layer: usize) -> Result<(), GpuError> {
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
        let on = match slots.slot(layer, e) {
            Some(Slot::Host) => continue,
            Some(Slot::Card(_)) => "the card",
            Some(Slot::Tier(_)) => "the tier card",
            None => {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "exclusion set of layer {layer}: expert {e} past the layer's {} experts",
                        slots.n_expert()
                    ),
                ));
            }
        };
        return Err(GpuError::shape(
            what,
            format!("exclusion set of layer {layer}: expert {e} is on {on}"),
        ));
    }
    Ok(())
}
