//! The batch port: an eager layer-batch's host leg, outside the go/wait
//! protocol. The card copies a block's activations and routing to the host
//! behind an event ([`BatchPort::download`]); the calling thread waits on
//! that event and serves the block's host experts in one union call
//! ([`BatchPort::serve`], through [`BatchService`]); one copy sends the sums
//! back ([`BatchPort::upload`]). The port holds two sets, taken in turn, so a
//! layer-batch's route copies into one set while the union still reads the
//! other.
//!
//! With expert tiers ([`super::tier::TierCard`]) a set also carries each
//! tier's leg of a tiered layer ([`BatchPort::attach_tier`], one a tier): the
//! route's download adds the slots' places of each tier that holds experts of
//! the layer; once the host has waited for the set's copies it enqueues each
//! such tier's service on that tier's stream before the union — the block's
//! f32 activations and places copied from the set to the tier, quantized
//! there, the tier's experts by the same tile path the stage card runs, their
//! rows copied back into the leg's host-mapped rows and an event recorded —
//! and after the union the host waits for every served leg's event under the
//! go deadline ([`HostTier::tier_rows_of`]) before the stage card's card sum
//! reads the rows in place. No launch of any card waits for another: the
//! host orders them through the set's events.

use super::slots::{HOST, MAX_TIERS, Slot, SlotMap};
use super::step::GO_DEADLINE;
use super::tier::{BlockRows, TierBlock, TierCard};
use super::{Health, HostExperts, HostTier, Refusal, name_refusal, nanos, non_finite, unknown_id};
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

/// Where one listed expert's columns of a layer-batch turn hot: past this
/// many, the expert's activation set no longer fits a core's L2 beside the
/// weight stream — the L2's capacity less the streaming margin, over the
/// bytes one quantized activation column occupies.
pub const HOT_COLS: usize = 136;

/// What one served layer-batch's union held, per listed expert: how many
/// experts its columns listed, the widest expert's columns, the experts past
/// [`HOT_COLS`] columns and the columns they took, and `Σ m²` over the
/// listed experts, which with the slots the serve listed gives the routing's
/// spread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnionCols {
    /// Distinct experts the layer-batch's columns listed.
    pub experts: usize,
    /// The most columns one listed expert took.
    pub m_max: usize,
    /// Listed experts past [`HOT_COLS`] columns.
    pub m_hot: usize,
    /// The columns those experts took.
    pub cols_hot: usize,
    /// `Σ m²` over the listed experts.
    pub m_sq: u64,
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
/// first slot, or packed: [`BlockRows::Packed`]) as the stage card's card
/// sum reads them, the event the tier's service completes at (the tier's
/// context), and its stage.
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

/// One tier's side of the port: its two legs, one a set, and its staging on
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

/// What a tier's batch services have done since load: services enqueued,
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
    /// Each tier's side, in tier order ([`BatchPort::attach_tier`]).
    tiers: Vec<TierPort>,
    n_embd: usize,
    n_used: usize,
    cap: usize,
    down: usize,
    serve: usize,
    up: usize,
}

/// What a tiered set hands one tier's service
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
            tiers: Vec::new(),
            n_embd,
            n_used,
            cap,
            down: 0,
            serve: 0,
            up: 0,
        })
    }

    /// The next tier's side of the port, for the expert tier on `tier` under
    /// the stage card of `stage`: per set the places' copy, the rows
    /// (host-mapped, seen from both cards) and the tier's event; on the tier
    /// card the staging for a block of up to the port's tokens. Tier `t` is
    /// the `t`-th call's. Load-time only; past [`super::slots::MAX_TIERS`]
    /// tiers it is refused by name.
    pub fn attach_tier(&mut self, stage: &Arc<CudaContext>, tier: &Gpu) -> Result<(), GpuError> {
        const WHAT: &str = "BatchPort::attach_tier";
        if self.tiers.len() >= super::slots::MAX_TIERS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a tier leg past the {} tiers a map names",
                    super::slots::MAX_TIERS
                ),
            ));
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
            act: Q8Act::with_tier_cols(ts, cap, n)?,
            y: DeviceBuffer::zeroed(ts, row_values)?,
            stats: TierBatchStats::default(),
        };
        stage.bind_to_thread()?;
        self.tiers.push(port);
        Ok(())
    }

    /// Tokens a block of this port holds at most.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Host bytes the tiers' legs hold, summed over the tiers: per set its
    /// rows and places.
    #[must_use]
    pub fn tier_host_bytes(&self) -> usize {
        self.tiers
            .iter()
            .flat_map(|t| &t.legs)
            .map(|l| 4 * l.rows_stage.len() + l.tsel.len() * size_of::<u32>())
            .sum()
    }

    /// Device bytes tier `tier`'s staging holds on its card; 0 for a tier the
    /// port has no side of.
    #[must_use]
    pub fn tier_device_bytes(&self, tier: usize) -> usize {
        self.tiers.get(tier).map_or(0, |t| {
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
        for l in self.tiers.iter_mut().flat_map(|t| &mut t.legs) {
            l.leg = Leg::Idle;
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
        self.download_with(stream, xw, ids, pitch, &[], key)
    }

    /// [`BatchPort::download`] of a tiered layer: for each `(t, places)` of
    /// `tsels`, in tier order, tier `t`'s places of the slots (`n_used` a
    /// token) copied into the set's leg of tier `t` before the event too, so
    /// that tier's service, enqueued once the host has waited for it, reads
    /// them. Refused for a tier the port has no side of.
    pub fn download_tiered(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        tsels: &[(usize, &DeviceBuffer<u32>)],
        key: BatchKey,
    ) -> Result<(), GpuError> {
        self.download_with(stream, xw, ids, self.n_used, tsels, key)
    }

    /// [`BatchPort::download_tiered`] from routing buffers of `pitch`
    /// entries a token whose first `n_used` are the routed slots
    /// ([`BatchPort::download_pitched`]); the tier places stay `n_used` a
    /// token.
    pub fn download_tiered_pitched(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        pitch: usize,
        tsels: &[(usize, &DeviceBuffer<u32>)],
        key: BatchKey,
    ) -> Result<(), GpuError> {
        self.download_with(stream, xw, ids, pitch, tsels, key)
    }

    fn download_with(
        &mut self,
        stream: &CudaStream,
        [x, w]: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        pitch: usize,
        tsels: &[(usize, &DeviceBuffer<u32>)],
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
        for &(t, _) in tsels {
            let Some(port) = self.tiers.get(t) else {
                return Err(GpuError::state(
                    PORT,
                    "a tier leg for a tiered layer's route (BatchPort::attach_tier)",
                ));
            };
            let l = &port.legs[self.down];
            if l.leg != Leg::Idle {
                return Err(GpuError::Shape {
                    what: PORT,
                    detail: format!(
                        "a tiered route of {key:?} into tier {t}'s leg that holds {:?}",
                        l.leg
                    ),
                });
            }
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
            for &(t, places) in tsels {
                dtoh(
                    stream,
                    &mut self.tiers[t].legs[self.down].tsel,
                    places,
                    at * s..u * s,
                )?;
            }
        }
        for &(t, _) in tsels {
            self.tiers[t].legs[self.down].leg = Leg::Routed(key);
        }
        set.routed.record(stream)?;
        set.stage = Stage::Routed(key);
        self.down ^= 1;
        Ok(())
    }

    /// Wait for the oldest unserved set's copies, which must be `key`'s, and
    /// hand back its routed ids (`n_used` a token of its tokens `at .. u`)
    /// without serving it: what a prompt call's pick counts before the
    /// serve. Refused by name when the oldest unserved set is not `key`'s.
    pub fn routed_ids(&mut self, key: BatchKey) -> Result<&[u32], GpuError> {
        let s = self.n_used;
        let BatchKey { at, u, .. } = key;
        let set = &mut self.sets[self.serve];
        if set.stage != Stage::Routed(key) {
            return Err(GpuError::Shape {
                what: PORT,
                detail: format!(
                    "the routed ids of {key:?}; the oldest unserved exchange set holds {:?}",
                    set.stage
                ),
            });
        }
        let routed = &set.routed;
        super::await_done("BatchPort::routed_ids", || routed.query())?;
        Ok(&set.ids[at * s..u * s])
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
        self.serve_tiered(key, |_, _| Ok(()), serve)
    }

    /// [`BatchPort::serve`], and for a tiered set, once its copies have
    /// landed and before the union, `tier` enqueues, in tier order, each
    /// tier's service of its routed leg of the set, given the tier; each such
    /// leg is then issued, and the set's upload waits for
    /// [`BatchPort::settle_tiers`]. `tier` is not called for a tier whose leg
    /// of the set is idle.
    fn serve_tiered(
        &mut self,
        key: BatchKey,
        mut tier: impl FnMut(usize, TierServe<'_>) -> Result<(), GpuError>,
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
        let routed = &set.routed;
        super::await_done("BatchPort::serve", || routed.query())?;
        let t1 = Instant::now();
        for (t, port) in self.tiers.iter_mut().enumerate() {
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
                    tier(
                        t,
                        TierServe {
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
                        },
                    )?;
                    l.leg = Leg::Issued(key);
                    stats.served += 1;
                }
                other => {
                    return Err(GpuError::Shape {
                        what: PORT,
                        detail: format!(
                            "the serve of {key:?}; the set's leg of tier {t} holds {other:?}"
                        ),
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

    /// Wait, until `deadline`, for every tier's service of the oldest
    /// served set — `key`'s, and tiered, else refused by name — after which
    /// each served leg's rows are what the stage card reads: that tier's down
    /// outputs of the block's slots, slot-major from its first slot
    /// ([`BatchPort::tier_rows`]). Returns the tiers, one bit a tier, whose
    /// event has not completed by `deadline`: the caller names them lost. A
    /// second call on a settled set returns 0 at once.
    pub fn settle_tiers(&mut self, key: BatchKey, deadline: Instant) -> Result<u32, GpuError> {
        const WHAT: &str = "BatchPort::settle_tiers";
        let set = &self.sets[self.up];
        if self.tiers.is_empty() {
            return Err(GpuError::state(WHAT, "a tier leg (BatchPort::attach_tier)"));
        }
        if set.stage != Stage::Served(key) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the tier's rows of {key:?}; the oldest served exchange set holds {:?}",
                    set.stage
                ),
            });
        }
        let mut served = false;
        for (t, port) in self.tiers.iter().enumerate() {
            match port.legs[self.up].leg {
                Leg::Idle => {}
                Leg::Settled(k) | Leg::Issued(k) if k == key => served = true,
                other => {
                    return Err(GpuError::Shape {
                        what: WHAT,
                        detail: format!(
                            "the tier's rows of {key:?}; the set's leg of tier {t} holds {other:?}"
                        ),
                    });
                }
            }
        }
        if !served {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("the tier's rows of {key:?}; the set's tier legs hold Idle"),
            });
        }
        let mut late = 0u32;
        for (t, port) in self.tiers.iter_mut().enumerate() {
            let l = &mut port.legs[self.up];
            if l.leg != Leg::Issued(key) {
                continue;
            }
            let t0 = Instant::now();
            let mut early = true;
            let mut landed = true;
            while !l.done.query()? {
                early = false;
                if Instant::now() > deadline {
                    landed = false;
                    break;
                }
                std::thread::yield_now();
            }
            if !landed {
                late |= 1 << t;
                continue;
            }
            let st = &mut port.stats;
            st.settles += 1;
            st.settle_early += u64::from(early);
            st.settle_ns += nanos(t0.elapsed());
            l.leg = Leg::Settled(key);
        }
        Ok(late)
    }

    /// Tier `tier`'s rows of the oldest served set — `key`'s, its leg
    /// settled ([`BatchPort::settle_tiers`]), else refused by name — as the
    /// stage card's card sum reads them.
    pub fn tier_rows(&self, tier: usize, key: BatchKey) -> Result<&DeviceBuffer<f32>, GpuError> {
        const WHAT: &str = "BatchPort::tier_rows";
        let l = self
            .tiers
            .get(tier)
            .map(|p| &p.legs[self.up])
            .ok_or(GpuError::state(WHAT, "a tier leg (BatchPort::attach_tier)"))?;
        if self.sets[self.up].stage != Stage::Served(key) || l.leg != Leg::Settled(key) {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "tier {tier}'s rows of {key:?}; the oldest served exchange set holds {:?} and \
                     its leg {:?}",
                    self.sets[self.up].stage, l.leg
                ),
            });
        }
        Ok(&l.rows_stage)
    }

    /// What each tier's batch services have done since load, in tier order.
    #[must_use]
    pub fn tier_stats(&self) -> Vec<TierBatchStats> {
        self.tiers.iter().map(|t| t.stats).collect()
    }

    /// Enqueue the copy of the oldest served set's sums, which must be
    /// `key`'s, to `hsum`; the set is free again. A tiered set's tier rows
    /// must have been waited for ([`BatchPort::settle_tiers`]) — a join
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
        for (t, port) in self.tiers.iter().enumerate() {
            match port.legs[self.up].leg {
                Leg::Idle => {}
                Leg::Settled(k) if k == key => {}
                other => {
                    return Err(GpuError::Shape {
                        what: PORT,
                        detail: format!(
                            "the upload of {key:?} before its tier rows were joined; the set's \
                             leg of tier {t} holds {other:?}"
                        ),
                    });
                }
            }
        }
        for port in &mut self.tiers {
            port.legs[self.up].leg = Leg::Idle;
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
/// down outputs by slot, and those copied into the set's rows — or, for a
/// [`BlockRows::Packed`] tier, its slots' rows packed at their front and
/// those alone copied, one row for each slot the places do not leave to the
/// host; the set's event.
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
    let on_tier = places.iter().filter(|&&p| p != HOST).count();
    tier.hit(layer, on_tier as u64);
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
    let copied = match tier.block_rows() {
        BlockRows::Staged => cols * s,
        BlockRows::Packed => on_tier,
    };
    tier.enqueue_block(
        layer,
        TierBlock {
            x: &*stage_x,
            act: &*act,
            sel: &*stage_tsel,
            cols,
            down: &mut *y,
        },
    )?;
    let stream = tier.gpu().stream();
    if copied > 0 {
        // SAFETY: `rows_len` is the rows' allocation's length (their window
        // spans all of it); the stage card reads them only after the host
        // has seen the event recorded below complete.
        unsafe { dtoh_mapped(stream, (rows, rows_len), y, copied * n)? };
    }
    done.record(stream)?;
    Ok(())
}

impl<H: HostExperts> HostTier<H> {
    /// The batch port's tier legs, once the host tier holds both the port
    /// ([`HostTier::prepare_batch`]) and its expert tiers: per tier and set
    /// the places, the rows and the tier's event, and the tier card's staging
    /// ([`BatchPort::attach_tier`]). Nothing without a tier; made once.
    /// Load-time or first-prompt only; what it holds is checked against the
    /// plan by [`HostTier::check_tier_reserves`].
    pub fn prepare_tier_batch(&mut self) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::prepare_tier_batch";
        let stage = Arc::clone(self.step.boundary.region.context());
        let Some(port) = self.port.as_mut() else {
            return Ok(());
        };
        if self.tiers.is_empty() || !port.tiers.is_empty() {
            return Ok(());
        }
        for tier in &self.tiers {
            let r = port.attach_tier(&stage, tier.gpu());
            let back = stage.bind_to_thread();
            r?;
            back.map_err(|e| GpuError::shape(WHAT, format!("rebinding the stage context: {e}")))?;
        }
        Ok(())
    }

    /// The expert tiers' prompt-batch bytes against the reserves `machine`'s
    /// plan carries for them: on each tier's card — tier `t`'s is index
    /// `cards[t]` of [`Machine::all_cards`] — its batch staging and the
    /// architecture's block scratch ([`TierCard::block_bytes`]) against that
    /// card's [`TIER_BATCH_RESERVE`] row; on the host, the tier legs' rows and
    /// places against the host's [`TIER_BATCH_HOST_RESERVE`] row. A card list
    /// that is not one card a tier, a missing row, a row named twice, or a
    /// difference is refused by name, naming the reserve. Load-time only, once
    /// [`HostTier::prepare_batch`] has made the legs.
    pub fn check_tier_reserves(&self, machine: &Machine, cards: &[usize]) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::check_tier_reserves";
        if self.tiers.is_empty() || cards.len() != self.tiers.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} plan cards for the host tier's {} expert tiers",
                    cards.len(),
                    self.tiers.len()
                ),
            ));
        }
        let port = self.port.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch port's sets (HostTier::prepare_batch)",
        })?;
        let mut rows = Vec::with_capacity(cards.len() + 1);
        for (i, (t, &index)) in self.tiers.iter().zip(cards).enumerate() {
            let card = machine.card(index).ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!(
                        "the placement has no card {index} for the expert tier {}",
                        t.name()
                    ),
                )
            })?;
            rows.push((
                card.name.as_str(),
                &card.reserves,
                TIER_BATCH_RESERVE,
                port.tier_device_bytes(i) + t.block_bytes(),
            ));
        }
        rows.push((
            "the host",
            &machine.host.reserves,
            TIER_BATCH_HOST_RESERVE,
            port.tier_host_bytes(),
        ));
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

    /// [`HostTier::enqueue_download`] of a tiered layer: the slots' places
    /// of each tier too, `tsels[t]` tier `t`'s (`n_used` a token), copied for
    /// each tier that holds experts of the layer, which that tier's service
    /// reads ([`BatchPort::download_tiered`]). Refused by name for a layer no
    /// tier holds an expert of, for a place list that is not one a tier, and
    /// without a tier or its leg.
    pub fn enqueue_download_tiered(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        tsels: &[&DeviceBuffer<u32>],
        key: BatchKey,
    ) -> Result<(), GpuError> {
        let pitch = self.step.boundary.layout.handoff().n_used;
        self.enqueue_download_tiered_pitched(stream, xw, ids, pitch, tsels, key)
    }

    /// [`HostTier::enqueue_download_tiered`] from routing buffers of `pitch`
    /// entries a token whose first `n_used` are the routed slots
    /// ([`BatchPort::download_tiered_pitched`]); the tier places stay
    /// `n_used` a token.
    pub fn enqueue_download_tiered_pitched(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        pitch: usize,
        tsels: &[&DeviceBuffer<u32>],
        key: BatchKey,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::enqueue_download_tiered";
        let mask = self.tier_mask(key.layer)?;
        if mask == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {}: a tiered route of a layer the tier holds no expert of",
                    key.layer
                ),
            ));
        }
        if tsels.len() != self.tiers.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {}: {} place lists for {} expert tiers",
                    key.layer,
                    tsels.len(),
                    self.tiers.len()
                ),
            ));
        }
        let mut routed: [(usize, &DeviceBuffer<u32>); MAX_TIERS] = [(0, tsels[0]); MAX_TIERS];
        let mut k = 0;
        for (t, &places) in tsels.iter().enumerate().filter(|(t, _)| mask >> t & 1 == 1) {
            routed[k] = (t, places);
            k += 1;
        }
        self.port_mut(WHAT)?
            .download_tiered_pitched(stream, xw, ids, pitch, &routed[..k], key)
    }

    /// The oldest download not served yet — `key`'s, else refused by name —
    /// waited for; each of a tiered set's tier services enqueued on that
    /// tier's stream, in tier order ([`enqueue_tier_leg`]); then its layer's
    /// host experts served for its tokens in one union call, less the ones
    /// the expert stream sends to the card ([`HostTier::xstream_layer`]), the
    /// sums into the set the upload sends. A failed tier enqueue poisons the host tier
    /// and releases every card's waits; a poisoned host tier is refused by
    /// name before any card is given work. Returns the host time outside the
    /// union call.
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
        let (batch, experts, health, tiers) = (
            &mut self.batch,
            &mut self.experts,
            &mut self.health,
            &mut self.tiers,
        );
        let (slots, fault) = (&self.slots, self.fault.as_ref());
        let exclude = self.xstream.as_ref().map_or(&[][..], |x| x.excluded(key));
        let mut tier_failed = false;
        let r = port.serve_tiered(
            key,
            |t, io| {
                let r = match tiers.get_mut(t) {
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
                batch.serve_guarded(t, layer, x, ids, w, exclude, out)
            },
        );
        if let Err(e) = &r
            && tier_failed
        {
            self.health
                .set_poison(SERVE_BATCH, key.layer, Some(e), None);
            self.release_all();
        }
        r.map_err(|e| self.noted(e))
    }

    /// The last batch service's [`UnionCols`] — the per-expert column counts
    /// of the layer-batch `key`'s serve listed — read between that serve and
    /// its upload by a walk that times its serves (the batch leg's timed
    /// serve): the service must still be the port's oldest served, which
    /// proves the lists the counts come from are exactly that serve's. Any
    /// other state is refused by name.
    pub fn served_union_cols(&mut self, key: BatchKey) -> Result<UnionCols, GpuError> {
        const WHAT: &str = "HostTier::served_union_cols";
        let port = self.port.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "the batch port's sets (HostTier::prepare_batch)",
        })?;
        let cols = key.u.checked_sub(key.at).ok_or(GpuError::shape(
            WHAT,
            format!(
                "the counts of {key:?}: its columns run {}..{}",
                key.at, key.u
            ),
        ))?;
        if port.sets[port.up].stage != Stage::Served(key) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the counts of {key:?}; the oldest served set holds {:?}",
                    port.sets[port.up].stage
                ),
            ));
        }
        let n_used = self.step.boundary.layout.handoff().n_used;
        Ok(self.batch.col_counts(cols, n_used))
    }

    /// Tier `tier`'s rows of `key`'s set as the stage card's card sum reads
    /// them ([`BatchPort::tier_rows`]), once the host has seen every tier's
    /// service of the set complete ([`BatchPort::settle_tiers`]), under the
    /// go deadline: a tier that has not finished by then is lost — the host
    /// tier is poisoned as a lost card, every card's waits released, and the
    /// error names each card late.
    pub fn tier_rows_of(
        &mut self,
        key: BatchKey,
        tier: usize,
    ) -> Result<&DeviceBuffer<f32>, GpuError> {
        const WHAT: &str = "HostTier::tier_rows_of";
        let deadline = Instant::now() + GO_DEADLINE;
        let late = self.port_mut(WHAT)?.settle_tiers(key, deadline)?;
        if late != 0 {
            return Err(self.lose_tiers_as(SERVE_BATCH, key.layer, late, |t, _| {
                format!(
                    "the expert tier on {} did not finish its service of the prompt block's \
                     columns {}..{} in {GO_DEADLINE:?}: the card is lost",
                    t.name(),
                    key.at,
                    key.u
                )
            }));
        }
        self.port_mut(WHAT)?.tier_rows(tier, key)
    }

    /// The expert tiers' fault words once their streams have drained,
    /// merged (the first layer wins); `None` without a tier or while every
    /// word is clean, and for a tier lost as a card
    /// ([`super::PoisonKind::CardLost`]), whose stream may never drain — the
    /// loss is the error, no reset lifts it, and every later call is refused
    /// at its group's start ([`HostTier::begin_group`]) before it touches a
    /// set that stream could still touch. The stage card's context is current
    /// again on return. A prompt call reads it at a group's end and when a
    /// group fails, beside the stage card's.
    pub fn tier_fault(&mut self) -> Result<Option<Fault>, GpuError> {
        if self.tiers.is_empty()
            || self
                .health
                .poison
                .as_ref()
                .is_some_and(|p| p.mark().kind == super::PoisonKind::CardLost)
        {
            return Ok(None);
        }
        let stage = Arc::clone(self.step.boundary.region.context());
        let mut words = [None; MAX_TIERS];
        for (w, t) in words.iter_mut().zip(&self.tiers) {
            let r = t.gpu().fault();
            let back = stage.bind_to_thread();
            *w = r?;
            back?;
        }
        Ok(crate::fault::read_cards(&words[..self.tiers.len()]))
    }

    /// Device bytes the tiers' batch staging holds on the tier cards,
    /// summed ([`BatchPort::tier_device_bytes`]); 0 before it is made.
    #[must_use]
    pub fn tier_batch_bytes(&self) -> usize {
        self.port.as_ref().map_or(0, |p| {
            (0..self.tiers.len()).map(|t| p.tier_device_bytes(t)).sum()
        })
    }

    /// What each tier's batch services have done since load, in tier order;
    /// empty before the legs are made.
    #[must_use]
    pub fn tier_batch_stats(&self) -> Vec<TierBatchStats> {
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
    /// The per-expert column counts of the last served layer-batch, a column
    /// a listed expert: [`BatchService::col_counts`]'s scratch, grown once to
    /// the widest expert id any serve listed and read only by a walk that
    /// times its serves.
    counts: Vec<u32>,
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
                    Some(Slot::Card(_) | Slot::Tier { .. }) => {}
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
            // A prompt call's union has consumed what it read: the host lets
            // go of the pages it does not keep.
            let r = r.and_then(|()| t.experts.release_union(layer, &lists));
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
            .experts_union_into(layer, clean.view(), &lists, out)
            .and_then(|()| t.experts.release_union(layer, &lists));
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

    /// The last served layer-batch's [`UnionCols`], counted over the lists
    /// [`BatchService::serve_one`] built for its `cols` columns of the routed
    /// width `n_used` — the same (expert, column) pairs its union plan
    /// sorts, a column that names an expert twice counting it once, as the
    /// plan does. The scratch grows once, to the widest expert id a serve
    /// listed; no serve allocates and no lock is taken.
    fn col_counts(&mut self, cols: usize, n_used: usize) -> UnionCols {
        let (lens, lists, counts) = (&self.lens, &self.lists, &mut self.counts);
        counts.fill(0);
        for j in 0..cols {
            let list = &lists[j * n_used..][..lens[j]];
            for (i, &(id, _)) in list.iter().enumerate() {
                if list[..i].iter().any(|&(seen, _)| seen == id) {
                    continue;
                }
                let id = id as usize;
                if id >= counts.len() {
                    counts.resize(id + 1, 0);
                }
                counts[id] += 1;
            }
        }
        let mut r = UnionCols::default();
        for &m in counts.iter() {
            if m == 0 {
                continue;
            }
            r.experts += 1;
            r.m_max = r.m_max.max(m as usize);
            if m as usize > HOT_COLS {
                r.m_hot += 1;
                r.cols_hot += m as usize;
            }
            r.m_sq += u64::from(m) * u64::from(m);
        }
        r
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
            Some(Slot::Tier { .. }) => "the tier card",
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
