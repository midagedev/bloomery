//! The RDMA end of mirrored pages: one page registered with its NIC, one
//! queue pair to the other end, and the pump thread that drives it. Each
//! post is one chain of RDMA writes on that queue pair, in the post's order,
//! each signaled; the page is registered without relaxed ordering,
//! so on the NIC this link runs on (mlx5, whose writes on one queue pair
//! land in their order) the generation lands last (`docs/contrib/dgx-spark.md`
//! 9.5, item 4). A write that completes with an error breaks the end by its
//! status name. A release takes the queue pair of an end whose writes do not
//! finish to its error state, which flushes each of them to a completion
//! ([`Wire::abort`]).

use std::sync::Arc;

use bloomery_verbs::{Endpoint, MAX_CHAIN, Nic, Qp, Region, WC_RDMA_WRITE, Wc, Write};

use super::pump::{Crossing, Half, Move, Reader, Wire};
use super::{End, Pump, Side};
use crate::GpuError;
use crate::graph::MappedHost;

/// The RDMA wire: this end's page as one region, its queue pair, and the
/// other page's address and key once connected.
///
/// Field order is drop order: the queue pair goes before the region.
struct Rdma {
    qp: Qp,
    region: Region,
    peer: Option<(u64, u32)>,
    /// Writes posted whose completion was not yet polled.
    in_flight: usize,
    wc: [Wc; 16],
    /// One post's writes, with room for the longest chain.
    writes: Vec<Write>,
}

impl Wire for Rdma {
    fn post(&mut self, moves: &[Move]) -> Result<(), String> {
        let (addr, rkey) = self
            .peer
            .ok_or("the link's RDMA end posted before it was connected")?;
        self.writes.clear();
        for m in moves.iter().filter(|m| m.len > 0) {
            self.writes.push(Write {
                from: m.from as u64,
                to: addr + m.to as u64,
                len: u32::try_from(m.len).map_err(|_| format!("a write of {} bytes", m.len))?,
            });
        }
        // SAFETY: region and page outlive the queue pair (field order here and in `End`), so
        // a range stays registered and allocated while a write reads it (dgx-spark.md 9.5, item 2).
        let posted = unsafe { self.qp.post_writes(&self.region, &self.writes, rkey) };
        self.in_flight += match &posted {
            Ok(()) => self.writes.len(),
            Err((taken, _)) => *taken,
        };
        posted.map_err(|(_, e)| format!("the link's RDMA post: {e}"))
    }

    fn reap(&mut self) -> Result<bool, String> {
        if self.in_flight == 0 {
            return Ok(true);
        }
        let nic = self.qp.nic();
        let n = nic
            .poll(&mut self.wc)
            .map_err(|e| format!("the link's completion queue: {e}"))?;
        let mut failed = None;
        for w in &self.wc[..n] {
            self.in_flight = self
                .in_flight
                .checked_sub(1)
                .ok_or("the link's completion queue returned a write never posted")?;
            // A failed completion's opcode is undefined (`ibv_poll_cq(3)`).
            let why = if w.status != 0 {
                format!(
                    "the link's RDMA write failed: {} (status {}, vendor error {:#x})",
                    nic.verbs().status_name(w.status),
                    w.status,
                    w.vendor_err
                )
            } else if w.opcode != WC_RDMA_WRITE {
                format!("the link's completion queue returned opcode {}", w.opcode)
            } else {
                continue;
            };
            failed.get_or_insert(why);
        }
        failed.map_or(Ok(self.in_flight == 0), Err)
    }

    fn abort(&mut self) -> Result<(), String> {
        self.qp
            .to_error()
            .map_err(|e| format!("the link's queue pair: {e}"))
    }
}

/// One RDMA end of mirrored pages, as a node holds it: its page, the
/// crossings whose tail lines start at byte `tail` of both pages, its queue
/// pair and, once started, its pump thread.
///
/// Field order is drop order: the pump stops before the end goes.
pub struct RdmaEnd {
    pump: Option<Pump>,
    end: Arc<End<Rdma>>,
}

impl RdmaEnd {
    /// The end on side `side` over `page`, carrying `crossings` with their
    /// tail lines from byte `tail` ([`super::link_tail`]), on port 1 of the
    /// RoCE device `dev` with GID index `gid`. The relay takes every word as
    /// `page` holds it now. A crossing whose counter, progress word or span
    /// is not 4-aligned before the tail is refused by name. Load-time only.
    pub fn new(
        page: Arc<MappedHost>,
        tail: usize,
        crossings: Vec<Crossing>,
        side: Reader,
        dev: &str,
        gid: u8,
    ) -> Result<RdmaEnd, GpuError> {
        let what = "the link's RDMA end";
        let depth: usize = crossings.iter().map(|c| c.spans.len() + 2).sum();
        let half =
            Half::new(crossings, side, tail, &*page).map_err(|e| GpuError::shape(what, e))?;
        let nic = Nic::open(dev, gid).map_err(|e| GpuError::shape(what, e.to_string()))?;
        // SAFETY: the end holds `page` after its wire, so the page drops after
        // the region: the range stays allocated while it is registered.
        let region = unsafe { nic.register(page.host_at(0), page.bytes()) }
            .map_err(|e| GpuError::shape(what, e.to_string()))?;
        let qp = nic
            .queue_pair(
                u32::try_from(depth.max(1)).map_err(|_| GpuError::shape(what, "queue depth"))?,
            )
            .map_err(|e| GpuError::shape(what, e.to_string()))?;
        let wire = Rdma {
            qp,
            region,
            peer: None,
            in_flight: 0,
            wc: [Wc::default(); 16],
            writes: Vec::with_capacity(MAX_CHAIN),
        };
        Ok(RdmaEnd {
            pump: None,
            end: Arc::new(End::new(half, wire, page)),
        })
    }

    /// What the other end needs to connect to this one, starting this end's
    /// sends at `psn`.
    pub fn endpoint(&self, psn: u32) -> Endpoint {
        let wire = &self.end.lock().1;
        Endpoint {
            qpn: wire.qp.qpn(),
            psn,
            gid: wire.qp.nic().gid(),
            addr: wire.region.addr(),
            rkey: wire.region.rkey(),
        }
    }

    /// Connect to the other end's `remote`, this end's sends starting at
    /// `psn`, and start the pump thread.
    pub fn connect(&mut self, psn: u32, remote: &Endpoint) -> Result<(), GpuError> {
        let what = "the link's RDMA end";
        {
            let wire = &mut self.end.lock().1;
            wire.qp
                .connect(psn, remote)
                .map_err(|e| GpuError::shape(what, e.to_string()))?;
            wire.peer = Some((remote.addr, remote.rkey));
        }
        let side: Arc<dyn Side> = self.end.clone();
        self.pump = Some(Pump::start(vec![side]).map_err(|e| GpuError::plan(what, e))?);
        Ok(())
    }

    /// Release every counter `ends` carry until the ends drop: each end stops
    /// carrying all of them, every end settles (its posts over, or its queue
    /// pair taken to its error state and its writes flushed), and only then
    /// does each end that settled set the counters it reads to `v`. Ends over
    /// one page go in one call: a released card rewrites spans that any of
    /// their posts may read. An end that does not settle leaves its counters
    /// as they are, so a card waiting on them keeps waiting: the
    /// [`GpuError::Protocol`] names it.
    pub fn release(ends: &[RdmaEnd], v: u32) -> Result<(), GpuError> {
        let sides: Vec<&dyn Side> = ends.iter().map(|e| &*e.end as &dyn Side).collect();
        super::release_settled(&sides, v).map_err(|e| GpuError::protocol("the link's RDMA end", e))
    }

    /// Why the end stopped carrying for good: a step that was neither 0 nor
    /// 1, a failed write, or a pump thread that ended.
    pub fn broken(&self) -> Option<String> {
        self.end
            .broken()
            .or_else(|| self.pump.as_ref().and_then(Pump::ended))
    }

    /// The end's pump and the end itself, for a [`super::Link`] to hold.
    pub(super) fn into_parts(self) -> (Option<Pump>, Arc<dyn Side>) {
        (self.pump, self.end)
    }
}
