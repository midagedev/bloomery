//! The RDMA end of mirrored pages: one page registered with its NIC, one
//! queue pair to the other end, and the pump thread that drives it. Each
//! post is one chain of RDMA writes on that queue pair, in the post's order,
//! only the last signaled; the page is registered without relaxed ordering,
//! so on the NIC this link runs on (mlx5, whose writes on one queue pair
//! land in their order) the generation lands last (`docs/contrib/dgx-spark.md`
//! 9.5, item 4). A write that completes with an error breaks the end by its
//! status name.

use std::sync::Arc;

use bloomery_verbs::{Endpoint, Nic, Qp, Region, WC_RDMA_WRITE, Wc, Write};

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
    /// Posts not yet completed.
    in_flight: usize,
    wc: [Wc; 16],
}

impl Wire for Rdma {
    fn post(&mut self, moves: &[Move]) -> Result<(), String> {
        let (addr, rkey) = self
            .peer
            .ok_or("the link's RDMA end posted before it was connected")?;
        let writes = moves
            .iter()
            .filter(|m| m.len > 0)
            .map(|m| {
                Ok(Write {
                    from: m.from as u64,
                    to: addr + m.to as u64,
                    len: u32::try_from(m.len).map_err(|_| format!("a write of {} bytes", m.len))?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        // SAFETY: the end holds the region and its page past the queue pair
        // (field order and `End`'s), so every range stays registered and
        // allocated until the queue pair is gone. A posted range is rewritten
        // only by the next generation of its counter, which the other side's
        // take of this one gates: a row has at most one go or done in flight.
        unsafe { self.qp.post_writes(&self.region, &writes, rkey, 0) }
            .map_err(|e| format!("the link's RDMA post: {e}"))?;
        self.in_flight += 1;
        Ok(())
    }

    fn reap(&mut self) -> Result<bool, String> {
        if self.in_flight == 0 {
            return Ok(true);
        }
        let nic = self.qp.nic();
        let n = nic
            .poll(&mut self.wc)
            .map_err(|e| format!("the link's completion queue: {e}"))?;
        for w in &self.wc[..n] {
            if w.status != 0 {
                return Err(format!(
                    "the link's RDMA write failed: {} (status {}, vendor error {:#x})",
                    nic.verbs().status_name(w.status),
                    w.status,
                    w.vendor_err
                ));
            }
            if w.opcode != WC_RDMA_WRITE {
                return Err(format!(
                    "the link's completion queue returned opcode {}",
                    w.opcode
                ));
            }
            self.in_flight = self
                .in_flight
                .checked_sub(1)
                .ok_or("the link's completion queue returned a write never posted")?;
        }
        Ok(self.in_flight == 0)
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
        // SAFETY: the end holds `page` after its wire, so the page drops
        // after the region: the range stays allocated while it is
        // registered.
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

    /// Stop carrying the counter at byte `off` until the end drops; if this
    /// end reads it, set it to `v`.
    pub fn release(&self, off: usize, v: u32) {
        self.end.release(off, v);
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
