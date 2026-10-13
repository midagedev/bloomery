//! The link the tier page crosses: how the stage card, each tier card and
//! the host reach the page's bytes ([`super::tier::TierPage`]). The tier
//! protocol takes every device address it enqueues from one side's end
//! ([`LinkEnd`]) and every word the host touches from the [`Link`], so where
//! the bytes sit is this module's alone.
//!
//! A link is one of three kinds ([`Link::new`]), and the tier protocol's code
//! is the same over all of them.
//!
//! - The shared page ([`Link::shared`]): one host-mapped allocation
//!   ([`MappedHost`]) that the stage card's context allocates and every tier
//!   card's context reaches at the same device address, which a tier's end
//!   checks when it is made ([`Link::tier_end`]). Each word is one location
//!   that both cards and the host read and write.
//! - Mirrored pages: the stage card's page and the tier cards' page, each
//!   held by one end ([`Side`]) that writes only its own page and reaches the
//!   other through its wire ([`pump`]), as two linked nodes do. Each counter
//!   has one writer card on each page; the cards' operations are the shared
//!   page's. The loopback ([`Link::loopback`]) carries a post by copying it
//!   in process, one pump thread driving both ends; the RDMA link
//!   ([`Link::rdma`]) carries it as RDMA writes between two queue pairs on
//!   one RoCE port, each end with its own device context and pump thread
//!   ([`rdma::RdmaEnd`]), which is what one node holds of a link between
//!   two.
//!
//! The host touches the page's flag words through the link alone: it reads a
//! word ([`Link::load`]), stores one ([`Link::store`]),
//! checks at rest and resets a counter ([`Link::at_rest`], [`Link::reset`]),
//! and releases a tier's counters, all of them in one call
//! ([`Link::release`]). On the shared page each word access is one load or
//! store, and a release one store a counter; on mirrored pages each end
//! acts on its own page, and a release stops all of its counters on both
//! ends before it raises any.

mod pump;
mod rdma;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bloomery_levers::TierLink;
use cuda_core::{CudaContext, sys};
use threads::helper::{HelperError, Placement, spawn_helper};

use crate::GpuError;
use crate::graph::{MappedHost, cu};
pub use pump::{Crossing, Reader};
use pump::{Half, Move, Page, TAIL_LINE, Wire};
pub use rdma::RdmaEnd;

/// How long a reset waits for an end's posts in flight to finish.
const RESET_DRAIN: Duration = Duration::from_secs(1);

/// The tier page's bytes as every side reaches them: the stage card's page,
/// which every tier card reaches too on the shared page, and on mirrored
/// pages the two ends.
///
/// Field order is drop order: the pumps stop before the pages go.
pub(super) struct Link {
    mirror: Option<Mirror>,
    page: Arc<MappedHost>,
}

/// The two ends of mirrored pages and the pump threads that drive them.
///
/// Field order is drop order: the pumps stop before the ends go.
struct Mirror<P = MappedHost> {
    pumps: Vec<Pump>,
    stage: Arc<dyn Side<P>>,
    tier: Arc<dyn Side<P>>,
    /// A test's tier card that runs on once released ([`Link::run_on`]).
    #[cfg(test)]
    run_on: Option<(usize, u32)>,
}

impl pump::Page for MappedHost {
    fn word(&self, off: usize) -> &AtomicU32 {
        self.atomic_u32(off)
            .expect("the tier page's layout keeps every word the link touches 4-aligned inside it")
    }
}

/// The byte where a mirrored page's tail starts after `bytes` of tier
/// layout, and the page's bytes with the tail of `crossings` crossings.
#[must_use]
pub fn link_tail(bytes: usize, crossings: usize) -> (usize, usize) {
    let tail = bytes.next_multiple_of(64);
    (tail, tail + crossings * TAIL_LINE)
}

/// One end of mirrored pages: its page, its relay and its wire.
///
/// Field order is drop order: the wire (an RDMA region over the page) goes
/// before this end's hold on the page.
struct End<W, P = MappedHost> {
    state: Mutex<(Half, W)>,
    page: Arc<P>,
}

/// What a pump thread and the host do with an end, whatever its wire.
trait Side<P = MappedHost>: Send + Sync {
    /// One pass of the end's relay; `true` when anything moved.
    fn pass(&self) -> bool;
    /// Stop carrying the counter at byte `off`; the end that reads it sets
    /// it to `v`.
    fn release(&self, off: usize, v: u32);
    /// Stop carrying the counter at byte `off`.
    fn stop(&self, off: usize);
    /// Whether this end's side of the counter at byte `off` is at rest.
    fn at_rest(&self, off: usize) -> bool;
    /// The writer's reset of the counter at byte `off`, if this end writes it.
    fn reset_writer(&self, off: usize);
    /// The reader's reset of the counter at byte `off`, if this end reads it.
    fn reset_reader(&self, off: usize);
    /// Why the end stopped carrying for good.
    fn broken(&self) -> Option<String>;
    /// This end's page.
    fn page(&self) -> &P;
}

impl<W, P> End<W, P> {
    fn new(half: Half, wire: W, page: Arc<P>) -> End<W, P> {
        End {
            state: Mutex::new((half, wire)),
            page,
        }
    }

    fn lock(&self) -> MutexGuard<'_, (Half, W)> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<W: Wire, P: Page> Side<P> for End<W, P> {
    fn pass(&self) -> bool {
        let (half, wire) = &mut *self.lock();
        half.pass(&*self.page, wire)
    }

    fn release(&self, off: usize, v: u32) {
        self.lock().0.release(off, v, &*self.page);
    }

    fn stop(&self, off: usize) {
        self.lock().0.stop(off);
    }

    fn at_rest(&self, off: usize) -> bool {
        let (half, wire) = &mut *self.lock();
        half.at_rest(off, &*self.page, wire)
    }

    fn reset_writer(&self, off: usize) {
        let until = Instant::now() + RESET_DRAIN;
        let (half, wire) = &mut *self.lock();
        half.reset_writer(off, &*self.page, wire, &|| Instant::now() < until);
    }

    fn reset_reader(&self, off: usize) {
        self.lock().0.reset_reader(off, &*self.page);
    }

    fn broken(&self) -> Option<String> {
        self.lock().0.broken().map(str::to_string)
    }

    fn page(&self) -> &P {
        &self.page
    }
}

/// A pump thread: a pass over each of its ends until it drops, spinning
/// while nothing moves. It floats ([`threads::helper::spawn_helper`]): a
/// load on a step thread pinned to one cpu would otherwise hand it that cpu.
struct Pump {
    quit: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Pump {
    fn start<P: 'static>(sides: Vec<Arc<dyn Side<P>>>) -> Result<Pump, HelperError> {
        let quit = Arc::new(AtomicBool::new(false));
        let (thread, _) = {
            let quit = Arc::clone(&quit);
            spawn_helper("tier-link-pump", Placement::Float, move || {
                while !quit.load(Ordering::Acquire) {
                    let mut moved = false;
                    for s in &sides {
                        moved |= s.pass();
                    }
                    if !moved {
                        std::thread::yield_now();
                    }
                }
            })?
        };
        Ok(Pump {
            quit,
            thread: Some(thread),
        })
    }

    /// Why the link stopped carrying, if the thread ended while the pump
    /// still lives (it panicked).
    fn ended(&self) -> Option<String> {
        self.thread
            .as_ref()
            .is_some_and(JoinHandle::is_finished)
            .then(|| "the link's pump thread has ended".to_string())
    }
}

impl Drop for Pump {
    fn drop(&mut self) {
        self.quit.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            // A pump that panicked has stopped; there is nothing to carry.
            let _ = t.join();
        }
    }
}

/// The loopback's wire: a post's words copied from one page into the other
/// in order, the last with Release.
struct Memory<P = MappedHost> {
    from: Arc<P>,
    to: Arc<P>,
}

impl<P: Page> Wire for Memory<P> {
    fn post(&mut self, moves: &[Move]) -> Result<(), String> {
        for (k, m) in moves.iter().enumerate() {
            for w in (0..m.len).step_by(4) {
                let v = Page::word(&*self.from, m.from + w).load(Ordering::Relaxed);
                let last = k + 1 == moves.len() && w + 4 >= m.len;
                let o = if last {
                    Ordering::Release
                } else {
                    Ordering::Relaxed
                };
                Page::word(&*self.to, m.to + w).store(v, o);
            }
        }
        Ok(())
    }

    fn reap(&mut self) -> Result<bool, String> {
        Ok(true)
    }
}

/// One side's view of the page: the device address of each byte as that
/// side's launches, copies and stream memory operations take it.
#[derive(Clone, Copy, Debug)]
pub(super) struct LinkEnd {
    dev: sys::CUdeviceptr,
}

impl LinkEnd {
    /// The device address of byte `off` of the page, on this side.
    pub(super) fn dev(self, off: usize) -> sys::CUdeviceptr {
        self.dev + off as u64
    }
}

impl Link {
    /// The link of kind `kind` over `bytes` zeroed bytes of page: the shared
    /// page, allocated in `stage`, or mirrored pages, whose ends carry
    /// `crossings` ([`Link::loopback`], [`Link::rdma`]) and refuse one by name
    /// ([`Half::new`]). Load-time only; `stage` is current on return.
    pub(super) fn new(
        kind: TierLink,
        stage: &Arc<CudaContext>,
        tier: &Arc<CudaContext>,
        bytes: usize,
        crossings: Vec<Crossing>,
        what: &'static str,
    ) -> Result<Link, GpuError> {
        match kind {
            TierLink::Shared => Link::shared(stage, bytes),
            TierLink::Loopback => Link::loopback(stage, tier, bytes, crossings, what),
            TierLink::Rdma => Link::rdma(stage, tier, bytes, crossings, what),
        }
    }

    /// `bytes` zeroed bytes of shared page, allocated in the stage card's
    /// context `ctx`. Load-time only.
    fn shared(ctx: &Arc<CudaContext>, bytes: usize) -> Result<Link, GpuError> {
        Ok(Link {
            mirror: None,
            page: Arc::new(MappedHost::new(ctx, bytes, "cuMemHostAlloc (tier page)")?),
        })
    }

    /// The two mirrored pages: `bytes` of tier layout and the link's tail
    /// for `crossings` crossings, the stage card's allocated in `stage` and
    /// the tier cards' in `tier`. `stage` is current on return.
    fn pages(
        stage: &Arc<CudaContext>,
        tier: &Arc<CudaContext>,
        bytes: usize,
        crossings: usize,
    ) -> Result<(usize, Arc<MappedHost>, Arc<MappedHost>), GpuError> {
        let (tail, total) = link_tail(bytes, crossings);
        let tier = Arc::new(MappedHost::new(
            tier,
            total,
            "cuMemHostAlloc (tier page, tier side)",
        )?);
        let page = Arc::new(MappedHost::new(stage, total, "cuMemHostAlloc (tier page)")?);
        Ok((tail, page, tier))
    }

    /// The loopback: the two pages ([`Link::pages`]) and one pump thread
    /// that drives both ends, each copying its posts into the other page.
    /// Load-time only; `stage` is current on return.
    fn loopback(
        stage: &Arc<CudaContext>,
        tier: &Arc<CudaContext>,
        bytes: usize,
        crossings: Vec<Crossing>,
        what: &'static str,
    ) -> Result<Link, GpuError> {
        let (tail, page, tier) = Link::pages(stage, tier, bytes, crossings.len())?;
        let end = |side, own: &Arc<MappedHost>, other: &Arc<MappedHost>| {
            let half = Half::new(crossings.clone(), side, tail, &**own)
                .map_err(|e| GpuError::shape(what, e))?;
            let wire = Memory {
                from: Arc::clone(own),
                to: Arc::clone(other),
            };
            Ok::<_, GpuError>(Arc::new(End::new(half, wire, Arc::clone(own))) as Arc<dyn Side>)
        };
        let stage_end = end(Reader::Stage, &page, &tier)?;
        let tier_end = end(Reader::Tier, &tier, &page)?;
        let pump = Pump::start(vec![Arc::clone(&stage_end), Arc::clone(&tier_end)])
            .map_err(|e| GpuError::plan(what, e))?;
        Ok(Link {
            mirror: Some(Mirror {
                pumps: vec![pump],
                stage: stage_end,
                tier: tier_end,
                #[cfg(test)]
                run_on: None,
            }),
            page,
        })
    }

    /// The RDMA link on one machine: the two pages ([`Link::pages`]), each
    /// held by an RDMA end with its own device context, queue pair and pump
    /// thread, the two queue pairs connected to each other on the first RoCE
    /// port the machine reports ([`bloomery_verbs::first_roce_v2`]).
    /// Load-time only; `stage` is current on return.
    fn rdma(
        stage: &Arc<CudaContext>,
        tier: &Arc<CudaContext>,
        bytes: usize,
        crossings: Vec<Crossing>,
        what: &'static str,
    ) -> Result<Link, GpuError> {
        let (tail, page, tier) = Link::pages(stage, tier, bytes, crossings.len())?;
        let (dev, gid) = bloomery_verbs::first_roce_v2()
            .map_err(|e| GpuError::shape(what, format!("the RDMA link: {e}")))?;
        let open = |side, page: &Arc<MappedHost>| {
            RdmaEnd::new(Arc::clone(page), tail, crossings.clone(), side, &dev, gid)
        };
        let (mut s, mut t) = (open(Reader::Stage, &page)?, open(Reader::Tier, &tier)?);
        let (s_psn, t_psn) = (0x100, 0x200);
        let (se, te) = (s.endpoint(s_psn), t.endpoint(t_psn));
        s.connect(s_psn, &te)?;
        t.connect(t_psn, &se)?;
        let (s_pump, stage_end) = s.into_parts();
        let (t_pump, tier_end) = t.into_parts();
        Ok(Link {
            mirror: Some(Mirror {
                pumps: s_pump.into_iter().chain(t_pump).collect(),
                stage: stage_end,
                tier: tier_end,
                #[cfg(test)]
                run_on: None,
            }),
            page,
        })
    }

    /// The page the tier cards reach: the shared page, or the tier cards'
    /// own.
    fn tier_page(&self) -> &MappedHost {
        match &self.mirror {
            None => &self.page,
            Some(m) => m.tier.page(),
        }
    }

    /// The stage card's end.
    pub(super) fn stage_end(&self) -> LinkEnd {
        LinkEnd {
            dev: self.page.dev_at(0),
        }
    }

    /// The end of the tier card named `name` in its context `ctx`, which is
    /// current on return: the tier cards' page must sit at one address in
    /// every context that reaches it (unified addressing), the stage card's
    /// included on the shared page, else the end is refused by name as
    /// `what`.
    pub(super) fn tier_end(
        &self,
        ctx: &CudaContext,
        what: &'static str,
        name: &str,
    ) -> Result<LinkEnd, GpuError> {
        ctx.bind_to_thread()?;
        let mut dev: sys::CUdeviceptr = 0;
        let page = self.tier_page();
        // SAFETY: the tier's context is current on this thread (bound above),
        // `dev` is a live local the call writes, and the page's first byte is
        // the start of its live mapped allocation; the flags must be 0.
        let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, page.host_at(0).cast(), 0) };
        cu(rc, "cuMemHostGetDevicePointer_v2 (tier page, tier context)")?;
        if dev != page.dev_at(0) {
            let made = if self.mirror.is_none() {
                "the stage card's context"
            } else {
                "the context that made it"
            };
            return Err(GpuError::shape(
                what,
                format!(
                    "the tier page is at {:#x} in {made} and at {dev:#x} in the tier {name}'s: the \
                     tier's copies and the stage's waits need one address",
                    page.dev_at(0),
                ),
            ));
        }
        Ok(LinkEnd { dev })
    }

    /// The u32 at byte `off` of the stage card's page, which the caller's
    /// layout keeps 4-aligned and inside the page.
    fn word(&self, off: usize) -> &AtomicU32 {
        Page::word(&*self.page, off)
    }

    /// The word at byte `off` as the stage card's side holds it.
    pub(super) fn load(&self, off: usize) -> u32 {
        self.word(off).load(Ordering::Acquire)
    }

    /// Store `v` into the word at byte `off` on every page, the tier cards'
    /// (whose card writes the words a done publishes) first.
    pub(super) fn store(&self, off: usize, v: u32) {
        if let Some(m) = &self.mirror {
            Page::word(m.tier.page(), off).store(v, Ordering::Release);
        }
        self.word(off).store(v, Ordering::Release);
    }

    /// Release every wait on the counters at bytes `offs`, a tier's every
    /// counter, until their resets: the counter each reader waits on goes to
    /// `v`, far past anything a step subtracts. A tier's counters go in one
    /// call: mirrored pages stop carrying all of them on both ends before any
    /// reader's counter moves, since a released card runs on and writes the
    /// counters that would still carry.
    pub(super) fn release(&self, offs: &[usize], v: u32) {
        match &self.mirror {
            None => offs.iter().for_each(|&off| self.store(off, v)),
            Some(m) => m.release(offs, v),
        }
    }

    /// Whether the counter at byte `off` holds what a drained step leaves:
    /// no go or signal pending, and on mirrored pages every generation
    /// carried and no end broken.
    pub(super) fn at_rest(&self, off: usize) -> bool {
        match &self.mirror {
            None => self.load(off) == 0,
            Some(m) => m.at_rest(off),
        }
    }

    /// The counter at byte `off` back to rest once every stream that reaches
    /// it has drained: 0 on the shared page; on mirrored pages the counter
    /// stops on both ends, the writer's end takes its generation as carried
    /// and sends it and the progress word to the reader's page, then the
    /// reader's end takes its inbox as delivered, its counter goes to 0 and
    /// the counter carries again. Mirrored pages that stopped carrying for
    /// good, before the reset or in it, fail it by why ([`Link::broken`]).
    pub(super) fn reset(&self, off: usize) -> Result<(), String> {
        match &self.mirror {
            None => {
                self.store(off, 0);
                Ok(())
            }
            Some(m) => m.reset(off),
        }
    }

    /// Why mirrored pages stopped carrying for good: a relay that stepped by
    /// neither 0 nor 1, a post the wire failed, or a pump thread that ended.
    /// `None` on the shared page.
    pub(super) fn broken(&self) -> Option<String> {
        self.mirror.as_ref()?.broken()
    }
}

impl<P: Page> Mirror<P> {
    /// [`Link::release`] on both ends: every counter of `offs` stopped on
    /// both before any reader's counter goes to `v`.
    fn release(&self, offs: &[usize], v: u32) {
        for &off in offs {
            self.stage.stop(off);
            self.tier.stop(off);
        }
        for &off in offs {
            self.stage.release(off, v);
            self.tier.release(off, v);
        }
        #[cfg(test)]
        self.after_release();
    }

    /// [`Link::at_rest`] on both ends.
    fn at_rest(&self, off: usize) -> bool {
        self.stage.at_rest(off) && self.tier.at_rest(off)
    }

    /// [`Link::reset`] on both ends: each end resets only its own side of
    /// the counter, the writer's before the reader's.
    fn reset(&self, off: usize) -> Result<(), String> {
        self.stage.stop(off);
        self.tier.stop(off);
        self.stage.reset_writer(off);
        self.tier.reset_writer(off);
        self.stage.reset_reader(off);
        self.tier.reset_reader(off);
        self.broken().map_or(Ok(()), Err)
    }

    /// [`Link::broken`] of both ends and the pumps.
    fn broken(&self) -> Option<String> {
        self.stage
            .broken()
            .or_else(|| self.tier.broken())
            .or_else(|| self.pumps.iter().find_map(Pump::ended))
    }
}

#[cfg(test)]
impl Link {
    /// A tier card that runs on once released, for a test over mirrored
    /// pages: after each release the counter at byte `off` steps by `by` on
    /// the tier cards' page, and each end makes one pass before the host
    /// goes on.
    pub(super) fn run_on(&mut self, off: usize, by: u32) {
        self.mirror.as_mut().expect("mirrored pages").run_on = Some((off, by));
    }
}

#[cfg(test)]
impl<P: Page> Mirror<P> {
    /// One pass of each end; `true` when anything moved.
    fn pass(&self) -> bool {
        let s = self.stage.pass();
        s | self.tier.pass()
    }

    /// [`Link::run_on`]'s step, after a release has dropped the ends' locks.
    fn after_release(&self) {
        if let Some((off, by)) = self.run_on {
            Page::word(self.tier.page(), off).fetch_add(by, Ordering::Release);
            self.pass();
        }
    }
}
