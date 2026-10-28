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
//! A tier served by another process on the same machine (a tier node,
//! [`super::node`]) reaches a shared page through a file under `/dev/shm`
//! that the stage card's process makes ([`Link::across`]) and the node's
//! process opens and registers in its own tier card's context
//! ([`Link::open_across`]): one physical page, at its own address on each
//! side, so each word is still one location and each side's host reaches it
//! as the shared page's. The loopback's tier side is a page of the stage
//! card's process, so a tier on a node takes no loopback.
//!
//! The host touches the page's flag words through the link alone: it reads a
//! word ([`Link::load`]), stores one ([`Link::store`]),
//! checks at rest and resets a counter ([`Link::at_rest`], [`Link::reset`]),
//! and releases every counter of the page in one call ([`Link::release`]).
//! On the shared page each word access is one load or store, and a release
//! one store a counter; on mirrored pages each end acts on its own page, and
//! a release stops every counter on both ends and settles both ends before
//! it raises any.

mod gather;
mod pump;
mod rdma;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bloomery_levers::TierLink;
use cuda_core::{CudaContext, sys};
use threads::helper::{HelperError, Placement, spawn_helper};

use crate::GpuError;
use crate::graph::{MappedHost, SHARED_PAGE, cu};
pub use gather::{GatherLoop, GatherSet, GatherShape, Gathers};
pub use pump::{Crossing, PostCols, Reader, Span};
use pump::{Half, Move, Page, TAIL_LINE, Wire};
pub use rdma::RdmaEnd;

/// How long a reset, release or drop waits for an end's posts in flight to
/// finish, and a release or drop then for the posts it aborted.
const DRAIN: Duration = Duration::from_secs(1);

/// Where a page shared with a tier node's process lives, and how its file's
/// name starts ([`Link::across`]); [`Link::open_across`] opens no other.
const ACROSS_DIR: &str = "/dev/shm";
const ACROSS_NAME: &str = "bloomery-tier-";

/// The tier page's bytes as every side reaches them: the stage card's page,
/// which every tier card reaches too on the shared page, and on mirrored
/// pages the two ends.
///
/// Field order is drop order: the pumps stop before the pages go.
pub(super) struct Link {
    mirror: Option<Mirror>,
    remote: Option<Remote>,
    page: Arc<MappedHost>,
    /// On a shared page the counters this end releases; mirrored ends hold
    /// their own ([`Side::counters`]).
    counters: Vec<usize>,
    /// The file the page is shared through, while its name stands for a tier
    /// node's process to open ([`Link::unlink`]); `None` for a page of this
    /// process alone and on the side that opened it.
    path: Option<PathBuf>,
    /// The columns every end over this page posts ([`PostCols`]).
    post_cols: PostCols,
}

/// The two ends of mirrored pages and the pump threads that drive them.
///
/// The pumps stop before the ends settle and drop. An end whose posts
/// cannot be shown over keeps both ends alive ([`Mirror`]'s `Drop`).
struct Mirror<P = MappedHost> {
    pumps: Vec<Pump>,
    stage: Arc<dyn Side<P>>,
    tier: Arc<dyn Side<P>>,
    /// A test's tier card that runs on once released ([`Link::run_on`]).
    #[cfg(test)]
    run_on: Option<(usize, u32)>,
}

/// This process's ends of remote links, sharing its one page.
#[derive(Default)]
struct Remote {
    pumps: Vec<Pump>,
    ends: Vec<Arc<dyn Side>>,
    counters: Vec<Vec<usize>>,
}

impl Drop for Remote {
    fn drop(&mut self) {
        self.pumps.clear();
        drop_settled(&self.ends);
    }
}

impl pump::Page for MappedHost {
    fn bytes(&self) -> usize {
        self.bytes()
    }

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
    /// Bring the end to where nothing it posted can still be read
    /// ([`Half::settle`], within twice [`DRAIN`]): `false`, with the end
    /// broken, when a post is not known to be over.
    fn settle(&self) -> bool;
    /// The counter of every crossing the end carries.
    fn counters(&self) -> Vec<usize>;
    /// Why the end is broken, if the last release left its counters where
    /// they were ([`Half::settle`]).
    fn unsettled(&self) -> Option<String>;
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

    fn settle(&self) -> bool {
        let (half, wire) = &mut *self.lock();
        half.settle(wire, DRAIN)
    }

    fn counters(&self) -> Vec<usize> {
        self.lock().0.counters()
    }

    fn unsettled(&self) -> Option<String> {
        let half = &self.lock().0;
        half.broken()
            .filter(|_| half.unsettled())
            .map(str::to_string)
    }

    fn at_rest(&self, off: usize) -> bool {
        let (half, wire) = &mut *self.lock();
        half.at_rest(off, &*self.page, wire)
    }

    fn reset_writer(&self, off: usize) {
        let until = Instant::now() + DRAIN;
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

    fn abort(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// Release every counter each end of `ends` carries: every counter stops on
/// its end, then every end settles ([`Side::settle`]), and only then does any
/// counter move, since a released card runs on, steps the counters it writes
/// and rewrites the spans its end's posts read. An end that does not settle
/// keeps its counters where they are, so its cards keep waiting rather than
/// rewrite a span a write may still read: `Err` names each such end.
fn release_settled<P>(ends: &[&dyn Side<P>], v: u32) -> Result<(), String> {
    let offs: Vec<Vec<usize>> = ends.iter().map(|end| end.counters()).collect();
    for (end, offs) in ends.iter().zip(&offs) {
        for &off in offs.iter() {
            end.stop(off);
        }
    }
    let settled: Vec<bool> = ends.iter().map(|end| end.settle()).collect();
    let mut held = Vec::new();
    for ((end, offs), settled) in ends.iter().zip(&offs).zip(settled) {
        if settled {
            for &off in offs.iter() {
                end.release(off, v);
            }
        } else {
            held.extend(end.unsettled());
        }
    }
    if held.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the tier link left the waits on its counters standing, so the cards that read them keep \
         waiting and a synchronize of their streams does not return: {}",
        held.join("; ")
    ))
}

/// After their pumps stop, settle the ends before their wires and pages
/// drop. An end a release left unsettled still has cards waiting on its
/// page, so it keeps every end alive even if its posts have since finished.
fn drop_settled<P>(ends: &[Arc<dyn Side<P>>]) {
    let held: Vec<String> = ends
        .iter()
        .filter_map(|end| {
            end.unsettled().or_else(|| {
                end.settle();
                end.unsettled()
            })
        })
        .collect();
    if held.is_empty() {
        return;
    }
    for end in ends {
        std::mem::forget(Arc::clone(end));
    }
    // `eprintln!` panics when stderr fails, and a panic in a drop aborts.
    let _ = writeln!(
        std::io::stderr(),
        "tier link drop: leaked its ends, their queue pairs and pages: {}",
        held.join("; ")
    );
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
            TierLink::Shared => {
                Link::shared(stage, bytes, crossings.iter().map(|c| c.counter).collect())
            }
            TierLink::Loopback => Link::loopback(stage, tier, bytes, crossings, what),
            TierLink::Rdma => Link::rdma(stage, tier, bytes, crossings, what),
        }
    }

    /// `bytes` zeroed bytes of shared page, allocated in the stage card's
    /// context `ctx`, whose counters are at bytes `counters`. Load-time only.
    fn shared(
        ctx: &Arc<CudaContext>,
        bytes: usize,
        counters: Vec<usize>,
    ) -> Result<Link, GpuError> {
        Ok(Link {
            mirror: None,
            remote: None,
            page: Arc::new(MappedHost::new(ctx, bytes, "cuMemHostAlloc (tier page)")?),
            counters,
            path: None,
            post_cols: PostCols::default(),
        })
    }

    /// At least `bytes` zeroed bytes of page in a new file under `/dev/shm`,
    /// registered in the stage card's context `ctx`, for a tier node's
    /// process to open by its path ([`Link::path`], [`Link::open_across`]),
    /// readable and writable by its user alone, whose counters are at bytes
    /// `counters`. The name goes at [`Link::unlink`], or with the link.
    /// Load-time only.
    pub(super) fn across(
        ctx: &Arc<CudaContext>,
        bytes: usize,
        counters: Vec<usize>,
    ) -> Result<Link, GpuError> {
        // An atomic count, so two loads of one process never name one file.
        static MADE: AtomicU32 = AtomicU32::new(0);
        let n = MADE.fetch_add(1, Ordering::Relaxed);
        let path = Path::new(ACROSS_DIR).join(format!("{ACROSS_NAME}{}-{n}", std::process::id()));
        let bytes = bytes.next_multiple_of(SHARED_PAGE);
        let page = MappedHost::shared_file(ctx, &path, bytes, true, "the tier page's file")?;
        Ok(Link {
            mirror: None,
            remote: None,
            page: Arc::new(page),
            counters,
            path: Some(path),
            post_cols: PostCols::default(),
        })
    }

    /// The page another process made ([`Link::across`]): the file at `path`,
    /// `bytes` long, registered in this process's tier card's context `ctx`,
    /// whose tier's counters are at bytes `counters`.
    /// A path that is not a file [`Link::across`] names (in `/dev/shm`, its
    /// name starting `bloomery-tier-`) is refused by name: the path comes
    /// over a channel with no authentication. Load-time only.
    pub(super) fn open_across(
        ctx: &Arc<CudaContext>,
        path: &Path,
        bytes: usize,
        counters: Vec<usize>,
    ) -> Result<Link, GpuError> {
        if !across_named(path) {
            return Err(GpuError::shape(
                "Link::open_across",
                format!(
                    "a page at {}: a tier page is a file {ACROSS_DIR}/{ACROSS_NAME}*",
                    path.display()
                ),
            ));
        }
        let page = MappedHost::shared_file(ctx, path, bytes, false, "the tier page's file")?;
        Ok(Link {
            mirror: None,
            remote: None,
            page: Arc::new(page),
            counters,
            path: None,
            post_cols: PostCols::default(),
        })
    }

    /// The file the page is shared through, while its name stands.
    pub(super) fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The page's bytes: what a process that opens it maps.
    pub(super) fn bytes(&self) -> usize {
        self.page.bytes()
    }

    /// Remove the file's name once every tier node has opened it: the
    /// processes that mapped the page keep it, and no other can open it.
    pub(super) fn unlink(&mut self) -> Result<(), GpuError> {
        match self.path.take() {
            Some(p) => std::fs::remove_file(&p)
                .map_err(|e| GpuError::link("Link::unlink", format!("{}: {e}", p.display()))),
            None => Ok(()),
        }
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
        let post_cols = PostCols::default();
        let end = |side, own: &Arc<MappedHost>, other: &Arc<MappedHost>| {
            let half = Half::new(crossings.clone(), side, tail, &**own, post_cols.clone())
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
            remote: None,
            page,
            counters: Vec::new(),
            path: None,
            post_cols,
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
        let post_cols = PostCols::default();
        let open = |side, page: &Arc<MappedHost>| {
            RdmaEnd::new(
                Arc::clone(page),
                tail,
                crossings.clone(),
                side,
                &dev,
                gid,
                post_cols.clone(),
            )
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
            remote: None,
            page,
            counters: Vec::new(),
            path: None,
            post_cols,
        })
    }

    /// A local page with room for every remote end's relay tail.
    pub(super) fn remote(
        ctx: &Arc<CudaContext>,
        bytes: usize,
        crossings: usize,
    ) -> Result<Link, GpuError> {
        let (_, total) = link_tail(bytes, crossings);
        let mut link = Self::shared(ctx, total, Vec::new())?;
        link.remote = Some(Remote::default());
        Ok(link)
    }

    /// An unconnected end over this page, on the control socket's RoCE port.
    pub(super) fn rdma_end(
        &self,
        tail: usize,
        crossings: Vec<Crossing>,
        side: Reader,
        ip: std::net::IpAddr,
    ) -> Result<RdmaEnd, GpuError> {
        let std::net::IpAddr::V4(ip) = ip else {
            return Err(GpuError::shape(
                "the remote tier link",
                "an IPv4 RoCE port is required",
            ));
        };
        let (dev, gid) = bloomery_verbs::roce_v2_at(ip)
            .map_err(|e| GpuError::link("the remote tier link", e.to_string()))?;
        RdmaEnd::new(
            Arc::clone(&self.page),
            tail,
            crossings,
            side,
            &dev,
            gid,
            self.post_cols.clone(),
        )
    }

    /// The columns every end over this page posts, which the tier's work sets
    /// for each chain it runs.
    pub(super) fn post_cols(&self) -> PostCols {
        self.post_cols.clone()
    }

    /// Keep a connected end and its pump until this page drops.
    pub(super) fn keep_remote(&mut self, end: RdmaEnd) {
        let remote = self.remote.as_mut().expect("a remote page holds its ends");
        let (pump, side) = end.into_parts();
        remote.pumps.extend(pump);
        remote.counters.push(side.counters());
        remote.ends.push(side);
    }

    pub(super) fn is_remote(&self) -> bool {
        self.remote.is_some()
    }

    /// One reset phase on this process's side, after both cards have drained.
    pub(super) fn reset_remote(&self, phase: u8, counters: &[usize]) -> Result<(), String> {
        let remote = self.remote.as_ref().ok_or("not a remote page")?;
        for &off in counters {
            let end = remote
                .ends
                .iter()
                .zip(&remote.counters)
                .find(|(_, counters)| counters.contains(&off))
                .map(|(end, _)| end)
                .ok_or_else(|| format!("no remote end carries counter {off}"))?;
            match phase {
                0 => end.stop(off),
                1 => end.reset_writer(off),
                2 => end.reset_reader(off),
                _ => return Err(format!("unknown remote reset phase {phase}")),
            }
        }
        self.broken().map_or(Ok(()), Err)
    }

    /// The page the tier cards reach: the shared page, or the tier cards'
    /// own.
    fn tier_page(&self) -> &MappedHost {
        match &self.mirror {
            None => &self.page,
            Some(m) => m.tier.page(),
        }
    }

    /// The end of the context that made or opened the page, on the
    /// loopback the stage card's page: the device address of its first byte
    /// there.
    pub(super) fn own_end(&self) -> LinkEnd {
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

    /// Release every wait on this end's counters until their resets: the
    /// counter each reader waits on goes to `v`, far past
    /// anything a step subtracts. Mirrored pages stop carrying every counter
    /// on both ends and settle both ends before any reader's counter moves
    /// ([`release_settled`]). An end whose posts cannot be shown over leaves
    /// its own counters as they are, so its cards keep waiting: `Err` names
    /// it, and the link's drop never frees its pages.
    pub(super) fn release(&self, v: u32) -> Result<(), String> {
        if let Some(remote) = &self.remote {
            let ends: Vec<&dyn Side> = remote.ends.iter().map(|e| &**e).collect();
            return release_settled(&ends, v);
        }
        match &self.mirror {
            None => {
                self.counters.iter().for_each(|&off| self.store(off, v));
                Ok(())
            }
            Some(m) => m.release(v),
        }
    }

    /// Whether the counter at byte `off` holds what a drained step leaves:
    /// no go or signal pending, and on mirrored pages every generation
    /// carried and no end broken.
    pub(super) fn at_rest(&self, off: usize) -> bool {
        if let Some(remote) = &self.remote {
            return remote
                .ends
                .iter()
                .zip(&remote.counters)
                .find(|(_, counters)| counters.contains(&off))
                .is_some_and(|(end, _)| end.at_rest(off));
        }
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
        if self.remote.is_some() {
            return Err("a remote page resets both ends over its control channel".to_string());
        }
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
        if let Some(remote) = &self.remote {
            return remote
                .ends
                .iter()
                .find_map(|e| e.broken())
                .or_else(|| remote.pumps.iter().find_map(Pump::ended));
        }
        self.mirror.as_ref()?.broken()
    }
}

impl<P: Page> Mirror<P> {
    /// [`Link::release`] on both ends.
    fn release(&self, v: u32) -> Result<(), String> {
        let released = release_settled(&[&*self.stage, &*self.tier], v);
        #[cfg(test)]
        self.after_release();
        released
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

impl<P> Drop for Mirror<P> {
    fn drop(&mut self) {
        self.pumps.clear();
        drop_settled(&[Arc::clone(&self.stage), Arc::clone(&self.tier)]);
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

impl Drop for Link {
    fn drop(&mut self) {
        if let Some(p) = self.path.take() {
            drop(std::fs::remove_file(p));
        }
    }
}

/// Whether `path` is a name [`Link::across`] gives.
fn across_named(path: &Path) -> bool {
    path.parent() == Some(Path::new(ACROSS_DIR))
        && path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with(ACROSS_NAME) && n != ACROSS_NAME)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in crate::host) fn remote_with_wire(
        ctx: &Arc<CudaContext>,
        bytes: usize,
        crossings: Vec<Crossing>,
        stalled: bool,
    ) -> Link {
        struct Held {
            stalled: bool,
            posted: bool,
        }
        impl Wire for Held {
            fn post(&mut self, _: &[Move]) -> Result<(), String> {
                self.posted = true;
                Ok(())
            }
            fn reap(&mut self) -> Result<bool, String> {
                if self.stalled && self.posted {
                    Err("completion queue polling failed".into())
                } else {
                    Ok(true)
                }
            }
            fn abort(&mut self) -> Result<(), String> {
                Ok(())
            }
        }
        let mut link = Link::remote(ctx, bytes, crossings.len()).expect("a remote page");
        let (tail, _) = link_tail(bytes, crossings.len());
        let half = Half::new(
            crossings.clone(),
            Reader::Tier,
            tail,
            &*link.page,
            link.post_cols(),
        )
        .expect("the crossings");
        let end: Arc<dyn Side> = Arc::new(End::new(
            half,
            Held {
                stalled,
                posted: false,
            },
            Arc::clone(&link.page),
        ));
        if stalled {
            let done = crossings
                .iter()
                .find(|c| c.reader == Reader::Stage)
                .expect("a writer");
            link.store(done.counter, 1);
            assert!(!end.pass(), "the completion queue must fail after posting");
            assert_eq!(
                end.broken().as_deref(),
                Some("completion queue polling failed")
            );
        }
        link.remote = Some(Remote {
            pumps: Vec::new(),
            counters: vec![end.counters()],
            ends: vec![end],
        });
        link
    }
    use super::across_named;
    use std::path::Path;

    /// A tier node opens only a file the stage side's process names.
    #[test]
    fn a_tier_node_opens_only_a_tier_pages_name() {
        assert!(across_named(Path::new("/dev/shm/bloomery-tier-41-0")));
        for p in [
            "/dev/shm/bloomery-tier-",
            "/dev/shm/other",
            "/var/tmp/bloomery-tier-1",
            "/dev/shm/x/bloomery-tier-1",
            "/dev/shm/..",
            "bloomery-tier-1",
        ] {
            assert!(!across_named(Path::new(p)), "{p}");
        }
    }
}
