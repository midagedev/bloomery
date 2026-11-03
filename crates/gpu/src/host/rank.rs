//! A tensor-parallel group of ranks, one process a node, and the shared
//! pieces every model's rank body runs on.
//!
//! - [`RankList`]: each rank's control address (the TCP address its bootstrap
//!   reaches) and its data addresses (the fabric addresses a peer's RoCE
//!   queue pair may reach it at). The two differ on a ring: a relayed fabric
//!   address carries marked RDMA packets and nothing else, so a control
//!   channel never assumes it.
//! - [`RankSession`]: the group's open over the control channel (rank 0
//!   listens, every other rank dials it), the agreement on the group's
//!   [`Fingerprint`] and limits, the RDMA end to every peer on its path
//!   ([`super::paths`]) over one gather page, the round record rank 0 sends
//!   every step ([`Round`]), and the failure contract ([`RankError`]).
//! - [`Sums`]: a rank's side of the sums over the ranks on a gather page,
//!   for a loopback ([`super::GatherLoop`]) or a session alike.
//! - The step's checks every rank body makes ([`streams_at`], [`alike`],
//!   [`at_pos`]).
//!
//! One rank's failure is the group's. A rank that fails releases its ends:
//! when every end settles ([`super::RdmaEnd::release`]) the group ends with
//! the failure that came first ([`RankError::Settled`]); when one does not,
//! the session names it and keeps its page and ends alive, since a card may
//! still wait on the page, and the run ends.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bloomery_verbs::Endpoint;
use cuda_core::{CudaContext, DeviceBuffer, sys};

use super::link::{GatherSet, GatherShape, Gathers, PeerPath, RdmaEnd, paths};
use super::node::wire::{In, MAX_FRAME, Out};
use super::node::{read_endpoint, write_endpoint};
use super::step::RELEASE;
use crate::graph::{MappedHost, mem_batch};
use crate::tensor::WindowMut;
use crate::{Gpu, GpuError};

pub mod dist;

/// One rank's addresses: where its control channel listens, and every
/// fabric address it holds, in the order a peer tries them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankAddr {
    pub control: SocketAddr,
    pub data: Vec<Ipv4Addr>,
}

/// The ranks of one group in rank order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankList {
    ranks: Vec<RankAddr>,
}

impl RankList {
    /// The list a rank file holds: one line a rank, in rank order,
    /// `<control ip:port> <data ip>[,<data ip>...]`; blank lines and text
    /// after `#` are ignored. No rank, a line of another shape, a control
    /// address given twice or a data address held by two ranks is refused by
    /// line.
    pub fn parse(text: &str) -> Result<RankList, GpuError> {
        const WHAT: &str = "the rank list";
        let mut ranks: Vec<RankAddr> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let bad = |why: &str| GpuError::shape(WHAT, format!("line {}: {why}", n + 1));
            let mut words = line.split_whitespace();
            let (Some(control), Some(data), None) = (words.next(), words.next(), words.next())
            else {
                return Err(bad("want <control ip:port> <data ip>[,<data ip>...]"));
            };
            let control: SocketAddr = control
                .parse()
                .map_err(|_| bad(&format!("{control:?} is not an ip:port")))?;
            let data = data
                .split(',')
                .map(|a| {
                    a.parse()
                        .map_err(|_| bad(&format!("{a:?} is not an IPv4 address")))
                })
                .collect::<Result<Vec<Ipv4Addr>, _>>()?;
            if ranks.iter().any(|r| r.control == control) {
                return Err(bad(&format!("control address {control} is given twice")));
            }
            if let Some(a) = data
                .iter()
                .enumerate()
                .find(|&(i, a)| data[..i].contains(a) || ranks.iter().any(|r| r.data.contains(a)))
            {
                return Err(bad(&format!("data address {} is given twice", a.1)));
            }
            ranks.push(RankAddr { control, data });
        }
        if ranks.is_empty() {
            return Err(GpuError::shape(WHAT, "no rank"));
        }
        Ok(RankList { ranks })
    }

    /// The list in the rank file at `path` ([`RankList::parse`]).
    pub fn read(path: &std::path::Path) -> Result<RankList, GpuError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| GpuError::shape("the rank list", format!("{}: {e}", path.display())))?;
        RankList::parse(&text)
    }

    /// The ranks in the group.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranks.len()
    }

    /// Whether the list holds no rank; a parsed list never does.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    /// Rank `rank`'s addresses; a rank past the list panics.
    #[must_use]
    pub fn rank(&self, rank: usize) -> &RankAddr {
        &self.ranks[rank]
    }
}

/// What a rank's load is, compared item by item when a group opens: the
/// source's identity (the SHA-256 over the verified digests of the files the
/// group reads), the identity of the tensor headers this rank reads (names,
/// types, shapes, companions and cuts), the normalization's form version,
/// the build, the group's rank count and this rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    pub source: [u8; 32],
    pub headers: [u8; 32],
    pub form: u32,
    pub build: String,
    pub n: u16,
    pub rank: u16,
}

/// One round of a group: what rank 0 decides and every rank runs, the slot,
/// the position its first token takes, the tokens, the prompt range they
/// belong to when they are a prompt's, and whether the slot stops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Round {
    pub slot: u32,
    pub pos: u32,
    pub tokens: Vec<u32>,
    pub prompt: Option<Range<u32>>,
    pub stop: bool,
}

/// Why a group stopped.
#[derive(Debug)]
pub enum RankError {
    /// Rank `rank` failed, `why` its own error, and every end of this rank
    /// settled.
    Settled { rank: usize, why: String },
    /// A rank failed and the named end did not settle: a card may still wait
    /// on the page, which stays allocated with the end; the run ends.
    Unsettled { end: String },
    /// A rank's load differs from rank 0's in the named item.
    Mismatch { item: &'static str },
    /// The group could not open or carry a round, as the error says.
    Gpu(GpuError),
}

impl std::fmt::Display for RankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RankError::Settled { rank, why } => {
                write!(
                    f,
                    "the rank group failed, rank {rank} first: {why}; every end settled"
                )
            }
            RankError::Unsettled { end } => {
                write!(f, "the rank group failed and an end did not settle: {end}")
            }
            RankError::Mismatch { item } => {
                write!(f, "a rank's load differs from rank 0's in its {item}")
            }
            RankError::Gpu(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RankError {}

impl From<GpuError> for RankError {
    fn from(e: GpuError) -> RankError {
        RankError::Gpu(e)
    }
}

const WHAT: &str = "the rank group";

/// The control channel's protocol version, carried in every hello.
const VERSION: u32 = 1;

/// The items a hello is compared by, in the order rank 0 compares them.
const ITEMS: [&str; 8] = [
    "version",
    "rank count",
    "rank",
    "source",
    "headers",
    "form",
    "build",
    "limits",
];

/// A rank's side of the sums over the ranks on one gather page: each slot of
/// the layout is a sum a step posts its part of ([`Sums::post`]) and later
/// adds in rank order ([`Sums::sum`]), alternating the slot's two span sets,
/// so a part posted early (an input that depends on the token alone) waits
/// on its own slot while other slots post and sum. Between two posts of one
/// slot the rank sums it, which keeps every set until each rank has read it
/// ([`Gathers`]).
pub struct Sums {
    layout: Gathers,
    page: Arc<MappedHost>,
    rank: usize,
    slots: Vec<SumSlot>,
}

/// One slot's operations: per set the post, the wait and the two in one
/// batch, the set the next post takes, and the set posted and not summed yet
/// with the values its part holds.
struct SumSlot {
    shape: GatherShape,
    post: [Vec<sys::CUstreamBatchMemOpParams>; Gathers::SETS],
    wait: [Vec<sys::CUstreamBatchMemOpParams>; Gathers::SETS],
    joined: [Vec<sys::CUstreamBatchMemOpParams>; Gathers::SETS],
    next: usize,
    posted: Option<(usize, usize)>,
}

impl Sums {
    /// Rank `rank` of `ranks`' sums of `shapes` on a page of its own in
    /// `ctx`, for a loopback to join. Fewer than two ranks, a rank past them
    /// and an empty shape are refused by name ([`Gathers::new`]).
    pub fn new(
        ctx: &Arc<CudaContext>,
        rank: usize,
        ranks: usize,
        shapes: &[GatherShape],
    ) -> Result<Sums, GpuError> {
        let layout = Gathers::new(0, ranks, rank, shapes)?;
        let page = MappedHost::new(ctx, layout.page_bytes(), "the rank's gather page")?;
        Ok(Sums::over(layout, Arc::new(page)))
    }

    /// The sums of `layout` on `page`, which holds the layout's bytes.
    fn over(layout: Gathers, page: Arc<MappedHost>) -> Sums {
        let dev = page.dev_at(0);
        let slots = (0..layout.slots())
            .map(|slot| {
                let at = |set| GatherSet { dev, slot, set };
                let post: [Vec<_>; Gathers::SETS] =
                    std::array::from_fn(|set| layout.post(at(set)).to_vec());
                let wait: [Vec<_>; Gathers::SETS] = std::array::from_fn(|set| layout.wait(at(set)));
                SumSlot {
                    shape: layout.shape(slot),
                    joined: std::array::from_fn(|set| [&post[set][..], &wait[set][..]].concat()),
                    post,
                    wait,
                    next: 0,
                    posted: None,
                }
            })
            .collect();
        Sums {
            rank: layout.rank(),
            layout,
            page,
            slots,
        }
    }

    /// The page's layout and the page, for the link that joins it to the
    /// other ranks'.
    #[must_use]
    pub fn link(&self) -> (&Gathers, &Arc<MappedHost>) {
        (&self.layout, &self.page)
    }

    /// Enqueue the post of `part`, the rank's part of slot `slot`'s sum over
    /// `part.len() / width` rows: the part copied into the rank's place of
    /// the slot's next set, then that set's post. A slot past the page, a
    /// part that is not whole rows within the slot's, and a post while the
    /// slot's last one is not summed are refused by name.
    pub fn post(
        &mut self,
        gpu: &Gpu,
        slot: usize,
        part: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let set = self.place(gpu, slot, part, "Sums::post")?;
        let s = &mut self.slots[slot];
        // SAFETY: every counter the operations name lies in the page, held as in `place`.
        unsafe { mem_batch(gpu.stream(), &mut s.post[set], "the rank's post")? };
        s.posted = Some((set, part.len()));
        s.next = (set + 1) % Gathers::SETS;
        Ok(())
    }

    /// Enqueue the wait for every other rank's part of slot `slot`'s last
    /// post and the parts' sum in rank order into `out`, over the posted
    /// part's rows. A slot with no post to sum, and an `out` of another
    /// length than the posted part, are refused by name.
    pub fn sum(
        &mut self,
        gpu: &Gpu,
        slot: usize,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Sums::sum";
        let s = self
            .slots
            .get_mut(slot)
            .ok_or_else(|| GpuError::shape(WHAT, format!("slot {slot} of a page")))?;
        let Some((set, _)) = s.posted.filter(|&(_, n)| n == out.len()) else {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "slot {slot}: a sum into {} values, posted (set, values) {:?}",
                    out.len(),
                    s.posted
                ),
            ));
        };
        // SAFETY: every counter the operations name lies in the page, held as in `place`.
        unsafe { mem_batch(gpu.stream(), &mut s.wait[set], "the rank's wait")? };
        s.posted = None;
        self.add(gpu, slot, set, out)
    }

    /// [`Sums::post`] of `out` then [`Sums::sum`] into it, the post and the
    /// wait in one batch.
    pub fn join(
        &mut self,
        gpu: &Gpu,
        slot: usize,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let set = self.place(gpu, slot, out, "Sums::join")?;
        let s = &mut self.slots[slot];
        // SAFETY: every counter the operations name lies in the page, held as in `place`.
        unsafe { mem_batch(gpu.stream(), &mut s.joined[set], "the rank's post and wait")? };
        s.next = (set + 1) % Gathers::SETS;
        self.add(gpu, slot, set, out)
    }

    /// Check `part` against slot `slot` and copy it into the rank's place of
    /// the slot's next set, which is returned.
    fn place(
        &mut self,
        gpu: &Gpu,
        slot: usize,
        part: &DeviceBuffer<f32>,
        what: &'static str,
    ) -> Result<usize, GpuError> {
        let dev = self.page.dev_at(0);
        let s = self
            .slots
            .get(slot)
            .ok_or_else(|| GpuError::shape(what, format!("slot {slot} of a page")))?;
        let w = s.shape.width;
        if part.is_empty() || !part.len().is_multiple_of(w) || part.len() / w > s.shape.rows {
            return Err(GpuError::shape(
                what,
                format!("slot {slot}: a part of {} values, rows of {w}", part.len()),
            ));
        }
        if let Some((set, _)) = s.posted {
            return Err(GpuError::shape(
                what,
                format!("slot {slot}: set {set} is posted and not summed"),
            ));
        }
        let set = s.next;
        let own = dev + self.layout.part(slot, set, self.rank).start as u64;
        // SAFETY: the part's values lie in the rank's own place of the page, mapped in this
        // context, which the sums hold until their owner has waited for every launch it enqueued.
        let mut place = unsafe { WindowMut::<f32>::from_raw(own, part.len(), gpu.context()) };
        place.copy_from_device_async(part, gpu.stream())?;
        Ok(set)
    }

    /// Enqueue the sum in rank order of slot `slot`'s set `set` into `out`.
    fn add(
        &self,
        gpu: &Gpu,
        slot: usize,
        set: usize,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let w = self.slots[slot].shape.width;
        let at = GatherSet {
            dev: self.page.dev_at(0),
            slot,
            set,
        };
        // SAFETY: `dev` is the page of this layout, mapped in this context and held as in `place`.
        unsafe {
            self.layout
                .enqueue_sum(gpu.elem(), gpu.stream(), at, out.len() / w, out)
        }
    }
}

/// How a rank's group bootstraps: the fabric manifest the peer paths read
/// their relay flow label from ([`super::paths`]), and how long rank 0 waits
/// for every other rank to connect and every other rank keeps dialing it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bootstrap {
    pub fabric: Option<std::path::PathBuf>,
    pub wait: Duration,
}

/// A group of ranks opened on their nodes (module doc): the control channel,
/// the gather page and the RDMA end to every peer.
///
/// Field order is drop order: the ends settle before the page drops.
pub struct RankSession {
    me: usize,
    n: usize,
    limits: Vec<(&'static str, u64)>,
    ctx: Arc<CudaContext>,
    control: Option<Control>,
    ends: Vec<RdmaEnd>,
    paths: Vec<PeerPath>,
    sums: Option<Sums>,
    layout: Option<Gathers>,
    page: Option<Arc<MappedHost>>,
}

impl RankSession {
    /// Open rank `me` of `list` on `gpu`'s context: at one rank, no link;
    /// above, the control channel to rank 0, the comparison of every rank's
    /// `fp` with rank 0's and of its `limits` names (the group runs at each
    /// limit's smallest value, [`RankSession::limits`]), a gather page of
    /// `shapes`, and the RDMA end to every peer on its path, as `how` says.
    /// Refused by name: a fingerprint that is not this rank's place in the
    /// list, a rank whose load differs ([`RankError::Mismatch`]), a peer with
    /// no path, and a rank that does not connect within `how.wait`.
    /// Load-time only.
    pub fn open(
        gpu: &Gpu,
        list: &RankList,
        me: usize,
        how: &Bootstrap,
        fp: &Fingerprint,
        limits: &[(&'static str, u64)],
        shapes: &[GatherShape],
    ) -> Result<RankSession, RankError> {
        if me >= list.len() || usize::from(fp.n) != list.len() || usize::from(fp.rank) != me {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "rank {me} of a list of {} with the fingerprint of rank {} of {}",
                    list.len(),
                    fp.rank,
                    fp.n
                ),
            )
            .into());
        }
        let n = list.len();
        let mut s = RankSession {
            me,
            n,
            limits: limits.to_vec(),
            ctx: Arc::clone(gpu.context()),
            control: None,
            ends: Vec::new(),
            paths: Vec::new(),
            sums: None,
            layout: None,
            page: None,
        };
        if n == 1 {
            return Ok(s);
        }
        let mut control = Control::open(list, me, how.wait)?;
        let hello = Hello {
            version: VERSION,
            fp: fp.clone(),
            limits: limits.iter().map(|&(k, v)| (k.to_string(), v)).collect(),
        };
        s.limits = control.agree(&hello, limits)?;
        let all = paths(list, me, how.fabric.as_deref(), &mut control)?;
        let layout = Gathers::new(0, n, me, shapes)?;
        let page = Arc::new(MappedHost::new(
            &s.ctx,
            layout.page_bytes(),
            "the rank group's gather page",
        )?);
        let mut ends = Vec::with_capacity(n - 1);
        for path in &all {
            let j = path.peer;
            ends.push(RdmaEnd::new(
                Arc::clone(&page),
                layout.tail(j),
                layout.crossings(j),
                layout.side(j),
                path,
                crate::host::PostCols::default(),
            )?);
        }
        let local: Vec<(usize, Endpoint)> = ends
            .iter()
            .zip(&all)
            .map(|(e, p)| (p.peer, e.endpoint(psn(me, p.peer))))
            .collect();
        let remote = control.endpoints(&local)?;
        for (((e, &(_, l)), r), path) in ends.iter_mut().zip(&local).zip(&remote).zip(&all) {
            e.connect(l.psn, r).map_err(|e| {
                GpuError::link(
                    WHAT,
                    format!(
                        "rank {} from {} on {} GID {} to {}: {e}",
                        path.peer, path.local, path.dev, path.gid, path.remote
                    ),
                )
            })?;
        }
        control.barrier()?;
        s.control = Some(control);
        s.ends = ends;
        s.paths = all;
        s.sums = Some(Sums::over(layout.clone(), Arc::clone(&page)));
        s.layout = Some(layout);
        s.page = Some(page);
        Ok(s)
    }

    /// Each limit's smallest value over the ranks, in this rank's order.
    #[must_use]
    pub fn limits(&self) -> &[(&'static str, u64)] {
        &self.limits
    }

    /// The gather page's layout; `None` at one rank, which gathers nothing.
    #[must_use]
    pub fn gathers(&self) -> Option<&Gathers> {
        self.layout.as_ref()
    }

    /// The sums on the session's page, once: the rank body posts and sums on
    /// them. `None` at one rank or once taken.
    pub fn sums(&mut self) -> Option<Sums> {
        self.sums.take()
    }

    /// The gather page; `None` at one rank.
    #[must_use]
    pub fn page(&self) -> Option<&Arc<MappedHost>> {
        self.page.as_ref()
    }

    /// The path of every end, in rank order ([`RankSession::ends`]).
    #[must_use]
    pub fn paths(&self) -> &[PeerPath] {
        &self.paths
    }

    /// The RDMA end to every peer, in rank order: what a stall report reads
    /// and a release releases ([`RdmaEnd::broken`], [`RdmaEnd::release`]).
    #[must_use]
    pub fn ends(&self) -> &[RdmaEnd] {
        &self.ends
    }

    /// Every rank sends rank 0 `words`, its figures to compare (a digest of
    /// its sums, its tokens), and rank 0 gets every rank's in rank order, its
    /// own first; every other rank gets none. Rank 0 answers each rank once it
    /// holds them all, so the call is also a barrier. A control channel that
    /// fails fails the group ([`RankSession::fail`]).
    pub fn collect(&mut self, words: &[u64]) -> Result<Vec<Vec<u64>>, RankError> {
        if self.n == 1 {
            return Ok(vec![words.to_vec()]);
        }
        let Some(control) = self.control.as_mut() else {
            return Err(GpuError::state(WHAT, "an open group, not one that failed").into());
        };
        let got = control.exchange(words);
        got.map_err(|f| self.fault(f))
    }

    /// Rank 0 sends `r` to every rank and every other rank receives it: rank 0
    /// passes `Some`, every other rank `None`, and each gets the round back.
    /// A round on the wrong side, and a control channel that fails, fail the
    /// group ([`RankSession::fail`]). At one rank the round comes back as it
    /// went.
    pub fn round(&mut self, r: Option<&Round>) -> Result<Round, RankError> {
        if self.n == 1 {
            return r
                .cloned()
                .ok_or_else(|| GpuError::shape(WHAT, "rank 0 decides the round").into());
        }
        let Some(control) = self.control.as_mut() else {
            return Err(GpuError::state(WHAT, "an open group, not one that failed").into());
        };
        let sent = match (self.me, r) {
            (0, Some(r)) => control.send_round(r).map(|()| r.clone()),
            (0, None) => Err(Fault::Link("rank 0 decides the round".into())),
            (_, Some(_)) => Err(Fault::Link("only rank 0 decides the round".into())),
            (_, None) => control.recv_round(),
        };
        sent.map_err(|f| self.fault(f))
    }

    /// [`RankSession::fail`] from a fault of the control channel: a peer's notice keeps the rank
    /// that failed first and its error.
    fn fault(&mut self, f: Fault) -> RankError {
        match f {
            Fault::Peer { rank, why } => self.fail_from(rank, &why),
            Fault::Link(why) => self.fail(&why),
        }
    }

    /// Fail the group from this rank, `why` saying why: close the control
    /// channel, so every other rank's next round fails too, release the
    /// ends and finish the context's queued work. Every end settled and the
    /// work finished: [`RankError::Settled`]. Otherwise [`RankError::Unsettled`] names why, and the session keeps
    /// its page and ends alive for good.
    pub fn fail(&mut self, why: &str) -> RankError {
        // A peer's notice already here names the failure that came first; this rank's own error
        // may only follow from it.
        match self.control.as_mut().and_then(Control::notice) {
            Some((rank, first)) => self.fail_from(rank, &first),
            None => self.fail_from(self.me, why),
        }
    }

    /// [`RankSession::fail`] for a failure rank `rank` had first, `why` its error: every peer
    /// this rank reaches is told before the channel closes (rank 0 tells every rank, every other
    /// rank tells rank 0, which tells the rest at its next exchange), so each rank's error names
    /// the rank and its own error.
    fn fail_from(&mut self, rank: usize, why: &str) -> RankError {
        if let Some(mut control) = self.control.take() {
            control.notify(rank, why);
        }
        self.sums = None;
        if self.ends.is_empty() {
            return RankError::Settled {
                rank,
                why: why.to_string(),
            };
        }
        // The released counters let the card's waits pass; the context's queued work then
        // finishes before the page can go.
        let released = RdmaEnd::release(&self.ends, RELEASE).and_then(|()| {
            crate::capsync::ctx_sync(&self.ctx, "RankSession::fail").map_err(GpuError::from)
        });
        let settled = settle(released, (rank, why));
        if matches!(settled, RankError::Unsettled { .. }) {
            std::mem::forget(std::mem::take(&mut self.ends));
            if let Some(page) = self.page.take() {
                std::mem::forget(page);
            }
        }
        settled
    }
}

/// The error a failure rank `rank` had first, `why` its error, leaves, the ends' release
/// `released`.
fn settle(released: Result<(), GpuError>, (rank, why): (usize, &str)) -> RankError {
    match released {
        Ok(()) => RankError::Settled {
            rank,
            why: why.to_string(),
        },
        Err(e) => RankError::Unsettled {
            end: format!("{e} (after rank {rank}: {why})"),
        },
    }
}

/// The first packet sequence number of `me`'s queue pair to `peer`, which
/// `peer` computes the same way for that pair: the low byte of each rank, 16
/// of the 24 bits a sequence number holds.
pub fn psn(me: usize, peer: usize) -> u32 {
    ((me as u32 & 0xff) << 8) | (peer as u32 & 0xff)
}

/// One hello: what a rank opens with.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Hello {
    version: u32,
    fp: Fingerprint,
    limits: Vec<(String, u64)>,
}

impl Hello {
    fn encode(&self) -> Vec<u8> {
        let mut o = Out::default();
        o.u32(self.version)
            .u32(self.fp.n.into())
            .u32(self.fp.rank.into())
            .bytes(&self.fp.source)
            .bytes(&self.fp.headers)
            .u32(self.fp.form)
            .str(&self.fp.build)
            .len(self.limits.len());
        for (k, v) in &self.limits {
            o.str(k).u64(*v);
        }
        o.done().to_vec()
    }

    fn decode(b: &[u8]) -> Result<Hello, String> {
        let mut r = In::new(b);
        let version = r.u32("version")?;
        let n = u16_at(&mut r, "rank count")?;
        let rank = u16_at(&mut r, "rank")?;
        let source = array_at(&mut r, "source digest")?;
        let headers = array_at(&mut r, "header digest")?;
        let form = r.u32("form")?;
        let build = r.str("build")?;
        let limits = (0..r.len("limits")?)
            .map(|_| Ok((r.str("limit name")?, r.u64("limit")?)))
            .collect::<Result<Vec<_>, String>>()?;
        r.end()?;
        Ok(Hello {
            version,
            fp: Fingerprint {
                source,
                headers,
                form,
                build,
                n,
                rank,
            },
            limits,
        })
    }

    /// The first item in which `peer`, rank `from`'s hello, differs from
    /// rank 0's `self`.
    fn differs(&self, peer: &Hello, from: usize) -> Option<&'static str> {
        let limit_names = |h: &Hello| h.limits.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>();
        let same = [
            self.version == peer.version,
            self.fp.n == peer.fp.n,
            usize::from(peer.fp.rank) == from,
            self.fp.source == peer.fp.source,
            self.fp.headers == peer.fp.headers,
            self.fp.form == peer.fp.form,
            self.fp.build == peer.fp.build,
            limit_names(self) == limit_names(peer),
        ];
        ITEMS.iter().zip(same).find(|(_, s)| !s).map(|(i, _)| *i)
    }
}

/// A `u16` a payload carries as a `u32`.
fn u16_at(r: &mut In, what: &str) -> Result<u16, String> {
    let v = r.u32(what)?;
    u16::try_from(v).map_err(|_| format!("its {what}, {v}, passes u16"))
}

/// `N` bytes a payload carries as a length-prefixed string.
fn array_at<const N: usize>(r: &mut In, what: &str) -> Result<[u8; N], String> {
    let b = r.bytes(what)?;
    b.try_into()
        .map_err(|_| format!("its {what} holds {} bytes, not {N}", b.len()))
}

fn write_frame(s: &mut TcpStream, b: &[u8]) -> Result<(), String> {
    let n = u32::try_from(b.len()).map_err(|_| "a frame past u32".to_string())?;
    s.write_all(&n.to_le_bytes())
        .and_then(|()| s.write_all(b))
        .map_err(|e| format!("a control write: {e}"))
}

/// A run-time frame's first byte: the exchange's data, or a failure notice.
const DATA: u8 = 0;
const FAILED: u8 = 1;

/// Why a run-time exchange of the control channel did not finish.
enum Fault {
    /// A peer's failure notice: the rank that failed first and its error.
    Peer { rank: usize, why: String },
    /// The channel itself, as the text says.
    Link(String),
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fault::Peer { rank, why } => write!(f, "rank {rank} failed: {why}"),
            Fault::Link(why) => write!(f, "{why}"),
        }
    }
}

/// Write `b` as a data frame of the run-time exchange. A write the peer refused may follow its
/// failure notice, which then comes back in place of the write's error.
fn send_data(s: &mut TcpStream, b: &[u8]) -> Result<(), Fault> {
    let mut f = Vec::with_capacity(1 + b.len());
    f.push(DATA);
    f.extend(b);
    write_frame(s, &f).map_err(|e| {
        let _ = s.set_read_timeout(Some(Duration::from_secs(1)));
        match recv_data(s) {
            Err(peer @ Fault::Peer { .. }) => peer,
            _ => Fault::Link(e),
        }
    })
}

/// Read a data frame of the run-time exchange; a failure notice comes back as the peer's fault.
fn recv_data(s: &mut TcpStream) -> Result<Vec<u8>, Fault> {
    let mut b = read_frame(s).map_err(Fault::Link)?;
    match b.first() {
        Some(&DATA) => {
            b.remove(0);
            Ok(b)
        }
        Some(&FAILED) => {
            let mut r = In::new(&b[1..]);
            let notice = r
                .u32("rank")
                .and_then(|rank| Ok((rank as usize, r.str("error")?)));
            Err(match notice {
                Ok((rank, why)) => Fault::Peer { rank, why },
                Err(e) => Fault::Link(format!("a failure notice: {e}")),
            })
        }
        _ => Err(Fault::Link("a frame of no kind".into())),
    }
}

fn read_frame(s: &mut TcpStream) -> Result<Vec<u8>, String> {
    let mut n = [0u8; 4];
    s.read_exact(&mut n)
        .map_err(|e| format!("a control read: {e}"))?;
    let n = u32::from_le_bytes(n) as usize;
    if n > MAX_FRAME {
        return Err(format!("a frame of {n} bytes"));
    }
    let mut b = vec![0u8; n];
    s.read_exact(&mut b)
        .map_err(|e| format!("a control read: {e}"))?;
    Ok(b)
}

/// A group's control channel: rank 0 holds a socket to every other rank, in
/// rank order; every other rank holds one to rank 0. The session's, and a
/// probe's that opens its own ends.
pub struct Control {
    me: usize,
    socks: Vec<TcpStream>,
    root: bool,
    /// The bound of a bootstrap read and of a collect's.
    wait: Duration,
}

impl Control {
    /// Rank 0 listens on its control address and takes one connection from
    /// every other rank, each naming its rank; every other rank dials rank 0
    /// until `deadline` passes.
    pub fn open(list: &RankList, me: usize, deadline: Duration) -> Result<Control, GpuError> {
        let at = list.rank(0).control;
        let link = |e: String| GpuError::link(WHAT, format!("bootstrap: {e}"));
        let start = Instant::now();
        if me == 0 {
            let l = TcpListener::bind(at).map_err(|e| link(format!("listen on {at}: {e}")))?;
            l.set_nonblocking(true).map_err(|e| link(e.to_string()))?;
            let mut socks: Vec<Option<TcpStream>> = (1..list.len()).map(|_| None).collect();
            while socks.iter().any(Option::is_none) {
                match l.accept() {
                    Ok((mut s, _)) => {
                        s.set_nonblocking(false).map_err(|e| link(e.to_string()))?;
                        s.set_read_timeout(Some(deadline))
                            .map_err(|e| link(e.to_string()))?;
                        let mut r = [0u8; 4];
                        s.read_exact(&mut r).map_err(|e| link(e.to_string()))?;
                        let r = u32::from_le_bytes(r) as usize;
                        match socks.get_mut(r.wrapping_sub(1)) {
                            Some(slot @ None) => *slot = Some(s),
                            _ => return Err(link(format!("a rank named itself {r}"))),
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if start.elapsed() > deadline {
                            let missing: Vec<usize> =
                                (1..list.len()).filter(|r| socks[r - 1].is_none()).collect();
                            return Err(link(format!(
                                "ranks {missing:?} did not connect within {deadline:?}"
                            )));
                        }
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(e) => return Err(link(e.to_string())),
                }
            }
            Ok(Control {
                me,
                socks: socks.into_iter().flatten().collect(),
                root: true,
                wait: deadline,
            })
        } else {
            let mut s = loop {
                match TcpStream::connect_timeout(&at, Duration::from_secs(5)) {
                    Ok(s) => break s,
                    Err(e) if start.elapsed() > deadline => {
                        return Err(link(format!("cannot reach rank 0 at {at}: {e}")));
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(100)),
                }
            };
            s.set_read_timeout(Some(deadline))
                .map_err(|e| link(e.to_string()))?;
            s.set_nodelay(true).map_err(|e| link(e.to_string()))?;
            s.write_all(&(me as u32).to_le_bytes())
                .map_err(|e| link(e.to_string()))?;
            Ok(Control {
                me,
                socks: vec![s],
                root: false,
                wait: deadline,
            })
        }
    }

    /// Compare every rank's `hello` with rank 0's and take each limit's
    /// smallest value: rank 0 reads every hello, then sends every rank the
    /// verdict, the first item a rank differs in or the agreed limits.
    fn agree(
        &mut self,
        hello: &Hello,
        limits: &[(&'static str, u64)],
    ) -> Result<Vec<(&'static str, u64)>, RankError> {
        let link = |e: String| RankError::Gpu(GpuError::link(WHAT, format!("bootstrap: {e}")));
        let verdict: Result<Vec<u64>, &'static str> = if self.root {
            let mut agreed: Vec<u64> = limits.iter().map(|&(_, v)| v).collect();
            let mut differs = None;
            for (i, s) in self.socks.iter_mut().enumerate() {
                let peer = Hello::decode(&read_frame(s).map_err(link)?).map_err(link)?;
                if differs.is_none() {
                    differs = hello.differs(&peer, i + 1);
                }
                for (a, (_, v)) in agreed.iter_mut().zip(&peer.limits) {
                    *a = (*a).min(*v);
                }
            }
            let verdict = differs.map_or(Ok(agreed), Err);
            let b = encode_verdict(&verdict);
            for s in &mut self.socks {
                write_frame(s, &b).map_err(link)?;
            }
            verdict
        } else {
            write_frame(&mut self.socks[0], &hello.encode()).map_err(link)?;
            decode_verdict(&read_frame(&mut self.socks[0]).map_err(link)?).map_err(link)?
        };
        match verdict {
            Ok(v) => Ok(limits.iter().zip(v).map(|(&(k, _), v)| (k, v)).collect()),
            Err(item) => Err(RankError::Mismatch { item }),
        }
    }

    /// Send each peer named in `out` its bytes and receive, in the same
    /// order, the bytes that peer sent this rank; rank 0 carries every other
    /// pair's. Every rank names every peer once.
    pub fn swap(&mut self, out: &[(usize, Vec<u8>)]) -> Result<Vec<Vec<u8>>, GpuError> {
        let link = |e: String| GpuError::link(WHAT, format!("bootstrap: {e}"));
        let me = self.me;
        let own: Vec<(usize, usize, Vec<u8>)> =
            out.iter().map(|(to, b)| (me, *to, b.clone())).collect();
        let mine = if self.root {
            let mut all = own;
            for s in &mut self.socks {
                all.extend(decode_swap(&read_frame(s).map_err(link)?).map_err(link)?);
            }
            for (i, s) in self.socks.iter_mut().enumerate() {
                let to: Vec<_> = all.iter().filter(|m| m.1 == i + 1).cloned().collect();
                write_frame(s, &encode_swap(&to)).map_err(link)?;
            }
            all.into_iter().filter(|m| m.1 == 0).collect::<Vec<_>>()
        } else {
            write_frame(&mut self.socks[0], &encode_swap(&own)).map_err(link)?;
            decode_swap(&read_frame(&mut self.socks[0]).map_err(link)?).map_err(link)?
        };
        out.iter()
            .map(|&(peer, _)| {
                let mut of = mine.iter().filter(|m| m.0 == peer && m.1 == me);
                match (of.next(), of.next()) {
                    (Some(m), None) => Ok(m.2.clone()),
                    _ => Err(link(format!("not one message of rank {peer} to rank {me}"))),
                }
            })
            .collect()
    }

    /// Trade queue pair endpoints: `local` holds this rank's end to each
    /// peer, and the result each peer's end to this rank, in the same order.
    pub fn endpoints(&mut self, local: &[(usize, Endpoint)]) -> Result<Vec<Endpoint>, GpuError> {
        let out: Vec<(usize, Vec<u8>)> = local
            .iter()
            .map(|(to, e)| {
                let mut o = Out::default();
                write_endpoint(e, &mut o);
                (*to, o.done().to_vec())
            })
            .collect();
        self.swap(&out)?
            .iter()
            .map(|b| {
                let mut r = In::new(b);
                read_endpoint(&mut r)
                    .and_then(|e| r.end().map(|()| e))
                    .map_err(|e| GpuError::link(WHAT, format!("bootstrap: {e}")))
            })
            .collect()
    }

    /// A barrier, a byte each way, so every queue pair of the group is
    /// connected before any rank posts; then the reads wait without bound,
    /// a round coming whenever rank 0 has one.
    pub fn barrier(&mut self) -> Result<(), GpuError> {
        let link = |e: std::io::Error| GpuError::link(WHAT, format!("bootstrap: {e}"));
        let read = |s: &mut TcpStream| s.read_exact(&mut [0u8; 1]);
        if self.root {
            self.socks.iter_mut().try_for_each(read).map_err(link)?;
            self.socks
                .iter_mut()
                .try_for_each(|s| s.write_all(b"R"))
                .map_err(link)?;
        } else {
            self.socks
                .iter_mut()
                .try_for_each(|s| s.write_all(b"R"))
                .map_err(link)?;
            self.socks.iter_mut().try_for_each(read).map_err(link)?;
        }
        for s in &mut self.socks {
            s.set_read_timeout(None).map_err(link)?;
            s.set_nodelay(true).map_err(link)?;
        }
        Ok(())
    }

    /// Every rank sends rank 0 `words`, as many on every rank; rank 0 gets
    /// every rank's in rank order, its own first, and every other rank none.
    /// Rank 0 answers each rank once it holds them all, so the call is also a
    /// barrier. Each read waits at most the channel's open bound.
    pub fn collect(&mut self, words: &[u64]) -> Result<Vec<Vec<u64>>, GpuError> {
        self.exchange(words)
            .map_err(|f| GpuError::link(WHAT, format!("collect: {f}")))
    }

    /// [`Control::collect`], a peer's failure notice kept apart.
    fn exchange(&mut self, words: &[u64]) -> Result<Vec<Vec<u64>>, Fault> {
        self.bound(Some(self.wait)).map_err(Fault::Link)?;
        let got = self.gather_words(words);
        self.bound(None).map_err(Fault::Link)?;
        got
    }

    /// Tell every peer this channel reaches that rank `rank` failed, `why` its error; a peer
    /// already gone is passed over, since the group fails either way.
    fn notify(&mut self, rank: usize, why: &str) {
        let mut o = Out::default();
        o.u8(FAILED).u32(rank as u32).str(why);
        for s in &mut self.socks {
            let _ = write_frame(s, o.done());
        }
    }

    /// The first failure notice a peer already sent, the frames before it read and passed
    /// over; `None` when none has arrived. Reads only what is here, without waiting.
    fn notice(&mut self) -> Option<(usize, String)> {
        let mut found = None;
        for s in &mut self.socks {
            if s.set_nonblocking(true).is_err() {
                continue;
            }
            while found.is_none() {
                match recv_data(s) {
                    Ok(_) => {}
                    Err(Fault::Peer { rank, why }) => found = Some((rank, why)),
                    Err(Fault::Link(_)) => break,
                }
            }
            let _ = s.set_nonblocking(false);
        }
        found
    }

    /// Set every socket's read bound.
    fn bound(&mut self, wait: Option<Duration>) -> Result<(), String> {
        self.socks
            .iter_mut()
            .try_for_each(|s| s.set_read_timeout(wait))
            .map_err(|e| e.to_string())
    }

    /// [`Control::collect`]'s exchange.
    fn gather_words(&mut self, words: &[u64]) -> Result<Vec<Vec<u64>>, Fault> {
        let encode = |w: &[u64]| w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        if !self.root {
            send_data(&mut self.socks[0], &encode(words))?;
            recv_data(&mut self.socks[0])?;
            return Ok(Vec::new());
        }
        let mut all = vec![words.to_vec()];
        for (i, s) in self.socks.iter_mut().enumerate() {
            let b = recv_data(s)?;
            if b.len() != 8 * words.len() {
                return Err(Fault::Link(format!(
                    "rank {} sent {} bytes, not {} words",
                    i + 1,
                    b.len(),
                    words.len()
                )));
            }
            all.push(
                b.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|&c| u64::from_le_bytes(c))
                    .collect(),
            );
        }
        for s in &mut self.socks {
            send_data(s, &[])?;
        }
        Ok(all)
    }

    fn send_round(&mut self, r: &Round) -> Result<(), Fault> {
        let b = encode_round(r);
        for (i, s) in self.socks.iter_mut().enumerate() {
            send_data(s, &b).map_err(|f| match f {
                Fault::Link(e) => Fault::Link(format!("round to rank {}: {e}", i + 1)),
                peer @ Fault::Peer { .. } => peer,
            })?;
        }
        Ok(())
    }

    fn recv_round(&mut self) -> Result<Round, Fault> {
        let at = |e: String| Fault::Link(format!("round from rank 0: {e}"));
        let b = recv_data(&mut self.socks[0]).map_err(|f| match f {
            Fault::Link(e) => at(e),
            peer @ Fault::Peer { .. } => peer,
        })?;
        decode_round(&b).map_err(at)
    }
}

fn encode_verdict(v: &Result<Vec<u64>, &'static str>) -> Vec<u8> {
    let mut o = Out::default();
    match v {
        Ok(limits) => o.u8(0).u64s(limits),
        Err(item) => o.u8(1).str(item),
    };
    o.done().to_vec()
}

fn decode_verdict(b: &[u8]) -> Result<Result<Vec<u64>, &'static str>, String> {
    let mut r = In::new(b);
    let v = if r.flag("verdict")? {
        let item = r.str("item")?;
        Err(ITEMS
            .iter()
            .find(|&&i| i == item)
            .copied()
            .ok_or_else(|| format!("an item {item:?}"))?)
    } else {
        Ok(r.u64s("limits")?)
    };
    r.end()?;
    Ok(v)
}

fn encode_swap(m: &[(usize, usize, Vec<u8>)]) -> Vec<u8> {
    let mut o = Out::default();
    o.len(m.len());
    for (from, to, bytes) in m {
        o.u32(*from as u32).u32(*to as u32).bytes(bytes);
    }
    o.done().to_vec()
}

fn decode_swap(b: &[u8]) -> Result<Vec<(usize, usize, Vec<u8>)>, String> {
    let mut r = In::new(b);
    let v = (0..r.len("messages")?)
        .map(|_| {
            let (from, to) = (r.u32("from")? as usize, r.u32("to")? as usize);
            Ok((from, to, r.bytes("message")?.to_vec()))
        })
        .collect::<Result<Vec<_>, String>>()?;
    r.end()?;
    Ok(v)
}

fn encode_round(r: &Round) -> Vec<u8> {
    let mut o = Out::default();
    o.u32(r.slot).u32(r.pos).u8(u8::from(r.stop));
    match &r.prompt {
        Some(p) => o.u8(1).u32(p.start).u32(p.end),
        None => o.u8(0),
    };
    o.u32s(&r.tokens).done().to_vec()
}

fn decode_round(b: &[u8]) -> Result<Round, String> {
    let mut r = In::new(b);
    let (slot, pos, stop) = (r.u32("slot")?, r.u32("pos")?, r.flag("stop")?);
    let prompt = if r.flag("prompt")? {
        Some(r.u32("prompt start")?..r.u32("prompt end")?)
    } else {
        None
    };
    let tokens = r.u32s("tokens")?;
    r.end()?;
    Ok(Round {
        slot,
        pos,
        tokens,
        prompt,
        stop,
    })
}

/// Each rank uses its opener's stream, distinct from every earlier rank's:
/// rank `rank` opened on `opened`, offered `offered`, the earlier ranks on
/// `prior`. Refused by name otherwise.
pub fn streams_at<'a, T: 'a>(
    rank: usize,
    opened: &Arc<T>,
    offered: &T,
    prior: impl IntoIterator<Item = &'a Arc<T>>,
) -> Result<(), GpuError> {
    const WHAT: &str = "a rank's stream";
    if !std::ptr::eq(offered, opened.as_ref()) {
        return Err(GpuError::shape(
            WHAT,
            format!("rank {rank}: use the GPU that opened it"),
        ));
    }
    if prior.into_iter().any(|p| Arc::ptr_eq(p, opened)) {
        return Err(GpuError::shape(
            WHAT,
            format!("rank {rank}: each rank must use a distinct stream"),
        ));
    }
    Ok(())
}

/// Rank 0's shape of ranks whose shapes `shapes` one step joins, each the
/// figures a step reads alike on every rank: refused by name for no rank, or
/// a rank whose shape is unlike rank 0's.
pub fn alike<T: PartialEq + std::fmt::Debug>(
    shapes: impl IntoIterator<Item = T>,
) -> Result<T, GpuError> {
    const WHAT: &str = "a step's ranks";
    let mut shapes = shapes.into_iter();
    let first = shapes
        .next()
        .ok_or_else(|| GpuError::shape(WHAT, "a step over no ranks"))?;
    for (r, t) in shapes.enumerate() {
        if t != first {
            return Err(GpuError::shape(
                WHAT,
                format!("rank {}'s {t:?} unlike rank 0's {first:?}", r + 1),
            ));
        }
    }
    Ok(first)
}

/// A step at `pos` inside stores of `ctx` positions, refused by name past them.
pub fn check_pos(pos: u32, ctx: usize) -> Result<(), GpuError> {
    if pos as usize >= ctx {
        return Err(GpuError::shape(
            "a step",
            format!("a step at position {pos} in stores of {ctx}"),
        ));
    }
    Ok(())
}

/// Refused by name unless stores of `ctx` positions that stand at `next` (a
/// rank's next position, `None` after a step failed once its launches went
/// out) take a step at `pos`: inside them, and the position they stand at.
pub fn at_pos(next: Option<u32>, ctx: usize, pos: u32) -> Result<(), GpuError> {
    const WHAT: &str = "a rank's step";
    check_pos(pos, ctx)?;
    match next {
        Some(p) if p == pos => Ok(()),
        Some(p) => Err(GpuError::shape(
            WHAT,
            format!("position {pos}, where the stores hold {p} positions"),
        )),
        None => Err(GpuError::shape(
            WHAT,
            format!(
                "position {pos}: a step failed once its launches went out, so the stores hold \
                 part of it; open the rank again"
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Control, Fingerprint, GpuError, Hello, RankError, RankList, Round, alike, at_pos, settle,
        streams_at,
    };
    use std::sync::Arc;
    use std::time::Duration;

    /// A list of `n` ranks whose control addresses are loopback ports from `port`.
    fn local(n: usize, port: u16) -> RankList {
        let lines: Vec<String> = (0..n)
            .map(|r| format!("127.0.0.{}:{} 198.18.0.{}", r + 1, port, r + 1))
            .collect();
        let mut text = lines.join("\n");
        // Every rank dials rank 0's address; the others' ports are never bound.
        text.push('\n');
        RankList::parse(&text).unwrap()
    }

    fn hello(rank: usize, n: usize, build: &str) -> Hello {
        Hello {
            version: super::VERSION,
            fp: Fingerprint {
                source: [1; 32],
                headers: [2; 32],
                form: 3,
                build: build.into(),
                n: n as u16,
                rank: rank as u16,
            },
            limits: vec![("positions".into(), 100 + rank as u64)],
        }
    }

    /// A rank's agreed limits and the rounds it received, or why it failed.
    type Outcome = Result<(Vec<u64>, Vec<Round>), String>;

    /// Every rank of a group of `n` opens its control channel and agrees on `hello(rank)`;
    /// rank 0 then sends `rounds`, which every other rank receives. Each rank's outcome.
    fn group(
        n: usize,
        port: u16,
        build: impl Fn(usize) -> &'static str + Sync,
        rounds: &[Round],
    ) -> Vec<Outcome> {
        let list = local(n, port);
        std::thread::scope(|s| {
            let jobs: Vec<_> = (0..n)
                .map(|r| {
                    let (list, build) = (&list, &build);
                    s.spawn(move || {
                        let mut c = Control::open(list, r, Duration::from_secs(10))
                            .map_err(|e| e.to_string())?;
                        let h = hello(r, n, build(r));
                        let limits = [("positions", 100 + r as u64)];
                        let agreed = c.agree(&h, &limits).map_err(|e| e.to_string())?;
                        c.barrier().map_err(|e| e.to_string())?;
                        let mut got = Vec::new();
                        for round in rounds {
                            if r == 0 {
                                c.send_round(round).map_err(|e| e.to_string())?;
                                got.push(round.clone());
                            } else {
                                got.push(c.recv_round().map_err(|e| e.to_string())?);
                            }
                        }
                        Ok((agreed.iter().map(|&(_, v)| v).collect(), got))
                    })
                })
                .collect();
            jobs.into_iter().map(|j| j.join().unwrap()).collect()
        })
    }

    fn port(k: u16) -> u16 {
        40_000 + (std::process::id() % 2000) as u16 * 5 + k
    }

    // At N = 2 and 4 every rank agrees on the smallest limit and receives rank 0's rounds as
    // they went; a rank whose build differs is refused on every rank by the item's name.
    #[test]
    fn a_group_agrees_on_its_load_and_carries_rank_zeros_rounds() {
        let rounds = [
            Round {
                slot: 0,
                pos: 0,
                tokens: vec![5, 6, 7],
                prompt: Some(0..3),
                stop: false,
            },
            Round {
                slot: 2,
                pos: 3,
                tokens: vec![9],
                prompt: None,
                stop: true,
            },
        ];
        for (k, n) in [(0u16, 2usize), (1, 4)] {
            for r in group(n, port(k), |_| "b1", &rounds) {
                let (agreed, got) = r.expect("an open group");
                assert_eq!(agreed, [100], "{n} ranks");
                assert_eq!(got, rounds, "{n} ranks");
            }
        }
        let out = group(4, port(2), |r| if r == 3 { "b2" } else { "b1" }, &[]);
        for r in out {
            let e = r.unwrap_err();
            assert!(e.contains("in its build"), "{e}");
        }
    }

    // A rank whose rank 0 is gone fails its next round by name, so one rank's failure is the
    // group's; the failure's ends settled or not decide what follows.
    #[test]
    fn a_lost_rank_fails_the_group_and_its_ends_decide_what_follows() {
        let list = local(2, port(3));
        let e = std::thread::scope(|s| {
            let root = s.spawn(|| {
                let mut c = Control::open(&list, 0, Duration::from_secs(10)).unwrap();
                c.agree(&hello(0, 2, "b"), &[("positions", 100)]).unwrap();
                c.barrier().unwrap();
            });
            let mut c = Control::open(&list, 1, Duration::from_secs(10)).unwrap();
            c.agree(&hello(1, 2, "b"), &[("positions", 101)]).unwrap();
            c.barrier().unwrap();
            root.join().unwrap();
            c.recv_round().unwrap_err()
        });
        assert!(e.to_string().contains("round from rank 0"), "{e}");
        // A failing rank's notice reaches rank 0's next exchange with its rank and its error.
        let list = local(2, port(4));
        let got = std::thread::scope(|s| {
            let root = s.spawn(|| {
                let mut c = Control::open(&list, 0, Duration::from_secs(10)).unwrap();
                c.agree(&hello(0, 2, "b"), &[("positions", 100)]).unwrap();
                c.barrier().unwrap();
                c.exchange(&[7]).map(|_| ()).unwrap_err()
            });
            let mut c = Control::open(&list, 1, Duration::from_secs(10)).unwrap();
            c.agree(&hello(1, 2, "b"), &[("positions", 101)]).unwrap();
            c.barrier().unwrap();
            c.notify(1, "wo_a refused");
            root.join().unwrap()
        });
        assert!(
            matches!(&got, super::Fault::Peer { rank: 1, why } if why == "wo_a refused"),
            "{got}"
        );
        // A notice already at rank 0 when its own step fails is the failure it reports.
        let list = local(2, port(3));
        let first = std::sync::Barrier::new(2);
        let got = std::thread::scope(|s| {
            let root = s.spawn(|| {
                let mut c = Control::open(&list, 0, Duration::from_secs(10)).unwrap();
                c.agree(&hello(0, 2, "b"), &[("positions", 100)]).unwrap();
                c.barrier().unwrap();
                assert_eq!(c.notice(), None);
                first.wait();
                first.wait();
                c.notice()
            });
            let mut c = Control::open(&list, 1, Duration::from_secs(10)).unwrap();
            c.agree(&hello(1, 2, "b"), &[("positions", 101)]).unwrap();
            c.barrier().unwrap();
            first.wait();
            c.notify(1, "wo_a refused");
            first.wait();
            root.join().unwrap()
        });
        assert_eq!(got, Some((1, "wo_a refused".to_string())));
        assert!(matches!(
            settle(Ok(()), (1, "x")),
            RankError::Settled { rank: 1, .. }
        ));
        let held = settle(Err(GpuError::shape("end", "rank 2's end")), (1, "x"));
        assert!(matches!(&held, RankError::Unsettled { end } if end.contains("rank 2's end")));
    }

    #[test]
    fn a_hello_reads_back_as_written() {
        let h = hello(1, 4, "build");
        assert_eq!(Hello::decode(&h.encode()).unwrap(), h);
        assert!(Hello::decode(&h.encode()[..10]).is_err());
    }

    #[test]
    fn ranks_use_their_openers_distinct_streams() {
        let a = Arc::new(0u8);
        let b = Arc::new(0u8);
        assert!(streams_at(0, &a, &a, []).is_ok());
        assert!(streams_at(1, &b, &b, [&a]).is_ok());
        for (opened, offered, prior, want) in [
            (&a, &b, &b, "use the GPU that opened it"),
            (&a, &a, &a, "each rank must use a distinct stream"),
        ] {
            let e = streams_at(1, opened, offered, [prior]).unwrap_err();
            assert!(e.to_string().contains(want), "{e}");
        }
    }

    /// Stores take the position they stand at, inside them; any other position, the one past
    /// them and every position after a failed step are refused.
    #[test]
    fn a_step_takes_only_the_position_its_stores_stand_at() {
        assert!(at_pos(Some(0), 4, 0).is_ok() && at_pos(Some(3), 4, 3).is_ok());
        for (next, pos) in [
            (Some(0), 1),
            (Some(2), 1),
            (Some(4), 4),
            (None, 0),
            (None, 2),
        ] {
            assert!(at_pos(next, 4, pos).is_err(), "{next:?} at {pos}");
        }
    }

    /// A step joins ranks of one shape, at N = 1, 2 and 4; no rank, and a rank of another
    /// shape, are refused by name.
    #[test]
    fn a_step_joins_only_ranks_of_one_shape() {
        let t = (3, 64, 8);
        for n in [1, 2, 4] {
            assert_eq!(alike(vec![t; n]).unwrap(), t, "{n} ranks");
        }
        assert!(alike(Vec::<(i32, i32, i32)>::new()).is_err());
        for other in [(2, 64, 8), (3, 32, 8), (3, 64, 16)] {
            for n in [2, 4] {
                let mut shapes = vec![t; n];
                shapes[n - 1] = other;
                let e = alike(shapes).unwrap_err().to_string();
                assert!(e.contains(&format!("rank {}'s", n - 1)), "{e}");
            }
        }
    }

    #[test]
    fn a_rank_file_reads_in_rank_order_and_refuses_by_line() {
        let list = RankList::parse(
            "# rank 0 first\n192.168.1.10:7000 198.18.0.1,198.18.1.1\n\n\
             192.168.1.11:7000 198.18.0.2 # rank 1\n",
        )
        .expect("a rank list");
        assert_eq!(list.len(), 2);
        assert_eq!(list.rank(0).data.len(), 2);
        assert_eq!(list.rank(1).control.to_string(), "192.168.1.11:7000");
        for (text, says) in [
            ("", "no rank"),
            ("192.168.1.10:7000\n", "line 1: want"),
            ("192.168.1.10 198.18.0.1\n", "is not an ip:port"),
            ("192.168.1.10:7000 fe80::1\n", "is not an IPv4"),
            (
                "10.0.0.1:1 198.18.0.1\n10.0.0.1:1 198.18.0.2\n",
                "line 2: control",
            ),
            (
                "10.0.0.1:1 198.18.0.1\n10.0.0.2:1 198.18.0.1\n",
                "line 2: data",
            ),
            ("10.0.0.1:1 198.18.0.1,198.18.0.1\n", "line 1: data"),
        ] {
            let e = RankList::parse(text).unwrap_err().to_string();
            assert!(e.contains(says), "{text:?}: {e}");
        }
    }
}
