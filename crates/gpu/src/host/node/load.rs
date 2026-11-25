//! A tier node with no model file of its own: node 0 streams its tier's rows
//! to it at open (`docs/contrib/dgx-spark.md` 9.5, item 5).
//!
//! The rule: a `tier_server` started with no model file (no `-m`, no
//! `$BLOOMERY_REF_MODEL`) says so in its census ([`super::NodeCensus::file`]),
//! and node 0 opens its tier as a fed open ([`super::TierNode::open_fed`]).
//! The open carries, after the tier's rows, a stream head ([`StreamHead`]):
//! node 0's shards' headers and file lengths, their digest, each row's scales
//! and the wire the bytes take. The node builds node 0's split from the
//! headers alone ([`gguf::Split::from_headers`]), checks the open against it
//! as it checks one against its own file ([`super::NodeOpen::rows_on`]),
//! opens the card, and answers with its staging ([`Ready`]). Node 0 then
//! sends each row's bytes, in the open's order: by RDMA writes into the
//! node's registered slots (`BLOOMERY_TIER_LINK=rdma`), or as load requests
//! that carry them over the control channel (any other link). Each row ends
//! with a request that names its slot, its length and the SHA-256 of each
//! [`CHUNK`] of its bytes, hashed by node 0 as it read them from its file;
//! the node hashes what landed, uploads it as a load from its own file
//! would ([`crate::weights::SegmentLoad`]), and answers once both are done. A chunk whose digest differs is the stream's corruption: the node
//! refuses it by name and ends, and node 0's open fails with that refusal, so
//! no byte that did not arrive as node 0 read it is served. The last request
//! ends the load: the family's computation is built over the uploaded rows
//! and the open's reply is the reply a node with a file gives.
//!
//! Node 0 reads each row from its file's mapping, the pages of the rows after
//! it read ahead ([`model::placement::read_ahead`]) and, in a load that drops
//! its card segments' pages (`BLOOMERY_CARD_DONTNEED`), released once copied
//! ([`PageDrop`]); one thread fills the next row while the row before it is
//! on the wire, copying and hashing it on threads that float off the load
//! thread's cpu (`threads::helper::float_caller`). Over RDMA the node holds two slots, so node 0 writes a row
//! while the node hashes and uploads the one before it; over the control
//! channel each request waits for its reply.
//!
//! A paged tier (rows on the node's host or drive) reads the node's own
//! file's runs and is refused by name on a node with no file.

use std::collections::VecDeque;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use bloomery_verbs::{Endpoint, MAX_CHAIN, Nic, Qp, Region, WC_RDMA_WRITE, Wc, Write};
use gguf::Split;
use gguf::digest::{Sha256, hex};
use model::placement::host_lock::PageDrop;
use model::placement::read_ahead;
use model::placement::{ModelTensor, ModelTensors, Row, Segment};

use super::wire::{In, Out, Tag};
use super::{
    NodeFamily, NodeOpen, OPEN_BASE, OPEN_RATE, REPLY, SERVER, TierNode, WHAT, read_endpoint,
    read_err, write_endpoint,
};
use crate::host::tier::TierWork;
use crate::weights::{SegmentFile, SegmentLoad, segment_ranges, segment_scales};
use crate::{Gpu, GpuError};

/// The bytes each digest of a streamed row covers: a row's run is cut into
/// chunks of this many bytes, each hashed on its own, so both sides hash a
/// row on several threads.
pub(super) const CHUNK: usize = 16 << 20;

/// The longest RDMA write node 0 posts.
const WRITE: usize = 64 << 20;

/// The most RDMA writes node 0 holds in flight, the depth of its queue pair.
const DEPTH: usize = 32;

/// The longest run of row bytes one load request carries over the control
/// channel, well under [`super::wire::MAX_FRAME`].
const FRAME_BYTES: usize = 32 << 20;

/// The packet sequence numbers each end's sends start at.
const PSN_NODE0: u32 = 0x300;
const PSN_NODE: u32 = 0x400;

/// A load request's kinds.
const LOAD_BYTES: u8 = 0;
const LOAD_ROW: u8 = 1;
const LOAD_END: u8 = 2;

/// Threads that copy and hash a row's chunks on node 0, and hash them on the
/// node: the calling thread's cpus, so node 0's filler counts them once it
/// floats.
fn workers() -> usize {
    std::thread::available_parallelism()
        .map_or(4, std::num::NonZeroUsize::get)
        .clamp(2, 8)
}

/// How node 0's rows reach the node: RDMA writes into its registered slots,
/// or load requests over the control channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadWire {
    Rdma,
    Tcp,
}

impl LoadWire {
    fn byte(self) -> u8 {
        match self {
            LoadWire::Rdma => 1,
            LoadWire::Tcp => 2,
        }
    }

    fn of(b: u8) -> Result<LoadWire, String> {
        match b {
            1 => Ok(LoadWire::Rdma),
            2 => Ok(LoadWire::Tcp),
            w => Err(format!("a load wire {w}")),
        }
    }
}

/// What node 0 feeds a node with no file from: its split, the plan's
/// tensors, whether its card segments' pages go once read, and the wire.
pub struct Feed<'a> {
    pub file: &'a Split,
    pub model: &'a ModelTensors,
    pub release: bool,
    pub wire: LoadWire,
}

/// What node 0's sending thread waited on over a feed: the filler (its reads
/// of the file, copies and digests, on `workers` threads), the wire (RDMA
/// writes, or load frames and their replies over the control channel), and
/// the node's row replies.
#[derive(Default)]
struct Waited {
    fill: Duration,
    wire: Duration,
    reply: Duration,
    workers: usize,
}

/// What follows a fed open's rows: the wire, node 0's RDMA endpoint on it,
/// each shard's name, file length and header bytes, their digest
/// ([`model::fixture::header_sha256`]), and each row's scales
/// ([`segment_scales`]).
pub(super) struct StreamHead {
    wire: LoadWire,
    endpoint: Option<Endpoint>,
    shards: Vec<(String, u64, Vec<u8>)>,
    header_sha256: String,
    scales: Vec<Option<Vec<f32>>>,
}

impl StreamHead {
    fn write(&self, o: &mut Out) {
        o.u8(1).u8(self.wire.byte());
        if let Some(e) = &self.endpoint {
            write_endpoint(e, o);
        }
        o.len(CHUNK).len(self.shards.len());
        for (name, len, header) in &self.shards {
            o.str(name).u64(*len).bytes(header);
        }
        o.str(&self.header_sha256).len(self.scales.len());
        for s in &self.scales {
            match s {
                Some(v) => o
                    .u8(1)
                    .u32s(&v.iter().map(|x| x.to_bits()).collect::<Vec<_>>()),
                None => o.u8(0),
            };
        }
    }

    /// The head as [`StreamHead::write`] wrote it; a chunk of another size
    /// than this build's is refused by name.
    pub(super) fn read(i: &mut In<'_>) -> Result<StreamHead, String> {
        if i.u8("open source")? != 1 {
            return Err("an open source other than a stream".into());
        }
        let wire = LoadWire::of(i.u8("load wire")?)?;
        let endpoint = match wire {
            LoadWire::Rdma => Some(read_endpoint(i)?),
            LoadWire::Tcp => None,
        };
        let chunk = i.len("chunk bytes")?;
        if chunk != CHUNK {
            return Err(format!(
                "digests of {chunk}-byte chunks; this node checks {CHUNK}-byte chunks"
            ));
        }
        let n = i.len("shard count")?;
        let mut shards = Vec::new();
        for _ in 0..n {
            let name = i.str("shard name")?;
            let len = i.u64("shard length")?;
            shards.push((name, len, i.bytes("shard header")?.to_vec()));
        }
        let header_sha256 = i.str("header digest")?;
        let n = i.len("scale rows")?;
        let mut scales = Vec::new();
        for _ in 0..n {
            scales.push(if i.flag("scales")? {
                Some(
                    i.u32s("scale bits")?
                        .into_iter()
                        .map(f32::from_bits)
                        .collect(),
                )
            } else {
                None
            });
        }
        Ok(StreamHead {
            wire,
            endpoint,
            shards,
            header_sha256,
            scales,
        })
    }
}

/// The node's answer to a fed open: its wire, its RDMA endpoint (its slots'
/// region) on it, and its slots' size and count.
struct Ready {
    wire: LoadWire,
    endpoint: Option<Endpoint>,
    slot: usize,
    slots: usize,
}

impl Ready {
    fn write(&self, o: &mut Out) {
        o.u8(self.wire.byte());
        if let Some(e) = &self.endpoint {
            write_endpoint(e, o);
        }
        o.len(self.slot).len(self.slots);
    }

    fn read(body: &[u8]) -> Result<Ready, String> {
        let mut i = In::new(body);
        let wire = LoadWire::of(i.u8("load wire")?)?;
        let endpoint = match wire {
            LoadWire::Rdma => Some(read_endpoint(&mut i)?),
            LoadWire::Tcp => None,
        };
        let (slot, slots) = (i.len("slot bytes")?, i.len("slots")?);
        i.end()?;
        Ok(Ready {
            wire,
            endpoint,
            slot,
            slots,
        })
    }
}

/// The bytes of segment `seg` of `t`, a stack of `experts` when it is one:
/// its spans' bytes, one after another.
fn row_bytes(t: &ModelTensor, seg: &Segment, experts: u64) -> Result<usize, GpuError> {
    let spans = seg.spans(t, experts).map_err(|e| GpuError::plan(WHAT, e))?;
    let n: u64 = spans.iter().map(|s| s.bytes.end - s.bytes.start).sum();
    usize::try_from(n).map_err(|_| GpuError::shape(WHAT, format!("{}: {n} bytes", t.name)))
}

/// The SHA-256 of each [`CHUNK`] of `bytes`, hashed on `workers` threads.
fn chunk_digests(bytes: &[u8], workers: usize) -> Vec<[u8; 32]> {
    let chunks: Vec<&[u8]> = bytes.chunks(CHUNK).collect();
    let mut out = vec![[0u8; 32]; chunks.len()];
    let per = chunks.len().div_ceil(workers.max(1)).max(1);
    std::thread::scope(|s| {
        for (part, slots) in chunks.chunks(per).zip(out.chunks_mut(per)) {
            s.spawn(move || {
                for (c, d) in part.iter().zip(slots) {
                    let mut sha = Sha256::new();
                    sha.update(c);
                    *d = sha.finish();
                }
            });
        }
    });
    out
}

/// Copy the run `src` (spans one after another) into `dest`, its length, and
/// hash each [`CHUNK`] of it on `workers` threads, each thread copying the
/// chunks it hashes; the digests in chunk order.
fn fill(dest: &mut [u8], src: &[&[u8]], workers: usize) -> Vec<[u8; 32]> {
    let mut starts = Vec::with_capacity(src.len());
    let mut at = 0;
    for s in src {
        starts.push(at);
        at += s.len();
    }
    let copy = |from: usize, out: &mut [u8]| {
        let mut done = 0;
        let mut k = starts.partition_point(|&s| s <= from).saturating_sub(1);
        while done < out.len() {
            let off = from + done - starts[k];
            let n = (src[k].len() - off).min(out.len() - done);
            out[done..done + n].copy_from_slice(&src[k][off..off + n]);
            done += n;
            k += 1;
        }
    };
    let n = dest.len().div_ceil(CHUNK);
    let mut out = vec![[0u8; 32]; n];
    let per = n.div_ceil(workers.max(1)).max(1);
    std::thread::scope(|s| {
        for ((w, part), slots) in dest
            .chunks_mut(per * CHUNK)
            .enumerate()
            .zip(out.chunks_mut(per))
        {
            let copy = &copy;
            s.spawn(move || {
                for ((j, c), d) in part.chunks_mut(CHUNK).enumerate().zip(slots) {
                    copy((w * per + j) * CHUNK, c);
                    let mut sha = Sha256::new();
                    sha.update(c);
                    *d = sha.finish();
                }
            });
        }
    });
    out
}

// ------------------------------------------------------------- node 0

/// Node 0's RDMA end of a stream: its source buffers' region and its queue
/// pair to the node's slots. Field order is drop order: the queue pair goes
/// before the region, and the caller's buffer outlives both.
struct Source {
    qp: Qp,
    region: Region,
}

impl Source {
    /// An end over `buf` on the RoCE port of `ip`, the control connection's
    /// local address.
    ///
    /// # Safety
    ///
    /// `buf` stays allocated and unmoved until the end drops.
    unsafe fn new(buf: &mut [u8], ip: IpAddr) -> Result<Source, GpuError> {
        let what = "the tier load's RDMA end";
        let IpAddr::V4(ip) = ip else {
            return Err(GpuError::shape(what, "an IPv4 RoCE port is required"));
        };
        let (dev, gid) =
            bloomery_verbs::roce_v2_at(ip).map_err(|e| GpuError::link(what, e.to_string()))?;
        let nic = Nic::open(&dev, gid).map_err(|e| GpuError::link(what, e.to_string()))?;
        // SAFETY: the caller keeps `buf` allocated while the region lives.
        let region = unsafe { nic.register(buf.as_mut_ptr(), buf.len()) }
            .map_err(|e| GpuError::link(what, e.to_string()))?;
        let qp = nic
            .queue_pair(DEPTH as u32)
            .map_err(|e| GpuError::link(what, e.to_string()))?;
        Ok(Source { qp, region })
    }

    fn endpoint(&self) -> Endpoint {
        Endpoint {
            qpn: self.qp.qpn(),
            psn: PSN_NODE0,
            gid: self.qp.nic().gid(),
            addr: self.region.addr(),
            rkey: self.region.rkey(),
        }
    }

    /// Write `len` bytes at byte `from` of the region to the peer's `to`,
    /// keyed `rkey`, and wait by `deadline` for every write's completion. A
    /// failed write is named by its status; a write not complete by the
    /// deadline takes the queue pair to its error state, which flushes it.
    fn write(
        &self,
        (from, len): (usize, usize),
        (to, rkey): (u64, u32),
        deadline: Instant,
    ) -> Result<(), String> {
        let nic = self.qp.nic();
        let mut writes = Vec::new();
        let mut at = 0;
        while at < len {
            let n = (len - at).min(WRITE);
            writes.push(Write {
                from: (from + at) as u64,
                to: to + at as u64,
                len: n as u32,
            });
            at += n;
        }
        let mut wc = [Wc::default(); 16];
        let (mut posted, mut in_flight) = (0, 0usize);
        while posted < writes.len() || in_flight > 0 {
            while posted < writes.len() && in_flight + MAX_CHAIN.min(writes.len() - posted) <= DEPTH
            {
                let chain = &writes[posted..(posted + MAX_CHAIN).min(writes.len())];
                // SAFETY: the region and its buffer outlive every write: this
                // call returns only once each posted write completed, or after
                // the queue pair's error state flushed them all.
                if let Err((taken, e)) = unsafe { self.qp.post_writes(&self.region, chain, rkey) } {
                    in_flight += taken;
                    self.flush(&mut wc, in_flight);
                    return Err(format!("the RDMA post: {e}"));
                }
                posted += chain.len();
                in_flight += chain.len();
            }
            let n = nic.poll(&mut wc).map_err(|e| e.to_string())?;
            for w in &wc[..n] {
                in_flight = in_flight.saturating_sub(1);
                if w.status != 0 {
                    let why = format!(
                        "an RDMA write failed: {} (status {}, vendor error {:#x})",
                        nic.verbs().status_name(w.status),
                        w.status,
                        w.vendor_err
                    );
                    self.flush(&mut wc, in_flight);
                    return Err(why);
                }
                if w.opcode != WC_RDMA_WRITE {
                    return Err(format!("the completion queue returned opcode {}", w.opcode));
                }
            }
            if n == 0 && Instant::now() > deadline {
                self.flush(&mut wc, in_flight);
                return Err(format!(
                    "{in_flight} RDMA writes not complete by their deadline"
                ));
            }
        }
        Ok(())
    }

    /// Take the queue pair to its error state and reap the `in_flight`
    /// writes it flushes, so none reads the buffer after this end drops.
    fn flush(&self, wc: &mut [Wc], mut in_flight: usize) {
        if self.qp.to_error().is_err() {
            return;
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while in_flight > 0 && Instant::now() < deadline {
            match self.qp.nic().poll(wc) {
                Ok(n) => in_flight = in_flight.saturating_sub(n),
                Err(_) => return,
            }
        }
    }
}

/// One row filled on node 0: its index, the source half it is in and the
/// half's bytes, its length, its chunks' digests and the threads that copied
/// and hashed it.
struct Filled<'b> {
    row: usize,
    half: usize,
    buf: &'b mut [u8],
    len: usize,
    digests: Vec<[u8; 32]>,
    workers: usize,
}

impl TierNode {
    /// The tier `open` names, opened on the node at `addr`, which holds no
    /// model file: its rows go from `feed`'s file over `feed`'s wire as the
    /// module describes ([`super::load`]). Load-time only.
    pub fn open_fed(
        addr: std::net::SocketAddr,
        open: &NodeOpen,
        feed: &Feed<'_>,
    ) -> Result<TierNode, GpuError> {
        if open.nvme.is_some() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the tier {} on the node at {addr}, which holds no model file: a paged tier \
                     reads the node's own file's runs",
                    open.name
                ),
            ));
        }
        let rows = open.rows_on(feed.model)?;
        let lens = rows
            .iter()
            .map(|r| {
                row_bytes(
                    &feed.model.tensors[r.tensor],
                    &r.segments[0],
                    feed.model.experts,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let slot = lens.iter().copied().max().unwrap_or(0).max(1);
        let mut node = TierNode::connected(addr, open)?;
        let mut buf = Vec::new();
        buf.try_reserve_exact(2 * slot).map_err(|e| {
            GpuError::shape(WHAT, format!("node 0's two {slot}-byte load buffers: {e}"))
        })?;
        buf.resize(2 * slot, 0);
        let ip = node
            .ch
            .local_ip()
            .map_err(|e| node.lose_now("open", e.to_string()))?;
        let source = match feed.wire {
            // SAFETY: `buf` is not moved or freed until `source` drops below.
            LoadWire::Rdma => Some(unsafe { Source::new(&mut buf, ip)? }),
            LoadWire::Tcp => None,
        };
        let scales = rows
            .iter()
            .map(|r| {
                segment_scales(
                    feed.file,
                    feed.model.experts,
                    &feed.model.tensors[r.tensor],
                    &r.segments[0],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let head = StreamHead {
            wire: feed.wire,
            endpoint: source.as_ref().map(Source::endpoint),
            shards: (0..feed.file.shard_count())
                .map(|s| {
                    let g = feed
                        .file
                        .shard(s)
                        .expect("a split reads each of its shards");
                    let name = feed
                        .file
                        .shard_path(s)
                        .and_then(|p| p.file_name())
                        .map_or_else(
                            || format!("shard-{s}"),
                            |n| n.to_string_lossy().into_owned(),
                        );
                    (name, g.mapping().len() as u64, g.header_bytes().to_vec())
                })
                .collect(),
            header_sha256: model::fixture::header_sha256(feed.file),
            scales,
        };
        open.write(node.req.clear())?;
        head.write(&mut node.req);
        node.call(Tag::Open, OPEN_BASE, "open")?;
        let ready = Ready::read(node.ch.body()).map_err(|e| node.lose_now("open", e))?;
        let slots = match feed.wire {
            LoadWire::Rdma => 2,
            LoadWire::Tcp => 1,
        };
        if ready.wire != feed.wire || ready.slot < slot || ready.slots != slots {
            let why = format!(
                "a load ready on {:?} with {} slots of {} bytes; node 0 streams over {:?} into {slots} \
                 of {slot}",
                ready.wire, ready.slots, ready.slot, feed.wire
            );
            return Err(node.lose_now("open", why));
        }
        let peer = match (&source, ready.endpoint) {
            (Some(s), Some(e)) => {
                s.qp.connect(PSN_NODE0, &e, 0)
                    .map_err(|e| node.lose_now("open", format!("the load's queue pair: {e}")))?;
                Some((e.addr, e.rkey, ready.slot))
            }
            (None, None) => None,
            _ => return Err(node.lose_now("open", "a load ready of another wire".into())),
        };
        let t0 = Instant::now();
        let streamed = node.feed_rows(feed, &rows, &lens, &mut buf, source.as_ref(), peer);
        drop(source);
        let waited = streamed?;
        let secs = t0.elapsed().as_secs_f64();
        let bytes: usize = lens.iter().sum();
        eprintln!(
            "tier node {addr}: {} rows, {bytes} B fed over {:?} in {secs:.2} s ({:.2} GB/s; \
             waiting {:.2} s on the filler (the file's reads, copied and hashed on {} threads), \
             {:.2} s on the wire, {:.2} s on the node's replies)",
            rows.len(),
            feed.wire,
            bytes as f64 / secs.max(1e-9) / 1e9,
            waited.fill.as_secs_f64(),
            waited.workers,
            waited.wire.as_secs_f64(),
            waited.reply.as_secs_f64()
        );
        node.req.clear().u8(LOAD_END);
        node.call(Tag::Load, OPEN_BASE, "load")?;
        node.read_open_reply()?;
        Ok(node)
    }

    /// Send each of `rows` (their lengths `lens`) from `feed`'s file, a
    /// thread filling the next half of `buf` while this one sends the last;
    /// what the sending thread waited on, by phase.
    fn feed_rows(
        &mut self,
        feed: &Feed<'_>,
        rows: &[Row],
        lens: &[usize],
        buf: &mut [u8],
        source: Option<&Source>,
        peer: Option<(u64, u32, usize)>,
    ) -> Result<Waited, GpuError> {
        let (model, file) = (feed.model, feed.file);
        let mut waited = Waited::default();
        let waited_on = &mut waited;
        let ranges = rows
            .iter()
            .map(|r| {
                segment_ranges(
                    file,
                    model.experts,
                    &model.tensors[r.tensor],
                    &r.segments[0],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let half = buf.len() / 2;
        let ranges = &ranges;
        let ((), _) = read_ahead::run(
            file,
            ranges,
            |e| GpuError::plan(WHAT, e),
            |ahead| {
                std::thread::scope(|s| {
                    let (a, b) = buf.split_at_mut(half);
                    let (free_tx, free_rx) = mpsc::channel::<(usize, &mut [u8])>();
                    let (full_tx, full_rx) = mpsc::sync_channel::<Result<Filled<'_>, GpuError>>(1);
                    let _ = free_tx.send((0, a));
                    let _ = free_tx.send((1, b));
                    s.spawn(move || {
                        // The filler and the threads it copies and hashes on float off
                        // the caller's cpu mask, as a binary pins its load thread to one
                        // cpu: on it they would take turns with each other and the sender.
                        let workers = match threads::helper::float_caller() {
                            Ok(()) => workers(),
                            Err(e) => {
                                let _ = full_tx.send(Err(GpuError::plan(WHAT, e)));
                                return;
                            }
                        };
                        let mut pages = feed.release.then(|| PageDrop::new(file));
                        for (i, r) in rows.iter().enumerate() {
                            let Ok((h, buf)) = free_rx.recv() else { return };
                            let filled = (|| {
                                ahead.begin(i).map_err(|e| GpuError::plan(WHAT, e))?;
                                let t = &model.tensors[r.tensor];
                                let f = SegmentFile::of(file, model.experts, t, &r.segments[0])?;
                                let len = lens[i];
                                let digests = fill(&mut buf[..len], &f.bytes, workers);
                                ahead.retire(i);
                                if let Some(pages) = pages.as_mut() {
                                    for (shard, at) in &ranges[i] {
                                        pages
                                            .release(*shard, at.clone())
                                            .map_err(|e| GpuError::plan(WHAT, e))?;
                                    }
                                }
                                Ok((len, digests))
                            })();
                            let sent = match filled {
                                Ok((len, digests)) => full_tx.send(Ok(Filled {
                                    row: i,
                                    half: h,
                                    buf,
                                    len,
                                    digests,
                                    workers,
                                })),
                                Err(e) => full_tx.send(Err(e)),
                            };
                            if sent.is_err() {
                                return;
                            }
                        }
                    });
                    // Rows whose end request is sent and whose reply is unread.
                    let mut unread: VecDeque<usize> = VecDeque::new();
                    for _ in 0..rows.len() {
                        let t = Instant::now();
                        let f = full_rx.recv().map_err(|_| {
                            GpuError::plan(WHAT, "node 0's load filler ended early")
                        })??;
                        waited_on.fill += t.elapsed();
                        waited_on.workers = f.workers;
                        if let (Some(src), Some((to, rkey, slot))) = (source, peer) {
                            // The node's slot `row % 2` is free once the row before
                            // the last in it is answered.
                            let t = Instant::now();
                            while unread.len() > 1 {
                                self.load_reply(&mut unread, lens)?;
                            }
                            waited_on.reply += t.elapsed();
                            let dst = to + ((f.row % 2) * slot) as u64;
                            let deadline = Instant::now()
                                + REPLY
                                + Duration::from_secs(f.len as u64 / OPEN_RATE);
                            let t = Instant::now();
                            src.write((f.half * half, f.len), (dst, rkey), deadline)
                                .map_err(|e| self.lose_now("load", e))?;
                            waited_on.wire += t.elapsed();
                            self.send_row(&f, f.row % 2)?;
                        } else {
                            let t = Instant::now();
                            while !unread.is_empty() {
                                self.load_reply(&mut unread, lens)?;
                            }
                            waited_on.reply += t.elapsed();
                            let t = Instant::now();
                            let mut at = 0;
                            while at < f.len {
                                let n = (f.len - at).min(FRAME_BYTES);
                                self.req
                                    .clear()
                                    .u8(LOAD_BYTES)
                                    .len(f.row)
                                    .len(at)
                                    .bytes(&f.buf[at..at + n]);
                                let within = REPLY + Duration::from_secs(n as u64 / OPEN_RATE);
                                self.call(Tag::Load, within, "load")?;
                                at += n;
                            }
                            waited_on.wire += t.elapsed();
                            self.send_row(&f, 0)?;
                        }
                        unread.push_back(f.row);
                        let _ = free_tx.send((f.half, f.buf));
                    }
                    let t = Instant::now();
                    while !unread.is_empty() {
                        self.load_reply(&mut unread, lens)?;
                    }
                    waited_on.reply += t.elapsed();
                    Ok(())
                })
            },
        )?;
        Ok(waited)
    }

    /// Send row `f`'s end request, naming slot `slot`, without waiting for
    /// its reply.
    fn send_row(&mut self, f: &Filled<'_>, slot: usize) -> Result<(), GpuError> {
        self.req
            .clear()
            .u8(LOAD_ROW)
            .len(f.row)
            .len(slot)
            .len(f.len)
            .len(f.digests.len());
        for d in &f.digests {
            self.req.raw(d);
        }
        let deadline = Instant::now() + REPLY;
        let sent = self.ch.send(Tag::Load, self.req.done(), deadline);
        sent.map_err(|e| self.lose_now("load", e.to_string()))
    }

    /// Read the reply of the oldest row of `unread`, within the bound its
    /// bytes give the node's hash and upload. A refusal ends the node: the
    /// error is the node's.
    fn load_reply(&mut self, unread: &mut VecDeque<usize>, lens: &[usize]) -> Result<(), GpuError> {
        let Some(row) = unread.pop_front() else {
            return Ok(());
        };
        let within = REPLY + Duration::from_secs(lens[row] as u64 / OPEN_RATE);
        match self.ch.recv(Instant::now() + within) {
            Ok(Tag::Ok) => {
                self.control_round_trips += 1;
                Ok(())
            }
            Ok(Tag::Err) => {
                let e = match read_err(self.addr, "load", self.ch.body()) {
                    Ok(e) => e,
                    Err(why) => return Err(self.lose_now("load", why)),
                };
                self.lost = Some(format!("the load request of row {row}: {e}"));
                self.ch.shutdown();
                Err(e)
            }
            Ok(t) => Err(self.lose_now("load", format!("a {t:?} frame where a reply belongs"))),
            Err(e) => Err(self.lose_now("load", format!("row {row}'s reply: {e}"))),
        }
    }
}

// ------------------------------------------------------------- the node

/// The node's RDMA end of a stream: its slots' region and its queue pair to
/// node 0's. Field order is drop order: the queue pair goes before the
/// region, and the region before the slots it registers.
struct SlotsRdma {
    qp: Qp,
    region: Region,
}

/// A fed open on the node: the card, node 0's split from its headers, the
/// tensors and rows of the open, its family's index, the rows' scales, the
/// slots and their RDMA end, the upload, and the next row and the bytes of
/// it a control channel has carried.
pub(super) struct NodeLoad {
    // The upload holds the split; it drops before the card's context.
    upload: SegmentLoad,
    rdma: Option<SlotsRdma>,
    slots: Vec<u8>,
    slot: usize,
    lens: Vec<usize>,
    gpu: Gpu,
    split: Arc<Split>,
    tensors: ModelTensors,
    rows: Vec<Row>,
    open: NodeOpen,
    family: usize,
    scales: Vec<Option<Vec<f32>>>,
    wire: LoadWire,
    next: usize,
    carried: usize,
    t0: Instant,
    /// Where the node's time went: hashing and uploading rows (each on its
    /// own thread, so they overlap) and the rows' wall.
    spent: Spent,
}

/// The node's time by phase over a fed load, printed with its end.
#[derive(Default)]
struct Spent {
    sha: Duration,
    upload: Duration,
    rows: Duration,
}

impl NodeLoad {
    /// Open `open` as fed by `head`: the family by its word among
    /// `families`, node 0's split from the head's headers and their digest,
    /// the open's rows against it, a paged tier refused, the card, the
    /// slots, and the ready reply into `reply`. Every refusal is by name,
    /// before node 0 sends a byte of a row.
    pub(super) fn open(
        families: &[NodeFamily],
        open: NodeOpen,
        head: StreamHead,
        ip: IpAddr,
        reply: &mut Out,
    ) -> Result<NodeLoad, GpuError> {
        let family = families
            .iter()
            .position(|f| f.word == open.family)
            .ok_or_else(|| {
                GpuError::shape(
                    SERVER,
                    format!(
                        "a tier of the family {:?}; this server opens {:?}",
                        open.family,
                        families.iter().map(|f| f.word).collect::<Vec<_>>()
                    ),
                )
            })?;
        if open.nvme.is_some() {
            return Err(GpuError::shape(
                SERVER,
                "a paged tier on a node with no model file: the paged tier reads the node's own \
                 file's runs",
            ));
        }
        let split = Split::from_headers(
            head.shards
                .into_iter()
                .map(|(name, len, header)| (name.into(), header, len))
                .collect(),
        )
        .map_err(|e| GpuError::shape(SERVER, format!("node 0's shard headers: {e}")))?;
        let digest = model::fixture::header_sha256(&split);
        if digest != head.header_sha256 {
            return Err(GpuError::shape(
                SERVER,
                format!(
                    "node 0's shard headers arrived with the digest {digest}, node 0 sent {}: the \
                     stream corrupted them",
                    head.header_sha256
                ),
            ));
        }
        let split = Arc::new(split);
        let tensors = (families[family].tensors)(&split)?;
        let rows = open.rows_on(&tensors)?;
        if head.scales.len() != rows.len() {
            return Err(GpuError::shape(
                SERVER,
                format!("{} rows' scales for {} rows", head.scales.len(), rows.len()),
            ));
        }
        let lens = rows
            .iter()
            .map(|r| row_bytes(&tensors.tensors[r.tensor], &r.segments[0], tensors.experts))
            .collect::<Result<Vec<_>, _>>()?;
        let slot = lens.iter().copied().max().unwrap_or(0).max(1);
        let n = match head.wire {
            LoadWire::Rdma => 2,
            LoadWire::Tcp => 1,
        };
        let gpu = Gpu::open_card(&open.name, open.device)?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(n * slot)
            .map_err(|e| GpuError::shape(SERVER, format!("{n} load slots of {slot} bytes: {e}")))?;
        slots.resize(n * slot, 0);
        let rdma = match (head.wire, head.endpoint) {
            (LoadWire::Rdma, Some(peer)) => {
                let what = "tier_server: the load's RDMA end";
                let IpAddr::V4(v4) = ip else {
                    return Err(GpuError::shape(what, "an IPv4 RoCE port is required"));
                };
                let (dev, gid) = bloomery_verbs::roce_v2_at(v4)
                    .map_err(|e| GpuError::link(what, e.to_string()))?;
                let nic = Nic::open(&dev, gid).map_err(|e| GpuError::link(what, e.to_string()))?;
                // SAFETY: the load holds `slots` unmoved after this region,
                // which drops first (field order).
                let region = unsafe { nic.register(slots.as_mut_ptr(), slots.len()) }
                    .map_err(|e| GpuError::link(what, e.to_string()))?;
                let qp = nic
                    .queue_pair(1)
                    .map_err(|e| GpuError::link(what, e.to_string()))?;
                qp.connect(PSN_NODE, &peer, 0)
                    .map_err(|e| GpuError::link(what, e.to_string()))?;
                Some(SlotsRdma { qp, region })
            }
            (LoadWire::Tcp, None) => None,
            _ => return Err(GpuError::shape(SERVER, "a load wire without its endpoint")),
        };
        Ready {
            wire: head.wire,
            endpoint: rdma.as_ref().map(|r| Endpoint {
                qpn: r.qp.qpn(),
                psn: PSN_NODE,
                gid: r.qp.nic().gid(),
                addr: r.region.addr(),
                rkey: r.region.rkey(),
            }),
            slot,
            slots: n,
        }
        .write(reply);
        let upload =
            SegmentLoad::new(gpu.stream(), Arc::clone(&split), &tensors, &rows, open.card)?;
        Ok(NodeLoad {
            upload,
            rdma,
            slots,
            slot,
            lens,
            gpu,
            split,
            tensors,
            rows,
            open,
            family,
            scales: head.scales,
            wire: head.wire,
            next: 0,
            carried: 0,
            t0: Instant::now(),
            spent: Spent::default(),
        })
    }

    /// The rows taken of the open's, and their bytes.
    pub(super) fn state(&self) -> String {
        format!(
            "{} of the {} rows node 0 streams taken ({} of {} bytes)",
            self.next,
            self.rows.len(),
            self.lens[..self.next].iter().sum::<usize>(),
            self.lens.iter().sum::<usize>()
        )
    }

    /// One load request: bytes of the next row over the channel, or the
    /// next row's end (its bytes hashed against node 0's digests and
    /// uploaded); `true` for the load's end, which [`NodeLoad::end`] takes.
    /// Anything out of the open's order is refused by name.
    pub(super) fn take(&mut self, i: &mut In<'_>) -> Result<bool, String> {
        match i.u8("load kind")? {
            LOAD_BYTES => {
                let (row, at) = (i.len("row")?, i.len("at")?);
                let bytes = i.bytes("row bytes")?;
                i.end()?;
                let len = self.lens.get(row).copied().unwrap_or(0);
                if self.wire != LoadWire::Tcp
                    || row != self.next
                    || at != self.carried
                    || at + bytes.len() > len
                {
                    return Err(format!(
                        "bytes {at}..{} of row {row} over {:?}; the load takes row {}'s bytes from \
                         {} of {len} over its slots' wire",
                        at + bytes.len(),
                        self.wire,
                        self.next,
                        self.carried
                    ));
                }
                self.slots[at..at + bytes.len()].copy_from_slice(bytes);
                self.carried += bytes.len();
                Ok(false)
            }
            LOAD_ROW => {
                let (row, slot, len, n) = (
                    i.len("row")?,
                    i.len("slot")?,
                    i.len("row bytes")?,
                    i.len("digests")?,
                );
                let mut sent = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    let mut d = [0u8; 32];
                    for b in &mut d {
                        *b = i.u8("digest")?;
                    }
                    sent.push(d);
                }
                i.end()?;
                self.row(row, slot, len, &sent).map_err(|e| e.to_string())?;
                Ok(false)
            }
            LOAD_END => {
                i.end()?;
                if self.next != self.rows.len() {
                    return Err(format!("the load's end with {}", self.state()));
                }
                Ok(true)
            }
            k => Err(format!("a load request of kind {k}")),
        }
    }

    /// Row `row`'s end: `len` bytes in slot `slot`, `sent` its chunks'
    /// digests as node 0 read them; hashed here while they upload.
    fn row(
        &mut self,
        row: usize,
        slot: usize,
        len: usize,
        sent: &[[u8; 32]],
    ) -> Result<(), GpuError> {
        let want = self.lens.get(row).copied();
        let slots = self.slots.len() / self.slot;
        let carried = self.wire == LoadWire::Rdma || self.carried == len;
        if row != self.next
            || want != Some(len)
            || slot >= slots
            || !carried
            || sent.len() != len.div_ceil(CHUNK)
        {
            return Err(GpuError::shape(
                SERVER,
                format!(
                    "row {row}'s end in slot {slot} of {slots}: {len} bytes, {} digests; the load \
                     takes row {} of {want:?} bytes, {} carried",
                    sent.len(),
                    self.next,
                    self.carried
                ),
            ));
        }
        let at = slot * self.slot;
        let bytes = &self.slots[at..at + len];
        let r = &self.rows[row];
        let t = &self.tensors.tensors[r.tensor];
        let scale = self.scales[row].take();
        let stream = self.gpu.stream();
        let upload = &mut self.upload;
        let t_row = Instant::now();
        let (got, uploaded) = std::thread::scope(|s| {
            let hashing = s.spawn(|| {
                let t = Instant::now();
                (chunk_digests(bytes, workers()), t.elapsed())
            });
            let t_up = Instant::now();
            let uploaded = upload.upload(stream, t, &r.segments[0], bytes, scale);
            (hashing.join(), (uploaded, t_up.elapsed()))
        });
        let (uploaded, up) = uploaded;
        let (got, sha) = got.map_err(|_| GpuError::plan(SERVER, "a load hash thread panicked"))?;
        self.spent.sha += sha;
        self.spent.upload += up;
        self.spent.rows += t_row.elapsed();
        if let Some(j) = (0..sent.len()).find(|&j| got[j] != sent[j]) {
            return Err(GpuError::shape(
                SERVER,
                format!(
                    "{} (experts {:?}): chunk {j} of its {len} bytes arrived with sha256 {}, node 0 \
                     read {}: the stream corrupted it",
                    t.name,
                    r.segments[0].experts.as_ref().map(|e| e.ids().len()),
                    hex(&got[j]),
                    hex(&sent[j])
                ),
            ));
        }
        uploaded?;
        self.next += 1;
        self.carried = 0;
        Ok(())
    }

    /// The load's end, once [`NodeLoad::take`] took it: the slots released,
    /// the uploads synchronized, the family's computation built over them
    /// and the tier's work opened, with the open it serves.
    pub(super) fn end(self, families: &[NodeFamily]) -> Result<(TierWork, NodeOpen), GpuError> {
        let NodeLoad {
            upload,
            rdma,
            slots,
            lens,
            gpu,
            split,
            tensors,
            open,
            family,
            t0,
            spent,
            ..
        } = self;
        drop(rdma);
        drop(slots);
        let w = upload.finish(gpu.stream())?;
        let experts = (families[family].experts)(&gpu, &split, &tensors, &w, &open)?;
        eprintln!(
            "tier_server: node 0 fed {} rows, {} B, in {:.2} s (rows {:.2} s: sha256 {:.2} s beside \
             upload {:.2} s)",
            lens.len(),
            lens.iter().sum::<usize>(),
            t0.elapsed().as_secs_f64(),
            spent.rows.as_secs_f64(),
            spent.sha.as_secs_f64(),
            spent.upload.as_secs_f64()
        );
        let work = TierWork::open(gpu, w, experts, open.shape, open.cols)?;
        Ok((work, open))
    }
}
