//! The TCP end of mirrored pages: one full-duplex TCP stream between the two
//! ends, a receiver thread on each, and the pump thread that drives the end.
//!
//! A post is one write of framed moves. A frame is a header of [`HEAD`]
//! bytes (its tag, the move's length, the byte of the other end's page it
//! lands at and its sequence number, each direction counting from 1) and the
//! move's bytes, read from this end's page during the post. The other end's
//! receiver thread lands each frame into its own page in stream order, a
//! frame's last word with Release, and acks a post's last frame once it has
//! landed. One stream keeps its bytes in order, so a post's generation lands
//! after every byte before it, as the RDMA end's writes on one queue pair do
//! ([`super::rdma`]).
//!
//! The stream is full duplex: an ack rides the direction that carries the
//! other end's posts, written once its post lands and so before any reply
//! that post releases, and one reader on each end sees a peer that closed or
//! failed. The two threads that write to a stream (the pump's posts, the
//! receiver's acks) take turns under one lock, so frames never interleave.
//!
//! A post is in flight from its write until the ack of its last frame
//! arrives ([`Wire::reap`]); no later write reads its bytes. A frame out of
//! sequence, a frame outside the page or not whole words, a short read, a
//! closed peer, a failed write and an abort end the wire for good, by name:
//! every later post fails with that name, and the next reap returns it once,
//! after which no post is in flight (the unacknowledged ones finished,
//! failed), as the RDMA end's flushed writes do. An end that refuses what it
//! reads sends the other end a last frame with why, so both ends name the
//! cause.
//!
//! The ends meet through a listener on the stage's side, which the tier's
//! end dials. Each end's first frame is a hello carrying the stage's nonce
//! and its page's bytes; an end refuses by name a hello whose nonce or page
//! differs from its own.

use std::io::{self, BufReader, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use threads::helper::{Placement, spawn_helper};

use super::pump::{Crossing, Half, Move, Page, PostCols, Reader, Wire};
use super::{DRAIN, End, Pump, Side};
use crate::GpuError;
use crate::graph::MappedHost;

/// Bytes of a frame's header: tag (u32), length (u32), the byte it lands at
/// (u64) and its sequence number (u64), little-endian.
const HEAD: usize = 24;

/// A move that does not end its post.
const MOVE: u32 = 1;
/// A post's last move: acked once it has landed.
const LAST: u32 = 2;
/// An ack: its sequence number is the last frame of a post that landed.
const ACK: u32 = 3;
/// The first frame each way: the page's bytes where a move's byte goes and
/// the stage's nonce where its sequence number goes.
const HELLO: u32 = u32::from_le_bytes(*b"BTW1");
/// The sender's last frame: why it refused the stream, as text in place of a
/// move's bytes.
const BYE: u32 = 4;
/// Bytes of a bye's text at most.
const BYE_MAX: usize = 1024;

/// How long the tier's end may take to dial and both hellos to cross.
const MEET: Duration = Duration::from_secs(10);
/// How long one write may block: a post's under its end's lock, which a
/// settle waits for within [`DRAIN`].
const WRITE: Duration = DRAIN;
/// How often an accept that found no dial looks again.
const ACCEPT_POLL: Duration = Duration::from_millis(1);
/// Bytes the receiver reads and lands at a time.
const CHUNK: usize = 64 << 10;
/// Microseconds a blocking read on the stream busy-polls its device queue
/// before it sleeps (`SO_BUSY_POLL`): the rigs' `net.core.busy_read`.
#[cfg(target_os = "linux")]
const BUSY_POLL: libc::c_int = 50;

/// One frame's header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Head {
    tag: u32,
    len: u32,
    at: u64,
    seq: u64,
}

impl Head {
    fn bytes(self) -> [u8; HEAD] {
        let mut b = [0; HEAD];
        b[..4].copy_from_slice(&self.tag.to_le_bytes());
        b[4..8].copy_from_slice(&self.len.to_le_bytes());
        b[8..16].copy_from_slice(&self.at.to_le_bytes());
        b[16..].copy_from_slice(&self.seq.to_le_bytes());
        b
    }

    fn of(b: &[u8; HEAD]) -> Head {
        let (tag, rest) = b.split_first_chunk::<4>().expect("a header holds its tag");
        let (len, rest) = rest
            .split_first_chunk::<4>()
            .expect("a header holds its length");
        let (at, seq) = rest
            .split_first_chunk::<8>()
            .expect("a header holds its byte");
        let seq: &[u8; 8] = seq
            .try_into()
            .expect("a header ends with its sequence number");
        Head {
            tag: u32::from_le_bytes(*tag),
            len: u32::from_le_bytes(*len),
            at: u64::from_le_bytes(*at),
            seq: u64::from_le_bytes(*seq),
        }
    }
}

/// What a hello carries: the stage's nonce and the sender's page's bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Hello {
    nonce: u64,
    bytes: u64,
}

impl Hello {
    fn head(self) -> Head {
        Head {
            tag: HELLO,
            len: 0,
            at: self.bytes,
            seq: self.nonce,
        }
    }

    /// `h` as the other end's hello to this one, or why not.
    fn check(self, h: Head) -> Result<(), String> {
        if h.tag != HELLO || h.len != 0 {
            return Err(format!(
                "the first frame is not a tier link's hello (tag {:#x}, {} bytes)",
                h.tag, h.len
            ));
        }
        if h.seq != self.nonce {
            return Err("a hello with another nonce: not this link's other end".to_string());
        }
        if h.at != self.bytes {
            return Err(format!(
                "a hello for a page of {} bytes, where this end's page holds {}",
                h.at, self.bytes
            ));
        }
        Ok(())
    }
}

/// What one end tells the other so they meet: on the stage's side its
/// listener's address and its nonce; on the tier's side the address it
/// dialed from once connected (port 0 before), and the nonce it dialed with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TcpPeer {
    pub addr: SocketAddr,
    pub nonce: u64,
}

/// What the pump's posts and the receiver thread share.
#[derive(Default)]
struct Shared {
    /// The sequence number of the last frame posted, which an ack may name.
    sent: AtomicU64,
    /// The sequence number of the last frame acked.
    acked: AtomicU64,
    /// Why the wire ended: its first cause.
    ended: OnceLock<String>,
}

impl Shared {
    /// Take an ack of frame `seq`: past every ack before it, at most the last
    /// frame posted.
    fn take_ack(&self, seq: u64) -> Result<(), String> {
        let (acked, sent) = (
            self.acked.load(Ordering::Relaxed),
            self.sent.load(Ordering::Acquire),
        );
        if seq <= acked || seq > sent {
            return Err(format!(
                "an ack of frame {seq} out of sequence: frames up to {acked} acked, {sent} posted"
            ));
        }
        self.acked.store(seq, Ordering::Release);
        Ok(())
    }

    /// End the wire for `why`, unless it ended already, and close `s`'s
    /// stream both ways, so a blocked read or write returns; the first cause.
    fn end(&self, why: String, s: &TcpStream) -> String {
        let why = self.ended.get_or_init(|| why).clone();
        drop(s.shutdown(Shutdown::Both));
        why
    }
}

fn lock(tx: &Mutex<TcpStream>) -> MutexGuard<'_, TcpStream> {
    tx.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Why a receiver thread stopped.
#[derive(Debug)]
enum Stop {
    /// The stream closed or failed, or the other end said why: there is no
    /// one to tell.
    Ended(String),
    /// This end refused what it read: the other end is told why.
    Refused(String),
}

/// Send the other end a bye with `why`, cut to [`BYE_MAX`] bytes. A write
/// that fails is not reported: the wire ends by `why` either way.
fn tell(mut s: impl Write, why: &str) {
    let mut n = why.len().min(BYE_MAX);
    while !why.is_char_boundary(n) {
        n -= 1;
    }
    let head = Head {
        tag: BYE,
        len: u32::try_from(n).expect("a bye's text fits its length"),
        at: 0,
        seq: 0,
    };
    let mut frame = head.bytes().to_vec();
    frame.extend_from_slice(&why.as_bytes()[..n]);
    drop(s.write_all(&frame));
}

/// Read the text of the bye `h` heads from `rx`: why the other end ended.
fn bye(rx: &mut impl Read, h: Head) -> Stop {
    let len = h.len as usize;
    if len > BYE_MAX {
        return Stop::Refused(format!("a bye of {len} bytes, past {BYE_MAX}"));
    }
    let mut text = vec![0; len];
    match rx.read_exact(&mut text) {
        Ok(()) => Stop::Ended(format!(
            "the other end ended the link: {}",
            String::from_utf8_lossy(&text)
        )),
        Err(e) => Stop::Ended(format!("a short read: a bye of {len} bytes: {e}")),
    }
}

/// A connected wire's stream and receiver thread.
///
/// Its drop closes the stream both ways and joins the receiver.
struct Conn {
    /// The stream the pump's posts and the receiver's acks write to.
    tx: Arc<Mutex<TcpStream>>,
    /// The stream again, to close it without the writers' lock.
    ctl: TcpStream,
    shared: Arc<Shared>,
    receiver: Option<JoinHandle<()>>,
}

impl Conn {
    fn end(&self, why: String) -> String {
        self.shared.end(why, &self.ctl)
    }
}

impl Drop for Conn {
    fn drop(&mut self) {
        drop(self.ctl.shutdown(Shutdown::Both));
        if let Some(t) = self.receiver.take() {
            // A receiver that panicked has stopped; there is nothing to land.
            drop(t.join());
        }
    }
}

/// The TCP wire of one end: this end's page, which a post reads, and once
/// connected its stream.
pub(super) struct Tcp<P = MappedHost> {
    page: Arc<P>,
    conn: Option<Conn>,
    /// The sequence number of the last frame posted.
    seq: u64,
    /// The last post's last frame: the post is in flight until it is acked.
    last: u64,
    /// Whether a reap returned why the wire ended.
    told: bool,
    /// One post's frames, with room at load for the longest post.
    buf: Vec<u8>,
}

impl<P: Page + 'static> Tcp<P> {
    /// An unconnected wire over `page`, its post buffer `cap` bytes.
    fn new(page: Arc<P>, cap: usize) -> Tcp<P> {
        Tcp {
            page,
            conn: None,
            seq: 0,
            last: 0,
            told: false,
            buf: Vec::with_capacity(cap),
        }
    }

    /// Carry posts over `stream` and land the other end's into this page on
    /// a receiver thread, which first checks the other end's hello against
    /// `hello` when this end has not yet read it.
    fn attach(&mut self, stream: TcpStream, hello: Option<Hello>) -> Result<(), String> {
        let clone = || {
            stream
                .try_clone()
                .map_err(|e| format!("the link's stream: {e}"))
        };
        let (rx, ctl) = (clone()?, clone()?);
        let tx = Arc::new(Mutex::new(stream));
        let shared = Arc::new(Shared::default());
        let (page, acks, state) = (Arc::clone(&self.page), Arc::clone(&tx), Arc::clone(&shared));
        let (receiver, _) = spawn_helper("tier-link-tcp", Placement::Float, move || {
            let mut rx = BufReader::with_capacity(CHUNK, rx);
            let why = match receive(&mut rx, &acks, &*page, &state, hello) {
                Stop::Ended(why) => why,
                Stop::Refused(why) => {
                    tell(&mut *lock(&acks), &why);
                    why
                }
            };
            state.end(why, rx.get_ref());
        })
        .map_err(|e| format!("the link's receiver thread: {e}"))?;
        self.conn = Some(Conn {
            tx,
            ctl,
            shared,
            receiver: Some(receiver),
        });
        Ok(())
    }
}

impl<P: Page> Wire for Tcp<P> {
    fn post(&mut self, moves: &[Move]) -> Result<(), String> {
        let conn = self
            .conn
            .as_ref()
            .ok_or("the link's TCP end posted before it was connected")?;
        if let Some(why) = conn.shared.ended.get() {
            return Err(why.clone());
        }
        let Some(last) = moves.iter().rposition(|m| m.len > 0) else {
            return Ok(());
        };
        self.buf.clear();
        for (k, m) in moves.iter().enumerate().filter(|(_, m)| m.len > 0) {
            let len = u32::try_from(m.len).map_err(|_| format!("a move of {} bytes", m.len))?;
            self.seq += 1;
            let head = Head {
                tag: if k == last { LAST } else { MOVE },
                len,
                at: m.to as u64,
                seq: self.seq,
            };
            self.buf.extend_from_slice(&head.bytes());
            for off in (m.from..m.from + m.len).step_by(4) {
                let w = self.page.word(off).load(Ordering::Relaxed);
                self.buf.extend_from_slice(&w.to_le_bytes());
            }
        }
        self.last = self.seq;
        conn.shared.sent.store(self.seq, Ordering::Release);
        let wrote = lock(&conn.tx).write_all(&self.buf);
        wrote.map_err(|e| {
            conn.end(format!(
                "the link's TCP post of {} bytes: {e}",
                self.buf.len()
            ))
        })
    }

    fn reap(&mut self) -> Result<bool, String> {
        let Some(conn) = &self.conn else {
            return Ok(true);
        };
        if let Some(why) = conn.shared.ended.get() {
            if self.told {
                return Ok(true);
            }
            self.told = true;
            return Err(why.clone());
        }
        Ok(conn.shared.acked.load(Ordering::Acquire) >= self.last)
    }

    fn abort(&mut self) -> Result<(), String> {
        if let Some(conn) = &self.conn {
            conn.end("the link's TCP wire was aborted".to_string());
        }
        Ok(())
    }
}

/// The receiver thread's work: check the other end's hello against `hello`
/// if given, then land every frame from `rx` into `page` in sequence, ack
/// each post's last frame over `tx` once it has landed, and take each of the
/// other end's acks into `shared`, until the stream ends, the other end says
/// why, or a frame breaks the protocol: why it stopped.
fn receive<P: Page + ?Sized>(
    rx: &mut impl Read,
    tx: &Mutex<TcpStream>,
    page: &P,
    shared: &Shared,
    hello: Option<Hello>,
) -> Stop {
    let mut chunk = vec![0; CHUNK];
    let mut head = [0; HEAD];
    if let Some(hello) = hello {
        if let Err(why) = read_head(rx, &mut head) {
            return Stop::Ended(format!("the other end's hello: {why}"));
        }
        let h = Head::of(&head);
        if h.tag == BYE {
            return bye(rx, h);
        }
        if let Err(why) = hello.check(h) {
            return Stop::Refused(format!("the other end's hello: {why}"));
        }
    }
    let mut next = 1;
    loop {
        if let Err(why) = read_head(rx, &mut head) {
            return Stop::Ended(why);
        }
        let h = Head::of(&head);
        match h.tag {
            ACK => {
                if let Err(why) = shared.take_ack(h.seq) {
                    return Stop::Refused(why);
                }
            }
            MOVE | LAST => {
                if h.seq != next {
                    return Stop::Refused(format!(
                        "a frame out of sequence: frame {} where frame {next} belongs",
                        h.seq
                    ));
                }
                if let Err(stop) = land(rx, &mut chunk, page, h) {
                    return stop;
                }
                next += 1;
                if h.tag == LAST {
                    let ack = Head {
                        tag: ACK,
                        len: 0,
                        at: 0,
                        seq: h.seq,
                    };
                    if let Err(e) = lock(tx).write_all(&ack.bytes()) {
                        return Stop::Ended(format!("the ack of frame {}: {e}", h.seq));
                    }
                }
            }
            BYE => return bye(rx, h),
            tag => return Stop::Refused(format!("a frame of unknown tag {tag:#x}")),
        }
    }
}

/// Read one header into `head`: a stream that ends before its first byte is
/// a closed peer, one that ends inside it a short read.
fn read_head(rx: &mut impl Read, head: &mut [u8; HEAD]) -> Result<(), String> {
    let mut at = 0;
    while at < HEAD {
        match rx.read(&mut head[at..]) {
            Ok(0) if at == 0 => return Err("the other end closed the link".to_string()),
            Ok(0) => {
                return Err(format!(
                    "a short read: the stream ended {at} bytes into a frame's header"
                ));
            }
            Ok(n) => at += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) if at == 0 => return Err(format!("the link's stream failed: {e}")),
            Err(e) => {
                return Err(format!(
                    "a short read: {e} {at} bytes into a frame's header"
                ));
            }
        }
    }
    Ok(())
}

/// Land the bytes of the move `h` heads from `rx` into `page`, `chunk` at a
/// time, its last word with Release. A move that is not whole words inside
/// the page is refused by name before any byte lands.
fn land<P: Page + ?Sized>(
    rx: &mut impl Read,
    chunk: &mut [u8],
    page: &P,
    h: Head,
) -> Result<(), Stop> {
    let len = h.len as usize;
    let at = usize::try_from(h.at)
        .ok()
        .filter(|&at| {
            len > 0
                && at.is_multiple_of(4)
                && len.is_multiple_of(4)
                && at.checked_add(len).is_some_and(|end| end <= page.bytes())
        })
        .ok_or_else(|| {
            Stop::Refused(format!(
                "frame {}: {len} bytes at byte {}, not whole words inside the page's {} bytes",
                h.seq,
                h.at,
                page.bytes()
            ))
        })?;
    let mut done = 0;
    while done < len {
        let n = (len - done).min(chunk.len());
        rx.read_exact(&mut chunk[..n]).map_err(|e| {
            Stop::Ended(format!(
                "a short read: frame {} of {len} bytes ended between bytes {done} and {}: {e}",
                h.seq,
                done + n
            ))
        })?;
        let (words, _) = chunk[..n].as_chunks::<4>();
        for (i, w) in words.iter().enumerate() {
            let off = done + 4 * i;
            let o = if off + 4 == len {
                Ordering::Release
            } else {
                Ordering::Relaxed
            };
            page.word(at + off).store(u32::from_le_bytes(*w), o);
        }
        done += n;
    }
    Ok(())
}

/// Set `s` up for the link: no Nagle delay, the write bound, and busy
/// polling where the platform has it.
fn tune(s: &TcpStream) -> io::Result<()> {
    s.set_nodelay(true)?;
    s.set_write_timeout(Some(WRITE))?;
    busy_poll(s);
    Ok(())
}

/// `SO_BUSY_POLL` at [`BUSY_POLL`] on `s`. A refusal (raising it past
/// `net.core.busy_read` needs `CAP_NET_ADMIN`) is a warning, once a
/// process: the sysctl may already cover it.
#[cfg(target_os = "linux")]
fn busy_poll(s: &TcpStream) {
    use std::os::fd::AsRawFd;
    static REFUSED: std::sync::Once = std::sync::Once::new();
    let us = BUSY_POLL;
    let len = libc::socklen_t::try_from(size_of::<libc::c_int>())
        .expect("an int's size fits a socklen_t");
    // SAFETY: the descriptor is `s`'s, open for the call; the value is a live
    // `c_int` and `len` its size.
    let rc = unsafe {
        libc::setsockopt(
            s.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_BUSY_POLL,
            (&raw const us).cast(),
            len,
        )
    };
    if rc != 0 {
        let e = io::Error::last_os_error();
        REFUSED.call_once(|| {
            // `eprintln!` panics when stderr fails.
            drop(writeln!(
                io::stderr(),
                "tier link: SO_BUSY_POLL refused ({e}); its reads busy-poll as \
                 net.core.busy_read says"
            ));
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn busy_poll(_: &TcpStream) {}

/// Dial the stage's end at `peer` and send this end's hello, `hello`.
fn dial(peer: &TcpPeer, hello: Hello) -> Result<TcpStream, String> {
    let s = TcpStream::connect_timeout(&peer.addr, MEET)
        .map_err(|e| format!("dialing the stage's end at {}: {e}", peer.addr))?;
    tune(&s)
        .and_then(|()| (&s).write_all(&hello.head().bytes()))
        .map_err(|e| format!("the hello to the stage's end at {}: {e}", peer.addr))?;
    Ok(s)
}

/// Accept the tier's end on `l` within [`MEET`] (from `peer`'s address when
/// it names a port), check its hello against `hello` and answer with it.
fn accept(l: &TcpListener, hello: Hello, peer: &TcpPeer) -> Result<TcpStream, String> {
    let until = Instant::now() + MEET;
    let at = l.local_addr().map_err(|e| format!("the listener: {e}"))?;
    l.set_nonblocking(true)
        .map_err(|e| format!("the listener at {at}: {e}"))?;
    let (s, from) = loop {
        match l.accept() {
            Ok(a) => break a,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock && Instant::now() < until => {
                std::thread::sleep(ACCEPT_POLL);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                return Err(format!("no tier end dialed {at} within {MEET:?}"));
            }
            Err(e) => return Err(format!("accepting on {at}: {e}")),
        }
    };
    if peer.addr.port() != 0 && from != peer.addr {
        return Err(format!(
            "a dial from {from} on {at}, where the tier's end at {} belongs",
            peer.addr
        ));
    }
    let mut head = [0; HEAD];
    let left = until
        .saturating_duration_since(Instant::now())
        .max(ACCEPT_POLL);
    s.set_nonblocking(false)
        .and_then(|()| tune(&s))
        .and_then(|()| s.set_read_timeout(Some(left)))
        .and_then(|()| (&s).read_exact(&mut head))
        .map_err(|e| format!("the hello of the tier's end from {from}: {e}"))?;
    if let Err(why) = hello.check(Head::of(&head)) {
        tell(&s, &why);
        return Err(format!("the tier's end from {from}: {why}"));
    }
    s.set_read_timeout(None)
        .and_then(|()| (&s).write_all(&hello.head().bytes()))
        .map_err(|e| format!("the hello to the tier's end at {from}: {e}"))?;
    Ok(s)
}

/// A nonce for a stage's listener, which only its tier's end learns (over
/// the control channel or in process).
fn nonce() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u32(std::process::id());
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos()),
    );
    h.finish()
}

/// Bytes of the longest post over `crossings`: every span and the
/// generation's word, or a reset's two words, each with its header.
fn post_bytes(crossings: &[Crossing]) -> usize {
    crossings
        .iter()
        .map(|c| {
            let spans: usize = c.spans.iter().map(|s| HEAD + s.bytes.len()).sum();
            spans.max(HEAD + 4) + HEAD + 4
        })
        .max()
        .unwrap_or(0)
}

/// One TCP end of mirrored pages, as a node holds it: its page, the
/// crossings whose tail lines start at byte `tail` of both pages, on the
/// stage's side its listener until the tier's end dials it, and once
/// connected its stream, receiver thread and pump thread.
///
/// The pump stops before the end settles and drops its stream and page. An
/// end whose posts cannot be shown over stays alive.
pub struct TcpEnd {
    pump: Option<Pump>,
    end: Option<Arc<End<Tcp>>>,
    listener: Option<TcpListener>,
    /// The nonce this end's hello carries and its page's bytes.
    hello: Hello,
    /// What [`TcpEnd::endpoint`] names: the listener's address on the
    /// stage's side; on the tier's side the address it dialed from once
    /// connected, port 0 before.
    addr: SocketAddr,
}

impl TcpEnd {
    /// The end on side `side` over `page`, carrying `crossings` with their
    /// tail lines from byte `tail` ([`super::link_tail`]), each post
    /// carrying the columns `cols` holds. The stage's side listens on `ip`,
    /// a port the system picks; the tier's side dials the stage's at
    /// [`TcpEnd::connect`]. The relay takes every word as `page` holds it
    /// now. A crossing whose counter, progress word or span's columns are not
    /// 4-aligned before the tail is refused by name. Load-time only.
    pub fn new(
        page: Arc<MappedHost>,
        tail: usize,
        crossings: Vec<Crossing>,
        side: Reader,
        ip: IpAddr,
        cols: PostCols,
    ) -> Result<TcpEnd, GpuError> {
        let what = "the link's TCP end";
        let cap = post_bytes(&crossings);
        let half =
            Half::new(crossings, side, tail, &*page, cols).map_err(|e| GpuError::shape(what, e))?;
        let bytes = page.bytes() as u64;
        let (listener, addr, nonce) = match side {
            Reader::Stage => {
                let l = TcpListener::bind((ip, 0))
                    .map_err(|e| GpuError::link(what, format!("listening on {ip}: {e}")))?;
                let addr = l
                    .local_addr()
                    .map_err(|e| GpuError::link(what, format!("the listener on {ip}: {e}")))?;
                (Some(l), addr, nonce())
            }
            Reader::Tier => (None, SocketAddr::new(ip, 0), 0),
        };
        let wire = Tcp::new(Arc::clone(&page), cap);
        Ok(TcpEnd {
            pump: None,
            end: Some(Arc::new(End::new(half, wire, page))),
            listener,
            hello: Hello { nonce, bytes },
            addr,
        })
    }

    /// What the other end needs to meet this one.
    #[must_use]
    pub fn endpoint(&self) -> TcpPeer {
        TcpPeer {
            addr: self.addr,
            nonce: self.hello.nonce,
        }
    }

    /// Meet the other end, `peer`, and start the pump thread: the stage's
    /// side accepts the tier's dial (from `peer`'s address when it names a
    /// port) and checks its hello for this end's nonce; the tier's side dials
    /// `peer` with `peer`'s nonce and checks the stage's hello on its
    /// receiver thread, whose failure ends the wire by name.
    pub fn connect(&mut self, peer: &TcpPeer) -> Result<(), GpuError> {
        let what = "the link's TCP end";
        let (stream, unread) = match self.listener.take() {
            Some(l) => (accept(&l, self.hello, peer), None),
            None => {
                self.hello.nonce = peer.nonce;
                (dial(peer, self.hello), Some(self.hello))
            }
        };
        let stream = stream.map_err(|e| GpuError::link(what, e))?;
        if unread.is_some()
            && let Ok(a) = stream.local_addr()
        {
            self.addr = a;
        }
        self.end()
            .lock()
            .1
            .attach(stream, unread)
            .map_err(|e| GpuError::link(what, e))?;
        let side: Arc<dyn Side> = self.end().clone();
        self.pump = Some(Pump::start(vec![side]).map_err(|e| GpuError::plan(what, e))?);
        Ok(())
    }

    /// Release every counter `ends` carry until the ends drop
    /// ([`super::RdmaEnd::release`]'s contract, over TCP ends).
    pub fn release(ends: &[TcpEnd], v: u32) -> Result<(), GpuError> {
        let sides: Vec<&dyn Side> = ends.iter().map(|e| &**e.end() as &dyn Side).collect();
        super::release_settled(&sides, v).map_err(|e| GpuError::protocol("the link's TCP end", e))
    }

    /// Why the end stopped carrying for good: a step that was neither 0 nor
    /// 1, a wire that ended (by its name), or a pump thread that ended.
    pub fn broken(&self) -> Option<String> {
        self.end()
            .broken()
            .or_else(|| self.pump.as_ref().and_then(Pump::ended))
    }

    /// The end's pump and the end itself, for a [`super::Link`] to hold.
    pub(super) fn into_parts(mut self) -> (Option<Pump>, Arc<dyn Side>) {
        let end = self
            .end
            .take()
            .expect("a TCP end holds its page until it is taken into a link");
        (self.pump.take(), end)
    }

    fn end(&self) -> &Arc<End<Tcp>> {
        self.end
            .as_ref()
            .expect("a TCP end holds its page until it is taken into a link")
    }
}

impl Drop for TcpEnd {
    fn drop(&mut self) {
        drop(self.pump.take());
        if let Some(end) = self.end.take() {
            super::drop_settled(&[end]);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::Ipv4Addr;
    use std::ops::RangeInclusive;
    use std::sync::atomic::AtomicU32;

    use super::super::pump::TAIL_LINE;
    use super::super::pump::tests::Mem;
    use super::super::{Mirror, Span};
    use super::*;
    use crate::host::step::RELEASE;

    // The model row: an image the go publishes, rows the done publishes, the
    // two counters and the done's progress word, then the tail.
    const SPAN: usize = 1024;
    const IMAGE: usize = 0;
    const ROWS: usize = SPAN;
    const GO: usize = 2 * SPAN;
    const CNT: usize = GO + 4;
    const PROG: usize = GO + 8;
    const TAIL: usize = GO + 64;
    const BYTES: usize = TAIL + 2 * TAIL_LINE;
    const NONCE: u64 = 0x5eed;
    const BOUND: Duration = Duration::from_secs(5);

    fn page() -> Arc<Mem> {
        Arc::new(Mem((0..BYTES / 4).map(|_| AtomicU32::new(0)).collect()))
    }

    fn row() -> Vec<Crossing> {
        vec![
            Crossing {
                counter: GO,
                reader: Reader::Tier,
                spans: vec![Span::whole(IMAGE..IMAGE + SPAN)],
                prog: None,
            },
            Crossing {
                counter: CNT,
                reader: Reader::Stage,
                spans: vec![Span::whole(ROWS..ROWS + SPAN)],
                prog: Some(PROG),
            },
        ]
    }

    fn hello() -> Hello {
        Hello {
            nonce: NONCE,
            bytes: BYTES as u64,
        }
    }

    /// The image and the generation of a go, as the stage's end posts them.
    fn go_post() -> [Move; 2] {
        [
            Move {
                from: IMAGE,
                to: IMAGE,
                len: SPAN,
            },
            Move {
                from: GO,
                to: TAIL,
                len: 4,
            },
        ]
    }

    /// The stage's wire over `stage`, accepted on loopback, and the stream of
    /// the tier's dial, its hello sent.
    fn stage_and_dial(stage: &Arc<Mem>) -> (Tcp<Mem>, TcpStream) {
        let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("a loopback listener");
        let at = TcpPeer {
            addr: l.local_addr().expect("the listener's address"),
            nonce: NONCE,
        };
        let dialed = dial(&at, hello()).expect("the tier's dial");
        let from = TcpPeer {
            addr: dialed.local_addr().expect("the dial's address"),
            nonce: NONCE,
        };
        let accepted = accept(&l, hello(), &from).expect("the stage's accept");
        let mut s = Tcp::new(Arc::clone(stage), 0);
        s.attach(accepted, None).expect("the stage's receiver");
        (s, dialed)
    }

    /// Two connected wires: the stage's over `stage`, the tier's over `tier`.
    fn pair(stage: &Arc<Mem>, tier: &Arc<Mem>) -> (Tcp<Mem>, Tcp<Mem>) {
        let (s, dialed) = stage_and_dial(stage);
        let mut t = Tcp::new(Arc::clone(tier), 0);
        t.attach(dialed, Some(hello()))
            .expect("the tier's receiver");
        (s, t)
    }

    /// Reap `w` until it fails, within [`BOUND`]: the failure.
    fn failed(w: &mut Tcp<Mem>) -> Option<String> {
        let until = Instant::now() + BOUND;
        while Instant::now() < until {
            if let Err(why) = w.reap() {
                return Some(why);
            }
            std::thread::yield_now();
        }
        None
    }

    /// The raw frames `heads`, each followed by `body` bytes of zeros.
    fn frames(heads: &[(Head, usize)]) -> Vec<u8> {
        let mut b = Vec::new();
        for &(h, body) in heads {
            b.extend_from_slice(&h.bytes());
            b.resize(b.len() + body, 0);
        }
        b
    }

    fn head(tag: u32, len: usize, at: usize, seq: u64) -> Head {
        Head {
            tag,
            len: u32::try_from(len).expect("a test frame's length fits"),
            at: at as u64,
            seq,
        }
    }

    // ------------------------------------------------ the model cards

    const ROWS_OF: u32 = 1 << 20;

    fn fill(p: &Mem, at: usize, v: u32) {
        for off in (at..at + SPAN).step_by(4) {
            p.word(off).store(v, Ordering::Relaxed);
        }
    }

    /// The first word of the span at `at` that is not `v`, as (byte, word).
    fn stray(p: &Mem, at: usize, v: u32) -> Option<(usize, u32)> {
        (at..at + SPAN)
            .step_by(4)
            .map(|off| (off, p.word(off).load(Ordering::Relaxed)))
            .find(|&(_, w)| w != v)
    }

    fn wait_take(p: &Mem, off: usize, until: Instant) -> bool {
        while p.word(off).load(Ordering::Acquire) < 1 {
            if Instant::now() > until {
                return false;
            }
            std::hint::spin_loop();
        }
        p.word(off).fetch_sub(1, Ordering::AcqRel);
        true
    }

    // Goes `ks` carry image k and dones rows k + ROWS_OF in every word of
    // the span; the bytes are checked for every go up to `checked`, the ones
    // before a release.
    fn tier_card(
        tier: Arc<Mem>,
        ks: RangeInclusive<u32>,
        checked: u32,
        until: Instant,
    ) -> JoinHandle<Result<(), String>> {
        std::thread::spawn(move || {
            for k in ks {
                if !wait_take(&tier, GO, until) {
                    return Err(format!("the tier's go {k} never came"));
                }
                if k <= checked
                    && let Some((off, img)) = stray(&tier, IMAGE, k)
                {
                    return Err(format!(
                        "the tier's go {k} carried the image of go {img} at byte {off}"
                    ));
                }
                fill(&tier, ROWS, k + ROWS_OF);
                tier.word(CNT).fetch_add(1, Ordering::Release);
                tier.word(PROG).fetch_add(1, Ordering::Release);
            }
            Ok(())
        })
    }

    fn stage_card(
        stage: &Mem,
        ks: RangeInclusive<u32>,
        checked: u32,
        until: Instant,
    ) -> Result<(), String> {
        for k in ks {
            fill(stage, IMAGE, k);
            stage.word(GO).fetch_add(1, Ordering::Release);
            if !wait_take(stage, CNT, until) {
                return Err(format!("the stage's done {k} never came"));
            }
            if k <= checked
                && let Some((off, rows)) = stray(stage, ROWS, k + ROWS_OF)
            {
                return Err(format!(
                    "the stage's done {k} carried rows {rows:#x} at byte {off}"
                ));
            }
        }
        Ok(())
    }

    // ------------------------------------------------ the clauses

    // The ordering clause. Two model cards and the link's two ends over one
    // loopback TCP stream, each end on its own pump thread: every go carries
    // its image and every done its rows, so each post's bytes land before its
    // generation. Then the link's release, K more layers carried by nothing,
    // the resets of the go and of the done (each drained by its acks), and N
    // more goes and dones as before. Both counters end at rest, the
    // progress counts every done, and no end broke.
    #[test]
    fn the_ends_carry_every_go_and_done_in_order_over_tcp() {
        const N: u32 = 2000;
        const K: u32 = 3;
        let (stage, tier) = (page(), page());
        let (sw, tw) = pair(&stage, &tier);
        let half = |side, p: &Arc<Mem>| {
            Half::new(row(), side, TAIL, &**p, PostCols::default())
                .expect("the model row fits before the tail")
        };
        let mut m = Mirror {
            pumps: Vec::new(),
            stage: Arc::new(End::new(
                half(Reader::Stage, &stage),
                sw,
                Arc::clone(&stage),
            )),
            tier: Arc::new(End::new(half(Reader::Tier, &tier), tw, Arc::clone(&tier))),
            run_on: None,
        };
        m.pumps = [&m.stage, &m.tier]
            .map(|e| Pump::start(vec![Arc::clone(e)]).expect("a pump thread"))
            .into();
        let until = Instant::now() + Duration::from_secs(20);
        let run = || -> Result<(), String> {
            let card = tier_card(tier.clone(), 1..=N + K, N, until);
            stage_card(&stage, 1..=N, N, until)?;
            m.release(RELEASE)?;
            stage_card(&stage, N + 1..=N + K, N, until)?;
            card.join().expect("the tier card's thread")?;
            m.reset(GO)?;
            m.reset(CNT)?;
            let card = tier_card(tier.clone(), N + K + 1..=2 * N + K, 2 * N + K, until);
            stage_card(&stage, N + K + 1..=2 * N + K, 2 * N + K, until)?;
            card.join().expect("the tier card's thread")
        };
        let ran = run();
        m.pumps.clear();
        assert_eq!(ran, Ok(()), "the ends: {:?}", m.broken());
        assert_eq!(m.broken(), None, "no end broke");
        assert!(
            m.at_rest(GO) && m.at_rest(CNT),
            "both counters rest once every post is acked"
        );
        assert_eq!(
            stage.word(PROG).load(Ordering::Acquire),
            2 * N + K,
            "the progress counts every done delivered"
        );
    }

    // A frame whose sequence number is not the next one ends the wire by
    // name at the next reap.
    #[test]
    fn a_frame_out_of_sequence_ends_the_wire_by_name() {
        let stage = page();
        let (mut s, mut dialed) = stage_and_dial(&stage);
        let raw = frames(&[(head(MOVE, 4, IMAGE, 2), 4), (head(LAST, 4, TAIL, 1), 4)]);
        dialed.write_all(&raw).expect("the raw frames");
        assert_eq!(
            failed(&mut s).as_deref(),
            Some("a frame out of sequence: frame 2 where frame 1 belongs"),
            "the receiver checks each frame's sequence number before it lands"
        );
        assert_eq!(
            s.reap(),
            Ok(true),
            "the reap that named why ended every post"
        );
    }

    // The abort clause: the receiver thread's peer goes (its end drops, as
    // a killed process's stream closes), and the next reap names why; a later
    // post fails with the same name, and the reap after that finds no post in
    // flight.
    #[test]
    fn a_closed_peer_ends_the_wire_at_the_next_reap_by_name() {
        let (stage, tier) = (page(), page());
        let (mut s, t) = pair(&stage, &tier);
        stage.word(GO).store(7, Ordering::Release);
        s.post(&go_post()).expect("a post on a connected wire");
        let until = Instant::now() + BOUND;
        while !s.reap().expect("a carrying wire") {
            assert!(Instant::now() < until, "the post was never acked");
            std::thread::yield_now();
        }
        assert_eq!(
            tier.word(TAIL).load(Ordering::Acquire),
            7,
            "an ack follows its post's landing: the generation is in the tier's inbox"
        );
        drop(t);
        let closed = "the other end closed the link";
        assert_eq!(
            failed(&mut s).as_deref(),
            Some(closed),
            "the receiver names the closed peer"
        );
        assert_eq!(s.post(&go_post()), Err(closed.to_string()));
        assert_eq!(
            s.reap(),
            Ok(true),
            "the reap that named why ended every post"
        );
    }

    // An end that refuses what it reads tells the other end why: the tier's
    // end refuses a frame past its page, and the stage's end, told, ends by
    // the tier's cause.
    #[test]
    fn the_end_that_refuses_a_frame_tells_the_other_end_why() {
        let (stage, tier) = (page(), page());
        let (mut s, mut t) = pair(&stage, &tier);
        let past = Move {
            from: IMAGE,
            to: BYTES,
            len: 4,
        };
        s.post(&[past]).expect("a post on a connected wire");
        let why = format!(
            "frame 1: 4 bytes at byte {BYTES}, not whole words inside the page's {BYTES} bytes"
        );
        assert_eq!(
            failed(&mut t),
            Some(why.clone()),
            "the tier's end refuses it"
        );
        assert_eq!(
            failed(&mut s),
            Some(format!("the other end ended the link: {why}")),
            "the stage's end names the tier's cause"
        );
    }

    // An abort ends the wire by its own name: the next reap returns it
    // once, its unacknowledged posts finished, failed, and a later post fails
    // with it.
    #[test]
    fn an_abort_ends_the_wire_by_name() {
        let (stage, tier) = (page(), page());
        let (mut s, _t) = pair(&stage, &tier);
        s.post(&go_post()).expect("a post on a connected wire");
        s.abort().expect("an abort");
        let aborted = "the link's TCP wire was aborted";
        assert_eq!(s.reap(), Err(aborted.to_string()));
        assert_eq!(s.reap(), Ok(true), "the aborted posts finished, failed");
        assert_eq!(s.post(&go_post()), Err(aborted.to_string()));
    }

    // A frame that does not fit the page is refused by name before any of
    // its bytes land.
    #[test]
    fn a_frame_past_the_page_is_refused_by_name() {
        let stage = page();
        let (mut s, mut dialed) = stage_and_dial(&stage);
        let at = BYTES - 32;
        dialed
            .write_all(&frames(&[(head(LAST, 64, at, 1), 64)]))
            .expect("the raw frame");
        assert_eq!(
            failed(&mut s),
            Some(format!(
                "frame 1: 64 bytes at byte {at}, not whole words inside the page's {BYTES} bytes"
            )),
            "the bound is the receiver's own page"
        );
        let (mut s, mut dialed) = stage_and_dial(&stage);
        dialed
            .write_all(&frames(&[(head(MOVE, 6, 0, 1), 6)]))
            .expect("the raw frame");
        assert_eq!(
            failed(&mut s),
            Some(format!(
                "frame 1: 6 bytes at byte 0, not whole words inside the page's {BYTES} bytes"
            )),
            "a frame of part of a word is refused"
        );
    }

    // A stream that ends inside a frame is a short read, by name.
    #[test]
    fn a_short_read_ends_the_wire_by_name() {
        let stage = page();
        let (mut s, mut dialed) = stage_and_dial(&stage);
        let mut raw = frames(&[(head(MOVE, 64, IMAGE, 1), 0)]);
        raw.extend_from_slice(&[0; 8]);
        dialed.write_all(&raw).expect("the raw frame");
        dialed
            .shutdown(Shutdown::Write)
            .expect("the dial's write side closed");
        let why = failed(&mut s).expect("a short read ends the wire");
        assert!(
            why.starts_with("a short read: frame 1 of 64 bytes ended between bytes 0 and 64"),
            "{why}"
        );
    }

    // An ack of a frame never posted ends the wire by name.
    #[test]
    fn an_ack_of_a_frame_never_posted_ends_the_wire_by_name() {
        let stage = page();
        let (mut s, mut dialed) = stage_and_dial(&stage);
        dialed
            .write_all(&frames(&[(head(ACK, 0, 0, 5), 0)]))
            .expect("the raw ack");
        assert_eq!(
            failed(&mut s).as_deref(),
            Some("an ack of frame 5 out of sequence: frames up to 0 acked, 0 posted"),
        );
    }

    // The stage's end takes only a hello with its own nonce and page size,
    // and tells the tier's end why it refused one.
    #[test]
    fn a_hello_of_another_link_is_refused_by_name() {
        for (other, why) in [
            (
                Hello {
                    nonce: NONCE + 1,
                    bytes: BYTES as u64,
                },
                "a hello with another nonce: not this link's other end".to_string(),
            ),
            (
                Hello {
                    nonce: NONCE,
                    bytes: 64,
                },
                format!("a hello for a page of 64 bytes, where this end's page holds {BYTES}"),
            ),
        ] {
            let l = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("a loopback listener");
            let at = TcpPeer {
                addr: l.local_addr().expect("the listener's address"),
                nonce: other.nonce,
            };
            let dialed = dial(&at, other).expect("the dial");
            let from = dialed.local_addr().expect("the dial's address");
            let refused = accept(
                &l,
                hello(),
                &TcpPeer {
                    addr: from,
                    nonce: NONCE,
                },
            )
            .expect_err("a hello of another link");
            assert_eq!(refused, format!("the tier's end from {from}: {why}"));
            let mut t = Tcp::new(page(), 0);
            t.attach(dialed, Some(other)).expect("the tier's receiver");
            assert_eq!(
                failed(&mut t),
                Some(format!("the other end ended the link: {why}")),
                "the stage tells the tier's end why it refused its hello"
            );
        }
    }
}
