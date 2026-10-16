//! The tier node channel: frames of a `u32` length, a `u8` tag and a payload
//! of little-endian integers and length-prefixed strings ([`Out`], [`In`]),
//! over a TCP stream whose every read and write is bounded ([`Channel`]).
//! Every request gets one reply, [`Tag::Ok`] with its payload or [`Tag::Err`]
//! with the server's error; the census goes from the server unasked, once,
//! when a connection opens.

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::time::{Duration, Instant};

/// The protocol's version, the census's first word: a node of another
/// version is refused by name.
pub(super) const VERSION: u32 = 4;

/// The longest frame either side takes: an open's rows are the largest, a
/// few hundred kilobytes.
pub const MAX_FRAME: usize = 1 << 26;

/// How long one read or write of the stream blocks: a wait looks at its
/// deadline, and the server at its stage side's progress, this often.
const SLICE: Duration = Duration::from_millis(50);

/// A frame's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    Census = 1,
    Open = 2,
    Bind = 3,
    Launch = 4,
    Capture = 5,
    Enqueue = 6,
    Drain = 8,
    ClearFault = 9,
    Fault = 10,
    Close = 11,
    Reset = 12,
    Release = 13,
    Ok = 100,
    Err = 101,
}

impl Tag {
    fn of(b: u8) -> Option<Tag> {
        [
            Tag::Census,
            Tag::Open,
            Tag::Bind,
            Tag::Launch,
            Tag::Capture,
            Tag::Enqueue,
            Tag::Drain,
            Tag::ClearFault,
            Tag::Fault,
            Tag::Close,
            Tag::Reset,
            Tag::Release,
            Tag::Ok,
            Tag::Err,
        ]
        .into_iter()
        .find(|&t| t as u8 == b)
    }
}

/// A payload being written.
#[derive(Default)]
pub struct Out(Vec<u8>);

impl Out {
    /// Empty again, its buffer kept for the next payload.
    pub fn clear(&mut self) -> &mut Out {
        self.0.clear();
        self
    }

    pub fn u8(&mut self, v: u8) -> &mut Out {
        self.0.push(v);
        self
    }

    pub fn u32(&mut self, v: u32) -> &mut Out {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Out {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    /// A `usize` as a `u64`.
    pub fn len(&mut self, v: usize) -> &mut Out {
        self.u64(v as u64)
    }

    pub fn bytes(&mut self, v: &[u8]) -> &mut Out {
        self.len(v.len());
        self.0.extend_from_slice(v);
        self
    }

    pub fn str(&mut self, v: &str) -> &mut Out {
        self.bytes(v.as_bytes())
    }

    pub fn u32s(&mut self, v: &[u32]) -> &mut Out {
        self.len(v.len());
        for &x in v {
            self.u32(x);
        }
        self
    }

    pub fn u64s(&mut self, v: &[u64]) -> &mut Out {
        self.len(v.len());
        for &x in v {
            self.u64(x);
        }
        self
    }

    /// The payload written.
    #[must_use]
    pub fn done(&self) -> &[u8] {
        &self.0
    }
}

/// A payload being read: each read past its end, and a payload with bytes
/// left at [`In::end`], is an error naming what was read.
pub struct In<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> In<'a> {
    #[must_use]
    pub fn new(buf: &'a [u8]) -> In<'a> {
        In { buf, at: 0 }
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.buf.len());
        let Some(end) = end else {
            return Err(format!(
                "a payload of {} bytes ends inside its {what} at byte {}",
                self.buf.len(),
                self.at
            ));
        };
        let s = &self.buf[self.at..end];
        self.at = end;
        Ok(s)
    }

    pub fn u8(&mut self, what: &str) -> Result<u8, String> {
        Ok(self.take(1, what)?[0])
    }

    /// A flag byte: 0 or 1, any other refused.
    pub fn flag(&mut self, what: &str) -> Result<bool, String> {
        match self.u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            b => Err(format!("its {what} is {b}, a flag 0 or 1")),
        }
    }

    pub fn u32(&mut self, what: &str) -> Result<u32, String> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&mut self, what: &str) -> Result<u64, String> {
        let b = self.take(8, what)?;
        let mut w = [0u8; 8];
        w.copy_from_slice(b);
        Ok(u64::from_le_bytes(w))
    }

    pub fn len(&mut self, what: &str) -> Result<usize, String> {
        let v = self.u64(what)?;
        usize::try_from(v).map_err(|_| format!("its {what}, {v}, passes usize"))
    }

    pub fn bytes(&mut self, what: &str) -> Result<&'a [u8], String> {
        let n = self.len(what)?;
        self.take(n, what)
    }

    pub fn str(&mut self, what: &str) -> Result<String, String> {
        let b = self.bytes(what)?;
        String::from_utf8(b.to_vec()).map_err(|_| format!("its {what} is not UTF-8"))
    }

    /// A list of `u32`, its length checked against the payload first.
    pub fn u32s(&mut self, what: &str) -> Result<Vec<u32>, String> {
        let n = self.len(what)?;
        if n > (self.buf.len() - self.at) / 4 {
            return Err(format!("its {what} of {n} words passes the payload"));
        }
        (0..n).map(|_| self.u32(what)).collect()
    }

    pub fn u64s(&mut self, what: &str) -> Result<Vec<u64>, String> {
        let n = self.len(what)?;
        if n > (self.buf.len() - self.at) / 8 {
            return Err(format!("its {what} of {n} words passes the payload"));
        }
        (0..n).map(|_| self.u64(what)).collect()
    }

    /// The payload read to its end, else an error.
    pub fn end(&self) -> Result<(), String> {
        if self.at == self.buf.len() {
            Ok(())
        } else {
            Err(format!(
                "a payload of {} bytes holds {} more past what its message reads",
                self.buf.len(),
                self.buf.len() - self.at
            ))
        }
    }
}

/// What one [`Channel::poll`] found.
pub(super) enum Polled {
    /// A whole frame of this tag; its payload is [`Channel::body`].
    Frame(Tag),
    /// No whole frame within a slice.
    Waiting,
    /// The peer ended the stream between frames.
    Closed,
}

/// One end of a tier node's channel: its stream, each read and write
/// blocking at most a slice, so every wait checks a deadline; the frame
/// being read and the one being written, their buffers reused.
pub(super) struct Channel {
    s: TcpStream,
    /// The frame being read: its first `have` bytes, and when the first came.
    inbuf: Vec<u8>,
    have: usize,
    started: Option<Instant>,
    /// Whether `inbuf` holds a whole frame the caller has not passed yet.
    ready: bool,
    outbuf: Vec<u8>,
}

/// Whether `e` is a read or write that ran out its slice.
fn slice_over(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    )
}

/// The error of a wait `what` past `deadline`.
fn late(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        format!("{what} passed its deadline"),
    )
}

impl Channel {
    pub(super) fn local_ip(&self) -> io::Result<std::net::IpAddr> {
        self.s.local_addr().map(|a| a.ip())
    }

    pub(super) fn new(s: TcpStream) -> io::Result<Channel> {
        s.set_nodelay(true)?;
        s.set_read_timeout(Some(SLICE))?;
        s.set_write_timeout(Some(SLICE))?;
        Ok(Channel {
            s,
            inbuf: Vec::new(),
            have: 0,
            started: None,
            ready: false,
            outbuf: Vec::new(),
        })
    }

    /// Send one frame, whole by `deadline`.
    pub(super) fn send(&mut self, tag: Tag, body: &[u8], deadline: Instant) -> io::Result<()> {
        let len = u32::try_from(body.len() + 1)
            .ok()
            .filter(|&n| n as usize <= MAX_FRAME)
            .ok_or_else(|| io::Error::other(format!("a frame of {} bytes", body.len())))?;
        self.outbuf.clear();
        self.outbuf.extend_from_slice(&len.to_le_bytes());
        self.outbuf.push(tag as u8);
        self.outbuf.extend_from_slice(body);
        let mut at = 0;
        while at < self.outbuf.len() {
            match self.s.write(&self.outbuf[at..]) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => at += n,
                Err(e) if slice_over(&e) || e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
            if Instant::now() > deadline {
                return Err(late("a frame's write"));
            }
        }
        Ok(())
    }

    /// Read on for a slice: a whole frame, the stream ended between frames,
    /// or neither yet. A frame past [`MAX_FRAME`] or of a tag the protocol
    /// does not name is an error, and so is a stream ended inside a frame.
    pub(super) fn poll(&mut self) -> io::Result<Polled> {
        if self.ready {
            (self.have, self.started, self.ready) = (0, None, false);
        }
        let t0 = Instant::now();
        loop {
            let want = if self.have < 4 {
                4
            } else {
                let len = u32::from_le_bytes([
                    self.inbuf[0],
                    self.inbuf[1],
                    self.inbuf[2],
                    self.inbuf[3],
                ]) as usize;
                if len == 0 || len > MAX_FRAME {
                    return Err(io::Error::other(format!("a frame of {len} bytes")));
                }
                4 + len
            };
            if self.have == want && want > 4 {
                let tag = Tag::of(self.inbuf[4])
                    .ok_or_else(|| io::Error::other(format!("a frame tagged {}", self.inbuf[4])))?;
                self.ready = true;
                return Ok(Polled::Frame(tag));
            }
            if self.inbuf.len() < want {
                self.inbuf.resize(want, 0);
            }
            match self.s.read(&mut self.inbuf[self.have..want]) {
                Ok(0) if self.have == 0 => return Ok(Polled::Closed),
                Ok(0) => return Err(io::ErrorKind::UnexpectedEof.into()),
                Ok(n) => {
                    self.started.get_or_insert_with(Instant::now);
                    self.have += n;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) if slice_over(&e) => return Ok(Polled::Waiting),
                Err(e) => return Err(e),
            }
            if t0.elapsed() > SLICE {
                return Ok(Polled::Waiting);
            }
        }
    }

    /// End the stream both ways: the peer's next read finds it ended.
    pub(super) fn shutdown(&self) {
        drop(self.s.shutdown(Shutdown::Both));
    }

    /// The payload of the frame the last [`Channel::poll`] found.
    pub(super) fn body(&self) -> &[u8] {
        &self.inbuf[5..self.have]
    }

    /// When the frame being read began, while one is part read.
    pub(super) fn started(&self) -> Option<Instant> {
        self.started.filter(|_| !self.ready)
    }

    /// The next frame, whole by `deadline`; a stream that ends is an error.
    /// The read is late only when a slice that began past the deadline found
    /// no frame, so a frame that came while this process was stopped is
    /// taken when it runs again.
    pub(super) fn recv(&mut self, deadline: Instant) -> io::Result<Tag> {
        loop {
            let past = Instant::now() > deadline;
            match self.poll()? {
                Polled::Frame(t) => return Ok(t),
                Polled::Closed => return Err(io::ErrorKind::UnexpectedEof.into()),
                Polled::Waiting if past => return Err(late("a frame's read")),
                Polled::Waiting => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Channel, In, MAX_FRAME, Out, Polled, Tag};
    use std::io::{ErrorKind, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::{Duration, Instant};

    /// What [`Out`] writes, [`In`] reads back, in order; a read past the
    /// payload, a list longer than the payload, a flag other than 0 or 1
    /// and bytes left over are refused.
    #[test]
    fn a_payload_reads_back_what_was_written() {
        let mut o = Out::default();
        o.u8(7).u32(0xdead_beef).u64(1 << 40).str("qwen4exp");
        o.u32s(&[1, 2, 3]).u64s(&[9, 1 << 33]).u8(1);
        let mut i = In::new(o.done());
        assert_eq!(i.u8("a").unwrap(), 7);
        assert_eq!(i.u32("b").unwrap(), 0xdead_beef);
        assert_eq!(i.u64("c").unwrap(), 1 << 40);
        assert_eq!(i.str("d").unwrap(), "qwen4exp");
        assert_eq!(i.u32s("e").unwrap(), [1, 2, 3]);
        assert_eq!(i.u64s("f").unwrap(), [9, 1 << 33]);
        assert!(i.flag("g").unwrap());
        assert!(i.end().is_ok());
        assert!(i.u8("past").is_err());
        assert!(In::new(&[2]).flag("flag").is_err());
        let mut o = Out::default();
        o.len(1 << 40);
        assert!(In::new(o.done()).u32s("long").is_err());
        let mut o = Out::default();
        o.u32(1).u8(0);
        let mut i = In::new(o.done());
        i.u32("x").unwrap();
        assert!(i.end().is_err());
    }

    /// Over loopback, with no card: a frame written in two parts reads back
    /// whole, the poll between them waiting; a frame already in is taken
    /// past its deadline, and none by then is late; the stream ended between
    /// frames is closed, inside one an error; a length of [`MAX_FRAME`]
    /// waits for its bytes, and one past it is refused; a send past its
    /// deadline is late.
    #[test]
    fn a_channel_reads_whole_frames_and_its_ends() {
        let l = TcpListener::bind("127.0.0.1:0").expect("a loopback port");
        let pair = || {
            let w = TcpStream::connect(l.local_addr().expect("its address")).expect("a connect");
            let ch = Channel::new(l.accept().expect("an accept").0).expect("a channel");
            (w, ch)
        };
        let head = |len: usize, tag: Tag| {
            let mut f = u32::try_from(len).expect("a u32").to_le_bytes().to_vec();
            f.push(tag as u8);
            f
        };

        let (mut w, mut ch) = pair();
        let mut f = head(11, Tag::Bind);
        f.extend_from_slice(b"0123456789");
        w.write_all(&f[..7]).expect("a write");
        assert!(matches!(ch.poll().expect("a poll"), Polled::Waiting));
        w.write_all(&f[7..]).expect("a write");
        assert_eq!(
            ch.recv(Instant::now() + Duration::from_secs(5))
                .expect("a frame"),
            Tag::Bind
        );
        assert_eq!(ch.body(), b"0123456789");
        w.write_all(&head(1, Tag::ClearFault)).expect("a write");
        assert_eq!(ch.recv(Instant::now()).expect("a frame"), Tag::ClearFault);
        assert_eq!(
            ch.recv(Instant::now()).unwrap_err().kind(),
            ErrorKind::TimedOut
        );
        drop(w);
        assert!(matches!(ch.poll().expect("a poll"), Polled::Closed));

        let (mut w, mut ch) = pair();
        w.write_all(&head(0, Tag::Open)).expect("a write");
        let error = ch.poll().err().expect("an empty frame is refused");
        assert!(error.to_string().contains("a frame of 0 bytes"), "{error}");

        let (mut w, mut ch) = pair();
        w.write_all(&head(MAX_FRAME, Tag::Open)).expect("a write");
        assert!(matches!(ch.poll().expect("a poll"), Polled::Waiting));
        drop(w);
        let ended = ch.poll().err().map(|e| e.kind());
        assert_eq!(ended, Some(ErrorKind::UnexpectedEof));

        let (mut w, mut ch) = pair();
        w.write_all(&head(MAX_FRAME + 1, Tag::Open))
            .expect("a write");
        assert!(ch.poll().is_err());

        let (_w, mut ch) = pair();
        assert_eq!(
            ch.send(Tag::Close, &[], Instant::now() - Duration::from_secs(1))
                .unwrap_err()
                .kind(),
            ErrorKind::TimedOut
        );
    }
}
