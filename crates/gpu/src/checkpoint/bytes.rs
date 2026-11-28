//! The byte form a host state is written in and read back from, what a slot
//! file carries after the server's header: little-endian values through one
//! buffer of [`CHUNK`] bytes each way, a count a u64, a name a u32 length
//! and its bytes. Each owner of a state lays out its own parts with
//! [`Put`] and reads them back with [`Take`] in the same order
//! ([`super::saved::SeqState::put`], the DeepSeek V4.1 body's snapshot).
//!
//! A reader takes its counts from the stream it reads, so no count sizes an
//! allocation before the bytes it covers have arrived: [`Take::vals`] grows
//! its array a chunk at a time, and a stream that ends early is refused by
//! name with the part it ends in.
//!
//! A part that lives on the card moves through one staging buffer of at most
//! [`CARD_CHUNK`] bytes, whatever its length: [`Put::from_card`] reads a
//! chunk and writes it, [`Take::to_card`] reads a chunk and hands it on, so
//! the host holds one chunk of a state, never the state. The bytes are those
//! of [`Put::vals`] and [`Take::vals`] over the whole part.

use std::fmt;
use std::io::{self, Read, Write};
use std::ops::Range;

use crate::GpuError;

const WHAT: &str = "state bytes";

/// Bytes of the write and read buffer.
pub const CHUNK: usize = 1 << 16;

/// Bytes of the staging buffer a part of the card moves through
/// ([`Put::from_card`], [`Take::to_card`]): the most a copy holds of it.
pub const CARD_CHUNK: usize = 1 << 20;

/// The longest name the form carries: a path's limit.
const NAME_MAX: usize = 4096;

/// Why a write or a read of the byte form failed.
#[derive(Debug)]
pub enum Fail {
    /// Refused by name ([`GpuError::Shape`]): a stream that is not a state
    /// this reader takes (another tag, format, model or layout, a short
    /// stream, trailing bytes, a count out of range), or a state that cannot
    /// be written as it stands. [`Take::to_card`] hands each chunk on before
    /// it reads the next, so a stream that ends inside such a part is refused
    /// after the chunks before it reached the card; every other refusal
    /// touched nothing on the device.
    Refused(GpuError),
    /// The stream itself failed: a write or a read it refused (a closed
    /// pipe, a full disk), neither the request's form nor the model's.
    Io(io::Error),
    /// The card failed during a copy of a part ([`Put::from_card`],
    /// [`Take::to_card`]): the model's error, neither the request's form nor
    /// the stream's.
    Card(GpuError),
    /// A restore that failed as `restore` says, and then the reset that was
    /// to leave the model empty after it failed too (`reset`): what the
    /// model holds is unknown. The restore's own failure is kept whole; the
    /// failure is the card's whatever kind `restore` is.
    Reset { restore: Box<Fail>, reset: GpuError },
}

impl Fail {
    /// `e`, the error of a copy of a part on the card, as this form's: a
    /// refusal by name ([`GpuError::Shape`], [`GpuError::State`]) is the
    /// request's, every other error the card's.
    #[must_use]
    pub fn card(e: GpuError) -> Fail {
        match e {
            GpuError::Shape { .. } | GpuError::State { .. } => Fail::Refused(e),
            e => Fail::Card(e),
        }
    }

    /// This failure of a restore, after `reset` ran to leave the model
    /// empty: itself when the reset succeeded, else [`Fail::Reset`] naming
    /// both, so a failing reset never replaces the failure that called for
    /// it.
    #[must_use]
    pub fn with_reset(self, reset: Result<(), GpuError>) -> Fail {
        match reset {
            Ok(()) => self,
            Err(reset) => Fail::Reset {
                restore: Box::new(self),
                reset,
            },
        }
    }
}

impl fmt::Display for Fail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fail::Refused(e) => write!(f, "refused: {e}"),
            Fail::Io(e) => write!(f, "the stream failed: {e}"),
            Fail::Card(e) => write!(f, "the card failed: {e}"),
            Fail::Reset { restore, reset } => write!(
                f,
                "{restore}; and the model's reset after it failed too: {reset}"
            ),
        }
    }
}

impl std::error::Error for Fail {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Fail::Refused(e) | Fail::Card(e) => Some(e),
            Fail::Io(e) => Some(e),
            Fail::Reset { reset, .. } => Some(reset),
        }
    }
}

/// A refusal by name of the byte form ([`Fail::Refused`]).
#[must_use]
pub fn refused(detail: impl Into<String>) -> Fail {
    Fail::Refused(GpuError::shape(WHAT, detail))
}

/// A value the byte form carries, little-endian.
pub trait Le: Copy {
    const SIZE: usize;
    fn put(self, out: &mut Vec<u8>);
    /// The value of `b`, [`Le::SIZE`] bytes.
    fn get(b: &[u8]) -> Self;
}

macro_rules! le {
    ($($t:ty),*) => {$(
        impl Le for $t {
            const SIZE: usize = size_of::<$t>();
            fn put(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn get(b: &[u8]) -> $t {
                let mut a = [0u8; size_of::<$t>()];
                a.copy_from_slice(b);
                <$t>::from_le_bytes(a)
            }
        }
    )*};
}

le!(u8, u16, u32, u64, f32);

/// A buffer of `len` values for a chunk of a part on the card, refused by
/// name when the host has no memory for it.
fn staging<T: Le + Default>(len: usize) -> Result<Vec<T>, Fail> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|e| refused(format!("{len} values of host memory for a card copy: {e}")))?;
    v.resize(len, T::default());
    Ok(v)
}

/// The values of `T` a card chunk of `card` bytes holds. A chunk that holds
/// none is a caller's undefined input, never rounded up to one value.
///
/// # Panics
/// When `card` is smaller than one value of `T`.
fn per_chunk<T: Le>(card: usize) -> usize {
    let per = card / T::SIZE;
    assert!(
        per > 0,
        "a card chunk of {card} bytes holds no whole {}-byte value",
        T::SIZE
    );
    per
}

/// The writer: `out` through a buffer of at most [`CHUNK`] bytes, the bytes
/// written counted. An error of `out` is the stream's ([`Fail::Io`]).
pub struct Put<'a> {
    out: &'a mut dyn Write,
    buf: Vec<u8>,
    bytes: u64,
    card: usize,
}

impl<'a> Put<'a> {
    pub fn new(out: &'a mut dyn Write) -> Put<'a> {
        Put {
            out,
            buf: Vec::with_capacity(CHUNK),
            bytes: 0,
            card: CARD_CHUNK,
        }
    }

    /// The writer with a card chunk of `card` bytes ([`Put::from_card`]) in
    /// place of [`CARD_CHUNK`]: what a gate sets to run a part over many
    /// chunk boundaries.
    ///
    /// # Panics
    /// When `card` is 0, and in [`Put::from_card`] when it is smaller than
    /// one value of the part's type.
    #[must_use]
    pub fn with_card_chunk(mut self, card: usize) -> Put<'a> {
        assert!(card > 0, "a card chunk of 0 bytes holds nothing");
        self.card = card;
        self
    }

    /// The bytes written to `out` so far: every one once [`Put::flush`] ran.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn one<T: Le>(&mut self, v: T) -> Result<(), Fail> {
        self.vals(&[v])
    }

    pub fn vals<T: Le>(&mut self, vals: &[T]) -> Result<(), Fail> {
        for c in vals.chunks(CHUNK / T::SIZE) {
            if self.buf.len() + c.len() * T::SIZE > CHUNK {
                self.flush()?;
            }
            for &v in c {
                v.put(&mut self.buf);
            }
        }
        Ok(())
    }

    /// `len` values of a part on the card, as [`Put::vals`] writes them: the
    /// part read a chunk at a time through one buffer of at most
    /// [`CARD_CHUNK`] bytes, `read` called for the values `at..at + n` with
    /// the buffer's first `n` to fill, in order, each range at most the
    /// buffer's and none empty. A chunk is written before the next is read,
    /// so the host holds one chunk of the part. An error of `read` is the
    /// card's ([`Fail::card`]); a buffer that cannot be had is refused by
    /// name.
    ///
    /// # Panics
    /// When the writer's card chunk is smaller than one value of `T`.
    pub fn from_card<T: Le + Default>(
        &mut self,
        len: usize,
        mut read: impl FnMut(Range<usize>, &mut [T]) -> Result<(), GpuError>,
    ) -> Result<(), Fail> {
        let per = per_chunk::<T>(self.card);
        let mut stage = staging::<T>(len.min(per))?;
        let mut at = 0;
        while at < len {
            let n = per.min(len - at);
            read(at..at + n, &mut stage[..n]).map_err(Fail::card)?;
            self.vals(&stage[..n])?;
            at += n;
        }
        Ok(())
    }

    pub fn count(&mut self, v: usize) -> Result<(), Fail> {
        self.one(v as u64)
    }

    /// `vals`' count, then its values.
    pub fn array<T: Le>(&mut self, vals: &[T]) -> Result<(), Fail> {
        self.count(vals.len())?;
        self.vals(vals)
    }

    /// `b` as a name: its length (u32), then its bytes; refused by name past
    /// [`NAME_MAX`].
    pub fn name(&mut self, b: &[u8]) -> Result<(), Fail> {
        let len = u32::try_from(b.len())
            .ok()
            .filter(|&l| l as usize <= NAME_MAX)
            .ok_or_else(|| {
                refused(format!(
                    "a name of {} bytes, past the {NAME_MAX} the form carries",
                    b.len()
                ))
            })?;
        self.one(len)?;
        self.vals(b)
    }

    /// `name`, one of `names`, as one past its place there (u8), none as 0;
    /// refused by name for a name `names` does not hold.
    pub fn named(
        &mut self,
        name: Option<&str>,
        names: &[&'static str],
        part: &str,
    ) -> Result<(), Fail> {
        let code = match name {
            None => 0,
            Some(n) => names
                .iter()
                .position(|&m| m == n)
                .and_then(|i| u8::try_from(i + 1).ok())
                .ok_or_else(|| refused(format!("{part} the form names no code for: {n}")))?,
        };
        self.one(code)
    }

    pub fn flush(&mut self) -> Result<(), Fail> {
        self.out.write_all(&self.buf).map_err(Fail::Io)?;
        self.bytes += self.buf.len() as u64;
        self.buf.clear();
        Ok(())
    }
}

/// The reader: `input` through a buffer of at most [`CHUNK`] bytes, the
/// bytes read counted. A stream that ends early is refused by name, with the
/// part it ends in; any other error of `input` is the stream's
/// ([`Fail::Io`]).
pub struct Take<'a> {
    input: &'a mut dyn Read,
    buf: Vec<u8>,
    bytes: u64,
    card: usize,
}

impl<'a> Take<'a> {
    pub fn new(input: &'a mut dyn Read) -> Take<'a> {
        Take {
            input,
            buf: Vec::with_capacity(CHUNK),
            bytes: 0,
            card: CARD_CHUNK,
        }
    }

    /// The reader with a card chunk of `card` bytes ([`Take::to_card`]) in
    /// place of [`CARD_CHUNK`].
    ///
    /// # Panics
    /// When `card` is 0, and in [`Take::to_card`] when it is smaller than
    /// one value of the part's type.
    #[must_use]
    pub fn with_card_chunk(mut self, card: usize) -> Take<'a> {
        assert!(card > 0, "a card chunk of 0 bytes holds nothing");
        self.card = card;
        self
    }

    /// The bytes read from `input` so far.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The next `len` bytes, `len` at most [`CHUNK`].
    fn exact(&mut self, len: usize, part: &str) -> Result<&[u8], Fail> {
        self.buf.resize(len, 0);
        self.input
            .read_exact(&mut self.buf)
            .map_err(|e| match e.kind() {
                io::ErrorKind::UnexpectedEof => refused(format!(
                    "a short stream: it ends inside {part}, past byte {}",
                    self.bytes
                )),
                _ => Fail::Io(e),
            })?;
        self.bytes += len as u64;
        Ok(&self.buf)
    }

    pub fn one<T: Le>(&mut self, part: &str) -> Result<T, Fail> {
        Ok(T::get(self.exact(T::SIZE, part)?))
    }

    pub fn count(&mut self, part: &str) -> Result<usize, Fail> {
        let v: u64 = self.one(part)?;
        usize::try_from(v).map_err(|_| refused(format!("{part}: {v} passes usize")))
    }

    /// `len` values, the array grown a chunk at a time as they arrive.
    pub fn vals<T: Le>(&mut self, len: usize, part: &str) -> Result<Vec<T>, Fail> {
        let mut v = Vec::new();
        let mut left = len;
        while left > 0 {
            let n = left.min(CHUNK / T::SIZE);
            let b = self.exact(n * T::SIZE, part)?;
            v.try_reserve(n)
                .map_err(|e| refused(format!("{part}: {len} values of host memory: {e}")))?;
            v.extend(b.chunks_exact(T::SIZE).map(T::get));
            left -= n;
        }
        Ok(v)
    }

    /// `len` values of a part that goes to the card, which [`Put::vals`] or
    /// [`Put::from_card`] wrote: read a chunk at a time into one buffer of at
    /// most [`CARD_CHUNK`] bytes and handed to `write` as the values
    /// `at..at + n`, in order, each range at most the buffer's and none
    /// empty. A stream that ends inside the part is refused by name after
    /// the chunks before it were handed on; an error of `write` is the
    /// card's ([`Fail::card`]).
    ///
    /// # Panics
    /// When the reader's card chunk is smaller than one value of `T`.
    pub fn to_card<T: Le + Default>(
        &mut self,
        len: usize,
        part: &str,
        mut write: impl FnMut(Range<usize>, &[T]) -> Result<(), GpuError>,
    ) -> Result<(), Fail> {
        let per = per_chunk::<T>(self.card);
        let mut stage = staging::<T>(len.min(per))?;
        let mut at = 0;
        while at < len {
            let n = per.min(len - at);
            stage.clear();
            let mut left = n;
            while left > 0 {
                let m = left.min(CHUNK / T::SIZE);
                let b = self.exact(m * T::SIZE, part)?;
                stage.extend(b.chunks_exact(T::SIZE).map(T::get));
                left -= m;
            }
            write(at..at + n, &stage).map_err(Fail::card)?;
            at += n;
        }
        Ok(())
    }

    /// [`Put::array`]'s count and values.
    pub fn array<T: Le>(&mut self, part: &str) -> Result<Vec<T>, Fail> {
        let n = self.count(part)?;
        self.vals(n, part)
    }

    pub fn name(&mut self, part: &str) -> Result<Vec<u8>, Fail> {
        let len = self.one::<u32>(part)? as usize;
        if len > NAME_MAX {
            return Err(refused(format!(
                "{part} of {len} bytes, past the {NAME_MAX} the form carries"
            )));
        }
        Ok(self.exact(len, part)?.to_vec())
    }

    /// What [`Put::named`] wrote of `names`; refused by name for a code past
    /// them.
    pub fn named(
        &mut self,
        names: &[&'static str],
        part: &str,
    ) -> Result<Option<&'static str>, Fail> {
        match self.one::<u8>(part)? {
            0 => Ok(None),
            c => names
                .get(usize::from(c) - 1)
                .copied()
                .map(Some)
                .ok_or_else(|| refused(format!("{part} of code {c}"))),
        }
    }

    /// Refused by name unless the stream ends here.
    pub fn end(&mut self) -> Result<(), Fail> {
        let mut b = [0u8; 1];
        loop {
            match self.input.read(&mut b) {
                Ok(0) => return Ok(()),
                Ok(_) => {
                    return Err(refused(format!(
                        "trailing bytes past the state's {}: the form names none",
                        self.bytes
                    )));
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(Fail::Io(e)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fmt::Debug;
    use std::rc::Rc;

    use super::*;

    /// A writer that keeps what it is given and says how much that is.
    struct Sink {
        data: Vec<u8>,
        delivered: Rc<Cell<usize>>,
    }

    impl Sink {
        fn new() -> Sink {
            Sink {
                data: Vec::new(),
                delivered: Rc::new(Cell::new(0)),
            }
        }
    }

    impl Write for Sink {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.data.extend_from_slice(b);
            self.delivered.set(self.data.len());
            Ok(b.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A reader of `data` that counts the bytes taken from it.
    struct Counted<'a> {
        data: &'a [u8],
        taken: Rc<Cell<usize>>,
    }

    impl Read for Counted<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.data.read(buf)?;
            self.taken.set(self.taken.get() + n);
            Ok(n)
        }
    }

    /// `vals` written whole through [`Put::vals`]: the bytes a part of the
    /// card must come to.
    fn whole<T: Le>(vals: &[T]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut w = Put::new(&mut out);
        w.vals(vals).expect("values into memory");
        w.flush().expect("a flush into memory");
        out
    }

    /// The lengths that end at, just short of and just past a chunk, and
    /// span several.
    fn lens(per: usize) -> [usize; 6] {
        [0, 1, per - 1, per, per + 1, 3 * per + 7]
    }

    /// A part of `len` values read from the card in chunks is the bytes of
    /// the whole part, its ranges tile `0..len` in order, none empty or past
    /// a chunk.
    fn from_card_tiles<T: Le + Default + Debug>(card: usize, mk: impl Fn(usize) -> T) {
        let per = card / T::SIZE;
        for len in lens(per) {
            let all: Vec<T> = (0..len).map(&mk).collect();
            let mut sink = Sink::new();
            let mut ranges = Vec::new();
            {
                let mut w = Put::new(&mut sink).with_card_chunk(card);
                w.from_card::<T>(len, |r, out| {
                    assert_eq!(r.len(), out.len());
                    for (i, o) in out.iter_mut().enumerate() {
                        *o = mk(r.start + i);
                    }
                    ranges.push(r);
                    Ok(())
                })
                .expect("a part from a card of memory");
                w.flush().expect("a flush into memory");
            }
            assert_eq!(sink.data, whole(&all), "{len} values of {}", T::SIZE);
            let mut next = 0;
            for r in &ranges {
                assert_eq!(r.start, next, "ranges tile {len} values");
                assert!(!r.is_empty() && r.len() <= per, "range {r:?} of {per}");
                next = r.end;
            }
            assert_eq!(next, len);
            assert_eq!(ranges.len(), len.div_ceil(per));
        }
    }

    /// A chunk that is no multiple of the value sizes: the seams fall where
    /// no array boundary is.
    const ODD_CHUNK: usize = 1002;

    #[test]
    fn a_part_from_the_card_is_the_whole_part_in_chunks() {
        for card in [CARD_CHUNK, ODD_CHUNK] {
            from_card_tiles(card, |i| (i % 251) as u8);
            from_card_tiles(card, |i| (i * 7 % 65521) as u16);
            from_card_tiles(card, |i| i as f32 * 0.5);
        }
        // The smallest chunk a type takes: one value.
        from_card_tiles(1, |i| (i % 251) as u8);
        from_card_tiles(2, |i| (i * 7 % 65521) as u16);
        from_card_tiles(4, |i| i as f32 * 0.5);
    }

    /// The host holds one chunk of the part and the writer's buffer: at each
    /// read, the bytes pulled from the card less those that reached `out`
    /// stay within [`CARD_CHUNK`] and [`CHUNK`], whatever the length.
    #[test]
    fn a_part_from_the_card_is_read_ahead_by_one_chunk_and_one_buffer() {
        let len = 8 * (CARD_CHUNK / 2) + 5;
        let mut sink = Sink::new();
        let delivered = Rc::clone(&sink.delivered);
        let (mut pulled, mut worst) = (0usize, 0usize);
        {
            let mut w = Put::new(&mut sink);
            w.from_card::<u16>(len, |r, out| {
                pulled += r.len() * 2;
                worst = worst.max(pulled - delivered.get());
                out.fill(1);
                Ok(())
            })
            .expect("a part from a card of memory");
            w.flush().expect("a flush into memory");
        }
        assert_eq!(sink.data.len(), len * 2);
        assert!(
            worst <= CARD_CHUNK + CHUNK,
            "{worst} bytes read ahead of what reached the stream"
        );
    }

    /// A part to the card is the values written, its ranges tile `0..len` in
    /// order, and the chunk read ahead of the card's copy stays within
    /// [`CARD_CHUNK`] and [`CHUNK`].
    fn to_card_tiles<T: Le + Default + Debug + PartialEq>(card: usize, mk: impl Fn(usize) -> T) {
        let per = card / T::SIZE;
        for len in lens(per) {
            let all: Vec<T> = (0..len).map(&mk).collect();
            let bytes = whole(&all);
            let taken = Rc::new(Cell::new(0));
            let mut input = Counted {
                data: &bytes,
                taken: Rc::clone(&taken),
            };
            let mut r = Take::new(&mut input).with_card_chunk(card);
            let (mut got, mut next, mut handed, mut worst) = (Vec::new(), 0, 0usize, 0usize);
            r.to_card::<T>(len, "a part", |range, chunk| {
                assert_eq!(range.start, next, "ranges tile {len} values");
                assert_eq!(range.len(), chunk.len());
                assert!(!chunk.is_empty() && chunk.len() <= per);
                worst = worst.max(taken.get() - handed);
                handed += chunk.len() * T::SIZE;
                got.extend_from_slice(chunk);
                next = range.end;
                Ok(())
            })
            .expect("a part to a card of memory");
            assert_eq!(got, all, "{len} values of {}", T::SIZE);
            assert_eq!(next, len);
            assert!(worst <= card + CHUNK, "{worst} bytes read ahead");
            r.end().expect("the part is the stream");
        }
    }

    #[test]
    fn a_part_to_the_card_is_the_whole_part_in_chunks() {
        for card in [CARD_CHUNK, ODD_CHUNK] {
            to_card_tiles(card, |i| (i % 251) as u8);
            to_card_tiles(card, |i| (i * 7 % 65521) as u16);
            to_card_tiles(card, |i| i as f32 * 0.5);
        }
        // The smallest chunk a type takes: one value.
        to_card_tiles(1, |i| (i % 251) as u8);
        to_card_tiles(2, |i| (i * 7 % 65521) as u16);
        to_card_tiles(4, |i| i as f32 * 0.5);
    }

    /// A card chunk that holds no whole value is refused by name, never
    /// rounded up to one value: a chunk of 0 bytes where it is set, one
    /// smaller than the part's value where the part is copied, each way.
    #[test]
    fn a_card_chunk_that_holds_no_value_is_refused_by_name() {
        fn panics_with(says: &str, f: impl FnOnce()) {
            let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
                .expect_err("a chunk that holds no value is refused");
            let text = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            assert!(text.contains(says), "{text:?}: not {says:?}");
        }
        panics_with("a card chunk of 0 bytes holds nothing", || {
            let mut out = Vec::new();
            drop(Put::new(&mut out).with_card_chunk(0));
        });
        panics_with("a card chunk of 0 bytes holds nothing", || {
            let mut input: &[u8] = &[];
            drop(Take::new(&mut input).with_card_chunk(0));
        });
        panics_with(
            "a card chunk of 3 bytes holds no whole 4-byte value",
            || {
                let mut out = Vec::new();
                let mut w = Put::new(&mut out).with_card_chunk(3);
                drop(w.from_card::<f32>(5, |_, _| Ok(())));
            },
        );
        panics_with(
            "a card chunk of 1 bytes holds no whole 2-byte value",
            || {
                let mut input: &[u8] = &[0; 10];
                let mut r = Take::new(&mut input).with_card_chunk(1);
                drop(r.to_card::<u16>(5, "a part", |_, _| Ok(())));
            },
        );
    }

    /// A stream that ends inside a part is refused by name after the chunks
    /// before it were handed on, and a count past the stream sizes no
    /// allocation.
    #[test]
    fn a_part_to_the_card_that_ends_early_is_a_short_stream() {
        let per = CARD_CHUNK / 2;
        let bytes = whole(&vec![3u16; per + 1]);
        let mut input = bytes.as_slice();
        let mut r = Take::new(&mut input);
        let mut chunks = 0;
        let got = r.to_card::<u16>(2 * per, "a part", |_, _| {
            chunks += 1;
            Ok(())
        });
        match got {
            Err(Fail::Refused(e)) => {
                assert!(
                    e.to_string()
                        .contains("a short stream: it ends inside a part")
                );
            }
            other => panic!("{other:?}: not a short stream"),
        }
        assert_eq!(chunks, 1);
        let mut input = [0u8; 10].as_slice();
        let got = Take::new(&mut input).to_card::<u8>(usize::MAX / 4, "a part", |_, _| Ok(()));
        assert!(matches!(got, Err(Fail::Refused(_))), "{got:?}");
    }

    /// The card's refusals by name are the request's, its other errors the
    /// card's, in both directions; a stream that fails is the stream's.
    #[test]
    fn a_failure_of_the_card_or_the_stream_keeps_its_kind() {
        let shape = || GpuError::shape("a test copy", "out of range");
        let driver = || GpuError::Protocol {
            what: "a test copy",
            detail: "the card went".to_owned(),
        };
        let mut sink = Sink::new();
        let mut w = Put::new(&mut sink);
        let r = w.from_card::<u8>(4, |_, _| Err(shape()));
        assert!(matches!(r, Err(Fail::Refused(_))), "{r:?}");
        let r = w.from_card::<u8>(4, |_, _| Err(driver()));
        assert!(matches!(r, Err(Fail::Card(_))), "{r:?}");
        let mut input = [0u8; 4].as_slice();
        let mut t = Take::new(&mut input);
        let r = t.to_card::<u8>(4, "a part", |_, _| Err(shape()));
        assert!(matches!(r, Err(Fail::Refused(_))), "{r:?}");
        let mut input = [0u8; 4].as_slice();
        let mut t = Take::new(&mut input);
        let r = t.to_card::<u8>(4, "a part", |_, _| Err(driver()));
        assert!(matches!(r, Err(Fail::Card(_))), "{r:?}");

        struct Full;
        impl Write for Full {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(
                    io::ErrorKind::StorageFull,
                    "the disk is full",
                ))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut full = Full;
        let mut w = Put::new(&mut full);
        let r = w
            .from_card::<u8>(2 * CHUNK, |_, out| {
                out.fill(0);
                Ok(())
            })
            .and_then(|()| w.flush());
        assert!(
            matches!(&r, Err(Fail::Io(e)) if e.kind() == io::ErrorKind::StorageFull),
            "{r:?}"
        );
    }

    /// A restore's failure survives a failing reset: the failure that follows
    /// it names both, whatever kind the restore's is, and a reset that
    /// worked changes nothing.
    #[test]
    fn a_failing_reset_does_not_replace_the_restores_failure() {
        let driver = || GpuError::Protocol {
            what: "a test reset",
            detail: "the card went".to_owned(),
        };
        let restores: [fn() -> Fail; 3] = [
            || refused("a short stream: it ends inside a part"),
            || Fail::Io(io::Error::new(io::ErrorKind::BrokenPipe, "the pipe closed")),
            || {
                Fail::Card(GpuError::Protocol {
                    what: "a test copy",
                    detail: "the copy failed".to_owned(),
                })
            },
        ];
        for mk in restores {
            let said = mk().to_string();
            let kept = mk().with_reset(Ok(()));
            assert_eq!(kept.to_string(), said);
            assert_eq!(std::mem::discriminant(&kept), std::mem::discriminant(&mk()));
            match mk().with_reset(Err(driver())) {
                Fail::Reset { restore, reset } => {
                    assert_eq!(
                        restore.to_string(),
                        said,
                        "the restore's failure kept whole"
                    );
                    assert_eq!(
                        std::mem::discriminant(&*restore),
                        std::mem::discriminant(&mk())
                    );
                    assert!(matches!(
                        reset,
                        GpuError::Protocol {
                            what: "a test reset",
                            ..
                        }
                    ));
                }
                other => panic!("{other}: a failed reset is not named"),
            }
        }
        let text = restores[0]().with_reset(Err(driver())).to_string();
        assert!(
            text.contains("a short stream")
                && text.contains("reset")
                && text.contains("the card went"),
            "{text}"
        );
    }
}
