//! A recurrent model's sequence state on the host ([`SeqState`]): what a
//! server's prompt cache holds for a session it switches away from, and puts
//! back when the session returns.
//!
//! A state of `n` positions holds three parts, each copied once between the
//! card and the host:
//! - the positional stores' rows of positions `0..n` (the K and V planes,
//!   the keys a selector scores, every f16 store indexed by position), as the
//!   body lists them ([`Spans`]): rows past `n` are never read before a later
//!   position writes them, so they are not carried;
//! - the recurrent stores (the body's checkpoint list,
//!   [`Checkpoints::lens`]) at `n`, read from the card;
//! - the recurrent stores at each point it carries, read from the
//!   checkpoints' host slots ([`Checkpoints::read_point`]): a later request
//!   that shares only a shorter prefix (a resend with the reasoning taken
//!   out of the last turn) cuts to one of them.
//!
//! Each position the state carries has a host value of the body's own `T`
//! beside it (a history the card does not hold), handed back with it.
//!
//! A resume ([`SeqState::load`]) puts every part back: the positional rows
//! and the recurrent stores at `n` on the card, each carried point into the
//! checkpoints ([`Checkpoints::adopt`]), which then hold those points and no
//! other. It is refused by name, nothing copied, for a state of another
//! model, card or context ([`Identity`]), of another store layout, or onto
//! checkpoints that hold a point.
//!
//! The host memory is pageable: each run is one synchronous copy on the
//! engine stream, through a checked window of its store.

use std::fmt;
use std::path::PathBuf;

use cuda_core::{CudaStream, DeviceBuffer};

use super::Checkpoints;
use crate::GpuError;
use crate::tensor::{Window, WindowMut};

const WHAT: &str = "sequence state";

/// What a state belongs to: a resume onto anything else is refused by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// The architecture's name.
    pub arch: &'static str,
    /// The model file's first shard.
    pub file: PathBuf,
    /// The card the stores live on.
    pub card: String,
    /// The positions every store was sized for.
    pub ctx: usize,
}

impl fmt::Display for Identity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {} on {} at {} positions",
            self.arch,
            self.file.display(),
            self.card,
            self.ctx
        )
    }
}

/// The rows of one positional f16 store a state carries, under one borrow
/// of it: `planes` runs of `len` values, the first at value 0 and each
/// `stride` values past the one before.
pub struct Spans<'a> {
    buf: &'a mut DeviceBuffer<u16>,
    planes: usize,
    stride: usize,
    len: usize,
}

impl<'a> Spans<'a> {
    /// The runs of `buf` (the type's doc); refused by name when one would
    /// leave the buffer or overlap the next.
    pub fn planes(
        buf: &'a mut DeviceBuffer<u16>,
        planes: usize,
        stride: usize,
        len: usize,
    ) -> Result<Spans<'a>, GpuError> {
        let end = planes
            .checked_sub(1)
            .map_or(Some(0), |p| p.checked_mul(stride)?.checked_add(len));
        if end.is_none_or(|e| e > buf.len()) || (planes > 1 && stride < len) {
            return Err(shape(format!(
                "{planes} planes of {len} values {stride} apart in a store of {}",
                buf.len()
            )));
        }
        Ok(Spans {
            buf,
            planes,
            stride,
            len,
        })
    }

    /// The values of each run, in order.
    fn sizes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.planes).map(|_| self.len)
    }

    /// Each run into `host` from `*off` on, `*off` past them.
    fn read(&self, stream: &CudaStream, host: &mut [u16], off: &mut usize) -> Result<(), GpuError> {
        for p in 0..self.planes {
            if self.len > 0 {
                let byte = p * self.stride * size_of::<u16>();
                let w: Window<'_, u16> = Window::of(self.buf, byte, self.len)?;
                w.copy_to_host(stream, &mut host[*off..*off + self.len])?;
            }
            *off += self.len;
        }
        Ok(())
    }

    /// Each run from `host` from `*off` on, `*off` past them.
    fn write(
        &mut self,
        stream: &CudaStream,
        host: &[u16],
        off: &mut usize,
    ) -> Result<(), GpuError> {
        for p in 0..self.planes {
            if self.len > 0 {
                let byte = p * self.stride * size_of::<u16>();
                let mut w: WindowMut<'_, u16> = WindowMut::of_mut(self.buf, byte, self.len)?;
                w.copy_from_host(stream, &host[*off..*off + self.len])?;
            }
            *off += self.len;
        }
        Ok(())
    }
}

/// The stores a state is read from or put back into, lent for one call:
/// the positional rows the body lists, and the recurrent stores in the
/// checkpoints' order ([`Checkpoints::lens`]).
pub struct Lent<'s, 'p, 'r> {
    pub positional: &'s mut [Spans<'p>],
    pub recurrent: &'s mut [&'r mut DeviceBuffer<f32>],
}

impl Lent<'_, '_, '_> {
    fn spans(&self) -> Vec<usize> {
        self.positional.iter().flat_map(Spans::sizes).collect()
    }

    fn lens(&self) -> Vec<usize> {
        self.recurrent.iter().map(|s| s.len()).collect()
    }
}

/// The host values a resume hands back: the current position's, and each
/// carried point's with its position.
pub type Hosts<'s, T> = (&'s T, &'s [(u32, T)]);

/// A sequence state of `positions` positions on the host (the module doc),
/// with the body's host value `T` beside the current position and each
/// carried point.
pub struct SeqState<T> {
    who: Identity,
    positions: u32,
    /// The positional rows, in the order the body lists them.
    rows: Vec<u16>,
    /// The recurrent stores at `positions`, then at each point.
    recurrent: Vec<f32>,
    /// The values of each positional run, in order.
    spans: Vec<usize>,
    /// f32s of each recurrent store.
    lens: Vec<usize>,
    current: T,
    /// Ascending, each below `positions`.
    points: Vec<(u32, T)>,
}

impl<T> SeqState<T> {
    /// The state of `who`'s model standing at `positions`: `lent`'s
    /// positional rows, its recurrent stores (the checkpoints' list, which
    /// hold `positions`), and the stores of each of `points` from `ckpt`,
    /// each position with its host value. Every copy is ordered after what
    /// the stream ran before. Refused by name when the points are not
    /// ascending from above 0 and below `positions`, `ckpt` holds no such
    /// point, or the stores are not the checkpoints'.
    pub fn save(
        stream: &CudaStream,
        who: Identity,
        positions: u32,
        lent: &Lent<'_, '_, '_>,
        current: T,
        points: Vec<(u32, T)>,
        ckpt: &Checkpoints,
    ) -> Result<SeqState<T>, GpuError> {
        let lens = lent.lens();
        if lens != ckpt.lens() {
            return Err(shape(format!(
                "recurrent stores of {lens:?} f32s; the checkpoints copy {:?}",
                ckpt.lens()
            )));
        }
        check_points(positions, points.iter().map(|&(p, _)| p))?;
        let spans = lent.spans();
        let one: usize = lens.iter().sum();
        let mut rows = vec![0u16; spans.iter().sum()];
        let mut recurrent = vec![0f32; one * (1 + points.len())];
        let mut off = 0;
        for s in lent.positional.iter() {
            s.read(stream, &mut rows, &mut off)?;
        }
        let mut off = 0;
        for s in lent.recurrent.iter() {
            s.copy_to_host(stream, &mut recurrent[off..off + s.len()])?;
            off += s.len();
        }
        for (i, &(at, _)) in points.iter().enumerate() {
            let from = one * (1 + i);
            ckpt.read_point(at, &mut recurrent[from..from + one])?;
        }
        Ok(SeqState {
            who,
            positions,
            rows,
            recurrent,
            spans,
            lens,
            current,
            points,
        })
    }

    /// The state put back on `who`'s model: `lent`'s positional rows and
    /// recurrent stores from the host, each carried point adopted by `ckpt`;
    /// the host values of the current position and of each point returned.
    /// Refused by name, nothing copied, when `who` is another model's, the
    /// runs or the stores are another layout, or `ckpt` holds a point.
    pub fn load(
        &self,
        stream: &CudaStream,
        who: &Identity,
        lent: &mut Lent<'_, '_, '_>,
        ckpt: &mut Checkpoints,
    ) -> Result<Hosts<'_, T>, GpuError> {
        if *who != self.who {
            return Err(shape(format!("a state of {} put back on {who}", self.who)));
        }
        let spans = lent.spans();
        if spans != self.spans {
            return Err(shape(format!(
                "a state of {} positional runs ({} values) put back into {} ({} values)",
                self.spans.len(),
                self.spans.iter().sum::<usize>(),
                spans.len(),
                spans.iter().sum::<usize>()
            )));
        }
        let lens = lent.lens();
        if lens != self.lens || lens != ckpt.lens() {
            return Err(shape(format!(
                "a state of recurrent stores of {:?} f32s put back into {lens:?} (the \
                 checkpoints copy {:?})",
                self.lens,
                ckpt.lens()
            )));
        }
        if !ckpt.positions().is_empty() || ckpt.pending() {
            return Err(shape(format!(
                "a state put back beside checkpoints at {:?}: resume onto an empty model",
                ckpt.positions()
            )));
        }
        let mut off = 0;
        for s in lent.positional.iter_mut() {
            s.write(stream, &self.rows, &mut off)?;
        }
        let mut off = 0;
        for s in lent.recurrent.iter_mut() {
            let n = s.len();
            s.copy_from_host(stream, &self.recurrent[off..off + n])?;
            off += n;
        }
        let one: usize = self.lens.iter().sum();
        for (i, &(at, _)) in self.points.iter().enumerate() {
            let from = one * (1 + i);
            ckpt.adopt(at, &self.recurrent[from..from + one])?;
        }
        Ok((&self.current, &self.points))
    }

    /// The positions it holds.
    #[must_use]
    pub fn positions(&self) -> u32 {
        self.positions
    }

    /// The host bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.rows.len() * size_of::<u16>() + self.recurrent.len() * size_of::<f32>()
    }

    /// The positions of the points it carries, ascending.
    #[must_use]
    pub fn points(&self) -> Vec<u32> {
        self.points.iter().map(|&(p, _)| p).collect()
    }

    /// The longest prefix of at most `n` positions a model holds right after
    /// the state is put back: every position when `n` reaches them, else the
    /// nearest carried point at or below `n`, else none.
    #[must_use]
    pub fn keep_point(&self, n: u32) -> u32 {
        if n >= self.positions {
            return self.positions;
        }
        self.points
            .iter()
            .rev()
            .map(|&(p, _)| p)
            .find(|&p| p <= n)
            .unwrap_or(0)
    }
}

/// Refused by name unless `points` ascend, each above 0 and below
/// `positions`.
fn check_points(positions: u32, points: impl Iterator<Item = u32>) -> Result<(), GpuError> {
    let mut last = 0;
    for p in points {
        if p <= last || p >= positions {
            return Err(shape(format!(
                "a point at {p} after {last} in a state of {positions} positions: the points \
                 ascend from above 0 and lie below the positions"
            )));
        }
        last = p;
    }
    Ok(())
}

fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_core::CudaContext;

    /// A state of 5 positions carries rows 0..5 of a two-plane store, the
    /// recurrent stores at 5 and the point at 3; put back on an empty model it
    /// restores the rows and the stores, the checkpoints hold exactly the
    /// point, and the host values come back. A state put back on another
    /// model's identity is refused by name, nothing copied.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_state_goes_and_comes_back() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("a stream");
        let who = Identity {
            arch: "test",
            file: PathBuf::from("/x"),
            card: "card".to_owned(),
            ctx: 8,
        };
        let mut c = Checkpoints::new(&ctx, vec![4], 1 << 20, 512).expect("checkpoints");
        // Two planes of 8 positions, a row of 2 values.
        let plane: Vec<u16> = (0..32).collect();
        let mut kv = DeviceBuffer::from_host(&stream, &plane).expect("a store");
        let mut rec = DeviceBuffer::from_host(&stream, &[3.0f32; 4]).expect("a store");
        assert!(matches!(
            c.take(&stream, 3, &mut [&mut rec]).expect("a take at 3"),
            runtime::seqstate::Take::Copy { .. }
        ));
        rec.copy_from_host(&stream, &[5.0; 4]).expect("rec");
        let s = {
            let mut spans = [Spans::planes(&mut kv, 2, 16, 10).expect("spans")];
            let lent = Lent {
                positional: &mut spans,
                recurrent: &mut [&mut rec],
            };
            SeqState::save(&stream, who.clone(), 5, &lent, 'c', vec![(3, 'p')], &c).expect("a save")
        };
        assert_eq!(s.bytes(), 2 * 10 * 2 + 2 * 16);
        assert_eq!(
            (s.keep_point(4), s.keep_point(2), s.keep_point(9)),
            (3, 0, 5)
        );
        let mut d = Checkpoints::new(&ctx, vec![4], 1 << 20, 512).expect("checkpoints");
        kv.copy_from_host(&stream, &[9u16; 32]).expect("kv");
        rec.copy_from_host(&stream, &[0.0; 4]).expect("rec");
        let other = Identity {
            ctx: 16,
            ..who.clone()
        };
        {
            let mut spans = [Spans::planes(&mut kv, 2, 16, 10).expect("spans")];
            let mut lent = Lent {
                positional: &mut spans,
                recurrent: &mut [&mut rec],
            };
            let err = s
                .load(&stream, &other, &mut lent, &mut d)
                .expect_err("another model's identity refused");
            assert!(
                err.to_string()
                    .contains("put back on test /x on card at 16"),
                "{err}"
            );
        }
        assert_eq!(rec.to_host_vec(&stream).expect("rec"), [0.0; 4]);
        let (cur, pts) = {
            let mut spans = [Spans::planes(&mut kv, 2, 16, 10).expect("spans")];
            let mut lent = Lent {
                positional: &mut spans,
                recurrent: &mut [&mut rec],
            };
            let (cur, pts) = s.load(&stream, &who, &mut lent, &mut d).expect("a load");
            (*cur, pts.to_vec())
        };
        assert_eq!((cur, pts), ('c', vec![(3, 'p')]));
        let back = kv.to_host_vec(&stream).expect("kv");
        assert_eq!(&back[..10], &plane[..10]);
        assert_eq!(&back[16..26], &plane[16..26]);
        assert_eq!(back[10], 9, "a row past the positions is not carried");
        assert_eq!(rec.to_host_vec(&stream).expect("rec"), [5.0; 4]);
        assert_eq!(d.positions(), [3]);
        d.cut(3, 5).expect("a cut to the adopted point");
        d.apply(&stream, &mut [&mut rec]).expect("its restore");
        assert_eq!(rec.to_host_vec(&stream).expect("rec"), [3.0; 4]);
    }
}
