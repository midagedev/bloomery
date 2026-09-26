//! A streaming GGUF v3 writer, the reader's twin. The header — every
//! metadata pair and every tensor's name, dims, type id and offset — is laid
//! out whole first ([`Layout`]) and written first; then each tensor's bytes
//! go out in declaration order ([`Writer`]), so a file far larger than memory
//! is written one tensor at a time.
//!
//! Offsets follow ggml's `gguf_write_to_buf`: a tensor starts at the data
//! base plus the padded sizes of the tensors before it, and every tensor's
//! bytes are followed by zeros up to the next multiple of the alignment —
//! `general.alignment` when the metadata sets it, else 32. The data base is
//! the end of the header rounded up to the same alignment, as the reader
//! computes it.
//!
//! What a reader would refuse is refused here by name, before a byte is
//! written: a repeated key or tensor name, an alignment that is not a
//! power-of-two u32, an array whose elements carry different value tags or
//! none (an empty array has no element type), dims outside 1..=4 or holding
//! a 0, and, for a type id ggml's table sizes ([`crate::ggml_type_info`]), a
//! first dim off the block or a byte count its dims do not make. A type id no
//! table knows is taken with the byte count its declaration states.

use std::collections::HashSet;
use std::io::Write;
use std::ops::Range;

use crate::{GENERAL_ALIGNMENT, Value, ggml_type_info};

/// The alignment of a file whose metadata does not set one.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// The alignment a `general.alignment` value sets: a power-of-two u32, else
/// [`WriteError::Alignment`].
pub fn alignment_value(v: &Value) -> Result<u64, WriteError> {
    match v {
        Value::U32(a) if a.is_power_of_two() => Ok(u64::from(*a)),
        other => Err(WriteError::Alignment {
            value: format!("{other:?}"),
        }),
    }
}

/// The alignment a file of `kvs` is written at: the first
/// `general.alignment` by [`alignment_value`], else [`DEFAULT_ALIGNMENT`].
pub fn file_alignment(kvs: &[(String, Value)]) -> Result<u64, WriteError> {
    kvs.iter()
        .find(|(k, _)| k == GENERAL_ALIGNMENT)
        .map_or(Ok(DEFAULT_ALIGNMENT), |(_, v)| alignment_value(v))
}

/// One tensor as the header declares it: `dims` in ggml's `ne[]` order
/// (`dims[0]` the contiguous axis), its ggml type id, and the bytes its data
/// holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDecl {
    pub name: String,
    pub dims: Vec<u64>,
    pub type_id: u32,
    pub nbytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("metadata key {key:?} appears twice")]
    DuplicateKey { key: String },
    #[error("metadata {key:?}: an empty array has no element type")]
    EmptyArray { key: String },
    #[error("metadata {key:?}: an array mixes value tags {first} and {other}")]
    MixedArray { key: String, first: u32, other: u32 },
    #[error(
        "{} is {value}: the alignment is a power-of-two u32",
        GENERAL_ALIGNMENT
    )]
    Alignment { value: String },
    #[error("tensor {name:?} is declared twice")]
    DuplicateTensor { name: String },
    #[error("tensor {name:?}: dims {dims:?}; a tensor has 1..=4 dims, each at least 1")]
    BadDims { name: String, dims: Vec<u64> },
    #[error("tensor {name:?}: ne[0] {ne0} is not a multiple of type {type_id}'s block of {blck}")]
    UnalignedRow {
        name: String,
        type_id: u32,
        ne0: u64,
        blck: u64,
    },
    #[error(
        "tensor {name:?}: {nbytes} bytes declared, but type {type_id} at dims {dims:?} is {want}"
    )]
    TensorBytes {
        name: String,
        type_id: u32,
        dims: Vec<u64>,
        nbytes: u64,
        want: u64,
    },
    #[error("tensor {name:?}: its bytes end past the largest file offset")]
    TooLarge { name: String },
    #[error("tensor {got:?} written out of order: {}", next_text(.next))]
    OutOfOrder { got: String, next: Option<String> },
    #[error("tensor {name:?}: {got} bytes written, {want} declared")]
    Length { name: String, got: u64, want: u64 },
    #[error("finished with {left} of {declared} tensors unwritten, the next {next:?}")]
    Unfinished {
        next: String,
        left: usize,
        declared: usize,
    },
    #[error("an earlier write failed and the file is incomplete; drop the writer")]
    Broken,
}

/// [`WriteError::OutOfOrder`]'s tail.
fn next_text(next: &Option<String>) -> String {
    match next {
        Some(n) => format!("the next declared tensor is {n:?}"),
        None => "every declared tensor is written".to_string(),
    }
}

/// A file's layout, fixed before any byte is written: the header's bytes up
/// to the data base, and where each tensor's bytes go.
#[derive(Clone, Debug)]
pub struct Layout {
    /// Magic through the last tensor description, zero-padded to the data base.
    header: Vec<u8>,
    alignment: u64,
    /// Each tensor with its offset from the data base, in declaration order.
    tensors: Vec<(TensorDecl, u64)>,
    /// From the data base to the end of the last tensor's padding.
    data_len: u64,
}

impl Layout {
    /// Lay out a file of `kvs`, in order, and `tensors`, in order; every
    /// refusal the module header lists is an error naming the key or tensor.
    pub fn new(kvs: &[(String, Value)], tensors: Vec<TensorDecl>) -> Result<Layout, WriteError> {
        let mut alignment = DEFAULT_ALIGNMENT;
        for (i, (key, v)) in kvs.iter().enumerate() {
            if kvs[..i].iter().any(|(k, _)| k == key) {
                return Err(WriteError::DuplicateKey { key: key.clone() });
            }
            check_value(key, v)?;
            if key == GENERAL_ALIGNMENT {
                alignment = alignment_value(v)?;
            }
        }
        let mut names = HashSet::with_capacity(tensors.len());
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut end = 0u64;
        for t in &tensors {
            if !names.insert(t.name.as_str()) {
                return Err(WriteError::DuplicateTensor {
                    name: t.name.clone(),
                });
            }
            check_tensor(t)?;
            offsets.push(end);
            end = t
                .nbytes
                .checked_add(end)
                .and_then(|e| e.checked_next_multiple_of(alignment))
                .ok_or_else(|| WriteError::TooLarge {
                    name: t.name.clone(),
                })?;
        }
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3u32.to_le_bytes());
        header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        header.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (key, v) in kvs {
            put_str(&mut header, key);
            header.extend_from_slice(&tag(v).to_le_bytes());
            put_value(&mut header, v);
        }
        for (t, off) in tensors.iter().zip(&offsets) {
            put_str(&mut header, &t.name);
            header.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                header.extend_from_slice(&d.to_le_bytes());
            }
            header.extend_from_slice(&t.type_id.to_le_bytes());
            header.extend_from_slice(&off.to_le_bytes());
        }
        let data_base = (header.len() as u64).next_multiple_of(alignment);
        header.resize(
            usize::try_from(data_base).expect("the padded header is in memory already"),
            0,
        );
        if data_base.checked_add(end).is_none() {
            let last = tensors.last().map(|t| t.name.clone()).unwrap_or_default();
            return Err(WriteError::TooLarge { name: last });
        }
        Ok(Layout {
            header,
            alignment,
            tensors: tensors.into_iter().zip(offsets).collect(),
            data_len: end,
        })
    }

    /// The file offset of the data section: the header's padded length.
    pub fn data_base(&self) -> u64 {
        self.header.len() as u64
    }

    /// The alignment every tensor's offset and the data base keep.
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// The bytes the finished file holds.
    pub fn file_len(&self) -> u64 {
        self.data_base() + self.data_len
    }
}

/// A tensor declaration's own refusals: its dims, and for a type ggml's
/// table sizes, its block alignment and byte count.
fn check_tensor(t: &TensorDecl) -> Result<(), WriteError> {
    if !(1..=4).contains(&t.dims.len()) || t.dims.contains(&0) {
        return Err(WriteError::BadDims {
            name: t.name.clone(),
            dims: t.dims.clone(),
        });
    }
    let Some((_, blck, tsz)) = ggml_type_info(t.type_id) else {
        return Ok(());
    };
    if !t.dims[0].is_multiple_of(blck) {
        return Err(WriteError::UnalignedRow {
            name: t.name.clone(),
            type_id: t.type_id,
            ne0: t.dims[0],
            blck,
        });
    }
    let want = t.dims[1..]
        .iter()
        .try_fold(tsz * (t.dims[0] / blck), |acc, d| acc.checked_mul(*d))
        .ok_or_else(|| WriteError::TooLarge {
            name: t.name.clone(),
        })?;
    if want != t.nbytes {
        return Err(WriteError::TensorBytes {
            name: t.name.clone(),
            type_id: t.type_id,
            dims: t.dims.clone(),
            nbytes: t.nbytes,
            want,
        });
    }
    Ok(())
}

/// An array's elements must share one value tag, and there must be one to
/// share; nested arrays are checked the same way.
fn check_value(key: &str, v: &Value) -> Result<(), WriteError> {
    let Value::Array(items) = v else {
        return Ok(());
    };
    let Some(first) = items.first() else {
        return Err(WriteError::EmptyArray {
            key: key.to_string(),
        });
    };
    for item in items {
        if tag(item) != tag(first) {
            return Err(WriteError::MixedArray {
                key: key.to_string(),
                first: tag(first),
                other: tag(item),
            });
        }
        check_value(key, item)?;
    }
    Ok(())
}

/// A value's GGUF type tag, the reader's numbering.
fn tag(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

/// A GGUF string: u64 byte length, then the bytes.
fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

/// A value without its tag; an array is its element tag, its count and its
/// elements (checked non-empty and uniform by [`check_value`]).
fn put_value(b: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => b.push(*x),
        Value::I8(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => b.extend_from_slice(&x.to_bits().to_le_bytes()),
        Value::Bool(x) => b.push(u8::from(*x)),
        Value::String(s) => put_str(b, s),
        Value::Array(items) => {
            let elem = items.first().map_or(0, tag);
            b.extend_from_slice(&elem.to_le_bytes());
            b.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                put_value(b, item);
            }
        }
        Value::U64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => b.extend_from_slice(&x.to_bits().to_le_bytes()),
    }
}

/// `n` zero bytes, a page at a time.
fn write_zeros(out: &mut impl Write, mut n: u64) -> std::io::Result<()> {
    const ZEROS: [u8; 4096] = [0; 4096];
    while n > 0 {
        let step = n.min(ZEROS.len() as u64);
        out.write_all(&ZEROS[..step as usize])?;
        n -= step;
    }
    Ok(())
}

/// A file being written: its header is out, and [`Writer::tensor`] takes
/// the declared tensors' bytes in order. After any error the file is
/// incomplete and every later call is [`WriteError::Broken`].
pub struct Writer<W: Write> {
    out: W,
    layout: Layout,
    next: usize,
    broken: bool,
}

impl<W: Write> Writer<W> {
    /// Write `layout`'s header to `out`.
    pub fn new(mut out: W, layout: Layout) -> Result<Writer<W>, WriteError> {
        out.write_all(&layout.header)?;
        Ok(Writer {
            out,
            layout,
            next: 0,
            broken: false,
        })
    }

    /// The next declared tensor's bytes, then its padding. Returns the file
    /// range the bytes occupy, padding excluded.
    pub fn tensor(&mut self, name: &str, bytes: &[u8]) -> Result<Range<u64>, WriteError> {
        if self.broken {
            return Err(WriteError::Broken);
        }
        let Some((decl, offset)) = self.layout.tensors.get(self.next) else {
            return Err(WriteError::OutOfOrder {
                got: name.to_string(),
                next: None,
            });
        };
        if decl.name != name {
            return Err(WriteError::OutOfOrder {
                got: name.to_string(),
                next: Some(decl.name.clone()),
            });
        }
        let got = bytes.len() as u64;
        if got != decl.nbytes {
            return Err(WriteError::Length {
                name: name.to_string(),
                got,
                want: decl.nbytes,
            });
        }
        let start = self.layout.data_base() + offset;
        let pad = got.next_multiple_of(self.layout.alignment) - got;
        let written = self
            .out
            .write_all(bytes)
            .and_then(|()| write_zeros(&mut self.out, pad));
        if let Err(e) = written {
            self.broken = true;
            return Err(e.into());
        }
        self.next += 1;
        Ok(start..start + got)
    }

    /// Every declared tensor must be written; flushes and hands `out` back.
    pub fn finish(mut self) -> Result<W, WriteError> {
        if self.broken {
            return Err(WriteError::Broken);
        }
        if let Some((decl, _)) = self.layout.tensors.get(self.next) {
            return Err(WriteError::Unfinished {
                next: decl.name.clone(),
                left: self.layout.tensors.len() - self.next,
                declared: self.layout.tensors.len(),
            });
        }
        self.out.flush()?;
        Ok(self.out)
    }
}

/// A whole file of `kvs` and `tensors` at `path`: the unit tests' writer.
#[cfg(test)]
pub(crate) fn write_file(
    path: &std::path::Path,
    kvs: &[(String, Value)],
    tensors: &[(TensorDecl, Vec<u8>)],
) -> Layout {
    let layout = Layout::new(kvs, tensors.iter().map(|(t, _)| t.clone()).collect()).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut w = Writer::new(std::io::BufWriter::new(file), layout.clone()).unwrap();
    for (t, bytes) in tensors {
        w.tensor(&t.name, bytes).unwrap();
    }
    w.finish().unwrap();
    layout
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GENERAL_ARCHITECTURE, GgmlType, Gguf};
    use std::path::PathBuf;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gguf-write-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn decl(name: &str, dims: &[u64], type_id: u32, nbytes: u64) -> TensorDecl {
        TensorDecl {
            name: name.to_string(),
            dims: dims.to_vec(),
            type_id,
            nbytes,
        }
    }

    /// Deterministic bytes (xorshift64*), so a data mismatch names a byte.
    fn bytes(seed: u64, n: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
            })
            .collect()
    }

    /// Two values are the same value: the same variant and, for floats, the
    /// same bits (`PartialEq` would pass `-0.0` for `0.0` and fail a NaN).
    fn same(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::F32(x), Value::F32(y)) => x.to_bits() == y.to_bits(),
            (Value::F64(x), Value::F64(y)) => x.to_bits() == y.to_bits(),
            (Value::Array(x), Value::Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same(p, q))
            }
            _ => a == b,
        }
    }

    /// Every value type the reader parses, arrays nested and not.
    fn every_value() -> Vec<(String, Value)> {
        let s = |x: &str| Value::String(x.to_string());
        [
            (GENERAL_ARCHITECTURE, s("roundtrip")),
            ("t.u8", Value::U8(0xAB)),
            ("t.i8", Value::I8(-5)),
            ("t.u16", Value::U16(0xBEEF)),
            ("t.i16", Value::I16(-1234)),
            ("t.u32", Value::U32(0xDEAD_BEEF)),
            ("t.i32", Value::I32(-123_456)),
            ("t.f32", Value::F32(-0.0)),
            ("t.f32.nan", Value::F32(f32::from_bits(0x7FC0_1234))),
            ("t.bool.t", Value::Bool(true)),
            ("t.bool.f", Value::Bool(false)),
            ("t.string", s("héllo ✓")),
            ("t.string.empty", s("")),
            ("t.u64", Value::U64(u64::MAX - 1)),
            ("t.i64", Value::I64(i64::MIN)),
            ("t.f64", Value::F64(f64::from_bits(0xFFF8_0000_0000_0001))),
            (
                "t.arr.u32",
                Value::Array(vec![Value::U32(1), Value::U32(2), Value::U32(u32::MAX)]),
            ),
            ("t.arr.str", Value::Array(vec![s("a"), s(""), s("ccc")])),
            (
                "t.arr.nested",
                Value::Array(vec![
                    Value::Array(vec![Value::I8(-1), Value::I8(2)]),
                    Value::Array(vec![s("x")]),
                    Value::Array(vec![Value::Array(vec![Value::F64(1.5)])]),
                ]),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    /// The writer round trip at `alignment` (`None`: no alignment key, so
    /// 32): every value type, nested arrays and several tensors of known
    /// types read back through the strict reader with the same pairs in the
    /// same order, the same names, dims, types, offsets, byte counts and
    /// bytes, and the file ends where the layout said.
    fn round_trip(tag: &str, alignment: Option<u32>) {
        let d = dir(tag);
        let p = d.join("rt.gguf");
        let mut kvs = every_value();
        if let Some(a) = alignment {
            kvs.push((GENERAL_ALIGNMENT.to_string(), Value::U32(a)));
        }
        let tensors: Vec<(TensorDecl, Vec<u8>)> = [
            decl("t.f32", &[3, 2], 0, 24),
            decl("t.f16", &[5], 1, 10),
            decl("t.q8_0", &[64, 2], 8, 136),
            decl("t.q3_k", &[256, 3], 11, 330),
            decl("t.q4_k", &[512, 1, 2], 12, 576),
            decl("t.bf16", &[7], 30, 14),
            decl("t.i32", &[1, 1, 1, 3], 26, 12),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            let b = bytes(i as u64 + 7, t.nbytes);
            (t, b)
        })
        .collect();
        let layout = write_file(&p, &kvs, &tensors);
        let a = u64::from(alignment.unwrap_or(32));
        assert_eq!(layout.alignment(), a);
        let g = Gguf::open(&p).unwrap();
        assert_eq!(g.alignment(), a);
        assert_eq!(g.data_base(), layout.data_base());
        assert_eq!(g.data_base() % a, 0);
        assert_eq!(
            std::fs::metadata(&p).unwrap().len(),
            layout.file_len(),
            "file length"
        );
        let back: Vec<(&str, &Value)> = g.iter_kv().collect();
        assert_eq!(back.len(), kvs.len());
        for ((k, v), (bk, bv)) in kvs.iter().zip(&back) {
            assert_eq!(k, bk);
            assert!(same(v, bv), "{k}: wrote {v:?}, read {bv:?}");
        }
        let mut offset = 0;
        assert_eq!(g.tensor_count(), tensors.len());
        for ((t, b), info) in tensors.iter().zip(g.iter_tensors()) {
            assert_eq!(info.name, t.name);
            assert_eq!(info.dims, t.dims);
            assert_eq!(info.ty, GgmlType::from_u32(t.type_id));
            assert_eq!(info.offset, offset, "{}", t.name);
            assert_eq!(info.offset % a, 0);
            assert_eq!(info.nbytes, t.nbytes);
            assert_eq!(g.data(info).unwrap(), b.as_slice(), "{} bytes", t.name);
            offset += t.nbytes.next_multiple_of(a);
        }
        assert_eq!(layout.file_len(), layout.data_base() + offset);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_writer_round_trips_at_the_default_alignment() {
        round_trip("align32", None);
    }

    #[test]
    fn the_writer_round_trips_at_alignment_64() {
        round_trip("align64", Some(64));
    }

    /// A sink that takes `room` bytes, then fails every write.
    struct Full(usize);

    impl Write for Full {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.0 == 0 {
                return Err(std::io::Error::other("full"));
            }
            let n = buf.len().min(self.0);
            self.0 -= n;
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn layout_err(kvs: &[(String, Value)], tensors: Vec<TensorDecl>) -> WriteError {
        match Layout::new(kvs, tensors) {
            Ok(_) => panic!("the layout must be refused"),
            Err(e) => e,
        }
    }

    /// Every refusal names its key or tensor: a declaration refused at
    /// layout, and a write out of order, of the wrong length, past the
    /// declared tensors, a finish with a tensor unwritten, and every call
    /// after a failed write.
    #[test]
    fn the_writer_refuses_by_name() {
        let kv = |k: &str, v: Value| vec![(k.to_string(), v)];
        let f32x4 = |n: &str| decl(n, &[4], 0, 16);
        match layout_err(&[], vec![decl("q", &[256, 2], 11, 219)]) {
            WriteError::TensorBytes {
                name, nbytes, want, ..
            } => assert_eq!((name.as_str(), nbytes, want), ("q", 219, 220)),
            other => panic!("wrong refusal: {other}"),
        }
        match layout_err(&[], vec![decl("q", &[100, 2], 11, 220)]) {
            WriteError::UnalignedRow {
                name, ne0, blck, ..
            } => {
                assert_eq!((name.as_str(), ne0, blck), ("q", 100, 256));
            }
            other => panic!("wrong refusal: {other}"),
        }
        let cases: [&[u64]; 3] = [&[], &[4, 0], &[1, 1, 1, 1, 1]];
        for dims in cases {
            match layout_err(&[], vec![decl("d", dims, 1011, 16)]) {
                WriteError::BadDims { name, .. } => assert_eq!(name, "d"),
                other => panic!("wrong refusal for {dims:?}: {other}"),
            }
        }
        match layout_err(&[], vec![f32x4("a"), f32x4("a")]) {
            WriteError::DuplicateTensor { name } => assert_eq!(name, "a"),
            other => panic!("wrong refusal: {other}"),
        }
        match layout_err(&[], vec![decl("big", &[4], 1011, u64::MAX - 8)]) {
            WriteError::TooLarge { name } => assert_eq!(name, "big"),
            other => panic!("wrong refusal: {other}"),
        }
        let mut twice = kv("k", Value::U8(1));
        twice.push(("k".to_string(), Value::U8(2)));
        match layout_err(&twice, vec![]) {
            WriteError::DuplicateKey { key } => assert_eq!(key, "k"),
            other => panic!("wrong refusal: {other}"),
        }
        match layout_err(&kv("e", Value::Array(vec![])), vec![]) {
            WriteError::EmptyArray { key } => assert_eq!(key, "e"),
            other => panic!("wrong refusal: {other}"),
        }
        let inner_empty = Value::Array(vec![Value::Array(vec![])]);
        match layout_err(&kv("n", inner_empty), vec![]) {
            WriteError::EmptyArray { key } => assert_eq!(key, "n"),
            other => panic!("wrong refusal: {other}"),
        }
        match layout_err(
            &kv("m", Value::Array(vec![Value::U8(1), Value::I8(1)])),
            vec![],
        ) {
            WriteError::MixedArray { key, first, other } => {
                assert_eq!((key.as_str(), first, other), ("m", 0, 1));
            }
            other => panic!("wrong refusal: {other}"),
        }
        for bad in [Value::U32(48), Value::U32(0), Value::U16(64)] {
            match layout_err(&kv(GENERAL_ALIGNMENT, bad.clone()), vec![]) {
                WriteError::Alignment { .. } => {}
                other => panic!("wrong refusal for {bad:?}: {other}"),
            }
        }
        let two = || Layout::new(&[], vec![f32x4("a"), f32x4("b")]).unwrap();
        let mut w = Writer::new(Vec::new(), two()).unwrap();
        match w.tensor("b", &[0; 16]) {
            Err(WriteError::OutOfOrder { got, next }) => {
                assert_eq!((got.as_str(), next.as_deref()), ("b", Some("a")));
            }
            other => panic!("wrong refusal: {other:?}"),
        }
        match w.tensor("a", &[0; 15]) {
            Err(WriteError::Length { name, got, want }) => {
                assert_eq!((name.as_str(), got, want), ("a", 15, 16));
            }
            other => panic!("wrong refusal: {other:?}"),
        }
        w.tensor("a", &[0; 16]).unwrap();
        match w.finish() {
            Err(WriteError::Unfinished {
                next,
                left,
                declared,
            }) => assert_eq!((next.as_str(), left, declared), ("b", 1, 2)),
            other => panic!("wrong refusal: {:?}", other.map(|_| ())),
        }
        let mut w = Writer::new(Vec::new(), two()).unwrap();
        w.tensor("a", &[0; 16]).unwrap();
        w.tensor("b", &[0; 16]).unwrap();
        match w.tensor("a", &[0; 16]) {
            Err(WriteError::OutOfOrder { got, next: None }) => assert_eq!(got, "a"),
            other => panic!("wrong refusal: {other:?}"),
        }
        let header = two().data_base() as usize;
        let mut w = Writer::new(Full(header + 8), two()).unwrap();
        assert!(matches!(w.tensor("a", &[0; 16]), Err(WriteError::Io(_))));
        assert!(matches!(w.tensor("a", &[0; 16]), Err(WriteError::Broken)));
        assert!(matches!(w.finish(), Err(WriteError::Broken)));
    }
}
