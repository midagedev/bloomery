//! The body's sequence state saved to the host and put back: the live
//! sequence's every saved part as one value ([`SeqSnapshot`], [`snapshot`])
//! and back into the model ([`resume`]) — what the server's prompt cache
//! holds — and the same state as bytes ([`save_state`], [`restore_state`]),
//! what a slot file carries.
//!
//! A snapshot copies the sequence's own parts ([`super::Seq`]): per layer the
//! window ring whole, the compressed rows and index keys of the positions
//! held, the compressor state whole; the ring shadows' rows a cut may restore
//! (from `shadow_from`, outside the prompt calls' holes); the token history,
//! the slots' record, the holes and a pending ring restore. It names what it
//! belongs to: the model file, the stage card and a slot's positions
//! ([`Identity`]), and the slots the load's plan counts. A state of another
//! one, or laid out otherwise, is refused by name before any copy
//! ([`check`]), in either form.
//!
//! The byte form, written and read through the shared
//! [`bloomery_gpu::checkpoint::bytes`] (little-endian throughout; a count is
//! a u64, a name a u32 length and its bytes):
//!
//! | part | fields, in order |
//! |---|---|
//! | header | [`STATE_TAG`] (8 bytes), [`STATE_FORMAT`] (u32) |
//! | | the identity: the architecture, the first shard's path, the card (names); the context (count) |
//! | | the slots the load's plan counts; the positions `n` |
//! | | the layers' start and end; per layer the values of its ring, compressed rows, index keys, state values and state scores |
//! | | the shadow's width; the ring's slots; the state rings (a count, then each ratio) |
//! | | `shadow_from`; the holes, then the shadow's runs (each a count, then each start and end) |
//! | | the history's media spans (each a count, then each start and end) |
//! | values | the history: `n` ids (u32) |
//! | | the slots' record: per ring slot, then per state ring and slot, the position it holds (u64, `u64::MAX` for none) |
//! | | a pending ring restore (u8, 0 or 1) |
//! | | per layer: its ring, compressed rows and index keys (f16 bits, u16), its state values and scores (f32 bits) |
//! | | the shadow rows: per run, per layer, the run's rows (u16) |
//!
//! "Another build" is the bytes, not a commit: a state of this format and
//! layout reads the same whichever commit wrote it, and a change to any
//! part, its order or its encoding raises [`STATE_FORMAT`].

use std::ffi::OsStr;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::io::{self, Read, Write};
use std::ops::Range;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;

use bloomery_gpu::checkpoint::bytes::{self, Put, Take};
use bloomery_gpu::checkpoint::saved::Identity;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use cuda_core::{DeviceBuffer, DeviceCopy};

use super::seq::{History, keep_rule};
use super::{Body, Deepseek41Model, Holds, PAIR_ROWS, Shadows};
use crate::span::{span, span_mut};

const WHAT: &str = "deepseek41 sequence state";

/// The architecture a state names ([`Identity::arch`]).
const ARCH: &str = "deepseek41";

/// The byte form's first eight bytes.
const STATE_TAG: [u8; 8] = *b"BLMV41SQ";

/// The byte form's layout (the module table): raised whenever a part, its
/// order or its encoding changes.
const STATE_FORMAT: u32 = 2;

/// Why [`save_state`] or [`restore_state`] failed.
#[derive(Debug)]
pub enum StateFail {
    /// Refused by name: a state of another model file, build, layout or
    /// context, a stream that is not one, or a sequence that cannot be saved
    /// as it stands. A save leaves the sequence as it was; a restore may have
    /// read part of the stream, and the caller resets the sequence.
    Refused(GpuError),
    /// The card failed mid-copy: the model's error.
    Card(GpuError),
    /// The stream itself failed: a write or a read it refused (a closed
    /// pipe, a full disk) — neither the request's form nor the model's. A
    /// stream that ends early is a short stream, refused by name. A save
    /// leaves the sequence as it was; a restore touched nothing on the
    /// device, and the caller resets the sequence as for a refusal.
    Io(io::Error),
}

impl fmt::Display for StateFail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StateFail::Refused(e) => write!(f, "refused: {e}"),
            StateFail::Card(e) => write!(f, "the card failed: {e}"),
            StateFail::Io(e) => write!(f, "the stream failed: {e}"),
        }
    }
}

impl std::error::Error for StateFail {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StateFail::Refused(e) | StateFail::Card(e) => Some(e),
            StateFail::Io(e) => Some(e),
        }
    }
}

/// `e` as a failure of the byte form: a refusal by name
/// ([`GpuError::Shape`], [`GpuError::State`]) is the request's, every other
/// error the model's.
fn fail(e: GpuError) -> StateFail {
    match e {
        GpuError::Shape { .. } | GpuError::State { .. } => StateFail::Refused(e),
        e => StateFail::Card(e),
    }
}

/// A refusal by name of the byte form ([`StateFail::Refused`]).
fn refused(detail: String) -> StateFail {
    StateFail::Refused(refuse(detail))
}

/// A failure of the shared byte form as this state's: its refusal by name or
/// its stream's error, never the model's.
impl From<bytes::Fail> for StateFail {
    fn from(e: bytes::Fail) -> StateFail {
        match e {
            bytes::Fail::Refused(e) => StateFail::Refused(e),
            bytes::Fail::Io(e) => StateFail::Io(e),
        }
    }
}

/// The selected sequence's state ([`snapshot`]) written to `out` in the byte
/// form (the module table), part by part through one buffer, the sequence
/// unchanged; the bytes written. The host holds one snapshot and the buffer.
pub fn save_state(m: &mut Deepseek41Model, out: &mut dyn Write) -> Result<u64, StateFail> {
    let s = snapshot(m).map_err(fail)?;
    let mut w = Put::new(out);
    s.write(&mut w)?;
    w.flush()?;
    Ok(w.bytes())
}

/// The state `input` carries, which [`save_state`] wrote, put back into the
/// selected sequence as [`resume`] puts a snapshot back; the positions it
/// holds. `input` runs to the state's end. The header is read and checked
/// first ([`check`]), then the values, then the stream's end — a refusal in
/// any of them is [`StateFail::Refused`], and a failure of the stream
/// [`StateFail::Io`], with nothing on the device touched; a failure of the
/// put back is the model's.
pub fn restore_state(m: &mut Deepseek41Model, input: &mut dyn Read) -> Result<usize, StateFail> {
    let mut r = Take::new(input);
    let h = Header::read(&mut r)?;
    let (gpu, _, body) = m.body_parts(WHAT).map_err(fail)?;
    body.check_state(gpu, &h).map_err(fail)?;
    let s = SeqSnapshot::read(&mut r, h)?;
    r.end()?;
    resume(m, &s).map_err(fail)?;
    Ok(s.positions())
}

/// One layer's cache as saved: the ring whole, the compressed rows and index
/// keys of the positions held, the compressor state whole.
struct LayerSaved {
    ring: Vec<u16>,
    rows: Vec<u16>,
    keys: Vec<u16>,
    values: Vec<f32>,
    scores: Vec<f32>,
}

/// A body's sequence state on the host ([`snapshot`]) and what it belongs
/// to; [`resume`] puts it back into a body of the same identity and slot
/// count.
pub struct SeqSnapshot {
    /// The model file, the stage card and a slot's positions.
    who: Identity,
    /// The resident sequences the load's plan counts.
    slots: usize,
    layers: Range<usize>,
    kv: Vec<LayerSaved>,
    /// The shadow's width and its positions per layer, which the saved rows
    /// are laid out by.
    width: usize,
    /// The shadow positions saved, as runs; per run, per layer, its rows.
    runs: Vec<Range<usize>>,
    shadow: Vec<u16>,
    history: History,
    holds: Holds,
    shadow_from: usize,
    holes: Vec<Range<usize>>,
    restore: bool,
}

/// Every part a save copies, in the order [`snapshot`] lays it out, the f32
/// states by their bits, and what it belongs to: two snapshots hash alike
/// when a [`resume`] of either puts back the same sequence.
impl Hash for SeqSnapshot {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.who.arch.hash(h);
        self.who.file.hash(h);
        self.who.card.hash(h);
        self.who.ctx.hash(h);
        self.slots.hash(h);
        self.layers.hash(h);
        for l in &self.kv {
            l.ring.hash(h);
            l.rows.hash(h);
            l.keys.hash(h);
            for v in l.values.iter().chain(&l.scores) {
                v.to_bits().hash(h);
            }
        }
        self.width.hash(h);
        self.runs.hash(h);
        self.shadow.hash(h);
        self.history.hash(h);
        self.holds.hash(h);
        self.shadow_from.hash(h);
        self.holes.hash(h);
        self.restore.hash(h);
    }
}

impl SeqSnapshot {
    /// The positions it holds.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.history.len()
    }

    /// Its host bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        state_bytes(
            self.kv.iter().map(|l| {
                [
                    l.ring.len(),
                    l.rows.len(),
                    l.keys.len(),
                    l.values.len(),
                    l.scores.len(),
                ]
            }),
            self.shadow.len(),
            self.history.len(),
        )
        .expect("a saved state's host arrays fit the address space")
    }

    /// [`Body::keep_point`] of the body right after a [`resume`] of this
    /// state.
    #[must_use]
    pub fn keep_point(&self, n: usize) -> usize {
        keep_rule(
            self.history.len(),
            &self.holds,
            self.shadow_from,
            &self.holes,
            n,
        )
        .0
    }

    /// The positions whose ring rows a cut to `k` of the body right after a
    /// [`resume`] of this state restores from the shadows, as runs: those
    /// the step at `k` reads whose ring slot holds another position's row.
    #[must_use]
    pub fn restores(&self, k: usize) -> Vec<Range<usize>> {
        self.holds.stale_runs(k)
    }

    /// Its header: what it belongs to and the shape of every part.
    fn header(&self) -> Header {
        Header {
            who: self.who.clone(),
            slots: self.slots,
            positions: self.positions(),
            shape: Shape {
                layers: self.layers.clone(),
                kv: self
                    .kv
                    .iter()
                    .map(|l| {
                        [
                            l.ring.len(),
                            l.rows.len(),
                            l.keys.len(),
                            l.values.len(),
                            l.scores.len(),
                        ]
                    })
                    .collect(),
                width: self.width,
                ring: self.holds.ring.len(),
                ratios: self.holds.states.iter().map(|&(r, _)| r).collect(),
            },
            shadow_from: self.shadow_from,
            holes: self.holes.clone(),
            runs: self.runs.clone(),
            media: self
                .history
                .media()
                .iter()
                .map(|r| r.start as usize..r.end as usize)
                .collect(),
        }
    }

    /// The byte form (the module table) into `w`.
    fn write(&self, w: &mut Put<'_>) -> Result<(), StateFail> {
        self.header().write(w)?;
        w.vals(self.history.ids())?;
        let held = |q: &Option<usize>| q.map_or(u64::MAX, |q| q as u64);
        let ring: Vec<u64> = self.holds.ring.iter().map(held).collect();
        w.vals(&ring)?;
        for (_, s) in &self.holds.states {
            let s: Vec<u64> = s.iter().map(held).collect();
            w.vals(&s)?;
        }
        w.vals(&[u8::from(self.restore)])?;
        for l in &self.kv {
            w.vals(&l.ring)?;
            w.vals(&l.rows)?;
            w.vals(&l.keys)?;
            w.vals(&l.values)?;
            w.vals(&l.scores)?;
        }
        Ok(w.vals(&self.shadow)?)
    }

    /// The values after header `h` (the module table), read from `r`.
    fn read(r: &mut Take<'_>, h: Header) -> Result<SeqSnapshot, StateFail> {
        let ids = r.vals::<u32>(h.positions, "the history")?;
        let media = h
            .media
            .iter()
            .map(|r| {
                u32::try_from(r.start)
                    .and_then(|s| u32::try_from(r.end).map(|e| s..e))
                    .map_err(|_| refused(format!("a media span {r:?} of positions that pass u32")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut history = History::default();
        history
            .set_media(&ids, &media)
            .map_err(|e| refused(format!("{e}")))?;
        let held = |q: u64| -> Result<Option<usize>, StateFail> {
            if q == u64::MAX {
                return Ok(None);
            }
            usize::try_from(q)
                .map(Some)
                .map_err(|_| refused(format!("a slot that holds position {q}, past usize")))
        };
        let ring = r
            .vals::<u64>(h.shape.ring, "the ring slots' record")?
            .into_iter()
            .map(held)
            .collect::<Result<Vec<_>, _>>()?;
        let mut states = Vec::with_capacity(h.shape.ratios.len());
        for &ratio in &h.shape.ratios {
            let slots = r
                .vals::<u64>(ratio, "the state slots' record")?
                .into_iter()
                .map(held)
                .collect::<Result<Vec<_>, _>>()?;
            states.push((ratio, slots));
        }
        let restore = match r.one::<u8>("the pending restore")? {
            0 => false,
            1 => true,
            b => return Err(refused(format!("a pending restore of {b}: 0 or 1"))),
        };
        let mut kv = Vec::with_capacity(h.shape.kv.len());
        for &[ring, rows, keys, values, scores] in &h.shape.kv {
            kv.push(LayerSaved {
                ring: r.vals(ring, "a layer's ring")?,
                rows: r.vals(rows, "a layer's compressed rows")?,
                keys: r.vals(keys, "a layer's index keys")?,
                values: r.vals(values, "a layer's state values")?,
                scores: r.vals(scores, "a layer's state scores")?,
            });
        }
        let positions: usize = h.runs.iter().map(Range::len).sum();
        let shadow = r.vals(
            positions * h.shape.kv.len() * h.shape.width,
            "the shadow rows",
        )?;
        Ok(SeqSnapshot {
            who: h.who,
            slots: h.slots,
            layers: h.shape.layers,
            kv,
            width: h.shape.width,
            runs: h.runs,
            shadow,
            history,
            holds: Holds { ring, states },
            shadow_from: h.shadow_from,
            holes: h.holes,
            restore,
        })
    }
}

/// The shape of every part of a state, as a body of a given layout holds it
/// at a state's positions.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Shape {
    layers: Range<usize>,
    /// Per layer: the values of its ring, compressed rows, index keys, state
    /// values and state scores.
    kv: Vec<[usize; 5]>,
    /// The shadow's width.
    width: usize,
    /// The ring's slots the slots' record holds.
    ring: usize,
    /// Each state ring's ratio, its slots.
    ratios: Vec<usize>,
}

impl Shape {
    /// The first part in which a state's shape and the body's differ, in
    /// words; `None` when they are one layout.
    fn differs(&self, body: &Shape) -> Option<String> {
        if self.layers != body.layers || self.kv.len() != body.kv.len() {
            return Some(format!(
                "layers {:?} ({} of them), the body's {:?} ({})",
                self.layers,
                self.kv.len(),
                body.layers,
                body.kv.len()
            ));
        }
        let layer = self.kv.iter().zip(&body.kv).position(|(a, b)| a != b);
        if let Some(i) = layer {
            return Some(format!(
                "layer {}'s ring, compressed rows, index keys, state values and scores of {:?} \
                 values, the body's {:?}",
                self.layers.start + i,
                self.kv[i],
                body.kv[i]
            ));
        }
        if (self.width, self.ring) != (body.width, body.ring) {
            return Some(format!(
                "a shadow {} wide and {} ring slots, the body's {} and {}",
                self.width, self.ring, body.width, body.ring
            ));
        }
        (self.ratios != body.ratios).then(|| {
            format!(
                "state rings of ratios {:?}, the body's {:?}",
                self.ratios, body.ratios
            )
        })
    }
}

/// The byte form's header (the module table): what a state belongs to and
/// the shape of every part, read before any value.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Header {
    who: Identity,
    slots: usize,
    positions: usize,
    shape: Shape,
    shadow_from: usize,
    holes: Vec<Range<usize>>,
    runs: Vec<Range<usize>>,
    /// The history's media spans, of the positions the history holds.
    media: Vec<Range<usize>>,
}

impl Header {
    fn write(&self, w: &mut Put<'_>) -> Result<(), StateFail> {
        w.vals(&STATE_TAG)?;
        w.vals(&[STATE_FORMAT])?;
        w.name(self.who.arch.as_bytes())?;
        w.name(self.who.file.as_os_str().as_bytes())?;
        w.name(self.who.card.as_bytes())?;
        w.count(self.who.ctx)?;
        w.count(self.slots)?;
        w.count(self.positions)?;
        let s = &self.shape;
        w.count(s.layers.start)?;
        w.count(s.layers.end)?;
        for &c in s.kv.iter().flatten() {
            w.count(c)?;
        }
        w.count(s.width)?;
        w.count(s.ring)?;
        w.count(s.ratios.len())?;
        for &r in &s.ratios {
            w.count(r)?;
        }
        w.count(self.shadow_from)?;
        put_runs(w, &self.holes)?;
        put_runs(w, &self.runs)?;
        put_runs(w, &self.media)
    }

    /// The header `r` starts with: refused by name when the stream ends
    /// inside it, or when its tag, format or architecture is not this
    /// reader's — whatever follows those is another layout.
    fn read(r: &mut Take<'_>) -> Result<Header, StateFail> {
        let tag = r.vals::<u8>(STATE_TAG.len(), "the tag")?;
        let format: u32 = r.one("the format")?;
        if tag != STATE_TAG || format != STATE_FORMAT {
            return Err(refused(format!(
                "a stream of tag {tag:02x?} and format {format}; this build reads tag \
                 {STATE_TAG:02x?} and format {STATE_FORMAT}"
            )));
        }
        let arch = r.name("the architecture")?;
        if arch != ARCH.as_bytes() {
            return Err(refused(format!(
                "a state of architecture {:?}; this body is {ARCH}",
                String::from_utf8_lossy(&arch)
            )));
        }
        let file = PathBuf::from(OsStr::from_bytes(&r.name("the model file")?));
        let card = String::from_utf8(r.name("the card")?)
            .map_err(|e| refused(format!("a card name that is not UTF-8: {e}")))?;
        let ctx = r.count("the context")?;
        let who = Identity {
            arch: ARCH,
            file,
            card,
            ctx,
        };
        let slots = r.count("the slot count")?;
        let positions = r.count("the positions")?;
        let start = r.count("the layers")?;
        let layers = start..r.count("the layers")?;
        let mut kv = Vec::new();
        for _ in 0..layers.len() {
            let mut part = [0; 5];
            for c in &mut part {
                *c = r.count("the layers' parts")?;
            }
            kv.push(part);
        }
        let width = r.count("the shadow width")?;
        let ring = r.count("the ring slots")?;
        let rings = r.count("the state rings")?;
        let ratios = (0..rings)
            .map(|_| r.count("the state rings"))
            .collect::<Result<Vec<_>, _>>()?;
        let shadow_from = r.count("the shadow's start")?;
        let holes = take_runs(r, "the holes")?;
        let runs = take_runs(r, "the shadow's runs")?;
        let media = take_runs(r, "the history's media spans")?;
        Ok(Header {
            who,
            slots,
            positions,
            shape: Shape {
                layers,
                kv,
                width,
                ring,
                ratios,
            },
            shadow_from,
            holes,
            runs,
            media,
        })
    }
}

/// Whether a body takes a state of header `h`: the body is `who`, its
/// load's plan counts `slots`, and `shape` gives its layout at a state's
/// positions. Refused by name at the first that differs, in this order: the
/// owner ([`same_owner`]), the positions against the context, the shadow's
/// runs against the holes, the layout.
fn check(
    h: &Header,
    who: &Identity,
    slots: usize,
    shape: impl FnOnce(usize) -> Result<Shape, GpuError>,
) -> Result<(), GpuError> {
    same_owner(h, who, slots)?;
    let n = h.positions;
    if n > who.ctx {
        return Err(refuse(format!(
            "a state of {n} positions restored into a slot of {} positions",
            who.ctx
        )));
    }
    let holes = h.shadow_from <= n
        && h.holes.iter().all(|r| r.start < r.end && r.end <= n)
        && h.holes.windows(2).all(|w| w[0].end <= w[1].start);
    if !holes || h.runs != kept_runs(h.shadow_from, n, &h.holes) {
        return Err(refuse(format!(
            "a state of {n} positions whose shadow runs {:?} are not those from {} outside its \
             holes {:?}",
            h.runs, h.shadow_from, h.holes
        )));
    }
    if let Some(d) = h.shape.differs(&shape(n)?) {
        return Err(refuse(format!(
            "a state of {n} positions laid out as another body's: {d}"
        )));
    }
    Ok(())
}

/// Refused by name unless a state of header `h` belongs to a body that is
/// `who` and whose load's plan counts `slots`.
fn same_owner(h: &Header, who: &Identity, slots: usize) -> Result<(), GpuError> {
    if h.who != *who {
        let (a, b) = (&h.who, who);
        let differs = [
            (a.arch != b.arch).then(|| "another architecture".to_string()),
            (a.file != b.file).then(|| "another model file".to_string()),
            (a.card != b.card).then(|| "another card".to_string()),
            (a.ctx != b.ctx)
                .then(|| format!("a context of {} positions, the load's {}", a.ctx, b.ctx)),
        ];
        let differs: Vec<String> = differs.into_iter().flatten().collect();
        return Err(refuse(format!(
            "a state of {a} restored on {b}: {}",
            differs.join(", ")
        )));
    }
    if h.slots != slots {
        return Err(refuse(format!(
            "a state of a load whose plan counts {} slots restored on one whose plan counts \
             {slots}",
            h.slots
        )));
    }
    Ok(())
}

fn refuse(detail: String) -> GpuError {
    GpuError::Shape { what: WHAT, detail }
}

/// `runs` in the byte form: their count, then each run's start and end.
fn put_runs(w: &mut Put<'_>, runs: &[Range<usize>]) -> Result<(), StateFail> {
    w.count(runs.len())?;
    for r in runs {
        w.count(r.start)?;
        w.count(r.end)?;
    }
    Ok(())
}

/// What [`put_runs`] wrote.
fn take_runs(r: &mut Take<'_>, part: &str) -> Result<Vec<Range<usize>>, StateFail> {
    let n = r.count(part)?;
    let mut runs = Vec::new();
    for _ in 0..n {
        let start = r.count(part)?;
        runs.push(start..r.count(part)?);
    }
    Ok(runs)
}

/// `len` values of host memory, refused by name when the allocation fails.
fn host_vec<T: Clone + Default>(len: usize) -> Result<Vec<T>, GpuError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|e| GpuError::Shape {
        what: WHAT,
        detail: format!("{len} values of host memory for a snapshot: {e}"),
    })?;
    v.resize(len, T::default());
    Ok(v)
}

/// The first `len` values of `buf`, device to host (a blocking copy).
fn read<T: DeviceCopy + Clone + Default>(
    gpu: &Gpu,
    buf: &DeviceBuffer<T>,
    len: usize,
) -> Result<Vec<T>, GpuError> {
    let mut v = host_vec(len)?;
    if len > 0 {
        span(WHAT, buf, 0, len)?.copy_to_host(gpu.stream(), &mut v)?;
    }
    Ok(v)
}

/// [`read`] of a buffer the layer may not hold: none read when it does not.
fn read_held<T: DeviceCopy + Clone + Default>(
    gpu: &Gpu,
    t: Option<&DeviceTensor<T>>,
    len: usize,
) -> Result<Vec<T>, GpuError> {
    t.map_or_else(|| Ok(Vec::new()), |t| read(gpu, t.buf(), len))
}

/// `vals` into the first values of `buf`, host to device (a blocking copy).
fn write<T: DeviceCopy>(gpu: &Gpu, buf: &mut DeviceBuffer<T>, vals: &[T]) -> Result<(), GpuError> {
    if vals.is_empty() {
        return Ok(());
    }
    span_mut(WHAT, buf, 0, vals.len())?.copy_from_host(gpu.stream(), vals)?;
    Ok(())
}

impl Shadows {
    /// Layer `i`'s shadow rows `rows` appended to `out`, after the stream
    /// finished every append (the caller synchronized): a copy from the host
    /// allocation, or a blocking device-to-host copy of the room's rows.
    pub(super) fn read_rows_into(
        &self,
        gpu: &Gpu,
        i: usize,
        rows: Range<usize>,
        out: &mut Vec<u16>,
    ) -> Result<(), GpuError> {
        let (width, per) = (self.width(), self.rows());
        let refuse = || GpuError::Shape {
            what: WHAT,
            detail: format!("shadow {i} rows {rows:?} of shadows of {per} rows of {width}"),
        };
        if rows.end > per || rows.start > rows.end {
            return Err(refuse());
        }
        if rows.is_empty() {
            return Ok(());
        }
        match self {
            Shadows::Pinned(p) => {
                let at = (i * per + rows.start) * width;
                let src = p.host.get(at..at + rows.len() * width).ok_or_else(refuse)?;
                out.extend_from_slice(src);
            }
            Shadows::Room(r) => {
                let t = r.layers.get(i).ok_or_else(refuse)?;
                out.extend_from_slice(&read_vec(
                    gpu,
                    t.buf(),
                    rows.start * width,
                    rows.len() * width,
                )?);
            }
        }
        Ok(())
    }

    /// `vals` into layer `i`'s shadow from row `row` on, the stream idle
    /// (the caller synchronized): a copy into the host allocation, or a
    /// blocking host-to-device copy into the room, whose rows the caller
    /// fitted.
    pub(super) fn write_rows(
        &mut self,
        gpu: &Gpu,
        i: usize,
        row: usize,
        vals: &[u16],
    ) -> Result<(), GpuError> {
        let (width, per) = (self.width(), self.rows());
        let refuse = || GpuError::Shape {
            what: WHAT,
            detail: format!(
                "shadow {i} from row {row}, {} values, of shadows of {per} rows of {width}",
                vals.len()
            ),
        };
        if vals.is_empty() {
            return Ok(());
        }
        if row * width + vals.len() > per * width {
            return Err(refuse());
        }
        match self {
            Shadows::Pinned(p) => {
                let at = (i * per + row) * width;
                p.host
                    .as_mut_slice()
                    .get_mut(at..at + vals.len())
                    .ok_or_else(refuse)?
                    .copy_from_slice(vals);
            }
            Shadows::Room(r) => {
                let t = r.layers.get_mut(i).ok_or_else(refuse)?;
                span_mut(WHAT, t.buf_mut(), row * width, vals.len())?
                    .copy_from_host(gpu.stream(), vals)?;
            }
        }
        Ok(())
    }
}

/// `len` values of `buf` from value `off`, device to host (a blocking copy).
fn read_vec<T: DeviceCopy + Clone + Default>(
    gpu: &Gpu,
    buf: &DeviceBuffer<T>,
    off: usize,
    len: usize,
) -> Result<Vec<T>, GpuError> {
    let mut v = host_vec(len)?;
    span(WHAT, buf, off, len)?.copy_to_host(gpu.stream(), &mut v)?;
    Ok(v)
}

impl Body {
    /// What a state of this body's sequences belongs to: the model file's
    /// first shard, the stage card, a slot's positions.
    fn identity(&self, gpu: &Gpu) -> Result<Identity, GpuError> {
        Identity::of_load(ARCH, &self.file, gpu, self.positions())
    }

    /// The shape of every part of a state of `n` positions, as this body
    /// holds it: per layer the ring and the compressor state whole, the
    /// compressed rows and index keys of `⌈n / ratio⌉` rows.
    fn shape(&self, n: usize) -> Result<Shape, GpuError> {
        let kv = self
            .seq
            .kv
            .iter()
            .zip(self.layers.clone())
            .map(|(l, layer)| {
                let held = |t: &DeviceTensor<u16>| {
                    let ratio = self.hp.layers.get(layer).map_or(0, |k| k.ratio() as usize);
                    if ratio == 0 {
                        return Err(GpuError::Shape {
                            what: WHAT,
                            detail: format!("layer {layer} holds compressed rows and has no ratio"),
                        });
                    }
                    Ok(n.div_ceil(ratio).min(t.rows()) * t.cols())
                };
                let whole = |t: Option<&DeviceTensor<f32>>| t.map_or(0, |t| t.buf().len());
                Ok([
                    l.ring.buf().len(),
                    l.rows.as_deref().map_or(Ok(0), held)?,
                    l.keys.as_deref().map_or(Ok(0), held)?,
                    whole(l.values.as_ref()),
                    whole(l.scores.as_ref()),
                ])
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(Shape {
            layers: self.layers.clone(),
            kv,
            width: self.seq.shadows.width(),
            ring: self.seq.ring_rows(),
            ratios: self.seq.holds.states.iter().map(|&(r, _)| r).collect(),
        })
    }

    /// Whether this body takes a state of header `h` ([`check`]).
    fn check_state(&self, gpu: &Gpu, h: &Header) -> Result<(), GpuError> {
        check(h, &self.identity(gpu)?, self.slots_planned, |n| {
            self.shape(n)
        })
    }

    /// The sequence state to the host: the rows in flight delivered, the
    /// stream finished, then every saved part copied (module comment).
    fn save(&mut self, gpu: &Gpu) -> Result<SeqSnapshot, GpuError> {
        self.arrive()?;
        self.rows.finish()?;
        if self.seq.rows_failed {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a reset: an earlier step's engram rows failed after its launch",
            });
        }
        gpu.stream().synchronize()?;
        let n = self.seq.history.len();
        let shape = self.shape(n)?;
        let mut kv = Vec::with_capacity(self.seq.kv.len());
        for (l, &[ring, rows, keys, values, scores]) in self.seq.kv.iter().zip(&shape.kv) {
            kv.push(LayerSaved {
                ring: read(gpu, l.ring.buf(), ring)?,
                rows: read_held(gpu, l.rows.as_deref(), rows)?,
                keys: read_held(gpu, l.keys.as_deref(), keys)?,
                values: read_held(gpu, l.values.as_ref(), values)?,
                scores: read_held(gpu, l.scores.as_ref(), scores)?,
            });
        }
        let runs = kept_runs(self.seq.shadow_from, n, &self.seq.holes);
        let width = self.seq.shadows.width();
        let layers = self.seq.kv.len();
        let total = runs.iter().map(|r| r.len()).sum::<usize>() * layers * width;
        let mut shadow: Vec<u16> = Vec::new();
        shadow
            .try_reserve_exact(total)
            .map_err(|e| GpuError::Shape {
                what: WHAT,
                detail: format!("{total} shadow values of host memory for a snapshot: {e}"),
            })?;
        for r in &runs {
            for i in 0..layers {
                self.seq
                    .shadows
                    .read_rows_into(gpu, i, r.clone(), &mut shadow)?;
            }
        }
        Ok(SeqSnapshot {
            who: self.identity(gpu)?,
            slots: self.slots_planned,
            layers: self.layers.clone(),
            kv,
            width,
            runs,
            shadow,
            history: self.seq.history.clone(),
            holds: self.seq.holds.clone(),
            shadow_from: self.seq.shadow_from,
            holes: self.seq.holes.clone(),
            restore: self.seq.restore,
        })
    }

    /// `s` back into a body that was just reset: refused by name, before any
    /// copy, unless this body takes it ([`check`]: its identity, slot count,
    /// positions and layout); then every saved part copied in and the host
    /// record set to the state's.
    fn load(&mut self, gpu: &Gpu, s: &SeqSnapshot) -> Result<(), GpuError> {
        self.check_state(gpu, &s.header())?;
        // The saved rows land on pages of this sequence's own: the held positions' rooms are
        // backed before the copies, outside any capture.
        self.fit_on(None, 0, s.history.len())?;
        for (l, saved) in self.seq.kv.iter_mut().zip(&s.kv) {
            write(gpu, l.ring.buf_mut(), &saved.ring)?;
            if let Some(t) = l.rows.as_mut() {
                write(gpu, t.buf_mut(), &saved.rows)?;
            }
            if let Some(t) = l.keys.as_mut() {
                write(gpu, t.buf_mut(), &saved.keys)?;
            }
            if let Some(t) = l.values.as_mut() {
                write(gpu, t.buf_mut(), &saved.values)?;
            }
            if let Some(t) = l.scores.as_mut() {
                write(gpu, t.buf_mut(), &saved.scores)?;
            }
        }
        // No step in flight writes a shadow row while the host does.
        gpu.stream().synchronize()?;
        let (width, layers) = (self.seq.shadows.width(), self.seq.kv.len());
        let mut from = 0;
        for r in &s.runs {
            for i in 0..layers {
                let len = r.len() * width;
                self.seq
                    .shadows
                    .write_rows(gpu, i, r.start, &s.shadow[from..from + len])?;
                from += len;
            }
        }
        self.seq.history.clone_from(&s.history);
        self.seq.holds.clone_from(&s.holds);
        self.seq.shadow_from = s.shadow_from;
        self.seq.holes.clone_from(&s.holes);
        self.seq.restore = s.restore;
        self.need = None;
        if let Some(tap) = self.tap.as_mut() {
            tap.pos = [None; PAIR_ROWS];
        }
        Ok(())
    }
}

/// The positions `shadow_from .. n` outside `holes` (ascending, disjoint), as
/// runs.
fn kept_runs(shadow_from: usize, n: usize, holes: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut p = shadow_from;
    for h in holes {
        if h.start > p {
            runs.push(p..h.start.min(n));
        }
        p = p.max(h.end);
    }
    if p < n {
        runs.push(p..n);
    }
    runs.retain(|r| !r.is_empty());
    runs
}

/// `m`'s sequence state on the host. Refused on a poisoned model: its caches
/// hold what a fault condemned.
pub fn snapshot(m: &mut Deepseek41Model) -> Result<SeqSnapshot, GpuError> {
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!("a snapshot of a model a fault poisoned ({fault:?})"),
        });
    }
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.save(gpu)
}

fn state_bytes(
    kv: impl IntoIterator<Item = [usize; 5]>,
    shadow: usize,
    history: usize,
) -> Option<usize> {
    kv.into_iter()
        .try_fold(0usize, |n, [ring, rows, keys, values, scores]| {
            let f16 = ring.checked_add(rows)?.checked_add(keys)?.checked_mul(2)?;
            let f32 = values.checked_add(scores)?.checked_mul(4)?;
            n.checked_add(f16)?.checked_add(f32)
        })?
        .checked_add(shadow.checked_mul(2)?)?
        .checked_add(history.checked_mul(4)?)
}

/// The host array bytes a snapshot holds, without reading the card.
pub fn snapshot_size(m: &mut Deepseek41Model) -> Result<u64, GpuError> {
    let (_, _, b) = m.body_parts(WHAT)?;
    let n = b.seq.history.len();
    let shape = b.shape(n)?;
    let shadow = kept_runs(b.seq.shadow_from, n, &b.seq.holes)
        .iter()
        .try_fold(0usize, |v, r| v.checked_add(r.len()))
        .and_then(|v| v.checked_mul(b.seq.kv.len()))
        .and_then(|v| v.checked_mul(b.seq.shadows.width()));
    shadow
        .and_then(|shadow| state_bytes(shape.kv, shadow, n))
        .map(|bytes| bytes as u64)
        .ok_or_else(|| GpuError::shape(WHAT, "snapshot host bytes pass the address space"))
}

/// Replace `m`'s sequence state with `s`, which [`snapshot`] took of a model
/// of the same identity and slot count (refused by name otherwise, before
/// any copy): a reset, then — as a pass of `s.positions()` positions whose
/// work is the copies — the state put back; the model stands at
/// `s.positions()`, and the steps after it are bit for bit those after the
/// state was taken.
pub fn resume(m: &mut Deepseek41Model, s: &SeqSnapshot) -> Result<(), GpuError> {
    m.reset()?;
    let n = s.positions();
    if n == 0 {
        return Ok(());
    }
    m.run_rows(n, WHAT, |gpu, _, body, _, _| {
        body.load(gpu, s)?;
        Ok(false)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A header of two layers, the second without a compressor, 40 positions
    /// whose two prompt calls each left a hole.
    fn header() -> Header {
        let holes = vec![0..8, 16..20];
        Header {
            who: Identity {
                arch: ARCH,
                file: PathBuf::from("/models/v41/V4.1-Flash-00001-of-00009.gguf"),
                card: "card".to_owned(),
                ctx: 64,
            },
            slots: 2,
            positions: 40,
            shape: Shape {
                layers: 3..5,
                kv: vec![[16, 10, 4, 6, 6], [16, 0, 0, 0, 0]],
                width: 8,
                ring: 4,
                ratios: vec![2],
            },
            shadow_from: 0,
            runs: kept_runs(0, 40, &holes),
            media: vec![4..12, 30..31],
            holes,
        }
    }

    fn bytes(h: &Header) -> Vec<u8> {
        let mut out = Vec::new();
        let written = {
            let mut w = Put::new(&mut out);
            h.write(&mut w).expect("a header into memory");
            w.flush().expect("a flush into memory");
            w.bytes()
        };
        assert_eq!(written, out.len() as u64);
        out
    }

    fn read_header(b: &[u8]) -> Result<Header, StateFail> {
        let mut input = b;
        Header::read(&mut Take::new(&mut input))
    }

    /// `r` is a refusal by name whose words hold `says`.
    fn named(r: Result<impl fmt::Debug, StateFail>, says: &str) {
        match r {
            Err(StateFail::Refused(e @ GpuError::Shape { .. })) => {
                assert!(e.to_string().contains(says), "{e} does not say {says:?}");
            }
            r => panic!("{r:?}: not refused by name ({says:?})"),
        }
    }

    /// [`named`] of [`check`]'s answer.
    fn named_check(r: Result<(), GpuError>, says: &str) {
        named(r.map_err(StateFail::Refused), says);
    }

    /// A writer that takes `left` bytes, then fails as a full disk does.
    struct FullAfter {
        left: usize,
    }

    impl Write for FullAfter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.left == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::StorageFull,
                    "the disk is full",
                ));
            }
            let n = buf.len().min(self.left);
            self.left -= n;
            Ok(n)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// A reader that hands `bytes`, then fails as a closed pipe does.
    struct BrokenAfter<'a> {
        bytes: &'a [u8],
    }

    impl Read for BrokenAfter<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.bytes.is_empty() {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "the pipe closed"));
            }
            self.bytes.read(buf)
        }
    }

    #[test]
    fn kept_runs_skip_the_holes() {
        let first = 0..40;
        let one = std::slice::from_ref(&first);
        assert_eq!(kept_runs(0, 100, &[]), vec![0..100]);
        assert_eq!(kept_runs(0, 100, one), vec![40..100]);
        assert_eq!(kept_runs(10, 100, &[0..40, 60..70]), vec![40..60, 70..100]);
        assert_eq!(kept_runs(50, 100, &[0..40, 60..70]), vec![50..60, 70..100]);
        assert_eq!(kept_runs(0, 30, one), Vec::<std::ops::Range<usize>>::new());
    }

    /// A header written and read back is itself, every byte of it read and
    /// the stream's end there.
    #[test]
    fn a_header_goes_and_comes_back() {
        let h = header();
        let b = bytes(&h);
        let mut input = b.as_slice();
        let mut r = Take::new(&mut input);
        assert_eq!(Header::read(&mut r).expect("its own header"), h);
        assert_eq!(r.bytes(), b.len() as u64);
        r.end().expect("the stream ends with the header");
    }

    /// Another tag or another format is refused by name, and so is a byte
    /// past the state.
    #[test]
    fn another_tag_or_format_is_refused() {
        let b = bytes(&header());
        let mut tag = b.clone();
        tag[0] ^= 1;
        named(read_header(&tag), "a stream of tag");
        let mut format = b.clone();
        format[8..12].copy_from_slice(&(STATE_FORMAT + 1).to_le_bytes());
        named(
            read_header(&format),
            &format!("format {};", STATE_FORMAT + 1),
        );
        let mut past = b;
        past.push(0);
        let mut input = past.as_slice();
        let mut r = Take::new(&mut input);
        Header::read(&mut r).expect("its own header");
        named(
            r.end().map_err(StateFail::from),
            "trailing bytes past the state's",
        );
    }

    /// A header cut anywhere is a short stream, refused by name.
    #[test]
    fn a_cut_header_is_a_short_stream() {
        let b = bytes(&header());
        for cut in 0..b.len() {
            named(read_header(&b[..cut]), "a short stream: it ends inside");
        }
    }

    /// A stream that fails is the stream's failure, never the card's: a
    /// writer that fails after some bytes, at any of them, and a reader that
    /// fails inside the header or at its end.
    #[test]
    fn a_failing_stream_is_io() {
        let h = header();
        let b = bytes(&h);
        for left in [0, 10, b.len() - 1] {
            let mut out = FullAfter { left };
            let mut w = Put::new(&mut out);
            let r = h.write(&mut w).and_then(|()| Ok(w.flush()?));
            assert!(
                matches!(&r, Err(StateFail::Io(e)) if e.kind() == io::ErrorKind::StorageFull),
                "a write failing after {left} bytes: {r:?}"
            );
        }
        for at in [0, 30, b.len() - 1] {
            let mut input = BrokenAfter { bytes: &b[..at] };
            let r = Header::read(&mut Take::new(&mut input));
            assert!(
                matches!(&r, Err(StateFail::Io(e)) if e.kind() == io::ErrorKind::BrokenPipe),
                "a read failing after {at} bytes: {r:?}"
            );
        }
        let mut input = BrokenAfter { bytes: &b };
        let mut r = Take::new(&mut input);
        Header::read(&mut r).expect("its own header");
        assert!(matches!(r.end(), Err(bytes::Fail::Io(_))));
    }

    /// The owner, the positions, the runs and the layout: each other one is
    /// refused by name, the context's naming both counts, and the body's own
    /// header taken.
    #[test]
    fn another_owner_or_layout_is_refused() {
        let h = header();
        let shape = |_: usize| Ok(header().shape);
        let (who, slots) = (h.who.clone(), h.slots);
        check(&h, &who, slots, shape).expect("its own body takes it");
        let file = Identity {
            file: PathBuf::from("/models/other.gguf"),
            ..who.clone()
        };
        named_check(
            check(&h, &file, slots, shape),
            "restored on deepseek41 /models/other.gguf on card at 64 positions: another model file",
        );
        let card = Identity {
            card: "another card".to_owned(),
            ..who.clone()
        };
        named_check(check(&h, &card, slots, shape), ": another card");
        let ctx = Identity {
            ctx: 65,
            ..who.clone()
        };
        named_check(
            check(&h, &ctx, slots, shape),
            "a context of 64 positions, the load's 65",
        );
        named_check(check(&h, &who, 3, shape), "plan counts 2 slots restored");
        let past = Header {
            positions: 65,
            ..h.clone()
        };
        named_check(
            check(&past, &who, slots, shape),
            "65 positions restored into",
        );
        let runs = Header {
            runs: vec![0..20, 20..40],
            ..h.clone()
        };
        named_check(check(&runs, &who, slots, shape), "are not those from 0");
        let mut layout = header().shape;
        layout.kv[1][0] = 32;
        named_check(
            check(&h, &who, slots, |_| Ok(layout.clone())),
            "layer 4's ring",
        );
    }
}
