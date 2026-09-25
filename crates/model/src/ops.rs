//! The two primitives every module needs, written once — reference
//! implementations, fast path in `crates/q3k-cpu`. Accumulation is f32 in
//! ggml's own order (one row at a time, k ascending): a different order moves
//! the last bit, which is what lets the gates stay tight.

use crate::ffn::swiglu_timed;
use crate::profile;
use gguf::{GgmlType, Gguf, Split, TensorInfo, dequant_row, quantize_activations};
use std::cell::RefCell;
use std::ops::Range;
use std::time::Instant;

/// A 2-D activation block in ggml's layout: `ne0` contiguous, `ne1` strides by
/// `ne0`; token `t` lives at `data[t*ne0..(t+1)*ne0]` (oracle, `docs/oracle.md`).
///
/// Buffers are recycled: `Drop` hands `data` to a per-thread free list and the
/// constructors take from it, so a steady-state step does not go to malloc for
/// activations (`tests/alloc.rs` pins the count).
#[derive(PartialEq, Debug)]
pub struct Tensor2 {
    pub ne0: usize,
    pub ne1: usize,
    pub data: Vec<f32>,
}

thread_local! {
    /// Per-thread recycled `Vec<f32>` storage, bucketed by size class: the
    /// activations (`Tensor2`) and the scalar-path quantization buffers are the
    /// same shape of allocation, and a step cycles several sizes at once — a
    /// flat list that fills rejects every later drop and starves exactly the
    /// sizes it exists to serve.
    static F32_BLOCKS: RefCell<Vec<Vec<Vec<f32>>>> = const { RefCell::new(Vec::new()) };
    /// The byte-shaped sibling, for the fused path's quantized columns.
    static U8_BLOCKS: RefCell<Vec<Vec<Vec<u8>>>> = const { RefCell::new(Vec::new()) };
}
/// Free-list depth per size class. One decode step's working set fits: the
/// widest class holds the sixteen per-head activation blocks `wv_b_heads`
/// gathers, plus headroom for transients.
const CLASS_DEPTH: usize = 32;

/// The bucket a capacity of `n` lives in: the power-of-two floor. A take's
/// capacity window `[n, 2n)` never crosses more than one bucket boundary, so a
/// search walks at most two.
fn class_of(n: usize) -> usize {
    n.ilog2() as usize
}

fn take_f32(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let hit = F32_BLOCKS
        .try_with(|p| {
            let mut p = p.borrow_mut();
            let c = class_of(n);
            [c, c + 1].into_iter().find_map(|cls| {
                p.get_mut(cls).and_then(|b| {
                    let i = b
                        .iter()
                        .position(|b| b.capacity() >= n && b.capacity() <= 2 * n)?;
                    Some(b.swap_remove(i))
                })
            })
        })
        .ok()
        .flatten();
    match hit {
        Some(mut b) => {
            // Stale cells stay: only a block shorter than `n` is extended.
            if b.len() >= n {
                b.truncate(n);
            } else {
                b.resize(n, 0.0);
            }
            b
        }
        None => vec![0.0f32; n],
    }
}

fn give_f32(b: Vec<f32>) {
    if b.capacity() == 0 {
        return;
    }
    let _ = F32_BLOCKS.try_with(|p| {
        let mut p = p.borrow_mut();
        let cls = class_of(b.capacity());
        if p.len() <= cls {
            p.resize_with(cls + 1, Vec::new);
        }
        if p[cls].len() < CLASS_DEPTH {
            p[cls].push(b);
        }
    });
}

fn take_u8(n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    let hit = U8_BLOCKS
        .try_with(|p| {
            let mut p = p.borrow_mut();
            let c = class_of(n);
            [c, c + 1].into_iter().find_map(|cls| {
                p.get_mut(cls).and_then(|b| {
                    let i = b
                        .iter()
                        .position(|b| b.capacity() >= n && b.capacity() <= 2 * n)?;
                    Some(b.swap_remove(i))
                })
            })
        })
        .ok()
        .flatten();
    match hit {
        Some(mut b) => {
            // Stale cells stay: only a block shorter than `n` is extended.
            if b.len() >= n {
                b.truncate(n);
            } else {
                b.resize(n, 0);
            }
            b
        }
        None => vec![0u8; n],
    }
}

fn give_u8(b: Vec<u8>) {
    if b.capacity() == 0 {
        return;
    }
    let _ = U8_BLOCKS.try_with(|p| {
        let mut p = p.borrow_mut();
        let cls = class_of(b.capacity());
        if p.len() <= cls {
            p.resize_with(cls + 1, Vec::new);
        }
        if p[cls].len() < CLASS_DEPTH {
            p[cls].push(b);
        }
    });
}

/// `BLOOMERY_POISON=1` fills every `scratch` block with NaN, so a kernel that
/// leaves a cell unwritten fails the gates loudly instead of reading a stale value.
fn poison() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var("BLOOMERY_POISON").is_ok_and(|v| v != "0"))
}

/// A block of `n` f32 off the free list, contents unspecified (stale, never
/// uninitialized). The capacity window keeps a logits-sized block from being spent
/// on a 2048-wide request.
fn take_block(n: usize) -> Vec<f32> {
    take_f32(n)
}

impl Drop for Tensor2 {
    fn drop(&mut self) {
        // `try_with`: a block dropped during thread teardown just frees.
        give_f32(std::mem::take(&mut self.data));
    }
}

impl Clone for Tensor2 {
    fn clone(&self) -> Self {
        let mut out = Self::scratch(self.ne0, self.ne1);
        out.data.copy_from_slice(&self.data);
        out
    }
}

impl Tensor2 {
    pub fn zeros(ne0: usize, ne1: usize) -> Self {
        let mut data = take_block(ne0 * ne1);
        data.fill(0.0);
        Self { ne0, ne1, data }
    }

    /// A block whose every cell the caller is about to write; reading one first
    /// is a bug (`BLOOMERY_POISON` makes it a NaN).
    pub fn scratch(ne0: usize, ne1: usize) -> Self {
        let mut data = take_block(ne0 * ne1);
        if poison() {
            data.fill(f32::NAN);
        }
        Self { ne0, ne1, data }
    }

    pub fn from_vec(ne0: usize, ne1: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            ne0 * ne1,
            "Tensor2 data length must be ne0 * ne1"
        );
        Self { ne0, ne1, data }
    }

    /// The values, leaving the free list out of it (`Drop` forbids moving `data` out).
    pub fn into_data(mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }

    /// Column `i`, i.e. token `i`'s `ne0` contiguous values.
    pub fn col(&self, i: usize) -> &[f32] {
        &self.data[i * self.ne0..(i + 1) * self.ne0]
    }

    pub fn col_mut(&mut self, i: usize) -> &mut [f32] {
        &mut self.data[i * self.ne0..(i + 1) * self.ne0]
    }

    /// Give the block `ne1` columns of its `ne0` in place, inside the capacity
    /// it was made with, so a block made once for its widest use serves
    /// narrower calls without a take. Cells past the old length are stale
    /// zeros (NaN under `BLOOMERY_POISON`); the caller writes every cell it
    /// reads. Growing past the capacity would allocate and panics instead.
    pub(crate) fn set_cols(&mut self, ne1: usize) {
        let n = self.ne0 * ne1;
        assert!(
            n <= self.data.capacity(),
            "Tensor2::set_cols: {ne1} columns of {} past the block's capacity {}",
            self.ne0,
            self.data.capacity()
        );
        self.data.resize(n, if poison() { f32::NAN } else { 0.0 });
        self.ne1 = ne1;
    }
}

/// An f32 tensor straight from the file — norm gains, router biases, anything the quantizer left alone; the crate's one little-endian byte walk.
pub fn f32_tensor(gguf: &Gguf, t: &TensorInfo) -> Result<Vec<f32>, crate::ModelError> {
    // Profiler hook: a flat read — `rows`/`k` are 0; the weight MB is the work.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let bytes = gguf.data(t)?;
    let out = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    if let Some(t_call) = t_call {
        let acc = profile::CallAcc::new();
        profile::record(
            "f32_tensor",
            GgmlType::F32,
            0,
            0,
            bytes.len() as u64,
            t_call.elapsed().as_nanos() as u64,
            &acc,
        );
    }
    Ok(out)
}

/// RMS norm with a learned gain, per column — ggml's form: mean of squares over the row, `eps` inside the sqrt (added to the mean, not the result).
pub fn rms_norm(x: &Tensor2, gain: &[f32], eps: f32) -> Tensor2 {
    // Profiler hook: `rows` = columns normed, `k` = elements each walks.
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    assert_eq!(gain.len(), x.ne0, "rms_norm gain must be ne0 long");
    let mut out = Tensor2::scratch(x.ne0, x.ne1);
    for t in 0..x.ne1 {
        let src = x.col(t);
        // The reference's fused norm: f32 squares summed in f64, the mean
        // narrowed to f32, then `(scale · gain) · x` in that order.
        let sum = qdot::sum_sq_f64(src);
        let mean = (sum / x.ne0 as f64) as f32;
        let scale = 1.0f32 / (mean + eps).sqrt();
        let dst = out.col_mut(t);
        for i in 0..x.ne0 {
            dst[i] = scale * gain[i] * src[i];
        }
    }
    if let Some(t_call) = t_call {
        let acc = profile::CallAcc::new();
        profile::record(
            "rms_norm",
            GgmlType::F32,
            x.ne1 as u64,
            (x.ne0 * x.ne1) as u64,
            gain.len() as u64 * 4,
            t_call.elapsed().as_nanos() as u64,
            &acc,
        );
    }
    out
}

// Per-thread dequantized-row scratch for `matmul_q`: rows run on the pool, so every worker needs its own, grown to the largest `k` seen.
thread_local! {
    static ROW_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// One pool chunk's bookkeeping: row-range start (lowest-row-first error
/// precedence), stage timers, first error (quant or `qdot`). Values go to
/// `out` via `out_ptr`; crosses threads in one mutex take per chunk.
struct RowChunk {
    start: usize,
    /// The chunk's own wall, profiled runs only.
    busy_ns: u64,
    acc: profile::CallAcc,
    err: Option<crate::ModelError>,
}

/// `matmul_q`'s activation columns in the form the row loop consumes — f32 round trip (scalar) or `qdot::quantize_col` bytes (fused, `cb`/column); an enum so the paths cannot swap inputs.
///
/// The buffers are recycled per thread: quantized activations are per-call
/// values in stable storage, so a steady decode step takes from the pool and
/// returns instead of going to malloc (`Tensor2`'s free list is the same
/// argument). Contents are rewritten before use — every column is a pure
/// function of its input, and the quantizer writes whole columns.
enum QuantCols {
    /// `quantize_activations(w.ty, ..)` — f32 round-trip representation.
    F32(Vec<f32>),
    /// `qdot::quantize_col(w.ty, ..)` — one `cb`-byte column per token.
    Bytes { cb: usize, buf: Vec<u8> },
    /// Columns the caller quantized ([`QuantizedCols`]): its read view, no
    /// buffer of the dispatch's own.
    Held(SharedQuantCols),
}

impl QuantCols {
    /// A buffer for one distinct input of `ne1` columns, off the pool.
    fn new(cb: Option<usize>, x: &Tensor2) -> QuantCols {
        // Stale contents, like `Tensor2::scratch`; poisoned the same way, so a
        // quantizer that leaves a byte of its column unwritten shows in the gates.
        match cb {
            Some(cb) => {
                let mut buf = take_u8(x.ne1 * cb);
                if poison() {
                    buf.fill(0xA5);
                }
                QuantCols::Bytes { cb, buf }
            }
            None => {
                let mut buf = take_f32(x.data.len());
                if poison() {
                    buf.fill(f32::NAN);
                }
                QuantCols::F32(buf)
            }
        }
    }

    /// The write view the pool workers quantize through; SAFETY: the pointer
    /// aliases this buffer, owned by the caller for the whole dispatch — the
    /// disjoint-cells argument at `run_group`'s construction site.
    fn shared(&mut self) -> SharedQuantCols {
        match self {
            QuantCols::Bytes { cb, buf } => SharedQuantCols::Bytes {
                cb: *cb,
                ptr: buf.as_mut_ptr(),
            },
            QuantCols::F32(q) => SharedQuantCols::F32(q.as_mut_ptr()),
            QuantCols::Held(view) => *view,
        }
    }

    /// Hand the storage back to the pool.
    fn release(self) {
        match self {
            QuantCols::Bytes { buf, .. } => give_u8(buf),
            QuantCols::F32(q) => give_f32(q),
            QuantCols::Held(_) => {}
        }
    }
}

/// `matmul_q`'s output pointer in the one form the pool closure can capture
/// (raw pointers are neither `Send` nor `Sync`); the impls add no safety of
/// their own — see the construction site.
pub(crate) struct SharedOut(pub(crate) *mut f32);
// SAFETY: the pointer is into the dispatch's own output block, which outlives the
// split, and it is only ever written through — never read, never freed, by a worker.
unsafe impl Send for SharedOut {}
// SAFETY: sharing adds no aliasing — the split's row ranges are disjoint, so a worker
// writes only cells no other worker touches, and the join precedes every read.
unsafe impl Sync for SharedOut {}

impl SharedOut {
    /// Write one output cell — a method so the call captures `&SharedOut` (`Sync`), not the raw pointer.
    ///
    /// # Safety
    ///
    /// `idx` must name a cell inside the caller's own row range of the split.
    pub(crate) unsafe fn write(&self, idx: usize, v: f32) {
        // SAFETY: the caller guarantees the disjointness contract above.
        unsafe { *self.0.add(idx) = v };
    }
}

/// [`SharedOut`] for a distinct input's quantized-column buffer; the impls add no safety of their own — see the `run_group` construction site.
#[derive(Clone, Copy)]
enum SharedQuantCols {
    /// `quantize_activations(w.ty, ..)` — f32 round trip, `k` cells per column.
    F32(*mut f32),
    /// `qdot::quantize_col(w.ty, ..)` — one `cb`-byte column per token.
    Bytes { cb: usize, ptr: *mut u8 },
}
// SAFETY: the pointer is into the dispatch's own quantized-column buffer, which
// outlives the split, and a worker only writes through it — never reads, never frees.
unsafe impl Send for SharedQuantCols {}
// SAFETY: sharing adds no aliasing — one column is one cell, and a column is written
// by exactly one participant (its pre-pass sub-range, or the claim that won the slot's
// compare-exchange); the join or the slot's DONE precedes every read.
unsafe impl Sync for SharedQuantCols {}

impl SharedQuantCols {
    /// Quantize one column into its cell — the exact call a serial pre-pass would make; the split changes WHO calls it, never the call.
    ///
    /// # Safety
    ///
    /// The caller owns column `t`'s cells — its own sub-range of the pre-pass
    /// split, or the whole slot through a deferred claim's compare-exchange;
    /// `x_col` is column `t` of this buffer's input.
    unsafe fn quantize_into(&self, ty: GgmlType, k: usize, x_col: &[f32], t: usize) {
        match self {
            SharedQuantCols::F32(ptr) => {
                // SAFETY: the disjoint-cell contract above; the cell is `k` f32s, as allocated.
                let cell = unsafe { std::slice::from_raw_parts_mut(ptr.add(t * k), k) };
                quantize_activations(ty, x_col, cell)
                    .expect("the scalar arm of the fused decision refused this type before its slot was built");
            }
            SharedQuantCols::Bytes { cb, ptr } => {
                // SAFETY: same contract; the cell is `cb` bytes, as `col_bytes` allocated.
                let cell = unsafe { std::slice::from_raw_parts_mut(ptr.add(t * cb), *cb) };
                qdot::quantize_col(ty, x_col, cell);
            }
        }
    }
}

/// A SwiGLU slot's par block in the one form the dispatch state can carry
/// (raw pointers are neither `Send` nor `Sync`); the impls add no safety of
/// their own — see the construction site in `matmul_q_group_swiglu`.
#[derive(Clone, Copy)]
pub(crate) struct ParWrite(*mut f32, usize);
// SAFETY: the pointer is into the SwiGLU slot's par block, which outlives the
// dispatch, and the length travels with it so the slice it becomes is in bounds.
unsafe impl Send for ParWrite {}
// SAFETY: sharing adds no aliasing — the block's only writer is the participant that
// claimed the slot, and no reader runs before that slot observes DONE or the join.
unsafe impl Sync for ParWrite {}

/// `y = W · x` for a quantized 2-D `W` straight from the file, ggml's
/// convention: `W.dims == [k, n]`, `k` contiguous; `x` must be `[k, n_tokens]`;
/// dequantizes one row at a time (the row buffer stays in L1).
///
/// Activations are quantized first into the format the weight type implies
/// (Q8_K for Q3_K, Q8_2_X4 for Q4_K/Q5_K/Q6_K/Q5_0/Q5_1, none for F32;
/// `gguf::activation_format` owns the table) — the oracle is ggml's output,
/// and the wrong format would force every gate open.
///
/// A quantized type with a fused kernel takes the fused path (`crates/qdot`):
/// k a multiple of `qdot::k_granularity(ty)` — 256 for Q3_K/Q4_K/Q6_K, 32
/// for Q5_0/Q5_1. The fused dot is deliberately NOT bit-identical to the
/// scalar path (it is more accurate than the dequant round trip) — gates
/// across it assert closeness to the oracle, never bit equality; all other
/// types keep the scalar path.
///
/// The single-pair shape: no `Vec` of pairs, no dedupe table, the one
/// quantization buffer off the per-thread pool — the form a decode step calls
/// hundreds of times per second.
pub fn matmul_q(gguf: &Gguf, w: &TensorInfo, x: &Tensor2) -> Result<Tensor2, crate::ModelError> {
    matmul_q_one("matmul_q", gguf, w, x)
}

/// The batched sibling of [`matmul_q`]: `y_i = W_i · x_i` for same-`(k, n)`
/// weight views, one pool dispatch over the concatenated row space; rows split
/// only on the output row of one pair, so bit identity is unchanged
/// (`tests/ops.rs`: batched ≡ sequential). Pairs may share an activation
/// block — each DISTINCT block is quantized once (deterministic encoders);
/// `W_i` all agree on dims and type, `x_i` may differ in token count. The
/// homogeneity contract is [`batch_shape`]'s, checked here before the group
/// engine runs — a mixed list is [`matmul_q_group`]'s job, not this one's.
pub fn matmul_q_batch(
    gguf: &Gguf,
    ws: &[&TensorInfo],
    xs: &[&Tensor2],
) -> Result<Vec<Tensor2>, crate::ModelError> {
    batch_shape(ws, xs)?;
    matmul_q_multi("matmul_q_batch", gguf, ws, xs)
}

/// The heterogeneous sibling of [`matmul_q_batch`]: `y_i = W_i · x_i` where
/// every pair carries its own type, `k` and `n` — output `i` is
/// `{n_i, xs[i].ne1}`. One row dispatch over all pairs (lanes balanced by
/// weight bytes) and one quantization per DISTINCT (input, encoding). The
/// per-pair shape contract is `matmul_q`'s own input-width check, so a
/// mismatched pair errors exactly as the single-pair call would, and the
/// result is bit-identical to the per-pair `matmul_q` sequence
/// (`tests/ops.rs`).
pub fn matmul_q_group(
    gguf: &Gguf,
    ws: &[&TensorInfo],
    xs: &[&Tensor2],
) -> Result<Vec<Tensor2>, crate::ModelError> {
    matmul_q_multi("matmul_q_group", gguf, ws, xs)
}

/// One pair's input for [`matmul_q_group_swiglu`] and
/// [`matmul_q_group_swiglu_into`]: a ready activation block, some of a ready
/// block's columns, or the SwiGLU pair whose combine is the input.
#[derive(Clone, Copy)]
pub enum GroupInput<'a> {
    Ready(&'a Tensor2),
    /// (`x`, `cols`) — the pair's output column `t` is its weight times
    /// column `cols[t]` of `x`. `x` is quantized once, whole, like a ready
    /// block, and every pair that reads it shares that one quantization; a
    /// column past `x` is refused by name. The row loop dots each weight row
    /// against the listed columns only, so pairs of one expert over
    /// different token subsets read the weight once each.
    Cols(&'a Tensor2, &'a [usize]),
    /// (`x`, `xq`, `cols`) — [`GroupInput::Cols`] of `x` whose columns `xq`
    /// already holds quantized (filled from this `x`, in the encoding the
    /// pair's weight type reads): the pair reads `xq`'s bytes and no dispatch
    /// quantizes `x`. Any other block or encoding is refused by name.
    Quantized(&'a Tensor2, &'a QuantizedCols, &'a [usize]),
    /// (`gate`, `up`) — `silu(gate) · up` feeds the pair's weight.
    Swiglu(&'a Tensor2, &'a Tensor2),
    /// (`gate`, `up`, `limit`) — `clamp(up, ±limit) · min(silu(gate), limit)`
    /// feeds the pair's weight ([`qdot::swiglu_clamp`]; a limit at or below
    /// 1e-6 clamps nothing).
    SwigluClamp(&'a Tensor2, &'a Tensor2, f32),
}

impl<'a> GroupInput<'a> {
    /// The block whose shape is the pair's input shape: the ready block, or
    /// the gate the combine is made from.
    fn shape_of(self) -> &'a Tensor2 {
        match self {
            GroupInput::Ready(x) | GroupInput::Cols(x, _) | GroupInput::Quantized(x, _, _) => x,
            GroupInput::Swiglu(gate, _) | GroupInput::SwigluClamp(gate, _, _) => gate,
        }
    }

    /// The column map of a [`GroupInput::Cols`] or [`GroupInput::Quantized`]
    /// input; `None` reads every column in order.
    fn cols(self) -> Option<&'a [usize]> {
        match self {
            GroupInput::Cols(_, cols) | GroupInput::Quantized(_, _, cols) => Some(cols),
            _ => None,
        }
    }

    /// The columns the pair's output has.
    fn out_cols(self) -> usize {
        self.cols().map_or(self.shape_of().ne1, <[usize]>::len)
    }

    /// The combine this input stands for — gate, up and the clamp limit — or
    /// `None` for a ready block.
    fn combine(self) -> Option<(&'a Tensor2, &'a Tensor2, Option<f32>)> {
        match self {
            GroupInput::Ready(_) | GroupInput::Cols(..) | GroupInput::Quantized(..) => None,
            GroupInput::Swiglu(gate, up) => Some((gate, up, None)),
            GroupInput::SwigluClamp(gate, up, limit) => Some((gate, up, Some(limit))),
        }
    }
}

/// Every column of one activation block quantized once, in one weight type's
/// encoding, for many group dispatches to read ([`GroupInput::Quantized`]) —
/// each column's bytes exactly what a dispatch's own quantization writes for
/// it. Made empty; the first fill reserves its room and later fills reuse the
/// storage.
pub struct QuantizedCols {
    /// What the bytes are: `None` until the first fill.
    key: Option<QuantKey>,
    buf: Vec<u8>,
}

/// The block and encoding a [`QuantizedCols`] holds.
#[derive(Clone, Copy)]
struct QuantKey {
    /// The weight type whose activation format the bytes are in.
    ty: GgmlType,
    /// Values per column and columns.
    k: usize,
    ne1: usize,
    /// `qdot::col_bytes(ty, k)`.
    cb: usize,
    /// Address of the source block's data: identity only, never dereferenced.
    src: usize,
}

impl QuantizedCols {
    pub(crate) fn new() -> QuantizedCols {
        QuantizedCols {
            key: None,
            buf: Vec::new(),
        }
    }

    /// Quantize every column of `x` over the pool in `ty`'s fused encoding —
    /// the one encoder call per column a dispatch makes. Room for `room`
    /// columns is reserved the first time, so a later fill of up to that many
    /// allocates nothing. `false` (and nothing held): `ty` has no fused
    /// kernel at `x`'s width, or `x` has no column.
    pub(crate) fn fill(&mut self, ty: GgmlType, x: &Tensor2, room: usize) -> bool {
        self.key = None;
        let k = x.ne0;
        if x.ne1 == 0 || !qdot::fuses(ty, k) {
            return false;
        }
        let cb = qdot::col_bytes(ty, k);
        self.buf
            .reserve_exact((room.max(x.ne1) * cb).saturating_sub(self.buf.len()));
        self.buf.resize(x.ne1 * cb, 0);
        // Poisoned like a dispatch's own buffer, so an unwritten byte shows.
        if poison() {
            self.buf.fill(0xA5);
        }
        let shared = SharedQuantCols::Bytes {
            cb,
            ptr: self.buf.as_mut_ptr(),
        };
        // SAFETY (construction site): `buf` holds `x.ne1 · cb` bytes, borrowed
        // for the whole dispatch; each participant writes only the columns of
        // its own chunk of `0..x.ne1`, the chunks partition them, and the join
        // publishes the writes.
        threads::pool().for_each_chunk(x.ne1, |cols| {
            for t in cols {
                // SAFETY: column `t` is in this participant's chunk — see the construction site.
                unsafe { shared.quantize_into(ty, k, x.col(t), t) };
            }
        });
        self.key = Some(QuantKey {
            ty,
            k,
            ne1: x.ne1,
            cb,
            src: x.data.as_ptr() as usize,
        });
        true
    }

    /// Whether a pair whose weight is `ty` with `k` values per row reads
    /// exactly these bytes: fused at `k`, and the same column bytes and
    /// activation format — the dispatch's sharing rule.
    pub(crate) fn serves(&self, ty: GgmlType, k: usize) -> bool {
        self.key.is_some_and(|q| {
            q.k == k
                && qdot::fuses(ty, k)
                && qdot::col_bytes(ty, k) == q.cb
                && gguf::activation_format(ty) == gguf::activation_format(q.ty)
        })
    }

    /// Whether the bytes were filled from `x`.
    fn is_of(&self, x: &Tensor2) -> bool {
        self.key
            .is_some_and(|q| (q.src, q.k, q.ne1) == (x.data.as_ptr() as usize, x.ne0, x.ne1))
    }

    /// The read view a pair's rows take; nothing writes through it — a slot
    /// of these bytes has no columns to quantize and no claim.
    fn shared(&self) -> SharedQuantCols {
        SharedQuantCols::Bytes {
            cb: self.key.map_or(0, |q| q.cb),
            ptr: self.buf.as_ptr().cast_mut(),
        }
    }
}

/// The heterogeneous group whose pairs' inputs may be SwiGLU combines:
/// `y_i = W_i · silu(gate_i) · up_i` (clamped for a `SwigluClamp` input), one
/// pool dispatch for every projection. For a decode group (every input one
/// column, lever on) the dispatch's claimant produces each combine into a
/// poisoned scratch par (`qdot::swiglu`, the exact call `ffn::swiglu` makes,
/// or `qdot::swiglu_clamp`) and quantizes it before any row of its pair
/// runs — the caller never serializes on the combine or the encoder. Inputs
/// of more than one column, or `BLOOMERY_DEFER_QUANT=0`, keep today's shape:
/// the combine runs on the caller (its own `swiglu` profiler site) and the
/// group takes the ordinary pre-pass.
///
/// Returns the outputs and the produced `par` blocks — one per combine
/// input, pair order. A `Ready` or `Cols` input's block is the caller's own
/// and is not returned.
pub fn matmul_q_group_swiglu(
    gguf: &Gguf,
    ws: &[&TensorInfo],
    srcs: &[GroupInput<'_>],
) -> Result<(Vec<Tensor2>, Vec<Tensor2>), crate::ModelError> {
    let (mut outs, mut pars) = (Vec::new(), Vec::new());
    group_core(
        Weights::File(gguf, ws),
        Inputs::Mixed(srcs),
        Dest::Alloc {
            site: "matmul_q_group",
            outs: &mut outs,
            pars: &mut pars,
        },
        Entry::Plain,
    )?;
    Ok((outs, pars))
}

/// A tensor of a split model together with the shard that holds its bytes.
/// It is made only by a lookup in the split ([`ShardTensor::find`]), and the
/// bytes a dispatch reads through it ([`ShardTensor::weight`],
/// [`ShardTensor::expert`]) come from that shard — no caller hands a reader
/// in, so a header cannot be paired with another shard's file. Owned: a plan
/// holds these past the lookup.
///
/// ```compile_fail,E0451
/// # fn forge(up: &model::ops::ShardTensor) -> model::ops::ShardTensor {
/// // A handle naming another shard than the one its header came from.
/// model::ops::ShardTensor { shard: up.shard() + 1, info: up.info().clone() }
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct ShardTensor {
    shard: usize,
    info: TensorInfo,
}

impl ShardTensor {
    /// The tensor `name` and the shard the split found it in.
    pub fn find(split: &Split, name: &str) -> Result<ShardTensor, crate::ModelError> {
        let (shard, info) = split
            .find(name)
            .ok_or_else(|| crate::ModelError::MissingTensor(name.to_string()))?;
        Ok(ShardTensor {
            shard,
            info: info.clone(),
        })
    }

    pub fn info(&self) -> &TensorInfo {
        &self.info
    }

    pub fn shard(&self) -> usize {
        self.shard
    }

    /// The shard's reader in `split`; a handle from a split of fewer shards
    /// is refused by name.
    fn file<'a>(&self, split: &'a Split) -> Result<&'a Gguf, crate::ModelError> {
        split.shard(self.shard).ok_or_else(|| {
            crate::ModelError::MissingTensor(format!("{} (shard {})", self.info.name, self.shard))
        })
    }

    /// The whole tensor — one or two dims — as a dispatch weight. A stacked
    /// tensor is cut per expert by [`ShardTensor::expert`] instead.
    pub fn weight<'a>(&self, split: &'a Split) -> Result<Weight<'a>, crate::ModelError> {
        if self.info.dims.len() > 2 {
            return Err(crate::ModelError::Shape {
                what: "weight dims (a stack is cut per expert)",
                want_ne0: 2,
                want_ne1: 1,
                got_ne0: self.info.dims.len(),
                got_ne1: 1,
            });
        }
        Ok(Weight {
            ty: self.info.ty,
            k: k_of(&self.info),
            n: n_of(&self.info),
            bytes: self.file(split)?.data(&self.info)?,
        })
    }

    /// Matrix `e` of a stacked tensor `{k, n, n_expert}` as a dispatch weight:
    /// the `e`-th of the stack's `n_expert` equal byte runs — the cut
    /// `moe::expert_view` makes, without a header of its own.
    pub fn expert<'a>(&self, split: &'a Split, e: usize) -> Result<Weight<'a>, crate::ModelError> {
        let dims = &self.info.dims;
        let n_expert = dims.get(2).copied().unwrap_or(0);
        if dims.len() != 3 || n_expert == 0 || !self.info.nbytes.is_multiple_of(n_expert) {
            return Err(crate::ModelError::Shape {
                what: "expert stack",
                want_ne0: 3,
                want_ne1: n_expert as usize,
                got_ne0: dims.len(),
                got_ne1: n_expert as usize,
            });
        }
        if e as u64 >= n_expert {
            return Err(crate::ModelError::MissingTensor(format!(
                "expert {e} of {}",
                self.info.name
            )));
        }
        let per = (self.info.nbytes / n_expert) as usize;
        let stack = self.file(split)?.data(&self.info)?;
        Ok(Weight {
            ty: self.info.ty,
            k: dims[0] as usize,
            n: dims[1] as usize,
            bytes: &stack[e * per..(e + 1) * per],
        })
    }
}

/// One weight matrix as a dispatch reads it — type, `k` × `n` and the bytes —
/// resolved from the file that holds it. Outside this crate the only way to
/// make one is through a [`ShardTensor`] and its split; a one-file model's
/// own headers make one inside it. Copy and allocation-free: a host call
/// cuts its experts' matrices per token.
#[derive(Clone, Copy)]
pub struct Weight<'a> {
    ty: GgmlType,
    k: usize,
    n: usize,
    bytes: &'a [u8],
}

impl<'a> Weight<'a> {
    /// `info` read from `gguf`, which must be the file `info` came from — a
    /// one-file model, whose headers have no other.
    pub(crate) fn in_file(
        gguf: &'a Gguf,
        info: &TensorInfo,
    ) -> Result<Weight<'a>, crate::ModelError> {
        Ok(Weight {
            ty: info.ty,
            k: k_of(info),
            n: n_of(info),
            bytes: gguf.data(info)?,
        })
    }

    pub fn ty(&self) -> GgmlType {
        self.ty
    }

    /// The contracted dimension: values per row.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Rows: values per output column.
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// `k`, the contracted dimension of a header (0 for a header without dims).
fn k_of(w: &TensorInfo) -> usize {
    w.dims.first().copied().unwrap_or(0) as usize
}

/// `n`, the rows of a header (1 for a one-dim tensor).
fn n_of(w: &TensorInfo) -> usize {
    if w.dims.len() > 1 {
        w.dims[1] as usize
    } else {
        1
    }
}

/// What an `_into` group call cost, at profile level 1 and above (zeros when
/// the profiler is off): its wall, and the part of it the activation stage
/// held — the caller-side combines and pre-pass, or the longest claim pass of
/// a deferred dispatch (a claim is one input's combine and quantization, and
/// every row that reads that input waits for it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupTimes {
    pub wall_ns: u64,
    pub stage_ns: u64,
}

/// [`matmul_q_group`] into the caller's blocks: `outs[i]` must already be
/// `{ws[i].n(), xs[i].ne1}`, and nothing is allocated for the outputs. The
/// profiler rows go under `site`, the activation stage left out of them —
/// the caller records it from the returned [`GroupTimes`]. The values are
/// the allocating entry's, bit for bit (`tests/ops.rs`).
pub fn matmul_q_group_into(
    site: &'static str,
    ws: &[Weight<'_>],
    xs: &[&Tensor2],
    outs: &mut [Tensor2],
) -> Result<GroupTimes, crate::ModelError> {
    group_core(
        Weights::Resolved(ws),
        Inputs::Ready(xs),
        Dest::Into {
            site,
            outs,
            pars: &mut [],
        },
        Entry::Plain,
    )
}

/// [`matmul_q_group_swiglu`] into the caller's blocks: `outs[i]` must be
/// `{ws[i].n(), ne1}` and `pars[j]` `{k, ne1}` for the `j`-th combine input,
/// pair order; each par receives its combine. Profiler rows as
/// [`matmul_q_group_into`]'s — the combines are part of the stage.
pub fn matmul_q_group_swiglu_into(
    site: &'static str,
    ws: &[Weight<'_>],
    srcs: &[GroupInput<'_>],
    outs: &mut [Tensor2],
    pars: &mut [Tensor2],
) -> Result<GroupTimes, crate::ModelError> {
    group_core(
        Weights::Resolved(ws),
        Inputs::Mixed(srcs),
        Dest::Into { site, outs, pars },
        Entry::Plain,
    )
}

/// The most columns a slot of [`matmul_q_group_cols_into`] may carry and
/// still be claimed inside the dispatch: a claim quantizes (and combines)
/// the whole slot, so this bounds how long a row waits on one.
pub const DEFER_MAX_COLS: usize = 8;

/// [`matmul_q_group_swiglu_into`] for the few-column shape — a union of
/// experts over up to [`DEFER_MAX_COLS`] tokens, its inputs
/// [`GroupInput::Cols`] of one block or combines of a few columns each.
/// With the deferral lever on, a slot of up to that many columns is claimed
/// whole by one participant of the row dispatch (its combine, then each of
/// its columns quantized on its own) instead of the caller combining and the
/// pool pre-quantizing. The other entries keep the claim to one-column slots,
/// so a prefill group of a few tokens runs as before. A group with an input
/// wider than [`DEFER_MAX_COLS`] (a union chunk of a prefill ubatch) takes the
/// pre-pass, which then also produces each combine, column by column, and its
/// rows are cut into lanes by the tile kernel's cost.
/// Values are the pre-pass arm's, bit for bit: only who runs the combine and
/// the encoder, and which participant runs which rows, change.
pub fn matmul_q_group_cols_into(
    site: &'static str,
    ws: &[Weight<'_>],
    srcs: &[GroupInput<'_>],
    outs: &mut [Tensor2],
    pars: &mut [Tensor2],
) -> Result<GroupTimes, crate::ModelError> {
    group_core(
        Weights::Resolved(ws),
        Inputs::Mixed(srcs),
        Dest::Into { site, outs, pars },
        Entry::Cols,
    )
}

/// The entry a group came through, for the rules only the union's entry has.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Entry {
    /// A deferred claim takes one-column slots only.
    Plain,
    /// [`matmul_q_group_cols_into`]: a claim takes slots of up to
    /// [`DEFER_MAX_COLS`] columns; a group with a wider input produces its
    /// combines in the pre-pass and cuts its lanes by the tile's cost.
    Cols,
}

impl Entry {
    /// The widest slot a deferred claim of this entry takes.
    fn defer_cols(self) -> usize {
        match self {
            Entry::Plain => 1,
            Entry::Cols => DEFER_MAX_COLS,
        }
    }
}

/// Where a group's weights come from: headers of one file and that file
/// (the entries that take a `Gguf`), or weights already resolved from the
/// file that holds each ([`Weight`]).
#[derive(Clone, Copy)]
enum Weights<'a> {
    File(&'a Gguf, &'a [&'a TensorInfo]),
    Resolved(&'a [Weight<'a>]),
}

impl<'a> Weights<'a> {
    fn len(self) -> usize {
        match self {
            Weights::File(_, ws) => ws.len(),
            Weights::Resolved(ws) => ws.len(),
        }
    }

    /// Pair `i`'s type, `k` and `n` — no byte read yet, so a bad input width
    /// is reported before a bad placement, as the single-pair call does.
    fn shape(self, i: usize) -> (GgmlType, usize, usize) {
        match self {
            Weights::File(_, ws) => (ws[i].ty, k_of(ws[i]), n_of(ws[i])),
            Weights::Resolved(ws) => (ws[i].ty, ws[i].k, ws[i].n),
        }
    }

    fn bytes(self, i: usize) -> Result<&'a [u8], crate::ModelError> {
        match self {
            Weights::File(gguf, ws) => Ok(gguf.data(ws[i])?),
            Weights::Resolved(ws) => Ok(ws[i].bytes),
        }
    }
}

/// A group's inputs: ready blocks only, or blocks and SwiGLU combines.
#[derive(Clone, Copy)]
enum Inputs<'a, 'x> {
    /// The deferral decision is `run_group`'s, by slot.
    Ready(&'a [&'x Tensor2]),
    /// The decision is made before any par exists: the claimants produce the
    /// combines only when every input is one column and the claim states
    /// fit; otherwise the caller combines and the group takes the pre-pass.
    Mixed(&'a [GroupInput<'x>]),
}

impl<'a, 'x> Inputs<'a, 'x> {
    fn len(self) -> usize {
        match self {
            Inputs::Ready(xs) => xs.len(),
            Inputs::Mixed(srcs) => srcs.len(),
        }
    }

    fn get(self, i: usize) -> GroupInput<'x> {
        match self {
            Inputs::Ready(xs) => GroupInput::Ready(xs[i]),
            Inputs::Mixed(srcs) => srcs[i],
        }
    }
}

/// Where a group's outputs and the pars of its combines go, and what its
/// profiler rows leave out.
enum Dest<'o> {
    /// Fresh blocks off the pool, one per pair and one per combine, pushed
    /// in pair order; the rows under `site` leave out the combines, which
    /// the `swiglu` site records.
    Alloc {
        site: &'static str,
        outs: &'o mut Vec<Tensor2>,
        pars: &'o mut Vec<Tensor2>,
    },
    /// The caller's blocks, already shaped; the rows under `site` leave out
    /// the activation stage, which the caller records from [`GroupTimes`].
    Into {
        site: &'static str,
        outs: &'o mut [Tensor2],
        pars: &'o mut [Tensor2],
    },
}

/// `silu(gate) · up` into `out`, clamped when the pair carries a limit — the
/// one call every combine makes, on a claimant or on the caller.
fn swiglu_into(gate: &[f32], up: &[f32], limit: Option<f32>, out: &mut [f32]) {
    match limit {
        None => qdot::swiglu(gate, up, out),
        Some(limit) => qdot::swiglu_clamp(gate, up, limit, out),
    }
}

/// The allocating entry's caller-side combine: a fresh block and its time,
/// recorded under the `swiglu` site. A plain pair is `ffn::swiglu_timed`
/// itself; a clamped one is the same shape with the clamped call.
fn combine_timed(gate: &Tensor2, up: &Tensor2, limit: Option<f32>) -> (Tensor2, u64) {
    if limit.is_none() {
        return swiglu_timed(gate, up);
    }
    let lvl = profile::level();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut out = Tensor2::scratch(gate.ne0, gate.ne1);
    swiglu_into(&gate.data, &up.data, limit, &mut out.data);
    let ns = match t_call {
        Some(t_call) => {
            let ns = t_call.elapsed().as_nanos() as u64;
            profile::record_time("swiglu", ns);
            ns
        }
        None => 0,
    };
    (out, ns)
}

/// The group engine every multi-pair entry rides: per pair the input-width
/// check, the fused decision, the combine's par (produced by a claimant, by
/// the pre-pass or here), the slot and the output block; then `run_group` and
/// the profiler rows. `entry` sets the widest slot the deferred claim may
/// take and whether a wide group takes the union's pre-pass combine and tile
/// lane cut ([`Entry`]).
fn group_core(
    ws: Weights<'_>,
    inputs: Inputs<'_, '_>,
    dest: Dest<'_>,
    entry: Entry,
) -> Result<GroupTimes, crate::ModelError> {
    let n_pairs = ws.len();
    if n_pairs != inputs.len() {
        return Err(crate::ModelError::Shape {
            what: "matmul_q batch",
            want_ne0: n_pairs,
            want_ne1: n_pairs,
            got_ne0: inputs.len(),
            got_ne1: inputs.len(),
        });
    }
    if n_pairs == 0 {
        return Ok(GroupTimes::default());
    }
    // Profiler hook: level 0 one compare, level 1 one `Instant` pair, level 2
    // two more per row, none touching the arithmetic; chunks merge into one
    // `CallAcc` per weight type, so `record` fires once per (call, type).
    let lvl = profile::level();
    let mut pacc = profile::CallAcc::new();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let n_combine = (0..n_pairs)
        .filter(|&i| inputs.get(i).combine().is_some())
        .count();
    let defer = match inputs {
        Inputs::Ready(_) => defer_quant(),
        Inputs::Mixed(srcs) => {
            defer_quant()
                && n_pairs <= MAX_DEFER_SLOTS
                && srcs.iter().all(|s| s.shape_of().ne1 <= entry.defer_cols())
        }
    };
    // A union chunk of a prefill ubatch: an input wider than any claim takes,
    // so the group is on the pre-pass arm whatever the lever says. Narrower
    // groups — every decode and verify call — never set it.
    let wide =
        entry == Entry::Cols && (0..n_pairs).any(|i| inputs.get(i).shape_of().ne1 > DEFER_MAX_COLS);
    // Who writes a combine's par: the claimant (deferred), the pre-pass
    // column by column (wide), or the caller before the dispatch.
    let caller_combines = !defer && !wide;
    let (site, mut sink) = match dest {
        Dest::Alloc { site, outs, pars } => {
            // Exact capacities: a slot names its par where it sits, so the
            // Vec must never grow past its reservation.
            pars.reserve_exact(n_combine);
            outs.reserve_exact(n_pairs);
            (site, Sink::Alloc { outs, pars })
        }
        Dest::Into { site, outs, pars } => {
            if outs.len() != n_pairs || pars.len() != n_combine {
                return Err(crate::ModelError::Shape {
                    what: "group into: one out per pair, one par per combine",
                    want_ne0: n_pairs,
                    want_ne1: n_combine,
                    got_ne0: outs.len(),
                    got_ne1: pars.len(),
                });
            }
            // One base pointer for the whole call: a slot names each par
            // through it while that par's claimant writes the cells.
            (
                site,
                Sink::Into {
                    outs,
                    pars: pars.as_mut_ptr(),
                },
            )
        }
    };
    let into = matches!(sink, Sink::Into { .. });
    let mut bytes: Flex<&[u8]> = Flex::new();
    let mut meta: Flex<PairMeta> = Flex::new();
    let mut slot_of: Flex<usize> = Flex::new();
    let mut cols_of: Flex<Option<&[usize]>> = Flex::new();
    let mut slots: Flex<QuantSlot> = Flex::new();
    let mut caller_combine_ns: u64 = 0;
    let mut j = 0usize;
    for i in 0..n_pairs {
        let (ty, k, n) = ws.shape(i);
        let src = inputs.get(i);
        let xin = src.shape_of();
        if xin.ne0 != k {
            return Err(crate::ModelError::Shape {
                what: "matmul_q input",
                want_ne0: k,
                want_ne1: xin.ne1,
                got_ne0: xin.ne0,
                got_ne1: xin.ne1,
            });
        }
        if let Some(&c) = src
            .cols()
            .and_then(|cols| cols.iter().find(|&&c| c >= xin.ne1))
        {
            return Err(crate::ModelError::Shape {
                what: "group cols: a listed column past the block",
                want_ne0: k,
                want_ne1: xin.ne1,
                got_ne0: k,
                got_ne1: c,
            });
        }
        let out_cols = src.out_cols();
        let b = ws.bytes(i)?;
        // Fused path decided per pair, exactly the single-pair rule: qdot
        // dots the quantized codes directly, so activations are quantized
        // into `qdot::quantize_col`'s layout, and `qdot::fuses` owns the
        // per-type block contract (256 for the K-quants, 32 for Q5_0/Q5_1).
        let fused = qdot::fuses(ty, k);
        let cb = if fused {
            Some(qdot::col_bytes(ty, k))
        } else {
            // The scalar path quantizes into the weight type's activation format;
            // a type without one is refused here, before any column exists.
            gguf::activation_format(ty)?;
            None
        };
        // The combine: produced by the dispatch's claimant (deferred), by the
        // pre-pass (wide) or here on the caller — the same call either way,
        // into the pair's par.
        let (x, swiglu) = match src.combine() {
            None => (xin, None),
            Some((gate, up, limit)) => {
                if (up.ne0, up.ne1) != (gate.ne0, gate.ne1) {
                    return Err(crate::ModelError::Shape {
                        what: "swiglu up",
                        want_ne0: gate.ne0,
                        want_ne1: gate.ne1,
                        got_ne0: up.ne0,
                        got_ne1: up.ne1,
                    });
                }
                let blk: *mut Tensor2 = match &mut sink {
                    Sink::Alloc { pars, .. } => {
                        let par = if caller_combines {
                            let (par, ns) = combine_timed(gate, up, limit);
                            caller_combine_ns += ns;
                            par
                        } else {
                            Tensor2::scratch(k, xin.ne1)
                        };
                        pars.push(par);
                        pars.last_mut().expect("just pushed")
                    }
                    Sink::Into { pars, .. } => {
                        // SAFETY: `j < n_combine`, the length checked above;
                        // each index is taken once.
                        let blk = unsafe { pars.add(j) };
                        // SAFETY: the caller's block, lent for the call; no
                        // slot names it yet, so this read and the combine
                        // below are its only accesses.
                        let (ne0, ne1, len) =
                            unsafe { ((*blk).ne0, (*blk).ne1, (*blk).data.len()) };
                        if (ne0, ne1, len) != (k, xin.ne1, k * xin.ne1) {
                            return Err(crate::ModelError::Shape {
                                what: "group into par",
                                want_ne0: k,
                                want_ne1: xin.ne1,
                                got_ne0: ne0,
                                got_ne1: ne1,
                            });
                        }
                        if caller_combines {
                            let t_c = if lvl > 0 { Some(Instant::now()) } else { None };
                            // SAFETY: as the read above.
                            swiglu_into(&gate.data, &up.data, limit, unsafe { &mut (*blk).data });
                            if let Some(t_c) = t_c {
                                caller_combine_ns += t_c.elapsed().as_nanos() as u64;
                            }
                        }
                        blk
                    }
                };
                j += 1;
                // The cells, not the block: the claimant's `&mut` covers the
                // heap cells only, never the header the slot table reads.
                // SAFETY: `blk` is the par just placed or checked above.
                let write = unsafe { ParWrite((*blk).data.as_mut_ptr(), (*blk).data.len()) };
                // SAFETY: the block never moves during the call (the Vec sits
                // at its exact final capacity, or the caller lent it); the
                // write view goes to exactly one claimant through the slot,
                // and no reader runs before the slot observes DONE or the
                // dispatch joins — this shared ref only names the block for
                // the slot table (metadata reads).
                let par: &Tensor2 = unsafe { &*blk };
                (
                    par,
                    Some(Combine {
                        gate,
                        up,
                        limit,
                        par: write,
                    }),
                )
            }
        };
        // The sharing rule — one quantized buffer per DISTINCT (input,
        // encoding): same input block AND same fused-ness AND same activation
        // format. Sound both ways. Sharers write identical bytes: `k` is
        // pinned to the input's `ne0` by the check above, the encoders are
        // pure functions of (activation format, k) — `quantize_col`'s arm
        // and `quantize_activations` both key on the format, never the
        // weight type — and equal format with equal `k` yields equal
        // `col_bytes`. Non-sharers never collide: a different format or
        // fused-ness means the encoders disagree on at least the buffer's
        // representation, so sharing would feed one pair the other's bytes.
        // A combine's par is a distinct block by construction, so it never
        // shares. Bytes the caller quantized ([`GroupInput::Quantized`]) are a
        // slot of their own kind, shared by every pair that names them, after
        // the same rule decided they are this pair's bytes.
        let pre = match src {
            GroupInput::Quantized(_, xq, _) => {
                if !xq.is_of(x) || !xq.serves(ty, k) {
                    return Err(crate::ModelError::Shape {
                        what: "group quantized cols: filled from another block or for another encoding",
                        want_ne0: x.ne0,
                        want_ne1: x.ne1,
                        got_ne0: xq.key.map_or(0, |q| q.k),
                        got_ne1: xq.key.map_or(0, |q| q.ne1),
                    });
                }
                Some(xq)
            }
            _ => None,
        };
        let slot = match (swiglu, pre) {
            (Some(..), _) => {
                slots.push(QuantSlot {
                    x,
                    ty,
                    k,
                    cb,
                    swiglu,
                    produced: caller_combines,
                    pre: None,
                });
                slots.len() - 1
            }
            (None, Some(xq)) => (0..slots.len())
                .find(|&s| slots.get(s).pre.is_some_and(|q| std::ptr::eq(q, xq)))
                .unwrap_or_else(|| {
                    slots.push(QuantSlot {
                        x,
                        ty,
                        k,
                        cb,
                        swiglu: None,
                        produced: false,
                        pre: Some(xq),
                    });
                    slots.len() - 1
                }),
            (None, None) => (0..slots.len())
                .find(|&s| {
                    let q = slots.get(s);
                    q.swiglu.is_none()
                        && q.pre.is_none()
                        && std::ptr::eq(q.x, x)
                        && q.cb == cb
                        && gguf::activation_format(q.ty) == gguf::activation_format(ty)
                })
                .unwrap_or_else(|| {
                    slots.push(QuantSlot {
                        x,
                        ty,
                        k,
                        cb,
                        swiglu: None,
                        produced: false,
                        pre: None,
                    });
                    slots.len() - 1
                }),
        };
        bytes.push(b);
        meta.push(PairMeta {
            ty,
            k,
            n,
            row_bytes: b.len() / n,
        });
        slot_of.push(slot);
        cols_of.push(src.cols());
        match &mut sink {
            Sink::Alloc { outs, .. } => outs.push(Tensor2::scratch(n, out_cols)),
            Sink::Into { outs, .. } => {
                let o = &outs[i];
                if (o.ne0, o.ne1, o.data.len()) != (n, out_cols, n * out_cols) {
                    return Err(crate::ModelError::Shape {
                        what: "group into out",
                        want_ne0: n,
                        want_ne1: out_cols,
                        got_ne0: o.ne0,
                        got_ne1: o.ne1,
                    });
                }
            }
        }
    }
    LAST_QUANT_SLOTS.store(slots.len(), std::sync::atomic::Ordering::Relaxed);

    let outs: &mut [Tensor2] = match sink {
        Sink::Alloc { outs, .. } => outs,
        Sink::Into { outs, .. } => outs,
    };
    // SAFETY: the construction-site argument in `run_group`'s pair build.
    let rt = run_group(
        &slots,
        &PairMap {
            slot_of: &slot_of,
            cols_of: &cols_of,
        },
        &bytes,
        &meta,
        outs,
        lvl,
        &mut pacc,
        Arm {
            defer_cols: defer.then_some(entry.defer_cols()),
            tile_lanes: wide,
        },
    )?;
    // What the rows leave out: the combines, which the allocating entry's
    // `swiglu` site records; or the whole activation stage, which an `_into`
    // caller records.
    let left_out = if into {
        caller_combine_ns + rt.stage_ns
    } else {
        if defer && lvl > 0 && n_combine > 0 {
            profile::record_time("swiglu", rt.swiglu_ns);
        }
        caller_combine_ns + rt.swiglu_ns
    };
    let mut times = GroupTimes::default();
    if let Some(t_call) = t_call {
        let wall_ns = t_call.elapsed().as_nanos() as u64;
        record_rows(site, &meta, &bytes, wall_ns.saturating_sub(left_out), &pacc);
        if into {
            times = GroupTimes {
                wall_ns,
                stage_ns: left_out,
            };
        }
    }
    Ok(times)
}

/// [`Dest`] inside the pair loop: the allocating entry's Vecs, or the
/// caller's outputs and the base pointer of its pars.
enum Sink<'o> {
    Alloc {
        outs: &'o mut Vec<Tensor2>,
        pars: &'o mut Vec<Tensor2>,
    },
    Into {
        outs: &'o mut [Tensor2],
        pars: *mut Tensor2,
    },
}

/// One profiler row per distinct weight type the group carried: rows, `k`
/// and weight bytes exact per type; `ns` and the stage accumulators split by
/// weight-byte share — the honest per-type statement without per-type
/// timers. The shares sum to `ns`, so `instrumented_ns` never counts a group
/// twice, and a single-type group records exactly the homogeneous values.
fn record_rows(
    site: &'static str,
    meta: &Flex<PairMeta>,
    bytes: &Flex<&[u8]>,
    ns: u64,
    pacc: &profile::CallAcc,
) {
    let wb_total: u64 = (0..bytes.len()).map(|i| bytes.get(i).len() as u64).sum();
    let mut tys: Flex<(GgmlType, u64, u64, u64)> = Flex::new();
    for i in 0..meta.len() {
        let m = *meta.get(i);
        let wb = bytes.get(i).len() as u64;
        let mut j = 0;
        while j < tys.len() && tys.get(j).0 != m.ty {
            j += 1;
        }
        if j == tys.len() {
            tys.push((m.ty, m.n as u64, m.k as u64 * m.n as u64, wb));
        } else {
            let e = tys.get_mut(j);
            *e = (
                e.0,
                e.1 + m.n as u64,
                e.2 + m.k as u64 * m.n as u64,
                e.3 + wb,
            );
        }
    }
    for t in 0..tys.len() {
        let &(ty, rows, k_total, wb) = tys.get(t);
        let share = |ns: u64| ((ns as u128 * wb as u128) / wb_total as u128) as u64;
        profile::record(
            site,
            ty,
            rows,
            k_total,
            wb,
            share(ns),
            &pacc.scaled(wb, wb_total),
        );
    }
}

/// Pairs a group carries before its bookkeeping spills to the heap. The
/// step's widest group is MoE gate/up — `2·n_used + 2 = 14` at `n_used = 6` —
/// and `wv_b`'s per-head batch is 16; both stay inline, so a decode step's
/// groups allocate nothing but their outputs.
const GROUP_INLINE: usize = 16;

/// Per-call group bookkeeping: a fixed stack block for the first
/// [`GROUP_INLINE`] entries, the heap past that. Indexed access only — the
/// walkers advance monotonically and never need a contiguous slice.
struct Flex<T> {
    inline: [Option<T>; GROUP_INLINE],
    heap: Vec<T>,
    len: usize,
}

impl<T> Flex<T> {
    fn new() -> Self {
        Flex {
            inline: std::array::from_fn(|_| None),
            heap: Vec::new(),
            len: 0,
        }
    }

    fn push(&mut self, v: T) {
        if self.len < GROUP_INLINE {
            self.inline[self.len] = Some(v);
        } else {
            self.heap.push(v);
        }
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    /// An index below `len`; the inline tail above it is never read.
    fn get(&self, i: usize) -> &T {
        if i < GROUP_INLINE {
            self.inline[i].as_ref().expect("index below len is filled")
        } else {
            &self.heap[i - GROUP_INLINE]
        }
    }

    fn get_mut(&mut self, i: usize) -> &mut T {
        if i < GROUP_INLINE {
            self.inline[i].as_mut().expect("index below len is filled")
        } else {
            &mut self.heap[i - GROUP_INLINE]
        }
    }
}

impl Flex<QuantCols> {
    /// Hand every buffer back to its pool — the heap arm drains from the
    /// end because `swap_remove` shortens it.
    fn release_all(&mut self) {
        while let Some(q) = self.heap.pop() {
            q.release();
        }
        for slot in &mut self.inline {
            if let Some(q) = slot.take() {
                q.release();
            }
        }
        self.len = 0;
    }
}

impl<T> Default for Flex<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// One (weight view, activation block) pair plus everything a row needs; the
/// row loop exists once, here — a duplicate is how the shapes would drift bit for bit.
/// A group is heterogeneous: `ty`/`k`/`n` are this pair's own, not the call's.
struct PairWork<'a> {
    ty: GgmlType,
    k: usize,
    n: usize,
    bytes: &'a [u8],
    row_bytes: usize,
    /// Fused qdot columns: (stride `cb`, quantized byte buffer) as raw parts
    /// — a deferred dispatch's claimer may still be writing them when this
    /// pair is built, so `compute_rows` forms the slice only after the slot
    /// observes DONE.
    fused_cols: Option<(usize, *const u8)>,
    /// Scalar-path f32 round trip; null on the fused path. Same raw-parts
    /// contract as `fused_cols`.
    q_f32: *const f32,
    /// The slot's columns — what `fused_cols` / `q_f32` hold.
    src_ne1: usize,
    /// Output column `t` reads slot column `cols[t]`; `None` is `t` itself.
    cols: Option<&'a [usize]>,
    /// Output columns.
    ne1: usize,
    out: SharedOut,
    /// The slot whose columns this pair reads, and the claim table when
    /// quantization is deferred into the dispatch (`None`: the pre-pass
    /// completed every column before any row runs).
    slot: usize,
    defer: Option<&'a DeferredSlots<'a>>,
}
// SAFETY: `fused_cols` and `q_f32` are bare pointers — the type does not bind them
// to `'a`, so the contract is the `run_group` construction site's: they point into
// column buffers that outlive the dispatch, and `compute_rows` forms a slice only
// after the slot observes DONE (deferred) or the pre-pass join (ordinary).
unsafe impl<'a> Send for PairWork<'a> {}
// SAFETY: sharing one table across the dispatch adds no aliasing — the column cells
// are read-only once published, and each participant writes only the output cells its
// own row range owns.
unsafe impl<'a> Sync for PairWork<'a> {}

/// The per-pair facts resolved before any output block exists.
#[derive(Clone, Copy)]
struct PairMeta {
    ty: GgmlType,
    k: usize,
    n: usize,
    row_bytes: usize,
}

impl<'a> PairWork<'a> {
    /// The one construction path both dispatch shapes ride; the SAFETY
    /// argument for the output pointer lives at the callers, where the split
    /// is decided. `src` is the slot's write view, its column count and the
    /// pair's column map.
    fn new(
        out: &mut Tensor2,
        bytes: &'a [u8],
        src: (SharedQuantCols, usize, Option<&'a [usize]>),
        m: PairMeta,
        slot: usize,
        defer: Option<&'a DeferredSlots<'a>>,
    ) -> PairWork<'a> {
        let (q, src_ne1, cols) = src;
        let out_ptr = SharedOut(out.data.as_mut_ptr());
        let (fused_cols, q_f32) = match q {
            SharedQuantCols::Bytes { cb, ptr } => (Some((cb, ptr as *const u8)), std::ptr::null()),
            SharedQuantCols::F32(ptr) => (None, ptr as *const f32),
        };
        PairWork {
            ty: m.ty,
            k: m.k,
            n: m.n,
            bytes,
            row_bytes: m.row_bytes,
            fused_cols,
            q_f32,
            src_ne1,
            cols,
            ne1: out.ne1,
            out: out_ptr,
            slot,
            defer,
        }
    }

    /// The slot column output column `t` reads.
    #[inline(always)]
    fn col(&self, t: usize) -> usize {
        match self.cols {
            Some(cols) => cols[t],
            None => t,
        }
    }

    /// Rows `rows` of this pair across every token column → cells `t * n + r`; ROW_BUF is borrowed once per call.
    fn compute_rows(
        &self,
        rows: Range<usize>,
        lvl: u8,
        acc: &mut profile::CallAcc,
    ) -> Result<(), crate::ModelError> {
        let (ty, k, n) = (self.ty, self.k, self.n);
        // A deferred slot's columns must be complete before its rows read
        // them; on the pre-pass arm this is a no-op (no claim table).
        if let Some(defer) = self.defer {
            defer.wait_ready(self.slot);
        }
        if let Some((cb, aptr)) = self.fused_cols {
            // SAFETY: the slot's buffer is `src_ne1 * cb` bytes as allocated,
            // and its writes are ordered before this read — the pre-pass
            // join, or the DONE store the wait above just observed.
            let acol = unsafe { std::slice::from_raw_parts(aptr, self.src_ne1 * cb) };
            // Fused rows: `qdot` dots the weight bytes and the quantized
            // columns directly — no f32 row, ROW_BUF untouched;
            // `add_dequant_w` stays 0 (the dequant is in the dot timer). A
            // lone column takes `qdot::dot_row`; runs of up to
            // `qdot::TILE_COLS` columns take `qdot::dot_row_cols`, which
            // unpacks the row's blocks once per run and writes each column's
            // `dot_row` value bit for bit.
            for r in rows {
                let src = &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
                let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                let mut t0 = 0;
                while t0 < self.ne1 {
                    let m = (self.ne1 - t0).min(qdot::TILE_COLS);
                    if m == 1 {
                        let c = self.col(t0);
                        let v = qdot::dot_row(ty, src, &acol[c * cb..(c + 1) * cb], k)?;
                        // SAFETY: cell `t0 * n + r` is inside this chunk's row range — see the SharedOut construction site.
                        unsafe { self.out.write(t0 * n + r, v) };
                    } else {
                        let cols: [&[u8]; qdot::TILE_COLS] = std::array::from_fn(|i| {
                            let c = self.col(t0 + i.min(m - 1));
                            &acol[c * cb..(c + 1) * cb]
                        });
                        let mut v = [0.0f32; qdot::TILE_COLS];
                        qdot::dot_row_cols(ty, src, &cols[..m], k, &mut v[..m])?;
                        for (i, &v) in v[..m].iter().enumerate() {
                            // SAFETY: cell `(t0 + i) * n + r` is inside this chunk's row range — see the SharedOut construction site.
                            unsafe { self.out.write((t0 + i) * n + r, v) };
                        }
                    }
                    t0 += m;
                }
                if let Some(t_dot) = t_dot {
                    acc.add_dot(t_dot.elapsed().as_nanos() as u64);
                }
            }
            Ok(())
        } else {
            // SAFETY: `k * src_ne1` f32s as allocated, ordered before this
            // read exactly as the fused arm above.
            let q_f32 = unsafe { std::slice::from_raw_parts(self.q_f32, k * self.src_ne1) };
            ROW_BUF.with(|cell| {
                let mut scratch = cell.borrow_mut();
                if scratch.len() < k {
                    scratch.resize(k, 0.0);
                }
                let row = &mut scratch[..k];
                for r in rows {
                    let src = &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
                    // An F32 row against F32 columns: the reference's float
                    // kernel order, straight off the file bytes.
                    if ty == GgmlType::F32 && qdot::dot_f32(src, &q_f32[..k]).is_some() {
                        let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                        for t in 0..self.ne1 {
                            let c = self.col(t);
                            let xc = &q_f32[c * k..(c + 1) * k];
                            let v = qdot::dot_f32(src, xc).expect("support is per (cpu, k)");
                            // SAFETY: cell `t * n + r` is inside this chunk's row range — see the SharedOut construction site.
                            unsafe { self.out.write(t * n + r, v) };
                        }
                        if let Some(t_dot) = t_dot {
                            acc.add_dot(t_dot.elapsed().as_nanos() as u64);
                        }
                        continue;
                    }
                    // Weight side: dequant_row, or the byte walk in its place for F32 rows.
                    let t_d = if lvl >= 2 { Some(Instant::now()) } else { None };
                    if ty == GgmlType::F32 {
                        let (words, _) = src.as_chunks::<4>();
                        let words = &words[..row.len()];
                        for (v, w) in row.iter_mut().zip(words) {
                            *v = f32::from_le_bytes(*w);
                        }
                    } else {
                        dequant_row(ty, src, row)?;
                    }
                    if let Some(t_d) = t_d {
                        acc.add_dequant_w(t_d.elapsed().as_nanos() as u64);
                    }
                    let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                    for t in 0..self.ne1 {
                        let c = self.col(t);
                        let xc = &q_f32[c * k..(c + 1) * k];
                        let mut a = 0.0f32;
                        for i in 0..k {
                            a += row[i] * xc[i];
                        }
                        // SAFETY: cell `t * n + r` is inside this chunk's row range — see the SharedOut construction site.
                        unsafe { self.out.write(t * n + r, a) };
                    }
                    if let Some(t_dot) = t_dot {
                        acc.add_dot(t_dot.elapsed().as_nanos() as u64);
                    }
                }
                Ok(())
            })
        }
    }
}

/// Shape contract of a batch: equal-length lists; every `W_i` agrees on
/// `k`, `n`, type (one expert stack) and every input is `k` wide. `Ok(None)`
/// is the empty batch, not an error.
fn batch_shape(
    ws: &[&TensorInfo],
    xs: &[&Tensor2],
) -> Result<Option<(usize, usize, GgmlType)>, crate::ModelError> {
    if ws.len() != xs.len() {
        return Err(crate::ModelError::Shape {
            what: "matmul_q batch",
            want_ne0: ws.len(),
            want_ne1: ws.len(),
            got_ne0: xs.len(),
            got_ne1: xs.len(),
        });
    }
    let Some(w0) = ws.first() else {
        return Ok(None);
    };
    let k = w0.dims[0] as usize;
    let n = if w0.dims.len() > 1 {
        w0.dims[1] as usize
    } else {
        1
    };
    let ty = w0.ty;
    for (w, x) in ws.iter().zip(xs) {
        let wk = w.dims.first().copied().unwrap_or(0) as usize;
        let wn = if w.dims.len() > 1 {
            w.dims[1] as usize
        } else {
            1
        };
        if wk != k || wn != n || w.ty != ty {
            return Err(crate::ModelError::Shape {
                what: "matmul_q batch",
                want_ne0: k,
                want_ne1: n,
                got_ne0: wk,
                got_ne1: wn,
            });
        }
        if x.ne0 != k {
            return Err(crate::ModelError::Shape {
                what: "matmul_q input",
                want_ne0: k,
                want_ne1: x.ne1,
                got_ne0: x.ne0,
                got_ne1: x.ne1,
            });
        }
    }
    Ok(Some((k, n, ty)))
}

/// One distinct quantization job: an input block plus the single encoding
/// every pair that lands on this slot reads. The dedupe key at the
/// construction site is (input identity, fused, activation format); `k` is
/// pinned to the input's `ne0` by the per-pair shape check, so key equality
/// implies equal columns.
struct QuantSlot<'a> {
    /// The block whose columns are quantized — a ready input, or the par
    /// block the slot's SwiGLU source produces (see `swiglu`).
    x: &'a Tensor2,
    ty: GgmlType,
    k: usize,
    /// `qdot::col_bytes(ty, k)` on the fused path, `None` on the scalar one.
    cb: Option<usize>,
    /// The SwiGLU producer, when `x` is not a ready input: the claimer (or,
    /// unless `produced`, the pre-pass) writes the combine through its `par`
    /// (which aliases `x`) before quantizing `x`'s columns. `None` on ready
    /// inputs — nothing writes `x`.
    swiglu: Option<Combine<'a>>,
    /// The caller wrote the combine before the dispatch; the pre-pass only
    /// quantizes.
    produced: bool,
    /// The caller quantized `x` already: the slot has no column to quantize
    /// and no claim, and its rows read these bytes.
    pre: Option<&'a QuantizedCols>,
}

impl QuantSlot<'_> {
    /// Columns the pre-pass quantizes for this slot: none for bytes the
    /// caller quantized.
    fn todo_cols(&self) -> usize {
        if self.pre.is_some() { 0 } else { self.x.ne1 }
    }
}

/// A combine slot's producer: gate, up, the clamp limit when the pair
/// carries one, and the write view of the par block the slot's `x` names.
#[derive(Clone, Copy)]
struct Combine<'a> {
    gate: &'a Tensor2,
    up: &'a Tensor2,
    limit: Option<f32>,
    par: ParWrite,
}

/// The column walker both dispatch shapes ride: quantize columns `cols` of the
/// concatenated slot space, one `quantize_into` call per column,
/// chunk-straddling slots handled segment-wise.
fn quant_range(
    slots: &Flex<QuantSlot<'_>>,
    col_starts: &Flex<usize>,
    shared: &Flex<SharedQuantCols>,
    cols: Range<usize>,
    lvl: u8,
    acc: &mut profile::CallAcc,
) {
    let n = slots.len();
    let mut c = cols.start;
    // The slot whose column range contains `c`: `col_starts` is sorted and
    // `c` only advances within a call, so a forward scan never rewinds and
    // skips zero-width slots instead of landing inside one. Progress holds —
    // `c < cols.end` sits strictly inside slot `d`'s range, so every pass
    // writes at least one column.
    let mut d = 0usize;
    while c < cols.end {
        while d + 1 < n && *col_starts.get(d + 1) <= c {
            d += 1;
        }
        let t0 = c - *col_starts.get(d);
        let t_end = t0 + (cols.end - c).min(slots.get(d).todo_cols() - t0);
        let t_q = if lvl >= 2 { Some(Instant::now()) } else { None };
        let s = slots.get(d);
        match s.swiglu {
            // A combine nobody produced yet: this column's combine, then its
            // quantization — the calls a claimant makes, one column at a time.
            Some(comb) if !s.produced => {
                for t in t0..t_end {
                    debug_assert!(
                        (t + 1) * s.k <= comb.par.1,
                        "column {t} inside the par block"
                    );
                    // SAFETY: `comb.par` holds the par block's `k · ne1` cells, lent for the
                    // call, and column `t` of it is this chunk's alone (the construction
                    // site in `quantize_slots`); nothing reads the block before the join.
                    let par =
                        unsafe { std::slice::from_raw_parts_mut(comb.par.0.add(t * s.k), s.k) };
                    swiglu_into(comb.gate.col(t), comb.up.col(t), comb.limit, par);
                    // SAFETY: column `t` of slot `d` is inside this chunk's sub-range — see the construction site in `quantize_slots`.
                    unsafe { shared.get(d).quantize_into(s.ty, s.k, par, t) };
                }
            }
            _ => {
                for t in t0..t_end {
                    // SAFETY: column `t` of slot `d` is inside this chunk's sub-range — see the construction site in `quantize_slots`.
                    unsafe { shared.get(d).quantize_into(s.ty, s.k, s.x.col(t), t) };
                }
            }
        }
        if let Some(t_q) = t_q {
            acc.add_quant_act(t_q.elapsed().as_nanos() as u64);
        }
        c = *col_starts.get(d) + t_end;
    }
}

/// Column count under which the caller-side pre-pass runs inline. A pool
/// dispatch has a fixed floor — the pool bench's empty-closure dispatch —
/// that a handful of columns cannot repay, and a pre-pass group carries one
/// column per slot. Either way the encoder runs once per column, so the
/// bytes are identical; only WHO calls it changes. The decode shape pays no
/// pre-pass at all while the deferral lever is on: the row dispatch's
/// participants claim the slots themselves ([`DeferredSlots`]), so the cap
/// bounds the pre-pass arms only.
const QUANT_INLINE_COLS: usize = 8;

/// Ceiling on the slots a deferred dispatch carries: the claim states sit
/// inline in the dispatch state, so a deferred dispatch never allocates. A
/// wider one-column group keeps the caller-side pre-pass; the step's decode
/// groups (one slot for the attention pair, one for gate/up, one `par` per
/// down pair) all sit far below it.
const MAX_DEFER_SLOTS: usize = 16;

/// `BLOOMERY_DEFER_QUANT=0` forces today's caller-side pre-pass — the A/B
/// lever. Same binary, same bytes: only WHO quantizes changes.
fn defer_quant_env() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("BLOOMERY_DEFER_QUANT").map_or(true, |v| v != "0"))
}

/// The in-process override the deferral tests drive: the env var is read
/// once per process, so this is how one binary exercises both arms.
static DEFER_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Test override for the deferral lever: `Some(true)` forces the deferred
/// path, `Some(false)` forces the caller-side pre-pass, `None` follows
/// `BLOOMERY_DEFER_QUANT`.
#[doc(hidden)]
pub fn set_defer_quant(mode: Option<bool>) {
    DEFER_OVERRIDE.store(
        match mode {
            Some(true) => 1,
            Some(false) => 2,
            None => 0,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

fn defer_quant() -> bool {
    match DEFER_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => defer_quant_env(),
    }
}

/// A deferred slot's claim state.
const SLOT_TODO: u8 = 0;
const SLOT_CLAIMED: u8 = 1;
const SLOT_DONE: u8 = 2;

/// Publishes DONE when a claim's scope ends, on unwind too: a producer that
/// panics must not leave waiters spinning past the pool's barrier — the
/// panic unwinds every participant's chunk and the dispatcher rethrows it
/// once the join completes, discarding the outputs with it.
struct ClaimDone<'a>(&'a std::sync::atomic::AtomicU8);
impl Drop for ClaimDone<'_> {
    fn drop(&mut self) {
        self.0
            .store(SLOT_DONE, std::sync::atomic::Ordering::Release);
    }
}

/// The claim table of a deferred dispatch: the slot list, the shared write
/// views, and one state per slot. The decode shape carries one column per
/// slot ([`matmul_q_group_cols_into`] at most [`DEFER_MAX_COLS`]), so a
/// claim is one SwiGLU combine at most plus one encoder call per column,
/// and every wait is bounded by exactly that.
struct DeferredSlots<'a> {
    slots: &'a Flex<QuantSlot<'a>>,
    shared: &'a Flex<SharedQuantCols>,
    states: [std::sync::atomic::AtomicU8; MAX_DEFER_SLOTS],
    /// SwiGLU combine time, summed by the claimers (level >= 1) and recorded
    /// once on the caller after the join — the profiler's lock stays out of
    /// the workers.
    swiglu_ns: std::sync::atomic::AtomicU64,
    /// The longest claim pass of the dispatch, from its first claim to its
    /// end (level >= 1): the stage every row of a claimed slot waits behind.
    claim_ns: std::sync::atomic::AtomicU64,
}

impl<'a> DeferredSlots<'a> {
    fn new(slots: &'a Flex<QuantSlot<'a>>, shared: &'a Flex<SharedQuantCols>) -> Self {
        DeferredSlots {
            slots,
            shared,
            // Bytes the caller quantized start DONE: nothing claims or writes them.
            states: std::array::from_fn(|s| {
                let held = s < slots.len() && slots.get(s).pre.is_some();
                std::sync::atomic::AtomicU8::new(if held { SLOT_DONE } else { SLOT_TODO })
            }),
            swiglu_ns: std::sync::atomic::AtomicU64::new(0),
            claim_ns: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// A participant's first work in the dispatch: walk every slot and claim
    /// each still-todo one, producing and quantizing it. A claim is held
    /// only across the slot's own produce+quantize, which waits on
    /// nothing — so no participant can wait on a slot whose claim is
    /// stalled behind another wait, and the wait graph stays acyclic.
    fn claim_pass(&self, lvl: u8, acc: &mut profile::CallAcc) {
        let mut t_first: Option<Instant> = None;
        for s in 0..self.slots.len() {
            // Cheap look first: a finished or claimed slot costs a load.
            if self.states[s].load(std::sync::atomic::Ordering::Relaxed) != SLOT_TODO {
                continue;
            }
            if self.states[s]
                .compare_exchange(
                    SLOT_TODO,
                    SLOT_CLAIMED,
                    std::sync::atomic::Ordering::Acquire,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_err()
            {
                continue;
            }
            if lvl >= 1 && t_first.is_none() {
                t_first = Some(Instant::now());
            }
            let _done = ClaimDone(&self.states[s]);
            self.produce(s, lvl, acc);
        }
        if let Some(t_first) = t_first {
            self.claim_ns.fetch_max(
                t_first.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }

    /// Produce and quantize claimed slot `s` — the exact calls the inline
    /// pre-pass and `ffn::swiglu` make today, whole columns, one encoder
    /// call per column; only WHICH thread runs them changes. Timers match
    /// those call sites: the combine at level 1, the encoder at level 2.
    fn produce(&self, s: usize, lvl: u8, acc: &mut profile::CallAcc) {
        let q = self.slots.get(s);
        let ne1 = q.x.ne1;
        if let Some(c) = q.swiglu {
            let t_s = if lvl >= 1 { Some(Instant::now()) } else { None };
            // SAFETY: `c.par` aliases this slot's `x`, the par block the entry
            // placed and handed off here. This claimer is the block's only
            // writer: no other participant touches it before the slot
            // observes DONE, and the caller reads it only after the join.
            let par = unsafe { std::slice::from_raw_parts_mut(c.par.0, c.par.1) };
            swiglu_into(&c.gate.data, &c.up.data, c.limit, par);
            if let Some(t_s) = t_s {
                self.swiglu_ns.fetch_add(
                    t_s.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            let t_q = if lvl >= 2 { Some(Instant::now()) } else { None };
            for t in 0..ne1 {
                // SAFETY: every column of the slot is this claimer's
                // exclusively (the compare-exchange above); the par block
                // was written by this same thread just above.
                unsafe {
                    self.shared
                        .get(s)
                        .quantize_into(q.ty, q.k, &par[t * q.k..(t + 1) * q.k], t);
                }
            }
            if let Some(t_q) = t_q {
                acc.add_quant_act(t_q.elapsed().as_nanos() as u64);
            }
        } else {
            let t_q = if lvl >= 2 { Some(Instant::now()) } else { None };
            for t in 0..ne1 {
                // SAFETY: every column of the slot is this claimer's
                // exclusively (the compare-exchange above).
                unsafe { self.shared.get(s).quantize_into(q.ty, q.k, q.x.col(t), t) };
            }
            if let Some(t_q) = t_q {
                acc.add_quant_act(t_q.elapsed().as_nanos() as u64);
            }
        }
    }

    /// Block until slot `s` is quantized. Bounded: the claimer is a
    /// participant already inside this dispatch, and a slot is at most
    /// [`DEFER_MAX_COLS`] columns of production and quantization.
    fn wait_ready(&self, s: usize) {
        while self.states[s].load(std::sync::atomic::Ordering::Acquire) != SLOT_DONE {
            std::hint::spin_loop();
        }
    }

    /// The claimers' SwiGLU time, read on the caller after the join (the
    /// pool's completion protocol orders the adds before the return).
    fn total_swiglu_ns(&self) -> u64 {
        self.swiglu_ns.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The longest claim pass, read on the caller after the join.
    fn longest_claim_ns(&self) -> u64 {
        self.claim_ns.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// The caller-side pre-pass: quantize every slot's activation columns before
/// any row runs — one pool dispatch over the concatenated column space
/// (inline for the small counts below), chunk-straddling slots handled
/// segment-wise. The decode shape does not come through here while the
/// deferral lever is on (see [`run_group`]); `BLOOMERY_DEFER_QUANT=0` and
/// multi-column inputs do (past [`DEFER_MAX_COLS`] for
/// [`matmul_q_group_cols_into`]).
///
/// Bit identity: each column is a pure function of its own input, and the
/// split partitions the output cells; splitting WITHIN a column would need
/// its own argument (block-local scales).
fn quantize_slots(
    slots: &Flex<QuantSlot<'_>>,
    shared: &Flex<SharedQuantCols>,
    lvl: u8,
    pacc: &mut profile::CallAcc,
) {
    let n = slots.len();
    let mut col_starts: Flex<usize> = Flex::new();
    let mut total_cols = 0usize;
    for s in 0..n {
        col_starts.push(total_cols);
        total_cols += slots.get(s).todo_cols();
    }
    // SAFETY (construction site, referenced by every write): the pointers
    // alias buffers owned for the whole call; each participant writes only
    // its own sub-range's columns — and, for a combine the pre-pass produces,
    // only those columns of the par block — the join publishes the writes,
    // and a boundary-straddling column is two segments, never quantized twice.
    // Only a profiled dispatch pays for the chunk collector; the quantizer
    // cannot fail, so an unprofiled one needs no error channel either.
    let collected = if lvl > 0 {
        Some(profile::ChunkSlots::<profile::CallAcc>::new())
    } else {
        None
    };
    // Inline only the decode shape — one column per slot. A multi-column input
    // (prefill) is real work and goes to the pool whatever the count.
    if total_cols > QUANT_INLINE_COLS || total_cols > n {
        threads::pool().for_each_chunk(total_cols, |cols| {
            let mut acc = profile::CallAcc::new();
            quant_range(slots, &col_starts, shared, cols, lvl, &mut acc);
            if let Some(collected) = &collected {
                collected.push(acc);
            }
        });
    } else if total_cols > 0 {
        // The inline shape: the walker runs on the caller, no split —
        // deterministic, `tests/mt.rs` covers this branch too.
        let mut acc = profile::CallAcc::new();
        quant_range(slots, &col_starts, shared, 0..total_cols, lvl, &mut acc);
        if let Some(collected) = &collected {
            collected.push(acc);
        }
    }
    if let Some(collected) = collected {
        for acc in collected.into_vec() {
            pacc.add_acc(&acc);
        }
    }
}

/// Which slot each pair reads, and the columns of it (`None`: all, in order).
struct PairMap<'m, 'c> {
    slot_of: &'m Flex<usize>,
    cols_of: &'m Flex<Option<&'c [usize]>>,
}

/// How [`run_group`] quantizes a group's slots and cuts its rows into lanes.
#[derive(Clone, Copy)]
struct Arm {
    /// The caller's lever reading ([`defer_quant`], or the entry's own
    /// shape-constrained variant) with the widest slot the entry lets a claim
    /// take; `None` is the pre-pass.
    defer_cols: Option<usize>,
    /// Lanes cut by the tile kernel's cost, participant `t` on lane `t`
    /// ([`run_row_pool`]) — a wide group of [`Entry::Cols`].
    tile_lanes: bool,
}

/// Quantize `slots` and run every pair's rows over one pool dispatch — the
/// engine all three entry shapes ride. The shape checks here are the single
/// owner of the deferral decision ([`Arm::defer_cols`]): the decode shape
/// (every slot one column — or, for [`matmul_q_group_cols_into`], at most
/// [`DEFER_MAX_COLS`]) with inline claim states skips the pre-pass, and the
/// row dispatch's participants claim the slots as their first work. A slot
/// of caller-quantized bytes ([`GroupInput::Quantized`]) is neither claimed
/// nor pre-quantized: its rows read those bytes. Returns,
/// at level >= 1, the SwiGLU nanoseconds the claims spent (worker-side, so
/// the caller's profiler wall can hand them to the `swiglu` site instead of
/// counting them twice) and the activation stage's share of the wall (the
/// pre-pass, or the longest claim pass).
#[allow(clippy::too_many_arguments)]
fn run_group(
    slots: &Flex<QuantSlot<'_>>,
    map: &PairMap<'_, '_>,
    bytes: &Flex<&[u8]>,
    meta: &Flex<PairMeta>,
    outs: &mut [Tensor2],
    lvl: u8,
    pacc: &mut profile::CallAcc,
    arm: Arm,
) -> Result<RunTimes, crate::ModelError> {
    let n_slots = slots.len();
    let mut quantized: Flex<QuantCols> = Flex::new();
    for s in 0..n_slots {
        let q = slots.get(s);
        quantized.push(match q.pre {
            Some(xq) => QuantCols::Held(xq.shared()),
            None => QuantCols::new(q.cb, q.x),
        });
    }
    let mut shared: Flex<SharedQuantCols> = Flex::new();
    for s in 0..n_slots {
        shared.push(quantized.get_mut(s).shared());
    }
    let deferred = arm
        .defer_cols
        .filter(|&w| {
            n_slots > 0
                && n_slots <= MAX_DEFER_SLOTS
                && (0..n_slots).all(|s| slots.get(s).x.ne1 <= w)
        })
        .map(|_| DeferredSlots::new(slots, &shared));
    let mut prepass_ns = 0u64;
    if deferred.is_none() {
        let t_q = if lvl > 0 { Some(Instant::now()) } else { None };
        quantize_slots(slots, &shared, lvl, pacc);
        if let Some(t_q) = t_q {
            prepass_ns = t_q.elapsed().as_nanos() as u64;
        }
    }
    let mut pairs: Flex<PairWork> = Flex::new();
    for (i, out) in outs.iter_mut().enumerate() {
        // SAFETY (construction site, referenced by every write): each pair's
        // output pointer aliases that pair's `out.data`, caller-owned for the
        // whole call. Each participant writes only cells `t * n + r` with
        // `r` inside its own sub-range of the row split, which partitions
        // the rows; the join publishes the writes; an early error discards
        // the outputs either way. The quantized-column pointers alias the
        // slot buffers, caller-owned for the whole call: on the pre-pass arm
        // the columns are complete before any row runs; on the deferred arm
        // a participant forms the slice only after the slot's state observes
        // DONE (the claim's release store), which orders the claimer's
        // writes before its read. A column-mapped pair reads only the
        // slot's columns its map names, each checked against the slot's
        // width at the entry.
        let s = *map.slot_of.get(i);
        pairs.push(PairWork::new(
            out,
            bytes.get(i),
            (*shared.get(s), slots.get(s).x.ne1, *map.cols_of.get(i)),
            *meta.get(i),
            s,
            deferred.as_ref(),
        ));
    }
    run_row_pool(&pairs, lvl, pacc, deferred.as_ref(), arm.tile_lanes)?;
    quantized.release_all();
    Ok(match deferred {
        Some(d) => RunTimes {
            swiglu_ns: d.total_swiglu_ns(),
            stage_ns: d.longest_claim_ns(),
        },
        None => RunTimes {
            swiglu_ns: 0,
            stage_ns: prepass_ns,
        },
    })
}

/// What [`run_group`] hands back, level >= 1 only (zeros when off).
struct RunTimes {
    /// The claims' SwiGLU combines, summed over the claimers.
    swiglu_ns: u64,
    /// The activation stage's share of the wall: the pre-pass, or the
    /// longest claim pass.
    stage_ns: u64,
}

/// What one row of a pair costs a lane: its weight bytes once per input column
/// — a row is dotted against every column, and a prefill group mixes pairs of
/// one column with pairs of many.
fn row_cost(p: &PairWork<'_>) -> u64 {
    p.row_bytes as u64 * p.ne1.max(1) as u64
}

/// Instructions per super-block of qdot's tile kernels, in half-instruction
/// units: the fixed unpack of one run of a weight row, and each column's
/// share (the loop body of `qdot::tile::<C>` in the disassembly, C = 2, 3, 4,
/// fitted as fixed + C · per column). A lone column's `dot_row` costs about a
/// run of one. `None`: no tile kernel.
fn tile_units(ty: GgmlType) -> Option<(u64, u64)> {
    match ty {
        GgmlType::Q3_K => Some((192, 63)),
        GgmlType::Q4_K => Some((96, 97)),
        GgmlType::Q5_K => Some((156, 98)),
        _ => None,
    }
}

/// What one row of a pair costs a lane when its columns go through the tile
/// kernel in runs of up to [`qdot::TILE_COLS`], as `compute_rows` walks them:
/// per super-block one fixed unpack per run plus each column's share. `None`
/// for a pair without a tile kernel on this machine.
fn tile_cost(p: &PairWork<'_>) -> Option<u64> {
    let (fixed, per_col) = tile_units(p.ty)?;
    if p.fused_cols.is_none() || !qdot::has_tile(p.ty) {
        return None;
    }
    let m = p.ne1.max(1) as u64;
    let sb = (p.k / qdot::k_granularity(p.ty)) as u64;
    Some(sb * (m.div_ceil(qdot::TILE_COLS as u64) * fixed + m * per_col))
}

/// One lane of a row dispatch: the next unclaimed row and the lane's end, on
/// its own cache line so an owner's claims do not fight its neighbours'.
#[repr(align(64))]
struct Lane {
    next: std::sync::atomic::AtomicUsize,
    end: usize,
    /// Rows per claim. Per lane, not per dispatch: cost-cut lanes differ in row
    /// count, and a lane of few expensive rows claimed as one block leaves its
    /// tail nothing to share.
    block: usize,
}
const MAX_LANES: usize = 64;
fn steal_enabled() -> bool {
    static S: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("BLOOMERY_STEAL").map_or(true, |v| v != "0"))
}
/// Blocks a lane is cut into: the barrier waits for at most one block per
/// participant once every block is claimed. `BLOOMERY_STEAL_BLOCKS` is the
/// same-binary lever for that granularity.
fn steal_blocks() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("BLOOMERY_STEAL_BLOCKS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&b| b > 0)
            .unwrap_or(4)
    })
}

/// The unprofiled dispatch's error channel: no chunk collector, no
/// allocation — a clean chunk touches nothing, and the mutex only locks when
/// an error exists. The winning error is the lowest failing row's, exactly
/// what the sorted scan decided on the profiled path.
pub(crate) struct ErrGate {
    any: std::sync::atomic::AtomicBool,
    slot: std::sync::Mutex<Option<(usize, crate::ModelError)>>,
}

impl ErrGate {
    pub(crate) fn new() -> Self {
        ErrGate {
            any: std::sync::atomic::AtomicBool::new(false),
            slot: std::sync::Mutex::new(None),
        }
    }

    /// Offer the error of the chunk that starts at `start`; a lower start
    /// replaces a higher one. Compared under the lock, so the payload and the
    /// minimum cannot come from different chunks.
    pub(crate) fn offer(&self, start: usize, e: crate::ModelError) {
        let mut slot = self.slot.lock().expect("row error slot");
        if slot.as_ref().is_none_or(|(s, _)| start < *s) {
            *slot = Some((start, e));
        }
        self.any.store(true, std::sync::atomic::Ordering::Release);
    }

    pub(crate) fn take(&self) -> Option<crate::ModelError> {
        if !self.any.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        self.slot
            .lock()
            .expect("row error slot")
            .take()
            .map(|(_, e)| e)
    }
}

/// The row dispatch and its gather: one pool job over the concatenated rows
/// of every pair (a chunk may straddle pair boundaries; the walker hands each
/// pair its own sub-range), then the sort, error scan and accumulator fold.
/// The split axis is the output row `r` and only that — each element still
/// accumulates its k products ascending, so the worker split cannot change a
/// bit (`tests/mt.rs`, `tests/ops.rs`); splitting k or the token axis is
/// forbidden.
///
/// Lanes are cut by cumulative COST (`row_cost`: weight bytes × input columns), not row count: a group's pairs
/// differ in per-row cost, so a row-count cut would stack the cheap pairs on
/// some lanes and the expensive ones on others. Lane `t` is the rows where
/// the running byte cost crosses `t/nlanes` of the total — deterministic,
/// row-granular; a uniform group degenerates to a row-count cut.
///
/// With `tile_lanes` (a wide union group) and a tile kernel for every pair,
/// the cost is `tile_cost` — a fixed unpack per run of columns plus each
/// column — and participant `t` starts on lane `t`: the pool splits the lane
/// indices, not the rows, so every lane the cut balanced has the participant
/// it was cut for. Otherwise the row split names the home lanes, as below.
///
/// A profiled dispatch pays the chunk collector (one `Vec` of pool-width
/// slots); an unprofiled one runs the error channel above and allocates
/// nothing unless a chunk fails. A dispatch with a claim table runs it as
/// the participants' first work (`DeferredSlots::claim_pass`); a claimed
/// slot's state gates every read of its columns (`PairWork::compute_rows`).
fn run_row_pool(
    pairs: &Flex<PairWork<'_>>,
    lvl: u8,
    pacc: &mut profile::CallAcc,
    deferred: Option<&DeferredSlots<'_>>,
    tile_lanes: bool,
) -> Result<(), crate::ModelError> {
    let npairs = pairs.len();
    let mut pair_starts: Flex<usize> = Flex::new();
    let mut total_rows = 0usize;
    pair_starts.push(0);
    for p in 0..npairs {
        total_rows += pairs.get(p).n;
        pair_starts.push(total_rows);
    }
    let collected = if lvl > 0 {
        Some(profile::ChunkSlots::<RowChunk>::new())
    } else {
        None
    };
    let gate = ErrGate::new();
    let t_span = if lvl >= 1 { Some(Instant::now()) } else { None };
    // Lanes: the pool's own static split, one cursor each. A participant walks
    // its home lanes front to back in blocks, then takes blocks off the front of
    // whatever lanes are still unfinished. The slowest chunk of a dispatch runs
    // 20–40 % over the mean and which chunk that is changes every dispatch, so
    // the barrier waits on noise; the tail is the only part worth sharing, and
    // the owner still streams its rows contiguously. A row's value does not
    // depend on who computes it.
    let nlanes = threads::pool().threads();
    assert!(nlanes <= MAX_LANES, "row pool lanes: {nlanes} threads");
    let tile = tile_lanes && (0..npairs).all(|p| tile_cost(pairs.get(p)).is_some());
    // One cost model for the whole cut: the tile's for every pair, or bytes.
    let cost = |p: &PairWork<'_>| match tile.then(|| tile_cost(p)).flatten() {
        Some(c) => c,
        None => row_cost(p),
    };
    let mut lane_bounds = [0usize; MAX_LANES + 1];
    {
        let total_cost: u64 = (0..npairs)
            .map(|p| pairs.get(p).n as u64 * cost(pairs.get(p)))
            .sum();
        // Closed form per pair — a pair's rows cost the same, so the first row
        // where `nlanes·cost` reaches `t·total_cost` is a division, not a walk
        // over the rows (a walk is O(rows) on the calling thread, per dispatch).
        let nl = nlanes as u64;
        let mut p = 0usize;
        let mut before = 0u64; // cost of every pair ahead of `p`
        for (t, bound) in lane_bounds.iter_mut().enumerate().take(nlanes).skip(1) {
            let target = t as u64 * total_cost;
            loop {
                if p >= npairs {
                    *bound = total_rows;
                    break;
                }
                let c = cost(pairs.get(p));
                let n_p = pairs.get(p).n as u64;
                if c == 0 || (before + n_p * c) * nl < target {
                    before += n_p * c;
                    p += 1;
                    continue;
                }
                // Smallest j with (before + j·c)·nl >= target.
                let need = target.saturating_sub(before * nl);
                let j = need.div_ceil(c * nl).min(n_p);
                *bound = *pair_starts.get(p) + j as usize;
                break;
            }
        }
        lane_bounds[nlanes] = total_rows;
    }
    // `BLOOMERY_STEAL=0` is the A/B lever: whole-lane blocks, home lanes only.
    let steal = steal_enabled();
    let steal_blocks = steal_blocks();
    let lanes: [Lane; MAX_LANES] = std::array::from_fn(|t| {
        let (start, end) = if t < nlanes {
            (lane_bounds[t], lane_bounds[t + 1])
        } else {
            (0, 0)
        };
        let block = if steal {
            ((end - start) / steal_blocks).max(1)
        } else {
            total_rows.max(1)
        };
        Lane {
            next: std::sync::atomic::AtomicUsize::new(start),
            end,
            block,
        }
    });
    // The tile cut hands participant `t` the index `t` (`nlanes` is the pool's
    // width, so each chunk is one index); the byte cut hands it rows.
    let items = if tile { nlanes } else { total_rows };
    threads::pool().for_each_chunk(items, |r| {
        let t_busy = if lvl >= 1 { Some(Instant::now()) } else { None };
        let mut chunk = RowChunk {
            // A tile participant's errors are ordered by its lane's first row.
            start: if tile { lane_bounds[r.start] } else { r.start },
            busy_ns: 0,
            acc: profile::CallAcc::new(),
            err: None,
        };
        // Deferred quantization: claims come first, before any row and any
        // wait — a claim never waits on anything, so claim order cannot cycle.
        if let Some(deferred) = deferred {
            deferred.claim_pass(lvl, &mut chunk.acc);
        }
        let (origin, span) = if tile {
            // Lane `t` is this participant's (the pool hands each participant
            // one index here); with stealing it walks on through the others.
            let span = if r.is_empty() {
                0
            } else if steal {
                nlanes
            } else {
                r.len()
            };
            (r.start, span)
        } else {
            // Home lanes: the lane containing this chunk's start, plus every
            // lane that starts inside the chunk. Byte-cut boundaries need not
            // align with the pool's row-count chunks, and a start-match lookup
            // would leave an interior-starting lane ownerless when stealing is
            // off — the chunk that contains a lane's start owns it, so every
            // lane has exactly one. An empty chunk owns none. The aligned
            // (uniform) case reduces to one home lane per chunk.
            let rows = r;
            let mut first = 0usize;
            while first + 1 < nlanes && lane_bounds[first + 1] <= rows.start {
                first += 1;
            }
            let mut last = first + 1;
            while last < nlanes && lane_bounds[last] < rows.end {
                last += 1;
            }
            if rows.is_empty() {
                (first, 0)
            } else if steal {
                (first, nlanes)
            } else {
                (first, last - first)
            }
        };
        'lanes: for off in 0..span {
            let lane = &lanes[(origin + off) % nlanes];
            loop {
                // A cheap look first: a finished lane costs a read, not a write.
                if lane.next.load(std::sync::atomic::Ordering::Relaxed) >= lane.end {
                    break;
                }
                let from = lane
                    .next
                    .fetch_add(lane.block, std::sync::atomic::Ordering::Relaxed);
                if from >= lane.end {
                    break;
                }
                let to = lane.end.min(from + lane.block);
                let mut gr = from;
                let mut p = 0usize;
                while p + 1 < npairs && gr >= *pair_starts.get(p + 1) {
                    p += 1;
                }
                while gr < to {
                    let sub_end = to.min(*pair_starts.get(p + 1));
                    let r0 = gr - *pair_starts.get(p);
                    if let Err(e) =
                        pairs
                            .get(p)
                            .compute_rows(r0..r0 + (sub_end - gr), lvl, &mut chunk.acc)
                    {
                        chunk.err = Some(e);
                        break 'lanes;
                    }
                    gr = sub_end;
                    while p + 1 < npairs && gr >= *pair_starts.get(p + 1) {
                        p += 1;
                    }
                }
            }
        }
        if let Some(t_busy) = t_busy {
            chunk.busy_ns = t_busy.elapsed().as_nanos() as u64;
        }
        // A clean unprofiled chunk has nothing to report.
        if let Some(collected) = &collected {
            collected.push(chunk);
        } else if let Some(e) = chunk.err {
            gate.offer(chunk.start, e);
        }
    });

    let span_ns = t_span.map(|t| t.elapsed().as_nanos() as u64);
    if let Some(collected) = collected {
        // Arrival order is nondeterministic; sorting by `start` keeps the first
        // error the lowest failing row's, as a sequential `?` would return.
        let t_gather = if lvl >= 2 { Some(Instant::now()) } else { None };
        let mut chunks = collected.into_vec();
        chunks.sort_by_key(|c| c.start);
        if let Some(e) = chunks.iter_mut().find_map(|c| c.err.take()) {
            return Err(e);
        }
        for c in &chunks {
            pacc.add_acc(&c.acc);
        }
        if lvl >= 1 {
            for (i, c) in chunks.iter().enumerate() {
                profile::add_chunk_busy(i, c.busy_ns);
            }
        }
        if let Some(span_ns) = span_ns {
            pacc.add_span(span_ns, chunks.iter().map(|c| c.busy_ns).max().unwrap_or(0));
        }
        if let Some(t_gather) = t_gather {
            pacc.add_gather(t_gather.elapsed().as_nanos() as u64);
        }
    } else if let Some(e) = gate.take() {
        return Err(e);
    }
    Ok(())
}

/// The single-pair dispatch `matmul_q` rides: one weight, one input, no
/// per-call `Vec`s — the quantization buffer off the thread pool, the one
/// `PairWork` inline. Same row loop, same `batch_shape` contract, same
/// profiler row as the batch shape; only the bookkeeping is narrower. The
/// decode shape (one column) defers its quantization into the row dispatch
/// like every other group.
fn matmul_q_one(
    site: &'static str,
    gguf: &Gguf,
    w: &TensorInfo,
    x: &Tensor2,
) -> Result<Tensor2, crate::ModelError> {
    let Some((k, n, ty)) = batch_shape(std::slice::from_ref(&w), std::slice::from_ref(&x))? else {
        unreachable!("a one-pair batch is not empty")
    };
    // Profiler hook: level 0 one compare, level 1 one `Instant` pair, level 2
    // two more per row, none touching the arithmetic; chunks merge into one
    // `CallAcc`, so `record` fires once per call.
    let lvl = profile::level();
    let mut pacc = profile::CallAcc::new();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let bytes = gguf.data(w)?;
    let row_bytes = bytes.len() / n;
    let fused = qdot::fuses(ty, k);
    let cb = if fused {
        Some(qdot::col_bytes(ty, k))
    } else {
        // The scalar path quantizes into the weight type's activation format;
        // a type without one is refused here, before any column exists.
        gguf::activation_format(ty)?;
        None
    };

    let mut slots: Flex<QuantSlot> = Flex::new();
    slots.push(QuantSlot {
        x,
        ty,
        k,
        cb,
        swiglu: None,
        produced: false,
        pre: None,
    });
    let mut slot_of: Flex<usize> = Flex::new();
    slot_of.push(0);
    let mut cols_of: Flex<Option<&[usize]>> = Flex::new();
    cols_of.push(None);
    let mut bytes_f: Flex<&[u8]> = Flex::new();
    bytes_f.push(bytes);
    let mut meta: Flex<PairMeta> = Flex::new();
    meta.push(PairMeta {
        ty,
        k,
        n,
        row_bytes,
    });
    let mut out = Tensor2::scratch(n, x.ne1);
    // SAFETY: the construction-site argument in `run_group`'s pair build —
    // one pair, the whole row split, the join publishes the writes.
    run_group(
        &slots,
        &PairMap {
            slot_of: &slot_of,
            cols_of: &cols_of,
        },
        &bytes_f,
        &meta,
        std::slice::from_mut(&mut out),
        lvl,
        &mut pacc,
        Arm {
            defer_cols: defer_quant().then_some(1),
            tile_lanes: false,
        },
    )?;

    if let Some(t_call) = t_call {
        profile::record(
            site,
            ty,
            n as u64,
            k as u64 * n as u64,
            bytes.len() as u64,
            t_call.elapsed().as_nanos() as u64,
            &pacc,
        );
    }
    Ok(out)
}

/// `matmul_q_one` for a one-column input, on the calling thread, no pool
/// dispatch: the fused attention round calls it from inside one pool row,
/// where a nested dispatch would idle every worker for a barrier per head.
/// This is the `ne1 = 1` shape of `PairWork::compute_rows` — same `fused`
/// decision, same `quantize_col`/`quantize_activations` encoder into the
/// same recycled poisoned buffers, same per-row op order — so the values are
/// those of a one-column `matmul_q` and only WHICH thread computes them
/// changes (`tests/attn.rs` pins the bit identity end to end).
pub(crate) fn matvec_q_local(
    gguf: &Gguf,
    w: &TensorInfo,
    x_col: &[f32],
    out: &mut [f32],
) -> Result<(), crate::ModelError> {
    let k = w.dims.first().copied().unwrap_or(0) as usize;
    let n = if w.dims.len() > 1 {
        w.dims[1] as usize
    } else {
        1
    };
    let ty = w.ty;
    if x_col.len() != k {
        return Err(crate::ModelError::Shape {
            what: "matmul_q input",
            want_ne0: k,
            want_ne1: 1,
            got_ne0: x_col.len(),
            got_ne1: 1,
        });
    }
    if out.len() != n {
        return Err(crate::ModelError::Shape {
            what: "matvec_q out",
            want_ne0: n,
            want_ne1: 1,
            got_ne0: out.len(),
            got_ne1: 1,
        });
    }
    let bytes = gguf.data(w)?;
    let row_bytes = bytes.len() / n;
    let fused = qdot::fuses(ty, k);
    if fused {
        // The QuantCols::new contract: recycled byte buffer, poisoned under
        // BLOOMERY_POISON, one `quantize_col` writing the whole column.
        let mut buf = take_u8(qdot::col_bytes(ty, k));
        if poison() {
            buf.fill(0xA5);
        }
        qdot::quantize_col(ty, x_col, &mut buf);
        for r in 0..n {
            let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
            out[r] = qdot::dot_row(ty, src, &buf, k)?;
        }
        give_u8(buf);
    } else {
        let mut qbuf = take_f32(k);
        if poison() {
            qbuf.fill(f32::NAN);
        }
        quantize_activations(ty, x_col, &mut qbuf)?;
        ROW_BUF.with(|cell| -> Result<(), crate::ModelError> {
            let mut scratch = cell.borrow_mut();
            if scratch.len() < k {
                scratch.resize(k, 0.0);
            }
            let row = &mut scratch[..k];
            for r in 0..n {
                let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
                if ty == GgmlType::F32 {
                    if let Some(v) = qdot::dot_f32(src, &qbuf[..k]) {
                        out[r] = v;
                        continue;
                    }
                }
                // Weight side: dequant_row, or the byte walk in its place for F32 rows.
                if ty == GgmlType::F32 {
                    let (words, _) = src.as_chunks::<4>();
                    let words = &words[..row.len()];
                    for (v, w) in row.iter_mut().zip(words) {
                        *v = f32::from_le_bytes(*w);
                    }
                } else {
                    dequant_row(ty, src, row)?;
                }
                let xc = &qbuf[..k];
                let mut a = 0.0f32;
                for i in 0..k {
                    a += row[i] * xc[i];
                }
                out[r] = a;
            }
            Ok(())
        })?;
        give_f32(qbuf);
    }
    Ok(())
}

/// Quantization slots the most recent `matmul_q_multi` call built — the
/// sharing rule's observable for `tests/ops.rs`: pairs over one input with
/// one encoding must collapse to a single slot, pairs whose encodings differ
/// must not.
static LAST_QUANT_SLOTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The reading side of [`LAST_QUANT_SLOTS`].
#[doc(hidden)]
pub fn last_quant_slots() -> usize {
    LAST_QUANT_SLOTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// One pool dispatch over the concatenated rows of every `(W_i, x_i)` pair —
/// heterogeneous: each pair carries its own type, `k`, `n` and row bytes, and
/// the lanes balance weight bytes. `site` names the profiler row;
/// [`matmul_q_batch`] is the homogeneous front (`batch_shape` checked first)
/// and [`matmul_q_group`] the heterogeneous one.
fn matmul_q_multi(
    site: &'static str,
    gguf: &Gguf,
    ws: &[&TensorInfo],
    xs: &[&Tensor2],
) -> Result<Vec<Tensor2>, crate::ModelError> {
    let (mut outs, mut pars) = (Vec::new(), Vec::new());
    group_core(
        Weights::File(gguf, ws),
        Inputs::Ready(xs),
        Dest::Alloc {
            site,
            outs: &mut outs,
            pars: &mut pars,
        },
        Entry::Plain,
    )?;
    Ok(outs)
}

#[cfg(test)]
mod tests {
    use super::{
        GroupInput, Tensor2, matmul_q, matmul_q_group, matmul_q_group_swiglu, matvec_q_local,
    };
    use gguf::{GgmlType, Gguf};

    /// A GGUF v3 file holding one `[32, 2]` tensor `w` of type `ty` over zero bytes —
    /// the least the strict reader opens.
    fn one_tensor_file(ty: GgmlType) -> std::path::PathBuf {
        let rows = 2u64;
        let bytes = ty.type_size().unwrap() * (32 / ty.blck_size().unwrap()) * rows;
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensors
        b.extend_from_slice(&0u64.to_le_bytes()); // metadata pairs
        b.extend_from_slice(&1u64.to_le_bytes());
        b.push(b'w');
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&32u64.to_le_bytes());
        b.extend_from_slice(&rows.to_le_bytes());
        b.extend_from_slice(&ty.as_u32().to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // offset
        b.resize(b.len().div_ceil(32) * 32 + bytes as usize, 0);
        let path =
            std::env::temp_dir().join(format!("bloomery-ops-{}-{ty}.gguf", std::process::id()));
        std::fs::write(&path, b).unwrap();
        path
    }

    fn refusal<T>(r: Result<T, crate::ModelError>) -> String {
        match r {
            Ok(_) => panic!("a CPU matmul over this weight type must be refused"),
            Err(e) => e.to_string(),
        }
    }

    /// No CPU matmul takes a Q8_0 or BF16 weight: every entry that reaches the scalar
    /// arm of the fused decision refuses it with an error naming the type, instead of
    /// running it against f32 activations.
    #[test]
    fn matmul_refuses_weight_types_without_an_activation_format() {
        for ty in [GgmlType::Q8_0, GgmlType::BF16] {
            let path = one_tensor_file(ty);
            let g = Gguf::open(&path).unwrap();
            let w = g.find("w").unwrap();
            let x = Tensor2::zeros(32, 1);
            let mut out = [0.0f32; 2];
            let errors = [
                refusal(matmul_q(&g, w, &x)),
                refusal(matmul_q_group(&g, &[w], &[&x])),
                refusal(matmul_q_group_swiglu(&g, &[w], &[GroupInput::Ready(&x)])),
                refusal(matvec_q_local(&g, w, x.col(0), &mut out)),
            ];
            for e in &errors {
                assert!(
                    e.contains(&ty.to_string()),
                    "{ty}: the refusal must name the type, got {e:?}"
                );
            }
            std::fs::remove_file(&path).unwrap();
        }
    }

    /// A column-mapped input's output has one column per listed column, a
    /// repeat included, and a listed column past the block is refused by
    /// name before any byte is read — never clamped to the last column.
    #[test]
    fn cols_input_shapes_its_output_and_refuses_a_column_past_the_block() {
        let path = one_tensor_file(GgmlType::F32);
        let g = Gguf::open(&path).unwrap();
        let w = g.find("w").unwrap();
        let x = Tensor2::zeros(32, 2);
        let (outs, pars) =
            matmul_q_group_swiglu(&g, &[w], &[GroupInput::Cols(&x, &[1, 0, 1])]).unwrap();
        assert_eq!(
            (outs[0].ne0, outs[0].ne1),
            (2, 3),
            "one output column per listed column"
        );
        assert!(
            pars.is_empty(),
            "a column-mapped input is the caller's block"
        );
        let e = refusal(matmul_q_group_swiglu(
            &g,
            &[w],
            &[GroupInput::Cols(&x, &[0, 2])],
        ));
        assert!(
            e.contains("past the block") && e.contains("got [32, 2]"),
            "the refusal names the column, got {e:?}"
        );
        std::fs::remove_file(&path).unwrap();
    }
}
