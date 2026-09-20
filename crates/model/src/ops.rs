//! The two primitives every module needs, written once — reference
//! implementations, fast path in `crates/q3k-cpu`. Accumulation is f32 in
//! ggml's own order (one row at a time, k ascending): a different order moves
//! the last bit, which is what lets the gates stay tight.

use crate::profile;
use gguf::{GgmlType, Gguf, TensorInfo, dequant_row, quantize_activations};
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
        let mut sum = 0.0f32;
        for &v in src {
            sum += v * v;
        }
        let scale = 1.0f32 / (sum / x.ne0 as f32 + eps).sqrt();
        let dst = out.col_mut(t);
        for i in 0..x.ne0 {
            dst[i] = src[i] * scale * gain[i];
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
    /// disjoint-cells argument at `quantize_distinct`'s construction site.
    fn shared(&mut self) -> SharedQuantCols {
        match self {
            QuantCols::Bytes { cb, buf } => SharedQuantCols::Bytes {
                cb: *cb,
                ptr: buf.as_mut_ptr(),
            },
            QuantCols::F32(q) => SharedQuantCols::F32(q.as_mut_ptr()),
        }
    }

    /// Hand the storage back to the pool.
    fn release(self) {
        match self {
            QuantCols::Bytes { buf, .. } => give_u8(buf),
            QuantCols::F32(q) => give_f32(q),
        }
    }
}

/// `matmul_q`'s output pointer in the one form the pool closure can capture
/// (raw pointers are neither `Send` nor `Sync`); the impls add no safety of
/// their own — see the construction site.
pub(crate) struct SharedOut(pub(crate) *mut f32);
// SAFETY: see the struct doc — disjoint cells and join ordering, argued at the construction site.
unsafe impl Send for SharedOut {}
// SAFETY: same argument; workers write only cells their own row range owns.
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

/// [`SharedOut`] for a distinct input's quantized-column buffer; the impls add no safety of their own — see the `quantize_distinct` construction site.
#[derive(Clone, Copy)]
enum SharedQuantCols {
    /// `quantize_activations(w.ty, ..)` — f32 round trip, `k` cells per column.
    F32(*mut f32),
    /// `qdot::quantize_col(w.ty, ..)` — one `cb`-byte column per token.
    Bytes { cb: usize, ptr: *mut u8 },
}
// SAFETY: see the enum doc — disjoint cells and join ordering, argued at the construction site.
unsafe impl Send for SharedQuantCols {}
// SAFETY: same argument; workers write only their own column range's cells.
unsafe impl Sync for SharedQuantCols {}

impl SharedQuantCols {
    /// Quantize one column into its cell — the exact call a serial pre-pass would make; the split changes WHO calls it, never the call.
    ///
    /// # Safety
    ///
    /// `t` inside the caller's sub-range of the split; `x_col` is column
    /// `t` of this buffer's input.
    unsafe fn quantize_into(&self, ty: GgmlType, k: usize, x_col: &[f32], t: usize) {
        match self {
            SharedQuantCols::F32(ptr) => {
                // SAFETY: the disjoint-cell contract above; the cell is `k` f32s, as allocated.
                let cell = unsafe { std::slice::from_raw_parts_mut(ptr.add(t * k), k) };
                quantize_activations(ty, x_col, cell);
            }
            SharedQuantCols::Bytes { cb, ptr } => {
                // SAFETY: same contract; the cell is `cb` bytes, as `col_bytes` allocated.
                let cell = unsafe { std::slice::from_raw_parts_mut(ptr.add(t * cb), *cb) };
                qdot::quantize_col(ty, x_col, cell);
            }
        }
    }
}

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

    /// Move entry `i` out — the single-entry handoff in `quantize_one`.
    fn take(&mut self, i: usize) -> T {
        self.len -= 1;
        if i < GROUP_INLINE {
            self.inline[i].take().expect("index below len is filled")
        } else {
            self.heap.swap_remove(i - GROUP_INLINE)
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
    /// Fused qdot columns: (stride `cb`, quantized byte buffer).
    fused_cols: Option<(usize, &'a [u8])>,
    /// Scalar-path f32 round trip; empty on the fused path.
    q_f32: &'a [f32],
    ne1: usize,
    out: SharedOut,
}

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
    /// is decided.
    fn new(out: &mut Tensor2, bytes: &'a [u8], q: &'a QuantCols, m: PairMeta) -> PairWork<'a> {
        let out_ptr = SharedOut(out.data.as_mut_ptr());
        let (fused_cols, q_f32) = match q {
            QuantCols::Bytes { cb, buf } => (Some((*cb, buf.as_slice())), &[][..]),
            QuantCols::F32(q) => (None, q.as_slice()),
        };
        PairWork {
            ty: m.ty,
            k: m.k,
            n: m.n,
            bytes,
            row_bytes: m.row_bytes,
            fused_cols,
            q_f32,
            ne1: out.ne1,
            out: out_ptr,
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
        if let Some((cb, acol)) = self.fused_cols {
            // Fused rows: `qdot::dot_row` dots the weight bytes and the
            // quantized column directly — no f32 row, ROW_BUF untouched;
            // `add_dequant_w` stays 0 (the dequant is in the dot timer).
            for r in rows {
                let src = &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
                let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                for t in 0..self.ne1 {
                    let v = qdot::dot_row(ty, src, &acol[t * cb..(t + 1) * cb], k)?;
                    // SAFETY: cell `t * n + r` is inside this chunk's row range — see the SharedOut construction site.
                    unsafe { self.out.write(t * n + r, v) };
                }
                if let Some(t_dot) = t_dot {
                    acc.add_dot(t_dot.elapsed().as_nanos() as u64);
                }
            }
            Ok(())
        } else {
            ROW_BUF.with(|cell| {
                let mut scratch = cell.borrow_mut();
                if scratch.len() < k {
                    scratch.resize(k, 0.0);
                }
                let row = &mut scratch[..k];
                for r in rows {
                    let src = &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
                    // Weight side: dequant_row, or the byte walk in its place for F32 rows.
                    let t_d = if lvl >= 2 { Some(Instant::now()) } else { None };
                    if ty == GgmlType::F32 {
                        for (i, v) in row.iter_mut().enumerate() {
                            *v = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
                        }
                    } else {
                        dequant_row(ty, src, row)?;
                    }
                    if let Some(t_d) = t_d {
                        acc.add_dequant_w(t_d.elapsed().as_nanos() as u64);
                    }
                    let t_dot = if lvl >= 2 { Some(Instant::now()) } else { None };
                    for t in 0..self.ne1 {
                        let xc = &self.q_f32[t * k..(t + 1) * k];
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
    x: &'a Tensor2,
    ty: GgmlType,
    k: usize,
    /// `qdot::col_bytes(ty, k)` on the fused path, `None` on the scalar one.
    cb: Option<usize>,
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
        let t_end = t0 + (cols.end - c).min(slots.get(d).x.ne1 - t0);
        let t_q = if lvl >= 2 { Some(Instant::now()) } else { None };
        let s = slots.get(d);
        for t in t0..t_end {
            // SAFETY: column `t` of slot `d` is inside this chunk's sub-range — see the construction site in `quantize_slots`.
            unsafe { shared.get(d).quantize_into(s.ty, s.k, s.x.col(t), t) };
        }
        if let Some(t_q) = t_q {
            acc.add_quant_act(t_q.elapsed().as_nanos() as u64);
        }
        c = *col_starts.get(d) + t_end;
    }
}

/// Quantize one input's columns — the single-pair shape: one slot, no
/// per-call lists. Bit-identity argument as [`quantize_slots`].
fn quantize_one(
    ty: GgmlType,
    k: usize,
    x: &Tensor2,
    fused: bool,
    lvl: u8,
    pacc: &mut profile::CallAcc,
) -> QuantCols {
    let cb = if fused {
        Some(qdot::col_bytes(ty, k))
    } else {
        None
    };
    let mut slots: Flex<QuantSlot> = Flex::new();
    slots.push(QuantSlot { x, ty, k, cb });
    let mut out: Flex<QuantCols> = Flex::new();
    quantize_slots(&slots, &mut out, lvl, pacc);
    out.take(0)
}

/// Column count under which the pre-pass runs inline on the caller. A pool
/// dispatch has a fixed floor — the pool bench's empty-closure dispatch —
/// that a handful of columns cannot repay, and a decode group carries one
/// column per slot. Either way the encoder runs once per column, so the
/// bytes are identical; only WHO calls it changes.
const QUANT_INLINE_COLS: usize = 8;

/// Quantize every slot's activation columns once: one pool dispatch over the
/// concatenated column space (inline for the small counts above),
/// chunk-straddling slots handled segment-wise. Each slot's `(ty, k)` pick
/// its encoder — every pair on the slot is format-equal by the dedupe key,
/// and both encoders are pure functions of (activation format, k).
///
/// Bit identity: each column is a pure function of its own input, and the
/// split partitions the output cells; splitting WITHIN a column would need
/// its own argument (block-local scales).
fn quantize_slots(
    slots: &Flex<QuantSlot<'_>>,
    out: &mut Flex<QuantCols>,
    lvl: u8,
    pacc: &mut profile::CallAcc,
) {
    let n = slots.len();
    for s in 0..n {
        out.push(QuantCols::new(slots.get(s).cb, slots.get(s).x));
    }
    let mut col_starts: Flex<usize> = Flex::new();
    let mut total_cols = 0usize;
    for s in 0..n {
        col_starts.push(total_cols);
        total_cols += slots.get(s).x.ne1;
    }
    let mut shared: Flex<SharedQuantCols> = Flex::new();
    for s in 0..n {
        shared.push(out.get_mut(s).shared());
    }
    // SAFETY (construction site, referenced by every write): the pointers
    // alias buffers owned for the whole call; each participant writes only
    // its own sub-range's columns, the join publishes the writes, and a
    // boundary-straddling column is two segments, never quantized twice.
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
            quant_range(slots, &col_starts, &shared, cols, lvl, &mut acc);
            if let Some(collected) = &collected {
                collected.push(acc);
            }
        });
    } else if total_cols > 0 {
        // The inline shape: the walker runs on the caller, no split —
        // deterministic, `tests/mt.rs` covers this branch too.
        let mut acc = profile::CallAcc::new();
        quant_range(slots, &col_starts, &shared, 0..total_cols, lvl, &mut acc);
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

/// What one row of a pair costs a lane: its weight bytes once per input column
/// — a row is dotted against every column, and a prefill group mixes pairs of
/// one column with pairs of many.
fn row_cost(p: &PairWork<'_>) -> u64 {
    p.row_bytes as u64 * p.ne1.max(1) as u64
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
/// Blocks a lane is cut into.
const STEAL_BLOCKS: usize = 4;

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
/// A profiled dispatch pays the chunk collector (one `Vec` of pool-width
/// slots); an unprofiled one runs the error channel above and allocates
/// nothing unless a chunk fails.
fn run_row_pool(
    pairs: &Flex<PairWork<'_>>,
    lvl: u8,
    pacc: &mut profile::CallAcc,
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
    let mut lane_bounds = [0usize; MAX_LANES + 1];
    {
        let total_cost: u64 = (0..npairs)
            .map(|p| pairs.get(p).n as u64 * row_cost(pairs.get(p)))
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
                let c = row_cost(pairs.get(p));
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
    assert!(nlanes <= MAX_LANES, "row pool lanes: {nlanes} threads");
    // `BLOOMERY_STEAL=0` is the A/B lever: whole-lane blocks, home lanes only.
    let steal = steal_enabled();
    let lanes: [Lane; MAX_LANES] = std::array::from_fn(|t| {
        let (start, end) = if t < nlanes {
            (lane_bounds[t], lane_bounds[t + 1])
        } else {
            (0, 0)
        };
        let block = if steal {
            ((end - start) / STEAL_BLOCKS).max(1)
        } else {
            total_rows.max(1)
        };
        Lane {
            next: std::sync::atomic::AtomicUsize::new(start),
            end,
            block,
        }
    });
    threads::pool().for_each_chunk(total_rows, |rows| {
        let t_busy = if lvl >= 1 { Some(Instant::now()) } else { None };
        let mut chunk = RowChunk {
            start: rows.start,
            busy_ns: 0,
            acc: profile::CallAcc::new(),
            err: None,
        };
        // Home lanes: the lane containing this chunk's start, plus every lane
        // that starts inside the chunk. Byte-cut boundaries need not align
        // with the pool's row-count chunks, and a start-match lookup would
        // leave an interior-starting lane ownerless when stealing is off —
        // the chunk that contains a lane's start owns it, so every lane has
        // exactly one. An empty chunk owns none. The aligned (uniform) case
        // reduces to one home lane per chunk.
        let mut first = 0usize;
        while first + 1 < nlanes && lane_bounds[first + 1] <= rows.start {
            first += 1;
        }
        let mut last = first + 1;
        while last < nlanes && lane_bounds[last] < rows.end {
            last += 1;
        }
        let (origin, span) = if rows.is_empty() {
            (first, 0)
        } else if steal {
            (first, nlanes)
        } else {
            (first, last - first)
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
/// profiler row as the batch shape; only the bookkeeping is narrower.
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
    let fused = qdot::supports(ty) && k.is_multiple_of(qdot::k_granularity(ty));

    let qc = quantize_one(ty, k, x, fused, lvl, &mut pacc);
    let mut out = Tensor2::scratch(n, x.ne1);
    let mut pairs: Flex<PairWork> = Flex::new();
    // SAFETY: the construction-site argument in `matmul_q_multi`'s pair build —
    // one pair, the whole row split, the join publishes the writes.
    pairs.push(PairWork::new(
        &mut out,
        bytes,
        &qc,
        PairMeta {
            ty,
            k,
            n,
            row_bytes,
        },
    ));
    run_row_pool(&pairs, lvl, &mut pacc)?;
    qc.release();

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
    let fused = qdot::supports(ty) && k.is_multiple_of(qdot::k_granularity(ty));
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
        quantize_activations(ty, x_col, &mut qbuf);
        ROW_BUF.with(|cell| -> Result<(), crate::ModelError> {
            let mut scratch = cell.borrow_mut();
            if scratch.len() < k {
                scratch.resize(k, 0.0);
            }
            let row = &mut scratch[..k];
            for r in 0..n {
                let src = &bytes[r * row_bytes..(r + 1) * row_bytes];
                // Weight side: dequant_row, or the byte walk in its place for F32 rows.
                if ty == GgmlType::F32 {
                    for (i, v) in row.iter_mut().enumerate() {
                        *v = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
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
    if ws.len() != xs.len() {
        return Err(crate::ModelError::Shape {
            what: "matmul_q batch",
            want_ne0: ws.len(),
            want_ne1: ws.len(),
            got_ne0: xs.len(),
            got_ne1: xs.len(),
        });
    }
    if ws.is_empty() {
        return Ok(Vec::new());
    }
    // Profiler hook: level 0 one compare, level 1 one `Instant` pair, level 2
    // two more per row, none touching the arithmetic; chunks merge into one
    // `CallAcc` per weight type, so `record` fires once per (call, type).
    let lvl = profile::level();
    let mut pacc = profile::CallAcc::new();
    let t_call = if lvl > 0 { Some(Instant::now()) } else { None };
    let mut bytes: Flex<&[u8]> = Flex::new();
    let mut meta: Flex<PairMeta> = Flex::new();
    let mut slot_of: Flex<usize> = Flex::new();
    let mut slots: Flex<QuantSlot> = Flex::new();
    let mut outs: Vec<Tensor2> = Vec::with_capacity(ws.len());
    for (w, x) in ws.iter().zip(xs) {
        let k = w.dims.first().copied().unwrap_or(0) as usize;
        let n = if w.dims.len() > 1 {
            w.dims[1] as usize
        } else {
            1
        };
        let ty = w.ty;
        if x.ne0 != k {
            return Err(crate::ModelError::Shape {
                what: "matmul_q input",
                want_ne0: k,
                want_ne1: x.ne1,
                got_ne0: x.ne0,
                got_ne1: x.ne1,
            });
        }
        let b = gguf.data(w)?;
        // Fused path decided per pair, exactly the single-pair rule: qdot
        // dots the quantized codes directly, so activations are quantized
        // into `qdot::quantize_col`'s layout, and `k_granularity` owns the
        // per-type block contract (256 for the K-quants, 32 for Q5_0/Q5_1).
        let fused = qdot::supports(ty) && k.is_multiple_of(qdot::k_granularity(ty));
        let cb = if fused {
            Some(qdot::col_bytes(ty, k))
        } else {
            None
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
        let slot = (0..slots.len())
            .find(|&s| {
                let q = slots.get(s);
                std::ptr::eq(q.x, *x)
                    && q.cb == cb
                    && gguf::activation_format(q.ty) == gguf::activation_format(ty)
            })
            .unwrap_or_else(|| {
                slots.push(QuantSlot { x, ty, k, cb });
                slots.len() - 1
            });
        bytes.push(b);
        meta.push(PairMeta {
            ty,
            k,
            n,
            row_bytes: b.len() / n,
        });
        slot_of.push(slot);
        outs.push(Tensor2::scratch(n, x.ne1));
    }
    LAST_QUANT_SLOTS.store(slots.len(), std::sync::atomic::Ordering::Relaxed);

    let mut quantized: Flex<QuantCols> = Flex::new();
    quantize_slots(&slots, &mut quantized, lvl, &mut pacc);

    let mut pairs: Flex<PairWork> = Flex::new();
    for (i, out) in outs.iter_mut().enumerate() {
        // SAFETY (construction site, referenced by every write): each pair's
        // output pointer aliases that pair's `out.data`, caller-owned for the
        // whole call. Each participant writes only cells `t * n + r` with
        // `r` inside its own sub-range of the row split, which partitions
        // the rows; the join publishes the writes; an early error discards
        // the outputs either way.
        pairs.push(PairWork::new(
            out,
            bytes.get(i),
            quantized.get(*slot_of.get(i)),
            *meta.get(i),
        ));
    }
    run_row_pool(&pairs, lvl, &mut pacc)?;

    if let Some(t_call) = t_call {
        // One row per distinct weight type the group carried. Rows, `k` and
        // weight bytes are exact per type; wall and the stage accumulators
        // split by weight-byte share — the honest per-type statement without
        // per-type timers. The shares sum to the whole call, so
        // `instrumented_ns` never counts a group twice, and a single-type
        // group records exactly the homogeneous values.
        let wall_ns = t_call.elapsed().as_nanos() as u64;
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
                share(wall_ns),
                &pacc.scaled(wb, wb_total),
            );
        }
    }
    quantized.release_all();
    Ok(outs)
}
