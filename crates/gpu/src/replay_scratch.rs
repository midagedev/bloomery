//! The scratch of a captured replay, for a gate that holds a replay to
//! owning it: the plain device allocations a captured graph addresses, found
//! from the graph's own nodes, and the poison a gate fills each with right
//! before a replay.
//!
//! A replay writes every cell of its scratch that it reads before it reads
//! it. So a span filled with [`POISON`] right before the launch moves no bit
//! of the replay's outputs unless the replay reads what it does not write:
//! what the eager path or the replay before left there, or the zero a buffer
//! was allocated with. What a replay reads before it writes by design — the
//! weights, the card's fault words, host memory, a by-position or recurrent
//! state, a load-time constant, a launch's block ticket count — is the
//! gate's to account for, by what the code that allocated it says it is.

use crate::graph::{NodeInfo, cu};
use crate::model::{ChainBody, GpuModel};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights};
use crate::{Fault, Gpu, GpuError};
use cuda_core::{CudaContext, CudaStream, DeviceCopy, sys};
use std::ffi::c_void;
use std::sync::Arc;

/// The byte a [`ReplayScratch`] fills a span with before a replay: all-one
/// bytes, a NaN as f32, f16 and bf16 and the largest value as an index or a
/// count, so a launch that reads a cell the replay has not written makes a
/// value no finite result hides, or an index its own check refuses on the
/// fault word.
pub const POISON: u8 = 0xff;

/// The zero bytes on either side of a run of non-zero load-time bytes that a
/// poison writes back with the run: a table's zero entry at its edge.
const TABLE_MARGIN: usize = 16;
/// The zero bytes inside one run of load-time bytes that do not end it: a
/// table's zero entry between non-zero ones.
const TABLE_GAP: usize = 64;

/// A plain device allocation (`cuMemAlloc`) a captured graph's nodes address:
/// the bytes `base .. base + bytes`, and the places of the nodes, in the node
/// list [`ReplayScratch::of`] read, whose parameters name an address in it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplaySpan {
    /// The allocation's first device address.
    pub base: sys::CUdeviceptr,
    /// The allocation's length in bytes.
    pub bytes: usize,
    /// The nodes that address the span, ascending, once each.
    pub nodes: Vec<usize>,
}

/// The addresses in the nodes' parameters that a [`ReplayScratch`] leaves
/// alone, each counted once, by what it is inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReplayLeft {
    /// A buffer of the model's weights.
    pub weights: usize,
    /// The card's fault words.
    pub fault_words: usize,
    /// Memory the virtual memory manager maps (a room: a K/V room, a stack
    /// held as a room), whose pages a memset may not reach.
    pub rooms: usize,
    /// Mapped host memory: staging, flags, a handoff page — the host's to
    /// write, on its own protocol.
    pub host: usize,
    /// Device memory the driver ranges as no plain allocation: an array, a
    /// pool's.
    pub unranged: usize,
}

/// One step of a [`ReplayScratch`] trial: what the model's step call gave
/// back.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayStep {
    /// The step's argmax token; 0 when the step raised the fault word.
    pub id: u32,
    /// The step's logits row; empty when the step raised the fault word.
    pub logits: Vec<f32>,
    /// The fault the step raised, if it did.
    pub fault: Option<Fault>,
    /// Whether a span was poisoned before the step this record describes:
    /// `false` for a reference step, and for one the walk before it cut
    /// short by raising the fault word.
    pub poisoned: bool,
    /// Bytes of the poisoned span the replay left at [`POISON`]: what no
    /// launch of the replay wrote. Read when one span was poisoned, else 0.
    pub still: usize,
}

impl ReplayStep {
    /// The record of a step the walk before it cut short: the walk's step
    /// raised the fault word, and the step itself did not run.
    fn cut_short(fault: Option<Fault>) -> ReplayStep {
        ReplayStep {
            id: 0,
            logits: Vec::new(),
            fault,
            poisoned: false,
            still: 0,
        }
    }
}

/// The plain device allocations a captured graph addresses, for a gate that
/// holds a replay to owning its scratch: a replay writes every cell it reads
/// before it reads it, so a span filled with [`POISON`] right before the
/// launch ([`ReplayScratch::poison`]) moves no bit of the replay's outputs
/// unless the replay reads what it does not write — what the eager path or
/// the replay before left there, or the zero a buffer was allocated with.
///
/// The candidates come from the graph itself: the pointer-sized words of
/// every kernel node's parameters ([`crate::graph::KernelNode::words`]), each
/// asked of the
/// driver what it names, so no model lists its buffers. Left alone, and
/// counted in [`ReplayLeft`]: the weights, the card's fault words, host
/// memory and the rooms, and an address the driver ranges as no plain
/// allocation.
///
/// What is an input inside a candidate, a poison keeps: it fills the span,
/// then writes back the bytes that were non-zero when the span's image was
/// taken ([`ReplayScratch::image`]) — what the load wrote into a buffer, a
/// table or a record's constants. A cell that was zero then is the replay's
/// to write: a zero the replay reads before writing is correct by the
/// allocation's accident, and the poison shows it. What else a replay reads
/// before it writes by design — a state it updates in place, a count one
/// launch resets for the next — is the gate's to account for by the code
/// that allocated it; [`ReplayScratch::still`] and [`ReplayScratch::read`]
/// are what it holds those to.
pub struct ReplayScratch {
    /// The context the spans live in, bound to the calling thread before
    /// each copy and each of the driver's answers.
    ctx: Arc<CudaContext>,
    spans: Vec<ReplaySpan>,
    left: ReplayLeft,
}

/// What a parameter's word names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    /// Nothing the driver holds: a count, a float, a host address.
    Nothing,
    /// An address a poison must not reach.
    Held,
    /// A byte of the plain allocation at `base`, of `bytes`.
    Plain(sys::CUdeviceptr, usize),
}

/// Every device buffer `weights` holds, `(first address, bytes)`.
fn weight_spans(weights: &Weights) -> Vec<(sys::CUdeviceptr, usize)> {
    fn span<T: DeviceCopy>(t: &DeviceTensor<T>) -> (sys::CUdeviceptr, usize) {
        (t.buf().cu_deviceptr(), t.buf().num_bytes())
    }
    let mut out = Vec::new();
    for name in weights.names() {
        let Some(w) = weights.get(name) else {
            continue;
        };
        match w {
            DevWeight::KQuant { w, .. }
            | DevWeight::Q5_0 { w, .. }
            | DevWeight::Q5_1 { w, .. }
            | DevWeight::Bf16 { w, .. } => out.push(span(w)),
            DevWeight::F32 { w, .. } => out.push(span(w)),
            DevWeight::Q8_0 { qs, d, .. } | DevWeight::Q8_0Derived { qs, d, .. } => {
                out.push(span(qs));
                out.push(span(d));
            }
            DevWeight::Nvfp4 { qs, e, scales, .. }
            | DevWeight::Nvfp4Stack { qs, e, scales, .. } => {
                out.push(span(qs));
                out.push(span(e));
                out.push(span(scales));
            }
            DevWeight::Mxfp8 { qs, e, .. } | DevWeight::Fp8B128 { qs, e, .. } => {
                out.push(span(qs));
                out.push(span(e));
            }
        }
    }
    out
}

/// Where the word `word` names, counting in `left` the address left alone.
/// `fault` and `weights` are the buffers no poison reaches, asked first: a
/// weight viewed over a room's range is a weight, not a room.
fn place_of(
    word: sys::CUdeviceptr,
    fault: (sys::CUdeviceptr, usize),
    weights: &[(sys::CUdeviceptr, usize)],
    left: &mut ReplayLeft,
) -> Result<Place, GpuError> {
    let inside =
        |(base, bytes): (sys::CUdeviceptr, usize)| word >= base && word < base + bytes as u64;
    if inside(fault) {
        left.fault_words += 1;
        return Ok(Place::Held);
    }
    if weights.iter().any(|w| inside(*w)) {
        left.weights += 1;
        return Ok(Place::Held);
    }
    let refused = sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE;
    let mut kind: sys::CUmemorytype = 0;
    // SAFETY: `kind` is a local the call writes; `word` is any value a
    // parameter holds, which the driver answers or refuses.
    let rc = unsafe {
        sys::cuPointerGetAttribute(
            (&mut kind as *mut sys::CUmemorytype).cast(),
            sys::CUpointer_attribute_enum_CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            word,
        )
    };
    if rc == refused {
        return Ok(Place::Nothing);
    }
    cu(rc, "cuPointerGetAttribute")?;
    if kind == sys::CUmemorytype_enum_CU_MEMORYTYPE_HOST {
        left.host += 1;
        return Ok(Place::Held);
    }
    if kind != sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE {
        left.unranged += 1;
        return Ok(Place::Held);
    }
    // A room is memory cuMemMap mapped, found by the handle the address
    // retains; any other answer reads as no room, and the range query below
    // says what the address is, naming the driver when it cannot.
    let mut handle: sys::CUmemGenericAllocationHandle = 0;
    // SAFETY: `handle` is a local the call writes when it answers; `word`
    // names device memory.
    if unsafe { sys::cuMemRetainAllocationHandle(&mut handle, word as *mut c_void) }
        == sys::cudaError_enum_CUDA_SUCCESS
    {
        // SAFETY: `handle` is the reference the call above took; it is
        // released once, here.
        cu(unsafe { sys::cuMemRelease(handle) }, "cuMemRelease")?;
        left.rooms += 1;
        return Ok(Place::Held);
    }
    let (mut base, mut bytes) = (0, 0usize);
    // SAFETY: both locals are what the call writes; `word` names device
    // memory.
    let rc = unsafe { sys::cuMemGetAddressRange_v2(&mut base, &mut bytes, word) };
    if rc == refused || rc == sys::cudaError_enum_CUDA_ERROR_NOT_FOUND {
        left.unranged += 1;
        return Ok(Place::Held);
    }
    cu(rc, "cuMemGetAddressRange_v2")?;
    Ok(Place::Plain(base, bytes))
}

/// The runs of `was`, a span's image, that a poison writes back over the
/// memset: every stretch of bytes holding a non-zero one, zero gaps of at
/// most [`TABLE_GAP`] bytes folded in and [`TABLE_MARGIN`] zero bytes on
/// either side, each as `(offset, bytes)`. Nothing for an all-zero image.
fn table_runs(was: &[u8]) -> Vec<(usize, &[u8])> {
    let run = |first: usize, end: usize| {
        let from = first.saturating_sub(TABLE_MARGIN);
        let to = (end + TABLE_MARGIN).min(was.len());
        (from, &was[from..to])
    };
    let mut runs = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    for (i, &b) in was.iter().enumerate() {
        if b == 0 {
            continue;
        }
        open = match open {
            Some((first, end)) if i - end <= TABLE_GAP => Some((first, i + 1)),
            Some((first, end)) => {
                runs.push(run(first, end));
                Some((i, i + 1))
            }
            None => Some((i, i + 1)),
        };
    }
    if let Some((first, end)) = open {
        runs.push(run(first, end));
    }
    runs
}

/// One step call of `m`: its token, `None` when the step raised the fault
/// word (the word is that read's defined answer, and the next reset lifts
/// the poison), and any other error.
fn stepped<B: ChainBody>(m: &mut GpuModel<B>, token: u32) -> Result<Option<u32>, GpuError> {
    match m.step(&[token]) {
        Ok(id) => Ok(Some(id)),
        Err(GpuError::Fault { .. }) => Ok(None),
        Err(e) => Err(e),
    }
}

impl ReplayScratch {
    /// The plain device allocations the kernel nodes among `nodes` address
    /// on `gpu`, in address order and never overlapping; inside `weights` and
    /// the card's fault words, host memory and the rooms are left alone and
    /// counted. `gpu`'s context is bound to the calling thread for the
    /// driver's answers. Refused by name when no node addresses a plain
    /// allocation: then there is nothing a replay could be held to.
    pub fn of(gpu: &Gpu, weights: &Weights, nodes: &[NodeInfo]) -> Result<ReplayScratch, GpuError> {
        const WHAT: &str = "ReplayScratch::of";
        let ctx = Arc::clone(gpu.context());
        ctx.bind_to_thread()?;
        let fault = gpu.fault_word();
        let fault = (fault.cu_deviceptr(), fault.num_bytes());
        let weights = weight_spans(weights);
        let mut left = ReplayLeft::default();
        let mut seen: Vec<(sys::CUdeviceptr, Place)> = Vec::new();
        let mut spans: Vec<ReplaySpan> = Vec::new();
        for (i, n) in nodes.iter().enumerate() {
            for &word in n.kernel.iter().flat_map(|k| k.words.iter()) {
                let place = match seen.iter().find(|(w, _)| *w == word) {
                    Some((_, place)) => *place,
                    None => {
                        let place = place_of(word, fault, &weights, &mut left)?;
                        seen.push((word, place));
                        place
                    }
                };
                if let Place::Plain(base, bytes) = place {
                    match spans.iter_mut().find(|s| s.base == base) {
                        Some(span) => span.nodes.push(i),
                        None => spans.push(ReplaySpan {
                            base,
                            bytes,
                            nodes: vec![i],
                        }),
                    }
                }
            }
        }
        for span in &mut spans {
            span.nodes.dedup();
        }
        spans.sort_unstable_by_key(|s| s.base);
        if let Some(pair) = spans
            .windows(2)
            .find(|p| p[0].base + p[0].bytes as u64 > p[1].base)
        {
            let (a, b) = (&pair[0], &pair[1]);
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "two allocations that overlap: {:#x} + {} and {:#x}",
                    a.base, a.bytes, b.base
                ),
            ));
        }
        if spans.is_empty() {
            return Err(GpuError::state(
                WHAT,
                "no plain device allocation among the addresses the nodes name",
            ));
        }
        Ok(ReplayScratch { ctx, spans, left })
    }

    /// The candidate spans, in address order.
    #[must_use]
    pub fn spans(&self) -> &[ReplaySpan] {
        &self.spans
    }

    /// The addresses left alone, by what each is inside.
    #[must_use]
    pub fn left(&self) -> ReplayLeft {
        self.left
    }

    /// Span `at`, or the refusal of a place past the spans.
    fn span(&self, at: usize) -> Result<&ReplaySpan, GpuError> {
        const WHAT: &str = "ReplayScratch::span";
        let held = self.spans.len();
        self.spans
            .get(at)
            .ok_or_else(|| GpuError::shape(WHAT, format!("span {at} of {held}")))
    }

    /// The spans' bytes as they stand, in [`ReplayScratch::spans`]'s order;
    /// `stream` is synchronized. Taken at the load, it is the image a poison
    /// keeps the non-zero bytes of.
    pub fn image(&self, stream: &CudaStream) -> Result<Vec<Vec<u8>>, GpuError> {
        self.ctx.bind_to_thread()?;
        let mut image: Vec<Vec<u8>> = self.spans.iter().map(|s| vec![0; s.bytes]).collect();
        for (at, dst) in image.iter_mut().enumerate() {
            self.read_span(stream, at, dst)?;
        }
        stream.synchronize()?;
        Ok(image)
    }

    /// Fill the spans `at` (their places in [`ReplayScratch::spans`]) with
    /// [`POISON`] on `stream`, then write back the runs of bytes that were
    /// non-zero in `image` ([`table_runs`]): the load's input stays, the
    /// cells that were zero are the replay's to write. Asynchronous, behind
    /// what the stream carries.
    pub fn poison(
        &self,
        stream: &CudaStream,
        at: &[usize],
        image: &[Vec<u8>],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "ReplayScratch::poison";
        self.ctx.bind_to_thread()?;
        for &i in at {
            let span = self.span(i)?;
            let was = self.image_of(i, image, WHAT)?;
            // SAFETY: `span.base` is the first byte of a live plain
            // allocation of `span.bytes` on `stream`'s context.
            let rc =
                unsafe { sys::cuMemsetD8Async(span.base, POISON, span.bytes, stream.cu_stream()) };
            cu(rc, "cuMemsetD8Async")?;
            for (off, run) in table_runs(was) {
                self.write_span(stream, i, off, run)?;
            }
        }
        Ok(())
    }

    /// Write `image` back into the spans `at`, whole: what a trial leaves
    /// behind when it ends, so the next starts from the load's image.
    /// Asynchronous, behind what the stream carries.
    pub fn restore(
        &self,
        stream: &CudaStream,
        at: &[usize],
        image: &[Vec<u8>],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "ReplayScratch::restore";
        self.ctx.bind_to_thread()?;
        for &i in at {
            let was = self.image_of(i, image, WHAT)?;
            self.write_span(stream, i, 0, was)?;
        }
        Ok(())
    }

    /// `image`'s entry for span `i`, held to the span's length.
    fn image_of<'a>(
        &self,
        i: usize,
        image: &'a [Vec<u8>],
        what: &'static str,
    ) -> Result<&'a [u8], GpuError> {
        let span = self.span(i)?;
        let held = image.len();
        let was = image
            .get(i)
            .ok_or_else(|| GpuError::shape(what, format!("span {i} of {held} images")))?;
        if was.len() != span.bytes {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} bytes of image for a span of {} bytes",
                    was.len(),
                    span.bytes
                ),
            ));
        }
        Ok(was)
    }

    /// The bytes of span `at` that hold [`POISON`] now; `stream` is
    /// synchronized.
    pub fn still(&self, stream: &CudaStream, at: usize) -> Result<usize, GpuError> {
        Ok(self
            .read(stream, at)?
            .iter()
            .filter(|&&b| b == POISON)
            .count())
    }

    /// Span `at`'s bytes as they stand; `stream` is synchronized. A gate
    /// reads a span with it between steps, to hold its owner to what it
    /// leaves there.
    pub fn read(&self, stream: &CudaStream, at: usize) -> Result<Vec<u8>, GpuError> {
        self.ctx.bind_to_thread()?;
        let mut now = vec![0; self.span(at)?.bytes];
        self.read_span(stream, at, &mut now)?;
        stream.synchronize()?;
        Ok(now)
    }

    /// Copy span `at`'s bytes into `dst`, which holds exactly their number.
    fn read_span(&self, stream: &CudaStream, at: usize, dst: &mut [u8]) -> Result<(), GpuError> {
        const WHAT: &str = "ReplayScratch::read_span";
        let span = self.span(at)?;
        if dst.len() != span.bytes {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} bytes of room for a span of {} bytes",
                    dst.len(),
                    span.bytes
                ),
            ));
        }
        // SAFETY: `dst` holds `span.bytes` writable bytes for the call, and
        // `span.base` is the first byte of a live plain allocation of that
        // length on `stream`'s context, which outlives the copy.
        let rc = unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr().cast(),
                span.base,
                span.bytes,
                stream.cu_stream(),
            )
        };
        cu(rc, "cuMemcpyDtoHAsync_v2")
    }

    /// Copy `src` into span `at` from its byte `off`.
    fn write_span(
        &self,
        stream: &CudaStream,
        at: usize,
        off: usize,
        src: &[u8],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "ReplayScratch::write_span";
        let span = self.span(at)?;
        if off
            .checked_add(src.len())
            .is_none_or(|end| end > span.bytes)
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} bytes at {} of a span of {} bytes",
                    src.len(),
                    off,
                    span.bytes
                ),
            ));
        }
        // SAFETY: `src` is `src.len()` host bytes, and `span.base + off` is
        // inside the live plain allocation for that length (checked above),
        // on `stream`'s context.
        let rc = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                span.base + off as u64,
                src.as_ptr().cast(),
                src.len(),
                stream.cu_stream(),
            )
        };
        cu(rc, "cuMemcpyHtoDAsync_v2")
    }
    /// The kernel entries of the nodes that address span `at`, once each and
    /// in the node list's order.
    #[must_use]
    pub fn entries<'a>(&self, nodes: &'a [NodeInfo], at: usize) -> Vec<&'a str> {
        let mut names: Vec<&str> = self
            .spans
            .get(at)
            .into_iter()
            .flat_map(|span| span.nodes.iter())
            .filter_map(|&i| nodes.get(i))
            .filter_map(|n| n.kernel.as_ref())
            .map(|k| k.name.as_str())
            .collect();
        names.dedup();
        names
    }

    /// `token`'s step after `prefix` fed one id a step from a reset, through
    /// the model's own calls: the reference a poisoned step is held to.
    pub fn step<B: ChainBody>(
        &self,
        m: &mut GpuModel<B>,
        prefix: &[u32],
        token: u32,
    ) -> Result<ReplayStep, GpuError> {
        self.run(m, prefix, token, None)
    }

    /// [`ReplayScratch::step`] with the spans `at` poisoned right before the
    /// step's replay: behind `prefix`, in front of the refresh and the launch
    /// the step call makes. However the step ends — in a token, in the fault
    /// word or in an error — the spans hold `image` again when this returns.
    pub fn poisoned_step<B: ChainBody>(
        &self,
        m: &mut GpuModel<B>,
        prefix: &[u32],
        token: u32,
        at: &[usize],
        image: &[Vec<u8>],
    ) -> Result<ReplayStep, GpuError> {
        self.run(m, prefix, token, Some((at, image)))
    }

    /// The body of [`ReplayScratch::step`] and [`ReplayScratch::poisoned_step`].
    fn run<B: ChainBody>(
        &self,
        m: &mut GpuModel<B>,
        prefix: &[u32],
        token: u32,
        poison: Option<(&[usize], &[Vec<u8>])>,
    ) -> Result<ReplayStep, GpuError> {
        let stream = m.gpu().stream_handle();
        m.reset()?;
        for &t in prefix {
            if stepped(m, t)?.is_none() {
                return Ok(ReplayStep::cut_short(m.poisoned()));
            }
        }
        if let Some((at, image)) = poison {
            self.poison(&stream, at, image)?;
        }
        // Read back in this order: the token's logits, then what the replay
        // left at the poison, then the image over the span.
        let got = stepped(m, token).and_then(|id| match id {
            Some(id) => m.logits().map(|logits| Some((id, logits))),
            None => Ok(None),
        });
        let left = match poison {
            Some(([at], _)) => Some(self.still(&stream, *at)),
            _ => None,
        };
        if let Some((at, image)) = poison {
            self.restore(&stream, at, image)?;
        }
        let still = match left {
            Some(still) => still?,
            None => 0,
        };
        match got? {
            Some((id, logits)) => Ok(ReplayStep {
                id,
                logits,
                fault: None,
                poisoned: poison.is_some(),
                still,
            }),
            None => Ok(ReplayStep {
                id: 0,
                logits: Vec::new(),
                fault: m.poisoned(),
                poisoned: poison.is_some(),
                still,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{POISON, ReplayScratch};
    use crate::Gpu;
    use crate::weights::Weights;
    use cuda_core::DeviceBuffer;

    /// Values in each of the fixture's buffers.
    const N: usize = 4096;

    /// Two exact elementwise launches over plain allocations:
    /// `t = a·ga + b·gb` ([`crate::elem::ElemKernels::enqueue_mix_rows`]), then
    /// `y = t + b` ([`crate::elem::ElemKernels::enqueue_add`]). The mix writes
    /// `t` before the add reads it, and the add writes `y`: both the replay's
    /// own. The mix reads `a`, `ga`, `b` and `gb` and writes none of them:
    /// inputs. Neither launch carries a fault sink.
    struct Step {
        a: DeviceBuffer<f32>,
        ga: DeviceBuffer<f32>,
        b: DeviceBuffer<f32>,
        gb: DeviceBuffer<f32>,
        t: DeviceBuffer<f32>,
        y: DeviceBuffer<f32>,
    }

    impl Step {
        fn new(gpu: &Gpu) -> Step {
            let s = gpu.stream();
            let row = |f: fn(usize) -> f32| {
                let v: Vec<f32> = (0..N).map(f).collect();
                DeviceBuffer::from_host(s, &v).expect("a buffer")
            };
            Step {
                a: row(|i| 0.25 + i as f32 / 8.0),
                ga: row(|_| 1.0),
                // Zero at one index: a zero f32 between non-zero ones.
                b: row(|i| 1.5 - i as f32 / 16.0),
                gb: row(|_| 2.0),
                t: DeviceBuffer::zeroed(s, N).expect("the mix's output"),
                y: DeviceBuffer::zeroed(s, N).expect("the output"),
            }
        }

        fn enqueue(&mut self, gpu: &Gpu) -> Result<(), crate::GpuError> {
            let s = gpu.stream();
            gpu.elem()
                .enqueue_mix_rows(s, &self.a, &self.ga, &self.b, &self.gb, N, &mut self.t)?;
            gpu.elem().enqueue_add(s, &self.t, &self.b, N, &mut self.y)
        }
    }

    /// A replay owns its scratch and reads its inputs. The two launches' nodes
    /// name every plain allocation they run over — the four inputs, the mix's
    /// output, the sum's — and nothing is left alone. With a span poisoned
    /// right before the replay and nothing of the image kept, the replay over
    /// the two the launches write is the eager bits (it writes them before it
    /// reads), and over an input it is not (it reads it and never writes it).
    /// An input poisoned with its own image kept is the eager bits again: the
    /// load's bytes stay, a zero entry between non-zero ones among them. Each
    /// trial ends with the span at the image again, so the last replay is the
    /// eager bits.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_poisoned_replay_owns_its_scratch_and_reads_its_inputs() {
        let gpu = Gpu::new().expect("a Gpu on device 0");
        let mut step = Step::new(&gpu);
        step.enqueue(&gpu).expect("the eager sequence");
        let s = gpu.stream();
        s.synchronize().expect("the eager run");
        let eager = step.y.to_host_vec(s).expect("readback");
        assert!(
            eager.iter().all(|v| v.is_finite()) && eager.iter().any(|&v| v != 0.0),
            "the eager output is not finite and nonzero, so a replay that runs nothing \
             could match it"
        );
        let graph = gpu.capture(|_| step.enqueue(&gpu)).expect("the capture");
        let nodes = graph.nodes().expect("the node list");
        assert_eq!(nodes.len(), 2, "{nodes:?}");
        assert!(
            nodes
                .iter()
                .all(|n| n.kernel.as_ref().is_some_and(|k| !k.words.is_empty())),
            "every kernel's parameters name an address: {nodes:?}"
        );
        let scratch = ReplayScratch::of(&gpu, &Weights::default(), &nodes).expect("the scratch");
        let spans = scratch.spans();
        let left = scratch.left();
        assert_eq!(
            spans.len(),
            6,
            "four inputs, the mix's output, the sum: {spans:?}"
        );
        assert_eq!(
            (
                left.weights,
                left.fault_words,
                left.rooms,
                left.host,
                left.unranged
            ),
            (0, 0, 0, 0, 0),
            "{left:?}"
        );
        let at_of = |want: u64| {
            spans
                .iter()
                .position(|span| span.base == want)
                .unwrap_or_else(|| panic!("no span starts at {want:#x}: {spans:?}"))
        };
        let inputs = [
            at_of(step.a.cu_deviceptr()),
            at_of(step.ga.cu_deviceptr()),
            at_of(step.b.cu_deviceptr()),
            at_of(step.gb.cu_deviceptr()),
        ];
        let owned: Vec<usize> = (0..spans.len()).filter(|at| !inputs.contains(at)).collect();
        assert_eq!(owned.len(), 2, "the mix's output and the sum: {spans:?}");
        let image = scratch.image(s).expect("the image");
        let none: Vec<Vec<u8>> = spans.iter().map(|span| vec![0; span.bytes]).collect();
        // One replay, then how many outputs are not the eager bits.
        let replay = || -> usize {
            graph.launch(s).expect("a replay");
            s.synchronize().expect("the replay");
            step.y
                .to_host_vec(s)
                .expect("readback")
                .iter()
                .zip(&eager)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count()
        };
        for &at in &owned {
            scratch.poison(s, &[at], &none).expect("the poison");
            let off = replay();
            scratch.restore(s, &[at], &image).expect("the restore");
            assert_eq!(off, 0, "span {at} poisoned: the replay writes it first");
        }
        for &at in &inputs {
            scratch.poison(s, &[at], &none).expect("the poison");
            let off = replay();
            scratch.restore(s, &[at], &image).expect("the restore");
            assert_ne!(
                off, 0,
                "span {at} poisoned: the replay reads it, never writes it"
            );
        }
        // `b` holds a zero entry: its image keeps the whole column.
        let column = at_of(step.b.cu_deviceptr());
        scratch.poison(s, &[column], &image).expect("the poison");
        let kept = replay();
        scratch.restore(s, &[column], &image).expect("the restore");
        assert_eq!(
            kept, 0,
            "the column at its own image: the load's bytes stay"
        );
        assert_eq!(replay(), 0, "every span at the image again");
        assert_eq!(POISON, 0xff);
    }
}
