//! The hybrid MoE boundary: the experts a layer's [`SlotMap`] puts on the
//! card run there, the rest on the host, and the handoff between the two sits
//! inside the captured step. The prefix `[0, n_l)` of every MoE layer is the
//! V2-Lite (deepseek2) convention ([`HybridConfig`], [`SlotMap::prefix`]); a
//! V4.1 card holds its plan's `ExpertList` per layer — the id prefix or a hot
//! list's ranked ids.
//!
//! Per hybrid layer the chain carries, after the router:
//!
//! ```text
//! router → D2H(handoff) → go → [card experts, shared expert] → wait → add(hsum) → combine
//! ```
//!
//! The handoff is one device region the router's launches write — a
//! sequence word, the ids, the weights and the f32 normed activation — and
//! one copy node lands it in the host-mapped page. `go` is one batch of
//! stream memory operations: a system-scope barrier (the copy lands first),
//! the layer's id written into the page, a second barrier, then an atomic add
//! of one to the host-mapped generation word and to the device sequence word
//! the next handoff carries. `wait` waits until the host-mapped counter is at
//! least one and adds minus one. With the overlap lever off
//! (`BLOOMERY_HYBRID_OVERLAP=0`) the wait sits right after the go instead.
//!
//! No word is written by the card and the host at the same time: the
//! generation, sequence and layer words only by the card; the counter by the
//! host between a go and its wait, and by the card after the wait passes —
//! and the host touches it again only after the next go, which the stream
//! orders behind that add and a system barrier.
//!
//! A boundary of two rows carries a pass whose two tokens run one layer
//! apart, so a row's go can land while the other row's wait is pending: each
//! row has its own layer word, counter, image and sum, and the rule above
//! holds per row, since a row's next go sits behind its own wait. The
//! generation and sequence count every go in stream order, and the host
//! serves in that order.
//!
//! The host side is the decode thread. After a graph launch it serves the
//! captured chain's hybrid layers in order (`Hybrid::serve_captured`); an
//! eager chain is served layer by layer as it is enqueued, so the stream never
//! holds more than one layer of work behind a wait. A service waits for its go
//! with the whole pool spinning on the generation word (a pool job: a worker
//! inside a job is not parked, and the expert dispatch that follows starts
//! inside the spin window the job leaves it in), checks the handoff's sequence
//! number and layer, has the architecture's host computation ([`HostExperts`])
//! write the host experts' sum into the host-mapped page and adds one to the
//! counter. The combine reads that sum in place through the mapping. The
//! protocol — the words, the sequence, the service loop — is this file's
//! alone; an architecture supplies only what one service computes.
//!
//! An eager batch of prompt tokens is served outside that protocol
//! ([`Hybrid::serve_batch`]): the chain brings the batch's handoffs to the host
//! itself, and one service computes a layer's host experts for every token of
//! the batch in one union call ([`HostExperts::experts_union_into`]). It moves
//! no flag word and no sequence number, so the next step's handoff finds the
//! host where the last step left it.

use crate::GpuError;
use crate::graph::cu;
use crate::tensor::window;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, sys};
use gguf::Split;
use model::Tensor2;
use model::moe::EXPERTS_INTO_MAX;
use model::placement::host_lock::{HostLock, HostSet, Walk};
use model::placement::{ModelTensor, Plan};
use std::cell::RefCell;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

// ----------------------------------------------------------------- levers

/// The hybrid path's two runtime levers, as read from the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Levers {
    /// `BLOOMERY_HYBRID_NL`: experts per MoE layer kept on the card; `None`
    /// when unset.
    pub n_l: Option<usize>,
    /// `BLOOMERY_HYBRID_OVERLAP`: `false` (`0`) puts each hybrid layer's wait
    /// right after its go; unset or `1` puts it just before the combine.
    pub overlap: bool,
}

/// The levers, read once per process (a `OnceLock`: the environment is read at
/// first use and never again).
pub fn levers() -> Result<Levers, GpuError> {
    static LEVERS: OnceLock<Result<Levers, String>> = OnceLock::new();
    LEVERS
        .get_or_init(read_levers)
        .clone()
        .map_err(|detail| GpuError::shape("hybrid::levers", detail))
}

fn read_levers() -> Result<Levers, String> {
    let n_l = match std::env::var("BLOOMERY_HYBRID_NL") {
        Ok(v) => Some(
            v.trim()
                .parse::<usize>()
                .map_err(|e| format!("BLOOMERY_HYBRID_NL={v:?}: {e}"))?,
        ),
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => return Err(format!("BLOOMERY_HYBRID_NL: {e}")),
    };
    let overlap = match std::env::var("BLOOMERY_HYBRID_OVERLAP") {
        Err(std::env::VarError::NotPresent) => true,
        Ok(v) if v.trim() == "1" => true,
        Ok(v) if v.trim() == "0" => false,
        Ok(v) => return Err(format!("BLOOMERY_HYBRID_OVERLAP={v:?}: want 0 or 1")),
        Err(e) => return Err(format!("BLOOMERY_HYBRID_OVERLAP: {e}")),
    };
    Ok(Levers { n_l, overlap })
}

/// A V2-Lite (deepseek2) hybrid load's parameters: experts `[0, n_l)` of
/// every MoE layer on the card and the rest on the host, and where each
/// layer's wait sits. A V4.1 load takes its card set from the plan instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridConfig {
    pub n_l: usize,
    pub overlap: bool,
}

impl HybridConfig {
    /// What the levers ask of `file`'s model: `None` when `BLOOMERY_HYBRID_NL`
    /// is unset or keeps every expert on the card — today's all-card path — and
    /// an error when it asks for more experts than a layer has.
    pub fn from_levers(file: &Split) -> Result<Option<HybridConfig>, GpuError> {
        let l = levers()?;
        let Some(n_l) = l.n_l else {
            return Ok(None);
        };
        let what = "HybridConfig::from_levers";
        let n_expert = file
            .arch_get_u64("expert_count")
            .ok_or(GpuError::metadata(what, "expert_count"))?;
        let n_expert = usize::try_from(n_expert)
            .map_err(|_| GpuError::shape(what, format!("expert_count {n_expert}")))?;
        if n_l > n_expert {
            return Err(GpuError::shape(
                what,
                format!("BLOOMERY_HYBRID_NL={n_l} keeps more experts than a layer's {n_expert}"),
            ));
        }
        Ok((n_l < n_expert).then_some(HybridConfig {
            n_l,
            overlap: l.overlap,
        }))
    }
}

// ------------------------------------------------------- the host set at load

/// The host tier's load levers, as read from the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostLevers {
    /// `BLOOMERY_HOST_POPULATE`: unset or `1` reads the plan's host set into
    /// the page cache and maps it at load; `0` leaves it to the steps' first
    /// touches.
    pub populate: bool,
    /// `BLOOMERY_HOST_LOCK`: `1` locks the host set in RAM for the model's
    /// lifetime ([`HostLock`]); unset or `0` does not.
    pub lock: bool,
    /// `BLOOMERY_CARD_DONTNEED`: unset or `1` releases each card segment's
    /// file pages from the page cache once uploaded
    /// ([`crate::weights::Weights::load_placed`]); `0` keeps them cached.
    pub card_dontneed: bool,
}

/// The host levers, read once per process.
pub fn host_levers() -> Result<HostLevers, GpuError> {
    static LEVERS: OnceLock<Result<HostLevers, String>> = OnceLock::new();
    LEVERS
        .get_or_init(|| {
            Ok(HostLevers {
                populate: flag("BLOOMERY_HOST_POPULATE", true)?,
                lock: flag("BLOOMERY_HOST_LOCK", false)?,
                card_dontneed: flag("BLOOMERY_CARD_DONTNEED", true)?,
            })
        })
        .clone()
        .map_err(|detail| GpuError::shape("hybrid::host_levers", detail))
}

/// A `0`/`1` lever; `default` when unset.
fn flag(name: &str, default: bool) -> Result<bool, String> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(default),
        Ok(v) if v.trim() == "1" => Ok(true),
        Ok(v) if v.trim() == "0" => Ok(false),
        Ok(v) => Err(format!("{name}={v:?}: want 0 or 1")),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

/// What a placed load did to its plan's host set: populated it, locked it,
/// both or neither ([`HostLevers`]). Holds the lock, when there is one, for
/// as long as it lives; its owner keeps the split's mappings alive longer.
pub struct HostResidency {
    set: HostSet,
    populate: Option<Walk>,
    lock: Option<HostLock>,
}

impl HostResidency {
    /// The host set of `plan` over `split` — the host segments whose tensor
    /// `keep` selects — populated, then locked, as `levers` ask. Populating
    /// first makes the lock a walk over resident pages.
    pub fn at_load(
        split: &Split,
        plan: &Plan<'_>,
        keep: impl Fn(&ModelTensor) -> bool,
        levers: HostLevers,
    ) -> Result<HostResidency, GpuError> {
        const WHAT: &str = "HostResidency::at_load";
        let set = HostSet::of(split, plan, keep).map_err(|e| GpuError::plan(WHAT, e))?;
        let populate = levers
            .populate
            .then(|| set.populate(split))
            .transpose()
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let lock = levers
            .lock
            .then(|| HostLock::lock(split, &set))
            .transpose()
            .map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(HostResidency {
            set,
            populate,
            lock,
        })
    }

    /// The host set the load walked.
    #[must_use]
    pub fn set(&self) -> &HostSet {
        &self.set
    }

    /// The populate walk; `None` with `BLOOMERY_HOST_POPULATE=0`.
    #[must_use]
    pub fn populated(&self) -> Option<&Walk> {
        self.populate.as_ref()
    }

    /// The lock; `None` unless `BLOOMERY_HOST_LOCK=1`.
    #[must_use]
    pub fn lock(&self) -> Option<&HostLock> {
        self.lock.as_ref()
    }
}

// --------------------------------------------------------------- slot map

/// The slot map's entry for an expert the card does not hold: the host
/// computes it.
pub const HOST: u32 = u32::MAX;

/// Which experts of each MoE layer the card holds, host side: per layer of
/// `layers`, a row of `n_expert` entries, each the slot of the layer's routed
/// stack that holds the expert or [`HOST`]. The one owner of which experts
/// run on the host: the tier serves an id exactly when the map sends it
/// there, and a chain that reads the map on the card uploads this one
/// ([`SlotMap::as_slice`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotMap {
    layers: Range<usize>,
    n_expert: usize,
    slots: Vec<u32>,
    /// Per row, the experts on the card.
    on_card: Vec<usize>,
}

impl SlotMap {
    /// Experts `[0, n_l)` of every layer of `layers` in slots `0..n_l`, the
    /// rest on the host.
    pub fn prefix(layers: Range<usize>, n_expert: usize, n_l: usize) -> Result<SlotMap, GpuError> {
        let what = "SlotMap::prefix";
        let n = u32::try_from(n_expert)
            .map_err(|_| GpuError::shape(what, format!("{n_expert} experts pass u32")))?;
        let cut = u32::try_from(n_l)
            .ok()
            .filter(|&c| c <= n)
            .ok_or_else(|| GpuError::shape(what, format!("{n_l} experts on the card of {n}")))?;
        let row: Vec<u32> = (0..n).map(|e| if e < cut { e } else { HOST }).collect();
        let slots = row.repeat(layers.len());
        SlotMap::from_rows(layers, n_expert, slots)
    }

    /// The map from its rows: `layers.len()` rows of `n_expert` entries. The
    /// entries of a row that are not [`HOST`] are the slots `0..k` of the
    /// row's `k` experts on the card, each once.
    pub fn from_rows(
        layers: Range<usize>,
        n_expert: usize,
        slots: Vec<u32>,
    ) -> Result<SlotMap, GpuError> {
        let what = "SlotMap::from_rows";
        if n_expert == 0 || layers.len().checked_mul(n_expert) != Some(slots.len()) {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} entries for layers {layers:?} of {n_expert} experts",
                    slots.len()
                ),
            ));
        }
        let mut on_card = Vec::with_capacity(layers.len());
        let mut taken = vec![false; n_expert];
        for (row, l) in slots.chunks_exact(n_expert).zip(layers.clone()) {
            let k = row.iter().filter(|&&s| s != HOST).count();
            taken.fill(false);
            for &s in row.iter().filter(|&&s| s != HOST) {
                let slot = usize::try_from(s)
                    .ok()
                    .filter(|&s| s < k)
                    .and_then(|s| taken.get_mut(s))
                    .ok_or_else(|| {
                        GpuError::shape(
                            what,
                            format!(
                                "layer {l}: slot {s}, and the row puts {k} experts on the card"
                            ),
                        )
                    })?;
                if std::mem::replace(slot, true) {
                    return Err(GpuError::shape(
                        what,
                        format!("layer {l}: slot {s} holds two experts"),
                    ));
                }
            }
            on_card.push(k);
        }
        Ok(SlotMap {
            layers,
            n_expert,
            slots,
            on_card,
        })
    }

    /// The layers the map has rows for.
    #[must_use]
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    /// Entries per row: the file's experts per layer.
    #[must_use]
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }

    /// Every row in layer order, as a card-side copy holds it.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.slots
    }

    /// Layer `layer`'s row; `None` for a layer the map has no row for.
    #[must_use]
    pub fn row(&self, layer: usize) -> Option<&[u32]> {
        let i = layer.checked_sub(self.layers.start)?;
        self.slots.chunks_exact(self.n_expert).nth(i)
    }

    /// The experts of layer `layer` the card holds; 0 for a layer the map has
    /// no row for.
    #[must_use]
    pub fn on_card(&self, layer: usize) -> usize {
        layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.on_card.get(i))
            .copied()
            .unwrap_or(0)
    }
}

// ------------------------------------------------------ memory and windows

/// A flag word of the host-mapped page. Each has a 64-byte line of its own,
/// so a thread spinning on one never shares a line with another.
#[derive(Clone, Copy)]
enum Word {
    /// The card adds one per go; the host reads it.
    Gen,
    /// The host adds one per layer of row `.0` served; the card waits for it
    /// and takes it back.
    Cnt(usize),
    /// The card writes the layer of each go of row `.0`; the host reads it.
    Lyr(usize),
}

impl Word {
    const fn offset(self) -> usize {
        match self {
            Word::Gen => 0,
            Word::Cnt(row) => 64 + 128 * row,
            Word::Lyr(row) => 128 + 128 * row,
        }
    }
}

/// Rows a boundary carries at most: tokens whose handoffs can be in flight
/// at once, each with its own layer word, image and sum. Every flag word
/// sits in the page's first [`PAYLOAD_OFF`] bytes.
pub const MAX_ROWS: usize = 2;

const _: () = assert!(
    Word::Cnt(MAX_ROWS - 1).offset() + 64 <= PAYLOAD_OFF
        && Word::Lyr(MAX_ROWS - 1).offset() + 64 <= PAYLOAD_OFF
);

/// Byte offset of row 0's handoff image in the page; row `r`'s sits
/// [`Boundary`]'s image stride after it.
const PAYLOAD_OFF: usize = (128 * MAX_ROWS + 64).next_multiple_of(256);

/// Word offsets in the handoff region, and in its host image.
const SEQ_W: usize = 0;
const IDS_W: usize = 16;
const WTS_W: usize = 32;
const X_W: usize = 64;

/// The widest routing a handoff holds: ids and weights each have the 16
/// words before the next field.
const MAX_USED: usize = WTS_W - IDS_W;

const _: () = assert!(X_W - WTS_W >= MAX_USED && MAX_USED >= EXPERTS_INTO_MAX);

/// What the counter is set to when a service fails: every wait still pending
/// in the stream passes, so a failed step drains instead of hanging.
const RELEASE: u32 = 1 << 30;

/// How long a service waits for its go before it gives up and releases the
/// stream.
const GO_DEADLINE: Duration = Duration::from_secs(10);

/// Spins between two clock reads while waiting for a go.
const DEADLINE_POLL: u32 = 1 << 12;

/// Pinned host memory mapped into the device's address space
/// (`cuMemHostAlloc` with `DEVICEMAP`): the host reaches it at `host`,
/// kernels, copies and stream memory operations at `dev`. Zeroed at
/// allocation, freed on drop.
struct MappedHost {
    host: *mut u8,
    dev: sys::CUdeviceptr,
    bytes: usize,
}

// SAFETY: the allocation is owned by this value alone and freed once, in its
// drop; the host pointer is plain process memory any thread may reach, and
// every access to it goes through `&self`/`&mut self` methods below.
unsafe impl Send for MappedHost {}

impl MappedHost {
    fn new(ctx: &Arc<CudaContext>, bytes: usize) -> Result<MappedHost, GpuError> {
        ctx.bind_to_thread()?;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: the context is current on this thread (bound above) and
        // `host` is a live local the call writes.
        let rc = unsafe { sys::cuMemHostAlloc(&mut host, bytes, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        cu(rc, "cuMemHostAlloc")?;
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `host` is the mapped allocation just made; the flags must be 0.
        let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0) };
        if let Err(e) = cu(rc, "cuMemHostGetDevicePointer_v2") {
            // SAFETY: `host` came from cuMemHostAlloc and is freed once, here.
            unsafe { sys::cuMemFreeHost(host) };
            return Err(e);
        }
        // SAFETY: the allocation holds `bytes` writable bytes and nothing else
        // references it yet.
        unsafe { std::ptr::write_bytes(host.cast::<u8>(), 0, bytes) };
        Ok(MappedHost {
            host: host.cast(),
            dev,
            bytes,
        })
    }

    /// The device address of byte `off`, which the layout keeps inside the
    /// allocation.
    fn dev_at(&self, off: usize) -> sys::CUdeviceptr {
        self.dev + off as u64
    }

    /// The host address of byte `off`. Computing it touches nothing; each
    /// access below proves its own span.
    fn host_at(&self, off: usize) -> *mut u8 {
        self.host.wrapping_add(off)
    }

    /// Flag word `w`, as an atomic.
    fn word(&self, w: Word) -> &AtomicU32 {
        // SAFETY: every `Word` offset is a 4-aligned byte inside the first
        // PAYLOAD_OFF bytes of the allocation (`Boundary::new` sizes it past
        // them); `AtomicU32` has `u32`'s layout, and the host reaches a flag
        // word only through this view.
        unsafe { &*self.host_at(w.offset()).cast::<AtomicU32>() }
    }

    /// Word `i` of the handoff image of `words` words at byte `at`; `None`
    /// past it.
    fn payload_word(&self, at: usize, i: usize, words: usize) -> Option<u32> {
        if i >= words {
            return None;
        }
        // SAFETY: i < words, and an image's `words` words start at a
        // word-aligned `at` inside the allocation (`Boundary::new` lays the
        // images out). The write that filled them finished before the go the
        // caller acquired, and none writes them again before the caller's
        // signal.
        Some(unsafe { self.host_at(at + 4 * i).cast::<u32>().read_volatile() })
    }

    /// Copy `dst.len()` f32 of the image of `words` words at byte `at` from
    /// word `from` into `dst`; `false` when the span passes the image.
    fn payload_f32_into(&self, at: usize, from: usize, words: usize, dst: &mut [f32]) -> bool {
        if from + dst.len() > words {
            return false;
        }
        // SAFETY: the span is inside the image (checked above), which is
        // f32-aligned at `at` + 4·from; `dst` is a distinct host slice.
        // Ordered after the go as in `payload_word`.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.host_at(at + 4 * from).cast::<f32>(),
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
        true
    }

    /// The `len` f32 at byte `off`, for the host to write. The caller holds
    /// the page mutably, and the card reads the span only between a wait and
    /// the next go — never while the host writes it.
    fn f32_mut(&mut self, off: usize, len: usize) -> Option<&mut [f32]> {
        if !off.is_multiple_of(4) || off + 4 * len > self.bytes {
            return None;
        }
        // SAFETY: the span is inside the allocation and 4-aligned (checked
        // above); `&mut self` makes it the only host reference.
        Some(unsafe { std::slice::from_raw_parts_mut(self.host_at(off).cast::<f32>(), len) })
    }

    /// A copy of the `len` f32 at byte `off`, when inside the allocation.
    fn f32_copy(&self, off: usize, len: usize) -> Option<Vec<f32>> {
        if !off.is_multiple_of(4) || off + 4 * len > self.bytes {
            return None;
        }
        let mut out = vec![0.0f32; len];
        // SAFETY: the span is inside the allocation and 4-aligned (checked
        // above); `out` is a distinct host buffer of `len` f32.
        unsafe {
            std::ptr::copy_nonoverlapping(self.host_at(off).cast::<f32>(), out.as_mut_ptr(), len);
        }
        Some(out)
    }
}

impl Drop for MappedHost {
    fn drop(&mut self) {
        // SAFETY: `host` came from cuMemHostAlloc and is freed once, here —
        // after every graph that names it, since the graphs are declared
        // before the body that owns this page. A failure on the drop path is
        // unreportable and ignored.
        unsafe { sys::cuMemFreeHost(self.host.cast()) };
    }
}

// ------------------------------------------------------ stream memory ops

/// An all-zero batch operation.
fn op_zero() -> sys::CUstreamBatchMemOpParams {
    // SAFETY: every member of the union is a plain C struct of integers, so
    // all-zero bytes are a valid value.
    unsafe { std::mem::zeroed() }
}

/// Wait until the u32 at `addr` is at least `value`.
fn op_wait_geq(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
    let mut p = op_zero();
    p.waitValue = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st {
        operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_WAIT_VALUE_32,
        address: addr,
        __bindgen_anon_1:
            sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1 {
                value,
            },
        flags: sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ,
        alias: 0,
    };
    p
}

/// Write `value` to the u32 at `addr`.
fn op_write(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
    let mut p = op_zero();
    p.writeValue = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st {
        operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_WRITE_VALUE_32,
        address: addr,
        __bindgen_anon_1:
            sys::CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1 {
                value,
            },
        flags: sys::CUstreamWriteValue_flags_enum_CU_STREAM_WRITE_VALUE_DEFAULT,
        alias: 0,
    };
    p
}

/// A system-scope memory barrier: everything the stream wrote before it is
/// visible system-wide before anything after it.
fn op_barrier_sys() -> sys::CUstreamBatchMemOpParams {
    let mut p = op_zero();
    p.memoryBarrier = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st {
        operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_BARRIER,
        flags: sys::CUstreamMemoryBarrier_flags_enum_CU_STREAM_MEMORY_BARRIER_TYPE_SYS,
    };
    p
}

/// Atomic reduction `*addr += value` on a u32 (wrapping).
fn op_add(addr: sys::CUdeviceptr, value: u32) -> sys::CUstreamBatchMemOpParams {
    let mut p = op_zero();
    p.atomicReduction = sys::CUstreamBatchMemOpParams_union_CUstreamMemOpAtomicReductionParams_st {
        operation: sys::CUstreamBatchMemOpType_enum_CU_STREAM_MEM_OP_ATOMIC_REDUCTION,
        flags: 0,
        reductionOp: sys::CUstreamAtomicReductionOpType_enum_CU_STREAM_ATOMIC_REDUCTION_OP_ADD,
        dataType: sys::CUstreamAtomicReductionDataType_enum_CU_STREAM_ATOMIC_REDUCTION_UNSIGNED_32,
        address: addr,
        value: u64::from(value),
        alias: 0,
    };
    p
}

/// Enqueue `ops` on `stream` as one batch of stream memory operations — one
/// graph node when captured.
fn mem_batch(
    stream: &CudaStream,
    ops: &mut [sys::CUstreamBatchMemOpParams],
    what: &'static str,
) -> Result<(), GpuError> {
    let n = u32::try_from(ops.len()).map_err(|_| GpuError::shape(what, "batch too long"))?;
    // SAFETY: `ops` is a live array of `n` initialized operations the driver
    // reads during the call (a capture copies them into the node); the flags
    // must be 0.
    let rc = unsafe { sys::cuStreamBatchMemOp_v2(stream.cu_stream(), n, ops.as_mut_ptr(), 0) };
    cu(rc, what)
}

/// Whether `stream` is recording a capture right now.
fn capturing(stream: &CudaStream) -> Result<bool, GpuError> {
    let mut status: sys::CUstreamCaptureStatus = 0;
    // SAFETY: the stream is live and `status` is a local the call writes.
    let rc = unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), &mut status) };
    cu(rc, "cuStreamIsCapturing")?;
    Ok(status != sys::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_NONE)
}

/// Serial-number order on a wrapping u32: `a` comes strictly before `b`.
fn before(a: u32, b: u32) -> bool {
    a != b && b.wrapping_sub(a) < 1 << 31
}

// --------------------------------------------------------------- boundary

/// The shapes a boundary is cut for.
#[derive(Clone, Copy, Debug)]
pub struct BoundaryShape {
    /// The model width: the normed activation, the host sum.
    pub hidden: usize,
    /// Routed slots per token.
    pub n_used: usize,
}

/// One row's part of the host-mapped page: its handoff image and the host
/// sum its combine reads, with their windows.
pub(crate) struct RowPage {
    /// Byte offsets of the image and of the sum in the page.
    image_off: usize,
    hsum_off: usize,
    image: ManuallyDrop<DeviceBuffer<u32>>,
    /// The row's host experts' weighted sum, read in place through the
    /// mapping.
    pub(crate) hsum: ManuallyDrop<DeviceBuffer<f32>>,
}

/// The card side of the boundary, allocated once at load (decision 4 of
/// docs/gpu-design.md: a captured graph bakes every address in): the handoff
/// region the router's launches write, the host-mapped page, and the sum the
/// combine reads.
///
/// A boundary of several rows lets that many tokens' handoffs be in flight
/// at once, served in go order: each row has its own layer word, image and
/// sum in the page; the region, the generation, the counter and the
/// sequence are shared — the region's readers of one row finish on the
/// stream before the next row's norm writes it, and the other three count
/// every go and service in one order.
pub struct Boundary {
    /// Windows into `region` — what the norm and the router write in place
    /// of the MoE arena's own buffers on a hybrid layer.
    pub(crate) normed: ManuallyDrop<DeviceBuffer<f32>>,
    pub(crate) ids: ManuallyDrop<DeviceBuffer<u32>>,
    pub(crate) weights: ManuallyDrop<DeviceBuffer<f32>>,
    /// The region's sequence word: what a chain's own launch copies into a
    /// row's image in place of the copy node ([`Boundary::handoff_target`]).
    seq: ManuallyDrop<DeviceBuffer<u32>>,
    /// Per row, its image and sum in the page; never empty. A one-row
    /// chain reads row 0's sum as `pages[0].hsum`.
    pub(crate) pages: Vec<RowPage>,
    /// `shared expert + hsum`: the combine's shared-expert input.
    pub(crate) sum: DeviceBuffer<f32>,
    /// Which experts each layer's card stack holds; the host serves the rest.
    pub(crate) slots: SlotMap,
    /// Whether the wait sits before the combine (`true`) or right after the
    /// go.
    pub(crate) overlap: bool,
    region: DeviceBuffer<u32>,
    page: MappedHost,
    shape: BoundaryShape,
}

/// The handoff's layout in its image: the word offsets of the sequence, the
/// routing's ids and weights (`n_used` each) and the activation (`hidden`
/// f32, the image's tail).
#[derive(Clone, Copy, Debug)]
pub struct HandoffLayout {
    pub seq: usize,
    pub ids: usize,
    pub weights: usize,
    pub x: usize,
    pub n_used: usize,
    pub hidden: usize,
}

/// What a chain's own handoff launch reads and writes
/// ([`Boundary::handoff_target`]).
pub struct HandoffTarget<'a> {
    /// The activation the norm wrote into the region, `hidden` f32.
    pub x: &'a DeviceBuffer<f32>,
    /// The region's sequence word: the image carries it as it stands before
    /// the go adds one.
    pub seq: &'a DeviceBuffer<u32>,
    /// The handoff's image in the host-mapped page, `layout.x + hidden`
    /// words; the launch writes every field of it.
    pub image: &'a mut DeviceBuffer<u32>,
    pub layout: HandoffLayout,
}

impl Drop for RowPage {
    fn drop(&mut self) {
        // SAFETY: each window is taken once, here, and never read again; its
        // raw parts are dropped (the context handle with them) and no memory
        // is freed — the boundary's page frees its allocation after the rows.
        unsafe {
            drop(ManuallyDrop::take(&mut self.image).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.hsum).into_raw_parts());
        }
    }
}

impl Drop for Boundary {
    fn drop(&mut self) {
        // SAFETY: each window is taken once, here, and never read again; its
        // raw parts are dropped (the context handle with them) and no memory
        // is freed — `region` frees the allocation after this.
        unsafe {
            drop(ManuallyDrop::take(&mut self.normed).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.ids).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.weights).into_raw_parts());
            drop(ManuallyDrop::take(&mut self.seq).into_raw_parts());
        }
    }
}

impl Boundary {
    /// Allocate the region, the page and the sum for `shape`; the host
    /// serves the experts `slots` sends to it, and `overlap` places each
    /// layer's wait. One row. Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        shape: BoundaryShape,
        slots: SlotMap,
        overlap: bool,
    ) -> Result<Boundary, GpuError> {
        Boundary::with_rows(ctx, stream, shape, slots, overlap, 1)
    }

    /// [`Boundary::new`] with `rows` rows (1..=[`MAX_ROWS`]): row 0's image
    /// and sum sit where a one-row boundary has them, the other rows' images
    /// after it and every sum after the images, each at a 256-byte boundary.
    /// Load-time only.
    pub fn with_rows(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        shape: BoundaryShape,
        slots: SlotMap,
        overlap: bool,
        rows: usize,
    ) -> Result<Boundary, GpuError> {
        let what = "Boundary::new";
        if shape.hidden == 0 || shape.n_used == 0 || shape.n_used > EXPERTS_INTO_MAX {
            return Err(GpuError::shape(
                what,
                format!(
                    "hidden {} and {} routed slots: the handoff carries 1..={EXPERTS_INTO_MAX} slots",
                    shape.hidden, shape.n_used
                ),
            ));
        }
        if !(1..=MAX_ROWS).contains(&rows) {
            return Err(GpuError::shape(
                what,
                format!("{rows} rows: a boundary carries 1..={MAX_ROWS}"),
            ));
        }
        let words = X_W + shape.hidden;
        let region = DeviceBuffer::<u32>::zeroed(stream, words)?;
        let image_stride = (4 * words).next_multiple_of(256);
        let hsum_stride = (4 * shape.hidden).next_multiple_of(256);
        let sums_off = PAYLOAD_OFF + rows * image_stride;
        let page = MappedHost::new(ctx, sums_off + rows * hsum_stride)?;
        let base = region.cu_deviceptr();
        // SAFETY: the ids span is n_used <= MAX_USED words at IDS_W, inside the
        // region's X_W + hidden words; `region` moves into the boundary beside
        // the window (a move of the handle, not of the allocation), where it
        // outlives it and is never reallocated.
        let ids = unsafe { window::<u32>(base + 4 * IDS_W as u64, shape.n_used, ctx) };
        // SAFETY: as above, for the weights' n_used words at WTS_W.
        let weights = unsafe { window::<f32>(base + 4 * WTS_W as u64, shape.n_used, ctx) };
        // SAFETY: as above, for the activation's `hidden` words at X_W, the
        // region's tail.
        let normed = unsafe { window::<f32>(base + 4 * X_W as u64, shape.hidden, ctx) };
        // SAFETY: as for the ids, for the sequence's one word at SEQ_W.
        let seq = unsafe { window::<u32>(base + 4 * SEQ_W as u64, 1, ctx) };
        let pages = (0..rows)
            .map(|r| {
                let (image_off, hsum_off) =
                    (PAYLOAD_OFF + r * image_stride, sums_off + r * hsum_stride);
                // SAFETY: row r's image — `words` words from a 256-aligned
                // offset — and its sum — `hidden` f32 from a 256-aligned
                // offset — lie inside the page (sized above past the last
                // sum), apart from each other and from every other row's;
                // the page moves into the boundary beside the windows and is
                // freed only when the boundary drops.
                let (image, hsum) = unsafe {
                    (
                        window::<u32>(page.dev_at(image_off), words, ctx),
                        window::<f32>(page.dev_at(hsum_off), shape.hidden, ctx),
                    )
                };
                RowPage {
                    image_off,
                    hsum_off,
                    image,
                    hsum,
                }
            })
            .collect::<Vec<_>>();
        Ok(Boundary {
            normed,
            ids,
            weights,
            seq,
            pages,
            sum: DeviceBuffer::<f32>::zeroed(stream, shape.hidden)?,
            slots,
            overlap,
            region,
            page,
            shape,
        })
    }

    /// Rows the boundary carries.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.pages.len()
    }

    /// Row `row`'s part of the page; a row the boundary does not carry is
    /// refused.
    fn page_of(&self, row: usize, what: &'static str) -> Result<&RowPage, GpuError> {
        self.pages.get(row).ok_or_else(|| {
            GpuError::shape(
                what,
                format!("row {row} of a boundary of {} rows", self.pages.len()),
            )
        })
    }

    /// The slot map the host tier serves by.
    #[must_use]
    pub fn slots(&self) -> &SlotMap {
        &self.slots
    }

    /// Whether each layer's wait sits just before the combine (`true`) or
    /// right after its go.
    #[must_use]
    pub fn overlap(&self) -> bool {
        self.overlap
    }

    /// The handoff's activation, `hidden` f32: what the norm writes on a
    /// hybrid layer and the host reads.
    #[must_use]
    pub fn normed(&self) -> &DeviceBuffer<f32> {
        &self.normed
    }

    /// The handoff's activation, for the norm to write.
    pub fn normed_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.normed
    }

    /// The host experts' weighted sum, `hidden` f32 in the host-mapped page:
    /// what a combine reads in place after the layer's wait. Row 0's.
    #[must_use]
    pub fn hsum(&self) -> &DeviceBuffer<f32> {
        &self.pages[0].hsum
    }

    /// Row `row`'s host sum ([`Boundary::hsum`]).
    pub fn hsum_of(&self, row: usize) -> Result<&DeviceBuffer<f32>, GpuError> {
        Ok(&self.page_of(row, "Boundary::hsum_of")?.hsum)
    }

    /// What a chain's own launch writes the handoff through, in place of the
    /// copy node: the activation the norm wrote into the region, the region's
    /// sequence word, and the handoff's image in the host-mapped page with its
    /// layout. The launch copies the sequence word, the routing and the
    /// activation into the image; [`Boundary::enqueue_go`] follows it. Row
    /// 0's image.
    pub fn handoff_target(&mut self) -> HandoffTarget<'_> {
        self.handoff_target_of(0)
            .expect("every boundary carries row 0")
    }

    /// [`Boundary::handoff_target`] into row `row`'s image.
    pub fn handoff_target_of(&mut self, row: usize) -> Result<HandoffTarget<'_>, GpuError> {
        let rows = self.pages.len();
        let page = self.pages.get_mut(row).ok_or_else(|| {
            GpuError::shape(
                "Boundary::handoff_target_of",
                format!("row {row} of a boundary of {rows} rows"),
            )
        })?;
        Ok(HandoffTarget {
            x: &self.normed,
            seq: &self.seq,
            image: &mut page.image,
            layout: HandoffLayout {
                seq: SEQ_W,
                ids: IDS_W,
                weights: WTS_W,
                x: X_W,
                n_used: self.shape.n_used,
                hidden: self.shape.hidden,
            },
        })
    }

    /// Words in the handoff region and in its host image.
    fn words(&self) -> usize {
        X_W + self.shape.hidden
    }

    /// Bytes the handoff's copy node moves.
    pub(crate) fn handoff_bytes(&self) -> usize {
        4 * self.words()
    }

    /// Device bytes the boundary holds: the region and the sum. The page is
    /// host memory.
    pub fn device_bytes(&self) -> usize {
        self.region.num_bytes() + self.sum.num_bytes()
    }

    /// Enqueue layer `layer`'s handoff: the region's image into the page (one
    /// copy node), then the go batch ([`Boundary::enqueue_go`]).
    pub(crate) fn enqueue_out(&self, stream: &CudaStream, layer: usize) -> Result<(), GpuError> {
        let what = "Boundary::enqueue_out";
        let layer = u32::try_from(layer).map_err(|_| GpuError::shape(what, "layer passes u32"))?;
        let bytes = self.handoff_bytes();
        // SAFETY: the page holds the region's image of `bytes` bytes at
        // PAYLOAD_OFF (sized so in `new`); both allocations live as long as
        // the boundary, which outlives every graph that captured this copy.
        let rc = unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                self.page.host_at(PAYLOAD_OFF).cast(),
                self.region.cu_deviceptr(),
                bytes,
                stream.cu_stream(),
            )
        };
        cu(rc, "cuMemcpyDtoHAsync_v2 (hybrid handoff)")?;
        self.go_batch(stream, layer, 0)
    }

    /// Enqueue layer `layer`'s go alone, for a chain whose own launch wrote the
    /// handoff's image ([`Boundary::handoff_target`]): a system barrier (the
    /// image lands first), the layer into the page, a second barrier, and one
    /// added to the generation word and to the sequence word the next handoff
    /// carries. The host serves it once the chain says the layer is enqueued
    /// ([`Hybrid::layer_enqueued`]). Row 0's go.
    pub fn enqueue_go(&self, stream: &CudaStream, layer: usize) -> Result<(), GpuError> {
        self.enqueue_go_of(stream, layer, 0)
    }

    /// [`Boundary::enqueue_go`] of row `row`: its layer word takes the layer.
    pub fn enqueue_go_of(
        &self,
        stream: &CudaStream,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        let what = "Boundary::enqueue_go";
        let layer = u32::try_from(layer).map_err(|_| GpuError::shape(what, "layer passes u32"))?;
        self.page_of(row, what)?;
        self.go_batch(stream, layer, row)
    }

    fn go_batch(&self, stream: &CudaStream, layer: u32, row: usize) -> Result<(), GpuError> {
        mem_batch(
            stream,
            &mut [
                op_barrier_sys(),
                op_write(self.page.dev_at(Word::Lyr(row).offset()), layer),
                op_barrier_sys(),
                op_add(self.page.dev_at(Word::Gen.offset()), 1),
                op_add(self.region.cu_deviceptr() + 4 * SEQ_W as u64, 1),
            ],
            "cuStreamBatchMemOp_v2 (hybrid go)",
        )
    }

    /// Enqueue the wait: until the counter is at least one, then minus one.
    /// Row 0's.
    pub fn enqueue_back(&self, stream: &CudaStream) -> Result<(), GpuError> {
        self.enqueue_back_of(stream, 0)
    }

    /// [`Boundary::enqueue_back`] on row `row`'s counter.
    pub fn enqueue_back_of(&self, stream: &CudaStream, row: usize) -> Result<(), GpuError> {
        self.page_of(row, "Boundary::enqueue_back")?;
        let cnt = self.page.dev_at(Word::Cnt(row).offset());
        mem_batch(
            stream,
            &mut [op_wait_geq(cnt, 1), op_add(cnt, u32::MAX)],
            "cuStreamBatchMemOp_v2 (hybrid wait)",
        )
    }

    /// Release every wait still pending in the stream, for good: every
    /// row's counter goes far past anything a step subtracts.
    fn release(&self) {
        for row in 0..self.pages.len() {
            self.page
                .word(Word::Cnt(row))
                .store(RELEASE, Ordering::Release);
        }
    }
}

// ------------------------------------------------------------------- host

/// What one host service computes, supplied by the architecture: the
/// weighted sum of the listed routed experts of layer `layer` for the
/// activation `x` (one column of the model width), written over `out` (that
/// width; an empty list writes zeros). The list is the handoff's host slots
/// in slot order, at most [`EXPERTS_INTO_MAX`], each id with its routing
/// weight. The protocol around the call — waiting for the go, checking the
/// handoff, signalling the card, releasing it on a failure — is
/// [`Hybrid`]'s alone.
pub trait HostExperts {
    fn experts_into(
        &mut self,
        layer: usize,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
    ) -> Result<(), GpuError>;

    /// The batch form: `lists[j]` for column `j` of `x`, its sum over
    /// `out[j · width ..][.. width]`, each column bit for bit what
    /// [`HostExperts::experts_into`] writes for that column and list alone. An
    /// architecture without a batched host path refuses, by name.
    fn experts_union_into(
        &mut self,
        layer: usize,
        x: &Tensor2,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let _ = (layer, x, lists, out);
        Err(GpuError::state(
            "HostExperts::experts_union_into",
            "a batched host path: this architecture's host tier serves one column a call",
        ))
    }
}

/// What the host side has done since load. Counted by the decode thread
/// alone; nothing here takes a lock.
#[derive(Clone, Copy, Debug, Default)]
pub struct HybridStats {
    /// Layers served.
    pub served: u64,
    /// Services whose go had already landed when the host began to wait for
    /// it — the card was waiting on the host.
    pub go_early: u64,
    /// Of those, the ones that opened a replay: the go of the chain's first
    /// hybrid layer landed before the launch returned to the host.
    pub go_early_first: u64,
    /// Host slots computed, summed over services.
    pub host_slots: u64,
    /// The host's share of each service's routed weight squared,
    /// `Σ_host w² / Σ_all w²`, summed over services.
    pub host_w2: f64,
    /// Host wall time from the go seen to the signal, summed (ns).
    pub leg_ns: u64,
    /// Of every watched replay ([`serving_replay`]), the time from just before
    /// its graph launch was issued to the entry of its first service, summed
    /// (ns). Lever off that holds the whole launch call; with the launch on
    /// its own thread only the post and the hand-over to the service.
    pub first_serve_lag_ns: u64,
    /// Pool workers that parked between the start of a service's wait and
    /// its signal, summed.
    pub parks_in_service: u64,
    /// Time from the decode thread seeing a go to the last pool thread
    /// leaving the wait — what a worker that was not spinning when the go
    /// landed (parked, or preempted) adds before the experts start. Summed
    /// and worst, over services whose go was not already there (ns).
    pub straggle_ns: u64,
    pub straggle_max_ns: u64,
    /// Of a two-row pass ([`Chain::Pair`]), the host slot ids of row 1's
    /// service of a layer that row 0's service of the same layer also
    /// listed, summed over row 1's services. The one-token step adds nothing.
    pub overlap_slots: u64,
    /// Host slots computed by row 1's services of a two-row pass, summed:
    /// the denominator of the row overlap `overlap_slots / pair_row1_slots`.
    /// The union of a layer's two host lists is `host_slots − overlap_slots`
    /// over any span of whole passes.
    pub pair_row1_slots: u64,
    /// Batch services ([`Hybrid::serve_batch`]): layers served, the columns
    /// they carried, the host slots those columns listed, and the host wall
    /// time of the union calls (ns), summed. None of them counts above.
    pub batch_served: u64,
    pub batch_cols: u64,
    pub batch_host_slots: u64,
    pub batch_ns: u64,
}

/// Row 0's host slot ids of the layer its two-row pass served last — what
/// row 1's service of that layer counts its overlap against. Written by
/// [`Chain::Pair`] services only.
#[derive(Clone, Copy, Debug)]
struct PairRow0 {
    layer: usize,
    n: usize,
    ids: [u32; EXPERTS_INTO_MAX],
}

/// A chain the host tier serves: the one-token step, or the two-row pass
/// whose rows run one layer apart (row `r`'s go of layer `l` and the other
/// row's of the layer before can both be in flight).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    Step,
    Pair,
}

impl Chain {
    fn index(self) -> usize {
        match self {
            Chain::Step => 0,
            Chain::Pair => 1,
        }
    }

    /// Goes of the chain that can have landed past the one being served,
    /// plus one: the rows in flight.
    fn in_flight(self) -> usize {
        match self {
            Chain::Step => 1,
            Chain::Pair => 2,
        }
    }
}

/// The replay the decode thread is about to serve: when its graph launch was
/// issued, and the flag its launch thread raises when that launch fails —
/// `None` when the decode thread made the call itself and has its result.
#[derive(Clone, Debug)]
pub(crate) struct ReplayWatch {
    pub(crate) issued: Instant,
    pub(crate) failed: Option<Arc<AtomicBool>>,
}

thread_local! {
    /// The watch of the replay this thread serves inside [`serving_replay`].
    /// A scope, not state: `ChainBody::serve_replay` takes no argument and
    /// every architecture's body forwards it to [`Hybrid::serve_captured`],
    /// so the skeleton hands the replay's watch to the tier through the
    /// thread that serves it.
    static REPLAY: RefCell<Option<ReplayWatch>> = const { RefCell::new(None) };
}

/// Run `serve` — the body's share of one replay — with `watch` as the
/// replay's watch; the previous one is back when it returns or unwinds.
pub(crate) fn serving_replay<R>(watch: ReplayWatch, serve: impl FnOnce() -> R) -> R {
    struct Restore(Option<ReplayWatch>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let prev = self.0.take();
            REPLAY.with(|r| *r.borrow_mut() = prev);
        }
    }
    let _restore = Restore(REPLAY.with(|r| r.borrow_mut().replace(watch)));
    serve()
}

/// The boundary and the host tier that serves it: the architecture's host
/// computation and the protocol state.
pub struct Hybrid<H> {
    pub(crate) boundary: Boundary,
    host: H,
    /// The host's copy of a handoff's activation — one column, allocated at
    /// load.
    x: Tensor2,
    /// Services done: the sequence number the next handoff carries.
    served: u32,
    /// Per [`Chain`], the (layer, row) services its last capture recorded,
    /// in go order — what a replay of it asks the host to serve.
    captured: [Vec<(usize, usize)>; 2],
    /// The chain being enqueued, and whether it is a capture, not an eager
    /// step.
    chain: Chain,
    capturing: bool,
    /// A service failed and released the stream; nothing is served again.
    poisoned: bool,
    stats: HybridStats,
    /// For the row overlap in `stats`; `None` until a pair's row 0 is served.
    pair_row0: Option<PairRow0>,
    /// A batch service's host lists, [`EXPERTS_INTO_MAX`] entries a column
    /// and each column's length; grown to the widest batch served, once.
    batch_lists: Vec<(u32, f32)>,
    batch_lens: Vec<usize>,
}

impl<H: HostExperts> Hybrid<H> {
    /// The host tier over `boundary`: `host` computes every expert the
    /// boundary's slot map sends to the host, and `layers` bounds the chain.
    /// Load-time only.
    pub fn new(boundary: Boundary, host: H, layers: usize) -> Result<Hybrid<H>, GpuError> {
        let hidden = boundary.shape.hidden;
        Ok(Hybrid {
            boundary,
            host,
            x: Tensor2::zeros(hidden, 1),
            served: 0,
            captured: [
                Vec::with_capacity(layers),
                Vec::with_capacity(Chain::Pair.in_flight() * layers),
            ],
            chain: Chain::Step,
            capturing: false,
            poisoned: false,
            stats: HybridStats::default(),
            pair_row0: None,
            batch_lists: Vec::new(),
            batch_lens: Vec::new(),
        })
    }

    /// The boundary the chain enqueues its handoffs and waits on.
    #[must_use]
    pub fn boundary(&self) -> &Boundary {
        &self.boundary
    }

    /// The boundary, for a chain whose launches write its handoff.
    pub fn boundary_mut(&mut self) -> &mut Boundary {
        &mut self.boundary
    }

    /// What the host side has done since load.
    #[must_use]
    pub fn stats(&self) -> HybridStats {
        self.stats
    }

    /// The host experts, for a caller that prepares their scratch ahead of a
    /// service.
    pub fn host_mut(&mut self) -> &mut H {
        &mut self.host
    }

    /// Row 0's host sum as the page holds it now: the last layer served
    /// for that row.
    pub fn hsum_copy(&self) -> Result<Vec<f32>, GpuError> {
        let what = "Hybrid::hsum_copy";
        let off = self.boundary.page_of(0, what)?.hsum_off;
        self.boundary
            .page
            .f32_copy(off, self.boundary.shape.hidden)
            .ok_or(GpuError::state(what, "the sum is outside the page"))
    }

    /// Open a one-token chain on `stream`: a capture records which layers
    /// its replays will ask for; an eager chain is served as it goes.
    pub fn begin_chain(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.begin_chain_of(stream, Chain::Step)
    }

    /// [`Hybrid::begin_chain`] of `chain`; a pair needs a boundary of two
    /// rows.
    pub fn begin_chain_of(&mut self, stream: &CudaStream, chain: Chain) -> Result<(), GpuError> {
        let what = "Hybrid::begin_chain";
        self.refuse_if_poisoned(what)?;
        if chain.in_flight() > self.boundary.rows() {
            return Err(GpuError::shape(
                what,
                format!(
                    "a {chain:?} chain on a boundary of {} rows",
                    self.boundary.rows()
                ),
            ));
        }
        self.chain = chain;
        self.capturing = capturing(stream)?;
        if self.capturing {
            self.captured[chain.index()].clear();
        }
        Ok(())
    }

    /// Layer `layer`'s hybrid work is enqueued: a capture notes it, an eager
    /// chain serves it now, before anything more joins the stream behind its
    /// wait. Row 0's.
    pub fn layer_enqueued(&mut self, layer: usize) -> Result<(), GpuError> {
        self.row_enqueued(layer, 0)
    }

    /// [`Hybrid::layer_enqueued`] for row `row`. The chain enqueues its
    /// services' waits in go order, and so is served in it.
    pub fn row_enqueued(&mut self, layer: usize, row: usize) -> Result<(), GpuError> {
        if self.capturing {
            self.captured[self.chain.index()].push((layer, row));
            Ok(())
        } else {
            self.serve(layer, row, false, self.chain, None)
        }
    }

    /// Serve every hybrid layer a replay of the captured one-token chain
    /// submitted, in chain order.
    pub fn serve_captured(&mut self) -> Result<(), GpuError> {
        self.serve_captured_of(Chain::Step)
    }

    /// [`Hybrid::serve_captured`] for a replay of `chain`'s capture.
    ///
    /// Inside [`serving_replay`] the first service adds its lag to
    /// [`HybridStats::first_serve_lag_ns`], and every service stops waiting
    /// for its go once the watch's launch has failed.
    pub fn serve_captured_of(&mut self, chain: Chain) -> Result<(), GpuError> {
        let watch = REPLAY.with(|r| r.borrow().clone());
        let c = chain.index();
        for i in 0..self.captured[c].len() {
            let (layer, row) = self.captured[c][i];
            self.serve(layer, row, i == 0, chain, watch.as_ref())?;
        }
        Ok(())
    }

    /// Serve layer `layer` for an eager batch of `x.ne1` tokens in one union
    /// call, outside the go/wait protocol: the caller has brought the
    /// batch's handoffs to the host — `x`, one normed activation per column,
    /// and `ids` and `weights`, the routing, `n_used` per column in slot
    /// order — and reads the host sums back from `out`, `hidden` per column.
    /// Column `j`'s list is its slots the slot map sends to the host or does
    /// not know, in slot order: the list a step's service builds for that
    /// token. A non-finite activation is undefined input the card's norm has
    /// already raised as a fault, which the batch's readback turns into the
    /// named error: every sum is NaN then and no host expert runs. Refused
    /// on a poisoned tier; a failure, or a panic inside the host experts,
    /// poisons it.
    pub fn serve_batch(
        &mut self,
        layer: usize,
        x: &Tensor2,
        ids: &[u32],
        weights: &[f32],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Hybrid::serve_batch";
        self.refuse_if_poisoned(WHAT)?;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.serve_batch_one(layer, x, ids, weights, out)
        }));
        match r {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                self.poisoned = true;
                Err(e)
            }
            Err(p) => {
                self.poisoned = true;
                std::panic::resume_unwind(p)
            }
        }
    }

    fn serve_batch_one(
        &mut self,
        layer: usize,
        x: &Tensor2,
        ids: &[u32],
        weights: &[f32],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Hybrid::serve_batch";
        let (hidden, n_used) = (self.boundary.shape.hidden, self.boundary.shape.n_used);
        let cols = x.ne1;
        if x.ne0 != hidden
            || cols == 0
            || ids.len() != cols * n_used
            || weights.len() != cols * n_used
            || out.len() != cols * hidden
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{cols} columns of {} values with {} ids, {} weights and {} sums; the \
                     boundary carries {n_used} slots of {hidden} values a column",
                    x.ne0,
                    ids.len(),
                    weights.len(),
                    out.len()
                ),
            ));
        }
        let t0 = Instant::now();
        // A fold, not `all`: no early exit, so the test vectorizes.
        let finite = x.data.iter().fold(true, |ok, v| ok & v.is_finite());
        if !finite {
            out.fill(f32::NAN);
            return Ok(());
        }
        let map_row = self.boundary.slots.row(layer).ok_or(GpuError::state(
            WHAT,
            "a hybrid layer without a slot map row",
        ))?;
        if self.batch_lens.len() < cols {
            self.batch_lens.resize(cols, 0);
            self.batch_lists
                .resize(cols * EXPERTS_INTO_MAX, (0, 0.0f32));
        }
        let mut host_slots = 0u64;
        for j in 0..cols {
            let list = &mut self.batch_lists[j * EXPERTS_INTO_MAX..][..EXPERTS_INTO_MAX];
            let mut n = 0usize;
            for s in 0..n_used {
                let (id, w) = (ids[j * n_used + s], weights[j * n_used + s]);
                let slot = usize::try_from(id).ok().and_then(|id| map_row.get(id));
                if slot.is_none_or(|&slot| slot == HOST) {
                    list[n] = (id, w);
                    n += 1;
                }
            }
            self.batch_lens[j] = n;
            host_slots += n as u64;
        }
        let lists: Vec<&[(u32, f32)]> = self.batch_lens[..cols]
            .iter()
            .enumerate()
            .map(|(j, &n)| &self.batch_lists[j * EXPERTS_INTO_MAX..][..n])
            .collect();
        self.host.experts_union_into(layer, x, &lists, out)?;
        let s = &mut self.stats;
        s.batch_served += 1;
        s.batch_cols += cols as u64;
        s.batch_host_slots += host_slots;
        s.batch_ns += u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok(())
    }

    fn refuse_if_poisoned(&self, what: &'static str) -> Result<(), GpuError> {
        if self.poisoned {
            return Err(GpuError::state(
                what,
                "an earlier hybrid service failed and released the stream",
            ));
        }
        Ok(())
    }

    /// Serve layer `layer` of row `row` in `chain`. Any failure — an error
    /// or a panic inside the host experts — releases every pending wait
    /// first, so the stream drains instead of hanging a later synchronize.
    fn serve(
        &mut self,
        layer: usize,
        row: usize,
        opens_replay: bool,
        chain: Chain,
        watch: Option<&ReplayWatch>,
    ) -> Result<(), GpuError> {
        self.refuse_if_poisoned("Hybrid::serve")?;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.serve_one(layer, row, opens_replay, chain, watch)
        }));
        match r {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                self.poisoned = true;
                self.boundary.release();
                Err(e)
            }
            Err(p) => {
                self.poisoned = true;
                self.boundary.release();
                std::panic::resume_unwind(p)
            }
        }
    }

    fn serve_one(
        &mut self,
        layer: usize,
        row: usize,
        opens_replay: bool,
        chain: Chain,
        watch: Option<&ReplayWatch>,
    ) -> Result<(), GpuError> {
        let what = "Hybrid::serve";
        let entered = Instant::now();
        if opens_replay && let Some(w) = watch {
            let lag = entered.saturating_duration_since(w.issued).as_nanos();
            self.stats.first_serve_lag_ns += u64::try_from(lag).unwrap_or(u64::MAX);
        }
        let failed = watch.and_then(|w| w.failed.as_deref());
        let (image_off, hsum_off) = {
            let p = self.boundary.page_of(row, what)?;
            (p.image_off, p.hsum_off)
        };
        let want = self.served.wrapping_add(1);
        let generation = self.boundary.page.word(Word::Gen);
        let early = !before(generation.load(Ordering::Acquire), want);
        let parks = threads::pool().stats().worker_parks;
        let (seen, straggle) = wait_go(generation, want, entered + GO_DEADLINE, failed);
        if early {
            self.stats.go_early += 1;
            self.stats.go_early_first += u64::from(opens_replay);
        } else {
            self.stats.straggle_ns += straggle;
            self.stats.straggle_max_ns = self.stats.straggle_max_ns.max(straggle);
        }
        // The goes of the rows in flight after this one may have landed too;
        // one past them means a wait is missing.
        let ahead = usize::try_from(seen.wrapping_sub(want)).unwrap_or(usize::MAX);
        if before(seen, want) && failed.is_some_and(|f| f.load(Ordering::Acquire)) {
            return Err(GpuError::protocol(
                what,
                format!("the replay's graph launch failed: the go of layer {layer} never lands"),
            ));
        }
        if before(seen, want) || ahead >= chain.in_flight() {
            let detail = if before(seen, want) {
                format!(
                    "the go of layer {layer} did not land in {GO_DEADLINE:?} (generation {seen}, want {want})"
                )
            } else {
                format!(
                    "generation {seen} is {ahead} past {want} with {} row(s) in flight: the card \
                     ran ahead of the host — a hybrid layer's wait is missing",
                    chain.in_flight()
                )
            };
            return Err(GpuError::protocol(what, detail));
        }
        let t0 = Instant::now();
        let page = &self.boundary.page;
        let words = self.boundary.words();
        let seq = page.payload_word(image_off, SEQ_W, words);
        if seq != Some(self.served) {
            return Err(GpuError::protocol(
                what,
                format!(
                    "the handoff carries sequence {seq:?}, the host is at {}: a stale handoff",
                    self.served
                ),
            ));
        }
        let lyr = page.word(Word::Lyr(row)).load(Ordering::Acquire);
        if usize::try_from(lyr).ok() != Some(layer) {
            return Err(GpuError::protocol(
                what,
                format!("row {row}'s go is layer {lyr}'s, the host serves layer {layer}"),
            ));
        }
        let map_row = self.boundary.slots.row(layer).ok_or(GpuError::state(
            what,
            "a hybrid layer without a slot map row",
        ))?;
        // The host's slots, in slot order: every routed id the slot map sends
        // to the host or does not know, with its weight.
        let mut list = [(0u32, 0.0f32); EXPERTS_INTO_MAX];
        let (mut n, mut w2_host, mut w2_all) = (0usize, 0.0f64, 0.0f64);
        for s in 0..self.boundary.shape.n_used {
            let (Some(id), Some(wb)) = (
                page.payload_word(image_off, IDS_W + s, words),
                page.payload_word(image_off, WTS_W + s, words),
            ) else {
                return Err(GpuError::shape(what, "the routing is outside the handoff"));
            };
            let w = f32::from_bits(wb);
            w2_all += f64::from(w) * f64::from(w);
            let slot = usize::try_from(id).ok().and_then(|id| map_row.get(id));
            if slot.is_none_or(|&slot| slot == HOST) {
                list[n] = (id, w);
                n += 1;
                w2_host += f64::from(w) * f64::from(w);
            }
        }
        if !page.payload_f32_into(image_off, X_W, words, &mut self.x.data) {
            return Err(GpuError::shape(
                what,
                "the activation is outside the handoff",
            ));
        }
        let hidden = self.boundary.shape.hidden;
        let out = self
            .boundary
            .page
            .f32_mut(hsum_off, hidden)
            .ok_or(GpuError::state(what, "the sum is outside the page"))?;
        // A non-finite handoff activation is undefined input the card has
        // already refused: the norm that wrote it tests every value it
        // writes and raised its fault, which the step's readback turns into
        // the named error. The host sum carries the NaN on instead of the
        // host encoders panicking first, so that error, with its layer and
        // site, is the one the caller sees.
        // A fold, not `all`: no early exit, so the test vectorizes.
        let finite = self.x.data.iter().fold(true, |ok, v| ok & v.is_finite());
        if finite {
            self.host.experts_into(layer, &self.x, &list[..n], out)?;
        } else {
            out.fill(f32::NAN);
        }
        self.boundary
            .page
            .word(Word::Cnt(row))
            .fetch_add(1, Ordering::Release);
        self.served = want;
        let s = &mut self.stats;
        s.served += 1;
        s.host_slots += n as u64;
        s.host_w2 += if w2_all > 0.0 { w2_host / w2_all } else { 0.0 };
        s.leg_ns += u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        s.parks_in_service += threads::pool().stats().worker_parks.saturating_sub(parks);
        if chain == Chain::Pair {
            self.count_pair_overlap(layer, row, &list[..n]);
        }
        Ok(())
    }

    /// The row overlap of a two-row pass: row 0's service keeps its host ids,
    /// row 1's service of the same layer — the next one served — counts its
    /// ids found among them. A copy of at most [`EXPERTS_INTO_MAX`] ids and
    /// that many squared compares, no lock; the one-token step never calls it.
    fn count_pair_overlap(&mut self, layer: usize, row: usize, list: &[(u32, f32)]) {
        if row == 0 {
            let mut ids = [0u32; EXPERTS_INTO_MAX];
            for (d, &(id, _)) in ids.iter_mut().zip(list) {
                *d = id;
            }
            self.pair_row0 = Some(PairRow0 {
                layer,
                n: list.len(),
                ids,
            });
        } else if let Some(r0) = self.pair_row0.take_if(|r0| r0.layer == layer) {
            let row0 = &r0.ids[..r0.n];
            let overlap = list.iter().filter(|(id, _)| row0.contains(id)).count();
            self.stats.overlap_slots += overlap as u64;
            self.stats.pair_row1_slots += list.len() as u64;
        }
    }
}

/// Wait until the generation word has reached `want`, `deadline` has passed
/// or `failed` (the replay's launch, when another thread issued it) is
/// raised, with every pool thread spinning on it — a pool job, so no worker
/// is parked when the go lands and the dispatch that follows starts inside
/// the spin window this job leaves them in. Returns the word as last read,
/// and the nanoseconds from the calling thread seeing it to the job's end
/// (the calling thread runs the last chunk and alone writes that stamp).
fn wait_go(
    generation: &AtomicU32,
    want: u32,
    deadline: Instant,
    failed: Option<&AtomicBool>,
) -> (u32, u64) {
    let pool = threads::pool();
    let caller = pool.threads() - 1;
    let base = Instant::now();
    let seen_at = AtomicU64::new(0);
    pool.for_each_chunk(pool.threads(), |chunk| {
        let mut spins = 0u32;
        while before(generation.load(Ordering::Acquire), want) {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(DEADLINE_POLL)
                && (Instant::now() > deadline || failed.is_some_and(|f| f.load(Ordering::Relaxed)))
            {
                break;
            }
            std::hint::spin_loop();
        }
        if chunk.start == caller {
            let ns = u64::try_from(base.elapsed().as_nanos()).unwrap_or(u64::MAX);
            seen_at.store(ns, Ordering::Relaxed);
        }
    });
    let end = u64::try_from(base.elapsed().as_nanos()).unwrap_or(u64::MAX);
    (
        generation.load(Ordering::Acquire),
        end.saturating_sub(seen_at.load(Ordering::Relaxed)),
    )
}

#[cfg(test)]
mod tests {
    use super::{Boundary, BoundaryShape, HOST, SlotMap};
    use cuda_core::CudaContext;
    use std::sync::Arc;

    /// A boundary gives back every context handle its windows took: the
    /// context's count after one is dropped is the count before it was made.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_boundary_gives_back_its_context_handles() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("a stream");
        let shape = BoundaryShape {
            hidden: 256,
            n_used: 6,
        };
        let before = Arc::strong_count(&ctx);
        let slots = SlotMap::prefix(0..2, 16, 8).expect("a prefix of 8 of 16");
        let b = Boundary::with_rows(&ctx, &stream, shape, slots, true, 2).expect("a boundary");
        let held = Arc::strong_count(&ctx);
        drop(b);
        let after = Arc::strong_count(&ctx);
        eprintln!("boundary context handles: before {before} held {held} after {after}");
        assert!(held > before, "the boundary's buffers hold the context");
        assert_eq!(after, before, "a dropped boundary leaves no context handle");
    }

    /// The prefix map sends to the host exactly the ids at or past `n_l`, on
    /// every layer it has a row for, and knows no other layer.
    #[test]
    fn prefix_sends_the_ids_past_n_l_to_the_host() {
        let map = SlotMap::prefix(1..4, 64, 22).expect("a prefix of 22 of 64");
        for layer in 1..4 {
            let row = map.row(layer).expect("a row per layer of the range");
            for (id, &slot) in row.iter().enumerate() {
                assert_eq!(slot == HOST, id >= 22, "layer {layer} id {id}: {slot}");
            }
            assert_eq!(map.on_card(layer), 22);
        }
        assert!(map.row(0).is_none() && map.row(4).is_none());
        assert_eq!(map.on_card(4), 0);
        assert!(SlotMap::prefix(0..1, 64, 65).is_err());
    }

    /// A row's card slots are `0..k` for its `k` experts on the card, each
    /// once; any other row is refused, and so is a wrong entry count.
    #[test]
    fn from_rows_refuses_a_slot_twice_or_past_the_card_count() {
        assert!(SlotMap::from_rows(0..1, 4, vec![1, HOST, 0, HOST]).is_ok());
        assert!(SlotMap::from_rows(0..1, 4, vec![0, 0, HOST, HOST]).is_err());
        assert!(SlotMap::from_rows(0..1, 4, vec![0, 2, HOST, HOST]).is_err());
        assert!(SlotMap::from_rows(0..2, 4, vec![0, 1, HOST, HOST]).is_err());
    }
}
