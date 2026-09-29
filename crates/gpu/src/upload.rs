//! The load's staging ring: pinned host slots a weight upload copies
//! through, so the host memcpy of one piece overlaps the device copy of the
//! piece before it, instead of every segment staging its own packing and
//! waiting for its own copy. A slot is reused only once the event recorded
//! after its copy has passed; one stream synchronize at the end of the load
//! covers every copy ([`UploadRing::finish`]), and the drop drains each
//! slot's event, so a load abandoned mid-upload frees nothing a copy still
//! reads.

use crate::GpuError;
use crate::tensor::window;
use cuda_core::{CudaStream, DeviceCopy, PinnedHostBuffer, sys};
use std::mem::ManuallyDrop;

// The staged uploads are the file's bytes read as the device's little-endian
// words; a target of another endianness would need a repacking path, not a
// silent reinterpretation.
#[cfg(not(target_endian = "little"))]
compile_error!(
    "the staged weight uploads treat file bytes as little-endian device words; this target is not little-endian"
);

/// Slots in the ring: one being copied to the device, one being filled, one
/// more so a fault burst on the filling side does not stall the pipe.
const SLOTS: usize = 3;
/// The most bytes one slot holds: pieces of this size amortize the copy
/// enqueue and the event around them, and larger slots would pin host
/// memory the tiers need for little more overlap.
const SLOT_MAX: usize = 32 << 20;
/// The least bytes one slot holds, so a small load — a gate's few tensors —
/// pins a small ring rather than a maximum one.
const SLOT_MIN: usize = 256 << 10;
/// The least bytes a piece is worth the worker pool for; shorter pieces copy
/// on the calling thread, the dispatch costing more than the copy.
const TEAM_MIN: usize = 1 << 20;

/// One staging slot: its pinned bytes and the event recorded after the copy
/// that last read them.
struct Slot {
    host: PinnedHostBuffer<u8>,
    done: cuda_core::CudaEvent,
}

/// A bounded ring of pinned staging slots, one per load: the uploads copy
/// each piece into the next free slot and enqueue the device copy from it,
/// so at most [`SLOTS`] pieces are in flight and the host side runs ahead of
/// the device by that much. Not sync: a load is one thread's work.
pub(crate) struct UploadRing {
    slots: Vec<Slot>,
    next: usize,
}

/// One tensor's staged upload: where its bytes go on the device, the size
/// of the allocation there, and how its refusals name it.
pub(crate) struct Stage<'a> {
    pub(crate) what: &'static str,
    pub(crate) name: &'a str,
    pub(crate) dst: sys::CUdeviceptr,
    pub(crate) dst_bytes: usize,
}

impl UploadRing {
    /// A ring for a load of `budget` upload bytes: [`SLOTS`] slots of the
    /// budget's share, each between [`SLOT_MIN`] and [`SLOT_MAX`]; a load of
    /// nothing takes no slots. `stream`'s context must be current (the
    /// pinned allocation and the events are made in it).
    pub(crate) fn new(stream: &CudaStream, budget: usize) -> Result<UploadRing, GpuError> {
        let ctx = stream.context();
        let n = usize::from(budget != 0) * SLOTS;
        let slot = budget.div_ceil(SLOTS).clamp(SLOT_MIN, SLOT_MAX);
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            slots.push(Slot {
                host: PinnedHostBuffer::zeroed(ctx, slot).map_err(|source| GpuError::Driver {
                    op: Some("cuMemAllocHost (the upload ring)"),
                    source,
                })?,
                done: ctx.new_event(None).map_err(|source| GpuError::Driver {
                    op: Some("cuEventCreate (the upload ring)"),
                    source,
                })?,
            });
        }
        Ok(UploadRing { slots, next: 0 })
    }

    /// Copy the first `take` bytes of `src`'s slices, in order, into the
    /// bytes at `stage.dst` on `stream`, through the ring. The copy stays
    /// inside `stage.dst_bytes` and never touches the bytes past `take`,
    /// which the caller's zeroed buffer owns (a padded format's tail is zero
    /// because nothing was copied over it). Nothing reads the slices once a
    /// piece's host memcpy has returned. A refusal from a piece already
    /// enqueued drains the stream first, so the buffer this error is about
    /// to drop is not freed under a running copy.
    pub(crate) fn copy_bytes(
        &mut self,
        stream: &CudaStream,
        stage: &Stage<'_>,
        src: &[&[u8]],
        take: usize,
    ) -> Result<(), GpuError> {
        let Stage {
            what,
            name,
            dst,
            dst_bytes,
        } = *stage;
        let held: usize = src.iter().map(|s| s.len()).sum();
        if held < take {
            return Err(GpuError::shape(
                what,
                format!("tensor {name} holds {held} bytes, its upload takes {take}"),
            ));
        }
        if take > dst_bytes {
            return Err(GpuError::shape(
                what,
                format!("tensor {name}: an upload of {take} bytes into a {dst_bytes}-byte buffer"),
            ));
        }
        let Some(slot_bytes) = self.slots.first().map(|s| s.host.len()) else {
            return if take == 0 {
                Ok(())
            } else {
                Err(GpuError::shape(
                    what,
                    format!("tensor {name}: the ring holds no slots for {take} bytes"),
                ))
            };
        };
        let mut at = 0usize;
        let mut left = take;
        for slice in src {
            if left == 0 {
                break;
            }
            let mut run = &slice[..slice.len().min(left)];
            while !run.is_empty() {
                let piece = run.len().min(slot_bytes);
                let off = u64::try_from(at).map_err(|_| {
                    GpuError::shape(what, format!("tensor {name}: byte {at} passes u64"))
                })?;
                self.push(stream, dst + off, &run[..piece])?;
                at += piece;
                run = &run[piece..];
                left -= piece;
            }
        }
        Ok(())
    }

    /// Wait for every copy this ring enqueued: the load's one synchronize,
    /// which the caller owes before anything it staged is read or released.
    pub(crate) fn finish(&self, stream: &CudaStream) -> Result<(), GpuError> {
        stream.synchronize()?;
        Ok(())
    }

    /// One piece: wait the next slot's earlier copy out, memcpy the piece
    /// into it, enqueue the device copy, and record the slot's event after
    /// it — the event is what the next use of the slot and the drop wait.
    fn push(
        &mut self,
        stream: &CudaStream,
        at: sys::CUdeviceptr,
        piece: &[u8],
    ) -> Result<(), GpuError> {
        let i = self.next;
        self.next = (self.next + 1) % self.slots.len();
        let slot = &mut self.slots[i];
        slot.done.synchronize().map_err(|source| GpuError::Driver {
            op: Some("cuEventSynchronize (an upload slot)"),
            source,
        })?;
        let staging = &mut slot.host.as_mut_slice()[..piece.len()];
        memcpy_pool(staging, piece);
        // SAFETY: the window is the `piece.len()` bytes at `at` of the
        // caller's allocation at `dst`, inside the `dst_bytes` copy_bytes
        // checked; the allocation is a cuMemAlloc one that outlives this
        // window, which is released below and frees nothing.
        let mut window = unsafe { window::<u8>(at, piece.len(), stream.context()) };
        // SAFETY: `staging` is pinned memory this ring owns and nothing
        // rewrites before `done` — recorded right below, after the copy —
        // has passed: the next use of this slot waits it, and the ring's
        // drop waits every slot's. The copy is enqueued, not run, here.
        let enqueue = unsafe { window.copy_from_host_async_unchecked(stream, staging) };
        drop(ManuallyDrop::into_inner(window).into_raw_parts());
        if let Err(source) = enqueue {
            // A piece before this one may still be copying into the caller's
            // buffer, which this error is about to drop: drain first, so
            // nothing is freed under a running copy.
            let _ = stream.synchronize();
            return Err(GpuError::Driver {
                op: Some("the staged copy of an upload piece"),
                source,
            });
        }
        if let Err(source) = slot.done.record(stream) {
            // The copy above sits enqueued without its event: nothing may
            // free the slot before it completes, so the stream is drained
            // here rather than left to the drop's event waits.
            let _ = stream.synchronize();
            return Err(GpuError::Driver {
                op: Some("the upload event of a staged piece"),
                source,
            });
        }
        Ok(())
    }
}

impl Drop for UploadRing {
    fn drop(&mut self) {
        // The pinned bytes outlive any copy still reading them: a slot's
        // event was recorded after its last copy, and one whose record
        // failed had its stream drained at once. A failure here is
        // unreportable and ignored — the load that owned this ring is
        // already refusing.
        for slot in &self.slots {
            let _ = slot.done.synchronize();
        }
    }
}

/// `src` into `dst` over the resident worker pool: the host side of the
/// upload is the memcpy of the source pages plus their first faults, which
/// one core serves slower than the device copies drain. The pool's workers
/// run on their own cores; threads spawned here would inherit the calling
/// thread's affinity, which a binary pins to one core before its load. A
/// short piece copies on the calling thread.
fn memcpy_pool(dst: &mut [u8], src: &[u8]) {
    let n = dst.len();
    debug_assert_eq!(n, src.len());
    if n < TEAM_MIN {
        dst.copy_from_slice(src);
        return;
    }
    let out = SharedDst(dst.as_mut_ptr());
    threads::pool().for_each_chunk(n, |r| {
        let out = &out;
        // SAFETY: `for_each_chunk` hands each call a disjoint range inside
        // `0..n`, `src` and `dst` both hold `n` bytes and do not overlap (one
        // is the ring's pinned slot, the other a file mapping or a host
        // buffer), and `dst` stays borrowed until every chunk has returned.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(r.start), out.0.add(r.start), r.len());
        }
    });
}

/// The destination of [`memcpy_pool`], shared with the pool's workers, each
/// of which writes only its own chunk.
struct SharedDst(*mut u8);

// SAFETY: the pointer is written only through disjoint chunk ranges, each by
// one worker, while the `&mut` it came from is held by `memcpy_pool`.
unsafe impl Sync for SharedDst {}

/// `data` as its bytes, for a copy that treats them as raw: every element of
/// a `DeviceCopy` type is initialized plain bytes (a byte-for-byte copy is
/// the trait's own identity), and u8's alignment divides every alignment.
pub(crate) fn bytes_of<T: DeviceCopy>(data: &[T]) -> &[u8] {
    // SAFETY: the length covers exactly `data`'s initialized bytes and
    // reading bytes through a lower alignment is always sound.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}
