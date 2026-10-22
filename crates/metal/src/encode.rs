//! The encoding of a kernel: the bindings, the dispatch, the end.
//!
//! A token group is the dispatches between barriers.
//! [`ComputeEncoder::memory_barrier`] is that boundary, not a name for the
//! group. The serial encoder's dispatches run in encode order, so a reader
//! after a writer on that encoder needs no barrier. The concurrent encoder's
//! dispatches overlap until `memory_barrier`.
//!
//! The command buffer the encoder is made on retains every object bound to
//! it, so a buffer dropped by the host after its dispatch was encoded still
//! lives until the work is done; a command buffer made with unretained
//! references would need that line rewritten.

use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr::NonNull;

use objc2::ffi::NSUInteger;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLDispatchType, MTLSize,
};

use bloomery_device::{Bytes, DeviceError, Elem, Window, WindowMut};

use crate::buffer::MetalRaw;
use crate::device::MetalDevice;
use crate::library::Pipeline;

/// The buffer argument slots a compute function has.
pub const MAX_BINDINGS: usize = 31;

/// The most bytes `setBytes` takes.
pub const MAX_BYTES_BINDING: usize = 4096;

/// A kernel's binding: a buffer at a byte offset, or a value. Made from a
/// window, a whole buffer or a value, so its offset is inside its buffer and
/// aligned for the element it was made for.
#[derive(Clone, Copy)]
pub struct Binding<'w> {
    kind: BindingKind<'w>,
}

#[derive(Clone, Copy)]
enum BindingKind<'w> {
    Buffer {
        buffer: &'w ProtocolObject<dyn MTLBuffer>,
        offset_bytes: Bytes,
    },
    Bytes(&'w [u8]),
}

impl<'w> Binding<'w> {
    /// The whole of a buffer.
    #[must_use]
    pub fn whole(buffer: &'w ProtocolObject<dyn MTLBuffer>) -> Self {
        Self {
            kind: BindingKind::Buffer {
                buffer,
                offset_bytes: Bytes::ZERO,
            },
        }
    }

    /// A read window: the buffer, at the window's byte offset.
    pub fn of<T: Elem>(window: &'w Window<'_, T, MetalDevice>) -> Result<Self, DeviceError> {
        Self::at::<T>(window.raw(), window.byte_range()?.start)
    }

    /// A mutable window: the buffer, at the window's byte offset.
    pub fn of_mut<T: Elem>(window: &'w WindowMut<'_, T, MetalDevice>) -> Result<Self, DeviceError> {
        Self::at::<T>(window.raw(), window.byte_range()?.start)
    }

    /// The buffer of `raw`, `offset` bytes in, refused by name when the
    /// offset is past the buffer or misaligned for `T`.
    fn at<T: Elem>(raw: &'w MetalRaw, offset: Bytes) -> Result<Self, DeviceError> {
        // SAFETY: `raw` is lent by a window of a live buffer, and this
        // binding borrows the window for `'w`.
        let buffer = unsafe { raw.buffer() };
        let length = buffer.length();
        if offset.get() > length {
            return Err(DeviceError::Range {
                what: "binding",
                detail: format!(
                    "the offset of {} bytes is past the buffer's {length} bytes",
                    offset.get()
                ),
            });
        }
        let align = core::mem::align_of::<T>();
        if !offset.get().is_multiple_of(align) {
            return Err(DeviceError::Range {
                what: "binding",
                detail: format!(
                    "the offset of {} bytes is not a multiple of the element's alignment {align}",
                    offset.get()
                ),
            });
        }
        Ok(Self {
            kind: BindingKind::Buffer {
                buffer,
                offset_bytes: offset,
            },
        })
    }

    /// A value, as its bytes: one to [`MAX_BYTES_BINDING`] of them.
    pub fn bytes(data: &'w [u8]) -> Result<Self, DeviceError> {
        if data.is_empty() || data.len() > MAX_BYTES_BINDING {
            return Err(DeviceError::Range {
                what: "binding",
                detail: format!(
                    "a value of {} bytes is not between 1 and {MAX_BYTES_BINDING}",
                    data.len()
                ),
            });
        }
        Ok(Self {
            kind: BindingKind::Bytes(data),
        })
    }
}

/// The compute encoder of one command buffer. It ends with
/// [`ComputeEncoder::end`], or when it is dropped; the recording it belongs to
/// cannot be submitted while it lives.
pub struct ComputeEncoder<'r> {
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// The group size the current pipeline allows, once one is set.
    max_threads_per_group: Option<usize>,
    /// What this encoder has encoded so far.
    count: EncodeCount,
    /// A dispatch was encoded since the last barrier.
    group_open: bool,
    _recording: PhantomData<&'r ()>,
}

/// What one encoder encoded: its groups (runs of dispatches between
/// barriers, empty runs not counted), its dispatches and its barriers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncodeCount {
    pub groups: usize,
    pub dispatches: usize,
    pub barriers: usize,
}

impl ComputeEncoder<'_> {
    /// The serial compute encoder of a command buffer.
    pub(crate) fn new(
        buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<ComputeEncoder<'_>, DeviceError> {
        Self::with_encoder(buffer, |buffer| buffer.computeCommandEncoder())
    }

    /// A compute encoder of a command buffer whose dispatches may overlap.
    pub(crate) fn concurrent(
        buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<ComputeEncoder<'_>, DeviceError> {
        Self::with_encoder(buffer, |buffer| {
            buffer.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
        })
    }

    /// An encoder from `make`, refused by name when the command buffer makes none.
    fn with_encoder<'a>(
        buffer: &'a ProtocolObject<dyn MTLCommandBuffer>,
        make: impl FnOnce(
            &ProtocolObject<dyn MTLCommandBuffer>,
        ) -> Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    ) -> Result<ComputeEncoder<'a>, DeviceError> {
        let encoder = make(buffer).ok_or_else(|| {
            DeviceError::backend(
                "compute encoder",
                "the command buffer made no compute encoder".to_string(),
            )
        })?;
        Ok(ComputeEncoder {
            encoder,
            max_threads_per_group: None,
            count: EncodeCount::default(),
            group_open: false,
            _recording: PhantomData,
        })
    }

    /// The pipeline the next dispatches run.
    pub fn set_pipeline(&mut self, pipeline: &Pipeline) {
        self.encoder.setComputePipelineState(pipeline.metal());
        self.max_threads_per_group = Some(pipeline.max_threads_per_group());
    }

    /// The binding at `index`; an index past [`MAX_BINDINGS`] is refused by
    /// name.
    pub fn set_binding(&mut self, index: usize, binding: Binding<'_>) -> Result<(), DeviceError> {
        if index >= MAX_BINDINGS {
            return Err(DeviceError::Range {
                what: "binding",
                detail: format!("index {index} is not below the {MAX_BINDINGS} slots"),
            });
        }
        let slot = to_ns(index);
        match binding.kind {
            BindingKind::Buffer {
                buffer,
                offset_bytes,
            } => {
                // SAFETY: the offset was checked against the buffer's length
                // and the element's alignment when the binding was made, and
                // the index is below the slot count (checked above).
                unsafe {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(buffer),
                        to_ns(offset_bytes.get()),
                        slot,
                    );
                }
            }
            BindingKind::Bytes(data) => {
                let ptr = NonNull::from(data).cast::<c_void>();
                // SAFETY: the encoder copies the bytes into its own argument
                // buffer before the call returns, `data` outlives the call,
                // and its length is 1..=4096 (checked at construction).
                unsafe {
                    self.encoder
                        .setBytes_length_atIndex(ptr, to_ns(data.len()), slot);
                }
            }
        }
        Ok(())
    }

    /// The dispatch: `groups_x` by `groups_y` threadgroups, each of
    /// `threads_per_group` threads. The group's height is one, so the grid's
    /// height is `groups_y` and its width is `groups_x` groups. A pipeline not
    /// set, an empty grid, a group larger than the pipeline allows, or a width
    /// that wraps is refused by name.
    pub fn dispatch_groups(
        &mut self,
        groups_x: usize,
        groups_y: usize,
        threads_per_group: usize,
    ) -> Result<(), DeviceError> {
        self.check_group(groups_x, groups_y, threads_per_group)?;
        let width = groups_x
            .checked_mul(threads_per_group)
            .ok_or_else(|| DeviceError::Range {
                what: "dispatch",
                detail: format!("{groups_x} groups of {threads_per_group} threads wrap"),
            })?;
        self.submit(width, groups_y, threads_per_group);
        self.count.dispatches += 1;
        if !self.group_open {
            self.group_open = true;
            self.count.groups += 1;
        }
        Ok(())
    }

    /// What this encoder has encoded so far.
    pub fn count(&self) -> EncodeCount {
        self.count
    }

    /// A pipeline is set, the grid is not empty, and the group is within the
    /// pipeline's maximum. The width's wrap is checked by the caller.
    fn check_group(
        &self,
        groups_x: usize,
        groups_y: usize,
        threads_per_group: usize,
    ) -> Result<(), DeviceError> {
        let Some(max) = self.max_threads_per_group else {
            return Err(DeviceError::Range {
                what: "dispatch",
                detail: "no pipeline is set".to_string(),
            });
        };
        if groups_x == 0 || groups_y == 0 || threads_per_group == 0 {
            return Err(DeviceError::Range {
                what: "dispatch",
                detail: format!(
                    "{groups_x} by {groups_y} groups of {threads_per_group} is refused"
                ),
            });
        }
        if threads_per_group > max {
            return Err(DeviceError::Range {
                what: "dispatch",
                detail: format!(
                    "a group of {threads_per_group} threads passes the pipeline's {max}"
                ),
            });
        }
        Ok(())
    }

    /// Record the grid. `width` is the threads across, `height` the
    /// threadgroups down, and every group holds `threads_per_group` threads.
    fn submit(&self, width: usize, height: usize, threads_per_group: usize) {
        self.encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: to_ns(width),
                height: to_ns(height),
                depth: 1,
            },
            MTLSize {
                width: to_ns(threads_per_group),
                height: 1,
                depth: 1,
            },
        );
    }

    /// A barrier on the encoder's buffers: the dispatches before it are one
    /// group, and on a concurrent encoder the dispatches after it see what
    /// that group wrote. On a serial encoder Metal ignores the barrier; the
    /// call is still encoded. The barrier is the group's boundary, not a
    /// name for the group.
    pub fn memory_barrier(&mut self) {
        self.encoder
            .memoryBarrierWithScope(MTLBarrierScope::Buffers);
        self.count.barriers += 1;
        self.group_open = false;
    }

    /// The encoder is done; the recording may be submitted.
    pub fn end(self) {
        // `Drop` ends the encoding.
    }
}

/// Bindings filled from slot 0. The nth call is the kernel's `[[buffer(n)]]`.
pub struct Bindings<'a, 'b> {
    encoder: &'a mut ComputeEncoder<'b>,
    next: usize,
}

impl<'a, 'b> Bindings<'a, 'b> {
    /// Bindings on `encoder`, starting at slot 0.
    pub fn new(encoder: &'a mut ComputeEncoder<'b>) -> Self {
        Self { encoder, next: 0 }
    }

    /// The next slot.
    pub fn set(&mut self, binding: Binding<'_>) -> Result<(), DeviceError> {
        let index = self.next;
        self.next += 1;
        self.encoder.set_binding(index, binding)
    }

    /// [`ComputeEncoder::dispatch_groups`] on the encoder these bindings fill.
    pub fn dispatch_groups(
        &mut self,
        groups_x: usize,
        groups_y: usize,
        threads_per_group: usize,
    ) -> Result<(), DeviceError> {
        self.encoder
            .dispatch_groups(groups_x, groups_y, threads_per_group)
    }
}

impl Drop for ComputeEncoder<'_> {
    fn drop(&mut self) {
        self.encoder.endEncoding();
    }
}

/// A count as Metal's unsigned integer: both are the pointer width on the
/// 64-bit targets Metal runs on.
fn to_ns(n: usize) -> NSUInteger {
    NSUInteger::try_from(n).expect("NSUInteger holds a usize on every Metal target")
}
