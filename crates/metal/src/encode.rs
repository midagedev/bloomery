//! The encoding of a kernel: the bindings, the dispatch, the end.
//!
//! The encoder is a serial one: its dispatches run in the order they were
//! encoded, so a dispatch that reads what the one before it wrote needs no
//! barrier. (A concurrent encoder would need `memoryBarrier` between them.)
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
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
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
    _recording: PhantomData<&'r ()>,
}

impl ComputeEncoder<'_> {
    /// The compute encoder of a command buffer.
    pub(crate) fn new(
        buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<ComputeEncoder<'_>, DeviceError> {
        let encoder = buffer.computeCommandEncoder().ok_or_else(|| {
            DeviceError::backend(
                "compute encoder",
                "the command buffer made no compute encoder".to_string(),
            )
        })?;
        Ok(ComputeEncoder {
            encoder,
            max_threads_per_group: None,
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

    /// The dispatch: `threads` threads, `threads_per_group` of them in a
    /// group. A pipeline not set, no threads, no group, or a group larger than
    /// the pipeline allows is refused by name.
    pub fn dispatch(
        &mut self,
        threads: usize,
        threads_per_group: usize,
    ) -> Result<(), DeviceError> {
        let refuse = |detail: String| DeviceError::Range {
            what: "dispatch",
            detail,
        };
        let Some(max) = self.max_threads_per_group else {
            return Err(refuse("no pipeline is set".to_string()));
        };
        if threads == 0 || threads_per_group == 0 {
            return Err(refuse(format!(
                "{threads} threads in groups of {threads_per_group} is refused"
            )));
        }
        if threads_per_group > max {
            return Err(refuse(format!(
                "a group of {threads_per_group} threads passes the pipeline's {max}"
            )));
        }
        self.encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: to_ns(threads),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: to_ns(threads_per_group),
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    /// The encoder is done; the recording may be submitted.
    pub fn end(self) {
        // `Drop` ends the encoding.
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
