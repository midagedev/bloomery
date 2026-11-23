//! Enqueue-only fills and copies on a [`DeviceBuffer`]: no allocation lease and no completion event.
//!
//! cuda-core's own `zero_async` and `copy_from_device_async` retain a lease on the buffer until an event
//! recorded behind the work has passed, and one thread of cuda-core polls those events every millisecond.
//! Inside a stream capture the event is recorded into the graph: the poll of it invalidates the capture
//! (`CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` at `end_capture`) when the capture body outlives the first
//! poll, and a captured event never completes, so the lease would never be released. A chain that is
//! captured enqueues through these methods; the buffers they touch outlive the work by the caller's
//! ordering, as every raw launch's do.

use cuda_core::{CudaStream, DeviceBuffer, DriverError, memory, sys};

/// The raw twins of `DeviceBuffer::zero_async` and `DeviceBuffer::copy_from_device_async`.
pub trait DeviceBufferRaw<T> {
    /// Zeroes every byte of the buffer, enqueued on `stream`, with no lease on it.
    fn zero_raw(&mut self, stream: &CudaStream) -> Result<(), DriverError>;

    /// Copies `other` into `self` device-to-device, enqueued on `stream`, with no lease on either.
    /// Panics if the lengths differ; refuses overlapping ranges and a buffer of another context.
    fn copy_from_device_raw(
        &mut self,
        other: &DeviceBuffer<T>,
        stream: &CudaStream,
    ) -> Result<(), DriverError>;
}

impl<T> DeviceBufferRaw<T> for DeviceBuffer<T> {
    fn zero_raw(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        let bytes = self.num_bytes();
        if bytes == 0 {
            return Ok(());
        }
        if self.context().as_ref() != stream.context().as_ref() {
            return Err(DriverError(sys::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT));
        }
        stream.context().bind_to_thread()?;
        // SAFETY: the buffer's `bytes` bytes at its device pointer, in the stream's context.
        unsafe { memory::memset_d8_async(self.cu_deviceptr(), 0, bytes, stream.cu_stream()) }
    }

    fn copy_from_device_raw(
        &mut self,
        other: &DeviceBuffer<T>,
        stream: &CudaStream,
    ) -> Result<(), DriverError> {
        assert_eq!(
            self.len(),
            other.len(),
            "device-to-device copy length mismatch"
        );
        let bytes = self.num_bytes();
        if bytes == 0 {
            return Ok(());
        }
        if self.context().as_ref() != stream.context().as_ref()
            || other.context().as_ref() != self.context().as_ref()
        {
            return Err(DriverError(sys::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT));
        }
        let (dst, src) = (self.cu_deviceptr(), other.cu_deviceptr());
        if dst == src {
            return Ok(());
        }
        if dst < src + bytes as u64 && src < dst + bytes as u64 {
            return Err(DriverError(sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE));
        }
        stream.context().bind_to_thread()?;
        // SAFETY: two disjoint `bytes`-byte ranges of buffers of the stream's context.
        unsafe { memory::memcpy_dtod_async(dst, src, bytes, stream.cu_stream()) }
    }
}
