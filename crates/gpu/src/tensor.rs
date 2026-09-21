//! Device-resident buffers with the shapes the kernels' launch contracts
//! speak in, allocated once and reused across steps (docs/gpu-design.md
//! decision 4: nothing is allocated inside `step`, so a captured graph can
//! replay against fixed addresses).

use crate::GpuError;
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};

/// A 2-D device tensor: `rows * cols` elements of `T`, row-major, uploaded
/// once. The element type is whatever the consuming kernel loads (`u32`
/// words for the K-quant gemvs, `f32` for activations, `u16` for F16 KV).
pub struct DeviceTensor<T> {
    buf: DeviceBuffer<T>,
    rows: usize,
    cols: usize,
}

impl<T: DeviceCopy> DeviceTensor<T> {
    /// Upload `data` (exactly `rows * cols` elements) on `stream`. This is a
    /// load-time operation: `from_host` synchronizes, so it must never run
    /// inside a graph capture.
    pub fn upload(
        stream: &CudaStream,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self, GpuError> {
        if data.len() != rows * cols {
            return Err(format!(
                "DeviceTensor::upload: data.len() {} != rows*cols = {}*{}",
                data.len(),
                rows,
                cols
            )
            .into());
        }
        let buf = DeviceBuffer::from_host(stream, data)?;
        Ok(DeviceTensor { buf, rows, cols })
    }

    /// Zero-filled tensor (scratch / KV). Load-time only, like `upload`.
    pub fn zeroed(stream: &CudaStream, rows: usize, cols: usize) -> Result<Self, GpuError> {
        let buf = DeviceBuffer::zeroed(stream, rows * cols)?;
        Ok(DeviceTensor { buf, rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The flat buffer the generated launchers take (`&DeviceBuffer<T>` for
    /// `&[T]` kernel parameters).
    pub fn buf(&self) -> &DeviceBuffer<T> {
        &self.buf
    }

    /// Writable view for `DisjointSlice<T>` / `&mut [T]` kernel parameters.
    pub fn buf_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buf
    }
}

/// q8_1 activation scratch for up to `m` columns of K=2048 (the stage-0
/// geometry; P1 generalizes K as a launch argument and this struct grows a
/// `k` field then). One set per distinct input site: sites that read the same
/// activation (gate·up, q·kv_a) share one (decision 2, MUL-37 on the CPU).
///
/// Buffer sizes are the quantizer's launch contract verbatim:
/// q3 `m*256` u64, q4/q6 `m*512` u32, s8 `m*64` i32, d8 `m*16` f32.
pub struct Q8Act {
    pub(crate) q3: DeviceBuffer<u64>,
    pub(crate) q4: DeviceBuffer<u32>,
    pub(crate) q6: DeviceBuffer<u32>,
    pub(crate) s8: DeviceBuffer<i32>,
    pub(crate) d8: DeviceBuffer<f32>,
    m: usize,
}

impl Q8Act {
    /// Allocate scratch for `m` (1..=8) columns. Load-time only.
    pub fn new(stream: &CudaStream, m: usize) -> Result<Self, GpuError> {
        if !(1..=8).contains(&m) {
            return Err(format!("Q8Act::new: 1 <= m <= 8, got {m}").into());
        }
        Ok(Q8Act {
            q3: DeviceBuffer::zeroed(stream, m * 256)?,
            q4: DeviceBuffer::zeroed(stream, m * 512)?,
            q6: DeviceBuffer::zeroed(stream, m * 512)?,
            s8: DeviceBuffer::zeroed(stream, m * 64)?,
            d8: DeviceBuffer::zeroed(stream, m * 16)?,
            m,
        })
    }

    pub fn m(&self) -> usize {
        self.m
    }
}
