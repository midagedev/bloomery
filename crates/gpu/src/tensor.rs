//! Device-resident buffers with the shapes the kernels' launch contracts
//! speak in, allocated once and reused across steps (docs/gpu-design.md
//! decision 4: nothing is allocated inside `step`, so a captured graph can
//! replay against fixed addresses).

use crate::GpuError;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, sys};
use std::mem::ManuallyDrop;
use std::sync::Arc;

/// A non-owning window of `len` `T` at device address `ptr` in `ctx`. The
/// launches read it exactly as they read a buffer of their own;
/// `ManuallyDrop` keeps it from ever freeing memory it does not own.
///
/// # Safety
///
/// - `ptr .. ptr + len * size_of::<T>()` must be memory the device reaches in
///   `ctx` — a `cuMemAlloc` allocation or a host-mapped one — aligned for `T`.
/// - That memory must outlive the window and stay in place: a captured graph
///   bakes the address in.
pub unsafe fn window<T>(
    ptr: sys::CUdeviceptr,
    len: usize,
    ctx: &Arc<CudaContext>,
) -> ManuallyDrop<DeviceBuffer<T>> {
    // SAFETY: the range is the caller's contract. `from_raw_parts` asks for a
    // `cuMemAlloc` pointer because its drop frees one; a window is never
    // dropped, so the only uses left are the address and the length.
    ManuallyDrop::new(unsafe { DeviceBuffer::from_raw_parts(ptr, len, ctx.clone()) })
}

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
            return Err(GpuError::shape(
                "DeviceTensor::upload",
                format!("data.len() {} != rows*cols = {}*{}", data.len(), rows, cols),
            ));
        }
        let buf = DeviceBuffer::from_host(stream, data)?;
        Ok(DeviceTensor { buf, rows, cols })
    }

    /// Zero-filled tensor (scratch / KV). Load-time only, like `upload`.
    pub fn zeroed(stream: &CudaStream, rows: usize, cols: usize) -> Result<Self, GpuError> {
        let buf = DeviceBuffer::zeroed(stream, rows * cols)?;
        Ok(DeviceTensor { buf, rows, cols })
    }

    /// Row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Elements per row.
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

/// The largest K a [`Q8Act`] takes: the widest K-quant input of the models
/// this engine runs — DeepSeek-V4.1's `hc_{attn,ffn}_fn` read all four
/// residual streams, 4 × 5120 values.
pub(crate) const Q8ACT_MAX_K: usize = 20_480;

/// q8_1 activation scratch for up to `m` columns of `k` values each. One
/// set per distinct input site: sites that read the same activation
/// (gate·up, q·kv_a) share one (decision 2, as the CPU engine does).
///
/// `k` is a multiple of 256 with 256 <= k <= [`Q8ACT_MAX_K`]. The cap is a
/// sanity bound, not a layout limit: no kernel stages a K-sized array, and
/// every index the layout computes stays far inside u32 at the cap. With
/// n_sb = k/256 super-blocks per row, buffer sizes in elements per column
/// are: q3 `64 * ceil(n_sb/2)` u64, q4 `256 * ceil(n_sb/4)` u32,
/// q6 `128 * ceil(n_sb/2)` u32, s8 `8 * n_sb` i32, d8 `2 * n_sb` f32.
/// The q3/q6 buffers are permuted per 2-super-block group and q4 per
/// 4-super-block group; a partial final group (n_sb odd, or n_sb not a
/// multiple of 4) leaves the group's tail slots allocated but never written
/// or read — hence the ceil. At k = 2048 (n_sb = 8) these are the stage-0
/// constants: 256 u64 / 512 / 512 / 64 / 16 per column.
pub struct Q8Act {
    pub(crate) q3: DeviceBuffer<u64>,
    pub(crate) q4: DeviceBuffer<u32>,
    pub(crate) q6: DeviceBuffer<u32>,
    pub(crate) s8: DeviceBuffer<i32>,
    pub(crate) d8: DeviceBuffer<f32>,
    m: usize,
    k: usize,
}

impl Q8Act {
    /// Allocate scratch for `m` (1..=8) columns of k = 2048 values — the
    /// stage-0 geometry. Load-time only.
    pub fn new(stream: &CudaStream, m: usize) -> Result<Self, GpuError> {
        Q8Act::with_k(stream, m, 2048)
    }

    /// Allocate scratch for `m` (1..=8) columns of `k` values. Load-time
    /// only.
    pub fn with_k(stream: &CudaStream, m: usize, k: usize) -> Result<Self, GpuError> {
        if !(1..=8).contains(&m) {
            return Err(GpuError::shape(
                "Q8Act::with_k",
                format!("1 <= m <= 8, got {m}"),
            ));
        }
        if !k.is_multiple_of(256) || !(256..=Q8ACT_MAX_K).contains(&k) {
            return Err(GpuError::shape(
                "Q8Act::with_k",
                format!("k must be a multiple of 256 in 256..={Q8ACT_MAX_K}, got {k}"),
            ));
        }
        let n_sb = k / 256;
        Ok(Q8Act {
            q3: DeviceBuffer::zeroed(stream, m * 64 * n_sb.div_ceil(2))?,
            q4: DeviceBuffer::zeroed(stream, m * 256 * n_sb.div_ceil(4))?,
            q6: DeviceBuffer::zeroed(stream, m * 128 * n_sb.div_ceil(2))?,
            s8: DeviceBuffer::zeroed(stream, m * 8 * n_sb)?,
            d8: DeviceBuffer::zeroed(stream, m * 2 * n_sb)?,
            m,
            k,
        })
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub(crate) fn k(&self) -> usize {
        self.k
    }

    /// Super-blocks per row (k/256) — the row geometry every K-quant gemv
    /// of this engine launches with.
    #[must_use]
    pub fn n_sb(&self) -> usize {
        self.k / 256
    }

    /// The codes `cores::q3k_row_dot` reads: `64 * ceil(n_sb/2)` u64 per
    /// column in the quantizer's pair permutation, for a Q3_K kernel outside
    /// this crate.
    #[must_use]
    pub fn q3(&self) -> &DeviceBuffer<u64> {
        &self.q3
    }

    /// The block scales that go with [`Q8Act::q3`]: `2 * n_sb` f32 per
    /// column.
    #[must_use]
    pub fn d8(&self) -> &DeviceBuffer<f32> {
        &self.d8
    }
}
