//! Device-resident buffers with the shapes the kernels' launch contracts
//! speak in, allocated once and reused across steps (docs/gpu-design.md
//! decision 4: nothing is allocated inside `step`, so a captured graph can
//! replay against fixed addresses).

use crate::GpuError;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, sys};
use std::mem::{ManuallyDrop, size_of};
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

/// One device allocation cut into `N` consecutive parts, each of which the
/// launches read and write as a buffer of its own, while a launch handed the
/// whole writes every part at once — a gemv over row-concatenated weights
/// fills each projection's output in place. The parts are [`window`]s into
/// the whole, which this type owns: they live exactly as long as it does
/// and free nothing.
pub struct PartedBuffer<T, const N: usize> {
    parts: [ManuallyDrop<DeviceBuffer<T>>; N],
    whole: DeviceBuffer<T>,
}

impl<T: DeviceCopy, const N: usize> PartedBuffer<T, N> {
    /// A zero-filled allocation of `lens` summed, part `i` the `lens[i]`
    /// elements after the parts before it. Load-time only.
    pub fn zeroed(stream: &CudaStream, lens: [usize; N]) -> Result<Self, GpuError> {
        let refuse = || GpuError::shape("PartedBuffer::zeroed", format!("parts {lens:?}"));
        // Each part's byte offset, and the total in elements.
        let mut offs = [0u64; N];
        let mut total = 0usize;
        for (off, &len) in offs.iter_mut().zip(&lens) {
            let bytes = total.checked_mul(size_of::<T>()).ok_or_else(refuse)?;
            *off = u64::try_from(bytes).map_err(|_| refuse())?;
            total = total.checked_add(len).ok_or_else(refuse)?;
        }
        let whole = DeviceBuffer::<T>::zeroed(stream, total)?;
        let base = whole.cu_deviceptr();
        let mut i = 0;
        let parts = lens.map(|len| {
            let off = offs[i];
            i += 1;
            // SAFETY: the part spans `len` elements from byte `off` of `whole`,
            // inside its `total` (the sum of `lens`); `whole` is a `cuMemAlloc`
            // allocation aligned for `T`, owned by `self` and freed after the
            // windows, which free nothing.
            unsafe { window::<T>(base + off, len, whole.context()) }
        });
        Ok(PartedBuffer { parts, whole })
    }
}

impl<T, const N: usize> PartedBuffer<T, N> {
    /// The whole allocation, every part in order.
    pub fn whole(&self) -> &DeviceBuffer<T> {
        &self.whole
    }

    /// The whole allocation, for a launch that writes every part.
    pub fn whole_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.whole
    }

    /// Part `i` (`i < N`).
    pub fn part(&self, i: usize) -> &DeviceBuffer<T> {
        &self.parts[i]
    }

    /// Part `i` (`i < N`), for a launch that writes it alone.
    pub fn part_mut(&mut self, i: usize) -> &mut DeviceBuffer<T> {
        &mut self.parts[i]
    }
}

impl<T, const N: usize> Drop for PartedBuffer<T, N> {
    fn drop(&mut self) {
        for part in &mut self.parts {
            // SAFETY: each window is taken once, here, and never read again;
            // its raw parts are dropped (the context handle with them) and no
            // memory is freed — `whole` frees the allocation after this.
            let buf = unsafe { ManuallyDrop::take(part) };
            drop(buf.into_raw_parts());
        }
    }
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

    /// A `rows × cols` tensor over memory it does not own, for a launcher
    /// that takes a tensor: [`window`] with a shape. Give it back with
    /// [`DeviceTensor::release`].
    ///
    /// # Safety
    ///
    /// [`window`]'s contract, for the `rows * cols` `T` at `ptr`.
    pub unsafe fn window(
        ptr: sys::CUdeviceptr,
        rows: usize,
        cols: usize,
        ctx: &Arc<CudaContext>,
    ) -> ManuallyDrop<DeviceTensor<T>> {
        // SAFETY: the span is the caller's, under `window`'s contract.
        let buf = unsafe { window::<T>(ptr, rows * cols, ctx) };
        ManuallyDrop::new(DeviceTensor {
            buf: ManuallyDrop::into_inner(buf),
            rows,
            cols,
        })
    }

    /// Give back a [`DeviceTensor::window`]: its handle on the context is
    /// dropped and no memory is freed.
    pub fn release(t: ManuallyDrop<DeviceTensor<T>>) {
        drop(ManuallyDrop::into_inner(t).buf.into_raw_parts());
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

/// The most columns a per-slot [`Q8Act`] takes ([`Q8Act::with_slots`]):
/// eight tokens of eight routed slots, the down input of a prefill pass.
pub(crate) const Q8ACT_MAX_SLOTS: usize = 64;

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
        Q8Act::alloc("Q8Act::with_k", stream, m, k)
    }

    /// Allocate scratch for `cols` (1..=[`Q8ACT_MAX_SLOTS`]) columns of `k`
    /// values, one per expert slot: the input of a `_sel` down over several
    /// tokens' slots, which reads column `s` for slot `s`, and of the
    /// quantizer that writes it. The multi-column gemvs refuse more than
    /// eight columns by their own check. Load-time only.
    pub(crate) fn with_slots(stream: &CudaStream, cols: usize, k: usize) -> Result<Self, GpuError> {
        if !(1..=Q8ACT_MAX_SLOTS).contains(&cols) {
            return Err(GpuError::shape(
                "Q8Act::with_slots",
                format!("1 <= cols <= {Q8ACT_MAX_SLOTS}, got {cols}"),
            ));
        }
        Q8Act::alloc("Q8Act::with_slots", stream, cols, k)
    }

    /// The planes for `m` columns of `k` values, `k` checked here.
    fn alloc(
        what: &'static str,
        stream: &CudaStream,
        m: usize,
        k: usize,
    ) -> Result<Self, GpuError> {
        if !k.is_multiple_of(256) || !(256..=Q8ACT_MAX_K).contains(&k) {
            return Err(GpuError::shape(
                what,
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
