//! The GEMM's q8_1 activations, the quantizer launch that fills them, and
//! the launch that fills them from a `Q8Act`'s q3 plane and scales alone.

use super::{GEMM_MAX_SLOTS, GemmKernels};
use crate::cores::{q3_slot, q4_slot, q6_slot};
use crate::fault::FaultSink;
use crate::tensor::{Q8ACT_MAX_K, Q8Act};
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, warp};

/// q8_1 activations for up to `cols` columns of `k` values each, in the
/// layout the quantizer writes for a `Q8Act` (the q3/q4/q6 permutations, the
/// 32-value code sums and the 128-value scales), for more columns than a
/// `Q8Act` holds. The GEMM reads the q6 permutation, `s8` and `d8`.
pub struct GemmAct {
    pub(super) q3: DeviceBuffer<u64>,
    pub(super) q4: DeviceBuffer<u32>,
    pub(super) q6: DeviceBuffer<u32>,
    pub(super) s8: DeviceBuffer<i32>,
    pub(super) d8: DeviceBuffer<f32>,
    pub(super) cols: usize,
    pub(super) k: usize,
}

/// A [`GemmAct`]'s five planes borrowed for writing, for a quantizer outside
/// this module that writes the layout [`GemmAct`] documents.
pub(crate) struct GemmActPlanesMut<'a> {
    /// The codes in the q3 pair permutation: `64 * ceil(n_sb/2)` u64 per column.
    pub(crate) q3: &'a mut DeviceBuffer<u64>,
    /// The codes in the q4 permutation: `256 * ceil(n_sb/4)` u32 per column.
    pub(crate) q4: &'a mut DeviceBuffer<u32>,
    /// The codes in the q6 permutation the GEMM stages: `128 * ceil(n_sb/2)` u32 per column.
    pub(crate) q6: &'a mut DeviceBuffer<u32>,
    /// The 32-value code sums: `8 * n_sb` i32 per column.
    pub(crate) s8: &'a mut DeviceBuffer<i32>,
    /// The 128-value block scales: `2 * n_sb` f32 per column.
    pub(crate) d8: &'a mut DeviceBuffer<f32>,
}

impl GemmAct {
    /// Scratch for `cols` (1..=[`GEMM_MAX_SLOTS`]) columns of `k` values, `k`
    /// a multiple of 256 up to the quantizer's cap. Load-time only.
    pub fn new(stream: &CudaStream, cols: usize, k: usize) -> Result<GemmAct, GpuError> {
        let what = "GemmAct::new";
        if !(1..=GEMM_MAX_SLOTS).contains(&cols) {
            return Err(GpuError::shape(
                what,
                format!("1 <= cols <= {GEMM_MAX_SLOTS}, got {cols}"),
            ));
        }
        if !k.is_multiple_of(256) || !(256..=Q8ACT_MAX_K).contains(&k) {
            return Err(GpuError::shape(
                what,
                format!("k must be a multiple of 256 in 256..={Q8ACT_MAX_K}, got {k}"),
            ));
        }
        let n_sb = k / 256;
        Ok(GemmAct {
            q3: DeviceBuffer::zeroed(stream, cols * 64 * n_sb.div_ceil(2))?,
            q4: DeviceBuffer::zeroed(stream, cols * 256 * n_sb.div_ceil(4))?,
            q6: DeviceBuffer::zeroed(stream, cols * 128 * n_sb.div_ceil(2))?,
            s8: DeviceBuffer::zeroed(stream, cols * 8 * n_sb)?,
            d8: DeviceBuffer::zeroed(stream, cols * 2 * n_sb)?,
            cols,
            k,
        })
    }

    /// Columns this scratch holds.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Values per column.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Super-blocks per column (`k / 256`).
    #[must_use]
    pub fn n_sb(&self) -> usize {
        self.k / 256
    }

    /// The 128-value block scales: `2 * n_sb` f32 per column.
    #[must_use]
    pub fn d8(&self) -> &DeviceBuffer<f32> {
        &self.d8
    }

    /// The 32-value code sums: `8 * n_sb` i32 per column.
    #[must_use]
    pub fn s8(&self) -> &DeviceBuffer<i32> {
        &self.s8
    }

    /// The codes in the q3 pair permutation: `64 * ceil(n_sb/2)` u64 per
    /// column.
    #[must_use]
    pub fn q3(&self) -> &DeviceBuffer<u64> {
        &self.q3
    }

    /// The codes in the q4 permutation: `256 * ceil(n_sb/4)` u32 per column.
    #[must_use]
    pub fn q4(&self) -> &DeviceBuffer<u32> {
        &self.q4
    }

    /// The codes in the q6 permutation the GEMM stages: `128 * ceil(n_sb/2)`
    /// u32 per column.
    #[must_use]
    pub fn q6(&self) -> &DeviceBuffer<u32> {
        &self.q6
    }

    /// The five planes, writable, for a quantizer outside this module that
    /// writes the layout above (`gated_quant`).
    pub(crate) fn planes_mut(&mut self) -> GemmActPlanesMut<'_> {
        GemmActPlanesMut {
            q3: &mut self.q3,
            q4: &mut self.q4,
            q6: &mut self.q6,
            s8: &mut self.s8,
            d8: &mut self.d8,
        }
    }

    /// Device bytes of the five planes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.q3.num_bytes()
            + self.q4.num_bytes()
            + self.q6.num_bytes()
            + self.s8.num_bytes()
            + self.d8.num_bytes()
    }
}

impl Gpu {
    /// Enqueue the q8_1 quantization of the first `n_cols` columns of `x`
    /// (`act.k()` f32 each) into `act`, a block holding a non-finite value
    /// stored refused and raised into `fault` — `q3k_quantize_q8_1`, the kernel every gemv's
    /// activations come from, so column `c` holds the bytes a `Q8Act`
    /// quantized from the same values holds. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_quantize_gemm(
        &self,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        self.enqueue_quantize_gemm_(x, n_cols, act, false, fault)
    }

    /// [`Gpu::enqueue_quantize_gemm`] through `q3k_quantize_q8_1_relu_sqr`:
    /// the up rows' buffer quantized for the down GEMM with the squared
    /// ReLU applied inside the load, no elem pass over an activation buffer.
    pub fn enqueue_quantize_gemm_relu_sqr(
        &self,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        self.enqueue_quantize_gemm_(x, n_cols, act, true, fault)
    }

    fn enqueue_quantize_gemm_(
        &self,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct,
        relu_sqr: bool,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "enqueue_quantize_gemm";
        if n_cols == 0 || n_cols > act.cols {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= act.cols() = {}, got {n_cols}", act.cols),
            ));
        }
        if x.len() < n_cols * act.k {
            return Err(GpuError::shape(
                what,
                format!("x.len() {} < n_cols*k = {n_cols}*{}", x.len(), act.k),
            ));
        }
        let n_sb = act.n_sb();
        let grid = launch_u32(what, "grid", n_cols * n_sb * 2)?;
        let m = launch_u32(what, "n_cols", n_cols)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        if relu_sqr {
            let prep = self
                .module
                .prepare_q3k_quantize_q8_1_relu_sqr(LaunchConfig1D::new(grid, 32, 0))?;
            self.module.q3k_quantize_q8_1_relu_sqr(
                &self.stream,
                &prep,
                x,
                0,
                m,
                n_sb,
                n_sb.div_ceil(2),
                n_sb.div_ceil(4),
                &mut act.q3,
                &mut act.q4,
                &mut act.q6,
                &mut act.s8,
                &mut act.d8,
                fault,
            )?;
        } else {
            let prep = self
                .module
                .prepare_q3k_quantize_q8_1(LaunchConfig1D::new(grid, 32, 0))?;
            self.module.q3k_quantize_q8_1(
                &self.stream,
                &prep,
                x,
                0,
                m,
                n_sb,
                n_sb.div_ceil(2),
                n_sb.div_ceil(4),
                &mut act.q3,
                &mut act.q4,
                &mut act.q6,
                &mut act.s8,
                &mut act.d8,
                fault,
            )?;
        }
        Ok(())
    }
}

impl GemmKernels {
    /// Enqueue `act`'s five planes over the first `n_cols` columns from the
    /// q3 plane and block scales of `src` alone (`gemm_act_from_q3`, declared
    /// in `kernels.rs`): column `c` then holds the bytes
    /// [`Gpu::enqueue_quantize_gemm`] writes from the values `src`'s column
    /// `c` was quantized from — the GEMM's activations where the producer
    /// handed on only those two planes (a tier node's block). A column count
    /// past either side, or a `k` the two do not share, is refused by name.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_act_from_q8(
        &self,
        stream: &CudaStream,
        src: &Q8Act,
        n_cols: usize,
        act: &mut GemmAct,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_act_from_q8";
        if n_cols == 0 || n_cols > src.m() || n_cols > act.cols || src.k() != act.k {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_cols} columns from a Q8Act of {} columns of {} values into a GemmAct of \
                     {} columns of {}: 1 to both column counts, one k",
                    src.m(),
                    src.k(),
                    act.cols,
                    act.k
                ),
            ));
        }
        let n_sb = act.n_sb();
        let grid = launch_u32(what, "grid", n_cols * n_sb * 2)?;
        let m = launch_u32(what, "n_cols", n_cols)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_gemm_act_from_q3(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.gemm_act_from_q3(
            stream,
            &prep,
            &src.q3,
            &src.d8,
            m,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
        )?;
        Ok(())
    }
}

/// One warp's 128-value block `b` of column `col` of `gemm_act_from_q3`:
/// lane `l` reads the column's value-order word `v4 = 32·b + l` from the q3
/// plane where `q8_1_quant_vals` put it — the u64 pair at `q3_slot(v4)`, its
/// low half when bit 3 of `v4` is clear and its high half when set — and
/// stores what `q8_1_quant_vals` stores for that word: the pair (by the
/// bit-3-clear lanes, as there), the word in the q4 and q6 permutations, the
/// signed sum of its four codes folded over its 32-value group by the same
/// butterfly, and the block's scale.
///
/// # Safety
///
/// All 32 lanes of one warp call it with one `(col, b)`, `lane` the calling
/// lane's id, and no other warp of the grid with that pair; `b < 2 · n_sb`;
/// `half_it` and `quad_it` are `n_sb.div_ceil(2)` and `n_sb.div_ceil(4)`; and
/// every plane holds at least `col + 1` columns of its per-column size —
/// `q3_in` and `q3` `64 · half_it`, `q4` `256 · quad_it`, `q6` `128 ·
/// half_it`, `s8` `8 · n_sb`, `d8_in` and `d8` `2 · n_sb`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(super) unsafe fn act_from_q3_block(
    (q3_in, d8_in): (&[u64], &[f32]),
    [col, b, n_sb, lane]: [usize; 4],
    (half_it, quad_it): (u32, u32),
    q3: &mut DisjointSlice<u64>,
    q4: &mut DisjointSlice<u32>,
    q6: &mut DisjointSlice<u32>,
    s8: &mut DisjointSlice<i32>,
    d8: &mut DisjointSlice<f32>,
) {
    let v4 = (32 * b + lane) as u32;
    let at3 = col * 64 * half_it as usize + q3_slot(v4) as usize;
    // SAFETY: q3_slot maps the column's 64·n_sb words, bit 3 dropped, onto
    // its 64·half_it pair slots, so `at3` is inside column `col` of `q3_in`.
    let pair = unsafe { *q3_in.get_unchecked(at3) };
    let word = (pair >> (32 * ((v4 >> 3) & 1))) as u32;
    let quad = i32::from(word as u8 as i8)
        + i32::from((word >> 8) as u8 as i8)
        + i32::from((word >> 16) as u8 as i8)
        + i32::from((word >> 24) as u8 as i8);
    // SAFETY: q4_slot < 256·quad_it and q6_slot < 128·half_it per column
    // (the quantizer's bijections), so every store is inside column `col`;
    // the bit-3-clear lanes of a block write disjoint pair slots and every
    // lane its own q4 and q6 word.
    unsafe {
        if lane & 8 == 0 {
            *q3.get_unchecked_mut(at3) = pair;
        }
        *q4.get_unchecked_mut(col * 256 * quad_it as usize + q4_slot(v4) as usize) = word;
        *q6.get_unchecked_mut(col * 128 * half_it as usize + q6_slot(v4) as usize) = word;
    }
    // The 32-value group sums: `q8_1_quant_vals`'s butterfly over the lanes'
    // code sums (masks 1, 2, 4); lanes 8k write group 4b + k.
    let mut g = quad;
    g += warp::shuffle_xor(g as u32, 1) as i32;
    g += warp::shuffle_xor(g as u32, 2) as i32;
    g += warp::shuffle_xor(g as u32, 4) as i32;
    if lane & 7 == 0 {
        // SAFETY: group 4b + lane/8 < 8·n_sb, inside column `col` of s8;
        // one lane writes each group.
        unsafe { *s8.get_unchecked_mut(col * 8 * n_sb + 4 * b + (lane >> 3)) = g };
    }
    if lane == 0 {
        let at = col * 2 * n_sb + b;
        // SAFETY: b < 2·n_sb, inside column `col` of `d8_in` and `d8`; lane 0
        // of this block's one warp writes it.
        unsafe { *d8.get_unchecked_mut(at) = *d8_in.get_unchecked(at) };
    }
}
