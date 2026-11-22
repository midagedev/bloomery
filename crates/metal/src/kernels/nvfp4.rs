//! The NVFP4 mat-vec of Inkling 2's routed experts (`kernels/nvfp4.metal`):
//! a SIMD group a row, eight rows a threadgroup, over the codes as the files
//! store them, the expert's global scale a launch word applied after the dot.
//!
//! Two layouts, one kernel source: the blocks of the GGUF file (mainline's
//! `block_nvfp4`: 64 values in 36 bytes, [`NvfpLayout::Blocks`]) and the rows
//! of the MLX folder's tensors (bloomery's `NVFP4_ROWS`: a UE4M3 scale byte
//! and 8 code bytes a 16-value group, [`NvfpLayout::Rows`]).

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{NVFP4_BYTES, NVFP4_VALUES, nvfp4_stack_bytes};

use super::{Kernels, WeightAlign, check_weights_align, check_window_len, pack, range, word};

const WHAT: &str = "nvfp4";

/// Rows a threadgroup holds: one a SIMD group, eight of them.
const ROWS_PER_GROUP: usize = 8;
/// Threads in the threadgroup.
const THREADS: usize = ROWS_PER_GROUP * 32;
/// Values a group of the rows layout holds.
const ROWS_GROUP_VALUES: usize = 16;
/// Bytes a group of the rows layout holds.
const ROWS_GROUP_BYTES: usize = 9;

/// How the weights are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NvfpLayout {
    /// Mainline's `block_nvfp4`: 64 values in 36 bytes, the four sub-blocks'
    /// scale bytes first. `k` is a multiple of 64.
    Blocks,
    /// Bloomery's NVFP4 rows: 16 values in 9 bytes, the scale byte first, value
    /// `2i` the low nibble of code byte `i`. `k` is a multiple of 16.
    Rows,
}

impl NvfpLayout {
    /// The values one scale group covers: the block's or the row group's.
    #[must_use]
    pub const fn step(self) -> usize {
        match self {
            Self::Blocks => NVFP4_VALUES,
            Self::Rows => ROWS_GROUP_VALUES,
        }
    }

    /// The bytes of `rows` rows of `k` values, or `None` when `k` is not whole
    /// groups or the product wraps.
    #[must_use]
    pub fn bytes(self, rows: usize, k: usize) -> Option<usize> {
        match self {
            Self::Blocks => nvfp4_stack_bytes(rows, k),
            Self::Rows => {
                if !k.is_multiple_of(ROWS_GROUP_VALUES) {
                    return None;
                }
                rows.checked_mul(k / ROWS_GROUP_VALUES)?
                    .checked_mul(ROWS_GROUP_BYTES)
            }
        }
    }
}

/// One NVFP4 mat-vec launch of one expert's rows.
pub struct Nvfp4Params<'w> {
    /// The weights' layout.
    pub layout: NvfpLayout,
    /// Output rows.
    pub rows: usize,
    /// The input's values.
    pub k: usize,
    /// The rows' bytes, exactly [`NvfpLayout::bytes`].
    pub weights: Window<'w, u8, MetalDevice>,
    /// The input, `k`.
    pub x: Window<'w, f32, MetalDevice>,
    /// The output, `rows`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The expert's global scale, applied after the dot.
    pub w_scale: f32,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

impl Kernels {
    /// Encode the mat-vec: one dispatch of `⌈rows / 8⌉` threadgroups. The shape
    /// and every window are checked first; a refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn nvfp4_matvec(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: Nvfp4Params<'_>,
    ) -> Result<(), DeviceError> {
        if p.rows == 0 || p.k == 0 {
            return Err(refuse(format!("{} rows of {} values", p.rows, p.k)));
        }
        let need = p.layout.bytes(p.rows, p.k).ok_or_else(|| {
            refuse(format!(
                "{} values a row are not whole groups of {} (or {} rows wrap)",
                p.k,
                p.layout.step(),
                p.rows
            ))
        })?;
        if !p.w_scale.is_finite() {
            return Err(refuse(format!("scale {} is not finite", p.w_scale)));
        }
        check_window_len(WHAT, "weights", p.weights.len(), need)?;
        check_window_len(WHAT, "x", p.x.len(), p.k)?;
        check_window_len(WHAT, "y", p.y.len(), p.rows)?;
        if p.layout == NvfpLayout::Blocks {
            // The kernel reads each block's scale word as a 32-bit word.
            check_weights_align(
                WHAT,
                "weights",
                p.weights.byte_range()?.start.get(),
                WeightAlign::Word,
            )?;
            debug_assert!(need.is_multiple_of(NVFP4_BYTES));
        }
        let args = [
            word(WHAT, "k", p.k)?,
            word(WHAT, "rows", p.rows)?,
            p.layer,
            p.w_scale.to_bits(),
        ]
        .map(u32::to_ne_bytes);
        let pipeline = match p.layout {
            NvfpLayout::Blocks => &self.ink2.nvfp4,
            NvfpLayout::Rows => &self.ink2.nvfp4_rows,
        };
        encoder.set_pipeline(pipeline);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(p.rows.div_ceil(ROWS_PER_GROUP), 1, THREADS)
    }
}
