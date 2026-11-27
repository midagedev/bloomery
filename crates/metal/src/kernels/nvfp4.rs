//! The scaled mat-vec of Inkling 2's routed experts (`kernels/nvfp4.metal`):
//! a SIMD group a row, eight rows a threadgroup, every slot of a route in one
//! launch (the grid's height a slot), each slot's expert id read on the card
//! out of the router's ids and its rows' finished dots scaled by its expert's
//! global scale, times its route weight on a routed down.
//!
//! Five layouts, one kernel source: the NVFP4 blocks of the GGUF file
//! (mainline's `block_nvfp4`: 64 values in 36 bytes, [`NvfpLayout::Blocks`]),
//! the MLX folder's NVFP4 as two planes an expert ([`NvfpLayout::Planes`]:
//! the codes, then the UE4M3 scale bytes, one a 16-value group; sixteen rows
//! a SIMD group), and the two codas' forms: the GGUF's Q8_0 blocks
//! ([`NvfpLayout::Q8_0`]) and the MLX folder's MXFP8 rows
//! ([`NvfpLayout::Mxfp8Rows`]) or, the same bytes with four groups' codes
//! interleaved a lane, its quads ([`NvfpLayout::Mxfp8Quads`]).
//!
//! Every layout but the planes takes a SIMD group a row, eight rows a
//! threadgroup; the planes take sixteen rows a SIMD group, two SIMD groups a
//! threadgroup.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{NVFP4_BYTES, NVFP4_VALUES, TOP_K, nvfp4_stack_bytes};

use super::{Kernels, WeightAlign, check_weights_align, check_window_len, pack, range, word};

const WHAT: &str = "nvfp4";

/// Rows a threadgroup holds: one a SIMD group, eight of them.
const ROWS_PER_GROUP: usize = 8;
/// Threads in the threadgroup.
const THREADS: usize = ROWS_PER_GROUP * 32;
/// Rows a planes threadgroup holds: sixteen a SIMD group (`NVP_SG_ROWS`),
/// two of them (`NVP_SGS`).
const PLANES_ROWS_PER_GROUP: usize = 32;
/// Threads in the planes threadgroup.
const PLANES_THREADS: usize = 64;
/// Bytes a code word of the planes layout holds: a group's sixteen codes,
/// loaded as one 8-byte word.
const PLANES_CODE_WORD: usize = 8;
/// Values a group of the planes layout holds: one scale byte's.
const PLANES_GROUP_VALUES: usize = 16;
/// Bytes a group of the planes layout holds over its two planes: 8 code
/// bytes and the scale byte.
const PLANES_GROUP_BYTES: usize = 9;
/// Values a block of the Q8_0 and MXFP8 layouts holds.
const WIDE_GROUP_VALUES: usize = 32;
/// Bytes a Q8_0 block holds: the f16 scale and 32 codes.
const Q8_0_BYTES: usize = 34;
/// Bytes an MXFP8 group holds: the E8M0 scale byte and 32 codes.
const MXFP8_BYTES: usize = 33;
/// Values a block of the MXFP8 quads layout holds: four groups.
const QUADS_VALUES: usize = 4 * WIDE_GROUP_VALUES;
/// Bytes an MXFP8 quads block holds: the four groups' 128 code bytes and
/// their four scale bytes.
const QUADS_BYTES: usize = 4 * MXFP8_BYTES;

/// How the weights are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NvfpLayout {
    /// Mainline's `block_nvfp4`: 64 values in 36 bytes, the four sub-blocks'
    /// scale bytes first. `k` is a multiple of 64.
    Blocks,
    /// Bloomery's NVFP4 planes: an expert's codes plane (`k / 2` bytes a
    /// row, value `2i` the low nibble of byte `i`), then its scales plane
    /// (`k / 16` UE4M3 bytes a row), the rows in the launch's order. `k` is a
    /// multiple of 16.
    Planes,
    /// GGUF `Q8_0`: 34-byte blocks of 32 values, the f16 scale first. The
    /// GGUF coda's stacks; `k` is a multiple of 32.
    Q8_0,
    /// Bloomery's MXFP8 rows: 33 bytes a 32-value group, the E8M0 scale byte
    /// first, value `i` E4M3 code byte `i`. The MLX folder coda's stacks; `k`
    /// is a multiple of 32.
    Mxfp8Rows,
    /// Bloomery's MXFP8 quads: the rows' bytes in 132-byte blocks of four
    /// groups (128 values), a block's 128 code bytes first (lane `l`'s four
    /// codes, groups 0 to 3, the word at byte `4 l`) then the four scale
    /// bytes. The MLX folder coda's stacks as the session loads them; `k` is a
    /// multiple of 128. Declared after [`NvfpLayout::Mxfp8Rows`]: the clauses
    /// seed from the discriminant, so the earlier ones must not move.
    Mxfp8Quads,
}

impl NvfpLayout {
    /// The layout of the folder's MXFP8 coda stacks a run asks for: the word
    /// of its lever (`quads`, or unset, the quads; `rows`).
    ///
    /// # Errors
    /// A word that names neither layout.
    pub fn mx8_of(word: Option<&str>) -> Result<Self, String> {
        match word {
            None | Some("quads") => Ok(Self::Mxfp8Quads),
            Some("rows") => Ok(Self::Mxfp8Rows),
            Some(other) => Err(format!("{other}: the MXFP8 coda is quads or rows")),
        }
    }

    /// The values one scale group covers: the block's or the row group's.
    #[must_use]
    pub const fn step(self) -> usize {
        match self {
            Self::Blocks => NVFP4_VALUES,
            Self::Planes => PLANES_GROUP_VALUES,
            Self::Q8_0 | Self::Mxfp8Rows => WIDE_GROUP_VALUES,
            Self::Mxfp8Quads => QUADS_VALUES,
        }
    }

    /// The bytes of `rows` rows of `k` values, or `None` when `k` is not whole
    /// groups or the product wraps.
    #[must_use]
    pub fn bytes(self, rows: usize, k: usize) -> Option<usize> {
        match self {
            Self::Blocks => nvfp4_stack_bytes(rows, k),
            Self::Planes => {
                if !k.is_multiple_of(PLANES_GROUP_VALUES) {
                    return None;
                }
                rows.checked_mul(k / PLANES_GROUP_VALUES)?
                    .checked_mul(PLANES_GROUP_BYTES)
            }
            Self::Q8_0 | Self::Mxfp8Rows => {
                if !k.is_multiple_of(WIDE_GROUP_VALUES) {
                    return None;
                }
                let per = if self == Self::Q8_0 {
                    Q8_0_BYTES
                } else {
                    MXFP8_BYTES
                };
                rows.checked_mul(k / WIDE_GROUP_VALUES)?.checked_mul(per)
            }
            Self::Mxfp8Quads => {
                if !k.is_multiple_of(QUADS_VALUES) {
                    return None;
                }
                rows.checked_mul(k / QUADS_VALUES)?.checked_mul(QUADS_BYTES)
            }
        }
    }
}

/// One NVFP4 mat-vec launch over a route's slots.
pub struct Nvfp4Params<'w> {
    /// The weights' layout.
    pub layout: NvfpLayout,
    /// An expert's rows, every one of them a slot's.
    pub rows: usize,
    /// The input's values.
    pub k: usize,
    /// The stack: every expert's rows, expert `e`'s at `e` times
    /// [`NvfpLayout::bytes`], exactly one expert a scale.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The route's expert ids, one a slot, at most [`TOP_K`]: a slot whose id
    /// is past the stack raises the expert-id site and its rows are NaN.
    pub ids: Window<'w, u32, MetalDevice>,
    /// Each expert's global scale, one an expert of the stack.
    pub scales: Window<'w, f32, MetalDevice>,
    /// Each slot's route weight, which its expert's scale is multiplied by
    /// (a routed down), or none (a gate·up).
    pub route_weights: Option<Window<'w, f32, MetalDevice>>,
    /// The input: `k` values every slot reads, or a slot's own `k` each
    /// ([`Nvfp4Params::x_per_slot`]).
    pub x: Window<'w, f32, MetalDevice>,
    pub x_per_slot: bool,
    /// The output, `slots · rows`, in planes of `plane` rows: row `r` of slot
    /// `s` at `((r / plane) · slots + s) · plane + r % plane`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// Rows an output plane holds; it divides `rows`.
    pub plane: usize,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

impl Kernels {
    /// Encode the mat-vec: one dispatch of `⌈rows / 8⌉ × slots` threadgroups
    /// (`⌈rows / 32⌉ × slots` of the planes).
    /// The shape and every window are checked first; a refused launch encodes
    /// nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn nvfp4_matvec(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: Nvfp4Params<'_>,
    ) -> Result<(), DeviceError> {
        if p.rows == 0 || p.k == 0 {
            return Err(refuse(format!("{} rows of {} values", p.rows, p.k)));
        }
        let per = p.layout.bytes(p.rows, p.k).ok_or_else(|| {
            refuse(format!(
                "{} values a row are not whole groups of {} (or {} rows wrap)",
                p.k,
                p.layout.step(),
                p.rows
            ))
        })?;
        let slots = p.ids.len().get();
        if slots == 0 || slots > TOP_K {
            return Err(refuse(format!("{slots} slots; a route holds 1 to {TOP_K}")));
        }
        if p.plane == 0 || !p.rows.is_multiple_of(p.plane) {
            return Err(refuse(format!(
                "a plane of {} rows does not divide {}",
                p.plane, p.rows
            )));
        }
        let experts = p.scales.len().get();
        let need = per
            .checked_mul(experts)
            .ok_or_else(|| refuse(format!("{experts} experts of {per} bytes wrap")))?;
        check_window_len(WHAT, "weights", p.weights.len(), need)?;
        check_window_len(
            WHAT,
            "x",
            p.x.len(),
            if p.x_per_slot { slots * p.k } else { p.k },
        )?;
        check_window_len(WHAT, "y", p.y.len(), slots * p.rows)?;
        if let Some(w) = &p.route_weights {
            check_window_len(WHAT, "route weights", w.len(), slots)?;
        }
        if p.layout == NvfpLayout::Mxfp8Quads {
            // The kernel reads a block's code and scale words as 32-bit words;
            // a block is 132 bytes, so an expert's rows keep the alignment.
            check_weights_align(
                WHAT,
                "weights",
                p.weights.byte_range()?.start.get(),
                WeightAlign::Word,
            )?;
        }
        if p.layout == NvfpLayout::Blocks {
            // The kernel reads each block's scale word as a 32-bit word.
            check_weights_align(
                WHAT,
                "weights",
                p.weights.byte_range()?.start.get(),
                WeightAlign::Word,
            )?;
            debug_assert!(per.is_multiple_of(NVFP4_BYTES));
        }
        if p.layout == NvfpLayout::Planes {
            // The kernel reads a group's codes as one 8-byte word and its
            // inputs as float4s.
            let start = p.weights.byte_range()?.start.get();
            if !start.is_multiple_of(PLANES_CODE_WORD)
                || (experts > 1 && !per.is_multiple_of(PLANES_CODE_WORD))
            {
                return Err(refuse(format!(
                    "the weights start {start} bytes into their buffer, an expert {per} bytes: \
                     the planes' code words are 8-byte words"
                )));
            }
            check_weights_align(
                WHAT,
                "x",
                p.x.byte_range()?.start.get(),
                WeightAlign::Float4,
            )?;
        }
        let args = [
            word(WHAT, "k", p.k)?,
            word(WHAT, "rows", p.rows)?,
            p.layer,
            word(WHAT, "experts", experts)?,
            word(WHAT, "slots", slots)?,
            word(WHAT, "plane", p.plane)?,
            if p.x_per_slot {
                word(WHAT, "k", p.k)?
            } else {
                0
            },
            u32::from(p.route_weights.is_some()),
        ]
        .map(u32::to_ne_bytes);
        let pipeline = match p.layout {
            NvfpLayout::Blocks => &self.ink2.nvfp4,
            NvfpLayout::Planes => &self.ink2.nvfp4_planes,
            NvfpLayout::Q8_0 => &self.ink2.q8_0_scaled,
            NvfpLayout::Mxfp8Rows => &self.ink2.mxfp8_rows,
            NvfpLayout::Mxfp8Quads => &self.ink2.mxfp8_quads,
        };
        encoder.set_pipeline(pipeline);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.set(Binding::of(&p.ids)?)?;
        slot.set(Binding::of(&p.scales)?)?;
        // A gate·up reads no route weight: the scales stand in the slot.
        slot.set(match &p.route_weights {
            Some(w) => Binding::of(w)?,
            None => Binding::of(&p.scales)?,
        })?;
        if p.layout == NvfpLayout::Planes {
            return slot.dispatch_groups(
                p.rows.div_ceil(PLANES_ROWS_PER_GROUP),
                slots,
                PLANES_THREADS,
            );
        }
        slot.dispatch_groups(p.rows.div_ceil(ROWS_PER_GROUP), slots, THREADS)
    }
}

#[cfg(test)]
mod tests {
    use crate::experts_geom::routed_ids;
    use crate::inkling2_geom::TOP_K;
    use crate::instance::INKLING2_MINI;

    use super::NvfpLayout;

    /// The quads hold the rows' bytes (an MXFP8 stack's size does not move with
    /// its layout) and take only whole blocks of four groups; the lever's word
    /// names a layout, and no other word does.
    #[test]
    fn the_quads_hold_the_rows_bytes_in_blocks_of_four_groups() {
        for (rows, k) in [(1, 128), (37, 2304), (2304, 1152), (4608, 2304)] {
            assert_eq!(
                NvfpLayout::Mxfp8Quads.bytes(rows, k),
                NvfpLayout::Mxfp8Rows.bytes(rows, k),
                "{rows} rows of {k}"
            );
        }
        assert_eq!(
            NvfpLayout::Mxfp8Quads.step(),
            4 * NvfpLayout::Mxfp8Rows.step()
        );
        assert!(NvfpLayout::Mxfp8Rows.bytes(1, 96).is_some());
        assert_eq!(NvfpLayout::Mxfp8Quads.bytes(1, 96), None);
        assert_eq!(NvfpLayout::mx8_of(None), Ok(NvfpLayout::Mxfp8Quads));
        assert_eq!(
            NvfpLayout::mx8_of(Some("quads")),
            Ok(NvfpLayout::Mxfp8Quads)
        );
        assert_eq!(NvfpLayout::mx8_of(Some("rows")), Ok(NvfpLayout::Mxfp8Rows));
        assert!(NvfpLayout::mx8_of(Some("planes")).is_err());
    }

    /// The family's row compiles at least the route's slots as routed ids,
    /// so no slot of a launch is the shared slot the id read leaves unread.
    #[test]
    fn the_rows_routed_ids_cover_a_route() {
        assert!(routed_ids(&INKLING2_MINI) >= TOP_K);
    }
}
