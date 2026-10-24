//! The rows arena: the buffers a pass of up to `cap` rows touches, opened on
//! the first pass and kept.
//!
//! A pass walks the chain at `m` rows with the residual stream, the normed
//! input, the P1 projection, the delta numerator or the attention output,
//! the router's logits, picks and route table, and the experts' activations
//! and outputs each `m` rows deep, a row after another at the stride the
//! kernels read. The weights, caches and recurrent stores are the token
//! arena's: a pass reads and writes the same ones, so a pass and a step
//! continue each other.
//!
//! The expert buffers hold a row per slot (`m * 9`), the pair's output and
//! the down's output at their widths: they are most of the arena.

use bloomery_device::{Buffer, Elems};
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::arena::Arena;
use super::session::SessionError;
use crate::buffer::MetalBuffer;
use crate::kernels::{
    EXPERT_K, EXPERT_SLOTS, LOGIT_STRIDE, PARTIALS_PER_HEAD, VALUE_HEADS, partials_len,
    route_table_words, segments_for_keys,
};

/// The most rows of one attention launch. The launch's partials are held for
/// this many rows of the cache's most segments.
pub(super) const ATTN_RUN_ROWS: usize = 32;

/// The widest a row of the delta numerator or the attention output is.
pub(super) const MIX_STRIDE: usize = VALUE_HEADS * crate::kernels::HEAD_COLS;

/// The partial sums of squares a row of the delta step leaves.
pub(super) const DELTA_PARTIALS_STRIDE: usize = VALUE_HEADS * PARTIALS_PER_HEAD;

const WHAT: &str = "qwen35moe rows arena";

fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// A buffer of `n` elements, zeroed.
fn zeroed<T: bloomery_device::Elem>(
    device: &ProtocolObject<dyn MTLDevice>,
    n: usize,
    what: &str,
) -> Result<MetalBuffer<T>, SessionError> {
    if n == 0 {
        return Err(refuse(format!("{what} of no elements")));
    }
    let mut buffer = MetalBuffer::new(device, Elems(n))?;
    buffer.zero();
    Ok(buffer)
}

/// `a * b`, refused by name when it wraps.
fn product(a: usize, b: usize, what: &str) -> Result<usize, SessionError> {
    a.checked_mul(b)
        .ok_or_else(|| refuse(format!("{what}: {a} x {b} does not fit")))
}

/// The buffers of a pass.
pub(super) struct RowsArena {
    /// The most rows a pass holds.
    pub(super) cap: usize,
    /// The pass's ids, written by the host.
    pub(super) ids: MetalBuffer<u32>,
    /// The residual stream, a row of the hidden width each.
    pub(super) x: MetalBuffer<f32>,
    /// The normed input of the layer's projections and of its experts.
    pub(super) xn: MetalBuffer<f32>,
    /// The P1 output, `proj_stride` values a row.
    pub(super) proj: MetalBuffer<f32>,
    pub(super) proj_stride: usize,
    /// The delta numerator, or the attention output: [`MIX_STRIDE`] a row.
    pub(super) mix: MetalBuffer<f32>,
    /// The delta step's partial sums of squares, a row each.
    pub(super) delta_partials: MetalBuffer<f32>,
    /// One attention launch's segment partials.
    pub(super) attn_partials: MetalBuffer<f32>,
    /// The attention folds' tickets, one word a row.
    pub(super) attn_ticket: MetalBuffer<u32>,
    /// The router's logits, [`LOGIT_STRIDE`] a row.
    pub(super) logits: MetalBuffer<f32>,
    /// The picks of each row, the shared expert's id last.
    pub(super) pick_ids: MetalBuffer<u32>,
    /// The picks' weights, the shared expert's gate last.
    pub(super) pick_weights: MetalBuffer<f32>,
    /// The slot at each place of the expert-sorted list.
    pub(super) sorted: MetalBuffer<u32>,
    /// The place of each slot.
    pub(super) inv: MetalBuffer<u32>,
    /// The route table's tiles.
    pub(super) tiles: MetalBuffer<u32>,
    /// The pair's outputs, a row of [`EXPERT_K`] a place.
    pub(super) h: MetalBuffer<f32>,
    /// The down's outputs, a row of the hidden width a place.
    pub(super) y: MetalBuffer<f32>,
    /// The bytes the buffers take.
    pub(super) bytes: u64,
}

impl RowsArena {
    /// The rows arena of `cap` rows for the token arena `arena`, opened on
    /// `device`: every length from the file's shapes and the launches'
    /// geometry, refused by name when one does not fit.
    pub(super) fn open(
        device: &ProtocolObject<dyn MTLDevice>,
        arena: &Arena,
        cap: usize,
    ) -> Result<RowsArena, SessionError> {
        if cap == 0 {
            return Err(refuse("a pass of no rows"));
        }
        let hidden = arena.hidden;
        let slots = product(cap, EXPERT_SLOTS, "the pass's slots")?;
        // The widest P1 output a layer writes, rounded up to the four floats
        // the GEMM's output rows are aligned to.
        let proj_stride = arena.proj.len().get().next_multiple_of(4);
        let segments = segments_for_keys(
            u32::try_from(arena.ctx)
                .map_err(|_| refuse(format!("a cache of {} rows does not fit u32", arena.ctx)))?,
        ) as usize;
        let run_partials = product(
            partials_len(segments)
                .ok_or_else(|| refuse(format!("the partials of {segments} segments")))?,
            ATTN_RUN_ROWS,
            "an attention run's partials",
        )?;
        let words = route_table_words(slots, arena.n_experts);
        let ids = zeroed(device, cap, "the ids")?;
        let x = zeroed(
            device,
            product(cap, hidden, "the residual rows")?,
            "the residual",
        )?;
        let xn = zeroed(
            device,
            product(cap, hidden, "the normed rows")?,
            "the normed input",
        )?;
        let proj = zeroed(
            device,
            product(cap, proj_stride, "the P1 rows")?,
            "the P1 output",
        )?;
        let mix = zeroed(
            device,
            product(cap, MIX_STRIDE, "the mixer rows")?,
            "the mixer rows",
        )?;
        let delta_partials = zeroed(
            device,
            product(cap, DELTA_PARTIALS_STRIDE, "the delta partials")?,
            "the delta partials",
        )?;
        let attn_partials = zeroed(device, run_partials, "the attention partials")?;
        let attn_ticket = zeroed(device, cap, "the attention tickets")?;
        let logits = zeroed(
            device,
            product(cap, LOGIT_STRIDE, "the logit rows")?,
            "the logits",
        )?;
        let pick_ids = zeroed(device, slots, "the picks")?;
        let pick_weights = zeroed(device, slots, "the pick weights")?;
        let sorted = zeroed(device, slots, "the sorted slots")?;
        let inv = zeroed(device, slots, "the slots' places")?;
        let tiles = zeroed(device, words, "the route table")?;
        let h = zeroed(
            device,
            product(slots, EXPERT_K, "the pair rows")?,
            "the pair outputs",
        )?;
        let y = zeroed(
            device,
            product(slots, hidden, "the down rows")?,
            "the down outputs",
        )?;
        let f32s = [
            x.len().get(),
            xn.len().get(),
            proj.len().get(),
            mix.len().get(),
            delta_partials.len().get(),
            attn_partials.len().get(),
            logits.len().get(),
            pick_weights.len().get(),
            h.len().get(),
            y.len().get(),
        ];
        let u32s = [
            ids.len().get(),
            attn_ticket.len().get(),
            pick_ids.len().get(),
            sorted.len().get(),
            inv.len().get(),
            tiles.len().get(),
        ];
        let bytes = 4 * (f32s.iter().sum::<usize>() + u32s.iter().sum::<usize>()) as u64;
        Ok(RowsArena {
            cap,
            ids,
            x,
            xn,
            proj,
            proj_stride,
            mix,
            delta_partials,
            attn_partials,
            attn_ticket,
            logits,
            pick_ids,
            pick_weights,
            sorted,
            inv,
            tiles,
            h,
            y,
            bytes,
        })
    }
}
