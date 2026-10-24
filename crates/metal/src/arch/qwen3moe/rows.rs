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

use super::arena::{Arena, MixerBufs};
use super::session::SessionError;
use crate::buffer::MetalBuffer;
use crate::kernels::{
    EXPERT_K, EXPERT_SLOTS, LOGIT_STRIDE, PARTIALS_PER_HEAD, VALUE_HEADS, prepared_elems,
    route_table_words,
};
use crate::residency::Alloc;

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
    device: Alloc<'_>,
    n: usize,
    what: &str,
) -> Result<MetalBuffer<T>, SessionError> {
    if n == 0 {
        return Err(refuse(format!("{what} of no elements")));
    }
    let mut buffer = device.buffer(Elems(n))?;
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
    /// The delta launch's prepared rows, [`RowsArena::delta_prep_stride`] a
    /// row: the widest delta layer's prepared row.
    pub(super) delta_prep: MetalBuffer<f32>,
    /// The f32s of one prepared delta row.
    pub(super) delta_prep_stride: usize,
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
        device: Alloc<'_>,
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
        let delta_prep_stride = arena
            .layers
            .iter()
            .filter_map(|l| match &l.mixer {
                MixerBufs::Delta(d) => Some(d.n_k),
                MixerBufs::Attention(_) => None,
            })
            .map(|n_k| {
                prepared_elems(n_k)
                    .ok_or_else(|| refuse(format!("a delta layer of {n_k} key heads")))
            })
            .try_fold(0, |w, e| e.map(|e| w.max(e)))?;
        let delta_prep = zeroed(
            device,
            product(cap, delta_prep_stride.max(1), "the delta's prepared rows")?,
            "the delta's prepared rows",
        )?;
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
            delta_prep.len().get(),
            logits.len().get(),
            pick_weights.len().get(),
            h.len().get(),
            y.len().get(),
        ];
        let u32s = [
            ids.len().get(),
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
            delta_prep,
            delta_prep_stride,
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
