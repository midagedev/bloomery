//! The rows arena: the buffers a pass of up to `cap` rows touches, opened
//! with the session and kept.
//!
//! A pass walks the chain at `m` rows with the residual stream, the normed
//! input, the P1 projection, the delta numerator or the attention output,
//! the router's logits, picks and route table, and the experts' activations
//! and outputs each `m` rows deep, a row after another at the stride the
//! kernels read. The weights, caches and recurrent stores are the token
//! arena's: a pass reads and writes the same ones, so a pass and a step
//! continue each other.
//!
//! The expert buffers hold a row per slot (`m` times the instance's slots),
//! the pair's output and the down's output at their widths: they are most of
//! the arena.
//!
//! The copies a pass leaves after rows inside it ([`SnapBufs`]) are opened
//! apart from the arena, by the open of a session that marks or else by the
//! first pass that takes one: a session that takes none never pays for them.
//!
//! No buffer is written by the host at its open. A new buffer's pages are
//! zeros (Metal clears a buffer it makes), and no kernel of a pass reads a
//! cell no earlier op of the same pass wrote: the GEMMs clamp their row
//! loads to the pass's rows, the routed GEMMs theirs to the tile's places,
//! the route table counts in threadgroup memory it clears, and the pick
//! reads only the router's columns.

use bloomery_device::{Buffer, Elems};

use super::arena::{Arena, MixerBufs};
use super::session::SessionError;
use crate::buffer::MetalBuffer;
use crate::geom::Geometry;
use crate::kernels::{SNAPS, route_table_words};
use crate::residency::Alloc;

/// The widest a row of the delta numerator or the attention output is, at
/// `g`: the value heads' columns or the query heads' values.
fn mix_stride(g: &Geometry) -> usize {
    (g.delta.value_heads * g.delta.head_cols).max(g.attn.query_heads * g.attn.head)
}

const WHAT: &str = "qwen35moe rows arena";

fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
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
    /// The delta numerator, or the attention output: `mix_stride` a row.
    pub(super) mix: MetalBuffer<f32>,
    pub(super) mix_stride: usize,
    /// The delta step's partial sums of squares, a row each.
    pub(super) delta_partials: MetalBuffer<f32>,
    /// The delta launch's prepared rows, [`RowsArena::delta_prep_stride`] a
    /// row: the widest delta layer's prepared row.
    pub(super) delta_prep: MetalBuffer<f32>,
    /// The f32s of one prepared delta row.
    pub(super) delta_prep_stride: usize,
    /// The router's logits, `RouterGeom::logit_stride` a row.
    pub(super) logits: MetalBuffer<f32>,
    /// The picks of each row, the shared expert's id last when the instance
    /// has one.
    pub(super) pick_ids: MetalBuffer<u32>,
    /// The picks' weights, the shared expert's gate last when the instance
    /// has one.
    pub(super) pick_weights: MetalBuffer<f32>,
    /// The slot at each place of the expert-sorted list.
    pub(super) sorted: MetalBuffer<u32>,
    /// The place of each slot.
    pub(super) inv: MetalBuffer<u32>,
    /// The route table's tiles.
    pub(super) tiles: MetalBuffer<u32>,
    /// The pair's outputs, a row of `ExpertsGeom::k` a place.
    pub(super) h: MetalBuffer<f32>,
    /// The down's outputs, a row of the hidden width a place.
    pub(super) y: MetalBuffer<f32>,
    /// The bytes the buffers take.
    pub(super) bytes: u64,
    /// The copies of the delta layers' stores, once a pass has taken one.
    pub(super) snaps: Option<SnapBufs>,
}

/// Where one delta layer's copies sit: its [`SNAPS`] lanes from `lane` in
/// [`SnapBufs::lane`], its [`SNAPS`] rings from `ring` in [`SnapBufs::ring`],
/// a ring being `ring_len` values.
#[derive(Clone, Copy, Debug)]
pub(super) struct SnapAt {
    pub(super) lane: usize,
    pub(super) ring: usize,
    pub(super) ring_len: usize,
}

/// The copies a pass leaves after rows inside it: for each delta layer,
/// [`SNAPS`] lanes of its state and [`SNAPS`] of its conv ring, each laid out
/// as the store it copies, and the position each copy holds. A copy holds a
/// position only once the pass that wrote it has settled clean, and only
/// until the next pass, cut or reset: its buffers are read for nothing else,
/// so they are not zeroed.
pub(super) struct SnapBufs {
    pub(super) lane: MetalBuffer<f32>,
    pub(super) ring: MetalBuffer<f32>,
    /// Each layer's copies; none for an attention layer.
    pub(super) at: Vec<Option<SnapAt>>,
    /// The position copy `k` holds: the position after the row it follows.
    pub(super) holds: [Option<u32>; SNAPS],
}

impl SnapBufs {
    /// The copies of the delta layers of `arena`, opened on `device`.
    pub(super) fn open(device: Alloc<'_>, arena: &Arena) -> Result<SnapBufs, SessionError> {
        let lane_len = arena
            .geom
            .delta
            .state_elems(1)
            .ok_or_else(|| refuse("a lane of the state does not fit"))?;
        let (mut lanes, mut rings) = (0usize, 0usize);
        let mut at = Vec::with_capacity(arena.layers.len());
        for layer in &arena.layers {
            let MixerBufs::Delta(d) = &layer.mixer else {
                at.push(None);
                continue;
            };
            let ring_len = d.ring.len().get();
            at.push(Some(SnapAt {
                lane: lanes,
                ring: rings,
                ring_len,
            }));
            lanes = product(SNAPS, lane_len, "a layer's lane copies")?
                .checked_add(lanes)
                .ok_or_else(|| refuse("the lane copies do not fit"))?;
            rings = product(SNAPS, ring_len, "a layer's ring copies")?
                .checked_add(rings)
                .ok_or_else(|| refuse("the ring copies do not fit"))?;
        }
        if lanes == 0 {
            return Err(refuse("copies of a model with no delta layer"));
        }
        let lane = device.buffer(Elems(lanes))?;
        let ring = device.buffer(Elems(rings))?;
        Ok(SnapBufs {
            lane,
            ring,
            at,
            holds: [None; SNAPS],
        })
    }
}

/// The element counts of the rows arena's buffers, from the file's shapes and
/// the launches' geometry: the one owner of the sizes, so the open and a
/// header-only plan make the same count.
pub(super) struct RowsLens {
    pub(super) ids: usize,
    pub(super) x: usize,
    pub(super) xn: usize,
    pub(super) proj: usize,
    pub(super) proj_stride: usize,
    pub(super) mix: usize,
    pub(super) mix_stride: usize,
    pub(super) delta_partials: usize,
    pub(super) delta_prep: usize,
    pub(super) delta_prep_stride: usize,
    pub(super) logits: usize,
    pub(super) slots: usize,
    pub(super) tiles: usize,
    pub(super) h: usize,
    pub(super) y: usize,
}

impl RowsLens {
    /// The lens of a pass of `cap` rows at the launch geometry `g` over a token
    /// arena of residual width `hidden`, a P1 output of `proj` values at its
    /// widest, `n_experts` routed experts and delta layers of the key heads
    /// `delta_heads` names, each refused by name when it does not fit.
    pub(super) fn of(
        g: &Geometry,
        cap: usize,
        hidden: usize,
        proj: usize,
        n_experts: usize,
        delta_heads: impl Iterator<Item = usize>,
    ) -> Result<RowsLens, SessionError> {
        if cap == 0 {
            return Err(refuse("a pass of no rows"));
        }
        let slots = product(cap, g.experts.slots, "the pass's slots")?;
        // The widest P1 output a layer writes, rounded up to the four floats
        // the GEMM's output rows are aligned to.
        let proj_stride = proj.next_multiple_of(4);
        let words = route_table_words(slots, n_experts);
        let delta_prep_stride = delta_heads
            .map(|n_k| {
                g.delta
                    .prepared_elems(n_k)
                    .ok_or_else(|| refuse(format!("a delta layer of {n_k} key heads")))
            })
            .try_fold(0, |w, e| e.map(|e| w.max(e)))?;
        let nonzero = |n: usize, what: &str| {
            if n == 0 {
                Err(refuse(format!("{what} of no elements")))
            } else {
                Ok(n)
            }
        };
        let mix_stride = mix_stride(g);
        Ok(RowsLens {
            ids: nonzero(cap, "the ids")?,
            x: nonzero(product(cap, hidden, "the residual rows")?, "the residual")?,
            xn: nonzero(product(cap, hidden, "the normed rows")?, "the normed input")?,
            proj: nonzero(product(cap, proj_stride, "the P1 rows")?, "the P1 output")?,
            proj_stride,
            mix: nonzero(
                product(cap, mix_stride, "the mixer rows")?,
                "the mixer rows",
            )?,
            mix_stride,
            delta_partials: nonzero(
                product(cap, g.delta.partials_stride, "the delta partials")?,
                "the delta partials",
            )?,
            delta_prep: nonzero(
                product(cap, delta_prep_stride.max(1), "the delta's prepared rows")?,
                "the delta's prepared rows",
            )?,
            delta_prep_stride,
            logits: nonzero(
                product(cap, g.router.logit_stride, "the logit rows")?,
                "the logits",
            )?,
            slots: nonzero(slots, "the picks")?,
            tiles: nonzero(words, "the route table")?,
            h: nonzero(
                product(slots, g.experts.k, "the pair rows")?,
                "the pair outputs",
            )?,
            y: nonzero(product(slots, hidden, "the down rows")?, "the down outputs")?,
        })
    }

    /// Each buffer's bytes: the residual, the normed input, the P1 output,
    /// the mixer rows, the delta's partials and prepared rows, the logits,
    /// the pair's and the down's outputs (f32); the ids, the picks, the sorted
    /// slots, their places and the route table (u32); the pick weights (f32).
    pub(super) fn buffer_bytes(&self) -> [u64; 15] {
        let w = |n: usize| 4 * u64::try_from(n).expect("a length fits u64");
        [
            w(self.ids),
            w(self.x),
            w(self.xn),
            w(self.proj),
            w(self.mix),
            w(self.delta_partials),
            w(self.delta_prep),
            w(self.logits),
            w(self.slots),
            w(self.slots),
            w(self.slots),
            w(self.slots),
            w(self.tiles),
            w(self.h),
            w(self.y),
        ]
    }

    /// The bytes the buffers ask of the driver.
    pub(super) fn bytes(&self) -> u64 {
        self.buffer_bytes().iter().sum()
    }
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
        let hidden = arena.hidden;
        let delta_heads = arena.layers.iter().filter_map(|l| match &l.mixer {
            MixerBufs::Delta(d) => Some(d.n_k),
            MixerBufs::Attention(_) => None,
        });
        let lens = RowsLens::of(
            &arena.geom,
            cap,
            hidden,
            arena.proj.len().get(),
            arena.n_experts,
            delta_heads,
        )?;
        let ids = device.buffer(Elems(lens.ids))?;
        let x = device.buffer(Elems(lens.x))?;
        let xn = device.buffer(Elems(lens.xn))?;
        let proj = device.buffer(Elems(lens.proj))?;
        let mix = device.buffer(Elems(lens.mix))?;
        let delta_partials = device.buffer(Elems(lens.delta_partials))?;
        let delta_prep = device.buffer(Elems(lens.delta_prep))?;
        let logits = device.buffer(Elems(lens.logits))?;
        let pick_ids = device.buffer(Elems(lens.slots))?;
        let pick_weights = device.buffer(Elems(lens.slots))?;
        let sorted = device.buffer(Elems(lens.slots))?;
        let inv = device.buffer(Elems(lens.slots))?;
        let tiles = device.buffer(Elems(lens.tiles))?;
        let h = device.buffer(Elems(lens.h))?;
        let y = device.buffer(Elems(lens.y))?;
        let (proj_stride, mix_stride, delta_prep_stride, bytes) = (
            lens.proj_stride,
            lens.mix_stride,
            lens.delta_prep_stride,
            lens.bytes(),
        );
        Ok(RowsArena {
            cap,
            ids,
            x,
            xn,
            proj,
            proj_stride,
            mix,
            mix_stride,
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
            snaps: None,
        })
    }
}
