//! The V4.1 layer program of the step and the pair pass ([`StepProgram`]), and the
//! port it hands its host leg over through ([`HostLeg`]). Both walks are
//! [`runtime::sched::walk`] over one point of [`Overlap`]: the step is one
//! row, `(1, 1, Step)`; the pair pass two rows one layer apart, `(2, 1,
//! Step)`. The program's parts per row and layer:
//!
//! - the front: the engram step where the layer carries a site, the
//!   attention sub-layer, and the MoE sub-layer up to its go
//!   ([`FfnPiece::enqueue_go_front`](crate::chain::ffn::FfnPiece::enqueue_go_front));
//! - the shadow: the MoE sub-layer's own card work under the host leg, then
//!   the next site's token-only engram work;
//! - the back: the wait, the join, and the feature tap where one reads the
//!   layer.
//!
//! A row's begin is its gather and embedding broadcast, its end the collapse
//! into its head. An observer sees each [`Seam`] once the launches before it
//! are enqueued; the engine's step and the pair pass observe nothing.

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::{Chain, Hybrid};
use bloomery_gpu::model::Rows;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::CudaStream;
use runtime::sched::{self, At, LayerProgram, Overlap, Port, PortKind, Refused};

use super::{Body, Cursor, ListOf, PAIR_ROWS, Parts, Seam};
use crate::chain::ffn::{Ds41Host, GoFront};

/// What the walks' errors name.
const WHAT: &str = "deepseek41 Body::enqueue_chain";

// The step port opens a walk of `PAIR_ROWS` rows as `Chain::Pair`; a replay
// of the pair pass is served as `Rows::CHAIN`: the two must name one chain.
const _: () = assert!(matches!(<Body as Rows>::CHAIN, Chain::Pair) && PAIR_ROWS == 2);

/// An observer of the step's seams ([`super::Body::enqueue_observed`]).
pub(super) type Observe<'o> = dyn FnMut(&Gpu, Seam<'_>) -> Result<(), GpuError> + 'o;

/// Walk the step (one head) or the pair pass (two) over `parts`, the host
/// tier opened for it first, row `r` into `heads[r]`.
pub(super) fn walk_step(
    gpu: &Gpu,
    w: &Weights,
    parts: Parts<'_>,
    hybrid: &mut Hybrid<Ds41Host>,
    heads: [Option<&mut Head>; PAIR_ROWS],
    observe: &mut Observe<'_>,
) -> Result<(), GpuError> {
    let units = heads.iter().flatten().count();
    let layers = parts.layers.len();
    let o = Overlap {
        units,
        cols: 1,
        port: PortKind::Step,
    };
    let mut port = HostLeg {
        stream: gpu.stream(),
        hybrid,
    };
    let mut prog = StepProgram {
        gpu,
        w,
        parts,
        cur: [Cursor::default(); PAIR_ROWS],
        go: [None, None],
        heads,
        observe,
    };
    sched::walk(o, layers, &mut port, &mut prog)
}

/// The step's host leg through the body's host tier: the port of the
/// one-token step (`Chain::Step`) and of the pair pass (the chain its replay
/// is served as, [`Rows::CHAIN`]).
pub(super) struct HostLeg<'a> {
    stream: &'a CudaStream,
    hybrid: &'a mut Hybrid<Ds41Host>,
}

impl Port for HostLeg<'_> {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Step;

    /// Opens the host tier's step port on the point's rows
    /// ([`Hybrid::open_step`]): one row of one column is the step,
    /// [`PAIR_ROWS`] the pair pass; any other point is refused by name.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        self.hybrid.open_step(self.stream, o.units, o.cols)
    }

    fn refused(why: Refused) -> GpuError {
        GpuError::Shape {
            what: WHAT,
            detail: why.to_string(),
        }
    }
}

/// The V4.1 program over one walk of the step or the pair pass: the body's
/// parts, each row's cursor and head, and each row's MoE front between its
/// front and its shadow.
struct StepProgram<'s, 'w> {
    gpu: &'w Gpu,
    w: &'w Weights,
    parts: Parts<'s>,
    cur: [Cursor; PAIR_ROWS],
    go: [Option<GoFront<'w>>; PAIR_ROWS],
    heads: [Option<&'s mut Head>; PAIR_ROWS],
    observe: &'s mut Observe<'s>,
}

/// Row `row`'s entry of a per-row array.
fn row_of<T>(v: &mut [T; PAIR_ROWS], row: usize) -> Result<&mut T, GpuError> {
    v.get_mut(row).ok_or(GpuError::State {
        what: WHAT,
        missing: "the row's buffers",
    })
}

impl<'s, 'w> LayerProgram for StepProgram<'s, 'w> {
    type Port = HostLeg<'s>;

    /// The row's gather of its step words and its embedding broadcast.
    fn begin(&mut self, unit: usize) -> Result<(), GpuError> {
        let cur = row_of(&mut self.cur, unit)?;
        self.parts.begin_row(self.gpu, unit, cur)
    }

    /// The engram step where the layer carries a site, the attention
    /// sub-layer, the MoE sub-layer up to its go; the engram and attention
    /// seams.
    fn front(&mut self, port: &mut HostLeg<'s>, at: At) -> Result<(), GpuError> {
        let (gpu, w) = (self.gpu, self.w);
        let (i, row) = (at.layer, at.unit);
        let l = self.parts.layers.start + i;
        let cur = row_of(&mut self.cur, row)?;
        let p = &mut self.parts;
        if p.engram(gpu, w, i, l, row, cur)? {
            let lane = &p.lanes[row];
            (self.observe)(
                gpu,
                Seam::Engram {
                    layer: l,
                    streams: lane.hc[cur.s].buf(),
                    fold: &lane.folds[cur.f],
                },
            )?;
        }
        p.attn(gpu, w, i, l, row, cur)?;
        {
            let step = p.steps[i];
            let lane = &p.lanes[row];
            (self.observe)(
                gpu,
                Seam::Attn {
                    layer: l,
                    streams: lane.hc[cur.s].buf(),
                    fold: &lane.folds[cur.f],
                    taps: p.attn.taps(),
                    list: match step.list {
                        ListOf::None => None,
                        ListOf::Reads(j) | ListOf::Writes { list: j, .. } => {
                            p.lists.get(row).and_then(|r| r.get(j))
                        }
                    },
                },
            )?;
        }
        let go = p.ffn_front(gpu, w, i, l, row, *cur, port.hybrid)?;
        *row_of(&mut self.go, row)? = Some(go);
        Ok(())
    }

    /// The MoE sub-layer's card work under the host leg and the next site's
    /// token-only engram work, after the row's front of the same layer.
    fn shadow(&mut self, port: &mut HostLeg<'s>, at: At) -> Result<(), GpuError> {
        let (i, row) = (at.layer, at.unit);
        let l = self.parts.layers.start + i;
        let go = row_of(&mut self.go, row)?
            .take()
            .filter(|g| g.at() == (l, row))
            .ok_or(GpuError::State {
                what: WHAT,
                missing: "the row's MoE front of the layer before its shadow",
            })?;
        let cur = *row_of(&mut self.cur, row)?;
        self.parts
            .ffn_shadow(self.gpu, self.w, i, row, cur, go, port.hybrid)
    }

    /// The wait, the join and the feature tap; the MoE seam.
    fn back(&mut self, port: &mut HostLeg<'s>, at: At) -> Result<(), GpuError> {
        let gpu = self.gpu;
        let (i, row) = (at.layer, at.unit);
        let l = self.parts.layers.start + i;
        let cur = row_of(&mut self.cur, row)?;
        let p = &mut self.parts;
        p.ffn_join(gpu, i, l, row, cur, port.hybrid)?;
        let step = p.steps[i];
        let lane = &p.lanes[row];
        (self.observe)(
            gpu,
            Seam::Ffn {
                layer: l,
                streams: lane.hc[cur.s].buf(),
                fold: step.folds.then_some(&lane.folds[cur.f]),
                taps: p.ffn.taps_of(row)?,
            },
        )
    }

    /// The row's streams collapsed into its head, and the head.
    fn end(&mut self, unit: usize) -> Result<(), GpuError> {
        let cur = *row_of(&mut self.cur, unit)?;
        let head = row_of(&mut self.heads, unit)?
            .as_deref_mut()
            .ok_or(GpuError::State {
                what: WHAT,
                missing: "the row's head",
            })?;
        self.parts.head(self.gpu, self.w, unit, cur, head)
    }
}
