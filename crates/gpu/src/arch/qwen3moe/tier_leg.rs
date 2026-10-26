//! The Qwen3-30B and Qwen3.6 expert tier: one card beside the stage card,
//! hung through the common card leg ([`crate::host::leg_tier`]). What is
//! Qwen's here is the node family — the file's reading, `silu(g)·u`, and the
//! placed leg's arms — and the join of the tier's slots with the stage
//! card's.
//!
//! The join is the host leg's combine (`hostleg`) with the tier's slots in
//! the same slot order. A slot the stage card holds is the stage card's down;
//! a slot the tier holds is that slot of the tier's rows; a slot at [`HOST`]
//! is the host's and adds nothing; a place neither side serves raises
//! [`FaultSite::ExpertId`] and the token's value is NaN. The sum is
//! `(card + hsum) + resid`, and with the shared expert
//! `+ w[used] · sh` last. This order is the gate.

use std::error::Error;
use std::ops::Range;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::Split;
use model::arch::models::{Act as SpecAct, LayerSpec, Moe};
use model::moe::HostLayer;
use model::placement::ModelTensors;
use model::placement::accepts::{self, LegArms};
use model::r8file::R8Source;

use crate::fault::{FaultSink, FaultSite};
use crate::host::leg_tier::{self, LegFamily, LegWidths};
use crate::host::node::NodeFamily;
use crate::hybrid::HOST;
use crate::kquant::Act;
use crate::{GpuError, launch_u32};
use std::sync::Arc;

/// What the tier's errors name.
const WHAT: &str = "qwen3moe tier";

/// The tier page's rows: one column. The batch port's width is the placed
/// load's unit rows, passed as [`leg_tier::Hang::cols`], not as
/// [`leg_tier::LegOpen::rows`].
pub(super) const PAGE_ROWS: usize = 1;

/// Tokens one tier run covers. Both files' plans reserve the tier's scratch
/// at this chunk.
const LEG_CHUNK: usize = 8;

const _: () = assert!(model::arch::qwen3moe::place::TIER_CHUNK == LEG_CHUNK as u64);
const _: () = assert!(model::arch::qwen35moe::place::LEG_TIER_CHUNK == LEG_CHUNK as u64);

/// Threads per block of both join entries.
const THREADS: u32 = 256;

/// A gate or up stack the placed leg does not run.
const GATE_UP: &str = "a resident routed gate or up stack of a type the placed leg runs \
                       (`accepts::prompt_leg_gate_up`)";

/// A down stack the placed leg does not run.
const DOWN: &str =
    "a resident routed down stack of a type the placed leg runs (`accepts::prompt_leg_down`)";

/// The routed activation the tier runs: `silu(g)·u`. A clamped SwiGLU, and
/// any other activation, is refused by name.
fn qwen_act(moe: &Moe) -> Result<Act, GpuError> {
    match moe.act {
        SpecAct::SwiGlu { limit: None } => Ok(Act::SiluMul),
        SpecAct::SwiGlu { limit: Some(limit) } => Err(GpuError::shape(
            WHAT,
            format!("a routed SwiGLU clamped at {limit}; the tier runs silu(g)·u"),
        )),
        other => Err(GpuError::shape(
            WHAT,
            format!(
                "a routed {} activation; the tier runs silu(g)·u",
                other.what()
            ),
        )),
    }
}

/// The placed leg's arms ([`accepts::leg_arms`] of [`accepts::Program::Qwen3moeBody`]).
///
/// # Panics
///
/// When that program names no card-leg arms.
fn qwen_arms() -> (LegArms, (&'static str, &'static str)) {
    let arms = match accepts::leg_arms(accepts::Program::Qwen3moeBody) {
        Some(arms) => arms,
        None => panic!("{WHAT}: Program::Qwen3moeBody names no card-leg arms"),
    };
    (arms, (GATE_UP, DOWN))
}

/// The host views of the routed layers `run`, the refusal naming the run as
/// `word`'s. `n_layer` is the description's layer count, so a trunk layer is
/// not refused for sitting past a next-token block.
fn qwen_views(
    word: &'static str,
    n_layer: usize,
    w: LegWidths,
    src: R8Source<'_>,
    run: Range<usize>,
) -> Result<Vec<HostLayer>, Box<dyn Error + Send + Sync + 'static>> {
    let dims = model::arch::qwen35moe::host::RoutedDims {
        n_layer,
        n_expert: w.n_expert,
        embd: w.n_embd,
        ff: w.ff,
    };
    model::arch::qwen35moe::host::routed_layers(src, dims, run, word).map_err(|e| Box::new(e) as _)
}

/// Qwen3-30B as the common tier node reads it: the reader's description and
/// the hyperparameters the widths come from.
struct Q3File {
    hp: model::arch::qwen3moe::hparams::Hparams,
    tensors: ModelTensors,
    layers: Vec<LayerSpec>,
}

/// Qwen3-30B's facts for the common tier node service.
struct Qwen3;

impl LegFamily for Qwen3 {
    const WHAT: &'static str = WHAT;
    const WORD: &'static str = "qwen3moe";
    const ROWS: usize = PAGE_ROWS;
    const CHUNK: usize = LEG_CHUNK;
    type Inputs = Q3File;

    fn describe(file: &Split) -> Result<Q3File, GpuError> {
        let read = model::arch::qwen3moe::spec::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let hp = model::arch::qwen3moe::hparams::Hparams::read(file)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(Q3File {
            hp,
            tensors: read.tensors,
            layers: read.spec.layers,
        })
    }

    fn into_model(inputs: Q3File) -> ModelTensors {
        inputs.tensors
    }

    fn layers(inputs: &Q3File) -> &[LayerSpec] {
        &inputs.layers
    }

    fn widths(inputs: &Q3File) -> LegWidths {
        let e = &inputs.hp.experts;
        LegWidths {
            n_expert: e.n_expert,
            n_used: e.n_used,
            n_embd: inputs.hp.n_embd,
            ff: e.ff,
        }
    }

    fn routed_views(
        src: R8Source<'_>,
        inputs: &Q3File,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, Box<dyn Error + Send + Sync + 'static>> {
        qwen_views(
            Self::WORD,
            Self::layers(inputs).len(),
            Self::widths(inputs),
            src,
            run,
        )
    }

    fn tier_act(moe: &Moe) -> Result<Act, GpuError> {
        qwen_act(moe)
    }

    fn tier_arms() -> Option<(LegArms, (&'static str, &'static str))> {
        Some(qwen_arms())
    }
}

/// Qwen3.6's facts for the common tier node service. The description's
/// layers are the trunk's.
struct Qwen35;

impl LegFamily for Qwen35 {
    const WHAT: &'static str = WHAT;
    const WORD: &'static str = "qwen35moe";
    const ROWS: usize = PAGE_ROWS;
    const CHUNK: usize = LEG_CHUNK;
    type Inputs = model::arch::qwen35moe::place::PlanInputs;

    fn describe(file: &Split) -> Result<Self::Inputs, GpuError> {
        model::arch::qwen35moe::place::PlanInputs::describe(file)
            .map_err(|e| GpuError::plan(WHAT, e))
    }

    fn into_model(inputs: Self::Inputs) -> ModelTensors {
        inputs.model
    }

    fn layers(inputs: &Self::Inputs) -> &[LayerSpec] {
        &inputs.spec.layers
    }

    fn widths(inputs: &Self::Inputs) -> LegWidths {
        // A tier leg places routed experts; `describe` of a file it opens has a routed layer.
        let moe = inputs
            .spec
            .routed()
            .expect("a qwen35moe tier leg's description holds a routed layer");
        LegWidths {
            n_expert: moe.experts as usize,
            n_used: moe.top_k as usize,
            n_embd: inputs.spec.hidden as usize,
            ff: moe.expert_ff as usize,
        }
    }

    fn routed_views(
        src: R8Source<'_>,
        inputs: &Self::Inputs,
        run: Range<usize>,
    ) -> Result<Vec<HostLayer>, Box<dyn Error + Send + Sync + 'static>> {
        qwen_views(
            Self::WORD,
            Self::layers(inputs).len(),
            Self::widths(inputs),
            src,
            run,
        )
    }

    fn tier_act(moe: &Moe) -> Result<Act, GpuError> {
        qwen_act(moe)
    }

    fn tier_arms() -> Option<(LegArms, (&'static str, &'static str))> {
        Some(qwen_arms())
    }
}

/// Qwen3-30B's tier node service ([`leg_tier::node_family`]).
pub(super) const QWEN3_FAMILY: NodeFamily = leg_tier::node_family::<Qwen3>();

/// Qwen3.6's tier node service ([`leg_tier::node_family`]).
pub(super) const QWEN35_FAMILY: NodeFamily = leg_tier::node_family::<Qwen35>();

/// One token's slots, the stage card's and the tier's, in slot order
/// (module doc). `j < n_used` reads the tier place; a shared slot past
/// `n_used` keeps the stage card's rule and does not read a tier place.
///
/// # Safety
///
/// `t < m`, `d < rows`, `sel.len()` and `w.len() >= (t + 1)·pitch`,
/// `down.len() >= (t + 1)·pitch·rows`, and for every `j < n_used`
/// `tsel.len() >= t·n_used + j + 1` and, when that slot's tier place is
/// below `n_tier`, `trows.len() >= (t·n_used + j + 1)·rows`.
#[inline(always)]
#[allow(
    clippy::too_many_arguments,
    reason = "one token's rows, weights, both place maps, the dims, the token, the row and the fault (rust-quality R8)"
)]
unsafe fn tier_sum(
    down: &[f32],
    trows: &[f32],
    w: &[f32],
    sel: &[u32],
    tsel: &[u32],
    (rows, pitch, n_used, n_card, n_tier): (usize, usize, usize, u32, u32),
    t: usize,
    d: usize,
    fault: FaultSink,
) -> f32 {
    let mut acc = 0.0f32;
    let mut j = 0usize;
    while j < pitch {
        let s = t * pitch + j;
        // SAFETY: s < (t + 1)·pitch, inside sel and w by this fn's contract.
        let place = unsafe { *sel.get_unchecked(s) };
        if j < n_used {
            if place < n_card {
                // SAFETY: s·rows + d < (t + 1)·pitch·rows <= down.len().
                acc += unsafe { *w.get_unchecked(s) * *down.get_unchecked(s * rows + d) };
            } else {
                let slot = t * n_used + j;
                // SAFETY: slot < (t + 1)·n_used, inside tsel by this fn's contract.
                let tp = unsafe { *tsel.get_unchecked(slot) };
                if tp < n_tier {
                    // SAFETY: slot·rows + d < (t + 1)·n_used·rows <= trows.len()
                    // when the tier place is below n_tier.
                    acc += unsafe { *w.get_unchecked(s) * *trows.get_unchecked(slot * rows + d) };
                } else if place != HOST {
                    fault.raise(FaultSite::ExpertId);
                    acc = f32::NAN;
                }
            }
        } else if place < n_card {
            // SAFETY: as the stage-card arm above.
            acc += unsafe { *w.get_unchecked(s) * *down.get_unchecked(s * rows + d) };
        } else if place != HOST {
            fault.raise(FaultSite::ExpertId);
            acc = f32::NAN;
        }
        j += 1;
    }
    acc
}

#[cuda_module]
mod qwen3moe_tier_join_kernels {
    use super::*;

    /// The join without a shared expert (module doc): thread `i < rows·m`
    /// writes `y[i]`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= rows * pitch * m,
            trows.len() >= rows * n_used * m,
            w.len() >= pitch * m,
            sel.len() >= pitch * m,
            tsel.len() >= n_used * m,
            hsum.len() >= rows * m,
            resid.len() >= rows * m,
            y.len() >= rows * m
        )
    )]
    pub fn qwen3moe_combine_tier(
        down: &[f32],
        trows: &[f32],
        w: &[f32],
        sel: &[u32],
        tsel: &[u32],
        hsum: &[f32],
        resid: &[f32],
        rows: u32,
        pitch: u32,
        n_used: u32,
        n_card: u32,
        n_tier: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let rows_u = rows as usize;
        if i >= rows_u * m as usize {
            return;
        }
        let t = i / rows_u;
        // SAFETY: i < rows·m gives t < m and i − t·rows < rows; the launch
        // contract bounds every buffer for every token below m.
        let card = unsafe {
            tier_sum(
                down,
                trows,
                w,
                sel,
                tsel,
                (rows_u, pitch as usize, n_used as usize, n_card, n_tier),
                t,
                i - t * rows_u,
                fault,
            )
        };
        // SAFETY: i < rows·m bounds the hsum and resid reads and the y store
        // by the launch contract; thread i is y[i]'s only writer. This order
        // is the gate.
        unsafe {
            let v = (card + *hsum.get_unchecked(i)) + *resid.get_unchecked(i);
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// The join with the shared expert's output `sh` and its weight at slot
    /// `used` of each token (module doc): thread `i < rows·m` writes `y[i]`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            used < pitch,
            down.len() >= rows * pitch * m,
            trows.len() >= rows * n_used * m,
            w.len() >= pitch * m,
            sel.len() >= pitch * m,
            tsel.len() >= n_used * m,
            hsum.len() >= rows * m,
            resid.len() >= rows * m,
            sh.len() >= rows * m,
            y.len() >= rows * m
        )
    )]
    pub fn qwen3moe_combine_tier_sh(
        down: &[f32],
        trows: &[f32],
        w: &[f32],
        sel: &[u32],
        tsel: &[u32],
        hsum: &[f32],
        resid: &[f32],
        sh: &[f32],
        rows: u32,
        pitch: u32,
        n_used: u32,
        used: u32,
        n_card: u32,
        n_tier: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let rows_u = rows as usize;
        if i >= rows_u * m as usize {
            return;
        }
        let t = i / rows_u;
        let pitch_u = pitch as usize;
        // SAFETY: as qwen3moe_combine_tier's.
        let card = unsafe {
            tier_sum(
                down,
                trows,
                w,
                sel,
                tsel,
                (rows_u, pitch_u, n_used as usize, n_card, n_tier),
                t,
                i - t * rows_u,
                fault,
            )
        };
        // SAFETY: i < rows·m bounds hsum, resid, sh and y; t·pitch + used <
        // (t + 1)·pitch <= w.len() since used < pitch; thread i is y[i]'s
        // only writer. This order is the gate.
        unsafe {
            let w_sh = *w.get_unchecked(t * pitch_u + used as usize);
            let v = ((card + *hsum.get_unchecked(i)) + *resid.get_unchecked(i))
                + w_sh * *sh.get_unchecked(i);
            *y.get_unchecked_mut(i) = v;
        }
    }
}

/// [`TierJoin::enqueue`]'s arguments: the stage card's downs and the tier's
/// rows (per token slot-major), the router's weights and the slots' places
/// (`pitch` a token), the tier places of the routed slots (`n_used` a token),
/// the host tier's routed sums, the residual and the output, the stage
/// card's experts of the layer and the tier's, and with the shared expert
/// its output at weight slot `used`.
pub(super) struct TierCombineArgs<'a> {
    pub(super) down: &'a DeviceBuffer<f32>,
    pub(super) trows: &'a DeviceBuffer<f32>,
    pub(super) w: &'a DeviceBuffer<f32>,
    pub(super) sel: &'a DeviceBuffer<u32>,
    pub(super) tsel: &'a DeviceBuffer<u32>,
    pub(super) hsum: &'a DeviceBuffer<f32>,
    pub(super) resid: &'a DeviceBuffer<f32>,
    pub(super) shared: Option<(&'a DeviceBuffer<f32>, usize)>,
    pub(super) rows: usize,
    pub(super) pitch: usize,
    pub(super) n_used: usize,
    pub(super) n_card: usize,
    pub(super) n_tier: usize,
    pub(super) m: usize,
    pub(super) fault: FaultSink,
    pub(super) y: &'a mut DeviceBuffer<f32>,
}

/// The loaded join. Owns no stream: each enqueue takes the engine stream.
pub(super) struct TierJoin {
    module: qwen3moe_tier_join_kernels::LoadedModule,
}

impl TierJoin {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub(super) fn load(ctx: &Arc<CudaContext>) -> Result<TierJoin, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(qwen3moe_tier_join_kernels, ctx)? };
        Ok(TierJoin { module })
    }

    /// Enqueue the join of `a.m` tokens (module doc). One column: the tier
    /// page holds one. A shape whose buffers are short, more routed slots
    /// than the pitch, or a shared slot outside the pitch, is refused by
    /// name. Asynchronous, allocation-free, capturable.
    pub(super) fn enqueue(
        &self,
        stream: &CudaStream,
        a: TierCombineArgs<'_>,
    ) -> Result<(), GpuError> {
        const WHAT_J: &str = "qwen3moe_combine_tier";
        let TierCombineArgs {
            down,
            trows,
            w,
            sel,
            tsel,
            hsum,
            resid,
            shared,
            rows,
            pitch,
            n_used,
            n_card,
            n_tier,
            m,
            fault,
            y,
        } = a;
        if m != 1 {
            return Err(GpuError::shape(
                WHAT_J,
                format!("m {m}: the tier join runs one column"),
            ));
        }
        if n_used > pitch {
            return Err(GpuError::shape(
                WHAT_J,
                format!("n_used {n_used} past pitch {pitch}"),
            ));
        }
        let short = rows == 0
            || pitch == 0
            || n_used == 0
            || down.len() < rows * pitch * m
            || trows.len() < rows * n_used * m
            || w.len() < pitch * m
            || sel.len() < pitch * m
            || tsel.len() < n_used * m
            || hsum.len() < rows * m
            || resid.len() < rows * m
            || y.len() < rows * m
            || shared.is_some_and(|(sh, slot)| slot >= pitch || sh.len() < rows * m);
        if short {
            return Err(GpuError::shape(
                WHAT_J,
                format!(
                    "rows {rows}, pitch {pitch}, n_used {n_used}, m {m}, shared slot {:?}: \
                     down.len() {} trows.len() {} w.len() {} sel.len() {} tsel.len() {} \
                     hsum.len() {} resid.len() {} y.len() {} shared.len() {:?}",
                    shared.map(|(_, s)| s),
                    down.len(),
                    trows.len(),
                    w.len(),
                    sel.len(),
                    tsel.len(),
                    hsum.len(),
                    resid.len(),
                    y.len(),
                    shared.map(|(sh, _)| sh.len())
                ),
            ));
        }
        let cfg = LaunchConfig1D::new(
            launch_u32(WHAT_J, "grid", (rows * m).div_ceil(THREADS as usize))?,
            THREADS,
            0,
        );
        let (rows_u, pitch_u, n_used_u, n_card_u, n_tier_u, m_u) = (
            launch_u32(WHAT_J, "rows", rows)?,
            launch_u32(WHAT_J, "pitch", pitch)?,
            launch_u32(WHAT_J, "n_used", n_used)?,
            launch_u32(WHAT_J, "n_card", n_card)?,
            launch_u32(WHAT_J, "n_tier", n_tier)?,
            launch_u32(WHAT_J, "m", m)?,
        );
        match shared {
            None => {
                let prep = self.module.prepare_qwen3moe_combine_tier(cfg)?;
                self.module.qwen3moe_combine_tier(
                    stream, &prep, down, trows, w, sel, tsel, hsum, resid, rows_u, pitch_u,
                    n_used_u, n_card_u, n_tier_u, m_u, fault, y,
                )?;
            }
            Some((sh, slot)) => {
                let prep = self.module.prepare_qwen3moe_combine_tier_sh(cfg)?;
                self.module.qwen3moe_combine_tier_sh(
                    stream,
                    &prep,
                    down,
                    trows,
                    w,
                    sel,
                    tsel,
                    hsum,
                    resid,
                    sh,
                    rows_u,
                    pitch_u,
                    n_used_u,
                    launch_u32(WHAT_J, "used", slot)?,
                    n_card_u,
                    n_tier_u,
                    m_u,
                    fault,
                    y,
                )?;
            }
        }
        Ok(())
    }
}
