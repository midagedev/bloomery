//! The common dense block for every model: the gate·up·SwiGLU and the down projection over one row, of a dense
//! layer or a shared expert, launched by the weight types of its three matrices.
//!
//! A model keeps its tensor names, which layers run it, and where its limit comes from; everything the types
//! decide lives here. The module holds no model's name and borrows only [`ExpertKernels`] from this crate. The plan is read once at load from the file's header ([`BlockPlan::of_file`]) and checked
//! against the resident weights ([`BlockPlan::check_resident`]); the scratch is sized at load
//! ([`BlockScratch::new`]); [`enqueue_block`] is the one launch sequence. A row's launches by plan:
//! - the q8_0 pair with a q8_0 down ([`BlockPlan::Q8`]): the fused pair `ds41_shexp_gate_up` (the clamp fused,
//!   f32 `h` out), then the q8_0 gemv over `h`;
//! - one K-quant type for gate and up, Q4_K or Q5_K: the row's q8_1 blocks, then the fused clamped `_sel` over a
//!   stack of one expert (f32 `h` out);
//! - any other K-quant pair: the row's q8_1 blocks, a gemv each, and the clamped SwiGLU written into the
//!   down's q8_1 blocks (no f32 `h`);
//! - the down by its type: the q8_0 gemv over `h`, or a K-quant gemv over `h`'s q8_1 blocks.
//!
//! A K-quant block runs one row: the K-quant gemvs of Q4_K, Q6_K and Q3_K write row-major and the swiglu
//! quantizer and the model's outputs read token-major, so more than one column is the q8_0 block's alone
//! ([`enqueue_block_cols`], [`BlockPlan::check_cols`]).
//!
//! The launches beyond the q8_0 block's two are [`BlockPlan::extra_launches`], read from the plan by whoever
//! counts a graph's nodes.

use bloomery_gpu::gemm::GemmKernels;
use bloomery_gpu::kquant::{Act, GateUpAct, KquantKernels};
use bloomery_gpu::site::{self, KGemvKernels, SiteTy};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;

use crate::experts::ExpertKernels;

/// What the launches' refusals name.
const BLOCK: &str = "gated block";

/// The types a block's matrices read, one launch family each ([`site::file_site`]'s `allowed`); a model's
/// coverage pin for the block lists the same set.
pub const BLOCK_TYPES: [SiteTy; 4] = [SiteTy::Q8_0, SiteTy::Q4K, SiteTy::Q5K, SiteTy::Q6K];

/// The most columns the q8_0 block's multi-column launches run.
const COLS_MAX: usize = 8;

/// A shape no block launch reads, by name.
fn refuse(what: &'static str, detail: String) -> GpuError {
    GpuError::Shape { what, detail }
}

/// A block's gate·up half, picked once at load from the file's types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateUp {
    /// The q8_0 pair: `ds41_shexp_gate_up`, the clamp fused, f32 `h` out.
    Ds41,
    /// One K-quant type for both, Q4_K or Q5_K: the fused clamped `_sel` over a stack of one expert, f32 `h` out.
    Fused(SiteTy),
    /// A gemv each, then the clamped SwiGLU into the down's q8_1 blocks: no f32 `h`.
    Split { gate: SiteTy, up: SiteTy },
}

/// The launches of one block by its file's types, read once at load ([`BlockPlan::of_file`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPlan {
    gate_up: GateUp,
    down: SiteTy,
}

impl BlockPlan {
    /// A block of q8_0 matrices: the pair and the q8_0 gemv.
    pub const Q8: BlockPlan = BlockPlan {
        gate_up: GateUp::Ds41,
        down: SiteTy::Q8_0,
    };

    /// Whether the block is [`BlockPlan::Q8`].
    #[must_use]
    pub fn is_q8(self) -> bool {
        self == BlockPlan::Q8
    }

    /// The plan of a block whose gate, up and down are `tys`: the q8_0 pair or two K-quants for gate·up (the
    /// same Q4_K or Q5_K fused, else split), any of [`BLOCK_TYPES`] for the down, which a split gate·up needs a
    /// K-quant one of. Another mix is refused by name as `what`'s.
    ///
    /// # Errors
    /// A gate and up that are not two q8_0 or two K-quants, or a split pair over a down that is not a K-quant.
    pub fn of_types(
        what: &'static str,
        [gate, up, down]: [SiteTy; 3],
    ) -> Result<BlockPlan, GpuError> {
        let gate_up = match (gate, up) {
            (SiteTy::Q8_0, SiteTy::Q8_0) => GateUp::Ds41,
            (g, u) if g.kquant() && u.kquant() => match (g == u, g) {
                (true, SiteTy::Q4K | SiteTy::Q5K) => GateUp::Fused(g),
                _ => GateUp::Split { gate: g, up: u },
            },
            _ => {
                return Err(refuse(
                    what,
                    format!(
                        "gate is {gate} and up is {up}: a block takes two q8_0 or two K-quants"
                    ),
                ));
            }
        };
        if matches!(gate_up, GateUp::Split { .. }) && !down.kquant() {
            return Err(refuse(
                what,
                format!(
                    "gate {gate} and up {up} split, and the split writes q8_1 blocks a {down} down cannot read"
                ),
            ));
        }
        Ok(BlockPlan { gate_up, down })
    }

    /// The plan of the block `names` (gate, up, down) of `ff` rows over `n` values in `file`, read from the
    /// header before any upload.
    ///
    /// # Errors
    /// A tensor missing, of another shape or of a type outside [`BLOCK_TYPES`], or a mix [`of_types`] refuses.
    ///
    /// [`of_types`]: BlockPlan::of_types
    pub fn of_file(
        file: &Split,
        what: &'static str,
        names: [&str; 3],
        (ff, n): (usize, usize),
    ) -> Result<BlockPlan, GpuError> {
        let mut tys = [SiteTy::Q8_0; 3];
        for ((ty, name), shape) in tys.iter_mut().zip(names).zip(shapes(ff, n)) {
            *ty = site::file_site(file, what, name, shape, &BLOCK_TYPES)?;
        }
        BlockPlan::of_types(what, tys)
    }

    /// The launches [`enqueue_block`] enqueues beyond the q8_0 block's two (the pair, the down): a fused gate·up
    /// quantizes the row and launches the `_sel` (+1), a split one quantizes it, runs both gemvs and the clamped
    /// SwiGLU's quantizer (+3), and a K-quant down over an f32 `h` quantizes it first (+1; a split block's
    /// SwiGLU already wrote it).
    #[must_use]
    pub fn extra_launches(self) -> usize {
        let down_quantize = usize::from(self.down.kquant());
        match self.gate_up {
            GateUp::Ds41 => down_quantize,
            GateUp::Fused(_) => 1 + down_quantize,
            GateUp::Split { .. } => 3,
        }
    }

    /// The types of gate, up and down.
    fn types(self) -> [SiteTy; 3] {
        match self.gate_up {
            GateUp::Ds41 => [SiteTy::Q8_0, SiteTy::Q8_0, self.down],
            GateUp::Fused(t) => [t, t, self.down],
            GateUp::Split { gate, up } => [gate, up, self.down],
        }
    }

    /// Refused by name as `what`'s unless the resident weights `names` are the types the plan was read as, at
    /// the block's shape.
    ///
    /// # Errors
    /// A weight absent, resident in another form or of another shape.
    pub fn check_resident(
        self,
        w: &Weights,
        what: &'static str,
        names: [&str; 3],
        (ff, n): (usize, usize),
    ) -> Result<(), GpuError> {
        for ((ty, name), shape) in self.types().into_iter().zip(names).zip(shapes(ff, n)) {
            site::site(w, what, name, shape, ty)?;
        }
        Ok(())
    }

    /// Refused by name as `what`'s, naming `layer`, unless the block can run over several columns: only the
    /// q8_0 block does ([`enqueue_block_cols`]).
    ///
    /// # Errors
    /// A plan that is not [`BlockPlan::Q8`].
    pub fn check_cols(self, what: &'static str, layer: usize) -> Result<(), GpuError> {
        if self.is_q8() {
            return Ok(());
        }
        let [gate, up, down] = self.types();
        Err(refuse(
            what,
            format!(
                "layer {layer}'s block is gate {gate}, up {up}, down {down}: a launch over several \
                 columns reads q8_0 planes"
            ),
        ))
    }
}

/// The shapes of gate, up and down: `ff` rows of `n` values, then `n` rows of `ff`.
fn shapes(ff: usize, n: usize) -> [(usize, usize); 3] {
    [(ff, n), (ff, n), (n, ff)]
}

/// The kernels a K-quant block launches beyond the q8_0 block's: the K-quant gemvs, the fused `_sel` gate·up and
/// the clamped SwiGLU's quantizer. Made at load for a file with a K-quant block only.
pub struct KBlockKernels {
    gemv: KGemvKernels,
    sel: KquantKernels,
    swiglu: GemmKernels,
}

impl KBlockKernels {
    /// Load the three into `gpu`'s context. Load-time only.
    ///
    /// # Errors
    /// A module that does not load.
    pub fn load(gpu: &Gpu) -> Result<KBlockKernels, GpuError> {
        Ok(KBlockKernels {
            gemv: KGemvKernels::load(gpu)?,
            sel: KquantKernels::load(gpu.context(), gpu.fault_word())?,
            swiglu: GemmKernels::load(gpu.context())?,
        })
    }
}

/// The kernels one [`enqueue_block`] reads, both borrowed from the body that holds them: the pair's, which the
/// body's routed experts or drafts may also use, and the K-quant block's, absent on a file whose blocks are
/// all q8_0.
#[derive(Clone, Copy)]
pub struct BlockKernels<'k> {
    pub experts: &'k ExpertKernels,
    pub kquant: Option<&'k KBlockKernels>,
}

/// One row's buffers for its K-quant blocks: the normed row and each block width's `h` as q8_1 blocks, the
/// split gate and up in f32, and the one expert id the fused `_sel` reads.
pub struct BlockScratch {
    xa: Q8Act,
    ha: Vec<(usize, Q8Act)>,
    gu: Option<[DeviceBuffer<f32>; 2]>,
    one: DeviceBuffer<u32>,
}

impl BlockScratch {
    /// The scratch of a row that runs `blocks` (plan, width) over `n` values; `None` when every block is
    /// q8_0. Load-time only.
    ///
    /// # Errors
    /// An allocation that fails.
    pub fn new(
        stream: &CudaStream,
        n: usize,
        blocks: impl IntoIterator<Item = (BlockPlan, usize)>,
    ) -> Result<Option<BlockScratch>, GpuError> {
        let mut ha: Vec<(usize, Q8Act)> = Vec::new();
        let (mut any, mut split_ff) = (false, 0);
        for (plan, ff) in blocks.into_iter().filter(|(p, _)| !p.is_q8()) {
            any = true;
            if plan.down.kquant() && !ha.iter().any(|(f, _)| *f == ff) {
                ha.push((ff, Q8Act::with_k(stream, 1, ff)?));
            }
            if matches!(plan.gate_up, GateUp::Split { .. }) {
                split_ff = split_ff.max(ff);
            }
        }
        if !any {
            return Ok(None);
        }
        let gu = if split_ff > 0 {
            Some([
                DeviceBuffer::zeroed(stream, split_ff)?,
                DeviceBuffer::zeroed(stream, split_ff)?,
            ])
        } else {
            None
        };
        Ok(Some(BlockScratch {
            xa: Q8Act::with_k(stream, 1, n)?,
            ha,
            gu,
            one: DeviceBuffer::zeroed(stream, 1)?,
        }))
    }

    /// Device bytes held.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.xa.device_bytes()
            + self.ha.iter().map(|(_, a)| a.device_bytes()).sum::<usize>()
            + self
                .gu
                .as_ref()
                .map_or(0, |g| g.iter().map(DeviceBuffer::num_bytes).sum())
            + self.one.num_bytes()
    }
}

/// One block's inputs: its plan, tensor names (gate, up, down), width, SwiGLU limit and layer.
pub struct Block<'a> {
    pub plan: BlockPlan,
    pub names: [&'a str; 3],
    pub ff: usize,
    pub limit: f32,
    pub layer: usize,
}

/// One block's row buffers: the normed row in, `h` between, the block's output.
pub struct BlockIo<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub h: &'a mut DeviceBuffer<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// The resident K-quant words of `name`.
fn kq_words<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<u32>, GpuError> {
    match w.resident(BLOCK, name)? {
        DevWeight::KQuant { w, .. } => Ok(w),
        _ => Err(GpuError::Tensor {
            what: BLOCK,
            name: name.to_string(),
            need: "a K-quant weight",
        }),
    }
}

/// The K-quant kernels and scratch of a block that needs them, or a named error: the load made them for
/// every plan that is not q8_0.
fn kit<'a>(
    k: &BlockKernels<'a>,
    s: &'a mut Option<BlockScratch>,
) -> Result<(&'a KBlockKernels, &'a mut BlockScratch), GpuError> {
    match (k.kquant, s.as_mut()) {
        (Some(kk), Some(sc)) => Ok((kk, sc)),
        _ => Err(GpuError::State {
            what: BLOCK,
            missing: "the K-quant block's kernels and scratch",
        }),
    }
}

/// The q8_1 blocks of the width-`ff` `h`, or a named error.
fn h_blocks(ha: &mut [(usize, Q8Act)], ff: usize) -> Result<&mut Q8Act, GpuError> {
    ha.iter_mut()
        .find(|(f, _)| *f == ff)
        .map(|(_, a)| a)
        .ok_or(GpuError::State {
            what: BLOCK,
            missing: "the down's q8_1 blocks at the block's width",
        })
}

/// Enqueue one block over the one row `io.x`: a q8_0 block is the pair and the q8_0 gemv; any other is the
/// plan's gate·up, then the down by its type ([`site::kgemv`] over the q8_1 blocks, the q8_0 gemv over `h`).
/// Asynchronous, allocation-free, capturable; [`BlockPlan::extra_launches`] launches more than the q8_0
/// block's two on a K-quant one.
///
/// # Errors
/// A weight not resident as the plan says, a K-quant plan without its kernels or scratch, or a launch's
/// refusal.
pub fn enqueue_block(
    gpu: &Gpu,
    w: &Weights,
    k: &BlockKernels<'_>,
    s: &mut Option<BlockScratch>,
    blk: &Block<'_>,
    io: BlockIo<'_>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let [gate, up, down] = blk.names;
    let BlockIo { x, h, y } = io;
    if blk.plan.gate_up == GateUp::Ds41 {
        k.experts.enqueue_shexp_gate_up(
            stream,
            w.resident(BLOCK, gate)?,
            w.resident(BLOCK, up)?,
            x,
            blk.limit,
            h,
        )?;
        if !blk.plan.down.kquant() {
            return w.q8_gemv(gpu, BLOCK, down, h, y);
        }
    }
    let rule = Act::SwigluClamp { limit: blk.limit };
    let fault = gpu.layer_sink(blk.layer)?;
    match blk.plan.gate_up {
        // The pair above wrote `h`.
        GateUp::Ds41 => {}
        GateUp::Fused(ty) => {
            let (kk, sc) = kit(k, s)?;
            gpu.enqueue_quantize_q8_1_layer(x, &mut sc.xa, blk.layer)?;
            let a = GateUpAct {
                wg: kq_words(w, gate)?,
                wu: kq_words(w, up)?,
                act: &sc.xa,
                sel: &sc.one,
                n_slots: 1,
                rows_per_expert: blk.ff,
                slots_per_col: 1,
                rule,
            };
            match ty {
                SiteTy::Q4K => kk.sel.enqueue_gate_up_q4k(stream, &a, fault, h)?,
                _ => kk.sel.enqueue_gate_up_q5k(stream, &a, fault, h)?,
            }
        }
        GateUp::Split { gate: gt, up: ut } => {
            let (kk, sc) = kit(k, s)?;
            let BlockScratch { xa, ha, gu, .. } = sc;
            let Some([g, u]) = gu.as_mut() else {
                return Err(GpuError::State {
                    what: BLOCK,
                    missing: "the split block's f32 gate and up",
                });
            };
            let act = h_blocks(ha, blk.ff)?;
            gpu.enqueue_quantize_q8_1_layer(x, xa, blk.layer)?;
            site::kgemv(gpu, &kk.gemv, (gt, w, gate), xa, fault, g)?;
            site::kgemv(gpu, &kk.gemv, (ut, w, up), xa, fault, u)?;
            kk.swiglu
                .enqueue_swiglu_act_quant_q8(stream, g, u, rule, 1, act, fault)?;
            return site::kgemv(gpu, &kk.gemv, (blk.plan.down, w, down), act, fault, y);
        }
    }
    if !blk.plan.down.kquant() {
        return w.q8_gemv(gpu, BLOCK, down, h, y);
    }
    let (kk, sc) = kit(k, s)?;
    let act = h_blocks(&mut sc.ha, blk.ff)?;
    gpu.enqueue_quantize_q8_1_layer(h, act, blk.layer)?;
    site::kgemv(gpu, &kk.gemv, (blk.plan.down, w, down), act, fault, y)
}

/// Enqueue the q8_0 block over the first `cols` (1..=8) columns of `io.x`, token-major: the pair over the
/// columns in one launch, then the q8_0 gemv over them, each column's rows the one-row launch's bit for bit.
/// Asynchronous, allocation-free, capturable.
///
/// # Errors
/// A plan that is not [`BlockPlan::Q8`] ([`BlockPlan::check_cols`]), `cols` outside 1..=8, or a launch's
/// refusal.
pub fn enqueue_block_cols(
    gpu: &Gpu,
    w: &Weights,
    experts: &ExpertKernels,
    blk: &Block<'_>,
    cols: usize,
    io: BlockIo<'_>,
) -> Result<(), GpuError> {
    blk.plan.check_cols(BLOCK, blk.layer)?;
    if !(1..=COLS_MAX).contains(&cols) {
        return Err(refuse(
            BLOCK,
            format!("{cols} columns: a launch over columns takes 1..={COLS_MAX}"),
        ));
    }
    let [gate, up, down] = blk.names;
    let BlockIo { x, h, y } = io;
    experts.enqueue_shexp_gate_up_mcol(
        gpu.stream(),
        w.resident(BLOCK, gate)?,
        w.resident(BLOCK, up)?,
        x,
        cols,
        blk.limit,
        h,
    )?;
    w.q8_gemv_mcol(gpu, BLOCK, down, h, cols, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHAT: &str = "test";

    fn plan(tys: [SiteTy; 3]) -> Result<BlockPlan, GpuError> {
        BlockPlan::of_types(WHAT, tys)
    }

    fn refused(r: Result<BlockPlan, GpuError>, words: &[&str]) {
        let msg = match r {
            Err(GpuError::Shape { what, detail }) => format!("{what}: {detail}"),
            other => panic!("expected a named shape refusal, got {other:?}"),
        };
        for w in words {
            assert!(msg.contains(w), "{msg:?} lacks {w:?}");
        }
    }

    /// The plan is the types' dispatch: the q8_0 triple is `BlockPlan::Q8`, one Q4_K or Q5_K for gate and up
    /// is fused, any other K-quant pair splits, and every plan reads back the types it came from.
    #[test]
    fn the_plan_is_the_types_dispatch() {
        use SiteTy::{Q3K, Q4K, Q5K, Q6K, Q8_0};
        assert_eq!(plan([Q8_0, Q8_0, Q8_0]).unwrap(), BlockPlan::Q8);
        assert_eq!(plan([Q8_0, Q8_0, Q6K]).unwrap().gate_up, GateUp::Ds41);
        assert_eq!(plan([Q4K, Q4K, Q6K]).unwrap().gate_up, GateUp::Fused(Q4K));
        assert_eq!(plan([Q5K, Q5K, Q8_0]).unwrap().gate_up, GateUp::Fused(Q5K));
        let split = |g, u| GateUp::Split { gate: g, up: u };
        assert_eq!(plan([Q6K, Q6K, Q5K]).unwrap().gate_up, split(Q6K, Q6K));
        assert_eq!(plan([Q4K, Q5K, Q4K]).unwrap().gate_up, split(Q4K, Q5K));
        assert_eq!(plan([Q3K, Q3K, Q6K]).unwrap().gate_up, split(Q3K, Q3K));
        for tys in [
            [Q8_0, Q8_0, Q8_0],
            [Q8_0, Q8_0, Q5K],
            [Q4K, Q4K, Q8_0],
            [Q6K, Q6K, Q5K],
            [Q4K, Q5K, Q6K],
        ] {
            assert_eq!(plan(tys).unwrap().types(), tys);
        }
    }

    /// A mix no arm covers is refused by name, with the types: gate and up of two kinds, and a split pair over
    /// a q8_0 down.
    #[test]
    fn a_mix_no_arm_covers_is_refused() {
        use SiteTy::{Q4K, Q6K, Q8_0};
        refused(
            plan([Q8_0, Q4K, Q8_0]),
            &["test", "gate is Q8_0", "up is Q4_K"],
        );
        refused(plan([Q6K, Q8_0, Q6K]), &["gate is Q6_K", "up is Q8_0"]);
        refused(plan([Q6K, Q6K, Q8_0]), &["split", "Q8_0 down"]);
    }

    /// The launches beyond the q8_0 block's two, by plan: 0 for the q8_0 block, and the K-quant arms' derived
    /// counts (module doc) — one more for a K-quant down over an f32 `h`, three for a split pair.
    #[test]
    fn the_extra_launches_are_the_arms_counts() {
        use SiteTy::{Q4K, Q5K, Q6K, Q8_0};
        let extra = |tys| plan(tys).unwrap().extra_launches();
        assert_eq!(extra([Q8_0, Q8_0, Q8_0]), 0);
        assert_eq!(extra([Q8_0, Q8_0, Q6K]), 1);
        assert_eq!(extra([Q4K, Q4K, Q8_0]), 1);
        assert_eq!(extra([Q5K, Q5K, Q6K]), 2);
        assert_eq!(extra([Q6K, Q6K, Q5K]), 3);
        assert_eq!(extra([Q4K, Q5K, Q4K]), 3);
    }

    /// Only the q8_0 block runs over columns; every other plan is refused by name with its layer.
    #[test]
    fn only_the_q8_block_runs_over_columns() {
        use SiteTy::{Q4K, Q6K, Q8_0};
        assert!(BlockPlan::Q8.check_cols(WHAT, 0).is_ok());
        for tys in [[Q8_0, Q8_0, Q6K], [Q4K, Q4K, Q4K], [Q6K, Q6K, Q4K]] {
            match plan(tys).unwrap().check_cols(WHAT, 7) {
                Err(GpuError::Shape { detail, .. }) => {
                    assert!(detail.contains("layer 7"), "{detail}");
                }
                other => panic!("{tys:?}: {other:?}"),
            }
        }
    }
}
