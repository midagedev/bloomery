//! The prompt path: a prompt runs layer-major over passes of the rows the
//! load chose ([`choose_pass_rows`]: at most [`PASS_ROWS_MAX`]), each
//! launch over the pass's rows.
//!
//! - a projection is its site's gemv over column runs of at most
//!   [`COLS`] rows ([`project_rows`]): Q8_0 `q8_0_gemv_mcol` on the f32 rows,
//!   NVFP4 `nvfp4_gemv` on the run's q8_1 columns (at a width the q8_1 form
//!   does not take, the `_sel` gemv on its q8 blocks of 32), each weight read
//!   once a run; past [`COLS`] rows a Q8_0 site is one 32-value GEMM over the
//!   pass and an NVFP4 site the same GEMM over its resident NVFP4 planes
//!   ([`site::gemm`]);
//! - a Mamba-2 layer is one pass over its tokens
//!   ([`Mamba2Kernels::enqueue_pass`] with the chunked SSD scan,
//!   [`PassScan::Ssd`]: within its derived f16 bound of the exact scan, not
//!   the steps' bits);
//! - an attention layer appends the pass's K/V rows to its planes and runs
//!   the prefill flash's split entry over the pass's whole row set
//!   ([`crate::flash_gqa_prefill::FlashGqaPrefill::enqueue_split`]), each
//!   row over the keys up to its own position;
//! - a latent MoE layer routes the pass's rows in two launches
//!   ([`super::router::Routers::enqueue_rows`]), runs its kept experts as
//!   one `iq2_xxs_gemv_sel` over the pass's `TOP_K · m` slots each for the
//!   up and the down (past [`COLS`] rows the grouped IQ2_XXS GEMMs), or, for
//!   32-value stacks, as one 32-value GEMM each at any row count
//!   ([`experts_gemm32`]), sums each row's slots by its weights, and runs the
//!   shared expert as a one-expert `_sel` over the pass's `m` slots — the
//!   step's launches at `m` rows, in the step's order.
//!
//! A placed load ([`super::placed`]) walks the same layers through the host
//! tier's batch port ([`PlacedPass`], [`sched::walk`]): a layer with no host
//! share runs as above; a hosted MoE layer runs three parts around the
//! host's union over the pass's rows —
//!
//! - **front**: the norm, the router, the latent down, the download of the
//!   rows, weights and ids, and each slot's place in the card's stacks
//!   (`ds41_ffn_places`);
//! - **shadow**, under the host's union: the card's slots over those places
//!   (past [`COLS`] rows the grouped GEMMs over the layer's remapped route
//!   table, `gemm_route_remap`; up to them the `_sel` launches, a
//!   [`HOST`](crate::hybrid::HOST) slot's warps returning first) and the
//!   shared expert, none of the card's when the layer holds no expert;
//! - **back**: the card slots' weighted sum (`ds41_ffn_card_acc_9`) plus the
//!   host's sums, the latent up, the shared expert, the residual.
//!
//! The head reads the last row of the prompt's last pass only.
//!
//! After [`GpuModel::prefill_with`] the stores (the scan states, the conv
//! rings and the K/V planes) and the position are those of the prompt's ids
//! run as steps; the next step continues from them.
//!
//! The pass's rows live in [`Prompt`], cut at load for the rows the card's
//! room holds ([`choose_pass_rows`]), so a pass allocates nothing. A run's
//! multi-column gemv sums each output as the one-column gemv of its family
//! does, so how far a pass's rows agree with the steps' is the
//! multi-column launches' own contract.

use super::body::{Body, Layer};
use super::moe::{MoeLayer, MoeShape, Shared};
use super::placed::{Placed, PlacedSide};
use super::router::Routers;
use super::router::{COUNTS, MAX_TOP_K};
use crate::arch::hybrid::{RowBuffers, Shape, Site, norm_rows, residual_rows};
use crate::card_leg::{RouteW, StackW};
use crate::card_slots::CardAcc;
use crate::fault::FaultSink;
use crate::flash::PlaneAppend;
use crate::flash_gqa::HEAD;
use crate::flash_gqa_prefill::{FlashGqaPrefill, GqaPrefillArgs, SplitScratch};
use crate::gemm::GEMM_MAX_SLOTS;
use crate::gemm::Gemm32Kernels;
use crate::gemm::{
    Gemm32Args, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput, GemmRoute, GemmWeight,
};
use crate::head::Head;
use crate::host::BatchLeg;
use crate::host::run::HostRun;
use crate::kquant::SelDown;
use crate::mamba2::{Mamba2Kernels, Mamba2PassScratch, PassScan};
use crate::model::GpuModel;
use crate::model::fit::{FIT_RESERVE, largest_fit};
use crate::q5::Q8Blocks32;
use crate::q38::{EmbedArgs, EmbedRows};
use crate::quant_sel::QuantSelKernels;
use crate::site::{self, Form, SiteTy};
use crate::site::{KGemvKernels, SiteWeight, WideActs, WideForms, WideKernels};
use crate::tensor::{Pitched, WindowMut, dtod_2d};
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError, Q8Act, Window, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;
use model::arch::nemotron_h_moe::place::{self, PassDims};
use runtime::sched::{self, At, LayerProgram, Overlap, PortKind};

const WHAT: &str = "nemotron_h_moe::prefill";

/// The most ids one pass of [`GpuModel::prefill`] takes: the most whose
/// `TOP_K` slots one route table and one GEMM form hold
/// ([`GemmRoute::new`], [`GemmAct::new`]); the pass's rows themselves are
/// chosen at load ([`choose_pass_rows`]).
pub const PASS_ROWS_MAX: usize = 4096;
const _: () = assert!(MAX_TOP_K * PASS_ROWS_MAX <= GEMM_MAX_SLOTS);

// The plan's reserve for the pass is the formula of `place::pass_bytes`, which counts these sizes
// as the kernels fix them.
const _: () = assert!(COLS == place::COLS);
const _: () = assert!(P_ROWS == place::WORD_HEAD);
const _: () = assert!(HEAD == place::HEAD);
const _: () = assert!(crate::flash_gqa::GROUP == place::SPLIT_GROUP);
const _: () = assert!(crate::flash_gqa::SEGMENTS == place::FLASH_SEGMENTS);
const _: () = assert!(crate::model::MAX_PASS_ROWS == place::MAX_PASS_ROWS);
const _: () = assert!(crate::gemm::GEMM_BN == place::GEMM_BN);
const _: () = assert!(crate::gemm::GEMM32_STEP == place::GEMM32_STEP);
const _: () = {
    let mut widest = 0;
    let mut i = 0;
    while i < COUNTS.len() {
        if COUNTS[i] > widest {
            widest = COUNTS[i];
        }
        i += 1;
    }
    assert!(widest == place::WIDEST_ROUTER);
};

/// The columns one gemv run takes: the multi-column gemvs' most.
const COLS: usize = 8;

/// The pass's sizes: every width its buffers are cut from — the shape, the
/// MoE widths and the layers' projection widths — collected once at load
/// and read by both the fit check ([`Prompt::need`]) and the allocation
/// ([`Prompt::new`]), so the bytes the one counts and the other holds
/// cannot drift. The widths are the family's architecture facts; how many
/// rows a pass takes is the card's, decided apart ([`choose_pass_rows`]).
pub(super) struct PassSizes {
    sh: Shape,
    moe: MoeShape,
    /// The query heads a row's `q` holds.
    n_head: usize,
    /// The widest routed expert's up width; 0 where no layer is a MoE layer.
    ff: usize,
    /// The widest router table's expert count (the `probs` rows).
    n_expert: usize,
    /// The q8_1 slot forms the `_sel` fallback reads ([`slot_form`]): the
    /// fallback runs only for passes of at most [`COLS`] rows, so a form
    /// holds [`COLS`] columns (`TOP_K · COLS` for the routed up's input).
    slots: Vec<(usize, usize)>,
    /// The routed tables' expert counts, one a distinct count.
    route_n: Vec<usize>,
    /// The experts' GEMM forms ([`gemm_form`]): `(k, slots a row)`, a
    /// form's columns the pass's cut times that.
    gemm_acts: Vec<(usize, usize)>,
    /// The 32-value GEMM forms of the layers whose card experts are 32-value
    /// stacks ([`experts_gemm32`]), as `gemm_acts`.
    gemm32_acts: Vec<(usize, usize)>,
    /// Whether a layer's card experts are NVFP4 planes: the 32-value GEMM
    /// forms are then made with the native FP4 columns beside them where the
    /// build has them ([`GemmAct32::new_routed_for`]).
    routed_fp4: bool,
    /// The NVFP4 gemv run forms' widths, where the q8_1 form takes them.
    run_k: Vec<usize>,
    /// The NVFP4 gemv run forms' widths where it does not: [`COLS`] columns
    /// of q8 blocks of 32 for the `_sel` gemv ([`site::reads_at`]).
    run32_k: Vec<usize>,
    /// The q8 blocks of 32 the wide GEMMs read, one a site width.
    wide_k: Vec<usize>,
    /// Whether the pass runs on a placed load: its rows' places and the
    /// host's sums are buffers of the pass.
    placed: bool,
}

impl PassSizes {
    /// The sizes for `sh`, the MoE shape `moe`, `n_head` query heads and
    /// `layers`' widths, for a placed load or not. Host-side only; allocates
    /// nothing on the card.
    pub(super) fn of(
        (sh, moe, n_head): (&Shape, MoeShape, usize),
        layers: &[Layer],
        placed: bool,
    ) -> PassSizes {
        let moes = || {
            layers.iter().filter_map(|l| match l {
                Layer::Moe(m) => Some(m),
                _ => None,
            })
        };
        let mut slots: Vec<(usize, usize)> = Vec::new();
        let mut want_slots = |k: usize, cols: usize| {
            if !slots.iter().any(|s| s.0 == k && s.1 == cols) {
                slots.push((k, cols));
            }
        };
        for m in moes() {
            // The q8_1 slot forms the `_sel` fallback reads: the routed
            // experts' input and the slots' squared ReLU where the card
            // holds IQ2_XXS experts of the layer, the normed row and the
            // shared expert's where the shared expert is an IQ2_XXS stack. A
            // launch that reads no q8_1 form asks for none: the form's block
            // is the K-quant's 256 values, and a width no K-quant form takes
            // would refuse the load for a form nothing reads.
            if m.n_card > 0 && m.relu2_ff().is_none() {
                want_slots(moe.width, COLS);
                want_slots(m.ff, moe.top_k * COLS);
            }
            if m.shared_is_sel() {
                want_slots(sh.hidden, COLS);
                want_slots(moe.shared_ff, COLS);
            }
        }
        let mut route_n: Vec<usize> = Vec::new();
        let mut gemm_acts: Vec<(usize, usize)> = Vec::new();
        let mut want_acts = |k: usize, per_row: usize| {
            if !gemm_acts.iter().any(|a| a.0 == k && a.1 == per_row) {
                gemm_acts.push((k, per_row));
            }
        };
        let mut gemm32_acts: Vec<(usize, usize)> = Vec::new();
        let mut want_acts32 = |k: usize, per_row: usize| {
            if !gemm32_acts.iter().any(|a| a.0 == k && a.1 == per_row) {
                gemm32_acts.push((k, per_row));
            }
        };
        let mut routed_fp4 = false;
        for m in moes() {
            // The table runs over the card's experts of the layer: every
            // expert on a whole-card load, the plan's share on a placed one.
            let n = m.n_card;
            if n > 0 {
                if !route_n.contains(&n) {
                    route_n.push(n);
                }
                match m.relu2_ff() {
                    None => {
                        want_acts(moe.width, 1);
                        want_acts(m.ff, moe.top_k);
                    }
                    Some(ff) => {
                        want_acts32(moe.width, 1);
                        want_acts32(ff, moe.top_k);
                        routed_fp4 |= m.relu2_fp4();
                    }
                }
            }
            if m.shared_is_sel() {
                want_acts(sh.hidden, 1);
                want_acts(moe.shared_ff, 1);
            }
        }
        let mut run_k: Vec<usize> = Vec::new();
        let mut run32_k: Vec<usize> = Vec::new();
        let mut wide_k: Vec<usize> = Vec::new();
        for s in layers.iter().flat_map(Layer::sites) {
            if s.ty == SiteTy::Nvfp4 {
                // The q8_1 run form where the site reads it, the 32-value one
                // at a width it does not take.
                let runs = if s.reads() == Form::Q8Blocks32 {
                    &mut run32_k
                } else {
                    &mut run_k
                };
                if !runs.contains(&s.k) {
                    runs.push(s.k);
                }
            }
            if matches!(s.ty, SiteTy::Q8_0 | SiteTy::Nvfp4) && !wide_k.contains(&s.k) {
                wide_k.push(s.k);
            }
        }
        PassSizes {
            sh: *sh,
            moe,
            n_head,
            ff: moes().map(|m| m.ff).max().unwrap_or(0),
            n_expert: COUNTS.iter().copied().max().unwrap_or(0),
            slots,
            route_n,
            gemm_acts,
            gemm32_acts,
            routed_fp4,
            run_k,
            run32_k,
            wide_k,
            placed,
        }
    }

    /// The width of a row of the host's sums: the latent on a placed load,
    /// none on a whole-card one.
    fn hsum_width(&self) -> usize {
        if self.placed { self.moe.width } else { 0 }
    }

    /// The widths of the pass's buffers as the plan's formula takes them
    /// ([`place::pass_bytes`]).
    fn dims(&self) -> PassDims {
        let geom = self.sh.mamba.geom();
        PassDims {
            hidden: self.sh.hidden,
            n_kv: self.sh.n_kv,
            n_head: self.n_head,
            in_proj: self.sh.mamba.in_proj_width(),
            d_inner: geom.d_inner,
            ssm_heads: geom.n_heads,
            conv_dim: self.sh.mamba.conv_dim(),
            width: self.moe.width,
            top_k: self.moe.top_k,
            shared_ff: self.moe.shared_ff,
            ff: self.ff,
            n_expert: self.n_expert,
            slots: self.slots.clone(),
            route_n: self.route_n.clone(),
            gemm_acts: self.gemm_acts.clone(),
            gemm32_acts: self.gemm32_acts.clone(),
            native32_acts: self.native32_acts(),
            run_k: self.run_k.clone(),
            run32_k: self.run32_k.clone(),
            wide_k: self.wide_k.clone(),
            placed: self.placed,
        }
    }

    /// The 32-value GEMM forms that hold the native FP4 columns beside them
    /// ([`GemmAct32::new_routed_for`]): where a layer's card experts are NVFP4
    /// planes and the build has native FP4 products, those of a width of whole
    /// 64-value steps.
    fn native32_acts(&self) -> Vec<(usize, usize)> {
        if cfg!(bloomery_nvfp4_mma) && self.routed_fp4 {
            self.gemm32_acts
                .iter()
                .copied()
                .filter(|&(k, _)| k.is_multiple_of(place::GEMM32_STEP))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// The device bytes the pass's buffers hold at `rows` rows — every
    /// buffer [`Prompt::new`] allocates, by the same widths: what a fit
    /// check ([`choose_pass_rows`]) counts, and what the plan keeps for them
    /// ([`place::pass_bytes`]). Pure; [`Prompt::new`] asserts the buffers it
    /// allocated hold exactly this.
    pub(super) fn bytes(&self, rows: usize) -> usize {
        place::pass_bytes(&self.dims(), rows)
    }
}

/// The rows one pass takes on `free` card bytes, at most `cap` (the load's
/// most: [`PASS_ROWS_MAX`] on a whole-card load, the host union's columns on
/// a placed one). With `asked` unset, the
/// largest multiple of 64 in `64..=min(cap, ctx)` — the whole `ctx`
/// where that is below 64 — whose buffers ([`Prompt::need`]) plus
/// [`FIT_RESERVE`] fit ([`largest_fit`]), or a named refusal when not even
/// the smallest does: a default the user did not set shrinks and never
/// refuses past the 64-row floor. With `asked` set, exactly those rows or a
/// named refusal that gives the free bytes, the need and the largest size
/// that fits: a size the user named is never shrunk. Load-time only;
/// allocates nothing.
pub(super) fn choose_pass_rows(
    sizes: &PassSizes,
    (ctx, asked): (usize, Option<usize>),
    free: usize,
    cap: usize,
) -> Result<usize, GpuError> {
    const WHAT_FIT: &str = "nemotron_h_moe::pass_rows";
    let hi = cap.min(ctx);
    if hi == 0 {
        return Err(GpuError::shape(
            WHAT_FIT,
            "a pass holds at least one row; the context left none",
        ));
    }
    let candidates: Vec<usize> = if hi < 64 {
        vec![hi]
    } else {
        (1..=hi / 64).map(|k| 64 * k).rev().collect()
    };
    let need = |rows| Prompt::need(sizes, rows);
    let largest = largest_fit(candidates.iter().copied(), need, free, FIT_RESERVE);
    let refuse = |rows: usize| {
        GpuError::shape(
            WHAT_FIT,
            format!(
                "the prefill pass buffers for passes of {rows} rows need {} bytes and \
                 {FIT_RESERVE} more stay free for what the load allocates after them; the card \
                 has {free} free: {}",
                need(rows),
                largest.map_or("no pass size fits".to_string(), |v| {
                    format!("passes of {v} rows fit")
                })
            ),
        )
    };
    match asked {
        None => largest.ok_or_else(|| refuse(candidates[0])),
        Some(rows) if rows == 0 || rows > hi || (rows >= 64 && !rows.is_multiple_of(64)) => {
            Err(GpuError::shape(
                WHAT_FIT,
                format!(
                    "a pass of {rows} rows; a pass holds 1..={hi} rows here (at most \
                     {cap} and the context {ctx}), at 64 past it a multiple of 64"
                ),
            ))
        }
        Some(rows) if largest_fit(std::iter::once(rows), need, free, FIT_RESERVE).is_none() => {
            Err(refuse(rows))
        }
        Some(rows) => Ok(rows),
    }
}

/// The pass's word record: the first position, then one row word a key head
/// (head `h` at `P_ROWS + h`, the plane row `h · ctx + pos` the pass's first
/// K/V row lands at), then the pass's ids from [`Prompt::ids_at`].
const P_POS: usize = 0;
const P_ROWS: usize = 1;

/// The buffers one pass runs over: `rows` rows each, every row buffer cut at
/// `rows` rounded up to [`COLS`], so a run of [`COLS`] columns past the
/// pass's last row writes inside its buffer.
pub(super) struct Prompt {
    rows: usize,
    /// The query heads' width a row, `n_head · HEAD`.
    q_width: usize,
    /// The pass's two residual streams, each layer reading one and writing
    /// the other.
    x: [DeviceBuffer<f32>; 2],
    /// Every other buffer of the pass.
    b: PassBufs,
    /// The host's sums for the pass's rows, `latent` values a row: what the
    /// host tier's batch port uploads into on a placed load.
    hsum: DeviceBuffer<f32>,
    /// What the Mamba-2 pass precomputes: every token's conv outputs and
    /// per-head `σ`/`dA` ([`Mamba2PassScratch`]), sized to these `rows`.
    mamba2: Mamba2PassScratch,
}

/// [`Prompt`]'s buffers beside the residual streams.
struct PassBufs {
    /// The trunk's row buffers.
    buf: RowBuffers,
    /// The pass's K and V rows regrouped head-major: head `h`'s rows at
    /// `h · rows · HEAD`, what the plane append reads.
    kh: DeviceBuffer<f32>,
    vh: DeviceBuffer<f32>,
    /// The word record ([`P_POS`], [`P_ROWS`], the ids) and its host copy.
    words: DeviceBuffer<u32>,
    host_words: Vec<u32>,
    /// The router's outputs: every expert's score (the widest layer's count
    /// a row), then `TOP_K` ids and weights a row.
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    /// Each routed slot's place in the card's stacks (`TOP_K` a row), what a
    /// placed load's card slots and sum read.
    sel: DeviceBuffer<u32>,
    /// The latent down's rows; the slots' up rows (the widest expert); the
    /// slots' down rows; the rows' weighted sums.
    lat: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    /// The shared expert's up rows, its relu² rows and its down rows, then
    /// `moe_out + shared`.
    sh_up: DeviceBuffer<f32>,
    sh_relu: DeviceBuffer<f32>,
    sh: DeviceBuffer<f32>,
    sum: DeviceBuffer<f32>,
    /// One id 0 a row: the shared expert's slots.
    zero_ids: DeviceBuffer<u32>,
    /// The q8_1 slot forms the `_sel` fallback reads, one a width:
    /// `(k, columns, form)`, each [`COLS`] columns wide — the fallback runs
    /// only for passes of at most [`COLS`] rows, whatever the pass's cut.
    slots: Vec<(usize, usize, Q8Act)>,
    /// The projections' forms ([`project_rows`]).
    arms: Arms,
    /// The experts' GEMM forms ([`experts_gemm`]).
    ex: ExpertArms,
    /// The prefill flash's sub-group buffers: one plain launch's worth of
    /// query and output rows, what the pass's whole-row split entry moves
    /// through.
    split: SplitScratch,
}

/// What the experts' GEMMs read past [`COLS`] rows.
struct ExpertArms {
    /// One routed table a layer expert count, `TOP_K` slots a row.
    routes: Vec<(usize, GemmRoute)>,
    /// The q8_1 blocks of 128 the GEMMs read: `(k, columns, form)`.
    acts: Vec<(usize, usize, GemmAct)>,
    /// The q8 blocks of 32 the 32-value GEMMs read: `(k, columns, form)`.
    acts32: Vec<(usize, usize, GemmAct32)>,
}

/// What a projection over the pass's rows reads beside its f32 rows.
struct Arms {
    /// The q8_1 forms of one run of [`COLS`] columns the NVFP4 gemv reads,
    /// one a width.
    runs: Vec<Q8Act>,
    /// The q8 blocks of 32 of one run of [`COLS`] columns the NVFP4 `_sel`
    /// gemv reads at a width the q8_1 form does not take, one a width.
    runs32: Vec<Q8Blocks32>,
    /// The q8 blocks of 32 a Q8_0 GEMM reads, the pass's rows, one a width.
    wide: Vec<(usize, WideActs)>,
    /// The pass's one-expert table: slot `s` reads column `s`, filled once a
    /// pass.
    route: GemmRoute,
}

impl Prompt {
    /// The device bytes the pass's buffers hold at `rows` rows of `sizes`
    /// ([`PassSizes::bytes`]): what the load's fit check counts before
    /// anything is allocated.
    pub(super) fn need(sizes: &PassSizes, rows: usize) -> usize {
        sizes.bytes(rows)
    }

    /// The buffers for passes of up to `rows` rows over `sizes`' widths,
    /// which must hold exactly the bytes [`Prompt::need`] counted for them
    /// or the load fails by name. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        sizes: &PassSizes,
        rows: usize,
    ) -> Result<Prompt, GpuError> {
        if rows == 0 || rows > PASS_ROWS_MAX {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {rows} rows; a pass holds 1..={PASS_ROWS_MAX}"),
            ));
        }
        let (sh, moe, n_head) = (sizes.sh, sizes.moe, sizes.n_head);
        let cut = rows.next_multiple_of(COLS);
        let z = |n: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, cut * n.max(1))?)
        };
        let kv_width = sh.n_kv * HEAD;
        let mut slots: Vec<(usize, usize, Q8Act)> = Vec::new();
        for &(k, cols) in &sizes.slots {
            slots.push((k, cols, Q8Act::with_slots(stream, cols, k)?));
        }
        let mut ex = ExpertArms {
            routes: Vec::new(),
            acts: Vec::new(),
            acts32: Vec::new(),
        };
        for &n in &sizes.route_n {
            ex.routes
                .push((n, GemmRoute::new(stream, moe.top_k * cut, n)?));
        }
        for &(k, per_row) in &sizes.gemm_acts {
            let cols = per_row * cut;
            ex.acts.push((k, cols, GemmAct::new(stream, cols, k)?));
        }
        // NVFP4 planes take the native FP4 columns beside the q8 blocks where
        // the build has them; every other type's forms are the q8 blocks alone.
        let routed = sizes.routed_fp4.then_some(GgmlType::NVFP4);
        for &(k, per_row) in &sizes.gemm32_acts {
            let cols = per_row * cut;
            ex.acts32
                .push((k, cols, GemmAct32::new_routed_for(stream, cols, k, routed)?));
        }
        let mut runs: Vec<Q8Act> = Vec::new();
        for &k in &sizes.run_k {
            runs.push(Q8Act::with_k(stream, COLS, k)?);
        }
        let mut runs32: Vec<Q8Blocks32> = Vec::new();
        for &k in &sizes.run32_k {
            runs32.push(Q8Blocks32::new(stream, k, COLS)?);
        }
        let mut wide: Vec<(usize, WideActs)> = Vec::new();
        for &k in &sizes.wide_k {
            wide.push((
                k,
                WideActs::new(stream, WideForms::q8(false, true), cut, k)?,
            ));
        }
        let n_words = P_ROWS + sh.n_kv + rows;
        let prompt = Prompt {
            rows,
            q_width: n_head * HEAD,
            x: [z(sh.hidden)?, z(sh.hidden)?],
            b: PassBufs {
                buf: RowBuffers::new(stream, cut, &sh, (n_head, 0))?,
                kh: z(kv_width)?,
                vh: z(kv_width)?,
                words: DeviceBuffer::zeroed(stream, n_words)?,
                host_words: vec![0; n_words],
                probs: z(sizes.n_expert)?,
                ids: DeviceBuffer::zeroed(stream, cut * moe.top_k)?,
                weights: z(moe.top_k)?,
                sel: DeviceBuffer::zeroed(stream, cut * moe.top_k)?,
                lat: z(moe.width)?,
                up: z(moe.top_k * sizes.ff)?,
                down: z(moe.top_k * moe.width)?,
                moe: z(moe.width)?,
                sh_up: z(moe.shared_ff)?,
                sh_relu: z(moe.shared_ff)?,
                sh: z(sh.hidden)?,
                sum: z(sh.hidden)?,
                zero_ids: DeviceBuffer::zeroed(stream, cut)?,
                slots,
                ex,
                arms: Arms {
                    runs,
                    runs32,
                    wide,
                    route: GemmRoute::new(stream, cut, 1)?,
                },
                split: SplitScratch::new(stream, rows, sh.n_kv)?,
            },
            hsum: z(sizes.hsum_width())?,
            mamba2: Mamba2PassScratch::new(stream, sh.mamba, rows)?,
        };
        let need = Prompt::need(sizes, rows);
        if prompt.bytes() != need {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the pass buffers for passes of {rows} rows hold {} bytes, their fit check \
                     counted {need}",
                    prompt.bytes()
                ),
            ));
        }
        Ok(prompt)
    }

    /// The most ids a pass takes.
    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    /// The slots' up rows back to zero. A placed pass's up launch skips the
    /// slots the host serves and the quantize after it reads every slot's
    /// row, so rows a faulted pass left non-finite fault the next valid one.
    pub(super) fn clear_expert_up(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.b.up.zero_async(stream)?;
        Ok(())
    }

    /// The device bytes the pass's buffers hold.
    pub(super) fn bytes(&self) -> usize {
        let b = &self.b;
        let f = [
            &self.x[0], &self.x[1], &b.kh, &b.vh, &b.probs, &b.weights, &b.lat, &b.up, &b.down,
            &b.moe, &b.sh_up, &b.sh_relu, &b.sh, &b.sum, &self.hsum,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>();
        f + b.buf.bytes()
            + b.words.num_bytes()
            + b.ids.num_bytes()
            + b.sel.num_bytes()
            + b.zero_ids.num_bytes()
            + b.slots.iter().map(|s| s.2.device_bytes()).sum::<usize>()
            + b.arms.runs.iter().map(Q8Act::device_bytes).sum::<usize>()
            + b.arms
                .runs32
                .iter()
                .map(|a| a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes())
                .sum::<usize>()
            + b.arms.wide.iter().map(|a| a.1.bytes()).sum::<usize>()
            + b.arms.route.bytes()
            + b.ex.routes.iter().map(|r| r.1.bytes()).sum::<usize>()
            + b.ex.acts.iter().map(|a| a.2.bytes()).sum::<usize>()
            + b.ex.acts32.iter().map(|a| a.2.bytes()).sum::<usize>()
            + b.split.num_bytes()
            + self.mamba2.device_bytes()
    }

    /// The word index the pass's ids start at.
    fn ids_at(n_kv: usize) -> usize {
        P_ROWS + n_kv
    }
}

/// The q8_1 slot form of `cols` columns of `k` values in `slots`.
fn slot_form(
    slots: &mut [(usize, usize, Q8Act)],
    k: usize,
    cols: usize,
) -> Result<&mut Q8Act, GpuError> {
    slots
        .iter_mut()
        .find(|s| s.0 == k && s.1 >= cols)
        .map(|s| &mut s.2)
        .ok_or_else(|| GpuError::shape(WHAT, format!("no q8_1 slot form of {cols} columns of {k}")))
}

/// What every launch of a pass reads.
struct Cx<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    wk: WideKernels<'a>,
    g32: &'a Gemm32Kernels,
    kgemv: &'a KGemvKernels,
    routers: &'a Routers,
    iq: &'a QuantSelKernels,
    m: usize,
}

/// `y = W · x` for site `s` over the pass's `m` rows of `x` (`s.k` values a
/// row), `rows` output values a row, token-major. Up to [`COLS`] rows runs of
/// up to [`COLS`] columns, each its family's multi-column gemv (the step's
/// sums; an NVFP4 site at a width the q8_1 form does not take, the `_sel` gemv
/// over the run's q8 blocks of 32, [`site::nvfp4_sel32_gemv`]); a Q8_0 or
/// NVFP4 site past [`COLS`] rows is one 32-value GEMM over the pass's table
/// ([`site::gemm`]), on the rows' q8 blocks of 32. A type the pass does not
/// run is refused by name.
fn project_rows(
    cx: &Cx<'_>,
    arms: &mut Arms,
    layer: usize,
    (s, rows): (&Site, usize),
    x: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (gpu, f) = (cx.gpu, size_of::<f32>());
    let key = (s.ty, cx.w, s.name.as_str());
    if matches!(s.ty, SiteTy::Q8_0 | SiteTy::Nvfp4) && cx.m > COLS {
        let a =
            arms.wide.iter_mut().find(|a| a.0 == s.k).ok_or_else(|| {
                GpuError::shape(WHAT, format!("no q8 blocks of 32 of width {}", s.k))
            })?;
        // Both GEMMs read the q8 blocks of 32, quantized as a Q8_0 site's are.
        a.1.quantize(
            gpu,
            &cx.wk,
            gpu.layer_sink(layer)?,
            x,
            cx.m,
            &[SiteTy::Q8_0],
        )?;
        return site::gemm(
            gpu,
            &cx.wk,
            (s.ty, cx.w, SiteWeight::Resident(&s.name)),
            rows,
            &a.1.input(x),
            (&arms.route, GemmInput::PerSlot),
            cx.m,
            y,
        );
    }
    if s.ty == SiteTy::Nvfp4 && s.reads() == Form::Q8Blocks32 {
        // At most COLS rows, one run: the form quantizes its COLS columns (the
        // pass's rows and, past them, rows the buffer's cut holds) and the
        // gemv writes its COLS columns of `rows` values inside the buffer's cut.
        let act = arms
            .runs32
            .iter_mut()
            .find(|a| a.k() == s.k)
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("no q8 blocks of 32 run form of width {}", s.k),
                )
            })?;
        let fault = gpu.layer_sink(layer)?;
        gpu.q5().enqueue_quantize_q8(gpu.stream(), x, act, fault)?;
        return site::nvfp4_sel32_gemv(gpu, cx.kgemv, key, act, fault, y);
    }
    let mut c0 = 0;
    while c0 < cx.m {
        let c = (cx.m - c0).min(COLS);
        let xw = Window::<f32>::of(x, c0 * s.k * f, c * s.k)?;
        match s.ty {
            SiteTy::Q8_0 => {
                let mut yw = WindowMut::<f32>::of_mut(y, c0 * rows * f, c * rows)?;
                site::gemv(gpu, cx.g32, key, &xw, c, &mut yw)?;
            }
            SiteTy::Nvfp4 => {
                let act = arms.runs.iter_mut().find(|a| a.k() == s.k).ok_or_else(|| {
                    GpuError::shape(WHAT, format!("no q8_1 run form of width {}", s.k))
                })?;
                gpu.quantize_q8_1_cols_at(x, c0 * s.k, act, c, gpu.layer_sink(layer)?)?;
                // The gemv runs the form's COLS columns: the run's output
                // window is COLS rows, inside the buffer's cut.
                let mut yw = WindowMut::<f32>::of_mut(y, c0 * rows * f, COLS * rows)?;
                site::kgemv(gpu, cx.kgemv, key, act, gpu.layer_sink(layer)?, &mut yw)?;
            }
            other => {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{} is {other}: the pass runs Q8_0 and NVFP4 sites", s.name),
                ));
            }
        }
        c0 += c;
    }
    Ok(())
}

/// The resident stack `name` a `_sel` launch reads.
fn stack<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<u32>, GpuError> {
    match w.resident(WHAT, name)? {
        DevWeight::KQuant { w, .. } => Ok(w),
        _ => Err(GpuError::tensor(
            WHAT,
            name,
            "a weight in the file's blocks",
        )),
    }
}

/// Where a layer's card slots read their experts from.
#[derive(Clone, Copy)]
enum Places<'a> {
    /// Every expert of the layer is resident, in the router's id order: the
    /// ids are the places.
    Ids,
    /// The plan's share of the layer is resident: `sel` holds each slot's
    /// place in the stacks (or [`HOST`](crate::hybrid::HOST)), and `map` the
    /// layer's expert to place, which the GEMMs' route table is built from.
    Cut { map: &'a DeviceBuffer<u32> },
}

/// Layer `l`'s router over the pass's `m` rows of `xn` (the layer's normed
/// rows): every expert's score, the kept ids and their weights, and the last
/// row's kept in the layer's own results.
fn moe_route(
    cx: &Cx<'_>,
    p: &mut PassBufs,
    (l, ml, sh): (usize, &mut MoeLayer, MoeShape),
) -> Result<(), GpuError> {
    let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    cx.routers.enqueue_rows(
        stream,
        cx.gpu,
        cx.g32,
        ml.route.n_expert(),
        w.f32_tensor(WHAT, &ml.router)?,
        &p.buf.xn,
        w.f32_buf(WHAT, &ml.bias)?,
        sh.scale,
        m,
        (&mut p.probs, &mut p.ids, &mut p.weights),
        fault,
    )?;
    ml.route
        .enqueue_row_from(stream, (&p.probs, &p.ids, &p.weights), m - 1)
}

/// Layer `l`'s routed experts over the pass's `TOP_K · m` slots into `p.up`
/// and `p.down` (the up and the squared-ReLU'd down rows of every slot a card
/// expert serves), from the experts' input rows (the latent rows `p.lat`, or
/// the normed rows where the file has no latent): for IQ2_XXS stacks past
/// [`COLS`] rows the grouped IQ2_XXS GEMMs, up to them the `_sel` launches;
/// for 32-value stacks the 32-value GEMMs at any row count. A slot the host
/// serves ([`Places::Cut`]) is neither read nor written.
fn moe_card_experts(
    cx: &Cx<'_>,
    p: &mut PassBufs,
    (l, ml, sh): (usize, &MoeLayer, MoeShape),
    places: Places<'_>,
) -> Result<(), GpuError> {
    let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    let slots = sh.top_k * m;
    let x = if ml.latent_down.is_some() {
        &p.lat
    } else {
        &p.buf.xn
    };
    if ml.relu2_ff().is_some() {
        return experts_gemm32(
            cx,
            &mut p.ex,
            (fault, ml, sh.width),
            (&p.ids, places),
            (x, &mut p.up, &mut p.down),
        );
    }
    // The grouped GEMM's IQ2_XXS entry, where the build has one.
    if let Some(ty) = GemmWeight::from_ggml(GgmlType::IQ2_XXS)
        .ok()
        .filter(|_| m > COLS)
    {
        return experts_gemm(
            cx,
            &mut p.ex,
            (fault, ml, sh.width, ty),
            (&p.ids, places),
            (x, &mut p.up, &mut p.down),
        );
    }
    let sel = match places {
        Places::Ids => &p.ids,
        Places::Cut { .. } => &p.sel,
    };
    {
        let act = slot_form(&mut p.slots, sh.width, m)?;
        gpu.quantize_q8_1_cols_at(x, 0, act, m, fault)?;
        let up = SelDown {
            w: stack(w, ml.up()?)?,
            act,
            sel,
            n_slots: slots,
            rows_per_expert: ml.ff,
        };
        cx.iq
            .enqueue_gemv_sel_cols(stream, GgmlType::IQ2_XXS, &up, sh.top_k, fault, &mut p.up)?;
    }
    let act = slot_form(&mut p.slots, ml.ff, slots)?;
    gpu.quantize_q8_1_cols_at_relu_sqr(&p.up, 0, act, slots, fault)?;
    let down = SelDown {
        w: stack(w, ml.down()?)?,
        act,
        sel,
        n_slots: slots,
        rows_per_expert: sh.width,
    };
    cx.iq
        .enqueue_gemv_sel(stream, GgmlType::IQ2_XXS, &down, fault, &mut p.down)
}

/// Layer `l`'s shared expert over the pass's `m` rows of `xn` into `p.sh`:
/// past [`COLS`] rows its GEMMs, up to them the one-expert `_sel` launches.
fn moe_shared(
    cx: &Cx<'_>,
    p: &mut PassBufs,
    (l, ml, sh): (usize, &MoeLayer, MoeShape),
    hidden: usize,
) -> Result<(), GpuError> {
    let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
    let stream = gpu.stream();
    let fault = gpu.layer_sink(l)?;
    match &ml.shared {
        // The Q8_0 sites: the dense projections, over the pass's rows.
        Shared::Site { up, down } => {
            project_rows(
                cx,
                &mut p.arms,
                l,
                (up, sh.shared_ff),
                &p.buf.xn,
                &mut p.sh_up,
            )?;
            gpu.elem()
                .enqueue_relu_sqr(stream, &p.sh_up, m * sh.shared_ff, &mut p.sh_relu)?;
            project_rows(cx, &mut p.arms, l, (down, hidden), &p.sh_relu, &mut p.sh)
        }
        // The IQ2_XXS stacks: the grouped GEMM past [`COLS`] rows, the
        // one-expert `_sel` launches up to them.
        Shared::Sel { up, down } => {
            if let Some(ty) = GemmWeight::from_ggml(GgmlType::IQ2_XXS)
                .ok()
                .filter(|_| m > COLS)
            {
                return shared_gemm(
                    cx,
                    (&mut p.ex, &p.arms.route),
                    (fault, ml, sh.shared_ff, hidden, ty),
                    &p.buf.xn,
                    (&mut p.sh_up, &mut p.sh),
                );
            }
            {
                let act = slot_form(&mut p.slots, hidden, m)?;
                gpu.quantize_q8_1_cols_at(&p.buf.xn, 0, act, m, fault)?;
                let up = SelDown {
                    w: stack(w, up)?,
                    act,
                    sel: &p.zero_ids,
                    n_slots: m,
                    rows_per_expert: sh.shared_ff,
                };
                cx.iq
                    .enqueue_gemv_sel(stream, GgmlType::IQ2_XXS, &up, fault, &mut p.sh_up)?;
            }
            let act = slot_form(&mut p.slots, sh.shared_ff, m)?;
            gpu.quantize_q8_1_cols_at_relu_sqr(&p.sh_up, 0, act, m, fault)?;
            let down = SelDown {
                w: stack(w, down)?,
                act,
                sel: &p.zero_ids,
                n_slots: m,
                rows_per_expert: hidden,
            };
            cx.iq
                .enqueue_gemv_sel(stream, GgmlType::IQ2_XXS, &down, fault, &mut p.sh)
        }
    }
}

/// One latent MoE layer, every expert resident, over the pass's `m` rows of
/// `xn` (the layer's normed rows), the layer's delta into `p.sum`: the step's
/// order (`moe.rs`'s module doc) at `m` rows.
fn moe_rows(
    cx: &Cx<'_>,
    p: &mut PassBufs,
    (l, ml, sh): (usize, &mut MoeLayer, MoeShape),
    hidden: usize,
) -> Result<(), GpuError> {
    let (gpu, m) = (cx.gpu, cx.m);
    let stream = gpu.stream();
    moe_route(cx, p, (l, ml, sh))?;
    if let Some(site) = &ml.latent_down {
        project_rows(cx, &mut p.arms, l, (site, sh.width), &p.buf.xn, &mut p.lat)?;
    }
    moe_card_experts(cx, p, (l, ml, sh), Places::Ids)?;
    gpu.elem().enqueue_weighted_sum(
        stream,
        &p.down,
        &p.weights,
        sh.width,
        sh.top_k as u32,
        m,
        &mut p.moe,
    )?;
    moe_shared(cx, p, (l, ml, sh), hidden)?;
    // The layer's delta: the latent up's projection, or the experts' sum
    // itself where the file has no latent.
    match &ml.latent_up {
        Some(site) => {
            project_rows(cx, &mut p.arms, l, (site, hidden), &p.moe, &mut p.buf.delta)?;
            gpu.elem()
                .enqueue_add(stream, &p.buf.delta, &p.sh, m * hidden, &mut p.sum)
        }
        None => gpu
            .elem()
            .enqueue_add(stream, &p.moe, &p.sh, m * hidden, &mut p.sum),
    }
}

/// The GEMM form of `cols` columns of `k` values in `acts`.
fn gemm_form(
    acts: &mut [(usize, usize, GemmAct)],
    k: usize,
    cols: usize,
) -> Result<&mut GemmAct, GpuError> {
    acts.iter_mut()
        .find(|a| a.0 == k && a.1 >= cols)
        .map(|a| &mut a.2)
        .ok_or_else(|| GpuError::shape(WHAT, format!("no GEMM form of {cols} columns of {k}")))
}

/// The routed experts over the pass's `TOP_K · m` slots past [`COLS`] rows:
/// the slots' table from `ids` (through the layer's map to the card's places
/// under [`Places::Cut`], a slot the host serves in no tile), the up as one
/// grouped IQ2_XXS GEMM on the latent rows `lat` (each row's `TOP_K` slots
/// reading its column), the squared ReLU, the down as one GEMM on each slot's
/// own row.
fn experts_gemm(
    cx: &Cx<'_>,
    ex: &mut ExpertArms,
    (fault, ml, latent, ty): (FaultSink, &MoeLayer, usize, GemmWeight),
    (ids, places): (&DeviceBuffer<u32>, Places<'_>),
    (lat, up, down): (
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
) -> Result<(), GpuError> {
    let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
    let stream = gpu.stream();
    let slots = ml.top_k * m;
    let route = fill_route(cx, &mut ex.routes, ml, (ids, places), fault)?;
    {
        let a = gemm_form(&mut ex.acts, latent, m)?;
        gpu.enqueue_quantize_gemm(lat, m, a, fault)?;
        cx.wk.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty,
                w: stack(w, ml.up()?)?,
                rows_per_expert: ml.ff,
                act: a,
                route,
                input: GemmInput::Shared { top_k: ml.top_k },
                y: up,
            },
        )?;
    }
    let a = gemm_form(&mut ex.acts, ml.ff, slots)?;
    gpu.enqueue_quantize_gemm_relu_sqr(up, slots, a, fault)?;
    cx.wk.gemm.enqueue_gemm(
        stream,
        GemmArgs {
            ty,
            w: stack(w, ml.down()?)?,
            rows_per_expert: latent,
            act: a,
            route,
            input: GemmInput::PerSlot,
            y: down,
        },
    )
}

/// Layer `ml`'s routed table over the pass's `TOP_K · m` slots, filled from
/// `ids` (through the layer's map to the card's places under
/// [`Places::Cut`], a slot the host serves in no tile): the table of the
/// layer's card experts in `routes`, or a named refusal where the load made
/// none.
fn fill_route<'r>(
    cx: &Cx<'_>,
    routes: &'r mut [(usize, GemmRoute)],
    ml: &MoeLayer,
    (ids, places): (&DeviceBuffer<u32>, Places<'_>),
    fault: FaultSink,
) -> Result<&'r GemmRoute, GpuError> {
    let stream = cx.gpu.stream();
    let slots = ml.top_k * cx.m;
    let n = ml.n_card;
    let route = routes
        .iter_mut()
        .find(|r| r.0 == n)
        .map(|r| &mut r.1)
        .ok_or_else(|| GpuError::shape(WHAT, format!("no routed table of {n} experts")))?;
    match places {
        Places::Ids => cx.wk.gemm.enqueue_route(stream, ids, slots, route, fault)?,
        Places::Cut { map } => cx
            .g32
            .enqueue_route_remap(stream, ids, map, slots, route, fault)?,
    }
    Ok(route)
}

/// The routed experts on 32-value stacks over the pass's `TOP_K · m` slots,
/// at any row count: the slots' table ([`fill_route`]), the up as one
/// 32-value GEMM on the experts' input rows `x` (each row's `TOP_K` slots
/// reading its column), the squared ReLU folded into the down's q8 blocks of
/// 32, the down as one 32-value GEMM on each slot's own row. Each stack is
/// read by the GEMM the kernel table names for its type ([`StackW::route`]).
fn experts_gemm32(
    cx: &Cx<'_>,
    ex: &mut ExpertArms,
    (fault, ml, width): (FaultSink, &MoeLayer, usize),
    (ids, places): (&DeviceBuffer<u32>, Places<'_>),
    (x, up, down): (
        &DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
) -> Result<(), GpuError> {
    let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
    let stream = gpu.stream();
    let slots = ml.top_k * m;
    let route = fill_route(cx, &mut ex.routes, ml, (ids, places), fault)?;
    {
        let a = gemm32_form(&mut ex.acts32, width, m)?;
        cx.g32.enqueue_quantize_gemm32(stream, x, m, a, fault)?;
        cx.g32.enqueue_gemm32(
            stream,
            Gemm32Args {
                w: gemm32_stack(w, ml.up()?)?,
                rows_per_expert: ml.ff,
                act: a,
                route,
                input: GemmInput::Shared { top_k: ml.top_k },
                y: up,
            },
        )?;
    }
    let a = gemm32_form(&mut ex.acts32, ml.ff, slots)?;
    cx.g32
        .enqueue_quantize_gemm32_relu_sqr(stream, up, slots, a, fault)?;
    cx.g32.enqueue_gemm32(
        stream,
        Gemm32Args {
            w: gemm32_stack(w, ml.down()?)?,
            rows_per_expert: width,
            act: a,
            route,
            input: GemmInput::PerSlot,
            y: down,
        },
    )
}

/// The 32-value GEMM form of `cols` columns of `k` values in `acts`.
fn gemm32_form(
    acts: &mut [(usize, usize, GemmAct32)],
    k: usize,
    cols: usize,
) -> Result<&mut GemmAct32, GpuError> {
    acts.iter_mut()
        .find(|a| a.0 == k && a.1 >= cols)
        .map(|a| &mut a.2)
        .ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!("no 32-value GEMM form of {cols} columns of {k}"),
            )
        })
}

/// The resident routed stack `name` as the 32-value GEMM reads it: the file's
/// blocks as words, or Q5_0 packed, or NVFP4's planes, of a type the kernel
/// table names a 32-value GEMM for.
fn gemm32_stack<'w>(w: &'w Weights, name: &str) -> Result<Gemm32Weight<'w>, GpuError> {
    let (ty, stack) = StackW::resident(WHAT, w, name)?;
    match stack.route(ty) {
        Some(RouteW::Blocks32(g)) => Ok(g),
        _ => Err(GpuError::tensor(
            WHAT,
            name,
            "a stack of a type the 32-value GEMM reads",
        )),
    }
}

/// The shared expert over the pass's `m` rows of `xn` past [`COLS`] rows:
/// its up and down each one IQ2_XXS GEMM over the pass's one-expert table
/// `dense`, the squared ReLU between.
fn shared_gemm(
    cx: &Cx<'_>,
    (ex, dense): (&mut ExpertArms, &GemmRoute),
    (fault, ml, shared_ff, hidden, ty): (FaultSink, &MoeLayer, usize, usize, GemmWeight),
    xn: &DeviceBuffer<f32>,
    (up, y): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
    let stream = gpu.stream();
    let Shared::Sel {
        up: up_stack,
        down: down_stack,
    } = &ml.shared
    else {
        return Err(GpuError::state(
            WHAT,
            "a shared expert of q8_0 sites (the grouped GEMM is the iq2_xxs stack's)",
        ));
    };
    {
        let a = gemm_form(&mut ex.acts, hidden, m)?;
        gpu.enqueue_quantize_gemm(xn, m, a, fault)?;
        cx.wk.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty,
                w: stack(w, up_stack)?,
                rows_per_expert: shared_ff,
                act: a,
                route: dense,
                input: GemmInput::PerSlot,
                y: up,
            },
        )?;
    }
    let a = gemm_form(&mut ex.acts, shared_ff, m)?;
    gpu.enqueue_quantize_gemm_relu_sqr(up, m, a, fault)?;
    cx.wk.gemm.enqueue_gemm(
        stream,
        GemmArgs {
            ty,
            w: stack(w, down_stack)?,
            rows_per_expert: hidden,
            act: a,
            route: dense,
            input: GemmInput::PerSlot,
            y,
        },
    )
}

/// A pass's layer walk: what every layer of the pass reads and writes. The
/// residual stream alternates between the two buffers `x`, `cur` the one the
/// next layer reads.
struct PassWalk<'a> {
    cx: Cx<'a>,
    sh: Shape,
    moe: MoeShape,
    /// The query heads' width a row, `n_head · HEAD`.
    q_width: usize,
    layers: &'a mut [Layer],
    x: &'a mut [DeviceBuffer<f32>; 2],
    b: &'a mut PassBufs,
    mamba: &'a Mamba2Kernels,
    mamba2: &'a mut Mamba2PassScratch,
    flash: &'a FlashGqaPrefill,
    cur: usize,
}

impl PassWalk<'_> {
    /// Layer `l` whole over the pass's rows, its residual into the next
    /// stream: a Mamba-2 layer, an attention layer, or a MoE layer with every
    /// expert resident.
    fn layer(&mut self, l: usize) -> Result<(), GpuError> {
        let PassWalk {
            cx,
            sh,
            moe,
            q_width,
            layers,
            x,
            b,
            mamba,
            mamba2,
            flash,
            cur,
        } = self;
        let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
        let (sh, moe, q_width) = (*sh, *moe, *q_width);
        let n_kv = sh.n_kv;
        let stream = gpu.stream();
        let f = size_of::<f32>();
        let [x0, x1] = &mut **x;
        let (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>) =
            if *cur == 0 { (x0, x1) } else { (x1, x0) };
        let fault = gpu.layer_sink(l)?;
        let b = &mut **b;
        let delta = match &mut layers[l] {
            Layer::Mamba(mm) => {
                let g = sh.mamba.geom();
                let (inner, conv_dim) = (g.d_inner, sh.mamba.conv_dim());
                let ld = sh.mamba.in_proj_width();
                let buf = &mut b.buf;
                norm_rows((gpu, w), (sh.eps, sh.hidden, m), &mm.norm, src, &mut buf.xn)?;
                project_rows(
                    cx,
                    &mut b.arms,
                    l,
                    (&mm.ssm_in, ld),
                    &buf.xn,
                    &mut buf.zxbcdt,
                )?;
                mamba.enqueue_pass(
                    stream,
                    crate::mamba2::StepsArgs {
                        shape: sh.mamba,
                        u: m,
                        rows: &buf.zxbcdt,
                        ld,
                        z0: 0,
                        x0: inner,
                        dt0: inner + conv_dim,
                        dt_bias: w.f32_buf(WHAT, &mm.dt_bias)?,
                        a: w.f32_buf(WHAT, &mm.a)?,
                        d: w.f32_buf(WHAT, &mm.d)?,
                        conv_w: w.f32_buf(WHAT, &mm.conv_w)?,
                        conv_b: w.f32_buf(WHAT, &mm.conv_b)?,
                        norm: Some(w.f32_buf(WHAT, &mm.ssm_norm)?),
                        eps: sh.eps,
                        pos: &buf.pos,
                        pos_at: 0,
                        fault,
                        y: &mut buf.y,
                        ring: &mut mm.store.ring,
                        state: &mut mm.store.state,
                        scratch: mamba2,
                    },
                    PassScan::Ssd,
                )?;
                project_rows(
                    cx,
                    &mut b.arms,
                    l,
                    (&mm.ssm_out, sh.hidden),
                    &buf.y,
                    &mut buf.delta,
                )?;
                &buf.delta
            }
            Layer::Attn(a) => {
                let kv_width = n_kv * HEAD;
                let buf = &mut b.buf;
                norm_rows((gpu, w), (sh.eps, sh.hidden, m), &a.norm, src, &mut buf.xn)?;
                project_rows(cx, &mut b.arms, l, (&a.q, q_width), &buf.xn, &mut buf.q)?;
                project_rows(cx, &mut b.arms, l, (&a.k, kv_width), &buf.xn, &mut buf.k)?;
                project_rows(cx, &mut b.arms, l, (&a.v, kv_width), &buf.xn, &mut buf.v)?;
                for h in 0..n_kv {
                    let shape = Pitched {
                        width: HEAD * f,
                        rows: m,
                        dev_pitch: kv_width * f,
                        host_pitch: HEAD * f,
                    };
                    dtod_2d(
                        stream,
                        (&buf.k, h * HEAD * f),
                        (&mut b.kh, h * m * HEAD * f),
                        shape,
                    )?;
                    dtod_2d(
                        stream,
                        (&buf.v, h * HEAD * f),
                        (&mut b.vh, h * m * HEAD * f),
                        shape,
                    )?;
                }
                let append = PlaneAppend {
                    words: &b.words,
                    first_word: P_ROWS,
                    heads: n_kv,
                    m,
                    fault,
                };
                let attn = gpu.flash();
                let (kh, vh) = (&b.kh, &b.vh);
                attn.enqueue_kv_append_planes(
                    stream,
                    &append,
                    |h| Window::<f32>::of(kh, h * m * HEAD * f, m * HEAD),
                    &mut a.kc,
                )?;
                attn.enqueue_kv_append_planes(
                    stream,
                    &append,
                    |h| Window::<f32>::of(vh, h * m * HEAD * f, m * HEAD),
                    &mut a.vc,
                )?;
                flash.enqueue_split(
                    stream,
                    GqaPrefillArgs {
                        q: &buf.q,
                        kc: a.kc.buf(),
                        vc: a.vc.buf(),
                        n_keys: &buf.n_keys,
                        scale: 1.0 / (HEAD as f32).sqrt(),
                        n_head: q_width / HEAD,
                        n_kv,
                        ctx: sh.ctx,
                        t: m,
                        fault,
                        y: &mut buf.attn,
                    },
                    &mut b.split,
                )?;
                project_rows(
                    cx,
                    &mut b.arms,
                    l,
                    (&a.o, sh.hidden),
                    &buf.attn,
                    &mut buf.delta,
                )?;
                &b.buf.delta
            }
            Layer::Moe(ml) => {
                norm_rows(
                    (gpu, w),
                    (sh.eps, sh.hidden, m),
                    &ml.norm,
                    src,
                    &mut b.buf.xn,
                )?;
                moe_rows(cx, b, (l, ml, moe), sh.hidden)?;
                &b.sum
            }
        };
        residual_rows(gpu, m * sh.hidden, src, delta, dst)?;
        *cur = 1 - *cur;
        Ok(())
    }

    /// Hosted layer `at.layer`'s front: the norm, the router, the latent down,
    /// the download of the latent rows, the weights and the ids to the host
    /// ([`BatchLeg::key`]'s exchange), and each routed slot's place in the
    /// card's stacks (module doc).
    fn hosted_front(
        &mut self,
        side: &PlacedSide,
        leg: &mut BatchLeg<'_, HostRun>,
        at: At,
    ) -> Result<(), GpuError> {
        let l = at.layer;
        let PassWalk {
            cx,
            sh,
            moe,
            layers,
            x,
            b,
            cur,
            ..
        } = self;
        let (gpu, w, m) = (cx.gpu, cx.w, cx.m);
        let stream = gpu.stream();
        let Layer::Moe(ml) = &mut layers[l] else {
            return Err(GpuError::state(WHAT, "a hosted layer is a MoE layer"));
        };
        let b = &mut **b;
        norm_rows(
            (gpu, w),
            (sh.eps, sh.hidden, m),
            &ml.norm,
            &x[*cur],
            &mut b.buf.xn,
        )?;
        moe_route(cx, b, (l, ml, *moe))?;
        // The rows the host's experts read: the latent the file's down
        // projection makes, or the normed rows where the file has no latent.
        if let Some(site) = &ml.latent_down {
            project_rows(cx, &mut b.arms, l, (site, moe.width), &b.buf.xn, &mut b.lat)?;
        }
        let lat: &DeviceBuffer<f32> = if ml.latent_down.is_some() {
            &b.lat
        } else {
            &b.buf.xn
        };
        let key = leg.key(at);
        leg.hybrid()
            .enqueue_download_pitched(stream, [lat, &b.weights], &b.ids, moe.top_k, key)?;
        side.enqueue_places(
            stream,
            (l, ml.route.n_expert(), moe.top_k),
            (&b.ids, &mut b.sel, m),
            gpu.layer_sink(l)?,
        )
    }

    /// Hosted layer `l`'s shadow, the card's work under the host's union: the
    /// card's slots over their places when the layer holds an expert there,
    /// and the shared expert.
    fn hosted_shadow(&mut self, side: &PlacedSide, l: usize) -> Result<(), GpuError> {
        let PassWalk {
            cx,
            sh,
            moe,
            layers,
            b,
            ..
        } = self;
        let Layer::Moe(ml) = &layers[l] else {
            return Err(GpuError::state(WHAT, "a hosted layer is a MoE layer"));
        };
        if ml.n_card > 0 {
            let map = side.map_of(l)?;
            moe_card_experts(cx, b, (l, ml, *moe), Places::Cut { map })?;
        }
        moe_shared(cx, b, (l, ml, *moe), sh.hidden)
    }

    /// Hosted layer `l`'s back: the card slots' weighted sum and the host's
    /// sums, the latent up, the shared expert and the residual.
    fn hosted_back(
        &mut self,
        side: &PlacedSide,
        leg: &BatchLeg<'_, HostRun>,
        l: usize,
    ) -> Result<(), GpuError> {
        let PassWalk {
            cx,
            sh,
            moe,
            layers,
            x,
            b,
            cur,
            ..
        } = self;
        let (gpu, m) = (cx.gpu, cx.m);
        let stream = gpu.stream();
        let Layer::Moe(ml) = &layers[l] else {
            return Err(GpuError::state(WHAT, "a hosted layer is a MoE layer"));
        };
        let b = &mut **b;
        let hsum = leg.hsum();
        let merged: &DeviceBuffer<f32> = if ml.n_card > 0 {
            side.acc().enqueue_card_acc(
                stream,
                &CardAcc {
                    down: &b.down,
                    w: &b.weights,
                    sel: &b.sel,
                    n: moe.width,
                    m,
                    n_card: ml.n_card,
                    n_used: moe.top_k,
                    w_pitch: moe.top_k,
                    fault: gpu.layer_sink(l)?,
                },
                &mut b.moe,
            )?;
            // The latent rows are spent once the card's slots ran: they take
            // the card's sum and the host's.
            gpu.elem()
                .enqueue_add(stream, &b.moe, hsum, m * moe.width, &mut b.lat)?;
            &b.lat
        } else {
            hsum
        };
        // The layer's delta: the latent up's projection, or the experts' sum
        // itself where the file has no latent.
        match &ml.latent_up {
            Some(site) => {
                project_rows(
                    cx,
                    &mut b.arms,
                    l,
                    (site, sh.hidden),
                    merged,
                    &mut b.buf.delta,
                )?;
                gpu.elem()
                    .enqueue_add(stream, &b.buf.delta, &b.sh, m * sh.hidden, &mut b.sum)?;
            }
            None => {
                gpu.elem()
                    .enqueue_add(stream, merged, &b.sh, m * sh.hidden, &mut b.sum)?;
            }
        }
        let [x0, x1] = &mut **x;
        let (src, dst): (&DeviceBuffer<f32>, &mut DeviceBuffer<f32>) =
            if *cur == 0 { (x0, x1) } else { (x1, x0) };
        residual_rows(gpu, m * sh.hidden, src, &b.sum, dst)?;
        *cur = 1 - *cur;
        Ok(())
    }
}

/// A placed load's pass `(1, m, Batch)` through the host tier's batch port
/// (module doc): the layers with no host share run whole in their front, a
/// hosted one in its three parts around the host's union.
struct PlacedPass<'a, 'b> {
    walk: PassWalk<'a>,
    side: &'b PlacedSide,
}

impl<'a> LayerProgram for PlacedPass<'a, '_> {
    type Port = BatchLeg<'a, HostRun>;

    fn host_leg(&self, at: At) -> bool {
        self.side.hosted(at.layer)
    }

    fn front(&mut self, leg: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if self.side.hosted(at.layer) {
            self.walk.hosted_front(self.side, leg, at)
        } else {
            self.walk.layer(at.layer)
        }
    }

    fn shadow(&mut self, _: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if self.side.hosted(at.layer) {
            self.walk.hosted_shadow(self.side, at.layer)?;
        }
        Ok(())
    }

    fn back(&mut self, leg: &mut BatchLeg<'a, HostRun>, at: At) -> Result<(), GpuError> {
        if self.side.hosted(at.layer) {
            self.walk.hosted_back(self.side, leg, at.layer)?;
        }
        Ok(())
    }
}

impl Body {
    /// One pass of `ids` (at most the prompt buffers' rows) from position
    /// `pos`, layer-major; with `head`, the head over the pass's last row.
    /// Each MoE layer's results ([`Body::routing`]) are its last row's. A
    /// placed load walks the layers through the host tier's batch port
    /// ([`PlacedPass`]).
    pub(super) fn enqueue_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (ids, pos): (&[u32], u32),
        head: Option<&mut Head>,
    ) -> Result<(), GpuError> {
        let m = ids.len();
        let (sh, moe) = (self.sh, self.moe);
        let n_kv = sh.n_kv;
        let stream = gpu.stream();
        let p = &mut self.prompt;
        let q_width = p.q_width;
        let rows = p.rows;
        if m == 0 || m > rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {m} ids; the prompt buffers hold 1..={rows}"),
            ));
        }
        let Prompt {
            x, b, hsum, mamba2, ..
        } = p;
        b.host_words[P_POS] = pos;
        for h in 0..n_kv {
            b.host_words[P_ROWS + h] = h
                .checked_mul(sh.ctx)
                .and_then(|r| r.checked_add(pos as usize))
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| {
                    GpuError::shape(
                        WHAT,
                        format!("row {h} · {} + {pos} does not fit a u32 row word", sh.ctx),
                    )
                })?;
        }
        let at = Prompt::ids_at(n_kv);
        b.host_words[at..at + m].copy_from_slice(ids);
        b.words.copy_from_host(stream, &b.host_words)?;
        if m > COLS {
            self.k
                .gemm
                .enqueue_route_dense(stream, m, &mut b.arms.route, gpu.unlabelled_sink())?;
        }
        {
            let (qs, d) = w.q8_planes(WHAT, &self.embed)?;
            let ids_w = Window::<u32>::of(&b.words, at * size_of::<u32>(), m)?;
            let pos0 = Window::<u32>::of(&b.words, P_POS * size_of::<u32>(), 1)?;
            self.q38.enqueue_embed_rows(
                stream,
                EmbedArgs {
                    table: EmbedRows::Q8_0 { qs, d },
                    ids: &ids_w,
                    pos0: &pos0,
                    first: 0,
                    fault: gpu.unlabelled_sink(),
                    y: &mut x[0],
                    pos: &mut b.buf.pos,
                    n_keys: &mut b.buf.n_keys,
                },
            )?;
        }
        let cx = Cx {
            gpu,
            w,
            wk: WideKernels {
                gemm: &self.k.gemm,
                g32: Some(&self.k.g32),
            },
            g32: &self.k.g32,
            kgemv: &self.k.kgemv,
            routers: &self.mk.routers,
            iq: &self.mk.iq,
            m,
        };
        let n = self.layers.len();
        let mut walk = PassWalk {
            cx,
            sh,
            moe,
            q_width,
            layers: &mut self.layers,
            x,
            b,
            mamba: &self.k.mamba,
            mamba2,
            flash: &self.prefill,
            cur: 0,
        };
        match self.placed.as_mut() {
            None => {
                for l in 0..n {
                    walk.layer(l)?;
                }
            }
            Some(Placed { hybrid, side }) => {
                let mut leg = BatchLeg::new(stream, hybrid, hsum, rows);
                let o = Overlap {
                    units: 1,
                    cols: m,
                    port: PortKind::Batch,
                };
                let mut pass = PlacedPass { walk, side: &*side };
                sched::walk(o, n, &mut leg, &mut pass)?;
                walk = pass.walk;
            }
        }
        let cur = walk.cur;
        if let Some(head) = head {
            let f = size_of::<f32>();
            let row = Window::<f32>::of(&p.x[cur], (m - 1) * sh.hidden * f, sh.hidden)?;
            head.input_mut().copy_from_device_async(&row, stream)?;
            head.enqueue(gpu, w)?;
        }
        Ok(())
    }
}

impl GpuModel<Body> {
    /// The rows one pass of [`GpuModel::prefill`] takes: the size the load
    /// chose from the card's room ([`choose_pass_rows`]).
    pub fn pass_rows(&self) -> Result<usize, GpuError> {
        Ok(self.body(WHAT)?.prompt.rows())
    }

    /// [`GpuModel::prefill_with`] in passes of the most ids the prompt
    /// buffers hold: the rows the load chose ([`pass_rows`]).
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32, GpuError> {
        let rows = self.body(WHAT)?.prompt.rows();
        self.prefill_with(ids, rows)
    }

    /// Feed `ids` through the chain in passes of up to `rows` ids (module
    /// doc) and return the greedy next token after the last one. Positions
    /// continue from wherever the model stands, and the stores after it are
    /// those of the ids run as steps. `rows` past the prompt buffers' rows,
    /// or 0, and an empty `ids` are refused by name. The passes run eager in
    /// either step mode; a fault any pass raised is read behind the last
    /// pass's head (`GpuModel::run_rows`, the call-end read).
    pub fn prefill_with(&mut self, ids: &[u32], rows: usize) -> Result<u32, GpuError> {
        if ids.is_empty() {
            return Err(GpuError::shape(WHAT, "empty token slice"));
        }
        let cap = self.body(WHAT)?.prompt.rows();
        if rows == 0 || rows > cap {
            return Err(GpuError::shape(
                WHAT,
                format!("passes of {rows} ids; the prompt buffers hold 1..={cap}"),
            ));
        }
        let pos0 = self.pos();
        self.check_pos(pos0 + launch_u32(WHAT, "ids", ids.len())? - 1, WHAT)?;
        let n = ids.len().div_ceil(rows);
        let mut next = None;
        for (i, pass) in ids.chunks(rows).enumerate() {
            let last = i + 1 == n;
            next = self.run_rows(pass.len(), WHAT, |gpu, w, body, head, pos| {
                body.enqueue_pass(gpu, w, (pass, pos), last.then_some(head))?;
                Ok(last)
            })?;
        }
        next.ok_or(GpuError::state(WHAT, "a token read after the last pass"))
    }
}
