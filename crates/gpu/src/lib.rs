//! bloomery-gpu — the stage-0 CUDA kernels packaged as a library.
//!
//! The kernels are the algorithms of `crates/q3k-gemv/src/main.rs` with K
//! (values per activation column) as a launch argument instead of the
//! stage-0 constant 2048; the arithmetic bodies live in `cores::` as
//! device-callable functions (docs/gpu-design.md decision 6). The crate
//! must be compiled with `cargo oxide`; the codegen backend embeds the
//! compiled device bundle in a `.oxart` member of this crate's rlib, and
//! the anchor reference emitted inside `kernels::load` pulls that member
//! into any final binary that calls this API.
//!
//! K enters the kernels as `n_sb` (super-blocks per row, K = 256·n_sb) plus
//! the iteration counts the q8 scratch layout needs (`half_it` =
//! ceil(n_sb/2) two-super-block groups, `quad_it` = ceil(n_sb/4)
//! four-super-block groups); `Q8Act` owns the pairing between them. The
//! launch contracts bind every buffer length as products of those scalars —
//! the grammar has no division, so the ceils travel as arguments.

#![allow(
    rustdoc::private_intra_doc_links,
    reason = "public docs name crate-private constants and kernels on purpose: the crate is \
              not published, its docs are read in source or with --document-private-items, \
              and the link keeps the name checkable"
)]

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

pub mod arch;
pub mod cores;
pub mod elem;
pub mod fault;
pub mod flash;
pub mod flash_gqa;
pub mod flash_gqa_prefill;
pub mod fused;
pub mod gemm;
pub(crate) mod graph;
pub mod head;
pub mod hybrid;
pub mod iq;
pub mod model;
pub mod moe_fused;
pub mod mxfp4;
pub mod probe;
pub mod q4k_sel;
pub mod q5;
pub mod q6k_sel;
pub mod q8f32;
pub mod rope_neox;
pub mod rope_table;
pub mod route_core;
pub mod router;
pub(crate) mod tensor;
pub mod weights;

pub use fault::{FAULT_NONE, FAULT_WORDS, Fault, FaultSink, FaultSite, LAYER_HEAD, LAYER_NONE};
pub use graph::{Branch, FLAG_WAIT_OPS, Graph, HostFlags, NodeInfo, capturing};
pub use model::GpuModel;
/// The engine over the DeepSeek-V2-Lite chain — what `GpuModel` alone named
/// before the skeleton became generic over its architecture.
pub type Deepseek2Model = GpuModel<arch::deepseek2::Body>;
/// The qwen3moe engine: the whole model on one card.
pub type Qwen3moeModel = GpuModel<arch::qwen3moe::Body>;
pub use tensor::{DeviceTensor, PartedBuffer, Q8Act, window};

/// Host-side failure: context creation, module loading, device allocation,
/// launch, capture, or copy-back.
///
/// The variants are the kinds a reader has to tell apart when a gate prints
/// one: a driver refusal, a caller's geometry mistake, a missing tensor or
/// metadata key, a model object that is not ready, or a failure another
/// crate already described. Every variant names the entry point it came
/// from, because the same geometry complaint reaches a dozen kernels.
#[derive(Debug)]
pub enum GpuError {
    /// A CUDA driver entry point returned a failure code. `op` names the
    /// driver call our wrapper made; a plain `?` on a cuda-core call has no
    /// name of ours to add.
    Driver {
        op: Option<&'static str>,
        source: cuda_core::DriverError,
    },
    /// The embedded device module could not be loaded into the context.
    Module(Box<cuda_host::EmbeddedModuleError>),
    /// A launch configuration does not meet a kernel's launch contract.
    Launch(Box<cuda_core::LaunchContractError>),
    /// The GGUF file could not be read.
    Load(Box<gguf::LoadError>),
    /// An argument breaks an entry point's geometry contract: `what` is the
    /// entry point, `detail` the lengths or shapes that disagree.
    Shape { what: &'static str, detail: String },
    /// A GGUF metadata key the loader needs is absent from the file.
    Metadata {
        what: &'static str,
        key: &'static str,
    },
    /// A tensor is not resident, or is resident in a device format other
    /// than the one the caller reads it as.
    Tensor {
        what: &'static str,
        name: String,
        need: &'static str,
    },
    /// The model does not hold what the call needs: a stage, that stage's
    /// residency, an output head, or a captured graph.
    State {
        what: &'static str,
        missing: &'static str,
    },
    /// Planning, metadata or dequantization carried up from the model crate.
    Model(Box<::model::ModelError>),
    /// The host's plan for the card was refused — its placement, a step's
    /// plan, a step's host rows — by `what`. `source` is the planner's own
    /// error, whatever crate made it, so the chain keeps it.
    Plan {
        what: &'static str,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// The file declares an architecture the model crate knows but this
    /// crate has no engine for; the string is the name the file declares.
    UnsupportedArch(String),
    /// The card and the hybrid host tier lost step with each other: a go that
    /// did not land in time, a card that ran ahead of the host, a handoff of
    /// another sequence number or layer. The tier has released every pending
    /// wait and refuses to serve again, so the model is unusable, not the
    /// call's arguments wrong.
    Protocol { what: &'static str, detail: String },
    /// A kernel raised the card's fault word (crate::fault): an input it has
    /// no defined answer for, at `fault`'s layer and site. The step that
    /// read it back returned no token, and the state it wrote — caches,
    /// rings — holds what the fault condemned. `behind` is the call's other
    /// error when the word was read after it failed (a later token refused,
    /// a launch failed, a host refusal the card did not make at or before
    /// that layer): the fault is the call's error because it poisons the
    /// model, and that error's text rides along.
    Fault {
        what: &'static str,
        fault: Fault,
        behind: Option<Box<GpuError>>,
    },
    /// A model refused a step because an earlier step raised `fault`; its
    /// state is condemned until a `reset`.
    Poisoned { what: &'static str, fault: Fault },
}

impl GpuError {
    /// A geometry contract broken by the caller of `what`.
    pub(crate) fn shape(what: &'static str, detail: impl Into<String>) -> GpuError {
        GpuError::Shape {
            what,
            detail: detail.into(),
        }
    }

    /// A tensor `what` looked up that is absent or in the wrong format;
    /// `need` completes "is not …".
    pub(crate) fn tensor(
        what: &'static str,
        name: impl Into<String>,
        need: &'static str,
    ) -> GpuError {
        GpuError::Tensor {
            what,
            name: name.into(),
            need,
        }
    }

    /// A model object `what` needs that the caller has not built yet.
    pub(crate) fn state(what: &'static str, missing: &'static str) -> GpuError {
        GpuError::State { what, missing }
    }

    /// A metadata key `what` reads that the file does not carry.
    pub(crate) fn metadata(what: &'static str, key: &'static str) -> GpuError {
        GpuError::Metadata { what, key }
    }

    /// The card and the host tier `what` serves out of step, described by
    /// `detail`.
    pub(crate) fn protocol(what: &'static str, detail: impl Into<String>) -> GpuError {
        GpuError::Protocol {
            what,
            detail: detail.into(),
        }
    }

    /// The fault `fault` that `what` read back from the card's fault word.
    #[must_use]
    pub fn fault(what: &'static str, fault: Fault) -> GpuError {
        GpuError::Fault {
            what,
            fault,
            behind: None,
        }
    }

    /// A host plan `what` asked for, refused with `source`.
    pub fn plan(
        what: &'static str,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> GpuError {
        GpuError::Plan {
            what,
            source: source.into(),
        }
    }
}

/// `v` as the `u32` a kernel scalar or a launch dimension takes. `as` would
/// wrap a count past `u32::MAX` into a smaller geometry the launch contract
/// may still accept, so a value that does not fit is a `Shape` error of the
/// entry point `what`, naming the argument `name`.
pub fn launch_u32(what: &'static str, name: &'static str, v: usize) -> Result<u32, GpuError> {
    u32::try_from(v)
        .map_err(|_| GpuError::shape(what, format!("{name} = {v} does not fit the kernel's u32")))
}

/// Split width [`Q3K_SPLIT_WIDTH`] and the walk length
/// [`Q3K_SPLIT_MIN_ITERS`] from which a Q3_K row is split: split-K pays only
/// where a launch ends on a long, mostly empty last wave of long walks; a
/// grid of many waves, or one already at the bandwidth floor, only gains the
/// combine's cost. The default width is 1 — no launch is split and every
/// bit is the unsplit kernel's; the split is the `BLOOMERY_Q3K_SPLIT=2` arm
/// of a same-binary A/B, since on V4.1 only `wo_b` qualifies and its
/// predicted gain sits under the ruler's resolution while the changed sum
/// order moves the greedy trajectory.
pub const Q3K_SPLIT_WIDTH: usize = 1;
/// See [`Q3K_SPLIT_WIDTH`].
pub const Q3K_SPLIT_MIN_ITERS: usize = 16;

/// Warps [`Gpu::enqueue_gemv_q3k`] splits a Q3_K row of `n_sb` super-blocks
/// across: the width W when the row walk has at least T two-super-block
/// iterations and W divides them (equal shares), else 1 — the unsplit
/// `q3k_gemv`. A function of `n_sb` alone, never of the row or column count:
/// a joined launch, a gate's row-capped upload and an m-column pass must
/// write each row the bits its own one-column launch writes, and the split
/// width is part of those bits.
///
/// W and T default to [`Q3K_SPLIT_WIDTH`] and [`Q3K_SPLIT_MIN_ITERS`];
/// `BLOOMERY_Q3K_SPLIT=1|2|4|8` sets W (1 keeps every launch unsplit) and
/// `BLOOMERY_Q3K_SPLIT_ITERS=<T>` sets T, for same-binary A/B arms. Read
/// once, at first use: the width fixes a captured graph's grid and its bits.
/// A value that is set but unusable panics rather than falling back.
pub fn q3k_splits(n_sb: usize) -> usize {
    let (width, min_iters) = q3k_split_rule();
    let iters = n_sb.div_ceil(2);
    if width > 1 && iters >= min_iters && iters.is_multiple_of(width) {
        width
    } else {
        1
    }
}

/// The `(W, T)` of [`q3k_splits`], read once.
fn q3k_split_rule() -> (usize, usize) {
    static RULE: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    *RULE.get_or_init(|| {
        let width = match std::env::var("BLOOMERY_Q3K_SPLIT") {
            Err(_) => Q3K_SPLIT_WIDTH,
            Ok(v) => match v.parse::<usize>() {
                Ok(n @ (1 | 2 | 4 | 8)) => n,
                _ => panic!("BLOOMERY_Q3K_SPLIT={v} is not 1, 2, 4 or 8"),
            },
        };
        let min_iters = match std::env::var("BLOOMERY_Q3K_SPLIT_ITERS") {
            Err(_) => Q3K_SPLIT_MIN_ITERS,
            Ok(v) => match v.parse::<usize>() {
                Ok(n) if n >= 1 => n,
                _ => panic!("BLOOMERY_Q3K_SPLIT_ITERS={v} is not a positive iteration count"),
            },
        };
        (width, min_iters)
    })
}

/// Columns a group of [`ColGroups`] holds at most: the m-column cores'.
pub const COL_GROUP: usize = 8;

/// Columns `col0 .. col0 + cols` of an activation cut into the column groups
/// a grouped launch runs ([`Gpu::enqueue_gemv_q3k_groups`] and the grouped
/// entries built on it): the first group of `lead` columns, then groups of
/// [`COL_GROUP`], the last the rest. Each group is the m-column launch over
/// its own columns — column `c` bit for bit the one-column launch on that
/// column (`cores::q3k_row_dot`) — and a row-major output keeps group `g`'s
/// `rows × m` block at `rows · c0`, `c0` its first column counted from
/// `col0`: the groups' own outputs laid end to end. A prompt batch cuts its
/// tokens into chunks the same way, so each chunk's block is the one its own
/// launch writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColGroups {
    col0: usize,
    lead: usize,
    cols: usize,
}

impl ColGroups {
    /// The groups of `cols` columns from `col0` whose first holds `lead`;
    /// refused unless `1 <= lead <= min(COL_GROUP, cols)`.
    pub fn new(col0: usize, lead: usize, cols: usize) -> Result<ColGroups, GpuError> {
        if lead == 0 || lead > COL_GROUP.min(cols) {
            return Err(GpuError::shape(
                "ColGroups::new",
                format!(
                    "a first group of {lead} columns out of {cols}: it holds 1..={COL_GROUP} of them"
                ),
            ));
        }
        Ok(ColGroups { col0, lead, cols })
    }

    /// The first column, counted in the activation.
    #[must_use]
    pub fn col0(&self) -> usize {
        self.col0
    }

    /// The first group's columns.
    #[must_use]
    pub fn lead(&self) -> usize {
        self.lead
    }

    /// Columns over every group.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Groups: the first, then `⌈(cols − lead) / COL_GROUP⌉`.
    #[must_use]
    pub fn count(&self) -> usize {
        col_group_count(self.lead, self.cols)
    }

    /// Group `g`'s first column counted from `col0`, and its columns (0 past
    /// the last group).
    #[must_use]
    pub fn group(&self, g: usize) -> (usize, usize) {
        col_group(g, self.lead, self.cols)
    }
}

/// Group `g` of the column groups whose first holds `lead` of `cols`
/// columns ([`ColGroups`]): its first column and its column count, 0 for a
/// group past the last. Integer arithmetic only: kernels and hosts read the
/// groups through this one rule.
#[inline(always)]
#[must_use]
pub fn col_group(g: usize, lead: usize, cols: usize) -> (usize, usize) {
    let c0 = if g == 0 {
        0
    } else {
        lead + COL_GROUP * (g - 1)
    };
    let len = if g == 0 { lead } else { COL_GROUP };
    (c0, len.min(cols.saturating_sub(c0)))
}

/// Groups of `cols` columns whose first holds `lead` ([`col_group`]): the
/// first, then `⌈(cols − lead) / COL_GROUP⌉`. `1 <= lead <= cols`.
#[inline(always)]
#[must_use]
pub fn col_group_count(lead: usize, cols: usize) -> usize {
    1 + (cols - lead).div_ceil(COL_GROUP)
}

/// The group column `c` belongs to among the column groups whose first holds
/// `lead` ([`col_group`]).
#[inline(always)]
#[must_use]
pub fn col_group_of(c: usize, lead: usize) -> usize {
    if c < lead {
        0
    } else {
        1 + (c - lead) / COL_GROUP
    }
}

/// A row's `m` sums from the per-lane partials of an m-column core:
/// `warp::reduce_sum_f32` per column, the one-column kernels' tree, the
/// columns past `m` left at 0.0. `m` must be warp-uniform — a launch-wide or
/// group-wide constant in every caller.
#[inline(always)]
#[must_use]
pub fn col_sums(f: [f32; 8], m: usize) -> [f32; 8] {
    let mut s = [0.0f32; 8];
    s[0] = warp::reduce_sum_f32(f[0]);
    if m > 1 {
        s[1] = warp::reduce_sum_f32(f[1]);
    }
    if m > 2 {
        s[2] = warp::reduce_sum_f32(f[2]);
    }
    if m > 3 {
        s[3] = warp::reduce_sum_f32(f[3]);
    }
    if m > 4 {
        s[4] = warp::reduce_sum_f32(f[4]);
    }
    if m > 5 {
        s[5] = warp::reduce_sum_f32(f[5]);
    }
    if m > 6 {
        s[6] = warp::reduce_sum_f32(f[6]);
    }
    if m > 7 {
        s[7] = warp::reduce_sum_f32(f[7]);
    }
    s
}

/// Lane 0's store of a row's `m` values: `y[base + c·stride] = v[c]` for
/// `c < m`, one guarded store per column with a constant index (a loop over
/// `c` would index `v` at run time and put it in a local depot).
///
/// # Safety
///
/// `1 <= m <= 8`, every slot `base + c·stride` for `c < m` inside `y`, and
/// no other thread of the launch writes any of them.
#[inline(always)]
pub unsafe fn store_cols(
    y: &mut DisjointSlice<f32>,
    base: usize,
    stride: usize,
    m: usize,
    v: [f32; 8],
) {
    // SAFETY: each store is guarded by c < m, so its slot is one this fn's
    // contract puts inside y and gives to this thread alone.
    unsafe {
        *y.get_unchecked_mut(base) = v[0];
        if m > 1 {
            *y.get_unchecked_mut(base + stride) = v[1];
        }
        if m > 2 {
            *y.get_unchecked_mut(base + 2 * stride) = v[2];
        }
        if m > 3 {
            *y.get_unchecked_mut(base + 3 * stride) = v[3];
        }
        if m > 4 {
            *y.get_unchecked_mut(base + 4 * stride) = v[4];
        }
        if m > 5 {
            *y.get_unchecked_mut(base + 5 * stride) = v[5];
        }
        if m > 6 {
            *y.get_unchecked_mut(base + 6 * stride) = v[6];
        }
        if m > 7 {
            *y.get_unchecked_mut(base + 7 * stride) = v[7];
        }
    }
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Driver {
                op: Some(op),
                source,
            } => write!(f, "{op}: {source}"),
            GpuError::Driver { op: None, source } => write!(f, "{source}"),
            GpuError::Module(e) => write!(f, "{e}"),
            GpuError::Launch(e) => write!(f, "{e}"),
            GpuError::Load(e) => write!(f, "{e}"),
            GpuError::Shape { what, detail } => write!(f, "{what}: {detail}"),
            GpuError::Metadata { what, key } => write!(f, "{what}: metadata key {key} missing"),
            GpuError::Tensor { what, name, need } => write!(f, "{what}: {name} is not {need}"),
            GpuError::State { what, missing } => write!(f, "{what}: {missing}"),
            GpuError::Model(e) => write!(f, "{e}"),
            GpuError::Plan { what, source } => write!(f, "{what}: {source}"),
            GpuError::UnsupportedArch(name) => write!(f, "unsupported architecture {name:?}"),
            GpuError::Protocol { what, detail } => write!(f, "{what}: {detail}"),
            GpuError::Fault {
                what,
                fault,
                behind,
            } => {
                write!(f, "{what}: device fault at {fault}")?;
                match behind {
                    Some(e) => write!(f, "; the call also failed: {e}"),
                    None => Ok(()),
                }
            }
            GpuError::Poisoned { what, fault } => write!(
                f,
                "{what}: the model is poisoned by an earlier device fault at {fault}; reset() \
                 clears it"
            ),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuError::Driver { source, .. } => Some(source),
            GpuError::Module(e) => Some(&**e),
            GpuError::Launch(e) => Some(&**e),
            GpuError::Load(e) => Some(&**e),
            GpuError::Model(e) => Some(&**e),
            GpuError::Plan { source, .. } => Some(&**source),
            GpuError::Fault {
                behind: Some(e), ..
            } => Some(&**e),
            _ => None,
        }
    }
}

impl From<cuda_core::DriverError> for GpuError {
    fn from(source: cuda_core::DriverError) -> GpuError {
        GpuError::Driver { op: None, source }
    }
}

impl From<cuda_host::EmbeddedModuleError> for GpuError {
    fn from(e: cuda_host::EmbeddedModuleError) -> GpuError {
        GpuError::Module(Box::new(e))
    }
}

impl From<cuda_core::LaunchContractError> for GpuError {
    fn from(e: cuda_core::LaunchContractError) -> GpuError {
        GpuError::Launch(Box::new(e))
    }
}

impl From<gguf::LoadError> for GpuError {
    fn from(e: gguf::LoadError) -> GpuError {
        GpuError::Load(Box::new(e))
    }
}

impl From<::model::ModelError> for GpuError {
    fn from(e: ::model::ModelError) -> GpuError {
        GpuError::Model(Box::new(e))
    }
}

/// One 128-value q8_1 block of column `col` of the activation plane based at
/// `x0`, quantized and stored in the three gemv permutations plus the group
/// sums and the block scale. The body of `kernels::q3k_quantize_q8_1` — the
/// whole numeric contract of q8_1 activations lives here once, and the
/// single- and pair-destination kernels are the two ways of reaching it, so
/// their bytes agree by construction rather than by inspection.
///
/// Lives outside the `#[cuda_module]` for the reason `cores` does: a
/// device-callable body. It is not in `cores` because its stores need the
/// module's `DisjointSlice` outputs, which no core takes.
///
/// A block holding a non-finite value has no q8_1 form:
/// [`q8_1_quant_vals`] refuses it (NaN scale, zero codes and sums), and this
/// raises `site` on `fault` for it.
///
/// SAFETY: the caller guarantees `col < m_cols`, `b < 2 * n_sb`, the launch
/// contract's bounds on `x` at base `x0` and on the five outputs, and that
/// all 32 lanes of one warp enter with the same `(col, b)` — the collectives
/// below are warp-wide.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub fn q8_1_quant_block(
    x: &[f32],
    x0: usize,
    col: usize,
    b: usize,
    n_sb: usize,
    half_it: u32,
    quad_it: u32,
    lane: usize,
    q3: &mut DisjointSlice<u64>,
    q4: &mut DisjointSlice<u32>,
    q6: &mut DisjointSlice<u32>,
    s8: &mut DisjointSlice<i32>,
    d8: &mut DisjointSlice<f32>,
    fault: FaultSink,
    site: FaultSite,
) {
    // Lane covers the four consecutive values 4*lane .. 4*lane+3 of the
    // block; the warp max over the four per-lane maxima is the block amax
    // (no cross-lane byte packing needed, unlike the 32-value geometry where
    // one value per lane forced two shuffle_downs).
    let base = x0 + col * 256 * n_sb + 128 * b + 4 * lane;
    // SAFETY: base + 3 < x0 + (col+1)*256*n_sb <= x.len() by the caller's
    // contract.
    let v = unsafe {
        [
            *x.get_unchecked(base),
            *x.get_unchecked(base + 1),
            *x.get_unchecked(base + 2),
            *x.get_unchecked(base + 3),
        ]
    };
    // SAFETY: this fn's own contract gives `col < m_cols`, `b < 2*n_sb`, the
    // output bounds and the warp-uniform `(col, b)` that
    // [`q8_1_quant_vals`] requires, and `v` holds exactly its values
    // `128*b + 4*lane .. +3` of column `col` — the rest of its contract.
    let refused =
        unsafe { q8_1_quant_vals(v, col, b, n_sb, half_it, quad_it, lane, q3, q4, q6, s8, d8) };
    if refused && lane == 0 {
        fault.raise(site);
    }
}

/// The same 128-value q8_1 quantization from the four values already in a
/// lane's registers, for a producer that holds them. The partition is the
/// caller's to keep: the warp entering this must be the same 32 lanes
/// holding the same block's 128 values in the same order, because the amax
/// below is a warp collective and a scale computed over a different set of
/// values changes every byte quantized with it.
///
/// The one owner of the q8_1 refusal. A block holding a non-finite value
/// has no q8_1 form (a NaN lane would round to code 0 and the scale come
/// from the finite lanes, a plausible block), and `f32::max` drops a NaN, so
/// the amax cannot tell: each lane tests its four values and one ballot
/// refuses the block. A refused block is stored with a NaN scale and zero
/// codes and code sums, so every dot over it is NaN. The return says
/// whether this block was refused — the same on every lane — for the caller
/// to raise its own fault site; a caller that ignores it still stores no
/// plausible block. A finite block is stored as it always was.
///
/// SAFETY: as [`q8_1_quant_block`], and `v` must be values `128 * b + 4 *
/// lane .. +3` of column `col`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(crate) unsafe fn q8_1_quant_vals(
    v: [f32; 4],
    col: usize,
    b: usize,
    n_sb: usize,
    half_it: u32,
    quad_it: u32,
    lane: usize,
    q3: &mut DisjointSlice<u64>,
    q4: &mut DisjointSlice<u32>,
    q6: &mut DisjointSlice<u32>,
    s8: &mut DisjointSlice<i32>,
    d8: &mut DisjointSlice<f32>,
) -> bool {
    use crate::cores::{q3_slot, q4_slot, q6_slot, q8_quad};

    let (v0, v1, v2, v3) = (v[0], v[1], v[2], v[3]);
    let refused = warp::ballot(!crate::fault::quad_finite(v)) != 0;
    let amax = warp::reduce_max_f32(v0.abs().max(v1.abs()).max(v2.abs()).max(v3.abs()));
    let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let (word, quad) = q8_quad([v0, v1, v2, v3], d);
    let (word, quad, d) = if refused {
        (0, 0, f32::NAN)
    } else {
        (word, quad, d)
    };

    // v4 = this word's index in value order within the column: the word
    // covers values 128b + 4*lane .. +3, so v4 = 32b + lane (32 words per
    // 128-value block, 2*n_sb blocks per column). Each q*_slot maps it to
    // the load slot of the named gemv geometry — permutations per
    // 2-super-block (q3/q6) or 4-super-block (q4) group, host-verified
    // including this value-span tie (an earlier draft said 16b + lane, which
    // the load-site check alone could not catch because both sides then
    // agree on a wrong bijection over half the column).
    let v4 = (32 * b + lane) as u32;
    // Q3_K u64 pairing: the two fields of a gemv load PAIR (j, j^1) are
    // always held by quantize lanes lane and lane^8 of one block (v4 differs
    // only in bit 3), so q3_slot(v4) is a bijection onto the column's
    // 64*half_it u64 slots with v4's bit 3 selecting the half. All lanes run
    // the collective shuffle; the bit3-clear half stores.
    let g3 = q3_slot(v4);
    let partner = warp::shuffle_xor(word, 8);
    let cb = col * 64 * half_it as usize;
    let p4 = q4_slot(v4);
    let p6 = q6_slot(v4);
    let cu4 = col * 256 * quad_it as usize;
    let cu6 = col * 128 * half_it as usize;
    // SAFETY: q3_slot < 64*half_it, q4_slot < 256*quad_it and q6_slot <
    // 128*half_it per column (permutations of the column's value words onto
    // its group slots, host-verified bijections incl. the q3 pair check);
    // the three stores hit three distinct buffers, bit3-clear lanes of a
    // block write disjoint u64 positions, every lane its own u32 position.
    unsafe {
        if lane & 8 == 0 {
            *q3.get_unchecked_mut(cb + g3 as usize) = (word as u64) | ((partner as u64) << 32);
        }
        *q4.get_unchecked_mut(cu4 + p4 as usize) = word;
        *q6.get_unchecked_mut(cu6 + p6 as usize) = word;
    }

    // 32-value-group signed sums: butterfly over the lane-local quad sums
    // (masks 1, 2, 4); afterwards every lane holds its octet's total and
    // lanes 8k write group 4b + k of the column.
    let mut g = quad;
    g += warp::shuffle_xor(g as u32, 1) as i32;
    g += warp::shuffle_xor(g as u32, 2) as i32;
    g += warp::shuffle_xor(g as u32, 4) as i32;
    if lane & 7 == 0 {
        // SAFETY: group index 4b + lane/8 < 8*n_sb per column; s8 holds
        // m_cols*8*n_sb words and one lane writes each group.
        unsafe {
            *s8.get_unchecked_mut(col * 8 * n_sb + 4 * b + (lane >> 3)) = g;
        }
    }
    if lane == 0 {
        // SAFETY: lane 0 of each warp writes its own d8 slot.
        unsafe {
            *d8.get_unchecked_mut(col * 2 * n_sb + b) = d;
        }
    }
    refused
}

#[cuda_module]
mod kernels {
    use super::*;
    use crate::cores::{
        funnel16, half_to_f32, q3k_row_dot, q3k_row_dot_1col, q3k_row_dot_1col_span,
        q3k_row_dot_cols_span, q4k_row_dot, q4k_row_dot_1col, q6k_chain, q6k_dequant,
        q6k_sub_scale,
    };

    /// Global thread `g`'s warp in its block, the row that warp serves and
    /// its split index, for the split-K Q3_K entries: `1 << split_shift` warps
    /// to a row, consecutive warps, the row's split 0 first.
    #[inline(always)]
    fn q3k_split_warp(g: usize, split_shift: u32) -> (usize, usize, u32) {
        let warp_of = (g % 256) / 32;
        let sh = split_shift as usize;
        let row = ((g / 256) << (3 - sh)) + (warp_of >> sh);
        (warp_of, row, (warp_of & ((1 << sh) - 1)) as u32)
    }

    /// The split-K fold: the `1 << split_shift` warp sums of one row and one
    /// column, at `slots[first + s * stride]` for split s, added in split
    /// order, `((s_0 + s_1) + s_2) + …`, one plain f32 add each. This order is
    /// the gate.
    ///
    /// # Safety
    ///
    /// Every `slots.add(first + s * stride)` for `s < 1 << split_shift` is a
    /// written shared slot the caller's barrier has made visible.
    #[inline(always)]
    unsafe fn q3k_split_fold(
        slots: *const f32,
        first: usize,
        stride: usize,
        split_shift: u32,
    ) -> f32 {
        // SAFETY: split 0's slot, by this fn's contract.
        let mut acc = unsafe { *slots.add(first) };
        let mut s = 1usize;
        while s < (1usize << split_shift) {
            // SAFETY: split s's slot, by this fn's contract.
            acc += unsafe { *slots.add(first + s * stride) };
            s += 1;
        }
        acc
    }

    // Q3_K packing recap (decode verified against ggml's
    // `dequantize_row_q3_K` in the round-1 kernel at 1e-7):
    // super-block = hmask[32] @ +0, qs[64] @ +32, scales[12] @ +96, d @ +108.
    // Weight k within the super-block (k = 128c+32j+16h+8p+o) reads its low
    // 2 bits from qs byte 32c+16h+8p+o, field j, and its high bit from
    // hmask byte (k mod 32), bit 4c+j. Inverting: qs word w (bytes 4w..4w+3),
    // field j, covers the four CONSECUTIVE weights
    // k = 128*(w/8) + 32*j + 4*(w%8) + b — one dp4a per (word, field)
    // against u32 word (k/4) of the q8_1 activation, with the sub-block
    // scale (16 weights = one field-group of the word) applied per dp4a.

    /// Quantize f32 activations to q8_1: per 128-value block, d = amax/127 and
    /// int8 q = round(x/d). The m columns of `k = 256*n_sb` values each are
    /// read from base `x0`, so a caller can quantize a slice of a wider
    /// buffer without copying it. One warp per block; each lane quantizes
    /// its four consecutive values and writes the packed u32 word once per
    /// gemv lane geometry — the same bytes in the permutation that makes its format's
    /// gemv load instruction address 32 lane-consecutive words (one 128B L1
    /// line) instead of four strided clusters (four lines, four wavefronts
    /// per load — the M>1 marginal cost; the index identity
    /// p(old_slot) == new_slot is host-verified for every load site of all
    /// three formats, so the values are bit-identical to the single linear
    /// store). Q3_K goes one step further: its gemv reads the four words of
    /// an iteration as TWO u64 loads (field pairs share a slot; the pairing
    /// partner is always quantize lane lane^8 of the same block, so one
    /// shuffle builds the u64 — the load-issue count was the residual
    /// per-column cost after the permutation), so q3 is a u64 buffer and
    /// only the bit3-clear lanes store. Also emits s8: the signed-byte sum
    /// of each 32-value group, a 1-2-4 xor butterfly over lane-local quad
    /// sums — exactly the integer the q4k B chain's dp4a(0x01010101, qv)
    /// accumulated, so that chain becomes one i32 load. A 128-value block is
    /// one Q3_K half super-block, the exact span a gemv lane's four fields
    /// cover in one step: all four dp4a share this block's scale, so they
    /// chain in int behind ONE f32 FMA instead of one scale load + FMA per
    /// field.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= x0 + m_cols * 256 * n_sb,
            q3.len() >= m_cols * 64 * half_it,
            q4.len() >= m_cols * 256 * quad_it,
            q6.len() >= m_cols * 128 * half_it,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb
        )
    )]
    pub fn q3k_quantize_q8_1(
        x: &[f32],
        x0: u32,
        m_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        // One 32-thread block per 128-value quant block: blk is the BLOCK
        // index (global tid / 32), not the thread id.
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let blocks_per_col = 2 * n_sb; // 128-value blocks per column
        let total = m_cols as usize * blocks_per_col;
        if blk >= total {
            return;
        }
        let col = blk / blocks_per_col;
        let b = blk % blocks_per_col;
        let lane = warp::lane_id() as usize;
        // SAFETY: col < m_cols and b < 2*n_sb by the two lines above; the
        // launch contract carries the rest of `q8_1_quant_block`'s
        // preconditions, and the block index is warp-uniform.
        q8_1_quant_block(
            x,
            x0 as usize,
            col,
            b,
            n_sb,
            half_it,
            quad_it,
            lane,
            &mut q3,
            &mut q4,
            &mut q6,
            &mut s8,
            &mut d8,
            fault,
            FaultSite::QuantColumn,
        );
    }

    /// Two independent q8_1 quantizations of the SAME source buffer in one
    /// launch: blocks below `m_cols * 2 * n_sb` quantize the plane at `x0_a`
    /// into the `_a` outputs, the rest the plane at `x0_b` into `_b`. Both
    /// planes are `m_cols` columns of `256 * n_sb` values, so the grid is
    /// twice the single form's and the arm is a block-uniform branch — no
    /// warp splits, every collective inside the body still warp-wide.
    ///
    /// The body is `q8_1_quant_block` either way, so each output is the
    /// bytes its own `q3k_quantize_q8_1` launch would have written.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= x0_a + m_cols * 256 * n_sb,
            x.len() >= x0_b + m_cols * 256 * n_sb,
            q3a.len() >= m_cols * 64 * half_it,
            q4a.len() >= m_cols * 256 * quad_it,
            q6a.len() >= m_cols * 128 * half_it,
            s8a.len() >= m_cols * 8 * n_sb,
            d8a.len() >= m_cols * 2 * n_sb,
            q3b.len() >= m_cols * 64 * half_it,
            q4b.len() >= m_cols * 256 * quad_it,
            q6b.len() >= m_cols * 128 * half_it,
            s8b.len() >= m_cols * 8 * n_sb,
            d8b.len() >= m_cols * 2 * n_sb
        )
    )]
    pub fn q3k_quantize_q8_1_pair(
        x: &[f32],
        x0_a: u32,
        x0_b: u32,
        m_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3a: DisjointSlice<u64>,
        mut q4a: DisjointSlice<u32>,
        mut q6a: DisjointSlice<u32>,
        mut s8a: DisjointSlice<i32>,
        mut d8a: DisjointSlice<f32>,
        mut q3b: DisjointSlice<u64>,
        mut q4b: DisjointSlice<u32>,
        mut q6b: DisjointSlice<u32>,
        mut s8b: DisjointSlice<i32>,
        mut d8b: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let blocks_per_col = 2 * n_sb;
        let total = m_cols as usize * blocks_per_col;
        if blk >= 2 * total {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY (both arms): the arm's index is below `total`, so col <
        // m_cols and b < 2*n_sb; the launch contract bounds `x` at both
        // bases and each arm's five outputs. The arm is chosen by the block
        // index, so a warp never splits across it.
        if blk < total {
            let (col, b) = (blk / blocks_per_col, blk % blocks_per_col);
            q8_1_quant_block(
                x,
                x0_a as usize,
                col,
                b,
                n_sb,
                half_it,
                quad_it,
                lane,
                &mut q3a,
                &mut q4a,
                &mut q6a,
                &mut s8a,
                &mut d8a,
                fault,
                FaultSite::QuantColumn,
            );
        } else {
            let k = blk - total;
            let (col, b) = (k / blocks_per_col, k % blocks_per_col);
            q8_1_quant_block(
                x,
                x0_b as usize,
                col,
                b,
                n_sb,
                half_it,
                quad_it,
                lane,
                &mut q3b,
                &mut q4b,
                &mut q6b,
                &mut s8b,
                &mut d8b,
                fault,
                FaultSite::QuantColumn,
            );
        }
    }

    /// Q4_K packing recap (word arithmetic verified against ggml's
    /// `dequantize_row_q4_K` on a real attention-output row at 3.6e-8 before
    /// this kernel was written): super-block = d f16 @0, dmin f16 @2, scales[12]
    /// @4, qs[128] @16 — 144 bytes, always 4-aligned, no funnel needed.
    /// Sub-block s (32 consecutive weights, values 32s..32s+32) is nibble
    /// s&1 of qs bytes 32*(s>>1) .. +32: one nibble per byte, the sibling
    /// nibble in the same byte belongs to sub-block s^1. Scale s and min s
    /// unpack from the 12 scale bytes with ggml's get_scale_min_k4.
    /// The min offset forces a second dp4a chain: weight = d*sc*nib -
    /// dmin*mi, so the sub-block dot on quantized activations is
    /// d8*(d*sc*A + (8*d*sc - dmin*mi)*B) with A = dp4a(nib-8, q8) and
    /// B = dp4a(1s, q8) = sum q8 (the -8 offset moves 8*B between the
    /// chains). 16 dp4a cover 32 values — twice Q3_K's density, inherent to
    /// the nibble-sibling layout.
    ///
    /// Q4_K (K = 256*n_sb values, n_sb super-blocks per row) times ONE
    /// q8_1 activation column — the decode shape; `q4k_gemv_mcol` takes
    /// 1..=8. One warp per row. The warp covers FOUR super-blocks per
    /// iteration (lane L owns the 32-value sub-block s = L&7 of super-block
    /// 4*it + L>>3), so the row takes iters = ceil(n_sb/4) iterations; when
    /// n_sb is not a multiple of 4 the last iteration's high octets have no
    /// super-block and their lanes contribute nothing (no warp-collective op
    /// lives in the loop). Each lane keeps its whole sub-block local: the
    /// min-offset B chain reduces to one s8 group-sum load (the quantize
    /// kernel precomputed the exact integer the dp4a(0x01010101) chain
    /// accumulated). The per-row body is `cores::q4k_row_dot_1col`.
    ///
    /// The column count is not a parameter: registers are allocated per
    /// entry, and an entry holding both walks is given more than either
    /// needs alone, so the decode shape would run below the occupancy its
    /// own walk allows.
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
            w.len() >= n_rows * 36 * n_sb,
            q.len() >= 256 * iters,
            s8.len() >= 8 * n_sb,
            d8.len() >= 2 * n_sb,
            y.len() >= n_rows
        )
    )]
    pub fn q4k_gemv(
        w: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        n_rows: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let f0 = q4k_row_dot_1col(w, q, s8, d8, n_sb as usize, iters, row, 0, lane);
        let s0 = warp::reduce_sum_f32(f0);
        if lane == 0 {
            // SAFETY: only lane 0 writes; warp `row` owns y[row].
            unsafe {
                *y.get_unchecked_mut(row) = s0;
            }
        }
    }

    /// `q4k_gemv` over `m_cols` (1..=8) q8_1 activation columns, `y[r·m +
    /// c]`: one warp per row, the qs window decoded once per iteration into
    /// vi[8] for every column, guarded scalar accumulators. The per-row body
    /// is `cores::q4k_row_dot`, whose column c is `q4k_row_dot_1col` on that
    /// column bit for bit — so column c here is the `q4k_gemv` launch of
    /// column c.
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
            w.len() >= n_rows * 36 * n_sb,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= n_rows * m_cols,
            m_cols >= 1,
            m_cols <= 8
        )
    )]
    pub fn q4k_gemv_mcol(
        w: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        n_rows: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        let f = q4k_row_dot(
            w,
            q,
            s8,
            d8,
            n_sb as usize,
            iters,
            row,
            0,
            m_cols as usize,
            lane,
        );

        // Warp-uniform reduction (m is a launch-wide constant). Column c's
        // reduction runs only when m > c; every lane takes the same branch,
        // so the shuffles stay warp-collective.
        let s0 = warp::reduce_sum_f32(f[0]);
        let s1 = if m > 1 {
            warp::reduce_sum_f32(f[1])
        } else {
            0.0
        };
        let s2 = if m > 2 {
            warp::reduce_sum_f32(f[2])
        } else {
            0.0
        };
        let s3 = if m > 3 {
            warp::reduce_sum_f32(f[3])
        } else {
            0.0
        };
        let s4 = if m > 4 {
            warp::reduce_sum_f32(f[4])
        } else {
            0.0
        };
        let s5 = if m > 5 {
            warp::reduce_sum_f32(f[5])
        } else {
            0.0
        };
        let s6 = if m > 6 {
            warp::reduce_sum_f32(f[6])
        } else {
            0.0
        };
        let s7 = if m > 7 {
            warp::reduce_sum_f32(f[7])
        } else {
            0.0
        };
        if lane == 0 {
            // SAFETY: only lane 0 of each warp writes the disjoint segment
            // y[row*m .. row*m+m]: store c is guarded by m > c, so exactly
            // the first m slots of the row are touched.
            unsafe {
                let b = row * m;
                *y.get_unchecked_mut(b) = s0;
                if m > 1 {
                    *y.get_unchecked_mut(b + 1) = s1;
                }
                if m > 2 {
                    *y.get_unchecked_mut(b + 2) = s2;
                }
                if m > 3 {
                    *y.get_unchecked_mut(b + 3) = s3;
                }
                if m > 4 {
                    *y.get_unchecked_mut(b + 4) = s4;
                }
                if m > 5 {
                    *y.get_unchecked_mut(b + 5) = s5;
                }
                if m > 6 {
                    *y.get_unchecked_mut(b + 6) = s6;
                }
                if m > 7 {
                    *y.get_unchecked_mut(b + 7) = s7;
                }
            }
        }
    }

    /// Q6_K packing recap (word arithmetic verified against ggml's
    /// `dequantize_row_q6_K` on output.weight bytes at 5.2e-8): super-block
    /// = ql[128] @0, qh[64] @128, scales int8[16] @192, d f16 @208 — 210
    /// bytes, so odd super-blocks sit 2 mod 4 and every window takes the
    /// Q3_K 16-bit funnel. Sub-block s (16 consecutive weights, values
    /// 16s..16s+16, scale byte s signed) reads its low nibble from ql byte
    /// 64*(s>>3) + 16*(s&3) + i (nibble s&4 selects high) and its two high
    /// bits from qh byte 128 + 32*(s>>3) + 16*(s&1) + i, bit pair
    /// 2*((s>>1)&3) — the pair index is per HALF (qh's 2-bit fields repeat
    /// every 32 bytes), which is easy to get wrong. q6 - 32 in [-32,31]
    /// fits a signed byte, so one SWAR subtract folds the offset and the
    /// whole sub-block is a single dp4a chain — no B chain, no min term.
    ///
    /// Q6_K (K = 256*n_sb values, n_sb super-blocks per row) times q8_1
    /// activations, M <= 8. Same lane geometry as q3k_gemv: one warp per
    /// row, iteration covers two super-blocks (lanes 0..15 even, 16..31
    /// odd; iters = ceil(n_sb/2), an odd n_sb guards the last iteration's
    /// odd half), lane owns one 16-value sub-block = one scale = four dp4a
    /// per column.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 210 * n_sb,
            q.len() >= m_cols * 128 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q6k_gemv(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        n_rows: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        let n_sb = n_sb as usize;
        let row_bytes = 210 * n_sb;
        let q_col = 128 * iters as usize; // q8 words per column
        let d8_col = 2 * n_sb; // 128-value blocks per column

        let w16 = lane & 15; // sub-block within the super-block
        let half = lane >> 4; // 0: even super-block, 1: odd

        // Per-lane constants (see the packing comment above).
        let qlok = 16 * (w16 & 3); // ql word base, +16 per word, inside ql
        let qhok = 16 * (w16 & 1); // qh byte base, +16 per word, inside qh
        let nib_sh = ((w16 >> 2) & 1) as u32 * 4; // low nibble vs high
        let hib_sh = 2 * ((w16 >> 1) as u32 & 3); // per-half bit pair

        let mut f0 = 0.0f32;
        let mut f1 = 0.0f32;
        let mut f2 = 0.0f32;
        let mut f3 = 0.0f32;
        let mut f4 = 0.0f32;
        let mut f5 = 0.0f32;
        let mut f6 = 0.0f32;
        let mut f7 = 0.0f32;

        let mut it: u32 = 0;
        while it < iters {
            let sbp = ((it << 1) | half as u32) as usize;
            // The sbp guard makes a partial final iteration safe; always
            // true when n_sb is even.
            if sbp < n_sb {
                let base = row * row_bytes + sbp * 210; // byte offset
                // A super-block sits 0 or 2 mod 4 depending on the row's own
                // offset too (odd n_sb shifts every other row), so the
                // funnel select comes from the byte window, not from sbp.
                let par = (base >> 1) & 1;

                // ql window: 4 words at super-block byte 64*(w16>>3) + qlok.
                let lk = (base + 64 * (w16 >> 3) + qlok) >> 2;
                // SAFETY: the last window of the last row ends at byte
                // <= row*row_bytes + row_bytes, and the floored word index
                // stays inside the row's ceil(row_bytes/4) words (see the
                // scales window below for the tight bound).
                let (l0, l1, l2, l3, l4) = unsafe {
                    (
                        *w.get_unchecked(lk),
                        *w.get_unchecked(lk + 1),
                        *w.get_unchecked(lk + 2),
                        *w.get_unchecked(lk + 3),
                        *w.get_unchecked(lk + 4),
                    )
                };
                let ql0 = if par == 0 { l0 } else { funnel16(l0, l1) };
                let ql1 = if par == 0 { l1 } else { funnel16(l1, l2) };
                let ql2 = if par == 0 { l2 } else { funnel16(l2, l3) };
                let ql3 = if par == 0 { l3 } else { funnel16(l3, l4) };

                // qh window: 4 words at super-block byte 128 + 32*(w16>>3) + qhok.
                let hk = (base + 128 + 32 * (w16 >> 3) + qhok) >> 2;
                // SAFETY: same row bounds as the ql window; the qh section
                // ends at byte 192 of the super-block, before the scales.
                let (h0, h1, h2, h3, h4) = unsafe {
                    (
                        *w.get_unchecked(hk),
                        *w.get_unchecked(hk + 1),
                        *w.get_unchecked(hk + 2),
                        *w.get_unchecked(hk + 3),
                        *w.get_unchecked(hk + 4),
                    )
                };
                let qh0 = if par == 0 { h0 } else { funnel16(h0, h1) };
                let qh1 = if par == 0 { h1 } else { funnel16(h1, h2) };
                let qh2 = if par == 0 { h2 } else { funnel16(h2, h3) };
                let qh3 = if par == 0 { h3 } else { funnel16(h3, h4) };

                // scales (16 int8 at super-block byte 192) + d (f16 @208):
                // one 5-word window, k..k+4. Even: scales are words k..k+3
                // and d is the low half of word k+4. Odd: the window is
                // funneled and d is the HIGH half of word k+4 (the same
                // load f3 uses).
                let ak = (base + 192) >> 2;
                // SAFETY: ak+4 <= ceil((row+1)*row_bytes/4) - 1: for the
                // last super-block of the last row, ak+4 is at most the
                // buffer's final (possibly zero-padded) word; the launch
                // contract bounds 4*w.len() >= n_rows*210*n_sb.
                let (a0, a1, a2, a3, a4) = unsafe {
                    (
                        *w.get_unchecked(ak),
                        *w.get_unchecked(ak + 1),
                        *w.get_unchecked(ak + 2),
                        *w.get_unchecked(ak + 3),
                        *w.get_unchecked(ak + 4),
                    )
                };
                let (sw0, sw1, sw2, sw3, d_bits) = if par == 0 {
                    (a0, a1, a2, a3, (a4 & 0xffff) as u16)
                } else {
                    (
                        funnel16(a0, a1),
                        funnel16(a1, a2),
                        funnel16(a2, a3),
                        funnel16(a3, a4),
                        (a4 >> 16) as u16,
                    )
                };
                // Sub-block scale: byte w16 of the scales window, signed.
                let sc = q6k_sub_scale(&[sw0, sw1, sw2, sw3], w16);
                let drow = half_to_f32(d_bits);

                // q6 - 32 per byte: nibble | high-bits<<4, then one SWAR
                // subtract of 32 (borrow-free with the |0x80 bias).
                let vi = [
                    q6k_dequant(ql0, qh0, nib_sh, hib_sh),
                    q6k_dequant(ql1, qh1, nib_sh, hib_sh),
                    q6k_dequant(ql2, qh2, nib_sh, hib_sh),
                    q6k_dequant(ql3, qh3, nib_sh, hib_sh),
                ];

                // q8_1 words in the q6 permutation: word i of column c lives
                // at q_col*c + 128it + 32i + lane (host-verified identity
                // with the value-order slot the linear layout used); each
                // load is 32 lane-consecutive words across the warp.
                let qb = 128 * it as usize + lane;
                let d8b = 2 * sbp + (w16 >> 3);

                // Column 0 (always active): one dp4a chain, one FMA.
                {
                    // SAFETY: qb + 96 + 3 <= q_col - 1 — this lane's four
                    // q8 words are inside the column's q_col words (the
                    // permutation's group bound); d8b < d8_col by the sbp
                    // guard.
                    let (q0, q1, q2, q3, e0) = unsafe {
                        (
                            *q.get_unchecked(qb),
                            *q.get_unchecked(qb + 32),
                            *q.get_unchecked(qb + 64),
                            *q.get_unchecked(qb + 96),
                            *d8.get_unchecked(d8b),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f0 += (a as f32) * (e0 * drow * sc as f32);
                }
                // Columns 1..7, one launch-uniform guard per column so the
                // work scales with m (the amortization curve over m). Column c
                // reads q8 words at q_col*c + qb (+32 per word) and block
                // d8b + d8_col*c.
                // SAFETY: guard m > c means q.len() >= (c+1)*q_col >
                // q_col*c + qb + 99 and d8.len() >= (c+1)*d8_col >
                // d8b + d8_col*c, launch-uniform.
                if m > 1 {
                    let cb = q_col + qb;
                    let d8c = d8b + d8_col;
                    // SAFETY: m > 1, so q.len() >= 2*q_col > cb + 99 and
                    // d8.len() >= 2*d8_col > d8c.
                    let (q0, q1, q2, q3, e1) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f1 += (a as f32) * (e1 * drow * sc as f32);
                }
                if m > 2 {
                    let cb = 2 * q_col + qb;
                    let d8c = d8b + 2 * d8_col;
                    // SAFETY: m > 2, so q.len() >= 3*q_col > cb + 99 and
                    // d8.len() >= 3*d8_col > d8c.
                    let (q0, q1, q2, q3, e2) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f2 += (a as f32) * (e2 * drow * sc as f32);
                }
                if m > 3 {
                    let cb = 3 * q_col + qb;
                    let d8c = d8b + 3 * d8_col;
                    // SAFETY: m > 3, so q.len() >= 4*q_col > cb + 99 and
                    // d8.len() >= 4*d8_col > d8c.
                    let (q0, q1, q2, q3, e3) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f3 += (a as f32) * (e3 * drow * sc as f32);
                }
                if m > 4 {
                    let cb = 4 * q_col + qb;
                    let d8c = d8b + 4 * d8_col;
                    // SAFETY: m > 4, so q.len() >= 5*q_col > cb + 99 and
                    // d8.len() >= 5*d8_col > d8c.
                    let (q0, q1, q2, q3, e4) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f4 += (a as f32) * (e4 * drow * sc as f32);
                }
                if m > 5 {
                    let cb = 5 * q_col + qb;
                    let d8c = d8b + 5 * d8_col;
                    // SAFETY: m > 5, so q.len() >= 6*q_col > cb + 99 and
                    // d8.len() >= 6*d8_col > d8c.
                    let (q0, q1, q2, q3, e5) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f5 += (a as f32) * (e5 * drow * sc as f32);
                }
                if m > 6 {
                    let cb = 6 * q_col + qb;
                    let d8c = d8b + 6 * d8_col;
                    // SAFETY: m > 6, so q.len() >= 7*q_col > cb + 99 and
                    // d8.len() >= 7*d8_col > d8c.
                    let (q0, q1, q2, q3, e6) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f6 += (a as f32) * (e6 * drow * sc as f32);
                }
                if m > 7 {
                    let cb = 7 * q_col + qb;
                    let d8c = d8b + 7 * d8_col;
                    // SAFETY: m > 7, so q.len() >= 8*q_col > cb + 99 and
                    // d8.len() >= 8*d8_col > d8c.
                    let (q0, q1, q2, q3, e7) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f7 += (a as f32) * (e7 * drow * sc as f32);
                }
            }

            it += 1;
        }

        // Warp-uniform reduction (m is a launch-wide constant). Column c's
        // reduction runs only when m > c; every lane takes the same branch,
        // so the shuffles stay warp-collective, and the tail scales with m
        // exactly like the column bodies above.
        let s0 = warp::reduce_sum_f32(f0);
        if m == 1 {
            if lane == 0 {
                // SAFETY: only lane 0 writes; warp `row` owns y[row].
                unsafe {
                    *y.get_unchecked_mut(row) = s0;
                }
            }
        } else {
            let s1 = warp::reduce_sum_f32(f1);
            let s2 = if m > 2 { warp::reduce_sum_f32(f2) } else { 0.0 };
            let s3 = if m > 3 { warp::reduce_sum_f32(f3) } else { 0.0 };
            let s4 = if m > 4 { warp::reduce_sum_f32(f4) } else { 0.0 };
            let s5 = if m > 5 { warp::reduce_sum_f32(f5) } else { 0.0 };
            let s6 = if m > 6 { warp::reduce_sum_f32(f6) } else { 0.0 };
            let s7 = if m > 7 { warp::reduce_sum_f32(f7) } else { 0.0 };
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes the disjoint
                // segment y[row*m .. row*m+m]: store c is guarded by m > c,
                // so exactly the first m slots of the row are touched.
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = s0;
                    if m > 1 {
                        *y.get_unchecked_mut(b + 1) = s1;
                    }
                    if m > 2 {
                        *y.get_unchecked_mut(b + 2) = s2;
                    }
                    if m > 3 {
                        *y.get_unchecked_mut(b + 3) = s3;
                    }
                    if m > 4 {
                        *y.get_unchecked_mut(b + 4) = s4;
                    }
                    if m > 5 {
                        *y.get_unchecked_mut(b + 5) = s5;
                    }
                    if m > 6 {
                        *y.get_unchecked_mut(b + 6) = s6;
                    }
                    if m > 7 {
                        *y.get_unchecked_mut(b + 7) = s7;
                    }
                }
            }
        }
    }

    /// Q3_K (K = 256*n_sb values, n_sb super-blocks per row) times q8_1
    /// activations, M <= 8. ggml-mmvq-shaped redesign (round 2): the
    /// activation is quantized once to q8_1 by `q3k_quantize_q8_1`, so the
    /// inner product is a hardware `dp4a` over packed 4xint8 words instead
    /// of per-weight f32 FMAs, and x costs 1/4 the bytes. One warp owns one
    /// row; per iteration the warp covers two super-blocks (lanes 0..15 ->
    /// even sb, 16..31 -> odd sb; iters = ceil(n_sb/2), an odd n_sb guards
    /// the last iteration's odd half) so every lane's 16-weight qs word is
    /// one u32 and the warp's weight loads are contiguous runs. Odd
    /// super-blocks sit 2 mod 4, so every word is assembled from two
    /// aligned u32 loads with a 16-bit funnel select. Each (word, field)
    /// quad is dequantized in registers with SWAR byte arithmetic
    /// (vi = vil - 4*(1-hbit) as signed bytes) and dotted with one u32 of
    /// q8_1 x via dp4a. The q8_1 block is the 128-value half super-block,
    /// the exact span of a lane's four fields (round 3): per column the
    /// four dp4a results are multiplied by their 6-bit sub-block scales and
    /// summed in int, then ONE f32 FMA applies the shared q8_1 scale and
    /// the super-block scale (round 2 loaded d8 and multiplied per field:
    /// the structural M=8 cost). Reduced with a warp shuffle sum.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= m_cols * 64 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q3k_gemv(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        // One column takes the single-column body directly — see the same
        // shape in `q4k_gemv`. The entries whose column count is a literal 1
        // (`q3k_gemv_sel`, the fused block and expert kernels) already get
        // this by constant folding; an entry that takes it at launch does not.
        if m == 1 {
            let f0 = q3k_row_dot_1col(w, q, d8, n_sb as usize, iters, row, 0, lane);
            let s0 = warp::reduce_sum_f32(f0);
            if lane == 0 {
                // SAFETY: only lane 0 writes; warp `row` owns y[row].
                unsafe {
                    *y.get_unchecked_mut(row) = s0;
                }
            }
            return;
        }
        // The per-row body (lane constants, the two-super-block iteration
        // with the SWAR weight decode, the guarded column chains) is
        // `cores::q3k_row_dot`, shared with `q3k_gemv_sel` and the fused
        // block kernels.
        let f = q3k_row_dot(
            w,
            q,
            d8,
            n_sb as usize,
            iters,
            row,
            0,
            m_cols as usize,
            lane,
        );

        // Warp-uniform reduction (m is a launch-wide constant, and > 1 here).
        // Column c's reduction runs only when m > c; every lane takes the same
        // branch, so the shuffles stay warp-collective, and the tail scales
        // with m exactly like the column bodies above.
        let s0 = warp::reduce_sum_f32(f[0]);
        {
            let s1 = warp::reduce_sum_f32(f[1]);
            let s2 = if m > 2 {
                warp::reduce_sum_f32(f[2])
            } else {
                0.0
            };
            let s3 = if m > 3 {
                warp::reduce_sum_f32(f[3])
            } else {
                0.0
            };
            let s4 = if m > 4 {
                warp::reduce_sum_f32(f[4])
            } else {
                0.0
            };
            let s5 = if m > 5 {
                warp::reduce_sum_f32(f[5])
            } else {
                0.0
            };
            let s6 = if m > 6 {
                warp::reduce_sum_f32(f[6])
            } else {
                0.0
            };
            let s7 = if m > 7 {
                warp::reduce_sum_f32(f[7])
            } else {
                0.0
            };
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes the disjoint
                // segment y[row*m .. row*m+m]: store c is guarded by m > c,
                // so exactly the first m slots of the row are touched.
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = s0;
                    if m > 1 {
                        *y.get_unchecked_mut(b + 1) = s1;
                    }
                    if m > 2 {
                        *y.get_unchecked_mut(b + 2) = s2;
                    }
                    if m > 3 {
                        *y.get_unchecked_mut(b + 3) = s3;
                    }
                    if m > 4 {
                        *y.get_unchecked_mut(b + 4) = s4;
                    }
                    if m > 5 {
                        *y.get_unchecked_mut(b + 5) = s5;
                    }
                    if m > 6 {
                        *y.get_unchecked_mut(b + 6) = s6;
                    }
                    if m > 7 {
                        *y.get_unchecked_mut(b + 7) = s7;
                    }
                }
            }
        }
    }
    /// `q3k_gemv` at m = 1 with K split across warps: `1 << split_shift` (2,
    /// 4 or 8) warps share a row, `8 >> split_shift` rows to a 256-thread
    /// block, and warp `s` of a row walks the iterations `[s * iters >>
    /// split_shift, (s + 1) * iters >> split_shift)` in the row walk's own
    /// per-lane order (`cores::q3k_row_dot_1col_span`). Each warp reduces its
    /// partial with `q3k_gemv`'s shuffle sum, lane 0 parks it in shared
    /// memory, and past the block barrier lane 0 of the row's split-0 warp
    /// adds the splits' sums in split order ([`q3k_split_fold`]). No atomics
    /// and no scratch: a row's bits are a function of its inputs, `n_sb` and
    /// the split width alone — not of the grid or the row count. Width 1 is
    /// `q3k_gemv`, whose single fold this does not reproduce;
    /// `enqueue_gemv_q3k` owns the choice (`q3k_splits`). The m-column shape
    /// is [`q3k_gemv_split_mcol`], a separate entry so its column chains do
    /// not set this one's register count.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes its arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= 64 * iters,
            d8.len() >= 2 * n_sb,
            y.len() >= n_rows,
            split_shift >= 1,
            split_shift <= 3
        )
    )]
    pub fn q3k_gemv_split(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        n_sb: u32,
        iters: u32,
        split_shift: u32,
        mut y: DisjointSlice<f32>,
    ) {
        // One reduced partial per warp.
        static mut PART: SharedArray<f32, 8> = SharedArray::UNINIT;

        let (warp_of, row, split) = q3k_split_warp(thread::index_1d().get(), split_shift);
        // Warp-uniform: a warp past the last row walks nothing but still
        // reaches the barrier below.
        let live = row < n_rows as usize;
        let lane = warp::lane_id() as usize;
        // SAFETY: PART is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every index is a warp index < 8, and the barrier orders each warp's
        // store before the reads.
        let part = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PART) };
        if live {
            let it0 = (split * iters) >> split_shift;
            let it1 = ((split + 1) * iters) >> split_shift;
            let f0 = q3k_row_dot_1col_span(w, q, d8, n_sb as usize, iters, row, 0, lane, it0, it1);
            let s0 = warp::reduce_sum_f32(f0);
            if lane == 0 {
                // SAFETY: warp_of < 8; one lane per warp writes its own slot.
                unsafe {
                    *part.add(warp_of) = s0;
                }
            }
        }
        thread::sync_threads();
        if live && split == 0 && lane == 0 {
            // SAFETY: slots warp_of .. warp_of + (1 << split_shift) are this
            // row's warps, all < 8 and written before the barrier.
            let v = unsafe { q3k_split_fold(part, warp_of, 1, split_shift) };
            // SAFETY: only this lane writes y[row], row < n_rows <= y.len().
            unsafe {
                *y.get_unchecked_mut(row) = v;
            }
        }
    }

    /// [`q3k_gemv_split`] with `m_cols` (2..=8) activation columns: the same
    /// geometry, every column folded over each warp's range by
    /// `cores::q3k_row_dot_cols_span` — which is the single-column span on
    /// that column bit for bit — reduced per column, parked in shared memory
    /// and added in split order by the same [`q3k_split_fold`]. So column c
    /// of an m-column launch is [`q3k_gemv_split`] on column c.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes its arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= m_cols * 64 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            y.len() >= n_rows * m_cols,
            m_cols <= 8,
            split_shift >= 1,
            split_shift <= 3
        )
    )]
    pub fn q3k_gemv_split_mcol(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        split_shift: u32,
        mut y: DisjointSlice<f32>,
    ) {
        // One reduced partial per (warp, column): 8 warps x 8 columns.
        static mut PART: SharedArray<f32, 64> = SharedArray::UNINIT;

        let (warp_of, row, split) = q3k_split_warp(thread::index_1d().get(), split_shift);
        // Warp-uniform: a warp past the last row walks nothing but still
        // reaches the barrier below.
        let live = row < n_rows as usize;
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        // SAFETY: PART is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every index is `warp_of * 8 + c` with warp_of < 8 and c < 8, and
        // the barrier orders each warp's stores before the reads.
        let part = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PART) };
        if live {
            let it0 = (split * iters) >> split_shift;
            let it1 = ((split + 1) * iters) >> split_shift;
            let f =
                q3k_row_dot_cols_span(w, q, d8, n_sb as usize, iters, row, 0, 1, m, lane, it0, it1);
            // Column c's reduction runs only when m > c; `m` is launch-wide,
            // so every lane takes the same branch and the shuffles stay
            // warp-collective.
            let s0 = warp::reduce_sum_f32(f[0]);
            let s1 = if m > 1 {
                warp::reduce_sum_f32(f[1])
            } else {
                0.0
            };
            let s2 = if m > 2 {
                warp::reduce_sum_f32(f[2])
            } else {
                0.0
            };
            let s3 = if m > 3 {
                warp::reduce_sum_f32(f[3])
            } else {
                0.0
            };
            let s4 = if m > 4 {
                warp::reduce_sum_f32(f[4])
            } else {
                0.0
            };
            let s5 = if m > 5 {
                warp::reduce_sum_f32(f[5])
            } else {
                0.0
            };
            let s6 = if m > 6 {
                warp::reduce_sum_f32(f[6])
            } else {
                0.0
            };
            let s7 = if m > 7 {
                warp::reduce_sum_f32(f[7])
            } else {
                0.0
            };
            if lane == 0 {
                let b = warp_of * 8;
                // SAFETY: b + 7 < 64; one lane per warp writes its own eight
                // slots (those past m are never read).
                unsafe {
                    *part.add(b) = s0;
                    *part.add(b + 1) = s1;
                    *part.add(b + 2) = s2;
                    *part.add(b + 3) = s3;
                    *part.add(b + 4) = s4;
                    *part.add(b + 5) = s5;
                    *part.add(b + 6) = s6;
                    *part.add(b + 7) = s7;
                }
            }
        }
        thread::sync_threads();
        if live && split == 0 && lane == 0 {
            let b = row * m;
            let mut c = 0usize;
            while c < m {
                // SAFETY: the row's warps' slots (warp_of + s) * 8 + c, all
                // < 64 and written before the barrier.
                let v = unsafe { q3k_split_fold(part.add(c), warp_of * 8, 8, split_shift) };
                // SAFETY: only this lane writes y[row*m .. row*m+m], and
                // row < n_rows with y.len() >= n_rows*m by the contract.
                unsafe {
                    *y.get_unchecked_mut(b + c) = v;
                }
                c += 1;
            }
        }
    }

    /// [`q3k_gemv`] over the column groups of a [`ColGroups`] in one grid:
    /// block `b` runs group `b % n_groups` ([`col_group_count`]) on the eight
    /// rows from `8 · (b / n_groups)`, the group's `m` columns from `col0 +
    /// c0` through `cores::q3k_row_dot` — `q3k_gemv`'s core, its one-column
    /// body at m = 1 included, so a group's rows are that launch's bits on
    /// the group's columns alone — and stores them at `y[n_rows·c0 + row·m +
    /// c]`: the groups' own row-major outputs laid end to end. A row
    /// stretch's blocks are consecutive, so its groups run side by side on
    /// the same weight words.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes its arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= col0 * 64 * iters + cols * 64 * iters,
            d8.len() >= col0 * 2 * n_sb + cols * 2 * n_sb,
            y.len() >= n_rows * cols,
            lead >= 1,
            lead <= 8,
            lead <= cols
        )
    )]
    pub fn q3k_gemv_groups(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        col0: u32,
        lead: u32,
        cols: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let (lead, cols) = (lead as usize, cols as usize);
        let n_groups = col_group_count(lead, cols);
        let b = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let row = (b / n_groups) * 8 + t / 32;
        let (c0, m) = col_group(b % n_groups, lead, cols);
        // Both tests are block-uniform: a block past the rows, or past the
        // last group, stores nothing.
        if row >= n_rows as usize || m == 0 {
            return;
        }
        let lane = warp::lane_id() as usize;
        // q3k_row_dot's contract: the row inside w and columns col0 + c0 ..
        // col0 + c0 + m (c0 + m <= cols) inside q and d8, by the launch
        // contract; every lane of the warp calls it.
        let f = q3k_row_dot(
            w,
            q,
            d8,
            n_sb as usize,
            iters,
            row,
            col0 as usize + c0,
            m,
            lane,
        );
        let s = col_sums(f, m);
        if lane == 0 {
            // SAFETY: slots n_rows·c0 + row·m + c for c < m are the group's
            // block, inside y (n_rows·(c0 + m) <= n_rows·cols <= y.len()),
            // and belong to this row's warp alone.
            unsafe { store_cols(&mut y, n_rows as usize * c0 + row * m, 1, m, s) };
        }
    }

    /// [`q3k_gemv_split_mcol`] over the column groups of a [`ColGroups`] in
    /// one grid, laid out as [`q3k_gemv_groups`] lays them: block `b` runs
    /// group `b % n_groups` on the `8 >> split_shift` rows of row stretch `b
    /// / n_groups`, each row's warps, spans and fold those of
    /// `q3k_gemv_split_mcol` (`cores::q3k_row_dot_cols_span`, the
    /// single-column span at m = 1, so a one-column group is
    /// [`q3k_gemv_split`]'s bits), the group's block at `y[n_rows·c0 ..]`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes its arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= col0 * 64 * iters + cols * 64 * iters,
            d8.len() >= col0 * 2 * n_sb + cols * 2 * n_sb,
            y.len() >= n_rows * cols,
            lead >= 1,
            lead <= 8,
            lead <= cols,
            split_shift >= 1,
            split_shift <= 3
        )
    )]
    pub fn q3k_gemv_split_groups(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        col0: u32,
        lead: u32,
        cols: u32,
        n_sb: u32,
        iters: u32,
        split_shift: u32,
        mut y: DisjointSlice<f32>,
    ) {
        // One reduced partial per (warp, column): 8 warps x 8 columns.
        static mut PART: SharedArray<f32, 64> = SharedArray::UNINIT;

        let (lead, cols) = (lead as usize, cols as usize);
        let n_groups = col_group_count(lead, cols);
        let b = thread::blockIdx_x() as usize;
        let (c0, m) = col_group(b % n_groups, lead, cols);
        let (warp_of, row, split) = q3k_split_warp(
            (b / n_groups) * 256 + thread::threadIdx_x() as usize,
            split_shift,
        );
        // Warp-uniform: a warp past the last row walks nothing but still
        // reaches the barrier below.
        let live = row < n_rows as usize && m > 0;
        let lane = warp::lane_id() as usize;
        // SAFETY: PART is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every index is `warp_of * 8 + c` with warp_of < 8 and c < 8, and
        // the barrier orders each warp's stores before the reads.
        let part = unsafe { SharedArray::as_raw_mut_ptr(&raw mut PART) };
        if live {
            let it0 = (split * iters) >> split_shift;
            let it1 = ((split + 1) * iters) >> split_shift;
            let f = q3k_row_dot_cols_span(
                w,
                q,
                d8,
                n_sb as usize,
                iters,
                row,
                col0 as usize + c0,
                1,
                m,
                lane,
                it0,
                it1,
            );
            let s = col_sums(f, m);
            if lane == 0 {
                let b8 = warp_of * 8;
                // SAFETY: b8 + 7 < 64; one lane per warp writes its own eight
                // slots (those past m are never read).
                unsafe {
                    *part.add(b8) = s[0];
                    *part.add(b8 + 1) = s[1];
                    *part.add(b8 + 2) = s[2];
                    *part.add(b8 + 3) = s[3];
                    *part.add(b8 + 4) = s[4];
                    *part.add(b8 + 5) = s[5];
                    *part.add(b8 + 6) = s[6];
                    *part.add(b8 + 7) = s[7];
                }
            }
        }
        thread::sync_threads();
        if live && split == 0 && lane == 0 {
            let base = n_rows as usize * c0 + row * m;
            let mut c = 0usize;
            while c < m {
                // SAFETY: the row's warps' slots (warp_of + s) * 8 + c, all
                // < 64 and written before the barrier.
                let v = unsafe { q3k_split_fold(part.add(c), warp_of * 8, 8, split_shift) };
                // SAFETY: only this lane writes the group's slots base ..
                // base + m, inside y as in `q3k_gemv_groups`.
                unsafe {
                    *y.get_unchecked_mut(base + c) = v;
                }
                c += 1;
            }
        }
    }

    /// Q3_K gemv over expert slots selected on the device (the MoE decode
    /// shape): one launch computes `n_slots` experts of a resident flat
    /// stack — `n_experts * rows_per_expert` rows of `110 * n_sb` bytes —
    /// against the ONE quantized activation column of `q`/`d8` (m = 1;
    /// gate and up read the same input). The selected ids are read from
    /// `sel`, a device buffer, so the launch is addressable from inside a
    /// captured graph whose replay consumes whatever a router kernel last
    /// wrote there. Thread geometry as `q3k_gemv` (one warp per output
    /// row); thread row `n = slot * rows_per_expert + row_in_expert`
    /// stores `y[n]` and reads weight row `sel[slot] * rows_per_expert +
    /// row_in_expert`. The arithmetic is the m = 1 path of `q3k_gemv`
    /// verbatim (same cores, same accumulation order), so a slot's output
    /// is bit-identical to `q3k_gemv` run on that expert alone.
    ///
    /// An id >= n_experts cannot be rejected by the host contract (it
    /// lives in device memory): the slot's warps return before their first
    /// load — warp-uniform, no divergent branch — leaving that slot of `y`
    /// untouched and every other slot unaffected. [`hybrid::HOST`] is a slot
    /// the host tier serves, skipped by contract; any other id past the stack
    /// raises [`FaultSite::ExpertId`] on `fault` first.
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
            4 * w.len() >= n_experts * rows_per_expert * 110 * n_sb,
            q.len() >= 64 * iters,
            d8.len() >= 2 * n_sb,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q3k_gemv_sel(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_sb: u32,
        iters: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (all 32 lanes of the warp share `row`, hence
        // `slot`), so the out-of-range return below never diverges a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != hybrid::HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        let id = id as usize;
        let row_abs = id * rows_per_expert as usize + row % rows_per_expert as usize;
        // The m = 1 per-row body is `cores::q3k_row_dot`, shared with
        // `q3k_gemv` and the fused block kernels.
        let f = q3k_row_dot(w, q, d8, n_sb as usize, iters, row_abs, 0, 1, lane);
        let s0 = warp::reduce_sum_f32(f[0]);
        if lane == 0 {
            // SAFETY: row < n_slots*rows_per_expert <= y.len() by the launch
            // contract; only lane 0 of the warp writes y[row].
            unsafe {
                *y.get_unchecked_mut(row) = s0;
            }
        }
    }
}

/// Device `dev`'s name, asked of the driver without a context.
fn raw_device_name(dev: cuda_core::sys::CUdevice) -> Result<String, GpuError> {
    let mut buf: [std::ffi::c_char; 256] = [0; 256];
    let len = std::ffi::c_int::try_from(buf.len()).expect("256 fits a c_int");
    // SAFETY: `buf` is a live, writable buffer of `len` bytes for the call,
    // and the driver writes at most that many, a NUL included.
    let rc = unsafe { cuda_core::sys::cuDeviceGetName(buf.as_mut_ptr(), len, dev) };
    graph::cu(rc, "cuDeviceGetName")?;
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c.to_ne_bytes()[0])
        .collect();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// A CUDA context on one device ([`Gpu::new`]: device 0; [`Gpu::for_card`]: the
/// card a name picks) with this crate's device module loaded and one
/// non-blocking stream that every launch and copy of this engine goes on.
///
/// The stream is a real `cuStreamCreate` stream, not the legacy default: the
/// driver refuses graph capture on the null stream, and `Graph::capture`
/// records whatever is enqueued on `self.stream()` between begin and end.
pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    q4k_sel: q4k_sel::Q4kSelKernels,
    q5: q5::Q5Kernels,
    q8f32: q8f32::Q8F32Kernels,
    elem: elem::ElemKernels,
    flash: flash::FlashKernels,
    router: router::RouterKernels,
    fused: fused::FusedKernels,
    moe_fused: moe_fused::MoeFusedKernels,
    /// The fault word (crate::fault): the allocation's first u32,
    /// [`FAULT_NONE`] while clean. Every launch on this context's stream that
    /// can refuse its input carries a [`FaultSink`] over it. Shared with the
    /// modules given it at load ([`Gpu::fault_word`]), which keep it alive
    /// while they can launch.
    fault: Arc<DeviceBuffer<u32>>,
}

/// The sink over a `Gpu`'s fault allocation ([`Gpu::fault_word`]), for
/// launches of `layer`. A layer past [`LAYER_NONE`] has no site mask and is
/// refused by panic: every caller passes a constant or a
/// [`Gpu::layer_sink`]-checked layer.
#[must_use]
pub(crate) fn sink_over(word: &Arc<DeviceBuffer<u32>>, layer: u32) -> FaultSink {
    assert!(
        layer <= LAYER_NONE && word.len() >= FAULT_WORDS,
        "sink_over: layer {layer} over a fault allocation of {} words (want layer <= {LAYER_NONE}, \
         {FAULT_WORDS} words)",
        word.len()
    );
    // SAFETY: `word` holds FAULT_WORDS device u32s of the context the
    // holder's launches run in ([`module_fault_word`] checks both at load,
    // the assert above the length again) and `layer <= LAYER_NONE`; the
    // holder keeps the Arc, so the allocation stays alive while any of its
    // launches can run (its drop synchronizes the context first), and a
    // graph drops before the buffers it addresses.
    unsafe { FaultSink::new(word.cu_deviceptr() as *mut u32, layer) }
}

/// A kernel module's handle on the fault allocation it is loaded with:
/// `word` shared, after checking that it is a buffer of `ctx` (the context
/// the module's launches run in) holding the first-layer word and every site
/// mask ([`FAULT_WORDS`]). Anything else is refused by `what`'s name.
pub(crate) fn module_fault_word(
    ctx: &Arc<CudaContext>,
    word: &Arc<DeviceBuffer<u32>>,
    what: &'static str,
) -> Result<Arc<DeviceBuffer<u32>>, GpuError> {
    if word.len() < FAULT_WORDS || !Arc::ptr_eq(word.context(), ctx) {
        return Err(GpuError::state(
            what,
            "the fault word of the Gpu that owns this context (Gpu::fault_word)",
        ));
    }
    Ok(Arc::clone(word))
}

/// The fault word's allocation, in u32s: one whole 2 MiB allocation granule.
/// A smaller allocation would open, before any model loads, the shared
/// granule a load's small buffers are planned into (`model::placement`'s
/// heap), and every card's planned rounding would be one granule off what the
/// load then takes.
const FAULT_ALLOC_WORDS: usize = (2 << 20) / 4;

// The first-layer word and every layer's site mask fit the granule.
const _: () = assert!(fault::FAULT_WORDS <= FAULT_ALLOC_WORDS);

/// The fault allocation's clean contents: the first-layer word at
/// [`FAULT_NONE`], every site mask (and the unused tail) at 0.
fn clean_fault_words() -> Vec<u32> {
    let mut v = vec![0u32; FAULT_ALLOC_WORDS];
    v[0] = FAULT_NONE;
    v
}

impl Gpu {
    /// `with_device(0)`: under the box environment device 0 is the dev card.
    pub fn new() -> Result<Gpu, GpuError> {
        Gpu::with_device(0)
    }

    /// Create the context on CUDA device `device`, the engine stream, and
    /// load every device module of this crate into that context — the
    /// K-quant module here and one per kernel file. Load-time only.
    pub(crate) fn with_device(device: usize) -> Result<Gpu, GpuError> {
        let ctx = CudaContext::new(device)?;
        let stream = ctx.new_stream()?;
        // First: the modules below that raise are given it at load.
        let fault = Arc::new(DeviceBuffer::from_host(&stream, &clean_fault_words())?);
        // SAFETY: this package owns the embedded device bundle produced for
        // the kernels module above; every launcher checks its launch
        // contract before launching.
        let module = unsafe { kernels::load(&ctx)? };
        Ok(Gpu {
            q4k_sel: q4k_sel::Q4kSelKernels::load(&ctx, &fault)?,
            q5: q5::Q5Kernels::load(&ctx, &fault)?,
            q8f32: q8f32::Q8F32Kernels::load(&ctx)?,
            elem: elem::ElemKernels::load(&ctx, &fault)?,
            flash: flash::FlashKernels::load(&ctx)?,
            router: router::RouterKernels::load(&ctx)?,
            fused: fused::FusedKernels::load(&ctx)?,
            moe_fused: moe_fused::MoeFusedKernels::load(&ctx, &fault)?,
            fault,
            ctx,
            stream,
            module,
        })
    }

    /// The sink a launch of `layer` raises into: this context's fault word.
    /// [`LAYER_HEAD`] for the output head, [`LAYER_NONE`] for a launch no
    /// layer owns; a model layer comes through [`Gpu::layer_sink`], which
    /// checks that the word can carry it.
    #[must_use]
    pub(crate) fn fault_sink(&self, layer: u32) -> FaultSink {
        sink_over(&self.fault, layer)
    }

    /// This `Gpu`'s fault word, for a kernel module loaded outside the `Gpu`
    /// that raises into it: the module keeps the `Arc` while it can launch.
    #[must_use]
    pub fn fault_word(&self) -> &Arc<DeviceBuffer<u32>> {
        &self.fault
    }

    /// The sink of a launch no layer owns — a gate's direct launch: its
    /// fault reads as [`LAYER_NONE`].
    #[must_use]
    pub fn unlabelled_sink(&self) -> FaultSink {
        self.fault_sink(LAYER_NONE)
    }

    /// [`Gpu::fault_sink`] for model layer `layer`: a layer index the word
    /// cannot carry (at or past [`LAYER_HEAD`]) is refused, not wrapped into
    /// another layer's code.
    pub fn layer_sink(&self, layer: usize) -> Result<FaultSink, GpuError> {
        match u32::try_from(layer) {
            Ok(l) if l < LAYER_HEAD => Ok(self.fault_sink(l)),
            _ => Err(GpuError::shape(
                "Gpu::layer_sink",
                format!("layer {layer} does not fit the fault word (layers below {LAYER_HEAD})"),
            )),
        }
    }

    /// The fault the allocation holds, if any — the first-layer word and
    /// that layer's site mask: blocking reads on the engine stream, so they
    /// see every launch enqueued before them.
    pub fn fault(&self) -> Result<Option<Fault>, GpuError> {
        let word = self.fault_u32(0)?;
        if word == FAULT_NONE {
            return Ok(None);
        }
        let sites = match usize::try_from(word >> 8) {
            Ok(l) if l <= LAYER_NONE as usize => self.fault_u32(fault::mask_index(word >> 8))?,
            _ => 0,
        };
        Ok(Fault::from_words(word, sites))
    }

    /// Word `at` of the fault allocation, read on the engine stream.
    fn fault_u32(&self, at: usize) -> Result<u32, GpuError> {
        if at >= FAULT_WORDS {
            return Err(GpuError::shape(
                "Gpu::fault_u32",
                format!("word {at} of a fault allocation of {FAULT_WORDS}"),
            ));
        }
        self.ctx.bind_to_thread()?;
        let mut word = FAULT_NONE;
        // SAFETY: the source is word `at < FAULT_WORDS` of this Gpu's own
        // live allocation of FAULT_ALLOC_WORDS; the destination is `word`,
        // four bytes that outlive the copy because the stream is synchronized
        // before it goes out of scope; this context is current on the
        // calling thread (bound above).
        let rc = unsafe {
            cuda_core::sys::cuMemcpyDtoHAsync_v2(
                (&raw mut word).cast(),
                self.fault.cu_deviceptr() + 4 * at as u64,
                std::mem::size_of::<u32>(),
                self.stream.cu_stream(),
            )
        };
        graph::cu(rc, "cuMemcpyDtoHAsync_v2 (fault word)")?;
        self.stream.synchronize()?;
        Ok(word)
    }

    /// Put the first-layer word back to [`FAULT_NONE`] and every site mask
    /// to 0, in stream order: every launch enqueued before this may still
    /// raise into what it clears, every launch after it starts clean. Never
    /// inside a capture.
    pub fn clear_fault(&self) -> Result<(), GpuError> {
        self.ctx.bind_to_thread()?;
        // SAFETY: both ranges lie inside this Gpu's own live allocation of
        // FAULT_ALLOC_WORDS >= FAULT_WORDS u32s, the stream handle is this
        // Gpu's engine stream, and this context is current on the calling
        // thread (bound above); the memsets are ordered on that stream like
        // every launch that touches the allocation.
        let rc = unsafe {
            cuda_core::sys::cuMemsetD32Async(
                self.fault.cu_deviceptr(),
                FAULT_NONE,
                1,
                self.stream.cu_stream(),
            )
        };
        graph::cu(rc, "cuMemsetD32Async (fault word)")?;
        // SAFETY: as above; the masks are words 1..FAULT_WORDS.
        let rc = unsafe {
            cuda_core::sys::cuMemsetD32Async(
                self.fault.cu_deviceptr() + 4,
                0,
                FAULT_WORDS - 1,
                self.stream.cu_stream(),
            )
        };
        graph::cu(rc, "cuMemsetD32Async (fault site masks)")?;
        Ok(())
    }

    /// [`Gpu::fault`], then [`Gpu::clear_fault`]: what a gate that planted
    /// a fault reads, leaving the word clean for its next case.
    pub fn take_fault(&self) -> Result<Option<Fault>, GpuError> {
        let f = self.fault()?;
        self.clear_fault()?;
        Ok(f)
    }

    /// `with_device` on the one visible CUDA device whose name contains
    /// `name` — a placement card's name. A plan names its cards and
    /// `CUDA_VISIBLE_DEVICES` orders them, so a card is never found by
    /// ordinal. No device, or two, is an error that says so.
    pub fn for_card(name: &str) -> Result<Gpu, GpuError> {
        let n = usize::try_from(cuda_core::Device::device_count()?)
            .map_err(|_| GpuError::shape("Gpu::for_card", "negative device count"))?;
        let mut named = Vec::new();
        let mut seen = Vec::with_capacity(n);
        for ordinal in 0..n {
            let full = raw_device_name(cuda_core::Device::raw_device(ordinal)?)?;
            if full.contains(name) {
                named.push(ordinal);
            }
            seen.push(full);
        }
        match named.as_slice() {
            &[ordinal] => Gpu::with_device(ordinal),
            _ => Err(GpuError::shape(
                "Gpu::for_card",
                format!(
                    "{} visible devices are named like {name:?}, not one: {seen:?}",
                    named.len()
                ),
            )),
        }
    }

    /// The device's name as the driver reports it.
    pub fn device_name(&self) -> Result<String, GpuError> {
        Ok(self.ctx.device_name()?)
    }

    /// `(free, total)` device bytes of this context's device, as
    /// `cuMemGetInfo` reports them.
    pub fn mem_info(&self) -> Result<(usize, usize), GpuError> {
        self.ctx.bind_to_thread()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both out-pointers are live locals for the duration of the
        // call, and this context is current on the calling thread (bound above).
        let rc = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
        graph::cu(rc, "cuMemGetInfo")?;
        Ok((free, total))
    }

    /// The driver's minimum allocation granularity for device memory on this
    /// device (`cuMemGetAllocationGranularity`): the page size an allocation's
    /// physical backing is counted in.
    pub fn allocation_granularity(&self) -> Result<usize, GpuError> {
        self.ctx.bind_to_thread()?;
        Ok(cuda_core::vmm::allocation_granularity(
            self.ctx.cu_device(),
        )?)
    }

    /// Q4_K gemv over expert slots selected on the device (the down shape).
    pub fn q4k_sel(&self) -> &q4k_sel::Q4kSelKernels {
        &self.q4k_sel
    }

    /// Q5_0 / Q5_1 gemv and the 32-value activation quantizer.
    pub fn q5(&self) -> &q5::Q5Kernels {
        &self.q5
    }

    /// Q8_0 and F32 gemv over f32 activations.
    pub fn q8f32(&self) -> &q8f32::Q8F32Kernels {
        &self.q8f32
    }

    /// Element-wise and reduction kernels.
    pub fn elem(&self) -> &elem::ElemKernels {
        &self.elem
    }

    /// KV append and latent flash attention.
    pub fn flash(&self) -> &flash::FlashKernels {
        &self.flash
    }

    /// Router top-6 and the expert offset table.
    pub fn router(&self) -> &router::RouterKernels {
        &self.router
    }

    /// The fused block kernels (P0b): norm+quantize, gate·up·swiglu,
    /// down+residual — bit-identical to the per-op path they replace.
    pub(crate) fn fused(&self) -> &fused::FusedKernels {
        &self.fused
    }

    /// The fused MoE kernels: six experts' gate·up·swiglu in one launch and
    /// the weighted combine (+shexp, +residual) in one — bit-identical to
    /// the per-op `_sel` path they replace.
    pub(crate) fn moe_fused(&self) -> &moe_fused::MoeFusedKernels {
        &self.moe_fused
    }

    /// The CUDA context this device's buffers and modules live in.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The engine stream. Allocations at load time and the per-step
    /// synchronize go here too, so nothing ever orders against the null
    /// stream.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Capture `body`'s enqueues on the engine stream into a replayable
    /// graph. See `Graph::capture` for what the body may not do.
    pub fn capture<F>(&self, body: F) -> Result<Graph, GpuError>
    where
        F: FnOnce(&CudaStream) -> Result<(), GpuError>,
    {
        Graph::capture(&self.stream, body)
    }

    /// Enqueue the q8_1 quantization of `x` (`act.m()` columns of `act.k()`
    /// f32 each) into `act`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_quantize_q8_1(
        &self,
        x: &DeviceBuffer<f32>,
        act: &mut Q8Act,
    ) -> Result<(), GpuError> {
        self.quantize_q8_1_at(x, 0, act, self.unlabelled_sink())
    }

    /// [`Gpu::enqueue_quantize_q8_1`] for a launch of model layer `layer`:
    /// a non-finite value raises the fault word with that layer.
    pub fn enqueue_quantize_q8_1_layer(
        &self,
        x: &DeviceBuffer<f32>,
        act: &mut Q8Act,
        layer: usize,
    ) -> Result<(), GpuError> {
        let sink = self.layer_sink(layer)?;
        self.quantize_q8_1_at(x, 0, act, sink)
    }

    /// The head's quantization: [`Gpu::enqueue_quantize_q8_1`] raising with
    /// [`LAYER_HEAD`].
    pub(crate) fn enqueue_quantize_q8_1_head(
        &self,
        x: &DeviceBuffer<f32>,
        act: &mut Q8Act,
    ) -> Result<(), GpuError> {
        self.quantize_q8_1_at(x, 0, act, self.fault_sink(LAYER_HEAD))
    }

    /// Enqueue the q8_1 quantization of `x[x0 .. x0 + m*k]` (`m = act.m()`
    /// columns of `act.k()` f32 each) into `act` — the same quantization
    /// `enqueue_quantize_q8_1` runs, on a base-offset slice of a wider
    /// buffer, so the quantized bytes equal a copy-then-quantize. Asynchronous,
    /// allocation-free, capturable.
    pub(crate) fn enqueue_quantize_q8_1_at(
        &self,
        x: &DeviceBuffer<f32>,
        x0: usize,
        act: &mut Q8Act,
    ) -> Result<(), GpuError> {
        self.quantize_q8_1_at(x, x0, act, self.unlabelled_sink())
    }

    /// [`Gpu::enqueue_quantize_q8_1_layer`] over the first `cols` columns of
    /// `act` alone (`1 ..=` its `m()`), `x` holding at least `cols · k` f32:
    /// each column the bytes the full launch writes it — the quantizer is
    /// column-local — and the columns past `cols` untouched. The per-slot
    /// activation of a prompt pass, quantized for the tokens the pass holds.
    pub fn enqueue_quantize_q8_1_cols(
        &self,
        x: &DeviceBuffer<f32>,
        act: &mut Q8Act,
        cols: usize,
        layer: usize,
    ) -> Result<(), GpuError> {
        let sink = self.layer_sink(layer)?;
        self.quantize_q8_1_cols_at(x, 0, act, cols, sink)
    }

    /// The launcher of `q3k_quantize_q8_1` into a [`Q8Act`], raising into
    /// `fault`; [`Gpu::enqueue_quantize_gemm`] is its other launcher, into a
    /// [`gemm::GemmAct`].
    fn quantize_q8_1_at(
        &self,
        x: &DeviceBuffer<f32>,
        x0: usize,
        act: &mut Q8Act,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let m = act.m();
        self.quantize_q8_1_cols_at(x, x0, act, m, fault)
    }

    /// [`Gpu::quantize_q8_1_at`] over `act`'s first `m` columns, refused
    /// unless `1 <= m <= act.m()`.
    fn quantize_q8_1_cols_at(
        &self,
        x: &DeviceBuffer<f32>,
        x0: usize,
        act: &mut Q8Act,
        m: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let n_sb = act.n_sb();
        if !(1..=act.m()).contains(&m) {
            return Err(GpuError::shape(
                "enqueue_quantize_q8_1_at",
                format!("{m} columns of an activation of {}", act.m()),
            ));
        }
        if x.len() < x0 + m * act.k() {
            return Err(GpuError::shape(
                "enqueue_quantize_q8_1_at",
                format!(
                    "x.len() {} < x0 + m*k = {} + {}*{}",
                    x.len(),
                    x0,
                    m,
                    act.k()
                ),
            ));
        }
        let what = "enqueue_quantize_q8_1_at";
        let grid = launch_u32(what, "grid", m * n_sb * 2)?;
        let x0 = launch_u32(what, "x0", x0)?;
        let m = launch_u32(what, "m", m)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        // Prepared per call for now: the prepare step is host-only contract
        // validation, and it is exactly the kind of per-launch host cost a
        // captured graph removes. Caching per shape is P8's business.
        let prep = self
            .module
            .prepare_q3k_quantize_q8_1(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.q3k_quantize_q8_1(
            &self.stream,
            &prep,
            x,
            x0,
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
        Ok(())
    }

    /// Enqueue the q8_1 quantization of TWO slices of the same buffer in one
    /// launch: `x[a0 .. a0 + m*k]` into `a` and `x[b0 .. b0 + m*k]` into `b`,
    /// the same bytes the two `enqueue_quantize_q8_1_at` calls write. Both
    /// scratches must have the same shape (one grid covers both halves).
    /// Asynchronous, allocation-free, capturable.
    pub(crate) fn enqueue_quantize_q8_1_pair(
        &self,
        x: &DeviceBuffer<f32>,
        a0: usize,
        a: &mut Q8Act,
        b0: usize,
        b: &mut Q8Act,
    ) -> Result<(), GpuError> {
        let (m, n_sb) = (a.m(), a.n_sb());
        if b.m() != m || b.k() != a.k() {
            return Err(GpuError::shape(
                "enqueue_quantize_q8_1_pair",
                format!(
                    "both halves must share a shape, got a {}x{} b {}x{}",
                    a.m(),
                    a.k(),
                    b.m(),
                    b.k()
                ),
            ));
        }
        let need = m * a.k();
        if x.len() < a0 + need || x.len() < b0 + need {
            return Err(GpuError::shape(
                "enqueue_quantize_q8_1_pair",
                format!("x.len() {} < max(a0 {a0}, b0 {b0}) + m*k = {need}", x.len()),
            ));
        }
        let what = "enqueue_quantize_q8_1_pair";
        let grid = launch_u32(what, "grid", m * n_sb * 4)?;
        let a0 = launch_u32(what, "a0", a0)?;
        let b0 = launch_u32(what, "b0", b0)?;
        let m = launch_u32(what, "m", m)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_q3k_quantize_q8_1_pair(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.q3k_quantize_q8_1_pair(
            &self.stream,
            &prep,
            x,
            a0,
            b0,
            m,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            &mut a.q3,
            &mut a.q4,
            &mut a.q6,
            &mut a.s8,
            &mut a.d8,
            &mut b.q3,
            &mut b.q4,
            &mut b.q6,
            &mut b.s8,
            &mut b.d8,
            self.unlabelled_sink(),
        )?;
        Ok(())
    }

    /// Enqueue `y = w · act` for a Q4_K weight of `w.rows()` rows — 36 u32
    /// words per super-block, `36 * n_sb` per row — against the quantized
    /// activations in `act`, which supply K. `y` holds `rows * m` f32,
    /// row-major with `m` outputs per row. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_gemv_q4k(
        &self,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, m) = (w.rows(), act.m());
        let n_sb = act.n_sb();
        if w.cols() != 36 * n_sb {
            return Err(GpuError::shape(
                "enqueue_gemv_q4k",
                format!(
                    "Q4_K rows are 36*{} = {} words at K={}, got {}",
                    n_sb,
                    36 * n_sb,
                    act.k(),
                    w.cols()
                ),
            ));
        }
        if y.len() < n_rows * m {
            return Err(GpuError::shape(
                "enqueue_gemv_q4k",
                format!("y.len() {} < rows*m = {}", y.len(), n_rows * m),
            ));
        }
        let what = "enqueue_gemv_q4k";
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let cfg = LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0);
        // One column is the decode entry; more go to the m-column entry,
        // whose column c is the decode entry's launch of column c.
        if m == 1 {
            let prep = self.module.prepare_q4k_gemv(cfg)?;
            self.module.q4k_gemv(
                &self.stream,
                &prep,
                w.buf(),
                &act.q4,
                &act.s8,
                &act.d8,
                n_rows,
                n_sb,
                n_sb.div_ceil(4),
                y,
            )?;
        } else {
            let m = launch_u32(what, "m", m)?;
            let prep = self.module.prepare_q4k_gemv_mcol(cfg)?;
            self.module.q4k_gemv_mcol(
                &self.stream,
                &prep,
                w.buf(),
                &act.q4,
                &act.s8,
                &act.d8,
                n_rows,
                m,
                n_sb,
                n_sb.div_ceil(4),
                y,
            )?;
        }
        Ok(())
    }

    /// Enqueue `y = w · act` for a Q3_K weight of `w.rows()` rows — 110
    /// bytes per super-block, the rows' byte stream as u32 words, zero-padded
    /// at its end to a whole number of words per row (`CardFormat::KQuant`:
    /// `110 * n_sb / 4` words a row for even n_sb; an odd n_sb starts every
    /// other row on a half word, which the row walk's byte addressing reads)
    /// — against the quantized activations in `act`, which supply K. `y`
    /// holds `rows * m` f32, row-major with `m` outputs per row. The launch
    /// is `q3k_gemv`, or `q3k_gemv_split` (`_mcol` at m > 1) when
    /// [`q3k_splits`] splits rows of this K. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q3k(
        &self,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, m) = (w.rows(), act.m());
        let n_sb = act.n_sb();
        let words = (n_rows * 110 * n_sb).div_ceil(4).div_ceil(n_rows.max(1));
        if w.cols() != words {
            return Err(GpuError::shape(
                "enqueue_gemv_q3k",
                format!(
                    "Q3_K rows are {} words at K={}, got {}",
                    words,
                    act.k(),
                    w.cols()
                ),
            ));
        }
        if y.len() < n_rows * m {
            return Err(GpuError::shape(
                "enqueue_gemv_q3k",
                format!("y.len() {} < rows*m = {}", y.len(), n_rows * m),
            ));
        }
        let what = "enqueue_gemv_q3k";
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let m = launch_u32(what, "m", m)?;
        let splits = q3k_splits(n_sb);
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let iters = n_sb.div_ceil(2);
        if splits == 1 {
            let prep =
                self.module
                    .prepare_q3k_gemv(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
            self.module.q3k_gemv(
                &self.stream,
                &prep,
                w.buf(),
                &act.q3,
                &act.d8,
                n_rows,
                m,
                n_sb,
                iters,
                y,
            )?;
        } else {
            let split_shift = splits.trailing_zeros();
            let rows_per_block = launch_u32(what, "rows_per_block", 8 / splits)?;
            let grid = LaunchConfig1D::new(n_rows.div_ceil(rows_per_block), 256, 0);
            if m == 1 {
                let prep = self.module.prepare_q3k_gemv_split(grid)?;
                self.module.q3k_gemv_split(
                    &self.stream,
                    &prep,
                    w.buf(),
                    &act.q3,
                    &act.d8,
                    n_rows,
                    n_sb,
                    iters,
                    split_shift,
                    y,
                )?;
            } else {
                let prep = self.module.prepare_q3k_gemv_split_mcol(grid)?;
                self.module.q3k_gemv_split_mcol(
                    &self.stream,
                    &prep,
                    w.buf(),
                    &act.q3,
                    &act.d8,
                    n_rows,
                    m,
                    n_sb,
                    iters,
                    split_shift,
                    y,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue `y = w · act` over the column groups `groups` of `act`
    /// ([`ColGroups`]) in one launch: group `g`'s `rows × m` block at `y[rows
    /// · c0 ..]`, bit for bit [`Gpu::enqueue_gemv_q3k`] on an activation of
    /// the group's columns alone, its split-K rule ([`q3k_splits`])
    /// included. Refused when the groups pass `act`'s columns or `y` holds
    /// fewer than `rows · cols`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q3k_groups(
        &self,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        groups: ColGroups,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q3k_groups";
        let (n_rows, n_sb) = (w.rows(), act.n_sb());
        let words = (n_rows * 110 * n_sb).div_ceil(4).div_ceil(n_rows.max(1));
        if w.cols() != words {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q3_K rows are {words} words at K={}, got {}",
                    act.k(),
                    w.cols()
                ),
            ));
        }
        let (col0, lead, cols) = (groups.col0(), groups.lead(), groups.cols());
        if col0 + cols > act.m() || y.len() < n_rows * cols {
            return Err(GpuError::shape(
                what,
                format!(
                    "columns {col0}..{} of an activation of {}; y.len() {} (need rows*cols = {})",
                    col0 + cols,
                    act.m(),
                    y.len(),
                    n_rows * cols
                ),
            ));
        }
        let splits = q3k_splits(n_sb);
        let grid = launch_u32(what, "grid", groups.count() * n_rows.div_ceil(8 / splits))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let col0 = launch_u32(what, "col0", col0)?;
        let lead = launch_u32(what, "lead", lead)?;
        let cols = launch_u32(what, "cols", cols)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let iters = n_sb.div_ceil(2);
        let cfg = LaunchConfig1D::new(grid, 256, 0);
        if splits == 1 {
            let prep = self.module.prepare_q3k_gemv_groups(cfg)?;
            self.module.q3k_gemv_groups(
                &self.stream,
                &prep,
                w.buf(),
                &act.q3,
                &act.d8,
                n_rows,
                col0,
                lead,
                cols,
                n_sb,
                iters,
                y,
            )?;
        } else {
            let prep = self.module.prepare_q3k_gemv_split_groups(cfg)?;
            self.module.q3k_gemv_split_groups(
                &self.stream,
                &prep,
                w.buf(),
                &act.q3,
                &act.d8,
                n_rows,
                col0,
                lead,
                cols,
                n_sb,
                iters,
                splits.trailing_zeros(),
                y,
            )?;
        }
        Ok(())
    }

    /// Enqueue the device-indirect MoE decode step for a Q3_K expert stack:
    /// one launch computes `n_slots` experts, slot s writing
    /// `y[s*rows_per_expert .. (s+1)*rows_per_expert]` as
    /// `w[sel[s]*rows_per_expert .. +rows_per_expert] · act` — every slot
    /// dots the ONE activation column of `act` (m = 1: gate and up read the
    /// same input). `w` is the full resident stack, `w.rows()` a multiple of
    /// `rows_per_expert` (`n_experts = w.rows()/rows_per_expert`), rows as in
    /// `enqueue_gemv_q3k` (`110 * n_sb / 4` words, even n_sb). `sel` is a
    /// device buffer of at least `n_slots` ids read by the kernel per launch,
    /// so a captured graph replay picks up new ids written between replays;
    /// an id >= n_experts leaves that slot of `y` untouched, and one that is
    /// not [`hybrid::HOST`] raises [`FaultSite::ExpertId`] on this `Gpu`'s
    /// fault word as an unlabelled launch ([`LAYER_NONE`]: the launcher
    /// knows no layer). Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q3k_sel(
        &self,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        sel: &DeviceBuffer<u32>,
        n_slots: usize,
        rows_per_expert: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n_sb = act.n_sb();
        if act.m() != 1 {
            return Err(GpuError::shape(
                "enqueue_gemv_q3k_sel",
                format!(
                    "m = 1 only (the shared expert input), got act.m() = {}",
                    act.m()
                ),
            ));
        }
        if !n_sb.is_multiple_of(2) {
            return Err(GpuError::shape(
                "enqueue_gemv_q3k_sel",
                format!(
                    "odd super-block count {n_sb} (K={}) leaves rows \
                 unaligned; repack rows at load time",
                    act.k()
                ),
            ));
        }
        if w.cols() != 110 * n_sb / 4 {
            return Err(GpuError::shape(
                "enqueue_gemv_q3k_sel",
                format!(
                    "Q3_K rows are 110*{n_sb}/4 = {} words at K={}, got {}",
                    110 * n_sb / 4,
                    act.k(),
                    w.cols()
                ),
            ));
        }
        if rows_per_expert == 0 || !w.rows().is_multiple_of(rows_per_expert) {
            return Err(GpuError::shape(
                "enqueue_gemv_q3k_sel",
                format!(
                    "w.rows() {} must be a positive multiple of \
                 rows_per_expert {rows_per_expert}",
                    w.rows()
                ),
            ));
        }
        if n_slots == 0 || sel.len() < n_slots {
            return Err(GpuError::shape(
                "enqueue_gemv_q3k_sel",
                format!(
                    "need n_slots >= 1 and sel.len() >= n_slots, got \
                 n_slots {n_slots} sel.len() {}",
                    sel.len()
                ),
            ));
        }
        if y.len() < n_slots * rows_per_expert {
            return Err(GpuError::shape(
                "enqueue_gemv_q3k_sel",
                format!(
                    "y.len() {} < n_slots*rows_per_expert = {}",
                    y.len(),
                    n_slots * rows_per_expert
                ),
            ));
        }
        let what = "enqueue_gemv_q3k_sel";
        let grid = launch_u32(what, "grid", (n_slots * rows_per_expert).div_ceil(8))?;
        let n_experts = launch_u32(what, "n_experts", w.rows() / rows_per_expert)?;
        let rows_per_expert = launch_u32(what, "rows_per_expert", rows_per_expert)?;
        let n_slots = launch_u32(what, "n_slots", n_slots)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_q3k_gemv_sel(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.q3k_gemv_sel(
            &self.stream,
            &prep,
            w.buf(),
            &act.q3,
            &act.d8,
            sel,
            n_experts,
            rows_per_expert,
            n_slots,
            n_sb,
            n_sb.div_ceil(2),
            self.unlabelled_sink(),
            y,
        )?;
        Ok(())
    }

    /// Enqueue `y = w · act` for a Q6_K weight of `w.rows()` rows — 210
    /// bytes per super-block, the row stream as u32 words with the final
    /// word zero-padded, `210 * n_sb / 4` words per row (an integer only
    /// for even n_sb; an odd n_sb leaves rows unaligned and needs load-time
    /// repacking, rejected here) — against the quantized activations in
    /// `act`, which supply K. `y` holds `rows * m` f32, row-major with `m`
    /// outputs per row. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q6k(
        &self,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, m) = (w.rows(), act.m());
        let n_sb = act.n_sb();
        if !n_sb.is_multiple_of(2) {
            return Err(GpuError::shape(
                "enqueue_gemv_q6k",
                format!(
                    "odd super-block count {n_sb} (K={}) leaves rows \
                 unaligned; repack rows at load time",
                    act.k()
                ),
            ));
        }
        if w.cols() != 210 * n_sb / 4 {
            return Err(GpuError::shape(
                "enqueue_gemv_q6k",
                format!(
                    "Q6_K rows are 210*{n_sb}/4 = {} words at K={}, got {}",
                    210 * n_sb / 4,
                    act.k(),
                    w.cols()
                ),
            ));
        }
        if y.len() < n_rows * m {
            return Err(GpuError::shape(
                "enqueue_gemv_q6k",
                format!("y.len() {} < rows*m = {}", y.len(), n_rows * m),
            ));
        }
        let what = "enqueue_gemv_q6k";
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let m = launch_u32(what, "m", m)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_q6k_gemv(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module.q6k_gemv(
            &self.stream,
            &prep,
            w.buf(),
            &act.q6,
            &act.d8,
            n_rows,
            m,
            n_sb,
            n_sb.div_ceil(2),
            y,
        )?;
        Ok(())
    }

    /// Q4_K (K=2048) gemv through the shared q8_1 activation quantizer, with
    /// upload, scratch allocation and copy-back inside the call — the
    /// stage-0 correctness shape, not the step shape.
    ///
    /// `w` is `n_rows * 288` little-endian u32 words of Q4_K rows (144 B
    /// per super-block, 8 super-blocks per row); `x` is `m` activation
    /// columns of 2048 f32 each, concatenated; the result is `n_rows * m`
    /// f32, row-major with `m` outputs per row.
    pub fn gemv_q4k(
        &self,
        w: &[u32],
        x: &[f32],
        n_rows: usize,
        m: usize,
    ) -> Result<Vec<f32>, GpuError> {
        check_q4k_geometry(w.len(), x.len(), n_rows, m)?;
        let stream = &self.stream;
        let w_dev = DeviceTensor::upload(stream, &w[..n_rows * 288], n_rows, 288)?;
        let x_dev = DeviceBuffer::from_host(stream, &x[..m * 2048])?;
        let mut act = Q8Act::new(stream, m)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_rows * m)?;
        self.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        self.enqueue_gemv_q4k(&w_dev, &act, &mut y_dev)?;
        stream.synchronize()?;
        Ok(y_dev.to_host_vec(stream)?)
    }

    /// Rough launch-cost probe for design, NOT a benchmark of record: stages
    /// `w`/`x` on the device and quantizes once, then times `iters` bare
    /// q4k_gemv launches, each immediately followed by a stream
    /// synchronize. Returns the mean microseconds per launch+sync.
    pub fn probe_q4k_launch_us(
        &self,
        w: &[u32],
        x: &[f32],
        n_rows: usize,
        m: usize,
        iters: u32,
    ) -> Result<f64, GpuError> {
        check_q4k_geometry(w.len(), x.len(), n_rows, m)?;
        if iters == 0 {
            return Err(GpuError::shape("probe_q4k_launch_us", "iters must be >= 1"));
        }
        let stream = &self.stream;
        let w_dev = DeviceTensor::upload(stream, &w[..n_rows * 288], n_rows, 288)?;
        let x_dev = DeviceBuffer::from_host(stream, &x[..m * 2048])?;
        let mut act = Q8Act::new(stream, m)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n_rows * m)?;
        self.enqueue_quantize_q8_1(&x_dev, &mut act)?;

        let mut launch_gemv = || -> Result<(), GpuError> {
            self.enqueue_gemv_q4k(&w_dev, &act, &mut y_dev)?;
            stream.synchronize()?;
            Ok(())
        };
        for _ in 0..2 {
            launch_gemv()?;
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            launch_gemv()?;
        }
        Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters))
    }
}

/// Reject geometry the kernels' launch contracts do not cover: the q4k gemv
/// reads 288 words per row, the quantizer consumes 2048 values per column
/// and both support 1..=8 columns.
fn check_q4k_geometry(w_len: usize, x_len: usize, n_rows: usize, m: usize) -> Result<(), GpuError> {
    if n_rows == 0 || !(1..=8).contains(&m) {
        return Err(GpuError::shape(
            "gemv_q4k",
            format!("need n_rows >= 1 and 1 <= m <= 8, got n_rows={n_rows} m={m}"),
        ));
    }
    if w_len < n_rows * 288 {
        return Err(GpuError::shape(
            "gemv_q4k",
            format!("w.len() {w_len} < n_rows*288 = {}", n_rows * 288),
        ));
    }
    if x_len < m * 2048 {
        return Err(GpuError::shape(
            "gemv_q4k",
            format!("x.len() {x_len} < m*2048 = {}", m * 2048),
        ));
    }
    Ok(())
}
