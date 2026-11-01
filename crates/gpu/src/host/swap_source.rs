//! The model side of the residency machine as common code
//! ([`super::swap`]): where any placed load's routed experts come from
//! ([`FileSwap`], a [`SwapSource`] over the model file and the load's host
//! set), and the [`ResidencyGlue`] a body delegates its residency calls to.
//!
//! **Parts.** An expert is its model's parts in stack order — V4.1's gate,
//! up and down; a model names its own ([`FileStacks`]). On the card each
//! part of slot `s` is the file's bytes of that expert at byte `s · part` of
//! its layer's stack, which the placed load uploads as file bytes in slot
//! order; an NVFP4 stack, which the card holds as planes, is three parts a
//! stack, each plane staged from the file's bytes by its transform
//! ([`Transform`]: a part is its target's length, not its source's). The parts are each layer's own as the file holds them: a model
//! whose layers quantize their stacks differently (a layer's gate and up of
//! another type) still stages every expert as its layer lays it out, and the
//! staging ring's slot is sized by the widest layer.
//!
//! **Sources.** The host reads a part from the r8 sidecar when the load
//! reads one (`BLOOMERY_R8`) and the sidecar holds its stack, else from the
//! source file — the bytes the load's host set holds. A part whose staged
//! bytes are not its slot's layout is turned into them in place by the
//! model's hook ([`Convert`]): V4.1's gates and ups under the sidecar hold
//! the r8 row-lane layout, unpacked into Q3_K on the copy stream.
//!
//! **The arena.** On a paged plan ([`NvTier`]) an id the arena holds filled
//! is read from its slot, part by part, the bytes the arena's fill wrote —
//! the file's, a planes matrix's codes then its block scales — while the
//! lane's open read pins it ([`NvTier::open_read`]);
//! a part the sidecar holds still reads the sidecar, whose bytes are not the
//! arena's, and an id unfilled when the lane opens it reads the file's
//! mapping. An id the card then holds gives its slot back when its flip
//! lands ([`SwapSource::release_host`]), so an expert is in one tier, and a
//! victim the arena serves fills its slot on the lane's own thread.
//!
//! **Host residency.** The host serves an expert from resident pages when
//! every byte it reads for it lies in the load's host set and is in the page
//! cache now ([`HostSet::serves`], `mincore`): the set says what the load
//! read in (and locked, under `BLOOMERY_HOST_LOCK`), the page cache what a
//! step would fault on. A victim in the set whose pages the page cache let
//! go since the load or the staging thread's prepare (a set not locked) is
//! read in again by the machine before it decides
//! ([`SwapSource::prepare_victim`]) and counted ([`PassReport::rereads`]): a
//! page fault's cost once. A victim outside the set, or one whose pages the
//! page cache lets go again before the machine looks, is not host-resident:
//! the machine counts it ([`PassReport::unresident`]) and refuses a call's
//! pick, before any move, or sends it on all the same, the host reading it
//! from the file's mapping, and counts the passes it is served so
//! ([`PassReport::faulting`]). The
//! load puts each layer's churn pool — its stage card experts past the
//! pinned ones ([`ChurnPool`]) — in the set
//! ([`crate::model::GpuModel::load_placed_with`]): read in with it where
//! [`ResidencySpec::pool`] says hosted, else known only, so that each victim
//! of it is read back from the mapping by the prepare.

use std::borrow::Cow;
use std::ops::Range;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use bloomery_levers::ChurnHosting;
use cuda_core::{CudaContext, CudaStream, sys};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::names::routed_stacks;
use model::placement::host_lock::{HostFile, HostSet, expert_runs};
use model::placement::{ModelTensor, ModelTensors, Plan};
use model::r8file::R8Pair;
use runtime::swaprule::KeptRows;

use super::HostExperts;
use super::nvtier::NvTier;
use super::swap::{
    BoundaryAt, MachineCfg, Nvfp4From, PassReport, Piece, ResetReport, Residency, SwapSource,
    Transform,
};
use super::{HostTier, PassKind};
use crate::GpuError;
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights};

// ----------------------------------------------------------- the stacks

/// A model's routed expert stacks for [`FileSwap`]: where each layer's stacks
/// are in the file and on the card, the types its kernels read them in, and
/// the convert step its card layout needs. Every offloading model implements
/// this once beside its body; V4.1's is `Ds41Stacks`
/// (`gpu_deepseek41::swap`).
pub trait FileStacks: Send + Sync {
    /// The file names of layer `layer`'s routed expert stacks, in stack
    /// order.
    fn names(&self, layer: usize) -> Vec<String>;

    /// The K-quant type the model's kernels read layer `layer`'s stacks in,
    /// in the same order; a card stack of another type is refused by name.
    /// A layer's list is its own as the file holds it: layers may differ.
    fn types(&self, layer: usize) -> &[GgmlType];

    /// The layers the body's slot map holds, when it is not the load's whole
    /// card range — a body whose host tier serves a run of the card's layers
    /// builds its map over the run, and the machine's per-layer lists (its
    /// pinned counts) cover the map's layers. `None`: the card's layers.
    fn map_layers(&self) -> Option<usize> {
        None
    }

    /// The map's layers no pass routes, by their model numbers
    /// ([`super::swap::MachineCfg::unrouted`]): a draft layer the host tier
    /// serves beside the chain, with no card slot. Empty by default: every
    /// layer of the map is routed by the passes.
    fn unrouted(&self) -> Vec<usize> {
        Vec::new()
    }

    /// The load's convert step over the stacks the load found: `dims` the
    /// first card layer's first stack's dims, `parts` that layer's parts,
    /// each part's bytes an expert, `sidecar` whether the host reads the r8
    /// sidecar — `None` when the staged bytes are the slot's. A model with a
    /// shape contract of its own on the stacks refuses here, by name; a
    /// convert step whose sizes the layers do not share refuses the layers
    /// that differ. Load-time only.
    fn open(
        &self,
        dims: &[u64],
        parts: &[usize],
        sidecar: bool,
        ctx: &Arc<CudaContext>,
    ) -> Result<Option<Arc<dyn Convert>>, GpuError>;
}

/// The convert step of a [`FileSwap`] ([`FileStacks::open`]): what turns a
/// staged part's bytes, just copied to their slot, into the slot's layout in
/// place, on the machine's copy stream. The copies of later parts and the
/// flip's event follow it on the stream.
pub trait Convert: Send + Sync {
    /// Enqueue on `stream` the conversion of part `part` at `dst`.
    fn convert(
        &self,
        part: usize,
        dst: sys::CUdeviceptr,
        stream: &CudaStream,
    ) -> Result<(), GpuError>;
}

/// Passes from the boundary that makes a flip to the one it lands at, for a
/// model whose card holds the file's bytes ([`PlanStacks`]). The victims are
/// host-resident (the churn pool), so a flip waits on no NVMe read, only on
/// its staging and its copy: a planning pass makes at most the rule's `cap`
/// flips ([`runtime::swaprule::SwapParams::mid`]), each one expert's memcpy
/// into the pinned ring on a lane thread and one H2D copy over the card's
/// link, an expert's copy running under the next one's memcpy (the ring holds
/// [`super::lane::RING_SLOTS`] experts). The lane stages on up to
/// [`super::lane::LANE_THREADS`] threads, and a flip the window opened on
/// one at a time. A whole pass's staging and copies —
/// its experts' bytes over the ring's rate beside the host leg — stay far
/// inside two passes' wall of decode steps at the model's width [derived],
/// so two passes hold it. The staging runs in the host leg's wait window, a
/// share of the step this derivation does not price: a late staging makes
/// the host wait at the landing and a late copy the engine stream, and
/// neither moves the boundary a flip lands at.
pub const LIVE_DELAY: u64 = 2;

/// The bound on every host wait of the machine. A boundary runs after the
/// last launched pass's host service has returned: before a launch, or ahead
/// of the next pass after a step's kept row and before its readback
/// (`GpuModel::run_tokens`), while the engine stream still runs that step's
/// last kernels and its head. So the engine stream need not be empty, but
/// nothing on it waits on this thread: every host word its waits read was
/// written by a service that has returned. The one host wait a boundary
/// makes is for a landing flip's staging. The flip's job `n` was issued
/// `LIVE_DELAY` = 2 boundaries back (a live delay of 1 or more, which the
/// machine holds), and its staging waits only for its ring slot's previous
/// copy, an earlier job issued at or before that boundary. That copy
/// waits on the copy stream for its own staging (an earlier, due job: the
/// same argument) and for the boundary event recorded when it was issued,
/// which precedes the last launched pass in the engine stream's order, so
/// the stream reaches it with no further host action — the waits before it
/// are copies whose staging the host already waited for at their own
/// landing, and the host words of passes already served. No wait can
/// close a cycle through the engine stream, and a wait is for the staging
/// thread and the copy stream alone: at most a planning pass's `cap`
/// experts ([`runtime::swaprule::SwapParams::mid`]), far inside this bound.
/// A late copy landing at a boundary made ahead delays the step's readback,
/// which the engine stream orders after that boundary's wait.
pub const DEADLINE: Duration = Duration::from_secs(30);

/// The family's contract on one complete layer's stack types
/// ([`StackFacts::check`]): the layer's number and its gate, up and down
/// types as the file holds them, `Err` the family's own refusal.
pub type LayerCheck = fn(usize, [GgmlType; 3]) -> Result<(), GpuError>;

/// The family facts a [`PlanStacks`] read takes: what differs per model
/// about its three routed stacks, every field the family's own.
pub struct StackFacts {
    /// The refusal name of the family's constructor, the `what` its
    /// load-time refusals carry.
    pub what: &'static str,
    /// Layer `l`'s gate, up and down file names, in stack order.
    pub names: fn(usize) -> [String; 3],
    /// The family's contract on a complete layer's stack types, checked
    /// layer by layer; `None`: the family's place rule owns it.
    pub check: Option<LayerCheck>,
    /// Whether a layer may hold none of its stacks (a dense block).
    pub empty: bool,
    /// The layers the body's slot map holds ([`FileStacks::map_layers`]).
    pub map_layers: Option<usize>,
    /// The map's layers no pass routes ([`FileStacks::unrouted`]).
    pub unrouted: Vec<usize>,
}

/// The [`FileStacks`] of a model whose card holds the file's bytes: each
/// layer's gate, up and down types read from the plan's tensors, nothing to
/// convert, every family fact an input ([`StackFacts`]). With nothing to
/// convert, an r8 sidecar beside the file is refused by name.
pub struct PlanStacks {
    what: &'static str,
    names_of: fn(usize) -> [String; 3],
    /// Per layer, its three stacks' types in stack order; `None` on a layer
    /// that holds none of them.
    types: Vec<Option<[GgmlType; 3]>>,
    map_layers: Option<usize>,
    unrouted: Vec<usize>,
}

/// The stacks of a hybrid family whose routed layers are one run of `run_len` layers: the common stack names
/// ([`routed_stacks`]), no family check, the map holding the run's layers and none unrouted; `empty` whether a
/// layer of the file may hold none of its stacks (a dense block). Refused by name as `what`'s, as [`PlanStacks::of`].
/// Load-time only.
pub fn hosted_stacks(
    model: &ModelTensors,
    what: &'static str,
    (run_len, empty): (usize, bool),
) -> Result<PlanStacks, GpuError> {
    PlanStacks::of(
        model,
        StackFacts {
            what,
            names: routed_stacks,
            check: None,
            empty,
            map_layers: Some(run_len),
            unrouted: Vec::new(),
        },
    )
}

impl PlanStacks {
    /// The stacks of `model`'s layers by the facts' names. Refused by name
    /// under the facts' `what`: a layer missing some of its three stacks
    /// (all three, unless the facts allow an empty layer), and a complete
    /// layer the facts' `check` refuses. Load-time only.
    pub fn of(model: &ModelTensors, facts: StackFacts) -> Result<PlanStacks, GpuError> {
        let find = |name: &str| model.tensors.iter().find(|t| t.name == name).map(|t| t.ty);
        let mut types = Vec::with_capacity(model.layers);
        for l in 0..model.layers {
            let names = (facts.names)(l);
            let found: [Option<GgmlType>; 3] = [find(&names[0]), find(&names[1]), find(&names[2])];
            types.push(match found {
                [None, None, None] if facts.empty => None,
                [Some(gate), Some(up), Some(down)] => {
                    if let Some(check) = facts.check {
                        check(l, [gate, up, down])?;
                    }
                    Some([gate, up, down])
                }
                _ => {
                    let (name, _) = names
                        .iter()
                        .zip(found)
                        .find(|(_, t)| t.is_none())
                        .expect("a layer with a missing stack names it");
                    return Err(GpuError::Tensor {
                        what: facts.what,
                        name: name.clone(),
                        need: if facts.empty {
                            "all three routed stacks: a layer holds either all of them or none"
                        } else {
                            "every layer's three routed stacks"
                        },
                    });
                }
            });
        }
        Ok(PlanStacks {
            what: facts.what,
            names_of: facts.names,
            types,
            map_layers: facts.map_layers,
            unrouted: facts.unrouted,
        })
    }

    /// Layer `l`'s gate·up type and down type, as [`FileStacks::types`] lists
    /// them; `None` past the model's layers and on a layer that holds none.
    #[must_use]
    pub fn pair(&self, l: usize) -> Option<(GgmlType, GgmlType)> {
        self.types
            .get(l)
            .and_then(Option::as_ref)
            .map(|t| (t[0], t[2]))
    }

    /// The r8 sidecar's refusal ([`FileStacks::open`]): the sidecar is the
    /// Q3_K stacks' r8 layout (`qdot::repack_q3k_r8`) and this table has no
    /// unpack from it, so a part a sidecar held would stage into its slot as
    /// r8 bytes. Refused by name under the facts' `what` whatever the file
    /// holds, the message listing the types its stacks are. `Ok` without a
    /// sidecar.
    fn check_sidecar(&self, sidecar: bool) -> Result<(), GpuError> {
        if !sidecar {
            return Ok(());
        }
        let mut found: Vec<GgmlType> = Vec::new();
        for &ty in self.types.iter().flatten().flatten() {
            if !found.contains(&ty) {
                found.push(ty);
            }
        }
        let found: Vec<String> = found.iter().map(ToString::to_string).collect();
        Err(GpuError::shape(
            self.what,
            format!(
                "an r8 sidecar beside a file whose routed stacks are {}: the sidecar's format is \
                 Q3_K's (qdot::repack_q3k_r8) and this table has no unpack from it, so a staged \
                 part is the file's bytes as they stand (BLOOMERY_R8=off)",
                found.join(", ")
            ),
        ))
    }
}

impl FileStacks for PlanStacks {
    fn names(&self, layer: usize) -> Vec<String> {
        (self.names_of)(layer).to_vec()
    }

    fn types(&self, layer: usize) -> &[GgmlType] {
        self.types
            .get(layer)
            .and_then(Option::as_ref)
            .map_or(&[], |t| t.as_slice())
    }

    fn map_layers(&self) -> Option<usize> {
        self.map_layers
    }

    fn unrouted(&self) -> Vec<usize> {
        self.unrouted.clone()
    }

    fn open(
        &self,
        _dims: &[u64],
        _parts: &[usize],
        sidecar: bool,
        _ctx: &Arc<CudaContext>,
    ) -> Result<Option<Arc<dyn Convert>>, GpuError> {
        self.check_sidecar(sidecar)?;
        Ok(None)
    }
}

/// What a placed load under [`Residency`] asks the load path for
/// ([`crate::model::GpuModel::load_placed_with`]): the lever's value with the
/// model's own constants and stacks. The live delay and the deadline are the
/// model's step-length function, not the machine's; a model passes its own.
pub struct ResidencySpec {
    /// The lever's value ([`Residency::parse`]): `off`, or the rule's `mid`
    /// parameters with the pinned seed experts a layer.
    pub lever: Residency,
    /// Whether the load's host set holds the churn pool or only knows it,
    /// as the word's resolution decided ([`ChurnHosting`]).
    pub pool: ChurnHosting,
    /// Passes from the boundary that makes a flip to the one it lands at.
    pub delay: u64,
    /// The bound on every host wait the machine makes.
    pub deadline: Duration,
    /// Ids a row routes a layer, 1 to [`super::swap::TALLY_TOP_K`].
    pub top_k: usize,
    /// The most rows one pass runs.
    pub max_rows: usize,
    /// The model's routed stacks.
    pub stacks: Arc<dyn FileStacks>,
}

// --------------------------------------------------------------- source

/// One layer's parts: the file tensors (its stacks), the stage stacks' base
/// addresses and slots, and each part's bytes and what of its stack it is —
/// this layer's own, as the file holds its stacks. A stack is one part, or
/// three for an NVFP4 stack the card holds as planes ([`Plane`]), so `base`,
/// `parts` and `of` run over the parts and `tensors` over the stacks.
struct LayerParts {
    tensors: Vec<ModelTensor>,
    base: Vec<sys::CUdeviceptr>,
    slots: usize,
    parts: Vec<usize>,
    of: Vec<(usize, Plane)>,
    /// Per stack, its experts' tensor scales as little-endian bytes, 4 an
    /// expert: the [`Plane::Scale`] part's source; empty for a stack read
    /// whole.
    scales: Vec<Vec<u8>>,
}

/// What a part of an expert is of its stack: the stack's bytes whole (a
/// K-quant, IQ, Q5_1, Q8_0 or MXFP4 stack, the card holding the file's
/// blocks), or one plane of an NVFP4 stack the card holds as planes
/// (`DevWeight::Nvfp4Stack`): the codes, staged in the card's block order
/// ([`Transform::Nvfp4Codes`]), the block scales, staged checked
/// ([`Transform::Ue4m3Scales`]), and the expert's tensor scale. The two
/// planes come from the expert's matrix where a checkpoint folder keeps it
/// (NVFP4 rows, `gguf::Place::Stack`: [`Nvfp4From::Plane`]) or from its run
/// of GGUF's NVFP4 blocks, each plane split from the same run
/// ([`Nvfp4From::Blocks`]); the tensor scale is the f32 the folder's header
/// or the stack's `.scale` holds for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plane {
    Whole,
    Codes,
    BlockScales,
    Scale,
}

/// What the per-layer parts check reads of one stack name's resident weight
/// ([`Weights::get`]): `None` when the load holds no weight of the name,
/// else the resident stack's type, its words and rows, and its buffer's
/// device address — a file-block type (`DevWeight::KQuant`: a K-quant, IQ,
/// Q5_1, Q8_0 or MXFP4 stack as the file stores its blocks), or NVFP4
/// planes (`DevWeight::Nvfp4Stack`, `ty` `None`: either NVFP4 file type loads
/// in them, [`FoundStack::holds`]), whose words, rows and address are its
/// code plane's and `nvfp4` its block-scale plane's words and address, then
/// its tensor scales' count and address. A weight of another device format
/// has neither `ty` nor `nvfp4` — the rest stands unread — and the check
/// refuses it beside its layer's other parts.
struct FoundStack {
    ty: Option<GgmlType>,
    words: usize,
    rows: usize,
    ptr: sys::CUdeviceptr,
    nvfp4: Option<[(usize, sys::CUdeviceptr); 2]>,
}

impl FoundStack {
    /// The parts check's view of one resident weight.
    fn of(d: &DevWeight) -> FoundStack {
        match d {
            DevWeight::KQuant { ty, w, .. } => FoundStack {
                ty: Some(*ty),
                words: w.buf().len(),
                rows: w.rows(),
                ptr: w.buf().cu_deviceptr(),
                nvfp4: None,
            },
            DevWeight::Nvfp4Stack { qs, e, scales, .. } => FoundStack {
                ty: None,
                words: qs.buf().len(),
                rows: qs.rows(),
                ptr: qs.buf().cu_deviceptr(),
                nvfp4: Some([
                    (e.buf().len(), e.buf().cu_deviceptr()),
                    (scales.buf().len(), scales.buf().cu_deviceptr()),
                ]),
            },
            _ => FoundStack {
                ty: None,
                words: 0,
                rows: 0,
                ptr: 0,
                nvfp4: None,
            },
        }
    }

    /// Whether the weight is a resident stack of type `want`: its file
    /// blocks, or NVFP4 planes for either NVFP4 file type.
    fn holds(&self, want: GgmlType) -> bool {
        match self.nvfp4 {
            Some(_) => matches!(want, GgmlType::NVFP4 | GgmlType::NVFP4_ROWS),
            None => self.ty == Some(want),
        }
    }

    /// The stack's parts in stage order, each its base address and what of
    /// the stack it is: the stack whole, or its three planes.
    fn parts(&self) -> Vec<(sys::CUdeviceptr, Plane)> {
        match self.nvfp4 {
            None => vec![(self.ptr, Plane::Whole)],
            Some([(_, e), (_, s)]) => vec![
                (self.ptr, Plane::Codes),
                (e, Plane::BlockScales),
                (s, Plane::Scale),
            ],
        }
    }
}

/// Layer `l`'s parts and slots from what the load holds of its stacks:
/// `names` its stack names in stack order, `types` the types its kernels
/// read them in, `found` and `tensors` what the load's weights and the plan
/// hold of each name, and `experts` the routed experts a stack holds. A
/// stack of whole experts is one part of its tensor's `file_bytes /
/// experts`; an NVFP4 stack held as card planes is three: an expert's `rows ·
/// k / 2` code bytes, `rows · k / 16` block-scale bytes and its 4-byte
/// tensor scale (`dims` `[k, rows, experts]`), whose first two make the
/// tensor's bytes an expert. The slots are the last stack's rows over its
/// tensor's `dims[1]`. `Ok(None)`: the load holds none of the layer's
/// stacks. Refused by name, every error naming `FileSwap::new`, the entry
/// point the check serves: a names and types count that differs, a stack
/// not resident as its listed type, a name the plan does not hold, a part
/// not a whole number of 4-byte words, planes that are not the tensor's
/// bytes or not whole 64-value blocks, and a stack too short for its slots.
/// Pure host code: the tests below exercise layers whose layouts differ
/// without a model file.
fn layer_parts(
    l: usize,
    names: &[String],
    types: &[GgmlType],
    found: &[Option<FoundStack>],
    tensors: &[Option<&ModelTensor>],
    experts: u64,
) -> Result<Option<(Vec<usize>, usize)>, GpuError> {
    const WHAT: &str = "FileSwap::new";
    // All of the layer's stacks or none: a layer none of whose stacks the
    // load holds is skipped whatever its types list says.
    if found.iter().all(Option::is_none) {
        return Ok(None);
    }
    if names.len() != types.len() {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {l}: {} stack names for {} types",
                names.len(),
                types.len()
            ),
        ));
    }
    let stacks_of = found
        .iter()
        .zip(names)
        .zip(types)
        .map(|((d, n), want)| match d {
            Some(st) if st.holds(*want) => Ok(st),
            _ => Err(GpuError::tensor(
                WHAT,
                n.clone(),
                "a resident routed stack of its listed type (a K-quant, or NVFP4 as card planes) \
                 beside the layer's other parts",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tensors = names
        .iter()
        .zip(tensors)
        .map(|(n, t)| {
            t.ok_or_else(|| GpuError::tensor(WHAT, n.clone(), "a routed stack of the plan"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let per: Vec<usize> = tensors.iter().map(|t| file_part(t, experts)).collect();
    let slots =
        stacks_of[stacks_of.len() - 1].rows / tensors[tensors.len() - 1].dims[1].max(1) as usize;
    let mut parts = Vec::with_capacity(per.len());
    for (i, st) in stacks_of.iter().enumerate() {
        let Some(planes) = st.nvfp4 else {
            if per[i] == 0 || !per[i].is_multiple_of(4) || st.words * 4 < slots * per[i] {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l} part {i}: {} bytes an expert, a stack of {} words for \
                         {slots} slots",
                        per[i], st.words
                    ),
                ));
            }
            parts.push(per[i]);
            continue;
        };
        let dim = |d: usize| tensors[i].dims.get(d).copied().unwrap_or(0) as usize;
        let (k, rows) = (dim(0), dim(1));
        // Bytes an expert in each plane, and the words (f32s for the scales)
        // each plane holds.
        let bytes = [rows * (k / 2), rows * (k / 16), 4];
        let words = [st.words, planes[0].0, planes[1].0];
        if k == 0
            || !k.is_multiple_of(64)
            || rows == 0
            || bytes[0] + bytes[1] != per[i]
            || words.iter().zip(&bytes).any(|(&w, &b)| w * 4 < slots * b)
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "stack {}: NVFP4 planes of {k} values a row, {rows} rows and {} bytes an \
                     expert, planes of {}, {} and {} words for {slots} slots",
                    names[i], per[i], st.words, planes[0].0, planes[1].0
                ),
            ));
        }
        parts.extend(bytes);
    }
    Ok(Some((parts, slots)))
}

/// The file bytes one expert of stack `t` of `experts` experts takes: its
/// run, 0 for a stack of no expert or one past `usize`.
fn file_part(t: &ModelTensor, experts: u64) -> usize {
    usize::try_from(t.file_bytes / experts.max(1)).unwrap_or(0)
}

/// The tensor scales of the NVFP4 stack `t` of `experts` experts, as
/// little-endian bytes, 4 an expert: the [`Plane::Scale`] parts' source. NVFP4
/// rows read them where the folder's header holds them (`gguf::Place::Stack`),
/// GGUF's NVFP4 blocks from the stack's `.scale` (`gguf::tensor_scales`, the
/// load's one reader of them). Refused by name: NVFP4 rows the split does not
/// locate as matrices, scales the reader refuses, and scales that do not
/// number the stack's experts.
fn tensor_scales(split: &Split, t: &ModelTensor, experts: u64) -> Result<Vec<u8>, GpuError> {
    const WHAT: &str = "FileSwap::new";
    let scales = match split.place(&t.name) {
        Some(gguf::Place::Stack { scales, .. }) => scales.clone(),
        _ if t.ty == GgmlType::NVFP4 => {
            let (_, info) = split.find(&t.name).ok_or(GpuError::tensor(
                WHAT,
                t.name.clone(),
                "a routed stack of the split",
            ))?;
            crate::weights::tensor_scales(WHAT, info, |n| {
                let (s, i) = split.find(n)?;
                Some((split.shard(s)?, i))
            })?
            .unwrap_or_default()
        }
        _ => {
            return Err(GpuError::tensor(
                WHAT,
                t.name.clone(),
                "NVFP4 rows a checkpoint folder locates, each expert's matrix where the folder \
                 keeps it, beside its tensor scale",
            ));
        }
    };
    if scales.len() as u64 != experts {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "stack {}: {} tensor scales for {experts} experts",
                t.name,
                scales.len()
            ),
        ));
    }
    Ok(scales.iter().flat_map(|s| s.to_le_bytes()).collect())
}

/// The one convert step is built from the first card layer's parts: a card
/// layer of other parts would be unpacked with the wrong sizes, so a load
/// with a convert step refuses it by name. `first` is the layer `layers[0]`
/// holds.
fn one_layout(first: usize, layers: &[Option<LayerParts>]) -> Result<(), GpuError> {
    let mut held = layers
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|p| (first + i, p)));
    let Some((l0, p0)) = held.next() else {
        return Ok(());
    };
    match held.find(|(_, p)| p.parts != p0.parts) {
        None => Ok(()),
        Some((l, p)) => Err(GpuError::shape(
            "FileSwap::new",
            format!(
                "layer {l}: parts {:?} beside layer {l0}'s {:?}, which the convert step unpacks",
                p.parts, p0.parts
            ),
        )),
    }
}

/// A placed load's [`SwapSource`] over its model file: each part of an expert
/// as the file (or the r8 sidecar the host reads) holds it, into the slot's
/// place in the layer's stage stack, with the load's host set saying what the
/// host can serve.
pub struct FileSwap {
    /// The NVMe expert tier's arena, on a paged plan, attached by the body
    /// once it built the tier ([`FileSwap::attach_tier`]): the residency
    /// seam answers for the ids the arena serves from its books. Unset on a
    /// resident plan, and for every id the mapping serves.
    tier: OnceLock<Arc<NvTier>>,
    pair: R8Pair,
    set: HostSet,
    experts: u64,
    first: usize,
    layers: Vec<Option<LayerParts>>,
    /// The host layout of every routed tensor, including layers with no stage stack.
    union_tag: String,
    /// The model's convert step, when its staged bytes are not its slots'.
    convert: Option<Arc<dyn Convert>>,
}

/// The whole model's host expert layout, including wholly host-resident layers. Every routed
/// tensor's type is included: a gate/down pair alone does not identify a separate up stack.
#[must_use]
pub fn union_identity(model: &ModelTensors, r8: bool) -> String {
    runtime::costs::union_source_tag(
        model
            .tensors
            .iter()
            .filter(|t| t.role == model::placement::Role::RoutedExperts)
            .map(|t| (t.layer, t.ty.as_u32(), t.dims.as_slice())),
        r8,
    )
}

// SAFETY: the stack addresses are plain device pointers into allocations the
// model keeps in place until the machine's copy stream has drained (the
// model's drop stops the machine before it frees them); the convert step is
// used from the one thread that drives the machine, and everything the
// staging thread reads (`source`, `prepare_victim`) is the split's and the
// sidecar's read-only mappings and the set.
unsafe impl Send for FileSwap {}
// SAFETY: as for `Send`: no call mutates shared state, the tier cell is set once
// at load before the machine starts, and the arena is `Sync` by its own contract.
unsafe impl Sync for FileSwap {}

impl FileSwap {
    /// The source of `plan`'s layers `layers`, whose routed stacks the load
    /// holds in `w`: the split `file` and the host reading `r8` the load took
    /// (the same sidecar open as the host set's), `set`, the host set the
    /// load read in (`populated` saying it was), and `stacks`, the model's
    /// stacks. Refused by name: a set the load did not populate, a layer
    /// whose stacks are not all resident in the types the model reads, a
    /// stack the plan does not name, parts not whole words, and the model's
    /// own contract
    /// ([`FileStacks::open`]). Load-time only.
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's plan, layers, file, host reading, set, weights, stacks and context (rust-quality R8)"
    )]
    pub fn new(
        plan: &Plan<'_>,
        layers: Range<usize>,
        file: Arc<Split>,
        r8: bool,
        set: HostSet,
        populated: bool,
        w: &Weights,
        stacks: &dyn FileStacks,
        ctx: &Arc<CudaContext>,
    ) -> Result<FileSwap, GpuError> {
        const WHAT: &str = "FileSwap::new";
        if !populated {
            return Err(GpuError::shape(
                WHAT,
                "adaptive residency on a load whose host set was not read in \
                 (BLOOMERY_HOST_POPULATE=0): no expert would be host-resident",
            ));
        }
        let pair = R8Pair::at_load(file, r8).map_err(|e| GpuError::plan(WHAT, e))?;
        let union_tag = union_identity(plan.model, pair.r8().sidecar().is_some());
        let experts = plan.model.experts;
        let mut out = Vec::with_capacity(layers.len());
        for l in layers.clone() {
            let names = stacks.names(l);
            // The card's stacks first: all of the layer's or none. A layer
            // none of whose stacks the load holds (a dense block, a layer the
            // plan keeps whole on the host) is skipped whatever its types
            // list says.
            let found = names
                .iter()
                .map(|n| w.get(n).map(FoundStack::of))
                .collect::<Vec<_>>();
            let tensors = names
                .iter()
                .map(|n| plan.model.tensors.iter().find(|t| t.name == *n))
                .collect::<Vec<_>>();
            let Some((parts, slots)) =
                layer_parts(l, &names, stacks.types(l), &found, &tensors, experts)?
            else {
                out.push(None);
                continue;
            };
            // The check held every name: a resident stack of its listed type
            // and a tensor of the plan.
            let tensors: Vec<ModelTensor> = tensors
                .into_iter()
                .map(|t| t.cloned().expect("a plan tensor the check held"))
                .collect();
            let (mut base, mut of) = (Vec::new(), Vec::new());
            for (i, st) in found.iter().enumerate() {
                for (ptr, plane) in st
                    .as_ref()
                    .expect("a resident stack the check held")
                    .parts()
                {
                    base.push(ptr);
                    of.push((i, plane));
                }
            }
            let scales = tensors
                .iter()
                .enumerate()
                .map(|(i, t)| match of.contains(&(i, Plane::Scale)) {
                    true => tensor_scales(pair.source().split(), t, experts),
                    false => Ok(Vec::new()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            out.push(Some(LayerParts {
                tensors,
                base,
                slots,
                parts,
                of,
                scales,
            }));
        }
        let first = out
            .iter()
            .flatten()
            .next()
            .ok_or_else(|| GpuError::shape(WHAT, "no layer of the card holds routed stacks"))?;
        let t0 = first.tensors[0].clone();
        let convert = stacks.open(&t0.dims, &first.parts, pair.r8().sidecar().is_some(), ctx)?;
        if convert.is_some() {
            one_layout(layers.start, &out)?;
        }
        Ok(FileSwap {
            tier: OnceLock::new(),
            pair,
            set,
            experts,
            first: layers.start,
            layers: out,
            union_tag,
            convert,
        })
    }

    /// Give the source the NVMe expert tier's arena the body built for the
    /// plan's paged layers. Load-time only, before the machine starts and
    /// once: a second attach is refused by name, and so is a tier over
    /// another open of the model file (its drops would not reach the pages
    /// the lanes read) and a load whose host set holds a page the tier drops
    /// after each read — a churn pool on a paged stack, whose pages the
    /// drops would take from under the host ([`NvTier::refuse_held_pages`]).
    pub fn attach_tier(&self, tier: &Arc<NvTier>) -> Result<(), GpuError> {
        const WHAT: &str = "FileSwap::attach_tier";
        if !tier.drops_reach(self.pair.split()) {
            return Err(GpuError::shape(
                WHAT,
                "the NVMe tier was built over another open of the model file than the source \
                 reads: its drops would leave the lanes' pages cached",
            ));
        }
        tier.refuse_held_pages(&self.set)?;
        self.tier
            .set(Arc::clone(tier))
            .map_err(|_| GpuError::state(WHAT, "a source with no tier yet"))
    }

    /// The arena, when it serves layer `layer`'s expert `id` ([`NvTier::serves`]).
    fn tier_serving(&self, layer: usize, id: u32) -> Option<&Arc<NvTier>> {
        self.tier.get().filter(|t| t.serves(layer, id))
    }

    /// Whether the load converts a staged part on the card before it goes
    /// live ([`Convert`]): under the r8 sidecar, the unpack of a staged gate
    /// or up.
    #[must_use]
    pub fn unpacks(&self) -> bool {
        self.convert.is_some()
    }

    /// Part `part` of layer `layer`'s expert `id` as a static load uploads it
    /// to a card slot, from the source file's bytes whatever the host reads:
    /// a stack read whole is the expert's run of the file; a plane of GGUF's
    /// NVFP4 blocks is that plane of the run as the load splits its rows
    /// ([`crate::planes::split_rows`]), and its tensor scale the f32 the
    /// stack's `.scale` holds. A plane of NVFP4 rows a checkpoint folder
    /// locates is that plane of the expert's matrix in the card's order: the
    /// codes moved into its blocks by the scalar rule
    /// (`qdot::nvfp4_codes_to_blocks_scalar`), the block scales as stored.
    pub fn card_bytes(
        &self,
        layer: usize,
        id: u32,
        part: usize,
    ) -> Result<Cow<'_, [u8]>, GpuError> {
        const WHAT: &str = "FileSwap::card_bytes";
        let p = self.layer(layer, WHAT)?;
        let (per, &(stack, plane)) = p
            .parts
            .get(part)
            .zip(p.of.get(part))
            .ok_or_else(|| GpuError::shape(WHAT, format!("part {part} of an expert")))?;
        let t = &p.tensors[stack];
        match plane {
            Plane::Whole => self.file_run(WHAT, layer, id, t, *per).map(Cow::Borrowed),
            Plane::Scale => p.scales[stack]
                .get(4 * id as usize..4 * id as usize + 4)
                .map(Cow::Borrowed)
                .ok_or_else(|| {
                    GpuError::shape(WHAT, format!("layer {layer} expert {id}: its scale"))
                }),
            Plane::Codes | Plane::BlockScales if t.ty == GgmlType::NVFP4 => {
                use crate::planes::{Format, split_rows};
                let run = self.file_run(WHAT, layer, id, t, file_part(t, self.experts))?;
                let k = t.dims.first().copied().unwrap_or(0) as usize;
                let n = run.len() / Format::Nvfp4.record_bytes();
                let mut codes = vec![0u8; n * Format::Nvfp4.code_bytes()];
                let mut heads = vec![0u8; n * Format::Nvfp4.head_bytes()];
                split_rows(Format::Nvfp4, k / 64, 0, run, &mut codes, &mut heads).map_err(|e| {
                    GpuError::shape(WHAT, format!("layer {layer} expert {id}: {e}"))
                })?;
                Ok(Cow::Owned(if plane == Plane::Codes {
                    codes
                } else {
                    heads
                }))
            }
            Plane::Codes | Plane::BlockScales => {
                let m = self
                    .pair
                    .source()
                    .split()
                    .matrix(&t.name, id as usize)
                    .map_err(|e| GpuError::plan(WHAT, e))?;
                if plane == Plane::BlockScales {
                    return Ok(Cow::Borrowed(m.block_scales));
                }
                let mut blocks = vec![0u8; m.codes.len()];
                qdot::nvfp4_codes_to_blocks_scalar(m.codes, &mut blocks).map_err(|e| {
                    GpuError::shape(WHAT, format!("layer {layer} expert {id}: {e}"))
                })?;
                Ok(Cow::Owned(blocks))
            }
        }
    }

    /// The `per` bytes of layer `layer`'s expert `id` in stack `t` as the
    /// split file holds them, refused by name as `what`'s.
    fn file_run(
        &self,
        what: &'static str,
        layer: usize,
        id: u32,
        t: &ModelTensor,
        per: usize,
    ) -> Result<&[u8], GpuError> {
        let split = self.pair.source().split();
        let (s, info) = split.find(&t.name).ok_or(GpuError::tensor(
            what,
            t.name.clone(),
            "a routed stack of the split",
        ))?;
        let whole = split
            .shard(s)
            .ok_or_else(|| GpuError::shape(what, format!("shard {s} of the split")))?
            .data(info)
            .map_err(|e| GpuError::plan(what, e))?;
        let at = id as usize * per;
        whole.get(at..at + per).ok_or_else(|| {
            GpuError::shape(
                what,
                format!(
                    "layer {layer} expert {id}: past the {} bytes of {}",
                    whole.len(),
                    t.name
                ),
            )
        })
    }

    /// Stack `t`'s bytes of layer `layer`'s expert `id` in the arena's slot,
    /// when the arena serves the id and the lane's open read found it filled
    /// ([`NvTier::staged_slot`]); `None` when the mapping serves it.
    fn slot_of(&self, layer: usize, id: u32, t: &ModelTensor) -> Result<Option<&[u8]>, GpuError> {
        match self.tier_serving(layer, id) {
            Some(tier) => tier.staged_slot(layer, id, &t.name),
            None => Ok(None),
        }
    }

    /// Matrix `id` of the stack `t` a checkpoint folder locates, its codes
    /// and its block scales: the arena's slot when it holds the id filled
    /// under the lane's open read (the codes, then the scales the fill moved
    /// after them), else where the folder keeps them
    /// ([`gguf::Split::matrix`]). A slot of another length is refused by
    /// name as `what`'s.
    fn matrix(
        &self,
        what: &'static str,
        layer: usize,
        id: u32,
        t: &ModelTensor,
    ) -> Result<(&[u8], &[u8]), GpuError> {
        let m = self
            .pair
            .source()
            .split()
            .matrix(&t.name, id as usize)
            .map_err(|e| GpuError::plan(what, e))?;
        let Some(slot) = self.slot_of(layer, id, t)? else {
            return Ok((m.codes, m.block_scales));
        };
        slot.split_at_checked(m.codes.len())
            .filter(|(_, scales)| scales.len() == m.block_scales.len())
            .ok_or_else(|| {
                GpuError::shape(
                    what,
                    format!(
                        "layer {layer} expert {id}: its slot holds {} bytes of {}, the matrix is \
                         {} + {}",
                        slot.len(),
                        t.name,
                        m.codes.len(),
                        m.block_scales.len()
                    ),
                )
            })
    }

    fn layer(&self, layer: usize, what: &'static str) -> Result<&LayerParts, GpuError> {
        layer
            .checked_sub(self.first)
            .and_then(|i| self.layers.get(i))
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                GpuError::shape(
                    what,
                    format!("layer {layer} holds no routed stack on the stage card"),
                )
            })
    }

    /// Where the host reads stack `stack` of layer `layer`'s expert `id`: the
    /// sidecar's run when it holds the stack and the pair reads one, else the
    /// source's; a planes stack's codes run then its scales run, each in its
    /// file, the second empty for a stack of whole runs ([`expert_runs`]).
    fn runs(
        &self,
        layer: usize,
        id: u32,
        stack: usize,
        what: &'static str,
    ) -> Result<[(HostFile, Range<u64>); 2], GpuError> {
        let t = &self.layer(layer, what)?.tensors[stack];
        expert_runs(self.pair.source(), t, self.experts, id, true)
            .map_err(|e| GpuError::plan(what, e))
    }

    /// The `per` bytes of layer `layer`'s expert `id` in stack `t` as the
    /// host's source holds them, one run: the r8 sidecar's when it holds the
    /// stack, else the arena's slot when it holds the id filled under the
    /// lane's open read (the file's bytes), else the file's.
    fn run(&self, layer: usize, id: u32, t: &ModelTensor, per: usize) -> Result<&[u8], GpuError> {
        const WHAT: &str = "FileSwap::source";
        let side = self
            .pair
            .source()
            .sidecar()
            .filter(|side| side.file_bytes(&t.name).is_some());
        // A part the sidecar does not hold, of an id the arena's slot holds
        // filled under the lane's open read: the slot's bytes, which are the
        // file's. The sidecar's bytes are not the arena's, so its parts read
        // the sidecar as they always did.
        if side.is_none()
            && let Some(bytes) = self.slot_of(layer, id, t)?
        {
            if bytes.len() != per {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {layer} expert {id}: its slot holds {} bytes of {}, the run is {per}",
                        bytes.len(),
                        t.name
                    ),
                ));
            }
            return Ok(bytes);
        }
        let Some(side) = side else {
            return self.file_run(WHAT, layer, id, t, per);
        };
        let whole = side.data(&t.name).map_err(|e| GpuError::plan(WHAT, e))?;
        let at = id as usize * per;
        whole.get(at..at + per).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!(
                    "layer {layer} expert {id}: past the {} bytes of {} in the sidecar",
                    whole.len(),
                    t.name
                ),
            )
        })
    }
}

impl SwapSource for FileSwap {
    fn union_tag(&self) -> Result<String, GpuError> {
        Ok(self.union_tag.clone())
    }

    fn part_bytes(&self, layer: usize) -> &[usize] {
        layer
            .checked_sub(self.first)
            .and_then(|i| self.layers.get(i))
            .and_then(Option::as_ref)
            .map_or(&[], |p| p.parts.as_slice())
    }

    /// A plane of NVFP4 rows reads the folder's matrix ([`FileSwap::matrix`]:
    /// the arena's slot, or where the folder keeps it) — its codes or its
    /// block scales — a plane of
    /// GGUF's NVFP4 blocks the expert's run of them ([`FileSwap::run`]), and a
    /// tensor scale the one the header or the stack's `.scale` holds, each
    /// with the transform that makes it the card plane's bytes ([`Plane`]); a
    /// stack read whole reads its run as it is.
    fn source(&self, layer: usize, id: u32, part: usize) -> Result<Piece<'_>, GpuError> {
        const WHAT: &str = "FileSwap::source";
        let p = self.layer(layer, WHAT)?;
        let (per, &(stack, plane)) = p
            .parts
            .get(part)
            .zip(p.of.get(part))
            .ok_or_else(|| GpuError::shape(WHAT, format!("part {part} of an expert")))?;
        let per = *per;
        let t = &p.tensors[stack];
        let (bytes, transform) = match plane {
            Plane::Whole => (Some(self.run(layer, id, t, per)?), Transform::Identity),
            Plane::Scale => (
                p.scales[stack].get(4 * id as usize..4 * id as usize + 4),
                Transform::Identity,
            ),
            Plane::Codes | Plane::BlockScales if t.ty == GgmlType::NVFP4 => {
                let run = self.run(layer, id, t, file_part(t, self.experts))?;
                let from = Nvfp4From::Blocks;
                match plane {
                    Plane::Codes => (Some(run), Transform::Nvfp4Codes(from)),
                    _ => (Some(run), Transform::Ue4m3Scales(from)),
                }
            }
            Plane::Codes | Plane::BlockScales => {
                let (codes, block_scales) = self.matrix(WHAT, layer, id, t)?;
                let from = Nvfp4From::Plane;
                match plane {
                    Plane::Codes => (Some(codes), Transform::Nvfp4Codes(from)),
                    _ => (Some(block_scales), Transform::Ue4m3Scales(from)),
                }
            }
        };
        match bytes.map(|bytes| Piece { bytes, transform }) {
            Some(piece) if piece.target_len() == Some(per) => Ok(piece),
            got => Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer} expert {id}: {plane:?} of {} is {} source bytes ({transform:?}), \
                     the part {per}",
                    t.name,
                    got.map_or(0, |p| p.bytes.len())
                ),
            )),
        }
    }

    /// An id the arena serves is open on the tier while a lane reads it
    /// ([`NvTier::open_read`]): filled, its slot is pinned against the LRU
    /// and read in place; unfilled, no union's drop of its layer takes the
    /// pages the lane has yet to copy. An id the mapping serves needs
    /// nothing.
    fn open_read(&self, layer: usize, id: u32) -> Result<(), GpuError> {
        match self.tier_serving(layer, id) {
            Some(tier) => tier.open_read(layer, id),
            None => Ok(()),
        }
    }

    /// An id the arena serves is let go once a lane is done with it
    /// ([`NvTier::end_read`]): a slot read unpins the slot; a mapping read
    /// leaves the mapping and the page cache, its pages having been the
    /// read's alone (the next read faults them in again). An id the mapping
    /// serves keeps its pages: the host set's.
    fn release_read(&self, layer: usize, id: u32) -> Result<(), GpuError> {
        match self.tier_serving(layer, id) {
            Some(tier) => tier.end_read(layer, id),
            None => Ok(()),
        }
    }

    fn dest(&self, layer: usize, part: usize, slot: u32) -> Result<sys::CUdeviceptr, GpuError> {
        const WHAT: &str = "FileSwap::dest";
        let p = self.layer(layer, WHAT)?;
        if slot as usize >= p.slots || part >= p.parts.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer} part {part} slot {slot}: {} slots of {} parts",
                    p.slots,
                    p.parts.len()
                ),
            ));
        }
        Ok(p.base[part] + (slot as usize * p.parts[part]) as u64)
    }

    /// The model's convert step of the staged part just copied to `dst`
    /// ([`Convert`]).
    fn convert(
        &self,
        _layer: usize,
        part: usize,
        dst: sys::CUdeviceptr,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        match &self.convert {
            Some(c) => c.convert(part, dst, stream),
            None => Ok(()),
        }
    }

    fn prepare_victim(&self, layer: usize, id: u32) -> Result<(), GpuError> {
        const WHAT: &str = "FileSwap::prepare_victim";
        // A victim the arena serves fills its slot, never the page cache:
        // the tier's own read, on its fill reader behind the prompt batches
        // — never the pool's, whose dispatch mutex the chain's compute
        // takes — and published before this returns, inside the deadline.
        if let Some(tier) = self.tier_serving(layer, id) {
            return tier.ensure_victim(layer, &[id]);
        }
        let stacks = self.layer(layer, WHAT)?.tensors.len();
        for stack in 0..stacks {
            for (file, at) in self.runs(layer, id, stack, WHAT)? {
                if at.is_empty() {
                    continue;
                }
                self.set
                    .populate_run(self.pair.source(), &file, &at)
                    .map_err(|e| GpuError::plan(WHAT, e))?;
            }
        }
        Ok(())
    }

    fn host_resident(&self, layer: usize, id: u32) -> Result<bool, GpuError> {
        const WHAT: &str = "FileSwap::host_resident";
        // The arena's books are the answer for an id it serves — never
        // mincore, never the page cache's residency.
        if let Some(tier) = self.tier_serving(layer, id) {
            return Ok(tier.slot_filled(layer, id));
        }
        let stacks = self.layer(layer, WHAT)?.tensors.len();
        for stack in 0..stacks {
            for (file, at) in self.runs(layer, id, stack, WHAT)? {
                if at.is_empty() {
                    continue;
                }
                let serves = self
                    .set
                    .serves(self.pair.source(), &file, &at)
                    .map_err(|e| GpuError::plan(WHAT, e))?;
                if !serves {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// An id the arena serves: the tier pages it.
    fn paged(&self, layer: usize, id: u32) -> bool {
        self.tier_serving(layer, id).is_some()
    }

    /// An id the arena serves goes to the card holding no slot: its slot is
    /// freed ([`NvTier::release`]) and the bytes it took returned. An id the
    /// mapping serves frees nothing: the churn pool the load's host set holds
    /// for the model's life is the host's, and the host never reads the rest
    /// of a card expert's bytes (the source's gate and up under the
    /// sidecar), which the load already released.
    fn release_host(&self, layer: usize, id: u32) -> Result<u64, GpuError> {
        match self.tier_serving(layer, id) {
            Some(tier) => tier.release(layer, id),
            None => Ok(0),
        }
    }
}

// ----------------------------------------------------------------- glue

/// The residency side of a body's [`HostServed`](crate::model::HostServed):
/// the machine's source and shape, the boundaries' log, and the delegation to
/// the host tier. A body holds one and points its trait's residency calls at
/// it; a load with no machine holds [`ResidencyGlue::off`].
pub struct ResidencyGlue {
    /// The boundaries' reports since the last [`ResidencyGlue::take_passes`],
    /// kept only once a binary that prints them asked
    /// ([`ResidencyGlue::log_passes`]).
    log: Option<Vec<(PassKind, PassReport)>>,
    /// The machine's source, when the load runs one: what a gate asks of it
    /// directly.
    source: Option<Arc<FileSwap>>,
    /// The machine's shape, until [`ResidencyGlue::start`] spends it.
    cfg: Option<MachineCfg>,
}

impl ResidencyGlue {
    /// No machine: the load's slot map for the model's life, every call
    /// nothing.
    #[must_use]
    pub fn off() -> ResidencyGlue {
        ResidencyGlue {
            log: None,
            source: None,
            cfg: None,
        }
    }

    /// The machine a load under [`Residency`] prepared: its file `source` and
    /// the shape [`ResidencyGlue::start`] starts it with.
    #[must_use]
    pub fn new(source: FileSwap, cfg: MachineCfg) -> ResidencyGlue {
        ResidencyGlue {
            log: None,
            source: Some(Arc::new(source)),
            cfg: Some(cfg),
        }
    }

    /// Keep every boundary's report from now on, for
    /// [`ResidencyGlue::take_passes`]: a binary that prints the `residency
    /// pass` records asks once, with the boundaries it takes at most between
    /// two takes, `passes`, so a pass logs its report without growing the
    /// log. Nothing is kept until then.
    pub fn log_passes(&mut self, passes: usize) {
        self.log.get_or_insert_with(Vec::new).reserve(passes);
    }

    /// The boundaries' reports since the last take, in order, each with the
    /// kind of the pass it ended; the log keeps its capacity.
    pub fn take_passes(&mut self) -> Vec<(PassKind, PassReport)> {
        self.log
            .as_mut()
            .map(|log| std::mem::replace(log, Vec::with_capacity(log.capacity())))
            .unwrap_or_default()
    }

    /// The machine's source, when the load runs one.
    #[must_use]
    pub fn source(&self) -> Option<&FileSwap> {
        self.source.as_deref()
    }

    /// Give the machine's source the NVMe expert tier's arena, when the
    /// load runs a machine and built a tier ([`FileSwap::attach_tier`]);
    /// nothing otherwise. Load-time only, before [`ResidencyGlue::start`].
    pub fn attach_tier(&self, tier: Option<&Arc<NvTier>>) -> Result<(), GpuError> {
        match (&self.source, tier) {
            (Some(source), Some(tier)) => source.attach_tier(tier),
            _ => Ok(()),
        }
    }

    /// Run the machine over `tier`'s slot map
    /// ([`HostTier::start_swap`]), `view` the stage card's copy of the map:
    /// called once the body's pieces are sized, because the machine empties
    /// each layer's spare slots, so every piece that sizes itself from the
    /// map's capacity is made first. Load-time only; nothing without a
    /// machine.
    pub fn start<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        view: Arc<DeviceTensor<u32>>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "ResidencyGlue::start";
        let Some(cfg) = self.cfg.take() else {
            return match self.source.is_some() {
                true => Err(GpuError::state(
                    WHAT,
                    "the residency machine was started already: load-time, once",
                )),
                false => Ok(()),
            };
        };
        let source = self
            .source
            .clone()
            .ok_or_else(|| GpuError::state(WHAT, "the machine's source"))?;
        tier.start_swap(ctx, stream, view, source, cfg)
    }

    /// The tier's residency boundary at `at` ([`HostTier::swap_at`]), the
    /// report of a boundary it made logged when a binary asked
    /// ([`ResidencyGlue::log_passes`]). Nothing without a machine.
    pub fn at_boundary<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        stream: &CudaStream,
        at: BoundaryAt,
    ) -> Result<(), GpuError> {
        if let Some(r) = tier.swap_at(stream, at)?
            && let Some(log) = self.log.as_mut()
        {
            log.push(r);
        }
        Ok(())
    }

    /// The pass the last boundary opened, a `kind`, keeps `kept` rows
    /// ([`HostTier::keep_rows`]). Nothing without a machine.
    pub fn keep_rows<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        kept: KeptRows,
        kind: PassKind,
    ) -> Result<(), GpuError> {
        tier.keep_rows(kept, kind)
    }

    /// The residency back to its seed at a quiet boundary
    /// ([`HostTier::swap_reset`]); `None` without a machine.
    pub fn reset<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        stream: &CudaStream,
    ) -> Result<Option<ResetReport>, GpuError> {
        tier.swap_reset(stream)
    }

    /// The residency to a new seed at a quiet boundary
    /// ([`HostTier::swap_reseed`]); `None` without a machine.
    pub fn reseed<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        stream: &CudaStream,
        order: &[&[u32]],
    ) -> Result<Option<ResetReport>, GpuError> {
        tier.swap_reseed(stream, order)
    }

    /// Stop the machine before anything of the model is freed
    /// ([`HostTier::stop_swap`]); nothing without one.
    pub fn stop<H>(&mut self, tier: &mut HostTier<H>) {
        tier.stop_swap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::placement::Role;

    /// The plan's tensor of one routed stack: `dims` in gguf `ne[]` order,
    /// `per` bytes an expert, its `file_bytes` `per * 8` — the eight experts
    /// every check here runs. No real tensor data, only what the parts
    /// check reads.
    fn stack(l: usize, name: &str, ty: GgmlType, dims: &[u64], per: u64) -> ModelTensor {
        ModelTensor {
            name: name.to_string(),
            shard: 0,
            layer: Some(l),
            role: Role::RoutedExperts,
            ty,
            dims: dims.to_vec(),
            file_bytes: per * 8,
            gathered_rows: None,
        }
    }

    /// A located planes stack's parts come from the arena's slot where the lane's open read finds
    /// its id filled, and from where the folder keeps the matrix otherwise
    /// ([`FileSwap::matrix`]): after the fill the folder's bytes change, and the filled id's codes
    /// and block scales are the bytes the fill wrote, the unfilled id's the folder's new ones.
    /// FAIL-first: the matrix read from the mapping whatever the arena holds (the filled id reads
    /// the folder's new bytes).
    #[test]
    fn a_filled_id_stages_its_planes_from_its_slot() {
        use super::super::nvtier::tests::{
            LOCATED_CODES, LOCATED_PER, Scratch, located_split, located_tier, routed,
        };
        use model::placement::workstation;
        use model::placement::{KvBytes, ModelTensors, PlanLevers, plan_host_routed};
        use std::os::unix::fs::FileExt;

        struct NoKv;
        impl KvBytes for NoKv {
            fn layer_bytes(&self, _: usize, _: u64) -> u64 {
                0
            }
        }
        let scratch = Scratch::new("swap-located");
        let split = located_split(&scratch);
        let tier = Arc::new(located_tier(&split));
        let t = routed(&split, "s.weight", None);
        // The host set holds no page: the arena serves every id of the stack.
        let model = ModelTensors {
            tensors: Vec::new(),
            layers: 1,
            experts: 3,
            experts_used: 1,
            bf16_words: |_| false,
        };
        let machine = workstation::plan_on(workstation::A6000, 1);
        let plan = plan_host_routed(&model, &machine, 4096, &NoKv, &PlanLevers::default())
            .expect("the plan");
        let pair = R8Pair::at_load(Arc::clone(&split), false).expect("the pair");
        let set = HostSet::of(pair.source(), &plan, |_| false).expect("an empty host set");
        let parts = vec![LOCATED_CODES, LOCATED_PER - LOCATED_CODES];
        let swap = FileSwap {
            tier: OnceLock::new(),
            pair,
            set,
            experts: 3,
            first: 0,
            layers: vec![Some(LayerParts {
                tensors: vec![t],
                base: vec![0; 2],
                slots: 2,
                parts,
                of: vec![(0, Plane::Codes), (0, Plane::BlockScales)],
                scales: vec![Vec::new()],
            })],
            union_tag: String::new(),
            convert: None,
        };
        assert!(swap.tier.set(Arc::clone(&tier)).is_ok());
        let matrix = |id: usize| {
            let m = split.matrix("s.weight", id).expect("the matrix");
            [m.codes.to_vec(), m.block_scales.to_vec()]
        };
        tier.ensure(0, &[1]).expect("the fill");
        let filled = matrix(1);
        for shard in 0..2 {
            let path = scratch.0.join(format!("folder-{shard}.bin"));
            let flipped: Vec<u8> = std::fs::read(&path).unwrap().iter().map(|b| !b).collect();
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.write_all_at(&flipped, 0).unwrap();
        }
        assert_ne!(
            matrix(1),
            filled,
            "the folder's bytes changed under the mapping"
        );
        for (id, want) in [(1, filled), (0, matrix(0))] {
            swap.open_read(0, id).expect("the open read");
            for (part, want) in want.iter().enumerate() {
                let got = swap.source(0, id, part).expect("the part");
                assert_eq!(got.bytes, &want[..], "expert {id} part {part}");
            }
            swap.release_read(0, id).expect("the read's end");
        }
    }

    /// A resident K-quant stack of `ty`: `words` words, `rows` rows.
    fn resident(ty: GgmlType, words: usize, rows: usize) -> Option<FoundStack> {
        Some(FoundStack {
            ty: Some(ty),
            words,
            rows,
            ptr: 0,
            nvfp4: None,
        })
    }

    /// A resident NVFP4 planes stack of `slots` experts of `rows` rows of
    /// `k` values: its code plane, block-scale plane and tensor scales whole.
    fn planes(slots: usize, rows: usize, k: usize) -> Option<FoundStack> {
        Some(FoundStack {
            ty: None,
            words: slots * rows * k / 8,
            rows: slots * rows,
            ptr: 0x1000,
            nvfp4: Some([(slots * rows * k / 64, 0x2000), (slots, 0x3000)]),
        })
    }

    /// A resident weight that is no K-quant stack at all.
    fn other_format() -> Option<FoundStack> {
        Some(FoundStack {
            ty: None,
            words: 0,
            rows: 0,
            ptr: 0,
            nvfp4: None,
        })
    }

    /// An NVFP4 stack held as card planes, NVFP4 rows or GGUF's NVFP4 blocks
    /// alike, is three parts — the codes (`rows · k / 2` bytes), the block
    /// scales (`rows · k / 16`) and the 4-byte tensor scale — in plane order,
    /// at the planes' own addresses; the slots are the down's rows over its
    /// rows an expert. Planes that are not the tensor's bytes, a scale plane
    /// short of the slots, and planes listed as another type are refused by
    /// name. Mutants: a planes stack read as one part of `file_bytes /
    /// experts` — the nine-part arm takes three; planes held only for NVFP4
    /// rows — the GGUF arm is refused.
    #[test]
    fn nvfp4_planes_stacks_hold_three_parts_a_matrix() {
        // MiniMax-M3.1's shape, small: gate and up 64 rows of 128 values, the
        // down 128 rows of 64; four slots of eight experts.
        let names = names(&["layer.3.gate", "layer.3.up", "layer.3.down"]);
        let types = [GgmlType::NVFP4_ROWS; 3];
        let per = |rows: u64, k: u64| rows * k * 9 / 16;
        let t = [
            stack(
                3,
                "layer.3.gate",
                GgmlType::NVFP4_ROWS,
                &[128, 64, 8],
                per(64, 128),
            ),
            stack(
                3,
                "layer.3.up",
                GgmlType::NVFP4_ROWS,
                &[128, 64, 8],
                per(64, 128),
            ),
            stack(
                3,
                "layer.3.down",
                GgmlType::NVFP4_ROWS,
                &[64, 128, 8],
                per(128, 64),
            ),
        ];
        let tensors: Vec<Option<&ModelTensor>> = t.iter().map(Some).collect();
        let found = [planes(4, 64, 128), planes(4, 64, 128), planes(4, 128, 64)];
        match layer_parts(3, &names, &types, &found, &tensors, 8) {
            Ok(Some((parts, slots))) => {
                let m = [64 * 64, 64 * 8, 4];
                assert_eq!(parts, [m, m, m].concat());
                assert_eq!(slots, 4);
            }
            other => panic!("NVFP4 planes stacks hold three parts each, got {other:?}"),
        }
        let blocks = t.clone().map(|mut t| {
            t.ty = GgmlType::NVFP4;
            t
        });
        let of_blocks: Vec<Option<&ModelTensor>> = blocks.iter().map(Some).collect();
        let parts = layer_parts(3, &names, &[GgmlType::NVFP4; 3], &found, &of_blocks, 8);
        assert_eq!(
            parts
                .expect("GGUF's NVFP4 blocks in planes")
                .map(|(p, _)| p[..3].to_vec()),
            Some(vec![64 * 64, 64 * 8, 4])
        );
        assert!(layer_parts(3, &names, &[GgmlType::Q4_K; 3], &found, &tensors, 8).is_err());
        let held = found[0].as_ref().expect("a planes stack").parts();
        assert_eq!(
            held,
            [
                (0x1000, Plane::Codes),
                (0x2000, Plane::BlockScales),
                (0x3000, Plane::Scale)
            ]
        );
        let off = stack(3, "layer.3.gate", GgmlType::NVFP4_ROWS, &[128, 64, 8], 999);
        let wrong = vec![Some(&off), tensors[1], tensors[2]];
        match layer_parts(3, &names, &types, &found, &wrong, 8) {
            Err(GpuError::Shape { detail, .. }) => {
                assert!(
                    detail.starts_with("stack layer.3.gate: NVFP4 planes"),
                    "{detail}"
                );
            }
            other => panic!("planes that are not the tensor's bytes are refused, got {other:?}"),
        }
        let short = [
            planes(4, 64, 128),
            planes(4, 64, 128),
            planes(3, 128, 64).map(|mut f| {
                f.rows = 4 * 128;
                f
            }),
        ];
        match layer_parts(3, &names, &types, &short, &tensors, 8) {
            Err(GpuError::Shape { detail, .. }) => {
                assert!(
                    detail.starts_with("stack layer.3.down: NVFP4 planes"),
                    "{detail}"
                );
            }
            other => panic!("planes short of the slots are refused, got {other:?}"),
        }
    }

    fn names(of: &[&str]) -> Vec<String> {
        of.iter().map(|s| s.to_string()).collect()
    }

    /// Two layers whose layouts differ — [Q4_K, Q4_K, Q5_K] against
    /// [Q5_K, Q5_K, Q6_K], each part its own bytes an expert and the down
    /// stack its own rows — hold their own parts and slots, each layer as
    /// the file lays it out. Mutant: parts taken from the first card layer
    /// for every layer — the layer 1 arm catches it.
    #[test]
    fn layers_of_different_layouts_hold_their_own_parts() {
        // Layer 0: 4096, 12288 and 8192 bytes an expert, the down stack 4096
        // rows of 512 an expert — 8 slots.
        let names0 = names(&["layer.0.gate", "layer.0.up", "layer.0.down"]);
        let types0 = [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q5_K];
        let t0 = [
            stack(0, "layer.0.gate", GgmlType::Q4_K, &[512, 128, 8], 4096),
            stack(0, "layer.0.up", GgmlType::Q4_K, &[1536, 128, 8], 12288),
            stack(0, "layer.0.down", GgmlType::Q5_K, &[2048, 512, 8], 8192),
        ];
        let tensors0: Vec<Option<&ModelTensor>> = t0.iter().map(Some).collect();
        let found0 = [
            resident(GgmlType::Q4_K, 8192, 1024),
            resident(GgmlType::Q4_K, 24576, 1024),
            resident(GgmlType::Q5_K, 16384, 4096),
        ];
        match layer_parts(0, &names0, &types0, &found0, &tensors0, 8) {
            Ok(Some((parts, slots))) => {
                assert_eq!(parts, vec![4096, 12288, 8192]);
                assert_eq!(slots, 8);
            }
            other => panic!("layer 0 holds its parts, got {other:?}"),
        }
        // Layer 1: 6144, 6144 and 16384 bytes an expert, the down stack 1536
        // rows of 256 an expert — 6 slots.
        let names1 = names(&["layer.1.gate", "layer.1.up", "layer.1.down"]);
        let types1 = [GgmlType::Q5_K, GgmlType::Q5_K, GgmlType::Q6_K];
        let t1 = [
            stack(1, "layer.1.gate", GgmlType::Q5_K, &[768, 96, 8], 6144),
            stack(1, "layer.1.up", GgmlType::Q5_K, &[768, 96, 8], 6144),
            stack(1, "layer.1.down", GgmlType::Q6_K, &[2048, 256, 8], 16384),
        ];
        let tensors1: Vec<Option<&ModelTensor>> = t1.iter().map(Some).collect();
        let found1 = [
            resident(GgmlType::Q5_K, 9216, 576),
            resident(GgmlType::Q5_K, 9216, 576),
            resident(GgmlType::Q6_K, 24576, 1536),
        ];
        match layer_parts(1, &names1, &types1, &found1, &tensors1, 8) {
            Ok(Some((parts, slots))) => {
                assert_eq!(parts, vec![6144, 6144, 16384]);
                assert_eq!(slots, 6);
            }
            other => panic!("layer 1 holds its own parts, not layer 0's, got {other:?}"),
        }
    }

    /// An MXFP4 stack is its file's 17-byte blocks, 32 values each, so a part
    /// is `file_bytes / experts` whatever the type: a gate or up expert is 2,048
    /// rows of 4,096 values (128 blocks of 17 bytes a row), the down 4,096 rows of
    /// 2,048 (64 blocks), the three parts the same 4,456,448 bytes, the rows a
    /// whole number of words, and a card of three experts holds three slots.
    #[test]
    fn mxfp4_stacks_hold_whole_experts_in_three_equal_parts() {
        const PER: u64 = 2048 * 128 * 17;
        let names = names(&["layer.1.gate", "layer.1.up", "layer.1.down"]);
        let types = [GgmlType::MXFP4; 3];
        let t = [
            stack(1, "layer.1.gate", GgmlType::MXFP4, &[4096, 2048, 8], PER),
            stack(1, "layer.1.up", GgmlType::MXFP4, &[4096, 2048, 8], PER),
            stack(1, "layer.1.down", GgmlType::MXFP4, &[2048, 4096, 8], PER),
        ];
        let tensors: Vec<Option<&ModelTensor>> = t.iter().map(Some).collect();
        // Three card experts: 3 · 2,048 rows of 544 words, and 3 · 4,096 rows of 272.
        let found = [
            resident(GgmlType::MXFP4, 3 * 2048 * 544, 3 * 2048),
            resident(GgmlType::MXFP4, 3 * 2048 * 544, 3 * 2048),
            resident(GgmlType::MXFP4, 3 * 4096 * 272, 3 * 4096),
        ];
        match layer_parts(1, &names, &types, &found, &tensors, 8) {
            Ok(Some((parts, slots))) => {
                assert_eq!(parts, vec![4_456_448; 3]);
                assert_eq!(slots, 3);
            }
            other => panic!("an MXFP4 layer holds three equal parts, got {other:?}"),
        }
    }

    /// A layer none of whose stacks the load holds is skipped whatever its
    /// types list says: the skip reads only the found stacks, before the
    /// names-and-types count it would otherwise refuse.
    #[test]
    fn a_layer_with_no_resident_stack_is_none_whatever_its_types() {
        let names = names(&["layer.4.gate", "layer.4.up", "layer.4.down"]);
        let types = [GgmlType::Q4_K];
        let found = [None, None, None];
        let tensors: Vec<Option<&ModelTensor>> = vec![None, None, None];
        match layer_parts(4, &names, &types, &found, &tensors, 8) {
            Ok(None) => {}
            other => panic!("a layer the load holds nothing of is skipped, got {other:?}"),
        }
    }

    /// A stack resident as another K-quant than the one its layer lists —
    /// and a resident weight that is no K-quant stack at all — is refused
    /// by name, beside its layer's other parts.
    #[test]
    fn a_stack_of_another_type_than_listed_is_refused_by_name() {
        let names = names(&["layer.2.gate", "layer.2.down"]);
        let types = [GgmlType::Q4_K, GgmlType::Q5_K];
        let gate = stack(2, "layer.2.gate", GgmlType::Q4_K, &[512, 128, 8], 4096);
        let down = stack(2, "layer.2.down", GgmlType::Q5_K, &[2048, 512, 8], 8192);
        let tensors = vec![Some(&gate), Some(&down)];
        let wrong_type = [
            resident(GgmlType::Q6_K, 8192, 1024),
            resident(GgmlType::Q5_K, 16384, 4096),
        ];
        match layer_parts(2, &names, &types, &wrong_type, &tensors, 8) {
            Err(GpuError::Tensor { what, name, need }) => {
                assert_eq!(
                    (what, need),
                    (
                        "FileSwap::new",
                        "a resident routed stack of its listed type (a K-quant, or NVFP4 as \
                         card planes) beside the layer's other parts"
                    )
                );
                assert_eq!(name, "layer.2.gate");
            }
            other => panic!("a stack of another type than listed is refused, got {other:?}"),
        }
        let no_kquant = [other_format(), resident(GgmlType::Q5_K, 16384, 4096)];
        match layer_parts(2, &names, &types, &no_kquant, &tensors, 8) {
            Err(GpuError::Tensor { name, .. }) => assert_eq!(name, "layer.2.gate"),
            other => panic!("a weight of another device format is refused, got {other:?}"),
        }
    }

    /// A layer whose stack names and types differ in count is refused
    /// before any stack is read.
    #[test]
    fn names_and_types_of_different_counts_are_refused() {
        let names = names(&["layer.1.gate", "layer.1.up", "layer.1.down"]);
        let types = [GgmlType::Q4_K, GgmlType::Q4_K];
        let gate = stack(1, "layer.1.gate", GgmlType::Q4_K, &[512, 128, 8], 4096);
        let tensors: Vec<Option<&ModelTensor>> = vec![Some(&gate), None, None];
        let found = [resident(GgmlType::Q4_K, 8192, 1024), None, None];
        match layer_parts(1, &names, &types, &found, &tensors, 8) {
            Err(GpuError::Shape { what, detail }) => {
                assert_eq!(what, "FileSwap::new");
                assert_eq!(detail, "layer 1: 3 stack names for 2 types");
            }
            other => panic!("names and types of different counts are refused, got {other:?}"),
        }
    }

    /// A part that is not a whole number of 4-byte words — zero bytes an
    /// expert included — or a stack too short for its slots is refused with
    /// its bytes and words. One stack, 8 slots of 512 rows.
    #[test]
    fn a_part_not_whole_words_or_a_stack_too_short_is_refused() {
        let names = names(&["layer.0.gate"]);
        let types = [GgmlType::Q4_K];
        let case = |per: u64, words: usize| {
            let t = stack(0, "layer.0.gate", GgmlType::Q4_K, &[512, 512, 8], per);
            let tensors = vec![Some(&t)];
            let found = [resident(GgmlType::Q4_K, words, 4096)];
            layer_parts(0, &names, &types, &found, &tensors, 8)
        };
        match case(6, 12) {
            Err(GpuError::Shape { what, detail }) => {
                assert_eq!(what, "FileSwap::new");
                assert_eq!(
                    detail,
                    "layer 0 part 0: 6 bytes an expert, a stack of 12 words for 8 slots"
                );
            }
            other => panic!("a part that is not whole words is refused, got {other:?}"),
        }
        match case(0, 100) {
            Err(GpuError::Shape { detail, .. }) => assert_eq!(
                detail,
                "layer 0 part 0: 0 bytes an expert, a stack of 100 words for 8 slots"
            ),
            other => panic!("a part of no bytes is refused, got {other:?}"),
        }
        match case(4096, 8191) {
            Err(GpuError::Shape { detail, .. }) => assert_eq!(
                detail,
                "layer 0 part 0: 4096 bytes an expert, a stack of 8191 words for 8 slots"
            ),
            other => panic!("a stack too short for its slots is refused, got {other:?}"),
        }
    }

    /// Layer parts of `parts` bytes: no tensors or stacks, only what the
    /// convert-layout check reads.
    fn held(parts: &[usize]) -> Option<LayerParts> {
        Some(LayerParts {
            tensors: Vec::new(),
            base: Vec::new(),
            slots: 8,
            parts: parts.to_vec(),
            of: Vec::new(),
            scales: Vec::new(),
        })
    }

    /// A load with a convert step takes card layers of one layout only, and
    /// names the first layer of another layout. Mutant: the check skipped (or
    /// comparing only neighbours past the first) — the layer 7 arm catches it.
    #[test]
    fn a_convert_step_refuses_a_layer_of_other_parts_by_name() {
        let same = [None, held(&[4096, 4096, 8192]), held(&[4096, 4096, 8192])];
        assert!(one_layout(4, &same).is_ok(), "one layout passes");
        assert!(one_layout(4, &[None, None]).is_ok(), "no held layer passes");
        let other = [
            held(&[4096, 4096, 8192]),
            None,
            held(&[4096, 4096, 8192]),
            held(&[6144, 6144, 16384]),
        ];
        match one_layout(4, &other) {
            Err(GpuError::Shape { what, detail }) => {
                assert_eq!(what, "FileSwap::new");
                assert_eq!(
                    detail,
                    "layer 7: parts [6144, 6144, 16384] beside layer 4's [4096, 4096, 8192], \
                     which the convert step unpacks"
                );
            }
            r => panic!("a layer of other parts is refused, got {r:?}"),
        }
    }

    /// A plan of `layers` layers whose layer `l` holds the stacks `of(l)`
    /// names — a tensor per `Some`, in stack order, no real tensor data.
    fn plan(layers: usize, of: impl Fn(usize) -> [Option<GgmlType>; 3]) -> ModelTensors {
        let mut tensors = Vec::new();
        for l in 0..layers {
            for (i, ty) in of(l).into_iter().enumerate() {
                if let Some(ty) = ty {
                    tensors.push(stack(
                        l,
                        &format!("layer.{l}.{}", ["gate", "up", "down"][i]),
                        ty,
                        &[8, 8, 8],
                        1,
                    ));
                }
            }
        }
        ModelTensors {
            tensors,
            layers,
            experts: 8,
            experts_used: 8,
            bf16_words: |_| false,
        }
    }

    /// Layer `l`'s three stack names as [`plan`] writes them.
    fn three_names(l: usize) -> [String; 3] {
        ["gate", "up", "down"].map(|part| format!("layer.{l}.{part}"))
    }

    /// A layer that holds some of its three stacks but not all is refused by
    /// the name of the first one missing, under the caller's own `what`, in a
    /// family whose layers may hold none. Mutant: the refusal's `need` reads
    /// the all-three family's string (the emptiness arms swapped) — the need
    /// assert catches it.
    #[test]
    fn a_partial_layer_is_refused_by_its_first_missing_stack() {
        let plan = plan(2, |l| {
            if l == 0 {
                [Some(GgmlType::Q4_K); 3]
            } else {
                [Some(GgmlType::Q4_K), Some(GgmlType::Q4_K), None]
            }
        });
        match PlanStacks::of(
            &plan,
            StackFacts {
                what: "of",
                names: three_names,
                check: None,
                empty: true,
                map_layers: None,
                unrouted: Vec::new(),
            },
        ) {
            Err(GpuError::Tensor { what, name, need }) => {
                assert_eq!(
                    (what, name.as_str(), need),
                    (
                        "of",
                        "layer.1.down",
                        "all three routed stacks: a layer holds either all of them or none"
                    )
                );
            }
            _ => panic!("a partial layer is refused"),
        }
    }

    /// A layer that holds none of its stacks enters the table empty in a
    /// family whose layers may hold none: its `types` list is empty and its
    /// `pair` absent, the layers beside it untouched. Mutant: the empty row
    /// stored as a stack of default types — the empty-layer arm catches it.
    #[test]
    fn a_layer_with_no_stacks_enters_the_table_empty() {
        let plan = plan(3, |l| match l {
            0 => [
                Some(GgmlType::Q4_K),
                Some(GgmlType::Q5_K),
                Some(GgmlType::Q6_K),
            ],
            1 => [None; 3],
            _ => [Some(GgmlType::Q5_K); 3],
        });
        let stacks = PlanStacks::of(
            &plan,
            StackFacts {
                what: "of",
                names: three_names,
                check: None,
                empty: true,
                map_layers: None,
                unrouted: Vec::new(),
            },
        )
        .expect("a family with empty layers reads its layers");
        assert_eq!(
            stacks.types(0),
            &[GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K]
        );
        assert_eq!(stacks.types(1), &[]);
        assert_eq!(stacks.types(2), &[GgmlType::Q5_K; 3]);
        assert_eq!(stacks.pair(0), Some((GgmlType::Q4_K, GgmlType::Q6_K)));
        assert_eq!(stacks.pair(1), None);
    }

    /// In a family every layer of which holds all three stacks, a layer that
    /// holds none — and one that holds some — is refused by the name of the
    /// first stack missing. Mutant: as the partial layer's.
    #[test]
    fn a_missing_stack_is_refused_where_every_layer_holds_all_three() {
        let cases = [
            ([None, None, None], "layer.0.gate"),
            ([Some(GgmlType::Q4_K), None, None], "layer.0.up"),
        ];
        for (held, name) in cases {
            let plan = plan(1, |_| held);
            match PlanStacks::of(
                &plan,
                StackFacts {
                    what: "of",
                    names: three_names,
                    check: None,
                    empty: false,
                    map_layers: None,
                    unrouted: Vec::new(),
                },
            ) {
                Err(GpuError::Tensor {
                    what,
                    name: n,
                    need,
                }) => {
                    assert_eq!((what, n.as_str()), ("of", name), "of {held:?}");
                    assert_eq!(need, "every layer's three routed stacks", "of {held:?}");
                }
                _ => panic!("a layer missing a stack is refused, of {held:?}"),
            }
        }
    }

    /// The family's check holds each complete layer's types and spares the
    /// empty rows: its refusal of layer 1 reaches the caller, and the same
    /// check passes a family whose layer 1 holds no stacks. Mutant: the
    /// check never called — the layer 1 arm catches it.
    #[test]
    fn the_family_check_holds_each_complete_layer() {
        fn not_q6(l: usize, [gate, _, _]: [GgmlType; 3]) -> Result<(), GpuError> {
            if l == 1 && gate == GgmlType::Q6_K {
                return Err(GpuError::shape("the check", format!("layer {l}")));
            }
            Ok(())
        }
        let q6 = plan(2, |l| {
            if l == 1 {
                [Some(GgmlType::Q6_K); 3]
            } else {
                [Some(GgmlType::Q4_K); 3]
            }
        });
        match PlanStacks::of(
            &q6,
            StackFacts {
                what: "of",
                names: three_names,
                check: Some(not_q6),
                empty: true,
                map_layers: None,
                unrouted: Vec::new(),
            },
        ) {
            Err(GpuError::Shape { what, detail }) => {
                assert_eq!((what, detail.as_str()), ("the check", "layer 1"));
            }
            _ => panic!("the family check's refusal reaches the caller"),
        }
        let empty = plan(2, |l| {
            if l == 1 {
                [None; 3]
            } else {
                [Some(GgmlType::Q4_K); 3]
            }
        });
        let stacks = PlanStacks::of(
            &empty,
            StackFacts {
                what: "of",
                names: three_names,
                check: Some(not_q6),
                empty: true,
                map_layers: None,
                unrouted: Vec::new(),
            },
        )
        .expect("the check spares an empty layer");
        assert_eq!(stacks.types(1), &[]);
    }

    /// An r8 sidecar beside the table's file is refused by name under the
    /// facts' `what`, the message listing the types the stacks are, whether
    /// or not one of them is Q3_K: the table has no unpack from the
    /// sidecar's layout. Without a sidecar the check passes. Mutants: the
    /// refusal dropped — the no-Q3_K arm catches it; made only where no stack
    /// is Q3_K — the Q3_K arm catches it.
    #[test]
    fn an_r8_sidecar_is_refused_whatever_the_stacks_are() {
        let table = |plan: &ModelTensors| {
            PlanStacks::of(
                plan,
                StackFacts {
                    what: "of",
                    names: three_names,
                    check: None,
                    empty: true,
                    map_layers: None,
                    unrouted: Vec::new(),
                },
            )
            .expect("a table of complete layers")
        };
        let (q3, q4, q5, q6) = (
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
        );
        let cases = [
            (
                plan(2, |l| {
                    if l == 0 {
                        [Some(q4), Some(q4), Some(q5)]
                    } else {
                        [Some(q4), Some(q4), Some(q6)]
                    }
                }),
                format!("{q4}, {q5}, {q6}"),
            ),
            (
                plan(1, |_| [Some(q3), Some(q3), Some(q4)]),
                format!("{q3}, {q4}"),
            ),
        ];
        for (plan, types) in cases {
            let stacks = table(&plan);
            assert!(
                stacks.check_sidecar(false).is_ok(),
                "no sidecar, of {types}"
            );
            match stacks.check_sidecar(true) {
                Err(GpuError::Shape { what, detail }) => {
                    assert_eq!(what, "of", "of {types}");
                    assert!(
                        detail.contains(&format!("routed stacks are {types}:")),
                        "{detail}, of {types}"
                    );
                }
                r => panic!("a sidecar is refused whatever the stacks, of {types}, got {r:?}"),
            }
        }
    }
}
