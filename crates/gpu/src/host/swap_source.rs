//! The model side of the residency machine as common code
//! ([`super::swap`]): where any placed load's routed experts come from
//! ([`FileSwap`], a [`SwapSource`] over the model file and the load's host
//! set), and the [`ResidencyGlue`] a body delegates its residency calls to.
//!
//! **Parts.** An expert is its model's parts in stack order — V4.1's gate,
//! up and down; a model names its own ([`FileStacks`]). On the card each
//! part of slot `s` is the file's bytes of that expert at byte `s · part` of
//! its layer's stack, which the placed load uploads as file bytes in slot
//! order. The parts are each layer's own as the file holds them: a model
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
//! ([`crate::model::GpuModel::load_placed_with`]).

use std::ops::Range;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use cuda_core::{CudaContext, CudaStream, sys};
use gguf::Split;
use gguf::quant::GgmlType;
use model::placement::host_lock::{HostFile, HostSet, expert_run};
use model::placement::{ModelTensor, ModelTensors, Plan};
use model::r8file::R8Pair;
use runtime::swaprule::KeptRows;

use super::HostExperts;
use super::nvtier::NvTier;
use super::swap::{
    BoundaryAt, MachineCfg, PassReport, Piece, ResetReport, Residency, SwapSource, Transform,
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
    /// The r8 sidecar's refusal ([`FileStacks::open`]), in the family's
    /// words.
    pub sidecar: fn() -> GpuError,
}

/// The [`FileStacks`] of a model whose card holds the file's bytes: each
/// layer's gate, up and down types read from the plan's tensors, nothing to
/// convert, every family fact an input ([`StackFacts`]).
pub struct PlanStacks {
    names_of: fn(usize) -> [String; 3],
    /// Per layer, its three stacks' types in stack order; `None` on a layer
    /// that holds none of them.
    types: Vec<Option<[GgmlType; 3]>>,
    map_layers: Option<usize>,
    unrouted: Vec<usize>,
    sidecar: fn() -> GpuError,
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
            names_of: facts.names,
            types,
            map_layers: facts.map_layers,
            unrouted: facts.unrouted,
            sidecar: facts.sidecar,
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
        if sidecar {
            return Err((self.sidecar)());
        }
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

/// One layer's parts: the file tensors, the stage stacks' base addresses and
/// slots, and each part's bytes — this layer's own, as the file holds its
/// stacks.
struct LayerParts {
    tensors: Vec<ModelTensor>,
    base: Vec<sys::CUdeviceptr>,
    slots: usize,
    parts: Vec<usize>,
}

/// What the per-layer parts check reads of one stack name's resident weight
/// ([`Weights::get`]): `None` when the load holds no weight of the name,
/// else the resident stack's K-quant type, its words and rows, and its
/// buffer's device address. `ty` is `None` — and the words and rows stand
/// unread — for a weight of another device format, which the check refuses
/// beside its layer's other parts.
struct FoundStack {
    ty: Option<GgmlType>,
    words: usize,
    rows: usize,
    ptr: sys::CUdeviceptr,
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
            },
            _ => FoundStack {
                ty: None,
                words: 0,
                rows: 0,
                ptr: 0,
            },
        }
    }
}

/// Layer `l`'s parts and slots from what the load holds of its stacks:
/// `names` its stack names in stack order, `types` the K-quants its kernels
/// read them in, `found` and `tensors` what the load's weights and the plan
/// hold of each name, and `experts` the routed experts a stack holds. Each
/// part is its tensor's `file_bytes / experts`; the slots are the last
/// stack's rows over its tensor's `dims[1]`. `Ok(None)`: the load holds
/// none of the layer's stacks. Refused by name, every error naming
/// `FileSwap::new`, the entry point the check serves: a names and types
/// count that differs, a stack not a resident K-quant of its listed type, a
/// name the plan does not hold, a part not a whole number of 4-byte words,
/// and a stack too short for its slots. Pure host code: the tests below
/// exercise layers whose layouts differ without a model file.
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
            Some(st) if st.ty == Some(*want) => Ok(st),
            _ => Err(GpuError::tensor(
                WHAT,
                n.clone(),
                "a resident K-quant routed stack beside the layer's other parts",
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
    let per: Vec<usize> = tensors
        .iter()
        .map(|t| usize::try_from(t.file_bytes / experts.max(1)).unwrap_or(0))
        .collect();
    let slots =
        stacks_of[stacks_of.len() - 1].rows / tensors[tensors.len() - 1].dims[1].max(1) as usize;
    for (i, st) in stacks_of.iter().enumerate() {
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
    }
    Ok(Some((per, slots)))
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
    /// The model's convert step, when its staged bytes are not its slots'.
    convert: Option<Arc<dyn Convert>>,
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
            // The check held every name: a resident K-quant stack of its
            // listed type and a tensor of the plan.
            out.push(Some(LayerParts {
                tensors: tensors
                    .into_iter()
                    .map(|t| t.cloned().expect("a plan tensor the check held"))
                    .collect(),
                base: found
                    .iter()
                    .map(|st| st.as_ref().expect("a resident stack the check held").ptr)
                    .collect(),
                slots,
                parts,
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
    /// to a card slot: the source file's bytes, whatever the host reads.
    pub fn card_bytes(&self, layer: usize, id: u32, part: usize) -> Result<&[u8], GpuError> {
        const WHAT: &str = "FileSwap::card_bytes";
        let p = self.layer(layer, WHAT)?;
        let per = *p
            .parts
            .get(part)
            .ok_or_else(|| GpuError::shape(WHAT, format!("part {part} of an expert")))?;
        let t = &p.tensors[part];
        let split = self.pair.source().split();
        let (s, info) = split.find(&t.name).ok_or(GpuError::tensor(
            WHAT,
            t.name.clone(),
            "a routed stack of the split",
        ))?;
        let whole = split
            .shard(s)
            .ok_or_else(|| GpuError::shape(WHAT, format!("shard {s} of the split")))?
            .data(info)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let at = id as usize * per;
        whole.get(at..at + per).ok_or_else(|| {
            GpuError::shape(WHAT, format!("layer {layer} expert {id}: past {}", t.name))
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

    /// Where the host reads part `part` of layer `layer`'s expert `id`: the
    /// sidecar's run when it holds the stack and the pair reads one, else the
    /// source's.
    fn run(
        &self,
        layer: usize,
        id: u32,
        part: usize,
        what: &'static str,
    ) -> Result<(HostFile, Range<u64>), GpuError> {
        let t = &self.layer(layer, what)?.tensors[part];
        expert_run(self.pair.source(), t, self.experts, id, true)
            .map_err(|e| GpuError::plan(what, e))
    }
}

impl SwapSource for FileSwap {
    fn part_bytes(&self, layer: usize) -> &[usize] {
        layer
            .checked_sub(self.first)
            .and_then(|i| self.layers.get(i))
            .and_then(Option::as_ref)
            .map_or(&[], |p| p.parts.as_slice())
    }

    fn source(&self, layer: usize, id: u32, part: usize) -> Result<Piece<'_>, GpuError> {
        const WHAT: &str = "FileSwap::source";
        let p = self.layer(layer, WHAT)?;
        let per = *p
            .parts
            .get(part)
            .ok_or_else(|| GpuError::shape(WHAT, format!("part {part} of an expert")))?;
        let t = &p.tensors[part];
        let src = self.pair.source();
        let whole = match src
            .sidecar()
            .filter(|side| side.file_bytes(&t.name).is_some())
        {
            Some(side) => side.data(&t.name).map_err(|e| GpuError::plan(WHAT, e))?,
            None => {
                let split = src.split();
                let (s, info) = split.find(&t.name).ok_or(GpuError::tensor(
                    WHAT,
                    t.name.clone(),
                    "a routed stack of the split",
                ))?;
                split
                    .shard(s)
                    .ok_or_else(|| GpuError::shape(WHAT, format!("shard {s} of the split")))?
                    .data(info)
                    .map_err(|e| GpuError::plan(WHAT, e))?
            }
        };
        let at = id as usize * per;
        let bytes = whole.get(at..at + per).ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!(
                    "layer {layer} expert {id}: past the {} bytes of {}",
                    whole.len(),
                    t.name
                ),
            )
        })?;
        Ok(Piece {
            bytes,
            transform: Transform::Identity,
        })
    }

    /// An id the arena serves is open on the tier while a lane reads it
    /// ([`NvTier::open_read`]): no union's drop of its layer takes the pages
    /// the lane has yet to copy out. An id the mapping serves needs nothing.
    fn open_read(&self, layer: usize, id: u32) -> Result<(), GpuError> {
        match self.tier_serving(layer, id) {
            Some(tier) => tier.open_read(layer, id),
            None => Ok(()),
        }
    }

    /// An id the arena serves leaves the mapping and the page cache once a
    /// lane is done with it ([`NvTier::end_read`]): its pages were the
    /// read's alone, and the next read faults them in again. An id the
    /// mapping serves keeps its pages: the host set's.
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
        // the tier's own read, on this thread, inside the deadline.
        if let Some(tier) = self.tier_serving(layer, id) {
            return tier.ensure(layer, &[id]);
        }
        let parts = self.layer(layer, WHAT)?.parts.len();
        for part in 0..parts {
            let (file, at) = self.run(layer, id, part, WHAT)?;
            self.set
                .populate_run(self.pair.source(), &file, &at)
                .map_err(|e| GpuError::plan(WHAT, e))?;
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
        let parts = self.layer(layer, WHAT)?.parts.len();
        for part in 0..parts {
            let (file, at) = self.run(layer, id, part, WHAT)?;
            let serves = self
                .set
                .serves(self.pair.source(), &file, &at)
                .map_err(|e| GpuError::plan(WHAT, e))?;
            if !serves {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Nothing: every stage card expert the machine can move is the churn
    /// pool's, which the load's host set holds for the model's life, and the
    /// host never reads the rest of a card expert's bytes (the source's gate
    /// and up under the sidecar), which the load already released. The reset
    /// lets go of no byte the host set held, so it reports 0.
    fn release_host(&self, _layer: usize, _id: u32) -> Result<u64, GpuError> {
        Ok(0)
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

    /// A resident K-quant stack of `ty`: `words` words, `rows` rows.
    fn resident(ty: GgmlType, words: usize, rows: usize) -> Option<FoundStack> {
        Some(FoundStack {
            ty: Some(ty),
            words,
            rows,
            ptr: 0,
        })
    }

    /// A resident weight that is no K-quant stack at all.
    fn other_format() -> Option<FoundStack> {
        Some(FoundStack {
            ty: None,
            words: 0,
            rows: 0,
            ptr: 0,
        })
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
                        "a resident K-quant routed stack beside the layer's other parts"
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
        }
    }

    /// Layer `l`'s three stack names as [`plan`] writes them.
    fn three_names(l: usize) -> [String; 3] {
        ["gate", "up", "down"].map(|part| format!("layer.{l}.{part}"))
    }

    /// The sidecar refusal of the tests' families, its text never read.
    fn no_sidecar() -> GpuError {
        GpuError::State {
            what: "the tests' stacks",
            missing: "no sidecar",
        }
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
                sidecar: no_sidecar,
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
                sidecar: no_sidecar,
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
                    sidecar: no_sidecar,
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
                sidecar: no_sidecar,
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
                sidecar: no_sidecar,
            },
        )
        .expect("the check spares an empty layer");
        assert_eq!(stacks.types(1), &[]);
    }
}
