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
//! read in and locked, the page cache what a step would fault on. A victim
//! outside the set is not host-resident whatever the page cache holds, so
//! the machine refuses its flip by name. The load puts each layer's churn
//! pool — its stage card experts past the pinned ones ([`ChurnPool`]) — in
//! the set ([`crate::model::GpuModel::load_placed_with`]).

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use cuda_core::{CudaContext, CudaStream, sys};
use gguf::Split;
use gguf::quant::GgmlType;
use model::placement::host_lock::{HostFile, HostSet, expert_run};
use model::placement::{ModelTensor, Plan};
use model::r8file::R8Pair;

use super::HostExperts;
use super::swap::{MachineCfg, PassReport, Piece, ResetReport, Residency, SwapSource, Transform};
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

/// A placed load's [`SwapSource`] over its model file: each part of an expert
/// as the file (or the r8 sidecar the host reads) holds it, into the slot's
/// place in the layer's stage stack, with the load's host set saying what the
/// host can serve.
pub struct FileSwap {
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
// SAFETY: as for `Send`: no call mutates shared state.
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
        let find = |name: String| {
            plan.model
                .tensors
                .iter()
                .find(|t| t.name == name)
                .cloned()
                .ok_or(GpuError::tensor(WHAT, name, "a routed stack of the plan"))
        };
        let mut out = Vec::with_capacity(layers.len());
        for l in layers.clone() {
            let names = stacks.names(l);
            // The card's stacks first: all of the layer's or none. A layer
            // none of whose stacks the load holds (a dense block, a layer the
            // plan keeps whole on the host) is skipped whatever its types
            // list says.
            let found = names.iter().map(|n| w.get(n)).collect::<Vec<_>>();
            if found.iter().all(Option::is_none) {
                out.push(None);
                continue;
            }
            let types = stacks.types(l);
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
                .zip(&names)
                .zip(types)
                .map(|((d, n), want)| match d {
                    Some(DevWeight::KQuant { ty, w, .. }) if ty == want => Ok(w),
                    _ => Err(GpuError::tensor(
                        WHAT,
                        n.clone(),
                        "a resident K-quant routed stack beside the layer's other parts",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let tensors = names
                .iter()
                .cloned()
                .map(find)
                .collect::<Result<Vec<_>, _>>()?;
            let per: Vec<usize> = tensors
                .iter()
                .map(|t| usize::try_from(t.file_bytes / experts.max(1)).unwrap_or(0))
                .collect();
            let slots = stacks_of[stacks_of.len() - 1].rows()
                / tensors[tensors.len() - 1].dims[1].max(1) as usize;
            for (i, st) in stacks_of.iter().enumerate() {
                if per[i] == 0 || !per[i].is_multiple_of(4) || st.buf().len() * 4 < slots * per[i] {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "layer {l} part {i}: {} bytes an expert, a stack of {} words for \
                             {slots} slots",
                            per[i],
                            st.buf().len()
                        ),
                    ));
                }
            }
            out.push(Some(LayerParts {
                tensors,
                base: stacks_of.iter().map(|st| st.buf().cu_deviceptr()).collect(),
                slots,
                parts: per,
            }));
        }
        let first = out
            .iter()
            .flatten()
            .next()
            .ok_or_else(|| GpuError::shape(WHAT, "no layer of the card holds routed stacks"))?;
        let t0 = first.tensors[0].clone();
        let convert = stacks.open(&t0.dims, &first.parts, pair.r8().sidecar().is_some(), ctx)?;
        Ok(FileSwap {
            pair,
            set,
            experts,
            first: layers.start,
            layers: out,
            convert,
        })
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
        let t = &p.tensors[part.min(p.tensors.len() - 1)];
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

    /// The tier's residency boundary ([`HostTier::swap_boundary`]), its
    /// report logged when a binary asked
    /// ([`ResidencyGlue::log_passes`]). Nothing without a machine.
    pub fn at_boundary<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if let Some(r) = tier.swap_boundary(stream)?
            && let Some(log) = self.log.as_mut()
        {
            log.push(r);
        }
        Ok(())
    }

    /// The pass the last boundary opened, a `kind`, keeps its first `kept`
    /// rows ([`HostTier::keep_rows`]). Nothing without a machine.
    pub fn keep_rows<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        kept: usize,
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
