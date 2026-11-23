//! A placed load's host tier and the stage card's side of it: what every body whose routed
//! experts the stage card holds a share of keeps the same way beside its layers (Kimi K3,
//! Hy4-flash). The plan's card share of each routed layer's stacks is resident (the id prefix
//! `[0, n_card)`); the host tier serves the rest. [`PlacedSide`] is which layers the host serves a
//! share of, each routed layer's card count and slot-map row, the slot map's card copy, and the
//! step's handoff and go, and a prompt batch's counterparts of them (the places of its rows and
//! their download to the host's batch set); [`PlacedHost`] holds it with the tier's shell. A
//! body's own walk (the three parts of its block around the host's go or around a batch's
//! download and join) stays the body's.
//!
//! Under a residency machine the slot map's card copy is the machine's view of the map: its
//! flips write their words there on the engine stream at a boundary, so the step's handoff and a
//! batch's places, which read the copy at their launch after the boundary, follow the live map
//! (the stacks are rooms the common load made, [`crate::StackRoom`]). The card count of a layer
//! is its stacks' slots, which the machine never grows, and every routed layer has a host share
//! the machine moves experts through ([`PlacedHost::new`]).

use crate::DeviceBufferRaw;
use std::sync::Arc;

use cuda_core::{CudaStream, DeviceBuffer};

use runtime::sched::At;

use super::BatchLeg;
use super::handoff::{Handoff, HandoffKernels, HandoffSlots, TierSink};
use super::run::HostRun;
use super::served::{ResidencyParts, TierBody};
use super::shell::TierShell;
use crate::card_leg::{LegRun, LegStacks};
use crate::card_slots::Places;
use crate::hybrid::{Chain, Hybrid, SlotMap};
use crate::kquant::Act;
use crate::room::kv::FreedBytes;
use crate::tensor::{Window, WindowMut};
use crate::{DeviceTensor, Gpu, GpuError};

/// The routed run a [`PlacedSide`] is made for: experts a routed layer, slots a token, and the
/// width of the row the experts read (the latent's, on a latent block).
#[derive(Clone, Copy, Debug)]
pub struct SideShape {
    pub n_expert: usize,
    pub n_used: usize,
    pub width: usize,
}

/// The stage card's side of a placed load: which layers the host serves a share of, each routed
/// layer's card count and slot-map row, the slot map's card copy, the handoff launches, the
/// places of one layer's slots, and the tier sink of a slot count no per-count handoff entry is
/// built for ([`PlacedSide::sink_any_count`]). The step's half is [`PlacedSide::go`]; a prompt
/// batch's is [`PlacedSide::places_rows`] and [`PlacedSide::download_rows`], over the same map,
/// row offsets and shape.
pub struct PlacedSide {
    hosted: Vec<bool>,
    n_card: Vec<usize>,
    row_off: Vec<usize>,
    shape: SideShape,
    /// The slot map's card copy, one row of `n_expert` places a routed layer: the residency
    /// machine's view, whose words its flips write ([`PlacedHost`]).
    slots: Arc<DeviceTensor<u32>>,
    handoff: HandoffKernels,
    /// The routed slots' places in the card's stacks, or `HOST`.
    sel: DeviceBuffer<u32>,
    sink: Option<TierSink>,
}

impl PlacedSide {
    /// The side of a model of `n` layers over the slot map `map` of the routed run at `shape`:
    /// layer `l` is hosted when its card count is under its experts. `check(l, on_card)` is the
    /// body's refusal of a layer whose card stacks are not its card count's, called for each
    /// routed layer with experts on the card. A routed layer with no map row is refused by
    /// name (`what`). Load-time only.
    pub fn new(
        gpu: &Gpu,
        map: &SlotMap,
        (n, shape): (usize, SideShape),
        what: &'static str,
        mut check: impl FnMut(usize, usize) -> Result<(), GpuError>,
    ) -> Result<PlacedSide, GpuError> {
        let (mut hosted, mut n_card, mut row_off) = (vec![false; n], vec![0; n], vec![0; n]);
        let run = map.layers();
        for l in run.clone() {
            let on_card = map.on_card(l)?;
            n_card[l] = on_card;
            row_off[l] = map.row_offset(l).ok_or_else(|| {
                GpuError::shape(what, format!("layer {l}: a routed layer with no map row"))
            })?;
            hosted[l] = on_card < shape.n_expert;
            if on_card > 0 {
                check(l, on_card)?;
            }
        }
        let stream = gpu.stream();
        let slots = Arc::new(DeviceTensor::upload(
            stream,
            &map.stage_view(),
            run.len(),
            shape.n_expert,
        )?);
        Ok(PlacedSide {
            hosted,
            n_card,
            row_off,
            shape,
            slots,
            handoff: HandoffKernels::load(gpu.context())?,
            sel: DeviceBuffer::zeroed(stream, shape.n_used)?,
            sink: None,
        })
    }

    /// The tier sink of a host handoff at a slot count no per-count entry is built for, so
    /// [`PlacedSide::go`] hands a layer no tier card holds experts of over through the any-count
    /// tier entry; a count a per-count entry is built for takes none. Without it such a count is
    /// refused by name at the go. Load-time only.
    pub fn sink_any_count(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        let SideShape { n_used, width, .. } = self.shape;
        if HandoffSlots::of(n_used).is_err() {
            self.sink = Some(TierSink::new(stream, n_used, width)?);
        }
        Ok(())
    }

    /// The layers the host serves a share of, in order.
    #[must_use]
    pub fn hosted_set(&self) -> Vec<usize> {
        (0..self.hosted.len()).filter(|&l| self.hosted[l]).collect()
    }

    /// Layer `l` of the model: a layer past it panics by name (`what`), as no accessor answers for
    /// a layer the model does not have.
    #[must_use]
    pub fn layer(&self, what: &str, l: usize) -> usize {
        assert!(
            l < self.hosted.len(),
            "{what}: layer {l} is past the model's {} layers",
            self.hosted.len()
        );
        l
    }

    /// Whether the host serves a share of layer `l`. A layer past the model panics by name.
    #[must_use]
    pub fn hosted(&self, l: usize) -> bool {
        self.hosted[self.layer("hosted", l)]
    }

    /// The first layer the host serves a share of, if any.
    #[must_use]
    pub fn first_hosted(&self) -> Option<usize> {
        self.hosted.iter().position(|&h| h)
    }

    /// The routed layers whose experts the card holds whole: the step runs them with no host
    /// leg, in order.
    #[must_use]
    pub fn whole_set(&self) -> Vec<usize> {
        (0..self.hosted.len())
            .filter(|&l| self.n_card[l] > 0 && !self.hosted[l])
            .collect()
    }

    /// The card's experts of layer `l`: 0 on a layer with no routed block. A layer past the
    /// model panics by name.
    #[must_use]
    pub fn n_card(&self, l: usize) -> usize {
        self.n_card[self.layer("n_card", l)]
    }

    /// Layer `l`'s row in the slot map's card copy. A layer past the model panics by name.
    #[must_use]
    pub fn row_off(&self, l: usize) -> usize {
        self.row_off[self.layer("row_off", l)]
    }

    /// The slot map's card copy.
    #[must_use]
    pub fn slots(&self) -> &Arc<DeviceTensor<u32>> {
        &self.slots
    }

    /// The places of the last handoff's slots in the card's stacks, or `HOST`.
    #[must_use]
    pub fn sel(&self) -> &DeviceBuffer<u32> {
        &self.sel
    }

    /// Layer `l`'s handoff of the router's `ids` and `weights` over the slot map's card copy.
    #[must_use]
    pub fn handoff<'a>(
        &'a self,
        l: usize,
        (ids, weights): (&'a DeviceBuffer<u32>, &'a DeviceBuffer<f32>),
    ) -> Handoff<'a> {
        Handoff {
            ids,
            weights,
            map: self.slots.buf(),
            row_off: self.row_off(l),
            n_expert: self.shape.n_expert,
        }
    }

    /// `f` over layer `l`'s handoff of `picks` (the router's ids and weights), the handoff
    /// launches and the places they write: a handoff of the body's own (a tier layer's, which
    /// writes the tier's image beside the host's).
    pub fn with_handoff<T>(
        &mut self,
        l: usize,
        picks: (&DeviceBuffer<u32>, &DeviceBuffer<f32>),
        f: impl FnOnce(&HandoffKernels, &Handoff<'_>, &mut DeviceBuffer<u32>) -> T,
    ) -> T {
        let row_off = self.row_off(l);
        let h = Handoff {
            ids: picks.0,
            weights: picks.1,
            map: self.slots.buf(),
            row_off,
            n_expert: self.shape.n_expert,
        };
        f(&self.handoff, &h, &mut self.sel)
    }

    /// Layer `l`'s step handoff of the router's `ids` and `weights` into the host tier's row 0,
    /// each slot's place into [`PlacedSide::sel`], then the go that releases the host leg: the
    /// per-count entry of the slot count, or the any-count tier entry into the sink
    /// ([`HandoffKernels::enqueue_handoff_sunk`]). One launch and the go.
    pub fn go(
        &mut self,
        gpu: &Gpu,
        picks: (&DeviceBuffer<u32>, &DeviceBuffer<f32>),
        hy: &mut Hybrid<HostRun>,
        l: usize,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let fault = gpu.layer_sink(l)?;
        let row_off = self.row_off(l);
        let PlacedSide {
            shape,
            slots,
            handoff,
            sel,
            sink,
            ..
        } = self;
        let h = Handoff {
            ids: picks.0,
            weights: picks.1,
            map: slots.buf(),
            row_off,
            n_expert: shape.n_expert,
        };
        let target = hy.boundary_mut().handoff_target_of(0)?;
        match sink {
            Some(sink) => handoff.enqueue_handoff_sunk(stream, &h, target, sink, fault, sel)?,
            None => handoff.enqueue_handoff(stream, &h, target, fault, sel)?,
        }
        hy.boundary().enqueue_go_of(stream, l, 0)
    }

    /// Layer `l`'s places of a batch's `m` rows of the router's `ids` (`n_used` words a row,
    /// row-major) over the slot map's card copy, into `sel`: each slot's place in the card's
    /// stacks, or `HOST` for an expert the card does not hold. The batch's counterpart of the
    /// places the step's handoff writes ([`PlacedSide::go`]). One launch.
    pub fn places_rows(
        &self,
        gpu: &Gpu,
        l: usize,
        rows: (&DeviceBuffer<u32>, usize),
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.places_rows_in(gpu, l, self.slots.buf(), rows, sel)
    }

    /// [`PlacedSide::places_rows`] over `map`, another view of the experts' places laid out as the
    /// slot map's card copy (one row of `n_expert` places a routed layer, at the same row
    /// offsets): the expert tier's places on a load that hangs one.
    pub fn places_rows_in(
        &self,
        gpu: &Gpu,
        l: usize,
        map: &DeviceBuffer<u32>,
        (ids, m): (&DeviceBuffer<u32>, usize),
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.handoff.places().enqueue_places(
            gpu.stream(),
            &Places {
                ids,
                n: m * self.shape.n_used,
                map,
                row_off: self.row_off(l),
                n_expert: self.shape.n_expert,
            },
            gpu.layer_sink(l)?,
            sel,
        )
    }

    /// Layer `l`'s row of the slot map's card copy into `map`, in stream order: each expert's
    /// place in the card's stacks, or `HOST` for one the card does not hold, the map a route table
    /// over the layer's ids is built through (`Gemm32Kernels::enqueue_route_remap`: the host's
    /// slots get no tile). One device-to-device copy; a `map` of other than `n_expert` words is
    /// refused by name.
    pub fn remap_row(
        &self,
        stream: &CudaStream,
        l: usize,
        map: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let n = self.shape.n_expert;
        if map.len() != n {
            return Err(GpuError::shape(
                "PlacedSide::remap_row",
                format!("a map of {} words for the layer's {n} experts", map.len()),
            ));
        }
        let row = Window::<u32>::of(self.slots.buf(), self.row_off(l) * size_of::<u32>(), n)?;
        let mut to = WindowMut::<u32>::of_mut(map, 0, n)?;
        to.copy_from_device_raw(&row, stream)?;
        Ok(())
    }

    /// The download of hosted layer `at.layer`'s batch rows to the host tier's next batch set: the
    /// latent rows `act` the host reads, the router's `weights` and `ids`, and `tsels`, the places
    /// of each expert tier that holds experts of the layer ([`PlacedSide::places_rows_in`]), none
    /// when no tier does. The caller's contract: a body's walk calls it only from the front of a
    /// layer its host leg covers ([`PlacedSide::hosted`]); any other layer is refused by name, not
    /// downloaded.
    pub fn download_rows(
        &self,
        port: &mut BatchLeg<'_, HostRun>,
        at: At,
        (act, weights, ids): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<u32>),
        tsels: &[&DeviceBuffer<u32>],
    ) -> Result<(), GpuError> {
        if !self.hosted(at.layer) {
            return Err(GpuError::shape(
                "PlacedSide::download_rows",
                format!(
                    "layer {}: a download of a layer the host serves no share of",
                    at.layer
                ),
            ));
        }
        let stream = port.stream();
        let key = port.key(at);
        let hy = port.hybrid();
        if tsels.is_empty() {
            hy.enqueue_download(stream, [act, weights], ids, key)
        } else {
            hy.enqueue_download_tiered(stream, [act, weights], ids, tsels, key)
        }
    }

    /// Layer `l`'s card slots over `normed`, one row, under `rule` at the places the last
    /// handoff wrote; `None` on a layer with no experts on the card.
    #[must_use]
    pub fn card_run<'a>(
        &'a self,
        l: usize,
        stacks: &'a LegStacks<'a>,
        normed: &'a DeviceBuffer<f32>,
        rule: Act,
    ) -> Option<LegRun<'a>> {
        let n_card = self.n_card(l);
        (n_card > 0).then_some(LegRun {
            layer: l,
            stacks,
            normed,
            m: 1,
            sel: &self.sel,
            n_card,
            rule,
        })
    }

    /// Device bytes the side holds beside the plan's terms: the slot map's card copy, the
    /// places and the sink.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.slots.buf().num_bytes()
            + self.sel.num_bytes()
            + self.sink.as_ref().map_or(0, TierSink::bytes)
    }
}

/// A placed load's host tier, its glue and the stage card's side. The shell is the first field
/// and the drop stops the residency machine before any card or pinned memory frees. The machine,
/// when the glue holds one, runs over the side's slot map copy ([`TierBody::residency_parts`]),
/// started once the body's pieces are sized ([`crate::model::HostServed::start_residency`]).
pub struct PlacedHost {
    pub shell: TierShell<HostRun>,
    pub side: PlacedSide,
    /// The engine stream, which the machine's give-backs run on.
    stream: Arc<CudaStream>,
}

impl Drop for PlacedHost {
    fn drop(&mut self) {
        self.shell.hybrid.stop_swap();
    }
}

impl PlacedHost {
    /// The placed host of `shell` and `side` on `gpu`'s engine stream. Refused by name: a
    /// residency machine (the glue holds one) beside a routed layer whose experts the card holds
    /// whole. The machine empties a spare slot of every routed layer and moves experts to the
    /// host, which serves no share of a whole layer, and the step runs that layer with no host
    /// leg. Load-time only.
    pub fn new(
        gpu: &Gpu,
        shell: TierShell<HostRun>,
        side: PlacedSide,
    ) -> Result<PlacedHost, GpuError> {
        let whole = side.whole_set();
        if shell.glue.source().is_some() && !whole.is_empty() {
            return Err(GpuError::shape(
                shell.what,
                format!(
                    "a residency machine beside routed layers whose experts the card holds whole                      ({whole:?}): the machine moves their slots to a host that serves no share of                      them (BLOOMERY_RESIDENCY off)"
                ),
            ));
        }
        Ok(PlacedHost {
            shell,
            side,
            stream: gpu.stream_handle(),
        })
    }

    /// At least `want` device bytes of the stage card's expert stacks given back by the residency
    /// machine ([`ResidencyGlue::free_bytes`]): the per-layer report folded to the stores' tally.
    /// Refused by name, the machine unchanged, on a load that runs none and as the machine
    /// refuses.
    ///
    /// [`ResidencyGlue::free_bytes`]: super::swap_source::ResidencyGlue::free_bytes
    pub fn free_expert_bytes(&mut self, want: u64) -> Result<FreedBytes, GpuError> {
        let TierShell { glue, hybrid, .. } = &mut self.shell;
        let freed = glue.free_bytes(hybrid, &self.stream, want)?;
        Ok(FreedBytes {
            bytes: freed.bytes,
            slots: freed.k,
            layers: freed.layers.len(),
        })
    }

    /// The host tier's union slabs for the one-column step ([`HostRun::prepare_union`] at one
    /// column): the resident path of the host layer serves the step only through them.
    /// Load-time only.
    pub fn prepare_step_union(&mut self) -> Result<(), GpuError> {
        self.shell.hybrid.host_mut().prepare_union(1)
    }

    /// The host tier's batch sets and union slabs for prompt passes of up to `rows` rows
    /// ([`crate::host::HostTier::prepare_batch`], [`HostRun::prepare_union`]): a load that runs
    /// its prompt as passes makes them once, and they also serve its one-column step. Load-time
    /// only.
    pub fn prepare_pass_rows(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError> {
        self.prepare_pass_walk(gpu, (rows, 1))
    }

    /// [`PlacedHost::prepare_pass_rows`] for walks of `units` units of up to `rows` rows each
    /// whose fronts all go ahead of the serves ([`crate::host::xstream::GROUP_WALK_ORDER`]): one
    /// exchange set a unit, two at least ([`crate::host::HostTier::prepare_batch_sets`]).
    /// Load-time only.
    pub fn prepare_pass_walk(
        &mut self,
        gpu: &Gpu,
        (rows, units): (usize, usize),
    ) -> Result<(), GpuError> {
        let sets = units.max(crate::host::batch::BATCH_SETS);
        self.shell
            .hybrid
            .prepare_batch_sets(gpu.context(), rows, sets)?;
        self.shell.hybrid.host_mut().prepare_union(rows)?;
        gpu.stream().synchronize()?;
        Ok(())
    }

    /// Device bytes the placed side holds beside the plan's terms: the side's and the boundary.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.side.device_bytes() + self.shell.hybrid.boundary().device_bytes()
    }

    /// The host tier, for a gate's instrument.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.shell.hybrid
    }

    /// The host tier's settling ([`crate::host::HostTier::settle`]).
    pub fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.shell.hybrid.settle(stream)
    }
}

impl TierBody for PlacedHost {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.shell.hybrid
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        &mut self.shell.hybrid
    }

    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.shell.residency_parts(&self.side.slots)
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.shell.hybrid.serve_captured_of(chain)
    }
}
