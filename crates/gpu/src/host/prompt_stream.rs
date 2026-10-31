//! The prompt stream: the one owner of a streaming prompt call's host
//! streaming, which any MoE family's body calls with its counts and its card
//! map and no stream code of its own.
//!
//! Under host streaming (the [`XMode`] lever, a residency machine, a prompt
//! call fed by the walk's units) every walk of the call moves each card
//! layer's residency pool toward the experts the unit routes most: in the
//! shadow, the host waits for the front's download, counts the unit's routed
//! ids and the machine's pick sends the pool's coldest residents to the host
//! and copies the hottest host experts over them; the card route and the
//! serve then run under the moved map. Under [`XMode::Split`] the expert
//! stream ([`super::xstream`]) sends the host experts its rule picks to a
//! half of its ring for the layer's units, and the serve's union leaves the
//! streamed experts out. The body hands the model's own facts in as data:
//! the least count its pick admits with no probe ([`PromptStream::begin`]'s
//! `floor`), the stream's priced card column (`tail`), its
//! layers, and — walk by walk — whether a layer holds card experts and the
//! walk's last position.
//!
//! A call that returns to its start (a decode history before it) returns
//! each layer once its host service has run, asked at the next layer's
//! shadow, whose download wait the call's queued return copies fill
//! ([`PromptStream::return_at_shadow`]); a split pick at the last walk's
//! last card layer prices an admit's way back, which no front follows
//! ([`PromptStream::return_cost`]), and the call's end issues what is left.
//! The stream's placement never stays.
//!
//! A walk of several units that share one pick a layer runs the same steps
//! in every family: the first unit picks at its shadow and the later units'
//! card routes read the pick it holds for them
//! ([`PromptStream::unit_pick`], [`PromptStream::unit_read`]), the last
//! unit's back owes the layer's return ([`PromptStream::unit_back`]), and
//! the walk's end returns the last layer and refuses a pick still held
//! ([`PromptStream::end_walk`]). The guards every family's `set_xstream`
//! shares are [`PromptStream::check_mode`]'s, and the stream's start at a
//! first `split` is [`PromptStream::start_split`].

use cuda_core::CudaStream;
use runtime::sched::At;
use runtime::xsplit::ReturnCost;

use super::BatchLeg;
use super::HostExperts;
use super::HostTier;
use super::swap::{CallCfg, CallPick, CallReport};
use super::xstream::{Costs, LandBatch, PROBE, RingLayer, XCfg, XLayer, XMode, XReport};
use crate::{Gpu, GpuError};

/// What the stream's errors name.
const WHAT: &str = "prompt stream";

/// A prompt call's host streaming (the [`XMode`] lever, the residency
/// machine's call mode and the expert stream): what the next prompt calls
/// move, whether one is streaming now, the pick's count buffer, and the
/// call's pick, stream and end records for the binary that prints them.
#[derive(Default)]
pub struct PromptStream {
    /// What the next prompt calls move (the body's `set_xstream`).
    mode: Option<XMode>,
    /// A call is streaming: every walk of it picks.
    on: bool,
    /// The streaming call runs the expert stream too (`split`).
    split: bool,
    /// The card tail rate a split pick's walk prices its unit with [µs per
    /// card pick], set when a streaming call opens: the stream's priced card
    /// column, 0 before the stream started.
    card_tail_us_per_pick: f64,
    /// The least count the pick admits on a load whose lane never probed,
    /// set when a streaming call opens: the body's own floor.
    floor: u32,
    /// The map's layers, set when a streaming call opens: the last card
    /// layer's return is priced against the ones after it.
    layers: usize,
    /// The walk the call runs, from 0, the pick records' group.
    ubatch: usize,
    /// The streaming call returns to its start
    /// ([`PromptStream::returns_here`]): decided at its begin, where the
    /// decode history is read.
    returning: bool,
    /// The streaming call's end position: the walk of the unit that ends
    /// there is the call's last.
    call_end: usize,
    /// The layer of the last unit whose host service ran and whose return
    /// the walk asks for at the next layer's shadow.
    owed: Option<usize>,
    /// The call's return runs whole at its end instead of in the walk: the
    /// twin a gate holds the walk's return to
    /// ([`PromptStream::set_return_at_end`]).
    return_at_end: bool,
    counts: Vec<u32>,
    picks: Vec<(usize, CallPick)>,
    end: Option<CallReport>,
    /// Per streamed layer of the call, its walk and what it streamed; the
    /// stream's end.
    xlayers: Vec<(usize, XLayer)>,
    xend: Option<XReport>,
    /// The walk's first unit's pick of a layer, held for its later units'
    /// card routes ([`PromptStream::unit_pick`]).
    held: Option<Held>,
}

/// The first unit's pick of a layer, for the walk's later units' card
/// routes: the layer, its landing batches, and whether the layer streams.
struct Held {
    layer: usize,
    land: Vec<LandBatch>,
    streams: bool,
}

impl PromptStream {
    /// What the next prompt calls move; [`XMode::Off`] until the body's
    /// `set_xstream` sets it.
    #[must_use]
    pub fn mode(&self) -> XMode {
        self.mode.unwrap_or(XMode::Off)
    }

    /// The body's `set_xstream` ends with this: the mode the next prompt
    /// calls move.
    pub fn set_mode(&mut self, mode: XMode) {
        self.mode = Some(mode);
    }

    /// The guards every family's `set_xstream` runs before it sets `mode`
    /// ([`PromptStream::set_mode`]), `tier_refused` whether the load hangs an
    /// expert tier whose layers the family's card route runs no ring beside
    /// (a route that sums a tier layer's card slots only in the tier's join).
    /// Refused by name: any change while a call streams, `Admit` or `Split`
    /// on a tier with no residency machine, and `Split` where `tier_refused`.
    pub fn check_mode<H: HostExperts>(
        &self,
        tier: &HostTier<H>,
        (mode, tier_refused): (XMode, bool),
    ) -> Result<(), GpuError> {
        if self.on {
            return Err(GpuError::state(
                "PromptStream::check_mode",
                "no prompt call streaming",
            ));
        }
        if mode != XMode::Off && tier.swap().is_none() {
            return Err(GpuError::state(
                "BLOOMERY_XSTREAM",
                "a residency machine (BLOOMERY_RESIDENCY=mid-p<P>-s<S>) for admit or split",
            ));
        }
        if mode == XMode::Split && tier_refused {
            return Err(GpuError::shape(
                "BLOOMERY_XSTREAM",
                "split on a load with an expert tier: this family's ring serves the stage card \
                 of a load with none",
            ));
        }
        Ok(())
    }

    /// The expert stream on `gpu` at a family's first `Split`
    /// ([`HostTier::xstream_start`], its probe [`PROBE`] host experts, `cfg`
    /// the model's expert shape): the ring's slots
    /// a half, which the family's card route arms its ring pass for; `None`
    /// when the stream runs already — it stays for the load's life, idle
    /// under `Admit` and `Off`. The stream's own refusals at its start (the
    /// card's room among them).
    pub fn start_split<H: HostExperts>(
        tier: &mut HostTier<H>,
        gpu: &Gpu,
        cfg: impl FnOnce() -> XCfg,
    ) -> Result<Option<usize>, GpuError> {
        if tier.xstream().is_some() {
            return Ok(None);
        }
        let x = tier.xstream_start(gpu.context(), gpu.stream(), cfg(), PROBE)?;
        Ok(Some(x.half_slots()))
    }

    /// A prompt call is streaming: no lever change and no cost change while
    /// one is.
    #[must_use]
    pub fn streaming(&self) -> bool {
        self.on
    }

    /// The call's return runs whole at its end instead of in the walk
    /// ([`PromptStream::return_at_shadow`]). Refused by name while a call
    /// streams.
    pub fn set_return_at_end(&mut self, on: bool) -> Result<(), GpuError> {
        if self.on {
            return Err(GpuError::state(
                "PromptStream::set_return_at_end",
                "no prompt call streaming",
            ));
        }
        self.return_at_end = on;
        Ok(())
    }

    /// The expert stream's rule costs in place of the ones the load started
    /// it with ([`super::xstream::XStream::set_costs`]). Refused by name
    /// before `split` started the stream and while a call streams.
    pub fn set_costs<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        costs: Costs,
    ) -> Result<(), GpuError> {
        const WHAT_C: &str = "PromptStream::set_costs";
        if self.on {
            return Err(GpuError::state(WHAT_C, "no prompt call streaming"));
        }
        tier.xstream_mut()
            .ok_or(GpuError::state(
                WHAT_C,
                "an expert stream (the load's split setting starts it)",
            ))?
            .set_costs(costs)
    }

    /// The load line's `xstream=` word: the mode, and under a started stream
    /// its ring's slots a half, its pinned staging slots (0: pageable), the
    /// lane's rate its probe measured and the rate the rule prices in GB/s,
    /// the least unit width that streams, the costs the rule prices its
    /// first streaming layer at in µs (the host union's a column and its
    /// floor, the card's fixed and a column) with where the lane's and the
    /// union's values came from (`stored`, `measured`, or `set` by a gate).
    #[must_use]
    pub fn word<H: HostExperts>(&self, tier: &HostTier<H>) -> String {
        let mode = self.mode().word();
        match tier.xstream() {
            Some(x) => {
                let costs = x.priced().map_or("none".to_string(), |(c, src)| {
                    format!(
                        "{:.3},{:.3},{:.3},{:.4} xstream_lane_source={src} \
                         xstream_union_source={}",
                        c.host_us_per_col,
                        c.host_us_fixed,
                        c.card_us_fixed,
                        c.card_us_per_col,
                        tier.host_costs().map_or("none", |h| h.source.word())
                    )
                });
                format!(
                    "{mode} xstream_half={} xstream_staging={} xstream_lane_gbs={:.2} \
                     xstream_priced_gbs={:.2} xstream_m_min={} xstream_costs={costs}",
                    x.half_slots(),
                    x.staging_slots(),
                    x.probed_b_per_us() / 1000.0,
                    x.lane_b_per_us() / 1000.0,
                    x.m_min().map_or("none".to_string(), |m| m.to_string())
                )
            }
            None => mode.to_string(),
        }
    }

    /// Open the prompt call's streaming when the lever is `admit` or
    /// `split`, the call is the walk's (`walk`: the units' path) and its `n`
    /// ids reach the least gate a pick admits from (`floor` under `admit`,
    /// the stream rule's walk gate under `split`): the residency machine's
    /// call ([`HostTier::call_begin`]) inside the pass the boundary just
    /// opened, and under `split` the expert stream's
    /// ([`HostTier::xstream_begin`]). Nothing otherwise: a row routes an
    /// expert once at most, so a call of fewer ids than the gate — and every
    /// call the walks do not feed — gives no expert a count its pick would
    /// admit, and its walks would only wait on each layer's download for
    /// nothing. A call of `n` ids from `end - n` that `returns` to its start
    /// (a decode history) returns each layer in the walk of its last unit,
    /// the one ending at `end`, unless the gate's twin is set
    /// ([`PromptStream::set_return_at_end`]). `layers` is the map's. Refused
    /// by name: a call streaming already, and a mode with no machine.
    #[allow(
        clippy::too_many_arguments,
        reason = "the tier and its stream, the call's own facts (its path, ids, end and return), and the body's data (its floor, its tail rate and its layers) (rust-quality R8)"
    )]
    pub fn begin<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        stream: &CudaStream,
        walk: bool,
        n: usize,
        end: usize,
        returns: bool,
        floor: u32,
        tail: f64,
        layers: usize,
    ) -> Result<(), GpuError> {
        const WHAT_B: &str = "prompt streaming";
        if self.on {
            return Err(GpuError::state(WHAT_B, "no call streaming (end)"));
        }
        self.picks.clear();
        self.end = None;
        self.xlayers.clear();
        self.xend = None;
        self.ubatch = 0;
        self.owed = None;
        self.held = None;
        self.returning = returns && !self.return_at_end;
        self.call_end = end;
        let mode = self.mode();
        self.card_tail_us_per_pick = tail;
        self.floor = floor;
        self.layers = layers;
        let floor = match (mode, tier.xstream()) {
            (XMode::Split, Some(x)) => (0..layers)
                .map(|l| {
                    x.constants(l)
                        .map_or(floor, |k| runtime::xsplit::walk_gate(&k))
                })
                .chain([floor])
                .min()
                .unwrap_or(floor),
            _ => floor,
        };
        if mode == XMode::Off || !walk || n < floor as usize {
            return Ok(());
        }
        let cfg = CallCfg { floor };
        if !tier.call_begin(stream, cfg)? {
            return Err(GpuError::state(
                "BLOOMERY_XSTREAM",
                "a residency machine (BLOOMERY_RESIDENCY=mid-p<P>-s<S>)",
            ));
        }
        self.on = true;
        self.split = mode == XMode::Split && tier.xstream_begin();
        if mode == XMode::Split && !self.split {
            self.on = false;
            let _ = tier.call_end(stream, false);
            return Err(GpuError::state(
                "BLOOMERY_XSTREAM=split",
                "an expert stream (the load's split setting starts it)",
            ));
        }
        Ok(())
    }

    /// End the prompt call's streaming, if it streams: the expert stream's
    /// end under `split` ([`HostTier::xstream_end`]), then the residency
    /// machine's ([`HostTier::call_end`]) whatever the first returned, its
    /// placement `kept` for the decode after it, else each layer back at the
    /// set the call started with; the ends' reports kept for
    /// [`PromptStream::take_pick_records`] and
    /// [`PromptStream::take_xstream_records`]. The first error is returned.
    pub fn end<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        stream: &CudaStream,
        kept: bool,
    ) -> Result<(), GpuError> {
        if !std::mem::take(&mut self.on) {
            return Ok(());
        }
        let x = match std::mem::take(&mut self.split) {
            true => tier.xstream_end(),
            false => Ok(None),
        };
        let ended = tier.call_end(stream, kept);
        self.xend = x?;
        self.end = ended?;
        // The stamps are final now: the end waited every layer's landed
        // event, and every copy landed behind its staged ticket.
        tier.call_staged_us(&mut self.picks);
        Ok(())
    }

    /// Under host streaming, on a layer with card experts (`card`), the
    /// layer's pick before its card route, for the first unit of `m` rows of
    /// a walk of `units` units: the host waits for the front's download of
    /// the unit's routed ids, counts them, and the residency machine's pick
    /// sends the pool's coldest residents to the host and copies the unit's
    /// hottest host experts over them — the least count admitted
    /// [`PromptStream::begin`]'s `floor` on a load whose lane never probed,
    /// the walk's own floor and backlog bound
    /// ([`HostTier::call_pick_walk_scaled`], every copy at the
    /// beside-union rate) where it did — in victim-slot order, a landing
    /// batch a third of them, each batch's event on the copy stream after
    /// its last copy ([`LandBatch`]) — the host map and the card's copy of
    /// the layer's words moved at once, so the union the walk's serve runs
    /// next, and the card route, run under the moved map, each GEMM batch
    /// waiting its own batch's event and no later copy. Under `split` the
    /// expert stream then takes the layer ([`HostTier::xstream_layer`]): the
    /// host experts its rule sends to the card stream into a half of its
    /// ring, its copies batched the same way, and the serve leaves them out;
    /// the layer's ring view for its card route is returned. Nothing else,
    /// and nothing in a walk narrower than the gate the pick admits from.
    /// `walk_end` is the walk's last position (`pos0 + m`).
    ///
    /// One pick a layer for the walk's units: the first unit's routed ids
    /// count `units` times each (the group's width, `units · m`), so the
    /// pick, its floor and the stream's rule price the group; under `split`
    /// every later unit joins the layer's stream
    /// ([`HostTier::xstream_share`]) before any card route reads it, so its
    /// serve leaves out the same experts and the stream stays open for its
    /// card route. A later unit's card route reads the same landing batches
    /// and ring (the walk's own held set).
    pub fn pick<H: HostExperts>(
        &mut self,
        port: &mut BatchLeg<'_, H>,
        at: At,
        (m, units): (usize, usize),
        walk_end: usize,
        card: impl Fn(usize) -> bool,
    ) -> Result<(Option<RingLayer>, Vec<LandBatch>), GpuError> {
        let l = at.layer;
        if !self.on || !card(l) {
            return Ok((None, Vec::new()));
        }
        let scale = u32::try_from(units)
            .map_err(|_| GpuError::shape(WHAT, format!("a walk of {units} units")))?;
        let group = units * m;
        let split = self.split;
        // The walk only where the lane probed: its gate is the least width
        // whose picks could pay a first admit, derived without the tail (the
        // counts it prices from do not exist yet). A load with no probe
        // admits from the stream floor, so short prompts move what they did.
        let k = port.hybrid().xstream().and_then(|x| x.constants(l));
        let walked = split && k.is_some();
        let gate = match k.as_ref() {
            Some(k) if split => runtime::xsplit::walk_gate(k),
            _ => self.floor,
        };
        // A walk of fewer rows than the gate gives no expert a count the
        // pick admits (a call's last walk can be one row).
        if group < gate as usize {
            return Ok((None, Vec::new()));
        }
        let key = port.key(at);
        let stream = port.stream();
        let ret = self.return_cost(l, walk_end, &card);
        let mut pick = match (split, k) {
            (true, Some(k)) => port.hybrid().call_pick_walk_scaled(
                stream,
                key,
                &mut self.counts,
                k,
                self.card_tail_us_per_pick,
                ret,
                scale,
            )?,
            _ => port.hybrid().call_pick_routed_scaled(
                stream,
                key,
                &mut self.counts,
                usize::MAX,
                scale,
            )?,
        };
        // The no-probe floor witness: the pick admitted from the stream
        // floor rather than the walk's measured plan.
        pick.fallback = u32::from(!walked);
        let admitted = pick.admitted;
        let land = std::mem::take(&mut pick.land);
        self.picks.push((self.ubatch, pick));
        if !split {
            return Ok((None, land));
        }
        let x = port
            .hybrid()
            .xstream_layer(stream, key, group, &self.counts, admitted)?;
        self.xlayers.push((self.ubatch, x));
        for unit in 1..units {
            let other = port.key(At { unit, layer: l });
            port.hybrid().xstream_share(l, other)?;
        }
        Ok((port.hybrid().xstream_ring(l), land))
    }

    /// Whether this walk is the last of a call that returns to its start: it
    /// returns each layer once the layer's host service has run.
    /// `walk_end` is the walk's last position.
    #[must_use]
    pub fn returns_here(&self, walk_end: usize) -> bool {
        self.on && self.returning && walk_end == self.call_end
    }

    /// In the returning call's last walk, at a layer's shadow — its front
    /// already enqueued, so the card never waits on this — the return of the
    /// layer whose host service ran last ([`HostTier::call_return`]), then
    /// the call's queued return copies pumped while the host waits for this
    /// layer's download ([`HostTier::call_return_pump`]): the copies fill
    /// the host's idle wait for the card's front, and the layer's pick
    /// issues its own behind a ring's worth of them at most. Nothing else.
    /// `walk_end` is the walk's last position.
    pub fn return_at_shadow<H: HostExperts>(
        &mut self,
        port: &mut BatchLeg<'_, H>,
        at: At,
        walk_end: usize,
    ) -> Result<(), GpuError> {
        if !self.returns_here(walk_end) {
            return Ok(());
        }
        let stream = port.stream();
        if let Some(l) = self.owed.take() {
            port.hybrid().call_return(l, stream)?;
        }
        let key = port.key(at);
        port.hybrid().call_return_pump(key)?;
        Ok(())
    }

    /// The return cost of an admit at layer `l`'s pick
    /// ([`ReturnCost`]): exposed at the returning call's last walk's last
    /// card layer, whose return no front follows, and hidden everywhere else
    /// — by the layer's place in the walk alone, never by how far the lane
    /// has got, so the pick is a function of the prompt.
    fn return_cost(&self, l: usize, walk_end: usize, card: &dyn Fn(usize) -> bool) -> ReturnCost {
        if self.returns_here(walk_end) && !(l + 1..self.layers).any(card) {
            ReturnCost::Exposed
        } else {
            ReturnCost::Hidden
        }
    }

    /// Under host streaming, on a layer with card experts (`card`), a unit's
    /// read of the layer after its card route. The walk's `last` unit's is
    /// the layer's last read of its slots in the call so far
    /// ([`HostTier::call_reader`]): the next walk's pick of the layer and
    /// the call's return copies of it wait behind that read — recorded at an
    /// earlier unit, they would land while a later unit's route reads the
    /// slots. Under `split` every unit reads the layer's stream
    /// ([`HostTier::xstream_read`]), which closes at the group's last, the
    /// half's next stream copying behind it. Nothing else.
    pub fn read<H: HostExperts>(
        &mut self,
        port: &mut BatchLeg<'_, H>,
        l: usize,
        last: bool,
        card: impl Fn(usize) -> bool,
    ) -> Result<(), GpuError> {
        if !self.on || !card(l) {
            return Ok(());
        }
        let stream = port.stream();
        if last {
            port.hybrid().call_reader(l, stream)?;
        }
        if self.split {
            port.hybrid().xstream_read(l, stream)?;
        }
        Ok(())
    }

    /// The last unit's host service of layer `l` has run: in the returning
    /// call's last walk the layer's return is due, asked at the next layer's
    /// shadow ([`PromptStream::return_at_shadow`]).
    pub fn owe_return(&mut self, l: usize) {
        self.owed = Some(l);
    }

    /// The return of the layer whose host service ran last, asked where no
    /// shadow follows it (the walk's last layer): the copies are the call
    /// end's.
    pub fn return_owed<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if let Some(l) = self.owed.take() {
            tier.call_return(l, stream)?;
        }
        Ok(())
    }

    /// Under host streaming, unit `at.unit` of a walk of `units` units at
    /// layer `at.layer`'s shadow, before its card route: the return the walk
    /// owes here ([`PromptStream::return_at_shadow`]), then the first unit's
    /// pick of the layer from its `m` rows ([`PromptStream::pick`]) or, for
    /// a later unit, the pick the first unit held for it
    /// ([`PromptStream::unit_read`]). The layer's ring view and landing
    /// batches, which the unit's card route reads, each GEMM batch behind
    /// its own batch's event. Refused by name: a later unit with no pick of
    /// the layer held. Nothing when no call streams.
    pub fn unit_pick<H: HostExperts>(
        &mut self,
        port: &mut BatchLeg<'_, H>,
        at: At,
        (m, units): (usize, usize),
        walk_end: usize,
        card: impl Fn(usize) -> bool,
    ) -> Result<(Option<RingLayer>, Vec<LandBatch>), GpuError> {
        if !self.on {
            return Ok((None, Vec::new()));
        }
        self.return_at_shadow(port, at, walk_end)?;
        if at.unit == 0 {
            return self.pick(port, at, (m, units), walk_end, card);
        }
        let l = at.layer;
        let held = self.held.take().filter(|h| h.layer == l).ok_or_else(|| {
            GpuError::protocol(
                WHAT,
                format!(
                    "unit {}'s card route of layer {l} with no pick of the layer's first unit \
                     held",
                    at.unit
                ),
            )
        })?;
        let ring = if held.streams {
            port.hybrid().xstream_ring(l)
        } else {
            None
        };
        Ok((ring, held.land))
    }

    /// Under host streaming, after unit `at.unit`'s card route of layer
    /// `at.layer` over what [`PromptStream::unit_pick`] gave it (`land`, and
    /// whether the layer streams): the pick held for the walk's next unit
    /// unless this is the last of `units`, then the unit's read
    /// ([`PromptStream::read`]), the last unit's the layer's last read of
    /// its slots. Nothing when no call streams.
    pub fn unit_read<H: HostExperts>(
        &mut self,
        port: &mut BatchLeg<'_, H>,
        (at, units): (At, usize),
        (land, streams): (Vec<LandBatch>, bool),
        card: impl Fn(usize) -> bool,
    ) -> Result<(), GpuError> {
        if !self.on {
            return Ok(());
        }
        let last = at.unit + 1 == units;
        if !last {
            self.held = Some(Held {
                layer: at.layer,
                land,
                streams,
            });
        }
        self.read(port, at.layer, last, card)
    }

    /// After unit `at.unit`'s back of layer `at.layer` in a walk of `units`
    /// units ending at `walk_end`: once the last unit's host service of a
    /// layer with card experts (`card`) has run, the returning call's last
    /// walk owes the layer's return ([`PromptStream::owe_return`]).
    pub fn unit_back(&mut self, (at, units): (At, usize), walk_end: usize, card: bool) {
        if at.unit + 1 == units && card && self.returns_here(walk_end) {
            self.owe_return(at.layer);
        }
    }

    /// After a walk: refused by name while a layer's pick is still held (a
    /// later unit's card route did not read it); then the return of the
    /// walk's last returned layer, which no shadow follows
    /// ([`PromptStream::return_owed`]).
    pub fn end_walk<H: HostExperts>(
        &mut self,
        tier: &mut HostTier<H>,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if let Some(h) = self.held.take() {
            return Err(GpuError::protocol(
                WHAT,
                format!(
                    "layer {}'s pick held past the walk: a unit's card route did not read it",
                    h.layer
                ),
            ));
        }
        self.return_owed(tier, stream)
    }

    /// A walk of the streaming call ran: its records name it by the next
    /// walk's index.
    pub fn walk_done(&mut self) {
        self.ubatch += 1;
    }

    /// The last streaming call's streamed layers, each with its walk, and
    /// the stream's end; empty and `None` after a call that did not stream.
    /// Taken: a second read is empty.
    pub fn take_xstream_records(&mut self) -> (Vec<(usize, XLayer)>, Option<XReport>) {
        (std::mem::take(&mut self.xlayers), self.xend.take())
    }

    /// The last streaming call's picks, each with its walk, and its end;
    /// empty and `None` after a call that did not stream. Taken: a second
    /// read is empty.
    pub fn take_pick_records(&mut self) -> (Vec<(usize, CallPick)>, Option<CallReport>) {
        (std::mem::take(&mut self.picks), self.end.take())
    }
}
