//! The routed block's card experts, common to every program whose card runs
//! the experts the slot map puts on it by the kernel table: the experts of a
//! layer computed by the common card leg ([`crate::card_leg`]: the per-slot
//! pair of the kernel table, launched by the entries the table names for a
//! stack's type) in the host leg's shadow, and summed here.
//!
//! One run is six launches, over `m` tokens of a layer's normed rows:
//! - the card leg's four (`card_leg`): the normed rows in q8_1, the gate·up
//!   `_sel` with the routed SwiGLU (slot `s` against column `s / n_used`), the
//!   q8_1 of the card slots' `h` columns (the host's slots are left alone) and
//!   the down `_sel`;
//! - the card slots' weighted sum, a token's slots in slot order (the leg's
//!   sum stage, `card_slots`);
//! - the sum plus the base the layer's back adds the host's sum to
//!   ([`Io::x1`]): the residual after the attention, with a shared expert's
//!   output summed into it by the caller where the family has one.
//!
//! The one-token step is the run at `m = 1`, its places written by the
//! handoff; a prompt batch is the run at `m = t`, one launch each over the
//! batch's `t · n_used` slots, its places written by one launch
//! ([`CardSlotKernels::enqueue_places`]). Each slot's rows are what the step's
//! launches write for it, whatever `m`, so the two feeds leave the same bits.
//!
//! What a family fixes is a [`CardFamily`]: the slot map's row width, the
//! slots a token picks, the stacks' names and what the errors name. A load
//! whose map names an expert tier card builds them with [`CardExperts::tiered`]:
//! a layer the tier holds experts of runs the card leg without its sum, the
//! stage card's slots' downs stay in the leg's rows ([`TierJoin`]) and the
//! layer's join sums them with the tier's rows in one slot order. [`CardExperts::new`]
//! refuses a map that names a tier by name: those experts hang none.

use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::models::{Ffn, LayerSpec};
use model::placement::accepts::{self, LegArms, Program};
use model::placement::kernels::{self, DownForms};

use crate::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum};
use crate::card_slots::{CardSlotKernels, Places};
use crate::hybrid::SlotMap;
use crate::kquant::Act;
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError};

/// The selected program, rather than a union of unrelated launches, admits stacks.
const STACK_ROLES: (&str, &str) = (
    "a resident routed gate or up stack the selected card gate·up arm runs",
    "a resident routed down stack the selected card down arm runs",
);

/// What a family fixes of its card experts.
#[derive(Clone, Copy)]
pub struct CardFamily {
    /// What the card experts' errors name.
    pub what: &'static str,
    /// The routed experts of a layer: the slot map's row width.
    pub n_expert: usize,
    /// The routed experts a token picks.
    pub n_used: usize,
    /// Layer `l`'s gate, up and down stacks' names.
    pub stacks: fn(usize) -> [String; 3],
    /// The program whose selected gate·up and down launches run the stacks.
    pub program: Program,
}

/// The kernels a run launches: the card leg's (its sum stage included), and
/// the places' of a prompt batch.
struct CardKernels {
    leg: CardLeg,
    places: CardSlotKernels,
    arms: LegArms,
}

impl CardFamily {
    /// The card leg's shape for runs of up to `cap` tokens of `n` values
    /// through experts of `ff` rows.
    fn shape(&self, n: usize, ff: usize, cap: usize) -> LegShape {
        LegShape {
            cap,
            slots_per_col: self.n_used,
            n,
            ff,
        }
    }
}

/// One run's buffers for up to `cap` tokens: the card leg's, the card sum and
/// the sum plus the base.
struct CardBufs {
    leg: LegBufs,
    acc: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
}

impl CardBufs {
    /// The buffers for runs of up to `cap` tokens of `n` values through
    /// experts of `ff` rows. Load-time or first-prompt only.
    fn new(
        stream: &CudaStream,
        family: &CardFamily,
        n: usize,
        ff: usize,
        cap: usize,
        downs: DownForms,
    ) -> Result<CardBufs, GpuError> {
        Ok(CardBufs {
            leg: LegBufs::for_downs(stream, family.shape(n, ff, cap), downs)?,
            acc: DeviceBuffer::zeroed(stream, cap * n)?,
            pre: DeviceBuffer::zeroed(stream, cap * n)?,
        })
    }

    /// Tokens the buffers hold.
    fn cap(&self) -> usize {
        self.leg.shape().cap
    }

    /// The buffers a tier layer's join reads and writes, the slots' places
    /// `sel` as the run read them.
    fn join<'a>(&'a mut self, sel: &'a DeviceBuffer<u32>) -> TierJoin<'a> {
        TierJoin {
            down: self.leg.down(),
            sel,
            acc: &mut self.acc,
            pre: &mut self.pre,
        }
    }

    /// Device bytes of the buffers.
    fn bytes(&self) -> usize {
        self.leg.bytes() + self.acc.num_bytes() + self.pre.num_bytes()
    }
}

/// A prompt batch's card buffers: one run's buffers for the batch's tokens
/// and the places its slots resolve to.
pub struct CardBatch {
    bufs: CardBufs,
    sel: DeviceBuffer<u32>,
}

impl CardBatch {
    /// Device bytes of the buffers: what [`CardExperts::batch`] allocates,
    /// counted before it does by [`CardExperts::batch_bytes`].
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bufs.bytes() + self.sel.num_bytes()
    }

    /// The sum plus the base of the last run over the batch, `t` rows.
    #[must_use]
    pub fn pre(&self) -> &DeviceBuffer<f32> {
        &self.bufs.pre
    }

    /// The buffers a tier layer's join reads and writes over the batch.
    pub fn join(&mut self) -> TierJoin<'_> {
        self.bufs.join(&self.sel)
    }
}

/// A tier layer's join over one run's buffers: the stage card's slots' down
/// rows (slot-major, `n` values a slot, left by the run that did not sum
/// them) and each slot's place in the card's stacks, the card sum the join
/// writes and the sum plus the base it then writes
/// ([`CardExperts::step_join`], [`CardBatch::join`]).
pub struct TierJoin<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub acc: &'a mut DeviceBuffer<f32>,
    pub pre: &'a mut DeviceBuffer<f32>,
}

/// What [`CardExperts::enqueue_step`], [`CardExperts::enqueue_pass`] and
/// [`CardExperts::enqueue_rows`] read: `m` tokens' normed rows, their slots'
/// routing weights, and the base the card sum is added to.
pub struct Io<'a> {
    pub normed: &'a DeviceBuffer<f32>,
    pub weights: &'a DeviceBuffer<f32>,
    /// The residual after the attention, with a shared expert's output
    /// summed into it by the caller where the family has one.
    pub x1: &'a DeviceBuffer<f32>,
    pub m: usize,
}

/// A layer's card experts: how many its stacks hold, and the stacks' names
/// (gate, up, down), made once at load.
struct CardLayer {
    n_card: usize,
    names: [String; 3],
}

impl CardLayer {
    /// The stacks' names, as the card leg reads them.
    fn names(&self) -> [&str; 3] {
        self.names.each_ref().map(String::as_str)
    }
}

/// Layer `l`'s card experts of `layers`, refused by name (`what`) on a layer
/// with none.
fn layer_of<'a>(
    what: &'static str,
    layers: &'a [Option<CardLayer>],
    l: usize,
) -> Result<&'a CardLayer, GpuError> {
    layers
        .get(l)
        .and_then(Option::as_ref)
        .ok_or_else(|| GpuError::Shape {
            what,
            detail: format!("layer {l}: no card experts to run"),
        })
}

/// The card's routed experts of the step: the kernels, each layer's count
/// from the slot map, and the step's buffers. Made at load, after the
/// weights, from the stacks they hold; a layer the map puts experts on holds
/// its three stacks in a type the card leg runs, at `n_card` experts each, and
/// a layer it puts none on holds none — anything else is refused by name.
pub struct CardExperts {
    family: CardFamily,
    k: CardKernels,
    /// Per layer of the description, its card experts; `None` without.
    layers: Vec<Option<CardLayer>>,
    /// Per layer of the description, the experts the expert tier holds: 0
    /// off the tier, and on every layer of a load with no tier.
    tier: Vec<usize>,
    /// Routed expert width: a gate·up slot's rows.
    ff: usize,
    /// The model's width.
    n: usize,
    /// The union of the resident layers' selected down activation forms.
    downs: DownForms,
    /// The one-token step's buffers.
    step: CardBufs,
    /// The buffers of a pass of several rows, on a load that runs one.
    pass: Option<CardBufs>,
}

impl CardExperts {
    /// The card experts of `layers` (the description) as `map` places them and
    /// `w` holds them, `n` wide with routed experts `ff` wide, with buffers
    /// for a pass of `rows` rows when `rows` is more than one. A map that names
    /// an expert tier is refused by name ([`CardExperts::tiered`]). Load-time
    /// only.
    pub fn new(
        gpu: &Gpu,
        w: &Weights,
        family: CardFamily,
        layers: &[LayerSpec],
        map: &SlotMap,
        dims: (usize, usize),
        rows: usize,
    ) -> Result<CardExperts, GpuError> {
        Self::build(gpu, w, family, layers, map, dims, rows, false)
    }

    /// [`CardExperts::new`] for a load that hangs the map's expert tier card:
    /// a layer the tier holds experts of runs as a tier layer (module doc).
    pub fn tiered(
        gpu: &Gpu,
        w: &Weights,
        family: CardFamily,
        layers: &[LayerSpec],
        map: &SlotMap,
        dims: (usize, usize),
        rows: usize,
    ) -> Result<CardExperts, GpuError> {
        Self::build(gpu, w, family, layers, map, dims, rows, true)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the constructors' arguments and whether a tier hangs (rust-quality R8)"
    )]
    fn build(
        gpu: &Gpu,
        w: &Weights,
        family: CardFamily,
        layers: &[LayerSpec],
        map: &SlotMap,
        (n, ff): (usize, usize),
        rows: usize,
        tiers: bool,
    ) -> Result<CardExperts, GpuError> {
        let what = family.what;
        let arms = accepts::leg_arms(family.program).ok_or(GpuError::state(
            what,
            "the selected program's card leg arms",
        ))?;
        let mut per = Vec::with_capacity(layers.len());
        let mut tier_k = Vec::with_capacity(layers.len());
        let mut downs = DownForms::NONE;
        for (l, spec) in layers.iter().enumerate() {
            // A dense block is outside the host run and has no row: no card
            // experts, which the resident-rows check below holds it to. A
            // routed layer the map has no row for is refused by name.
            let (n_card, on_tier) = match spec.ffn {
                Ffn::Dense { .. } => (0, 0),
                Ffn::Moe(_) => {
                    let (card, tier) = (map.on_card(l)?, map.on_tier(l)?);
                    if tier > 0 && !tiers {
                        return Err(GpuError::Shape {
                            what,
                            detail: format!(
                                "layer {l}: the plan puts {tier} routed experts on an expert \
                                 tier; these card experts hang no tier"
                            ),
                        });
                    }
                    (card, tier)
                }
            };
            tier_k.push(on_tier);
            let layer = CardLayer {
                n_card,
                names: (family.stacks)(l),
            };
            if n_card == 0 {
                if let Some(name) = layer.names.iter().find(|name| w.get(name).is_some()) {
                    return Err(GpuError::Tensor {
                        what,
                        name: name.clone(),
                        need: "no resident rows: the slot map puts no expert of its layer on the \
                               card",
                    });
                }
                per.push(None);
                continue;
            }
            let stacks = LegStacks::of_arms(w, layer.names(), arms, STACK_ROLES)?;
            stacks.check(n_card, family.shape(n, ff, 1))?;
            let (_, act) = kernels::down(stacks.down_ty()).ok_or(GpuError::state(
                what,
                "the selected card down's activation form",
            ))?;
            downs = downs.with(act);
            per.push(Some(layer));
        }
        Ok(CardExperts {
            family,
            k: CardKernels {
                leg: CardLeg::load(gpu)?,
                places: CardSlotKernels::load(gpu.context())?,
                arms,
            },
            layers: per,
            tier: tier_k,
            ff,
            n,
            downs,
            step: CardBufs::new(gpu.stream(), &family, n, ff, 1, downs)?,
            pass: (rows > 1)
                .then(|| CardBufs::new(gpu.stream(), &family, n, ff, rows, downs))
                .transpose()?,
        })
    }

    /// Layer `l` computes experts on the card.
    #[must_use]
    pub fn has(&self, l: usize) -> bool {
        layer_of(self.family.what, &self.layers, l).is_ok()
    }

    /// Some layer computes experts on the card.
    #[must_use]
    pub fn any(&self) -> bool {
        self.layers.iter().any(Option::is_some)
    }

    /// The experts of layer `l` the expert tier holds: 0 off the tier, and on
    /// every layer of a load with none.
    #[must_use]
    pub fn tier_k(&self, l: usize) -> usize {
        self.tier.get(l).copied().unwrap_or(0)
    }

    /// Layer `l` has a card sum: card experts on the stage card or the tier.
    #[must_use]
    pub fn sums(&self, l: usize) -> bool {
        self.has(l) || self.tier_k(l) > 0
    }

    /// Some layer holds experts on the expert tier.
    #[must_use]
    pub fn any_tier(&self) -> bool {
        self.tier.iter().any(|&k| k > 0)
    }

    /// The stage card's experts of layer `l`: 0 off the card.
    #[must_use]
    pub fn n_card(&self, l: usize) -> usize {
        layer_of(self.family.what, &self.layers, l).map_or(0, |c| c.n_card)
    }

    /// The layers that compute experts on the card.
    #[must_use]
    pub fn card_layers(&self) -> Vec<bool> {
        self.layers.iter().map(Option::is_some).collect()
    }

    /// The routed experts' width: a gate·up slot's rows.
    #[must_use]
    pub fn ff(&self) -> usize {
        self.ff
    }

    /// The step's sum plus the base of the last card layer enqueued.
    #[must_use]
    pub fn pre(&self) -> &DeviceBuffer<f32> {
        &self.step.pre
    }

    /// The buffers a tier layer's join reads and writes over the step's row,
    /// the slots' places `sel` as the handoff wrote them.
    pub fn step_join<'a>(&'a mut self, sel: &'a DeviceBuffer<u32>) -> TierJoin<'a> {
        self.step.join(sel)
    }

    /// The sum plus the base of the last card layer a pass enqueued, one row
    /// a token of the pass.
    pub fn pass_pre(&self) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.pass.as_ref().map(|p| &p.pre).ok_or(GpuError::State {
            what: self.family.what,
            missing: "the buffers of a pass of several rows",
        })
    }

    /// Device bytes of the step's buffers and a pass's.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.step.bytes() + self.pass.as_ref().map_or(0, CardBufs::bytes)
    }

    /// The buffers for prompt batches of up to `cap` tokens. Load-time or
    /// first-prompt only.
    pub fn batch(&self, gpu: &Gpu, cap: usize) -> Result<CardBatch, GpuError> {
        let stream = gpu.stream();
        Ok(CardBatch {
            bufs: CardBufs::new(stream, &self.family, self.n, self.ff, cap, self.downs)?,
            sel: DeviceBuffer::zeroed(stream, cap * self.family.n_used)?,
        })
    }

    /// Device bytes of [`CardExperts::batch`]'s buffers for `cap` tokens
    /// ([`CardBatch::bytes`]'s formula): the card leg's, the
    /// card sum and the sum plus the base, and the places.
    #[must_use]
    pub fn batch_bytes(&self, cap: usize) -> usize {
        let leg = kernels::leg_bytes(
            cap as u64,
            self.family.n_used as u64,
            self.n as u64,
            self.ff as u64,
            self.downs,
        );
        usize::try_from(leg).expect("the card leg's bytes fit a usize")
            + 2 * size_of::<f32>() * cap * self.n
            + size_of::<u32>() * cap * self.family.n_used
    }

    /// Layer `l`'s card slots of the step (module doc, `m = 1`), the slots'
    /// places `sel` as the handoff wrote them, into [`CardExperts::pre`].
    pub fn enqueue_step(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        rule: Act,
        io: &Io<'_>,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let layer = layer_of(self.family.what, &self.layers, l)?;
        let tiered = self.tier_k(l) > 0;
        self.k.enqueue(
            gpu,
            w,
            &mut self.step,
            (l, layer, self.n),
            (rule, tiered),
            io,
            sel,
        )
    }

    /// Layer `l`'s card slots of a pass of `io.m` rows (module doc), the
    /// slots' places `sel` as the handoffs wrote them, column after column,
    /// into [`CardExperts::pass_pre`]. A pass of more rows than the load's
    /// buffers hold is refused by name.
    pub fn enqueue_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        rule: Act,
        io: &Io<'_>,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = self.family.what;
        let layer = layer_of(what, &self.layers, l)?;
        let bufs = self.pass.as_mut().ok_or(GpuError::State {
            what,
            missing: "the buffers of a pass of several rows",
        })?;
        if io.m == 0 || io.m > bufs.cap() {
            return Err(GpuError::Shape {
                what,
                detail: format!("a pass of {} rows in buffers of {}", io.m, bufs.cap()),
            });
        }
        let tiered = self.tier.get(l).is_some_and(|&k| k > 0);
        self.k
            .enqueue(gpu, w, bufs, (l, layer, self.n), (rule, tiered), io, sel)
    }

    /// Layer `l`'s card slots over a prompt batch of `io.m` tokens (module
    /// doc): every slot's place from the slot map `map` in one launch, `ids`
    /// the router's picks, then the run, into [`CardBatch::pre`]. The batch's
    /// buffers hold at least `io.m` tokens.
    #[allow(
        clippy::too_many_arguments,
        reason = "one batch's inputs: the layer and its rule, the batch's rows and routing, the \
                  slot map and its row"
    )]
    pub fn enqueue_rows(
        &self,
        gpu: &Gpu,
        w: &Weights,
        cb: &mut CardBatch,
        l: usize,
        rule: Act,
        io: &Io<'_>,
        (ids, map, row_off): (&DeviceBuffer<u32>, &DeviceTensor<u32>, usize),
    ) -> Result<(), GpuError> {
        let layer = layer_of(self.family.what, &self.layers, l)?;
        if io.m == 0 || io.m > cb.bufs.cap() {
            return Err(GpuError::Shape {
                what: self.family.what,
                detail: format!("a batch of {} tokens in buffers of {}", io.m, cb.bufs.cap()),
            });
        }
        self.k.places.enqueue_places(
            gpu.stream(),
            &Places {
                ids,
                n: io.m * self.family.n_used,
                map: map.buf(),
                row_off,
                n_expert: self.family.n_expert,
            },
            gpu.layer_sink(l)?,
            &mut cb.sel,
        )?;
        let tiered = self.tier.get(l).is_some_and(|&k| k > 0);
        self.k.enqueue(
            gpu,
            w,
            &mut cb.bufs,
            (l, layer, self.n),
            (rule, tiered),
            io,
            &cb.sel,
        )
    }
}

impl CardKernels {
    /// The six launches (module doc) of layer `l` over `io.m` tokens, `layer`
    /// its card experts and `n` the model's width. On a `tiered` layer the
    /// card leg's four and no more: the slots' downs stay in the buffers for
    /// the layer's join, which sums them with the tier's rows.
    #[allow(
        clippy::too_many_arguments,
        reason = "one run's inputs: the weights, the buffers, the layer's shape and rule, the \
                  rows and their places"
    )]
    fn enqueue(
        &self,
        gpu: &Gpu,
        w: &Weights,
        bufs: &mut CardBufs,
        (l, layer, n): (usize, &CardLayer, usize),
        (rule, tiered): (Act, bool),
        io: &Io<'_>,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let stacks = LegStacks::of_arms(w, layer.names(), self.arms, STACK_ROLES)?;
        let run = LegRun {
            layer: l,
            stacks: &stacks,
            normed: io.normed,
            m: io.m,
            sel,
            n_card: layer.n_card,
            rule,
        };
        let sum = (!tiered).then_some(LegSum {
            w: io.weights,
            acc: &mut bufs.acc,
        });
        self.leg.enqueue(gpu, &mut bufs.leg, &run, sum)?;
        if tiered {
            return Ok(());
        }
        let stream = gpu.stream();
        gpu.elem()
            .enqueue_add(stream, &bufs.acc, io.x1, io.m * n, &mut bufs.pre)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::GgmlType;
    use gguf::write::{Layout, TensorDecl, Writer};
    use model::arch::models::{self, Mixer, Moe, Residual, Router, RouterNorm, Score};

    use crate::card_leg::tests::stack_bytes;
    use crate::host::leg_tier::LegTier;
    use crate::host::tier::{TierBlock, TierExperts, TierInput, TierIo, TierSet};
    use crate::hybrid::HOST;

    const N: usize = 256;
    const FF: usize = 256;
    const USED: usize = 8;
    const EXPERTS: usize = 16;
    const CARD: usize = 4;
    const CAP: usize = 3;
    const FAMILY: CardFamily = CardFamily {
        what: "mixed card wrapper",
        n_expert: EXPERTS,
        n_used: USED,
        stacks: model::arch::names::routed_stacks,
        program: Program::Qwen35Body,
    };

    fn layers(n: usize) -> Vec<LayerSpec> {
        vec![
            LayerSpec {
                mixer: Mixer::None,
                ffn: Ffn::Moe(Moe {
                    experts: EXPERTS as u32,
                    top_k: USED as u32,
                    expert_ff: FF as u32,
                    latent: None,
                    act: models::Act::SwiGlu { limit: None },
                    router: Router {
                        score: Score::Sigmoid,
                        bias: false,
                        norm: RouterNorm::SelectedSum,
                        scale: 1.0,
                        hash: false,
                    },
                    shared: None,
                }),
                residual: Residual::Plain,
                extras: Vec::new(),
            };
            n
        ]
    }

    fn weights(gpu: &Gpu, triples: &[[GgmlType; 3]]) -> Weights {
        let mut decls = Vec::new();
        let mut data = Vec::new();
        for (l, triple) in triples.iter().enumerate() {
            for (part, (name, &ty)) in (FAMILY.stacks)(l).into_iter().zip(triple).enumerate() {
                let (rows, k) = if part == 2 { (N, FF) } else { (FF, N) };
                let bytes = stack_bytes(ty, CARD * rows, k, 7 + (3 * l + part) as u64);
                decls.push(TensorDecl {
                    name,
                    dims: vec![k as u64, rows as u64, CARD as u64],
                    type_id: ty.as_u32(),
                    nbytes: bytes.len() as u64,
                });
                data.push(bytes);
            }
        }
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "bloomery-mixed-wrappers-{}-{call}.gguf",
            std::process::id()
        ));
        let layout = Layout::new(&[], decls.clone()).expect("a layout");
        let file = std::io::BufWriter::new(std::fs::File::create(&path).expect("a file"));
        let mut writer = Writer::new(file, layout).expect("a writer");
        for (decl, bytes) in decls.iter().zip(data) {
            writer.tensor(&decl.name, &bytes).expect("a tensor");
        }
        writer.finish().expect("a finished file");
        let file = gguf::Split::open(&path).expect("the file");
        let weights = Weights::load_where(gpu.stream(), &file, |_| true).expect("resident stacks");
        std::fs::remove_file(path).expect("remove the fixture");
        weights
    }

    fn assert_bits(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(g.is_finite() && w.is_finite(), "lane {i}: {g} vs {w}");
            assert_eq!(g.to_bits(), w.to_bits(), "lane {i}: {g} vs {w}");
        }
    }

    /// Each real wrapper uses its resident layers' union, not a model-wide
    /// quant: both one-form loads and mixed loads conserve bytes and execute
    /// step, pass, batch and tier chunks as the same common leg. Host slots
    /// stay untouched on the tier, including a chunk's short final run.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_mixed_card_and_tier_wrappers_conserve_forms_and_rows() {
        use GgmlType::{IQ3_S, IQ4_XS, Q5_K, Q6_K, Q8_0};
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the reference leg");
        for triples in [
            vec![[IQ3_S, IQ3_S, Q5_K], [IQ4_XS; 3]],
            vec![[Q5_K, Q5_K, Q8_0]],
            vec![[Q5_K, Q5_K, Q6_K]],
            vec![
                [IQ3_S, IQ3_S, Q5_K],
                [Q5_K, Q5_K, Q8_0],
                [Q5_K, Q5_K, Q6_K],
                [IQ4_XS; 3],
            ],
        ] {
            let w = weights(&gpu, &triples);
            let specs = layers(triples.len());
            let map = SlotMap::prefix(0..triples.len(), EXPERTS, CARD).expect("a map");
            let mut card = CardExperts::new(&gpu, &w, FAMILY, &specs, &map, (N, FF), CAP)
                .expect("the card wrapper");
            let mut batch = card.batch(&gpu, CAP).expect("the batch");
            assert_eq!(card.batch_bytes(CAP), batch.bytes());
            let forms = triples.iter().fold(DownForms::NONE, |forms, tys| {
                forms.with(kernels::down(tys[2]).expect("a down").1)
            });
            let counted = |cap| {
                kernels::leg_bytes(cap as u64, USED as u64, N as u64, FF as u64, forms) as usize
            };
            assert_eq!(
                card.bytes(),
                counted(1) + counted(CAP) + 2 * size_of::<f32>() * (1 + CAP) * N,
            );
            assert_eq!(
                batch.bytes(),
                counted(CAP) + 2 * size_of::<f32>() * CAP * N + size_of::<u32>() * CAP * USED,
            );
            let shape = FAMILY.shape(N, FF, 2);
            let set = TierSet::new(
                0..triples.len(),
                EXPERTS,
                vec![(0..CARD as u32).collect(); triples.len()],
            )
            .expect("the tier set");
            let mut tier = LegTier::new(&gpu, &specs, &set, &w, shape, FAMILY.program)
                .expect("the tier wrapper");
            assert_eq!(tier.block_bytes(), counted(shape.cap));
            let host_x: Vec<f32> = (0..CAP * N).map(|i| (i % 71) as f32 / 71.0 - 0.5).collect();
            let x = DeviceBuffer::from_host(stream, &host_x).expect("the rows");
            let base: Vec<f32> = (0..CAP * N).map(|i| (i % 13) as f32 * 0.125).collect();
            let x1 = DeviceBuffer::from_host(stream, &base).expect("the base");
            let route = DeviceBuffer::from_host(stream, &[0.125f32; CAP * USED]).expect("weights");
            let host_sel: Vec<u32> = (0..CAP * USED)
                .map(|i| {
                    if i % USED == USED - 1 {
                        HOST
                    } else {
                        (i % CARD) as u32
                    }
                })
                .collect();
            let sel = DeviceBuffer::from_host(stream, &host_sel).expect("the places");
            let ids = DeviceBuffer::from_host(
                stream,
                &host_sel
                    .iter()
                    .map(|&s| if s == HOST { (EXPERTS - 1) as u32 } else { s })
                    .collect::<Vec<_>>(),
            )
            .expect("the ids");
            let places = DeviceTensor::upload(stream, &map.stage_view(), triples.len(), EXPERTS)
                .expect("the map's card view");
            let act = crate::Q8Act::with_tier_cols(stream, CAP, N).expect("the tier block's form");
            for (l, tys) in triples.iter().enumerate() {
                let names = (FAMILY.stacks)(l);
                let stacks = LegStacks::of_arms(
                    &w,
                    names.each_ref().map(String::as_str),
                    accepts::leg_arms(FAMILY.program).expect("the selected arms"),
                    STACK_ROLES,
                )
                .expect("the reference stacks");
                let mut reference = LegBufs::for_down(
                    stream,
                    FAMILY.shape(N, FF, CAP),
                    kernels::down(tys[2]).expect("a down").1,
                )
                .expect("the reference buffers");
                let mut acc = DeviceBuffer::zeroed(stream, CAP * N).expect("the reference sum");
                for m in [1, CAP] {
                    let io = Io {
                        normed: &x,
                        weights: &route,
                        x1: &x1,
                        m,
                    };
                    let rule = Act::SwigluClamp { limit: 0.0 };
                    leg.enqueue(
                        &gpu,
                        &mut reference,
                        &LegRun {
                            layer: l,
                            stacks: &stacks,
                            normed: &x,
                            m,
                            sel: &sel,
                            n_card: CARD,
                            rule,
                        },
                        Some(LegSum {
                            w: &route,
                            acc: &mut acc,
                        }),
                    )
                    .expect("the reference");
                    let sums = acc.to_host_vec(stream).expect("the reference sums");
                    let want: Vec<f32> = sums[..m * N]
                        .iter()
                        .zip(&base)
                        .map(|(&a, &b)| a + b)
                        .collect();
                    let got = if m == 1 {
                        card.enqueue_step(&gpu, &w, l, rule, &io, &sel)
                            .expect("step");
                        card.pre()
                    } else {
                        card.enqueue_pass(&gpu, &w, l, rule, &io, &sel)
                            .expect("pass");
                        card.pass_pre().expect("pass output")
                    };
                    assert_bits(&got.to_host_vec(stream).expect("output")[..m * N], &want);
                    card.enqueue_rows(
                        &gpu,
                        &w,
                        &mut batch,
                        l,
                        rule,
                        &io,
                        (&ids, &places, l * EXPERTS),
                    )
                    .expect("batch");
                    assert_bits(
                        &batch.pre().to_host_vec(stream).expect("batch output")[..m * N],
                        &want,
                    );
                    let mut down =
                        DeviceBuffer::from_host(stream, &vec![1.0e30f32; CAP * USED * N])
                            .expect("the tier output");
                    if m == 1 {
                        tier.enqueue_layer(
                            &gpu,
                            &w,
                            l,
                            TierIo {
                                act: TierInput::F32(&x),
                                sel: &sel,
                                rows: &mut down,
                                cols: m,
                            },
                        )
                        .expect("tier step");
                    } else {
                        tier.enqueue_block(
                            &gpu,
                            &w,
                            l,
                            TierBlock {
                                x: &x,
                                act: &act,
                                sel: &sel,
                                cols: m,
                                down: &mut down,
                            },
                        )
                        .expect("tier chunks");
                    }
                    let got = down.to_host_vec(stream).expect("tier output");
                    let want = reference
                        .down()
                        .to_host_vec(stream)
                        .expect("reference rows");
                    for (s, &place) in host_sel[..m * USED].iter().enumerate() {
                        if place == HOST {
                            assert!(got[s * N..(s + 1) * N].iter().all(|&v| v == 1.0e30));
                        } else {
                            assert_bits(&got[s * N..(s + 1) * N], &want[s * N..(s + 1) * N]);
                        }
                    }
                }
            }
            assert!(gpu.take_fault().expect("the fault word").is_none());
        }
        for (triple, program) in [
            ([IQ3_S, Q5_K, Q8_0], Program::Qwen35Body),
            ([Q5_K, Q5_K, Q6_K], Program::Solar2Body),
        ] {
            let w = weights(&gpu, &[triple]);
            let specs = layers(1);
            let map = SlotMap::prefix(0..1, EXPERTS, CARD).expect("a map");
            let family = CardFamily { program, ..FAMILY };
            assert!(
                matches!(
                    CardExperts::new(&gpu, &w, family, &specs, &map, (N, FF), CAP),
                    Err(GpuError::Tensor { .. }),
                ),
                "{program:?}: {triple:?}"
            );
            let set =
                TierSet::new(0..1, EXPERTS, vec![(0..CARD as u32).collect()]).expect("a tier set");
            assert!(
                matches!(
                    LegTier::new(&gpu, &specs, &set, &w, family.shape(N, FF, 2), program),
                    Err(GpuError::Tensor { .. }),
                ),
                "{program:?}: {triple:?}"
            );
        }
    }
}
