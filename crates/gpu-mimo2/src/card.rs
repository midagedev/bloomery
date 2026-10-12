//! The routed block's card experts: the experts the slot map puts on the card,
//! computed by the common card leg (`bloomery_gpu::card_leg`: the per-slot
//! pair of the kernel table, launched by the entries the table names for a
//! stack's type) in the host leg's shadow, and summed here.
//!
//! One run is six launches, over `m` tokens of a layer's normed rows:
//! - the card leg's four (`card_leg`): the normed rows in q8_1, the gate·up
//!   `_sel` with the routed SwiGLU (slot `s` against column `s / N_USED`), the
//!   q8_1 of the card slots' `h` columns (the host's slots are left alone) and
//!   the down `_sel`;
//! - the card slots' weighted sum, a token's slots in slot order (the leg's
//!   sum stage, `card_slots`);
//! - the sum plus the residual after the attention, the `pre` the layer's back
//!   adds the host's sum to.
//!
//! The one-token step is the run at `m = 1`, its places written by the
//! handoff; a prompt batch is the run at `m = t`, one launch each over the
//! batch's `t · N_USED` slots, its places written by one launch
//! (`CardSlotKernels::enqueue_places`). Each slot's rows are what the step's
//! launches write for it, whatever `m`, so the two feeds leave the same bits.
//!
//! MiMo facts that fix this file: the routed block has no shared expert, so
//! the shadow is the card slots alone and the layer's back adds the host's sum
//! to `acc + x1`; the routed SwiGLU is the description's limit (0: plain);
//! no expert tier hangs on these experts.

use bloomery_gpu::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum, leg_bytes};
use bloomery_gpu::card_slots::{CardSlotKernels, Places};
use bloomery_gpu::hybrid::SlotMap;
use bloomery_gpu::kquant::Act;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use bloomery_gpu_deepseek41::router::mimo2::{N_EXPERT, N_USED};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::mimo2::names;
use models::{Ffn, LayerSpec};

/// What the card experts' errors name.
const CARD: &str = "mimo2 CardExperts";

/// The kernels a run launches: the card leg's (its sum stage included), and
/// the places' of a prompt batch.
struct CardKernels {
    leg: CardLeg,
    places: CardSlotKernels,
}

/// The card leg's shape for runs of up to `cap` tokens of `n` values through
/// experts of `ff` rows.
const fn shape(n: usize, ff: usize, cap: usize) -> LegShape {
    LegShape {
        cap,
        slots_per_col: N_USED,
        n,
        ff,
    }
}

/// One run's buffers for up to `cap` tokens: the card leg's, the card sum and
/// the sum plus the residual.
pub(crate) struct CardBufs {
    leg: LegBufs,
    acc: DeviceBuffer<f32>,
    pre: DeviceBuffer<f32>,
}

impl CardBufs {
    /// The buffers for runs of up to `cap` tokens of `n` values through
    /// experts of `ff` rows. Load-time or first-prompt only.
    fn new(stream: &CudaStream, n: usize, ff: usize, cap: usize) -> Result<CardBufs, GpuError> {
        Ok(CardBufs {
            leg: LegBufs::new(stream, shape(n, ff, cap))?,
            acc: DeviceBuffer::zeroed(stream, cap * n)?,
            pre: DeviceBuffer::zeroed(stream, cap * n)?,
        })
    }

    /// Tokens the buffers hold.
    fn cap(&self) -> usize {
        self.leg.shape().cap
    }

    /// Device bytes of the buffers.
    fn bytes(&self) -> usize {
        self.leg.bytes() + self.acc.num_bytes() + self.pre.num_bytes()
    }

    /// The sum plus the residual of the last run, `m` rows.
    pub(crate) fn pre(&self) -> &DeviceBuffer<f32> {
        &self.pre
    }
}

/// A prompt batch's card buffers: [`CardBufs`] for the batch's tokens and the
/// places its slots resolve to.
pub(crate) struct CardBatch {
    bufs: CardBufs,
    sel: DeviceBuffer<u32>,
}

impl CardBatch {
    /// The buffers for batches of up to `cap` tokens. Load-time or
    /// first-prompt only.
    pub(crate) fn new(gpu: &Gpu, n: usize, ff: usize, cap: usize) -> Result<CardBatch, GpuError> {
        let stream = gpu.stream();
        Ok(CardBatch {
            bufs: CardBufs::new(stream, n, ff, cap)?,
            sel: DeviceBuffer::zeroed(stream, cap * N_USED)?,
        })
    }

    /// Device bytes of the buffers: what [`CardBatch::new`] allocates for `cap`
    /// tokens of `n` values through experts of `ff` rows, counted before it
    /// does ([`card_batch_bytes`]).
    pub(crate) fn bytes(&self) -> usize {
        self.bufs.bytes() + self.sel.num_bytes()
    }

    /// The sum plus the residual of the last run over the batch, `t` rows.
    pub(crate) fn pre(&self) -> &DeviceBuffer<f32> {
        self.bufs.pre()
    }
}

/// What [`CardExperts::enqueue_step`] and [`CardExperts::enqueue_rows`] read:
/// `m` tokens' normed rows, their slots' places and routing weights, and the
/// residual after the attention.
pub(crate) struct Io<'a> {
    pub normed: &'a DeviceBuffer<f32>,
    pub weights: &'a DeviceBuffer<f32>,
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
    fn of(l: usize, n_card: usize) -> CardLayer {
        CardLayer {
            n_card,
            names: [
                names::ffn_gate_exps(l),
                names::ffn_up_exps(l),
                names::ffn_down_exps(l),
            ],
        }
    }

    /// The stacks' names, as the card leg reads them.
    fn names(&self) -> [&str; 3] {
        self.names.each_ref().map(String::as_str)
    }
}

/// Layer `l`'s card experts of `layers`, refused by name on a layer with none.
fn layer_of(layers: &[Option<CardLayer>], l: usize) -> Result<&CardLayer, GpuError> {
    layers
        .get(l)
        .and_then(Option::as_ref)
        .ok_or_else(|| GpuError::Shape {
            what: CARD,
            detail: format!("layer {l}: no card experts to run"),
        })
}

/// Device bytes of the card buffers of a prompt batch of up to `cap` tokens of
/// `n` values through experts of `ff` rows ([`CardBatch::bytes`]'s formula):
/// the card leg's ([`leg_bytes`]), the card sum and the sum plus the
/// residual, and the places.
#[must_use]
pub(crate) fn card_batch_bytes(n: usize, ff: usize, cap: usize) -> usize {
    leg_bytes(shape(n, ff, cap)) + 2 * size_of::<f32>() * cap * n + size_of::<u32>() * cap * N_USED
}

/// The card's routed experts of the step: the kernels, each layer's count
/// from the slot map, and the step's buffers. Made at load, after the
/// weights, from the stacks they hold; a layer the map puts experts on holds
/// its three stacks in a type the card leg runs, at `n_card` experts each, and
/// a layer it puts none on holds none — anything else is refused by name.
pub(crate) struct CardExperts {
    k: CardKernels,
    /// Per layer of the description, its card experts; `None` without.
    layers: Vec<Option<CardLayer>>,
    /// Routed expert width: a gate·up slot's rows.
    ff: usize,
    /// The model's width.
    n: usize,
    /// The one-token step's buffers.
    step: CardBufs,
    /// The buffers of a pass of several rows, on a load that runs one.
    pass: Option<CardBufs>,
}

impl CardExperts {
    /// The card experts of `layers` (the description) as `map` places them and
    /// `w` holds them, `n` wide with routed experts `ff` wide, with buffers
    /// for a pass of `rows` rows when `rows` is more than one. Load-time only.
    pub(crate) fn new(
        gpu: &Gpu,
        w: &Weights,
        layers: &[LayerSpec],
        map: &SlotMap,
        n: usize,
        ff: usize,
        rows: usize,
    ) -> Result<CardExperts, GpuError> {
        let mut per = Vec::with_capacity(layers.len());
        for (l, spec) in layers.iter().enumerate() {
            // A dense block is outside the host run and has no row: no card
            // experts, which the resident-rows check below holds it to. A
            // routed layer the map has no row for is refused by name.
            let n_card = match spec.ffn {
                Ffn::Dense { .. } => 0,
                Ffn::Moe(_) => {
                    let (card, tier) = (map.on_card(l)?, map.on_tier(l)?);
                    if tier > 0 {
                        return Err(GpuError::Shape {
                            what: CARD,
                            detail: format!(
                                "layer {l}: the plan puts {tier} routed experts on an expert \
                                 tier; this program hangs no tier"
                            ),
                        });
                    }
                    card
                }
            };
            let layer = CardLayer::of(l, n_card);
            if n_card == 0 {
                if let Some(name) = layer.names.iter().find(|name| w.get(name).is_some()) {
                    return Err(GpuError::Tensor {
                        what: CARD,
                        name: name.clone(),
                        need: "no resident rows: the slot map puts no expert of its layer on the \
                               card",
                    });
                }
                per.push(None);
                continue;
            }
            LegStacks::of(w, layer.names())?.check(n_card, shape(n, ff, 1))?;
            per.push(Some(layer));
        }
        Ok(CardExperts {
            k: CardKernels {
                leg: CardLeg::load(gpu)?,
                places: CardSlotKernels::load(gpu.context())?,
            },
            layers: per,
            ff,
            n,
            step: CardBufs::new(gpu.stream(), n, ff, 1)?,
            pass: (rows > 1)
                .then(|| CardBufs::new(gpu.stream(), n, ff, rows))
                .transpose()?,
        })
    }

    /// Layer `l` computes experts on the card.
    pub(crate) fn has(&self, l: usize) -> bool {
        layer_of(&self.layers, l).is_ok()
    }

    /// Some layer computes experts on the card.
    pub(crate) fn any(&self) -> bool {
        self.layers.iter().any(Option::is_some)
    }

    /// The layers that compute experts on the card.
    pub(crate) fn card_layers(&self) -> Vec<bool> {
        self.layers.iter().map(Option::is_some).collect()
    }

    /// The routed experts' width: a gate·up slot's rows.
    pub(crate) fn ff(&self) -> usize {
        self.ff
    }

    /// The step's sum plus the residual of the last card layer enqueued.
    pub(crate) fn pre(&self) -> &DeviceBuffer<f32> {
        self.step.pre()
    }

    /// The sum plus the residual of the last card layer a pass enqueued,
    /// one row a token of the pass.
    pub(crate) fn pass_pre(&self) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.pass
            .as_ref()
            .map(CardBufs::pre)
            .ok_or(GpuError::State {
                what: CARD,
                missing: "the buffers of a pass of several rows",
            })
    }

    /// Device bytes of the step's buffers and a pass's.
    pub(crate) fn bytes(&self) -> usize {
        self.step.bytes() + self.pass.as_ref().map_or(0, CardBufs::bytes)
    }

    /// Layer `l`'s card slots of the step (module doc, `m = 1`), the slots'
    /// places `sel` as the handoff wrote them, into [`CardExperts::pre`].
    pub(crate) fn enqueue_step(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        rule: Act,
        io: &Io<'_>,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let layer = layer_of(&self.layers, l)?;
        self.k
            .enqueue(gpu, w, &mut self.step, (l, layer, self.n), rule, io, sel)
    }

    /// Layer `l`'s card slots of a pass of `io.m` rows (module doc), the
    /// slots' places `sel` as the handoffs wrote them, column after column,
    /// into [`CardExperts::pass_pre`]. A pass of more rows than the load's
    /// buffers hold is refused by name.
    pub(crate) fn enqueue_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        rule: Act,
        io: &Io<'_>,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let layer = layer_of(&self.layers, l)?;
        let bufs = self.pass.as_mut().ok_or(GpuError::State {
            what: CARD,
            missing: "the buffers of a pass of several rows",
        })?;
        if io.m == 0 || io.m > bufs.cap() {
            return Err(GpuError::Shape {
                what: CARD,
                detail: format!("a pass of {} rows in buffers of {}", io.m, bufs.cap()),
            });
        }
        self.k
            .enqueue(gpu, w, bufs, (l, layer, self.n), rule, io, sel)
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
    pub(crate) fn enqueue_rows(
        &self,
        gpu: &Gpu,
        w: &Weights,
        cb: &mut CardBatch,
        l: usize,
        rule: Act,
        io: &Io<'_>,
        (ids, map, row_off): (&DeviceBuffer<u32>, &DeviceTensor<u32>, usize),
    ) -> Result<(), GpuError> {
        let layer = layer_of(&self.layers, l)?;
        if io.m == 0 || io.m > cb.bufs.cap() {
            return Err(GpuError::Shape {
                what: CARD,
                detail: format!("a batch of {} tokens in buffers of {}", io.m, cb.bufs.cap()),
            });
        }
        self.k.places.enqueue_places(
            gpu.stream(),
            &Places {
                ids,
                n: io.m * N_USED,
                map: map.buf(),
                row_off,
                n_expert: N_EXPERT,
            },
            gpu.layer_sink(l)?,
            &mut cb.sel,
        )?;
        self.k
            .enqueue(gpu, w, &mut cb.bufs, (l, layer, self.n), rule, io, &cb.sel)
    }
}

impl CardKernels {
    /// The six launches (module doc) of layer `l` over `io.m` tokens, `layer`
    /// its card experts and `n` the model's width.
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
        rule: Act,
        io: &Io<'_>,
        sel: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let stacks = LegStacks::of(w, layer.names())?;
        let run = LegRun {
            layer: l,
            stacks: &stacks,
            normed: io.normed,
            m: io.m,
            sel,
            n_card: layer.n_card,
            rule,
        };
        let sum = LegSum {
            w: io.weights,
            acc: &mut bufs.acc,
        };
        self.leg.enqueue(gpu, &mut bufs.leg, &run, Some(sum))?;
        let stream = gpu.stream();
        gpu.elem()
            .enqueue_add(stream, &bufs.acc, io.x1, io.m * n, &mut bufs.pre)
    }
}
