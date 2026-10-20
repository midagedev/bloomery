//! The card leg of a routed layer's experts, common to every program that runs
//! experts on the card by the kernel table (`model::placement::kernels`):
//! the four launches from a layer's normed rows to its slots' down rows.
//!
//! One run covers `m` tokens of `slots_per_col` routed slots each:
//! - the normed rows in q8_1 (`m` columns, `Gpu::enqueue_quantize_q8_1_cols`);
//! - the gate·up `_sel` of the entry [`kernels::gate_up`] names for the
//!   stacks' type, the activation rule a launch argument, slot `s` against
//!   column `s / slots_per_col`, writing `h` (`ff` rows a slot);
//! - the q8_1 of the card slots' `h` columns in the form the down entry reads
//!   ([`DownAct`]: `Q4kSelKernels::enqueue_quantize_sel` for 128-value
//!   blocks, `Q5Kernels::enqueue_quantize_q8_sel` for 32-value blocks; a slot
//!   the run places off the card is left alone);
//! - the down `_sel` of the entry [`kernels::down`] names, writing `down`
//!   (`n` rows a slot).
//!
//! - optionally the card slots' weighted sum ([`LegSum`], `crate::card_slots`):
//!   a token's slots in slot order, written to a caller's `acc`.
//!
//! A slot's rows are what a one-token run writes for it, whatever `m`: the
//! launches are slot-local and read their columns by slot, so a unit of a
//! prompt batch and a decode step leave the same bits for the same slot.
//!
//! The relu² form ([`Relu2Leg`]) is the same four launches for an expert with
//! no gate, `h = relu(up · x)²`, over the 32-value-block types
//! ([`kernels::card_sel32_runs`]) alone:
//! - the token's normed row in 32-value q8_1, once in the column of each of
//!   its slots (`Q5Kernels::enqueue_quantize_q8_rep`), so the up is a
//!   down-shaped `_sel` that reads column `s` for slot `s` ([`kernels::up`])
//!   and no `_sel` entry has a column rule of its own;
//! - the up `_sel`, writing `h` (`ff` rows a slot);
//! - the q8_1 of the card slots' `max(h, 0)²` in the form the down reads
//!   (`Q5Kernels::enqueue_quantize_q8_sel_relu_sqr`; a slot the run places
//!   off the card is left alone);
//! - the down `_sel` and, optionally, the card slots' weighted sum.
//!
//! The widths need not be multiples of 256: `n` and `ff` are multiples of 32.
//!
//! What stays with the caller, because it is the family's: the slots' places
//! `sel` (the handoff's, or a batch's `CardSlotKernels::enqueue_places`), what
//! the sum is combined with (a shared expert, the residual, the host tier's
//! rows) or its own sum where the common one is not its rule (the expert
//! tier's rows beside the card's, another slot count or weight pitch), and
//! the routing weights. A caller whose down rows live in a buffer of its own
//! (the expert tier's staged rows) runs [`CardLeg::enqueue_rows`].
//!
//! The run launches the entries [`kernels::card_gate_up_runs`] and
//! [`kernels::card_down_runs`] admit, each stack by its role: those whose
//! launchers read an activation of more columns than a run uses (a
//! [`LegBufs`] sized for `cap` tokens serves a unit of fewer in one launch).
//! The buffers hold the down's activation in one form ([`LegBufs::for_down`]);
//! a stack whose down entry reads the other is refused by name before the
//! first launch.
//! A stack of any other type is refused by name when the stacks are read
//! ([`LegStacks::of`]), and an entry the table names but the run has no arm
//! for is refused by name at the launch.

use crate::card_slots::{CardAcc, CardSlotKernels, CardSlots};
use crate::fault::FaultSink;
use crate::iq_sel::{IqDown, IqGateUp};
use crate::kquant::{Act, GateUpAct, SelDown};
use crate::mxfp4_sel::{Mxfp4SelKernels, Nvfp4Down, Nvfp4GateUp, Nvfp4SelKernels, Nvfp4Stack};
use crate::q4k_sel::QuantSel;
use crate::q5::Q8Blocks32;
use crate::q5_1_sel::{Q51SelDown, Q51SelKernels};
use crate::q8_0_sel32::{Q80SelDown, Q80SelKernels};
use crate::quant_sel::QuantSelKernels;
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError, Q8Act};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;
use model::placement::kernels::{self, DownAct, DownEntry, GateUpEntry, q8_act_bytes};

/// What the leg's errors name.
const WHAT: &str = "CardLeg";

/// What a [`LegBufs`] is sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegShape {
    /// Tokens (columns of the normed rows) one run covers at most.
    pub cap: usize,
    /// Routed slots a token carries: the gate·up's `slots_per_col`.
    pub slots_per_col: usize,
    /// The model's width: a normed row's values and a down slot's rows.
    pub n: usize,
    /// The routed experts' width: a gate·up slot's rows.
    pub ff: usize,
}

impl LegShape {
    /// Slots of a run of `cap` tokens.
    const fn slots(self) -> usize {
        self.cap * self.slots_per_col
    }
}

/// The card slots' q8_1 columns in the form a down entry reads.
enum ActH {
    Q8Act(Q8Act),
    Blocks32(Q8Blocks32),
}

impl ActH {
    fn form(&self) -> DownAct {
        match self {
            ActH::Q8Act(_) => DownAct::Q8Act,
            ActH::Blocks32(_) => DownAct::Q8Blocks32,
        }
    }

    fn bytes(&self) -> usize {
        match self {
            ActH::Q8Act(a) => a.device_bytes(),
            ActH::Blocks32(a) => a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes(),
        }
    }
}

/// One run's buffers for up to `shape.cap` tokens: the normed rows' q8_1 form,
/// the gate·up's rows, the card slots' q8_1 columns and the down's rows.
pub struct LegBufs {
    shape: LegShape,
    act_x: Q8Act,
    h: DeviceBuffer<f32>,
    act_h: ActH,
    down: DeviceBuffer<f32>,
}

impl LegBufs {
    /// The buffers for runs of `shape` whose down entries read [`DownAct::
    /// Q8Act`] (the K-quant, MXFP4 and IQ2_XS, IQ3_XXS, IQ3_S and IQ4_XS downs).
    /// Load-time or first-prompt only.
    pub fn new(stream: &CudaStream, shape: LegShape) -> Result<LegBufs, GpuError> {
        LegBufs::for_down(stream, shape, DownAct::Q8Act)
    }

    /// The buffers for runs of `shape` whose down entries read `act`, the form
    /// [`kernels::down`] names for the stacks' type. Load-time or first-prompt
    /// only.
    pub fn for_down(
        stream: &CudaStream,
        shape: LegShape,
        act: DownAct,
    ) -> Result<LegBufs, GpuError> {
        let slots = shape.slots();
        let act_h = match act {
            DownAct::Q8Act => ActH::Q8Act(Q8Act::with_slots(stream, slots, shape.ff)?),
            DownAct::Q8Blocks32 => ActH::Blocks32(Q8Blocks32::with_slots(stream, shape.ff, slots)?),
        };
        Ok(LegBufs {
            shape,
            act_x: Q8Act::with_tier_cols(stream, shape.cap, shape.n)?,
            h: DeviceBuffer::zeroed(stream, slots * shape.ff)?,
            act_h,
            down: DeviceBuffer::zeroed(stream, slots * shape.n)?,
        })
    }

    /// The shape the buffers were made for.
    #[must_use]
    pub fn shape(&self) -> LegShape {
        self.shape
    }

    /// Device bytes of the buffers: [`leg_bytes_for`] of their shape and
    /// form, as allocated.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.act_x.device_bytes() + self.h.num_bytes() + self.act_h.bytes() + self.down.num_bytes()
    }

    /// The down rows of the last run: `n` values a slot, slot-major, the
    /// slots the run placed on the card (the rest are as an earlier run left
    /// them).
    #[must_use]
    pub fn down(&self) -> &DeviceBuffer<f32> {
        &self.down
    }
}

/// Device bytes a [`LegBufs::new`] of `shape` allocates, counted before it
/// does: [`leg_bytes_for`] of the [`DownAct::Q8Act`] form.
#[must_use]
pub fn leg_bytes(shape: LegShape) -> usize {
    leg_bytes_for(shape, DownAct::Q8Act)
}

/// Device bytes a [`LegBufs::for_down`] of `shape` and `act` allocates,
/// counted before it does: the normed rows' q8_1 form (`cap` columns of `n`),
/// the gate·up's rows, the card slots' q8_1 columns (`cap · slots_per_col` of
/// `ff`) and the down's rows.
#[must_use]
pub fn leg_bytes_for(shape: LegShape, act: DownAct) -> usize {
    let slots = shape.slots();
    let f = size_of::<f32>();
    let act_h = match act {
        DownAct::Q8Act => usize::try_from(q8_act_bytes(slots as u64, shape.ff as u64))
            .expect("the q8_1 activation's bytes fit a usize"),
        DownAct::Q8Blocks32 => q8_blocks32_bytes(slots, shape.ff),
    };
    usize::try_from(q8_act_bytes(shape.cap as u64, shape.n as u64))
        .expect("the q8_1 activation's bytes fit a usize")
        + f * slots * shape.ff
        + act_h
        + f * slots * shape.n
}

/// Device bytes of a [`Q8Blocks32`] of `cols` columns of `k` values (a multiple
/// of 32): per column `256 · ceil(k_blocks / 32)` words of `q` and `k_blocks`
/// values each of `s8` and `d8`, of 4 bytes, `k_blocks = k / 32`.
const fn q8_blocks32_bytes(cols: usize, k: usize) -> usize {
    let k_blocks = k / 32;
    cols * 4 * (256 * k_blocks.div_ceil(32) + 2 * k_blocks)
}

/// A resident routed stack's device form: the file's blocks as words
/// (`DevWeight::KQuant`), or NVFP4's planes and its experts' scales
/// (`DevWeight::Nvfp4Stack`).
#[derive(Clone, Copy)]
enum StackW<'a> {
    Words(&'a DeviceTensor<u32>),
    Nvfp4(Nvfp4Stack<'a>),
}

/// One resident routed stack: its name, file type, device form and row width.
struct LegStack<'a> {
    name: &'a str,
    ty: GgmlType,
    w: StackW<'a>,
    k: usize,
}

impl<'a> LegStack<'a> {
    /// The resident stack `name`: a `KQuant` stack of a type `runs` admits in
    /// the stack's role, or an NVFP4 stack in its planes where `planes`
    /// admits NVFP4; `role` is the refusal's words.
    fn of(
        w: &'a Weights,
        name: &'a str,
        runs: fn(GgmlType) -> bool,
        planes: fn(GgmlType) -> bool,
        role: &'static str,
    ) -> Result<LegStack<'a>, GpuError> {
        match w.get(name) {
            Some(DevWeight::KQuant { ty, w, k }) if runs(*ty) => Ok(LegStack {
                name,
                ty: *ty,
                w: StackW::Words(w),
                k: *k,
            }),
            Some(DevWeight::Nvfp4Stack { k, .. }) if planes(GgmlType::NVFP4) => Ok(LegStack {
                name,
                ty: GgmlType::NVFP4,
                w: StackW::Nvfp4(w.nvfp4_stack(WHAT, name)?),
                k: *k,
            }),
            _ => Err(GpuError::Tensor {
                what: WHAT,
                name: name.to_owned(),
                need: role,
            }),
        }
    }

    /// Rows the stack holds, every expert's.
    fn rows(&self) -> usize {
        match self.w {
            StackW::Words(w) => w.rows(),
            StackW::Nvfp4(s) => s.qs.rows(),
        }
    }

    /// The stack's words, for an entry over the file's blocks.
    fn words(&self) -> Result<&'a DeviceTensor<u32>, GpuError> {
        match self.w {
            StackW::Words(w) => Ok(w),
            StackW::Nvfp4(_) => Err(self.unrun()),
        }
    }

    /// The stack's planes and scales, for an NVFP4 entry.
    fn planes(&self) -> Result<Nvfp4Stack<'a>, GpuError> {
        match self.w {
            StackW::Nvfp4(s) => Ok(s),
            StackW::Words(_) => Err(self.unrun()),
        }
    }

    /// The stack's type has an entry in the table that this leg has no arm
    /// for.
    fn unrun(&self) -> GpuError {
        GpuError::Tensor {
            what: WHAT,
            name: self.name.to_owned(),
            need: "a type whose gate·up and down entries the card leg launches",
        }
    }
}

/// A layer's three resident routed stacks, the gate and up of one type.
pub struct LegStacks<'a> {
    gate: LegStack<'a>,
    up: LegStack<'a>,
    down: LegStack<'a>,
}

impl<'a> LegStacks<'a> {
    /// The stacks named `[gate, up, down]` in `w`, each a resident
    /// `DevWeight::KQuant` or an NVFP4 stack in its planes: the gate and up of
    /// one type [`kernels::card_gate_up_runs`] or [`kernels::card_planes_runs`]
    /// admits, the down of a type [`kernels::card_down_runs`] or
    /// [`kernels::card_planes_runs`] admits (not necessarily the same one).
    /// Allocation-free on success.
    pub fn of(w: &'a Weights, names: [&'a str; 3]) -> Result<LegStacks<'a>, GpuError> {
        const GATE_UP: &str = "a resident routed gate or up stack of a type the card leg runs (`kernels::card_gate_up_runs`)";
        const DOWN: &str =
            "a resident routed down stack of a type the card leg runs (`kernels::card_down_runs`)";
        let [gate, up, down] = names;
        let (gate, up, down) = (
            LegStack::of(
                w,
                gate,
                kernels::card_gate_up_runs,
                kernels::card_planes_runs,
                GATE_UP,
            )?,
            LegStack::of(
                w,
                up,
                kernels::card_gate_up_runs,
                kernels::card_planes_runs,
                GATE_UP,
            )?,
            LegStack::of(
                w,
                down,
                kernels::card_down_runs,
                kernels::card_planes_runs,
                DOWN,
            )?,
        );
        if up.ty != gate.ty {
            return Err(GpuError::Tensor {
                what: WHAT,
                name: up.name.to_owned(),
                need: "the gate stack's type",
            });
        }
        Ok(LegStacks { gate, up, down })
    }

    /// The stacks hold `n_card` experts of `shape` each: the gate's and up's
    /// `ff` rows of `n` values, the down's `n` rows of `ff`.
    pub fn check(&self, n_card: usize, shape: LegShape) -> Result<(), GpuError> {
        let LegShape { n, ff, .. } = shape;
        for (st, rows, k) in [(&self.gate, ff, n), (&self.up, ff, n), (&self.down, n, ff)] {
            if st.rows() != n_card * rows || st.k != k {
                return Err(GpuError::Tensor {
                    what: WHAT,
                    name: st.name.to_owned(),
                    need: "the rows of the layer's card experts, of the shape's width",
                });
            }
        }
        Ok(())
    }
}

/// What [`CardLeg::enqueue`] reads: layer `layer`'s `stacks`, `m` tokens'
/// normed rows `normed`, the slots' places `sel` (`m · slots_per_col` ids; an
/// id below `n_card` is a card expert, the rest are left alone) and the
/// activation rule.
pub struct LegRun<'a> {
    pub layer: usize,
    pub stacks: &'a LegStacks<'a>,
    pub normed: &'a DeviceBuffer<f32>,
    pub m: usize,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_card: usize,
    pub rule: Act,
}

/// The card slots' sum a run ends with ([`CardLeg::enqueue`]): `w` the
/// slots' routing weights, `slots_per_col` a token and token-major, and `acc`
/// the sum, `m` rows of `n` values. A token's slots are summed in slot order
/// over the slots whose place is below `n_card`; `slots_per_col` is 6, 8 or
/// 10, any other count is refused by name before the run launches anything.
pub struct LegSum<'a> {
    pub w: &'a DeviceBuffer<f32>,
    pub acc: &'a mut DeviceBuffer<f32>,
}

/// The modules the leg's launches run on.
pub struct CardLeg {
    quant: QuantSelKernels,
    mxfp4: Mxfp4SelKernels,
    nvfp4: Nvfp4SelKernels,
    q51: Q51SelKernels,
    q80: Q80SelKernels,
    slots: CardSlotKernels,
}

/// The buffers of a [`LegBufs`] a run writes besides its down rows.
struct Scratch<'a> {
    shape: LegShape,
    act_x: &'a mut Q8Act,
    h: &'a mut DeviceBuffer<f32>,
    act_h: &'a mut ActH,
}

impl CardLeg {
    /// The modules on `gpu`. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<CardLeg, GpuError> {
        let (ctx, word) = (gpu.context(), gpu.fault_word());
        Ok(CardLeg {
            quant: QuantSelKernels::load(ctx, word)?,
            mxfp4: Mxfp4SelKernels::load(ctx, word)?,
            nvfp4: Nvfp4SelKernels::load(ctx, word)?,
            q51: Q51SelKernels::load(ctx, word)?,
            q80: Q80SelKernels::load(ctx, word)?,
            slots: CardSlotKernels::load(ctx)?,
        })
    }

    /// The modules the 32-value-block `_sel`s run on.
    fn sel32(&self) -> Sel32<'_> {
        Sel32 {
            q51: &self.q51,
            q80: &self.q80,
            iq: &self.quant,
        }
    }

    /// Enqueue on `gpu`'s stream the four launches (module doc) of `run` into
    /// `bufs`, and with `sum` the card slots' sum after them; `run.m` outside
    /// `1..=bufs.shape().cap`, a down entry that reads the other activation
    /// form than `bufs` holds, or a sum over a slot count the sum has no entry
    /// for, is refused by name before the first launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        gpu: &Gpu,
        bufs: &mut LegBufs,
        run: &LegRun<'_>,
        sum: Option<LegSum<'_>>,
    ) -> Result<(), GpuError> {
        let LegBufs {
            shape,
            act_x,
            h,
            act_h,
            down,
        } = bufs;
        let shape = *shape;
        if sum.is_some() {
            check_sum(shape)?;
        }
        let scratch = Scratch {
            shape,
            act_x,
            h,
            act_h,
        };
        self.launch(gpu, scratch, run, down)?;
        let Some(sum) = sum else {
            return Ok(());
        };
        enqueue_sum(
            &self.slots,
            gpu,
            shape,
            (down, run.sel),
            (run.layer, run.m, run.n_card),
            sum,
        )
    }

    /// [`CardLeg::enqueue`]'s four launches, the down rows written to `down`
    /// (`shape.n` values a slot, slot-major, at least `m · slots_per_col`
    /// slots) instead of `bufs`'s own: for a caller whose rows live in a
    /// buffer of its own, and which sums them itself. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_rows(
        &self,
        gpu: &Gpu,
        bufs: &mut LegBufs,
        run: &LegRun<'_>,
        down: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let LegBufs {
            shape,
            act_x,
            h,
            act_h,
            ..
        } = bufs;
        let scratch = Scratch {
            shape: *shape,
            act_x,
            h,
            act_h,
        };
        self.launch(gpu, scratch, run, down)
    }

    /// The four launches of `run` into `scratch` and `down`.
    fn launch(
        &self,
        gpu: &Gpu,
        scratch: Scratch<'_>,
        run: &LegRun<'_>,
        down: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Scratch {
            shape:
                LegShape {
                    cap,
                    slots_per_col,
                    n,
                    ff,
                },
            act_x,
            h,
            act_h,
        } = scratch;
        if run.m == 0 || run.m > cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a run of {} tokens in buffers of {cap}", run.m),
            });
        }
        let st = run.stacks;
        if let Some((_, form)) = kernels::down(st.down.ty)
            && form != act_h.form()
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the down stack {} of {} reads {form:?} columns; the buffers hold {:?}",
                    st.down.name,
                    st.down.ty,
                    act_h.form()
                ),
            });
        }
        let stream = gpu.stream();
        let fault = gpu.layer_sink(run.layer)?;
        let slots = run.m * slots_per_col;
        gpu.enqueue_quantize_q8_1_cols(run.normed, act_x, run.m, run.layer)?;
        let words = || -> Result<GateUpAct<'_>, GpuError> {
            Ok(GateUpAct {
                wg: st.gate.words()?,
                wu: st.up.words()?,
                act: act_x,
                sel: run.sel,
                n_slots: slots,
                rows_per_expert: ff,
                slots_per_col,
                rule: run.rule,
            })
        };
        match kernels::gate_up(st.gate.ty) {
            Some(GateUpEntry::Q4k) => {
                self.quant
                    .enqueue_gate_up_q4k(stream, &words()?, fault, h)?
            }
            Some(GateUpEntry::Q5k) => {
                self.quant
                    .enqueue_gate_up_q5k(stream, &words()?, fault, h)?
            }
            Some(GateUpEntry::Mxfp4) => self.mxfp4.enqueue_gate_up(stream, &words()?, fault, h)?,
            Some(GateUpEntry::Nvfp4) => {
                let a = Nvfp4GateUp {
                    gate: st.gate.planes()?,
                    up: st.up.planes()?,
                    act: act_x,
                    sel: run.sel,
                    n_slots: slots,
                    rows_per_expert: ff,
                    slots_per_col,
                    rule: run.rule,
                };
                self.nvfp4.enqueue_gate_up(stream, &a, fault, h)?;
            }
            Some(
                GateUpEntry::Iq2xs | GateUpEntry::Iq3xxs | GateUpEntry::Iq3s | GateUpEntry::Iq4xs,
            ) => {
                let a = words()?;
                let iq = IqGateUp {
                    ty: st.gate.ty,
                    wg: a.wg,
                    wu: a.wu,
                    act: a.act,
                    sel: a.sel,
                    n_slots: a.n_slots,
                    rows_per_expert: a.rows_per_expert,
                    slots_per_col: a.slots_per_col,
                    rule: a.rule,
                };
                self.quant.enqueue_gate_up(stream, &iq, fault, h)?;
            }
            _ => return Err(st.gate.unrun()),
        }
        let q = QuantSel {
            x: h,
            cols: 0..slots,
            sel: run.sel,
            n_card: run.n_card,
        };
        match act_h {
            ActH::Q8Act(act) => gpu.q4k_sel().enqueue_quantize_sel(stream, &q, fault, act)?,
            ActH::Blocks32(act) => gpu.q5().enqueue_quantize_q8_sel(stream, &q, act, fault)?,
        }
        let (sel, rows_per_expert) = (run.sel, n);
        match (kernels::down(st.down.ty), &*act_h) {
            (Some((DownEntry::Q4k, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                gpu.q4k_sel().enqueue_gemv_q4k_sel(
                    stream,
                    st.down.words()?,
                    act,
                    sel,
                    slots,
                    rows_per_expert,
                    down,
                )?;
            }
            (Some((DownEntry::Q5k, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                let d = SelDown {
                    w: st.down.words()?,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.quant.enqueue_gemv_q5k_sel(stream, &d, fault, down)?;
            }
            (Some((DownEntry::Mxfp4, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                let d = SelDown {
                    w: st.down.words()?,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.mxfp4.enqueue_down(stream, &d, fault, down)?;
            }
            (Some((DownEntry::Nvfp4, DownAct::Q8Blocks32)), ActH::Blocks32(act)) => {
                let d = Nvfp4Down {
                    w: st.down.planes()?,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.nvfp4.enqueue_down(stream, &d, fault, down)?;
            }
            (Some((entry, DownAct::Q8Blocks32)), ActH::Blocks32(act)) => {
                let run = Sel32Run {
                    entry,
                    stack: &st.down,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.sel32().enqueue(stream, &run, fault, down)?;
            }
            (
                Some((
                    DownEntry::Iq2xs | DownEntry::Iq3xxs | DownEntry::Iq3s | DownEntry::Iq4xs,
                    DownAct::Q8Act,
                )),
                ActH::Q8Act(act),
            ) => {
                let d = SelDown {
                    w: st.down.words()?,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.quant
                    .enqueue_gemv_sel(stream, st.down.ty, &d, fault, down)?;
            }
            _ => return Err(st.down.unrun()),
        }
        Ok(())
    }
}

/// The sum a run ends with is over a slot count the sum has an entry for:
/// refused by name before the run launches anything.
fn check_sum(shape: LegShape) -> Result<(), GpuError> {
    if CardSlots::of(shape.slots_per_col).is_none() {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "the card sum: {} routed slots a token: the sum's entries are built for 6, 8 \
                 and 10",
                shape.slots_per_col
            ),
        });
    }
    Ok(())
}

/// The card slots' weighted sum of `down` (the run's rows, `m` tokens of
/// `shape.slots_per_col` slots) under the places `sel`, after the run's
/// launches: [`LegSum`] says what it reads and writes. `(layer, m, n_card)` are
/// the run's.
fn enqueue_sum(
    slots: &CardSlotKernels,
    gpu: &Gpu,
    shape: LegShape,
    (down, sel): (&DeviceBuffer<f32>, &DeviceBuffer<u32>),
    (layer, m, n_card): (usize, usize, usize),
    sum: LegSum<'_>,
) -> Result<(), GpuError> {
    let LegSum { w, acc } = sum;
    let a = CardAcc {
        down,
        w,
        sel,
        n: shape.n,
        m,
        n_card,
        n_used: shape.slots_per_col,
        w_pitch: shape.slots_per_col,
        fault: gpu.layer_sink(layer)?,
    };
    slots.enqueue_card_acc(gpu.stream(), &a, acc)
}

/// The modules a 32-value-block `_sel` runs on, borrowed from a leg.
struct Sel32<'a> {
    q51: &'a Q51SelKernels,
    q80: &'a Q80SelKernels,
    iq: &'a QuantSelKernels,
}

/// One 32-value-block `_sel` launch: slot `s` writes the `rows_per_expert`
/// rows of expert `sel[s]` of `stack` dotted with column `s` of `act`, to
/// `y[s · rows_per_expert ..]`. `entry` is the table's row for the stack's
/// type, [`kernels::down`]'s for a down and [`kernels::up`]'s for an up (one
/// entry: an up is a down-shaped `_sel`).
#[derive(Clone, Copy)]
struct Sel32Run<'a> {
    entry: DownEntry,
    stack: &'a LegStack<'a>,
    act: &'a Q8Blocks32,
    sel: &'a DeviceBuffer<u32>,
    n_slots: usize,
    rows_per_expert: usize,
}

impl Sel32<'_> {
    /// Enqueue the launch of `run`; an entry the table names that has no arm
    /// here is refused by name.
    fn enqueue(
        &self,
        stream: &CudaStream,
        run: &Sel32Run<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let Sel32Run {
            entry,
            stack,
            act,
            sel,
            n_slots,
            rows_per_expert,
        } = *run;
        match entry {
            DownEntry::Q5_1 => {
                let d = Q51SelDown {
                    w: stack.words()?,
                    act,
                    sel,
                    n_slots,
                    rows_per_expert,
                };
                self.q51.enqueue_gemv_q5_1_sel(stream, &d, fault, y)
            }
            DownEntry::Q8_0 => {
                let d = Q80SelDown {
                    w: stack.words()?,
                    act,
                    sel,
                    n_slots,
                    rows_per_expert,
                };
                self.q80.enqueue_gemv_q8_0_sel32(stream, &d, fault, y)
            }
            DownEntry::Iq4nl => {
                let d = IqDown {
                    w: stack.words()?,
                    act,
                    sel,
                    n_slots,
                    rows_per_expert,
                };
                self.iq.enqueue_down(stream, &d, fault, y)
            }
            _ => Err(stack.unrun()),
        }
    }
}

/// A layer's two resident routed stacks of an expert with no gate: the up and
/// the down, each of a type [`kernels::card_sel32_runs`] admits (not
/// necessarily the same one).
pub struct Relu2Stacks<'a> {
    up: LegStack<'a>,
    down: LegStack<'a>,
}

impl<'a> Relu2Stacks<'a> {
    /// The stacks named `[up, down]` in `w`, each a resident
    /// `DevWeight::KQuant` of a 32-value-block type the relu² form runs.
    /// Allocation-free on success.
    pub fn of(w: &'a Weights, names: [&'a str; 2]) -> Result<Relu2Stacks<'a>, GpuError> {
        const UP: &str = "a resident routed up stack of a 32-value-block type the card leg runs (`kernels::card_sel32_runs`)";
        const DOWN: &str = "a resident routed down stack of a 32-value-block type the card leg runs (`kernels::card_sel32_runs`)";
        let [up, down] = names;
        Ok(Relu2Stacks {
            up: LegStack::of(w, up, kernels::card_sel32_runs, |_| false, UP)?,
            down: LegStack::of(w, down, kernels::card_sel32_runs, |_| false, DOWN)?,
        })
    }

    /// The stacks hold `n_card` experts of `shape` each: the up's `ff` rows of
    /// `n` values, the down's `n` rows of `ff`.
    pub fn check(&self, n_card: usize, shape: LegShape) -> Result<(), GpuError> {
        let LegShape { n, ff, .. } = shape;
        for (st, rows, k) in [(&self.up, ff, n), (&self.down, n, ff)] {
            if st.rows() != n_card * rows || st.k != k {
                return Err(GpuError::Tensor {
                    what: WHAT,
                    name: st.name.to_owned(),
                    need: "the rows of the layer's card experts, of the shape's width",
                });
            }
        }
        Ok(())
    }
}

/// One relu² run's buffers for up to `shape.cap` tokens: the normed row's
/// 32-value q8_1 form once a slot, the up's rows, the card slots' q8_1 columns
/// of `max(h, 0)²` and the down's rows. `shape.n` and `shape.ff` are
/// multiples of 32.
pub struct Relu2Bufs {
    shape: LegShape,
    act_x: Q8Blocks32,
    h: DeviceBuffer<f32>,
    act_h: Q8Blocks32,
    down: DeviceBuffer<f32>,
}

impl Relu2Bufs {
    /// The buffers for runs of `shape`. Load-time or first-prompt only.
    pub fn new(stream: &CudaStream, shape: LegShape) -> Result<Relu2Bufs, GpuError> {
        let slots = shape.slots();
        Ok(Relu2Bufs {
            shape,
            act_x: Q8Blocks32::with_slots(stream, shape.n, slots)?,
            h: DeviceBuffer::zeroed(stream, slots * shape.ff)?,
            act_h: Q8Blocks32::with_slots(stream, shape.ff, slots)?,
            down: DeviceBuffer::zeroed(stream, slots * shape.n)?,
        })
    }

    /// The shape the buffers were made for.
    #[must_use]
    pub fn shape(&self) -> LegShape {
        self.shape
    }

    /// Device bytes of the buffers: [`relu2_leg_bytes`] of their shape, as
    /// allocated.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let q8 = |a: &Q8Blocks32| a.q.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes();
        q8(&self.act_x) + self.h.num_bytes() + q8(&self.act_h) + self.down.num_bytes()
    }

    /// The down rows of the last run: `n` values a slot, slot-major, the slots
    /// the run placed on the card (the rest are as an earlier run left them).
    #[must_use]
    pub fn down(&self) -> &DeviceBuffer<f32> {
        &self.down
    }
}

/// Device bytes a [`Relu2Bufs::new`] of `shape` allocates, counted before it
/// does: the normed row's q8_1 form and the card slots' (`cap · slots_per_col`
/// columns of `n` and of `ff`), the up's rows and the down's rows.
#[must_use]
pub fn relu2_leg_bytes(shape: LegShape) -> usize {
    let slots = shape.slots();
    let f = size_of::<f32>();
    q8_blocks32_bytes(slots, shape.n)
        + f * slots * shape.ff
        + q8_blocks32_bytes(slots, shape.ff)
        + f * slots * shape.n
}

/// What [`Relu2Leg::enqueue`] reads: layer `layer`'s `stacks`, `m` tokens'
/// normed rows `normed`, and the slots' places `sel` (`m · slots_per_col` ids;
/// an id below `n_card` is a card expert, the rest are left alone).
pub struct Relu2Run<'a> {
    pub layer: usize,
    pub stacks: &'a Relu2Stacks<'a>,
    pub normed: &'a DeviceBuffer<f32>,
    pub m: usize,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_card: usize,
}

/// The modules the relu² form's launches run on, beside the quantizers of
/// `Gpu::q5`.
pub struct Relu2Leg {
    q51: Q51SelKernels,
    q80: Q80SelKernels,
    iq: QuantSelKernels,
    slots: CardSlotKernels,
}

impl Relu2Leg {
    /// The modules on `gpu`. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<Relu2Leg, GpuError> {
        let (ctx, word) = (gpu.context(), gpu.fault_word());
        Ok(Relu2Leg {
            q51: Q51SelKernels::load(ctx, word)?,
            q80: Q80SelKernels::load(ctx, word)?,
            iq: QuantSelKernels::load(ctx, word)?,
            slots: CardSlotKernels::load(ctx)?,
        })
    }

    /// The modules the 32-value-block `_sel`s run on.
    fn sel32(&self) -> Sel32<'_> {
        Sel32 {
            q51: &self.q51,
            q80: &self.q80,
            iq: &self.iq,
        }
    }

    /// Enqueue on `gpu`'s stream the four launches (module doc) of `run` into
    /// `bufs`, and with `sum` the card slots' sum after them; `run.m` outside
    /// `1..=bufs.shape().cap`, or a sum over a slot count the sum has no entry
    /// for, is refused by name before the first launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        gpu: &Gpu,
        bufs: &mut Relu2Bufs,
        run: &Relu2Run<'_>,
        sum: Option<LegSum<'_>>,
    ) -> Result<(), GpuError> {
        let Relu2Bufs {
            shape,
            act_x,
            h,
            act_h,
            down,
        } = bufs;
        let LegShape {
            cap,
            slots_per_col,
            n,
            ff,
        } = *shape;
        if run.m == 0 || run.m > cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a run of {} tokens in buffers of {cap}", run.m),
            });
        }
        if sum.is_some() {
            check_sum(*shape)?;
        }
        let st = run.stacks;
        let entry = |stack: &LegStack<'_>| kernels::up(stack.ty).ok_or_else(|| stack.unrun());
        let (up_entry, down_entry) = (entry(&st.up)?, entry(&st.down)?);
        let stream = gpu.stream();
        let fault = gpu.layer_sink(run.layer)?;
        let slots = run.m * slots_per_col;
        let sel32 = self.sel32();
        let q = QuantSel {
            x: run.normed,
            cols: 0..slots,
            sel: run.sel,
            n_card: run.n_card,
        };
        gpu.q5()
            .enqueue_quantize_q8_rep(stream, &q, slots_per_col, act_x, fault)?;
        let up = Sel32Run {
            entry: up_entry,
            stack: &st.up,
            act: act_x,
            sel: run.sel,
            n_slots: slots,
            rows_per_expert: ff,
        };
        sel32.enqueue(stream, &up, fault, h)?;
        let q = QuantSel {
            x: h,
            cols: 0..slots,
            sel: run.sel,
            n_card: run.n_card,
        };
        gpu.q5()
            .enqueue_quantize_q8_sel_relu_sqr(stream, &q, act_h, fault)?;
        let dn = Sel32Run {
            entry: down_entry,
            stack: &st.down,
            act: act_h,
            sel: run.sel,
            n_slots: slots,
            rows_per_expert: n,
        };
        sel32.enqueue(stream, &dn, fault, down)?;
        let Some(sum) = sum else {
            return Ok(());
        };
        enqueue_sum(
            &self.slots,
            gpu,
            *shape,
            (down, run.sel),
            (run.layer, run.m, run.n_card),
            sum,
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::hybrid::HOST;
    use crate::q4k_sel::QuantSel;
    use gguf::Split;
    use gguf::write::{Layout, TensorDecl, Writer};
    use model::placement::CardFormat;
    use runtime::words::stream_words;

    /// The normed width: the gate·up stacks' K (eight super-blocks).
    const N: usize = 2048;
    /// The routed width: a gate·up slot's rows and the down stack's K (four
    /// super-blocks).
    const FF: usize = 1024;
    /// Experts the stacks hold.
    const N_CARD: usize = 4;
    /// Slots a token carries.
    const SPC: usize = 8;
    /// Tokens the leg's buffers are sized for.
    const CAP: usize = 3;
    /// What a slot the run leaves alone still holds.
    const SENT: f32 = 1.0e30;
    /// Three tokens' places: duplicates, every card expert, and slots the
    /// card does not serve.
    const SEL: [u32; CAP * SPC] = [
        0, 1, 2, 3, 0, 1, HOST, 2, 3, 3, 1, 0, HOST, 2, 1, 0, 2, HOST, 0, 1, 3, 2, 1, 0,
    ];
    /// The shape of the small tests' buffers.
    const SHAPE: LegShape = LegShape {
        cap: CAP,
        slots_per_col: SPC,
        n: N,
        ff: FF,
    };

    /// Next value of a xorshift64 stream.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// `rows` rows of `k` values of `ty` in the file's block bytes: every
    /// byte random but each block's scales, which are finite and small, so no
    /// dot is NaN or infinite — the f16 `d` (and `dmin` of Q4_K and Q5_K, `m`
    /// of Q5_1; Q6_K's last in its block) or MXFP4's E8M0 byte. Every other byte of these types is a
    /// valid code, so any draw is a valid block.
    pub(crate) fn stack_bytes(ty: GgmlType, rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let block = usize::try_from(ty.type_size().expect("a block size")).expect("a size");
        let width = usize::try_from(ty.blck_size().expect("a block width")).expect("a width");
        let f16s = if matches!(ty, GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q5_1) {
            2
        } else {
            1
        };
        let mut state = seed;
        let mut out = vec![0u8; rows * (k / width) * block];
        for b in out.chunks_exact_mut(block) {
            for chunk in b.chunks_mut(8) {
                let v = next(&mut state).to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
            if ty == GgmlType::MXFP4 {
                b[0] = 118 + u8::try_from(next(&mut state) % 4).expect("a byte");
                continue;
            }
            // Q6_K's `d` closes its block; every other type's scales open it.
            let at = if ty == GgmlType::Q6_K { block - 2 } else { 0 };
            for scale in b[at..at + 2 * f16s].chunks_mut(2) {
                let (exp, mant) = (5 + next(&mut state) % 4, next(&mut state) & 0x3ff);
                let half = u16::try_from(exp << 10 | mant).expect("an f16");
                scale.copy_from_slice(&half.to_le_bytes());
            }
        }
        out
    }

    /// The little-endian words of a byte stream of `rows` rows as the card
    /// holds it (`runtime::words::stream_words`: zero-padded to whole words a
    /// row).
    pub(crate) fn words_of(bytes: &[u8], rows: usize) -> Vec<u32> {
        let total = stream_words(bytes.len() as u64, rows as u64).expect("a stack of rows");
        let mut w = vec![0u32; usize::try_from(total).expect("a word count")];
        for (word, c) in w.iter_mut().zip(bytes.chunks(4)) {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            *word = u32::from_le_bytes(b);
        }
        w
    }

    /// The loader keeps a tensor of `ty` as the file's rows.
    fn loads_as_file_rows(ty: GgmlType) -> bool {
        CardFormat::of(ty) == Some(CardFormat::KQuant)
    }

    /// The resident name of the `role` stack of `ty` that [`weights`] files:
    /// the file's own tensor for the types the loader keeps as the file's
    /// rows, a derived weight for the rest.
    fn stack_name(role: &str, ty: GgmlType) -> String {
        if loads_as_file_rows(ty) {
            format!("{role}_{ty}")
        } else {
            format!("derived.{role}_{ty}")
        }
    }

    /// The stacks `<role>_<ty>` of each type in `tys`, one for each role the
    /// leg runs the type in — gate and up `N_CARD · ff` rows of `n`, the down
    /// `N_CARD · n` rows of `ff` — resident in a [`Weights`] as
    /// `DevWeight::KQuant`, named by [`stack_name`]. The types the loader
    /// keeps as the file's rows (Q4_K, Q5_K) come through a file written for
    /// them, so `tys` holds one; the other types' rows are the same word
    /// stream, filed as derived weights.
    fn weights(gpu: &Gpu, shape: LegShape, tys: &[GgmlType]) -> Weights {
        weights_for(gpu, shape, tys, |role, ty| match role {
            "down" => kernels::card_down_runs(ty),
            _ => kernels::card_gate_up_runs(ty),
        })
    }

    /// [`weights`] with the roles each type is filed in chosen by `admits`
    /// (`"gate"`, `"up"` or `"down"`, and the type).
    fn weights_for(
        gpu: &Gpu,
        shape: LegShape,
        tys: &[GgmlType],
        admits: impl Fn(&str, GgmlType) -> bool,
    ) -> Weights {
        let LegShape { n, ff, .. } = shape;
        let mut decls = Vec::new();
        let mut data = Vec::new();
        let mut derived = Vec::new();
        for (i, &ty) in tys.iter().enumerate() {
            for (role, rows, k) in [
                ("gate", N_CARD * ff, n),
                ("up", N_CARD * ff, n),
                ("down", N_CARD * n, ff),
            ] {
                if !admits(role, ty) {
                    continue;
                }
                let seed = 0x9e37_79b9_7f4a_7c15 ^ ((i as u64) << 8 | role.len() as u64);
                let bytes = stack_bytes(ty, rows, k, seed);
                let name = stack_name(role, ty);
                if !loads_as_file_rows(ty) {
                    derived.push((name, ty, rows, k, bytes));
                    continue;
                }
                decls.push(TensorDecl {
                    name,
                    dims: vec![k as u64, rows as u64],
                    type_id: ty.as_u32(),
                    nbytes: bytes.len() as u64,
                });
                data.push(bytes);
            }
        }
        // One file per call: the tests run on parallel threads, and a second
        // File::create of a path the first test still maps is a SIGBUS.
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("bloomery-card-leg-{}-{call}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("a temp dir");
        let path = dir.join("stacks.gguf");
        let layout = Layout::new(&[], decls.clone()).expect("a layout");
        let file = std::io::BufWriter::new(std::fs::File::create(&path).expect("the file"));
        let mut w = Writer::new(file, layout).expect("a writer");
        for (d, bytes) in decls.iter().zip(&data) {
            w.tensor(&d.name, bytes).expect("a tensor");
        }
        w.finish().expect("a finished file");
        let split = Split::open(&path).expect("the file opens");
        let mut weights = Weights::load_where(gpu.stream(), &split, |_| true).expect("resident");
        let _ = std::fs::remove_dir_all(&dir);
        for (name, ty, rows, k, bytes) in derived {
            let words = words_of(&bytes, rows);
            let w = DeviceTensor::upload(gpu.stream(), &words, rows, words.len() / rows)
                .expect("the stack's words");
            weights
                .insert_derived(name, DevWeight::KQuant { ty, w, k })
                .expect("a free derived name");
        }
        weights
    }

    /// The resident words of the stack `name`.
    fn words<'a>(w: &'a Weights, name: &str) -> &'a DeviceTensor<u32> {
        match w.get(name) {
            Some(DevWeight::KQuant { w, .. }) => w,
            _ => panic!("{name} is not a resident K-quant"),
        }
    }

    /// The leg's K-quant arms against the launches the table names, run by
    /// hand on buffers of exactly the run's slots: for each type, a run of
    /// every token its buffers hold and of fewer. The down of a run of fewer
    /// reads an activation of more columns than the run has slots. A slot the
    /// card serves must hold a finite value the run wrote, a slot it does not
    /// must still hold what it held, and the rows are the hand launches' bit
    /// for bit.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_runs_the_kquant_arms_as_their_sel_launches() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let kq = QuantSelKernels::load(gpu.context(), gpu.fault_word()).expect("the kquant module");
        let tys = [GgmlType::Q4_K, GgmlType::Q5_K];
        let w = weights(&gpu, SHAPE, &tys);
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let shape = SHAPE;
        let rule = Act::SiluMul;
        for ty in tys {
            let names = [
                format!("gate_{ty}"),
                format!("up_{ty}"),
                format!("down_{ty}"),
            ];
            let stacks = LegStacks::of(&w, [&names[0], &names[1], &names[2]]).expect("the stacks");
            stacks.check(N_CARD, shape).expect("the stacks' shape");
            let mut bufs = LegBufs::new(stream, shape).expect("the leg's buffers");
            for m in [CAP, CAP - 1] {
                let case = format!("{ty}, {m} tokens");
                let slots = m * SPC;
                bufs.down
                    .copy_from_host(stream, &vec![SENT; CAP * SPC * N])
                    .expect("the sentinel");
                let run = LegRun {
                    layer: 0,
                    stacks: &stacks,
                    normed: &normed,
                    m,
                    sel: &sel,
                    n_card: N_CARD,
                    rule,
                };
                leg.enqueue(&gpu, &mut bufs, &run, None)
                    .unwrap_or_else(|e| panic!("{case}: the leg: {e}"));

                let mut act_x = Q8Act::with_tier_cols(stream, m, N).expect("act_x");
                let mut h = DeviceBuffer::<f32>::zeroed(stream, slots * FF).expect("h");
                let mut act_h = Q8Act::with_slots(stream, slots, FF).expect("act_h");
                let mut want =
                    DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("the rows");
                let sink = gpu.layer_sink(0).expect("a sink");
                gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, m, 0)
                    .expect("q8_1");
                let a = GateUpAct {
                    wg: words(&w, &names[0]),
                    wu: words(&w, &names[1]),
                    act: &act_x,
                    sel: &sel,
                    n_slots: slots,
                    rows_per_expert: FF,
                    slots_per_col: SPC,
                    rule,
                };
                match ty {
                    GgmlType::Q4_K => kq.enqueue_gate_up_q4k(stream, &a, sink, &mut h),
                    _ => kq.enqueue_gate_up_q5k(stream, &a, sink, &mut h),
                }
                .unwrap_or_else(|e| panic!("{case}: the gate·up: {e}"));
                let q = QuantSel {
                    x: &h,
                    cols: 0..slots,
                    sel: &sel,
                    n_card: N_CARD,
                };
                gpu.q4k_sel()
                    .enqueue_quantize_sel(stream, &q, sink, &mut act_h)
                    .unwrap_or_else(|e| panic!("{case}: the quantizer: {e}"));
                let d = SelDown {
                    w: words(&w, &names[2]),
                    act: &act_h,
                    sel: &sel,
                    n_slots: slots,
                    rows_per_expert: N,
                };
                match ty {
                    GgmlType::Q4_K => gpu
                        .q4k_sel()
                        .enqueue_gemv_q4k_sel(stream, d.w, d.act, d.sel, slots, N, &mut want),
                    _ => kq.enqueue_gemv_q5k_sel(stream, &d, sink, &mut want),
                }
                .unwrap_or_else(|e| panic!("{case}: the down: {e}"));

                let want = want.to_host_vec(stream).expect("the hand rows");
                let got = bufs.down.to_host_vec(stream).expect("the leg's rows");
                assert!(
                    gpu.take_fault().expect("the fault word").is_none(),
                    "{case}"
                );
                assert_hand_rows(&case, &got, &want);
            }
        }
    }

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_gate_up_sums_fault_before_activation() {
        use crate::fault::{Fault, FaultSite, LAYER_NONE};
        use crate::iq_sel::IqGateUp;
        use crate::kquant::{
            EntryPlanes, GateUpAt, KquantTileKernels, TileGather, TileTable, TiledGateUpAct,
        };
        const K: usize = 256;
        const ROWS: usize = 8;
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let sink = gpu.unlabelled_sink();
        let kernels = QuantSelKernels::load(gpu.context(), gpu.fault_word()).expect("kernels");
        let tiles = KquantTileKernels::load(gpu.context(), gpu.fault_word()).expect("tiles");
        let x = DeviceBuffer::from_host(stream, &[1e38f32; K]).expect("x");
        let mut act = Q8Act::with_k(stream, 1, K).expect("act");
        gpu.enqueue_quantize_q8_1(&x, &mut act).expect("quantize");
        let sel = DeviceBuffer::from_host(stream, &[0u32]).expect("sel");
        let start = DeviceBuffer::from_host(stream, &[0u32, 1]).expect("start");
        let mut table_tiles =
            DeviceBuffer::zeroed(stream, crate::q4k_sel::tile_cap(1, 1) + 1).expect("tiles");
        gpu.q4k_sel()
            .enqueue_grouped_tiles(stream, &start, 1, 1, sink, &mut table_tiles)
            .expect("tile table");
        let table = || TileTable {
            order: &sel,
            start: &start,
            tiles: &table_tiles,
            n_experts: 1,
            n_slots: 1,
        };
        let mut planes = EntryPlanes::new(stream, 1, K).expect("planes");
        tiles
            .enqueue_card_gather(
                stream,
                &TileGather {
                    x: &act,
                    cols: 1,
                    slots_per_col: 1,
                    table: table(),
                },
                sink,
                &mut planes,
            )
            .expect("gather");
        assert_eq!(gpu.take_fault().expect("input fault"), None);
        let mut failures = Vec::new();
        for ty in [
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q8_0,
            GgmlType::IQ2_XS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
        ] {
            let mut bytes = stack_bytes(ty, ROWS, K, 42);
            let bb = ty.type_size().expect("block bytes") as usize;
            for block in bytes.chunks_exact_mut(bb) {
                block[..2].copy_from_slice(&0x7bffu16.to_le_bytes());
            }
            let up = words_of(&bytes, ROWS);
            let gate = vec![0u32; up.len()];
            let wg = DeviceTensor::upload(stream, &gate, ROWS, gate.len() / ROWS).expect("gate");
            let wu = DeviceTensor::upload(stream, &up, ROWS, up.len() / ROWS).expect("up");
            let arena = DeviceBuffer::from_host(stream, &[gate, up].concat()).expect("arena");
            let offs = DeviceBuffer::from_host(stream, &[0u64, (bytes.len() / 4) as u64, 0])
                .expect("offsets");
            let modes: &[&str] = match ty {
                GgmlType::Q4_K => &["sel", "sel_gate", "arena", "tiles"],
                GgmlType::Q5_K => &["sel", "arena", "tiles"],
                _ => &["sel"],
            };
            for &mode in modes {
                let (gate, up) = if mode == "sel_gate" {
                    (&wu, &wg)
                } else {
                    (&wg, &wu)
                };
                let mut h = DeviceBuffer::zeroed(stream, ROWS).expect("h");
                let rule = Act::SwigluClamp { limit: 7.0 };
                let a = GateUpAct {
                    wg: gate,
                    wu: up,
                    act: &act,
                    sel: &sel,
                    n_slots: 1,
                    rows_per_expert: ROWS,
                    slots_per_col: 1,
                    rule,
                };
                match mode {
                    "arena" => {
                        let a = GateUpAt {
                            arena: &arena,
                            offs: &offs,
                            act: &act,
                            n_slots: 1,
                            rows_per_expert: ROWS,
                            slots_per_col: 1,
                            rule,
                        };
                        match ty {
                            GgmlType::Q4_K => {
                                kernels.enqueue_gate_up_q4k_at(stream, &a, sink, &mut h)
                            }
                            _ => kernels.enqueue_gate_up_q5k_at(stream, &a, sink, &mut h),
                        }
                    }
                    "tiles" => tiles.enqueue_gate_up_tiles(
                        stream,
                        ty,
                        &TiledGateUpAct {
                            wg: &wg,
                            wu: &wu,
                            act: &planes,
                            table: table(),
                            rows_per_expert: ROWS,
                            rule,
                        },
                        sink,
                        &mut h,
                    ),
                    _ => match ty {
                        GgmlType::Q4_K => kernels.enqueue_gate_up_q4k(stream, &a, sink, &mut h),
                        GgmlType::Q5_K => kernels.enqueue_gate_up_q5k(stream, &a, sink, &mut h),
                        GgmlType::Q8_0 => kernels.enqueue_gate_up_q8_0(stream, &a, sink, &mut h),
                        _ => kernels.enqueue_gate_up(
                            stream,
                            &IqGateUp {
                                ty,
                                wg: &wg,
                                wu: &wu,
                                act: &act,
                                sel: &sel,
                                n_slots: 1,
                                rows_per_expert: ROWS,
                                slots_per_col: 1,
                                rule,
                            },
                            sink,
                            &mut h,
                        ),
                    },
                }
                .expect("gate up");
                let fault = gpu.take_fault().expect("fault");
                let rows = h.to_host_vec(stream).expect("rows");
                let ok = fault == Some(Fault::at(LAYER_NONE, FaultSite::F32Product))
                    && rows.iter().all(|v| v.is_nan());
                println!(
                    "{ty:?} {mode}: fault={fault:?} nan={} {ok}",
                    rows.iter().filter(|v| v.is_nan()).count()
                );
                if !ok {
                    failures.push(format!("{ty:?} {mode}: {fault:?} {rows:?}"));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// An NVFP4 stack of `N_CARD` experts of `rows` rows of `k` values in its
    /// planes, random codes under small UE4M3 scale bytes, and a scale an
    /// expert that differs between experts, filed as derived weight `name`.
    fn nvfp4_stack(gpu: &Gpu, w: &mut Weights, name: &str, rows: usize, k: usize, seed: u64) {
        let mut state = seed;
        let mut bytes = vec![0u8; N_CARD * rows * k / 64 * 36];
        for b in bytes.as_chunks_mut::<36>().0 {
            for (i, v) in b.iter_mut().enumerate() {
                let r = u8::try_from(next(&mut state) & 0xff).expect("a byte");
                *v = if i < 4 { 0x30 | (r & 0x0f) } else { r };
            }
        }
        let p = crate::mxfp4::repack::<crate::mxfp4::Nvfp4>(&bytes, N_CARD * rows, k)
            .expect("the planes");
        let scales: Vec<f32> = (0..N_CARD).map(|e| 0.75 + 0.125 * e as f32).collect();
        let stream = gpu.stream();
        let stack = DevWeight::Nvfp4Stack {
            qs: DeviceTensor::upload(stream, &p.qs, N_CARD * rows, k / 8).expect("qs"),
            e: DeviceTensor::upload(stream, &p.e, N_CARD * rows, k / 64).expect("e"),
            scales: DeviceTensor::upload(stream, &scales, N_CARD, 1).expect("the scales"),
            k,
        };
        w.insert_derived(name.to_owned(), stack)
            .expect("a free derived name");
    }

    /// The leg's NVFP4 arms against the launches the table names, run by hand
    /// on buffers of exactly the run's slots: the gate·up over a `Q8Act`, the
    /// down over the 32-value form the buffers hold for it, for a run of every
    /// token the buffers hold and of fewer. A slot the card serves holds a
    /// finite value the run wrote, a host slot what it held, and the rows are
    /// the hand launches' bit for bit; a stack the leg holds in planes is
    /// refused by the relu² form by name.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_runs_the_nvfp4_arms_as_their_sel_launches() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let nv = Nvfp4SelKernels::load(gpu.context(), gpu.fault_word()).expect("the nvfp4 module");
        let mut w = weights(&gpu, SHAPE, &[]);
        let names = ["derived.gate_nv", "derived.up_nv", "derived.down_nv"];
        nvfp4_stack(&gpu, &mut w, names[0], FF, N, 0x11);
        nvfp4_stack(&gpu, &mut w, names[1], FF, N, 0x22);
        nvfp4_stack(&gpu, &mut w, names[2], N, FF, 0x33);
        let stacks = LegStacks::of(&w, names).expect("the stacks");
        stacks.check(N_CARD, SHAPE).expect("the stacks' shape");
        assert!(Relu2Stacks::of(&w, [names[1], names[2]]).is_err());
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let rule = Act::SwigluClamp { limit: 7.0 };
        let mut bufs =
            LegBufs::for_down(stream, SHAPE, DownAct::Q8Blocks32).expect("the leg's buffers");
        let plane = |name: &str| w.nvfp4_stack(WHAT, name).expect("a resident stack");
        for m in [CAP, CAP - 1] {
            let case = format!("NVFP4, {m} tokens");
            let slots = m * SPC;
            bufs.down
                .copy_from_host(stream, &vec![SENT; CAP * SPC * N])
                .expect("the sentinel");
            let run = LegRun {
                layer: 0,
                stacks: &stacks,
                normed: &normed,
                m,
                sel: &sel,
                n_card: N_CARD,
                rule,
            };
            leg.enqueue(&gpu, &mut bufs, &run, None)
                .unwrap_or_else(|e| panic!("{case}: the leg: {e}"));

            let mut act_x = Q8Act::with_tier_cols(stream, m, N).expect("act_x");
            let mut h = DeviceBuffer::<f32>::zeroed(stream, slots * FF).expect("h");
            let mut act_h = Q8Blocks32::with_slots(stream, FF, slots).expect("act_h");
            let mut want =
                DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("the rows");
            let sink = gpu.layer_sink(0).expect("a sink");
            gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, m, 0)
                .expect("q8_1");
            let a = Nvfp4GateUp {
                gate: plane(names[0]),
                up: plane(names[1]),
                act: &act_x,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: FF,
                slots_per_col: SPC,
                rule,
            };
            nv.enqueue_gate_up(stream, &a, sink, &mut h)
                .unwrap_or_else(|e| panic!("{case}: the gate·up: {e}"));
            let q = QuantSel {
                x: &h,
                cols: 0..slots,
                sel: &sel,
                n_card: N_CARD,
            };
            gpu.q5()
                .enqueue_quantize_q8_sel(stream, &q, &mut act_h, sink)
                .unwrap_or_else(|e| panic!("{case}: the quantizer: {e}"));
            let d = Nvfp4Down {
                w: plane(names[2]),
                act: &act_h,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: N,
            };
            nv.enqueue_down(stream, &d, sink, &mut want)
                .unwrap_or_else(|e| panic!("{case}: the down: {e}"));

            let want = want.to_host_vec(stream).expect("the hand rows");
            let got = bufs.down.to_host_vec(stream).expect("the leg's rows");
            assert!(
                gpu.take_fault().expect("the fault word").is_none(),
                "{case}"
            );
            assert_hand_rows(&case, &got, &want);
        }
    }

    /// Tokens the wide test's buffers hold: more than a warp's 32 columns.
    const WIDE_CAP: usize = 40;
    /// The wide test's normed width: ten super-blocks, so the last group of
    /// four in each column's q8_1 planes is partial.
    const WIDE_N: usize = 2560;
    /// The wide test's routed width: five super-blocks and forty 32-value
    /// blocks, a partial last group in both forms of the down's activation.
    const WIDE_FF: usize = 1280;
    /// The shape of the wide test's buffers.
    const WIDE_SHAPE: LegShape = LegShape {
        cap: WIDE_CAP,
        slots_per_col: SPC,
        n: WIDE_N,
        ff: WIDE_FF,
    };
    /// The tokens of the wide test's runs: one, one past a group of eight
    /// columns, and the buffers' cap.
    const WIDE_RUNS: [usize; 3] = [1, 9, WIDE_CAP];
    /// The wide test's cases, `(gate·up type, down type)`: the seven types of
    /// `kernels::card_gate_up_runs` each as a gate·up beside a down of its own
    /// type, then the three 32-value downs (`kernels::card_down_runs`) beside
    /// a gate·up that ran before.
    const WIDE_CASES: [(GgmlType, GgmlType); 10] = [
        (GgmlType::Q4_K, GgmlType::Q4_K),
        (GgmlType::Q5_K, GgmlType::Q5_K),
        (GgmlType::MXFP4, GgmlType::MXFP4),
        (GgmlType::IQ2_XS, GgmlType::IQ2_XS),
        (GgmlType::IQ3_XXS, GgmlType::IQ3_XXS),
        (GgmlType::IQ3_S, GgmlType::IQ3_S),
        (GgmlType::IQ4_XS, GgmlType::IQ4_XS),
        (GgmlType::Q4_K, GgmlType::Q5_1),
        (GgmlType::Q5_K, GgmlType::Q8_0),
        (GgmlType::IQ4_XS, GgmlType::IQ4_NL),
    ];

    /// Token `t`'s normed row: a function of `t` alone, so a column holds the
    /// same values at every `m`.
    fn wide_row(t: usize) -> Vec<f32> {
        row_of(t, WIDE_N)
    }

    /// Token `t`'s normed row of `n` values, a function of `t` alone.
    fn row_of(t: usize, n: usize) -> Vec<f32> {
        let mut state = (t as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        (0..n)
            .map(|_| ((next(&mut state) >> 33) % 2001) as f32 / 1000.0 - 1.0)
            .collect()
    }

    /// Token `t`'s places, a function of `t` alone: slot `j` serves expert
    /// `(j + t) % N_CARD`, but the slot `(3t + 1) % SPC`, which the card does
    /// not serve. A run of any `m` therefore mixes duplicates (seven expert
    /// slots of four experts a token), every card expert and host slots.
    fn wide_places(t: usize) -> [u32; SPC] {
        places_of(t)
    }

    /// [`wide_places`] for tokens of `S` slots.
    fn places_of<const S: usize>(t: usize) -> [u32; S] {
        std::array::from_fn(|j| {
            if j == (3 * t + 1) % S {
                HOST
            } else {
                ((j + t) % N_CARD) as u32
            }
        })
    }

    /// `sel` repeats a card expert, names every card expert, and holds a slot
    /// the card does not serve.
    fn mixes(sel: &[u32]) -> bool {
        let mut seen = [0usize; N_CARD];
        for &place in sel {
            if let Some(count) = seen.get_mut(place as usize) {
                *count += 1;
            }
        }
        seen.iter().all(|&c| c > 0) && seen.iter().any(|&c| c > 1) && sel.contains(&HOST)
    }

    /// What a run's rows of one slot must hold: a host slot's are the
    /// sentinel, a card slot's are finite, not the sentinel and not all zero.
    fn slot_state(rows: &[f32], place: u32) -> Result<(), &'static str> {
        if place == HOST {
            if rows.iter().all(|v| *v == SENT) {
                return Ok(());
            }
            return Err("a host slot was written");
        }
        if !rows.iter().all(|v| v.is_finite() && *v != SENT) {
            return Err("a card slot holds no finite row the run wrote");
        }
        if rows.iter().all(|v| *v == 0.0) {
            return Err("a card slot's rows are all zero");
        }
        Ok(())
    }

    fn assert_hand_rows(case: &str, got: &[f32], want: &[f32]) {
        for (s, &place) in SEL[..want.len() / N].iter().enumerate() {
            let rows = s * N..(s + 1) * N;
            slot_state(&got[rows.clone()], place)
                .unwrap_or_else(|why| panic!("{case}: slot {s} (place {place}): {why}"));
            if let Some(r) = rows
                .clone()
                .find(|&r| got[r].to_bits() != want[r].to_bits())
            {
                panic!(
                    "{case}: slot {s} row {}: leg {:#010x}, hand {:#010x}",
                    r - rows.start,
                    got[r].to_bits(),
                    want[r].to_bits()
                );
            }
        }
        assert!(
            got[want.len()..].iter().all(|v| *v == SENT),
            "{case}: the leg wrote past its run's slots"
        );
    }

    /// The down rows of a run of the first `m` tokens of `normed` against
    /// `sel` in `bufs`, filled with the sentinel before it, synchronized and
    /// fault-free.
    fn leg_rows(
        gpu: &Gpu,
        leg: &CardLeg,
        bufs: &mut LegBufs,
        stacks: &LegStacks<'_>,
        normed: &DeviceBuffer<f32>,
        sel: &DeviceBuffer<u32>,
        m: usize,
    ) -> Vec<f32> {
        let stream = gpu.stream();
        let LegShape {
            cap,
            slots_per_col,
            n,
            ..
        } = bufs.shape();
        bufs.down
            .copy_from_host(stream, &vec![SENT; cap * slots_per_col * n])
            .expect("the sentinel");
        let run = LegRun {
            layer: 0,
            stacks,
            normed,
            m,
            sel,
            n_card: N_CARD,
            rule: Act::SiluMul,
        };
        leg.enqueue(gpu, bufs, &run, None)
            .unwrap_or_else(|e| panic!("a run of {m} tokens: the leg: {e}"));
        let rows = bufs.down.to_host_vec(stream).expect("the leg's rows");
        assert!(
            gpu.take_fault().expect("the fault word").is_none(),
            "a run of {m} tokens raised a fault"
        );
        rows
    }

    /// A slot's rows are what a one-token run writes for it, whatever `m`
    /// (the module doc), for every type [`kernels::card_gate_up_runs`] and
    /// [`kernels::card_down_runs`] admit, in ten cases (`WIDE_CASES`). Runs of
    /// `m` in `WIDE_RUNS` tokens of eight slots go through fresh buffers of
    /// `WIDE_CAP` tokens, so the run's columns beyond the eighth, and beyond a
    /// warp's 32, are read through the leg's own quantize steps and buffers;
    /// the places and rows are functions of the token, so the same token holds
    /// the same slots in every run. The oracle is the leg's own run of that one
    /// token in buffers of one token. Each card slot's rows must equal its
    /// oracle's bit for bit, a host slot must keep its sentinel, and nothing
    /// past the run's slots may be written. Every case and `m` is judged, and
    /// the failures are listed together, each naming the slot and its column.
    ///
    /// The K-quant test above holds the leg's Q4_K and Q5_K arms to the launches
    /// the table names at one `m`; this one holds each arm's slots to the
    /// one-token run across `m`.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_writes_each_slots_one_token_rows_at_any_m() {
        let ids = |tys: Vec<GgmlType>| {
            let mut ids: Vec<u32> = tys.into_iter().map(GgmlType::as_u32).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        let admitted = |runs: fn(GgmlType) -> bool| {
            ids((0..64)
                .map(GgmlType::from_u32)
                .filter(|&ty| runs(ty))
                .collect())
        };
        assert_eq!(
            ids(WIDE_CASES.iter().map(|c| c.0).collect()),
            admitted(kernels::card_gate_up_runs),
            "the cases' gate·up types are not the types the leg admits"
        );
        assert_eq!(
            ids(WIDE_CASES.iter().map(|c| c.1).collect()),
            admitted(kernels::card_down_runs),
            "the cases' down types are not the types the leg admits"
        );
        let sel_all: Vec<u32> = (0..WIDE_CAP).flat_map(wide_places).collect();
        for m in WIDE_RUNS {
            assert!(
                mixes(&sel_all[..m * SPC]),
                "the places of {m} tokens must mix duplicates, every card expert and host slots"
            );
        }

        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let mut tys: Vec<GgmlType> = WIDE_CASES.iter().flat_map(|&(g, d)| [g, d]).collect();
        tys.sort_by_key(|ty| ty.as_u32());
        tys.dedup();
        let w = weights(&gpu, WIDE_SHAPE, &tys);
        let host_normed: Vec<f32> = (0..WIDE_CAP).flat_map(wide_row).collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &sel_all).expect("the places");
        let one = LegShape {
            cap: 1,
            ..WIDE_SHAPE
        };

        let mut failures = Vec::new();
        for (gu, dn) in WIDE_CASES {
            let names = [
                stack_name("gate", gu),
                stack_name("up", gu),
                stack_name("down", dn),
            ];
            let stacks = LegStacks::of(&w, [&names[0], &names[1], &names[2]]).expect("the stacks");
            stacks.check(N_CARD, WIDE_SHAPE).expect("the stacks' shape");
            let form = kernels::down(dn).expect("the down's entry").1;
            let mut one_bufs = LegBufs::for_down(stream, one, form).expect("one token's buffers");
            let oracle: Vec<Vec<f32>> = (0..WIDE_CAP)
                .map(|t| {
                    let row = DeviceBuffer::from_host(stream, &wide_row(t)).expect("a normed row");
                    let places = wide_places(t);
                    let sel_t = DeviceBuffer::from_host(stream, &places).expect("a token's places");
                    let rows = leg_rows(&gpu, &leg, &mut one_bufs, &stacks, &row, &sel_t, 1);
                    for (j, &place) in places.iter().enumerate() {
                        if let Err(why) = slot_state(&rows[j * WIDE_N..(j + 1) * WIDE_N], place) {
                            panic!("{gu}/{dn}: the one-token run of token {t}, slot {j}: {why}");
                        }
                    }
                    rows
                })
                .collect();

            for m in WIDE_RUNS {
                let mut bufs = LegBufs::for_down(stream, WIDE_SHAPE, form).expect("the buffers");
                let got = leg_rows(&gpu, &leg, &mut bufs, &stacks, &normed, &sel, m);
                let slots = m * SPC;
                let mut bad = Vec::new();
                for (s, &place) in sel_all[..slots].iter().enumerate() {
                    let (token, j) = (s / SPC, s % SPC);
                    let got_rows = &got[s * WIDE_N..(s + 1) * WIDE_N];
                    let want_rows = &oracle[token][j * WIDE_N..(j + 1) * WIDE_N];
                    let first_diff = (0..WIDE_N)
                        .find(|&r| got_rows[r].to_bits() != want_rows[r].to_bits())
                        .map(|r| {
                            format!(
                                "row {r}: {:#010x}, the one-token run's {:#010x}",
                                got_rows[r].to_bits(),
                                want_rows[r].to_bits()
                            )
                        });
                    let why = match slot_state(got_rows, place) {
                        Err(why) => Some(why.to_string()),
                        Ok(()) => first_diff,
                    };
                    if let Some(why) = why {
                        let at = match place {
                            HOST => "host".to_string(),
                            expert => format!("expert {expert}"),
                        };
                        bad.push(format!("slot {s} (column {token}, {at}): {why}"));
                    }
                }
                if got[slots * WIDE_N..].iter().any(|v| *v != SENT) {
                    bad.push("the run wrote past its slots".to_string());
                }
                let verdict = if bad.is_empty() { "ok" } else { "FAIL" };
                println!("leg_wide[gate·up {gu}, down {dn}, m = {m}] {verdict}");
                if let Some(first) = bad.first() {
                    failures.push(format!(
                        "gate·up {gu}, down {dn}, m = {m}: {} of {slots} slots; first: {first}",
                        bad.len()
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    /// The buffers of each activation form hold the bytes [`leg_bytes_for`]
    /// counts for their shape.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_buffers_hold_the_counted_bytes_of_each_form() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let shape = SHAPE;
        for form in [DownAct::Q8Act, DownAct::Q8Blocks32] {
            let bufs = LegBufs::for_down(gpu.stream(), shape, form).expect("the buffers");
            assert_eq!(bufs.bytes(), leg_bytes_for(shape, form), "{form:?}");
        }
        let bufs = LegBufs::new(gpu.stream(), shape).expect("the buffers");
        assert_eq!(bufs.bytes(), leg_bytes(shape));
    }

    /// A run whose down entry reads the other activation form than the
    /// buffers hold is refused by name before the first launch (the down's
    /// rows stay as they were), and [`CardLeg::enqueue_rows`] writes the rows
    /// [`CardLeg::enqueue`] writes into the caller's buffer, leaving the
    /// buffers' own as they were.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_leg_refuses_the_other_form_and_writes_the_callers_rows() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = CardLeg::load(&gpu).expect("the leg");
        let ty = GgmlType::Q4_K;
        let w = weights(&gpu, SHAPE, &[ty, GgmlType::Q5_K]);
        let names = [
            format!("gate_{ty}"),
            format!("up_{ty}"),
            format!("down_{ty}"),
        ];
        let stacks = LegStacks::of(&w, [&names[0], &names[1], &names[2]]).expect("the stacks");
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let shape = SHAPE;
        let run = LegRun {
            layer: 0,
            stacks: &stacks,
            normed: &normed,
            m: CAP,
            sel: &sel,
            n_card: N_CARD,
            rule: Act::SiluMul,
        };
        let sentinel = vec![SENT; CAP * SPC * N];

        let mut other = LegBufs::for_down(stream, shape, DownAct::Q8Blocks32).expect("buffers");
        other
            .down
            .copy_from_host(stream, &sentinel)
            .expect("sentinel");
        let refused = leg.enqueue(&gpu, &mut other, &run, None);
        assert!(
            matches!(&refused, Err(GpuError::Shape { what: WHAT, .. })),
            "a Q8Blocks32 buffer for a Q4_K down: {refused:?}"
        );
        let rows = other.down.to_host_vec(stream).expect("the rows");
        assert!(rows.iter().all(|v| *v == SENT), "a refused run wrote rows");

        let mut own = LegBufs::new(stream, shape).expect("buffers");
        own.down
            .copy_from_host(stream, &sentinel)
            .expect("sentinel");
        leg.enqueue(&gpu, &mut own, &run, None).expect("the run");
        let want = own.down.to_host_vec(stream).expect("the rows");
        let mut elsewhere = LegBufs::new(stream, shape).expect("buffers");
        elsewhere
            .down
            .copy_from_host(stream, &sentinel)
            .expect("sentinel");
        let mut mine = DeviceBuffer::from_host(stream, &sentinel).expect("the caller's rows");
        leg.enqueue_rows(&gpu, &mut elsewhere, &run, &mut mine)
            .expect("the run into the caller's rows");
        let got = mine.to_host_vec(stream).expect("the rows");
        assert!(
            gpu.take_fault().expect("the fault word").is_none(),
            "a fault"
        );
        assert!(
            got.iter()
                .zip(&want)
                .all(|(g, w)| g.to_bits() == w.to_bits()),
            "enqueue_rows wrote other rows than enqueue"
        );
        let own_rows = elsewhere.down.to_host_vec(stream).expect("the rows");
        assert!(
            own_rows.iter().all(|v| *v == SENT),
            "enqueue_rows wrote the buffers' own rows"
        );
    }

    /// The relu² tests' normed width and routed width: neither a multiple of
    /// 256 (84 and 58 blocks of 32), the second with a partial last group of
    /// four blocks in the quantizers.
    const R2_N: usize = 2688;
    /// See [`R2_N`].
    const R2_FF: usize = 1856;
    /// Slots a token carries in the relu² tests.
    const R2_SPC: usize = 6;
    /// Tokens the relu² buffers hold: 54 slot columns, more than a warp's 32.
    const R2_CAP: usize = 9;
    /// The shape of the relu² tests' buffers.
    const R2_SHAPE: LegShape = LegShape {
        cap: R2_CAP,
        slots_per_col: R2_SPC,
        n: R2_N,
        ff: R2_FF,
    };
    /// The tokens of the relu² runs: one, a few, and the buffers' cap.
    const R2_RUNS: [usize; 3] = [1, 4, R2_CAP];
    /// The relu² cases, `(up type, down type)`: the three 32-value types
    /// `kernels::card_sel32_runs` admits, each in both roles, with the mixes a
    /// real file holds (an IQ4_NL up under an IQ4_NL, a Q8_0 and a Q5_1 down).
    const R2_CASES: [(GgmlType, GgmlType); 5] = [
        (GgmlType::IQ4_NL, GgmlType::IQ4_NL),
        (GgmlType::IQ4_NL, GgmlType::Q8_0),
        (GgmlType::IQ4_NL, GgmlType::Q5_1),
        (GgmlType::Q8_0, GgmlType::Q8_0),
        (GgmlType::Q5_1, GgmlType::Q5_1),
    ];

    /// The launches the relu² form is made of, run by hand on buffers of
    /// exactly the run's slots: the plain quantizer over the rows repeated once
    /// a slot, the up and down `_sel` entries of the type, `elem`'s `relu_sqr`
    /// and the select quantizer.
    struct Hand {
        q51: Q51SelKernels,
        q80: Q80SelKernels,
        iq: QuantSelKernels,
    }

    impl Hand {
        fn load(gpu: &Gpu) -> Hand {
            let (ctx, word) = (gpu.context(), gpu.fault_word());
            Hand {
                q51: Q51SelKernels::load(ctx, word).expect("the q5_1 module"),
                q80: Q80SelKernels::load(ctx, word).expect("the q8_0 module"),
                iq: QuantSelKernels::load(ctx, word).expect("the iq module"),
            }
        }

        /// The `_sel` of the 32-value type `ty` over `act`.
        fn sel32(
            &self,
            gpu: &Gpu,
            (ty, w): (GgmlType, &DeviceTensor<u32>),
            (act, sel): (&Q8Blocks32, &DeviceBuffer<u32>),
            (n_slots, rows_per_expert): (usize, usize),
            y: &mut DeviceBuffer<f32>,
        ) {
            let (stream, sink) = (gpu.stream(), gpu.layer_sink(0).expect("a sink"));
            match ty {
                GgmlType::Q5_1 => {
                    let d = Q51SelDown {
                        w,
                        act,
                        sel,
                        n_slots,
                        rows_per_expert,
                    };
                    self.q51.enqueue_gemv_q5_1_sel(stream, &d, sink, y)
                }
                GgmlType::Q8_0 => {
                    let d = Q80SelDown {
                        w,
                        act,
                        sel,
                        n_slots,
                        rows_per_expert,
                    };
                    self.q80.enqueue_gemv_q8_0_sel32(stream, &d, sink, y)
                }
                GgmlType::IQ4_NL => {
                    let d = IqDown {
                        w,
                        act,
                        sel,
                        n_slots,
                        rows_per_expert,
                    };
                    self.iq.enqueue_down(stream, &d, sink, y)
                }
                other => panic!("no hand launch for {other}"),
            }
            .unwrap_or_else(|e| panic!("the hand {ty} launch: {e}"));
        }

        /// The first `m` tokens of `normed` (host values, `R2_N` a token)
        /// through the whole form: `(act_x, act_h, rows)`, the rows of the
        /// card slots written and the sentinel elsewhere.
        fn run(
            &self,
            gpu: &Gpu,
            w: &Weights,
            names: [&str; 2],
            (up_ty, dn_ty): (GgmlType, GgmlType),
            (normed, sel): (&[f32], &DeviceBuffer<u32>),
            m: usize,
        ) -> (Q8Blocks32, Q8Blocks32, Vec<f32>) {
            let (stream, sink) = (gpu.stream(), gpu.layer_sink(0).expect("a sink"));
            let slots = m * R2_SPC;
            let rep: Vec<f32> = (0..slots)
                .flat_map(|s| normed[(s / R2_SPC) * R2_N..][..R2_N].to_vec())
                .collect();
            let x = DeviceBuffer::from_host(stream, &rep).expect("the repeated rows");
            let mut act_x = Q8Blocks32::with_slots(stream, R2_N, slots).expect("act_x");
            gpu.q5()
                .enqueue_quantize_q8(stream, &x, &mut act_x, sink)
                .expect("the plain quantizer");
            let mut h = DeviceBuffer::<f32>::zeroed(stream, slots * R2_FF).expect("h");
            self.sel32(
                gpu,
                (up_ty, words(w, names[0])),
                (&act_x, sel),
                (slots, R2_FF),
                &mut h,
            );
            let mut sq = DeviceBuffer::<f32>::zeroed(stream, slots * R2_FF).expect("relu²");
            gpu.elem()
                .enqueue_relu_sqr(stream, &h, slots * R2_FF, &mut sq)
                .expect("relu_sqr");
            let mut act_h = Q8Blocks32::with_slots(stream, R2_FF, slots).expect("act_h");
            let q = QuantSel {
                x: &sq,
                cols: 0..slots,
                sel,
                n_card: N_CARD,
            };
            gpu.q5()
                .enqueue_quantize_q8_sel(stream, &q, &mut act_h, sink)
                .expect("the select quantizer");
            let mut down =
                DeviceBuffer::from_host(stream, &vec![SENT; slots * R2_N]).expect("the rows");
            self.sel32(
                gpu,
                (dn_ty, words(w, names[1])),
                (&act_h, sel),
                (slots, R2_N),
                &mut down,
            );
            let rows = down.to_host_vec(stream).expect("the hand rows");
            (act_x, act_h, rows)
        }
    }

    /// The first difference between two quantized activations over the card
    /// slots' columns, named by plane and column, or `None`.
    fn blocks32_diff(
        gpu: &Gpu,
        (got, want): (&Q8Blocks32, &Q8Blocks32),
        places: &[u32],
    ) -> Option<String> {
        let (g, w) = (
            got.readback(gpu.stream()).expect("readback"),
            want.readback(gpu.stream()).expect("readback"),
        );
        let (q_stride, k_blocks) = (got.q_stride(), got.k() / 32);
        for (c, &place) in places.iter().enumerate() {
            if place == HOST {
                continue;
            }
            let q = c * q_stride..(c + 1) * q_stride;
            let b = c * k_blocks..(c + 1) * k_blocks;
            if g.q[q.clone()] != w.q[q] {
                return Some(format!("column {c}: the codes differ"));
            }
            if g.s8[b.clone()] != w.s8[b.clone()] {
                return Some(format!("column {c}: the block sums differ"));
            }
            if let Some(i) = b.clone().find(|&i| g.d8[i].to_bits() != w.d8[i].to_bits()) {
                return Some(format!("column {c}: block {} scale differs", i - b.start));
            }
        }
        None
    }

    /// The relu² form against the launches it is made of, run by hand on
    /// buffers of exactly the run's slots, for every type of both roles
    /// (`R2_CASES`) at Lightning's widths and `R2_RUNS` tokens of six slots
    /// through buffers of `R2_CAP`: the q8_1 columns after the repeating
    /// quantizer and after the squared-ReLU select quantizer are the plain
    /// quantizer's of the repeated rows and the select quantizer's of
    /// `elem`'s `relu_sqr`, bit for bit; the down rows are the hand launches'
    /// bit for bit and, token by token, the rows a one-token run writes; a
    /// host slot keeps its sentinel; the card sum is the sum entry's over the
    /// leg's own rows; nothing past the run's slots is written.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_relu2_leg_runs_its_launches_as_the_hand_launches() {
        let ids = |tys: Vec<GgmlType>| {
            let mut ids: Vec<u32> = tys.into_iter().map(GgmlType::as_u32).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        let admitted = ids((0..64)
            .map(GgmlType::from_u32)
            .filter(|&ty| kernels::card_sel32_runs(ty))
            .collect());
        assert_eq!(
            ids(R2_CASES.iter().map(|c| c.0).collect()),
            admitted,
            "the cases' up types are not the types the relu² form admits"
        );
        assert_eq!(
            ids(R2_CASES.iter().map(|c| c.1).collect()),
            admitted,
            "the cases' down types are not the types the relu² form admits"
        );
        let sel_all: Vec<u32> = (0..R2_CAP).flat_map(places_of::<R2_SPC>).collect();
        for m in R2_RUNS {
            assert!(
                mixes(&sel_all[..m * R2_SPC]),
                "the places of {m} tokens must mix duplicates, every card expert and host slots"
            );
        }

        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (stream, sink) = (gpu.stream(), gpu.layer_sink(0).expect("a sink"));
        let leg = Relu2Leg::load(&gpu).expect("the leg");
        let hand = Hand::load(&gpu);
        let slots_kernels = CardSlotKernels::load(gpu.context()).expect("the sum entries");
        let mut tys: Vec<GgmlType> = R2_CASES.iter().flat_map(|&(u, d)| [u, d]).collect();
        tys.sort_by_key(|ty| ty.as_u32());
        tys.dedup();
        let w = weights_for(&gpu, R2_SHAPE, &tys, |role, ty| {
            role != "gate" && kernels::card_sel32_runs(ty)
        });
        let host_normed: Vec<f32> = (0..R2_CAP).flat_map(|t| row_of(t, R2_N)).collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &sel_all).expect("the places");
        let route: Vec<f32> = (0..R2_CAP * R2_SPC)
            .map(|i| 0.0625 + (i % 7) as f32 / 16.0)
            .collect();
        let route = DeviceBuffer::from_host(stream, &route).expect("the routing weights");
        let one = LegShape { cap: 1, ..R2_SHAPE };

        let mut failures = Vec::new();
        for (up_ty, dn_ty) in R2_CASES {
            let names = [stack_name("up", up_ty), stack_name("down", dn_ty)];
            let stacks = Relu2Stacks::of(&w, [&names[0], &names[1]]).expect("the stacks");
            stacks.check(N_CARD, R2_SHAPE).expect("the stacks' shape");
            let mut one_bufs = Relu2Bufs::new(stream, one).expect("one token's buffers");
            let oracle: Vec<Vec<f32>> = (0..R2_CAP)
                .map(|t| {
                    let row = DeviceBuffer::from_host(stream, &row_of(t, R2_N)).expect("a row");
                    let sel_t = DeviceBuffer::from_host(stream, &places_of::<R2_SPC>(t))
                        .expect("a token's places");
                    one_bufs
                        .down
                        .copy_from_host(stream, &vec![SENT; R2_SPC * R2_N])
                        .expect("the sentinel");
                    let run = Relu2Run {
                        layer: 0,
                        stacks: &stacks,
                        normed: &row,
                        m: 1,
                        sel: &sel_t,
                        n_card: N_CARD,
                    };
                    leg.enqueue(&gpu, &mut one_bufs, &run, None)
                        .unwrap_or_else(|e| panic!("{up_ty}/{dn_ty}: token {t}: {e}"));
                    one_bufs.down.to_host_vec(stream).expect("the rows")
                })
                .collect();

            for m in R2_RUNS {
                let slots = m * R2_SPC;
                let mut bufs = Relu2Bufs::new(stream, R2_SHAPE).expect("the buffers");
                bufs.down
                    .copy_from_host(stream, &vec![SENT; R2_CAP * R2_SPC * R2_N])
                    .expect("the sentinel");
                let mut acc = DeviceBuffer::from_host(stream, &vec![SENT; R2_CAP * R2_N])
                    .expect("the sum's rows");
                let run = Relu2Run {
                    layer: 0,
                    stacks: &stacks,
                    normed: &normed,
                    m,
                    sel: &sel,
                    n_card: N_CARD,
                };
                let sum = LegSum {
                    w: &route,
                    acc: &mut acc,
                };
                leg.enqueue(&gpu, &mut bufs, &run, Some(sum))
                    .unwrap_or_else(|e| panic!("{up_ty}/{dn_ty}, m = {m}: the leg: {e}"));
                let got = bufs.down.to_host_vec(stream).expect("the leg's rows");
                let got_acc = acc.to_host_vec(stream).expect("the leg's sum");
                let (hx, hh, want) = hand.run(
                    &gpu,
                    &w,
                    [&names[0], &names[1]],
                    (up_ty, dn_ty),
                    (&host_normed, &sel),
                    m,
                );
                let mut want_acc = DeviceBuffer::from_host(stream, &vec![SENT; R2_CAP * R2_N])
                    .expect("the sum's rows");
                let a = CardAcc {
                    down: &bufs.down,
                    w: &route,
                    sel: &sel,
                    n: R2_N,
                    m,
                    n_card: N_CARD,
                    n_used: R2_SPC,
                    w_pitch: R2_SPC,
                    fault: sink,
                };
                slots_kernels
                    .enqueue_card_acc(stream, &a, &mut want_acc)
                    .expect("the hand sum");
                let want_acc = want_acc.to_host_vec(stream).expect("the hand sum");
                let faulted = gpu.take_fault().expect("the fault word");

                let mut bad = Vec::new();
                if let Some(f) = faulted {
                    bad.push(format!("the run raised {f:?}"));
                }
                let places = &sel_all[..slots];
                if let Some(d) = blocks32_diff(&gpu, (&bufs.act_x, &hx), places) {
                    bad.push(format!("the repeated q8_1 columns: {d}"));
                }
                if let Some(d) = blocks32_diff(&gpu, (&bufs.act_h, &hh), places) {
                    bad.push(format!("the relu² q8_1 columns: {d}"));
                }
                for (s, &place) in places.iter().enumerate() {
                    let (token, j) = (s / R2_SPC, s % R2_SPC);
                    let rows = s * R2_N..(s + 1) * R2_N;
                    if let Err(why) = slot_state(&got[rows.clone()], place) {
                        bad.push(format!("slot {s} (token {token}): {why}"));
                        continue;
                    }
                    let one_token = &oracle[token][j * R2_N..(j + 1) * R2_N];
                    for (what, other) in [
                        ("the hand launches'", &want[rows.clone()]),
                        ("the one-token run's", one_token),
                    ] {
                        if let Some(r) = (0..R2_N).find(|&r| {
                            place != HOST && got[rows.start + r].to_bits() != other[r].to_bits()
                        }) {
                            bad.push(format!(
                                "slot {s} (token {token}) row {r}: {:#010x}, {what} {:#010x}",
                                got[rows.start + r].to_bits(),
                                other[r].to_bits()
                            ));
                        }
                    }
                }
                if got[slots * R2_N..].iter().any(|v| *v != SENT) {
                    bad.push("the run wrote past its slots".to_string());
                }
                let sum_rows = m * R2_N;
                if let Some(i) =
                    (0..sum_rows).find(|&i| got_acc[i].to_bits() != want_acc[i].to_bits())
                {
                    bad.push(format!(
                        "card sum value {i}: {:#010x}, the sum entry's {:#010x}",
                        got_acc[i].to_bits(),
                        want_acc[i].to_bits()
                    ));
                }
                if got_acc[..sum_rows]
                    .iter()
                    .any(|v| !v.is_finite() || *v == SENT)
                {
                    bad.push("a card sum value is no finite value the run wrote".to_string());
                }
                if got_acc[sum_rows..].iter().any(|v| *v != SENT) {
                    bad.push("the sum wrote past its tokens".to_string());
                }
                let verdict = if bad.is_empty() { "ok" } else { "FAIL" };
                println!("leg_relu2[up {up_ty}, down {dn_ty}, m = {m}] {verdict}");
                if let Some(first) = bad.first() {
                    failures.push(format!(
                        "up {up_ty}, down {dn_ty}, m = {m}: {} findings; first: {first}",
                        bad.len()
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    /// The relu² form's buffers hold the bytes [`relu2_leg_bytes`] counts, and
    /// its refusals are by name before the first launch: a stack of a type the
    /// form has no launch for, stacks of another shape, a run of no tokens or
    /// more than the buffers hold, a sum over seven slots, widths off the
    /// 32-value block. A NaN in a card slot's normed row raises `Q5Quant` from
    /// the repeating quantizer.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_relu2_leg_counts_its_bytes_and_refuses_by_name() {
        use crate::fault::FaultSite;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let stream = gpu.stream();
        let leg = Relu2Leg::load(&gpu).expect("the leg");
        for shape in [
            R2_SHAPE,
            LegShape {
                cap: 2,
                slots_per_col: 7,
                n: 96,
                ff: 32,
            },
        ] {
            let bufs = Relu2Bufs::new(stream, shape).expect("the buffers");
            assert_eq!(bufs.bytes(), relu2_leg_bytes(shape), "{shape:?}");
        }
        for (n, ff) in [(R2_N - 16, R2_FF), (R2_N, R2_FF + 8)] {
            let shape = LegShape { n, ff, ..R2_SHAPE };
            let refused = Relu2Bufs::new(stream, shape);
            assert!(
                matches!(&refused, Err(GpuError::Shape { .. })),
                "n = {n}, ff = {ff} are not multiples of 32: {:?}",
                refused.map(|_| ())
            );
        }

        // Q4_K is no 32-value type: refused when the stacks are read, in
        // either role; the 32-value types come through.
        let mixed = weights_for(
            &gpu,
            SHAPE,
            &[GgmlType::Q4_K, GgmlType::IQ4_NL],
            |role, ty| role != "gate" && (kernels::card_sel32_runs(ty) || ty == GgmlType::Q4_K),
        );
        let (k_up, k_down) = (
            stack_name("up", GgmlType::Q4_K),
            stack_name("down", GgmlType::Q4_K),
        );
        let (nl_up, nl_down) = (
            stack_name("up", GgmlType::IQ4_NL),
            stack_name("down", GgmlType::IQ4_NL),
        );
        for (role, names) in [("up", [&k_up, &nl_down]), ("down", [&nl_up, &k_down])] {
            let refused = Relu2Stacks::of(&mixed, [names[0], names[1]]);
            assert!(
                matches!(&refused, Err(GpuError::Tensor { what: WHAT, .. })),
                "a Q4_K {role} stack: {:?}",
                refused.map(|_| ())
            );
        }
        Relu2Stacks::of(&mixed, [&nl_up, &nl_down]).expect("IQ4_NL stacks");

        let tys = [GgmlType::IQ4_NL, GgmlType::Q8_0];
        let w = weights_for(&gpu, R2_SHAPE, &tys, |role, ty| {
            role != "gate" && kernels::card_sel32_runs(ty)
        });
        let names = [
            stack_name("up", GgmlType::IQ4_NL),
            stack_name("down", GgmlType::Q8_0),
        ];
        let stacks = Relu2Stacks::of(&w, [&names[0], &names[1]]).expect("the stacks");
        stacks.check(N_CARD, R2_SHAPE).expect("the stacks' shape");
        for (n_card, shape) in [
            (N_CARD + 1, R2_SHAPE),
            (
                N_CARD,
                LegShape {
                    ff: R2_FF - 32,
                    ..R2_SHAPE
                },
            ),
        ] {
            let refused = stacks.check(n_card, shape);
            assert!(
                matches!(&refused, Err(GpuError::Tensor { what: WHAT, .. })),
                "{n_card} experts of {shape:?}: {refused:?}"
            );
        }

        let host_normed: Vec<f32> = (0..R2_CAP).flat_map(|t| row_of(t, R2_N)).collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel_all: Vec<u32> = (0..R2_CAP).flat_map(places_of::<R2_SPC>).collect();
        let sel = DeviceBuffer::from_host(stream, &sel_all).expect("the places");
        let sentinel = vec![SENT; R2_CAP * R2_SPC * R2_N];
        let run = |m: usize| Relu2Run {
            layer: 0,
            stacks: &stacks,
            normed: &normed,
            m,
            sel: &sel,
            n_card: N_CARD,
        };
        let mut bufs = Relu2Bufs::new(stream, R2_SHAPE).expect("the buffers");
        bufs.down
            .copy_from_host(stream, &sentinel)
            .expect("the sentinel");
        for m in [0, R2_CAP + 1] {
            let refused = leg.enqueue(&gpu, &mut bufs, &run(m), None);
            assert!(
                matches!(&refused, Err(GpuError::Shape { what: WHAT, .. })),
                "a run of {m} tokens: {refused:?}"
            );
        }
        let route = DeviceBuffer::from_host(stream, &[0.5f32; 2 * 7]).expect("the weights");
        let mut acc = DeviceBuffer::<f32>::zeroed(stream, 2 * 96).expect("the sum");
        let seven = LegShape {
            cap: 2,
            slots_per_col: 7,
            n: 96,
            ff: 32,
        };
        let mut small = Relu2Bufs::new(stream, seven).expect("the buffers");
        let refused = leg.enqueue(
            &gpu,
            &mut small,
            &Relu2Run { m: 1, ..run(1) },
            Some(LegSum {
                w: &route,
                acc: &mut acc,
            }),
        );
        assert!(
            matches!(&refused, Err(GpuError::Shape { what: WHAT, .. })),
            "a sum over seven slots: {refused:?}"
        );
        assert!(
            bufs.down
                .to_host_vec(stream)
                .expect("the rows")
                .iter()
                .all(|v| *v == SENT),
            "a refused run wrote rows"
        );
        assert!(
            gpu.take_fault().expect("the fault word").is_none(),
            "a refusal raised a fault"
        );

        // A NaN in token 1's row: the repeating quantizer refuses the blocks
        // of its card slots' columns and raises Q5Quant.
        let mut nan_normed = host_normed.clone();
        nan_normed[R2_N + 5] = f32::NAN;
        let nan_normed = DeviceBuffer::from_host(stream, &nan_normed).expect("the rows");
        leg.enqueue(
            &gpu,
            &mut bufs,
            &Relu2Run {
                normed: &nan_normed,
                ..run(2)
            },
            None,
        )
        .expect("the run");
        let fault = gpu.take_fault().expect("the fault word");
        assert!(
            fault
                .as_ref()
                .is_some_and(|f| f.sites & (1 << FaultSite::Q5Quant as u32) != 0 && f.layer == 0),
            "a NaN normed row raised {fault:?}"
        );
    }

    /// The arena entries' cases, `(gate·up type, down type)`: each arena
    /// entry runs in one, in GLM's pairs.
    const AT_CASES: [(GgmlType, GgmlType); 2] = [
        (GgmlType::Q4_K, GgmlType::Q5_K),
        (GgmlType::Q5_K, GgmlType::Q6_K),
    ];

    /// The arena entries against the stack `_sel` launches of their format:
    /// each expert's three parts copied as their file bytes into pinned host
    /// memory the card maps, out of stack order and each at its own word
    /// skew, and a table naming them by slot. The gate·up rows and, from the
    /// stack run's q8_1 columns, the down rows are the stack launches' bit
    /// for bit; a skipped slot keeps what it held. A part one word past the
    /// arena raises `ExpertId`: the gate·up writes NaN, the down leaves the
    /// slot alone.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_an_arena_part_computes_as_its_stack_rows() {
        use crate::fault::FaultSite;
        use crate::graph::MappedHost;
        use crate::kquant::{
            ARENA_PARTS, ARENA_SKIP, GateUpAt, PART_DOWN, PART_GATE, PART_UP, SelDownAt,
        };
        use crate::q6k_sel::Q6kSelKernels;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (stream, ctx) = (gpu.stream(), gpu.context());
        let kq = QuantSelKernels::load(ctx, gpu.fault_word()).expect("the kquant module");
        let q6 = Q6kSelKernels::load(ctx, gpu.fault_word()).expect("the q6k module");
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let slots = CAP * SPC;
        let rule = Act::SwigluClamp { limit: 0.25 };
        let sink = gpu.layer_sink(0).expect("a sink");
        let mut act_x = Q8Act::with_tier_cols(stream, CAP, N).expect("act_x");
        gpu.enqueue_quantize_q8_1_cols(&normed, &mut act_x, CAP, 0)
            .expect("q8_1");
        for (gty, dty) in AT_CASES {
            let case = format!("{gty} gate·up, {dty} down");
            // Per part: the stack's file bytes, rows a part, and the rows' k.
            let roles = [(gty, FF, N, 1u64), (gty, FF, N, 2), (dty, N, FF, 3)];
            let bytes: Vec<Vec<u8>> = roles
                .iter()
                .map(|&(ty, rows, k, seed)| stack_bytes(ty, N_CARD * rows, k, seed << 32 | 0x51))
                .collect();
            let stacks: Vec<DeviceTensor<u32>> = roles
                .iter()
                .zip(&bytes)
                .map(|(&(_, rows, _, _), b)| {
                    let words = words_of(b, N_CARD * rows);
                    let n = words.len();
                    DeviceTensor::upload(stream, &words, N_CARD * rows, n / (N_CARD * rows))
                        .expect("a stack")
                })
                .collect();
            let part_len: Vec<usize> = bytes.iter().map(|b| b.len() / N_CARD).collect();
            // Every expert's parts, down first, each 1..=11 words past the
            // previous one's end.
            let mut offs_of = [[0u64; ARENA_PARTS]; N_CARD];
            let mut cursor = 3usize;
            for (i, e) in [2usize, 0, 3, 1].into_iter().enumerate() {
                for p in [PART_DOWN, PART_GATE, PART_UP] {
                    offs_of[e][p] = cursor as u64;
                    cursor += part_len[p].div_ceil(4) + 1 + (5 * i + 3 * p) % 11;
                }
            }
            let arena_words = cursor;
            let widest = part_len.iter().max().expect("parts").div_ceil(4);
            let page =
                MappedHost::read_in_place(ctx, 4 * (arena_words + widest), "the test's arena")
                    .expect("pinned host memory the card maps");
            for (e, offs) in offs_of.iter().enumerate() {
                for p in 0..ARENA_PARTS {
                    let src = &bytes[p][e * part_len[p]..(e + 1) * part_len[p]];
                    // SAFETY: the part's bytes lie inside the allocation (the
                    // cursor kept every part below arena_words words), and no
                    // launch reads the page yet.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src.as_ptr(),
                            page.host_at(4 * offs[p] as usize),
                            src.len(),
                        );
                    }
                }
            }
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            // SAFETY: arena_words words of the live allocation the card maps,
            // which outlives the window.
            let arena = unsafe { crate::tensor::window::<u32>(page.dev_at(0), arena_words, ctx) };
            let table = |past: Option<usize>| -> Vec<u64> {
                let mut t = Vec::with_capacity(slots * ARENA_PARTS);
                for (s, &id) in SEL.iter().enumerate() {
                    if id == HOST {
                        t.extend([ARENA_SKIP; ARENA_PARTS]);
                        continue;
                    }
                    let mut o = offs_of[id as usize];
                    if past == Some(s) {
                        for p in 0..ARENA_PARTS {
                            o[p] = (arena_words + 1 - part_len[p].div_ceil(4)) as u64;
                        }
                    }
                    t.extend(o);
                }
                t
            };
            let offs = DeviceBuffer::from_host(stream, &table(None)).expect("the table");

            let mut h_ref = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("h");
            let mut h_at = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("h");
            let a = GateUpAct {
                wg: &stacks[PART_GATE],
                wu: &stacks[PART_UP],
                act: &act_x,
                sel: &sel,
                n_slots: slots,
                rows_per_expert: FF,
                slots_per_col: SPC,
                rule,
            };
            let at = GateUpAt {
                arena: &arena,
                offs: &offs,
                act: &act_x,
                n_slots: slots,
                rows_per_expert: FF,
                slots_per_col: SPC,
                rule,
            };
            match gty {
                GgmlType::Q4_K => kq
                    .enqueue_gate_up_q4k(stream, &a, sink, &mut h_ref)
                    .and_then(|()| kq.enqueue_gate_up_q4k_at(stream, &at, sink, &mut h_at)),
                _ => kq
                    .enqueue_gate_up_q5k(stream, &a, sink, &mut h_ref)
                    .and_then(|()| kq.enqueue_gate_up_q5k_at(stream, &at, sink, &mut h_at)),
            }
            .unwrap_or_else(|e| panic!("{case}: the gate·up: {e}"));
            let mut act_h = Q8Act::with_slots(stream, slots, FF).expect("act_h");
            let q = QuantSel {
                x: &h_ref,
                cols: 0..slots,
                sel: &sel,
                n_card: N_CARD,
            };
            gpu.q4k_sel()
                .enqueue_quantize_sel(stream, &q, sink, &mut act_h)
                .unwrap_or_else(|e| panic!("{case}: the quantizer: {e}"));
            let mut y_ref = DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("y");
            let mut y_at = DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("y");
            let d_at = SelDownAt {
                arena: &arena,
                offs: &offs,
                act: &act_h,
                n_slots: slots,
                rows_per_expert: N,
            };
            let down_at = |d: &SelDownAt<'_>, y: &mut DeviceBuffer<f32>| match dty {
                GgmlType::Q5_K => kq.enqueue_gemv_q5k_sel_at(stream, d, sink, y),
                _ => q6.enqueue_gemv_q6k_sel_at(stream, d, sink, y),
            };
            match dty {
                GgmlType::Q5_K => {
                    let d = SelDown {
                        w: &stacks[PART_DOWN],
                        act: &act_h,
                        sel: &sel,
                        n_slots: slots,
                        rows_per_expert: N,
                    };
                    kq.enqueue_gemv_q5k_sel(stream, &d, sink, &mut y_ref)
                }
                _ => q6.enqueue_gemv_q6k_sel(
                    stream,
                    &stacks[PART_DOWN],
                    &act_h,
                    &sel,
                    slots,
                    N,
                    &mut y_ref,
                ),
            }
            .and_then(|()| down_at(&d_at, &mut y_at))
            .unwrap_or_else(|e| panic!("{case}: the down: {e}"));
            let (h_ref, h_got) = (
                h_ref.to_host_vec(stream).expect("h"),
                h_at.to_host_vec(stream).expect("h"),
            );
            let (y_ref, y_got) = (
                y_ref.to_host_vec(stream).expect("y"),
                y_at.to_host_vec(stream).expect("y"),
            );
            assert!(
                gpu.take_fault().expect("the fault word").is_none(),
                "{case}: a fault"
            );
            for (what, rpe, want, got) in [("h", FF, &h_ref, &h_got), ("down", N, &y_ref, &y_got)] {
                for (s, &id) in SEL.iter().enumerate() {
                    let rows = s * rpe..(s + 1) * rpe;
                    if id == HOST {
                        assert!(
                            got[rows].iter().all(|v| *v == SENT),
                            "{case}: {what} of slot {s} (skipped) was written"
                        );
                        continue;
                    }
                    assert!(
                        got[rows.clone()]
                            .iter()
                            .all(|v| v.is_finite() && *v != SENT)
                            && got[rows.clone()].iter().any(|v| *v != 0.0),
                        "{case}: {what} of slot {s} (expert {id}) holds no finite rows the \
                         entry wrote"
                    );
                    if let Some(r) = rows
                        .clone()
                        .find(|&r| got[r].to_bits() != want[r].to_bits())
                    {
                        panic!(
                            "{case}: {what} of slot {s} (expert {id}) row {}: arena {:#010x}, \
                             stack {:#010x}",
                            r - rows.start,
                            got[r].to_bits(),
                            want[r].to_bits()
                        );
                    }
                }
            }

            let past = DeviceBuffer::from_host(stream, &table(Some(0))).expect("the table");
            let mut h = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("h");
            let at = GateUpAt { offs: &past, ..at };
            match gty {
                GgmlType::Q4_K => kq.enqueue_gate_up_q4k_at(stream, &at, sink, &mut h),
                _ => kq.enqueue_gate_up_q5k_at(stream, &at, sink, &mut h),
            }
            .unwrap_or_else(|e| panic!("{case}: the gate·up past the arena: {e}"));
            let h = h.to_host_vec(stream).expect("h");
            let fault = gpu.take_fault().expect("the fault word");
            assert!(
                fault
                    .as_ref()
                    .is_some_and(|f| f.code == FaultSite::ExpertId as u32 && f.layer == 0),
                "{case}: a gate·up part past the arena raised {fault:?}"
            );
            assert!(
                h[..FF].iter().all(|v| v.is_nan()),
                "{case}: a gate·up part past the arena left rows that are not NaN"
            );
            let mut y = DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("y");
            down_at(
                &SelDownAt {
                    offs: &past,
                    ..d_at
                },
                &mut y,
            )
            .unwrap_or_else(|e| panic!("{case}: the down past the arena: {e}"));
            let y = y.to_host_vec(stream).expect("y");
            let fault = gpu.take_fault().expect("the fault word");
            assert!(
                fault
                    .as_ref()
                    .is_some_and(|f| f.code == FaultSite::ExpertId as u32 && f.layer == 0),
                "{case}: a down part past the arena raised {fault:?}"
            );
            assert!(
                y[..N].iter().all(|v| *v == SENT),
                "{case}: a down part past the arena wrote its slot"
            );
            // The window frees nothing; the page goes after every launch
            // that read it drained (the reads above).
            drop(std::mem::ManuallyDrop::into_inner(arena).into_raw_parts());
            drop(page);
        }
    }
}
