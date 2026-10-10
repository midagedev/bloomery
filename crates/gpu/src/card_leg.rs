//! The card leg of a routed layer's experts, common to every program that runs
//! experts on the card by the kernel table (`model::placement::kernels`):
//! the four launches from a layer's normed rows to its slots' down rows.
//!
//! One run covers `m` tokens of `slots_per_col` routed slots each:
//! - the normed rows in q8_1 (`m` columns, `Gpu::enqueue_quantize_q8_1_cols`);
//! - the gate·up `_sel` of the entry [`kernels::gate_up`] names for the
//!   stacks' type, the activation rule a launch argument, slot `s` against
//!   column `s / slots_per_col`, writing `h` (`ff` rows a slot);
//! - the q8_1 of the card slots' `h` columns (`Q4kSelKernels::
//!   enqueue_quantize_sel`; a slot the run places off the card is left alone);
//! - the down `_sel` of the entry [`kernels::down`] names, writing `down`
//!   (`n` rows a slot).
//!
//! A slot's rows are what a one-token run writes for it, whatever `m`: the
//! launches are slot-local and read their columns by slot, so a unit of a
//! prompt batch and a decode step leave the same bits for the same slot.
//!
//! What stays with the caller, because it is the family's: the slots' places
//! `sel` (the handoff's, or a batch's `enqueue_places`), the weighted sum of
//! the down rows and what it is combined with (a shared expert, the residual,
//! the host tier's rows), and the routing weights. The sum kernels live in the
//! families' crates, above this one.
//!
//! The run launches the entries [`kernels::card_runs`] admits: those whose
//! launchers read an activation of more columns than a run uses (a
//! [`LegBufs`] sized for `cap` tokens serves a unit of fewer in one launch).
//! A stack of any other type is refused by name when the stacks are read
//! ([`LegStacks::of`]), and an entry the table names but the run has no arm
//! for is refused by name at the launch.

use crate::kquant::{Act, GateUpAct, SelDown};
use crate::mxfp4_sel::Mxfp4SelKernels;
use crate::q4k_sel::QuantSel;
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError, Q8Act};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;
use model::placement::kernels::{self, DownAct, DownEntry, GateUpEntry};

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

/// One run's buffers for up to `shape.cap` tokens: the normed rows' q8_1 form,
/// the gate·up's rows, the card slots' q8_1 columns and the down's rows.
pub struct LegBufs {
    shape: LegShape,
    act_x: Q8Act,
    h: DeviceBuffer<f32>,
    act_h: Q8Act,
    down: DeviceBuffer<f32>,
}

impl LegBufs {
    /// The buffers for runs of `shape`. Load-time or first-prompt only.
    pub fn new(stream: &CudaStream, shape: LegShape) -> Result<LegBufs, GpuError> {
        let slots = shape.slots();
        Ok(LegBufs {
            shape,
            act_x: Q8Act::with_tier_cols(stream, shape.cap, shape.n)?,
            h: DeviceBuffer::zeroed(stream, slots * shape.ff)?,
            act_h: Q8Act::with_slots(stream, slots, shape.ff)?,
            down: DeviceBuffer::zeroed(stream, slots * shape.n)?,
        })
    }

    /// The shape the buffers were made for.
    #[must_use]
    pub fn shape(&self) -> LegShape {
        self.shape
    }

    /// Device bytes of the buffers: [`leg_bytes`] of their shape, as
    /// allocated.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.act_x.device_bytes()
            + self.h.num_bytes()
            + self.act_h.device_bytes()
            + self.down.num_bytes()
    }

    /// The down rows of the last run: `n` values a slot, slot-major, the
    /// slots the run placed on the card (the rest are as an earlier run left
    /// them).
    #[must_use]
    pub fn down(&self) -> &DeviceBuffer<f32> {
        &self.down
    }
}

/// Device bytes a [`LegBufs`] of `shape` allocates, counted before it does:
/// the normed rows' q8_1 form (`cap` columns of `n`), the gate·up's rows, the
/// card slots' q8_1 columns (`cap · slots_per_col` of `ff`) and the down's
/// rows.
#[must_use]
pub fn leg_bytes(shape: LegShape) -> usize {
    let slots = shape.slots();
    let f = size_of::<f32>();
    q8_act_bytes(shape.cap, shape.n)
        + f * slots * shape.ff
        + q8_act_bytes(slots, shape.ff)
        + f * slots * shape.n
}

/// Device bytes of a [`Q8Act`] of `cols` columns of `k` values (a multiple of
/// 256): the five planes at `n_sb = k / 256` super-blocks a column: q3 `64 ·
/// ceil(n_sb / 2)` words of 8 bytes, q4 `256 · ceil(n_sb / 4)` and q6 `128 ·
/// ceil(n_sb / 2)` words of 4, s8 `8 · n_sb` and d8 `2 · n_sb` values of 4.
const fn q8_act_bytes(cols: usize, k: usize) -> usize {
    let n_sb = k / 256;
    cols * (8 * 64 * n_sb.div_ceil(2)
        + 4 * 256 * n_sb.div_ceil(4)
        + 4 * 128 * n_sb.div_ceil(2)
        + 4 * 8 * n_sb
        + 4 * 2 * n_sb)
}

/// One resident routed stack: its name, file type, words and row width.
struct LegStack<'a> {
    name: &'a str,
    ty: GgmlType,
    w: &'a DeviceTensor<u32>,
    k: usize,
}

impl<'a> LegStack<'a> {
    fn of(w: &'a Weights, name: &'a str) -> Result<LegStack<'a>, GpuError> {
        match w.get(name) {
            Some(DevWeight::KQuant { ty, w, k }) if kernels::card_runs(*ty) => Ok(LegStack {
                name,
                ty: *ty,
                w,
                k: *k,
            }),
            _ => Err(GpuError::Tensor {
                what: WHAT,
                name: name.to_owned(),
                need: "a resident routed stack of a type the card leg runs (`kernels::card_runs`)",
            }),
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
    /// `DevWeight::KQuant` of a type [`kernels::card_runs`] admits, the gate
    /// and up of one type. Allocation-free on success.
    pub fn of(w: &'a Weights, names: [&'a str; 3]) -> Result<LegStacks<'a>, GpuError> {
        let [gate, up, down] = names;
        let (gate, up, down) = (
            LegStack::of(w, gate)?,
            LegStack::of(w, up)?,
            LegStack::of(w, down)?,
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
            if st.w.rows() != n_card * rows || st.k != k {
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

/// The modules the leg's launches run on.
pub struct CardLeg {
    mxfp4: Mxfp4SelKernels,
}

impl CardLeg {
    /// The modules on `gpu`. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<CardLeg, GpuError> {
        Ok(CardLeg {
            mxfp4: Mxfp4SelKernels::load(gpu.context(), gpu.fault_word())?,
        })
    }

    /// Enqueue on `gpu`'s stream the four launches (module doc) of `run` into
    /// `bufs`; `run.m` outside `1..=bufs.shape().cap` is refused by name.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue(&self, gpu: &Gpu, bufs: &mut LegBufs, run: &LegRun<'_>) -> Result<(), GpuError> {
        let LegShape {
            cap,
            slots_per_col,
            n,
            ff,
        } = bufs.shape;
        if run.m == 0 || run.m > cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a run of {} tokens in buffers of {cap}", run.m),
            });
        }
        let stream = gpu.stream();
        let fault = gpu.layer_sink(run.layer)?;
        let st = run.stacks;
        let slots = run.m * slots_per_col;
        gpu.enqueue_quantize_q8_1_cols(run.normed, &mut bufs.act_x, run.m, run.layer)?;
        let a = GateUpAct {
            wg: st.gate.w,
            wu: st.up.w,
            act: &bufs.act_x,
            sel: run.sel,
            n_slots: slots,
            rows_per_expert: ff,
            slots_per_col,
            rule: run.rule,
        };
        match kernels::gate_up(st.gate.ty) {
            Some(GateUpEntry::Mxfp4) => {
                self.mxfp4.enqueue_gate_up(stream, &a, fault, &mut bufs.h)?
            }
            _ => return Err(st.gate.unrun()),
        }
        let q = QuantSel {
            x: &bufs.h,
            cols: 0..slots,
            sel: run.sel,
            n_card: run.n_card,
        };
        gpu.q4k_sel()
            .enqueue_quantize_sel(stream, &q, fault, &mut bufs.act_h)?;
        let d = SelDown {
            w: st.down.w,
            act: &bufs.act_h,
            sel: run.sel,
            n_slots: slots,
            rows_per_expert: n,
        };
        match kernels::down(st.down.ty) {
            Some((DownEntry::Mxfp4, DownAct::Q8Act)) => {
                self.mxfp4.enqueue_down(stream, &d, fault, &mut bufs.down)
            }
            _ => Err(st.down.unrun()),
        }
    }
}
