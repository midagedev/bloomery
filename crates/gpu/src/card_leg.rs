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
use crate::iq_sel::{IqDown, IqGateUp, IqSelKernels};
use crate::kquant::{Act, GateUpAct, KquantKernels, SelDown};
use crate::mxfp4_sel::Mxfp4SelKernels;
use crate::q4k_sel::QuantSel;
use crate::q5::Q8Blocks32;
use crate::q5_1_sel::{Q51SelDown, Q51SelKernels};
use crate::q8_0_sel32::{Q80SelDown, Q80SelKernels};
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
    /// Q8Act`] (the K-quant and MXFP4 downs). Load-time or first-prompt only.
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
        DownAct::Q8Act => q8_act_bytes(slots, shape.ff),
        DownAct::Q8Blocks32 => q8_blocks32_bytes(slots, shape.ff),
    };
    q8_act_bytes(shape.cap, shape.n) + f * slots * shape.ff + act_h + f * slots * shape.n
}

/// Device bytes of a [`Q8Blocks32`] of `cols` columns of `k` values (a multiple
/// of 32): per column `256 · ceil(k_blocks / 32)` words of `q` and `k_blocks`
/// values each of `s8` and `d8`, of 4 bytes, `k_blocks = k / 32`.
const fn q8_blocks32_bytes(cols: usize, k: usize) -> usize {
    let k_blocks = k / 32;
    cols * 4 * (256 * k_blocks.div_ceil(32) + 2 * k_blocks)
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
    /// The resident `KQuant` stack `name`, of a type `runs` admits in the
    /// stack's role (`role`, for the refusal's words).
    fn of(
        w: &'a Weights,
        name: &'a str,
        runs: fn(GgmlType) -> bool,
        role: &'static str,
    ) -> Result<LegStack<'a>, GpuError> {
        match w.get(name) {
            Some(DevWeight::KQuant { ty, w, k }) if runs(*ty) => Ok(LegStack {
                name,
                ty: *ty,
                w,
                k: *k,
            }),
            _ => Err(GpuError::Tensor {
                what: WHAT,
                name: name.to_owned(),
                need: role,
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
    /// `DevWeight::KQuant`: the gate and up of one type [`kernels::
    /// card_gate_up_runs`] admits, the down of a type [`kernels::
    /// card_down_runs`] admits (not necessarily the same one). Allocation-free
    /// on success.
    pub fn of(w: &'a Weights, names: [&'a str; 3]) -> Result<LegStacks<'a>, GpuError> {
        const GATE_UP: &str = "a resident routed gate or up stack of a type the card leg runs (`kernels::card_gate_up_runs`)";
        const DOWN: &str =
            "a resident routed down stack of a type the card leg runs (`kernels::card_down_runs`)";
        let [gate, up, down] = names;
        let (gate, up, down) = (
            LegStack::of(w, gate, kernels::card_gate_up_runs, GATE_UP)?,
            LegStack::of(w, up, kernels::card_gate_up_runs, GATE_UP)?,
            LegStack::of(w, down, kernels::card_down_runs, DOWN)?,
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
    kquant: KquantKernels,
    mxfp4: Mxfp4SelKernels,
    q51: Q51SelKernels,
    q80: Q80SelKernels,
    iq: IqSelKernels,
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
            kquant: KquantKernels::load(ctx, word)?,
            mxfp4: Mxfp4SelKernels::load(ctx, word)?,
            q51: Q51SelKernels::load(ctx, word)?,
            q80: Q80SelKernels::load(ctx, word)?,
            iq: IqSelKernels::load(ctx, word)?,
            slots: CardSlotKernels::load(ctx)?,
        })
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
        if sum.is_some() && CardSlots::of(shape.slots_per_col).is_none() {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "the card sum: {} routed slots a token: the sum's entries are built for 6, 8 \
                     and 10",
                    shape.slots_per_col
                ),
            });
        }
        let scratch = Scratch {
            shape,
            act_x,
            h,
            act_h,
        };
        self.launch(gpu, scratch, run, down)?;
        let Some(LegSum { w, acc }) = sum else {
            return Ok(());
        };
        let a = CardAcc {
            down,
            w,
            sel: run.sel,
            n: shape.n,
            m: run.m,
            n_card: run.n_card,
            n_used: shape.slots_per_col,
            w_pitch: shape.slots_per_col,
            fault: gpu.layer_sink(run.layer)?,
        };
        self.slots.enqueue_card_acc(gpu.stream(), &a, acc)
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
        let a = GateUpAct {
            wg: st.gate.w,
            wu: st.up.w,
            act: act_x,
            sel: run.sel,
            n_slots: slots,
            rows_per_expert: ff,
            slots_per_col,
            rule: run.rule,
        };
        match kernels::gate_up(st.gate.ty) {
            Some(GateUpEntry::Q4k) => self.kquant.enqueue_gate_up_q4k(stream, &a, fault, h)?,
            Some(GateUpEntry::Q5k) => self.kquant.enqueue_gate_up_q5k(stream, &a, fault, h)?,
            Some(GateUpEntry::Mxfp4) => self.mxfp4.enqueue_gate_up(stream, &a, fault, h)?,
            Some(GateUpEntry::Iq3xxs | GateUpEntry::Iq4xs) => {
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
                self.iq.enqueue_gate_up(stream, &iq, fault, h)?;
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
        let w = st.down.w;
        let (sel, rows_per_expert) = (run.sel, n);
        match (kernels::down(st.down.ty), &*act_h) {
            (Some((DownEntry::Q4k, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                gpu.q4k_sel().enqueue_gemv_q4k_sel(
                    stream,
                    w,
                    act,
                    sel,
                    slots,
                    rows_per_expert,
                    down,
                )?;
            }
            (Some((DownEntry::Q5k, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                let d = SelDown {
                    w,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.kquant.enqueue_gemv_q5k_sel(stream, &d, fault, down)?;
            }
            (Some((DownEntry::Mxfp4, DownAct::Q8Act)), ActH::Q8Act(act)) => {
                let d = SelDown {
                    w,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.mxfp4.enqueue_down(stream, &d, fault, down)?;
            }
            (Some((DownEntry::Q5_1, DownAct::Q8Blocks32)), ActH::Blocks32(act)) => {
                let d = Q51SelDown {
                    w,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.q51.enqueue_gemv_q5_1_sel(stream, &d, fault, down)?;
            }
            (Some((DownEntry::Q8_0, DownAct::Q8Blocks32)), ActH::Blocks32(act)) => {
                let d = Q80SelDown {
                    w,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.q80.enqueue_gemv_q8_0_sel32(stream, &d, fault, down)?;
            }
            (Some((DownEntry::Iq4nl, DownAct::Q8Blocks32)), ActH::Blocks32(act)) => {
                let d = IqDown {
                    w,
                    act,
                    sel,
                    n_slots: slots,
                    rows_per_expert,
                };
                self.iq.enqueue_down(stream, &d, fault, down)?;
            }
            _ => return Err(st.down.unrun()),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid::HOST;
    use crate::q4k_sel::QuantSel;
    use gguf::Split;
    use gguf::write::{Layout, TensorDecl, Writer};

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

    /// Next value of a xorshift64 stream.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// `rows` rows of `k` values of `ty` (Q4_K or Q5_K) in the file's block
    /// bytes: every byte random but each block's two f16 scales, which are
    /// finite and small, so no dot is NaN.
    fn stack_bytes(ty: GgmlType, rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let block = usize::try_from(ty.type_size().expect("a K-quant block")).expect("a size");
        let mut state = seed;
        let mut out = vec![0u8; rows * (k / 256) * block];
        for b in out.chunks_exact_mut(block) {
            for chunk in b.chunks_mut(8) {
                let v = next(&mut state).to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
            for scale in b[..4].chunks_mut(2) {
                let (exp, mant) = (5 + next(&mut state) % 4, next(&mut state) & 0x3ff);
                let half = u16::try_from(exp << 10 | mant).expect("an f16");
                scale.copy_from_slice(&half.to_le_bytes());
            }
        }
        out
    }

    /// A file of the six stacks `<role>_<ty>` — gate and up `N_CARD · FF`
    /// rows of `N`, the down `N_CARD · N` rows of `FF` — resident in a
    /// [`Weights`].
    fn weights(gpu: &Gpu, tys: [GgmlType; 2]) -> Weights {
        let mut decls = Vec::new();
        let mut data = Vec::new();
        for (i, ty) in tys.into_iter().enumerate() {
            for (role, rows, k) in [
                ("gate", N_CARD * FF, N),
                ("up", N_CARD * FF, N),
                ("down", N_CARD * N, FF),
            ] {
                let seed = 0x9e37_79b9_7f4a_7c15 ^ ((i as u64) << 8 | role.len() as u64);
                let bytes = stack_bytes(ty, rows, k, seed);
                decls.push(TensorDecl {
                    name: format!("{role}_{ty}"),
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
        let weights = Weights::load_where(gpu.stream(), &split, |_| true).expect("resident");
        let _ = std::fs::remove_dir_all(&dir);
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
        let kq = KquantKernels::load(gpu.context(), gpu.fault_word()).expect("the kquant module");
        let tys = [GgmlType::Q4_K, GgmlType::Q5_K];
        let w = weights(&gpu, tys);
        let host_normed: Vec<f32> = (0..CAP * N)
            .map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let normed = DeviceBuffer::from_host(stream, &host_normed).expect("the normed rows");
        let sel = DeviceBuffer::from_host(stream, &SEL).expect("the places");
        let shape = LegShape {
            cap: CAP,
            slots_per_col: SPC,
            n: N,
            ff: FF,
        };
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
                for (s, &place) in SEL[..slots].iter().enumerate() {
                    let rows = s * N..(s + 1) * N;
                    if place == HOST {
                        assert!(
                            got[rows.clone()].iter().all(|v| *v == SENT),
                            "{case}: slot {s} (HOST) was written"
                        );
                    } else {
                        assert!(
                            got[rows.clone()]
                                .iter()
                                .all(|v| v.is_finite() && *v != SENT),
                            "{case}: slot {s} (expert {place}) holds no finite row the run wrote"
                        );
                    }
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
                    got[slots * N..].iter().all(|v| *v == SENT),
                    "{case}: the leg wrote past its run's slots"
                );
            }
        }
    }
    /// The buffers of each activation form hold the bytes [`leg_bytes_for`]
    /// counts for their shape.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_buffers_hold_the_counted_bytes_of_each_form() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let shape = LegShape {
            cap: CAP,
            slots_per_col: SPC,
            n: N,
            ff: FF,
        };
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
        let w = weights(&gpu, [ty, GgmlType::Q5_K]);
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
        let shape = LegShape {
            cap: CAP,
            slots_per_col: SPC,
            n: N,
            ff: FF,
        };
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
}
