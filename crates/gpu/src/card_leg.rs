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
//! tier's rows beside the card's, another slot count), and the routing
//! weights.
//!
//! The run launches the entries [`kernels::card_gate_up_runs`] and
//! [`kernels::card_down_runs`] admit, each stack by its role: those whose
//! launchers read an activation of more columns than a run uses (a
//! [`LegBufs`] sized for `cap` tokens serves a unit of fewer in one launch).
//! A stack of any other type is refused by name when the stacks are read
//! ([`LegStacks::of`]), and an entry the table names but the run has no arm
//! for is refused by name at the launch.

use crate::card_slots::{CardAcc, CardSlotKernels};
use crate::kquant::{Act, GateUpAct, KquantKernels, SelDown};
use crate::mxfp4_sel::Mxfp4SelKernels;
use crate::q4k_sel::QuantSel;
use crate::weights::{DevWeight, Weights};
use crate::{DeviceTensor, Gpu, GpuError, Q8Act};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::GgmlType;
use model::placement::kernels::{self, DownAct, DownEntry, GateUpEntry};
use runtime::combine::Slots;

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
/// over the slots whose place is below `n_card`; `slots_per_col` is 6 or 8,
/// any other count is refused by name before the run launches anything.
pub struct LegSum<'a> {
    pub w: &'a DeviceBuffer<f32>,
    pub acc: &'a mut DeviceBuffer<f32>,
}

/// The modules the leg's launches run on.
pub struct CardLeg {
    kquant: KquantKernels,
    mxfp4: Mxfp4SelKernels,
    slots: CardSlotKernels,
}

impl CardLeg {
    /// The modules on `gpu`. Load-time only.
    pub fn load(gpu: &Gpu) -> Result<CardLeg, GpuError> {
        Ok(CardLeg {
            kquant: KquantKernels::load(gpu.context(), gpu.fault_word())?,
            mxfp4: Mxfp4SelKernels::load(gpu.context(), gpu.fault_word())?,
            slots: CardSlotKernels::load(gpu.context())?,
        })
    }

    /// Enqueue on `gpu`'s stream the four launches (module doc) of `run` into
    /// `bufs`, and with `sum` the card slots' sum after them; `run.m` outside
    /// `1..=bufs.shape().cap`, or a sum over a slot count the sum has no entry
    /// for, is refused by name before the first launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        gpu: &Gpu,
        bufs: &mut LegBufs,
        run: &LegRun<'_>,
        sum: Option<LegSum<'_>>,
    ) -> Result<(), GpuError> {
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
        if sum.is_some() {
            Slots::of(slots_per_col).map_err(|e| GpuError::Shape {
                what: WHAT,
                detail: format!("the card sum: {e}"),
            })?;
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
            Some(GateUpEntry::Q4k) => {
                self.kquant
                    .enqueue_gate_up_q4k(stream, &a, fault, &mut bufs.h)?
            }
            Some(GateUpEntry::Q5k) => {
                self.kquant
                    .enqueue_gate_up_q5k(stream, &a, fault, &mut bufs.h)?
            }
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
            Some((DownEntry::Q4k, DownAct::Q8Act)) => gpu.q4k_sel().enqueue_gemv_q4k_sel(
                stream,
                d.w,
                d.act,
                d.sel,
                d.n_slots,
                d.rows_per_expert,
                &mut bufs.down,
            )?,
            Some((DownEntry::Q5k, DownAct::Q8Act)) => {
                self.kquant
                    .enqueue_gemv_q5k_sel(stream, &d, fault, &mut bufs.down)?
            }
            Some((DownEntry::Mxfp4, DownAct::Q8Act)) => {
                self.mxfp4.enqueue_down(stream, &d, fault, &mut bufs.down)?
            }
            _ => return Err(st.down.unrun()),
        }
        let Some(LegSum { w, acc }) = sum else {
            return Ok(());
        };
        let a = CardAcc {
            down: &bufs.down,
            w,
            sel: run.sel,
            n,
            m: run.m,
            n_card: run.n_card,
            n_used: slots_per_col,
        };
        self.slots.enqueue_card_acc(stream, &a, acc)
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
        let dir = std::env::temp_dir().join(format!("bloomery-card-leg-{}", std::process::id()));
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
}
