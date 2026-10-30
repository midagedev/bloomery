//! The routed batch a prompt walk's card experts run between the model's
//! places launch and its card sum: [`enqueue_gemm_batch`] fills the remapped
//! route table over the batch's ids and the slot map's row, runs the gate and
//! up GEMMs, the SwiGLU of the card slots' places and the down GEMM — the
//! same sequence over the same kernel sets for every model with a remapped
//! card route. The places that write `sel` and the buffers stay the model's;
//! the batch reads `sel` and writes the down rows the model's sum reads.

use super::{
    GateUp32Args, Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32, GemmArgs, GemmInput,
    GemmKernels, GemmRoute, GemmWeight,
};
use crate::GpuError;
use crate::fault::FaultSink;
use crate::kquant::Act;
use crate::tensor::{DeviceTensor, WindowMut};
use cuda_core::{CudaStream, DeviceBuffer};

/// A layer's gate and up as the route's GEMMs read them (the card leg's
/// `StackW::route` names them for a stack), each with the run's normed rows
/// in the form it reads: the grouped GEMM's words over the q8_1 rows, or the
/// 32-value GEMM's weights over the 32-value rows.
#[derive(Clone, Copy)]
pub enum GateUp<'a> {
    Grouped(GemmWeight, [&'a DeviceTensor<u32>; 2], &'a GemmAct),
    Blocks32([Gemm32Weight<'a>; 2], &'a GemmAct32),
}

/// The down stack as the batch's down GEMM reads it, carrying everything
/// that GEMM's launch owns — the activation its SwiGLU writes and the rows
/// it stores — so the two forms cannot be swapped: the 32-value GEMM's
/// weight over the 32-value quantizer's rows, into a run's window of the
/// model's down rows ([`DownWin`]); or a K-quant stack through the grouped
/// GEMM (`rows_per_expert` the batch's `hidden`) over the [`GemmAct`]
/// `GemmKernels::enqueue_swiglu_act_quant_sel` writes from the gate/up rows
/// under `rule`, into F32 rows (the grouped GEMM's `y` is a device buffer,
/// slot-major — no BF16 form).
pub enum DownW<'a> {
    Blocks32(Gemm32Weight<'a>, &'a mut GemmAct32, &'a mut DownWin),
    Grouped(
        GemmWeight,
        &'a DeviceTensor<u32>,
        Act,
        &'a mut GemmAct,
        &'a mut DeviceBuffer<f32>,
    ),
}

/// A run's window of a card route's down rows: F32, or the BF16 bits a
/// native FP4 down product stores.
pub enum DownWin {
    F32(WindowMut<'static, f32>),
    Bf16(WindowMut<'static, u16>),
}

/// One GEMM batch of a card route's pass over `map` — the model's places
/// already in `sel` — with the two kernel sets `(gemm, g32)`: the remapped
/// route table from `ids` over `map`, the gate and up GEMMs of `gate_up`
/// (each slot reading its token's input column, `top_k` slots a token), the
/// SwiGLU over `sel` (every place below `n_card` a column it writes) in the
/// form `down` carries — a 32-value down the 32-value quantizer's rows (its
/// NVFP4 gate·up folds it into its own launch,
/// `Gemm32Kernels::enqueue_gate_up_sel32`), a grouped down the K-quant
/// quantizer's under the case's rule — and the down GEMM of `down` into the
/// rows its case holds: `ff` rows a gate and up expert, `hidden` a down
/// expert. Five launches, four with the fold; only the batch's slots are
/// touched, so a pass's batches write the rows the whole pass always wrote,
/// the same values in the same order — the bits do not move.
#[allow(
    clippy::too_many_arguments,
    reason = "the two GEMM kernel sets, the ids and map, the weights, the widths and places bound, the sink and the buffers (rust-quality R8)"
)]
pub fn enqueue_gemm_batch(
    (gemm, g32): (&GemmKernels, &Gemm32Kernels),
    stream: &CudaStream,
    ids: &DeviceBuffer<u32>,
    map: &DeviceBuffer<u32>,
    gate_up: GateUp<'_>,
    down: DownW<'_>,
    (n, n_card): (usize, usize),
    (top_k, ff, hidden): (usize, usize, usize),
    sink: FaultSink,
    (g, u): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    (sel, route): (&DeviceBuffer<u32>, &mut GemmRoute),
) -> Result<(), GpuError> {
    // The 32-value gate·up fold writes a `GemmAct32`; the grouped GEMM reads
    // the `GemmAct` the sel quantizer writes.
    if let (GateUp::Blocks32(..), DownW::Grouped(..)) = (&gate_up, &down) {
        return Err(GpuError::shape(
            "gemm::enqueue_gemm_batch",
            "a Blocks32 gate·up over a Grouped down: the fold's GemmAct32 rows are not the \
             grouped GEMM's input",
        ));
    }
    g32.enqueue_route_remap(stream, ids, map, n * top_k, route, sink)?;
    let input = GemmInput::Shared { top_k };
    if let GateUp::Grouped(ty, [gate, up], x) = gate_up {
        for (w, y) in [(gate, &mut *g), (up, &mut *u)] {
            gemm.enqueue_gemm(
                stream,
                GemmArgs {
                    ty,
                    w,
                    rows_per_expert: ff,
                    act: x,
                    route,
                    input,
                    y,
                },
            )?;
        }
    }
    match down {
        DownW::Blocks32(w, act, y) => {
            match gate_up {
                GateUp::Grouped(..) => {
                    g32.enqueue_swiglu_quant32_sel(stream, g, u, sel, n_card, n * top_k, act, sink)?
                }
                GateUp::Blocks32(w, x) => g32.enqueue_gate_up_sel32(
                    stream,
                    GateUp32Args {
                        w,
                        rows_per_expert: ff,
                        x,
                        route,
                        input,
                        sel,
                        n_card,
                        g,
                        u,
                        act,
                    },
                    sink,
                )?,
            }
            macro_rules! down_args {
                ($y:expr) => {
                    Gemm32Args {
                        w,
                        rows_per_expert: hidden,
                        act,
                        route,
                        input: GemmInput::PerSlot,
                        y: $y,
                    }
                };
            }
            match y {
                DownWin::F32(y) => g32.enqueue_gemm32(stream, down_args!(y)),
                DownWin::Bf16(y) => g32.enqueue_gemm32_bf16(stream, down_args!(y)),
            }
        }
        DownW::Grouped(ty, w, rule, act, y) => {
            match gate_up {
                GateUp::Grouped(..) => gemm.enqueue_swiglu_act_quant_sel(
                    stream,
                    g,
                    u,
                    rule,
                    sel,
                    n_card,
                    n * top_k,
                    act,
                    sink,
                )?,
                GateUp::Blocks32(..) => unreachable!("refused before the batch's first launch"),
            }
            gemm.enqueue_gemm(
                stream,
                GemmArgs {
                    ty,
                    w,
                    rows_per_expert: hidden,
                    act,
                    route,
                    input: GemmInput::PerSlot,
                    y,
                },
            )
        }
    }
}
