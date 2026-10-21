//! The wide arm of Raon-1's walk: a unit of more than [`PASS_ROWS`] rows, up
//! to [`WIDE_ROWS`], the prompt's bulk. One layer walk as the passes run, each
//! op choosing its launch family from the rows (`qwen3moe::wide`'s shape): the
//! projections through the grouped int8 tensor-core GEMM over a one-expert
//! table ([`site::gemm`]) — a K-quant over q8_1 blocks of 128, a Q8_0 over q8
//! blocks of 32, an F32 site on the F32 tile — the routed experts through one
//! route table over the unit's slots, the router, the conv, the delta step, the
//! norms and the card sum as the passes launch them (they take any row count).
//!
//! A unit's rows are not each the one-row step's bits: the GEMM sums 128-value
//! integer blocks into one f32 accumulator where the gemv sums lane partials
//! through a warp tree. What stays the step's: the router (the expert ids), the
//! card sum's order and every element-wise op. A unit of up to [`PASS_ROWS`]
//! rows is the pass of [`super::rows`], bit for bit the steps.
//!
//! The pieces by owner:
//! - the RDA layer: [`rda_rows`], the common mixer
//!   ([`crate::linear::mixer_kda`]) over [`RdaWide`]'s projections;
//! - the latent layer: [`super::mla::rows_wide`], the projections wide, the
//!   cache append, the gather, the per-head key absorption ([`absorb_heads`])
//!   and value projection ([`value_heads`]) and the output gate over the
//!   unit, the latent flash in chunks of [`super::mla::ATTN_CHUNK`] query
//!   tokens (the partials' size bounds them);
//! - the feed-forward block: [`ffn`], below.

use super::body::Scratch;
use super::ffn::{Ffn, Glu, Routed, sigmoid_128_6};
use super::mhc::{MAX_TOKENS, STREAMS};
use super::rda::RdaSites;
use super::rows::RowsIn;
use super::state::PASS_ROWS;
use crate::arch::hybrid::{GqaCx, Site};
use crate::card_leg::{RouteW, StackW};
use crate::card_slots::{CardAcc, CardSlotKernels};
use crate::fault::FaultSink;
use crate::gemm::{
    Gemm32Args, Gemm32Kernels, GemmAct, GemmAct32, GemmArgs, GemmInput, GemmKernels, GemmRoute,
};
use crate::kquant::Act;
use crate::latent::LATENT;
use crate::linear::mixer_kda::{self, KdaBufs, KdaDims, KdaIn, KdaProj, KdaState};
use crate::linear::{HEAD, LinearKernels, LinearShape};
use crate::model::fit::{act_col_bytes, act32_col_bytes, route_bytes};
use crate::site::{self, Gated, GatedIo, SiteTy, SiteWeight, SwigluRows, WideActs, WideIn};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::GgmlType;
use std::sync::Arc;

/// What the refusals of this file name.
const WHAT: &str = "raon1 wide";

/// The most rows a wide unit holds: its buffers' rows. At most the mHC route
/// launch's tokens ([`MAX_TOKENS`]).
pub(super) const WIDE_ROWS: usize = 256;

/// The fewest rows of a unit that runs wide; fewer run as the pass does.
pub(super) const WIDE_FROM: usize = PASS_ROWS + 1;

const _: () = assert!(WIDE_ROWS <= MAX_TOKENS && WIDE_ROWS >= WIDE_FROM);

#[cuda_module]
mod wide_kernels {
    use super::*;

    /// The joined q·k·v rows of the conv: row `t` of `y` is `q[t]`, `k[t]`,
    /// `v[t]` end to end (`q` and `k` `wq` values a row, `v` `wv`) — a copy,
    /// so exact. One thread per value.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            q.len() >= wq * m,
            k.len() >= wq * m,
            v.len() >= wv * m,
            y.len() >= (2 * wq + wv) * m
        )
    )]
    pub fn raon_join_qkv(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        wq: u32,
        wv: u32,
        m: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let (wq, wv, m) = (wq as usize, wv as usize, m as usize);
        let row = 2 * wq + wv;
        if i >= row * m {
            return;
        }
        let t = i / row;
        let c = i - t * row;
        // SAFETY: i < row·m, so t < m and c < row; the source index of each
        // arm is below its buffer's m rows by the launch contract; thread i is
        // y[i]'s only writer.
        let x = unsafe {
            if c < wq {
                *q.get_unchecked(t * wq + c)
            } else if c < 2 * wq {
                *k.get_unchecked(t * wq + (c - wq))
            } else {
                *v.get_unchecked(t * wv + (c - 2 * wq))
            }
        };
        // SAFETY: i < row·m <= y.len() by the launch contract.
        unsafe { *y.get_unchecked_mut(i) = x };
    }

    /// `rows` windows of `width` values copied between two row-strided
    /// buffers: `y[r·y_stride + y_off + c] = x[r·x_stride + x_off + c]` for
    /// `r < rows`, `c < width` — a copy, so exact. One thread per value; the
    /// host holds each window inside its stride, so no two threads store one
    /// slot.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= rows * x_stride, y.len() >= rows * y_stride)
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
    )]
    pub fn raon_row_window(
        x: &[f32],
        x_stride: u32,
        x_off: u32,
        width: u32,
        rows: u32,
        y_stride: u32,
        y_off: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let (width, rows) = (width as usize, rows as usize);
        if i >= width * rows {
            return;
        }
        let r = i / width;
        let c = i - r * width;
        // SAFETY: r < rows and c < width, with x_off + width <= x_stride and
        // y_off + width <= y_stride (host-checked), put both indices below
        // rows·stride, inside x and y by the launch contract; thread i is the
        // only writer of its (r, c) slot of y.
        unsafe {
            *y.get_unchecked_mut(r * y_stride as usize + y_off as usize + c) =
                *x.get_unchecked(r * x_stride as usize + x_off as usize + c);
        }
    }
}

/// The module of the join and of the window copy.
pub(super) struct JoinKernels {
    module: wide_kernels::LoadedModule,
}

impl JoinKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub(super) fn load(ctx: &Arc<CudaContext>) -> Result<JoinKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(wide_kernels, ctx)? };
        Ok(JoinKernels { module })
    }

    /// Enqueue the join of `m` rows of `q` and `k` (`wq` values a row) and
    /// `v` (`wv`) into `y`'s `m` rows of `2 · wq + wv`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the three parts, their widths, the rows and the joined rows (rust-quality R8)"
    )]
    pub(super) fn enqueue_join_qkv(
        &self,
        stream: &CudaStream,
        (q, k, v): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, &DeviceBuffer<f32>),
        (wq, wv): (usize, usize),
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 wide join";
        let row = 2 * wq + wv;
        if m == 0 || q.len() < wq * m || k.len() < wq * m || v.len() < wv * m || y.len() < row * m {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} rows of {wq}+{wq}+{wv}: q {}, k {}, v {}, y {}",
                    q.len(),
                    k.len(),
                    v.len(),
                    y.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (row * m).div_ceil(256))?;
        let (wq, wv, m) = (
            launch_u32(what, "wq", wq)?,
            launch_u32(what, "wv", wv)?,
            launch_u32(what, "m", m)?,
        );
        let prep = self
            .module
            .prepare_raon_join_qkv(LaunchConfig1D::new(grid, 256, 0))?;
        self.module
            .raon_join_qkv(stream, &prep, q, k, v, wq, wv, m, y)?;
        Ok(())
    }

    /// Enqueue the copy of `rows` windows of `width` values: row `r` of `x`
    /// starts at `r · x_stride + x_off`, row `r` of `y` at `r · y_stride +
    /// y_off`. A window past its stride, or a buffer short of its rows, is
    /// refused by name.
    pub(super) fn enqueue_window(
        &self,
        stream: &CudaStream,
        (x, x_stride, x_off): (&DeviceBuffer<f32>, usize, usize),
        (width, rows): (usize, usize),
        (y, y_stride, y_off): (&mut DeviceBuffer<f32>, usize, usize),
    ) -> Result<(), GpuError> {
        let what = "raon1 wide window";
        if width == 0
            || rows == 0
            || x_off + width > x_stride
            || y_off + width > y_stride
            || x.len() < rows * x_stride
            || y.len() < rows * y_stride
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{rows} windows of {width}: x stride {x_stride} from {x_off} over {} values, \
                     y stride {y_stride} from {y_off} over {}",
                    x.len(),
                    y.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (width * rows).div_ceil(256))?;
        let (x_stride, x_off, width, rows, y_stride, y_off) = (
            launch_u32(what, "x_stride", x_stride)?,
            launch_u32(what, "x_off", x_off)?,
            launch_u32(what, "width", width)?,
            launch_u32(what, "rows", rows)?,
            launch_u32(what, "y_stride", y_stride)?,
            launch_u32(what, "y_off", y_off)?,
        );
        let prep = self
            .module
            .prepare_raon_row_window(LaunchConfig1D::new(grid, 256, 0))?;
        self.module.raon_row_window(
            stream, &prep, x, x_stride, x_off, width, rows, y_stride, y_off, y,
        )?;
        Ok(())
    }
}

/// The activation forms a wide unit's sites read, one buffer per input the
/// walk quantizes: each holds the unit's rows (or slots) of its width.
pub(super) struct Forms {
    /// The normed rows: the model width, in q8_1 blocks of 128 (the K-quant
    /// sites) and of 32 (a Q8_0 site).
    hid128: GemmAct,
    hid32: GemmAct32,
    /// The mixers' gated rows into the output projection, `heads · v_head`.
    out32: GemmAct32,
    /// The latent layer's normed query latent into `attn_q_b`.
    qb32: GemmAct32,
    /// The dense block's SiTU rows into its down, the dense width.
    dense128: GemmAct,
    /// The shared expert's SiTU rows into its down, the shared width.
    shared32: GemmAct32,
    /// The routed experts' SiTU rows, a column a slot, into the stacks' down.
    slots32: GemmAct32,
    /// The latent attention's output, a column a (row, head), into the value
    /// projection.
    lat128: GemmAct,
    /// The queries' nope parts, a column a (row, head), into the key
    /// absorption.
    nope32: GemmAct32,
}

/// The widths a wide unit is cut for.
#[derive(Clone, Copy)]
pub(super) struct WideDims {
    pub embd: usize,
    /// `heads · v_head`: the output projections' input.
    pub attn: usize,
    pub q_lora: usize,
    /// The latent layers' heads, and the nope values of a query head the key
    /// absorption reads.
    pub heads: usize,
    pub nope: usize,
    pub dense_ff: usize,
    pub shared_ff: usize,
    pub expert_ff: usize,
    /// The widest feed-forward block's gate rows a unit holds: the dense
    /// block's, or the routed slots'.
    pub ff_rows: usize,
}

impl Forms {
    fn new(stream: &CudaStream, d: &WideDims) -> Result<Forms, GpuError> {
        let rows = WIDE_ROWS;
        let slots = rows * sigmoid_128_6::N_USED;
        Ok(Forms {
            hid128: GemmAct::new(stream, rows, d.embd)?,
            hid32: GemmAct32::new(stream, rows, d.embd)?,
            out32: GemmAct32::new(stream, rows, d.attn)?,
            qb32: GemmAct32::new(stream, rows, d.q_lora)?,
            dense128: GemmAct::new(stream, rows, d.dense_ff)?,
            shared32: GemmAct32::new(stream, rows, d.shared_ff)?,
            slots32: GemmAct32::new(stream, slots, d.expert_ff)?,
            lat128: GemmAct::new(stream, rows * d.heads, LATENT)?,
            nope32: GemmAct32::new(stream, rows * d.heads, d.nope)?,
        })
    }

    fn bytes_of(d: &WideDims) -> usize {
        let rows = WIDE_ROWS;
        let slots = rows * sigmoid_128_6::N_USED;
        rows * (act_col_bytes(d.embd) + act_col_bytes(d.dense_ff))
            + rows * (act32_col_bytes(d.embd) + act32_col_bytes(d.attn))
            + rows * (act32_col_bytes(d.q_lora) + act32_col_bytes(d.shared_ff))
            + slots * act32_col_bytes(d.expert_ff)
            + rows * d.heads * act_col_bytes(LATENT)
            + rows * d.heads * act32_col_bytes(d.nope)
    }
}

/// The feed-forward block's buffers over a unit's rows, token-major.
pub(super) struct WideFfn {
    /// The gate and up rows of the dense block, the shared expert (a row
    /// each) and the routed slots (a slot each): `ff_rows` values a row.
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    routed: DeviceBuffer<f32>,
    /// The slots' down rows, `embd` a slot.
    down: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    /// The route table over the unit's slots and the experts.
    moe: GemmRoute,
}

impl WideFfn {
    fn new(stream: &CudaStream, d: &WideDims) -> Result<WideFfn, GpuError> {
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n.max(1));
        let (rows, used) = (WIDE_ROWS, sigmoid_128_6::N_USED);
        Ok(WideFfn {
            g: z(rows * d.ff_rows)?,
            u: z(rows * d.ff_rows)?,
            shared: z(rows * d.embd)?,
            routed: z(rows * d.embd)?,
            down: z(rows * used * d.embd)?,
            probs: z(rows * sigmoid_128_6::N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, rows * used)?,
            weights: z(rows * used)?,
            moe: GemmRoute::new(stream, rows * used, sigmoid_128_6::N_EXPERT)?,
        })
    }

    fn bytes_of(d: &WideDims) -> usize {
        let (rows, used) = (WIDE_ROWS, sigmoid_128_6::N_USED);
        let f32s = 2 * rows * d.ff_rows
            + 2 * rows * d.embd
            + rows * used * d.embd
            + rows * sigmoid_128_6::N_EXPERT
            + rows * used;
        4 * (f32s + rows * used) + route_bytes(rows * used, sigmoid_128_6::N_EXPERT)
    }
}

/// The latent layers' per-head projections' table: a slot a (row, head), slot
/// `s` routed to expert `s % heads`, the head's rows of the stack, shared by
/// the key absorption and the value projection; and the rows between the
/// absorption's steps.
pub(super) struct HeadSlots {
    heads: usize,
    ids: DeviceBuffer<u32>,
    route: GemmRoute,
    /// The absorption's input, the queries' nope values a slot, and its output,
    /// [`LATENT`] values a slot.
    nope: DeviceBuffer<f32>,
    lat: DeviceBuffer<f32>,
}

impl HeadSlots {
    fn new(stream: &CudaStream, heads: usize, nope: usize) -> Result<HeadSlots, GpuError> {
        let slots = WIDE_ROWS * heads;
        let ids = (0..slots)
            .map(|s| launch_u32(WHAT, "head", s % heads))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(HeadSlots {
            heads,
            ids: DeviceBuffer::from_host(stream, &ids)?,
            route: GemmRoute::new(stream, slots, heads)?,
            nope: DeviceBuffer::zeroed(stream, slots * nope)?,
            lat: DeviceBuffer::zeroed(stream, slots * LATENT)?,
        })
    }

    fn bytes_of(heads: usize, nope: usize) -> usize {
        let slots = WIDE_ROWS * heads;
        4 * slots + route_bytes(slots, heads) + 4 * slots * (nope + LATENT)
    }

    fn bytes(&self) -> usize {
        self.ids.num_bytes() + self.route.bytes() + self.nope.num_bytes() + self.lat.num_bytes()
    }
}

/// A wide unit's own buffers and kernels beyond the passes': the staging, the
/// walk's rows, the forms and the feed-forward block's buffers, the join's
/// parts and the GEMM family.
pub(super) struct Wide {
    pub(super) rin: RowsIn,
    pub(super) s: Scratch,
    ffn: WideFfn,
    forms: Forms,
    /// `q`, `k` and `v`'s rows before the join, `attn` values a row each.
    parts: [DeviceBuffer<f32>; 3],
    /// The one-expert table every projection of a unit reads.
    dense: GemmRoute,
    heads: HeadSlots,
    gemm: GemmKernels,
    join: JoinKernels,
    slots: CardSlotKernels,
}

/// The kernels and the table a wide launch reads.
#[derive(Clone, Copy)]
pub(super) struct Kern<'a> {
    pub(super) gemm: &'a GemmKernels,
    pub(super) g32: &'a Gemm32Kernels,
    pub(super) join: &'a JoinKernels,
    pub(super) slots: &'a CardSlotKernels,
    pub(super) router: &'a sigmoid_128_6::RouterKernels,
    pub(super) dense: &'a GemmRoute,
}

/// The buffers a wide launch writes beyond a layer's own.
pub(super) struct Bufs<'a> {
    pub(super) forms: &'a mut Forms,
    pub(super) parts: &'a mut [DeviceBuffer<f32>; 3],
    pub(super) ffn: &'a mut WideFfn,
    pub(super) heads: &'a mut HeadSlots,
}

/// What a wide launch borrows beyond a layer's buffers.
pub(super) struct WideKit<'a> {
    pub(super) k: Kern<'a>,
    pub(super) b: Bufs<'a>,
}

impl Wide {
    /// The unit's buffers for `d`, and the kernels of the GEMM family, over
    /// a walk's `scratch` of [`WIDE_ROWS`] rows. Load-time only.
    pub(super) fn new(gpu: &Gpu, d: &WideDims, scratch: Scratch) -> Result<Wide, GpuError> {
        let (stream, ctx) = (gpu.stream(), gpu.context());
        let part = || DeviceBuffer::<f32>::zeroed(stream, WIDE_ROWS * d.attn);
        Ok(Wide {
            rin: RowsIn::new(stream, WIDE_ROWS)?,
            s: scratch,
            ffn: WideFfn::new(stream, d)?,
            forms: Forms::new(stream, d)?,
            parts: [part()?, part()?, part()?],
            dense: GemmRoute::new(stream, WIDE_ROWS, 1)?,
            heads: HeadSlots::new(stream, d.heads, d.nope)?,
            gemm: GemmKernels::load(ctx)?,
            join: JoinKernels::load(ctx)?,
            slots: CardSlotKernels::load(ctx)?,
        })
    }

    /// Card bytes of what [`Wide::new`] allocates beyond the walk's rows.
    pub(super) fn bytes_of(d: &WideDims) -> usize {
        let stage = (STREAMS + 2) * WIDE_ROWS * size_of::<u32>() + 2 * STREAMS * WIDE_ROWS * 4;
        Forms::bytes_of(d)
            + WideFfn::bytes_of(d)
            + 3 * 4 * WIDE_ROWS * d.attn
            + route_bytes(WIDE_ROWS, 1)
            + HeadSlots::bytes_of(d.heads, d.nope)
            + stage
    }

    /// Card bytes held, the walk's rows included.
    pub(super) fn bytes(&self) -> usize {
        self.s.bytes()
            + self.rin.bytes()
            + self.forms.bytes()
            + self.ffn.bytes()
            + self
                .parts
                .iter()
                .map(DeviceBuffer::num_bytes)
                .sum::<usize>()
            + self.dense.bytes()
            + self.heads.bytes()
    }

    /// The unit's one-expert table for `m` rows, once behind the embedding.
    pub(super) fn route_dense(&mut self, gpu: &Gpu, m: usize) -> Result<(), GpuError> {
        self.gemm
            .enqueue_route_dense(gpu.stream(), m, &mut self.dense, gpu.unlabelled_sink())
    }

    /// The kit over this unit's pieces beside the passes' `g32` and router,
    /// the walk's rows and the staging.
    pub(super) fn kit<'a>(
        &'a mut self,
        g32: &'a Gemm32Kernels,
        router: &'a sigmoid_128_6::RouterKernels,
    ) -> (WideKit<'a>, &'a mut Scratch, &'a mut RowsIn) {
        let Wide {
            rin,
            s,
            ffn,
            forms,
            parts,
            dense,
            heads,
            gemm,
            join,
            slots,
        } = self;
        let k = Kern {
            gemm,
            g32,
            join,
            slots,
            router,
            dense,
        };
        (
            WideKit {
                k,
                b: Bufs {
                    forms,
                    parts,
                    ffn,
                    heads,
                },
            },
            s,
            rin,
        )
    }
}

impl Forms {
    fn bytes(&self) -> usize {
        self.hid128.bytes()
            + self.hid32.bytes()
            + self.out32.bytes()
            + self.qb32.bytes()
            + self.dense128.bytes()
            + self.shared32.bytes()
            + self.slots32.bytes()
            + self.lat128.bytes()
            + self.nope32.bytes()
    }
}

impl WideFfn {
    fn bytes(&self) -> usize {
        [
            &self.g,
            &self.u,
            &self.shared,
            &self.routed,
            &self.down,
            &self.probs,
            &self.weights,
        ]
        .into_iter()
        .map(DeviceBuffer::num_bytes)
        .sum::<usize>()
            + self.ids.num_bytes()
            + self.moe.bytes()
    }
}

/// Which of the unit's forms a projection reads.
#[derive(Clone, Copy)]
pub(super) enum Input {
    /// The normed rows' forms ([`quantize_hid`]).
    Hid,
    /// The gated rows into an output projection ([`quantize_out`]).
    Out,
    /// The query latent into `attn_q_b` ([`quantize_qb`]).
    Qb,
    /// The queries' nope parts into the key absorption ([`absorb_heads`]).
    Nope,
    /// The f32 rows alone (an F32 site).
    Plain,
}

/// The input `x` of a projection as `which` reads it.
pub(super) fn input<'f>(forms: &'f Forms, which: Input, x: &'f DeviceBuffer<f32>) -> WideIn<'f> {
    let (q128, q32) = match which {
        Input::Hid => (Some(&forms.hid128), Some(&forms.hid32)),
        Input::Out => (None, Some(&forms.out32)),
        Input::Qb => (None, Some(&forms.qb32)),
        Input::Nope => (None, Some(&forms.nope32)),
        Input::Plain => (None, None),
    };
    WideIn {
        q128,
        q32,
        f32: Some(x),
    }
}

impl Kern<'_> {
    fn kernels(&self) -> site::WideKernels<'_> {
        site::WideKernels {
            gemm: self.gemm,
            g32: Some(self.g32),
        }
    }

    /// Site `t` (`rows` outputs) over the unit's `m` rows of `x` in the forms
    /// `x_in` carries, into `y`.
    pub(super) fn project(
        &self,
        cx: &GqaCx<'_>,
        (t, rows): (&Site, usize),
        x_in: &WideIn<'_>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        site::gemm(
            cx.gpu,
            &self.kernels(),
            (t.ty(), cx.w, SiteWeight::Resident(t.name())),
            rows,
            x_in,
            (self.dense, GemmInput::PerSlot),
            m,
            y,
        )
    }
}

/// The normed rows `x` (`m` of the model width) in the forms asked for:
/// `(q128, q32)`, as [`WideActs::wants_of`] names them for a set of sites.
pub(super) fn quantize_hid(
    gpu: &Gpu,
    g32: &Gemm32Kernels,
    forms: &mut Forms,
    (x, m): (&DeviceBuffer<f32>, usize),
    (sink, (q128, q32)): (FaultSink, (bool, bool)),
) -> Result<(), GpuError> {
    if q128 {
        gpu.enqueue_quantize_gemm(x, m, &mut forms.hid128, sink)?;
    }
    if q32 {
        g32.enqueue_quantize_gemm32(gpu.stream(), x, m, &mut forms.hid32, sink)?;
    }
    Ok(())
}

/// The head table ([`HeadSlots`]) filled for a unit's `m` rows: `m · heads`
/// slots, which [`absorb_heads`] and [`value_heads`] read.
pub(super) fn route_heads(
    gpu: &Gpu,
    (k, b): (&Kern<'_>, &mut Bufs<'_>),
    (m, sink): (usize, FaultSink),
) -> Result<(), GpuError> {
    let h = &mut *b.heads;
    k.gemm
        .enqueue_route(gpu.stream(), &h.ids, m * h.heads, &mut h.route, sink)
}

/// A named refusal of a head table not filled for exactly `slots` slots.
fn heads_filled(route: &GemmRoute, slots: usize) -> Result<(), GpuError> {
    if route.filled() == Some(slots) {
        return Ok(());
    }
    Err(GpuError::shape(
        WHAT,
        format!(
            "the head table filled for the unit's {slots} slots (route_heads); it is filled for \
             {:?}",
            route.filled()
        ),
    ))
}

/// The widths of a latent layer's queries the key absorption reads: a head's
/// query row is `head_k` values, its nope part the first `nope`, and the
/// absorbed row `row` values with the latent last.
#[derive(Clone, Copy)]
pub(super) struct AbsorbShape {
    pub head_k: usize,
    pub nope: usize,
    pub row: usize,
}

/// The latent layers' per-head key absorption over a unit's `m` rows: the nope
/// part of each (row, head) slot of `q` in q8 blocks of 32, a slot routed to
/// its head ([`route_heads`]), times `k_b`, a Q8_0 stack of `heads` experts of
/// [`LATENT`] rows (`attn_k_b`), one 32-value GEMM, the result into each slot's
/// absorbed row `qabs` after its first `row − LATENT` values, which the caller
/// has laid there.
pub(super) fn absorb_heads(
    gpu: &Gpu,
    (k, b): (&Kern<'_>, &mut Bufs<'_>),
    (w, k_b): (&Weights, &str),
    (q, m): (&DeviceBuffer<f32>, usize),
    (shape, sink, qabs): (AbsorbShape, FaultSink, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let AbsorbShape { head_k, nope, row } = shape;
    let (stream, h) = (gpu.stream(), &mut *b.heads);
    let slots = m * h.heads;
    heads_filled(&h.route, slots)?;
    k.join.enqueue_window(
        stream,
        (q, head_k, 0),
        (nope, slots),
        (&mut h.nope, nope, 0),
    )?;
    k.g32
        .enqueue_quantize_gemm32(stream, &h.nope, slots, &mut b.forms.nope32, sink)?;
    let x_in = input(b.forms, Input::Nope, &h.nope);
    site::gemm(
        gpu,
        &k.kernels(),
        (SiteTy::Q8_0, w, SiteWeight::Resident(k_b)),
        LATENT,
        &x_in,
        (&h.route, GemmInput::PerSlot),
        slots,
        &mut h.lat,
    )?;
    k.join.enqueue_window(
        stream,
        (&h.lat, LATENT, 0),
        (LATENT, slots),
        (qabs, row, row - LATENT),
    )
}

/// The latent layers' per-head value projection over a unit's `m` rows: `att`
/// (`m · heads` rows of [`LATENT`] values, the attention's output) times
/// `wv`, a Q6_K stack of `heads` experts of `rows` rows (`attn_v_b`), into `av`
/// (`m · heads` rows of `rows` values): the rows in q8_1 blocks, a slot a
/// (row, head) routed to its head ([`route_heads`]), one grouped GEMM.
pub(super) fn value_heads(
    gpu: &Gpu,
    (k, b): (&Kern<'_>, &mut Bufs<'_>),
    wv: &DeviceTensor<u32>,
    (att, m): (&DeviceBuffer<f32>, usize),
    (rows, sink, av): (usize, FaultSink, &mut DeviceBuffer<f32>),
) -> Result<(), GpuError> {
    let Some(RouteW::Grouped(ty, w)) = StackW::Words(wv).route(GgmlType::Q6_K) else {
        return Err(GpuError::shape(
            WHAT,
            "the value projection's Q6_K words have no grouped GEMM route",
        ));
    };
    let (stream, heads) = (gpu.stream(), b.heads.heads);
    heads_filled(&b.heads.route, m * heads)?;
    gpu.enqueue_quantize_gemm(att, m * heads, &mut b.forms.lat128, sink)?;
    k.gemm.enqueue_gemm(
        stream,
        GemmArgs {
            ty,
            w,
            rows_per_expert: rows,
            act: &b.forms.lat128,
            route: &b.heads.route,
            input: GemmInput::PerSlot,
            y: av,
        },
    )
}

/// The mixers' gated rows `x` (`m` of `heads · v_head`) in the q8 blocks of 32
/// the output projection of type `ty` reads.
pub(super) fn quantize_out(
    gpu: &Gpu,
    g32: &Gemm32Kernels,
    forms: &mut Forms,
    (x, m): (&DeviceBuffer<f32>, usize),
    (sink, ty): (FaultSink, SiteTy),
) -> Result<(), GpuError> {
    only_blocks32(ty, "an output projection")?;
    g32.enqueue_quantize_gemm32(gpu.stream(), x, m, &mut forms.out32, sink)
}

/// The latent layer's normed query latent `x` in the q8 blocks of 32
/// `attn_q_b` of type `ty` reads.
pub(super) fn quantize_qb(
    gpu: &Gpu,
    g32: &Gemm32Kernels,
    forms: &mut Forms,
    (x, m): (&DeviceBuffer<f32>, usize),
    (sink, ty): (FaultSink, SiteTy),
) -> Result<(), GpuError> {
    only_blocks32(ty, "a query up projection")?;
    g32.enqueue_quantize_gemm32(gpu.stream(), x, m, &mut forms.qb32, sink)
}

/// A named refusal of a projection `what` of type `ty` that reads another
/// form than q8 blocks of 32: the wide arm's forms for those inputs are
/// Q8_0's.
fn only_blocks32(ty: SiteTy, what: &str) -> Result<(), GpuError> {
    let (q128, q32) = WideActs::wants(ty);
    if q128 || !q32 {
        return Err(GpuError::shape(
            WHAT,
            format!("{what} of {ty}: the wide arm's form for its input is q8 blocks of 32 (Q8_0)"),
        ));
    }
    Ok(())
}

/// The RDA layer's wide projections ([`KdaProj`]): q, k, v and the gate over
/// the normed rows' q8_1 form, β's, the write gate's and the forget
/// projection's halves on the F32 tile, the output projection over the gated
/// rows' q8 blocks of 32.
struct RdaWide<'a, 'k> {
    cx: &'a GqaCx<'a>,
    sites: &'a RdaSites,
    shape: LinearShape,
    kit: &'a mut WideKit<'k>,
}

impl KdaProj for RdaWide<'_, '_> {
    /// q, k and v into three row sets joined into the conv's channels, `z`
    /// from the gate projection, β's and the write gate's raw rows, `fa`, then
    /// `f = f_b · fa`; the K-quant sites share one quantization of `xn`.
    fn input(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        y: KdaIn<'_>,
    ) -> Result<(), GpuError> {
        let (cx, s, shape) = (self.cx, self.sites, self.shape);
        let sink = gpu.layer_sink(l)?;
        let (qk, v) = (shape.n_k * HEAD, shape.n_v * HEAD);
        let all = s.sites();
        let tys: Vec<SiteTy> = all[..7].iter().map(|t| t.ty()).collect();
        let KdaIn {
            qkv,
            fa,
            beta_raw,
            write_raw,
            f,
            z,
            ..
        } = y;
        let write_raw = write_raw.ok_or(GpuError::state(
            WHAT,
            "the write projection's rows (the rule has a write gate)",
        ))?;
        let WideKit { k, b } = &mut *self.kit;
        let Bufs { forms, parts, .. } = b;
        quantize_hid(gpu, k.g32, forms, (xn, m), (sink, WideActs::wants_of(&tys)))?;
        let x_in = input(forms, Input::Hid, xn);
        let [pq, pk, pv] = &mut **parts;
        for ((t, rows), y) in [(&s.qkv[0], qk), (&s.qkv[1], qk), (&s.qkv[2], v)]
            .into_iter()
            .zip([&mut *pq, &mut *pk, &mut *pv])
        {
            k.project(cx, (t, rows), &x_in, m, y)?;
        }
        k.project(cx, (&s.g, v), &x_in, m, z)?;
        k.project(cx, (&s.beta, shape.n_v), &x_in, m, beta_raw)?;
        k.project(cx, (&s.write, shape.n_v), &x_in, m, write_raw)?;
        k.project(cx, (&s.f_a, s.f_b.k()), &x_in, m, fa)?;
        let fa_in = input(forms, Input::Plain, &*fa);
        k.project(cx, (&s.f_b, v), &fa_in, m, f)?;
        k.join
            .enqueue_join_qkv(gpu.stream(), (pq, pk, pv), (qk, v), m, qkv)
    }

    /// `out = o_proj · gated` over the unit's rows.
    fn output(
        &mut self,
        gpu: &Gpu,
        l: usize,
        m: usize,
        gated: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (cx, t) = (self.cx, &self.sites.out);
        let sink = gpu.layer_sink(l)?;
        let WideKit { k, b } = &mut *self.kit;
        quantize_out(gpu, k.g32, b.forms, (gated, m), (sink, t.ty()))?;
        let x_in = input(b.forms, Input::Out, gated);
        k.project(cx, (t, cx.hidden), &x_in, m, out)
    }
}

/// Enqueue RDA layer `l` over a wide unit of `m` rows (module doc): the
/// common mixer over [`RdaWide`]'s projections, in position order, the store
/// `st` as the unit's lanes name it.
#[allow(
    clippy::too_many_arguments,
    reason = "the launch context, the unit's kit, the layer and its store and buffers (rust-quality R8)"
)]
pub(super) fn rda_rows(
    cx: &GqaCx<'_>,
    kit: &mut WideKit<'_>,
    linear: &LinearKernels,
    sites: &RdaSites,
    (l, d, m): (usize, &KdaDims, usize),
    st: KdaState<'_>,
    bufs: KdaBufs<'_>,
) -> Result<(), GpuError> {
    let gpu = cx.gpu;
    let vecs = sites.vecs(cx.w)?;
    let mut proj = RdaWide {
        cx,
        sites,
        shape: d.shape,
        kit,
    };
    mixer_kda::enqueue(
        gpu,
        gpu.stream(),
        linear,
        &mut proj,
        (l, m, d),
        &vecs,
        st,
        bufs,
    )
}

/// The stack `name` as the route GEMM reads it: its file type and form.
fn stack<'w>(w: &'w Weights, name: &str) -> Result<(GgmlType, StackW<'w>), GpuError> {
    match w.get(name) {
        Some(DevWeight::KQuant { ty, w, .. }) => Ok((*ty, StackW::Words(w))),
        Some(DevWeight::Q5_0 { w, .. }) => Ok((GgmlType::Q5_0, StackW::Q5_0(w))),
        _ => Err(GpuError::tensor(
            WHAT,
            name,
            "a resident routed stack (the file's blocks as words, or Q5_0 packed)",
        )),
    }
}

/// A dense block (the leading layer's, or a shared expert) over the unit's
/// `m` rows of the normed rows whose forms the caller made: the gate and up
/// GEMMs, the rule's SiTU quantized into the form the down reads, the down
/// GEMM into `y`. `g_buf` and `u_buf` hold the gate and up rows.
#[allow(
    clippy::too_many_arguments,
    reason = "the block's context, rule, rows, forms and buffers (rust-quality R8)"
)]
fn glu(
    cx: &GqaCx<'_>,
    k: &Kern<'_>,
    rule: Act,
    (l, m): (usize, usize),
    (g, forms): (&Glu, &mut Forms),
    (g_buf, u_buf): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let blk = Gated {
        w: cx.w,
        route: k.dense,
        gate: (g.gate.ty(), SiteWeight::Resident(g.gate.name())),
        up: (g.up.ty(), SiteWeight::Resident(g.up.name())),
        down: (g.down.ty(), SiteWeight::Resident(g.down.name())),
        ff: g.ff,
        hidden: cx.hidden,
        rule,
        input: GemmInput::PerSlot,
    };
    let x = WideIn {
        q128: Some(&forms.hid128),
        q32: Some(&forms.hid32),
        f32: None,
    };
    let rows = match WideActs::wants(g.down.ty()) {
        (true, _) if forms.dense128.k() == g.ff => SwigluRows {
            q128: Some(&mut forms.dense128),
            q32: None,
            f32: None,
        },
        (false, true) if forms.shared32.k() == g.ff => SwigluRows {
            q128: None,
            q32: Some(&mut forms.shared32),
            f32: None,
        },
        _ => {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a block of width {} with a down of {}: the wide arm holds q8_1 \
                     rows of the dense width and q8 blocks of 32 of the shared width",
                    g.ff,
                    g.down.ty()
                ),
            ));
        }
    };
    let io = GatedIo {
        x: &x,
        gate: g_buf,
        up: u_buf,
        rows,
        y,
    };
    site::gated(
        cx.gpu,
        &k.kernels(),
        &blk,
        (m, m),
        io,
        cx.gpu.layer_sink(l)?,
    )
}

/// Layer `l`'s block over the unit's `m` rows of `xn` into `y`: the dense
/// block, or the routed sum plus the shared expert ([`super::ffn::ffn`]'s
/// launches over a route table): the router, the gate and up GEMMs over the
/// slots, the SiTU into q8 blocks of 32 a slot, the down GEMM of the stack's
/// type, the card sum in slot order, the shared expert, the add.
#[allow(
    clippy::too_many_arguments,
    reason = "ffn's inputs and the unit's kit (rust-quality R8)"
)]
pub(super) fn ffn(
    cx: &GqaCx<'_>,
    kit: &mut WideKit<'_>,
    (rule, scale): (Act, f32),
    (l, f): (usize, &Ffn),
    m: usize,
    xn: &DeviceBuffer<f32>,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let gpu = cx.gpu;
    let (stream, sink) = (gpu.stream(), gpu.layer_sink(l)?);
    let WideKit { k, b } = kit;
    let Bufs { forms, ffn: wf, .. } = b;
    match f {
        Ffn::Dense(gl) => {
            let want = WideActs::wants_of(&[gl.gate.ty(), gl.up.ty()]);
            quantize_hid(gpu, k.g32, forms, (xn, m), (sink, want))?;
            glu(cx, k, rule, (l, m), (gl, forms), (&mut wf.g, &mut wf.u), y)
        }
        Ffn::Routed(r) => {
            // The routed stacks' gate and up read the q8_1 rows; the shared
            // expert's sites read theirs.
            let (_, q32) = WideActs::wants_of(&[r.shared.gate.ty(), r.shared.up.ty()]);
            quantize_hid(gpu, k.g32, forms, (xn, m), (sink, (true, q32)))?;
            routed(cx, k, rule, (l, m), (r, forms, wf), xn, scale)?;
            glu(
                cx,
                k,
                rule,
                (l, m),
                (&r.shared, forms),
                (&mut wf.g, &mut wf.u),
                &mut wf.shared,
            )?;
            gpu.elem()
                .enqueue_add(stream, &wf.routed, &wf.shared, m * cx.hidden, y)
        }
    }
}

/// The routed experts of layer `l` over the unit's `m` rows of `xn` into
/// `wf.routed`: the router, the route table over its ids, the gate and up
/// GEMMs over the q8_1 rows, the rule into q8 blocks of 32 a slot, the down
/// GEMM, the card sum.
#[allow(
    clippy::too_many_arguments,
    reason = "the routed block's context, rule, rows, buffers and the router's scale (rust-quality R8)"
)]
fn routed(
    cx: &GqaCx<'_>,
    k: &Kern<'_>,
    rule: Act,
    (l, m): (usize, usize),
    (r, forms, wf): (&Routed, &mut Forms, &mut WideFfn),
    xn: &DeviceBuffer<f32>,
    scale: f32,
) -> Result<(), GpuError> {
    let (gpu, w) = (cx.gpu, cx.w);
    let (stream, sink) = (gpu.stream(), gpu.layer_sink(l)?);
    let used = sigmoid_128_6::N_USED;
    let (gty, gate) = stack(w, &r.stacks[0])?;
    let (uty, up) = stack(w, &r.stacks[1])?;
    let (dty, down) = stack(w, &r.stacks[2])?;
    let (Some(RouteW::Grouped(ty, gate)), Some(RouteW::Grouped(_, up))) =
        (gate.route(gty), up.route(uty))
    else {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {l}: gate and up stacks of {gty} and {uty}: the wide arm runs the grouped K-quant GEMM over them"
            ),
        ));
    };
    let Some(RouteW::Blocks32(down)) = down.route(dty) else {
        return Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a down stack of {dty}, which no 32-value route GEMM reads"),
        ));
    };
    k.router.enqueue_router_rows(
        stream,
        gpu,
        k.g32,
        w.f32_tensor(WHAT, &r.router)?,
        xn,
        w.f32_buf(WHAT, &r.bias)?,
        scale,
        m,
        &mut wf.probs,
        &mut wf.ids,
        &mut wf.weights,
        sink,
    )?;
    k.gemm
        .enqueue_route(stream, &wf.ids, m * used, &mut wf.moe, sink)?;
    let ff = forms.slots32.k();
    for (w, y) in [(gate, &mut wf.g), (up, &mut wf.u)] {
        k.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty,
                w,
                rows_per_expert: ff,
                act: &forms.hid128,
                route: &wf.moe,
                input: GemmInput::Shared { top_k: used },
                y,
            },
        )?;
    }
    k.g32.enqueue_swiglu_act_quant32(
        stream,
        &wf.g,
        &wf.u,
        rule,
        m * used,
        &mut forms.slots32,
        sink,
    )?;
    k.g32.enqueue_gemm32(
        stream,
        Gemm32Args {
            w: down,
            rows_per_expert: cx.hidden,
            act: &forms.slots32,
            route: &wf.moe,
            input: GemmInput::PerSlot,
            y: &mut wf.down,
        },
    )?;
    let acc = CardAcc {
        down: &wf.down,
        w: &wf.weights,
        sel: &wf.ids,
        n: cx.hidden,
        m,
        n_card: sigmoid_128_6::N_EXPERT,
        n_used: used,
        w_pitch: used,
        fault: sink,
    };
    k.slots.enqueue_card_acc(stream, &acc, &mut wf.routed)
}

#[cfg(test)]
mod tests {
    use super::{Gemm32Args, Gemm32Kernels, GemmAct32, GemmInput, GemmKernels, GemmRoute, LATENT};
    use crate::Gpu;
    use crate::arch::raon1::rda::tests::{Lcg, q8_0, resident};
    use crate::gemm::Gemm32Weight;
    use crate::q8f32::Q8_0GemvHeadsMcolArgs;
    use crate::tensor::{Window, WindowMut};
    use cuda_core::DeviceBuffer;
    use model::arch::raon1::names;

    /// The query's nope values the absorption reads a head: the file's `attn_k_b` is
    /// 128 values (4 blocks of 32) a row.
    const NOPE: usize = 128;

    /// The key absorption's 32-value GEMM ([`super::absorb_heads`]'s core) over the file's
    /// geometry is the f32 gemv's rows ([`crate::q8f32`]'s `q8_0_gemv_heads_mcol`, the pass's
    /// absorb) to the activation's quantization: the rows' relative rms distance stays under
    /// [`BAND`], where a step, a scale or a head's rows read wrong is O(1). Both the unit's
    /// shape (40 heads, 256 rows: the slab entry's launch) and a small one.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_absorb_gemm_is_the_gemv_to_the_activation_quantization() {
        // 8-bit codes of a block of 32 with the block's own scale: ~0.4 % of the rms of a result
        // of 128 products; 1.0e-2 is 2.5 times that.
        const BAND: f64 = 1.0e-2;
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let (s, ctx) = (gpu.stream(), gpu.context());
        let gemm = GemmKernels::load(ctx).expect("the route kernels");
        let g32 = Gemm32Kernels::load(ctx).expect("the 32-value GEMM");
        for (heads, rows) in [(40, 256), (6, 40)] {
            let mut r = Lcg(0x0061_6273 + heads as u64);
            let name = names::attn_k_b(3);
            let w = resident(
                &gpu,
                &[q8_0(&mut r, name.clone(), vec![NOPE, LATENT, heads])],
            );
            let (qs, d) = w.q8_planes("test", &name).expect("the planes");
            let slots = rows * heads;
            let x = r.fill(slots * NOPE, -1.0, 1.0);
            let x = DeviceBuffer::from_host(s, &x).expect("the rows up");
            // The gemv, eight tokens a launch.
            let mut want = DeviceBuffer::<f32>::zeroed(s, slots * LATENT).expect("the output");
            for t0 in (0..rows).step_by(8) {
                let mc = 8.min(rows - t0);
                let xw = Window::<f32>::of(&x, t0 * heads * NOPE * 4, mc * heads * NOPE)
                    .expect("the rows");
                let mut yw = WindowMut::<f32>::of_mut(
                    &mut want,
                    t0 * heads * LATENT * 4,
                    mc * heads * LATENT,
                )
                .expect("the output window");
                gpu.q8f32()
                    .enqueue_q8_0_gemv_heads_mcol(
                        s,
                        Q8_0GemvHeadsMcolArgs {
                            qs,
                            d,
                            x: &xw,
                            rows_per_head: LATENT,
                            x_head_stride: NOPE,
                            y_head_stride: LATENT,
                            y_off: 0,
                            m: mc,
                            x_col_stride: heads * NOPE,
                            y_col_stride: heads * LATENT,
                            y: &mut yw,
                        },
                    )
                    .expect("the gemv");
            }
            // The GEMM: a slot a (row, head), slot s on head s % heads.
            let mut act = GemmAct32::new(s, slots, NOPE).expect("the form");
            g32.enqueue_quantize_gemm32(s, &x, slots, &mut act, gpu.unlabelled_sink())
                .expect("the quantization");
            let ids: Vec<u32> = (0..slots).map(|i| (i % heads) as u32).collect();
            let ids = DeviceBuffer::from_host(s, &ids).expect("the ids up");
            let mut route = GemmRoute::new(s, slots, heads).expect("the table");
            gemm.enqueue_route(s, &ids, slots, &mut route, gpu.unlabelled_sink())
                .expect("the route");
            let mut got = DeviceBuffer::<f32>::zeroed(s, slots * LATENT).expect("the output");
            g32.enqueue_gemm32(
                s,
                Gemm32Args {
                    w: Gemm32Weight::Q8_0Plane { qs, d },
                    rows_per_expert: LATENT,
                    act: &act,
                    route: &route,
                    input: GemmInput::PerSlot,
                    y: &mut got,
                },
            )
            .expect("the GEMM");
            let (got, want) = (
                got.to_host_vec(s).expect("the GEMM back"),
                want.to_host_vec(s).expect("the gemv back"),
            );
            assert!(
                got.iter().chain(&want).all(|v| v.is_finite()),
                "{heads} heads, {rows} rows: finite outputs"
            );
            let (num, den) = got
                .iter()
                .zip(&want)
                .fold((0.0f64, 0.0f64), |(n, d), (&g, &w)| {
                    (n + f64::from(g - w).powi(2), d + f64::from(w).powi(2))
                });
            let rel = (num / den).sqrt();
            println!("absorb gemm vs gemv: {heads} heads, {rows} rows: rel rms {rel:.3e}");
            assert!(
                rel <= BAND,
                "{heads} heads, {rows} rows: the GEMM is {rel:.3e} (rel rms) from the gemv; band {BAND:.1e}"
            );
        }
    }
}
