//! The prompt batch's Q8_0 projections on the tensor-core GEMM
//! (`bloomery_gpu::gemm`'s `gemm_q8_0p` over a one-expert route table), for
//! up to [`FrontShape::cols`] token columns at once: each input quantized
//! once to the GEMM's 32-value activations (`quantize_gemm32`), then every
//! projection that reads it, each output token-major as `q8_0_gemv_mcol`
//! writes it (`y[t · rows + r]`). A dense block's three projections go
//! through the GEMM their file types pick instead ([`GemmFront::dense_ffn`]).
//!
//! The entries are the sites whose inputs and outputs a batch holds whole:
//! - [`GemmFront::kda_in`]: a KDA mixer's projections of its normed input —
//!   `qkv`, `g_a` and `beta` from `xn`, then `g_b` from `g_a`'s rows (two
//!   inputs). The decay's pair (`f_a`, then `f_b`) is not the front's: an
//!   error in a log decay rescales every older term of the state, so it grows
//!   with the state's memory, which no bound on the activations' error fixes;
//! - [`GemmFront::kda_out`]: its output projection of the gated rows;
//! - [`GemmFront::latent_in`]: a latent mixer's joined projection of its
//!   normed input, as its two row ranges (the query's low rank, then the
//!   latent, the index key and the pool gate), each a window of the stack;
//! - [`GemmFront::latent_q`]: its query heads (`q_b`) from the normed low
//!   rank, into the front's own query rows ([`GemmFront::heads_rows`]);
//! - [`GemmFront::latent_out`]: its output projection of the heads' outputs,
//!   which the caller writes into the front's own rows between the two. The
//!   absorbed pair between them (`k_b`, `v_b`) is not the front's: each is a
//!   per-head product around the attention, which runs by chunk;
//! - [`GemmFront::dense_ffn`]: a dense block on `site::gated`, whatever the
//!   types of its three matrices: its normed input quantized once into the
//!   form its gate and up read, their rows through the clamped SwiGLU
//!   (`swiglu_clamp` at the layer's limit) quantized in one launch into the
//!   form its down reads, then the down. A routed layer's shared expert is
//!   not the front's: it runs under the host's serve.
//!
//! Numerics are the GEMM's (`gemm32.rs`'s and `grouped.rs`'s contracts): an
//! output's bits are a function of its weight row and its column's quantized
//! codes alone, so a batch's rows are those of any other column count over
//! the same inputs — but not the one-column gemv's, whose activations stay
//! f32 where the Q8_0 GEMM's are 32-value int8 blocks.
//!
//! The forms the dense block's sites read ([`DenseForms`]) are held only
//! when a dense layer's types need them: the normed input's q8 blocks of 32
//! values beside the mixers' own, q8_1 blocks of 128 for a K-quant gate and
//! up, and the SwiGLU rows in q8 blocks of 32 for a Q8_0 down or q8_1
//! blocks of 128 for a K-quant one.
//!
//! Refused by name, before any launch: a Q8_0 plane that is not the one a
//! projection reads, a weight whose K is not the width its input's
//! activations were sized for, a weight whose rows are not the width the
//! front's own rows were sized for, no columns or more than the scratch
//! holds, a row range outside its stack, a dense block on a front sized for
//! none or without a form its types read. A front whose made bytes are not
//! [`front_bytes`]' is refused at open.

use std::ops::Range;

use bloomery_gpu::fault::FaultSink;
use bloomery_gpu::gemm::{
    GEMM_BN, GEMM32_STEP, Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct, GemmAct32, GemmInput,
    GemmKernels, GemmRoute,
};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::site::{self, Form, Gated, GatedIo, SwigluRows, WideIn, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use bloomery_gpu_deepseek41::block::Block;
use cuda_core::DeviceBuffer;
use model::arch::glm5next::place::{self, FrontWidths};

use crate::body::q8;

/// What the front's errors name.
const WHAT: &str = "glm5next GemmFront";

/// The widths the front's scratch is sized for: the token columns, and the
/// widths of each input it quantizes and each row it holds whole
/// ([`FrontWidths`], the plan's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontShape {
    pub cols: usize,
    pub widths: FrontWidths,
}

/// The front's launches since it opened, by kind: the quantizers (the
/// SwiGLU's among them), the GEMM, the dense table's fill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrontStats {
    pub quantize: usize,
    pub gemm: usize,
    pub route: usize,
}

/// A KDA mixer's input projections by name ([`GemmFront::kda_in`]).
#[derive(Clone, Copy, Debug)]
pub struct KdaInNames<'a> {
    pub qkv: &'a str,
    pub g_a: &'a str,
    pub beta: &'a str,
    pub g_b: &'a str,
}

/// [`GemmFront::kda_in`]'s outputs, token-major: the joined q·k·v rows, the
/// gate's low-rank rows, the raw β and the gate's `z`.
pub struct KdaInRows<'a> {
    pub qkv: &'a mut DeviceBuffer<f32>,
    pub ga: &'a mut DeviceBuffer<f32>,
    pub beta_raw: &'a mut DeviceBuffer<f32>,
    pub z: &'a mut DeviceBuffer<f32>,
}

/// [`GemmFront::latent_in`]'s outputs, token-major: the query's low rank
/// (`q_lora` rows), and the latent, index key and pool gate (`kv` rows).
pub struct LatentInRows<'a> {
    pub qa: &'a mut DeviceBuffer<f32>,
    pub kv: &'a mut DeviceBuffer<f32>,
}

/// Which activation scratch an input quantizes into.
#[derive(Clone, Copy, Debug)]
enum Input {
    Embd,
    Low,
    Gated,
    QLow,
    Heads,
}

impl Input {
    fn name(self) -> &'static str {
        match self {
            Input::Embd => "normed input",
            Input::Low => "low-rank rows",
            Input::Gated => "gated rows",
            Input::QLow => "query low rank",
            Input::Heads => "heads' outputs",
        }
    }
}

/// One projection of an input: the Q8_0 weight `name` (its rows `rows`, or
/// every row) into `y`.
struct Proj<'a, 'y> {
    name: &'a str,
    rows: Option<Range<usize>>,
    y: &'y mut DeviceBuffer<f32>,
}

/// A dense block's scratch: the SwiGLU rows in the form each down type reads, and the gate's and up's f32 rows.
struct FfScratch {
    /// q8 blocks of 32 values, for a Q8_0 down.
    rows32: Option<GemmAct32>,
    /// q8_1 blocks of 128 values, for a K-quant down.
    rows128: Option<GemmAct>,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
}

/// The front: the GEMM's two device modules (the dense table's fill is the
/// K-quant family's `gemm_route`), the one-expert table, an activation
/// scratch per input width and the rows it holds whole, all made once at
/// open.
pub struct GemmFront {
    route_k: GemmKernels,
    k32: Gemm32Kernels,
    dense: GemmRoute,
    embd: GemmAct32,
    low: GemmAct32,
    gated: GemmAct32,
    q_low: GemmAct32,
    heads: GemmAct32,
    /// The normed input as q8_1 blocks of 128 values, for a K-quant gate and up; `None` when no dense layer
    /// has one.
    in128: Option<GemmAct>,
    /// `None` on a front sized for no dense block.
    ff: Option<FfScratch>,
    /// The latent query heads, token-major.
    q: DeviceBuffer<f32>,
    /// The latent heads' outputs, token-major.
    av: DeviceBuffer<f32>,
    shape: FrontShape,
    stats: FrontStats,
}

impl GemmFront {
    /// The front for `shape` on `gpu`'s context: [`front_bytes`]' bytes,
    /// refused by name otherwise. Load-time only.
    pub fn open(gpu: &Gpu, shape: FrontShape) -> Result<GemmFront, GpuError> {
        let stream = gpu.stream();
        let ctx = gpu.context();
        let (cols, w) = (shape.cols, shape.widths);
        let rows = |n: usize| DeviceBuffer::<f32>::zeroed(stream, cols * n);
        let ff = match w.ff {
            0 => None,
            ff => Some(FfScratch {
                rows32: w
                    .forms
                    .rows32
                    .then(|| GemmAct32::new(stream, cols, ff))
                    .transpose()?,
                rows128: w
                    .forms
                    .rows128
                    .then(|| GemmAct::new(stream, cols, ff))
                    .transpose()?,
                g: rows(ff)?,
                u: rows(ff)?,
            }),
        };
        let in128 = (w.ff > 0 && w.forms.in128)
            .then(|| GemmAct::new(stream, cols, w.embd))
            .transpose()?;
        let front = GemmFront {
            route_k: GemmKernels::load(ctx)?,
            k32: Gemm32Kernels::load(ctx)?,
            dense: GemmRoute::new(stream, cols, 1)?,
            embd: GemmAct32::new(stream, cols, w.embd)?,
            low: GemmAct32::new(stream, cols, w.low)?,
            gated: GemmAct32::new(stream, cols, w.gated)?,
            q_low: GemmAct32::new(stream, cols, w.q_low)?,
            heads: GemmAct32::new(stream, cols, w.heads_v)?,
            in128,
            ff,
            q: rows(w.q)?,
            av: rows(w.heads_v)?,
            shape,
            stats: FrontStats::default(),
        };
        let want = front_bytes(shape);
        if front.bytes() != want {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a front of {shape:?} takes {} B; front_bytes counts {want}",
                    front.bytes()
                ),
            });
        }
        Ok(front)
    }

    /// The widths the scratch was sized for.
    #[must_use]
    pub fn shape(&self) -> FrontShape {
        self.shape
    }

    /// The launches since open.
    #[must_use]
    pub fn stats(&self) -> FrontStats {
        self.stats
    }

    /// Device bytes of the table, the scratch and the rows.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let acts = [&self.embd, &self.low, &self.gated, &self.q_low, &self.heads]
            .iter()
            .map(|a| a.bytes())
            .sum::<usize>();
        let ff = self.ff.as_ref().map_or(0, |s| {
            s.rows32.as_ref().map_or(0, GemmAct32::bytes)
                + s.rows128.as_ref().map_or(0, GemmAct::bytes)
                + s.g.num_bytes()
                + s.u.num_bytes()
        });
        let in128 = self.in128.as_ref().map_or(0, GemmAct::bytes);
        self.dense.bytes() + acts + in128 + ff + self.q.num_bytes() + self.av.num_bytes()
    }

    /// The latent query heads and the heads' outputs, token-major, each a
    /// column's `widths.q` and `widths.heads_v` values: what
    /// [`GemmFront::latent_q`] writes and [`GemmFront::latent_out`] reads,
    /// for the caller's per-head launches between them.
    pub fn heads_rows(&mut self) -> (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>) {
        (&mut self.q, &mut self.av)
    }

    /// Layer `l`'s KDA input projections over the first `m` columns of its
    /// normed input `xn` (module doc): `xn` quantized once for `qkv`, `g_a`
    /// and `beta`, then `g_a`'s rows once for `g_b`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the step's context, the layer, the width, the input, the names and the outputs"
    )]
    pub fn kda_in(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        nm: KdaInNames<'_>,
        y: KdaInRows<'_>,
    ) -> Result<(), GpuError> {
        let KdaInRows {
            qkv,
            ga,
            beta_raw,
            z,
        } = y;
        // The second input's projection before the first launch, so a
        // refusal leaves no output half written.
        self.check(w, m, (Input::Low, ga.len()), &[(nm.g_b, None, z.len())])?;
        self.project(
            gpu,
            w,
            l,
            m,
            (Input::Embd, xn),
            [
                Proj::all(nm.qkv, qkv),
                Proj::all(nm.g_a, &mut *ga),
                Proj::all(nm.beta, beta_raw),
            ],
        )?;
        self.project(gpu, w, l, m, (Input::Low, ga), [Proj::all(nm.g_b, z)])
    }

    /// Layer `l`'s KDA output projection `out` over the first `m` columns of
    /// its gated rows, into `y`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the step's context, the layer, the width, the input, the name and the output"
    )]
    pub fn kda_out(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        m: usize,
        gated: &DeviceBuffer<f32>,
        out: &str,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.project(gpu, w, l, m, (Input::Gated, gated), [Proj::all(out, y)])
    }

    /// Layer `l`'s latent joined projection `stack` over the first `m`
    /// columns of its normed input `xn`, quantized once, as its two row
    /// ranges: `0 .. q_lora` into `y.qa`, `q_lora .. q_lora + kv` into `y.kv`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the step's context, the layer, the width, the input, the stack and its split, the outputs"
    )]
    pub fn latent_in(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        m: usize,
        xn: &DeviceBuffer<f32>,
        (stack, q_lora, kv): (&str, usize, usize),
        y: LatentInRows<'_>,
    ) -> Result<(), GpuError> {
        self.project(
            gpu,
            w,
            l,
            m,
            (Input::Embd, xn),
            [
                Proj {
                    name: stack,
                    rows: Some(0..q_lora),
                    y: y.qa,
                },
                Proj {
                    name: stack,
                    rows: Some(q_lora..q_lora + kv),
                    y: y.kv,
                },
            ],
        )
    }

    /// Layer `l`'s latent query heads `q_b` over the first `m` columns of its
    /// normed query low rank `qr`, into the front's query rows
    /// ([`GemmFront::heads_rows`]): `q_b`'s rows must be their width.
    pub fn latent_q(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (l, m): (usize, usize),
        qr: &DeviceBuffer<f32>,
        q_b: &str,
    ) -> Result<(), GpuError> {
        self.check(w, m, (Input::QLow, qr.len()), &[(q_b, None, self.q.len())])?;
        check_rows(w, q_b, self.shape.widths.q, "query rows")?;
        let fault = self.fill(gpu, l, m)?;
        self.k32
            .enqueue_quantize_gemm32(gpu.stream(), qr, m, &mut self.q_low, fault)?;
        self.stats.quantize += 1;
        gemm(
            (&self.k32, &self.dense),
            gpu,
            q8(w, q_b)?,
            &self.q_low,
            &mut self.q,
        )?;
        self.stats.gemm += 1;
        Ok(())
    }

    /// Layer `l`'s latent output projection `out` over the first `m` columns
    /// of the front's heads' outputs ([`GemmFront::heads_rows`], which the
    /// caller wrote), into `y`.
    pub fn latent_out(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (l, m): (usize, usize),
        out: &str,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.check(w, m, (Input::Heads, self.av.len()), &[(out, None, y.len())])?;
        let fault = self.fill(gpu, l, m)?;
        self.k32
            .enqueue_quantize_gemm32(gpu.stream(), &self.av, m, &mut self.heads, fault)?;
        self.stats.quantize += 1;
        gemm((&self.k32, &self.dense), gpu, q8(w, out)?, &self.heads, y)?;
        self.stats.gemm += 1;
        Ok(())
    }

    /// The dense block `blk` over the first `m` columns of its normed input `xn`, into `y`, on
    /// [`site::gated`]: `xn` quantized once into the form its gate and up read, their rows through the clamped
    /// SwiGLU at `blk.limit` (`swiglu_clamp`: none at or under 1e-6) quantized in the same launch into the form
    /// its down reads, then the down. The launches are those of the three types ([`site::gemm`]); `blk.ff`
    /// must be the front's SwiGLU width.
    pub fn dense_ffn(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        m: usize,
        xn: &DeviceBuffer<f32>,
        blk: &Block<'_>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.check_dense(w, m, (xn.len(), y.len()), blk)?;
        let fault = self.fill(gpu, blk.layer, m)?;
        let [gate, up, down] = blk.plan.types();
        let [gate_n, up_n, down_n] = blk.names;
        let hidden = self.shape.widths.embd;
        let stream = gpu.stream();
        let GemmFront {
            route_k,
            k32,
            dense,
            embd,
            in128,
            ff,
            stats,
            ..
        } = self;
        let missing = |missing| GpuError::State {
            what: WHAT,
            missing,
        };
        let FfScratch {
            rows32,
            rows128,
            g,
            u,
        } = ff.as_mut().ok_or(missing("the dense block's scratch"))?;
        let x = if gate.reads() == Form::Q8x128 {
            let act = in128
                .as_mut()
                .ok_or(missing("the normed input's q8_1 blocks of 128"))?;
            gpu.enqueue_quantize_gemm(xn, m, act, fault)?;
            WideIn {
                q128: Some(&*act),
                q32: None,
                f32: None,
            }
        } else {
            k32.enqueue_quantize_gemm32(stream, xn, m, embd, fault)?;
            WideIn {
                q128: None,
                q32: Some(&*embd),
                f32: None,
            }
        };
        let rows = if down.reads() == Form::Q8x128 {
            SwigluRows {
                q128: rows128.as_mut(),
                q32: None,
                f32: None,
            }
        } else {
            SwigluRows {
                q128: None,
                q32: rows32.as_mut(),
                f32: None,
            }
        };
        let block = Gated {
            w,
            route: &*dense,
            gate: (gate, gate_n),
            up: (up, up_n),
            down: (down, down_n),
            ff: blk.ff,
            hidden,
            rule: Act::SwigluClamp { limit: blk.limit },
            input: GemmInput::PerSlot,
        };
        let io = GatedIo {
            x: &x,
            gate: g,
            up: u,
            rows,
            y,
        };
        let k = WideKernels {
            gemm: &*route_k,
            g32: &*k32,
        };
        site::gated(gpu, &k, &block, (m, m), io, fault)?;
        stats.quantize += 2;
        stats.gemm += 3;
        Ok(())
    }

    /// The dense block `blk` over `m` columns launchable: the front sized for a dense block of `blk.ff` rows
    /// and holding the forms its types read, `m` inside the scratch, the three weights resident as the plan's
    /// types at the block's shape, and the input and output long enough; refused by name otherwise.
    fn check_dense(
        &self,
        w: &Weights,
        m: usize,
        (xn_len, y_len): (usize, usize),
        blk: &Block<'_>,
    ) -> Result<(), GpuError> {
        let refuse = |detail: String| Err(GpuError::Shape { what: WHAT, detail });
        let l = blk.layer;
        let Some(ff) = self.ff.as_ref() else {
            return refuse(format!(
                "layer {l}'s dense block on a front sized for none (a SwiGLU width of 0)"
            ));
        };
        let widths = self.shape.widths;
        if blk.ff != widths.ff {
            return refuse(format!(
                "layer {l}'s dense block has {} rows; the front's SwiGLU rows were sized for {}",
                blk.ff, widths.ff
            ));
        }
        if !(1..=self.shape.cols).contains(&m) {
            return refuse(format!(
                "{m} columns: the scratch holds 1 ..= {} columns",
                self.shape.cols
            ));
        }
        let need = m * widths.embd;
        if xn_len < need || y_len < need {
            return refuse(format!(
                "layer {l}'s dense block over {m} columns of {} values needs {need} in and out; the \
                 normed input holds {xn_len} and the output {y_len}",
                widths.embd
            ));
        }
        blk.plan
            .check_resident(w, WHAT, blk.names, (blk.ff, widths.embd))?;
        let [gate, _, down] = blk.plan.types();
        let has = |held: bool, ty: site::SiteTy, what: &str| {
            if held {
                Ok(())
            } else {
                refuse(format!(
                    "layer {l}'s dense block has a {ty} {what}, whose form the front was sized without"
                ))
            }
        };
        if gate.reads() == Form::Q8x128 {
            has(self.in128.is_some(), gate, "gate and up")?;
        }
        if down.reads() == Form::Q8x128 {
            has(ff.rows128.is_some(), down, "down")
        } else {
            has(ff.rows32.is_some(), down, "down")
        }
    }

    /// The scratch input `which` quantizes into.
    fn act(&self, which: Input) -> &GemmAct32 {
        match which {
            Input::Embd => &self.embd,
            Input::Low => &self.low,
            Input::Gated => &self.gated,
            Input::QLow => &self.q_low,
            Input::Heads => &self.heads,
        }
    }

    /// The projections `outs` (a weight's name, its row range or every row,
    /// the length of its output) of `m` columns of input `which` (the length
    /// of its values) launchable: `m` inside the scratch, each weight a
    /// resident Q8_0 plane of the K the input's scratch was sized for, each
    /// row range inside its weight, the values and every output long enough;
    /// refused by name otherwise.
    fn check(
        &self,
        w: &Weights,
        m: usize,
        (which, x_len): (Input, usize),
        outs: &[(&str, Option<Range<usize>>, usize)],
    ) -> Result<(), GpuError> {
        let refuse = |detail: String| Err(GpuError::Shape { what: WHAT, detail });
        if !(1..=self.shape.cols).contains(&m) {
            return refuse(format!(
                "{m} columns: the scratch holds 1 ..= {} columns",
                self.shape.cols
            ));
        }
        let k = self.act(which).k();
        if x_len < m * k {
            return refuse(format!(
                "the {} hold {x_len} values, {m} columns of K = {k} need {}",
                which.name(),
                m * k
            ));
        }
        for (name, rows, y_len) in outs {
            let (qs, _) = q8(w, name)?;
            if 4 * qs.cols() != k {
                return refuse(format!(
                    "{name} is a projection of K = {}; the {} activations were sized for K = {k}",
                    4 * qs.cols(),
                    which.name()
                ));
            }
            let n = match rows {
                Some(r) if r.is_empty() || r.end > qs.rows() => {
                    return refuse(format!("rows {r:?} of {name}, which has {}", qs.rows()));
                }
                Some(r) => r.len(),
                None => qs.rows(),
            };
            if *y_len < m * n {
                return refuse(format!(
                    "{name}'s output holds {y_len} values, {m} columns of {n} rows need {}",
                    m * n
                ));
            }
        }
        Ok(())
    }

    /// Layer `l`'s fault sink, the dense table refilled first when it was
    /// last filled for another count than `m`.
    fn fill(&mut self, gpu: &Gpu, l: usize, m: usize) -> Result<FaultSink, GpuError> {
        let fault = gpu.layer_sink(l)?;
        if self.dense.filled() != Some(m) {
            self.route_k
                .enqueue_route_dense(gpu.stream(), m, &mut self.dense, fault)?;
            self.stats.route += 1;
        }
        Ok(fault)
    }

    /// The first `m` columns of `x` quantized once into input `which`'s
    /// scratch, then each projection of `projs` over them. Every refusal
    /// ([`GemmFront::check`]) comes before the first launch.
    fn project<const N: usize>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        l: usize,
        m: usize,
        (which, x): (Input, &DeviceBuffer<f32>),
        projs: [Proj<'_, '_>; N],
    ) -> Result<(), GpuError> {
        let outs = projs
            .each_ref()
            .map(|p| (p.name, p.rows.clone(), p.y.len()));
        self.check(w, m, (which, x.len()), &outs)?;
        let fault = self.fill(gpu, l, m)?;
        let act = match which {
            Input::Embd => &mut self.embd,
            Input::Low => &mut self.low,
            Input::Gated => &mut self.gated,
            Input::QLow => &mut self.q_low,
            Input::Heads => &mut self.heads,
        };
        self.k32
            .enqueue_quantize_gemm32(gpu.stream(), x, m, act, fault)?;
        self.stats.quantize += 1;
        for p in projs {
            let (qs, d) = q8(w, p.name)?;
            let act = self.act(which);
            let k = (&self.k32, &self.dense);
            match p.rows {
                None => gemm(k, gpu, (qs, d), act, p.y)?,
                Some(rows) => {
                    let (n, cq, cd) = (rows.len(), qs.cols(), d.cols());
                    let wq = DeviceTensor::<u32>::window_of(qs.buf(), rows.start * cq * 4, n, cq)?;
                    let wd = DeviceTensor::<u16>::window_of(d.buf(), rows.start * cd * 2, n, cd)?;
                    gemm(k, gpu, (&wq, &wd), act, p.y)?;
                }
            }
            self.stats.gemm += 1;
        }
        Ok(())
    }
}

/// `name`'s rows exactly `want`, the width of the front's rows (`what`)
/// they are written into or read beside; refused by name otherwise.
fn check_rows(w: &Weights, name: &str, want: usize, what: &str) -> Result<(), GpuError> {
    let (qs, _) = q8(w, name)?;
    if qs.rows() == want {
        return Ok(());
    }
    Err(GpuError::Shape {
        what: WHAT,
        detail: format!(
            "{name} has {} rows; the front's {what} were sized for {want}",
            qs.rows()
        ),
    })
}

/// `y = W · act` over the dense table's columns for the planes `(qs, d)`.
fn gemm(
    (k32, route): (&Gemm32Kernels, &GemmRoute),
    gpu: &Gpu,
    (qs, d): (&DeviceTensor<u32>, &DeviceTensor<u16>),
    act: &GemmAct32,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    k32.enqueue_gemm32(
        gpu.stream(),
        Gemm32Args {
            w: Gemm32Weight::Q8_0Plane { qs, d },
            rows_per_expert: qs.rows(),
            act,
            route,
            input: GemmInput::PerSlot,
            y,
        },
    )
}

/// Card bytes of a front of `shape` ([`GemmFront::bytes`]): the plan's
/// formula (`place::front_bytes`), which the plan reserves on the stage card.
#[must_use]
pub fn front_bytes(shape: FrontShape) -> usize {
    place::front_bytes(shape.cols, shape.widths)
}

const _: () = assert!(GEMM_BN == place::FRONT_TILE_COLS);
const _: () = assert!(GEMM32_STEP == place::FRONT_STEP);

impl<'a, 'y> Proj<'a, 'y> {
    /// Every row of `name` into `y`.
    fn all(name: &'a str, y: &'y mut DeviceBuffer<f32>) -> Proj<'a, 'y> {
        Proj {
            name,
            rows: None,
            y,
        }
    }
}
