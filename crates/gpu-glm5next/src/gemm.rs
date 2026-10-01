//! The prompt batch's Q8_0 projections on the tensor-core GEMM
//! (`bloomery_gpu::gemm`'s `gemm_q8_0p` over a one-expert route table), for
//! up to [`FrontShape::cols`] token columns at once: each input quantized
//! once to the GEMM's 32-value activations (`quantize_gemm32`), then every
//! projection that reads it, each output token-major as `q8_0_gemv_mcol`
//! writes it (`y[t · rows + r]`).
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
//!   latent, the index key and the pool gate), each a window of the stack.
//!
//! Numerics are the GEMM's (`gemm32.rs`'s contract): an output's bits are a
//! function of its weight row and its column's quantized codes alone, so a
//! batch's rows are those of any other column count over the same inputs —
//! but not the one-column gemv's, whose activations stay f32.
//!
//! Refused by name, before any launch: a weight that is not a resident
//! Q8_0 plane, a weight whose K is not the width its input's activations
//! were sized for, no columns or more than the scratch holds, a row range
//! outside its stack.

use std::ops::Range;

use bloomery_gpu::gemm::{
    Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct32, GemmInput, GemmKernels, GemmRoute,
};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
use cuda_core::DeviceBuffer;

use crate::body::q8;

/// What the front's errors name.
const WHAT: &str = "glm5next GemmFront";

/// The widths the front's scratch is sized for: the token columns, and the
/// K of each input — the normed input (`embedding_length`), a KDA mixer's
/// low-rank gate rows (`g_a`'s output, one head) and its gated rows (the
/// value heads).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontShape {
    pub cols: usize,
    pub embd: usize,
    pub low: usize,
    pub gated: usize,
}

/// The front's launches since it opened, by kind: the quantizer, the
/// GEMM, the dense table's fill.
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
}

impl Input {
    fn name(self) -> &'static str {
        match self {
            Input::Embd => "normed input",
            Input::Low => "low-rank rows",
            Input::Gated => "gated rows",
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

/// The front: the GEMM's two device modules (the dense table's fill is the
/// K-quant family's `gemm_route`), the one-expert table and an activation
/// scratch per input width, all made once at open.
pub struct GemmFront {
    route_k: GemmKernels,
    k32: Gemm32Kernels,
    dense: GemmRoute,
    embd: GemmAct32,
    low: GemmAct32,
    gated: GemmAct32,
    shape: FrontShape,
    stats: FrontStats,
}

impl GemmFront {
    /// The front for `shape` on `gpu`'s context. Load-time only.
    pub fn open(gpu: &Gpu, shape: FrontShape) -> Result<GemmFront, GpuError> {
        let stream = gpu.stream();
        let ctx = gpu.context();
        Ok(GemmFront {
            route_k: GemmKernels::load(ctx)?,
            k32: Gemm32Kernels::load(ctx)?,
            dense: GemmRoute::new(stream, shape.cols, 1)?,
            embd: GemmAct32::new(stream, shape.cols, shape.embd)?,
            low: GemmAct32::new(stream, shape.cols, shape.low)?,
            gated: GemmAct32::new(stream, shape.cols, shape.gated)?,
            shape,
            stats: FrontStats::default(),
        })
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

    /// Device bytes of the table and the scratch.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.dense.bytes() + self.embd.bytes() + self.low.bytes() + self.gated.bytes()
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

    /// The scratch input `which` quantizes into.
    fn act(&self, which: Input) -> &GemmAct32 {
        match which {
            Input::Embd => &self.embd,
            Input::Low => &self.low,
            Input::Gated => &self.gated,
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

    /// The first `m` columns of `x` quantized once into input `which`'s
    /// scratch, then each projection of `projs` over them; the dense table
    /// refilled first when it was last filled for another count. Every
    /// refusal ([`GemmFront::check`]) comes before the first launch.
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
        let stream = gpu.stream();
        let fault = gpu.layer_sink(l)?;
        if self.dense.filled() != Some(m) {
            self.route_k
                .enqueue_route_dense(stream, m, &mut self.dense, fault)?;
            self.stats.route += 1;
        }
        let act = match which {
            Input::Embd => &mut self.embd,
            Input::Low => &mut self.low,
            Input::Gated => &mut self.gated,
        };
        self.k32.enqueue_quantize_gemm32(stream, x, m, act, fault)?;
        self.stats.quantize += 1;
        for p in projs {
            let (qs, d) = q8(w, p.name)?;
            let act = self.act(which);
            match p.rows {
                None => self.gemm(gpu, (qs, d), act, p.y)?,
                Some(rows) => {
                    let (n, cq, cd) = (rows.len(), qs.cols(), d.cols());
                    let wq = DeviceTensor::<u32>::window_of(qs.buf(), rows.start * cq * 4, n, cq)?;
                    let wd = DeviceTensor::<u16>::window_of(d.buf(), rows.start * cd * 2, n, cd)?;
                    self.gemm(gpu, (&wq, &wd), act, p.y)?;
                }
            }
            self.stats.gemm += 1;
        }
        Ok(())
    }

    /// `y = W · act` over the dense table's columns for the planes `(qs, d)`.
    fn gemm(
        &self,
        gpu: &Gpu,
        (qs, d): (&DeviceTensor<u32>, &DeviceTensor<u16>),
        act: &GemmAct32,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.k32.enqueue_gemm32(
            gpu.stream(),
            Gemm32Args {
                w: Gemm32Weight::Q8_0Plane { qs, d },
                rows_per_expert: qs.rows(),
                act,
                route: &self.dense,
                input: GemmInput::PerSlot,
                y,
            },
        )
    }
}

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
