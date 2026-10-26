//! The prompt batch's mixer projections on the tensor-core GEMM, for up to
//! [`FrontShape::cols`] token columns at once, each by the type its weight
//! is resident as ([`site::resident_gemm`]): Q8_0 on `gemm_q8_0p` over a
//! one-expert route table, the K-quants (Q3_K, Q4_K, Q5_K, Q6_K) on the grouped
//! GEMM over q8_1 blocks of 128 values, BF16 on its exact tile over the f32
//! rows. Each input is quantized once into the forms the projections that read
//! it take ([`WideActs`]: `quantize_gemm32` for the Q8_0 sites'
//! 32-value activations, `quantize_gemm` for the K-quants'), then every
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
//! - [`GemmFront::latent_in`]: a latent mixer's projections of its normed
//!   input, the joined stack as its two row ranges (the query's low rank, then
//!   the latent, the index key and the pool gate), each a window of the
//!   stack, or the four weights the load left apart, each into its rows;
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
//! A projection of a weight the load left apart (a KDA's `q`, `k` and `v` of
//! three types, a latent stack's four) writes its rows into its place of the
//! joined layout through the front's query rows as a stage and one strided copy.
//!
//! Numerics are each GEMM's (`gemm32.rs`'s and `grouped.rs`'s contracts): an
//! output's bits are a function of its weight row and its column's quantized
//! codes alone, so a batch's rows are those of any other column count over
//! the same inputs — but not the one-column gemv's, whose activations stay
//! f32 where the Q8_0 GEMM's are 32-value int8 blocks.
//!
//! The forms the sites read ([`place::DenseForms`], [`place::MixerForms`]) are held only
//! when the file's types need them: each input's q8 blocks of 32 values, the
//! normed input's q8_1 blocks of 128 for a K-quant gate and up or mixer
//! site, the other inputs' for a K-quant mixer site, and the SwiGLU rows in q8
//! blocks of 32 for a Q8_0 down or q8_1 blocks of 128 for a K-quant one.
//!
//! Refused by name, before any launch: a type with no wide GEMM (the
//! front's own list, [`GemmFront::covers`]), a weight whose K is not the
//! width its input's activations were sized for, a weight whose rows are not
//! the width the front's own rows were sized for, no columns or more than the
//! scratch holds, a row range outside its stack or of a weight that is not
//! Q8_0 planes, a K-quant of rows that are not a multiple of 16, a dense block
//! on a front sized for none or without a form its types read. A front whose
//! made bytes are not [`front_bytes`]' is refused at open.

use std::ops::Range;

use bloomery_gpu::fault::FaultSink;
use bloomery_gpu::gemm::{GEMM_BN, GEMM32_STEP, Gemm32Kernels, GemmInput, GemmKernels, GemmRoute};
use bloomery_gpu::kquant::Act;
use bloomery_gpu::site::{
    self, Gated, GatedIo, SiteTy, SiteWeight, WideActs, WideForms, WideIn, WideKernels,
};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::block::Block;
use cuda_core::DeviceBuffer;
use model::arch::glm5next::place::{self, FrontWidths};

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

/// The front's launches since it opened, by kind: the quantizers (one a form
/// of each input, the SwiGLU's among them), the GEMM, the dense table's fill
/// and the strided copies of a part into its joined layout.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrontStats {
    pub quantize: usize,
    pub gemm: usize,
    pub route: usize,
    pub copy: usize,
}

/// A projection's weights as the load left them: the one joined weight (a
/// single entry at row 0) or the parts, each with its first row in the
/// joined layout, and the joined layout's rows ([`crate::tensors::Joined::launches`]).
#[derive(Clone, Copy, Debug)]
pub struct Launches<'a> {
    pub parts: &'a [(&'a str, usize)],
    pub rows: usize,
}

/// A KDA mixer's input projections by name ([`GemmFront::kda_in`]).
#[derive(Clone, Copy, Debug)]
pub struct KdaInNames<'a> {
    pub qkv: Launches<'a>,
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

/// The f32 rows an input quantizes from: the caller's, or the front's own
/// heads' outputs ([`GemmFront::heads_rows`]).
enum Src<'a> {
    Caller(&'a DeviceBuffer<f32>),
    Heads,
}

/// Where a projection writes: a caller's buffer, or the front's own query rows.
enum Dst<'y> {
    Caller(&'y mut DeviceBuffer<f32>),
    Query,
}

/// One projection of an input: the weight `name` (its rows `rows`, or every
/// row) into output `out`, whole at `stride == 0`, else into rows `at ..` of
/// each `stride`-value column (through the stage and one strided copy).
struct Proj<'a> {
    name: &'a str,
    rows: Option<Range<usize>>,
    out: usize,
    at: usize,
    stride: usize,
}

impl<'a> Proj<'a> {
    /// Every row of `name` into output `out`.
    fn all(name: &'a str, out: usize) -> Proj<'a> {
        Proj {
            name,
            rows: None,
            out,
            at: 0,
            stride: 0,
        }
    }

    /// The projections of a weight the load may have left apart
    /// ([`Launches`]) into output `out`: the one joined weight whole, or each
    /// part into its rows of the joined layout.
    fn launches(l: Launches<'a>, out: usize) -> Vec<Proj<'a>> {
        match l.parts {
            [(name, 0)] => vec![Proj::all(name, out)],
            parts => parts
                .iter()
                .map(|&(name, at)| Proj {
                    name,
                    rows: None,
                    out,
                    at,
                    stride: l.rows,
                })
                .collect(),
        }
    }
}

/// A dense block's scratch: the SwiGLU rows in the form each down type reads (q8 blocks of 32 values for a
/// Q8_0 down, q8_1 blocks of 128 for a K-quant one), and the gate's and up's f32 rows.
struct FfScratch {
    rows: WideActs,
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
    embd: WideActs,
    low: WideActs,
    gated: WideActs,
    q_low: WideActs,
    heads: WideActs,
    /// `None` on a front sized for no dense block.
    ff: Option<FfScratch>,
    /// The latent query heads, token-major: also the stage a part's rows
    /// are made in before their strided copy.
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
                rows: WideActs::new(
                    stream,
                    WideForms::q8(w.forms.rows128, w.forms.rows32),
                    cols,
                    ff,
                )?,
                g: rows(ff)?,
                u: rows(ff)?,
            }),
        };
        // Every input holds its q8 blocks of 32; the q8_1 blocks of 128 are held where a K-quant site
        // reads them, the normed input's for a K-quant dense gate and up too.
        let m = w.mixer;
        let in128 = (w.ff > 0 && w.forms.in128) || m.embd;
        let acts = |q128: bool, k: usize| WideActs::new(stream, WideForms::q8(q128, true), cols, k);
        let front = GemmFront {
            route_k: GemmKernels::load(ctx)?,
            k32: Gemm32Kernels::load(ctx)?,
            dense: GemmRoute::new(stream, cols, 1)?,
            embd: acts(in128, w.embd)?,
            low: acts(m.low, w.low)?,
            gated: acts(m.gated, w.gated)?,
            q_low: acts(m.q_low, w.q_low)?,
            heads: acts(m.heads, w.heads_v)?,
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
        let ff = self
            .ff
            .as_ref()
            .map_or(0, |s| s.rows.bytes() + s.g.num_bytes() + s.u.num_bytes());
        self.dense.bytes() + acts + ff + self.q.num_bytes() + self.av.num_bytes()
    }

    /// Whether the front runs every weight of `names`: resident as Q8_0
    /// planes, as a K-quant (Q3_K, Q4_K, Q5_K or Q6_K, whose rows are a
    /// multiple of 16 and which holds an even number of super-blocks a row
    /// at Q6_K) or as BF16 words. Any other type, or a name that is not
    /// resident, has no wide GEMM: the caller runs it by chunk.
    #[must_use]
    pub fn covers<'n>(w: &Weights, names: impl IntoIterator<Item = &'n str>) -> bool {
        names.into_iter().all(|n| {
            w.get(n)
                .is_some_and(|weight| wide_ty(weight).is_ok() && weight_rows_ok(weight))
        })
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
        let gate = [Proj::all(nm.g_b, 0)];
        self.check(w, m, (Input::Low, ga.len()), &gate, &[z.len()])?;
        let mut first = Proj::launches(nm.qkv, 0);
        first.push(Proj::all(nm.g_a, 1));
        first.push(Proj::all(nm.beta, 2));
        self.project(
            gpu,
            w,
            (l, m),
            (Input::Embd, Src::Caller(xn)),
            &first,
            &mut [
                Dst::Caller(qkv),
                Dst::Caller(&mut *ga),
                Dst::Caller(beta_raw),
            ],
        )?;
        self.project(
            gpu,
            w,
            (l, m),
            (Input::Low, Src::Caller(ga)),
            &gate,
            &mut [Dst::Caller(z)],
        )
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
        self.project(
            gpu,
            w,
            (l, m),
            (Input::Gated, Src::Caller(gated)),
            &[Proj::all(out, 0)],
            &mut [Dst::Caller(y)],
        )
    }

    /// Layer `l`'s latent projections of the normed input `xn` over its first
    /// `m` columns, quantized once: the joined `stack` as its two row ranges
    /// (`0 .. q_lora` into `y.qa`, `q_lora .. q_lora + kv` into `y.kv`), or
    /// its parts apart, the first (`q_lora` rows) into `y.qa` and each of the
    /// others into its rows of a `kv`-value column of `y.kv`.
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
        (stack, q_lora, kv): (Launches<'_>, usize, usize),
        y: LatentInRows<'_>,
    ) -> Result<(), GpuError> {
        let projs = match stack.parts {
            [(name, 0)] => vec![
                Proj {
                    name,
                    rows: Some(0..q_lora),
                    out: 0,
                    at: 0,
                    stride: 0,
                },
                Proj {
                    name,
                    rows: Some(q_lora..q_lora + kv),
                    out: 1,
                    at: 0,
                    stride: 0,
                },
            ],
            parts => {
                let mut v = Vec::with_capacity(parts.len());
                for &(name, at) in parts {
                    v.push(if at < q_lora {
                        if at != 0 || w.resident(WHAT, name)?.rows() != q_lora {
                            return Err(GpuError::Shape {
                                what: WHAT,
                                detail: format!(
                                    "{name}: the stack's first part is the query's {q_lora} rows"
                                ),
                            });
                        }
                        Proj::all(name, 0)
                    } else {
                        Proj {
                            name,
                            rows: None,
                            out: 1,
                            at: at - q_lora,
                            stride: kv,
                        }
                    });
                }
                v
            }
        };
        self.project(
            gpu,
            w,
            (l, m),
            (Input::Embd, Src::Caller(xn)),
            &projs,
            &mut [Dst::Caller(y.qa), Dst::Caller(y.kv)],
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
        check_rows(w, q_b, self.shape.widths.q, "query rows")?;
        self.project(
            gpu,
            w,
            (l, m),
            (Input::QLow, Src::Caller(qr)),
            &[Proj::all(q_b, 0)],
            &mut [Dst::Query],
        )
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
        self.project(
            gpu,
            w,
            (l, m),
            (Input::Heads, Src::Heads),
            &[Proj::all(out, 0)],
            &mut [Dst::Caller(y)],
        )
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
        let GemmFront {
            route_k,
            k32,
            dense,
            embd,
            ff,
            stats,
            ..
        } = self;
        let missing = |missing| GpuError::State {
            what: WHAT,
            missing,
        };
        let FfScratch { rows, g, u } = ff.as_mut().ok_or(missing("the dense block's scratch"))?;
        let k = WideKernels {
            gemm: &*route_k,
            g32: Some(&*k32),
        };
        embd.quantize(gpu, &k, fault, xn, m, &[gate, up])?;
        let want = WideActs::wants_of(&[gate, up]);
        stats.quantize += usize::from(want.q128) + usize::from(want.q32);
        let x = embd.input(xn);
        let block = Gated {
            w,
            route: &*dense,
            gate: (gate, SiteWeight::Resident(gate_n)),
            up: (up, SiteWeight::Resident(up_n)),
            down: (down, SiteWeight::Resident(down_n)),
            ff: blk.ff,
            hidden,
            rule: Act::SwigluClamp { limit: blk.limit },
            input: GemmInput::PerSlot,
        };
        let io = GatedIo {
            x: &x,
            gate: g,
            up: u,
            rows: rows.swiglu_rows(),
            y,
        };
        site::gated(gpu, &k, &block, (m, m), io, fault)?;
        stats.quantize += 1;
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
        let [gate, up, down] = blk.plan.types();
        let has = |held: bool, ty: SiteTy, what: &str| {
            if held {
                Ok(())
            } else {
                refuse(format!(
                    "layer {l}'s dense block has a {ty} {what}, whose form the front was sized without"
                ))
            }
        };
        let want = WideActs::wants_of(&[gate, up]);
        has(
            (!want.q128 || self.embd.q128.is_some()) && (!want.q32 || self.embd.q32.is_some()),
            gate,
            "gate and up",
        )?;
        let want = WideActs::wants(down);
        has(
            (!want.q128 || ff.rows.q128.is_some())
                && (!want.q32 || ff.rows.q32.is_some())
                && (!want.f32 || ff.rows.f32.is_some()),
            down,
            "down",
        )
    }

    /// The width of input `which`'s values.
    fn width(&self, which: Input) -> usize {
        let w = self.shape.widths;
        match which {
            Input::Embd => w.embd,
            Input::Low => w.low,
            Input::Gated => w.gated,
            Input::QLow => w.q_low,
            Input::Heads => w.heads_v,
        }
    }

    /// The projections `projs` of `m` columns of input `which` (the length
    /// of its values) into outputs of `out_lens` values launchable: `m`
    /// inside the scratch, each weight resident as a type the front runs
    /// ([`GemmFront::covers`]) at the K the input's scratch was sized for,
    /// each row range inside a Q8_0 weight, each part inside its stride and
    /// the stage, and the values and every output long enough; refused by
    /// name otherwise.
    fn check(
        &self,
        w: &Weights,
        m: usize,
        (which, x_len): (Input, usize),
        projs: &[Proj<'_>],
        out_lens: &[usize],
    ) -> Result<(), GpuError> {
        let refuse = |detail: String| Err(GpuError::Shape { what: WHAT, detail });
        if !(1..=self.shape.cols).contains(&m) {
            return refuse(format!(
                "{m} columns: the scratch holds 1 ..= {} columns",
                self.shape.cols
            ));
        }
        let k = self.width(which);
        if x_len < m * k {
            return refuse(format!(
                "the {} hold {x_len} values, {m} columns of K = {k} need {}",
                which.name(),
                m * k
            ));
        }
        for p in projs {
            let name = p.name;
            let weight = w.resident(WHAT, name)?;
            let ty = wide_ty(weight).map_err(|why| GpuError::Shape {
                what: WHAT,
                detail: format!("{name}: {why}"),
            })?;
            if let Some(wk) = weight_k(weight)
                && wk != k
            {
                return refuse(format!(
                    "{name} is a projection of K = {wk}; the {} activations were sized for K = {k}",
                    which.name()
                ));
            }
            let all = weight.rows();
            let n = match &p.rows {
                Some(_) if ty != SiteTy::Q8_0 => {
                    return refuse(format!(
                        "{name} is {ty}: only Q8_0 planes launch a window of rows"
                    ));
                }
                Some(r) if r.is_empty() || r.end > all => {
                    return refuse(format!("rows {r:?} of {name}, which has {all}"));
                }
                Some(r) => r.len(),
                None => all,
            };
            let Some(&len) = out_lens.get(p.out) else {
                return refuse(format!("{name}: output {} of {}", p.out, out_lens.len()));
            };
            if p.stride == 0 {
                if len < m * n {
                    return refuse(format!(
                        "{name}'s output holds {len} values, {m} columns of {n} rows need {}",
                        m * n
                    ));
                }
                continue;
            }
            let stage = self.q.len() / self.shape.cols;
            if p.at + n > p.stride || len < m * p.stride || n > stage {
                return refuse(format!(
                    "{name}: {n} rows at {} of a {}-value column of an output of {len} values; \
                     the stage holds {stage} rows a column",
                    p.at, p.stride
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

    /// The first `m` columns of `src` quantized once into input `which`'s
    /// forms the projections of `projs` read, then each projection over them
    /// by its weight's type into its output (`outs`, indexed by
    /// [`Proj::out`]); a part goes through the stage and one strided copy.
    /// Every refusal ([`GemmFront::check`]) comes before the first launch.
    fn project(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (l, m): (usize, usize),
        (which, src): (Input, Src<'_>),
        projs: &[Proj<'_>],
        outs: &mut [Dst<'_>],
    ) -> Result<(), GpuError> {
        let x_len = match &src {
            Src::Caller(x) => x.len(),
            Src::Heads => self.av.len(),
        };
        let out_lens: Vec<usize> = outs
            .iter()
            .map(|o| match o {
                Dst::Caller(y) => y.len(),
                Dst::Query => self.q.len(),
            })
            .collect();
        self.check(w, m, (which, x_len), projs, &out_lens)?;
        if let Some(p) = projs
            .iter()
            .find(|p| p.stride != 0 && matches!(outs[p.out], Dst::Query))
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{}: a part cannot be staged in the stage itself", p.name),
            });
        }
        let fault = self.fill(gpu, l, m)?;
        let tys = projs
            .iter()
            .map(|p| Ok(site::of_resident(w.resident(WHAT, p.name)?).ok_or(shape_ty(p.name))?))
            .collect::<Result<Vec<SiteTy>, GpuError>>()?;
        let GemmFront {
            route_k,
            k32,
            dense,
            embd,
            low,
            gated,
            q_low,
            heads,
            q,
            av,
            stats,
            ..
        } = self;
        let k = WideKernels {
            gemm: &*route_k,
            g32: Some(&*k32),
        };
        let acts = match which {
            Input::Embd => embd,
            Input::Low => low,
            Input::Gated => gated,
            Input::QLow => q_low,
            Input::Heads => heads,
        };
        let x: &DeviceBuffer<f32> = match src {
            Src::Caller(x) => x,
            Src::Heads => &*av,
        };
        acts.quantize(gpu, &k, fault, x, m, &tys)?;
        let want = WideActs::wants_of(&tys);
        stats.quantize += usize::from(want.q128) + usize::from(want.q32);
        let x_in = acts.input(x);
        for p in projs {
            let n = match &p.rows {
                Some(r) => r.len(),
                None => w.resident(WHAT, p.name)?.rows(),
            };
            match (&mut outs[p.out], p.stride) {
                (Dst::Caller(y), 0) => gemm(gpu, &k, dense, (w, p), &x_in, m, y)?,
                (Dst::Query, _) => gemm(gpu, &k, dense, (w, p), &x_in, m, q)?,
                (Dst::Caller(y), stride) => {
                    gemm(gpu, &k, dense, (w, p), &x_in, m, q)?;
                    site::scatter(gpu.stream(), &*q, (n, m), y, (p.at, stride))?;
                    stats.copy += 1;
                }
            }
            stats.gemm += 1;
        }
        Ok(())
    }
}

/// The error of a weight no entry of the kernel table reads.
fn shape_ty(name: &str) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: format!("{name}: a resident form no dense entry reads"),
    }
}

/// The site type of resident weight `weight` when the front runs it: Q8_0
/// planes, a K-quant or BF16 words; else why it does not.
fn wide_ty(weight: &DevWeight) -> Result<SiteTy, String> {
    let ty = site::of_resident(weight).ok_or("a resident form no dense entry reads")?;
    match ty {
        SiteTy::Q8_0 | SiteTy::Q3K | SiteTy::Q4K | SiteTy::Q5K | SiteTy::Q6K | SiteTy::Bf16 => {
            Ok(ty)
        }
        other => Err(format!(
            "{other} has a gemv and no wide GEMM the front runs (Q8_0, the K-quants and BF16 do)"
        )),
    }
}

/// Whether the rows of `weight` are launchable on the grouped GEMM when it is
/// a K-quant: a multiple of 16, and at Q6_K an even number of super-blocks a
/// row (`site::kgemv`'s rule for its row addresses).
fn weight_rows_ok(weight: &DevWeight) -> bool {
    match (site::of_resident(weight), weight_k(weight)) {
        (Some(SiteTy::Q6K), Some(k)) => {
            weight.rows().is_multiple_of(16) && (k / 256).is_multiple_of(2)
        }
        (Some(ty), _) if ty.kquant() => weight.rows().is_multiple_of(16),
        _ => true,
    }
}

/// The values a row of `weight` holds, for the forms the front runs.
fn weight_k(weight: &DevWeight) -> Option<usize> {
    match weight {
        DevWeight::Q8_0 { k, .. } | DevWeight::KQuant { k, .. } | DevWeight::Bf16 { k, .. } => {
            Some(*k)
        }
        _ => None,
    }
}

/// `name`'s rows exactly `want`, the width of the front's rows (`what`)
/// they are written into or read beside; refused by name otherwise.
fn check_rows(w: &Weights, name: &str, want: usize, what: &str) -> Result<(), GpuError> {
    let rows = w.resident(WHAT, name)?.rows();
    if rows == want {
        return Ok(());
    }
    Err(GpuError::Shape {
        what: WHAT,
        detail: format!("{name} has {rows} rows; the front's {what} were sized for {want}"),
    })
}

/// `y = W · x` over the dense table's columns for the projection `p` of
/// weight `w`, by the type it is resident as ([`site::resident_gemm`]).
fn gemm(
    gpu: &Gpu,
    k: &WideKernels<'_>,
    route: &GemmRoute,
    (w, p): (&Weights, &Proj<'_>),
    x: &WideIn<'_>,
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    match &p.rows {
        None => site::resident_gemm(gpu, k, (w, p.name), x, route, m, y),
        Some(r) => {
            site::resident_gemm_rows(gpu, k, (w, p.name, (r.start, r.len())), x, route, m, y)
        }
    }
}

/// Card bytes of a front of `shape` ([`GemmFront::bytes`]): the plan's
/// formula (`place::front_bytes`), which the plan reserves on the stage card.
#[must_use]
pub fn front_bytes(shape: FrontShape) -> usize {
    place::front_bytes(shape.cols, shape.widths)
}

const _: () = assert!(GEMM_BN == place::FRONT_TILE_COLS);
const _: () = assert!(GEMM32_STEP == place::FRONT_STEP);
