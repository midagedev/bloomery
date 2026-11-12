//! The per-layer input site of the Nano dialect (Raon-1-nano's
//! `Extra::PerLayerEmbed`, layout 2): one gather-and-combine that turns the
//! walk's ids into every layer's input vector, and one gated add a layer runs
//! after its feed-forward post-mix (llama.cpp `src/models/raon1nano.cpp`'s
//! graph, conversion/raon1_nano.py's folds).
//!
//! The combine, once a walk: `p = (RMSNorm_w(W_mp·e) + √P·table[id]) / √2`
//! for every layer's slice, where `e` is the embedding row the walk's own
//! gather reads (`token_embd`), `W_mp` is `per_layer_model_proj`
//! (`embedding_length_per_layer_input · trunk` rows), the gain
//! `per_layer_proj_norm` norms each layer's slice of the projection at the
//! block epsilon, and the table row `per_layer_token_embd[id]` holds the
//! trunk's slices in order. The launch order is the reference's: the table's
//! scale (`ggml_scale` by `√P`), the add, then the `1/√2` — each rounding on
//! its own. [`raon1_ple_combine`] writes the sums layer-major, so a layer's
//! slice is one span the gated add reads ([`PleBufs::p`]).
//!
//! The gated add, once a layer: the streams' mean (the head's own rule,
//! [`super::mhc`]'s `raon_hc_mean`) through the layer's gate `inp_gate`, the
//! GELU-tanh gated unit with the layer's slice of `p` (`ggml_geglu_split`:
//! `gelu(a)·b`, [`ElemKernels::enqueue_geglu`]), the back-projection `proj`,
//! its RMS norm's gain `post_norm` at the block epsilon, and the result
//! added to every stream ([`raon1_ple_add`]). The conversion pre-divides the
//! gate by the stream count, so the gate reads the mean the head's rule
//! computes, not the sum the reference feeds it — the same values, `Σ/4·W`
//! against `Σ·(W/4)`, every term an exact quarter.
//!
//! Every f32 product and sum the two entries make is an explicit
//! round-to-nearest intrinsic on the device. A non-finite value passes
//! through as itself, as the reference's elementwise ops carry it; the sites
//! and the norms raise their own faults before it reaches these entries.

use super::gather::{self, RowsTable};
use super::mhc::{MhcKernels, STREAMS};
use super::state::PASS_ROWS;
use crate::GpuError;
use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::arch::rows_kit::{RowsKit, project_rows};
use crate::launch_u32;
use crate::q38::Q38Kernels;
use crate::site::{SiteTy, vector};
use crate::tensor::Window;
use crate::weights::Weights;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::Split;
use model::arch::raon1::hparams::Hparams;
use model::arch::raon1::names;
use std::sync::Arc;

/// What the refusals of this site name.
pub(super) const WHAT: &str = "raon1 ple";

/// The types a per-layer input projection runs: the site families the body's
/// other projections take (`gpu/src/arch/raon1/rda.rs` `DENSE`), the F16 rows
/// the load widens among them.
pub(super) const TYPES: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];

/// Threads a block of the entries holds.
const BLOCK: u32 = 256;

#[allow(
    dead_code,
    reason = "#[cuda_module] derives an `_unchecked` launcher per entry that no caller uses"
)]
#[cuda_module]
mod raon1_ple_kernels {
    use super::*;

    /// `y[(l·m + t)·P + d] = (a[(t·T + l)·P + d] + s·rows[(t·T + l)·P + d])·c`
    /// — the combine of the token-major normed projection `a` and the
    /// gathered table rows `rows` into the layer-major per-layer inputs, `T`
    /// layers of `m` tokens of `P` values: one thread a (layer, token,
    /// value), each product and sum rounded on its own in the reference's
    /// order.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            a.len() >= layers * m * width,
            rows.len() >= layers * m * width,
            y.len() >= layers * m * width
        )
    )]
    pub fn raon1_ple_combine(
        a: &[f32],
        rows: &[f32],
        m: u32,
        layers: u32,
        width: u32,
        s: f32,
        c: f32,
        mut y: DisjointSlice<f32>,
    ) {
        let g = thread::index_1d().get();
        let (m, layers, width) = (m as usize, layers as usize, width as usize);
        if g >= layers * m * width {
            return;
        }
        let (l, r) = (g / (m * width), g % (m * width));
        let (t, d) = (r / width, r % width);
        let i = t * (layers * width) + l * width + d;
        // SAFETY: `i` and `g` are both below `layers·m·width`, which `a`,
        // `rows` and `y` hold at least by the contract; thread g alone
        // writes entry g.
        unsafe {
            let (av, rv) = (*a.get_unchecked(i), *rows.get_unchecked(i));
            *y.get_unchecked_mut(g) = mul_rn_f32(add_rn_f32(av, mul_rn_f32(rv, s)), c);
        }
    }

    /// `x[t·4n + i·n + d] += g[t·n + d]`, in place: the per-layer input added
    /// to every stream of every token — one thread a (token, stream, value),
    /// the sum rounded on its own.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= m * 4 * n, g.len() >= m * n)
    )]
    pub fn raon1_ple_add(g: &[f32], n: u32, m: u32, mut x: DisjointSlice<f32>) {
        let id = thread::index_1d().get();
        let (n, m) = (n as usize, m as usize);
        if id >= m * STREAMS * n {
            return;
        }
        let (t, r) = (id / (STREAMS * n), id % (STREAMS * n));
        let d = r % n;
        // SAFETY: `id < m·4n <= x.len()` and `t·n + d < m·n <= g.len()` by
        // the contract; thread id alone writes entry id.
        unsafe {
            let add = add_rn_f32(*x.get_unchecked_mut(id), *g.get_unchecked(t * n + d));
            *x.get_unchecked_mut(id) = add;
        }
    }
}

/// The loaded module.
pub struct PleKernels {
    module: raon1_ple_kernels::LoadedModule,
}

impl PleKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<PleKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(raon1_ple_kernels, ctx)? };
        Ok(PleKernels { module })
    }

    /// The combine of the normed projection `a` and the gathered rows `rows`
    /// of `m` tokens over `layers` layers of `width` values into the
    /// layer-major `y` ([`raon1_ple_combine`]): `s` the table's `√width`
    /// scale and `c` the `1/√2`, both finite and nonzero. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_combine(
        &self,
        stream: &CudaStream,
        (a, rows): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        (m, layers, width): (usize, usize, usize),
        (s, c): (f32, f32),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 ple::enqueue_combine";
        let n = layers
            .checked_mul(m)
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| GpuError::shape(what, "the per-layer inputs pass usize values"))?;
        if m == 0 || layers == 0 || width == 0 || a.len() < n || rows.len() < n || y.len() < n {
            return Err(GpuError::shape(
                what,
                format!(
                    "m={m}, layers={layers}, width={width}: a {} (want {n}), rows {} (want {n}), \
                     y {} (want {n})",
                    a.len(),
                    rows.len(),
                    y.len()
                ),
            ));
        }
        for (name, v) in [("the table's scale", s), ("the combine's constant", c)] {
            if !v.is_finite() || v == 0.0 {
                return Err(GpuError::shape(what, format!("{name} is {v}")));
            }
        }
        let total = launch_u32(what, "layers·m·width", n)?;
        let prep = self.module.prepare_raon1_ple_combine(LaunchConfig1D::new(
            total.div_ceil(BLOCK),
            BLOCK,
            0,
        ))?;
        self.module.raon1_ple_combine(
            stream,
            &prep,
            a,
            rows,
            launch_u32(what, "m", m)?,
            launch_u32(what, "layers", layers)?,
            launch_u32(what, "width", width)?,
            s,
            c,
            y,
        )?;
        Ok(())
    }

    /// The per-layer input `g` (`n` values a token, `m` tokens) added to
    /// every one of the [`STREAMS`] streams of `x` in place
    /// ([`raon1_ple_add`]). Asynchronous, allocation-free, capturable.
    pub fn enqueue_add(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        (n, m): (usize, usize),
        x: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "raon1 ple::enqueue_add";
        if m == 0 || x.len() < m * STREAMS * n || g.len() < m * n {
            return Err(GpuError::shape(
                what,
                format!(
                    "m={m}, n={n}: x {} (want {}), g {} (want {})",
                    x.len(),
                    m * STREAMS * n,
                    g.len(),
                    m * n
                ),
            ));
        }
        let total = launch_u32(what, "m·4n", m * STREAMS * n)?;
        let prep = self.module.prepare_raon1_ple_add(LaunchConfig1D::new(
            total.div_ceil(BLOCK),
            BLOCK,
            0,
        ))?;
        self.module.raon1_ple_add(
            stream,
            &prep,
            g,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            x,
        )?;
        Ok(())
    }
}

/// The per-layer input site of a file that carries one: its projections and
/// gains, and the widths the file's keys state.
pub(super) struct Ple {
    /// `embedding_length_per_layer_input`.
    pub(super) width: usize,
    /// The trunk's layers, each with its own slice and gate.
    pub(super) trunk: usize,
    /// `per_layer_model_proj`, the token row to every layer's input.
    model_proj: Site,
    /// A trunk layer's gate (from the streams' mean) and back-projection,
    /// and its `post_norm` gain's name.
    layers: Vec<(Site, Site, String)>,
    /// `per_layer_proj_norm`, the combine's norm gain.
    proj_norm: String,
}

impl Ple {
    /// The site of a file that carries a per-layer input ([`Hparams::ple`]),
    /// or `None` for one that carries none: the model projection over the
    /// whole trunk, every trunk layer's gate, back-projection and gain, and
    /// the combine's gain, each held to the file's dims. Load-time only.
    pub(super) fn load(file: &Split, w: &Weights, hp: &Hparams) -> Result<Option<Ple>, GpuError> {
        let Some(width) = hp.ple else {
            return Ok(None);
        };
        if !width.is_multiple_of(32) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the per-layer input is {width} values wide; the combine's norm takes a \
                     multiple of 32"
                ),
            ));
        }
        let (n, trunk) = (hp.n_embd, hp.n_trunk);
        let model_proj = Site::of_file(
            (file, w),
            WHAT,
            names::PER_LAYER_MODEL_PROJ.to_owned(),
            (width * trunk, n),
            TYPES,
        )?;
        let proj_norm = vector(w, WHAT, names::PER_LAYER_PROJ_NORM.to_owned(), width)?;
        let mut layers = Vec::with_capacity(trunk);
        for l in 0..trunk {
            let gate = Site::of_file(
                (file, w),
                WHAT,
                format!("blk.{l}.{}", names::INP_GATE),
                (width, n),
                TYPES,
            )?;
            let proj = Site::of_file(
                (file, w),
                WHAT,
                format!("blk.{l}.{}", names::PROJ),
                (n, width),
                TYPES,
            )?;
            let post_norm = vector(w, WHAT, format!("blk.{l}.{}", names::POST_NORM), n)?;
            layers.push((gate, proj, post_norm));
        }
        Ok(Some(Ple {
            width,
            trunk,
            model_proj,
            layers,
            proj_norm,
        }))
    }

    /// Every site the walks launch, for the load's [`Acts::new`].
    pub(super) fn sites(&self) -> Vec<&Site> {
        let mut out = vec![&self.model_proj];
        out.extend(self.layers.iter().flat_map(|(gate, proj, _)| [gate, proj]));
        out
    }

    /// The walk's combine (module doc): the token rows and the table's rows
    /// of `m` tokens gathered through [`super::gather`], the projection
    /// through [`Ple::model_proj`], its norm at `cx.eps` and the combine into
    /// `b.p`. Once a walk, before the layer loop; `ids` and `pos0` are the
    /// walk's own gather's, so the two reads agree on the tokens.
    pub(super) fn enqueue_setup(
        &self,
        cx: &GqaCx<'_>,
        (acts, kit, kq, ple): (&mut Acts, &mut RowsKit<'_>, &Q38Kernels, &PleKernels),
        (ids, pos0): (&DeviceBuffer<u32>, &DeviceBuffer<u32>),
        m: usize,
        b: &mut PleBufs,
    ) -> Result<(), GpuError> {
        let stream = cx.gpu.stream();
        let (width, trunk) = (self.width, self.trunk);
        gather::enqueue_rows(
            cx.gpu,
            kq,
            &RowsTable::of(cx.w, WHAT, &names::token_embd())?,
            (ids, pos0),
            (&mut b.e, &mut b.pos, &mut b.n_keys),
        )?;
        gather::enqueue_rows(
            cx.gpu,
            kq,
            &RowsTable::of(cx.w, WHAT, names::PER_LAYER_TOKEN_EMBD)?,
            (ids, pos0),
            (&mut b.rows, &mut b.pos, &mut b.n_keys),
        )?;
        project_rows(
            cx,
            kit,
            acts,
            (0, m),
            (&self.model_proj, width * trunk),
            &b.e,
            &mut b.proj,
        )?;
        let gain = cx.w.f32_buf(WHAT, &self.proj_norm)?;
        cx.gpu.elem().enqueue_rms_norm(
            stream,
            &b.proj,
            gain,
            cx.eps,
            width,
            m * trunk,
            &mut b.proj_n,
        )?;
        // The reference's two scales: `ggml_scale` by `sqrtf(n_ple)` on the
        // table rows, then the `1.0f/sqrtf(2.0f)` on their sum with the
        // projection (raon1nano.cpp's graph).
        let scale = (width as f32).sqrt();
        let combine = 1.0f32 / 2.0f32.sqrt();
        ple.enqueue_combine(
            stream,
            (&b.proj_n, &b.rows),
            (m, trunk, width),
            (scale, combine),
            &mut b.p,
        )
    }

    /// Layer `l`'s gated add (module doc): the streams `x`'s mean through
    /// the layer's gate, the gated unit with the layer's slice of `b.p`, the
    /// back-projection, its norm, and the sum added to every stream of `x`
    /// in place. Once a layer, after the feed-forward post-mix.
    pub(super) fn enqueue_layer(
        &self,
        cx: &GqaCx<'_>,
        (acts, kit, hc, ple): (&mut Acts, &mut RowsKit<'_>, &MhcKernels, &PleKernels),
        (l, m): (usize, usize),
        x: &mut DeviceBuffer<f32>,
        b: &mut PleBufs,
    ) -> Result<(), GpuError> {
        let stream = cx.gpu.stream();
        let (width, n) = (self.width, cx.hidden);
        let Some((gate, proj, post_norm)) = self.layers.get(l) else {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {l} of a trunk of {}", self.trunk),
            ));
        };
        hc.enqueue_mean(stream, x, (n, m), &mut b.mean)?;
        project_rows(cx, kit, acts, (l, m), (gate, width), &b.mean, &mut b.gate)?;
        let up = Window::<f32>::of(&b.p, l * m * width * size_of::<f32>(), m * width)?;
        cx.gpu
            .elem()
            .enqueue_geglu(stream, &b.gate, &up, m * width, &mut b.act)?;
        project_rows(cx, kit, acts, (l, m), (proj, n), &b.act, &mut b.g)?;
        let gain = cx.w.f32_buf(WHAT, post_norm)?;
        cx.gpu
            .elem()
            .enqueue_rms_norm(stream, &b.g, gain, cx.eps, n, m, &mut b.gn)?;
        ple.enqueue_add(stream, &b.gn, (n, m), x)
    }
}

/// The per-layer input's buffers over a walk's rows, token-major except `p`
/// (layer-major, a layer's slice one span): the step's one row or a pass's
/// [`PASS_ROWS`], every buffer at least the rows' width.
pub(super) struct PleBufs {
    /// The embedding rows a second gather reads, `n` a token.
    e: DeviceBuffer<f32>,
    /// The model projection and its norm, `width · trunk` a token each.
    proj: DeviceBuffer<f32>,
    proj_n: DeviceBuffer<f32>,
    /// The gathered table rows, `width · trunk` a token.
    rows: DeviceBuffer<f32>,
    /// The combined per-layer inputs, `width` a (layer, token).
    p: DeviceBuffer<f32>,
    /// A layer's gate rows, the gated unit's output, its back-projection and
    /// the normed projection, `width`/`width`/`n`/`n` a token.
    gate: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    g: DeviceBuffer<f32>,
    gn: DeviceBuffer<f32>,
    /// The streams' mean, `n` a token.
    mean: DeviceBuffer<f32>,
    /// The gathers' position and live key count side outputs, never read.
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
}

impl PleBufs {
    /// The buffers for walks of up to [`PASS_ROWS`] rows over a model of
    /// `n` values, `trunk` layers of `width`-value per-layer inputs.
    /// Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        (n, trunk, width): (usize, usize, usize),
    ) -> Result<PleBufs, GpuError> {
        let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, (PASS_ROWS * len).max(1));
        let u = |len: usize| DeviceBuffer::<u32>::zeroed(stream, PASS_ROWS.max(1) * len);
        Ok(PleBufs {
            e: z(n)?,
            proj: z(trunk * width)?,
            proj_n: z(trunk * width)?,
            rows: z(trunk * width)?,
            p: z(trunk * width)?,
            gate: z(width)?,
            act: z(width)?,
            g: z(n)?,
            gn: z(n)?,
            mean: z(n)?,
            pos: u(1)?,
            n_keys: u(1)?,
        })
    }

    /// Card bytes of the buffers.
    pub(super) fn bytes(&self) -> usize {
        [
            &self.e,
            &self.proj,
            &self.proj_n,
            &self.rows,
            &self.p,
            &self.gate,
            &self.act,
            &self.g,
            &self.gn,
            &self.mean,
        ]
        .into_iter()
        .map(DeviceBuffer::num_bytes)
        .sum::<usize>()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
    }

    /// Card bytes of the buffers [`PleBufs::new`] makes for a model of `n`
    /// values, `trunk` layers of `width`-value per-layer inputs: ten f32
    /// buffers — the rows `n + 4·trunk·width + 2·width + 2·n` wide — and the
    /// two u32 side outputs.
    pub(super) fn bytes_of((n, trunk, width): (usize, usize, usize)) -> usize {
        let f32s = PASS_ROWS * (n + 4 * trunk * width + 2 * width + 2 * n);
        4 * f32s + 2 * 4 * PASS_ROWS
    }
}
