//! Kimi K3's routed block on the card, over one row ([`Moe::step`]) or a pass's rows
//! ([`Moe::rows`]), from llama.cpp `src/models/kimi-k3.cpp` (`build_latent_moe`):
//!
//! 1. the layer's RMS norm of the row (`ffn_norm`);
//! 2. the router on the full-width normed row: the F32 rows, the sigmoid, the selection bias
//!    that steers the pick only, the top 16 of 896 renormalized (scale 1);
//! 3. the latent down (`ffn_routed_down`, Q6_K, hidden to latent) of the normed row;
//! 4. each kept expert on the latent row: the `situ` of its gate and up rows (IQ1_S or IQ2_XXS),
//!    the down rows (IQ1_S or IQ2_XXS), the weighted sum in slot order;
//! 5. the routed norm (`ffn_routed_norm`, RMS at the block's eps) of the sum;
//! 6. the latent up (`ffn_routed_up`, Q6_K, latent to hidden);
//! 7. the shared stack on the full-width normed row: the `situ` of its gate and up (Q6_K, two
//!    experts as one gated block of width 2 × `expert_ff`), its down;
//! 8. the sum of 6 and 7 added to the row.
//!
//! Every stage is the common routed block's (`routed_ffn`: the router instance, the card leg, the
//! `situ` GLU, the shared block) or one of its optional stages: the latent pair and the sum norm
//! of [`Routed`], both `None` for the families without them. The router instance is the one
//! constant of this file ([`Instance`]): a count no instance is built for is refused by name at
//! load.

use crate::arch::hybrid::{Acts, GqaCx, Site};
use crate::arch::routed_ffn::{
    self as ffn, Ffn, FfnKernels, FfnRows, FfnScratch, Glu, Latent, Routed, RoutedShape,
};
use crate::arch::rows_kit::{RowKernels, RowsKit, add_row_forms};
use crate::card_leg::LegStacks;
use crate::flash_gqa::FlashGqaKernels;
use crate::gemm::Gemm32Kernels;
use crate::kquant::Act;
use crate::linear::{LinearKernels, PASS_ROWS};
use crate::route_body::{sigmoid_8_2, sigmoid_896_16};
use crate::routed_block::RouterEntry;
use crate::site::{self, KGemvKernels, SiteTy};
use crate::weights::Weights;
use crate::{Gpu, GpuError};
use cuda_core::DeviceBuffer;
use gguf::Split;
use model::arch::kimi_k3::hparams::Hparams;
use model::arch::kimi_k3::names;
use model::placement::kernels::DownAct;

/// What the refusals of this file name.
const WHAT: &str = "kimi_k3 moe";

/// The types the latent pair and the shared stack have in the released set (`gguf::arch::kimi_k3`).
const SITE_TYPES: &[SiteTy] = &[SiteTy::Q6K];

/// Values in a super-block of the routed stacks' types (IQ1_S, IQ2_XXS) and of Q6_K: every row
/// of the block is a whole number of them.
const SUPER: usize = 256;

/// The `(experts, kept)` pairs the router instances are built for: the file's 896 with the top
/// 16 and the fixture's 8 with the top 2.
pub const INSTANCES: [(usize, usize); 2] = [
    (sigmoid_896_16::N_EXPERT, sigmoid_896_16::N_USED),
    (sigmoid_8_2::N_EXPERT, sigmoid_8_2::N_USED),
];

/// The widths and rules of one routed layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeDims {
    /// `embedding_length`: the row's width, the router's and the shared stack's input.
    pub hidden: usize,
    /// `expert_latent_length`: the width the routed experts run at.
    pub latent: usize,
    /// `expert_feed_forward_length`: one routed expert's width.
    pub expert_ff: usize,
    /// The shared stack's width: `expert_shared_count` experts of `expert_ff` as one block.
    pub shared_ff: usize,
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_weights_scale`.
    pub scale: f32,
    /// `expert_weights_norm`: the kept weights renormalize to sum to one.
    pub weights_norm: bool,
    /// `attention.layer_norm_rms_epsilon`: the block's norm and the routed sum's.
    pub eps: f32,
    /// `activation.situ_beta`: the gate's bound.
    pub situ_beta: f32,
    /// `activation.situ_linear_beta`: the up path's bound.
    pub situ_linear_beta: f32,
}

impl MoeDims {
    /// The widths and rules `hp` reads from a file's header.
    #[must_use]
    pub fn of(hp: &Hparams) -> MoeDims {
        MoeDims {
            hidden: hp.n_embd,
            latent: hp.latent,
            expert_ff: hp.experts.ff,
            shared_ff: hp.experts.shared_ff(),
            n_expert: hp.experts.n_expert,
            n_used: hp.experts.n_used,
            scale: hp.experts.weights_scale,
            weights_norm: hp.experts.weights_norm,
            eps: hp.rms_eps,
            situ_beta: hp.situ.gate_beta,
            situ_linear_beta: hp.situ.value_beta,
        }
    }

    /// The `situ` rule the block's gate·up launches run.
    pub(super) fn rule(&self) -> Act {
        Act::Situ {
            beta: self.situ_beta,
            linear_beta: self.situ_linear_beta,
        }
    }

    /// The routed experts' shape: latent rows, the routed sum normed, the down stack's
    /// activation form.
    pub(super) fn shape(&self) -> RoutedShape {
        RoutedShape {
            expert_ff: self.expert_ff,
            latent: Some(self.latent),
            sum_norm: true,
            down: DownAct::Q8Act,
        }
    }

    /// Whether the widths and the router fit what the block runs, else the refusal naming the
    /// one that does not.
    pub(super) fn check(&self) -> Result<(), GpuError> {
        let refuse = |what: &str, v: usize| {
            GpuError::shape(
                WHAT,
                format!(
                    "{what} is {v}: the routed stacks (IQ1_S, IQ2_XXS) and the Q6_K sites hold \
                     rows of whole super-blocks of {SUPER} values"
                ),
            )
        };
        for (what, v) in [
            (
                "the hidden (the router's, the latent down's and the shared stack's rows)",
                self.hidden,
            ),
            (
                "the latent (the gate and up stacks' rows, the latent up's rows)",
                self.latent,
            ),
            ("the experts' width (the down stack's rows)", self.expert_ff),
            ("the shared stack's width (its down's rows)", self.shared_ff),
        ] {
            if v == 0 || !v.is_multiple_of(SUPER) {
                return Err(refuse(what, v));
            }
        }
        if !self.weights_norm {
            return Err(GpuError::shape(
                WHAT,
                "the router keeps its weights unnormalized: the card's router renormalizes the \
                 kept scores to sum to one",
            ));
        }
        if !INSTANCES.contains(&(self.n_expert, self.n_used)) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a router of {} experts, top {}: the instances route over {INSTANCES:?}",
                    self.n_expert, self.n_used
                ),
            ));
        }
        self.rule().code(WHAT).map(drop)
    }
}

/// The kernels a block launches beside the common routed block's: the sites' and the row
/// passes'.
struct Kern<R: RouterEntry> {
    kgemv: KGemvKernels,
    g32: Gemm32Kernels,
    flash: FlashGqaKernels,
    linear: LinearKernels,
    rows: RowKernels,
    ffn: FfnKernels<R>,
}

impl<R: RouterEntry> Kern<R> {
    fn load(gpu: &Gpu) -> Result<Kern<R>, GpuError> {
        let ctx = gpu.context();
        Ok(Kern {
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash: FlashGqaKernels::load(ctx)?,
            linear: LinearKernels::load(ctx)?,
            rows: RowKernels::load(ctx)?,
            ffn: FfnKernels::load(gpu)?,
        })
    }
}

/// Layer `l`'s routed block of `file`'s resident weights `w` at `dims` (the sites, the router
/// rows, the routed stacks, the latent pair and the sum norm, each held to its width and type),
/// and the name of the layer's `ffn_norm`. `stacks` is whether the layer's routed stacks are
/// resident: false on a placed load for a layer whose experts the card holds none of. Load-time
/// only.
pub(super) fn routed_of(
    (file, w): (&Split, &Weights),
    l: usize,
    dims: &MoeDims,
    stacks: bool,
) -> Result<(Routed, String), GpuError> {
    let MoeDims {
        hidden: n,
        latent,
        shared_ff,
        n_expert,
        ..
    } = *dims;
    let site = |name: String, (rows, k): (usize, usize)| {
        Site::of_file((file, w), WHAT, name, (rows, k), SITE_TYPES)
    };
    let glu = || -> Result<Glu, GpuError> {
        Ok(Glu {
            gate: site(names::ffn_gate_shexp(l), (shared_ff, n))?,
            up: site(names::ffn_up_shexp(l), (shared_ff, n))?,
            down: site(names::ffn_down_shexp(l), (n, shared_ff))?,
            ff: shared_ff,
        })
    };
    let router = names::ffn_gate_inp(l);
    let got = w.f32_tensor(WHAT, &router)?;
    if (got.rows(), got.cols()) != (n_expert, n) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{router} is {} rows of {}: the router scores {n_expert} experts of the \
                 {n}-wide row",
                got.rows(),
                got.cols()
            ),
        ));
    }
    let routed = Routed {
        bias: site::vector(w, WHAT, names::exp_probs_b(l), n_expert)?,
        router,
        stacks: [
            names::ffn_gate_exps(l),
            names::ffn_up_exps(l),
            names::ffn_down_exps(l),
        ],
        shared: glu().map_err(|e| layer_refusal(l, "the shared stack", &e))?,
        latent: Some(Box::new(Latent {
            down: site(names::ffn_routed_down(l), (latent, n))?,
            up: site(names::ffn_routed_up(l), (n, latent))?,
            width: latent,
        })),
        sum_norm: Some(site::vector(w, WHAT, names::ffn_routed_norm(l), latent)?),
    };
    // The stacks' types and shapes, refused here by name rather than at the first launch.
    if stacks {
        LegStacks::of_resident(w, routed.stacks.each_ref().map(String::as_str)).map(drop)?;
    }
    let norm = site::vector(w, WHAT, names::ffn_norm(l), n)?;
    Ok((routed, norm))
}

/// One layer's block at one router instance.
struct Block<R: RouterEntry> {
    dims: MoeDims,
    layer: usize,
    norm: String,
    ffn: Ffn,
    k: Kern<R>,
    acts: Acts,
    step: FfnScratch<R>,
    pass: FfnRows<R>,
    stage: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    delta: DeviceBuffer<f32>,
}

impl<R: RouterEntry> Block<R> {
    fn load(
        gpu: &Gpu,
        (file, w): (&Split, &Weights),
        l: usize,
        dims: MoeDims,
    ) -> Result<Block<R>, GpuError> {
        let stream = gpu.stream();
        let MoeDims {
            hidden: n,
            latent,
            shared_ff,
            ..
        } = dims;
        let (routed, norm) = routed_of((file, w), l, &dims, true)?;
        let ffn = Ffn::Routed(routed);
        let sites = match &ffn {
            Ffn::Routed(r) => r.sites(),
            Ffn::Dense(_) => Vec::new(),
        };
        let mut acts = Acts::new(stream, sites.iter().copied())?;
        add_row_forms(stream, &mut acts, sites.iter().copied())?;
        let shape = dims.shape();
        let wide = n.max(latent).max(shared_ff);
        Ok(Block {
            dims,
            layer: l,
            norm,
            k: Kern::load(gpu)?,
            acts,
            step: FfnScratch::new(stream, n, shared_ff, shape)?,
            pass: FfnRows::new(stream, n, (shared_ff, shape), PASS_ROWS)?,
            stage: DeviceBuffer::zeroed(stream, PASS_ROWS * wide)?,
            xn: DeviceBuffer::zeroed(stream, PASS_ROWS * n)?,
            delta: DeviceBuffer::zeroed(stream, PASS_ROWS * n)?,
            ffn,
        })
    }

    /// `y = x + block(x)` over `m` rows of `x`: one row through [`ffn::ffn`], 2 to
    /// [`PASS_ROWS`] through [`ffn::ffn_rows`].
    fn run(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (m, x): (usize, &DeviceBuffer<f32>),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n, l) = (self.dims.hidden, self.layer);
        if m == 0 || m > PASS_ROWS {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {l}: a pass of {m} rows (1..={PASS_ROWS})"),
            ));
        }
        let stream = gpu.stream();
        let rule = self.dims.rule();
        let scale = self.dims.scale;
        gpu.elem().enqueue_rms_norm(
            stream,
            x,
            w.f32_buf(WHAT, &self.norm)?,
            self.dims.eps,
            n,
            m,
            &mut self.xn,
        )?;
        let cx = GqaCx {
            what: WHAT,
            gpu,
            w,
            kgemv: &self.k.kgemv,
            g32: &self.k.g32,
            flash: &self.k.flash,
            mma: false,
            out_gate: None,
            hidden: n,
            eps: self.dims.eps,
            n_kv: 1,
            ctx: 1,
        };
        if m == 1 {
            ffn::ffn(
                &cx,
                &self.k.ffn,
                &mut self.acts,
                (rule, scale),
                (l, &self.ffn),
                &self.xn,
                &mut self.step,
                &mut self.delta,
            )?;
        } else {
            let kit = RowsKit {
                linear: &self.k.linear,
                rows: &self.k.rows,
                stage: &mut self.stage,
            };
            ffn::ffn_rows(
                &cx,
                &self.k.ffn,
                &mut self.acts,
                (rule, scale),
                (l, &self.ffn),
                m,
                &self.xn,
                &mut self.pass,
                &mut self.delta,
                kit,
            )?;
        }
        gpu.elem().enqueue_add(stream, x, &self.delta, m * n, y)
    }

    fn picks(&self, gpu: &Gpu, rows: bool) -> Result<(Vec<u32>, Vec<f32>), GpuError> {
        if rows {
            self.pass.picks(gpu.stream())
        } else {
            self.step.picks(gpu.stream())
        }
    }

    fn bytes(&self) -> usize {
        self.step.bytes()
            + self.pass.bytes()
            + self.stage.num_bytes()
            + self.xn.num_bytes()
            + self.delta.num_bytes()
    }
}

/// `err` as layer `l`'s refusal about `part`.
fn layer_refusal(l: usize, part: &str, err: &GpuError) -> GpuError {
    GpuError::shape(WHAT, format!("layer {l}'s {part}: {err}"))
}

/// The router instance a layer's counts select.
enum Instance {
    /// The file's 896 experts, the top 16.
    E896(Block<sigmoid_896_16::RouterKernels>),
    /// The fixture's 8 experts, the top 2.
    E8(Block<sigmoid_8_2::RouterKernels>),
}

/// One Kimi K3 routed layer on the card: its resident pieces, kernels and scratch, at the router
/// instance its expert counts select.
pub struct Moe(Instance);

impl Moe {
    /// Layer `l`'s block of `file`'s resident weights `w` at `dims` (a file's header through
    /// [`MoeDims::of`]). A width, a router or a weight the block does not run is refused by
    /// name. Load-time only.
    ///
    /// # Errors
    /// [`GpuError::Shape`] or a weight error naming the width, tensor or type.
    pub fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        l: usize,
        dims: MoeDims,
    ) -> Result<Moe, GpuError> {
        dims.check()?;
        let counts = (dims.n_expert, dims.n_used);
        Ok(Moe(if counts == INSTANCES[0] {
            Instance::E896(Block::load(gpu, (file, w), l, dims)?)
        } else {
            Instance::E8(Block::load(gpu, (file, w), l, dims)?)
        }))
    }

    /// `y = x + block(norm(x))` for the one row `x` of `hidden` values.
    ///
    /// # Errors
    /// [`GpuError`] from a launch or a weight.
    pub fn step(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.rows(gpu, w, (1, x), y)
    }

    /// [`Moe::step`] over `m` rows of `x` (1 to [`PASS_ROWS`], token-major): each row the
    /// one-row step's bits.
    ///
    /// # Errors
    /// [`GpuError`] from a launch or a weight, or a row count the block does not take.
    pub fn rows(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        (m, x): (usize, &DeviceBuffer<f32>),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        match &mut self.0 {
            Instance::E896(b) => b.run(gpu, w, (m, x), y),
            Instance::E8(b) => b.run(gpu, w, (m, x), y),
        }
    }

    /// The last launch's picks read back: each row's kept expert ids in rank order and its
    /// weights, row-major (`rows`: the last launch was a pass of rows, not a step). Synchronizes
    /// the stream.
    ///
    /// # Errors
    /// [`GpuError`] from the readback.
    pub fn picks(&self, gpu: &Gpu, rows: bool) -> Result<(Vec<u32>, Vec<f32>), GpuError> {
        match &self.0 {
            Instance::E896(b) => b.picks(gpu, rows),
            Instance::E8(b) => b.picks(gpu, rows),
        }
    }

    /// Device bytes of the block's scratch.
    #[must_use]
    pub fn scratch_bytes(&self) -> usize {
        match &self.0 {
            Instance::E896(b) => b.bytes(),
            Instance::E8(b) => b.bytes(),
        }
    }
}
