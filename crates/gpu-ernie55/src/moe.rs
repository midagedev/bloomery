//! The latent mixture of experts of an ERNIE 5.5 routed layer, the feed-forward block past the
//! leading dense layer: over one row ([`Moe::enqueue_step`]) or a prompt unit's rows
//! ([`Moe::enqueue_prompt`]), and in the three parts a layer whose experts the host tier serves a
//! share of runs around the host leg's go and wait ([`Moe::enqueue_front`],
//! [`Moe::enqueue_shadow`], [`Moe::enqueue_back`], and their rows forms for a prompt batch).
//! `tools/ref/ernie55_ref.py` `_moe_ffn` is the oracle (llama.cpp PR #31925 `build_latent_moe`,
//! vLLM-ERNIE `moe.py`):
//!
//! 1. the router on the full-width normed row `x`: two projections (BF16 in the file, f32 on the
//!    card), an expert's score the sum of their sigmoids ([`sigmoid_split_512_10`]); the selection
//!    bias steers the pick only, the kept scores renormalize, each weight times a learned
//!    per-expert scale;
//! 2. the latent down (`ffn_routed_down`, hidden to latent) of `x`: the experts' rows;
//! 3. the ten picked experts over a latent row: the MXFP4 gate, up and down stacks under SiTU
//!    ([`CardLeg`]: the gate·up `_sel`, the down `_sel`, the slots' weighted sum in slot order).
//!    On the card, the experts the plan keeps there ([`LegRun::n_card`]); the rest on the common
//!    host tier ([`host_layer`]: `HostLayer::build_situ` over the same stacks in the file's
//!    blocks), the host's sum added to the card's before the norm (the routed sum is linear in
//!    the experts' rows);
//! 4. the routed sum's RMS norm (`ffn_routed_norm`);
//! 5. the latent up (`ffn_routed_up`, latent to hidden);
//! 6. the shared expert on `x`: the FP8 128-block projections and SiTU, no gate;
//! 7. the sum of 5 and 6 into `out`, for the layer's post mix.
//!
//! Every launch is one a decode step runs, so the whole is graph-capturable: no host sync and no
//! allocation between a layer's front and its back. The step's picks stay in the buffers for a
//! gate's instrument ([`MoeBufs::picks`]).
//!
//! What the file forces here, each named where it is used: the split-feature router and its
//! learned scale, the latent width, the routed sum's norm, the SiTU betas, the ten slots. The
//! rest is the common owners': the router bodies, the card leg, the dense sites, the host tier's
//! SiTU form. The residency machine's expert store needs nothing of this family: the stacks sit
//! under the common names in the file's blocks, the shape `hosted_stacks` reads.
//!
//! [`sigmoid_split_512_10`]: bloomery_gpu::route_body::sigmoid_split_512_10
//! [`hosted_stacks`]: bloomery_gpu::host::swap_source::hosted_stacks

use bloomery_gpu::arch::hybrid::{Acts, GqaCx, Site, WideCx, project_wide, wide_acts_at};
use bloomery_gpu::card_leg::{CardLeg, LegBufs, LegRun, LegShape, LegStacks, LegSum};
use bloomery_gpu::gemm::{Gemm32Kernels, GemmKernels, GemmRoute};
use bloomery_gpu::host::placed::SideShape;
use bloomery_gpu::host::run::HostWidths;
use bloomery_gpu::kquant::Act;
use bloomery_gpu::route_body::sigmoid_split_512_10 as split;
use bloomery_gpu::routed_block::RoutedRows;
use bloomery_gpu::site::{self, SiteTy, WideActs, WideKernels};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::ernie55::hparams::Hparams;
use model::arch::ernie55::names;
use model::arch::names::routed_stacks;
use model::moe::{HostLayer, SituSpec};
use model::placement::kernels::{DownAct, DownForms};
use model::r8file::R8Source;

/// What the refusals of this file name.
pub const WHAT: &str = "ernie55 moe";

/// The types the latent pair and the shared expert's projections run: the checkpoint's FP8 in
/// 128-value blocks, and the Q8_0 a reader writes an unquantized projection as. A site of another
/// type is refused by name at the load.
const SITE_TYPES: &[SiteTy] = &[SiteTy::Fp8B128, SiteTy::Q8_0];

/// The widths and rules of the routed block, from the file's header.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeDims {
    /// `embedding_length`: the row's width, the router's and the shared expert's input.
    pub hidden: usize,
    /// `expert_latent_length`: the width the routed experts run at.
    pub latent: usize,
    /// `expert_feed_forward_length`: one routed expert's width.
    pub expert_ff: usize,
    /// The shared expert's width (`expert_shared_feed_forward_length`).
    pub shared_ff: usize,
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_weights_norm`: the kept weights renormalize to sum to one.
    pub weights_norm: bool,
    /// `attention.layer_norm_rms_epsilon`: the routed sum's norm reads it, as every norm's.
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
            weights_norm: hp.experts.weights_norm,
            eps: hp.rms_eps,
            situ_beta: hp.situ.gate_beta,
            situ_linear_beta: hp.situ.value_beta,
        }
    }

    /// The `situ` rule the experts' and the shared expert's gate·up launches run.
    #[must_use]
    pub fn rule(&self) -> Act {
        Act::Situ {
            beta: self.situ_beta,
            linear_beta: self.situ_linear_beta,
        }
    }

    /// The card leg's shape for a run of up to `cap` tokens: [`MoeDims::n_used`] slots a token
    /// over rows of the latent's width, through experts of their own width.
    #[must_use]
    pub const fn leg_shape(&self, cap: usize) -> LegShape {
        LegShape {
            cap,
            slots_per_col: self.n_used,
            n: self.latent,
            ff: self.expert_ff,
        }
    }

    /// The routed run a placed load's host side is made for: the experts, the slots and the
    /// latent's width.
    #[must_use]
    pub const fn side_shape(&self) -> SideShape {
        SideShape {
            n_expert: self.n_expert,
            n_used: self.n_used,
            width: self.latent,
        }
    }

    /// The widths a host call serves: the latent's, the experts' and the slots a token.
    #[must_use]
    pub const fn host_widths(&self) -> HostWidths {
        HostWidths {
            embd: self.latent,
            ff: self.expert_ff,
            n_used: self.n_used,
        }
    }

    /// Whether the widths and the router fit what the block runs, else the refusal naming the one
    /// that does not.
    pub fn check(&self) -> Result<(), GpuError> {
        let refuse = |what: &str, v: usize, unit: usize| {
            GpuError::shape(
                WHAT,
                format!("{what} is {v}: the block runs it in whole units of {unit}"),
            )
        };
        // The rows the FP8_B128 blocks of 128 cover, and the router tile's staged lines of 64.
        for (what, v) in [
            ("the hidden", self.hidden),
            ("the shared expert's width", self.shared_ff),
        ] {
            if v == 0 || !v.is_multiple_of(128) {
                return Err(refuse(what, v, 128));
            }
        }
        // The card leg's q8_1 columns: super-blocks of 256.
        for (what, v) in [
            ("the latent", self.latent),
            ("the routed experts' width", self.expert_ff),
        ] {
            if v == 0 || !v.is_multiple_of(256) {
                return Err(refuse(what, v, 256));
            }
        }
        if !self.weights_norm {
            return Err(GpuError::shape(
                WHAT,
                "the router keeps its weights unnormalized: the card's router renormalizes the \
                 kept scores over their sum",
            ));
        }
        if (self.n_expert, self.n_used) != (split::N_EXPERT, split::N_USED) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a router of {} experts, top {}: the split-feature instance routes over {} and {}",
                    self.n_expert,
                    self.n_used,
                    split::N_EXPERT,
                    split::N_USED
                ),
            ));
        }
        if !(self.eps.is_finite() && self.eps >= 0.0) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the norm epsilon is {}, not finite and not negative",
                    self.eps
                ),
            ));
        }
        self.rule().code(WHAT).map(drop)
    }
}

/// The family's part of the host tier: layer `l`'s routed experts as the common host tier serves
/// them ([`HostLayer::build_situ`]), the file's three stacks at the latent's width combined by
/// SiTU at the file's betas. Load-time only.
///
/// # Errors
///
/// [`model::ModelError`]: a stack missing or of another shape, a width no host kernel serves.
pub fn host_layer(
    src: R8Source<'_>,
    dims: &MoeDims,
    l: usize,
) -> Result<HostLayer, model::ModelError> {
    HostLayer::build_situ(
        src,
        &SituSpec {
            gate: &names::ffn_gate_exps(l),
            up: &names::ffn_up_exps(l),
            down: &names::ffn_down_exps(l),
            n_expert: dims.n_expert,
            embd: dims.latent,
            ff: dims.expert_ff,
            beta: dims.situ_beta,
            linear_beta: dims.situ_linear_beta,
        },
    )
}

/// The check a placed load's host side runs on layer `l` with `on_card` of its experts on the
/// card (`PlacedSide::new`'s `check`): the card's stacks hold exactly those experts, of the
/// shape the leg runs.
pub fn check_card_stacks(
    w: &Weights,
    dims: &MoeDims,
    l: usize,
    on_card: usize,
) -> Result<(), GpuError> {
    LegStacks::of(w, routed_stacks(l).each_ref().map(String::as_str))?
        .check(on_card, dims.leg_shape(1))
}

/// A gate·up·down block's three sites and its width.
struct Glu {
    gate: Site,
    up: Site,
    down: Site,
    /// The block's width: the gate's and up's rows.
    ff: usize,
}

impl Glu {
    fn sites(&self) -> [&Site; 3] {
        [&self.gate, &self.up, &self.down]
    }
}

/// One routed layer's resident pieces, by the description's names.
struct Routed {
    /// The two router matrices, `ffn_gate_inp` and `ffn_gate_inp_1`.
    router: [String; 2],
    /// The selection bias (`exp_probs_b`), for the pick only.
    bias: String,
    /// The learned per-expert scale (`ffn_exp_scale`).
    escale: String,
    /// The gate, up and down stacks' names.
    stacks: [String; 3],
    /// `ffn_routed_down`: the normed row to the experts' latent row.
    down_lat: Site,
    /// `ffn_routed_up`: the normed routed sum to the hidden row.
    up_lat: Site,
    /// `ffn_routed_norm`: the gain of the routed sum's norm.
    sum_norm: String,
    shared: Glu,
}

impl Routed {
    /// Layer `l`'s pieces of `file`'s resident weights `w` at `dims`, each held to its width and
    /// type. A layer whose stacks `w` holds some of but not all of is refused by name; the stacks
    /// it holds are whole experts, the first of the file's (a placed load's card share, the rest the
    /// host tier's) or all of them; a layer whose stacks it holds none of is a placed load's, the
    /// host tier serving it alone. Load-time only.
    fn load((file, w): (&Split, &Weights), l: usize, dims: &MoeDims) -> Result<Routed, GpuError> {
        let MoeDims {
            hidden: n,
            latent,
            shared_ff,
            n_expert: e,
            ..
        } = *dims;
        let site = |name: String, (rows, k): (usize, usize)| {
            Site::of_file((file, w), WHAT, name, (rows, k), SITE_TYPES)
        };
        let router = [names::ffn_gate_inp(l), names::ffn_gate_inp_1(l)];
        for name in &router {
            let got = w.f32_tensor(WHAT, name)?;
            if (got.rows(), got.cols()) != (e, n) {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{name} is {} rows of {}: the router scores {e} experts of the {n}-wide row",
                        got.rows(),
                        got.cols()
                    ),
                ));
            }
        }
        let stacks = [
            names::ffn_gate_exps(l),
            names::ffn_up_exps(l),
            names::ffn_down_exps(l),
        ];
        let resident = stacks.iter().filter(|s| w.get(s).is_some()).count();
        if resident != 0 && resident != 3 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: {resident} of its three routed stacks are resident; a layer holds \
                     all of them or none"
                ),
            ));
        }
        if resident == 3 {
            // The stacks' types and shapes, refused here by name rather than at the first launch:
            // whole experts of the shape, the first of the file's (a placed load's card share) or
            // all of them.
            let shape = dims.leg_shape(1);
            let gate_rows = w.resident(WHAT, &stacks[0])?.rows();
            let n_card = match (gate_rows % shape.ff, gate_rows / shape.ff) {
                (0, n) if n > 0 && n <= e => n,
                _ => {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "layer {l}: the card's stacks hold {gate_rows} gate rows: whole experts \
                             of {} rows, 1 to {e} of them",
                            shape.ff
                        ),
                    ));
                }
            };
            LegStacks::of(w, stacks.each_ref().map(String::as_str))?.check(n_card, shape)?;
        }
        Ok(Routed {
            router,
            bias: site::vector(w, WHAT, names::exp_probs_b(l), e)?,
            escale: site::vector(w, WHAT, names::ffn_exp_scale(l), e)?,
            stacks,
            down_lat: site(names::ffn_routed_down(l), (latent, n))?,
            up_lat: site(names::ffn_routed_up(l), (n, latent))?,
            sum_norm: site::vector(w, WHAT, names::ffn_routed_norm(l), latent)?,
            shared: Glu {
                gate: site(names::ffn_gate_shexp(l), (shared_ff, n))?,
                up: site(names::ffn_up_shexp(l), (shared_ff, n))?,
                down: site(names::ffn_down_shexp(l), (n, shared_ff))?,
                ff: shared_ff,
            },
        })
    }

    /// The sites the layer's launches read: the shared expert's, then the latent pair's.
    fn sites(&self) -> Vec<&Site> {
        let mut out: Vec<&Site> = self.shared.sites().to_vec();
        out.extend([&self.down_lat, &self.up_lat]);
        out
    }

    /// The stacks' names as the card leg reads them.
    fn stack_names(&self) -> [&str; 3] {
        self.stacks.each_ref().map(String::as_str)
    }
}

/// The routed block of every routed layer: the layers' pieces and the two kernel sets the common
/// owners take at this family's counts, the split-feature router's and the card leg's. Read-only
/// after the load; what a launch writes is a [`MoeBufs`].
pub struct Moe {
    dims: MoeDims,
    /// Per layer of the model, `None` on a layer with no routed block.
    layers: Vec<Option<Routed>>,
    router: split::RouterKernels,
    leg: CardLeg,
}

/// The block's buffers: a step's one row and a prompt unit's up to `rows` rows, token-major, and
/// the forms its sites read on either arm. A step and a prompt unit share them, so one runs at a
/// time.
pub struct MoeBufs {
    rows: usize,
    /// The step's router output: each buffer one token's.
    route: split::RouterOut,
    /// Whether the last router launch was a unit's rows (its picks in `ids` and `weights`),
    /// else a step's (in `route`).
    unit_route: bool,
    /// A unit's router rows: the scores, router 1's logits, the picks' ids and weights.
    probs: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    /// The experts' rows: the latent down's output.
    lat: DeviceBuffer<f32>,
    /// The card slots' weighted sum.
    routed: DeviceBuffer<f32>,
    /// The card's sum and the host's, added.
    summed: DeviceBuffer<f32>,
    /// The routed sum through its norm.
    normed: DeviceBuffer<f32>,
    /// The latent up's output.
    lat_up: DeviceBuffer<f32>,
    /// The shared expert's gate and up sums, their SiTU and its down's output.
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    act: DeviceBuffer<f32>,
    shared: DeviceBuffer<f32>,
    /// The card leg's buffers, for runs of up to `rows` tokens.
    leg: LegBufs,
    /// The narrow arm's quantized forms of the sites.
    acts: Acts,
    /// The wide arm's forms, per input width a site reads.
    wide: Vec<(usize, WideActs)>,
    /// The wide arm's one-expert table.
    dense: GemmRoute,
}

/// The last step's router output read back by [`MoeBufs::scores`], each [`split::N_EXPERT`] long.
#[derive(Clone, Debug)]
pub struct RouterRead {
    /// Router 0's logits.
    pub logits_0: Vec<f32>,
    /// Router 1's logits.
    pub logits_1: Vec<f32>,
    /// The summed scores of the two routers' sigmoids.
    pub probs: Vec<f32>,
}

/// A stage of the block whose rows [`MoeBufs::tap`] reads back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tap {
    /// The latent down's output: the experts' rows.
    Lat,
    /// The card slots' weighted sum.
    Routed,
    /// The card's sum and the host's, added.
    Summed,
    /// The routed sum through its norm.
    Normed,
    /// The latent up's output.
    LatUp,
    /// The shared expert's output.
    Shared,
    /// The card leg's gate·up rows under SiTU, a slot's `ff` rows a row.
    Act,
    /// The card leg's down rows, a slot's `latent` rows a row.
    Down,
    /// A unit's router scores, [`split::N_EXPERT`] a row.
    Probs,
}

impl MoeBufs {
    /// The buffers for units of up to `rows` rows over the sites of the layers: the leg's columns
    /// in each form of `downs`, the sites' forms on either arm. Load-time only.
    fn new(
        stream: &CudaStream,
        (dims, rows): (&MoeDims, usize),
        downs: DownForms,
        sites: &[&Site],
    ) -> Result<MoeBufs, GpuError> {
        let z = |len: usize| -> Result<DeviceBuffer<f32>, GpuError> {
            Ok(DeviceBuffer::zeroed(stream, len.max(1))?)
        };
        let MoeDims {
            hidden,
            latent,
            shared_ff,
            n_expert,
            n_used,
            ..
        } = *dims;
        let mut widths: Vec<usize> = sites.iter().map(|s| s.k()).collect();
        widths.sort_unstable();
        widths.dedup();
        let wide = widths
            .into_iter()
            .map(|k| Ok((k, wide_acts_at(stream, sites.iter().copied(), (rows, k))?)))
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok(MoeBufs {
            rows,
            route: split::RouterOut::new(stream)?,
            unit_route: false,
            probs: z(rows * n_expert)?,
            logits: z(rows * n_expert)?,
            ids: DeviceBuffer::zeroed(stream, rows * n_used)?,
            weights: z(rows * n_used)?,
            lat: z(rows * latent)?,
            routed: z(rows * latent)?,
            summed: z(rows * latent)?,
            normed: z(rows * latent)?,
            lat_up: z(rows * hidden)?,
            gate: z(rows * shared_ff)?,
            up: z(rows * shared_ff)?,
            act: z(rows * shared_ff)?,
            shared: z(rows * hidden)?,
            leg: LegBufs::for_downs(stream, dims.leg_shape(rows), downs)?,
            acts: Acts::new(stream, sites.iter().copied())?,
            wide,
            dense: GemmRoute::new(stream, rows, 1)?,
        })
    }

    /// The rows a prompt unit holds at most.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The last router launch's picks on the card, row-major `[m, n_used]`: a step's one row
    /// (the router's output) or a unit's rows, the ids and their final weights (renormalized,
    /// times the learned scale). The placed handoff's input and a gate's taps; no read back.
    #[must_use]
    pub fn route(&self) -> (&DeviceBuffer<u32>, &DeviceBuffer<f32>) {
        if self.unit_route {
            (&self.ids, &self.weights)
        } else {
            (&self.route.ids, &self.route.weights)
        }
    }

    /// The last step's picks read back: the kept expert ids in rank order and their weights (a
    /// gate's instrument; it synchronizes the stream).
    pub fn picks(&self, stream: &CudaStream) -> Result<(Vec<u32>, Vec<f32>), GpuError> {
        Ok((
            self.route.ids.to_host_vec(stream)?,
            self.route.weights.to_host_vec(stream)?,
        ))
    }

    /// The last unit's picks read back: each of its `m` rows' kept expert ids in rank order and
    /// their weights, row-major (a gate's instrument; it synchronizes the stream).
    pub fn picks_rows(
        &self,
        (stream, m): (&CudaStream, usize),
    ) -> Result<(Vec<u32>, Vec<f32>), GpuError> {
        if m > self.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("{m} rows of a unit on buffers of {}", self.rows),
            ));
        }
        let n = m * split::N_USED;
        Ok((
            self.ids.to_host_vec(stream)?[..n].to_vec(),
            self.weights.to_host_vec(stream)?[..n].to_vec(),
        ))
    }

    /// The latent rows the last front or prompt wrote: the experts' input, up to the buffers'
    /// rows of the latent's width, token-major.
    #[must_use]
    pub fn lat(&self) -> &DeviceBuffer<f32> {
        &self.lat
    }

    /// The last step's router output read back: router 0's logits, router 1's, the summed scores
    /// (each [`split::N_EXPERT`]; a gate's instrument, it synchronizes the stream).
    pub fn scores(&self, stream: &CudaStream) -> Result<RouterRead, GpuError> {
        Ok(RouterRead {
            logits_0: self.route.logits_0.to_host_vec(stream)?,
            logits_1: self.route.logits_1.to_host_vec(stream)?,
            probs: self.route.probs.to_host_vec(stream)?,
        })
    }

    /// A stage's rows as the last launch left them (a gate's instrument): `m` rows of the
    /// unit's, row-major. [`Tap::Act`] and [`Tap::Down`] are the card leg's, a slot's rows a
    /// row (`m · 10` of them). It synchronizes the stream.
    pub fn tap(&self, (stream, m): (&CudaStream, usize), tap: Tap) -> Result<Vec<f32>, GpuError> {
        if m == 0 || m > self.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("{m} rows of a unit on buffers of {}", self.rows),
            ));
        }
        let (src, width): (&DeviceBuffer<f32>, usize) = match tap {
            Tap::Lat => (&self.lat, self.leg.shape().n),
            Tap::Routed => (&self.routed, self.leg.shape().n),
            Tap::Summed => (&self.summed, self.leg.shape().n),
            Tap::Normed => (&self.normed, self.leg.shape().n),
            Tap::LatUp => (&self.lat_up, self.shared.len() / self.rows),
            Tap::Shared => (&self.shared, self.shared.len() / self.rows),
            Tap::Act => (self.leg.act_rows(), self.leg.shape().ff),
            Tap::Down => (self.leg.down(), self.leg.shape().n),
            Tap::Probs => (&self.probs, split::N_EXPERT),
        };
        let rows = match tap {
            Tap::Act | Tap::Down => m * split::N_USED,
            _ => m,
        };
        let n = rows * width;
        if src.len() < n {
            return Err(GpuError::State {
                what: WHAT,
                missing: "the stage's rows: the buffers hold fewer",
            });
        }
        Ok(src.to_host_vec(stream)?[..n].to_vec())
    }

    /// Device bytes of the buffers.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.route.bytes()
            + [
                &self.probs,
                &self.logits,
                &self.weights,
                &self.lat,
                &self.routed,
                &self.summed,
                &self.normed,
                &self.lat_up,
                &self.gate,
                &self.up,
                &self.act,
                &self.shared,
            ]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.ids.num_bytes()
            + self.leg.bytes()
            + self.wide.iter().map(|(_, a)| a.bytes()).sum::<usize>()
            + self.dense.bytes()
    }
}

/// Layer `l`'s routed block; a layer with no routed block is refused by name.
fn layer(layers: &[Option<Routed>], l: usize) -> Result<&Routed, GpuError> {
    layers.get(l).and_then(Option::as_ref).ok_or_else(|| {
        GpuError::shape(
            WHAT,
            format!(
                "layer {l}: no routed block (a program of {} layers)",
                layers.len()
            ),
        )
    })
}

/// `m` rows of the hidden's width: a count the buffers do not hold or a shorter row is refused by
/// name.
fn check_rows(
    (rows, hidden): (usize, usize),
    m: usize,
    x: &DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    if m == 0 || m > rows || x.len() < m * hidden {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{m} rows of {hidden} values on buffers of {rows}: x holds {}",
                x.len()
            ),
        ));
    }
    Ok(())
}

/// The step's router: layer `l`'s two matrices, the bias and the learned scale over the one row
/// `x` into `out`.
fn route_1(
    router: &split::RouterKernels,
    (gpu, w): (&Gpu, &Weights),
    r: &Routed,
    l: usize,
    x: &DeviceBuffer<f32>,
    out: &mut split::RouterOut,
) -> Result<(), GpuError> {
    let (w0, w1) = (
        w.f32_tensor(WHAT, &r.router[0])?,
        w.f32_tensor(WHAT, &r.router[1])?,
    );
    router.enqueue_router(
        gpu.stream(),
        (w0, w1),
        x,
        w.f32_buf(WHAT, &r.bias)?,
        w.f32_buf(WHAT, &r.escale)?,
        out,
        gpu.layer_sink(l)?,
    )
}

/// A unit's router over `m` rows of `x`: the scores into `probs`, router 1's logits into
/// `logits`, the picks' ids and weights.
#[allow(
    clippy::too_many_arguments,
    reason = "the router, the card and weights, the layer and rows, the input and the four row buffers"
)]
fn route_rows(
    router: &split::RouterKernels,
    (gpu, w, g32): (&Gpu, &Weights, &Gemm32Kernels),
    r: &Routed,
    (l, m): (usize, usize),
    x: &DeviceBuffer<f32>,
    (probs, logits, ids, weights): (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<u32>,
        &mut DeviceBuffer<f32>,
    ),
) -> Result<(), GpuError> {
    let (w0, w1) = (
        w.f32_tensor(WHAT, &r.router[0])?,
        w.f32_tensor(WHAT, &r.router[1])?,
    );
    router.enqueue_router_rows(
        (gpu.stream(), g32),
        (w0, w1),
        x,
        w.f32_buf(WHAT, &r.bias)?,
        w.f32_buf(WHAT, &r.escale)?,
        m,
        RoutedRows {
            probs,
            logits: Some(logits),
            ids,
            weights,
        },
        gpu.layer_sink(l)?,
    )
}

/// `y = down(situ(gate(x), up(x)))` over one row: the shared expert's block.
fn glu_1(
    cx: &GqaCx<'_>,
    acts: &mut Acts,
    (rule, l): (Act, usize),
    g: &Glu,
    x: &DeviceBuffer<f32>,
    (gate, up, act): (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    g.gate.project(cx, acts, l, x, gate)?;
    g.up.project(cx, acts, l, x, up)?;
    cx.gpu
        .elem()
        .enqueue_situ(cx.gpu.stream(), rule, (gate, up), g.ff, act)?;
    g.down.project(cx, acts, l, act, y)
}

/// `y = W · x` of each of `sites` over `m` rows of `x`, all of one input width: the rows
/// quantized once into the forms the sites of that width read ([`project_wide`]).
#[allow(
    clippy::too_many_arguments,
    reason = "the wide arm's forms and table, the card, weights and kernels, the layer and rows, the input and the sites"
)]
fn project(
    (wide, dense): (&mut [(usize, WideActs)], &GemmRoute),
    (gpu, w, gemm, g32): (&Gpu, &Weights, &GemmKernels, &Gemm32Kernels),
    (l, m): (usize, usize),
    x: &DeviceBuffer<f32>,
    sites: &mut [(&Site, usize, &mut DeviceBuffer<f32>)],
) -> Result<(), GpuError> {
    let k = sites
        .first()
        .map(|s| s.0.k())
        .ok_or_else(|| GpuError::shape(WHAT, "a projection of no site"))?;
    let acts = wide
        .iter_mut()
        .find(|a| a.0 == k)
        .map(|a| &mut a.1)
        .ok_or_else(|| GpuError::shape(WHAT, format!("no wide forms of width {k}")))?;
    let cx = WideCx {
        what: WHAT,
        gpu,
        w,
        wk: WideKernels {
            gemm,
            g32: Some(g32),
        },
        m,
    };
    project_wide(&cx, (acts, dense), l, x, sites)
}

/// The experts the card's stacks of layer `l` hold, for a whole-card launch: refused by name when
/// they hold fewer than the file's (a placed load's share, whose steps and units run the front,
/// the shadow and the back with the host's sum).
fn whole_card((_, w): (&Gpu, &Weights), r: &Routed, dims: &MoeDims) -> Result<usize, GpuError> {
    let n_card = w.resident(WHAT, &r.stacks[0])?.rows() / dims.expert_ff;
    if n_card != dims.n_expert {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the card holds {n_card} of {} experts: the split form (the front, the shadow and \
                 the back) serves a layer with a host share",
                dims.n_expert
            ),
        ));
    }
    Ok(n_card)
}

/// The card's slots of layer `l` over `m` latent rows: the leg's four launches and the slots'
/// weighted sum into `routed` (`m` rows). The places `sel` name a card slot below `n_card` or
/// the host mark, which the leg leaves alone.
#[allow(
    clippy::too_many_arguments,
    reason = "the leg and its buffers, the card and weights, the layer and rows, the latent rows, the places, weights and count, the rule, the sum"
)]
fn card_slots(
    (leg, bufs): (&CardLeg, &mut LegBufs),
    (gpu, w): (&Gpu, &Weights),
    (r, l, m): (&Routed, usize, usize),
    lat: &DeviceBuffer<f32>,
    (sel, weights, n_card): (&DeviceBuffer<u32>, &DeviceBuffer<f32>, usize),
    rule: Act,
    routed: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stacks = LegStacks::of(w, r.stack_names())?;
    let run = LegRun {
        layer: l,
        stacks: &stacks,
        normed: lat,
        m,
        sel,
        n_card,
        rule,
    };
    leg.enqueue(
        gpu,
        bufs,
        &run,
        Some(LegSum {
            w: weights,
            acc: routed,
        }),
    )
}

/// The layer's routed sum of `m` rows: the card's (`routed`, where `card`) plus the host's
/// (`hsum`, where the layer has a host share), the add into `summed`; a block with neither is
/// refused by name. The add's order is `card + host`: up to rounding the whole-card block's,
/// which sums every slot in one launch. This order is the gate.
fn routed_sum<'a>(
    gpu: &Gpu,
    l: usize,
    (card, hsum): (bool, Option<&'a DeviceBuffer<f32>>),
    (routed, summed): (&'a DeviceBuffer<f32>, &'a mut DeviceBuffer<f32>),
    (m, latent): (usize, usize),
) -> Result<&'a DeviceBuffer<f32>, GpuError> {
    match (card, hsum) {
        (true, Some(h)) => {
            gpu.elem()
                .enqueue_add(gpu.stream(), routed, h, m * latent, summed)?;
            Ok(summed)
        }
        (true, None) => Ok(routed),
        (false, Some(h)) => Ok(h),
        (false, None) => Err(GpuError::shape(
            WHAT,
            format!("layer {l}: a routed block with neither a card slot nor a host sum"),
        )),
    }
}

/// The block's last three stages over the one row `summed`: the routed sum through the layer's
/// norm, the latent up, the shared expert's row added, into `out`.
#[allow(
    clippy::too_many_arguments,
    reason = "the context, the forms, the layer, the sum, three buffers, the dims and the output"
)]
fn tail_1(
    (gpu, w, cx): (&Gpu, &Weights, &GqaCx<'_>),
    acts: &mut Acts,
    (r, l): (&Routed, usize),
    summed: &DeviceBuffer<f32>,
    (normed, lat_up, shared): (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
    ),
    dims: &MoeDims,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    gpu.elem().enqueue_rms_norm(
        stream,
        summed,
        w.f32_buf(WHAT, &r.sum_norm)?,
        dims.eps,
        dims.latent,
        1,
        normed,
    )?;
    r.up_lat.project(cx, acts, l, normed, lat_up)?;
    gpu.elem()
        .enqueue_add(stream, lat_up, shared, dims.hidden, out)
}

/// The shared expert over `m` rows of `x` into `shared` (its gate and up sums over the same
/// quantization of `x` as the latent down's, [`project`]); `hidden` is the down's rows.
#[allow(
    clippy::too_many_arguments,
    reason = "the wide arm's forms and kernels, the context, the layer and rows, the input, four buffers, the rule and the width"
)]
fn shared_rows(
    (wide, dense, gemm): (&mut [(usize, WideActs)], &GemmRoute, &GemmKernels),
    cx: &GqaCx<'_>,
    (r, l, m): (&Routed, usize, usize),
    x: &DeviceBuffer<f32>,
    (gate, up, act, shared): (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    rule: Act,
    hidden: usize,
) -> Result<(), GpuError> {
    let (gpu, w) = (cx.gpu, cx.w);
    let ff = r.shared.ff;
    project(
        (wide, dense),
        (gpu, w, gemm, cx.g32),
        (l, m),
        x,
        &mut [(&r.shared.gate, ff, gate), (&r.shared.up, ff, up)],
    )?;
    gpu.elem()
        .enqueue_situ(gpu.stream(), rule, (gate, up), m * ff, act)?;
    project(
        (wide, dense),
        (gpu, w, gemm, cx.g32),
        (l, m),
        act,
        &mut [(&r.shared.down, hidden, shared)],
    )
}

/// The block's last three stages over `m` rows: [`tail_1`] with the latent up on the wide arm.
#[allow(
    clippy::too_many_arguments,
    reason = "the context, the wide arm's forms, table and kernels, the layer and rows, the sum, three buffers, the dims and the output"
)]
fn tail_rows(
    (gpu, w, cx, wide, dense, gemm): (
        &Gpu,
        &Weights,
        &GqaCx<'_>,
        &mut [(usize, WideActs)],
        &GemmRoute,
        &GemmKernels,
    ),
    (r, l, m): (&Routed, usize, usize),
    summed: &DeviceBuffer<f32>,
    (normed, lat_up, shared): (
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &DeviceBuffer<f32>,
    ),
    dims: &MoeDims,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    gpu.elem().enqueue_rms_norm(
        stream,
        summed,
        w.f32_buf(WHAT, &r.sum_norm)?,
        dims.eps,
        dims.latent,
        m,
        normed,
    )?;
    project(
        (wide, dense),
        (gpu, w, gemm, cx.g32),
        (l, m),
        normed,
        &mut [(&r.up_lat, dims.hidden, lat_up)],
    )?;
    gpu.elem()
        .enqueue_add(stream, lat_up, shared, m * dims.hidden, out)
}

impl Moe {
    /// Every routed layer of `hp`'s file: the pieces of each, and the buffers for a step and for
    /// prompt units of up to `rows` rows. Refused by name: a width, a rule or a router count the
    /// block does not run ([`MoeDims::check`]), and what each layer's load refuses. Load-time only.
    pub fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        hp: &Hparams,
        rows: usize,
    ) -> Result<(Moe, MoeBufs), GpuError> {
        let dims = MoeDims::of(hp);
        let layers: Vec<usize> = (0..hp.n_layer).filter(|&l| !hp.dense(l)).collect();
        Self::load_layers(gpu, file, w, &dims, &layers, rows)
    }

    /// [`Moe::load`] of the routed layers `layers` of `file` at `dims` (a file's header through
    /// [`MoeDims::of`]), in the model's layer numbers: a gate's load of one layer or a few. The
    /// buffers serve a step and prompt units of up to `rows` rows. Load-time only.
    pub fn load_layers(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        dims: &MoeDims,
        layers: &[usize],
        rows: usize,
    ) -> Result<(Moe, MoeBufs), GpuError> {
        dims.check()?;
        if rows == 0 {
            return Err(GpuError::shape(WHAT, "buffers of no row"));
        }
        let last = layers.iter().copied().max().unwrap_or(0);
        let mut by: Vec<Option<Routed>> = (0..=last).map(|_| None).collect();
        for &l in layers {
            if by[l].is_some() {
                return Err(GpuError::shape(WHAT, format!("layer {l} listed twice")));
            }
            by[l] = Some(Routed::load((file, w), l, dims)?);
        }
        let sites: Vec<&Site> = by.iter().flatten().flat_map(Routed::sites).collect();
        // The down form each resident layer's stacks read; a load whose card holds no stack of
        // the family holds the MXFP4 down's, the family's stacks as the file stores them.
        let mut downs = DownForms::NONE.with(DownAct::Q8Act);
        for r in by.iter().flatten() {
            if let Some(stacks) = LegStacks::of(w, r.stack_names()).ok()
                && let Some(form) = stacks.down_act()
            {
                downs = downs.with(form);
            }
        }
        let bufs = MoeBufs::new(gpu.stream(), (dims, rows), downs, &sites)?;
        Ok((
            Moe {
                dims: *dims,
                layers: by,
                router: split::RouterKernels::load(gpu.context())?,
                leg: CardLeg::load(gpu)?,
            },
            bufs,
        ))
    }

    /// The block's widths and rules.
    #[must_use]
    pub fn dims(&self) -> &MoeDims {
        &self.dims
    }

    /// Layer `l`'s block over the one normed row `x` into `out`, every expert on the card: the
    /// front, the card's slots at the picks' ids, and the back with no host sum.
    pub fn enqueue_step(
        &self,
        (gpu, w, cx): (&Gpu, &Weights, &GqaCx<'_>),
        bufs: &mut MoeBufs,
        l: usize,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let r = layer(&self.layers, l)?;
        check_rows((bufs.rows, self.dims.hidden), 1, x)?;
        bufs.unit_route = false;
        let dims = &self.dims;
        let MoeBufs {
            route,
            lat,
            routed,
            summed,
            normed,
            lat_up,
            gate,
            up,
            act,
            shared,
            leg,
            acts,
            ..
        } = bufs;
        route_1(&self.router, (gpu, w), r, l, x, route)?;
        r.down_lat.project(cx, acts, l, x, lat)?;
        let rule = dims.rule();
        glu_1(cx, acts, (rule, l), &r.shared, x, (gate, up, act), shared)?;
        card_slots(
            (&self.leg, leg),
            (gpu, w),
            (r, l, 1),
            lat,
            (&route.ids, &route.weights, whole_card((gpu, w), r, dims)?),
            rule,
            routed,
        )?;
        let summed_row = routed_sum(gpu, l, (true, None), (routed, summed), (1, dims.latent))?;
        tail_1(
            (gpu, w, cx),
            acts,
            (r, l),
            summed_row,
            (normed, lat_up, shared),
            dims,
            out,
        )
    }

    /// The front of layer `l`'s step, a layer with a host share: the router into the buffers'
    /// step output and the latent down of `x` into `lat`, the row the host leg reads (a placed
    /// load's boundary hands it over). The caller runs the handoff and the go between this and
    /// [`Moe::enqueue_shadow`].
    pub fn enqueue_front(
        &self,
        (gpu, w, cx): (&Gpu, &Weights, &GqaCx<'_>),
        bufs: &mut MoeBufs,
        l: usize,
        x: &DeviceBuffer<f32>,
        lat: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let r = layer(&self.layers, l)?;
        check_rows((bufs.rows, self.dims.hidden), 1, x)?;
        bufs.unit_route = false;
        if lat.len() < self.dims.latent {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: a latent row of {} values into a buffer of {}",
                    self.dims.latent,
                    lat.len()
                ),
            ));
        }
        let MoeBufs { route, acts, .. } = bufs;
        route_1(&self.router, (gpu, w), r, l, x, route)?;
        r.down_lat.project(cx, acts, l, x, lat)
    }

    /// The shadow of layer `l`'s step, under the host leg's work: the shared expert of `x` into
    /// the buffers' `shared` row, and the card's slots over the latent row `lat` at the places
    /// `sel` (a card slot below `n_card`, or the host mark), the picks' weights the sum's. Whether
    /// the card's slots ran: false on a layer the card holds no expert of, whose back takes the
    /// host's sum alone.
    pub fn enqueue_shadow(
        &self,
        (gpu, w, cx): (&Gpu, &Weights, &GqaCx<'_>),
        bufs: &mut MoeBufs,
        l: usize,
        x: &DeviceBuffer<f32>,
        lat: &DeviceBuffer<f32>,
        (sel, n_card): (&DeviceBuffer<u32>, usize),
    ) -> Result<bool, GpuError> {
        let r = layer(&self.layers, l)?;
        check_rows((bufs.rows, self.dims.hidden), 1, x)?;
        let MoeBufs {
            route,
            gate,
            up,
            act,
            shared,
            leg,
            routed,
            acts,
            ..
        } = bufs;
        let rule = self.dims.rule();
        glu_1(cx, acts, (rule, l), &r.shared, x, (gate, up, act), shared)?;
        if n_card == 0 {
            return Ok(false);
        }
        card_slots(
            (&self.leg, leg),
            (gpu, w),
            (r, l, 1),
            lat,
            (sel, &route.weights, n_card),
            rule,
            routed,
        )?;
        Ok(true)
    }

    /// The back of layer `l`'s step from the wait on: the routed sum (the card's where `card`,
    /// plus `hsum`, the host leg's, where the layer has a host share) through the norm, the latent
    /// up and the shared expert, into `out`.
    pub fn enqueue_back(
        &self,
        (gpu, w, cx): (&Gpu, &Weights, &GqaCx<'_>),
        bufs: &mut MoeBufs,
        l: usize,
        (card, hsum): (bool, Option<&DeviceBuffer<f32>>),
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let r = layer(&self.layers, l)?;
        let dims = &self.dims;
        if let Some(h) = hsum
            && h.len() < dims.latent
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: the host's sum holds {} values, {} needed",
                    h.len(),
                    dims.latent
                ),
            ));
        }
        let MoeBufs {
            routed,
            summed,
            normed,
            lat_up,
            shared,
            acts,
            ..
        } = bufs;
        let summed_row = routed_sum(gpu, l, (card, hsum), (routed, summed), (1, dims.latent))?;
        tail_1(
            (gpu, w, cx),
            acts,
            (r, l),
            summed_row,
            (normed, lat_up, shared),
            dims,
            out,
        )
    }

    /// Layer `l`'s block over the `m` rows of `x`, every expert on the card: [`Moe::enqueue_step`]
    /// a row, the card's slots one run over the unit's rows. `m` is 1 to the rows the buffers
    /// were made for.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, weights, the two contexts, the buffers, the layer, the rows, the input and the output"
    )]
    pub fn enqueue_prompt(
        &self,
        (gpu, w, cx, gemm): (&Gpu, &Weights, &GqaCx<'_>, &GemmKernels),
        bufs: &mut MoeBufs,
        l: usize,
        m: usize,
        x: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let r = layer(&self.layers, l)?;
        check_rows((bufs.rows, self.dims.hidden), m, x)?;
        bufs.unit_route = true;
        let dims = &self.dims;
        let MoeBufs {
            probs,
            logits,
            ids,
            weights,
            lat,
            routed,
            summed,
            normed,
            lat_up,
            gate,
            up,
            act,
            shared,
            leg,
            wide,
            dense,
            ..
        } = bufs;
        route_rows(
            &self.router,
            (gpu, w, cx.g32),
            r,
            (l, m),
            x,
            (probs, logits, ids, weights),
        )?;
        gemm.enqueue_route_dense(gpu.stream(), m, dense, gpu.layer_sink(l)?)?;
        project(
            (wide, &*dense),
            (gpu, w, gemm, cx.g32),
            (l, m),
            x,
            &mut [(&r.down_lat, dims.latent, lat)],
        )?;
        let rule = dims.rule();
        shared_rows(
            (wide, &*dense, gemm),
            cx,
            (r, l, m),
            x,
            (gate, up, act, shared),
            rule,
            dims.hidden,
        )?;
        card_slots(
            (&self.leg, leg),
            (gpu, w),
            (r, l, m),
            lat,
            (&*ids, &*weights, whole_card((gpu, w), r, dims)?),
            rule,
            routed,
        )?;
        let summed_rows = routed_sum(gpu, l, (true, None), (routed, summed), (m, dims.latent))?;
        tail_rows(
            (gpu, w, cx, wide, &*dense, gemm),
            (r, l, m),
            summed_rows,
            (normed, lat_up, shared),
            dims,
            out,
        )
    }

    /// The front of layer `l` over the `m` rows of `x`, a layer with a host share: the router
    /// over the rows into the buffers' `probs`, `logits`, `ids` and `weights`, and the latent
    /// down into the buffers' `lat` ([`MoeBufs::lat`]: `m` rows, the rows a prompt batch's
    /// download hands the host and the card's slots read). The caller runs the places and the
    /// download between this and [`Moe::enqueue_shadow_rows`].
    pub fn enqueue_front_rows(
        &self,
        (gpu, w, cx, gemm): (&Gpu, &Weights, &GqaCx<'_>, &GemmKernels),
        bufs: &mut MoeBufs,
        l: usize,
        m: usize,
        x: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let r = layer(&self.layers, l)?;
        check_rows((bufs.rows, self.dims.hidden), m, x)?;
        bufs.unit_route = true;
        let dims = &self.dims;
        let MoeBufs {
            probs,
            logits,
            ids,
            weights,
            lat,
            wide,
            dense,
            ..
        } = bufs;
        route_rows(
            &self.router,
            (gpu, w, cx.g32),
            r,
            (l, m),
            x,
            (probs, logits, ids, weights),
        )?;
        gemm.enqueue_route_dense(gpu.stream(), m, dense, gpu.layer_sink(l)?)?;
        project(
            (wide, &*dense),
            (gpu, w, gemm, cx.g32),
            (l, m),
            x,
            &mut [(&r.down_lat, dims.latent, lat)],
        )
    }

    /// [`Moe::enqueue_shadow`] over the `m` rows: the shared expert into the buffers' `shared`
    /// rows, the card's slots one run over the buffers' latent rows at the places `sel`
    /// (`m · 10` places), the buffers' `weights` the sum's. Whether the card's slots ran.
    pub fn enqueue_shadow_rows(
        &self,
        (gpu, w, cx, gemm): (&Gpu, &Weights, &GqaCx<'_>, &GemmKernels),
        bufs: &mut MoeBufs,
        l: usize,
        m: usize,
        x: &DeviceBuffer<f32>,
        (sel, n_card): (&DeviceBuffer<u32>, usize),
    ) -> Result<bool, GpuError> {
        let r = layer(&self.layers, l)?;
        check_rows((bufs.rows, self.dims.hidden), m, x)?;
        let dims = &self.dims;
        let MoeBufs {
            weights,
            gate,
            up,
            act,
            shared,
            leg,
            lat,
            routed,
            wide,
            dense,
            ..
        } = bufs;
        let rule = dims.rule();
        shared_rows(
            (wide, &*dense, gemm),
            cx,
            (r, l, m),
            x,
            (gate, up, act, shared),
            rule,
            dims.hidden,
        )?;
        if n_card == 0 {
            return Ok(false);
        }
        card_slots(
            (&self.leg, leg),
            (gpu, w),
            (r, l, m),
            lat,
            (sel, &*weights, n_card),
            rule,
            routed,
        )?;
        Ok(true)
    }

    /// [`Moe::enqueue_back`] over the `m` rows: the routed sums plus the host's `hsum` rows
    /// (`m` of the latent's width) through the norm, the latent up and the shared expert, into
    /// `out`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the card, weights, the two contexts, the buffers, the layer, the rows, the join and the output"
    )]
    pub fn enqueue_back_rows(
        &self,
        (gpu, w, cx, gemm): (&Gpu, &Weights, &GqaCx<'_>, &GemmKernels),
        bufs: &mut MoeBufs,
        l: usize,
        m: usize,
        (card, hsum): (bool, Option<&DeviceBuffer<f32>>),
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let r = layer(&self.layers, l)?;
        let dims = &self.dims;
        if m == 0 || m > bufs.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("{m} rows of a unit on buffers of {}", bufs.rows),
            ));
        }
        if let Some(h) = hsum
            && h.len() < m * dims.latent
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {l}: the host's sum of {m} rows holds {} values, {} needed",
                    h.len(),
                    m * dims.latent
                ),
            ));
        }
        let MoeBufs {
            routed,
            summed,
            normed,
            lat_up,
            shared,
            wide,
            dense,
            ..
        } = bufs;
        let summed_rows = routed_sum(gpu, l, (card, hsum), (routed, summed), (m, dims.latent))?;
        tail_rows(
            (gpu, w, cx, wide, &*dense, gemm),
            (r, l, m),
            summed_rows,
            (normed, lat_up, shared),
            dims,
            out,
        )
    }
}
