//! `GpuModel<B>` — the GPU engine; an architecture's CPU `forward::step` is its reference
//! (docs/gpu-design.md decisions 1 and 7). Weights and everything a step
//! touches live on the device from `load`, split into stages by layer range;
//! `step` enqueues one decode step stage by stage and synchronizes once for
//! the argmax.
//!
//! This file is the part that is the same for every architecture
//! (docs/arch-split.md): the graph drop order, the capture identity
//! `(layer, embed)`, the rule that a mode change discards a capture, the `pos`
//! advance and the single argmax readback. Which weights a model derives at
//! load, what a layer chain enqueues, which rows it appends to which cache and
//! what one step's host input is belong to a [`ChainBody`] under
//! [`crate::arch`]; `GpuModel` is generic over it and
//! monomorphic at every call site, so nothing on the token path is a virtual
//! call. [`GpuModel::step`] stitches the body's layers into the whole chain —
//! layer 0 embedding its token, every later layer reading the previous
//! layer's output residual, the `head.rs` output head on the last — and
//! returns the argmax of the last token it was given. The chain submits in one
//! of two modes ([`StepMode`]): eager, which enqueues the body per token, or
//! graph, which captures the body once and replays it. Only the argmax
//! readback synchronizes.

pub(crate) mod kernels;
pub(crate) mod launcher;
pub(crate) mod lookup;
pub(crate) mod probe;

pub use kernels::{Q8_0GemvHeadsArgs, StepKernels};
pub use launcher::LaunchStats;
pub(crate) use lookup::f32_gain;
pub use probe::{OpTime, StepProbe};

use crate::fault::Fault;
use crate::head::Head;
use crate::hybrid::{Chain, HostResidency, HybridConfig, Refusal, host_levers, name_refusal};
use crate::weights::Weights;
use crate::{Gpu, GpuError, Graph, NodeInfo};
use cuda_core::CudaStream;
use gguf::Split;
use model::arch::Arch;
use model::placement::Plan;
use std::ops::Range;

// -------------------------------------------------------------- chain body

/// One architecture's layer chain: what the shared skeleton drives at capture
/// time, and everything the step touches that is shaped by the model rather
/// than by the runtime — the derived weights, the KV row store, the load-time
/// arena, the weight names, the step's own kernels (docs/arch-split.md,
/// 「트레이트 둘」).
///
/// Monomorphic by construction: `GpuModel<B>` names one body, and a binary
/// that can open two architectures branches once, on the enum around the
/// model, never inside a layer.
pub trait ChainBody: Sized {
    /// The per-replay host values of one step. deepseek2 is the token and the
    /// position; deepseek41 adds the engram rows and the compressed-index
    /// plan, which is why this is a type and not two arguments.
    type Input;

    /// What a placed load hands the body from the file's headers, read once
    /// by whoever made the placement plan — deepseek41's hyperparameters, so
    /// the plan and the body are sized from one reading. `()` for an
    /// architecture without a placed load.
    type Meta;

    /// The architecture this body is the chain of.
    fn arch() -> Arch;

    /// File into `w` the weights this architecture's plan computes at load
    /// for the blocks of `layers` — after the file tensors are resident,
    /// before [`ChainBody::load`] reads them. Required, with no default body,
    /// so an architecture cannot forget its derived weights silently.
    fn derive(
        stream: &CudaStream,
        file: &Split,
        layers: Range<usize>,
        w: &mut Weights,
    ) -> Result<(), GpuError>;

    /// Everything the chain of `layers` needs resident, over the weights the
    /// caller has already loaded and derived: the caches, the arena and the
    /// step module.
    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        layers: Range<usize>,
        ctx_max: usize,
    ) -> Result<Self, GpuError>;

    /// The input of one decode step at `(token, pos)`. A body whose next step
    /// needs work issued ahead of it does that here, not in `refresh`.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<Self::Input, GpuError>;

    /// Write `input` into the device buffers the captured graph reads, so one
    /// capture serves every position. Runs before an eager enqueue or a replay
    /// and never inside a capture.
    fn refresh(&mut self, stream: &CudaStream, input: &Self::Input) -> Result<(), GpuError>;

    /// Enqueue the whole chain — every resident layer and then `head`.
    /// Asynchronous throughout: no allocation, no synchronization, no host
    /// round trip, so this is both the eager body and what the capture
    /// records.
    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError>;

    /// Rewind every cache to empty. The weights, the arena and any captured
    /// chain stay: none of them depends on the cache contents.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError>;

    /// Fill the first `rows` cache rows of every resident layer with the
    /// body's own synthetic pattern — the state a prompt of `rows` tokens
    /// leaves behind, without decoding one. An instrument: the rows are not
    /// what the model would have written.
    fn seed_depth(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError>;

    /// Arm (or disarm) the node-price probe on this body's arena. The
    /// skeleton owns dropping the captures the change invalidates.
    fn set_probe(&mut self, probe: StepProbe) -> Result<(), GpuError>;

    /// The rms epsilon the output head normalizes with, read from the file at
    /// load by this body's plan.
    fn head_eps(&self) -> f32;

    /// Device bytes this body holds resident: caches, arena, step module.
    fn resident_bytes(&self) -> usize;

    /// The file tensors of a hybrid load of `layers` (crate::hybrid): every
    /// tensor a whole-model load uploads, except that each routed expert
    /// stack keeps only its experts `[0, n_l)` — the leading rows the `_sel`
    /// kernels address; that prefix is the V2-Lite (deepseek2) convention. The default refuses: an architecture without a
    /// hybrid plan has no hybrid load.
    fn hybrid_weights(
        _stream: &CudaStream,
        _file: &gguf::Split,
        _layers: Range<usize>,
        _n_l: usize,
    ) -> Result<Weights, GpuError> {
        Err(GpuError::state(
            "ChainBody::hybrid_weights",
            "this architecture has no hybrid load",
        ))
    }

    /// [`ChainBody::derive`] and [`ChainBody::load`] of a hybrid load in one:
    /// the derived weights filed into `w`, then the body over them, whose
    /// routed layers hand experts `[cfg.n_l, n_expert)` to a host tier that
    /// keeps `file` (the V2-Lite prefix cut). The default refuses.
    fn load_hybrid(
        _gpu: &Gpu,
        _file: gguf::Split,
        _w: &mut Weights,
        _layers: Range<usize>,
        _ctx_max: usize,
        _cfg: HybridConfig,
    ) -> Result<Self, GpuError> {
        Err(GpuError::state(
            "ChainBody::load_hybrid",
            "this architecture has no hybrid load",
        ))
    }

    /// The body of card `card` of `plan`, over the weights the caller has
    /// already loaded ([`Weights::load_placed`]) and derived for the card's
    /// layers: every routed stack holds the experts the plan's segments put
    /// on the card, and the body keeps `file` for what the plan leaves on the
    /// host. Its caches hold the plan's `ctx_max` rows. The default refuses:
    /// an architecture without a placement has no placed load.
    fn load_placed(
        _gpu: &Gpu,
        _file: Split,
        _w: &Weights,
        _plan: &Plan<'_>,
        _card: usize,
        _meta: &Self::Meta,
    ) -> Result<Self, GpuError> {
        Err(GpuError::state(
            "ChainBody::load_placed",
            "this architecture has no placed load",
        ))
    }

    /// Serve the host's share of the chain a graph replay just submitted, in
    /// chain order, before anything else waits on the stream. A body whose
    /// chain holds no host work has nothing to do, which is the default.
    fn serve_replay(&mut self) -> Result<(), GpuError> {
        Ok(())
    }

    /// The pair pass's host half: the steps of `tokens[0]` at `pos` and
    /// `tokens[1]` at `pos + 1`, planned in turn and written into the
    /// buffers the captured pair reads. Never inside a capture. The default
    /// refuses: a body without a second row has no pair pass.
    fn decode_pair(
        &mut self,
        _stream: &CudaStream,
        _tokens: [u32; 2],
        _pos: u32,
    ) -> Result<(), GpuError> {
        Err(GpuError::state(
            "ChainBody::decode_pair",
            "this architecture has no pair pass",
        ))
    }

    /// Enqueue the pair pass — the two tokens [`ChainBody::decode_pair`]
    /// planned, through every resident layer and into `heads[0]` and
    /// `heads[1]` — bit for bit two steps in turn. Asynchronous as
    /// [`ChainBody::enqueue_chain`] is. The default refuses.
    fn enqueue_pair(
        &mut self,
        _gpu: &Gpu,
        _w: &Weights,
        _heads: [&mut Head; 2],
    ) -> Result<(), GpuError> {
        Err(GpuError::state(
            "ChainBody::enqueue_pair",
            "this architecture has no pair pass",
        ))
    }

    /// The refusal that failed the step's host service, once
    /// ([`crate::hybrid::Hybrid::take_step_refusal`]): what the step's
    /// failure is named from after the stream has drained. A body without a
    /// host tier has none, which is the default.
    fn take_host_refusal(&mut self) -> Option<Refusal> {
        None
    }

    /// [`ChainBody::serve_replay`] for a replay of the captured pair pass.
    fn serve_replay_pair(&mut self) -> Result<(), GpuError> {
        Err(GpuError::state(
            "ChainBody::serve_replay_pair",
            "this architecture has no pair pass",
        ))
    }

    /// Take back the positions from `pos` on, so the next step runs at
    /// `pos`. The default refuses.
    fn rollback(&mut self, _pos: u32) -> Result<(), GpuError> {
        Err(GpuError::state(
            "ChainBody::rollback",
            "this architecture takes no position back",
        ))
    }
}

/// What a binary or a gate drives, once per token — the surface that does not
/// name an architecture. Static dispatch only: it exists so `generate` and the
/// gate harnesses can be written once over `impl Engine`, not for virtual
/// calls.
pub trait Engine {
    fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError>;
    fn reset(&mut self) -> Result<(), GpuError>;
    fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError>;
    fn pos(&self) -> u32;
    fn resident_bytes(&self) -> usize;
    fn arch(&self) -> Arch;
}

// ------------------------------------------------------------------- stage

/// A contiguous range of blocks resident on one device (docs/gpu-design.md
/// decision 7). A stage owns its `Gpu` — context, stream, modules — and its
/// weights and the body's resident buffers once loaded with residency; what
/// crosses a stage boundary is one hidden vector. Two stages may sit on the
/// same card: that is the shape the 2-stage = 1-stage bit-identity gate runs
/// in.
///
/// `residency` is filled by [`GpuModel::load_blocks`]; a stage from
/// [`GpuModel::load_staged`] carries only the metadata (that entry must stay
/// cheap — metadata probes call it).
pub struct Stage<B> {
    gpu: Gpu,
    /// Blocks `layers.start..layers.end` of the model, in order.
    layers: std::ops::Range<usize>,
    /// The captured decode step of this stage, once one is assembled.
    graph: Option<Graph>,
    /// What `graph` recorded: the layer and whether it embeds in front. A
    /// replay names what it expects, so a graph of another layer is an
    /// error, not a silent replay of the wrong chain.
    graph_of: Option<(usize, bool)>,
    /// Weights and the architecture's body — present once loaded with
    /// residency.
    residency: Option<Residency<B>>,
}

/// Everything a stage's step touches, allocated at load (decision 4): the
/// weights every architecture reads the same way, and the body that knows
/// what to do with them.
struct Residency<B> {
    weights: Weights,
    body: B,
}

impl<B: ChainBody> Stage<B> {
    /// The device this stage's layers run on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The model layers this stage carries, as a range of layer indices.
    pub fn layers(&self) -> std::ops::Range<usize> {
        self.layers.clone()
    }

    /// Device bytes held by the residency: weights, caches, arena. Zero for a
    /// metadata-only stage.
    pub fn resident_bytes(&self) -> usize {
        self.residency
            .as_ref()
            .map_or(0, |r| r.weights.resident_bytes() + r.body.resident_bytes())
    }

    /// The stage's resident weights; `None` for a metadata-only stage.
    pub fn weights(&self) -> Option<&Weights> {
        self.residency.as_ref().map(|r| &r.weights)
    }
}

/// How [`GpuModel::step`] submits the chain. Both modes run the same body
/// over the same resident buffers; the graph mode records it once and
/// replays, so it pays the host submit of ~700 launches only at capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMode {
    /// Enqueue the whole body per token.
    Eager,
    /// Capture the body on first use, then replay it per token.
    Graph,
}

/// One resident model: its stages in layer order, covering every block
/// exactly once. Everything `step` touches is allocated at load, never per
/// step.
pub struct GpuModel<B: ChainBody> {
    /// The whole chain (every layer plus the head) captured once and
    /// replayed per token. Deliberately not `Stage::graph`, whose
    /// `(layer, embed)` identity the block gates own.
    ///
    /// Declared FIRST, and the head second, because fields drop in
    /// declaration order and a graph must be destroyed while every buffer
    /// it addresses is still alive — the same reason `Stage` declares its
    /// `graph` above `residency`.
    step_graph: Option<Graph>,
    /// The pair pass ([`GpuModel::step_pair`]) captured once and replayed
    /// per pass; declared before the heads for the same reason.
    pair_graph: Option<Graph>,
    /// The output head — present only on a model that holds every block
    /// ([`GpuModel::load_full`]). A partial stage has no logits to take, so
    /// `step` refuses on one.
    head: Option<Head>,
    /// The pair pass's second row's head, made at the first
    /// [`GpuModel::step_pair`]; the first row's is `head`.
    pair_head: Option<Head>,
    /// What a placed load did to the plan's host set, and the lock over it
    /// when one was asked for. Declared before `stages` so that it drops
    /// first: the lock's spans are pages of the file mappings a stage's body
    /// keeps.
    host: Option<HostResidency>,
    /// `BLOOMERY_LAUNCH_THREAD=1`'s launch thread for stage 0's stream, from
    /// open ([`launcher`]); `None` launches on the decode thread. Idle at
    /// drop: every launch it took was answered before its step returned.
    launcher: Option<launcher::Launcher>,
    /// What the replays' launches have cost since load.
    launch_stats: LaunchStats,
    stages: Vec<Stage<B>>,
    /// KV rows the resident caches were sized for; `step` refuses to grow them.
    ctx_max: usize,
    mode: StepMode,
    /// The cache row the next `step` token lands in.
    pos: u32,
    /// The fault a step read back (crate::fault): the caches and rings hold
    /// what it condemned, so every later step refuses until `reset`.
    poisoned: Option<Fault>,
}

impl<B: ChainBody> GpuModel<B> {
    /// The whole model as one stage (metadata only — a resident stage comes
    /// from [`GpuModel::load_blocks`]).
    pub fn load(file: &Split, ctx_max: usize) -> Result<GpuModel<B>, GpuError> {
        GpuModel::load_staged(file, ctx_max, &[])
    }

    /// Split the blocks at `cuts` (strictly ascending, each in
    /// `1..block_count`): `cuts.len() + 1` stages. Reads the metadata `step`
    /// needs; the stages carry no residency until [`GpuModel::load_blocks`]
    /// fills one.
    pub(crate) fn load_staged(
        file: &Split,
        ctx_max: usize,
        cuts: &[usize],
    ) -> Result<GpuModel<B>, GpuError> {
        if ctx_max == 0 {
            return Err(GpuError::shape("GpuModel::load", "ctx_max must be >= 1"));
        }
        let n_layers = block_count(file, "GpuModel::load")?;
        let mut bounds = vec![0usize];
        for &c in cuts {
            if c <= *bounds.last().unwrap_or(&0) || c >= n_layers {
                return Err(GpuError::shape(
                    "GpuModel::load_staged",
                    format!("cuts must ascend strictly inside 1..{n_layers}, got {cuts:?}"),
                ));
            }
            bounds.push(c);
        }
        bounds.push(n_layers);
        let mut stages = Vec::with_capacity(bounds.len() - 1);
        for w in bounds.windows(2) {
            stages.push(Stage {
                gpu: Gpu::new()?,
                layers: w[0]..w[1],
                graph: None,
                graph_of: None,
                residency: None,
            });
        }
        Ok(GpuModel {
            stages,
            ctx_max,
            head: None,
            step_graph: None,
            pair_graph: None,
            pair_head: None,
            host: None,
            launcher: None,
            launch_stats: LaunchStats::default(),
            mode: StepMode::Graph,
            pos: 0,
            poisoned: None,
        })
    }

    /// One stage over `layers` WITH residency: every tensor of that range
    /// plus the globals in its kernels' device format, the weights the body
    /// derives from them, and the body those weights drive — its caches, its
    /// m = 1 arena and its step module. The geometry checks run at load, not
    /// mid-step. The file is one shard, refused otherwise before anything is
    /// uploaded: this is the whole-tensor upload of [`Weights::load`], and a
    /// model of several shards loads by its placement plan
    /// ([`GpuModel::load_placed`]).
    pub fn load_blocks(
        file: &Split,
        ctx_max: usize,
        layers: Range<usize>,
    ) -> Result<GpuModel<B>, GpuError> {
        let what = "GpuModel::load_blocks";
        if ctx_max == 0 {
            return Err(GpuError::shape(what, "ctx_max must be >= 1"));
        }
        let n_layers = block_count(file, what)?;
        if layers.start >= layers.end || layers.end > n_layers {
            return Err(GpuError::shape(
                what,
                format!("layer range {layers:?} outside 0..{n_layers}"),
            ));
        }
        one_shard(file, what)?;
        let gpu = Gpu::new()?;
        let mut weights = Weights::load(gpu.stream(), file, layers.clone(), true)?;
        B::derive(gpu.stream(), file, layers.clone(), &mut weights)?;
        let body = B::load(&gpu, file, &weights, layers.clone(), ctx_max)?;
        let launcher = launcher::at_open(&gpu.stream)?;
        Ok(GpuModel {
            stages: vec![Stage {
                gpu,
                layers,
                graph: None,
                graph_of: None,
                residency: Some(Residency { weights, body }),
            }],
            ctx_max,
            head: None,
            step_graph: None,
            pair_graph: None,
            pair_head: None,
            host: None,
            launcher,
            launch_stats: LaunchStats::default(),
            mode: StepMode::Graph,
            pos: 0,
            poisoned: None,
        })
    }

    /// Every block of `file` resident, plus the output head: the model
    /// [`GpuModel::step`] needs. `load_blocks(0..block_count)` first (so the
    /// head's `output_norm.weight` / `output.weight` arrive with the
    /// globals), then the head over those same resident weights, normalizing
    /// with the epsilon the body's plan read.
    ///
    /// The model takes the file: `BLOOMERY_HYBRID_NL` below the file's expert
    /// count makes this the hybrid load of [`GpuModel::load_hybrid`], whose
    /// host tier keeps it for the experts the card does not hold; unset or
    /// equal to the expert count, every weight is on the card and the file is
    /// closed once they are.
    pub fn load_full(file: Split, ctx_max: usize) -> Result<GpuModel<B>, GpuError> {
        if let Some(cfg) = HybridConfig::from_levers(&file)? {
            return GpuModel::load_hybrid(file, ctx_max, cfg);
        }
        let n_layers = block_count(&file, "GpuModel::load_full")?;
        let mut m = GpuModel::<B>::load_blocks(&file, ctx_max, 0..n_layers)?;
        let head = {
            let (gpu, weights, body) = m.body_parts("GpuModel::load_full")?;
            let eps = body.head_eps();
            Head::new(gpu, weights, eps)?
        };
        m.head = Some(head);
        Ok(m)
    }

    /// Every block of `file` resident, plus the output head, with each MoE
    /// layer's experts `[0, cfg.n_l)` on the card — the V2-Lite (deepseek2)
    /// prefix cut — and the rest computed on the host inside the captured
    /// step (crate::hybrid). The model keeps
    /// `file` for its host experts.
    pub fn load_hybrid(
        file: Split,
        ctx_max: usize,
        cfg: HybridConfig,
    ) -> Result<GpuModel<B>, GpuError> {
        let what = "GpuModel::load_hybrid";
        if ctx_max == 0 {
            return Err(GpuError::shape(what, "ctx_max must be >= 1"));
        }
        let n_layers = block_count(&file, what)?;
        let gpu = Gpu::new()?;
        let mut weights = B::hybrid_weights(gpu.stream(), &file, 0..n_layers, cfg.n_l)?;
        let body = B::load_hybrid(&gpu, file, &mut weights, 0..n_layers, ctx_max, cfg)?;
        let head = Head::new(&gpu, &weights, body.head_eps())?;
        let launcher = launcher::at_open(&gpu.stream)?;
        Ok(GpuModel {
            stages: vec![Stage {
                gpu,
                layers: 0..n_layers,
                graph: None,
                graph_of: None,
                residency: Some(Residency { weights, body }),
            }],
            ctx_max,
            head: Some(head),
            step_graph: None,
            pair_graph: None,
            pair_head: None,
            host: None,
            launcher,
            launch_stats: LaunchStats::default(),
            mode: StepMode::Graph,
            pos: 0,
            poisoned: None,
        })
    }

    /// Card `card` of `plan` resident, as one stage: the segments the plan
    /// puts on the card ([`Weights::load_placed`] — whole tensors, and each
    /// routed stack's card `ExpertList` of its layer, the id prefix or a hot
    /// list's ranked ids), the weights the body
    /// derives for the card's layers, and the body over them
    /// ([`ChainBody::load_placed`]), which keeps `file` for what the plan
    /// leaves on the host; plus the output head when the card carries it.
    /// Between the uploads and the body the plan's host set is read in and,
    /// when asked, locked ([`HostResidency::at_load`], [`crate::hybrid::HostLevers`]),
    /// so that no step takes the first touch of a host expert page.
    /// The caches hold the plan's `ctx_max` rows, the context its budget was
    /// made for. The card is found by its name in the plan
    /// ([`Gpu::for_card`]), never by ordinal. `meta` is what the plan was made
    /// from.
    pub fn load_placed(
        file: Split,
        plan: &Plan<'_>,
        card: usize,
        meta: &B::Meta,
    ) -> Result<GpuModel<B>, GpuError> {
        let what = "GpuModel::load_placed";
        let spec = plan
            .machine
            .cards
            .get(card)
            .ok_or_else(|| GpuError::shape(what, format!("the plan has no card {card}")))?;
        let ctx_max = usize::try_from(plan.ctx_max)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| GpuError::shape(what, format!("the plan's ctx_max {}", plan.ctx_max)))?;
        let layers = spec.layers.clone();
        let gpu = Gpu::for_card(&spec.name)?;
        let mut weights = Weights::load_placed(gpu.stream(), &file, plan, card)?;
        B::derive(gpu.stream(), &file, layers.clone(), &mut weights)?;
        // After the uploads, so the card's file bytes have left the page
        // cache before the host set is read in; before the body, which
        // takes `file`.
        let host = HostResidency::at_load(&file, plan, |_| true, host_levers()?)?;
        let body = B::load_placed(&gpu, file, &weights, plan, card, meta)?;
        let head = if spec.head {
            Some(Head::new(&gpu, &weights, body.head_eps())?)
        } else {
            None
        };
        let launcher = launcher::at_open(&gpu.stream)?;
        Ok(GpuModel {
            stages: vec![Stage {
                gpu,
                layers,
                graph: None,
                graph_of: None,
                residency: Some(Residency { weights, body }),
            }],
            ctx_max,
            head,
            step_graph: None,
            pair_graph: None,
            pair_head: None,
            host: Some(host),
            launcher,
            launch_stats: LaunchStats::default(),
            mode: StepMode::Graph,
            pos: 0,
            poisoned: None,
        })
    }

    /// With the replays' launches on their own thread
    /// (`BLOOMERY_LAUNCH_THREAD=1`, read at open), `Some` of where that thread
    /// runs: the SMT sibling of the opening thread's cpu when that thread was
    /// pinned to one, `None` when it floats. `None` launches on the decode
    /// thread.
    #[must_use]
    pub fn launch_thread(&self) -> Option<Option<usize>> {
        self.launcher.as_ref().map(launcher::Launcher::cpu)
    }

    /// What the replays' launches have cost since load.
    #[must_use]
    pub fn launch_stats(&self) -> LaunchStats {
        self.launch_stats
    }

    /// What the load did to the plan's host set — populated, locked — on a
    /// placed load ([`GpuModel::load_placed`]); `None` on any other.
    pub fn host_residency(&self) -> Option<&HostResidency> {
        self.host.as_ref()
    }

    /// The model's stages, in layer order.
    pub fn stages(&self) -> &[Stage<B>] {
        &self.stages
    }

    /// Device bytes of everything this model holds resident: the stage's
    /// weights and body, plus the heads' scratch — the pair pass's second
    /// head once a pair has run.
    pub fn resident_bytes(&self) -> usize {
        self.stages.iter().map(Stage::resident_bytes).sum::<usize>()
            + self.head.as_ref().map_or(0, Head::resident_bytes)
            + self.pair_head.as_ref().map_or(0, Head::resident_bytes)
    }

    /// The cache row the next [`GpuModel::step`] token lands in.
    pub fn pos(&self) -> u32 {
        self.pos
    }

    /// How the chain submits ([`GpuModel::set_mode`]).
    #[must_use]
    pub fn mode(&self) -> StepMode {
        self.mode
    }

    /// Choose how the chain submits. Changing the mode drops any captured
    /// chain: the graph is a recording of this body over these buffers, and
    /// a later `Graph` run recaptures rather than replay a stale one.
    pub fn set_mode(&mut self, mode: StepMode) {
        if mode != self.mode {
            self.step_graph = None;
            self.pair_graph = None;
        }
        self.mode = mode;
    }

    /// Arm (or disarm) the node-price probe. Like [`GpuModel::set_mode`] this
    /// drops any captured chain, since the probe changes which launches the
    /// body issues. A probe with either lever set makes the chain a timing
    /// instrument: `skip_quant` leaves activation buffers unwritten, so the
    /// tokens that come out are not the model's answer.
    pub fn set_probe(&mut self, probe: StepProbe) -> Result<(), GpuError> {
        probe.check(self.ctx_max)?;
        let (_, _, body) = self.body_parts("GpuModel::set_probe")?;
        body.set_probe(probe)?;
        self.step_graph = None;
        self.pair_graph = None;
        if let Some(stage) = self.stages.first_mut() {
            stage.graph = None;
            stage.graph_of = None;
        }
        Ok(())
    }

    /// Rewind to position 0 with empty caches — the fresh-context state for
    /// the next prompt. The weights, the arena and any captured chain stay
    /// (they do not depend on the cache contents).
    pub fn reset(&mut self) -> Result<(), GpuError> {
        // The body owns its row store, so it owns what "empty" means there.
        let (gpu, _, body) = self.body_parts("GpuModel::reset")?;
        body.reset(gpu)?;
        // Empty caches hold nothing a fault condemned.
        gpu.clear_fault()?;
        self.poisoned = None;
        self.pos = 0;
        Ok(())
    }

    /// Fill the first `rows` cache rows of every resident layer and stand at
    /// position `rows` — the state a prompt of `rows` tokens leaves behind,
    /// without decoding one. There is no prefill kernel, so a deep prompt
    /// costs one body per token; a prepared cache buys the same step shape
    /// for a copy. This is an instrument: the rows are not what the model
    /// would have written, so the tokens that come out are meaningless.
    ///
    /// It is a step-shape equivalence, not a value one. The step's cost does
    /// not depend on the key values — no kernel branches on them, and the
    /// one value-dependent guard drops keys past the causal limit rather
    /// than reading their size.
    ///
    /// Synchronizes; never inside a capture.
    pub fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError> {
        if rows == 0 {
            return Err(GpuError::shape(
                "GpuModel::seed_depth",
                "rows must be at least 1",
            ));
        }
        if rows >= self.ctx_max {
            return Err(GpuError::shape(
                "GpuModel::seed_depth",
                format!(
                    "{rows} seeded rows leave no room for a step in the \
                 resident cache's {} rows",
                    self.ctx_max
                ),
            ));
        }
        let pos = crate::launch_u32("GpuModel::seed_depth", "rows", rows)?;
        let (gpu, _, body) = self.body_parts("GpuModel::seed_depth")?;
        body.seed_depth(gpu, rows)?;
        self.pos = pos;
        Ok(())
    }

    /// Capture the whole chain — every resident layer plus the head — into
    /// one graph over the resident buffers, and return its node count. One
    /// graph, not a chain of per-layer graphs: the layers differ only in
    /// which weights and which cache they address, all of them frozen at
    /// load, and a single `cuGraphLaunch` is the whole point of the capture
    /// (a chain of 27 launches would pay 27 host submits per token).
    pub fn capture_step(&mut self) -> Result<usize, GpuError> {
        let GpuModel {
            step_graph,
            head,
            stages,
            ..
        } = self;
        let head = head.as_mut().ok_or(GpuError::state(
            "GpuModel::capture_step",
            "no output head — load with load_full",
        ))?;
        let stage = stages
            .first_mut()
            .ok_or(GpuError::state("GpuModel::capture_step", "no stage"))?;
        let Some(Residency { weights, body }) = stage.residency.as_mut() else {
            return Err(GpuError::state(
                "GpuModel::capture_step",
                "stage carries no residency",
            ));
        };
        let gpu = &stage.gpu;
        let graph = gpu.capture(|_| body.enqueue_chain(gpu, weights, head))?;
        let nodes = graph.node_count();
        *step_graph = Some(graph);
        Ok(nodes)
    }

    /// Every node of the captured whole chain, as the driver lists them —
    /// what a structure gate counts the kinds of.
    pub fn step_graph_nodes(&self) -> Result<Vec<NodeInfo>, GpuError> {
        self.step_graph
            .as_ref()
            .ok_or(GpuError::state(
                "GpuModel::step_graph_nodes",
                "no captured chain",
            ))?
            .nodes()
    }

    /// Enqueue the whole chain eagerly on the engine stream. Pure enqueues —
    /// the same body [`GpuModel::capture_step`] records.
    fn enqueue_chain_step(&mut self) -> Result<(), GpuError> {
        let GpuModel { head, stages, .. } = self;
        let head = head.as_mut().ok_or(GpuError::state(
            "GpuModel::step",
            "no output head — load with load_full",
        ))?;
        let stage = stages
            .first_mut()
            .ok_or(GpuError::state("GpuModel::step", "no stage"))?;
        let Some(Residency { weights, body }) = stage.residency.as_mut() else {
            return Err(GpuError::state(
                "GpuModel::step",
                "stage carries no residency",
            ));
        };
        body.enqueue_chain(&stage.gpu, weights, head)
    }

    /// Feed `tokens` through the chain one position at a time and return the
    /// argmax of the LAST one — the greedy next token. Each token gets its
    /// own parameter refresh (outside any capture) and one body; only the
    /// argmax readback at the end synchronizes the stream, so a prompt of P
    /// tokens is P bodies and one sync.
    ///
    /// Positions continue from wherever the model stands: a prompt then its
    /// continuation is `step(&prompt)` followed by one `step(&[tok])` per
    /// generated token. [`GpuModel::reset`] rewinds. A fault any token
    /// raised is the call's error, also when a later token then fails for
    /// another reason ([`GpuModel::note_fault`]).
    pub fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        self.refuse_if_poisoned("GpuModel::step")?;
        if tokens.is_empty() {
            return Err(GpuError::shape("GpuModel::step", "empty token slice"));
        }
        if self.head.is_none() {
            return Err(GpuError::state(
                "GpuModel::step",
                "no output head — load with load_full",
            ));
        }
        if self.stages.len() != 1 || self.stages[0].layers.start != 0 {
            return Err(GpuError::shape(
                "GpuModel::step",
                "the token loop needs the single whole-model stage of \
                 load_full",
            ));
        }
        if self.mode == StepMode::Graph && self.step_graph.is_none() {
            self.capture_step()?;
        }
        let token = self.run_tokens(tokens);
        self.note_fault("GpuModel::step", token)
    }

    /// [`GpuModel::step`]'s tokens once its checks have passed: each
    /// position's refresh and chain, then the head's readback. An error
    /// here can follow launches, so the caller passes it through
    /// [`GpuModel::note_fault`].
    fn run_tokens(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        for &token in tokens {
            let pos = self.pos;
            self.check_pos(pos, "GpuModel::step")?;
            self.arm_launcher();
            self.refresh_params(token, pos)?;
            let r = match self.mode {
                StepMode::Eager => self.enqueue_chain_step(),
                StepMode::Graph => self.replay_graph(Chain::Step),
            };
            self.name_host_refusal(r)?;
            self.pos = pos + 1;
        }
        let gpu = &self.stages[0].gpu;
        self.head
            .as_ref()
            .ok_or(GpuError::state("GpuModel::step", "no output head"))?
            .token(gpu)
    }

    /// A step's enqueue or replay result, with a host refusal named: when the
    /// body's host service refused input the card should already have
    /// refused, it released every wait of the step, so the step drains, and
    /// the fault word read on the engine stream behind it names the refusal
    /// ([`name_refusal`]) — the card's fault when the card raised one at or
    /// before that layer, else the host's error. The caller's
    /// [`GpuModel::note_fault`] then turns a host error into the fault when
    /// the word holds a later layer's, so the call still ends poisoned; the
    /// refusal itself stays in the tier's record. A failed read names the
    /// refusal and the step's error beside its own. Any other result passes
    /// through.
    pub(crate) fn name_host_refusal(&mut self, r: Result<(), GpuError>) -> Result<(), GpuError> {
        const WHAT: &str = "GpuModel::name_host_refusal";
        let Err(e) = r else {
            return Ok(());
        };
        let Ok((gpu, _, body)) = self.body_parts(WHAT) else {
            return Err(e);
        };
        let Some(refusal) = body.take_host_refusal() else {
            return Err(e);
        };
        match gpu.fault() {
            Ok(word) => Err(name_refusal(&refusal, word)),
            Err(s) => Err(GpuError::shape(
                WHAT,
                format!(
                    "{} refused the input of layer {} ({}) and the step failed ({e}), and reading \
                     the fault word behind it failed too ({s})",
                    refusal.what, refusal.layer, refusal.detail
                ),
            )),
        }
    }

    /// A step on a poisoned model is refused, naming the fault.
    fn refuse_if_poisoned(&self, what: &'static str) -> Result<(), GpuError> {
        match self.poisoned {
            Some(fault) => Err(GpuError::Poisoned { what, fault }),
            None => Ok(()),
        }
    }

    /// Pass through the result of `what`, a call that may have launched,
    /// remembering a fault it carries: the state the faulted call wrote
    /// poisons every later call until [`GpuModel::reset`]. A fault is the
    /// call's error whatever else failed: any other error is read behind
    /// ([`GpuModel::fault_behind`]). The fault prints its site mask in this
    /// body's step order.
    fn note_fault<T>(&mut self, what: &'static str, r: Result<T, GpuError>) -> Result<T, GpuError> {
        let mut e = match r {
            Ok(v) => return Ok(v),
            Err(e @ GpuError::Fault { .. }) => e,
            Err(e) => self.fault_behind(what, e),
        };
        if let GpuError::Fault { fault, .. } = &mut e {
            *fault = fault.in_arch(B::arch());
            self.poisoned = Some(*fault);
        }
        Err(e)
    }

    /// `e`, the error of `what` after launches that may have raised the
    /// fault word, named by the word: the engine stream waited for, then the
    /// word read. A raised word is the error whatever `e` was, so the fault
    /// never outlives the call that raised it — the next call's readback
    /// would name it. A clean word leaves `e`. A failed wait or read names
    /// `e` beside its own error.
    fn fault_behind(&self, what: &'static str, e: GpuError) -> GpuError {
        let Some(gpu) = self.stages.first().map(|s| &s.gpu) else {
            return e;
        };
        let read = gpu
            .stream()
            .synchronize()
            .map_err(GpuError::from)
            .and_then(|()| gpu.fault());
        match read {
            Ok(Some(fault)) => GpuError::Fault { what, fault },
            Ok(None) => e,
            Err(s) => GpuError::shape(
                what,
                format!(
                    "the call failed ({e}), and waiting for its launches or reading the fault \
                     word failed too ({s})"
                ),
            ),
        }
    }

    /// The fault that poisons this model, if a step has read one back.
    #[must_use]
    pub fn poisoned(&self) -> Option<Fault> {
        self.poisoned
    }

    /// Run `t` at the next position and `t1` at the one after as one pair
    /// pass ([`ChainBody::enqueue_pair`]) and return each row's greedy next
    /// token: after `t`, and after `t1`. Bit for bit `step(&[t])` then
    /// `step(&[t1])`; the model stands two positions on. The first call
    /// makes the second row's head, and in graph mode captures the pass.
    /// A caller that does not keep `t1` takes it back with
    /// [`GpuModel::rollback`]. A fault the pass raised is its error, as
    /// [`GpuModel::step`]'s is.
    pub fn step_pair(&mut self, t: u32, t1: u32) -> Result<[u32; 2], GpuError> {
        const WHAT: &str = "GpuModel::step_pair";
        self.refuse_if_poisoned(WHAT)?;
        if self.head.is_none() {
            return Err(GpuError::state(
                WHAT,
                "no output head — load with load_full",
            ));
        }
        if self.stages.len() != 1 || self.stages[0].layers.start != 0 {
            return Err(GpuError::shape(
                WHAT,
                "the token loop needs the single whole-model stage of load_full",
            ));
        }
        let pos = self.pos;
        self.check_pos(pos + 1, WHAT)?;
        if self.pair_head.is_none() {
            let (gpu, w, body) = self.body_parts(WHAT)?;
            let head = Head::new(gpu, w, body.head_eps())?;
            self.pair_head = Some(head);
        }
        if self.mode == StepMode::Graph && self.pair_graph.is_none() {
            self.capture_pair()?;
        }
        self.arm_launcher();
        {
            let (gpu, _, body) = self.body_parts(WHAT)?;
            body.decode_pair(gpu.stream(), [t, t1], pos)?;
        }
        let tokens = self.run_pair(pos);
        self.note_fault(WHAT, tokens)
    }

    /// [`GpuModel::step_pair`]'s pass at `pos` once its rows are planned:
    /// the enqueue or replay, then both heads' readbacks. An error here can
    /// follow launches, so the caller passes it through
    /// [`GpuModel::note_fault`].
    fn run_pair(&mut self, pos: u32) -> Result<[u32; 2], GpuError> {
        let r = match self.mode {
            StepMode::Eager => self.enqueue_pair_step(),
            StepMode::Graph => self.replay_graph(Chain::Pair),
        };
        self.name_host_refusal(r)?;
        self.pos = pos + 2;
        let gpu = &self.stages[0].gpu;
        let [a, b] = self.pair_heads()?;
        Ok([a.token(gpu)?, b.token(gpu)?])
    }

    /// Wake the launch thread, if any, ahead of a graph replay's refresh.
    fn arm_launcher(&self) {
        if self.mode == StepMode::Graph
            && let Some(l) = &self.launcher
        {
            l.arm();
        }
    }

    /// Launch the captured `chain` — the one-token step's or the pair's — and
    /// serve the host's share of the replay ([`launcher::replay`]: the only
    /// place a replay of either is launched).
    fn replay_graph(&mut self, chain: Chain) -> Result<(), GpuError> {
        const WHAT: &str = "GpuModel::replay_graph";
        let GpuModel {
            step_graph,
            pair_graph,
            launcher,
            launch_stats,
            stages,
            ..
        } = self;
        let graph = match chain {
            Chain::Step => step_graph.as_ref(),
            Chain::Pair => pair_graph.as_ref(),
        }
        .ok_or(GpuError::state(WHAT, "no captured chain"))?;
        let Some(Stage { gpu, residency, .. }) = stages.first_mut() else {
            return Err(GpuError::state(WHAT, "no stage"));
        };
        let Some(Residency { body, .. }) = residency.as_mut() else {
            return Err(GpuError::state(WHAT, "stage carries no residency"));
        };
        launcher::replay(
            launcher.as_ref(),
            launch_stats,
            graph,
            gpu.stream(),
            || match chain {
                Chain::Step => body.serve_replay(),
                Chain::Pair => body.serve_replay_pair(),
            },
        )
    }

    /// Capture the pair pass into its own graph over the resident buffers
    /// and return its node count. The one-token step's capture stays.
    pub fn capture_pair(&mut self) -> Result<usize, GpuError> {
        const WHAT: &str = "GpuModel::capture_pair";
        let GpuModel {
            pair_graph,
            head,
            pair_head,
            stages,
            ..
        } = self;
        let heads = [
            head.as_mut()
                .ok_or(GpuError::state(WHAT, "no output head"))?,
            pair_head
                .as_mut()
                .ok_or(GpuError::state(WHAT, "no second head: step_pair makes it"))?,
        ];
        let stage = stages
            .first_mut()
            .ok_or(GpuError::state(WHAT, "no stage"))?;
        let Some(Residency { weights, body }) = stage.residency.as_mut() else {
            return Err(GpuError::state(WHAT, "stage carries no residency"));
        };
        let gpu = &stage.gpu;
        let graph = gpu.capture(|_| body.enqueue_pair(gpu, weights, heads))?;
        let nodes = graph.node_count();
        *pair_graph = Some(graph);
        Ok(nodes)
    }

    /// Every node of the captured pair pass, as the driver lists them.
    pub fn pair_graph_nodes(&self) -> Result<Vec<NodeInfo>, GpuError> {
        self.pair_graph
            .as_ref()
            .ok_or(GpuError::state(
                "GpuModel::pair_graph_nodes",
                "no captured pair",
            ))?
            .nodes()
    }

    /// Enqueue the pair pass eagerly on the engine stream.
    fn enqueue_pair_step(&mut self) -> Result<(), GpuError> {
        const WHAT: &str = "GpuModel::step_pair";
        let GpuModel {
            head,
            pair_head,
            stages,
            ..
        } = self;
        let heads = [
            head.as_mut()
                .ok_or(GpuError::state(WHAT, "no output head"))?,
            pair_head
                .as_mut()
                .ok_or(GpuError::state(WHAT, "no second head"))?,
        ];
        let stage = stages
            .first_mut()
            .ok_or(GpuError::state(WHAT, "no stage"))?;
        let Some(Residency { weights, body }) = stage.residency.as_mut() else {
            return Err(GpuError::state(WHAT, "stage carries no residency"));
        };
        body.enqueue_pair(&stage.gpu, weights, heads)
    }

    fn pair_heads(&self) -> Result<[&Head; 2], GpuError> {
        const WHAT: &str = "GpuModel::pair_heads";
        Ok([
            self.head
                .as_ref()
                .ok_or(GpuError::state(WHAT, "no output head"))?,
            self.pair_head
                .as_ref()
                .ok_or(GpuError::state(WHAT, "no pair pass has run"))?,
        ])
    }

    /// Each row's logits of the last [`GpuModel::step_pair`] (`n_vocab` f32
    /// each). Blocking read; gate/debug use.
    pub fn pair_logits(&self) -> Result<[Vec<f32>; 2], GpuError> {
        let gpu = &self
            .stages
            .first()
            .ok_or(GpuError::state("GpuModel::pair_logits", "no stage"))?
            .gpu;
        let [a, b] = self.pair_heads()?;
        Ok([a.logits_to_host(gpu)?, b.logits_to_host(gpu)?])
    }

    /// Take back the positions from `pos` on ([`ChainBody::rollback`]): the
    /// next step runs at `pos`.
    pub fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        if pos > self.pos {
            return Err(GpuError::shape(
                "GpuModel::rollback",
                format!("back to position {pos} from {}", self.pos),
            ));
        }
        let (_, _, body) = self.body_parts("GpuModel::rollback")?;
        body.rollback(pos)?;
        self.pos = pos;
        Ok(())
    }

    /// The head's logits of the last `step` (`n_vocab` f32). Blocking read;
    /// gate/debug use. After [`GpuModel::step_pair`] these are row A's, the
    /// logits after `t`: the pair pass writes row A through this same head
    /// and row B through the pair head ([`GpuModel::pair_logits`]).
    pub fn logits(&self) -> Result<Vec<f32>, GpuError> {
        let head = self
            .head
            .as_ref()
            .ok_or(GpuError::state("GpuModel::logits", "no output head"))?;
        let gpu = &self
            .stages
            .first()
            .ok_or(GpuError::state("GpuModel::logits", "no stage"))?
            .gpu;
        head.logits_to_host(gpu)
    }

    // --------------------------------------------- what the body's own impl
    // --------------------------------------------- blocks are handed

    /// The one resident stage, split into the three things a body's chain
    /// needs — for the body's own instruments, in whichever crate the body
    /// lives. Every assembled path needs exactly one stage carrying
    /// residency; `what` names the caller in the error.
    pub fn body_parts(&mut self, what: &'static str) -> Result<(&Gpu, &Weights, &mut B), GpuError> {
        if self.stages.len() != 1 {
            return Err(GpuError::state(
                what,
                "the assembled step needs the single stage of load_blocks",
            ));
        }
        let Stage { gpu, residency, .. } = &mut self.stages[0];
        let Some(Residency { weights, body }) = residency.as_mut() else {
            return Err(GpuError::state(
                what,
                "stage carries no residency (load_blocks fills it)",
            ));
        };
        Ok((gpu, weights, body))
    }

    /// The one resident stage's body, for an instrument that only reads it.
    pub fn body(&self, what: &'static str) -> Result<&B, GpuError> {
        match self.stages.first().and_then(|s| s.residency.as_ref()) {
            Some(Residency { body, .. }) => Ok(body),
            None => Err(GpuError::state(
                what,
                "stage carries no residency (load_blocks fills it)",
            )),
        }
    }

    /// As [`GpuModel::body_parts`], requiring the stage to hold block 0.
    pub(crate) fn block0_parts(
        &mut self,
        what: &'static str,
    ) -> Result<(&Gpu, &Weights, &mut B), GpuError> {
        if self.stages.len() != 1 || self.stages[0].layers.start != 0 {
            return Err(GpuError::state(
                what,
                "the assembled block-0 step needs a stage of load_blocks \
                 starting at layer 0",
            ));
        }
        self.body_parts(what)
    }

    /// The slot of layer `l` inside the resident range — the index of its
    /// cache and of its names in the body.
    pub(crate) fn layer_slot(&self, l: usize, what: &'static str) -> Result<usize, GpuError> {
        if self.stages.len() != 1 {
            return Err(GpuError::state(
                what,
                "the assembled step needs the single stage of load_blocks",
            ));
        }
        let layers = self.stages[0].layers.clone();
        if !layers.contains(&l) {
            return Err(GpuError::shape(
                what,
                format!("layer {l} is outside the resident range {layers:?}"),
            ));
        }
        Ok(l - layers.start)
    }

    /// Build the body's input for `(token, pos)` and write it to the device.
    /// Runs before an eager enqueue or a graph replay; a captured graph reads
    /// those buffers at run time, which is what lets one graph serve every
    /// position.
    pub(crate) fn refresh_params(&mut self, token: u32, pos: u32) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("GpuModel::refresh_params")?;
        let stream = gpu.stream();
        let input = body.decode_input(token, pos)?;
        body.refresh(stream, &input)
    }

    /// Capture one body of the stage's own graph — the per-layer capture the
    /// block gates drive — and record what it is a capture OF. `want` is the
    /// `(layer, embeds in front)` identity a later replay must name.
    pub(crate) fn capture_stage(
        &mut self,
        want: (usize, bool),
        f: impl FnOnce(&Gpu, &Weights, &mut B) -> Result<(), GpuError>,
    ) -> Result<usize, GpuError> {
        let Stage {
            gpu,
            graph,
            graph_of,
            residency,
            ..
        } = &mut self.stages[0];
        let Some(Residency { weights, body }) = residency.as_mut() else {
            return Err(GpuError::state(
                "GpuModel::capture_stage",
                "stage carries no residency",
            ));
        };
        let captured = gpu.capture(|_| f(gpu, weights, body))?;
        let nodes = captured.node_count();
        *graph = Some(captured);
        *graph_of = Some(want);
        Ok(nodes)
    }

    /// Launch the stage's graph, requiring it to be the capture of `want`
    /// (layer, embeds in front).
    pub(crate) fn launch_graph(&self, want: (usize, bool)) -> Result<(), GpuError> {
        let stage = &self.stages[0];
        let graph = stage.graph.as_ref().ok_or(GpuError::state(
            "GpuModel::launch_graph",
            "no captured graph",
        ))?;
        if stage.graph_of != Some(want) {
            return Err(GpuError::shape(
                "GpuModel::launch_graph",
                format!(
                    "the captured graph is {:?} (layer, embed), the replay \
                 wants {want:?}",
                    stage.graph_of
                ),
            ));
        }
        graph.launch(stage.gpu.stream())
    }

    /// The engine stream of the one stage — what an instrument synchronizes
    /// on after driving a replay.
    pub(crate) fn stage_stream(&self) -> Result<&CudaStream, GpuError> {
        Ok(self
            .stages
            .first()
            .ok_or(GpuError::state("GpuModel::stage_stream", "no stage"))?
            .gpu
            .stream())
    }

    /// Err unless `pos` leaves room for one more row in the resident cache.
    pub(crate) fn check_pos(&self, pos: u32, what: &'static str) -> Result<(), GpuError> {
        if pos as usize + 1 > self.ctx_max {
            return Err(GpuError::shape(
                what,
                format!(
                    "pos {pos} + 1 exceeds the resident cache's {} rows",
                    self.ctx_max
                ),
            ));
        }
        Ok(())
    }

    /// Run `n` positions as one eager pass the body enqueues itself — `pass`
    /// gets the stage's parts, the output head and the first position — and
    /// stand `n` positions on. When `pass` reports that it enqueued the
    /// head, return the head's token (a blocking read); a pass that does not
    /// is not read back, and leaves the fault word to the head of a later
    /// pass of the same call. A fault the pass returns, the head's readback
    /// carries, or the word holds behind any other error of the pass
    /// ([`GpuModel::note_fault`]) poisons the model.
    pub fn run_rows(
        &mut self,
        n: usize,
        what: &'static str,
        pass: impl FnOnce(&Gpu, &Weights, &mut B, &mut Head, u32) -> Result<bool, GpuError>,
    ) -> Result<Option<u32>, GpuError> {
        self.refuse_if_poisoned(what)?;
        if n == 0 || self.stages.len() != 1 || self.stages[0].layers.start != 0 {
            return Err(GpuError::shape(
                what,
                "a pass needs positions and the single whole-model stage",
            ));
        }
        let (pos, n) = (self.pos, crate::launch_u32(what, "positions", n)?);
        self.check_pos(pos + n - 1, what)?;
        let GpuModel { head, stages, .. } = self;
        let head = head
            .as_mut()
            .ok_or(GpuError::state(what, "no output head"))?;
        let Stage { gpu, residency, .. } = &mut stages[0];
        let Some(Residency { weights, body }) = residency.as_mut() else {
            return Err(GpuError::state(what, "stage carries no residency"));
        };
        let token = match pass(gpu, weights, body, head, pos) {
            Ok(true) => head.token(gpu).map(Some),
            Ok(false) => Ok(None),
            Err(e) => Err(e),
        };
        let token = self.note_fault(what, token)?;
        self.pos = pos + n;
        Ok(token)
    }
}

/// `block_count` of `file`: the layer count.
fn block_count(file: &Split, what: &'static str) -> Result<usize, GpuError> {
    let n = file
        .arch_get_u64("block_count")
        .ok_or(GpuError::metadata(what, "block_count"))?;
    usize::try_from(n).map_err(|_| GpuError::shape(what, format!("block_count {n}")))
}

/// The reader of `file`'s one shard, for a caller that reads a single file;
/// a split set of several shards is refused.
pub(crate) fn one_shard<'a>(
    file: &'a Split,
    what: &'static str,
) -> Result<&'a gguf::Gguf, GpuError> {
    match (file.shard_count(), file.shard(0)) {
        (1, Some(g)) => Ok(g),
        (n, _) => Err(GpuError::shape(
            what,
            format!("the file is {n} shards, and this reads one"),
        )),
    }
}

impl<B: ChainBody> Engine for GpuModel<B> {
    fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        GpuModel::step(self, tokens)
    }

    fn reset(&mut self) -> Result<(), GpuError> {
        GpuModel::reset(self)
    }

    fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError> {
        GpuModel::seed_depth(self, rows)
    }

    fn pos(&self) -> u32 {
        GpuModel::pos(self)
    }

    fn resident_bytes(&self) -> usize {
        GpuModel::resident_bytes(self)
    }

    fn arch(&self) -> Arch {
        B::arch()
    }
}
