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
pub(crate) mod lookup;
pub(crate) mod probe;

pub use kernels::{Q8_0GemvHeadsArgs, StepKernels};
pub(crate) use lookup::f32_gain;
pub use probe::{OpTime, StepProbe};

use crate::head::Head;
use crate::hybrid::HybridConfig;
use crate::weights::Weights;
use crate::{Gpu, GpuError, Graph, NodeInfo};
use cuda_core::CudaStream;
use model::arch::Arch;
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

    /// The architecture this body is the chain of.
    fn arch() -> Arch;

    /// File into `w` the weights this architecture's plan computes at load
    /// for the blocks of `layers` — after the file tensors are resident,
    /// before [`ChainBody::load`] reads them. Required, with no default body,
    /// so an architecture cannot forget its derived weights silently.
    fn derive(
        stream: &CudaStream,
        gguf: &gguf::Gguf,
        layers: Range<usize>,
        w: &mut Weights,
    ) -> Result<(), GpuError>;

    /// Everything the chain of `layers` needs resident, over the weights the
    /// caller has already loaded and derived: the caches, the arena and the
    /// step module.
    fn load(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
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
    /// kernels address. The default refuses: an architecture without a
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
    /// keeps `file`. The default refuses.
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

    /// Serve the host's share of the chain a graph replay just submitted, in
    /// chain order, before anything else waits on the stream. A body whose
    /// chain holds no host work has nothing to do, which is the default.
    fn serve_replay(&mut self) -> Result<(), GpuError> {
        Ok(())
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
    /// The output head — present only on a model that holds every block
    /// ([`GpuModel::load_full`]). A partial stage has no logits to take, so
    /// `step` refuses on one.
    head: Option<Head>,
    stages: Vec<Stage<B>>,
    /// KV rows the resident caches were sized for; `step` refuses to grow them.
    ctx_max: usize,
    mode: StepMode,
    /// The cache row the next `step` token lands in.
    pos: u32,
}

impl<B: ChainBody> GpuModel<B> {
    /// The whole model as one stage (metadata only — a resident stage comes
    /// from [`GpuModel::load_blocks`]).
    pub fn load(gguf: &gguf::Gguf, ctx_max: usize) -> Result<GpuModel<B>, GpuError> {
        GpuModel::load_staged(gguf, ctx_max, &[])
    }

    /// Split the blocks at `cuts` (strictly ascending, each in
    /// `1..block_count`): `cuts.len() + 1` stages. Reads the metadata `step`
    /// needs; the stages carry no residency until [`GpuModel::load_blocks`]
    /// fills one.
    pub(crate) fn load_staged(
        gguf: &gguf::Gguf,
        ctx_max: usize,
        cuts: &[usize],
    ) -> Result<GpuModel<B>, GpuError> {
        if ctx_max == 0 {
            return Err(GpuError::shape("GpuModel::load", "ctx_max must be >= 1"));
        }
        let n_layers =
            gguf.block_count()
                .ok_or(GpuError::metadata("GpuModel::load", "block_count"))? as usize;
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
            mode: StepMode::Graph,
            pos: 0,
        })
    }

    /// One stage over `layers` WITH residency: every tensor of that range
    /// plus the globals in its kernels' device format, the weights the body
    /// derives from them, and the body those weights drive — its caches, its
    /// m = 1 arena and its step module. The geometry checks run at load, not
    /// mid-step.
    pub fn load_blocks(
        gguf: &gguf::Gguf,
        ctx_max: usize,
        layers: Range<usize>,
    ) -> Result<GpuModel<B>, GpuError> {
        if ctx_max == 0 {
            return Err(GpuError::shape(
                "GpuModel::load_blocks",
                "ctx_max must be >= 1",
            ));
        }
        let n_layers = gguf
            .block_count()
            .ok_or(GpuError::metadata("GpuModel::load_blocks", "block_count"))?
            as usize;
        if layers.start >= layers.end || layers.end > n_layers {
            return Err(GpuError::shape(
                "GpuModel::load_blocks",
                format!("layer range {layers:?} outside 0..{n_layers}"),
            ));
        }
        let gpu = Gpu::new()?;
        let mut weights = Weights::load(gpu.stream(), gguf, layers.clone(), true)?;
        B::derive(gpu.stream(), gguf, layers.clone(), &mut weights)?;
        let body = B::load(&gpu, gguf, &weights, layers.clone(), ctx_max)?;
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
            mode: StepMode::Graph,
            pos: 0,
        })
    }

    /// Every block of the file resident, plus the output head: the model
    /// [`GpuModel::step`] needs. `load_blocks(0..block_count)` first (so the
    /// head's `output_norm.weight` / `output.weight` arrive with the
    /// globals), then the head over those same resident weights, normalizing
    /// with the epsilon the body's plan read.
    ///
    /// `BLOOMERY_HYBRID_NL` below the file's expert count makes this the
    /// hybrid load of [`GpuModel::load_hybrid`] over the same file, reopened
    /// by path (`hybrid::reopen`); unset or equal to the expert count, the
    /// path below is taken unchanged.
    pub fn load_full(gguf: &gguf::Gguf, ctx_max: usize) -> Result<GpuModel<B>, GpuError> {
        if let Some(cfg) = HybridConfig::from_levers(gguf)? {
            return GpuModel::load_hybrid(crate::hybrid::reopen(gguf)?, ctx_max, cfg);
        }
        let n_layers = gguf
            .block_count()
            .ok_or(GpuError::metadata("GpuModel::load_full", "block_count"))?
            as usize;
        let mut m = GpuModel::<B>::load_blocks(gguf, ctx_max, 0..n_layers)?;
        let head = {
            let (gpu, weights, body) = m.body_parts("GpuModel::load_full")?;
            let eps = body.head_eps();
            Head::new(gpu, weights, eps)?
        };
        m.head = Some(head);
        Ok(m)
    }

    /// Every block of `file` resident, plus the output head, with each MoE
    /// layer's experts `[0, cfg.n_l)` on the card and the rest computed on
    /// the host inside the captured step (crate::hybrid). The model keeps
    /// `file` for its host experts.
    pub fn load_hybrid(
        file: gguf::Split,
        ctx_max: usize,
        cfg: HybridConfig,
    ) -> Result<GpuModel<B>, GpuError> {
        let what = "GpuModel::load_hybrid";
        if ctx_max == 0 {
            return Err(GpuError::shape(what, "ctx_max must be >= 1"));
        }
        let n_layers = file
            .shard(0)
            .and_then(gguf::Gguf::block_count)
            .ok_or(GpuError::metadata(what, "block_count"))?;
        let n_layers = usize::try_from(n_layers)
            .map_err(|_| GpuError::shape(what, format!("block_count {n_layers}")))?;
        let gpu = Gpu::new()?;
        let mut weights = B::hybrid_weights(gpu.stream(), &file, 0..n_layers, cfg.n_l)?;
        let body = B::load_hybrid(&gpu, file, &mut weights, 0..n_layers, ctx_max, cfg)?;
        let head = Head::new(&gpu, &weights, body.head_eps())?;
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
            mode: StepMode::Graph,
            pos: 0,
        })
    }

    /// The model's stages, in layer order.
    pub fn stages(&self) -> &[Stage<B>] {
        &self.stages
    }

    /// Device bytes of everything this model holds resident: the stage's
    /// weights and body, plus the head's scratch when it has one.
    pub fn resident_bytes(&self) -> usize {
        self.stages.iter().map(Stage::resident_bytes).sum::<usize>()
            + self.head.as_ref().map_or(0, Head::resident_bytes)
    }

    /// The cache row the next [`GpuModel::step`] token lands in.
    pub fn pos(&self) -> u32 {
        self.pos
    }

    /// Choose how the chain submits. Changing the mode drops any captured
    /// chain: the graph is a recording of this body over these buffers, and
    /// a later `Graph` run recaptures rather than replay a stale one.
    pub fn set_mode(&mut self, mode: StepMode) {
        if mode != self.mode {
            self.step_graph = None;
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
    /// generated token. [`GpuModel::reset`] rewinds.
    pub fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
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
        for &token in tokens {
            let pos = self.pos;
            self.check_pos(pos, "GpuModel::step")?;
            self.refresh_params(token, pos)?;
            match self.mode {
                StepMode::Eager => self.enqueue_chain_step()?,
                StepMode::Graph => {
                    self.step_graph
                        .as_ref()
                        .ok_or(GpuError::state("GpuModel::step", "no captured chain"))?
                        .launch(self.stages[0].gpu.stream())?;
                    self.body_parts("GpuModel::step")?.2.serve_replay()?;
                }
            }
            self.pos = pos + 1;
        }
        let gpu = &self.stages[0].gpu;
        self.head
            .as_ref()
            .ok_or(GpuError::state("GpuModel::step", "no output head"))?
            .token(gpu)
    }

    /// The head's logits of the last `step` (`n_vocab` f32). Blocking read;
    /// gate/debug use.
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
    /// needs. Every assembled path needs exactly one stage carrying
    /// residency; `what` names the caller in the error.
    pub(crate) fn body_parts(
        &mut self,
        what: &'static str,
    ) -> Result<(&Gpu, &Weights, &mut B), GpuError> {
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
    pub(crate) fn body(&self, what: &'static str) -> Result<&B, GpuError> {
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
