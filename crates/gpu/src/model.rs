//! `GpuModel<B>` — the GPU engine; an architecture's CPU `forward::step` is its reference
//! (docs/gpu-design.md decisions 1 and 7). One card holds the model: its
//! weights, the body's resident buffers and the output head live on the device
//! from load, and `step` enqueues one decode step and synchronizes once for
//! the argmax.
//!
//! This file is the part that is the same for every architecture
//! (docs/arch-split.md): the drop order of the captured chains, the graph
//! cache keyed by the rows a chain runs, the rule that a mode change discards
//! a capture, the position and its advance and the single argmax readback.
//! Which weights a model derives at load, what a layer chain enqueues, which
//! rows it appends to which cache and what one step's host input is belong to
//! a [`ChainBody`] under [`crate::arch`], and each model's constructor lives
//! beside its body; `GpuModel` is generic over it and monomorphic at every
//! call site, so nothing on the token path is a virtual call.
//! [`GpuModel::step`] runs the body's whole chain — layer 0 embedding its
//! token, every later layer reading the previous layer's output residual, the
//! `head.rs` output head on the last — and returns the argmax of the last
//! token it was given. The chain submits in one of two modes ([`StepMode`]):
//! eager, which enqueues the body per token, or graph, which captures the
//! body once and replays it. Only the argmax readback synchronizes.
//!
//! What only some bodies can do is a capability trait, and the method that
//! drives it exists only where the body implements it: [`HostServed`] (a host
//! tier inside the step), [`Rows`] (a pass of several rows), [`Rollback`],
//! [`Instrumented`] (a synthetic depth) and [`Probed`] (the node-price probe).

pub(crate) mod kernels;
pub(crate) mod lookup;
pub(crate) mod probe;

pub use kernels::{Q8_0GemvHeadsArgs, StepKernels};
pub(crate) use lookup::f32_gain;
pub use probe::{OpTime, StepProbe};

use crate::fault::Fault;
use crate::head::Head;
use crate::hybrid::{Chain, HostResidency, Refusal, name_refusal};
use crate::weights::Weights;
use crate::{Gpu, GpuError, Graph, NodeInfo};
use bloomery_levers::HostCfg;
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
/// model, never inside a layer. How a body loads is its own constructor's
/// (`open`, `open_hybrid`, `open_placed` beside the body), which returns the
/// resident [`GpuModel`].
pub trait ChainBody: Sized {
    /// The per-replay host values of one step. deepseek2 is the token and the
    /// position; deepseek41 adds the engram rows and the compressed-index
    /// plan, which is why this is a type and not two arguments.
    type Input;

    /// The host service a replay of this body's chain needs ([`HostServed`]):
    /// the body itself when its chain can hold host work, [`NoHost`] when it
    /// runs on the card alone.
    type Host: HostServed;

    /// The architecture this body is the chain of.
    fn arch() -> Arch;

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
    /// chain stay: none of them depends on the cache contents. The contract
    /// is the load's: every state a call writes and a later call reads before
    /// writing it is back where the load left it, so the next prompt writes
    /// the bits it writes in a fresh process ([`GpuModel::reset`]).
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError>;

    /// The rms epsilon the output head normalizes with, read from the file at
    /// load by this body's plan.
    fn head_eps(&self) -> f32;

    /// Device bytes this body holds resident: caches, arena, step module.
    fn resident_bytes(&self) -> usize;

    /// The model layers this load carries, in order: its cache slot `i` is
    /// layer `layers().start + i`.
    fn layers(&self) -> Range<usize>;

    /// The host service of this load, `None` when the chain holds no host
    /// work: what a replay is served by and a failed step's refusal is named
    /// from.
    fn host(&mut self) -> Option<&mut Self::Host>;
}

/// A body whose chain holds host work: a host tier computes part of the
/// captured step, and a replay of the chain is served on the calling thread
/// right after its launch ([`GpuModel::step`], [`GpuModel::step_rows`]).
pub trait HostServed {
    /// Serve the host's share of the chain `chain` a graph replay just
    /// submitted, in chain order, before anything else waits on the stream:
    /// [`Chain::Step`] for the one-token step, [`Rows::CHAIN`] for a
    /// multi-row pass.
    fn serve_captured(&mut self, chain: Chain) -> Result<(), GpuError>;

    /// The refusal that failed the step's host service, once
    /// ([`crate::hybrid::Hybrid::take_step_refusal`]): what the step's
    /// failure is named from after the stream has drained.
    fn take_host_refusal(&mut self) -> Option<Refusal>;
}

/// The host service of a body whose chain runs on the card alone: no value
/// of it exists, so its [`ChainBody::host`] is always `None`.
pub enum NoHost {}

impl HostServed for NoHost {
    fn serve_captured(&mut self, _chain: Chain) -> Result<(), GpuError> {
        match *self {}
    }

    fn take_host_refusal(&mut self) -> Option<Refusal> {
        match *self {}
    }
}

/// The most rows one captured chain runs: the size of the graph cache, and
/// the bound on every body's [`Rows::MAX_ROWS`].
pub const MAX_PASS_ROWS: usize = 8;

/// A body that runs several consecutive positions as one pass of `m` rows,
/// bit for bit `m` one-token steps in turn — what a verify of drafted tokens
/// runs.
pub trait Rows: ChainBody {
    /// The most rows one pass takes, 2 to [`MAX_PASS_ROWS`]. A pass of more
    /// rows does not compile ([`GpuModel::step_rows`]).
    const MAX_ROWS: usize;

    /// How the pass lays its rows over the chain, as a host tier serves a
    /// replay of it ([`HostServed::serve_captured`]).
    const CHAIN: Chain;

    /// The pass's host half: the steps of `tokens[r]` at `pos + r`, planned
    /// in turn and written into the buffers the captured pass reads.
    /// `tokens` holds 2 to [`Rows::MAX_ROWS`] ids. Never inside a capture.
    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError>;

    /// Enqueue the pass the last [`Rows::plan_rows`] planned, through every
    /// resident layer and row `r` into `heads[r]`, one head a row.
    /// Asynchronous as [`ChainBody::enqueue_chain`] is.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError>;
}

/// A body that can take positions back.
pub trait Rollback: ChainBody {
    /// Take back the positions from `pos` on, so the next step runs at `pos`.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError>;
}

/// A body that can stand at a synthetic depth.
pub trait Instrumented: ChainBody {
    /// Fill the first `rows` cache rows of every resident layer with the
    /// body's own synthetic pattern — the state a prompt of `rows` tokens
    /// leaves behind, without decoding one. An instrument: the rows are not
    /// what the model would have written.
    fn seed_depth(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError>;
}

/// A body whose chain carries the node-price probe's arms.
pub trait Probed: ChainBody {
    /// Arm (or disarm) the node-price probe on this body's arena, and drop
    /// whatever the body captured itself. The skeleton drops its own
    /// captures.
    fn set_probe(&mut self, probe: StepProbe) -> Result<(), GpuError>;
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

// ------------------------------------------------------------ graph cache

/// The captured chains, one per key: the rows the chain runs — 1 for the
/// one-token step, `m` for a pass of [`Rows`]. Nothing else keys a capture:
/// a chain reads every per-call value (position, live key count, a row's
/// lane) from device buffers refreshed before the replay, so one capture
/// serves every call of its shape.
struct Graphs([Option<Graph>; MAX_PASS_ROWS]);

impl Graphs {
    fn new() -> Graphs {
        Graphs([const { None }; MAX_PASS_ROWS])
    }

    /// The capture of `rows` rows, if one is held.
    fn get(&self, rows: usize) -> Option<&Graph> {
        self.0.get(rows.wrapping_sub(1)).and_then(Option::as_ref)
    }

    /// The slot of `rows` rows: 1 to [`MAX_PASS_ROWS`], else refused.
    fn slot(&mut self, rows: usize, what: &'static str) -> Result<&mut Option<Graph>, GpuError> {
        self.0.get_mut(rows.wrapping_sub(1)).ok_or_else(|| {
            GpuError::shape(
                what,
                format!("a chain of {rows} rows; the cache keys 1 to {MAX_PASS_ROWS}"),
            )
        })
    }

    fn any(&self) -> bool {
        self.0.iter().any(Option::is_some)
    }

    fn clear(&mut self) {
        self.0 = [const { None }; MAX_PASS_ROWS];
    }
}

// ------------------------------------------------------------------ model

/// Everything one card holds for a model, as the model's constructor hands it
/// to [`GpuModel::new`]: every piece is resident already.
pub struct Resident<B> {
    pub gpu: Gpu,
    pub weights: Weights,
    pub body: B,
    /// The output head, when the load carries it: a load of some blocks has
    /// no logits to take.
    pub head: Option<Head>,
    /// What a placed load did to the plan's host set ([`HostResidency`]).
    pub host: Option<HostResidency>,
    /// KV rows the resident caches were sized for.
    pub ctx_max: usize,
}

/// One resident model on one card. Everything `step` touches is allocated at
/// load, never per step. A second card is another `GpuModel` (a draft's).
///
/// Fields drop in declaration order, and a graph must be destroyed while
/// every buffer it addresses is still alive: the captured chains first, then
/// the heads, the host set, the body (which drops its own captures first),
/// the weights they all address, and the card last.
pub struct GpuModel<B: ChainBody> {
    /// The captured chains, keyed by rows.
    graphs: Graphs,
    /// Row `r`'s output head: row 0's is the one-token step's, made at load
    /// when the load carries the head; rows 1 on are made by the first pass
    /// of [`Rows`] that needs them. Empty on a load without the head.
    heads: Vec<Head>,
    /// What a placed load did to the plan's host set, and the lock over it
    /// when one was asked for. Declared before `body` so that it drops
    /// first: the lock's spans are pages of the file mappings the body keeps.
    host: Option<HostResidency>,
    /// On the heap, so moving the model never moves the body's own state.
    body: Box<B>,
    weights: Weights,
    gpu: Gpu,
    /// KV rows the resident caches were sized for; `step` refuses to grow them.
    ctx_max: usize,
    mode: StepMode,
    /// The cache row the next `step` token lands in. Written only through
    /// [`GpuModel::stand_at`].
    pos: u32,
    /// The fault a step read back (crate::fault): the caches and rings hold
    /// what it condemned, so every later step refuses until `reset`.
    poisoned: Option<Fault>,
}

impl<B: ChainBody> GpuModel<B> {
    /// The model over `r`, standing at position 0 in graph mode with nothing
    /// captured.
    pub fn new(r: Resident<B>) -> GpuModel<B> {
        let Resident {
            gpu,
            weights,
            body,
            head,
            host,
            ctx_max,
        } = r;
        GpuModel {
            graphs: Graphs::new(),
            heads: head.into_iter().collect(),
            host,
            body: Box::new(body),
            weights,
            gpu,
            ctx_max,
            mode: StepMode::Graph,
            pos: 0,
            poisoned: None,
        }
    }

    /// Blocks `layers` of the one-shard `file` resident on `gpu`: every
    /// tensor of that range plus the globals in its kernels' device format
    /// ([`Weights::load`]), then `body` over them — the constructor's own
    /// derived weights filed into the set, and the body with its caches, its
    /// m = 1 arena and its step module — and, when `head`, the output head
    /// over those same weights, normalizing with the epsilon the body read.
    /// The geometry checks run before anything is uploaded: a cache of 0
    /// rows, a range outside the file, a file of several shards (a model of
    /// several shards loads by its placement plan, [`GpuModel::load_placed`]).
    pub(crate) fn load_blocks(
        gpu: Gpu,
        file: &Split,
        ctx_max: usize,
        layers: Range<usize>,
        head: bool,
        body: impl FnOnce(&Gpu, &mut Weights) -> Result<B, GpuError>,
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
        let mut weights = Weights::load(gpu.stream(), file, layers, true)?;
        let body = body(&gpu, &mut weights)?;
        let head = if head {
            Some(Head::new(&gpu, &weights, body.head_eps())?)
        } else {
            None
        };
        Ok(GpuModel::new(Resident {
            gpu,
            weights,
            body,
            head,
            host: None,
            ctx_max,
        }))
    }

    /// Card `card` of `plan` resident: the segments the plan puts on the
    /// card ([`Weights::load_placed`] — whole tensors, and each routed
    /// stack's card `ExpertList` of its layer, the id prefix or a hot list's
    /// ranked ids), the weights `derive` files for the card's layers, and the
    /// body `body` builds over them, which keeps `file` for what the plan
    /// leaves on the host; plus the output head when the card carries it.
    /// Between the uploads and the body the plan's host set is read in and
    /// locked as `host` asks ([`HostResidency::at_load`]), so that no step
    /// takes the first touch of a host expert page; `host` also says whether
    /// the card segments' file pages are released once uploaded. The caches hold the plan's `ctx_max` rows, the
    /// context its budget was made for. The card is found by its name in the
    /// plan ([`Gpu::for_card`]), never by ordinal.
    pub fn load_placed(
        file: Split,
        plan: &Plan<'_>,
        card: usize,
        host: HostCfg,
        derive: impl FnOnce(&CudaStream, &Split, Range<usize>, &mut Weights) -> Result<(), GpuError>,
        body: impl FnOnce(&Gpu, Split, &Weights) -> Result<B, GpuError>,
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
        let mut weights =
            Weights::load_placed(gpu.stream(), &file, plan, card, host.card_dontneed)?;
        derive(gpu.stream(), &file, layers, &mut weights)?;
        // After the uploads, so the card's file bytes have left the page
        // cache before the host set is read in; before the body, which
        // takes `file`.
        let host = HostResidency::at_load(&file, plan, |_| true, host)?;
        let body = body(&gpu, file, &weights)?;
        let head = if spec.head {
            Some(Head::new(&gpu, &weights, body.head_eps())?)
        } else {
            None
        };
        Ok(GpuModel::new(Resident {
            gpu,
            weights,
            body,
            head,
            host: Some(host),
            ctx_max,
        }))
    }

    /// What the load did to the plan's host set — populated, locked — on a
    /// placed load ([`GpuModel::load_placed`]); `None` on any other.
    pub fn host_residency(&self) -> Option<&HostResidency> {
        self.host.as_ref()
    }

    /// The card this model runs on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The model's resident weights.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The model layers this load carries ([`ChainBody::layers`]).
    pub fn layers(&self) -> Range<usize> {
        self.body.layers()
    }

    /// Device bytes of everything this model holds resident: the weights and
    /// the body, plus the heads' scratch — a multi-row pass's heads once one
    /// has run.
    pub fn resident_bytes(&self) -> usize {
        self.weights.resident_bytes()
            + self.body.resident_bytes()
            + self.heads.iter().map(Head::resident_bytes).sum::<usize>()
    }

    /// The cache row the next [`GpuModel::step`] token lands in.
    pub fn pos(&self) -> u32 {
        self.pos
    }

    /// Stand at `pos`: the one write of the position.
    fn stand_at(&mut self, pos: u32) {
        self.pos = pos;
    }

    /// How the chain submits ([`GpuModel::set_mode`]).
    #[must_use]
    pub fn mode(&self) -> StepMode {
        self.mode
    }

    /// Choose how the chain submits. Changing the mode drops every captured
    /// chain: a graph is a recording of this body over these buffers, and a
    /// later `Graph` run recaptures rather than replay a stale one.
    pub fn set_mode(&mut self, mode: StepMode) {
        if mode != self.mode {
            self.graphs.clear();
        }
        self.mode = mode;
    }

    /// Whether any chain is captured.
    #[must_use]
    pub fn has_capture(&self) -> bool {
        self.graphs.any()
    }

    /// Rewind to position 0 with empty caches — the fresh-context state for
    /// the next prompt: a prompt after it gives the tokens and logits it
    /// gives in a fresh process right after the load, bit for bit. The
    /// weights, the arena and any captured chain stay (they do not depend on
    /// the cache contents). It also lifts a fault; a caller that must not
    /// continue past one checks [`GpuModel::poisoned`] first.
    pub fn reset(&mut self) -> Result<(), GpuError> {
        // The body owns its row store, so it owns what "empty" means there.
        self.body.reset(&self.gpu)?;
        // Empty caches hold nothing a fault condemned.
        self.gpu.clear_fault()?;
        self.poisoned = None;
        self.stand_at(0);
        Ok(())
    }

    /// Capture the whole chain — every resident layer plus the head — into
    /// one graph over the resident buffers, key 1, and return its node count.
    /// One graph, not a chain of per-layer graphs: the layers differ only in
    /// which weights and which cache they address, all of them frozen at
    /// load, and a single `cuGraphLaunch` is the whole point of the capture
    /// (a chain of 27 launches would pay 27 host submits per token).
    pub fn capture_step(&mut self) -> Result<usize, GpuError> {
        const WHAT: &str = "GpuModel::capture_step";
        let GpuModel {
            graphs,
            heads,
            body,
            weights,
            gpu,
            ..
        } = self;
        let head = heads.first_mut().ok_or(no_head(WHAT))?;
        let graph = gpu.capture(|_| body.enqueue_chain(gpu, weights, head))?;
        let nodes = graph.node_count();
        *graphs.slot(1, WHAT)? = Some(graph);
        Ok(nodes)
    }

    /// Every node of the captured whole chain, as the driver lists them —
    /// what a structure gate counts the kinds of.
    pub fn step_graph_nodes(&self) -> Result<Vec<NodeInfo>, GpuError> {
        self.graphs
            .get(1)
            .ok_or(GpuError::state(
                "GpuModel::step_graph_nodes",
                "no captured chain",
            ))?
            .nodes()
    }

    /// Enqueue the whole chain eagerly on the engine stream. Pure enqueues —
    /// the same body [`GpuModel::capture_step`] records.
    fn enqueue_chain_step(&mut self) -> Result<(), GpuError> {
        let GpuModel {
            heads,
            body,
            weights,
            gpu,
            ..
        } = self;
        let head = heads.first_mut().ok_or(no_head("GpuModel::step"))?;
        body.enqueue_chain(gpu, weights, head)
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
    /// another reason ([`GpuModel::note_fault`]). A model that carries the
    /// head carries every layer: each constructor that makes the head loads
    /// the whole chain.
    pub fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        self.refuse_if_poisoned("GpuModel::step")?;
        if tokens.is_empty() {
            return Err(GpuError::shape("GpuModel::step", "empty token slice"));
        }
        if self.heads.is_empty() {
            return Err(no_head("GpuModel::step"));
        }
        if self.mode == StepMode::Graph && self.graphs.get(1).is_none() {
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
            self.refresh_params(token, pos)?;
            let r = match self.mode {
                StepMode::Eager => self.enqueue_chain_step(),
                StepMode::Graph => self.replay(1, Chain::Step),
            };
            self.name_host_refusal(r)?;
            self.stand_at(pos + 1);
        }
        self.heads
            .first()
            .ok_or(no_head("GpuModel::step"))?
            .token(&self.gpu)
    }

    /// Launch the captured chain of `rows` rows and serve the host's share
    /// of the replay, as `chain`, when the body has a host service: the only
    /// place a captured chain is replayed.
    fn replay(&mut self, rows: usize, chain: Chain) -> Result<(), GpuError> {
        let GpuModel {
            graphs, body, gpu, ..
        } = self;
        graphs
            .get(rows)
            .ok_or(GpuError::state("GpuModel::replay", "no captured chain"))?
            .launch(gpu.stream())?;
        match body.host() {
            Some(host) => host.serve_captured(chain),
            None => Ok(()),
        }
    }

    /// A step's enqueue or replay result, with a host refusal named: when the
    /// body's host service refused input the card should already have
    /// refused, it released every wait of the step, so the step drains, and
    /// the fault word read on the engine stream behind it names the refusal
    /// ([`name_refusal`]) — the card's fault when the card raised one at or
    /// before that layer, else the host's error. When the word holds a later
    /// layer's fault, the caller's [`GpuModel::note_fault`] returns that fault
    /// with the host's error behind it, so the call ends poisoned and still
    /// names the refusal's layer, which came first. A failed read names the
    /// refusal and the step's error beside its own. Any other result passes
    /// through.
    pub(crate) fn name_host_refusal(&mut self, r: Result<(), GpuError>) -> Result<(), GpuError> {
        const WHAT: &str = "GpuModel::name_host_refusal";
        let Err(e) = r else {
            return Ok(());
        };
        let Some(refusal) = self.body.host().and_then(HostServed::take_host_refusal) else {
            return Err(e);
        };
        match self.gpu.fault() {
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
    /// word read. A raised word is the error whatever `e` was, with `e`
    /// behind it, so the fault never outlives the call that raised it — the
    /// next call's readback would name it. A clean word leaves `e`. A failed
    /// wait or read names `e` beside its own error.
    fn fault_behind(&self, what: &'static str, e: GpuError) -> GpuError {
        let gpu = &self.gpu;
        let read = gpu.fault();
        match read {
            Ok(Some(fault)) => GpuError::Fault {
                what,
                fault,
                behind: Some(Box::new(e)),
            },
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

    /// The head's logits of the last `step` (`n_vocab` f32). Blocking read;
    /// gate/debug use. After [`GpuModel::step_rows`] these are row 0's: the
    /// pass writes row 0 through this same head and row `r` through head `r`
    /// ([`GpuModel::rows_logits`]).
    pub fn logits(&self) -> Result<Vec<f32>, GpuError> {
        self.heads
            .first()
            .ok_or(no_head("GpuModel::logits"))?
            .logits_to_host(&self.gpu)
    }

    /// [`GpuModel::logits`] into `out` (`n_vocab` f32), for a caller that reads
    /// them every token into one buffer. Blocking read.
    pub fn logits_into(&self, out: &mut [f32]) -> Result<(), GpuError> {
        self.heads
            .first()
            .ok_or(no_head("GpuModel::logits_into"))?
            .logits_into_host(&self.gpu, out)
    }

    // --------------------------------------------- what the body's own impl
    // --------------------------------------------- blocks are handed

    /// The card, the weights and the body — for the body's own instruments,
    /// in whichever crate the body lives. Never fails: a model is resident
    /// from its constructor on. The `Result` and `what` are its callers'
    /// shape.
    pub fn body_parts(
        &mut self,
        _what: &'static str,
    ) -> Result<(&Gpu, &Weights, &mut B), GpuError> {
        Ok((&self.gpu, &self.weights, &mut *self.body))
    }

    /// The body, for an instrument that only reads it. Never fails, as
    /// [`GpuModel::body_parts`].
    pub fn body(&self, _what: &'static str) -> Result<&B, GpuError> {
        Ok(&*self.body)
    }

    /// Build the body's input for `(token, pos)` and write it to the device.
    /// Runs before an eager enqueue or a graph replay; a captured graph reads
    /// those buffers at run time, which is what lets one graph serve every
    /// position.
    pub(crate) fn refresh_params(&mut self, token: u32, pos: u32) -> Result<(), GpuError> {
        let input = self.body.decode_input(token, pos)?;
        self.body.refresh(self.gpu.stream(), &input)
    }

    /// The slot of layer `l` inside the resident range — the index of its
    /// cache and of its names in the body.
    pub(crate) fn layer_slot(&self, l: usize, what: &'static str) -> Result<usize, GpuError> {
        let layers = self.body.layers();
        if !layers.contains(&l) {
            return Err(GpuError::shape(
                what,
                format!("layer {l} is outside the resident range {layers:?}"),
            ));
        }
        Ok(l - layers.start)
    }

    /// The engine stream — what an instrument synchronizes on after driving
    /// a replay. Never fails, as [`GpuModel::body_parts`].
    pub(crate) fn stage_stream(&self) -> Result<&CudaStream, GpuError> {
        Ok(self.gpu.stream())
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
    /// gets the card, the weights, the body, the output head and the first
    /// position — and stand `n` positions on. When `pass` reports that it
    /// enqueued the head, return the head's token (a blocking read); a pass
    /// that does not is not read back. A fault the pass returns, the head's
    /// readback carries, or the word holds behind any other error of the pass
    /// ([`GpuModel::note_fault`]) poisons the model.
    ///
    /// The prompt call's fault-read contract has two named modes, and each
    /// body runs one:
    /// - **call-end read** (qwen3moe's prefill): only the call's last pass
    ///   enqueues the head, so the word is read once, behind the head at the
    ///   call's end; a fault an earlier pass raised is named there, and the
    ///   passes between it and the end ran on the condemned state.
    /// - **group-end read** (deepseek41's prompt call): every group's pass
    ///   returns `true` only for the group that holds the call's last
    ///   position, and a group that fails is read behind at once
    ///   ([`GpuModel::fault_behind`]), so a fault does not cross the failed
    ///   call; the call takes its positions back before it returns the error.
    pub fn run_rows(
        &mut self,
        n: usize,
        what: &'static str,
        pass: impl FnOnce(&Gpu, &Weights, &mut B, &mut Head, u32) -> Result<bool, GpuError>,
    ) -> Result<Option<u32>, GpuError> {
        self.refuse_if_poisoned(what)?;
        if n == 0 {
            return Err(GpuError::shape(what, "a pass needs positions"));
        }
        let (pos, n) = (self.pos, crate::launch_u32(what, "positions", n)?);
        self.check_pos(pos + n - 1, what)?;
        let GpuModel {
            heads,
            body,
            weights,
            gpu,
            ..
        } = self;
        let head = heads.first_mut().ok_or(no_head(what))?;
        let token = match pass(gpu, weights, &mut **body, head, pos) {
            Ok(true) => head.token(gpu).map(Some),
            Ok(false) => Ok(None),
            Err(e) => Err(e),
        };
        let token = self.note_fault(what, token)?;
        self.stand_at(pos + n);
        Ok(token)
    }
}

// -------------------------------------------------------- the capabilities

impl<B: Rows> GpuModel<B> {
    /// Run `tokens[r]` at position `pos + r`, for the model's position `pos`,
    /// as one pass of `M` rows ([`Rows::enqueue_rows`]) and return each row's
    /// greedy next token. Bit for bit `M` calls of `step(&[tokens[r]])`; the
    /// model stands `M` positions on. `M` is 2 to the body's
    /// [`Rows::MAX_ROWS`], or the call does not compile. The first pass of
    /// `M` rows makes the heads it needs, and in graph mode captures the
    /// pass (key `M`). A caller that does not keep the later rows takes them
    /// back with [`GpuModel::rollback`]. A fault the pass raised is its
    /// error, as [`GpuModel::step`]'s is.
    pub fn step_rows<const M: usize>(&mut self, tokens: [u32; M]) -> Result<[u32; M], GpuError> {
        const WHAT: &str = "GpuModel::step_rows";
        const { rows_fit::<B>(M) };
        self.refuse_if_poisoned(WHAT)?;
        let pos = self.pos;
        self.check_pos(pos + crate::launch_u32(WHAT, "rows", M - 1)?, WHAT)?;
        self.make_heads(M, WHAT)?;
        if self.mode == StepMode::Graph && self.graphs.get(M).is_none() {
            self.capture_rows::<M>()?;
        }
        self.body.plan_rows(self.gpu.stream(), &tokens, pos)?;
        let tokens = self.run_pass::<M>(pos);
        self.note_fault(WHAT, tokens)
    }

    /// [`GpuModel::step_rows`]'s pass at `pos` once its rows are planned:
    /// the enqueue or replay, then every row's readback, in row order. An
    /// error here can follow launches, so the caller passes it through
    /// [`GpuModel::note_fault`].
    fn run_pass<const M: usize>(&mut self, pos: u32) -> Result<[u32; M], GpuError> {
        const WHAT: &str = "GpuModel::step_rows";
        let r = match self.mode {
            StepMode::Eager => {
                let GpuModel {
                    heads,
                    body,
                    weights,
                    gpu,
                    ..
                } = self;
                let heads = heads.get_mut(..M).ok_or(no_head(WHAT))?;
                body.enqueue_rows(gpu, weights, heads)
            }
            StepMode::Graph => self.replay(M, B::CHAIN),
        };
        self.name_host_refusal(r)?;
        self.stand_at(pos + crate::launch_u32(WHAT, "rows", M)?);
        let mut out = [0u32; M];
        let heads = self.heads.get(..M).ok_or(no_head(WHAT))?;
        for (o, head) in out.iter_mut().zip(heads) {
            *o = head.token(&self.gpu)?;
        }
        Ok(out)
    }

    /// Capture the pass of `M` rows into its own graph over the resident
    /// buffers (key `M`), making the heads it needs first, and return its
    /// node count. The one-token step's capture stays.
    pub fn capture_rows<const M: usize>(&mut self) -> Result<usize, GpuError> {
        const WHAT: &str = "GpuModel::capture_rows";
        const { rows_fit::<B>(M) };
        self.make_heads(M, WHAT)?;
        let GpuModel {
            graphs,
            heads,
            body,
            weights,
            gpu,
            ..
        } = self;
        let heads = heads.get_mut(..M).ok_or(no_head(WHAT))?;
        let graph = gpu.capture(|_| body.enqueue_rows(gpu, weights, heads))?;
        let nodes = graph.node_count();
        *graphs.slot(M, WHAT)? = Some(graph);
        Ok(nodes)
    }

    /// Every node of the captured pass of `M` rows, as the driver lists them.
    pub fn rows_graph_nodes<const M: usize>(&self) -> Result<Vec<NodeInfo>, GpuError> {
        const { rows_fit::<B>(M) };
        self.graphs
            .get(M)
            .ok_or(GpuError::state(
                "GpuModel::rows_graph_nodes",
                "no captured pass of these rows",
            ))?
            .nodes()
    }

    /// Each row's logits of the last [`GpuModel::step_rows`] of `M` rows
    /// (`n_vocab` f32 each). Blocking read; gate/debug use.
    pub fn rows_logits<const M: usize>(&self) -> Result<[Vec<f32>; M], GpuError> {
        const WHAT: &str = "GpuModel::rows_logits";
        const { rows_fit::<B>(M) };
        let heads = self
            .heads
            .get(..M)
            .ok_or(GpuError::state(WHAT, "no pass of these rows has run"))?;
        let mut out: [Vec<f32>; M] = std::array::from_fn(|_| Vec::new());
        for (o, head) in out.iter_mut().zip(heads) {
            *o = head.logits_to_host(&self.gpu)?;
        }
        Ok(out)
    }

    /// Row `r`'s head for every `r < m`: row 0's is the load's, the rest are
    /// made here over the resident weights.
    fn make_heads(&mut self, m: usize, what: &'static str) -> Result<(), GpuError> {
        if self.heads.is_empty() {
            return Err(no_head(what));
        }
        while self.heads.len() < m {
            let head = Head::new(&self.gpu, &self.weights, self.body.head_eps())?;
            self.heads.push(head);
        }
        Ok(())
    }
}

/// Holds when a pass of `m` rows is one `B` takes: 2 to [`Rows::MAX_ROWS`],
/// which is at most [`MAX_PASS_ROWS`]. Evaluated at compile time by every
/// method of [`Rows`]'s skeleton.
const fn rows_fit<B: Rows>(m: usize) {
    assert!(
        B::MAX_ROWS <= MAX_PASS_ROWS,
        "a body's Rows::MAX_ROWS is at most MAX_PASS_ROWS"
    );
    assert!(
        m >= 2 && m <= B::MAX_ROWS,
        "a pass of this many rows is not one this body takes (Rows::MAX_ROWS)"
    );
}

impl<B: Rollback> GpuModel<B> {
    /// Take back the positions from `pos` on ([`Rollback::rollback`]): the
    /// next step runs at `pos`.
    pub fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        if pos > self.pos {
            return Err(GpuError::shape(
                "GpuModel::rollback",
                format!("back to position {pos} from {}", self.pos),
            ));
        }
        self.body.rollback(pos)?;
        self.stand_at(pos);
        Ok(())
    }
}

impl<B: Instrumented> GpuModel<B> {
    /// Fill the first `rows` cache rows of every resident layer and stand at
    /// position `rows` — the state a prompt of `rows` tokens leaves behind,
    /// without decoding one. A prepared cache buys the step shape of a deep
    /// prompt for a copy. This is an instrument: the rows are not what the
    /// model would have written, so the tokens that come out are meaningless.
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
        self.body.seed_depth(&self.gpu, rows)?;
        self.stand_at(pos);
        Ok(())
    }
}

impl<B: Probed> GpuModel<B> {
    /// Arm (or disarm) the node-price probe. Like [`GpuModel::set_mode`] this
    /// drops every captured chain, since the probe changes which launches the
    /// body issues. A probe with either lever set makes the chain a timing
    /// instrument: `skip_quant` leaves activation buffers unwritten, so the
    /// tokens that come out are not the model's answer.
    pub fn set_probe(&mut self, probe: StepProbe) -> Result<(), GpuError> {
        self.body.set_probe(probe)?;
        self.graphs.clear();
        Ok(())
    }
}

/// The refusal of a call that needs the output head on a load without it.
fn no_head(what: &'static str) -> GpuError {
    GpuError::state(what, "no output head: this load carries none")
}

/// `block_count` of `file`: the layer count.
pub(crate) fn block_count(file: &Split, what: &'static str) -> Result<usize, GpuError> {
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

impl<B: Instrumented> Engine for GpuModel<B> {
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
