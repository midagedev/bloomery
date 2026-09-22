//! `GpuModel` — the GPU engine that stands beside `forward::step`
//! (docs/gpu-design.md decisions 1 and 7). Weights, KV and scratch live on
//! the device from `load`, split into stages by layer range; `step` enqueues
//! one decode step stage by stage and synchronizes once for the argmax.
//!
//! The chain is layer-indexed at m = 1: an attention half shared by every
//! layer (MLA over the layer's own KV cache) followed by one of two FFN
//! halves — the fused dense FFN for a layer without a router, the routed
//! MoE half (router → six experts through `sel` → shared expert → combine)
//! for a layer with one. Block 0 additionally embeds its token in front.
//! Every per-replay quantity (position, live key count, token id, rope
//! cos/sin cache) lives in a device buffer refreshed before the
//! enqueue/replay, so one captured graph serves every position, and the
//! routed expert ids live in a device buffer the expert kernels read per
//! launch, so one captured graph serves every routing.
//! [`GpuModel::step`] stitches those layers into the whole chain — layer 0
//! embedding its token, every later layer reading the previous layer's
//! output residual, the `head.rs` output head on the last — and returns the
//! argmax of the last token it was given. The chain submits in one of two
//! modes ([`StepMode`]): eager, which enqueues the body per token, or graph,
//! which captures the body once and replays it. Only the argmax readback
//! synchronizes.

mod dispatch;
mod kernels;
mod lookup;
mod probe;
mod scratch;
mod seed;

pub use kernels::StepKernels;
pub(crate) use lookup::f32_gain;
pub(crate) use probe::Bytes;
pub use probe::{Block0Taps, LayerTaps, OpTime, StepProbe};

use crate::head::Head;
use crate::tensor::DeviceTensor;
use crate::weights::Weights;
use crate::{Gpu, GpuError, Graph};
use cuda_core::{CudaStream, DeviceBuffer};
use dispatch::{enqueue_chain, enqueue_layer};
use model::attn::MlaParams;
use probe::{Observer, ProfRec};
use scratch::{LayerNames, LayerScratch, MoeDims, MoeScratch, SP_N_KEYS, SP_POS};
use seed::{seed_cache, seed_pattern};
use std::ops::Range;

// ------------------------------------------------------------------- stage

/// A contiguous range of blocks resident on one device (docs/gpu-design.md
/// decision 7). A stage owns its `Gpu` — context, stream, modules — and its
/// weights, KV rows and scratch once loaded with residency; what crosses a
/// stage boundary is one hidden vector. Two stages may sit on the same
/// card: that is the shape the 2-stage = 1-stage bit-identity gate runs in.
///
/// `residency` is filled by [`GpuModel::load_blocks`]; a stage from
/// [`GpuModel::load_staged`] carries only the metadata (that entry must stay
/// cheap — metadata probes call it).
pub struct Stage {
    gpu: Gpu,
    /// Blocks `layers.start..layers.end` of the model, in order.
    layers: std::ops::Range<usize>,
    /// The captured decode step of this stage, once one is assembled.
    graph: Option<Graph>,
    /// What `graph` recorded: the layer and whether it embeds in front. A
    /// replay names what it expects, so a graph of another layer is an
    /// error, not a silent replay of the wrong chain.
    graph_of: Option<(usize, bool)>,
    /// Weights, KV and scratch — present once loaded with residency.
    residency: Option<Residency>,
}

/// Everything a stage's step touches, allocated at load (decision 4).
struct Residency {
    weights: Weights,
    /// One `[ctx_max, kv_width]` u16 cache per layer, `kvr` row layout.
    kv: Vec<DeviceTensor<u16>>,
    /// The weight names of each layer of the stage, in layer order.
    names: Vec<LayerNames>,
    scratch: LayerScratch,
    step: StepKernels,
}

impl Stage {
    /// The device this stage's layers run on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The model layers this stage carries, as a range of layer indices.
    pub fn layers(&self) -> std::ops::Range<usize> {
        self.layers.clone()
    }

    /// Device bytes held by the residency: weights, KV caches, scratch.
    /// Zero for a metadata-only stage.
    pub fn resident_bytes(&self) -> usize {
        self.residency.as_ref().map_or(0, |r| {
            r.weights.resident_bytes()
                + r.kv.iter().map(|c| c.buf().len() * 2).sum::<usize>()
                + r.scratch.bytes()
        })
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
pub struct GpuModel {
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
    stages: Vec<Stage>,
    mla: MlaParams,
    /// The MoE shapes — `None` when no resident layer routes.
    moe: Option<MoeDims>,
    /// KV rows the resident cache was sized for; `step` refuses to grow it.
    ctx_max: usize,
    mode: StepMode,
    /// The cache row the next `step` token lands in.
    pos: u32,
}

impl GpuModel {
    /// The whole model as one stage (metadata only — a resident stage comes
    /// from [`GpuModel::load_blocks`]).
    pub fn load(gguf: &gguf::Gguf, ctx_max: usize) -> Result<GpuModel, GpuError> {
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
    ) -> Result<GpuModel, GpuError> {
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
        let mla = MlaParams::read(gguf, 0)?;
        Ok(GpuModel {
            stages,
            mla,
            moe: None,
            ctx_max,
            head: None,
            step_graph: None,
            mode: StepMode::Graph,
            pos: 0,
        })
    }

    /// One stage over `layers` WITH residency: every tensor of that range
    /// plus the globals in its kernels' device format, one KV cache per
    /// layer, the m = 1 layer scratch and this file's step module. The
    /// geometry checks below run at load, not mid-step; a range holding a
    /// routed layer also sizes the MoE half's arena.
    pub fn load_blocks(
        gguf: &gguf::Gguf,
        ctx_max: usize,
        layers: Range<usize>,
    ) -> Result<GpuModel, GpuError> {
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
        let mla = MlaParams::read(gguf, 0)?;
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let weights = Weights::load(stream, gguf, layers.clone(), true)?;
        let names: Vec<LayerNames> = layers
            .clone()
            .map(|l| LayerNames::new(&weights, l))
            .collect();
        let moe = match names.iter().find(|n| n.routed) {
            Some(n) => Some(MoeDims::read(gguf, &weights, n)?),
            None => None,
        };
        let scratch = LayerScratch::new(
            gpu.context(),
            stream,
            &weights,
            &mla,
            &names,
            moe.as_ref(),
            ctx_max,
        )?;
        let kv_width = mla.latent + mla.rope_dims;
        let kv = (0..layers.len())
            .map(|_| DeviceTensor::<u16>::zeroed(stream, ctx_max, kv_width))
            .collect::<Result<Vec<_>, _>>()?;
        let step = StepKernels::load(gpu.context())?;
        Ok(GpuModel {
            stages: vec![Stage {
                gpu,
                layers,
                graph: None,
                graph_of: None,
                residency: Some(Residency {
                    weights,
                    kv,
                    names,
                    scratch,
                    step,
                }),
            }],
            mla,
            moe,
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
    /// globals), then the head over those same resident weights.
    pub fn load_full(gguf: &gguf::Gguf, ctx_max: usize) -> Result<GpuModel, GpuError> {
        let n_layers = gguf
            .block_count()
            .ok_or(GpuError::metadata("GpuModel::load_full", "block_count"))?
            as usize;
        let mut m = GpuModel::load_blocks(gguf, ctx_max, 0..n_layers)?;
        let eps = m.mla.eps;
        let head = {
            let (gpu, residency) = m.stage_parts("GpuModel::load_full")?;
            Head::new(gpu, &residency.weights, eps)?
        };
        m.head = Some(head);
        Ok(m)
    }

    /// The model's stages, in layer order.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// The attention geometry read from the model file at load.
    pub fn mla(&self) -> &MlaParams {
        &self.mla
    }

    /// Device bytes of everything this model holds resident: the stage's
    /// weights, caches and scratch, plus the head's scratch when it has one.
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
        let (_, residency) = self.stage_parts("GpuModel::set_probe")?;
        residency.scratch.probe_cfg = probe;
        self.step_graph = None;
        if let Some(stage) = self.stages.first_mut() {
            stage.graph = None;
            stage.graph_of = None;
        }
        Ok(())
    }

    /// Rewind to position 0 with empty caches — the fresh-context state for
    /// the next prompt. The weights, the scratch and any captured chain stay
    /// (they do not depend on the cache contents); every layer's cache is
    /// zeroed, because the flash walks whole key segments and a stale row
    /// inside the last segment of a short run is a real key row, not a
    /// skipped one.
    pub fn reset(&mut self) -> Result<(), GpuError> {
        let Some(r) = self.stages.first().and_then(|s| s.residency.as_ref()) else {
            return Err(GpuError::state(
                "GpuModel::reset",
                "stage carries no residency",
            ));
        };
        let slots = r.kv.len();
        let Some(kv0) = r.kv.first() else {
            return Err(GpuError::state(
                "GpuModel::reset",
                "residency carries no cache slots",
            ));
        };
        let zero_row = vec![0u16; kv0.cols()];
        for slot in 0..slots {
            let (gpu, residency) = self.stage_parts("GpuModel::reset")?;
            seed_cache(gpu, &mut residency.kv[slot], &zero_row, "reset")?;
        }
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
    /// The pattern is deterministic in `rows` alone: the same `rows` gives
    /// the same bytes. Every row differs (a repeated row makes the softmax
    /// uniform, a different path from a real cache) and no value is zero,
    /// inf or NaN by construction.
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
        let (slots, width) = match self.stages.first().and_then(|s| s.residency.as_ref()) {
            Some(r) => match r.kv.first() {
                Some(kv0) => (r.kv.len(), kv0.cols()),
                None => {
                    return Err(GpuError::state(
                        "GpuModel::seed_depth",
                        "residency carries no cache slots",
                    ));
                }
            },
            None => {
                return Err(GpuError::state(
                    "GpuModel::seed_depth",
                    "stage carries no residency",
                ));
            }
        };
        let block = seed_pattern(rows, width);
        for slot in 0..slots {
            let (gpu, residency) = self.stage_parts("GpuModel::seed_depth")?;
            seed_cache(gpu, &mut residency.kv[slot], &block, "seed_depth")?;
        }
        self.pos = rows as u32;
        Ok(())
    }

    /// The step parameters as the DEVICE holds them: `(pos_buf[0],
    /// n_keys_buf[0])`, both written by the last `refresh_params`. The host
    /// `pos` is the row the next token lands in; these are what the launches
    /// actually read, and a gate that asserts a prepared cache stands where a
    /// decoded prompt would needs the device side of that claim.
    pub fn device_step_params(&mut self) -> Result<(u32, u32), GpuError> {
        let (gpu, residency) = self.stage_parts("GpuModel::device_step_params")?;
        let stream = gpu.stream();
        let params = residency.scratch.step_params.to_host_vec(stream)?;
        match (params.get(SP_POS), params.get(SP_N_KEYS)) {
            (Some(p), Some(k)) => Ok((*p, *k)),
            _ => Err(GpuError::state(
                "GpuModel::device_step_params",
                "empty parameter buffer",
            )),
        }
    }

    /// Capture the whole chain — every resident layer plus the head — into
    /// one graph over the resident buffers, and return its node count. One
    /// graph, not a chain of per-layer graphs: the layers differ only in
    /// which weights and which cache they address, all of them frozen at
    /// load, and a single `cuGraphLaunch` is the whole point of the capture
    /// (a chain of 27 launches would pay 27 host submits per token).
    pub fn capture_step(&mut self) -> Result<usize, GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let head = self.head.as_mut().ok_or(GpuError::state(
            "GpuModel::capture_step",
            "no output head — load with load_full",
        ))?;
        let stage = self
            .stages
            .first_mut()
            .ok_or(GpuError::state("GpuModel::capture_step", "no stage"))?;
        let Some(Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        }) = stage.residency.as_mut()
        else {
            return Err(GpuError::state(
                "GpuModel::capture_step",
                "stage carries no residency",
            ));
        };
        let gpu = &stage.gpu;
        let graph = gpu.capture(|_| {
            enqueue_chain(
                gpu,
                step,
                weights,
                names,
                kv,
                scratch,
                &mla,
                moe.as_ref(),
                head,
            )
        })?;
        let nodes = graph.node_count();
        self.step_graph = Some(graph);
        Ok(nodes)
    }

    /// Enqueue the whole chain eagerly on the engine stream. Pure enqueues —
    /// the same body [`GpuModel::capture_step`] records.
    fn enqueue_chain_step(&mut self) -> Result<(), GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let head = self.head.as_mut().ok_or(GpuError::state(
            "GpuModel::step",
            "no output head — load with load_full",
        ))?;
        let stage = self
            .stages
            .first_mut()
            .ok_or(GpuError::state("GpuModel::step", "no stage"))?;
        let Some(Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        }) = stage.residency.as_mut()
        else {
            return Err(GpuError::state(
                "GpuModel::step",
                "stage carries no residency",
            ));
        };
        enqueue_chain(
            &stage.gpu,
            step,
            weights,
            names,
            kv,
            scratch,
            &mla,
            moe.as_ref(),
            head,
        )
    }

    /// Feed `tokens` through the chain one position at a time and return the
    /// argmax of the LAST one — the greedy next token. Each token gets its
    /// own `refresh_params` (outside any capture) and one body; only the
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
                StepMode::Graph => self
                    .step_graph
                    .as_ref()
                    .ok_or(GpuError::state("GpuModel::step", "no captured chain"))?
                    .launch(self.stages[0].gpu.stream())?,
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

    // ------------------------------------------------- assembled layer step

    /// The one resident stage. Every assembled path needs exactly one stage
    /// carrying residency; `what` names the caller in the error.
    fn stage_parts(&mut self, what: &'static str) -> Result<(&mut Gpu, &mut Residency), GpuError> {
        if self.stages.len() != 1 {
            return Err(GpuError::state(
                what,
                "the assembled step needs the single stage of load_blocks",
            ));
        }
        let stage = &mut self.stages[0];
        let Some(residency) = stage.residency.as_mut() else {
            return Err(GpuError::state(
                what,
                "stage carries no residency (load_blocks fills it)",
            ));
        };
        Ok((&mut stage.gpu, residency))
    }

    /// The slot of layer `l` inside the resident range — the index of its KV
    /// cache and of its names.
    fn layer_slot(&self, l: usize, what: &'static str) -> Result<usize, GpuError> {
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

    /// The one stage, requiring it to hold block 0 with residency.
    fn block0_parts(&mut self) -> Result<(&mut Gpu, &mut Residency), GpuError> {
        if self.stages.len() != 1 || self.stages[0].layers.start != 0 {
            return Err(GpuError::state(
                "GpuModel::block0",
                "the assembled block-0 step needs a stage of load_blocks starting \
                 at layer 0",
            ));
        }
        self.stage_parts("GpuModel::block0")
    }

    /// Refresh every per-step device parameter for `(token, pos)`: the rope
    /// cos/sin cache (host YaRN math, one position), the token id, the KV
    /// landing row and the live key count. All four share `step_params`, so
    /// the refresh is one host-to-device copy. Runs before an eager enqueue or
    /// a graph replay; a captured graph reads that buffer at run time, which
    /// is what lets one graph serve every position.
    fn refresh_params(&mut self, token: u32, pos: u32) -> Result<(), GpuError> {
        let mut cs = Vec::new();
        self.mla.rope.cache_into(pos, &mut cs);
        let (gpu, residency) = self.stage_parts("GpuModel::refresh_params")?;
        let stream = gpu.stream();
        let s = &mut residency.scratch;
        s.params_host.clear();
        s.params_host.push(token);
        s.params_host.push(pos);
        s.params_host.push(pos + 1);
        s.params_host.extend(cs.iter().map(|v| v.to_bits()));
        let (params, image) = (&mut s.step_params, &s.params_host);
        params.copy_from_host(stream, image)?;
        s.pos_host = pos;
        Ok(())
    }

    /// Enqueue the block-0 step on the engine stream (eager form). Pure
    /// enqueues — no allocation, no synchronization — so the same body is
    /// what [`GpuModel::capture_block0`] records.
    fn enqueue_step(&mut self) -> Result<(), GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let (gpu, residency) = self.block0_parts()?;
        let Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        } = residency;
        enqueue_layer(
            gpu,
            step,
            weights,
            &names[0],
            &mut kv[0],
            scratch,
            &mla,
            moe.as_ref(),
            true,
            &mut |_, _, _| Ok(()),
        )
    }

    /// Enqueue layer `l`'s step on the engine stream (eager form), the
    /// layer's input residual already in the resident input buffer. Pure
    /// enqueues — the same body [`GpuModel::capture_layer`] records.
    fn enqueue_layer_step(&mut self, l: usize) -> Result<(), GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let slot = self.layer_slot(l, "GpuModel::enqueue_layer_step")?;
        let (gpu, residency) = self.stage_parts("GpuModel::enqueue_layer_step")?;
        let Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        } = residency;
        enqueue_layer(
            gpu,
            step,
            weights,
            &names[slot],
            &mut kv[slot],
            scratch,
            &mla,
            moe.as_ref(),
            false,
            &mut |_, _, _| Ok(()),
        )
    }

    /// Eagerly run block 0's step for `token` at `pos` (`pos + 1` live keys,
    /// rows `0..pos` already in the cache — this call appends row `pos`) and
    /// read every tap back. Synchronizes; gate/debug use.
    pub fn step_block0_taps(&mut self, token: u32, pos: u32) -> Result<Block0Taps, GpuError> {
        self.check_pos(pos, "GpuModel::step_block0_taps")?;
        self.refresh_params(token, pos)?;
        self.enqueue_step()?;
        self.block0_taps()
    }

    /// Read the tap tensors of the last run (eager or replay). Synchronizes
    /// per readback; gate/debug use.
    pub fn block0_taps(&mut self) -> Result<Block0Taps, GpuError> {
        let mla = self.mla.clone();
        let (gpu, residency) = self.block0_parts()?;
        let stream = gpu.stream();
        read_block_taps(stream, &residency.scratch, &mla)
    }

    /// Read layer `l`'s tap tensors of the last run (eager or replay). The
    /// MoE spans come back empty for a layer without a router.
    /// Synchronizes per readback; gate/debug use.
    pub fn layer_taps(&mut self, l: usize) -> Result<LayerTaps, GpuError> {
        let mla = self.mla.clone();
        let slot = self.layer_slot(l, "GpuModel::layer_taps")?;
        let (gpu, residency) = self.stage_parts("GpuModel::layer_taps")?;
        let stream = gpu.stream();
        let routed = residency.names[slot].routed;
        let s = &residency.scratch;
        let a = read_block_taps(stream, s, &mla)?;
        let moe = if routed { s.moe.as_ref() } else { None };
        let moe_rd = |pick: fn(&MoeScratch) -> &DeviceBuffer<f32>| -> Result<Vec<f32>, GpuError> {
            match moe {
                Some(m) => Ok(pick(m).to_host_vec(stream)?),
                None => Ok(Vec::new()),
            }
        };
        Ok(LayerTaps {
            layer: l,
            attn_norm: a.attn_norm,
            q: a.q,
            kv_rope_compressed: a.kv_rope_compressed,
            q_rope: a.q_rope,
            k_rope: a.k_rope,
            kv_compressed: a.kv_compressed,
            kqv_compressed: a.kqv_compressed,
            kqv_out: a.kqv_out,
            ffn_inp: a.ffn_inp,
            ffn_norm: moe_rd(|m| &m.normed)?,
            moe_logits: moe_rd(|m| &m.logits)?,
            moe_ids: match moe {
                Some(m) => m.ids.to_host_vec(stream)?,
                None => Vec::new(),
            },
            moe_weights: moe_rd(|m| &m.weights)?,
            expert_down: moe_rd(|m| &m.down)?,
            ffn_shexp: moe_rd(|m| &m.shexp)?,
            l_out: a.l_out,
        })
    }

    /// Write `x_in` into the resident input buffer — the layer's input
    /// residual, which a lone layer has no embedding in front of to produce.
    /// Synchronizes; never inside a capture.
    pub fn set_layer_input(&mut self, x_in: &[f32]) -> Result<(), GpuError> {
        let (gpu, residency) = self.stage_parts("GpuModel::set_layer_input")?;
        let s = &mut residency.scratch;
        if x_in.len() != s.dims.hidden {
            return Err(GpuError::shape(
                "GpuModel::set_layer_input",
                format!(
                    "{} values, the hidden width is {}",
                    x_in.len(),
                    s.dims.hidden
                ),
            ));
        }
        s.x.copy_from_host(gpu.stream(), x_in)?;
        Ok(())
    }

    /// Eagerly run layer `l`'s step for the input residual `x_in` at `pos`
    /// (`pos + 1` live keys, rows `0..pos` already in that layer's cache —
    /// this call appends row `pos`) and read every tap back. Synchronizes;
    /// gate/debug use.
    pub fn step_layer_taps(
        &mut self,
        l: usize,
        x_in: &[f32],
        pos: u32,
    ) -> Result<LayerTaps, GpuError> {
        self.check_pos(pos, "GpuModel::step_layer_taps")?;
        self.set_layer_input(x_in)?;
        self.refresh_params(0, pos)?;
        self.enqueue_layer_step(l)?;
        self.layer_taps(l)
    }

    /// Capture layer `l`'s step into the stage's graph over the resident
    /// buffers (their addresses freeze — they were allocated at load). The
    /// input residual is read from the resident input buffer and the routed
    /// expert ids from the router's own device buffer, so one graph serves
    /// every input and every routing. Returns the node count.
    pub fn capture_layer(&mut self, l: usize) -> Result<usize, GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let slot = self.layer_slot(l, "GpuModel::capture_layer")?;
        let stage = &mut self.stages[0];
        let Some(Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        }) = stage.residency.as_mut()
        else {
            return Err(GpuError::state(
                "GpuModel::capture_layer",
                "stage carries no residency",
            ));
        };
        let gpu = &stage.gpu;
        let graph = gpu.capture(|_| {
            enqueue_layer(
                gpu,
                step,
                weights,
                &names[slot],
                &mut kv[slot],
                scratch,
                &mla,
                moe.as_ref(),
                false,
                &mut |_, _, _| Ok(()),
            )
        })?;
        let nodes = graph.node_count();
        stage.graph = Some(graph);
        stage.graph_of = Some((l, false));
        Ok(nodes)
    }

    /// Refresh the step parameters for `(x_in, pos)` and replay the captured
    /// layer graph. Synchronizes.
    pub fn replay_layer(&mut self, l: usize, x_in: &[f32], pos: u32) -> Result<(), GpuError> {
        self.check_pos(pos, "GpuModel::replay_layer")?;
        self.layer_slot(l, "GpuModel::replay_layer")?;
        self.set_layer_input(x_in)?;
        self.refresh_params(0, pos)?;
        self.launch_graph((l, false))?;
        self.stages[0].gpu.stream().synchronize()?;
        Ok(())
    }

    /// Write `rows` (whole `kv_width`-wide rows) at the head of layer `l`'s
    /// cache, zeroing the rest — the seeding path the gate uses to give the
    /// step a prefix of oracle rows. Synchronizes; never inside a capture.
    pub fn seed_layer_cache(&mut self, l: usize, rows: &[u16]) -> Result<(), GpuError> {
        let slot = self.layer_slot(l, "GpuModel::seed_layer_cache")?;
        let (gpu, residency) = self.stage_parts("GpuModel::seed_layer_cache")?;
        seed_cache(gpu, &mut residency.kv[slot], rows, "seed_layer_cache")
    }

    /// Capture the block-0 step into the stage's graph over the resident
    /// buffers (their addresses freeze — they were allocated at load).
    /// Returns the node count.
    pub fn capture_block0(&mut self) -> Result<usize, GpuError> {
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        if self.stages.len() != 1 || self.stages[0].layers.start != 0 {
            return Err(GpuError::state(
                "GpuModel::capture_block0",
                "needs a stage of load_blocks starting at layer 0",
            ));
        }
        let stage = &mut self.stages[0];
        let Some(Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        }) = stage.residency.as_mut()
        else {
            return Err(GpuError::state(
                "GpuModel::capture_block0",
                "stage carries no residency",
            ));
        };
        let gpu = &stage.gpu;
        let graph = gpu.capture(|_| {
            enqueue_layer(
                gpu,
                step,
                weights,
                &names[0],
                &mut kv[0],
                scratch,
                &mla,
                moe.as_ref(),
                true,
                &mut |_, _, _| Ok(()),
            )
        })?;
        let nodes = graph.node_count();
        stage.graph = Some(graph);
        stage.graph_of = Some((0, true));
        Ok(nodes)
    }

    /// Enqueue one replay of the captured graph — no parameter refresh, no
    /// synchronization. The timing arm of the gate drives this in a loop.
    pub fn launch_block0_graph(&self) -> Result<(), GpuError> {
        self.launch_graph((0, true))
    }

    /// Enqueue one replay of a captured layer graph ([`GpuModel::capture_layer`])
    /// — no parameter refresh, no synchronization. The profile's layer-replay
    /// arm drives this in a loop, the way [`GpuModel::launch_block0_graph`]
    /// serves block 0: a per-op table taken eagerly cannot say how much of a
    /// row is the launch, and only a replay of the same chain can.
    pub fn launch_layer_graph(&self, l: usize) -> Result<(), GpuError> {
        self.launch_graph((l, false))
    }

    /// Launch the stage's graph, requiring it to be the capture of `want`
    /// (layer, embeds in front).
    fn launch_graph(&self, want: (usize, bool)) -> Result<(), GpuError> {
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

    /// Refresh the step parameters for `(token, pos)` and replay the
    /// captured block-0 graph. Synchronizes.
    pub fn replay_block0(&mut self, token: u32, pos: u32) -> Result<(), GpuError> {
        self.check_pos(pos, "GpuModel::replay_block0")?;
        self.refresh_params(token, pos)?;
        self.launch_block0_graph()?;
        let stream = self.stages[0].gpu.stream();
        stream.synchronize()?;
        Ok(())
    }

    /// [`GpuModel::profile_layer`] of block 0 — the model's first layer,
    /// which embeds its token in front.
    pub fn profile_block0(
        &mut self,
        token: u32,
        pos: u32,
        reps: u32,
    ) -> Result<Vec<OpTime>, GpuError> {
        self.profile_layer(0, token, pos, reps)
    }

    /// Time layer `l`'s step op by op: `reps` measured chain runs (after 20
    /// warm-up runs, discarded), each enqueued eagerly at `(token, pos)` with
    /// an observer that synchronizes after every op and records wall time
    /// since the previous sync. Each sample is therefore an eager launch,
    /// body and one synchronize — it overstates every op by the same host
    /// sync cost, which the caller calibrates against a bare launch+sync
    /// constant. Layer 0 embeds `token` in front; every other layer reads
    /// the input residual the caller left in the resident input buffer
    /// ([`GpuModel::set_layer_input`]) and ignores `token`. Parameters are
    /// refreshed once before the loop; the chain is idempotent at a fixed
    /// `(token, pos)` (it rewrites the same KV row and every scratch buffer
    /// it reads), so the runs leave the model exactly where one eager step
    /// would. Errs if the observed op count differs from the node count of a
    /// capture of the same chain — one tick per launch is what makes a row's
    /// time that row's op. A routed layer additionally reads its expert ids
    /// back afterwards and errs unless they are distinct: the expert ops'
    /// byte counts are `n_used` whole expert blocks, which is the traffic
    /// only when no slot repeats. Debug/profiling use — never inside a
    /// capture (the observer synchronizes).
    pub fn profile_layer(
        &mut self,
        l: usize,
        token: u32,
        pos: u32,
        reps: u32,
    ) -> Result<Vec<OpTime>, GpuError> {
        self.check_pos(pos, "GpuModel::profile_layer")?;
        if reps == 0 {
            return Err(GpuError::shape("profile_layer", "reps must be >= 1"));
        }
        let slot = self.layer_slot(l, "GpuModel::profile_layer")?;
        let embed = l == 0;
        self.refresh_params(token, pos)?;
        let mla = self.mla.clone();
        let moe = self.moe.clone();
        let (gpu, residency) = self.stage_parts("GpuModel::profile_layer")?;
        let gpu: &Gpu = gpu;
        let stream = gpu.stream();
        let Residency {
            weights,
            kv,
            names,
            scratch,
            step,
        } = residency;
        let routed = names[slot].routed;
        // One run of layer `l`'s chain under the given observer: the rep
        // loop times it, the node-count check captures it.
        let mut run = |obs: &mut Observer<'_>| {
            enqueue_layer(
                gpu,
                step,
                weights,
                &names[slot],
                &mut kv[slot],
                scratch,
                &mla,
                moe.as_ref(),
                embed,
                obs,
            )
        };
        let rec = profile_reps(stream, reps, &mut run)?;
        stream.synchronize()?;
        check_one_node_per_tick(gpu, rec.ops.len(), &mut run)?;
        if routed {
            check_distinct_ids(scratch.moe.as_ref(), stream, l)?;
        }
        Ok(rec
            .ops
            .into_iter()
            .enumerate()
            .map(|(index, (name, bytes, s))| OpTime {
                index,
                name,
                us_mean: s.iter().sum::<f64>() / s.len() as f64,
                us_min: s.iter().cloned().fold(f64::INFINITY, f64::min),
                bytes,
            })
            .collect())
    }

    /// Mean host wall time (µs) of one `refresh_params` call — the four
    /// host→device parameter copies that sit outside the captured graph but
    /// inside every real decode step. Same 20-run warm-up convention as
    /// [`GpuModel::profile_block0`]; the timed window is the call itself,
    /// not the copies' stream completion.
    pub fn refresh_params_us(&mut self, token: u32, pos: u32, reps: u32) -> Result<f64, GpuError> {
        self.check_pos(pos, "GpuModel::refresh_params_us")?;
        if reps == 0 {
            return Err(GpuError::shape("refresh_params_us", "reps must be >= 1"));
        }
        for _ in 0..20 {
            self.refresh_params(token, pos)?;
        }
        self.stages[0].gpu.stream().synchronize()?;
        let mut total = 0.0f64;
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            self.refresh_params(token, pos)?;
            total += t0.elapsed().as_secs_f64() * 1e6;
        }
        self.stages[0].gpu.stream().synchronize()?;
        Ok(total / f64::from(reps))
    }

    /// Write `rows` (whole `kv_width`-wide rows, `rows.len() <= ctx_max * kv_width`)
    /// at the head of layer 0's cache, zeroing the rest — the seeding path
    /// the gate uses to give the step a prefix of oracle rows. Synchronizes;
    /// never inside a capture.
    pub fn seed_block0_cache(&mut self, rows: &[u16]) -> Result<(), GpuError> {
        let (gpu, residency) = self.block0_parts()?;
        seed_cache(gpu, &mut residency.kv[0], rows, "seed_block0_cache")
    }

    fn check_pos(&self, pos: u32, what: &'static str) -> Result<(), GpuError> {
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

/// [`GpuModel::profile_layer`]'s rep loop: 20 warm-up runs whose samples
/// are discarded, then `reps` measured runs of `run` under the profiling
/// observer.
fn profile_reps(
    stream: &CudaStream,
    reps: u32,
    run: &mut impl FnMut(&mut Observer<'_>) -> Result<(), GpuError>,
) -> Result<ProfRec, GpuError> {
    let mut rec = ProfRec {
        // Slot i is created by op i's first firing (indices arrive in
        // chain order); the name and the byte count are cross-checked on
        // every later rep.
        ops: Vec::new(),
        last: std::time::Instant::now(),
    };
    const WARMUP: u32 = 20;
    for rep in 0..(WARMUP + reps) {
        rec.last = std::time::Instant::now();
        let mut obs = |i: usize, name: &'static str, b: Bytes| rec.observe(i, name, b, stream);
        run(&mut obs)?;
        if rep < WARMUP {
            for (_, _, s) in rec.ops.iter_mut() {
                s.clear();
            }
        }
    }
    Ok(rec)
}

/// Err unless a capture of `run` holds exactly `observed` nodes.
fn check_one_node_per_tick(
    gpu: &Gpu,
    observed: usize,
    run: &mut impl FnMut(&mut Observer<'_>) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    // The observer bills the window between two ticks to one op, so a
    // tick that covered two launches would fold them into one row with
    // no sign of it. Capture the same chain into a throwaway graph (the
    // stage's own capture is untouched) and require one node per tick:
    // the node count is where a folded pair shows.
    let probe = gpu.capture(|_| run(&mut |_, _, _| Ok(())))?;
    if observed != probe.node_count() {
        return Err(GpuError::shape(
            "profile_layer",
            format!(
                "{} ops observed but the same chain captures {} nodes — an op \
             issued more than one launch before its tick, so its neighbours' times are \
             mis-attributed",
                observed,
                probe.node_count()
            ),
        ));
    }
    Ok(())
}

/// Err unless routed layer `l`'s expert ids, read back from `moe`, are
/// distinct: the expert ops' byte counts are `n_used` whole expert blocks,
/// which is the traffic only when no slot repeats.
fn check_distinct_ids(
    moe: Option<&MoeScratch>,
    stream: &CudaStream,
    l: usize,
) -> Result<(), GpuError> {
    let ids = match moe {
        Some(m) => m.ids.to_host_vec(stream)?,
        None => {
            return Err(GpuError::shape(
                "profile_layer",
                format!("layer {l} routes but the stage carries no MoE arena"),
            ));
        }
    };
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != ids.len() {
        return Err(GpuError::shape(
            "profile_layer",
            format!(
                "layer {l} routed to {ids:?} — a repeated slot makes the \
             expert ops' byte counts an overcount of the rows actually read"
            ),
        ));
    }
    Ok(())
}

/// Read the taps every layer's attention half and output share, from the
/// arena `s` of the last run. The flash q rows' rope spans are cut out per
/// head into `q_rope`. Synchronizes per readback.
fn read_block_taps(
    stream: &CudaStream,
    s: &LayerScratch,
    mla: &MlaParams,
) -> Result<Block0Taps, GpuError> {
    let rd = |b: &DeviceBuffer<f32>| -> Result<Vec<f32>, GpuError> { Ok(b.to_host_vec(stream)?) };
    let f_rows = rd(&s.f_rows)?;
    let width = mla.rope_dims + mla.latent;
    let mut q_rope = Vec::with_capacity(mla.n_head * mla.rope_dims);
    for h in 0..mla.n_head {
        q_rope.extend_from_slice(&f_rows[h * width..h * width + mla.rope_dims]);
    }
    let kv_s = rd(&s.kv_s)?;
    let kvr = rd(&s.kvr)?;
    Ok(Block0Taps {
        attn_norm: rd(&s.normed)?,
        q: rd(&s.q)?,
        kv_rope_compressed: rd(&s.kv_a)?,
        q_rope,
        k_rope: kvr[..mla.rope_dims].to_vec(),
        kv_compressed: kv_s[..mla.latent].to_vec(),
        kqv_compressed: rd(&s.kqvc)?,
        kqv_out: rd(&s.attn_out)?,
        ffn_inp: rd(&s.ffn_inp)?,
        l_out: rd(&s.l_out)?,
    })
}
