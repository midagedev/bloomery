//! The DeepSeek-V2-Lite chain: the weights it derives at load, the layer
//! bodies [`crate::model::GpuModel`] drives, the KV rows they append to, the
//! load-time arena they run over and the taps a gate reads back.
//!
//! One layer is an MLA attention half over that layer's own latent cache
//! followed by one of two FFN halves — the fused dense FFN for a layer without
//! a router, the routed MoE half (router → six experts through `sel` → shared
//! expert → combine) for a layer with one. Which half a layer takes is read
//! from the file at load, not from its index: a layer routes exactly when its
//! router weight is resident. Block 0 additionally embeds its token in front.
//!
//! Every per-replay quantity — position, live key count, token id, rope
//! cos/sin cache — lives in a device buffer refreshed before the enqueue or
//! replay, so one captured graph serves every position, and the routed expert
//! ids live in a device buffer the expert kernels read per launch, so one
//! captured graph serves every routing.
//!
//! A hybrid load (crate::hybrid) keeps experts `[0, n_l)` of every routed
//! layer on the card and hands the rest to a host tier; its routed layers run
//! the hybrid MoE half, and nothing else in the chain changes.

mod dispatch;
mod names;
mod scratch;
mod seed;
mod taps;

/// deepseek2's MLA geometry as the CPU model reads it from the file.
pub use model::arch::deepseek2::attn::MlaParams;
pub use names::derived_name;
pub use taps::{Block0Taps, LayerTaps};

use crate::hybrid::{Boundary, BoundaryShape, ExpertPlans, Hybrid, HybridConfig, HybridStats};
use crate::model::probe::Observer;
use crate::model::{ChainBody, GpuModel, StepKernels, StepProbe};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights, q8_0_planes};
use crate::{Gpu, GpuError};
use cuda_core::CudaStream;
use gguf::Split;
use model::arch::Arch;
use model::arch::deepseek2::derived::Derived;
use model::moe::MoeBlockPlan;
use model::placement::{
    Card, CardFormat, CardTotals, Device, Format, Host, HostTotals, Machine, ModelTensor,
    ModelTensors, Plan, Role, Row, Segment,
};
use names::LayerNames;
use scratch::{LayerScratch, MoeDims};
use seed::{seed_cache, seed_pattern};
use std::num::NonZeroU64;
use std::ops::Range;

/// deepseek2's per-replay host values: the token the chain embeds and the
/// cache row it lands in. Everything else the captured graph reads — the rope
/// cos/sin table, the live key count — is a function of those two.
pub struct DecodeInput {
    token: u32,
    pos: u32,
}

/// Everything deepseek2's chain owns on the device: the attention and MoE
/// geometry read from the file at load, one latent KV cache per resident
/// layer, the weight names each layer looks up, the m = 1 arena every layer
/// shares and this architecture's own step module.
pub struct Body {
    mla: MlaParams,
    /// The MoE shapes — `None` when no resident layer routes.
    moe: Option<MoeDims>,
    /// One `[ctx_max, kv_width]` u16 cache per resident layer, `kvr` row
    /// layout.
    kv: Vec<DeviceTensor<u16>>,
    /// The weight names of each layer of the stage, in layer order.
    names: Vec<LayerNames>,
    scratch: LayerScratch,
    step: StepKernels,
    /// The hybrid boundary and its host tier — `None` on an all-card load.
    hybrid: Option<Hybrid<Derived>>,
}

/// The CPU engine's plan of every routed block is what the host tier
/// computes from.
impl ExpertPlans for Derived {
    fn experts(&self, l: usize) -> Option<&MoeBlockPlan> {
        self.block_plan(l).ok()?.moe().ok()
    }
}

impl Body {
    /// Enqueue the layer in cache slot `slot` (embedding its token in front
    /// when `embed`). Pure enqueues — no allocation, no synchronization — so
    /// this is both the eager body and what a per-layer capture records. A
    /// hybrid routed layer is refused: it needs its host service, which only
    /// the whole-chain step and [`GpuModel::step_layer_hybrid`] give it.
    fn enqueue_layer_at(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        slot: usize,
        embed: bool,
        obs: &mut Observer<'_>,
    ) -> Result<(), GpuError> {
        if self.hybrid.is_some() && self.names.get(slot).is_some_and(|n| n.routed) {
            return Err(GpuError::state(
                "Body::enqueue_layer_at",
                "a hybrid MoE layer runs only in the whole-chain step or step_layer_hybrid",
            ));
        }
        dispatch::enqueue_layer(
            gpu,
            &self.step,
            w,
            &self.names[slot],
            &mut self.kv[slot],
            &mut self.scratch,
            &self.mla,
            self.moe.as_ref(),
            embed,
            None,
            obs,
        )
    }

    /// Everything [`ChainBody::load`] builds over the resident weights, with
    /// the routed stacks holding `resident_experts` experts each (`None`:
    /// all of them). No hybrid tier yet.
    fn assemble(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
        w: &Weights,
        layers: Range<usize>,
        ctx_max: usize,
        resident_experts: Option<usize>,
    ) -> Result<Body, GpuError> {
        let stream = gpu.stream();
        let mla = MlaParams::read(gguf, 0)?;
        let names: Vec<LayerNames> = layers.clone().map(|l| LayerNames::new(w, l)).collect();
        let moe = match names.iter().find(|n| n.routed) {
            Some(n) => Some(MoeDims::read(gguf, w, n, resident_experts)?),
            None => None,
        };
        let scratch = LayerScratch::new(
            gpu.context(),
            stream,
            w,
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
        Ok(Body {
            mla,
            moe,
            kv,
            names,
            scratch,
            step,
            hybrid: None,
        })
    }

    /// Enqueue the hybrid routed layer in slot `slot` on its own and serve
    /// its host share — the single-layer run [`GpuModel::step_layer_hybrid`]
    /// reads back.
    fn enqueue_hybrid_layer(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        slot: usize,
    ) -> Result<(), GpuError> {
        let what = "Body::enqueue_hybrid_layer";
        let h = self
            .hybrid
            .as_mut()
            .ok_or(GpuError::state(what, "not a hybrid load"))?;
        let names = self
            .names
            .get(slot)
            .filter(|n| n.routed)
            .ok_or(GpuError::state(what, "the slot holds no routed layer"))?;
        h.begin_chain(gpu.stream())?;
        dispatch::enqueue_layer(
            gpu,
            &self.step,
            w,
            names,
            &mut self.kv[slot],
            &mut self.scratch,
            &self.mla,
            self.moe.as_ref(),
            false,
            Some(&mut h.boundary),
            &mut |_, _, _| Ok(()),
        )?;
        h.layer_enqueued(names.layer)
    }

    /// The width of one cache row, and the caches themselves — the shape the
    /// seeding paths write whole rows of.
    fn cache_cols(&self, what: &'static str) -> Result<usize, GpuError> {
        self.kv
            .first()
            .map(DeviceTensor::cols)
            .ok_or(GpuError::state(what, "residency carries no cache slots"))
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;

    fn arch() -> Arch {
        Arch::Deepseek2
    }

    /// The derived q_nope2 weights of every block in `layers`: `wk_b`
    /// requantized to Q8_0 by the CPU crate's [`Derived`] (never re-derived
    /// here), in the q8f32 two-plane layout over `rows = n_head · latent`
    /// rows of `k = nope` (`qs` rows × k/4 words, `d` rows × k/32 scales),
    /// head-major, block (row, b) at `wblocks[row·nope/32 + b]`, each filed
    /// under [`derived_name`]. An empty range derives nothing.
    fn derive(
        stream: &CudaStream,
        gguf: &gguf::Gguf,
        layers: Range<usize>,
        w: &mut Weights,
    ) -> Result<(), GpuError> {
        if !layers.is_empty() {
            derive_blocks(stream, &Derived::new(gguf)?, layers, w)?;
        }
        Ok(())
    }

    fn load(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
        w: &Weights,
        layers: Range<usize>,
        ctx_max: usize,
    ) -> Result<Body, GpuError> {
        Body::assemble(gpu, gguf, w, layers, ctx_max, None)
    }

    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        Ok(DecodeInput { token, pos })
    }

    /// The token id, the KV landing row, the live key count and the rope
    /// cos/sin cache (host YaRN math, one position) all share `step_params`,
    /// so the refresh is one host-to-device copy.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        let DecodeInput { token, pos } = *input;
        let mut cs = Vec::new();
        self.mla.rope.cache_into(pos, &mut cs);
        let s = &mut self.scratch;
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

    fn enqueue_chain(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        head: &mut crate::head::Head,
    ) -> Result<(), GpuError> {
        dispatch::enqueue_chain(
            gpu,
            &self.step,
            w,
            &self.names,
            &mut self.kv,
            &mut self.scratch,
            &self.mla,
            self.moe.as_ref(),
            head,
            self.hybrid.as_mut(),
        )
    }

    /// Every layer's cache is zeroed, because the flash walks whole key
    /// segments and a stale row inside the last segment of a short run is a
    /// real key row, not a skipped one.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let zero_row = vec![0u16; self.cache_cols("GpuModel::reset")?];
        for cache in self.kv.iter_mut() {
            seed_cache(gpu, cache, &zero_row, "reset")?;
        }
        Ok(())
    }

    /// The pattern is deterministic in `rows` alone: the same `rows` gives the
    /// same bytes. Every row differs (a repeated row makes the softmax
    /// uniform, a different path from a real cache) and no value is zero, inf
    /// or NaN by construction.
    fn seed_depth(&mut self, gpu: &Gpu, rows: usize) -> Result<(), GpuError> {
        let block = seed_pattern(rows, self.cache_cols("GpuModel::seed_depth")?);
        for cache in self.kv.iter_mut() {
            seed_cache(gpu, cache, &block, "seed_depth")?;
        }
        Ok(())
    }

    fn set_probe(&mut self, probe: StepProbe) -> Result<(), GpuError> {
        self.scratch.probe_cfg = probe;
        Ok(())
    }

    /// One architecture-wide rms epsilon serves every norm, the head's
    /// included.
    fn head_eps(&self) -> f32 {
        self.mla.eps
    }

    fn resident_bytes(&self) -> usize {
        self.kv.iter().map(|c| c.buf().len() * 2).sum::<usize>()
            + self.scratch.bytes()
            + self
                .hybrid
                .as_ref()
                .map_or(0, |h| h.boundary.device_bytes())
    }

    fn hybrid_weights(
        stream: &CudaStream,
        file: &Split,
        layers: Range<usize>,
        n_l: usize,
    ) -> Result<Weights, GpuError> {
        let model = hybrid_tensors(file, &layers)?;
        let rows = model
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| hybrid_row(i, t, model.experts, n_l))
            .collect::<Result<Vec<_>, _>>()?;
        let machine = Machine {
            cards: vec![Card {
                name: "hybrid load".to_string(),
                usable_bytes: 0,
                context_bytes: 0,
                scratch_bytes: 0,
                margin_bytes: 0,
                granule_bytes: NonZeroU64::MIN,
                layers: layers.clone(),
                head: true,
            }],
            host: Host {
                usable_bytes: 0,
                reserves: Vec::new(),
            },
        };
        let n_l_u64 = u64::try_from(n_l)
            .map_err(|_| GpuError::shape("Body::hybrid_weights", "n_l passes u64"))?;
        let plan = Plan {
            model: &model,
            machine: &machine,
            ctx_max: 0,
            rows,
            cards: vec![CardTotals {
                dense_bytes: 0,
                expert_bytes: 0,
                rounding_bytes: 0,
                experts: 0,
                kv_bytes: 0,
                scratch_bytes: 0,
                context_bytes: 0,
                headroom_bytes: 0,
            }],
            host: HostTotals {
                expert_bytes: 0,
                experts: 0,
                table_bytes: 0,
                reserve_bytes: 0,
                headroom_bytes: 0,
            },
            nvme_bytes: 0,
            n_l: vec![n_l_u64; model.layers],
        };
        Weights::load_placed(stream, file, &plan, 0)
    }

    /// One `Derived` serves both the derived weights and the host tier's
    /// expert plans.
    fn load_hybrid(
        gpu: &Gpu,
        file: Split,
        w: &mut Weights,
        layers: Range<usize>,
        ctx_max: usize,
        cfg: HybridConfig,
    ) -> Result<Body, GpuError> {
        let what = "Body::load_hybrid";
        let gguf = file
            .shard(0)
            .ok_or(GpuError::state(what, "the file has no shard 0"))?;
        let derived = Derived::new(gguf)?;
        derive_blocks(gpu.stream(), &derived, layers.clone(), w)?;
        let mut body = Body::assemble(gpu, gguf, w, layers.clone(), ctx_max, Some(cfg.n_l))?;
        let n_used = body
            .moe
            .as_ref()
            .ok_or(GpuError::state(
                what,
                "no resident layer routes: nothing to split",
            ))?
            .n_used;
        let shape = BoundaryShape {
            hidden: body.scratch.dims.hidden,
            n_used,
        };
        let boundary = Boundary::new(gpu.context(), gpu.stream(), shape, cfg)?;
        body.hybrid = Some(Hybrid::new(boundary, derived, file, layers.len())?);
        Ok(body)
    }

    fn serve_replay(&mut self) -> Result<(), GpuError> {
        match self.hybrid.as_mut() {
            Some(h) => h.serve_captured(),
            None => Ok(()),
        }
    }
}

/// The derived q_nope2 weights of every block in `layers`, from `derived`,
/// filed into `w` (see [`ChainBody::derive`]).
fn derive_blocks(
    stream: &CudaStream,
    derived: &Derived,
    layers: Range<usize>,
    w: &mut Weights,
) -> Result<(), GpuError> {
    for l in layers {
        let params = &derived.block_plan(l)?.attn.params;
        let (rows, k) = (params.n_head * params.latent, params.nope);
        let blocks = derived.wk_b_all_heads(l)?;
        // The resident shape is the q8f32 kernel's: rows = n_head·
        // latent rows of k = nope. The block count must be exactly
        // rows·k/32 — a disagreement is a load error naming the
        // geometry, never a silent reshape.
        let want = rows * (k / 32);
        if blocks.len() != want {
            return Err(GpuError::shape(
                "Body::derive",
                format!(
                    "block {l}: {} q8 blocks, want rows·k/32 = {rows}·{k}/32 = {want}",
                    blocks.len()
                ),
            ));
        }
        let (qs, d) = q8_0_planes(blocks);
        w.insert_derived(
            derived_name(l),
            DevWeight::Q8_0Derived {
                qs: DeviceTensor::upload(stream, &qs, rows, k / 4)?,
                d: DeviceTensor::upload(stream, &d, rows, k / 32)?,
                k,
            },
        )?;
    }
    Ok(())
}

// ------------------------------------------------------ the hybrid load plan
//
// A load plan, not a fit: `Weights::load_placed` reads its rows and segments
// and checks each upload against the segment's `resident_bytes`; the roles are
// the file's, and the totals and budgets a planner would fill stay zero.

/// Every tensor of `file` in a block of `layers` or in no block, with its
/// role, and the model's layer and expert counts.
fn hybrid_tensors(file: &Split, layers: &Range<usize>) -> Result<ModelTensors, GpuError> {
    let what = "Body::hybrid_weights";
    let g = file
        .shard(0)
        .ok_or(GpuError::state(what, "the file has no shard 0"))?;
    let count = |v: Option<u64>, key: &'static str| -> Result<u64, GpuError> {
        v.ok_or(GpuError::metadata(what, key))
    };
    let n_layers = count(g.block_count(), "block_count")?;
    let experts = count(g.expert_count(), "expert_count")?;
    let experts_used = count(g.arch_get_u64("expert_used_count"), "expert_used_count")?;
    let mut tensors = Vec::new();
    for (shard, t) in file.iter_tensors() {
        let layer = block_of(&t.name);
        if layer.is_some_and(|l| !layers.contains(&l)) {
            continue;
        }
        tensors.push(ModelTensor {
            name: t.name.clone(),
            shard,
            layer,
            role: role_of(&t.name),
            ty: t.ty,
            dims: t.dims.clone(),
            file_bytes: t.nbytes,
            gathered_rows: (role_of(&t.name) == Role::TokenEmbedding).then_some(1),
        });
    }
    Ok(ModelTensors {
        tensors,
        layers: usize::try_from(n_layers)
            .map_err(|_| GpuError::shape(what, format!("block_count {n_layers}")))?,
        experts,
        experts_used,
    })
}

/// Tensor `t`'s placement: whole on card 0 in its card format, or — a
/// routed stack — experts `[0, n_l)` on the card and the rest in the host
/// file.
fn hybrid_row(i: usize, t: &ModelTensor, experts: u64, n_l: usize) -> Result<Row, GpuError> {
    let what = "Body::hybrid_weights";
    let format = CardFormat::of(t.ty).ok_or_else(|| {
        GpuError::shape(
            what,
            format!("tensor {} has type {} with no card format", t.name, t.ty),
        )
    })?;
    let rows: u64 = t.dims[1..].iter().product();
    let card = |rows: u64| -> Result<u64, GpuError> {
        format.resident_bytes(t.ty, t.dims[0], rows).ok_or_else(|| {
            GpuError::shape(
                what,
                format!("tensor {}: {rows} rows have no card layout", t.name),
            )
        })
    };
    let n_l = u64::try_from(n_l).map_err(|_| GpuError::shape(what, "n_l passes u64"))?;
    let segments = if t.role == Role::RoutedExperts {
        if t.dims.len() != 3 || t.dims[2] != experts || n_l > experts {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} {:?} is not a stack of {experts} experts to split at {n_l}",
                    t.name, t.dims
                ),
            ));
        }
        let per = rows / experts;
        let mut segments = Vec::with_capacity(2);
        if n_l > 0 {
            segments.push(Segment {
                device: Device::Card(0),
                format: Format::Card(format),
                experts: Some(0..n_l),
                resident_bytes: card(n_l * per)?,
            });
        }
        if n_l < experts {
            segments.push(Segment {
                device: Device::Host,
                format: Format::HostFile,
                experts: Some(n_l..experts),
                resident_bytes: t.file_bytes / experts * (experts - n_l),
            });
        }
        segments
    } else {
        vec![Segment {
            device: Device::Card(0),
            format: Format::Card(format),
            experts: None,
            resident_bytes: card(rows)?,
        }]
    };
    Ok(Row {
        tensor: i,
        segments,
        read_bytes: 0,
        stage: 0,
    })
}

/// `Some(L)` for a `blk.L.*` name.
fn block_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// A deepseek2 tensor's role by name. The block without a router carries a
/// dense FFN, which takes the shared expert's role: both are read whole by
/// every token, which is all a load plan asks of a role.
fn role_of(name: &str) -> Role {
    let Some((_, stem)) = name.strip_prefix("blk.").and_then(|r| r.split_once('.')) else {
        return if name.starts_with("token_embd") {
            Role::TokenEmbedding
        } else {
            Role::Head
        };
    };
    if stem.contains("_exps") {
        Role::RoutedExperts
    } else if stem.starts_with("ffn_gate_inp") {
        Role::Router
    } else if stem.starts_with("ffn_norm") {
        Role::FfnNorm
    } else if stem.starts_with("ffn_") {
        Role::SharedExpert
    } else {
        Role::Attention
    }
}

// --------------------------------------------------- the hybrid instruments

/// What one hybrid layer's run leaves, read back for a gate.
pub struct HybridTaps {
    /// The MoE half's input residual.
    pub ffn_inp: Vec<f32>,
    /// The routing as the router wrote it into the handoff.
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
    /// The card's expert outputs by slot, `n_used × hidden`: a host slot's
    /// are the zeros the layer wrote there.
    pub down: Vec<f32>,
    /// The shared expert's output.
    pub shexp: Vec<f32>,
    /// The host experts' weighted sum, as the host wrote it.
    pub hsum: Vec<f32>,
    /// The layer's output residual.
    pub l_out: Vec<f32>,
}

impl GpuModel<Body> {
    /// Eagerly run hybrid routed layer `l` for the input residual `x_in` at
    /// `pos` (rows `0..pos` already in that layer's cache — this call appends
    /// row `pos`), serve its host share, and read the layer back.
    /// Synchronizes; gate/debug use.
    pub fn step_layer_hybrid(
        &mut self,
        l: usize,
        x_in: &[f32],
        pos: u32,
    ) -> Result<HybridTaps, GpuError> {
        let what = "GpuModel::step_layer_hybrid";
        self.check_pos(pos, what)?;
        self.set_layer_input(x_in)?;
        self.refresh_params(0, pos)?;
        let slot = self.layer_slot(l, what)?;
        let (gpu, w, body) = self.body_parts(what)?;
        body.enqueue_hybrid_layer(gpu, w, slot)?;
        let stream = gpu.stream();
        let h = body
            .hybrid
            .as_ref()
            .ok_or(GpuError::state(what, "not a hybrid load"))?;
        let s = &body.scratch;
        let m = s
            .moe
            .as_ref()
            .ok_or(GpuError::state(what, "no MoE arena"))?;
        Ok(HybridTaps {
            ffn_inp: s.ffn_inp.to_host_vec(stream)?,
            ids: h.boundary.ids.to_host_vec(stream)?,
            weights: h.boundary.weights.to_host_vec(stream)?,
            down: m.down.to_host_vec(stream)?,
            shexp: m.shexp.to_host_vec(stream)?,
            hsum: h.hsum_copy()?,
            l_out: s.l_out.to_host_vec(stream)?,
        })
    }

    /// The host tier's counters, on a hybrid load.
    pub fn hybrid_stats(&self) -> Option<HybridStats> {
        self.body("GpuModel::hybrid_stats")
            .ok()?
            .hybrid
            .as_ref()
            .map(Hybrid::stats)
    }
}
