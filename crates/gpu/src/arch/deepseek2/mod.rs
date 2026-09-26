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
mod pins;
mod scratch;
mod seed;
mod taps;

/// deepseek2's MLA geometry as the CPU model reads it from the file.
pub use model::arch::deepseek2::attn::MlaParams;
pub use names::derived_name;
pub use taps::{Block0Taps, LayerTaps};

use crate::hybrid::{
    Boundary, BoundaryShape, HostExperts, Hybrid, HybridConfig, HybridStats, HybridWords, Refusal,
    SlotMap,
};
use crate::model::probe::Observer;
use crate::model::{ChainBody, GpuModel, StepKernels, StepProbe, one_shard};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights, q8_0_planes};
use crate::{Gpu, GpuError};
use cuda_core::CudaStream;
use gguf::Split;
use model::Tensor2;
use model::arch::Arch;
use model::arch::deepseek2::derived::Derived;
use model::arch::deepseek2::hparams::Hparams;
use model::moe::{HostScratch, experts_into};
use model::placement::{self, ExpertList, ModelTensor, ModelTensors, Role, Row};
use names::LayerNames;
use scratch::{LayerScratch, MoeDims};
use seed::{seed_cache, seed_pattern};
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
    hybrid: Option<Hybrid<PlanHost>>,
}

/// deepseek2's host tier: the CPU engine's plan of every routed block over
/// the one-shard file, and the scratch its calls write, made at load for the
/// layers' widths.
pub(crate) struct PlanHost {
    plans: Derived,
    file: Split,
    scratch: HostScratch,
}

impl PlanHost {
    /// The tier for experts that map `hidden -> ff -> hidden`. Load-time
    /// only.
    fn new(plans: Derived, file: Split, hidden: usize, ff: usize) -> Result<PlanHost, GpuError> {
        if file.shard(0).is_none() {
            return Err(GpuError::state("PlanHost::new", "the file has no shard 0"));
        }
        Ok(PlanHost {
            plans,
            file,
            scratch: HostScratch::new(hidden, ff),
        })
    }
}

impl HostExperts for PlanHost {
    fn experts_into(
        &mut self,
        layer: usize,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let what = "Hybrid::serve";
        let gguf = self
            .file
            .shard(0)
            .ok_or(GpuError::state(what, "the file has no shard 0"))?;
        let plan = self
            .plans
            .block_plan(layer)
            .ok()
            .and_then(|b| b.moe().ok())
            .ok_or(GpuError::state(
                what,
                "a hybrid layer without an expert plan",
            ))?;
        experts_into(gguf, plan, x, experts, out, &mut self.scratch)?;
        Ok(())
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
    /// all of them). No hybrid tier yet. The file's hyperparameters are read
    /// here, once per load, and every value the kernels were not built for is
    /// refused before anything is allocated.
    fn assemble(
        gpu: &Gpu,
        gguf: &gguf::Gguf,
        w: &Weights,
        layers: Range<usize>,
        ctx_max: usize,
        resident_experts: Option<usize>,
    ) -> Result<Body, GpuError> {
        let stream = gpu.stream();
        let hp = Hparams::read(gguf).map_err(|e| GpuError::plan("Body::load", e))?;
        let mla = MlaParams::read(gguf, 0)?;
        pins::attention(&hp, mla.latent, &|s| gguf.arch_key(s))?;
        let names: Vec<LayerNames> = layers.clone().map(|l| LayerNames::new(w, l)).collect();
        let moe = MoeDims::read(gguf, &hp, w, &names, resident_experts)?;
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
    type Meta = ();

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
        file: &Split,
        layers: Range<usize>,
        w: &mut Weights,
    ) -> Result<(), GpuError> {
        if !layers.is_empty() {
            let gguf = one_shard(file, "Body::derive")?;
            derive_blocks(stream, &Derived::new(gguf)?, layers, w)?;
        }
        Ok(())
    }

    fn load(
        gpu: &Gpu,
        file: &Split,
        w: &Weights,
        layers: Range<usize>,
        ctx_max: usize,
    ) -> Result<Body, GpuError> {
        let gguf = one_shard(file, "Body::load")?;
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
    /// real key row, not a skipped one. The host tier's reset comes first
    /// ([`Hybrid::reset`]): a poison it cannot lift fails the reset before
    /// anything is zeroed. On a hybrid load the routed experts' gate/up rows
    /// are zeroed too: a hybrid step quantizes every slot's row and writes
    /// only the card's, so a host slot's row keeps what an earlier step left
    /// there — NaN, after a step that faulted.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if let Some(h) = self.hybrid.as_mut() {
            h.reset(gpu.stream())?;
            if let Some(m) = self.scratch.moe.as_mut() {
                m.h_exp.zero_async(gpu.stream())?;
            }
        }
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
        let n_l = u64::try_from(n_l)
            .map_err(|_| GpuError::shape("Body::hybrid_weights", "n_l passes u64"))?;
        let rows = model
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| hybrid_row(i, t, &model, n_l))
            .collect::<Result<Vec<_>, _>>()?;
        Weights::load_rows(stream, file, &model, &rows, 0)
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
        let gguf = one_shard(&file, what)?;
        let derived = Derived::new(gguf)?;
        derive_blocks(gpu.stream(), &derived, layers.clone(), w)?;
        let mut body = Body::assemble(gpu, gguf, w, layers.clone(), ctx_max, Some(cfg.n_l))?;
        let (n_used, n_expert, ff) = body
            .moe
            .as_ref()
            .map(|m| (m.n_used, m.n_expert, m.ff))
            .ok_or(GpuError::state(
                what,
                "no resident layer routes: nothing to split",
            ))?;
        let hidden = body.scratch.dims.hidden;
        let shape = BoundaryShape { hidden, n_used };
        let slots = SlotMap::prefix(layers.clone(), n_expert, cfg.n_l)?;
        let boundary = Boundary::new(gpu.context(), gpu.stream(), shape, slots, cfg.overlap)?;
        let host = PlanHost::new(derived, file, hidden, ff)?;
        body.hybrid = Some(Hybrid::new(boundary, host, layers.len())?);
        Ok(body)
    }

    fn serve_replay(&mut self) -> Result<(), GpuError> {
        match self.hybrid.as_mut() {
            Some(h) => h.serve_captured(),
            None => Ok(()),
        }
    }

    fn take_host_refusal(&mut self) -> Option<Refusal> {
        self.hybrid.as_mut().and_then(Hybrid::take_step_refusal)
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

// ------------------------------------------------------ the hybrid load rows
//
// Load rows, not a fit: `Weights::load_rows` reads their segments and checks
// each upload against the segment's `resident_bytes`; the roles are the
// file's, and no budget is planned.

/// Every tensor of `file` in a block of `layers` or in no block, with its
/// role, and the model's layer and expert counts as [`Hparams`] reads them.
fn hybrid_tensors(file: &Split, layers: &Range<usize>) -> Result<ModelTensors, GpuError> {
    let what = "Body::hybrid_weights";
    let g = file
        .shard(0)
        .ok_or(GpuError::state(what, "the file has no shard 0"))?;
    let hp = Hparams::read(g).map_err(|e| GpuError::plan(what, e))?;
    let as_u64 = |n: usize, key: &str| {
        u64::try_from(n).map_err(|_| GpuError::shape(what, format!("{key} {n} passes u64")))
    };
    let experts = as_u64(hp.experts.n_expert, "expert_count")?;
    let experts_used = as_u64(hp.experts.n_used, "expert_used_count")?;
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
        layers: hp.n_layer,
        experts,
        experts_used,
    })
}

/// Tensor `t`'s load row: whole on card 0 in its card format, or — a routed
/// stack of `model`'s — experts `[0, n_l)` on the card and the rest in the
/// host file (the `_sel` kernels here read an id as its slot, so the card
/// list is the id prefix).
fn hybrid_row(i: usize, t: &ModelTensor, model: &ModelTensors, n_l: u64) -> Result<Row, GpuError> {
    let what = "Body::hybrid_weights";
    if t.role != Role::RoutedExperts {
        return placement::whole_on_card(i, t, 0).map_err(|e| GpuError::plan(what, e));
    }
    if t.dims.len() != 3 || n_l > model.experts {
        return Err(GpuError::shape(
            what,
            format!(
                "{} {:?} is not a stack of {} experts to split at {n_l}",
                t.name, t.dims, model.experts
            ),
        ));
    }
    ExpertList::prefix(n_l)
        .and_then(|on_card| placement::routed_row(i, t, 0, on_card, model))
        .map_err(|e| GpuError::plan(what, e))
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
        let r = body.enqueue_hybrid_layer(gpu, w, slot);
        // A fault printed in this architecture's step order.
        self.name_host_refusal(r).map_err(|e| match e {
            GpuError::Fault {
                what,
                fault,
                behind,
            } => GpuError::Fault {
                what,
                fault: fault.in_arch(Body::arch()),
                behind,
            },
            e => e,
        })?;
        let (gpu, _, body) = self.body_parts(what)?;
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

    /// The host tier's protocol words, on a hybrid load.
    pub fn hybrid_words(&self) -> Option<HybridWords> {
        self.body("GpuModel::hybrid_words")
            .ok()?
            .hybrid
            .as_ref()
            .map(Hybrid::words)
    }
}
