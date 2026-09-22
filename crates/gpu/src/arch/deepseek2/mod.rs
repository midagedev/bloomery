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

mod dispatch;
mod names;
mod scratch;
mod seed;
mod taps;

/// deepseek2's MLA geometry as the CPU model reads it from the file.
pub use model::arch::deepseek2::attn::MlaParams;
pub use names::derived_name;
pub use taps::{Block0Taps, LayerTaps};

use crate::model::probe::Observer;
use crate::model::{ChainBody, StepKernels, StepProbe};
use crate::tensor::DeviceTensor;
use crate::weights::{DevWeight, Weights, q8_0_planes};
use crate::{Gpu, GpuError};
use cuda_core::CudaStream;
use model::arch::Arch;
use model::arch::deepseek2::derived::Derived;
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
}

impl Body {
    /// Enqueue the layer in cache slot `slot` (embedding its token in front
    /// when `embed`). Pure enqueues — no allocation, no synchronization — so
    /// this is both the eager body and what a per-layer capture records.
    fn enqueue_layer_at(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        slot: usize,
        embed: bool,
        obs: &mut Observer<'_>,
    ) -> Result<(), GpuError> {
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
            obs,
        )
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
            let derived = Derived::new(gguf)?;
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
        let stream = gpu.stream();
        let mla = MlaParams::read(gguf, 0)?;
        let names: Vec<LayerNames> = layers.clone().map(|l| LayerNames::new(w, l)).collect();
        let moe = match names.iter().find(|n| n.routed) {
            Some(n) => Some(MoeDims::read(gguf, w, n)?),
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
        })
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
        self.kv.iter().map(|c| c.buf().len() * 2).sum::<usize>() + self.scratch.bytes()
    }
}
