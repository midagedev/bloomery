//! The deepseek2 tap snapshots: host copies of the spans a gate compares
//! against the reference dump, for block 0 and for any layer.

/// Host copies of block 0's tap tensors for one position, in the dump's
/// logical order at that position. The fused FFN exposes neither
/// `ffn_norm-0` nor the down-projection `ffn_out-0` (the norm feeds the
/// quantizer in registers; the down store folds the residual) — `l_out-0`
/// carries that span.
pub struct Block0Taps {
    pub attn_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub kv_rope_compressed: Vec<f32>,
    /// Head-major `q_rope(h)` per head, the dump's `(d, h, t)` at one t.
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub kv_compressed: Vec<f32>,
    /// Head-major `kqv(h)` per head.
    pub kqv_compressed: Vec<f32>,
    pub kqv_out: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub l_out: Vec<f32>,
}

impl Block0Taps {
    /// First tap holding a non-finite value, if any — the gate's finiteness
    /// check over every span, including the ones not compared.
    pub fn non_finite(&self) -> Option<&'static str> {
        let all: [&str; 10] = [
            "attn_norm",
            "q",
            "kv_rope_compressed",
            "q_rope",
            "k_rope",
            "kv_compressed",
            "kqv_compressed",
            "kqv_out",
            "ffn_inp",
            "l_out",
        ];
        let vals: [&[f32]; 10] = [
            &self.attn_norm,
            &self.q,
            &self.kv_rope_compressed,
            &self.q_rope,
            &self.k_rope,
            &self.kv_compressed,
            &self.kqv_compressed,
            &self.kqv_out,
            &self.ffn_inp,
            &self.l_out,
        ];
        all.iter()
            .zip(vals)
            .find(|(_, v)| v.iter().any(|x| !x.is_finite()))
            .map(|(n, _)| *n)
    }

    /// Bit equality of every tap — the gate's rerun/replay checks.
    pub fn bits_equal(&self, other: &Block0Taps) -> bool {
        let eq = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        eq(&self.attn_norm, &other.attn_norm)
            && eq(&self.q, &other.q)
            && eq(&self.kv_rope_compressed, &other.kv_rope_compressed)
            && eq(&self.q_rope, &other.q_rope)
            && eq(&self.k_rope, &other.k_rope)
            && eq(&self.kv_compressed, &other.kv_compressed)
            && eq(&self.kqv_compressed, &other.kqv_compressed)
            && eq(&self.kqv_out, &other.kqv_out)
            && eq(&self.ffn_inp, &other.ffn_inp)
            && eq(&self.l_out, &other.l_out)
    }
}

/// Host copies of one layer's tap tensors for one position, in the dump's
/// logical order at that position. The MoE spans are empty for a layer
/// without a router, and so is `ffn_norm` (the fused dense FFN keeps its
/// normed vector in registers). The routed
/// half's `ffn_moe_out` and `ffn_out` are not here: `moe_combine` folds the
/// weighted sum, the shared expert and the residual into one store, so
/// `l_out` carries that span — `expert_down` and `moe_weights` are the
/// operands a caller can recombine.
pub struct LayerTaps {
    pub layer: usize,
    pub attn_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub kv_rope_compressed: Vec<f32>,
    /// Head-major `q_rope(h)` per head, the dump's `(d, h, t)` at one t.
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub kv_compressed: Vec<f32>,
    /// Head-major `kqv(h)` per head.
    pub kqv_compressed: Vec<f32>,
    pub kqv_out: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub moe_logits: Vec<f32>,
    /// The router's chosen expert ids, rank order — the `sel` the expert
    /// kernels read.
    pub moe_ids: Vec<u32>,
    /// The router weight of each chosen expert, same order.
    pub moe_weights: Vec<f32>,
    /// Each slot's down projection, slot-major (`n_used * hidden`).
    pub expert_down: Vec<f32>,
    pub ffn_shexp: Vec<f32>,
    pub l_out: Vec<f32>,
}

impl LayerTaps {
    /// Every f32 span with its name, in forward order — the finiteness and
    /// bit-equality checks walk this one list so a new tap cannot be added
    /// to the struct and forgotten by the checks.
    fn spans(&self) -> [(&'static str, &[f32]); 14] {
        [
            ("attn_norm", &self.attn_norm),
            ("q", &self.q),
            ("kv_rope_compressed", &self.kv_rope_compressed),
            ("q_rope", &self.q_rope),
            ("k_rope", &self.k_rope),
            ("kv_compressed", &self.kv_compressed),
            ("kqv_compressed", &self.kqv_compressed),
            ("kqv_out", &self.kqv_out),
            ("ffn_inp", &self.ffn_inp),
            ("ffn_norm", &self.ffn_norm),
            ("moe_logits", &self.moe_logits),
            ("moe_weights", &self.moe_weights),
            ("expert_down", &self.expert_down),
            ("ffn_shexp", &self.ffn_shexp),
        ]
    }

    /// First tap holding a non-finite value, if any — the gate's finiteness
    /// check over every span, including the ones not compared.
    pub fn non_finite(&self) -> Option<&'static str> {
        self.spans()
            .into_iter()
            .chain([("l_out", self.l_out.as_slice())])
            .find(|(_, v)| v.iter().any(|x| !x.is_finite()))
            .map(|(n, _)| n)
    }

    /// Bit equality of every tap, the routed ids included — the gate's
    /// rerun/replay checks.
    pub fn bits_equal(&self, other: &LayerTaps) -> bool {
        let eq = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        self.layer == other.layer
            && self.moe_ids == other.moe_ids
            && eq(&self.l_out, &other.l_out)
            && self
                .spans()
                .into_iter()
                .zip(other.spans())
                .all(|((_, a), (_, b))| eq(a, b))
    }
}

// ---------------------------------------------- the deepseek2 instruments
//
// Everything below is `GpuModel<Body>` and only that: eager single-layer
// runs, the per-layer capture/replay the block gates drive, the per-op
// profile and the cache seeding they need. Monomorphic and inherent, not a
// trait, so every gate keeps calling the same method names it always did.

use super::Body;
use super::scratch::{LayerScratch, MoeScratch, SP_N_KEYS, SP_POS};
use crate::model::probe::{Observer, check_one_node_per_tick, profile_reps};
use crate::model::{GpuModel, OpTime};
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::attn::MlaParams;

impl GpuModel<Body> {
    /// The attention geometry read from the model file at load.
    pub fn mla(&self) -> Result<&MlaParams, GpuError> {
        Ok(&self.body("GpuModel::mla")?.mla)
    }

    /// The step parameters as the DEVICE holds them: `(pos_buf[0],
    /// n_keys_buf[0])`, both written by the last parameter refresh. The host
    /// `pos` is the row the next token lands in; these are what the launches
    /// actually read, and a gate that asserts a prepared cache stands where a
    /// decoded prompt would needs the device side of that claim.
    pub fn device_step_params(&mut self) -> Result<(u32, u32), GpuError> {
        let (gpu, _, body) = self.body_parts("GpuModel::device_step_params")?;
        let params = body.scratch.step_params.to_host_vec(gpu.stream())?;
        match (params.get(SP_POS), params.get(SP_N_KEYS)) {
            (Some(p), Some(k)) => Ok((*p, *k)),
            _ => Err(GpuError::state(
                "GpuModel::device_step_params",
                "empty parameter buffer",
            )),
        }
    }

    /// Eagerly run block 0's step for `token` at `pos` (`pos + 1` live keys,
    /// rows `0..pos` already in the cache — this call appends row `pos`) and
    /// read every tap back. Synchronizes; gate/debug use.
    pub fn step_block0_taps(&mut self, token: u32, pos: u32) -> Result<Block0Taps, GpuError> {
        self.check_pos(pos, "GpuModel::step_block0_taps")?;
        self.refresh_params(token, pos)?;
        let (gpu, w, body) = self.block0_parts("GpuModel::block0")?;
        body.enqueue_layer_at(gpu, w, 0, true, &mut |_, _, _| Ok(()))?;
        self.block0_taps()
    }

    /// Read the tap tensors of the last run (eager or replay). Synchronizes
    /// per readback; gate/debug use.
    pub fn block0_taps(&mut self) -> Result<Block0Taps, GpuError> {
        let (gpu, _, body) = self.block0_parts("GpuModel::block0")?;
        read_block_taps(gpu.stream(), &body.scratch, &body.mla)
    }

    /// Read layer `l`'s tap tensors of the last run (eager or replay). The
    /// MoE spans come back empty for a layer without a router.
    /// Synchronizes per readback; gate/debug use.
    pub fn layer_taps(&mut self, l: usize) -> Result<LayerTaps, GpuError> {
        let slot = self.layer_slot(l, "GpuModel::layer_taps")?;
        let (gpu, _, body) = self.body_parts("GpuModel::layer_taps")?;
        let stream = gpu.stream();
        let routed = body.names[slot].routed;
        let s = &body.scratch;
        let a = read_block_taps(stream, s, &body.mla)?;
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
        let (gpu, _, body) = self.body_parts("GpuModel::set_layer_input")?;
        let s = &mut body.scratch;
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
        let slot = self.layer_slot(l, "GpuModel::enqueue_layer_step")?;
        let (gpu, w, body) = self.body_parts("GpuModel::enqueue_layer_step")?;
        body.enqueue_layer_at(gpu, w, slot, false, &mut |_, _, _| Ok(()))?;
        self.layer_taps(l)
    }

    /// Capture layer `l`'s step into the stage's graph over the resident
    /// buffers (their addresses freeze — they were allocated at load). The
    /// input residual is read from the resident input buffer and the routed
    /// expert ids from the router's own device buffer, so one graph serves
    /// every input and every routing. Returns the node count.
    pub fn capture_layer(&mut self, l: usize) -> Result<usize, GpuError> {
        let slot = self.layer_slot(l, "GpuModel::capture_layer")?;
        self.capture_stage((l, false), |gpu, w, body| {
            body.enqueue_layer_at(gpu, w, slot, false, &mut |_, _, _| Ok(()))
        })
    }

    /// Refresh the step parameters for `(x_in, pos)` and replay the captured
    /// layer graph. Synchronizes.
    pub fn replay_layer(&mut self, l: usize, x_in: &[f32], pos: u32) -> Result<(), GpuError> {
        self.check_pos(pos, "GpuModel::replay_layer")?;
        self.layer_slot(l, "GpuModel::replay_layer")?;
        self.set_layer_input(x_in)?;
        self.refresh_params(0, pos)?;
        self.launch_graph((l, false))?;
        self.stage_stream()?.synchronize()?;
        Ok(())
    }

    /// Write `rows` (whole `kv_width`-wide rows) at the head of layer `l`'s
    /// cache, zeroing the rest — the seeding path the gate uses to give the
    /// step a prefix of oracle rows. Synchronizes; never inside a capture.
    pub fn seed_layer_cache(&mut self, l: usize, rows: &[u16]) -> Result<(), GpuError> {
        let slot = self.layer_slot(l, "GpuModel::seed_layer_cache")?;
        let (gpu, _, body) = self.body_parts("GpuModel::seed_layer_cache")?;
        super::seed::seed_cache(gpu, &mut body.kv[slot], rows, "seed_layer_cache")
    }

    /// Write `rows` (whole `kv_width`-wide rows, `rows.len() <= ctx_max * kv_width`)
    /// at the head of layer 0's cache, zeroing the rest — the seeding path
    /// the gate uses to give the step a prefix of oracle rows. Synchronizes;
    /// never inside a capture.
    pub fn seed_block0_cache(&mut self, rows: &[u16]) -> Result<(), GpuError> {
        let (gpu, _, body) = self.block0_parts("GpuModel::block0")?;
        super::seed::seed_cache(gpu, &mut body.kv[0], rows, "seed_block0_cache")
    }

    /// Capture the block-0 step into the stage's graph over the resident
    /// buffers (their addresses freeze — they were allocated at load).
    /// Returns the node count.
    pub fn capture_block0(&mut self) -> Result<usize, GpuError> {
        self.block0_parts("GpuModel::capture_block0")?;
        self.capture_stage((0, true), |gpu, w, body| {
            body.enqueue_layer_at(gpu, w, 0, true, &mut |_, _, _| Ok(()))
        })
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

    /// Refresh the step parameters for `(token, pos)` and replay the
    /// captured block-0 graph. Synchronizes.
    pub fn replay_block0(&mut self, token: u32, pos: u32) -> Result<(), GpuError> {
        self.check_pos(pos, "GpuModel::replay_block0")?;
        self.refresh_params(token, pos)?;
        self.launch_block0_graph()?;
        self.stage_stream()?.synchronize()?;
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
        let (gpu, w, body) = self.body_parts("GpuModel::profile_layer")?;
        let gpu: &Gpu = gpu;
        let stream = gpu.stream();
        let routed = body.names[slot].routed;
        // One run of layer `l`'s chain under the given observer: the rep
        // loop times it, the node-count check captures it.
        let mut run = |obs: &mut Observer<'_>| body.enqueue_layer_at(gpu, w, slot, embed, obs);
        let rec = profile_reps(stream, reps, &mut run)?;
        stream.synchronize()?;
        check_one_node_per_tick(gpu, rec.ops.len(), &mut run)?;
        if routed {
            check_distinct_ids(body.scratch.moe.as_ref(), stream, l)?;
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

    /// Mean host wall time (µs) of one parameter refresh — the host→device
    /// parameter copy that sits outside the captured graph but inside every
    /// real decode step. Same 20-run warm-up convention as
    /// [`GpuModel::profile_block0`]; the timed window is the call itself,
    /// not the copy's stream completion.
    pub fn refresh_params_us(&mut self, token: u32, pos: u32, reps: u32) -> Result<f64, GpuError> {
        self.check_pos(pos, "GpuModel::refresh_params_us")?;
        if reps == 0 {
            return Err(GpuError::shape("refresh_params_us", "reps must be >= 1"));
        }
        for _ in 0..20 {
            self.refresh_params(token, pos)?;
        }
        self.stage_stream()?.synchronize()?;
        let mut total = 0.0f64;
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            self.refresh_params(token, pos)?;
            total += t0.elapsed().as_secs_f64() * 1e6;
        }
        self.stage_stream()?.synchronize()?;
        Ok(total / f64::from(reps))
    }
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
