//! The instruments a gate drives on the qwen3moe chain: one layer run on its
//! own from a given input (the teacher-forced arm), the per-layer copies of
//! the output residual the whole chain leaves when asked, and the cache rows
//! a run wrote.

use super::body::Body;
use super::dispatch;
use super::scratch::KvPlanes;
use crate::GpuError;
use crate::flash_gqa::HEAD;
use crate::model::{GpuModel, StepMode};
use cuda_core::DeviceBuffer;

/// One layer's q8_0 cache planes on the host, rows `0..rows` per head, for
/// a gate to compare bit for bit ([`GpuModel::kv_q8_rows`]): per side the
/// codes words and the scales.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvQ8Host {
    pub kq: Vec<u32>,
    pub kd: Vec<u16>,
    pub vq: Vec<u32>,
    pub vd: Vec<u16>,
}

/// What one layer's run leaves, read back.
pub struct LayerRun {
    /// The attention residual: the FFN half's input.
    pub ffn_inp: Vec<f32>,
    /// The router's ids, in slot order.
    pub ids: Vec<u32>,
    /// The layer's output residual.
    pub l_out: Vec<f32>,
}

impl GpuModel<Body> {
    /// Run layer `l` eagerly from input residual `x_in` at `pos` (rows
    /// `0..pos` already in that layer's planes — this call appends row
    /// `pos`) and read it back. The step's front runs first, for the row's
    /// position and live key count; `x_in` then replaces its embedding row.
    /// Synchronizes; gate use.
    pub fn step_layer(&mut self, l: usize, x_in: &[f32], pos: u32) -> Result<LayerRun, GpuError> {
        let what = "qwen3moe::step_layer";
        self.check_pos(pos, what)?;
        let slot = self.layer_slot(l, what)?;
        self.refresh_params(0, pos)?;
        let (gpu, w, body) = self.body_parts(what)?;
        let stream = gpu.stream();
        if x_in.len() != body.s.x.len() {
            return Err(GpuError::shape(
                what,
                format!(
                    "x_in has {} values, the residual {}",
                    x_in.len(),
                    body.s.x.len()
                ),
            ));
        }
        dispatch::enqueue_front(gpu, w, body)?;
        body.s.x.copy_from_host(stream, x_in)?;
        dispatch::enqueue_layer(gpu, w, body, slot, false, None)?;
        let s = &body.s;
        Ok(LayerRun {
            ffn_inp: s.ffn_inp.to_host_vec(stream)?,
            ids: s.route.ids().to_host_vec(stream)?,
            l_out: s.x.to_host_vec(stream)?,
        })
    }

    /// Run layer `l`'s FFN half alone eagerly from its input residual
    /// `ffn_inp` and read it back (`ffn_inp` echoes the input). Synchronizes;
    /// gate use.
    pub fn step_ffn(&mut self, l: usize, ffn_inp: &[f32]) -> Result<LayerRun, GpuError> {
        let what = "qwen3moe::step_ffn";
        let slot = self.layer_slot(l, what)?;
        let (gpu, w, body) = self.body_parts(what)?;
        let stream = gpu.stream();
        if ffn_inp.len() != body.s.ffn_inp.len() {
            return Err(GpuError::shape(
                what,
                format!(
                    "ffn_inp has {} values, the residual {}",
                    ffn_inp.len(),
                    body.s.ffn_inp.len()
                ),
            ));
        }
        body.s.ffn_inp.copy_from_host(stream, ffn_inp)?;
        dispatch::enqueue_ffn(gpu, w, body, slot)?;
        let s = &body.s;
        Ok(LayerRun {
            ffn_inp: s.ffn_inp.to_host_vec(stream)?,
            ids: s.route.ids().to_host_vec(stream)?,
            l_out: s.x.to_host_vec(stream)?,
        })
    }

    /// Keep (or stop keeping) a copy of every layer's output residual after
    /// each layer of the chain. Switches the model to eager mode: the copies are
    /// read by [`GpuModel::layer_taps`] after an eager step.
    pub fn set_layer_taps(&mut self, on: bool) -> Result<(), GpuError> {
        self.set_mode(StepMode::Eager);
        let (gpu, _, body) = self.body_parts("qwen3moe::set_layer_taps")?;
        body.set_taps(gpu.stream(), on)
    }

    /// Run the chain's attention on the tensor-core flash pass (`mma`, the
    /// pass a load runs) or the scalar one from here on — the other of the
    /// chain's two gated attention arithmetics, a same-class ruler for a
    /// gate, and not an engine path. Eager mode only: a captured graph holds
    /// the pass it was captured with, so in graph mode this is refused; set
    /// the tensor-core pass back before replaying one.
    pub fn set_flash_mma(&mut self, mma: bool) -> Result<(), GpuError> {
        const WHAT: &str = "qwen3moe::set_flash_mma";
        if self.mode() != StepMode::Eager {
            return Err(GpuError::state(
                WHAT,
                "eager mode (the captured graphs hold the flash pass of their capture)",
            ));
        }
        self.body_parts(WHAT)?.2.mma = mma;
        Ok(())
    }

    /// Rows `0..rows` of every layer's K and V planes as f16 bits: per
    /// layer, K then V, each head's `rows` rows of [`HEAD`] values in turn —
    /// refused by name on a q8_0 cache, which holds no f16 rows
    /// ([`GpuModel::kv_q8_rows`]). Synchronizes; gate use.
    pub fn kv_rows(&mut self, rows: usize) -> Result<Vec<Vec<u16>>, GpuError> {
        let what = "qwen3moe::kv_rows";
        let (gpu, _, body) = self.body_parts(what)?;
        let d = body.s.dims;
        if rows > d.ctx {
            return Err(GpuError::shape(
                what,
                format!("{rows} rows of a {}-row cache", d.ctx),
            ));
        }
        let stream = gpu.stream();
        body.kv
            .iter()
            .map(|p| {
                let (k, v) = p.f16(what)?;
                let mut out = Vec::with_capacity(2 * d.n_kv * rows * HEAD);
                for plane in [k, v] {
                    let all = plane.to_host_vec(stream)?;
                    for h in 0..d.n_kv {
                        out.extend_from_slice(&all[h * d.ctx * HEAD..][..rows * HEAD]);
                    }
                }
                Ok(out)
            })
            .collect()
    }

    /// Rows `0..rows` of every layer's q8_0 cache planes, per head the
    /// head's `rows` rows of the two-plane layout (`q8_plane_lens`'s
    /// per-head stride): per layer the codes words and the scales of K then
    /// V — refused by name on an f16 cache ([`GpuModel::kv_rows`]).
    /// Synchronizes; gate use.
    pub fn kv_q8_rows(&mut self, rows: usize) -> Result<Vec<KvQ8Host>, GpuError> {
        let what = "qwen3moe::kv_q8_rows";
        let (gpu, _, body) = self.body_parts(what)?;
        let d = body.s.dims;
        if rows > d.ctx {
            return Err(GpuError::shape(
                what,
                format!("{rows} rows of a {}-row cache", d.ctx),
            ));
        }
        let (words, scales) = crate::rope_neox::q8_plane_lens(d.head, 1, d.ctx);
        let stream = gpu.stream();
        let mut out = Vec::with_capacity(body.kv.len());
        for p in &body.kv {
            let KvPlanes::Q8 { kq, kd, vq, vd } = p else {
                return Err(GpuError::state(
                    what,
                    "the q8_0 planes (this cache runs f16)",
                ));
            };
            let per_head = |codes: &DeviceBuffer<u32>, scl: &DeviceBuffer<u16>| {
                let (all_q, all_d) = (codes.to_host_vec(stream)?, scl.to_host_vec(stream)?);
                let (mut cq, mut cd) = (Vec::new(), Vec::new());
                for h in 0..d.n_kv {
                    cq.extend_from_slice(&all_q[h * words..][..rows * d.head / 4]);
                    cd.extend_from_slice(&all_d[h * scales..][..rows * d.head / 32]);
                }
                Ok::<_, GpuError>((cq, cd))
            };
            let (kq_rows, kd_rows) = per_head(kq, kd)?;
            let (vq_rows, vd_rows) = per_head(vq, vd)?;
            out.push(KvQ8Host {
                kq: kq_rows,
                kd: kd_rows,
                vq: vq_rows,
                vd: vd_rows,
            });
        }
        Ok(out)
    }

    /// Every layer's output residual from the last step, layer by layer.
    pub fn layer_taps(&mut self) -> Result<Vec<Vec<f32>>, GpuError> {
        let what = "qwen3moe::layer_taps";
        let (gpu, _, body) = self.body_parts(what)?;
        let t = body
            .taps
            .as_ref()
            .ok_or(GpuError::state(what, "layer taps are off (set_layer_taps)"))?;
        let all = t.buf.to_host_vec(gpu.stream())?;
        Ok(all
            .chunks(body.s.dims.hidden)
            .map(<[f32]>::to_vec)
            .collect())
    }
}
