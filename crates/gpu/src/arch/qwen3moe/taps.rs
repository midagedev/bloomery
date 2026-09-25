//! The instruments a gate drives on the qwen3moe chain: one layer run on its
//! own from a given input (the teacher-forced arm), the per-layer copies of
//! the output residual the whole chain leaves when asked, and the cache rows
//! a run wrote.

use super::body::Body;
use super::dispatch;
use crate::GpuError;
use crate::flash_gqa::HEAD;
use crate::model::{GpuModel, StepMode};

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
    /// `pos`) and read it back. Synchronizes; gate use.
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
        body.s.x.copy_from_host(stream, x_in)?;
        dispatch::enqueue_layer(gpu, w, body, slot, false, None)?;
        let s = &body.s;
        Ok(LayerRun {
            ffn_inp: s.ffn_inp.to_host_vec(stream)?,
            ids: s.route.ids.to_host_vec(stream)?,
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
            ids: s.route.ids.to_host_vec(stream)?,
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

    /// Rows `0..rows` of every layer's K and V planes as f16 bits: per
    /// layer, K then V, each head's `rows` rows of [`HEAD`] values in turn.
    /// Synchronizes; gate use.
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
                let mut out = Vec::with_capacity(2 * d.n_kv * rows * HEAD);
                for plane in [&p.k, &p.v] {
                    let all = plane.to_host_vec(stream)?;
                    for h in 0..d.n_kv {
                        out.extend_from_slice(&all[h * d.ctx * HEAD..][..rows * HEAD]);
                    }
                }
                Ok(out)
            })
            .collect()
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
