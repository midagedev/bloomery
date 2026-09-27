//! The instruments a gate drives on the Qwen3.6 chain: one layer run on its
//! own at `m` rows from a given input (the teacher-forced arm), its FFN half
//! alone, every intermediate either leaves read back; a layer's store read
//! and written (the state a set's prefill left, loaded before its step);
//! the per-layer copies of the output residual the decode chain leaves when
//! asked; and the decode flash pass switched for an eager ruler.

use super::body35::Body35;
use super::dispatch::{self, Ctx};
use super::plan::MixerPlan;
use super::scratch::{LayerStore, f32_view};
use crate::model::{GpuModel, MAX_PASS_ROWS, StepMode};
use crate::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};

/// What an attention layer's mixer leaves for its `m` rows, token-major:
/// the query projection's `[q | gate]` rows (`m · 16 · 512`), the normed and
/// turned queries (`m · 16 · 256`), keys and values (`m · 2 · 256`), and the
/// flash output (`m · 16 · 256`).
pub struct Gqa35Run {
    pub qg: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub fa: Vec<f32>,
}

/// What a delta layer's mixer leaves for its `m` rows, token-major: the
/// q·k·v projection (`m · C`), the gate `z` (`m · n_v · 128`), β's and α's
/// raw projections (`m · n_v`), the conv's output (`m · C`: q normed and
/// scaled, k normed, v), β and decay (`exp(g)`, `m · n_v`), the delta
/// output `o` and the gated norm's output (`m · n_v · 128`).
pub struct Delta35Run {
    pub x: Vec<f32>,
    pub z: Vec<f32>,
    pub b: Vec<f32>,
    pub a: Vec<f32>,
    pub conv: Vec<f32>,
    pub beta: Vec<f32>,
    pub decay: Vec<f32>,
    pub o: Vec<f32>,
    pub y: Vec<f32>,
}

/// A layer's mixer's intermediates, by kind.
pub enum Mixer35Run {
    Attention(Gqa35Run),
    Delta(Delta35Run),
}

/// What the FFN half leaves for its `m` rows: the router's 257 logits a
/// token (the shared gate's last), each token's nine slots' ids and weights
/// (the shared expert's last, its weight the sigmoid of its logit), and the
/// output residual.
pub struct Ffn35Run {
    pub logits: Vec<f32>,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
    pub l_out: Vec<f32>,
}

/// What one layer's run leaves.
pub struct Layer35Run {
    pub mixer: Mixer35Run,
    /// The FFN's input residual: the input plus the mixer's output.
    pub ffn_inp: Vec<f32>,
    pub ffn: Ffn35Run,
}

/// A layer's store on the host, in the card's layouts: the K and V planes
/// (`[n_kv][ctx][256]` f16 bits each), or the recurrent state (every lane,
/// `[lanes][n_v][128 v][128 k]`) and the conv ring (`[RING_ROWS][C]`, the
/// input of position `p` in slot `p mod RING_ROWS`).
pub enum StoreHost {
    Kv { k: Vec<u16>, v: Vec<u16> },
    Rec { state: Vec<f32>, ring: Vec<f32> },
}

/// The first `n` values of `b`, read back.
fn head_of(stream: &CudaStream, b: &DeviceBuffer<f32>, n: usize) -> Result<Vec<f32>, GpuError> {
    let mut v = b.to_host_vec(stream)?;
    v.truncate(n);
    Ok(v)
}

impl Body35 {
    /// Layer `l`'s store, read back. Synchronizes; gate use.
    pub fn store(&self, gpu: &Gpu, l: usize) -> Result<StoreHost, GpuError> {
        let stream = gpu.stream();
        Ok(match self.store_at(l, "qwen35moe::store")? {
            LayerStore::Kv(p) => StoreHost::Kv {
                k: p.k.to_host_vec(stream)?,
                v: p.v.to_host_vec(stream)?,
            },
            LayerStore::Rec(r) => StoreHost::Rec {
                state: r.state.to_host_vec(stream)?,
                ring: r.ring.to_host_vec(stream)?,
            },
        })
    }

    /// Write `h` into layer `l`'s store: every value of a store of its kind
    /// and length, else refused by name. Synchronizes; gate use.
    pub fn set_store(&mut self, gpu: &Gpu, l: usize, h: &StoreHost) -> Result<(), GpuError> {
        const WHAT: &str = "qwen35moe::set_store";
        let stream = gpu.stream();
        let bad = |want: String| GpuError::shape(WHAT, format!("layer {l}: {want}"));
        match (self.store_at_mut(l, WHAT)?, h) {
            (LayerStore::Kv(p), StoreHost::Kv { k, v }) => {
                if k.len() != p.k.len() || v.len() != p.v.len() {
                    return Err(bad(format!(
                        "K/V planes of {} values, given {} and {}",
                        p.k.len(),
                        k.len(),
                        v.len()
                    )));
                }
                p.k.copy_from_host(stream, k)?;
                p.v.copy_from_host(stream, v)?;
            }
            (LayerStore::Rec(r), StoreHost::Rec { state, ring }) => {
                if state.len() != r.state.len() || ring.len() != r.ring.len() {
                    return Err(bad(format!(
                        "a state of {} and a ring of {} values, given {} and {}",
                        r.state.len(),
                        r.ring.len(),
                        state.len(),
                        ring.len()
                    )));
                }
                r.state.copy_from_host(stream, state)?;
                r.ring.copy_from_host(stream, ring)?;
            }
            _ => return Err(bad("a store of the layer's own kind".into())),
        }
        stream.synchronize()?;
        Ok(())
    }

    fn store_at(&self, l: usize, what: &'static str) -> Result<&LayerStore, GpuError> {
        self.stores
            .get(l)
            .ok_or_else(|| GpuError::shape(what, format!("layer {l} of {}", self.stores.len())))
    }

    fn store_at_mut(&mut self, l: usize, what: &'static str) -> Result<&mut LayerStore, GpuError> {
        let n = self.stores.len();
        self.stores
            .get_mut(l)
            .ok_or_else(|| GpuError::shape(what, format!("layer {l} of {n}")))
    }

    /// The model's vocabulary.
    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Values of a residual row.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.s.dims.hidden
    }

    /// Rows of every cache: K/V planes and the rope table.
    #[must_use]
    pub fn ctx_rows(&self) -> usize {
        self.s.dims.ctx
    }

    /// The pass arena's first `m` rows of every intermediate the last layer
    /// run left, read back, and its output residual (`x`).
    fn read_run(&self, gpu: &Gpu, l: usize, m: usize) -> Result<Layer35Run, GpuError> {
        const WHAT: &str = "qwen35moe::read_run";
        let stream = gpu.stream();
        let a = &self.a;
        let d = a.dims;
        let mixer = match &self.plans[l].mixer {
            MixerPlan::Gqa(_) => {
                let q_out = a
                    .q_out
                    .as_ref()
                    .ok_or(GpuError::state(WHAT, "the gated queries' buffer"))?;
                Mixer35Run::Attention(Gqa35Run {
                    qg: head_of(stream, &a.q, m * d.q_rows)?,
                    q: head_of(stream, q_out, m * d.attn_len())?,
                    k: head_of(stream, &a.k, m * d.kv_len())?,
                    v: head_of(stream, &a.v, m * d.kv_len())?,
                    fa: head_of(stream, &a.attn, m * d.attn_len())?,
                })
            }
            MixerPlan::Delta(p) => {
                let g = a
                    .gdn
                    .as_ref()
                    .ok_or(GpuError::state(WHAT, "the delta intermediates"))?;
                let (c, nv) = (p.shape.channels(), p.shape.n_v);
                let zl = nv * crate::linear::HEAD;
                Mixer35Run::Delta(Delta35Run {
                    x: head_of(stream, &g.x, m * c)?,
                    z: head_of(stream, &g.z, m * zl)?,
                    b: head_of(stream, &g.b, m * nv)?,
                    a: head_of(stream, &g.a, m * nv)?,
                    conv: head_of(stream, &g.conv, m * c)?,
                    beta: head_of(stream, &g.beta, m * nv)?,
                    decay: head_of(stream, &g.decay, m * nv)?,
                    o: head_of(stream, &g.o, m * zl)?,
                    y: head_of(stream, &a.attn, m * zl)?,
                })
            }
        };
        Ok(Layer35Run {
            mixer,
            ffn_inp: head_of(stream, &a.ffn_inp, m * d.hidden)?,
            ffn: self.read_ffn(gpu, m)?,
        })
    }

    fn read_ffn(&self, gpu: &Gpu, m: usize) -> Result<Ffn35Run, GpuError> {
        let stream = gpu.stream();
        let a = &self.a;
        let (hidden, slots) = (a.dims.hidden, a.dims.slots());
        let super::scratch::Route::Gated(r) = &a.route else {
            return Err(GpuError::state(
                "qwen35moe::read_ffn",
                "the gated router's buffers",
            ));
        };
        let mut ids = r.ids.to_host_vec(stream)?;
        ids.truncate(m * slots);
        Ok(Ffn35Run {
            logits: head_of(stream, &r.logits, m * a.dims.router.logits())?,
            ids,
            weights: head_of(stream, &r.weights, m * slots)?,
            l_out: head_of(stream, &a.x, m * hidden)?,
        })
    }
}

/// Err unless `v` holds whole rows of `hidden` for `1..=MAX_PASS_ROWS` rows:
/// their count.
fn rows_of(what: &'static str, v: &[f32], hidden: usize) -> Result<usize, GpuError> {
    let m = v.len() / hidden.max(1);
    if m * hidden != v.len() || !(1..=MAX_PASS_ROWS).contains(&m) {
        return Err(GpuError::shape(
            what,
            format!("{} values: 1..={MAX_PASS_ROWS} rows of {hidden}", v.len()),
        ));
    }
    Ok(m)
}

impl GpuModel<Body35> {
    /// Run layer `l` eagerly at the rows of `x_in` (1 to [`MAX_PASS_ROWS`]
    /// rows of the residual) from position `pos` on the pass arena, and read
    /// back everything it left. The pass's front runs first for the rows'
    /// positions and live key counts (the embedding of id 0 at each), and
    /// `x_in` then replaces the embedding rows; the layer's store is read
    /// and written as a pass writes it (the rows' K/V appended, the state
    /// and ring moved `m` positions on). Synchronizes; gate use.
    pub fn layer_rows(&mut self, l: usize, x_in: &[f32], pos: u32) -> Result<Layer35Run, GpuError> {
        const WHAT: &str = "qwen35moe::layer_rows";
        let hidden = self.body(WHAT)?.hidden();
        let m = rows_of(WHAT, x_in, hidden)?;
        self.check_pos(pos + crate::launch_u32(WHAT, "rows", m - 1)?, WHAT)?;
        let slot = self.layer_slot(l, WHAT)?;
        let (gpu, w, b) = self.body_parts(WHAT)?;
        let stream = gpu.stream();
        b.rp.write(stream, &vec![0; m], pos)?;
        {
            let io = b.rp.io(m)?;
            dispatch::embed_rows(gpu, w, &io, &mut b.a)?;
        }
        // SAFETY: `m · hidden` values lie inside `x` (`MAX_PASS_ROWS ·
        // hidden`); the window lives for this copy alone.
        let mut rows = unsafe { f32_view(&b.a.x, 0, m * hidden) };
        rows.copy_from_host(stream, x_in)?;
        let c = Ctx::new(
            gpu,
            w,
            (&b.plans[slot], slot),
            &b.k,
            b.mma,
            b.eps,
            &b.rope.table,
        )?;
        let io = b.rp.io(m)?;
        dispatch::layer(&c, b.stores[slot].as_mut(), &mut b.a, &io, m, false, None)?;
        b.read_run(gpu, slot, m)
    }

    /// Run layer `l`'s FFN half alone eagerly at the rows of `ffn_inp` (1 to
    /// [`MAX_PASS_ROWS`] rows of its input residual) on the pass arena, and
    /// read back what it left. Synchronizes; gate use.
    pub fn ffn_rows(&mut self, l: usize, ffn_inp: &[f32]) -> Result<Ffn35Run, GpuError> {
        const WHAT: &str = "qwen35moe::ffn_rows";
        let hidden = self.body(WHAT)?.hidden();
        let m = rows_of(WHAT, ffn_inp, hidden)?;
        let slot = self.layer_slot(l, WHAT)?;
        let (gpu, w, b) = self.body_parts(WHAT)?;
        let stream = gpu.stream();
        // SAFETY: `m · hidden` values lie inside `ffn_inp` (`MAX_PASS_ROWS ·
        // hidden`); the window lives for this copy alone.
        let mut rows = unsafe { f32_view(&b.a.ffn_inp, 0, m * hidden) };
        rows.copy_from_host(stream, ffn_inp)?;
        let c = Ctx::new(
            gpu,
            w,
            (&b.plans[slot], slot),
            &b.k,
            b.mma,
            b.eps,
            &b.rope.table,
        )?;
        dispatch::ffn(&c, &b.plans[slot].ffn, &mut b.a, m, None)?;
        b.read_ffn(gpu, m)
    }

    /// Keep (or stop keeping) a copy of every layer's output residual after
    /// each layer of the decode chain. Switches the model to eager mode: the
    /// copies are read by [`GpuModel::layer_taps`] after an eager step.
    pub fn set_layer_taps(&mut self, on: bool) -> Result<(), GpuError> {
        self.set_mode(StepMode::Eager);
        let (gpu, _, body) = self.body_parts("qwen35moe::set_layer_taps")?;
        body.set_taps(gpu.stream(), on)
    }

    /// Every layer's output residual from the last step, layer by layer.
    pub fn layer_taps(&mut self) -> Result<Vec<Vec<f32>>, GpuError> {
        const WHAT: &str = "qwen35moe::layer_taps";
        let (gpu, _, body) = self.body_parts(WHAT)?;
        let t = body
            .taps
            .as_ref()
            .ok_or(GpuError::state(WHAT, "layer taps are off (set_layer_taps)"))?;
        let all = t.buf.to_host_vec(gpu.stream())?;
        Ok(all.chunks(body.hidden()).map(<[f32]>::to_vec).collect())
    }

    /// Run the attention layers on the tensor-core flash pass (`mma`, the
    /// pass a load runs) or the scalar one from here on — a same-class
    /// ruler for a gate, not an engine path. Eager mode only: a captured
    /// graph holds the pass it was captured with.
    pub fn set_flash_mma(&mut self, mma: bool) -> Result<(), GpuError> {
        const WHAT: &str = "qwen35moe::set_flash_mma";
        if self.mode() != StepMode::Eager {
            return Err(GpuError::state(
                WHAT,
                "eager mode (the captured graphs hold the flash pass of their capture)",
            ));
        }
        self.body_parts(WHAT)?.2.mma = mma;
        Ok(())
    }
}
