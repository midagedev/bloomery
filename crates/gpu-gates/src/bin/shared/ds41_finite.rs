//! The V4.1 step's finite probe, for the bins that read a step's seams
//! (`generate_ds41`'s `BLOOMERY_CHECK_FINITE`, `gate_deepseek41_skew`'s cuts,
//! `gate_deepseek41_long`): one step of the body run eagerly through
//! `Body::enqueue_observed`, each sub-layer's streams read back where it wrote
//! them, every seam whose streams hold a NaN or an infinity named, and at the
//! first such MoE seam every buffer the sub-layer wrote, in launch order.
//!
//! The logits alone cannot see this. A q8_1 quantizer takes its scale from
//! the block's `max(|v|)` and falls back to 1 when that is not above 0 — a
//! NaN is not — and a NaN code converts to 0: the head's quantized input is
//! then all zeros, every logit is 0, and the argmax is id 0. The residual
//! streams keep the NaN from the seam where it appeared to the last layer,
//! so the streams are where it shows.
//!
//! It sits beside the bins, each including it with `#[path]`, not in the
//! gpu-gates library: it names the V4.1 device crate, and a crate the library
//! names is linked, device bundle and all, into every binary built with the
//! `deepseek41` feature.
//!
//! An observed step writes the caches at its position and appends its token
//! to the body's history, and leaves `GpuModel::pos` where it was. A caller
//! that goes on takes the position back (`GpuModel::rollback`, which grants
//! the last position) and steps the token through the engine.

use std::fmt;

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::HOST;
use bloomery_gpu::model::ChainBody;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::body::{Deepseek41Model, Seam};
use bloomery_gpu_deepseek41::chain::ffn::FfnTaps;
use bloomery_gpu_deepseek41::router::N_USED;
use bloomery_gpu_gates::GateError;
use cuda_core::DeviceBuffer;

/// A seam of the step: the layer, and the piece whose streams it shows
/// (`"engram"`, `"attention"` or `"moe"`).
pub type Site = (usize, &'static str);

/// What one observed step read.
pub struct Observed {
    /// The head's logits, as bits.
    pub logits: Vec<u32>,
    /// Every seam whose streams hold a NaN or an infinity, in step order.
    pub nonfinite: Vec<Site>,
    /// At the first non-finite MoE seam, the sub-layer's buffers.
    pub moe: Option<MoeFault>,
}

impl Observed {
    /// The first seam whose streams are not finite.
    pub fn first_nonfinite(&self) -> Option<Site> {
        self.nonfinite.first().copied()
    }

    /// The logits' largest finite magnitude and how many are not finite.
    pub fn logit_stats(&self) -> (f32, usize) {
        self.logits.iter().fold((0.0f32, 0), |(m, bad), &b| {
            let x = f32::from_bits(b);
            if x.is_finite() {
                (m.max(x.abs()), bad)
            } else {
                (m, bad + 1)
            }
        })
    }

    /// `ok`, or the non-finite seams, the first one, the logits' state and,
    /// at a MoE seam, the sub-layer's routing and buffers.
    pub fn describe(&self) -> String {
        let Some(first) = self.first_nonfinite() else {
            return "ok".to_string();
        };
        let (absmax, bad) = self.logit_stats();
        let seams: Vec<String> = self.nonfinite.iter().map(|&s| site_name(s)).collect();
        let mut line = format!(
            "nonfinite_seams={} first={} ({}) logits absmax={absmax:e} nonfinite={bad}",
            self.nonfinite.len(),
            site_name(first),
            seams.join(" ")
        );
        if let Some(f) = &self.moe {
            line.push_str(&format!(" {f}"));
        }
        line
    }

    /// The largest logit's id, the first of equals; a NaN is never the
    /// largest.
    pub fn token(&self) -> u32 {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, &b) in self.logits.iter().enumerate() {
            let x = f32::from_bits(b);
            if x > best.1 {
                best = (i, x);
            }
        }
        u32::try_from(best.0).expect("a vocabulary index fits in u32")
    }
}

/// A seam as the lines print it: `(layer, piece)`.
pub fn site_name((layer, piece): Site) -> String {
    format!("({layer}, {piece})")
}

/// One buffer of a MoE sub-layer as the probe read it.
pub struct Buf {
    pub name: String,
    pub len: usize,
    /// Values that are NaN or infinite, and the first one's index.
    pub bad: usize,
    pub first_bad: Option<usize>,
    /// The finite values' largest magnitude.
    pub absmax: f32,
}

impl Buf {
    fn of(name: String, v: &[f32]) -> Buf {
        let mut bad = 0;
        let mut first_bad = None;
        let mut absmax = 0.0f32;
        for (i, &x) in v.iter().enumerate() {
            if x.is_finite() {
                absmax = absmax.max(x.abs());
            } else {
                bad += 1;
                first_bad.get_or_insert(i);
            }
        }
        Buf {
            name,
            len: v.len(),
            bad,
            first_bad,
            absmax,
        }
    }
}

/// A MoE sub-layer whose streams came out non-finite: its routing and its
/// buffers in launch order — the input it read (the fold its attention seam
/// left), the router, HC_PRE, each card slot's SwiGLU and down, the shared
/// expert, the combine, the new streams and the next fold. A host slot's h
/// and down are not the card's work and are left out; the host's partial sum
/// is not a buffer the seam shows, so where the combine is the first bad
/// buffer with every card slot and the shared expert finite, the sum it added
/// is the one left.
pub struct MoeFault {
    pub layer: usize,
    pub ids: [u32; N_USED],
    pub weights: [f32; N_USED],
    /// Each slot's place in the card's stacks, or [`HOST`].
    pub places: [u32; N_USED],
    /// The sub-layer's input before its norm.
    pub fold_in: Vec<f32>,
    pub buffers: Vec<Buf>,
}

impl MoeFault {
    /// The first buffer in launch order with a non-finite value.
    pub fn first_bad(&self) -> Option<&Buf> {
        self.buffers.iter().find(|b| b.bad > 0)
    }

    /// Whether the combine is the first bad buffer: every card slot and the
    /// shared expert finite, so the host's partial sum carried it.
    pub fn host_sum_carried(&self) -> bool {
        self.first_bad().is_some_and(|b| b.name == "y")
    }
}

impl fmt::Display for MoeFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "layer {} moe: routing", self.layer)?;
        for s in 0..N_USED {
            let place = if self.places[s] == HOST {
                "host".to_string()
            } else {
                format!("card {}", self.places[s])
            };
            write!(f, " [{} w={:.6} {place}]", self.ids[s], self.weights[s])?;
        }
        match self.first_bad() {
            Some(b) => write!(
                f,
                "; first bad buffer {} ({}/{} non-finite, first at {:?}){}",
                b.name,
                b.bad,
                b.len,
                b.first_bad,
                if self.host_sum_carried() {
                    " — every card slot and the shared expert are finite: the host's partial sum"
                } else {
                    ""
                }
            )?,
            None => write!(f, "; every buffer the seam shows is finite")?,
        }
        let ss: f64 = self
            .fold_in
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum();
        let rms = (ss / self.fold_in.len().max(1) as f64).sqrt();
        write!(f, "; fold_in rms={rms:e}")?;
        for b in &self.buffers {
            write!(f, "; {} absmax={:e} bad={}", b.name, b.absmax, b.bad)?;
        }
        Ok(())
    }
}

/// `buf` read back as f32 values.
fn host_f32(gpu: &Gpu, buf: &DeviceBuffer<f32>) -> Result<Vec<f32>, GpuError> {
    Ok(buf.to_host_vec(gpu.stream())?)
}

/// The MoE seam's buffers, read after the stream has synchronized.
fn moe_fault(
    gpu: &Gpu,
    layer: usize,
    taps: &FfnTaps<'_>,
    streams: &[f32],
    fold: Option<&DeviceBuffer<f32>>,
    fold_in: &[f32],
) -> Result<MoeFault, GpuError> {
    let stream = gpu.stream();
    let six = |v: Vec<u32>| -> [u32; N_USED] { std::array::from_fn(|s| v[s]) };
    let ids = six(taps.router.ids.to_host_vec(stream)?);
    let places = six(taps.sel.to_host_vec(stream)?);
    let w = host_f32(gpu, &taps.router.weights)?;
    let weights: [f32; N_USED] = std::array::from_fn(|s| w[s]);
    let mut buffers = vec![
        Buf::of("fold_in".into(), fold_in),
        Buf::of("router.logits".into(), &host_f32(gpu, &taps.router.logits)?),
        Buf::of("router.probs".into(), &host_f32(gpu, &taps.router.probs)?),
        Buf::of("router.weights".into(), &w),
        Buf::of("hc_pre".into(), &host_f32(gpu, taps.hc)?),
    ];
    let h = host_f32(gpu, taps.h)?;
    let down = host_f32(gpu, taps.down)?;
    let (ff, n) = (h.len() / N_USED, down.len() / N_USED);
    for s in (0..N_USED).filter(|&s| places[s] != HOST) {
        let tag = format!("[slot {s} id {}]", ids[s]);
        buffers.push(Buf::of(format!("h{tag}"), &h[s * ff..(s + 1) * ff]));
        buffers.push(Buf::of(format!("down{tag}"), &down[s * n..(s + 1) * n]));
    }
    buffers.push(Buf::of("shexp_h".into(), &host_f32(gpu, taps.shexp_h)?));
    buffers.push(Buf::of("shexp".into(), &host_f32(gpu, taps.shexp)?));
    buffers.push(Buf::of("y".into(), &host_f32(gpu, taps.y)?));
    buffers.push(Buf::of("streams".into(), streams));
    if let Some(fold) = fold {
        buffers.push(Buf::of("fold_out".into(), &host_f32(gpu, fold)?));
    }
    Ok(MoeFault {
        layer,
        ids,
        weights,
        places,
        fold_in: fold_in.to_vec(),
        buffers,
    })
}

/// What a caller of [`observed_step`] is shown at each seam once the probe
/// has read it: the seam's buffers, and the streams' values as the probe read
/// them.
pub type SeamHook<'h> = dyn FnMut(&Gpu, &Seam<'_>, &[f32]) -> Result<(), GpuError> + 'h;

/// The step of `token` at `pos` on `m`'s body, eagerly, through `head` (the
/// caller's, not the engine's): every seam's streams read where the step
/// wrote them and shown to `hook`, the head's logits read at the end.
pub fn observed_step(
    m: &mut Deepseek41Model,
    head: &mut Head,
    token: u32,
    pos: u32,
    hook: &mut SeamHook<'_>,
) -> Result<Observed, GateError> {
    let (gpu, w, body) = m.body_parts("finite probe")?;
    let input = body.decode_input(token, pos)?;
    body.refresh(gpu.stream(), &input)?;
    let mut nonfinite = Vec::new();
    let mut moe = None;
    let mut fold_in = Vec::new();
    body.enqueue_observed(gpu, w, head, &mut |gpu, seam| {
        gpu.stream().synchronize()?;
        let (site, streams) = match &seam {
            Seam::Engram { layer, streams, .. } => ((*layer, "engram"), *streams),
            Seam::Attn {
                layer,
                streams,
                fold,
                ..
            } => {
                fold_in = host_f32(gpu, fold)?;
                ((*layer, "attention"), *streams)
            }
            Seam::Ffn { layer, streams, .. } => ((*layer, "moe"), *streams),
        };
        let v = host_f32(gpu, streams)?;
        if !v.iter().all(|x| x.is_finite()) {
            nonfinite.push(site);
            if moe.is_none()
                && let Seam::Ffn {
                    layer, fold, taps, ..
                } = &seam
            {
                moe = Some(moe_fault(gpu, *layer, taps, &v, *fold, &fold_in)?);
            }
        }
        hook(gpu, &seam, &v)
    })?;
    gpu.stream().synchronize()?;
    let logits = head
        .logits_to_host(gpu)?
        .into_iter()
        .map(f32::to_bits)
        .collect();
    Ok(Observed {
        logits,
        nonfinite,
        moe,
    })
}
