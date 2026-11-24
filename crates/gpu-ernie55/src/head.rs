//! ERNIE 5.5's side of the output head: the mix of the four streams into the
//! common head's input — the four-row `output_hc_fn` under the V4 RMS rule
//! (`hc_head_f32`: the F32 piece entry of `bloomery_gpu::hc_f32_pieces`), then the
//! common fold `ds41_hc_fold` by its four `pre` gates ([`Head::enqueue_fold`]) —
//! and the MultiMax epilogue on the common head's logits
//! ([`Head::enqueue_multimax`], [`bloomery_gpu::multimax`]), between its
//! projection and its argmax, so the token and the logits a sampler reads are the
//! MultiMax ones. The common head itself (`output_norm`, the BF16 `output`, the
//! argmax: [`bloomery_gpu::head::Head`]) is the model's, not this part's.
//!
//! What the file forces here, each named where it is used: the head keeps the V4
//! factor `rsqrt(mean + eps)` where every site's mix divides by `sqrt(mean) + eps`;
//! the fold by the head's own gates, not the last sub-layer's; the eight MultiMax
//! values, finite or refused by name at load.

use bloomery_gpu::hc::{HC_MIX, HC_STREAMS, HcKernels, HcPreScratch};
use bloomery_gpu::hc_f32::{HcF32Args, HcF32Params};
use bloomery_gpu::hc_f32_pieces::{HcF32PiecesKernels, LEAD};
use bloomery_gpu::multimax::{MULTIMAX, MultiMaxArgs, MultiMaxKernels};
use bloomery_gpu::site;
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{Gpu, GpuError, LAYER_HEAD, Window, col_group_count};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::ernie55::names::{
    OUTPUT_HC_BASE, OUTPUT_HC_FN, OUTPUT_HC_SCALE, OUTPUT_MULTIMAX_RANGES, OUTPUT_MULTIMAX_TS,
};

use crate::trunk::Facts;

/// What the refusals of this file name.
const WHAT: &str = "ernie55::Head";

const _: () = assert!(model::arch::ernie55::hparams::MULTIMAX == MULTIMAX);

/// Launches of the head past the common head's own: the stream mix and its fold
/// before it, the MultiMax epilogue between its projection and its argmax.
pub const HEAD_EXTRA_LAUNCHES: usize = 3;

/// ERNIE 5.5's side of the output head: the stream mix's kernels and parameters,
/// the MultiMax values, and the buffers a run writes.
pub struct Head {
    pieces: HcF32PiecesKernels,
    hc: HcKernels,
    multimax: MultiMaxKernels,
    scratch: HcPreScratch,
    /// The head's scaled mixes: four a token, in [`HC_MIX`]-wide rows.
    mixes: DeviceBuffer<f32>,
    /// The head's pre gates: four a token, in [`HC_MIX`]-wide rows the fold reads.
    pre: DeviceBuffer<f32>,
    ranges: DeviceBuffer<f32>,
    ts: DeviceBuffer<f32>,
    hidden: usize,
    hc_eps: f32,
    iters: u32,
    rms_eps: f32,
    m: usize,
}

/// The four values of the resident f32 vector `name`, read back once: each finite,
/// or refused by name.
fn multimax_values(w: &Weights, stream: &CudaStream, name: &str) -> Result<Vec<f32>, GpuError> {
    let values = w.f32_buf(WHAT, name)?.to_host_vec(stream)?;
    if let Some(i) = values.iter().position(|v| !v.is_finite()) {
        return Err(GpuError::shape(
            WHAT,
            format!("{name}[{i}] is {}, not a finite value", values[i]),
        ));
    }
    Ok(values)
}

impl Head {
    /// The head's side of the description's `facts` over the resident `w`, for
    /// `m` rows (1 ..= 8) a run, beside a common head of as many rows:
    /// `output_hc_fn` held to [`HC_STREAMS`] F32 rows of the four streams' width,
    /// its base and scale, and the eight MultiMax values. Refused by name: no row,
    /// another form or shape of any of them, a MultiMax value that is not finite.
    /// Load-time only.
    pub fn load(gpu: &Gpu, w: &Weights, facts: &Facts, m: usize) -> Result<Head, GpuError> {
        let stream = gpu.stream();
        let k = HC_STREAMS * facts.hidden;
        if m == 0 {
            return Err(GpuError::shape(WHAT, "a head of no row a run"));
        }
        match w.resident(WHAT, OUTPUT_HC_FN)? {
            DevWeight::F32 { w, .. } if (w.rows(), w.cols()) == (HC_STREAMS, k) => {}
            DevWeight::F32 { w, .. } => {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "{OUTPUT_HC_FN} is {} rows of {} values; the head takes {HC_STREAMS} of {k}",
                        w.rows(),
                        w.cols()
                    ),
                ));
            }
            _ => {
                return Err(GpuError::tensor(
                    WHAT,
                    OUTPUT_HC_FN,
                    "F32 rows (the head runs the F32 piece entry)",
                ));
            }
        }
        site::vector(w, WHAT, OUTPUT_HC_BASE.to_string(), HC_STREAMS)?;
        site::vector(w, WHAT, OUTPUT_HC_SCALE.to_string(), 1)?;
        let ranges = multimax_values(w, stream, OUTPUT_MULTIMAX_RANGES)?;
        let ts = multimax_values(w, stream, OUTPUT_MULTIMAX_TS)?;
        if ranges.len() != MULTIMAX || ts.len() != MULTIMAX {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{OUTPUT_MULTIMAX_RANGES} holds {} and {OUTPUT_MULTIMAX_TS} {} values; the \
                     epilogue takes {MULTIMAX} of each",
                    ranges.len(),
                    ts.len()
                ),
            ));
        }
        let ranges = DeviceBuffer::from_host(stream, &ranges)?;
        let ts = DeviceBuffer::from_host(stream, &ts)?;
        let z = |len: usize| DeviceBuffer::zeroed(stream, len);
        Ok(Head {
            pieces: HcF32PiecesKernels::load(gpu.context())?,
            hc: HcKernels::load(gpu.context())?,
            multimax: MultiMaxKernels::load(gpu.context())?,
            scratch: HcPreScratch::with_groups(stream, k, col_group_count(LEAD, m))?,
            mixes: z(m * HC_MIX)?,
            pre: z(m * HC_MIX)?,
            ranges,
            ts,
            hidden: facts.hidden,
            hc_eps: facts.hc_eps,
            iters: facts.sinkhorn_iters,
            rms_eps: facts.rms_eps,
            m,
        })
    }

    /// The head's mix parameters over the weights.
    fn params<'w>(&self, w: &'w Weights) -> Result<HcF32Params<'w>, GpuError> {
        let DevWeight::F32 { w: proj, .. } = w.resident(WHAT, OUTPUT_HC_FN)? else {
            return Err(GpuError::tensor(
                WHAT,
                OUTPUT_HC_FN,
                "F32 rows (the head runs the F32 piece entry)",
            ));
        };
        Ok(HcF32Params {
            w: proj,
            scale: w.f32_buf(WHAT, OUTPUT_HC_SCALE)?,
            base: w.f32_buf(WHAT, OUTPUT_HC_BASE)?,
            eps: self.hc_eps,
            iters: self.iters,
        })
    }

    /// The rows a run folds.
    #[must_use]
    pub fn m(&self) -> usize {
        self.m
    }

    /// The launches of one run past the common head's: the stream mix, its fold,
    /// the MultiMax epilogue.
    #[must_use]
    pub fn launches(&self) -> usize {
        HEAD_EXTRA_LAUNCHES
    }

    /// The device bytes of its scratch and values.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.scratch.bytes()
            + self.mixes.num_bytes()
            + self.pre.num_bytes()
            + self.ranges.num_bytes()
            + self.ts.num_bytes()
    }

    /// The head's mix of the last [`Head::m`] of `tokens` tokens' streams
    /// (four streams of [`Facts::hidden`] values a token) and their fold into
    /// `input`, the common head's input (`m` rows of [`Facts::hidden`]). A prompt
    /// chunk needs no logits before its last rows. Refused by name: fewer tokens
    /// than the rows the head folds, or streams that do not hold `tokens` tokens.
    /// Allocation-free and asynchronous.
    pub fn enqueue_fold(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        streams: &DeviceBuffer<f32>,
        tokens: usize,
        input: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (rows, k) = (self.m, HC_STREAMS * self.hidden);
        if tokens < rows || streams.len() < tokens * k {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a prompt chunk of {tokens} tokens in {} values: the head folds the last \
                     {rows}, {k} values a token",
                    streams.len()
                ),
            ));
        }
        let last = Window::<f32>::of(streams, 4 * (tokens - rows) * k, rows * k)?;
        let stream = gpu.stream();
        let params = self.params(w)?;
        let a = HcF32Args {
            params: &params,
            x: &last,
            tokens: rows,
            rms_eps: self.rms_eps,
            fault: head_sink(gpu),
        };
        self.pieces.enqueue_head(
            stream,
            &a,
            LEAD,
            &mut self.scratch,
            &mut self.mixes,
            &mut self.pre,
        )?;
        self.hc
            .enqueue_fold(stream, &last, &self.pre, self.hidden, rows, input)
    }

    /// MultiMax over the `n` logits in place (the common head's, `n_vocab · m`):
    /// the head's epilogue, before the argmax. Allocation-free and asynchronous.
    pub fn enqueue_multimax(
        &self,
        gpu: &Gpu,
        logits: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        self.multimax.enqueue(
            gpu.stream(),
            MultiMaxArgs {
                y: logits,
                n,
                ranges: &self.ranges,
                ts: &self.ts,
                fault: head_sink(gpu),
            },
        )
    }
}

/// The head's fault sink: the head's layer code.
fn head_sink(gpu: &Gpu) -> bloomery_gpu::FaultSink {
    // SAFETY: `LAYER_HEAD` is a layer code below `LAYER_NONE`, the bound
    // `FaultSink::at_layer` asks; the sink is the `Gpu`'s own allocation.
    unsafe { gpu.unlabelled_sink().at_layer(LAYER_HEAD) }
}
