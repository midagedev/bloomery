//! The draft's head: the block's folded rows through the draft's own
//! `output_norm` and the target's Q6_K projection, then the Markov loop.
//!
//! `bloomery_gpu::head::Head` reads its norm gain and its projection from
//! one `Weights`; here the gain is the draft file's and the projection the
//! target's (borrowed at load), so this module launches the same chain
//! itself: `rms_norm` → the shared q8_1 quantizer → `gemv_q6k`, one launch
//! each over the block's `m` rows (each row bit for bit its `m = 1` launch,
//! as `Head` states for its own chain). The logits land in the gemv's
//! layout, `logits[v · m + c]`, which is `ds41_markov`'s.
//!
//! The Markov loop is `m` serial steps ([`MarkovKernels::enqueue_step`]):
//! row 0's previous token is the block's first id (`id_last`, read from the
//! device, where the pass's staging put it), row `c`'s the argmax the step
//! before wrote. The last step's argmax buffer holds the block's proposal
//! and, after it, the card's fault word, so the head adds no argmax of its
//! own and [`DraftHead::tokens`]' one readback turns a fault any launch of
//! the pass raised into [`GpuError::Fault`], as `bloomery_gpu::head::Head`
//! does for the target.

use bloomery_gpu::weights::DevWeight;
use bloomery_gpu::{DeviceTensor, Fault, Gpu, GpuError, Q8Act};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::quant::GgmlType;
use model::arch::deepseek41::names as target_names;
use model::arch::dspark::names;

use super::load::DraftWeights;
use crate::markov::{MarkovKernels, MarkovWeights};

const WHAT: &str = "draft::head";

/// The head's buffers for blocks of up to `max_width` rows.
pub struct DraftHead {
    normed: DeviceBuffer<f32>,
    /// The quantized rows, one scratch per width (`Q8Act` fixes its rows).
    acts: Vec<Q8Act>,
    logits: DeviceBuffer<f32>,
    /// Each row's argmax, then the fault's first-layer word at `tok[m]` and
    /// its layer's site mask at `tok[m + 1]`; after the loop, the block's
    /// proposal. `max_width + 2` words.
    tok: DeviceBuffer<u32>,
    markov: MarkovKernels,
    n_embd: usize,
    n_vocab: usize,
    eps: f32,
}

/// The target's Q6_K projection as the loader holds it.
fn projection(w: &DraftWeights) -> Result<&DeviceTensor<u32>, GpuError> {
    let name = target_names::output();
    match w.head().get(&name) {
        Some(DevWeight::KQuant {
            ty: GgmlType::Q6_K,
            w,
            ..
        }) => Ok(w),
        _ => Err(GpuError::Tensor {
            what: WHAT,
            name,
            need: "a resident Q6_K projection",
        }),
    }
}

impl DraftHead {
    /// Buffers for blocks of `1..=max_width` rows. Load-time only.
    pub fn new(gpu: &Gpu, w: &DraftWeights, max_width: usize) -> Result<DraftHead, GpuError> {
        let s = gpu.stream();
        let hp = w.hp();
        let n_vocab = projection(w)?.rows();
        let acts = (1..=max_width)
            .map(|m| Q8Act::with_k(s, m, hp.n_embd))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(DraftHead {
            normed: DeviceBuffer::zeroed(s, max_width * hp.n_embd)?,
            acts,
            logits: DeviceBuffer::zeroed(s, max_width * n_vocab)?,
            tok: DeviceBuffer::zeroed(s, max_width + 2)?,
            markov: MarkovKernels::load(gpu.context())?,
            n_embd: hp.n_embd,
            n_vocab,
            eps: hp.rms_eps,
        })
    }

    /// Rows of the projection: the vocabulary.
    #[must_use]
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// Refuse a width the head holds no scratch for.
    fn check(&self, m: usize) -> Result<(), GpuError> {
        if (1..=self.acts.len()).contains(&m) {
            Ok(())
        } else {
            Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a block of {m} rows; the head holds 1..={}",
                    self.acts.len()
                ),
            })
        }
    }

    /// Enqueue the norm, the quantizer and the projection over `m` folded
    /// rows `x` (`m · n_embd`, token-major). Three launches. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_logits(
        &mut self,
        gpu: &Gpu,
        w: &DraftWeights,
        x: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        self.check(m)?;
        let s = gpu.stream();
        let gain = w.gain(&names::output_norm(), self.n_embd)?;
        gpu.elem()
            .enqueue_rms_norm(s, x, gain, self.eps, self.n_embd, m, &mut self.normed)?;
        let act = &mut self.acts[m - 1];
        gpu.enqueue_quantize_q8_1(&self.normed, act)?;
        gpu.enqueue_gemv_q6k(projection(w)?, act, &mut self.logits)
    }

    /// Enqueue the Markov loop over the `m` rows' logits, row 0's previous
    /// token `first[0]`: `m` steps of two launches. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_markov(
        &mut self,
        gpu: &Gpu,
        w: &DraftWeights,
        m: usize,
        first: &DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.check(m)?;
        let (w1, w2) = w.markov();
        let mw = MarkovWeights { w1, w2 };
        for row in 0..m {
            self.markov.enqueue_step(
                gpu.stream(),
                gpu.elem(),
                &mw,
                first,
                &mut self.tok,
                m,
                row,
                gpu.unlabelled_sink(),
                &mut self.logits,
            )?;
        }
        Ok(())
    }

    /// Launches one block of any width makes before the Markov loop.
    pub const LOGIT_LAUNCHES: usize = 3;

    /// Launches the Markov loop of an `m`-row block makes.
    #[must_use]
    pub fn markov_launches(m: usize) -> usize {
        2 * m
    }

    /// The `m` rows' normed input, token-major. Blocking; gate use.
    pub fn normed_to_host(&self, stream: &CudaStream, m: usize) -> Result<Vec<f32>, GpuError> {
        Ok(self.normed.to_host_vec(stream)?[..m * self.n_embd].to_vec())
    }

    /// The `m` rows' logits in the gemv's layout (`v · m + c`). Blocking;
    /// gate use.
    pub fn logits_to_host(&self, stream: &CudaStream, m: usize) -> Result<Vec<f32>, GpuError> {
        Ok(self.logits.to_host_vec(stream)?[..m * self.n_vocab].to_vec())
    }

    /// The `m` rows' proposal. Blocking. A fault the last step copied is
    /// [`GpuError::Fault`], not a proposal; it stays raised on the card until
    /// the `Gpu`'s owner clears it.
    pub fn tokens(&self, stream: &CudaStream, m: usize) -> Result<Vec<u32>, GpuError> {
        self.check(m)?;
        let out = self.tok.to_host_vec(stream)?;
        if let Some(fault) = Fault::from_words(out[m], out[m + 1]) {
            return Err(GpuError::Fault {
                what: "draft::head::tokens",
                fault,
            });
        }
        Ok(out[..m].to_vec())
    }
}
