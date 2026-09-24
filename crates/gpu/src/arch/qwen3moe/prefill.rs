//! qwen3moe's prompt prefill: a prompt fed [`MAX_TOKENS`] positions per pass
//! instead of one decode step per token. A pass is the decode chain's own
//! layer body at `m` rows (`dispatch::enqueue_prefill_pass`), and every
//! launch in it computes each token's values the way the one-row launch
//! does: the gemv, norm and quantizer bodies take a column index; the flash
//! runs each query row against its own live key count; the router routes
//! each token with its own warp; the experts' gate·up dots each token's
//! column, and the down `_sel` runs per token. So a prefilled prompt leaves
//! the K/V rows, and the head the logits, that feeding it one step per token
//! leaves.
//!
//! Passes run eagerly and are not captured: the decode step's graph stays
//! valid beside them. The prefill arena is allocated at the first prefill;
//! the prompt's parameter image — ids, positions, live key counts and rope
//! rows for every token — is written and copied once per prompt, and each
//! pass reads its tokens' windows of it.

use super::body::Body;
use super::dispatch;
use super::router::MAX_TOKENS;
use super::scratch::{Arena, Io, param_view};
use crate::flash_gqa::HEAD;
use crate::model::GpuModel;
use crate::rope_table::{Direction, RopeTable};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use std::mem::ManuallyDrop;

/// The prefill arena and the current prompt's parameter image.
pub(super) struct Prefill {
    pub(super) a: Arena,
    /// The image on the card: ids, positions and live key counts
    /// (`len` u32 each), then one rope row of [`HEAD`] f32 per token as
    /// bits. Grown to the longest prompt so far.
    params: DeviceBuffer<u32>,
    params_host: Vec<u32>,
    cs_host: Vec<f32>,
    /// Tokens of the prompt the image holds.
    len: usize,
}

/// One pass's windows onto the image, owned so that the arena can be
/// borrowed beside them.
pub(super) struct Windows {
    tokens: ManuallyDrop<DeviceBuffer<u32>>,
    pos: ManuallyDrop<DeviceBuffer<u32>>,
    n_keys: ManuallyDrop<DeviceBuffer<u32>>,
    cs: ManuallyDrop<DeviceBuffer<f32>>,
}

impl Windows {
    pub(super) fn io(&self) -> Io<'_> {
        Io {
            tokens: &self.tokens,
            pos: &self.pos,
            n_keys: &self.n_keys,
            cs: &self.cs,
        }
    }
}

impl Prefill {
    /// The prefill state over arena `a`, with an empty image.
    fn new(stream: &CudaStream, a: Arena) -> Result<Prefill, GpuError> {
        Ok(Prefill {
            a,
            params: DeviceBuffer::zeroed(stream, 1)?,
            params_host: Vec::new(),
            cs_host: Vec::new(),
            len: 0,
        })
    }

    /// Write the image of `tokens` at positions `pos0 ..` and copy it to the
    /// card. Synchronizes (the copy of a borrowed host slice does).
    fn write(
        &mut self,
        stream: &CudaStream,
        rope: &RopeTable,
        tokens: &[u32],
        pos0: u32,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::prefill";
        let n = launch_u32(what, "tokens", tokens.len())?;
        self.params_host.clear();
        self.params_host.extend_from_slice(tokens);
        self.params_host.extend((0..n).map(|t| pos0 + t));
        self.params_host.extend((0..n).map(|t| pos0 + t + 1));
        self.cs_host.clear();
        for t in 0..n {
            rope.push(pos0 + t, Direction::Forward, &mut self.cs_host);
        }
        self.params_host
            .extend(self.cs_host.iter().map(|v| v.to_bits()));
        let len = self.params_host.len();
        if self.params.len() < len {
            self.params = DeviceBuffer::zeroed(stream, len)?;
        }
        // SAFETY: `len <= self.params.len()`, so the window is inside it; it
        // lives for this copy alone.
        let mut dst = unsafe { param_view::<u32>(&self.params, 0, len) };
        dst.copy_from_host(stream, &self.params_host)?;
        self.len = tokens.len();
        Ok(())
    }

    /// Pass `chunk`'s windows: its `m` tokens start at token `chunk ·
    /// MAX_TOKENS` of the image.
    pub(super) fn windows(&self, chunk: usize, m: usize) -> Result<Windows, GpuError> {
        let t0 = chunk * MAX_TOKENS;
        if m == 0 || m > MAX_TOKENS || t0 + m > self.len {
            return Err(GpuError::shape(
                "qwen3moe::prefill",
                format!(
                    "pass {chunk} of {m} tokens past the {}-token image",
                    self.len
                ),
            ));
        }
        let n = self.len;
        // SAFETY: the image holds `3·n + n·HEAD` u32 (`write`), and each
        // window lies inside its array: tokens `t0 .. t0 + m <= n` of the ids,
        // positions and counts at `0`, `n` and `2·n`, and their rope rows at
        // `3·n + t0·HEAD`. The windows live for one pass, while the image
        // stays in place.
        unsafe {
            Ok(Windows {
                tokens: param_view::<u32>(&self.params, t0, m),
                pos: param_view::<u32>(&self.params, n + t0, m),
                n_keys: param_view::<u32>(&self.params, 2 * n + t0, m),
                cs: param_view::<f32>(&self.params, 3 * n + t0 * HEAD, m * HEAD),
            })
        }
    }

    /// Device bytes of the arena and the image.
    pub(super) fn bytes(&self) -> usize {
        self.a.bytes() + self.params.num_bytes()
    }
}

impl GpuModel<Body> {
    /// Feed `tokens` through the chain [`MAX_TOKENS`] positions per pass
    /// (module doc) and return the greedy next token after the last one —
    /// what [`GpuModel::step`] returns for the same tokens from the same
    /// position, and the same cache rows behind it. Positions continue from
    /// wherever the model stands. The first call allocates the prefill
    /// arena; the layer taps must be off.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        const WHAT: &str = "qwen3moe::prefill";
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT, "empty token slice"));
        }
        let pos0 = self.pos();
        self.check_pos(pos0 + launch_u32(WHAT, "tokens", tokens.len())? - 1, WHAT)?;
        {
            let (gpu, _, body) = self.body_parts(WHAT)?;
            if body.taps.is_some() {
                return Err(GpuError::state(
                    WHAT,
                    "layer taps are on; a pass writes none",
                ));
            }
            let stream = gpu.stream();
            if body.prefill.is_none() {
                let a = Arena::new(stream, body.s.dims, MAX_TOKENS)?;
                body.prefill = Some(Prefill::new(stream, a)?);
            }
            let Body { prefill, rope, .. } = body;
            prefill
                .as_mut()
                .ok_or(GpuError::state(WHAT, "no prefill arena"))?
                .write(stream, rope, tokens, pos0)?;
        }
        let passes = tokens.len().div_ceil(MAX_TOKENS);
        let mut next = None;
        for (chunk, run) in tokens.chunks(MAX_TOKENS).enumerate() {
            let last = chunk + 1 == passes;
            next = self.run_rows(run.len(), WHAT, |gpu, w, body, head, _| {
                dispatch::enqueue_prefill_pass(
                    gpu,
                    w,
                    body,
                    chunk,
                    run.len(),
                    last.then_some(head),
                )?;
                Ok(last)
            })?;
        }
        next.ok_or(GpuError::state(WHAT, "the last pass read no token"))
    }

    /// The passes [`GpuModel::prefill`] takes for a prompt of `tokens` ids.
    #[must_use]
    pub fn prefill_passes(tokens: usize) -> usize {
        tokens.div_ceil(MAX_TOKENS)
    }
}
