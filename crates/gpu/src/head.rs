//! The GPU output head: `result_norm → lm_head (Q6_K) → argmax` as one
//! capturable sequence over resident scratch (docs/gpu-design.md decisions 4
//! and 6). No kernels of its own — the chain is the gated `rms_norm`, the
//! shared q8_1 quantizer, the Q6_K gemv and `argmax` exactly as the block
//! path launches them, so their gates are this chain's gates.
//!
//! A head carries `m` rows (1..=8), fixed at construction: the decode step
//! samples one position (`m = 1`), a k-token step every row. Each row's
//! normed vector, logits and token are bit for bit that row's `m = 1` head —
//! the norm and quantizer work per row, the Q6_K gemv keeps each column's
//! own accumulation order, and the argmax walks each row alone. The input
//! buffer is exposed mutable so the assembly round can have the last block's
//! residual store write straight into it, and `set_input` is the gate's way
//! in.
//!
//! The argmax also carries the card's fault word (crate::fault) out: the
//! readback buffer holds the `m` tokens and then the word, one copy, and a
//! raised word turns the readback into [`GpuError::Fault`] instead of a
//! token.

use crate::GpuError;
use crate::fault::{Fault, FaultSink, LAYER_HEAD};
use crate::tensor::{DeviceTensor, Q8Act};
use crate::weights::{DevWeight, Weights};
use crate::{Gpu, Graph};
use cuda_core::DeviceBuffer;
use gguf::quant::GgmlType;

/// The final norm's gain (`output_norm.weight`), resident F32 — the shared
/// weights reader, so the head's residency error reads like every other
/// stage's.
fn head_gain(w: &Weights) -> Result<&DeviceBuffer<f32>, GpuError> {
    crate::model::f32_gain(w, "output_norm.weight")
}

/// The lm_head weight (`output.weight`): its Q6_K word plane and row width.
/// The type is checked, not assumed — a future file whose lm_head is another
/// K-quant needs a different gemv, and a silent word-plane reuse would
/// misread its bytes.
fn head_out_w(w: &Weights) -> Result<(&DeviceTensor<u32>, usize), GpuError> {
    match w.get("output.weight") {
        Some(DevWeight::KQuant { ty, w, k }) => {
            if *ty != GgmlType::Q6_K {
                return Err(GpuError::shape(
                    "head_out_w",
                    format!(
                        "output.weight is {ty}, want Q6_K (the gemv \
                     this head launches reads Q6_K rows)"
                    ),
                ));
            }
            Ok((w, *k))
        }
        Some(_) => Err(GpuError::tensor(
            "head_out_w",
            "output.weight",
            "a K-quant word plane",
        )),
        None => Err(GpuError::tensor(
            "head_out_w",
            "output.weight",
            "resident — load Weights with globals",
        )),
    }
}

/// The resident output head for `m` rows: input `m * hidden` f32 (row-major,
/// one row per token), the normed rows, their q8_1 quantization, `n_vocab * m`
/// logits (`logits[v*m + c]`, the gemv's layout) and `m` argmax tokens. Every
/// buffer is allocated at construction; `enqueue` and a graph replay touch
/// addresses only, so one captured graph serves every input written into `x`.
pub struct Head {
    /// Declared FIRST: fields drop in declaration order, and a captured graph
    /// must be destroyed while every buffer it addresses is still alive
    /// (`GpuModel` and `Stage` order theirs the same way — the reverse order
    /// segfaulted the 12.4 GB model in the e2e round; this head survived at
    /// 432 KB only by luck).
    graph: Option<Graph>,
    eps: f32,
    hidden: usize,
    n_vocab: usize,
    m: usize,
    /// The chain's input — the last block's output vector. The assembly round
    /// writes here through `input_mut`; the gate through `set_input`.
    x: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    act: Q8Act,
    logits: DeviceBuffer<f32>,
    /// The `m` argmax tokens, then the fault word as the argmax found it.
    token_out: DeviceBuffer<u32>,
}

impl Head {
    /// The decode head, `m = 1`: [`Head::with_m`] with one row.
    pub fn new(gpu: &Gpu, w: &Weights, eps: f32) -> Result<Head, GpuError> {
        Head::with_m(gpu, w, eps, 1)
    }

    /// Cross-check the two weights against each other and allocate the
    /// scratch for `m` rows (1..=8). Load-time only.
    pub fn with_m(gpu: &Gpu, w: &Weights, eps: f32, m: usize) -> Result<Head, GpuError> {
        if !(1..=8).contains(&m) {
            return Err(GpuError::shape(
                "Head::with_m",
                format!("need 1 <= m <= 8, got m={m}"),
            ));
        }
        let stream = gpu.stream();
        let hidden = head_gain(w)?.len();
        let (out_w, k) = head_out_w(w)?;
        if k != hidden {
            return Err(GpuError::shape(
                "Head::new",
                format!(
                    "output.weight rows are {k} values wide, the norm \
                 gain is {hidden}"
                ),
            ));
        }
        let n_vocab = out_w.rows();
        if n_vocab == 0 {
            return Err(GpuError::shape("Head::new", "output.weight has no rows"));
        }
        Ok(Head {
            eps,
            hidden,
            n_vocab,
            m,
            x: DeviceBuffer::zeroed(stream, m * hidden)?,
            normed: DeviceBuffer::zeroed(stream, m * hidden)?,
            act: Q8Act::with_k(stream, m, hidden)?,
            logits: DeviceBuffer::zeroed(stream, m * n_vocab)?,
            token_out: DeviceBuffer::from_host(stream, &vec![0u32; m + 1])?,
            graph: None,
        })
    }

    /// The hidden width the head's norm and projection take.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The vocabulary size, the logit count the projection writes per row.
    pub fn n_vocab(&self) -> usize {
        self.n_vocab
    }

    /// The rows this head was built for.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Device bytes of the scratch (the weights are counted by their owner).
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.x.num_bytes()
            + self.normed.num_bytes()
            + self.logits.num_bytes()
            + self.token_out.num_bytes()
            + self.act.q3.num_bytes()
            + self.act.q4.num_bytes()
            + self.act.q6.num_bytes()
            + self.act.s8.num_bytes()
            + self.act.d8.num_bytes()
    }

    /// Enqueue the whole head for the rows in `x`: rms_norm → quantize_q8_1
    /// → gemv_q6k → argmax (`argmax_rows` when `m > 1`), the argmax copying
    /// the fault word after the tokens. Pure enqueues — no allocation, no
    /// synchronization — so the same body is what `capture` records.
    pub fn enqueue(&mut self, gpu: &Gpu, w: &Weights) -> Result<(), GpuError> {
        let stream = gpu.stream();
        gpu.elem().enqueue_rms_norm(
            stream,
            &self.x,
            head_gain(w)?,
            self.eps,
            self.hidden,
            self.m,
            &mut self.normed,
        )?;
        gpu.enqueue_quantize_q8_1_head(&self.normed, &mut self.act)?;
        gpu.enqueue_gemv_q6k(head_out_w(w)?.0, &self.act, &mut self.logits)?;
        let fault = gpu.fault_sink(LAYER_HEAD);
        if self.m == 1 {
            gpu.elem().enqueue_argmax_fault(
                stream,
                &self.logits,
                self.n_vocab,
                fault,
                &mut self.token_out,
            )?;
        } else {
            gpu.elem().enqueue_argmax_rows_fault(
                stream,
                &self.logits,
                self.n_vocab,
                self.m,
                fault,
                &mut self.token_out,
            )?;
        }
        Ok(())
    }

    /// Enqueue the head with its projection and argmax handed to `tail`, for
    /// an architecture whose chain folds the two into one launch: the same
    /// rms_norm and q8_1 quantization [`Head::enqueue`] runs, then `tail(act,
    /// out_w, fault, logits, token_out)` over the quantized rows, the Q6_K
    /// lm_head, the head's fault sink and the two outputs the readbacks read
    /// (`logits` as `logits_to_host` takes them, `token_out` as the `m` tokens
    /// then the fault word). `tail` must write both as `enqueue`'s gemv and
    /// argmax do; [`Head::token`] and [`Head::logits_to_host`] are unchanged.
    /// Pure enqueues, so the same body is what a capture records.
    pub(crate) fn enqueue_with_tail<F>(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        tail: F,
    ) -> Result<(), GpuError>
    where
        F: FnOnce(
            &Q8Act,
            &DeviceTensor<u32>,
            FaultSink,
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<u32>,
        ) -> Result<(), GpuError>,
    {
        gpu.elem().enqueue_rms_norm(
            gpu.stream(),
            &self.x,
            head_gain(w)?,
            self.eps,
            self.hidden,
            self.m,
            &mut self.normed,
        )?;
        gpu.enqueue_quantize_q8_1_head(&self.normed, &mut self.act)?;
        tail(
            &self.act,
            head_out_w(w)?.0,
            gpu.fault_sink(LAYER_HEAD),
            &mut self.logits,
            &mut self.token_out,
        )
    }

    /// Capture `enqueue` over the resident buffers into this head's graph and
    /// return the node count. The buffers' addresses freeze — they were
    /// allocated at `new`.
    pub fn capture(&mut self, gpu: &Gpu, w: &Weights) -> Result<usize, GpuError> {
        let graph = gpu.capture(|_| self.enqueue(gpu, w))?;
        let nodes = graph.node_count();
        self.graph = Some(graph);
        Ok(nodes)
    }

    /// Enqueue one replay of the captured graph — no input write, no
    /// synchronization; the caller reads back what it needs.
    pub fn launch(&self, gpu: &Gpu) -> Result<(), GpuError> {
        self.graph
            .as_ref()
            .ok_or(GpuError::state("Head::launch", "no captured graph"))?
            .launch(gpu.stream())
    }

    /// Write the chain's input from the host. Synchronizing copy on the
    /// engine stream; never inside a capture.
    pub fn set_input(&mut self, gpu: &Gpu, host: &[f32]) -> Result<(), GpuError> {
        if host.len() != self.m * self.hidden {
            return Err(GpuError::shape(
                "Head::set_input",
                format!(
                    "{} values, the head takes {} rows of {}",
                    host.len(),
                    self.m,
                    self.hidden
                ),
            ));
        }
        self.x.copy_from_host(gpu.stream(), host)?;
        Ok(())
    }

    /// The chain's input buffer — the write target of the step's last launch
    /// before the head, so that launch's output lands directly in the head.
    /// Public because an architecture crate's chain writes it too.
    pub fn input_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.x
    }

    /// The normed rows of the last run (the dump's `result_norm` tap).
    /// Blocking read; gate/debug use.
    pub fn normed_to_host(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        Ok(self.normed.to_host_vec(gpu.stream())?)
    }

    /// The logits of the last run (`n_vocab * m` f32, `[v*m + c]`). Blocking
    /// read; gate/debug use.
    pub fn logits_to_host(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        Ok(self.logits.to_host_vec(gpu.stream())?)
    }

    /// The argmax token of the last run's first row: a blocking read — the
    /// decode step's single synchronize-shaped retrieval. A fault raised by
    /// any launch before the argmax is [`GpuError::Fault`], not a token.
    pub fn token(&self, gpu: &Gpu) -> Result<u32, GpuError> {
        Ok(self.tokens(gpu)?[0])
    }

    /// The argmax tokens of the last run, one per row. Blocking read; a
    /// raised fault word is [`GpuError::Fault`], as [`Head::token`].
    pub fn tokens(&self, gpu: &Gpu) -> Result<Vec<u32>, GpuError> {
        let mut out = self.token_out.to_host_vec(gpu.stream())?;
        let word = out.pop().ok_or(GpuError::state(
            "Head::tokens",
            "the readback buffer holds no fault word",
        ))?;
        if let Some(fault) = Fault::from_word(word) {
            return Err(GpuError::Fault {
                what: "Head::tokens",
                fault,
            });
        }
        Ok(out)
    }
}
