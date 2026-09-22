//! The GPU output head: `result_norm → lm_head (Q6_K) → argmax` as one
//! capturable sequence over resident scratch (docs/gpu-design.md decisions 4
//! and 6). No kernels of its own — the chain is the gated `rms_norm`, the
//! shared q8_1 quantizer, the Q6_K gemv and `argmax` exactly as the block
//! path launches them, so their gates are this chain's gates.
//!
//! m = 1 only: the decode step samples the last position, and the lm_head is
//! the one site whose row count is the vocabulary. The input buffer is
//! exposed mutable so the assembly round can have the last block's residual
//! store write straight into it, and `set_input` is the gate's way in.

use crate::GpuError;
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
                return Err(format!(
                    "head_out_w: output.weight is {ty}, want Q6_K (the gemv \
                     this head launches reads Q6_K rows)"
                )
                .into());
            }
            Ok((w, *k))
        }
        Some(_) => Err("head_out_w: output.weight is not a K-quant word plane".into()),
        None => Err(
            "head_out_w: output.weight not resident — load Weights with \
             globals"
                .into(),
        ),
    }
}

/// The resident output head for m = 1: input `hidden` f32, the normed vector,
/// the q8_1 quantization of it, `n_vocab` logits and the argmax token. Every
/// buffer is allocated at `new`; `enqueue` and a graph replay touch addresses
/// only, so one captured graph serves every input written into `x`.
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
    /// The chain's input — the last block's output vector. The assembly round
    /// writes here through `input_mut`; the gate through `set_input`.
    x: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    act: Q8Act,
    logits: DeviceBuffer<f32>,
    token_out: DeviceBuffer<u32>,
}

impl Head {
    /// Cross-check the two weights against each other and allocate the
    /// scratch. Load-time only.
    pub fn new(gpu: &Gpu, w: &Weights, eps: f32) -> Result<Head, GpuError> {
        let stream = gpu.stream();
        let hidden = head_gain(w)?.len();
        let (out_w, k) = head_out_w(w)?;
        if k != hidden {
            return Err(format!(
                "Head::new: output.weight rows are {k} values wide, the norm \
                 gain is {hidden}"
            )
            .into());
        }
        let n_vocab = out_w.rows();
        if n_vocab == 0 {
            return Err("Head::new: output.weight has no rows".into());
        }
        Ok(Head {
            eps,
            hidden,
            n_vocab,
            x: DeviceBuffer::zeroed(stream, hidden)?,
            normed: DeviceBuffer::zeroed(stream, hidden)?,
            act: Q8Act::with_k(stream, 1, hidden)?,
            logits: DeviceBuffer::zeroed(stream, n_vocab)?,
            token_out: DeviceBuffer::from_host(stream, &[0u32])?,
            graph: None,
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn n_vocab(&self) -> usize {
        self.n_vocab
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

    /// Enqueue the whole head for the vector in `x`: rms_norm → quantize_q8_1
    /// → gemv_q6k → argmax. Pure enqueues — no allocation, no synchronization
    /// — so the same body is what `capture` records.
    pub fn enqueue(&mut self, gpu: &Gpu, w: &Weights) -> Result<(), GpuError> {
        assert_eq!(self.act.m(), 1, "Head: the decode head runs m = 1 only");
        let stream = gpu.stream();
        gpu.elem().enqueue_rms_norm(
            stream,
            &self.x,
            head_gain(w)?,
            self.eps,
            self.hidden,
            1,
            &mut self.normed,
        )?;
        gpu.enqueue_quantize_q8_1(&self.normed, &mut self.act)?;
        gpu.enqueue_gemv_q6k(head_out_w(w)?.0, &self.act, &mut self.logits)?;
        gpu.elem()
            .enqueue_argmax(stream, &self.logits, self.n_vocab, &mut self.token_out)?;
        Ok(())
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
            .ok_or("Head::launch: no captured graph")?
            .launch(gpu.stream())
    }

    /// Write the chain's input from the host. Synchronizing copy on the
    /// engine stream; never inside a capture.
    pub fn set_input(&mut self, gpu: &Gpu, host: &[f32]) -> Result<(), GpuError> {
        if host.len() != self.hidden {
            return Err(format!(
                "Head::set_input: {} values, the head takes {}",
                host.len(),
                self.hidden
            )
            .into());
        }
        self.x.copy_from_host(gpu.stream(), host)?;
        Ok(())
    }

    /// The chain's input buffer — the assembly round's write target, so the
    /// last block's residual add lands directly in the head.
    pub fn input_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.x
    }

    /// The normed vector of the last run (the dump's `result_norm` tap).
    /// Blocking read; gate/debug use.
    pub fn normed_to_host(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        Ok(self.normed.to_host_vec(gpu.stream())?)
    }

    /// The logits of the last run (`n_vocab` f32). Blocking read; gate/debug
    /// use.
    pub fn logits_to_host(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        Ok(self.logits.to_host_vec(gpu.stream())?)
    }

    /// The argmax token of the last run: a blocking read of one u32 — the
    /// step's single synchronize-shaped retrieval.
    pub fn token(&self, gpu: &Gpu) -> Result<u32, GpuError> {
        Ok(self.token_out.to_host_vec(gpu.stream())?[0])
    }
}
