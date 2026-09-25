//! qwen3moe's prompt prefill: a prompt fed [`MAX_TOKENS`] positions per pass
//! instead of one decode step per token. A pass is the decode chain's own
//! layer body at `m` rows (`dispatch::enqueue_pass`), and every launch in it
//! computes each token's values the way the one-row launch does: the gemv,
//! norm and quantizer bodies take a column index; the flash runs each query
//! row against its own live key count; the router routes each token with its
//! own warp; the experts' gate·up dots each token's column, and the down
//! `_sel` runs every token's slots, each against its own column. So a
//! prefilled prompt leaves the K/V rows, and the head the logits, that
//! feeding it one step per token leaves.
//!
//! In graph mode ([`StepMode::Graph`]) a pass is a replay of the pass of its
//! `m` tokens, captured once ([`GpuModel::capture_prefill`], or by the first
//! prompt that needs it); in eager mode the same enqueues run one by one —
//! the twin every replay is gated against. The head after the last pass runs
//! eager either way. The decode step's graph stays valid beside these: the
//! two share the weights and the cache planes and nothing else.
//!
//! Inputs: the prompt's image — one [`BLOCK`] per pass of ids, positions,
//! live key counts and rope rows — is written and copied to the card once
//! per prompt. An eager pass reads its block where it lies. A captured pass
//! reads the fixed-address slot instead, and each pass's block is copied
//! into the slot in stream order right before its replay, so a replay reads
//! this pass's tokens, not the ones it was captured over. The arena, the
//! image (room for a prompt as long as the cache) and the slot are allocated
//! at load; a prompt allocates nothing.

use super::body::Body;
use super::dispatch::{self, PassCtx};
use super::router::MAX_TOKENS;
use super::scratch::{Arena, Dims, Io, param_view};
use crate::flash_gqa::HEAD;
use crate::model::{GpuModel, StepMode};
use crate::rope_table::{Direction, RopeTable};
use crate::weights::Weights;
use crate::{Gpu, GpuError, Graph, NodeInfo, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use std::mem::ManuallyDrop;

/// Element offsets into one pass's block, image and slot alike:
/// [`MAX_TOKENS`] ids, positions and live key counts (u32 each), then
/// [`MAX_TOKENS`] rope rows of [`HEAD`] f32 as bits. A pass of `m` tokens
/// fills the first `m` of each and reads only those.
const B_TOKENS: usize = 0;
const B_POS: usize = MAX_TOKENS;
const B_N_KEYS: usize = 2 * MAX_TOKENS;
const B_CS: usize = 3 * MAX_TOKENS;
/// u32 words of one block.
const BLOCK: usize = B_CS + MAX_TOKENS * HEAD;

/// The prefill arena, the current prompt's image, the slot the captured
/// passes read and the passes themselves.
pub(super) struct Prefill {
    /// The pass of `m` tokens captured over the slot, at `m − 1`. Declared
    /// first: fields drop in declaration order, and a graph is destroyed
    /// while the arena and the slot it addresses are alive (and, since
    /// `Body` declares this struct first, the cache planes too).
    graphs: Vec<Option<Graph>>,
    pub(super) a: Arena,
    /// Pass `c`'s block of the current prompt at `c · BLOCK`, for as many
    /// passes as a prompt of the cache's height takes.
    image: DeviceBuffer<u32>,
    image_host: Vec<u32>,
    /// One position's rope row, reused for every token of a prompt.
    cs_host: Vec<f32>,
    /// The block every captured pass reads.
    slot: DeviceBuffer<u32>,
    /// The captured passes' windows onto the slot, `m` tokens at `m − 1`.
    slot_windows: Vec<Windows>,
    /// Tokens of the prompt the image holds.
    len: usize,
    /// The pass whose block the slot holds, copied and not yet replayed.
    staged: Option<usize>,
}

/// One pass's windows onto a block, owned so that the arena can be borrowed
/// beside them.
pub(super) struct Windows {
    tokens: ManuallyDrop<DeviceBuffer<u32>>,
    pos: ManuallyDrop<DeviceBuffer<u32>>,
    n_keys: ManuallyDrop<DeviceBuffer<u32>>,
    cs: ManuallyDrop<DeviceBuffer<f32>>,
}

impl Windows {
    /// The first `m` entries of each array of the block at element `base`
    /// of `parent`.
    ///
    /// # Safety
    ///
    /// `base + BLOCK <= parent.len()`, `m <= MAX_TOKENS`, and `parent`
    /// outlives the windows unmoved in memory (a captured graph bakes the
    /// addresses in).
    unsafe fn over(parent: &DeviceBuffer<u32>, base: usize, m: usize) -> Windows {
        // SAFETY: each window lies inside the block (`B_*` layout, `m <=
        // MAX_TOKENS`), which lies inside `parent` — the caller's contract.
        unsafe {
            Windows {
                tokens: param_view::<u32>(parent, base + B_TOKENS, m),
                pos: param_view::<u32>(parent, base + B_POS, m),
                n_keys: param_view::<u32>(parent, base + B_N_KEYS, m),
                cs: param_view::<f32>(parent, base + B_CS, m * HEAD),
            }
        }
    }

    pub(super) fn io(&self) -> Io<'_> {
        Io {
            tokens: &self.tokens,
            pos: &self.pos,
            n_keys: &self.n_keys,
            cs: &self.cs,
        }
    }
}

const WHAT: &str = "qwen3moe::prefill";

impl Prefill {
    /// The arena for [`MAX_TOKENS`] rows of `d`, an image for a prompt of
    /// `ctx_max` tokens, the slot and its windows, no pass captured.
    /// Load-time only.
    pub(super) fn new(stream: &CudaStream, d: Dims, ctx_max: usize) -> Result<Prefill, GpuError> {
        let blocks = ctx_max.div_ceil(MAX_TOKENS);
        let slot = DeviceBuffer::zeroed(stream, BLOCK)?;
        // SAFETY: the slot is one BLOCK and every `m <= MAX_TOKENS`; the slot
        // moves into the struct beside its windows (a move of the handle,
        // not of the allocation), where it outlives them.
        let slot_windows = unsafe {
            (1..=MAX_TOKENS)
                .map(|m| Windows::over(&slot, 0, m))
                .collect()
        };
        Ok(Prefill {
            graphs: (0..MAX_TOKENS).map(|_| None).collect(),
            a: Arena::new(stream, d, MAX_TOKENS)?,
            image: DeviceBuffer::zeroed(stream, blocks * BLOCK)?,
            image_host: Vec::with_capacity(blocks * BLOCK),
            cs_host: Vec::with_capacity(HEAD),
            slot,
            slot_windows,
            len: 0,
            staged: None,
        })
    }

    /// Write the image of `tokens` at positions `pos0 ..` and copy it to the
    /// card. Synchronizes (the copy of a borrowed host slice does). A prompt
    /// of more passes than the image holds is refused.
    fn write(
        &mut self,
        stream: &CudaStream,
        rope: &RopeTable,
        tokens: &[u32],
        pos0: u32,
    ) -> Result<(), GpuError> {
        launch_u32(WHAT, "tokens", tokens.len())?;
        let passes = tokens.len().div_ceil(MAX_TOKENS);
        let words = passes * BLOCK;
        if words > self.image.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} tokens take {passes} passes; the image holds {}",
                    tokens.len(),
                    self.image.len() / BLOCK
                ),
            ));
        }
        self.staged = None;
        self.image_host.clear();
        self.image_host.resize(words, 0);
        let mut pos = pos0;
        for (chunk, run) in tokens.chunks(MAX_TOKENS).enumerate() {
            let block = &mut self.image_host[chunk * BLOCK..(chunk + 1) * BLOCK];
            for (t, &token) in run.iter().enumerate() {
                block[B_TOKENS + t] = token;
                block[B_POS + t] = pos;
                block[B_N_KEYS + t] = pos + 1;
                self.cs_host.clear();
                rope.push(pos, Direction::Forward, &mut self.cs_host);
                for (dst, v) in block[B_CS + t * HEAD..][..HEAD]
                    .iter_mut()
                    .zip(&self.cs_host)
                {
                    *dst = v.to_bits();
                }
                pos += 1;
            }
        }
        // SAFETY: `words <= self.image.len()` (checked above), so the window
        // is inside it; it lives for this copy alone.
        let mut dst = unsafe { param_view::<u32>(&self.image, 0, words) };
        dst.copy_from_host(stream, &self.image_host)?;
        self.len = tokens.len();
        Ok(())
    }

    /// Err unless pass `chunk` of `m` tokens is a pass of the prompt in the
    /// image: `m` in `1..=MAX_TOKENS` and its tokens inside the prompt.
    fn check_pass(&self, chunk: usize, m: usize) -> Result<(), GpuError> {
        let t0 = chunk * MAX_TOKENS;
        if m == 0 || m > MAX_TOKENS || t0 + m > self.len {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "pass {chunk} of {m} tokens is not a pass of the {}-token prompt in the image \
                     (1..={MAX_TOKENS} tokens a pass)",
                    self.len
                ),
            ));
        }
        Ok(())
    }

    /// Pass `chunk`'s windows onto its block of the image, for an eager pass.
    fn image_windows(&self, chunk: usize, m: usize) -> Result<Windows, GpuError> {
        self.check_pass(chunk, m)?;
        // SAFETY: `check_pass` puts the pass inside the prompt, whose
        // `ceil(len / MAX_TOKENS)` blocks `write` checked against the image,
        // so block `chunk` ends inside it; the windows live for one pass,
        // while the image stays in place.
        Ok(unsafe { Windows::over(&self.image, chunk * BLOCK, m) })
    }

    /// Copy pass `chunk`'s block of the image into the slot, in stream order
    /// behind every launch before it — the replay that follows reads it.
    fn stage(&mut self, stream: &CudaStream, chunk: usize, m: usize) -> Result<(), GpuError> {
        self.check_pass(chunk, m)?;
        // SAFETY: block `chunk` lies inside the image (`image_windows`'s
        // argument); the window lives for this enqueue, and the image stays
        // in place until the copy has run (it is reallocated only at load).
        let src = unsafe { param_view::<u32>(&self.image, chunk * BLOCK, BLOCK) };
        self.slot.copy_from_device_async(&src, stream)?;
        self.staged = Some(chunk);
        Ok(())
    }

    /// Replay the captured pass of `m` tokens for pass `chunk`, whose block
    /// must be the one staged since the last replay.
    fn replay(&mut self, stream: &CudaStream, chunk: usize, m: usize) -> Result<(), GpuError> {
        self.check_pass(chunk, m)?;
        if self.staged != Some(chunk) {
            return Err(GpuError::state(
                WHAT,
                "this pass's block in the slot (stage it before the replay)",
            ));
        }
        let graph = self.graphs[m - 1]
            .as_ref()
            .ok_or(GpuError::state(WHAT, "a captured pass of this many tokens"))?;
        graph.launch(stream)?;
        self.staged = None;
        Ok(())
    }

    /// Device bytes of the arena, the image and the slot.
    pub(super) fn bytes(&self) -> usize {
        self.a.bytes() + self.image.num_bytes() + self.slot.num_bytes()
    }
}

/// Err unless `m` is a pass's token count, `1..=MAX_TOKENS`.
fn check_m(m: usize) -> Result<(), GpuError> {
    if m == 0 || m > MAX_TOKENS {
        return Err(GpuError::shape(
            WHAT,
            format!("a pass of {m} tokens (1..={MAX_TOKENS})"),
        ));
    }
    Ok(())
}

impl Body {
    /// Capture the pass of `m` tokens over the slot and keep it, its node
    /// count checked against the launches the pass makes
    /// (`dispatch::pass_launches`). Returns the node count.
    fn capture_pass(&mut self, gpu: &Gpu, w: &Weights, m: usize) -> Result<usize, GpuError> {
        check_m(m)?;
        let Body {
            hp,
            names,
            kv,
            k,
            mma,
            prefill,
            ..
        } = self;
        let c = PassCtx {
            gpu,
            w,
            names,
            k,
            mma: *mma,
            eps: hp.rms_eps,
        };
        let Prefill {
            graphs,
            a,
            slot_windows,
            ..
        } = prefill;
        let io = slot_windows[m - 1].io();
        let graph = gpu.capture(|_| dispatch::enqueue_pass(&c, kv, a, &io, m))?;
        let (nodes, launches) = (graph.node_count(), dispatch::pass_launches(names, m));
        if nodes != launches {
            return Err(GpuError::shape(
                WHAT,
                format!("the pass of {m} tokens captured {nodes} nodes for {launches} launches"),
            ));
        }
        graphs[m - 1] = Some(graph);
        Ok(nodes)
    }

    /// Run pass `chunk` of `m` tokens of the prompt in the image: with
    /// `graph`, its block copied into the slot and the captured pass of `m`
    /// replayed behind it; without, the same enqueues reading the block
    /// where it lies.
    fn run_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        chunk: usize,
        m: usize,
        graph: bool,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        if graph {
            self.prefill.stage(stream, chunk, m)?;
            return self.prefill.replay(stream, chunk, m);
        }
        let win = self.prefill.image_windows(chunk, m)?;
        let Body {
            hp,
            names,
            kv,
            k,
            mma,
            prefill,
            ..
        } = self;
        let c = PassCtx {
            gpu,
            w,
            names,
            k,
            mma: *mma,
            eps: hp.rms_eps,
        };
        dispatch::enqueue_pass(&c, kv, &mut prefill.a, &win.io(), m)
    }
}

impl GpuModel<Body> {
    /// Feed `tokens` through the chain [`MAX_TOKENS`] positions per pass
    /// (module doc) and return the greedy next token after the last one —
    /// what [`GpuModel::step`] returns for the same tokens from the same
    /// position, and the same cache rows behind it. Positions continue from
    /// wherever the model stands. In graph mode a pass size not captured yet
    /// is captured here, inside the call; the layer taps must be off.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT, "empty token slice"));
        }
        let pos0 = self.pos();
        self.check_pos(pos0 + launch_u32(WHAT, "tokens", tokens.len())? - 1, WHAT)?;
        let graph = self.mode() == StepMode::Graph;
        let passes = tokens.len().div_ceil(MAX_TOKENS);
        {
            let (gpu, w, body) = self.body_parts(WHAT)?;
            if body.taps.is_some() {
                return Err(GpuError::state(WHAT, "layer taps off (a pass writes none)"));
            }
            let Body { prefill, rope, .. } = body;
            prefill.write(gpu.stream(), rope, tokens, pos0)?;
            if graph {
                let last = tokens.len() - (passes - 1) * MAX_TOKENS;
                let full = (passes > 1).then_some(MAX_TOKENS);
                for m in full.into_iter().chain([last]) {
                    if body.prefill.graphs[m - 1].is_none() {
                        body.capture_pass(gpu, w, m)?;
                    }
                }
            }
        }
        let mut next = None;
        for (chunk, run) in tokens.chunks(MAX_TOKENS).enumerate() {
            let (m, last) = (run.len(), chunk + 1 == passes);
            next = self.run_rows(m, WHAT, |gpu, w, body, head, _| {
                body.run_pass(gpu, w, chunk, m, graph)?;
                if last {
                    let Body {
                        k,
                        head_state,
                        prefill,
                        ..
                    } = body;
                    dispatch::enqueue_pass_head(gpu, w, k, head_state, &prefill.a, m, head)?;
                }
                Ok(last)
            })?;
        }
        next.ok_or(GpuError::state(WHAT, "a token read after the last pass"))
    }

    /// Capture the prefill pass of every token count in `1..=MAX_TOKENS` not
    /// captured yet, and return each pass's node count, `m` tokens at `m −
    /// 1`. [`GpuModel::prefill`] in graph mode captures a missing count
    /// itself, inside its call; a timed caller captures here first.
    pub fn capture_prefill(&mut self) -> Result<Vec<usize>, GpuError> {
        let (gpu, w, body) = self.body_parts("qwen3moe::capture_prefill")?;
        for m in 1..=MAX_TOKENS {
            if body.prefill.graphs[m - 1].is_none() {
                body.capture_pass(gpu, w, m)?;
            }
        }
        Ok(body
            .prefill
            .graphs
            .iter()
            .map(|g| g.as_ref().map_or(0, Graph::node_count))
            .collect())
    }

    /// Every node of the captured prefill pass of `m` tokens, as the driver
    /// lists them.
    pub fn prefill_graph_nodes(&self, m: usize) -> Result<Vec<NodeInfo>, GpuError> {
        const WHAT_NODES: &str = "qwen3moe::prefill_graph_nodes";
        check_m(m)?;
        self.body(WHAT_NODES)?.prefill.graphs[m - 1]
            .as_ref()
            .ok_or(GpuError::state(
                WHAT_NODES,
                "a captured pass of this many tokens",
            ))?
            .nodes()
    }

    /// The passes [`GpuModel::prefill`] takes for a prompt of `tokens` ids.
    #[must_use]
    pub fn prefill_passes(tokens: usize) -> usize {
        tokens.div_ceil(MAX_TOKENS)
    }
}
