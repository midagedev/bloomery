//! qwen3moe's prompt prefill: a prompt fed several positions per launch
//! instead of one decode step per token, by one of two paths
//! ([`PrefillPlan`]). A prompt that fits one pass takes the pass path; a
//! longer one runs as GEMM ubatches of up to the model's ubatch size
//! (`super::ubatch`: at most [`UBATCH`](super::ubatch::UBATCH), set at
//! load, changed by [`GpuModel::set_ubatch`]), every weight read once per
//! ubatch, and a tail of at most one pass after them takes the pass path
//! again. At nine tokens a
//! ubatch reads fewer expert bytes than two passes and its dense GEMMs cost
//! less than a pass's launches, so the cut sits there. The GEMM path sums
//! its products in another order than the one-token path and is gated by a
//! band; the pass path is bit-equal to it.
//!
//! The pass path feeds [`MAX_TOKENS`] positions per pass. A pass is the decode chain's own
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
//! Inputs: the prompt's image — one [`BLOCK`] per pass, the pass's input
//! record (the position of its first token, then its ids) — is written into
//! pinned host words and copied to the card once per prompt, asynchronously.
//! A pass's embedding launch derives its rows' positions and live key counts
//! from it on the card. An eager pass reads its block where it lies. A
//! captured pass reads the fixed-address slot instead, and each pass's block
//! is copied into the slot in stream order right before its replay, so a
//! replay reads this pass's tokens, not the ones it was captured over. The
//! arena, the image (room for a prompt as long as the cache) and the slot
//! are allocated at load; a prompt allocates nothing.

use super::body::Body;
use super::dispatch::{self, PassCtx};
use super::router::MAX_TOKENS;
use super::scratch::{Arena, Dims, IN_IDS, IN_POS0, Inbox, Io, param_view, put_input};
use super::ubatch::UbCtx;
use crate::model::{GpuModel, StepMode};
use crate::weights::Weights;
use crate::{Gpu, GpuError, Graph, NodeInfo, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::Range;

/// u32 words of one pass's block, image and slot alike: an input record of
/// [`MAX_TOKENS`] ids. A pass of `m` tokens fills the first `m + 1` words
/// and its launches read only those.
const BLOCK: usize = IN_IDS + MAX_TOKENS;

/// The prefill arena, the current prompt's image, the slot the captured
/// passes read and the passes themselves.
pub(super) struct Prefill {
    /// The pass of `m` tokens captured over the slot, at `m − 1`. Declared
    /// first: fields drop in declaration order, and a graph is destroyed
    /// while the arena and the slot it addresses are alive (and, since
    /// `Body` declares this struct first, the cache planes and the rope
    /// table too).
    graphs: Vec<Option<Graph>>,
    pub(super) a: Arena,
    /// The captured passes' windows onto the slot, `m` tokens at `m − 1`.
    /// Declared before the slot, so they drop first (a window frees nothing
    /// either way).
    slot_windows: Vec<Windows>,
    /// The block every captured pass reads.
    slot: DeviceBuffer<u32>,
    /// Pass `c`'s block of the current prompt at `c · BLOCK`, for as many
    /// passes as a prompt of the cache's height takes.
    image: Inbox,
    /// Tokens of the prompt the image holds.
    len: usize,
    /// The pass whose block the slot holds, copied and not yet replayed.
    staged: Option<usize>,
}

/// One pass's windows onto a block, owned so that the arena can be borrowed
/// beside them.
pub(super) struct Windows {
    ids: ManuallyDrop<DeviceBuffer<u32>>,
    pos0: ManuallyDrop<DeviceBuffer<u32>>,
}

impl Windows {
    /// The pass of `m` tokens' windows onto the block at element `base` of
    /// `parent`: its first `m` ids and its position word.
    ///
    /// # Safety
    ///
    /// `base + BLOCK <= parent.len()`, `m <= MAX_TOKENS`, and `parent`
    /// outlives the windows unmoved in memory (a captured graph bakes the
    /// addresses in).
    unsafe fn over(parent: &DeviceBuffer<u32>, base: usize, m: usize) -> Windows {
        // SAFETY: each window lies inside the block (`IN_POS0`, `IN_IDS` + `m`
        // <= BLOCK), which lies inside `parent` — the caller's contract.
        unsafe {
            Windows {
                ids: param_view::<u32>(parent, base + IN_IDS, m),
                pos0: param_view::<u32>(parent, base + IN_POS0, 1),
            }
        }
    }

    pub(super) fn io(&self) -> Io<'_> {
        Io {
            ids: &self.ids,
            pos0: &self.pos0,
            first: 0,
            lane: None,
        }
    }
}

const WHAT: &str = "qwen3moe::prefill";

impl Prefill {
    /// The arena for [`MAX_TOKENS`] rows of `d`, an image for a prompt as
    /// long as the cache (`d.ctx` tokens), the slot and its windows, no pass
    /// captured. Load-time only.
    pub(super) fn new(stream: &CudaStream, d: Dims) -> Result<Prefill, GpuError> {
        let blocks = d.ctx.div_ceil(MAX_TOKENS);
        let slot = DeviceBuffer::zeroed(stream, BLOCK)?;
        // SAFETY: the slot is one BLOCK and every `m <= MAX_TOKENS`; the slot
        // moves into the struct beside its windows (a move of the handle,
        // not of the allocation), declared after them, so it outlives them.
        let slot_windows = unsafe {
            (1..=MAX_TOKENS)
                .map(|m| Windows::over(&slot, 0, m))
                .collect()
        };
        Ok(Prefill {
            graphs: (0..MAX_TOKENS).map(|_| None).collect(),
            a: Arena::new(stream, d, MAX_TOKENS)?,
            slot_windows,
            slot,
            image: Inbox::new(stream, blocks * BLOCK)?,
            len: 0,
            staged: None,
        })
    }

    /// Write the image of `tokens` at positions `pos0 ..` — pass `c`'s block
    /// the record of its tokens at `pos0 + c · MAX_TOKENS` — and enqueue its
    /// copy to the card. Asynchronous: the passes behind it read it. A prompt
    /// of more passes than the image holds is refused.
    fn write(&mut self, stream: &CudaStream, tokens: &[u32], pos0: u32) -> Result<(), GpuError> {
        launch_u32(WHAT, "tokens", tokens.len())?;
        let passes = tokens.len().div_ceil(MAX_TOKENS);
        let held = self.image.dev().len() / BLOCK;
        if passes > held {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} tokens take {passes} passes; the image holds {held}",
                    tokens.len()
                ),
            ));
        }
        self.staged = None;
        let host = self.image.host_mut()?;
        let mut words = 0;
        for (chunk, run) in tokens.chunks(MAX_TOKENS).enumerate() {
            let first = launch_u32(WHAT, "a pass's first token", chunk * MAX_TOKENS)?;
            let at = pos0.checked_add(first).ok_or_else(|| {
                GpuError::shape(WHAT, format!("position {pos0} + {first} past a u32"))
            })?;
            put_input(&mut host[chunk * BLOCK..], run, at)?;
            words = chunk * BLOCK + IN_IDS + run.len();
        }
        self.image.upload(stream, words)?;
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
        Ok(unsafe { Windows::over(self.image.dev(), chunk * BLOCK, m) })
    }

    /// Copy pass `chunk`'s block of the image into the slot, in stream order
    /// behind every launch before it — the replay that follows reads it.
    fn stage(&mut self, stream: &CudaStream, chunk: usize, m: usize) -> Result<(), GpuError> {
        self.check_pass(chunk, m)?;
        // SAFETY: block `chunk` lies inside the image (`image_windows`'s
        // argument); the window lives for this enqueue, and the image stays
        // in place until the copy has run (it is reallocated only at load).
        let src = unsafe { param_view::<u32>(self.image.dev(), chunk * BLOCK, BLOCK) };
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
        self.a.bytes() + self.image.bytes() + self.slot.num_bytes()
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
            plans,
            kv,
            rope,
            k,
            mma,
            prefill,
            ..
        } = self;
        let c = PassCtx {
            gpu,
            w,
            plans,
            k,
            mma: *mma,
            eps: hp.rms_eps,
            table: &rope.table,
        };
        let Prefill {
            graphs,
            a,
            slot_windows,
            ..
        } = prefill;
        let io = slot_windows[m - 1].io();
        let graph = gpu.capture(|_| dispatch::enqueue_pass(&c, kv, a, &io, m))?;
        let (nodes, launches) = (graph.node_count(), dispatch::pass_launches(plans, m));
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
            plans,
            kv,
            rope,
            k,
            mma,
            prefill,
            ..
        } = self;
        let c = PassCtx {
            gpu,
            w,
            plans,
            k,
            mma: *mma,
            eps: hp.rms_eps,
            table: &rope.table,
        };
        dispatch::enqueue_pass(&c, kv, &mut prefill.a, &win.io(), m)
    }
}

/// Which path a prompt's positions take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillPath {
    /// The cut in the module doc: a prompt of at most [`MAX_TOKENS`] tokens
    /// takes one pass; a longer one GEMM ubatches of up to the ubatch size,
    /// and a tail of at most [`MAX_TOKENS`] after them one pass.
    Auto,
    /// Passes of up to [`MAX_TOKENS`] positions, the whole prompt: the path
    /// that is bit-equal to one step per token.
    Pass,
    /// GEMM ubatches of up to the ubatch size, the whole prompt.
    Gemm,
}

/// One launch unit of a prompt: a GEMM ubatch or a pass, and its tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillStep {
    Ubatch(usize),
    Pass(usize),
}

/// The units a prompt runs as, in order: the ubatches, then the passes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefillPlan {
    pub steps: Vec<PrefillStep>,
}

impl PrefillPlan {
    /// The plan of a prompt of `tokens` ids by `path` on a model whose
    /// ubatch size is `ubatch` ([`GpuModel::prefill_plan`] passes the
    /// model's).
    #[must_use]
    pub fn new(tokens: usize, path: PrefillPath, ubatch: NonZeroUsize) -> PrefillPlan {
        let ub = ubatch.get();
        let chunked = |n: usize, w: usize, f: fn(usize) -> PrefillStep| {
            (0..n.div_ceil(w)).map(move |i| f(w.min(n - i * w)))
        };
        let steps = match path {
            PrefillPath::Pass => chunked(tokens, MAX_TOKENS, PrefillStep::Pass).collect(),
            PrefillPath::Gemm => chunked(tokens, ub, PrefillStep::Ubatch).collect(),
            PrefillPath::Auto if tokens <= MAX_TOKENS => {
                chunked(tokens, MAX_TOKENS, PrefillStep::Pass).collect()
            }
            PrefillPath::Auto => {
                let tail = tokens % ub;
                let mut v: Vec<PrefillStep> =
                    chunked(tokens - tail, ub, PrefillStep::Ubatch).collect();
                match tail {
                    0 => {}
                    t if t <= MAX_TOKENS => v.push(PrefillStep::Pass(t)),
                    t => v.push(PrefillStep::Ubatch(t)),
                }
                v
            }
        };
        PrefillPlan { steps }
    }

    /// Tokens the ubatches take: a prefix of the prompt.
    #[must_use]
    pub fn ubatch_tokens(&self) -> usize {
        self.steps
            .iter()
            .map(|s| match s {
                PrefillStep::Ubatch(t) => *t,
                PrefillStep::Pass(_) => 0,
            })
            .sum()
    }

    /// `gemm` when a ubatch runs, else `prefill` (the pass path's name).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        if self.ubatch_tokens() > 0 {
            "gemm"
        } else {
            "prefill"
        }
    }
}

impl std::fmt::Display for PrefillPlan {
    /// `ubatch:512x2,276 pass:1` — each path's unit sizes in order, a run of
    /// `k > 1` equal sizes as `<size>x<k>`, a path with none left out.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let list = |ub: bool| {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for s in &self.steps {
                let t = match (s, ub) {
                    (PrefillStep::Ubatch(t), true) | (PrefillStep::Pass(t), false) => *t,
                    _ => continue,
                };
                match runs.last_mut() {
                    Some((size, k)) if *size == t => *k += 1,
                    _ => runs.push((t, 1)),
                }
            }
            runs.iter()
                .map(|&(size, k)| {
                    if k == 1 {
                        size.to_string()
                    } else {
                        format!("{size}x{k}")
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let parts: Vec<String> = [("ubatch", list(true)), ("pass", list(false))]
            .into_iter()
            .filter(|(_, l)| !l.is_empty())
            .map(|(n, l)| format!("{n}:{l}"))
            .collect();
        write!(f, "{}", parts.join(" "))
    }
}

impl Body {
    /// Enqueue the ubatch of the ubatch image's `tokens`, the first at
    /// position `pos`.
    fn run_ubatch(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        tokens: Range<usize>,
        pos: u32,
    ) -> Result<(), GpuError> {
        let Body {
            hp,
            plans,
            kv,
            rope,
            k,
            ub,
            ..
        } = self;
        let c = UbCtx {
            gpu,
            w,
            plans,
            k,
            eps: hp.rms_eps,
            table: &rope.table,
        };
        ub.enqueue(&c, kv, tokens, pos)
    }
}

impl GpuModel<Body> {
    /// [`GpuModel::prefill_with`] by [`PrefillPath::Auto`].
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        self.prefill_with(tokens, PrefillPath::Auto)
    }

    /// The GEMM prefill's ubatch size: the most tokens one ubatch takes.
    pub fn ubatch(&self) -> Result<usize, GpuError> {
        Ok(self.body("qwen3moe::ubatch")?.ub.size().get())
    }

    /// Run the GEMM prefill in ubatches of up to `u` tokens from here on:
    /// `u` in `1..=UBATCH`, else refused with the old size kept. The
    /// ubatch arena is reallocated for `min(u, ctx)` rows (load-time
    /// allocation); nothing captured refers to it, and a token's values do
    /// not depend on the ubatch it lands in, so the prompt's K/V rows and
    /// logits are the same bits at every size.
    pub fn set_ubatch(&mut self, u: usize) -> Result<(), GpuError> {
        let (gpu, _, body) = self.body_parts("qwen3moe::set_ubatch")?;
        body.ub.resize(gpu.stream(), u)
    }

    /// The units [`GpuModel::prefill_with`] runs a prompt of `tokens` ids
    /// as by `path`, at this model's ubatch size.
    pub fn prefill_plan(&self, tokens: usize, path: PrefillPath) -> Result<PrefillPlan, GpuError> {
        let ub = self.body("qwen3moe::prefill_plan")?.ub.size();
        Ok(PrefillPlan::new(tokens, path, ub))
    }

    /// Feed `tokens` through the chain by the plan of `path` (module doc)
    /// and return the greedy next token after the last one. Positions
    /// continue from wherever the model stands. On the pass path this is
    /// what [`GpuModel::step`] returns for the same tokens from the same
    /// position, with the same cache rows behind it, bit for bit; the GEMM
    /// ubatches agree with it to their band. In graph mode a pass size not
    /// captured yet is captured here, inside the call; the ubatches run
    /// eager in either mode. The layer taps must be off, and every id must
    /// be below the vocabulary (the embedding would read another row).
    pub fn prefill_with(&mut self, tokens: &[u32], path: PrefillPath) -> Result<u32, GpuError> {
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT, "empty token slice"));
        }
        let pos0 = self.pos();
        self.check_pos(pos0 + launch_u32(WHAT, "tokens", tokens.len())? - 1, WHAT)?;
        let graph = self.mode() == StepMode::Graph;
        let plan = self.prefill_plan(tokens.len(), path)?;
        let n_ub = plan.ubatch_tokens();
        let passes: Vec<usize> = plan
            .steps
            .iter()
            .filter_map(|s| match s {
                PrefillStep::Pass(m) => Some(*m),
                PrefillStep::Ubatch(_) => None,
            })
            .collect();
        {
            let (gpu, w, body) = self.body_parts(WHAT)?;
            if body.taps.is_some() {
                return Err(GpuError::state(WHAT, "layer taps off (a pass writes none)"));
            }
            let n_vocab = body.hp.n_vocab;
            if let Some((i, &id)) = tokens
                .iter()
                .enumerate()
                .find(|&(_, &id)| id as usize >= n_vocab)
            {
                return Err(GpuError::shape(
                    WHAT,
                    format!("token {i} is id {id}, past the vocabulary of {n_vocab}"),
                ));
            }
            let Body { prefill, ub, .. } = body;
            if n_ub > 0 {
                ub.write(gpu.stream(), &tokens[..n_ub], pos0)?;
            }
            if !passes.is_empty() {
                let pos_p = pos0 + launch_u32(WHAT, "ubatch tokens", n_ub)?;
                prefill.write(gpu.stream(), &tokens[n_ub..], pos_p)?;
                if graph {
                    for &m in &passes {
                        if body.prefill.graphs[m - 1].is_none() {
                            body.capture_pass(gpu, w, m)?;
                        }
                    }
                }
            }
        }
        let n_steps = plan.steps.len();
        let (mut s, mut chunk) = (0usize, 0usize);
        let mut next = None;
        for (i, &step) in plan.steps.iter().enumerate() {
            let last = i + 1 == n_steps;
            next = match step {
                PrefillStep::Ubatch(t) => {
                    let tokens = s..s + t;
                    s += t;
                    self.run_rows(t, WHAT, |gpu, w, body, head, pos| {
                        body.run_ubatch(gpu, w, tokens, pos)?;
                        if last {
                            let row = body.ub.last_row(t)?;
                            head.input_mut()
                                .copy_from_device_async(&row, gpu.stream())?;
                            dispatch::enqueue_head(gpu, w, &body.k, &mut body.head_state, head)?;
                        }
                        Ok(last)
                    })?
                }
                PrefillStep::Pass(m) => {
                    let c = chunk;
                    chunk += 1;
                    self.run_rows(m, WHAT, |gpu, w, body, head, _| {
                        body.run_pass(gpu, w, c, m, graph)?;
                        if last {
                            let Body {
                                k,
                                head_state,
                                prefill,
                                ..
                            } = body;
                            dispatch::enqueue_pass_head(
                                gpu, w, k, head_state, &prefill.a, m, head,
                            )?;
                        }
                        Ok(last)
                    })?
                }
            };
        }
        next.ok_or(GpuError::state(WHAT, "a token read after the last unit"))
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

    /// The passes the pass path ([`PrefillPath::Pass`]) takes for a prompt
    /// of `tokens` ids.
    #[must_use]
    pub fn prefill_passes(tokens: usize) -> usize {
        tokens.div_ceil(MAX_TOKENS)
    }
}
