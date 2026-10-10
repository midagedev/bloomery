//! The next-token (NextN, MTP) layer on the target's card ([`Nextn`]): its
//! weights, its store and its walk's arena, opened once at load beside the
//! target from the NextN plan (`place::PlanInputs::plan_nextn`), and its
//! program ([`nextn_walk`], [`nextn_chain`]).
//!
//! The program is ik's GLM5NEXT MTP graph (`build_glm5next_mtp`,
//! `build_mtp_input`) over `m` rows (1..=[`WALK_ROWS`]): row `t` is the token
//! at position `pos0 + t` beside the target's hidden row at `pos0 + t − 1`,
//! the hidden row being the target's streams' mean normed by `output_norm`
//! (ik's `result_mtp_embd`). Every launch is an entry the target's chain
//! already has:
//! - the store part, every row: the token's embedding row read from the file
//!   on the host, position 0 included (ik's GLM MTP input masks no row);
//!   the hidden rows — a target arena's streams' mean (`ds41_hc_mean`) then
//!   the `output_norm` RMS, or rows from the host already normed; each row's
//!   `[enorm(e) | hnorm(h)]`, two RMS norms into the halves of its packed
//!   row; `nextn.eh_proj` over the `m` packed columns (`q8_0_gemv_mcol`);
//!   `attn_norm`, the joined `[q_a; latent; key; gate]` projection over the
//!   columns, the latent and index rows appended at the rows' positions and
//!   the pools they complete — a later row's attention reads nothing else of
//!   a row, so [`NextnMode::Store`] stops here;
//! - the full part, the last row alone: the latent mixer of a trunk layer on
//!   the one-row buffers (`crate::mla::latent_row`, which appends the row
//!   again, the same bits), the plain residual `h = x + attn`, `ffn_norm`,
//!   the router over 288 experts with the selection bias, the download of the
//!   row and its routing to the host tier's batch port, the shared expert
//!   under the host's union call over the layer's routed experts (every one
//!   on the host), the upload of their sum, `out = (routed + shared) + h`,
//!   `shared_head_norm` into the head's input, and the target's `output`
//!   projection and its argmax (a `HeadNorm::Mixed` head: the norm is the
//!   layer's own launch).
//!
//! The walk runs launch by launch: the host's union call sits in the middle
//! of it, so no walk is captured ([`NextnMode::Graph`] is refused by name).
//! The store is by position: one walk may start at or below the end of the
//! last walk of the sequence, never past it ([`Nextn::held`]); the model's
//! reset and a cut behind it move that end back, and a walk that fails
//! between its appends leaves it at the walk's first position.
//!
//! A walk that reads the target's rows reads them by position: each target
//! arena records the positions its rows hold ([`Wrote`]) — the step, the
//! verify and the prompt batch that last wrote it, less what a cut took
//! back — and a walk from `pos0` reads rows that hold positions `pos0 − 1 ..
//! pos0 + m − 2` of it, or is refused by name.
//!
//! A fault a launch raises poisons the model as soon as a call of the
//! program meets it ([`GpuModel::note_fault`]): the chain's readback, or a
//! walk's host leg; until then it stays on the card's fault word, which the
//! target's next readback names.

use bloomery_gpu::head::{Head, HeadNorm};
use bloomery_gpu::host::BatchLeg;
use bloomery_gpu::latent::{
    INDEX_HEAD, INDEX_ROW, IndexKeyArgs, LATENT, LatentAppendArgs, Rows, pools_for,
};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{COL_GROUP, DeviceTensor, Gpu, GpuError, GpuModel};
use bloomery_gpu_deepseek41::hc::HC_STREAMS;
use bloomery_gpu_deepseek41::span::{span, span_mut};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use model::arch::glm5next::names;
use model::arch::glm5next::place::{NEXTN_ARENA_BYTES, NextnInputs, NextnPlan, PlanInputs};
use models::Ffn;
use runtime::layer::{FfnKind, Layer, MixerKind};
use runtime::sched::{At, Overlap, Port, PortKind};

use super::WHAT as BODY;
use super::prefill::PromptState;
use super::{Body, Dims, Kernels, LANES, RowScratch, Scratch, f32t, f32v, gemv, weight};
use crate::mla::{self, LatentStore};
use crate::tensors::{FfnNames, LatentNames, LayerNames, MixerNames};

/// What the NextN walk's errors name.
const WHAT: &str = "glm5next NextN";

/// The most rows one walk takes: the m-column kernels' width.
pub const WALK_ROWS: usize = COL_GROUP;

/// A target arena whose final hidden rows a NextN walk reads: where a step,
/// a verify and a prompt batch leave their rows' streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlmArena {
    /// The step's one row: the step's row buffers.
    Step,
    /// A verify's two rows: row 0 copied out of the step's row buffers at the
    /// verify's end (the next step writes them), row 1 in its own.
    Pair,
    /// A prompt batch's rows, row `i` the batch's position `i`.
    Prefill,
}

/// Where a walk's hidden rows come from.
#[derive(Clone, Copy, Debug)]
pub enum NextnHidden<'a> {
    /// Rows written from the host, [`Body::nextn_hidden_width`] values a row,
    /// normed as the target's head norms them: the row at position 0 reads a
    /// zero row, the target holding nothing before it.
    Host(&'a [f32]),
    /// The target's rows in the arena `walk`, from its row `first` on, as the
    /// target's last call of that arena left them.
    Target { walk: GlmArena, first: usize },
}

/// A walk's rows: `tokens` at positions `pos0 ..`, each beside its hidden row.
#[derive(Clone, Copy, Debug)]
pub struct NextnFeed<'a> {
    pub tokens: &'a [u32],
    pub pos0: u32,
    pub hidden: NextnHidden<'a>,
}

/// Which projection the draft's head runs: the target's `output`, the whole
/// vocabulary (the load opens no row list).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextnHead {
    Full,
}

/// How a walk runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextnMode {
    /// Enqueued launch by launch, the last row through the head.
    Eager,
    /// Replayed from a capture: refused by name, the walk's host union call
    /// sits between its launches.
    Graph,
    /// Enqueued launch by launch through the store's appends and no further
    /// (module doc): no head, no readback.
    Store,
}

/// The layer's tensor names, and the target's `output_norm` its hidden rows
/// are normed by, made at load.
struct NextnNames {
    output_norm: String,
    eh: String,
    enorm: String,
    hnorm: String,
    head_norm: String,
    latent: LatentNames,
    ffn_norm: String,
    router: String,
    bias: String,
    sh_gate: String,
    sh_up: String,
    sh_down: String,
}

/// The walk's rows' buffers, up to [`WALK_ROWS`] rows, and the full row's
/// own: its residual, its block's input and output, the host sums.
struct Arena {
    /// The rows' embedding values on the host before their upload.
    e_host: Vec<f32>,
    pos_host: Vec<u32>,
    cnt_host: Vec<u32>,
    emb: DeviceBuffer<f32>,
    /// The hidden rows' streams' means, and the rows normed by `output_norm`.
    hid: DeviceBuffer<f32>,
    hn: DeviceBuffer<f32>,
    /// Each row's `[enorm(e) | hnorm(h)]`.
    pack: DeviceBuffer<f32>,
    /// `eh_proj`'s rows (the residual), their `attn_norm` and the joined
    /// projection's rows.
    x: DeviceBuffer<f32>,
    xn: DeviceBuffer<f32>,
    stack: DeviceBuffer<f32>,
    pos: DeviceBuffer<u32>,
    cnt: DeviceBuffer<u32>,
    /// The full row's `h = x + attn`, its `ffn_norm`, the routed and shared
    /// sum and the layer's output.
    res: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    moe: DeviceBuffer<f32>,
    out: DeviceBuffer<f32>,
    /// The host's routed sum the batch port uploads.
    hsum: DeviceBuffer<f32>,
}

impl Arena {
    fn new(stream: &CudaStream, n: usize, stack: usize) -> Result<Arena, GpuError> {
        let r = WALK_ROWS;
        let z = |len: usize| DeviceBuffer::<f32>::zeroed(stream, len);
        Ok(Arena {
            e_host: vec![0.0; r * n],
            pos_host: vec![0; r],
            cnt_host: vec![0; r],
            emb: z(r * n)?,
            hid: z(r * n)?,
            hn: z(r * n)?,
            pack: z(r * 2 * n)?,
            x: z(r * n)?,
            xn: z(r * n)?,
            stack: z(r * stack)?,
            pos: DeviceBuffer::zeroed(stream, r)?,
            cnt: DeviceBuffer::zeroed(stream, r)?,
            res: z(n)?,
            normed: z(n)?,
            moe: z(n)?,
            out: z(n)?,
            hsum: z(n)?,
        })
    }

    fn bytes(&self) -> usize {
        [
            &self.emb,
            &self.hid,
            &self.hn,
            &self.pack,
            &self.x,
            &self.xn,
            &self.stack,
            &self.res,
            &self.normed,
            &self.moe,
            &self.out,
            &self.hsum,
        ]
        .iter()
        .map(|b| b.num_bytes())
        .sum::<usize>()
            + self.pos.num_bytes()
            + self.cnt.num_bytes()
    }
}

/// One resident sequence's side of the layer ([`Nextn::new_seq`]): its
/// store's three planes, the positions they hold and the verify's row-0
/// copy — what a select of the target's slots exchanges
/// ([`Nextn::swap_seq`]). The layer's weights, head and walk buffers are the
/// load's.
pub struct NextnSeq {
    latent: DeviceTensor<u16>,
    index_rows: DeviceTensor<u16>,
    pooled: DeviceTensor<u16>,
    held: usize,
    pair0: DeviceBuffer<f32>,
}

impl NextnSeq {
    /// A cut of the sequence's target to `pos` ([`Nextn::cut`]'s rule on a
    /// parked sequence): the store holds no position past it.
    pub(super) fn cut(&mut self, pos: u32) {
        self.held = self.held.min(pos as usize);
    }

    /// The parked sequence's verify row-0 copy, which the end of a pass of
    /// several slots that runs its verify rows fills (`Body::copy_pair0`);
    /// the live sequence's is the layer's ([`Nextn::pair0_mut`]).
    pub(super) fn pair0_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.pair0
    }
}

/// The NextN layer resident on the target's card. See the module doc.
pub struct Nextn {
    /// The layer's `blk.` index in the file.
    index: usize,
    /// The layer's own weights: its tensors but the routed stacks, and the
    /// joined projection.
    w: Weights,
    names: NextnNames,
    latent: DeviceTensor<u16>,
    index_rows: DeviceTensor<u16>,
    pooled: DeviceTensor<u16>,
    /// The full row's buffers: a trunk step's one-row buffers.
    s: RowScratch,
    a: Arena,
    /// The verify's row 0 streams, copied at the verify's end.
    pair0: DeviceBuffer<f32>,
    /// The head over the target's `output`, its input the layer's own norm.
    head: Head,
    /// The shared expert's SwiGLU limit, and whether the file holds the
    /// router's selection bias.
    limit: f32,
    bias: bool,
    /// Positions of this sequence the store holds: the end of its last walk.
    held: usize,
    ctx: usize,
}

impl Nextn {
    /// Positions of the current sequence the store holds: a walk may start at
    /// or below it, never past it.
    #[must_use]
    pub fn held(&self) -> usize {
        self.held
    }

    /// Whether the head is a row list: never, this load opens the full head.
    #[must_use]
    pub fn head_rows(&self) -> bool {
        false
    }

    /// The layer's `blk.` index in the file.
    #[must_use]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Positions the store holds.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// The layer's own weights, which the load uploaded beside the
    /// target's. Gate use: a fault clause patches one in place.
    #[must_use]
    pub fn weights(&self) -> &Weights {
        &self.w
    }

    /// Device bytes of the layer's weights and store.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.w.resident_bytes() + self.store_bytes()
    }

    /// Device bytes of the walk's arena: its rows, the full row's buffers,
    /// the verify's row-0 copy and the head.
    #[must_use]
    pub fn arena_bytes(&self) -> usize {
        self.a.bytes() + self.s.bytes() + self.pair0.num_bytes() + self.head.resident_bytes()
    }

    fn store_bytes(&self) -> usize {
        self.latent.buf().num_bytes()
            + self.index_rows.buf().num_bytes()
            + self.pooled.buf().num_bytes()
    }

    /// A new sequence: the store holds no position of it.
    pub(super) fn forget(&mut self) {
        self.held = 0;
    }

    /// A cut of the target to `pos`: the store holds no position past it.
    pub(super) fn cut(&mut self, pos: u32) {
        self.held = self.held.min(pos as usize);
    }

    /// The verify's row-0 streams buffer, which the verify's end fills.
    pub(super) fn pair0_mut(&mut self) -> &mut DeviceBuffer<f32> {
        &mut self.pair0
    }

    /// The store's three planes — latent rows, index rows, pool keys — lent
    /// for a sequence state's copies (`body::seq`).
    pub(super) fn store_mut(&mut self) -> [&mut DeviceTensor<u16>; 3] {
        [&mut self.latent, &mut self.index_rows, &mut self.pooled]
    }

    /// A resident sequence's side in the state the load leaves: its store
    /// zeroed at the layer's context, no position held, its row-0 copy
    /// zeroed. Load-time allocation.
    pub(super) fn new_seq(&self, stream: &CudaStream) -> Result<NextnSeq, GpuError> {
        Ok(NextnSeq {
            latent: DeviceTensor::zeroed(stream, self.ctx, LATENT)?,
            index_rows: DeviceTensor::zeroed(stream, self.ctx, INDEX_ROW)?,
            pooled: DeviceTensor::zeroed(stream, pools_for(self.ctx), INDEX_HEAD)?,
            held: 0,
            pair0: DeviceBuffer::zeroed(stream, self.pair0.len())?,
        })
    }

    /// Exchange the live sequence's side with `s`: pointer moves only, so a
    /// slot's captured verify keeps copying into its own row-0 buffer.
    pub(super) fn swap_seq(&mut self, s: &mut NextnSeq) {
        std::mem::swap(&mut self.latent, &mut s.latent);
        std::mem::swap(&mut self.index_rows, &mut s.index_rows);
        std::mem::swap(&mut self.pooled, &mut s.pooled);
        std::mem::swap(&mut self.held, &mut s.held);
        std::mem::swap(&mut self.pair0, &mut s.pair0);
    }

    /// Device bytes one sequence's side holds: the store and the row-0 copy.
    pub(super) fn seq_bytes(&self) -> usize {
        self.store_bytes() + self.pair0.num_bytes()
    }

    /// The layer the NextN plan `plan` places, as `inputs` and `nextn`
    /// describe it, resident on `gpu` beside the target's weights `tw`, from
    /// `file`: its tensors uploaded from the plan's rows and checked against
    /// its bytes, the joined projection derived, its store at `ctx`
    /// positions — the plan counting one for each of the load's `slots`
    /// resident sequences — and its arena, held to [`NEXTN_ARENA_BYTES`].
    /// Refused by name: a layer other than a routed latent one, a tensor
    /// absent or of another type, an upload or a store whose bytes are not
    /// the plan's, an arena past its bound. Load-time only.
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's card, file, plans, inputs, the target's weights and widths, its context and resident sequences, and page release (rust-quality R8)"
    )]
    pub(super) fn open(
        gpu: &Gpu,
        file: &Split,
        plan: &NextnPlan<'_>,
        inputs: &PlanInputs,
        nextn: &NextnInputs,
        tw: &Weights,
        d: &Dims,
        (ctx, slots): (usize, usize),
        card_dontneed: bool,
    ) -> Result<Nextn, GpuError> {
        let index = nextn.index;
        let spec =
            inputs.spec.mtp.first().ok_or_else(|| {
                shape("the file's description has no next-token layer".to_string())
            })?;
        let kind = Layer::of(spec);
        if kind.mixer != MixerKind::Latent || kind.ffn != FfnKind::Moe {
            return Err(shape(format!(
                "layer {index} runs a {:?} mixer and a {:?} block; the NextN program runs a \
                 latent mixer and a routed block",
                kind.mixer, kind.ffn
            )));
        }
        let Ffn::Moe(moe) = &spec.ffn else {
            return Err(shape(format!(
                "layer {index}: a routed block without its routing"
            )));
        };
        let shared = moe.shared.ok_or_else(|| {
            shape(format!(
                "layer {index}: a routed block without its shared expert"
            ))
        })?;
        let ln = LayerNames::of(index, kind)?;
        let (
            MixerNames::Latent(latent_names),
            FfnNames::Moe {
                norm,
                router,
                bias,
                sh_gate,
                sh_up,
                sh_down,
            },
        ) = (ln.mixer, ln.ffn)
        else {
            return Err(shape(format!("layer {index}: names of another kind")));
        };
        let names = NextnNames {
            output_norm: names::output_norm(),
            eh: names::nextn_eh_proj(index),
            enorm: names::nextn_enorm(index),
            hnorm: names::nextn_hnorm(index),
            head_norm: names::nextn_shared_head_norm(index),
            latent: latent_names,
            ffn_norm: norm,
            router,
            bias,
            sh_gate,
            sh_up,
            sh_down,
        };
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let mut w = Weights::load_placed(stream, file, &plan.nextn, 0, card_dontneed)?;
        let parts = [
            names::attn_q_a(index),
            names::attn_kv_a_mqa(index),
            names::indexer_attn_k(index),
            names::indexer_compressor_gate(index),
        ];
        let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
        w.join_rows(stream, &parts, names::attn_a_stack(index))?;
        let held_bytes = w.resident_bytes() as u64;
        if held_bytes != plan.nextn_resident_bytes() {
            return Err(shape(format!(
                "the layer's weights hold {held_bytes} device bytes; the NextN plan has {}",
                plan.nextn_resident_bytes()
            )));
        }
        for name in [&names.eh, &names.enorm, &names.hnorm, &names.head_norm] {
            weight(&w, name)?;
        }
        f32v(tw, &names.output_norm)?;
        let bias_held = w.get(&names.bias).is_some();
        if bias_held != moe.router.bias {
            return Err(shape(format!(
                "layer {index}: the selection bias resident {bias_held}, the description {}",
                moe.router.bias
            )));
        }
        let head = Head::with_norm(gpu, tw, d.rms_eps, 1, HeadNorm::Mixed)?;
        if head.hidden() != d.embd {
            return Err(shape(format!(
                "the head projects rows of {} values; the layer's are {}",
                head.hidden(),
                d.embd
            )));
        }
        let n = d.embd;
        let body = Nextn {
            index,
            w,
            names,
            latent: DeviceTensor::zeroed(stream, ctx, LATENT)?,
            index_rows: DeviceTensor::zeroed(stream, ctx, INDEX_ROW)?,
            pooled: DeviceTensor::zeroed(stream, pools_for(ctx), INDEX_HEAD)?,
            s: RowScratch::new(stream, d, shared.ff as usize, ctx)?,
            a: Arena::new(stream, n, d.stack())?,
            pair0: DeviceBuffer::zeroed(stream, HC_STREAMS * n)?,
            head,
            limit: shared.act.swiglu_limit(),
            bias: moe.router.bias,
            held: 0,
            ctx,
        };
        stream.synchronize()?;
        let store = body.store_bytes() as u64;
        if store * slots as u64 != plan.nextn.cards[0].kv_bytes {
            return Err(shape(format!(
                "the layer's store holds {store} device bytes a sequence, for {slots} resident \
                 sequences; the NextN plan has {}",
                plan.nextn.cards[0].kv_bytes
            )));
        }
        let arena = body.arena_bytes() as u64;
        if arena > NEXTN_ARENA_BYTES {
            return Err(shape(format!(
                "the walk's arena holds {arena} device bytes, past the plan's bound of \
                 {NEXTN_ARENA_BYTES}"
            )));
        }
        Ok(body)
    }
}

/// A shape the walk refuses, by name.
fn shape(detail: String) -> GpuError {
    GpuError::Shape { what: WHAT, detail }
}

/// The positions a target arena's rows hold, row `i` position `first + i`:
/// what the call that last wrote the arena left there, less the positions a
/// cut took back since. No rows: the arena holds no position.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Held {
    first: u32,
    rows: u32,
}

impl Held {
    /// No position.
    pub(crate) const NONE: Held = Held { first: 0, rows: 0 };

    /// Rows `0 .. rows` at positions `first ..`.
    pub(crate) fn at(first: u32, rows: u32) -> Held {
        Held { first, rows }
    }

    /// The positions from `pos` on taken back.
    fn cut(&mut self, pos: u32) {
        self.rows = self.rows.min(pos.saturating_sub(self.first));
    }

    /// Whether rows `row .. row + m` hold positions `at .. at + m`.
    fn holds(&self, row: usize, m: usize, at: u32) -> bool {
        row + m <= self.rows as usize && self.first as usize + row == at as usize
    }

    /// The positions held, for a refusal's words.
    fn shown(&self) -> String {
        match self.rows {
            0 => "no position".to_string(),
            r => format!("positions {}..{}", self.first, self.first + r),
        }
    }
}

/// Each target arena's [`Held`]: the body's record of where the step, the
/// verify and the prompt batch left their rows.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Wrote {
    pub(crate) step: Held,
    pub(crate) pair: Held,
    pub(crate) prefill: Held,
}

impl Wrote {
    /// Every arena's positions from `pos` on taken back.
    pub(crate) fn cut(&mut self, pos: u32) {
        for h in [&mut self.step, &mut self.pair, &mut self.prefill] {
            h.cut(pos);
        }
    }

    fn of(&self, walk: GlmArena) -> Held {
        match walk {
            GlmArena::Step => self.step,
            GlmArena::Pair => self.pair,
            GlmArena::Prefill => self.prefill,
        }
    }
}

/// What a walk's hidden rows are gathered for: the feed's source, its first
/// position, its rows and their width.
#[derive(Clone, Copy)]
struct Want<'h> {
    hidden: NextnHidden<'h>,
    pos0: u32,
    m: usize,
    n: usize,
}

/// The target's streams a [`NextnHidden::Target`] feed reads: one buffer of
/// rows of `4 · n_embd` values, or the verify's two rows apart.
enum Src<'a> {
    Rows {
        buf: &'a DeviceBuffer<f32>,
        rows: usize,
    },
    Pair {
        row0: &'a DeviceBuffer<f32>,
        row1: &'a DeviceBuffer<f32>,
    },
}

impl Src<'_> {
    /// Row `r`'s streams, `wide` values.
    fn row(
        &self,
        r: usize,
        wide: usize,
    ) -> Result<bloomery_gpu_deepseek41::span::Span<'_, f32>, GpuError> {
        match *self {
            Src::Rows { buf, rows } if r < rows => span(WHAT, buf, r * wide, wide),
            Src::Pair { row0, .. } if r == 0 => span(WHAT, row0, 0, wide),
            Src::Pair { row1, .. } if r == 1 => span(WHAT, row1, 0, wide),
            _ => Err(shape(format!("row {r} of the arena's rows"))),
        }
    }
}

/// The target's rows a walk of `want.m` rows of `want.n` values from
/// `want.pos0` reads for `want.hidden`: `None` for rows from the host, else
/// the arena's rows from `first` on, which must hold positions `pos0 − 1 ..
/// pos0 + m − 2` ([`Wrote`]). Refused by name: a host slice of other than
/// `m · n` values, prompt-batch rows before any batch buffers, rows past the
/// arena's buffers, a walk at position 0 (no target row before it), rows
/// that hold other positions or none.
fn hidden_src<'a>(
    s: &'a Scratch,
    prompt: &'a PromptState,
    pair0: &'a DeviceBuffer<f32>,
    wrote: &Wrote,
    layers: usize,
    want: Want<'_>,
) -> Result<Option<(Src<'a>, usize)>, GpuError> {
    let Want { hidden, pos0, m, n } = want;
    let fin = crate::program::final_streams(layers);
    let (walk, first) = match hidden {
        NextnHidden::Host(v) if v.len() != m * n => {
            return Err(shape(format!(
                "{} hidden values for {m} rows of {n}",
                v.len()
            )));
        }
        NextnHidden::Host(_) => return Ok(None),
        NextnHidden::Target { walk, first } => (walk, first),
    };
    let src = match walk {
        GlmArena::Step => Src::Rows {
            buf: &s.rows[0].streams[fin],
            rows: 1,
        },
        GlmArena::Pair => Src::Pair {
            row0: pair0,
            row1: &s.rows[1].streams[fin],
        },
        GlmArena::Prefill => {
            let (buf, rows) = prompt
                .final_streams(fin)
                .ok_or_else(|| shape("prompt-batch rows before any batch buffers".to_string()))?;
            Src::Rows { buf, rows }
        }
    };
    let rows = match src {
        Src::Rows { rows, .. } => rows,
        Src::Pair { .. } => 2,
    };
    if first + m > rows {
        return Err(shape(format!(
            "rows {first}..{} of the {walk:?} arena's {rows}",
            first + m
        )));
    }
    let at = pos0.checked_sub(1).ok_or_else(|| {
        shape(format!(
            "the {walk:?} arena's rows for a walk at position 0: the target holds no row before \
             it (a host zero row is position 0's)"
        ))
    })?;
    let held = wrote.of(walk);
    if !held.holds(first, m, at) {
        return Err(shape(format!(
            "rows {first}..{} of the {walk:?} arena as positions {at}..{} for a walk from {pos0}: \
             the arena holds {}",
            first + m,
            at as usize + m,
            held.shown()
        )));
    }
    Ok(Some((src, first)))
}

/// The walk's `m` hidden rows into `a.hn`, normed as the target's head
/// norms them: copied from the host, or each target row's streams' mean
/// then the RMS by the target's `output_norm` (`norm`).
#[allow(
    clippy::too_many_arguments,
    reason = "the card, the target's weights and its norm's name, the kernels and widths, the feed's source resolved and its rows, and the arena (rust-quality R8)"
)]
fn enqueue_hidden(
    gpu: &Gpu,
    tw: &Weights,
    norm: &str,
    k: &Kernels,
    d: &Dims,
    hidden: NextnHidden<'_>,
    src: Option<(Src<'_>, usize)>,
    m: usize,
    a: &mut Arena,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let n = d.embd;
    match (hidden, src) {
        (NextnHidden::Host(v), _) => {
            span_mut(WHAT, &mut a.hn, 0, m * n)?.copy_from_host(stream, v)?;
        }
        (NextnHidden::Target { .. }, Some((src, first))) => {
            for t in 0..m {
                let rowv = src.row(first + t, HC_STREAMS * n)?;
                k.hc.enqueue_mean(stream, &rowv, n, t * n, &mut a.hid)?;
            }
            gpu.elem().enqueue_rms_norm(
                stream,
                &a.hid,
                f32v(tw, norm)?,
                d.rms_eps,
                n,
                m,
                &mut a.hn,
            )?;
        }
        (NextnHidden::Target { .. }, None) => {
            return Err(shape("the target's rows to read".to_string()));
        }
    }
    Ok(())
}

/// What a sequence state carries of a NextN load beside the store's rows
/// (`body::seq`): the positions the store holds ([`Nextn::held`]), and the
/// step's and the verify's arena rows with the positions they hold
/// ([`Wrote`]) — the target's hidden rows a draft's waiting rows read when it
/// rejoins the sequence. The prompt batch's rows are not carried: after a
/// resume that arena holds no position, and a walk that names it is refused.
#[derive(Clone, Debug)]
pub struct DraftRows {
    held: usize,
    step: (Held, Vec<f32>),
    pair: (Held, Vec<f32>),
}

impl DraftRows {
    /// The host bytes it holds.
    #[must_use]
    pub fn bytes(&self) -> usize {
        (self.step.1.len() + self.pair.1.len()) * size_of::<f32>()
    }
}

/// The rows `held` names of an arena whose row `r` is `rows[r]`, each `wide`
/// values, read back; refused by name past the arena's rows.
fn read_rows(
    stream: &CudaStream,
    held: Held,
    rows: &[&DeviceBuffer<f32>],
    wide: usize,
) -> Result<(Held, Vec<f32>), GpuError> {
    let r = held.rows as usize;
    let bufs = rows.get(..r).ok_or_else(|| {
        shape(format!(
            "{} as rows of an arena of {}",
            held.shown(),
            rows.len()
        ))
    })?;
    let mut v = Vec::with_capacity(r * wide);
    for b in bufs {
        v.extend(span(WHAT, b, 0, wide)?.to_host_vec(stream)?);
    }
    Ok((held, v))
}

/// `v`'s rows back into the arena whose row `r` is `rows[r]`, each `wide`
/// values, as many as `v` holds ([`Body::draft_fits`] held them to the
/// arena).
fn write_rows(
    stream: &CudaStream,
    v: &[f32],
    rows: &mut [&mut DeviceBuffer<f32>],
    wide: usize,
) -> Result<(), GpuError> {
    for (b, row) in rows.iter_mut().zip(v.chunks_exact(wide)) {
        span_mut(WHAT, b, 0, wide)?.copy_from_host(stream, row)?;
    }
    Ok(())
}

impl Body {
    /// The step arena's rows written by a call that is no step (a gate's
    /// forced row): the arena holds no position until the next step.
    pub(crate) fn overwrote_step_arena(&mut self) {
        self.wrote.step = Held::NONE;
    }

    /// The NextN side of a sequence state at `pos` ([`DraftRows`]); `None`
    /// on a load without the layer. Refused by name when the store holds
    /// positions past `pos`. Blocking.
    pub(super) fn draft_rows(
        &self,
        stream: &CudaStream,
        pos: u32,
    ) -> Result<Option<DraftRows>, GpuError> {
        let Some(nx) = self.nextn.as_deref() else {
            return Ok(None);
        };
        if nx.held > pos as usize {
            return Err(shape(format!(
                "a state at {pos} of a NextN store that holds {} positions",
                nx.held
            )));
        }
        let fin = crate::program::final_streams(self.cfg.len());
        let wide = HC_STREAMS * self.dims.embd;
        let step = read_rows(
            stream,
            self.wrote.step,
            &[&self.s.rows[0].streams[fin]],
            wide,
        )?;
        let pair_rows = [&nx.pair0, &self.s.rows[1].streams[fin]];
        let pair = read_rows(stream, self.wrote.pair, &pair_rows, wide)?;
        Ok(Some(DraftRows {
            held: nx.held,
            step,
            pair,
        }))
    }

    /// Refused by name unless `d` fits this load standing at `pos`, nothing
    /// copied: the layer loaded, the store's positions and each arena's at
    /// or below `pos`, each arena's rows within its buffers (the step's one,
    /// the verify's two) and of the model's width.
    pub(super) fn draft_fits(&self, d: &DraftRows, pos: u32) -> Result<(), GpuError> {
        if self.nextn.is_none() {
            return Err(shape(
                "the NextN rows of a state put back on a load without the layer".into(),
            ));
        }
        if d.held > pos as usize {
            return Err(shape(format!(
                "a NextN store of {} positions put back at {pos}",
                d.held
            )));
        }
        let wide = HC_STREAMS * self.dims.embd;
        for (what, (held, v), rows) in [("step", &d.step, 1), ("verify", &d.pair, LANES)] {
            let r = held.rows as usize;
            if r > rows || v.len() != r * wide || (r > 0 && held.first + held.rows > pos) {
                return Err(shape(format!(
                    "the {what} arena's {} ({} values) put back into {rows} rows of {wide} at \
                     position {pos}",
                    held.shown(),
                    v.len()
                )));
            }
        }
        Ok(())
    }

    /// `d` put back on a model that stands at `pos` ([`Body::draft_rows`]'s
    /// rows): the store holding `d`'s positions, the step's and the verify's
    /// arenas their rows, the prompt batch's arena none. Refused by name, as
    /// [`Body::draft_fits`] refuses, before any copy.
    pub(super) fn put_draft_rows(
        &mut self,
        stream: &CudaStream,
        d: &DraftRows,
        pos: u32,
    ) -> Result<(), GpuError> {
        self.draft_fits(d, pos)?;
        let fin = crate::program::final_streams(self.cfg.len());
        let wide = HC_STREAMS * self.dims.embd;
        let Body {
            s, nextn, wrote, ..
        } = self;
        let nx = nextn.as_deref_mut().ok_or_else(|| {
            shape("the NextN rows of a state put back on a load without the layer".into())
        })?;
        write_rows(stream, &d.step.1, &mut [&mut s.rows[0].streams[fin]], wide)?;
        let mut pair_rows = [&mut nx.pair0, &mut s.rows[1].streams[fin]];
        write_rows(stream, &d.pair.1, &mut pair_rows, wide)?;
        nx.held = d.held;
        *wrote = Wrote {
            step: d.step.0,
            pair: d.pair.0,
            prefill: Held::NONE,
        };
        Ok(())
    }

    /// The NextN layer, when the load carries it.
    #[must_use]
    pub fn nextn(&self) -> Option<&Nextn> {
        self.nextn.as_deref()
    }

    /// Values of one hidden row a NextN walk reads: the model's width.
    #[must_use]
    pub fn nextn_hidden_width(&self) -> usize {
        self.dims.embd
    }

    /// One NextN walk of `feed` in `mode` (module doc), with no readback;
    /// `tw` the target's weights (the head's projection, `output_norm`).
    /// Refused by name before anything moves: a load without the layer, a
    /// captured walk, rows outside 1..=[`WALK_ROWS`], a token past the
    /// vocabulary, rows past the store, a walk from past [`Nextn::held`], a
    /// hidden slice of other than the rows' values, target rows the arena
    /// does not hold at the walk's positions ([`Wrote`]).
    fn nextn_run(
        &mut self,
        gpu: &Gpu,
        tw: &Weights,
        feed: NextnFeed<'_>,
        mode: NextnMode,
    ) -> Result<(), GpuError> {
        let layers = self.cfg.len();
        let Body {
            hybrid,
            dims,
            k,
            s,
            prompt,
            embd,
            nextn,
            wrote,
            ..
        } = self;
        let nx = nextn
            .as_deref_mut()
            .ok_or_else(|| shape("a walk on a load without the NextN layer".to_string()))?;
        if mode == NextnMode::Graph {
            return Err(shape(
                "a captured NextN walk: the host's union call over the layer's routed experts \
                 sits between its launches"
                    .to_string(),
            ));
        }
        let d = *dims;
        let n = d.embd;
        let m = feed.tokens.len();
        let pos0 = feed.pos0 as usize;
        if !(1..=WALK_ROWS).contains(&m) {
            return Err(shape(format!(
                "a walk of {m} rows; a walk takes 1..={WALK_ROWS}"
            )));
        }
        if pos0 + m > nx.ctx {
            return Err(shape(format!(
                "{m} rows from position {pos0} in a store of {}",
                nx.ctx
            )));
        }
        if pos0 > nx.held {
            return Err(shape(format!(
                "a walk from position {pos0}: the store holds this sequence's positions below {}, \
                 and the walk would read the rows between",
                nx.held
            )));
        }
        if let Some(&t) = feed.tokens.iter().find(|&&t| t as usize >= embd.n_vocab) {
            return Err(shape(format!(
                "token {t} is past the {} embedding rows",
                embd.n_vocab
            )));
        }
        let want = Want {
            hidden: feed.hidden,
            pos0: feed.pos0,
            m,
            n,
        };
        let src = hidden_src(s, prompt, &nx.pair0, wrote, layers, want)?;
        let stream = gpu.stream();
        let fault = gpu.layer_sink(nx.index)?;
        let Nextn {
            w,
            names: nm,
            latent,
            index_rows,
            pooled,
            s: row,
            a,
            head,
            limit,
            bias,
            held,
            index,
            ..
        } = nx;
        // The rows' embeddings and positions.
        for t in 0..m {
            let p = u32::try_from(pos0 + t).expect("a walk's positions are under the store's rows");
            let e = &mut a.e_host[t * n..(t + 1) * n];
            embd.row_into(feed.tokens[t], e)?;
            a.pos_host[t] = p;
            a.cnt_host[t] = p + 1;
        }
        span_mut(WHAT, &mut a.emb, 0, m * n)?.copy_from_host(stream, &a.e_host[..m * n])?;
        span_mut(WHAT, &mut a.pos, 0, m)?.copy_from_host(stream, &a.pos_host[..m])?;
        span_mut(WHAT, &mut a.cnt, 0, m)?.copy_from_host(stream, &a.cnt_host[..m])?;
        enqueue_hidden(gpu, tw, &nm.output_norm, k, &d, feed.hidden, src, m, a)?;
        // Each row's [enorm(e) | hnorm(h)], then eh_proj over the columns.
        let (enorm, hnorm) = (f32v(w, &nm.enorm)?, f32v(w, &nm.hnorm)?);
        for t in 0..m {
            let e = span(WHAT, &a.emb, t * n, n)?;
            let h = span(WHAT, &a.hn, t * n, n)?;
            gpu.elem().enqueue_rms_norm(
                stream,
                &e,
                enorm,
                d.rms_eps,
                n,
                1,
                &mut *span_mut(WHAT, &mut a.pack, t * 2 * n, n)?,
            )?;
            gpu.elem().enqueue_rms_norm(
                stream,
                &h,
                hnorm,
                d.rms_eps,
                n,
                1,
                &mut *span_mut(WHAT, &mut a.pack, t * 2 * n + n, n)?,
            )?;
        }
        w.q8_gemv_mcol(
            gpu,
            BODY,
            &nm.eh,
            &*span(WHAT, &a.pack, 0, m * 2 * n)?,
            m,
            &mut a.x,
        )?;
        // The store: the joined projection's rows, the latent and index rows
        // at the rows' positions, the pools they complete.
        let ln = &nm.latent;
        let stride = d.stack();
        gpu.elem().enqueue_rms_norm(
            stream,
            &a.x,
            f32v(w, &ln.norm)?,
            d.rms_eps,
            n,
            m,
            &mut a.xn,
        )?;
        w.q8_gemv_mcol(
            gpu,
            BODY,
            &ln.stack,
            &*span(WHAT, &a.xn, 0, m * n)?,
            m,
            &mut a.stack,
        )?;
        // From the first append on, rows from pos0 may be overwritten: a
        // failure before the last leaves the store holding positions below
        // pos0 alone.
        *held = (*held).min(pos0);
        {
            let rows = span(WHAT, &a.stack, 0, m * stride)?;
            let pos = span(WHAT, &a.pos, 0, m)?;
            k.latent.enqueue_latent_append(
                stream,
                LatentAppendArgs {
                    rows: Rows {
                        x: &rows,
                        stride,
                        m,
                    },
                    off: d.q_lora,
                    gain: f32v(w, &ln.kv_a_norm)?,
                    pos: &pos,
                    eps: d.rms_eps,
                    fault,
                    cache: latent,
                },
            )?;
            k.latent.enqueue_index_key_append(
                stream,
                IndexKeyArgs {
                    rows: Rows {
                        x: &rows,
                        stride,
                        m,
                    },
                    k_off: d.q_lora + LATENT,
                    g_off: d.q_lora + LATENT + d.index_d,
                    w: f32v(w, &ln.index_norm)?,
                    b: f32v(w, &ln.index_norm_bias)?,
                    pos: &pos,
                    eps: d.norm_eps,
                    fault,
                    cache: index_rows,
                },
            )?;
            let cnt = span(WHAT, &a.cnt, 0, m)?;
            mla::pool(stream, w, k, &ln.sel, index_rows, &cnt, m, fault, pooled)?;
        }
        *held = pos0 + m;
        if mode == NextnMode::Store {
            return Ok(());
        }
        // The full part, the last row alone.
        let r = m - 1;
        let p = u32::try_from(pos0 + r).expect("a walk's positions are under the store's rows");
        row.x
            .copy_from_device_async(&*span(WHAT, &a.x, r * n, n)?, stream)?;
        row.pos.copy_from_host(stream, &[p])?;
        row.vis.copy_from_host(stream, &[0, p + 1])?;
        row.cnt.copy_from_host(stream, &[p + 1])?;
        mla::latent_row(
            gpu,
            w,
            k,
            &d,
            ln,
            LatentStore {
                latent,
                index: index_rows,
                pooled,
            },
            row,
            fault,
        )?;
        gpu.elem()
            .enqueue_add(stream, &row.out, &row.x, n, &mut a.res)?;
        gpu.elem().enqueue_rms_norm(
            stream,
            &a.res,
            f32v(w, &nm.ffn_norm)?,
            d.rms_eps,
            n,
            1,
            &mut a.normed,
        )?;
        let sel_bias = if *bias {
            f32v(w, &nm.bias)?
        } else {
            &row.no_bias
        };
        k.router.enqueue_router(
            stream,
            f32t(w, &nm.router)?,
            &a.normed,
            sel_bias,
            d.scale,
            &mut row.rout,
            fault,
        )?;
        let at = At {
            unit: 0,
            layer: *index,
        };
        {
            let mut leg = BatchLeg::new(stream, hybrid, &mut a.hsum, 1);
            leg.open(Overlap {
                units: 1,
                cols: 1,
                port: PortKind::Batch,
            })?;
            let key = leg.key(at);
            leg.hybrid().enqueue_download(
                stream,
                [&a.normed, &row.rout.weights],
                &row.rout.ids,
                key,
            )?;
            k.experts.enqueue_shexp_gate_up(
                stream,
                weight(w, &nm.sh_gate)?,
                weight(w, &nm.sh_up)?,
                &a.normed,
                *limit,
                &mut row.h,
            )?;
            gemv(gpu, w, &nm.sh_down, &row.h, &mut row.sh_y)?;
            leg.serve(at)?;
            gpu.elem()
                .enqueue_add(stream, leg.hsum(), &row.sh_y, n, &mut a.moe)?;
        }
        gpu.elem()
            .enqueue_add(stream, &a.moe, &a.res, n, &mut a.out)?;
        gpu.elem().enqueue_rms_norm(
            stream,
            &a.out,
            f32v(w, &nm.head_norm)?,
            d.rms_eps,
            n,
            1,
            head.input_mut(),
        )?;
        head.enqueue(gpu, tw)
    }
}

/// One NextN walk of `feed` into `head` in `mode`, with no readback
/// ([`Body::nextn_run`]'s refusals, and a poisoned model's). A fault the
/// walk's own calls meet is its error and poisons the model; one they do not
/// read stays on the card's fault word, which the next readback names.
pub fn nextn_walk(
    m: &mut GpuModel<Body>,
    feed: NextnFeed<'_>,
    head: NextnHead,
    mode: NextnMode,
) -> Result<(), GpuError> {
    let NextnHead::Full = head;
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned { what: WHAT, fault });
    }
    let (gpu, tw, body) = m.body_parts(WHAT)?;
    let r = body.nextn_run(gpu, tw, feed, mode);
    m.note_fault(WHAT, r)
}

/// One window's chain, one readback: `refresh` walked eager through the head,
/// its last row's prediction the proposal's one id, written to `out[0]`; the
/// count, 1. `own` walks past it are refused by name (a proposal holds one
/// id), as are a store walk's mode, a captured one and an `out` with no
/// place. A fault any launch raised is the chain's error and poisons the
/// model as the target's would.
pub fn nextn_chain(
    m: &mut GpuModel<Body>,
    refresh: NextnFeed<'_>,
    own: usize,
    head: NextnHead,
    mode: NextnMode,
    out: &mut [u32],
) -> Result<usize, GpuError> {
    if own != 0 {
        return Err(shape(format!(
            "a chain of {own} own walks; a GLM proposal holds one id"
        )));
    }
    if mode == NextnMode::Store {
        return Err(shape(
            "a chain in the store walk's mode: it reads the head".to_string(),
        ));
    }
    let place = out
        .first_mut()
        .ok_or_else(|| shape("a proposal of 1 id into 0 places".to_string()))?;
    nextn_walk(m, refresh, head, mode)?;
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let nx = body
        .nextn
        .as_deref()
        .ok_or_else(|| shape("a chain on a load without the NextN layer".to_string()))?;
    let read = nx.head.tokens(gpu);
    let tokens = m.note_fault(WHAT, read)?;
    *place = *tokens
        .first()
        .ok_or_else(|| shape("a head readback of no token".to_string()))?;
    Ok(1)
}

/// The `rows` hidden rows a walk from `pos0` fed `hidden` reads, as the
/// walk norms them (the model's width a row), read back: the walk's own
/// gather and nothing after it, the store untouched. Refused as a walk
/// refuses its hidden rows and its rows' count. Blocking; gate use.
pub fn nextn_hidden(
    m: &mut GpuModel<Body>,
    pos0: u32,
    hidden: NextnHidden<'_>,
    rows: usize,
) -> Result<Vec<f32>, GpuError> {
    if !(1..=WALK_ROWS).contains(&rows) {
        return Err(shape(format!(
            "{rows} hidden rows; a walk takes 1..={WALK_ROWS}"
        )));
    }
    let (gpu, tw, body) = m.body_parts(WHAT)?;
    let layers = body.cfg.len();
    let d = body.dims;
    let Body {
        k,
        s,
        prompt,
        nextn,
        wrote,
        ..
    } = body;
    let nx = nextn
        .as_deref_mut()
        .ok_or_else(|| shape("hidden rows on a load without the NextN layer".to_string()))?;
    let want = Want {
        hidden,
        pos0,
        m: rows,
        n: d.embd,
    };
    let src = hidden_src(s, prompt, &nx.pair0, wrote, layers, want)?;
    enqueue_hidden(
        gpu,
        tw,
        &nx.names.output_norm,
        k,
        &d,
        hidden,
        src,
        rows,
        &mut nx.a,
    )?;
    let mut v = nx.a.hn.to_host_vec(gpu.stream())?;
    v.truncate(rows * d.embd);
    Ok(v)
}

/// The target's streams the `rows` hidden rows of a walk from `pos0` fed
/// `hidden` are made from, read back: each row's `4 · n_embd` values as its
/// arena holds them (stream-major), the rows the walk's gather reads.
/// Refused as a walk refuses its target rows, and by name for rows from the
/// host, which have no target streams. Blocking; gate use.
pub fn nextn_target_streams(
    m: &mut GpuModel<Body>,
    pos0: u32,
    hidden: NextnHidden<'_>,
    rows: usize,
) -> Result<Vec<f32>, GpuError> {
    if !(1..=WALK_ROWS).contains(&rows) {
        return Err(shape(format!(
            "{rows} hidden rows; a walk takes 1..={WALK_ROWS}"
        )));
    }
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let layers = body.cfg.len();
    let n = body.dims.embd;
    let Body {
        s,
        prompt,
        nextn,
        wrote,
        ..
    } = body;
    let nx = nextn
        .as_deref()
        .ok_or_else(|| shape("target rows on a load without the NextN layer".to_string()))?;
    let want = Want {
        hidden,
        pos0,
        m: rows,
        n,
    };
    let (src, first) = hidden_src(s, prompt, &nx.pair0, wrote, layers, want)?
        .ok_or_else(|| shape("rows from the host have no target streams".to_string()))?;
    let wide = HC_STREAMS * n;
    let mut out = Vec::with_capacity(rows * wide);
    for t in 0..rows {
        out.extend(src.row(first + t, wide)?.to_host_vec(gpu.stream())?);
    }
    Ok(out)
}

/// The last full walk's head logits (`n_vocab` f32). Blocking; gate use.
pub fn nextn_logits(m: &mut GpuModel<Body>) -> Result<Vec<f32>, GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let nx = body
        .nextn
        .as_deref()
        .ok_or_else(|| shape("logits on a load without the NextN layer".to_string()))?;
    nx.head.logits_to_host(gpu)
}

/// The last full walk's router as the card holds it: the layer's router logits, one per expert
/// (`N_EXPERT` f32, before the sigmoid and the selection bias), and the `N_USED` experts it picked,
/// in the router's slot order. They are the full row's, the one the head read. Blocking; gate use.
pub fn nextn_router(m: &mut GpuModel<Body>) -> Result<(Vec<f32>, Vec<u32>), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let nx = body
        .nextn
        .as_deref()
        .ok_or_else(|| shape("a router on a load without the NextN layer".to_string()))?;
    let stream = gpu.stream();
    Ok((
        nx.s.rout.logits.to_host_vec(stream)?,
        nx.s.rout.ids.to_host_vec(stream)?,
    ))
}

/// The NextN layer's store as the card holds it: its latent rows and index
/// rows, `n` positions of each. Blocking; gate use.
pub fn nextn_store(m: &mut GpuModel<Body>, n: usize) -> Result<(Vec<u16>, Vec<u16>), GpuError> {
    let (gpu, _, body) = m.body_parts(WHAT)?;
    let nx = body
        .nextn
        .as_deref()
        .ok_or_else(|| shape("a store on a load without the NextN layer".to_string()))?;
    if n > nx.ctx {
        return Err(shape(format!("{n} positions of a store of {}", nx.ctx)));
    }
    let stream = gpu.stream();
    let mut lat = nx.latent.buf().to_host_vec(stream)?;
    let mut idx = nx.index_rows.buf().to_host_vec(stream)?;
    lat.truncate(n * LATENT);
    idx.truncate(n * INDEX_ROW);
    Ok((lat, idx))
}
