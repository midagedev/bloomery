//! The Qwen3.8-Flash-Next (`qwen4exp`) `ChainBody`: 48 layers in four
//! gated-residual streams — 36 sigmoid-gated delta-rule layers and 12
//! attention layers whose positions a mean-pool selector picks — each
//! routing to 512 experts on the host tier beside a sigmoid-gated shared
//! expert on the card, a PLE site on layer 1, and a head whose mix is the
//! final norm ([`HeadNorm::Mixed`]).
//!
//! Load ([`Body38::open_placed`]) reads the description once
//! (`model::arch::qwen35moe::place::PlanInputs`), checks the chain's shape
//! against the kernels' geometry (`plan38`), refuses by name every feature
//! of the file the program does not run past the two it allows
//! ([`ALLOWED`]), joins each delta layer's β and α into one stack and each
//! router with the shared expert's gate as its last row, and builds the
//! stores, two arenas (the step's one row, an eager pass's
//! [`PASS_ROWS`](super::scratch38::PASS_ROWS)), the input records, and the
//! host tier over every layer's routed stacks (no routed expert on the
//! card).
//!
//! The walk is `program38`'s. The decode step is captured; a prompt runs
//! either as captured steps, one a position ([`Prompt38::Step`]), or as
//! eager passes of up to eight positions through the batch port
//! ([`Prompt38::Pass`]) — each row bit for bit its step.
//!
//! A verify ([`Rows`]) runs 2 to [`VERIFY_ROWS`] consecutive positions as
//! one captured pass through the step port's `Cols` chain into one head of
//! its rows, each row bit for bit its step, and stands the model that many
//! positions on until its commit ([`Rollback`], `GpuModel::rollback` to the
//! first position not kept) keeps the first `k`: every delta layer's state
//! after each row is in a lane of its own, so the commit moves the lane
//! word to row `k − 1`'s ([`kept_lane`]) and copies nothing; the PLE hash's
//! history is replayed over the kept rows. Every other store is by
//! position, and nothing reads a row past the count: a pooled key a
//! rejected row completed is read only at a count that completes it again,
//! and the row at that count writes it before its select reads it. A step,
//! a pass or a verify is refused by name while a verify waits for its
//! commit.
//!
//! The PLE rows are the host's: each position's n-gram hash
//! (`engram::Hash::ple_rows_into`) names 16 rows of the IQ4_NL table, which
//! stays in the file; they are decoded (`gguf::quant::dequant_row`) into the
//! position's 2,560 values and copied to the arena ahead of the step. The
//! hash's history is the sequence's: a step at any position other than the
//! next is refused by name, and `reset` starts a new one.

use super::plan38::{self, GDN, Kind38, Layer38, Shape38, beta_alpha, geo, router};
use super::program38::{
    Ctx38, Kernels38, Parts38, Pass38, STEP_MEMOPS, Step38, Verify38, step_launches,
    verify_launches,
};
use super::scratch::{Io, LANE, RopeRows, StepParams, f32_view};
use super::scratch38::{
    Arena38, LANES, LaneWord, PASS_ROWS, PassRecord, Store38, Taps38, VERIFY_ROWS, dims, kept_lane,
    store_rule_bytes,
};
use crate::head::{Head, HeadNorm};
use crate::host::run::{HostRun, HostWidths};
use crate::host::{BatchLeg, StepLeg};
use crate::hybrid::{Boundary, BoundaryShape, Chain, HostResidency, Hybrid, Refusal, SlotMap};
use crate::model::{ChainBody, GpuModel, HostServed, Rollback, RowHeads, Rows};
use crate::rope_table::{RopeSpec, RopeTable};
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError, launch_u32, ple};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use engram::Hash;
use engram::hash::{History, Window};
use gguf::quant::dequant_row;
use gguf::{GgmlType, Split, TensorInfo};
use model::arch::Arch;
use model::placement::Plan;
use std::ops::Range;
use std::sync::Arc;

const WHAT: &str = "qwen4exp Body38";

/// The coverage items the program opens a file with: the tokenizer's
/// pre-tokenizer and the template's tool-call parser are the chat surface's,
/// which the program does not use. Every other item the check lists is
/// refused by name.
pub const ALLOWED: &[&str] = &[
    "pre-tokenizer qwen35",
    "a tool-call parser for this template",
];

/// The Qwen3.8 model: one card, the skeleton over this body.
pub type Qwen38Model = GpuModel<Body38>;

/// How a prompt runs: captured steps, one a position — the decode step's
/// graph — or eager passes of up to eight positions through the host tier's
/// batch port. Both leave every position's state and logits bit for bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prompt38 {
    Step,
    Pass,
}

impl Prompt38 {
    /// Positions an eager pass takes at most.
    pub const PASS_ROWS: usize = PASS_ROWS;

    /// The path a command line names: `step` or `pass`. The ubatch GEMM
    /// paths (`gemm`, `auto`) are refused by name with what they lack here,
    /// as is anything else.
    pub fn parse(s: &str) -> Result<Prompt38, GpuError> {
        match s {
            "step" => Ok(Prompt38::Step),
            "pass" => Ok(Prompt38::Pass),
            "gemm" | "auto" => Err(GpuError::shape(
                WHAT,
                format!(
                    "prompt path {s:?}: qwen4exp has no ubatch GEMM path — no gemm_q8_0 entry \
                     (every card matrix is Q8_0 with f32 activations), GemmAct takes K a \
                     multiple of 256 (the shared expert's down reads K 640, the \
                     hyper-connection up K 320), and GEMM_MAX_SLOTS bounds one route table; \
                     the prompt runs `step` (captured steps) or `pass` (eager passes of up to \
                     {PASS_ROWS})"
                ),
            )),
            other => Err(GpuError::shape(
                WHAT,
                format!("prompt path {other:?}: `step` or `pass`"),
            )),
        }
    }

    /// The name a record prints.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Prompt38::Step => "step",
            Prompt38::Pass => "pass",
        }
    }
}

/// A layer's mixer, as a gate lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind38 {
    /// The sigmoid-gated delta rule over its recurrent store.
    Gdn,
    /// Selecting gated attention over its K/V planes.
    Qsa,
}

/// The PLE site's host side: the hash and the sequence's history, the table
/// in the file, and the rows of the positions last filled, decoded.
struct PleHost {
    hash: Hash,
    hist: History,
    file: Arc<Split>,
    shard: usize,
    info: TensorInfo,
    /// Table rows, values a row, bytes a row.
    rows: usize,
    width: usize,
    row_bytes: usize,
    /// Row ids of up to a pass's positions, and their decoded values
    /// (`HIDDEN` a position).
    ids: Vec<u32>,
    e: Vec<f32>,
}

impl PleHost {
    /// The table `per_layer_token_embd` of `file`, IQ4_NL rows of the hash's
    /// width, and the hash from the first shard's keys. Load-time only.
    fn new(file: Arc<Split>, n_vocab: usize) -> Result<PleHost, GpuError> {
        let name = model::arch::qwen35moe::names::per_layer_token_embd();
        let refuse = |need: &'static str| GpuError::Tensor {
            what: WHAT,
            name: name.clone(),
            need,
        };
        let path = file
            .shard_path(0)
            .ok_or(GpuError::state(WHAT, "the file's first shard"))?;
        let meta = gguf::inventory_of(path).map_err(|e| GpuError::plan(WHAT, e))?;
        let vocab = u32::try_from(n_vocab)
            .map_err(|_| GpuError::shape(WHAT, format!("a vocabulary of {n_vocab}")))?;
        let hash =
            Hash::ple_from_gguf(&meta, "qwen4exp", vocab).map_err(|e| GpuError::plan(WHAT, e))?;
        let hist = hash
            .new_history()
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        let (shard, info) = file
            .find(&name)
            .map(|(s, t)| (s, t.clone()))
            .ok_or_else(|| refuse("in the file"))?;
        let width = info.dims.first().copied().unwrap_or(0) as usize;
        let rows = info.dims.get(1).copied().unwrap_or(0) as usize;
        let per_token = (geo::HIDDEN).checked_div(width).unwrap_or(0);
        if info.ty != GgmlType::IQ4_NL
            || width == 0
            || !width.is_multiple_of(32)
            || per_token * width != geo::HIDDEN
        {
            return Err(refuse("IQ4_NL rows of a whole part of the model width"));
        }
        let row_bytes = width / 32 * 18;
        if rows == 0 || info.nbytes != (rows * row_bytes) as u64 {
            return Err(refuse("a whole number of IQ4_NL rows"));
        }
        Ok(PleHost {
            hash,
            hist,
            file,
            shard,
            info,
            rows,
            width,
            row_bytes,
            ids: vec![0; PASS_ROWS * per_token],
            e: vec![0.0; PASS_ROWS * geo::HIDDEN],
        })
    }

    /// The rows of `tokens` at positions `pos ..`, the history moved past
    /// them, decoded into the first `tokens.len()` positions of `e`. A token
    /// the hash refuses (the image placeholder, an id past the vocabulary),
    /// a position other than the history's next, or a row past the table is
    /// refused by name before the history moves.
    fn fill(&mut self, pos: u32, tokens: &[u32]) -> Result<(), GpuError> {
        let n = tokens.len();
        let per_token = geo::HIDDEN / self.width;
        let ids = self
            .ids
            .get_mut(..n * per_token)
            .ok_or_else(|| GpuError::shape(WHAT, format!("{n} PLE positions at once")))?;
        let mut hist = self.hist.clone();
        self.hash
            .ple_rows_into(&mut hist, u64::from(pos), tokens, ids)
            .map_err(|e| GpuError::shape(WHAT, format!("PLE rows at position {pos}: {e}")))?;
        if let Some(&r) = ids.iter().find(|&&r| r as usize >= self.rows) {
            return Err(GpuError::shape(
                WHAT,
                format!("PLE row {r} past the table's {} rows", self.rows),
            ));
        }
        let data = self
            .file
            .shard(self.shard)
            .ok_or(GpuError::state(WHAT, "the PLE table's shard"))?
            .data(&self.info)?;
        for (j, &r) in ids.iter().enumerate() {
            let src = &data[r as usize * self.row_bytes..][..self.row_bytes];
            let out = &mut self.e[j * self.width..][..self.width];
            dequant_row(GgmlType::IQ4_NL, src, out).map_err(model::ModelError::from)?;
        }
        self.hist = hist;
        Ok(())
    }

    /// The history a verify's `fill` moved past its rows, moved instead
    /// from `before` (the history as it stood before that fill) past the
    /// `kept` rows from position `pos`: the history a fill of those rows
    /// alone leaves.
    fn keep(&mut self, before: History, pos: u32, kept: &[u32]) -> Result<(), GpuError> {
        let per_token = geo::HIDDEN / self.width;
        let ids = self.ids.get_mut(..kept.len() * per_token).ok_or_else(|| {
            GpuError::shape(WHAT, format!("{} PLE positions at once", kept.len()))
        })?;
        let mut hist = before;
        self.hash
            .ple_rows_into(&mut hist, u64::from(pos), kept, ids)
            .map_err(|e| GpuError::shape(WHAT, format!("PLE rows at position {pos}: {e}")))?;
        self.hist = hist;
        Ok(())
    }

    /// A new sequence's history.
    fn restart(&mut self) -> Result<(), GpuError> {
        self.hist = self
            .hash
            .new_history()
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        Ok(())
    }
}

/// Refused by name, `what` the caller: the first of `tokens` the raw-id PLE
/// hash of `window` does not take (the image placeholder, an id past the
/// vocabulary), with its index in the call, or a hash that is not raw-id.
/// The hash's own test ([`engram::hash::EosWindow::check`]), so a call's
/// check before its first launch and each fill refuse the same ids.
fn ple_takes(what: &'static str, window: &Window, tokens: &[u32]) -> Result<(), GpuError> {
    let Window::Eos(w) = window else {
        return Err(GpuError::state(what, "a raw-id PLE hash"));
    };
    for (i, &t) in tokens.iter().enumerate() {
        w.check(t)
            .map_err(|e| GpuError::shape(what, format!("token {i}: {e}")))?;
    }
    Ok(())
}

/// A layer's store read back ([`Body38::stores_host`]): a delta layer's
/// state and conv ring, or a selecting layer's K/V planes (f16 bits) and its
/// raw and pooled indexer keys.
#[derive(Clone, Debug, PartialEq)]
pub enum Store38Host {
    Rec {
        state: Vec<f32>,
        ring: Vec<f32>,
    },
    Qsa {
        k: Vec<u16>,
        v: Vec<u16>,
        raw: Vec<u16>,
        pooled: Vec<u16>,
    },
}

impl Store38Host {
    /// Bit equality: the f32 values by their bits, so a NaN equals itself
    /// and `-0.0` differs from `0.0`.
    #[must_use]
    pub fn same_bits(&self, other: &Store38Host) -> bool {
        let bits = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        match (self, other) {
            (
                Store38Host::Rec { state: s, ring: r },
                Store38Host::Rec {
                    state: s2,
                    ring: r2,
                },
            ) => bits(s, s2) && bits(r, r2),
            (Store38Host::Qsa { .. }, Store38Host::Qsa { .. }) => self == other,
            _ => false,
        }
    }
}

/// A layer's router at the step's row, read back
/// ([`Body38::route_taps`]).
#[derive(Clone, Debug)]
pub struct RouteTap {
    /// The routed experts' logits, the shared expert's gate left out.
    pub logits: Vec<f32>,
    /// The routed slots' expert ids, in slot order.
    pub ids: Vec<u32>,
}

/// A verify that waits for its commit: its first position, its tokens, and
/// the PLE history as it stood before them.
struct Pending38 {
    pos0: u32,
    tokens: Vec<u32>,
    hist: History,
}

/// One step's host values: its token and position.
#[derive(Clone, Copy, Debug)]
pub struct DecodeInput38 {
    token: u32,
    pos: u32,
}

/// Everything Qwen3.8's chain owns. Field order is drop order: the host tier
/// (its lock over the host set, its page windows) before the buffers.
pub struct Body38 {
    hybrid: Hybrid<HostRun>,
    plans: Vec<Layer38>,
    stores: Vec<Store38>,
    ple_ring: DeviceBuffer<f32>,
    ple: PleHost,
    rope: RopeRows,
    /// The step's one-row arena and record; an eager pass's arena, record
    /// and host sums.
    s: Arena38,
    sp: StepParams,
    a: Arena38,
    rp: PassRecord,
    pass_hsum: DeviceBuffer<f32>,
    k: Kernels38,
    /// The slot map's card copy: every place the host's.
    slots: DeviceTensor<u32>,
    /// The lane word every delta launch reads, and the verify waiting for
    /// its commit.
    lane: LaneWord,
    pending: Option<Pending38>,
    /// Each layer's streams after it and its route, when a gate armed them.
    taps: Option<Taps38>,
    eps: f32,
    vocab: usize,
    ctx: usize,
    /// Positions fed since the load or the last reset.
    fed: u32,
}

impl Body38 {
    /// Card `card` of `plan`, which `inputs` made, resident: the plan's
    /// segments, the joins ([`Body38::derive`]) and the body over them with
    /// the host tier over every layer's routed experts, holding the load's
    /// host set as `host` asks. Refused by name: a coverage item past
    /// [`ALLOWED`], a plan of more than one card or not every layer, a layer
    /// or a width the kernels do not take.
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<Qwen38Model, GpuError> {
        let refused: Vec<String> = inputs
            .unimplemented()
            .into_iter()
            .map(|u| match u.layer {
                Some(l) => format!("{} (layer {l})", u.feature),
                None => u.feature,
            })
            .filter(|f| !ALLOWED.contains(&f.as_str()))
            .collect();
        if !refused.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the file needs what the program does not run: {}",
                    refused.join("; ")
                ),
            ));
        }
        let n = inputs.spec.layers.len();
        let layers = plan
            .machine
            .cards
            .get(card)
            .map(|c| c.layers.clone())
            .ok_or_else(|| GpuError::shape(WHAT, format!("the plan has no card {card}")))?;
        if plan.machine.cards.len() != 1 || layers != (0..n) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "card {card} runs layers {layers:?} of a plan over {} cards; the program runs \
                     every layer (0..{n}) on one card",
                    plan.machine.cards.len()
                ),
            ));
        }
        let shape = plan38::read(&inputs.spec)?;
        GpuModel::load_placed(
            file,
            plan,
            card,
            host,
            |stream, _, layers, w| Body38::derive(stream, &shape.kinds, layers, w),
            |gpu, file, w, residency| {
                Body38::load_placed(gpu, file, w, plan, inputs, &shape, host, residency)
            },
        )
    }

    /// The joins: each delta layer's β and α into one F32 stack (β's rows
    /// first), each router with the shared expert's gate as row 512. Each
    /// row keeps its file bits.
    fn derive(
        stream: &CudaStream,
        kinds: &[Kind38],
        layers: Range<usize>,
        w: &mut Weights,
    ) -> Result<(), GpuError> {
        for l in layers {
            let kind = kinds
                .get(l)
                .ok_or_else(|| GpuError::shape(WHAT, format!("layer {l} past the description")))?;
            if *kind == Kind38::Gdn {
                let (b, a) = (
                    model::arch::qwen35moe::names::ssm_beta(l),
                    model::arch::qwen35moe::names::ssm_alpha(l),
                );
                w.join_rows(stream, &[&b, &a], beta_alpha(l))?;
            }
            let (r, sh) = (
                model::arch::qwen35moe::names::ffn_gate_inp(l),
                model::arch::qwen35moe::names::ffn_gate_inp_shexp(l),
            );
            w.join_rows(stream, &[&r, &sh], router(l))?;
        }
        Ok(())
    }

    /// The body of the plan's one card over the chain's `shape` (module doc).
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's card handle, file, weights, plan, inputs and chain shape, and the host tier's residency and levers (rust-quality R8)"
    )]
    fn load_placed(
        gpu: &Gpu,
        file: Split,
        w: &Weights,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        shape: &Shape38,
        host: HostCfg,
        residency: HostResidency,
    ) -> Result<Body38, GpuError> {
        let (spec, hp) = (&inputs.spec, &inputs.hp);
        let n = spec.layers.len();
        let ctx = usize::try_from(plan.ctx_max)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| GpuError::shape(WHAT, format!("ctx_max {}", plan.ctx_max)))?;
        let Shape38 {
            kinds,
            router: router_dims,
            base,
            ple: ple_layer,
        } = shape;
        let (router_dims, base) = (*router_dims, *base);
        let plans = plan38::plans(w, kinds, *ple_layer)?;
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let d = dims(router_dims, ctx);
        let stores = kinds
            .iter()
            .map(|k| match k {
                Kind38::Gdn => Store38::rec(stream),
                Kind38::Qsa => Store38::qsa(stream, &d),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (rec, sel) = store_rule_bytes(ctx);
        for (l, (st, k)) in stores.iter().zip(kinds).enumerate() {
            let want = match k {
                Kind38::Gdn => rec,
                Kind38::Qsa => sel,
            };
            if st.bytes() as u64 != want {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}'s store holds {} bytes, runtime::stores counts {want}",
                        st.bytes()
                    ),
                ));
            }
        }
        let ple_ring = DeviceBuffer::zeroed(stream, ple::ring_len(geo::STREAMS))?;
        let rope = RopeTable::new(&RopeSpec::window(base, geo::ROPE))?;
        let rope = RopeRows::new(stream, &rope, geo::ROPE, ctx)?;
        let s = Arena38::new(stream, router_dims, 1, ctx)?;
        let sp = StepParams::new(stream, true)?;
        let a = Arena38::new(stream, router_dims, PASS_ROWS, ctx)?;
        let rp = PassRecord::new(stream)?;
        let pass_hsum = DeviceBuffer::zeroed(stream, PASS_ROWS * geo::HIDDEN)?;
        let run = 0..n;
        let map = SlotMap::prefix(run.clone(), geo::EXPERTS, 0)?;
        let slots = DeviceTensor::upload(stream, map.as_slice(), n, geo::EXPERTS)?;
        let boundary = Boundary::with_cols(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: geo::HIDDEN,
                n_used: geo::N_USED,
            },
            1,
            VERIFY_ROWS,
        )?;
        let file = Arc::new(file);
        let widths = HostWidths {
            embd: geo::HIDDEN,
            ff: geo::FF,
            n_used: geo::N_USED,
        };
        let mut experts = HostRun::build(Arc::clone(&file), 0, host.r8, widths, |src| {
            model::arch::qwen35moe::host::layers(src, hp, run.clone())
        })?;
        experts.prepare_union(PASS_ROWS)?;
        let mut hybrid = Hybrid::new(boundary, map, experts, n)?;
        hybrid.watch_fault(gpu.fault_word())?;
        hybrid.keep_residency(residency);
        hybrid.prepare_batch(gpu.context(), PASS_ROWS)?;
        let ple = PleHost::new(file, spec.vocab as usize)?;
        let mut body = Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            ple,
            rope,
            s,
            sp,
            a,
            rp,
            pass_hsum,
            k: Kernels38::load(gpu)?,
            slots,
            lane: LaneWord::new(stream)?,
            pending: None,
            taps: None,
            eps: spec.rms_eps,
            vocab: spec.vocab as usize,
            ctx,
            fed: 0,
        };
        body.sp.write(stream, 0, 0)?;
        stream.synchronize()?;
        Ok(body)
    }

    /// Every layer's mixer, in order.
    #[must_use]
    pub fn kinds(&self) -> Vec<LayerKind38> {
        self.plans
            .iter()
            .map(|p| match p.mixer {
                super::plan38::Mixer38::Gdn(_) => LayerKind38::Gdn,
                super::plan38::Mixer38::Qsa(_) => LayerKind38::Qsa,
            })
            .collect()
    }

    /// The captured step's launches ([`super::program38`]'s count), and how
    /// many of them are stream memory-operation batches.
    #[must_use]
    pub fn step_launches(&self) -> (usize, usize) {
        (step_launches(&self.plans), STEP_MEMOPS * self.plans.len())
    }

    /// Positions every store holds.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// Tokens of the vocabulary.
    #[must_use]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Device bytes of the layers' stores and the PLE ring.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(Store38::bytes).sum::<usize>() + self.ple_ring.num_bytes()
    }

    /// The host tier.
    #[must_use]
    pub fn hybrid(&self) -> &Hybrid<HostRun> {
        &self.hybrid
    }

    /// Arm (or disarm) the per-layer taps ([`GpuModel::set_layer_taps`], the
    /// one caller, drops the captures first). Load-time allocation.
    fn set_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        self.taps = if on {
            Some(Taps38::new(gpu.stream(), self.plans.len(), &self.s.route)?)
        } else {
            None
        };
        Ok(())
    }

    /// Every layer's streams after the last step, `4 · 2560` a layer, stream
    /// `s` of layer `l` at `l · 10240 + s · 2560`. Blocking; refused when the
    /// taps are not armed.
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        let taps = self
            .taps
            .as_ref()
            .ok_or(GpuError::state(WHAT, "armed taps (set_layer_taps)"))?;
        let mut out = Vec::with_capacity(taps.out.len() * geo::STREAMS * geo::HIDDEN);
        for t in &taps.out {
            out.extend(t.to_host_vec(gpu.stream())?);
        }
        Ok(out)
    }

    /// Every layer's router after the last step: its [`geo::EXPERTS`]
    /// logits and its [`geo::N_USED`] routed ids, in slot order. Blocking;
    /// refused when the taps are not armed.
    pub fn route_taps(&self, gpu: &Gpu) -> Result<Vec<RouteTap>, GpuError> {
        let taps = self
            .taps
            .as_ref()
            .ok_or(GpuError::state(WHAT, "armed taps (set_layer_taps)"))?;
        taps.logits
            .iter()
            .zip(&taps.ids)
            .map(|(lg, id)| {
                let (mut lg, mut id) =
                    (lg.to_host_vec(gpu.stream())?, id.to_host_vec(gpu.stream())?);
                if lg.len() < geo::EXPERTS || id.len() < geo::N_USED {
                    return Err(GpuError::shape(
                        WHAT,
                        format!("route taps of {} logits and {} ids", lg.len(), id.len()),
                    ));
                }
                lg.truncate(geo::EXPERTS);
                id.truncate(geo::N_USED);
                Ok(RouteTap {
                    logits: lg,
                    ids: id,
                })
            })
            .collect()
    }

    /// Every layer's store and the PLE ring, read back: what a run leaves
    /// behind, for a gate to compare bit for bit — a delta store's state as
    /// the committed lane holds it. Blocking.
    pub fn stores_host(&self, gpu: &Gpu) -> Result<(Vec<Store38Host>, Vec<f32>), GpuError> {
        let stream = gpu.stream();
        let stores = self
            .stores
            .iter()
            .map(|s| {
                Ok(match s {
                    Store38::Rec { rec, .. } => Store38Host::Rec {
                        state: lane_of(rec.state.to_host_vec(stream)?, self.lane.lane())?,
                        ring: rec.ring.to_host_vec(stream)?,
                    },
                    Store38::Qsa { kv, raw, pooled } => Store38Host::Qsa {
                        k: kv.k.to_host_vec(stream)?,
                        v: kv.v.to_host_vec(stream)?,
                        raw: raw.to_host_vec(stream)?,
                        pooled: pooled.to_host_vec(stream)?,
                    },
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        Ok((stores, self.ple_ring.to_host_vec(stream)?))
    }

    /// Every selecting store's K/V planes and raw and pooled keys set to
    /// `bits`, an f16 pattern: a gate's stand-in for the rows `reset` leaves
    /// as they were, so a path that reads a row it did not write reads the
    /// pattern, not a previous run's value. Synchronizes; gate use.
    pub fn fill_planes(&mut self, gpu: &Gpu, bits: u16) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for s in &mut self.stores {
            if let Store38::Qsa { kv, raw, pooled } = s {
                for buf in [&mut kv.k, &mut kv.v, raw, pooled] {
                    buf.copy_from_host(stream, &vec![bits; buf.len()])?;
                }
            }
        }
        Ok(())
    }

    /// Every selecting store's pooled keys from pool `from` on set to
    /// `bits`, an f16 pattern: a gate's stand-in for pools a rejected row
    /// left past the count, which no select may read before the row that
    /// completes them writes them again. Synchronizes; gate use.
    pub fn poison_pools_from(&mut self, gpu: &Gpu, from: usize, bits: u16) -> Result<(), GpuError> {
        let stream = gpu.stream();
        for s in &mut self.stores {
            if let Store38::Qsa { pooled, .. } = s {
                let mut rows = pooled.to_host_vec(stream)?;
                let at = (from * crate::qsa::DIM).min(rows.len());
                rows[at..].fill(bits);
                pooled.copy_from_host(stream, &rows)?;
            }
        }
        Ok(())
    }

    /// The lane the lane word holds: the committed delta state's.
    #[must_use]
    pub fn lane(&self) -> u32 {
        self.lane.lane()
    }

    /// The lane word set to `lane` with no verify behind it — a gate's
    /// stand-in for a commit that moved the word to a lane no call wrote:
    /// the next launch that reads it raises the stamp's fault. Gate use.
    pub fn plant_lane(&mut self, gpu: &Gpu, lane: u32) -> Result<(), GpuError> {
        self.lane.set(gpu.stream(), lane)
    }

    /// The captured verify's launches at `m` rows ([`super::program38`]'s
    /// count).
    #[must_use]
    pub fn verify_launches(&self, m: usize) -> usize {
        verify_launches(&self.plans, m)
    }

    /// The step's position `pos` is the next one and inside the stores, and
    /// no verify waits for its commit, else refused by name.
    fn check_next(&self, pos: u32, n: usize) -> Result<(), GpuError> {
        if let Some(p) = &self.pending {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{n} positions from {pos} while the verify of {} rows at {} waits for its \
                     commit (GpuModel::rollback to the first position not kept)",
                    p.tokens.len(),
                    p.pos0
                ),
            ));
        }
        let end = pos as usize + n;
        if pos != self.fed || end > self.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{n} positions from {pos} after {} positions, in stores of {}",
                    self.fed, self.ctx
                ),
            ));
        }
        Ok(())
    }

    /// Plan an eager pass of `tokens` (1..=8) at `pos`, the next position:
    /// its PLE rows, its record and its rows' copy to the pass arena.
    fn plan_pass(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        let n = tokens.len();
        if self.taps.is_some() {
            return Err(GpuError::state(WHAT, "layer taps off (a pass writes none)"));
        }
        self.check_next(pos, n)?;
        self.ple.fill(pos, tokens)?;
        self.rp.write(stream, tokens, pos)?;
        // SAFETY: the pass arena holds `PASS_ROWS · HIDDEN` rows of `e` and
        // `n <= PASS_ROWS` (the record's write refused more), and `e` stays in
        // place while the window lives (one synchronous copy).
        let mut e = unsafe { f32_view(&self.a.ple.e, 0, n * geo::HIDDEN) };
        e.copy_from_host(stream, &self.ple.e[..n * geo::HIDDEN])?;
        self.fed += launch_u32(WHAT, "positions", n)?;
        Ok(())
    }

    /// Enqueue the planned pass of `m` rows through the batch port, then the
    /// head of its last row into `head` when given.
    fn walk_pass(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        m: usize,
        head: Option<&mut Head>,
    ) -> Result<(), GpuError> {
        if m > self.a.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a pass of {m} rows on an arena of {}", self.a.rows),
            ));
        }
        let io = Io {
            lane: Some(self.lane.word()),
            ..self.rp.io(m)?
        };
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            rope,
            a,
            pass_hsum,
            k,
            slots,
            eps,
            ctx,
            ..
        } = self;
        let mut prog = Pass38 {
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: &rope.table,
                    ctx: *ctx,
                },
                plans,
                stores,
                ple_ring,
                s: a,
                io: &io,
                m,
                cur: 0,
                taps: None,
                slots,
                each: false,
            },
        };
        let mut leg = BatchLeg::new(gpu.stream(), hybrid, pass_hsum, PASS_ROWS);
        prog.walk(&mut leg)?;
        match head {
            Some(h) => prog.head(h),
            None => Ok(()),
        }
    }
}

impl GpuModel<Body38> {
    /// Arm (or disarm) the per-layer taps: after each layer the step copies
    /// its streams into the layer's tap, which [`Body38::taps`] reads back,
    /// and after each router its logits and ids ([`Body38::route_taps`]).
    /// Every captured chain is dropped first, the mode kept: a capture
    /// records the copies into the taps armed at its time, so a replay after
    /// a change would write freed memory or leave the new taps unwritten.
    /// Load-time allocation.
    pub fn set_layer_taps(&mut self, on: bool) -> Result<(), GpuError> {
        self.drop_captures();
        let (gpu, _, body) = self.body_parts("qwen4exp set_layer_taps")?;
        body.set_taps(gpu, on)
    }

    /// Feed `tokens` from where the model stands by `path` and return the
    /// greedy next token after the last one: captured steps, one a position,
    /// or eager passes of up to eight positions, the last one ending in its
    /// last row's head. Either leaves every position's state and logits bit
    /// for bit. A prompt past the stores, or with an id the embedding or the
    /// PLE hash does not take (one past the vocabulary, the image
    /// placeholder), is refused before any launch, so either path leaves the
    /// model where it stood; the layer taps must be off for passes.
    pub fn prompt38(&mut self, tokens: &[u32], path: Prompt38) -> Result<u32, GpuError> {
        const WHAT_P: &str = "qwen4exp prompt";
        if tokens.is_empty() {
            return Err(GpuError::shape(WHAT_P, "empty token slice"));
        }
        if let Some(fault) = self.poisoned() {
            return Err(GpuError::Poisoned {
                what: WHAT_P,
                fault,
            });
        }
        let body = self.body(WHAT_P)?;
        super::refuse_past_vocab(WHAT_P, tokens, body.vocab)?;
        ple_takes(WHAT_P, body.ple.hash.window(), tokens)?;
        let (pos, ctx) = (self.pos() as usize, body.ctx());
        if pos + tokens.len() > ctx {
            return Err(GpuError::shape(
                WHAT_P,
                format!(
                    "a prompt of {} ids from position {pos} passes the stores' {ctx} positions",
                    tokens.len()
                ),
            ));
        }
        match path {
            Prompt38::Step => self.step(tokens),
            Prompt38::Pass => {
                let n = tokens.len().div_ceil(PASS_ROWS);
                let mut next = None;
                for (i, chunk) in tokens.chunks(PASS_ROWS).enumerate() {
                    let last = i + 1 == n;
                    let pos = self.pos();
                    {
                        let (gpu, _, body) = self.body_parts(WHAT_P)?;
                        body.plan_pass(gpu.stream(), chunk, pos)?;
                    }
                    let m = chunk.len();
                    next = self.run_rows(m, WHAT_P, |gpu, w, body, head, _| {
                        body.walk_pass(gpu, w, m, last.then_some(head))?;
                        Ok(last)
                    })?;
                }
                next.ok_or(GpuError::state(WHAT_P, "a token read after the last pass"))
            }
        }
    }
}

impl ChainBody for Body38 {
    type Input = DecodeInput38;
    type Host = Body38;

    fn arch() -> Arch {
        Arch::Qwen35moe
    }

    /// The step at `pos`, the next position: its PLE rows read from the file.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput38, GpuError> {
        self.check_next(pos, 1)?;
        self.ple.fill(pos, &[token])?;
        self.fed += 1;
        Ok(DecodeInput38 { token, pos })
    }

    /// The step's input record (its position, its token, the lane word) and
    /// its PLE rows: two copies ahead of the step's launches.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput38) -> Result<(), GpuError> {
        self.s
            .ple
            .e
            .copy_from_host(stream, &self.ple.e[..geo::HIDDEN])?;
        self.sp.write(stream, input.token, input.pos)
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        let io = Io {
            lane: Some(self.lane.word()),
            ..self.sp.io()
        };
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            rope,
            s,
            k,
            slots,
            taps,
            eps,
            ctx,
            ..
        } = self;
        let prog = Step38 {
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: &rope.table,
                    ctx: *ctx,
                },
                plans,
                stores,
                ple_ring,
                s,
                io: &io,
                m: 1,
                cur: 0,
                taps: taps.as_mut(),
                slots,
                each: false,
            },
            head,
        };
        let mut leg = StepLeg::new(gpu.stream(), hybrid);
        prog.walk(&mut leg)
    }

    /// Every delta store's lanes and the PLE ring back to zero, every lane's
    /// stamp to a fresh store's, the lane word to 0, a new PLE history, no
    /// verify waiting, after the host tier's reset. The K/V planes and the
    /// raw and pooled keys need nothing: nothing reads a row at or past a
    /// live count, and every row below it is written by its own step first.
    /// Synchronizes.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        self.hybrid.reset(stream)?;
        let most = self
            .stores
            .iter()
            .map(|s| match s {
                Store38::Rec { rec, .. } => rec.state.len().max(rec.ring.len()),
                Store38::Qsa { .. } => 0,
            })
            .max()
            .unwrap_or(0);
        let zeros = vec![0.0f32; most];
        for s in &mut self.stores {
            if let Store38::Rec { rec, .. } = s {
                rec.clear(stream, &zeros)?;
            }
            s.restamp(stream)?;
        }
        self.lane.set(stream, 0)?;
        self.pending = None;
        self.k
            .ple
            .reset_ring(stream, &mut self.ple_ring, geo::STREAMS)?;
        self.ple.restart()?;
        self.fed = 0;
        self.sp.write(stream, 0, 0)?;
        debug_assert_eq!(LANE, 0);
        stream.synchronize()?;
        Ok(())
    }

    fn head_eps(&self) -> f32 {
        self.eps
    }

    /// The head's mix is the final norm.
    fn head_norm(&self) -> HeadNorm {
        HeadNorm::Mixed
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes()
            + self.rope.table.num_bytes()
            + self.s.bytes()
            + self.sp.bytes()
            + self.a.bytes()
            + self.rp.bytes()
            + self.pass_hsum.num_bytes()
            + self.slots.buf().num_bytes()
            + self.lane.bytes()
            + self.hybrid.boundary().device_bytes()
            + self.taps.as_ref().map_or(0, Taps38::bytes)
    }

    fn layers(&self) -> Range<usize> {
        0..self.plans.len()
    }

    fn host(&mut self) -> Option<&mut Body38> {
        Some(self)
    }
}

impl HostServed for Body38 {
    fn serve_captured(&mut self, chain: Chain) -> Result<(), GpuError> {
        self.hybrid.serve_captured_of(chain)
    }

    fn take_host_refusal(&mut self) -> Option<Refusal> {
        self.hybrid.take_step_refusal()
    }

    fn host_residency(&self) -> Option<&HostResidency> {
        self.hybrid.residency()
    }
}

impl Body38 {
    /// Plan a verify of `tokens` (2..=[`VERIFY_ROWS`]) at `pos`, the next
    /// position: the ids checked as a prompt's are, the PLE rows, the record
    /// and the rows' copy to the pass arena, and the verify left waiting for
    /// its commit. Refused by name before anything moves: another row count, the
    /// taps armed, a verify already waiting, a position other than the next
    /// or past the stores, an id the embedding or the PLE hash does not take.
    fn plan_verify(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        const WHAT_V: &str = "qwen4exp verify";
        let n = tokens.len();
        if !(2..=VERIFY_ROWS).contains(&n) {
            return Err(GpuError::shape(
                WHAT_V,
                format!("a verify of {n} rows; the lanes hold 2..={VERIFY_ROWS}"),
            ));
        }
        if self.taps.is_some() {
            return Err(GpuError::state(
                WHAT_V,
                "layer taps off (a verify writes none)",
            ));
        }
        self.check_next(pos, n)?;
        super::refuse_past_vocab(WHAT_V, tokens, self.vocab)?;
        ple_takes(WHAT_V, self.ple.hash.window(), tokens)?;
        let before = self.ple.hist.clone();
        self.ple.fill(pos, tokens)?;
        self.rp.write(stream, tokens, pos)?;
        // SAFETY: the pass arena holds `PASS_ROWS · HIDDEN` rows of `e` and
        // `n <= VERIFY_ROWS <= PASS_ROWS`, and `e` stays in place while the
        // window lives (one synchronous copy).
        let mut e = unsafe { f32_view(&self.a.ple.e, 0, n * geo::HIDDEN) };
        e.copy_from_host(stream, &self.ple.e[..n * geo::HIDDEN])?;
        self.fed += launch_u32(WHAT_V, "positions", n)?;
        self.pending = Some(Pending38 {
            pos0: pos,
            tokens: tokens.to_vec(),
            hist: before,
        });
        Ok(())
    }

    /// Enqueue the planned verify of `head.m()` rows through the step port
    /// into `head`, one head of the verify's rows. An eager verify needs its
    /// plan; a capture records the launches only.
    fn walk_verify(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        const WHAT_V: &str = "qwen4exp verify";
        let m = head.m();
        let planned = self.pending.as_ref().map(|p| p.tokens.len());
        if planned != Some(m) && !crate::capturing(gpu.stream())? {
            return Err(GpuError::state(
                WHAT_V,
                "a verify planned for this head's rows (plan_rows before the pass)",
            ));
        }
        let io = Io {
            lane: Some(self.lane.word()),
            ..self.rp.io(m)?
        };
        let Body38 {
            hybrid,
            plans,
            stores,
            ple_ring,
            rope,
            a,
            k,
            slots,
            eps,
            ctx,
            ..
        } = self;
        let prog = Verify38 {
            p: Parts38 {
                c: Ctx38 {
                    gpu,
                    w,
                    k,
                    eps: *eps,
                    table: &rope.table,
                    ctx: *ctx,
                },
                plans,
                stores,
                ple_ring,
                s: a,
                io: &io,
                m,
                cur: 0,
                taps: None,
                slots,
                each: true,
            },
            head,
        };
        let mut leg = StepLeg::new(gpu.stream(), hybrid);
        prog.walk(&mut leg)
    }

    /// Keep the first `pos − pos0` rows of the verify waiting for its commit
    /// and take the rest back: the lane word to the last kept row's lane
    /// ([`kept_lane`]), the PLE history over the kept rows. With no
    /// verify waiting only the position the body stands at is taken; any
    /// other, and a count outside the verify's rows, is refused by name.
    fn commit(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        const WHAT_C: &str = "qwen4exp commit";
        let Some(p) = self.pending.take() else {
            if pos == self.fed {
                return Ok(());
            }
            return Err(GpuError::shape(
                WHAT_C,
                format!(
                    "back to position {pos} from {} with no verify waiting: a delta layer keeps \
                     no state for an earlier position",
                    self.fed
                ),
            ));
        };
        let rows = p.tokens.len();
        let kept = pos
            .checked_sub(p.pos0)
            .map(|k| k as usize)
            .filter(|k| (1..=rows).contains(k));
        let Some(kept) = kept else {
            let e = GpuError::shape(
                WHAT_C,
                format!(
                    "back to position {pos}: the verify of {rows} rows at {} keeps 1..={rows} \
                     (row 0 always)",
                    p.pos0
                ),
            );
            self.pending = Some(p);
            return Err(e);
        };
        let stream = gpu.stream();
        self.lane.set(stream, kept_lane(self.lane.lane(), kept))?;
        if kept < rows {
            self.ple.keep(p.hist, p.pos0, &p.tokens[..kept])?;
        }
        self.fed = pos;
        Ok(())
    }
}

/// Lane `lane` of a delta state read back whole (`LANES` lanes of
/// `GDN.state_len()`).
fn lane_of(mut state: Vec<f32>, lane: u32) -> Result<Vec<f32>, GpuError> {
    let len = GDN.state_len();
    let at = lane as usize * len;
    if state.len() != LANES * len || at + len > state.len() {
        return Err(GpuError::shape(
            WHAT,
            format!("lane {lane} of a state of {} values", state.len()),
        ));
    }
    state.truncate(at + len);
    Ok(state.split_off(at))
}

impl Rows for Body38 {
    const MAX_ROWS: usize = VERIFY_ROWS;
    /// Unread: the verify's chain is its rows' ([`Rows::chain_of`]).
    const CHAIN: Chain = Chain::Step;
    const HEADS: RowHeads = RowHeads::One;

    /// One row of `m` columns through the step port.
    fn chain_of(m: usize) -> Chain {
        Chain::Cols(m)
    }

    fn plan_rows(&mut self, stream: &CudaStream, tokens: &[u32], pos: u32) -> Result<(), GpuError> {
        self.plan_verify(stream, tokens, pos)
    }

    /// The verify into `heads[0]`, a head of its rows; any other heads are
    /// refused by name.
    fn enqueue_rows(&mut self, gpu: &Gpu, w: &Weights, heads: &mut [Head]) -> Result<(), GpuError> {
        match heads {
            [head] => self.walk_verify(gpu, w, head),
            _ => Err(GpuError::shape(
                "qwen4exp verify",
                format!(
                    "{} heads; a verify runs into one head of its rows",
                    heads.len()
                ),
            )),
        }
    }
}

impl Rollback for Body38 {
    /// Refused by name: the commit enqueues copies, so it runs with the
    /// card at hand ([`Rollback::rollback_on`], what `GpuModel::rollback`
    /// calls).
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        Err(GpuError::shape(
            "qwen4exp commit",
            format!("back to position {pos} without the card: GpuModel::rollback commits"),
        ))
    }

    fn rollback_on(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        self.commit(gpu, pos)
    }
}

const _: () = assert!(GDN.n_v == geo::V_HEADS);
const _: () = assert!(VERIFY_ROWS <= PASS_ROWS);

#[cfg(test)]
mod tests {
    use super::ple_takes;
    use engram::hash::{EosWindow, Window};

    /// The call's check is the hash's: a text id passes; the image
    /// placeholder and an id past the vocabulary are refused by name with
    /// their index wherever they stand; a hash that is not raw-id is
    /// refused.
    #[test]
    fn a_call_is_checked_by_the_hashs_own_test() {
        let w = Window::Eos(EosWindow {
            eos: 1,
            image: Some(9),
            n_vocab: 16,
        });
        assert!(ple_takes("t", &w, &[0, 1, 15]).is_ok());
        for (ids, at) in [([2, 3, 9], "token 2:"), ([15, 16, 2], "token 1:")] {
            let e = ple_takes("t", &w, &ids).expect_err("the id is refused");
            assert!(e.to_string().contains(at), "{ids:?}: {e}");
        }
        let mapped = Window::Mapped {
            token_map: vec![0; 16],
            pad: 0,
        };
        let e = ple_takes("t", &mapped, &[0]).expect_err("a mapped hash is refused");
        assert!(e.to_string().contains("a raw-id PLE hash"), "{e}");
    }
}
