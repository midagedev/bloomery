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
//! The PLE rows are the host's: each position's n-gram hash
//! (`engram::Hash::ple_rows_into`) names 16 rows of the IQ4_NL table, which
//! stays in the file; they are decoded (`gguf::quant::dequant_row`) into the
//! position's 2,560 values and copied to the arena ahead of the step. The
//! hash's history is the sequence's: a step at any position other than the
//! next is refused by name, and `reset` starts a new one.

use super::plan38::{self, GDN, Kind38, Layer38, beta_alpha, geo, router};
use super::program38::{Ctx38, Kernels38, Parts38, Pass38, STEP_MEMOPS, Step38, step_launches};
use super::scratch::{LANE, RopeRows, StepParams, f32_view};
use super::scratch38::{Arena38, PASS_ROWS, PassRecord, Store38, Taps38, dims, store_rule_bytes};
use crate::head::{Head, HeadNorm};
use crate::host::run::{HostRun, HostWidths};
use crate::host::{BatchLeg, StepLeg};
use crate::hybrid::{Boundary, BoundaryShape, Chain, HostResidency, Hybrid, Refusal, SlotMap};
use crate::model::{ChainBody, GpuModel, HostServed};
use crate::rope_table::{RopeSpec, RopeTable};
use crate::weights::Weights;
use crate::{DeviceTensor, Gpu, GpuError, launch_u32, ple};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use engram::Hash;
use engram::hash::History;
use gguf::quant::dequant_row;
use gguf::{GgmlType, Split, TensorInfo};
use model::arch::Arch;
use model::placement::Plan;
use std::ops::Range;
use std::sync::Arc;

const WHAT: &str = "qwen4exp Body38";

/// The coverage items the program opens a file with: the tokenizer's
/// pre-tokenizer and the template's tool-call parser are the chat surface's,
/// which the program does not use. The coverage check lists them until
/// proposal P1 lands the program's rows, and every other item it lists is
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

    /// A new sequence's history.
    fn restart(&mut self) -> Result<(), GpuError> {
        self.hist = self
            .hash
            .new_history()
            .map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        Ok(())
    }
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
        let (kinds, _, _, _) = plan38::read(&inputs.spec)?;
        GpuModel::load_placed(
            file,
            plan,
            card,
            host,
            |stream, _, layers, w| Body38::derive(stream, &kinds, layers, w),
            |gpu, file, w, residency| {
                Body38::load_placed(gpu, file, w, plan, inputs, card, host, residency)
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

    /// The body of card `card` (module doc).
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's card, file, weights, plan and inputs, and the host tier's residency and levers (rust-quality R8)"
    )]
    fn load_placed(
        gpu: &Gpu,
        file: Split,
        w: &Weights,
        plan: &Plan<'_>,
        inputs: &model::arch::qwen35moe::place::PlanInputs,
        card: usize,
        host: HostCfg,
        residency: HostResidency,
    ) -> Result<Body38, GpuError> {
        let (spec, hp) = (&inputs.spec, &inputs.hp);
        let n = spec.layers.len();
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
        let ctx = usize::try_from(plan.ctx_max)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| GpuError::shape(WHAT, format!("ctx_max {}", plan.ctx_max)))?;
        let (kinds, router_dims, base, ple_layer) = plan38::read(spec)?;
        let plans = plan38::plans(w, &kinds, ple_layer)?;
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
        for (l, (st, k)) in stores.iter().zip(&kinds).enumerate() {
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
        let boundary = Boundary::with_rows(
            gpu.context(),
            stream,
            BoundaryShape {
                hidden: geo::HIDDEN,
                n_used: geo::N_USED,
            },
            1,
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

    /// Arm (or disarm) the per-layer taps: after each layer the step copies
    /// its streams into the layer's tap, which [`Body38::taps`] reads back,
    /// and after each router its logits and ids ([`Body38::route_taps`]).
    /// For an eager step: a capture taken while they are armed records the
    /// copies. Load-time allocation.
    pub fn set_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
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
            .ok_or(GpuError::state(WHAT, "armed taps (Body38::set_taps)"))?;
        let mut out = Vec::with_capacity(taps.out.len() * geo::STREAMS * geo::HIDDEN);
        for t in &taps.out {
            out.extend(t.to_host_vec(gpu.stream())?);
        }
        Ok(out)
    }

    /// Every layer's router after the last step: its [`geo::EXPERTS`]
    /// logits and its [`geo::N_USED`] routed ids, in slot order. Blocking;
    /// refused when the taps are not armed.
    pub fn route_taps(&self, gpu: &Gpu) -> Result<Vec<(Vec<f32>, Vec<u32>)>, GpuError> {
        let taps = self
            .taps
            .as_ref()
            .ok_or(GpuError::state(WHAT, "armed taps (Body38::set_taps)"))?;
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
                Ok((lg, id))
            })
            .collect()
    }

    /// Every layer's store and the PLE ring, read back: what a run leaves
    /// behind, for a gate to compare bit for bit. Blocking.
    pub fn stores_host(&self, gpu: &Gpu) -> Result<(Vec<Store38Host>, Vec<f32>), GpuError> {
        let stream = gpu.stream();
        let stores = self
            .stores
            .iter()
            .map(|s| {
                Ok(match s {
                    Store38::Rec(r) => Store38Host::Rec {
                        state: r.state.to_host_vec(stream)?,
                        ring: r.ring.to_host_vec(stream)?,
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

    /// The step's position `pos` is the next one and inside the stores, else
    /// refused by name.
    fn check_next(&self, pos: u32, n: usize) -> Result<(), GpuError> {
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
        let io = self.rp.io(m)?;
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
    /// Feed `tokens` from where the model stands by `path` and return the
    /// greedy next token after the last one: captured steps, one a position,
    /// or eager passes of up to eight positions, the last one ending in its
    /// last row's head. Either leaves every position's state and logits bit
    /// for bit. The layer taps must be off for passes.
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
        let io = self.sp.io();
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
            },
            head,
        };
        let mut leg = StepLeg::new(gpu.stream(), hybrid);
        prog.walk(&mut leg)
    }

    /// Every delta store and the PLE ring back to zero, a new PLE history,
    /// the record's lane word to [`LANE`], after the host tier's reset. The
    /// K/V planes and the raw and pooled keys need nothing: nothing reads a
    /// row at or past a live count, and every row below it is written by its
    /// own step first. Synchronizes.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        self.hybrid.reset(stream)?;
        let most = self
            .stores
            .iter()
            .map(|s| match s {
                Store38::Rec(r) => r.state.len().max(r.ring.len()),
                Store38::Qsa { .. } => 0,
            })
            .max()
            .unwrap_or(0);
        let zeros = vec![0.0f32; most];
        for s in &mut self.stores {
            if let Store38::Rec(r) = s {
                r.clear(stream, &zeros)?;
            }
        }
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

const _: () = assert!(GDN.n_v == geo::V_HEADS);
