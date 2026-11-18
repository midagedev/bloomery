//! The whole-card load of a Hy4-flash file and the state one decode step reads and writes: the
//! layers' sites, the stores by layer kind, the step's buffers and words, the host side of the
//! embedding and the engram rows, and the gate's taps.
//!
//! The stores, by layer kind:
//! - a sparse (DSA) layer keeps a 576-wide f16 row `[k_rope | latent]` for every position in a
//!   room ([`bloomery_gpu::RoomTensor`]: the pages the sequence reaches are backed, on the card or,
//!   by the plan's `BLOOMERY_KV_HOST` word, on the host; [`crate::rooms`]), the k-pool selector's
//!   index rows (`[value | gate]`, 256 f16 a position) in a ring of
//!   [`crate::batch::INDEX_RING`] positions (the most a prompt batch's launch appends, and a pool),
//!   its pooled keys (128 f16 a pool of four), its scores and its list of picked rows;
//! - a window (SWA) layer keeps a ring of `W` 576-wide rows, position `p` at slot `p % W`.

use std::ops::Range;
use std::sync::Arc;

use bloomery_gpu::arch::hybrid::{Acts, GqaCx, Site};
use bloomery_gpu::card_leg::{CardLeg, LegBufs, LegShape, LegStacks};
use bloomery_gpu::engram_delta::EngramDeltaKernels;
use bloomery_gpu::flash::{FlashKernels, partials_ms_len, partials_v_len, window_partials_lens};
use bloomery_gpu::flash_gqa::FlashGqaKernels;
use bloomery_gpu::fused::FusedKernels;
use bloomery_gpu::gemm::Gemm32Kernels;
use bloomery_gpu::hc_ident::{HC, HcIdentKernels, MIX};
use bloomery_gpu::head::Head;
use bloomery_gpu::host::run::HostRun;
use bloomery_gpu::host::served::{ResidencyParts, TierBody};
use bloomery_gpu::host::stream_side::StreamSide;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::swap_source::{ResidencySpec, hosted_stacks};
use bloomery_gpu::hybrid::{Chain, Hybrid, SlotMap};
use bloomery_gpu::kpool::KpoolKernels;
use bloomery_gpu::latent::{INDEX_HEAD, INDEX_ROW, LatentKernels, POOL};
use bloomery_gpu::linear::out_gate::OutGateKernels;
use bloomery_gpu::model::{
    ChainBody, HostServed, Resident, Rollback, Slots, StepKernels, refuse_reserve_past, stage_run,
};
use bloomery_gpu::qsa::QsaKernels;
use bloomery_gpu::rope_table::{Direction, RopeRows, RopeSpec, RopeTable};
use bloomery_gpu::route_body::sigmoid_512_16;
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu::site::{KGemvKernels, SiteTy};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel, KvBackings, RoomTensor};
use bloomery_levers::HostCfg;
use cuda_core::{CudaStream, DeviceBuffer};
use engram::hash::{Hash, History};
use gguf::{GgmlType, Split};
use model::arch::Arch;
use model::arch::hy4::hparams::Hparams;
use model::arch::hy4::names;
use model::arch::hy4::place::{CACHE_ROWS, PlanInputs};
use model::arch::place_hybrid;
use model::placement::Plan;
use model::placement::kernels::DownForms;
use runtime::layer::hosted;
use runtime::stores::PosBuf;

use crate::batch::{PassShape, PlacedPass};
use crate::placed::{LOAD_ROWS, Open, Placed};

/// What the refusals of this file name.
pub(crate) const WHAT: &str = "hy4::Body";

/// The router instance: the file's 512 experts and its top 16 (the fixture keeps both).
pub(crate) type Router = sigmoid_512_16::RouterKernels;

/// The router's output for one token.
pub(crate) type RouteOut = <Router as RouterEntry>::Out;

/// The rotated values of a query head and of a cache row (`rope.dimension_count`).
pub(crate) const ROPE: usize = 64;
/// The latent width of a cache row (`attention.kv_lora_rank`), the flash's value width.
pub(crate) const LATENT: usize = 512;
/// A cache row: the rotated part, then the latent.
pub(crate) const ROW: usize = ROPE + LATENT;
/// Values in one engram table row (`embedding_length_per_layer_input`): one byte each.
pub(crate) const ENGRAM_W: usize = 96;
/// The engram path's norm epsilon: the reference's RMSNorm with eps unset (fp32's epsilon).
pub(crate) const ENGRAM_EPS: f32 = bloomery_gpu::engram_delta::EPS;

/// The stages of a layer the taps copy, in their slot order; each slot holds the four streams'
/// width, a narrower stage its first `hidden` values. The names are the reference graph's
/// `cb()` names (`<stage>-<layer>`); `engram_out` is written on an engram site only.
pub const TAP_STAGES: [&str; 7] = [
    "engram_out",
    "attn_norm",
    "attn_out",
    "hc_attn_out",
    "ffn_norm",
    "ffn_out",
    "l_out",
];

/// The dense sites' file types the step's projections take.
const DENSE: &[SiteTy] = &[
    SiteTy::Q8_0,
    SiteTy::Q6K,
    SiteTy::Q5K,
    SiteTy::Q4K,
    SiteTy::Bf16,
];
/// The indexer's head weights: F32 in the file.
const F32: &[SiteTy] = &[SiteTy::F32];

/// The model's widths, from the file's header.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Shape {
    pub(crate) hidden: usize,
    pub(crate) eps: f32,
    pub(crate) q_lora: usize,
    /// A query head's values before the absorption: nope, then rope.
    pub(crate) head_k: usize,
    pub(crate) nope: usize,
    pub(crate) head_v: usize,
    /// The widest layer's heads (the window layers').
    pub(crate) max_heads: usize,
    /// The window layers' ring rows (`attention.sliding_window`).
    pub(crate) window: usize,
    /// A sparse layer's list width (`attention.indexer.top_k`) and the pools it keeps beside its
    /// own (`top_k / kpool − 1`).
    pub(crate) top_k: usize,
    pub(crate) kept: usize,
    pub(crate) idx_heads: usize,
    pub(crate) hc_eps: f32,
    pub(crate) hc_mag: f32,
    pub(crate) latent: usize,
    pub(crate) expert_ff: usize,
    pub(crate) weights_scale: f32,
    pub(crate) ctx: usize,
}

impl Shape {
    /// `hp`'s widths at `ctx` positions; a width the composed kernels are not built for is refused
    /// by name (each is an architecture fact of the released file).
    pub(crate) fn of(hp: &Hparams, ctx: usize) -> Result<Shape, GpuError> {
        let m = &hp.mla;
        let fixed = [
            ("rope.dimension_count", m.pe, ROPE),
            ("attention.kv_lora_rank", m.kv_lora, LATENT),
            ("attention.indexer.key_length", hp.indexer.d, INDEX_HEAD),
            ("attention.indexer.kpool", hp.indexer.kpool, POOL),
            ("hc.streams", hp.hc.streams, HC),
        ];
        for (key, have, want) in fixed {
            if have != want {
                return Err(GpuError::shape(
                    WHAT,
                    format!("{key} is {have}; this body is built for {want}"),
                ));
            }
        }
        let top_k = hp.indexer.top_k;
        if m.head_k <= m.pe || top_k < 2 * POOL || !top_k.is_multiple_of(POOL) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a query head of {} values over {} rotated, a top-k of {top_k}",
                    m.head_k, m.pe
                ),
            ));
        }
        if hp.experts.n_expert != Router::N_EXPERT || hp.experts.n_used != Router::N_USED {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} experts, the top {}; the router is built for {} and {}",
                    hp.experts.n_expert,
                    hp.experts.n_used,
                    Router::N_EXPERT,
                    Router::N_USED
                ),
            ));
        }
        Ok(Shape {
            hidden: hp.n_embd,
            eps: hp.rms_eps,
            q_lora: m.q_lora,
            head_k: m.head_k,
            nope: m.head_k - m.pe,
            head_v: m.head_v,
            max_heads: hp.heads.iter().copied().max().unwrap_or(0),
            window: hp.window,
            top_k,
            kept: top_k / POOL - 1,
            idx_heads: hp.indexer.heads,
            hc_eps: hp.hc.eps,
            hc_mag: hp.hc.magnitude,
            latent: hp.latent,
            expert_ff: hp.experts.ff,
            weights_scale: hp.experts.weights_scale,
            ctx,
        })
    }

    /// The flash's scale: `1/√(nope + rope)`.
    pub(crate) fn kq_scale(&self) -> f32 {
        1.0 / (self.head_k as f32).sqrt()
    }
}

/// A sparse layer's indexer sites.
pub(crate) struct IdxSites {
    /// `indexer.attn_q_b`: the query off the q residual, `heads × 128`.
    pub(crate) q_b: Site,
    /// `indexer.attn_k`: the raw token key off the normed row.
    pub(crate) k: Site,
    /// `indexer.proj`: the heads' weights off the normed row (F32).
    pub(crate) proj: Site,
}

/// A SiLU-gated block's three sites and its width.
pub(crate) struct Glu {
    pub(crate) gate: Site,
    pub(crate) up: Site,
    pub(crate) down: Site,
    pub(crate) ff: usize,
}

impl Glu {
    fn sites(&self) -> [&Site; 3] {
        [&self.gate, &self.up, &self.down]
    }
}

/// A layer's feed-forward block.
pub(crate) enum Ffn {
    /// The dense lead's SiLU block.
    Dense(Glu),
    /// The routed block: the latent pair, the stacks' names, the shared SiLU expert.
    Routed {
        down_lat: Site,
        up_lat: Site,
        stacks: [String; 3],
        shared: Glu,
        /// The routed experts' gate-first clamp (`swiglu_clamp_exp`).
        clamp: f32,
    },
}

/// One layer's sites and facts.
pub(crate) struct Layer {
    pub(crate) dsa: bool,
    pub(crate) heads: usize,
    pub(crate) q_a: Site,
    pub(crate) q_b: Site,
    pub(crate) kv_a: Site,
    pub(crate) gate: Site,
    pub(crate) wo: Site,
    pub(crate) idx: Option<IdxSites>,
    pub(crate) ffn: Ffn,
    /// The engram site's index among the sites and its `engram_kv` site, on a site layer.
    pub(crate) engram: Option<(usize, Site)>,
}

/// A layer's stores.
pub(crate) enum Store {
    Dsa(Box<Sparse>),
    Swa { ring: DeviceTensor<u16> },
}

/// A sparse layer's stores: the cache row of every position (a room), the selector's index ring,
/// pool plane and scores, and the list it hands the flash.
pub(crate) struct Sparse {
    pub(crate) cache: RoomTensor<u16>,
    pub(crate) index: DeviceTensor<u16>,
    pub(crate) pooled: DeviceTensor<u16>,
    pub(crate) scores: DeviceBuffer<f32>,
    pub(crate) list: DeviceBuffer<u32>,
    pub(crate) n_sel: DeviceBuffer<u32>,
}

/// The kernels the step launches beside the `Gpu`'s own.
pub(crate) struct Kernels {
    pub(crate) kgemv: KGemvKernels,
    pub(crate) g32: Gemm32Kernels,
    pub(crate) flash_gqa: FlashGqaKernels,
    pub(crate) step: StepKernels,
    pub(crate) fused: FusedKernels,
    pub(crate) flash: FlashKernels,
    pub(crate) latent: LatentKernels,
    pub(crate) kpool: KpoolKernels,
    pub(crate) qsa: QsaKernels,
    pub(crate) out_gate: OutGateKernels,
    pub(crate) hc: HcIdentKernels,
    pub(crate) engram: EngramDeltaKernels,
    pub(crate) router: Router,
    pub(crate) leg: CardLeg,
}

/// The step's device buffers, one row each.
pub(crate) struct Scratch {
    /// The four streams, ping-ponged: each write goes to the other buffer.
    pub(crate) hs: [DeviceBuffer<f32>; 2],
    pub(crate) mix: DeviceBuffer<f32>,
    pub(crate) y: DeviceBuffer<f32>,
    pub(crate) xn: DeviceBuffer<f32>,
    pub(crate) qa: DeviceBuffer<f32>,
    pub(crate) qr: DeviceBuffer<f32>,
    pub(crate) q: DeviceBuffer<f32>,
    pub(crate) q_rope: DeviceBuffer<f32>,
    pub(crate) q_cur: DeviceBuffer<f32>,
    pub(crate) kv_a: DeviceBuffer<f32>,
    pub(crate) kv_s: DeviceBuffer<f32>,
    pub(crate) kvr: DeviceBuffer<f32>,
    pub(crate) iq: DeviceBuffer<f32>,
    pub(crate) ik: DeviceBuffer<f32>,
    pub(crate) wi: DeviceBuffer<f32>,
    pub(crate) part_v: DeviceBuffer<f32>,
    pub(crate) part_ms: DeviceBuffer<f32>,
    pub(crate) att: DeviceBuffer<f32>,
    pub(crate) av: DeviceBuffer<f32>,
    pub(crate) g: DeviceBuffer<f32>,
    pub(crate) gated: DeviceBuffer<f32>,
    pub(crate) o: DeviceBuffer<f32>,
    pub(crate) out: DeviceBuffer<f32>,
    pub(crate) fg: DeviceBuffer<f32>,
    pub(crate) fu: DeviceBuffer<f32>,
    pub(crate) fa: DeviceBuffer<f32>,
    pub(crate) lat: DeviceBuffer<f32>,
    pub(crate) routed: DeviceBuffer<f32>,
    pub(crate) lat_up: DeviceBuffer<f32>,
    pub(crate) shared: DeviceBuffer<f32>,
    pub(crate) f: DeviceBuffer<f32>,
    pub(crate) leg: LegBufs,
    /// The gather tables of the rotated query parts: head `h`'s value `j` from `q_rope`'s
    /// `h·head_k + nope + j` to `q_cur`'s `h·ROW + j`.
    pub(crate) rope_src: DeviceBuffer<u32>,
    pub(crate) rope_dst: DeviceBuffer<u32>,
    /// The engram path's buffers (one site at a time) and the key norm's unit gain.
    pub(crate) e_norm: DeviceBuffer<f32>,
    pub(crate) e_kv: DeviceBuffer<f32>,
    pub(crate) e_kn: DeviceBuffer<f32>,
    pub(crate) e_gate: DeviceBuffer<f32>,
    pub(crate) e_ones: DeviceBuffer<f32>,
}

/// The step's words and rows the host writes before every eager enqueue or replay
/// ([`ChainBody::refresh`]).
pub(crate) struct Words {
    /// The position, its window slot, the live key count.
    pub(crate) pos: DeviceBuffer<u32>,
    pub(crate) ring_pos: DeviceBuffer<u32>,
    pub(crate) n_keys: DeviceBuffer<u32>,
    /// The position's cos/sin rows of the sparse and the window rope.
    pub(crate) cs_dsa: DeviceBuffer<f32>,
    pub(crate) cs_swa: DeviceBuffer<f32>,
    /// Each engram site's gathered rows (`cols × 96`) and its mask (`cols`).
    pub(crate) e_rows: Vec<DeviceBuffer<f32>>,
    pub(crate) e_valid: Vec<DeviceBuffer<f32>>,
}

/// The host side of a step's inputs: the file (the embedding and the engram table are read from
/// it), the rope tables and the engram hash with its history.
pub(crate) struct HostIn {
    pub(crate) file: Arc<Split>,
    /// The embedding's type and bytes a row.
    pub(crate) embd: (GgmlType, usize),
    pub(crate) rope: [RopeTable; 2],
    pub(crate) hash: Option<(Hash, History)>,
    /// The engram sites' columns.
    pub(crate) cols: usize,
    streams: Vec<f32>,
    cs: Vec<f32>,
    rows: Vec<u32>,
    vals: Vec<f32>,
    valid: Vec<f32>,
}

/// The gate's copies of one step's stages ([`TAP_STAGES`]), every layer, and of the head's fold.
pub(crate) struct Taps {
    pub(crate) stages: Vec<DeviceBuffer<f32>>,
    pub(crate) head: DeviceBuffer<f32>,
}

/// One step's taps read back ([`Body::taps`]).
pub struct TapRows {
    /// Layer-major, [`TAP_STAGES`] a layer, each `4·hidden` values (a narrower stage's first
    /// `hidden` values hold it).
    pub stages: Vec<Vec<f32>>,
    /// Each layer's picked experts and their weights, in the router's slot order; empty on a
    /// dense layer.
    pub picks: Vec<(Vec<u32>, Vec<f32>)>,
    /// The head's fold of the streams (`hc_head`).
    pub head: Vec<f32>,
}

/// The Hy4-flash body: the whole model on one card.
pub struct Body {
    pub(crate) sh: Shape,
    pub(crate) layers: Vec<Layer>,
    pub(crate) k: Kernels,
    pub(crate) acts: Acts,
    pub(crate) s: Scratch,
    /// Each routed layer's router output (`None` on a dense layer), so a step's picks stay
    /// readable.
    pub(crate) routes: Vec<Option<RouteOut>>,
    pub(crate) words: Words,
    pub(crate) stores: Vec<Store>,
    /// Each layer's by-position buffers as the plan declares them ([`Hparams::pos_bufs`]), which a
    /// sequence's rooms are made from, and what the load backs their pages by.
    pub(crate) decl: Vec<Vec<PosBuf>>,
    pub(crate) kv: KvBackings,
    pub(crate) host: HostIn,
    /// Every position's rope row at the sparse layers' base: the selector's table.
    pub(crate) rope_rows: RopeRows,
    pub(crate) taps: Option<Taps>,
    /// The position the stores stand at: the next step's.
    pub(crate) held: u32,
    /// The host tier's side of a placed load ([`crate::placed`]); `None` on the whole-card load.
    pub(crate) placed: Option<Placed>,
    /// The placed prompt pass's buffers ([`crate::batch::PlacedPass`]), on a placed load opened with
    /// its pass; `None` on a load that steps its prompt and on the whole-card load.
    pub(crate) pass: Option<Box<PlacedPass>>,
    /// The prompt calls' host streaming and its ring pass's buffers (`BLOOMERY_XSTREAM`), off until
    /// set ([`bloomery_gpu::host::xstream::XStreamBody`]).
    pub(crate) xs: StreamSide,
}

/// What one resident sequence owns ([`Slots`]): every layer's stores, the engram history and the
/// position the stores stand at.
pub struct Seq {
    stores: Vec<Store>,
    hist: Option<History>,
    held: u32,
}

/// A step's host input.
pub struct DecodeInput {
    pub token: u32,
    pub pos: u32,
}

fn z(stream: &CudaStream, n: usize) -> Result<DeviceBuffer<f32>, GpuError> {
    Ok(DeviceBuffer::zeroed(stream, n.max(1))?)
}

impl Glu {
    fn load(
        fw: (&Split, &Weights),
        names: [String; 3],
        n: usize,
        ff: usize,
    ) -> Result<Glu, GpuError> {
        let [gate, up, down] = names;
        Ok(Glu {
            gate: Site::of_file(fw, WHAT, gate, (ff, n), DENSE)?,
            up: Site::of_file(fw, WHAT, up, (ff, n), DENSE)?,
            down: Site::of_file(fw, WHAT, down, (n, ff), DENSE)?,
            ff,
        })
    }
}

impl Layer {
    fn load(file: &Split, w: &Weights, hp: &Hparams, l: usize) -> Result<Layer, GpuError> {
        let fw = (file, w);
        let site = |name: String, rows: usize, k: usize, allowed: &[SiteTy]| {
            Site::of_file(fw, WHAT, name, (rows, k), allowed)
        };
        let n = hp.n_embd;
        let heads = hp.heads(l);
        let m = &hp.mla;
        let dsa = hp.dsa(l);
        let idx = if dsa {
            let ix = &hp.indexer;
            Some(IdxSites {
                q_b: site(names::indexer_attn_q_b(l), ix.heads * ix.d, m.q_lora, DENSE)?,
                k: site(names::indexer_attn_k(l), ix.d, n, DENSE)?,
                proj: site(names::indexer_proj(l), ix.heads, n, F32)?,
            })
        } else {
            None
        };
        let ffn = if hp.dense(l) {
            let names3 = [names::ffn_gate(l), names::ffn_up(l), names::ffn_down(l)];
            Ffn::Dense(Glu::load(fw, names3, n, hp.dense_ff)?)
        } else {
            let shexp = [
                names::ffn_gate_shexp(l),
                names::ffn_up_shexp(l),
                names::ffn_down_shexp(l),
            ];
            Ffn::Routed {
                down_lat: site(names::ffn_routed_down(l), hp.latent, n, DENSE)?,
                up_lat: site(names::ffn_routed_up(l), n, hp.latent, DENSE)?,
                stacks: [
                    names::ffn_gate_exps(l),
                    names::ffn_up_exps(l),
                    names::ffn_down_exps(l),
                ],
                shared: Glu::load(fw, shexp, n, hp.experts.shared_width)?,
                clamp: hp.clamp[l],
            }
        };
        let engram = match &hp.engram {
            Some(e) if e.site(l) => {
                let at = e.layers.iter().position(|&s| s == l).ok_or_else(|| {
                    GpuError::shape(WHAT, format!("layer {l} is a site the sites do not list"))
                })?;
                let k = e.n_cols() * e.key_length;
                Some((at, site(names::engram_kv(l), (HC + 1) * n, k, DENSE)?))
            }
            _ => None,
        };
        Ok(Layer {
            dsa,
            heads,
            q_a: site(names::attn_q_a(l), m.q_lora, n, DENSE)?,
            q_b: site(names::attn_q_b(l), heads * m.head_k, m.q_lora, DENSE)?,
            kv_a: site(names::attn_kv_a_mqa(l), m.kv_lora + m.pe, n, DENSE)?,
            gate: site(names::attn_gate(l), heads * m.head_v, n, DENSE)?,
            wo: site(names::attn_output(l), n, heads * m.head_v, DENSE)?,
            idx,
            ffn,
            engram,
        })
    }

    pub(crate) fn sites(&self) -> Vec<&Site> {
        let mut v = vec![&self.q_a, &self.q_b, &self.kv_a, &self.gate, &self.wo];
        if let Some(ix) = &self.idx {
            v.extend([&ix.q_b, &ix.k, &ix.proj]);
        }
        match &self.ffn {
            Ffn::Dense(g) => v.extend(g.sites()),
            Ffn::Routed {
                down_lat,
                up_lat,
                shared,
                ..
            } => {
                v.extend([down_lat, up_lat]);
                v.extend(shared.sites());
            }
        }
        if let Some((_, s)) = &self.engram {
            v.push(s);
        }
        v
    }
}

impl Store {
    /// A layer's store of `sh.ctx` positions: a sparse layer's cache is the room `decl` declares,
    /// backed as `kv` says.
    fn new(
        stream: &Arc<CudaStream>,
        sh: &Shape,
        (decl, kv): (&[PosBuf], KvBackings),
        dsa: bool,
    ) -> Result<Store, GpuError> {
        Ok(if dsa {
            let pools = sh.ctx.div_ceil(POOL);
            Store::Dsa(Box::new(Sparse {
                cache: kv.room(stream, decl, CACHE_ROWS, sh.ctx)?,
                index: DeviceTensor::zeroed(stream, crate::batch::INDEX_RING, INDEX_ROW)?,
                pooled: DeviceTensor::zeroed(stream, pools, INDEX_HEAD)?,
                scores: z(stream, pools)?,
                list: DeviceBuffer::zeroed(stream, sh.top_k)?,
                n_sel: DeviceBuffer::zeroed(stream, 1)?,
            }))
        } else {
            Store::Swa {
                ring: DeviceTensor::zeroed(stream, sh.window, ROW)?,
            }
        })
    }

    /// The store's address range in bytes: a room's whole range, backed or not.
    pub(crate) fn bytes(&self) -> usize {
        match self {
            Store::Dsa(d) => d.cache.buf().num_bytes() + self.fixed_bytes(),
            Store::Swa { .. } => self.fixed_bytes(),
        }
    }

    /// The bytes the store holds whatever its sequence reaches: every buffer but the cache room.
    pub(crate) fn fixed_bytes(&self) -> usize {
        match self {
            Store::Dsa(d) => {
                let Sparse {
                    cache: _,
                    index,
                    pooled,
                    scores,
                    list,
                    n_sel,
                } = &**d;
                index.buf().num_bytes()
                    + pooled.buf().num_bytes()
                    + scores.num_bytes()
                    + list.num_bytes()
                    + n_sel.num_bytes()
            }
            Store::Swa { ring } => ring.buf().num_bytes(),
        }
    }
}

impl Kernels {
    fn load(gpu: &Gpu) -> Result<Kernels, GpuError> {
        let ctx = gpu.context();
        Ok(Kernels {
            kgemv: KGemvKernels::load(gpu)?,
            g32: Gemm32Kernels::load(ctx)?,
            flash_gqa: FlashGqaKernels::load(ctx)?,
            step: StepKernels::load(ctx)?,
            fused: FusedKernels::load(ctx)?,
            flash: FlashKernels::load(ctx)?,
            latent: LatentKernels::load(ctx)?,
            kpool: KpoolKernels::load(ctx)?,
            qsa: QsaKernels::load(ctx)?,
            out_gate: OutGateKernels::load(ctx)?,
            hc: HcIdentKernels::load(ctx)?,
            engram: EngramDeltaKernels::load(ctx)?,
            router: Router::load(ctx)?,
            leg: CardLeg::load(gpu)?,
        })
    }
}

impl Scratch {
    fn new(
        stream: &CudaStream,
        sh: &Shape,
        hp: &Hparams,
        leg: LegBufs,
    ) -> Result<Scratch, GpuError> {
        let n = sh.hidden;
        let h = sh.max_heads;
        let ff = hp.dense_ff.max(hp.experts.shared_width);
        // The flash's partials: the wider of a sparse layer's list and a window ring, at the
        // widest head count.
        let (wv, wms) = window_partials_lens(h, sh.window);
        let pv = partials_v_len(h, sh.top_k).max(wv);
        let pms = partials_ms_len(h, sh.top_k).max(wms);
        let index = |v: usize| {
            u32::try_from(v).map_err(|_| GpuError::shape(WHAT, "a query row index past u32"))
        };
        let mut src = Vec::with_capacity(h * ROPE);
        let mut dst = Vec::with_capacity(h * ROPE);
        for hd in 0..h {
            for j in 0..ROPE {
                src.push(index(hd * sh.head_k + sh.nope + j)?);
                dst.push(index(hd * ROW + j)?);
            }
        }
        let cols = hp.engram.as_ref().map_or(1, |e| e.n_cols());
        Ok(Scratch {
            hs: [z(stream, HC * n)?, z(stream, HC * n)?],
            mix: z(stream, MIX)?,
            y: z(stream, n)?,
            xn: z(stream, n)?,
            qa: z(stream, sh.q_lora)?,
            qr: z(stream, sh.q_lora)?,
            q: z(stream, h * sh.head_k)?,
            q_rope: z(stream, h * sh.head_k)?,
            q_cur: z(stream, h * ROW)?,
            kv_a: z(stream, ROW)?,
            kv_s: z(stream, ROW)?,
            kvr: z(stream, ROW)?,
            iq: z(stream, sh.idx_heads * INDEX_HEAD)?,
            ik: z(stream, INDEX_HEAD)?,
            wi: z(stream, sh.idx_heads)?,
            part_v: z(stream, pv)?,
            part_ms: z(stream, pms)?,
            att: z(stream, h * LATENT)?,
            av: z(stream, h * sh.head_v)?,
            g: z(stream, h * sh.head_v)?,
            gated: z(stream, h * sh.head_v)?,
            o: z(stream, n)?,
            out: z(stream, n)?,
            fg: z(stream, ff)?,
            fu: z(stream, ff)?,
            fa: z(stream, ff)?,
            lat: z(stream, sh.latent)?,
            routed: z(stream, sh.latent)?,
            lat_up: z(stream, n)?,
            shared: z(stream, n)?,
            f: z(stream, n)?,
            leg,
            rope_src: DeviceBuffer::from_host(stream, &src)?,
            rope_dst: DeviceBuffer::from_host(stream, &dst)?,
            e_norm: z(stream, cols * ENGRAM_W)?,
            e_kv: z(stream, (HC + 1) * n)?,
            e_kn: z(stream, HC * n)?,
            e_gate: z(stream, HC)?,
            e_ones: DeviceBuffer::from_host(stream, &vec![1.0f32; HC * n])?,
        })
    }
}

impl Body {
    /// The whole model of `file` resident on `gpu` with stores of `ctx` positions, plus the
    /// output head. The embedding and the engram table stay in the file: the step reads a token's
    /// rows on the host. A file whose card tensors exceed the card's free bytes is refused by name
    /// before any is read: the released file does not fit one card. The stores' rooms back their
    /// pages on the card ([`Body::open_rooms`] with [`KvBackings::CARD`]).
    pub fn open(gpu: Gpu, file: Split, ctx: usize) -> Result<GpuModel<Body>, GpuError> {
        Body::open_rooms(gpu, file, ctx, KvBackings::CARD)
    }

    /// [`Body::open`] with the sparse layers' cache rooms backed as `kv` says: the pages a
    /// sequence reaches are backed as it grows ([`Body::fit_through`] backs the rest at once).
    pub fn open_rooms(
        gpu: Gpu,
        file: Split,
        ctx: usize,
        kv: KvBackings,
    ) -> Result<GpuModel<Body>, GpuError> {
        const OPEN: &str = "hy4::Body::open";
        let hp = Hparams::read(&file).map_err(|e| GpuError::plan(OPEN, e))?;
        if ctx == 0 || ctx > hp.n_ctx_train {
            return Err(GpuError::shape(
                OPEN,
                format!(
                    "a context of {ctx} positions; the file trains {}",
                    hp.n_ctx_train
                ),
            ));
        }
        let embd = names::token_embd();
        let host_side =
            |name: &str| name == names::ENGRAM_TABLE || name == embd || nextn(&hp, name);
        let (free, total) = gpu.mem_info()?;
        let bytes: u64 = file
            .iter_tensors()
            .filter(|(_, t)| !host_side(&t.name))
            .map(|(_, t)| t.nbytes)
            .sum();
        if usize::try_from(bytes).map_or(true, |b| b > free) {
            return Err(GpuError::shape(
                OPEN,
                format!(
                    "the file's card tensors are {bytes} B and the card has {free} of {total} B \
                     free: this body holds the whole model on one card"
                ),
            ));
        }
        let weights = Weights::load_where(gpu.stream(), &file, |name| !host_side(name))?;
        let body = Body::load(&gpu, Arc::new(file), &weights, (&hp, ctx), kv)?;
        let head = Head::of(&gpu, &weights, body.head_eps(), body.head_norm())?;
        Ok(GpuModel::new(Resident {
            gpu,
            weights,
            body,
            head: Some(head),
            ctx_max: ctx,
        }))
    }
}

impl Body {
    /// The placed load of `file` by `plan` on stage card `card`: every layer's dense part, the
    /// stores and each routed layer's card share of its stacks on the card, the rest of the
    /// routed experts on the host tier ([`crate::placed`]), the embedding and the engram table
    /// read from the file on the host. The load runs no residency machine ([`Residency::Off`];
    /// [`Body::open_placed_with`] takes a word). Refused by name: a plan of another layer range or of more than one
    /// stage card ([`stage_run`]), an expert tier card, a routed run with a gap. The sparse layers'
    /// cache rooms back their pages as the plan's `kv_host` word says, a spill keeping the stage
    /// card's margin free ([`KvBackings::of_plan`]).
    pub fn open_placed(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
    ) -> Result<GpuModel<Body>, GpuError> {
        Body::open_placed_with(file, (plan, inputs), card, host, (None, Residency::Off))
    }

    /// [`Body::open_placed`] with the placed prompt pass of one unit of `rows` rows
    /// ([`crate::batch`]'s `PlacedPass`): its buffers made beside the placed side and the host
    /// tier's batch sets and union slabs for its rows, so a prompt runs as passes of up to `rows`
    /// rows ([`crate::batch::prompt_placed`]). The plan's prompt reserve holds the pass's bytes
    /// ([`crate::batch::pass_bytes`] at the plan's context). Refused by name as
    /// [`Body::open_placed`] refuses, and the pass's buffers past the plan's reserve for them. The
    /// load runs no residency machine.
    pub fn open_placed_pass(
        file: Split,
        plan: &Plan<'_>,
        inputs: &PlanInputs,
        card: usize,
        host: HostCfg,
        rows: usize,
    ) -> Result<GpuModel<Body>, GpuError> {
        Body::open_placed_with(
            file,
            (plan, inputs),
            card,
            host,
            (Some(PassShape::of(rows)), Residency::Off),
        )
    }

    /// [`Body::open_placed`], or with the placed pass of `pass` ([`Body::open_placed_pass`]; a walk
    /// of several units holds one batch set a unit, [`PlacedHost::prepare_pass_walk`]), under
    /// `residency`: a `mid` word runs the common machine over the stage card's routed
    /// stacks, held as rooms, and the slot map's card copy the step and the pass read
    /// ([`bloomery_gpu::host::placed`]), its churn pool hosted beside the plan's host set. Refused
    /// by name as those refuse, and a machine beside a routed layer the card holds whole
    /// ([`PlacedHost::new`]). The plan's ring term, when it holds one
    /// ([`model::arch::hy4::place::stream_ring`]), caps the ring a `split` stream makes.
    ///
    /// [`PlacedHost::new`]: bloomery_gpu::host::placed::PlacedHost::new
    /// [`PlacedHost::prepare_pass_walk`]: bloomery_gpu::host::placed::PlacedHost::prepare_pass_walk
    pub fn open_placed_with(
        file: Split,
        (plan, inputs): (&Plan<'_>, &PlanInputs),
        card: usize,
        host: HostCfg,
        (pass, residency): (Option<PassShape>, Residency),
    ) -> Result<GpuModel<Body>, GpuError> {
        const PLAN: &str = "hy4::Body::open_placed";
        let hp = &inputs.hp;
        let (_, ctx) = stage_run(PLAN, plan, card, hp.n_layer)?;
        let run = hosted(&inputs.spec.layers)
            .map_err(|e| GpuError::shape(PLAN, format!("the routed layers: {e}")))?;
        let spec = ResidencySpec::hosted(
            residency,
            hp.experts.n_used,
            LOAD_ROWS,
            Arc::new(hosted_stacks(&inputs.model, PLAN, (run.len(), true))?),
        );
        GpuModel::load_placed_with(
            file,
            plan,
            card,
            host,
            spec,
            |_, _, _, _| Ok(()),
            |gpu, file, w, set, glue| {
                let kv = KvBackings::of_plan(plan, card, gpu.opened_free());
                let mut body = Body::load(gpu, Arc::clone(file), w, (hp, ctx), kv)?;
                let map = SlotMap::of_plan(plan, card, None, run.clone(), hp.experts.n_expert)?;
                let open = Open {
                    file,
                    plan,
                    hp,
                    host,
                    set,
                    glue,
                };
                let mut placed = crate::placed::load((gpu, w), &body.layers, map, open)?;
                match pass {
                    Some(shape) => {
                        // The batch sets and union slabs for the pass's rows, which also serve
                        // the step's one column.
                        placed.prepare_pass_walk(gpu, (shape.rows, shape.units))?;
                        let made = PlacedPass::new(gpu, w, &body, &placed.side, shape)?;
                        let reserve = inputs
                            .prompt_batch_bytes(plan.machine)
                            .map_err(|e| GpuError::plan(PLAN, e))?;
                        refuse_reserve_past(
                            PLAN,
                            "placed pass buffers",
                            reserve,
                            made.bytes() as u64,
                            "the prompt batch's rows, the host's sums, the layer's map, the \
                             slots' places and the card slots' route tables",
                        )?;
                        body.pass = Some(Box::new(made));
                        if let Some(cap) = place_hybrid::planned_room_cap(plan, inputs) {
                            body.xs.set_ring_room(cap);
                        }
                    }
                    // The step's union slabs for one column: the host tier's resident path runs
                    // only through them.
                    None => placed.prepare_step_union()?,
                }
                placed.start_residency(gpu)?;
                body.placed = Some(placed);
                Ok(body)
            },
        )
    }
}

/// Whether `name` is a tensor of a next-token block (`blk.{l}.*`, `l` from `n_layer`), which this
/// body does not run.
fn nextn(hp: &Hparams, name: &str) -> bool {
    (hp.n_layer..hp.n_block).any(|l| name.starts_with(&format!("blk.{l}.")))
}

impl Body {
    pub(crate) fn load(
        gpu: &Gpu,
        file: Arc<Split>,
        w: &Weights,
        (hp, ctx): (&Hparams, usize),
        kv: KvBackings,
    ) -> Result<Body, GpuError> {
        let stream = gpu.stream();
        let handle = gpu.stream_handle();
        let sh = Shape::of(hp, ctx)?;
        let layers = (0..hp.n_layer)
            .map(|l| Layer::load(&file, w, hp, l))
            .collect::<Result<Vec<_>, _>>()?;
        let acts = Acts::new(stream, layers.iter().flat_map(Layer::sites))?;
        // The leg's buffers hold the form each routed layer's down stack reads.
        let mut downs = DownForms::NONE;
        for layer in &layers {
            if let Ffn::Routed { stacks, .. } = &layer.ffn {
                // A placed load's layer whose experts the host serves whole holds no stack here.
                if w.get(&stacks[2]).is_none() {
                    continue;
                }
                // The common leg's arms (`LegStacks::of`): the downs are Q4_K and Q6_K in the
                // file's blocks, both read through the 128-value activation form these buffers hold.
                let st = LegStacks::of(w, stacks.each_ref().map(String::as_str))?;
                let act = st.down_act().ok_or_else(|| {
                    GpuError::tensor(WHAT, stacks[2].clone(), "a down stack the card leg reads")
                })?;
                downs = downs.with(act);
            }
        }
        let shape = LegShape {
            cap: 1,
            slots_per_col: Router::N_USED,
            n: sh.latent,
            ff: sh.expert_ff,
        };
        let leg = LegBufs::for_downs(stream, shape, downs)?;
        let s = Scratch::new(stream, &sh, hp, leg)?;
        let routes = layers
            .iter()
            .map(|layer| match layer.ffn {
                Ffn::Routed { .. } => Router::new_out(stream).map(Some),
                Ffn::Dense(_) => Ok(None),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let decl: Vec<Vec<PosBuf>> = (0..hp.n_layer).map(|l| hp.pos_bufs(l)).collect();
        let stores = layers
            .iter()
            .zip(&decl)
            .map(|(layer, d)| Store::new(&handle, &sh, (d, kv), layer.dsa))
            .collect::<Result<Vec<_>, _>>()?;
        let rope = [
            RopeTable::new(&RopeSpec::window(hp.rope_base, ROPE))?,
            RopeTable::new(&RopeSpec::window(hp.rope_base_swa, ROPE))?,
        ];
        let rope_rows = RopeRows::new(stream, &rope[0], ROPE, ctx)?;
        let embd = embd_rows(&file, sh.hidden, hp.n_vocab)?;
        let (hash, cols, sites) = match &hp.engram {
            Some(e) => {
                check_table(&file, e.rows)?;
                if e.key_length != ENGRAM_W {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "engram rows of {} values; the table's are {ENGRAM_W}",
                            e.key_length
                        ),
                    ));
                }
                let meta: Vec<_> = file
                    .iter_kv()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect();
                let n_vocab = u32::try_from(hp.n_vocab)
                    .map_err(|_| GpuError::shape(WHAT, "a vocabulary past u32"))?;
                let arch = model::arch::hy4::ARCH;
                let hash = Hash::ple_mapped_from_keys(&meta, arch, n_vocab)
                    .map_err(|e| GpuError::plan(WHAT, e))?;
                let hist = hash
                    .new_mapped_history()
                    .map_err(|e| GpuError::plan(WHAT, e))?;
                let ids: Vec<usize> = hash.layer_ids().iter().map(|&l| l as usize).collect();
                if hash.n_cols() != e.n_cols() || ids != e.layers {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "the hash's sites {ids:?} of {} columns; the header's {:?} of {}",
                            hash.n_cols(),
                            e.layers,
                            e.n_cols()
                        ),
                    ));
                }
                (Some((hash, hist)), e.n_cols(), e.layers.len())
            }
            None => (None, 1, 0),
        };
        let words = Words {
            pos: DeviceBuffer::zeroed(stream, 1)?,
            ring_pos: DeviceBuffer::zeroed(stream, 1)?,
            n_keys: DeviceBuffer::zeroed(stream, 1)?,
            cs_dsa: z(stream, ROPE)?,
            cs_swa: z(stream, ROPE)?,
            e_rows: (0..sites)
                .map(|_| z(stream, cols * ENGRAM_W))
                .collect::<Result<_, _>>()?,
            e_valid: (0..sites)
                .map(|_| z(stream, cols))
                .collect::<Result<_, _>>()?,
        };
        let k = Kernels::load(gpu)?;
        Ok(Body {
            sh,
            layers,
            k,
            acts,
            s,
            routes,
            words,
            stores,
            decl,
            kv,
            host: HostIn {
                file,
                embd,
                rope,
                hash,
                cols,
                streams: vec![0.0; HC * sh.hidden],
                cs: Vec::with_capacity(ROPE),
                rows: vec![0; sites * cols],
                vals: vec![0.0; cols * ENGRAM_W],
                valid: vec![0.0; cols],
            },
            rope_rows,
            taps: None,
            held: 0,
            placed: None,
            pass: None,
            xs: StreamSide::default(),
        })
    }

    /// Arm the taps: every later step copies its stages ([`TAP_STAGES`]) and the head's fold. The
    /// copies are launches of the chain, so a graph captured while they are armed holds them.
    pub fn arm_taps(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let n = self.sh.hidden;
        self.taps = Some(Taps {
            stages: (0..self.layers.len() * TAP_STAGES.len())
                .map(|_| z(stream, HC * n))
                .collect::<Result<_, _>>()?,
            head: z(stream, n)?,
        });
        Ok(())
    }

    /// Drop the taps: the chain copies nothing.
    pub fn disarm_taps(&mut self) {
        self.taps = None;
    }

    /// The last step's taps and picks read back. Synchronizes `stream`.
    pub fn taps(&self, stream: &CudaStream) -> Result<TapRows, GpuError> {
        let t = self.taps.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "armed taps",
        })?;
        let picks = self
            .routes
            .iter()
            .map(|r| match r {
                Some(out) => Ok((
                    Router::out_ids(out).to_host_vec(stream)?,
                    Router::out_weights(out).to_host_vec(stream)?,
                )),
                None => Ok((Vec::new(), Vec::new())),
            })
            .collect::<Result<_, GpuError>>()?;
        Ok(TapRows {
            stages: t
                .stages
                .iter()
                .map(|b| b.to_host_vec(stream))
                .collect::<Result<_, _>>()?,
            picks,
            head: t.head.to_host_vec(stream)?,
        })
    }

    /// The model's width.
    #[must_use]
    pub fn hidden(&self) -> usize {
        self.sh.hidden
    }

    /// The stores' bytes: every layer's caches, rings and selector state.
    #[must_use]
    pub fn store_bytes(&self) -> usize {
        self.stores.iter().map(Store::bytes).sum()
    }

    /// Each layer's kind in layer order: whether it is sparse (DSA) and whether it carries an
    /// engram site.
    #[must_use]
    pub fn kinds(&self) -> Vec<(bool, bool)> {
        self.layers
            .iter()
            .map(|l| (l.dsa, l.engram.is_some()))
            .collect()
    }

    /// Position `pos`'s words (the position, its window slot, the live key count) and the cos/sin
    /// rows of both rope bases.
    pub(crate) fn write_position(&mut self, stream: &CudaStream, pos: u32) -> Result<(), GpuError> {
        let window = u32::try_from(self.sh.window)
            .map_err(|_| GpuError::shape(WHAT, "a window past u32"))?;
        self.words.pos.copy_from_host(stream, &[pos])?;
        self.words
            .ring_pos
            .copy_from_host(stream, &[pos % window])?;
        self.words.n_keys.copy_from_host(stream, &[pos + 1])?;
        let h = &mut self.host;
        for (table, dst) in h
            .rope
            .iter()
            .zip([&mut self.words.cs_dsa, &mut self.words.cs_swa])
        {
            h.cs.clear();
            table.push(pos, Direction::Forward, &mut h.cs);
            dst.copy_from_host(stream, &h.cs)?;
        }
        Ok(())
    }

    /// The host side of step `pos` of `token`: the embedding row into the four streams, the
    /// position's words and cos/sin rows, each engram site's rows and mask.
    fn host_inputs(&mut self, stream: &CudaStream, token: u32, pos: u32) -> Result<(), GpuError> {
        let n = self.sh.hidden;
        let h = &mut self.host;
        // 1. The embedding row, dequantized from the file, repeated into the four streams.
        let (ty, row_bytes) = h.embd;
        let embd = h.file.data(&names::token_embd())?;
        let at = token as usize * row_bytes;
        let src = embd.get(at..at + row_bytes).ok_or_else(|| {
            GpuError::shape(WHAT, format!("token {token} past the embedding's rows"))
        })?;
        gguf::dequant_row(ty, src, &mut h.streams[..n]).map_err(|e| GpuError::plan(WHAT, e))?;
        let (first, rest) = h.streams.split_at_mut(n);
        for c in rest.chunks_mut(n) {
            c.copy_from_slice(first);
        }
        self.s.hs[0].copy_from_host(stream, &h.streams)?;
        // 2. The words and the cos/sin rows of both rope bases.
        self.write_position(stream, pos)?;
        let h = &mut self.host;
        // 3. The engram rows: the hash of the token and its history, each site's rows dequantized
        //    from the table, and the token's mask on every column.
        if let Some((hash, hist)) = h.hash.as_mut() {
            let mut live = [false];
            hash.mapped_rows_into(hist, u64::from(pos), &[Some(token)], &mut h.rows, &mut live)
                .map_err(|e| GpuError::plan(WHAT, e))?;
            let table = h.file.data(names::ENGRAM_TABLE)?;
            h.valid.fill(if live[0] { 1.0 } else { 0.0 });
            let sites = self
                .words
                .e_rows
                .iter_mut()
                .zip(self.words.e_valid.iter_mut());
            for (e, (rows, mask)) in sites.enumerate() {
                for c in 0..h.cols {
                    let r = h.rows[e * h.cols + c] as usize;
                    let src = table.get(r * ENGRAM_W..(r + 1) * ENGRAM_W).ok_or_else(|| {
                        GpuError::shape(WHAT, format!("engram row {r} past the table"))
                    })?;
                    let dst = &mut h.vals[c * ENGRAM_W..(c + 1) * ENGRAM_W];
                    gguf::dequant_row(GgmlType::F8E4M3, src, dst)
                        .map_err(|e| GpuError::plan(WHAT, e))?;
                }
                rows.copy_from_host(stream, &h.vals)?;
                mask.copy_from_host(stream, &h.valid)?;
            }
        }
        Ok(())
    }
}

/// The embedding's type and bytes a row: `[hidden, vocab]`, read on the host; another shape is
/// refused by name.
fn embd_rows(file: &Split, hidden: usize, vocab: usize) -> Result<(GgmlType, usize), GpuError> {
    let name = names::token_embd();
    let Some((_, t)) = file.find(&name) else {
        return Err(GpuError::tensor(WHAT, name, "the token embedding"));
    };
    let dims: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    if dims != [hidden, vocab] || !t.ty.has_dequant() {
        return Err(GpuError::tensor(
            WHAT,
            name,
            "a [hidden, vocab] embedding the host dequantizes",
        ));
    }
    let bytes = usize::try_from(t.nbytes)
        .map_err(|_| GpuError::shape(WHAT, "an embedding past the address space"))?;
    Ok((t.ty, bytes / vocab))
}

/// The engram table as the step reads it: scale-less F8 E4M3 rows of [`ENGRAM_W`] bytes, `rows`
/// of them; another type or shape is refused by name.
fn check_table(file: &Split, rows: u64) -> Result<(), GpuError> {
    let name = names::ENGRAM_TABLE;
    let Some((_, t)) = file.find(name) else {
        return Err(GpuError::tensor(WHAT, name, "the engram table"));
    };
    if t.ty != GgmlType::F8E4M3 || t.dims != [ENGRAM_W as u64, rows] {
        return Err(GpuError::tensor(
            WHAT,
            name,
            "scale-less F8 E4M3 rows of 96 values, as many as the header's engram rows",
        ));
    }
    Ok(())
}

/// The composition context the shared sites take: the gemv kernels and the model's widths.
pub(crate) fn cx<'a>(gpu: &'a Gpu, w: &'a Weights, k: &'a Kernels, sh: &Shape) -> GqaCx<'a> {
    GqaCx {
        what: WHAT,
        gpu,
        w,
        kgemv: &k.kgemv,
        g32: &k.g32,
        flash: &k.flash_gqa,
        mma: false,
        out_gate: Some(&k.out_gate),
        hidden: sh.hidden,
        eps: sh.eps,
        n_kv: 1,
        ctx: sh.ctx,
    }
}

impl ChainBody for Body {
    type Input = DecodeInput;
    type Host = Placed;
    const UNHOSTED_POOL: bloomery_levers::UnhostedPool = bloomery_levers::UnhostedPool::NotCarried;

    fn arch() -> Arch {
        Arch::Hy4
    }

    /// The step at `pos`; a position past the stores is refused by name.
    fn decode_input(&mut self, token: u32, pos: u32) -> Result<DecodeInput, GpuError> {
        if pos as usize >= self.sh.ctx {
            return Err(GpuError::shape(
                WHAT,
                format!("a step at position {pos} in stores of {}", self.sh.ctx),
            ));
        }
        Ok(DecodeInput { token, pos })
    }

    /// The host inputs of the step, after the cache rooms are fitted through its position
    /// ([`Body::fit`]); refused by name off the position the stores stand at, so the engram
    /// history and the stores never part.
    fn refresh(&mut self, stream: &CudaStream, input: &DecodeInput) -> Result<(), GpuError> {
        if input.pos != self.held {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a step at position {}; the stores stand at {} (reset after a call that \
                     failed part-way)",
                    input.pos, self.held
                ),
            ));
        }
        self.fit(input.pos as usize, 1)?;
        self.host_inputs(stream, input.token, input.pos)?;
        self.held = input.pos + 1;
        Ok(())
    }

    fn enqueue_chain(&mut self, gpu: &Gpu, w: &Weights, head: &mut Head) -> Result<(), GpuError> {
        crate::step::enqueue(gpu, w, self, head)
    }

    /// The engram history back to a fresh sequence, the position to 0, the cache rooms' pages
    /// given back and the host tier settled. The other stores need nothing: a window layer's flash
    /// reads only the slots its position reaches, a sparse layer's selector only the pools and
    /// rows before its count, and each row below the count is written by its own step first.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if let Some((hash, hist)) = self.host.hash.as_mut() {
            *hist = hash
                .new_mapped_history()
                .map_err(|e| GpuError::plan(WHAT, e))?;
        }
        self.settle_rooms()?;
        if let Some(p) = self.placed.as_mut() {
            p.reset(gpu.stream())?;
        }
        self.held = 0;
        Ok(())
    }

    /// `attention.layer_norm_rms_epsilon`: every block norm's, the head's included.
    fn head_eps(&self) -> f32 {
        self.sh.eps
    }

    fn resident_bytes(&self) -> usize {
        self.store_bytes()
            + self.acts.bytes()
            + self.s.leg.bytes()
            + self.placed.as_ref().map_or(0, Placed::device_bytes)
            + self.pass_device_bytes()
    }

    fn layers(&self) -> Range<usize> {
        0..self.layers.len()
    }

    fn host(&mut self) -> Option<&mut Placed> {
        self.placed.as_mut()
    }
}

/// Why a tier accessor of a whole-card load panics: the seat reads the host tier only of the
/// placed load it opened.
const WHOLE_CARD: &str =
    "hy4::Body: a whole-card load has no host tier (the seat opens the placed load)";

impl TierBody for Body {
    type Experts = HostRun;

    fn hybrid(&self) -> &Hybrid<HostRun> {
        self.placed.as_ref().expect(WHOLE_CARD).hybrid()
    }

    fn hybrid_mut(&mut self) -> &mut Hybrid<HostRun> {
        self.placed.as_mut().expect(WHOLE_CARD).hybrid_mut()
    }

    fn residency_parts(&mut self) -> Option<ResidencyParts<'_, HostRun>> {
        self.placed.as_mut().and_then(TierBody::residency_parts)
    }

    fn serve_chain(&mut self, chain: Chain) -> Result<(), GpuError> {
        match self.placed.as_mut() {
            Some(placed) => placed.serve_chain(chain),
            None => Err(GpuError::shape(WHAT, WHOLE_CARD)),
        }
    }
}

impl Rollback for Body {
    /// Refused by name, always: a window layer's ring and the engram history hold the position
    /// they stand at alone, so the body takes no position back. The session's cut keeps every
    /// held position or nothing (a reset) and never asks for another.
    fn rollback(&mut self, pos: u32) -> Result<(), GpuError> {
        Err(GpuError::shape(
            WHAT,
            format!(
                "a rollback to position {pos}: the window rings and the engram history take no \
                 position back"
            ),
        ))
    }
}

impl Slots for Body {
    type Seq = Seq;

    /// A sequence of the live one's store shapes at position 0 with a fresh engram history. Load
    /// time only; a card without the stores' bytes that no position's reach decides is refused by
    /// name (the cache rooms back their pages as the sequence grows).
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Seq, GpuError> {
        let need: usize = self.stores.iter().map(Store::fixed_bytes).sum();
        let free = gpu.alloc_room()?;
        if need > free {
            return Err(GpuError::shape(
                WHAT,
                format!("a sequence's stores are {need} B and the card has {free} B free"),
            ));
        }
        let handle = gpu.stream_handle();
        let stores = self
            .layers
            .iter()
            .zip(&self.decl)
            .map(|(layer, d)| Store::new(&handle, &self.sh, (d, self.kv), layer.dsa))
            .collect::<Result<Vec<_>, _>>()?;
        let hist = match &self.host.hash {
            Some((hash, _)) => Some(
                hash.new_mapped_history()
                    .map_err(|e| GpuError::plan(WHAT, e))?,
            ),
            None => None,
        };
        Ok(Seq {
            stores,
            hist,
            held: 0,
        })
    }

    /// The stores, the engram history and the position exchanged: pointer moves only. A parked
    /// sequence of another layout is refused by name.
    fn swap_seq(&mut self, _gpu: &Gpu, seq: &mut Seq) -> Result<(), GpuError> {
        let same = self.stores.len() == seq.stores.len()
            && self
                .stores
                .iter()
                .zip(&seq.stores)
                .all(|(a, b)| a.bytes() == b.bytes())
            && self.host.hash.is_some() == seq.hist.is_some();
        if !same {
            return Err(GpuError::shape(
                WHAT,
                "a parked sequence of another store layout",
            ));
        }
        std::mem::swap(&mut self.stores, &mut seq.stores);
        if let (Some((_, hist)), Some(other)) = (self.host.hash.as_mut(), seq.hist.as_mut()) {
            std::mem::swap(hist, other);
        }
        std::mem::swap(&mut self.held, &mut seq.held);
        Ok(())
    }

    fn seq_bytes(&self) -> usize {
        self.store_bytes()
    }
}
