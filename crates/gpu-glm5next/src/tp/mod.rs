//! A tensor-parallel load of GLM-5.3-Flash: N ranks, each a card holding its part of every layer,
//! which together run the one-card step ([`Rank`], [`step`]). N is the load's count, one and up;
//! nothing here names a rank count, a pair or a node.
//!
//! Each rank holds every layer whole in its streams. The attention, KDA and latent alike, runs
//! the rank's part of the heads; the dense blocks, the routed experts and the shared expert run
//! the rank's part of their inner width; the head projects the rank's part of the vocabulary.
//! The router, the hyper-connection glue, the latent cache and its index, the norms, the
//! low-rank inputs and the embedding run whole on every rank. The cut is the family's rule
//! (`model::arch::glm5next::tp::cut_of`, [`RankOf`]), applied to the tensors of the source a
//! rank loads from ([`load`], `model::part::Source`): a GGUF file or a checkpoint folder.
//!
//! A step is the one-card step's launches at the rank's widths, the mixers' and blocks' own
//! functions ([`crate::kda`], [`crate::mla`], [`crate::ffn`]), with every expert on the card at
//! its own id ([`crate::ffn::rank_moe`]) and no host tier. Each sub-layer's output is then a
//! part of the sum over heads or over the inner width: the ranks gather their parts and each
//! adds them in rank order (`bloomery_gpu::host::Gathers`), so every rank holds the same bits,
//! and `hc_post` runs on the sum. Two gathers a layer. Each rank's head picks its part's argmax,
//! and the step's token is their merge ([`head::merge`]). At N = 1 there is no gather and the
//! launches are the one-card step's with every expert on the card.
//!
//! The ranks' launches go out a sub-layer at a time, every rank's before the next sub-layer's
//! ([`step`]), so no rank's stream holds more than one sub-layer's work queued behind a wait.
//! The NextN layer and a prompt batch are not run here.

mod head;
mod source;

use std::sync::Arc;

use bloomery_gpu::head::Head;
use bloomery_gpu::host::{GatherLoop, GatherSet, GatherShape, Gathers};
use bloomery_gpu::hybrid::SlotMap;
use bloomery_gpu::weights::{Weights, block_index};
use bloomery_gpu::{Branch, DeviceTensor, Gpu, GpuError, MappedHost, WindowMut, mem_batch};
use bloomery_gpu_deepseek41::hc::HC_STREAMS;
use bloomery_gpu_deepseek41::router::glm5next::{N_EXPERT, N_USED};
use cuda_core::{CudaStream, DeviceBuffer, sys};
use gguf::Split;
use model::arch::glm5next::names::Sub;
use model::arch::glm5next::place::{KdaLanes, PlanInputs};
use model::part::{Source, Take};
use runtime::layer::{FfnKind, MixerKind};

use crate::body::pair::write_row;
use crate::body::{
    Body, Dims, Embedding, Kernels, LayerCfg, Parts, Scratch, Store, check_ctx, check_pos, dims_of,
    host_run, layer_cfg, store, trunk_plans,
};
use crate::ffn::{CardExperts, block_names};
use crate::program::{hc_in, hc_out};
use crate::tensors::LayerNames;
use crate::{ffn, kda, mla};

pub use source::RankOf;

/// What the errors name.
const WHAT: &str = "glm5next tp rank";

/// A rank's side of the load's gathers: its page's layout, the page, each span set's post and
/// wait operations on the page, and the gathers it has run, which alternate the span sets.
struct RankLink {
    layout: Gathers,
    page: Arc<MappedHost>,
    ops: [Vec<sys::CUstreamBatchMemOpParams>; Gathers::SETS],
    gathered: usize,
}

/// One rank of a tensor-parallel load (module doc): its parts of the weights, the one-card
/// step's buffers and stores at its widths, its card experts and its head. Its stores take each
/// position once, from 0 up, and a step that fails once its launches go out leaves them holding
/// part of it: the rank takes no further step, and is opened again. Its drop waits for
/// every launch it enqueued before any field frees memory those launches name ([`Drop`]); field
/// order is then drop order, the buffers before the page the gathers read.
pub struct Rank {
    at: RankOf,
    /// The card's stream every launch of the rank goes on.
    stream: Arc<CudaStream>,
    w: Weights,
    k: Kernels,
    branch: Branch,
    dims: Dims,
    cfg: Vec<LayerCfg>,
    names: Vec<LayerNames>,
    s: Scratch,
    stores: Vec<Store>,
    slots: DeviceTensor<u32>,
    card: CardExperts,
    embd: Embedding,
    head: Head,
    ctx: usize,
    /// The position the next step takes, every one before it in the stores; `None` from a step's
    /// first launch until the step completes ([`step`]).
    next_pos: Option<u32>,
    /// The stream buffer the next sub-layer reads.
    cur: usize,
    /// Each layer's streams and each routed layer's picks after the last step, when armed.
    taps: Option<Vec<DeviceBuffer<f32>>>,
    routes: Option<Vec<Option<RouteTap>>>,
    link: Option<RankLink>,
}

/// A routed layer's tap: the router's chosen ids and its scores after the last step.
struct RouteTap {
    ids: DeviceBuffer<u32>,
    probs: DeviceBuffer<f32>,
}

/// A routed layer's picks at one step: the chosen ids in slot order and every expert's sigmoid
/// score, which the selection bias is added to for the ranking.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutePick {
    pub ids: Vec<u32>,
    pub probs: Vec<f32>,
}

impl Rank {
    /// Rank `at` of the trunk of `file` at `ctx` positions on `gpu`, with every routed expert
    /// on the card and widths and tensor parts derived from the file. With more than one
    /// rank the rank makes its gather page in `gpu`'s context ([`Rank::link`]), which the caller
    /// joins to the other ranks' before a step. A context of no positions or past the positions a
    /// reference set checks the selector at, a width the ranks do not divide and a part the
    /// source refuses are refused by name.
    pub fn open(gpu: &Gpu, file: Arc<Split>, at: RankOf, ctx: usize) -> Result<Rank, GpuError> {
        check_ctx(ctx)?;
        let inputs = PlanInputs::read(&file).map_err(|e| GpuError::plan(WHAT, e))?;
        let spec = &inputs.spec;
        let n = at.ranks();
        let whole = dims_of(&inputs)?;
        let vocab = at.part("the vocabulary", inputs.hp.n_vocab)?;
        let embd = Embedding::new(file.clone(), whole.embd, inputs.hp.n_vocab)?;
        let mut dims = whole;
        dims.heads = at.part("heads", whole.heads)?;
        dims.kda.n_k = at.part("KDA key heads", whole.kda.n_k)?;
        dims.kda.n_v = at.part("KDA value heads", whole.kda.n_v)?;
        let ff = at.part("expert width", inputs.hp.expert_ff)?;
        let run = host_run(&spec.layers, None)?;
        let map = SlotMap::prefix(run.clone(), N_EXPERT, N_EXPERT)?;
        let plans = trunk_plans(&file, &inputs)?;
        let mut cfg = Vec::with_capacity(spec.layers.len());
        for (l, (s, &plan)) in spec.layers.iter().zip(&plans).enumerate() {
            let mut c = layer_cfg(l, s, &map, plan)?;
            c.ff = at.part("a layer's block width", c.ff)?;
            cfg.push(c);
        }
        let names = cfg
            .iter()
            .enumerate()
            .map(|(l, c)| LayerNames::of(l, c.kind))
            .collect::<Result<Vec<_>, _>>()?;
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let mut w = load(gpu, &*file, at, spec.layers.len(), inputs.hp.n_head)?;
        let kinds: Vec<_> = cfg.iter().map(|c| c.kind).collect();
        Body::derive(stream, &kinds, 0..kinds.len(), &mut w)?;
        for (c, n) in cfg.iter().zip(&names) {
            c.block
                .check_resident(&w, WHAT, block_names(n), (c.ff, dims.embd))?;
        }
        let k_blocks = cfg.iter().any(|c| !c.block.is_q8());
        let widest = cfg.iter().map(|c| c.ff).max().unwrap_or(0);
        let blocks: Vec<_> = cfg.iter().map(|c| (c.block, c.ff)).collect();
        let s = Scratch::new(stream, &dims, (widest, ctx), &blocks)?;
        let stores = cfg
            .iter()
            .map(|c| store(stream, c.kind, &dims, ctx, KdaLanes::One))
            .collect::<Result<Vec<_>, _>>()?;
        let slots = DeviceTensor::upload(stream, &map.stage_view(), run.len(), N_EXPERT)?;
        let card = CardExperts::rank(gpu, &w, &spec.layers, dims.embd, ff)?;
        let head = Head::new(gpu, &w, dims.rms_eps)?;
        if head.n_vocab() != vocab {
            return Err(shape(format!(
                "the head projects {} rows a rank; the vocabulary is {} over {n} ranks",
                head.n_vocab(),
                inputs.hp.n_vocab
            )));
        }
        let link = (n > 1)
            .then(|| -> Result<RankLink, GpuError> {
                let shape = GatherShape {
                    rows: 1,
                    width: dims.embd,
                };
                let layout = Gathers::new(0, n, at.rank(), &[shape])?;
                let page = MappedHost::new(gpu.context(), layout.page_bytes(), "tp gather page")?;
                let ops = std::array::from_fn(|set| {
                    let at = GatherSet {
                        dev: page.dev_at(0),
                        slot: 0,
                        set,
                    };
                    let mut ops = layout.post(at).to_vec();
                    ops.extend(layout.wait(at));
                    ops
                });
                Ok(RankLink {
                    layout,
                    page: Arc::new(page),
                    ops,
                    gathered: 0,
                })
            })
            .transpose()?;
        Ok(Rank {
            at,
            stream: gpu.stream_handle(),
            w,
            k: Kernels::load(gpu, k_blocks)?,
            branch: Branch::new(gpu.stream_handle())?,
            dims,
            cfg,
            names,
            s,
            stores,
            slots,
            card,
            embd,
            head,
            ctx,
            next_pos: Some(0),
            cur: 0,
            taps: None,
            routes: None,
            link,
        })
    }

    /// The rank's gather layout and page, for the link that joins it to the other ranks' (a
    /// loopback's, `bloomery_gpu::host::GatherLoop`); `None` at one rank.
    #[must_use]
    pub fn link(&self) -> Option<(&Gathers, &Arc<MappedHost>)> {
        self.link.as_ref().map(|l| (&l.layout, &l.page))
    }

    /// Keep every layer's streams and every routed layer's picks after each step, or stop. Taps
    /// are armed before the first step ([`arm_at`]): arming later is refused by name, so the
    /// readers never hand back buffers no step wrote.
    pub fn set_taps(&mut self, gpu: &Gpu, on: bool) -> Result<(), GpuError> {
        streams_at(self.at.rank(), &self.stream, gpu.stream(), [])?;
        if on {
            arm_at(self.next_pos)?;
        }
        let stream = on_card(gpu)?;
        (self.taps, self.routes) = if on {
            let n = HC_STREAMS * self.dims.embd;
            let taps = self
                .cfg
                .iter()
                .map(|_| DeviceBuffer::zeroed(stream, n))
                .collect::<Result<Vec<_>, _>>()?;
            let routes = self
                .cfg
                .iter()
                .map(|c| {
                    (c.kind.ffn == FfnKind::Moe)
                        .then(|| {
                            Ok::<_, GpuError>(RouteTap {
                                ids: DeviceBuffer::zeroed(stream, N_USED)?,
                                probs: DeviceBuffer::zeroed(stream, N_EXPERT)?,
                            })
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?;
            (Some(taps), Some(routes))
        } else {
            (None, None)
        };
        Ok(())
    }

    /// Every layer's streams after the last step, `4 · n_embd` a layer, as the one-card body's
    /// taps lay them; refused when the taps are not armed or no step completed last
    /// ([`Rank::stepped`]). Blocking.
    pub fn taps(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        streams_at(self.at.rank(), &self.stream, gpu.stream(), [])?;
        self.stepped()?;
        let taps = self.taps.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "armed taps (Rank::set_taps)",
        })?;
        let stream = on_card(gpu)?;
        let mut out = Vec::with_capacity(taps.len() * HC_STREAMS * self.dims.embd);
        for t in taps {
            out.extend(t.to_host_vec(stream)?);
        }
        Ok(out)
    }

    /// Each routed layer's picks after the last step, `None` for a dense layer; refused when
    /// the taps are not armed or no step completed last ([`Rank::stepped`]). Blocking.
    pub fn routes(&self, gpu: &Gpu) -> Result<Vec<Option<RoutePick>>, GpuError> {
        streams_at(self.at.rank(), &self.stream, gpu.stream(), [])?;
        self.stepped()?;
        let routes = self.routes.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "armed taps (Rank::set_taps)",
        })?;
        let stream = on_card(gpu)?;
        let mut out = Vec::with_capacity(routes.len());
        for r in routes {
            out.push(match r {
                Some(t) => Some(RoutePick {
                    ids: t.ids.to_host_vec(stream)?,
                    probs: t.probs.to_host_vec(stream)?,
                }),
                None => None,
            });
        }
        Ok(out)
    }

    /// The last step's logits over the rank's part of the vocabulary; refused when no step
    /// completed last ([`Rank::stepped`]). Blocking.
    pub fn logits(&self, gpu: &Gpu) -> Result<Vec<f32>, GpuError> {
        streams_at(self.at.rank(), &self.stream, gpu.stream(), [])?;
        self.stepped()?;
        on_card(gpu)?;
        self.head.logits_to_host(gpu)
    }

    /// Refused by name unless the last step on the rank completed: none ran yet, or one failed
    /// once its launches went out.
    fn stepped(&self) -> Result<(), GpuError> {
        match self.next_pos {
            Some(p) if p > 0 => Ok(()),
            _ => Err(GpuError::State {
                what: WHAT,
                missing: "a completed step (tp::step)",
            }),
        }
    }

    /// The checked token's embedding at `pos` into the step's row.
    fn begin(&mut self, gpu: &Gpu, pos: u32) -> Result<(), GpuError> {
        write_row(on_card(gpu)?, &mut self.s.rows[0], &self.embd.streams, pos)?;
        self.cur = 0;
        Ok(())
    }

    /// Sub-layer `sub` of layer `l`: its input, the mixer or the block at the rank's widths, the
    /// ranks' parts gathered and summed, `hc_post`, and after the block the layer's taps.
    fn enqueue_sub(&mut self, gpu: &Gpu, l: usize, sub: Sub) -> Result<(), GpuError> {
        let Rank {
            w,
            k,
            branch,
            dims,
            cfg,
            names,
            s,
            stores,
            slots,
            card,
            taps,
            routes,
            link,
            cur,
            at,
            ..
        } = self;
        on_card(gpu)?;
        let kind = cfg
            .get(l)
            .ok_or_else(|| shape(format!("layer {l} past the trunk's {}", cfg.len())))?
            .kind;
        let mut p = Parts::of(
            k,
            branch,
            dims,
            cfg,
            names,
            s,
            stores,
            slots,
            card,
            taps.as_deref_mut(),
        );
        hc_in(gpu, w, &mut p, *cur, l, sub)?;
        match (sub, kind.mixer, kind.ffn) {
            (Sub::Attn, MixerKind::DeltaRule, _) => kda::kda(gpu, w, &mut p, l)?,
            (Sub::Attn, MixerKind::Latent, _) => mla::mla(gpu, w, &mut p, l)?,
            (Sub::Ffn, _, FfnKind::Dense) => ffn::dense(gpu, w, &mut p, l)?,
            (Sub::Ffn, _, FfnKind::Moe) => ffn::rank_moe(gpu, w, &mut p, l)?,
            (Sub::Attn, other, _) => {
                return Err(shape(format!(
                    "layer {l}: a {other:?} mixer, which glm5next has none of"
                )));
            }
        }
        if let Some(link) = link {
            gather(gpu, link, at.rank(), &mut p.s.out)?;
        }
        *cur = hc_out(gpu, &mut p, *cur)?;
        if sub == Sub::Ffn {
            p.tap(gpu, l, *cur)?;
            if let Some(Some(t)) = routes.as_mut().and_then(|r| r.get_mut(l)) {
                t.ids.copy_from_device_async(&p.s.rout.ids, gpu.stream())?;
                t.probs
                    .copy_from_device_async(&p.s.rout.probs, gpu.stream())?;
            }
        }
        Ok(())
    }

    /// The streams' mean into the head, and the head over the rank's part of the vocabulary.
    fn enqueue_head(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        let stream = on_card(gpu)?;
        let streams = &self.s.rows[0].streams[self.cur];
        self.k
            .hc
            .enqueue_mean(stream, streams, self.dims.embd, 0, self.head.input_mut())?;
        self.head.enqueue(gpu, &self.w)
    }

    /// The rank's pick after the last step: its part's largest logit and that logit's id in the
    /// whole vocabulary. Blocking.
    fn pick(&self, gpu: &Gpu) -> Result<(f32, u32), GpuError> {
        on_card(gpu)?;
        let id = self.head.token(gpu)?;
        let logits = self.head.logits_to_host(gpu)?;
        let v = *logits.get(id as usize).ok_or_else(|| {
            shape(format!(
                "the head's pick {id} past its {} rows",
                logits.len()
            ))
        })?;
        let off = self
            .at
            .rank()
            .checked_mul(self.head.n_vocab())
            .and_then(|o| u32::try_from(o).ok())
            .and_then(|o| o.checked_add(id))
            .ok_or_else(|| shape(format!("rank {}'s pick {id} past u32", self.at.rank())))?;
        Ok((v, off))
    }
}

/// One step of `token` at `pos` on every rank of a load, each with its card, rank `r` at
/// `ranks[r]`: the ranks' launches a sub-layer at a time (module doc), every head, then the
/// merge of the ranks' picks, the step's token. `link` joins these ranks' pages and stays live
/// for the step; `None` at one rank. Each rank uses its opener's distinct stream. Blocking.
/// Refused by name before any launch: ranks out of rank order or of another
/// count than their own, ranks of unlike trunks ([`alike`]), a position past a rank's stores or
/// other than the one its stores stand at ([`at_pos`]), and a token past the vocabulary. A step
/// that fails once its launches go out leaves every rank refusing the next (`Rank`).
pub fn step(
    ranks: &mut [(&Gpu, &mut Rank)],
    link: Option<&GatherLoop>,
    token: u32,
    pos: u32,
) -> Result<u32, GpuError> {
    let n = ranks.len();
    let layers = alike(
        ranks
            .iter()
            .map(|(_, r)| (r.cfg.len(), r.dims.embd, r.head.n_vocab())),
    )?;
    if (n > 1) != link.is_some() {
        return Err(shape(format!(
            "a step over {n} ranks needs its joined link only at N > 1"
        )));
    }
    for (i, (gpu, r)) in ranks.iter().enumerate() {
        if r.at.rank() != i || r.at.ranks() != n {
            return Err(shape(format!(
                "rank {} of {} at place {i} of a step over {n}",
                r.at.rank(),
                r.at.ranks()
            )));
        }
        streams_at(
            i,
            &r.stream,
            gpu.stream(),
            ranks[..i].iter().map(|(_, p)| &p.stream),
        )?;
        at_pos(r.next_pos, r.ctx, pos)?;
        if let Some(link) = link {
            let (layout, page) = r
                .link()
                .ok_or_else(|| shape(format!("rank {i}: no gather page")))?;
            link.check(layout, page, 1)?;
        }
    }
    for (_, r) in ranks.iter_mut() {
        r.embd.fill(token)?;
    }
    for (_, r) in ranks.iter_mut() {
        r.next_pos = None;
    }
    for (gpu, r) in ranks.iter_mut() {
        r.begin(gpu, pos)?;
    }
    for l in 0..layers {
        for sub in [Sub::Attn, Sub::Ffn] {
            for (gpu, r) in ranks.iter_mut() {
                r.enqueue_sub(gpu, l, sub)?;
            }
        }
    }
    for (gpu, r) in ranks.iter_mut() {
        r.enqueue_head(gpu)?;
    }
    let picks = ranks
        .iter()
        .map(|(gpu, r)| r.pick(gpu))
        .collect::<Result<Vec<_>, _>>()?;
    let token = head::merge(&picks)?;
    for (_, r) in ranks.iter_mut() {
        r.next_pos = Some(pos + 1);
    }
    Ok(token)
}

/// Each rank uses its opener's stream, distinct from every earlier rank's.
fn streams_at<'a, T: 'a>(
    rank: usize,
    opened: &Arc<T>,
    offered: &T,
    prior: impl IntoIterator<Item = &'a Arc<T>>,
) -> Result<(), GpuError> {
    if !std::ptr::eq(offered, opened.as_ref()) {
        return Err(shape(format!("rank {rank}: use the GPU that opened it")));
    }
    if prior.into_iter().any(|p| Arc::ptr_eq(p, opened)) {
        return Err(shape(format!(
            "rank {rank}: each rank must use a distinct stream"
        )));
    }
    Ok(())
}

/// The layer count of ranks whose trunks are `(layers, embedding width, head rows)` each, which
/// one step joins: refused by name for no rank, or a rank whose trunk is unlike rank 0's.
fn alike(trunks: impl IntoIterator<Item = (usize, usize, usize)>) -> Result<usize, GpuError> {
    let mut trunks = trunks.into_iter();
    let first = trunks
        .next()
        .ok_or_else(|| shape("a step over no ranks".into()))?;
    for (r, t) in trunks.enumerate() {
        if t != first {
            return Err(shape(format!(
                "rank {}'s (layers, width, head rows) {t:?} unlike rank 0's {first:?}",
                r + 1
            )));
        }
    }
    Ok(first.0)
}

/// Refused by name unless stores of `ctx` positions that stand at `next` (a rank's
/// `next_pos`) take a step at `pos`: inside them ([`check_pos`]), and the position they stand at.
fn at_pos(next: Option<u32>, ctx: usize, pos: u32) -> Result<(), GpuError> {
    check_pos(pos, ctx)?;
    match next {
        Some(p) if p == pos => Ok(()),
        Some(p) => Err(shape(format!(
            "position {pos}, where the stores hold {p} positions"
        ))),
        None => Err(shape(format!(
            "position {pos}: a step failed once its launches went out, so the stores hold part \
             of it; open the rank again"
        ))),
    }
}

/// Refused by name unless stores that stand at `next` (a rank's `next_pos`) have taken no step,
/// the one state where armed taps are written before they are read.
fn arm_at(next: Option<u32>) -> Result<(), GpuError> {
    match next {
        Some(0) => Ok(()),
        _ => Err(GpuError::State {
            what: WHAT,
            missing: "taps armed before the first step (open the rank again)",
        }),
    }
}

/// The rank's part `out` of a sub-layer's output posted to the other ranks, their parts waited
/// for, and the parts' rank-order sum into `out`, on the link's next span set.
fn gather(
    gpu: &Gpu,
    link: &mut RankLink,
    rank: usize,
    out: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let set = link.gathered % Gathers::SETS;
    link.gathered += 1;
    let dev = link.page.dev_at(0);
    let at = GatherSet { dev, slot: 0, set };
    let own = dev + link.layout.part(0, set, rank).start as u64;
    // SAFETY: the part's values lie in the rank's own page, mapped in this context, which the
    // rank holds until its drop has waited for every launch it enqueued.
    let mut part = unsafe { WindowMut::<f32>::from_raw(own, out.len(), gpu.context()) };
    part.copy_from_device_async(out, stream)?;
    // SAFETY: every counter the operations name lies in the rank's page, held as above.
    unsafe { mem_batch(stream, &mut link.ops[set], "glm5next tp gather")? };
    // SAFETY: `dev` is the rank's page of this layout, mapped in this context and held as above.
    unsafe { link.layout.enqueue_sum(gpu.elem(), stream, at, 1, out) }
}

impl Drop for Rank {
    /// Wait for every launch the rank enqueued before its fields free the buffers and the page
    /// those launches name (`MappedHost`'s owner contract): a step that failed part way leaves
    /// launches queued. A rank still waiting on a part that will not come runs on once its link is
    /// released (`GatherLoop::release`). An error is kept on the card's context.
    fn drop(&mut self) {
        let ctx = self.stream.context();
        ctx.record_err(ctx.bind_to_thread());
        ctx.record_err(self.stream.synchronize());
    }
}

/// Rank `at`'s parts of the trunk's `layers` layers of `heads` heads and of the head, from
/// `source` on `gpu` ([`bloomery_gpu::parts::load`]): every tensor the card holds
/// (`model::arch::glm5next::tp::card_holds`: not the token embedding, which the host reads, nor
/// the layers past the trunk's), each the part the family's cut gives the rank
/// ([`RankOf::take`]), in its type's card format.
pub fn load(
    gpu: &Gpu,
    source: &(impl Source + Sync),
    at: RankOf,
    layers: usize,
    heads: usize,
) -> Result<Weights, GpuError> {
    bloomery_gpu::parts::load(
        gpu.stream(),
        source,
        |name, ty, dims| {
            if !model::arch::glm5next::tp::card_holds(name, block_index(name)?, layers) {
                return Ok(Take::Skip);
            }
            at.take(name, ty, dims, heads as u64)
        },
        |_| None,
        false,
    )
}

/// `gpu`'s context current on this thread, which runs every rank of a loopback in turn, and its
/// stream.
fn on_card(gpu: &Gpu) -> Result<&CudaStream, GpuError> {
    gpu.context().bind_to_thread()?;
    Ok(gpu.stream())
}

fn shape(detail: String) -> GpuError {
    GpuError::shape(WHAT, detail)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    #[test]
    fn ranks_use_their_openers_distinct_streams() {
        let a = Arc::new(0u8);
        let b = Arc::new(0u8);
        assert!(super::streams_at(0, &a, &a, []).is_ok());
        assert!(super::streams_at(1, &b, &b, [&a]).is_ok());
        for (opened, offered, prior, want) in [
            (&a, &b, &b, "use the GPU that opened it"),
            (&a, &a, &a, "each rank must use a distinct stream"),
        ] {
            let e = super::streams_at(1, opened, offered, [prior]).unwrap_err();
            assert!(e.to_string().contains(want), "{e}");
        }
    }
    use super::{alike, arm_at, at_pos, check_ctx};
    use model::arch::glm5next::place::ORACLE_POSITIONS;

    /// A load of no positions or past the reference sets' positions is refused.
    #[test]
    fn a_context_of_no_positions_or_past_the_sets_is_refused() {
        let max = usize::try_from(ORACLE_POSITIONS).unwrap();
        assert!(check_ctx(1).is_ok() && check_ctx(max).is_ok());
        assert!(check_ctx(0).is_err() && check_ctx(max + 1).is_err());
    }

    /// Stores take the position they stand at, inside them; any other position, the one past
    /// them and every position after a failed step are refused.
    #[test]
    fn a_step_takes_only_the_position_its_stores_stand_at() {
        assert!(at_pos(Some(0), 4, 0).is_ok() && at_pos(Some(3), 4, 3).is_ok());
        for (next, pos) in [
            (Some(0), 1),
            (Some(2), 1),
            (Some(4), 4),
            (None, 0),
            (None, 2),
        ] {
            assert!(at_pos(next, 4, pos).is_err(), "{next:?} at {pos}");
        }
    }

    /// A step joins ranks of one trunk, at N = 1, 2 and 4; no rank, and a rank of another layer
    /// count, width or head part, are refused.
    #[test]
    fn a_step_joins_only_ranks_of_one_trunk() {
        let t = (3, 64, 8);
        for n in [1, 2, 4] {
            assert_eq!(alike(vec![t; n]).unwrap(), 3, "{n} ranks");
        }
        let mut accepted = Vec::new();
        if alike([]).is_ok() {
            accepted.push(("no ranks", 0, t));
        }
        for other in [(2, 64, 8), (4, 64, 8), (3, 32, 8), (3, 64, 16)] {
            for n in [2, 4] {
                let mut trunks = vec![t; n];
                trunks[n - 1] = other;
                let refused = alike(trunks)
                    .is_err_and(|e| e.to_string().contains(&format!("rank {}'s", n - 1)));
                if !refused {
                    accepted.push(("last rank unlike", n, other));
                }
            }
        }
        assert!(accepted.is_empty(), "not refused by name: {accepted:?}");
    }

    /// Taps arm only on stores that have taken no step; after a step or a failed one, arming is
    /// refused by name.
    #[test]
    fn taps_arm_only_before_the_first_step() {
        assert!(arm_at(Some(0)).is_ok());
        for next in [Some(1), Some(3), None] {
            let e = arm_at(next).unwrap_err().to_string();
            assert!(
                e.contains("taps armed before the first step"),
                "{next:?}: {e}"
            );
        }
    }
}
