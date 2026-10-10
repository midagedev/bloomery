//! A GLM sequence state on the host ([`GlmSeq`], over the shared
//! `bloomery_gpu::checkpoint::saved`): what a server's prompt cache or a
//! slot's save holds of a sequence, put back on the slot that resumes it.
//!
//! The parts, each forced by a fact of the architecture:
//! - every latent layer's latent rows, index rows and pool keys of the held
//!   positions — the attention's cache, indexed by position;
//! - every KDA layer's committed state lane and conv ring at the held
//!   position, and at the last checkpoint below it (the last prompt call's
//!   end), so a later request that shares only that prefix cuts there — a
//!   KDA layer keeps no earlier position but a copy;
//! - on a NextN load, the layer's store rows of the held positions (its
//!   latent rows, index rows and pool keys), the positions the store holds,
//!   and the step's and the verify's arena rows ([`DraftRows`]): what a
//!   draft's waiting rows read when it rejoins the sequence.
//!
//! The residency is the model's, not a sequence's: the slot map names what
//! the card holds, and a flip lands both at a boundary, so a state carries
//! no map and a resume runs on the map as it stands. A save and a resume
//! run between calls, outside every residency boundary.
//!
//! A resume onto another model, card or context, onto another store layout
//! (a NextN load and a plain one), or onto a model that is not empty is
//! refused by name, nothing copied.

use std::sync::Arc;

use bloomery_gpu::checkpoint::Checkpoints;
use bloomery_gpu::checkpoint::saved::{
    Identity, Lent, Pending, SaveLane, SeqState, SeqView, Spans, Streams,
};
use bloomery_gpu::checkpoint::staging::NoRoom;
use bloomery_gpu::latent::{INDEX_HEAD, INDEX_ROW, LATENT, POOL, pools_for};
use bloomery_gpu::linear::KHeadMap;
use bloomery_gpu::linear::LinearShape;
use bloomery_gpu::{DeviceTensor, GpuError, GpuModel};
use bloomery_gpu_deepseek41::hc::HC_STREAMS;
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::glm5next::place::PlanInputs;
use models::Mixer;

use super::nextn::{DraftRows, Nextn};
use super::{Body, Store, shape};

/// A GLM sequence state ([`seq_save`]): the shared state of the module
/// doc's rows and stores, and on a NextN load the layer's side beside it.
pub struct GlmSeq {
    state: Arc<SeqState<()>>,
    draft: Option<DraftRows>,
}

impl GlmSeq {
    /// The positions it holds.
    #[must_use]
    pub fn positions(&self) -> u32 {
        self.state.positions()
    }

    /// The host bytes it holds: the shared state's and the NextN side's.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.state.bytes() + self.draft.as_ref().map_or(0, DraftRows::bytes)
    }

    /// The positions of the points it carries, ascending.
    #[must_use]
    pub fn points(&self) -> Vec<u32> {
        self.state.points()
    }

    /// The longest prefix of at most `n` positions a model holds right after
    /// the state is put back ([`SeqState::keep_point`]).
    #[must_use]
    pub fn keep_point(&self, n: u32) -> u32 {
        self.state.keep_point(n)
    }
}

impl SeqView for GlmSeq {
    fn positions(&self) -> u32 {
        GlmSeq::positions(self)
    }

    fn bytes(&self) -> usize {
        GlmSeq::bytes(self)
    }

    fn keep_point(&self, n: u32) -> u32 {
        GlmSeq::keep_point(self, n)
    }
}

/// The rows of positions below `n` of one latent store's three planes: its
/// latent rows, its index rows and the keys of the pools those positions
/// reach, each growing with the positions ([`Spans::by_position`],
/// [`Spans::by_pool`]), so a prefix of the state moves only its own rows.
fn latent_spans<'a>(
    [latent, index, pooled]: [&'a mut DeviceTensor<u16>; 3],
    n: usize,
) -> Result<[Spans<'a>; 3], GpuError> {
    Ok([
        Spans::by_position(latent.buf_mut(), 1, 0, n, LATENT)?,
        Spans::by_position(index.buf_mut(), 1, 0, n, INDEX_ROW)?,
        Spans::by_pool(pooled.buf_mut(), 1, 0, n, (POOL, INDEX_HEAD))?,
    ])
}

/// The stores a state copies, lent for one call: each latent layer's rows of
/// positions below `n` in layer order, then the NextN layer's; and each KDA
/// layer's committed lane `lane` and conv ring in layer order — the
/// checkpoints' list (`body::copied`'s).
fn lend<'a>(
    stores: &'a mut [Store],
    nextn: Option<&'a mut Nextn>,
    lane: u32,
    n: usize,
) -> Result<(Vec<Spans<'a>>, Vec<&'a mut DeviceBuffer<f32>>), GpuError> {
    let mut spans = Vec::new();
    let mut rec = Vec::new();
    for s in stores {
        match s {
            Store::Kda { state, ring, .. } => {
                rec.push(state.part_mut(lane)?);
                rec.push(ring);
            }
            Store::Latent {
                latent,
                index,
                pooled,
            } => spans.extend(latent_spans([latent, index, pooled], n)?),
        }
    }
    if let Some(nx) = nextn {
        spans.extend(latent_spans(nx.store_mut(), n)?);
    }
    Ok((spans, rec))
}

impl Body {
    /// What every save takes before its copy: a waiting cut carried out, the
    /// NextN side read, the last checkpoint below `pos` as the point, the
    /// stores lent, and `take` (the copy, on the engine stream or on a lane)
    /// run on them. Refused by name while a verify waits for its commit,
    /// after a step failed past its launch, and on a NextN load whose store
    /// holds positions past `pos`.
    fn take_state<R>(
        &mut self,
        stream: &CudaStream,
        pos: u32,
        take: impl FnOnce(
            Identity,
            &Lent<'_, '_, '_>,
            Vec<(u32, ())>,
            &Checkpoints,
        ) -> Result<R, GpuError>,
    ) -> Result<(R, Option<DraftRows>), GpuError> {
        self.stores_at(pos)?;
        self.apply_cut(stream)?;
        let draft = self.draft_rows(stream, pos)?;
        let points: Vec<(u32, ())> = self
            .ckpt
            .positions()
            .into_iter()
            .rev()
            .find(|&p| p < pos)
            .map(|p| (p, ()))
            .into_iter()
            .collect();
        let lane = self.s.lanes.committed();
        let (mut positional, mut recurrent) = lend(
            &mut self.stores,
            self.nextn.as_deref_mut(),
            lane,
            pos as usize,
        )?;
        let lent = Lent {
            positional: &mut positional,
            recurrent: &mut recurrent,
        };
        let taken = take(self.who.clone(), &lent, points, &self.ckpt)?;
        Ok((taken, draft))
    }

    /// The state at `pos`, where the stores stand ([`Body::take_state`]): the
    /// rows and stores the module doc lists, copied on the engine stream.
    fn save_state(&mut self, stream: &CudaStream, pos: u32) -> Result<GlmSeq, GpuError> {
        let (state, draft) = self.take_state(stream, pos, |who, lent, points, ckpt| {
            SeqState::save(stream, who, pos, lent, (), points, ckpt)
        })?;
        Ok(GlmSeq {
            state: Arc::new(state),
            draft,
        })
    }

    /// [`Body::save_state`] with the copy on `lane`'s stream into its pinned
    /// staging ([`SeqState::save_async`]): the state and its copy in flight,
    /// or [`NoRoom`], nothing begun. The NextN side is read here, blocking.
    fn save_state_async(
        &mut self,
        stream: &CudaStream,
        lane: &SaveLane,
        pos: u32,
    ) -> Result<Result<(GlmSeq, Pending<()>), NoRoom>, GpuError> {
        let (taken, draft) = self.take_state(stream, pos, |who, lent, points, ckpt| {
            let streams = Streams {
                lane,
                engine: stream,
            };
            SeqState::save_async(streams, who, pos, lent, (), points, ckpt)
        })?;
        Ok(taken.map(|pending| {
            let state = pending.shared();
            (GlmSeq { state, draft }, pending)
        }))
    }

    /// The first `ask` positions of `s` (or the nearest point below, `at`:
    /// [`SeqState::keep_point`]) put back on the empty model, the state a
    /// whole resume and a cut to `at` leave: the stores standing at `at`,
    /// every KDA layer's committed lane stamped there, the points up to `at`
    /// the checkpoints, and on a NextN load the layer's side cut to `at` too
    /// (the store holds no position past it, the arenas' rows past it are
    /// taken back). Only the rows of the first `at` positions go to the card.
    /// Returns `at`. Refused by name, nothing copied, onto a model that is
    /// not empty, for a NextN side the load does not take (a state with it
    /// onto a load without the layer, or the other way: the store layouts
    /// differ), and for another model's state or layout
    /// ([`SeqState::load_prefix`]).
    fn load_state(&mut self, stream: &CudaStream, s: &GlmSeq, ask: u32) -> Result<u32, GpuError> {
        const WHAT_L: &str = "glm5next resume";
        self.stores_at(0).map_err(|e| GpuError::Shape {
            what: WHAT_L,
            detail: format!("a resume onto a model that is not empty ({e}): reset first"),
        })?;
        let n = s.positions();
        match &s.draft {
            Some(d) => self.draft_fits(d, n)?,
            None if self.nextn.is_some() => {
                return Err(shape(
                    "a state without the NextN rows put back on a load with the layer".into(),
                ));
            }
            None => {}
        }
        let lane = self.s.lanes.committed();
        let at = {
            let (mut positional, mut recurrent) = lend(
                &mut self.stores,
                self.nextn.as_deref_mut(),
                lane,
                n as usize,
            )?;
            let mut lent = Lent {
                positional: &mut positional,
                recurrent: &mut recurrent,
            };
            s.state
                .load_prefix(stream, &self.who, &mut lent, &mut self.ckpt, ask)?
                .at
        };
        if let Some(d) = &s.draft {
            self.put_draft_rows(stream, d, n, at)?;
        }
        for st in &mut self.stores {
            st.restamp(stream, lane, at)?;
        }
        self.held = at;
        Ok(at)
    }
}

/// The sequence state on the host ([`GlmSeq`], the module doc): the stores'
/// rows below `m`'s position, the KDA layers' stores there and at the last
/// checkpoint below it, and on a NextN load the layer's side. A waiting cut
/// is carried out first. Refused by name on a poisoned model, while a verify
/// waits for its commit, and after a step failed past its launch.
pub fn seq_save(m: &mut GpuModel<Body>) -> Result<GlmSeq, GpuError> {
    const WHAT_S: &str = "glm5next snapshot";
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned {
            what: WHAT_S,
            fault,
        });
    }
    let pos = m.pos();
    let (gpu, _, body) = m.body_parts(WHAT_S)?;
    body.save_state(gpu.stream(), pos)
}

/// [`seq_save`] whose copy runs on `lane`'s stream into its pinned staging and
/// whose call returns once it is enqueued ([`SeqState::save_async`]): the
/// state and its copy in flight, which the model's rows are not cut, reset or
/// replaced before; or [`NoRoom`], nothing begun, when the staging holds no
/// free extent for it. Refused by name as [`seq_save`] refuses.
pub fn seq_save_async(
    m: &mut GpuModel<Body>,
    lane: &SaveLane,
) -> Result<Result<(GlmSeq, Pending<()>), NoRoom>, GpuError> {
    const WHAT_S: &str = "glm5next snapshot";
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned {
            what: WHAT_S,
            fault,
        });
    }
    let pos = m.pos();
    let (gpu, _, body) = m.body_parts(WHAT_S)?;
    body.save_state_async(gpu.stream(), lane, pos)
}

/// A save lane on `m`'s card sized to `limit` pinned bytes of states
/// ([`SaveLane::new`]): load-time only.
pub fn seq_lane(m: &mut GpuModel<Body>, limit: u64) -> Result<SaveLane, GpuError> {
    let (gpu, _, _) = m.body_parts("glm5next save lane")?;
    SaveLane::new(gpu.context(), limit)
}

/// Replace `m`'s sequence with `s`, which [`seq_save`] took of this load: a
/// reset unless the model is empty already (a reset just before, or the
/// load; the residency stays where use has taken it), then — as a pass of
/// `s`'s positions whose work is the copies — the state put back; the model
/// stands at its positions, its checkpoints the point `s` carried and no
/// other, and the calls after it are bit for bit those after the state was
/// taken. Refused by name, nothing copied, for another model's state (its
/// file, card or context) or layout.
pub fn seq_resume(m: &mut GpuModel<Body>, s: &GlmSeq) -> Result<(), GpuError> {
    if s.positions() == 0 {
        return seq_reset_for(m, "glm5next resume");
    }
    seq_resume_prefix(m, s, s.positions()).map(drop)
}

/// The model reset unless it is empty already: what every resume starts from.
fn seq_reset_for(m: &mut GpuModel<Body>, what: &'static str) -> Result<(), GpuError> {
    let b = m.body(what)?;
    let empty = m.pos() == 0
        && b.stores_at(0).is_ok()
        && b.ckpt.positions().is_empty()
        && !b.ckpt.pending();
    if !empty {
        m.reset()?;
    }
    Ok(())
}

/// [`seq_resume`] of the first `n` positions of `s` the state can be cut to
/// ([`GlmSeq::keep_point`]): the model stands at that many, which it
/// returns, as a whole resume and a rollback to it leave it, with only those
/// positions' rows moved to the card ([`SeqState::load_prefix`]). Refused by
/// name as [`seq_resume`] refuses, and for a prefix of no positions (reset the
/// model instead). `s`'s copy must have landed.
pub fn seq_resume_prefix(m: &mut GpuModel<Body>, s: &GlmSeq, n: u32) -> Result<u32, GpuError> {
    const WHAT_R: &str = "glm5next resume";
    seq_reset_for(m, WHAT_R)?;
    let at = s.keep_point(n);
    if at == 0 {
        return Err(shape(format!(
            "a prefix of 0 positions of a state of {} (asked {n}): reset the model instead",
            s.positions()
        )));
    }
    m.run_rows(at as usize, WHAT_R, |gpu, _, body, _, _| {
        let put = body.load_state(gpu.stream(), s, n)?;
        if put != at {
            return Err(shape(format!(
                "a prefix of {at} positions put back as {put}"
            )));
        }
        Ok(false)
    })?;
    Ok(at)
}

/// The host bytes a sequence state of `ctx` positions holds on the file
/// `inputs` describes, with the NextN layer's side when the load drafts
/// ([`GlmSeq::bytes`]'s terms at plan time):
/// every latent layer's rows of the positions and their pool keys (the
/// NextN layer's store beside them under the draft), two recurrent copies
/// of each KDA layer's committed lane and conv ring, and the draft's two
/// arenas' rows — the step's one and the verify's two.
#[must_use]
pub fn seq_bytes(inputs: &PlanInputs, ctx: usize, drafts: bool) -> u64 {
    let hp = &inputs.hp;
    let f16 = size_of::<u16>() as u64;
    let f32 = size_of::<f32>() as u64;
    let (kda, latent) = inputs
        .spec
        .layers
        .iter()
        .fold((0u64, 0u64), |(kda, latent), l| match l.mixer {
            Mixer::DeltaRule(_) => (kda + 1, latent),
            Mixer::Latent(_) => (kda, latent + 1),
            _ => (kda, latent),
        });
    let positional = |n: usize| (n * (LATENT + INDEX_ROW) + pools_for(n) * INDEX_HEAD) as u64 * f16;
    // The load's every KDA layer one shape (`body::dims_of`): heads of the
    // head count each, the head dim the kernels'.
    let shape = LinearShape {
        n_k: hp.n_head,
        n_v: hp.n_head,
        map: KHeadMap::Tiled,
    };
    let recurrent = 2 * kda * (shape.state_len() + shape.ring_len()) as u64 * f32;
    let draft = |n: usize| positional(n) + 3 * (HC_STREAMS * hp.n_embd) as u64 * f32;
    latent * positional(ctx) + recurrent + u64::from(drafts) * draft(ctx)
}
