//! A GLM sequence state on the host ([`GlmSeq`], over the shared
//! `bloomery_gpu::checkpoint::saved`): what a server's prompt cache or its
//! slot swap holds for a sequence it switches away from.
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

use bloomery_gpu::checkpoint::saved::{Lent, SeqState, Spans};
use bloomery_gpu::latent::{INDEX_HEAD, INDEX_ROW, LATENT, pools_for};
use bloomery_gpu::{DeviceTensor, GpuError, GpuModel};
use cuda_core::{CudaStream, DeviceBuffer};

use super::nextn::{DraftRows, Nextn};
use super::{Body, Store, shape};

/// A GLM sequence state ([`seq_save`]): the shared state of the module
/// doc's rows and stores, and on a NextN load the layer's side beside it.
pub struct GlmSeq {
    state: SeqState<()>,
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

/// The rows of positions below `n` of one latent store's three planes: its
/// latent rows, its index rows and the keys of the pools those positions
/// reach.
fn latent_spans<'a>(
    [latent, index, pooled]: [&'a mut DeviceTensor<u16>; 3],
    n: usize,
) -> Result<[Spans<'a>; 3], GpuError> {
    Ok([
        Spans::planes(latent.buf_mut(), 1, 0, n * LATENT)?,
        Spans::planes(index.buf_mut(), 1, 0, n * INDEX_ROW)?,
        Spans::planes(pooled.buf_mut(), 1, 0, pools_for(n) * INDEX_HEAD)?,
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
    /// The state at `pos`, where the stores stand, a waiting cut carried out
    /// first: the rows and stores the module doc lists, the point the last
    /// checkpoint below `pos`. Refused by name while a verify waits for its
    /// commit, after a step failed past its launch, and on a NextN load whose
    /// store holds positions past `pos`.
    fn save_state(&mut self, stream: &CudaStream, pos: u32) -> Result<GlmSeq, GpuError> {
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
        let state = SeqState::save(stream, self.who.clone(), pos, &lent, (), points, &self.ckpt)?;
        Ok(GlmSeq { state, draft })
    }

    /// `s` put back on the empty model: the stores standing at its
    /// positions, every KDA layer's committed lane stamped there, the points
    /// it carried the checkpoints, its NextN side the layer's. Refused by
    /// name, nothing copied, onto a model that is not empty, for a NextN
    /// side the load does not take (a state with it onto a load without the
    /// layer, or the other way: the store layouts differ), and for another
    /// model's state or layout ([`SeqState::load`]).
    fn load_state(&mut self, stream: &CudaStream, s: &GlmSeq) -> Result<(), GpuError> {
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
        {
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
            s.state.load(stream, &self.who, &mut lent, &mut self.ckpt)?;
        }
        if let Some(d) = &s.draft {
            self.put_draft_rows(stream, d, n)?;
        }
        for st in &mut self.stores {
            st.restamp(stream, lane, n)?;
        }
        self.held = n;
        Ok(())
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

/// Replace `m`'s sequence with `s`, which [`seq_save`] took of this load: a
/// reset unless the model is empty already (a reset just before, or the
/// load; the residency stays where use has taken it), then — as a pass of
/// `s`'s positions whose work is the copies — the state put back; the model
/// stands at its positions, its checkpoints the point `s` carried and no
/// other, and the calls after it are bit for bit those after the state was
/// taken. Refused by name, nothing copied, for another model's state (its
/// file, card or context) or layout.
pub fn seq_resume(m: &mut GpuModel<Body>, s: &GlmSeq) -> Result<(), GpuError> {
    const WHAT_R: &str = "glm5next resume";
    let b = m.body(WHAT_R)?;
    let empty = m.pos() == 0
        && b.stores_at(0).is_ok()
        && b.ckpt.positions().is_empty()
        && !b.ckpt.pending();
    if !empty {
        m.reset()?;
    }
    let n = s.positions() as usize;
    if n == 0 {
        return Ok(());
    }
    m.run_rows(n, WHAT_R, |gpu, _, body, _, _| {
        body.load_state(gpu.stream(), s)?;
        Ok(false)
    })?;
    Ok(())
}
