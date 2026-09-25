//! The DSpark draft a V4.1 decode serves, at block width [`WIDTH`]: the draft
//! body (`bloomery_gpu_deepseek41::draft::DraftBody`) on its own card, fed the
//! target's features (`body::attach_features`, `Body::read_features`), in the
//! order the draft module's contract fixes:
//!
//! 1. [`Dspark::reset`], then [`Dspark::append`] of the fed positions'
//!    features in order: every position's when fed step by step, the last
//!    [`Dspark::window`]'s when batched ([`Dspark::skip_to`] past the rest);
//! 2. per pass ([`pass`]): `d = propose(next)`, the target's pair pass over
//!    `[next, d]`; row A's argmax equal to `d` accepts both positions and
//!    appends both rows' features, otherwise the second position is taken
//!    back and row A's features alone are appended.
//!
//! The proposal changes which passes run, never a token: every emitted token
//! is the target's own argmax, so the tokens are the plain greedy run's.
//!
//! The draft's card is `BLOOMERY_DSPARK_CARD` (a placement card name), the
//! 3090 when unset — plan (a)'s idle card. It may be the target's own card
//! (the gate runs both on the 3090 under a card budget); the two then share
//! the card's primary context. Every call binds the draft's context and binds
//! the target's back before it returns, so the target's next launch never runs
//! under the draft's context. A device fault the draft raises comes back as
//! the draft's `GpuError::Fault` — from a proposal's readback, or from
//! [`Dspark::check_fault`] after the last append — and ends the run: no pass
//! falls back to a plain step.
//!
//! This is a bin module, not the gates library's: the library never names
//! the V4.1 device crate.

use std::path::PathBuf;
use std::sync::Arc;

use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::body::{self, Deepseek41Model};
use bloomery_gpu_deepseek41::draft::DraftBody;
use bloomery_gpu_deepseek41::draft::kv::GROUP;
use bloomery_gpu_deepseek41::draft::load::DraftWeights;
use bloomery_gpu_gates::GateError;
use cuda_core::CudaContext;
use gguf::Split;
use model::arch::dspark::DraftHparams;
use model::placement::workstation;

/// The block width every proposal is pinned to.
pub const WIDTH: usize = 1;

const WHAT: &str = "dspark loop";

/// The draft file: `$BLOOMERY_DSPARK_MODEL`, which the recipes export from
/// the V4.1 profile's `DSPARK_MODEL` (`tools/ref/models/deepseek41.sh`).
pub fn draft_path() -> Result<PathBuf, GateError> {
    match std::env::var_os("BLOOMERY_DSPARK_MODEL") {
        Some(p) if !p.is_empty() => Ok(PathBuf::from(p)),
        _ => Err(
            "BLOOMERY_DSPARK_MODEL unset — export the V4.1 profile's DSPARK_MODEL \
                  (`. tools/ref/ref-paths.sh` under BLOOMERY_MODEL=deepseek41), as the dspark \
                  recipes do"
                .into(),
        ),
    }
}

/// The draft file's hyperparameters, read from its header alone.
pub fn draft_hparams() -> Result<(Split, DraftHparams), GateError> {
    let path = draft_path()?;
    let draft = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let hp = DraftHparams::read(&draft)?;
    Ok((draft, hp))
}

/// `BLOOMERY_DSPARK_CARD`: a placement card name; unset is the 3090.
pub fn draft_card() -> Result<&'static str, GateError> {
    let cards = [workstation::RTX_3090.name, workstation::A6000.name];
    match std::env::var("BLOOMERY_DSPARK_CARD") {
        Err(std::env::VarError::NotPresent) => Ok(workstation::RTX_3090.name),
        Ok(v) => cards
            .into_iter()
            .find(|c| *c == v)
            .ok_or_else(|| format!("BLOOMERY_DSPARK_CARD is one of {cards:?}, not {v:?}").into()),
        Err(e) => Err(format!("BLOOMERY_DSPARK_CARD: {e}").into()),
    }
}

/// The draft on its card, and the target's context to hand back.
pub struct Dspark {
    /// Declared first: dropped before its card's `Gpu`.
    body: DraftBody,
    gpu: Gpu,
    target: Arc<CudaContext>,
    card: &'static str,
    /// Prompt features waiting for a whole group, `GROUP` rows at most.
    pending: Vec<f32>,
    width: usize,
}

impl Dspark {
    /// The draft of `draft` on card `card`, borrowing `target_file`'s head and
    /// embedding rows, and `m`'s feature tap built for its `target_layers`
    /// (before `m` captures anything). Load-time only.
    pub fn open(
        m: &mut Deepseek41Model,
        draft: &Split,
        hp: &DraftHparams,
        target_file: Arc<Split>,
        card: &'static str,
    ) -> Result<Dspark, GateError> {
        body::attach_features(m, &hp.target_layers)?;
        let target = target_ctx(m)?;
        let width = m.body(WHAT)?.feature_width();
        let gpu = Gpu::for_card(card)?;
        let loaded = (|| {
            let w = DraftWeights::load(gpu.stream(), draft, &target_file)?;
            DraftBody::new(&gpu, w, target_file)
        })();
        target.bind_to_thread()?;
        let body = loaded?;
        if body.weights().hp().target_layers.len() * body.weights().hp().n_embd != width {
            return Err(format!(
                "the draft reads {} features a row, the target's tap writes {width}",
                body.weights().hp().target_layers.len() * body.weights().hp().n_embd
            )
            .into());
        }
        Ok(Dspark {
            body,
            gpu,
            target,
            card,
            pending: Vec::with_capacity(GROUP * width),
            width,
        })
    }

    /// The card the draft runs on.
    #[must_use]
    pub fn card(&self) -> &'static str {
        self.card
    }

    /// `(free, total)` device bytes of the draft's card.
    pub fn mem_info(&mut self) -> Result<(usize, usize), GpuError> {
        self.on_card(|gpu, _| gpu.mem_info())
    }

    /// Positions appended so far.
    #[must_use]
    pub fn committed(&self) -> u32 {
        self.body.committed()
    }

    /// The draft's window, from its file: the trailing positions of a prompt
    /// whose features it keeps.
    #[must_use]
    pub fn window(&self) -> usize {
        self.body.weights().hp().window
    }

    /// Commit the positions before `to` without their features, the queued
    /// ones appended first: a batched prompt hands over only the rows the
    /// draft's window keeps.
    pub fn skip_to(&mut self, to: u32) -> Result<(), GpuError> {
        if to > self.body.committed() + (self.pending.len() / self.width) as u32 {
            self.flush()?;
            self.body.skip(to)?;
        }
        Ok(())
    }

    /// `f` under the draft's context, the target's bound again after it.
    fn on_card<T>(
        &mut self,
        f: impl FnOnce(&Gpu, &mut DraftBody) -> Result<T, GpuError>,
    ) -> Result<T, GpuError> {
        self.gpu.context().bind_to_thread()?;
        let r = f(&self.gpu, &mut self.body);
        self.target.bind_to_thread()?;
        r
    }

    /// Forget the sequence.
    pub fn reset(&mut self) -> Result<(), GpuError> {
        self.pending.clear();
        self.on_card(|gpu, body| body.reset(gpu))
    }

    /// Queue a fed position's features; a whole group is appended at once,
    /// through its captured graph. [`Dspark::flush`] appends the rest.
    pub fn feed(&mut self, feats: &[f32]) -> Result<(), GpuError> {
        self.pending.extend_from_slice(feats);
        if self.pending.len() == GROUP * self.width {
            self.flush()?;
        }
        Ok(())
    }

    /// Append the queued fed positions.
    pub fn flush(&mut self) -> Result<(), GpuError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let rows = std::mem::take(&mut self.pending);
        let r = self.append(&rows);
        self.pending = rows;
        self.pending.clear();
        r
    }

    /// Append the features of the next committed positions (whole rows).
    pub fn append(&mut self, feats: &[f32]) -> Result<(), GpuError> {
        self.on_card(|gpu, body| body.append(gpu, feats).map(|_| ()))
    }

    /// The draft's id after `id_last`, the token at position [`Dspark::committed`].
    pub fn propose(&mut self, id_last: u32) -> Result<u32, GpuError> {
        let ids = self.on_card(|gpu, body| body.propose(gpu, id_last, WIDTH))?;
        match ids.as_slice() {
            &[d] => Ok(d),
            other => Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a proposal of width {WIDTH} returned {} ids", other.len()),
            }),
        }
    }

    /// The draft card's fault word, read after the last append: raised is
    /// the draft's `GpuError::Fault`.
    pub fn check_fault(&mut self) -> Result<(), GpuError> {
        match self.on_card(|gpu, _| gpu.fault())? {
            None => Ok(()),
            Some(fault) => Err(GpuError::Fault { what: WHAT, fault }),
        }
    }
}

/// The context of `m`'s one stage.
fn target_ctx(m: &Deepseek41Model) -> Result<Arc<CudaContext>, GateError> {
    Ok(m.stages()
        .first()
        .ok_or("dspark loop: the model has no stage")?
        .gpu()
        .context()
        .clone())
}

/// Step the fed `ids` one position at a time, each position's features into
/// the draft after a reset, and return the first generated token. One step
/// per id: `step(&ids)` runs every body before its one readback, so only the
/// last position's features would survive it.
pub fn feed(m: &mut Deepseek41Model, d: &mut Dspark, ids: &[u32]) -> Result<u32, GateError> {
    d.reset()?;
    let mut next = None;
    for &id in ids {
        next = Some(m.step(&[id])?);
        let (gpu, _, b) = m.body_parts(WHAT)?;
        d.feed(b.read_features(gpu, 1)?.values)?;
    }
    d.flush()?;
    next.ok_or_else(|| "dspark loop: no id to feed".into())
}

/// What one pass verified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Row A's argmax was the proposal: row A's and row B's tokens.
    Accept([u32; 2]),
    /// It was not: row A's token; the second position was taken back.
    Reject(u32),
}

/// One pass from `next`, the token at the model's position: propose, the
/// pair pass over `[next, d]`, the verdict, and the accepted positions'
/// features appended.
pub fn pass(m: &mut Deepseek41Model, d: &mut Dspark, next: u32) -> Result<Verdict, GateError> {
    let pos = m.pos();
    if d.committed() != pos {
        return Err(format!(
            "dspark loop: the draft has {} positions, the target stands at {pos}",
            d.committed()
        )
        .into());
    }
    let draft = d.propose(next)?;
    let [ta, tb] = m.step_pair(next, draft)?;
    let accept = ta == draft;
    let rows = if accept { 2 } else { 1 };
    {
        let (gpu, _, b) = m.body_parts(WHAT)?;
        d.append(b.read_features(gpu, rows)?.values)?;
    }
    if accept {
        Ok(Verdict::Accept([ta, tb]))
    } else {
        m.rollback(pos + 1)?;
        Ok(Verdict::Reject(ta))
    }
}
