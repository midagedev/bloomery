//! V4.1's DSpark draft, driven by the common loop ([`crate::dspark`]): the
//! owner over V4.1's layers (`bloomery_gpu_deepseek41::draft::DraftBody`) on a
//! card of its own, fed the target's feature tap (`body::attach_features`,
//! `Body::read_features`). [`open_draft`] loads the draft file on that card
//! and attaches the tap to the loaded target; everything after is
//! [`CardDspark`]'s.

use std::sync::Arc;

use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::body::{self, Body};
use bloomery_gpu_deepseek41::draft::load::DraftWeights;
use bloomery_gpu_deepseek41::draft::{self, V41Layers};
use gguf::Split;
use model::arch::deepseek41::spec::draft_of;
use model::arch::dspark::{DraftHparams, shape};
use model::placement::workstation::DeviceId;
use runtime::Program;

use crate::dspark::CardDspark;
use crate::{Loaded, SessionError};

/// V4.1's DSpark draft: the common loop over V4.1's layers.
pub type Ds41Draft = CardDspark<V41Layers>;

/// The draft of `draft` on card `card` — its plan's device `device` when the
/// placement resolved one, else the one device of that name — borrowing
/// `target_file`'s head and embedding rows, and the loaded target's feature
/// tap built for the draft's target layers — before the target captures
/// anything, which [`Loaded`] guarantees. The draft's resident bytes are the
/// buffers it allocated ([`draft::held`]); the card's loss around the load is
/// kept as a diagnostic. Load-time only.
pub fn open_draft(
    target: &mut Loaded<Body>,
    draft: &Split,
    hp: &DraftHparams,
    target_file: Arc<Split>,
    (card, device): (&'static str, Option<DeviceId>),
) -> Result<Ds41Draft, SessionError> {
    let m = target.model_mut();
    let desc =
        draft_of(hp).map_err(|e| SessionError::Refused(format!("the draft's description: {e}")))?;
    let Program::Block(spec) = runtime::program(&desc)
        .map_err(|e| SessionError::Refused(format!("the draft's description: {e}")))?;
    let taps: Vec<usize> = spec.target_layers.iter().map(|&l| l as usize).collect();
    body::attach_features(m, &taps)?;
    let ctx = m.gpu().context().clone();
    let gpu = Gpu::open_card(card, device)?;
    let loaded = (|| {
        let (free, _) = gpu.mem_info()?;
        let w = DraftWeights::load(gpu.stream(), draft, &target_file)?;
        let body = draft::open(&gpu, w, target_file)?;
        let (left, _) = gpu.mem_info()?;
        Ok::<_, GpuError>((body, free.saturating_sub(left)))
    })();
    ctx.bind_to_thread().map_err(GpuError::from)?;
    let (body, taken) = loaded?;
    let held = draft::held(&body)?;
    let shape = shape(body.layers().weights().hp());
    CardDspark::new::<Body>(body, gpu, ctx, card, shape, held, taken)
}
