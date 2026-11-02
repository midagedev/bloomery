//! Mistral Large 4's DSpark draft, driven by the common loop
//! ([`crate::dspark`]): the owner over the file's MLA stages
//! (`bloomery_gpu_mistrallarge4::draft`) on a card of its own. [`taps`]
//! names the target layers whose rows the draft reads, for the target's
//! feature tap, which is attached before the target captures anything;
//! [`open_draft`] loads the draft file on its card; everything after is
//! [`CardDspark`]'s.

use std::sync::Arc;

use bloomery_gpu::model::ChainBody;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_mistrallarge4::draft::{self, Ml4Layers};
use cuda_core::CudaContext;
use gguf::Split;
use model::arch::mistrallarge4::dspark::{Hparams, shape};
use model::placement::workstation::DeviceId;
use runtime::Verify;

use crate::dspark::CardDspark;
use crate::{Session, SessionError};

/// Mistral Large 4's DSpark draft: the common loop over its MLA stages.
pub type Ml4Dspark = CardDspark<Ml4Layers>;

/// The target layers `draft` reads, as the file lists them (the entry
/// `n_layer` is the row the target's last layer leaves): the layers the
/// target's feature tap is attached for. A file that is not this family's
/// draft is refused by its key.
pub fn taps(draft: &Split) -> Result<Vec<usize>, SessionError> {
    let hp = Hparams::read(draft)
        .map_err(|e| SessionError::Refused(format!("the draft's statement: {e}")))?;
    Ok(hp.target_layers)
}

/// The draft of `draft` on card `card` — its plan's device `device` when the
/// placement resolved one, else the one device of that name — borrowing
/// `target_file`'s `output` (copied to the draft's card) and `token_embd`
/// (read on the host), for the target `B` whose context is `target`. The
/// target's feature tap for [`taps`] is the caller's, attached before the
/// target captures anything. The draft's resident bytes are the buffers it
/// allocated ([`draft::held`]); the card's loss around the load is kept as
/// a diagnostic. The calling thread is bound to `target` again before this
/// returns, the load failed or not. Load-time only.
pub fn open_draft<B>(
    target: Arc<CudaContext>,
    draft: &Split,
    target_file: Arc<Split>,
    (card, device): (&'static str, Option<DeviceId>),
) -> Result<Ml4Dspark, SessionError>
where
    B: ChainBody,
    Session<B>: Verify,
{
    let gpu = Gpu::open_card(card, device)?;
    let loaded = (|| {
        let (free, _) = gpu.mem_info()?;
        let body = draft::open(&gpu, draft, target_file)?;
        let (left, _) = gpu.mem_info()?;
        Ok::<_, GpuError>((body, free.saturating_sub(left)))
    })();
    target.bind_to_thread().map_err(GpuError::from)?;
    let (body, taken) = loaded?;
    let held = draft::held(&body);
    let shape = shape(body.layers().hp());
    CardDspark::new::<B>(body, gpu, target, card, shape, held, taken)
}
