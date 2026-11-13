//! Flint attention plugged into the common prompt walk and shared FFN.

use bloomery_gpu::attention_h128::{Binding, Cache, Kernels, Scratch};
use bloomery_gpu::head::Head;
use bloomery_gpu::model::ChainBody;
use bloomery_gpu::rope_table::RopeRows;
use bloomery_gpu::routed_block::RouterEntry;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use bloomery_gpu_deepseek41::prompt_walk::{Batch, BatchAttention, UnitRows};
use runtime::layer::{FfnKind, Layer};

use crate::body::{FlintBody, shape};

/// The plan and the ring both hold one common prompt unit.
pub const T_MAX: usize = model::arch::flint::place::PROMPT_ROWS;
const _: () = assert!(T_MAX as u64 == runtime::stores::PROMPT_BATCH);

/// Run a prompt through real attention and dense/routed blocks, then the final head.
pub fn prefill<R: RouterEntry>(
    gpu: &Gpu,
    w: &Weights,
    body: &mut FlintBody<R>,
    head: &mut Head,
    ids: &[u32],
    at: u32,
) -> Result<bool, GpuError> {
    call(gpu, w, body, head, ids, at, false)
}

/// [`prefill`] of one prompt unit (at most [`T_MAX`] ids) that also leaves the unit's final hidden rows
/// in the draft's prefill arena, with its position record ([`crate::draft`]). Refused by name: a load
/// without the draft and a call of more than one unit.
pub fn prefill_draft<R: RouterEntry>(
    gpu: &Gpu,
    w: &Weights,
    body: &mut FlintBody<R>,
    head: &mut Head,
    ids: &[u32],
    at: u32,
) -> Result<bool, GpuError> {
    if body.draft.is_none() {
        return Err(shape(
            "a drafted prompt unit on a load without the MTP layer",
        ));
    }
    if ids.len() > T_MAX {
        return Err(shape(format!(
            "a drafted prompt unit of {} ids; a unit holds {T_MAX}",
            ids.len()
        )));
    }
    call(gpu, w, body, head, ids, at, true)
}

fn call<R: RouterEntry>(
    gpu: &Gpu,
    w: &Weights,
    body: &mut FlintBody<R>,
    head: &mut Head,
    ids: &[u32],
    at: u32,
    tapped: bool,
) -> Result<bool, GpuError> {
    let to =
        runtime::prompt::call_end(at, ids.len(), body.ctx).map_err(|e| shape(e.to_string()))?;
    for &id in ids {
        if id as usize >= body.embed.n_vocab() {
            return Err(shape(format!(
                "prompt token {id} past vocabulary {}",
                body.embed.n_vocab()
            )));
        }
    }
    // A failed call may already have overwritten ring rows. Reset, not a lower logical position, repairs that state.
    body.at = body.at.max(to);
    let layers = body.cfg.len();
    let FlintBody {
        shell,
        cfg,
        bindings,
        stores,
        k,
        batch_attention,
        rope,
        units,
        ffn,
        embed,
        hidden,
        ctx,
        draft,
        ..
    } = body;
    let mut attention = Attention {
        gpu,
        w,
        cfg,
        bindings,
        stores,
        k,
        scratch: batch_attention,
        rope,
    };
    let mut walk = Batch {
        gpu,
        w,
        bufs: units,
        attention: &mut attention,
        ffn,
        hybrid: &mut shell.hybrid,
        embed,
        head,
        taps: shell.taps.as_deref_mut(),
        embd: *hidden,
        ctx: *ctx,
        layers,
    };
    let mut cursor = at;
    match draft.as_mut().filter(|_| tapped) {
        Some(d) => {
            let mut tap = d.tap(layers, at);
            let ran = walk.prompt_entering(ids, &mut cursor, &mut tap)?;
            d.wrote_prefill(at, ids.len());
            Ok(ran)
        }
        None => walk.prompt(ids, &mut cursor),
    }
}

struct Attention<'a> {
    gpu: &'a Gpu,
    w: &'a Weights,
    cfg: &'a [Layer],
    bindings: &'a [Binding],
    stores: &'a mut [Cache],
    k: &'a Kernels,
    scratch: &'a mut Scratch,
    rope: &'a RopeRows,
}

impl BatchAttention for Attention<'_> {
    fn attend(
        &mut self,
        l: usize,
        _u: usize,
        m: usize,
        rows: UnitRows<'_>,
    ) -> Result<(), GpuError> {
        // SAFETY: the common walk stages consecutive positions and each count is position + 1.
        unsafe {
            self.bindings[l].enqueue(
                self.gpu,
                self.w,
                self.k,
                self.scratch,
                &mut self.stores[l],
                rows.x,
                rows.x1,
                rows.pos,
                rows.n_keys,
                m,
                self.rope,
            )
        }
    }
    fn ffn(&self, l: usize) -> FfnKind {
        self.cfg[l].ffn
    }
}

/// Change prompt grouping between calls; allocation stays within the plan's reserve.
pub fn set_prefill_group<R: RouterEntry>(
    m: &mut GpuModel<FlintBody<R>>,
    group: usize,
) -> Result<bool, GpuError>
where
    FlintBody<R>: ChainBody,
{
    let (gpu, _, body) = m.body_parts("flint prompt group")?;
    let made = body.units.set_group(gpu, body.hidden, group)?;
    body.check_prompt_reserve()?;
    Ok(made)
}
