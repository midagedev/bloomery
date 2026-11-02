//! What a server needs of the Mistral Large 4 body beside its step: the
//! sequence it holds as a state ([`PlaneBody`]), the cut it takes
//! ([`Rollback`]), the resident sequences it parks ([`Slots`]) and the media
//! rows its prompt call stages in place of token embeddings ([`MediaBody`]).
//!
//! The module is the body module's child (`body.rs` declares it), so it reads
//! the stores the body keeps private.
//!
//! Every store of a sequence is a plane of positions: a layer that owns a
//! cache holds its latent rows (`[latent | rope]` f16, a row a position) and a
//! layer with an indexer of its own holds its index keys (f16, a row a
//! position). The list and the indexer's scores are the step's scratch,
//! recomputed from the keys each step, and a layer that reads another's cache
//! or list holds nothing. So a sequence is those two kinds of rows of the
//! held positions and nothing more, a step writes its position's row before
//! it reads it, and any prefix of a sequence is a valid cut.
//!
//! A sequence parked with the model ([`Seq`]) is those planes alone: a select
//! exchanges the planes of every store and leaves the step's scratch where it
//! is, which is why the plan counts a sequence's K/V bytes and no more
//! (`PlanInputs::plan_with_slots`).

use bloomery_gpu::checkpoint::planes::{PlaneBody, head_major};
use bloomery_gpu::checkpoint::saved::{Identity, Spans};
use bloomery_gpu::media::{MediaBody, MediaSpan, check_spans};
use bloomery_gpu::model::{Rollback, Slots};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, GpuModel};
use gguf::Split;
use model::arch::Arch;
use model::placement::Plan;

use super::{Body, LayerStore};
use crate::prefill::prefill;

/// What the sequence operations name in their refusals.
const WHAT: &str = "mistrallarge4 sequence";

/// What the served body carries beside the step: whom a saved state belongs
/// to, and the resident sequences the load's plan counted against the ones
/// made.
pub(crate) struct State {
    who: Identity,
    planned: usize,
    made: usize,
}

impl State {
    /// The state of a load of `file` on `gpu` with stores of `ctx` positions
    /// over `plan`'s card `card`: `one` is the bytes of one sequence's K/V
    /// planes, and the plan must count a whole number of them.
    pub(crate) fn of_load(
        gpu: &Gpu,
        file: &Split,
        plan: &Plan<'_>,
        card: usize,
        (one, ctx): (u64, usize),
    ) -> Result<State, GpuError> {
        let counted = plan
            .cards
            .get(card)
            .map(|c| c.kv_bytes)
            .ok_or_else(|| GpuError::shape(WHAT, format!("the plan has no card {card}")))?;
        if one == 0 || counted < one || counted % one != 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the plan counts {counted} store bytes on card {card}: not a whole number of \
                     sequences of {one} bytes"
                ),
            ));
        }
        let planned = usize::try_from(counted / one)
            .map_err(|_| GpuError::shape(WHAT, "a sequence count past usize"))?;
        Ok(State {
            who: Identity::of_load(Arch::MistralLarge4.name(), file, gpu, ctx)?,
            planned,
            made: 1,
        })
    }
}

impl Body {
    /// The batches a prompt group runs layer by layer, as the body holds them: the load's one
    /// until [`crate::prefill::set_prefill_group`] sets the lever's.
    #[must_use]
    pub fn prefill_group(&self) -> usize {
        self.units.group()
    }
}

impl Rollback for Body {
    /// A cut to `pos`: every store is a plane of positions the launches read
    /// only below a row's own live count, and the step and the prompt batch
    /// both write a position's rows before any launch reads them, so a cut
    /// leaves nothing to take back.
    fn rollback(&mut self, _pos: u32) -> Result<(), GpuError> {
        Ok(())
    }
}

impl PlaneBody for Body {
    const WHAT: &'static str = WHAT;

    fn who(&self) -> &Identity {
        &self.seat.who
    }

    /// Each layer's latent rows, then its index keys, in layer order; a layer
    /// that holds one kind lends that one.
    fn lend(&mut self, n: usize) -> Result<Vec<Spans<'_>>, GpuError> {
        let ctx = self.sh.ctx;
        let mut spans = Vec::with_capacity(2 * self.stores.len());
        for store in &mut self.stores {
            let LayerStore { latent, idx } = store;
            if let Some(t) = latent {
                let width = t.cols();
                spans.push(head_major(t, (1, ctx, width), n)?);
            }
            if let Some(i) = idx {
                let width = i.keys.cols();
                spans.push(head_major(&mut i.keys, (1, ctx, width), n)?);
            }
        }
        Ok(spans)
    }
}

/// One layer's K/V planes of a parked sequence, in the shape of the live
/// store's: a latent cache on a layer that owns one, index keys on a layer
/// whose indexer is its own.
struct Planes {
    latent: Option<DeviceTensor<u16>>,
    keys: Option<DeviceTensor<u16>>,
}

/// One resident sequence ([`Slots`], what a select exchanges): every layer's
/// K/V planes.
pub struct Seq {
    planes: Vec<Planes>,
}

impl Slots for Body {
    type Seq = Seq;

    /// A sequence in the state the load leaves ([`bloomery_gpu::model::ChainBody::reset`]'s
    /// contract): zeroed planes of the live stores' shapes. Refused by name
    /// past the sequences the load's plan counts, and when the card has fewer
    /// free bytes than the planes take. Load-time allocation.
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Seq, GpuError> {
        if self.seat.made >= self.seat.planned {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "sequence {} of a plan that counts {} resident sequences; plan the load for \
                     as many as it serves (PlanInputs::plan_with_slots)",
                    self.seat.made + 1,
                    self.seat.planned
                ),
            ));
        }
        let need = self.store_bytes();
        let free = gpu.alloc_room()?;
        if need > free {
            return Err(GpuError::shape(
                WHAT,
                format!("a sequence of {need} B of stores, and the card has {free} B free"),
            ));
        }
        let stream = gpu.stream();
        let zeroed = |t: &DeviceTensor<u16>| DeviceTensor::zeroed(stream, t.rows(), t.cols());
        let planes = self
            .stores
            .iter()
            .map(|s| {
                Ok(Planes {
                    latent: s.latent.as_ref().map(zeroed).transpose()?,
                    keys: s.idx.as_ref().map(|i| zeroed(&i.keys)).transpose()?,
                })
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        // After the allocation, so a sequence that failed to allocate is not counted.
        self.seat.made += 1;
        Ok(Seq { planes })
    }

    /// Exchange the live sequence's planes with `seq`'s: pointer moves, no
    /// device work. Refused by name, nothing moved, when `seq`'s layers do
    /// not hold the kinds of planes the live stores do.
    fn swap_seq(&mut self, _gpu: &Gpu, seq: &mut Seq) -> Result<(), GpuError> {
        let alike = seq.planes.len() == self.stores.len()
            && self.stores.iter().zip(&seq.planes).all(|(s, p)| {
                s.latent.is_some() == p.latent.is_some() && s.idx.is_some() == p.keys.is_some()
            });
        if !alike {
            return Err(GpuError::shape(
                WHAT,
                "a parked sequence whose layers hold other planes than the live stores",
            ));
        }
        for (store, parked) in self.stores.iter_mut().zip(&mut seq.planes) {
            std::mem::swap(&mut store.latent, &mut parked.latent);
            if let (Some(idx), Some(keys)) = (store.idx.as_mut(), parked.keys.as_mut()) {
                std::mem::swap(&mut idx.keys, keys);
            }
        }
        Ok(())
    }

    /// Device bytes one sequence holds: its planes. The step's scratch is the
    /// model's, one for every sequence.
    fn seq_bytes(&self) -> usize {
        self.store_bytes()
    }
}

impl MediaBody for Body {
    type Span<'a> = MediaSpan<'a>;

    fn media_span<'s>(s: &'s MediaSpan<'_>) -> &'s MediaSpan<'s> {
        s
    }

    /// The prompt call of `ids` from the model's position, one pass of the batched walk
    /// ([`prefill`]) inside the residency pass the session opens around a call
    /// (`app::Session::prompt_media`): a span's positions stage the span's rows in place of their
    /// ids' embeddings ([`bloomery_gpu_deepseek41::prompt_walk::BatchBufs::set_media`]) and every
    /// position takes the walk's bits, so the call leaves what a text prompt of the same length
    /// does, and the greedy token after the last id is the head's. Refused by name before anything
    /// runs: a span the call does not hold whole, spans out of order, a call of no id; and by name
    /// from the walk, a call that staged fewer rows than the spans hold.
    fn prefill_media(
        m: &mut GpuModel<Body>,
        ids: &[u32],
        spans: &[MediaSpan<'_>],
    ) -> Result<u32, GpuError> {
        let n_embd = m.body(WHAT)?.sh.hidden;
        if ids.is_empty() {
            return Err(GpuError::shape(WHAT, "a media prompt of no ids"));
        }
        check_spans(WHAT, spans, ids.len(), n_embd)?;
        let first = m.pos();
        m.body_parts(WHAT)?
            .2
            .units
            .set_media(first, n_embd, spans)?;
        let ran = m.run_rows(ids.len(), WHAT, |gpu, w, body, head, at| {
            prefill(gpu, w, body, head, ids, at)
        });
        // The walk takes the rows it was given; rows left are those of a pass refused before it.
        m.body_parts(WHAT)?.2.units.clear_media();
        ran?.ok_or(GpuError::State {
            what: WHAT,
            missing: "the head's token after the media prompt",
        })
    }
}
