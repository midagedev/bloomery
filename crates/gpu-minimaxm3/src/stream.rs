//! MiniMax-M3.1's prompt calls through the common walk's host streaming (`BLOOMERY_XSTREAM`,
//! [`bloomery_gpu_deepseek41::prompt_walk`]): the family hands the walk its facts as data — the
//! floor a pick admits from, no card tail record, and under `split` the stream's wait bound (the
//! walk prices a routed expert on this device and host from its widths) — and runs no stream code
//! of its own. The guards, the stream's start, the picks, the
//! ring pass (route GEMMs over the ring's half, a batch a landing batch) and the join are the walk's.

use bloomery_gpu::host::swap_source::DEADLINE;
use bloomery_gpu::host::xstream::{Costs, XLayer, XMode, XReport, XStreamBody};
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::prompt_walk::{SplitFacts, StreamFacts};
use bloomery_gpu_deepseek41::router::sigmoid_128_4::RouterKernels;
use model::arch::minimaxm3::place;

use crate::body::Body;

/// The least count a pick admits from on a load whose lane never probed: the V4.1 walk's floor (its
/// flow model's threshold), which no M3.1 record has priced for its own routing, held as a share of
/// the unit's rows. A pick compares one unit's counts times the walk's units with it, so at the
/// default group of two units, 32 a 128-row unit admits an expert one row in eight of the unit routes
/// to, and a wider [`place::PROMPT_ROWS`] raises the count with the rows rather than admitting colder
/// experts.
pub const STREAM_FLOOR: u32 = 32 * (place::PROMPT_ROWS / 128) as u32;
const _: () = assert!(place::PROMPT_ROWS.is_multiple_of(128));

/// What the family hands the walk ([`StreamFacts`]).
const FACTS: StreamFacts = StreamFacts {
    floor: STREAM_FLOOR,
    tail_us_per_card_pick: 0.0,
    split: Some(SplitFacts { deadline: DEADLINE }),
};

impl Body {
    /// What the next prompt calls move (`BLOOMERY_XSTREAM`): the walk's
    /// [`bloomery_gpu_deepseek41::prompt_walk::BatchBufs::set_xstream`] with the family's facts,
    /// `tiered` whether the load hangs an expert tier. Refused by name as there: `admit` or `split`
    /// with no residency machine, `split` on a load with an expert tier or when the stream cannot
    /// start (the card's room for the ring among them), and any change while a call streams.
    pub fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        let hybrid = self.shell.hybrid_mut();
        let tiered = !hybrid.tiers().is_empty();
        self.units
            .set_xstream::<RouterKernels>(gpu, (hybrid, &mut self.ffn), (mode, tiered), FACTS)
    }

    /// The load line's `xstream=` word ([`bloomery_gpu::host::prompt_stream::PromptStream::word`]).
    #[must_use]
    pub fn xstream_word(&self) -> String {
        self.units.stream().word(self.shell.hybrid())
    }

    /// What the next prompt calls move ([`Body::set_xstream`]).
    #[must_use]
    pub fn xstream_mode(&self) -> XMode {
        self.units.stream().mode()
    }

    /// The expert stream's rule costs in place of the priced ones
    /// ([`bloomery_gpu::host::prompt_stream::PromptStream::set_costs`]), for a gate that drives the
    /// rule to stream. Refused by name before `split` started the stream and while a call streams.
    pub fn set_xstream_costs(&mut self, costs: Costs) -> Result<(), GpuError> {
        let hybrid = self.shell.hybrid_mut();
        self.units.stream_mut().set_costs(hybrid, costs)
    }

    /// The last streaming call's expert stream records, each layer's of each unit and the call's
    /// ([`bloomery_gpu::host::prompt_stream::PromptStream::take_xstream_records`]).
    pub fn take_xstream_records(&mut self) -> (Vec<(usize, XLayer)>, Option<XReport>) {
        self.units.stream_mut().take_xstream_records()
    }
}

impl XStreamBody for Body {
    /// [`Body::set_xstream`].
    fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError> {
        Body::set_xstream(self, gpu, mode)
    }

    /// [`Body::xstream_word`].
    fn xstream_word(&self) -> String {
        Body::xstream_word(self)
    }
}
