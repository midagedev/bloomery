//! One image's place in the prompt: the reference `image_token_types` and the ids beside them.
//!
//! The span is `[START] + ([IMAGE] * n_llm_w + [NEW_LINE]) * n_llm_h + [END]`, and every position of
//! it carries the same input id (the image placeholder token); only the type tells the positions
//! apart. The IMAGE positions take the aligner rows in reading order, the other three kinds take
//! the learned delimiter rows.

use crate::grid::GridPlan;

/// The type of one position in an image span, with the reference's values (`TEXT = -1` outside
/// any span).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanType {
    Start,
    Image,
    NewLine,
    End,
}

impl SpanType {
    /// The value `image_processor.py` gives this type (`IMAGE_START, IMAGE, IMAGE_NEW_LINE,
    /// IMAGE_END = range(4)`).
    #[must_use]
    pub fn code(self) -> i32 {
        match self {
            SpanType::Start => 0,
            SpanType::Image => 1,
            SpanType::NewLine => 2,
            SpanType::End => 3,
        }
    }
}

/// An image's span: one input id and one type per position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSpan {
    pub ids: Vec<u32>,
    pub types: Vec<SpanType>,
}

/// The span of an image with this plan, every position carrying `image_token_id`.
#[must_use]
pub fn image_span(plan: &GridPlan, image_token_id: u32) -> ImageSpan {
    let mut types = Vec::with_capacity(plan.n_tokens());
    types.push(SpanType::Start);
    for _ in 0..plan.n_llm_h {
        types.extend(std::iter::repeat_n(SpanType::Image, plan.n_llm_w));
        types.push(SpanType::NewLine);
    }
    types.push(SpanType::End);
    ImageSpan {
        ids: vec![image_token_id; types.len()],
        types,
    }
}
