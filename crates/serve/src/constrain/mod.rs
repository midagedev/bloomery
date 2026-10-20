//! Structured output: a request's schema or forced tool call as a segment
//! program, and the step arm that follows it, one masked argmax per token.
//!
//! - [`schema`]: the request's fields (`response_format` `json_schema`,
//!   `tool_choice` `required` or named) to a program as text. The JSON-schema
//!   subset is an object (every property written, in the schema's order),
//!   string, integer and boolean enums, `const`, booleans and free strings
//!   with an optional `maxLength`; anything else is a 400 naming the keyword.
//! - [`program`]: the program as ids, built with the serve `Tokenizer`, and
//!   the [`Walk`] over it that picks every token. A forced text is one id
//!   after another, a choice is a trie, a free string takes any token of its
//!   class and ends on a token that opens the text after it.
//! - [`vocab`]: the vocabulary read once at bind, which the free-string
//!   classes come from.
//!
//! The tool-call text comes from the chat format's own module (`hermes`,
//! `qwenxml`, `glmxml`, `dsml`): each has an inverse of its parser, a
//! `call_parts` beside it, and none of this module knows a model.
//!
//! The generation loop ([`crate::genloop`]) asks the walk for each token
//! through the same logits-row path a banned id takes, and feeds the chosen
//! id like any other: the step arm. A constrained request decodes greedily:
//! the pick is the largest allowed logit, so its answer is plain greedy
//! decoding wherever greedy already obeys the program. A request that asks
//! for sampling is refused by name.
//!
//! The jump arm feeds a run of ids the program fixes ([`Walk::forced_run`])
//! as one prompt call instead of one step each, and reads the next decision
//! from the row of the step that closes the run. It takes the same picks as
//! the step arm from the same rows: nothing is taken back.

mod program;
mod schema;
mod vocab;

pub(crate) use program::{Constraint, Decision, Part, StepError, Style, Walk, Watch};
pub(crate) use schema::{Ask, Fun, call, is_text, request, value};
pub(crate) use vocab::Vocab;

/// A request the constrained path will not take, named. The route answers it
/// as a 400, or as a 501 when the chat template's tool-call markup is one no
/// inverse here spells.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub(crate) struct Refusal {
    pub message: String,
    /// The markup is unsupported, not the request invalid.
    pub unsupported: bool,
}

impl Refusal {
    pub(crate) fn invalid(message: String) -> Refusal {
        Refusal {
            message,
            unsupported: false,
        }
    }
}

#[cfg(test)]
mod tests;
