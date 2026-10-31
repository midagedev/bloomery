//! Structured output: a request's schema, JSON object or forced tool call as a
//! segment program, and the step arm that follows it, one masked pick per
//! token.
//!
//! - [`schema`]: the request's fields (`response_format` `json_schema` or
//!   `json_object`, `tool_choice` `required` or named) to a program as text.
//!   The JSON-schema subset is an object (every property written, in the
//!   schema's order), arrays with `items`, `anyOf`, `$ref` into the schema,
//!   string, integer and boolean enums, `const`, booleans, null, integers,
//!   numbers and free strings with an optional `maxLength`; anything else is a
//!   400 naming the keyword.
//! - [`program`]: the program as ids, built with the serve `Tokenizer`, and
//!   the [`Walk`] over it that picks every token. A forced text is one id
//!   after another, a choice is a trie, a free string or number takes any
//!   token of its class and ends on a token that opens the text after it, and
//!   a fork's options are followed side by side until the ids part them.
//! - [`vocab`]: the vocabulary read once at bind, which the free-string
//!   classes come from.
//!
//! The tool-call text comes from the chat format's own module (`hermes`,
//! `qwenxml`, `glmxml`, `dsml`): each has an inverse of its parser, a
//! `call_parts` beside it, and none of this module knows a model.
//!
//! The generation loop ([`crate::genloop`]) asks the walk for each token
//! through the same logits-row path a banned id takes, and feeds the chosen
//! id like any other: the step arm. A greedy request's pick is the largest
//! allowed logit, so its answer is plain greedy decoding wherever greedy
//! already obeys the program. A request that samples or penalizes draws with
//! its own sampler from the row with every id the program does not allow
//! masked out ([`Draw`]). On an engine that drafts, the same pick takes
//! each kept id of a pass from its verified row; a pass stops after the
//! first pick that is not its draft's, so the walk moves through kept ids
//! alone.
//!
//! The jump arm feeds a run of ids the program fixes ([`Walk::forced_run`])
//! as one prompt call instead of one step each, and reads the next decision
//! from the row of the step that closes the run. It takes the same picks as
//! the step arm from the same rows: nothing is taken back.

mod program;
mod schema;
mod vocab;

#[cfg(test)]
pub(crate) use program::Kid;
pub(crate) use program::{Constraint, Decision, Draw, Part, StepError, Style, Walk, Watch};
pub(crate) use schema::{Ask, Fun, call, is_text, refuse, request, value};
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
    /// The path of the request field refused (`response_format.json_schema.schema.items`),
    /// when the refusal is about one.
    pub field: Option<String>,
}

impl Refusal {
    pub(crate) fn invalid(message: String) -> Refusal {
        Refusal {
            message,
            unsupported: false,
            field: None,
        }
    }
}

#[cfg(test)]
mod tests;
