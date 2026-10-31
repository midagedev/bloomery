//! Structured output: a request's schema, JSON object or forced tool call as a
//! segment program, and the step arm that follows it, one masked pick per
//! token.
//!
//! - [`schema`]: the request's fields (`response_format` `json_schema` or
//!   `json_object`, `tool_choice` `required` or named) to a program as text.
//!   The JSON-schema subset is an object (the properties `required` names
//!   written in the schema's order, then the others in order, each of which
//!   may be left out — llama.cpp's object rule), arrays with `items`,
//!   `anyOf`, `$ref` into the schema, string, integer and boolean enums,
//!   `const`, booleans, null, integers, numbers and free strings with an
//!   optional `maxLength`; anything else is a 400 naming the keyword.
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
//! `call_parts` beside it, and none of this module knows a model. It names
//! the call opener apart from the rest, so one function serves a forced
//! call (the program writes the opener too) and a `tool_choice: auto`
//! request's free text, which is watched for the opener: the generation is
//! free until the model opens a call, the program runs from the opener to
//! the call's close, and free text follows again, each further opener
//! starting the program again. Muse Spark's markup names no such opener (its
//! recipient slot names the answer's channels as well as a tool), so its
//! replies stay free under auto.
//!
//! The generation loop ([`crate::genloop`]) asks the walk for each token
//! through the same logits-row path a banned id takes, and feeds the chosen
//! id like any other: the step arm. A greedy request's pick is the largest
//! allowed logit, so its answer is plain greedy decoding wherever greedy
//! already obeys the program. A request that samples or penalizes draws with
//! its own sampler from the row with every id the program does not allow
//! masked out ([`Draw`]).
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
pub(crate) use schema::{
    Arg, Ask, Call, CallMode, Fun, any_start, call, is_text, object_parameters, outside, request,
    value,
};
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
    /// The request is valid and names a keyword or shape the compiler does
    /// not serve (a schema keyword outside the subset, text the vocabulary
    /// cannot spell). A forced choice refuses it all the same; a watched
    /// call serves it with no watch.
    pub outside_subset: bool,
    /// The path of the request field refused (`response_format.json_schema.schema.items`),
    /// when the refusal is about one.
    pub field: Option<String>,
}

impl Refusal {
    pub(crate) fn invalid(message: String) -> Refusal {
        Refusal {
            message,
            unsupported: false,
            outside_subset: false,
            field: None,
        }
    }

    /// A valid request the compiler does not serve ([`Refusal::outside_subset`]).
    pub(crate) fn outside_subset(message: String) -> Refusal {
        Refusal {
            outside_subset: true,
            ..Refusal::invalid(message)
        }
    }
}

#[cfg(test)]
mod tests;
