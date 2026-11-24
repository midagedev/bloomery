//! ERNIE 5.5 Flash Preview's chat protocol on the prompt side: the template
//! the seat serves ([`TEMPLATE`]), the tags of its think span ([`THINK`]) and
//! the markers the protocol writes, which its tool-call scan
//! ([`crate::erniexml`]) reads.
//!
//! The checkpoint folder ships no chat template: its renderer is Python, and
//! [`TEMPLATE`] writes the text that renderer writes. A seat serves it as
//! every template: `chat_template` in the server's config, and
//! [`crate::Server::with_think_tags`] with [`THINK`]. The request's thinking
//! knobs reach it as the Python renderer's keyword arguments do: the OpenAI
//! `reasoning_effort` (`low`, `medium`, `high`; `"none"` as
//! `enable_thinking: false`), and `chat_template_kwargs` carrying `options`
//! (`thinking_mode`, `parallel_tool_calls`), `system_settings`,
//! `agent_storage`, `examples` and `sort_keys`.
//!
//! The protocol's markers are added tokens of the folder's vocabulary:
//! `<|ERNIE_MASK|>`, `<|ernie:im_start|>` and `<|ernie:im_end|>` are special
//! tokens, and `<ernie:tool_call>`, `</ernie:tool_call>`, `<ernie:think>` and
//! `</ernie:think>` are user-defined ones. The server reads a marker as
//! markup only where its token wrote it ([`crate::reasoning::SpecialTags`]),
//! and keeps a marker's spelling in a client's text as text
//! ([`crate::guard`]). The renderer opens the prompt with `<|ERNIE_MASK|>`,
//! the token the folder's `tokenizer.json` prepends to a bare text and the
//! vocabulary's exported BOS: a chat prompt is the rendered text as it is, so
//! it holds that one mask, and only a raw text prompt gets the BOS added
//! ([`crate::engine::Tokenizer::add_bos`]); a message ends on
//! `<|ernie:im_end|>` or on `<|ERNIE_MASK|>`, the two ids of
//! `generation_config.json`'s `eos_token_id`, which the folder's vocabulary
//! keys name as its stops ([`crate::engine::Tokenizer::stops`]).

use crate::reasoning::ThinkTags;

/// The checkpoint's chat protocol as a template: the text of its Python
/// renderer (`encoding_ernie.py`, `render_chat`) for a request, turn by turn.
pub const TEMPLATE: &str = include_str!("templates/ernie55.jinja");

/// `<|ERNIE_MASK|>`: opens the prompt; a message also ends on it.
pub const MASK: &str = "<|ERNIE_MASK|>";

/// `<|ernie:im_start|>`: opens a turn.
pub const START: &str = "<|ernie:im_start|>";

/// `<|ernie:im_end|>`: ends a turn, and a message.
pub const END: &str = "<|ernie:im_end|>";

/// `<ernie:think>`: opens the think span, which the generation prompt leaves
/// open unless thinking is off.
pub const THINK_OPEN: &str = "<ernie:think>";

/// `</ernie:think>`: closes the think span.
pub const THINK_CLOSE: &str = "</ernie:think>";

/// `<ernie:tool_call>`: opens the block that holds a message's calls.
pub const CALL_OPEN: &str = "<ernie:tool_call>";

/// `</ernie:tool_call>`: closes the block that holds a message's calls.
pub const CALL_CLOSE: &str = "</ernie:tool_call>";

/// The tags of the think span: the generation prompt ends with the open tag
/// when thinking is on, and with the closed pair when it is off.
pub const THINK: ThinkTags = ThinkTags {
    open: THINK_OPEN,
    close: THINK_CLOSE,
    response: None,
};
