//! OpenAI's Responses API on the chat path: `POST /v1/responses` and
//! `POST /responses` generate, `POST /v1/responses/input_tokens` and
//! `POST /responses/input_tokens` count, as llama-server serves them
//! (`server_chat_convert_responses_to_chatcmpl`, `to_json_oaicompat_resp`,
//! `to_json_oaicompat_resp_stream`, `handle_count_tokens`).
//!
//! A request converts into the OpenAI chat body `/v1/chat/completions` reads
//! and runs the chat path's own steps on it ([`chat_plan`]): the sampling
//! fields, the reasoning format, the chat template, the tool-call scan, the
//! think-span budget, the generation and the output parser. Only the two shapes
//! are this module's: the request's conversion ([`to_chat`]) and the answer's.
//!
//! # The request
//!
//! As llama-server converts it, the chat body is the request's own body with
//! five fields converted, so every chat field the request carries reaches the
//! chat path as it is (`temperature`, `top_p`, `stream`, `tool_choice`, `stop`,
//! `seed`, `top_k`, `timings_per_token`, `return_progress`, `cache_prompt`,
//! `chat_template_kwargs`, `reasoning_format`, …), and the chat path's own
//! refusals answer them (`tool_choice` other than `auto` or `none`, a
//! `response_format` other than text, …):
//! - `instructions` (a string): a `system` message before the input's.
//! - `input`: a string is one `user` message. A list holds items, each by its
//!   `type` (`message` when absent):
//!   - a `message` of role `user`, `system` or `developer`: that role's
//!     message, its content a string or `input_text` and `input_image` parts,
//!     as the chat path's `text` and `image_url` parts; the role is kept, as
//!     llama-server's conversion keeps it;
//!   - a `message` of role `assistant`: its `output_text` (or `input_text`)
//!     parts as the assistant's text parts;
//!   - `function_call`: a `tool_calls` entry, its `call_id` the call's id and
//!     its `arguments` the call's arguments text;
//!   - `function_call_output`: a `tool` message for its `call_id`, its
//!     `output` a string or `input_text` and `input_image` parts;
//!   - `reasoning`: the assistant's `reasoning_content`, the text of its
//!     `content` parts (`summary` must be a list, and is not read).
//!
//!   An assistant item (a message, a call or a reasoning) joins the message
//!   before it when that one is the assistant's, so one turn's reasoning, text
//!   and calls are one assistant message, as llama-server merges them. Every
//!   message an item converts to has its content as parts; an assistant
//!   message of calls or reasoning alone has none.
//! - `tools`: each `function` tool as the chat path's function tool, its
//!   fields but `type` the function's and `strict` `true` when absent. A
//!   `web_search` tool (`web_search_preview` and the like) is dropped, as
//!   llama-server drops every tool that is not a function's: it runs on the
//!   provider's side, and Codex sends one by default.
//! - `max_output_tokens`: `max_tokens`.
//! - `reasoning`: its `effort` (a string) as `reasoning_effort`, which the
//!   template reads. Its `summary` asks OpenAI's models for a summary of their
//!   reasoning; this server answers with the reasoning itself, as llama-server
//!   does, so the field is read as part of the protocol and ignored.
//!
//! Ignored, as bookkeeping or as a request this answer already meets:
//! `model` (echoed), `store: false`, `include` (but its logprobs),
//! `metadata`, `user`, `safety_identifier`, `prompt_cache_key`,
//! `prompt_cache_retention`, `service_tier`, `client_metadata`,
//! `stream_options`, `max_tool_calls` (it counts built-in tools, which this
//! server has none of), `text.verbosity` (a length hint to OpenAI's models),
//! `parallel_tool_calls` (the chat path ignores it too), and `truncation`
//! (`auto` asks to drop input that overflows the context; a prompt that does
//! not fit is the chat path's 400, never a quiet cut).
//!
//! # The answer
//!
//! `{id, object: "response", created_at, completed_at, status, model, output,
//! usage}`. `output` holds a `reasoning` item (the think span as one
//! `reasoning_text` content part, `summary` empty, `encrypted_content` empty),
//! a `message` item (one `output_text` part), then one `function_call` item per
//! parsed call (`call_id`, `name`, `arguments`), each present only when it
//! has something. `usage` is the chat path's: `input_tokens` the whole
//! prompt, `input_tokens_details.cached_tokens` what the prompt cache kept,
//! `output_tokens` the generated ids, `total_tokens` their sum. `status` is
//! `completed`, or `incomplete` with `incomplete_details.reason`
//! `max_output_tokens` when the generation stopped at the request's or the
//! context's end. Ids: the response `resp_<32 hex>`; an item
//! `<rs|msg>_<output index>_<16 hex>`; a call `fc_<n>_<16 hex>`, its `call_id`
//! `call_<n>_<16 hex>`, `n` the call's index and the hex the response's.
//!
//! # The stream
//!
//! Each frame is `event: <type>` and `data: <json>`, every data object with
//! its `sequence_number`, from 0. In order:
//! - `response.created`, then `response.in_progress`, each with the response
//!   in progress (`output` empty), sent with the first event; with
//!   `return_progress` the prompt's `prompt_progress` rides on the
//!   `response.in_progress` that reports the evaluated prompt;
//! - a reasoning item: `response.output_item.added`, its
//!   `response.reasoning_text.delta`s, `response.output_item.done`;
//! - a message item: `response.output_item.added`,
//!   `response.content_part.added`, its `response.output_text.delta`s,
//!   `response.output_text.done`, `response.content_part.done`,
//!   `response.output_item.done`;
//! - a call: `response.output_item.added`,
//!   `response.function_call_arguments.delta` (the call's whole arguments),
//!   `response.output_item.done`;
//! - `response.completed`, or `response.incomplete`, with the whole response
//!   and the generation's `timings`.
//!
//! Each item is done before the next one is added. Item events carry
//! `output_index`, part and text events `item_id` and `content_index`. With
//! `timings_per_token` the last frame of each step carries `timings`. Text the
//! model writes after a call opens another message item; the whole answer joins
//! all its text in one message item, as the chat path's `content` does.
//!
//! # Refusals and errors
//!
//! Refused by name with a 400, where llama-server drops or defaults:
//! `previous_response_id`, `conversation`, `store: true`, `background: true`,
//! a stored `prompt` and an `item_reference` item (this server keeps no
//! responses); `logprobs`, `top_logprobs` above 0 and `include` of
//! `message.output_text.logprobs` (the answer carries no probabilities);
//! `n` above 1; `text.format` other than `text`; `input_file`, an
//! `input_image` by `file_id`, and an assistant's `refusal` part; a tool that
//! is neither a function nor a web search; an item, a part or a role the
//! conversion does not know; and a field of the wrong type. An error is the
//! chat path's OpenAI error object; inside a stream that has started, a
//! `data: {"error": …}` frame with no `event:` line, as llama-server sends it.
//!
//! # Where this differs from llama-server
//!
//! - An unserved field is refused by name (above) where llama-server ignores
//!   it or reads the wrong type as empty.
//! - An assistant message with no `type` is read as a `message`, as OpenAI
//!   reads it; llama-server refuses it.
//! - Each message is built from its role and content alone; llama-server
//!   copies the item's other fields (`id`, `status`) into the chat message.
//! - A reasoning item's text is all its content parts', and it is appended to
//!   the assistant message it joins; llama-server reads the first part and
//!   overwrites.
//! - A generation stopped at its limit is `incomplete` and its stream ends with
//!   `response.incomplete`, as OpenAI reports it; llama-server reports it
//!   `completed`, which hides the cut.
//! - The stream closes each item before the next opens and numbers its
//!   frames (`sequence_number`, `output_index`, `content_index`), and its
//!   `response.created` carries the whole response in progress, as OpenAI's
//!   stream does, so a client that reads those fields finds them; llama-server
//!   sends every item's done events after the last delta, and no index.
//! - A `null` `instructions` is absent, as `null` is everywhere here;
//!   llama-server writes it as an empty system message.
//! - The non-stream answer's item ids are the stream's (one per output index),
//!   not fresh ones.

use std::io;
use std::net::TcpStream;

use serde_json::{Map, Value, json};

use super::{
    ApiError, ChatPlan, State, body, chat_input, chat_plan, error_body, finish_stream, get_i,
    invalid, json_type, progress, run_gen, run_whole, send_error, send_json, sse,
    tool_markup_error, unix_now,
};
use crate::dsml::{Message, ToolCall};
use crate::genloop::{Event, Outcome, StopKind, Timings};
use crate::http::{EventStream, Request};

/// The request's fields the conversion replaces; the rest of the body goes to
/// the chat path as it is.
const CONVERTED: [&str; 5] = [
    "input",
    "instructions",
    "tools",
    "max_output_tokens",
    "reasoning",
];

/// `include`'s value that asks for the output's probabilities.
const INCLUDE_LOGPROBS: &str = "message.output_text.logprobs";

/// `POST /v1/responses` and `POST /responses`.
pub(super) fn create(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let parsed = body(req).and_then(|b| {
        let chat = to_chat(&b)?;
        let plan = chat_plan(state, &chat, chat.get("reasoning_format"))?;
        Ok((Ids::new(state, &b), plan))
    });
    let (ids, plan) = match parsed {
        Ok(x) => x,
        Err(e) => return send_error(w, req, &e),
    };
    if plan.p.stream {
        stream(state, req, w, &ids, plan)
    } else {
        whole(state, req, w, &ids, plan)
    }
}

/// `POST /v1/responses/input_tokens` and `POST /responses/input_tokens`:
/// `{"input_tokens": N, "object": "response.input_tokens"}`, N the ids of the
/// prompt the same request renders, by the template and the tokenizer the
/// generation uses.
pub(super) fn input_tokens(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let counted = body(req)
        .and_then(|b| to_chat(&b))
        .and_then(|chat| chat_input(state, &chat));
    match counted {
        Ok((_, prompt)) => send_json(
            w,
            req,
            200,
            &json!({ "input_tokens": prompt.held.ids.len(), "object": "response.input_tokens" }),
        ),
        Err(e) => send_error(w, req, &e),
    }
}

// ---------------------------------------------------------------- the request

/// A field that is present and not `null`.
fn set<'a>(o: &'a Map<String, Value>, k: &str) -> Option<&'a Value> {
    o.get(k).filter(|v| !v.is_null())
}

/// The string field `k` of `what`.
fn text_of<'a>(o: &'a Map<String, Value>, k: &str, what: &str) -> Result<&'a str, ApiError> {
    o.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{what} needs a string '{k}'")))
}

/// The OpenAI chat body a Responses request converts to, or the first field it
/// refuses, by name.
fn to_chat(b: &Map<String, Value>) -> Result<Map<String, Value>, ApiError> {
    unserved(b)?;
    let mut messages = Vec::new();
    if let Some(v) = set(b, "instructions") {
        let Value::String(s) = v else {
            return Err(invalid(format!(
                "instructions must be a string, not {}",
                json_type(v)
            )));
        };
        messages.push(json!({ "role": "system", "content": s }));
    }
    match set(b, "input") {
        Some(Value::String(s)) => messages.push(json!({ "role": "user", "content": s })),
        Some(Value::Array(items)) => {
            for it in items {
                item(it, &mut messages)?;
            }
        }
        Some(v) => {
            return Err(invalid(format!(
                "input must be a string or an array of items, not {}",
                json_type(v)
            )));
        }
        None => return Err(invalid("input is required")),
    }
    let mut chat: Map<String, Value> = b
        .iter()
        .filter(|(k, _)| !CONVERTED.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    chat.insert("messages".into(), Value::Array(messages));
    if let Some(t) = set(b, "tools") {
        let tools = tools(t)?;
        if !tools.is_empty() {
            chat.insert("tools".into(), Value::Array(tools));
        }
    }
    if let Some(n) = b.get("max_output_tokens") {
        chat.insert("max_tokens".into(), n.clone());
    }
    if let Some(effort) = reasoning_effort(set(b, "reasoning"))? {
        chat.insert("reasoning_effort".into(), effort.clone());
    }
    Ok(chat)
}

/// The fields that ask for what this server does not serve, refused by name.
fn unserved(b: &Map<String, Value>) -> Result<(), ApiError> {
    let refused = [
        (
            "previous_response_id",
            set(b, "previous_response_id").is_some_and(|v| v.as_str() != Some("")),
            "it keeps no responses",
        ),
        (
            "conversation",
            set(b, "conversation").is_some(),
            "it keeps no conversations",
        ),
        (
            "store: true",
            set(b, "store").is_some_and(|v| v.as_bool() != Some(false)),
            "it keeps no responses (send store: false or leave it out)",
        ),
        (
            "background: true",
            set(b, "background").is_some_and(|v| v.as_bool() != Some(false)),
            "it keeps no responses to poll",
        ),
        (
            "prompt",
            set(b, "prompt").is_some(),
            "it keeps no stored prompts",
        ),
        (
            "logprobs",
            set(b, "logprobs").is_some_and(|v| v.as_bool() != Some(false)),
            "the answer carries no probabilities",
        ),
        (
            "top_logprobs",
            get_i(b, "top_logprobs")?.is_some_and(|n| n > 0),
            "the answer carries no probabilities",
        ),
        (
            "n",
            get_i(b, "n")?.is_some_and(|n| n > 1),
            "a response holds one answer",
        ),
    ];
    if let Some((field, _, why)) = refused.iter().find(|(_, hit, _)| *hit) {
        return Err(invalid(format!(
            "{field} is not supported by this server: {why}"
        )));
    }
    match set(b, "include") {
        None => {}
        Some(Value::Array(a)) => {
            for x in a {
                match x.as_str() {
                    Some(INCLUDE_LOGPROBS) => {
                        return Err(invalid(format!(
                            "include {INCLUDE_LOGPROBS} is not supported by this server: the \
                             answer carries no probabilities"
                        )));
                    }
                    Some(_) => {}
                    None => return Err(invalid(format!("include must hold strings, not {x}"))),
                }
            }
        }
        Some(v) => {
            return Err(invalid(format!(
                "include must be an array of strings, not {}",
                json_type(v)
            )));
        }
    }
    match set(b, "text") {
        None => Ok(()),
        Some(Value::Object(t)) => match set(t, "format") {
            None => Ok(()),
            Some(f) => match f.get("type").and_then(Value::as_str) {
                Some("text") => Ok(()),
                Some(kind) => Err(invalid(format!(
                    "text.format of type {kind} is not supported by this server: it answers in \
                     plain text"
                ))),
                None => Err(invalid(format!(
                    "text.format must be an object with a string 'type', not {f}"
                ))),
            },
        },
        Some(v) => Err(invalid(format!(
            "text must be an object, not {}",
            json_type(v)
        ))),
    }
}

/// `reasoning.effort`, a string, which the chat path hands the template as
/// `reasoning_effort`.
fn reasoning_effort(v: Option<&Value>) -> Result<Option<&Value>, ApiError> {
    match v {
        None => Ok(None),
        Some(Value::Object(r)) => match set(r, "effort") {
            None => Ok(None),
            Some(e @ Value::String(_)) => Ok(Some(e)),
            Some(e) => Err(invalid(format!(
                "reasoning.effort must be a string, not {}",
                json_type(e)
            ))),
        },
        Some(v) => Err(invalid(format!(
            "reasoning must be an object, not {}",
            json_type(v)
        ))),
    }
}

/// Appends one input item as the chat path's messages.
fn item(it: &Value, out: &mut Vec<Value>) -> Result<(), ApiError> {
    let Value::Object(it) = it else {
        return Err(invalid(format!(
            "each input item must be an object, not {}",
            json_type(it)
        )));
    };
    let kind = match set(it, "type") {
        None => "message",
        Some(Value::String(s)) => s.as_str(),
        Some(v) => {
            return Err(invalid(format!(
                "an input item's type must be a string, not {}",
                json_type(v)
            )));
        }
    };
    match kind {
        "message" => message(it, out),
        "function_call" => function_call(it, out),
        "function_call_output" => function_call_output(it, out),
        "reasoning" => reasoning(it, out),
        "item_reference" => Err(invalid(
            "input item type item_reference is not supported by this server: it keeps no items \
             to refer to",
        )),
        other => Err(invalid(format!(
            "input item type \"{other}\" is not supported (message, function_call, \
             function_call_output, reasoning)"
        ))),
    }
}

/// The assistant message an assistant item joins: the last message when it is
/// the assistant's, else a new one with no content.
fn assistant(out: &mut Vec<Value>) -> &mut Map<String, Value> {
    if !out.last().is_some_and(|m| m["role"] == "assistant") {
        out.push(json!({ "role": "assistant", "content": [] }));
    }
    out.last_mut()
        .and_then(Value::as_object_mut)
        .expect("every message this conversion writes is an object")
}

/// A message's `content` as the chat path's parts: a string as one text part
/// (llama-server reads it as one `input_text` part), an array part by part
/// through `part`, absent as none.
fn content_parts(
    m: &Map<String, Value>,
    part: impl Fn(&Value) -> Result<Value, ApiError>,
) -> Result<Vec<Value>, ApiError> {
    match set(m, "content") {
        None => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![json!({ "type": "text", "text": s })]),
        Some(Value::Array(parts)) => parts.iter().map(part).collect(),
        Some(v) => Err(invalid(format!(
            "a message's content must be a string or an array of content parts, not {}",
            json_type(v)
        ))),
    }
}

/// A content part's `type`.
fn part_type(p: &Value) -> Result<&str, ApiError> {
    p.get("type").and_then(Value::as_str).ok_or_else(|| {
        invalid(format!(
            "a content part must be an object with a string 'type', not {p}"
        ))
    })
}

/// An `input_text` part as the chat path's text part.
fn text_part(p: &Value, what: &str) -> Result<Value, ApiError> {
    let text = p
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{what} part needs a string 'text'")))?;
    Ok(json!({ "type": "text", "text": text }))
}

/// An `input_image` part as the chat path's `image_url` part: its `image_url`
/// (a URL or a `data:` URL) is the image. A `file_id` names an uploaded file,
/// which this server does not keep.
fn image_part(p: &Value) -> Result<Value, ApiError> {
    match p.get("image_url").filter(|v| !v.is_null()) {
        Some(Value::String(url)) => Ok(json!({ "type": "image_url", "image_url": { "url": url } })),
        Some(v) => Err(invalid(format!(
            "an input_image's image_url must be a string, not {}",
            json_type(v)
        ))),
        None if p.get("file_id").is_some_and(|v| !v.is_null()) => Err(invalid(
            "an input_image by file_id is not supported by this server: it keeps no files (send \
             its image_url)",
        )),
        None => Err(invalid("an input_image needs a string 'image_url'")),
    }
}

/// An `input_text` or `input_image` part of a user, system or developer
/// message or of a call's output, as the chat path's part.
fn input_part(p: &Value, what: &str) -> Result<Value, ApiError> {
    match part_type(p)? {
        "input_text" => text_part(p, "an input_text"),
        "input_image" => image_part(p),
        "input_file" => Err(invalid(
            "input_file is not supported by this server: it reads text and images",
        )),
        other => Err(invalid(format!(
            "content part type \"{other}\" is not supported in {what} (input_text, input_image)"
        ))),
    }
}

/// A `message` item: a user, system or developer message of its own, or an
/// assistant's text joining the assistant message before it.
fn message(it: &Map<String, Value>, out: &mut Vec<Value>) -> Result<(), ApiError> {
    let role = text_of(it, "role", "an input message")?;
    match role {
        "user" | "system" | "developer" => {
            if set(it, "content").is_none() {
                return Err(invalid(format!("a {role} message needs its content")));
            }
            let what = format!("a {role} message");
            let parts = content_parts(it, |p| input_part(p, &what))?;
            out.push(json!({ "role": role, "content": parts }));
            Ok(())
        }
        "assistant" => {
            let parts = content_parts(it, |p| match part_type(p)? {
                "output_text" | "input_text" => text_part(p, "an output_text"),
                "refusal" => Err(invalid(
                    "an assistant message's refusal part is not supported by this server: the \
                     chat path takes its text parts",
                )),
                other => Err(invalid(format!(
                    "content part type \"{other}\" is not supported in an assistant message \
                     (output_text)"
                ))),
            })?;
            if let Some(Value::Array(content)) = assistant(out).get_mut("content") {
                content.extend(parts);
            }
            Ok(())
        }
        other => Err(invalid(format!(
            "an input message's role must be user, system, developer or assistant, not \"{other}\""
        ))),
    }
}

/// A `function_call` item as a call of the assistant message it joins.
fn function_call(it: &Map<String, Value>, out: &mut Vec<Value>) -> Result<(), ApiError> {
    let what = "a function_call item";
    let call = json!({
        "id": text_of(it, "call_id", what)?,
        "type": "function",
        "function": {
            "name": text_of(it, "name", what)?,
            "arguments": text_of(it, "arguments", what)?,
        },
    });
    let message = assistant(out);
    match message.get_mut("tool_calls") {
        Some(Value::Array(calls)) => calls.push(call),
        _ => {
            message.insert("tool_calls".into(), json!([call]));
        }
    }
    Ok(())
}

/// A `function_call_output` item as a `tool` message: its output a string, or
/// `input_text` and `input_image` parts.
fn function_call_output(it: &Map<String, Value>, out: &mut Vec<Value>) -> Result<(), ApiError> {
    let call_id = text_of(it, "call_id", "a function_call_output item")?;
    let content = match set(it, "output") {
        Some(Value::String(s)) => json!(s),
        Some(Value::Array(parts)) => Value::Array(
            parts
                .iter()
                .map(|p| input_part(p, "a function_call_output"))
                .collect::<Result<_, _>>()?,
        ),
        Some(v) => {
            return Err(invalid(format!(
                "a function_call_output's output must be a string or an array of parts, not {}",
                json_type(v)
            )));
        }
        None => {
            return Err(invalid(
                "a function_call_output item needs its output, a string or an array of parts",
            ));
        }
    };
    out.push(json!({ "role": "tool", "tool_call_id": call_id, "content": content }));
    Ok(())
}

/// A `reasoning` item as the `reasoning_content` of the assistant message it
/// joins: its content parts' text, in order. The reasoning is read back from
/// that text; an item without it (OpenAI's encrypted reasoning) is refused.
fn reasoning(it: &Map<String, Value>, out: &mut Vec<Value>) -> Result<(), ApiError> {
    if !matches!(it.get("summary"), Some(Value::Array(_))) {
        return Err(invalid("a reasoning item needs a 'summary' array"));
    }
    let parts = match set(it, "content") {
        Some(Value::Array(parts)) if !parts.is_empty() => parts,
        _ => {
            return Err(invalid(
                "a reasoning item needs a non-empty 'content' array of reasoning_text parts: this \
                 server reads the reasoning from its text",
            ));
        }
    };
    let mut text = String::new();
    for p in parts {
        match part_type(p)? {
            "reasoning_text" | "text" => {
                text.push_str(
                    p.get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| invalid("a reasoning_text part needs a string 'text'"))?,
                );
            }
            other => {
                return Err(invalid(format!(
                    "content part type \"{other}\" is not supported in a reasoning item \
                     (reasoning_text)"
                )));
            }
        }
    }
    let message = assistant(out);
    match message.get_mut("reasoning_content") {
        Some(Value::String(r)) => r.push_str(&text),
        _ => {
            message.insert("reasoning_content".into(), json!(text));
        }
    }
    Ok(())
}

/// `tools` as the chat path's function tools; a web search is dropped.
fn tools(v: &Value) -> Result<Vec<Value>, ApiError> {
    let Value::Array(tools) = v else {
        return Err(invalid(format!(
            "tools must be an array, not {}",
            json_type(v)
        )));
    };
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        let Value::Object(t) = t else {
            return Err(invalid("each tool must be an object"));
        };
        match t.get("type").and_then(Value::as_str) {
            Some("function") => out.push(function_tool(t)?),
            Some(kind) if kind.starts_with("web_search") => {}
            Some(kind) => {
                return Err(invalid(format!(
                    "tool type {kind} is not supported: this server runs function tools only"
                )));
            }
            None => return Err(invalid("each tool needs a string 'type'")),
        }
    }
    Ok(out)
}

/// One function tool as the chat path's: its fields but `type` the function's,
/// `strict` `true` when absent, as llama-server converts it. Its `name` is a
/// non-empty string, its `description` a string or absent, its `parameters` an
/// object or absent.
fn function_tool(t: &Map<String, Value>) -> Result<Value, ApiError> {
    let name = match t.get("name") {
        Some(Value::String(s)) if !s.is_empty() => s,
        _ => {
            return Err(invalid(
                "each function tool needs a non-empty string 'name'",
            ));
        }
    };
    if let Some(v) = set(t, "description").filter(|v| !v.is_string()) {
        return Err(invalid(format!(
            "tool {name}: 'description' must be a string, not {}",
            json_type(v)
        )));
    }
    if let Some(v) = set(t, "parameters").filter(|v| !v.is_object()) {
        return Err(invalid(format!(
            "tool {name}: 'parameters' must be an object, not {}",
            json_type(v)
        )));
    }
    let mut function = t.clone();
    function.shift_remove("type");
    if !function.contains_key("strict") {
        function.insert("strict".into(), Value::Bool(true));
    }
    Ok(json!({ "type": "function", "function": function }))
}

// ---------------------------------------------------------------- the answer

/// The answer's names: the response's id, its creation time and the request's
/// `model` (the server's alias when it names none), as the chat path echoes it.
struct Ids {
    /// `resp_<32 hex>`.
    id: String,
    created_at: u64,
    model: String,
}

impl Ids {
    fn new(state: &State, b: &Map<String, Value>) -> Self {
        Ids {
            id: format!("resp_{}", state.random_id()),
            created_at: unix_now(),
            model: b
                .get("model")
                .and_then(Value::as_str)
                .map_or_else(|| state.alias.clone(), str::to_owned),
        }
    }

    /// `<prefix>_<n>_<16 hex>`: an item's id, `n` its output index (or a
    /// call's index), the hex the response's first 16.
    fn item(&self, prefix: &str, n: usize) -> String {
        let nonce = self.id.strip_prefix("resp_").unwrap_or(&self.id);
        format!("{prefix}_{n}_{}", nonce.get(..16).unwrap_or(nonce))
    }
}

/// A finished reasoning item.
fn reasoning_item(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "summary": [],
        "content": [{ "type": "reasoning_text", "text": text }],
        "encrypted_content": "",
        "status": "completed",
    })
}

/// A message's `output_text` part.
fn output_text(text: &str) -> Value {
    json!({ "type": "output_text", "annotations": [], "logprobs": [], "text": text })
}

/// A finished message item.
fn message_item(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [output_text(text)],
    })
}

/// A call's item, `arguments` and `status` as the stream reports it.
fn call_item(ids: &Ids, c: &ToolCall, arguments: &str, status: &str) -> Value {
    json!({
        "id": ids.item("fc", c.index),
        "type": "function_call",
        "status": status,
        "call_id": ids.item("call", c.index),
        "name": c.name,
        "arguments": arguments,
    })
}

/// The open item's kind.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Reasoning,
    Message,
}

/// The answer's output items, in order, built from the output parser's steps:
/// the stream's events and the answer's `output` come from the same items. An
/// item is done before the next one is added.
struct Items<'a> {
    ids: &'a Ids,
    /// The done items, in output order; the open one's index is their count.
    done: Vec<Value>,
    /// The open item and its text so far.
    open: Option<(Kind, String)>,
}

impl<'a> Items<'a> {
    fn new(ids: &'a Ids) -> Self {
        Items {
            ids,
            done: Vec::new(),
            open: None,
        }
    }

    /// The open item's id.
    fn open_id(&self, kind: Kind) -> String {
        let prefix = match kind {
            Kind::Reasoning => "rs",
            Kind::Message => "msg",
        };
        self.ids.item(prefix, self.done.len())
    }

    /// The events one step of the output parser adds, in the answer's order:
    /// reasoning, text, then each call whole.
    fn push(&mut self, d: &Message, ev: &mut Vec<Value>) {
        if !d.reasoning.is_empty() {
            self.text(Kind::Reasoning, &d.reasoning, ev);
        }
        if !d.content.is_empty() {
            self.text(Kind::Message, &d.content, ev);
        }
        for c in &d.calls {
            self.call(c, ev);
        }
    }

    /// A delta of the open `kind` item, which is added first when another
    /// item (or none) is open.
    fn text(&mut self, kind: Kind, delta: &str, ev: &mut Vec<Value>) {
        if self.open.as_ref().map(|(k, _)| *k) != Some(kind) {
            self.close(ev);
            self.add(kind, ev);
        }
        let (index, id) = (self.done.len(), self.open_id(kind));
        ev.push(match kind {
            Kind::Reasoning => json!({
                "type": "response.reasoning_text.delta",
                "item_id": id, "output_index": index, "content_index": 0, "delta": delta,
            }),
            Kind::Message => json!({
                "type": "response.output_text.delta",
                "item_id": id, "output_index": index, "content_index": 0, "delta": delta,
            }),
        });
        if let Some((_, text)) = &mut self.open {
            text.push_str(delta);
        }
    }

    /// Adds a `kind` item, empty and in progress.
    fn add(&mut self, kind: Kind, ev: &mut Vec<Value>) {
        let (index, id) = (self.done.len(), self.open_id(kind));
        match kind {
            Kind::Reasoning => ev.push(json!({
                "type": "response.output_item.added",
                "output_index": index,
                "item": {
                    "id": id, "type": "reasoning", "summary": [], "content": [],
                    "encrypted_content": "", "status": "in_progress",
                },
            })),
            Kind::Message => {
                ev.push(json!({
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {
                        "id": id, "type": "message", "role": "assistant", "status": "in_progress",
                        "content": [],
                    },
                }));
                ev.push(json!({
                    "type": "response.content_part.added",
                    "item_id": id, "output_index": index, "content_index": 0,
                    "part": { "type": "output_text", "text": "" },
                }));
            }
        }
        self.open = Some((kind, String::new()));
    }

    /// One parsed call: added, its whole arguments, done.
    fn call(&mut self, c: &ToolCall, ev: &mut Vec<Value>) {
        self.close(ev);
        let index = self.done.len();
        ev.push(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": call_item(self.ids, c, "", "in_progress"),
        }));
        ev.push(json!({
            "type": "response.function_call_arguments.delta",
            "item_id": self.ids.item("fc", c.index), "output_index": index, "delta": c.arguments,
        }));
        let item = call_item(self.ids, c, &c.arguments, "completed");
        ev.push(
            json!({ "type": "response.output_item.done", "output_index": index, "item": item }),
        );
        self.done.push(item);
    }

    /// Finishes the open item, if any.
    fn close(&mut self, ev: &mut Vec<Value>) {
        let Some((kind, text)) = self.open.take() else {
            return;
        };
        let (index, id) = (self.done.len(), self.open_id(kind));
        let item = match kind {
            Kind::Reasoning => reasoning_item(&id, &text),
            Kind::Message => {
                ev.push(json!({
                    "type": "response.output_text.done",
                    "item_id": id, "output_index": index, "content_index": 0, "text": text,
                }));
                ev.push(json!({
                    "type": "response.content_part.done",
                    "item_id": id, "output_index": index, "content_index": 0,
                    "part": output_text(&text),
                }));
                message_item(&id, &text)
            }
        };
        ev.push(
            json!({ "type": "response.output_item.done", "output_index": index, "item": item }),
        );
        self.done.push(item);
    }

    /// The answer's `output`, the open item finished.
    fn finish(mut self, ev: &mut Vec<Value>) -> Vec<Value> {
        self.close(ev);
        self.done
    }
}

/// `usage`: the whole prompt, what the prompt cache kept of it, and the
/// generated ids, as the chat path counts them.
fn usage(t: &Timings) -> Value {
    json!({
        "input_tokens": t.n_prompt,
        "input_tokens_details": { "cached_tokens": t.cache_n },
        "output_tokens": t.predicted_n,
        "total_tokens": t.n_prompt + t.predicted_n,
    })
}

/// `incomplete` when the generation stopped at its limit, else `completed`.
fn status(o: &Outcome) -> &'static str {
    match o.stop {
        StopKind::Limit => "incomplete",
        StopKind::Eos | StopKind::Word => "completed",
    }
}

/// The response in progress: `response.created`'s and
/// `response.in_progress`'s.
fn in_progress(ids: &Ids) -> Value {
    json!({
        "id": ids.id,
        "object": "response",
        "created_at": ids.created_at,
        "status": "in_progress",
        "model": ids.model,
        "output": [],
    })
}

/// The finished response: llama-server's `to_json_oaicompat_resp`, its
/// `status` the generation's.
fn finished(ids: &Ids, o: &Outcome, output: Vec<Value>) -> Value {
    let mut v = json!({
        "id": ids.id,
        "object": "response",
        "created_at": ids.created_at,
        "completed_at": unix_now(),
        "status": status(o),
        "model": ids.model,
        "output": output,
        "usage": usage(&o.timings),
    });
    if o.stop == StopKind::Limit {
        v["incomplete_details"] = json!({ "reason": "max_output_tokens" });
    }
    v
}

/// The non-stream answer.
fn whole(
    state: &State,
    req: &Request,
    w: &mut TcpStream,
    ids: &Ids,
    plan: ChatPlan,
) -> io::Result<bool> {
    let ChatPlan {
        p,
        input,
        prompt,
        mut parser,
    } = plan;
    let (o, _) = match run_whole(state, &input, prompt, &p, w)? {
        Err(e) => return send_error(w, req, &e),
        Ok(done) => done,
    };
    if let Err(e) = parser
        .try_push(&o.content)
        .and_then(|_| parser.try_finish())
    {
        return send_error(w, req, &tool_markup_error(&e));
    }
    let mut items = Items::new(ids);
    let mut events = Vec::new();
    items.push(parser.message(), &mut events);
    let output = items.finish(&mut events);
    send_json(w, req, 200, &finished(ids, &o, output))
}

// ---------------------------------------------------------------- the stream

/// The stream's frames: `event: <type>` and `data: <json>`, each data object
/// numbered from 0, the opening pair sent once.
#[derive(Default)]
struct Frames {
    sequence: u64,
    opened: bool,
}

impl Frames {
    /// `response.created` and `response.in_progress` the first time, nothing
    /// after.
    fn opening(&mut self, ids: &Ids) -> Vec<Value> {
        if std::mem::replace(&mut self.opened, true) {
            return Vec::new();
        }
        vec![
            json!({ "type": "response.created", "response": in_progress(ids) }),
            json!({ "type": "response.in_progress", "response": in_progress(ids) }),
        ]
    }

    fn send(&mut self, s: &mut EventStream<'_>, events: Vec<Value>) -> io::Result<()> {
        for mut v in events {
            v["sequence_number"] = json!(self.sequence);
            self.sequence += 1;
            let kind = v["type"]
                .as_str()
                .expect("every event this module builds names its type");
            s.send(format!("event: {kind}\ndata: {v}\n\n").as_bytes())?;
        }
        Ok(())
    }
}

/// The stream. The head and the opening pair go out with the first event. An
/// error before the head is the request's answer; after it, the chat path's
/// error frame ([`finish_stream`]).
fn stream(
    state: &State,
    req: &Request,
    w: &mut TcpStream,
    ids: &Ids,
    plan: ChatPlan,
) -> io::Result<bool> {
    let ChatPlan {
        p,
        input,
        prompt,
        mut parser,
    } = plan;
    let mut stream: Option<EventStream<'_>> = None;
    let mut w_opt = Some(w);
    let mut frames = Frames::default();
    let mut items = Items::new(ids);
    let tpt = p.timings_per_token;
    // Markup that does not parse stops the generation (the sink fails) and
    // ends the stream with an error frame instead of a dropped connection.
    let mut markup = None;
    let r = {
        let mut sink = |ev: Event<'_>, _slot: usize| -> io::Result<()> {
            if stream.is_none() {
                let w = w_opt
                    .take()
                    .ok_or_else(|| io::Error::other("stream writer taken"))?;
                stream = Some(EventStream::start(w, req, 200, "text/event-stream")?);
            }
            let s = stream
                .as_mut()
                .ok_or_else(|| io::Error::other("no stream"))?;
            let mut events = frames.opening(ids);
            match ev {
                Event::Prompt(t) => {
                    if events.is_empty() {
                        events.push(
                            json!({ "type": "response.in_progress", "response": in_progress(ids) }),
                        );
                    }
                    if let Some(last) = events.last_mut() {
                        last["prompt_progress"] = progress(t);
                    }
                }
                Event::Text(text, t) => {
                    let d = parser.try_push(text).map_err(|e| {
                        markup = Some(e);
                        io::Error::other("tool-call markup does not parse")
                    })?;
                    items.push(&d, &mut events);
                    if tpt && let Some(last) = events.last_mut() {
                        last["timings"] = t.to_json();
                    }
                }
            }
            frames.send(s, events)
        };
        run_gen(state, &input, prompt, &p, &mut sink).map(|(o, _)| o)
    };
    let r = match markup {
        Some(e) => Err(tool_markup_error(&e)),
        None => r,
    };
    finish_stream(req, stream, w_opt, r, |s, o| {
        let mut events = frames.opening(ids);
        match parser.try_finish() {
            Ok(d) => items.push(&d, &mut events),
            Err(e) => {
                frames.send(s, events)?;
                let e = tool_markup_error(&e);
                return sse(
                    s,
                    &json!({ "error": error_body(e.code, e.kind, &e.message)["error"] }),
                );
            }
        }
        let output = items.finish(&mut events);
        let mut end = json!({
            "type": format!("response.{}", status(o)),
            "response": finished(ids, o, output),
        });
        end["timings"] = o.timings.to_json();
        events.push(end);
        frames.send(s, events)
    })
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use serde_json::{Value, json};

    use super::super::testserve;
    use crate::{Engine, MockEngine, ScriptedEngine};

    /// The tests' chat template: the mock's, with each message's reasoning
    /// (`<think>…</think>`), its calls in Hermes' markup, each followed by its
    /// id in parentheses, and a tool message's call id in brackets, after a
    /// line for the reasoning effort and one for the tools.
    /// Its rendered call types it Hermes', so the server scans a scripted call.
    const TEMPLATE: &str = concat!(
        "{%- if reasoning_effort is defined %}",
        "{{- '[effort ' + reasoning_effort + ']\\n' }}{%- endif %}",
        "{%- if tools is defined %}{{- '[tools ' }}{{- tools | tojson }}{{- ']\\n' }}{%- endif %}",
        "{%- for message in messages %}",
        "{{- '<' + message.role + '>\\n' }}",
        "{%- if message.reasoning_content is defined %}",
        "{{- '<think>' + message.reasoning_content + '</think>' }}{%- endif %}",
        "{{- message.content }}",
        "{%- if message.tool_calls is defined %}{%- for call in message.tool_calls %}",
        "{{- '<tool_call>\\n{\"name\": \"' + call.function.name + '\", \"arguments\": ' }}",
        "{%- if call.function.arguments is string %}{{- call.function.arguments }}",
        "{%- else %}{{- call.function.arguments | tojson }}{%- endif %}",
        "{{- '}\\n</tool_call>' }}",
        "{%- if call.id is defined %}{{- '(' + call.id + ')' }}{%- endif %}",
        "{%- endfor %}{%- endif %}",
        "{%- if message.tool_call_id is defined %}{{- '[' + message.tool_call_id + ']' }}{%- endif %}",
        "{%- endfor %}",
        "{%- if add_generation_prompt %}{{- '<assistant>\\n' }}{%- endif %}",
    );

    /// A Hermes call of `get_weather` for Seoul, as a model writes it.
    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Seoul\"}}\n</tool_call>";

    /// `engine` served under [`TEMPLATE`].
    fn serve(engine: Box<dyn Engine>) -> SocketAddr {
        let mut config = testserve::mock_config();
        config.chat_template = TEMPLATE.to_owned();
        testserve::spawn(engine, config).0
    }

    /// A server whose every generation is `text`, then the end of generation.
    fn scripted(text: &str) -> SocketAddr {
        serve(Box::new(ScriptedEngine::new(4096, text)))
    }

    /// One `POST` of `body`: the status and the answer's JSON.
    fn post(addr: SocketAddr, path: &str, body: &Value) -> (u16, Value) {
        let (status, text) = testserve::roundtrip(addr, "POST", path, &[], &body.to_string());
        let v = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{path}: the answer is not JSON ({e}): {text}"));
        (status, v)
    }

    /// One stream `POST` of `body`: its status and its whole body.
    fn post_stream(addr: SocketAddr, path: &str, body: &Value) -> (u16, String) {
        testserve::roundtrip(addr, "POST", path, &[], &body.to_string())
    }

    /// A 200's JSON.
    fn answered(addr: SocketAddr, path: &str, body: &Value) -> Value {
        let (status, v) = post(addr, path, body);
        assert_eq!(status, 200, "{body} -> {v}");
        v
    }

    /// The rendered prompt of the last request the server's one slot ran.
    fn last_prompt(addr: SocketAddr) -> String {
        let (status, text) = testserve::roundtrip(addr, "GET", "/slots", &[], "");
        assert_eq!(status, 200, "{text}");
        let v: Value = serde_json::from_str(&text).expect("/slots is JSON");
        v[0]["prompt"].as_str().expect("a slot's prompt").to_owned()
    }

    /// The prompt the chat body `chat` renders on the same server: the
    /// conversion's oracle, the body written as llama-server converts.
    fn rendered(addr: SocketAddr, chat: &Value) -> String {
        let v = answered(addr, "/apply-template", chat);
        v["prompt"].as_str().expect("a prompt").to_owned()
    }

    /// The 16 hex digits every item id of the response `id` carries.
    fn nonce(id: &Value) -> String {
        let id = id.as_str().expect("a string id");
        assert!(
            id.starts_with("resp_") && id.len() == 5 + 32,
            "a response id is resp_ and 32 hex digits: {id}"
        );
        id[5..21].to_owned()
    }

    /// The data objects of a stream, in order. Every frame is an `event:` line
    /// naming its data's `type` and a `data:` line, and the sequence numbers
    /// count from 0.
    fn frames(status: u16, body: &str) -> Vec<Value> {
        assert_eq!(status, 200, "{body}");
        let frames: Vec<Value> = body
            .split("\n\n")
            .filter(|f| !f.is_empty())
            .map(|f| {
                let (ev, data) = f
                    .split_once('\n')
                    .unwrap_or_else(|| panic!("a frame of one line: {f:?}"));
                let ev = ev
                    .strip_prefix("event: ")
                    .unwrap_or_else(|| panic!("no event line: {f:?}"));
                let data = data
                    .strip_prefix("data: ")
                    .unwrap_or_else(|| panic!("no data line: {f:?}"));
                let v: Value = serde_json::from_str(data)
                    .unwrap_or_else(|e| panic!("data is not JSON ({e}): {data}"));
                assert_eq!(v["type"], ev, "the event's name is its data's type: {f:?}");
                v
            })
            .collect();
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(f["sequence_number"], i, "{f}");
        }
        frames
    }

    /// The frames' types, a run of one type as one.
    fn kinds(frames: &[Value]) -> Vec<&str> {
        let mut k: Vec<&str> = frames
            .iter()
            .map(|f| f["type"].as_str().expect("a type"))
            .collect();
        k.dedup();
        k
    }

    /// The frames of type `kind`.
    fn of<'a>(frames: &'a [Value], kind: &str) -> Vec<&'a Value> {
        frames.iter().filter(|f| f["type"] == kind).collect()
    }

    /// Asserts a refusal: a 400 carrying exactly the chat path's error object,
    /// its message holding `part`.
    fn assert_refused(addr: SocketAddr, path: &str, body: &Value, part: &str) {
        let (status, v) = post(addr, path, body);
        assert_eq!(status, 400, "{body} -> {v}");
        let e = v["error"].as_object().unwrap_or_else(|| panic!("{v}"));
        let mut keys: Vec<&String> = e.keys().collect();
        keys.sort();
        assert_eq!(keys, ["code", "message", "type"], "{v}");
        assert_eq!(v.as_object().map(|o| o.len()), Some(1), "{v}");
        assert_eq!(e["code"], 400, "{v}");
        assert_eq!(e["type"], "invalid_request_error", "{v}");
        assert!(
            e["message"].as_str().is_some_and(|m| m.contains(part)),
            "{body}: the message must hold {part:?}: {v}"
        );
    }

    /// llama-server's case's request (`test_compat_oai_responses.py`).
    fn book(extra: Value) -> Value {
        let mut b = json!({
            "model": "gpt-4.1",
            "input": [
                {"role": "system", "content": "Book"},
                {"role": "user", "content": "What is the best book"},
            ],
            "max_output_tokens": 8,
            "temperature": 0.8,
        });
        if let (Value::Object(b), Value::Object(e)) = (&mut b, extra) {
            b.extend(e);
        }
        b
    }

    /// Ports `test_responses_with_openai_library`, held to exact values: the
    /// response's id is `resp_` and its message's `msg_`, its text the
    /// generation's. The script's eight ids fill `max_output_tokens` before
    /// its end, so the answer is `incomplete` (`max_output_tokens`), where
    /// llama-server says `completed`; with room for the end it is
    /// `completed`, and a stop at the context's end is `incomplete` as the
    /// limit's is. `usage` counts the whole prompt, the generated ids (the
    /// end's id with them) and nothing cached.
    #[test]
    fn whole_answer_ports_the_library_case() {
        let addr = scripted("Suddenly");
        let v = answered(addr, "/v1/responses", &book(json!({})));
        let n = nonce(&v["id"]);
        let prompt = last_prompt(addr);
        assert_eq!(
            prompt,
            "<system>\nBook<user>\nWhat is the best book<assistant>\n"
        );
        // The mock's ids are the prompt's bytes: it holds no special.
        let input = prompt.len();
        let created = v["created_at"].as_u64().expect("created_at");
        assert!(v["completed_at"].as_u64().expect("completed_at") >= created);
        let message = json!({
            "id": format!("msg_0_{n}"), "type": "message", "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "Suddenly"}],
        });
        assert_eq!(
            v,
            json!({
                "id": v["id"], "object": "response", "created_at": created,
                "completed_at": v["completed_at"], "status": "incomplete", "model": "gpt-4.1",
                "output": [message],
                "usage": {
                    "input_tokens": input, "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 8, "total_tokens": input + 8,
                },
                "incomplete_details": {"reason": "max_output_tokens"},
            })
        );
        let v = answered(
            addr,
            "/v1/responses",
            &book(json!({"max_output_tokens": 16})),
        );
        assert_eq!(v["status"], "completed", "{v}");
        assert!(v.get("incomplete_details").is_none(), "{v}");
        assert_eq!(v["output"][0]["content"][0]["text"], "Suddenly", "{v}");
        assert_eq!(v["usage"]["output_tokens"], 9, "{v}");
        // The context's end is the same cut: a context three ids past the prompt.
        let short = serve(Box::new(ScriptedEngine::new(input + 3, "Suddenly")));
        let v = answered(
            short,
            "/v1/responses",
            &book(json!({"max_output_tokens": 16})),
        );
        assert_eq!(v["status"], "incomplete", "{v}");
        assert_eq!(
            v["incomplete_details"],
            json!({"reason": "max_output_tokens"}),
            "{v}"
        );
    }

    /// Ports `test_responses_stream_with_openai_library`: `response.created`
    /// and `response.in_progress` name one `resp_` id; the message item is
    /// added, its part added, its text streamed and done, its part done, the
    /// item done, every event naming the item's `msg_` id at output index 0;
    /// the deltas join into the final response's text. The stream ends with
    /// `response.incomplete` here (the limit), whose response is the whole
    /// answer.
    #[test]
    fn stream_ports_the_library_case() {
        let addr = scripted("Suddenly");
        let (status, body) = post_stream(addr, "/v1/responses", &book(json!({"stream": true})));
        let f = frames(status, &body);
        assert_eq!(
            kinds(&f),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.incomplete",
            ]
        );
        let id = &f[0]["response"]["id"];
        let n = nonce(id);
        let msg = format!("msg_0_{n}");
        for opening in &f[..2] {
            assert_eq!(
                opening["response"],
                json!({
                    "id": id, "object": "response", "created_at": f[0]["response"]["created_at"],
                    "status": "in_progress", "model": "gpt-4.1", "output": [],
                }),
                "{opening}"
            );
        }
        assert_eq!(f[2]["item"]["id"], msg.as_str(), "{}", f[2]);
        for e in &f[2..f.len() - 1] {
            assert_eq!(e["output_index"], 0, "{e}");
            if e.get("item_id").is_some() {
                assert_eq!(e["item_id"], msg.as_str(), "{e}");
            }
        }
        let text: String = of(&f, "response.output_text.delta")
            .iter()
            .map(|d| d["delta"].as_str().expect("a delta"))
            .collect();
        assert_eq!(text, "Suddenly");
        assert_eq!(of(&f, "response.output_text.done")[0]["text"], "Suddenly");
        let done = of(&f, "response.output_item.done")[0];
        let end = f.last().expect("a last frame");
        assert_eq!(end["response"]["id"], *id, "{end}");
        assert_eq!(end["response"]["status"], "incomplete", "{end}");
        assert_eq!(end["response"]["output"], json!([done["item"]]), "{end}");
        assert_eq!(
            end["response"]["output"][0]["content"][0]["text"],
            "Suddenly"
        );
        assert!(end["timings"].is_object(), "{end}");
    }

    /// Ports `test_responses_stream_with_llama_telemetry` on `/responses`:
    /// with `return_progress`, the prompt's `prompt_progress` rides on a
    /// `response.in_progress` (its total the prompt, `processed` at least
    /// `cache`); with `timings_per_token` every text delta carries the
    /// generation's `timings`; the last frame carries `usage` and `timings`.
    #[test]
    fn stream_telemetry_ports_the_llama_case() {
        let addr = scripted("Suddenly");
        let body = json!({
            "input": "This is a test".repeat(10),
            "max_output_tokens": 8,
            "temperature": 0.8,
            "stream": true,
            "timings_per_token": true,
            "return_progress": true,
        });
        let (status, text) = post_stream(addr, "/responses", &body);
        let f = frames(status, &text);
        let progress: Vec<&Value> = f
            .iter()
            .filter(|e| e.get("prompt_progress").is_some())
            .collect();
        assert_eq!(progress.len(), 1, "{text}");
        let p = &progress[0];
        assert_eq!(p["type"], "response.in_progress", "{p}");
        let pp = &p["prompt_progress"];
        assert_eq!(pp["total"], last_prompt(addr).len(), "{p}");
        let count = |k: &str| {
            pp[k]
                .as_u64()
                .unwrap_or_else(|| panic!("no {k} count: {p}"))
        };
        assert!(count("processed") >= count("cache"), "{p}");
        let deltas = of(&f, "response.output_text.delta");
        assert!(!deltas.is_empty(), "{text}");
        for d in deltas {
            let t = d["timings"].as_object().unwrap_or_else(|| panic!("{d}"));
            assert!(
                t.contains_key("prompt_per_second") && t.contains_key("predicted_per_second"),
                "{d}"
            );
        }
        let end = f.last().expect("a last frame");
        assert!(end["response"]["usage"].is_object(), "{end}");
        assert!(end["timings"].is_object(), "{end}");
    }

    /// Every input item converts as llama-server converts it, the prompt the
    /// same as its chat body renders: `instructions` first; user, system and
    /// developer messages (a string or `input_text` parts) with their roles,
    /// a developer message rendered as system (`render_chat`'s mapping);
    /// an assistant's `output_text`, its call and its reasoning joining one
    /// assistant message; a call's output as a `tool` message of its parts.
    /// Two lines are this server's reading where llama-server's differs: the
    /// assistant message without a `type` (refused there) and the reasoning's
    /// two parts (its first part alone there).
    #[test]
    fn input_items_render_as_llama_server_converts() {
        let addr = scripted("ok");
        let text = |t: &str| json!({"type": "input_text", "text": t});
        let out = |t: &str| json!({"type": "output_text", "text": t, "annotations": []});
        let body = json!({
            "instructions": "I",
            "input": [
                {"role": "system", "content": "S"},
                {"type": "message", "role": "developer", "content": [text("D1"), text("D2")]},
                {"role": "user", "content": [text("U")]},
                {"type": "message", "role": "assistant", "id": "msg_a", "status": "completed",
                 "content": [out("A1")]},
                {"role": "assistant", "content": "A2"},
                {"type": "function_call", "id": "fc_a", "call_id": "c1", "name": "f",
                 "arguments": "{\"x\":1}"},
                {"type": "function_call_output", "call_id": "c1", "output": [text("R1"), text("R2")]},
                {"type": "reasoning", "id": "rs_a", "summary": [],
                 "content": [{"type": "reasoning_text", "text": "T1"},
                             {"type": "reasoning_text", "text": "T2"}]},
                {"type": "message", "role": "assistant", "content": [out("A3")]},
                {"role": "user", "content": "U2"},
            ],
        });
        answered(addr, "/v1/responses", &body);
        let prompt = last_prompt(addr);
        let part = |t: &str| json!({"text": t, "type": "text"});
        let chat = json!({"messages": [
            {"role": "system", "content": "I"},
            {"role": "system", "content": [part("S")]},
            {"role": "developer", "content": [part("D1"), part("D2")]},
            {"role": "user", "content": [part("U")]},
            {"role": "assistant", "content": [part("A1"), part("A2")], "tool_calls": [
                {"function": {"arguments": "{\"x\":1}", "name": "f"}, "id": "c1", "type": "function"},
            ]},
            {"content": [part("R1"), part("R2")], "role": "tool", "tool_call_id": "c1"},
            {"role": "assistant", "content": [part("A3")], "reasoning_content": "T1T2"},
            {"role": "user", "content": [part("U2")]},
        ]});
        assert_eq!(prompt, rendered(addr, &chat));
        assert_eq!(
            prompt,
            concat!(
                "<system>\nI<system>\nS<system>\nD1\nD2<user>\nU<assistant>\nA1\nA2",
                "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\":1}}\n</tool_call>(c1)",
                "<tool>\nR1\nR2[c1]<assistant>\n<think>T1T2</think>A3<user>\nU2<assistant>\n",
            )
        );
    }

    /// A function call's round trip: the function tool converts to the chat
    /// path's (`strict` true when absent); the scripted Hermes call comes back
    /// as one `function_call` item, its `call_id` and `fc_` id the response's;
    /// the next request, carrying that item as it came and the call's output,
    /// renders the prompt llama-server's conversion renders: the call as the
    /// assistant's `tool_calls` entry under its `call_id`, the output as a
    /// `tool` message for it.
    #[test]
    fn function_call_round_trip_renders_as_llama_server_converts() {
        let addr = scripted(CALL);
        let parameters = json!({"type": "object", "properties": {"location": {"type": "string"}}});
        let tools = json!([{
            "type": "function", "name": "get_weather", "description": "the weather",
            "parameters": parameters,
        }]);
        let chat_tools = json!([{"type": "function", "function": {
            "name": "get_weather", "description": "the weather", "parameters": parameters,
            "strict": true,
        }}]);
        let v = answered(
            addr,
            "/v1/responses",
            &json!({"input": "weather in Seoul?", "tools": tools}),
        );
        let n = nonce(&v["id"]);
        assert_eq!(v["status"], "completed", "{v}");
        let call_id = format!("call_0_{n}");
        let call = json!({
            "id": format!("fc_0_{n}"), "type": "function_call", "status": "completed",
            "call_id": call_id, "name": "get_weather", "arguments": "{\"location\":\"Seoul\"}",
        });
        assert_eq!(v["output"], json!([call]), "{v}");
        assert_eq!(
            last_prompt(addr),
            rendered(
                addr,
                &json!({
                    "messages": [{"role": "user", "content": "weather in Seoul?"}],
                    "tools": chat_tools,
                })
            )
        );
        let next = json!({
            "input": [
                {"role": "user", "content": "weather in Seoul?"},
                v["output"][0],
                {"type": "function_call_output", "call_id": call_id, "output": "sunny"},
            ],
            "tools": tools,
        });
        answered(addr, "/v1/responses", &next);
        let prompt = last_prompt(addr);
        let chat = json!({
            "messages": [
                {"role": "user", "content": [{"text": "weather in Seoul?", "type": "text"}]},
                {"role": "assistant", "tool_calls": [{
                    "function": {"arguments": "{\"location\":\"Seoul\"}", "name": "get_weather"},
                    "id": call_id, "type": "function",
                }]},
                {"content": "sunny", "role": "tool", "tool_call_id": call_id},
            ],
            "tools": chat_tools,
        });
        assert_eq!(prompt, rendered(addr, &chat));
        let turns = format!(
            "<user>\nweather in Seoul?<assistant>\n<tool_call>\n{{\"name\": \"get_weather\", \
             \"arguments\": {{\"location\":\"Seoul\"}}}}\n</tool_call>({call_id})<tool>\n\
             sunny[{call_id}]<assistant>\n"
        );
        assert!(prompt.ends_with(&turns), "{prompt}");
    }

    /// The think span the chat path splits off comes back as a `reasoning`
    /// item (one `reasoning_text` part, `summary` and `encrypted_content`
    /// empty) before the message; the next request carrying both items as they
    /// came renders one assistant message, the reasoning its
    /// `reasoning_content`, as llama-server's conversion renders it.
    #[test]
    fn reasoning_item_goes_out_and_comes_back_in() {
        let addr = scripted("<think>plan</think>answer");
        let v = answered(addr, "/v1/responses", &json!({"input": "hi"}));
        let n = nonce(&v["id"]);
        assert_eq!(
            v["output"],
            json!([
                {"id": format!("rs_0_{n}"), "type": "reasoning", "summary": [],
                 "content": [{"type": "reasoning_text", "text": "plan"}],
                 "encrypted_content": "", "status": "completed"},
                {"id": format!("msg_1_{n}"), "type": "message", "role": "assistant",
                 "status": "completed",
                 "content": [{"type": "output_text", "annotations": [], "logprobs": [],
                              "text": "answer"}]},
            ]),
            "{v}"
        );
        let next = json!({"input": [
            {"role": "user", "content": "hi"},
            v["output"][0],
            v["output"][1],
            {"role": "user", "content": "again"},
        ]});
        answered(addr, "/v1/responses", &next);
        let prompt = last_prompt(addr);
        let chat = json!({"messages": [
            {"role": "user", "content": [{"text": "hi", "type": "text"}]},
            {"role": "assistant", "content": [{"text": "answer", "type": "text"}],
             "reasoning_content": "plan"},
            {"role": "user", "content": [{"text": "again", "type": "text"}]},
        ]});
        assert_eq!(prompt, rendered(addr, &chat));
        assert_eq!(
            prompt,
            "<user>\nhi<assistant>\n<think>plan</think>answer<user>\nagain<assistant>\n"
        );
    }

    /// A stream of reasoning, text and a call: each item is added, streamed
    /// and done before the next is added, in output order (reasoning, message,
    /// call); the final response's `output` is the done items, and is the
    /// whole answer's to the same request but for the response's own hex.
    #[test]
    fn stream_items_close_in_order_and_match_the_whole_answer() {
        let addr = scripted(&format!("<think>plan</think>answer{CALL}"));
        let body = json!({
            "input": "hi",
            "tools": [{"type": "function", "name": "get_weather"}],
        });
        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let (status, text) = post_stream(addr, "/v1/responses", &streamed);
        let f = frames(status, &text);
        assert_eq!(
            kinds(&f),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_text.delta",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let added: Vec<&Value> = of(&f, "response.output_item.added");
        let done: Vec<&Value> = of(&f, "response.output_item.done");
        for (i, (a, d)) in added.iter().zip(&done).enumerate() {
            assert_eq!(a["output_index"], i, "{a}");
            assert_eq!(d["output_index"], i, "{d}");
            assert_eq!(a["item"]["id"], d["item"]["id"], "{a} {d}");
            assert_eq!(a["item"]["status"], "in_progress", "{a}");
        }
        let end = f.last().expect("a last frame");
        let output: Vec<&Value> = done.iter().map(|d| &d["item"]).collect();
        assert_eq!(end["response"]["output"], json!(output), "{end}");
        let whole = answered(addr, "/v1/responses", &body);
        let same = whole["output"]
            .to_string()
            .replace(&nonce(&whole["id"]), &nonce(&end["response"]["id"]));
        assert_eq!(same, end["response"]["output"].to_string());
    }

    /// `tools`: a function tool converts with its fields, `strict` true when
    /// absent and kept when set; a web search (Codex sends one on every
    /// request) is dropped, as llama-server drops it, and a request of web
    /// searches alone renders no tools; every other type is refused by name,
    /// where llama-server drops it.
    #[test]
    fn tools_convert_drop_web_search_and_refuse_the_rest() {
        let addr = scripted("ok");
        let body = json!({"input": "hi", "tools": [
            {"type": "function", "name": "a"},
            {"type": "web_search", "external_web_access": false},
            {"type": "function", "name": "b", "description": "B", "strict": false,
             "parameters": {"type": "object"}},
            {"type": "web_search_preview"},
        ]});
        answered(addr, "/v1/responses", &body);
        let chat = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "a", "strict": true}},
                {"type": "function", "function": {"name": "b", "description": "B", "strict": false,
                                                  "parameters": {"type": "object"}}},
            ],
        });
        assert_eq!(last_prompt(addr), rendered(addr, &chat));
        let searches = json!({"input": "hi", "tools": [{"type": "web_search"}]});
        answered(addr, "/v1/responses", &searches);
        assert_eq!(last_prompt(addr), "<user>\nhi<assistant>\n");
        for kind in [
            "custom",
            "namespace",
            "tool_search",
            "file_search",
            "code_interpreter",
            "image_generation",
            "computer_use_preview",
            "mcp",
            "local_shell",
        ] {
            let body = json!({"input": "hi", "tools": [{"type": kind, "name": "x"}]});
            assert_refused(
                addr,
                "/v1/responses",
                &body,
                &format!("tool type {kind} is not supported"),
            );
        }
    }

    /// The request Codex sends is served: its bookkeeping (`store: false`,
    /// `include` of the encrypted reasoning, `prompt_cache_key`,
    /// `client_metadata`, `parallel_tool_calls`, `reasoning.summary`,
    /// `text.verbosity`, …) is ignored; `max_output_tokens` and the sampling
    /// fields reach the generation, llama-server's own (`top_k`, `seed`) too,
    /// as the body passes whole; `reasoning.effort` reaches the template as
    /// `reasoning_effort`.
    #[test]
    fn a_codex_request_is_served_and_its_fields_reach_the_generation() {
        let addr = scripted("ok");
        let body = json!({
            "model": "bloomery",
            "input": [
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": "D"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            ],
            "tools": [{"type": "web_search", "external_web_access": false}],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "reasoning": {"effort": "high", "summary": "auto"},
            "store": false,
            "stream": false,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "k",
            "client_metadata": {"a": "b"},
            "service_tier": "auto",
            "text": {"verbosity": "low"},
            "metadata": {"x": "y"},
            "user": "u",
            "truncation": "auto",
            "max_output_tokens": 5,
            "temperature": 0.25,
            "top_p": 0.5,
            "top_k": 7,
            "seed": 42,
        });
        let v = answered(addr, "/v1/responses", &body);
        assert_eq!(v["model"], "bloomery", "{v}");
        let (status, text) = testserve::roundtrip(addr, "GET", "/slots", &[], "");
        assert_eq!(status, 200, "{text}");
        let slot: Value = serde_json::from_str(&text).expect("/slots is JSON");
        let slot = &slot[0];
        for (k, want) in [
            ("n_predict", json!(5)),
            ("temperature", json!(0.25)),
            ("top_p", json!(0.5)),
            ("top_k", json!(7)),
            ("seed", json!(42)),
        ] {
            assert_eq!(slot[k], want, "{k}: {slot}");
        }
        assert_eq!(
            slot["prompt"],
            "[effort high]\n<system>\nD<user>\nhi<assistant>\n"
        );
    }

    /// What needs a store is refused by name — `previous_response_id`,
    /// `conversation`, `store: true`, `background: true`, a stored `prompt`,
    /// an `item_reference` item — on both routes; `store: false`, an empty or
    /// `null` `previous_response_id` and `background: false` pass.
    #[test]
    fn stored_state_is_refused_by_name() {
        let addr = scripted("ok");
        let with = |k: &str, v: Value| json!({"input": "hi", k: v});
        for (body, part) in [
            (
                with("previous_response_id", json!("resp_1")),
                "previous_response_id is not supported",
            ),
            (
                with("conversation", json!("conv_1")),
                "conversation is not supported",
            ),
            (
                with("conversation", json!({"id": "conv_1"})),
                "conversation is not supported",
            ),
            (with("store", json!(true)), "store: true is not supported"),
            (
                with("background", json!(true)),
                "background: true is not supported",
            ),
            (
                with("prompt", json!({"id": "pmpt_1"})),
                "prompt is not supported",
            ),
            (
                json!({"input": [{"type": "item_reference", "id": "msg_1"}]}),
                "item_reference is not supported",
            ),
        ] {
            for path in ["/v1/responses", "/v1/responses/input_tokens"] {
                assert_refused(addr, path, &body, part);
            }
        }
        for body in [
            with("store", json!(false)),
            with("previous_response_id", json!("")),
            with("previous_response_id", Value::Null),
            with("background", json!(false)),
        ] {
            answered(addr, "/v1/responses", &body);
        }
    }

    /// The answer carries no probabilities and one answer: `logprobs`,
    /// `top_logprobs` above 0, `include` of `message.output_text.logprobs`
    /// and `n` above 1 are refused by name; `logprobs: false`,
    /// `top_logprobs: 0`, `n: 1` and the other `include` values pass.
    #[test]
    fn probabilities_and_several_answers_are_refused_by_name() {
        let addr = scripted("ok");
        let with = |k: &str, v: Value| json!({"input": "hi", k: v});
        for (body, part) in [
            (with("logprobs", json!(true)), "logprobs is not supported"),
            (
                with("top_logprobs", json!(3)),
                "top_logprobs is not supported",
            ),
            (
                with("include", json!(["message.output_text.logprobs"])),
                "include message.output_text.logprobs is not supported",
            ),
            (with("n", json!(2)), "n is not supported"),
        ] {
            assert_refused(addr, "/v1/responses", &body, part);
        }
        for body in [
            with("logprobs", json!(false)),
            with("top_logprobs", json!(0)),
            with("n", json!(1)),
            with(
                "include",
                json!([
                    "reasoning.encrypted_content",
                    "message.input_image.image_url"
                ]),
            ),
        ] {
            answered(addr, "/v1/responses", &body);
        }
    }

    /// `text.format` other than `text` (a JSON schema, a JSON object) asks for
    /// structured output, which this server does not constrain: refused by
    /// name. `text` passes.
    #[test]
    fn structured_output_is_refused_by_name() {
        let addr = scripted("ok");
        for kind in ["json_schema", "json_object"] {
            let body = json!({"input": "hi", "text": {"format": {"type": kind, "name": "x",
                                                                  "schema": {}}}});
            assert_refused(
                addr,
                "/v1/responses",
                &body,
                &format!("text.format of type {kind} is not supported"),
            );
        }
        answered(
            addr,
            "/v1/responses",
            &json!({"input": "hi", "text": {"format": {"type": "text"}}}),
        );
    }

    /// An `input_image` is the chat path's `image_url` part, so a text-only
    /// engine answers it as `/v1/chat/completions` answers that part; an
    /// `input_image` by `file_id` and an `input_file`, in a message or in a
    /// call's output, are refused by name.
    #[test]
    fn images_are_the_chat_paths_and_files_are_refused() {
        let addr = scripted("ok");
        let image = json!({"type": "input_image", "image_url": "data:image/png;base64,AAAA"});
        let responses = json!({"input": [{"role": "user", "content": [image]}]});
        let chat = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]});
        let (status, v) = post(addr, "/v1/responses", &responses);
        let (chat_status, chat_v) = post(addr, "/v1/chat/completions", &chat);
        assert_eq!((status, &v), (chat_status, &chat_v));
        assert_eq!(status, 400, "{v}");
        let by_file = json!({"type": "input_image", "file_id": "file_1"});
        let file = json!({"type": "input_file", "file_id": "file_1"});
        for (part, refusal) in [
            (by_file, "an input_image by file_id is not supported"),
            (file, "input_file is not supported"),
        ] {
            let in_message = json!({"input": [{"role": "user", "content": [part]}]});
            let in_output = json!({"input": [
                {"type": "function_call_output", "call_id": "c", "output": [part]},
            ]});
            for body in [in_message, in_output] {
                assert_refused(addr, "/v1/responses", &body, refusal);
            }
        }
    }

    /// A request the conversion cannot read is refused by name: a missing or
    /// mistyped field, an item, a part or a role it does not know, and a
    /// reasoning item without its text (OpenAI's encrypted reasoning).
    #[test]
    fn unreadable_requests_are_refused_by_name() {
        let addr = scripted("ok");
        let input = |items: Value| json!({"input": items});
        let user = |content: Value| input(json!([{"role": "user", "content": content}]));
        let assistant = |content: Value| input(json!([{"role": "assistant", "content": content}]));
        let reasoning = |r: Value| input(json!([r]));
        let tool = |t: Value| json!({"input": "hi", "tools": [t]});
        let cases = [
            (json!({}), "input is required"),
            (
                json!({"input": 3}),
                "input must be a string or an array of items",
            ),
            (input(json!([3])), "each input item must be an object"),
            (
                input(json!([{"type": 3}])),
                "an input item's type must be a string",
            ),
            (
                input(json!([{"type": "custom_tool_call", "call_id": "c", "input": "x"}])),
                "input item type \"custom_tool_call\" is not supported",
            ),
            (
                input(json!([{"content": "x"}])),
                "an input message needs a string 'role'",
            ),
            (
                input(json!([{"role": "tool", "content": "x"}])),
                "role must be user, system, developer or assistant",
            ),
            (
                input(json!([{"role": "user"}])),
                "a user message needs its content",
            ),
            (
                user(json!(3)),
                "content must be a string or an array of content parts",
            ),
            (
                user(json!([{"text": "x"}])),
                "a content part must be an object with a string 'type'",
            ),
            (
                user(json!([{"type": "input_text", "text": 3}])),
                "an input_text part needs a string 'text'",
            ),
            (
                user(json!([{"type": "output_text", "text": "x"}])),
                "content part type \"output_text\" is not supported in a user message",
            ),
            (
                user(json!([{"type": "input_image"}])),
                "an input_image needs a string 'image_url'",
            ),
            (
                assistant(json!([{"type": "refusal", "refusal": "no"}])),
                "refusal part is not supported",
            ),
            (
                assistant(json!([{"type": "input_image", "image_url": "x"}])),
                "content part type \"input_image\" is not supported in an assistant message",
            ),
            (
                input(json!([{"type": "function_call", "name": "f", "arguments": "{}"}])),
                "a function_call item needs a string 'call_id'",
            ),
            (
                input(
                    json!([{"type": "function_call", "call_id": "c", "name": "f",
                              "arguments": {}}]),
                ),
                "a function_call item needs a string 'arguments'",
            ),
            (
                input(json!([{"type": "function_call_output", "call_id": "c"}])),
                "a function_call_output item needs its output",
            ),
            (
                input(json!([{"type": "function_call_output", "call_id": "c", "output": 3}])),
                "output must be a string or an array of parts",
            ),
            (
                reasoning(
                    json!({"type": "reasoning", "content": [{"type": "reasoning_text",
                                                                  "text": "t"}]}),
                ),
                "a reasoning item needs a 'summary' array",
            ),
            (
                reasoning(json!({"type": "reasoning", "summary": [], "encrypted_content": "x"})),
                "a reasoning item needs a non-empty 'content' array",
            ),
            (
                reasoning(json!({"type": "reasoning", "summary": [], "content": []})),
                "a reasoning item needs a non-empty 'content' array",
            ),
            (
                reasoning(json!({"type": "reasoning", "summary": [],
                                 "content": [{"type": "summary_text", "text": "t"}]})),
                "content part type \"summary_text\" is not supported in a reasoning item",
            ),
            (
                json!({"input": "hi", "instructions": 3}),
                "instructions must be a string",
            ),
            (
                json!({"input": "hi", "tools": {}}),
                "tools must be an array",
            ),
            (tool(json!(3)), "each tool must be an object"),
            (
                tool(json!({"name": "f"})),
                "each tool needs a string 'type'",
            ),
            (
                tool(json!({"type": "function"})),
                "each function tool needs a non-empty string 'name'",
            ),
            (
                tool(json!({"type": "function", "name": "f", "parameters": "x"})),
                "tool f: 'parameters' must be an object",
            ),
            (
                tool(json!({"type": "function", "name": "f", "description": 3})),
                "tool f: 'description' must be a string",
            ),
            (
                json!({"input": "hi", "reasoning": "high"}),
                "reasoning must be an object",
            ),
            (
                json!({"input": "hi", "reasoning": {"effort": 3}}),
                "reasoning.effort must be a string",
            ),
            (
                json!({"input": "hi", "include": "x"}),
                "include must be an array of strings",
            ),
            (
                json!({"input": "hi", "include": [3]}),
                "include must hold strings",
            ),
            (
                json!({"input": "hi", "text": "x"}),
                "text must be an object",
            ),
            (
                json!({"input": "hi", "text": {"format": {}}}),
                "text.format must be an object with a string 'type'",
            ),
        ];
        for (body, part) in cases {
            assert_refused(addr, "/v1/responses", &body, part);
        }
    }

    /// `input_tokens` on both routes answers `{"input_tokens": N, "object":
    /// "response.input_tokens"}`, N the ids of the prompt the same request
    /// generates from — its `usage.input_tokens` and the tokenizer's count of
    /// the slot's prompt, special tokens counted as one.
    #[test]
    fn input_tokens_count_the_generations_prompt() {
        let addr = scripted("ok");
        let body = json!({
            "instructions": "be brief",
            "input": [
                {"role": "user", "content": "hi"},
                {"type": "reasoning", "summary": [],
                 "content": [{"type": "reasoning_text", "text": "think"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text",
                                                                      "text": "hello"}]},
                {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c", "output": "done"},
                {"role": "user", "content": "again"},
            ],
            "tools": [{"type": "function", "name": "f"}],
        });
        let counted: Vec<Value> = ["/v1/responses/input_tokens", "/responses/input_tokens"]
            .iter()
            .map(|path| answered(addr, path, &body))
            .collect();
        assert_eq!(counted[0], counted[1]);
        let n = counted[0]["input_tokens"].as_u64().expect("a count");
        assert_eq!(
            counted[0],
            json!({"input_tokens": n, "object": "response.input_tokens"})
        );
        let v = answered(addr, "/v1/responses", &body);
        assert_eq!(v["usage"]["input_tokens"], n, "{v}");
        let prompt = last_prompt(addr);
        assert!(prompt.contains("<think>think</think>"), "{prompt}");
        let ids = answered(
            addr,
            "/tokenize",
            &json!({"content": prompt, "add_special": false}),
        );
        assert_eq!(
            ids["tokens"].as_array().map(Vec::len),
            usize::try_from(n).ok()
        );
    }

    /// Errors are the chat path's OpenAI error object: a prompt past the
    /// context is the plain 400 even when a stream is asked for (nothing has
    /// started); an engine failure after the stream started is one last
    /// `data: {"error": …}` frame with no `event:` line, as llama-server ends
    /// the stream, after the frames already sent.
    #[test]
    fn errors_are_the_chat_paths_and_end_a_started_stream() {
        let small = serve(Box::new(MockEngine::new(16)));
        let long = json!({"input": "a prompt past the context of sixteen ids", "stream": true});
        let (status, v) = post(small, "/v1/responses", &long);
        assert_eq!(status, 400, "{v}");
        assert_eq!(v["error"]["type"], "exceed_context_size_error", "{v}");
        assert_eq!(v["error"]["code"], 400, "{v}");

        let failing = serve(Box::new(MockEngine::failing_at(4096, 2)));
        let body = json!({"input": "hi", "stream": true, "temperature": 0});
        let (status, text) = post_stream(failing, "/v1/responses", &body);
        let (sent, last) = text
            .trim_end()
            .rsplit_once("\n\n")
            .unwrap_or_else(|| panic!("frames before the error: {text}"));
        let f = frames(status, sent);
        assert_eq!(f[0]["type"], "response.created", "{text}");
        let error = last
            .strip_prefix("data: ")
            .unwrap_or_else(|| panic!("the error frame is one data line: {last:?}"));
        let error: Value = serde_json::from_str(error).expect("the error frame is JSON");
        let e = &error["error"];
        assert_eq!(
            (&e["code"], &e["type"]),
            (&json!(500), &json!("server_error")),
            "{error}"
        );
        assert!(
            e["message"]
                .as_str()
                .is_some_and(|m| m.contains("injected failure")),
            "{error}"
        );
        assert_eq!(error.as_object().map(|o| o.len()), Some(1), "{error}");
    }
}
