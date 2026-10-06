//! Anthropic's Messages API on the chat path: `POST /v1/messages` and
//! `POST /v1/messages/count_tokens`, as llama-server serves them
//! (`server_chat_convert_anthropic_to_oai`, `to_json_anthropic`).
//!
//! A request converts into the OpenAI chat body `/v1/chat/completions` reads,
//! and runs the chat path's own steps on it: the sampling fields, the chat
//! template, the tool-call scan the template's markup selects, the think-span
//! budget, the generation and the output parser. Only the two shapes are this
//! module's: the request's conversion ([`to_chat`]) and the answer's.
//!
//! The conversion maps what llama-server maps — `system` (a string or text
//! blocks, joined), `messages` with `text`, `image`, `thinking`, `tool_use` and
//! `tool_result` blocks, `tools`, `tool_choice`, `stop_sequences`, `max_tokens`,
//! `temperature`, `top_p`, `top_k`, `stream`, `chat_template_kwargs` — and
//! `thinking.budget_tokens` to the chat path's `reasoning_budget`. `thinking`
//! does not turn the template's thinking on: that stays the template's default
//! or the request's `chat_template_kwargs`, as in llama-server. A system text that is
//! Claude Code's billing header has its per-request `cch` stamp written as
//! `fffff` ([`normalize_billing_header`]), so the prompt cache reaches past it.
//! Other fields are ignored, as on the OpenAI path (`cache_control` and
//! `metadata` among them). Refused by name where llama-server defaults or drops:
//! a missing `max_tokens`, one below 1, a block of a type the conversion does not
//! know, a malformed tool or a tool of a type other than `custom`, a
//! `tool_choice` or `thinking` of an unknown type, an enabled `thinking` without
//! its `budget_tokens`, or an image source other than `base64` or `url`. An `image`
//! block becomes the OpenAI path's `image_url` part, so an image is answered as
//! that path answers it.
//!
//! The answer: a thinking block, a text block, then one `tool_use` block per
//! call, each present only when it has something; `stop_reason` `max_tokens` at
//! the request's or the context's end, `stop_sequence` on a stop sequence,
//! `tool_use` when a call parsed, else `end_turn`; `usage` the prompt's ids the
//! cache kept (`cache_read_input_tokens`), the rest (`input_tokens`) and the
//! generated ids. A stream is `message_start` (sent with the first text, when
//! the prompt's counts are known), then each block's `content_block_start`, its
//! deltas (`thinking_delta` then `signature_delta`, `text_delta`,
//! `input_json_delta` holding the call's whole input) and its
//! `content_block_stop` before the next block starts, then `message_delta` and
//! `message_stop`; every frame is `event: <type>` and `data: <json>`.
//!
//! An error is Anthropic's envelope around the OpenAI path's error object, its
//! status and `type` unchanged: `{"type":"error","error":{"code","message","type"}}`,
//! and inside a stream that has started, an `error` event carrying the same.

use std::io;
use std::net::TcpStream;

use serde_json::{Map, Value, json};

use super::{
    ApiError, JSON, RETRY_AFTER_SECS, State, body, engine_error, error_body, gate_reasoning_budget,
    gen_params, get_i, invalid, json_type, render_chat, run_gen, send_json, tool_markup_error,
    tool_scan,
};
use crate::dsml::{ChatParser, Message, ToolCall};
use crate::genloop::{Event, GenError, GenParams, Outcome, StopKind, Timings};
use crate::http::{self, EventStream, Request};
use crate::reasoning::ReasoningFormat;

/// The prefix of the system text Claude Code sends first.
const BILLING_HEADER: &str = "x-anthropic-billing-header:";
/// The billing header's `cch` stamp: five characters, then `;`.
const CCH: &str = "cch=";
const CCH_LEN: usize = 5;

/// `POST /v1/messages`.
pub(super) fn messages(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let parsed = body(req).and_then(|b| {
        let chat = to_chat(&b, true)?;
        Ok((Ids::new(state, &b), plan(state, &chat)?))
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

/// `POST /v1/messages/count_tokens`: `{"input_tokens": N}`, N the ids of the
/// prompt the same request renders, by the template and the tokenizer the
/// generation uses. `max_tokens` is not asked for.
pub(super) fn count_tokens(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let counted = body(req)
        .and_then(|b| to_chat(&b, false))
        .and_then(|chat| prompt_of(state, &chat));
    match counted {
        Ok((_, ids)) => send_json(w, req, 200, &json!({ "input_tokens": ids.len() })),
        Err(e) => send_error(w, req, &e),
    }
}

// ---------------------------------------------------------------- the request

/// A field that is present and not `null`.
fn set<'a>(o: &'a Map<String, Value>, k: &str) -> Option<&'a Value> {
    o.get(k).filter(|v| !v.is_null())
}

/// The OpenAI chat body an Anthropic request converts to, or the first field it
/// refuses, by name. `generate` is a `/v1/messages` request, which needs
/// `max_tokens`; a token count does not.
fn to_chat(b: &Map<String, Value>, generate: bool) -> Result<Map<String, Value>, ApiError> {
    let mut messages = Vec::new();
    if let Some(system) = set(b, "system") {
        messages.push(json!({ "role": "system", "content": system_text(system)? }));
    }
    match set(b, "messages") {
        Some(Value::Array(ms)) => {
            for m in ms {
                message(m, &mut messages)?;
            }
        }
        Some(v) => {
            return Err(invalid(format!(
                "'messages' must be an array, not {}",
                json_type(v)
            )));
        }
        None => return Err(invalid("'messages' is required")),
    }
    let mut chat = Map::new();
    chat.insert("messages".into(), Value::Array(messages));
    if let Some(t) = set(b, "tools") {
        chat.insert("tools".into(), tools(t)?);
    }
    if let Some(c) = set(b, "tool_choice") {
        chat.insert("tool_choice".into(), tool_choice(c)?);
    }
    if let Some(s) = set(b, "stop_sequences") {
        chat.insert("stop".into(), stop_sequences(s)?);
    }
    if generate {
        chat.insert("max_tokens".into(), json!(max_tokens(b)?));
    }
    for key in [
        "temperature",
        "top_p",
        "top_k",
        "stream",
        "chat_template_kwargs",
    ] {
        if let Some(v) = b.get(key) {
            chat.insert(key.into(), v.clone());
        }
    }
    if let Some(budget) = thinking_budget(set(b, "thinking"))? {
        chat.insert("reasoning_budget".into(), json!(budget));
    }
    Ok(chat)
}

/// `max_tokens`: required, an integer of at least 1.
fn max_tokens(b: &Map<String, Value>) -> Result<i64, ApiError> {
    match (set(b, "max_tokens"), get_i(b, "max_tokens")?) {
        (_, Some(n)) if n >= 1 => Ok(n),
        (_, Some(n)) => Err(invalid(format!("max_tokens must be at least 1, not {n}"))),
        (None, None) => Err(invalid("max_tokens is required")),
        (Some(v), None) => Err(invalid(format!(
            "max_tokens must be an integer, not {}",
            json_type(v)
        ))),
    }
}

/// The think-span budget `thinking` asks for: `enabled` with its
/// `budget_tokens` (an integer of at least 0, required), `disabled` and
/// `adaptive` none.
fn thinking_budget(v: Option<&Value>) -> Result<Option<u64>, ApiError> {
    let Some(v) = v else {
        return Ok(None);
    };
    match v.get("type").and_then(Value::as_str) {
        Some("enabled") => {
            let Some(n) = v.get("budget_tokens").filter(|n| !n.is_null()) else {
                return Err(invalid(
                    "thinking.budget_tokens is required when thinking.type is enabled",
                ));
            };
            let budget = n.as_u64().or_else(|| {
                n.as_f64()
                    .filter(|f| f.fract() == 0.0 && *f >= 0.0 && *f < u64::MAX as f64)
                    .map(|f| f as u64)
            });
            budget.map(Some).ok_or_else(|| {
                invalid(format!(
                    "thinking.budget_tokens must be an integer of at least 0, not {n}"
                ))
            })
        }
        Some("disabled" | "adaptive") => Ok(None),
        _ => Err(invalid(format!(
            "thinking must be an object whose type is enabled, disabled or adaptive, not {v}"
        ))),
    }
}

/// `stop_sequences` as the chat path's `stop`: an array of strings.
fn stop_sequences(v: &Value) -> Result<Value, ApiError> {
    let Value::Array(a) = v else {
        return Err(invalid(format!(
            "stop_sequences must be an array of strings, not {}",
            json_type(v)
        )));
    };
    if let Some(x) = a.iter().find(|x| !x.is_string()) {
        return Err(invalid(format!(
            "stop_sequences must hold strings, not {x}"
        )));
    }
    Ok(v.clone())
}

/// The system prompt: a string, or text blocks joined with nothing between
/// them, as llama-server joins them; each text normalised
/// ([`normalize_billing_header`]).
fn system_text(v: &Value) -> Result<String, ApiError> {
    match v {
        Value::String(s) => Ok(normalize_billing_header(s)),
        Value::Array(blocks) => blocks
            .iter()
            .map(|b| match block_type(b)? {
                "text" => Ok(normalize_billing_header(text_of(
                    b,
                    "text",
                    "a text block",
                )?)),
                other => Err(invalid(format!(
                    "a system block of type \"{other}\": the system prompt takes text blocks only"
                ))),
            })
            .collect(),
        other => Err(invalid(format!(
            "system must be a string or an array of text blocks, not {}",
            json_type(other)
        ))),
    }
}

/// A system text that starts with Claude Code's billing header has its
/// `cch=` stamp, five characters before a `;`, written as `fffff`: the stamp
/// changes on every request and would end the prompt cache's prefix there.
/// Any other text, and a header whose stamp is not five characters before a
/// `;`, is returned as it is (llama-server's `normalize_anthropic_billing_header`).
fn normalize_billing_header(text: &str) -> String {
    let mut out = text.to_owned();
    if !text.starts_with(BILLING_HEADER) {
        return out;
    }
    let Some(at) = text[BILLING_HEADER.len()..]
        .find(CCH)
        .map(|i| BILLING_HEADER.len() + i + CCH.len())
    else {
        return out;
    };
    let end = at + CCH_LEN;
    if text.as_bytes().get(end) == Some(&b';')
        && text.is_char_boundary(at)
        && text.is_char_boundary(end)
    {
        out.replace_range(at..end, &"f".repeat(CCH_LEN));
    }
    out
}

/// A content block's `type`.
fn block_type(b: &Value) -> Result<&str, ApiError> {
    b.get("type").and_then(Value::as_str).ok_or_else(|| {
        invalid(format!(
            "a content block must be an object with a string 'type', not {b}"
        ))
    })
}

/// The string field `k` of `what`.
fn text_of<'a>(b: &'a Value, k: &str, what: &str) -> Result<&'a str, ApiError> {
    b.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{what} needs a string '{k}'")))
}

/// Appends one Anthropic message as the chat path's messages: the message with
/// its text and image parts, its calls (`tool_calls`) and its thinking
/// (`reasoning_content`), then one `tool` message per `tool_result`, as
/// llama-server orders them. A message of blocks that leave nothing (an empty
/// array) and an assistant message without content add nothing.
fn message(m: &Value, out: &mut Vec<Value>) -> Result<(), ApiError> {
    let Value::Object(m) = m else {
        return Err(invalid("each message must be an object"));
    };
    let Some(role) = m.get("role").and_then(Value::as_str) else {
        return Err(invalid("each message needs a string 'role'"));
    };
    let blocks = match set(m, "content") {
        None if role == "assistant" => return Ok(()),
        None => {
            out.push(json!({ "role": role }));
            return Ok(());
        }
        Some(Value::String(s)) => {
            out.push(json!({ "role": role, "content": s }));
            return Ok(());
        }
        Some(Value::Array(blocks)) => blocks,
        Some(other) => {
            return Err(invalid(format!(
                "a message's content must be a string or an array of content blocks, not {}",
                json_type(other)
            )));
        }
    };
    let mut parts = Vec::new();
    let mut calls = Vec::new();
    let mut reasoning = String::new();
    let mut results = Vec::new();
    for b in blocks {
        match block_type(b)? {
            "text" => {
                parts.push(json!({ "type": "text", "text": text_of(b, "text", "a text block")? }))
            }
            "image" => parts.push(image_part(b)?),
            "thinking" => reasoning.push_str(text_of(b, "thinking", "a thinking block")?),
            "tool_use" => calls.push(tool_call(b)?),
            "tool_result" => results.push(tool_result(b)?),
            other => {
                return Err(invalid(format!(
                    "content block type \"{other}\" is not supported (text, image, thinking, \
                     tool_use, tool_result)"
                )));
            }
        }
    }
    if !parts.is_empty() || !calls.is_empty() || !reasoning.is_empty() {
        let mut msg = Map::new();
        msg.insert("role".into(), json!(role));
        let content = if parts.is_empty() {
            json!("")
        } else {
            Value::Array(parts)
        };
        msg.insert("content".into(), content);
        if !calls.is_empty() {
            msg.insert("tool_calls".into(), Value::Array(calls));
        }
        if !reasoning.is_empty() {
            msg.insert("reasoning_content".into(), json!(reasoning));
        }
        out.push(Value::Object(msg));
    }
    out.extend(results);
    Ok(())
}

/// An `image` block as the OpenAI path's `image_url` part: a `base64` source as
/// a `data:` URL (its `media_type`, `image/jpeg` when absent, as llama-server
/// defaults it), a `url` source as its URL.
fn image_part(b: &Value) -> Result<Value, ApiError> {
    let Some(source) = b.get("source").filter(|s| s.is_object()) else {
        return Err(invalid("an image block needs a 'source' object"));
    };
    let url = match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media = match source.get("media_type").filter(|v| !v.is_null()) {
                None => "image/jpeg",
                Some(v) => v
                    .as_str()
                    .ok_or_else(|| invalid("an image source's 'media_type' must be a string"))?,
            };
            let data = text_of(source, "data", "a base64 image source")?;
            format!("data:{media};base64,{data}")
        }
        Some("url") => text_of(source, "url", "a url image source")?.to_owned(),
        _ => {
            return Err(invalid(format!(
                "image source type {}: only \"base64\" and \"url\" sources are accepted",
                source.get("type").unwrap_or(&Value::Null)
            )));
        }
    };
    Ok(json!({ "type": "image_url", "image_url": { "url": url } }))
}

/// A `tool_use` block as an OpenAI `tool_calls` entry, its `input` the
/// arguments' JSON text.
fn tool_call(b: &Value) -> Result<Value, ApiError> {
    let id = text_of(b, "id", "a tool_use block")?;
    let name = text_of(b, "name", "a tool_use block")?;
    let input = match b.get("input").filter(|v| !v.is_null()) {
        None => json!({}),
        Some(v @ Value::Object(_)) => v.clone(),
        Some(v) => {
            return Err(invalid(format!(
                "a tool_use block's 'input' must be an object, not {}",
                json_type(v)
            )));
        }
    };
    Ok(json!({
        "id": id,
        "type": "function",
        "function": { "name": name, "arguments": input.to_string() },
    }))
}

/// A `tool_result` block as a `tool` message: its content a string, text
/// blocks joined with nothing between them, or — when it holds an image — the
/// text and image parts. `is_error` is not carried, as in llama-server.
fn tool_result(b: &Value) -> Result<Value, ApiError> {
    let id = text_of(b, "tool_use_id", "a tool_result block")?;
    let content = match b.get("content").filter(|v| !v.is_null()) {
        None => json!(""),
        Some(Value::String(s)) => json!(s),
        Some(Value::Array(blocks)) => {
            let mut text = String::new();
            let mut parts = Vec::with_capacity(blocks.len());
            let mut images = false;
            for p in blocks {
                match block_type(p)? {
                    "text" => {
                        let t = text_of(p, "text", "a text block")?;
                        text.push_str(t);
                        parts.push(json!({ "type": "text", "text": t }));
                    }
                    "image" => {
                        images = true;
                        parts.push(image_part(p)?);
                    }
                    other => {
                        return Err(invalid(format!(
                            "a tool_result block of type \"{other}\" is not supported (text, image)"
                        )));
                    }
                }
            }
            if images {
                Value::Array(parts)
            } else {
                json!(text)
            }
        }
        Some(v) => {
            return Err(invalid(format!(
                "a tool_result's content must be a string or an array of blocks, not {}",
                json_type(v)
            )));
        }
    };
    Ok(json!({ "role": "tool", "tool_call_id": id, "content": content }))
}

/// `tools` as OpenAI function tools.
fn tools(v: &Value) -> Result<Value, ApiError> {
    let Value::Array(tools) = v else {
        return Err(invalid(format!(
            "tools must be an array, not {}",
            json_type(v)
        )));
    };
    tools
        .iter()
        .map(tool)
        .collect::<Result<_, _>>()
        .map(Value::Array)
}

/// One custom tool (`type` absent or `custom`) as an OpenAI function: its
/// `name` a non-empty string, its `description` a string or absent, its
/// `input_schema` an object or absent (no parameters). A server tool
/// (`web_search_…` and the like) runs on Anthropic's side and is refused.
fn tool(t: &Value) -> Result<Value, ApiError> {
    let Value::Object(t) = t else {
        return Err(invalid("each tool must be an object"));
    };
    if let Some(kind) = set(t, "type").filter(|k| k.as_str() != Some("custom")) {
        return Err(invalid(format!(
            "tool type {kind} is not supported: this server runs custom tools only"
        )));
    }
    let name = match t.get("name") {
        Some(Value::String(s)) if !s.is_empty() => s,
        _ => return Err(invalid("each tool needs a non-empty string 'name'")),
    };
    let description = match set(t, "description") {
        None => "",
        Some(Value::String(s)) => s,
        Some(v) => {
            return Err(invalid(format!(
                "tool {name}: 'description' must be a string, not {}",
                json_type(v)
            )));
        }
    };
    let parameters = match set(t, "input_schema") {
        None => json!({}),
        Some(v @ Value::Object(_)) => v.clone(),
        Some(v) => {
            return Err(invalid(format!(
                "tool {name}: 'input_schema' must be an object, not {}",
                json_type(v)
            )));
        }
    };
    Ok(json!({
        "type": "function",
        "function": { "name": name, "description": description, "parameters": parameters },
    }))
}

/// `tool_choice` as the chat path's: `auto` and `none` as they are, `any` and
/// `tool` (a call forced) as `required`, which the chat path refuses by name.
fn tool_choice(v: &Value) -> Result<Value, ApiError> {
    match v.get("type").and_then(Value::as_str) {
        Some(kind @ ("auto" | "none")) => Ok(json!(kind)),
        Some("any" | "tool") => Ok(json!("required")),
        _ => Err(invalid(format!(
            "tool_choice must be an object whose type is auto, any, tool or none, not {v}"
        ))),
    }
}

// ---------------------------------------------------------------- the chat path

/// A converted request after the chat path's own steps, in their order on
/// `/v1/chat/completions`: its sampling fields, its rendered prompt and the
/// prompt's ids, and the output parser of the tool-call markup the template
/// teaches.
struct Plan {
    p: GenParams,
    ids: Vec<u32>,
    prompt: Value,
    parser: ChatParser,
}

fn plan(state: &State, chat: &Map<String, Value>) -> Result<Plan, ApiError> {
    let mut p = gen_params(state, chat)?;
    // Anthropic has no `reasoning_format`: the chat path's default, which
    // splits the think span off into the thinking block.
    let format = ReasoningFormat::from_request(None).map_err(invalid)?;
    let (text, ids) = prompt_of(state, chat)?;
    let tools = tool_scan(state, chat)?;
    gate_reasoning_budget(&mut p, &text);
    let parser = ChatParser::with_tools(&text, format, tools);
    Ok(Plan {
        p,
        ids,
        prompt: Value::String(text),
        parser,
    })
}

/// The prompt a converted request runs, rendered by the chat template, and its
/// ids: where both endpoints take them from.
fn prompt_of(state: &State, chat: &Map<String, Value>) -> Result<(String, Vec<u32>), ApiError> {
    let text = render_chat(state, chat)?;
    let ids = state.tok.encode(&text);
    Ok((text, ids))
}

/// The answer's names: `msg_<32 hex>`, a tool use's `toolu_<index>_<16 hex>`
/// (the message's first 16), and the request's `model` (the server's alias
/// when it names none), as the chat path echoes it.
struct Ids {
    id: String,
    model: String,
}

impl Ids {
    fn new(state: &State, b: &Map<String, Value>) -> Self {
        Ids {
            id: format!("msg_{}", state.random_id()),
            model: b
                .get("model")
                .and_then(Value::as_str)
                .map_or_else(|| state.alias.clone(), str::to_owned),
        }
    }

    fn tool_id(&self, index: usize) -> String {
        let nonce = self.id.strip_prefix("msg_").unwrap_or(&self.id);
        format!("toolu_{index}_{}", nonce.get(..16).unwrap_or(nonce))
    }
}

/// The non-stream answer.
fn whole(
    state: &State,
    req: &Request,
    w: &mut TcpStream,
    ids: &Ids,
    plan: Plan,
) -> io::Result<bool> {
    let Plan {
        p,
        ids: prompt_ids,
        prompt,
        mut parser,
    } = plan;
    let o = match run_gen(state, &prompt_ids, prompt, &p, &mut |_, _| Ok(())) {
        Err(e) => return send_error(w, req, &e),
        Ok((Err(e), _)) => return send_error(w, req, &engine_error(&e)),
        Ok((Ok(o), _)) => o,
    };
    if let Err(e) = parser
        .try_push(&o.content)
        .and_then(|_| parser.try_finish())
    {
        return send_error(w, req, &tool_markup_error(&e));
    }
    match message_object(ids, &o, parser.message()) {
        Ok(v) => send_json(w, req, 200, &v),
        Err(e) => send_error(w, req, &e),
    }
}

/// The message object: llama-server's `to_json_anthropic`.
fn message_object(ids: &Ids, o: &Outcome, m: &Message) -> Result<Value, ApiError> {
    let mut content = Vec::with_capacity(2 + m.calls.len());
    if !m.reasoning.is_empty() {
        content.push(json!({ "type": "thinking", "thinking": m.reasoning, "signature": "" }));
    }
    if !m.content.is_empty() {
        content.push(json!({ "type": "text", "text": m.content }));
    }
    for c in &m.calls {
        content.push(tool_use(ids, c)?);
    }
    Ok(json!({
        "id": ids.id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": ids.model,
        "stop_reason": stop_reason(o, m),
        "stop_sequence": stop_sequence(o),
        "usage": usage(&o.timings, o.timings.predicted_n),
    }))
}

/// A parsed call as a `tool_use` block. The parsers write an object's JSON
/// text; text that does not parse as one is the server's 500, never `{}`.
fn tool_use(ids: &Ids, c: &ToolCall) -> Result<Value, ApiError> {
    let input: Value = serde_json::from_str(&c.arguments)
        .ok()
        .filter(Value::is_object)
        .ok_or_else(|| {
            engine_error(&format!(
                "tool call {}: its arguments are not a JSON object: {}",
                c.name, c.arguments
            ))
        })?;
    Ok(json!({ "type": "tool_use", "id": ids.tool_id(c.index), "name": c.name, "input": input }))
}

fn stop_reason(o: &Outcome, m: &Message) -> &'static str {
    match o.stop {
        StopKind::Limit => "max_tokens",
        StopKind::Word => "stop_sequence",
        StopKind::Eos if m.calls.is_empty() => "end_turn",
        StopKind::Eos => "tool_use",
    }
}

fn stop_sequence(o: &Outcome) -> Value {
    match o.stop {
        StopKind::Word => json!(o.stopping_word),
        StopKind::Eos | StopKind::Limit => Value::Null,
    }
}

/// `usage` with `output` generated ids: the prompt's kept ids, the rest, and
/// the output.
fn usage(t: &Timings, output: usize) -> Value {
    json!({
        "cache_read_input_tokens": t.cache_n,
        "input_tokens": t.n_prompt.saturating_sub(t.cache_n),
        "output_tokens": output,
    })
}

// ---------------------------------------------------------------- the stream

/// The stream's content blocks: each starts at the next index and stops when
/// the next one starts or the message ends. A thinking block's stop follows an
/// empty `signature_delta`, as the API requires of every thinking block.
#[derive(Default)]
struct Blocks {
    /// Blocks started so far; the open one's index is one less.
    started: usize,
    open: Option<Block>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Thinking,
    Text,
    ToolUse,
}

impl Blocks {
    /// The events one step of the output parser adds, in the message's order:
    /// thinking, text, then each call whole.
    fn push(&mut self, ids: &Ids, d: &Message) -> Vec<Value> {
        let mut ev = Vec::new();
        if !d.reasoning.is_empty() {
            self.enter(
                Block::Thinking,
                json!({ "type": "thinking", "thinking": "" }),
                &mut ev,
            );
            ev.push(self.delta(json!({ "type": "thinking_delta", "thinking": d.reasoning })));
        }
        if !d.content.is_empty() {
            self.enter(Block::Text, json!({ "type": "text", "text": "" }), &mut ev);
            ev.push(self.delta(json!({ "type": "text_delta", "text": d.content })));
        }
        for c in &d.calls {
            self.stop(&mut ev);
            let block = json!({
                "type": "tool_use", "id": ids.tool_id(c.index), "name": c.name, "input": {},
            });
            self.start(Block::ToolUse, block, &mut ev);
            ev.push(self.delta(json!({ "type": "input_json_delta", "partial_json": c.arguments })));
        }
        ev
    }

    /// Keeps the open block when it is a `kind` one, else stops it and starts
    /// `block`.
    fn enter(&mut self, kind: Block, block: Value, ev: &mut Vec<Value>) {
        if self.open != Some(kind) {
            self.stop(ev);
            self.start(kind, block, ev);
        }
    }

    fn start(&mut self, kind: Block, block: Value, ev: &mut Vec<Value>) {
        ev.push(
            json!({ "type": "content_block_start", "index": self.started, "content_block": block }),
        );
        self.started += 1;
        self.open = Some(kind);
    }

    /// A delta of the open block.
    fn delta(&self, delta: Value) -> Value {
        json!({ "type": "content_block_delta", "index": self.started - 1, "delta": delta })
    }

    /// Stops the open block, if any.
    fn stop(&mut self, ev: &mut Vec<Value>) {
        let Some(kind) = self.open.take() else {
            return;
        };
        let index = self.started - 1;
        if kind == Block::Thinking {
            ev.push(json!({
                "type": "content_block_delta", "index": index,
                "delta": { "type": "signature_delta", "signature": "" },
            }));
        }
        ev.push(json!({ "type": "content_block_stop", "index": index }));
    }
}

/// `message_start`: the message with no content yet, `usage` the prompt's.
fn message_start(ids: &Ids, t: &Timings) -> Value {
    json!({
        "type": "message_start",
        "message": {
            "id": ids.id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": ids.model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": usage(t, 0),
        },
    })
}

/// One frame: `event: <the data's type>` and `data: <json>`.
fn event(s: &mut EventStream<'_>, v: &Value) -> io::Result<()> {
    let kind = v["type"]
        .as_str()
        .expect("every event this module builds names its type");
    s.send(format!("event: {kind}\ndata: {v}\n\n").as_bytes())
}

/// The answer's writer: nothing until the first event, then the event stream.
struct Out<'r, 'w> {
    req: &'r Request,
    w: Option<&'w mut TcpStream>,
    stream: Option<EventStream<'w>>,
}

impl<'w> Out<'_, 'w> {
    fn open(&mut self) -> io::Result<&mut EventStream<'w>> {
        if self.stream.is_none() {
            let w = self
                .w
                .take()
                .ok_or_else(|| io::Error::other("stream writer taken"))?;
            self.stream = Some(EventStream::start(w, self.req, 200, "text/event-stream")?);
        }
        self.stream
            .as_mut()
            .ok_or_else(|| io::Error::other("no stream"))
    }
}

/// The stream. The head goes out with the first event; `message_start` with
/// the first text, whose timings carry the prompt's counts (an empty text event
/// comes between the calls of a prompt run a call a round, before them). An
/// error before the head is the request's answer; after it, an `error` event.
fn stream(
    state: &State,
    req: &Request,
    w: &mut TcpStream,
    ids: &Ids,
    plan: Plan,
) -> io::Result<bool> {
    let Plan {
        p,
        ids: prompt_ids,
        prompt,
        mut parser,
    } = plan;
    let mut out = Out {
        req,
        w: Some(w),
        stream: None,
    };
    let mut blocks = Blocks::default();
    let mut started = false;
    // Markup that does not parse stops the generation (the sink fails) and
    // ends the stream with an error event instead of a dropped connection.
    let mut markup = None;
    let r = {
        let mut sink = |ev: Event<'_>, _slot: usize| -> io::Result<()> {
            let s = out.open()?;
            let Event::Text(text, t) = ev else {
                return Ok(());
            };
            if text.is_empty() {
                return Ok(());
            }
            if !started {
                started = true;
                event(s, &message_start(ids, t))?;
            }
            let d = parser.try_push(text).map_err(|e| {
                markup = Some(e);
                io::Error::other("tool-call markup does not parse")
            })?;
            blocks.push(ids, &d).iter().try_for_each(|v| event(s, v))
        };
        run_gen(state, &prompt_ids, prompt, &p, &mut sink).map(|(o, _)| o)
    };
    let r = match markup {
        Some(e) => Err(tool_markup_error(&e)),
        None => r,
    };
    let mut s = match (out.stream, out.w) {
        (Some(s), _) => s,
        (None, Some(w)) => match &r {
            Err(e) => return send_error(w, req, e),
            Ok(Err(e)) => return send_error(w, req, &engine_error(e)),
            Ok(Ok(_)) => EventStream::start(w, req, 200, "text/event-stream")?,
        },
        (None, None) => return Ok(false),
    };
    match r {
        Ok(Ok(o)) => {
            if !started {
                event(&mut s, &message_start(ids, &o.timings))?;
            }
            match parser.try_finish() {
                Ok(d) => {
                    let mut ev = blocks.push(ids, &d);
                    blocks.stop(&mut ev);
                    ev.push(json!({
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason(&o, parser.message()),
                            "stop_sequence": stop_sequence(&o),
                        },
                        "usage": { "output_tokens": o.timings.predicted_n },
                    }));
                    ev.push(json!({ "type": "message_stop" }));
                    ev.iter().try_for_each(|v| event(&mut s, v))?;
                }
                Err(e) => event(&mut s, &envelope(&tool_markup_error(&e)))?,
            }
        }
        Ok(Err(GenError::Client(e))) => return Err(e),
        Ok(Err(e)) => event(&mut s, &envelope(&engine_error(&e)))?,
        Err(e) => event(&mut s, &envelope(&e))?,
    }
    let reusable = s.reusable();
    s.finish()?;
    Ok(reusable)
}

// ---------------------------------------------------------------- errors

/// Anthropic's error envelope around the OpenAI path's error object.
fn envelope(e: &ApiError) -> Value {
    json!({ "type": "error", "error": error_body(e.code, e.kind, &e.message)["error"] })
}

/// An error answer: the envelope, with the status and the `Retry-After` the
/// OpenAI path sends for it.
fn send_error(w: &mut TcpStream, req: &Request, e: &ApiError) -> io::Result<bool> {
    let retry = [("Retry-After", RETRY_AFTER_SECS.to_owned())];
    let headers: &[(&str, String)] = if e.retry_after { &retry } else { &[] };
    http::respond(
        w,
        req,
        e.code,
        JSON,
        headers,
        envelope(e).to_string().as_bytes(),
    )?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::normalize_billing_header;

    /// The stamp after `cch=` is written as `fffff` when it is five characters
    /// before a `;` in a billing header; any other text, a header without a
    /// stamp, and a stamp of another length are returned as they are.
    #[test]
    fn billing_header_stamp_is_normalised_only_in_its_place() {
        let head = "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; ";
        assert_eq!(
            normalize_billing_header(&format!("{head}cch=a5145;You are Claude Code.")),
            format!("{head}cch=fffff;You are Claude Code.")
        );
        for kept in [
            "You are Claude Code. cch=a5145;".to_owned(),
            format!("{head}You are Claude Code."),
            format!("{head}cch=a514;You are"),
            format!("{head}cch=a51456;You are"),
            format!("{head}cch=a5145"),
            format!(" {head}cch=a5145;x"),
        ] {
            assert_eq!(normalize_billing_header(&kept), kept);
        }
    }
}
