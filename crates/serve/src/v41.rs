//! DeepSeek's official V4.1 prompt encoding: the chat template the V4.1 seat
//! serves, and the request's shape the template reads.
//!
//! The template renders the prompt the way DeepSeek's own encoder renders
//! it: the system token opens the first system message, the reasoning effort
//! opens a thinking-mode prompt, the user and tool turns carry the user
//! token, an assistant turn carries the assistant token, its thinking markers
//! and its end-of-sentence token, and the tool prompt and the response
//! format join the first system message.
//!
//! Two rules of the encoding are not the template's to write: the tool
//! results of a turn answer the calls in the order the model asked them
//! (`order_tool_results`), and a tool call's arguments are read as the
//! object of the call's parameters (`tool_arguments`). The server applies
//! both to the request before the template renders it, through the family's
//! request preparation ([`prepare`]).

use serde_json::{Map, Value};

use crate::ChatRequest;

/// The official V4.1 chat template.
pub const TEMPLATE: &str = include_str!("templates/v41-official.jinja");

/// The V4.1 request preparation: the request as DeepSeek's own encoder reads
/// it.
///
/// The tool results of a turn answer the calls in the order the model asked
/// them, a tool call's arguments are the object of the call's parameters, and
/// the request's tool choice and response format reach the template, which
/// writes them into the prompt.
///
/// A tool call the encoding cannot read is refused by name: the field and
/// the message it is in.
pub fn prepare(request: &mut ChatRequest<'_>) -> Result<(), String> {
    tool_arguments(request.messages)?;
    order_tool_results(request.messages);
    for (name, value) in request_vars(request.body) {
        let Some(value) = value else {
            continue;
        };
        let mut value = value.clone();
        request.mark(&mut value);
        request.vars.insert(name.into(), value);
    }
    Ok(())
}

/// The request's tool choice and response format, the two fields the official
/// template reads and the other templates do not.
///
/// The tool choice says whether the tool prompt is rendered and whether the
/// prompt ends in a tool-call block; the response format is written into the
/// first system message.
fn request_vars(body: &Map<String, Value>) -> [(&'static str, Option<&Value>); 2] {
    [
        ("tool_choice", body.get("tool_choice")),
        ("response_format", body.get("response_format")),
    ]
}

/// The tool call arguments of the messages, as the official encoding reads
/// them: the JSON object of the call's parameters, which the template parses.
///
/// The official encoding parses a call's arguments as a JSON object; text
/// that is not one becomes a single `arguments` parameter holding the text.
///
/// A `tool_calls` that is not an array, and a call whose `function` is not
/// an object, are refused by name: the encoding reads neither, and the
/// template would walk them.
fn tool_arguments(messages: &mut [Value]) -> Result<(), String> {
    for (i, message) in messages.iter_mut().enumerate() {
        let calls = match message.get_mut("tool_calls") {
            None | Some(Value::Null) => continue,
            Some(Value::Array(calls)) => calls,
            Some(_) => return Err(format!("messages[{i}].tool_calls must be an array")),
        };
        for (j, call) in calls.iter_mut().enumerate() {
            let function = call
                .get_mut("function")
                .and_then(Value::as_object_mut)
                .ok_or_else(|| {
                    format!("messages[{i}].tool_calls[{j}].function must be an object")
                })?;
            let text = match function.get("arguments") {
                Some(Value::String(text)) => text.clone(),
                // An object is the object of the parameters already.
                Some(Value::Object(_)) => continue,
                Some(other) => other.to_string(),
                None => String::new(),
            };
            let arguments = match serde_json::from_str::<Map<String, Value>>(&text) {
                Ok(object) => Value::Object(object),
                Err(_) => Value::Object(Map::from_iter([(
                    "arguments".to_owned(),
                    Value::String(text),
                )])),
            };
            function.insert("arguments".into(), Value::String(arguments.to_string()));
        }
    }
    Ok(())
}

/// The tool results of `messages`, ordered the way the official encoding
/// orders them: the results that answer the calls of one assistant message
/// are ordered by the order of those calls.
///
/// A result that answers no call of that message sorts first, and a result
/// without a call id keeps its place.
fn order_tool_results(messages: &mut [Value]) {
    let mut call_ids: Vec<String> = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        match role_of(&messages[index]) {
            "assistant" => {
                if let Some(calls) = messages[index].get("tool_calls").and_then(Value::as_array)
                    && !calls.is_empty()
                {
                    call_ids = calls
                        .iter()
                        .map(|call| {
                            call.get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned()
                        })
                        .collect();
                }
                index += 1;
            }
            "user" | "tool" => {
                let start = index;
                while index < messages.len() && matches!(role_of(&messages[index]), "user" | "tool")
                {
                    index += 1;
                }
                let slots: Vec<usize> = (start..index)
                    .filter(|&at| call_id_of(&messages[at]).is_some())
                    .collect();
                if slots.len() > 1 && !call_ids.is_empty() {
                    let mut ordered: Vec<Value> =
                        slots.iter().map(|&at| messages[at].clone()).collect();
                    ordered.sort_by_key(|message| {
                        let id = call_id_of(message).unwrap_or_default();
                        call_ids.iter().position(|call| *call == id).unwrap_or(0)
                    });
                    for (&at, message) in slots.iter().zip(ordered) {
                        messages[at] = message;
                    }
                }
            }
            _ => index += 1,
        }
    }
}

/// The role of a message.
fn role_of(message: &Value) -> &str {
    message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default()
}

/// The id of the tool call a tool result answers.
fn call_id_of(message: &Value) -> Option<&str> {
    message.get("tool_call_id").and_then(Value::as_str)
}
