//! OpenAI's text completion API and the chat token count: `POST
//! /v1/completions`, and `POST /chat/completions/input_tokens` and
//! `/v1/chat/completions/input_tokens`, as llama-server serves them
//! (`post_completions_oai` over the `/completion` path, `to_json_oaicompat`
//! for the answer and the stream chunks; `handle_count_tokens`).
//!
//! `/v1/completions` runs the completion path's own steps
//! ([`completion_plan`], as `/completion` does): the same prompt forms, the
//! sampling fields, the generation and the stop rules. Only the answer's
//! shape is this module's: `object: "text_completion"`, `id` `cmpl-…`,
//! `created`, `model`, `choices: [{text, index, logprobs, finish_reason}]`
//! and `usage`; a stream sends one chunk a text event, then the chunk with
//! `finish_reason` and `usage`, then `data: [DONE]`, and opens with no chunk
//! of its own (OpenAI's text completion stream opens with text, unlike the
//! chat stream's role delta). `finish_reason` is llama-server's rule: `stop`
//! on EOS or a stop word, `length` at the limit.
//!
//! Divergences from llama-server, each keeping the shape this crate's chat
//! path already answers: `system_fingerprint` is not sent (the chat path
//! sends none either), `model` echoes the request's `model` when it names
//! one (llama-server answers its alias whatever the request names), and
//! `id` is `cmpl-…` where llama-server reuses the chat path's `chatcmpl-…`
//! prefix — OpenAI's own spelling, one no chat id can collide with.
//!
//! The request refuses by name what it cannot honour: a `prompt` array of
//! several prompts (llama-server, as OpenAI, answers one choice per prompt;
//! this server generates one completion a request — an array that mixes
//! strings and ids is one prompt, as `/completion` reads it, and a
//! one-element array is one prompt), `echo` other than `false` (llama-server
//! parses the field but never echoes the prompt in the answer), a non-empty
//! `suffix`, and `best_of > 1`. `logprobs` is refused through the shared
//! `gen_params`, here in its integer form. `return_tokens` is `/completion`'s
//! field: this shape carries no ids, as llama-server's own does not without
//! its `__verbose`.
//!
//! The token count routes answer `{"input_tokens": N}`, N the ids of the
//! prompt the same chat request renders ([`chat_input`], the same count the
//! Anthropic token counter answers): the template and the tokenizer the
//! generation uses, an image as its span, `max_tokens` not asked for.
//! llama-server adds `"object": "response.input_tokens"`; this server keeps
//! the Anthropic counter's shape, one shape for the same count on every
//! route.

use std::io;
use std::net::TcpStream;

use serde_json::{Map, Value, json};

use super::{
    ApiError, CompletionPlan, State, body, chat_input, completion_plan, finish_stream, get_i,
    invalid, progress, run_gen, run_whole, send_error, send_json, sse, unix_now, usage,
};
use crate::genloop::{Event, Outcome};
use crate::http::{EventStream, Request};

/// `POST /v1/completions`: the completion path's generation, OpenAI's answer.
pub(super) fn completions(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let parsed = body(req).and_then(|b| {
        openai_fields(&b)?;
        Ok((completion_plan(state, &b)?, b))
    });
    let (plan, b) = match parsed {
        Ok(x) => x,
        Err(e) => return send_error(w, req, &e),
    };
    let CompletionPlan {
        p,
        input,
        prompt,
        return_tokens: _,
    } = plan;
    let ids = Ids::new(state, &b);
    if !p.stream {
        return match run_whole(state, &input, prompt, &p, w)? {
            Err(e) => send_error(w, req, &e),
            Ok((o, _)) => send_json(w, req, 200, &answer(&ids, &o, &o.content)),
        };
    }
    let mut stream: Option<EventStream<'_>> = None;
    let mut w_opt = Some(w);
    let tpt = p.timings_per_token;
    let r = {
        let ids = &ids;
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
            let v = match ev {
                Event::Prompt(t) => {
                    let mut v = ids.chunk("", Value::Null);
                    v["prompt_progress"] = progress(t);
                    v
                }
                Event::Text(text, t) => {
                    let mut v = ids.chunk(text, Value::Null);
                    if tpt {
                        v["timings"] = t.to_json();
                    }
                    v
                }
            };
            sse(s, &v)
        };
        run_gen(state, &input, prompt, &p, &mut sink)
    };
    finish_stream(req, stream, w_opt, r.map(|(o, _)| o), |s, o| {
        sse(s, &answer(&ids, o, ""))?;
        s.send(b"data: [DONE]\n\n")
    })
}

/// `POST /chat/completions/input_tokens` and its `/v1/` spelling:
/// `{"input_tokens": N}`, N the ids of the prompt the same chat request
/// renders. `max_tokens` is not asked for, as llama-server does not ask the
/// counting route for it.
pub(super) fn count_tokens(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let counted = body(req).and_then(|b| chat_input(state, &b));
    match counted {
        Ok((_, prompt)) => send_json(
            w,
            req,
            200,
            &json!({ "input_tokens": prompt.held.ids.len() }),
        ),
        Err(e) => send_error(w, req, &e),
    }
}

/// The OpenAI-only fields of a `/v1/completions` body, before the completion
/// path reads it (the module doc names each rule and its reference). An array
/// holds several prompts exactly where llama-server splits it: no element of
/// it is a number, the form it tokenizes in place.
fn openai_fields(b: &Map<String, Value>) -> Result<(), ApiError> {
    let set = |k: &str| b.get(k).filter(|v| !v.is_null());
    if let Some(Value::Array(a)) = b.get("prompt")
        && a.len() > 1
        && !a.iter().any(|x| x.as_i64().is_some())
    {
        return Err(invalid(format!(
            "'prompt' holds {} prompts: this server generates one completion a request",
            a.len()
        )));
    }
    if set("echo").is_some_and(|v| v.as_bool() != Some(false)) {
        return Err(invalid("echo is not supported by this server"));
    }
    if set("suffix").is_some_and(|v| v.as_str() != Some("")) {
        return Err(invalid("suffix is not supported by this server"));
    }
    if get_i(b, "best_of")?.is_some_and(|n| n > 1) {
        return Err(invalid("best_of is not supported by this server"));
    }
    Ok(())
}

/// The answer's names, the same in every chunk of one answer.
struct Ids {
    id: String,
    created: u64,
    model: String,
}

impl Ids {
    fn new(state: &State, b: &Map<String, Value>) -> Self {
        Ids {
            id: format!("cmpl-{}", state.random_id()),
            created: unix_now(),
            model: b
                .get("model")
                .and_then(Value::as_str)
                .map_or_else(|| state.alias.clone(), str::to_owned),
        }
    }

    /// One frame of the answer: `choices` around `text` (`finish` `null`
    /// until the last frame) and the fields every frame carries.
    fn chunk(&self, text: &str, finish: Value) -> Value {
        json!({
            "choices": [{
                "text": text,
                "index": 0,
                "logprobs": null,
                "finish_reason": finish,
            }],
            "created": self.created,
            "model": self.model,
            "object": "text_completion",
            "id": self.id,
        })
    }
}

/// The whole answer, or the stream's last frame (`text` empty there: the
/// text is already sent) — llama-server's `to_json_oaicompat`, the final
/// form: the usage and the timings of the outcome, `logprobs` null (this
/// server serves no probabilities).
fn answer(ids: &Ids, o: &Outcome, text: &str) -> Value {
    let mut v = ids.chunk(text, json!(o.stop.finish_reason()));
    v["usage"] = usage(o);
    v["timings"] = o.timings.to_json();
    v
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::super::testserve::{mock_config, roundtrip, spawn};
    use crate::{MockTokenizer, ScriptedEngine, Tokenizer};

    /// One HTTP/1.0 request over a fresh connection (the harness of the
    /// in-crate tests): the status and the whole body, a stream's included.
    fn post(addr: std::net::SocketAddr, path: &str, body: &str) -> (u16, String) {
        roundtrip(addr, "POST", path, &[], body)
    }

    /// A scripted engine's answer and this route's shapes: `hello` plus EOS
    /// from the prompt `ab`.
    fn scripted() -> (std::net::SocketAddr, usize, usize) {
        let (addr, _state, _ended) =
            spawn(Box::new(ScriptedEngine::new(64, "hello")), mock_config());
        let prompt = MockTokenizer.encode("ab").len();
        (addr, prompt, 6)
    }

    /// Ports `test_completion_with_openai_library`: the whole answer carries
    /// every field llama-server's `to_json_oaicompat` sends (its
    /// `system_fingerprint` excepted, as the chat path's), `finish_reason`
    /// `stop` on EOS, and the usage of the outcome.
    #[test]
    fn whole_answer_is_openai_text_completion() {
        let (addr, n_prompt, n_out) = scripted();
        let (status, body) = post(
            addr,
            "/v1/completions",
            r#"{"prompt":"ab","max_tokens":8,"model":"m"}"#,
        );
        assert_eq!(status, 200, "{body}");
        let v: Value = serde_json::from_str(&body).expect("the answer is JSON");
        let mut keys: Vec<&String> = v.as_object().expect("an object").keys().collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "choices", "created", "id", "model", "object", "timings", "usage"
            ],
            "the answer's fields, no more: {body}"
        );
        assert_eq!(v["object"], "text_completion");
        assert!(
            v["id"].as_str().is_some_and(|id| id.starts_with("cmpl-")),
            "the id: {body}"
        );
        assert!(v["created"].as_u64().is_some_and(|t| t > 0), "{body}");
        assert_eq!(
            v["model"], "m",
            "the request's model, as the chat path echoes it"
        );
        let c = &v["choices"][0];
        assert_eq!(c["text"], "hello");
        assert_eq!(c["index"], 0);
        assert!(c["logprobs"].is_null(), "no probabilities are served");
        assert_eq!(c["finish_reason"], "stop");
        assert_eq!(
            v["usage"],
            json!({
                "prompt_tokens": n_prompt,
                "completion_tokens": n_out,
                "total_tokens": n_prompt + n_out,
                "prompt_tokens_details": { "cached_tokens": 0 },
            }),
            "{body}"
        );
    }

    /// Ports `test_completion_stream_with_openai_library`: one chunk a text
    /// event, the progress frame a prompt progress (`return_progress`), the
    /// same `id`, `created`, `model` and `object` in every chunk, a text
    /// chunk's `timings` under `timings_per_token`, the last frame's empty
    /// `text` with `finish_reason` and `usage`, then `data: [DONE]` — and no
    /// opening frame, unlike the chat stream's role.
    #[test]
    fn stream_chunks_end_in_done() {
        let (addr, n_prompt, n_out) = scripted();
        let (status, body) = post(
            addr,
            "/v1/completions",
            r#"{"prompt":"ab","max_tokens":8,"stream":true,"return_progress":true,"timings_per_token":true}"#,
        );
        assert_eq!(status, 200, "{body}");
        let data: Vec<&str> = body
            .split("\n\n")
            .filter_map(|f| f.strip_prefix("data: "))
            .collect();
        assert_eq!(
            data.last(),
            Some(&"[DONE]"),
            "the stream ends in [DONE]: {body:?}"
        );
        let chunks: Vec<Value> = data[..data.len() - 1]
            .iter()
            .map(|e| serde_json::from_str(e).expect("chunk JSON"))
            .collect();
        let id = chunks[0]["id"].as_str().expect("an id").to_owned();
        assert!(id.starts_with("cmpl-"), "{id}");
        for c in &chunks[..chunks.len() - 1] {
            assert_eq!(c["id"], json!(id), "one id every chunk carries: {body:?}");
            assert_eq!(c["object"], "text_completion");
            assert_eq!(c["model"], "mock");
            assert_eq!(c["choices"][0]["index"], 0);
            assert!(c["choices"][0]["logprobs"].is_null());
            assert!(
                c["choices"][0]["finish_reason"].is_null(),
                "not until the last: {body:?}"
            );
        }
        let first = &chunks[0];
        assert_eq!(
            first["choices"][0]["text"], "",
            "the progress frame carries no text"
        );
        assert_eq!(
            first["prompt_progress"]["total"],
            json!(n_prompt),
            "the one progress frame, once the prompt is evaluated: {first}"
        );
        let text: String = chunks[1..chunks.len() - 1]
            .iter()
            .map(|c| c["choices"][0]["text"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(text, "hello", "one chunk a text event: {body:?}");
        assert!(
            chunks[1]
                .get("timings")
                .is_some_and(|t| t["predicted_n"].is_u64()),
            "timings_per_token: a text chunk carries the timings: {body:?}"
        );
        let last = chunks.last().expect("the last frame");
        assert_eq!(last["id"], json!(id), "{body:?}");
        assert_eq!(last["object"], "text_completion");
        assert_eq!(last["choices"][0]["text"], "", "the text is already sent");
        assert_eq!(last["choices"][0]["finish_reason"], "stop");
        assert_eq!(last["usage"]["completion_tokens"], json!(n_out), "{last}");
        assert!(
            last.get("timings").is_some(),
            "the last frame carries the timings"
        );
    }

    /// `max_tokens` short of the script's end: the answer stops at the limit
    /// with `finish_reason` `length` (the `stop` side is the whole answer's
    /// test).
    #[test]
    fn finish_reason_is_length_at_the_limit() {
        let (addr, _, _) = scripted();
        let (status, body) = post(addr, "/v1/completions", r#"{"prompt":"ab","max_tokens":2}"#);
        assert_eq!(status, 200, "{body}");
        let v: Value = serde_json::from_str(&body).expect("the answer is JSON");
        assert_eq!(v["choices"][0]["text"], "he");
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["usage"]["completion_tokens"], 2, "{body}");
    }

    /// Where llama-server splits a `prompt` array into one prompt an element
    /// (no element a number) and answers one choice per prompt, this server
    /// refuses the several-prompt form by name; the one-prompt forms it keeps
    /// — a one-element array, the mixed in-place form, ids alone, a string —
    /// are one prompt each, as `/completion` reads them.
    #[test]
    fn several_prompts_are_refused_by_name() {
        let (addr, _, _) = scripted();
        for prompt in [r#"["ab","cd"]"#, r#"[["ab"],["cd"]]"#, r#"[{},{}]"#] {
            let body = format!(r#"{{"prompt":{prompt},"max_tokens":1}}"#);
            let (status, text) = post(addr, "/v1/completions", &body);
            assert_eq!(status, 400, "{prompt}: {text}");
            assert!(text.contains("prompt"), "the 400 names prompt: {text}");
            assert!(text.contains("one completion"), "{text}");
        }
        for prompt in [r#""ab""#, r#"["ab"]"#, r#"["ab",7]"#, r#"[6,7]"#] {
            let body = format!(r#"{{"prompt":{prompt},"max_tokens":1}}"#);
            let (status, text) = post(addr, "/v1/completions", &body);
            assert_eq!(status, 200, "{prompt}: {text}");
        }
    }

    /// `logprobs` in the integer form this route takes (and every other form
    /// but `false`) is a 400 naming it through the shared `gen_params`; the
    /// logprobs round that adds it sees this pin.
    #[test]
    fn logprobs_is_refused_in_its_count_form() {
        let (addr, _, _) = scripted();
        let (status, text) = post(
            addr,
            "/v1/completions",
            r#"{"prompt":"ab","max_tokens":1,"logprobs":5}"#,
        );
        assert_eq!(status, 400, "{text}");
        assert!(text.contains("logprobs"), "{text}");
        for lp in ["false", "null"] {
            let body = format!(r#"{{"prompt":"ab","max_tokens":1,"logprobs":{lp}}}"#);
            let (status, text) = post(addr, "/v1/completions", &body);
            assert_eq!(status, 200, "logprobs {lp}: {text}");
        }
    }

    /// `echo`, `suffix` and `best_of` — none of which llama-server honours on
    /// this route (`echo` it parses and never echoes; `suffix` and `best_of`
    /// it does not read) — are refused by name; their neutral forms (`false`,
    /// empty, `1`) and absence are taken.
    #[test]
    fn echo_suffix_and_best_of_are_refused_by_name() {
        let (addr, _, _) = scripted();
        for field in [r#""echo":true"#, r#""suffix":" more""#, r#""best_of":2"#] {
            let body = format!(r#"{{"prompt":"ab","max_tokens":1,{field}}}"#);
            let (status, text) = post(addr, "/v1/completions", &body);
            assert_eq!(status, 400, "{field}: {text}");
            let name = field.split(':').next().expect("a name");
            let name = name.trim_matches('"');
            assert!(text.contains(name), "the 400 names {name}: {text}");
        }
        for neutral in [
            r#""echo":false"#,
            "\"suffix\":\"\"",
            r#""best_of":1"#,
            r#""echo":null"#,
        ] {
            let body = format!(r#"{{"prompt":"ab","max_tokens":1,{neutral}}}"#);
            let (status, text) = post(addr, "/v1/completions", &body);
            assert_eq!(status, 200, "{neutral}: {text}");
        }
    }

    /// Ports `test_chat_completions_token_count`: both routes answer
    /// `input_tokens` alone (llama-server adds an `object`; the Anthropic
    /// counter's shape is this crate's one shape for the same count), the
    /// same N, no `max_tokens` asked for — and N is what the same chat
    /// request generates from: the mock template's render of the messages,
    /// the ids the chat request's own usage counts.
    #[test]
    fn input_tokens_routes_count_the_chat_prompt() {
        let (addr, _, _) = scripted();
        let messages = r#"[{"role":"system","content":"Book"},{"role":"user","content":"What is the best book"}]"#;
        let rendered = "<system>\nBook<user>\nWhat is the best book<assistant>\n";
        let n = MockTokenizer.encode(rendered).len() as u64;
        assert!(n > 5, "the reference case's count is above its floor");
        for path in [
            "/chat/completions/input_tokens",
            "/v1/chat/completions/input_tokens",
        ] {
            let body = format!(r#"{{"messages":{messages}}}"#);
            let (status, text) = post(addr, path, &body);
            assert_eq!(status, 200, "{path}: {text}");
            let v: Value = serde_json::from_str(&text).expect("the count is JSON");
            let keys: Vec<&String> = v.as_object().expect("an object").keys().collect();
            assert_eq!(keys, ["input_tokens"], "{path}: {text}");
            assert_eq!(v["input_tokens"], json!(n), "{path}: {text}");
        }
        let (status, text) = post(
            addr,
            "/v1/chat/completions",
            &format!(r#"{{"messages":{messages},"max_tokens":1}}"#),
        );
        assert_eq!(status, 200, "{text}");
        let v: Value = serde_json::from_str(&text).expect("the chat answer is JSON");
        assert_eq!(v["usage"]["prompt_tokens"], json!(n), "{text}");
    }
}
