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
//! and `usage`; a stream sends one chunk a text event, then a chunk with
//! each choice's `finish_reason`, the last with `usage`, then `data: [DONE]`,
//! and opens with no chunk
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
//! `n` asks for that many choices of the prompt ([`take_choices`]): choice
//! `i` samples with the request's seed plus `i`, `index` is its place, and
//! the usage counts the prompt once and every choice's completion tokens.
//!
//! The request refuses by name what it cannot honour: a `prompt` array of
//! several prompts (llama-server, as OpenAI, answers one choice per prompt;
//! this server takes one prompt a request — an array that mixes
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
    ApiError, Client, CompletionPlan, State, body, chat_input, choice_params, completion_plan,
    finish_stream, gen_body, get_i, invalid, progress, run_choices, run_whole_choices, send_error,
    send_json, sse, take_choices, unix_now, usage,
};
use crate::genloop::{Event, Outcome};
use crate::http::{EventStream, Request};

/// `POST /v1/completions`: the completion path's generation, OpenAI's answer.
pub(super) fn completions(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let parsed = gen_body(req).and_then(|mut b| {
        openai_fields(&b)?;
        let n = take_choices(state, &mut b)?;
        Ok((completion_plan(state, &b)?, n, b))
    });
    let (plan, n, b) = match parsed {
        Ok(x) => x,
        Err(e) => return send_error(w, req, &e),
    };
    let CompletionPlan {
        p,
        input,
        prompt,
        return_tokens: _,
    } = plan;
    let ps = choice_params(state, &p, n);
    let ids = Ids::new(state, &b);
    if !p.stream {
        return match run_whole_choices(state, &req.path, &input, prompt, &ps, w)? {
            Err(e) => send_error(w, req, &e),
            Ok(all) => {
                let choices = all
                    .iter()
                    .enumerate()
                    .map(|(i, (o, _))| choice(i, &o.content, json!(o.stop.finish_reason())));
                send_json(w, req, 200, &last(&ids, choices.collect(), &all))
            }
        };
    }
    let peer = w.try_clone()?;
    let mut stream: Option<EventStream<'_>> = None;
    let mut w_opt = Some(w);
    let tpt = p.timings_per_token;
    let r = {
        let ids = &ids;
        let mut sink = |ev: Event<'_>, i: usize, _slot: usize| -> io::Result<()> {
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
                    let mut v = ids.chunk(json!([choice(i, "", Value::Null)]));
                    v["prompt_progress"] = progress(t);
                    v
                }
                Event::Text(text, t) => {
                    let mut v = ids.chunk(json!([choice(i, text, Value::Null)]));
                    if tpt {
                        v["timings"] = t.to_json();
                    }
                    v
                }
            };
            sse(s, &v)
        };
        run_choices(
            state,
            &req.path,
            &input,
            prompt,
            &ps,
            Client::of(peer),
            &mut sink,
        )
    };
    finish_stream(req, stream, w_opt, r, |s, all| {
        // One frame a choice with its `finish_reason` (its text already
        // sent), the last frame with the usage and the timings.
        let mut frames: Vec<Value> = all
            .iter()
            .enumerate()
            .map(|(i, (o, _))| choice(i, "", json!(o.stop.finish_reason())))
            .collect();
        let end = frames.pop().map(|c| last(&ids, vec![c], all));
        for c in frames {
            sse(s, &ids.chunk(json!([c])))?;
        }
        if let Some(end) = end {
            sse(s, &end)?;
        }
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
            "'prompt' holds {} prompts: this server takes one prompt a request",
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

    /// One frame of the answer: its `choices` and the fields every frame
    /// carries.
    fn chunk(&self, choices: Value) -> Value {
        json!({
            "choices": choices,
            "created": self.created,
            "model": self.model,
            "object": "text_completion",
            "id": self.id,
        })
    }
}

/// Choice `index` of a frame: `text`, and `finish` `null` until its last
/// frame.
fn choice(index: usize, text: &str, finish: Value) -> Value {
    json!({
        "text": text,
        "index": index,
        "logprobs": null,
        "finish_reason": finish,
    })
}

/// The whole answer, or the stream's last frame, over `choices`: as
/// llama-server's `to_json_oaicompat` writes the final form, with the usage of
/// every choice ([`usage`]), the first choice's timings and `logprobs` null
/// (this route serves no probabilities).
fn last(ids: &Ids, choices: Vec<Value>, all: &[(Outcome, usize)]) -> Value {
    let mut v = ids.chunk(Value::Array(choices));
    v["usage"] = usage(all);
    if let Some((o, _)) = all.first() {
        v["timings"] = o.timings.to_json();
    }
    v
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::super::Server;
    use super::super::testserve::{mock_config, roundtrip, spawn};
    use crate::sched::SlotConfig;
    use crate::{MockEngine, MockTokenizer, ScriptedEngine, Tokenizer};

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
            assert!(text.contains("one prompt a request"), "{text}");
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

    /// The text of each choice of an answer on `path`, whole or streamed, in
    /// `index` order (every choice's frames carry its index, and a choice
    /// ends once with a `finish_reason`), and its `usage`.
    fn choices(addr: std::net::SocketAddr, path: &str, body: &Value) -> (Vec<String>, Value) {
        let (status, text) = post(addr, path, &body.to_string());
        assert_eq!(status, 200, "{path}: {text}");
        let piece = |c: &Value| {
            let t = if path.contains("chat") {
                c.get("message").unwrap_or(&c["delta"])["content"].as_str()
            } else {
                c["text"].as_str()
            };
            t.unwrap_or_default().to_owned()
        };
        if body["stream"] != true {
            let v: Value = serde_json::from_str(&text).expect("the answer is JSON");
            let all = v["choices"].as_array().expect("choices");
            for (i, c) in all.iter().enumerate() {
                assert_eq!(c["index"], json!(i), "{path}: {text}");
            }
            return (all.iter().map(piece).collect(), v["usage"].clone());
        }
        let mut texts: Vec<String> = Vec::new();
        let mut ended: Vec<usize> = Vec::new();
        let mut usage = Value::Null;
        for e in text.split("\n\n").filter_map(|f| f.strip_prefix("data: ")) {
            if e == "[DONE]" {
                continue;
            }
            let v: Value = serde_json::from_str(e).expect("chunk JSON");
            if !v["usage"].is_null() {
                usage = v["usage"].clone();
            }
            for c in v["choices"].as_array().expect("choices") {
                let i = usize::try_from(c["index"].as_u64().expect("an index")).expect("usize");
                if texts.len() <= i {
                    texts.resize(i + 1, String::new());
                }
                texts[i].push_str(&piece(c));
                if !c["finish_reason"].is_null() {
                    ended.push(i);
                }
            }
        }
        ended.sort_unstable();
        assert_eq!(
            ended,
            (0..texts.len()).collect::<Vec<_>>(),
            "{path}: {text}"
        );
        (texts, usage)
    }

    /// `n` choices of one prompt, on the chat and the text completion routes,
    /// whole and streamed, over one slot (the choices queue) and two (they
    /// run together): choice `i` is the request of one choice seeded with the
    /// request's seed plus `i`, so sampled choices differ, and the usage counts
    /// the prompt once and every choice's tokens.
    #[test]
    fn n_choices_are_the_requests_of_the_following_seeds() {
        let request = |path: &str, n: u64, seed: u64, stream: bool| {
            let mut b = json!({
                "max_tokens": 8, "temperature": 1.5, "seed": seed, "n": n, "stream": stream,
                "stream_options": {"include_usage": true},
            });
            let text = "abcabdabeabfabgab";
            if path.contains("chat") {
                b["messages"] = json!([{"role": "user", "content": text}]);
            } else {
                b["prompt"] = json!(text);
            }
            b
        };
        for slots in [1, 2] {
            let server = Server::bind_with(
                "127.0.0.1:0",
                Box::new(MockEngine::new(256).with_slots(slots)),
                mock_config(),
                SlotConfig {
                    parallel: slots,
                    ..SlotConfig::default()
                },
            )
            .expect("bind");
            let addr = server.local_addr().expect("addr");
            let _started = server.start().expect("start");
            for path in ["/v1/chat/completions", "/v1/completions"] {
                let alone: Vec<(String, Value)> = (0..3)
                    .map(|i| {
                        let (mut texts, usage) =
                            choices(addr, path, &request(path, 1, 7 + i, false));
                        (texts.remove(0), usage)
                    })
                    .collect();
                let texts: Vec<&String> = alone.iter().map(|(t, _)| t).collect();
                assert!(texts.iter().any(|t| *t != texts[0]), "{path}: {texts:?}");
                let tokens: u64 = alone
                    .iter()
                    .map(|(_, u)| u["completion_tokens"].as_u64().expect("a count"))
                    .sum();
                for stream in [false, true] {
                    let (got, usage) = choices(addr, path, &request(path, 3, 7, stream));
                    assert_eq!(got.iter().collect::<Vec<_>>(), texts, "{path} {stream}");
                    assert_eq!(usage["completion_tokens"], json!(tokens), "{path} {usage}");
                    assert_eq!(usage["prompt_tokens"], alone[0].1["prompt_tokens"]);
                }
            }
        }
    }

    /// `n` is a count of 1 up to the slots and the requests that may wait
    /// beside them (the choices queue together): 0, a negative count or one
    /// past that is a 400 naming it on both OpenAI routes, and `/completion`,
    /// which answers one generation, refuses `n > 1` by name.
    #[test]
    fn n_outside_one_to_the_slots_and_the_queue_is_refused_by_name() {
        let (addr, _, _) = scripted();
        let most = crate::MAX_CONNECTIONS;
        for path in ["/v1/chat/completions", "/v1/completions"] {
            for n in [0, -1, most as i64 + 1] {
                let body = format!(
                    r#"{{"messages":[{{"role":"user","content":"ab"}}],"prompt":"ab","max_tokens":1,"n":{n}}}"#
                );
                let (status, text) = post(addr, path, &body);
                assert_eq!(status, 400, "{path} n={n}: {text}");
                assert!(
                    text.contains(&format!("n must be between 1 and {most}")),
                    "{text}"
                );
            }
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":"ab"}}],"prompt":"ab","max_tokens":1,"n":{most}}}"#
            );
            let (status, text) = post(addr, path, &body);
            assert_eq!(status, 200, "{path} n={most}: {text}");
        }
        let (status, text) = post(addr, "/completion", r#"{"prompt":"ab","n":2}"#);
        assert_eq!(status, 400, "{text}");
        assert!(text.contains("n is not supported"), "{text}");
    }
}
