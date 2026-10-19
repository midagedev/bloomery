//! Gate: a tool call streams as it is written, in every markup the server parses.
//!
//! OpenAI's stream shape: the first `tool_calls` delta of a call carries its
//! `index`, `id`, `type` and `function.name` (with empty `arguments`), the
//! next ones `function.arguments` fragments, and the fragments concatenate to
//! the call's `arguments` — the string the non-streamed answer carries. A long
//! text parameter arrives in many fragments before the call closes, and a tail
//! that could still become a tag is held back. The cases are one `write` call
//! per markup (DSML in V4.1's spelling, Hermes, GLM, Qwen3-Coder) whose
//! `content` parameter is a long text with quotes, backslashes, a tab,
//! non-ASCII text and near-tags.
//!
//! The parser-level gates feed the same text under every cut and compare with
//! the parser that does not stream its calls; the divergence gates pin what a
//! call announced and then ended otherwise is (a named error, never a quietly
//! different call), and that the parser that does not stream is unchanged.

mod common;

use common::{V41_SPACED_TEMPLATE, fold_call_delta, post, start_templated};
use serde_json::{Value, json};
use serve::ScriptedEngine;
use serve::callstream::CallPiece;
use serve::dsml::{ChatParser, MarkupError, Message, Spelling, ToolCall, Tools};
use serve::glmxml::ArgTypes;
use serve::qwenxml::ParamKinds;
use serve::reasoning::ReasoningFormat;

const QWEN3: &str = include_str!("fixtures/qwen3-chat-template.jinja");
const QWEN38: &str = include_str!("fixtures/qwen38-chat-template.jinja");
const GLM5: &str = include_str!("fixtures/glm5-chat-template.jinja");

/// The long text of `content`: twelve lines, each with quotes, a backslash, a
/// tab, non-ASCII text and a near-tag of every markup (each shares a prefix
/// with a tag that ends a value or a call, and is not one).
fn content() -> String {
    (0..12)
        .map(|i| {
            format!(
                "line {i}: \"quoted\" \\ back\ttab 서울 ☃ </arg_value x> </｜DSML｜ parameter x> \
                 </parameter x> </tool_ca x>\n"
            )
        })
        .collect()
}

fn tools() -> Value {
    json!([{"type": "function", "function": {"name": "write", "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "mode": {"type": "integer"},
            "content": {"type": "string"},
        },
    }}}])
}

/// One markup: the template its server runs, what the model writes up to the
/// call, the call, and the parser tools that read it.
struct Case {
    label: &'static str,
    template: &'static str,
    /// Written before the call: closes the span the template's prompt opens.
    lead: &'static str,
    call: String,
    tools: Tools,
}

fn cases() -> Vec<Case> {
    let c = content();
    let hermes_content = serde_json::to_string(&c).expect("a string");
    vec![
        Case {
            label: "dsml",
            template: V41_SPACED_TEMPLATE,
            lead: "",
            call: format!(
                "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n\
                 <｜DSML｜ parameter name=\"path\" string=\"true\">/a.txt</｜DSML｜ parameter>\n\
                 <｜DSML｜ parameter name=\"mode\" string=\"false\">420</｜DSML｜ parameter>\n\
                 <｜DSML｜ parameter name=\"content\" string=\"true\">{c}</｜DSML｜ parameter>\n\
                 </｜DSML｜ invoke>\n</｜DSML｜ calls>"
            ),
            tools: Tools::Dsml(Spelling::Spaced),
        },
        Case {
            label: "hermes",
            template: QWEN3,
            lead: "",
            call: format!(
                "<tool_call>\n{{\"name\": \"write\", \"arguments\": {{\"path\": \"/a.txt\", \
                 \"mode\": 420, \"content\": {hermes_content}}}}}\n</tool_call>"
            ),
            tools: Tools::Hermes,
        },
        Case {
            label: "glm",
            template: GLM5,
            lead: "</think>",
            call: format!(
                "<tool_call>write<arg_key>path</arg_key><arg_value>/a.txt</arg_value>\
                 <arg_key>mode</arg_key><arg_value>420</arg_value>\
                 <arg_key>content</arg_key><arg_value>{c}</arg_value></tool_call>"
            ),
            tools: Tools::GlmXml(ArgTypes::of_tools(Some(&tools()))),
        },
        Case {
            label: "qwen",
            template: QWEN38,
            lead: "</think>",
            call: format!(
                "<tool_call>\n<function=write>\n<parameter=path>\n/a.txt\n</parameter>\n\
                 <parameter=mode>\n420\n</parameter>\n<parameter=content>\n{c}\n</parameter>\n\
                 </function>\n</tool_call>"
            ),
            tools: Tools::QwenXml(ParamKinds::of_tools(Some(&tools()))),
        },
    ]
}

fn body(stream: bool) -> Value {
    json!({
        "model": "m",
        "messages": [{"role": "user", "content": "write it"}],
        "temperature": 0,
        "tools": tools(),
        "stream": stream,
    })
}

/// The `tool_calls` deltas of a streamed chat body in order, and whether a
/// chunk with a `finish_reason` followed the last of them.
fn deltas(body: &str) -> (Vec<Value>, bool) {
    let mut out = Vec::new();
    let mut finished_after = false;
    for e in body.split("\n\n").filter_map(|e| e.strip_prefix("data: ")) {
        if e == "[DONE]" {
            continue;
        }
        let v: Value = serde_json::from_str(e).expect("chunk JSON");
        assert!(v.get("error").is_none(), "no error event: {e}");
        let Some(c) = v["choices"].get(0) else {
            continue;
        };
        for tc in c["delta"]["tool_calls"].as_array().into_iter().flatten() {
            out.push(tc.clone());
            finished_after = false;
        }
        if !c["finish_reason"].is_null() {
            finished_after = true;
        }
    }
    (out, finished_after)
}

/// A call streams in the OpenAI shape: opened once with its name, then its
/// arguments in fragments that concatenate to what the non-streamed answer
/// carries, the long text in many fragments before the call closes.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_call_streams_its_arguments_as_they_are_written() {
    for case in cases() {
        let label = case.label;
        let script = format!("{}{}", case.lead, case.call);
        let addr = start_templated(Box::new(ScriptedEngine::new(8192, &script)), case.template);

        let r = post(addr, "/v1/chat/completions", &body(false));
        assert_eq!(r.status, 200, "{label}: {}", r.body);
        let whole = r.json();
        let call = &whole["choices"][0]["message"]["tool_calls"][0]["function"];
        assert_eq!(call["name"], "write", "{label}: {whole}");
        let want = call["arguments"].as_str().expect("arguments").to_owned();

        let r = post(addr, "/v1/chat/completions", &body(true));
        assert_eq!(r.status, 200, "{label}: {}", r.body);
        let (deltas, finished_after) = deltas(&r.body);
        assert!(finished_after, "{label}: the finish chunk follows the call");
        let first = &deltas[0];
        assert!(
            first["id"]
                .as_str()
                .is_some_and(|i| i.starts_with("call_0_"))
                && first["type"] == "function"
                && first["index"] == 0
                && first["function"]["name"] == "write"
                && first["function"]["arguments"] == "",
            "{label}: the first delta names the call: {first}"
        );
        let fragments: Vec<&str> = deltas[1..]
            .iter()
            .map(|d| {
                assert!(
                    d.get("id").is_none() && d["function"].get("name").is_none(),
                    "{label}: a fragment carries only arguments: {d}"
                );
                d["function"]["arguments"].as_str().expect("text")
            })
            .collect();
        assert_eq!(fragments.concat(), want, "{label}: fragments == arguments");
        let args: Value = serde_json::from_str(&want).expect("arguments are JSON");
        assert_eq!(args["content"], content(), "{label}");
        assert_eq!(args["mode"], 420, "{label}");

        // The long text arrives in pieces, before the call closes: past the
        // fragment that opens `content`, many fragments carry its text.
        let mut seen = String::new();
        let mut opened = None;
        for (i, f) in fragments.iter().enumerate() {
            seen.push_str(f);
            if opened.is_none() && seen.contains(r#""content":""#) {
                opened = Some(i);
            }
        }
        let after = &fragments[opened.expect("content opens") + 1..];
        assert!(
            after.len() >= 100 && after.iter().map(|f| f.len()).sum::<usize>() >= content().len(),
            "{label}: the text streams: {} fragments after content opens",
            after.len()
        );
        let mut folded = Vec::new();
        for d in &deltas {
            fold_call_delta(&mut folded, d);
        }
        assert_eq!(folded[0]["function"]["arguments"], want, "{label}");
    }
}

/// The parser, with its calls streamed or not, over `pieces`: the message,
/// and the call pieces it reported.
fn run(
    tools: &Tools,
    streaming: bool,
    pieces: &[&str],
) -> Result<(Message, Vec<CallPiece>), MarkupError> {
    let mut p = ChatParser::with_tools("", ReasoningFormat::None, Some(tools.clone()));
    if streaming {
        p = p.streaming_calls();
    }
    let mut out = Vec::new();
    for piece in pieces {
        p.try_push(piece)?;
        out.extend(p.take_pieces());
    }
    p.try_finish()?;
    out.extend(p.take_pieces());
    Ok((p.message().clone(), out))
}

/// The whole text at once, a character at a time, split in two at every
/// character, and split in three at every pair of tag boundaries.
fn every_chunking(raw: &str) -> Vec<Vec<&str>> {
    let mut out = vec![vec![raw]];
    out.push(
        raw.char_indices()
            .map(|(i, c)| &raw[i..i + c.len_utf8()])
            .collect(),
    );
    for (at, _) in raw.char_indices().skip(1) {
        out.push(vec![&raw[..at], &raw[at..]]);
    }
    let tags: Vec<usize> = raw
        .char_indices()
        .filter(|&(_, c)| c == '<' || c == '>')
        .map(|(i, c)| if c == '>' { i + 1 } else { i })
        .filter(|&i| i > 0 && i < raw.len())
        .step_by(7)
        .collect();
    for (x, &a) in tags.iter().enumerate() {
        for &b in &tags[x + 1..] {
            out.push(vec![&raw[..a], &raw[a..b], &raw[b..]]);
        }
    }
    out
}

/// Each call's pieces: announced once, then fragments; the call's name and its
/// fragments' concatenation, by index.
fn fold_pieces(pieces: &[CallPiece]) -> Vec<(String, String)> {
    let mut calls: Vec<(String, String)> = Vec::new();
    for piece in pieces {
        match piece {
            CallPiece::Open { index, name } => {
                assert_eq!(*index, calls.len(), "calls open once, in order");
                calls.push((name.clone(), String::new()));
            }
            CallPiece::Args { index, text } => {
                assert_eq!(index + 1, calls.len(), "a fragment extends the open call");
                calls[*index].1.push_str(text);
            }
        }
    }
    calls
}

fn closed(m: &Message) -> Vec<(String, String)> {
    m.calls
        .iter()
        .map(|c| (c.name.clone(), c.arguments.clone()))
        .collect()
}

/// Under every cut, the pieces of a streamed call concatenate to the call the
/// parser closes with, which is the call the parser that does not stream
/// reads; the rest of the message is the same too.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_every_cut_streams_the_arguments_the_call_closes_with() {
    let mut texts: Vec<(String, Tools, String)> = cases()
        .into_iter()
        .map(|c| (c.label.to_owned(), c.tools, format!("note.\n{}", c.call)))
        .collect();
    // Hermes' arguments are re-written compactly: spaces, escapes, numbers,
    // nesting and surrogate pairs all come out as `serde_json` writes them.
    texts.push((
        "hermes-json".to_owned(),
        Tools::Hermes,
        "<tool_call>\n { \"name\" : \"f\" , \"arguments\" : { \"p\" : \"/a\" , \"n\" : \
         [ 1 , 2.50 , -3e2 , 1E-3 , true , false , null , { \"deep\" : { \"x\" : \"y\" } } ] , \
         \"s\" : \"caf\\u00e9 \\/ \\ud83d\\ude00 tab\\there \\u0001 \\\" \\\\\" , \"e\" : { } , \
         \"l\" : [ ] } }\n</tool_call>\n<tool_call>{\"name\":\"g\",\"arguments\":{}}</tool_call>"
            .to_owned(),
    ));
    for (label, tools, text) in &texts {
        let (want, _) = run(tools, false, &[text]).expect("the whole text parses");
        assert!(!want.calls.is_empty(), "{label}");
        for pieces in every_chunking(text) {
            let (got, calls) = run(tools, true, &pieces).unwrap_or_else(|e| {
                panic!(
                    "{label}: {e}: cut {:?}",
                    pieces.iter().map(|p| p.len()).collect::<Vec<_>>()
                )
            });
            assert_eq!(
                got, want,
                "{label}: the message does not depend on streaming"
            );
            assert_eq!(
                fold_pieces(&calls),
                closed(&want),
                "{label}: fragments == arguments, cut {:?}",
                pieces.iter().map(|p| p.len()).collect::<Vec<_>>()
            );
        }
    }
}

/// A call announced and then ended otherwise is a named error: DSML falls back
/// to content for a call it cannot close (kept by the parser that does not
/// stream), and a stream cannot take its fragments back.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_call_announced_and_ended_otherwise_is_a_named_error() {
    let dsml = Tools::Dsml(Spelling::Spaced);
    let head = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n";
    let param = |k: &str, v: &str| {
        format!("<｜DSML｜ parameter name=\"{k}\" string=\"true\">{v}</｜DSML｜ parameter>\n")
    };
    let tail = "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    let diverged = MarkupError::Diverged {
        index: 0,
        name: "write".to_owned(),
    };
    let char_by_char = |t: &str| -> Vec<String> { t.chars().map(String::from).collect() };
    let run_by_char = |t: &str, streaming: bool| {
        let pieces = char_by_char(t);
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        run(&dsml, streaming, &refs)
    };

    // A parameter given twice: the parser that does not stream keeps the last
    // value; a stream has sent the first.
    let dup = format!(
        "{head}{}{}{}{tail}",
        param("path", "/a"),
        param("mode", "x"),
        param("path", "/b")
    );
    let (m, _) = run(&dsml, false, &[&dup]).expect("parses");
    assert_eq!(m.calls[0].arguments, r#"{"path":"/b","mode":"x"}"#);
    assert_eq!(run_by_char(&dup, true).unwrap_err(), diverged);

    // Text that is not a parameter after the name: content for the parser
    // that does not stream.
    let bad = format!("{head}{}oops\n{tail}", param("path", "/a"));
    let (m, _) = run(&dsml, false, &[&bad]).expect("parses");
    assert!(m.calls.is_empty() && m.content == bad, "{m:?}");
    assert_eq!(run_by_char(&bad, true).unwrap_err(), diverged);

    // The generation ends inside the call (a token limit): content for the
    // parser that does not stream, the call the stream announced is cut.
    let cut = format!("{head}{}", &param("content", "abc")[..60]);
    let (m, _) = run(&dsml, false, &[&cut]).expect("parses");
    assert!(m.calls.is_empty() && m.content == cut, "{m:?}");
    assert_eq!(run_by_char(&cut, true).unwrap_err(), diverged);

    // Before the name is read nothing was announced: the same text is content
    // in both.
    let early = head[..head.len() - 8].to_owned();
    let (m, p) = run_by_char(&early, true).expect("content");
    assert!(
        m.calls.is_empty() && p.is_empty() && m.content == early,
        "{m:?}"
    );

    // A call that closes is exact with the other three markups' cuts too:
    // whole calls are untouched by the parser that streams.
    let ok = format!("{head}{}{tail}", param("path", "/a"));
    let (m, p) = run_by_char(&ok, true).expect("closes");
    assert_eq!(
        m.calls,
        [ToolCall {
            index: 0,
            name: "write".to_owned(),
            arguments: r#"{"path":"/a"}"#.to_owned()
        }]
    );
    assert_eq!(fold_pieces(&p), closed(&m));
}
