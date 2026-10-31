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

use common::{V41_SPACED_TEMPLATE, every_chunking, fold_call_delta, post, start_templated};
use serde_json::{Value, json};
use serve::ScriptedEngine;
use serve::callstream::CallPiece;
use serve::dsml::{ChatParser, MarkupError, Message, Spelling, ToolCall, Tools};
use serve::glmxml::{ArgTypes, KvError, KvSpelling};
use serve::qwenxml::{Framing, ParamKinds};
use serve::reasoning::ReasoningFormat;

const QWEN3: &str = include_str!("fixtures/qwen3-chat-template.jinja");
const QWEN38: &str = include_str!("fixtures/qwen38-chat-template.jinja");
const MIMO: &str = include_str!("fixtures/mimo-chat-template.jinja");
const GLM5: &str = include_str!("fixtures/glm5-chat-template.jinja");

/// The long text of `content`: twelve lines, each with quotes, a backslash, a
/// tab, non-ASCII text and a near-tag of every markup (each shares a prefix
/// with a tag that ends a value or a call, and is not one).
/// A value the watch's program writes holds none of what its quoting refuses
/// (a JSON string body takes no quote, backslash or control byte, and the
/// chat formats' raw values no byte of their own closers): words, digits and
/// unicode carry the streaming.
fn content() -> String {
    (0..12)
        .map(|i| format!("line {i}: 서울 ☃ snowman über {i}; "))
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
            tools: Tools::KvMarkup(KvSpelling::Glm, ArgTypes::of_tools(Some(&tools()))),
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
            tools: Tools::QwenXml(Framing::Lines, ParamKinds::of_tools(Some(&tools()))),
        },
        // The compact framing: the long text is the bytes between its tags,
        // the template's own spelling (no byte between one parameter and the
        // next).
        Case {
            label: "qwen-compact",
            template: MIMO,
            lead: "",
            call: format!(
                "<tool_call>\n<function=write>\n<parameter=path>/a.txt</parameter>\
                 <parameter=mode>420</parameter><parameter=content>{c}</parameter>\
                 </function>\n</tool_call>"
            ),
            tools: Tools::QwenXml(Framing::Compact, ParamKinds::of_tools(Some(&tools()))),
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
        p.try_push(piece, &[])?;
        out.extend(p.take_pieces());
    }
    p.try_finish()?;
    out.extend(p.take_pieces());
    Ok((p.message().clone(), out))
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
        for pieces in every_chunking(text, 7) {
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

    // A parser that keeps a cut call answers with the call the limit cut, not
    // with the one given back before it under the same index.
    let p = param("p", "abc");
    let recut = format!("{bad}{head}{}", &p[..p.find("abc").expect("value") + 2]);
    let mut parser =
        ChatParser::with_tools("", ReasoningFormat::None, Some(dsml.clone())).keeping_cut_call();
    for c in char_by_char(&recut) {
        parser.try_push(&c, &[]).expect("parses");
    }
    assert_eq!(
        parser.try_finish_at(true).expect("cut").calls,
        [ToolCall {
            index: 0,
            name: "write".to_owned(),
            arguments: r#"{"p":"ab"#.to_owned()
        }]
    );

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

/// A call announced under one name and closed under another is a named error:
/// Hermes' object may repeat `name`, and the last one wins when the call
/// closes, so text after the arguments can rename a call the stream has
/// already opened. A cut that has opened nothing yet reads the whole call, and
/// the call it closes with is the one a client sees.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_call_closed_under_another_name_is_a_named_error() {
    let tools = Tools::Hermes;
    let text = r#"<tool_call>{"name":"f","arguments":{},"name":"g"}</tool_call>"#;
    let (want, _) = run(&tools, false, &[text]).expect("the whole text parses");
    assert_eq!(closed(&want), [("g".to_owned(), "{}".to_owned())]);
    let renamed = MarkupError::Diverged {
        index: 0,
        name: "f".to_owned(),
    };

    let rename = text.find(r#","name":"g""#).expect("the second name");
    let two = [&text[..rename], &text[rename..]];
    assert_eq!(run(&tools, true, &two).unwrap_err(), renamed);

    for pieces in every_chunking(text, 1) {
        let lens: Vec<usize> = pieces.iter().map(|p| p.len()).collect();
        match run(&tools, true, &pieces) {
            Err(e) => assert_eq!(e, renamed, "cut {lens:?}"),
            Ok((got, calls)) => {
                assert_eq!(got, want, "cut {lens:?}");
                assert_eq!(fold_pieces(&calls), closed(&want), "cut {lens:?}");
            }
        }
    }
}

/// A token limit that cuts a streamed call inside its long text answers with
/// the call as read so far, and its pieces carry those arguments once: a scan
/// that reads by its own cursor only closes the call it already sent.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_a_cut_call_streams_its_arguments_once() {
    for case in cases() {
        let label = case.label;
        let cut = &case.call[..case.call.find("line 3").expect("the long text")];
        let mut p =
            ChatParser::with_tools("", ReasoningFormat::None, Some(case.tools)).streaming_calls();
        let mut pieces = Vec::new();
        for c in cut.chars() {
            p.try_push(&c.to_string(), &[])
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            pieces.extend(p.take_pieces());
        }
        p.try_finish_at(true)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        pieces.extend(p.take_pieces());
        let calls = closed(p.message());
        assert!(
            calls.len() == 1 && calls[0].1.contains("line 2"),
            "{label}: the cut call stands: {calls:?}"
        );
        assert_eq!(
            fold_pieces(&pieces),
            calls,
            "{label}: fragments == arguments"
        );
    }

    // An argument the cut call completed wrong stays its error, in both spellings.
    let glm = |k: &str, v: &str| format!("<arg_key>{k}</arg_key><arg_value>{v}</arg_value>");
    let solar =
        |k: &str, v: &str| format!("\n<|tool_arg:start|>{k}<|tool_arg:value|>{v}<|tool_arg:end|>");
    let spellings = [
        (
            KvSpelling::Glm,
            "<tool_call>write",
            glm as fn(&str, &str) -> String,
        ),
        (
            KvSpelling::Solar,
            "<|tool_call:start|>write",
            solar as fn(&str, &str) -> String,
        ),
    ];
    for (spelling, open, arg) in spellings {
        let wrong = [
            (arg("mode", "oops"), "NotJson"),
            (
                format!("{}{}", arg("path", "/a"), arg("path", "/b")),
                "DuplicateKey",
            ),
        ];
        let open_value = arg("content", "abc");
        let open_value = &open_value[..open_value.len() - 4];
        for (args, want) in wrong {
            let text = format!("{open}{args}{open_value}");
            let tools = Tools::KvMarkup(spelling, ArgTypes::of_tools(Some(&tools())));
            let mut p =
                ChatParser::with_tools("", ReasoningFormat::None, Some(tools)).streaming_calls();
            for c in text.chars() {
                p.try_push(&c.to_string(), &[]).expect("no close yet");
            }
            let e = p.try_finish_at(true).expect_err("a wrong argument");
            assert!(
                matches!(
                    (&e, want),
                    (MarkupError::Kv(KvError::NotJson { .. }), "NotJson")
                        | (
                            MarkupError::Kv(KvError::DuplicateKey { .. }),
                            "DuplicateKey"
                        )
                ),
                "{spelling:?} {want}: {e}"
            );
        }
    }
}

thread_local! {
    static ALLOCATED: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

struct CountAlloc;

// SAFETY: every allocation is delegated unchanged to the system allocator.
unsafe impl std::alloc::GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATED.with(|n| {
            if let Some(bytes) = n.get() {
                n.set(Some(bytes + layout.size()));
            }
        });
        // SAFETY: the caller supplies the allocator's layout contract.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: this pointer and layout came from the same system allocator.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNT_ALLOC: CountAlloc = CountAlloc;

/// A file-sized string streams with allocation work bounded by its length in
/// every markup that releases a text value as it arrives, preserving escapes
/// and delimiters under small, character-aligned chunks.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_long_string_stream_is_linear() {
    let content = "quoted \" \\ tab\t 서울 </parameter x> </arg_value x> <|tool_arg:end x|> \
                   </｜DSML｜ parameter x>\n"
        .repeat(3000);
    let kinds = || ParamKinds::of_tools(Some(&tools()));
    let types = || ArgTypes::of_tools(Some(&tools()));
    let cases = [
        (
            "qwen",
            Tools::QwenXml(Framing::Lines, kinds()),
            "<tool_call><function=write><parameter=content>\n",
            "\n</parameter>\n</function></tool_call>",
        ),
        (
            "qwen-compact",
            Tools::QwenXml(Framing::Compact, kinds()),
            "<tool_call><function=write><parameter=content>",
            "</parameter></function></tool_call>",
        ),
        (
            "glm",
            Tools::KvMarkup(KvSpelling::Glm, types()),
            "<tool_call>write<arg_key>content</arg_key><arg_value>",
            "</arg_value></tool_call>",
        ),
        (
            "solar",
            Tools::KvMarkup(KvSpelling::Solar, types()),
            "<|tool_call:start|>write\n<|tool_arg:start|>content<|tool_arg:value|>",
            "<|tool_arg:end|>\n<|tool_call:end|>",
        ),
        (
            "dsml",
            Tools::Dsml(Spelling::Spaced),
            "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n\
             <｜DSML｜ parameter name=\"content\" string=\"true\">",
            "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
        ),
    ];
    for (label, tools, prefix, suffix) in cases {
        let mut p =
            ChatParser::with_tools("", ReasoningFormat::None, Some(tools)).streaming_calls();
        let mut arguments = String::new();
        let began = std::time::Instant::now();
        ALLOCATED.with(|n| n.set(Some(0)));
        p.try_push(prefix, &[]).expect("call opens");
        for line in content.split_inclusive('\n') {
            p.try_push(line, &[]).expect("string advances");
            for piece in p.take_pieces() {
                if let CallPiece::Args { text, .. } = piece {
                    arguments.push_str(&text);
                }
            }
        }
        p.try_push(suffix, &[]).expect("call closes");
        for piece in p.take_pieces() {
            if let CallPiece::Args { text, .. } = piece {
                arguments.push_str(&text);
            }
        }
        p.try_finish().expect("stream ends");
        let bytes = ALLOCATED.with(|n| n.replace(None).expect("counting"));
        eprintln!(
            "long-string {label}: bytes={} allocated={bytes} host_us={}",
            content.len(),
            began.elapsed().as_micros()
        );
        assert_eq!(arguments, p.message().calls[0].arguments);
        assert_eq!(
            serde_json::from_str::<Value>(&arguments).unwrap()["content"],
            content
        );
        assert!(
            bytes < content.len() * 128,
            "{label}: {bytes} allocated for {} argument bytes",
            content.len()
        );
    }
}
