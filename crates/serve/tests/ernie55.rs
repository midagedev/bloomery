//! Gate: ERNIE 5.5 Flash Preview's chat in `/apply-template` and `/v1/chat/completions`, and its tool calls
//! (`crates/serve/src/ernie55.rs`, `templates/ernie55.jinja`, `erniexml.rs`).
//!
//! The checkpoint ships no chat template: its renderer is Python, and `tests/fixtures/ernie55-renders.json` holds
//! that renderer's text for every case of `render_ernie55.py` (plain chat, a system turn, multi-turn, tools offered,
//! an assistant's call and its result, thinking on and off, non-ASCII turns, the refusals). The template must write it
//! byte for byte (`hw_ernie55_template_renders_as_the_vendor`), and the server's prompt ids must be the vendor's: each
//! marker an added token's id, each text between them the BPE of the text, with no token matching inside a client's
//! spelling (`hw_ernie55_prompt_ids_are_the_vendors_runs`). `tests/fixtures/ernie55-parses.json` holds the vendor
//! decoder's reading of each output (`decode_ernie55.py`): the parser reads the same
//! (`hw_ernie55_parses_as_the_vendor`), under every cut of the stream
//! (`hw_ernie55_parses_the_same_however_the_stream_is_cut`), and refuses what it refuses by name
//! (`hw_ernie55_malformed_calls_are_named_errors`). Where the server reads on and the vendor refuses, the reading is
//! this server's rule for every markup, held by `hw_ernie55_reads_on_where_the_vendor_refuses` with its reason.
//!
//! `hw_`: the vocabulary clauses read the checkpoint folder's `tokenizer.json` on the box (`BLOOMERY_REF_MODEL`,
//! default `/models/ERNIE-5.5-Flash-Preview`; the fixture cut's is the same file): the prompt ids, the markers read
//! only where their tokens wrote them, and the two ids a message ends on. The other clauses need no file.

mod common;

use std::sync::Arc;

use common::{every_chunking, msg, post, tool_call};
use gguf::Value as Gguf;
use serde_json::{Value, json};
use serve::callstream::CallPiece;
use serve::dsml::{ChatParser, MarkupError, Message, ToolFormat, Tools};
use serve::ernie55::{CALL_CLOSE, CALL_OPEN, MASK, TEMPLATE, THINK};
use serve::erniexml::ErnieXmlError;
use serve::glmxml::ArgTypes;
use serve::reasoning::{ReasoningFormat, ThinkEntry};
use serve::vocab::Vocab;
use serve::{ChatTemplate, FATAL_LINGER, ScriptedEngine, Server, ServerConfig, Tokenizer as _};

/// The folder whose `tokenizer.json` the vocabulary clauses read.
fn folder() -> String {
    std::env::var("BLOOMERY_REF_MODEL")
        .ok()
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| "/models/ERNIE-5.5-Flash-Preview".to_owned())
}

/// The added-token ids of the protocol's markers, the folder's `tokenizer.json`.
const MASK_ID: u32 = 2;
const END_ID: u32 = 8;
const CALL_OPEN_ID: u32 = 17;
const CALL_CLOSE_ID: u32 = 18;
const THINK_CLOSE_ID: u32 = 20;

/// The folder's vocabulary as the server reads it: `tokenizer.json` as the header keys a GGUF file would carry
/// (the reading `crates/tokenizer/tests/ernie55.rs` makes), its stops the two ids a message ends on
/// (`generation_config.json`'s `eos_token_id`) and its BOS the mask, as `crates/hf/src/vocab.rs` exports them.
fn vocabulary() -> Vocab {
    let text = std::fs::read(format!("{}/tokenizer.json", folder()))
        .expect("the ERNIE 5.5 folder's tokenizer.json");
    let tok: Value = serde_json::from_slice(&text).expect("tokenizer.json parses");
    let vocab = tok["model"]["vocab"].as_object().expect("a vocabulary");
    let mut tokens = vec![String::new(); vocab.len()];
    for (token, id) in vocab {
        tokens[id.as_u64().expect("an id") as usize] = token.clone();
    }
    let mut types = vec![1i32; tokens.len()];
    for added in tok["added_tokens"].as_array().expect("added tokens") {
        let id = added["id"].as_u64().expect("an id") as usize;
        types[id] = if added["special"].as_bool().expect("special") {
            3
        } else {
            4
        };
    }
    let merges: Vec<Gguf> = tok["model"]["merges"]
        .as_array()
        .expect("merges")
        .iter()
        .map(|m| {
            let pair = m.as_array().expect("a merge is two tokens");
            Gguf::String(format!(
                "{} {}",
                pair[0].as_str().expect("a token"),
                pair[1].as_str().expect("a token")
            ))
        })
        .collect();
    let strings = |v: Vec<String>| Gguf::Array(v.into_iter().map(Gguf::String).collect());
    let keys = vec![
        (
            "tokenizer.ggml.model".to_owned(),
            Gguf::String("gpt2".to_owned()),
        ),
        (
            "tokenizer.ggml.pre".to_owned(),
            Gguf::String("ernie5".to_owned()),
        ),
        ("tokenizer.ggml.tokens".to_owned(), strings(tokens)),
        (
            "tokenizer.ggml.token_type".to_owned(),
            Gguf::Array(types.into_iter().map(Gguf::I32).collect()),
        ),
        ("tokenizer.ggml.merges".to_owned(), Gguf::Array(merges)),
        ("tokenizer.ggml.eos_token_id".to_owned(), Gguf::U32(END_ID)),
        ("tokenizer.ggml.bos_token_id".to_owned(), Gguf::U32(MASK_ID)),
        ("tokenizer.ggml.add_bos_token".to_owned(), Gguf::Bool(true)),
        (
            "tokenizer.bloomery.eog_token_ids".to_owned(),
            Gguf::Array(vec![Gguf::U32(MASK_ID), Gguf::U32(END_ID)]),
        ),
    ];
    let tok = tokenizer::Tokenizer::from_keys(&keys).expect("the ERNIE 5.5 vocabulary loads");
    Vocab::new(tok).expect("the vocabulary names its EOS")
}

/// A mark no prompt holds, for the plain reading of a text.
const MARK: &str = "\u{1}";

/// The ids of `text` spelled by ordinary tokens: no marker matches, however it is written. Read through the
/// tokenizer's own guard, not the server's `Vocab` wrapper of it, so a wrapper that lifts the guard shows.
fn plain(vocab: &Vocab, text: &str) -> Vec<u32> {
    let tok = vocab.tokenizer();
    tok.encode_marked(&tok.guard(text, MARK), MARK, false, true)
}

/// A server on `engine`, the ERNIE template and its think tags.
fn start(engine: Box<dyn serve::Engine>) -> std::net::SocketAddr {
    let config = ServerConfig {
        model_alias: "mock".to_owned(),
        model_path: "mock.gguf".to_owned(),
        chat_template: TEMPLATE.to_owned(),
        chat_preparation: None,
        sampler: None,
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
        api_keys: serve::flag::ApiKeys::default(),
    };
    let server =
        Server::bind("127.0.0.1:0", engine, config).unwrap_or_else(|e| panic!("bind: {e}"));
    server.with_think_tags(THINK).spawn().expect("spawn")
}

/// A mock server whose every reply is `script`.
fn mock(script: &str) -> std::net::SocketAddr {
    start(Box::new(ScriptedEngine::new(8192, script)))
}

/// A server whose every reply is the ids `script`, over `vocab`.
fn scripted(vocab: Arc<Vocab>, script: Vec<u32>) -> std::net::SocketAddr {
    start(Box::new(
        ScriptedEngine::from_ids(8192, script).with_tokenizer(vocab),
    ))
}

fn renders() -> Value {
    serde_json::from_str(include_str!("fixtures/ernie55-renders.json")).expect("fixture JSON")
}

fn parses() -> Value {
    serde_json::from_str(include_str!("fixtures/ernie55-parses.json")).expect("fixture JSON")
}

/// The case of `fixture` named `name`.
fn case<'a>(fixture: &'a Value, name: &str) -> &'a Value {
    fixture["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no case {name:?}"))
}

/// The vendor's generation prompt after one user message `hi`: the span open, or closed.
fn prompt(on: bool) -> String {
    case(
        &renders(),
        if on {
            "user only"
        } else {
            "thinking off: enable_thinking false in the kwargs"
        },
    )["prompt"]
        .as_str()
        .expect("a prompt")
        .to_owned()
}

/// The tools the parse fixtures call.
fn tools() -> Value {
    case(&parses(), "an answer")["tools"].clone()
}

/// The parser for `tools`' markup after the prompt.
fn parser(prompt: &str, tools: Option<&Value>) -> ChatParser {
    let scan = Tools::ErnieXml(ArgTypes::of_tools(tools));
    ChatParser::with_tags(prompt, ReasoningFormat::Deepseek, Some(scan), THINK)
}

/// `pieces` pushed in order and the finish, read whole.
fn read(p: &mut ChatParser, pieces: &[&str]) -> Result<Message, MarkupError> {
    common::sum_deltas(p, pieces)
}

/// A parse fixture read after its prompt: `Ok` with the message, `Err` with the markup error.
fn parse(fixture: &Value, pieces: &[&str]) -> Result<Message, MarkupError> {
    let on = fixture["thinking"].as_bool().expect("thinking");
    let tools = fixture.get("tools").filter(|t| !t.is_null());
    read(&mut parser(&prompt(on), tools), pieces)
}

/// `v` with every number as Python reads it back: one with a fraction or an exponent is a float.
fn canonical(v: &Value) -> Value {
    match v {
        Value::Number(n) if n.to_string().contains(['.', 'e', 'E']) => {
            Value::from(n.as_f64().expect("a float"))
        }
        Value::Array(a) => Value::Array(a.iter().map(canonical).collect()),
        Value::Object(o) => {
            Value::Object(o.iter().map(|(k, x)| (k.clone(), canonical(x))).collect())
        }
        _ => v.clone(),
    }
}

/// Whether `m` is the fixture's `want`: reasoning, content and calls in order, each argument the JSON it parses to.
fn is_want(m: &Message, want: &Value) -> bool {
    let calls = want["calls"].as_array().expect("calls");
    m.reasoning == want["reasoning"].as_str().expect("reasoning")
        && m.content == want["content"].as_str().expect("content")
        && m.calls.len() == calls.len()
        && m.calls.iter().zip(calls).all(|(got, want)| {
            got.name == want["name"].as_str().expect("a name")
                && serde_json::from_str::<Value>(&got.arguments)
                    .is_ok_and(|a| a.is_object() && canonical(&a) == canonical(&want["arguments"]))
        })
}

/// The message with each call's arguments as the JSON they parse to, for an assertion's print.
fn shown(m: &Message) -> Message {
    Message {
        calls: m
            .calls
            .iter()
            .map(|c| {
                tool_call(
                    c.index,
                    &c.name,
                    &serde_json::to_string(&canonical(
                        &serde_json::from_str::<Value>(&c.arguments).expect("JSON arguments"),
                    ))
                    .expect("JSON"),
                )
            })
            .collect(),
        ..m.clone()
    }
}

/// A streamed chat body folded back ([`common::fold_stream`]) under the block's tags.
fn fold_stream(body: &str) -> Result<(Value, Value), String> {
    common::fold_stream(body, &[CALL_OPEN, CALL_CLOSE])
}

// ---------------------------------------------------------------- the prompt

/// The template writes the vendor renderer's text, case by case, and refuses what the vendor refuses: a 400 naming
/// the fault. FAIL-first: with the template's effort default, mask or think span moved, or a refusal lifted, a case
/// differs.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_template_renders_as_the_vendor() {
    let addr = mock("");
    let fixture = renders();
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 60, "the case list");
    let mut wrong = Vec::new();
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let r = post(addr, "/apply-template", &case["body"]);
        match (case["prompt"].as_str(), case["refusal"].as_str()) {
            // A case whose ids differ from the vendor's is held to its text alone.
            (Some(text), _) => {
                if r.status != 200 || r.json()["prompt"] != text {
                    wrong.push(format!(
                        "{name}: {} {:?}, want {text:?}",
                        r.status,
                        r.json()["prompt"].as_str().unwrap_or(&r.body)
                    ));
                }
            }
            (None, Some(refusal)) => {
                if r.status != 400 || !r.body.contains(refusal) {
                    wrong.push(format!(
                        "{name}: {} {}, want a 400 naming {refusal:?}",
                        r.status,
                        r.json()["error"]["message"].as_str().unwrap_or(&r.body)
                    ));
                }
            }
            (None, None) => panic!("{name}: neither a prompt nor a refusal"),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s) wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The server's prompt ids are the vendor's runs: each marker the added token's id, each text between them the BPE
/// of the text, and a marker a client spelled as text. The prompt holds the one `<|ERNIE_MASK|>` the template
/// writes: the vocabulary's BOS is that token, and only a raw text prompt gets it added. FAIL-first: with the mask
/// left off the prompt, or a client's spelling matched as a token (the guard lifted), the ids differ.
#[test]
#[ignore = "gate: the ERNIE 5.5 folder's tokenizer.json on the box"]
fn hw_ernie55_prompt_ids_are_the_vendors_runs() {
    let vocab = Arc::new(vocabulary());
    for (text, id) in [
        (MASK, MASK_ID),
        (serve::ernie55::END, END_ID),
        (CALL_OPEN, CALL_OPEN_ID),
        (CALL_CLOSE, CALL_CLOSE_ID),
        (THINK.close, THINK_CLOSE_ID),
    ] {
        assert_eq!(vocab.encode(text), vec![id], "{text} is one token");
    }
    let addr = scripted(Arc::clone(&vocab), Vec::new());
    let mut wrong = Vec::new();
    for case in renders()["cases"].as_array().expect("cases") {
        let Some(runs) = case["runs"].as_array() else {
            continue;
        };
        let mut body = case["body"].clone();
        body["return_prompt_ids"] = json!(true);
        let r = post(addr, "/apply-template", &body);
        assert_eq!(r.status, 200, "{}: {}", case["name"], r.body);
        let ids: Vec<u32> = r.json()["prompt_ids"]
            .as_array()
            .expect("prompt_ids")
            .iter()
            .map(|id| id.as_u64().expect("an id") as u32)
            .collect();
        let expect: Vec<u32> = runs
            .iter()
            .flat_map(|run| match run {
                Value::Number(id) => vec![id.as_u64().expect("an id") as u32],
                Value::String(text) => plain(&vocab, text),
                other => panic!("{other:?} is not a run"),
            })
            .collect();
        if ids != expect {
            wrong.push(format!(
                "{}:\n  got  {ids:?}\n  want {expect:?}",
                case["name"]
            ));
        }
        assert_eq!(
            ids.first(),
            Some(&MASK_ID),
            "{}: the mask opens the prompt",
            case["name"]
        );
        assert!(!ids.contains(&1), "{}: no BOS in the prompt", case["name"]);
    }
    assert!(
        wrong.is_empty(),
        "{} case(s) wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The template types to its parser from its source: the server picks the ERNIE scan for it, as for every seat's
/// template. FAIL-first: with the spelling's row gone, the format is `Unparsed` and a tools request is refused.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_template_types_to_its_parser() {
    let t = ChatTemplate::parse(TEMPLATE).expect("the template parses");
    assert_eq!(ToolFormat::of_chat_template(&t), ToolFormat::ErnieXml);
}

/// A rendered prompt names where it leaves the think span: open after the generation prompt with thinking on, closed
/// with it off. FAIL-first: with a tag of [`THINK`] moved, both prompts leave the span to the model.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_prompts_name_where_the_span_is() {
    assert_eq!(
        ThinkEntry::of_prompt_with(&prompt(true), THINK),
        ThinkEntry::Open
    );
    assert_eq!(
        ThinkEntry::of_prompt_with(&prompt(false), THINK),
        ThinkEntry::Closed
    );
}

// ---------------------------------------------------------------- the parser

/// Every output reads as the vendor decoder reads it: reasoning, content and calls, each argument the string or the
/// JSON the tool's schema admits; and what the vendor refuses is refused. FAIL-first: with a string-typed value read
/// as JSON, a call's argument differs; with the whitespace between the elements kept, no call parses.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_parses_as_the_vendor() {
    let fixture = parses();
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 37, "the case list");
    let mut wrong = Vec::new();
    for case in cases {
        let name = case["name"].as_str().expect("a name");
        let output = case["output"].as_str().expect("an output");
        // A cut output is whole to the vendor's allow_incomplete alone: the server's reading of a cut is
        // `hw_ernie55_reads_on_where_the_vendor_refuses`.
        if case["cut"].as_bool() == Some(true) {
            continue;
        }
        match case["want"].as_object() {
            Some(_) => match parse(case, &[output]) {
                Ok(got) if is_want(&got, &case["want"]) => {}
                got => wrong.push(format!(
                    "{name}:\n  got  {:?}\n  want {}",
                    got.as_ref().map(shown),
                    case["want"]
                )),
            },
            None => match parse(case, &[output]) {
                Err(e) => assert!(
                    e.to_string().starts_with("tool call"),
                    "{name}: the error names the call: {e}"
                ),
                Ok(got) => wrong.push(format!(
                    "{name}: read {:?}, want an error ({})",
                    shown(&got),
                    case["error"]
                )),
            },
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s) wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A message never depends on how the stream was cut: every output the vendor reads whole reads the same pushed at
/// once, a character at a time, split in two at every character and in three at every pair of tag boundaries, and a
/// streamed call's fragments concatenate to the arguments the call closes with. FAIL-first: with a tag's partial
/// tail released as content, a cut inside `<ernie:tool_call>` leaks the tag into the content.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_parses_the_same_however_the_stream_is_cut() {
    let fixture = parses();
    let mut wrong = Vec::new();
    for case in fixture["cases"].as_array().expect("cases") {
        if case["want"].is_object() && case.get("tools").is_some_and(|t| !t.is_null()) {
            let output = case["output"].as_str().expect("an output");
            for pieces in every_chunking(output, 1) {
                match parse(case, &pieces) {
                    Ok(got) if is_want(&got, &case["want"]) => {}
                    got => {
                        wrong.push(format!(
                            "{}: {pieces:?}\n  got  {:?}\n  want {}",
                            case["name"],
                            got.as_ref().map(shown),
                            case["want"]
                        ));
                    }
                }
                // The streamed form: the pieces a push reported concatenate, per call, to the call's arguments.
                let on = case["thinking"].as_bool().expect("thinking");
                let tools = case.get("tools");
                let mut p = parser(&prompt(on), tools).streaming_calls();
                let mut streamed: Vec<(usize, String, String)> = Vec::new();
                let take = |p: &mut ChatParser, streamed: &mut Vec<(usize, String, String)>| {
                    for piece in p.take_pieces() {
                        match piece {
                            CallPiece::Open { index, name } => {
                                assert_eq!(index, streamed.len(), "calls open once, in order");
                                streamed.push((index, name, String::new()));
                            }
                            CallPiece::Args { index, text } => {
                                let last = streamed.last_mut().expect("a call opened");
                                assert_eq!(last.0, index, "a fragment extends the open call");
                                last.2.push_str(&text);
                            }
                        }
                    }
                };
                for piece in &pieces {
                    p.try_push(piece, &[]).expect("parses");
                    take(&mut p, &mut streamed);
                }
                p.try_finish().expect("closes");
                take(&mut p, &mut streamed);
                let closed: Vec<(usize, String, String)> = p
                    .message()
                    .calls
                    .iter()
                    .map(|c| (c.index, c.name.clone(), c.arguments.clone()))
                    .collect();
                if closed != streamed {
                    wrong.push(format!(
                        "{}: {pieces:?}: streamed {streamed:?}, closed {closed:?}",
                        case["name"]
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s) wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Markup that does not parse is a named error, never content and never a guessed call, under every cut of the
/// stream. FAIL-first: with the duplicate check lifted, a parameter given twice is a call.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_malformed_calls_are_named_errors() {
    const WEATHER: &str = "<ernie:tool_call><invoke name=\"get_weather\">";
    let unexpected = |name: &str, expected: &'static str, found: &str| ErnieXmlError::Unexpected {
        index: 0,
        name: name.to_owned(),
        expected,
        found: found.to_owned(),
    };
    let no_invoke = |found: &str| ErnieXmlError::NoInvoke {
        index: 0,
        found: found.to_owned(),
    };
    let cases: Vec<(&str, String, ErnieXmlError)> = vec![
        (
            "a block with no invoke",
            "</ernie:think><ernie:tool_call></ernie:tool_call>".to_owned(),
            ErnieXmlError::EmptyBlock { index: 0 },
        ),
        (
            "a block of whitespace only",
            "</ernie:think><ernie:tool_call> \n </ernie:tool_call>".to_owned(),
            ErnieXmlError::EmptyBlock { index: 0 },
        ),
        (
            "text where an invoke belongs",
            "</ernie:think><ernie:tool_call>hello".to_owned(),
            no_invoke("hello"),
        ),
        (
            "an opening tag without a name",
            "</ernie:think><ernie:tool_call><invoke></invoke></ernie:tool_call>".to_owned(),
            no_invoke("<invoke></invoke></ernie:tool_call>"),
        ),
        (
            "an opening tag with an empty name",
            "</ernie:think><ernie:tool_call><invoke name=\"\"></invoke></ernie:tool_call>"
                .to_owned(),
            no_invoke("<invoke name=\"\"></invoke></ernie:tool_ca"),
        ),
        (
            "an opening tag with no whitespace before its name",
            "</ernie:think><ernie:tool_call><invokename=\"x\"></invoke></ernie:tool_call>"
                .to_owned(),
            no_invoke("<invokename=\"x\"></invoke></ernie:tool_ca"),
        ),
        (
            "a name with a quote cut short",
            "</ernie:think><ernie:tool_call><invoke name=\"get\"time\"></invoke></ernie:tool_call>"
                .to_owned(),
            no_invoke("<invoke name=\"get\"time\"></invoke></ernie"),
        ),
        (
            "text where a parameter belongs",
            format!("</ernie:think>{WEATHER}<function=x></invoke></ernie:tool_call>"),
            unexpected(
                "get_weather",
                "</invoke>, or <parameter name=\"KEY\">",
                "<function=x></invoke></ernie:tool_call>",
            ),
        ),
        (
            "a duplicate parameter",
            "</ernie:think><ernie:tool_call><invoke name=\"get_weather\"><parameter \
             name=\"location\">Seoul</parameter><parameter name=\"location\">Oslo</parameter>\
             </invoke></ernie:tool_call>"
                .to_owned(),
            ErnieXmlError::DuplicateParameter {
                index: 0,
                name: "get_weather".to_owned(),
                key: "location".to_owned(),
            },
        ),
        (
            "a parameter never closed: the block ends inside it",
            format!("</ernie:think>{WEATHER}<parameter name=\"location\">Seoul</ernie:tool_call>"),
            unexpected("get_weather", "</parameter>", "</ernie:tool_call>"),
        ),
        (
            "the block's close inside a value",
            format!(
                "</ernie:think>{WEATHER}<parameter name=\"location\">Se</ernie:tool_call>\
                 </parameter></invoke></ernie:tool_call>"
            ),
            unexpected("get_weather", "</parameter>", CALL_CLOSE),
        ),
        (
            "an invoke never closed: the block's close inside it",
            format!("</ernie:think>{WEATHER}</ernie:tool_call>"),
            unexpected("get_weather", "</invoke>", "</ernie:tool_call>"),
        ),
        (
            "an integer-typed value that is not JSON",
            "</ernie:think><ernie:tool_call><invoke name=\"get_weather\"><parameter \
             name=\"location\">Seoul</parameter><parameter name=\"days\">three</parameter>\
             </invoke></ernie:tool_call>"
                .to_owned(),
            ErnieXmlError::NotJson {
                index: 0,
                name: "get_weather".to_owned(),
                key: "days".to_owned(),
                ty: "integer".to_owned(),
                value: "three".to_owned(),
            },
        ),
        (
            "a boolean-typed word",
            "</ernie:think><ernie:tool_call><invoke name=\"set_knob\"><parameter \
             name=\"gain\">1</parameter><parameter name=\"on\">True</parameter></invoke>\
             </ernie:tool_call>"
                .to_owned(),
            ErnieXmlError::NotJson {
                index: 0,
                name: "set_knob".to_owned(),
                key: "on".to_owned(),
                ty: "boolean".to_owned(),
                value: "True".to_owned(),
            },
        ),
        (
            "a block the stream ends inside",
            format!("</ernie:think>{WEATHER}"),
            ErnieXmlError::Unterminated {
                index: 0,
                text: WEATHER.to_owned(),
            },
        ),
        (
            "a block the stream ends inside after a call",
            "</ernie:think><ernie:tool_call><invoke name=\"get_time\"></invoke>".to_owned(),
            ErnieXmlError::Unterminated {
                index: 1,
                text: "<ernie:tool_call><invoke name=\"get_time\"></invoke>".to_owned(),
            },
        ),
    ];
    let tools = tools();
    let on = prompt(true);
    let mut wrong = Vec::new();
    for (name, output, e) in &cases {
        let expect = Err(e.clone().into());
        for pieces in every_chunking(output, 1) {
            let got = read(&mut parser(&on, Some(&tools)), &pieces);
            if got != expect {
                wrong.push(format!(
                    "{name}: {pieces:?}\n  got  {got:?}\n  want {expect:?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} case(s) wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Where the server reads on and the vendor decoder refuses, the reading is this server's rule for every markup it
/// parses, each held here with the vendor's reading beside it.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_reads_on_where_the_vendor_refuses() {
    let tools = tools();
    const WEATHER: &str = "<ernie:tool_call><invoke name=\"get_weather\"><parameter \
                          name=\"location\">Seoul</parameter><parameter name=\"days\">3</parameter>\
                          </invoke>\n</ernie:tool_call>";
    let weather = || tool_call(0, "get_weather", r#"{"location":"Seoul","days":3}"#);
    let parse = |output: &str| read(&mut parser(&prompt(true), Some(&tools)), &[output]);
    // Text after the block is content (the vendor: "Unexpected text after the tool-call block"), and a second
    // block's calls are numbered on.
    assert_eq!(
        parse(&format!("</ernie:think>{WEATHER} and then {WEATHER}")),
        Ok(msg(
            "",
            " and then ",
            vec![
                weather(),
                tool_call(1, "get_weather", r#"{"location":"Seoul","days":3}"#)
            ]
        ))
    );
    // The block's layout after its close is nothing.
    assert_eq!(
        parse(&format!("</ernie:think>{WEATHER}\n  ")),
        Ok(msg("", "", vec![weather()]))
    );
    // A name the tools do not list is a call (the vendor: "Generated unknown tool"), its arguments read by the
    // schema's light where the tools offer one, JSON else text.
    assert_eq!(
        parse(
            "</ernie:think><ernie:tool_call><invoke name=\"lookup\"><parameter name=\"q\">42</parameter>\
             <parameter name=\"w\">word</parameter><parameter name=\"o\">{\"a\": null}</parameter></invoke>\
             </ernie:tool_call>"
        ),
        Ok(msg(
            "",
            "",
            vec![tool_call(
                0,
                "lookup",
                r#"{"q":42,"w":"word","o":{"a":null}}"#
            )]
        ))
    );
    // A required parameter left out is a call (the vendor: "Arguments for tool ... do not match its schema").
    assert_eq!(
        parse(
            "</ernie:think><ernie:tool_call><invoke name=\"get_weather\"><parameter \
             name=\"location\">Seoul</parameter></invoke></ernie:tool_call>"
        ),
        Ok(msg(
            "",
            "",
            vec![tool_call(0, "get_weather", r#"{"location":"Seoul"}"#)]
        ))
    );
    // A parameter the schema types in several ways, a `null` among them, reads as the JSON it parses as (the
    // vendor: ambiguous, refused).
    assert_eq!(
        parse(
            "</ernie:think><ernie:tool_call><invoke name=\"save_note\"><parameter name=\"text\">t</parameter>\
             <parameter name=\"tone\">null</parameter></invoke></ernie:tool_call>"
        ),
        Ok(msg(
            "",
            "",
            vec![tool_call(0, "save_note", r#"{"text":"t","tone":null}"#)]
        ))
    );
    // A marker inside the think span is reasoning: the span ends at its close alone (the vendor: "Expected closing
    // think tag, found <ernie:tool_call>").
    assert_eq!(
        parse("a <ernie:tool_call> in the span"),
        Ok(msg("a <ernie:tool_call> in the span", "", vec![]))
    );
    // A span the limit cut leaves its text as reasoning, as the vendor reads it with allow_incomplete (and refuses
    // without).
    let cut = parses();
    let think_cut = case(&cut, "no think close, cut by the limit")["output"]
        .as_str()
        .expect("an output");
    assert_eq!(parse(think_cut), Ok(msg("still thinking", "", vec![])));
    // The limit's cut inside a block is an answer, not an error: the block's text as content, or with the call kept
    // once its name was read (the vendor: the block's text unparsed, no call).
    let call_cut = case(&cut, "cut inside a call after its name")["output"]
        .as_str()
        .expect("an output");
    let block_cut = case(&cut, "cut right after the block's opening")["output"]
        .as_str()
        .expect("an output");
    let cut_call = "<ernie:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Se";
    assert_eq!(
        parse(call_cut),
        Err(ErnieXmlError::Unterminated {
            index: 0,
            text: cut_call.to_owned()
        }
        .into())
    );
    assert_eq!(
        parse(block_cut),
        Err(ErnieXmlError::Unterminated {
            index: 0,
            text: CALL_OPEN.to_owned()
        }
        .into())
    );
    for (output, kept) in [(call_cut, true), (block_cut, false)] {
        let mut p = parser(&prompt(true), Some(&tools)).keeping_cut_call();
        p.try_push(output, &[]).expect("reads");
        let want = if kept {
            msg(
                "",
                "",
                vec![tool_call(0, "get_weather", r#"{"location":"Se"#)],
            )
        } else {
            msg("", CALL_OPEN, vec![])
        };
        assert_eq!(p.try_finish_at(true), Ok(want), "{output}");
    }
    let mut p = parser(&prompt(true), Some(&tools));
    p.try_push(call_cut, &[]).expect("reads");
    assert_eq!(
        p.try_finish_at(true),
        Ok(msg("", cut_call, vec![])),
        "the default gives the cut call's text back as content"
    );
    // With no tools offered the server scans nothing: the block stays content (the vendor: its calls, every
    // argument a string).
    assert_eq!(
        read(
            &mut ChatParser::with_tags(&prompt(true), ReasoningFormat::Deepseek, None, THINK),
            &[
                "</ernie:think><ernie:tool_call><invoke name=\"get_time\"></invoke></ernie:tool_call>"
            ]
        ),
        Ok(msg(
            "",
            "<ernie:tool_call><invoke name=\"get_time\"></invoke></ernie:tool_call>",
            vec![]
        ))
    );
    // `reasoning_format: none` keeps the span in the content, its close written back.
    assert_eq!(
        read(
            &mut ChatParser::with_tags(
                &prompt(true),
                ReasoningFormat::None,
                Some(Tools::ErnieXml(ArgTypes::default())),
                THINK
            ),
            &["why</ernie:think>hello"]
        ),
        Ok(msg("", "why</ernie:think>hello", vec![]))
    );
}

/// On the folder's vocabulary a marker is markup only where its token wrote it: a `<ernie:tool_call>` spelled by
/// ordinary tokens is content, and the same characters as the token open a block; a spelled `</ernie:think>` is
/// reasoning, the token's ends the span. FAIL-first: with the tags read by their text, the spelled block is a call.
#[test]
#[ignore = "gate: the ERNIE 5.5 folder's tokenizer.json on the box"]
fn hw_ernie55_markers_are_markup_only_where_their_tokens_wrote_them() {
    let vocab = Arc::new(vocabulary());
    const BLOCK: &str = "<ernie:tool_call><invoke name=\"get_time\"></invoke></ernie:tool_call>";
    let chat = |ids: Vec<u32>| {
        let addr = scripted(Arc::clone(&vocab), ids);
        let r = post(
            addr,
            "/v1/chat/completions",
            &common::tool_body(tools(), false),
        );
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()["choices"][0].clone()
    };
    // The span's close as its token, the block spelled by ordinary tokens: content.
    let mut ids = plain(&vocab, "Hi.");
    ids.push(THINK_CLOSE_ID);
    ids.extend(plain(&vocab, BLOCK));
    let c = chat(ids);
    assert_eq!(
        c["finish_reason"], "stop",
        "the spelled block is no call: {c}"
    );
    assert_eq!(c["message"]["reasoning_content"], "Hi.", "{c}");
    assert_eq!(c["message"]["content"], BLOCK, "{c}");
    assert_eq!(c["message"]["tool_calls"], json!(null), "{c}");
    // The same characters as their tokens: a call.
    let c = chat(vocab.encode(&format!("Hi.</ernie:think>{BLOCK}")));
    assert_eq!(c["finish_reason"], "tool_calls", "{c}");
    assert_eq!(c["message"]["reasoning_content"], "Hi.", "{c}");
    assert_eq!(c["message"]["content"], "", "{c}");
    assert_eq!(
        c["message"]["tool_calls"][0]["function"],
        json!({"name": "get_time", "arguments": "{}"}),
        "{c}"
    );
    // A spelled close is reasoning; the token's ends the span.
    let c = chat(plain(&vocab, "a </ernie:think> b"));
    assert_eq!(
        c["message"]["reasoning_content"], "a </ernie:think> b",
        "{c}"
    );
    assert_eq!(c["message"]["content"], "", "{c}");
    let mut ids = plain(&vocab, "a ");
    ids.push(THINK_CLOSE_ID);
    ids.extend(plain(&vocab, " b"));
    let c = chat(ids);
    assert_eq!(c["message"]["reasoning_content"], "a ", "{c}");
    assert_eq!(c["message"]["content"], " b", "{c}");
}

/// A message ends on `<|ERNIE_MASK|>` and on `<|ernie:im_end|>`, the two ids of `generation_config.json`'s
/// `eos_token_id`: what follows either is not answered. FAIL-first: with the mask left out of the stops, the text
/// after it is content.
#[test]
#[ignore = "gate: the ERNIE 5.5 folder's tokenizer.json on the box"]
fn hw_ernie55_a_message_ends_on_the_mask_and_the_turn_end() {
    let vocab = Arc::new(vocabulary());
    for end in [MASK, serve::ernie55::END] {
        let addr = scripted(
            Arc::clone(&vocab),
            vocab.encode(&format!("Hello!</ernie:think>Bye!{end}and a new turn")),
        );
        let r = post(
            addr,
            "/v1/chat/completions",
            &json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}),
        );
        assert_eq!(r.status, 200, "{}", r.body);
        let c = &r.json()["choices"][0];
        assert_eq!(c["finish_reason"], "stop", "{end}: {c}");
        assert_eq!(c["message"]["reasoning_content"], "Hello!", "{end}: {c}");
        assert_eq!(c["message"]["content"], "Bye!", "{end}: {c}");
    }
}

// ---------------------------------------------------------------- the server

/// A reply through `/v1/chat/completions`: the reasoning splits at the span's close, the block's tags are never
/// content, and the streamed calls fold to the non-streamed ones; `/v1/messages` answers `tool_use` blocks with the
/// parameters typed by each tool's schema.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_chat_tool_calls_stream_and_non_stream() {
    const TWO: &str = "Both.</ernie:think>Checking both.<ernie:tool_call>\n\
                       <invoke name=\"get_weather\"><parameter name=\"location\">Seoul</parameter></invoke>\n\
                       <invoke name=\"get_time\"></invoke>\n</ernie:tool_call>";
    // `tools()` holds a schema the structured-output subset cannot compile, so a `tool_choice: auto` request
    // leaves the reply to the model: the script's markup reaches the scan as it is.
    let addr = mock(TWO);
    let r = post(
        addr,
        "/v1/chat/completions",
        &common::tool_body(tools(), false),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    assert_eq!(c["message"]["reasoning_content"], "Both.", "{v}");
    assert_eq!(c["message"]["content"], "Checking both.", "{v}");
    let called = json!([
        {"name": "get_weather", "arguments": r#"{"location":"Seoul"}"#},
        {"name": "get_time", "arguments": "{}"},
    ]);
    let calls: Vec<Value> = c["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls")
        .iter()
        .map(|tc| tc["function"].clone())
        .collect();
    assert_eq!(json!(calls), called, "{v}");
    let r = post(
        addr,
        "/v1/chat/completions",
        &common::tool_body(tools(), true),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "tool_calls");
    assert_eq!(folded["content"], "Checking both.");
    assert_eq!(folded["reasoning_content"], "Both.");
    assert_eq!(folded["calls"], called, "streamed == non-streamed");
    let r = post(
        addr,
        "/v1/messages",
        &json!({"model": "m", "max_tokens": 256, "temperature": 0,
            "messages": [{"role": "user", "content": "weather and time?"}],
            "tools": [{"name": "get_weather", "description": "", "input_schema": {"type": "object", "properties": {
                "location": {"type": "string"}, "days": {"type": "integer"}}}},
                {"name": "get_time", "description": "", "input_schema": {"type": "object", "properties": {}}}]}),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    assert_eq!(v["stop_reason"], "tool_use", "{v}");
    let uses: Vec<Value> = v["content"]
        .as_array()
        .expect("blocks")
        .iter()
        .filter(|b| b["type"] == "tool_use")
        .map(|b| json!({"name": b["name"], "input": b["input"]}))
        .collect();
    assert_eq!(
        json!(uses),
        json!([
            {"name": "get_weather", "input": {"location": "Seoul"}},
            {"name": "get_time", "input": {}},
        ]),
        "{v}"
    );
}

/// Under a watch (tools the structured-output subset compiles, `tool_choice: auto`) every call of a block is held to
/// its schema, the second as the first. The reply is the protocol's own layout (the block's opening, a line feed, a
/// call, `</invoke>` and one line feed, the next call right after it, the block's close), and its second call writes
/// `x` where `days` is an integer: the watch turns it into an integer, read streamed and not. FAIL-first: with the
/// program holding the first call only, the second is free text, and `x` is not JSON for an integer.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_a_watch_holds_the_second_call_to_its_schema() {
    const SCRIPT: &str = "Need both.</ernie:think><ernie:tool_call>\n\
                          <invoke name=\"get_weather\"><parameter name=\"location\">Seoul</parameter></invoke>\n\
                          <invoke name=\"get_weather\"><parameter name=\"location\">Busan</parameter>\
                          <parameter name=\"days\">x</parameter></invoke>\n</ernie:tool_call>";
    let served = json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["location"]}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
    ]);
    let addr = mock(SCRIPT);
    for stream in [false, true] {
        let r = post(
            addr,
            "/v1/chat/completions",
            &common::tool_body(served.clone(), stream),
        );
        assert_eq!(r.status, 200, "stream {stream}: {}", r.body);
        let calls: Vec<Value> = if stream {
            let (finish, folded) = fold_stream(&r.body).expect("no error event");
            assert_eq!(finish, "tool_calls");
            assert_eq!(folded["reasoning_content"], "Need both.");
            folded["calls"].as_array().expect("calls").clone()
        } else {
            let c = &r.json()["choices"][0];
            assert_eq!(c["finish_reason"], "tool_calls", "{c}");
            c["message"]["tool_calls"]
                .as_array()
                .expect("tool_calls")
                .iter()
                .map(|tc| tc["function"].clone())
                .collect()
        };
        assert_eq!(calls.len(), 2, "stream {stream}: {calls:?}");
        assert_eq!(
            calls[0],
            json!({"name": "get_weather", "arguments": r#"{"location":"Seoul"}"#}),
            "stream {stream}"
        );
        assert_eq!(calls[1]["name"], "get_weather", "stream {stream}");
        let second: Value =
            serde_json::from_str(calls[1]["arguments"].as_str().expect("text")).expect("JSON");
        assert_eq!(second["location"], "Busan", "stream {stream}: {second}");
        assert!(
            second["days"].is_i64(),
            "stream {stream}: the watch held the second call's days to an integer: {second}"
        );
    }
}

/// Calls read from a reply render back through the template to the markup they were read from: the parser's typing
/// is the template's own (a string raw, the rest JSON with sorted keys), byte for byte.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_calls_round_trip_through_the_template() {
    let addr = mock("");
    let tools = tools();
    let r = post(
        addr,
        "/apply-template",
        &json!({"messages": [
            {"role": "user", "content": "Note it."},
            {"role": "assistant", "content": "", "reasoning_content": "why", "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "save_note", "arguments": {
                    "z": "last", "a": "first", "n": {"y": [1, 2], "x": true}}}},
                {"id": "c1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
            ]},
        ], "tools": tools, "add_generation_prompt": false}),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let text = v["prompt"].as_str().expect("a prompt");
    let turn = concat!(
        "<|ernie:im_start|>assistant\n<ernie:think>why</ernie:think><ernie:tool_call>\n",
        "<invoke name=\"save_note\"><parameter name=\"a\">first</parameter>",
        "<parameter name=\"n\">{\"x\": true, \"y\": [1, 2]}</parameter><parameter name=\"z\">last</parameter>",
        "</invoke>\n<invoke name=\"get_time\"></invoke>\n</ernie:tool_call><|ernie:im_end|>",
    );
    assert!(text.ends_with(turn), "{text}");
    // Read back by the parser, the calls are the ones sent.
    const HEADER: &str = "<|ernie:im_start|>assistant\n<ernie:think>";
    let output = &turn[HEADER.len()..turn.len() - "<|ernie:im_end|>".len()];
    let read = read(&mut parser(&prompt(true), Some(&tools)), &[output]).expect("parses");
    assert_eq!(read.reasoning, "why");
    assert_eq!(read.content, "");
    assert_eq!(
        read.calls
            .iter()
            .map(|c| (c.name.as_str(), c.arguments.as_str()))
            .collect::<Vec<_>>(),
        [
            (
                "save_note",
                r#"{"a":"first","n":{"x":true,"y":[1,2]},"z":"last"}"#
            ),
            ("get_time", "{}"),
        ]
    );
}

/// Markup that does not parse is the request's named error: a 500 without a stream, an error event that ends a
/// stream (no `[DONE]`), for a malformed call and for one the generation leaves open.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_malformed_call_is_an_api_error() {
    for (script, want) in [
        (
            "</ernie:think><ernie:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Seoul\
             </parameter><parameter name=\"days\">three</parameter></invoke></ernie:tool_call>",
            "tool-call markup: tool call 0 (get_weather): parameter days is typed integer, \
             and its value is not JSON",
        ),
        (
            "</ernie:think>Checking.<ernie:tool_call><invoke name=\"get_weather\"><parameter name=\"location\">Seoul",
            "tool-call markup: tool call block 0 is not closed by </ernie:tool_call>",
        ),
        (
            "</ernie:think><ernie:tool_call>hello",
            "tool-call markup: tool call block 0 opens no invoke",
        ),
    ] {
        common::check_api_error(
            mock(script),
            |stream| common::tool_body(tools(), stream),
            want,
            fold_stream,
        );
    }
}

/// A forced call follows the protocol's own text: a named tool after the model's own reasoning and the span's close,
/// in the markup the template writes a call in, on the chat route whole and streamed. A script that ends inside the
/// span is an error naming the missing call.
#[test]
#[ignore = "gate: just gate-serve"]
fn hw_ernie55_a_forced_call_follows_the_protocols_own_text() {
    const REASONING: &str = "r <ernie:tool_call> s";
    let named = json!({"type": "function", "function": {"name": "get_time"}});
    let called = json!([{"name": "get_time", "arguments": "{}"}]);
    // Two tools the structured-output subset compiles: a string and an integer parameter.
    let served = json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
            "properties": {"location": {"type": "string"}, "days": {"type": "integer"}}}}},
        {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
    ]);
    let mut body = common::tool_body(served, false);
    body["tool_choice"] = named;
    let addr = mock(&format!("{REASONING}</ernie:think>"));
    let r = post(addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = r.json();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{v}");
    assert_eq!(c["message"]["reasoning_content"], REASONING, "{v}");
    let calls: Vec<Value> = c["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls")
        .iter()
        .map(|tc| tc["function"].clone())
        .collect();
    assert_eq!(json!(calls), called, "{v}");
    body["stream"] = json!(true);
    let r = post(addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let (finish, folded) = fold_stream(&r.body).expect("no error event");
    assert_eq!(finish, "tool_calls");
    assert_eq!(folded["reasoning_content"], REASONING);
    assert_eq!(folded["calls"], called);
    let addr = mock(REASONING);
    body["stream"] = json!(false);
    let r = post(addr, "/v1/chat/completions", &body);
    assert_eq!(r.status, 500, "{}", r.body);
    assert!(r.body.contains("before the forced tool call"), "{}", r.body);
}
