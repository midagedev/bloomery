//! The step arm against scripted rows: a request's fields to a program, the
//! program's pick at every token, and each format's inverse read back by its
//! own scanner.

use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::program::{self, Env};
use super::{Ask, Constraint, Part, Refusal, Vocab, Walk, request};
use crate::api::logprobs::Ask as Probs;
use crate::dsml::{ChatParser, DsmlScan, MarkupError, ToolFormat, Tools};
use crate::engine::{Decoder, Engine, EngineError, SamplingParams, Tokenizer};
use crate::genloop::{GenParams, Outcome, Slot, StopKind, Timings, generate};
use crate::glmxml::{ArgTypes, GlmScan};
use crate::hermes::HermesScan;
use crate::mock::SPECIALS;
use crate::qwenxml::{ParamKinds, QwenScan};
use crate::reasoning::{ReasoningFormat, ThinkEntry};
use crate::sampling;

/// The mock's ids: its specials, then a byte each, then these pieces.
const BYTES: u32 = SPECIALS.len() as u32;
const PIECES: u32 = BYTES + 256;

/// The mock's vocabulary with merged pieces on top: a greedy longest match, so
/// a piece like `",` is one id where a byte-level vocabulary spells two.
struct PieceTok(&'static [&'static str]);

impl PieceTok {
    fn bytes_of(&self, id: u32) -> Vec<u8> {
        match id {
            i if i < BYTES => SPECIALS[i as usize].as_bytes().to_vec(),
            i if i < PIECES => vec![u8::try_from(i - BYTES).expect("a byte")],
            i => self.0[(i - PIECES) as usize].as_bytes().to_vec(),
        }
    }

    fn piece(&self, text: &str) -> u32 {
        PIECES + u32::try_from(self.0.iter().position(|p| *p == text).expect("a piece")).unwrap()
    }
}

struct PieceDecoder {
    tok: Arc<PieceTok>,
    held: Vec<u8>,
}

impl Decoder for PieceDecoder {
    fn push(&mut self, id: u32) -> Option<String> {
        self.held.extend(self.tok.bytes_of(id));
        let upto = match std::str::from_utf8(&self.held) {
            Ok(_) => self.held.len(),
            Err(e) => e.valid_up_to(),
        };
        if upto == 0 {
            return None;
        }
        let text = String::from_utf8_lossy(&self.held[..upto]).into_owned();
        self.held.drain(..upto);
        Some(text)
    }
    fn flush(&mut self) -> String {
        String::from_utf8_lossy(&std::mem::take(&mut self.held)).into_owned()
    }
}

/// A stop id besides the end of sequence, as a piece.
const STOP_PIECE: &str = "[END]";

/// A [`PieceTok`] the decoder can share.
struct Pieces(Arc<PieceTok>);

impl Tokenizer for Pieces {
    fn encode(&self, text: &str) -> Vec<u32> {
        let b = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            let specials = SPECIALS.iter().enumerate().map(|(k, s)| (k as u32, *s));
            let pieces = self
                .0
                .0
                .iter()
                .enumerate()
                .map(|(k, s)| (PIECES + k as u32, *s));
            let best = specials
                .chain(pieces)
                .filter(|(_, s)| b[i..].starts_with(s.as_bytes()))
                .max_by_key(|(_, s)| s.len());
            match best {
                Some((id, s)) => {
                    out.push(id);
                    i += s.len();
                }
                None => {
                    out.push(BYTES + u32::from(b[i]));
                    i += 1;
                }
            }
        }
        out
    }
    fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.0.bytes_of(i)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
    fn decoder(&self) -> Box<dyn Decoder> {
        Box::new(PieceDecoder {
            tok: Arc::clone(&self.0),
            held: Vec::new(),
        })
    }
    fn bos(&self) -> u32 {
        0
    }
    fn eos(&self) -> u32 {
        1
    }
    fn stops(&self) -> Vec<u32> {
        let piece = self.0.0.iter().position(|p| *p == STOP_PIECE);
        let extra = piece.map(|k| PIECES + u32::try_from(k).expect("a piece"));
        [self.eos()].into_iter().chain(extra).collect()
    }
    fn add_bos(&self) -> bool {
        false
    }
    fn n_vocab(&self) -> usize {
        PIECES as usize + self.0.0.len()
    }
}

fn bytes_tok() -> Arc<dyn Tokenizer> {
    Arc::new(Pieces(Arc::new(PieceTok(&[]))))
}

/// The bytes' vocabulary with [`STOP_PIECE`] a second stop id.
fn stop_tok() -> Arc<dyn Tokenizer> {
    Arc::new(Pieces(Arc::new(PieceTok(&[STOP_PIECE]))))
}

/// An engine whose row at its `k`-th step favours `want[k]` (the end of
/// generation once `want` is spent) over a fixed pseudo-random floor. `read`
/// holds, for each step, whether the caller asked for its row.
struct Rows {
    tok: Arc<dyn Tokenizer>,
    want: Vec<u32>,
    k: usize,
    read: Arc<Mutex<Vec<bool>>>,
}

impl Engine for Rows {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::clone(&self.tok)
    }
    fn prefill(&mut self, _ids: &[u32]) -> Result<(), EngineError> {
        Ok(())
    }
    fn next(&mut self, _last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.read.lock().expect("the log").push(out.is_some());
        let n = self.tok.n_vocab();
        let mut row: Vec<f32> = (0..n)
            .map(|i| ((i * 37 + self.k * 11) % 17) as f32 * 0.1)
            .collect();
        let favoured = self
            .want
            .get(self.k)
            .copied()
            .unwrap_or_else(|| self.tok.eos());
        row[favoured as usize] = 10.0;
        self.k += 1;
        let best = sampling::argmax(&row);
        if let Some(out) = out {
            out.copy_from_slice(&row);
        }
        Ok(best)
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.k = 0;
        Ok(())
    }
    fn ctx_max(&self) -> usize {
        4096
    }
    fn describe(&self) -> String {
        "rows".to_owned()
    }
}

fn params(constraint: Option<Constraint>, think: ThinkEntry) -> GenParams {
    GenParams {
        n_predict: 400,
        sampling: SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        },
        stop: Vec::new(),
        ignore_eos: false,
        stream: false,
        timings_per_token: false,
        return_progress: false,
        include_usage: false,
        cache_prompt: true,
        reasoning_budget: None,
        think_entry: think,
        logprobs: None,
        constraint,
    }
}

/// One generation over rows that favour `want`, with whether each step's row
/// was asked for.
fn run_read(
    tok: &Arc<dyn Tokenizer>,
    constraint: Option<Constraint>,
    think: ThinkEntry,
    want: &[u32],
    logprobs: Option<Probs>,
) -> (Outcome, Vec<bool>) {
    let read = Arc::new(Mutex::new(Vec::new()));
    let mut slot = Slot::new(Box::new(Rows {
        tok: Arc::clone(tok),
        want: want.to_vec(),
        k: 0,
        read: Arc::clone(&read),
    }));
    let mut p = params(constraint, think);
    p.logprobs = logprobs;
    let o = generate(
        &mut slot,
        &sampling::reference_factory(),
        &tok.encode("hi"),
        &p,
        &mut |_| Ok(()),
        &mut |_| {},
        &mut Timings::default(),
    )
    .expect("a generation");
    let read = read.lock().expect("the log").clone();
    (o, read)
}

/// One generation over rows that favour `want`.
fn run(
    tok: &Arc<dyn Tokenizer>,
    constraint: Option<Constraint>,
    think: ThinkEntry,
    want: &[u32],
) -> Outcome {
    run_read(tok, constraint, think, want, None).0
}

fn build(
    tok: &Arc<dyn Tokenizer>,
    format: ToolFormat,
    body: &Value,
) -> Result<Constraint, Refusal> {
    let vocab = Vocab::read(&**tok);
    request(
        body.as_object().expect("a body"),
        &Ask {
            tok: &**tok,
            vocab: &vocab,
            format,
        },
    )
    .map(|c| c.expect("the body asks for a constraint"))
}

fn schema_body(schema: &Value) -> Value {
    json!({"response_format": {"type": "json_schema",
        "json_schema": {"name": "x", "strict": true, "schema": schema}}})
}

fn ids(tok: &Arc<dyn Tokenizer>, text: &str) -> Vec<u32> {
    tok.encode(text)
}

fn with_eos(tok: &Arc<dyn Tokenizer>, mut v: Vec<u32>) -> Vec<u32> {
    v.push(tok.eos());
    v
}

fn weather_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "day": {"type": "integer", "enum": [1, 2, 3]},
            "ok": {"type": "boolean"},
            "kind": {"const": "weather"},
            "note": {"type": "string", "maxLength": 8}
        },
        "required": ["unit", "day", "ok", "kind", "note"],
        "additionalProperties": false
    })
}

const WEATHER: &str =
    r#"{"unit": "celsius", "day": 2, "ok": true, "kind": "weather", "note": "hi"}"#;

/// Where greedy already obeys the schema, the step arm is plain greedy: the
/// same ids, the same text, the same stop.
#[test]
fn the_step_arm_is_greedy_where_greedy_obeys() {
    let tok = bytes_tok();
    let want = ids(&tok, WEATHER);
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&weather_schema())).expect("served");
    let plain = run(&tok, None, ThinkEntry::Closed, &want);
    let stepped = run(&tok, Some(c), ThinkEntry::Closed, &want);
    assert_eq!(plain.tokens, with_eos(&tok, want));
    assert_eq!(stepped.tokens, plain.tokens);
    assert_eq!(stepped.content, WEATHER);
    assert_eq!(stepped.stop, StopKind::Eos);
}

/// Where greedy does not obey, the mask does: rows that favour one wrong
/// letter everywhere still give a document the schema accepts, the free
/// string cut at its `maxLength`.
#[test]
fn the_mask_holds_when_greedy_does_not() {
    let tok = bytes_tok();
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&weather_schema())).expect("served");
    let wrong = vec![BYTES + u32::from(b'Z'); 200];
    let plain = run(&tok, None, ThinkEntry::Closed, &wrong);
    assert!(serde_json::from_str::<Value>(&plain.content).is_err());
    let o = run(&tok, Some(c), ThinkEntry::Closed, &wrong);
    let v: Value =
        serde_json::from_str(&o.content).unwrap_or_else(|e| panic!("{e}: {}", o.content));
    assert!(["celsius", "fahrenheit"].contains(&v["unit"].as_str().expect("unit")));
    assert!([1, 2, 3].contains(&v["day"].as_i64().expect("day")));
    assert!(v["ok"].is_boolean());
    assert_eq!(v["kind"], "weather");
    assert_eq!(v["note"], "ZZZZZZZZ");
    assert_eq!(o.stop, StopKind::Eos);
}

/// A free string's end token takes the front of the text after it, and the
/// rest of that text is tokenised from there: `", "` is one id, `b": ` four.
#[test]
fn an_end_token_absorbs_the_front_of_the_next_text() {
    const PIECE: &str = "\", \"";
    let pieces = Arc::new(PieceTok(&[PIECE, "\"}", "\","]));
    let merged = pieces.piece(PIECE);
    let tok: Arc<dyn Tokenizer> = Arc::new(Pieces(pieces));
    let schema = json!({"type": "object", "properties": {
        "a": {"type": "string"}, "b": {"type": "boolean"}}});
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&schema)).expect("served");
    let mut want = ids(&tok, "{\"a\": \"");
    want.push(BYTES + u32::from(b'x'));
    want.push(merged);
    want.extend(ids(&tok, "b\": "));
    want.extend(ids(&tok, "true}"));
    let o = run(&tok, Some(c), ThinkEntry::Closed, &want);
    assert_eq!(o.tokens, with_eos(&tok, want));
    assert_eq!(o.content, "{\"a\": \"x\", \"b\": true}");
}

/// A prompt that left the think span open gets it closed first.
#[test]
fn an_open_think_span_is_closed_before_the_program() {
    let tok = bytes_tok();
    let c = build(
        &tok,
        ToolFormat::Hermes,
        &schema_body(&json!({"type": "boolean"})),
    )
    .unwrap();
    let wrong = vec![BYTES + u32::from(b'Z'); 50];
    let open = run(&tok, Some(c.clone()), ThinkEntry::Open, &wrong);
    let close = ids(&tok, "</think>\n\n");
    assert_eq!(open.tokens[..close.len()], close);
    let closed = run(&tok, Some(c), ThinkEntry::Closed, &wrong);
    assert!(closed.content.starts_with(['t', 'f']), "{}", closed.content);
}

/// The step errors are named: a free string at its limit with no end id in
/// the row, and a row that holds none of the ids.
#[test]
fn a_walk_that_cannot_step_is_a_named_error() {
    let tok = bytes_tok();
    let c = build(
        &tok,
        ToolFormat::Hermes,
        &schema_body(&json!({"type": "string", "maxLength": 0})),
    )
    .unwrap();
    let mut w = Walk::new(&c, false);
    assert_eq!(w.choose(&[]), Ok(BYTES + u32::from(b'"')));
    let e = w.choose(&[]).expect_err("no id in an empty row");
    assert!(e.to_string().contains("length limit"), "{e}");
    let c = build(
        &tok,
        ToolFormat::Hermes,
        &schema_body(&json!({"enum": ["a", "b"]})),
    )
    .unwrap();
    let mut w = Walk::new(&c, false);
    w.choose(&[]).expect("the opening quote is forced");
    let e = w.choose(&[]).expect_err("no option id in an empty row");
    assert!(e.to_string().contains("none of the option ids"), "{e}");
}

/// Every refusal is named, with the path of the keyword.
#[test]
fn each_refusal_names_its_field() {
    let tok = bytes_tok();
    let node = |s: Value| schema_body(&json!({"type": "object", "properties": {"p": s}}));
    let tools = json!([{"type": "function", "function": {"name": "f"}}]);
    let cases: Vec<(Value, ToolFormat, &str)> = vec![
        (
            node(json!({"type": "string", "pattern": "a+"})),
            ToolFormat::Hermes,
            "schema.properties.p: the keyword \"pattern\"",
        ),
        (
            node(json!({"$ref": "#/x"})),
            ToolFormat::Hermes,
            "the keyword \"$ref\"",
        ),
        (
            node(json!({"anyOf": [{"type": "string"}]})),
            ToolFormat::Hermes,
            "the keyword \"anyOf\"",
        ),
        (
            node(json!({"type": "integer"})),
            ToolFormat::Hermes,
            "type \"integer\" is not supported by this server without an enum",
        ),
        (
            node(json!({"type": "array", "items": {"type": "string"}})),
            ToolFormat::Hermes,
            "the keyword \"items\"",
        ),
        (
            node(json!({"type": "number", "enum": [1.5]})),
            ToolFormat::Hermes,
            "not an integer",
        ),
        (
            node(json!({"enum": [1, "a"]})),
            ToolFormat::Hermes,
            "mixed kinds",
        ),
        (
            node(json!({"enum": ["a", "a"]})),
            ToolFormat::Hermes,
            "listed twice",
        ),
        (
            node(json!({"enum": ["a", ""]})),
            ToolFormat::Hermes,
            "an empty string",
        ),
        (
            node(json!({"enum": [1, 10]})),
            ToolFormat::Hermes,
            "one is a prefix of the other",
        ),
        (
            node(json!({"enum": ["a", "ab"]})),
            ToolFormat::Hermes,
            "one is a prefix of the other",
        ),
        (
            node(json!({"type": "string", "enum": [1]})),
            ToolFormat::Hermes,
            "does not match",
        ),
        (
            node(json!({"type": "string", "maxLength": -1})),
            ToolFormat::Hermes,
            "maxLength must be",
        ),
        (
            node(json!({"type": "boolean", "maxLength": 3})),
            ToolFormat::Hermes,
            "beside type boolean",
        ),
        (
            node(json!({"description": "no type"})),
            ToolFormat::Hermes,
            "needs a type, an enum or a const",
        ),
        (
            schema_body(&json!({"type": "object", "required": ["q"]})),
            ToolFormat::Hermes,
            "required names \"q\"",
        ),
        (
            json!({"response_format": {"type": "json_object"}}),
            ToolFormat::Hermes,
            "json_object",
        ),
        (
            json!({"response_format": {"type": "xml"}}),
            ToolFormat::Hermes,
            "response_format",
        ),
        (
            json!({"response_format": {"type": "json_schema"}}),
            ToolFormat::Hermes,
            "json_schema.schema",
        ),
        (
            json!({"json_schema": {"type": "object"}}),
            ToolFormat::Hermes,
            "json_schema is not supported",
        ),
        (
            json!({"tool_choice": "required"}),
            ToolFormat::Hermes,
            "needs the request's tools",
        ),
        (
            json!({"tool_choice": "sometimes", "tools": tools}),
            ToolFormat::Hermes,
            "\"sometimes\"",
        ),
        (
            json!({"tool_choice": {"type": "function", "function": {"name": "g"}}, "tools": tools}),
            ToolFormat::Hermes,
            "the function g is not among",
        ),
        (
            json!({"tool_choice": "required", "tools": tools,
                "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "boolean"}}}}),
            ToolFormat::Hermes,
            "has one program",
        ),
        (
            json!({"tool_choice": "required", "tools": [{"function": {"name": "a<b"}}]}),
            ToolFormat::Hermes,
            "not a plain name",
        ),
        (
            json!({"tool_choice": "required", "tools": [{"function": {"name": "f", "parameters":
            {"type": "object", "properties": {"q": {"enum": ["a"]}}}}}]}),
            ToolFormat::GlmXml,
            "under \"type\": \"string\"",
        ),
    ];
    for (body, format, needle) in cases {
        let e = build(&tok, format, &body).expect_err(&body.to_string());
        assert!(e.message.contains(needle), "{body}: {e}");
        assert!(!e.unsupported, "{body}: {e}");
    }
    let e = build(
        &tok,
        ToolFormat::Unparsed,
        &json!({"tool_choice": "required", "tools": tools}),
    )
    .expect_err("no markup");
    assert!(e.unsupported && e.message.contains("no inverse"), "{e}");
}

fn forecast_tools() -> Value {
    json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer", "enum": [1, 2, 3]}}}}},
        {"type": "function", "function": {"name": "search", "parameters": {
            "type": "object", "properties": {
                "query": {"type": "string", "maxLength": 16},
                "deep": {"type": "boolean"}}}}}
    ])
}

/// The text each format writes for `get_weather(unit="celsius", days=2)` and
/// for `search(query="cats", deep=false)`.
fn markup(format: ToolFormat) -> [&'static str; 2] {
    match format {
        ToolFormat::Hermes => [
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"unit\": \"celsius\", \"days\": 2}}\n</tool_call>",
            "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"query\": \"cats\", \"deep\": false}}\n</tool_call>",
        ],
        ToolFormat::QwenXml => [
            "<tool_call>\n<function=get_weather>\n<parameter=unit>\ncelsius\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=search>\n<parameter=query>\ncats\n</parameter>\n<parameter=deep>\nfalse\n</parameter>\n</function>\n</tool_call>",
        ],
        ToolFormat::GlmXml => [
            "<tool_call>get_weather<arg_key>unit</arg_key><arg_value>celsius</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>",
            "<tool_call>search<arg_key>query</arg_key><arg_value>cats</arg_value><arg_key>deep</arg_key><arg_value>false</arg_value></tool_call>",
        ],
        ToolFormat::Dsml => [
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"search\">\n<｜DSML｜parameter name=\"query\" string=\"true\">cats</｜DSML｜parameter>\n<｜DSML｜parameter name=\"deep\" string=\"false\">false</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ],
        ToolFormat::Unparsed => unreachable!("no markup"),
    }
}

/// The calls the format's own scanner reads out of `text`: name and
/// arguments.
fn scanned(format: ToolFormat, tools: &Value, text: &str) -> Vec<(String, Value)> {
    let calls = match format {
        ToolFormat::Hermes => {
            let mut s = HermesScan::default();
            let mut out = s.push(text).expect("parses");
            out.calls.extend(s.finish().expect("closes").calls);
            out
        }
        ToolFormat::QwenXml => {
            let mut s = QwenScan::new(ParamKinds::of_tools(Some(tools)));
            let mut out = s.push(text).expect("parses");
            out.calls.extend(s.finish().expect("closes").calls);
            out
        }
        ToolFormat::GlmXml => {
            let mut s = GlmScan::new(ArgTypes::of_tools(Some(tools)));
            let mut out = s.push(text).expect("parses");
            out.calls.extend(s.finish().expect("closes").calls);
            out
        }
        ToolFormat::Dsml => {
            let mut s = DsmlScan::default();
            let mut out = s.push(text);
            out.calls.extend(s.finish().calls);
            out
        }
        ToolFormat::Unparsed => unreachable!("no markup"),
    };
    assert!(
        calls.content.is_empty(),
        "{format:?}: content {:?}",
        calls.content
    );
    calls
        .calls
        .into_iter()
        .map(|c| {
            (
                c.name,
                serde_json::from_str(&c.arguments).expect("JSON arguments"),
            )
        })
        .collect()
}

/// Each format's inverse is read back by its own scanner: a forced call of
/// either tool (a choice of name, then that tool's arguments, a free string
/// among them) is one call with the arguments written, and a named call
/// skips the choice.
#[test]
fn each_formats_inverse_reads_back_through_its_scanner() {
    let tok = bytes_tok();
    let tools = forecast_tools();
    for format in [
        ToolFormat::Hermes,
        ToolFormat::QwenXml,
        ToolFormat::GlmXml,
        ToolFormat::Dsml,
    ] {
        let [weather, search] = markup(format);
        let required = json!({"tools": tools, "tool_choice": "required"});
        let named = json!({"tools": tools, "tool_choice":
            {"type": "function", "function": {"name": "search"}}});
        let flat = json!({"tools": tools, "tool_choice": {"type": "function", "name": "search"}});
        for (body, text, name, args) in [
            (
                &required,
                weather,
                "get_weather",
                json!({"unit": "celsius", "days": 2}),
            ),
            (
                &required,
                search,
                "search",
                json!({"query": "cats", "deep": false}),
            ),
            (
                &named,
                search,
                "search",
                json!({"query": "cats", "deep": false}),
            ),
            (
                &flat,
                search,
                "search",
                json!({"query": "cats", "deep": false}),
            ),
        ] {
            let c = build(&tok, format, body).unwrap_or_else(|e| panic!("{format:?}: {e}"));
            let want = ids(&tok, text);
            let o = run(&tok, Some(c), ThinkEntry::Closed, &want);
            assert_eq!(o.tokens, with_eos(&tok, want), "{format:?} {name}");
            assert_eq!(o.content, text, "{format:?} {name}");
            assert_eq!(
                scanned(format, &tools, &o.content),
                vec![(name.to_owned(), args)],
                "{format:?}"
            );
        }
    }
}

/// A forced call obeys the tools when greedy would not: rows that favour one
/// wrong letter give a call to one of the tools, its enum values among the
/// listed ones.
#[test]
fn a_forced_call_obeys_the_tools_when_greedy_does_not() {
    let tok = bytes_tok();
    let tools = forecast_tools();
    let wrong = vec![BYTES + u32::from(b'Z'); 300];
    for format in [
        ToolFormat::Hermes,
        ToolFormat::QwenXml,
        ToolFormat::GlmXml,
        ToolFormat::Dsml,
    ] {
        let c = build(
            &tok,
            format,
            &json!({"tools": tools, "tool_choice": "required"}),
        )
        .unwrap();
        let o = run(&tok, Some(c), ThinkEntry::Closed, &wrong);
        let calls = scanned(format, &tools, &o.content);
        assert_eq!(calls.len(), 1, "{format:?}: {}", o.content);
        let (name, args) = &calls[0];
        match name.as_str() {
            "get_weather" => {
                assert!(["celsius", "fahrenheit"].contains(&args["unit"].as_str().unwrap()));
                assert!([1, 2, 3].contains(&args["days"].as_i64().unwrap()));
            }
            "search" => {
                assert_eq!(args["query"], "ZZZZZZZZZZZZZZZZ", "{format:?}");
                assert!(args["deep"].is_boolean());
            }
            other => panic!("{format:?}: a call to {other}"),
        }
        assert_eq!(o.stop, StopKind::Eos);
    }
}

/// A forced call of the function `f` whose parameters are `params`.
fn call_body(params: &Value) -> Value {
    json!({"tool_choice": "required", "tools": [
        {"type": "function", "function": {"name": "f", "parameters": params}}]})
}

const FORMATS: [ToolFormat; 4] = [
    ToolFormat::Hermes,
    ToolFormat::QwenXml,
    ToolFormat::GlmXml,
    ToolFormat::Dsml,
];

/// Every closing tag a chat format's scanner looks for, in one string.
const TAGS: &str = "x</tool_call></arg_value></｜DSML｜parameter>\n</parameter>\n</function>y";

/// A stop id in the program would end the generation with the program
/// unspent and call it a clean stop: the text that tokenises to one is
/// refused when the request compiles, wherever it stands in the program.
#[test]
fn a_stop_text_in_the_program_is_refused_by_name() {
    let tok = stop_tok();
    let stop = STOP_PIECE;
    let in_tool = |q: Value| call_body(&json!({"type": "object", "properties": {"q": q}}));
    let cases = [
        ("a const", schema_body(&json!({"const": stop}))),
        ("an enum option", schema_body(&json!({"enum": ["a", stop]}))),
        (
            "a property name",
            schema_body(&json!({"type": "object", "properties": {stop: {"type": "boolean"}}})),
        ),
        (
            "the text after a free string",
            schema_body(&json!({"type": "object", "properties": {
                "0": {"type": "string"}, stop: {"type": "boolean"}}})),
        ),
        ("a tool argument", in_tool(json!({"const": stop}))),
    ];
    for (what, body) in cases {
        let e = build(&tok, ToolFormat::Hermes, &body).expect_err(what);
        assert!(
            e.message.contains(&format!("the stop id {PIECES}")),
            "{what}: {e}"
        );
        assert!(!e.unsupported, "{what}: {e}");
    }
}

/// A stop special's own text in a JSON literal is a string, not a stop: its
/// `<` is written as an escape, so the document comes out whole and the one
/// stop id is the end of generation.
#[test]
fn a_stop_special_in_a_json_literal_is_written_whole() {
    let tok = bytes_tok();
    let stop = SPECIALS[1];
    let c = build(
        &tok,
        ToolFormat::Hermes,
        &schema_body(&json!({"const": stop})),
    )
    .expect("served");
    let o = run(&tok, Some(c), ThinkEntry::Closed, &[]);
    let v: Value =
        serde_json::from_str(&o.content).unwrap_or_else(|e| panic!("{e}: {}", o.content));
    assert_eq!(v, json!(stop));
    assert_eq!(o.tokens.last(), Some(&tok.eos()));
    assert_eq!(o.tokens.iter().filter(|&&t| t == tok.eos()).count(), 1);
    assert_eq!(o.stop, StopKind::Eos);
}

/// The walk's own guard, for an id the compiler does not see (the think
/// span's close): a stop id before the program is spent is a named error, and
/// after it the stop ids are what the walk takes.
#[test]
fn a_walk_takes_no_stop_id_before_the_program_is_spent() {
    let tok = bytes_tok();
    let vocab = Vocab::read(&*tok);
    let env = Env {
        tok: &*tok,
        vocab: &vocab,
    };
    let prog = program::compile(&[Part::Text("{".to_owned())], &env).expect("compiles");
    let c = Constraint::new(prog, vec![tok.eos()], Arc::clone(vocab.stops()));
    let e = Walk::new(&c, true)
        .choose(&[])
        .expect_err("a stop id before the program is spent");
    assert!(e.to_string().contains("before the program is spent"), "{e}");
    let mut w = Walk::new(&c, false);
    assert_eq!(w.choose(&[]), Ok(BYTES + u32::from(b'{')));
    let mut row = vec![0.0; tok.n_vocab()];
    row[tok.eos() as usize] = 1.0;
    assert_eq!(w.choose(&row), Ok(tok.eos()));
}

/// A tool's parameters are read by the one schema compiler: a keyword it does
/// not read, or a `required` naming nothing, is refused by name, never
/// dropped.
#[test]
fn a_tool_parameter_schema_is_read_whole() {
    let tok = bytes_tok();
    let q = json!({"q": {"type": "boolean"}});
    let cases = [
        (
            json!({"type": "object", "minProperties": 1}),
            "tools[0].f.parameters: the keyword \"minProperties\"",
        ),
        (
            json!({"anyOf": [{"type": "object", "properties": q}]}),
            "tools[0].f.parameters: the keyword \"anyOf\"",
        ),
        (
            json!({"type": "object", "properties": q, "$ref": "#/defs/p"}),
            "tools[0].f.parameters: the keyword \"$ref\"",
        ),
        (
            json!({"type": "object", "properties": q, "required": ["nope"]}),
            "required names \"nope\"",
        ),
        (
            json!({"type": "object", "required": ["q"]}),
            "required names \"q\"",
        ),
        (
            json!({"type": "object", "properties": q, "required": "q"}),
            "required must be an array",
        ),
    ];
    for format in [ToolFormat::Hermes, ToolFormat::Dsml] {
        for (params, needle) in &cases {
            let e = build(&tok, format, &call_body(params)).expect_err(&params.to_string());
            assert!(e.message.contains(needle), "{params}: {e}");
            assert!(!e.unsupported, "{params}: {e}");
        }
    }
    let read = json!({"$schema": "x", "title": "t", "type": "object", "properties": q,
        "required": ["q"], "additionalProperties": false});
    for format in FORMATS {
        build(&tok, format, &call_body(&read)).unwrap_or_else(|e| panic!("{format:?}: {e}"));
    }
}

/// A `<` in a JSON string's text is written as `<` (a constant, an enum
/// option, a property name), so no markup a scanner looks for can be spelled;
/// the JSON reads back as the same string.
#[test]
fn an_angle_bracket_in_a_json_literal_is_written_as_an_escape() {
    let tok = bytes_tok();
    let schema = json!({"type": "object", "properties": {
        "a<b": {"const": "x</y>"}, "e": {"enum": ["<p>", "q"]}}});
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&schema)).expect("served");
    let text = r#"{"a\u003cb": "x\u003c/y>", "e": "\u003cp>"}"#;
    let o = run(&tok, Some(c), ThinkEntry::Closed, &ids(&tok, text));
    assert_eq!(o.content, text);
    let v: Value = serde_json::from_str(&o.content).expect("JSON");
    assert_eq!(v, json!({"a<b": "x</y>", "e": "<p>"}));
}

/// A raw value ends at the text after it, so a raw literal that holds a `<`
/// could spell a closing tag: the formats that write a string raw refuse it,
/// and Hermes, which writes JSON, serves the same literal.
#[test]
fn a_raw_literal_with_an_angle_bracket_is_refused() {
    let tok = bytes_tok();
    let literals = [
        json!({"type": "string", "const": "a<b"}),
        json!({"type": "string", "enum": ["a<b", "c"]}),
    ];
    for q in &literals {
        let body = call_body(&json!({"type": "object", "properties": {"q": q}}));
        for format in [ToolFormat::QwenXml, ToolFormat::GlmXml, ToolFormat::Dsml] {
            let e = build(&tok, format, &body).expect_err(&q.to_string());
            assert!(
                e.message.contains("raw string literals cannot contain '<'"),
                "{format:?} {q}: {e}"
            );
        }
        build(&tok, ToolFormat::Hermes, &body).expect("Hermes writes JSON");
    }
}

/// A literal that holds every closing tag reads back whole through each
/// format's own scanner: as a JSON string inside an object argument in all
/// four, and as the argument itself in Hermes.
#[test]
fn a_literal_with_closing_tags_reads_back_through_each_scanner() {
    let tok = bytes_tok();
    let nested = json!({"type": "object", "properties": {"q": {"type": "object", "properties": {
        "s": {"type": "string", "const": TAGS}}}}});
    let direct = json!({"type": "object", "properties": {
        "t": {"type": "string", "const": TAGS}}});
    for format in FORMATS {
        let mut served = vec![(&nested, json!({"q": {"s": TAGS}}))];
        if format == ToolFormat::Hermes {
            served.push((&direct, json!({"t": TAGS})));
        }
        for (params, args) in served {
            let body = call_body(params);
            let c = build(&tok, format, &body).unwrap_or_else(|e| panic!("{format:?}: {e}"));
            let o = run(&tok, Some(c), ThinkEntry::Closed, &[]);
            let tools = body["tools"].clone();
            assert_eq!(
                scanned(format, &tools, &o.content),
                vec![("f".to_owned(), args)],
                "{format:?}"
            );
        }
    }
}

/// Rows that favour a format's own closing tag, one byte a step, cannot make a
/// free string spell it: a string inside a call's markup, as the argument
/// itself or inside a JSON object a parameter holds, takes no token with a `<`
/// (or, for Qwen's raw string, a newline), and the call reads back as one call.
#[test]
fn a_free_string_cannot_spell_its_formats_closing_tag() {
    let tok = bytes_tok();
    let direct = call_body(&json!({"type": "object", "properties": {
        "q": {"type": "string", "maxLength": 40}}}));
    let nested = call_body(&json!({"type": "object", "properties": {
        "q": {"type": "object", "properties": {"s": {"type": "string", "maxLength": 40}}}}}));
    let tags = [
        (ToolFormat::Hermes, "</tool_call>"),
        (ToolFormat::QwenXml, "\n</parameter>\n"),
        (ToolFormat::GlmXml, "</arg_value>"),
        (ToolFormat::Dsml, "</｜DSML｜parameter>"),
    ];
    for (format, tag) in tags {
        for (body, nested) in [(&direct, false), (&nested, true)] {
            let c = build(&tok, format, body).unwrap_or_else(|e| panic!("{format:?}: {e}"));
            let want: Vec<u32> = tag
                .bytes()
                .map(|b| BYTES + u32::from(b))
                .cycle()
                .take(300)
                .collect();
            let o = run(&tok, Some(c), ThinkEntry::Closed, &want);
            let calls = scanned(format, &body["tools"], &o.content);
            assert_eq!(calls.len(), 1, "{format:?}: {}", o.content);
            let q = &calls[0].1["q"];
            let q = if nested { &q["s"] } else { q };
            let q = q
                .as_str()
                .unwrap_or_else(|| panic!("{format:?}: {}", o.content));
            assert!(!q.contains(tag), "{format:?}: {q:?}");
            if format != ToolFormat::QwenXml || nested {
                assert!(!q.contains('<'), "{format:?} nested {nested}: {q:?}");
            }
        }
    }
}

/// A JSON string in a reply of its own is not inside any markup, so it may
/// hold a `<`: code and HTML in a `response_format` string come out whole,
/// whatever the chat format.
#[test]
fn a_response_format_string_may_hold_an_angle_bracket() {
    let tok = bytes_tok();
    let schema = json!({"type": "object", "properties": {
        "code": {"type": "string", "maxLength": 40}}});
    let code = "for (i = 0; i < n; i++) </p>";
    let text = format!(r#"{{"code": "{code}"}}"#);
    for format in FORMATS {
        let c = build(&tok, format, &schema_body(&schema)).expect("served");
        let o = run(&tok, Some(c), ThinkEntry::Closed, &ids(&tok, &text));
        assert_eq!(o.content, text, "{format:?}");
        let v: Value = serde_json::from_str(&o.content).expect("JSON");
        assert_eq!(v, json!({"code": code}), "{format:?}");
    }
}

/// What the request's chat parser reads out of a reply: the content it
/// leaves and how many calls it found, or the markup it refuses.
fn parsed(format: ToolFormat, tools: &Value, text: &str) -> Result<(String, usize), MarkupError> {
    let scan = match format {
        ToolFormat::Hermes => Tools::Hermes,
        ToolFormat::QwenXml => Tools::QwenXml(ParamKinds::of_tools(Some(tools))),
        ToolFormat::GlmXml => Tools::GlmXml(ArgTypes::of_tools(Some(tools))),
        ToolFormat::Dsml => Tools::Dsml,
        ToolFormat::Unparsed => unreachable!("no markup"),
    };
    let mut p = ChatParser::with_tools("", ReasoningFormat::None, Some(scan));
    let (a, b) = (p.try_push(text)?, p.try_finish()?);
    Ok((a.content + &b.content, a.calls.len() + b.calls.len()))
}

/// A `response_format` beside tools the parser scans for: the reply is
/// scanned for each format's opening tag, so a free string cannot spell it and
/// the reply reads back as content with no call. With `tool_choice` "none"
/// nothing scans the reply and the string may hold a `<`.
#[test]
fn a_response_string_beside_scanned_tools_cannot_open_a_call() {
    let tok = bytes_tok();
    let schema = json!({"type": "object", "properties": {
        "code": {"type": "string", "maxLength": 100}}});
    let blocks = [
        (ToolFormat::Hermes, "<tool_call></tool_call>"),
        (ToolFormat::QwenXml, "<tool_call></tool_call>"),
        (ToolFormat::GlmXml, "<tool_call></tool_call>"),
        (
            ToolFormat::Dsml,
            "<｜DSML｜tool_calls></｜DSML｜tool_calls>",
        ),
    ];
    let mut bad = Vec::new();
    for (format, block) in blocks {
        let mut want = ids(&tok, r#"{"code": ""#);
        want.extend(
            block
                .bytes()
                .map(|b| BYTES + u32::from(b))
                .cycle()
                .take(300),
        );
        for choice in ["auto", "none"] {
            let mut body = schema_body(&schema);
            body["tools"] = forecast_tools();
            body["tool_choice"] = json!(choice);
            let c = build(&tok, format, &body).unwrap_or_else(|e| panic!("{format:?}: {e}"));
            let o = run(&tok, Some(c), ThinkEntry::Closed, &want);
            if choice == "none" {
                assert!(o.content.contains('<'), "{format:?}: {}", o.content);
                continue;
            }
            match parsed(format, &body["tools"], &o.content) {
                Ok((content, 0)) if content == o.content => {}
                Ok((_, calls)) => bad.push(format!("{format:?}: {calls} call(s) in {}", o.content)),
                Err(e) => bad.push(format!("{format:?}: {e}")),
            }
        }
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// An id the program fixes is taken without its row: the only child of a
/// choice's node and every forced id ask the engine for none, the decisions
/// and the stop ask for one, and a request that wants probabilities reads
/// every row.
#[test]
fn a_fixed_id_asks_for_no_row() {
    let tok = bytes_tok();
    let c = build(
        &tok,
        ToolFormat::Hermes,
        &schema_body(&json!({"type": "boolean"})),
    )
    .expect("served");
    let want = ids(&tok, "true");
    // t or f; then r, u, e, each the only child of its node; then a stop.
    let (o, read) = run_read(&tok, Some(c.clone()), ThinkEntry::Closed, &want, None);
    assert_eq!(o.content, "true");
    assert_eq!(read, [true, false, false, false, true]);
    let probs = Probs::new(1, tok.n_vocab());
    let (o, read) = run_read(&tok, Some(c), ThinkEntry::Closed, &want, Some(probs));
    assert_eq!(o.content, "true");
    assert_eq!(read, [true; 5]);
}
