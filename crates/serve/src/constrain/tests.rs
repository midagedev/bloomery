//! The step arm against scripted rows: a request's fields to a program, the
//! program's pick at every token, and each format's inverse read back by its
//! own scanner.

use std::io;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::program::{self, Env};
use super::{Ask, Constraint, Part, Refusal, Vocab, Walk, request};
use crate::api::logprobs::Ask as Probs;
use crate::dsml::{ChatParser, DsmlScan, MarkupError, Spelling, ToolFormat, Tools};
use crate::engine::{Decoder, Engine, EngineError, SamplingParams, Tokenizer};
use crate::genloop::{
    Event, GenError, GenParams, Outcome, Slot, StopKind, Timings, generate, generate_in_quanta,
};
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
    // The rows follow the engine's step count, which is the step arm's.
    let mut slot = Slot::new(Box::new(Rows {
        tok: Arc::clone(tok),
        want: want.to_vec(),
        k: 0,
        read: Arc::clone(&read),
    }))
    .schema_jump(false);
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
fn markup(format: ToolFormat) -> [String; 2] {
    let texts: [&str; 2] = match format {
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
        ToolFormat::Dsml(_) => [
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"search\">\n<｜DSML｜parameter name=\"query\" string=\"true\">cats</｜DSML｜parameter>\n<｜DSML｜parameter name=\"deep\" string=\"false\">false</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ],
        ToolFormat::Unparsed => unreachable!("no markup"),
    };
    texts.map(|t| match format {
        ToolFormat::Dsml(Spelling::Spaced) => t
            .replace("｜DSML｜tool_calls", "｜DSML｜ calls")
            .replace("｜DSML｜invoke", "｜DSML｜ invoke")
            .replace("｜DSML｜parameter", "｜DSML｜ parameter"),
        _ => t.to_owned(),
    })
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
        ToolFormat::Dsml(spelling) => {
            let mut s = DsmlScan::new(spelling);
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
        ToolFormat::Dsml(Spelling::Joined),
        ToolFormat::Dsml(Spelling::Spaced),
    ] {
        let [weather, search] = markup(format);
        let required = json!({"tools": tools, "tool_choice": "required"});
        let named = json!({"tools": tools, "tool_choice":
            {"type": "function", "function": {"name": "search"}}});
        let flat = json!({"tools": tools, "tool_choice": {"type": "function", "name": "search"}});
        for (body, text, name, args) in [
            (
                &required,
                weather.as_str(),
                "get_weather",
                json!({"unit": "celsius", "days": 2}),
            ),
            (
                &required,
                search.as_str(),
                "search",
                json!({"query": "cats", "deep": false}),
            ),
            (
                &named,
                search.as_str(),
                "search",
                json!({"query": "cats", "deep": false}),
            ),
            (
                &flat,
                search.as_str(),
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
        ToolFormat::Dsml(Spelling::Joined),
        ToolFormat::Dsml(Spelling::Spaced),
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

const FORMATS: [ToolFormat; 5] = [
    ToolFormat::Hermes,
    ToolFormat::QwenXml,
    ToolFormat::GlmXml,
    ToolFormat::Dsml(Spelling::Joined),
    ToolFormat::Dsml(Spelling::Spaced),
];

/// Every closing tag a chat format's scanner looks for, in one string.
const TAGS: &str = "x</tool_call></arg_value></｜DSML｜parameter></｜DSML｜ parameter>\n</parameter>\n</function>y";

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
    for format in [
        ToolFormat::Hermes,
        ToolFormat::Dsml(Spelling::Joined),
        ToolFormat::Dsml(Spelling::Spaced),
    ] {
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
        for format in [
            ToolFormat::QwenXml,
            ToolFormat::GlmXml,
            ToolFormat::Dsml(Spelling::Joined),
            ToolFormat::Dsml(Spelling::Spaced),
        ] {
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
        (ToolFormat::Dsml(Spelling::Joined), "</｜DSML｜parameter>"),
        (ToolFormat::Dsml(Spelling::Spaced), "</｜DSML｜ parameter>"),
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
        ToolFormat::Dsml(spelling) => Tools::Dsml(spelling),
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
            ToolFormat::Dsml(Spelling::Joined),
            "<｜DSML｜tool_calls></｜DSML｜tool_calls>",
        ),
        (
            ToolFormat::Dsml(Spelling::Spaced),
            "<｜DSML｜ calls></｜DSML｜ calls>",
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

// The jump arm: a run of ids the program fixes is one prefill call, the step
// that closes it gives the next decision's row.

/// The prompt every jump test generates from: two ids, the first fed by the
/// prompt's call and the last by its step.
const PROMPT: &str = "hi";

/// An engine call after the prompt's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    Prefill(usize),
    Next,
}

/// What an engine was fed.
#[derive(Default)]
struct Seen {
    /// Every id fed since the last reset, in order.
    fed: Vec<u32>,
    calls: Vec<Call>,
}

/// Ids to raise in a row, by the ids generated so far where they are raised.
type Peaks = Vec<(Vec<u32>, Vec<(u32, f32)>)>;

/// An engine whose row is a function of the ids it has been fed and nothing
/// else: a pseudo-random floor of the whole context, and the `peaks` that name
/// ids to raise at an exact context (the ids generated so far). The same
/// context gives the same row whichever calls fed it, so the two arms meet the
/// same rows at the same decisions.
struct Scripted {
    tok: Arc<dyn Tokenizer>,
    seed: u64,
    peaks: Peaks,
    ctx_max: usize,
    seen: Arc<Mutex<Seen>>,
}

impl Scripted {
    fn row(&self, fed: &[u32]) -> Vec<f32> {
        let hash = fed.iter().fold(self.seed ^ 0xcbf2_9ce4_8422_2325, |h, &i| {
            (h ^ u64::from(i)).wrapping_mul(0x0100_0000_01b3)
        });
        let mut row: Vec<f32> = (0..self.tok.n_vocab() as u64)
            .map(|i| {
                let x = (hash ^ i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                    .wrapping_mul(0xff51_afd7_ed55_8ccd);
                ((x >> 40) % 1000) as f32 * 0.004 - 2.0
            })
            .collect();
        let made = &fed[self.tok.encode(PROMPT).len()..];
        for (at, peaks) in &self.peaks {
            if at == made {
                for &(id, v) in peaks {
                    row[id as usize] = v;
                }
            }
        }
        row
    }
}

impl Engine for Scripted {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::clone(&self.tok)
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        let mut s = self.seen.lock().expect("seen");
        s.fed.extend_from_slice(ids);
        s.calls.push(Call::Prefill(ids.len()));
        Ok(())
    }
    fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        let row = {
            let mut s = self.seen.lock().expect("seen");
            s.fed.push(last);
            s.calls.push(Call::Next);
            self.row(&s.fed)
        };
        let best = sampling::argmax(&row);
        if let Some(out) = out {
            out.copy_from_slice(&row);
        }
        Ok(best)
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.seen.lock().expect("seen").fed.clear();
        Ok(())
    }
    fn ctx_max(&self) -> usize {
        self.ctx_max
    }
    fn describe(&self) -> String {
        "scripted".to_owned()
    }
}

/// One generation from [`PROMPT`] on a [`Scripted`] engine, the jump arm on or
/// off, with the engine's calls after the prompt's step and the ids it was fed.
struct Ran {
    out: Outcome,
    calls: Vec<Call>,
    fed: Vec<u32>,
}

/// What the `Scripted` engine of `seed` is asked, the jump arm `jump`, and
/// the request's `n_predict`.
fn scripted(
    tok: &Arc<dyn Tokenizer>,
    c: &Constraint,
    seed: u64,
    peaks: Peaks,
    jump: bool,
    n_predict: i64,
) -> Ran {
    scripted_in(tok, c, seed, peaks, jump, n_predict, 4096)
}

/// [`scripted`] on an engine whose context holds `ctx_max` positions.
fn scripted_in(
    tok: &Arc<dyn Tokenizer>,
    c: &Constraint,
    seed: u64,
    peaks: Peaks,
    jump: bool,
    n_predict: i64,
    ctx_max: usize,
) -> Ran {
    let seen = Arc::new(Mutex::new(Seen::default()));
    let mut slot = Slot::new(Box::new(Scripted {
        tok: Arc::clone(tok),
        seed,
        peaks,
        ctx_max,
        seen: Arc::clone(&seen),
    }))
    .schema_jump(jump);
    let mut p = params(Some(c.clone()), ThinkEntry::Closed);
    p.n_predict = n_predict;
    let out = generate(
        &mut slot,
        &sampling::reference_factory(),
        &tok.encode(PROMPT),
        &p,
        &mut |_| Ok(()),
        &mut |_| {},
        &mut Timings::default(),
    )
    .expect("a generation");
    let s = seen.lock().expect("seen");
    let first = s
        .calls
        .iter()
        .position(|c| *c == Call::Next)
        .expect("the prompt's step");
    Ran {
        out,
        calls: s.calls[first + 1..].to_vec(),
        fed: s.fed.clone(),
    }
}

/// The ids a generation fed are the prompt's and every id it generated but
/// the stop: the prompt cache holds the answer.
fn assert_fed_the_answer(tok: &Arc<dyn Tokenizer>, r: &Ran) {
    let mut want = tok.encode(PROMPT);
    let (stop, body) = r.out.tokens.split_last().expect("a stop");
    assert!(tok.stops().contains(stop), "ends on {stop}");
    want.extend_from_slice(body);
    assert_eq!(r.fed, want);
}

/// A program of text parts compiled on `tok`.
fn compiled(tok: &Arc<dyn Tokenizer>, parts: &[Part]) -> Constraint {
    let vocab = Vocab::read(&**tok);
    let prog = program::compile(
        parts,
        &Env {
            tok: &**tok,
            vocab: &vocab,
        },
    )
    .expect("a program");
    Constraint::new(prog, tok.encode("</think>\n\n"), Arc::clone(vocab.stops()))
}

/// The jump arm takes the ids the step arm takes, over programs that mix
/// forced text, choices (one-token and many-token options), free strings whose
/// end token absorbs the front of the text after it, and a tool call, on rows
/// that differ by seed; both feed the prompt cache the same ids.
#[test]
fn the_jump_arm_takes_the_step_arms_ids() {
    let merged: Arc<dyn Tokenizer> =
        Arc::new(Pieces(Arc::new(PieceTok(&["\", \"", "\"}", "\","]))));
    let schema = json!({"type": "object", "properties": {
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "tag": {"type": "string", "maxLength": 6},
        "day": {"type": "integer", "enum": [1, 2, 3]},
        "ok": {"type": "boolean"},
        "kind": {"const": "weather"},
        "note": {"type": "string", "maxLength": 5}}});
    let call = json!({"tools": forecast_tools(), "tool_choice": "required"});
    let mut spanned = 0;
    for (tok, format, body) in [
        (bytes_tok(), ToolFormat::Hermes, schema_body(&schema)),
        (
            Arc::clone(&merged),
            ToolFormat::Hermes,
            schema_body(&schema),
        ),
        (bytes_tok(), ToolFormat::QwenXml, call.clone()),
        (stop_tok(), ToolFormat::GlmXml, call.clone()),
    ] {
        let c = build(&tok, format, &body).expect("served");
        for seed in 0..8 {
            let step = scripted(&tok, &c, seed, Vec::new(), false, 400);
            let jump = scripted(&tok, &c, seed, Vec::new(), true, 400);
            assert_eq!(jump.out.tokens, step.out.tokens, "{format:?} seed {seed}");
            assert_eq!(jump.out.content, step.out.content, "{format:?} seed {seed}");
            assert_eq!(jump.out.stop, step.out.stop, "{format:?} seed {seed}");
            assert_fed_the_answer(&tok, &step);
            assert_fed_the_answer(&tok, &jump);
            let (js, ss) = (
                jump.out.timings.jump.unwrap(),
                step.out.timings.jump.unwrap(),
            );
            assert_eq!((js.decisions, js.free_steps), (ss.decisions, ss.free_steps));
            assert!(
                js.next_calls + js.prefill_calls < ss.next_calls,
                "{format:?} seed {seed}"
            );
            spanned += js.spans;
        }
    }
    assert!(spanned > 0);
}

/// A first step's lead that a later step reverses: with p(a) = .6, p(b|a) =
/// p(d|a) = .5 and p(c) = .4, and a forced end after every option, greedy
/// takes a then b (the lowest id of a tie), a path of .3 against c's .4. The
/// pick is the masked argmax at each node, in both arms, not the option with
/// the largest probability.
#[test]
fn a_lead_that_reverses_later_still_picks_greedy() {
    let tok = bytes_tok();
    let at = |ch: char| BYTES + u32::from(u8::try_from(ch).expect("ascii"));
    let (a, b, c, d) = (at('a'), at('b'), at('c'), at('d'));
    let alt = |text: &str| program::Alt {
        text: text.to_owned(),
        then: Vec::new(),
    };
    let con = compiled(
        &tok,
        &[
            Part::Pick(vec![alt("ab"), alt("ad"), alt("c")]),
            Part::Text("xyz!".to_owned()),
        ],
    );
    let low = -20.0;
    let peaks = vec![
        (
            vec![],
            vec![(a, 0.6f32.ln()), (c, 0.4f32.ln()), (b, low), (d, low)],
        ),
        (vec![a], vec![(b, 0.5f32.ln()), (d, 0.5f32.ln())]),
    ];
    let want = with_eos(&tok, [vec![a, b], ids(&tok, "xyz!")].concat());
    for jump in [false, true] {
        let r = scripted(&tok, &con, 3, peaks.clone(), jump, 400);
        assert_eq!(r.out.tokens, want, "jump {jump}");
        assert_eq!(r.out.content, "abxyz!");
        assert_fed_the_answer(&tok, &r);
        let j = r.out.timings.jump.expect("a constrained request");
        assert_eq!(j.decisions, 2);
        assert_eq!(j.spans, usize::from(jump));
    }
}

/// A forced call with eight functions of two enum arguments each, whose
/// options differ at their first byte (the shape of an agent's tool call),
/// on a vocabulary of one id a byte. The jump arm is four spans, each a
/// prefill of its run but the last id and a step of that id, whose row is the
/// next decision's:
///
/// - span 0: the call's opening text, `<tool_call>\n<function=`, taken from
///   the walk's first id; its step reads the function's decision;
/// - span 1: the function's name but its first byte, and the markup up to the
///   first argument's value; its step reads that value's decision;
/// - span 2: the value but its first byte, and the markup up to the second
///   value; its step reads that decision;
/// - span 3: the second value but its first byte, and the closing text; its
///   step reads the stop.
///
/// With D the indices of the three decisions in the call's text of E ids, the
/// spans feed D0 - 1, D1 - D0 - 1, D2 - D1 - 1 and E - D2 - 1 ids: E - 4, one
/// step each, against E steps: 4 prefills and 4 steps, a prefill and a step
/// for each of the 3 decisions plus the closing text. Counting the opening
/// text inside the prompt's call and the closing text as a free tail gives
/// 3 steps and 2 prefills (one step a decision, one prefill a run between two
/// decisions); neither holds. The prompt's call ends at the template's
/// generation prompt, so the opening text is the program's first generated
/// ids and costs span 0; the stop is picked from a row, so the closing text
/// is fed and its step is the row the stop is read from (the cache holds the
/// answer without the stop).
#[test]
fn a_forced_call_is_fed_in_the_planned_calls() {
    let tok = bytes_tok();
    let fun = |name: &str| {
        json!({"type": "function", "function": {"name": name, "parameters": {
            "type": "object", "properties": {
                "color": {"type": "string", "enum": ["red", "green", "blue"]},
                "mode": {"type": "string", "enum": ["yes", "no"]}}}}})
    };
    let tools: Vec<Value> = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    ]
    .iter()
    .map(|n| fun(n))
    .collect();
    let c = build(
        &tok,
        ToolFormat::QwenXml,
        &json!({"tools": tools, "tool_choice": "required"}),
    )
    .expect("served");
    let text = "<tool_call>\n<function=bravo>\n<parameter=color>\ngreen\n</parameter>\n\
                <parameter=mode>\nno\n</parameter>\n</function>\n</tool_call>";
    let want = ids(&tok, text);
    let e = want.len();
    // The decisions: the function's first byte, then each value's.
    let d = [
        text.find("bravo").unwrap(),
        text.find("green").unwrap(),
        text.find("no\n").unwrap(),
    ];
    // Rows that favour the call on its decisions (every other id is forced
    // or masked).
    let peaks: Vec<_> = d
        .iter()
        .map(|&at| (want[..at].to_vec(), vec![(want[at], 9.0)]))
        .collect();
    let step = scripted(&tok, &c, 1, peaks.clone(), false, 400);
    let jump = scripted(&tok, &c, 1, peaks, true, 400);
    for r in [&step, &jump] {
        assert_eq!(r.out.tokens, with_eos(&tok, want.clone()));
        assert_eq!(r.out.content, text);
        assert_fed_the_answer(&tok, r);
    }
    assert_eq!(step.calls, vec![Call::Next; e]);
    let mut planned = Vec::new();
    for len in [d[0] - 1, d[1] - d[0] - 1, d[2] - d[1] - 1, e - d[2] - 1] {
        planned.extend([Call::Prefill(len), Call::Next]);
    }
    assert_eq!(jump.calls, planned);
    let (s, j) = (
        step.out.timings.jump.unwrap(),
        jump.out.timings.jump.unwrap(),
    );
    assert_eq!(
        (s.arm, s.spans, s.span_ids, s.prefill_calls, s.next_calls),
        (false, 0, 0, 0, e)
    );
    assert_eq!(
        (j.arm, j.spans, j.span_ids, j.prefill_calls, j.next_calls),
        (true, 4, e - 4, 4, 4)
    );
    assert_eq!(
        (s.decisions, s.free_steps, j.decisions, j.free_steps),
        (3, 0, 3, 0)
    );
    let t = jump.out.timings.to_json();
    assert_eq!(t["constrained"]["arm"], "jump");
    assert_eq!(t["constrained"]["prefill_calls"], 4);
}

/// With the arm off a constrained request makes the step arm's calls and
/// only those: one step for every id it feeds, no prefill.
#[test]
fn the_jump_arm_off_makes_the_step_arms_calls() {
    let tok = bytes_tok();
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&weather_schema())).expect("served");
    let r = scripted(&tok, &c, 5, Vec::new(), false, 400);
    let fed = r.out.tokens.len() - 1;
    assert_eq!(r.calls, vec![Call::Next; fed]);
    let j = r.out.timings.jump.expect("a constrained request");
    assert_eq!(
        (j.arm, j.spans, j.span_ids, j.prefill_calls, j.next_calls),
        (false, 0, 0, 0, fed)
    );
    let on = scripted(&tok, &c, 5, Vec::new(), true, 400);
    assert_eq!(on.out.tokens, r.out.tokens);
    assert!(on.calls.iter().any(|c| matches!(c, Call::Prefill(_))));
    // Free text carries no counts.
    let plain = run(&tok, None, ThinkEntry::Closed, &ids(&tok, "ab"));
    assert!(plain.timings.jump.is_none());
    assert!(plain.timings.to_json().get("constrained").is_none());
}

/// A free string's end token takes the front of the text after it, and the
/// rest of that text is the next run's: the ids are those the text spells,
/// with the end token merged, none dropped and none repeated.
#[test]
fn a_free_strings_end_hands_the_rest_to_the_run() {
    const PIECE: &str = "\", \"";
    let pieces = Arc::new(PieceTok(&[PIECE, "\"}", "\","]));
    let merged = pieces.piece(PIECE);
    let tok: Arc<dyn Tokenizer> = Arc::new(Pieces(pieces));
    let schema = json!({"type": "object", "properties": {
        "a": {"type": "string"}, "bb": {"type": "boolean"}, "c": {"type": "string"}}});
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&schema)).expect("served");
    let x = BYTES + u32::from(b'x');
    // The first string says `x` and ends on the merged piece; the rest of
    // `", "bb": ` is forced; the boolean says true; the second string ends at
    // once on `"}`.
    let mut made = ids(&tok, "{\"a\": \"");
    let at_free = made.len();
    made.push(x);
    made.push(merged);
    made.extend(ids(&tok, "bb\": "));
    let at_bool = made.len();
    made.extend(ids(&tok, "true, \"c\": \""));
    let at_end = made.len();
    let close = tok.encode("\"}")[0];
    let peaks = vec![
        (made[..at_free].to_vec(), vec![(x, 9.0)]),
        (made[..=at_free].to_vec(), vec![(merged, 9.0)]),
        (made[..at_bool].to_vec(), vec![(made[at_bool], 9.0)]),
        (made[..at_end].to_vec(), vec![(close, 9.0)]),
    ];
    let mut want = made.clone();
    want.push(close);
    for jump in [false, true] {
        let r = scripted(&tok, &c, 2, peaks.clone(), jump, 400);
        assert_eq!(
            r.out.content, "{\"a\": \"x\", \"bb\": true, \"c\": \"\"}",
            "jump {jump}"
        );
        assert_eq!(r.out.tokens, with_eos(&tok, want.clone()), "jump {jump}");
        assert_fed_the_answer(&tok, &r);
    }
}

/// The ids a request may still generate and the positions the cache has left
/// bound a run: the jump arm stops where the step arm does, on the same ids.
#[test]
fn a_run_is_cut_at_the_budget_and_the_context() {
    let tok = bytes_tok();
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&weather_schema())).expect("served");
    let mut cut = 0;
    for n_predict in [1, 2, 3, 4, 7, 12, 33] {
        let step = scripted(&tok, &c, 4, Vec::new(), false, n_predict);
        let jump = scripted(&tok, &c, 4, Vec::new(), true, n_predict);
        assert_eq!(jump.out.tokens, step.out.tokens, "n_predict {n_predict}");
        assert_eq!(jump.out.stop, StopKind::Limit);
        assert_eq!(jump.fed, step.fed, "n_predict {n_predict}");
        cut += jump.out.timings.jump.unwrap().spans;
    }
    assert!(cut > 0);
    for ctx_max in [3, 4, 5, 9, 20, 41] {
        let step = scripted_in(&tok, &c, 4, Vec::new(), false, 400, ctx_max);
        let jump = scripted_in(&tok, &c, 4, Vec::new(), true, 400, ctx_max);
        assert_eq!(jump.out.tokens, step.out.tokens, "ctx {ctx_max}");
        assert_eq!(jump.out.truncated, step.out.truncated, "ctx {ctx_max}");
        assert_eq!(jump.fed, step.fed, "ctx {ctx_max}");
        assert!(jump.fed.len() <= ctx_max);
    }
}

/// A run of fixed ids fed in calls of a quantum asks the sink between the
/// calls, as a prompt's calls do: a client that went away ends the generation
/// at the next call with the rest of the run unfed, and one that stays gets
/// the whole run.
#[test]
fn a_run_fed_in_quanta_asks_the_sink_between_calls() {
    let tok = bytes_tok();
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&weather_schema())).expect("served");
    let run = |gone_at: usize| {
        let seen = Arc::new(Mutex::new(Seen::default()));
        let mut slot = Slot::new(Box::new(Scripted {
            tok: Arc::clone(&tok),
            seed: 4,
            peaks: Vec::new(),
            ctx_max: 4096,
            seen: Arc::clone(&seen),
        }))
        .schema_jump(true);
        let p = params(Some(c.clone()), ThinkEntry::Closed);
        let mut asked = 0;
        let (r, _) = generate_in_quanta(
            &mut slot,
            &sampling::reference_factory(),
            &tok.encode(PROMPT),
            &p,
            NonZeroUsize::new(2),
            &mut |ev| {
                if matches!(ev, Event::Text("", _)) {
                    asked += 1;
                    if asked == gone_at {
                        return Err(io::Error::new(io::ErrorKind::BrokenPipe, "gone"));
                    }
                }
                Ok(())
            },
            &mut |_| {},
        );
        let calls = seen.lock().expect("seen").calls.clone();
        let prompt_step = calls
            .iter()
            .position(|c| *c == Call::Next)
            .expect("the prompt");
        (r, asked, calls[prompt_step + 1..].to_vec())
    };
    let (whole, asked, calls) = run(usize::MAX);
    whole.expect("a client that stays gets the answer");
    assert!(
        asked > 2,
        "the first run is fed in several calls: asked {asked}"
    );
    assert!(calls.len() > 2);
    let (gone, asked, calls) = run(2);
    assert!(
        matches!(gone, Err(GenError::Client(_))),
        "a failed sink ends the generation"
    );
    assert_eq!(asked, 2);
    assert_eq!(
        calls,
        vec![Call::Prefill(2); 2],
        "no call after the failed one"
    );
}
