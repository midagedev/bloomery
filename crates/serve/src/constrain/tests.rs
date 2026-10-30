//! The step arm against scripted rows: a request's fields to a program, the
//! program's pick at every token, and each format's inverse read back by its
//! own scanner.

use std::io;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json};

use super::program::{self, Env};
use super::{Ask, Constraint, Part, Refusal, Vocab, Walk, request};
use crate::api::logprobs::Ask as Probs;
use crate::callstream::CallPiece;
use crate::dsml::{ChatParser, DsmlScan, MarkupError, Spelling, ToolFormat, Tools};
use crate::engine::{Decoder, Engine, EngineError, SamplingParams, Tokenizer};
use crate::genloop::{
    Event, GenError, GenParams, Outcome, Slot, StopKind, Timings, generate, generate_in_quanta,
};
use crate::glmxml::{ArgTypes, KvScan, KvSpelling};
use crate::hermes::HermesScan;
use crate::mistral::MistralScan;
use crate::mock::SPECIALS;
use crate::qwenxml::{Framing, ParamKinds, QwenScan};
use crate::reasoning::{ReasoningFormat, ThinkEntry, ThinkTags};
use crate::sampling;

/// The key/value markup in each spelling.
const GLM: ToolFormat = ToolFormat::KvMarkup(KvSpelling::Glm);
const SOLAR: ToolFormat = ToolFormat::KvMarkup(KvSpelling::Solar);
/// Qwen3-Coder's markup in each framing of a value.
const QWEN: ToolFormat = ToolFormat::QwenXml(Framing::Lines);
const QWEN_COMPACT: ToolFormat = ToolFormat::QwenXml(Framing::Compact);

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
    fn guard<'a>(&self, _text: &'a str, _mark: &str) -> std::borrow::Cow<'a, str> {
        unreachable!("the constrain tests render no chat prompt")
    }
    fn encode_marked(&self, _text: &str, _mark: &str) -> Vec<u32> {
        unreachable!("the constrain tests render no chat prompt")
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
    fn bos(&self) -> Option<u32> {
        Some(0)
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
        reasoning_loop: false,
        think_entry: think,
        think: ThinkTags::THINK,
        think_split: None,
        logprobs: None,
        constraint,
        logit_bias: Vec::new(),
    }
}

/// One generation of `p` over rows that favour `want`, with whether each
/// step's row was asked for.
fn run_read(tok: &Arc<dyn Tokenizer>, p: &GenParams, want: &[u32]) -> (Outcome, Vec<bool>) {
    let read = Arc::new(Mutex::new(Vec::new()));
    // The rows follow the engine's step count, which is the step arm's.
    let mut slot = Slot::new(Box::new(Rows {
        tok: Arc::clone(tok),
        want: want.to_vec(),
        k: 0,
        read: Arc::clone(&read),
    }))
    .schema_jump(false);
    let o = generate(
        &mut slot,
        &sampling::reference_factory(),
        &tok.encode("hi"),
        p,
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
    run_read(tok, &params(constraint, think), want).0
}

fn build(
    tok: &Arc<dyn Tokenizer>,
    format: ToolFormat,
    body: &Value,
) -> Result<Constraint, Refusal> {
    let vocab = Arc::new(Vocab::read(&**tok));
    request(
        body.as_object().expect("a body"),
        &Ask {
            tok: &**tok,
            vocab: &vocab,
            format,
            think: ThinkTags::THINK,
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

/// A `logit_bias` is added to the row before the mask: it reorders the ids
/// the program allows and lets in no other. +20 on `f` turns both of its
/// choices (`fahrenheit`, `false`), +100 on `Z` fills the free string alone,
/// and the document still follows the schema.
#[test]
fn a_bias_reorders_the_allowed_ids_and_lets_in_none() {
    let tok = bytes_tok();
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&weather_schema())).expect("served");
    let mut slot = Slot::new(Box::new(Rows {
        tok: Arc::clone(&tok),
        want: ids(&tok, WEATHER),
        k: 0,
        read: Arc::new(Mutex::new(Vec::new())),
    }))
    .schema_jump(false);
    let mut p = params(Some(c), ThinkEntry::Closed);
    p.logit_bias = vec![
        (BYTES + u32::from(b'Z'), 100.0),
        (BYTES + u32::from(b'f'), 20.0),
    ];
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
    let v: Value =
        serde_json::from_str(&o.content).unwrap_or_else(|e| panic!("{e}: {}", o.content));
    assert_eq!(v["unit"], "fahrenheit", "{v}");
    assert_eq!(v["ok"], false, "{v}");
    assert!([1, 2, 3].contains(&v["day"].as_i64().expect("day")), "{v}");
    assert_eq!(v["kind"], "weather", "{v}");
    assert_eq!(v["note"], "ZZZZZZZZ", "{v}");
    assert_eq!(o.stop, StopKind::Eos);
}

/// The subset past objects and scalars: an integer, an array of a `$ref`'s
/// strings, an `anyOf` of a number and null, an array of integers.
fn wide_schema() -> Value {
    json!({
        "$defs": {"tag": {"type": "string", "maxLength": 4}},
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "xs": {"type": "array", "items": {"$ref": "#/$defs/tag"}},
            "o": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "e": {"type": "array", "items": {"type": "integer"}}
        }
    })
}

/// Whether `text` is a document of [`wide_schema`].
fn wide_valid(text: &str) {
    let v: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("{e}: {text}"));
    let keys: Vec<&str> = v
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["n", "xs", "o", "e"], "{text}");
    assert!(v["n"].is_i64(), "{text}");
    for x in v["xs"].as_array().expect("xs") {
        assert!(x.as_str().expect("a tag").chars().count() <= 4, "{text}");
    }
    assert!(v["o"].is_number() || v["o"].is_null(), "{text}");
    assert!(
        v["e"].as_array().expect("e").iter().all(Value::is_i64),
        "{text}"
    );
}

/// The wider subset (integers, numbers, arrays, `$ref`, `anyOf`, null) and
/// `json_object` are greedy where greedy obeys, the documents' ids taken as
/// they are, but for an empty reply object, which `json_object` does not
/// write; where greedy does not obey the mask holds; a free string in an array ends
/// on a token that takes the text after it (`", "` and `"]`); and a
/// `json_object` value's first characters are tokenised with the `: ` before
/// them (` "`), as the document's own ids have them.
#[test]
fn the_wider_subset_and_json_object_are_greedy_where_greedy_obeys() {
    let tok = bytes_tok();
    // PIN(2026-10-29): arrays, anyOf, refs and free numbers replace subset refusals.
    let wide = build(&tok, ToolFormat::Hermes, &schema_body(&wide_schema())).expect("served");
    let object = json!({"response_format": {"type": "json_object"}});
    let any = build(&tok, ToolFormat::Hermes, &object).expect("served");
    for (c, doc) in [
        (
            &wide,
            r#"{"n": -12, "xs": ["ab", "c"], "o": 2.5e-3, "e": [0, 7]}"#,
        ),
        (&wide, r#"{"n": 0, "xs": [], "o": null, "e": []}"#),
        (
            &any,
            r#"{"a": [1, {"b": null}, []], "c": "x", "d": {}, "e": -0.5E+2, "f": [true, false]}"#,
        ),
    ] {
        let want = ids(&tok, doc);
        let o = run(&tok, Some(c.clone()), ThinkEntry::Closed, &want);
        assert_eq!(o.tokens, with_eos(&tok, want), "{doc}");
        assert_eq!(o.content, doc);
    }
    let empty = run(
        &tok,
        Some(any.clone()),
        ThinkEntry::Closed,
        &ids(&tok, "{}"),
    );
    assert!(empty.content.starts_with("{\""), "{}", empty.content);
    let wrong = vec![BYTES + u32::from(b'Z'); 400];
    wide_valid(&run(&tok, Some(wide), ThinkEntry::Closed, &wrong).content);
    let quotes = vec![BYTES + u32::from(b'"'); 400];
    let o = run(&tok, Some(any), ThinkEntry::Closed, &quotes);
    assert!(
        serde_json::from_str::<Value>(&o.content).is_ok_and(|v| v.is_object()),
        "{}",
        o.content
    );
    let pieces = Arc::new(PieceTok(&["[\"", "\", \"", "\"]", " \""]));
    let want = vec![
        pieces.piece("[\""),
        BYTES + u32::from(b'a'),
        pieces.piece("\", \""),
        BYTES + u32::from(b'b'),
        pieces.piece("\"]"),
    ];
    let tok: Arc<dyn Tokenizer> = Arc::new(Pieces(pieces));
    let list = json!({"type": "array", "items": {"type": "string"}});
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&list)).expect("served");
    let o = run(&tok, Some(c), ThinkEntry::Closed, &want);
    assert_eq!(o.tokens, with_eos(&tok, want));
    assert_eq!(o.content, r#"["a", "b"]"#);
    let c = build(&tok, ToolFormat::Hermes, &object).expect("served");
    let doc = r#"{"a": "b", "c": [1, "d"]}"#;
    let want = ids(&tok, doc);
    let o = run(&tok, Some(c), ThinkEntry::Closed, &want);
    assert_eq!(o.tokens, with_eos(&tok, want));
    assert_eq!(o.content, doc);
}

/// Rows that favour these bytes in turn, of which a number's end is one now
/// and then.
fn cycling(bytes: &[u8], n: usize) -> Vec<u32> {
    bytes
        .iter()
        .cycle()
        .take(n)
        .map(|&b| BYTES + u32::from(b))
        .collect()
}

/// A request that samples, or penalizes at temperature 0, draws among the ids
/// the program allows: rows that favour letters the schema mostly refuses
/// still give documents of the schema, and the draws differ by seed.
#[test]
fn a_sampled_request_draws_inside_the_mask() {
    let tok = bytes_tok();
    let mut schema = wide_schema();
    schema["properties"]["o"] = json!({"anyOf": [{"type": "integer"}, {"type": "null"}]});
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&schema)).expect("served");
    let wrong = cycling(b"Z7,]}n", 400);
    let mut texts = Vec::new();
    for (seed, temperature, repeat_penalty) in [(1, 1.5, 1.0), (2, 1.5, 1.0), (3, 0.0, 1.3)] {
        let mut p = params(Some(c.clone()), ThinkEntry::Closed);
        p.sampling = SamplingParams {
            temperature,
            repeat_penalty,
            seed,
            ..SamplingParams::default()
        };
        let o = run_read(&tok, &p, &wrong).0;
        wide_valid(&o.content);
        assert_eq!(o.stop, StopKind::Eos);
        texts.push(o.content);
    }
    assert_ne!(texts[0], texts[1]);
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
    // Each definition is an array of the one before it: shallow as written,
    // 129 arrays deep once its $refs are written out.
    let mut defs = Map::new();
    defs.insert("d0".to_owned(), json!({"type": "null"}));
    for i in 1..130 {
        let items = json!({"$ref": format!("#/$defs/d{}", i - 1)});
        defs.insert(format!("d{i}"), json!({"type": "array", "items": items}));
    }
    let deep = json!({"type": "object", "$defs": defs,
        "properties": {"x": {"$ref": "#/$defs/d129"}}});
    let cases: Vec<(Value, ToolFormat, &str)> = vec![
        (
            json!({"tool_choice": "required", "tools": [{"function": {"name": "f",
                "parameters": deep}}]}),
            ToolFormat::Dsml(Spelling::Joined),
            "tools[0].f.parameters: the schema is too deeply nested once its $refs are written out",
        ),
        (
            schema_body(&deep),
            ToolFormat::Hermes,
            "schema: the schema is too deeply nested once its $refs are written out",
        ),
        (
            node(json!({"type": "string", "pattern": "a+"})),
            ToolFormat::Hermes,
            "schema.properties.p: the keyword \"pattern\"",
        ),
        (
            node(json!({"$ref": "#/x"})),
            ToolFormat::Hermes,
            "$ref \"#/x\" does not point into this schema",
        ),
        (
            schema_body(&json!({"type": "object", "properties": {"p": {"$ref": "#"}}})),
            ToolFormat::Hermes,
            "$ref \"#\" is met again inside the schema it points to",
        ),
        (
            node(json!({"$ref": "#/$defs/d", "type": "string"})),
            ToolFormat::Hermes,
            "the keyword \"type\" beside $ref",
        ),
        (
            node(json!({"oneOf": [{"type": "string"}]})),
            ToolFormat::Hermes,
            "the keyword \"oneOf\"",
        ),
        (
            node(json!({"anyOf": []})),
            ToolFormat::Hermes,
            "anyOf must be a non-empty array",
        ),
        (
            node(json!({"type": "integer", "minimum": 0})),
            ToolFormat::Hermes,
            "the keyword \"minimum\"",
        ),
        (
            node(json!({"type": "array"})),
            ToolFormat::Hermes,
            "type array needs items",
        ),
        (
            node(json!({"type": "array", "items": {"type": "string"}, "minItems": 1})),
            ToolFormat::Hermes,
            "the keyword \"minItems\"",
        ),
        (
            json!({"tool_choice": "required", "tools": [{"function": {"name": "f", "parameters":
            {"type": "object", "properties": {"q": {"anyOf": [{"type": "integer"},
            {"type": "string"}]}}}}}]}),
            QWEN,
            "an anyOf with a text option",
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
            json!({"tool_choice": "required", "tools": tools,
                "response_format": {"type": "json_object"}}),
            ToolFormat::Hermes,
            "has one program",
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
            GLM,
            "under \"type\": \"string\"",
        ),
        (
            json!({"tool_choice": "required", "tools": [{"function": {"name": "f", "parameters":
            {"type": "object", "properties": {"q": {"enum": ["a"]}}}}}]}),
            SOLAR,
            "Solar's tool markup reads a text value as text only under \"type\": \"string\"",
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

/// A schema small as sent but large once written out or compiled is refused
/// by name before the copies or the tokenising are done: `$ref`s to a 65 KiB
/// constant, in one schema or across a request's tools, a long name above
/// many schemas, and an `anyOf` that repeats a 10,000-byte property name in
/// each of its options.
#[test]
fn a_schema_is_refused_before_it_grows_past_its_budget() {
    let tok = bytes_tok();
    let refused = |schema: Value| {
        build(&tok, ToolFormat::Hermes, &schema_body(&schema))
            .expect_err("over budget")
            .message
    };
    let over = "the schema passes 67108864 bytes once its $refs are written out";
    let large = |refs: usize| {
        let props: Map<String, Value> = (0..refs)
            .map(|i| (format!("p{i}"), json!({"$ref": "#/$defs/large"})))
            .collect();
        json!({"type": "object", "properties": props,
            "$defs": {"large": {"const": "a".repeat(65 * 1024)}}})
    };
    let e = refused(large(1024));
    assert!(e.contains(over), "{e}");
    // The tools of a request share one budget.
    let tools: Vec<Value> = (0..2)
        .map(|i| json!({"function": {"name": format!("f{i}"), "parameters": large(520)}}))
        .collect();
    let e = build(
        &tok,
        ToolFormat::Hermes,
        &json!({"tool_choice": "required", "tools": tools}),
    )
    .expect_err("over budget")
    .message;
    assert!(e.starts_with("tools[1]") && e.contains(over), "{e}");
    // Each schema's path is copied too: a long name above many schemas.
    let props: Map<String, Value> = (0..2000)
        .map(|i| (format!("p{i}"), json!({"type": "null"})))
        .collect();
    let e = refused(json!({"type": "object", "properties": {
        ("n".repeat(40 * 1024)): {"type": "object", "properties": props}}}));
    assert!(e.contains(over), "{e}");
    let named = |options: usize| {
        json!({"type": "object", "properties": {
            ("k".repeat(10_000)): {"anyOf": vec![json!({"type": "null"}); options]}}})
    };
    let e = refused(named(1000));
    assert!(
        e.contains("its program tokenises more than 16777216 bytes"),
        "{e}"
    );
    build(&tok, ToolFormat::Hermes, &schema_body(&named(50))).expect("50 options compile");
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
        QWEN => [
            "<tool_call>\n<function=get_weather>\n<parameter=unit>\ncelsius\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=search>\n<parameter=query>\ncats\n</parameter>\n<parameter=deep>\nfalse\n</parameter>\n</function>\n</tool_call>",
        ],
        QWEN_COMPACT => [
            "<tool_call>\n<function=get_weather>\n<parameter=unit>celsius</parameter>\n<parameter=days>2</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=search>\n<parameter=query>cats</parameter>\n<parameter=deep>false</parameter>\n</function>\n</tool_call>",
        ],
        GLM => [
            "<tool_call>get_weather<arg_key>unit</arg_key><arg_value>celsius</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>",
            "<tool_call>search<arg_key>query</arg_key><arg_value>cats</arg_value><arg_key>deep</arg_key><arg_value>false</arg_value></tool_call>",
        ],
        SOLAR => [
            "<|tool_call:start|>get_weather\n<|tool_arg:start|>unit<|tool_arg:value|>celsius<|tool_arg:end|>\n<|tool_arg:start|>days<|tool_arg:value|>2<|tool_arg:end|>\n<|tool_call:end|>",
            "<|tool_call:start|>search\n<|tool_arg:start|>query<|tool_arg:value|>cats<|tool_arg:end|>\n<|tool_arg:start|>deep<|tool_arg:value|>false<|tool_arg:end|>\n<|tool_call:end|>",
        ],
        ToolFormat::Dsml(_) => [
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"unit\" string=\"true\">celsius</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"search\">\n<｜DSML｜parameter name=\"query\" string=\"true\">cats</｜DSML｜parameter>\n<｜DSML｜parameter name=\"deep\" string=\"false\">false</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
        ],
        ToolFormat::MuseSpark => [
            " to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"unit\">celsius</atem:parameter>\n<atem:parameter name=\"days\">2</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
            " to=search<|message|><atem:function_calls>\n<atem:invoke name=\"search\">\n<atem:parameter name=\"query\">cats</atem:parameter>\n<atem:parameter name=\"deep\">false</atem:parameter>\n</atem:invoke>\n</atem:function_calls>",
        ],
        ToolFormat::Mistral => [
            "[TOOL_CALLS]get_weather[ARGS]{\"unit\": \"celsius\", \"days\": 2}",
            "[TOOL_CALLS]search[ARGS]{\"query\": \"cats\", \"deep\": false}",
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
        ToolFormat::QwenXml(framing) => {
            let mut s = QwenScan::new(ParamKinds::of_tools(Some(tools)), framing);
            let mut out = s.push(text).expect("parses");
            out.calls.extend(s.finish().expect("closes").calls);
            out
        }
        ToolFormat::KvMarkup(spelling) => {
            let mut s = KvScan::new(spelling, ArgTypes::of_tools(Some(tools)));
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
        ToolFormat::MuseSpark => {
            let mut s = crate::museatem::MuseScan::default();
            let mut out = s.push(text);
            out.calls.extend(s.finish().calls);
            out
        }
        ToolFormat::Mistral => {
            let mut s = MistralScan::default();
            let mut out = s.push(text).expect("parses");
            out.calls.extend(s.finish().expect("closes").calls);
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
        QWEN,
        QWEN_COMPACT,
        GLM,
        SOLAR,
        ToolFormat::Dsml(Spelling::Joined),
        ToolFormat::Dsml(Spelling::Spaced),
        ToolFormat::MuseSpark,
        ToolFormat::Mistral,
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
        QWEN,
        QWEN_COMPACT,
        GLM,
        SOLAR,
        ToolFormat::Dsml(Spelling::Joined),
        ToolFormat::Dsml(Spelling::Spaced),
        ToolFormat::MuseSpark,
        ToolFormat::Mistral,
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

/// A forced call whose arguments are an integer, an array of integers and an
/// `anyOf` of an integer and null is read back by each format's scanner as
/// those values, on rows that favour letters the schema mostly refuses.
#[test]
fn a_forced_calls_numbers_arrays_and_any_of_read_back() {
    let tok = bytes_tok();
    let params = json!({"type": "object", "properties": {
        "n": {"type": "integer"},
        "xs": {"type": "array", "items": {"type": "integer"}},
        "o": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}});
    let tools = json!([{"type": "function", "function": {"name": "f", "parameters": params}}]);
    let wrong = cycling(b"Z7,]}\n<n", 300);
    for format in FORMATS.into_iter().chain([ToolFormat::MuseSpark]) {
        let c =
            build(&tok, format, &call_body(&params)).unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let o = run(&tok, Some(c), ThinkEntry::Closed, &wrong);
        let calls = scanned(format, &tools, &o.content);
        let [(name, args)] = calls.as_slice() else {
            panic!("{format:?}: {}", o.content);
        };
        assert_eq!(name, "f");
        assert!(args["n"].is_i64(), "{format:?}: {args}");
        assert!(
            args["xs"]
                .as_array()
                .is_some_and(|a| a.iter().all(Value::is_i64)),
            "{format:?}: {args}"
        );
        assert!(
            args["o"].is_i64() || args["o"].is_null(),
            "{format:?}: {args}"
        );
        assert_eq!(o.stop, StopKind::Eos);
    }
}

/// Numeric arguments retain their values in complete calls and streamed
/// argument pieces, including numbers nested inside arrays and objects.
#[test]
fn numeric_tool_arguments_keep_their_values() {
    let mut wrong = Vec::new();
    for (value, want, schema) in [
        (
            "18446744073709551617",
            "18446744073709551617",
            json!({"type": "integer"}),
        ),
        ("1e999", "1e+999", json!({"type": "number"})),
        (
            "-18446744073709551617",
            "-18446744073709551617",
            json!({"type": "integer"}),
        ),
        ("-1e-999", "-1e-999", json!({"type": "number"})),
        (
            r#"[18446744073709551617,{"x":1e999}]"#,
            r#"[18446744073709551617,{"x":1e+999}]"#,
            json!({"type": "array"}),
        ),
    ] {
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"n": schema}}}}]);
        let verbatim = format!(r#"{{"n":{value}}}"#);
        let want = format!(r#"{{"n":{want}}}"#);
        for format in FORMATS.into_iter().chain([ToolFormat::MuseSpark]) {
            let raw = match format {
                ToolFormat::Hermes => format!(
                    "<tool_call>\n{{\"name\":\"f\",\"arguments\":{{\"n\":{value}}}}}\n</tool_call>"
                ),
                QWEN => format!(
                    "<tool_call>\n<function=f>\n<parameter=n>\n{value}\n</parameter>\n</function>\n</tool_call>"
                ),
                QWEN_COMPACT => format!(
                    "<tool_call>\n<function=f>\n<parameter=n>{value}</parameter>\n</function>\n</tool_call>"
                ),
                GLM => format!(
                    "<tool_call>f<arg_key>n</arg_key><arg_value>{value}</arg_value></tool_call>"
                ),
                SOLAR => format!(
                    "<|tool_call:start|>f\n<|tool_arg:start|>n<|tool_arg:value|>{value}<|tool_arg:end|>\n<|tool_call:end|>"
                ),
                ToolFormat::Dsml(spelling) => {
                    let raw = format!(
                        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"n\" string=\"false\">{value}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
                    );
                    if spelling == Spelling::Spaced {
                        raw.replace("｜DSML｜tool_calls", "｜DSML｜ calls")
                            .replace("｜DSML｜invoke", "｜DSML｜ invoke")
                            .replace("｜DSML｜parameter", "｜DSML｜ parameter")
                    } else {
                        raw
                    }
                }
                ToolFormat::MuseSpark => format!(
                    " to=f<|message|><atem:function_calls>\n<atem:invoke name=\"f\">\n<atem:parameter name=\"n\">{value}</atem:parameter>\n</atem:invoke>\n</atem:function_calls>"
                ),
                _ => unreachable!("a parsed format"),
            };
            let mut whole = parser(format, &tools);
            let result = whole.try_push(&raw).and_then(|_| whole.try_finish());
            match result {
                Ok(_)
                    if whole.message().calls.len() == 1
                        && whole.message().calls[0].name == "f"
                        && (whole.message().calls[0].arguments == want
                            || whole.message().calls[0].arguments == verbatim) => {}
                result => wrong.push(format!(
                    "{format:?} {value} whole: {result:?} {:?}",
                    whole.message()
                )),
            }
            let mut stream = parser(format, &tools).streaming_calls();
            let mut args = String::new();
            let result = raw
                .chars()
                .try_for_each(|ch| {
                    stream.try_push(&ch.to_string())?;
                    for piece in stream.take_pieces() {
                        if let CallPiece::Args { text, .. } = piece {
                            args.push_str(&text);
                        }
                    }
                    Ok::<_, MarkupError>(())
                })
                .and_then(|()| stream.try_finish().map(|_| ()));
            for piece in stream.take_pieces() {
                if let CallPiece::Args { text, .. } = piece {
                    args.push_str(&text);
                }
            }
            // Streaming may keep the exponent's original spelling.
            if result.is_err() || (args != want && args != verbatim) {
                wrong.push(format!("{format:?} {value} stream: {result:?} {args:?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A parameter whose `$ref` names a string reads back as that string through
/// the GLM and Solar scanners, whole and streamed byte by byte: its digits
/// stay text.
#[test]
fn a_string_ref_reads_back_as_text() {
    let tok = bytes_tok();
    let params = json!({"type": "object",
        "$defs": {"s": {"type": "string", "const": "123"}},
        "properties": {"s": {"$ref": "#/$defs/s"}}});
    let tools = json!([{"type": "function", "function": {"name": "f", "parameters": params}}]);
    let want = json!({"s": "123"});
    for (format, spelling) in [(GLM, KvSpelling::Glm), (SOLAR, KvSpelling::Solar)] {
        let c =
            build(&tok, format, &call_body(&params)).unwrap_or_else(|e| panic!("{format:?}: {e}"));
        let o = run(&tok, Some(c), ThinkEntry::Closed, &[]);
        assert_eq!(
            scanned(format, &tools, &o.content),
            vec![("f".to_owned(), want.clone())],
            "{format:?}: {}",
            o.content
        );
        let mut p = ChatParser::with_tools(
            "",
            ReasoningFormat::None,
            Some(Tools::KvMarkup(spelling, ArgTypes::of_tools(Some(&tools)))),
        )
        .streaming_calls();
        let mut args = String::new();
        let mut take = |p: &mut ChatParser| {
            for piece in p.take_pieces() {
                if let CallPiece::Args { text, .. } = piece {
                    args.push_str(&text);
                }
            }
        };
        for ch in o.content.chars() {
            p.try_push(&ch.to_string()).expect("parses");
            take(&mut p);
        }
        p.try_finish().expect("closes");
        take(&mut p);
        assert_eq!(
            serde_json::from_str::<Value>(&args).expect("JSON arguments"),
            want,
            "{format:?}"
        );
    }
}

/// A forced call of the function `f` whose parameters are `params`.
fn call_body(params: &Value) -> Value {
    json!({"tool_choice": "required", "tools": [
        {"type": "function", "function": {"name": "f", "parameters": params}}]})
}

const FORMATS: [ToolFormat; 7] = [
    ToolFormat::Hermes,
    QWEN,
    QWEN_COMPACT,
    GLM,
    SOLAR,
    ToolFormat::Dsml(Spelling::Joined),
    ToolFormat::Dsml(Spelling::Spaced),
];

/// Every closing tag a chat format's scanner looks for, in one string.
const TAGS: &str = "x</tool_call></arg_value><|tool_arg:end|><|tool_call:end|></｜DSML｜parameter></｜DSML｜ parameter>\n</parameter>\n</function>y";

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
        let field = if what == "a tool argument" {
            "tools"
        } else {
            "response_format.json_schema.schema"
        };
        assert_eq!(e.field.as_deref(), Some(field), "{what}: {e}");
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
    let vocab = Arc::new(Vocab::read(&*tok));
    let env = Env::new(&*tok, &vocab);
    let prog = program::compile(&[Part::Text("{".to_owned())], &env).expect("compiles");
    let c = Constraint::new(prog, Vec::new(), vec![tok.eos()], vocab);
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

/// The rows of `tools/ref/clef/json-inner-cases.tsv`: a case's name, the
/// string, and the text between the quotes that spell it.
fn json_inner_cases() -> Vec<(String, String, String)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tools/ref/clef/json-inner-cases.tsv");
    let table =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let literal = |name: &str, cell: &str| -> String {
        serde_json::from_str(cell)
            .unwrap_or_else(|e| panic!("{name}: {cell} is not a JSON string: {e}"))
    };
    let cases: Vec<_> = table
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let cells: Vec<&str> = l.split('\t').collect();
            let [name, input, body] = cells[..] else {
                panic!("{l:?} is not three tab-separated cells");
            };
            (name.to_owned(), literal(name, input), literal(name, body))
        })
        .collect();
    assert!(!cases.is_empty(), "{} holds no case", path.display());
    cases
}

/// A string in a JSON literal is spelled between its quotes as the table has
/// it, and the table's spelling reads back as the string. The Clef
/// comparison's `json_inner` (`tools/ref/clef/sjcmp.py`) is held to the same
/// table by its self-test, so neither side can change an escape alone.
#[test]
fn a_json_literal_is_spelled_as_the_table_has_it() {
    let tok = bytes_tok();
    for (name, input, body) in json_inner_cases() {
        let schema = schema_body(&json!({"const": input}));
        let c = build(&tok, ToolFormat::Hermes, &schema).expect("served");
        let o = run(&tok, Some(c), ThinkEntry::Closed, &[]);
        assert_eq!(o.content, format!("\"{body}\""), "{name}");
        let back: String = serde_json::from_str(&o.content).expect("a JSON string");
        assert_eq!(back, input, "{name}");
    }
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
            QWEN,
            QWEN_COMPACT,
            GLM,
            SOLAR,
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
/// but Hermes, and as the argument itself in Hermes.
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
        (QWEN, "\n</parameter>\n"),
        (QWEN_COMPACT, "</parameter>"),
        (GLM, "</arg_value>"),
        (SOLAR, "<|tool_arg:end|>"),
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
            if format != QWEN || nested {
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

/// The request's chat parser, with its format and parameter types.
fn parser(format: ToolFormat, tools: &Value) -> ChatParser {
    let scan = match format {
        ToolFormat::Hermes => Tools::Hermes,
        ToolFormat::QwenXml(framing) => Tools::QwenXml(framing, ParamKinds::of_tools(Some(tools))),
        ToolFormat::KvMarkup(spelling) => {
            Tools::KvMarkup(spelling, ArgTypes::of_tools(Some(tools)))
        }
        ToolFormat::Dsml(spelling) => Tools::Dsml(spelling),
        ToolFormat::MuseSpark => Tools::MuseSpark,
        ToolFormat::Mistral => Tools::Mistral,
        ToolFormat::Unparsed => unreachable!("no markup"),
    };
    ChatParser::with_tools("", ReasoningFormat::None, Some(scan))
}

/// The content and call count read by the request's parser.
fn parsed(format: ToolFormat, tools: &Value, text: &str) -> Result<(String, usize), MarkupError> {
    let mut p = parser(format, tools);
    let (a, b) = (p.try_push(text)?, p.try_finish()?);
    Ok((
        format!("{}{}", a.content, b.content),
        a.calls.len() + b.calls.len(),
    ))
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
        (QWEN, "<tool_call></tool_call>"),
        (QWEN_COMPACT, "<tool_call></tool_call>"),
        (GLM, "<tool_call></tool_call>"),
        (SOLAR, "<|tool_call:start|><|tool_call:end|>"),
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
    let mut p = params(Some(c), ThinkEntry::Closed);
    let (o, read) = run_read(&tok, &p, &want);
    assert_eq!(o.content, "true");
    assert_eq!(read, [true, false, false, false, true]);
    p.logprobs = Some(Probs::new(1, tok.n_vocab()));
    let (o, read) = run_read(&tok, &p, &want);
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
    /// [`Engine::next_after`] of this many ids: one call.
    NextAfter(usize),
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
        self.fed_row(&[last], Call::Next, out)
    }
    /// One call whose row is the one `prefill` and `next` of the same ids
    /// leave: the row is a function of the ids fed.
    fn next_after(&mut self, ids: &[u32], out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.fed_row(ids, Call::NextAfter(ids.len()), out)
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

impl Scripted {
    /// `ids` fed as the one call `call`, and the row after them.
    fn fed_row(
        &mut self,
        ids: &[u32],
        call: Call,
        out: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let row = {
            let mut s = self.seen.lock().expect("seen");
            s.fed.extend_from_slice(ids);
            s.calls.push(call);
            self.row(&s.fed)
        };
        let best = sampling::argmax(&row);
        if let Some(out) = out {
            out.copy_from_slice(&row);
        }
        Ok(best)
    }
}

/// One generation from [`PROMPT`] on a [`Scripted`] engine, the jump arm on or
/// off, with the engine's calls after the prompt's step, the prompt's own
/// calls and the ids it was fed.
struct Ran {
    out: Outcome,
    calls: Vec<Call>,
    prompt: Vec<Call>,
    fed: Vec<u32>,
}

/// Where the prompt's calls end in `calls`: at its step, a `next` (the step
/// arm's) or a `next_after` (the jump arm's, which carries the call).
fn prompt_end(calls: &[Call]) -> usize {
    calls
        .iter()
        .position(|c| matches!(c, Call::Next | Call::NextAfter(_)))
        .expect("the prompt's step")
        + 1
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
    let mut p = params(Some(c.clone()), ThinkEntry::Closed);
    p.n_predict = n_predict;
    scripted_in(tok, &p, seed, peaks, jump, 4096)
}

/// [`scripted`] of the request `p` on an engine whose context holds
/// `ctx_max` positions.
fn scripted_in(
    tok: &Arc<dyn Tokenizer>,
    p: &GenParams,
    seed: u64,
    peaks: Peaks,
    jump: bool,
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
    let out = generate(
        &mut slot,
        &sampling::reference_factory(),
        &tok.encode(PROMPT),
        p,
        &mut |_| Ok(()),
        &mut |_| {},
        &mut Timings::default(),
    )
    .expect("a generation");
    let s = seen.lock().expect("seen");
    let end = prompt_end(&s.calls);
    Ran {
        out,
        calls: s.calls[end..].to_vec(),
        prompt: s.calls[..end].to_vec(),
        fed: s.fed.clone(),
    }
}

// PIN(2026-10-20): the jump tail is not fed and names EOS;
// compare the cached prefix and stop class instead of full equality.
/// The ids a generation fed are the prompt's and every id it generated up to
/// the last one an engine call took, which no call fed: the stop, or the
/// last id under the bounds, or, past the program's last pick, the pick
/// itself (the jump arm's tail, [`crate::genloop::Jump`]'s `tail`, takes the
/// ids after it with no call). The prompt cache holds the answer up to there.
fn assert_fed_the_answer(tok: &Arc<dyn Tokenizer>, r: &Ran) {
    let mut want = tok.encode(PROMPT);
    let tail = r.out.timings.jump.map_or(0, |j| j.tail);
    let body = &r.out.tokens[..r.out.tokens.len() - tail - 1];
    want.extend_from_slice(body);
    assert_eq!(r.fed, want, "tail {tail}");
}

/// The jump arm's ids against the step arm's: the same but for a stop the
/// step arm picks from a row, which the jump arm's tail names (the
/// vocabulary's end of sequence).
fn assert_step_arms_ids(tok: &Arc<dyn Tokenizer>, jump: &Ran, step: &Ran, what: &str) {
    assert_eq!(jump.out.stop, step.out.stop, "{what}");
    assert_eq!(jump.out.content, step.out.content, "{what}");
    let (j, s) = (&jump.out.tokens, &step.out.tokens);
    assert_eq!(j.len(), s.len(), "{what}");
    if jump.out.stop == StopKind::Eos {
        assert_eq!(j[..j.len() - 1], s[..s.len() - 1], "{what}");
        assert!(tok.stops().contains(&s[s.len() - 1]), "{what}");
        assert_eq!(j[j.len() - 1], tok.eos(), "{what}: the tail names the stop");
    } else {
        assert_eq!(j, s, "{what}");
    }
}

/// A program of text parts compiled on `tok`.
fn compiled(tok: &Arc<dyn Tokenizer>, parts: &[Part]) -> Constraint {
    let vocab = Arc::new(Vocab::read(&**tok));
    let prog = program::compile(parts, &Env::new(&**tok, &vocab)).expect("a program");
    Constraint::new(prog, Vec::new(), tok.encode("</think>\n\n"), vocab)
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
        (bytes_tok(), QWEN, call.clone()),
        (bytes_tok(), QWEN_COMPACT, call.clone()),
        (stop_tok(), GLM, call),
        (bytes_tok(), ToolFormat::Hermes, schema_body(&wide_schema())),
        (
            bytes_tok(),
            ToolFormat::Hermes,
            json!({"response_format": {"type": "json_object"}}),
        ),
    ] {
        let c = build(&tok, format, &body).expect("served");
        for seed in 0..8 {
            let step = scripted(&tok, &c, seed, Vec::new(), false, 400);
            let jump = scripted(&tok, &c, seed, Vec::new(), true, 400);
            assert_step_arms_ids(&tok, &jump, &step, &format!("{format:?} seed {seed}"));
            assert_fed_the_answer(&tok, &step);
            assert_fed_the_answer(&tok, &jump);
            let (js, ss) = (
                jump.out.timings.jump.unwrap(),
                step.out.timings.jump.unwrap(),
            );
            assert_eq!((js.decisions, js.free_steps), (ss.decisions, ss.free_steps));
            assert!(
                js.next_calls + js.prefill_calls + js.next_after_calls < ss.next_calls,
                "{format:?} seed {seed}"
            );
            spanned += js.spans;
        }
    }
    assert!(spanned > 0);
}

/// A whole number with nothing after it may end the answer: rows that favour
/// `7` and then the end give `7` and the end on both arms, for an integer, a
/// number and an `anyOf` of the two at the root.
#[test]
fn a_whole_root_number_may_end_the_answer() {
    let tok = bytes_tok();
    let seven = BYTES + u32::from(b'7');
    for schema in [
        json!({"type": "integer"}),
        json!({"type": "number"}),
        json!({"anyOf": [{"type": "integer"}, {"type": "number"}]}),
    ] {
        let c = build(&tok, ToolFormat::Hermes, &schema_body(&schema)).expect("served");
        let peaks: Peaks = vec![
            (Vec::new(), vec![(seven, 10.0)]),
            (vec![seven], vec![(tok.eos(), 10.0)]),
        ];
        let step = scripted(&tok, &c, 0, peaks.clone(), false, 40);
        let jump = scripted(&tok, &c, 0, peaks, true, 40);
        for r in [&step, &jump] {
            assert_eq!(
                (r.out.content.as_str(), r.out.stop),
                ("7", StopKind::Eos),
                "{schema}"
            );
        }
        assert_step_arms_ids(&tok, &jump, &step, &schema.to_string());
    }
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
        // The text after the second pick is the jump arm's tail: its four
        // ids and the stop taken with no call, no span fed.
        assert_eq!((j.spans, j.tail), (0, 5 * usize::from(jump)));
    }
}

/// A free string's decode steps copy no walk: the jump arm looks along the
/// program for its tail only where an id is fixed or the program is spent, so
/// the copies a request makes do not grow with the string's length.
#[test]
fn a_free_strings_steps_copy_no_walk() {
    let tok = bytes_tok();
    let mut ran = Vec::new();
    for max in [4, 24] {
        let schema = json!({"type": "object", "properties": {
            "note": {"type": "string", "maxLength": max}}});
        let c = build(&tok, ToolFormat::Hermes, &schema_body(&schema)).expect("served");
        let before = program::aheads();
        let r = scripted(&tok, &c, 1, Vec::new(), true, 400);
        let j = r.out.timings.jump.expect("a constrained request");
        ran.push((j.free_steps, program::aheads() - before));
    }
    let [(short, short_copies), (long, long_copies)] = ran[..] else {
        unreachable!("two lengths")
    };
    assert!(long >= short + 16, "free steps {short} and {long}");
    assert_eq!(long_copies, short_copies, "free steps {short} and {long}");
}

/// A forced call with eight functions of two enum arguments each, whose
/// options differ at their first byte (the shape of an agent's tool call),
/// on a vocabulary of one id a byte. With D the indices of the three
/// decisions in the call's text of E ids, the jump arm makes four calls a
/// request where the step arm makes E + 2 (a prefill of the prompt but its
/// last id, its step, and a step for every id of the call):
///
/// - the prompt's call carries the call's opening text,
///   `<tool_call>\n<function=` (D0 ids, the answer's lead): a prefill of the
///   prompt's 2 ids and D0 but the last, whose step's row reads the
///   function's decision;
/// - span 1: the function's first byte, just picked, then its name and the
///   markup up to the first argument's value, D1 - D0 ids, the last of them
///   stepped in the call: one `next_after` whose row reads that value;
/// - span 2: the value and the markup up to the second value, D2 - D1 ids,
///   one `next_after` whose row reads it;
/// - the tail: the E - D2 - 1 ids after the last decision and the stop,
///   E - D2 ids, taken with no call (the stop named, the end of sequence).
///
/// The cache then holds the answer up to the last decision, which no call
/// fed.
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
        QWEN,
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
    assert_eq!(step.prompt, [Call::Prefill(1), Call::Next]);
    assert_eq!(step.calls, vec![Call::Next; e]);
    assert_eq!(jump.prompt, [Call::Prefill(1 + d[0]), Call::Next]);
    assert_eq!(
        jump.calls,
        [Call::NextAfter(d[1] - d[0]), Call::NextAfter(d[2] - d[1])]
    );
    let (s, j) = (
        step.out.timings.jump.unwrap(),
        jump.out.timings.jump.unwrap(),
    );
    assert_eq!(
        (s.arm, s.spans, s.span_ids, s.prefill_calls, s.next_calls),
        (false, 0, 0, 0, e)
    );
    assert_eq!(
        (s.next_after_calls, s.lead, s.tail),
        (0, 0, 0),
        "the step arm"
    );
    assert_eq!(
        (j.arm, j.spans, j.span_ids, j.prefill_calls, j.next_calls),
        (true, 2, d[2] - d[0], 0, 0)
    );
    assert_eq!((j.next_after_calls, j.lead, j.tail), (2, d[0], e - d[2]));
    assert_eq!(
        (s.decisions, s.free_steps, j.decisions, j.free_steps),
        (3, 0, 3, 0)
    );
    let t = jump.out.timings.to_json();
    assert_eq!(t["constrained"]["arm"], "jump");
    assert_eq!(t["constrained"]["next_after_calls"], 2);
    assert_eq!(t["constrained"]["lead"], d[0]);
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
    assert_eq!((j.next_after_calls, j.lead, j.tail), (0, 0, 0));
    assert_eq!(r.prompt, [Call::Prefill(1), Call::Next]);
    let on = scripted(&tok, &c, 5, Vec::new(), true, 400);
    assert_eq!(on.out.tokens, r.out.tokens);
    assert!(matches!(on.prompt[..], [Call::Prefill(n), Call::Next] if n > 1));
    assert!(on.calls.iter().any(|c| matches!(c, Call::NextAfter(_))));
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
        assert_fed_the_answer(&tok, &step);
        assert_fed_the_answer(&tok, &jump);
        let j = jump.out.timings.jump.unwrap();
        cut += j.spans + j.lead;
    }
    assert!(cut > 0);
    let p = params(Some(c), ThinkEntry::Closed);
    for ctx_max in [3, 4, 5, 9, 20, 41] {
        let step = scripted_in(&tok, &p, 4, Vec::new(), false, ctx_max);
        let jump = scripted_in(&tok, &p, 4, Vec::new(), true, ctx_max);
        assert_eq!(jump.out.tokens, step.out.tokens, "ctx {ctx_max}");
        assert_eq!(jump.out.truncated, step.out.truncated, "ctx {ctx_max}");
        assert_fed_the_answer(&tok, &step);
        assert_fed_the_answer(&tok, &jump);
        assert!(jump.fed.len() <= ctx_max);
    }
}

/// A penalized request's draws count the fixed ids before them: the lead
/// `{"a": "` holds `a` and the later run `", "c": "` holds `c`, so rows that
/// favour `a` over `b` and then `c` over `d` give `b` and `d` under a repeat
/// penalty of 2, the jump arm on or off.
#[test]
fn a_penalty_counts_the_fixed_ids_before_a_decision() {
    let tok = bytes_tok();
    let pair = |x: &str, y: &str| json!({"type": "string", "enum": [x, y]});
    let schema =
        json!({"type": "object", "properties": {"a": pair("a", "b"), "c": pair("c", "d")}});
    let c = build(&tok, ToolFormat::Hermes, &schema_body(&schema)).expect("served");
    let byte = |b: u8| BYTES + u32::from(b);
    let favour = |hi: u8, lo: u8| vec![(byte(hi), 10.0), (byte(lo), 9.0)];
    // The second decision's rows follow either first pick, so each run's
    // count is seen apart from the other's.
    let peaks: Peaks = vec![
        (ids(&tok, "{\"a\": \""), favour(b'a', b'b')),
        (ids(&tok, "{\"a\": \"a\", \"c\": \""), favour(b'c', b'd')),
        (ids(&tok, "{\"a\": \"b\", \"c\": \""), favour(b'c', b'd')),
    ];
    let mut p = params(Some(c), ThinkEntry::Closed);
    p.sampling.repeat_penalty = 2.0;
    for jump in [false, true] {
        let r = scripted_in(&tok, &p, 0, peaks.clone(), jump, 4096);
        assert_eq!(r.out.content, r#"{"a": "b", "c": "d"}"#, "jump {jump}");
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
        let (r, timings) = generate_in_quanta(
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
        assert_eq!(
            timings.n_past,
            slot.held_of(0).len(),
            "the positions the request reports are the cache's: gone_at {gone_at}"
        );
        let calls = seen.lock().expect("seen").calls.clone();
        (r, asked, calls[prompt_end(&calls)..].to_vec())
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

/// The suite's median request (`ko-06`: an intent of four options and a
/// boolean, Q = 2) as the program compiles it on Qwen's vocabulary,
/// `F[4] P{4} F[6] P{2} F[1]` — `{"` `intent` `":` ` "`, the option, `",`
/// ` "` `ang` `ry` `":` ` `, the boolean, `}` — on a vocabulary of those
/// pieces, its rows raising `refund` and `true`. The two arms of one seed;
/// the answer is 13 ids and the stop.
///
/// The engine calls a request, the prompt's included (prompt = 2 ids):
///
/// | arm | calls | count |
/// |---|---|---|
/// | step | `prefill(1)`, 14 `next` | 14 next + 1 prefill |
/// | jump at b0705d908 | `prefill(1)`, `next`, `prefill(3)`, `next`, `prefill(6)`, `next`, `next`, `next` | 5 next + 3 prefill (2 spans) |
/// | + L1 lead | `prefill(5)`, `next`, `prefill(6)`, `next`, `next`, `next` | 4 next + 2 prefill |
/// | + L2 no tail | `prefill(5)`, `next`, `prefill(6)`, `next` | 2 next + 2 prefill |
/// | + L3 `next_after` | `prefill(5)`, `next`, `next_after(7)` | 1 next + 1 prefill (the prompt's) + 1 next_after |
fn ko06(jump: bool) -> Ran {
    const PIECES_KO06: &[&str] = &[
        "{\"", "intent", "\":", " \"", "cancel", "other", "refund", "tracking", "\",", "ang", "ry",
        "true", "false", "}",
    ];
    let pieces = Arc::new(PieceTok(PIECES_KO06));
    let tok: Arc<dyn Tokenizer> = Arc::new(Pieces(Arc::clone(&pieces)));
    let alt = |text: &str| program::Alt {
        text: text.to_owned(),
        then: Vec::new(),
    };
    let con = compiled(
        &tok,
        &[
            Part::Text("{\"intent\": \"".to_owned()),
            Part::Pick(["cancel", "other", "refund", "tracking"].map(alt).to_vec()),
            Part::Text("\", \"angry\": ".to_owned()),
            Part::Pick(["true", "false"].map(alt).to_vec()),
            Part::Text("}".to_owned()),
        ],
    );
    let lead = ids(&tok, "{\"intent\": \"");
    let mid = ids(&tok, "{\"intent\": \"refund\", \"angry\": ");
    assert_eq!((lead.len(), mid.len()), (4, 11), "the program's shape");
    let peaks = vec![
        (lead, vec![(pieces.piece("refund"), 9.0)]),
        (mid, vec![(pieces.piece("true"), 9.0)]),
    ];
    let r = scripted(&tok, &con, 6, peaks, jump, 400);
    assert_eq!(r.out.content, "{\"intent\": \"refund\", \"angry\": true}");
    assert_eq!(r.out.tokens.len(), 14);
    assert_fed_the_answer(&tok, &r);
    r
}

/// L1: the jump arm's prompt call carries the answer's lead, the four ids
/// before the first pick, and the step of the lead's last id is the row the
/// first pick reads: no call runs between the prompt's and the first pick,
/// and the prompt's calls feed the prompt's 2 ids and the lead.
/// FAIL-first: a `take_lead` that returns no id (b0705d908's prompt call)
/// feeds 2 ids and spends a step and a span on the lead.
#[test]
fn the_prompt_call_carries_the_lead() {
    let (step, jump) = (ko06(false), ko06(true));
    assert_step_arms_ids(&bytes_tok(), &jump, &step, "ko-06");
    let j = jump.out.timings.jump.expect("a constrained request");
    assert_eq!(j.lead, 4);
    let fed: usize = jump
        .prompt
        .iter()
        .map(|c| match *c {
            Call::Prefill(n) | Call::NextAfter(n) => n,
            Call::Next => 1,
        })
        .sum();
    assert_eq!(fed, 2 + 4, "{:?}", jump.prompt);
    // The first call after the prompt's feeds the first pick, `refund`.
    assert_eq!(jump.fed[2 + 4], step.out.tokens[4]);
    assert_eq!(step.out.timings.jump.expect("constrained").lead, 0);
}

/// L2: after the last pick nothing is fed or stepped: `}` and the stop are
/// the tail, taken with no call, the stop named (the end of sequence); the
/// cache holds the answer up to the last pick, 2 + 11 ids. FAIL-first: a
/// `tail_id` that never starts the tail (b0705d908) steps the pick and `}`,
/// two `next` more.
#[test]
fn nothing_is_fed_after_the_last_pick() {
    let jump = ko06(true);
    let j = jump.out.timings.jump.expect("a constrained request");
    assert_eq!(j.tail, 2);
    assert_eq!(jump.fed.len(), 2 + 11);
    assert_eq!(jump.out.tokens.last(), Some(&jump_eos()));
    assert_eq!(jump.out.stop, StopKind::Eos);
    assert_eq!(j.next_calls, 0, "{:?}", jump.calls);
}

/// The end of sequence of every vocabulary these tests build.
fn jump_eos() -> u32 {
    bytes_tok().eos()
}

/// L3: a span and the step that closes it are one `next_after`: the request
/// is the prompt's call, `prefill(5)` and `next` (the prompt and the lead,
/// whose step's row reads `refund`; a prompt call takes its checkpoints, so
/// it stays a prefill), and `next_after(7)` (`refund` and the six ids up to
/// the boolean, whose row reads `true`). The step arm is 15 calls.
/// FAIL-first: a span that never closes (b0705d908) makes `prefill(6)` and
/// `next` after the prompt's call.
#[test]
fn a_span_and_its_step_are_one_call() {
    let (step, jump) = (ko06(false), ko06(true));
    assert_eq!(step.prompt.len() + step.calls.len(), 15);
    assert_eq!(jump.prompt, [Call::Prefill(5), Call::Next]);
    assert_eq!(jump.calls, [Call::NextAfter(7)]);
    let j = jump.out.timings.jump.expect("a constrained request");
    assert_eq!(
        (
            j.spans,
            j.span_ids,
            j.next_calls,
            j.prefill_calls,
            j.next_after_calls
        ),
        (1, 7, 0, 0, 1)
    );
    assert_eq!(j.decisions, 2);
}
