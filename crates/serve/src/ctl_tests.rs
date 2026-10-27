//! Gate: client text in a chat prompt tokenizes as text. A special token's spelling in a user, system or
//! tool message, a tool result, a tool call's name or arguments, or a tool's description is a run of
//! ordinary byte-level BPE tokens in the prompt the engine is fed, never the token itself; the tokens
//! the template writes (turn markers, think tags, DSML, the BOS) stay tokens. An assistant message's
//! `content` and reasoning are the model's own past output and still carry its special ids.
//!
//! The vocabularies are synthetic: the real byte-level BPE (`tokenizer`) over the 256 byte tokens and
//! the special tokens each template family writes, as CONTROL and USER_DEFINED tokens the way the
//! released files type them (V4.1's `<｜User｜>` and `｜DSML｜` and the Qwen tool and think tags are
//! USER_DEFINED, which `parse_special = false` does not stop).

use std::borrow::Cow;
use std::sync::Arc;

use jinja::ChatTemplate;
use serde_json::{Map, Value, json};

use super::fixture_templates::FIXTURE_TEMPLATES;
use super::{State, apply_template, chat_input, render_chat, testserve};
use crate::engine::{Decoder, EngineError, Tokenizer};
use crate::media::SharedMediaModel;
use crate::{Engine, MockEngine};

const SOLAR_TEMPLATE: &str = include_str!("../tests/fixtures/solar-chat-template.jinja");
const V41_SPACED_TEMPLATE: &str =
    include_str!("../tests/fixtures/v41-dsml-spaced-chat-template.jinja");

/// `(text, is_control)`: a family's special tokens.
type Specials = &'static [(&'static str, bool)];

const QWEN: Specials = &[
    ("<|endoftext|>", true),
    ("<|im_start|>", true),
    ("<|im_end|>", true),
    ("<|vision_start|>", true),
    ("<|vision_end|>", true),
    ("<|image_pad|>", true),
    ("<|video_pad|>", true),
    ("<think>", false),
    ("</think>", false),
    ("<tool_call>", false),
    ("</tool_call>", false),
    ("<tool_response>", false),
    ("</tool_response>", false),
];

const GLM: Specials = &[
    ("[gMASK]", true),
    ("<sop>", true),
    ("<|endoftext|>", true),
    ("<|system|>", true),
    ("<|user|>", true),
    ("<|assistant|>", true),
    ("<|observation|>", true),
    ("<think>", false),
    ("</think>", false),
    ("<tool_call>", false),
    ("</tool_call>", false),
    ("<tool_response>", false),
    ("</tool_response>", false),
    ("<arg_key>", false),
    ("</arg_key>", false),
    ("<arg_value>", false),
    ("</arg_value>", false),
];

const V41: Specials = &[
    ("<｜begin▁of▁sentence｜>", true),
    ("<｜end▁of▁sentence｜>", true),
    ("<｜System｜>", false),
    ("<｜User｜>", false),
    ("<｜Assistant｜>", false),
    ("<｜latest_reminder｜>", false),
    ("<think>", false),
    ("</think>", false),
    ("｜DSML｜", false),
    ("<｜tool▁calls▁begin｜>", false),
];

const NEMOTRON: Specials = &[
    ("<|endoftext|>", true),
    ("<|im_start|>", true),
    ("<|im_end|>", true),
    ("<think>", false),
    ("</think>", false),
    ("<tool_call>", false),
    ("</tool_call>", false),
    ("<tool_response>", false),
    ("</tool_response>", false),
    ("<image>", true),
    ("<video>", true),
];

const SOLAR: Specials = &[
    ("<|im:start|>", true),
    ("<|im:content|>", true),
    ("<|im:end|>", true),
    ("<|think:start|>", true),
    ("<|think:end|>", true),
    ("<|tool_call:start|>", true),
    ("<|tool_call:end|>", true),
    ("<|tool_arg:start|>", true),
    ("<|tool_arg:value|>", true),
    ("<|tool_arg:end|>", true),
    ("<|tool_response:start|>", true),
    ("<|tool_response:end|>", true),
    ("<|tool:start|>", true),
    ("<|tool:end|>", true),
];

/// The special tokens of the family the fixture template `name` belongs to.
fn specials_of(name: &str) -> Specials {
    match name {
        "qwen3" | "qwen36" | "qwen38" => QWEN,
        "glm5" => GLM,
        "v41" | "v41-spaced" | "v41-official" => V41,
        "nemotron" | "nemotron-bolt" => NEMOTRON,
        "solar" => SOLAR,
        other => panic!("{other} has no special-token table here"),
    }
}

/// Every chat template family the serve tests render: the shared fixtures, Solar's and the V4.1
/// release's spaced DSML spelling.
fn templates() -> Vec<(&'static str, &'static str)> {
    let mut all: Vec<_> = FIXTURE_TEMPLATES.to_vec();
    all.push(("solar", SOLAR_TEMPLATE));
    all.push(("v41-spaced", V41_SPACED_TEMPLATE));
    all.push(("v41-official", crate::v41::TEMPLATE));
    all
}

/// A byte-level BPE vocabulary of the 256 byte tokens, one merge and `specials`, the first the BOS and
/// the second the EOS.
fn vocabulary(specials: Specials) -> tokenizer::Tokenizer {
    use gguf::Value as G;
    use gguf::write::{Layout, Writer};
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let mut texts: Vec<String> = tokenizer::byte_to_cpt_table()
        .into_iter()
        .map(|cp| {
            char::from_u32(cp)
                .expect("a GPT-2 alphabet code point")
                .to_string()
        })
        .collect();
    let mut types = vec![1; texts.len()];
    texts.push("ab".to_owned());
    types.push(1);
    for &(text, control) in specials {
        texts.push(text.to_owned());
        types.push(if control { 3 } else { 4 });
    }
    let first = 257;
    let kvs: Vec<(String, G)> = vec![
        ("tokenizer.ggml.model".into(), G::String("gpt2".into())),
        ("tokenizer.ggml.pre".into(), G::String("qwen2".into())),
        (
            "tokenizer.ggml.tokens".into(),
            G::Array(texts.into_iter().map(G::String).collect()),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            G::Array(types.into_iter().map(G::I32).collect()),
        ),
        (
            "tokenizer.ggml.merges".into(),
            G::Array(vec![G::String("a b".into())]),
        ),
        ("tokenizer.ggml.bos_token_id".into(), G::U32(first)),
        ("tokenizer.ggml.eos_token_id".into(), G::U32(first + 1)),
    ];
    let layout = Layout::new(&kvs, Vec::new()).expect("a layout");
    let bytes = Writer::new(Vec::new(), layout)
        .and_then(Writer::finish)
        .expect("a header");
    let path = std::env::temp_dir().join(format!(
        "serve-ctl-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, bytes).expect("write the vocabulary");
    let tok = tokenizer::Tokenizer::from_gguf(&path);
    std::fs::remove_file(&path).expect("remove the vocabulary");
    tok.expect("the vocabulary loads")
}

/// The server's tokenizer over the real one: what `bind`'s `Vocab` does for a GGUF file.
struct Vocab(Arc<tokenizer::Tokenizer>);

impl Tokenizer for Vocab {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.0.encode(text, false, true)
    }
    fn guard<'a>(&self, text: &'a str, mark: &str) -> Cow<'a, str> {
        self.0.guard(text, mark)
    }
    fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
        self.0.encode_marked(text, mark, false, true)
    }
    fn decode(&self, ids: &[u32]) -> String {
        self.0.decode(ids)
    }
    fn decoder(&self) -> Box<dyn Decoder> {
        unreachable!("the chat-control tests do not generate")
    }
    fn bos(&self) -> Option<u32> {
        self.0.specials().bos
    }
    fn eos(&self) -> u32 {
        self.0.specials().eos.expect("an EOS")
    }
    fn add_bos(&self) -> bool {
        false
    }
    fn n_vocab(&self) -> usize {
        self.0.n_vocab()
    }
}

/// The mock engine behind a chosen vocabulary, and a media model when it takes images; only the
/// render and the prompt are asked of it.
struct Served(MockEngine, Arc<dyn Tokenizer>, Option<SharedMediaModel>);

impl Engine for Served {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::clone(&self.1)
    }
    fn media_model(&self) -> Option<SharedMediaModel> {
        self.2.clone()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.0.prefill(ids)
    }
    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.0.next(last, logits_out)
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.0.reset()
    }
    fn ctx_max(&self) -> usize {
        self.0.ctx_max()
    }
    fn describe(&self) -> String {
        self.0.describe()
    }
}

fn object(body: &Value) -> &Map<String, Value> {
    let Value::Object(map) = body else {
        unreachable!("a body is an object")
    };
    map
}

/// The request preparation the family's template needs: the official V4.1
/// encoding's own, the other families' none.
fn preparation_of(name: &str) -> Option<crate::ChatPreparation> {
    match name {
        "v41-official" => Some(crate::v41::prepare),
        _ => None,
    }
}

/// One family's server: its template over its vocabulary.
struct Family {
    state: Arc<State>,
    tok: Arc<tokenizer::Tokenizer>,
    specials: Specials,
}

fn family(name: &str, template: &str) -> Family {
    served(specials_of(name), template, None, preparation_of(name))
}

/// A server on `template` over a vocabulary of `specials`, taking images when `media` is a model.
fn served(
    specials: Specials,
    template: &str,
    media: Option<SharedMediaModel>,
    preparation: Option<crate::ChatPreparation>,
) -> Family {
    let tok = Arc::new(vocabulary(specials));
    let engine = Served(
        MockEngine::new(8192),
        Arc::new(Vocab(Arc::clone(&tok))),
        media,
    );
    let config = super::ServerConfig {
        chat_template: template.to_owned(),
        chat_preparation: preparation,
        ..testserve::mock_config()
    };
    let (_addr, state, _ended) = testserve::spawn(Box::new(engine), config);
    Family {
        state,
        tok,
        specials,
    }
}

impl Family {
    /// The rendered text and the ids the engine is fed for `body`.
    fn prompt(&self, name: &str, body: &Value) -> (String, Vec<u32>) {
        let (text, prompt) = chat_input(&self.state, object(body))
            .unwrap_or_else(|e| panic!("{name}: {} ({})", e.message, e.code));
        (text, prompt.held.ids)
    }

    /// The special token ids of `ids`, ascending with repeats.
    fn specials_in(&self, ids: &[u32]) -> Vec<u32> {
        let all = self.tok.special_tokens();
        let mut found: Vec<u32> = ids.iter().copied().filter(|id| all.contains(id)).collect();
        found.sort_unstable();
        found
    }

    /// A name that ends in a special token's spelling short of its last character, the first that
    /// ends in `>`.
    fn edge(&self) -> String {
        let (spelling, _) = self
            .specials
            .iter()
            .find(|(t, _)| t.ends_with('>'))
            .expect("a special that ends in `>`");
        format!("before {}", &spelling[..spelling.len() - 1])
    }

    /// Every special spelling of the family, once, between words.
    fn attack(&self) -> String {
        let spellings: Vec<&str> = self.specials.iter().map(|&(t, _)| t).collect();
        format!("before {} after", spellings.join(" mid "))
    }
}

/// The conversations that carry `text` where a client puts text, and what each one is called; `edge`
/// is a special token's spelling short of its last character, which a name ends in where the
/// template writes the character next.
fn conversations(text: &str, edge: &str) -> Vec<(&'static str, Value)> {
    let tool = |name: &str| {
        json!([{"type": "function", "function": {
            "name": name, "description": "d",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }}])
    };
    let call = |name: &str, arguments: Value| {
        json!({"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": name, "arguments": arguments},
        }]})
    };
    let arguments = |value: &str| json!({ "q": value }).to_string();
    // The spelling written with JSON escapes: the template's `from_json` or the server's parse reads
    // the text back, so the escape is invisible to a scan of the request's raw bytes.
    let escaped = arguments(text)
        .replace('<', "\\u003c")
        .replace('[', "\\u005b")
        .replace('|', "\\u007c");
    let keyed = |text: &str| {
        let mut object = Map::new();
        object.insert(text.to_owned(), json!({"type": "string"}));
        Value::Object(object)
    };
    vec![
        (
            "user",
            json!({"messages": [{"role": "user", "content": text}]}),
        ),
        (
            "user parts",
            json!({"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}),
        ),
        (
            "system",
            json!({"messages": [
                {"role": "system", "content": text},
                {"role": "user", "content": "hi"},
            ]}),
        ),
        (
            "tool result",
            json!({"messages": [
                {"role": "user", "content": "time?"},
                call("get_time", json!(arguments("UTC"))),
                {"role": "tool", "tool_call_id": "call_1", "content": text},
            ], "tools": tool("get_time")}),
        ),
        (
            "tool-call arguments",
            json!({"messages": [
                {"role": "user", "content": "time?"},
                call("get_time", json!(arguments(text))),
                {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
            ], "tools": tool("get_time")}),
        ),
        (
            "tool-call arguments with JSON escapes",
            json!({"messages": [
                {"role": "user", "content": "time?"},
                call("get_time", json!(escaped)),
                {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
            ], "tools": tool("get_time")}),
        ),
        (
            "tool-call argument name",
            json!({"messages": [
                {"role": "user", "content": "time?"},
                call("get_time", json!(keyed(text).to_string())),
                {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
            ], "tools": tool("get_time")}),
        ),
        (
            "tool parameter name",
            json!({"messages": [{"role": "user", "content": "time?"}], "tools": [{
                "type": "function", "function": {
                    "name": "get_time", "description": "d",
                    "parameters": {"type": "object", "properties": keyed(text)},
                },
            }]}),
        ),
        (
            "tool-call name finished by the template",
            json!({"messages": [
                {"role": "user", "content": "time?"},
                call(edge, json!(arguments("UTC"))),
                {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
            ], "tools": tool("get_time")}),
        ),
        (
            "tool-call name",
            json!({"messages": [
                {"role": "user", "content": "time?"},
                call(text, json!(arguments("UTC"))),
                {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
            ], "tools": tool("get_time")}),
        ),
        (
            "tool description",
            json!({"messages": [{"role": "user", "content": "time?"}], "tools": json!([{
                "type": "function", "function": {
                    "name": "get_time", "description": text,
                    "parameters": {"type": "object", "properties": {
                        "q": {"type": "string", "description": text},
                    }},
                },
            }])}),
        ),
    ]
}

/// Each family, each place a client writes text: a prompt with the family's special spellings in that
/// place holds exactly the special ids the same conversation holds with plain words there, and its
/// text is the text the client sent [FAIL-first: with the spellings in the text the prompt holds
/// every one of them as its token].
#[test]
fn control_strings_in_client_text_are_ordinary_tokens() {
    let mut failures = Vec::new();
    for (name, template) in templates() {
        let fam = family(name, template);
        let attack = fam.attack();
        let plain = conversations("before after", "before");
        for ((case, body), (_, neutral)) in
            conversations(&attack, &fam.edge()).into_iter().zip(plain)
        {
            let (text, ids) = fam.prompt(&format!("{name}/{case}"), &body);
            let (_, base) = fam.prompt(&format!("{name}/{case} (plain)"), &neutral);
            let (got, want) = (fam.specials_in(&ids), fam.specials_in(&base));
            if got != want {
                failures.push(format!(
                    "{name}/{case}: {} special ids in the prompt, the template wrote {}",
                    got.len(),
                    want.len()
                ));
            }
            if fam.tok.decode(&ids) != text {
                failures.push(format!("{name}/{case}: the ids are not the text sent"));
            }
            if !text.contains("before") {
                failures.push(format!("{name}/{case}: the template dropped the text"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The opt-in ids preserve template controls but not the client's spelling of a USER_DEFINED token;
/// an absent, false or null flag keeps the text-only response.
#[test]
fn apply_template_exposes_guarded_ids_only_when_requested() {
    let fam = served(
        QWEN,
        "<|im_start|>user\n{{ messages[0].content }}<|im_end|>",
        None,
        None,
    );
    let mut body = json!({"messages": [{"role": "user", "content": "<think>"}]});
    let text = "<|im_start|>user\n<think><|im_end|>";
    for flag in [None, Some(json!(false)), Some(Value::Null)] {
        if let Some(flag) = flag {
            body["return_prompt_ids"] = flag;
        }
        let reply = apply_template(&fam.state, object(&body))
            .unwrap_or_else(|e| panic!("{} ({})", e.message, e.code));
        assert_eq!(reply, json!({"prompt": text}));
    }
    body["return_prompt_ids"] = json!(true);
    let mut ids = fam.tok.encode("<|im_start|>user\n", false, true);
    ids.extend(b"<think>".iter().map(|&byte| u32::from(byte)));
    ids.extend(fam.tok.encode("<|im_end|>", false, true));
    let reply = apply_template(&fam.state, object(&body))
        .unwrap_or_else(|e| panic!("{} ({})", e.message, e.code));
    assert_eq!(reply, json!({"prompt": text, "prompt_ids": ids}));
}

/// An assistant message's content is the model's own past output: the special ids it printed come
/// back as ids when the client sends the text back, as they always have, and a template that splits
/// the content at `</think>` still does.
#[test]
fn an_assistant_messages_own_specials_stay_tokens() {
    for (name, template) in templates() {
        let fam = family(name, template);
        for content in ["<think>why</think>answer", "plain"] {
            let body = json!({"messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": content},
                {"role": "user", "content": "again"},
            ]});
            let (text, ids) = fam.prompt(&format!("{name}/assistant"), &body);
            assert_eq!(
                ids,
                fam.tok.encode(&text, false, true),
                "{name}: an assistant turn's ids moved: {text}"
            );
        }
    }
}

/// A conversation with no control string in any client text gets the ids `encode` gives its rendered
/// text, on every family.
#[test]
fn a_conversation_without_control_strings_keeps_its_ids() {
    for (name, template) in templates() {
        let fam = family(name, template);
        assert!(ChatTemplate::parse(template).is_ok(), "{name}");
        for (case, body) in conversations(
            "a plain request with <angle> brackets, | pipes and [squares]",
            "plain",
        ) {
            let (text, ids) = fam.prompt(&format!("{name}/{case}"), &body);
            assert_eq!(ids, fam.tok.encode(&text, false, true), "{name}/{case}");
        }
    }
}

/// Every fixture conversation of the families that have renders (`tests/fixtures/*-renders.json`),
/// the tool calls, tool results, reasoning and tools in them, gets the ids `encode` gives its
/// rendered text: nothing a template wrote or a client sent without a spelling in it moves
/// [`a_conversation_without_control_strings_keeps_its_ids`] holds the same of the shapes above. The
/// cases the template refuses are refused as they were.
#[test]
fn the_fixture_conversations_keep_their_ids() {
    let fixtures: [(&str, &str, &str); 4] = [
        (
            "qwen3",
            include_str!("../tests/fixtures/qwen3-chat-template.jinja"),
            include_str!("../tests/fixtures/qwen3-renders.json"),
        ),
        (
            "qwen38",
            include_str!("../tests/fixtures/qwen38-chat-template.jinja"),
            include_str!("../tests/fixtures/qwen38-renders.json"),
        ),
        (
            "glm5",
            include_str!("../tests/fixtures/glm5-chat-template.jinja"),
            include_str!("../tests/fixtures/glm5-renders.json"),
        ),
        (
            "solar",
            SOLAR_TEMPLATE,
            include_str!("../tests/fixtures/solar-renders.json"),
        ),
    ];
    let mut failures = Vec::new();
    let mut compared = 0;
    for (name, template, renders) in fixtures {
        let fam = family(name, template);
        let fixture: Value = serde_json::from_str(renders).expect("fixture JSON");
        for case in fixture["cases"].as_array().expect("cases") {
            let label = format!("{name}/{}", case["name"]);
            // Only Solar's template refuses a conversation itself (`raise_exception`); the other
            // cases jinja2 refuses are tool-call `arguments` strings the server parses.
            let refused = name == "solar"
                && case["error"]
                    .as_str()
                    .is_some_and(|e| e.starts_with("TemplateError: "));
            match chat_input(&fam.state, object(&case["body"])) {
                Err(_) if refused => {}
                Err(e) => failures.push(format!("{label}: refused: {}", e.message)),
                Ok(_) if refused => failures.push(format!("{label}: not refused")),
                Ok((text, prompt)) => {
                    compared += 1;
                    if prompt.held.ids != fam.tok.encode(&text, false, true) {
                        failures.push(format!("{label}: the ids moved"));
                    }
                }
            }
        }
    }
    assert!(compared > 40, "{compared} cases compared");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A template that slices a text through its mark leaves a piece of it in the render, and the request
/// is a 400 naming that; a slice that stops before the mark, and a text with no spelling, render as
/// they do without the guard.
#[test]
fn a_template_that_cuts_a_guarded_text_apart_is_refused_by_name() {
    let slicing = |n: usize| {
        let template = format!("{{%- for m in messages %}}{{{{ m.content[:{n}] }}}}{{% endfor %}}");
        served(QWEN, &template, None, None)
    };
    let body = |text: &str| json!({"messages": [{"role": "user", "content": text}]});
    let spelled = body("x<think>y");
    // The mark, four symbols, stands after `x`: a slice that keeps two of them leaves a piece of
    // the mark, one that keeps all four takes the mark whole, and both change the client's text.
    for keep in [2, 4, 5] {
        let refused = chat_input(&slicing(keep).state, object(&spelled))
            .expect_err("a slice that changes the text is refused");
        assert_eq!(refused.code, 400, "a slice of {keep}");
        assert!(
            refused
                .message
                .contains("cut a text that carries a special-token spelling apart"),
            "{}",
            refused.message
        );
    }
    // A slice that stops before the mark renders the text as it is.
    let fam = slicing(1);
    let (text, prompt) = chat_input(&fam.state, object(&spelled))
        .unwrap_or_else(|e| panic!("a slice of 1: {}", e.message));
    assert_eq!(text, "x");
    assert_eq!(prompt.held.ids, fam.tok.encode("x", false, true));
    let (text, _) = chat_input(&slicing(4).state, object(&body("no spelling here")))
        .unwrap_or_else(|e| panic!("plain: {}", e.message));
    assert_eq!(text, "no s");
}

/// A client text a template trims keeps the template's own tokens: the guard's mark is inside the
/// text, so the trim leaves it there, the turn marker the template writes beside the text stays a
/// token, and the client's own spelling stays text.
#[test]
fn a_trimmed_client_text_keeps_the_turn_markers() {
    let (name, template) = templates()
        .into_iter()
        .find(|(n, _)| *n == "qwen36")
        .expect("the qwen36 template");
    let fam = family(name, template);
    let plain = json!({"messages": [{"role": "user", "content": "hi"}]});
    let (_, base) = fam.prompt("qwen36/plain", &plain);
    // Jinja's trim takes Python's whitespace, which holds the four separators U+001C–U+001F.
    // A whole spelling, the end of one and the beginning of one, each beside the trimmed space.
    for space in ["\n", "\u{1c}", "\u{1d}", "\u{1e}", "\u{1f}"] {
        for (content, kept) in [
            (format!("<think>{space}"), "<think>"),
            (format!("think>{space}"), "think>"),
            (format!(">{space}"), ">"),
            (format!("{space}<thi"), "<thi"),
            (format!("{space}<"), "<"),
        ] {
            let body = json!({"messages": [{"role": "user", "content": content}]});
            let (text, ids) = fam.prompt("qwen36/trimmed", &body);
            assert_eq!(
                fam.specials_in(&ids),
                fam.specials_in(&base),
                "{content:?} {ids:?}"
            );
            assert!(text.contains(kept), "{text}");
            assert_eq!(fam.tok.decode(&ids), text);
        }
    }
}

/// Tool arguments nested as deep as the argument parser takes still render when a marked text
/// sits beside them: the unmarked render walks the variables, it does not parse them again.
#[test]
fn deep_tool_arguments_beside_a_marked_text_render() {
    let (name, template) = templates()
        .into_iter()
        .find(|(n, _)| *n == "qwen36")
        .expect("the qwen36 template");
    let fam = family(name, template);
    let deep = format!(r#"{{"q":{}{}}}"#, "[".repeat(124), "]".repeat(124));
    let body = json!({"messages": [
        {"role": "user", "content": "<think>"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "f", "arguments": deep},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]});
    let (text, ids) = fam.prompt("qwen36/deep arguments", &body);
    assert!(text.contains("<think>"), "{text}");
    assert_eq!(fam.tok.decode(&ids), text);
}

/// A tool call's arguments that hold a special token's spelling in JSON escapes keep their text:
/// the mark the guard uses is one the decoded arguments do not hold, so the text the template
/// renders is the text the client sent, byte for byte.
#[test]
fn escaped_tool_arguments_keep_their_text() {
    let (name, template) = templates()
        .into_iter()
        .find(|(n, _)| *n == "qwen3")
        .expect("the qwen3 template");
    let fam = family(name, template);
    let body = json!({"messages": [
        {"role": "user", "content": "time?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "get_time", "arguments": r#"{"q":"\u007e\u007e\u007e\u005e<think>"}"#},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "12:00"},
    ], "tools": [{
        "type": "function", "function": {
            "name": "get_time", "description": "d",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }]});
    let (text, ids) = fam.prompt("qwen3/escaped arguments", &body);
    assert!(text.contains("~~~^<think>"), "{text}");
    assert_eq!(fam.tok.decode(&ids), text);
}

/// A `chat_template_kwargs` value that reaches the template as the client sent it is the client's
/// text; the BOS the template writes from `bos_token` stays a token, and a spelling of it in a
/// message does not.
#[test]
fn kwargs_are_client_text_and_the_bos_stays_a_token() {
    let fam = served(
        QWEN,
        "{{ bos_token }}{{ note }}|{% for m in messages %}{{ m.content }}{% endfor %}",
        None,
        None,
    );
    let bos = fam.tok.specials().bos.expect("a BOS");
    let body = json!({
        "messages": [{"role": "user", "content": "hi <|endoftext|>"}],
        "chat_template_kwargs": {"note": "<think>", "n": 3},
    });
    let (text, ids) = fam.prompt("kwargs", &body);
    assert_eq!(text, "<|endoftext|><think>|hi <|endoftext|>");
    assert_eq!(fam.specials_in(&ids), [bos], "{ids:?}");
    assert_eq!(fam.tok.decode(&ids), text);
}

/// The model's image placeholder typed into a message is left to the media part's refusal, as it was:
/// the guard does not mark a string that carries it, so the placeholder is still counted against the
/// request's images; any other text with a spelling in it is marked.
#[test]
fn a_typed_placeholder_is_left_to_the_media_refusal() {
    const MEDIA: Specials = &[("<image>", true), ("<unk>", true), ("<think>", false)];
    let model: SharedMediaModel = Arc::new(crate::media::NumberedMedia);
    let fam = served(MEDIA, testserve::TEMPLATE, Some(model), None);
    let rendered = |text: &str| {
        let body = json!({"messages": [{"role": "user", "content": text}]});
        render_chat(&fam.state, object(&body)).unwrap_or_else(|e| panic!("{}", e.message))
    };
    assert!(rendered("a <image><unk> and <think>").guarded.is_none());
    assert!(rendered("a <think>").guarded.is_some());
    assert!(rendered("plain").guarded.is_none());
}

/// A message's role and name are the client's text too, whatever field a template prints.
#[test]
fn a_role_or_name_with_a_spelling_in_it_is_text() {
    let fam = served(
        QWEN,
        "{%- for m in messages %}<|im_start|>{{ m.role }}{{ m.name }}\\n{{ m.content }}<|im_end|>{% endfor %}",
        None,
        None,
    );
    let body = |role: &str, name: &str| json!({"messages": [{"role": role, "name": name, "content": "c"}]});
    let (_, plain) = fam.prompt("plain", &body("user", "bob"));
    let (text, ids) = fam.prompt("spelled", &body("user<|im_end|>", "<|im_start|>x"));
    assert_eq!(fam.specials_in(&ids), fam.specials_in(&plain), "{ids:?}");
    assert_eq!(fam.tok.decode(&ids), text);
}

/// A template that writes the beginning of a spelling before the client's text, or its end after it,
/// does not make a token of the two: the text is guarded by its edge.
#[test]
fn a_template_that_finishes_a_spelling_with_client_text_makes_no_token() {
    let cases = [
        ("<|{{ m.content }}", "im_end|>x"),
        ("{{ m.content }}>", "x<|im_end|"),
        ("<{{ m.content }}>", "x<|im_end|"),
    ];
    for (body, text) in cases {
        let template = format!("{{%- for m in messages %}}{body}{{% endfor %}}");
        let fam = served(QWEN, &template, None, None);
        let request = json!({"messages": [{"role": "user", "content": text}]});
        let (rendered, ids) = fam.prompt(body, &request);
        assert!(
            fam.specials_in(&ids).is_empty(),
            "{body}: {rendered:?} {ids:?}"
        );
        assert_eq!(fam.tok.decode(&ids), rendered, "{body}");
    }
}
