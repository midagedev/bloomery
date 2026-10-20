//! A generative model as a decision model: the SystemOne request as a chat with a JSON schema, and
//! the model's decisions as the SystemOne answers.
//!
//! The model is asked in the schema text Clef reads ([`schema_text`]), one system message (an
//! instruction beside it) and the state alone as the user message. Its reply is held to a JSON
//! object by a schema ([`Ask::schema`]): a question id a property, a `choice` a string enum in
//! Clef's option order, a `score` an integer enum of its levels, a `noul` a boolean. The model
//! decides each property at one token, the first that tells its options apart; the logits of
//! those tokens are the options' scores ([`Decision`]), and the answers are written by
//! [`answer::question_answer`], the formulas Clef's seat writes its own with.
//!
//! One pick scores the options of a question when its options' token sequences share a leading
//! run (possibly none) and the token after it differs between every pair. The tokenizer is the
//! caller's ([`Tokens`]), so [`ask`] reads that off the ids it gets. A `choice` whose options
//! are not told apart so (`ab`, `ac` and `d` start with the same token) is asked with the
//! labels `A`, `B`, ... in the option order instead: its enum holds the labels, the schema text
//! shows each beside its option id, and the label row is the question's one pick. The scores
//! it gives are in option order, so the answers name the options and no label. A `score` or a
//! `noul` that is not told apart is a named error.

use serde_json::{Map, Value, json};

use crate::Error;
use crate::answer::{probabilities, question_answer};
use crate::encode::{NO_IMAGE_INPUT, schema_text_labelled};
use crate::json::Json;
use crate::render::render;
use crate::request::{Kind, Question, Request};

/// The instruction beside the schema text in the system message.
pub const SYSTEM_PROMPT: &str = "Read the state and the schema fields. Decide every field. Reply \
with one JSON object: one key per field ID, each value exactly one of that field's allowed \
options.";

/// A request as a generative model is asked it.
#[derive(Clone, Debug, PartialEq)]
pub struct Ask {
    /// The system message: [`SYSTEM_PROMPT`] and the schema text.
    pub system: String,
    /// The user message: the state, rendered.
    pub user: String,
    /// The JSON schema the reply follows, its properties in the request's question order.
    pub schema: Value,
    /// A question a reply names its option by label, in request order: its labels in option
    /// order. `None` for a question answered by option id.
    pub labels: Vec<Option<Vec<String>>>,
}

/// The ids the model's tokenizer cuts a text into, on its own, as the reply's program cuts each
/// option: the text is the string between a value's quotes ([`json_body`]), a score's level or a
/// noul's boolean as written. No token is an empty answer.
pub type Tokens<'a> = dyn FnMut(&str) -> Vec<u32> + 'a;

/// The text between the quotes of the reply's string `s`: JSON's escapes of its characters, and a
/// `<` as `\u003c`. The server spells every string option of a schema with it.
#[must_use]
pub fn json_body(s: &str) -> String {
    let quoted = Value::String(s.to_owned()).to_string();
    quoted[1..quoted.len() - 1].replace('<', "\\u003c")
}

impl Ask {
    /// The chat body the model answers on `/v1/chat/completions`: the system and user messages,
    /// the schema as the `response_format`, thinking off. The route and anyone who wants the
    /// same request through `response_format` take it from here.
    #[must_use]
    pub fn chat_body(&self, cache_prompt: bool) -> Map<String, Value> {
        [
            (
                "messages",
                json!([
                    { "role": "system", "content": self.system },
                    { "role": "user", "content": self.user },
                ]),
            ),
            (
                "response_format",
                json!({
                    "type": "json_schema",
                    "json_schema": { "name": "answer", "schema": self.schema },
                }),
            ),
            ("chat_template_kwargs", json!({ "enable_thinking": false })),
            ("cache_prompt", json!(cache_prompt)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
    }
}

/// `req` as the chat a generative model answers; `tokens` is its tokenizer. A request with images
/// is [`Error::NotSupported`] (a text model reads none). A `choice` question whose options one
/// pick does not tell apart is answered by label ([`Ask::labels`]); the refusal that remains is
/// a question the labels cannot fix (more options than single-token labels), or a `score` or
/// `noul` not told apart, each by name.
pub fn ask(req: &Request, tokens: &mut Tokens<'_>) -> Result<Ask, Error> {
    if !req.images.is_empty() {
        return Err(Error::NotSupported(NO_IMAGE_INPUT.into()));
    }
    let labels = req
        .questions
        .iter()
        .map(|q| labels_of(q, tokens))
        .collect::<Result<Vec<_>, _>>()?;
    let ids: Vec<Value> = req.questions.iter().map(|q| json!(q.id)).collect();
    let properties: Map<String, Value> = req
        .questions
        .iter()
        .zip(&labels)
        .map(|(q, l)| (q.id.clone(), property(q, l.as_deref())))
        .collect();
    Ok(Ask {
        system: format!("{SYSTEM_PROMPT}{}", schema_text_labelled(req, &labels)),
        user: format!("STATE:\n{}", render(&req.state)),
        schema: json!({ "type": "object", "properties": properties, "required": ids }),
        labels,
    })
}

/// The labels `q`'s options are answered by, or `None` when one pick scores them by their ids.
fn labels_of(q: &Question, tokens: &mut Tokens<'_>) -> Result<Option<Vec<String>>, Error> {
    if !needs_decision(q) {
        return Ok(None);
    }
    let ids: Vec<String> = q.options().into_iter().map(|o| o.id).collect();
    let seqs: Vec<Vec<u32>> = ids.iter().map(|id| tokens(&json_body(id))).collect();
    if let Some(at) = seqs.iter().position(Vec::is_empty) {
        return Err(Error::Question {
            id: q.id.clone(),
            what: format!("the option {:?} is cut to no token", ids[at]),
        });
    }
    if told_apart_by_one_pick(&seqs) {
        return Ok(None);
    }
    match q.kind {
        Kind::Choice => labels_for(q, ids.len(), tokens).map(Some),
        Kind::Score | Kind::Noul => Err(Error::Question {
            id: q.id.clone(),
            what: format!(
                "the options {ids:?} start with the same token, and a {} is not asked by label",
                q.kind.word()
            ),
        }),
    }
}

/// Whether one pick scores `seqs` (two or more token sequences): they share a leading run,
/// possibly none, and the token after it differs between every pair. An option that is the
/// shared run itself, or a prefix of another, leaves no token to pick.
fn told_apart_by_one_pick(seqs: &[Vec<u32>]) -> bool {
    let Some(first) = seqs.first() else {
        return true;
    };
    let shared = (0..first.len())
        .take_while(|&i| seqs.iter().all(|s| s.get(i) == first.get(i)))
        .count();
    let mut next: Vec<u32> = Vec::with_capacity(seqs.len());
    seqs.iter().all(|s| match s.get(shared) {
        Some(t) if !next.contains(t) => {
            next.push(*t);
            true
        }
        _ => false,
    })
}

/// The first `n` of `A`..`Z` whose text is one token of its own, in order: the labels of a
/// question whose options start alike. Fewer than `n` is a named error.
fn labels_for(q: &Question, n: usize, tokens: &mut Tokens<'_>) -> Result<Vec<String>, Error> {
    let mut taken: Vec<u32> = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for letter in 'A'..='Z' {
        let label = letter.to_string();
        if let [id] = tokens(&label)[..]
            && !taken.contains(&id)
        {
            taken.push(id);
            labels.push(label);
            if labels.len() == n {
                return Ok(labels);
            }
        }
    }
    Err(Error::Question {
        id: q.id.clone(),
        what: format!(
            "{n} options start with the same token, and this tokenizer leaves {} single-token \
             labels (A to Z) to tell them apart; give the question fewer options or options \
             whose first tokens differ",
            labels.len()
        ),
    })
}

/// One question's property: the option ids of a choice in the order [`Question::options`] gives
/// them (or `labels`, one an option, when it is answered by label), a score's levels as integers,
/// a noul's boolean (the program writes `true` before `false`, as Clef lists a noul's options).
fn property(q: &Question, labels: Option<&[String]>) -> Value {
    match q.kind {
        Kind::Choice => json!({
            "type": "string",
            "enum": match labels {
                Some(l) => l.to_vec(),
                None => q.options().into_iter().map(|o| o.id).collect(),
            },
        }),
        Kind::Score => json!({
            "type": "integer",
            "enum": (0..q.options().len()).collect::<Vec<_>>(),
        }),
        Kind::Noul => json!({ "type": "boolean" }),
    }
}

/// Whether the model decides `q`: a question of one option has nothing to pick, and its
/// answer is certain.
#[must_use]
pub fn needs_decision(q: &Question) -> bool {
    q.options().len() > 1
}

/// What the model's row left at one question's decision.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    /// The logit of each option's first telling token, in the order [`Question::options`]
    /// gives the options.
    pub scores: Vec<f32>,
    /// The option the model picked (its greedy token among the options'), an index into
    /// `scores`.
    pub picked: usize,
}

/// The answers of `req`, one `(question id, answer)` pair a question in request order, from
/// `decisions`: one for each question [`needs_decision`], in request order. A score that is not
/// finite is refused by name (the model could not evaluate the decision), never read as a
/// probability.
pub fn answers(req: &Request, decisions: &[Decision]) -> Result<Vec<(String, Json)>, Error> {
    let wanted = req.questions.iter().filter(|q| needs_decision(q)).count();
    if decisions.len() != wanted {
        return Err(Error::Logits(format!(
            "{} decisions for {wanted} questions that need one",
            decisions.len()
        )));
    }
    let mut decided = decisions.iter();
    let mut out = Vec::with_capacity(req.questions.len());
    for q in &req.questions {
        let ids: Vec<String> = q.options().into_iter().map(|o| o.id).collect();
        let (p32, picked) = if needs_decision(q) {
            let d = decided
                .next()
                .expect("a decision for each counted question");
            if d.scores.len() != ids.len() || d.picked >= ids.len() {
                return Err(Error::Logits(format!(
                    "{}: {} scores and the pick {} for {} options",
                    q.id,
                    d.scores.len(),
                    d.picked,
                    ids.len()
                )));
            }
            if let Some(i) = d.scores.iter().position(|x| !x.is_finite()) {
                return Err(Error::NonFinite {
                    question: q.id.clone(),
                    option: ids[i].clone(),
                });
            }
            (probabilities(&d.scores), d.picked)
        } else {
            (vec![1.0], 0)
        };
        out.push((
            q.id.clone(),
            question_answer(q, &ids, &p32, Some(&ids[picked]))?,
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answer::{answer, body};
    use crate::encode::{encode_with, schema_text};
    use crate::json::parse;
    use crate::render::dumps;

    const REQUEST: &str = r#"{"model":"m","state":{"msg":"I was charged twice"},"questions":{
        "department":{"type":"choice","instructions":"Which team?",
            "criteria":{"technical":"Bugs","billing":"Pay","account":null}},
        "urgency":{"type":"score","criteria":["Can wait","This week","Today"]},
        "outage":{"type":"noul"},
        "region":{"type":"choice","criteria":{"eu":"Europe"}}}}"#;

    fn request(text: &str) -> Request {
        Request::from_json(&parse(text).unwrap()).unwrap()
    }

    /// A byte-level tokenizer: a token a byte, so options tell apart where their bytes do.
    fn bytes(text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    /// `req` asked of a byte-level tokenizer.
    fn asked(req: &Request) -> Ask {
        ask(req, &mut bytes).unwrap()
    }

    /// A request of one choice question `q` over `ids`.
    fn choice(ids: &[&str]) -> Request {
        let criteria: Vec<String> = ids.iter().map(|id| format!("\"{id}\":\"d\"")).collect();
        request(&format!(
            r#"{{"state":"s","questions":{{"q":{{"type":"choice","criteria":{{{}}}}}}}}}"#,
            criteria.join(",")
        ))
    }

    /// The system message is the instruction and Clef's schema block, the user message the state
    /// alone, and the schema holds each question as a property, in request order.
    #[test]
    fn the_chat_is_the_schema_text_and_the_state() {
        let req = request(REQUEST);
        let a = asked(&req);
        assert_eq!(a.system, format!("{SYSTEM_PROMPT}{}", schema_text(&req)));
        assert!(a.system.contains("\nFIELD 2\nID: urgency\nTYPE: score\n"));
        assert_eq!(a.user, "STATE:\n{\"msg\":\"I was charged twice\"}");
        let props = a.schema["properties"].as_object().unwrap();
        assert_eq!(
            props.keys().collect::<Vec<_>>(),
            ["department", "urgency", "outage", "region"]
        );
        assert_eq!(
            a.schema["required"],
            json!(["department", "urgency", "outage", "region"])
        );
    }

    /// Each property's enum is the order the answers score the options in: a choice's sorted
    /// ids, a score's levels, and a noul a boolean (`true` first, as the program writes it).
    #[test]
    fn the_schema_follows_the_option_order() {
        let a = asked(&request(REQUEST));
        let p = &a.schema["properties"];
        assert_eq!(
            p["department"],
            json!({"type":"string","enum":["account","billing","technical"]})
        );
        assert_eq!(p["urgency"], json!({"type":"integer","enum":[0,1,2]}));
        assert_eq!(p["outage"], json!({"type":"boolean"}));
        let req = request(REQUEST);
        let noul = req.questions[2].options();
        assert_eq!(
            noul.iter().map(|o| o.id.as_str()).collect::<Vec<_>>(),
            ["true", "false"]
        );
    }

    #[test]
    fn a_request_with_an_image_is_not_supported() {
        let req = request(
            r#"{"state":"s","images":["data:image/png;base64,AAAA"],"questions":{"q":{"type":"noul"}}}"#,
        );
        assert!(
            matches!(ask(&req, &mut bytes), Err(Error::NotSupported(m)) if m == NO_IMAGE_INPUT)
        );
    }

    /// The answers a generative model's decisions give are the ones Clef's head gives for the
    /// same scores: same keys, order and formulas, a question of one option certain.
    #[test]
    fn the_answers_are_clefs_for_the_same_scores() {
        let req = request(REQUEST);
        let decisions = [
            Decision {
                scores: vec![0.5, -1.0, 2.0],
                picked: 2,
            },
            Decision {
                scores: vec![0.0, 1.0, 0.5],
                picked: 1,
            },
            Decision {
                scores: vec![1.0, -1.0],
                picked: 0,
            },
        ];
        let got = Json::Object(answers(&req, &decisions).unwrap());
        // Clef scores the same options in span order, the lone option's row included.
        let enc = encode_with(
            &mut |t: &str| t.bytes().map(u32::from).collect(),
            &req,
            16384,
        )
        .unwrap();
        let logits: Vec<Vec<f32>> = vec![
            decisions[0].scores.clone(),
            decisions[1].scores.clone(),
            decisions[2].scores.clone(),
            vec![0.0],
        ];
        let Json::Object(clef) = answer(&req, &enc, &logits, "m").unwrap() else {
            panic!("a body is an object")
        };
        let theirs = clef.iter().find(|(k, _)| k == "answers").unwrap().1.clone();
        assert_eq!(dumps(&got, false), dumps(&theirs, false));
        assert!(dumps(&got, false).contains(r#""region":{"type":"choice","choice":"eu","confidence":1.0,"probabilities":{"eu":1.0}}"#));
        assert_eq!(
            dumps(&body("m", vec![("q".into(), Json::Null)], 7, 3), false),
            r#"{"model":"m","answers":{"q":null},"usage":{"input_tokens":7,"output_tokens":3}}"#
        );
    }

    /// A choice reports the model's pick even where the printed probabilities tie.
    #[test]
    fn a_choice_reports_the_models_pick() {
        let req = request(
            r#"{"state":"s","questions":{"q":{"type":"choice","criteria":{"a":"A","b":"B"}}}}"#,
        );
        let tie = |picked| {
            let d = [Decision {
                scores: vec![1.0, 1.0],
                picked,
            }];
            dumps(&Json::Object(answers(&req, &d).unwrap()), false)
        };
        assert!(tie(1).contains(r#""choice":"b","confidence":0.0"#));
        assert!(tie(0).contains(r#""choice":"a","confidence":0.0"#));
    }

    #[test]
    fn a_decision_that_cannot_be_read_is_refused_by_name() {
        let req = request(r#"{"state":"s","questions":{"q":{"type":"noul"}}}"#);
        let one = |scores: Vec<f32>, picked| answers(&req, &[Decision { scores, picked }]);
        assert!(matches!(
            one(vec![f32::NAN, 0.0], 0),
            Err(Error::NonFinite { question, option }) if question == "q" && option == "true"
        ));
        assert!(matches!(
            one(vec![f32::NEG_INFINITY, 0.0], 0),
            Err(Error::NonFinite { .. })
        ));
        assert!(matches!(one(vec![0.0], 0), Err(Error::Logits(_))));
        assert!(matches!(one(vec![0.0, 0.0], 2), Err(Error::Logits(_))));
        assert!(matches!(answers(&req, &[]), Err(Error::Logits(_))));
    }

    /// A request whose options one pick tells apart is asked exactly as it was before options
    /// could be labelled: the system message, the user message and the schema, byte for byte.
    #[test]
    fn a_request_told_apart_is_asked_without_labels() {
        let a = asked(&request(REQUEST));
        assert_eq!(a.labels, vec![None, None, None, None]);
        assert_eq!(
            a.system,
            "Read the state and the schema fields. Decide every field. Reply with one JSON \
object: one key per field ID, each value exactly one of that field's allowed options.\n\
\n\
SCHEMA FIELDS:\n\
\n\
FIELD 1\n\
ID: department\n\
TYPE: choice\n\
INSTRUCTION: Which team?\n\
ALLOWED OPTIONS:\n\
OPTION 1: {\"option_id\":\"account\"}\n\
OPTION 2: {\"description\":\"Pay\",\"option_id\":\"billing\"}\n\
OPTION 3: {\"description\":\"Bugs\",\"option_id\":\"technical\"}\n\
END FIELD\n\
\n\
FIELD 2\n\
ID: urgency\n\
TYPE: score\n\
INSTRUCTION: urgency\n\
ALLOWED OPTIONS:\n\
OPTION 1: {\"description\":\"Can wait\",\"option_id\":\"0\"}\n\
OPTION 2: {\"description\":\"This week\",\"option_id\":\"1\"}\n\
OPTION 3: {\"description\":\"Today\",\"option_id\":\"2\"}\n\
END FIELD\n\
\n\
FIELD 3\n\
ID: outage\n\
TYPE: noul\n\
INSTRUCTION: outage\n\
ALLOWED OPTIONS:\n\
OPTION 1: {\"description\":\"The proposition is true or the answer is yes.\",\"option_id\":\"true\"}\n\
OPTION 2: {\"description\":\"The proposition is false or the answer is no.\",\"option_id\":\"false\"}\n\
END FIELD\n\
\n\
FIELD 4\n\
ID: region\n\
TYPE: choice\n\
INSTRUCTION: region\n\
ALLOWED OPTIONS:\n\
OPTION 1: {\"description\":\"Europe\",\"option_id\":\"eu\"}\n\
END FIELD\n"
        );
        assert_eq!(a.user, "STATE:\n{\"msg\":\"I was charged twice\"}");
        assert_eq!(
            a.schema.to_string(),
            concat!(
                r#"{"type":"object","properties":{"department":{"type":"string","enum":["account","billing","technical"]},"#,
                r#""urgency":{"type":"integer","enum":[0,1,2]},"outage":{"type":"boolean"},"#,
                r#""region":{"type":"string","enum":["eu"]}},"#,
                r#""required":["department","urgency","outage","region"]}"#,
            )
        );
    }

    /// Options that share a leading run and differ at the token after it are scored there: the
    /// forced run is not a collision, and the ids stay.
    #[test]
    fn a_shared_prefix_is_told_apart_where_it_differs() {
        let a = asked(&choice(&["refund_part", "refund_full"]));
        assert_eq!(a.labels, vec![None]);
        assert_eq!(
            a.schema["properties"]["q"]["enum"],
            json!(["refund_full", "refund_part"])
        );
    }

    /// Options two of which start with the same token and are not told apart there are asked by
    /// label, in the option order: the enum holds the labels, the system message shows each
    /// beside its option id, and the question beside it is asked as before.
    #[test]
    fn options_that_start_alike_are_asked_by_label() {
        let req = request(
            r#"{"state":"s","questions":{
            "q":{"type":"choice","criteria":{"d":"Dee","ac":null,"ab":"Ab"}},
            "urgency":{"type":"score","criteria":["Low","High"]}}}"#,
        );
        let a = asked(&req);
        assert_eq!(
            a.labels,
            vec![Some(vec!["A".into(), "B".into(), "C".into()]), None]
        );
        assert_eq!(
            a.schema["properties"]["q"],
            json!({"type":"string","enum":["A","B","C"]})
        );
        assert_eq!(
            a.schema["properties"]["urgency"],
            json!({"type":"integer","enum":[0,1]})
        );
        assert!(a.system.contains(concat!(
            "INSTRUCTION: q\n",
            "ALLOWED OPTIONS (reply with the label before the =):\n",
            "OPTION 1: A = {\"description\":\"Ab\",\"option_id\":\"ab\"}\n",
            "OPTION 2: B = {\"option_id\":\"ac\"}\n",
            "OPTION 3: C = {\"description\":\"Dee\",\"option_id\":\"d\"}\n",
            "END FIELD\n",
        )));
        // The question told apart keeps its text: the system message differs only in the first.
        let plain = schema_text(&req);
        let field2 = |t: &str| t[t.find("FIELD 2").unwrap()..].to_owned();
        assert_eq!(field2(&a.system), field2(&plain));
        assert!(!field2(&plain).contains(" = "));
    }

    /// The scores of a label row are in option order, so the answer names the options: the
    /// original ids are `choice` and the keys of `probabilities`, and no label is in the body.
    #[test]
    fn a_label_row_scores_the_original_options() {
        let req = choice(&["d", "ab", "ac"]);
        assert_eq!(
            asked(&req).labels,
            vec![Some(vec!["A".into(), "B".into(), "C".into()])]
        );
        let d = [Decision {
            scores: vec![0.5, 2.0, 0.0],
            picked: 1,
        }];
        let got = dumps(&Json::Object(answers(&req, &d).unwrap()), false);
        let said: Value = serde_json::from_str(&got).unwrap();
        assert_eq!(said["q"]["choice"], "ac");
        let p = probabilities(&d[0].scores);
        let keys: Vec<&str> = said["q"]["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        // The keys are the request's criteria order, the scores the sorted option order.
        assert_eq!(keys, ["d", "ab", "ac"], "{got}");
        for (i, id) in ["ab", "ac", "d"].iter().enumerate() {
            let x = said["q"]["probabilities"][id].as_f64().unwrap();
            assert!(
                (x - f64::from(p[i])).abs() < 1e-4,
                "{id}: {x} against {}",
                p[i]
            );
        }
        for label in ["\"A\"", "\"B\"", "\"C\""] {
            assert!(!got.contains(label), "{got}");
        }
    }

    /// An option that is a prefix of another leaves no token to pick between them, so they are
    /// asked by label as well.
    #[test]
    fn an_option_that_is_a_prefix_is_asked_by_label() {
        let a = asked(&choice(&["a", "ab"]));
        assert_eq!(a.labels, vec![Some(vec!["A".into(), "B".into()])]);
    }

    /// A letter that is not one token, or whose token an earlier label took, is skipped: the
    /// labels are the first letters that stand alone.
    #[test]
    fn a_letter_that_is_not_a_token_of_its_own_is_skipped() {
        let mut odd = |t: &str| match t {
            "C" => vec![67, 67],
            "E" => vec![65],
            _ => bytes(t),
        };
        let a = ask(&choice(&["xa1", "xa2", "xb1", "xb2", "xc1"]), &mut odd).unwrap();
        let labels = a.labels[0].clone().unwrap();
        assert_eq!(labels, ["A", "B", "D", "F", "G"]);
        assert_eq!(
            a.schema["properties"]["q"]["enum"],
            json!(["A", "B", "D", "F", "G"])
        );
    }

    /// More options that start alike than single-token labels is a named error, the question
    /// and the counts in it; 26 is the most the letters hold.
    #[test]
    fn more_options_than_labels_is_a_named_error() {
        let ids: Vec<String> = (0..27).map(|i| format!("p{i:02}")).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let Err(Error::Question { id, what }) = ask(&choice(&refs), &mut bytes) else {
            panic!("27 options were asked")
        };
        assert_eq!(id, "q");
        assert!(
            what.contains("27 options") && what.contains("26 single-token labels"),
            "{what}"
        );
        let a = asked(&choice(&refs[..26]));
        assert_eq!(a.labels[0].as_ref().map(Vec::len), Some(26));
    }

    /// A score or a noul whose options are not told apart is a named error, never labelled:
    /// the levels of a score are the integers themselves.
    #[test]
    fn a_score_that_is_not_told_apart_is_a_named_error() {
        let levels: Vec<String> = (0..11).map(|i| format!("\"l{i}\"")).collect();
        let req = request(&format!(
            r#"{{"state":"s","questions":{{"s":{{"type":"score","criteria":[{}]}}}}}}"#,
            levels.join(",")
        ));
        let Err(Error::Question { id, what }) = ask(&req, &mut bytes) else {
            panic!("11 levels were asked")
        };
        assert_eq!(id, "s");
        assert!(what.contains("is not asked by label"), "{what}");
    }

    #[test]
    fn an_option_cut_to_no_token_is_a_named_error() {
        let mut none = |_: &str| Vec::new();
        assert!(matches!(
            ask(&choice(&["a", "b"]), &mut none),
            Err(Error::Question { id, what }) if id == "q" && what.contains("no token")
        ));
    }

    #[test]
    fn json_body_is_what_the_reply_writes_between_the_quotes() {
        assert_eq!(json_body("a\"b<c\\"), r#"a\"b\u003cc\\"#);
        assert_eq!(json_body("결제"), "결제");
    }
}
