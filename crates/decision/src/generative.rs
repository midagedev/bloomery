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
//! A question's options are told apart by their first token only when no two share it. The
//! consumer reads that off the model's tokenizer; this module takes the scores it gets.

use serde_json::{Map, Value, json};

use crate::Error;
use crate::answer::{probabilities, question_answer};
use crate::encode::{NO_IMAGE_INPUT, schema_text};
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

/// `req` as the chat a generative model answers. A request with images is
/// [`Error::NotSupported`] (a text model reads none).
pub fn ask(req: &Request) -> Result<Ask, Error> {
    if !req.images.is_empty() {
        return Err(Error::NotSupported(NO_IMAGE_INPUT.into()));
    }
    let ids: Vec<Value> = req.questions.iter().map(|q| json!(q.id)).collect();
    let properties: Map<String, Value> = req
        .questions
        .iter()
        .map(|q| (q.id.clone(), property(q)))
        .collect();
    Ok(Ask {
        system: format!("{SYSTEM_PROMPT}{}", schema_text(req)),
        user: format!("STATE:\n{}", render(&req.state)),
        schema: json!({ "type": "object", "properties": properties, "required": ids }),
    })
}

/// One question's property: the option ids of a choice in the order [`Question::options`] gives
/// them, a score's levels as integers, a noul's boolean (the program writes `true` before
/// `false`, as Clef lists a noul's options).
fn property(q: &Question) -> Value {
    match q.kind {
        Kind::Choice => json!({
            "type": "string",
            "enum": q.options().into_iter().map(|o| o.id).collect::<Vec<_>>(),
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
    use crate::encode::encode_with;
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

    /// The system message is the instruction and Clef's schema block, the user message the state
    /// alone, and the schema holds each question as a property, in request order.
    #[test]
    fn the_chat_is_the_schema_text_and_the_state() {
        let req = request(REQUEST);
        let a = ask(&req).unwrap();
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
        let a = ask(&request(REQUEST)).unwrap();
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
        assert!(matches!(ask(&req), Err(Error::NotSupported(m)) if m == NO_IMAGE_INPUT));
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
}
