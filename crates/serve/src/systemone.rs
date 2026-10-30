//! `POST /v1/systemone` on a generative seat: the decision models' request answered by a general
//! model, in their answer body.
//!
//! The body is read and validated by `bloomery-decision` (the request reader every decision model
//! shares), asked of the model as a chat whose reply a JSON schema holds ([`generative::ask`]), and
//! run on the structured-output path of `/v1/chat/completions` ([`chat_plan`], the schema as a
//! `response_format`), schema jump included. The reply is a program of forced text and one pick a
//! question, and the pick's row is the question's evidence: the logit of the token each option
//! starts with, read where the walk picks ([`Watch`]). Those logits are the options' scores, and
//! the answer body is written by the formulas of the decision models' own
//! ([`generative::answers`], [`answer::body`], [`with_timings`]): the same keys in the same
//! order, `choice`, `confidence` and `probabilities` of each question as a decision head's.
//!
//! What the model's values mean here, field by field:
//! - `model`: the name this server serves, as the decision seat's.
//! - `answers`: the probabilities are the softmax of the scores over the question's options only,
//!   and a choice is the model's own pick.
//! - `usage`: `input_tokens` is the whole prompt, `output_tokens` the ids the model generated
//!   (a decision head generates none).
//! - `timings`: `prompt_n` the prompt ids evaluated, `prompt_ms` the prompt call, `head_ms` the
//!   rest of the generation, `cache_n` the prompt ids the cache kept.
//!
//! A question's options are scored by one pick when the tokens of their ids share a leading run
//! and differ at the token after it. A `choice` whose options are not told apart so (two start
//! with the same token and part there) is asked by label: [`generative::ask`] gives its enum the
//! labels `A`, `B`, ... in the option order and the schema text shows each beside its option id,
//! so the label row is the question's one pick and its scores are the options'. The answer names
//! the options, never a label. What the labels cannot fix is refused by name: a choice of more
//! options than single-token labels (a 400), a score that is not finite (a 500).
//!
//! Beside the decision models' fields the body takes `cache_prompt`, a boolean (default `true`,
//! the chat path's), which a decision model ignores like any key it does not know.

use std::io;
use std::net::TcpStream;

use decision::answer;
use decision::generative::{self, Decision};
use decision::json::{self, Json};
use decision::render::dumps;
use decision::request::{Question, Request as Asked};

use super::{
    ApiError, ChatPlan, JSON, State, chat_plan, engine_error, invalid, run_whole, send_error,
};
use crate::constrain::{Decision as Picked, Watch};
use crate::decide::{Decided, with_timings};
use crate::genloop::Outcome;
use crate::http::{self, Request};

/// The routes a SystemOne request is posted to: the decision models' own.
pub(crate) const ROUTES: &[&str] = decision::request::ROUTES;

/// `POST /v1/systemone`.
pub(super) fn post(state: &State, req: &Request, w: &mut TcpStream) -> io::Result<bool> {
    let (asked, plan, watch) = match plan(state, &req.body) {
        Ok(x) => x,
        Err(e) => return send_error(w, req, &e),
    };
    let ChatPlan {
        p, input, prompt, ..
    } = plan;
    let outcome = match run_whole(state, &req.path, &input, prompt, &p, w)? {
        Ok((o, _)) => o,
        Err(e) => return send_error(w, req, &e),
    };
    match answered(state, &asked, &watch, &outcome) {
        Ok(text) => {
            http::respond(w, req, 200, JSON, &[], text.as_bytes())?;
            Ok(true)
        }
        Err(e) => send_error(w, req, &e),
    }
}

/// The request validated, and the chat plan it runs: the decision models' request check first
/// (their refusals, their words), then the chat path's own steps on the converted body.
fn plan(state: &State, body: &[u8]) -> Result<(Asked, ChatPlan, Watch), ApiError> {
    let text = std::str::from_utf8(body).map_err(|_| invalid("the request body is not UTF-8"))?;
    let parsed = json::parse(text).map_err(refusal)?;
    let asked = Asked::from_json(&parsed).map_err(refusal)?;
    let cache_prompt = match parsed.get("cache_prompt") {
        None | Some(Json::Null) => true,
        Some(Json::Bool(b)) => *b,
        Some(other) => {
            return Err(invalid(format!(
                "cache_prompt must be a boolean, not {}",
                other.kind()
            )));
        }
    };
    let tok = &*state.tok;
    let chat = generative::ask(&asked, &mut |text| tok.encode(text))
        .map_err(refusal)?
        .chat_body(cache_prompt);
    let mut plan = chat_plan(state, &chat, None)?;
    let watch = Watch::default();
    plan.p.constraint = Some(
        plan.p
            .constraint
            .take()
            .ok_or_else(|| engine_error(&"the answer schema compiled to no program"))?
            .watched(&watch),
    );
    Ok((asked, plan, watch))
}

/// The request's fault as the decision seats answer it: a 501 for an input a text model cannot
/// read, else a 400 carrying the reader's message.
fn refusal(e: decision::Error) -> ApiError {
    match e {
        decision::Error::NotSupported(m) => ApiError {
            retry_after: false,
            code: 501,
            kind: "not_supported_error",
            message: m,
        },
        e => invalid(e.to_string()),
    }
}

/// The answer body of a finished generation: the watched picks as the questions' decisions, written
/// as the decision seats write theirs, with `timings` last.
fn answered(
    state: &State,
    asked: &Asked,
    watch: &Watch,
    outcome: &Outcome,
) -> Result<String, ApiError> {
    let decisions = decisions(asked, &watch.take())?;
    let answers = generative::answers(asked, &decisions).map_err(|e| match e {
        e @ decision::Error::NonFinite { .. } => engine_error(&e),
        e => engine_error(&format!("the model's decisions cannot be answered: {e}")),
    })?;
    let t = &outcome.timings;
    let body = answer::body(&state.alias, answers, t.n_prompt, t.predicted_n);
    let decided = Decided {
        body: dumps(&body, false),
        prompt_n: t.prompt_n,
        prompt_ms: t.prompt_ms,
        head_ms: t.predicted_ms,
    };
    with_timings(&decided, t.cache_n).map_err(|e| engine_error(&e))
}

/// The decision of each question that needs one, from the picks the walk made in the reply's
/// order: one pick a question, since [`generative::ask`] asks only what one pick tells apart (a
/// choice that one does not is asked by label). A question's pick has the options for children.
fn decisions(asked: &Asked, picks: &[Picked]) -> Result<Vec<Decision>, ApiError> {
    let wanted: Vec<_> = asked
        .questions
        .iter()
        .filter(|q| generative::needs_decision(q))
        .collect();
    let decisions = wanted
        .iter()
        .enumerate()
        .map(|(at, q)| {
            let pick = picks.get(at).ok_or_else(|| {
                engine_error(&format!(
                    "the reply ended after {at} of {} decisions",
                    wanted.len()
                ))
            })?;
            decided(q, pick)
        })
        .collect::<Result<Vec<_>, _>>()?;
    match picks.len() == wanted.len() {
        true => Ok(decisions),
        false => Err(engine_error(&format!(
            "the reply made {} decisions for {} questions",
            picks.len(),
            wanted.len()
        ))),
    }
}

/// `q`'s decision from the pick that told its options apart: each child of the pick leads to one
/// option, so its logit is that option's score. A pick that does not is the server's fault, not
/// the request's ([`generative::ask`] asks only what one pick tells apart).
fn decided(q: &Question, pick: &Picked) -> Result<Decision, ApiError> {
    let options = q.options().len();
    let mut scores = vec![None; options];
    let mut picked = None;
    for kid in &pick.kids {
        if let Some(k) = kid.option.filter(|&k| k < options) {
            scores[k] = Some(kid.logit);
            if kid.id == pick.picked {
                picked = Some(k);
            }
        }
    }
    match (scores.iter().copied().collect::<Option<Vec<f32>>>(), picked) {
        (Some(scores), Some(picked)) if pick.kids.len() == options => {
            Ok(Decision { scores, picked })
        }
        _ => Err(engine_error(&format!(
            "question {}: the pick has {} children for {options} options, and not each leads \
             to one option",
            q.id,
            pick.kids.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use decision::request::Request as Asked;
    use serde_json::{Value, json};

    use super::super::testserve::{mock_config, roundtrip, spawn};
    use super::*;
    use crate::engine::{Engine, EngineError, Tokenizer};
    use crate::mock::{MockTokenizer, SPECIALS};
    use crate::sampling;

    /// The mock's id of a byte: its specials come first.
    fn byte(b: u8) -> u32 {
        u32::try_from(SPECIALS.len()).unwrap() + u32::from(b)
    }

    /// The first byte's id of `text`, the token an option of the byte-level mock starts with.
    fn first(text: &str) -> u32 {
        byte(text.as_bytes()[0])
    }

    /// Raises: after the context ends in the ids of the text, the ids' logits.
    type Raises = Vec<(u32, f32)>;
    type Rule = (String, Raises);

    /// An engine whose row is a floor, raised where the ids fed so far end in a rule's text: the
    /// decisions' rows the tests choose, and nothing else.
    struct Rows {
        tok: Arc<dyn Tokenizer>,
        fed: Vec<u32>,
        rules: Vec<(Vec<u32>, Raises)>,
    }

    impl Rows {
        fn new(rules: &[Rule]) -> Rows {
            Rows {
                tok: Arc::new(MockTokenizer),
                fed: Vec::new(),
                rules: rules
                    .iter()
                    .map(|(text, raise)| (MockTokenizer.encode(text), raise.clone()))
                    .collect(),
            }
        }
    }

    impl Engine for Rows {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            Arc::clone(&self.tok)
        }

        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.fed.extend_from_slice(ids);
            Ok(())
        }

        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.fed.push(last);
            let mut row = vec![-8.0f32; self.tok.n_vocab()];
            for (suffix, raise) in &self.rules {
                if self.fed.ends_with(suffix) {
                    for &(id, v) in raise {
                        row[id as usize] = v;
                    }
                }
            }
            let best = sampling::argmax(&row);
            if let Some(out) = out {
                out.copy_from_slice(&row);
            }
            Ok(best)
        }

        fn reset(&mut self) -> Result<(), EngineError> {
            self.fed.clear();
            Ok(())
        }

        fn ctx_max(&self) -> usize {
            8192
        }

        fn describe(&self) -> String {
            "rows".to_owned()
        }
    }

    const REQUEST: &str = r#"{"model":"clef-flash","state":"My invoice is wrong and the site is down","cache_prompt":false,"questions":{
        "department":{"type":"choice","instructions":"Which team?","criteria":{"technical":"Bugs","billing":"Pay"}},
        "urgency":{"type":"score","criteria":["Can wait","This week","Today"]},
        "outage":{"type":"noul"},
        "region":{"type":"choice","criteria":{"eu":"Europe"}}}}"#;

    /// The decision rows of [`REQUEST`]: the logits the answer test hand-computes from.
    fn rules() -> Vec<Rule> {
        vec![
            (
                "\"department\": \"".into(),
                vec![(first("billing"), 0.0), (first("technical"), 2.0)],
            ),
            (
                "\"urgency\": ".into(),
                vec![(first("0"), 0.0), (first("1"), 1.0), (first("2"), 0.5)],
            ),
            (
                "\"outage\": ".into(),
                vec![(first("true"), 1.0), (first("false"), -1.0)],
            ),
        ]
    }

    /// A server over [`Rows`] and one POST to it: the status and the body.
    fn post_to(rules: &[Rule], path: &str, body: &str) -> (u16, Value) {
        let (addr, _state, _end) = spawn(Box::new(Rows::new(rules)), mock_config());
        let (status, text) = roundtrip(addr, "POST", path, &[], body);
        let v = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
        (status, v)
    }

    /// The keys in order and the kind of each value, a body's shape.
    fn shape(v: &Value) -> String {
        match v {
            Value::Object(m) => format!(
                "{{{}}}",
                m.iter()
                    .map(|(k, v)| format!("{k}:{}", shape(v)))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Value::Array(a) => format!("[{}]", a.iter().map(shape).collect::<Vec<_>>().join(",")),
            Value::String(_) => "str".into(),
            Value::Number(_) => "num".into(),
            Value::Bool(_) => "bool".into(),
            Value::Null => "null".into(),
        }
    }

    /// The decide seat's body for `body` with `logits` a question, as the Clef seat writes it, and
    /// `timings` as its server appends them.
    fn clef_body(body: &str, logits: &[Vec<f32>]) -> Value {
        use decision::encode::{Encoded, EncodedQuestion};
        let req = Asked::from_json(&json::parse(body).unwrap()).unwrap();
        let enc = Encoded {
            ids: vec![0; 3],
            questions: req
                .questions
                .iter()
                .map(|q| EncodedQuestion {
                    id: q.id.clone(),
                    kind: q.kind,
                    question_span: (0, 1),
                    option_spans: Vec::new(),
                    option_ids: q.options().into_iter().map(|o| o.id).collect(),
                })
                .collect(),
            image_spans: Vec::new(),
        };
        let json = answer::answer(&req, &enc, logits, "mock").unwrap();
        let decided = Decided {
            body: dumps(&json, false),
            prompt_n: 3,
            prompt_ms: 1.0,
            head_ms: 1.0,
        };
        serde_json::from_str(&with_timings(&decided, 0).unwrap()).unwrap()
    }

    /// One request per question type (a choice, a score, a noul, and a choice of one option) on
    /// the rows of [`rules`]: the body has the decide seat's keys, types and order; the answers
    /// are the formulas on those rows (the same numbers the decision crate's own body test
    /// computes by hand); the usage and timings are this model's.
    #[test]
    fn each_question_type_is_answered_in_the_decide_seats_body() {
        let (status, got) = post_to(&rules(), "/v1/systemone", REQUEST);
        assert_eq!(status, 200, "{got}");
        let want = clef_body(
            REQUEST,
            &[
                vec![0.0, 2.0],
                vec![0.0, 1.0, 0.5],
                vec![1.0, -1.0],
                vec![0.0],
            ],
        );
        assert_eq!(shape(&got), shape(&want));
        assert_eq!(
            got["answers"].to_string(),
            concat!(
                r#"{"department":{"type":"choice","choice":"technical","confidence":0.7616,"#,
                r#""probabilities":{"technical":0.8808,"billing":0.1192}},"#,
                r#""urgency":{"type":"score","score":1.1209,"confidence":0.2597,"#,
                r#""legend":{"0":"Can wait","1":"This week","2":"Today"},"#,
                r#""probabilities":{"0":0.1863,"1":0.5065,"2":0.3072}},"#,
                r#""outage":{"type":"noul","noul":0.8808},"#,
                r#""region":{"type":"choice","choice":"eu","confidence":1.0,"probabilities":{"eu":1.0}}}"#,
            )
        );
        assert_eq!(got["answers"], want["answers"]);
        assert_eq!(got["model"], "mock");
        let (usage, t) = (&got["usage"], &got["timings"]);
        assert!(usage["input_tokens"].as_u64().unwrap() > 100);
        assert!(usage["output_tokens"].as_u64().unwrap() > 20);
        assert_eq!(t["cache_n"], 0);
        assert_eq!(
            usage["input_tokens"].as_u64().unwrap(),
            t["prompt_n"].as_u64().unwrap()
        );
    }

    /// The picks are the picks `response_format` makes of the converted chat on the same rows: the
    /// choice is the string it writes, the score the integer, the noul the boolean.
    #[test]
    fn the_picks_are_the_response_formats_picks() {
        let (_, got) = post_to(&rules(), "/v1/systemone", REQUEST);
        let req = Asked::from_json(&json::parse(REQUEST).unwrap()).unwrap();
        let chat = generative::ask(&req, &mut |t| MockTokenizer.encode(t))
            .unwrap()
            .chat_body(false);
        let (status, reply) = post_to(
            &rules(),
            "/v1/chat/completions",
            &Value::Object(chat).to_string(),
        );
        assert_eq!(status, 200, "{reply}");
        let said: Value =
            serde_json::from_str(reply["choices"][0]["message"]["content"].as_str().unwrap())
                .unwrap();
        assert_eq!(
            said,
            json!({"department":"technical","urgency":1,"outage":true,"region":"eu"})
        );
        // Both ran the jump arm: the forced text between the picks was fed as calls.
        assert_eq!(reply["timings"]["constrained"]["arm"], "jump", "{reply}");
        assert!(
            reply["timings"]["constrained"]["span_ids"]
                .as_u64()
                .unwrap()
                > 20
        );
        let a = &got["answers"];
        assert_eq!(said["department"], a["department"]["choice"]);
        let level = a["urgency"]["probabilities"]
            .as_object()
            .unwrap()
            .iter()
            .max_by(|x, y| x.1.as_f64().partial_cmp(&y.1.as_f64()).unwrap())
            .map(|(k, _)| k.parse::<u64>().unwrap());
        assert_eq!(said["urgency"].as_u64(), level);
        assert_eq!(
            said["outage"].as_bool(),
            Some(a["outage"]["noul"].as_f64().unwrap() > 0.5)
        );
        // The other way: rows that favour the other options give the other picks, in both.
        let flipped: Vec<Rule> = vec![
            (
                "\"department\": \"".into(),
                vec![(first("billing"), 3.0), (first("technical"), 0.0)],
            ),
            (
                "\"urgency\": ".into(),
                vec![(first("0"), 2.0), (first("1"), 0.0), (first("2"), 1.0)],
            ),
            (
                "\"outage\": ".into(),
                vec![(first("true"), -1.0), (first("false"), 1.0)],
            ),
        ];
        let (_, got) = post_to(&flipped, "/v1/systemone", REQUEST);
        let chat = generative::ask(&req, &mut |t| MockTokenizer.encode(t))
            .unwrap()
            .chat_body(false);
        let (_, reply) = post_to(
            &flipped,
            "/v1/chat/completions",
            &Value::Object(chat).to_string(),
        );
        let said: Value =
            serde_json::from_str(reply["choices"][0]["message"]["content"].as_str().unwrap())
                .unwrap();
        assert_eq!(
            said,
            json!({"department":"billing","urgency":0,"outage":false,"region":"eu"})
        );
        assert_eq!(got["answers"]["department"]["choice"], "billing");
        assert!(got["answers"]["outage"]["noul"].as_f64().unwrap() < 0.5);
        assert!(got["answers"]["urgency"]["score"].as_f64().unwrap() < 1.0);
    }

    /// A Korean request: ids, instructions, state and descriptions in Hangul, which the schema, the
    /// reply's keys and the body's keys carry unescaped.
    #[test]
    fn a_korean_request_is_answered() {
        let ko = r#"{"model":"clef-flash","state":"결제는 됐는데 주문 내역에 안 보여요.","cache_prompt":false,"questions":{
            "부서":{"type":"choice","instructions":"어느 팀이 맡아야 하나요?","criteria":{"결제":"결제, 환불","배송":"배송과 택배","취소":"주문 취소"}},
            "긴급도":{"type":"score","instructions":"얼마나 급한가요?","criteria":["나중에","오늘 중","지금 당장"]},
            "환불요청":{"type":"noul","instructions":"고객이 환불을 요구하나요?"}}}"#;
        let rules: Vec<Rule> = vec![
            (
                "\"부서\": \"".into(),
                vec![
                    (first("결제"), 0.0),
                    (first("배송"), 0.5),
                    (first("취소"), 1.5),
                ],
            ),
            (
                "\"긴급도\": ".into(),
                vec![(first("0"), 0.0), (first("1"), 2.0), (first("2"), 1.0)],
            ),
            (
                "\"환불요청\": ".into(),
                vec![(first("true"), 0.0), (first("false"), 0.0)],
            ),
        ];
        let (status, got) = post_to(&rules, "/v1/systemone", ko);
        assert_eq!(status, 200, "{got}");
        let want = clef_body(
            ko,
            &[vec![0.0, 0.5, 1.5], vec![0.0, 2.0, 1.0], vec![0.0, 0.0]],
        );
        assert_eq!(shape(&got), shape(&want));
        assert_eq!(got["answers"], want["answers"]);
        assert_eq!(got["answers"]["부서"]["choice"], "취소");
        assert_eq!(
            got["answers"]["부서"]["probabilities"]
                .as_object()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(got["answers"]["환불요청"]["noul"], 0.5);
    }

    /// An invalid body gets the decision seats' refusal: the status and the message of the
    /// decision crate's request reader, whichever model answers.
    #[test]
    fn an_invalid_body_is_refused_as_the_decide_seat_refuses() {
        for body in [
            r#"[1]"#,
            r#"{"questions":{"q":{"type":"noul"}}}"#,
            r#"{"state":"s","questions":{}}"#,
            r#"{"state":"s","questions":{"q":{"type":"rank"}}}"#,
            r#"{"state":"s","questions":{"q":{"type":"choice","criteria":{}}}}"#,
            r#"{"state":"s","questions":{"q":{"type":"score","criteria":{"a":1}}}}"#,
            r#"{"state":"s","videos":["x"],"questions":{"q":{"type":"noul"}}}"#,
            r#"{"state":NaN,"questions":{"q":{"type":"noul"}}}"#,
            "{",
        ] {
            let Err(want) = json::parse(body).and_then(|j| Asked::from_json(&j)) else {
                panic!("{body} is valid");
            };
            let (status, got) = post_to(&rules(), "/v1/systemone", body);
            let (code, kind) = match want {
                decision::Error::NotSupported(_) => (501, "not_supported_error"),
                _ => (400, "invalid_request_error"),
            };
            assert_eq!(status, code, "{body}: {got}");
            assert_eq!(got["error"]["type"], kind, "{body}");
            assert_eq!(got["error"]["message"], want.to_string(), "{body}");
        }
        let (status, got) = post_to(
            &rules(),
            "/v1/systemone",
            r#"{"state":"s","images":["data:image/png;base64,AAAA"],"questions":{"q":{"type":"noul"}}}"#,
        );
        assert_eq!(status, 501, "{got}");
        assert_eq!(got["error"]["message"], decision::encode::NO_IMAGE_INPUT);
        let (status, got) = post_to(
            &rules(),
            "/v1/systemone",
            r#"{"state":"s","cache_prompt":"no","questions":{"q":{"type":"noul"}}}"#,
        );
        assert_eq!(status, 400, "{got}");
        assert!(
            got["error"]["message"]
                .as_str()
                .unwrap()
                .contains("cache_prompt")
        );
    }

    /// A choice with options that start with the same token is asked by label and answered by
    /// name: `ab`, `ac` and `d` are the labels `A`, `B`, `C` to the model, the label row's logits
    /// are their scores, and the body is the decide seat's for those scores, with the options'
    /// ids as `choice` and as the keys of `probabilities`, and no label in it.
    #[test]
    fn options_that_start_alike_are_answered_by_name() {
        let body = r#"{"state":"s","cache_prompt":false,"questions":{
            "q":{"type":"choice","criteria":{"d":"Dee","ab":"Ab","ac":"Ac"}},
            "urgency":{"type":"score","criteria":["Can wait","Today"]}}}"#;
        let rows: Vec<Rule> = vec![
            (
                "\"q\": \"".into(),
                vec![(first("A"), 0.5), (first("B"), 2.0), (first("C"), 0.0)],
            ),
            (
                "\"urgency\": ".into(),
                vec![(first("0"), 1.0), (first("1"), 0.0)],
            ),
        ];
        let (status, got) = post_to(&rows, "/v1/systemone", body);
        assert_eq!(status, 200, "{got}");
        // Scores in the sorted option order (ab, ac, d); keys in the request's criteria order.
        let want = clef_body(body, &[vec![0.5, 2.0, 0.0], vec![1.0, 0.0]]);
        assert_eq!(shape(&got), shape(&want));
        assert_eq!(got["answers"], want["answers"]);
        let q = &got["answers"]["q"];
        assert_eq!(q["choice"], "ac");
        let keys: Vec<&str> = q["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["d", "ab", "ac"]);
        let p = decision::answer::probabilities(&[0.5, 2.0, 0.0]);
        for (i, id) in ["ab", "ac", "d"].iter().enumerate() {
            let x = q["probabilities"][id].as_f64().unwrap();
            assert!(
                (x - f64::from(p[i])).abs() < 1e-4,
                "{id}: {x} against {}",
                p[i]
            );
        }
        let text = got.to_string();
        for label in ["\"A\"", "\"B\"", "\"C\""] {
            assert!(!text.contains(label), "{text}");
        }
        assert_eq!(
            got["answers"]["urgency"]["score"],
            want["answers"]["urgency"]["score"]
        );
    }

    /// The label is the route's conversion only: the converted chat body on `/v1/chat/completions`
    /// writes the label the walk picked, which is the option the answer names.
    #[test]
    fn the_label_is_the_chat_bodys_pick() {
        let body = r#"{"state":"s","cache_prompt":false,"questions":{
            "q":{"type":"choice","criteria":{"ab":"Ab","ac":"Ac","d":"Dee"}}}}"#;
        let rows: Vec<Rule> = vec![(
            "\"q\": \"".into(),
            vec![(first("A"), 0.0), (first("B"), 1.0), (first("C"), 3.0)],
        )];
        let (_, got) = post_to(&rows, "/v1/systemone", body);
        let req = Asked::from_json(&json::parse(body).unwrap()).unwrap();
        let ask = generative::ask(&req, &mut |t| MockTokenizer.encode(t)).unwrap();
        let (status, reply) = post_to(
            &rows,
            "/v1/chat/completions",
            &Value::Object(ask.chat_body(false)).to_string(),
        );
        assert_eq!(status, 200, "{reply}");
        let said: Value =
            serde_json::from_str(reply["choices"][0]["message"]["content"].as_str().unwrap())
                .unwrap();
        assert_eq!(said, json!({"q": "C"}));
        let label = ask.labels[0].as_ref().unwrap();
        let at = label.iter().position(|l| l == "C").unwrap();
        assert_eq!(["ab", "ac", "d"][at], got["answers"]["q"]["choice"]);
        assert_eq!(got["answers"]["q"]["choice"], "d");
    }

    /// The letters hold 26 labels: 26 options that start alike are answered, a 27th has none, and
    /// that request is refused by name, the question and the counts in the message.
    #[test]
    fn more_options_than_labels_is_refused_by_name() {
        let request = |n: usize| {
            let options: Vec<String> = (0..n).map(|i| format!("\"p{i:02}\":\"d\"")).collect();
            format!(
                r#"{{"state":"s","cache_prompt":false,"questions":{{"q":{{"type":"choice","criteria":{{{}}}}}}}}}"#,
                options.join(",")
            )
        };
        let (status, got) = post_to(&[], "/v1/systemone", &request(26));
        assert_eq!(status, 200, "{got}");
        assert_eq!(
            got["answers"]["q"]["probabilities"]
                .as_object()
                .unwrap()
                .len(),
            26
        );
        let (status, got) = post_to(&[], "/v1/systemone", &request(27));
        assert_eq!(status, 400, "{got}");
        assert_eq!(got["error"]["type"], "invalid_request_error");
        let m = got["error"]["message"].as_str().unwrap();
        assert!(
            m.starts_with("q: ")
                && m.contains("27 options")
                && m.contains("26 single-token labels"),
            "{m}"
        );
    }

    /// A pick whose children do not each lead to one option is the server's fault, a 500 naming
    /// the question, never a tie it made up.
    #[test]
    fn a_pick_that_does_not_separate_the_options_is_a_server_error() {
        let req = Asked::from_json(
            &json::parse(
                r#"{"state":"s","questions":{"q":{"type":"choice","criteria":{"ab":"x","ac":"y","d":"z"}}}}"#,
            )
            .unwrap(),
        )
        .unwrap();
        let kid = |id, option| crate::constrain::Kid {
            id,
            logit: 0.0,
            option,
        };
        let pick = Picked {
            kids: vec![kid(1, None), kid(2, Some(2))],
            picked: 1,
        };
        let e = decided(&req.questions[0], &pick).unwrap_err();
        assert_eq!(e.code, 500);
        assert!(e.message.contains("question q"), "{}", e.message);
    }

    /// Options that share a prefix are scored where they differ: the walk takes the shared ids
    /// as forced text and the row at the first token that tells them apart is the evidence.
    #[test]
    fn options_with_a_shared_prefix_are_scored_where_they_differ() {
        let body = r#"{"state":"s","cache_prompt":false,"questions":{
            "q":{"type":"choice","criteria":{"refund_part":"Some","refund_full":"All"}}}}"#;
        let rows: Vec<Rule> = vec![(
            "\"q\": \"refund_".into(),
            vec![(first("full"), 2.0), (first("part"), 0.0)],
        )];
        let (status, got) = post_to(&rows, "/v1/systemone", body);
        assert_eq!(status, 200, "{got}");
        assert_eq!(
            got["answers"].to_string(),
            r#"{"q":{"type":"choice","choice":"refund_full","confidence":0.7616,"probabilities":{"refund_part":0.1192,"refund_full":0.8808}}}"#
        );
    }

    /// A score that is not finite is a named error, never a probability.
    #[test]
    fn a_score_that_is_not_finite_is_a_named_error() {
        let rows: Vec<Rule> = vec![(
            "\"outage\": ".into(),
            vec![(first("true"), f32::NAN), (first("false"), 1.0)],
        )];
        let body = r#"{"state":"s","questions":{"outage":{"type":"noul"}}}"#;
        let (status, got) = post_to(&rows, "/v1/systemone", body);
        assert_eq!(status, 500, "{got}");
        assert_eq!(got["error"]["type"], "server_error");
        assert!(
            got["error"]["message"]
                .as_str()
                .unwrap()
                .contains("outage/true: the logit is not finite"),
            "{got}"
        );
    }

    /// `/props` and the not-found text name the route.
    #[test]
    fn props_and_the_not_found_text_name_the_route() {
        let (addr, _state, _end) = spawn(Box::new(Rows::new(&[])), mock_config());
        let (status, props) = roundtrip(addr, "GET", "/props", &[], "");
        assert_eq!(status, 200);
        let props: Value = serde_json::from_str(&props).unwrap();
        assert_eq!(props["routes"], json!(["/v1/systemone"]));
        let (status, nf) = roundtrip(addr, "POST", "/v1/nothing", &[], "{}");
        assert_eq!(status, 404);
        let nf: Value = serde_json::from_str(&nf).unwrap();
        assert!(
            nf["error"]["message"]
                .as_str()
                .unwrap()
                .contains("POST /v1/systemone")
        );
    }
}
