//! A SystemOne `/v1/systemone` request: its validation and each question's allowed options.
//!
//! The refusals are the release's `systemone` checks in its order, then the ones its encoder meets
//! by raising (a criteria value of the wrong shape), then what this engine does not run (media).
//! Unknown top-level keys (a client's `id`) are ignored, as the release ignores them.

use crate::Error;
use crate::json::Json;

/// A question's type, with the release's numbering (`QUESTION_TYPES`): the type embedding's row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Noul = 0,
    Choice = 1,
    Score = 2,
}

impl Kind {
    /// The type's request word.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            Kind::Noul => "noul",
            Kind::Choice => "choice",
            Kind::Score => "score",
        }
    }

    /// The type embedding row.
    #[must_use]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// One question of a request.
#[derive(Clone, Debug)]
pub struct Question {
    pub id: String,
    pub kind: Kind,
    /// `instructions` as sent (`None` when absent or null).
    pub instructions: Option<Json>,
    /// `criteria` as sent (`None` when absent).
    pub criteria: Option<Json>,
}

/// A validated request.
#[derive(Clone, Debug)]
pub struct Request {
    pub model: String,
    pub state: Json,
    /// In request order: the order of the encoder's fields and of the answers.
    pub questions: Vec<Question>,
}

/// One allowed option: its id and its description (`None` renders no description key).
#[derive(Clone, Debug, PartialEq)]
pub struct Opt {
    pub id: String,
    pub description: Option<Json>,
}

const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE: &str = "The proposition is false or the answer is no.";

impl Request {
    /// Validate a parsed request body.
    pub fn from_json(body: &Json) -> Result<Request, Error> {
        let Json::Object(_) = body else {
            return Err(Error::Request(format!(
                "the body is {}, not an object",
                body.kind()
            )));
        };
        let model = body.get("model").and_then(Json::as_str);
        let (Some(model), Some(state)) = (model, body.get("state")) else {
            return Err(Error::Request("model and state are required".into()));
        };
        let questions = match body.get("questions") {
            Some(Json::Object(q)) if !q.is_empty() => q,
            _ => return Err(Error::Request("at least one question is required".into())),
        };
        for key in ["images", "videos"] {
            if body.get(key).is_some_and(Json::truthy) {
                return Err(Error::Media(key));
            }
        }
        let mut out = Vec::with_capacity(questions.len());
        for (id, q) in questions {
            let Json::Object(_) = q else {
                return Err(Error::Question {
                    id: id.clone(),
                    what: format!("the question is {}, not an object", q.kind()),
                });
            };
            let kind = match q.get("type").and_then(Json::as_str) {
                Some("noul") => Kind::Noul,
                Some("choice") => Kind::Choice,
                Some("score") => Kind::Score,
                _ => {
                    return Err(Error::Question {
                        id: id.clone(),
                        what: "type must be noul, choice, or score".into(),
                    });
                }
            };
            let criteria = q.get("criteria").cloned();
            if kind != Kind::Noul && !criteria.as_ref().is_some_and(Json::truthy) {
                return Err(Error::Question {
                    id: id.clone(),
                    what: "criteria must not be empty".into(),
                });
            }
            let shape_ok = match (kind, &criteria) {
                (Kind::Choice, Some(Json::Object(_))) | (Kind::Score, Some(Json::Array(_))) => true,
                (Kind::Noul, None | Some(Json::Object(_))) => true,
                (Kind::Noul, Some(c)) => !c.truthy(),
                _ => false,
            };
            if !shape_ok {
                let want = match kind {
                    Kind::Noul => "an object (overriding true or false), or empty",
                    Kind::Choice => "an object of option id to description",
                    Kind::Score => "an array of level descriptions",
                };
                return Err(Error::Question {
                    id: id.clone(),
                    what: format!(
                        "{} criteria must be {want}, not {}",
                        kind.word(),
                        criteria.as_ref().map_or("absent", Json::kind)
                    ),
                });
            }
            out.push(Question {
                id: id.clone(),
                kind,
                instructions: q.get("instructions").filter(|v| **v != Json::Null).cloned(),
                criteria,
            });
        }
        Ok(Request {
            model: model.to_string(),
            state: state.clone(),
            questions: out,
        })
    }
}

impl Question {
    /// The release's `question_options`: `noul` takes `true` then `false` (defaults, overridden by
    /// the criteria), `choice` its criteria sorted by key (code point order), `score` its levels by
    /// index.
    #[must_use]
    pub fn options(&self) -> Vec<Opt> {
        match (self.kind, &self.criteria) {
            (Kind::Noul, c) => {
                let over = |key: &str| match c {
                    Some(Json::Object(pairs)) => {
                        pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
                    }
                    _ => None,
                };
                [("true", NOUL_TRUE), ("false", NOUL_FALSE)]
                    .into_iter()
                    .map(|(key, default)| Opt {
                        id: key.to_string(),
                        description: match over(key) {
                            Some(Json::Null) => None,
                            Some(v) => Some(v.clone()),
                            None => Some(Json::Str(default.to_string())),
                        },
                    })
                    .collect()
            }
            (Kind::Choice, Some(Json::Object(pairs))) => {
                let mut opts: Vec<Opt> = pairs
                    .iter()
                    .map(|(k, v)| Opt {
                        id: k.clone(),
                        description: (*v != Json::Null).then(|| v.clone()),
                    })
                    .collect();
                opts.sort_by(|a, b| a.id.cmp(&b.id));
                opts
            }
            (Kind::Score, Some(Json::Array(levels))) => levels
                .iter()
                .enumerate()
                .map(|(i, v)| Opt {
                    id: i.to_string(),
                    description: (*v != Json::Null).then(|| v.clone()),
                })
                .collect(),
            _ => unreachable!("Request::from_json admits only these criteria shapes"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    fn req(text: &str) -> Result<Request, Error> {
        Request::from_json(&parse(text).unwrap())
    }

    fn ids(q: &Question) -> Vec<String> {
        q.options().into_iter().map(|o| o.id).collect()
    }

    #[test]
    fn choice_options_sort_by_code_point() {
        let r = req(
            r#"{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":
            {"zeta":"P","alpha":"Pa","Mid":"Mo","beta":"B","_misc":"N","é":"E"}}}}"#,
        )
        .unwrap();
        assert_eq!(
            ids(&r.questions[0]),
            ["Mid", "_misc", "alpha", "beta", "zeta", "é"]
        );
    }

    #[test]
    fn noul_defaults_and_overrides() {
        let r = req(r#"{"model":"m","state":1,"questions":{
            "a":{"type":"noul"},
            "b":{"type":"noul","criteria":{"true":"Yes, approved.","extra":"ignored"}},
            "c":{"type":"noul","criteria":{"false":null}},
            "d":{"type":"noul","criteria":[]}}}"#)
        .unwrap();
        let desc = |q: usize| -> Vec<Option<Json>> {
            r.questions[q]
                .options()
                .into_iter()
                .map(|o| o.description)
                .collect()
        };
        let s = |t: &str| Some(Json::Str(t.into()));
        assert_eq!(ids(&r.questions[0]), ["true", "false"]);
        assert_eq!(desc(0), [s(NOUL_TRUE), s(NOUL_FALSE)]);
        assert_eq!(desc(1), [s("Yes, approved."), s(NOUL_FALSE)]);
        assert_eq!(desc(2), [s(NOUL_TRUE), None]);
        assert_eq!(desc(3), [s(NOUL_TRUE), s(NOUL_FALSE)]);
    }

    #[test]
    fn score_levels_are_indexed() {
        let r = req(r#"{"model":"m","state":1,"questions":{"q":{"type":"score","criteria":["Low",null,"High"]}}}"#).unwrap();
        let o = r.questions[0].options();
        assert_eq!(
            o.iter().map(|o| o.id.as_str()).collect::<Vec<_>>(),
            ["0", "1", "2"]
        );
        assert_eq!(o[1].description, None);
    }

    #[test]
    fn question_order_is_request_order_and_unknown_keys_pass() {
        let r = req(r#"{"id":"x","model":"m","state":null,"questions":{"z":{"type":"noul"},"a":{"type":"noul"}}}"#).unwrap();
        assert_eq!(
            r.questions
                .iter()
                .map(|q| q.id.as_str())
                .collect::<Vec<_>>(),
            ["z", "a"]
        );
        assert_eq!(r.state, Json::Null);
    }

    #[test]
    fn refusals_are_named() {
        let cases = [
            (r#"[1]"#, "not an object"),
            (
                r#"{"state":1,"questions":{"q":{"type":"noul"}}}"#,
                "model and state are required",
            ),
            (
                r#"{"model":3,"state":1,"questions":{"q":{"type":"noul"}}}"#,
                "model and state are required",
            ),
            (
                r#"{"model":"m","questions":{"q":{"type":"noul"}}}"#,
                "model and state are required",
            ),
            (
                r#"{"model":"m","state":1}"#,
                "at least one question is required",
            ),
            (
                r#"{"model":"m","state":1,"questions":{}}"#,
                "at least one question is required",
            ),
            (
                r#"{"model":"m","state":1,"questions":[]}"#,
                "at least one question is required",
            ),
            (
                r#"{"model":"m","state":1,"questions":{"q":{"type":"bool"}}}"#,
                "q: type must be noul, choice, or score",
            ),
            (
                r#"{"model":"m","state":1,"questions":{"q":"noul"}}"#,
                "q: the question is a string, not an object",
            ),
            (
                r#"{"model":"m","state":1,"questions":{"q":{"type":"choice"}}}"#,
                "q: criteria must not be empty",
            ),
            (
                r#"{"model":"m","state":1,"questions":{"q":{"type":"score","criteria":[]}}}"#,
                "q: criteria must not be empty",
            ),
            (
                r#"{"model":"m","state":1,"questions":{"q":{"type":"choice","criteria":["a"]}}}"#,
                "q: choice criteria must be an object",
            ),
            (
                r#"{"model":"m","state":1,"questions":{"q":{"type":"score","criteria":{"a":1}}}}"#,
                "q: score criteria must be an array",
            ),
            (
                r#"{"model":"m","state":1,"questions":{"q":{"type":"noul","criteria":"yes"}}}"#,
                "q: noul criteria must be an object",
            ),
            (
                r#"{"model":"m","state":1,"images":["x.png"],"questions":{"q":{"type":"noul"}}}"#,
                "images",
            ),
        ];
        for (text, want) in cases {
            let e = req(text).expect_err(text).to_string();
            assert!(e.contains(want), "{text}: {e}");
        }
        assert!(req(r#"{"model":"m","state":1,"images":[],"videos":null,"questions":{"q":{"type":"noul"}}}"#).is_ok());
    }
}
