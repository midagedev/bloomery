//! A SystemOne `/v1/systemone` request: its validation and each question's allowed options.
//!
//! The refusals are the release's `systemone` checks in its order, then the ones its encoder meets
//! by raising (a criteria value of the wrong shape), then the media checks of llama.cpp's server
//! (`tools/server/server-decision.cpp` at `a4cb4c61`, `parse_state`): `images` and every `image_url`
//! part of a chat-messages state must be data URLs, at most [`MAX_IMAGES`] in all, and a request
//! that carries one, or `videos`, is one this text-only engine does not answer
//! ([`Error::NotSupported`]). `model` is read by no one: a server of one model answers whatever a
//! request names, as llama.cpp's does. Unknown top-level keys (a client's `id`) are ignored, as the
//! release ignores them.

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

/// The images a request may carry, `images` and a state's image parts together: llama.cpp's
/// `DECISION_MAX_IMAGES`.
pub const MAX_IMAGES: usize = 8;

const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE: &str = "The proposition is false or the answer is no.";

/// What a decision model's wire does to a request, beside the checks every model shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    /// A choice's options are sorted by key (code point order); else they keep the criteria's order.
    pub choice_sorted: bool,
    /// A noul's options are `true` then `false`; else `false` then `true`.
    pub noul_true_first: bool,
    /// A noul option the criteria do not describe has the release's sentence; else no description.
    pub noul_defaults: bool,
    /// The checks llama.cpp's server makes (`parse_questions`) that the release's Python does not:
    /// a `state` that is not null, every question's `instructions` given, a score's 2 to 10 levels, a
    /// noul's `criteria` absent, null or an object.
    pub strict: bool,
}

impl Rules {
    /// The release's, which Clef's seat follows: choice sorted, `true` first, the release's
    /// sentences, and only the release's checks.
    pub const CLEF: Rules = Rules {
        choice_sorted: true,
        noul_true_first: true,
        noul_defaults: true,
        strict: false,
    };
}

/// A score's number of levels, at least and at most, as llama.cpp's server accepts it.
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;

impl Request {
    /// Validate a parsed request body under the release's rules ([`Rules::CLEF`]).
    pub fn from_json(body: &Json) -> Result<Request, Error> {
        Request::from_json_with(body, &Rules::CLEF)
    }

    /// Validate a parsed request body under `rules`.
    pub fn from_json_with(body: &Json, rules: &Rules) -> Result<Request, Error> {
        let Json::Object(_) = body else {
            return Err(Error::Request(format!(
                "the body is {}, not an object",
                body.kind()
            )));
        };
        let Some(state) = body.get("state") else {
            return Err(Error::Request("state is required".into()));
        };
        if rules.strict && *state == Json::Null {
            return Err(Error::Request("state is required, and is not null".into()));
        }
        let questions = match body.get("questions") {
            Some(Json::Object(q)) if !q.is_empty() => q,
            _ => return Err(Error::Request("at least one question is required".into())),
        };
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
            if rules.strict {
                strict_checks(id, kind, q.get("instructions"), criteria.as_ref())?;
            }
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
        let images = image_count(body, state)?;
        if images > 0 {
            return Err(Error::NotSupported(format!(
                "the request carries {images} image(s); this server does not support image input \
                 for decisions (the backbone reads text only)"
            )));
        }
        if body.get("videos").is_some_and(Json::truthy) {
            return Err(Error::NotSupported(
                "the request carries videos; this server does not support video input for decisions"
                    .into(),
            ));
        }
        Ok(Request {
            state: state.clone(),
            questions: out,
        })
    }
}

/// [`Rules::strict`]'s checks on one question.
fn strict_checks(
    id: &str,
    kind: Kind,
    instructions: Option<&Json>,
    criteria: Option<&Json>,
) -> Result<(), Error> {
    let refuse = |what: String| Error::Question {
        id: id.to_owned(),
        what,
    };
    if instructions.is_none_or(|v| *v == Json::Null) {
        return Err(refuse("\"instructions\" must be provided".into()));
    }
    match (kind, criteria) {
        (Kind::Score, Some(Json::Array(levels))) if !SCORE_LEVELS.contains(&levels.len()) => {
            Err(refuse(format!(
                "\"criteria\" must be an array of {} to {} levels, not {}",
                SCORE_LEVELS.start(),
                SCORE_LEVELS.end(),
                levels.len()
            )))
        }
        (Kind::Noul, Some(c)) if !matches!(c, Json::Null | Json::Object(_)) => {
            Err(refuse(format!(
                "noul criteria must be an object, or absent, not {}",
                c.kind()
            )))
        }
        _ => Ok(()),
    }
}

/// The images of a request, as llama.cpp's server counts them: `images` (null, or an array) and
/// the `image_url` parts of a chat-messages `state` (an array of messages, or an object's
/// `messages`), each a data URL (`data:image/…`), at most [`MAX_IMAGES`]. A malformed image is the
/// request's error, named.
fn image_count(body: &Json, state: &Json) -> Result<usize, Error> {
    let mut n = 0usize;
    let mut load = |url: &Json| -> Result<(), Error> {
        if !url.as_str().is_some_and(|u| u.starts_with("data:image/")) {
            return Err(Error::Request(
                "images must be data URLs (data:image/...;base64,...)".into(),
            ));
        }
        if n >= MAX_IMAGES {
            return Err(Error::Request(format!(
                "too many images, the maximum is {MAX_IMAGES}"
            )));
        }
        n += 1;
        Ok(())
    };
    match body.get("images") {
        None | Some(Json::Null) => {}
        Some(Json::Array(urls)) => {
            for url in urls {
                load(url)?;
            }
        }
        Some(_) => return Err(Error::Request("\"images\" must be an array".into())),
    }
    let messages = match state {
        Json::Object(_) => state.get("messages"),
        _ => Some(state),
    };
    if let Some(Json::Array(messages)) = messages {
        for msg in messages {
            let Some(Json::Array(parts)) = msg.get("content") else {
                continue;
            };
            for part in parts {
                if part.get("type").and_then(Json::as_str) != Some("image_url") {
                    continue;
                }
                let Some(image) = part.get("image_url") else {
                    continue;
                };
                load(image.get("url").unwrap_or(image))?;
            }
        }
    }
    Ok(n)
}

impl Question {
    /// The release's `question_options` ([`Rules::CLEF`]): `noul` takes `true` then `false`
    /// (defaults, overridden by the criteria), `choice` its criteria sorted by key (code point
    /// order), `score` its levels by index.
    #[must_use]
    pub fn options(&self) -> Vec<Opt> {
        self.options_with(&Rules::CLEF)
    }

    /// The options in the order `rules` gives them: a `noul`'s two (described by the criteria, else
    /// by the release's sentence or none), a `choice`'s criteria (sorted by key, or as sent), a
    /// `score`'s levels by index.
    #[must_use]
    pub fn options_with(&self, rules: &Rules) -> Vec<Opt> {
        match (self.kind, &self.criteria) {
            (Kind::Noul, c) => {
                let over = |key: &str| match c {
                    Some(Json::Object(pairs)) => {
                        pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
                    }
                    _ => None,
                };
                let both = if rules.noul_true_first {
                    [("true", NOUL_TRUE), ("false", NOUL_FALSE)]
                } else {
                    [("false", NOUL_FALSE), ("true", NOUL_TRUE)]
                };
                both.into_iter()
                    .map(|(key, default)| Opt {
                        id: key.to_string(),
                        description: match over(key) {
                            Some(Json::Null) => None,
                            Some(v) => Some(v.clone()),
                            None => rules.noul_defaults.then(|| Json::Str(default.to_string())),
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
                if rules.choice_sorted {
                    opts.sort_by(|a, b| a.id.cmp(&b.id));
                }
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
                r#"{"model":"m","questions":{"q":{"type":"noul"}}}"#,
                "state is required",
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
        ];
        for (text, want) in cases {
            let e = req(text).expect_err(text).to_string();
            assert!(e.contains(want), "{text}: {e}");
        }
        // `model` is read by no one: absent, of another type, or naming another model.
        for text in [
            r#"{"state":1,"questions":{"q":{"type":"noul"}}}"#,
            r#"{"model":3,"state":1,"questions":{"q":{"type":"noul"}}}"#,
            r#"{"model":"other","state":1,"images":[],"videos":null,"questions":{"q":{"type":"noul"}}}"#,
        ] {
            assert!(req(text).is_ok(), "{text}");
        }
    }

    /// llama.cpp's media rules: a malformed image is the request's error (400), a well-formed one,
    /// in `images` or in a chat-messages state, and `videos` are not supported (501).
    #[test]
    fn media_is_checked_then_not_supported() {
        let png = "data:image/png;base64,AA==";
        let q = r#""questions":{"q":{"type":"noul"}}"#;
        let nine = vec![format!("{png:?}"); 9].join(",");
        for (text, want) in [
            (
                format!(r#"{{"state":1,"images":"x",{q}}}"#),
                "must be an array",
            ),
            (
                format!(r#"{{"state":1,"images":["x.png"],{q}}}"#),
                "must be data URLs",
            ),
            (
                format!(r#"{{"state":1,"images":[{nine}],{q}}}"#),
                "too many images, the maximum is 8",
            ),
            (
                format!(
                    r#"{{"state":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"http://x/a.png"}}}}]}}],{q}}}"#
                ),
                "must be data URLs",
            ),
        ] {
            match req(&text) {
                Err(Error::Request(e)) => assert!(e.contains(want), "{text}: {e}"),
                other => panic!("{text}: {other:?}"),
            }
        }
        for text in [
            format!(r#"{{"state":1,"images":[{png:?}],{q}}}"#),
            format!(
                r#"{{"state":[{{"role":"user","content":[{{"type":"text","text":"hi"}},{{"type":"image_url","image_url":{{"url":{png:?}}}}}]}}],{q}}}"#
            ),
            format!(
                r#"{{"state":{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{png:?}}}]}}]}},{q}}}"#
            ),
            format!(r#"{{"state":1,"videos":["v.mp4"],{q}}}"#),
        ] {
            assert!(matches!(req(&text), Err(Error::NotSupported(_))), "{text}");
        }
        // A text-only chat state is text.
        assert!(
            req(&format!(
                r#"{{"state":[{{"role":"user","content":[{{"type":"text","text":"hi"}}]}}],{q}}}"#
            ))
            .is_ok()
        );
    }
    /// llama.cpp's other models: criteria order, `false` first, no default sentences, strict.
    const SENT: Rules = Rules {
        choice_sorted: false,
        noul_true_first: false,
        noul_defaults: false,
        strict: true,
    };

    /// The rules order a question's options and describe a noul's: the release's sorted choice and
    /// `true` first with its sentences, llama.cpp's request order and `false` first with none.
    #[test]
    fn rules_order_the_options_and_describe_the_noul() {
        let r = req(r#"{"state":1,"questions":{
            "c":{"type":"choice","criteria":{"zeta":"Z","alpha":null,"Mid":"M"}},
            "n":{"type":"noul","criteria":{"true":"Yes."}},
            "m":{"type":"noul"}}}"#)
        .unwrap();
        let seen = |q: usize, rules: &Rules| -> Vec<(String, Option<Json>)> {
            r.questions[q]
                .options_with(rules)
                .into_iter()
                .map(|o| (o.id, o.description))
                .collect()
        };
        let s = |t: &str| Some(Json::Str(t.into()));
        let ids = |q: usize, rules: &Rules| -> Vec<String> {
            seen(q, rules).into_iter().map(|o| o.0).collect()
        };
        assert_eq!(ids(0, &Rules::CLEF), ["Mid", "alpha", "zeta"]);
        assert_eq!(ids(0, &SENT), ["zeta", "alpha", "Mid"]);
        assert_eq!(seen(0, &SENT)[1].1, None, "a null description is none");
        assert_eq!(ids(1, &Rules::CLEF), ["true", "false"]);
        assert_eq!(ids(1, &SENT), ["false", "true"]);
        assert_eq!(
            seen(1, &SENT),
            [("false".to_owned(), None), ("true".to_owned(), s("Yes."))]
        );
        assert_eq!(
            seen(2, &SENT),
            [("false".to_owned(), None), ("true".to_owned(), None)]
        );
        assert_eq!(
            seen(2, &Rules::CLEF),
            [
                ("true".to_owned(), s(NOUL_TRUE)),
                ("false".to_owned(), s(NOUL_FALSE))
            ]
        );
        // `options()` is the release's.
        assert_eq!(
            r.questions[0].options(),
            r.questions[0].options_with(&Rules::CLEF)
        );
    }

    /// [`Rules::strict`] is llama.cpp's `parse_questions`: each case is refused by name under it and
    /// is a request the release's rules take.
    #[test]
    fn strict_rules_are_llama_cpps_checks() {
        let ok =
            |q: &str| format!(r#"{{"state":"s","questions":{{"q":{{"instructions":"I",{q}}}}}}}"#);
        for (text, want) in [
            (
                r#"{"state":null,"questions":{"q":{"type":"noul","instructions":"I"}}}"#.to_owned(),
                "state is required, and is not null",
            ),
            (
                r#"{"state":1,"questions":{"q":{"type":"noul"}}}"#.to_owned(),
                "q: \"instructions\" must be provided",
            ),
            (
                r#"{"state":1,"questions":{"q":{"type":"noul","instructions":null}}}"#.to_owned(),
                "q: \"instructions\" must be provided",
            ),
            (
                ok(r#""type":"score","criteria":["only"]"#),
                "q: \"criteria\" must be an array of 2 to 10 levels, not 1",
            ),
            (
                ok(r#""type":"score","criteria":[0,1,2,3,4,5,6,7,8,9,10]"#),
                "q: \"criteria\" must be an array of 2 to 10 levels, not 11",
            ),
            (
                ok(r#""type":"noul","criteria":[]"#),
                "q: noul criteria must be an object, or absent, not an array",
            ),
            (
                ok(r#""type":"noul","criteria":"yes""#),
                "q: noul criteria must be an object, or absent, not a string",
            ),
        ] {
            let e = Request::from_json_with(&parse(&text).unwrap(), &SENT)
                .expect_err(&text)
                .to_string();
            assert!(e.contains(want), "{text}: {e}");
        }
        // The release takes the ones it has no rule against, and a null noul criteria passes both.
        for text in [
            r#"{"state":null,"questions":{"q":{"type":"noul"}}}"#.to_owned(),
            r#"{"state":1,"questions":{"q":{"type":"noul","instructions":null}}}"#.to_owned(),
            ok(r#""type":"score","criteria":["only"]"#),
            ok(r#""type":"noul","criteria":[]"#),
        ] {
            assert!(
                Request::from_json_with(&parse(&text).unwrap(), &Rules::CLEF).is_ok(),
                "{text}"
            );
        }
        let null = ok(r#""type":"noul","criteria":null"#);
        assert!(Request::from_json_with(&parse(&null).unwrap(), &SENT).is_ok());
        let two = ok(r#""type":"score","criteria":["a","b"]"#);
        assert!(Request::from_json_with(&parse(&two).unwrap(), &SENT).is_ok());
    }
}
