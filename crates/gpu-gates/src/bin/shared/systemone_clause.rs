//! `systemone_is_the_schema_path`, the clause of the generative serve gates that holds `POST
//! /v1/systemone` on a general model: the rows of `tools/ref/clef/suite.jsonl` through the route and
//! through `/v1/chat/completions` on the same request converted to its chat body.
//!
//! Each row that fits the server's context (a prompt past it is the context's refusal, printed and
//! left out) is POSTed to `/v1/systemone` with `cache_prompt` off. It holds when, for every such
//! row, the answer is a 200 with the decision seats' keys in their order — `model`, `answers` (a
//! question each, in the request's order: a `choice` with `type`, `choice`, `confidence`,
//! `probabilities`; a `score` with `type`, `score`, `confidence`, `legend`, `probabilities`; a
//! `noul` with `type`, `noul`), `usage` (`input_tokens`, `output_tokens`) and `timings`
//! (`prompt_n`, `prompt_ms`, `head_ms`, `cache_n`) — and its picks are the `response_format`
//! path's: the converted chat body (`decision::generative::Ask::chat_body`) POSTed to
//! `/v1/chat/completions` gives, question by question, the option the answer picks (a choice's
//! string; a score's level and a noul's side within the 4-decimal rounding of the printed
//! probabilities). A server without the route answers every row 404.

use std::path::Path;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::serve_client::{check, curl, tokenized};
use decision::generative;
use decision::json::{self, Json};
use decision::request::Request;
use serde_json::{Value, json};

/// The ids a prompt may take of the context: the chat template's own tokens are not counted here.
const TEMPLATE_ROOM: usize = 128;

/// The rows a context holds are all that must hold.
const MIN_ROWS: usize = 7;

/// The keys of each question type's answer, in the decision seats' order.
fn answer_keys(kind: &str) -> &'static [&'static str] {
    match kind {
        "choice" => &["type", "choice", "confidence", "probabilities"],
        "score" => &["type", "score", "confidence", "legend", "probabilities"],
        _ => &["type", "noul"],
    }
}

/// The keys of a JSON object, in order; empty for any other value.
fn keys(v: &Json) -> Vec<&str> {
    match v {
        Json::Object(pairs) => pairs.iter().map(|(k, _)| k.as_str()).collect(),
        _ => Vec::new(),
    }
}

/// Whether the answer `body` of `req` has the decision seats' keys.
fn has_the_keys(req: &Request, body: &Json) -> bool {
    let part = |k: &str| body.get(k).cloned().unwrap_or(Json::Null);
    let answers = part("answers");
    keys(body) == ["model", "answers", "usage", "timings"]
        && keys(&answers)
            == req
                .questions
                .iter()
                .map(|q| q.id.as_str())
                .collect::<Vec<_>>()
        && keys(&part("usage")) == ["input_tokens", "output_tokens"]
        && keys(&part("timings")) == ["prompt_n", "prompt_ms", "head_ms", "cache_n"]
        && req.questions.iter().all(|q| {
            answers
                .get(&q.id)
                .is_some_and(|a| keys(a) == answer_keys(q.kind.word()))
        })
}

/// Whether the model's reply `said` (the chat path's JSON object) picks what the answer `a` of
/// question `q` picks. The printed probabilities are rounded to 4 decimals, so a level or a side
/// counts when it is within that of the top.
fn picks_agree(kind: &str, a: &Value, said: &Value) -> bool {
    match kind {
        "choice" => a["choice"] == *said,
        "score" => {
            let all = a["probabilities"].as_object();
            let top = all.map_or(f64::NAN, |o| {
                o.values()
                    .filter_map(Value::as_f64)
                    .fold(f64::MIN, f64::max)
            });
            said.as_u64().is_some_and(|i| {
                a["probabilities"][i.to_string()]
                    .as_f64()
                    .is_some_and(|p| p >= top - 1e-4)
            })
        }
        _ => {
            let yes = a["noul"].as_f64().unwrap_or(f64::NAN);
            match said.as_bool() {
                Some(true) => yes >= 0.5 - 1e-4,
                Some(false) => yes <= 0.5 + 1e-4,
                None => false,
            }
        }
    }
}

/// The clause, on the server at `url`: each row's answer in `<dir>/systemone-<id>.json` and the
/// chat path's in `<dir>/systemone-<id>-chat.json`. Prints its `check` line and returns it.
pub fn is_the_schema_path(url: &dyn Fn(&str) -> String, dir: &Path) -> Result<bool, GateError> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/clef/suite.jsonl");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (st, props) = curl(&url("/props"), None, false)?;
    let props: Value = serde_json::from_str(&props)?;
    let ctx = props["n_ctx"]
        .as_u64()
        .ok_or_else(|| format!("/props {st}: no n_ctx"))?;
    let ctx = usize::try_from(ctx)?;
    let (mut ran, mut keyed, mut picked) = (0usize, 0usize, 0usize);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut row: Value = serde_json::from_str(line)?;
        let id = row["id"]
            .as_str()
            .ok_or("a suite row has no id")?
            .to_owned();
        if let Some(o) = row.as_object_mut() {
            o.remove("id");
        }
        row["cache_prompt"] = json!(false);
        let req = Request::from_json(&json::parse(&row.to_string())?)?;
        let ask = generative::ask(&req)?;
        let ids = tokenized(url, &format!("{}\n{}", ask.system, ask.user))?;
        if ids.len() + TEMPLATE_ROOM > ctx {
            println!(
                "systemone {id}: left out, its prompt takes {} ids of this server's {ctx}",
                ids.len()
            );
            continue;
        }
        ran += 1;
        let (st, body) = curl(&url("/v1/systemone"), Some(&row), false)?;
        std::fs::write(dir.join(format!("systemone-{id}.json")), &body)?;
        if st != 200 {
            println!("systemone {id}: HTTP {st}: {body}");
            continue;
        }
        let got = json::parse(&body)?;
        let key_set = has_the_keys(&req, &got);
        keyed += usize::from(key_set);
        let chat = Value::Object(ask.chat_body(false));
        let (cst, ctext) = curl(&url("/v1/chat/completions"), Some(&chat), false)?;
        std::fs::write(dir.join(format!("systemone-{id}-chat.json")), &ctext)?;
        let reply: Value = serde_json::from_str(&ctext).unwrap_or(Value::Null);
        let said: Value = reply["choices"][0]["message"]["content"]
            .as_str()
            .and_then(|c| serde_json::from_str(c).ok())
            .unwrap_or(Value::Null);
        let body: Value = serde_json::from_str(&body)?;
        let same = cst == 200
            && req
                .questions
                .iter()
                .all(|q| picks_agree(q.kind.word(), &body["answers"][&q.id], &said[&q.id]));
        picked += usize::from(same);
        println!(
            "systemone {id}: key set {key_set}, picks equal {same} (chat HTTP {cst}), said {said}, \
             answers {}, usage {}, timings {}",
            body["answers"], body["usage"], body["timings"]
        );
    }
    let mut pass = true;
    check(
        &mut pass,
        "systemone_is_the_schema_path",
        ran >= MIN_ROWS && keyed == ran && picked == ran,
    );
    Ok(pass)
}
