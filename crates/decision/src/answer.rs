//! Logits to the SystemOne response body, as the release's `systemone` and `systemone_answer` build it.
//!
//! Per question, the softmax runs in f32 over the options in span order; each probability is then the
//! f64 of that f32 (torch's `.tolist()`), and every printed number is Python's `round(x, 4)`. A `choice`
//! answer lists its options in the request's criteria order and picks the first highest there; a
//! `score` is `sum(i · p_i)` as Python 3.12's `sum` adds floats (Neumaier-compensated). `confidence`
//! is the wire's, llama.cpp's server (`tools/server/server-decision.cpp` at `a4cb4c61`,
//! `decision_confidence_choice` and `decision_confidence_score`, TypeSafe's formulas): a choice's
//! `max(0, (p_max − u) / (1 − u))` with `u` the uniform probability, a score's
//! `max(0, 1 − dist / dist_uniform)` with `dist` the mean distance of the levels to the most probable
//! one and `dist_uniform` that of the uniform distribution to the middle level. `model` is the
//! server's name for the model, as llama.cpp's server answers.

use crate::Error;
use crate::encode::Encoded;
use crate::json::Json;
use crate::request::{Kind, Request};

/// Python's `round(x, 4)` of a finite f64: the correctly rounded 4-decimal value (ties to even on the
/// exact binary value), read back as the nearest f64.
#[must_use]
pub fn round4(x: f64) -> f64 {
    format!("{x:.4}")
        .parse()
        .expect("a formatted f64 reads back")
}

/// CPython 3.12's `sum` over floats starting from the int 0.
fn python_sum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for x in values {
        let t = s + x;
        c += if s.abs() >= x.abs() {
            (s - t) + x
        } else {
            (x - t) + s
        };
        s = t;
    }
    if c != 0.0 && c.is_finite() { s + c } else { s }
}

/// The f32 softmax of one question's logits.
#[must_use]
pub fn probabilities(logits: &[f32]) -> Vec<f32> {
    let mut p = logits.to_vec();
    crate::ops::softmax(&mut p);
    p
}

fn num(x: f64) -> Json {
    Json::Float(x)
}

/// A choice's confidence over its options' probabilities `p`: `(p_max − u) / (1 − u)` clamped at
/// 0, `u = 1 / n`; one option is certain.
pub(crate) fn confidence_choice(p: &[f64]) -> f64 {
    if p.len() < 2 {
        return 1.0;
    }
    let u = 1.0 / p.len() as f64;
    let p_max = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    ((p_max - u) / (1.0 - u)).max(0.0)
}

/// A score's confidence over its levels' probabilities `p`: one less the mean distance to the first
/// most probable level over that of the uniform distribution to the middle level, clamped at 0; one
/// level is certain.
pub(crate) fn confidence_score(p: &[f64]) -> f64 {
    let n = p.len();
    if n < 2 {
        return 1.0;
    }
    let mut mode = 0;
    for (i, &x) in p.iter().enumerate() {
        if x > p[mode] {
            mode = i;
        }
    }
    let (mut dist, mut dist_uniform) = (0.0f64, 0.0f64);
    for (i, &x) in p.iter().enumerate() {
        dist += x * (i as f64 - mode as f64).abs();
        dist_uniform += (i as f64 - (n - 1) as f64 / 2.0).abs() / n as f64;
    }
    (1.0 - dist / dist_uniform).max(0.0)
}

/// llama.cpp's server's probabilities of a question asked in `variants` (`format_answer`): each
/// variant's scores through a softmax at `temperature` in f64 (the f32 difference to the variant's
/// largest score, divided by the temperature), the variants averaged, the second one in the reverse
/// order of the first (it showed the options reversed). A score that is not finite is refused by
/// name, where llama.cpp's would be NaN.
pub fn softmax_averaged(variants: &[Vec<f32>], temperature: f32) -> Result<Vec<f64>, Error> {
    let n = variants.first().map_or(0, Vec::len);
    if n == 0 {
        return Err(Error::Logits("no scores to answer from".into()));
    }
    let mut probs = vec![0.0f64; n];
    for (v, scores) in variants.iter().enumerate() {
        if scores.len() != n {
            return Err(Error::Logits(format!(
                "variant {v} has {} scores, variant 0 has {n}",
                scores.len()
            )));
        }
        if scores.iter().any(|x| !x.is_finite()) {
            return Err(Error::Logits(format!(
                "variant {v} has a score that is not finite: the model could not evaluate the decision"
            )));
        }
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f64> = scores
            .iter()
            .map(|&x| (f64::from(x - top) / f64::from(temperature)).exp())
            .collect();
        let mut sum = 0.0f64;
        for e in &exps {
            sum += e;
        }
        for (i, e) in exps.iter().enumerate() {
            let at = if v == 0 { i } else { n - 1 - i };
            probs[at] += e / sum / variants.len() as f64;
        }
    }
    Ok(probs)
}

/// A noul read on `probs.len()` ratings, 0 certainly no to the last certainly yes: the expected
/// rating over the scale, `Σ pᵢ · i / (n − 1)`, summed in llama.cpp's order.
#[must_use]
pub fn expected_rating(probs: &[f64]) -> f64 {
    let top = probs.len().saturating_sub(1) as f64;
    let mut expected = 0.0f64;
    for (i, p) in probs.iter().enumerate() {
        expected += p * i as f64 / top;
    }
    expected
}

/// A score's expected level, `Σ i · pᵢ`, summed in llama.cpp's order (a plain left-to-right sum, not
/// Python's `sum`).
#[must_use]
pub fn expected_level(probs: &[f64]) -> f64 {
    let mut expected = 0.0f64;
    for (i, p) in probs.iter().enumerate() {
        expected += i as f64 * p;
    }
    expected
}

/// The response body for `req`, encoded as `enc`, with `logits` from the head, answered by the
/// model the server names `model`.
pub fn answer(
    req: &Request,
    enc: &Encoded,
    logits: &[Vec<f32>],
    model: &str,
) -> Result<Json, Error> {
    if logits.len() != enc.questions.len() || req.questions.len() != enc.questions.len() {
        return Err(Error::Logits(format!(
            "{} logit rows for {} questions",
            logits.len(),
            enc.questions.len()
        )));
    }
    let mut answers = Vec::with_capacity(logits.len());
    for ((q, eq), l) in req.questions.iter().zip(&enc.questions).zip(logits) {
        if l.len() != eq.option_ids.len() || q.id != eq.id {
            return Err(Error::Logits(format!(
                "{}: {} logits for {} options",
                eq.id,
                l.len(),
                eq.option_ids.len()
            )));
        }
        let p32 = probabilities(l);
        let p = |id: &str| -> f64 {
            let i = eq
                .option_ids
                .iter()
                .position(|o| o == id)
                .expect("every option id has a logit");
            f64::from(p32[i])
        };
        let body = match q.kind {
            Kind::Noul => vec![
                ("type".into(), Json::Str("noul".into())),
                ("noul".into(), num(round4(p("true")))),
            ],
            Kind::Choice => {
                let Some(Json::Object(criteria)) = &q.criteria else {
                    unreachable!("a validated choice question has object criteria")
                };
                let order: Vec<&str> = criteria.iter().map(|(k, _)| k.as_str()).collect();
                let mut best = order[0];
                for &o in &order[1..] {
                    if p(o) > p(best) {
                        best = o;
                    }
                }
                vec![
                    ("type".into(), Json::Str("choice".into())),
                    ("choice".into(), Json::Str(best.into())),
                    (
                        "confidence".into(),
                        num(round4(confidence_choice(
                            &order.iter().map(|&o| p(o)).collect::<Vec<_>>(),
                        ))),
                    ),
                    (
                        "probabilities".into(),
                        Json::Object(
                            order
                                .iter()
                                .map(|&o| (o.to_string(), num(round4(p(o)))))
                                .collect(),
                        ),
                    ),
                ]
            }
            Kind::Score => {
                let Some(Json::Array(levels)) = &q.criteria else {
                    unreachable!("a validated score question has array criteria")
                };
                let ids: Vec<String> = (0..levels.len()).map(|i| i.to_string()).collect();
                let score = python_sum(ids.iter().enumerate().map(|(i, id)| i as f64 * p(id)));
                vec![
                    ("type".into(), Json::Str("score".into())),
                    ("score".into(), num(round4(score))),
                    (
                        "confidence".into(),
                        num(round4(confidence_score(
                            &ids.iter().map(|id| p(id)).collect::<Vec<_>>(),
                        ))),
                    ),
                    (
                        "legend".into(),
                        Json::Object(ids.iter().cloned().zip(levels.iter().cloned()).collect()),
                    ),
                    (
                        "probabilities".into(),
                        Json::Object(
                            ids.iter()
                                .map(|id| (id.clone(), num(round4(p(id)))))
                                .collect(),
                        ),
                    ),
                ]
            }
        };
        answers.push((q.id.clone(), Json::Object(body)));
    }
    Ok(Json::Object(vec![
        ("model".into(), Json::Str(model.to_owned())),
        ("answers".into(), Json::Object(answers)),
        (
            "usage".into(),
            Json::Object(vec![
                ("input_tokens".into(), Json::Int(enc.ids.len().to_string())),
                ("output_tokens".into(), Json::Int("0".into())),
            ]),
        ),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode_with;
    use crate::json::parse;
    use crate::render::dumps;

    /// Each right side is CPython's `round(left, 4)`.
    #[test]
    fn round4_is_pythons() {
        for (x, want) in [
            (0.12345, 0.1235), // 0.12345 is stored as 0.123450000000000004174…
            (0.00005, 0.0001), // stored as 5.00000000000000023960…e-05
            (0.00015, 0.0001), // stored as 1.49999999999999993…e-04
            (2.675e-5, 0.0),
            (0.99995, 1.0), // stored as 0.999950000000000027…
            (0.5, 0.5),
            (1.0 / 3.0, 0.3333),
            (f64::from(0.734_521_9f32), 0.7345),
        ] {
            assert_eq!(round4(x), want, "{x:e}");
        }
    }

    #[test]
    fn neumaier_sum_matches_python() {
        // sum([0.1] * 10) == 1.0 in Python 3.12 (plain left-to-right addition gives 0.9999999999999999).
        assert_eq!(python_sum(std::iter::repeat_n(0.1, 10)), 1.0);
        assert_eq!(python_sum([1e100, 1.0, -1e100].into_iter()), 1.0);
    }

    fn body(text: &str, logits: &[Vec<f32>]) -> (String, usize) {
        let req = Request::from_json(&parse(text).unwrap()).unwrap();
        let enc = encode_with(
            &mut |t: &str| t.bytes().map(u32::from).collect(),
            &req,
            16384,
        )
        .unwrap();
        (
            dumps(&answer(&req, &enc, logits, "served").unwrap(), false),
            enc.ids.len(),
        )
    }

    #[test]
    fn the_body_has_the_release_shape() {
        // Logits arrive in span order: choice options sorted (billing, technical), noul (true, false).
        let (got, n) = body(
            r#"{"model":"clef-flash","state":"s","questions":{
               "department":{"type":"choice","criteria":{"technical":"Bugs","billing":"Pay"}},
               "urgency":{"type":"score","criteria":["Can wait","This week","Today"]},
               "outage":{"type":"noul"}}}"#,
            &[vec![0.0, 2.0], vec![0.0, 1.0, 0.5], vec![1.0, -1.0]],
        );
        // p(technical) = 1/(1+e⁻²) = 0.880797, its confidence (0.880797 − ½)/½ = 0.761594; urgency
        // p = softmax(0, 1, 0.5) = (0.186324, 0.506480, 0.307196); score = 0.506480 + 2·0.307196 =
        // 1.120872; its confidence 1 − (0.186324 + 0.307196)/(2/3) = 0.259720. The body names the
        // served model, not the request's.
        let want = format!(
            "{}{}{}{}{}",
            r#"{"model":"served","answers":{"department":{"type":"choice","choice":"technical","confidence":0.7616,"#,
            r#""probabilities":{"technical":0.8808,"billing":0.1192}},"#,
            r#""urgency":{"type":"score","score":1.1209,"confidence":0.2597,"legend":{"0":"Can wait","1":"This week","2":"Today"},"#,
            r#""probabilities":{"0":0.1863,"1":0.5065,"2":0.3072}},"outage":{"type":"noul","noul":0.8808}},"#,
            format_args!(r#""usage":{{"input_tokens":{n},"output_tokens":0}}}}"#),
        );
        assert_eq!(got, want);
    }

    #[test]
    fn a_tie_goes_to_the_first_option_in_request_order() {
        let (got, _) = body(
            r#"{"model":"m","state":"s","questions":{"q":{"type":"choice","criteria":{"b":"B","a":"A"}}}}"#,
            &[vec![0.0, 0.0]],
        );
        assert!(
            got.contains(r#""choice":"b","confidence":0.0,"probabilities":{"b":0.5,"a":0.5}"#),
            "{got}"
        );
    }

    #[test]
    fn mismatched_logits_are_refused() {
        let req = Request::from_json(
            &parse(r#"{"model":"m","state":"s","questions":{"q":{"type":"noul"}}}"#).unwrap(),
        )
        .unwrap();
        let enc = encode_with(
            &mut |t: &str| t.bytes().map(u32::from).collect(),
            &req,
            16384,
        )
        .unwrap();
        assert!(matches!(
            answer(&req, &enc, &[vec![0.0]], "m"),
            Err(Error::Logits(_))
        ));
        assert!(matches!(
            answer(&req, &enc, &[], "m"),
            Err(Error::Logits(_))
        ));
    }

    /// llama.cpp's confidence formulas on hand-computed cases: a choice's distance of p_max above
    /// uniform, a score's distance to the mode against the uniform one, each clamped at 0; one
    /// option or level is certain.
    #[test]
    fn confidence_is_llama_cpps() {
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        // (0.7 − 1/3) / (2/3) = 0.55.
        assert!(close(confidence_choice(&[0.7, 0.2, 0.1]), 0.55));
        assert_eq!(confidence_choice(&[0.25; 4]), 0.0);
        assert_eq!(confidence_choice(&[1.0]), 1.0);
        // Mode 0: dist = 0.1·1 + 0.1·2 = 0.3; uniform's = (1 + 0 + 1)/3; 1 − 0.3·1.5 = 0.55.
        assert!(close(confidence_score(&[0.8, 0.1, 0.1]), 0.55));
        // Uniform levels: the first is the mode, dist 1 against 2/3, clamped.
        assert_eq!(confidence_score(&[1.0 / 3.0; 3]), 0.0);
        assert_eq!(confidence_score(&[1.0]), 1.0);
    }
}
