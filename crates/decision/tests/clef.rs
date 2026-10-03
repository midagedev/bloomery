//! `just gate-decision-clef`: Clef's encoder, head and answer against the release's own Python.
//!
//! The reference is `$BLOOMERY_DATA/clef/flash/ref/` (`tools/ref/clef_ref.py`): `reference.jsonl`
//! (the suite's 8 requests: ids, spans, the head's logits in bf16 as shipped, in f32 on the card and
//! in f64 on the CPU, the `systemone()` body), `edge.jsonl` (6 encode-only requests) and, per suite
//! request, `<id>.hidden.f32` and `<id>.lexical.{f32,ids}`. The requests themselves are
//! `tools/ref/clef/{suite,edge}.jsonl`. The tokenizer is the Flash GGUF [`CLEF_GGUF`]; the head is
//! the snapshot's `joint_head.safetensors` with its `joint_head_config.json` ([`CLEF_HEAD`]).

use std::collections::HashMap;
use std::path::PathBuf;

use decision::answer::{answer, probabilities};
use decision::encode::{Encoded, EncodedQuestion, MAX_LENGTH, encode};
use decision::head::ClefHead;
use decision::json::{self, Json};
use decision::render::render;
use decision::request::{Kind, Request};

fn data() -> PathBuf {
    PathBuf::from(std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into()))
        .join("clef/flash")
}

fn requests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/clef")
}

/// The Flash GGUF `tools/ref/clef_ref.py`'s round quantized; the tokenizer reads its header only.
const CLEF_GGUF: &str = "/models/clef-flash/clef-flash-Q4_K_M.gguf";

/// The release snapshot's head, beside the model files.
const CLEF_HEAD: &str = "/models/clef-flash/hf/joint_head.safetensors";

fn jsonl(path: &PathBuf) -> Vec<Json> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| json::parse(l).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .collect()
}

fn id_of(row: &Json) -> String {
    row.get("id")
        .and_then(Json::as_str)
        .expect("every row has an id")
        .to_string()
}

fn usize_of(v: &Json) -> usize {
    usize::try_from(v.as_u64().expect("an index")).unwrap()
}

fn array(v: &Json) -> &[Json] {
    match v {
        Json::Array(a) => a,
        other => panic!("{} is not an array", other.kind()),
    }
}

fn span(v: &Json) -> (usize, usize) {
    let a = array(v);
    (usize_of(&a[0]), usize_of(&a[1]))
}

/// A reference row's ids and spans as the encoder's type.
fn reference_encoding(row: &Json) -> Encoded {
    let ids = array(row.get("input_ids").unwrap())
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let questions = array(row.get("questions").unwrap())
        .iter()
        .map(|q| EncodedQuestion {
            id: q
                .get("question_id")
                .and_then(Json::as_str)
                .unwrap()
                .to_string(),
            kind: match usize_of(q.get("question_type").unwrap()) {
                0 => Kind::Noul,
                1 => Kind::Choice,
                2 => Kind::Score,
                t => panic!("question type {t}"),
            },
            question_span: span(q.get("question_span").unwrap()),
            option_spans: array(q.get("option_spans").unwrap())
                .iter()
                .map(span)
                .collect(),
            option_ids: array(q.get("option_ids").unwrap())
                .iter()
                .map(|o| o.as_str().unwrap().to_string())
                .collect(),
        })
        .collect();
    Encoded { ids, questions }
}

fn requests() -> HashMap<String, Json> {
    let dir = requests_dir();
    ["suite.jsonl", "edge.jsonl"]
        .iter()
        .flat_map(|f| jsonl(&dir.join(f)))
        .map(|r| (id_of(&r), r))
        .collect()
}

fn f32_file(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{}", path.display());
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// ids: our encoder over all 14 requests gives the release's ids and every span exactly, and our
/// render gives the release's state text.
#[test]
#[ignore = "needs the Clef reference under $BLOOMERY_DATA and the Flash GGUF; just gate-decision-clef"]
fn hw_clef_ids_equal_the_release() {
    let reqs = requests();
    let tok =
        tokenizer::Tokenizer::from_gguf(CLEF_GGUF).unwrap_or_else(|e| panic!("{CLEF_GGUF}: {e}"));
    let rows: Vec<Json> = ["reference.jsonl", "edge.jsonl"]
        .iter()
        .flat_map(|f| jsonl(&data().join("ref").join(f)))
        .collect();
    assert_eq!(rows.len(), 14, "8 suite rows and 6 edge rows");
    let mut bad = Vec::new();
    for row in &rows {
        let id = id_of(row);
        let body = reqs
            .get(&id)
            .unwrap_or_else(|| panic!("{id}: no request in {}", requests_dir().display()));
        let req = Request::from_json(body).unwrap();
        let want = reference_encoding(row);
        let got = encode(&tok, &req, MAX_LENGTH).unwrap();
        let render_ok =
            row.get("state_render").and_then(Json::as_str) == Some(render(&req.state).as_str());
        let first = got.ids.iter().zip(&want.ids).position(|(a, b)| a != b);
        let ok = got == want && render_ok;
        println!(
            "ids {id}: n={} ref_n={} first_diff={first:?} spans_equal={} render_equal={render_ok} {}",
            got.ids.len(),
            want.ids.len(),
            got.questions == want.questions,
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            bad.push(id);
        }
    }
    assert!(bad.is_empty(), "encodings differ from the release: {bad:?}");
}

fn load_head() -> ClefHead {
    let path = PathBuf::from(CLEF_HEAD);
    ClefHead::open(&path, None).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The head over one reference row's own ids, spans, hidden states and lexical rows.
fn our_logits(head: &ClefHead, row: &Json) -> (Encoded, Vec<Vec<f32>>) {
    let id = id_of(row);
    let enc = reference_encoding(row);
    let dir = data().join("ref");
    let hidden = f32_file(&dir.join(format!("{id}.hidden.f32")));
    let lex = f32_file(&dir.join(format!("{id}.lexical.f32")));
    let lex_ids: Vec<u32> = std::fs::read_to_string(dir.join(format!("{id}.lexical.ids")))
        .unwrap()
        .lines()
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let mut rows = |ids: &[u32]| -> Result<Vec<f32>, String> {
        if ids != lex_ids.as_slice() {
            return Err(format!(
                "{id}: the head asked for other ids than {id}.lexical.ids"
            ));
        }
        Ok(lex.clone())
    };
    let logits = head
        .forward(&hidden, &enc, &mut rows)
        .unwrap_or_else(|e| panic!("{id}: {e}"));
    (enc, logits)
}

fn logits_of(row: &Json, key: &str) -> Vec<Vec<f64>> {
    array(row.get(key).unwrap_or_else(|| panic!("no {key}")))
        .iter()
        .map(|q| array(q).iter().map(|v| v.as_f64().unwrap()).collect())
        .collect()
}

/// head: our f32 logits against the f64 referee's on the same inputs.
///
/// The band. The only difference between the two heads is arithmetic: the same f32 hidden states and
/// lexical rows, the same bf16-valued weights. A logit is `P·prior + g·(J·cosine + r)` with the head's
/// scalars P = J = exp(2.296875) = 9.943 and g = sigmoid(-2) = 0.1192 (read from the file), and
/// |logit| < 16, where one ulp is 2⁻²⁰ = 9.5e-7. Every vector the logit reads leaves a LayerNorm or a
/// normalize, so its error is relative to a vector of unit scale: about u = 2⁻²⁴ per stage it passed,
/// not a random walk over Σ|aᵢbᵢ| per dot.
///
/// - the prior path: two unit vectors (≤ 4u each after the normalize) and one product by P:
///   ≤ 4 ulps of the logit;
/// - the joint path: ≈ 64 stages through the 2 routing and 4 decoder layers, ≤ 64u = 3.8e-6 relative,
///   on g·J = 1.19 (the cosine) and on g·|r| ≤ 1.19 (the scorer's |r| ≲ 10): ≤ 5 + 5 ulps;
/// - the last product and sum: 2 ulps.
///
/// That is 16 ulps = 1.5e-5; the band is 2e-5.
const HEAD_BAND: f64 = 2e-5;

#[test]
#[ignore = "needs the Clef reference under $BLOOMERY_DATA; just gate-decision-clef"]
fn hw_clef_head_matches_the_f64_referee() {
    let head = load_head();
    let (p, j, g) = head.scales();
    println!("head scales: prior {p} joint {j} gate {g}");
    let rows = jsonl(&data().join("ref/reference.jsonl"));
    assert_eq!(rows.len(), 8);
    let mut worst = 0f64;
    let mut bad = Vec::new();
    for row in &rows {
        let id = id_of(row);
        let t0 = std::time::Instant::now();
        let (enc, ours) = our_logits(&head, row);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let referee = logits_of(row, "logits_f64_head");
        assert_eq!(ours.len(), referee.len(), "{id}: question count");
        let mut row_max = 0f64;
        let mut dp_max = 0f64;
        for (o, r) in ours.iter().zip(&referee) {
            assert_eq!(o.len(), r.len(), "{id}: option count");
            for (a, b) in o.iter().zip(r) {
                row_max = row_max.max((f64::from(*a) - b).abs());
            }
            let r32: Vec<f32> = r.iter().map(|&v| v as f32).collect();
            for (a, b) in probabilities(o).iter().zip(probabilities(&r32)) {
                dp_max = dp_max.max(f64::from((a - b).abs()));
            }
        }
        worst = worst.max(row_max);
        let ok = row_max <= HEAD_BAND;
        println!(
            "head {id}: n={} max|dlogit|={row_max:.3e} band={HEAD_BAND:.0e} max|dp|={dp_max:.3e} ms={ms:.0} (functional) {}",
            enc.ids.len(),
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            bad.push(id);
        }
    }
    println!("head: worst max|dlogit| {worst:.3e} against the band {HEAD_BAND:.0e}");
    assert!(bad.is_empty(), "logits outside the band: {bad:?}");
}

/// The top option of an answer: the choice, the most probable score level, or noul's side.
fn top(a: &Json) -> String {
    match a.get("type").and_then(Json::as_str) {
        Some("choice") => a.get("choice").and_then(Json::as_str).unwrap().to_string(),
        Some("noul") => (a.get("noul").and_then(Json::as_f64).unwrap() >= 0.5).to_string(),
        Some("score") => {
            let Some(Json::Object(p)) = a.get("probabilities") else {
                panic!("score without probabilities")
            };
            let mut best = &p[0];
            for e in &p[1..] {
                if e.1.as_f64() > best.1.as_f64() {
                    best = e;
                }
            }
            best.0.clone()
        }
        _ => panic!("answer without a type"),
    }
}

/// Every number of an answer by its key path, and the answer with its numbers blanked.
fn numbers(v: &Json, path: &str, out: &mut Vec<(String, f64)>) -> Json {
    match v {
        Json::Float(_) | Json::Int(_) if !path.ends_with("tokens") => {
            out.push((path.to_string(), v.as_f64().unwrap()));
            Json::Null
        }
        Json::Object(pairs) => Json::Object(
            pairs
                .iter()
                .map(|(k, x)| (k.clone(), numbers(x, &format!("{path}.{k}"), out)))
                .collect(),
        ),
        Json::Array(items) => Json::Array(items.iter().map(|x| numbers(x, path, out)).collect()),
        other => other.clone(),
    }
}

/// answer: our body from our f32 logits against the release's `systemone()` body: the same shape
/// (keys, order, legends, usage) and the same top option for every question; |Δp| against the bf16
/// body is a diagnostic, printed beside the same body built from the f64 referee's logits.
#[test]
#[ignore = "needs the Clef reference under $BLOOMERY_DATA; just gate-decision-clef"]
fn hw_clef_answer_matches_the_release() {
    let head = load_head();
    let reqs = requests();
    let rows = jsonl(&data().join("ref/reference.jsonl"));
    let mut bad = Vec::new();
    for row in &rows {
        let id = id_of(row);
        let req = Request::from_json(&reqs[&id]).unwrap();
        let (enc, ours) = our_logits(&head, row);
        // The release echoes the request's model; the engine names the one it serves.
        let model = reqs[&id].get("model").and_then(Json::as_str).unwrap();
        let body = answer(&req, &enc, &ours, model).unwrap();
        let referee: Vec<Vec<f32>> = logits_of(row, "logits_f64_head")
            .iter()
            .map(|q| q.iter().map(|&v| v as f32).collect())
            .collect();
        let body64 = answer(&req, &enc, &referee, model).unwrap();
        let release = row.get("response").unwrap();
        let (mut n_ours, mut n_rel, mut n_64) = (Vec::new(), Vec::new(), Vec::new());
        let shape_ok = numbers(&body, "", &mut n_ours) == numbers(release, "", &mut n_rel);
        numbers(&body64, "", &mut n_64);
        let paths_ok = n_ours.iter().map(|x| &x.0).eq(n_rel.iter().map(|x| &x.0));
        let dp_bf16 = n_ours
            .iter()
            .zip(&n_rel)
            .map(|(a, b)| (a.1 - b.1).abs())
            .fold(0.0, f64::max);
        let dp_f64 = n_ours
            .iter()
            .zip(&n_64)
            .map(|(a, b)| (a.1 - b.1).abs())
            .fold(0.0, f64::max);
        let (Some(Json::Object(a_ours)), Some(Json::Object(a_rel))) =
            (body.get("answers"), release.get("answers"))
        else {
            panic!("{id}: no answers object")
        };
        let mut tops_ok = a_ours.len() == a_rel.len();
        for ((q, a), (_, r)) in a_ours.iter().zip(a_rel) {
            let probs: Vec<f32> =
                probabilities(&ours[enc.questions.iter().position(|e| &e.id == q).unwrap()]);
            let mut sorted = probs.clone();
            sorted.sort_by(|x, y| y.total_cmp(x));
            let margin = sorted[0] - sorted.get(1).copied().unwrap_or(0.0);
            let same = top(a) == top(r);
            tops_ok &= same;
            println!(
                "answer {id}/{q}: top ours={} release={} {} p_ours={} p_release={} margin={margin:.4}",
                top(a),
                top(r),
                if same { "equal" } else { "DIFFERENT" },
                json_num(a),
                json_num(r),
            );
        }
        let ok = shape_ok && paths_ok && tops_ok;
        println!(
            "answer {id}: shape_equal={shape_ok} tops_equal={tops_ok} max|dp| vs bf16 body={dp_bf16:.4} vs f64-logit body={dp_f64:.4} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            bad.push(id);
        }
    }
    assert!(bad.is_empty(), "answers differ from the release: {bad:?}");
}

/// The probability an answer reports for its top option (`confidence` is llama.cpp's formula in
/// ours and the top probability in the release's, so it is not compared).
fn json_num(a: &Json) -> String {
    let p = if a.get("type").and_then(Json::as_str) == Some("noul") {
        a.get("noul")
    } else {
        a.get("probabilities").and_then(|p| p.get(&top(a)))
    };
    p.and_then(Json::as_f64)
        .map_or_else(|| "?".into(), |v| v.to_string())
}
