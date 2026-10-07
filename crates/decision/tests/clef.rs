//! `just gate-decision-clef`: Clef's encoder, head and answer against the release's own Python.
//!
//! The reference is `$BLOOMERY_DATA/clef/flash/ref/` (`tools/ref/clef_ref.py`): `reference.jsonl`
//! (the suite's 8 requests: ids, spans, the head's logits in bf16 as shipped, in f32 on the card and
//! in f64 on the CPU, the `systemone()` body), `edge.jsonl` (6 encode-only requests) and, per suite
//! request, `<id>.hidden.f32` and `<id>.lexical.{f32,ids}`. The requests themselves are
//! `tools/ref/clef/{suite,edge}.jsonl`. The tokenizer is the Flash GGUF [`CLEF_GGUF`]; the head is
//! the snapshot's `joint_head.safetensors` with its `joint_head_config.json` ([`CLEF_HEAD`]). The head
//! in llama.cpp's Clef layout, the same release's weights as bartowski's GGUF carries them
//! ([`CLEF_LAYOUT_GGUF`]), is held to the release's: tensor by tensor, and through the logits.

use std::collections::HashMap;
use std::path::PathBuf;

use decision::answer::{answer, probabilities};
use decision::encode::{Encoded, EncodedQuestion, MAX_LENGTH, encode};
use decision::gguf_head::GgufHead;
use decision::head::{ClefHead, SCALARS, Weights};
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

/// bartowski's Clef-Flash Q5_K_M in llama.cpp's Clef layout (`general.architecture = clef`, the
/// head inside the file, 6,982,686,880 bytes; HF commit `5fcdd9ba`).
const CLEF_LAYOUT_GGUF: &str = "/root/models/clef-flash/Cloudflare_clef-flash-Q5_K_M.gguf";

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

/// The head of the clef-layout file.
fn load_gguf_head(split: &gguf::Split) -> ClefHead {
    ClefHead::from_gguf(split).unwrap_or_else(|e| panic!("{CLEF_LAYOUT_GGUF}: {e}"))
}

/// layout, tensors: every tensor of the head in the clef-layout file against the release's
/// `joint_head.safetensors`, widened to f32 as the release stores it (bf16).
///
/// The conversion's dtype change, named: llama.cpp's converter writes the norms, biases and the
/// type embedding as f32 and the scorer's output weight as bf16 (widening a bf16 is exact), so
/// those equal the release's bit for bit; the six projections, the attention and feed-forward
/// matrices and the scorer's first weight are quantized to Q8_0. A Q8_0 block of 32 values
/// stores `d = amax / 127` as f16 and `q = round(x / d)`: the error is at most `d / 2` from the
/// rounding plus `127 · |d - f16(d)| <= 127 · d · 2^-11` from the scale, `0.5625 d`
/// (ggml `quantize_row_q8_0_ref`; the file is quantized from bf16 or f32 values, never through
/// f16, which would add `2 · 127 · 2^-11 d`). The three scalars are stored transformed: within
/// two ulps of the f32 formula (`clef.py` computes them in f64).
#[test]
#[ignore = "needs the clef-layout GGUF and the release's head; just gate-decision-clef"]
fn hw_clef_layout_tensors_are_the_releases_within_q8_0() {
    let release = decision::safetensors::Safetensors::open(&PathBuf::from(CLEF_HEAD)).unwrap();
    let split = gguf::Split::open(CLEF_LAYOUT_GGUF).unwrap_or_else(|e| panic!("{e}"));
    let src = GgufHead::open(&split).unwrap_or_else(|e| panic!("{e}"));
    let cfg = src.config();
    println!("config from the file: {cfg:?}");
    let (mut exact, mut q8, mut worst, mut bad) = (0usize, 0usize, 0f64, Vec::new());
    for (name, shape) in cfg.weights() {
        if SCALARS.contains(&name.as_str()) {
            continue;
        }
        assert_eq!(src.shape(&name).as_ref(), Some(&shape), "{name}");
        let want = release.tensor(&name).unwrap().data;
        let got = src.tensor(&name).unwrap();
        let types = src.stored_types(&name).unwrap();
        if types.iter().all(|t| *t == gguf::GgmlType::Q8_0) {
            q8 += 1;
            for (w, g) in want.chunks(32).zip(got.chunks(32)) {
                let d = w.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
                for (a, b) in w.iter().zip(g) {
                    let err = f64::from((a - b).abs());
                    if d > 0.0 {
                        worst = worst.max(err / f64::from(d));
                    }
                    if err > 0.5625 * f64::from(d) * (1.0 + 1e-6) {
                        bad.push(format!("{name}: {a} vs {b}, d {d}"));
                    }
                }
            }
        } else if types
            .iter()
            .all(|t| matches!(t, gguf::GgmlType::F32 | gguf::GgmlType::BF16))
        {
            exact += 1;
            if got
                .iter()
                .zip(&want)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                bad.push(format!("{name}: stored {types:?}, not the release's bits"));
            }
        } else {
            bad.push(format!("{name}: stored as {types:?}"));
        }
    }
    println!(
        "head tensors: {exact} stored exactly (f32/bf16, bits equal), {q8} in Q8_0 with the largest \
         error {worst:.4} of its block's step d = amax/127 (bound 0.5625)"
    );
    let release_scales = {
        let raw = |n: &str| release.tensor(n).unwrap().data[0];
        decision::head::scales_of(raw(SCALARS[0]), raw(SCALARS[1]), raw(SCALARS[2]))
    };
    let stored = src.scales().unwrap();
    println!("scales: release {release_scales:?} file {stored:?}");
    for (a, b) in [
        (release_scales.0, stored.0),
        (release_scales.1, stored.1),
        (release_scales.2, stored.2),
    ] {
        assert!(
            (a - b).abs() <= 2.0 * a.abs() * f32::EPSILON,
            "{a} vs {b}: more than two ulps"
        );
    }
    assert!(bad.is_empty(), "tensors off the release's: {bad:?}");
    assert!(
        q8 > 0 && exact > 0,
        "{q8} Q8_0 tensors and {exact} exact ones"
    );
}

/// What the head's logits move by with the file's Q8_0 matrices in place of the release's bf16
/// ones, on the same hidden states and lexical rows.
///
/// PIN(2026-10-07): 5e-3 on a logit. Measured on the eight reference requests (31 questions): the
/// worst request moves 1.4e-3 (the others 0.8e-3 to 1.3e-3), so the band is 3.6 times the worst;
/// the runs repeat exactly. Written before the run: 1e-2, in 2e-3 to 5e-2 — a Q8_0 step
/// `amax / 127` moves a weight by 0.45 % of its scale at random, and ~10 products on a path of
/// LayerNorms were taken to add in quadrature to 1.4 % of a unit vector, which the joint logit
/// `gate · (scale · cosine + residual)` (1.19 · cosine + 0.12 · residual) shows at 1e-2. The
/// measure is a seventh of that: the residual connections carry each block's clean input past its
/// noise, and the prior path (the output rows against the pooled states) reads no head weight.
/// A wrong mapping is far outside it: with the key and value projections swapped the same run
/// moves logits by 1.6 at the worst request and 0.4 to 1.2 at the others.
const LAYOUT_LOGIT_BAND: f64 = 5e-3;

/// A probability moves by `p (1 - p) <= 1/4` of its logit's move, and a softmax over two options
/// by the difference of two: half of [`LAYOUT_LOGIT_BAND`], which also bounds the noul's sigmoid
/// (measured worst 2.4e-4).
const LAYOUT_PROB_BAND: f64 = 2.5e-3;

/// The probability of an answer's top option, and how far its decision is from flipping: for a
/// noul `|p - 0.5|`; for a choice or score half the top two probabilities' difference. A
/// probability that moves by at most `b` flips the top only where this is at most `b`.
fn slack(a: &Json) -> f64 {
    if a.get("type").and_then(Json::as_str) == Some("noul") {
        return (a.get("noul").and_then(Json::as_f64).unwrap() - 0.5).abs();
    }
    let Some(Json::Object(p)) = a.get("probabilities") else {
        panic!("an answer without probabilities")
    };
    let mut v: Vec<f64> = p.iter().map(|e| e.1.as_f64().unwrap()).collect();
    v.sort_by(|x, y| y.total_cmp(x));
    (v[0] - v.get(1).copied().unwrap_or(0.0)) / 2.0
}

/// layout, logits: the clef-layout file's head and the release's, on the eight reference requests'
/// own hidden states: the same inputs, so the only difference is the head's dtype
/// ([`hw_clef_layout_tensors_are_the_releases_within_q8_0`]). Every logit within
/// [`LAYOUT_LOGIT_BAND`], every probability within [`LAYOUT_PROB_BAND`], and every question's top
/// the same unless its decision was within the probability band of flipping ([`slack`]).
#[test]
#[ignore = "needs the Clef reference under $BLOOMERY_DATA and the clef-layout GGUF; just gate-decision-clef"]
fn hw_clef_layout_head_scores_as_the_release_head() {
    let release = load_head();
    let split = gguf::Split::open(CLEF_LAYOUT_GGUF).unwrap_or_else(|e| panic!("{e}"));
    let layout = load_gguf_head(&split);
    assert_eq!(layout.config(), release.config());
    let reqs = requests();
    let rows = jsonl(&data().join("ref/reference.jsonl"));
    assert_eq!(rows.len(), 8);
    let (mut worst_l, mut worst_p, mut bad) = (0f64, 0f64, Vec::new());
    for row in &rows {
        let id = id_of(row);
        let req = Request::from_json(&reqs[&id]).unwrap();
        let (enc, a) = our_logits(&release, row);
        let (_, b) = our_logits(&layout, row);
        let dl = a
            .iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(x, y)| f64::from((x - y).abs()))
            .fold(0.0, f64::max);
        let dp = a
            .iter()
            .zip(&b)
            .flat_map(|(x, y)| probabilities(x).into_iter().zip(probabilities(y)))
            .map(|(x, y)| f64::from((x - y).abs()))
            .fold(0.0, f64::max);
        let model = reqs[&id].get("model").and_then(Json::as_str).unwrap();
        let (body_a, body_b) = (
            answer(&req, &enc, &a, model).unwrap(),
            answer(&req, &enc, &b, model).unwrap(),
        );
        let (Some(Json::Object(ra)), Some(Json::Object(rb))) =
            (body_a.get("answers"), body_b.get("answers"))
        else {
            panic!("{id}: no answers")
        };
        for ((q, x), (_, y)) in ra.iter().zip(rb) {
            let (tx, ty, s) = (top(x), top(y), slack(x));
            let flip = tx != ty;
            println!(
                "layout {id}/{q}: top release={tx} layout={ty} {} slack={s:.4} p_release={} p_layout={}",
                if flip { "DIFFERENT" } else { "equal" },
                json_num(x),
                json_num(y)
            );
            if flip && s > LAYOUT_PROB_BAND {
                bad.push(format!("{id}/{q}: top {tx} vs {ty} with slack {s:.4}"));
            }
        }
        println!(
            "layout {id}: n={} max|dlogit|={dl:.3e} (band {LAYOUT_LOGIT_BAND:.1e}) max|dp|={dp:.3e} (band {LAYOUT_PROB_BAND:.1e})",
            enc.ids.len()
        );
        if dl > LAYOUT_LOGIT_BAND || dp > LAYOUT_PROB_BAND {
            bad.push(format!("{id}: dlogit {dl:.3e}, dp {dp:.3e}"));
        }
        worst_l = worst_l.max(dl);
        worst_p = worst_p.max(dp);
    }
    println!("layout: worst max|dlogit| {worst_l:.3e}, worst max|dp| {worst_p:.3e}");
    assert!(
        bad.is_empty(),
        "the layout's head is off the release's: {bad:?}"
    );
}
