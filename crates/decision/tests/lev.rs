//! `just gate-decision-lev`: lev's label readout against llama.cpp mainline's `/v1/systemone` server.
//!
//! The reference is `$BLOOMERY_DATA/lev/4b/ref/` (`tools/ref/lev_ref.py`, the server at
//! [`LCPP_BUILD`] answering the 14 requests of `tools/ref/lev/{suite,edge}.jsonl` on [`LEV_GGUF`]):
//! `reference.jsonl` (per request, the ids of every prompt the server built and its body) and
//! `labels.json` (the label codes and token ids the server's own tokenizer gives). The first test holds
//! our template render, tokenizer and labels to those ids exactly; the other two are plain: the answer
//! math (temperatures, the softmax average, the rating scale, the body) by hand-computed cases, and
//! the policy (the options' order, the prompts' shape, the refusals) over a small template.

use std::path::PathBuf;

use decision::answer::{expected_level, expected_rating, softmax_averaged};
use decision::json::{self, Json};
use decision::label::{Asked, LabelModel, Labels, Temperatures, llama_float};
use decision::release::lev::{self, POLICY};
use decision::render::dumps;
use decision::request::{Kind, Request};
use gguf::Split;
use jinja::ChatTemplate;

fn data() -> PathBuf {
    PathBuf::from(std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into()))
        .join("lev/4b")
}

fn requests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/lev")
}

/// ggml-org's Q4_K_M of lev (a Qwen3.5-4B dense file, 3,011,777,440 bytes, sha256 `3f61b27c`), under
/// the box's `/root/models`.
const LEV_GGUF: &str = "/root/models/lev-4b/lev-Q4_K_M.gguf";

/// The oracle the reference was taken from: llama.cpp's commit that added the decision server, with
/// `tools/ref/lev/lcpp-ids.patch` (its sha256) applied. A dump of another build is refused.
const LCPP_BUILD: &str = "a4cb4c61fd9d9c2066c7c1747821d3d65b8943bd+ids-patch:a51d47c78ec585b5bedd51a90af74d2c12daa335375fe3f64f4a2464d6b19435";

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

fn array(v: &Json) -> &[Json] {
    match v {
        Json::Array(a) => a,
        other => panic!("{} is not an array", other.kind()),
    }
}

fn ids_of(v: &Json) -> Vec<u32> {
    array(v)
        .iter()
        .map(|x| u32::try_from(x.as_u64().expect("an id")).expect("an id fits u32"))
        .collect()
}

/// The suite and edge requests by id, each as the server is posted it (`id` dropped).
fn requests() -> std::collections::HashMap<String, Json> {
    let dir = requests_dir();
    ["suite.jsonl", "edge.jsonl"]
        .iter()
        .flat_map(|f| jsonl(&dir.join(f)))
        .map(|mut r| {
            let id = id_of(&r);
            if let Json::Object(pairs) = &mut r {
                pairs.retain(|(k, _)| k != "id");
            }
            (id, r)
        })
        .collect()
}

/// ids: the label list is the server's own, and our render, tokenizer and plan give every prompt of
/// the 14 requests the server's ids, one prompt each, in its order, and the body's `usage` is their
/// count.
#[test]
#[ignore = "needs the lev reference under $BLOOMERY_DATA and the lev GGUF; just gate-decision-lev"]
fn hw_lev_ids_equal_mainline() {
    let reqs = requests();
    let split = Split::open(LEV_GGUF).unwrap_or_else(|e| panic!("{LEV_GGUF}: {e}"));
    let tok =
        tokenizer::Tokenizer::from_gguf(LEV_GGUF).unwrap_or_else(|e| panic!("{LEV_GGUF}: {e}"));
    let model = LabelModel::open(POLICY, &split, |code| tok.encode(code, false, false))
        .unwrap_or_else(|e| panic!("{LEV_GGUF}: {e}"));

    let ref_dir = data().join("ref");
    let labels = jsonl(&ref_dir.join("labels.json"));
    let want: Vec<(String, u32)> = array(labels[0].get("labels").expect("a labels key"))
        .iter()
        .map(|l| {
            (
                l.get("code").and_then(Json::as_str).unwrap().to_string(),
                u32::try_from(l.get("id").and_then(Json::as_u64).unwrap()).unwrap(),
            )
        })
        .collect();
    let got: Vec<(String, u32)> = (0..model.labels().len())
        .map(|i| (model.labels().text(i).to_string(), model.labels().ids()[i]))
        .collect();
    let labels_ok = got == want;
    println!(
        "labels: n={} ref_n={} {} .. {} {}",
        got.len(),
        want.len(),
        got[0].0,
        got[got.len() - 1].0,
        if labels_ok { "PASS" } else { "FAIL" }
    );
    assert!(labels_ok, "the label list differs from the server's");

    let rows = jsonl(&ref_dir.join("reference.jsonl"));
    assert_eq!(rows.len(), 14, "8 suite rows and 6 edge rows");
    let mut bad = Vec::new();
    for row in &rows {
        let id = id_of(row);
        let build = row.get("lcpp_build").and_then(Json::as_str).unwrap_or("");
        assert_eq!(
            build, LCPP_BUILD,
            "{id}: the dump is of another oracle build"
        );
        let body = reqs
            .get(&id)
            .unwrap_or_else(|| panic!("{id}: no request in {}", requests_dir().display()));
        let req = Request::from_json_with(body, &POLICY.rules).unwrap();
        let asked = model.plan(&req).unwrap();
        let prompts = model.prompts(&req, &asked).unwrap();
        let tasks = array(row.get("tasks").unwrap());
        let ours: Vec<Vec<u32>> = prompts
            .iter()
            .map(|p| tok.encode(&p.text, false, true))
            .collect();
        let theirs: Vec<Vec<u32>> = tasks
            .iter()
            .map(|t| ids_of(t.get("ids").unwrap()))
            .collect();
        let plan_ok = prompts.len() == tasks.len()
            && prompts.iter().zip(tasks).all(|(p, t)| {
                t.get("question").and_then(Json::as_str) == Some(&req.questions[p.question].id)
                    && t.get("variant").and_then(Json::as_u64) == Some(p.variant as u64)
            });
        let n: usize = ours.iter().map(Vec::len).sum();
        let ref_n: usize = theirs.iter().map(Vec::len).sum();
        let usage = row
            .get("response")
            .and_then(|r| r.get("usage"))
            .and_then(|u| u.get("input_tokens"))
            .and_then(Json::as_u64);
        let first = ours
            .iter()
            .zip(&theirs)
            .enumerate()
            .find_map(|(i, (a, b))| {
                a.iter()
                    .zip(b)
                    .position(|(x, y)| x != y)
                    .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
                    .map(|at| (i, at))
            });
        let ok = plan_ok && ours == theirs && usage == Some(n as u64);
        println!(
            "ids {id}: prompts={} n={n} ref_n={ref_n} usage={usage:?} plan_equal={plan_ok} first_diff={first:?} {}",
            prompts.len(),
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            bad.push(id);
        }
    }
    assert!(bad.is_empty(), "prompts differ from the server's: {bad:?}");
}

/// `n` labels of a vocabulary that holds every code as one token up to the `n`-th and none after.
fn labels_of(n: usize) -> Labels {
    let mut seen = 0;
    Labels::from_vocab(|code| {
        seen += 1;
        if seen <= n {
            vec![1000 + u32::try_from(seen).unwrap()]
        } else {
            vec![u32::try_from(code.len()).unwrap(), 7]
        }
    })
}

/// A model over `labels` that reads a 4-wide hidden state: label `i`'s row is hand-made so that its
/// logit is easy to compute, and `template` renders the prompts.
fn model_of(labels: Labels, template: &str, temps: Temperatures) -> LabelModel {
    let n = labels.len();
    let mut rows = vec![0f32; n * 4];
    rows[..4].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    rows[4..8].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
    rows[8..12].copy_from_slice(&[0.5, 0.5, 0.5, 0.5]);
    LabelModel::new(
        POLICY,
        ChatTemplate::parse(template).expect("the template parses"),
        labels,
        rows,
        4,
        temps,
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

fn request(text: &str) -> Request {
    Request::from_json_with(&json::parse(text).unwrap(), &POLICY.rules)
        .unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
}

/// The answer math, each right side computed by hand (or in Python's f64, which does the same
/// operations in the same order): the temperature llama.cpp reads from an f32 key, the buckets that
/// pick it, the softmax average over variants (the second reversed), the rating scale's expected value,
/// and the body's shape, key order and numbers.
#[test]
fn lev_answer_math() {
    // The f32 key goes through std::to_string (six decimals) and strtof: the value is the f32 nearest to
    // the six-decimal text, not the stored one. lev's six calibrations, cast to f32 as the converter
    // writes them, each differ from what the server holds.
    let stored = |x: f64| x as f32;
    assert_ne!(
        stored(1.789_783_450_971_802),
        llama_float(stored(1.789_783_450_971_802))
    );
    for (calibrated, held) in [
        (1.789_783_450_971_802, 1.789_783_f32),
        (1.608_568_500_166_153_2, 1.608_569),
        (1.664_111_261_733_098_6, 1.664_111),
        (1.766_665_953_967_977_9, 1.766_666),
        (2.803_323_663_032_788, 2.803_324),
        (2.333_293_083_170_645_8, 2.333_293),
    ] {
        assert_eq!(
            llama_float(stored(calibrated)).to_bits(),
            held.to_bits(),
            "{calibrated}"
        );
    }
    let temps = Temperatures::from_stored(
        [
            ("choice.small", 1.789_783_450_971_802),
            ("choice.mid", 1.608_568_500_166_153_2),
            ("choice.large", 1.664_111_261_733_098_6),
            ("choice", 1.766_665_953_967_977_9),
            ("score", 2.803_323_663_032_788),
            ("noul", 2.333_293_083_170_645_8),
        ]
        .map(|(k, v)| (k.to_owned(), stored(v))),
    )
    .unwrap();
    // The bucket is the option count: up to 8 small, up to 26 mid, else large; a name the file lacks
    // falls back to the bare type, and a type it lacks to 1.
    for (kind, n, want) in [
        (Kind::Choice, 2, 1.789_783_f32),
        (Kind::Choice, 8, 1.789_783),
        (Kind::Choice, 9, 1.608_569),
        (Kind::Choice, 26, 1.608_569),
        (Kind::Choice, 27, 1.664_111),
        (Kind::Choice, 255, 1.664_111),
        (Kind::Noul, 2, 2.333_293),
        (Kind::Score, 3, 2.803_324),
        (Kind::Score, 10, 2.803_324),
    ] {
        assert_eq!(
            temps.of(&POLICY, kind, n).to_bits(),
            want.to_bits(),
            "{} of {n} options",
            kind.word()
        );
    }
    let only_noul = Temperatures::from_stored([("noul".to_owned(), 2.0)]).unwrap();
    assert_eq!(only_noul.of(&POLICY, Kind::Choice, 3), 1.0);
    for bad in [0.0, -1.5, f32::NAN, f32::INFINITY, 0.000_000_4] {
        let e = Temperatures::from_stored([("x".to_owned(), bad)])
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("invalid decision temperature x") && e.contains("above 0"),
            "{e}"
        );
    }

    // The softmax average: one variant at T = 1 and T = 2, two variants that agree (the second shows
    // the options reversed, so its scores are read back reversed) and two that disagree and cancel.
    let ps = |v: &[Vec<f32>], t: f32| softmax_averaged(v, t).unwrap();
    let (hi, lo) = (0.880_797_077_977_882_3, 0.119_202_922_022_117_55);
    for (got, want) in ps(&[vec![2.0, 0.0]], 1.0).iter().zip([hi, lo]) {
        close(*got, want, 1e-15, "one variant, T = 1");
    }
    for (got, want) in ps(&[vec![2.0, 0.0]], 2.0)
        .iter()
        .zip([0.731_058_578_630_004_9, 0.268_941_421_369_995_1])
    {
        close(*got, want, 1e-15, "one variant, T = 2");
    }
    for (got, want) in ps(&[vec![2.0, 0.0], vec![0.0, 2.0]], 1.0)
        .iter()
        .zip([hi, lo])
    {
        close(*got, want, 1e-15, "two variants that agree");
    }
    for got in ps(&[vec![2.0, 0.0], vec![2.0, 0.0]], 1.0) {
        close(got, 0.5, 1e-15, "two variants that cancel");
    }
    close(
        ps(&[vec![1.0, 0.0, 0.0]], 1.0)[0],
        0.576_116_884_765_829_1,
        1e-15,
        "three options",
    );
    // A score that is not finite, or variants of other lengths, are refused by name.
    for (v, want) in [
        (vec![vec![0.0, f32::NAN]], "not finite"),
        (vec![vec![f32::INFINITY, 0.0]], "not finite"),
        (
            vec![vec![0.0, 1.0], vec![0.0]],
            "variant 1 has 1 scores, variant 0 has 2",
        ),
        (vec![], "no scores"),
    ] {
        let e = softmax_averaged(&v, 1.0).unwrap_err().to_string();
        assert!(e.contains(want), "{e}");
    }

    // The rating scale: uniform over 9 ratings is the middle of the scale; a rating of 8 at 20 above
    // the rest at T = 2 is nearly yes. A score's level is `Σ i · pᵢ`.
    assert_eq!(expected_rating(&ps(&[vec![0.0; 9]], 1.0)), 0.5);
    let mut peaked = vec![0.0f32; 9];
    peaked[8] = 20.0;
    close(
        expected_rating(&ps(&[peaked], 2.0)),
        0.999_795_774_490_658_9,
        1e-14,
        "a peaked scale",
    );
    assert_eq!(expected_level(&ps(&[vec![0.0; 3]], 1.0)), 1.0);

    // The body, with no temperature in the file (1): a tied choice goes to the first option in the
    // criteria's order (not the sorted one), a score's level over uniform levels is 1 and its
    // confidence clamps at 0, a noul over uniform ratings is one half; keys in llama.cpp's order.
    let req = request(
        r#"{"state":"s","questions":{
        "pick":{"type":"choice","instructions":"I","criteria":{"b":"Bee","a":"Ay"}},
        "lvl":{"type":"score","instructions":"I","criteria":["Low",null,"High"]},
        "yes":{"type":"noul","instructions":"I"}}}"#,
    );
    let m = model_of(labels_of(255), "{{ id }}", Temperatures::default());
    let asked = m.plan(&req).unwrap();
    let scores = vec![vec![0.0; 2], vec![0.0; 2], vec![0.0; 3], vec![0.0; 9]];
    let answers = m.answers(&req, &asked, &scores).unwrap();
    let body = dumps(&m.body("served", answers, 123), false);
    assert_eq!(
        body,
        concat!(
            r#"{"model":"served","answers":{"#,
            r#""pick":{"type":"choice","choice":"b","probabilities":{"b":0.5,"a":0.5},"confidence":0.0},"#,
            r#""lvl":{"type":"score","score":1.0,"legend":{"0":"Low","1":null,"2":"High"},"#,
            r#""probabilities":{"0":0.3333333333333333,"1":0.3333333333333333,"2":0.3333333333333333},"confidence":0.0},"#,
            r#""yes":{"type":"noul","noul":0.5}},"#,
            r#""usage":{"input_tokens":123,"output_tokens":0}}"#
        )
    );
    // With a temperature of 2 for choices (held as 2.0) and the second variant reversed: the
    // variants agree on `b`, whose probability is the T = 2 softmax's, and its confidence
    // (p − ½) / ½.
    let temps = Temperatures::from_stored([("choice".to_owned(), 2.0)]).unwrap();
    let m = model_of(labels_of(255), "{{ id }}", temps);
    let req = request(
        r#"{"state":"s","questions":{"pick":{"type":"choice","instructions":"I","criteria":{"b":"Bee","a":"Ay"}}}}"#,
    );
    let asked = m.plan(&req).unwrap();
    let answers = m
        .answers(&req, &asked, &[vec![2.0, 0.0], vec![0.0, 2.0]])
        .unwrap();
    let text = dumps(&answers, false);
    let v = json::parse(&text).unwrap();
    let pick = v.get("pick").unwrap();
    assert_eq!(pick.get("choice").and_then(Json::as_str), Some("b"));
    let p = pick.get("probabilities").unwrap();
    close(
        p.get("b").and_then(Json::as_f64).unwrap(),
        0.731_058_578_630_004_9,
        1e-15,
        "p(b)",
    );
    close(
        p.get("a").and_then(Json::as_f64).unwrap(),
        0.268_941_421_369_995_1,
        1e-15,
        "p(a)",
    );
    close(
        pick.get("confidence").and_then(Json::as_f64).unwrap(),
        0.462_117_157_260_009_8,
        1e-14,
        "confidence",
    );
    // A wrong number of prompts' scores, or a prompt of another length, is refused by name.
    for (s, want) in [
        (vec![vec![0.0; 2]], "scores for 2 prompts"),
        (vec![vec![0.0; 2], vec![0.0; 3]], "3 scores for 2 outputs"),
        (
            vec![vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]],
            "3 scores for 2 prompts",
        ),
    ] {
        let e = m.answers(&req, &asked, &s).unwrap_err().to_string();
        assert!(e.contains(want), "{e}");
    }
    // The label logit: the hidden state dotted with the label's row, f64 sum, f32 result.
    let hidden = [1.5f32, -2.0, 0.5, 4.0];
    assert_eq!(m.logits(&hidden, 3).unwrap(), vec![1.5, -2.0, 2.0]);
    assert!(
        m.logits(&hidden[..3], 2)
            .unwrap_err()
            .to_string()
            .contains("not 1 rows of 4")
    );
    assert!(
        m.logits(&hidden, 256)
            .unwrap_err()
            .to_string()
            .contains("256 outputs for 255 labels")
    );
}

/// The policy: a choice's options in the request's order and a noul's `false` first with no default
/// sentences, how many prompts and outputs each kind takes, the labels (the single-token codes, at most
/// 255), the template's input (labels by display position, the second variant reversed, keys sorted,
/// floats as llama.cpp's runtime writes them, no images), and the refusals by name.
#[test]
fn lev_policy() {
    // The labels: every code that is one token, to the 255th; a code of two tokens is skipped.
    let all = Labels::from_vocab(|_| vec![5]);
    assert_eq!(
        (all.len(), all.text(0), all.text(26), all.text(254)),
        (255, "A", "AA", "IU")
    );
    let mut n = 0;
    let skipping = Labels::from_vocab(|code| {
        n += 1;
        if code == "AA" { vec![1, 2] } else { vec![n] }
    });
    assert_eq!((skipping.text(25), skipping.text(26)), ("Z", "AB"));
    assert_eq!(Labels::codes().count(), 26 + 26 * 26);

    let template = "{{ id }}/{{ type }}/{{ instructions if instructions is string else instructions | tojson }}/\
        {{ state if state is string else state | tojson }}/\
        {% for o in options %}[{{ o.label }} {{ o.key }} {{ o.description | tojson }}]{% endfor %}/{{ images | length }}";
    let m = model_of(labels_of(255), template, Temperatures::default());
    let req = request(
        r#"{"state":{"b":1250.0,"a":{"y":0.1,"x":[3,null]},"é":"ü"},"questions":{
        "c":{"type":"choice","instructions":{"z":1,"a":"I"},"criteria":{"zeta":{"k":2.0,"a":1},"alpha":"A text","mid":null}},
        "one":{"type":"choice","instructions":"O","criteria":{"only":"The one"}},
        "n":{"type":"noul","instructions":"N?","criteria":{"true":"Yes."}},
        "s":{"type":"score","instructions":"S","criteria":["Low","Mid",null,"High"]}}}"#,
    );
    let asked = m.plan(&req).unwrap();
    let shape = |a: &Asked| {
        (
            a.options
                .iter()
                .map(|o| o.id.as_str())
                .collect::<Vec<_>>()
                .join(","),
            a.variants,
            a.outputs,
        )
    };
    assert_eq!(
        shape(&asked[0]),
        ("zeta,alpha,mid".to_owned(), 2, 3),
        "insertion order, two prompts"
    );
    assert_eq!(shape(&asked[1]), ("only".to_owned(), 1, 1));
    assert_eq!(
        shape(&asked[2]),
        ("false,true".to_owned(), 1, 9),
        "false first, nine ratings"
    );
    assert_eq!(shape(&asked[3]), ("0,1,2,3".to_owned(), 1, 4));
    assert_eq!(asked[2].options[0].description, None, "no default sentence");

    let state = r#"{"a": {"x": [3, null], "y": 0.1}, "b": 1250, "é": "ü"}"#;
    let want = [
        format!(
            r#"c/choice/{{"a": "I", "z": 1}}/{state}/[A zeta {{"a": 1, "k": 2}}][B alpha "A text"][C mid null]/0"#
        ),
        format!(
            r#"c/choice/{{"a": "I", "z": 1}}/{state}/[A mid null][B alpha "A text"][C zeta {{"a": 1, "k": 2}}]/0"#
        ),
        format!(r#"one/choice/O/{state}/[A only "The one"]/0"#),
        format!(r#"n/noul/N?/{state}/[A false null][B true "Yes."]/0"#),
        format!(r#"s/score/S/{state}/[A 0 "Low"][B 1 "Mid"][C 2 null][D 3 "High"]/0"#),
    ];
    let prompts = m.prompts(&req, &asked).unwrap();
    let got: Vec<&str> = prompts.iter().map(|p| p.text.as_str()).collect();
    assert_eq!(got, want.iter().map(String::as_str).collect::<Vec<_>>());
    assert_eq!(
        prompts
            .iter()
            .map(|p| (p.question, p.variant, p.outputs))
            .collect::<Vec<_>>(),
        [(0, 0, 3), (0, 1, 3), (1, 0, 1), (2, 0, 9), (3, 0, 4)]
    );

    // An integer past 64 bits is a float to the server (nlohmann), `1e+29`-ish under %g.
    let big = request(
        r#"{"state":{"n":123456789012345678901234567890,"i":-7},"questions":{"q":{"type":"noul","instructions":"I"}}}"#,
    );
    let asked = m.plan(&big).unwrap();
    assert_eq!(
        m.prompts(&big, &asked).unwrap()[0].text,
        r#"q/noul/I/{"i": -7, "n": 1.23457e+29}/[A false null][B true null]/0"#
    );

    // More options than labels, and a vocabulary with too few labels, are refused by name.
    let ten = model_of(labels_of(10), template, Temperatures::default());
    let wide = request(&format!(
        r#"{{"state":"s","questions":{{"q":{{"type":"choice","instructions":"I","criteria":{{{}}}}}}}}}"#,
        (0..11)
            .map(|i| format!(r#""k{i}":"d""#))
            .collect::<Vec<_>>()
            .join(",")
    ));
    let e = ten.plan(&wide).unwrap_err().to_string();
    assert_eq!(
        e,
        "q: too many options (11), this model supports at most 10"
    );
    let e = LabelModel::new(
        POLICY,
        ChatTemplate::parse("x").unwrap(),
        labels_of(8),
        vec![0.0; 32],
        4,
        Temperatures::default(),
    )
    .err()
    .expect("8 labels cannot hold a 9-rating scale")
    .to_string();
    assert!(
        e.contains("8 single-token labels, the model reads at least 9"),
        "{e}"
    );
    // The row's own facts: lev's decision type names its file, its policy is the one under test.
    assert_eq!(
        (lev::NAME, lev::DECISION_TYPE, lev::BACKBONES),
        ("lev", "lev", &["qwen35"][..])
    );
}
