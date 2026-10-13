//! `gate_lev_serve` — the decide seat of `bloomery-serve` on lev (`ggml-org/lev-GGUF`'s Q4_K_M, a
//! `qwen35` file that carries `qwen35.decision.type = lev`): the model's own head, no head flag, one
//! process at a time on the card the runner pins, its answers held to llama.cpp mainline's server
//! (`tools/ref/lev_ref.py`'s dump).
//!
//!     gate_lev_serve --model <lev gguf> --head <a Clef joint_head.safetensors>
//!                    --ref <$BLOOMERY_DATA/lev/4b/ref> --dir <dir> [--hf <repo[:quant]>]
//!
//! Every server is `bloomery-serve` beside this binary with `BLOOMERY_REF_MODEL` removed from its
//! environment (the file is named once, by `-m`). It holds, over the 14 requests of
//! `tools/ref/lev/{suite,edge}.jsonl` and the dump of the server that answered them:
//!
//! - `serves_its_row`: `-m <gguf> --port 0` (no head flag) prints its listening line naming the model
//!   file, the model file as the head and the row `lev`; `/props` names the row and its routes; the
//!   first request is a 200 whose `answers` hold every question and whose last key is `timings`, with
//!   `prompt_n` above 0; `usage.input_tokens` is the sum of the lengths of the ids of the prompts the
//!   mainline server built for it (read from the dump, not from our own count); `/v1/chat/completions`
//!   is a 404 (the seat serves its row's routes only);
//! - `speaks_llama_cpps_wire`: the answer's `model` is the server's name for the model (the file's
//!   name under `-m`); `/v1/models` lists it in `data` and `models`; `/props` carries the server's
//!   `engine` object; a request with an image is a 501 `not_supported_error`; a question with no
//!   `instructions` is a 400 naming it (llama.cpp's server refuses it too);
//! - `a_repeat_is_the_same_body`: the same request again is the same body byte for byte, `timings`
//!   taken out (each prompt runs from a reset);
//! - `tops_match_mainline`: every request's body against the dump's: the same shape (keys and their
//!   order, the legends, the usage, the number kinds) exactly, and every question's top option the
//!   same; each question's margin in the dump is printed (a margin under the backbone's noise is a
//!   near-tie, its top pinned as it is);
//! - `probabilities_within_band`: the largest |dp| over every probability of the 14 requests at most
//!   [`DP_BAND`];
//! - `hf_serves_its_row` (only with `--hf`: it fetches the file from the hub, a network read): `--hf
//!   <repo[:quant]> --port 0` and no head flag lists the same model file, head and row, and answers the
//!   first request with the body of the `-m` server's, byte for byte but the `model` (the hub's repo, as
//!   llama.cpp's server names a hub model) and `timings` (the same file through the same engine);
//! - the refusals, each a process that exits non-zero before it listens, its stderr naming why:
//!   `lev_with_head` (`-m <lev> --head <head>`), `decide_with_head` (`--model decide -m <lev> --head
//!   <head>`), `kev_file` (a GGUF of arch `qwen35` and `qwen35.decision.type` kev, written into
//!   `<dir>`), `qwen35_with_no_head` (one with no decision keys: the refusal now lists lev beside
//!   clef), and `help_lists_lev` (`--help`'s models table has lev's `--hf` line).
//!
//! Logs per server in `<dir>/<clause>/` (`server.out`, `server.err`, the bodies).

#[cfg(not(all(feature = "deepseek41", feature = "clef")))]
fn main() {
    eprintln!(
        "gate_lev_serve: built without the `deepseek41` and `clef` features; see `just gate-gpu-lev-serve`."
    );
    std::process::exit(2);
}

#[cfg(all(feature = "deepseek41", feature = "clef"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_lev_serve", gate::run())
}

// The decide seat's card gates' shared helpers.
#[cfg(all(feature = "deepseek41", feature = "clef"))]
#[path = "shared/decide_gate.rs"]
mod decide_gate;

#[cfg(all(feature = "deepseek41", feature = "clef"))]
mod gate {
    use std::path::{Path, PathBuf};

    use bloomery_gpu_gates::serve_client::curl;
    use bloomery_gpu_gates::{GateError, checks_failed};
    use serde_json::Value;

    use crate::decide_gate::{check, compare, listening, post_raw, refused, spawn, untimed, utf8};

    const USAGE: &str = "usage: gate_lev_serve --model <lev gguf> --head <joint_head.safetensors> \\
                         --ref <dir> --dir <dir> [--hf <repo[:quant]>]";

    /// The suite and edge requests: the 14 the dump answers.
    const REQUEST_FILES: [&str; 2] = ["suite.jsonl", "edge.jsonl"];

    /// PIN(2026-10-08): the band on the largest |dp|, ours against mainline's server, over every
    /// probability of the 14 requests. Both read the same Q4_K_M codes and differ in the backbone's
    /// 8-bit activations and f32 sum order (the hidden-state gate's median relative distance is
    /// 0.045), which reads 0.0632 at the worst probability (`sentiment-07`'s `bug_report`). 0.25 is
    /// 4 times that. A variant not mapped back to its options' order (a second prompt that showed
    /// them reversed) reads 0.39 to 0.49 on the same requests, 1.6 times the band and more.
    const DP_BAND: f64 = 0.25;

    /// The models table line `bloomery-serve --help` prints for lev
    /// ([`drive::MODELS`] in `bloomery_serve.rs`).
    const HELP_LINE: &str = "decide  a decision model in its file --hf ggml-org/lev-GGUF:Q4_K_M";

    struct Args {
        model: PathBuf,
        head: PathBuf,
        reference: PathBuf,
        dir: PathBuf,
        /// The hub spec of the file, when the `--hf` clause runs.
        hf: Option<String>,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut model, mut head, mut reference, mut dir) = (None, None, None, None);
        let mut hf = None;
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(v)),
                "--head" => head = Some(PathBuf::from(v)),
                "--ref" => reference = Some(PathBuf::from(v)),
                "--dir" => dir = Some(PathBuf::from(v)),
                "--hf" => hf = Some(v),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (model, head, reference, dir) {
            (Some(model), Some(head), Some(reference), Some(dir)) => Ok(Args {
                model,
                head,
                reference,
                dir,
                hf,
            }),
            _ => Err(USAGE.into()),
        }
    }

    /// One request of the suite or edge file: its id, its text as the server is posted it (the line
    /// from `"model"` on, the `id` ours), and its questions in order.
    struct Request {
        id: String,
        text: String,
        questions: Vec<String>,
    }

    fn requests() -> Result<Vec<Request>, GateError> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/lev");
        let mut out = Vec::new();
        for file in REQUEST_FILES {
            let path = dir.join(file);
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                let at = line
                    .find("\"model\"")
                    .ok_or_else(|| format!("{file}: a request names no model"))?;
                let text = format!("{{{}", &line[at..]);
                let v: Value = serde_json::from_str(&text)?;
                let whole: Value = serde_json::from_str(line)?;
                out.push(Request {
                    id: whole["id"]
                        .as_str()
                        .ok_or_else(|| format!("{file}: a request has no id"))?
                        .to_owned(),
                    questions: v["questions"]
                        .as_object()
                        .ok_or("a request holds no questions")?
                        .keys()
                        .cloned()
                        .collect(),
                    text,
                });
            }
        }
        Ok(out)
    }

    /// The dump's row of every request: its ids' count and its body.
    struct Reference {
        n_tokens: u64,
        body: Value,
    }

    fn reference(dir: &Path) -> Result<std::collections::HashMap<String, Reference>, GateError> {
        let path = dir.join("reference.jsonl");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut out = std::collections::HashMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let row: Value = serde_json::from_str(line)?;
            let id = row["id"].as_str().ok_or("a dump row has no id")?.to_owned();
            let n_tokens = row["tasks"]
                .as_array()
                .ok_or("a dump row has no tasks")?
                .iter()
                .map(|t| t["ids"].as_array().map_or(0, |ids| ids.len() as u64))
                .sum();
            out.insert(
                id,
                Reference {
                    n_tokens,
                    body: row["response"].clone(),
                },
            );
        }
        Ok(out)
    }

    /// A GGUF of arch `qwen35` and its `embedding_length`, with `qwen35.decision.type` when `kind`
    /// names one, written into `dir` as `name`: a file the seat refuses on its metadata, so no tensor
    /// is needed.
    fn header_only(dir: &Path, name: &str, kind: Option<&str>) -> Result<PathBuf, GateError> {
        let mut kvs = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_owned(),
                gguf::Value::String("qwen35".to_owned()),
            ),
            ("qwen35.embedding_length".to_owned(), gguf::Value::U32(16)),
        ];
        if let Some(k) = kind {
            kvs.push((
                "qwen35.decision.type".to_owned(),
                gguf::Value::String(k.to_owned()),
            ));
        }
        let layout = gguf::write::Layout::new(&kvs, Vec::new())?;
        let path = dir.join(name);
        let file = std::fs::File::create(&path)?;
        gguf::write::Writer::new(file, layout)?.finish()?;
        Ok(path)
    }

    /// A body's structure: its keys in their order, each leaf as its kind (an integer by its digits, a
    /// float as `f`, a string by its text, except a choice's `choice`, which is a top and compared as
    /// one). Two bodies of one shape differ only in their floats and in the option a choice picks.
    fn shape(v: &Value, key: &str, out: &mut String) {
        match v {
            Value::Null => out.push_str("null"),
            Value::Bool(b) => out.push_str(&b.to_string()),
            Value::Number(n) if n.is_f64() => out.push('f'),
            Value::Number(n) => out.push_str(&n.to_string()),
            Value::String(_) if key == "choice" => out.push('s'),
            Value::String(s) => out.push_str(&format!("{s:?}")),
            Value::Array(a) => {
                out.push('[');
                for x in a {
                    shape(x, key, out);
                    out.push(',');
                }
                out.push(']');
            }
            Value::Object(o) => {
                out.push('{');
                for (k, x) in o {
                    out.push_str(&format!("{k:?}:"));
                    shape(x, k, out);
                    out.push(',');
                }
                out.push('}');
            }
        }
    }

    /// `body` with the `timings` object the server appends taken out, as a value.
    fn untimed_value(body: &str) -> Result<Value, GateError> {
        let Some(text) = untimed(body) else {
            return Err("the body has no trailing timings".into());
        };
        Ok(serde_json::from_str(&format!("{text}}}"))?)
    }

    /// A question's margin in an answer: a noul's `|2p − 1|`, else the top probability less the
    /// runner-up's.
    fn margin(a: &Value) -> f64 {
        if a["type"] == "noul" {
            return a["noul"].as_f64().map_or(0.0, |p| (2.0 * p - 1.0).abs());
        }
        let mut p: Vec<f64> = a["probabilities"]
            .as_object()
            .map(|o| o.values().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        p.sort_by(|x, y| y.total_cmp(x));
        p[0] - p.get(1).copied().unwrap_or(0.0)
    }

    pub fn run() -> Result<(), GateError> {
        let a = parse_args()?;
        std::fs::create_dir_all(&a.dir)?;
        // The shapes compare the order of keys: serde_json must keep it (the decision crate asks for
        // `preserve_order`, and cargo unifies it in).
        let ordered: Value = serde_json::from_str(r#"{"b":1,"a":2}"#)?;
        if ordered
            .as_object()
            .map(|o| o.keys().cloned().collect::<Vec<_>>())
            != Some(vec!["b".to_owned(), "a".to_owned()])
        {
            return Err("serde_json does not keep an object's key order here".into());
        }
        let (model, head) = (utf8(&a.model)?, utf8(&a.head)?);
        let file = a.model.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let reqs = requests()?;
        let refs = reference(&a.reference)?;
        if reqs.len() != 14 || reqs.iter().any(|r| !refs.contains_key(&r.id)) {
            return Err(format!(
                "the dump {} answers {} of the {} requests (tools/ref/lev_ref.py)",
                a.reference.display(),
                reqs.iter().filter(|r| refs.contains_key(&r.id)).count(),
                reqs.len()
            )
            .into());
        }
        let mut ok = true;

        // The refusals first: none of them loads the backbone.
        let pass = refused(
            &a.dir,
            "lev_with_head",
            &["-m", model, "--head", head],
            &["beside a lev file: its head is the model file's own", head],
        )?;
        check(&mut ok, "lev_with_head", pass);
        let pass = refused(
            &a.dir,
            "decide_with_head",
            &["--model", "decide", "-m", model, "--head", head],
            &["beside a lev file: its head is the model file's own"],
        )?;
        check(&mut ok, "decide_with_head", pass);
        let kev = header_only(&a.dir, "kev.gguf", Some("kev"))?;
        let pass = refused(
            &a.dir,
            "kev_file",
            &["-m", utf8(&kev)?],
            &[
                "a qwen35 file whose qwen35.decision.type is kev",
                "this server serves no such decision model; it serves lev by decision type and clef by architecture",
            ],
        )?;
        check(&mut ok, "kev_file", pass);
        let bare = header_only(&a.dir, "no-decision.gguf", None)?;
        let pass = refused(
            &a.dir,
            "qwen35_with_no_head",
            &["-m", utf8(&bare)?],
            &[
                "a qwen35 file with no head",
                "clef (head repo Cloudflare/clef-flash, backbone qwen35; --hf \
                 bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M)",
                "lev (the head is the model file's own: a qwen35 file whose decision type is lev, with no --head)",
            ],
        )?;
        check(&mut ok, "qwen35_with_no_head", pass);
        let pass = refused(&a.dir, "help_lists_lev", &["--help"], &[HELP_LINE])?;
        check(&mut ok, "help_lists_lev", pass);

        // The seat serving its row, one server for the clauses that follow.
        let (mut s, d) = spawn(&a.dir, "serve", &["-m", model, "--port", "0"])?;
        let line = listening(&mut s, &d.join("server.err"))?;
        println!("{line}");
        let addr = line
            .split_once("listening on http://")
            .and_then(|(_, r)| r.split_whitespace().next())
            .ok_or("no address in the listening record")?
            .to_owned();
        let url = |p: &str| format!("http://{addr}{p}");
        let names = |w: &str| line.contains(w);
        let mut serves =
            names(&format!("model={file}")) && names(&format!("head={file}")) && names("row=lev");
        let (st, props) = curl(&url("/props"), None, false)?;
        let props: Value = serde_json::from_str(&props)?;
        println!("/props {st} {props}");
        serves &= st == 200
            && props["row"] == "lev"
            && props["head"] == file
            && props["routes"] == serde_json::json!(["/v1/systemone"]);
        let first = &reqs[0];
        let (st, body) = post_raw(&url("/v1/systemone"), &first.text)?;
        std::fs::write(d.join("first.json"), &body)?;
        let v: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let answered = first
            .questions
            .iter()
            .all(|q| v["answers"].get(q).is_some());
        let timed =
            untimed(&body).is_some() && v["timings"]["prompt_n"].as_u64().is_some_and(|n| n > 0);
        let want = refs[&first.id].n_tokens;
        let usage = v["usage"]["input_tokens"].as_u64();
        println!(
            "/v1/systemone {st}: {} answered {answered}, timings last {timed}, usage.input_tokens {usage:?} = the dump's prompts' {want} ids",
            first.id
        );
        serves &= st == 200 && answered && timed && usage == Some(want);
        let (st, _) = post_raw(&url("/v1/chat/completions"), "{}")?;
        println!("/v1/chat/completions {st}");
        serves &= st == 404;
        check(&mut ok, "serves_its_row", serves);

        let mut wire = v["model"] == file && props["engine"]["name"] == "bloomery";
        println!(
            "answer model {}, /props engine {}",
            v["model"], props["engine"]
        );
        let (st, models) = curl(&url("/v1/models"), None, false)?;
        let models: Value = serde_json::from_str(&models)?;
        println!("/v1/models {st} {models}");
        wire &= st == 200 && models["data"][0]["id"] == file && models["models"][0]["name"] == file;
        let image = first.text.replacen(
            "{",
            r#"{"images": ["data:image/png;base64,iVBORw0KGgo="], "#,
            1,
        );
        let (st, refused_image) = post_raw(&url("/v1/systemone"), &image)?;
        println!("/v1/systemone with an image {st} {refused_image}");
        let refused_image: Value = serde_json::from_str(&refused_image).unwrap_or(Value::Null);
        wire &= st == 501 && refused_image["error"]["type"] == "not_supported_error";
        let bare_question =
            r#"{"model": "lev", "state": "s", "questions": {"q": {"type": "noul"}}}"#;
        let (st, refused_q) = post_raw(&url("/v1/systemone"), bare_question)?;
        println!("/v1/systemone with no instructions {st} {refused_q}");
        let refused_q: Value = serde_json::from_str(&refused_q).unwrap_or(Value::Null);
        wire &= st == 400
            && refused_q["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains("q: \"instructions\" must be provided"));
        check(&mut ok, "speaks_llama_cpps_wire", wire);

        let (st, again) = post_raw(&url("/v1/systemone"), &first.text)?;
        std::fs::write(d.join("again.json"), &again)?;
        let same = st == 200 && untimed(&body).is_some() && untimed(&body) == untimed(&again);
        check(&mut ok, "a_repeat_is_the_same_body", same);

        // Every request against the dump.
        let (mut shapes, mut tops, mut worst) = (true, true, 0f64);
        for r in &reqs {
            let (st, text) = post_raw(&url("/v1/systemone"), &r.text)?;
            std::fs::write(d.join(format!("{}.json", r.id)), &text)?;
            if st != 200 {
                println!("{}: HTTP {st}: {text}", r.id);
                shapes = false;
                continue;
            }
            let ours = untimed_value(&text)?;
            let theirs = &refs[&r.id].body;
            let (mut sa, mut sb) = (String::new(), String::new());
            shape(&ours, "", &mut sa);
            shape(theirs, "", &mut sb);
            let same_shape = sa == sb;
            let (same_tops, dp) = compare(&r.id, &ours, theirs, &r.questions);
            for q in &r.questions {
                println!(
                    "{}: {q}: dump margin {:.4}, ours {:.4}",
                    r.id,
                    margin(&theirs["answers"][q]),
                    margin(&ours["answers"][q])
                );
            }
            println!(
                "{}: usage {} / {} shape {} tops {same_tops} max |dp| {dp:.4}",
                r.id,
                ours["usage"]["input_tokens"],
                theirs["usage"]["input_tokens"],
                if same_shape { "equal" } else { "DIFFERENT" }
            );
            if !same_shape {
                println!("  ours  {sa}\n  theirs {sb}");
            }
            shapes &= same_shape;
            tops &= same_tops;
            worst = worst.max(dp);
        }
        println!("max |dp| over every question of the 14 requests {worst:.4} (band {DP_BAND})");
        check(&mut ok, "tops_match_mainline", shapes && tops);
        check(&mut ok, "probabilities_within_band", worst <= DP_BAND);
        println!("server stopped: {}", s.stop()?);

        if let Some(spec) = &a.hf {
            let (mut h, d) = spawn(&a.dir, "hf", &["--hf", spec, "--port", "0"])?;
            let line = listening(&mut h, &d.join("server.err"))?;
            println!("{line}");
            let addr = line
                .split_once("listening on http://")
                .and_then(|(_, r)| r.split_whitespace().next())
                .ok_or("no address in the listening record")?
                .to_owned();
            let named = line.contains(&format!("model={file}"))
                && line.contains(&format!("head={file}"))
                && line.contains("row=lev");
            let (st, again) = post_raw(&format!("http://{addr}/v1/systemone"), &reqs[0].text)?;
            std::fs::write(d.join("first.json"), &again)?;
            // The server names a hub model by its repo (`ggml-org/lev-GGUF`), a file by its name.
            let repo = spec.split_once(':').map_or(spec.as_str(), |(r, _)| r);
            let by_repo = again.replacen(
                &format!("\"model\":\"{repo}\""),
                &format!("\"model\":\"{file}\""),
                1,
            );
            let same = st == 200
                && by_repo != again
                && untimed(&body).is_some()
                && untimed(&body) == untimed(&by_repo);
            println!(
                "--hf {spec}: the first request {st}, the answer names the model {repo:?}, and its \
                 body is the -m server's, `model` and `timings` taken out: {same}"
            );
            println!("server stopped: {}", h.stop()?);
            check(&mut ok, "hf_serves_its_row", named && same);
        }

        if ok {
            println!("gate_lev_serve: all checks passed");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
