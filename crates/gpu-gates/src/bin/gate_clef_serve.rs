//! `gate_clef_serve` — the decide seat of `bloomery-serve` on Clef-Flash: a
//! `qwen35` file with the release's head, and a `clef` file (llama.cpp's
//! layout, the head inside the GGUF), one process at a time on the card the
//! runner pins.
//!
//!     gate_clef_serve --model <gguf> --head <joint_head.safetensors>
//!                     --clef-model <gguf> --dir <dir>
//!
//! Every server is `bloomery-serve` beside this binary with
//! `BLOOMERY_REF_MODEL` removed from its environment (the file is named
//! once, by `-m`). It holds:
//!
//! - `serves_its_row`: `-m <gguf> --head <head> --port 0` prints its
//!   listening line on stdout naming the model, the head and the row `clef`;
//!   `/props` names the row and its routes; the first request of
//!   `tools/ref/clef/suite.jsonl` POSTed to `/v1/systemone` is a 200 whose
//!   `answers` hold every question of the request and whose last key is
//!   `timings`, with `prompt_n` above 0; `/v1/chat/completions` is a 404
//!   (the seat serves its row's routes only);
//! - `speaks_llama_cpps_wire` (llama.cpp's `/v1/systemone`, its server at
//!   `a4cb4c61`): the answer's `model` is the server's name for the model
//!   (the file's name under `-m`), not the request's; `/v1/models` lists
//!   that name in `data` and `models`; `/props` carries the server's
//!   `engine` object; a request with an image is a 501
//!   `not_supported_error`;
//! - `a_repeat_is_the_same_body`: the same request again is the same body
//!   byte for byte, `timings` taken out (each request runs from a reset);
//! - the face both constants of `bloomery-serve`'s drive print
//!   ([`face_pins_the_constants`]): `--help` exits non-zero with the models
//!   table on its stderr, one line a seat with its `--hf` example exactly as
//!   the gate pins it, and a start that passes the help branch prints the
//!   first-start note before it refuses (here `--model qwen3` with no model
//!   file, refused before any load);
//! - the refusals, each a process that exits non-zero before it listens,
//!   its stderr naming why: `qwen35_with_no_head` (`-m <gguf>`),
//!   `decide_with_no_head` (`--model decide -m <gguf>`), `qwen3_with_head`
//!   (`--model qwen3 -m <gguf> --head <head>`), `unknown_head_config`
//!   (`--head-config` of a config no row knows), `narrow_head` (a head of
//!   hidden width 16, written into `<dir>`, against the file's
//!   `embedding_length`: refused before the backbone loads),
//!   `clef_layout_without_its_head` (a GGUF of arch `clef` and no decision
//!   head, written into `<dir>`, is refused for it: the file carries no
//!   head, and the refusal is before the backbone loads);
//! - the clef layout (`--clef-model`, the same Clef-Flash quantization
//!   bartowski publishes in llama.cpp's layout), each server alone on the
//!   card after the `qwen35` one stopped:
//!   `clef_layout_serves_its_head` (`-m <clef gguf>` with no `--head` prints
//!   its listening line naming the model file as the head and the row
//!   `clef`; `/props` names them; the first request answers every question
//!   with `timings` last) and `clef_layout_head_is_the_release_head` (the
//!   same server's first-request answers against `-m <clef gguf> --head
//!   <release head>`: the one difference between the two is the head, the
//!   file's Q8_0 matrices against the release's bf16, so every question's
//!   chosen option is the same and every probability within
//!   [`HEAD_DTYPE_BAND`]). The `qwen35` file's answers against the clef
//!   file's are printed beside them as a diagnostic: the two files are
//!   separate quantizations of the backbone.
//!
//! Logs per server in `<dir>/<clause>/` (`server.out`, `server.err`, the
//! bodies).

#[cfg(not(all(feature = "deepseek41", feature = "clef")))]
fn main() {
    eprintln!(
        "gate_clef_serve: built without the `deepseek41` and `clef` features; see `just gate-gpu-clef-serve`."
    );
    std::process::exit(2);
}

#[cfg(all(feature = "deepseek41", feature = "clef"))]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_clef_serve", gate::run())
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
    use decision::head::HeadConfig;
    use decision::safetensors::write;
    use serde_json::Value;

    use crate::decide_gate::{check, compare, listening, post_raw, refused, spawn, untimed, utf8};

    const USAGE: &str = "usage: gate_clef_serve --model <gguf> --head <joint_head.safetensors> \
                         --clef-model <gguf> --dir <dir>";

    struct Args {
        model: PathBuf,
        head: PathBuf,
        clef: PathBuf,
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut model, mut head, mut clef, mut dir) = (None, None, None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(v)),
                "--head" => head = Some(PathBuf::from(v)),
                "--clef-model" => clef = Some(PathBuf::from(v)),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (model, head, clef, dir) {
            (Some(model), Some(head), Some(clef), Some(dir)) => Ok(Args {
                model,
                head,
                clef,
                dir,
            }),
            _ => Err(USAGE.into()),
        }
    }

    /// The first request of the English suite, without its `id`, as text.
    fn first_request() -> Result<(String, Vec<String>), GateError> {
        let suite = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/clef/suite.jsonl");
        let text =
            std::fs::read_to_string(&suite).map_err(|e| format!("{}: {e}", suite.display()))?;
        let line = text.lines().next().ok_or("the suite holds no request")?;
        let at = line
            .find("\"model\"")
            .ok_or("the suite's first request names no model")?;
        // The line opens `{"id": "...", "model": ...`: the id out, the rest as written.
        let body = format!("{{{}", &line[at..]);
        let v: Value = serde_json::from_str(&body)?;
        let questions = v["questions"]
            .as_object()
            .ok_or("the first request holds no questions")?
            .keys()
            .cloned()
            .collect();
        Ok((body, questions))
    }

    /// A GGUF whose metadata names arch `clef` and its `embedding_length`
    /// and holds nothing else, written into `dir`: a clef file with no
    /// decision head, which the seat refuses on reading its keys, so no
    /// tensor is needed.
    fn clef_layout_file(dir: &Path) -> Result<PathBuf, GateError> {
        let kvs = vec![
            (
                gguf::GENERAL_ARCHITECTURE.to_owned(),
                gguf::Value::String("clef".to_owned()),
            ),
            ("clef.embedding_length".to_owned(), gguf::Value::U32(16)),
        ];
        let layout = gguf::write::Layout::new(&kvs, Vec::new())?;
        let path = dir.join("clef-no-head.gguf");
        let file = std::fs::File::create(&path)?;
        gguf::write::Writer::new(file, layout)?.finish()?;
        Ok(path)
    }

    /// A head of hidden width 16 and its config, written into `dir`: zero
    /// weights of every shape its config names.
    fn narrow_head(dir: &Path) -> Result<PathBuf, GateError> {
        let config = r#"{"hidden_size": 16, "width": 8, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 12}"#;
        let cfg = HeadConfig::parse(config)?;
        let weights = cfg.weights();
        let bytes: Vec<Vec<u8>> = weights
            .iter()
            .map(|(_, shape)| vec![0u8; 4 * shape.iter().product::<usize>()])
            .collect();
        let list: Vec<(&str, &str, &[usize], &[u8])> = weights
            .iter()
            .zip(&bytes)
            .map(|((n, s), b)| (n.as_str(), "F32", s.as_slice(), b.as_slice()))
            .collect();
        let d = dir.join("narrow-head");
        std::fs::create_dir_all(&d)?;
        std::fs::write(d.join("joint_head_config.json"), config)?;
        let head = d.join("joint_head.safetensors");
        std::fs::write(&head, write(&list))?;
        Ok(head)
    }

    /// The models table `bloomery-serve --help` prints beside its usage
    /// ([`drive::MODELS`] in `bloomery_serve.rs`): one line a seat, the seat
    /// word, what it serves and a working `--hf` example of it, and the
    /// header line above them.
    const MODELS_LINES: [&str; 6] = [
        "models (the --model word is optional with a model file: the file's architecture picks \
         the seat)",
        "ds41    DeepSeek-V4.1-Flash          --hf vcruz305/DeepSeek-V4.1-Flash-GGUF:Q3_K_M",
        "qwen38  Qwen3.8-Flash-Next           --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL",
        "glm     GLM-5.3-Flash                --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL",
        "qwen3   Qwen3-30B-A3B and Qwen3.6    --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M",
        "decide  a decision model by its head --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M",
    ];

    /// The note every start that passes the help branch prints on stderr
    /// before the load begins ([`drive::FIRST_START`] in `bloomery_serve.rs`).
    const FIRST_START: &str = "bloomery-serve: a first start on a new card compiles the GPU \
                               code for it (tens of seconds); later starts take seconds";

    /// The face (module header): the two constant lines of the drive, read
    /// from what the binary printed. `--help` exits non-zero with the usage
    /// and the models table on stderr, every line exactly as
    /// [`MODELS_LINES`] pins it; a start that passes the help branch — here
    /// `--model qwen3` with no model file, which refuses before any load —
    /// prints the first-start note first. Neither process opens a card or
    /// reads a model.
    fn face_pins_the_constants(dir: &Path) -> Result<bool, GateError> {
        let helps = refused(
            dir,
            "help_prints_the_models_table",
            &["--help"],
            &MODELS_LINES,
        )?;
        let starts = refused(
            dir,
            "every_start_notes_the_first_compile",
            &["--model", "qwen3"],
            &[FIRST_START],
        )?;
        Ok(helps && starts)
    }

    /// The band of one answer's probability between the clef file's own head (Q8_0 matrices) and
    /// the release's (bf16): the head is the only difference between the two servers, which run one
    /// file on one card.
    ///
    /// PIN(2026-10-07): 2.5e-3, the `LAYOUT_PROB_BAND` of `crates/decision/tests/clef.rs`, which
    /// derives it from the logit band (a probability moves by at most a quarter of the difference
    /// of two logits' moves) and measures the head alone on the eight reference requests: the worst
    /// probability moves 2.4e-4 (`route-01`, this gate's request, the worst of them), the band is
    /// ten times that.
    const HEAD_DTYPE_BAND: f64 = 2.5e-3;

    /// One server's first-request round: the listening line, `/props`, the first request's status
    /// and body.
    struct Once {
        line: String,
        props: Value,
        status: u16,
        body: String,
    }

    /// `bloomery-serve` with `args` in `<dir>/<name>/`, its `/props` and `req` posted to
    /// `/v1/systemone`, then stopped.
    fn serve_once(dir: &Path, name: &str, args: &[&str], req: &str) -> Result<Once, GateError> {
        let (mut s, d) = spawn(dir, name, args)?;
        let line = listening(&mut s, &d.join("server.err"))?;
        println!("{line}");
        let addr = line
            .split_once("listening on http://")
            .and_then(|(_, r)| r.split_whitespace().next())
            .ok_or("no address in the listening record")?
            .to_owned();
        let (st, props) = curl(&format!("http://{addr}/props"), None, false)?;
        let props: Value = serde_json::from_str(&props)?;
        println!("{name}: /props {st} {props}");
        let (status, body) = post_raw(&format!("http://{addr}/v1/systemone"), req)?;
        std::fs::write(d.join("first.json"), &body)?;
        println!("{name}: server stopped: {}", s.stop()?);
        Ok(Once {
            line,
            props,
            status,
            body,
        })
    }

    /// Whether the request was a 200 answering every one of `questions`, `timings` last.
    fn whole(o: &Once, questions: &[String]) -> bool {
        let v: Value = serde_json::from_str(&o.body).unwrap_or(Value::Null);
        o.status == 200
            && questions.iter().all(|q| v["answers"].get(q).is_some())
            && untimed(&o.body).is_some()
            && v["timings"]["prompt_n"].as_u64().is_some_and(|n| n > 0)
    }

    pub fn run() -> Result<(), GateError> {
        let a = parse_args()?;
        std::fs::create_dir_all(&a.dir)?;
        let (model, head) = (utf8(&a.model)?, utf8(&a.head)?);
        let mut ok = true;

        let pass = face_pins_the_constants(&a.dir)?;
        check(&mut ok, "face_pins_the_constants", pass);

        // The refusals first: none of them loads the backbone.
        let pass = refused(
            &a.dir,
            "qwen35_with_no_head",
            &["-m", model],
            &[
                "a qwen35 file with no head",
                "--head",
                "clef (head repo Cloudflare/clef-flash, backbone qwen35; --hf \
                 bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M)",
            ],
        )?;
        check(&mut ok, "qwen35_with_no_head", pass);
        let pass = refused(
            &a.dir,
            "decide_with_no_head",
            &["--model", "decide", "-m", model],
            &["--model decide: a qwen35 file with no head"],
        )?;
        check(&mut ok, "decide_with_no_head", pass);
        let pass = refused(
            &a.dir,
            "qwen3_with_head",
            &["--model", "qwen3", "-m", model, "--head", head],
            &["--model qwen3 with --head"],
        )?;
        check(&mut ok, "qwen3_with_head", pass);
        let other = a.dir.join("other_head_config.json");
        std::fs::write(&other, r#"{"num_labels": 2}"#)?;
        let pass = refused(
            &a.dir,
            "unknown_head_config",
            &["-m", model, "--head", head, "--head-config", utf8(&other)?],
            &[
                "no decision model knows the head",
                "clef: ",
                "unknown keys [\"num_labels\"]",
            ],
        )?;
        check(&mut ok, "unknown_head_config", pass);
        let narrow = narrow_head(&a.dir)?;
        let pass = refused(
            &a.dir,
            "narrow_head",
            &["-m", model, "--head", utf8(&narrow)?],
            &["the head reads hidden states of width 16"],
        )?;
        check(&mut ok, "narrow_head", pass);
        // PIN(2026-10-07): the clause that held llama.cpp's Clef layout refused ("does not serve
        // yet", with bartowski's `--hf` line) is this one. The layout is served now (bartowski's
        // current Q5_K_M is in it, and `--hf bartowski/Cloudflare_clef-flash-GGUF` fetches it), so
        // a file of arch `clef` is refused for what it can lack: its head. The header-only file
        // has no `clef.decision.type`.
        let clef_bare = clef_layout_file(&a.dir)?;
        let pass = refused(
            &a.dir,
            "clef_layout_without_its_head",
            &["-m", utf8(&clef_bare)?],
            &[
                "the head in the model file",
                "clef.decision.type is absent",
                "the file carries no decision head",
            ],
        )?;
        check(&mut ok, "clef_layout_without_its_head", pass);

        // The seat serving its row.
        let (mut s, d) = spawn(
            &a.dir,
            "serve",
            &["-m", model, "--head", head, "--port", "0"],
        )?;
        let line = listening(&mut s, &d.join("server.err"))?;
        println!("{line}");
        let addr = line
            .split_once("listening on http://")
            .and_then(|(_, r)| r.split_whitespace().next())
            .ok_or("no address in the listening record")?
            .to_owned();
        let url = |p: &str| format!("http://{addr}{p}");
        let names = |w: &str| line.contains(w);
        let mut serves = names(&format!(
            "model={}",
            a.model.file_name().and_then(|n| n.to_str()).unwrap_or("?")
        )) && names("head=joint_head.safetensors")
            && names("row=clef");
        let (st, props) = curl(&url("/props"), None, false)?;
        let props: Value = serde_json::from_str(&props)?;
        println!("/props {st} {props}");
        serves &= st == 200
            && props["row"] == "clef"
            && props["routes"] == serde_json::json!(["/v1/systemone"]);
        let (req, questions) = first_request()?;
        let (st, first) = post_raw(&url("/v1/systemone"), &req)?;
        std::fs::write(d.join("first.json"), &first)?;
        let v: Value = serde_json::from_str(&first).unwrap_or(Value::Null);
        let answered = questions.iter().all(|q| v["answers"].get(q).is_some());
        let timed =
            untimed(&first).is_some() && v["timings"]["prompt_n"].as_u64().is_some_and(|n| n > 0);
        println!("/v1/systemone {st}: every question answered {answered}, timings last {timed}");
        serves &= st == 200 && answered && timed;
        let (st, _) = post_raw(&url("/v1/chat/completions"), "{}")?;
        println!("/v1/chat/completions {st}");
        serves &= st == 404;
        check(&mut ok, "serves_its_row", serves);
        let file = a.model.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let mut wire = v["model"] == file && props["engine"]["name"] == "bloomery";
        println!(
            "answer model {}, /props engine {}",
            v["model"], props["engine"]
        );
        let (st, models) = curl(&url("/v1/models"), None, false)?;
        let models: Value = serde_json::from_str(&models)?;
        println!("/v1/models {st} {models}");
        wire &= st == 200 && models["data"][0]["id"] == file && models["models"][0]["name"] == file;
        let image = req.replacen(
            "{",
            r#"{"images": ["data:image/png;base64,iVBORw0KGgo="], "#,
            1,
        );
        let (st, refused) = post_raw(&url("/v1/systemone"), &image)?;
        println!("/v1/systemone with an image {st} {refused}");
        let refused: Value = serde_json::from_str(&refused).unwrap_or(Value::Null);
        wire &= st == 501 && refused["error"]["type"] == "not_supported_error";
        check(&mut ok, "speaks_llama_cpps_wire", wire);
        let (st, again) = post_raw(&url("/v1/systemone"), &req)?;
        std::fs::write(d.join("again.json"), &again)?;
        let same = st == 200 && untimed(&first).is_some() && untimed(&first) == untimed(&again);
        check(&mut ok, "a_repeat_is_the_same_body", same);
        println!("server stopped: {}", s.stop()?);

        // The clef layout: the same first request to the file alone (its own head) and to the file
        // with the release's head. Same card, same backbone bytes, same prompt pass: the two
        // answers differ by the head alone.
        let clef = utf8(&a.clef)?;
        let clef_file_name = a.clef.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let own = serve_once(&a.dir, "clef_layout", &["-m", clef, "--port", "0"], &req)?;
        let mut serves = own.line.contains(&format!("model={clef_file_name}"))
            && own.line.contains(&format!("head={clef_file_name}"))
            && own.line.contains("row=clef")
            && own.props["row"] == "clef"
            && own.props["head"] == clef_file_name
            && own.props["routes"] == serde_json::json!(["/v1/systemone"]);
        let own_body = whole(&own, &questions);
        println!(
            "clef layout: head {} in /props, answered whole {own_body}",
            own.props["head"]
        );
        serves &= own_body;
        check(&mut ok, "clef_layout_serves_its_head", serves);
        let given = serve_once(
            &a.dir,
            "clef_layout_release_head",
            &["-m", clef, "--head", head, "--port", "0"],
            &req,
        )?;
        let given_body = whole(&given, &questions);
        let (v_own, v_given): (Value, Value) = (
            serde_json::from_str(&own.body).unwrap_or(Value::Null),
            serde_json::from_str(&given.body).unwrap_or(Value::Null),
        );
        let (tops, dp) = compare("release head vs file's head", &v_given, &v_own, &questions);
        println!(
            "max |dp| over the file's head and the release's {dp:.4} (band {HEAD_DTYPE_BAND})"
        );
        check(
            &mut ok,
            "clef_layout_head_is_the_release_head",
            own_body && given_body && tops && dp <= HEAD_DTYPE_BAND,
        );
        // Diagnostic, no check: the `qwen35` file's answers against the clef file's. The two are
        // separate quantizations of the backbone, so their answers differ by more than the head.
        let (tops, dp) = compare("qwen35 file vs clef file", &v, &v_own, &questions);
        println!("diagnostic: qwen35 file against clef file: tops equal {tops}, max |dp| {dp:.4}");

        if ok {
            println!("gate_clef_serve: all checks passed");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
