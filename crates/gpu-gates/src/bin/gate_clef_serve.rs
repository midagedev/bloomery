//! `gate_clef_serve` — the decide seat of `bloomery-serve` on Clef-Flash: a
//! `qwen35` file with the release's head, one process at a time on the card
//! the runner pins.
//!
//!     gate_clef_serve --model <gguf> --head <joint_head.safetensors> --dir <dir>
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
//! - the refusals, each a process that exits non-zero before it listens,
//!   its stderr naming why: `qwen35_with_no_head` (`-m <gguf>`),
//!   `decide_with_no_head` (`--model decide -m <gguf>`), `qwen3_with_head`
//!   (`--model qwen3 -m <gguf> --head <head>`), `unknown_head_config`
//!   (`--head-config` of a config no row knows), `narrow_head` (a head of
//!   hidden width 16, written into `<dir>`, against the file's
//!   `embedding_length`: refused before the backbone loads),
//!   `clef_layout_gguf` (a GGUF of arch `clef`, written into `<dir>`, is
//!   refused as llama.cpp's Clef layout, with bartowski's `--hf` line:
//!   refused before the backbone loads).
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

#[cfg(all(feature = "deepseek41", feature = "clef"))]
mod gate {
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    use bloomery_gpu_gates::serve_client::{Served, curl};
    use bloomery_gpu_gates::{GateError, checks_failed, verdict};
    use decision::head::HeadConfig;
    use decision::safetensors::write;
    use serde_json::Value;

    const USAGE: &str =
        "usage: gate_clef_serve --model <gguf> --head <joint_head.safetensors> --dir <dir>";

    /// How long a refused server may take to exit: it refuses before any
    /// load of the backbone.
    const REFUSE_WITHIN: Duration = Duration::from_secs(120);

    struct Args {
        model: PathBuf,
        head: PathBuf,
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut model, mut head, mut dir) = (None, None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(v)),
                "--head" => head = Some(PathBuf::from(v)),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (model, head, dir) {
            (Some(model), Some(head), Some(dir)) => Ok(Args { model, head, dir }),
            _ => Err(USAGE.into()),
        }
    }

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    fn utf8(p: &Path) -> Result<&str, GateError> {
        Ok(p.to_str()
            .ok_or_else(|| format!("{} is not UTF-8", p.display()))?)
    }

    /// `bloomery-serve` beside this binary with `args`, `BLOOMERY_REF_MODEL`
    /// removed, its logs in `<dir>/<name>/`.
    fn spawn(dir: &Path, name: &str, args: &[&str]) -> Result<(Served, PathBuf), GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let mut cmd = Command::new(std::env::current_exe()?.with_file_name("bloomery-serve"));
        cmd.env_remove("BLOOMERY_REF_MODEL");
        Ok((Served::spawn_cmd(cmd, args, &d)?, d))
    }

    /// Whether the server of `args` exits non-zero within [`REFUSE_WITHIN`]
    /// with every one of `want` in its stderr; a server still running then
    /// is killed and fails the clause.
    fn refused(dir: &Path, name: &str, args: &[&str], want: &[&str]) -> Result<bool, GateError> {
        let (mut s, d) = spawn(dir, name, args)?;
        let t = Instant::now();
        let status = loop {
            if let Some(status) = s.child.try_wait()? {
                break Some(status);
            }
            if t.elapsed() > REFUSE_WITHIN {
                break None;
            }
            std::thread::sleep(Duration::from_millis(200));
        };
        let err = std::fs::read_to_string(d.join("server.err"))?;
        let pass = match status {
            Some(st) => {
                let named = want.iter().all(|w| err.contains(w));
                println!("{name}: exit {st}, names {want:?}: {named}");
                !st.success() && named
            }
            None => {
                println!(
                    "{name}: still running after {REFUSE_WITHIN:?}: {}",
                    s.stop()?
                );
                false
            }
        };
        if !pass {
            println!("{name} stderr:\n{err}");
        }
        Ok(pass)
    }

    /// How long the server may take to load and listen, and how often its
    /// stdout is read meanwhile.
    const LISTEN_WITHIN: Duration = Duration::from_secs(300);
    const LISTEN_POLL: Duration = Duration::from_millis(500);

    /// The listening line on the server's stdout, read every [`LISTEN_POLL`]
    /// for at most [`LISTEN_WITHIN`]; a server that exits first fails by name.
    fn listening(s: &mut Served, out: &Path) -> Result<String, GateError> {
        let t = Instant::now();
        while t.elapsed() < LISTEN_WITHIN {
            let text = std::fs::read_to_string(out).unwrap_or_default();
            if let Some(l) = text.lines().find(|l| l.contains("listening on http://")) {
                return Ok(l.to_owned());
            }
            if let Some(st) = s.child.try_wait()? {
                return Err(format!("the server exited ({st}) before listening").into());
            }
            std::thread::sleep(LISTEN_POLL);
        }
        Err(format!("the server did not listen within {LISTEN_WITHIN:?}").into())
    }

    /// `text` POSTed to `url` as it is (curl's stdin, no re-serialization:
    /// the request's key order is its questions' order): the status and the
    /// body.
    fn post_raw(url: &str, text: &str) -> Result<(u16, String), GateError> {
        let mut c = Command::new("curl")
            .args(["-sS", "--max-time", "600", "-w", "\n%{http_code}"])
            .args([
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
                url,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        c.stdin
            .take()
            .ok_or("curl has no stdin")?
            .write_all(text.as_bytes())?;
        let out = c.wait_with_output()?;
        let all = String::from_utf8(out.stdout)?;
        let (body, code) = all
            .rsplit_once('\n')
            .ok_or_else(|| format!("curl {url}: no status line"))?;
        Ok((code.trim().parse()?, body.to_owned()))
    }

    /// `body` with the `timings` object the server appends as its last key
    /// taken out, or `None` when it has none.
    fn untimed(body: &str) -> Option<&str> {
        let at = body.rfind(",\"timings\":")?;
        body.trim_end().ends_with("}}").then_some(&body[..at])
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

    /// A GGUF whose metadata names arch `clef` and holds nothing else,
    /// written into `dir`: the refusal of llama.cpp's Clef layout reads the
    /// header only, so no tensor is needed.
    fn clef_layout_file(dir: &Path) -> Result<PathBuf, GateError> {
        let kvs = vec![(
            gguf::GENERAL_ARCHITECTURE.to_owned(),
            gguf::Value::String("clef".to_owned()),
        )];
        let layout = gguf::write::Layout::new(&kvs, Vec::new())?;
        let path = dir.join("clef-layout.gguf");
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

    pub fn run() -> Result<(), GateError> {
        let a = parse_args()?;
        std::fs::create_dir_all(&a.dir)?;
        let (model, head) = (utf8(&a.model)?, utf8(&a.head)?);
        let mut ok = true;

        // The refusals first: none of them loads the backbone.
        let pass = refused(
            &a.dir,
            "qwen35_with_no_head",
            &["-m", model],
            &[
                "a qwen35 file with no head",
                "--head",
                "clef (head repo Cloudflare/clef-flash, backbone qwen35)",
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
        let clef = clef_layout_file(&a.dir)?;
        let pass = refused(
            &a.dir,
            "clef_layout_gguf",
            &["-m", utf8(&clef)?, "--head", head],
            &[
                "llama.cpp's Clef layout",
                "does not serve yet",
                "--hf bartowski/Cloudflare_clef-flash-GGUF",
            ],
        )?;
        check(&mut ok, "clef_layout_gguf", pass);

        // The seat serving its row.
        let (mut s, d) = spawn(
            &a.dir,
            "serve",
            &["-m", model, "--head", head, "--port", "0"],
        )?;
        let line = listening(&mut s, &d.join("server.out"))?;
        println!("{line}");
        let addr = line
            .split_once("listening on http://")
            .and_then(|(_, r)| r.split_whitespace().next())
            .ok_or("no address in the listening line")?
            .to_owned();
        let url = |p: &str| format!("http://{addr}{p}");
        let names = |w: &str| line.contains(w);
        let mut serves = names(&format!(
            "model {}",
            a.model.file_name().and_then(|n| n.to_str()).unwrap_or("?")
        )) && names("head joint_head.safetensors")
            && names("row clef)");
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

        if ok {
            println!("gate_clef_serve: all checks passed");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
