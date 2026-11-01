//! `gate_clef_serve` — the decide seat of `bloomery-serve` on Clef-Flash: a
//! `qwen35` file with the release's head, and a `clef` file (llama.cpp's
//! layout, the head inside the GGUF), one process at a time on the card the
//! runner pins.
//!
//!     gate_clef_serve --model <gguf> --head <joint_head.safetensors>
//!                     --clef-model <gguf> --image-model <gguf> --mmproj <gguf>
//!                     --dir <dir> [--hf <repo>[:<quant>]]
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
//! - `decide_ubatch_is_the_cards_pick`: the backbone is a dense whole load, so its prompt ubatch is the card's
//!   verdict: the server's stderr holds `ubatch` lines (`ubatch_pick`) none of which is the user's, each
//!   judged by its own terms (the largest of 4096, 2048, 1024 and 512 within what the card had) with a pick;
//!   `-ub 0` is refused by name before any load (`decide_ubatch_flag_is_checked`). FAIL-first: a seat that
//!   opens at its own size prints no line and the clause is red;
//! - `speaks_llama_cpps_wire` (llama.cpp's `/v1/systemone`, its server at
//!   `a4cb4c61`): the answer's `model` is the server's name for the model
//!   (the file's name under `-m`), not the request's; `/v1/models` lists
//!   that name in `data` and `models`; `/props` carries the server's
//!   `engine` object; a request with an image, on a server with no
//!   `--mmproj`, is a 501 `not_supported_error` that carries llama.cpp's hint;
//! - `a_repeat_is_the_same_body`: the same request again is the same body
//!   byte for byte, `timings` taken out (each request runs from a reset);
//! - the face both constants of `bloomery-serve`'s drive print
//!   ([`face_pins_the_constants`]): `--help` exits 0 with the models table
//!   on its stdout, one line a seat with its `--hf` example exactly as the
//!   gate pins it, and a start that passes the help branch prints the
//!   first-start note before it refuses (here `--model qwen3` offline
//!   against an empty cache, refused as the class `network` before any
//!   load);
//! - the refusals, each a process that exits non-zero before it listens,
//!   its stderr naming why: `decide_with_no_head` (`--model decide -m <gguf>`), `qwen3_with_head`
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
//! - the image input (`--image-model`, Clef-Flash's Q8_0 file the clefvis sets are dumped from, and
//!   `--mmproj`, its encoder file), each server alone on the card:
//!   `image_text_control` (the model with no `--mmproj` answers the suite's first request and set
//!   D's text-only request `t1`: the bodies a media load's text requests are held to);
//!   `image_encoder_resident` (the server with `--mmproj` prints its `vision` record before it
//!   listens); `set_d_answers` (every request of set D, `tools/ref/clefvis/requests.jsonl` with
//!   each image stem its PNG, is a 200 with the release's body shape, usage tokens included, and
//!   the release's top option on every question; each question's slack — how far the release's
//!   own decision is from flipping — and the largest |dp| are printed beside it, and `t1`, which
//!   carries no image, is the text path's own gap); `text_after_images_is_the_text_control` (the
//!   suite's first request and `t1` answer on the media server, after its image requests, with
//!   the control's bodies byte for byte: a media load moves no text bit, and a reset takes the
//!   images away); `image_refusals` (a placeholder typed beside an image, images that alone pass
//!   the encoder's length, images past the decoded-byte budget, a URL that is no data URL,
//!   nine images, a GIF, each a 400 whose message names its cause, and videos a 501);
//!   `image_after_refusals` (the first image request
//!   again is the same body: the refusals left the server and its tables as they were).
//!   `--hf` adds two servers started from the hub, each alone on the card: `hf_picks_the_projector`
//!   (with no projector flag the seat fetches the repo's own projector, `hf::mmproj_sibling`'s pick,
//!   prints the `hf file` line of it and the `vision` record, and answers the suite's first request)
//!   and `hf_no_mmproj_serves_no_image` (with `--no-mmproj` it fetches and loads no projector, and an
//!   image is a 501 with the hint).
//!   Set D has no identity of its own (`images.jsonl` states no revision and no trailer), so its
//!   rows are held to what the run left beside them, as `hw_clef_image_ids_equal_the_release` holds
//!   them: `reference.jsonl` names the recipe's revision and carries the same `input_ids`, and the
//!   image grids' counts are the pad runs' lengths. That stands in for a family check.
//!
//! Logs per server in `<dir>/<clause>/` (`server.out`, `server.err`, the
//! bodies).

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
    use std::process::Command;
    use std::time::{Duration, Instant};

    use bloomery_gpu_gates::serve_client::{self, Served, curl};
    use bloomery_gpu_gates::ubatch_pick;
    use bloomery_gpu_gates::{GateError, checks_failed};
    use decision::encode::NO_IMAGE_INPUT;
    use decision::head::HeadConfig;
    use decision::json::{self, Json};
    use decision::render::dumps;
    use safetensors::write;
    use serde_json::Value;
    use serve::event::SCHEMA;
    use serve::exit::Code;

    use crate::decide_gate::{
        REFUSE_WITHIN, answered, check, compare, listening, post_raw, probabilities, refused,
        spawn, untimed, utf8,
    };

    const USAGE: &str = "usage: gate_clef_serve --model <gguf> --head <joint_head.safetensors> \
                         --clef-model <gguf> --image-model <gguf> --mmproj <gguf> --dir <dir> \
                         [--hf <repo>[:<quant>]]";

    struct Args {
        model: PathBuf,
        head: PathBuf,
        clef: PathBuf,
        /// The `qwen35` file the image clauses run: the Q8_0 file the clefvis sets are dumped from.
        image_model: PathBuf,
        /// Its encoder file.
        mmproj: PathBuf,
        dir: PathBuf,
        /// A `--hf` start's spelling, `<repo>[:<quant>]`: the clauses of the repo's own projector.
        hf: Option<String>,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut model, mut head, mut clef, mut dir) = (None, None, None, None);
        let (mut image_model, mut mmproj, mut hf) = (None, None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(v)),
                "--head" => head = Some(PathBuf::from(v)),
                "--clef-model" => clef = Some(PathBuf::from(v)),
                "--image-model" => image_model = Some(PathBuf::from(v)),
                "--mmproj" => mmproj = Some(PathBuf::from(v)),
                "--hf" => hf = Some(v),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (model, head, clef, image_model, mmproj, dir) {
            (Some(model), Some(head), Some(clef), Some(image_model), Some(mmproj), Some(dir)) => {
                Ok(Args {
                    model,
                    head,
                    clef,
                    image_model,
                    mmproj,
                    dir,
                    hf,
                })
            }
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
    ///
    /// PIN(2026-10-10): a seventh row, the mimo2 seat's, joins the table.
    /// PIN(2026-10-15): an eighth row, the nemotron seat's, joins the table.
    const MODELS_LINES: [&str; 8] = [
        "models (the --model word is optional with a model file: the file's architecture picks \
         the seat)",
        "ds41    DeepSeek-V4.1-Flash          --hf vcruz305/DeepSeek-V4.1-Flash-GGUF:Q3_K_M",
        "qwen38  Qwen3.8-Flash-Next           --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL",
        "glm     GLM-5.3-Flash                --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL",
        "qwen3   Qwen3-30B-A3B and Qwen3.6    --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M",
        "mimo2   MiMo-V2.6-Flash              --hf ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:MXFP4",
        "nemotron Nemotron 3 Nano             --hf nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M",
        "decide  a decision model by its head --hf bartowski/Cloudflare_clef-flash-GGUF:Q5_K_M",
    ];

    /// The note every start that passes the help branch prints on stderr
    /// before the load begins ([`drive::FIRST_START`] in `bloomery_serve.rs`).
    const FIRST_START: &str = "bloomery-serve: a first start on a new card can compile the \
                               GPU code for it (tens of seconds); later starts take seconds";

    /// The server of `args` with `env` set (`BLOOMERY_REF_MODEL` removed), run to its end within [`REFUSE_WITHIN`]:
    /// its status and what it printed on stdout and stderr. `None` when it is still running then; it is stopped.
    fn run_to_end(
        dir: &Path,
        name: &str,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<Option<(Option<i32>, String, String)>, GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let mut cmd = Command::new(std::env::current_exe()?.with_file_name("bloomery-serve"));
        cmd.env_remove("BLOOMERY_REF_MODEL")
            .envs(env.iter().copied());
        let mut s = Served::spawn_cmd(cmd, args, &d)?;
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
        let Some(status) = status else {
            println!(
                "{name}: still running after {REFUSE_WITHIN:?}: {}",
                s.stop()?
            );
            return Ok(None);
        };
        let out = std::fs::read_to_string(d.join("server.out"))?;
        let err = std::fs::read_to_string(d.join("server.err"))?;
        Ok(Some((status.code(), out, err)))
    }

    /// Whether the server of `args` under `env` ends within [`REFUSE_WITHIN`] with the status of `class`, an `error`
    /// event of that class last on stdout (after at most the listener's `bound`) and every one of `want` on stderr.
    fn refused_as(
        dir: &Path,
        name: &str,
        args: &[&str],
        env: &[(&str, &str)],
        class: Code,
        want: &[&str],
    ) -> Result<bool, GateError> {
        let Some((code, out, err)) = run_to_end(dir, name, args, env)? else {
            return Ok(false);
        };
        // The events: the listener's `bound` (a start by name binds it before the pick) and the `error` last.
        let events: Vec<Value> = out
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        let status = u64::from(class.status());
        let one_event = out.lines().count() == events.len()
            && events.split_last().is_some_and(|(error, before)| {
                error["schema"] == SCHEMA
                    && error["event"] == "error"
                    && error["exit"].as_u64() == Some(status)
                    && error["code"] == class.name()
                    && before
                        .iter()
                        .all(|e| e["schema"] == SCHEMA && e["event"] == "bound")
            });
        let named = want.iter().all(|w| err.contains(w));
        println!(
            "{name}: exit {code:?}, the events end in a {class} one {one_event}, stderr names {want:?}: {named}"
        );
        let pass = code == Some(i32::from(class.status())) && one_event && named;
        if !pass {
            println!("{name} stdout:\n{out}\n{name} stderr:\n{err}");
        }
        Ok(pass)
    }

    /// The face (module header): the two constant lines of the drive, read
    /// from what the binary printed. `--help` exits 0 with the usage and the
    /// models table on stdout, every line exactly as [`MODELS_LINES`] pins it
    /// and the first line pointing an agent at `--recommend --json`; a start
    /// that passes the help branch, here `--model qwen3` offline against an
    /// empty cache (a start by name that finds no listing and no cached set,
    /// the class `network`), prints the first-start note first. Neither
    /// process loads a model; the second reads the cards' census.
    ///
    /// PIN(2026-10-29): the agent contract. `--help` exits 0 and prints on stdout (it exited non-zero
    /// with the usage on stderr), and `--model qwen3` alone is a start by name, so the first-start clause
    /// runs it offline against an empty cache, where it was the refusal "no model file".
    fn face_pins_the_constants(dir: &Path) -> Result<bool, GateError> {
        let mut want = MODELS_LINES.to_vec();
        want.push("bloomery-serve --recommend --json");
        let helps = answered(dir, "help_prints_the_models_table", &["--help"], &want)?;
        let cache = dir.join("offline_cache");
        std::fs::create_dir_all(&cache)?;
        let starts = refused_as(
            dir,
            "every_start_notes_the_first_compile",
            &["--model", "qwen3", "--host", "127.0.0.1", "--port", "0"],
            &[("HF_HUB_OFFLINE", "1"), ("BLOOMERY_CACHE", utf8(&cache)?)],
            Code::Network,
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

    /// The placeholder Clef's text carries for one image, which only an `images` entry may make.
    const PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

    /// A file of the repository.
    fn repo_file(rel: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel)
    }

    /// The rows of a JSON lines file, in order.
    fn jsonl(path: &Path) -> Result<Vec<Json>, GateError> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| Ok(json::parse(l).map_err(|e| format!("{}: {e}", path.display()))?))
            .collect()
    }

    fn id_of(row: &Json) -> Result<String, GateError> {
        Ok(row
            .get("id")
            .and_then(Json::as_str)
            .ok_or("a row of set D has no id")?
            .to_owned())
    }

    /// The test images the gate posts by their stems: the three set D's requests name, and
    /// `max-2600x1800`, which the length row and the decode-bound row post.
    const IMAGES: [(&str, &str); 4] = [
        ("free-448x448", "tools/ref/clefvis/images/free-448x448.png"),
        ("free-640x480", "tools/ref/clefvis/images/free-640x480.png"),
        (
            "free-1024x768",
            "tools/ref/clefvis/images/free-1024x768.png",
        ),
        (
            "max-2600x1800",
            "tools/ref/clefvis/images/max-2600x1800.png",
        ),
    ];

    /// An image of [`IMAGES`] as a data URL; a stem the gate does not post is refused by name.
    fn png_url(stem: &str) -> Result<String, GateError> {
        let (_, rel) = IMAGES
            .iter()
            .find(|(s, _)| *s == stem)
            .ok_or_else(|| format!("the image {stem:?} is not one of the gate's {IMAGES:?}"))?;
        let path = repo_file(rel);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(format!(
            "data:image/png;base64,{}",
            serve_client::base64(&bytes)
        ))
    }

    /// The snapshot revision the release's runs are made at: the recipe's own `REVISION`.
    fn release_revision() -> Result<String, GateError> {
        let path = repo_file("tools/ref/clefvis/clefvis.sh");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(text
            .lines()
            .find_map(|l| l.strip_prefix("REVISION="))
            .ok_or_else(|| format!("{}: no REVISION line", path.display()))?
            .to_owned())
    }

    /// One request of set D: the body a client posts and the release's reply to it.
    struct Case {
        id: String,
        /// The request of `tools/ref/clefvis/requests.jsonl` with each image stem its PNG as a data
        /// URL, and a `state_tokens` request's state the one the release rendered.
        body: String,
        images: usize,
        /// The release's `systemone()` body.
        release: Json,
        /// Whether set D's two files agree on what they hold ([module doc](self)): the recipe's
        /// revision, the same ids, the image grids' counts the pad runs' lengths.
        provenance: bool,
    }

    /// The `N` unsigned integers of a set D row's `[...]` entry `v` of `key`; anything else (no
    /// array, another length, an entry that is no unsigned integer) is refused by name.
    fn numbers<const N: usize>(id: &str, key: &str, v: &Json) -> Result<[u64; N], GateError> {
        let refused = || {
            format!(
                "{id}: {key} entry {} is not {N} unsigned integers",
                dumps(v, false)
            )
        };
        let Json::Array(a) = v else {
            return Err(refused().into());
        };
        let n: Vec<u64> = a
            .iter()
            .map(Json::as_u64)
            .collect::<Option<_>>()
            .ok_or_else(refused)?;
        Ok(n.try_into().map_err(|_| refused())?)
    }

    /// The array a set D row holds at `key`, refused by name when it is anything else.
    fn entries<'a>(id: &str, row: &'a Json, key: &str) -> Result<&'a [Json], GateError> {
        match row.get(key) {
            Some(Json::Array(a)) => Ok(a),
            other => Err(format!(
                "{id}: {key} is no array: {}",
                other.map_or("absent".to_owned(), |v| dumps(v, false))
            )
            .into()),
        }
    }

    /// A row's merged image tokens by the release's grids: a grid (t, h, w) is t·h·w / 4; a
    /// product past `u64` is refused by name.
    fn image_counts(id: &str, row: &Json) -> Result<Vec<u64>, GateError> {
        entries(id, row, "image_grid_thw")?
            .iter()
            .map(|g| {
                let n = numbers::<3>(id, "image_grid_thw", g)?
                    .into_iter()
                    .try_fold(1u64, u64::checked_mul)
                    .ok_or_else(|| {
                        format!("{id}: image_grid_thw {} overflows u64", dumps(g, false))
                    })?;
                Ok(n / 4)
            })
            .collect()
    }

    /// A row's pad run lengths by the release's spans (start, length); a row with no image may
    /// leave `image_spans` out.
    fn image_runs(id: &str, row: &Json) -> Result<Vec<u64>, GateError> {
        if row.get("image_spans").is_none() {
            return Ok(Vec::new());
        }
        entries(id, row, "image_spans")?
            .iter()
            .map(|s| Ok(numbers::<2>(id, "image_spans", s)?[1]))
            .collect()
    }

    /// Every request of set D (`$BLOOMERY_DATA/clefvis/ref/`), in the order of `images.jsonl`.
    fn set_d() -> Result<Vec<Case>, GateError> {
        let dir = refset::data_dir().join("clefvis/ref");
        let rows = jsonl(&dir.join("images.jsonl"))?;
        let reference = jsonl(&dir.join("reference.jsonl"))?;
        let sent = jsonl(&repo_file("tools/ref/clefvis/requests.jsonl"))?;
        let revision = release_revision()?;
        let find = |rows: &[Json], id: &str| -> Result<Json, GateError> {
            rows.iter()
                .find(|r| id_of(r).is_ok_and(|i| i == id))
                .cloned()
                .ok_or_else(|| format!("{id}: no such row in set D's files").into())
        };
        let mut cases = Vec::new();
        for row in &rows {
            let id = id_of(row)?;
            let pair = find(&reference, &id)?;
            let Json::Object(pairs) = find(&sent, &id)? else {
                return Err(format!("{id}: a request is an object").into());
            };
            let render = row
                .get("state_render")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("{id}: no state_render"))?;
            let mut out = Vec::new();
            let mut images = 0;
            for (k, v) in &pairs {
                match (k.as_str(), v) {
                    ("images", Json::Array(stems)) => {
                        images = stems.len();
                        let urls = stems
                            .iter()
                            .map(|s| {
                                let stem = s.as_str().ok_or("an image stem")?;
                                Ok(Json::Str(png_url(stem)?))
                            })
                            .collect::<Result<Vec<_>, GateError>>()?;
                        out.push((k.clone(), Json::Array(urls)));
                    }
                    ("state_tokens", _) => {
                        out.push(("state".to_owned(), Json::Str(render.to_owned())))
                    }
                    _ => out.push((k.clone(), v.clone())),
                }
            }
            // The release's own counts: its grids and its pad runs say the same.
            let counts = image_counts(&id, row)?;
            let runs = image_runs(&id, row)?;
            let said = pair
                .get("facts")
                .and_then(|f| f.get("revision"))
                .and_then(Json::as_str);
            let provenance = said == Some(revision.as_str())
                && pair.get("input_ids") == row.get("input_ids")
                && counts == runs;
            cases.push(Case {
                id,
                body: dumps(&Json::Object(out), false),
                images,
                release: pair.get("response").cloned().ok_or("no response")?,
                provenance,
            });
        }
        Ok(cases)
    }

    /// A body's structure: every number blanked but the token counts of `usage`, and the model's
    /// name (the release echoes the request's, the server names the one it serves).
    fn shape(v: &Json, path: &str) -> Json {
        match v {
            Json::Float(_) | Json::Int(_) if !path.ends_with("tokens") => Json::Null,
            Json::Str(_) if path == ".model" => Json::Null,
            Json::Object(pairs) => Json::Object(
                pairs
                    .iter()
                    .map(|(k, x)| (k.clone(), shape(x, &format!("{path}.{k}"))))
                    .collect(),
            ),
            Json::Array(items) => Json::Array(items.iter().map(|x| shape(x, path)).collect()),
            other => other.clone(),
        }
    }

    /// How far the decision of one answer is from flipping: for a noul `|p - 0.5|`, else half the
    /// gap of the two most probable options. A probability that moves by at most this flips no top.
    fn slack(a: &Value) -> f64 {
        let mut p: Vec<f64> = probabilities(a).into_iter().map(|x| x.1).collect();
        if a["type"] == "noul" {
            return p.first().map_or(0.0, |x| (x - 0.5).abs());
        }
        p.sort_by(|x, y| y.total_cmp(x));
        (p[0] - p.get(1).copied().unwrap_or(0.0)) / 2.0
    }

    /// The server's body for `c` against the release's: whether it has the release's shape and the
    /// release's top on every question, and the largest |dp|. The release's slack of each question
    /// is printed beside it.
    fn answers_like_the_release(c: &Case, served: &str) -> Result<(bool, f64), GateError> {
        let Some(plain) = untimed(served) else {
            println!("set D {}: no timings last in {served}", c.id);
            return Ok((false, 1.0));
        };
        // `untimed` cuts at the timings key: the body closes again.
        let whole = format!("{plain}}}");
        let ours = json::parse(&whole).map_err(|e| format!("{}: {e}", c.id))?;
        let shape_equal = shape(&ours, "") == shape(&c.release, "");
        let (ours_v, release_v): (Value, Value) = (
            serde_json::from_str(&whole)?,
            serde_json::from_str(&dumps(&c.release, false))?,
        );
        let questions: Vec<String> = release_v["answers"]
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        let (tops, worst) = compare(&format!("set D {}", c.id), &ours_v, &release_v, &questions);
        for q in &questions {
            println!(
                "set D {}: {q}: the release's slack {:.4}",
                c.id,
                slack(&release_v["answers"][q])
            );
        }
        println!(
            "set D {}: shape equal {shape_equal}, tops equal {tops}, max |dp| {worst:.4}, \
             {} images, provenance {}, {}",
            c.id,
            c.images,
            c.provenance,
            serde_json::from_str::<Value>(served).map_or_else(
                |_| String::new(),
                |v| serde_json::to_string(&v["timings"]).unwrap_or_default()
            )
        );
        Ok((shape_equal && tops && c.provenance, worst))
    }

    /// The value of `key=` in a record line.
    fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        line.split_whitespace()
            .find_map(|w| w.strip_prefix(key)?.strip_prefix('='))
    }

    /// The server stopped, its end printed.
    fn stop(mut s: Served) -> Result<(), GateError> {
        println!("server stopped: {}", s.stop()?);
        Ok(())
    }

    /// `bloomery-serve` with `args` in `<dir>/<name>/`, listening: the server, its logs' directory
    /// and its address.
    fn up(dir: &Path, name: &str, args: &[&str]) -> Result<(Served, PathBuf, String), GateError> {
        let (mut s, d) = spawn(dir, name, args)?;
        let line = listening(&mut s, &d.join("server.err"))?;
        println!("{line}");
        let addr = line
            .split_once("listening on http://")
            .and_then(|(_, r)| r.split_whitespace().next())
            .ok_or("no address in the listening record")?
            .to_owned();
        Ok((s, d, addr))
    }

    /// `body` POSTed to the server at `addr`: whether it is refused with `status`, an error of type
    /// `kind` and a message that holds every one of `want`.
    fn refused_with(
        addr: &str,
        name: &str,
        body: &str,
        (status, kind): (u16, &str),
        want: &[&str],
    ) -> Result<bool, GateError> {
        let (st, reply) = post_raw(&format!("http://{addr}/v1/systemone"), body)?;
        let v: Value = serde_json::from_str(&reply).unwrap_or(Value::Null);
        let message = v["error"]["message"].as_str().unwrap_or_default();
        let pass =
            st == status && v["error"]["type"] == kind && want.iter().all(|w| message.contains(w));
        println!("{name}: {st} {kind:?} names {want:?}: {pass}: {message}");
        Ok(pass)
    }

    /// The suite's first request, whose `state` is followed by `extra` and which carries `images`
    /// (data URLs).
    fn with_images(req: &str, images: &[String], state_extra: &str) -> Result<String, GateError> {
        let Json::Object(pairs) = json::parse(req)? else {
            return Err("the suite's request is an object".into());
        };
        let mut out = vec![(
            "images".to_owned(),
            Json::Array(images.iter().map(|u| Json::Str(u.clone())).collect()),
        )];
        for (k, v) in pairs {
            match (k.as_str(), v) {
                ("state", Json::Str(s)) => out.push((k, Json::Str(format!("{s}{state_extra}")))),
                (_, v) => out.push((k, v)),
            }
        }
        Ok(dumps(&Json::Object(out), false))
    }

    /// The seat started from the hub (`--hf`, module doc): with no projector flag it fetches and
    /// loads the repo's own projector and serves images; with `--no-mmproj` it serves none.
    fn hf_projector(a: &Args, hf: &str, req: &str, ok: &mut bool) -> Result<(), GateError> {
        let (s, d, addr) = up(&a.dir, "hf_projector", &["--hf", hf, "--port", "0"])?;
        let err = std::fs::read_to_string(d.join("server.err"))?;
        let (st, body) = post_raw(&format!("http://{addr}/v1/systemone"), req)?;
        let fetched = err
            .lines()
            .any(|l| l.starts_with("hf file") && l.contains("file=mmproj-"));
        let resident = err.lines().any(|l| l.starts_with("vision "));
        println!("hf: projector fetched {fetched}, vision record {resident}, request {st}");
        stop(s)?;
        check(
            ok,
            "hf_picks_the_projector",
            fetched && resident && st == 200 && untimed(&body).is_some(),
        );

        let (s, d, addr) = up(
            &a.dir,
            "hf_no_mmproj",
            &["--hf", hf, "--no-mmproj", "--port", "0"],
        )?;
        let err = std::fs::read_to_string(d.join("server.err"))?;
        let image = req.replacen(
            '{',
            r#"{"images": ["data:image/png;base64,iVBORw0KGgo="], "#,
            1,
        );
        let (st, reply) = post_raw(&format!("http://{addr}/v1/systemone"), &image)?;
        let v: Value = serde_json::from_str(&reply).unwrap_or(Value::Null);
        let none = !err
            .lines()
            .any(|l| l.contains("mmproj") || l.starts_with("vision "));
        println!("hf --no-mmproj: no projector line {none}, image request {st} {reply}");
        stop(s)?;
        check(
            ok,
            "hf_no_mmproj_serves_no_image",
            none && st == 501 && v["error"]["message"] == NO_IMAGE_INPUT,
        );
        Ok(())
    }

    /// The image input on Clef-Flash's Q8_0 file and its encoder (module doc).
    fn image_input(a: &Args, req: &str, ok: &mut bool) -> Result<(), GateError> {
        let (model, head, mmproj) = (utf8(&a.image_model)?, utf8(&a.head)?, utf8(&a.mmproj)?);
        let cases = set_d()?;
        let control = cases
            .iter()
            .find(|c| c.images == 0)
            .ok_or("set D holds no request without an image")?;

        // The control: the same file with no encoder.
        let (mut s, _d, addr) = up(
            &a.dir,
            "image_text_control",
            &["-m", model, "--head", head, "--port", "0"],
        )?;
        let url = format!("http://{addr}/v1/systemone");
        let (st_first, first) = post_raw(&url, req)?;
        let (st_t1, t1) = post_raw(&url, &control.body)?;
        println!("server stopped: {}", s.stop()?);
        let pass =
            st_first == 200 && st_t1 == 200 && untimed(&first).is_some() && untimed(&t1).is_some();
        check(ok, "image_text_control", pass);

        // The server with the encoder.
        let (mut s, d, addr) = up(
            &a.dir,
            "image_serve",
            &[
                "-m", model, "--head", head, "--mmproj", mmproj, "--port", "0",
            ],
        )?;
        let url = format!("http://{addr}/v1/systemone");
        let err = std::fs::read_to_string(d.join("server.err"))?;
        let record = err.lines().find(|l| l.starts_with("vision ")).unwrap_or("");
        println!("vision record: {record}");
        // Where the tower sits decides what the backbone's whole-fit verdict counts: a stage-card
        // seat holds the tower's bytes among the card's reserves, which the verdict's line names
        // with the record's weights and activations; a card off the plan holds none.
        let bytes = ["weights", "scratch"]
            .iter()
            .map(|k| field(record, k)?.parse::<u64>().ok())
            .sum::<Option<u64>>();
        let fit = err.lines().find(|l| l.starts_with("whole fit at ctx"));
        println!("whole fit: {fit:?}");
        let note = "the reserves term holds the image encoder's";
        let counted = match (field(record, "seat"), fit, bytes) {
            (Some("stage-reserve"), Some(l), Some(b)) => l.contains(&format!("{note} {b} B")),
            (Some("off-plan"), Some(l), Some(_)) => !l.contains(note),
            _ => false,
        };
        check(
            ok,
            "image_encoder_resident",
            ["card=", "cap=", "load_s="]
                .iter()
                .all(|k| record.contains(k))
                && counted,
        );

        let (mut pass, mut worst) = (true, 0f64);
        // The first image request's body, for the repeat after the refusals.
        let again = cases
            .iter()
            .find(|c| c.images > 0)
            .ok_or("set D holds no request with an image")?;
        let mut again_first = None;
        for c in &cases {
            let (st, body) = post_raw(&url, &c.body)?;
            std::fs::write(d.join(format!("{}.json", c.id)), &body)?;
            println!("set D {}: {st}", c.id);
            if c.id == again.id && st == 200 {
                again_first = untimed(&body).map(str::to_owned);
            }
            if st != 200 {
                println!("{body}");
                pass = false;
                continue;
            }
            let (like, dp) = answers_like_the_release(c, &body)?;
            pass &= like;
            // Every answer's |dp| counts but the text control's, the text path's own gap.
            if c.images > 0 {
                worst = worst.max(dp);
            }
        }
        println!("set D: worst |dp| over the image requests {worst:.4}");
        check(ok, "set_d_answers", pass);

        // A media load moves no text bit.
        let (_, text_first) = post_raw(&url, req)?;
        let (_, text_t1) = post_raw(&url, &control.body)?;
        check(
            ok,
            "text_after_images_is_the_text_control",
            untimed(&text_first).is_some()
                && untimed(&text_first) == untimed(&first)
                && untimed(&text_t1) == untimed(&t1),
        );

        // The refusals.
        let max = png_url("max-2600x1800")?;
        let wide = png_url("free-1024x768")?;
        let one = png_url("free-448x448")?;
        let nine = vec![one.clone(); 9];
        // PIN(2026-10-22): five max-2600x1800 meet the 64 MiB decode bound before the encoder length.
        // Align 32, one token per 32×32, at most 4096: 2600×1800 plans to 2432×1696 = 76×53 =
        // 4028 tokens, 1024×768 is 32×24 = 768. The piece is start, pad and end per image and a
        // newline, so the length is sum(counts) + 2*n + 1: 4*4028 + 768 + 2*5 + 1 = 16891 >
        // 16384. Decoded bytes are w*h*3: 4*2600*1800*3 + 1024*768*3 = 58,519,296 < 64*1024*1024.
        let length = vec![max.clone(), max.clone(), max.clone(), max.clone(), wide];
        let five = vec![max; 5];
        let gif = "data:image/gif;base64,R0lGODlhAQABAAAAACw=".to_owned();
        let videos = req.replacen('{', r#"{"videos": ["x"], "#, 1);
        let mut pass = true;
        for (name, body, status, kind, want) in [
            (
                "placeholder_typed_beside_an_image",
                with_images(req, std::slice::from_ref(&one), PLACEHOLDER)?,
                400,
                "invalid_request_error",
                vec!["image placeholder"],
            ),
            (
                "images_alone_pass_the_encoders_length",
                with_images(req, &length, "")?,
                400,
                "invalid_request_error",
                vec!["5 image(s) take", "maximum is 16384"],
            ),
            (
                "images_pass_the_decoded_byte_budget",
                with_images(req, &five, "")?,
                400,
                "invalid_request_error",
                vec!["image size limits"],
            ),
            (
                "not_a_data_url",
                with_images(req, &["http://example.com/a.png".to_owned()], "")?,
                400,
                "invalid_request_error",
                vec!["images must be data URLs"],
            ),
            (
                "nine_images",
                with_images(req, &nine, "")?,
                400,
                "invalid_request_error",
                vec!["too many images, the maximum is 8"],
            ),
            (
                "not_a_png_or_jpeg",
                with_images(req, &[gif], "")?,
                400,
                "invalid_request_error",
                vec!["images[0]", "only image/png and image/jpeg"],
            ),
            ("videos", videos, 501, "not_supported_error", vec!["videos"]),
        ] {
            pass &= refused_with(&addr, name, &body, (status, kind), &want)?;
        }
        check(ok, "image_refusals", pass);

        // The refusals left the server as it was.
        let (st, repeat) = post_raw(&url, &again.body)?;
        let same = again_first
            .as_deref()
            .is_some_and(|b| st == 200 && untimed(&repeat) == Some(b));
        check(ok, "image_after_refusals", same);
        println!("server stopped: {}", s.stop()?);
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let a = parse_args()?;
        std::fs::create_dir_all(&a.dir)?;
        let (model, head) = (utf8(&a.model)?, utf8(&a.head)?);
        let mut ok = true;

        let pass = face_pins_the_constants(&a.dir)?;
        check(&mut ok, "face_pins_the_constants", pass);

        // The refusals first: none of them loads the backbone.
        // `-m <gguf>` with no head is not refused: a qwen35 file a generative seat serves is that seat's, and
        // gate_lev_serve's qwen35_with_no_head holds the refusal of one no seat serves.
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
        let pass = refused(
            &a.dir,
            "decide_ubatch_flag_is_checked",
            &["-m", model, "--head", head, "-ub", "0"],
            // The server names the flag in its long spelling, whichever spelling was given.
            &["--ubatch-size takes a whole number of at least 1, not \"0\""],
        )?;
        check(&mut ok, "decide_ubatch_flag_is_checked", pass);
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
        let err = std::fs::read_to_string(d.join("server.err"))?;
        let picks = ubatch_pick::choices(&err);
        let picked = !picks.is_empty()
            && picks
                .iter()
                .all(|c| c.asked.is_none() && c.pick.is_some() && ubatch_pick::judge(c).is_ok());
        println!(
            "decide ubatch lines {picks:?}: each the card's pick, judged by its terms {picked}"
        );
        check(&mut ok, "decide_ubatch_is_the_cards_pick", picked);
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
        wire &= st == 501
            && refused["error"]["type"] == "not_supported_error"
            && refused["error"]["message"] == NO_IMAGE_INPUT;
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

        image_input(&a, &req, &mut ok)?;
        if let Some(hf) = &a.hf {
            hf_projector(&a, hf, &req, &mut ok)?;
        }

        if ok {
            println!("gate_clef_serve: all checks passed");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{GateError, Json, image_counts, image_runs, json};

        type Read = fn(&str, &Json) -> Result<Vec<u64>, GateError>;

        /// Set D's grids and spans are read whole or refused by name. Mutant: the old reader,
        /// which left out an entry that is no unsigned integer, so a short grid and its short
        /// pad run agreed.
        #[test]
        fn set_d_numbers_refuse_what_they_cannot_read() {
            let row = |s: &str| json::parse(s).unwrap();
            let good = row(r#"{"image_grid_thw": [[1, 4, 6]], "image_spans": [[9, 6]]}"#);
            assert_eq!(image_counts("i1", &good).unwrap(), [6]);
            assert_eq!(image_runs("i1", &good).unwrap(), [6]);
            let text = row(r#"{"image_grid_thw": []}"#);
            assert_eq!(image_runs("t1", &text).unwrap(), Vec::<u64>::new());
            let cases: [(&str, Read); 8] = [
                (r#"{"image_grid_thw": [[1, null, 2, 8]]}"#, image_counts),
                (
                    r#"{"image_grid_thw": [[1, 4, 4611686018427387910]]}"#,
                    image_counts,
                ),
                (r#"{"image_grid_thw": [[1, 4, 6.0]]}"#, image_counts),
                (r#"{"image_grid_thw": [[4, 6]]}"#, image_counts),
                (r#"{}"#, image_counts),
                (r#"{"image_spans": [[9, "6"]]}"#, image_runs),
                (r#"{"image_spans": [[9]]}"#, image_runs),
                (r#"{"image_spans": 6}"#, image_runs),
            ];
            for (bad, read) in cases {
                let e = read("i1", &row(bad)).expect_err(bad).to_string();
                assert!(e.starts_with("i1: image_"), "{bad}: {e}");
            }
        }
    }
}
