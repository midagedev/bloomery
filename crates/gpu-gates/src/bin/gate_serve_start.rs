//! `gate_serve_start` — the real `bloomery-serve`, started once per image flag on a family's fixture.
//!
//!     gate_serve_start --family glm|ds41 --dir <out>
//!
//! The server-side image flags (`--mmproj`, `--hf` with the repo's projector) run on the seat's startup
//! path, where one unregistered record kind or one projector the plan cannot seat ends the process. The
//! seat gates that start the server on a real file run only in the real tier, so a landing's fixture tier
//! met these paths with no test. This gate starts the shipped binary (`bloomery-serve`, built with the
//! release's features and beside this binary) on the family's fixture (`ref_model_path`: GLM-5.3 or V4.1) and
//! reads each server's stderr through the seat's own record kinds. One server runs at a time, each in its own
//! directory under `--dir`, the card pick `BLOOMERY_CARD=both` (the recipe's), every lever unset but the
//! fixture tier's `BLOOMERY_CARD_BUDGET` (`bloomery_levers::at_main` refuses any other).
//!
//! - (s1) the plain start: `-m <fixture>` with the serving defaults, no `--model` word (the file's
//!   architecture picks the seat), no placement, context or slot flag. `/health` answers 200 and one text
//!   chat is a 200 with a prompt; `/props` names no image input. The placement the unset rule chose is
//!   printed, never asserted.
//! - (s2) `--place a --mmproj <projector>`: the family's projector file (GLM: the pinned
//!   `refset::arch::glm5next::vis::MMPROJ_BF16`; V4.1: the `# mmproj` line of the vision set the V4.1 image
//!   gates read). With both cards in view the placement `a` leaves one card off the plan, so the tower sits
//!   there and no reserve enters the plan (`vision_seat::encoder_seat`): it loads whatever the fixture's
//!   card budget leaves. The server prints exactly one `vision` record, and it is the record the tower's
//!   own seat rule and card figure give (card, seat word, weights, scratch, the most image tokens one image
//!   takes). One image chat sends `tools/ref/clefvis/images/free-448x448.png` (its sha256 checked) twice in one
//!   user message: two `image_url` parts with a lone `.` text part between them, then the question. It is a
//!   200 whose `prompt_tokens` are the rendered prompt less the two placeholders plus two spans, each the
//!   span the projector's own `MediaModel::prepare` gives the image. Only a call of two or more images meets
//!   the spans' order and count (`media::check_spans`: ascending, each inside the prompt, disjoint). For GLM
//!   that is also the text chat's plus 2 x (16 x 16 + 2) plus the tokens of the `.` (one byte is one token of
//!   a byte-level BPE; the gate asks `/tokenize`), 516 + 1 = 517: the tower's 28-pixel align on a 448-pixel
//!   edge and the two delimiter ids give each span its 258.
//! - (s3) `--hf <owner>/<name>:fixture` with no `-m`, no `--mmproj` and no `--no-mmproj`, offline: the gate
//!   builds a cache under `--dir` (`$BLOOMERY_CACHE`), the repo's layout `<owner>/<name>/<path>`, the set's
//!   shards as symlinks to the fixture in a subdirectory and the projector at the repo root as unsloth's
//!   repos keep them, each link beside a `.verified` marker holding the sha256 of the file it points at
//!   (the gate hashes them, in a thread that runs beside (s1) and (s2)). The server runs under
//!   `HF_HUB_OFFLINE=1`: the set's `hf offline` record and the projector's, an `hf file` record per shard
//!   and one for the projector, all `offline-cached`. Whether the seat keeps the repo's projector is the
//!   plan's word (`vision_seat::kept`): the GLM seat's `--plan` run of the same arguments says so before the
//!   start, the V4.1 seat (which has no `--plan`) says so in its own start's log. Kept, the checks of (s2)
//!   follow; dropped, `/props` names no image input, no `vision` record is printed and an image chat is
//!   the 400 "image input is not supported". The cache is removed whatever the outcome.
//! - (s4) no unregistered kind: across every server stderr of the run, `--plan` runs included, no line holds
//!   "does not print this kind" or "panicked".
//!
//! A clause's failure never ends the run: an error inside a clause (a server that exits before it listens,
//! a kind the log is not read by) is that clause's FAIL line, and (s4) reads every log the run wrote.
//!
//! The GLM seat's `--plan` run of each start's arguments prints its records and exits before any card is
//! opened; its rc must be 0.
//!
//! The fixture tier is the only tier with a file this gate may start a server on; the real tier ends by
//! name before anything runs.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::JoinHandle;
use std::time::Duration;

use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::{self, Fields, Kind};
use bloomery_gpu_gates::serve_client::{
    self, Served, curl, json_of, rendered, server_log, tokenized,
};
use bloomery_gpu_gates::tier::{self, Tier};
use bloomery_gpu_gates::vision_seat::{self, EncoderSeat, Flags, Projector};
use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
use gguf::Split;
use refset::arch::deepseek41v::{VISION, VISION_SET};
use refset::arch::glm5next::vis::MMPROJ_BF16;
use refset::vision::VisionSet;
use serde_json::{Value, json};

const NAME: &str = "gate_serve_start";
const USAGE: &str = "usage: gate_serve_start --family glm|ds41 --dir <out>";

/// A server's polls for its `listening` line, 2 s apart: a fixture loads in a fraction of this.
const POLLS: usize = 60;
const POLL: Duration = Duration::from_secs(2);
/// The chat requests' length: the fixture's weights say nothing, the request only has to be served.
const MAX_TOKENS: u64 = 8;
/// The one question every chat asks.
const ASK: &str = "Describe the pattern in this image in one sentence.";
/// The image of the image chats, relative to the repository root, and its sha256.
const IMAGE: &str = "tools/ref/clefvis/images/free-448x448.png";
const IMAGE_SHA256: &str = "4646fb28084e9f52640ae7ab2ca1bbd48409aaeeb35c916299edb56196f2c4c2";
/// The images of the served image chat: the same image, this many times in one user message.
const IMAGES: usize = 2;
/// The text part between two images of that message: one byte, so one token of a byte-level BPE. GLM's join
/// rule (llama-server's) writes no separator after a placeholder, so this part adds its own tokens and no more.
const BETWEEN: &str = ".";
/// The positions the GLM tower's grid gives a 448 x 448 image over a text request's: the edge aligns to 28
/// pixels (a patch of 14, merged 2 x 2), so the grid is 16 x 16, and the two delimiter ids go round the
/// expanded middle id.
const GLM_IMAGE_EXTRA: u64 = 16 * 16 + 2;
/// The quant tag of the fixture's set name (`<stem>-fixture`) and the directory the cache keeps its shards in.
const QUANT: &str = "fixture";
const SET_DIR: &str = "fixture";
/// What a server's log must never hold.
const NEEDLES: [&str; 2] = ["does not print this kind", "panicked"];

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with(NAME, run())
}

/// The family whose seat the gate starts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    Glm,
    Ds41,
}

impl Family {
    fn parse(v: &str) -> Result<Family, GateError> {
        match v {
            "glm" => Ok(Family::Glm),
            "ds41" => Ok(Family::Ds41),
            other => Err(format!("--family is glm or ds41, not {other}").into()),
        }
    }

    fn word(self) -> &'static str {
        match self {
            Family::Glm => "glm",
            Family::Ds41 => "ds41",
        }
    }

    /// The kinds the family's seat registers: the list its stderr is read by.
    fn kinds(self) -> &'static [&'static Kind] {
        match self {
            Family::Glm => record::BLOOMERY_SERVE_GLM,
            Family::Ds41 => record::BLOOMERY_SERVE_DS41,
        }
    }

    /// The `clip.projector_type` of the family's projector.
    fn projector_type(self) -> &'static str {
        match self {
            Family::Glm => vision::arch::glm5v::PROJECTOR_TYPE,
            Family::Ds41 => vision::arch::deepseek41v::PROJECTOR_TYPE,
        }
    }

    /// The `--hf` repo the cache stands for.
    fn repo(self) -> &'static str {
        match self {
            Family::Glm => "bloomery-fixture/glm5next",
            Family::Ds41 => "bloomery-fixture/v41",
        }
    }

    /// The family's projector file: GLM's pinned one, V4.1's from the vision set's manifest (the file
    /// the V4.1 image gates load).
    fn projector(self) -> Result<PathBuf, GateError> {
        match self {
            Family::Glm => Ok(PathBuf::from(MMPROJ_BF16)),
            Family::Ds41 => {
                let dir = VISION.path(VISION_SET);
                let set = VisionSet::read(&dir)?;
                if !set.complete {
                    return Err(format!("{}: the set is not complete", dir.display()).into());
                }
                Ok(set.mmproj)
            }
        }
    }
}

struct Args {
    family: Family,
    dir: PathBuf,
}

fn parse_args() -> Result<Args, GateError> {
    let (mut family, mut dir) = (None, None);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
        match flag.as_str() {
            "--family" => family = Some(Family::parse(&v)?),
            "--dir" => dir = Some(PathBuf::from(v)),
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    match (family, dir) {
        (Some(family), Some(dir)) => Ok(Args { family, dir }),
        _ => Err(USAGE.into()),
    }
}

/// The projector opened on the host and the image the chats send, with what the image costs.
struct Images {
    /// The projector file.
    file: PathBuf,
    proj: Projector,
    png: Vec<u8>,
    /// Where the seat rule puts the tower under `--place a` with both cards in view, read before any
    /// server holds a card.
    seat: EncoderSeat,
    /// The positions the projector's own preparation gives the image's placeholder.
    span: usize,
    /// The id the placeholder is.
    token: u32,
}

/// The gate's image, its sha256 checked.
fn read_image() -> Result<Vec<u8>, GateError> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(IMAGE);
    let png = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let got = hf::fetch::digest(&path, hf::Check::Sha256, u64::try_from(png.len())?)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if got != IMAGE_SHA256 {
        return Err(format!(
            "{}: sha256 {got}, the gate's image is {IMAGE_SHA256}",
            path.display()
        )
        .into());
    }
    Ok(png)
}

fn open_images(family: Family) -> Result<Images, GateError> {
    let file = family.projector()?;
    // The cache names the projector for the `--hf` rule that finds it: `mmproj` in the file name.
    if !file_name(&file).contains("mmproj") {
        return Err(format!(
            "{}: a projector the cache can name has `mmproj` in its file name",
            file.display()
        )
        .into());
    }
    let proj = Projector::open(&file, &Flags::default(), &[family.projector_type()])?;
    let seat = vision_seat::encoder_seat(Place::A, proj.card.total())?;
    let png = read_image()?;
    let span = proj.media.prepare(&vision::Rgb8::from_png(&png)?)?.span_len;
    let token = proj.media.image_token();
    Ok(Images {
        file,
        proj,
        png,
        seat,
        span,
        token,
    })
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// One verdict line, `check <name>: PASS|FAIL`, as the bool it printed.
fn check(name: &str, pass: bool) -> bool {
    let mut ok = true;
    serve_client::check(&mut ok, name, pass);
    ok
}

/// A clause's answer: its verdict, or, for an error inside it, the error and a FAIL line.
fn guarded(clause: &str, r: Result<bool, GateError>) -> bool {
    match r {
        Ok(pass) => pass,
        Err(e) => {
            println!("{clause}: {e}");
            check(&format!("{clause}_ran_to_its_end"), false)
        }
    }
}

/// The last `n` characters of `s`: the end of a log, where a server says why it ended.
fn tail(s: &str, n: usize) -> &str {
    let start = s.char_indices().rev().nth(n).map_or(0, |(i, _)| i);
    &s[start..]
}

/// Whether the seat dropped the `--hf` repo's projector: the one named line `vision_seat::kept` prints
/// (it is not a record kind, so the log's kinds cannot read it).
fn projector_dropped(stderr: &str) -> bool {
    stderr.lines().any(|l| {
        l.starts_with("vision") && (l.contains("is not loaded") || l.contains("not-loaded"))
    })
}

/// A server's command: `bloomery-serve` beside this binary with no `BLOOMERY_REF_MODEL`, and the cache the
/// start reads (`HF_HUB_OFFLINE=1` and `BLOOMERY_CACHE`), or neither variable for a start that names a file.
fn command(cache: Option<&Path>) -> Result<Command, GateError> {
    let mut cmd = serve_client::serve_cmd()?;
    match cache {
        Some(root) => {
            cmd.env("HF_HUB_OFFLINE", "1").env("BLOOMERY_CACHE", root);
        }
        None => {
            cmd.env_remove("HF_HUB_OFFLINE")
                .env_remove("BLOOMERY_CACHE");
        }
    }
    Ok(cmd)
}

/// A running server: its handle, the address it printed and its stderr file.
struct Up {
    served: Served,
    addr: String,
    err: PathBuf,
}

impl Up {
    /// The server's stderr so far.
    fn text(&self) -> Result<String, GateError> {
        Ok(String::from_utf8_lossy(&std::fs::read(&self.err)?).into_owned())
    }
}

/// A `--plan` run of a start's arguments: its exit code and stderr.
struct PlanRun {
    rc: Option<i32>,
    text: String,
}

/// The throwaway `--hf` cache; its directory and the symlinks in it go when it drops (a symlink is removed,
/// never followed).
struct CacheDir(PathBuf);

impl Drop for CacheDir {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!("could not remove {}: {e}", self.0.display());
        }
    }
}

/// `target` linked into `dir` under its own name, beside a `.verified` marker holding `sha`.
fn link(target: &Path, dir: &Path, sha: &str) -> Result<(), GateError> {
    let target = std::path::absolute(target)?;
    let at = dir.join(file_name(&target));
    std::os::unix::fs::symlink(target, &at)?;
    let mut marker = at.into_os_string();
    marker.push(".verified");
    std::fs::write(marker, format!("{sha}\n"))?;
    Ok(())
}

/// The sha256 of each file, one thread a file; an error names the file.
type Hashed = JoinHandle<Result<Vec<String>, String>>;

fn hash_in_background(files: Vec<PathBuf>) -> Hashed {
    std::thread::spawn(move || {
        std::thread::scope(|s| {
            let jobs: Vec<_> = files.iter().map(|f| s.spawn(move || sha256(f))).collect();
            jobs.into_iter()
                .map(|j| {
                    j.join()
                        .map_err(|_| "a hashing thread panicked".to_owned())?
                })
                .collect::<Result<Vec<String>, String>>()
        })
    })
}

fn sha256(p: &Path) -> Result<String, String> {
    let size = std::fs::metadata(p)
        .map_err(|e| format!("{}: {e}", p.display()))?
        .len();
    hf::fetch::digest(p, hf::Check::Sha256, size).map_err(|e| format!("{}: {e}", p.display()))
}

/// `messages` of the text chat.
fn text_messages() -> Value {
    json!([{"role": "user", "content": ASK}])
}

/// `messages` of the image chat: `images` copies of the image as `data:` URLs, a [`BETWEEN`] text part
/// between each two, then the question.
fn image_messages(png: &[u8], images: usize) -> Value {
    let url = format!("data:image/png;base64,{}", serve_client::base64(png));
    let mut parts = Vec::new();
    for i in 0..images {
        if i > 0 {
            parts.push(json!({"type": "text", "text": BETWEEN}));
        }
        parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
    }
    parts.push(json!({"type": "text", "text": ASK}));
    json!([{"role": "user", "content": parts}])
}

/// One chat request at temperature 0: the status and the body.
fn chat(url: &dyn Fn(&str) -> String, messages: &Value) -> Result<(u16, String), GateError> {
    let body = json!({"messages": messages, "max_tokens": MAX_TOKENS, "temperature": 0});
    curl(&url("/v1/chat/completions"), Some(&body), false)
}

/// `usage.prompt_tokens` of a chat response body.
fn prompt_tokens(body: &str) -> Option<u64> {
    serde_json::from_str::<Value>(body)
        .ok()?
        .get("usage")?
        .get("prompt_tokens")?
        .as_u64()
}

fn health(url: &dyn Fn(&str) -> String) -> Result<bool, GateError> {
    let (st, body) = curl(&url("/health"), None, false)?;
    Ok(st == 200 && body.contains("\"ok\""))
}

/// Every `.err` file under `dir`: the stderr of each server and each `--plan` run.
fn logs_under(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            logs_under(&path, out)?;
        } else if path.extension().is_some_and(|x| x == "err") {
            out.push(path);
        }
    }
    Ok(())
}

struct Gate {
    family: Family,
    /// `--dir`, absolute: a start's directory is under it, and so is the cache.
    dir: PathBuf,
    /// The fixture's first shard.
    model: PathBuf,
    /// Every shard of the fixture's set.
    shards: Vec<PathBuf>,
    /// The projector and the image; the error is the reason they could not be opened, which (s2) and
    /// (s3) report.
    images: Result<Images, String>,
}

impl Gate {
    fn open(args: Args) -> Result<Gate, GateError> {
        std::fs::create_dir_all(&args.dir)?;
        let dir = args.dir.canonicalize()?;
        let model = ref_model_path()?;
        let split = Split::open(&model).map_err(|e| format!("open {}: {e}", model.display()))?;
        let shards = (0..split.shard_count())
            .map(|i| {
                split
                    .shard_path(i)
                    .map(Path::to_path_buf)
                    .ok_or_else(|| GateError::from(format!("shard {i} has no path")))
            })
            .collect::<Result<Vec<PathBuf>, GateError>>()?;
        let images = open_images(args.family).map_err(|e| e.to_string());
        Ok(Gate {
            family: args.family,
            dir,
            model,
            shards,
            images,
        })
    }

    fn images(&self) -> Result<&Images, GateError> {
        self.images.as_ref().map_err(|e| GateError::from(e.clone()))
    }

    /// The sha256 of the shards and then the projector, computed beside the plain and the `--mmproj` starts.
    fn hash_in_background(&self) -> Hashed {
        let mut files = self.shards.clone();
        if let Ok(im) = &self.images {
            files.push(im.file.clone());
        }
        hash_in_background(files)
    }

    /// The arguments every start shares: the loopback address and an ephemeral port.
    fn base_args() -> Vec<String> {
        ["--host", "127.0.0.1", "--port", "0"]
            .map(str::to_owned)
            .to_vec()
    }

    /// `-m <fixture>` and the base arguments.
    fn file_args(&self) -> Vec<String> {
        let mut args = vec!["-m".to_owned(), self.model.display().to_string()];
        args.extend(Self::base_args());
        args
    }

    /// The `--plan` run of `args` (the GLM seat's flag; the V4.1 seat has none, and `None` is its answer).
    fn plan_run(
        &self,
        tag: &str,
        args: &[String],
        cache: Option<&Path>,
    ) -> Result<Option<PlanRun>, GateError> {
        if self.family != Family::Glm {
            return Ok(None);
        }
        let dir = self.dir.join(format!("{tag}_plan"));
        std::fs::create_dir_all(&dir)?;
        let out = command(cache)?
            .args(args)
            .arg("--plan")
            .stdin(Stdio::null())
            .output()?;
        let text = String::from_utf8_lossy(&out.stderr).into_owned();
        std::fs::write(dir.join("plan.err"), &text)?;
        std::fs::write(dir.join("plan.out"), out.stdout)?;
        println!(
            "{tag}: --plan rc {:?}; {}",
            out.status.code(),
            self.line_of(&text, &record::PLAN)
        );
        Ok(Some(PlanRun {
            rc: out.status.code(),
            text,
        }))
    }

    /// The first record of `kind` in `text`, as its line.
    fn line_of(&self, text: &str, kind: &Kind) -> String {
        match record::Log::of(text, self.family.kinds()).first(kind) {
            Ok(Some(f)) => f.line().to_owned(),
            Ok(None) => format!("no `{}` record", kind.name),
            Err(e) => format!("unreadable: {e}"),
        }
    }

    /// A server started with `args` in `<dir>/<tag>` and waited on until it listens.
    fn start(&self, tag: &str, args: &[String], cache: Option<&Path>) -> Result<Up, GateError> {
        let dir = self.dir.join(tag);
        std::fs::create_dir_all(&dir)?;
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        println!("{tag}: bloomery-serve {}", argv.join(" "));
        let mut served = Served::spawn_cmd(command(cache)?, &argv, &dir)?;
        let err = dir.join("server.err");
        let addr = served
            .address(&err, POLLS, POLL)
            .map_err(|e| GateError::from(tail(&e.to_string(), 3000).to_owned()))?;
        println!("{tag}: pid {} listening on {addr}", served.child.id());
        Ok(Up { served, addr, err })
    }

    /// (s1).
    fn plain(&self) -> Result<bool, GateError> {
        let args = self.file_args();
        let mut ok = true;
        if let Some(p) = self.plan_run("s1", &args, None)? {
            ok &= check("s1_plan_run_exits_0", p.rc == Some(0));
        }
        let mut up = self.start("s1", &args, None)?;
        let addr = up.addr.clone();
        let url = |p: &str| format!("http://{addr}{p}");
        ok &= check("s1_health_ok", health(&url)?);
        let (st, body) = chat(&url, &text_messages())?;
        let t0 = prompt_tokens(&body);
        println!("s1: text chat HTTP {st}, prompt_tokens {t0:?}");
        ok &= check("s1_text_chat_200", st == 200 && t0.is_some_and(|n| n > 0));
        let (st, body) = curl(&url("/props"), None, false)?;
        let props = json_of("/props", st, &body)?;
        ok &= check(
            "s1_props_names_no_image_input",
            props["modalities"]["vision"] == json!(false),
        );
        let text = up.text()?;
        println!(
            "s1: {}\ns1: {}",
            self.line_of(&text, &record::PLACE_UNSET),
            self.line_of(&text, &record::PLAN)
        );
        println!("s1: server stopped: {}", up.served.stop()?);
        Ok(ok)
    }

    /// (s2).
    fn mmproj(&self) -> Result<bool, GateError> {
        let im = self.images()?;
        let mut args = self.file_args();
        args.extend(["--place".to_owned(), "a".to_owned()]);
        args.extend(["--mmproj".to_owned(), im.file.display().to_string()]);
        let mut ok = true;
        if let Some(p) = self.plan_run("s2", &args, None)? {
            ok &= check("s2_plan_run_exits_0", p.rc == Some(0));
        }
        let mut up = self.start("s2", &args, None)?;
        ok &= self.image_checks(&up, "s2", im, true)?;
        println!("s2: server stopped: {}", up.served.stop()?);
        Ok(ok)
    }

    /// (s3): `hashed` is the thread of [`Gate::hash_in_background`].
    fn hf(&self, hashed: Hashed) -> Result<bool, GateError> {
        let im = self.images()?;
        let hashes = hashed.join().map_err(|_| "the hashing thread panicked")??;
        let cache = self.build_cache(&hashes, im)?;
        let root = cache.0.as_path();
        let mut args = vec!["--hf".to_owned(), format!("{}:{QUANT}", self.family.repo())];
        args.extend(Self::base_args());
        args.extend(["--place".to_owned(), "a".to_owned()]);
        let mut ok = true;
        let plan = self.plan_run("s3", &args, Some(root))?;
        if let Some(p) = &plan {
            ok &= check("s3_plan_run_exits_0", p.rc == Some(0));
        }
        let mut up = self.start("s3", &args, Some(root))?;
        let dropped = projector_dropped(&up.text()?);
        // The plan decides what the start must do; the start's own log decides where there is no plan run.
        let planned = plan.as_ref().map(|p| projector_dropped(&p.text));
        if let Some(planned) = planned {
            ok &= check("s3_start_keeps_what_the_plan_kept", planned == dropped);
        }
        println!(
            "s3: the repo's projector is {}",
            if planned.unwrap_or(dropped) {
                "dropped by the plan"
            } else {
                "kept by the plan"
            }
        );
        ok &= self.hf_records(&up, im)?;
        ok &= self.image_checks(&up, "s3", im, !planned.unwrap_or(dropped))?;
        println!("s3: server stopped: {}", up.served.stop()?);
        Ok(ok)
    }

    /// The cache of (s3) in `<dir>/hf`: the shards under [`SET_DIR`] and the projector at the repo's
    /// root, each linked to its file beside the sha256 `hashes` holds (the shards', then the
    /// projector's).
    fn build_cache(&self, hashes: &[String], im: &Images) -> Result<CacheDir, GateError> {
        let root = self.dir.join("hf");
        let _ = std::fs::remove_dir_all(&root);
        let cache = CacheDir(root.clone());
        let repo = root.join(self.family.repo());
        let set_dir = repo.join(SET_DIR);
        std::fs::create_dir_all(&set_dir)?;
        let Some((projector_sha, shard_shas)) = hashes.split_last() else {
            return Err("the hashing thread returned no digest".into());
        };
        if shard_shas.len() != self.shards.len() {
            return Err(format!(
                "{} digests for {} shards and a projector",
                hashes.len(),
                self.shards.len()
            )
            .into());
        }
        for (shard, sha) in self.shards.iter().zip(shard_shas) {
            link(shard, &set_dir, sha)?;
        }
        link(&im.file, &repo, projector_sha)?;
        println!(
            "s3: cache {} holds {} shards under {SET_DIR}/ and {}",
            root.display(),
            self.shards.len(),
            file_name(&im.file)
        );
        Ok(cache)
    }

    /// The `hf` records of the offline start: the set's and the projector's `hf offline`, an `hf file`
    /// for each shard and one for the projector, every one `offline-cached`.
    fn hf_records(&self, up: &Up, im: &Images) -> Result<bool, GateError> {
        let log = server_log(&up.err, self.family.kinds())?;
        let offline = log.all(&record::HF_OFFLINE)?;
        let files = log.all(&record::HF_FILE)?;
        println!(
            "s3: {} hf offline records, {} hf file records",
            offline.len(),
            files.len()
        );
        let mut want: Vec<String> = self
            .shards
            .iter()
            .map(|s| format!("{SET_DIR}/{}", file_name(s)))
            .collect();
        want.push(file_name(&im.file));
        want.sort();
        let mut got = files
            .iter()
            .map(|f| f.word("file").map(str::to_owned))
            .collect::<Result<Vec<String>, _>>()?;
        got.sort();
        let cached = files
            .iter()
            .all(|f| f.word("state").is_ok_and(|s| s == "offline-cached"));
        let reasons = offline
            .iter()
            .all(|f| f.text("reason").is_ok_and(|r| r.contains("HF_HUB_OFFLINE")));
        let mut ok = check(
            "s3_hf_offline_records_set_and_projector",
            offline.len() == 2 && reasons,
        );
        ok &= check(
            "s3_hf_file_records_every_file_offline_cached",
            got == want && cached,
        );
        Ok(ok)
    }

    /// The requests and records of a start that serves `kept` image input or does not: `/health`, one
    /// text chat, `/props`, the `vision` record, and the image chat.
    fn image_checks(&self, up: &Up, tag: &str, im: &Images, kept: bool) -> Result<bool, GateError> {
        let addr = up.addr.clone();
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = check(&format!("{tag}_health_ok"), health(&url)?);
        let (st, body) = chat(&url, &text_messages())?;
        let t0 = prompt_tokens(&body);
        println!("{tag}: text chat HTTP {st}, prompt_tokens {t0:?}");
        ok &= check(
            &format!("{tag}_text_chat_200"),
            st == 200 && t0.is_some_and(|n| n > 0),
        );
        let (st, body) = curl(&url("/props"), None, false)?;
        let props = json_of("/props", st, &body)?;
        ok &= check(
            &format!("{tag}_props_vision_is_{kept}"),
            props["modalities"]["vision"] == json!(kept),
        );
        let records = server_log(&up.err, self.family.kinds())?.all(&record::VISION)?;
        if !kept {
            ok &= check(&format!("{tag}_no_vision_record"), records.is_empty());
            let (st, body) = chat(&url, &image_messages(&im.png, 1))?;
            println!("{tag}: image chat HTTP {st}: {}", tail(&body, 200));
            ok &= check(
                &format!("{tag}_image_chat_is_the_400_refusal"),
                st == 400 && body.contains("image input is not supported"),
            );
            return Ok(ok);
        }
        ok &= check(&format!("{tag}_one_vision_record"), records.len() == 1);
        if let [rec] = records.as_slice() {
            println!("{tag}: {}", rec.line());
            ok &= self.tower_checks(tag, rec, im)?;
        }
        let messages = image_messages(&im.png, IMAGES);
        let ids = rendered(&url, messages.clone())?;
        let between = tokenized(&url, BETWEEN)?.len();
        let (st, body) = chat(&url, &messages)?;
        let got = prompt_tokens(&body);
        // Each placeholder is one id of the rendered prompt and becomes a span of `im.span`.
        let want = (ids.len() + IMAGES * im.span).checked_sub(IMAGES);
        println!(
            "{tag}: {IMAGES}-image chat HTTP {st}, prompt_tokens {got:?}; rendered {} ids, span {}, \
             {between} id(s) between the images",
            ids.len(),
            im.span
        );
        ok &= check(&format!("{tag}_image_chat_200"), st == 200);
        ok &= check(
            &format!("{tag}_the_placeholders_expand_to_the_spans"),
            ids.iter().filter(|&&i| i == im.token).count() == IMAGES
                && got == want.and_then(|w| u64::try_from(w).ok()),
        );
        if self.family == Family::Glm {
            let extra = GLM_IMAGE_EXTRA * u64::try_from(IMAGES)? + u64::try_from(between)?;
            ok &= check(
                &format!("{tag}_image_prompt_is_the_text_prompt_plus_the_spans"),
                got.is_some() && got == t0.map(|t| t + extra),
            );
        }
        Ok(ok)
    }

    /// The tower's seat and bytes: both cards in view and the placement `a` leave the tower off the
    /// plan, and the `vision` record is the one the seat rule and the projector's card figure give.
    fn tower_checks(&self, tag: &str, rec: &Fields, im: &Images) -> Result<bool, GateError> {
        let want = &im.seat;
        let mut ok = check(
            &format!("{tag}_tower_sits_off_plan"),
            want.word() == "off-plan",
        );
        ok &= check(
            &format!("{tag}_vision_record_is_the_towers"),
            rec.word("card")? == bloomery_gpu_gates::record::name_token(&want.name)
                && rec.word("seat")? == want.word()
                && rec.u64("weights")? == im.proj.card.weights
                && rec.u64("scratch")? == im.proj.card.scratch
                && rec.u64("cap")? == u64::try_from(im.proj.cap)?,
        );
        Ok(ok)
    }

    /// (s4).
    fn no_unregistered_kind(&self) -> Result<bool, GateError> {
        let mut logs = Vec::new();
        logs_under(&self.dir, &mut logs)?;
        logs.sort();
        let mut hits = 0usize;
        for log in &logs {
            let text = String::from_utf8_lossy(&std::fs::read(log)?).into_owned();
            for line in text
                .lines()
                .filter(|l| NEEDLES.iter().any(|n| l.contains(n)))
            {
                hits += 1;
                if hits <= 5 {
                    println!("s4: {}: {}", log.display(), tail(line, 300));
                }
            }
        }
        println!(
            "s4: {} logs read, {hits} lines hold an unregistered kind or a panic",
            logs.len()
        );
        Ok(check(
            "s4_no_unregistered_kind_and_no_panic",
            !logs.is_empty() && hits == 0,
        ))
    }
}

fn run() -> Result<(), GateError> {
    let tier = Tier::from_env()?;
    if tier != Tier::Fixture {
        return Err(format!(
            "0 clauses on the {} tier: this gate starts the server on the family's fixture file and has no real-tier set. \
             Run it under BLOOMERY_TIER=fixture",
            tier.name()
        )
        .into());
    }
    let _levers = bloomery_levers::at_main(&tier::acts_on(&[])?)?;
    let gate = Gate::open(parse_args()?)?;
    println!(
        "{NAME}: --family {} on {}",
        gate.family.word(),
        gate.model.display()
    );
    let hashed = gate.hash_in_background();
    tier::sc("s1_plain_start")?;
    let mut ok = guarded("s1", gate.plain());
    tier::sc("s2_mmproj_start")?;
    ok &= guarded("s2", gate.mmproj());
    tier::sc("s3_hf_offline_start")?;
    ok &= guarded("s3", gate.hf(hashed));
    tier::sc("s4_no_unregistered_kind")?;
    ok &= guarded("s4", gate.no_unregistered_kind());
    println!("{}", tier::tally_line());
    if !ok {
        return Err(checks_failed());
    }
    println!(
        "{NAME}: PASS (fixture tier, --family {})",
        gate.family.word()
    );
    Ok(())
}
