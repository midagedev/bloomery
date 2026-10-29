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
//! - (s5) `--chat-template-file <file>` is a flag of every seat: a start on the fixture's own
//!   `tokenizer.chat_template` behind a marker, and `/apply-template` of a call-carrying conversation renders
//!   the marker and then exactly what a start on the same file without the marker rendered of it, so the file's
//!   text is the template served. A seat whose default is the file's own template (GLM) renders the unmarked file
//!   as (s1)'s plain start did; the V4.1 seat, whose default is DeepSeek's official encoding, renders it
//!   otherwise, so the file replaced the default. A chat is a 200. (s5 runs before (s4), so (s4) reads its
//!   log.)
//!
//! - (s6) the need hook (`BLOOMERY_NEED_HOOK`): `-m <fixture>` with a hook that reads its standard input and
//!   exits 75 with a line. The process ends 75 before it listens, its log holds the hook's line, and the hook's
//!   input is the `need` record the server printed at plan time, byte for byte. A second start with a hook that
//!   exits 0 listens, and that hook's input is its own `need` record. The hook is consulted by the placed load
//!   before its own open of the card: that start's hook waits `HOOK_WAIT_S` seconds, and the load's `context_s`
//!   phase (the open of the card) stays under half of them. A consult after the open puts the wait in that phase.
//! - (s7) a start by catalog name (`--model <name>`), offline from a cache laid out as the catalog row's repo: the
//!   family's default model's first verified GGUF row (read from `tools/catalog.tsv`), its shards the fixture's, linked
//!   under the row's own file names, each beside a `.verified` marker. The `pick` event names the cached row (variant,
//!   source, `cached`, nothing to download). `pick` and `bound` come before `ready`, which repeats the `bound` url; a
//!   text chat is a 200; every stdout line is an event of the one schema; a SIGTERM ends the server with exit 0 and a
//!   `stopped` event whose cause is `signal`.
//! - (s8) `--model <name> --dry-run --json` prints one JSON document and exits 0 without a start: one model entry whose
//!   pick is the cached row, and `start`, the argument list a real start would run: `--model <name>:<variant>` and the
//!   flags given, no `--hf` and no `--revision`, so the start replays the pick at the row's pinned commit. That start's
//!   `--plan --json` prints the same `pick` (variant, source, revision) and a plan that equals the dry run's pick in place,
//!   context, slots and paged bytes, and is the plan of the fixture file (the same need bytes and limit).
//! - (s9) the loading window of (s7)'s start: from the `bound` event on, `GET /health` answers 503 `loading` (phase,
//!   byte counts and the `error` object llama-server answers with) until the first 200; no 200 is answered before the
//!   `ready` event is on stdout, and no 503 after it.
//! - (s10) the exit table, on the refusals a start meets, each ending in an `error` event of the code's name and
//!   status: an unknown name is 64; a `--model` value that is a path is 64 with a `next` that names `-m`; an embedding
//!   file is 65 and names `--embd-model`; a card budget below the dense trunk is 80, and its `next` offers only variants
//!   smaller than the one asked, the next smaller one among them (the families' default models have none, so theirs
//!   offers no variant); a `BLOOMERY_*` value no lever takes is 78 and names it; a port another process holds is 83,
//!   found before the load, with `--port 0` in `next`.
//!
//! A clause's failure never ends the run: an error inside a clause (a server that exits before it listens,
//! a kind the log is not read by) is that clause's FAIL line, and (s4) reads every log the run wrote.
//!
//! The GLM seat's `--plan` run of each start's arguments prints its records and exits before any card is
//! opened; its rc must be 0.
//!
//! The fixture tier is the only tier with a file this gate may start a server on; the real tier ends by
//! name before anything runs.

use std::cell::{OnceCell, RefCell};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::{self, Fields, Kind};
use bloomery_gpu_gates::serve_client::{
    self, Served, curl, json_of, rendered, server_log, tokenized,
};
use bloomery_gpu_gates::side_seat;
use bloomery_gpu_gates::tier::{self, Tier};
use bloomery_gpu_gates::vision_seat::{self, EncoderSeat, Flags, Projector};
use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path};
use bloomery_levers::CARD_BUDGET;
use gguf::Split;
use gguf::write::{Layout, TensorDecl, Writer};
use refset::arch::deepseek41v::{VISION, VISION_SET};
use refset::arch::glm5next::vis::MMPROJ_BF16;
use refset::vision::VisionSet;
use serde_json::{Value, json};
use serve::event::SCHEMA as EVENT_SCHEMA;
use serve::exit::Code;

const NAME: &str = "gate_serve_start";
const USAGE: &str = "usage: gate_serve_start --family glm|ds41 --dir <out>";

/// A server's polls for its `listening` line, 2 s apart: a fixture loads in a fraction of this.
const POLLS: usize = 60;
const POLL: Duration = Duration::from_secs(2);
/// Seconds the go-ahead hook of (s6) waits before it answers: far longer than the card's open takes.
const HOOK_WAIT_S: u32 = 4;
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
/// The text a template file starts with, which the rendered prompt starts with if the file is the template.
const FILE_MARK: &str = "FILE-TEMPLATE|";
/// What a server's log must never hold.
const NEEDLES: [&str; 2] = ["does not print this kind", "panicked"];
/// The model catalog's committed print, relative to the repository root: the by-name clauses take the family's row
/// from it.
const CATALOG: &str = "tools/catalog.tsv";
/// The columns of a catalog row.
const CATALOG_COLUMNS: usize = 11;
/// The schema the `--plan --json` line names.
const PLAN_SCHEMA: &str = "bloomery.plan/1";
/// How often a start's stdout and `/health` are read while the gate waits for the server.
const EVENT_GAP: Duration = Duration::from_millis(50);
/// The longest a start that must refuse, or a stop that must be orderly, may take.
const END_WAIT: Duration = Duration::from_secs(120);
/// A card budget below the floor of any plan.
const BUDGET_BELOW_FLOOR: &str = "64M";
/// A card budget no lever parser takes.
const BUDGET_NOT_BYTES: &str = "banana";
/// A model name no catalog holds.
const NO_SUCH_MODEL: &str = "no-such-model";

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
    let got = gguf::digest::sha256_file(&path)?;
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
    let token = proj
        .media
        .image_token()
        .ok_or("the projector has no chat image token")?;
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

/// A hook script in `dir`: it writes its standard input to `<name>.saw`, waits `wait` seconds, prints `line` and
/// exits `rc`. The script and the file it writes.
fn hook_script(
    dir: &Path,
    name: &str,
    line: &str,
    rc: u8,
    wait: u32,
) -> Result<(PathBuf, PathBuf), GateError> {
    let saw = dir.join(format!("{name}.saw"));
    let script = dir.join(format!("{name}.sh"));
    // Only the writer child opens the script for writing, so concurrent spawns cannot
    // inherit a writable descriptor and make the hook's exec fail with ETXTBSY.
    let written = Command::new("/bin/sh")
        .args(["-c", "printf '%s\\n' \"$2\" > \"$1\"", "hook-script"])
        .arg(&script)
        .arg(format!(
            "#!/bin/sh\ncat > '{}'\nsleep {wait}\necho '{line}'\nexit {rc}\n",
            saw.display()
        ))
        .status()?;
    if !written.success() {
        return Err(format!("hook script writer ended {written}").into());
    }
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))?;
    Ok((script, saw))
}

/// The child's end status, `None` if it is still running after `bound`.
fn ended(child: &mut Child, bound: Duration) -> Result<Option<ExitStatus>, GateError> {
    let until = Instant::now() + bound;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        if Instant::now() >= until {
            return Ok(None);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
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
    link_as(target, dir, &file_name(target), sha)
}

/// `target` linked into `dir` as `name`, beside a `.verified` marker holding `sha`.
fn link_as(target: &Path, dir: &Path, name: &str, sha: &str) -> Result<(), GateError> {
    let target = std::path::absolute(target)?;
    let at = dir.join(name);
    std::os::unix::fs::symlink(target, &at)?;
    let mut marker = at.into_os_string();
    marker.push(".verified");
    std::fs::write(marker, format!("{sha}\n"))?;
    Ok(())
}

/// A row of the catalog print: the columns the by-name clauses read.
struct CatalogRow {
    name: String,
    aliases: Vec<String>,
    seat: String,
    variant: String,
    format: String,
    hf: String,
    bytes: Option<u64>,
    verified: bool,
}

/// The rows of the catalog print `text`: a line of another width, a count that is none and a verdict that is neither
/// `true` nor `false` are refused by name.
fn catalog_rows(text: &str) -> Result<Vec<CatalogRow>, GateError> {
    let mut rows = Vec::new();
    let mut header = false;
    for (n, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if !header {
            if cols.first() != Some(&"name") {
                return Err(format!("{CATALOG}:{}: the header line is missing", n + 1).into());
            }
            header = true;
            continue;
        }
        let &[
            name,
            aliases,
            seat,
            _arch,
            variant,
            format,
            hf,
            _revision,
            bytes,
            verified,
            _note,
        ] = cols.as_slice()
        else {
            return Err(format!(
                "{CATALOG}:{}: {} columns, not {CATALOG_COLUMNS}",
                n + 1,
                cols.len()
            )
            .into());
        };
        let bytes = match bytes {
            "-" => None,
            b => Some(
                b.parse::<u64>()
                    .map_err(|e| format!("{CATALOG}:{}: bytes {b:?}: {e}", n + 1))?,
            ),
        };
        let verified = match verified {
            "true" => true,
            "false" => false,
            v => return Err(format!("{CATALOG}:{}: verified is {v:?}", n + 1).into()),
        };
        rows.push(CatalogRow {
            name: name.to_owned(),
            aliases: aliases.split(',').map(str::to_owned).collect(),
            seat: seat.to_owned(),
            variant: variant.to_owned(),
            format: format.to_owned(),
            hf: hf.to_owned(),
            bytes,
            verified,
        });
    }
    if rows.is_empty() {
        return Err(format!("{CATALOG}: no rows").into());
    }
    Ok(rows)
}

/// The catalog row a by-name start of a seat takes.
struct NamedRow {
    /// The catalog name: the model whose alias is the seat's word.
    model: String,
    /// The model's first verified GGUF row the automatic pick takes (a format of 8 bits or more is explicit-only).
    variant: String,
    repo: String,
    /// The variants of the model the catalog records fewer bytes for.
    smaller: Vec<String>,
    /// The next smaller variant: the largest verified row of the smaller ones in an automatic format (2 to 4 bits).
    /// For the two families this gate runs no smaller row is of another automatic format.
    step: Option<String>,
}

impl NamedRow {
    /// The row of the model that has `seat` as an alias.
    fn of(seat: &str, rows: &[CatalogRow]) -> Result<NamedRow, GateError> {
        let model = rows
            .iter()
            .find(|r| r.seat == seat && r.aliases.iter().any(|a| a == seat))
            .map(|r| r.name.clone())
            .ok_or_else(|| format!("{CATALOG}: no model has the seat word {seat} as an alias"))?;
        let of_model: Vec<&CatalogRow> = rows.iter().filter(|r| r.name == model).collect();
        let row = of_model
            .iter()
            .find(|r| {
                r.verified
                    && r.hf.contains(':')
                    && !matches!(r.format.as_str(), "gguf8" | "gguf6" | "gguf5")
            })
            .ok_or_else(|| format!("{CATALOG}: {model} has no verified GGUF row to start from"))?;
        let (repo, variant) = row
            .hf
            .split_once(':')
            .ok_or_else(|| format!("{CATALOG}: {} names no quant", row.hf))?;
        let under: Vec<&&CatalogRow> = of_model
            .iter()
            .filter(|r| matches!((r.bytes, row.bytes), (Some(a), Some(b)) if a < b))
            .collect();
        let smaller = under.iter().map(|r| r.variant.clone()).collect();
        let step = under
            .iter()
            .filter(|r| r.verified && matches!(r.format.as_str(), "gguf2" | "gguf3" | "gguf4"))
            .max_by_key(|r| r.bytes)
            .map(|r| r.variant.clone());
        Ok(NamedRow {
            variant: variant.to_owned(),
            repo: repo.to_owned(),
            model,
            smaller,
            step,
        })
    }

    /// The `--hf` value that names the row.
    fn hf(&self) -> String {
        format!("{}:{}", self.repo, self.variant)
    }
}

/// The lines of a stdout `text` that are complete: a last line with no newline is still being written.
fn complete_lines(text: &str) -> impl Iterator<Item = &str> {
    text.rfind('\n')
        .map_or("", |i| &text[..i])
        .lines()
        .filter(|l| !l.trim().is_empty())
}

/// The event a stdout line is: a JSON object that names [`EVENT_SCHEMA`] and an event.
fn event_of(line: &str) -> Option<Value> {
    serde_json::from_str::<Value>(line)
        .ok()
        .filter(|v| v["schema"] == EVENT_SCHEMA && v["event"].is_string())
}

/// The events of a server's stdout `text`; the first line that is no event is refused by name.
fn events_in(text: &str) -> Result<Vec<Value>, GateError> {
    complete_lines(text)
        .map(|l| {
            event_of(l).ok_or_else(|| {
                GateError::from(format!(
                    "a stdout line that is no event of {EVENT_SCHEMA}: {}",
                    tail(l, 200)
                ))
            })
        })
        .collect()
}

/// The events of a stdout `text` so far, a line that is no event left out: the polls read a file the server is still
/// writing, and [`events_in`] judges it whole.
fn events_seen(text: &str) -> Vec<Value> {
    complete_lines(text).filter_map(event_of).collect()
}

/// The first event named `name`.
fn event<'a>(events: &'a [Value], name: &str) -> Option<&'a Value> {
    events.iter().find(|e| e["event"] == name)
}

/// The file at `path`, read as text.
fn read_lossy(path: &Path) -> Result<String, GateError> {
    Ok(String::from_utf8_lossy(&std::fs::read(path)?).into_owned())
}

/// Whether the `ready` event is on the server's stdout.
fn ready_printed(out: &Path) -> Result<bool, GateError> {
    Ok(event(&events_seen(&read_lossy(out)?), "ready").is_some())
}

/// The events of a server's stdout once `want` holds of them. The server ending first, or the wait running out, is an
/// error that names `what` and ends with the server's stderr.
fn events_until(
    served: &mut Served,
    out: &Path,
    err: &Path,
    what: &str,
    want: &dyn Fn(&[Value]) -> bool,
) -> Result<Vec<Value>, GateError> {
    let until = Instant::now() + POLL * u32::try_from(POLLS)?;
    loop {
        let events = events_seen(&read_lossy(out)?);
        if want(&events) {
            return Ok(events);
        }
        if let Some(status) = served.child.try_wait()? {
            return Err(format!(
                "the server ended ({status}) before {what}; {}",
                tail(&read_lossy(err)?, 1500)
            )
            .into());
        }
        if Instant::now() >= until {
            return Err(format!(
                "no {what} within the wait; {}",
                tail(&read_lossy(err)?, 1500)
            )
            .into());
        }
        std::thread::sleep(EVENT_GAP);
    }
}

/// Sends SIGTERM to the process `pid`, one this gate spawned.
fn terminate(pid: u32) -> Result<(), GateError> {
    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status()?;
    if !status.success() {
        return Err(format!("kill -TERM {pid} ended {status}").into());
    }
    Ok(())
}

/// A server's `--json` answer: exactly one stdout line, a JSON object.
fn one_json_line(what: &str, stdout: &str) -> Result<Value, GateError> {
    let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
    let &[line] = lines.as_slice() else {
        return Err(format!(
            "{what}: stdout holds {} lines, not one JSON object: {}",
            lines.len(),
            tail(stdout, 300)
        )
        .into());
    };
    serde_json::from_str(line)
        .map_err(|e| GateError::from(format!("{what}: {e}: {}", tail(line, 300))))
}

/// The plan line and the events of a `--plan --json` run's stdout: every line is a JSON object, exactly one names
/// [`PLAN_SCHEMA`], and the others are events of the one schema.
fn plan_and_events(what: &str, stdout: &str) -> Result<(Value, Vec<Value>), GateError> {
    let mut plan = None;
    let mut events = Vec::new();
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)
            .map_err(|e| GateError::from(format!("{what}: {e}: {}", tail(line, 300))))?;
        if v["schema"] == PLAN_SCHEMA {
            if plan.replace(v).is_some() {
                return Err(format!("{what}: stdout holds two plan lines").into());
            }
        } else if v["schema"] == EVENT_SCHEMA {
            events.push(v);
        } else {
            return Err(format!(
                "{what}: a stdout line names the schema {}, not {PLAN_SCHEMA} or {EVENT_SCHEMA}",
                v["schema"]
            )
            .into());
        }
    }
    let plan = plan.ok_or_else(|| {
        GateError::from(format!(
            "{what}: stdout holds no line of the schema {PLAN_SCHEMA}: {}",
            tail(stdout, 300)
        ))
    })?;
    Ok((plan, events))
}

/// Whether the `error` event `e` lists the argument list `argv` in its `next`.
fn next_has(e: &Value, argv: &[&str]) -> bool {
    e["next"].as_array().is_some_and(|next| {
        next.iter().any(|list| {
            list.as_array().is_some_and(|list| {
                list.len() == argv.len() && list.iter().zip(argv).all(|(a, b)| a == b)
            })
        })
    })
}

/// Whether every key of `keys` is set in `a` and `b` and equal.
fn same(a: &Value, b: &Value, keys: &[&str]) -> bool {
    keys.iter().all(|k| !a[*k].is_null() && a[*k] == b[*k])
}

/// A GGUF file at `path` whose architecture is `arch`, with one tensor: what a start reads of a file to seat it.
fn gguf_of_arch(path: &Path, arch: &str) -> Result<(), GateError> {
    let kvs = [
        (
            gguf::GENERAL_ARCHITECTURE.to_owned(),
            gguf::Value::String(arch.to_owned()),
        ),
        // The fixture tier opens only files its generator wrote: this one carries the generator's key, and
        // the server refuses it for what it is.
        (tier::VERSION_KEY.to_owned(), gguf::Value::U32(1)),
    ];
    let name = "token_embd.weight";
    let decl = TensorDecl {
        name: name.to_owned(),
        dims: vec![4],
        type_id: 0,
        nbytes: 16,
    };
    let mut writer = Writer::new(std::fs::File::create(path)?, Layout::new(&kvs, vec![decl])?)?;
    writer.tensor(name, &[0u8; 16])?;
    writer.finish()?;
    Ok(())
}

/// What one `GET /health` saw of a start that is loading, and whether the `ready` event was on the server's stdout
/// before the request and after its answer.
struct Probe {
    status: u16,
    body: String,
    ready_before: bool,
    ready_after: bool,
}

impl Probe {
    /// Whether the answer is the 503 a loading server gives: the status word `loading`, the phase, both byte counts and
    /// the `error` object llama-server answers with.
    fn is_loading(&self) -> bool {
        self.status == 503
            && serde_json::from_str::<Value>(&self.body).is_ok_and(|b| {
                b["status"] == "loading"
                    && matches!(b["phase"].as_str(), Some("fetch" | "plan" | "load"))
                    && b["bytes_done"].is_u64()
                    && b["bytes_total"].is_u64()
                    && b["error"]["code"].as_u64() == Some(503)
                    && b["error"]["type"] == "unavailable_error"
            })
    }
}

/// A start that must end by refusing: its exit status, the events it printed and its stderr file.
struct Refusal {
    status: Option<i32>,
    events: Vec<Value>,
    err: PathBuf,
}

impl Refusal {
    /// The `error` event the process printed last.
    fn error(&self) -> Option<&Value> {
        self.events.iter().rev().find(|e| e["event"] == "error")
    }

    /// Whether the process ended with `code`'s status and said so in an `error` event: the status, the name and a hint.
    fn is(&self, code: Code) -> bool {
        self.status == Some(i32::from(code.status()))
            && self.error().is_some_and(|e| {
                e["exit"].as_u64() == Some(u64::from(code.status()))
                    && e["code"] == code.name()
                    && e["hint"].as_str().is_some_and(|h| !h.is_empty())
            })
    }

    /// Whether the `error` event's message holds `needle`.
    fn says(&self, needle: &str) -> bool {
        self.error()
            .is_some_and(|e| e["message"].as_str().is_some_and(|m| m.contains(needle)))
    }
}

/// The sha256 of each file, one thread a file; an error names the file.
type Hashed = JoinHandle<Result<Vec<String>, String>>;

fn hash_in_background(files: Vec<PathBuf>) -> Hashed {
    std::thread::spawn(move || {
        std::thread::scope(|s| {
            let jobs: Vec<_> = files
                .iter()
                .map(|f| s.spawn(move || gguf::digest::sha256_file(f)))
                .collect();
            jobs.into_iter()
                .map(|j| {
                    j.join()
                        .map_err(|_| "a hashing thread panicked".to_owned())?
                        .map_err(|e| e.to_string())
                })
                .collect::<Result<Vec<String>, String>>()
        })
    })
}

/// `messages` of the text chat.
fn text_messages() -> Value {
    json!([{"role": "user", "content": ASK}])
}

/// `messages` of a conversation whose assistant turn calls a function: the user's question, then the call with
/// its `arguments` as the JSON string clients send.
fn call_messages() -> Value {
    json!([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"Seoul\"}"},
        }]},
    ])
}

/// The prompt `/apply-template` renders of `messages`.
fn applied(url: &dyn Fn(&str) -> String, messages: &Value) -> Result<String, GateError> {
    let (st, body) = curl(
        &url("/apply-template"),
        Some(&json!({ "messages": messages })),
        false,
    )?;
    Ok(json_of("/apply-template", st, &body)?["prompt"]
        .as_str()
        .ok_or("/apply-template: no prompt")?
        .to_owned())
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
    /// `/apply-template` of [`call_messages`] on (s1)'s server, which (s5) compares its render with.
    plain_render: OnceCell<String>,
    /// The thread that hashes the shards and the projector, until [`Gate::digests`] joins it.
    hashing: RefCell<Option<Hashed>>,
    /// The sha256 of the shards and then the projector (the shards alone when no projector opened), once the
    /// hashing thread is joined.
    digests: OnceCell<Result<Vec<String>, String>>,
    /// The family's catalog row and the offline cache that holds it, which (s7), (s8) and (s10) start from.
    named: OnceCell<Result<Named, String>>,
}

/// What the by-name clauses start from: the family's catalog row and an offline cache that holds it.
struct Named {
    row: NamedRow,
    cache: CacheDir,
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
            plain_render: OnceCell::new(),
            hashing: RefCell::new(None),
            digests: OnceCell::new(),
            named: OnceCell::new(),
        })
    }

    fn images(&self) -> Result<&Images, GateError> {
        self.images.as_ref().map_err(|e| GateError::from(e.clone()))
    }

    /// Starts hashing the shards and then the projector, beside the plain and the `--mmproj` starts.
    fn hash_in_background(&self) {
        let mut files = self.shards.clone();
        if let Ok(im) = &self.images {
            files.push(im.file.clone());
        }
        *self.hashing.borrow_mut() = Some(hash_in_background(files));
    }

    /// The digests of the shards and then the projector: the hashing thread's answer, joined at the first call.
    fn digests(&self) -> Result<&[String], GateError> {
        let done = self
            .digests
            .get_or_init(|| match self.hashing.borrow_mut().take() {
                Some(thread) => thread
                    .join()
                    .map_err(|_| "the hashing thread panicked".to_owned())
                    .and_then(|hashed| hashed),
                None => Err("no hashing thread was started".to_owned()),
            });
        done.as_deref().map_err(|e| GateError::from(e.clone()))
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
        let prompt = applied(&url, &call_messages())?;
        println!("s1: call-carrying prompt {}", tail(&prompt, 160));
        let _ = self.plain_render.set(prompt);
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

    /// (s3).
    fn hf(&self) -> Result<bool, GateError> {
        let im = self.images()?;
        let cache = self.build_cache(self.digests()?, im)?;
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
            rec.word("card")? == want.name.replace(' ', "_")
                && rec.word("seat")? == want.word()
                && rec.u64("weights")? == im.proj.card.weights
                && rec.u64("scratch")? == im.proj.card.scratch
                && rec.u64("cap")? == u64::try_from(im.proj.cap)?,
        );
        Ok(ok)
    }

    /// The plain start's arguments with `--chat-template-file <file>`.
    fn template_args(&self, file: &Path) -> Vec<String> {
        let mut args = self.file_args();
        args.extend([
            "--chat-template-file".to_owned(),
            file.display().to_string(),
        ]);
        args
    }

    /// (s5): the seat takes `--chat-template-file`, and the file's text is the template it serves.
    fn template_file(&self) -> Result<bool, GateError> {
        let Some(plain) = self.plain_render.get() else {
            return Err("(s1) rendered no call-carrying prompt to compare with".into());
        };
        let split =
            Split::open(&self.model).map_err(|e| format!("open {}: {e}", self.model.display()))?;
        let template = split
            .value("tokenizer.chat_template")
            .and_then(gguf::Value::as_str)
            .ok_or("the fixture has no tokenizer.chat_template")?;
        let dir = self.dir.join("s5");
        std::fs::create_dir_all(&dir)?;
        let unmarked = dir.join("unmarked.jinja");
        std::fs::write(&unmarked, template)?;
        let mut up = self.start("s5u", &self.template_args(&unmarked), None)?;
        let addr = up.addr.clone();
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = check("s5u_health_ok", health(&url)?);
        let files = applied(&url, &call_messages())?;
        println!("s5u: prompt {}", tail(&files, 160));
        println!("s5u: server stopped: {}", up.served.stop()?);
        // The GLM seat serves the file's own template by default; the V4.1 seat serves the official encoding.
        ok &= if self.family == Family::Glm {
            check("s5_default_is_the_files_template", files == *plain)
        } else {
            check("s5_file_replaces_the_default", files != *plain)
        };
        let file = dir.join("template.jinja");
        std::fs::write(&file, format!("{FILE_MARK}{template}"))?;
        let mut up = self.start("s5", &self.template_args(&file), None)?;
        let addr = up.addr.clone();
        let url = |p: &str| format!("http://{addr}{p}");
        ok &= check("s5_health_ok", health(&url)?);
        let prompt = applied(&url, &call_messages())?;
        println!("s5: prompt {}", tail(&prompt, 160));
        ok &= check(
            "s5_prompt_is_the_files_template",
            prompt.strip_prefix(FILE_MARK) == Some(files.as_str()),
        );
        let (st, _) = chat(&url, &text_messages())?;
        ok &= check("s5_text_chat_200", st == 200);
        println!("s5: server stopped: {}", up.served.stop()?);
        Ok(ok)
    }

    /// (s6): the need hook refuses a load with 75 before the server listens, and the hook reads the
    /// `need` record the server printed.
    fn need_hook(&self) -> Result<bool, GateError> {
        let dir = self.dir.join("s6");
        let args = self.file_args();
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        let line = "the box is held for a sitting";
        let mut ok = true;
        // The refusal: the process ends 75 and never listens.
        let refuse = dir.join("refuse");
        std::fs::create_dir_all(&refuse)?;
        let (hook, saw) = hook_script(&refuse, "hook", line, 75, 0)?;
        let mut cmd = command(None)?;
        cmd.env("BLOOMERY_NEED_HOOK", &hook);
        println!(
            "s6: bloomery-serve {} with a hook that exits 75",
            argv.join(" ")
        );
        let mut served = Served::spawn_cmd(cmd, &argv, &refuse)?;
        let status = ended(&mut served.child, POLL * u32::try_from(POLLS)?)?;
        let err = refuse.join("server.err");
        let text = String::from_utf8_lossy(&std::fs::read(&err)?).into_owned();
        println!("s6: refused run ended {status:?}; {}", tail(&text, 400));
        ok &= check(
            "s6_a_refused_load_ends_75",
            status.is_some_and(|s| s.code() == Some(75)),
        );
        ok &= check("s6_the_log_names_the_hooks_line", text.contains(line));
        ok &= check(
            "s6_a_refused_load_never_listens",
            !text.contains("listening on http://"),
        );
        let need = server_log(&err, self.family.kinds())?
            .one(&record::NEED)?
            .line()
            .to_owned();
        let saw = std::fs::read_to_string(&saw)?;
        ok &= check("s6_the_hook_read_the_need_record", saw.trim_end() == need);
        // The go-ahead: the same start with a hook that exits 0 listens. It waits first, so where the wait
        // lands in the load's phases tells whether the hook answered before the card was opened.
        let go = dir.join("go");
        std::fs::create_dir_all(&go)?;
        let (hook, saw) = hook_script(&go, "hook", line, 0, HOOK_WAIT_S)?;
        let mut cmd = command(None)?;
        cmd.env("BLOOMERY_NEED_HOOK", &hook);
        println!(
            "s6: bloomery-serve {} with a hook that exits 0",
            argv.join(" ")
        );
        let mut served = Served::spawn_cmd(cmd, &argv, &go)?;
        let up_err = go.join("server.err");
        let listening = served.address(&up_err, POLLS, POLL).is_ok();
        ok &= check("s6_a_hook_that_exits_0_goes_on", listening);
        println!("s6: server stopped: {}", served.stop()?);
        let need = server_log(&up_err, self.family.kinds())?
            .one(&record::NEED)?
            .line()
            .to_owned();
        let saw = std::fs::read_to_string(&saw)?;
        ok &= check(
            "s6_the_go_ahead_hook_read_its_need_record",
            saw.trim_end() == need,
        );
        let context = server_log(&up_err, self.family.kinds())?
            .one(&record::LOAD_PHASES)?
            .f64("context_s")?;
        println!(
            "s6: the open of the card took {context} s under a hook that waited {HOOK_WAIT_S} s"
        );
        ok &= check(
            "s6_the_hook_answers_before_the_card_is_opened",
            context < f64::from(HOOK_WAIT_S) / 2.0,
        );
        Ok(ok)
    }

    /// The family's catalog row and the offline cache that holds it, made at the first call.
    fn named(&self) -> Result<&Named, GateError> {
        self.named
            .get_or_init(|| self.make_named().map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| GateError::from(e.clone()))
    }

    fn make_named(&self) -> Result<Named, GateError> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(CATALOG);
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let row = NamedRow::of(self.family.word(), &catalog_rows(&text)?)?;
        let cache = self.build_named_cache(&row)?;
        Ok(Named { row, cache })
    }

    /// The cache of the by-name clauses in `<dir>/named_hf`: the layout of the catalog row's repo, the fixture's
    /// shards linked under the file names the row's set has (the repo's name less `-GGUF`, the variant, the shard
    /// counts), each beside the sha256 of the file it points at.
    fn build_named_cache(&self, row: &NamedRow) -> Result<CacheDir, GateError> {
        let digests = self
            .digests()?
            .get(..self.shards.len())
            .ok_or("the hashing thread returned fewer digests than shards")?;
        let root = self.dir.join("named_hf");
        let _ = std::fs::remove_dir_all(&root);
        let cache = CacheDir(root.clone());
        let set_dir = root.join(&row.repo).join(&row.variant);
        std::fs::create_dir_all(&set_dir)?;
        let stem = row.repo.rsplit('/').next().unwrap_or(&row.repo);
        let stem = stem.strip_suffix("-GGUF").unwrap_or(stem);
        let count = self.shards.len();
        for (i, (shard, sha)) in self.shards.iter().zip(digests).enumerate() {
            let name = if count == 1 {
                format!("{stem}-{}.gguf", row.variant)
            } else {
                format!("{stem}-{}-{:05}-of-{count:05}.gguf", row.variant, i + 1)
            };
            link_as(shard, &set_dir, &name, sha)?;
        }
        println!(
            "named: cache {} holds {count} shards as {}/{}",
            root.display(),
            row.repo,
            row.variant
        );
        Ok(cache)
    }

    /// (s7) and (s9): the start by catalog name from the offline cache. One server runs from the `bound` event through
    /// a SIGTERM; `GET /health` is read from the `bound` event to its first 200.
    fn by_name(&self) -> Result<bool, GateError> {
        let named = self.named()?;
        let dir = self.dir.join("s7");
        std::fs::create_dir_all(&dir)?;
        let mut args = vec!["--model".to_owned(), named.row.model.clone()];
        args.extend(Self::base_args());
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        println!("s7: bloomery-serve {}", argv.join(" "));
        let mut served = Served::spawn_cmd(command(Some(named.cache.0.as_path()))?, &argv, &dir)?;
        let (out, err) = (dir.join("server.out"), dir.join("server.err"));
        let bound = events_until(&mut served, &out, &err, "the bound event", &|events| {
            event(events, "bound").is_some()
        })?;
        let url = event(&bound, "bound")
            .and_then(|b| b["url"].as_str())
            .ok_or("the bound event names no url")?
            .to_owned();
        let mut ok = check(
            "s9_the_bound_event_says_loading",
            event(&bound, "bound").is_some_and(|b| b["phase"] == "loading"),
        );
        // The loading window: every answer before the first 200, with the `ready` event's presence on each side.
        let mut probes: Vec<Probe> = Vec::new();
        let until = Instant::now() + POLL * u32::try_from(POLLS)?;
        loop {
            let ready_before = ready_printed(&out)?;
            let (status, body) = curl(&format!("{url}/health"), None, false)?;
            let ready_after = ready_printed(&out)?;
            probes.push(Probe {
                status,
                body,
                ready_before,
                ready_after,
            });
            if status == 200 {
                break;
            }
            if let Some(exit) = served.child.try_wait()? {
                return Err(format!(
                    "the server ended ({exit}) while loading; {}",
                    tail(&read_lossy(&err)?, 1500)
                )
                .into());
            }
            if Instant::now() >= until {
                return Err("the server answered /health with a 503 for the whole wait".into());
            }
            std::thread::sleep(EVENT_GAP);
        }
        let loading: Vec<&Probe> = probes.iter().filter(|p| p.status != 200).collect();
        println!(
            "s9: {} reads of /health, {} before the first 200",
            probes.len(),
            loading.len()
        );
        if let Some(first) = loading.first() {
            println!("s9: the first answer: {}", tail(&first.body, 300));
        }
        ok &= check(
            "s9_health_is_503_loading_until_the_first_200",
            !loading.is_empty() && loading.iter().all(|p| p.is_loading()),
        );
        ok &= check(
            "s9_no_200_before_the_ready_event",
            probes.iter().all(|p| p.status != 200 || p.ready_after),
        );
        ok &= check(
            "s9_no_503_after_the_ready_event",
            probes.iter().all(|p| !p.ready_before || p.status == 200),
        );

        let events = events_seen(&read_lossy(&out)?);
        let at = |name: &str| events.iter().position(|e| e["event"] == name);
        println!(
            "s7: events {:?}",
            events
                .iter()
                .map(|e| e["event"].as_str())
                .collect::<Vec<_>>()
        );
        ok &= check(
            "s7_pick_event_names_the_cached_row",
            event(&events, "pick").is_some_and(|p| {
                p["variant"] == named.row.variant.as_str()
                    && p["source"] == named.row.hf().as_str()
                    && p["cached"].as_bool() == Some(true)
                    && p["download"].as_u64() == Some(0)
            }),
        );
        ok &= check(
            "s7_pick_and_bound_come_before_ready",
            matches!(
                (at("pick"), at("bound"), at("ready")),
                (Some(pick), Some(bound), Some(ready)) if pick < ready && bound < ready
            ),
        );
        ok &= check(
            "s7_ready_event_names_the_server",
            event(&events, "ready").is_some_and(|r| {
                r["url"].as_str() == Some(url.as_str())
                    && r["model"].as_str().is_some_and(|m| !m.is_empty())
                    && r["ctx"].as_u64().is_some_and(|c| c > 0)
                    && r["slots"].as_u64().is_some_and(|n| n > 0)
            }),
        );
        let url_of = |p: &str| format!("{url}{p}");
        ok &= check("s7_health_ok", health(&url_of)?);
        let (st, body) = chat(&url_of, &text_messages())?;
        println!(
            "s7: text chat HTTP {st}, prompt_tokens {:?}",
            prompt_tokens(&body)
        );
        ok &= check(
            "s7_text_chat_200",
            st == 200 && prompt_tokens(&body).is_some_and(|n| n > 0),
        );
        // The orderly stop: SIGTERM to the pid this gate spawned.
        terminate(served.child.id())?;
        let status = ended(&mut served.child, END_WAIT)?;
        let all = events_in(&read_lossy(&out)?);
        if let Err(e) = &all {
            println!("s7: {e}");
        }
        ok &= check("s7_stdout_holds_events_only", all.is_ok());
        let stopped = all.as_ref().ok().and_then(|e| event(e, "stopped"));
        println!("s7: ended {status:?}; stopped event {stopped:?}");
        ok &= check(
            "s7_a_sigterm_ends_it_orderly_with_a_stopped_event",
            status.is_some_and(|s| s.code() == Some(0))
                && stopped.is_some_and(|s| {
                    s["cause"] == "signal"
                        && s["exit"].as_u64() == Some(u64::from(Code::Ok.status()))
                }),
        );
        Ok(ok)
    }

    /// `args` with `--plan --json` run to its end: the plan the server printed, one JSON line that names
    /// [`PLAN_SCHEMA`], and the events before it (a start by name prints its `pick`; every other stdout line is
    /// refused by name). Its stderr is kept in `<dir>/<tag>/plan.err`.
    fn plan_json(
        &self,
        tag: &str,
        args: &[String],
        cache: Option<&Path>,
    ) -> Result<(Value, Vec<Value>), GateError> {
        let dir = self.dir.join(tag);
        std::fs::create_dir_all(&dir)?;
        let out = command(cache)?
            .args(args)
            .args(["--plan", "--json"])
            .stdin(Stdio::null())
            .output()?;
        let text = String::from_utf8_lossy(&out.stderr).into_owned();
        std::fs::write(dir.join("plan.err"), &text)?;
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        println!(
            "{tag}: --plan --json rc {:?}; {}",
            out.status.code(),
            tail(stdout.trim_end(), 300)
        );
        if out.status.code() != Some(0) {
            return Err(format!(
                "{tag}: --plan --json ended {}; {}",
                out.status,
                tail(&text, 1500)
            )
            .into());
        }
        plan_and_events(&format!("{tag}: --plan --json"), &stdout)
    }

    /// (s8): the dry run by name prints the pick, and the argument list of the start it would make, whose plan is the
    /// plan of the fixture file.
    fn dry_run(&self) -> Result<bool, GateError> {
        let named = self.named()?;
        let cache = Some(named.cache.0.as_path());
        let (file_plan, _) = self.plan_json("s8_file", &self.file_args(), None)?;
        let dir = self.dir.join("s8");
        std::fs::create_dir_all(&dir)?;
        let mut args = vec!["--model".to_owned(), named.row.model.clone()];
        args.extend(Self::base_args());
        let out = command(cache)?
            .args(args)
            .args(["--dry-run", "--json"])
            .stdin(Stdio::null())
            .output()?;
        std::fs::write(dir.join("dry.err"), &out.stderr)?;
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        println!(
            "s8: --model {} --dry-run --json rc {:?}; {}",
            named.row.model,
            out.status.code(),
            tail(stdout.trim_end(), 600)
        );
        let mut ok = check("s8_dry_run_exits_0", out.status.code() == Some(0));
        let doc = one_json_line("s8: --dry-run --json", &stdout)?;
        let entry = doc["models"].as_array().and_then(|m| match m.as_slice() {
            [e] => Some(e),
            _ => None,
        });
        ok &= check(
            "s8_one_model_entry_the_one_named",
            entry.is_some_and(|e| e["name"] == named.row.model.as_str()),
        );
        let pick = entry.map(|e| &e["pick"]);
        ok &= check(
            "s8_pick_names_the_cached_row",
            pick.is_some_and(|p| {
                p["variant"] == named.row.variant.as_str()
                    && p["source"] == named.row.hf().as_str()
                    && p["cached"].as_bool() == Some(true)
            }),
        );
        let start: Vec<String> = doc["start"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        // PIN(2026-10-29): the start replays the pick by name (`--model <name>:<variant>`), not by the repo it resolves to
        // (`--hf`): the name carries the row's pinned commit, which a repo spelling would read at `main`.
        let replay = [
            "--model".to_owned(),
            format!("{}:{}", named.row.model, named.row.variant),
        ]
        .into_iter()
        .chain(Self::base_args())
        .collect::<Vec<String>>();
        ok &= check(
            "s8_start_replays_the_pick_by_name_at_its_pinned_commit",
            start == replay,
        );
        if start.is_empty() {
            return Ok(false);
        }
        let (start_plan, replayed) = self.plan_json("s8_start", &start, cache)?;
        ok &= check(
            "s8_the_start_prints_the_pick_it_replays",
            matches!(
                (pick, event(&replayed, "pick")),
                (Some(dry), Some(made))
                    if same(dry, made, &["variant", "source", "revision", "place", "ctx", "slots", "paged"])
            ),
        );
        ok &= check(
            "s8_pick_equals_the_plan_of_its_start",
            pick.is_some_and(|p| same(p, &start_plan, &["place", "ctx", "slots", "paged"])),
        );
        ok &= check(
            "s8_the_start_plans_as_the_fixture_file_does",
            same(
                &file_plan,
                &start_plan,
                &["place", "ctx", "slots", "paged", "limit"],
            ) && same(&file_plan["need"], &start_plan["need"], &["bytes"]),
        );
        Ok(ok)
    }

    /// A start with `args` and `env` set on top of the inherited environment, run to its end: its exit status, the
    /// events it printed and its stderr file, in `<dir>/<tag>`.
    fn refusal(
        &self,
        tag: &str,
        env: &[(&str, &str)],
        args: &[String],
        cache: Option<&Path>,
    ) -> Result<Refusal, GateError> {
        let dir = self.dir.join(tag);
        std::fs::create_dir_all(&dir)?;
        let mut cmd = command(cache)?;
        for (key, value) in env {
            cmd.env(key, value);
        }
        let argv: Vec<&str> = args.iter().map(String::as_str).collect();
        println!("{tag}: bloomery-serve {}", argv.join(" "));
        let mut served = Served::spawn_cmd(cmd, &argv, &dir)?;
        let status = ended(&mut served.child, END_WAIT)?;
        let events = events_seen(&read_lossy(&dir.join("server.out"))?);
        let err = dir.join("server.err");
        println!("{tag}: ended {status:?}; {}", tail(&read_lossy(&err)?, 300));
        Ok(Refusal {
            status: status.and_then(|s| s.code()),
            events,
            err,
        })
    }

    /// The base arguments after `head`.
    fn args_of(head: &[&str]) -> Vec<String> {
        head.iter()
            .map(|a| (*a).to_owned())
            .chain(Self::base_args())
            .collect()
    }

    /// (s10): the exit table, on the refusals a start meets.
    fn exit_table(&self) -> Result<bool, GateError> {
        let named = self.named().map_err(|e| e.to_string());
        let model = self.model.display().to_string();
        let mut ok = true;
        // A name the catalog does not hold: the message lists the names it has.
        let r = self.refusal(
            "s10_unknown",
            &[],
            &Self::args_of(&["--model", NO_SUCH_MODEL]),
            None,
        )?;
        ok &= check(
            "s10_an_unknown_name_is_64_usage",
            r.is(Code::Usage)
                && r.says(NO_SUCH_MODEL)
                && named.as_ref().is_ok_and(|n| r.says(&n.row.model)),
        );
        // A path where a name goes: the `next` names the flag that takes a path.
        let r = self.refusal("s10_path", &[], &Self::args_of(&["--model", &model]), None)?;
        ok &= check(
            "s10_a_model_value_that_is_a_path_is_64_and_next_names_dash_m",
            r.is(Code::Usage) && r.error().is_some_and(|e| next_has(e, &["-m", &model])),
        );
        // A file no chat seat serves: an embedding model.
        let Some(embedding) = side_seat::ROWS.first() else {
            return Err("this build serves no embedding architecture".into());
        };
        let dir = self.dir.join("s10_embedding");
        std::fs::create_dir_all(&dir)?;
        let file = dir.join("embedding.gguf");
        gguf_of_arch(&file, embedding.arch)?;
        let r = self.refusal(
            "s10_embedding",
            &[],
            &Self::args_of(&["-m", &file.display().to_string()]),
            None,
        )?;
        ok &= check(
            "s10_an_embedding_file_is_65_unsupported_and_names_embd_model",
            r.is(Code::Unsupported) && r.says("--embd-model"),
        );
        // A card budget below the dense trunk: no fit, and every smaller variant of the model is a `next`.
        match &named {
            Ok(n) => {
                let target = format!("{}:{}", n.row.model, n.row.variant);
                let r = self.refusal(
                    "s10_no_fit",
                    &[(CARD_BUDGET, BUDGET_BELOW_FLOOR)],
                    &Self::args_of(&["--model", &target]),
                    Some(n.cache.0.as_path()),
                )?;
                ok &= check(
                    "s10_a_card_budget_below_the_dense_trunk_is_80_no_fit",
                    r.is(Code::NoFit),
                );
                println!(
                    "s10: {} variants of {} are smaller than {}; the next smaller is {:?}",
                    n.row.smaller.len(),
                    n.row.model,
                    n.row.variant,
                    n.row.step
                );
                // PIN(2026-10-29): `next` offers the next smaller variant (the design's exit table), not every smaller
                // one; the check holds both sides, since the default rows of both families have nothing smaller and a
                // check of "every smaller" would pass on an empty `next`.
                ok &= check(
                    "s10_no_fit_next_offers_the_next_smaller_variant_and_no_other",
                    r.error().is_some_and(|e| {
                        let offered =
                            e["next"]
                                .as_array()
                                .into_iter()
                                .flatten()
                                .filter_map(|list| {
                                    let list = list.as_array()?;
                                    match (list.first()?.as_str()?, list.get(1)?.as_str()?) {
                                        ("--model", value) => Some(value.to_owned()),
                                        _ => None,
                                    }
                                });
                        let want: Vec<String> = n
                            .row
                            .step
                            .iter()
                            .map(|v| format!("{}:{v}", n.row.model))
                            .collect();
                        offered.collect::<Vec<String>>() == want
                    }),
                );
            }
            Err(e) => {
                println!("s10: no catalog row to start from: {e}");
                ok &= check(
                    "s10_a_card_budget_below_the_dense_trunk_is_80_no_fit",
                    false,
                );
            }
        }
        // A `BLOOMERY_*` value no lever takes: the message names the lever.
        let r = self.refusal(
            "s10_lever",
            &[(CARD_BUDGET, BUDGET_NOT_BYTES)],
            &self.file_args(),
            None,
        )?;
        ok &= check(
            "s10_a_bad_lever_value_is_78_lever_and_names_the_lever",
            r.is(Code::Lever) && r.says(CARD_BUDGET),
        );
        // A port another process holds is found before the model is loaded.
        let held = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = held.local_addr()?.port().to_string();
        let args = ["-m", &model, "--host", "127.0.0.1", "--port", &port].map(str::to_owned);
        let r = self.refusal("s10_port", &[], &args, None)?;
        drop(held);
        ok &= check("s10_a_held_port_is_83_port_in_use", r.is(Code::PortInUse));
        ok &= check(
            "s10_port_in_use_next_names_port_0",
            r.error().is_some_and(|e| next_has(e, &["--port", "0"])),
        );
        ok &= check(
            "s10_a_held_port_is_found_before_the_load",
            server_log(&r.err, self.family.kinds())?
                .first(&record::LOAD_PHASES)?
                .is_none(),
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
    gate.hash_in_background();
    tier::sc("s1_plain_start")?;
    let mut ok = guarded("s1", gate.plain());
    tier::sc("s2_mmproj_start")?;
    ok &= guarded("s2", gate.mmproj());
    tier::sc("s3_hf_offline_start")?;
    ok &= guarded("s3", gate.hf());
    tier::sc("s5_template_file_start")?;
    ok &= guarded("s5", gate.template_file());
    tier::sc("s6_need_hook_start")?;
    ok &= guarded("s6", gate.need_hook());
    tier::sc("s7_start_by_name")?;
    ok &= guarded("s7", gate.by_name());
    tier::sc("s8_dry_run_by_name")?;
    ok &= guarded("s8", gate.dry_run());
    tier::sc("s10_exit_table")?;
    ok &= guarded("s10", gate.exit_table());
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
