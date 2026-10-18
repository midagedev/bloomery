//! Encoded video in the server: ffmpeg and ffprobe, run as subprocesses the way llama-server does.
//!
//! A clip arrives as base64 inside a request ([`crate::media::Part::Video`]). It is written to a
//! temp file (an mp4 with its index at the end needs to seek, which a pipe cannot), probed once
//! with ffprobe ([`Decoder::probe`]), planned by the one planner (`vision::video::plan`), and
//! decoded by ffmpeg at its native size: only the planned frames, streamed one at a time, each
//! handed on as soon as it arrives so the request thread resizes and patches a group while ffmpeg
//! decodes toward the next ([`video_part`]). Memory is one group of frames and the patches.
//!
//! ffmpeg and ffprobe come from `PATH`, found once at startup ([`FfmpegDecoder::detect`]); an absent
//! or too old one is a named error that names both remedies, never a fallback. The decoder is a
//! trait ([`Decoder`]) so the server's tests run a mock that makes frames from their indices.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use serde_json::Value;
use sha2::{Digest, Sha256};
use vision::media::{MediaModel, Piece};
use vision::video::{ClipMeta, ClipPlan};
use vision::{Prepared, Rgb8, VisionError};

use crate::media::{ImageKey, MediaError};

/// How much of ffmpeg's stderr an error quotes.
const STDERR_KEPT: u64 = 8 * 1024;

/// The oldest ffmpeg the decode takes: `-fps_mode` is from 5.1.
pub const MIN_FFMPEG: (u32, u32) = (5, 1);

/// Positions a chat prompt spends beyond its content (the template's turn markers, the first
/// generated token) that a clip's budget leaves free.
pub const PROMPT_FRAME_TOKENS: usize = 64;

/// Everything the video part refuses, each naming what it refused and what to do.
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    /// No usable ffmpeg and ffprobe on `PATH`.
    #[error(
        "video input needs ffmpeg and ffprobe on PATH ({0}); install ffmpeg, or send the clip as \
         decoded one-FPS image_url frames after a Video: text part"
    )]
    Absent(String),
    /// An ffmpeg without `-fps_mode`.
    #[error(
        "video input needs ffmpeg {major}.{minor} or newer; the one on PATH is {found}; upgrade ffmpeg",
        major = MIN_FFMPEG.0,
        minor = MIN_FFMPEG.1
    )]
    Old { found: String },
    /// The temp file the clip is written to.
    #[error("the clip's temp file: {0}")]
    Temp(String),
    /// ffprobe failed or said something this server does not read.
    #[error("ffprobe: {0}")]
    Probe(String),
    /// ffmpeg failed.
    #[error("ffmpeg: {0}")]
    Decode(String),
    /// ffmpeg ended with fewer frames than the plan asked for.
    #[error(
        "ffmpeg returned {got} of the {wanted} planned frames: the file ends early or a frame did \
         not decode"
    )]
    Short { wanted: usize, got: usize },
}

/// A clip of a request: its bytes, their sha256, the container's name for the temp file's suffix,
/// and the temp file ffprobe and ffmpeg read, written once and removed when the clip drops.
pub struct Clip {
    bytes: Vec<u8>,
    key: [u8; 32],
    suffix: String,
    file: OnceLock<Result<TempFile, String>>,
}

struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        // A file that is already gone is the state wanted.
        let _ = std::fs::remove_file(&self.0);
    }
}

impl Clip {
    /// A clip of `bytes`; `format` names the container (`mp4`, `mkv`), kept as the temp file's
    /// suffix when it is a short alphanumeric word, otherwise `bin`.
    #[must_use]
    pub fn new(bytes: Vec<u8>, format: Option<&str>) -> Clip {
        let suffix = format
            .filter(|f| (1..=8).contains(&f.len()) && f.bytes().all(|b| b.is_ascii_alphanumeric()))
            .unwrap_or("bin")
            .to_ascii_lowercase();
        Clip {
            key: Sha256::digest(&bytes).into(),
            bytes,
            suffix,
            file: OnceLock::new(),
        }
    }

    /// The sha256 of the clip's bytes.
    #[must_use]
    pub fn key(&self) -> [u8; 32] {
        self.key
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The temp file holding the clip, written on the first call.
    pub fn path(&self) -> Result<&Path, VideoError> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        self.file
            .get_or_init(|| {
                let name = format!(
                    "bloomery-serve-clip-{}-{}.{}",
                    std::process::id(),
                    COUNTER.fetch_add(1, Ordering::Relaxed),
                    self.suffix
                );
                let path = std::env::temp_dir().join(name);
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let guard = TempFile(path);
                std::io::Write::write_all(&mut file, &self.bytes)
                    .map_err(|e| format!("{}: {e}", guard.0.display()))?;
                Ok(guard)
            })
            .as_ref()
            .map(|f| f.0.as_path())
            .map_err(|e| VideoError::Temp(e.clone()))
    }
}

/// What reads a clip's frames: ffmpeg and ffprobe, or a test's mock.
pub trait Decoder: Send + Sync {
    /// The clip's frame count, rate and size.
    fn probe(&self, clip: &Clip) -> Result<ClipMeta, VideoError>;

    /// Decode the source frames `indices` (ascending, each once) at their native size, handing each
    /// to `each` with its index as it arrives; `each` returns `false` to stop. `threads` is the
    /// decoder's thread count, 0 for its own choice. Fewer frames than asked, without a stop, is
    /// [`VideoError::Short`].
    fn decode(
        &self,
        clip: &Clip,
        meta: &ClipMeta,
        indices: &[usize],
        threads: usize,
        each: &mut dyn FnMut(usize, Rgb8) -> bool,
    ) -> Result<(), VideoError>;
}

/// The decoder's thread count when the seat names none: ffmpeg's own choice takes every core, which
/// the engine's spinning workers need.
pub const DEFAULT_THREADS: usize = 4;

/// What the server found at startup: a decoder, or why there is none, and the threads a decode may
/// use.
#[derive(Clone)]
pub struct VideoSupport {
    decoder: Result<Arc<dyn Decoder>, String>,
    threads: usize,
}

impl Default for VideoSupport {
    /// No decoder: a server nothing told about video.
    fn default() -> VideoSupport {
        VideoSupport::absent("no decoder was set")
    }
}

impl VideoSupport {
    /// ffmpeg and ffprobe from `PATH`, or the reason they are not usable, with the startup line that
    /// says which.
    #[must_use]
    pub fn detect() -> (VideoSupport, String) {
        match FfmpegDecoder::detect() {
            Ok(ffmpeg) => {
                let line = format!("video: ffmpeg {}", ffmpeg.version);
                (VideoSupport::with(Arc::new(ffmpeg)), line)
            }
            Err(e) => {
                let line = format!("video: off ({e})");
                let reason = match e {
                    VideoError::Absent(reason) => reason,
                    other => other.to_string(),
                };
                (VideoSupport::absent(reason), line)
            }
        }
    }

    /// A decoder: ffmpeg, a bundled build, or a test's mock.
    #[must_use]
    pub fn with(decoder: Arc<dyn Decoder>) -> VideoSupport {
        VideoSupport {
            decoder: Ok(decoder),
            threads: std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
                .min(DEFAULT_THREADS),
        }
    }

    /// No decoder, for `reason`.
    #[must_use]
    pub fn absent(reason: impl Into<String>) -> VideoSupport {
        VideoSupport {
            decoder: Err(reason.into()),
            threads: 1,
        }
    }

    /// The decoder's thread count: the cores the engine's pool leaves, at least one.
    #[must_use]
    pub fn threads(mut self, threads: usize) -> VideoSupport {
        self.threads = threads.max(1);
        self
    }

    #[must_use]
    pub fn thread_count(&self) -> usize {
        self.threads
    }

    /// The decoder, or the named error for its absence.
    pub fn decoder(&self) -> Result<Arc<dyn Decoder>, VideoError> {
        self.decoder.clone().map_err(VideoError::Absent)
    }

    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.decoder.is_ok()
    }
}

/// ffmpeg and ffprobe as subprocesses.
pub struct FfmpegDecoder {
    ffmpeg: PathBuf,
    ffprobe: PathBuf,
    /// The version text ffmpeg printed, for the startup line.
    pub version: String,
}

impl FfmpegDecoder {
    /// Find ffmpeg and ffprobe on `PATH` and check ffmpeg's version against [`MIN_FFMPEG`]. A build
    /// whose version text is not a release number (a git build) is taken: its decode would say if
    /// it lacked the option.
    pub fn detect() -> Result<FfmpegDecoder, VideoError> {
        let first_line = |name: &str| -> Result<String, VideoError> {
            let out = Command::new(name)
                .arg("-version")
                .stdin(Stdio::null())
                .output()
                .map_err(|e| VideoError::Absent(format!("{name}: {e}")))?;
            if !out.status.success() {
                return Err(VideoError::Absent(format!(
                    "{name} -version: {}",
                    out.status
                )));
            }
            Ok(String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .to_owned())
        };
        let line = first_line("ffmpeg")?;
        first_line("ffprobe")?;
        let version = line
            .strip_prefix("ffmpeg version ")
            .and_then(|rest| rest.split_whitespace().next())
            .ok_or_else(|| VideoError::Absent(format!("ffmpeg -version said {line:?}")))?
            .to_owned();
        check_version(&version)?;
        Ok(FfmpegDecoder {
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
            version,
        })
    }
}

/// Refuse an ffmpeg older than [`MIN_FFMPEG`]. A version text that is not a release number (a git
/// build) is taken: its decode would say if it lacked the option.
fn check_version(version: &str) -> Result<(), VideoError> {
    match release(version) {
        Some(found) if found < MIN_FFMPEG => Err(VideoError::Old {
            found: version.to_owned(),
        }),
        _ => Ok(()),
    }
}

/// `(major, minor)` of a release version text (`6.1.1-3ubuntu5`, `n7.0`), `None` for any other.
fn release(version: &str) -> Option<(u32, u32)> {
    let text = version.strip_prefix('n').unwrap_or(version);
    let mut parts = text.split(|c: char| !c.is_ascii_digit());
    let major = parts.next()?.parse().ok()?;
    let rest = text.get(major_len(text)..)?.strip_prefix('.')?;
    let minor = rest
        .split(|c: char| !c.is_ascii_digit())
        .next()?
        .parse()
        .ok()?;
    Some((major, minor))
}

fn major_len(text: &str) -> usize {
    text.bytes().take_while(u8::is_ascii_digit).count()
}

/// The ffmpeg `select` filter that keeps the source frames `indices`.
fn select_filter(indices: &[usize]) -> String {
    let terms: Vec<String> = indices.iter().map(|i| format!("eq(n,{i})")).collect();
    format!("select='{}'", terms.join("+"))
}

/// A frame rate text `num/den` of ffprobe, `None` for `0/0` or any other unreadable one.
fn ratio(text: &str) -> Option<(u32, u32)> {
    let (num, den) = text.split_once('/')?;
    let (num, den) = (
        num.trim().parse::<u32>().ok()?,
        den.trim().parse::<u32>().ok()?,
    );
    (num > 0 && den > 0).then_some((num, den))
}

/// The probe's JSON as a [`ClipMeta`]: the first video stream's size and average rate; the frame
/// count its `nb_frames`, else its duration (the stream's, else the container's) times the rate. A
/// clip with neither is refused by name rather than guessed. A stream rotated by a quarter turn
/// decodes at the swapped size (ffmpeg applies the rotation), so the size is swapped here.
fn parse_probe(json: &str) -> Result<ClipMeta, VideoError> {
    let bad = |detail: &str| VideoError::Probe(detail.to_owned());
    let v: Value =
        serde_json::from_str(json).map_err(|e| VideoError::Probe(format!("not JSON: {e}")))?;
    let stream = v
        .get("streams")
        .and_then(Value::as_array)
        .and_then(|s| s.first())
        .ok_or_else(|| bad("the file has no video stream"))?;
    let number = |key: &str| -> Option<f64> {
        match stream.get(key)? {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    };
    let size = |key: &str| {
        number(key)
            .filter(|n| *n >= 1.0 && n.fract() == 0.0)
            .map(|n| n as usize)
            .ok_or_else(|| VideoError::Probe(format!("the stream has no {key}")))
    };
    let (mut width, mut height) = (size("width")?, size("height")?);
    let rotation = stream
        .get("side_data_list")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find_map(|d| d.get("rotation").and_then(Value::as_f64))
        .or_else(|| {
            stream
                .get("tags")
                .and_then(|t| t.get("rotate"))
                .and_then(Value::as_str)
                .and_then(|s| s.parse().ok())
        });
    if rotation.is_some_and(|r| (r.abs() % 180.0 - 90.0).abs() < 1.0) {
        std::mem::swap(&mut width, &mut height);
    }
    let duration = number("duration").or_else(|| {
        v.get("format")
            .and_then(|f| f.get("duration"))
            .and_then(|d| d.as_str()?.parse().ok())
    });
    let nb_frames = number("nb_frames")
        .filter(|n| *n >= 1.0)
        .map(|n| n as usize);
    let fps = stream
        .get("avg_frame_rate")
        .and_then(Value::as_str)
        .and_then(ratio)
        .or_else(|| {
            let (frames, seconds) = (nb_frames?, duration.filter(|d| *d > 0.0)?);
            let millis = (seconds * 1000.0).round() as u64;
            let num = u32::try_from(frames as u64 * 1000).ok()?;
            Some((num, u32::try_from(millis).ok()?)).filter(|&(n, d)| n > 0 && d > 0)
        })
        .ok_or_else(|| bad("the stream has no frame rate and no frame count over a duration"))?;
    let frames = match nb_frames {
        Some(n) => n,
        None => {
            let seconds = duration.filter(|d| *d > 0.0).ok_or_else(|| {
                bad("the stream has neither a frame count nor a duration; the clip cannot be sampled")
            })?;
            let n = (seconds * f64::from(fps.0) / f64::from(fps.1)).round();
            if n < 1.0 {
                return Err(bad("the clip's duration holds no frame"));
            }
            n as usize
        }
    };
    Ok(ClipMeta {
        frames,
        fps,
        width,
        height,
    })
}

/// The first lines of what a process wrote to stderr, for an error.
fn first_lines(text: &[u8]) -> String {
    let text = String::from_utf8_lossy(text);
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .take(3)
        .collect();
    if lines.is_empty() {
        "no message".into()
    } else {
        lines.join("; ")
    }
}

impl Decoder for FfmpegDecoder {
    fn probe(&self, clip: &Clip) -> Result<ClipMeta, VideoError> {
        let out = Command::new(&self.ffprobe)
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
            .arg(
                "stream=width,height,avg_frame_rate,nb_frames,duration:stream_side_data=rotation:\
                 stream_tags=rotate:format=duration",
            )
            .args(["-of", "json"])
            .arg(clip.path()?)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| VideoError::Probe(e.to_string()))?;
        if !out.status.success() {
            return Err(VideoError::Probe(format!(
                "{}: {}",
                out.status,
                first_lines(&out.stderr)
            )));
        }
        parse_probe(&String::from_utf8_lossy(&out.stdout))
    }

    fn decode(
        &self,
        clip: &Clip,
        meta: &ClipMeta,
        indices: &[usize],
        threads: usize,
        each: &mut dyn FnMut(usize, Rgb8) -> bool,
    ) -> Result<(), VideoError> {
        let mut cmd = Command::new(&self.ffmpeg);
        cmd.args(["-nostdin", "-v", "error"]);
        if threads > 0 {
            cmd.arg("-threads").arg(threads.to_string());
        }
        cmd.arg("-i")
            .arg(clip.path()?)
            .arg("-vf")
            .arg(select_filter(indices))
            .args(["-fps_mode", "passthrough", "-frames:v"])
            .arg(indices.len().to_string())
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped());
        // ffmpeg's messages go to a file beside the clip's: a pipe nobody reads while stdout is
        // being read would stall the decoder once it held more than a pipe does.
        let log = TempFile(clip.path()?.with_extension("log"));
        let file = std::fs::File::create(&log.0)
            .map_err(|e| VideoError::Temp(format!("{}: {e}", log.0.display())))?;
        cmd.stderr(Stdio::from(file));
        let mut child = cmd.spawn().map_err(|e| VideoError::Decode(e.to_string()))?;
        let got = read_frames(&mut child, meta, indices, each);
        // The stream ended or the reader stopped: the process is done with, whatever it was doing.
        let _ = child.kill();
        let waited = child.wait();
        let mut stderr = Vec::new();
        if let Ok(f) = std::fs::File::open(&log.0) {
            let _ = f.take(STDERR_KEPT).read_to_end(&mut stderr);
        }
        match got {
            Ok(Done::Stopped) => Ok(()),
            Ok(Done::All) => Ok(()),
            Err(ReadError::Short(got)) => {
                let said = first_lines(&stderr);
                if said == "no message" {
                    Err(VideoError::Short {
                        wanted: indices.len(),
                        got,
                    })
                } else {
                    Err(VideoError::Decode(format!(
                        "{said} (after {got} of {} planned frames; {})",
                        indices.len(),
                        waited.map_or_else(|e| e.to_string(), |s| s.to_string())
                    )))
                }
            }
            Err(ReadError::Size { got, frame }) => Err(VideoError::Decode(format!(
                "the output ended inside a {}x{} frame after {got} whole frames ({frame} bytes a \
                 frame); the clip's rotation or size is not what ffprobe said",
                meta.width, meta.height
            ))),
        }
    }
}

enum Done {
    All,
    Stopped,
}

enum ReadError {
    /// The stream ended with this many whole frames.
    Short(usize),
    /// The stream ended inside a frame of `frame` bytes.
    Size { got: usize, frame: usize },
}

/// Read the child's raw rgb24 frames, one per index, handing each on.
fn read_frames(
    child: &mut Child,
    meta: &ClipMeta,
    indices: &[usize],
    each: &mut dyn FnMut(usize, Rgb8) -> bool,
) -> Result<Done, ReadError> {
    let Some(stdout) = child.stdout.as_mut() else {
        return Err(ReadError::Short(0));
    };
    frames_from(stdout, meta, indices, each)
}

/// [`read_frames`] over any reader: the frame loop, testable without a process.
fn frames_from(
    reader: &mut dyn Read,
    meta: &ClipMeta,
    indices: &[usize],
    each: &mut dyn FnMut(usize, Rgb8) -> bool,
) -> Result<Done, ReadError> {
    let frame = meta.width * meta.height * 3;
    for (k, &index) in indices.iter().enumerate() {
        let mut data = vec![0u8; frame];
        let mut at = 0;
        while at < frame {
            match reader.read(&mut data[at..]) {
                Ok(0) => {
                    return Err(if at == 0 {
                        ReadError::Short(k)
                    } else {
                        ReadError::Size { got: k, frame }
                    });
                }
                Ok(n) => at += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => return Err(ReadError::Short(k)),
            }
        }
        let image = Rgb8 {
            width: meta.width,
            height: meta.height,
            data,
        };
        if !each(index, image) {
            return Ok(Done::Stopped);
        }
    }
    Ok(Done::All)
}

/// One group of a clip, decoded and prepared: the feed the engine takes, with the key that tells it
/// from any other medium of the same span length.
#[derive(Clone, Debug)]
pub struct ReadyGroup {
    pub key: ImageKey,
    pub prepared: Arc<Prepared>,
}

/// A clip a request carries, probed and planned: the text it becomes in the prompt, and what the
/// prompt's expansion needs to decode and prepare its groups. Nothing is decoded until then, so a
/// request that only renders the template ([`PlannedClip::pieces`]) costs a probe.
pub struct PlannedClip {
    clip: Clip,
    meta: ClipMeta,
    plan: ClipPlan,
    pieces: Vec<Piece>,
    decoder: Arc<dyn Decoder>,
    threads: usize,
}

impl std::fmt::Debug for PlannedClip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlannedClip")
            .field("meta", &self.meta)
            .field("frames", &self.plan.indices.len())
            .field("groups", &self.plan.groups.len())
            .finish_non_exhaustive()
    }
}

/// The key of group `g` of a plan: the clip, the group's place, size and source frames. The same
/// clip planned the same way is the same medium, so a prefix cache keeps its spans; any other plan
/// of it is another.
fn group_key(clip: &Clip, plan: &ClipPlan, g: usize) -> ImageKey {
    let group = &plan.groups[g];
    let mut h = Sha256::new();
    h.update(b"video-group\0");
    h.update(clip.key());
    for n in [g, group.size.0, group.size.1] {
        h.update((n as u64).to_le_bytes());
    }
    for &i in &plan.indices[group.frames.clone()] {
        h.update((i as u64).to_le_bytes());
    }
    ImageKey(h.finalize().into())
}

/// Probe `clip` and plan it under `budget` positions by the model's video rule: the one planner
/// every model's clip goes through.
pub fn plan_clip(
    model: &dyn MediaModel,
    decoder: Arc<dyn Decoder>,
    clip: Clip,
    budget: usize,
    threads: usize,
) -> Result<PlannedClip, MediaError> {
    let rule = model.video_rule().ok_or(MediaError::NoVideoRule)?;
    let meta = decoder.probe(&clip)?;
    let plan =
        vision::video::plan(rule, &meta, budget, model.video_sizer()).map_err(MediaError::Video)?;
    let pieces = model.video_text(&plan).map_err(MediaError::Video)?;
    Ok(PlannedClip {
        clip,
        meta,
        plan,
        pieces,
        decoder,
        threads,
    })
}

impl PlannedClip {
    /// The clip's text in prompt order: the model's delimiters and stamps, and each group's
    /// placeholder.
    #[must_use]
    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// The groups the clip makes, one placeholder each in [`PlannedClip::pieces`].
    #[must_use]
    pub fn groups(&self) -> usize {
        self.plan.groups.len()
    }

    /// The prompt positions the plan spends.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.plan.tokens()
    }

    /// The source frames the plan takes.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.plan.indices.len()
    }

    /// Decode the planned frames and prepare each group as its frames arrive, so the request thread
    /// resizes and patches one group while the decoder works toward the next. Memory is one group's
    /// frames and the prepared patches.
    pub fn prepare(&self, model: &dyn MediaModel) -> Result<Vec<ReadyGroup>, MediaError> {
        let rule = model.video_rule().ok_or(MediaError::NoVideoRule)?;
        let plan = &self.plan;
        let mut groups: Vec<ReadyGroup> = Vec::with_capacity(plan.groups.len());
        let mut held: Vec<Rgb8> = Vec::with_capacity(rule.temporal);
        let mut next = 0;
        let mut failed: Option<MediaError> = None;
        let mut deliver = |index: usize, frame: Rgb8| -> bool {
            let uses = plan.indices[next..]
                .iter()
                .take_while(|&&i| i == index)
                .count();
            let mut frame = Some(frame);
            for use_ in 0..uses {
                let one = if use_ + 1 == uses {
                    frame.take().expect("the last use takes the frame")
                } else {
                    frame.clone().expect("an earlier use leaves the frame")
                };
                held.push(one);
                next += 1;
                if held.len() < rule.temporal {
                    continue;
                }
                let g = groups.len();
                let group = &plan.groups[g];
                match model.prepare_group(&held, group.size) {
                    Ok(prepared) if prepared.span_len == group.tokens => groups.push(ReadyGroup {
                        key: group_key(&self.clip, plan, g),
                        prepared: Arc::new(prepared),
                    }),
                    Ok(prepared) => {
                        failed = Some(MediaError::Video(VisionError::Video(format!(
                            "group {g} prepared {} positions where the plan said {}",
                            prepared.span_len, group.tokens
                        ))));
                        return false;
                    }
                    Err(e) => {
                        failed = Some(MediaError::Video(VisionError::Video(format!(
                            "group {g}: {e}"
                        ))));
                        return false;
                    }
                }
                held.clear();
            }
            true
        };
        self.decoder.decode(
            &self.clip,
            &self.meta,
            &plan.decode_indices(),
            self.threads,
            &mut deliver,
        )?;
        if let Some(e) = failed {
            return Err(e);
        }
        if groups.len() != plan.groups.len() {
            return Err(MediaError::Video(VisionError::Video(format!(
                "{} of {} groups were prepared",
                groups.len(),
                plan.groups.len()
            ))));
        }
        Ok(groups)
    }
}

/// The positions of context a clip may take: the context less the prompt's own text and
/// [`PROMPT_FRAME_TOKENS`] of template.
#[must_use]
pub fn clip_budget(ctx: usize, text_tokens: usize) -> usize {
    ctx.saturating_sub(text_tokens.saturating_add(PROMPT_FRAME_TOKENS))
}

/// A decoder of data only, for the server's tests: a clip's bytes read `<frames> <num>/<den>
/// <width>x<height>` (`b"30 10/1 4x2"` is 30 frames at 10 fps of 4×2 pixels), frame `i` is every
/// byte `i`, and each decode is recorded with its threads.
#[cfg(test)]
pub(crate) struct MockClips {
    pub(crate) calls: std::sync::Mutex<Vec<(Vec<usize>, usize)>>,
}

#[cfg(test)]
impl MockClips {
    pub(crate) fn new() -> Arc<MockClips> {
        Arc::new(MockClips {
            calls: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// The server's video support with this decoder, `threads` for its decodes.
    pub(crate) fn support(self: &Arc<Self>, threads: usize) -> VideoSupport {
        VideoSupport::with(Arc::clone(self) as Arc<dyn Decoder>).threads(threads)
    }
}

#[cfg(test)]
impl Decoder for MockClips {
    fn probe(&self, clip: &Clip) -> Result<ClipMeta, VideoError> {
        let text = std::str::from_utf8(clip.bytes())
            .map_err(|_| VideoError::Probe("the mock reads text clips".into()))?;
        let bad = || VideoError::Probe(format!("the mock cannot read {text:?}"));
        let mut words = text.split(' ');
        let frames = words.next().and_then(|w| w.parse().ok()).ok_or_else(bad)?;
        let fps = words.next().and_then(ratio).ok_or_else(bad)?;
        let (width, height) = words
            .next()
            .and_then(|w| w.split_once('x'))
            .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
            .ok_or_else(bad)?;
        Ok(ClipMeta {
            frames,
            fps,
            width,
            height,
        })
    }

    fn decode(
        &self,
        _clip: &Clip,
        meta: &ClipMeta,
        indices: &[usize],
        threads: usize,
        each: &mut dyn FnMut(usize, Rgb8) -> bool,
    ) -> Result<(), VideoError> {
        self.calls
            .lock()
            .expect("the mock's calls")
            .push((indices.to_vec(), threads));
        for &index in indices {
            let frame = Rgb8 {
                width: meta.width,
                height: meta.height,
                data: vec![index as u8; meta.width * meta.height * 3],
            };
            if !each(index, frame) {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_read_as_releases() {
        for (text, want) in [
            ("6.1.1-3ubuntu5", Some((6, 1))),
            ("8.1.1", Some((8, 1))),
            ("n7.0", Some((7, 0))),
            ("5.1", Some((5, 1))),
            ("4.4.2-0ubuntu0.22.04.1", Some((4, 4))),
            ("N-114267-gd15a2e8a1b", None),
            ("git-2024-01-01", None),
        ] {
            assert_eq!(release(text), want, "{text}");
        }
        assert!(release("4.4.2") < Some(MIN_FFMPEG));
        assert!(release("5.1") >= Some(MIN_FFMPEG));
    }

    /// An ffmpeg before 5.1 is refused by name with the version it found; a release at or past
    /// it, and a git build, are taken.
    #[test]
    fn an_old_ffmpeg_is_refused_by_name() {
        let old = check_version("4.4.2-0ubuntu0.22.04.1").unwrap_err();
        assert!(matches!(&old, VideoError::Old { found } if found.starts_with("4.4.2")));
        assert_eq!(
            check_version("4.4.2").unwrap_err().to_string(),
            "video input needs ffmpeg 5.1 or newer; the one on PATH is 4.4.2; upgrade ffmpeg"
        );
        for taken in ["5.1", "6.1.1-3ubuntu5", "8.1.1", "N-114267-gd15a2e8a1b"] {
            assert!(check_version(taken).is_ok(), "{taken}");
        }
    }

    #[test]
    fn the_select_filter_names_each_frame() {
        assert_eq!(select_filter(&[0]), "select='eq(n,0)'");
        assert_eq!(
            select_filter(&[3, 15, 76]),
            "select='eq(n,3)+eq(n,15)+eq(n,76)'"
        );
    }

    /// A probe with a frame count reads it; without one, the duration times the rate; with neither,
    /// a named refusal. The size swaps under a quarter-turn rotation, both spellings of it.
    #[test]
    fn the_probe_reads_frames_rate_and_size() {
        let probe = |s: &str| parse_probe(s);
        let ok = probe(
            r#"{"streams":[{"width":1920,"height":1080,"avg_frame_rate":"30000/1001","nb_frames":"1800","duration":"60.06"}],"format":{"duration":"60.06"}}"#,
        )
        .unwrap();
        assert_eq!(
            ok,
            ClipMeta {
                frames: 1800,
                fps: (30000, 1001),
                width: 1920,
                height: 1080
            }
        );
        // No nb_frames (webm, some mkv): the stream duration, else the container's, times the rate.
        let by_duration = probe(
            r#"{"streams":[{"width":640,"height":360,"avg_frame_rate":"25/1"}],"format":{"duration":"10.000000"}}"#,
        )
        .unwrap();
        assert_eq!((by_duration.frames, by_duration.fps), (250, (25, 1)));
        // No frame rate either: nb_frames over the duration.
        let rate = probe(
            r#"{"streams":[{"width":64,"height":48,"avg_frame_rate":"0/0","nb_frames":"50","duration":"2.000000"}]}"#,
        )
        .unwrap();
        assert_eq!(rate.fps, (50_000, 2_000));
        for (json, words) in [
            (
                r#"{"streams":[{"width":64,"height":48,"avg_frame_rate":"25/1"}]}"#,
                "neither a frame count nor a duration",
            ),
            (
                r#"{"streams":[{"width":64,"height":48,"avg_frame_rate":"0/0","duration":"2.0"}]}"#,
                "no frame rate",
            ),
            (r#"{"streams":[]}"#, "no video stream"),
            (r#"{}"#, "no video stream"),
            (
                r#"{"streams":[{"height":48,"avg_frame_rate":"25/1","nb_frames":"3"}]}"#,
                "no width",
            ),
            ("not json", "not JSON"),
        ] {
            let e = probe(json).unwrap_err().to_string();
            assert!(
                e.starts_with("ffprobe: ") && e.contains(words),
                "{json}: {e}"
            );
        }
        for json in [
            r#"{"streams":[{"width":1920,"height":1080,"avg_frame_rate":"30/1","nb_frames":"9","side_data_list":[{"side_data_type":"Display Matrix","rotation":-90}]}]}"#,
            r#"{"streams":[{"width":1920,"height":1080,"avg_frame_rate":"30/1","nb_frames":"9","tags":{"rotate":"270"}}]}"#,
        ] {
            let m = probe(json).unwrap();
            assert_eq!((m.width, m.height), (1080, 1920), "{json}");
        }
        let flat = probe(
            r#"{"streams":[{"width":1920,"height":1080,"avg_frame_rate":"30/1","nb_frames":"9","side_data_list":[{"rotation":180}]}]}"#,
        )
        .unwrap();
        assert_eq!((flat.width, flat.height), (1920, 1080));
    }

    /// The frame loop hands each whole frame on with its index, stops when told, and names a short
    /// stream (whole frames missing) and a torn one (a partial frame) apart.
    #[test]
    fn the_frame_loop_reads_whole_frames() {
        let meta = ClipMeta {
            frames: 9,
            fps: (1, 1),
            width: 2,
            height: 1,
        };
        let stream: Vec<u8> = (0..4u8).flat_map(|f| [f; 6]).collect();
        let mut seen = Vec::new();
        let r = frames_from(&mut stream.as_slice(), &meta, &[4, 6, 7, 8], &mut |i, f| {
            seen.push((i, f.data[0]));
            true
        });
        assert!(matches!(r, Ok(Done::All)));
        assert_eq!(seen, [(4, 0), (6, 1), (7, 2), (8, 3)]);
        let mut n = 0;
        let r = frames_from(&mut stream.as_slice(), &meta, &[4, 6, 7, 8], &mut |_, _| {
            n += 1;
            n < 2
        });
        assert!(matches!(r, Ok(Done::Stopped)));
        assert_eq!(n, 2);
        let r = frames_from(&mut &stream[..12], &meta, &[4, 6, 7], &mut |_, _| true);
        assert!(matches!(r, Err(ReadError::Short(2))));
        let r = frames_from(&mut &stream[..14], &meta, &[4, 6, 7], &mut |_, _| true);
        assert!(matches!(r, Err(ReadError::Size { got: 2, frame: 6 })));
    }

    /// A clip's temp file holds its bytes, suffixes by its container name when that is a short
    /// word, and is gone when the clip drops.
    #[test]
    fn a_clip_writes_one_temp_file_and_removes_it() {
        let clip = Clip::new(b"abc".to_vec(), Some("MKV"));
        let path = clip.path().unwrap().to_path_buf();
        assert_eq!(clip.path().unwrap(), path);
        assert_eq!(std::fs::read(&path).unwrap(), b"abc");
        assert_eq!(path.extension().unwrap(), "mkv");
        drop(clip);
        assert!(!path.exists());
        for odd in ["../etc", "mp4;rm", "", "waytoolongname"] {
            let clip = Clip::new(vec![1], Some(odd));
            assert_eq!(clip.path().unwrap().extension().unwrap(), "bin", "{odd:?}");
        }
        assert_eq!(
            Clip::new(vec![1], None).key(),
            Clip::new(vec![1], Some("mp4")).key()
        );
    }
}
