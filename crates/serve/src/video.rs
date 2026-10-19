//! Encoded video in the server: ffmpeg and ffprobe, run as subprocesses the way llama-server does.
//!
//! A clip arrives as base64 inside a request ([`crate::media::Part::Video`]). It is written to a
//! temp file (an mp4 with its index at the end needs to seek, which a pipe cannot), probed with
//! ffprobe ([`Decoder::probe`]; a stream whose container states no frame count is decoded once to
//! count them, on the decoder's threads and within [`PROBE_LIMIT`]), planned by the one planner
//! (`vision::video::plan`), and decoded by ffmpeg at its native size: only the planned frames,
//! streamed one at a time, each handed on as soon as it arrives so the request thread resizes and
//! patches a group while ffmpeg decodes toward the next ([`video_part`]). Memory is one group of
//! frames and the patches. Both commands read the clip's file only through the demuxers of
//! [`VIDEO_FORMATS`] and the `file` protocol, so a playlist inside the bytes cannot lead them to
//! another file.
//!
//! ffmpeg and ffprobe come from `PATH`, found once at startup ([`FfmpegDecoder::detect`]); an absent
//! or too old one is a named error that names both remedies, never a fallback. The decoder is a
//! trait ([`Decoder`]) so the server's tests run a mock that makes frames from their indices.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use serde_json::Value;
use sha2::{Digest, Sha256};
use vision::media::{MediaModel, Piece};
use vision::video::{ClipMeta, ClipPlan, VideoRule};
use vision::{Prepared, Rgb8, VisionError};

use crate::media::{ImageKey, MediaError};

/// How much of ffmpeg's stderr an error quotes.
const STDERR_KEPT: u64 = 8 * 1024;

/// The demuxers ffprobe and ffmpeg may open on a client's bytes. Left to detect the format, they
/// open a playlist (HLS), a manifest (DASH) or a list (concat) that names other files, and read
/// media the client never uploaded. None of these follows a reference: `mov` takes external tracks
/// only under `enable_drefs` and `use_absolute_path`, both off by default, and the others have no
/// option that opens a second file.
pub const VIDEO_FORMATS: &str = "mov,matroska,webm,avi,flv,mpeg,mpegts,ogg";

/// Limit `cmd`, an ffprobe or ffmpeg that reads a clip, to the demuxers of [`VIDEO_FORMATS`] and
/// the `file` protocol, so the one file it is given is the only one it opens. These are input
/// options: the command gives them before its `-i`.
fn restrict_input(cmd: &mut Command) {
    cmd.args([
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        VIDEO_FORMATS,
    ]);
}

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

/// A new file at `path` that only this user reads: made exclusively (a path that exists, a symlink
/// among them, is refused, never followed or truncated) and, on Unix, with mode 0600 whatever the
/// umask says. The clip and ffmpeg's log are made here, in a directory other users share.
fn create_private(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

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

    #[cfg(test)]
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
                let mut file =
                    create_private(&path).map_err(|e| format!("{}: {e}", path.display()))?;
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
    /// The clip's frame count, rate and size. `threads` is the decoder's thread count for the
    /// frames a probe decodes to count them, 0 for its own choice.
    fn probe(&self, clip: &Clip, threads: usize) -> Result<ClipMeta, VideoError>;

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
    /// The longest one ffprobe run may take.
    probe_limit: Duration,
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
            probe_limit: PROBE_LIMIT,
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

/// The most terms one flat sum of a `select` expression holds.
const SELECT_RUN: usize = 16;

/// The ffmpeg `select` filter that keeps the source frames `indices`. The terms are summed in a
/// balanced tree of parentheses: ffmpeg's expression parser nests one level for each term of a
/// flat sum and refuses a sum of more than 100, which a clip of hundreds of planned frames passes.
fn select_filter(indices: &[usize]) -> String {
    format!("select='{}'", select_sum(indices))
}

fn select_sum(indices: &[usize]) -> String {
    if indices.len() <= SELECT_RUN {
        let terms: Vec<String> = indices.iter().map(|i| format!("eq(n,{i})")).collect();
        return terms.join("+");
    }
    let (head, tail) = indices.split_at(indices.len() / 2);
    format!("({})+({})", select_sum(head), select_sum(tail))
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

/// What ffprobe read of a clip's first video stream. The rate and the frame count are the
/// stream's own, never the container's: a container's duration spans every stream, so an audio
/// track longer than the video would stretch a count made from it.
struct Probed {
    width: usize,
    height: usize,
    rate: Option<(u32, u32)>,
    /// The container's stated count, or the count of frames the stream decodes to.
    frames: Option<usize>,
    /// The stream's own duration in seconds.
    duration: Option<f64>,
}

/// The probe's JSON as a [`Probed`]: the first video stream's size and average rate, its frame count
/// (`nb_read_frames` when ffprobe counted by decoding, else the container's `nb_frames`) and its
/// duration. A stream rotated by a quarter turn decodes at the swapped size (ffmpeg applies the
/// rotation), so the size is swapped here.
fn read_probe(json: &str) -> Result<Probed, VideoError> {
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
    Ok(Probed {
        width,
        height,
        rate: stream
            .get("avg_frame_rate")
            .and_then(Value::as_str)
            .and_then(ratio),
        frames: number("nb_read_frames")
            .or_else(|| number("nb_frames"))
            .filter(|n| *n >= 1.0)
            .map(|n| n as usize),
        duration: number("duration").filter(|d| *d > 0.0),
    })
}

impl Probed {
    /// The clip's [`ClipMeta`]. A stream that states no rate takes its frames over its own
    /// duration; one with no count, or neither a rate nor a duration, is refused by name rather
    /// than guessed.
    fn meta(self) -> Result<ClipMeta, VideoError> {
        let bad = |detail: &str| VideoError::Probe(detail.to_owned());
        let frames = self
            .frames
            .ok_or_else(|| bad("the stream has no frame count and none could be counted"))?;
        let fps = self
            .rate
            .or_else(|| {
                let millis = (self.duration? * 1000.0).round() as u64;
                let num = u32::try_from(frames as u64 * 1000).ok()?;
                Some((num, u32::try_from(millis).ok()?)).filter(|&(n, d)| n > 0 && d > 0)
            })
            .ok_or_else(|| bad("the stream has no frame rate and no duration to take one from"))?;
        Ok(ClipMeta {
            frames,
            fps,
            width: self.width,
            height: self.height,
        })
    }
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

/// The longest an ffprobe run on a client's clip may take before it is killed and the clip refused.
/// A container that states no frame count is decoded whole to count its frames, work that grows
/// with the frames the clip holds, which the request's byte limit does not bound.
const PROBE_LIMIT: Duration = Duration::from_secs(60);

/// How often a process run under a limit is checked for its end.
const LIMIT_POLL: Duration = Duration::from_millis(10);

/// `cmd` run to its end, as [`Command::output`] runs it, but killed once it has run for `limit`:
/// its output, or `None` when it was killed. The output goes to files beside `clip`'s, as
/// ffmpeg's messages do in [`Decoder::decode`]: a pipe is read by a thread of its own, or stalls
/// the process once it holds more than a pipe does.
fn output_within(
    cmd: &mut Command,
    clip: &Path,
    limit: Duration,
) -> Result<Option<Output>, VideoError> {
    // A clip's suffix is alphanumeric, so neither name is ever the clip's own.
    let scratch = |extension: &str| {
        let path = clip.with_extension(extension);
        let file = create_private(&path)
            .map_err(|e| VideoError::Temp(format!("{}: {e}", path.display())))?;
        Ok::<_, VideoError>((file, TempFile(path)))
    };
    let (out_file, out) = scratch("probe-out")?;
    let (err_file, err) = scratch("probe-err")?;
    let mut child = cmd
        .stdout(Stdio::from(out_file))
        .stderr(Stdio::from(err_file))
        .spawn()
        .map_err(|e| VideoError::Probe(e.to_string()))?;
    let end = Instant::now() + limit;
    let ended = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < end => std::thread::sleep(LIMIT_POLL),
            Ok(None) => break None,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(VideoError::Probe(e.to_string()));
            }
        }
    };
    let Some(status) = ended else {
        let _ = child.kill();
        let _ = child.wait();
        return Ok(None);
    };
    let stdout = std::fs::read(&out.0)
        .map_err(|e| VideoError::Probe(format!("{}: {e}", out.0.display())))?;
    let mut stderr = Vec::new();
    if let Ok(f) = std::fs::File::open(&err.0) {
        let _ = f.take(STDERR_KEPT).read_to_end(&mut stderr);
    }
    Ok(Some(Output {
        status,
        stdout,
        stderr,
    }))
}

impl FfmpegDecoder {
    /// ffprobe's report of the clip's first video stream (`v:0`, the stream [`Decoder::decode`]
    /// maps). With `count` ffprobe also decodes the stream and counts its frames, on that many
    /// threads (0: its own choice). A run past [`FfmpegDecoder::probe_limit`] is killed and refused.
    fn probe_stream(&self, clip: &Clip, count: Option<usize>) -> Result<Probed, VideoError> {
        let mut cmd = Command::new(&self.ffprobe);
        restrict_input(&mut cmd);
        cmd.args(["-v", "error", "-select_streams", "v:0"]);
        if let Some(threads) = count {
            cmd.arg("-count_frames");
            if threads > 0 {
                cmd.arg("-threads").arg(threads.to_string());
            }
        }
        cmd.arg("-show_entries")
            .arg(
                "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration:\
                 stream_side_data=rotation:stream_tags=rotate",
            )
            .args(["-of", "json"])
            .arg(clip.path()?)
            .stdin(Stdio::null());
        let out = output_within(&mut cmd, clip.path()?, self.probe_limit)?.ok_or_else(|| {
            let limit = self.probe_limit;
            VideoError::Probe(if count.is_some() {
                format!(
                    "counting the stream's frames ran past {limit:?} and was stopped; send a \
                     shorter clip, or one in a container that states its frame count (mp4)"
                )
            } else {
                format!("reading the file ran past {limit:?} and was stopped")
            })
        })?;
        if !out.status.success() {
            return Err(VideoError::Probe(format!(
                "{}: {}",
                out.status,
                first_lines(&out.stderr)
            )));
        }
        read_probe(&String::from_utf8_lossy(&out.stdout))
    }
}

impl Decoder for FfmpegDecoder {
    fn probe(&self, clip: &Clip, threads: usize) -> Result<ClipMeta, VideoError> {
        let mut stream = self.probe_stream(clip, None)?;
        if stream.frames.is_none() {
            // The container states no count (Matroska, WebM). Its duration is no stand-in, since it
            // spans every stream, so the stream is decoded and its frames counted, the number
            // `select` indexes.
            stream.frames = self.probe_stream(clip, Some(threads))?.frames;
        }
        stream.meta()
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
        restrict_input(&mut cmd);
        cmd.args(["-nostdin", "-v", "error"]);
        if threads > 0 {
            cmd.arg("-threads").arg(threads.to_string());
        }
        // ffprobe measured `v:0`; without the map ffmpeg picks the largest video stream, which may
        // be another one and has other frame sizes.
        cmd.arg("-i")
            .arg(clip.path()?)
            .args(["-map", "0:v:0"])
            .arg("-vf")
            .arg(select_filter(indices))
            .args(["-fps_mode", "passthrough", "-frames:v"])
            .arg(indices.len().to_string())
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped());
        // ffmpeg's messages go to a file beside the clip's: a pipe nobody reads while stdout is
        // being read would stall the decoder once it held more than a pipe does.
        let log_path = clip.path()?.with_extension("log");
        let file = create_private(&log_path)
            .map_err(|e| VideoError::Temp(format!("{}: {e}", log_path.display())))?;
        let log = TempFile(log_path);
        cmd.stderr(Stdio::from(file));
        let mut child = cmd.spawn().map_err(|e| VideoError::Decode(e.to_string()))?;
        let got = read_frames(&mut child, meta, indices, each);
        // A reader that stopped or failed is done with the process, whatever it was doing. One that
        // took every planned frame lets ffmpeg end on its own, so its status is the decode's.
        if !matches!(&got, Ok(Done::All)) {
            let _ = child.kill();
        }
        let waited = child.wait();
        let mut stderr = Vec::new();
        if let Ok(f) = std::fs::File::open(&log.0) {
            let _ = f.take(STDERR_KEPT).read_to_end(&mut stderr);
        }
        match got {
            Ok(Done::Stopped) => Ok(()),
            Ok(Done::All) => match waited {
                Ok(status) if status.success() => Ok(()),
                Ok(status) => Err(VideoError::Decode(format!(
                    "{status}: {}",
                    first_lines(&stderr)
                ))),
                Err(e) => Err(VideoError::Decode(format!("waiting for ffmpeg: {e}"))),
            },
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
    let meta = decoder.probe(&clip, threads)?;
    let (plan, pieces) = plan_under(model, rule, &meta, budget).map_err(MediaError::Video)?;
    Ok(PlannedClip {
        clip,
        meta,
        plan,
        pieces,
        decoder,
        threads,
    })
}

/// The plan of a probed clip under `budget` positions and the text the model writes for it.
fn plan_under(
    model: &dyn MediaModel,
    rule: &VideoRule,
    meta: &ClipMeta,
    budget: usize,
) -> Result<(ClipPlan, Vec<Piece>), VisionError> {
    let plan = vision::video::plan(rule, meta, budget, model.video_sizer())?;
    let pieces = model.video_text(&plan)?;
    Ok((plan, pieces))
}

/// [`plan_clip`] for an input that spends more on the clip than the plan's tokens: `spent` is what
/// a plan takes of the input, its text and delimiters with its tokens, and the plan returned, with
/// its `spent`, is the largest the planner makes that takes at most `room`. The clip is probed
/// once. If the plan under `room` spends more, the caps below it are searched for the largest plan
/// that does not (a plan grows with its cap): the interval of caps halves each round, so the search
/// ends, and the plan is returned only after `spent` has read it. A clip whose least plan spends
/// more than `room` is refused by name.
pub fn plan_clip_within(
    model: &dyn MediaModel,
    decoder: Arc<dyn Decoder>,
    clip: Clip,
    room: usize,
    threads: usize,
    spent: impl Fn(&PlannedClip) -> usize,
) -> Result<(PlannedClip, usize), MediaError> {
    let mut planned = plan_clip(model, decoder, clip, room, threads)?;
    let cost = spent(&planned);
    if cost <= room {
        return Ok((planned, cost));
    }
    let rule = model.video_rule().ok_or(MediaError::NoVideoRule)?;
    // At cap `fits` the planner refuses or its plan spends at most `room`; at `over` its plan
    // spends `least`, more. A cap of 0 is refused, and `room` is over.
    let (mut fits, mut over, mut least) = (0, room, cost);
    while over - fits > 1 {
        let mid = fits + (over - fits) / 2;
        match plan_under(model, rule, &planned.meta, mid) {
            Ok((plan, pieces)) => {
                (planned.plan, planned.pieces) = (plan, pieces);
                let cost = spent(&planned);
                if cost > room {
                    (over, least) = (mid, cost);
                } else {
                    fits = mid;
                }
            }
            Err(_) => fits = mid,
        }
    }
    if let Ok((plan, pieces)) = plan_under(model, rule, &planned.meta, fits) {
        (planned.plan, planned.pieces) = (plan, pieces);
        let cost = spent(&planned);
        if cost <= room {
            return Ok((planned, cost));
        }
    }
    Err(MediaError::Video(VisionError::Video(format!(
        "the clip's least plan takes {least} positions with its text and delimiters; the input \
         leaves {room}"
    ))))
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
    #[cfg(test)]
    #[must_use]
    pub fn frames(&self) -> usize {
        self.plan.indices.len()
    }

    /// The positions the clip takes in an assembled input, counted over [`PlannedClip::pieces`] as
    /// the assembly walks them: a text piece the `text` positions it tokenizes to, a group its
    /// tokens and `around` positions of delimiters ([`group_positions`]).
    #[must_use]
    pub fn positions(&self, text: impl Fn(&str) -> usize, around: usize) -> usize {
        self.pieces
            .iter()
            .map(|piece| match piece {
                Piece::Text(t) => text(t),
                Piece::Group(n) => group_positions(self.plan.groups[*n].tokens, around),
            })
            .sum()
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

/// The positions a medium of `tokens` takes in an assembled embedding input between its delimiters,
/// which are `around` positions together: a group of a clip, a still image or an audio span. The
/// one rule the budget of an input's clips and the assembly both follow.
#[must_use]
pub fn group_positions(tokens: usize, around: usize) -> usize {
    around + tokens
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
    fn probe(&self, clip: &Clip, _threads: usize) -> Result<ClipMeta, VideoError> {
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

    /// A long plan names every frame once and in order, nested no deeper than ffmpeg's parser
    /// takes: no run of flat terms past [`SELECT_RUN`] and parentheses a log of the count deep.
    #[test]
    fn a_long_select_filter_is_a_shallow_tree() {
        let indices: Vec<usize> = (0..768).map(|i| i * 7 + 1).collect();
        let filter = select_filter(&indices);
        let named: Vec<usize> = filter
            .split("eq(n,")
            .skip(1)
            .map(|t| t.split(')').next().unwrap().parse().unwrap())
            .collect();
        assert_eq!(named, indices);
        let (mut depth, mut deepest) = (0usize, 0usize);
        for c in filter.chars() {
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            deepest = deepest.max(depth);
        }
        // `eq(` is a level of its own.
        assert!(deepest <= 1 + 1 + 8, "{deepest} levels of parentheses");
        let mut terms = String::new();
        let mut calls = filter.split("eq(n,");
        terms.push_str(calls.next().unwrap());
        for call in calls {
            terms.push('T');
            terms.push_str(call.split_once(')').unwrap().1);
        }
        assert_eq!(terms.matches('T').count(), indices.len());
        let longest = terms
            .split(['(', ')'])
            .map(|run| run.matches('T').count())
            .max()
            .unwrap();
        assert!(longest <= SELECT_RUN, "a flat run of {longest} terms");
    }

    /// A probe reads the stream's frame count (a counted one over the container's) and its rate; a
    /// container's duration is never read, since it spans every stream. A stream with no count, or
    /// with neither a rate nor a duration of its own, is a named refusal. The size swaps under a
    /// quarter-turn rotation, both spellings of it.
    #[test]
    fn the_probe_reads_frames_rate_and_size() {
        let probe = |s: &str| read_probe(s).and_then(Probed::meta);
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
        // A counted stream (ffprobe -count_frames): the count of frames it decodes to, whatever
        // the container says it holds and however long its audio runs.
        let counted = probe(
            r#"{"streams":[{"width":64,"height":48,"avg_frame_rate":"10/1","nb_read_frames":"20","nb_frames":"22"}],"format":{"duration":"10.000000"}}"#,
        )
        .unwrap();
        assert_eq!((counted.frames, counted.fps), (20, (10, 1)));
        // No frame rate: the stream's frames over its own duration.
        let rate = probe(
            r#"{"streams":[{"width":64,"height":48,"avg_frame_rate":"0/0","nb_frames":"50","duration":"2.000000"}]}"#,
        )
        .unwrap();
        assert_eq!(rate.fps, (50_000, 2_000));
        for (json, words) in [
            // No count to read, and the container's duration is no count.
            (
                r#"{"streams":[{"width":64,"height":48,"avg_frame_rate":"25/1"}],"format":{"duration":"10.0"}}"#,
                "no frame count",
            ),
            (
                r#"{"streams":[{"width":64,"height":48,"avg_frame_rate":"25/1","duration":"2.0"}]}"#,
                "no frame count",
            ),
            // No rate, and the container's duration is not the stream's.
            (
                r#"{"streams":[{"width":64,"height":48,"avg_frame_rate":"0/0","nb_frames":"20"}],"format":{"duration":"10.0"}}"#,
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

    /// A script that stands in for ffmpeg or ffprobe: `body` runs under `sh`. It is executable and
    /// is removed when the returned guard drops. It is run once here (it exits at once when given
    /// `warm-up`), since a new file's first run is slow on some systems and a test's limit must
    /// not count it.
    #[cfg(unix)]
    fn fake_tool(name: &str, body: &str) -> TempFile {
        use std::os::unix::fs::PermissionsExt;
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "bloomery-serve-fake-{name}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = create_private(&path).expect("the script's file");
        let script = TempFile(path);
        let text = format!("#!/bin/sh\n[ \"$1\" = warm-up ] && exit 0\n{body}\n");
        std::io::Write::write_all(&mut file, text.as_bytes()).expect("the script's text");
        drop(file);
        std::fs::set_permissions(&script.0, std::fs::Permissions::from_mode(0o700))
            .expect("the script's mode");
        // A process another test thread forked while the file was open for writing holds it open
        // until it execs, and the kernel refuses to run the file meanwhile ("text file busy").
        let mut busy = 0;
        let warmed = loop {
            match Command::new(&script.0).arg("warm-up").status() {
                Err(e) if e.kind() == std::io::ErrorKind::ExecutableFileBusy && busy < 500 => {
                    busy += 1;
                    std::thread::sleep(Duration::from_millis(10));
                }
                other => break other,
            }
        };
        assert!(
            warmed.as_ref().is_ok_and(|s| s.success()),
            "the script's first run: {warmed:?}"
        );
        script
    }

    /// A decoder whose ffmpeg and ffprobe are the scripts `ffmpeg` and `ffprobe`, its ffprobe runs
    /// ending after `probe_limit`.
    #[cfg(unix)]
    fn fake_decoder(ffmpeg: &TempFile, ffprobe: &TempFile, probe_limit: Duration) -> FfmpegDecoder {
        FfmpegDecoder {
            ffmpeg: ffmpeg.0.clone(),
            ffprobe: ffprobe.0.clone(),
            probe_limit,
            version: String::new(),
        }
    }

    /// A decode that read every planned frame is taken only when ffmpeg itself ended well: a
    /// non-zero exit and a signal are named errors that carry what ffmpeg said, whole frames or not.
    #[cfg(unix)]
    #[test]
    fn a_decode_is_taken_only_when_ffmpeg_ended_well() {
        let meta = ClipMeta {
            frames: 3,
            fps: (1, 1),
            width: 2,
            height: 1,
        };
        let clip = Clip::new(b"not read".to_vec(), None);
        // Three whole 2x1 rgb24 frames, then the ending under test.
        let ends = |ending: &str| {
            let ffmpeg = fake_tool("ffmpeg", &format!("head -c 18 /dev/zero\n{ending}"));
            let ffprobe = fake_tool("ffprobe", "exit 99");
            let decoder = fake_decoder(&ffmpeg, &ffprobe, PROBE_LIMIT);
            let mut seen = 0;
            let r = decoder.decode(&clip, &meta, &[0, 1, 2], 0, &mut |_, _| {
                seen += 1;
                true
            });
            (r.map_err(|e| e.to_string()), seen)
        };
        assert_eq!(ends("exit 0"), (Ok(()), 3));
        for (ending, words) in [
            (
                "echo 'muxer: disk full' >&2\nexit 1",
                ["ffmpeg: ", "exit status: 1", "muxer: disk full"],
            ),
            ("kill -9 $$", ["ffmpeg: ", "signal: 9", "no message"]),
        ] {
            let (r, seen) = ends(ending);
            let said = r.expect_err(ending);
            assert!(words.iter().all(|w| said.contains(w)), "{ending}: {said}");
            assert_eq!(seen, 3, "{ending}: every frame arrived before the ending");
        }
    }

    /// An ffprobe that runs past its limit is killed and the clip refused by name, whether it was
    /// reading the file or counting the stream's frames; one that ends within it is taken.
    #[cfg(unix)]
    #[test]
    fn an_ffprobe_past_its_limit_is_killed_and_named() {
        let clip = Clip::new(b"not read".to_vec(), None);
        let ffmpeg = fake_tool("ffmpeg", "exit 99");
        let limit = Duration::from_secs(1);
        let ask = |body: &str| {
            let ffprobe = fake_tool("ffprobe", body);
            let started = Instant::now();
            let r = fake_decoder(&ffmpeg, &ffprobe, limit).probe(&clip, 1);
            let took = started.elapsed();
            for extension in ["probe-out", "probe-err"] {
                let left = clip.path().unwrap().with_extension(extension);
                assert!(!left.exists(), "{} is left behind", left.display());
            }
            (r.map_err(|e| e.to_string()), took)
        };
        // `exec` leaves no child of the script holding ffprobe's pipes open past the kill.
        let (r, took) = ask("exec sleep 20");
        let said = r.expect_err("a hung read");
        assert!(
            said.starts_with("ffprobe: reading the file ran past 1s and was stopped"),
            "{said}"
        );
        assert!(took < Duration::from_secs(10), "stopped after {took:?}");
        // A stream with no count in its header is counted by a second run, the one that hangs.
        let counting = r#"case " $* " in *" -count_frames "*) exec sleep 20;; esac
echo '{"streams":[{"width":2,"height":1,"avg_frame_rate":"1/1"}]}'"#;
        let (r, took) = ask(counting);
        let said = r.expect_err("a hung count");
        assert!(
            said.starts_with("ffprobe: counting the stream's frames ran past 1s and was stopped"),
            "{said}"
        );
        assert!(took < Duration::from_secs(10), "stopped after {took:?}");
        let stated =
            r#"echo '{"streams":[{"width":2,"height":1,"avg_frame_rate":"1/1","nb_frames":"5"}]}'"#;
        let (r, _) = ask(stated);
        assert!(r.is_ok(), "{r:?}");
    }

    /// The count runs on the threads it is given and the plain read, which decodes nothing, names
    /// none; 0 threads leaves the choice to ffprobe.
    #[cfg(unix)]
    #[test]
    fn a_counting_probe_runs_on_the_threads_it_is_given() {
        let clip = Clip::new(b"not read".to_vec(), None);
        let calls = TempFile(clip.path().unwrap().with_extension("calls"));
        let body = r#"echo "$*" >> 'CALLS'
case " $* " in
*" -count_frames "*) echo '{"streams":[{"width":2,"height":1,"avg_frame_rate":"1/1","nb_read_frames":"7"}]}';;
*) echo '{"streams":[{"width":2,"height":1,"avg_frame_rate":"1/1"}]}';;
esac"#
            .replace("CALLS", &calls.0.display().to_string());
        let (ffmpeg, ffprobe) = (fake_tool("ffmpeg", "exit 99"), fake_tool("ffprobe", &body));
        let decoder = fake_decoder(&ffmpeg, &ffprobe, PROBE_LIMIT);
        for threads in [3, 0] {
            let _ = std::fs::remove_file(&calls.0);
            let meta = decoder.probe(&clip, threads).expect("the probe");
            assert_eq!(meta.frames, 7);
            let text = std::fs::read_to_string(&calls.0).expect("the runs ffprobe saw");
            let runs: Vec<&str> = text.lines().collect();
            assert_eq!(runs.len(), 2, "{text}");
            assert!(!runs[0].contains("-threads"), "the plain read: {}", runs[0]);
            assert!(runs[1].contains("-count_frames"), "the count: {}", runs[1]);
            let named = runs[1].contains(&format!(" -threads {threads} "));
            assert_eq!(named, threads > 0, "{threads} threads: {}", runs[1]);
            assert_eq!(runs[1].contains("-threads"), threads > 0, "{}", runs[1]);
        }
    }

    /// The clip's temp file and ffmpeg's log are made private and exclusive: mode 0600 whatever the
    /// umask says, and a path that exists, a symlink among them, is refused, neither followed nor
    /// truncated.
    #[cfg(unix)]
    #[test]
    fn temp_files_are_private_and_exclusive() {
        use std::os::unix::fs::PermissionsExt;
        let clip = Clip::new(b"abc".to_vec(), None);
        let path = clip.path().unwrap().to_path_buf();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        let link = path.with_extension("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        for taken in [&path, &link] {
            let e = create_private(taken).unwrap_err();
            assert_eq!(e.kind(), std::io::ErrorKind::AlreadyExists, "{taken:?}");
        }
        std::fs::remove_file(&link).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"abc");
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
