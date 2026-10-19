//! The server's video decode against a real ffmpeg (`just gate-serve-video`): a lossless clip whose
//! every frame is one colour naming its index is probed, planned by two rules (a centred tick
//! rule, a linspace rule) and decoded, and the frames that arrive are the plan's, each whole and
//! uniform and in order, whatever the thread count. A clip of two video streams decodes the first,
//! the stream the probe measured, and a clip with audio longer than its video probes the video's
//! own frame count. A decode past the file's end and a file that is no video are named errors. A
//! 60-second 1080p H.264 clip is planned and decoded at four threads; every planned frame arrives
//! whole and in order. ffmpeg is a requirement of the gate, so its absence fails by name rather
//! than skipping.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serve::video::{Clip, Decoder, FfmpegDecoder, VideoError};
use vision::VisionError;
use vision::video::{ClipMeta, Overflow, Sampler, Sizer, Stamp, Tail, VideoRule, plan};

/// Frames of the lossless clip, its rate and its size: four seconds of 64×48 at ten a second.
const FRAMES: usize = 40;
const WIDTH: usize = 64;
const HEIGHT: usize = 48;

/// A directory under the system's temp dir that the test owns and removes.
struct Dir(PathBuf);

impl Dir {
    fn new(test: &str) -> Dir {
        let path = std::env::temp_dir().join(format!(
            "bloomery-serve-video-{test}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("the test's directory");
        Dir(path)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn decoder() -> FfmpegDecoder {
    match FfmpegDecoder::detect() {
        Ok(d) => {
            println!("video gate: ffmpeg {}", d.version);
            d
        }
        Err(e) => panic!("this gate needs ffmpeg and ffprobe on PATH: {e}"),
    }
}

/// ffmpeg with `args` writing `out`, its stdin the `stdin` bytes or nothing.
fn ffmpeg(args: &[&str], stdin: Option<&[u8]>, out: &Path) {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-v", "error"]);
    if stdin.is_none() {
        cmd.arg("-nostdin");
    }
    cmd.args(args).arg(out).stdin(if stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mut child = cmd.spawn().expect("ffmpeg starts");
    if let Some(bytes) = stdin {
        child
            .stdin
            .take()
            .expect("ffmpeg's stdin")
            .write_all(bytes)
            .expect("the frames reach ffmpeg");
    }
    let status = child.wait().expect("ffmpeg ends");
    assert!(status.success(), "ffmpeg {args:?}: {status}");
}

/// Frame `i`'s colour: its index in red and green, a fixed blue.
fn colour(i: usize) -> [u8; 3] {
    [(i & 255) as u8, ((i >> 8) & 255) as u8, 0xA5]
}

/// The lossless clip: `FRAMES` frames, frame `i` every pixel `colour(i)`.
fn index_clip(dir: &Dir) -> Clip {
    let mut raw = Vec::with_capacity(FRAMES * WIDTH * HEIGHT * 3);
    for i in 0..FRAMES {
        for _ in 0..WIDTH * HEIGHT {
            raw.extend(colour(i));
        }
    }
    let path = dir.file("index.mkv");
    ffmpeg(
        &[
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{WIDTH}x{HEIGHT}"),
            "-r",
            "10",
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
        ],
        Some(&raw),
        &path,
    );
    Clip::new(std::fs::read(&path).expect("the clip"), Some("mkv"))
}

/// One token a group at the clip's own size: the rule's numbers are all this gate reads.
struct Fixed;

impl Sizer for Fixed {
    fn group_size(
        &self,
        width: usize,
        height: usize,
        _cap: usize,
    ) -> Result<((usize, usize), usize), VisionError> {
        Ok(((width, height), 1))
    }
}

/// A tick rule: Gemma's, one frame a second at the middle of each second.
const TICKS: VideoRule = vision::arch::gemma4v::VIDEO_RULE;

/// A linspace rule: two frames a second spread over the clip, at most six here and 768 for the
/// long clip, a pair of frames to a group.
const fn linspace(most: usize) -> VideoRule {
    VideoRule {
        fps: (2, 1),
        frames: (2, most),
        sampler: Sampler::Linspace,
        overflow: Overflow::Subsample,
        temporal: 2,
        tail: Tail::RepeatLast,
        group_tokens: 768,
        clip_tokens: 12_288,
        overhead: 0,
        stamp: Stamp::GroupMean,
    }
}

/// A decoded frame's index, its colour when the frame is one colour throughout, and its size.
type Decoded = (usize, Option<[u8; 3]>, (usize, usize));

/// Decode `indices` at `threads`: each frame that arrives, as a [`Decoded`].
fn decode(
    decoder: &FfmpegDecoder,
    clip: &Clip,
    meta: &ClipMeta,
    indices: &[usize],
    threads: usize,
) -> Result<Vec<Decoded>, VideoError> {
    let mut got = Vec::new();
    decoder.decode(clip, meta, indices, threads, &mut |i, frame| {
        let first = [frame.data[0], frame.data[1], frame.data[2]];
        let uniform = frame.data.as_chunks::<3>().0.iter().all(|p| *p == first);
        assert_eq!(frame.data.len(), frame.width * frame.height * 3);
        got.push((i, uniform.then_some(first), (frame.width, frame.height)));
        true
    })?;
    Ok(got)
}

/// The probe reads the lossless clip's size, rate and, with no `nb_frames` in a Matroska stream,
/// its frame count from the duration; a plan's frames arrive in order, whole, each the colour of
/// its own index, whatever the thread count.
#[test]
#[ignore = "gate: just gate-serve-video"]
fn hw_ffv1_frames_arrive_as_the_plans_indices() {
    let decoder = decoder();
    let dir = Dir::new("ffv1");
    let clip = index_clip(&dir);
    let meta = decoder.probe(&clip).expect("the probe");
    assert_eq!(
        meta,
        ClipMeta {
            frames: FRAMES,
            fps: (10, 1),
            width: WIDTH,
            height: HEIGHT
        }
    );
    // Ticks at the middle of seconds 0 to 3, the last source frame before each midpoint; and six
    // frames spread evenly over the 40, in three pairs.
    let cases: [(&str, VideoRule, &[usize]); 2] = [
        ("ticks", TICKS, &[4, 14, 24, 34]),
        ("linspace", linspace(6), &[0, 8, 16, 23, 31, 39]),
    ];
    for (name, rule, want) in cases {
        let p = plan(&rule, &meta, 10_000, &Fixed).expect(name);
        assert_eq!(p.indices, want, "{name}: the plan's indices");
        let mut sequences = Vec::new();
        for threads in [1, 4] {
            let frames = decode(&decoder, &clip, &meta, &p.decode_indices(), threads)
                .unwrap_or_else(|e| panic!("{name} at {threads} threads: {e}"));
            let seen: Vec<usize> = frames.iter().map(|f| f.0).collect();
            assert_eq!(seen, p.decode_indices(), "{name}: the frames that arrive");
            for (i, colour_seen, size) in &frames {
                assert_eq!(size, &(WIDTH, HEIGHT), "{name}: frame {i}'s size");
                assert_eq!(
                    colour_seen,
                    &Some(colour(*i)),
                    "{name}: frame {i} is not the colour of its index"
                );
            }
            sequences.push(frames);
        }
        assert_eq!(
            sequences[0], sequences[1],
            "{name}: thread count changes a frame"
        );
        println!(
            "gate video {name}: indices {:?} decoded in order, colours match",
            p.indices
        );
    }
}

/// A clip with a 64×64 first video stream (red) and a 128×128 second (blue), neither flagged
/// default, decodes the first, the stream the probe measured. ffmpeg left to itself takes the
/// larger (a stream flagged default wins over size), and its frames, read 64×64 at a time, would
/// be whole and uniform: only their colour tells.
#[test]
#[ignore = "gate: just gate-serve-video"]
fn hw_the_decode_takes_the_stream_the_probe_measured() {
    let decoder = decoder();
    let dir = Dir::new("streams");
    let path = dir.file("two.mkv");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=64x64:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:size=128x128:rate=10:duration=2",
            "-map",
            "0:v",
            "-map",
            "1:v",
            "-c:v",
            "ffv1",
            "-disposition:v:0",
            "0",
            "-disposition:v:1",
            "0",
            "-default_mode",
            "passthrough",
        ],
        None,
        &path,
    );
    let clip = Clip::new(std::fs::read(&path).expect("the clip"), Some("mkv"));
    let meta = decoder.probe(&clip).expect("the probe");
    assert_eq!((meta.width, meta.height), (64, 64), "the probe's stream");
    let frames = decode(&decoder, &clip, &meta, &[2, 9, 15], 1).expect("the decode");
    assert_eq!(frames.len(), 3);
    for (i, colour_seen, size) in &frames {
        assert_eq!(size, &(64, 64), "frame {i}'s size");
        let [r, _, b] = colour_seen.unwrap_or_else(|| panic!("frame {i} is not one colour"));
        assert!(
            r > 200 && b < 60,
            "frame {i} is not the first stream's red: r {r}, b {b}"
        );
    }
    println!("gate video: two streams decode the 64x64 first");
}

/// A Matroska clip of 2 s of video at 10 fps beside 10 s of audio has 20 frames: the container
/// states neither a frame count nor the video stream's duration, and its own duration is the
/// audio's. The probe counts the video stream's frames, so the tick rule plans two ticks (frames 4
/// and 14) and both decode; the container's duration would make 100 frames, ten ticks, eight of
/// them past the end.
#[test]
#[ignore = "gate: just gate-serve-video"]
fn hw_audio_longer_than_the_video_does_not_stretch_its_frame_count() {
    let decoder = decoder();
    let dir = Dir::new("audio");
    let path = dir.file("audio.mkv");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=64x48:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=10",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
        ],
        None,
        &path,
    );
    let clip = Clip::new(std::fs::read(&path).expect("the clip"), Some("mkv"));
    let meta = decoder.probe(&clip).expect("the probe");
    assert_eq!(
        meta,
        ClipMeta {
            frames: 20,
            fps: (10, 1),
            width: WIDTH,
            height: HEIGHT
        },
        "the probe counts the video stream, not the container's duration"
    );
    let p = plan(&TICKS, &meta, 10_000, &Fixed).expect("the plan");
    assert_eq!(p.indices, [4, 14]);
    let frames = decode(&decoder, &clip, &meta, &p.decode_indices(), 1).expect("the decode");
    assert_eq!(frames.len(), 2);
    println!("gate video: a 20-frame clip beside 10 s of audio probes 20 frames");
}

/// A decode that asks past the file's end is `Short`, naming the counts; a file that is no video
/// is the probe's named refusal.
#[test]
#[ignore = "gate: just gate-serve-video"]
fn hw_a_decode_past_the_end_and_a_non_video_are_named_errors() {
    let decoder = decoder();
    let dir = Dir::new("errors");
    let clip = index_clip(&dir);
    let meta = decoder.probe(&clip).expect("the probe");
    match decode(&decoder, &clip, &meta, &[38, 39, 45], 2) {
        Err(VideoError::Short { wanted: 3, got: 2 }) => {}
        other => panic!("a decode past the end: {other:?}"),
    }
    let junk = Clip::new(b"this is not a video".to_vec(), Some("mp4"));
    match decoder.probe(&junk) {
        Err(VideoError::Probe(said)) => println!("gate video: a non-video probes as: {said}"),
        other => panic!("a non-video: {other:?}"),
    }
}

/// A 60-second 1080p H.264 clip at 30 fps (x264's default preset, so CABAC and the loop filter are
/// on) is probed, planned at two frames a second (120 frames, 60 pairs) and decoded at four
/// threads; every planned frame arrives whole and in order.
#[test]
#[ignore = "gate: just gate-serve-video"]
fn hw_h264_clip_decodes_its_plan() {
    let decoder = decoder();
    let dir = Dir::new("h264");
    let path = dir.file("long.mp4");
    ffmpeg(
        &[
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30:duration=60",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
        ],
        None,
        &path,
    );
    let clip = Clip::new(std::fs::read(&path).expect("the clip"), Some("mp4"));
    let meta = decoder.probe(&clip).expect("the probe");
    assert_eq!(
        meta,
        ClipMeta {
            frames: 1800,
            fps: (30, 1),
            width: 1920,
            height: 1080
        }
    );
    let p = plan(&linspace(768), &meta, 12_288, &Fixed).expect("the plan");
    assert_eq!(p.indices.len(), 120);
    assert_eq!(p.groups.len(), 60);
    let want = p.decode_indices();
    let mut n = 0;
    let mut last = None;
    decoder
        .decode(&clip, &meta, &want, 4, &mut |i, frame| {
            assert_eq!(frame.data.len(), 1920 * 1080 * 3);
            assert!(last.is_none_or(|l| i > l), "frames arrive in order");
            last = Some(i);
            n += 1;
            true
        })
        .expect("the decode");
    assert_eq!(n, want.len());
}
