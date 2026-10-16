//! The embedding model a server holds beside its chat model: its flags ([`Flags`]), the
//! architectures a side seat serves ([`ROWS`]), the file they name read and sized after the seat's
//! levers and before its plan ([`Flags::read`]), the card it sits on ([`Side::seat`], the image
//! encoder's rule, [`vision_seat::encoder_seat`], chosen before the plan so a stage-card seat's
//! plan holds its bytes as the `embedding model` reserve, [`crate::beside::Beside::embedding`]),
//! and the engine the server's embeddings endpoints call ([`SideEngine`]): a channel to the
//! thread that opened the model, which runs every pass apart from the chat engine's thread.

use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};

use gguf::Split;
use hf::source;
use serve::embed::{Embed, EmbedError};
use serve::engine::Tokenizer;

use crate::GateError;
use crate::generate::Place;
use crate::model_file;
use crate::record::{self, Record};
use crate::vision_seat::{self, EncoderSeat};

#[path = "arch/embedding.rs"]
mod architectures;

#[cfg(feature = "gemma-embedding2")]
mod gemma_embedding2;
pub use architectures::ROWS;

/// The side seat's flags, llama-server's `-m` and `-hf` behind `--embd-`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    model: source::Flags,
    image: vision_seat::Flags,
    ctx: Option<usize>,
}

impl Flags {
    /// `args` with the side seat's flags and their values taken out: the flags and the rest. No
    /// file is read here. A flag with no value or given twice, and `--embd-model` beside
    /// `--embd-hf`, are refused by name.
    pub fn take(args: &[String]) -> Result<(Flags, Vec<String>), GateError> {
        let mut f = Flags::default();
        let mut selected = Vec::new();
        let mut rest = Vec::with_capacity(args.len());
        let mut image_seen = std::collections::HashSet::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let flag = a.as_str();
            if let Some(image_flag) = match flag {
                "--embd-mmproj" => Some("--mmproj"),
                "--embd-no-mmproj" => Some("--no-mmproj"),
                "--embd-image-min-tokens" => Some("--image-min-tokens"),
                "--embd-image-max-tokens" => Some("--image-max-tokens"),
                _ => None,
            } {
                if !image_seen.insert(flag) {
                    return Err(format!("{flag} is given twice").into());
                }
                f.image
                    .take(image_flag, &mut it.by_ref().map(String::as_str))
                    .map_err(|e| format!("{flag}: {e}"))?;
                continue;
            }
            if flag == "--embd-ctx" {
                let v = it.next().ok_or("--embd-ctx needs a value")?;
                let ctx = v
                    .parse::<usize>()
                    .map_err(|e| format!("--embd-ctx {v:?}: {e}"))?;
                if f.ctx.is_some() {
                    return Err("--embd-ctx is given twice".into());
                }
                if ctx == 0 {
                    return Err("--embd-ctx needs one positive count".into());
                }
                f.ctx = Some(ctx);
                continue;
            }
            let flag = match flag {
                "--embd-model" => "--embd-model",
                "--embd-hf" => "--hf",
                _ => {
                    rest.push(a.clone());
                    continue;
                }
            };
            selected.push(flag.to_owned());
            selected.extend(it.next().cloned());
        }
        let (model, _) = source::take(&selected, &["--embd-model"])
            .map_err(|e| format!("embedding flags (--embd-model/--embd-hf): {e}"))?;
        f.model = model;
        if !f.names() && (!image_seen.is_empty() || f.ctx.is_some()) {
            return Err("embedding image/context flags need --embd-model or --embd-hf".into());
        }
        Ok((f, rest))
    }

    /// Whether the flags name an embedding model.
    #[must_use]
    pub fn names(&self) -> bool {
        self.model.path.is_some() || self.model.hf.is_some()
    }

    /// The side model the flags name, `None` when they name none: its file found, or fetched
    /// from `--embd-hf`'s repo; its architecture one of [`ROWS`]; its context the row's
    /// ([`Row::ctx`]); and its bytes. Each step refuses by name. A seat calls it after its levers
    /// and its own arguments, and before its plan.
    pub fn read(&self) -> Result<Option<Side>, GateError> {
        let Some(path) = model_file::resolve(&self.model, None)? else {
            return Ok(None);
        };
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let arch = split.architecture().unwrap_or("<missing>");
        let row = row(arch).ok_or_else(|| {
            let archs: Vec<&str> = ROWS.iter().map(|r| r.arch).collect();
            format!(
                "--embd-model {}: a {arch} file; a side seat serves {}",
                path.display(),
                archs.join(", ")
            )
        })?;
        let train = split
            .arch_get_u64("context_length")
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| format!("{}: no {arch}.context_length", path.display()))?;
        let limit = row.ctx.min(train);
        let ctx = self.ctx.unwrap_or(limit);
        if ctx > limit {
            return Err(
                format!("--embd-ctx {ctx}: the side model supports at most {limit}").into(),
            );
        }
        let mut bytes = (row.bytes)(&split, ctx)
            .map_err(|e| format!("--embd-model {}: {e}", path.display()))?;
        let projector_path = if let Some(p) = self.image.mmproj()? {
            Some(p.to_path_buf())
        } else if let (Some(hf), Some(mmproj)) = (&self.model.hf, row.projector_file) {
            if cfg!(feature = "vision") && !self.image.disabled() {
                let repo = hf::RepoRef::parse(hf)?;
                Some(
                    model_file::fetch_exact(&repo.repo, &[mmproj])?
                        .into_iter()
                        .next()
                        .ok_or("the projector fetch returned no file")?,
                )
            } else {
                None
            }
        } else {
            None
        };
        let projector = match projector_path.as_deref() {
            Some(_) if !cfg!(feature = "vision") => {
                return Err("--embd-mmproj needs the vision build feature".into());
            }
            Some(p) => {
                let p = vision_seat::Projector::open(p, &self.image, row.projector_types)?;
                let n_embd = split
                    .arch_get_u64("embedding_length")
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("the embedding width is absent")?;
                p.check_text(
                    &vision_seat::TextFacts {
                        n_embd,
                        imrope: None,
                        ple: vision_seat::PleImage::Absent,
                    },
                    &|_| None,
                )?;
                bytes = bytes
                    .checked_add(p.card.total())
                    .ok_or("embedding model and tower byte count overflows")?;
                Some(p)
            }
            None if self.image.sets_limits() => {
                return Err(
                    "--embd-image-min-tokens/--embd-image-max-tokens need --embd-mmproj".into(),
                );
            }
            None => None,
        };
        Ok(Some(Side {
            row,
            path,
            ctx,
            bytes,
            projector,
        }))
    }
}

/// An embedding architecture a side seat serves: the file's `general.architecture`, the most
/// positions one input takes, the bytes its model takes on its card, and its open.
pub struct Row {
    /// The architecture.
    pub arch: &'static str,
    /// The model-card input limit, capped further by the file's `context_length`.
    pub ctx: usize,
    /// The bytes the model of the file takes on its card at `ctx` positions: the reserve a plan
    /// holds for it on the stage card, and the figure its open holds itself to.
    pub bytes: fn(&Split, usize) -> Result<u64, String>,
    /// The model of `side`'s file opened on `seat`'s card, on the thread that runs its passes.
    pub open: Open,
    /// The BF16 projector this row reads from an `--embd-hf` repository.
    pub projector_file: Option<&'static str>,
    /// The projector types this embedding architecture accepts.
    pub projector_types: &'static [&'static str],
}

/// A row's open: the model of `side`'s file on `seat`'s card.
pub type Open = fn(&Side, &EncoderSeat) -> Result<Box<dyn Embed>, String>;

/// The row of an embedding architecture, `None` for any other.
#[must_use]
pub fn row(arch: &str) -> Option<&'static Row> {
    ROWS.iter().find(|r| r.arch == arch)
}

/// The refusal of a row this build has no encoder for.
#[cfg(not(feature = "gemma-embedding2"))]
fn no_encoder(arch: &str) -> String {
    format!("a {arch} file is an embedding model this build has no encoder for")
}

#[cfg(not(feature = "gemma-embedding2"))]
fn no_encoder_bytes(split: &Split, _: usize) -> Result<u64, String> {
    Err(no_encoder(split.architecture().unwrap_or("<missing>")))
}

#[cfg(not(feature = "gemma-embedding2"))]
fn no_encoder_open(side: &Side, _: &EncoderSeat) -> Result<Box<dyn Embed>, String> {
    Err(no_encoder(side.row.arch))
}

/// The side model the flags named, read and sized.
pub struct Side {
    /// Its architecture's row.
    pub row: &'static Row,
    /// Its file, a set's first shard.
    pub path: PathBuf,
    /// The most positions one input takes ([`Row::ctx`]).
    pub ctx: usize,
    /// The bytes it takes on its card ([`Row::bytes`]).
    pub bytes: u64,
    /// The image tower sized before the chat plan; its bytes share this side seat's reserve.
    pub projector: Option<vision_seat::Projector>,
}

impl Side {
    /// This model's seat under `place`, chosen before the plan: a card the plan leaves off, else
    /// the stage card, whose plan then holds [`Side::bytes`] as its
    /// [`crate::beside::Beside::embedding`] reserve ([`EncoderSeat::reserve`]).
    pub fn seat(&self, place: Place) -> Result<EncoderSeat, GateError> {
        record::require(record::groups::side_seat::SEAT, "side_seat::Side::seat");
        let seat = vision_seat::encoder_seat(place, self.bytes)?;
        Record::new(&record::EMBEDDING_SEAT)
            .w("arch", self.row.arch)
            .w("card", &seat.name)
            .w("seat", seat.word())
            .u("bytes", self.bytes)
            .u("reserve", seat.reserve.unwrap_or(0))
            .u("ctx", self.ctx)
            .eprint();
        Ok(seat)
    }

    /// Opens an optional model on the seat used by its plan, after the chat model loads.
    pub fn start(seated: Option<(Self, EncoderSeat)>) -> Result<Option<SideEngine>, GateError> {
        seated
            .map(|(side, seat)| SideEngine::spawn(move || (side.row.open)(&side, &seat)))
            .transpose()
    }
}

/// One pass: the input's ids and where its answer goes.
type Job = (
    Vec<u32>,
    Vec<serve::media::MediaFeed>,
    mpsc::Sender<Result<Vec<f32>, EmbedError>>,
);

/// The model the server's embeddings endpoints call, held on a thread of its own: the facts read
/// once at its open, and the channel each pass goes through. Dropping it closes the channel and
/// joins the thread, so the model is released when the drop returns.
pub struct SideEngine {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<JoinHandle<()>>,
    tokenizer: Arc<dyn Tokenizer>,
    name: String,
    ctx: usize,
    dims: Vec<usize>,
    media: Option<serve::media::SharedMediaModel>,
}

impl SideEngine {
    /// The model `open` opens on a new thread, which then runs each pass in order until the
    /// engine drops. An open that fails or panics is the refusal, its panic's message named; a
    /// pass that panics answers an [`EmbedError`] naming the panic's message and ends the
    /// thread, and every pass after it is an [`EmbedError`] too.
    pub fn spawn(
        open: impl FnOnce() -> Result<Box<dyn Embed>, String> + Send + 'static,
    ) -> Result<SideEngine, GateError> {
        let (jobs, rx) = mpsc::channel::<Job>();
        let (opened_tx, opened) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("embd".to_owned())
            .spawn(move || {
                let opened = catch_unwind(open).unwrap_or_else(|p| {
                    Err(format!("the embedding model panicked in its open: {p}"))
                });
                let mut model = match opened {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = opened_tx.send(Err(e));
                        return;
                    }
                };
                let facts = (
                    model.tokenizer(),
                    model.name(),
                    model.ctx(),
                    model.dims(),
                    model.media(),
                );
                if opened_tx.send(Ok(facts)).is_err() {
                    return;
                }
                for (ids, images, answer) in rx {
                    match catch_unwind(|| {
                        if images.is_empty() {
                            model.embed(&ids)
                        } else {
                            model.embed_media(&ids, &images)
                        }
                    }) {
                        Ok(v) => {
                            let _ = answer.send(v);
                        }
                        Err(p) => {
                            let _ = answer.send(Err(EmbedError(format!(
                                "the embedding model panicked: {p}"
                            ))));
                            return;
                        }
                    }
                }
            })
            .map_err(|e| format!("starting the embedding model's thread: {e}"))?;
        let (tokenizer, name, ctx, dims, media) = match opened.recv() {
            Ok(Ok(facts)) => facts,
            refused => {
                // The thread ended with its open; nothing is left on its card.
                let _ = thread.join();
                return Err(match refused {
                    Ok(Err(e)) => e.into(),
                    _ => "the embedding model's thread ended in its open".into(),
                });
            }
        };
        Ok(SideEngine {
            jobs: Some(jobs),
            thread: Some(thread),
            tokenizer,
            name,
            ctx,
            dims,
            media,
        })
    }
}

/// `f`'s value, or its panic's message ([`serve::panic_text`]).
fn catch_unwind<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|p| serve::panic_text(&*p))
}

impl Drop for SideEngine {
    fn drop(&mut self) {
        // Closing the channel ends the thread's loop; the model is dropped there.
        self.jobs = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Embed for SideEngine {
    fn media(&self) -> Option<serve::media::SharedMediaModel> {
        self.media.clone()
    }
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::clone(&self.tokenizer)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn ctx(&self) -> usize {
        self.ctx
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.clone()
    }

    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, EmbedError> {
        self.embed_media(ids, &[])
    }

    fn embed_media(
        &mut self,
        ids: &[u32],
        images: &[serve::media::MediaFeed],
    ) -> Result<Vec<f32>, EmbedError> {
        let ended = || EmbedError("the embedding model's thread ended".to_owned());
        let jobs = self.jobs.as_ref().ok_or_else(ended)?;
        let (answer, answered) = mpsc::channel();
        jobs.send((ids.to_vec(), images.to_vec(), answer))
            .map_err(|_| ended())?;
        answered.recv().map_err(|_| ended())?
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use serve::MockTokenizer;

    use super::*;

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| (*s).to_owned()).collect()
    }

    /// The flags come out of the seat's arguments and the rest stays in order; each misuse is
    /// refused by name.
    #[test]
    fn the_side_flags_come_out_and_misuse_is_refused() {
        let (f, rest) = Flags::take(&args(&[
            "--ctx",
            "4096",
            "--embd-model",
            "/e.gguf",
            "--port",
            "8080",
        ]))
        .expect("take");
        assert_eq!(rest, args(&["--ctx", "4096", "--port", "8080"]));
        assert_eq!(
            f.model.path,
            Some(("--embd-model".to_owned(), "/e.gguf".to_owned()))
        );
        assert!(f.names());
        let (hf, rest) =
            Flags::take(&args(&["--embd-hf", "x/y:Q8_0", "--ctx", "4096"])).expect("HF model");
        assert_eq!(hf.model.hf.as_deref(), Some("x/y:Q8_0"));
        assert!(hf.model.path.is_none());
        assert!(hf.names());
        assert_eq!(rest, args(&["--ctx", "4096"]));
        let (none, _) = Flags::take(&args(&["--port", "1"])).expect("none");
        assert_eq!(none, Flags::default());
        assert!(!none.names());
        let (image, rest) = Flags::take(&args(&[
            "--embd-model",
            "/e.gguf",
            "--embd-ctx",
            "8192",
            "--embd-mmproj",
            "/p.gguf",
            "--embd-image-min-tokens",
            "70",
            "--embd-image-max-tokens",
            "1120",
            "--port",
            "8080",
        ]))
        .expect("image flags");
        assert_eq!(rest, args(&["--port", "8080"]));
        assert_eq!(image.ctx, Some(8192));
        assert_eq!(
            image.image.mmproj().expect("projector"),
            Some(std::path::Path::new("/p.gguf"))
        );
        assert!(image.image.sets_limits());
        let default = crate::vision_seat::TokenLimits::new(1, 1).expect("limits");
        let limits = image.image.limits_over(default).expect("image limits");
        assert_eq!((limits.min(), limits.max()), (70, 1120));
        for flag in ["--embd-image-min-tokens", "--embd-image-max-tokens"] {
            let (zero, _) =
                Flags::take(&args(&["--embd-model", "/a", flag, "0"])).expect("count parsed");
            let e = zero
                .image
                .limits_over(default)
                .expect_err("zero limit refused")
                .to_string();
            assert!(
                e.starts_with("--image-min-tokens/--image-max-tokens: image size limits: "),
                "{flag}: {e}"
            );
            assert!(e.contains("need 1 <= min <= max"), "{flag}: {e}");
            let (one, _) =
                Flags::take(&args(&["--embd-model", "/a", flag, "1"])).expect("count parsed");
            let limits = one.image.limits_over(default).expect("one token limit");
            assert_eq!((limits.min(), limits.max()), (1, 1));
        }
        for (bad, why) in [
            (&["--embd-model"][..], "--embd-model needs a value"),
            (&["--embd-hf"], "--hf needs a value"),
            (&["--embd-model", "/a", "--embd-model", "/b"], "named twice"),
            (&["--embd-hf", "x/y", "--embd-hf", "a/b"], "named twice"),
            (&["--embd-model", "/a", "--embd-hf", "x/y"], "named twice"),
            (&["--embd-hf", "x/y", "--embd-model", "/a"], "named twice"),
            (&["--embd-mmproj", "/p"], "need --embd-model or --embd-hf"),
            (&["--embd-no-mmproj"], "need --embd-model or --embd-hf"),
            (
                &["--embd-image-min-tokens", "70"],
                "need --embd-model or --embd-hf",
            ),
            (
                &["--embd-image-max-tokens", "1120"],
                "need --embd-model or --embd-hf",
            ),
            (&["--embd-ctx", "8192"], "need --embd-model or --embd-hf"),
            (
                &["--embd-model", "/a", "--embd-ctx", "1", "--embd-ctx", "2"],
                "--embd-ctx is given twice",
            ),
            (
                &["--embd-model", "/a", "--embd-ctx", "0"],
                "needs one positive count",
            ),
            (
                &[
                    "--embd-model",
                    "/a",
                    "--embd-mmproj",
                    "/p",
                    "--embd-mmproj",
                    "/q",
                ],
                "--embd-mmproj is given twice",
            ),
            (
                &["--embd-model", "/a", "--embd-no-mmproj", "--embd-no-mmproj"],
                "--embd-no-mmproj is given twice",
            ),
            (
                &[
                    "--embd-model",
                    "/a",
                    "--embd-image-min-tokens",
                    "70",
                    "--embd-image-min-tokens",
                    "80",
                ],
                "--embd-image-min-tokens is given twice",
            ),
            (
                &[
                    "--embd-model",
                    "/a",
                    "--embd-image-max-tokens",
                    "1120",
                    "--embd-image-max-tokens",
                    "1200",
                ],
                "--embd-image-max-tokens is given twice",
            ),
            (&["--embd-model", "/a", "--embd-mmproj"], "--embd-mmproj"),
        ] {
            let e = Flags::take(&args(bad)).expect_err("refused").to_string();
            assert!(e.contains(why), "{bad:?}: {e}");
        }
    }

    /// A model that answers a pass, panics on the id 0, and says when it is dropped.
    struct Stub {
        dropped: Arc<AtomicBool>,
        thread: thread::ThreadId,
    }

    impl Embed for Stub {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            Arc::new(MockTokenizer)
        }
        fn name(&self) -> String {
            "stub".to_owned()
        }
        fn ctx(&self) -> usize {
            8
        }
        fn dims(&self) -> Vec<usize> {
            vec![1]
        }
        fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, EmbedError> {
            assert_eq!(thread::current().id(), self.thread);
            assert!(!ids.contains(&0), "the stub met id 0");
            Ok(vec![ids.len() as f32])
        }
    }

    impl Drop for Stub {
        fn drop(&mut self) {
            assert_eq!(thread::current().id(), self.thread);
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    /// The engine runs each pass on the model's thread; a panic there answers its message and
    /// ends the thread, and later passes name the end; an open that fails or panics is the
    /// refusal; the drop returns once the model is dropped.
    #[test]
    fn the_side_engine_answers_on_its_thread_and_names_its_end() {
        let caller = thread::current().id();
        let dropped = Arc::new(AtomicBool::new(false));
        let d = Arc::clone(&dropped);
        let mut engine = SideEngine::spawn(move || {
            let owner = thread::current().id();
            assert_ne!(owner, caller, "the model opens on a worker");
            Ok(Box::new(Stub {
                dropped: d,
                thread: owner,
            }) as Box<dyn Embed>)
        })
        .expect("open");
        assert_eq!(
            (engine.name(), engine.ctx(), engine.dims()),
            ("stub".to_owned(), 8, vec![1])
        );
        assert_eq!(engine.embed(&[5, 6]).expect("a pass"), vec![2.0]);
        let e = engine.embed(&[0]).expect_err("the panic");
        assert_eq!(e.0, "the embedding model panicked: the stub met id 0");
        let e = engine.embed(&[5]).expect_err("the end");
        assert_eq!(e.0, "the embedding model's thread ended");
        drop(engine);
        assert!(
            dropped.load(Ordering::SeqCst),
            "the model is dropped by the engine's drop"
        );

        let dropped = Arc::new(AtomicBool::new(false));
        let d = Arc::clone(&dropped);
        drop(
            SideEngine::spawn(move || {
                let owner = thread::current().id();
                assert_ne!(owner, caller, "the model opens on a worker");
                Ok(Box::new(Stub {
                    dropped: d,
                    thread: owner,
                }) as Box<dyn Embed>)
            })
            .expect("open"),
        );
        assert!(
            dropped.load(Ordering::SeqCst),
            "an idle engine's drop joins its thread"
        );

        let e = SideEngine::spawn(|| Err("no card".to_owned()))
            .err()
            .expect("refused");
        assert_eq!(e.to_string(), "no card");
        let e = SideEngine::spawn(|| panic!("the open broke"))
            .err()
            .expect("refused");
        assert_eq!(
            e.to_string(),
            "the embedding model panicked in its open: the open broke"
        );
    }
}
