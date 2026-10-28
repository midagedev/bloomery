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
use serve::embed::{Embed, EmbedError};
use serve::engine::Tokenizer;

use crate::GateError;
use crate::generate::Place;
use crate::record::{self, Record};
use crate::vision_seat::{self, EncoderSeat};

#[path = "arch/embedding.rs"]
mod architectures;

#[cfg(feature = "hf")]
mod flags;
#[cfg(feature = "gemma-embedding2")]
mod gemma_embedding2;
pub use architectures::ROWS;
#[cfg(feature = "hf")]
pub use flags::Flags;

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
    /// The optional audio tower's bytes in the same projector file.
    pub audio_bytes: fn(&gguf::Gguf) -> Result<u64, GateError>,
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
    /// Named prefixes read from the model's sentence-transformers config.
    prompts: std::collections::BTreeMap<String, String>,
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
    /// Audio buffers included beside the image buffers in this seat's named reserve.
    pub audio_bytes: u64,
    /// Host bytes shared by pooled vectors and image tower outputs; zero disables it.
    pub cache_bytes: usize,
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
            .map(|(mut side, seat)| {
                let prompts = std::mem::take(&mut side.prompts);
                let mut engine = SideEngine::spawn(move || (side.row.open)(&side, &seat))?;
                engine.prompts = prompts;
                Ok(engine)
            })
            .transpose()
    }
}

/// One pass: the input's ids and where its answer goes.
type Job = (
    Vec<u32>,
    Vec<serve::media::MediaFeed>,
    Vec<serve::embed::AudioFeed>,
    mpsc::Sender<Result<Vec<f32>, EmbedError>>,
);

/// The model the server's embeddings endpoints call, held on a thread of its own: the facts read
/// once at its open, and the channel each pass goes through. Dropping it closes the channel and
/// joins the thread, so the model is released when the drop returns, unless a pass panicked
/// ([`SideEngine::spawn`]).
pub struct SideEngine {
    prompts: std::collections::BTreeMap<String, String>,
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<JoinHandle<()>>,
    tokenizer: Arc<dyn Tokenizer>,
    name: String,
    path: String,
    ctx: usize,
    dims: Vec<usize>,
    media: Option<serve::media::SharedMediaModel>,
    audio: Option<serve::media::SharedMediaModel>,
    cache: Option<serve::embed_cache::Shared>,
}

impl SideEngine {
    /// The model `open` opens on a new thread, which then runs each pass in order until the
    /// engine drops. An open that fails or panics is the refusal, its panic's message named; a
    /// pass that panics answers an [`EmbedError`] naming the panic's message and ends the
    /// thread, and every pass after it is an [`EmbedError`] too. The panicked model is left to
    /// the process exit, never dropped: its Drop frees through the device driver, and once the
    /// error is answered the server may stop and exit, tearing the driver's libraries down
    /// beside a Drop still running.
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
                    model.path(),
                    model.ctx(),
                    model.dims(),
                    model.media(),
                    model.audio(),
                    model.cache(),
                );
                if opened_tx.send(Ok(facts)).is_err() {
                    return;
                }
                for (ids, images, audio, answer) in rx {
                    match catch_unwind(|| {
                        if images.is_empty() && audio.is_empty() {
                            model.embed(&ids)
                        } else {
                            model.embed_media(&ids, &images, &audio)
                        }
                    }) {
                        Ok(v) => {
                            let _ = answer.send(v);
                        }
                        Err(p) => {
                            std::mem::forget(model);
                            let _ = answer.send(Err(EmbedError(format!(
                                "the embedding model panicked: {p}"
                            ))));
                            return;
                        }
                    }
                }
            })
            .map_err(|e| format!("starting the embedding model's thread: {e}"))?;
        let (tokenizer, name, path, ctx, dims, media, audio, cache) = match opened.recv() {
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
            prompts: std::collections::BTreeMap::default(),
            jobs: Some(jobs),
            thread: Some(thread),
            tokenizer,
            name,
            path,
            ctx,
            dims,
            media,
            audio,
            cache,
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
    fn prompts(&self) -> std::collections::BTreeMap<String, String> {
        self.prompts.clone()
    }

    fn cache(&self) -> Option<serve::embed_cache::Shared> {
        self.cache.clone()
    }
    fn media(&self) -> Option<serve::media::SharedMediaModel> {
        self.media.clone()
    }
    fn audio(&self) -> Option<serve::media::SharedMediaModel> {
        self.audio.clone()
    }
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::clone(&self.tokenizer)
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn path(&self) -> String {
        self.path.clone()
    }

    fn ctx(&self) -> usize {
        self.ctx
    }

    fn dims(&self) -> Vec<usize> {
        self.dims.clone()
    }

    fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>, EmbedError> {
        self.embed_media(ids, &[], &[])
    }

    fn embed_media(
        &mut self,
        ids: &[u32],
        images: &[serve::media::MediaFeed],
        audio: &[serve::embed::AudioFeed],
    ) -> Result<Vec<f32>, EmbedError> {
        let ended = || EmbedError("the embedding model's thread ended".to_owned());
        let jobs = self.jobs.as_ref().ok_or_else(ended)?;
        let (answer, answered) = mpsc::channel();
        jobs.send((ids.to_vec(), images.to_vec(), audio.to_vec(), answer))
            .map_err(|_| ended())?;
        answered.recv().map_err(|_| ended())?
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use serve::MockTokenizer;

    use super::*;

    /// A model that answers a pass, panics on the id 0, and says when it is dropped.
    struct Stub {
        dropped: Arc<AtomicBool>,
        thread: thread::ThreadId,
        cache: Option<serve::embed_cache::Shared>,
    }

    impl Embed for Stub {
        fn cache(&self) -> Option<serve::embed_cache::Shared> {
            self.cache.clone()
        }
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            Arc::new(MockTokenizer)
        }
        fn name(&self) -> String {
            "stub".to_owned()
        }
        fn path(&self) -> String {
            "/models/stub.gguf".to_owned()
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
    /// ends the thread, leaving the model to the process exit, and later passes name the end; an
    /// open that fails or panics is the refusal; an idle engine's drop returns once the model is
    /// dropped.
    #[test]
    fn the_side_engine_answers_on_its_thread_and_names_its_end() {
        let caller = thread::current().id();
        let dropped = Arc::new(AtomicBool::new(false));
        let d = Arc::clone(&dropped);
        let cache = serve::embed_cache::Cache::shared(4096).unwrap();
        let expected_cache = cache.clone();
        let mut engine = SideEngine::spawn(move || {
            let owner = thread::current().id();
            assert_ne!(owner, caller, "the model opens on a worker");
            Ok(Box::new(Stub {
                dropped: d,
                thread: owner,
                cache: Some(cache),
            }) as Box<dyn Embed>)
        })
        .expect("open");
        assert!(Arc::ptr_eq(&engine.cache().unwrap(), &expected_cache));
        assert_eq!(
            (engine.name(), engine.path(), engine.ctx(), engine.dims()),
            (
                "stub".to_owned(),
                "/models/stub.gguf".to_owned(),
                8,
                vec![1]
            )
        );
        assert_eq!(engine.embed(&[5, 6]).expect("a pass"), vec![2.0]);
        let e = engine.embed(&[0]).expect_err("the panic");
        assert_eq!(e.0, "the embedding model panicked: the stub met id 0");
        let e = engine.embed(&[5]).expect_err("the end");
        assert_eq!(e.0, "the embedding model's thread ended");
        drop(engine);
        // PIN(2026-10-23): a panicked model is no longer dropped on its thread; the check that the
        // engine's drop released it became the check that it is left to the process exit.
        assert!(
            !dropped.load(Ordering::SeqCst),
            "a panicked side model is left to the process exit"
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
                    cache: None,
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
