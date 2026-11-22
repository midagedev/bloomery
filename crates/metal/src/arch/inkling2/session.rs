//! An Inkling 2 model on this backend: the open (the check, the arena, the
//! residency set), the token step over the visits, the read-backs and the
//! fault word's poison.
//!
//! A step is host-fed: the host embeds the id, the walk ([`super::ops`]) runs
//! the visits in order, and the head picks over the file's vocabulary cut. The
//! routed experts launch at host-known ids, so each visit's command buffer
//! ends at its router and the host reads the picks before it encodes the
//! experts; a pass the loop runs again also ends its buffer, where the host
//! keeps the pass's blocks for the next pass's messages. A prompt runs as
//! steps (`target.rs`); the batched pass is not this family's yet.

use std::fmt;

use bloomery_device::fault::{Fault, FaultWord};
use bloomery_device::{DeviceError, Elems};
use model_hf::ModelFile;

use super::arena::Arena;
use super::check::{self, Checked, Refusal};
use super::ops::Walk;
use crate::device::MetalDevice;
use crate::instance::{HOST, Instance};
use crate::kernels::Kernels;
use crate::library::Library;
use crate::residency::Alloc;
use crate::session::room::Bytes;
use crate::stream::MetalStream;

/// The family's name in an error.
const WHAT: &str = "inkling2 session";

/// Why a session refused or failed.
#[derive(Debug)]
pub enum SessionError {
    Device(DeviceError),
    /// The check's refusal of the file, by name.
    Refused(Refusal),
    Shape {
        what: &'static str,
        detail: String,
    },
    /// The step raised the fault word: the session is poisoned until reset.
    Fault(Fault),
    /// A step refused while the session is poisoned.
    Poisoned(Fault),
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionError::Device(e) => write!(f, "{WHAT}: {e}"),
            SessionError::Refused(r) => write!(f, "{r}"),
            SessionError::Shape { what, detail } => write!(f, "{what}: {detail}"),
            SessionError::Fault(fault) => write!(
                f,
                "{WHAT}: the step raised the fault word: {fault}; the session is poisoned until \
                 reset"
            ),
            SessionError::Poisoned(fault) => write!(
                f,
                "{WHAT}: a step refused: the session is poisoned by {fault}; reset it first"
            ),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<DeviceError> for SessionError {
    fn from(e: DeviceError) -> Self {
        SessionError::Device(e)
    }
}

/// A refusal of the session, by name.
pub(super) fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// A node of the walk, by the reference graph's name ([`Tap::name`]): what
/// the e2e gate reads a visit's values of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tap {
    /// The embedded and normed row the walk starts from (handed with visit 0).
    Embed,
    /// The q, k, v and relative rows after their convs and clips.
    Q,
    K,
    V,
    R,
    /// The attention's output before its gate.
    Kqv,
    /// The attention's output after the gate, `wo` and the post-conv (absent
    /// at a boundary visit, as in the reference).
    AttnOut,
    /// The stream after the attention's residual.
    FfnInp,
    /// The FFN's input norm.
    FfnNorm,
    /// The router's sigmoid scores.
    Probs,
    /// The router's kept ids and their scaled weights.
    TopK,
    Weights,
    /// The experts' sum, before the post-conv.
    MoeOut,
    /// The FFN's output after its post-conv, and the stream after it.
    FfnOut,
    LOut,
    /// The q path's and the FFN's inputs after the loop's controllers.
    LoopQ,
    LoopFfn,
}

impl Tap {
    /// The reference graph's node name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tap::Embed => "norm",
            Tap::Q => "Qcur",
            Tap::K => "Kcur",
            Tap::V => "Vcur",
            Tap::R => "Rcur",
            Tap::Kqv => "kqv_out",
            Tap::AttnOut => "attn_out",
            Tap::FfnInp => "ffn_inp",
            Tap::FfnNorm => "norm",
            Tap::Probs => "ffn_moe_probs",
            Tap::TopK => "ffn_moe_topk",
            Tap::Weights => "ffn_moe_weights_scaled",
            Tap::MoeOut => "ffn_moe_out",
            Tap::FfnOut => "ffn_out",
            Tap::LOut => "l_out",
            Tap::LoopQ => "loop_q_inp",
            Tap::LoopFfn => "loop_ffn_inp",
        }
    }
}

/// The receiver of a tapped step's nodes: the visit, the node, its values.
/// A visit's router hands on its own kept ids ([`Tap::TopK`]), whatever
/// route a forced step then takes.
pub type TapSink<'a> = &'a mut dyn FnMut(usize, Tap, &[f32]);

/// The route a forced step takes at a visit: the kept ids and their scaled
/// weights, in place of the router's (the e2e gate's replay of a reference
/// chain, whose near-tied scores another engine's roundings may order
/// otherwise), or the reason there is none.
pub type Routes<'a> = &'a mut dyn FnMut(usize) -> Result<([u32; 8], [f32; 8]), String>;

/// The rows a replayed step's visit starts from, given the visit and the
/// blocks the walk has closed: those blocks then the stream, in place of the
/// walk's own (the e2e gate's replay of a reference chain a visit at a time,
/// each visit fed the reference's own inputs); None keeps the walk's own; an
/// error is the reason there are none. The head is the visit after the last.
pub type Inputs<'a> = &'a mut dyn FnMut(usize, usize) -> Result<Option<Vec<f32>>, String>;

/// An Inkling 2 model, open on one stream.
pub struct Session {
    stream: MetalStream,
    kernels: Kernels,
    checked: Checked,
    arena: Arena,
    pos: u32,
    poisoned: Option<Fault>,
    /// The last step's logits after the epilogue, read on request.
    logits: Vec<f32>,
    logits_fresh: bool,
    /// The epilogue's scale before the cap.
    logit_scale: f32,
}

impl Session {
    /// The instance whose library a session of `file` runs on: the build's
    /// host instance. The family's own kernels (`kernels::inkling2`) read the
    /// family's geometry (`inkling2_geom`), and the shared launches it takes
    /// (the segmented mat-vec, the mat-vec, the row norm, the bias add, the
    /// combine and the head) take their shapes as launch words, so any
    /// library runs it; the Mini's own row ([`crate::instance::INKLING2_MINI`])
    /// cannot compile a library, its sinks and its biased sigmoid router being
    /// rows the shared flash and router refuse.
    ///
    /// # Errors
    /// The check's first refusal of the file at `ctx` rows.
    pub fn instance_of(file: &ModelFile, ctx: usize) -> Result<&'static Instance, SessionError> {
        check::check(file.header(), ctx)
            .checked_or_first()
            .map_err(SessionError::Refused)?;
        Ok(&HOST)
    }

    /// A session of `file` at a cache of `ctx` rows on `device`, its kernels
    /// built from `library`. Every buffer is made and the residency set
    /// committed here.
    ///
    /// # Errors
    /// The check's refusal, the room's, or the device's, by name.
    pub fn open(
        device: &MetalDevice,
        library: &Library,
        file: &ModelFile,
        ctx: usize,
    ) -> Result<Session, SessionError> {
        let split = file.header();
        let checked = check::check(split, ctx)
            .checked_or_first()
            .map_err(SessionError::Refused)?;
        let free = crate::plan::device_room().map(|d| d.free);
        if let Ok(free) = free {
            let held = checked.bytes().held().requested;
            if held > free {
                return Err(refuse(format!(
                    "the session asks {held} bytes of the device, which has {free} free"
                )));
            }
        }
        let logit_scale = model::arch::inkling2::spec::read(split)
            .ok()
            .and_then(|r| r.spec.logit_epilogue)
            .map_or(1.0, |e| e.scale);
        let kernels = Kernels::from_library(library)?;
        let mut stream = device.new_stream()?;
        stream.enable_residency(device.metal())?;
        let residency = stream
            .residency()
            .ok_or_else(|| refuse("the stream made no residency set"))?;
        let staged = residency.stage();
        let alloc = Alloc::new(device.handle(), &staged);
        let weights = file
            .weights()
            .map_err(|e| refuse(format!("the weights: {e}")))?;
        let arena = Arena::open(alloc, split, &weights, &checked)?;
        staged.commit();
        Ok(Session {
            stream,
            kernels,
            logits: vec![0.0; checked.cut],
            checked,
            arena,
            pos: 0,
            poisoned: None,
            logits_fresh: false,
            logit_scale,
        })
    }

    /// The next position a step runs at.
    #[must_use]
    pub fn pos(&self) -> u32 {
        self.pos
    }

    /// The cache's rows.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.checked.ctx
    }

    /// The vocabulary rows the head picks over: the file's cut.
    #[must_use]
    pub fn vocab_cut(&self) -> usize {
        self.checked.cut
    }

    /// The checked file the session runs.
    #[must_use]
    pub fn checked(&self) -> &Checked {
        &self.checked
    }

    /// The bytes the open's buffers were planned at.
    #[must_use]
    pub fn bytes(&self) -> Bytes {
        self.checked.bytes()
    }

    /// Run `id` at [`Session::pos`] and pick the next id.
    ///
    /// # Errors
    /// A poisoned session, a position past the cache, an id past the
    /// vocabulary, the device's error, or the fault word raised by the step.
    pub fn step(&mut self, id: u32) -> Result<u32, SessionError> {
        self.step_with(id, None, None, None)
    }

    /// [`Session::step`] that hands every visit's nodes to `taps` as the walk
    /// reads them back (each visit's experts in a command buffer of their own).
    ///
    /// # Errors
    /// As [`Session::step`].
    pub fn step_tapped(&mut self, id: u32, taps: TapSink<'_>) -> Result<u32, SessionError> {
        self.step_with(id, Some(taps), None, None)
    }

    /// [`Session::step_tapped`] whose experts run the routes `routes` names
    /// at each visit instead of the router's picks, and whose visits (and
    /// head) each start from the rows `inputs` gives, where it is given.
    ///
    /// # Errors
    /// As [`Session::step`], and a route or rows the hooks refuse, a route
    /// that names an expert past the file's, or rows of another length.
    pub fn step_replayed(
        &mut self,
        id: u32,
        taps: TapSink<'_>,
        routes: Routes<'_>,
        inputs: Option<Inputs<'_>>,
    ) -> Result<u32, SessionError> {
        self.step_with(id, Some(taps), Some(routes), inputs)
    }

    fn step_with(
        &mut self,
        id: u32,
        taps: Option<TapSink<'_>>,
        routes: Option<Routes<'_>>,
        inputs: Option<Inputs<'_>>,
    ) -> Result<u32, SessionError> {
        if let Some(f) = self.poisoned {
            return Err(SessionError::Poisoned(f));
        }
        if self.pos as usize >= self.checked.ctx {
            return Err(refuse(format!(
                "a step at position {} of a cache of {} rows",
                self.pos, self.checked.ctx
            )));
        }
        if id as usize >= self.checked.vocab {
            return Err(refuse(format!(
                "id {id} is past the vocabulary of {}",
                self.checked.vocab
            )));
        }
        self.logits_fresh = false;
        let mut walk = Walk {
            c: &self.checked,
            a: &mut self.arena,
            k: &self.kernels,
            stream: &self.stream,
        };
        let picked = walk.token(id, self.pos, taps, routes, inputs);
        // A raise outranks the walk's own error: the fault names the visit.
        let fault = self.arena.faults.read(&self.stream)?;
        if let Some(f) = fault {
            self.poisoned = Some(f);
            return Err(SessionError::Fault(f));
        }
        let picked = picked?;
        self.pos += 1;
        Ok(picked)
    }

    /// The last step's logits over the cut, after the epilogue (the scale and
    /// the soft cap, which leave the pick unmoved).
    ///
    /// # Errors
    /// No step has run since the open, the reset or the cut.
    pub fn logits_row(&mut self) -> Result<&[f32], SessionError> {
        if self.pos == 0 {
            return Err(refuse("the logits of no step"));
        }
        if !self.logits_fresh {
            self.arena
                .scratch
                .logits
                .pages()
                .read_at(Elems(0), &mut self.logits)?;
            let cap = self.checked.softcap;
            let scale = self.logit_scale;
            for x in &mut self.logits {
                *x *= scale;
                if cap > 0.0 {
                    *x = (*x / cap).tanh() * cap;
                }
            }
            self.logits_fresh = true;
        }
        Ok(&self.logits)
    }

    /// Empty caches at position 0, the fault word clean.
    ///
    /// # Errors
    /// The device's error.
    pub fn reset(&mut self) -> Result<(), SessionError> {
        self.arena.faults.clear(&self.stream)?;
        self.poisoned = None;
        self.pos = 0;
        self.logits_fresh = false;
        Ok(())
    }

    /// The positions a cut to `n` keeps: every visit's conv ring holds only
    /// the last inputs, so a cut keeps the whole sequence or nothing.
    #[must_use]
    pub fn keepable(&self, n: u32) -> u32 {
        if n >= self.pos { self.pos } else { 0 }
    }

    /// Take back the positions from `n` on: refused by name unless
    /// [`Session::keepable`] grants `n` whole.
    ///
    /// # Errors
    /// A cut inside the sequence.
    pub fn cut(&mut self, n: u32) -> Result<(), SessionError> {
        match n {
            0 => self.reset(),
            n if n == self.pos => Ok(()),
            n => Err(refuse(format!(
                "a cut to {n} of {} positions: the conv rings keep no earlier point",
                self.pos
            ))),
        }
    }
}
