//! [`Generator`] — the load-and-step loop a decode CLI drives: open the file
//! by a placement, load it, capture the step graph before the prompt, feed
//! the prompt one real step per token, then one step per chosen token. It
//! owns the [`GpuModel`] and nothing else: no sampling, no text.
//!
//! Generic over the chain body. The body's own entry (`body::open` of the
//! V4.1 device crate) and whatever the caller checks on the loaded body come
//! in as arguments, because this library's source does not name a device
//! crate: named here, it would be linked, device bundle and all, into every
//! gate built with the feature, the ones that launch none of its kernels
//! too (the reason [`crate::ds41_meta::RopeMeta::read`] takes its ropes as
//! functions). The plan line is the caller's for the same reason: the plan's
//! inputs are the architecture's type.

use std::io::Write;
use std::time::Instant;

use bloomery_gpu::model::{ChainBody, StepMode};
use bloomery_gpu::{GpuError, GpuModel};
use gguf::Split;
use model::placement::{Machine, workstation};

use crate::{GateError, ref_model_path};

/// Which placement the engine loads by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// The serving plan (`workstation::plan_a`), on the A6000.
    A,
    /// The step gate's plan (`workstation::plan_gate`), on the 3090.
    Gate,
}

impl Place {
    /// The flag value's placement: `a` or `gate`.
    pub fn parse(v: &str) -> Result<Place, GateError> {
        match v {
            "a" => Ok(Place::A),
            "gate" => Ok(Place::Gate),
            other => Err(format!("--place is a or gate, not {other}").into()),
        }
    }

    /// The machine the placement plans over, by layer count.
    pub fn machine(self) -> fn(usize) -> Machine {
        match self {
            Place::A => workstation::plan_a,
            Place::Gate => workstation::plan_gate,
        }
    }

    /// The flag value, as the `plan` and `load` lines print it.
    pub fn name(self) -> &'static str {
        match self {
            Place::A => "a",
            Place::Gate => "gate",
        }
    }
}

/// How [`Generator::open`] loads: the placement, the context the caches are
/// sized for, the step mode, and whether the calling thread pins itself to
/// the dispatcher's cpu slot (`BLOOMERY_PIN_MAIN` is the caller's to read).
#[derive(Debug, Clone, Copy)]
pub struct OpenArgs {
    pub place: Place,
    pub ctx: usize,
    pub mode: StepMode,
    pub pin_main: bool,
}

/// The body's loader: the file, the placement's machine and the context.
pub type BodyOpen<B> = fn(Split, fn(usize) -> Machine, usize) -> Result<GpuModel<B>, GpuError>;

/// A loaded model standing at a position, and the context it may reach.
pub struct Generator<B: ChainBody> {
    model: GpuModel<B>,
    ctx: usize,
}

impl<B: ChainBody> Generator<B> {
    /// Open `$BLOOMERY_REF_MODEL` by `args.place` through `open`, run `check`
    /// on the loaded model (it refuses a body that cannot run this file, and
    /// returns the fields it adds to the `load` line), then write the `load`
    /// line, the host set's lines of a placed load, and in graph mode capture
    /// the step before any token (the `capture` line) — all to `log`.
    pub fn open<C>(
        args: OpenArgs,
        open: BodyOpen<B>,
        check: C,
        log: &mut dyn Write,
    ) -> Result<Generator<B>, GateError>
    where
        C: FnOnce(&GpuModel<B>) -> Result<String, GateError>,
    {
        let pinned = args.pin_main && threads::pool().pin_caller();
        let path = ref_model_path()?;
        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut model = open(file, args.place.machine(), args.ctx)?;
        model.set_mode(args.mode);
        let fields = check(&model)?;
        writeln!(
            log,
            "load resident_bytes={} ctx={} {fields} mode={} place={} pin_main={} \
             pinned={pinned} in {:.1} s (runtime value)",
            model.resident_bytes(),
            args.ctx,
            mode_name(args.mode),
            args.place.name(),
            if args.pin_main { "on" } else { "off" },
            t.elapsed().as_secs_f64()
        )?;
        if let Some(h) = model.host_residency() {
            match h.populated() {
                Some(w) => writeln!(
                    log,
                    "host_populate={} in {:.1} s (runtime value)",
                    w.bytes(),
                    w.wall().as_secs_f64()
                )?,
                None => writeln!(log, "host_populate=off")?,
            }
            if let Some(l) = h.lock() {
                writeln!(log, "host_lock={} B", l.bytes())?;
            }
        }
        if args.mode == StepMode::Graph {
            // Captured before the prompt, so no token's step pays for it.
            writeln!(log, "capture graph_nodes={}", model.capture_step()?)?;
        }
        Ok(Generator {
            model,
            ctx: args.ctx,
        })
    }

    /// Feed `ids` from the current position, one real step per id, and
    /// return the argmax after the last. Refused before any step when the
    /// ids do not fit the context.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        if ids.is_empty() {
            return Err("prefill: no ids to feed".into());
        }
        let end = usize::try_from(self.model.pos())? + ids.len();
        if end > self.ctx {
            return Err(format!(
                "prefill: position {} + {} ids exceed the context {}",
                self.model.pos(),
                ids.len(),
                self.ctx
            )
            .into());
        }
        Ok(self.model.step(ids)?)
    }

    /// One step on `tok` at the current position; the argmax after it.
    pub fn step(&mut self, tok: u32) -> Result<u32, GateError> {
        if usize::try_from(self.model.pos())? >= self.ctx {
            return Err(format!("step: the context {} is full", self.ctx).into());
        }
        Ok(self.model.step(&[tok])?)
    }

    /// The head's logits after the last step (`n_vocab` f32). A blocking
    /// read of the whole row, on top of the step's own argmax readback.
    pub fn logits(&self) -> Result<Vec<f32>, GateError> {
        Ok(self.model.logits()?)
    }

    /// The position the next fed token lands in.
    pub fn pos(&self) -> usize {
        // u32 widens into usize on the 64-bit hosts this runs on.
        self.model.pos() as usize
    }

    /// The context the caches were sized for: `pos` never passes it.
    pub fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The model, for what the loop does not own (the pair pass, probes).
    pub fn model(&self) -> &GpuModel<B> {
        &self.model
    }

    /// See [`Generator::model`].
    pub fn model_mut(&mut self) -> &mut GpuModel<B> {
        &mut self.model
    }
}

/// The step mode as the `load` line prints it.
pub fn mode_name(mode: StepMode) -> &'static str {
    if mode == StepMode::Graph {
        "graph"
    } else {
        "eager"
    }
}
