//! [`AnyEngine`] — where a GPU driver turns a file into an engine. The file's
//! architecture is detected from its metadata before any weight is read, and
//! each arm holds one architecture's `GpuModel`. A driver that needs only the
//! [`Engine`] surface stays generic over it; one that needs more of the model
//! (step mode, probes, graph capture, an architecture's taps) takes its arm.
//!
//! The arms are the `bloomery-gpu` bodies. A deepseek41 file is refused by
//! name before anything is loaded: its engine lives in the V4.1 crate, which
//! this library never names (a crate named here is linked, whole device
//! bundle and all, into every binary of the crate), and its drivers
//! (`generate_ds41`, the V4.1 gates) open it themselves.

#![cfg(feature = "gpu")]

use bloomery_gpu::model::Engine;
use bloomery_gpu::{Deepseek2Model, Gpu, GpuError, Qwen3moeModel};
use gguf::Split;
use model::arch::{self, Arch, coverage, models};
use model::placement::PlacementError;

/// The engine for whichever architecture a file declares, one arm per
/// architecture this build can run. An architecture the model crate knows but
/// this enum has no arm for is [`GpuError::UnsupportedArch`] from
/// [`AnyEngine::open`]; adding an arm is what makes every `match` and
/// destructuring `let` over this enum name the new case.
pub enum AnyEngine {
    Deepseek2(Deepseek2Model),
    Qwen3moe(Qwen3moeModel),
}

impl AnyEngine {
    /// Detect `file`'s architecture, then load that architecture's whole
    /// chain and output head with a `ctx`-row cache: deepseek2 through its
    /// `open` (every weight on the card, or the hybrid load its levers ask
    /// for); any other architecture by its description
    /// ([`model::arch::spec`]) — qwen3moe through its `open` (the whole model
    /// on one card, with the ubatch size and the flash pass its levers name,
    /// read here) once the coverage check finds nothing it cannot run.
    /// A deepseek41 file is [`GpuError::UnsupportedArch`] here, before any
    /// load; a file whose description has parts no program runs is refused
    /// with every such part listed. deepseek2 has no description reader and
    /// dispatches on the architecture string. The engine takes the file.
    pub fn open(file: Split, ctx: usize) -> Result<AnyEngine, GpuError> {
        const WHAT: &str = "AnyEngine::open";
        let head = file.shard(0).ok_or(GpuError::State {
            what: WHAT,
            missing: "the file's first shard",
        })?;
        match Arch::detect(head) {
            Ok(Arch::Deepseek2) => {
                return Ok(AnyEngine::Deepseek2(Deepseek2Model::open(file, ctx)?));
            }
            Ok(a @ Arch::Deepseek41) => {
                return Err(GpuError::UnsupportedArch(a.name().to_string()));
            }
            Ok(Arch::Qwen3moe | Arch::Qwen35moe) | Err(_) => {}
        }
        let read = arch::spec(&file)?;
        let missing = coverage::check(&read.spec, &read.tensors);
        if !missing.is_empty() {
            return Err(GpuError::plan(WHAT, PlacementError::Unimplemented(missing)));
        }
        match read.spec.arch {
            models::Arch::Qwen3Moe => Ok(AnyEngine::Qwen3moe(Qwen3moeModel::open(
                Gpu::new()?,
                file,
                Qwen3moeModel::lever_opts(ctx)?,
            )?)),
            a => Err(GpuError::UnsupportedArch(a.name().to_string())),
        }
    }
}

impl Engine for AnyEngine {
    fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        match self {
            AnyEngine::Deepseek2(m) => m.step(tokens),
            AnyEngine::Qwen3moe(m) => m.step(tokens),
        }
    }

    fn reset(&mut self) -> Result<(), GpuError> {
        match self {
            AnyEngine::Deepseek2(m) => m.reset(),
            AnyEngine::Qwen3moe(m) => m.reset(),
        }
    }

    fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError> {
        match self {
            AnyEngine::Deepseek2(m) => m.seed_depth(rows),
            AnyEngine::Qwen3moe(m) => m.seed_depth(rows),
        }
    }

    fn pos(&self) -> u32 {
        match self {
            AnyEngine::Deepseek2(m) => m.pos(),
            AnyEngine::Qwen3moe(m) => m.pos(),
        }
    }

    fn resident_bytes(&self) -> usize {
        match self {
            AnyEngine::Deepseek2(m) => m.resident_bytes(),
            AnyEngine::Qwen3moe(m) => m.resident_bytes(),
        }
    }

    fn arch(&self) -> Arch {
        match self {
            AnyEngine::Deepseek2(m) => m.arch(),
            AnyEngine::Qwen3moe(m) => m.arch(),
        }
    }
}
