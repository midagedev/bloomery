//! [`AnyEngine`] — where a GPU driver turns a file into an engine. The file's
//! architecture is detected from its metadata before any weight is read, and
//! each arm holds one architecture's `GpuModel`. A driver that needs only the
//! [`Engine`] surface stays generic over it; one that needs more of the model
//! (step mode, probes, graph capture, an architecture's taps) takes its arm.
//!
//! The enum lives in this crate because it is the one that sees both bodies:
//! the deepseek41 body is in `bloomery-gpu-deepseek41`, which depends on
//! `bloomery-gpu`, so `bloomery-gpu` cannot name it. The deepseek41 arm exists
//! with this crate's `deepseek41` feature only, so a build without it never
//! compiles V4.1 device code.

#![cfg(feature = "gpu")]

use bloomery_gpu::model::Engine;
use bloomery_gpu::{Deepseek2Model, GpuError};
use gguf::Split;
use model::arch::Arch;

/// The engine for whichever architecture a file declares, one arm per
/// architecture this build can run. An architecture the model crate knows but
/// this enum has no arm for is [`GpuError::UnsupportedArch`] from
/// [`AnyEngine::open`]; adding an arm is what makes every `match` and
/// destructuring `let` over this enum name the new case.
pub enum AnyEngine {
    Deepseek2(Deepseek2Model),
    #[cfg(feature = "deepseek41")]
    Deepseek41(bloomery_gpu_deepseek41::body::Deepseek41Model),
}

impl AnyEngine {
    /// Detect `file`'s architecture, then load that architecture's whole
    /// chain and output head with a `ctx`-row cache: deepseek2 through
    /// `load_full` (every weight on the card, or the hybrid load its levers
    /// ask for), deepseek41 by this workstation's serving placement (design
    /// §5 (a), `workstation::plan_a`). The engine takes the file.
    pub fn open(file: Split, ctx: usize) -> Result<AnyEngine, GpuError> {
        let head = file.shard(0).ok_or(GpuError::State {
            what: "AnyEngine::open",
            missing: "the file's first shard",
        })?;
        match Arch::detect(head)? {
            Arch::Deepseek2 => Ok(AnyEngine::Deepseek2(Deepseek2Model::load_full(file, ctx)?)),
            #[cfg(feature = "deepseek41")]
            Arch::Deepseek41 => Ok(AnyEngine::Deepseek41(bloomery_gpu_deepseek41::body::open(
                file,
                model::placement::workstation::plan_a,
                ctx,
            )?)),
            #[cfg(not(feature = "deepseek41"))]
            a @ Arch::Deepseek41 => Err(GpuError::UnsupportedArch(a.name().to_string())),
            a @ Arch::Qwen3moe => Err(GpuError::UnsupportedArch(a.name().to_string())),
        }
    }
}

impl Engine for AnyEngine {
    fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        match self {
            AnyEngine::Deepseek2(m) => m.step(tokens),
            #[cfg(feature = "deepseek41")]
            AnyEngine::Deepseek41(m) => m.step(tokens),
        }
    }

    fn reset(&mut self) -> Result<(), GpuError> {
        match self {
            AnyEngine::Deepseek2(m) => m.reset(),
            #[cfg(feature = "deepseek41")]
            AnyEngine::Deepseek41(m) => m.reset(),
        }
    }

    fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError> {
        match self {
            AnyEngine::Deepseek2(m) => m.seed_depth(rows),
            #[cfg(feature = "deepseek41")]
            AnyEngine::Deepseek41(m) => m.seed_depth(rows),
        }
    }

    fn pos(&self) -> u32 {
        match self {
            AnyEngine::Deepseek2(m) => m.pos(),
            #[cfg(feature = "deepseek41")]
            AnyEngine::Deepseek41(m) => m.pos(),
        }
    }

    fn resident_bytes(&self) -> usize {
        match self {
            AnyEngine::Deepseek2(m) => m.resident_bytes(),
            #[cfg(feature = "deepseek41")]
            AnyEngine::Deepseek41(m) => m.resident_bytes(),
        }
    }

    fn arch(&self) -> Arch {
        match self {
            AnyEngine::Deepseek2(m) => m.arch(),
            #[cfg(feature = "deepseek41")]
            AnyEngine::Deepseek41(m) => m.arch(),
        }
    }
}
