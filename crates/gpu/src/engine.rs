//! [`AnyEngine`] — where a GPU driver turns a file into an engine. The file's
//! architecture is detected from its metadata before any weight is read, and
//! each arm holds one architecture's [`crate::GpuModel`]. A driver that needs
//! only the [`Engine`] surface stays generic over it; one that needs more of
//! the model (step mode, probes, graph capture, an architecture's taps) takes
//! its arm.

use crate::model::Engine;
use crate::{Deepseek2Model, GpuError};
use ::model::arch::Arch;
use gguf::Gguf;

/// The engine for whichever architecture a file declares, one arm per
/// architecture this crate can run. An architecture the model crate knows but
/// this enum has no arm for is [`GpuError::UnsupportedArch`] from
/// [`AnyEngine::open`]; adding an arm is what makes every `match` and
/// destructuring `let` over this enum name the new case.
pub enum AnyEngine {
    Deepseek2(Deepseek2Model),
}

impl AnyEngine {
    /// Detect `gguf`'s architecture, then load that architecture's whole
    /// chain and output head with a `ctx`-row cache.
    pub fn open(gguf: &Gguf, ctx: usize) -> Result<AnyEngine, GpuError> {
        match Arch::detect(gguf)? {
            Arch::Deepseek2 => Ok(AnyEngine::Deepseek2(Deepseek2Model::load_full(gguf, ctx)?)),
            a @ Arch::Deepseek41 => Err(GpuError::UnsupportedArch(a.name().to_string())),
        }
    }
}

impl Engine for AnyEngine {
    fn step(&mut self, tokens: &[u32]) -> Result<u32, GpuError> {
        match self {
            AnyEngine::Deepseek2(m) => m.step(tokens),
        }
    }

    fn reset(&mut self) -> Result<(), GpuError> {
        match self {
            AnyEngine::Deepseek2(m) => m.reset(),
        }
    }

    fn seed_depth(&mut self, rows: usize) -> Result<(), GpuError> {
        match self {
            AnyEngine::Deepseek2(m) => m.seed_depth(rows),
        }
    }

    fn pos(&self) -> u32 {
        match self {
            AnyEngine::Deepseek2(m) => m.pos(),
        }
    }

    fn resident_bytes(&self) -> usize {
        match self {
            AnyEngine::Deepseek2(m) => m.resident_bytes(),
        }
    }

    fn arch(&self) -> Arch {
        match self {
            AnyEngine::Deepseek2(m) => m.arch(),
        }
    }
}
