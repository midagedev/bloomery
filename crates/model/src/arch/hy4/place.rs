//! A Hy4-flash file planned onto a machine: the common hybrid owner's plan ([`place_hybrid`]) over
//! this family's reading ([`PlanInputs`]), and what that plan sizes beyond the file's tensors: the
//! by-position rooms each layer holds ([`Hparams::pos_bufs`]) and the window layers' ring
//! ([`Hparams::ring_bytes`]). A sparse-attention layer caches one latent row a position (the 512
//! latent values and the 64 rotated key values, f16), attends the `top_k` rows its indexer picks,
//! and holds one pooled index key (`d` values, f16) and its score every `kpool` positions, scored
//! whole every step; a window layer holds the last `window` rows in a ring whatever the context.
//! The cached rows are rooms, backed as far as the sequence reaches and, being selected, host-able;
//! the pooled keys and scores are plain buffers the plan counts at the context's cap: a buffer the
//! step scans whole is card-only, and nothing a plan with no residency machine holds could pay for
//! its growth past a floor. The next-token block holds none: it is not loaded.
//!
//! The plan serves the placed load (`crates/gpu-hy4`): the dense trunk, the head and each routed
//! layer's card share of experts (the id prefix the budget holds) on the stage card, the rest on
//! the host tier. A load that runs its prompt as the placed pass reserves the pass's card bytes
//! ([`PlanInputs::with_prompt_reserve`]); one that steps its prompt reserves none. The plan hangs no
//! expert tier card; the engram table and the token embedding stay in the file, a token's rows read
//! on the host.

use gguf::Split;
use runtime::stores::{Access, F16_BYTES, PosBuf, PosLayout};

use super::hparams::Hparams;
use super::spec;
use crate::arch::place_hybrid;
use crate::placement::{PlacementError, kernels};

pub use place_hybrid::{KvLayout, PlaceError};

/// What a plan of a Hy4-flash file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "Hy4-flash";

impl PlanInputs {
    /// [`PlanInputs::describe`], then the file refused with every feature the engine does not run
    /// ([`PlacementError::Unimplemented`]).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// A context of `ctx` positions as the body serves it: at least one position and at most the
    /// trained context (`context_length`), refused by name past either.
    pub fn check_ctx(&self, ctx: u64) -> Result<(), PlacementError> {
        let train = self.hp.n_ctx_train as u64;
        if ctx == 0 || ctx > train {
            return Err(PlacementError::Metadata {
                key: "context_length".to_string(),
                detail: format!("a context of {ctx} positions; the file trains {train}"),
            });
        }
        Ok(())
    }

    /// The file as it is, whatever the coverage list holds, and the host's room. The roles and the
    /// description are the registry's reader's ([`spec::read`]), so the plan and the description
    /// of a file are one reading. Refused by name, before any load: a dense site whose q8_1 input
    /// column is wider than the card's activation holds ([`kernels::check_q8_act_width`]).
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let read = spec::read(split)?;
        read.tensors
            .tensors
            .iter()
            .try_for_each(kernels::check_q8_act_width)?;
        let rooms = hp.pos_layout();
        Ok(PlanInputs::new(ENGINE, hp, read.tensors, read.spec)?.with_rooms(rooms))
    }
}

/// The name of a sparse-attention layer's cached rows in a room's refusal and the plan's lines.
pub const CACHE_ROWS: &str = "a Hy4 sparse layer's cached rows";

impl Hparams {
    /// The cached row's bytes: the latent and the rotated key values, f16.
    #[must_use]
    pub fn cache_row_bytes(&self) -> u64 {
        (self.mla.kv_lora + self.mla.pe) as u64 * F16_BYTES
    }

    /// The by-position buffers trunk layer `l` holds in rooms, the one declaration the plan's
    /// floor and padding terms and the card body's room constructor read: a sparse layer's cached
    /// rows, selected `top_k` of them a step. A window layer's ring is fixed
    /// ([`Hparams::ring_bytes`]) and the pooled keys are a plain buffer (the module doc), so
    /// neither is declared; the next-token block holds none.
    #[must_use]
    pub fn pos_bufs(&self, l: usize) -> Vec<PosBuf> {
        if self.is_nextn(l) || !self.dsa(l) {
            return Vec::new();
        }
        vec![PosBuf {
            what: CACHE_ROWS,
            row: self.cache_row_bytes(),
            per: 1,
            runs: 1,
            access: Access::Select {
                rows: self.indexer.top_k as u64,
            },
        }]
    }

    /// Every trunk layer's [`Hparams::pos_bufs`], one sequence's.
    #[must_use]
    pub fn pos_layout(&self) -> PosLayout {
        PosLayout::new((0..self.n_layer).map(|l| self.pos_bufs(l)).collect())
    }

    /// The ring trunk layer `l` holds, in bytes: `window` cached rows for a window layer, none for
    /// a sparse-attention layer or the next-token block.
    #[must_use]
    pub fn ring_bytes(&self, l: usize) -> u64 {
        if self.is_nextn(l) || self.dsa(l) {
            return 0;
        }
        self.window as u64 * self.cache_row_bytes()
    }
}
