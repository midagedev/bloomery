//! An ERNIE 5.5 file planned onto a machine: the common hybrid owner's plan ([`place_hybrid`]) over
//! this family's reading ([`PlanInputs`]). Each layer's stores are read off the description
//! ([`KvLayout::of`]):
//! - an HCA layer keeps a ring of its last `window` rows of its one 512-value key-value head (f16),
//!   one pooled row of that head every `ratio` positions (f16, every one attended each step) and
//!   its compressor's pooling state, the last `ratio` projections' values and scores (f32);
//! - a latent layer keeps one row a position of the absorbed latent and its 64 rotated key values
//!   (f16, a step selecting `top_k` of them) and one indexer key a position (f32: the lightning
//!   indexer's score pass reads full-width keys, [`KvLayout::with_f32_index_keys`]; every one
//!   scored each step).
//!
//! None is a room: the parts' caches are whole buffers sized from the context at the load, so the
//! plan counts every store at the context, and a K/V floor or host backing is refused by name.
//!
//! The plan serves the placed load (`crates/gpu-ernie55`): every layer's dense part and the stores
//! on the stage card, each routed layer's card share of its experts by the common card leg's rule
//! (the routed rule of `Program::Ernie55Body`) and the rest on the host tier. The body's host run is
//! the common one, so the plan pages the host experts the room cannot hold
//! ([`PlanInputs::with_nvme_tier`]). What the file asks beyond the description ([`Ops`]) is refused
//! by name ([`PlanInputs::read`]) until a program of this family runs it.

use gguf::Split;
use models::Unimplemented;

use super::hparams::Hparams;
use super::ops::Ops;
use super::spec;
use crate::arch::place_hybrid::{self, route_bytes, wide_bytes};
use crate::placement::PlacementError;
use crate::placement::kernels::{self, DownAct, DownForms};

pub use place_hybrid::{KvLayout, PlaceError};

/// What a plan of an ERNIE 5.5 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "ERNIE 5.5 Flash";

/// The rows of one prompt unit: every part's launches of the pass take up to this many, and the
/// body's pass is built for it (`bloomery_gpu_ernie55::PASS_ROWS`, held equal at the load).
pub const PROMPT_ROWS: usize = 128;

/// The columns of a latent layer's chain in one span (the by-name gemv's columns): the chain runs
/// a unit's rows in spans of this many, and its scratch is made for one span.
pub const MLA_UNIT: usize = 8;

/// The keys one flash segment covers: a launch over `n` keys is `n.div_ceil(FLASH_SEG_KEYS)`
/// segments, each a partial of the head's `Σ exp·V` row and a `(max, Σ exp)` pair.
pub const FLASH_SEG_KEYS: usize = 64;

/// The segments the split flash cuts a row into over a cache of more than one segment's keys,
/// whatever the height (`bloomery_gpu::flash::segments_for`).
pub const FLASH_SEGMENTS: usize = 80;

/// The columns of one chunk of the HCA layers' grouped output projection.
pub const HCA_COLS: usize = 8;

/// The values of one piece of the hyper-connections' pre, and the tokens a launch of it takes.
pub const HC_PIECE: usize = 512;
pub const HC_MAX_TOKENS: usize = 8;

/// The score pass's histogram bins a column.
pub const HIST_BINS: usize = 1024;

/// The bytes of a four-byte word.
const WORD: usize = 4;

/// What the load's buffers beside the stores hold, by part, for prompt units of [`PROMPT_ROWS`]
/// rows ([`PassTerms::of`]): the parts' buffers are made for the unit, and the step runs in the
/// first row of the same ones, so these are every non-store byte the parts make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassTerms {
    /// The walk's streams and mixes, the dense lead's blocks and the pre's partials.
    pub trunk: u64,
    /// The HCA layers' chain, staging and words, and the flash partials of the ring's segments;
    /// the partials of the pooled rows' segments are [`hca_partials_block`]'s.
    pub hca: u64,
    /// The latent layers' chain over one span ([`MLA_UNIT`] columns), its flash partials, the
    /// indexer's histogram and list, and the span's words; the score rows a position adds are
    /// [`mla_scores_row`]'s.
    pub mla: u64,
    /// The routed block's rows, the card leg's buffers and the sites' forms.
    pub moe: u64,
    /// The head's one row of folded streams.
    pub head: u64,
}

impl PassTerms {
    /// The terms of `hp`, in bytes, from the widths the parts build their buffers from (each
    /// term's lines in the body's allocation: `Trunk`, `HcaState::new`, `Scratch::new` of the
    /// latent chain, `MoeBufs::new`). The terms, in four-byte words a row unless said:
    ///
    /// - trunk: the two stream buffers (`2 · hc · embd`), the sub-layers' mixes and the pre's
    ///   rows (`2 · mix`), the fold, the norm and the post's fold (`3 · embd`), the dense lead's
    ///   block (`3 · dense_ff` each, with its route table and the forms of its two widths), and
    ///   the grouped pre's partials, at one group a row (a bound);
    /// - HCA: the normed input, the query's two latents, the query rows three times over (the
    ///   rows, their norm and rope, the output), the output's gate and two grouped rows
    ///   (`3 · grouped`), six latent-wide rows (the latent, the pool's projections, the table, the
    ///   scores and the latent after its norm), a column chunk's gathered input and group
    ///   outputs, the staging (f16), the words and rope rows of the unit, the pool's pre-rope rows
    ///   and the append's shadow row, and the flash partials of the ring's segments;
    /// - latent: per column of a span, the query's two latents, its rows twice, its cache row
    ///   over every head, the key and value rows, the indexer's heads and key and weights, the
    ///   absorbed output and the value rows four times, the flash partials over the list's width
    ///   (`top_k` and the window: one segment when they fit one, else [`FLASH_SEGMENTS`]), the
    ///   histogram, the list and the count, the span's words and rope rows;
    /// - routed block: the router's scores and logits, the picks' ids and weights, four latent
    ///   rows, the up's output and the shared expert's output, its gate, up and activation,
    ///   the card leg's buffers ([`kernels::leg_bytes`], both down forms: a bound), the forms of
    ///   the three widths a site reads (both where the width allows them) and a one-expert route;
    /// - head: the streams and their fold, one row (a bound).
    #[must_use]
    pub fn of(hp: &Hparams) -> PassTerms {
        let r = PROMPT_ROWS;
        let (h, streams, mix) = (hp.n_embd, hp.hc.streams, hp.hc.mix());
        let k = streams * h;
        let pieces = k / HC_PIECE;
        let dense_ff = hp.dense_ff;
        let trunk_words = r * (2 * k + 2 * mix + 3 * h)
            + hp.dense_lead * 3 * r * dense_ff
            + r * (mix * pieces * HC_MAX_TOKENS + pieces * HC_MAX_TOKENS + 1);
        let trunk_bytes = WORD * trunk_words
            + hp.dense_lead * (route_bytes(r) + wide_bytes(r, dense_ff))
            + wide_bytes(r, h);

        let hca = &hp.hca;
        let (lat, rope) = (hca.head, hp.mla.pe);
        let (q_lora, q_width, grouped) = (hp.mla.q_lora, hp.heads * lat, hca.grouped());
        let group_k = q_width / hca.o_groups.max(1);
        let groups = r.div_ceil(hca.ratio);
        let step_words = 2 + groups * (1 + hca.ratio) + 2 * hca.ratio;
        let hca_words = r * (h + 2 * q_lora + 3 * q_width + 3 * grouped + 6 * lat)
            + HCA_COLS * (group_k + hca.o_lora)
            + lat
            + 2 * r * rope
            + groups * (rope + lat)
            + hca.window.div_ceil(FLASH_SEG_KEYS) * r * hp.heads * (lat + 2);
        let hca_bytes = WORD * (hca_words + 3 * r + step_words) + 2 * (r + 1) * lat;

        let u = MLA_UNIT;
        let m = &hp.mla;
        let (idx, stride) = (&hp.indexer, hp.indexer.top_k + hca.window);
        let row = m.pe + m.kv_lora;
        let segs = if stride <= FLASH_SEG_KEYS {
            1
        } else {
            FLASH_SEGMENTS
        };
        let mla_words = 2 * u * q_lora
            + 2 * u * hp.heads * m.head_k
            + u * hp.heads * row
            + 3 * u * row
            + u * idx.heads * idx.d
            + u * idx.d
            + u * idx.heads
            + u * hp.heads * segs * (m.kv_lora + 2)
            + u * hp.heads * m.kv_lora
            + 4 * u * hp.heads * m.head_v
            + u * m.pe
            + u * HIST_BINS
            + u * stride
            + 4 * u
            + 1;

        let e = &hp.experts;
        let shared_ff = e.shared_ff();
        let moe_words = r * (2 * e.n_expert + 2 * e.n_used + 4 * hp.latent + 2 * h + 3 * shared_ff)
            + 3 * e.n_expert
            + 2 * e.n_used;
        let leg = kernels::leg_bytes(
            r as u64,
            e.n_used as u64,
            hp.latent as u64,
            e.ff as u64,
            DownForms::of(DownAct::Q8Act).with(DownAct::Q8Blocks32),
        ) as usize;
        let moe_bytes = WORD * moe_words
            + leg
            + wide_bytes(r, h)
            + wide_bytes(r, hp.latent)
            + wide_bytes(r, shared_ff)
            + route_bytes(r);

        PassTerms {
            trunk: trunk_bytes as u64,
            hca: hca_bytes as u64,
            mla: (WORD * mla_words) as u64,
            moe: moe_bytes as u64,
            head: (WORD * (k + h)) as u64,
        }
    }

    /// Every term: what the plan reserves ([`prompt_reserve_bytes`]).
    #[must_use]
    pub fn total(&self) -> u64 {
        self.trunk + self.hca + self.mla + self.moe + self.head
    }
}

/// The card bytes of the load's buffers beside the stores that do not grow with the context: the
/// plan's prompt reserve ([`PlanInputs::with_prompt_reserve`]), which the body's made bytes are
/// held to at the load. The context-sized ones are the stage's position rows
/// ([`hca_partials_block`], [`mla_scores_row`]).
#[must_use]
pub fn prompt_reserve_bytes(hp: &Hparams) -> u64 {
    PassTerms::of(hp).total()
}

/// The HCA layers' flash partials for each block of positions of the context, the last block
/// counted whole ([`PlanInputs::with_block_scratch`]): bytes and positions. The pooled rows'
/// segments grow with the context — one segment of [`FLASH_SEG_KEYS`] pooled rows is
/// `ratio · FLASH_SEG_KEYS` positions — and each segment of a unit's rows over every head is a
/// `Σ exp·V` row and a `(max, Σ exp)` pair.
#[must_use]
pub fn hca_partials_block(hp: &Hparams) -> (u64, u64) {
    let words = PROMPT_ROWS * hp.heads * (hp.hca.head + 2);
    (
        (WORD * words) as u64,
        (hp.hca.ratio * FLASH_SEG_KEYS) as u64,
    )
}

/// The latent layers' score rows for each position of the context, in bytes
/// ([`PlanInputs::with_position_scratch`]): one f32 a column of a span.
#[must_use]
pub fn mla_scores_row() -> u64 {
    (WORD * MLA_UNIT) as u64
}

impl PlanInputs {
    /// The file as it is, whatever the coverage list holds, and the host's room. The roles and the
    /// description are the registry's reader's ([`spec::read`]), so the plan and the description
    /// of a file are one reading.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let read = spec::read(split)?;
        let reserve = prompt_reserve_bytes(&hp);
        let (partials, block) = hca_partials_block(&hp);
        Ok(PlanInputs::new(ENGINE, hp, read.tensors, read.spec)?
            .with_f32_index_keys()
            .with_prompt_reserve(reserve)
            .with_position_scratch(mla_scores_row())
            .with_block_scratch(partials, block)
            .with_nvme_tier())
    }

    /// [`PlanInputs::describe`], then the file refused with every feature the engine does not run
    /// ([`PlacementError::Unimplemented`], [`PlanInputs::unrun`]).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        let missing = inputs.unrun()?;
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// Every feature of the file the engine does not run: the coverage check's items and the ops
    /// the description cannot state ([`Ops::unrun`]). Empty for a file a program runs.
    pub fn unrun(&self) -> Result<Vec<Unimplemented>, PlacementError> {
        let mut items = self.unimplemented();
        items.extend(Ops::of(&self.hp)?.unrun());
        Ok(items)
    }

    /// A context of `ctx` positions: at least one and at most the trained context
    /// (`context_length`), refused by name past either.
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
}
