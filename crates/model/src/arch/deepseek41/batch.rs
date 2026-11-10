//! The V4.1 prompt batch's card buffers as named parts ([`Parts`]): a pure
//! function of the shapes ([`BatchShape`]: the file's hyperparameters, the
//! stage card's layers, the context, the group and the levers that add
//! buffers), which the plan counts as the stage card's prompt term
//! ([`prompt_parts`] at [`BatchShape::planned`], [`Parts::card_bytes`] at the
//! card's granule) and the card body's `make_batch` allocates from where the
//! length is its own (the params, the batch sets, the staging, the
//! attention's batch-wide buffers) and holds every part to
//! ([`Parts::check`], one refusal listing every part apart).
//!
//! The parts, in `make_batch`'s allocation order:
//!
//! | part | what |
//! |---|---|
//! | [`ATTN_WORDS`] | the attention piece's rows of step words, a row per chunk of every batch of a group, and the gather's pairs |
//! | [`ATTN_SELECT`] | the compressor's, the indexer's and the candidate mask's scratch, per token count 1..=[`CHUNK`] |
//! | [`ATTN_SCRATCH`] | the attention piece's chunk scratch: HC_PRE, the projections, the partials, the q8_1 forms |
//! | [`PARAMS`] | the chunk images' card copy, [`CHUNKS_MAX`] a batch |
//! | [`GLUE`] | the engram chunk scratch |
//! | [`FFN_ROWS`] | the MoE sub-layer's rows over [`T_MAX`] tokens and its chunk scratch |
//! | [`FFN_TILES`] | the fused layers' tile and grouped-GEMM scratch |
//! | [`FFN_LEG`] | the common card leg's run of [`LEG_PIECE`] tokens |
//! | [`FFN_PLACES`] | the tier's and the host stream's places |
//! | [`SETS`] | per batch of a group its streams, folds, lists and feature rows |
//! | [`STAGING`] | a chunk's latent rows before their commit |
//! | [`ATTN_BATCH`] | the batch-wide attention projections ([`AttnBatchLens`]) |

use std::ops::Range;

use gguf::GgmlType;
use runtime::prompt::group_sets;

use super::hparams::{CandidateRole, Hparams, LayerKind};
use super::plan::Planner;
use crate::placement::PlacementError;
use crate::placement::accepts::leg_runs;
use crate::placement::accepts::{self, Program};
use crate::placement::kernels::{self, DownForms, q8_act_bytes};
use crate::placement::parts::{Part, Parts};
use crate::placement::workstation::tier_block_scratch_bytes;

/// Positions one batch runs at most: the host union's columns.
pub const T_MAX: usize = crate::moe::UNION_MAX_COLS;
/// Positions one chunk runs at most: the m-column kernels' and HC_PRE's
/// (`HC_MAX_TOKENS`, the card body binds it to this).
pub const CHUNK: usize = 8;
/// Chunks a batch cuts into at most: an unaligned first position adds one.
pub const CHUNKS_MAX: usize = T_MAX / CHUNK + 1;
/// Values one HC_PRE partial spans (`HC_PIECE`).
pub const HC_PIECE: usize = 512;
/// The indexer's heads and score histogram bins (`indexer::HEADS`,
/// `HIST_BINS`); its head width is the file's index key width.
pub const INDEX_HEADS: usize = 32;
pub const INDEX_HIST_BINS: usize = 1024;
/// Keys one attention partial spans (`attn::SEG_KEYS`).
pub const SEG_KEYS: usize = 64;
/// Chunks a sub-block of the batch-wide projections runs at most, and the
/// token counts a sub-block can have.
pub const SUB_CHUNKS: usize = 16;
pub const SUB_TOKENS: usize = SUB_CHUNKS * CHUNK;
pub const SUB_COUNTS: [usize; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128];
const _: () = assert!(SUB_COUNTS[SUB_COUNTS.len() - 1] == SUB_TOKENS);
const _: () = assert!(SUB_COUNTS[CHUNK - 1] == CHUNK);
/// The rope tables a token carries (`Table::ALL`).
pub const TABLES: usize = 4;
/// Tokens one run of the common card leg takes (`LEG_PIECE`).
pub const LEG_PIECE: usize = 64;
/// Words of a token's head, of a stream's header, and the sections'
/// alignment in a step image (`params.rs`).
const TOKEN_WORDS: usize = 4;
const STREAM_HEADER: usize = 4;
const ALIGN: usize = 4;

pub const ATTN_WORDS: &str = "attn words";
pub const ATTN_SELECT: &str = "attn select";
pub const ATTN_SCRATCH: &str = "attn scratch";
pub const PARAMS: &str = "params";
pub const GLUE: &str = "glue";
pub const FFN_ROWS: &str = "ffn rows";
pub const FFN_TILES: &str = "ffn tiles";
pub const FFN_LEG: &str = "ffn leg";
pub const FFN_PLACES: &str = "ffn places";
pub const SETS: &str = "batch sets";
pub const STAGING: &str = "staging";
pub const ATTN_BATCH: &str = "attn batch";

/// The activation forms of `h` a card's MoE layers read beside the f32 rows,
/// from their types: the rows and the batch hold these and no other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FfnReads {
    /// A fused layer ([`accepts::ds41_fused`]): its down reads the card
    /// slots' q8_1.
    pub fused: bool,
    /// A fused layer whose down is Q5_K ([`accepts::DS41_FUSED_DOWNS`]).
    pub fused_q5k_down: bool,
    /// A shared down of a K-quant reads `sh_h` in q8_1.
    pub sh_q8_1: bool,
    /// A shared gate and up of types read over the f32 row write rows of
    /// their own (MXFP8 or BF16, which no fused pair reads).
    pub sh_rows: bool,
    /// The forms the common card leg's downs read ([`kernels::down_act`] at
    /// `ff`).
    pub leg: DownForms,
}

impl FfnReads {
    /// These forms and those a layer of routed stacks `routed` (`[gate, up,
    /// down]`, none for a layer without them) and shared expert `[gate, up,
    /// down]` reads at `ff`.
    #[must_use]
    pub fn with_layer(
        mut self,
        routed: Option<[GgmlType; 3]>,
        [shared_gate, shared_up, shared_down]: [GgmlType; 3],
        ff: usize,
    ) -> FfnReads {
        if let Some(t) = routed {
            if accepts::ds41_fused(t) {
                self.fused = true;
                self.fused_q5k_down |= matches!(t[2], GgmlType::Q5_K);
            } else if leg_runs(Program::Deepseek41Chain, t) == Some(true) {
                self.leg = self.leg.with(kernels::down_act(t[2], ff as u64));
            }
        }
        self.sh_q8_1 |= matches!(shared_down, GgmlType::Q3_K | GgmlType::Q4_K);
        let rows = |t| matches!(t, GgmlType::MXFP8 | GgmlType::BF16);
        self.sh_rows |= rows(shared_gate) && rows(shared_up);
        self
    }

    /// The forms the layers `kinds` read at `ff`.
    #[must_use]
    pub fn of(kinds: &[LayerKind], ff: usize) -> FfnReads {
        kinds.iter().fold(FfnReads::default(), |r, k| {
            let [g, u] = k.shared_gate_up;
            r.with_layer(k.routed_types, [g, u, k.shared_down], ff)
        })
    }
}

/// The levers that add buffers to a batch: the group's batch sets, the
/// expert tier's places, the host stream's places and the feature tap's
/// rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchLevers {
    /// `BLOOMERY_PREFILL_GROUP`.
    pub group: usize,
    /// A load with an expert tier.
    pub tiered: bool,
    /// `BLOOMERY_HOSTSTREAM` as the load takes it.
    pub hoststream: bool,
    /// The feature tap's values a row; 0 without a tap.
    pub tap_width: usize,
}

impl BatchLevers {
    /// What a plan counts, for a load with an expert tier or not: the group
    /// lever's default, the host stream's places (a residency load streams;
    /// they are [`T_MAX`]·n_used words, counted whether it does or not), and
    /// no feature tap (a draft's tap rows are its own plan's).
    #[must_use]
    pub const fn planned(tiered: bool) -> BatchLevers {
        BatchLevers {
            group: bloomery_levers::PREFILL_GROUP_DEFAULT as usize,
            tiered,
            hoststream: true,
            tap_width: 0,
        }
    }
}

/// What a prompt batch's buffers are sized by.
#[derive(Clone, Debug)]
pub struct BatchShape<'a> {
    pub hp: &'a Hparams,
    /// The stage card's layers.
    pub layers: Range<usize>,
    /// Positions the caches hold.
    pub ctx_max: usize,
    pub levers: BatchLevers,
}

impl<'a> BatchShape<'a> {
    /// The shape a plan counts for the stage card holding `layers` of `hp`
    /// at `ctx_max` ([`BatchLevers::planned`]).
    #[must_use]
    pub fn planned(hp: &'a Hparams, layers: Range<usize>, ctx_max: usize, tiered: bool) -> Self {
        BatchShape {
            hp,
            layers,
            ctx_max,
            levers: BatchLevers::planned(tiered),
        }
    }
}

/// The lengths of the batch-wide attention projections' buffers
/// ([`ATTN_BATCH`]), from which the card body allocates them: the f32
/// buffers in its field order, the q8_1 forms (columns, values a column),
/// and HC_PRE's grouped scratch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttnBatchLens {
    /// Tokens of a sub-block.
    pub sub: usize,
    pub f32s: [usize; 16],
    pub acts: Vec<(usize, usize)>,
    pub hc_groups: usize,
    pub hc_k: usize,
    /// Values of a token's hyper-connection mix.
    pub hc_mix: usize,
}

impl AttnBatchLens {
    /// The lengths for groups of `sets` batches of up to `tokens` positions
    /// over the card's layers of `hp`, `any_source` and `any_indexer` whether
    /// a card layer owns a compressor or runs the indexer.
    #[must_use]
    pub fn of(
        hp: &Hparams,
        tokens: usize,
        sets: usize,
        any_source: bool,
        any_indexer: bool,
    ) -> Self {
        let s = SUB_TOKENS.min(tokens.next_multiple_of(CHUNK));
        let index_w = if any_indexer { INDEX_HEADS } else { 0 };
        let index_q = if any_indexer {
            INDEX_HEADS * hp.indexer.head_dim
        } else {
            0
        };
        let q_rows = hp.n_head * hp.head_dim;
        let group_k = q_rows / hp.o_groups.max(1);
        let groups = q_rows / group_k.max(1);
        let width = hp.head_dim;
        let comp_raw = if any_source { 2 * s * width } else { 0 };
        let comp = if any_source { (s + CHUNK) * width } else { 0 };
        let mix = hc_mix(hp);
        let counts = SUB_COUNTS.iter().copied().filter(|&c| c <= s);
        AttnBatchLens {
            sub: s,
            f32s: [
                tokens * mix,
                tokens * mix,
                s * hp.n_embd,
                s * (hp.q_lora_rank + hp.head_dim + index_w),
                s * hp.q_lora_rank,
                s * hp.head_dim,
                s * hp.q_lora_rank,
                s * (q_rows + index_q),
                s * q_rows,
                comp_raw,
                2 * comp,
                s * q_rows,
                s * groups * hp.o_lora_rank,
                s * hp.n_embd,
                tokens * hp.n_embd,
                sets * TABLES * tokens * hp.rope_dims,
            ],
            acts: counts
                .flat_map(|c| [(c, hp.n_embd), (c, hp.q_lora_rank)])
                .chain([(s * groups, group_k), (s, groups * hp.o_lora_rank)])
                .collect(),
            hc_groups: tokens.div_ceil(CHUNK) + 1,
            hc_k: hp.hc.streams * hp.n_embd,
            hc_mix: mix,
        }
    }

    /// The buffers as one part, in the card body's allocation order: HC_PRE's
    /// scratch, then the f32 buffers with each q8_1 form after the buffer it
    /// quantizes.
    #[must_use]
    pub fn part(&self) -> Part {
        let f = &self.f32s;
        let mut p = hc_pre(
            Part::new(ATTN_BATCH),
            self.hc_k,
            self.hc_groups,
            self.hc_mix,
        );
        let (normed, rest) = self.acts.split_at(self.acts.len() - 2);
        p = p.word(f[0]).word(f[1]).word(f[2]);
        for &(c, k) in normed.iter().step_by(2) {
            p = q8act(p, c, k);
        }
        p = p.word(f[3]).word(f[4]).word(f[5]).word(f[6]);
        for &(c, k) in normed.iter().skip(1).step_by(2) {
            p = q8act(p, c, k);
        }
        p = p.word(f[7]).word(f[8]);
        if f[9] > 0 {
            p = p.word(f[9]);
        }
        if f[10] > 0 {
            p = p.word(f[10]);
        }
        p = q8act(p.word(f[11]), rest[0].0, rest[0].1);
        p = q8act(p.word(f[12]), rest[1].0, rest[1].1);
        p.word(f[13]).word(f[14]).word(f[15])
    }

    /// The buffers' bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.part().bytes()
    }
}

/// Values of a token's hyper-connection mix, `(2 + s) · s` for `s` streams.
fn hc_mix(hp: &Hparams) -> usize {
    (2 + hp.hc.streams) * hp.hc.streams
}

/// `p` with an HC_PRE scratch for inputs of `k` values over `groups` token
/// groups: its partials, squares and ticket a group.
fn hc_pre(p: Part, k: usize, groups: usize, mix: usize) -> Part {
    let pieces = k / HC_PIECE;
    p.word(mix * pieces * CHUNK * groups)
        .word(pieces * CHUNK * groups)
        .word(groups)
}

/// `p` with a q8_1 form of `cols` columns of `k` values, its five planes
/// ([`q8_act_bytes`]: q3 u64, q4, q6, s8, d8).
fn q8act(p: Part, cols: usize, k: usize) -> Part {
    let n_sb = k / 256;
    let p = p
        .with(cols * 64 * n_sb.div_ceil(2), 8)
        .word(cols * 256 * n_sb.div_ceil(4))
        .word(cols * 128 * n_sb.div_ceil(2))
        .word(cols * 8 * n_sb)
        .word(cols * 2 * n_sb);
    debug_assert_eq!(
        p.buffers()[p.buffers().len() - 5..]
            .iter()
            .map(|b| b.bytes())
            .sum::<u64>(),
        q8_act_bytes(cols as u64, k as u64)
    );
    p
}

/// Bytes of one row of `values` values of type `ty`; refused by name for a
/// type whose blocks do not tile it.
fn row_bytes(ty: GgmlType, values: usize, what: &str) -> Result<usize, PlacementError> {
    ty.blck_size()
        .zip(ty.type_size())
        .filter(|&(b, _)| b > 0 && (values as u64).is_multiple_of(b))
        .map(|(b, t)| values / b as usize * t as usize)
        .ok_or_else(|| PlacementError::Metadata {
            key: what.to_owned(),
            detail: format!("rows of {values} values of {ty} are no whole number of its blocks"),
        })
}

/// Words of a step image of `m` tokens of `hp`'s file (`ImageLayout::new`):
/// the tokens, each stream's block at its capacities, the rope tables, the
/// embedding rows and the engram rows, each section on [`ALIGN`] words.
fn image_words(hp: &Hparams, ratios: &[u32], m: usize) -> Result<usize, PlacementError> {
    let align = |n: usize| n.next_multiple_of(ALIGN);
    let nd = hp.rope_dims;
    let mut at = align(TOKEN_WORDS * m);
    for &r in ratios {
        let r = r as usize;
        let (gs, ps) = (m.div_ceil(r), m.min(r));
        at = align(at + STREAM_HEADER + m + gs * r + gs + gs + ps + ps);
    }
    at += m * TABLES * nd;
    at += ratios
        .iter()
        .filter(|&&r| r > 1)
        .map(|&r| m.div_ceil(r as usize) * nd)
        .sum::<usize>();
    let embd = row_bytes(hp.rows.token_embd, hp.n_embd, "token_embd")?;
    let engram = match &hp.engram {
        Some(e) => {
            let ty = hp.rows.engram.ok_or_else(|| PlacementError::Metadata {
                key: "engram tables".to_owned(),
                detail: "engram sites with no table type".to_owned(),
            })?;
            e.layer_ids.len() * e.rows_per_token() * row_bytes(ty, e.key_length, "engram tables")?
        }
        None => 0,
    };
    let embd_at = align(at);
    let engram_at = align(embd_at + m * embd.div_ceil(4));
    Ok(align(engram_at + m * engram.div_ceil(4)))
}

/// The attention piece's step words over images of `m` tokens and its
/// gather's pairs (`plan_words`): words a row, pairs.
fn attn_words(hp: &Hparams, ratios: &[u32], m: usize) -> (usize, usize) {
    let nd = hp.rope_dims;
    let mut len = 0usize;
    let mut alloc = |n: usize| {
        let at = len.next_multiple_of(4);
        len = at + n;
    };
    // The positions, a pair a token; the window lengths, two words and a
    // pair a token.
    alloc(m);
    alloc(2 * m);
    let mut pairs = 2 * m;
    for _ in 0..TABLES {
        alloc(m * nd);
        pairs += m * nd;
    }
    // `top_k`, which no pair writes.
    alloc(1);
    for &r in ratios {
        let r = r as usize;
        let (gm, ps) = (m.div_ceil(r), m.min(r));
        alloc(2 * m);
        pairs += 2 * m;
        if m > 1 {
            alloc(m);
            pairs += m;
        }
        // The compressor's step buffer (`CompGeom::words`).
        alloc(2 + gm * (1 + r) + 2 * r);
        pairs += 1 + gm + gm * r;
        if r > 1 {
            pairs += 1 + 2 * ps;
            alloc(gm * nd);
            pairs += gm * nd;
        }
    }
    (len.next_multiple_of(4), pairs)
}

/// The prompt batch's parts for `shape` (see the module comment), refused by
/// name where the file's widths cannot be laid out.
pub fn prompt_parts(shape: &BatchShape<'_>) -> Result<Parts, PlacementError> {
    let hp = shape.hp;
    let lv = shape.levers;
    let sets = group_sets(lv.group);
    let ctx = shape.ctx_max;
    let m = CHUNK;
    let n = hp.n_embd;
    let refuse = |detail: String| PlacementError::Metadata {
        key: "the prompt batch's shape".to_owned(),
        detail,
    };
    let ratios_all: Vec<u32> = hp.layers.iter().map(|k| k.ratio()).collect();
    let max_ngram = hp.engram.as_ref().map_or(1, |e| e.max_ngram);
    let planner = Planner::new(
        u32::try_from(hp.window).map_err(|_| refuse(format!("a window of {}", hp.window)))?,
        &ratios_all,
        max_ngram,
        ctx as u64,
    )
    .map_err(|e| refuse(e.to_string()))?;
    let ratios = planner.stream_ratios();
    let card = shape.layers.clone();
    let kinds = hp
        .layers
        .get(card.clone())
        .ok_or_else(|| refuse(format!("layers {card:?} of {}", hp.layers.len())))?;
    let any_source = kinds.iter().any(|k| k.compressor.is_some());
    let any_indexer = kinds.iter().any(|k| k.indexer);
    let any_stream = card.clone().any(|l| planner.layer_stream(l).is_some());
    let writers = kinds
        .iter()
        .filter(|k| k.stream.is_some() && k.indexer)
        .count();
    let list_len = hp.indexer.top_k;
    let rows = CHUNKS_MAX * sets;
    let stream_rows = |s: usize| ctx.div_ceil(ratios[s] as usize);

    // The attention piece over the batch layout, `rows` rows of words.
    let (words, pairs) = attn_words(hp, ratios, m);
    let mut attn_words_part = Part::new(ATTN_WORDS);
    for _ in 0..rows {
        attn_words_part = attn_words_part.word(words);
    }
    let attn_words_part = attn_words_part.word(pairs).word(pairs);

    let gm = ratios
        .iter()
        .map(|&r| m.div_ceil(r as usize))
        .max()
        .unwrap_or(1);
    let mut select = Part::new(ATTN_SELECT);
    let width = hp.head_dim;
    let key = hp.indexer.head_dim;
    if any_source {
        select = q8act(select, 1, n).word(2 * m * width).word(gm * width);
        select = q8act(select, gm, width).word(gm * key);
    }
    if any_indexer {
        let mut indexed = vec![false; ratios.len()];
        for l in card.clone() {
            if hp.layers[l].indexer
                && let Some(s) = planner.layer_stream(l)
            {
                indexed[s] = true;
            }
        }
        for c in 1..=m {
            for (s, &on) in indexed.iter().enumerate() {
                if on {
                    select = select
                        .word(c * INDEX_HEADS * key)
                        .word(c * INDEX_HEADS)
                        .word(c * stream_rows(s))
                        .word(c * INDEX_HIST_BINS);
                }
            }
        }
        let cand = hp
            .candidates
            .filter(|_| card.clone().any(|l| hp.candidate_role(l).is_some()));
        if let Some(mask) = cand {
            let src = card
                .clone()
                .find(|&l| hp.candidate_role(l) == Some(CandidateRole::Source))
                .and_then(|l| planner.layer_stream(l))
                .ok_or_else(|| {
                    refuse(format!(
                        "the card's layers {card:?} hold a consumer of the candidate mask of \
                         layer {} and not that layer",
                        mask.source_layer
                    ))
                })?;
            let blocks_of = stream_rows(src).div_ceil(mask.block_size.max(1));
            for c in 1..=m {
                select = select.word(c * blocks_of);
            }
            select = select.word(m + 1);
            for _ in 0..rows {
                select = select.word(m * mask.topk_blocks);
            }
        }
        select = q8act(select, 1, n);
    }

    let q_rows = m * hp.n_head;
    let group_k = hp.n_head * hp.head_dim / hp.o_groups.max(1);
    let window_rows = ctx.min(hp.window);
    let comp_keys = if any_stream { list_len } else { 0 };
    let segs = window_rows.div_ceil(SEG_KEYS) + comp_keys.div_ceil(SEG_KEYS);
    let index_rows = if any_indexer { m } else { 0 };
    let mut scratch = q8act(Part::new(ATTN_SCRATCH), 1, n);
    scratch = q8act(scratch, 1, hp.q_lora_rank);
    scratch = q8act(scratch, hp.o_groups, group_k);
    scratch = q8act(scratch, 1, hp.o_groups * hp.o_lora_rank);
    if m > 1 && any_source {
        scratch = scratch.word(gm * key);
    }
    let scratch = hc_pre(scratch, hp.hc.streams * n, 1, hc_mix(hp))
        .word(hc_mix(hp) * m)
        .word(hc_mix(hp) * m)
        .word(m * n)
        .word(m * hp.q_lora_rank + m * hp.head_dim + index_rows * INDEX_HEADS)
        .word(m * hp.q_lora_rank)
        .word(q_rows * hp.head_dim + index_rows * INDEX_HEADS * key)
        .word(m * hp.head_dim)
        .word(q_rows * segs * hp.head_dim)
        .word(q_rows * segs * 2)
        .word(q_rows * hp.head_dim)
        .word(m * hp.o_groups * hp.o_lora_rank)
        .word(m * n);

    let image = image_words(hp, ratios, m)?;
    let params = Part::new(PARAMS).word(sets * CHUNKS_MAX * image);

    let glue = match &hp.engram {
        Some(e) => {
            let rows_values = e.rows_per_token() * e.key_length;
            let wide = (hp.hc.streams + 1) * n;
            let mut g = Part::new(GLUE).word(m * rows_values);
            for c in 1..=m {
                g = q8act(g, c, rows_values);
            }
            g.word(m * wide)
                .word(m * wide)
                .word(m * hp.hc.streams * n)
                .word(m * hp.hc.streams)
        }
        None => Part::new(GLUE),
    };

    let (ff, used, experts) = (hp.experts.ff, hp.experts.n_used, hp.experts.n_expert);
    let reads = FfnReads::of(kinds, ff);
    let cap = T_MAX;
    let slot_cols = cap * used;
    let n_sb = n / 256;
    let mix = hc_mix(hp);
    let mut ffn = Part::new(FFN_ROWS)
        .word(cap * n)
        .word(cap * used)
        .word(cap * used)
        .word(cap * used);
    for _ in 0..sets {
        ffn = ffn.word(cap * mix);
    }
    ffn = ffn
        .word(cap * n)
        .word(cap * n)
        .word(cap * n)
        .word(cap * experts)
        .word(m * n);
    for c in 1..=m {
        ffn = q8act(ffn, c, n);
    }
    ffn = ffn.word(m * mix).word(m * ff);
    if reads.sh_q8_1 {
        for c in 1..=m {
            ffn = q8act(ffn, c, ff);
        }
    }
    let ffn = ffn
        .word(m * n)
        .with(cap * 64 * n_sb.div_ceil(2), 8)
        .word(cap * 2 * n_sb)
        .word(slot_cols * n)
        .word(experts);
    let tiles = if reads.fused {
        let p = Part::new(FFN_TILES).bytes_buffer(tier_block_scratch_bytes(
            n as u64,
            ff as u64,
            slot_cols as u64,
        ));
        let p = q8act(p, cap, n).word(slot_cols * ff).word(slot_cols * ff);
        q8act(p, slot_cols, ff)
    } else {
        Part::new(FFN_TILES)
    };
    let leg = if hp.on_leg() {
        Part::new(FFN_LEG).bytes_buffer(kernels::leg_bytes(
            LEG_PIECE as u64,
            used as u64,
            n as u64,
            ff as u64,
            reads.leg,
        ))
    } else {
        Part::new(FFN_LEG)
    };
    let mut places = Part::new(FFN_PLACES);
    if lv.tiered {
        places = places.word(slot_cols);
        for _ in 0..sets {
            places = places.word(slot_cols).word(slot_cols).word(slot_cols);
        }
    }
    if lv.hoststream {
        places = places.word(cap * used);
    }

    let mut set_part = Part::new(SETS);
    for _ in 0..sets {
        let s = hp.hc.streams;
        set_part = set_part
            .word(T_MAX * s * n)
            .word(T_MAX * s * n)
            .word(T_MAX * n)
            .word(T_MAX * n);
        for _ in 0..CHUNKS_MAX * writers {
            set_part = set_part.word(CHUNK * list_len);
        }
        if lv.tap_width > 0 {
            set_part = set_part.word(T_MAX * lv.tap_width);
        }
    }
    let staging = Part::new(STAGING).with(CHUNK * hp.head_dim, 2);
    let attn_batch = AttnBatchLens::of(hp, T_MAX, sets, any_source, any_indexer).part();
    Ok(Parts::new(vec![
        attn_words_part,
        select,
        scratch,
        params,
        glue,
        ffn,
        tiles,
        leg,
        places,
        set_part,
        staging,
        attn_batch,
    ]))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;

    fn bytes_by_part(p: &Parts) -> Vec<(&'static str, u64)> {
        p.parts().iter().map(|q| (q.name(), q.bytes())).collect()
    }

    /// Four window-only layers of width 256 (`Hparams::tiny`) at 1,024
    /// positions, the planned levers (group 2: three sets, the host stream's
    /// places, no tier, no tap), derived by hand from the code:
    ///
    /// - attn words: an image row of 8 tokens is 2,076 words (positions 8,
    ///   windows 16 at 8, four tables of 8·64 from 24 to 2,072, `top_k` at
    ///   2,072, rounded to 4), 65·3 = 195 rows, and two pair buffers of
    ///   16 + 4·512 = 2,064: 4·(195·2,076 + 2·2,064) = 1,635,792.
    /// - attn select: no compressor, no indexer: 0.
    /// - attn scratch: four q8_1 forms of one column of 256 (2,088 B each:
    ///   512 + 1,024 + 512 + 32 + 8), HC_PRE over 4·256 (two pieces: 384 +
    ///   16 + 1 words), and 22,944 words of chunk buffers (24·8 twice,
    ///   8·256 eight times, 8·512, two segments of 64 keys: 8·2·256 and
    ///   8·2·2): 8,352 + 1,604 + 91,776 = 101,732.
    /// - params: a step image of 8 tokens is 4,128 words (tokens 32, tables
    ///   2,048, F32 embedding rows 2,048), 195 of them: 3,219,840.
    /// - ffn rows: 866,560 words (512·256 rows, three 512·2 lists, three
    ///   512·24 mixes, three 512·256 rows, 512·64 logits, 8·256, 8·24,
    ///   8·256, 8·256, 512·2 scales, 1,024·256 slot rows, 64 counts), the
    ///   chunk's q8_1 at 1..=8 columns (2,088·36) and the union's q3 plane
    ///   (512·64·8): 3,466,240 + 75,168 + 262,144 = 3,803,552.
    /// - ffn places: the host stream's 512·2 words: 4,096.
    /// - batch sets: three of 2·512·4·256 + 2·512·256 words, no writer:
    ///   15,728,640.
    /// - staging: 8·256 halves: 4,096.
    /// - attn batch: sub-blocks of 128; 909,312 f32 words (512·24 twice,
    ///   nine of 128·256, 128·512, 512·256, 3·4·512·64), the q8_1 forms at
    ///   the twelve counts (276 columns, twice) and two of 128 columns
    ///   (2,088·(552 + 256)), HC_PRE over 65 groups (26,065 words):
    ///   3,637,248 + 1,687,104 + 104,260 = 5,428,612.
    #[test]
    fn each_part_is_its_hand_derived_bytes() {
        let hp = Hparams::tiny(4);
        let parts = prompt_parts(&BatchShape::planned(&hp, 0..4, 1024, false)).expect("parts");
        assert_eq!(
            bytes_by_part(&parts),
            vec![
                (ATTN_WORDS, 1_635_792),
                (ATTN_SELECT, 0),
                (ATTN_SCRATCH, 101_732),
                (PARAMS, 3_219_840),
                (GLUE, 0),
                (FFN_ROWS, 3_803_552),
                (FFN_TILES, 0),
                (FFN_LEG, 0),
                (FFN_PLACES, 4_096),
                (SETS, 15_728_640),
                (STAGING, 4_096),
                (ATTN_BATCH, 5_428_612),
            ]
        );
        assert_eq!(parts.bytes(), 29_926_360);
    }

    /// The fixture's prompt term, the plan's `prompt=` on a card of 2 MiB
    /// granules with an expert tier, every fixture layer on the stage card:
    /// at the gates' 32,768 positions (`gate-gpu-ds41-residency-a`'s
    /// callstream) and at 4,096 (`engram-layout`'s and `dspark-loop`'s
    /// `--ctx 4096`). Only the indexer's scores follow the context:
    /// 36·(ctx/2 + ctx) words over the token counts 1..=8 at ratios 2 and 1.
    #[test]
    fn the_fixture_shape_sums_to_its_prompt_term() {
        let hp = Hparams::v41_fixture_model();
        let granule = NonZeroU64::new(2 << 20).expect("not zero");
        let at = |ctx| {
            let p =
                prompt_parts(&BatchShape::planned(&hp, 0..9, ctx, true)).expect("fixture parts");
            (p.bytes(), p.card_bytes(granule))
        };
        assert_eq!(
            [at(32_768), at(4096)],
            [(661_216_064, 675_282_944), (655_022_912, 668_991_488)]
        );
    }

    /// The made figures of a body that allocated a part short are refused,
    /// the part named.
    #[test]
    fn a_batch_off_its_parts_is_refused_by_part() {
        let hp = Hparams::tiny(4);
        let parts = prompt_parts(&BatchShape::planned(&hp, 0..4, 1024, false)).expect("parts");
        let mut made = bytes_by_part(&parts);
        assert_eq!(parts.check("ds41", &made), Ok(()));
        made[9].1 -= 4;
        let e = parts.check("ds41", &made).expect_err("the sets are short");
        assert_eq!(e.apart.len(), 1);
        assert!(
            e.to_string()
                .contains("batch sets made 15728636 B planned 15728640 B (-4 B)")
        );
    }
}
