//! The Inkling 2 gate fixtures' specs ([`spec`], [`plain_spec`]): the facts of an `inkling2` file
//! the common generator (`crate::fixture`) cannot know. The family's two cases in one family: the
//! Mini's looped trunk (a core of whole blocks of the block attention residual, three passes, the
//! message controllers a core layer carries) and the Small's loopless one (no loop keys, every
//! layer its own single visit) — each with windowed and windowless layers, the windowless layers'
//! dynamic k and v conv taps beside their three input norms, the padded vocabulary with its
//! unpadded cut, and the experts narrowed to [`FIXTURE_FF`] (the smallest whole block of every
//! type the files hold: 32 values covers Q8_0's blocks and the i-quants' 16).
//!
//! The looped fixture ([`spec`], the registry row's) is thirteen layers of the Mini — the stem
//! block whole (the trunk's first layer among them), one whole core block (a windowless core
//! layer with its controllers among them), one coda layer of the Q8_0 expert set — cut from the
//! box's UD-Q4_K_XL:
//!
//! | fixture layer | source layer | kind | why |
//! |---|---|---|---|
//! | f0 | L0 | windowed, first | the trunk's first layer, which carries no attention mix query |
//! | f1-f4 | L1-L4 | windowed | the windowed layers' shared input norm and one-row query |
//! | f5 | L5 | windowless | the stem's windowless layer: three input norms, dynamic conv taps |
//! | f6-f10 | L6-L10 | core, windowed | the loop's controllers beside the windowed core block |
//! | f11 | L11 | core, windowless | the controllers beside the dynamic conv taps |
//! | f12 | L20 | coda, windowed | a layer past the core, of the Q8_0 expert set |
//!
//! The loop keys move with the map: the fixture's core is its layers [6, 12) — whole blocks of
//! the copied `attn_res.block_size` — so its 13 stored layers expand to 25 visits, and the three
//! per-visit arrays (`attention.head_count`, `attention.head_count_kv`,
//! `attention.sliding_window_pattern`) are rebuilt one value a fixture visit, a visit's value its
//! stored layer's first visit's, exactly as the reader holds a looped file
//! ([`hparams`]'s visits agreement). The model-level message queries narrow with the core: two
//! rows a kept core layer.
//!
//! The loopless spec ([`plain_spec`]) is the same family without the loop: five layers of the
//! Small — a windowed first layer, a windowed one, a windowless one (the dynamic convs), another
//! windowed and a second windowed — under the plain rules (a per-layer array follows the map, a
//! scalar is copied). No file of the Small's quant mix can be cut yet: `iq2_xs` and `iq3_xxs`
//! have no fixture fill rule (`crate::fixture::fill`), so the spec is held to its round-trip on a
//! synthetic source until those land.
//!
//! The fills (the generator's matrix rule is σ 1/√K): the norm gains 1; the router's selection
//! bias a spread (so a body that broadcasts one value or skips it fails); the loop's controllers,
//! the bias banks and the conv taps the matrix rule. The card budget is fixed.

use gguf::Split;
use gguf::Value;
use models::LoopSpec;

use super::hparams::{Hparams, Loop};
use super::roles::LayerKind;
use crate::fixture::hybrid;
use crate::fixture::{
    CardBudget, DEFAULT_SHARD_BYTES, Family, FixtureError, FixtureSpec, KeyRule, Tables, Window,
    int_like, meta,
};

/// Fixture layer `f` holds source layer `LAYER_MAP[f]` (the table above): the Mini's trunk at
/// one whole stem block, one whole core block and a coda layer of the Q8_0 experts.
pub const LAYER_MAP: [usize; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20];

/// The loopless fixture's layer map: the Small's trunk at a first windowed layer, two windowed
/// ones, a windowless one and another windowed one.
pub const PLAIN_LAYER_MAP: [usize; 5] = [0, 1, 5, 6, 20];

/// The d window, [2^-14, 2^-6]: normal f16 values every type's scale fits at every K the fixtures
/// hold (2,304 for the projections, 4,608 for the loop controllers' down, the padded vocabulary's
/// 201,024 rows for the embedding and the head).
pub const D_MIN: f32 = 1.0 / 16384.0;
/// See [`D_MIN`].
pub const D_MAX: f32 = 1.0 / 64.0;

/// The routed experts' feed-forward width the fixtures write in place of the source's: 512, whole
/// blocks of Q8_0 (32) and of the i-quants and NVFP4 (16) on the stacks' ff axis.
pub const FIXTURE_FF: u64 = 512;

/// The half-width of the router's selection bias (`exp_probs_b.bias`), drawn per expert: wide
/// enough that it steers the pick, narrow enough that the sigmoid scores still decide it.
pub const BIAS_HALF_WIDTH: f32 = 0.05;

/// The card byte budget the fixtures record: above their tensor data (about 4.4 GB), with room
/// for the cache and the scratch of the longest context a gate runs.
pub const CARD_BUDGET: u64 = 8 << 30;

/// The looped fixture's target files are `<STEM>-00001-of-00001.gguf`.
pub const STEM: &str = "inkling2-fixture";

/// The loopless fixture's target files are `<PLAIN_STEM>-…`.
pub const PLAIN_STEM: &str = "inkling2-small-fixture";

/// The real file `verify` opens when none is named: the Mini's UD-Q4_K_XL, one unsplit shard.
pub const DEFAULT_MODEL: &str = "/models/Inkling-2-Mini-GGUF/Inkling-2-Mini-UD-Q4_K_XL.gguf";

/// The loopless spec's default source: the Small's UD-Q2_K_XL, six shards.
pub const DEFAULT_PLAIN_MODEL: &str =
    "/models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf";

const ARCH: &str = "inkling2";

/// The Mini's looped fixture: [`LAYER_MAP`], the window [[`D_MIN`], [`D_MAX`]], the ff
/// [`FIXTURE_FF`], the Mini's file as the default source and a fixed card budget.
pub fn spec() -> FixtureSpec {
    FixtureSpec {
        arch: ARCH,
        stem: STEM,
        layers: LAYER_MAP.to_vec(),
        ratios: Vec::new(),
        card_budget: CardBudget::Fixed(CARD_BUDGET),
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-6] is inside the normal f16 values"),
        // The ff override rides the family's tables, not the common one: the fused gate and up
        // pair carries twice the ff on its axis, which the common narrowing does not spell.
        ff: None,
        shard_bytes: DEFAULT_SHARD_BYTES,
        default_source,
        draft: None,
        sidecar: None,
        family: &Looped,
    }
}

/// The Small's loopless fixture: [`PLAIN_LAYER_MAP`] under the plain rules, the Small's shards as
/// the default source, its fixture a split set (a shard cap under the file's size).
pub fn plain_spec() -> FixtureSpec {
    FixtureSpec {
        arch: ARCH,
        stem: PLAIN_STEM,
        layers: PLAIN_LAYER_MAP.to_vec(),
        ratios: Vec::new(),
        card_budget: CardBudget::Fixed(CARD_BUDGET),
        window: Window::new(D_MIN, D_MAX).expect("[2^-14, 2^-6] is inside the normal f16 values"),
        ff: None,
        shard_bytes: 1 << 30,
        default_source: plain_default_source,
        draft: None,
        sidecar: None,
        family: &Plain,
    }
}

fn default_source() -> String {
    DEFAULT_MODEL.to_string()
}

fn plain_default_source() -> String {
    DEFAULT_PLAIN_MODEL.to_string()
}

/// The coverage table of the architecture keys, shared by both specs; `None` is a key the map does
/// not cover. The copy list is every key [`Hparams::read`] reads that is not a function of the
/// map; the looped spec's own keys (the loop's and the three per-visit arrays) are its tables'
/// ([`LoopTables`]), and the loopless spec refuses a loop key by name: its file holds none.
pub fn key_rule(suffix: &str) -> Option<KeyRule> {
    const COPY: [&str; 26] = [
        "context_length",
        "embedding_length",
        "vocab_size",
        "vocab_size_unpadded",
        // The keys the reader lists unread (the dense width no layer holds, the swa mirrors of
        // the key and value widths, the rope this architecture never turns): the file's own
        // facts, copied.
        "feed_forward_length",
        "attention.key_length_swa",
        "attention.value_length_swa",
        "rope.dimension_count",
        "attention.key_length",
        "attention.value_length",
        "attention.layer_norm_rms_epsilon",
        "attention.sliding_window",
        "attention.relative_length",
        "attention.relative_extent",
        "attention.qk_clip",
        "attention.relative_clip",
        "attention.log_scale_floor",
        "attention.log_scale_alpha",
        "attn_res.block_size",
        "shortconv.l_cache",
        "expert_count",
        "expert_used_count",
        "expert_weights_scale",
        "final_logit_softcapping",
        "activation.situ_beta",
        "activation.situ_linear_beta",
    ];
    Some(match suffix {
        "block_count" => KeyRule::BlockCount,
        "attention.head_count" | "attention.head_count_kv" | "attention.sliding_window_pattern" => {
            KeyRule::Table
        }
        "expert_feed_forward_length" => KeyRule::Table,
        s if COPY.contains(&s) => KeyRule::Copy,
        _ => return None,
    })
}

/// The value of an F32 tensor the family names, by its name past `blk.N.` (or the whole name of a
/// global): every norm's gain, 1.
pub fn const_value(leaf: &str) -> Option<f32> {
    // The NVFP4 stacks' per-tensor scale companions (`ffn_*_exps.scale`): the scale the
    // generator writes the blocks at.
    if leaf.ends_with(".scale") {
        return Some(crate::fixture::NVFP4_TENSOR_SCALE);
    }
    const ONES: [&str; 10] = [
        "attn_norm.weight",
        "attn_q_inp_norm.weight",
        "attn_k_inp_norm.weight",
        "attn_v_inp_norm.weight",
        "ffn_norm.weight",
        "token_embd_norm.weight",
        "output_norm.weight",
        // The mixes' query vectors (a windowed layer's `attn_res_query` is stored one row, `[n]`):
        // a one-row query every state scores alike, the mechanism the gates read; the learned
        // spread is the real file's.
        "attn_res_query.weight",
        "ffn_res_query.weight",
        "output_res_query.weight",
    ];
    ONES.contains(&leaf).then_some(1.0)
}

/// The spread of an F32 tensor the family names: the router's selection bias.
pub fn spread_value(leaf: &str) -> Option<f32> {
    (leaf == "exp_probs_b.bias").then_some(BIAS_HALF_WIDTH)
}

/// `layers` keeps the source's facts: one layer of each kind the source holds and every (stem,
/// type) pair any source layer holds ([`hybrid::check_map`]).
pub fn check_map(src: &Hparams, source: &Split, layers: &[usize]) -> Result<(), FixtureError> {
    let kinds: Vec<LayerKind> = (0..src.n_layer).map(|l| src.kind(l)).collect();
    let kept: Vec<LayerKind> = {
        let mut seen = Vec::new();
        for kind in &kinds {
            if !seen.contains(kind) {
                seen.push(*kind);
            }
        }
        seen
    };
    hybrid::check_map(&kinds, &kept, source, layers)
}

/// The fixture's core, in fixture coordinates: the map's run of the source's core layers — whole
/// blocks of the source's `attn_res.block_size`, starting at a block boundary of the fixture's own
/// coordinates — as `(start, end)`. Refused by name where the map holds none of the core, holds
/// only part of it, or holds it in pieces the block size cannot take.
fn fixture_core(src: &Hparams, map: &[usize]) -> Result<(usize, usize), FixtureError> {
    let Some(loop_) = src.loop_ else {
        return Err(meta(
            &format!("{ARCH}.loop.core_end"),
            "is absent: the looped fixture is cut from a looped file",
        ));
    };
    let mut held = map
        .iter()
        .enumerate()
        .filter(|&(_, &l)| (loop_.core_start..loop_.core_end).contains(&l))
        .map(|(f, &l)| (f, l));
    let Some((first, _)) = held.next() else {
        return Err(meta(
            &format!("{ARCH}.loop.core_start"),
            format!(
                "the map {map:?} holds none of the core [{}, {})",
                loop_.core_start, loop_.core_end
            ),
        ));
    };
    let mut end = first + 1;
    for (f, l) in held {
        if l != map[end - 1] + 1 {
            return Err(meta(
                &format!("{ARCH}.loop.core_start"),
                format!(
                    "the map {map:?} holds the core in pieces (fixture layers {first} and {f})"
                ),
            ));
        }
        end = f + 1;
    }
    let span = end - first;
    if !first.is_multiple_of(src.block) || !span.is_multiple_of(src.block) {
        return Err(meta(
            &format!("{ARCH}.loop.core_start"),
            format!(
                "the map {map:?} holds the core at fixture layers [{first}, {end}), not whole \
                 blocks of attn_res.block_size {}",
                src.block
            ),
        ));
    }
    Ok((first, end))
}

/// The first visit of source layer `l` in a looped file of `loop_` (llama-hparams.cpp:381-383):
/// the layer's own index inside the core, its index past the core's repeats after it.
fn first_visit(loop_: &Loop, l: usize) -> usize {
    let extra = (loop_.passes - 1) * (loop_.core_end - loop_.core_start);
    if l < loop_.core_end { l } else { l + extra }
}

/// The looped fixture's tables: the keys and the two model-level message-query tensors that move
/// with the map beyond renaming.
struct LoopTables {
    /// The source's loop.
    src: Loop,
    /// The fixture's core, in fixture coordinates.
    core: (usize, usize),
    /// The fixture's layer visits in run order ([`models::visits`]), each naming its fixture
    /// layer.
    visits: Vec<models::Visit>,
    /// The source's first-visit index of each map layer's source layer.
    src_first: Vec<usize>,
    /// The source's `block_count` (its visits under the unsloth dialect, its layers under the
    /// master's).
    src_blocks: usize,
}

impl LoopTables {
    /// The tables of a fixture of `spec` cut from `source`: the map held to [`check_map`] and to
    /// the loop's whole blocks ([`fixture_core`]) first, by name, before any byte is planned.
    fn read(source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        let src = Hparams::read(source)?;
        check_map(&src, source, &spec.layers)?;
        let core = fixture_core(&src, &spec.layers)?;
        let loop_ = src.loop_.expect("fixture_core read it");
        let visits = models::visits(
            spec.layers.len(),
            Some(LoopSpec {
                core_start: core.0 as u32,
                core_end: core.1 as u32,
                passes: loop_.passes as u32,
                gate_rank: loop_.gate_rank as u32,
            }),
        );
        let src_first = spec
            .layers
            .iter()
            .map(|&l| first_visit(&loop_, l))
            .collect::<Vec<_>>();
        Ok(Box::new(LoopTables {
            src: loop_,
            core,
            visits,
            src_first,
            src_blocks: src.n_blocks,
        }))
    }
}

impl Tables for LoopTables {
    /// The fixture's value of the keys that move with the map: `block_count` the fixture's visits,
    /// the loop's core the fixture's coordinates (the passes and the rank the source's), and the
    /// three per-visit arrays rebuilt one value a fixture visit — a visit's value its stored
    /// layer's first visit's, an array under either dialect, a scalar copied.
    fn key(&self, key: &str, suffix: &str, v: &Value) -> Result<Value, FixtureError> {
        let per_visit = |key: &str, v: &Value| -> Result<Value, FixtureError> {
            // A scalar is every layer's value: the map moves nothing.
            let Value::Array(a) = v else {
                return Ok(v.clone());
            };
            if a.len() != self.src_blocks {
                return Err(meta(
                    key,
                    format!(
                        "has {} values for the source's {} blocks",
                        a.len(),
                        self.src_blocks
                    ),
                ));
            }
            let at = |l: models::Visit| a[self.src_first[l.layer as usize]].clone();
            Ok(Value::Array(
                self.visits.iter().map(|l| at(*l)).collect::<Vec<_>>(),
            ))
        };
        Ok(match suffix {
            "block_count" => int_like(key, v, self.visits.len() as u64)?,
            "loop.core_start" => int_like(key, v, self.core.0 as u64)?,
            "loop.core_end" => int_like(key, v, self.core.1 as u64)?,
            "expert_feed_forward_length" => int_like(key, v, FIXTURE_FF)?,
            "num_loops" | "loop.gate_rank" => v.clone(),
            "attention.head_count"
            | "attention.head_count_kv"
            | "attention.sliding_window_pattern" => per_visit(key, v)?,
            _ => return Err(meta(key, "is not a key of the looped fixture's tables")),
        })
    }

    /// The model-level message queries narrow with the core (two rows a kept core layer), and the
    /// routed stacks narrow to [`FIXTURE_FF`]: the fused gate and up pair twice that on its
    /// second dim, the down it on its first.
    fn dims(
        &self,
        _name: &str,
        layer: Option<usize>,
        leaf: &str,
        dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError> {
        if layer.is_none()
            && (leaf == "loop_res_query.weight" || leaf == "loop_kv_res_query.weight")
        {
            let rows = 2 * (self.core.1 - self.core.0) as u64;
            return Ok(Some(vec![dims[0], rows]));
        }
        Ok(narrow_stacks(leaf, dims))
    }

    fn lines(&self) -> Vec<String> {
        vec![format!(
            "loop core [{}, {}) of the source's [{}, {}): {} stored layers expand to {} visits",
            self.core.0,
            self.core.1,
            self.src.core_start,
            self.src.core_end,
            self.src_first.len(),
            self.visits.len(),
        )]
    }
}

/// The looped fixture's rules.
struct Looped;

impl Family for Looped {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        match suffix {
            // The loop's keys and the per-visit arrays are the tables' own ([`LoopTables`]): the
            // block count is the fixture's visits, not its layers.
            "block_count" | "num_loops" | "loop.core_start" | "loop.core_end"
            | "loop.gate_rank" => Some(KeyRule::Table),
            s => key_rule(s),
        }
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        spread_value(leaf)
    }

    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        LoopTables::read(source, spec)
    }

    fn check_kinds(
        &self,
        spec: &FixtureSpec,
        fixture: &Split,
        source: &Split,
    ) -> Result<(), FixtureError> {
        check_kinds(spec, fixture, source).map(|_| ())
    }
}

/// The loopless fixture's tables: the ff key and the routed stacks' widths, the looped tables'
/// own without the loop.
struct PlainTables;

impl Tables for PlainTables {
    fn key(&self, key: &str, suffix: &str, v: &Value) -> Result<Value, FixtureError> {
        Ok(match suffix {
            "expert_feed_forward_length" => int_like(key, v, FIXTURE_FF)?,
            _ => return Err(meta(key, "is not a key of the loopless fixture's tables")),
        })
    }

    fn dims(
        &self,
        _name: &str,
        _layer: Option<usize>,
        leaf: &str,
        dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError> {
        Ok(narrow_stacks(leaf, dims))
    }

    fn lines(&self) -> Vec<String> {
        vec![format!("experts narrowed to {FIXTURE_FF}")]
    }
}

/// The routed stacks of `leaf` at `dims` narrowed to [`FIXTURE_FF`]: the fused gate and up pair
/// twice it on its second dim, the down it on its first; `None` for any other tensor.
fn narrow_stacks(leaf: &str, dims: &[u64]) -> Option<Vec<u64>> {
    match leaf {
        "ffn_gate_up_exps.weight" => Some(vec![dims[0], 2 * FIXTURE_FF, dims[2]]),
        "ffn_down_exps.weight" => Some(vec![FIXTURE_FF, dims[1], dims[2]]),
        _ => None,
    }
}

/// The loopless fixture's rules: the shared table ([`key_rule`]) with no loop keys — a source that
/// carries one is refused by its key's name.
struct Plain;

impl Family for Plain {
    fn key_rule(&self, suffix: &str) -> Option<KeyRule> {
        match suffix {
            "num_loops" | "loop.core_start" | "loop.core_end" | "loop.gate_rank"
            | "loop.varying" => None,
            // No loop: the blocks are the stored layers and a per-block array follows the map.
            "block_count" => Some(KeyRule::BlockCount),
            "attention.head_count"
            | "attention.head_count_kv"
            | "attention.sliding_window_pattern" => Some(KeyRule::PerLayer),
            s => key_rule(s),
        }
    }

    fn const_value(&self, leaf: &str) -> Option<f32> {
        const_value(leaf)
    }

    fn spread_value(&self, leaf: &str) -> Option<f32> {
        spread_value(leaf)
    }

    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        let src = Hparams::read(source)?;
        if let Some(loop_) = src.loop_ {
            return Err(meta(
                &format!("{ARCH}.loop.core_end"),
                format!(
                    "names a loop [{}, {}) of {} passes: the loopless fixture is cut from a \
                     loopless file",
                    loop_.core_start, loop_.core_end, loop_.passes
                ),
            ));
        }
        check_map(&src, source, &spec.layers)?;
        Ok(Box::new(PlainTables))
    }

    fn check_kinds(
        &self,
        spec: &FixtureSpec,
        fixture: &Split,
        source: &Split,
    ) -> Result<(), FixtureError> {
        check_kinds(spec, fixture, source).map(|_| ())
    }
}

/// A written fixture reads, through the family's own reader, as the source's hyperparameters at
/// the map's layers: the layer count, the per-layer arrays and kinds the map layers' own, the loop
/// the fixture's coordinates (the looped spec) or the source's none (the loopless one), every
/// other fact the source's own, under a map [`check_map`] accepts. Tensor names, dims and types
/// are the plan's, which `verify` holds the file to.
pub fn check_kinds(
    spec: &FixtureSpec,
    fixture: &Split,
    source: &Split,
) -> Result<Hparams, FixtureError> {
    let fx = Hparams::read(fixture)?;
    let src = Hparams::read(source)?;
    check_map(&src, source, &spec.layers)?;
    let at = |v: &[usize]| spec.layers.iter().map(|&l| v[l]).collect::<Vec<_>>();
    let loop_ = match (src.loop_, fixture_core(&src, &spec.layers)) {
        (None, _) => None,
        (Some(src), Ok((start, end))) => Some(Loop {
            core_start: start,
            core_end: end,
            passes: src.passes,
            gate_rank: src.gate_rank,
        }),
        (Some(_), Err(e)) => return Err(e),
    };
    let n_blocks = models::visits(spec.layers.len(), loop_.map(loop_of)).len();
    let want = Hparams {
        n_blocks,
        n_layer: spec.layers.len(),
        n_head: at(&src.n_head),
        n_head_kv: at(&src.n_head_kv),
        n_used: at(&src.n_used),
        // The family's tables narrow every layer's experts to the fixture's ff.
        expert_ff: vec![FIXTURE_FF as usize; spec.layers.len()],
        windowed: spec.layers.iter().map(|&l| src.windowed[l]).collect(),
        loop_,
        ..src.clone()
    };
    if fx != want {
        return Err(FixtureError::Mismatch {
            what: "the hyperparameters".into(),
            detail: format!("{fx:?}, the source's at the map {want:?}"),
        });
    }
    Ok(fx)
}

/// The description's loop of a file's ([`LoopTables::read`] builds the same value).
fn loop_of(loop_: Loop) -> LoopSpec {
    LoopSpec {
        core_start: loop_.core_start as u32,
        core_end: loop_.core_end as u32,
        passes: loop_.passes as u32,
        gate_rank: loop_.gate_rank as u32,
    }
}

#[cfg(test)]
mod tests {
    use gguf::Split;

    use super::super::spec;
    use super::{
        CARD_BUDGET, FIXTURE_FF, LAYER_MAP, PLAIN_LAYER_MAP, check_kinds, plain_spec, spec,
    };
    use crate::arch::synthetic::{Ty, V, header_typed};
    use crate::fixture::{self, FixtureSpec};

    /// The Mini's shape at the table's widths: 24 stored layers, the core [6, 18) of three
    /// passes, the windowless layers at 5, 11, 17 and 23, 8 experts of 64, the padded vocabulary
    /// of 4 cut to the writer's 2.
    const E: u64 = 128;
    const HEADS: u64 = 4;
    const KV: u64 = 2;
    const D: u64 = 16;
    const REL: u64 = 4;
    const EXTENT: u64 = 32;
    const SWA: u64 = 16;
    const CONV: u64 = 4;
    const BLOCK: u64 = 6;
    const N_EXP: u64 = 8;
    const EFF: u64 = 64;
    const RANK: u64 = 8;
    const CORE: [usize; 2] = [6, 18];
    const PASSES: usize = 3;
    const LAYERS: usize = 24;
    const VISITS: usize = 48;

    /// The stored layer of visit `v` (the reference's `loop_layer`).
    fn layer_of(v: usize) -> usize {
        match v {
            v if v < CORE[1] => v,
            v if v < CORE[1] + (PASSES - 1) * (CORE[1] - CORE[0]) => {
                CORE[0] + (v - CORE[0]) % (CORE[1] - CORE[0])
            }
            v => v - (PASSES - 1) * (CORE[1] - CORE[0]),
        }
    }

    /// The keys of a Mini-shaped file: the per-visit arrays one value a visit, the loop's keys,
    /// the rest scalar. `looped` names whether the loop keys ride along.
    fn mini_keys(looped: bool) -> Vec<(&'static str, V)> {
        let visits = if looped { VISITS } else { 48 };
        let heads = (0..visits).map(|_| HEADS as i32).collect();
        let kv = (0..visits).map(|_| KV as i32).collect();
        let mut kvs: Vec<(&'static str, V)> = vec![
            ("block_count", V::U32(VISITS as u32)),
            ("context_length", V::U32(4096)),
            ("embedding_length", V::U32(E as u32)),
            ("vocab_size", V::U32(4)),
            ("vocab_size_unpadded", V::U32(2)),
            ("attention.head_count", V::I32s(heads)),
            ("attention.head_count_kv", V::I32s(kv)),
            ("attention.key_length", V::U32(D as u32)),
            ("attention.value_length", V::U32(D as u32)),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-6)),
            ("attention.sliding_window", V::U32(SWA as u32)),
            (
                "attention.sliding_window_pattern",
                V::Bools((0..visits).map(|v| layer_of(v) % 6 != 5).collect()),
            ),
            ("attention.relative_length", V::U32(REL as u32)),
            ("attention.relative_extent", V::U32(EXTENT as u32)),
            ("attention.qk_clip", V::F32(22.627_417)),
            ("attention.relative_clip", V::F32(8.0)),
            ("attention.log_scale_floor", V::U32(8192)),
            ("attention.log_scale_alpha", V::F32(0.1)),
            ("attn_res.block_size", V::U32(BLOCK as u32)),
            ("shortconv.l_cache", V::U32(CONV as u32)),
            ("expert_count", V::U32(N_EXP as u32)),
            ("expert_used_count", V::U32(2)),
            ("expert_feed_forward_length", V::U32(EFF as u32)),
            ("expert_weights_scale", V::F32(8.0)),
            ("final_logit_softcapping", V::F32(30.0)),
            ("activation.situ_beta", V::F32(4.0)),
            ("activation.situ_linear_beta", V::F32(25.0)),
        ];
        if looped {
            kvs.extend([
                ("loop.core_start", V::U32(CORE[0] as u32)),
                ("loop.core_end", V::U32(CORE[1] as u32)),
                ("num_loops", V::U32(PASSES as u32)),
                ("loop.gate_rank", V::U32(RANK as u32)),
            ]);
        }
        kvs
    }

    /// One stored layer's tensors at the table's widths, the routed stacks NVFP4 and the
    /// projections Q8_0, `looped` naming whether the core layers carry the controllers.
    fn mini_tensors(looped: bool) -> Vec<(String, Vec<u64>, Ty)> {
        mini_tensors_of(looped, LAYERS)
    }

    /// [`mini_tensors`] over `n` stored layers (a loopless Small-shaped source's 48).
    fn mini_tensors_of(looped: bool, n: usize) -> Vec<(String, Vec<u64>, Ty)> {
        let mut t: Vec<(String, Vec<u64>, Ty)> = vec![
            ("token_embd.weight".into(), vec![E, 4], Ty::Q8_0),
            ("token_embd_norm.weight".into(), vec![E], Ty::F32),
            ("output.weight".into(), vec![E, 4], Ty::Q8_0),
            ("output_norm.weight".into(), vec![E], Ty::F32),
            ("output_res_query.weight".into(), vec![E], Ty::F32),
        ];
        if looped {
            t.push(("loop_res_query.weight".into(), vec![E, 24], Ty::F32));
            t.push(("loop_kv_res_query.weight".into(), vec![E, 24], Ty::F32));
        }
        for l in 0..n {
            let windowless = l % 6 == 5;
            let mut layer: Vec<(String, Vec<u64>, Ty)> = [
                ("attn_q.weight", vec![E, D * HEADS], Ty::Q8_0),
                ("attn_k.weight", vec![E, D * KV], Ty::Q8_0),
                ("attn_v.weight", vec![E, D * KV], Ty::Q8_0),
                ("attn_output.weight", vec![D * HEADS, E], Ty::Q8_0),
                ("attn_gate.weight", vec![E, D * HEADS], Ty::Q8_0),
                ("attn_q_clip.weight", vec![D, HEADS], Ty::F32),
                ("attn_sink_proj.weight", vec![E, HEADS], Ty::Bf16),
                ("attn_rel.weight", vec![E, REL * HEADS], Ty::Q8_0),
                (
                    "attn_rel_bias.weight",
                    vec![REL, if windowless { EXTENT } else { SWA }],
                    Ty::F32,
                ),
                ("attn_rel_clip.weight", vec![REL, HEADS], Ty::F32),
                ("attn_k_conv.weight", vec![CONV, D * KV], Ty::F32),
                ("attn_v_conv.weight", vec![CONV, D * KV], Ty::F32),
                ("attn_post_conv.weight", vec![CONV, E], Ty::F32),
                ("ffn_post_conv.weight", vec![CONV, E], Ty::F32),
            ]
            .into_iter()
            .map(|(n, d, ty)| (format!("blk.{l}.{n}"), d, ty))
            .collect();
            if windowless {
                layer.extend(
                    [
                        ("attn_q_inp_norm.weight", vec![E]),
                        ("attn_k_inp_norm.weight", vec![E]),
                        ("attn_v_inp_norm.weight", vec![E]),
                        ("attn_k_conv_proj.weight", vec![E, CONV * KV]),
                        ("attn_v_conv_proj.weight", vec![E, CONV * KV]),
                    ]
                    .into_iter()
                    .map(|(n, d)| (format!("blk.{l}.{n}"), d, Ty::F32)),
                );
            } else {
                layer.push((format!("blk.{l}.attn_norm.weight"), vec![E], Ty::F32));
            }
            if l > 0 {
                layer.push((
                    format!("blk.{l}.attn_res_query.weight"),
                    vec![E, u64::from(windowless) * 2 + 1],
                    Ty::F32,
                ));
            }
            layer.extend(
                [
                    ("ffn_res_query.weight", vec![E], Ty::F32),
                    ("ffn_norm.weight", vec![E], Ty::F32),
                    ("ffn_gate_inp.weight", vec![E, N_EXP], Ty::F32),
                    ("exp_probs_b.bias", vec![N_EXP], Ty::F32),
                    (
                        "ffn_gate_up_exps.weight",
                        vec![E, 2 * EFF, N_EXP],
                        Ty::Nvfp4,
                    ),
                    ("ffn_down_exps.weight", vec![EFF, E, N_EXP], Ty::Nvfp4),
                ]
                .into_iter()
                .map(|(n, d, ty)| (format!("blk.{l}.{n}"), d, ty)),
            );
            if looped && (CORE[0]..CORE[1]).contains(&l) {
                layer.extend(
                    ["attn", "k", "v", "mlp"]
                        .into_iter()
                        .flat_map(|m| ["down", "gain", "forget"].map(move |p| (m, p)))
                        .map(|(m, p)| {
                            let dims = if p == "down" {
                                vec![2 * E, RANK]
                            } else {
                                vec![RANK, E]
                            };
                            (format!("blk.{l}.loop_{m}_{p}.weight"), dims, Ty::F32)
                        }),
                );
            }
            t.extend(layer);
        }
        t
    }

    /// A synthetic Mini-shaped source (or a loopless Small-shaped one at the same widths, `looped`
    /// false: 48 stored layers, no loop keys, the windowless layers every sixth from 5).
    fn source(tag: &str, looped: bool) -> Split {
        let global = [
            ("tokenizer.ggml.pre", V::Str("inkling2")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = if looped {
            header_typed(
                tag,
                "inkling2",
                &mini_keys(true),
                &global,
                &mini_tensors(true),
            )
        } else {
            let mut kv = mini_keys(false);
            kv.retain(|(k, _)| *k != "block_count");
            kv.push(("block_count", V::U32(48)));
            kv.retain(|(k, _)| *k != "attention.sliding_window_pattern");
            kv.push((
                "attention.sliding_window_pattern",
                V::Bools((0..48).map(|l| l % 6 != 5).collect()),
            ));
            header_typed(tag, "inkling2", &kv, &global, &mini_tensors_of(false, 48))
        };
        let split = Split::open(&path).expect("the synthetic source opens");
        let _ = std::fs::remove_file(&path);
        split
    }

    /// The fixture of `spec_` cut from `source`, as a `Split` holding no tensor's bytes.
    fn fixture_of(spec_: &FixtureSpec, source: &Split, tag: &str) -> Split {
        let plan = fixture::plan(spec_, source, None, &spec_.options()).expect("the plan");
        let layouts = plan.target.layouts().expect("the plan lays out");
        let path = std::env::temp_dir().join(format!(
            "bloomery-inkling2-fixture-{tag}-{}.gguf",
            std::process::id()
        ));
        let split = gguf::Split::header_only(path.clone(), &layouts[0].1)
            .expect("the fixture's header opens");
        let _ = std::fs::remove_file(&path);
        split
    }

    /// The looped fixture reads back through the family's own reader as the source's facts at the
    /// map: 13 stored layers of 25 visits (the core [6, 12) three passes over one block), the
    /// per-visit arrays 25 values agreeing across the core's repeats, the loop queries narrowed
    /// to the kept core's 12 rows, the experts narrowed to [`FIXTURE_FF`] and every other fact the
    /// source's own — [`check_kinds`] and the description both. Mutants: the loop's core left at
    /// the source's coordinates (the reader refuses it by name), a visit's array value taken at
    /// its own visit instead of its layer's first.
    #[test]
    fn the_looped_fixture_reads_back() {
        let spec_ = spec();
        let src = source("inkling2-fx-src", true);
        let fx = fixture_of(&spec_, &src, "looped");
        let hp = check_kinds(&spec_, &fx, &src).expect("the fixture reads as the source's map");
        assert_eq!((hp.n_blocks, hp.n_layer), (25, 13));
        assert_eq!(
            hp.loop_.map(|l| (l.core_start, l.core_end, l.passes)),
            Some((6, 12, 3))
        );
        assert_eq!(LAYER_MAP.len(), 13);
        assert_eq!(
            hp.windowed,
            LAYER_MAP.iter().map(|&l| l % 6 != 5).collect::<Vec<_>>()
        );
        let read = spec::read(&fx).expect("the description reads");
        let s = &read.spec;
        assert_eq!(
            s.loop_,
            Some(models::LoopSpec {
                core_start: 6,
                core_end: 12,
                passes: 3,
                gate_rank: RANK as u32,
            })
        );
        assert_eq!(s.visits().len(), 25);
        assert_eq!(s.layers.len(), 13);
        // The message queries narrowed with the core, and the experts narrowed to the fixture's
        // ff: the plan's dims, which the reader holds them to.
        let (_, query) = fx.find("loop_res_query.weight").expect("the query");
        assert_eq!(query.dims, vec![E, 12]);
        let models::Ffn::Moe(moe) = &s.layers[0].ffn else {
            panic!("layer 0 is routed");
        };
        assert_eq!(moe.expert_ff, FIXTURE_FF as u32);
    }

    /// The loopless fixture reads back the same way: five stored layers of five visits (no loop
    /// keys, the pattern one value a layer taken at the map's layers), the fixture a split set
    /// under its shard cap. Mutant: the loopless spec on a looped source is refused by the loop
    /// key's name, and the looped spec on a loopless one by the missing loop.
    #[test]
    fn the_loopless_fixture_reads_back() {
        let spec_ = plain_spec();
        let src = source("inkling2-fx-plain-src", false);
        let fx = fixture_of(&spec_, &src, "plain");
        let hp = check_kinds(&spec_, &fx, &src).expect("the fixture reads as the source's map");
        assert_eq!((hp.n_blocks, hp.n_layer, hp.loop_), (5, 5, None));
        assert_eq!(PLAIN_LAYER_MAP.len(), 5);
        assert_eq!(
            hp.windowed,
            PLAIN_LAYER_MAP
                .iter()
                .map(|&l| l % 6 != 5)
                .collect::<Vec<_>>()
        );
        let read = spec::read(&fx).expect("the description reads");
        assert_eq!(read.spec.visits().len(), 5);
        assert_eq!(read.spec.loop_, None);
        // The refusals, each by name.
        let err = fixture::plan(
            &spec_,
            &source("inkling2-fx-looped", true),
            None,
            &spec_.options(),
        )
        .expect_err("a looped source is refused");
        assert!(err.to_string().contains("loop.core_end"), "{err}");
        let err = fixture::plan(
            &spec(),
            &source("inkling2-fx-noloop", false),
            None,
            &spec().options(),
        )
        .expect_err("a loopless source is refused");
        assert!(err.to_string().contains("loop.core_end"), "{err}");
    }

    /// The plan's notes name the loop's move, and the recorded card budget is the spec's fixed
    /// one. Mutant: the budget recorded from the plan of a family with none.
    #[test]
    fn the_plan_notes_the_loops_move() {
        let spec_ = spec();
        let src = source("inkling2-fx-notes", true);
        let plan = fixture::plan(&spec_, &src, None, &spec_.options()).expect("the plan");
        assert_eq!(plan.card_budget, CARD_BUDGET);
        assert!(
            plan.notes
                .iter()
                .any(|n| n.contains("[6, 12)") && n.contains("25 visits")),
            "{:?}",
            plan.notes
        );
        // The visits agree: the fixture's per-visit arrays are one value a visit of one layer.
        let kv: Vec<_> = plan
            .target
            .kvs
            .iter()
            .filter(|(k, _)| k.ends_with("attention.head_count"))
            .collect();
        assert_eq!(kv.len(), 1, "one head-count key");
    }
}
