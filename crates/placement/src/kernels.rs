//! The card kernels by weight type, by operation: which entry of the common
//! card kernel crate (`crates/gpu`) runs a file tensor of a given
//! [`GgmlType`]. Each operation reads one column of the type's row
//! ([`gguf::QRow::card`]), where a type's card support is stated with the
//! rest of its facts; `gguf::card` documents each entry's launcher and the
//! layouts the entries read. A tag with no row has no entry.
//!
//! The card bytes of an activation form are here too ([`q8_act_bytes`],
//! [`q8_blocks32_bytes`]) and of one run of the card leg's buffers
//! ([`leg_bytes`]): the forms every card leg, plan and family sizes, in one
//! place.

use gguf::GgmlType;
pub use gguf::card::{
    DownAct, DownEntry, EmbedEntry, GateUpEntry, GemmEntry, HeadEntry, SiteAct, SiteEntry,
};
use models::Role;

pub mod attention;
pub mod ffn;

/// The common card gate·up `_sel` entry that runs a stack of type `ty`.
#[must_use]
pub const fn gate_up(ty: GgmlType) -> Option<GateUpEntry> {
    ty.card().gate_up
}

/// The common card down `_sel` entry that runs a stack of type `ty`, and the
/// activation form it reads.
#[must_use]
pub const fn down(ty: GgmlType) -> Option<(DownEntry, DownAct)> {
    ty.card().down
}

/// The common card `_sel` entry that runs an ungated up stack of type `ty`
/// (an expert with no gate, `h = rule(up · x)`), which is the [`down`] entry
/// of the stack's type whose activation is the 32-value [`DownAct::Q8Blocks32`]:
/// an up is a down-shaped `_sel` that reads column `s` for slot `s`, over
/// columns that repeat the token's row once per slot, so no entry needs a
/// second column rule.
#[must_use]
pub const fn up(ty: GgmlType) -> Option<DownEntry> {
    match down(ty) {
        Some((entry, DownAct::Q8Blocks32)) => Some(entry),
        _ => None,
    }
}

/// The prompt-path GEMM entry that runs a stack of type `ty`.
#[must_use]
pub const fn gemm(ty: GgmlType) -> Option<GemmEntry> {
    ty.card().gemm
}

/// Whether the prompt GEMM `entry` reads its activation in 32-value blocks
/// (`Gemm32Weight`'s entries: Q8_0, Q5_0, Q5_1, IQ4_NL and NVFP4, over a
/// `GemmAct32`), else in the grouped GEMM's q8_1 blocks of 128 (`GemmAct`):
/// the form a prompt walk quantizes a unit's normed rows and SwiGLU rows into
/// for a stack of that type.
#[must_use]
pub const fn gemm_reads_blocks32(entry: GemmEntry) -> bool {
    match entry {
        GemmEntry::Q8_0
        | GemmEntry::Q5_0
        | GemmEntry::Q5_1
        | GemmEntry::Iq4nl
        | GemmEntry::Nvfp4 => true,
        GemmEntry::Q3k
        | GemmEntry::Q4k
        | GemmEntry::Q5k
        | GemmEntry::Q6k
        | GemmEntry::Iq2xxs
        | GemmEntry::Iq2xs
        | GemmEntry::Iq2s
        | GemmEntry::Iq1s
        | GemmEntry::Iq1m
        | GemmEntry::Iq3xxs
        | GemmEntry::Iq3s
        | GemmEntry::Iq4xs => false,
    }
}

/// The dense gemv site launch family for a file tensor of type `ty`.
#[must_use]
pub const fn dense(ty: GgmlType) -> Option<SiteEntry> {
    ty.card().dense
}

/// The lm_head launch family for an `output.weight` of type `ty`.
#[must_use]
pub const fn head(ty: GgmlType) -> Option<HeadEntry> {
    ty.card().head
}

/// The card embedding row lookup for a token table of type `ty`.
#[must_use]
pub const fn embed(ty: GgmlType) -> Option<EmbedEntry> {
    ty.card().embed
}

/// Whether the host tier runs a routed stack of type `ty` whose rows contract `k` values: the
/// host's side of the executor facts the card's entries above state. The host union's passes
/// run fused rows only, so `ty` needs a fused kernel on this machine ([`qdot::supports`], its ISA
/// included) and `k` whole units of its row ([`qdot::fuses`]); `Err` names which is missing.
/// Holding a stack's bytes is no such fact. The plan's split
/// ([`crate::placement::routed_row_on`]) and the host tier's load
/// (`model::ops::host_stack_runs`) ask this one rule.
pub fn host_stack(ty: GgmlType, k: usize) -> Result<(), qdot::QdotError> {
    if !qdot::supports(ty) {
        return Err(qdot::QdotError::UnsupportedType(ty));
    }
    if !qdot::fuses(ty, k) {
        return Err(qdot::QdotError::UnalignedK {
            k,
            gran: qdot::k_granularity(ty),
        });
    }
    Ok(())
}

/// The card bytes of the q8_1 activation of `cols` columns of `k` values
/// ([`DownAct::Q8Act`], [`SiteAct::Q8x128`]): the five planes at `n_sb = k / 256`
/// super-blocks a column — the codes at the q3 (`8 · 64 · ceil(n_sb / 2)`
/// bytes), q4 (`4 · 256 · ceil(n_sb / 4)`) and q6 (`4 · 128 · ceil(n_sb / 2)`)
/// permutations, the sums (`4 · 8 · n_sb`) and the scales (`4 · 2 · n_sb`).
/// A width that is not a whole number of super-blocks counts the super-blocks
/// it holds; a form whose last super-block is padded passes the padded width.
#[must_use]
pub const fn q8_act_bytes(cols: u64, k: u64) -> u64 {
    match try_q8_act_bytes(cols, k) {
        Some(bytes) => bytes,
        None => panic!("q8_act_bytes: activation bytes overflow u64"),
    }
}

/// [`q8_act_bytes`] with overflow returned to the placement boundary.
#[must_use]
pub(crate) const fn try_q8_act_bytes(cols: u64, k: u64) -> Option<u64> {
    let n_sb = (k / 256) as u128;
    // A u64 width's per-column count fits u128; its column product may not.
    let per_col = 512 * n_sb.div_ceil(2)
        + 1024 * n_sb.div_ceil(4)
        + 512 * n_sb.div_ceil(2)
        + 32 * n_sb
        + 8 * n_sb;
    let Some(bytes) = per_col.checked_mul(cols as u128) else {
        return None;
    };
    if bytes > u64::MAX as u128 {
        None
    } else {
        Some(bytes as u64)
    }
}

/// The card bytes of the 32-value q8_1 activation of `cols` columns of `k`
/// values ([`DownAct::Q8Blocks32`], `Q8Blocks32` in `crates/gpu/src/q5.rs`):
/// per column `256 · ceil(k_blocks / 32)` code words and `k_blocks` block sums
/// and scales, four bytes each, `k_blocks = k / 32`.
#[must_use]
pub const fn q8_blocks32_bytes(cols: u64, k: u64) -> u64 {
    match try_q8_blocks32_bytes(cols, k) {
        Some(bytes) => bytes,
        None => panic!("q8_blocks32_bytes: activation bytes overflow u64"),
    }
}

/// [`q8_blocks32_bytes`] with overflow returned to the placement boundary.
#[must_use]
const fn try_q8_blocks32_bytes(cols: u64, k: u64) -> Option<u64> {
    let k_blocks = (k / 32) as u128;
    let per_col = 4 * (256 * k_blocks.div_ceil(32) + 2 * k_blocks);
    let Some(bytes) = per_col.checked_mul(cols as u128) else {
        return None;
    };
    if bytes > u64::MAX as u128 {
        None
    } else {
        Some(bytes as u64)
    }
}

/// The down activation forms ([`DownAct`]) a card leg's buffers hold: the
/// forms the downs of the layers a load runs on them read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DownForms {
    q8_act: bool,
    blocks32: bool,
}

impl DownForms {
    /// No form: a load with no layer of the kind.
    pub const NONE: DownForms = DownForms {
        q8_act: false,
        blocks32: false,
    };

    /// The one form `act`.
    #[must_use]
    pub const fn of(act: DownAct) -> DownForms {
        DownForms::NONE.with(act)
    }

    /// These forms and `act`.
    #[must_use]
    pub const fn with(self, act: DownAct) -> DownForms {
        match act {
            DownAct::Q8Act => DownForms {
                q8_act: true,
                ..self
            },
            DownAct::Q8Blocks32 => DownForms {
                blocks32: true,
                ..self
            },
        }
    }

    /// Whether `act` is one of the forms.
    #[must_use]
    pub const fn holds(self, act: DownAct) -> bool {
        match act {
            DownAct::Q8Act => self.q8_act,
            DownAct::Q8Blocks32 => self.blocks32,
        }
    }
}

/// The card bytes of one run of the card leg's buffers
/// (`bloomery_gpu::card_leg::LegBufs`) for up to `cap` tokens of `n` values,
/// `slots_per_col` slots a token, through experts of `ff` rows, the slots'
/// columns held in each form of `downs`: the q8_1 of the tokens' rows (a
/// column a token), the gate·up's rows (`ff` values a slot), the slots'
/// columns in each form and the down's rows (`n` values a slot). The one
/// formula of those bytes: the leg's buffers and every plan that reserves
/// them read it.
#[must_use]
pub const fn leg_bytes(cap: u64, slots_per_col: u64, n: u64, ff: u64, downs: DownForms) -> u64 {
    match try_leg_bytes(cap, slots_per_col, n, ff, downs) {
        Some(bytes) => bytes,
        None => panic!("leg_bytes: scratch bytes overflow u64"),
    }
}

/// [`leg_bytes`] with overflow returned to the placement boundary. This
/// counts the same whole blocks and padded planes as the infallible sizing.
#[must_use]
pub(crate) const fn try_leg_bytes(
    cap: u64,
    slots_per_col: u64,
    n: u64,
    ff: u64,
    downs: DownForms,
) -> Option<u64> {
    let Some(slots) = cap.checked_mul(slots_per_col) else {
        return None;
    };
    let Some(row_bytes) = slots.checked_mul(4) else {
        return None;
    };
    let Some(h) = row_bytes.checked_mul(ff) else {
        return None;
    };
    let Some(down) = row_bytes.checked_mul(n) else {
        return None;
    };
    let Some(input) = try_q8_act_bytes(cap, n) else {
        return None;
    };
    let q8 = if downs.q8_act {
        let Some(bytes) = try_q8_act_bytes(slots, ff) else {
            return None;
        };
        bytes
    } else {
        0
    };
    let b32 = if downs.blocks32 {
        let Some(bytes) = try_q8_blocks32_bytes(slots, ff) else {
            return None;
        };
        bytes
    } else {
        0
    };
    let bytes = input as u128 + h as u128 + q8 as u128 + b32 as u128 + down as u128;
    if bytes > u64::MAX as u128 {
        None
    } else {
        Some(bytes as u64)
    }
}

/// Whether the common card leg (`crates/gpu/src/card_leg.rs`) launches the
/// gate·up entry of a routed gate or up stack of type `ty` in the file's
/// blocks ([`crate::placement::CardFormat::KQuant`]): the table's [`gate_up`]
/// row, for the entries whose launchers read a `Q8Act` of more columns than a
/// run uses. The Q8_0 and Q6_K rows have launchers of that shape and arms in the
/// leg, but only programs whose selected gate·up policy admits them read those
/// arms ([`crate::accepts::prompt_leg_gate_up`]).
/// A stack in its planes is [`card_planes_runs`]'.
#[must_use]
pub const fn card_gate_up_runs(ty: GgmlType) -> bool {
    matches!(
        gate_up(ty),
        Some(
            GateUpEntry::Q4k
                | GateUpEntry::Q5k
                | GateUpEntry::Mxfp4
                | GateUpEntry::Iq2xxs
                | GateUpEntry::Iq2xs
                | GateUpEntry::Iq2s
                | GateUpEntry::Iq1s
                | GateUpEntry::Iq1m
                | GateUpEntry::Iq3xxs
                | GateUpEntry::Iq3s
                | GateUpEntry::Iq4xs
        )
    )
}

/// Whether the common card leg launches the down entry of a routed down stack
/// of type `ty` in the file's blocks: the table's [`down`] row, for the entries whose launchers take
/// an activation of more columns than a run uses, with the form the leg
/// quantizes for them: the 128-value [`DownAct::Q8Act`] (Q4_K, Q5_K, Q6_K,
/// MXFP4, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS) or the 32-value [`DownAct::Q8Blocks32`]
/// (Q5_1, Q8_0, IQ4_NL). A Q6_K down reads [`DownAct::Q8Act`], the form the
/// common families' down buffers hold. The Q5_0 row reads the packed layout,
/// not the file's blocks.
#[must_use]
pub const fn card_down_runs(ty: GgmlType) -> bool {
    matches!(
        down(ty),
        Some((
            DownEntry::Q4k
                | DownEntry::Q5k
                | DownEntry::Q6k
                | DownEntry::Mxfp4
                | DownEntry::Iq2xxs
                | DownEntry::Iq2xs
                | DownEntry::Iq2s
                | DownEntry::Iq1s
                | DownEntry::Iq1m
                | DownEntry::Iq3xxs
                | DownEntry::Iq3s
                | DownEntry::Iq4xs,
            DownAct::Q8Act
        )) | Some((
            DownEntry::Q5_1 | DownEntry::Q8_0 | DownEntry::Iq4nl,
            DownAct::Q8Blocks32
        ))
    )
}

/// Whether the common card leg's relu² form launches the 32-value-block `_sel`
/// of a routed stack of type `ty`, in either role: the [`up`] entry of an
/// ungated up stack, or the [`down`] entry of the same form: Q5_1, Q8_0 and
/// IQ4_NL (the 32-value downs of [`card_down_runs`], stacks in the file's
/// blocks) and Q5_0, whose stack is in the packed layout its entry reads
/// ([`crate::placement::CardFormat::Q5_0`], `accepts::leg_card_format`).
#[must_use]
pub const fn card_sel32_runs(ty: GgmlType) -> bool {
    matches!(
        up(ty),
        Some(DownEntry::Q5_0 | DownEntry::Q5_1 | DownEntry::Q8_0 | DownEntry::Iq4nl)
    )
}

/// Whether a type runs in every role the common leg launches
/// ([`card_gate_up_runs`] and [`card_down_runs`]). The plans do not read
/// this: each role has its own arm, so a Q6_K down (in [`card_down_runs`],
/// not here) is on the card beside a gate and up [`card_gate_up_runs`] admits.
#[must_use]
pub const fn card_runs(ty: GgmlType) -> bool {
    card_gate_up_runs(ty) && card_down_runs(ty)
}

/// Whether the common card leg runs a routed down stack of `ty` over the
/// 128-value [`DownAct::Q8Act`]: the one form the mimo2 and solar2 bodies'
/// leg buffers and an expert tier card's leg hold, so their plans place a
/// down only by this. Q6_K is one; the 32-value downs (Q5_1, Q8_0, IQ4_NL)
/// are not.
#[must_use]
pub const fn card_q8act_down_runs(ty: GgmlType) -> bool {
    card_down_runs(ty) && matches!(down(ty), Some((_, DownAct::Q8Act)))
}

/// Whether the common card leg runs a routed stack of `ty` in its planes
/// ([`crate::placement::CardFormat::Nvfp4Planes`], a tensor scale an expert)
/// in every role: NVFP4, whose gate·up reads a `Q8Act` and whose down reads
/// the 32-value [`DownAct::Q8Blocks32`]. Its own rule beside [`card_runs`],
/// whose stacks are the file's blocks.
#[must_use]
pub const fn card_planes_runs(ty: GgmlType) -> bool {
    matches!(gate_up(ty), Some(GateUpEntry::Nvfp4))
        && matches!(down(ty), Some((DownEntry::Nvfp4, DownAct::Q8Blocks32)))
}

/// Whether the common card leg's relu² form ([`card_sel32_runs`]'s) launches a routed stack of
/// `ty`, in either role: the 32-value-block types and NVFP4 in its planes ([`card_planes_runs`]:
/// NVFP4 and a checkpoint folder's NVFP4_ROWS), whose up and down are one `_sel` over the
/// 32-value activation (`nvfp4_gemv_sel32`: a row width of any multiple of 64). The rule of
/// every caller that picks a relu² leg; [`card_sel32_runs`] keeps its meaning, the types whose
/// `_sel` the 32-value-block entries launch.
#[must_use]
pub const fn card_relu2_runs(ty: GgmlType) -> bool {
    card_sel32_runs(ty) || card_planes_runs(ty)
}

/// The K unit a card tensor's layout and kernel keep whole, beyond its format block.
/// MXFP4 stacks need 256 values and MXFP8 planes 128. Other entries follow their activation:
/// Q8Act needs 256, Q8Blocks32 needs 32, and f32 has no block. The tensor's role selects the
/// routed down or dense site entry, so NVFP4 needs 32 for routed down and 256 for a dense site.
/// A tensor-parallel part keeps this unit whole ([`crate::placement::tp::card_blocks`]).
#[must_use]
pub const fn card_k_unit(ty: GgmlType, role: Role) -> u64 {
    if matches!(gate_up(ty), Some(GateUpEntry::Mxfp4)) {
        return 256;
    }
    if matches!(dense(ty), Some(SiteEntry::Mxfp8)) {
        return 128;
    }
    match role {
        Role::RoutedExperts => match down(ty) {
            Some((_, DownAct::Q8Act)) => 256,
            Some((_, DownAct::Q8Blocks32)) => 32,
            None => 1,
        },
        _ => match dense(ty) {
            Some(entry) => match entry.reads() {
                SiteAct::Q8x128 => 256,
                SiteAct::Q8x32 | SiteAct::Q8Blocks32 => 32,
                SiteAct::F32 => 1,
            },
            None => 1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::CardFormat;

    /// Plane-padding transitions retain their byte counts, and both sizing
    /// contracts share the last representable column and aggregate boundary.
    #[test]
    fn activation_and_leg_bytes_keep_padding_and_overflow_boundaries() {
        for (k, bytes) in [(1024, 3232), (1280, 5320)] {
            assert_eq!(try_q8_act_bytes(1, k), Some(bytes));
            assert_eq!(q8_act_bytes(1, k), bytes);
        }
        for (k, bytes) in [(1024, 1280), (1056, 2312)] {
            assert_eq!(try_q8_blocks32_bytes(1, k), Some(bytes));
            assert_eq!(q8_blocks32_bytes(1, k), bytes);
        }
        let q8 = DownForms::of(DownAct::Q8Act);
        let b32 = DownForms::of(DownAct::Q8Blocks32);
        for (forms, bytes) in [
            (DownForms::NONE, 4136),
            (q8, 6224),
            (b32, 5224),
            (q8.with(DownAct::Q8Blocks32), 7312),
        ] {
            assert_eq!(try_leg_bytes(1, 1, 256, 256, forms), Some(bytes));
            assert_eq!(leg_bytes(1, 1, 256, 256, forms), bytes);
        }
        assert_eq!(try_leg_bytes(1, 1, 64, 24, DownForms::NONE), Some(352));
        let act_cols = u64::MAX / 2088;
        assert_eq!(try_q8_act_bytes(act_cols, 256), Some(act_cols * 2088));
        assert_eq!(q8_act_bytes(act_cols, 256), act_cols * 2088);
        assert_eq!(try_q8_act_bytes(act_cols + 1, 256), None);
        let block_cols = u64::MAX / 1088;
        assert_eq!(
            try_q8_blocks32_bytes(block_cols, 256),
            Some(block_cols * 1088)
        );
        assert_eq!(q8_blocks32_bytes(block_cols, 256), block_cols * 1088);
        assert_eq!(try_q8_blocks32_bytes(block_cols + 1, 256), None);
        let cap = u64::MAX / 6224;
        assert_eq!(try_leg_bytes(cap, 1, 256, 256, q8), Some(cap * 6224));
        assert_eq!(leg_bytes(cap, 1, 256, 256, q8), cap * 6224);
        assert_eq!(try_leg_bytes(cap + 1, 1, 256, 256, q8), None);
        assert_eq!(try_leg_bytes(1 << 61, 8, 256, 256, q8), None);
    }

    #[test]
    #[should_panic(expected = "leg_bytes: scratch bytes overflow u64")]
    fn infallible_leg_sizing_refuses_overflow() {
        let _ = leg_bytes(1 << 61, 8, 256, 256, DownForms::of(DownAct::Q8Act));
    }

    /// Every `GgmlType` variant in declaration order, then one id the enum
    /// does not name.
    const ALL: [GgmlType; 44] = [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::BF16,
        GgmlType::MXFP4,
        GgmlType::NVFP4,
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_XS,
        GgmlType::IQ3_XXS,
        GgmlType::IQ1_S,
        GgmlType::IQ4_NL,
        GgmlType::IQ3_S,
        GgmlType::IQ2_S,
        GgmlType::IQ4_XS,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
        GgmlType::IQ1_M,
        GgmlType::TQ1_0,
        GgmlType::TQ2_0,
        GgmlType::Q2_0,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q1_0,
        GgmlType::Q8_1,
        GgmlType::Q8_K,
        GgmlType::F8E4M3_ROWS,
        GgmlType::MXFP8,
        GgmlType::NVFP4_ROWS,
        GgmlType::FP8_B128,
        GgmlType::MLX_A4,
        GgmlType::MLX_A8,
        GgmlType::MLX_A4H,
        GgmlType::MLX_A8H,
        GgmlType::Unknown(99),
    ];

    /// A variant's place in [`ALL`]. Exhaustive: a type added to `GgmlType`
    /// does not compile here until it is placed.
    const fn place(ty: GgmlType) -> usize {
        match ty {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q5_0 => 2,
            GgmlType::Q5_1 => 3,
            GgmlType::Q8_0 => 4,
            GgmlType::Q2_K => 5,
            GgmlType::Q3_K => 6,
            GgmlType::Q4_K => 7,
            GgmlType::Q5_K => 8,
            GgmlType::Q6_K => 9,
            GgmlType::BF16 => 10,
            GgmlType::MXFP4 => 11,
            GgmlType::NVFP4 => 12,
            GgmlType::IQ2_XXS => 13,
            GgmlType::IQ2_XS => 14,
            GgmlType::IQ3_XXS => 15,
            GgmlType::IQ1_S => 16,
            GgmlType::IQ4_NL => 17,
            GgmlType::IQ3_S => 18,
            GgmlType::IQ2_S => 19,
            GgmlType::IQ4_XS => 20,
            GgmlType::I8 => 21,
            GgmlType::I16 => 22,
            GgmlType::I32 => 23,
            GgmlType::I64 => 24,
            GgmlType::F64 => 25,
            GgmlType::IQ1_M => 26,
            GgmlType::TQ1_0 => 27,
            GgmlType::TQ2_0 => 28,
            GgmlType::Q2_0 => 29,
            GgmlType::Q4_0 => 30,
            GgmlType::Q4_1 => 31,
            GgmlType::Q1_0 => 32,
            GgmlType::Q8_1 => 33,
            GgmlType::Q8_K => 34,
            GgmlType::F8E4M3_ROWS => 35,
            GgmlType::MXFP8 => 36,
            GgmlType::NVFP4_ROWS => 37,
            GgmlType::FP8_B128 => 38,
            GgmlType::MLX_A4 => 39,
            GgmlType::MLX_A8 => 40,
            GgmlType::MLX_A4H => 41,
            GgmlType::MLX_A8H => 42,
            GgmlType::Unknown(_) => 43,
        }
    }

    /// The `(type, row)` pairs an operation admits over [`ALL`].
    fn rows<T>(op: fn(GgmlType) -> Option<T>) -> Vec<(GgmlType, T)> {
        ALL.iter()
            .filter_map(|&ty| op(ty).map(|row| (ty, row)))
            .collect()
    }

    /// Whether each operation has a row for `ty`, by the operation's name.
    fn has_row_in(ty: GgmlType) -> [(&'static str, bool); 7] {
        [
            ("gate_up", gate_up(ty).is_some()),
            ("down", down(ty).is_some()),
            ("up", up(ty).is_some()),
            ("gemm", gemm(ty).is_some()),
            ("dense", dense(ty).is_some()),
            ("head", head(ty).is_some()),
            ("embed", embed(ty).is_some()),
        ]
    }

    /// The pin tests list their pairs in [`ALL`]'s order, so the list has to
    /// be every variant, each once.
    #[test]
    fn the_type_list_is_every_variant() {
        for (i, ty) in ALL.iter().enumerate() {
            assert_eq!(place(*ty), i, "{ty:?} is not at its place in ALL");
        }
    }

    // PIN(2026-10-12): the gate·up, down and activation-form pairs and the leg's
    // run sets gain IQ3_S (`iq3_s_gate_up_sel`, `iq3_s_gemv_sel`); every other
    // pair is unchanged.
    // PIN(2026-10-19): the gate·up, down, activation-form and up pairs gain
    // NVFP4 (`nvfp4_gate_up_sel`, `nvfp4_gemv_sel32` over its planes); the
    // file-block run sets (`card_runs` and its parts) are unchanged.
    // PIN(2026-10-23): the gate·up, down, activation-form pairs and the
    // file-block run sets gain IQ2_XXS (`iq2_xxs_gate_up_sel` through the
    // block's shared table, `iq2_xxs_gemv_sel` as its down); every other pair
    // is unchanged.
    // PIN(2026-10-23): the gate·up, down, activation-form and up pairs, the
    // planes run set and the planes layout gain NVFP4_ROWS, which takes
    // NVFP4's routed card columns (a folder's routed stack loads as NVFP4
    // planes); every other pair is unchanged.
    // PIN(2026-10-25): the gate·up, down, activation-form pairs and the
    // file-block run sets gain IQ2_S, IQ1_S and IQ1_M (`iq2_s_gate_up_sel`,
    // `iq1_s_gate_up_sel`, `iq1_m_gate_up_sel` through the block's shared
    // table, and their `_gemv_sel` downs); IQ2_XS also takes the file-block
    // layout and a routed column. Every other pair is unchanged.
    #[test]
    fn gate_up_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(gate_up),
            [
                (GgmlType::Q8_0, GateUpEntry::Q8_0),
                (GgmlType::Q4_K, GateUpEntry::Q4k),
                (GgmlType::Q5_K, GateUpEntry::Q5k),
                (GgmlType::Q6_K, GateUpEntry::Q6k),
                (GgmlType::MXFP4, GateUpEntry::Mxfp4),
                (GgmlType::NVFP4, GateUpEntry::Nvfp4),
                (GgmlType::IQ2_XXS, GateUpEntry::Iq2xxs),
                (GgmlType::IQ2_XS, GateUpEntry::Iq2xs),
                (GgmlType::IQ3_XXS, GateUpEntry::Iq3xxs),
                (GgmlType::IQ1_S, GateUpEntry::Iq1s),
                (GgmlType::IQ3_S, GateUpEntry::Iq3s),
                (GgmlType::IQ2_S, GateUpEntry::Iq2s),
                (GgmlType::IQ4_XS, GateUpEntry::Iq4xs),
                (GgmlType::IQ1_M, GateUpEntry::Iq1m),
                (GgmlType::NVFP4_ROWS, GateUpEntry::Nvfp4),
            ]
        );
    }

    #[test]
    fn down_admits_exactly_the_pinned_pairs() {
        let entries: Vec<_> = ALL
            .iter()
            .filter_map(|&ty| down(ty).map(|(entry, _)| (ty, entry)))
            .collect();
        assert_eq!(
            entries,
            [
                (GgmlType::Q5_0, DownEntry::Q5_0),
                (GgmlType::Q5_1, DownEntry::Q5_1),
                (GgmlType::Q8_0, DownEntry::Q8_0),
                (GgmlType::Q4_K, DownEntry::Q4k),
                (GgmlType::Q5_K, DownEntry::Q5k),
                (GgmlType::Q6_K, DownEntry::Q6k),
                (GgmlType::MXFP4, DownEntry::Mxfp4),
                (GgmlType::NVFP4, DownEntry::Nvfp4),
                (GgmlType::IQ2_XXS, DownEntry::Iq2xxs),
                (GgmlType::IQ2_XS, DownEntry::Iq2xs),
                (GgmlType::IQ3_XXS, DownEntry::Iq3xxs),
                (GgmlType::IQ1_S, DownEntry::Iq1s),
                (GgmlType::IQ4_NL, DownEntry::Iq4nl),
                (GgmlType::IQ3_S, DownEntry::Iq3s),
                (GgmlType::IQ2_S, DownEntry::Iq2s),
                (GgmlType::IQ4_XS, DownEntry::Iq4xs),
                (GgmlType::IQ1_M, DownEntry::Iq1m),
                (GgmlType::NVFP4_ROWS, DownEntry::Nvfp4),
            ]
        );
    }

    /// The K-quant, MXFP4 and i-quant (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S,
    /// IQ2_S, IQ4_XS, IQ1_M) downs read the 128-value `Q8Act`; every other down reads the 32-value
    /// `Q8Blocks32`.
    #[test]
    fn down_reads_the_activation_form_of_its_entry() {
        let forms: Vec<_> = ALL
            .iter()
            .filter_map(|&ty| down(ty).map(|(_, act)| (ty, act)))
            .collect();
        assert_eq!(
            forms,
            [
                (GgmlType::Q5_0, DownAct::Q8Blocks32),
                (GgmlType::Q5_1, DownAct::Q8Blocks32),
                (GgmlType::Q8_0, DownAct::Q8Blocks32),
                (GgmlType::Q4_K, DownAct::Q8Act),
                (GgmlType::Q5_K, DownAct::Q8Act),
                (GgmlType::Q6_K, DownAct::Q8Act),
                (GgmlType::MXFP4, DownAct::Q8Act),
                (GgmlType::NVFP4, DownAct::Q8Blocks32),
                (GgmlType::IQ2_XXS, DownAct::Q8Act),
                (GgmlType::IQ2_XS, DownAct::Q8Act),
                (GgmlType::IQ3_XXS, DownAct::Q8Act),
                (GgmlType::IQ1_S, DownAct::Q8Act),
                (GgmlType::IQ4_NL, DownAct::Q8Blocks32),
                (GgmlType::IQ3_S, DownAct::Q8Act),
                (GgmlType::IQ2_S, DownAct::Q8Act),
                (GgmlType::IQ4_XS, DownAct::Q8Act),
                (GgmlType::IQ1_M, DownAct::Q8Act),
                (GgmlType::NVFP4_ROWS, DownAct::Q8Blocks32),
            ]
        );
    }

    /// An ungated up runs the down entries of the 32-value form, and no other:
    /// the same entry as the type's down, never one of the `Q8Act` entries.
    #[test]
    fn up_is_the_32_value_down_entry() {
        assert_eq!(
            rows(up),
            [
                (GgmlType::Q5_0, DownEntry::Q5_0),
                (GgmlType::Q5_1, DownEntry::Q5_1),
                (GgmlType::Q8_0, DownEntry::Q8_0),
                (GgmlType::NVFP4, DownEntry::Nvfp4),
                (GgmlType::IQ4_NL, DownEntry::Iq4nl),
                (GgmlType::NVFP4_ROWS, DownEntry::Nvfp4),
            ]
        );
        for ty in ALL {
            assert_eq!(
                up(ty),
                down(ty).and_then(|(e, act)| (act == DownAct::Q8Blocks32).then_some(e)),
                "{ty:?}"
            );
        }
        let runs: Vec<GgmlType> = ALL.into_iter().filter(|&ty| card_sel32_runs(ty)).collect();
        assert_eq!(
            runs,
            [
                GgmlType::Q5_0,
                GgmlType::Q5_1,
                GgmlType::Q8_0,
                GgmlType::IQ4_NL
            ]
        );
        // Every type but Q5_0 is a down in the file's blocks; Q5_0's stack is the packed layout, a
        // down the whole-file bodies' leg also runs (`accepts::resident_down_runs`).
        for ty in runs {
            assert_eq!(card_down_runs(ty), ty != GgmlType::Q5_0, "{ty:?}");
        }
    }

    /// The leg runs NVFP4 and NVFP4_ROWS from their planes and no other type;
    /// no type runs both from the file's blocks and from planes.
    #[test]
    fn card_planes_runs_admits_exactly_nvfp4() {
        let runs: Vec<GgmlType> = ALL.into_iter().filter(|&ty| card_planes_runs(ty)).collect();
        assert_eq!(runs, [GgmlType::NVFP4, GgmlType::NVFP4_ROWS]);
        assert_eq!(
            CardFormat::of(GgmlType::NVFP4),
            Some(CardFormat::Nvfp4Planes)
        );
        assert!(!card_runs(GgmlType::NVFP4) && !card_sel32_runs(GgmlType::NVFP4));
    }

    /// The relu² form runs the four 32-value-block types and, beside them, the two NVFP4 planes
    /// types, and no other: exactly the union of [`card_sel32_runs`] and [`card_planes_runs`],
    /// each of which keeps its own set. Mutant: the rule that reads `card_sel32_runs` alone.
    #[test]
    fn card_relu2_runs_admits_nvfp4_planes() {
        let runs: Vec<GgmlType> = ALL.into_iter().filter(|&ty| card_relu2_runs(ty)).collect();
        assert_eq!(
            runs,
            [
                GgmlType::Q5_0,
                GgmlType::Q5_1,
                GgmlType::Q8_0,
                GgmlType::NVFP4,
                GgmlType::IQ4_NL,
                GgmlType::NVFP4_ROWS
            ]
        );
        for ty in ALL {
            assert_eq!(
                card_relu2_runs(ty),
                card_sel32_runs(ty) || card_planes_runs(ty),
                "{ty:?}"
            );
        }
        let sel32: Vec<GgmlType> = ALL.into_iter().filter(|&ty| card_sel32_runs(ty)).collect();
        assert_eq!(
            sel32,
            [
                GgmlType::Q5_0,
                GgmlType::Q5_1,
                GgmlType::Q8_0,
                GgmlType::IQ4_NL
            ]
        );
    }

    /// The GEMM entries that read 32-value blocks are the `Gemm32Weight`
    /// ones (the file's Q8_0, Q5_0, Q5_1 and IQ4_NL blocks, NVFP4's planes); the
    /// grouped GEMM's read q8_1 blocks of 128. A type's GEMM form is one fact:
    /// the table's `gemm` row read through this.
    #[test]
    fn gemm_form_is_the_32_value_entries() {
        let blocks32: Vec<_> = ALL
            .iter()
            .filter(|&&ty| gemm(ty).is_some_and(gemm_reads_blocks32))
            .copied()
            .collect();
        assert_eq!(
            blocks32,
            [
                GgmlType::Q5_0,
                GgmlType::Q5_1,
                GgmlType::Q8_0,
                GgmlType::NVFP4,
                GgmlType::IQ4_NL,
                GgmlType::NVFP4_ROWS
            ]
        );
    }

    // PIN(2026-10-16): the dense pairs gain NVFP4 (`nvfp4_gemv` through
    // `site::kgemv`); every other pair is unchanged.
    // PIN(2026-10-19): and bloomery's own MXFP8 (`mxfp8_gemv` through
    // `site::gemv`); every other pair is unchanged.
    // PIN(2026-10-20): and bloomery's own FP8_B128 (`fp8_b128_gemv` through
    // `site::gemv`); BF16 also uses `bf16_gemv` and `bf16_tile_gemm` over its words.
    // PIN(2026-10-26): IQ4_XS adds the existing selected GEMV as an ordinary site (UTC).
    #[test]
    fn dense_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(dense),
            [
                (GgmlType::F32, SiteEntry::F32),
                (GgmlType::Q5_0, SiteEntry::Q5_0),
                (GgmlType::Q8_0, SiteEntry::Q8_0),
                (GgmlType::Q3_K, SiteEntry::Q3k),
                (GgmlType::Q4_K, SiteEntry::Q4k),
                (GgmlType::Q5_K, SiteEntry::Q5k),
                (GgmlType::Q6_K, SiteEntry::Q6k),
                (GgmlType::BF16, SiteEntry::Bf16),
                (GgmlType::NVFP4, SiteEntry::Nvfp4),
                (GgmlType::IQ4_XS, SiteEntry::Iq4xs),
                (GgmlType::MXFP8, SiteEntry::Mxfp8),
                (GgmlType::FP8_B128, SiteEntry::Fp8B128),
            ]
        );
    }

    // PIN(2026-10-20): the GEMM pairs gain NVFP4 (`gemm_nvfp4p` over its
    // planes); every other pair is unchanged.
    // PIN(2026-10-21): and IQ2_XXS (`gemm_iq2xxs`, which `GemmWeight` already
    // decoded and the latent MoE body launches); every other pair is unchanged.
    // PIN(2026-10-22): and Q5_0 (`gemm_q5_0p` over a stack's `pack_q5_0` words,
    // which Raon-1's Q5_0 down stacks' prompt route reads), and IQ3_S
    // (`gemm_iq3s`, over the file's blocks); every other pair is unchanged.
    // PIN(2026-10-23): and NVFP4_ROWS (`gemm_nvfp4p` over NVFP4's planes, the
    // routed card columns it takes); every other pair is unchanged.
    // PIN(2026-10-25): and IQ2_XS, IQ2_S, IQ1_S and IQ1_M (`gemm_iq2xs`,
    // `gemm_iq2s`, `gemm_iq1s`, `gemm_iq1m` over the file's blocks); every
    // other pair is unchanged.
    #[test]
    fn gemm_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(gemm),
            [
                (GgmlType::Q5_0, GemmEntry::Q5_0),
                (GgmlType::Q5_1, GemmEntry::Q5_1),
                (GgmlType::Q8_0, GemmEntry::Q8_0),
                (GgmlType::Q3_K, GemmEntry::Q3k),
                (GgmlType::Q4_K, GemmEntry::Q4k),
                (GgmlType::Q5_K, GemmEntry::Q5k),
                (GgmlType::Q6_K, GemmEntry::Q6k),
                (GgmlType::NVFP4, GemmEntry::Nvfp4),
                (GgmlType::IQ2_XXS, GemmEntry::Iq2xxs),
                (GgmlType::IQ2_XS, GemmEntry::Iq2xs),
                (GgmlType::IQ3_XXS, GemmEntry::Iq3xxs),
                (GgmlType::IQ1_S, GemmEntry::Iq1s),
                (GgmlType::IQ4_NL, GemmEntry::Iq4nl),
                (GgmlType::IQ3_S, GemmEntry::Iq3s),
                (GgmlType::IQ2_S, GemmEntry::Iq2s),
                (GgmlType::IQ4_XS, GemmEntry::Iq4xs),
                (GgmlType::IQ1_M, GemmEntry::Iq1m),
                (GgmlType::NVFP4_ROWS, GemmEntry::Nvfp4),
            ]
        );
    }

    /// A dense entry is named by one type's row, so its file type is that
    /// row's type.
    #[test]
    fn a_dense_entry_names_the_type_of_its_row() {
        for (ty, entry) in rows(dense) {
            assert_eq!(entry.ty(), ty, "{entry:?}");
        }
    }

    // PIN(2026-10-20): the head and embedding pairs gain BF16 (`bf16_gemv`
    // over the head's words, `embed_rows_bf16`); every other pair is unchanged.
    // PIN(2026-10-25): the head gains Q5_K (the `_sel` Q5_K gemv over the matrix
    // as one expert, one row); every other pair is unchanged.
    // PIN(2026-10-28): the head gains NVFP4 (`site::kgemv`, or the one-expert `_sel32` gemv at a
    // width of no super-block, over the matrix's planes and the tensor's scale); every other pair
    // is unchanged. FAIL-first: the type's row with no head entry leaves the list one pair short.
    #[test]
    fn head_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(head),
            [
                (GgmlType::Q8_0, HeadEntry::Q8_0),
                (GgmlType::Q4_K, HeadEntry::Q4k),
                (GgmlType::Q5_K, HeadEntry::Q5k),
                (GgmlType::Q6_K, HeadEntry::Q6k),
                (GgmlType::BF16, HeadEntry::Bf16),
                (GgmlType::NVFP4, HeadEntry::Nvfp4),
            ]
        );
    }

    #[test]
    fn embed_admits_exactly_the_pinned_pairs() {
        assert_eq!(
            rows(embed),
            [
                (GgmlType::Q8_0, EmbedEntry::Q8_0),
                (GgmlType::Q3_K, EmbedEntry::Q3k),
                (GgmlType::Q4_K, EmbedEntry::Q4k),
                (GgmlType::Q5_K, EmbedEntry::Q5k),
                (GgmlType::Q6_K, EmbedEntry::Q6k),
                (GgmlType::BF16, EmbedEntry::Bf16),
            ]
        );
    }

    /// The layout of each file type and its default routed layout: Q5_K,
    /// MXFP8 and IQ4_XS require a program-specific routed rule.
    // PIN(2026-10-23): NVFP4_ROWS loads as NVFP4's planes (its routed card
    // columns); every other pair is unchanged.
    // PIN(2026-10-25): IQ2_XS, IQ1_S, IQ2_S and IQ1_M load as the file's
    // blocks, as IQ2_XXS does; every other pair is unchanged.
    // PIN(2026-10-26): ordinary IQ4_XS matrices retain their file blocks (UTC).
    #[test]
    fn the_card_layouts_are_the_pinned_pairs() {
        assert_eq!(
            rows(CardFormat::of),
            [
                (GgmlType::F32, CardFormat::F32),
                (GgmlType::F16, CardFormat::Bf16AsF32),
                (GgmlType::Q5_0, CardFormat::Q5_0),
                (GgmlType::Q5_1, CardFormat::Q5_1),
                (GgmlType::Q8_0, CardFormat::Q8_0Planes),
                (GgmlType::Q3_K, CardFormat::KQuant),
                (GgmlType::Q4_K, CardFormat::KQuant),
                (GgmlType::Q5_K, CardFormat::KQuant),
                (GgmlType::Q6_K, CardFormat::KQuant),
                (GgmlType::BF16, CardFormat::Bf16AsF32),
                (GgmlType::NVFP4, CardFormat::Nvfp4Planes),
                (GgmlType::IQ2_XXS, CardFormat::KQuant),
                (GgmlType::IQ2_XS, CardFormat::KQuant),
                (GgmlType::IQ1_S, CardFormat::KQuant),
                (GgmlType::IQ2_S, CardFormat::KQuant),
                (GgmlType::IQ4_XS, CardFormat::KQuant),
                (GgmlType::IQ1_M, CardFormat::KQuant),
                (GgmlType::MXFP8, CardFormat::Mxfp8Planes),
                (GgmlType::NVFP4_ROWS, CardFormat::Nvfp4Planes),
                (GgmlType::FP8_B128, CardFormat::Fp8B128Planes),
            ]
        );
        for ty in ALL {
            let routed = CardFormat::of(ty)
                .filter(|_| !matches!(ty, GgmlType::Q5_K | GgmlType::MXFP8 | GgmlType::IQ4_XS));
            assert_eq!(CardFormat::of_routed(ty), routed, "{ty:?}");
        }
    }

    /// The types each layout holds: the types it is the load layout of, the
    /// K-quants as decoded f32, bf16 as raw words, and in the file's blocks
    /// every type a routed entry reads in them.
    #[test]
    fn each_layout_holds_exactly_the_pinned_types() {
        let held = |f: CardFormat| -> Vec<GgmlType> {
            ALL.into_iter().filter(|&ty| f.holds(ty)).collect()
        };
        assert_eq!(
            held(CardFormat::KQuant),
            [
                GgmlType::Q5_1,
                GgmlType::Q8_0,
                GgmlType::Q3_K,
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::Q6_K,
                GgmlType::MXFP4,
                GgmlType::IQ2_XXS,
                GgmlType::IQ2_XS,
                GgmlType::IQ3_XXS,
                GgmlType::IQ1_S,
                GgmlType::IQ4_NL,
                GgmlType::IQ3_S,
                GgmlType::IQ2_S,
                GgmlType::IQ4_XS,
                GgmlType::IQ1_M,
            ]
        );
        // PIN(2026-10-25): and the four i-quants whose layout is now the file's
        // blocks.
        assert_eq!(
            held(CardFormat::Bf16AsF32),
            [
                GgmlType::F16,
                GgmlType::Q3_K,
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::Q6_K,
                GgmlType::BF16,
                GgmlType::IQ2_XXS,
                GgmlType::IQ2_XS,
                GgmlType::IQ1_S,
                GgmlType::IQ2_S,
                GgmlType::IQ4_XS,
                GgmlType::IQ1_M,
            ]
        );
        // PIN(2026-10-22): independent BF16 heads also occupy MXFP8 planes.
        assert_eq!(
            held(CardFormat::Mxfp8Planes),
            [GgmlType::BF16, GgmlType::MXFP8]
        );
        // PIN(2026-10-23): NVFP4_ROWS's routed stacks load as NVFP4's planes.
        assert_eq!(
            held(CardFormat::Nvfp4Planes),
            [GgmlType::NVFP4, GgmlType::NVFP4_ROWS]
        );
        for (f, ty) in [
            (CardFormat::Q5_0, GgmlType::Q5_0),
            (CardFormat::Q5_1, GgmlType::Q5_1),
            (CardFormat::Q8_0Planes, GgmlType::Q8_0),
            (CardFormat::Fp8B128Planes, GgmlType::FP8_B128),
            (CardFormat::F32, GgmlType::F32),
            (CardFormat::Bf16Raw, GgmlType::BF16),
        ] {
            assert_eq!(held(f), [ty], "{f:?}");
        }
    }
    /// A card entry reads a layout the card loader can build: every type a
    /// row names has a [`CardFormat`] of its own, or is one of the file-block
    /// types a program's routed stack keeps as `KQuant`.
    #[test]
    fn every_row_type_has_a_card_layout() {
        for ty in ALL {
            for (name, has_row) in has_row_in(ty) {
                assert!(
                    !has_row || CardFormat::of(ty).is_some() || CardFormat::KQuant.holds(ty),
                    "{name} has a row for {ty:?}, which no card layout holds"
                );
            }
        }
    }

    /// The common card leg runs exactly the pinned types, each of which has
    /// both rows: the K-quants, MXFP4 and the i-quants whose raw-block gate·up
    /// and `Q8Act` down entries `iq_sel.rs` launches.
    // PIN(2026-10-23): IQ2_XXS joins `card_runs` with its gate·up and down
    // entries (the PIN above the gate·up pairs); every other type is unchanged.
    // PIN(2026-10-25): and IQ1_S, IQ2_S and IQ1_M, with theirs.
    #[test]
    fn card_runs_admits_exactly_the_pinned_types() {
        let runs: Vec<GgmlType> = ALL.into_iter().filter(|&ty| card_runs(ty)).collect();
        assert_eq!(
            runs,
            [
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::MXFP4,
                GgmlType::IQ2_XXS,
                GgmlType::IQ2_XS,
                GgmlType::IQ3_XXS,
                GgmlType::IQ1_S,
                GgmlType::IQ3_S,
                GgmlType::IQ2_S,
                GgmlType::IQ4_XS,
                GgmlType::IQ1_M,
            ]
        );
        for ty in runs {
            assert!(gate_up(ty).is_some() && down(ty).is_some(), "{ty:?}");
        }
    }

    /// The leg checks each role by its own rule: the gate·up and down sets
    /// differ (Q5_1, Q8_0, IQ4_NL and Q6_K run only as downs; Q8_0's gate·up
    /// row has no arm) and the Q5_0 down row has no arm. A rank's downs are
    /// that same down set.
    // PIN(2026-10-22): Q6_K joined `card_down_runs` (the Q8Act down the leg
    // already launches), so the rank-only extra is empty. It was `[Q6_K]`
    // while the common down set left Q6_K out.
    #[test]
    fn the_leg_admits_each_role_by_its_own_rule() {
        let gate_up_runs: Vec<GgmlType> = ALL
            .into_iter()
            .filter(|&ty| card_gate_up_runs(ty))
            .collect();
        let down_runs: Vec<GgmlType> = ALL.into_iter().filter(|&ty| card_down_runs(ty)).collect();
        assert_eq!(
            gate_up_runs,
            [
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::MXFP4,
                GgmlType::IQ2_XXS,
                GgmlType::IQ2_XS,
                GgmlType::IQ3_XXS,
                GgmlType::IQ1_S,
                GgmlType::IQ3_S,
                GgmlType::IQ2_S,
                GgmlType::IQ4_XS,
                GgmlType::IQ1_M
            ]
        );
        assert_eq!(
            down_runs,
            [
                GgmlType::Q5_1,
                GgmlType::Q8_0,
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::Q6_K,
                GgmlType::MXFP4,
                GgmlType::IQ2_XXS,
                GgmlType::IQ2_XS,
                GgmlType::IQ3_XXS,
                GgmlType::IQ1_S,
                GgmlType::IQ4_NL,
                GgmlType::IQ3_S,
                GgmlType::IQ2_S,
                GgmlType::IQ4_XS,
                GgmlType::IQ1_M
            ]
        );
        assert!(gate_up(GgmlType::Q8_0).is_some() && !card_gate_up_runs(GgmlType::Q8_0));
        assert!(down(GgmlType::Q6_K).is_some() && card_down_runs(GgmlType::Q6_K));
        assert!(down(GgmlType::Q5_0).is_some() && !card_down_runs(GgmlType::Q5_0));
    }

    /// A type id the enum does not name has no row in any operation.
    #[test]
    fn an_unnamed_type_has_no_row() {
        for id in [0, 2, 3, 99, u32::MAX] {
            let ty = GgmlType::Unknown(id);
            assert_eq!(gate_up(ty), None, "gate_up({ty:?})");
            assert_eq!(down(ty), None, "down({ty:?})");
            assert_eq!(up(ty), None, "up({ty:?})");
            assert_eq!(gemm(ty), None, "gemm({ty:?})");
            assert_eq!(dense(ty), None, "dense({ty:?})");
            assert_eq!(head(ty), None, "head({ty:?})");
            assert_eq!(embed(ty), None, "embed({ty:?})");
        }
    }
}
