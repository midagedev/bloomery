//! The card buffers of the heads a load's passes make after it opened: the one owner of their
//! bytes. A pass of `m` rows writes into a head of `m` rows (a body whose rows share one head) or
//! into `m` heads of one row each (a body with a head a row); the model makes each such head the
//! first time a pass of its rows runs or is captured, past the load's own one-row head. The plan
//! counts the heads a load's passes reach as named parts ([`row_heads`]) in the stage card's
//! prompt term ([`CardTotals::grow_prompt`](super::CardTotals::grow_prompt)), and the card's
//! model holds every head it makes to its part ([`head_part`], [`Parts::check`]).
//!
//! Model-free: a head's buffers follow from its vocabulary, its width, its final norm and the form
//! its projection reads ([`HeadReads`]), which the head tensor's type decides
//! ([`HeadReads::of_type`]).

use std::ops::RangeInclusive;

use gguf::GgmlType;
use gguf::card::{HeadEntry, SiteAct};

use super::parts::{Part, Parts};
use super::{HEAD_TENSOR, ModelTensors, PlacementError, Role};
use crate::kernels;

/// The most rows one pass takes, and so the widest head a pass makes.
pub const MAX_HEAD_ROWS: usize = 8;

/// The form a head's projection reads, as its buffers depend on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadReads {
    /// The normed f32 rows (Q8_0, BF16, E4M3 planes): no quantized form, and past one row a
    /// ticket word the argmax's blocks draw.
    F32,
    /// The q8_1 rows of 128 values a K-quant gemv reads.
    Q8x128,
    /// The q8_1 rows of the `_sel` Q5_K gemv, its one expert's id a row, and past one row the
    /// token-major logits it writes before the copy into the argmax's layout.
    Q5kSel,
    /// NVFP4 over the q8_1 rows of 128 values; past one row its token-major logits.
    Nvfp4Q8x128,
    /// NVFP4 over the 32-value rows (a width of no whole super-block); past one row its
    /// token-major logits.
    Nvfp4Blocks32,
}

impl HeadReads {
    /// The form a head of type `ty` reads at a width of `k` values; `None` for a type no head
    /// entry reads.
    #[must_use]
    pub fn of_type(ty: GgmlType, k: u64) -> Option<HeadReads> {
        if matches!(ty, GgmlType::MXFP8 | GgmlType::FP8_B128) {
            return Some(HeadReads::F32);
        }
        Some(match kernels::head(ty)? {
            HeadEntry::Q6k | HeadEntry::Q4k => HeadReads::Q8x128,
            HeadEntry::Q8_0 | HeadEntry::Bf16 => HeadReads::F32,
            HeadEntry::Q5k => HeadReads::Q5kSel,
            HeadEntry::Nvfp4 => match kernels::dense(ty)?.reads_at(k) {
                SiteAct::Q8x128 => HeadReads::Nvfp4Q8x128,
                _ => HeadReads::Nvfp4Blocks32,
            },
        })
    }
}

/// What a head's card buffers follow from past its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadForm {
    /// The rows of its projection: the vocabulary.
    pub vocab: usize,
    /// The values a row of its input holds.
    pub k: usize,
    /// Its final norm writes normed rows of its own (an RMS norm), or reads its input in place.
    pub normed: bool,
    pub reads: HeadReads,
}

impl HeadForm {
    /// The head of a file whose head tensor is `ty` with rows of `k` values and `vocab` rows,
    /// `normed` as its body's final norm: `None` for a type no head entry reads.
    #[must_use]
    pub fn of_type(ty: GgmlType, k: usize, vocab: usize, normed: bool) -> Option<HeadForm> {
        Some(HeadForm {
            vocab,
            k,
            normed,
            reads: HeadReads::of_type(ty, k as u64)?,
        })
    }
}

/// The tensor a load reads as its head when the file has no `output.weight`: tied embeddings.
const TIED_HEAD: &str = "token_embd.weight";

impl HeadForm {
    /// The head of `model`: its `output.weight`, or its token embedding when it has none (tied
    /// embeddings, as the load reads them), `normed` as its body's final norm; `None` for a model
    /// with neither, or a head of a type no head entry reads.
    #[must_use]
    pub fn of_model(model: &ModelTensors, normed: bool) -> Option<HeadForm> {
        let t = model
            .tensors
            .iter()
            .find(|t| t.role == Role::Head && t.name == HEAD_TENSOR)
            .or_else(|| model.tensors.iter().find(|t| t.name == TIED_HEAD))?;
        let k = usize::try_from(*t.dims.first()?).ok()?;
        let vocab = usize::try_from(*t.dims.get(1)?).ok()?;
        HeadForm::of_type(t.ty, k, vocab, normed)
    }
}

/// The part name of a head of `m` rows (one head a pass's rows).
const ONE_HEAD: [&str; MAX_HEAD_ROWS] = [
    "pass head of 1 row",
    "pass head of 2 rows",
    "pass head of 3 rows",
    "pass head of 4 rows",
    "pass head of 5 rows",
    "pass head of 6 rows",
    "pass head of 7 rows",
    "pass head of 8 rows",
];

/// The part name of row `r`'s head of one row (a head a row).
const ROW_HEAD: [&str; MAX_HEAD_ROWS] = [
    "row head 1",
    "row head 2",
    "row head 3",
    "row head 4",
    "row head 5",
    "row head 6",
    "row head 7",
    "row head 8",
];

/// The expert ids an NVFP4 head's `_sel` gemv family holds, one a column of its widest launch.
const SEL_IDS: usize = 8;

/// How the heads of a pass's rows are laid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadLayout {
    /// One head of the pass's `m` rows.
    One,
    /// A head of one row for each of the pass's rows.
    PerRow,
}

/// The buffers of one head of `m` rows (1 to [`MAX_HEAD_ROWS`]) of `form`, in the order the
/// card's head allocates them, under `name`: past one row a ticket word (f32 rows); the
/// one-expert ids and past one row the token-major logits (`_sel` Q5_K); the `_sel` family's ids,
/// the 32-value rows of a width of no whole super-block and past one row the token-major logits
/// (NVFP4); then the input rows, the normed rows under an RMS norm, the q8_1 rows of a form that
/// reads them, the logits and the argmax's readback words (the `m` tokens and the fault pair).
///
/// # Panics
///
/// On `m` outside 1 to [`MAX_HEAD_ROWS`], or a width the q8 forms do not hold (not a multiple of
/// 256 for the q8_1 rows, of 32 for the 32-value rows): a head of either is no head the card makes.
#[must_use]
pub fn head_part(name: &'static str, form: HeadForm, m: usize) -> Part {
    assert!(
        (1..=MAX_HEAD_ROWS).contains(&m),
        "a head of {m} rows; a head holds 1 to {MAX_HEAD_ROWS}"
    );
    let HeadForm {
        vocab, k, normed, ..
    } = form;
    let logits = m * vocab;
    let mut p = Part::new(name);
    match form.reads {
        HeadReads::F32 => {
            if m > 1 {
                p = p.word(1);
            }
        }
        HeadReads::Q5kSel => {
            p = p.word(m);
            if m > 1 {
                p = p.word(logits);
            }
        }
        HeadReads::Nvfp4Blocks32 => {
            assert!(k > 0 && k.is_multiple_of(32), "32-value rows of {k} values");
            let blocks = k / 32;
            p = p
                .word(SEL_IDS)
                .word(256 * blocks.div_ceil(32) * m)
                .word(blocks * m)
                .word(blocks * m);
            if m > 1 {
                p = p.word(logits);
            }
        }
        HeadReads::Nvfp4Q8x128 => {
            p = p.word(SEL_IDS);
            if m > 1 {
                p = p.word(logits);
            }
        }
        HeadReads::Q8x128 => {}
    }
    p = p.word(m * k);
    if normed {
        p = p.word(m * k);
    }
    if matches!(
        form.reads,
        HeadReads::Q8x128 | HeadReads::Q5kSel | HeadReads::Nvfp4Q8x128
    ) {
        assert!(k.is_multiple_of(256), "q8_1 rows of {k} values");
        let sb = k / 256;
        p = p
            .with(m * 64 * sb.div_ceil(2), 8)
            .word(m * 256 * sb.div_ceil(4))
            .word(m * 128 * sb.div_ceil(2))
            .word(m * 8 * sb)
            .word(m * 2 * sb);
    }
    p.word(logits).word(m + 2)
}

/// The part of the head a pass of `m` rows makes under `layout` for its row `m`: under
/// [`HeadLayout::One`] the head of `m` rows, under [`HeadLayout::PerRow`] row `m`'s head of one
/// row. The part's name is the check's.
#[must_use]
pub fn layout_part(layout: HeadLayout, form: HeadForm, m: usize) -> Part {
    match layout {
        HeadLayout::One => head_part(ONE_HEAD[m - 1], form, m),
        HeadLayout::PerRow => head_part(ROW_HEAD[m - 1], form, 1),
    }
}

/// The heads a load's passes of `rows` rows make past the load's own one-row head, as parts
/// laid as `layout`: under [`HeadLayout::One`] a head of each row count in `rows`; under
/// [`HeadLayout::PerRow`] a head of one row for each row past the first up to the widest pass
/// (the load's head is row 1's). Empty for an empty range.
///
/// # Panics
///
/// On a range past [`MAX_HEAD_ROWS`] or one that starts at 0.
#[must_use]
pub fn row_heads(layout: HeadLayout, form: HeadForm, rows: RangeInclusive<usize>) -> Parts {
    if rows.is_empty() {
        return Parts::default();
    }
    assert!(
        *rows.start() >= 1 && *rows.end() <= MAX_HEAD_ROWS,
        "passes of {rows:?} rows; a pass takes 1 to {MAX_HEAD_ROWS}"
    );
    let parts = match layout {
        HeadLayout::One => rows.map(|m| layout_part(layout, form, m)).collect(),
        HeadLayout::PerRow => (2..=*rows.end())
            .map(|m| layout_part(layout, form, m))
            .collect(),
    };
    Parts::new(parts)
}

/// [`row_heads`] of `model`'s head ([`HeadForm::of_model`], `normed` as its body's final norm):
/// the parts a model's plan counts for the heads its load's passes of `rows` rows make. Empty for
/// an empty range, whatever the file's head.
///
/// # Errors
///
/// [`PlacementError::PassHeads`] for a non-empty range over a file with no head a card head reads.
pub fn model_row_heads(
    model: &ModelTensors,
    normed: bool,
    layout: HeadLayout,
    rows: RangeInclusive<usize>,
) -> Result<Parts, PlacementError> {
    if rows.is_empty() {
        return Ok(Parts::default());
    }
    let form = HeadForm::of_model(model, normed).ok_or(PlacementError::PassHeads)?;
    Ok(row_heads(layout, form, rows))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;

    const MIB: u64 = 1 << 20;

    /// Qwen3.8's head: a Q8_0 matrix of 248,320 rows over 2,560 values, read in place by the
    /// hyper-connection head's mixed norm.
    fn q38() -> HeadForm {
        HeadForm::of_type(GgmlType::Q8_0, 2_560, 248_320, false).expect("a Q8_0 head")
    }

    /// Qwen3.8's heads of 1 to 8 rows, one head a pass: 36 rows of 10,240 B of input and
    /// 993,280 B of logits, the readback 4·(m + 2) B a head and a ticket word past one row —
    /// 36,126,956 B. On a 2 MiB granule the logits of 3 rows or more take 18 granules of their
    /// own and the rest packs into two shared ones: 40 MiB.
    #[test]
    fn qwen38_pass_heads_pin() {
        let parts = row_heads(HeadLayout::One, q38(), 1..=8);
        let per_row = 10_240 + 993_280;
        let words: u64 = (1..=8u64).map(|m| 4 * (m + 2)).sum::<u64>() + 4 * 7;
        assert_eq!(parts.bytes(), 36 * per_row + words);
        assert_eq!(parts.bytes(), 36_126_956);
        let g = NonZeroU64::new(2 * MIB).expect("not zero");
        assert_eq!(parts.card_bytes(g), 40 * MIB);
        assert_eq!(parts.parts().len(), 8);
        assert_eq!(
            parts.part("pass head of 3 rows").map(Part::bytes),
            Some(3 * per_row + 4 + 4 * 5)
        );
    }

    /// A head a row: the load holds row 1's, so passes of up to 4 rows make three heads of one
    /// row; one head a pass makes every row count asked.
    #[test]
    fn per_row_heads_skip_the_loads_head() {
        let one = row_heads(HeadLayout::PerRow, q38(), 1..=4);
        let names: Vec<&str> = one.parts().iter().map(Part::name).collect();
        assert_eq!(names, ["row head 2", "row head 3", "row head 4"]);
        assert_eq!(one.bytes(), 3 * head_part("x", q38(), 1).bytes());
        assert_eq!(row_heads(HeadLayout::One, q38(), 2..=4).parts().len(), 3);
        assert!(
            row_heads(HeadLayout::One, q38(), RangeInclusive::new(2, 1))
                .parts()
                .is_empty()
        );
    }

    /// Each form's buffers at a width of 4,096 values, 1,000 rows, two rows, normed: the q8_1
    /// planes 2·(512·8 + 1,024·4 + 1,024·4 + 128·4 + 32·4) = 25,856 B at 16 super-blocks; the
    /// 32-value rows 2·4·(256·4 + 2·128) = 10,240 B; NVFP4's eight `_sel` ids 32 B.
    #[test]
    fn each_form_counts_its_buffers() {
        let form = |reads| HeadForm {
            vocab: 1_000,
            k: 4_096,
            normed: true,
            reads,
        };
        let rows = 2 * 2 * 4_096 * 4 + 2 * 1_000 * 4 + 4 * 4;
        let q8 = 25_856;
        let cases = [
            (HeadReads::F32, rows + 4),
            (HeadReads::Q8x128, rows + q8),
            (HeadReads::Q5kSel, rows + q8 + 2 * 4 + 2 * 1_000 * 4),
            (HeadReads::Nvfp4Q8x128, rows + 32 + q8 + 2 * 1_000 * 4),
            (HeadReads::Nvfp4Blocks32, rows + 32 + 10_240 + 2 * 1_000 * 4),
        ];
        for (reads, want) in cases {
            assert_eq!(head_part("h", form(reads), 2).bytes(), want, "{reads:?}");
        }
    }

    /// The type decides the form: Q8_0 and the E4M3 planes read f32 rows, the K-quants q8_1
    /// rows, Q5_K its `_sel` gemv; a type no head reads has none.
    #[test]
    fn the_type_decides_the_form() {
        assert_eq!(
            HeadReads::of_type(GgmlType::Q8_0, 2_560),
            Some(HeadReads::F32)
        );
        assert_eq!(
            HeadReads::of_type(GgmlType::MXFP8, 4_096),
            Some(HeadReads::F32)
        );
        assert_eq!(
            HeadReads::of_type(GgmlType::Q6_K, 4_096),
            Some(HeadReads::Q8x128)
        );
        assert_eq!(
            HeadReads::of_type(GgmlType::Q5_K, 4_096),
            Some(HeadReads::Q5kSel)
        );
        assert_eq!(HeadReads::of_type(GgmlType::IQ4_NL, 4_096), None);
    }
}
