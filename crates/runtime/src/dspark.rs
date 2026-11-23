//! A DSpark draft's geometry ([`DsparkShape`]): what the draft file says about
//! its block, the target layers it reads and how it turns them into context
//! rows ([`Context`]), its vocabulary against the target's, its mask token, its
//! attention window and its head ([`HeadKind`]). Every field is the file's
//! data, read once at load by the file's reader; no family's value is a
//! constant in code. The card-side owner that runs a draft of this shape is
//! `bloomery_gpu::dspark`.
//!
//! The plans a load makes from a shape and its target are here too, so a host
//! test holds them: each tap against the target's layer applications
//! ([`DsparkShape::tap_rows`]), the target's output matrix as the head reads it
//! ([`BaseHead::plan`]) and the fold of a recurrent head whose correction runs
//! through that matrix ([`DsparkShape::fold`]).

use std::fmt;

use models::Visit;

/// What a DSpark draft file says about its geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsparkShape {
    /// Tokens one block pass drafts: row 0 is the token at the target's
    /// position, rows 1.. the mask token's.
    pub block: usize,
    /// Whether row 0's output (the token at the target's position) drafts
    /// id 0: then a block of `w` rows drafts `w` ids; otherwise row 0 is
    /// only the anchor and `w` ids take `w + 1` rows. The file's convention,
    /// set by its reader.
    pub row0_drafts: bool,
    /// The target's layer applications whose entering rows the draft reads
    /// ([`crate::TapNeed::Layers`]), in the file's order: the residual row
    /// entering application `L` (HF `hidden_states[L]`). An application is one
    /// run of a stored layer in the target's run order, so a looped trunk's
    /// core layers count once a pass ([`models::visits`]). `L` equal to the
    /// application count is the row the target's head reads, past the last
    /// application and the output norm (HF's last `hidden_states`;
    /// [`DsparkShape::tap_rows`]). A file written in another convention is
    /// converted by its reader.
    pub taps: Vec<usize>,
    /// The target's hidden width; one tap row is `taps.len() * hidden` floats.
    pub hidden: usize,
    /// How a position's tapped rows become the draft's context rows.
    pub context: Context,
    /// The draft's vocabulary against the target's.
    pub vocab: DraftVocab,
    /// The token the block's rows 1.. embed.
    pub mask_id: u32,
    /// How far back the draft's attention reads its context ring: `None` is
    /// the whole context.
    pub window: Option<usize>,
    /// How the block's rows become ids.
    pub head: HeadKind,
}

/// What follows the head's projection: how a block's rows become ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadKind {
    /// V4.1's Markov head: each row's logits take a bigram correction of
    /// the id the row before took, `markov_w2[v] · markov_w1[prev]` over
    /// `rank` values, and the row's argmax is its id (serial over the rows).
    /// `confidence` says the file carries a confidence head.
    Markov { rank: usize, confidence: bool },
    /// The gated Markov recurrence (Mistral Large 4's, Inkling 2's): a state
    /// of `rank` values, zero at the first row that drafts, carried down the
    /// rows. Row `i`'s `joint_proj` over `[state | markov_w1[prev] | normed
    /// row]` (and the context's own row, [`RnnRow::ctx`]) gives a gate, a
    /// candidate and a correction vector; the state becomes `cand + gate ·
    /// (state − cand)` and the correction of `tanh(vector)` is added to the
    /// row's logits. The row's argmax is its id and the next row's `prev`
    /// (serial over the rows). The first drafting row's `prev` is the token at
    /// the target's position. [`RnnRow`] holds the four facts the forms differ
    /// in.
    MarkovRnn { rank: usize, row: RnnRow },
    /// DFlash2's selector: every row after the anchor keeps the `top_k` ids
    /// its logits rank highest and scores each against each id the row
    /// before kept through factors of `rank` values
    /// ([`crate::selector`]); a greedy walk takes one id a row. Row 0 is the
    /// anchor and drafts nothing, and the draft's attention is non-causal
    /// over the block, so a block always runs whole.
    Selector { rank: usize, top_k: usize },
}

/// The facts a recurrent Markov head's row is built from, each its file's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RnnRow {
    /// Where the recurrence starts.
    pub start: RnnStart,
    /// Whether the joint row ends with the context's own row,
    /// `[state | markov_w1[prev] | normed row | context row]`: the last fuse mix
    /// ([`Context::Fuse`]) of the position the block runs after, one row for
    /// every row of the block.
    pub ctx: bool,
    /// Where the logit soft cap applies.
    pub cap: RnnCap,
    /// Where the correction comes from.
    pub out: RnnOut,
}

/// The row a recurrent head's state starts at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RnnStart {
    /// Row 0 drafts, and the state starts there: zero, its `prev` the block's
    /// first id (the token at the target's position).
    Row0,
    /// Row 0 is the anchor and drafts nothing: the state starts at row 1, zero,
    /// its `prev` the anchor's id. The same recurrence one row later.
    Anchor,
}

/// Where a recurrent head's logit soft cap applies (its value is the file's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RnnCap {
    /// On the projection's logits before the correction is added; the
    /// correction is not capped.
    Projection,
    /// On the sum of the projection's logits and the correction; the row's id
    /// is the first maximum after the cap.
    Sum,
}

/// Where a recurrent head's correction comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RnnOut {
    /// The file's own `markov_w2`: one row of `rank` values a head id.
    Own,
    /// `hidden_out` (`rank` values to `hidden`, with a bias `b`) through the
    /// target's output matrix: the correction is `output · (hidden_out ·
    /// tanh(e) + b)`. The owner folds it at load ([`DsparkShape::fold`]).
    Hidden,
}

impl RnnRow {
    /// The joint row's width over normed rows of `hidden` values: the state,
    /// the previous id's row and the normed row, then the context's row when
    /// the head reads it.
    #[must_use]
    pub fn joint_in(&self, rank: usize, hidden: usize) -> usize {
        2 * rank + hidden * (1 + usize::from(self.ctx))
    }
}

/// How a draft turns a position's tapped rows into its context rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Context {
    /// `fc` over the tapped rows side by side (`taps.len() * hidden` values)
    /// and the hidden norm after it: one context row a position, which every
    /// draft layer reads.
    Fc,
    /// `mixes` softmax-weighted sums of the tapped rows, each RMS-normed by
    /// its own gain: the fuse table is a row of `taps.len()` weights a mix
    /// ([`DsparkShape::fuse_dims`]). Mix `l` is draft layer `l`'s context row;
    /// under a recurrent head that reads the context ([`RnnRow::ctx`]) the
    /// last mix is the head's.
    Fuse { mixes: usize },
}

/// What a tap reads of its target ([`DsparkShape::tap_rows`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapRow {
    /// The residual row entering the target's `application`-th layer
    /// application, a run of `visit.layer` in pass `visit.pass`.
    Entering { application: usize, visit: Visit },
    /// The row the target's head reads: past the last application and the
    /// output norm.
    Final,
}

/// The type of the target's output matrix a full-vocabulary draft's head
/// reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadMatrix {
    Q6K,
    Q8_0,
    Bf16,
}

/// Where a draft's head reads the target's output matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadPlace {
    /// A copy on the draft's card: the draft's own bytes.
    Copy,
    /// The target's resident matrix, on the target's card: none of its bytes
    /// the draft's.
    InPlace,
}

/// The target's output matrix as a full-vocabulary draft's head reads it:
/// the block's base projection, and the matrix a folded recurrent head's fold
/// multiplies at load ([`DsparkShape::fold`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseHead {
    pub matrix: HeadMatrix,
    pub place: HeadPlace,
}

impl BaseHead {
    /// The head planned for `matrix` at `place`: a Q6_K matrix copied or in
    /// place, a Q8_0 or BF16 matrix in place. A copy of a Q8_0 or BF16 matrix
    /// is refused by name: the matrix is read where the target holds it, and
    /// a copy would only take the draft's card its bytes.
    pub fn plan(matrix: HeadMatrix, place: HeadPlace) -> Result<BaseHead, DsparkError> {
        match (matrix, place) {
            (HeadMatrix::Q6K, _) | (_, HeadPlace::InPlace) => Ok(BaseHead { matrix, place }),
            (HeadMatrix::Q8_0 | HeadMatrix::Bf16, HeadPlace::Copy) => {
                Err(DsparkError::HeadCopy(matrix))
            }
        }
    }

    /// The draft's own card bytes for a matrix of `rows` rows of `cols`
    /// values: none in place, the matrix's bytes for a copy (Q6_K: 210 bytes a
    /// 256 values).
    #[must_use]
    pub fn own_bytes(&self, rows: usize, cols: usize) -> u64 {
        let values = rows as u64 * cols as u64;
        match (self.place, self.matrix) {
            (HeadPlace::InPlace, _) => 0,
            (HeadPlace::Copy, HeadMatrix::Q6K) => values / 256 * 210,
            (HeadPlace::Copy, HeadMatrix::Q8_0) => values / 32 * 34,
            (HeadPlace::Copy, HeadMatrix::Bf16) => values * 2,
        }
    }
}

/// A recurrent head whose correction runs through the target's output
/// matrix ([`RnnOut::Hidden`]), folded at load: `W2 = output · hidden_out`,
/// one Q8_0 row of `rank` values a head id, and `c = output · b`, one f32 a
/// head id. A row then reads `W2 · tanh(e) + c`, `rows · rank` values, in
/// place of the whole output matrix. The fold reorders the correction's float
/// sum, so a draft id can differ from the unfolded form's at a near-tie; the
/// verify decides every kept id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fold {
    /// W2's rows and c's values: the ids the head scores.
    pub rows: usize,
    /// W2's row length: the head's rank.
    pub rank: usize,
}

impl Fold {
    /// W2's bytes: Q8_0, 34 bytes a 32 values.
    #[must_use]
    pub fn w2_bytes(&self) -> u64 {
        self.rows as u64 * (self.rank as u64 / 32) * 34
    }

    /// c's bytes: one f32 a head id.
    #[must_use]
    pub fn c_bytes(&self) -> u64 {
        self.rows as u64 * 4
    }
}

/// The ids a draft's head scores.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DraftVocab {
    /// The target's own ids, `n` of them.
    Full { n: usize },
    /// A reduced head of `to_target.len()` ids: draft id `d` is target id
    /// `to_target[d]` (a file's `d2t` offsets already added), each inside
    /// `n_target`.
    Reduced {
        to_target: Vec<u32>,
        n_target: usize,
    },
}

/// A draft shape or id the owner refuses, naming the field and its value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DsparkError {
    /// A block too short to draft one id.
    Block(usize),
    /// No tap, or taps not strictly rising.
    Taps(Vec<usize>),
    /// A width of 0 (hidden, rank or top-k).
    Zero(&'static str),
    /// A selector head over a shape that cannot host it.
    Selector(&'static str),
    /// A recurrent Markov head over a shape that cannot host it.
    Rnn(&'static str),
    /// A tap past the target's layer applications and its final row.
    TapPast { tap: usize, applications: usize },
    /// A copy of a target output matrix the head reads in place.
    HeadCopy(HeadMatrix),
    /// A selector keeping more ids than the head scores.
    TopK { top_k: usize, rows: usize },
    /// A reduced map whose id `index` is `id`, past the target's `n_target`
    /// ids, or a repeat of an earlier one.
    Map {
        index: usize,
        id: u32,
        why: &'static str,
    },
    /// A mask id past the target's vocabulary.
    Mask { id: u32, n_target: usize },
    /// A window of 0 positions.
    Window,
    /// A draft id past the head's rows.
    Id { id: u32, rows: usize },
}

impl fmt::Display for DsparkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DsparkError::Block(b) => write!(
                f,
                "dspark: a block of {b}: a block drafts at least one id (2 rows)"
            ),
            DsparkError::Taps(t) => write!(f, "dspark: taps {t:?}: at least one, strictly rising"),
            DsparkError::Zero(what) => write!(f, "dspark: a {what} of 0"),
            DsparkError::Selector(why) => write!(f, "dspark: a selector head {why}"),
            DsparkError::Rnn(why) => write!(f, "dspark: a recurrent Markov head {why}"),
            DsparkError::TapPast { tap, applications } => write!(
                f,
                "dspark: tap {tap} past a target of {applications} layer applications (the \
                 final row is tap {applications})"
            ),
            DsparkError::HeadCopy(m) => write!(
                f,
                "dspark: a copy of the target's {m:?} output matrix: the head reads it in place"
            ),
            DsparkError::TopK { top_k, rows } => write!(
                f,
                "dspark: a selector keeping {top_k} ids of a head that scores {rows}"
            ),
            DsparkError::Map { index, id, why } => {
                write!(
                    f,
                    "dspark: the draft-to-target map's id {index} is {id}: {why}"
                )
            }
            DsparkError::Mask { id, n_target } => {
                write!(f, "dspark: mask id {id} past the target's {n_target} ids")
            }
            DsparkError::Window => f.write_str("dspark: an attention window of 0 positions"),
            DsparkError::Id { id, rows } => {
                write!(f, "dspark: draft id {id} past the head's {rows} rows")
            }
        }
    }
}

impl std::error::Error for DsparkError {}

impl DraftVocab {
    /// The target's vocabulary size.
    pub fn n_target(&self) -> usize {
        match self {
            DraftVocab::Full { n } => *n,
            DraftVocab::Reduced { n_target, .. } => *n_target,
        }
    }

    /// Ids the draft's head scores.
    pub fn rows(&self) -> usize {
        match self {
            DraftVocab::Full { n } => *n,
            DraftVocab::Reduced { to_target, .. } => to_target.len(),
        }
    }

    /// The target id of draft id `d`; a `d` past the head is refused by name.
    pub fn target(&self, d: u32) -> Result<u32, DsparkError> {
        let rows = self.rows();
        let past = || DsparkError::Id { id: d, rows };
        match self {
            DraftVocab::Full { .. } => (usize::try_from(d).map_err(|_| past())? < rows)
                .then_some(d)
                .ok_or_else(past),
            DraftVocab::Reduced { to_target, .. } => usize::try_from(d)
                .ok()
                .and_then(|i| to_target.get(i).copied())
                .ok_or_else(past),
        }
    }
}

impl DsparkShape {
    /// The most ids one block drafts.
    pub fn ids_per_block(&self) -> usize {
        if self.row0_drafts {
            self.block
        } else {
            self.block.saturating_sub(1)
        }
    }

    /// The most ids one proposal holds against a target that verifies
    /// `verify_rows` rows: the block's ids, and at most `verify_rows − 1`.
    pub fn width(&self, verify_rows: usize) -> usize {
        self.ids_per_block().min(verify_rows.saturating_sub(1))
    }

    /// The block rows a proposal of `width` ids runs: a selector's is the
    /// whole block, its rows attend to one another.
    pub fn rows(&self, width: usize) -> usize {
        match self.head {
            HeadKind::Selector { .. } => self.block,
            HeadKind::Markov { .. } | HeadKind::MarkovRnn { .. } if self.row0_drafts => width,
            HeadKind::Markov { .. } | HeadKind::MarkovRnn { .. } => width + 1,
        }
    }

    /// The block row of a proposal's first id.
    pub fn first_row(&self) -> usize {
        usize::from(!self.row0_drafts)
    }

    /// The fuse table's rows and row length, `[mixes, taps]`: a row of
    /// weights a mix, a weight a tap. `None` for an fc context.
    #[must_use]
    pub fn fuse_dims(&self) -> Option<[usize; 2]> {
        match self.context {
            Context::Fc => None,
            Context::Fuse { mixes } => Some([mixes, self.taps.len()]),
        }
    }

    /// The fold a recurrent head whose correction runs through the target's
    /// output matrix takes at load, over the ids the head scores; `None` for
    /// any other head.
    #[must_use]
    pub fn fold(&self) -> Option<Fold> {
        match self.head {
            HeadKind::MarkovRnn {
                rank,
                row:
                    RnnRow {
                        out: RnnOut::Hidden,
                        ..
                    },
            } => Some(Fold {
                rows: self.vocab.rows(),
                rank,
            }),
            _ => None,
        }
    }

    /// Each tap against a target whose trunk runs `applications`, in run
    /// order ([`models::visits`]): a tap below their count enters that
    /// application, a tap at it is the final row, and a tap past it is refused
    /// by name.
    pub fn tap_rows(&self, applications: &[Visit]) -> Result<Vec<TapRow>, DsparkError> {
        let n = applications.len();
        self.taps
            .iter()
            .map(|&tap| match applications.get(tap) {
                Some(&visit) => Ok(TapRow::Entering {
                    application: tap,
                    visit,
                }),
                None if tap == n => Ok(TapRow::Final),
                None => Err(DsparkError::TapPast {
                    tap,
                    applications: n,
                }),
            })
            .collect()
    }

    /// Every field's own check, a named error per field.
    pub fn check(&self) -> Result<(), DsparkError> {
        if self.ids_per_block() == 0 {
            return Err(DsparkError::Block(self.block));
        }
        if self.taps.is_empty() || self.taps.windows(2).any(|w| w[0] >= w[1]) {
            return Err(DsparkError::Taps(self.taps.clone()));
        }
        if self.hidden == 0 {
            return Err(DsparkError::Zero("hidden width"));
        }
        if self.context == (Context::Fuse { mixes: 0 }) {
            return Err(DsparkError::Zero("fuse mix count"));
        }
        let n_target = self.vocab.n_target();
        match self.head {
            HeadKind::Markov { rank: 0, .. } => return Err(DsparkError::Zero("Markov rank")),
            HeadKind::Markov { .. } => {}
            HeadKind::MarkovRnn { rank: 0, .. } => return Err(DsparkError::Zero("Markov rank")),
            HeadKind::MarkovRnn { rank, row } => {
                match (row.start, self.row0_drafts) {
                    (RnnStart::Row0, false) => {
                        return Err(DsparkError::Rnn(
                            "drafts from row 0: its state starts at the block's first id",
                        ));
                    }
                    (RnnStart::Anchor, true) => {
                        return Err(DsparkError::Rnn(
                            "starts after the anchor: row 0 is the anchor and drafts nothing",
                        ));
                    }
                    (RnnStart::Row0, true) | (RnnStart::Anchor, false) => {}
                }
                if matches!(self.vocab, DraftVocab::Reduced { .. }) {
                    return Err(DsparkError::Rnn(
                        "scores target ids: its tables are the target's vocabulary",
                    ));
                }
                if row.ctx && !matches!(self.context, Context::Fuse { mixes } if mixes >= 2) {
                    return Err(DsparkError::Rnn(
                        "reads the context's own row: a fuse of a mix a layer and one more",
                    ));
                }
                if row.out == RnnOut::Hidden && !rank.is_multiple_of(32) {
                    return Err(DsparkError::Rnn(
                        "folds its correction into Q8_0 rows: a rank that is a multiple of 32",
                    ));
                }
            }
            HeadKind::Selector { rank: 0, .. } => {
                return Err(DsparkError::Zero("selector rank"));
            }
            HeadKind::Selector { top_k: 0, .. } => {
                return Err(DsparkError::Zero("selector top-k"));
            }
            HeadKind::Selector { top_k, .. } => {
                if self.row0_drafts {
                    return Err(DsparkError::Selector(
                        "drafts from row 1: row 0 is the anchor",
                    ));
                }
                if matches!(self.vocab, DraftVocab::Reduced { .. }) {
                    return Err(DsparkError::Selector(
                        "scores target ids: its vocabulary is the target's",
                    ));
                }
                if top_k > self.vocab.rows() {
                    return Err(DsparkError::TopK {
                        top_k,
                        rows: self.vocab.rows(),
                    });
                }
            }
        }
        if self.window == Some(0) {
            return Err(DsparkError::Window);
        }
        if usize::try_from(self.mask_id).map_or(true, |m| m >= n_target) {
            return Err(DsparkError::Mask {
                id: self.mask_id,
                n_target,
            });
        }
        if let DraftVocab::Reduced { to_target, .. } = &self.vocab {
            let mut seen = vec![false; n_target];
            for (index, &id) in to_target.iter().enumerate() {
                let slot = usize::try_from(id)
                    .ok()
                    .and_then(|i| seen.get_mut(i))
                    .ok_or(DsparkError::Map {
                        index,
                        id,
                        why: "past the target's vocabulary",
                    })?;
                if std::mem::replace(slot, true) {
                    return Err(DsparkError::Map {
                        index,
                        id,
                        why: "a repeat of an earlier id",
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use models::{LoopSpec, Visit, visits};

    use super::{
        BaseHead, Context, DraftVocab, DsparkError, DsparkShape, Fold, HeadKind, HeadMatrix,
        HeadPlace, RnnCap, RnnOut, RnnRow, RnnStart, TapRow,
    };

    /// Mistral Large 4's recurrent row: row 0 drafts, no context row, the cap
    /// on the projection, its own `markov_w2`.
    const ROW0: RnnRow = RnnRow {
        start: RnnStart::Row0,
        ctx: false,
        cap: RnnCap::Projection,
        out: RnnOut::Own,
    };

    /// Inkling 2's: the anchor first, the context's row in the joint row, the
    /// cap on the sum, the correction through the target's output matrix.
    const ANCHOR: RnnRow = RnnRow {
        start: RnnStart::Anchor,
        ctx: true,
        cap: RnnCap::Sum,
        out: RnnOut::Hidden,
    };

    fn shape(block: usize, vocab: DraftVocab) -> DsparkShape {
        DsparkShape {
            block,
            row0_drafts: true,
            taps: vec![2, 10, 20, 30, 37],
            hidden: 2048,
            context: Context::Fc,
            vocab,
            mask_id: 5,
            window: Some(2048),
            head: HeadKind::Markov {
                rank: 256,
                confidence: true,
            },
        }
    }

    /// DFlash2's geometry: a block of 8 over the target's ids, the selector
    /// head, row 0 the anchor.
    fn selector_shape(top_k: usize, vocab: DraftVocab) -> DsparkShape {
        DsparkShape {
            row0_drafts: false,
            head: HeadKind::Selector { rank: 256, top_k },
            ..shape(8, vocab)
        }
    }

    #[test]
    fn a_proposal_is_bounded_by_the_block_and_the_targets_verify() {
        let full = DraftVocab::Full { n: 16 };
        assert_eq!(shape(8, full.clone()).width(8), 7);
        assert_eq!(shape(15, full.clone()).width(8), 7);
        assert_eq!(shape(5, full.clone()).width(2), 1);
        assert_eq!(shape(5, full.clone()).width(8), 5);
        assert_eq!(shape(5, full.clone()).rows(3), 3);
        let mut anchor = shape(5, full);
        anchor.row0_drafts = false;
        assert_eq!(anchor.width(8), 4);
        assert_eq!(anchor.rows(3), 4);
        anchor.block = 1;
        assert_eq!(anchor.check(), Err(DsparkError::Block(1)));
    }

    #[test]
    fn a_reduced_head_maps_each_draft_id_and_refuses_past_its_rows() {
        let v = DraftVocab::Reduced {
            to_target: vec![0, 7, 3],
            n_target: 16,
        };
        assert_eq!(v.rows(), 3);
        assert_eq!(v.target(1), Ok(7));
        assert_eq!(v.target(3), Err(DsparkError::Id { id: 3, rows: 3 }));
        let full = DraftVocab::Full { n: 16 };
        assert_eq!(full.target(15), Ok(15));
        assert_eq!(full.target(16), Err(DsparkError::Id { id: 16, rows: 16 }));
    }

    #[test]
    fn each_field_is_checked_by_name() {
        let ok = shape(
            8,
            DraftVocab::Reduced {
                to_target: vec![0, 7, 3],
                n_target: 16,
            },
        );
        assert_eq!(ok.check(), Ok(()));
        assert_eq!(
            shape(0, DraftVocab::Full { n: 16 }).check(),
            Err(DsparkError::Block(0))
        );
        let mut taps = ok.clone();
        taps.taps = vec![3, 3];
        assert_eq!(taps.check(), Err(DsparkError::Taps(vec![3, 3])));
        let mut mask = ok.clone();
        mask.mask_id = 16;
        assert_eq!(
            mask.check(),
            Err(DsparkError::Mask {
                id: 16,
                n_target: 16
            })
        );
        let past = shape(
            8,
            DraftVocab::Reduced {
                to_target: vec![0, 16],
                n_target: 16,
            },
        );
        assert!(matches!(
            past.check(),
            Err(DsparkError::Map {
                index: 1,
                id: 16,
                ..
            })
        ));
        let repeat = shape(
            8,
            DraftVocab::Reduced {
                to_target: vec![4, 4],
                n_target: 16,
            },
        );
        assert!(matches!(
            repeat.check(),
            Err(DsparkError::Map {
                index: 1,
                id: 4,
                ..
            })
        ));
        let mut window = ok;
        window.window = Some(0);
        assert_eq!(window.check(), Err(DsparkError::Window));
    }

    /// A selector runs the whole block whatever the proposal's width and
    /// drafts from row 1; a Markov block runs one row per id (plus the
    /// anchor when row 0 drafts nothing).
    #[test]
    fn a_selector_block_runs_whole_and_drafts_from_row_one() {
        let sel = selector_shape(16, DraftVocab::Full { n: 64 });
        assert_eq!(sel.check(), Ok(()));
        assert_eq!(sel.ids_per_block(), 7);
        assert_eq!(sel.width(8), 7);
        assert_eq!(sel.width(4), 3);
        for width in 1..=7 {
            assert_eq!(sel.rows(width), 8, "width {width}");
        }
        assert_eq!(sel.first_row(), 1);
        let mut markov = shape(8, DraftVocab::Full { n: 64 });
        markov.row0_drafts = false;
        assert_eq!((markov.rows(3), markov.first_row()), (4, 1));
        markov.row0_drafts = true;
        assert_eq!((markov.rows(3), markov.first_row()), (3, 0));
    }

    /// A recurrent Markov head runs one row per id like the bigram head, and
    /// is refused by name with a rank of 0, a row 0 that does not draft, or
    /// a reduced vocabulary.
    #[test]
    fn a_recurrent_head_runs_a_row_an_id_and_is_checked_by_name() {
        let mut rnn = shape(3, DraftVocab::Full { n: 64 });
        rnn.head = HeadKind::MarkovRnn {
            rank: 512,
            row: ROW0,
        };
        assert_eq!(rnn.check(), Ok(()));
        assert_eq!((rnn.rows(3), rnn.first_row(), rnn.width(8)), (3, 0, 3));
        let mut zero = rnn.clone();
        zero.head = HeadKind::MarkovRnn { rank: 0, row: ROW0 };
        assert_eq!(zero.check(), Err(DsparkError::Zero("Markov rank")));
        let mut anchor = rnn.clone();
        anchor.row0_drafts = false;
        assert!(matches!(anchor.check(), Err(DsparkError::Rnn(_))));
        rnn.vocab = DraftVocab::Reduced {
            to_target: vec![0, 7, 3],
            n_target: 64,
        };
        assert!(matches!(rnn.check(), Err(DsparkError::Rnn(_))));
    }

    /// Each selector field is refused by name: a width of 0, a row 0 that
    /// drafts, a reduced vocabulary, a top-k past the head's ids.
    #[test]
    fn a_selector_shape_is_checked_by_name() {
        let full = DraftVocab::Full { n: 64 };
        let mut rank = selector_shape(16, full.clone());
        rank.head = HeadKind::Selector { rank: 0, top_k: 16 };
        assert_eq!(rank.check(), Err(DsparkError::Zero("selector rank")));
        assert_eq!(
            selector_shape(0, full.clone()).check(),
            Err(DsparkError::Zero("selector top-k"))
        );
        let mut row0 = selector_shape(16, full.clone());
        row0.row0_drafts = true;
        assert!(matches!(row0.check(), Err(DsparkError::Selector(_))));
        let reduced = selector_shape(
            2,
            DraftVocab::Reduced {
                to_target: vec![0, 7, 3],
                n_target: 64,
            },
        );
        assert!(matches!(reduced.check(), Err(DsparkError::Selector(_))));
        assert_eq!(
            selector_shape(65, full).check(),
            Err(DsparkError::TopK {
                top_k: 65,
                rows: 64
            })
        );
        let mut markov = shape(8, DraftVocab::Full { n: 64 });
        markov.head = HeadKind::Markov {
            rank: 0,
            confidence: false,
        };
        assert_eq!(markov.check(), Err(DsparkError::Zero("Markov rank")));
    }

    /// Inkling 2's draft: a block of 8 whose row 0 is the anchor, four taps
    /// fused into a mix a layer of three and one for the head, the recurrent
    /// head of rank 256 over the target's 200,058 ids.
    fn anchor_shape() -> DsparkShape {
        DsparkShape {
            row0_drafts: false,
            taps: vec![5, 23, 41, 48],
            hidden: 2304,
            context: Context::Fuse { mixes: 4 },
            mask_id: 0,
            window: None,
            head: HeadKind::MarkovRnn {
                rank: 256,
                row: ANCHOR,
            },
            ..shape(8, DraftVocab::Full { n: 200_058 })
        }
    }

    /// The anchor form is a shape the owner takes: seven ids a block, eight
    /// rows for the widest proposal, the first id at row 1. Each start is held
    /// to the row 0 the file states: a row-0 start whose row 0 does not draft
    /// is refused as before, an anchor start whose row 0 drafts likewise.
    #[test]
    fn an_anchor_recurrent_head_starts_after_the_anchor() {
        let ink = anchor_shape();
        assert_eq!(ink.check(), Ok(()));
        assert_eq!(
            (
                ink.ids_per_block(),
                ink.width(8),
                ink.rows(7),
                ink.first_row()
            ),
            (7, 7, 8, 1)
        );
        let mut row0 = ink.clone();
        row0.head = HeadKind::MarkovRnn {
            rank: 256,
            row: RnnRow { ctx: true, ..ROW0 },
        };
        assert_eq!(
            row0.check(),
            Err(DsparkError::Rnn(
                "drafts from row 0: its state starts at the block's first id"
            ))
        );
        let mut drafts = ink;
        drafts.row0_drafts = true;
        assert_eq!(
            drafts.check(),
            Err(DsparkError::Rnn(
                "starts after the anchor: row 0 is the anchor and drafts nothing"
            ))
        );
    }

    /// A fuse table is a row of one weight a tap for each mix; a head that
    /// reads the context's own row needs a fuse with a mix past the layers',
    /// and its joint row grows by one hidden row.
    #[test]
    fn a_fuse_is_a_row_of_tap_weights_a_mix() {
        let mut three = anchor_shape();
        three.taps = vec![5, 23, 48];
        assert_eq!(three.check(), Ok(()));
        assert_eq!(three.fuse_dims(), Some([4, 3]));
        assert_eq!(anchor_shape().fuse_dims(), Some([4, 4]));
        assert_eq!(shape(8, DraftVocab::Full { n: 64 }).fuse_dims(), None);
        let mut none = anchor_shape();
        none.context = Context::Fuse { mixes: 0 };
        assert_eq!(none.check(), Err(DsparkError::Zero("fuse mix count")));
        for context in [Context::Fc, Context::Fuse { mixes: 1 }] {
            let mut fc = anchor_shape();
            fc.context = context;
            assert!(
                matches!(fc.check(), Err(DsparkError::Rnn(_))),
                "{context:?}"
            );
        }
        assert_eq!(ANCHOR.joint_in(256, 2304), 5120);
        assert_eq!(ROW0.joint_in(512, 7168), 8192);
    }

    /// Taps count the target's layer applications in run order: a looped
    /// trunk's core layers once a pass, and the count itself names the row
    /// the head reads. A tap past it is refused by name.
    #[test]
    fn a_tap_at_the_application_count_is_the_final_row() {
        let loop_ = LoopSpec {
            core_start: 6,
            core_end: 18,
            passes: 3,
            gate_rank: 64,
        };
        let runs = visits(24, Some(loop_));
        assert_eq!(runs.len(), 48);
        let at = |application, layer, pass| TapRow::Entering {
            application,
            visit: Visit { layer, pass },
        };
        assert_eq!(
            anchor_shape().tap_rows(&runs),
            Ok(vec![
                at(5, 5, 0),
                at(23, 11, 1),
                at(41, 17, 2),
                TapRow::Final
            ])
        );
        let mut past = anchor_shape();
        past.taps = vec![5, 49];
        assert_eq!(
            past.tap_rows(&runs),
            Err(DsparkError::TapPast {
                tap: 49,
                applications: 48
            })
        );
    }

    /// The target's Q8_0 or BF16 output matrix is planned in place, none of its
    /// bytes the draft's, and a copy of either is refused by name. The anchor
    /// head's correction folds into a Q8_0 W2 of a row a scored id and an f32
    /// constant a scored id; a head on its own `markov_w2` folds nothing.
    #[test]
    fn the_target_head_is_planned_in_place_and_folded() {
        for matrix in [HeadMatrix::Q6K, HeadMatrix::Q8_0, HeadMatrix::Bf16] {
            let base = BaseHead::plan(matrix, HeadPlace::InPlace);
            assert_eq!(
                base,
                Ok(BaseHead {
                    matrix,
                    place: HeadPlace::InPlace
                })
            );
            assert_eq!(base.map(|b| b.own_bytes(201_024, 2304)), Ok(0));
        }
        assert_eq!(
            BaseHead::plan(HeadMatrix::Q6K, HeadPlace::Copy).map(|b| b.own_bytes(256, 4)),
            Ok(840)
        );
        for matrix in [HeadMatrix::Q8_0, HeadMatrix::Bf16] {
            assert_eq!(
                BaseHead::plan(matrix, HeadPlace::Copy),
                Err(DsparkError::HeadCopy(matrix))
            );
        }
        let fold = anchor_shape().fold();
        assert_eq!(
            fold,
            Some(Fold {
                rows: 200_058,
                rank: 256
            })
        );
        assert_eq!(
            fold.map(|f| (f.w2_bytes(), f.c_bytes())),
            Some((54_415_776, 800_232))
        );
        let mut own = anchor_shape();
        own.head = HeadKind::MarkovRnn {
            rank: 256,
            row: RnnRow {
                out: RnnOut::Own,
                ..ANCHOR
            },
        };
        assert_eq!(own.fold(), None);
        own.head = HeadKind::MarkovRnn {
            rank: 48,
            row: ANCHOR,
        };
        assert!(matches!(own.check(), Err(DsparkError::Rnn(_))));
    }
}
