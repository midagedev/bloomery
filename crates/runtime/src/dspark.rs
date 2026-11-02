//! A DSpark draft's geometry ([`DsparkShape`]): what the draft file says about
//! its block, the target layers it reads, its vocabulary against the target's,
//! its mask token, its attention window and its head ([`HeadKind`]). Every
//! field is the file's data,
//! read once at load by the file's reader; no family's value is a constant in
//! code. The card-side owner that runs a draft of this shape is
//! `bloomery_gpu::dspark`.

use std::fmt;

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
    /// The target layers whose entering rows the draft reads
    /// ([`crate::TapNeed::Layers`]), in the file's order: the residual row
    /// entering layer `L` (HF `hidden_states[L]`). A file written in another
    /// convention is converted by its reader.
    pub taps: Vec<usize>,
    /// The target's hidden width; one tap row is `taps.len() * hidden` floats.
    pub hidden: usize,
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
    /// Mistral Large 4's gated Markov recurrence: a state of `rank` values,
    /// zero at the block's first row, carried down the rows. Row `i`'s
    /// `joint_proj` over `[state | markov_w1[prev] | normed row]` gives a
    /// gate, a candidate and a correction vector; the state becomes `cand +
    /// gate · (state − cand)` and `markov_w2 · tanh(correction)` is added to
    /// the row's logits after the projection's soft cap. The row's argmax is
    /// its id and the next row's `prev` (serial over the rows). Row 0
    /// drafts: its `prev` is the block's first id.
    MarkovRnn { rank: usize },
    /// DFlash2's selector: every row after the anchor keeps the `top_k` ids
    /// its logits rank highest and scores each against each id the row
    /// before kept through factors of `rank` values
    /// ([`crate::selector`]); a greedy walk takes one id a row. Row 0 is the
    /// anchor and drafts nothing, and the draft's attention is non-causal
    /// over the block, so a block always runs whole.
    Selector { rank: usize, top_k: usize },
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
        let n_target = self.vocab.n_target();
        match self.head {
            HeadKind::Markov { rank: 0, .. } => return Err(DsparkError::Zero("Markov rank")),
            HeadKind::Markov { .. } => {}
            HeadKind::MarkovRnn { rank: 0 } => return Err(DsparkError::Zero("Markov rank")),
            HeadKind::MarkovRnn { .. } => {
                if !self.row0_drafts {
                    return Err(DsparkError::Rnn(
                        "drafts from row 0: its state starts at the block's first id",
                    ));
                }
                if matches!(self.vocab, DraftVocab::Reduced { .. }) {
                    return Err(DsparkError::Rnn(
                        "scores target ids: its tables are the target's vocabulary",
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
    use super::{DraftVocab, DsparkError, DsparkShape, HeadKind};

    fn shape(block: usize, vocab: DraftVocab) -> DsparkShape {
        DsparkShape {
            block,
            row0_drafts: true,
            taps: vec![2, 10, 20, 30, 37],
            hidden: 2048,
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
        rnn.head = HeadKind::MarkovRnn { rank: 512 };
        assert_eq!(rnn.check(), Ok(()));
        assert_eq!((rnn.rows(3), rnn.first_row(), rnn.width(8)), (3, 0, 3));
        let mut zero = rnn.clone();
        zero.head = HeadKind::MarkovRnn { rank: 0 };
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
}
