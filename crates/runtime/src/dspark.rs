//! A DSpark draft's geometry ([`DsparkShape`]): what the draft file says about
//! its block, the target layers it reads, its vocabulary against the target's,
//! its mask token and its attention window. Every field is the file's data,
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
    /// The Markov head's rank.
    pub markov_rank: usize,
    /// Whether the file carries a confidence head.
    pub confidence: bool,
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
    /// A width of 0 (hidden or rank).
    Zero(&'static str),
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

    /// The block rows a proposal of `width` ids runs.
    pub fn rows(&self, width: usize) -> usize {
        if self.row0_drafts { width } else { width + 1 }
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
        if self.markov_rank == 0 {
            return Err(DsparkError::Zero("Markov rank"));
        }
        if self.window == Some(0) {
            return Err(DsparkError::Window);
        }
        let n_target = self.vocab.n_target();
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
    use super::{DraftVocab, DsparkError, DsparkShape};

    fn shape(block: usize, vocab: DraftVocab) -> DsparkShape {
        DsparkShape {
            block,
            row0_drafts: true,
            taps: vec![2, 10, 20, 30, 37],
            hidden: 2048,
            vocab,
            mask_id: 5,
            window: Some(2048),
            markov_rank: 256,
            confidence: true,
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
}
