//! The qwen35moe fixture's DSpark drafts: the reduced-vocabulary speculator shape and the
//! full-vocabulary one ([`drafts`]), and the whole set [`write_set`] writes beside the fixture's
//! first shard — the two drafts, the mismatched twin and the renamed target copy — through the
//! shared writer ([`crate::dspark_fixture`]).

use std::path::{Path, PathBuf};

use gguf::Split;
use model::fixture::FixtureError;
use model::fixture::dspark::{
    FULL_DIR, MISMATCHED_DIR, REDUCED_DIR, RENAMED_BASENAME, RENAMED_NAME,
};

use crate::dspark_fixture::{Draft, Target, Vocab, rename_target, write_dir};

/// How far the fixture's mask id sits below the top of the vocabulary: inside the mismatched
/// twin's stated vocabulary too, so that twin's refusal is the ids line the cross-target clause
/// reads, not the mask's.
const MASK_MARGIN: usize = 256;

/// The two drafts the fixture serves beside its target, one per head shape.
///
/// The reduced one is the RedHat speculator's shape: the speculator's block, window,
/// confidence head and vocabulary at a quarter of its width, three taps of the fixture's four
/// layers. No tap is the first layer: the instrument a gate compares the production taps with
/// holds every *layer's output*, so a first-layer tap's entering row — the embedding row — has
/// no twin there. The mask id sits [`MASK_MARGIN`] ids below the top of the vocabulary, the
/// real speculator's own distance in kind.
///
/// The full one is MiniCPM-V's shape: the target's own ids over the target's own width (the
/// borrowed table's rows), full attention (no window), so the two drafts between them hold both
/// attention reaches the loader reads.
///
/// Another family's fixture takes its own constructor over the same [`Target`] and [`Draft`]
/// in its own module: its taps, widths and mask id are facts of its target, not of this one.
#[derive(Debug)]
pub struct Drafts {
    /// The reduced-vocabulary draft (its own head, table and id map).
    pub reduced: Draft,
    /// The full-vocabulary draft (the target's ids, its head and table borrowed).
    pub full: Draft,
}

/// The fixture's two drafts, held to `target`'s own rules.
pub fn drafts(target: &Target) -> Result<Drafts, FixtureError> {
    let mask = target
        .n_vocab
        .checked_sub(MASK_MARGIN)
        .ok_or_else(|| FixtureError::Mismatch {
            what: "the DSpark draft's mask id".into(),
            detail: format!(
                "a vocabulary of {} ids holds none {} below its top",
                target.n_vocab, MASK_MARGIN
            ),
        })?;
    let mask = u32::try_from(mask).map_err(|_| FixtureError::Mismatch {
        what: "the DSpark draft's mask id".into(),
        detail: format!("{mask} is no u32 id"),
    })?;
    let at = |hidden: usize, inter: usize, window: Option<usize>| Draft {
        layers: 5,
        hidden,
        heads: 8,
        kv_heads: 2,
        inter,
        taps: vec![1, 2, 3],
        block: 8,
        mask_id: mask,
        window,
        confidence: true,
        rows: 32000,
    };
    let reduced = at(512, 1536, Some(2048));
    reduced.check(target)?;
    let full = at(target.hidden, target.hidden * 3, None);
    full.check(target)?;
    Ok(Drafts { reduced, full })
}

/// What [`write_set`] wrote: the artifact paths, for the caller's summary.
pub struct Written {
    pub reduced: PathBuf,
    pub full: PathBuf,
    pub mismatched: PathBuf,
    pub renamed: Vec<PathBuf>,
}

/// Write the whole DSpark fixture beside `first` (a fixture's first shard): the three drafts of
/// [`drafts`]' geometry (the reduced shape twice and the full one) and the renamed copy,
/// each refused when it exists.
pub fn write_set(first: &Path, seed: u64) -> Result<Written, FixtureError> {
    let split = Split::open(first).map_err(FixtureError::Load)?;
    let target = Target::of(&split)?;
    let drafts = drafts(&target)?;
    let at = |dir: &str| first.parent().unwrap_or(Path::new(".")).join(dir);
    let reduced = at(REDUCED_DIR);
    let full = at(FULL_DIR);
    let mismatched = at(MISMATCHED_DIR);
    write_dir(&reduced, &target, &drafts.reduced, Vocab::Reduced, seed)?;
    write_dir(&full, &target, &drafts.full, Vocab::Full, seed)?;
    write_dir(
        &mismatched,
        &target,
        &drafts.reduced,
        Vocab::Mismatched,
        seed,
    )?;
    let renamed = rename_target(first, RENAMED_NAME, RENAMED_BASENAME)?;
    Ok(Written {
        reduced,
        full,
        mismatched,
        renamed,
    })
}

#[cfg(test)]
mod tests {
    use super::{Target, drafts};

    /// The fixture's own drafts: their facts (the reduced one the speculator's shape,
    /// the full one the target's own width with no window), and their refusals on a target too
    /// small for their taps or their mask.
    #[test]
    fn the_drafts_hold_the_fixtures_facts() {
        let target = Target {
            hidden: 2048,
            n_vocab: 248_320,
            n_layers: 4,
        };
        let d = drafts(&target).expect("the fixture's target");
        assert_eq!(
            (
                d.reduced.layers,
                d.reduced.hidden,
                d.reduced.heads,
                d.reduced.kv_heads,
                d.reduced.inter,
                d.reduced.block,
                d.reduced.rows
            ),
            (5, 512, 8, 2, 1536, 8, 32000)
        );
        assert_eq!(d.reduced.taps, [1, 2, 3]);
        assert_eq!(d.reduced.window, Some(2048));
        assert_eq!(d.reduced.mask_id, 248_320 - 256);
        assert_eq!(
            (d.full.hidden, d.full.inter, d.full.window),
            (2048, 6144, None)
        );
        assert_eq!(d.full.taps, [1, 2, 3]);
        let mut small = target;
        small.n_layers = 3;
        let err = drafts(&small).expect_err("the third tap has no layer");
        assert!(err.to_string().contains("tap 3"), "{err}");
        let mut narrow = target;
        narrow.n_vocab = 200;
        let err = drafts(&narrow).expect_err("no mask id");
        assert!(err.to_string().contains("mask id"), "{err}");
    }
}
