//! The card budget a fixture records ([`KEY_CARD_BUDGET`]) when its family
//! plans one ([`CardBudget::Planned`]): the byte cap under which the written
//! file's own plan holds half of every card-eligible layer's routed experts
//! on the card, the rest on the host. The cap is chosen through the family's
//! planner by a binary search over it — a plan is monotone in its budget, its
//! cards capped by `min(usable, budget)` (placement's `capped`) — and
//! [`check`] is the proof, the one owner of what a recorded budget must make
//! of a file's plan: half on every layer the budgetless plan fills, and
//! every other layer as the budgetless plan has it. `choose` ends in it on
//! the header-only files it searched, and `verify` runs it on the written
//! file.
//!
//! The planner reads a file, so the plan being searched is materialized from
//! the [`FilePlan`] as header-only files under a fresh directory of the
//! system's temporary one, removed once the budget is chosen. The recorded
//! value is rounded up to a whole MiB when the rounding keeps the plan's
//! counts, so the number a person reads back names itself.

use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use gguf::Split;
use gguf::write::Writer;

use super::plan::FilePlan;
use super::spec::{Budget, CardExperts};
use super::{FixtureError, io_err};

/// Directories handed out, one per call.
static DIRS: AtomicUsize = AtomicUsize::new(0);

/// The plan's header-only files under a fresh temporary directory, removed
/// when dropped.
struct Headers {
    dir: PathBuf,
    first: Split,
}

impl Headers {
    /// `p` as its files' headers and a hole each: what the planner reads of
    /// a written fixture, without its bytes. A split set's first shard opens
    /// every shard, so each is written.
    fn of(p: &FilePlan) -> Result<Headers, FixtureError> {
        let dir = std::env::temp_dir().join(format!(
            "bloomery-fixture-budget-{}-{}",
            std::process::id(),
            DIRS.fetch_add(1, Ordering::Relaxed),
        ));
        if dir.try_exists().map_err(|e| io_err(&dir, "stat", e))? {
            std::fs::remove_dir_all(&dir).map_err(|e| io_err(&dir, "remove", e))?;
        }
        std::fs::create_dir(&dir).map_err(|e| io_err(&dir, "create the directory", e))?;
        let layouts = p.layouts()?;
        let mut first_path: Option<PathBuf> = None;
        for (name, layout) in layouts {
            let path = dir.join(name);
            let file = File::create(&path).map_err(|e| io_err(&path, "create", e))?;
            let len = layout.file_len();
            drop(Writer::new(&file, layout).map_err(|e| FixtureError::Write {
                path: path.clone(),
                source: e,
            })?);
            file.set_len(len).map_err(|e| io_err(&path, "extend", e))?;
            first_path.get_or_insert(path);
        }
        let first_path = first_path.ok_or_else(|| {
            FixtureError::Budget(format!(
                "{} holds no shard to plan",
                p.files.first().map(String::as_str).unwrap_or("the plan")
            ))
        })?;
        let first = Split::open(&first_path).map_err(FixtureError::Load)?;
        Ok(Headers { dir, first })
    }
}

impl Drop for Headers {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.dir)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!("could not remove {}: {e}", self.dir.display());
        }
    }
}

/// What the budgetless plan of `split` fills: its counts, the layers it puts
/// experts on (the card-eligible ones; it holds every expert on each), and
/// half the file's experts.
fn fills(planner: &Budget, split: &Split) -> Result<(CardExperts, Vec<usize>, u64), FixtureError> {
    let full = (planner.card_experts)(split, None)?;
    if full.per_layer.is_empty() || full.experts == 0 {
        return Err(FixtureError::Budget(format!(
            "the file's plan holds no routed experts ({} layers, {} experts)",
            full.per_layer.len(),
            full.experts
        )));
    }
    if full.experts % 2 != 0 {
        return Err(FixtureError::Budget(format!(
            "the file's {} experts have no half",
            full.experts
        )));
    }
    let eligible: Vec<usize> = full
        .per_layer
        .iter()
        .enumerate()
        .filter(|&(_, &n)| n > 0)
        .map(|(l, _)| l)
        .collect();
    if eligible.is_empty() {
        return Err(FixtureError::Budget(
            "no layer of the file holds a card expert under a budgetless plan".into(),
        ));
    }
    if let Some(&l) = eligible
        .iter()
        .find(|&&l| full.per_layer[l] != full.experts)
    {
        return Err(FixtureError::Budget(format!(
            "a budgetless plan holds {:?} of {} experts (layer {l}): the card is too small for \
             the half the fixture records a budget for",
            full.per_layer, full.experts
        )));
    }
    let half = full.experts / 2;
    Ok((full, eligible, half))
}

/// The per-layer counts `split`'s plan holds under `budget`: half on every
/// layer the budgetless plan fills, the budgetless plan's count elsewhere.
/// Returns them when they are those, else the error that lists both.
pub fn check(planner: &Budget, split: &Split, budget: u64) -> Result<CardExperts, FixtureError> {
    let (full, eligible, half) = fills(planner, split)?;
    let got = (planner.card_experts)(split, Some(budget)).map_err(|e| {
        FixtureError::Budget(format!("planning the file under the budget {budget}: {e}"))
    })?;
    let want: Vec<u64> = full
        .per_layer
        .iter()
        .enumerate()
        .map(|(l, &n)| if eligible.contains(&l) { half } else { n })
        .collect();
    if got.per_layer != want {
        return Err(FixtureError::Budget(format!(
            "the plan under the budget {budget} holds {:?} experts per layer, not {want:?} \
             (half of the file's {} on each layer the budgetless plan fills)",
            got.per_layer, full.experts
        )));
    }
    Ok(got)
}

/// The byte cap under which planning the written `target` holds half of
/// every card-eligible layer's experts on the card: the smallest cap that
/// does (rounded up to a whole MiB when that holds the same counts), as the
/// family's `planner` computes the counts, proved by [`check`] on the
/// header-only files.
pub fn choose(planner: &Budget, target: &FilePlan) -> Result<u64, FixtureError> {
    let headers = Headers::of(target)?;
    let (_, eligible, half) = fills(planner, &headers.first)?;
    // Whether `cap` holds at least half on every card-eligible layer — the
    // monotone predicate the search minimizes — with the planner's refusal
    // (a cap the dense tensors pass) as the reason when none fits.
    let attempt = |cap: u64| -> (bool, Option<String>) {
        match (planner.card_experts)(&headers.first, Some(cap)) {
            Ok(c) => (
                eligible
                    .iter()
                    .all(|&l| c.per_layer.get(l).is_some_and(|&n| n >= half)),
                None,
            ),
            Err(e) => (false, Some(e.to_string())),
        }
    };
    // Any cap past the card's usable bytes plans as no cap does, so the
    // search space is bounded however large.
    let (mut lo, mut hi) = (1u64, 1u64 << 45);
    let (fits, why_not) = attempt(hi);
    if !fits {
        return Err(FixtureError::Budget(format!(
            "no cap holds {half} experts on every card-eligible layer: {}",
            why_not.unwrap_or_else(|| "the plan at the largest cap".into())
        )));
    }
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if attempt(mid).0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // `hi` is the smallest cap that holds half; a whole MiB above it, where
    // the counts are still half, is the number recorded.
    let rounded = hi.next_multiple_of(1024 * 1024);
    let chosen = if rounded > hi && check(planner, &headers.first, rounded).is_ok() {
        rounded
    } else {
        hi
    };
    check(planner, &headers.first, chosen)?;
    Ok(chosen)
}
