//! The card budget a fixture records ([`KEY_CARD_BUDGET`]) when its family
//! plans one ([`CardBudget::Planned`]): the least byte cap, to the MiB, under
//! which the written file's own plan holds at least half of every
//! card-eligible layer's routed experts on the card, the rest on the host.
//! Exactly half when the plan can stop there; one or a few more when experts
//! smaller than the allocator's granule share the granule the half-th one
//! opens, so no cap holds exactly half. The cap is chosen through the family's
//! planner by a binary search over it — a plan is monotone in its budget, its
//! cards capped by `min(usable, budget)` (placement's `capped`) — and
//! [`check`] is the proof, the one owner of what a recorded budget must make
//! of a file's plan: at least half on every layer the budgetless plan fills,
//! every other layer as the budgetless plan has it, and a cap a MiB smaller
//! short of half somewhere (the least cap, so a larger one copied from
//! elsewhere is refused). `choose` ends in it on
//! the header-only files it searched, and `verify` runs it on the written
//! file, as a variant's check does on the variant's own ([`choose_written`]).
//!
//! A variant (`tools/fixture-variants.tsv`) may retype every routed stack to types its family's card leg
//! does not load, so the budgetless plan holds no card expert at all: the file is host-only
//! ([`host_only`] tells which case a file is). Its experts then do not depend on the cap, and the dense
//! tensors are the source's, so the budget is the one its header already records (the source's, which
//! the quantizer copied), held to plan the file with no expert on a card; the least cap that merely
//! plans is below the floor a load adds (its KV, context and margin). A base fixture with no
//! card-eligible layer stays an error: it could not exercise the card experts it exists to.
//!
//! The planner reads a file, so the plan being searched is materialized from
//! the [`FilePlan`] as header-only files under a fresh directory of the
//! system's temporary one, removed once the budget is chosen. The recorded
//! value is rounded up to a whole MiB when the rounding keeps the plan's
//! counts, so the number a person reads back names itself.

use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use gguf::write::Writer;
use gguf::{Split, Value};

use super::plan::FilePlan;
use super::spec::{Budget, CardExperts};
use super::{FixtureError, KEY_CARD_BUDGET, io_err};

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
/// half the file's experts. No card-eligible layer is an error unless
/// `host_only_ok` (a variant).
fn fills(
    planner: &Budget,
    split: &Split,
    host_only_ok: bool,
) -> Result<(CardExperts, Vec<usize>, u64), FixtureError> {
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
    if eligible.is_empty() && !host_only_ok {
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

/// The per-layer counts `split`'s plan holds under `budget`: at least half on
/// every layer the budgetless plan fills, the budgetless plan's count
/// elsewhere, and `budget` the least such cap to the MiB (the plan a MiB below
/// it falls short of half on some such layer, or is refused). Returns them
/// when they are those, else the error that names what differs.
pub fn check(planner: &Budget, split: &Split, budget: u64) -> Result<CardExperts, FixtureError> {
    check_in(planner, split, budget, false)
}

/// [`check`] for a variant's file, which may be host-only ([`host_only`]): the plan of a file with no
/// card-eligible layer holds no expert on a card under any budget that plans it.
pub fn check_variant(
    planner: &Budget,
    split: &Split,
    budget: u64,
) -> Result<CardExperts, FixtureError> {
    check_in(planner, split, budget, true)
}

/// [`check`], a host-only file allowed when `host_only_ok`.
fn check_in(
    planner: &Budget,
    split: &Split,
    budget: u64,
    host_only_ok: bool,
) -> Result<CardExperts, FixtureError> {
    let (full, eligible, half) = fills(planner, split, host_only_ok)?;
    let got = (planner.card_experts)(split, Some(budget)).map_err(|e| {
        FixtureError::Budget(format!("planning the file under the budget {budget}: {e}"))
    })?;
    let holds = |c: &CardExperts| {
        eligible
            .iter()
            .all(|&l| c.per_layer.get(l).is_some_and(|&n| n >= half))
    };
    let rest_full = full
        .per_layer
        .iter()
        .enumerate()
        .all(|(l, &n)| eligible.contains(&l) || got.per_layer.get(l) == Some(&n));
    if !holds(&got) || !rest_full || got.per_layer.len() != full.per_layer.len() {
        return Err(FixtureError::BudgetMismatch(format!(
            "the plan under the budget {budget} holds {:?} experts per layer, not at least {half} \
             (half of the file's {}) on each layer the budgetless plan fills {eligible:?} and \
             {:?} elsewhere",
            got.per_layer, full.experts, full.per_layer
        )));
    }
    let below = budget.saturating_sub(MIB);
    // A host-only file has no least cap to be: no cap puts an expert on a card.
    if !eligible.is_empty()
        && below > 0
        && let Ok(c) = (planner.card_experts)(split, Some(below))
        && holds(&c)
    {
        return Err(FixtureError::BudgetMismatch(format!(
            "the budget {budget} is not the least: {below}, a MiB smaller, holds {:?}, at least \
             {half} on each of {eligible:?} too",
            c.per_layer
        )));
    }
    Ok(got)
}

/// The grain of a recorded budget.
const MIB: u64 = 1024 * 1024;

/// The byte cap under which planning the written `target` holds at least
/// half of every card-eligible layer's experts on the card: the smallest cap
/// that does (rounded up to a whole MiB when that holds the same counts), as the
/// family's `planner` computes the counts, proved by [`check`] on the
/// header-only files.
pub fn choose(planner: &Budget, target: &FilePlan) -> Result<u64, FixtureError> {
    let headers = Headers::of(target)?;
    choose_in(planner, &headers.first, false)
}

/// Whether the budgetless plan of the written `split` holds experts and none of them on a card: a
/// host-only file, whose budget is the header's own.
///
/// # Errors
/// The planner's refusal of the budgetless plan.
pub fn host_only(planner: &Budget, split: &Split) -> Result<bool, FixtureError> {
    let full = (planner.card_experts)(split, None)?;
    Ok(full.experts > 0 && full.per_layer.iter().all(|&n| n == 0))
}

/// [`choose`] for a file already written (`split` opens it): a variant, whose
/// budget is planned on its own headers by the same search and the same
/// proof, not copied from its trunk.
pub fn choose_written(planner: &Budget, split: &Split) -> Result<u64, FixtureError> {
    choose_in(planner, split, true)
}

/// [`choose_written`] for a file whose plan may be host-only only when `host_only_ok`.
fn choose_in(planner: &Budget, split: &Split, host_only_ok: bool) -> Result<u64, FixtureError> {
    let (_, eligible, half) = fills(planner, split, host_only_ok)?;
    if eligible.is_empty() {
        // Host-only: the header's budget stays, once it plans the file with no expert on a card.
        let recorded = match split.value(KEY_CARD_BUDGET) {
            Some(Value::U64(b)) => *b,
            other => {
                return Err(FixtureError::Budget(format!(
                    "a host-only file keeps the budget its header records, and {KEY_CARD_BUDGET} is \
                     {other:?}, not a u64"
                )));
            }
        };
        check_in(planner, split, recorded, true)?;
        return Ok(recorded);
    }
    // Whether `cap` holds at least half on every card-eligible layer — the
    // monotone predicate the search minimizes — with the planner's refusal
    // (a cap the dense tensors pass) as the reason when none fits.
    let attempt = |cap: u64| -> (bool, Option<String>) {
        match (planner.card_experts)(split, Some(cap)) {
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
    // the counts are still `hi`'s, is the number recorded.
    let rounded = hi.next_multiple_of(MIB);
    let same = |cap: u64| {
        (planner.card_experts)(split, Some(cap))
            .ok()
            .map(|c| c.per_layer)
            == (planner.card_experts)(split, Some(hi))
                .ok()
                .map(|c| c.per_layer)
    };
    let chosen = if rounded > hi && same(rounded) {
        rounded
    } else {
        hi
    };
    check_in(planner, split, chosen, host_only_ok)?;
    Ok(chosen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::synthetic::{V, header, header_with};

    /// A planner whose count is the cap in MiB, except that 4 MiB, the
    /// granule a small expert shares, reaches 5: no cap holds exactly half
    /// (4 of 8), and the least that holds at least half is recorded.
    fn granular(_: &Split, cap: Option<u64>) -> Result<CardExperts, FixtureError> {
        let n = match cap {
            None => 8,
            Some(0) => {
                return Err(FixtureError::Budget("no room for dense weights".into()));
            }
            Some(c) => match c / MIB {
                4 => 5,
                m => m.min(8),
            },
        };
        Ok(CardExperts {
            per_layer: vec![n, 0, n],
            experts: 8,
        })
    }

    /// A planner whose experts never reach a card, at any cap, and which refuses a cap under 3 MiB,
    /// where the dense tensors do not fit.
    fn host_only_plan(_: &Split, cap: Option<u64>) -> Result<CardExperts, FixtureError> {
        match cap {
            Some(c) if c < 3 * MIB => Err(FixtureError::Budget("no room for dense weights".into())),
            _ => Ok(CardExperts {
                per_layer: vec![0, 0, 0],
                experts: 8,
            }),
        }
    }

    #[test]
    fn a_variant_may_be_host_only_and_a_base_fixture_may_not() {
        let path = header_with(
            "budget-host-only",
            "test",
            &[],
            &[(KEY_CARD_BUDGET, V::U64(5 * MIB))],
            &[],
        );
        let split = Split::open(&path).unwrap();
        let planner = Budget {
            ctx: 1,
            card_experts: host_only_plan,
            card_set_aside: None,
        };
        assert!(host_only(&planner, &split).unwrap());
        let granular = Budget {
            ctx: 1,
            card_experts: granular,
            card_set_aside: None,
        };
        assert!(!host_only(&granular, &split).unwrap());
        // A host-only variant keeps the budget its header records, once the file plans under it.
        assert_eq!(choose_written(&planner, &split).unwrap(), 5 * MIB);
        assert_eq!(
            check_variant(&planner, &split, 5 * MIB).unwrap().per_layer,
            [0, 0, 0]
        );
        // No cap puts an expert on a card, so a larger one that plans holds too; one the dense
        // tensors do not fit is the planner's refusal.
        assert!(check_variant(&planner, &split, 9 * MIB).is_ok());
        let short = check_variant(&planner, &split, 2 * MIB).unwrap_err();
        assert!(matches!(short, FixtureError::Budget(_)), "{short}");
        // A base fixture with no card-eligible layer is refused by name.
        let base = check(&planner, &split, 5 * MIB).unwrap_err();
        assert!(matches!(base, FixtureError::Budget(_)), "{base}");
        assert!(
            base.to_string()
                .contains("no layer of the file holds a card expert"),
            "{base}"
        );
        assert!(choose_in(&planner, &split, false).is_err());
        drop(split);
        std::fs::remove_file(&path).unwrap();
        // A host-only file whose header records no budget has none to keep.
        let bare = header("budget-host-only-bare", "test", &[], &[]);
        let split = Split::open(&bare).unwrap();
        let error = choose_written(&planner, &split).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("keeps the budget its header records"),
            "{error}"
        );
        drop(split);
        std::fs::remove_file(bare).unwrap();
    }

    #[test]
    fn a_budget_mismatch_is_distinct_from_a_planner_failure() {
        let path = header("budget-counts", "test", &[], &[]);
        let split = Split::open(&path).unwrap();
        let planner = Budget {
            ctx: 1,
            card_experts: granular,
            card_set_aside: None,
        };
        // The least cap holding at least half: 4 MiB, its granule past half.
        assert_eq!(
            check(&planner, &split, 4 * MIB).unwrap().per_layer,
            [5, 0, 5]
        );
        assert_eq!(choose_written(&planner, &split).unwrap(), 4 * MIB);
        // Short of half, and a larger cap (a trunk's, copied): each refused by name.
        let short = check(&planner, &split, 3 * MIB).unwrap_err();
        assert!(matches!(short, FixtureError::BudgetMismatch(_)), "{short}");
        assert!(short.to_string().contains("not at least 4"), "{short}");
        let large = check(&planner, &split, 6 * MIB).unwrap_err();
        assert!(matches!(large, FixtureError::BudgetMismatch(_)), "{large}");
        assert!(large.to_string().contains("is not the least"), "{large}");
        let error = check(&planner, &split, 0).unwrap_err();
        assert!(matches!(error, FixtureError::Budget(_)), "{error}");
        assert!(
            error.to_string().contains("no room for dense weights"),
            "{error}"
        );
        drop(split);
        std::fs::remove_file(path).unwrap();
    }
}
