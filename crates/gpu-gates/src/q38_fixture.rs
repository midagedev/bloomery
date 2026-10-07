//! What the three Qwen3.8 whole-model gates (`gate_qwen4exp_e2e`, `gate_qwen4exp_mtp`,
//! `gate_qwen38_twocard`) read from the file they open instead of carrying the real file's
//! numbers: its layer kinds from the header (`block_count` and `full_attention_interval`), the
//! captured graphs' node counts from those kinds, the card the plan is made on, the plan's card
//! budget, and the draft file beside the target.
//!
//! The real tier (`BLOOMERY_TIER` unset or `real`) plans on the 3090's bytes on whichever card the
//! gate runner put in view, with the budget lever the caller set; every figure derived here then
//! equals the literal it replaced, and a gate prints each beside it ([`witness`]). The fixture
//! tier (`crates/gpu-gates/src/tier.rs`) plans with `a` (the largest visible card, as `--place a`
//! does), under the budget the generator recorded in the fixture's header, and a gate's clauses
//! run by their tag ([`clause`]).

use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

use bloomery_levers::Levers;
use gguf::Split;
use model::fixture::KEY_CARD_BUDGET;
use model::placement::PlanLevers;

use crate::tier::{Tag, Tier};
use crate::{GateError, verdict};

/// Launches of the embedding row.
pub const EMBED_LAUNCHES: usize = 1;
/// Launches of a delta layer: its two mixes' 3 each, the mixer's 7 (q·k·v, z, β·α, the conv, the
/// delta step, the gated norm, the output projection) and the block's 9 (the router, the handoff,
/// the go, the shared expert's four, the wait, the gated sum).
pub const GDN_LAYER_LAUNCHES: usize = 6 + 7 + 9;
/// Launches of a selecting layer: the same 6 and 9 around the mixer's 14.
pub const QSA_LAYER_LAUNCHES: usize = 6 + 14 + 9;
/// Launches of the PLE site: the combine, key and value, the gate and the conv.
pub const PLE_SITE_LAUNCHES: usize = 5;
/// Launches of the head: its mix's three, the q8_0 gemv and the argmax.
pub const HEAD_LAUNCHES: usize = 5;
/// Nodes a verify of more than one row adds on a delta layer (β and α out of the joined
/// projection, two token-major copies) and on a selecting layer (the indexer queries, one).
pub const VERIFY_GDN_EXTRA: usize = 2;
pub const VERIFY_QSA_EXTRA: usize = 1;
/// Launches bound to one sequence on a selecting layer, and on a delta layer (its conv and delta
/// step); a captured pass of several slots' rows repeats them for each busy slot.
pub const SLOT_QSA_LAUNCHES: usize = 8;
pub const SLOT_GDN_LAUNCHES: usize = 2;
/// Launches the card leg adds on a card layer: the normed rows' q8_1, the gate·up, the card
/// slots' q8_1, the down and the card sum.
pub const CARD_LEG_LAUNCHES: usize = 5;

/// The layer kinds a file's header names: what a gate's structure clause is written against, and
/// the only source it reads them from (never the loaded body).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    /// `block_count`.
    pub n_layer: usize,
    /// `full_attention_interval`.
    pub interval: usize,
    /// The selecting layers: llama.cpp's `(il + 1) % interval == 0`.
    pub qsa: Vec<usize>,
    /// `ple.layers`: the layer the PLE site sits on; `None` when the file lists no site.
    pub ple_layer: Option<usize>,
    /// `ple.image_token_id`: the placeholder the file's PLE hash refuses as an input.
    pub image_token: Option<u32>,
}

/// The selecting layers of `n_layer` layers at `interval`.
#[must_use]
pub fn qsa_layers(n_layer: usize, interval: usize) -> Vec<usize> {
    (0..n_layer)
        .filter(|l| (l + 1).is_multiple_of(interval))
        .collect()
}

impl Shape {
    /// The shape of the header values themselves.
    ///
    /// # Errors
    /// A file with no layer or an interval of 0, or a PLE site past the layers.
    pub fn from_header(
        n_layer: usize,
        interval: usize,
        ple_layer: Option<usize>,
        image_token: Option<u32>,
    ) -> Result<Shape, GateError> {
        if n_layer == 0 || interval == 0 {
            return Err(
                format!("block_count {n_layer} and full_attention_interval {interval}").into(),
            );
        }
        if ple_layer.is_some_and(|l| l >= n_layer) {
            return Err(format!("the PLE site {ple_layer:?} is past the {n_layer} layers").into());
        }
        Ok(Shape {
            n_layer,
            interval,
            qsa: qsa_layers(n_layer, interval),
            ple_layer,
            image_token,
        })
    }

    /// Selecting layers.
    #[must_use]
    pub fn n_qsa(&self) -> usize {
        self.qsa.len()
    }

    /// Delta layers.
    #[must_use]
    pub fn n_gdn(&self) -> usize {
        self.n_layer - self.qsa.len()
    }

    /// The captured decode step's nodes: the embedding row, each layer's launches, the PLE site's
    /// and the head's.
    #[must_use]
    pub fn nodes_decode(&self) -> usize {
        EMBED_LAUNCHES
            + self.n_gdn() * GDN_LAYER_LAUNCHES
            + self.n_qsa() * QSA_LAYER_LAUNCHES
            + self.ple_layer.map_or(0, |_| PLE_SITE_LAUNCHES)
            + HEAD_LAUNCHES
    }

    /// The stream memory-operation batches of a captured step: each layer's go and wait.
    #[must_use]
    pub fn memops(&self) -> usize {
        2 * self.n_layer
    }

    /// The captured verify's nodes at 2, 3 and 4 rows.
    #[must_use]
    pub fn nodes_verify(&self) -> usize {
        self.nodes_decode() + self.n_gdn() * VERIFY_GDN_EXTRA + self.n_qsa() * VERIFY_QSA_EXTRA
    }

    /// The nodes one more busy slot adds to a captured pass of several slots' rows: the
    /// embedding row, the PLE conv, each layer's sequence-bound launches and the parked rows'
    /// copy.
    #[must_use]
    pub fn nodes_slot(&self) -> usize {
        EMBED_LAUNCHES
            + usize::from(self.ple_layer.is_some())
            + self.n_gdn() * SLOT_GDN_LAUNCHES
            + self.n_qsa() * SLOT_QSA_LAUNCHES
            + 1
    }

    /// The captured step's nodes under a card plan that puts routed experts on `card_layers`
    /// layers.
    #[must_use]
    pub fn nodes_decode_card(&self, card_layers: usize) -> usize {
        self.nodes_decode() + CARD_LEG_LAUNCHES * card_layers
    }

    /// The captured verify's nodes under that plan.
    #[must_use]
    pub fn nodes_verify_card(&self, card_layers: usize) -> usize {
        self.nodes_verify() + CARD_LEG_LAUNCHES * card_layers
    }
}

/// The gate's one tier: `BLOOMERY_TIER`, refused by name when it is neither.
///
/// # Errors
/// [`crate::tier::TierError::BadValue`].
pub fn tier() -> Result<Tier, GateError> {
    Ok(Tier::from_env()?)
}

static RAN: AtomicUsize = AtomicUsize::new(0);
static DEFERRED: AtomicUsize = AtomicUsize::new(0);

/// Whether this process's tier runs the clause `name`, which carries `tag`; when it does not, the
/// clause's [`crate::tier::DEFERRED`] line is printed (the batch counts it). Every verdict a gate
/// prints is decided by one of these calls (the clause's function, or the sub-assertion that
/// compares with ik), so a clause with no tag has no way to run.
///
/// # Errors
/// [`crate::tier::TierError::BadValue`].
pub fn clause(name: &str, tag: Tag) -> Result<bool, GateError> {
    let runs = tier()?.clause(name, Some(tag))?;
    if runs {
        RAN.fetch_add(1, Ordering::Relaxed);
    } else {
        DEFERRED.fetch_add(1, Ordering::Relaxed);
    }
    Ok(runs)
}

/// The clauses decided so far: how many ran and how many were deferred, as the gate's closing line
/// prints them.
#[must_use]
pub fn tally() -> (usize, usize) {
    (
        RAN.load(Ordering::Relaxed),
        DEFERRED.load(Ordering::Relaxed),
    )
}

/// The move proof of one derived value: in the real tier the value beside the literal it replaced,
/// and whether they are equal; in the fixture tier the derived value beside the real file's, with
/// no verdict (the fixture's shape differs by construction).
pub fn witness<T: PartialEq + Debug>(name: &str, derived: T, real: T) -> bool {
    match Tier::from_env() {
        Ok(Tier::Fixture) => {
            println!("derived {name} = {derived:?} (the real file's {real:?})");
            true
        }
        _ => {
            let ok = derived == real;
            println!(
                "move proof {name}: derived {derived:?}, the old literal {real:?} {}",
                verdict(ok)
            );
            ok
        }
    }
}

/// The key the generator records the single-card budget under, read from a header.
///
/// # Errors
/// A key of another type than u64.
pub fn header_budget(split: &Split) -> Result<Option<u64>, GateError> {
    match split.value(KEY_CARD_BUDGET) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{KEY_CARD_BUDGET} holds {v:?}, not a u64").into()),
    }
}

/// The plan's levers: the real tier's are the caller's (`BLOOMERY_CARD_BUDGET`, unset: none, as
/// every gate had them); the fixture tier's budget is the one the generator recorded in the header
/// — the budget that puts half the experts on the card of every card-eligible layer — plus `extra`,
/// the card bytes a drafting load reserves out of the same budget (the draft's `card_bytes`; 0 for
/// a target-only plan). A budget lever the caller set that differs from the header is refused by
/// name, never planned under.
///
/// # Errors
/// The fixture tier's file with no budget key, or a lever that names another budget.
pub fn plan_levers(split: &Split, levers: &Levers, extra: u64) -> Result<PlanLevers, GateError> {
    budget_levers(
        tier()?,
        header_budget(split)?,
        levers.card_budget_bytes(),
        extra,
    )
}

/// [`plan_levers`] of its three inputs: the tier, the header's budget and the caller's lever.
///
/// # Errors
/// As [`plan_levers`].
pub fn budget_levers(
    tier: Tier,
    header: Option<u64>,
    lever: Option<u64>,
    extra: u64,
) -> Result<PlanLevers, GateError> {
    match tier {
        Tier::Real => Ok(PlanLevers {
            card_budget_bytes: lever,
        }),
        Tier::Fixture => {
            let header =
                header.ok_or_else(|| format!("the fixture header has no {KEY_CARD_BUDGET}"))?;
            if let Some(l) = lever
                && l != header
            {
                return Err(format!(
                    "BLOOMERY_CARD_BUDGET={l} and the fixture's {KEY_CARD_BUDGET} is {header}: \
                     the fixture tier plans under the header's budget"
                )
                .into());
            }
            Ok(PlanLevers {
                card_budget_bytes: Some(header + extra),
            })
        }
    }
}

/// The card a plan is made on, and the machine's usable bytes with it. The real tier keeps what
/// the gates had (`real`: the 3090's bytes on the card the runner put in view). The fixture tier
/// resolves `a` against this process's devices (`--place a`): the largest visible card as the
/// device reports itself, its free bytes included.
///
/// # Errors
/// `a` finding no card.
#[cfg(feature = "gpu")]
pub fn card(
    real: impl FnOnce() -> Result<model::placement::workstation::CardSpec, GateError>,
) -> Result<model::placement::workstation::CardSpec, GateError> {
    use std::sync::OnceLock;

    use model::placement::workstation::CardSpec;

    static A: OnceLock<CardSpec> = OnceLock::new();
    match tier()? {
        Tier::Real => real(),
        Tier::Fixture => {
            if let Some(c) = A.get() {
                return Ok(*c);
            }
            let specs = crate::generate::Place::A.on_host()?.card_specs()?;
            let card = *specs.first().ok_or("--place a resolved to no card")?;
            Ok(*A.get_or_init(|| card))
        }
    }
}

/// The move proof of the card a plan is made on: its bytes (total, the driver's reserve, the
/// resolved device, the free bytes, the holders) beside `RTX_3090`'s, the lookup every plan made
/// before the card was derived. The real tier requires them equal (`gate_card::card` keeps every
/// byte of `RTX_3090`); the name is the one difference, printed apart: the plan lookup opens the
/// card in view by its own name.
#[cfg(feature = "gpu")]
pub fn witness_card(derived: &model::placement::workstation::CardSpec) -> bool {
    use model::placement::workstation::{CardSpec, RTX_3090};

    let bytes = |c: &CardSpec| {
        (
            c.total_bytes,
            c.driver_reserve_bytes,
            c.device,
            c.free_bytes,
            c.held_by,
        )
    };
    let ok = witness(
        "card bytes (total, driver reserve, device, free, holders)",
        bytes(derived),
        bytes(&RTX_3090),
    );
    println!(
        "named difference: the card's name {:?}, the old literal's {:?}",
        derived.name, RTX_3090.name
    );
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real file's header (48 layers, interval 4, the PLE site on layer 1) gives the literals
    /// the gates carried; the fixture's (4 layers, interval 2, the site on layer 2) gives the
    /// counts its derivation names. Mutant: any launch constant, the interval rule or a count off
    /// by one.
    #[test]
    fn the_header_gives_the_real_files_literals_and_the_fixtures_counts() {
        let real = Shape::from_header(48, 4, Some(1), Some(248_056)).expect("real");
        assert_eq!(real.qsa, (0..12).map(|i| 4 * i + 3).collect::<Vec<_>>());
        assert_eq!((real.n_qsa(), real.n_gdn()), (12, 36));
        assert_eq!(real.nodes_decode(), 1151);
        assert_eq!(real.memops(), 96);
        assert_eq!(real.nodes_verify(), 1235);
        assert_eq!(real.nodes_slot(), 171);
        assert_eq!(real.nodes_decode_card(48), 1391);
        assert_eq!(real.nodes_verify_card(48), 1475);
        let fix = Shape::from_header(4, 2, Some(2), Some(248_056)).expect("fixture");
        assert_eq!(fix.qsa, [1, 3]);
        assert_eq!((fix.n_qsa(), fix.n_gdn()), (2, 2));
        assert_eq!(fix.nodes_decode(), 1 + 2 * 22 + 2 * 29 + 5 + 5);
        assert_eq!(fix.nodes_decode(), 113);
        assert_eq!(fix.nodes_verify(), 119);
        assert_eq!(fix.nodes_slot(), 23);
        assert_eq!(fix.nodes_decode_card(4), 133);
        assert_eq!(fix.nodes_verify_card(4), 139);
    }

    /// The real tier plans under the caller's lever whatever the header holds; the fixture tier
    /// under the header's budget plus a drafting load's card bytes, and refuses a lever that
    /// names another budget and a header with none. Mutant: the extra dropped or doubled, the
    /// lever preferred, the real tier reading the header.
    #[test]
    fn the_budget_is_the_levers_in_the_real_tier_and_the_headers_plus_the_draft_in_the_fixture() {
        let b = |t, h, l, x| budget_levers(t, h, l, x).map(|p| p.card_budget_bytes);
        assert_eq!(b(Tier::Real, Some(9), None, 5).expect("real"), None);
        assert_eq!(b(Tier::Real, None, Some(7), 5).expect("real"), Some(7));
        assert_eq!(b(Tier::Fixture, Some(9), None, 0).expect("plain"), Some(9));
        assert_eq!(
            b(Tier::Fixture, Some(9), Some(9), 5).expect("draft"),
            Some(14)
        );
        assert!(b(Tier::Fixture, Some(9), Some(8), 0).is_err());
        assert!(b(Tier::Fixture, None, None, 0).is_err());
    }

    /// A header no gate can read a shape from is a named error, and a file with no PLE site has
    /// no site's launches.
    #[test]
    fn a_header_without_layers_or_interval_is_refused_and_no_site_adds_no_launches() {
        assert!(Shape::from_header(0, 4, None, None).is_err());
        assert!(Shape::from_header(48, 0, None, None).is_err());
        assert!(Shape::from_header(4, 2, Some(4), None).is_err());
        let bare = Shape::from_header(4, 2, None, None).expect("no site");
        assert_eq!(bare.nodes_decode(), 108);
        assert_eq!(bare.nodes_slot(), 22);
    }
}
