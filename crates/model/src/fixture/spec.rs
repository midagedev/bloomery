//! What a family supplies to the generator ([`FixtureSpec`]), and the
//! options a plan takes ([`Options`]).

use gguf::{Split, Value};

use super::{DEFAULT_SEED, DEFAULT_SHARD_BYTES, FixtureError, Kvs, Window, meta};

/// A family's fixture. Its facts are values, so a variant — a smaller map
/// for a smaller host, a narrower ff — is the same spec with fields replaced
/// (`FixtureSpec { layers, ratios, ..spec() }`); its rules are the methods of
/// `family`, written once per family and shared by every variant.
#[derive(Clone)]
pub struct FixtureSpec {
    /// The `general.architecture` a source declares; any other is refused
    /// by name.
    pub arch: &'static str,
    /// Shard file names are `<stem>-0000i-of-0000N.gguf`.
    pub stem: &'static str,
    /// Fixture layer `f` holds source layer `layers[f]`.
    pub layers: Vec<usize>,
    /// What a [`KeyRule::Ratios`] key reads at `layers`: the check that the
    /// map still picks the layer kinds it was chosen for. Empty when the
    /// family has no such key.
    pub ratios: Vec<u64>,
    /// The card budget a fixture records when the caller gives none.
    pub card_budget: CardBudget,
    /// The range every block's `d` (and `dmin`) is drawn inside, by the
    /// rules and checked by `verify`; a type whose scale cannot fit it at a
    /// tensor's K is refused by name.
    pub window: Window,
    /// The routed experts' feed-forward width the fixture writes in place
    /// of the source's: the [`KeyRule::Ff`] key, and the axis
    /// [`Family::ff_axis`] names on each tensor that carries it. `None`
    /// keeps the source's.
    pub ff: Option<u64>,
    /// The source a `verify` reads when none is named.
    pub default_source: fn() -> String,
    /// The draft fixture, for a family with a draft file.
    pub draft: Option<DraftSpec>,
    /// The family's rules.
    pub family: &'static dyn Family,
}

impl FixtureSpec {
    /// The options a plan takes when the caller sets none: [`DEFAULT_SEED`],
    /// this spec's card budget, [`DEFAULT_SHARD_BYTES`], every tensor.
    pub fn options(&self) -> Options {
        Options {
            seed: DEFAULT_SEED,
            card_budget: None,
            shard_bytes: DEFAULT_SHARD_BYTES,
            tensors: None,
            draft_tensors: None,
        }
    }
}

/// The card budget a spec records when the caller names none.
#[derive(Clone, Copy)]
pub enum CardBudget {
    /// This many bytes, whatever the written file plans.
    Fixed(u64),
    /// The cap [`budget::choose`](super::budget::choose) finds for the
    /// written file: the family's own plan of it holds half of every
    /// card-eligible layer's experts on the card.
    Planned(Budget),
}

/// A family's plan of its written file under a card budget, which
/// [`CardBudget::Planned`] searches the cap through. The chosen cap's
/// conditions — the context the plan runs at, and the machine and resident
/// slots [`Budget::card_experts`] names — are the family's; wherever the
/// recorded value is read they are named with it.
#[derive(Clone, Copy)]
pub struct Budget {
    /// The positions the choosing plan runs at.
    pub ctx: u64,
    /// The per-layer card expert counts of the written file planned under
    /// `budget` bytes (`None`: each card's own usable bytes), and the file's
    /// expert count a layer's share is half of; a planner refusal is the
    /// error.
    pub card_experts: fn(&Split, Option<u64>) -> Result<CardExperts, FixtureError>,
}

/// What [`Budget::card_experts`] returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardExperts {
    /// Per layer of the file: the experts its stage card holds.
    pub per_layer: Vec<u64>,
    /// The file's expert count, one number for every layer.
    pub experts: u64,
}

/// A family's draft file: a separate model whose fixture is written beside
/// the target's.
#[derive(Clone, Copy)]
pub struct DraftSpec {
    /// The `general.architecture` the real draft declares.
    pub arch: &'static str,
    /// The draft fixture's path inside the fixture directory. A directory
    /// of its own keeps a reader that takes every `*.gguf` beside the
    /// target's shards from meeting it.
    pub file: &'static str,
    /// The real draft a `verify` reads when none is named, if any.
    pub default_source: fn() -> Option<String>,
    /// The draft's own rules.
    pub rules: &'static dyn DraftRules,
}

/// What the fixture does with the source's architecture key `<arch>.<suffix>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyRule {
    /// The source's value.
    Copy,
    /// The fixture's layer count, in the source's integer type.
    BlockCount,
    /// An array of exactly the source's layer count, taken at the map's
    /// layers; a scalar is copied.
    PerLayer,
    /// A per-layer array that may run past the source's layer count: the
    /// map's layers, whose values must be [`FixtureSpec::ratios`], then the
    /// entries past the source's layer count.
    Ratios,
    /// A list of source layers, each moved to its fixture layer; a layer
    /// the map does not hold is refused.
    LayerIds,
    /// The source's value, which must be 0: the reason names the layers the
    /// map does not carry.
    Zero(&'static str),
    /// The rule's value in place of the source's, in the source's integer
    /// type: a fact of the map, not of the source (an interval the fixture
    /// re-derives its layer kinds from, say). The key must be in the source:
    /// [`Family::required_keys`] names it.
    Set(u64),
    /// [`FixtureSpec::ff`] when set, else the source's value.
    Ff,
    /// The family's [`Tables::key`] computes it.
    Table,
}

/// A family's rules: what every variant of its spec shares.
pub trait Family: Sync {
    /// The rule of the source's `<arch>.<suffix>`; `None` is a key the
    /// fixture does not cover, which plans refuse by name.
    fn key_rule(&self, suffix: &str) -> Option<KeyRule>;

    /// The value of every element of a 1-D F32 tensor, by its name past
    /// `blk.N.`; `None` is a 1-D tensor with no fill rule.
    fn const_value(&self, leaf: &str) -> Option<f32>;

    /// The architecture keys (past `<arch>.`) every source must carry,
    /// because the fixture's header is not right without them: a key a
    /// [`KeyRule::Set`] rewrites, which the plan only walks if the source
    /// has it. A source without one is refused by name.
    fn required_keys(&self) -> &'static [&'static str] {
        &[]
    }

    /// The axis of tensor `leaf` (its name past `blk.N.`) that is the
    /// routed experts' feed-forward width, which [`FixtureSpec::ff`]
    /// replaces. A family that names none refuses an ff override.
    fn ff_axis(&self, leaf: &str) -> Option<usize> {
        let _ = leaf;
        None
    }

    /// The family's tables for a fixture of `spec` cut from `source`: the
    /// keys and tensor dims that depend on the map beyond renaming.
    fn tables(&self, source: &Split, spec: &FixtureSpec) -> Result<Box<dyn Tables>, FixtureError> {
        let _ = (source, spec);
        Ok(Box::new(NoTables))
    }

    /// A whole fixture's header reads, through the engine's own reader, as
    /// the source's layer kinds with their layer references moved through
    /// `spec.layers`.
    fn check_kinds(
        &self,
        spec: &FixtureSpec,
        fixture: &Split,
        source: &Split,
    ) -> Result<(), FixtureError>;
}

/// State a family reads from the source for one plan.
pub trait Tables {
    /// The value of the [`KeyRule::Table`] key `key` (`<arch>.<suffix>`),
    /// whose source value is `v`.
    fn key(&self, key: &str, suffix: &str, v: &Value) -> Result<Value, FixtureError>;

    /// The fixture dims of source tensor `name` (in source layer `layer`,
    /// `None` for a global; `leaf` its name past `blk.N.`) whose source dims
    /// are `dims`; `None` keeps them.
    fn dims(
        &self,
        name: &str,
        layer: Option<usize>,
        leaf: &str,
        dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError>;

    /// One line per table for a plan's print.
    fn lines(&self) -> Vec<String>;
}

/// The tables of a family that has none.
struct NoTables;

impl Tables for NoTables {
    fn key(&self, key: &str, _suffix: &str, _v: &Value) -> Result<Value, FixtureError> {
        Err(meta(key, "is a table key of a family with no tables"))
    }

    fn dims(
        &self,
        _name: &str,
        _layer: Option<usize>,
        _leaf: &str,
        _dims: &[u64],
    ) -> Result<Option<Vec<u64>>, FixtureError> {
        Ok(None)
    }

    fn lines(&self) -> Vec<String> {
        Vec::new()
    }
}

/// A draft file's rules.
pub trait DraftRules: Sync {
    /// The draft fixture's metadata, in the real draft's order, for a target
    /// fixture of `n_fixture` layers cut from `n_source`. The draft fixture is
    /// one file: a split key is refused before this is asked.
    fn kvs(&self, draft: &Split, n_source: usize, n_fixture: usize) -> Result<Kvs, FixtureError>;

    /// The fixture name of the real draft's tensor `name`, for a target
    /// fixture of `n_fixture` layers cut from `n_source`: the identity — a
    /// draft whose own layer indices the target does not read.
    fn tensor(
        &self,
        name: &str,
        n_source: usize,
        n_fixture: usize,
    ) -> Result<String, FixtureError> {
        let _ = (n_source, n_fixture);
        Ok(name.to_string())
    }

    /// The draft fixture reads, through the engine's own reader, as a draft
    /// of `target`; with both files `whole`, its inventory against the
    /// target holds too.
    fn check(
        &self,
        spec: &FixtureSpec,
        draft: &Split,
        target: &Split,
        whole: bool,
    ) -> Result<(), FixtureError>;
}

/// What `plan`, `generate` and `verify` share.
#[derive(Clone, Debug)]
pub struct Options {
    pub seed: u64,
    /// The card budget the fixture records; `None` when the caller sets
    /// none — the spec's, or its planner's ([`Budget`]).
    pub card_budget: Option<u64>,
    pub shard_bytes: u64,
    /// Only these target tensors (a subset file), in plan order.
    pub tensors: Option<Vec<String>>,
    /// Only these draft tensors.
    pub draft_tensors: Option<Vec<String>>,
}
