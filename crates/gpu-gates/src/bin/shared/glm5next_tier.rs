//! What a GLM-5.3-Flash whole-model gate reads of the fixture tier beyond `bloomery_gpu_gates::tier`:
//! the model file (one owner: `ref_model_path`, and in the fixture tier a whole fixture), the card
//! the gate plan is made on, the card budget the plan runs under, the layer kinds and counts a
//! literal used to stand for, the ids a gate reads from the oracle sets (which a fixture has no
//! set for), and the open of an oracle set itself, which the fixture tier refuses by name.
//!
//! Every number here is read from the header (`Hparams`, the tensors' types) or from a plan, and
//! the gate that used its literal prints the two side by side (`tier::witness`): equal on the real
//! file, the real file's value beside the fixture's in the fixture tier. The `REAL_*` constants
//! are those literals, kept only as the witness's other side.

#![allow(
    dead_code,
    reason = "every GLM gate includes this file and reads the part of it its clauses use"
)]

use std::sync::OnceLock;

use bloomery_gpu_gates::tier::{self, Tier};
use bloomery_gpu_gates::{GateError, RefManifest, data_dir, ref_model_path};
use gguf::{GgmlType, Split};
use model::arch::glm5next::hparams::{Hparams, Kind};
use model::arch::glm5next::names;
use model::arch::glm5next::place::{PlanInputs, dense_positions};
use model::placement::PlanLevers;
use model::placement::workstation::CardSpec;
use refset::arch::glm5next::{BATCH, D1K, IK, MODEL, MTP, MTP_SET};
use refset::family::Family;
use refset::mtpref::MtpSet;

/// The tokens of the batch set (`REF_TOKENS` of `tools/ref/models/glm5next.sh`): the ids the
/// fixture tier runs where the real tier reads them from ik's set.
pub const BATCH_TOKENS: [u32; 5] = [785, 6722, 315, 9621, 374];

/// The prompt of the 1,024-position step set: its first 1,024 ids of the prose corpus.
pub const D1K_PREFILL: usize = 1024;

/// The prompt of ik's MTP draft set: its first 64 ids of the prose corpus.
pub const MTP_PROMPT: usize = 64;

/// The real file's shape, as the gates carried it: the trunk's layers, the dense prefix, the
/// latent layers, the routed layers whose downs are Q6_K, the decode step's captured nodes and the
/// memory-operation batches among them. Each derived value of [`Shape`] is held equal to its
/// literal in the real tier.
pub const REAL_N_LAYER: usize = 45;
pub const REAL_N_DENSE: usize = 3;
pub const REAL_N_LATENT: usize = 11;
pub const REAL_N_KDA: usize = 34;
pub const REAL_N_ROUTED: usize = 42;
pub const REAL_Q6K_DOWN: [usize; 3] = [11, 12, 44];
pub const REAL_NODES_DECODE: usize = 1169;
pub const REAL_MEMOPS: usize = 84;

/// Launches of a layer's two sub-layers' streams: `hc_pre_q8_0`, the fold and `hc_post`, three
/// each.
const STREAM_LAUNCHES: usize = 6;
/// Launches of a KDA mixer: the norm, four q8_0 projections of the normed row (the q·k·v joined,
/// the two low-rank halves, β) and two of the halves, the conv and prep, the delta step, the gated
/// norm, the output projection.
const KDA_LAUNCHES: usize = 11;
/// Launches of a latent mixer: the norm, the joined projection, the q_a norm, two appends (the
/// selector's pool, head weights, indexer query, scores and top-k are on the branch and add no
/// node), q_b, k_b, the attention's two launches, v_b, the output projection.
const LATENT_LAUNCHES: usize = 16;
/// Launches of a dense block: the norm, gate·up, down.
const DENSE_LAUNCHES: usize = 3;
/// Launches of a routed block: the norm, the router, the handoff, the go, the shared gate·up and
/// down, the wait, the sum.
const ROUTED_LAUNCHES: usize = 8;
/// Launches of the head: the streams' mean, the norm, the q8_0 gemv, the argmax.
const HEAD_LAUNCHES: usize = 4;

/// What a file's header and tensor types say of its trunk: the layer kinds the gates' structure
/// clause is written against, and the only source it reads them from (never the loaded body).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    /// The trunk's layers (`block_count` less the next-token layers).
    pub n_layer: usize,
    /// The leading dense blocks.
    pub n_dense: usize,
    /// The latent-attention layers, in order.
    pub latent: Vec<usize>,
    /// The routed layers whose down stack is Q6_K, which the card experts do not read.
    pub q6k_down: Vec<usize>,
}

impl Shape {
    /// The shape of `hp` and of the routed down stacks' types in `split`.
    ///
    /// # Errors
    /// A routed layer whose down stack is not in the file.
    pub fn read(hp: &Hparams, split: &Split) -> Result<Shape, GateError> {
        let latent = hp
            .kinds
            .iter()
            .take(hp.n_trunk)
            .enumerate()
            .filter(|(_, k)| **k == Kind::Latent)
            .map(|(l, _)| l)
            .collect();
        let mut q6k_down = Vec::new();
        for l in hp.dense_lead..hp.n_trunk {
            let name = names::ffn_down_exps(l);
            let (_, info) = split
                .find(&name)
                .ok_or_else(|| format!("{name} is not in the file"))?;
            if info.ty == GgmlType::Q6_K {
                q6k_down.push(l);
            }
        }
        Ok(Shape {
            n_layer: hp.n_trunk,
            n_dense: hp.dense_lead,
            latent,
            q6k_down,
        })
    }

    /// Latent-attention layers.
    #[must_use]
    pub fn n_latent(&self) -> usize {
        self.latent.len()
    }

    /// KDA layers.
    #[must_use]
    pub fn n_kda(&self) -> usize {
        self.n_layer - self.latent.len()
    }

    /// Routed layers.
    #[must_use]
    pub fn n_routed(&self) -> usize {
        self.n_layer - self.n_dense
    }

    /// PIN(2026-09-28): the captured decode step's node count, derived before the chain was built
    /// from the launches above: `n_layer`·6 + `n_kda`·11 + `n_latent`·16 + `n_dense`·3 +
    /// `n_routed`·8 + 4.
    #[must_use]
    pub fn nodes_decode(&self) -> usize {
        self.n_layer * STREAM_LAUNCHES
            + self.n_kda() * KDA_LAUNCHES
            + self.n_latent() * LATENT_LAUNCHES
            + self.n_dense * DENSE_LAUNCHES
            + self.n_routed() * ROUTED_LAUNCHES
            + HEAD_LAUNCHES
    }

    /// PIN(2026-09-27): each routed layer's go and wait.
    #[must_use]
    pub fn memops(&self) -> usize {
        2 * self.n_routed()
    }

    /// The real tier's move proof: every derived value beside the literal it replaced. In the
    /// fixture tier the same lines print the real file's value beside the fixture's, with no
    /// verdict.
    #[must_use]
    pub fn witness(&self) -> bool {
        let real_latent: Vec<usize> = (0..REAL_N_LAYER).filter(|l| l % 4 == 3).collect();
        let mut ok = true;
        for (name, derived, real) in [
            ("trunk layers", self.n_layer, REAL_N_LAYER),
            ("dense blocks", self.n_dense, REAL_N_DENSE),
            ("latent layers", self.n_latent(), REAL_N_LATENT),
            ("KDA layers", self.n_kda(), REAL_N_KDA),
            ("routed layers", self.n_routed(), REAL_N_ROUTED),
            ("decode step nodes", self.nodes_decode(), REAL_NODES_DECODE),
            ("memory-operation batches", self.memops(), REAL_MEMOPS),
        ] {
            ok &= tier::witness(name, derived, real);
        }
        ok &= tier::witness("latent layer ids", self.latent.clone(), real_latent);
        ok &= tier::witness(
            "routed layers with a Q6_K down",
            self.q6k_down.clone(),
            REAL_Q6K_DOWN.to_vec(),
        );
        ok
    }
}

/// What [`init`] read once: the file's path, its header's card budget, its hyperparameters and
/// its shape.
struct Facts {
    path: String,
    budget: Option<u64>,
    hp: Hparams,
    shape: Shape,
}

static FACTS: OnceLock<Facts> = OnceLock::new();

/// The model file a gate opens: `$BLOOMERY_REF_MODEL` (`ref_model_path`), which in the fixture
/// tier is proven to be a whole fixture, so a gate never runs on the real file under the
/// fixture's name. In the real tier it is [`MODEL`], the file ik's sets were dumped from, and the
/// real tier's move proof says so.
///
/// # Errors
/// The owner's refusal, or a real-tier path that is not [`MODEL`].
pub fn model_path() -> Result<String, GateError> {
    if let Some(f) = FACTS.get() {
        return Ok(f.path.clone());
    }
    let path = ref_model_path()?.display().to_string();
    if !tier::witness("model file", path.clone(), MODEL.to_string()) {
        return Err(format!(
            "the model file is {path}: the real tier's gates run on {MODEL}, the file their oracle \
             sets were dumped from"
        )
        .into());
    }
    Ok(path)
}

/// The model file as a split.
///
/// # Errors
/// As [`model_path`], or a file that cannot be opened.
pub fn open() -> Result<Split, GateError> {
    let path = model_path()?;
    Split::open(&path).map_err(|e| format!("open {path}: {e}").into())
}

/// [`init_file`] after fixing the card `tier::plan_gate` plans on (`tier::init_card`: in the real
/// tier `real_card`, the bin's `gate_card::init`).
///
/// # Errors
/// `tier::init_card`'s, or [`init_file`]'s.
pub fn init(real_card: impl FnOnce() -> Result<CardSpec, GateError>) -> Result<(), GateError> {
    tier::init_card(real_card)?;
    init_file()
}

/// Fixes everything a gate reads of the file once, before any load: the model file, its header's
/// card budget, its hyperparameters and its [`Shape`], each with its move proof. A proof that
/// fails is the error here, before the first load.
///
/// # Errors
/// A file the tier refuses, a header the reader refuses, or a real tier's value that is not the
/// literal it replaced.
pub fn init_file() -> Result<(), GateError> {
    let path = model_path()?;
    let split = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
    let inputs = PlanInputs::read(&split)?;
    let shape = Shape::read(&inputs.hp, &split)?;
    if !shape.witness() {
        return Err("the file's shape is not the literal one in the real tier".into());
    }
    let budget = tier::header_budget(&split)?;
    FACTS.get_or_init(|| Facts {
        path,
        budget,
        hp: inputs.hp,
        shape,
    });
    Ok(())
}

fn facts() -> &'static Facts {
    FACTS
        .get()
        .expect("glm5next_tier::init runs before the gate reads the file's facts")
}

/// The file's shape, read at [`init`].
///
/// # Panics
/// Before [`init`].
#[must_use]
pub fn shape() -> &'static Shape {
    &facts().shape
}

/// The plan's levers: the real tier's are the caller's; the fixture tier's budget is the header's
/// plus `extra`, the card bytes a drafting load reserves out of the same budget (`tier::plan_levers`
/// of the header read at [`init`]).
///
/// # Errors
/// A `BLOOMERY_TIER` that is neither tier, a fixture header with no budget, or a lever that names
/// another budget.
///
/// # Panics
/// Before [`init`].
pub fn plan_levers(levers: &bloomery_levers::Levers, extra: u64) -> Result<PlanLevers, GateError> {
    tier::budget_levers(
        Tier::from_env()?,
        facts().budget,
        levers.card_budget_bytes(),
        extra,
    )
}

/// The position the latent layers attend whole to (`place::dense_positions`), which a gate's load
/// at that many positions holds `literal` for; the same in both tiers (the generator copies the
/// indexer's keys), so a header that differs is the error here.
///
/// # Errors
/// A header whose count is not `literal`.
///
/// # Panics
/// Before [`init`].
pub fn dense_ctx(literal: usize) -> Result<usize, GateError> {
    let derived = usize::try_from(dense_positions(&facts().hp))?;
    if derived == literal {
        Ok(derived)
    } else {
        Err(format!(
            "the indexer keeps every position up to {derived}, the gate's contexts are written for \
             {literal}"
        )
        .into())
    }
}

/// An oracle set of ik's, opened: its manifest, read through its family. The sets were dumped
/// from the real file, so a gate that reads one is an Oracle clause, which the fixture tier defers
/// (`tier::run_clause`) and never opens: a fixture-tier open is refused by name here, whatever the
/// clause's tag says.
///
/// # Errors
/// The fixture tier, or the family's own refusal.
pub fn ik_set(name: &str, family: &Family) -> Result<RefManifest, GateError> {
    if Tier::from_env()? == Tier::Fixture {
        return Err(format!(
            "the oracle set {name} (family {}) was dumped from {MODEL}; BLOOMERY_TIER=fixture runs \
             on a fixture, which has no set of ik's: a clause that reads it is an Oracle clause, \
             deferred to the real tier (crates/gpu-gates/src/tier.rs)",
            family.name
        )
        .into());
    }
    Ok(RefManifest::open(&data_dir().join(name), family)?)
}

/// The batch set's tokens: in the real tier the set's own, held equal to [`BATCH_TOKENS`]; in the
/// fixture tier [`BATCH_TOKENS`] (the ids the set was dumped for, which a fixture's embedding
/// table holds).
///
/// # Errors
/// A set that cannot be opened or whose tokens are not [`BATCH_TOKENS`].
pub fn batch_tokens() -> Result<Vec<u32>, GateError> {
    match Tier::from_env()? {
        Tier::Fixture => Ok(BATCH_TOKENS.to_vec()),
        Tier::Real => {
            let man = ik_set(BATCH, &IK)?;
            let (_, toks, _) = man.step()?;
            let toks = toks.to_vec();
            if tier::witness("batch tokens", toks.clone(), BATCH_TOKENS.to_vec()) {
                Ok(toks)
            } else {
                Err("the batch set's tokens are not REF_TOKENS of the glm5next profile".into())
            }
        }
    }
}

/// The 1,024-position set's prefill ids: in the real tier the set's own, held equal to the first
/// [`D1K_PREFILL`] ids of the prose corpus (the profile's `GLM_PROSE`); in the fixture tier those
/// corpus ids.
///
/// # Errors
/// A set that cannot be opened, a corpus that is short, or a set whose prefill is not the corpus's.
pub fn d1k_prefill() -> Result<Vec<u32>, GateError> {
    static IDS: OnceLock<Vec<u32>> = OnceLock::new();
    if let Some(ids) = IDS.get() {
        return Ok(ids.clone());
    }
    let corpus = tier::prose_ids("glm5next", D1K_PREFILL)?;
    let ids = match Tier::from_env()? {
        Tier::Fixture => corpus,
        Tier::Real => {
            let man = ik_set(D1K, &IK)?;
            let (_, _, prefill) = man.step()?;
            let prefill = prefill.to_vec();
            if !tier::witness(
                "d1k prefill ids = the first corpus ids",
                prefill == corpus,
                true,
            ) {
                return Err("the d1k set's prefill is not the prose corpus's first ids".into());
            }
            prefill
        }
    };
    Ok(IDS.get_or_init(|| ids).clone())
}

/// ik's MTP draft set, opened once through its family. Like [`ik_set`], an Oracle clause's: the
/// fixture tier refuses the open by name.
///
/// # Errors
/// The fixture tier, or the family's own refusal.
pub fn mtp_set() -> Result<&'static MtpSet, GateError> {
    static SET: OnceLock<MtpSet> = OnceLock::new();
    if let Some(set) = SET.get() {
        return Ok(set);
    }
    if Tier::from_env()? == Tier::Fixture {
        return Err(format!(
            "the MTP draft set {MTP_SET} (family {}) was dumped from {MODEL}; BLOOMERY_TIER=fixture \
             runs on a fixture, which has no set of ik's: a clause that reads it is an Oracle \
             clause, deferred to the real tier (crates/gpu-gates/src/tier.rs)",
            MTP.name
        )
        .into());
    }
    let set = MtpSet::open(&MTP.path(MTP_SET), &MTP)?;
    Ok(SET.get_or_init(|| set))
}

/// The MTP draft set's prompt: in the real tier the set's own, held equal to the first
/// [`MTP_PROMPT`] ids of the prose corpus; in the fixture tier those corpus ids.
///
/// # Errors
/// A set that cannot be opened or has no prompt, a corpus that is short, or a set whose prompt is
/// not the corpus's.
pub fn mtp_prompt() -> Result<Vec<u32>, GateError> {
    let corpus = tier::prose_ids("glm5next", MTP_PROMPT)?;
    match Tier::from_env()? {
        Tier::Fixture => Ok(corpus),
        Tier::Real => {
            let dir = MTP.path(MTP_SET);
            let set = MtpSet::open(&dir, &MTP)?;
            let tokens = set
                .tokens
                .ok_or_else(|| format!("{}: no # tokens line", dir.display()))?;
            if tier::witness("MTP prompt = the first corpus ids", tokens == corpus, true) {
                Ok(tokens)
            } else {
                Err("the MTP set's prompt is not the prose corpus's first ids".into())
            }
        }
    }
}

/// The premise every flip-dependent clause of a residency gate shares: the history's flips land,
/// an expert is admitted, the card's copy moves off its start. It is the real file's routing skew
/// (file-bound): `true` when this tier asserts it, `false` when the fixture tier leaves it to the
/// real one, its line printed once. A fixture's generated router has no skew, so the clauses'
/// equalities hold the rest of each arm.
///
/// # Errors
/// A `BLOOMERY_TIER` that is neither tier.
pub fn skew() -> Result<bool, GateError> {
    static SKEW: OnceLock<bool> = OnceLock::new();
    if let Some(&s) = SKEW.get() {
        return Ok(s);
    }
    let s = tier::run_clause(
        "flips landed and an expert admitted (the file's routing skew)",
        tier::Tag::FileBound,
    )?;
    Ok(*SKEW.get_or_init(|| s))
}
