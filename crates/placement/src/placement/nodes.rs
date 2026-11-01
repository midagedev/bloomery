//! Several unified nodes over one fabric (`docs/contrib/dgx-spark.md` §9):
//! plan (b′) with each remote node in place of a tier card. Node 0 runs every
//! layer and the head; each other node holds a share of every routed layer's
//! ids in its own pool and computes them in place from the handoff, and the
//! link carries the handoff out and the rows back.
//!
//! A node is one unified machine with its own host and pool
//! ([`Machine::nodes`], this process's first). Each node's budget is the card
//! plan of that node alone ([`NodeBudget::of`]): every term its pool bound
//! counts ([`Plan::pool_need`]) but the routed ids, which this plan splits,
//! so the pool bound holds once a node ([`NodePlan::over`]) over the same
//! terms as the card plan's.
//!
//! The split counts ids, as if every id were used alike: the ids of a routed
//! layer go one at a time to the node whose window stays lowest after it
//! ([`NodePlan::owned`]), the same ids on every routed layer, and no node
//! pages an id while another's pool has room. The predictor
//! ([`NodePlan::predict`]) reads the per-token terms of node 0's card plan
//! ([`Terms`]), each node's read rate and NIC, and the fabric's round trip; a
//! token's slots land on the nodes by their shares, and a layer waits for its
//! slowest node. In a layer, node 0 sends the handoff only to the remotes
//! that hold one of the token's slots, and only those send rows back and
//! wait for the link. It reads no model file and runs no per-model code.
//!
//! Which nodes a plan uses is chosen ([`plan_nodes`]): a node is used where
//! the pools of the nodes before it cannot hold every id and its pool has
//! room; any other is used only where the plan with it exposes at most a
//! tenth of its step to the link and gains at least twice the machine's
//! ruler over the plan without it. A node whose own budget passes its pool is
//! never used. Every node the plan gives no id is named idle ([`NodeIdle`]).

use std::cmp::Ordering;
use std::ops::RangeInclusive;

use super::{
    Device, Format, HostTotals, Machine, ModelTensor, ModelTensors, Node, PlacementError, Plan,
    PlanLevers, PoolNeed, Role, RoutedRule, Row, Segment, card_bytes, per_expert, plan_routed,
    row_bytes,
};

/// Seed a routed plan and lower any explicit node ownership.
#[allow(
    clippy::too_many_arguments,
    reason = "the routed planner's inputs and the explicit node ownership (rust-quality R8)"
)]
pub fn loaded_routed<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx_max: u64,
    kv: &dyn super::KvBytes,
    levers: &PlanLevers,
    routed: RoutedRule,
    reserve: u64,
    owned: Option<&[u64]>,
) -> Result<Plan<'a>, PlacementError> {
    let stage = Machine {
        tiers: Vec::new(),
        nodes: machine.nodes.iter().take(1).cloned().collect(),
        ..machine.clone()
    };
    let seed = super::plan_routed_reserving(
        model,
        if owned.is_some() { &stage } else { machine },
        ctx_max,
        kv,
        levers,
        routed,
        reserve,
    )?;
    let mut plan = Plan {
        model,
        machine,
        ..seed
    };
    if let Some(owned) = owned {
        lower_routed(&mut plan, owned)?;
        plan.reserve_beside(reserve)?;
    }
    Ok(plan)
}

/// Lower a node assignment into the stage's loaded row map. Remote host
/// segments name their tier endpoint but allocate no stage-device buffers.
/// Each node's host/NVMe split is reconstructed against its own room at open.
pub fn lower_routed(plan: &mut Plan<'_>, owned: &[u64]) -> Result<(), PlacementError> {
    let bad = |detail: &str| PlacementError::Metadata {
        key: "node routed assignment".into(),
        detail: detail.into(),
    };
    if plan.machine.cards.len() != 1
        || owned.len() != plan.machine.tiers.len() + 1
        || owned.iter().try_fold(0u64, |s, &n| s.checked_add(n)) != Some(plan.model.experts)
        || owned[1..].contains(&0)
    {
        return Err(bad(
            "expected one stage and a nonempty partition for each loaded tier",
        ));
    }
    super::check_tiers(&plan.machine.tiers)?;
    super::check_devices(plan.machine, &plan.card_budget)?;
    let mut n_l = vec![0; plan.model.layers];
    let mut tier_n_l = vec![vec![0; plan.model.layers]; owned.len() - 1];
    let mut rows = plan.rows.clone();
    for row in &mut rows {
        let t = &plan.model.tensors[row.tensor];
        if t.role != Role::RoutedExperts {
            continue;
        }
        if plan.model.experts_of(t) != plan.model.experts {
            return Err(PlacementError::tensor(
                t,
                "a node assignment over unequal expert stacks",
            ));
        }
        let layer = t.layer.filter(|&l| l < n_l.len()).ok_or_else(|| {
            PlacementError::tensor(t, "a routed stack outside the node assignment")
        })?;
        let kept = plan.n_l[layer].min(owned[0]);
        n_l[layer] = kept;
        let (rows_per_expert, file_per_expert) = super::per_expert(t, plan.model.experts)?;
        let mut segments = Vec::with_capacity(owned.len() + 1);
        if kept != 0 {
            let format = row.segments.iter().find_map(|s| match s.format {
                Format::Card(format) => Some(format),
                _ => None,
            });
            segments.push(super::card_segment_as(
                t,
                0,
                kept * rows_per_expert,
                Some(super::ExpertList::range(0..kept)?),
                format,
            )?);
        }
        let mut first = owned[0];
        for (j, &n) in owned.iter().enumerate().skip(1) {
            segments.push(super::in_place(
                Device::Card(j),
                Format::HostFile,
                Some(super::ExpertList::range(first..first + n)?),
                0,
            ));
            tier_n_l[j - 1][layer] = n;
            first += n;
        }
        if kept < owned[0] {
            segments.push(super::in_place(
                Device::Host,
                Format::HostFile,
                Some(super::ExpertList::range(kept..owned[0])?),
                (owned[0] - kept) * file_per_expert,
            ));
        }
        row.segments = segments;
    }
    let mut kv: Vec<_> = plan
        .cards
        .iter()
        .map(|c| (c.kv_bytes, 0, c.stage_rows_bytes))
        .collect();
    kv.resize(
        plan.machine.cards.len() + plan.machine.tiers.len(),
        (0, 0, 0),
    );
    kv[0].1 = plan.host.shadow_bytes;
    let mut lowered = super::totals(
        plan.model,
        plan.machine,
        plan.ctx_max,
        rows,
        &kv,
        (n_l, tier_n_l),
        plan.card_budget.clone(),
    )?;
    lowered.host.arena_columns = plan.host.arena_columns;
    lowered.budget_from_pool = plan.budget_from_pool;
    *plan = lowered;
    Ok(())
}

/// The routed stacks assigned to `card`, as a host plan on `machine`'s own
/// node. Non-routed tensors stay with the stage. The existing NVMe split can
/// then divide this node's host rows without charging another node's pool.
pub fn routed_host<'a>(
    model: &'a ModelTensors,
    assigned: &[Row],
    card: usize,
    machine: &'a Machine,
    columns: u64,
) -> Result<Plan<'a>, PlacementError> {
    let mut bytes = 0u64;
    let mut counts = vec![0; model.layers];
    let mut ids_of = vec![None; model.layers];
    let mut rows: Vec<Row> = model
        .tensors
        .iter()
        .enumerate()
        .map(|(tensor, _)| Row {
            tensor,
            segments: vec![Segment {
                device: Device::Unused,
                format: Format::Unused,
                experts: None,
                resident_bytes: 0,
            }],
            read_bytes: 0,
            stage: 0,
        })
        .collect();
    for r in assigned {
        let t = model
            .tensors
            .get(r.tensor)
            .ok_or_else(|| PlacementError::Metadata {
                key: "routed host projection".into(),
                detail: format!("tensor {} is outside the model", r.tensor),
            })?;
        if t.role != Role::RoutedExperts {
            continue;
        }
        let Some(s) = r.segments.iter().find(|s| s.device == Device::Card(card)) else {
            continue;
        };
        let ids = s.experts.as_ref().ok_or_else(|| {
            PlacementError::tensor(t, "a routed card segment with no expert list")
        })?;
        let layer = t.layer.filter(|&l| l < model.layers).ok_or_else(|| {
            PlacementError::tensor(t, "a routed segment outside the model's layers")
        })?;
        if ids_of[layer]
            .replace(ids)
            .is_some_and(|previous| previous != ids)
        {
            return Err(PlacementError::tensor(
                t,
                "the layer's routed stacks hold different expert lists",
            ));
        }
        counts[layer] = ids.len();
        let (_, per) = per_expert(t, model.experts_of(t))?;
        let held = per
            .checked_mul(ids.len())
            .ok_or_else(|| PlacementError::tensor(t, "the routed host segment's bytes pass u64"))?;
        bytes = bytes
            .checked_add(held)
            .ok_or_else(|| PlacementError::tensor(t, "the routed host plan's bytes pass u64"))?;
        rows[r.tensor] = Row {
            tensor: r.tensor,
            segments: vec![Segment {
                device: Device::Host,
                format: Format::HostFile,
                experts: Some(ids.clone()),
                resident_bytes: held,
            }],
            read_bytes: 0,
            stage: 0,
        };
    }
    Ok(Plan {
        model,
        machine,
        ctx_max: 0,
        rows,
        cards: Vec::new(),
        host: HostTotals {
            expert_bytes: bytes,
            experts: counts.iter().sum(),
            nvme_expert_bytes: 0,
            nvme_arena_bytes: 0,
            arena_columns: columns,
            table_bytes: 0,
            shadow_bytes: 0,
            reserve_bytes: 0,
            row_reserve_bytes: 0,
            pool_card_bytes: 0,
            headroom_bytes: 0,
            room: None,
        },
        nvme_bytes: 0,
        n_l: vec![0; model.layers],
        tier_n_l: Vec::new(),
        card_budget: super::card_budget::CardBudget::Unset,
        budget_from_pool: false,
    })
}

/// The posts node 0's pump makes to reach a further remote node: each costs
/// this [assumed, `docs/contrib/dgx-spark.md` §9.8 criterion 6].
pub const POST_NS: u64 = 1_000;

/// Bytes a row the rows message carries a value: one f32.
const ROW_VALUE: u64 = 4;

/// Picoseconds a second: the predictor's sums run in picoseconds, so a
/// window of bytes over a rate keeps its sub-nanosecond part until the step.
const PS: u128 = 1_000_000_000_000;

/// A time in picoseconds, exact over a token's slot outcomes: `whole` and
/// `part` of `of` more, `part < of`, where `of` is `experts^k`. The times of
/// one model's plans share `of`, so they compare in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Ps {
    whole: u128,
    part: u128,
    of: u128,
}

impl Ps {
    /// `self` and `sum / of` more.
    fn plus(self, sum: u128) -> Option<Ps> {
        let part = self.part.checked_add(sum % self.of)?;
        Some(Ps {
            whole: self
                .whole
                .checked_add(sum / self.of)?
                .checked_add(part / self.of)?,
            part: part % self.of,
            of: self.of,
        })
    }

    /// `k` times `self`.
    pub(super) fn times(self, k: u128) -> Option<Ps> {
        let part = self.part.checked_mul(k)?;
        Some(Ps {
            whole: self.whole.checked_mul(k)?.checked_add(part / self.of)?,
            part: part % self.of,
            of: self.of,
        })
    }

    /// `self` less `other`, 0 where `other` is larger.
    fn less(self, other: Ps) -> Ps {
        if self <= other {
            return Ps {
                whole: 0,
                part: 0,
                ..self
            };
        }
        let (whole, part) = if self.part < other.part {
            (self.whole - 1, self.of - other.part + self.part)
        } else {
            (self.whole, self.part - other.part)
        };
        Ps {
            whole: whole - other.whole,
            part,
            of: self.of,
        }
    }

    /// `self` against `num / den`.
    fn cmp_ratio(self, num: u128, den: u128) -> Ordering {
        match self.whole.cmp(&(num / den)) {
            Ordering::Equal => cmp_ratio(self.part, self.of, num % den, den),
            whole => whole,
        }
    }

    /// Whole nanoseconds, rounded down.
    fn ns(self) -> Option<u64> {
        u64::try_from(self.whole / 1000).ok()
    }
}

/// What the link between two nodes costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinkCost {
    /// The exposed round trip a layer, link and protocol (λ): the handoff
    /// out and the rows back, the pumps' wake-ups and the stream waits.
    pub round_trip_ns: u64,
}

/// How the nodes are joined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fabric {
    /// One node: every machine today.
    Alone,
    /// Two nodes on one cable.
    Pair { link: LinkCost },
    /// Three nodes, each node's two ports to the other two.
    Ring { link: LinkCost },
    /// Up to four nodes through a switch, whose hop adds `hop_ns` to a round
    /// trip.
    Switch { link: LinkCost, hop_ns: u64 },
}

impl Fabric {
    /// The node counts the fabric joins.
    #[must_use]
    pub fn joins(self) -> RangeInclusive<usize> {
        match self {
            Fabric::Alone => 1..=1,
            Fabric::Pair { .. } => 2..=2,
            Fabric::Ring { .. } => 3..=3,
            Fabric::Switch { .. } => 2..=4,
        }
    }

    /// A remote node's round trip a layer: the link's, and the switch's hop;
    /// `None` past u64.
    fn round_trip_ns(self) -> Option<u64> {
        match self {
            Fabric::Alone => Some(0),
            Fabric::Pair { link } | Fabric::Ring { link } => Some(link.round_trip_ns),
            Fabric::Switch { link, hop_ns } => link.round_trip_ns.checked_add(hop_ns),
        }
    }
}

/// The nodes a plan may use, the fabric that joins them, the rates the
/// predictor reads of every node alike and the machine's ruler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nodes {
    /// Node 0 runs every layer and the head on its card.
    pub machine: Machine,
    pub fabric: Fabric,
    /// Bytes a second our kernels read a node's pool at (BW).
    pub read: u64,
    /// Bytes a second a direction through a node's NIC, the SoC side.
    pub nic: u64,
    /// The machine's 95 % interval on a difference of two arm means, in
    /// permille, 1 to 1000: a node is used only where it gains twice this.
    pub ruler_permille: u64,
}

/// What a node's pool holds beside the routed ids this plan gives it, and
/// the pool, from the card plan of that node alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeBudget {
    /// The node the plan was made on.
    node: Node,
    /// The plan's pool need ([`PoolNeed`]) less the routed ids it holds in
    /// the pool.
    fixed: u64,
    /// [`PoolNeed::pool`].
    pool: u64,
}

impl NodeBudget {
    /// The node whose card plan supplied this budget.
    #[must_use]
    pub fn node(&self) -> &Node {
        &self.node
    }

    /// The plan's pool need less the routed ids it holds in the pool.
    #[must_use]
    pub fn fixed(&self) -> u64 {
        self.fixed
    }

    /// The node's pool, as [`PoolNeed::pool`] reads it.
    #[must_use]
    pub fn pool(&self) -> u64 {
        self.pool
    }

    /// The budget of `model`'s routed stacks on one remote machine, with no
    /// cache or caller-given card budget. The remote keeps its own free reading
    /// and service reserves; no host-room term is read from the stage node.
    pub fn routed(
        model: &ModelTensors,
        machine: &Machine,
        routed: RoutedRule,
    ) -> Result<NodeBudget, NodesError> {
        let model = routed_only(model);
        let plan = plan_routed(
            &model,
            machine,
            0,
            &super::NoKv,
            &PlanLevers::default(),
            routed,
        )?;
        Self::of(&plan)
    }

    /// The budget `plan` gives its one node: node 0's from the plan of the
    /// model, a remote node's from the plan of [`routed_only`] with no
    /// cache. Every term of the plan's pool need stays in it (KV, the cards'
    /// set-aside and margins, the host's tables, reserves, shadows and arena)
    /// but the routed ids the pool holds, which [`plan_nodes`] splits at
    /// their file bytes. Refused by name: a plan of other than one node, a
    /// node with no unified pool, and a plan whose card lays the routed ids
    /// out in other bytes than the file's.
    pub fn of(plan: &Plan<'_>) -> Result<NodeBudget, NodesError> {
        let [node] = plan.machine.nodes.as_slice() else {
            return Err(NodesError::Undefined("a budget of other than one node"));
        };
        let Some(PoolNeed { cards, host, pool }) = plan.pool_need() else {
            return Err(NodesError::NotUnified(0));
        };
        let held = routed_held(plan)?;
        let overflow = || NodesError::Undefined("a plan whose bytes pass u64");
        let need = cards.checked_add(host).ok_or_else(overflow)?;
        Ok(NodeBudget {
            node: node.clone(),
            fixed: need.checked_sub(held).ok_or_else(overflow)?,
            pool,
        })
    }
}

/// The routed ids' bytes `plan` holds on its cards and the host, refused by
/// name where those and the NVMe tier's are not the file's: the node plans
/// split and read the ids at their file bytes.
fn routed_held(plan: &Plan<'_>) -> Result<u64, NodesError> {
    let card_ids = plan
        .cards
        .iter()
        .try_fold(0u64, |sum, c| sum.checked_add(c.expert_bytes));
    let held = card_ids.and_then(|c| c.checked_add(plan.host.expert_bytes));
    let file = plan
        .model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts)
        .try_fold(0u64, |sum, t| sum.checked_add(t.file_bytes));
    let (held, file) = held
        .zip(file)
        .ok_or(NodesError::Undefined("a plan whose bytes pass u64"))?;
    if held.checked_add(plan.host.nvme_expert_bytes) != Some(file) {
        return Err(NodesError::Undefined(
            "routed ids the card holds in other bytes than the file's",
        ));
    }
    Ok(held)
}

/// `model`'s routed stacks alone: what a remote node holds, whose budget is
/// the card plan of them ([`NodeBudget::of`]).
#[must_use]
pub fn routed_only(model: &ModelTensors) -> ModelTensors {
    ModelTensors {
        tensors: model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts)
            .cloned()
            .collect(),
        ..*model
    }
}

/// One routed layer's terms a token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerTerms {
    /// The shared expert and dense FFN bytes beside the routed stacks (Sh):
    /// read on node 0 in the layer's window.
    pub shared: u64,
    /// One expert's bytes over the layer's routed stacks (e).
    pub expert: u64,
}

/// What a decode token reads of a model, by role: node 0's card plan's own
/// bytes, so no model supplies a figure ([`Terms::of`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Terms {
    /// Attention, hyper-connections, norms, routers, engram weights, the
    /// head and the rows gathered from tables in the pool, and a layer's
    /// FFN with no routed stack: read on node 0 outside any window (A).
    pub serial: u64,
    /// File bytes filled from the NVMe tier on node 0 before those pool
    /// reads. Every gathered row is charged a fill, without cache hits.
    pub(super) serial_nvme: u64,
    /// Per routed layer, in layer order.
    pub layers: Vec<LayerTerms>,
    /// Experts in each routed stack, and experts one token uses.
    pub experts: u64,
    pub experts_used: u64,
    /// A row's values: the width the handoff and the rows carry.
    pub n_embd: u64,
}

impl Terms {
    /// The terms of `plan`, node 0's card plan, whose model's hidden rows
    /// are `n_embd` values (the model's embedding length). Each tensor counts
    /// the rows the plan reads of it a token ([`Row::read_bytes`]) in the
    /// bytes the plan holds them in: on a card in the card's format, on the
    /// host in the file's; a routed expert counts its file bytes, the bytes
    /// the node plans split it at. Refused by name: a plan of no unified
    /// pool, a routed stack that does not split into the model's experts,
    /// routed ids held in other bytes than the file's, a tensor in other than
    /// one segment, and sums past u64. Rows read from the NVMe tier count
    /// both their pool read and their serial fill.
    pub fn of(plan: &Plan<'_>, n_embd: u64) -> Result<Terms, NodesError> {
        if plan.machine.unified().is_none() {
            return Err(NodesError::NotUnified(0));
        }
        routed_held(plan)?;
        let model = plan.model;
        let past = |t| PlacementError::tensor(t, "the model's byte sums pass u64");
        let mut layers = vec![None::<LayerTerms>; model.layers];
        for t in model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts)
        {
            let Some(l) = t.layer.filter(|&l| l < model.layers) else {
                return Err(
                    PlacementError::tensor(t, "a routed stack past the model's layers").into(),
                );
            };
            let (_, bytes) = per_expert(t, model.experts)?;
            let layer = layers[l].get_or_insert(LayerTerms {
                shared: 0,
                expert: 0,
            });
            layer.expert = layer.expert.checked_add(bytes).ok_or_else(|| past(t))?;
        }
        let mut serial = 0u64;
        let mut serial_nvme = 0u64;
        for r in &plan.rows {
            let Some(t) = model.tensors.get(r.tensor) else {
                continue;
            };
            if t.role == Role::RoutedExperts {
                continue;
            }
            let routed = t
                .layer
                .and_then(|l| layers.get_mut(l))
                .and_then(Option::as_mut);
            let sum = match (t.role, routed) {
                (Role::SharedExpert | Role::DenseFfn, Some(layer)) => &mut layer.shared,
                _ => &mut serial,
            };
            *sum = sum.checked_add(pool_read(t, r)?).ok_or_else(|| past(t))?;
            if r.segments.iter().any(|s| s.device == Device::Nvme) {
                serial_nvme = serial_nvme
                    .checked_add(r.read_bytes)
                    .ok_or_else(|| past(t))?;
            }
        }
        Ok(Terms {
            serial,
            serial_nvme,
            layers: layers.into_iter().flatten().collect(),
            experts: model.experts,
            experts_used: model.experts_used,
            n_embd,
        })
    }

    /// One id's bytes over every routed layer.
    fn id_bytes(&self) -> Option<u64> {
        self.layers
            .iter()
            .try_fold(0u64, |sum, l| sum.checked_add(l.expert))
    }

    /// Node 0's shared bytes over the routed layers.
    fn shared_bytes(&self) -> Option<u64> {
        self.layers
            .iter()
            .try_fold(0u64, |sum, l| sum.checked_add(l.shared))
    }

    /// The bytes a token reads in the routed layers' windows on one node.
    fn window_bytes(&self) -> Option<u64> {
        self.layers.iter().try_fold(0u64, |sum, l| {
            self.experts_used
                .checked_mul(l.expert)?
                .checked_add(l.shared)?
                .checked_add(sum)
        })
    }
}

/// The bytes a token reads of `t` from a node's pool as `r` places it: the
/// rows the plan reads a token, in the card's format on a card and in the
/// file's on the host or after a fill from the NVMe tier; none of an unread
/// tensor. The fill is counted separately in [`Terms::serial_nvme`].
fn pool_read(t: &ModelTensor, r: &Row) -> Result<u64, PlacementError> {
    if r.read_bytes == 0 {
        return Ok(0);
    }
    let [s] = r.segments.as_slice() else {
        return Err(PlacementError::tensor(
            t,
            format!("a tensor in {} segments, not one", r.segments.len()),
        ));
    };
    let row = row_bytes(t)?;
    if !r.read_bytes.is_multiple_of(row) {
        return Err(PlacementError::tensor(
            t,
            format!(
                "{} B read a token are not whole rows of {row} B",
                r.read_bytes
            ),
        ));
    }
    match (s.device, s.format) {
        (Device::Card(_), Format::Card(format)) => card_bytes(t, format, r.read_bytes / row),
        (Device::Host, _) | (Device::Nvme, Format::NvmeFile) => Ok(r.read_bytes),
        (Device::Nvme, format) => Err(PlacementError::tensor(
            t,
            format!("an NVMe segment as {format}"),
        )),
        (Device::Unused, _) => Ok(0),
        (Device::Card(_), format) => Err(PlacementError::tensor(
            t,
            format!("a card segment as {format}"),
        )),
    }
}

/// The NVMe tier's misses on a paged id. They come from the arena's hit
/// rate and the drive, which the unified tier's own terms set, so the plan
/// takes them as given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fills {
    /// The share of a paged id's uses the arena serves, in permille.
    pub hit_permille: u64,
    /// One miss's fill.
    pub fill_ns: u64,
}

/// What the plan is asked to serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workload {
    /// One node's in-pool prompt rate at the prompt's length, tokens a second
    /// (S₁): a rate of a card and model pair, measured, so at least 1.
    pub prompt_rate: u64,
    /// Node 0's NVMe fill rate in file bytes a second. At least 1 when
    /// the plan reads rows from NVMe; 0 may mark an unused drive.
    pub nvme_read: u64,
    /// Every routed id held in a pool: a plan that would page one is refused.
    pub whole: bool,
    /// The paged routed ids' misses; `None` predicts their pool reads only.
    pub fills: Option<Fills>,
}

/// A node a plan leaves idle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeIdle {
    pub node: usize,
    pub why: IdleWhy,
}

/// Why a node is left idle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdleWhy {
    /// The node's own budget passes its pool ([`NodeBudget`]): it holds no
    /// id.
    Pool { fixed: u64, pool: u64 },
    /// With the node, the link's exposed term passes a tenth of the step.
    LinkShare { link_ns: u64, step_ns: u64 },
    /// With the node, the step gains less than twice the machine's ruler,
    /// or the split gives the node no id.
    Gain { without_ns: u64, with_ns: u64 },
}

/// A node's NIC busy past half of a decode step or a prompt batch: the
/// prompt mode to move ([`NodePlan::nic_bound`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NicBound {
    pub node: usize,
    pub busy_permille: u64,
}

/// Two remote nodes whose fills a token are more than a tenth apart
/// ([`NodePlan::fills_apart`]). Why: past every pool the split puts each
/// paged id where the windows, fills included, stay lowest, and does not
/// balance the fills themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FillsApart {
    pub low: usize,
    pub high: usize,
}

/// A node's planned bytes past its pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolOver {
    pub node: usize,
    pub need: u64,
    pub pool: u64,
}

/// Why nodes could not be planned.
#[derive(Debug, thiserror::Error)]
pub enum NodesError {
    #[error("{nodes} nodes on {fabric:?}, which joins {joins:?}")]
    Fabric {
        nodes: usize,
        fabric: Fabric,
        joins: RangeInclusive<usize>,
    },
    #[error("node {0} has no unified pool: a node is one unified machine")]
    NotUnified(usize),
    #[error("node {0}'s budget is not a card plan of that node")]
    Budget(usize),
    #[error("node {}: {} B planned pass its pool {} B by {} B", .0.node, .0.need, .0.pool, .0.need - .0.pool)]
    PoolOver(PoolOver),
    #[error("{0}")]
    Model(#[from] PlacementError),
    #[error("the nodes or the model have {0}: the plan is undefined")]
    Undefined(&'static str),
    #[error("the plan's sums overflow at {experts} experts and {used} a token")]
    Overflow { experts: u64, used: u64 },
}

/// A token's predicted decode on a plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prediction {
    pub step_ns: u64,
    /// What the link exposes of the step: the step less the step of the
    /// same nodes split with no remote term.
    pub link_ns: u64,
    /// Per node of [`Nodes::machine`].
    pub per_node: Vec<NodeLoad>,
}

/// One node's side of a prediction; all 0 on a node the plan leaves idle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeLoad {
    /// Bytes the plan holds in the node's pool.
    pub pool: u64,
    /// The node's mean window a token over the routed layers: its reads, its
    /// fills and, on a remote node, its link term in the layers the token's
    /// slots reach it.
    pub window_ns: u64,
    /// The NIC's busy share of a decode step or of a prompt batch, the
    /// larger, in permille, rounded down.
    pub nic_permille: u64,
    /// The messages the node posts and takes in the layer that reaches the
    /// most remotes: on node 0 a handoff out and rows back from each, at
    /// most `2 × min(k, remotes)`; on a remote node 2.
    pub msgs_per_layer: u64,
    /// NVMe fills a token, in thousandths.
    pub fills_milli: u64,
}

/// Several nodes' plan of one model.
#[derive(Clone, Debug)]
pub struct NodePlan<'a> {
    nodes: &'a Nodes,
    terms: &'a Terms,
    budgets: &'a [NodeBudget],
    workload: Workload,
    /// One id's bytes over every routed layer.
    id_bytes: u64,
    owned: Vec<u64>,
    idle: Vec<NodeIdle>,
    /// The plan with no remote term: what [`Prediction::link_ns`] is
    /// measured from.
    bare: bool,
}

/// The plan of `terms` on `nodes` for `workload`, each node within its
/// budget (`budgets[j]` node j's, [`NodeBudget::of`]): node 0 alone, then each
/// further node in order where the pools of the nodes before it cannot hold
/// every id and its own has room, or where it keeps the link within a tenth
/// of the step and gains twice the ruler. A node whose own budget passes its
/// pool is named idle for it, and every node the final split gives no id is
/// named idle. Refused by name: a node count the fabric does not join, a
/// node with no unified pool, a budget of another node, a rate, ruler, row
/// width, model or hit rate the plan is undefined at
/// ([`NodesError::Undefined`]), sums past their integers
/// ([`NodesError::Overflow`]), and a used node whose planned bytes pass its
/// pool ([`NodePlan::over`]).
pub fn plan_nodes<'a>(
    nodes: &'a Nodes,
    terms: &'a Terms,
    budgets: &'a [NodeBudget],
    workload: Workload,
) -> Result<NodePlan<'a>, NodesError> {
    let n = nodes.machine.nodes.len();
    let id_bytes = checked_id_bytes(nodes, terms, budgets, workload)?;
    let plan = |used: &[usize]| -> Result<NodePlan<'a>, NodesError> {
        let mut p = NodePlan {
            nodes,
            terms,
            budgets,
            workload,
            id_bytes,
            owned: vec![0; n],
            idle: Vec::new(),
            bare: false,
        };
        p.spread(used)?;
        Ok(p)
    };
    let mut used = vec![0];
    let mut why: Vec<Option<IdleWhy>> = vec![None; n];
    let mut now = plan(&used)?;
    for j in 1..n {
        let (fixed, pool) = (budgets[j].fixed, budgets[j].pool);
        if fixed > pool {
            why[j] = Some(IdleWhy::Pool { fixed, pool });
            continue;
        }
        let with: Vec<usize> = used.iter().copied().chain([j]).collect();
        let next = plan(&with)?;
        why[j] = idle_why(&now, &used, &next, j)?;
        if why[j].is_none() {
            (used, now) = (with, next);
        }
    }
    if let Some(&over) = now.over()?.first() {
        return Err(NodesError::PoolOver(over));
    }
    let empty: Vec<usize> = (1..n).filter(|&j| now.owned[j] == 0).collect();
    if empty.iter().any(|&j| why[j].is_none()) {
        let step_ns = now.predict()?.step_ns;
        for &j in &empty {
            why[j].get_or_insert(IdleWhy::Gain {
                without_ns: step_ns,
                with_ns: step_ns,
            });
        }
    }
    now.idle = empty
        .into_iter()
        .filter_map(|node| why[node].map(|why| NodeIdle { node, why }))
        .collect();
    Ok(now)
}

/// One id's bytes over every routed layer, once [`plan_nodes`]'s inputs are
/// defined: refused by name as it lists.
fn checked_id_bytes(
    nodes: &Nodes,
    terms: &Terms,
    budgets: &[NodeBudget],
    workload: Workload,
) -> Result<u64, NodesError> {
    let n = nodes.machine.nodes.len();
    let joins = nodes.fabric.joins();
    if !joins.contains(&n) {
        return Err(NodesError::Fabric {
            nodes: n,
            fabric: nodes.fabric,
            joins,
        });
    }
    if let Some(i) = nodes.machine.nodes.iter().position(|j| j.unified.is_none()) {
        return Err(NodesError::NotUnified(i));
    }
    if let Some(j) = (0..budgets.len().max(n))
        .find(|&j| budgets.get(j).map(NodeBudget::node) != nodes.machine.nodes.get(j))
    {
        return Err(NodesError::Budget(j));
    }
    let overflow = NodesError::Overflow {
        experts: terms.experts,
        used: terms.experts_used,
    };
    let undefined = if nodes.read == 0 {
        Some("a read rate of 0")
    } else if nodes.nic == 0 {
        Some("a NIC of 0")
    } else if !(1..=1000).contains(&nodes.ruler_permille) {
        Some("a ruler outside 1..=1000 permille")
    } else if workload.prompt_rate == 0 {
        Some("a prompt rate of 0")
    } else if terms.serial_nvme != 0 && workload.nvme_read == 0 {
        Some("an NVMe fill rate of 0 for rows read from the NVMe tier")
    } else if terms.n_embd == 0 {
        Some("a row width of 0")
    } else if terms.experts_used == 0 || terms.experts_used > terms.experts {
        Some("a token using none of the experts, or more than there are")
    } else if terms.id_bytes() == Some(0) {
        Some("routed experts of no bytes")
    } else if workload.fills.is_some_and(|f| f.hit_permille > 1000) {
        Some("a hit rate past 1000 permille")
    } else {
        None
    };
    if let Some(what) = undefined {
        return Err(NodesError::Undefined(what));
    }
    let (Some(id_bytes), Some(_), Some(_), Some(_)) = (
        terms.id_bytes(),
        terms.window_bytes(),
        terms.shared_bytes(),
        terms.experts.checked_mul(1000),
    ) else {
        return Err(overflow);
    };
    Ok(id_bytes)
}

/// Why the plan leaves node `j` idle, or `None` where it takes `next`, the
/// split over `used` and `j`, over `now`, the split over `used`: it takes it
/// where the pools of `used` cannot hold every id and `j`'s has room, and
/// otherwise where the link stays within a tenth of `next`'s step and `next`
/// gives `j` an id and gains twice the ruler.
fn idle_why(
    now: &NodePlan<'_>,
    used: &[usize],
    next: &NodePlan<'_>,
    j: usize,
) -> Result<Option<IdleWhy>, NodesError> {
    if now.room(used) < u128::from(now.terms.experts) && now.room_ids(j) > 0 {
        return Ok(None);
    }
    let (a, b) = (now.step_ps()?, next.step_ps()?);
    let link = next.exposed_ps(b)?;
    let ns = |ps: Ps| ps.ns().ok_or_else(|| now.overflow());
    let past = |x: Option<Ps>| x.ok_or_else(|| now.overflow());
    let ruler = 2 * u128::from(now.nodes.ruler_permille);
    Ok(if past(link.times(10))? > b {
        Some(IdleWhy::LinkShare {
            link_ns: ns(link)?,
            step_ns: ns(b)?,
        })
    } else if next.owned[j] == 0
        || b >= a
        || ruler >= 1000
        || past(a.times(1000 - ruler))? < past(b.times(1000))?
    {
        Some(IdleWhy::Gain {
            without_ns: ns(a)?,
            with_ns: ns(b)?,
        })
    } else {
        None
    })
}

/// A remote's link costs a layer: the round trip and a post in picoseconds,
/// and a slot's row in bytes, which crosses a node's NIC.
#[derive(Clone, Copy, Debug)]
struct LinkPs {
    /// The fabric's round trip.
    round_trip: u128,
    /// One post of node 0's pump.
    post: u128,
    /// One slot's row.
    row_bytes: u128,
}

impl<'a> NodePlan<'a> {
    #[must_use]
    pub fn nodes(&self) -> &'a Nodes {
        self.nodes
    }

    #[must_use]
    pub fn terms(&self) -> &'a Terms {
        self.terms
    }

    #[must_use]
    pub fn workload(&self) -> Workload {
        self.workload
    }

    /// Per node of [`Nodes::machine`]: the ids of every routed layer it
    /// owns, 0 on a node the plan leaves idle.
    #[must_use]
    pub fn owned(&self) -> &[u64] {
        &self.owned
    }

    #[must_use]
    pub fn idle(&self) -> &[NodeIdle] {
        &self.idle
    }

    /// The nodes the plan uses: those that own an id, node 0 always.
    fn used(&self) -> Vec<usize> {
        (0..self.owned.len())
            .filter(|&j| j == 0 || self.owned[j] > 0)
            .collect()
    }

    fn overflow(&self) -> NodesError {
        NodesError::Overflow {
            experts: self.terms.experts,
            used: self.terms.experts_used,
        }
    }

    /// The ids of every routed layer `node`'s pool holds beside its budget.
    fn room_ids(&self, node: usize) -> u64 {
        let b = &self.budgets[node];
        b.pool.saturating_sub(b.fixed) / self.id_bytes
    }

    /// The ids `nodes`' pools hold together.
    fn room(&self, nodes: &[usize]) -> u128 {
        nodes.iter().map(|&j| u128::from(self.room_ids(j))).sum()
    }

    /// The ids `node` owns but its pool does not hold: paged, unless the
    /// plan holds every id whole.
    fn paged(&self, node: usize, owned: u64) -> u64 {
        if self.workload.whole {
            0
        } else {
            owned.saturating_sub(self.room_ids(node))
        }
    }

    /// Bytes the plan holds in used `node`'s pool: its budget and the routed
    /// ids it holds.
    fn held_bytes(&self, node: usize) -> Result<u64, NodesError> {
        let held = self.owned[node] - self.paged(node, self.owned[node]);
        held.checked_mul(self.id_bytes)
            .and_then(|b| b.checked_add(self.budgets[node].fixed))
            .ok_or_else(|| self.overflow())
    }

    /// Each used node whose planned bytes ([`NodeLoad::pool`]) pass its
    /// pool.
    pub fn over(&self) -> Result<Vec<PoolOver>, NodesError> {
        let mut over = Vec::new();
        for node in self.used() {
            let (need, pool) = (self.held_bytes(node)?, self.budgets[node].pool);
            if need > pool {
                over.push(PoolOver { node, need, pool });
            }
        }
        Ok(over)
    }

    /// The ids every remote node owns.
    fn remote_ids(&self) -> u64 {
        self.owned[1..].iter().sum()
    }

    /// `experts^k` and `experts^(k − 1)`: the outcomes of a token's slots
    /// and of all but one.
    fn outcomes(&self) -> Option<(u128, u128)> {
        let t = self.terms;
        let k = u32::try_from(t.experts_used).ok()?;
        let e = u128::from(t.experts);
        Some((e.checked_pow(k)?, e.checked_pow(k - 1)?))
    }

    /// The outcomes of a token's slots, of `outcomes().0`, in which at least
    /// one lands on `node`'s ids.
    fn reached(&self, node: usize) -> Option<u128> {
        let k = u32::try_from(self.terms.experts_used).ok()?;
        let missed = u128::from(self.terms.experts - self.owned[node]).checked_pow(k)?;
        Some(self.outcomes()?.0 - missed)
    }

    /// The remote link's costs; all 0 on the plan with no remote term.
    fn link_ps(&self) -> Option<LinkPs> {
        if self.bare {
            return Some(LinkPs {
                round_trip: 0,
                post: 0,
                row_bytes: 0,
            });
        }
        Some(LinkPs {
            round_trip: u128::from(self.nodes.fabric.round_trip_ns()?).checked_mul(1000)?,
            post: u128::from(POST_NS) * 1000,
            row_bytes: u128::from(ROW_VALUE.checked_mul(self.terms.n_embd)?),
        })
    }

    /// Remote `node`'s mean link term a layer at the plan's split, in
    /// picoseconds, rounded down: in the layers a token's slots reach it,
    /// the round trip, a post for each other remote reached, and the rows
    /// of the other remotes' slots into node 0's NIC ahead of its own, each
    /// row's time rounded down.
    fn link_term_ps(&self, node: usize) -> Option<u128> {
        let t = self.terms;
        let (all, all_but_one) = self.outcomes()?;
        let k = u32::try_from(t.experts_used).ok()?;
        let e = u128::from(t.experts);
        let free = e - u128::from(self.owned[node]);
        let link = self.link_ps()?;
        let row = link.row_bytes.checked_mul(PS)? / u128::from(self.nodes.nic);
        let (mut both, mut ahead) = (0u128, 0u128);
        for i in (1..self.owned.len()).filter(|&i| i != node) {
            let o = u128::from(self.owned[i]);
            both = both.checked_add(
                self.reached(node)?
                    .checked_add((free - o).checked_pow(k)?)?
                    .checked_sub(all - self.reached(i)?)?,
            )?;
            ahead = ahead.checked_add(
                u128::from(t.experts_used)
                    .checked_mul(o)?
                    .checked_mul(all_but_one - free.checked_pow(k - 1)?)?,
            )?;
        }
        let sum = link
            .round_trip
            .checked_mul(self.reached(node)?)?
            .checked_add(link.post.checked_mul(both)?)?
            .checked_add(row.checked_mul(ahead)?)?;
        Some(sum / all)
    }

    /// A node's fill a slot on its ids, in picoseconds, as a fraction
    /// (numerator, denominator), where `paged` of the `owned` ids it owns
    /// page, both in one unit: the share of its slots that land on a paged id
    /// times the arena's misses and one fill.
    fn fill_ps(&self, paged: u128, owned: u128) -> Option<(u128, u128)> {
        let Some(f) = self.workload.fills else {
            return Some((0, 1));
        };
        if paged == 0 || self.workload.whole {
            return Some((0, 1));
        }
        Some((
            paged
                .checked_mul(u128::from(1000 - f.hit_permille))?
                .checked_mul(u128::from(f.fill_ns))?,
            owned,
        ))
    }

    /// `node`'s window a token over the routed layers, in picoseconds
    /// rounded down, at `milli` thousandths of an id owned: node 0's shared
    /// bytes and its slots' reads, a remote's link term at the plan's split
    /// ([`NodePlan::link_term_ps`]) and its slots' reads, and each node's
    /// fills.
    fn mean_ps(&self, node: usize, milli: u64) -> Option<u128> {
        let t = self.terms;
        let read = u128::from(self.nodes.read);
        let e1000 = u128::from(t.experts) * 1000;
        let slots = u128::from(t.experts_used).checked_mul(u128::from(milli))?;
        let layers = t.layers.len() as u128;
        let base = if node == 0 {
            u128::from(t.shared_bytes()?).checked_mul(PS)? / read
        } else {
            layers.checked_mul(self.link_term_ps(node)?)?
        };
        let reads = slots
            .checked_mul(u128::from(self.id_bytes))?
            .checked_mul(PS)?
            / e1000.checked_mul(read)?;
        let paged = milli.saturating_sub(self.room_ids(node).saturating_mul(1000));
        let (fill, per) = self.fill_ps(u128::from(paged), u128::from(milli))?;
        let fills = layers.checked_mul(slots)?.checked_mul(fill)? / per.checked_mul(e1000)?;
        base.checked_add(reads)?.checked_add(fills)
    }

    /// The split over `used`: every routed layer's ids, one at a time, to the
    /// node whose slowest window after it is lowest, then whose own window
    /// after it is, then the lower index; a node whose pool is full takes an
    /// id only when every node's is.
    fn spread(&mut self, used: &[usize]) -> Result<(), NodesError> {
        self.owned = vec![0; self.owned.len()];
        for _ in 0..self.terms.experts {
            let full: Vec<bool> = used
                .iter()
                .map(|&j| self.owned[j] >= self.room_ids(j))
                .collect();
            let open = full.contains(&false);
            let mut best: Option<(u128, u128, usize)> = None;
            for (&j, _) in used.iter().zip(&full).filter(|&(_, &f)| !open || !f) {
                self.owned[j] += 1;
                let (mut slowest, mut own) = (0, 0);
                for &i in used {
                    let w = self
                        .mean_ps(i, 1000 * self.owned[i])
                        .ok_or_else(|| self.overflow())?;
                    slowest = slowest.max(w);
                    if i == j {
                        own = w;
                    }
                }
                self.owned[j] -= 1;
                if best.is_none_or(|(w, o, _)| (slowest, own) < (w, o)) {
                    best = Some((slowest, own, j));
                }
            }
            if let Some((_, _, j)) = best {
                self.owned[j] += 1;
            }
        }
        Ok(())
    }

    /// The rule's optimum: the window every used node reads at a split by
    /// real shares, each remote's link term at the plan's split, each node's
    /// share up to its pool where the pools hold every id and from its pool
    /// where they do not, in picoseconds. A plan's slowest node
    /// ([`NodePlan::window_ps`]) is held to it.
    pub fn balanced_window_ps(&self) -> Result<u128, NodesError> {
        let used = self.used();
        let all = 1000 * self.terms.experts;
        let whole = self.room(&used) >= u128::from(self.terms.experts);
        let pool = |j: usize| all.min(self.room_ids(j).saturating_mul(1000));
        let bounds = |j: usize| if whole { (0, pool(j)) } else { (pool(j), all) };
        let mean = |j, milli| self.mean_ps(j, milli).ok_or_else(|| self.overflow());
        let share = |j: usize, level: u128| -> Result<u64, NodesError> {
            let (mut lo, mut hi) = bounds(j);
            if mean(j, lo)? > level {
                return Ok(lo);
            }
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                if mean(j, mid)? <= level {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            Ok(lo)
        };
        let (mut lo, mut hi) = (0, 0);
        for &j in &used {
            lo = lo.max(mean(j, bounds(j).0)?);
            hi = hi.max(mean(j, all)?);
        }
        while lo < hi {
            let mid = (lo + hi) / 2;
            let mut sum = 0u128;
            for &j in &used {
                sum += u128::from(share(j, mid)?);
            }
            if sum >= u128::from(all) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }

    /// The node's NIC busy terms, each a fraction (numerator, denominator):
    /// its time a decode token in picoseconds, and its share of a prompt
    /// batch, a token's mean bytes kept a fraction through both. A token's
    /// bytes at the plan's split are the rows of its slots on the remotes'
    /// ids: node 0 takes them, a remote sends the rows of its own. Each
    /// remote a token reaches returns a row for each of its slots, so at
    /// least one for its handoff, and the rows' direction is the busier. The
    /// prompt runs at S₁ times the one-node token's bytes over the split's: A
    /// and the slowest node's FFN, node 0's with the shared bytes. Both terms
    /// scale with the node's ids, which on node 0 are every remote's.
    fn nic_busy(&self, node: usize) -> Option<[(u128, u128); 2]> {
        let t = self.terms;
        let e = u128::from(t.experts);
        let ids = if node == 0 {
            self.remote_ids()
        } else {
            self.owned[node]
        };
        // A token's mean bytes times `experts`.
        let bytes = (t.layers.len() as u128)
            .checked_mul(u128::from(ROW_VALUE.checked_mul(t.n_embd)?))?
            .checked_mul(u128::from(t.experts_used))?
            .checked_mul(u128::from(ids))?;
        let nic = u128::from(self.nodes.nic);
        let decode = (bytes.checked_mul(PS)?, nic.checked_mul(e)?);
        let id_k = u128::from(self.id_bytes).checked_mul(u128::from(t.experts_used))?;
        let mut ffn = 0u128;
        for j in self.used() {
            let own = id_k.checked_mul(u128::from(self.owned[j]))?;
            let shared = if j == 0 {
                u128::from(t.shared_bytes()?).checked_mul(e)?
            } else {
                0
            };
            ffn = ffn.max(own.checked_add(shared)?);
        }
        let serial = u128::from(t.serial);
        let one = serial.checked_add(u128::from(t.window_bytes()?))?;
        let prompt = (
            u128::from(self.workload.prompt_rate)
                .checked_mul(one)?
                .checked_mul(bytes)?,
            serial.checked_mul(e)?.checked_add(ffn)?.checked_mul(nic)?,
        );
        Some([decode, prompt])
    }

    /// The node's NIC busy share of the decode step `step` or of a prompt
    /// batch, the larger, in permille, rounded down.
    fn nic_permille(&self, node: usize, step: Ps) -> Option<u64> {
        let [(time, per), (num, den)] = self.nic_busy(node)?;
        let busy = time.checked_mul(1000)?;
        let at = |whole: u128| Some(busy / per.checked_mul(whole)?);
        // The most p with p × step ≤ busy / per: the step lies in
        // [whole, whole + 1).
        let (mut lo, mut hi) = (at(step.whole.checked_add(1)?)?, at(step.whole)?);
        while lo < hi {
            let mid = hi - (hi - lo) / 2;
            if step.times(mid)?.cmp_ratio(busy, per) == Ordering::Greater {
                hi = mid - 1;
            } else {
                lo = mid;
            }
        }
        u64::try_from(lo.max(num.checked_mul(1000)? / den)).ok()
    }

    /// The node whose NIC is busy past half the decode step or the prompt
    /// batch, which is node 0 or none: node 0 takes the rows of every
    /// remote's slots, so each of its shares is at least any remote's
    /// ([`NodePlan::nic_busy`]). The plan's prompt mode is the one to move.
    pub fn nic_bound(&self) -> Result<Option<NicBound>, NodesError> {
        if self.used().len() < 2 {
            return Ok(None);
        }
        let step = self.step_ps()?;
        let [(time, per), (num, den)] = self.nic_busy(0).ok_or_else(|| self.overflow())?;
        let past = |x: Option<u128>| x.ok_or_else(|| self.overflow());
        if step.cmp_ratio(past(time.checked_mul(2))?, per) == Ordering::Less
            || past(num.checked_mul(2))? > den
        {
            let busy_permille = self.nic_permille(0, step).ok_or_else(|| self.overflow())?;
            return Ok(Some(NicBound {
                node: 0,
                busy_permille,
            }));
        }
        Ok(None)
    }

    /// `node`'s NVMe fills a token in thousandths, times `experts`: whole,
    /// so two nodes' fills compare exactly.
    fn fills_times_experts(&self, node: usize) -> Option<u128> {
        let Some(f) = self.workload.fills else {
            return Some(0);
        };
        let t = self.terms;
        (t.layers.len() as u128)
            .checked_mul(u128::from(t.experts_used))?
            .checked_mul(u128::from(self.paged(node, self.owned[node])))?
            .checked_mul(u128::from(1000 - f.hit_permille))
    }

    /// Two remote nodes whose fills a token are more than a tenth apart,
    /// the fewest's and the most's.
    pub fn fills_apart(&self) -> Result<Option<FillsApart>, NodesError> {
        let mut remotes: Vec<(usize, u128)> = Vec::new();
        for j in self.used().into_iter().skip(1) {
            remotes.push((
                j,
                self.fills_times_experts(j).ok_or_else(|| self.overflow())?,
            ));
        }
        let (Some(&(low, lo)), Some(&(high, hi))) = (
            remotes.iter().min_by_key(|&&(_, f)| f),
            remotes.iter().max_by_key(|&&(_, f)| f),
        ) else {
            return Ok(None);
        };
        let (Some(hi), Some(lo)) = (hi.checked_mul(10), lo.checked_mul(11)) else {
            return Err(self.overflow());
        };
        Ok((hi > lo).then_some(FillsApart { low, high }))
    }

    /// The expected max over the used nodes of a layer's windows, in
    /// picoseconds, times the outcomes of a token's slots (`outcomes().0`),
    /// so that the step divides once: each of the token's slots lands on a
    /// node with its owned share, independently of the others as §9.7 counts
    /// them, so the counts are multinomial, and every outcome is summed
    /// exactly. In an outcome, node 0 reads the shared bytes and its slots; a
    /// remote its slots reach waits for the round trip, a post for each other
    /// remote reached and the rows of the other remotes' slots ahead of its
    /// own; one they miss adds nothing. A window's reads, rows and fills share
    /// one denominator, so the window rounds down to whole picoseconds once.
    fn layer_ps(&self, used: &[usize], shared: u64, expert: u64) -> Option<u128> {
        let t = self.terms;
        let (read, nic) = (u128::from(self.nodes.read), u128::from(self.nodes.nic));
        let rate = read.checked_mul(nic)?;
        let link = self.link_ps()?;
        let (shared, expert) = (u128::from(shared), u128::from(expert));
        let mut fills: Vec<(u128, u128)> = Vec::with_capacity(used.len());
        for &j in used {
            let owned = self.owned[j];
            fills.push(self.fill_ps(u128::from(self.paged(j, owned)), u128::from(owned))?);
        }
        // A window that reads `bytes`, takes `rows` bytes through a NIC and
        // fills `slots` slots, in whole picoseconds.
        let window = |bytes: u128, rows: u128, slots: u128, (fill, per): (u128, u128)| {
            bytes
                .checked_mul(PS)?
                .checked_mul(nic)?
                .checked_add(rows.checked_mul(PS)?.checked_mul(read)?)?
                .checked_mul(per)?
                .checked_add(slots.checked_mul(fill)?.checked_mul(rate)?)?
                .checked_div(rate.checked_mul(per)?)
        };
        let owned: Vec<u128> = used.iter().map(|&j| u128::from(self.owned[j])).collect();
        let k = u32::try_from(t.experts_used).ok()?;
        let mut sum = 0u128;
        let mut counts = vec![0u32; used.len()];
        outcomes(&mut counts, 0, k, &mut |x| {
            let mut weight = multinomial(x)?;
            for (&n, &c) in owned.iter().zip(x) {
                weight = weight.checked_mul(n.checked_pow(c)?)?;
            }
            let reached = x[1..].iter().filter(|&&c| c > 0).count() as u128;
            let remote_slots: u128 = x[1..].iter().map(|&c| u128::from(c)).sum();
            let here = u128::from(x[0]);
            let mut slowest = window(
                shared.checked_add(here.checked_mul(expert)?)?,
                0,
                here,
                fills[0],
            )?;
            for (&fill, &c) in fills.iter().zip(x).skip(1).filter(|&(_, &c)| c > 0) {
                let c = u128::from(c);
                let rows = link.row_bytes.checked_mul(remote_slots - c)?;
                let w = link
                    .round_trip
                    .checked_add(link.post.checked_mul(reached - 1)?)?
                    .checked_add(window(c.checked_mul(expert)?, rows, c, fill)?)?;
                slowest = slowest.max(w);
            }
            sum = sum.checked_add(weight.checked_mul(slowest)?)?;
            Some(())
        })?;
        Some(sum)
    }

    /// The plan's decode step for one token: the serial bytes over the read
    /// rate, the row tables' serial fill over the NVMe read rate, rounded
    /// down to whole picoseconds, and each routed layer's expected window
    /// ([`NodePlan::layer_ps`]), exact over the slot outcomes.
    pub(super) fn step_ps(&self) -> Result<Ps, NodesError> {
        let t = self.terms;
        let used = self.used();
        let of = self.outcomes().ok_or_else(|| self.overflow())?.0;
        let serial = u128::from(t.serial)
            .checked_mul(PS)
            .ok_or_else(|| self.overflow())?
            / u128::from(self.nodes.read);
        let fill = if t.serial_nvme == 0 {
            0
        } else {
            u128::from(t.serial_nvme) * PS / u128::from(self.workload.nvme_read)
        };
        let mut step = Ps {
            whole: serial.checked_add(fill).ok_or_else(|| self.overflow())?,
            part: 0,
            of,
        };
        for l in &t.layers {
            step = self
                .layer_ps(&used, l.shared, l.expert)
                .and_then(|sum| step.plus(sum))
                .ok_or_else(|| self.overflow())?;
        }
        if step.whole == 0 {
            return Err(NodesError::Undefined("a step under 1 ps"));
        }
        Ok(step)
    }

    /// What the link exposes of the plan's step `step`: the step less the
    /// step of the same nodes split with no remote term.
    pub(super) fn exposed_ps(&self, step: Ps) -> Result<Ps, NodesError> {
        let used = self.used();
        if used.len() < 2 || self.bare {
            return Ok(Ps {
                whole: 0,
                part: 0,
                ..step
            });
        }
        let mut bare = self.clone();
        bare.bare = true;
        bare.spread(&used)?;
        Ok(step.less(bare.step_ps()?))
    }

    /// `node`'s window a token over the routed layers at the plan's split,
    /// in picoseconds ([`NodePlan::mean_ps`]).
    pub(super) fn window_ps(&self, node: usize) -> Result<u128, NodesError> {
        self.mean_ps(node, 1000 * self.owned[node])
            .ok_or_else(|| self.overflow())
    }

    /// The plan's decode prediction for one token.
    pub fn predict(&self) -> Result<Prediction, NodesError> {
        let t = self.terms;
        let used = self.used();
        let remotes = used.len() as u64 - 1;
        let step = self.step_ps()?;
        let ns = |ps: Ps| ps.ns().ok_or_else(|| self.overflow());
        let step_ns = ns(step)?;
        let link_ns = ns(self.exposed_ps(step)?)?;
        let mut per_node = vec![
            NodeLoad {
                pool: 0,
                window_ns: 0,
                nic_permille: 0,
                msgs_per_layer: 0,
                fills_milli: 0,
            };
            self.owned.len()
        ];
        for &j in &used {
            let fills = self
                .fills_times_experts(j)
                .map(|f| f / u128::from(t.experts));
            let nic_permille = if remotes == 0 {
                0
            } else {
                self.nic_permille(j, step).ok_or_else(|| self.overflow())?
            };
            per_node[j] = NodeLoad {
                pool: self.held_bytes(j)?,
                window_ns: u64::try_from(self.window_ps(j)? / 1000).map_err(|_| self.overflow())?,
                nic_permille,
                msgs_per_layer: match (remotes, j) {
                    (0, _) => 0,
                    (_, 0) => 2 * remotes.min(t.experts_used),
                    _ => 2,
                },
                fills_milli: fills
                    .and_then(|f| u64::try_from(f).ok())
                    .ok_or_else(|| self.overflow())?,
            };
        }
        Ok(Prediction {
            step_ns,
            link_ns,
            per_node,
        })
    }
}

/// `a / b` against `c / d`, by their continued fractions, so no product
/// passes u128.
fn cmp_ratio(mut a: u128, mut b: u128, mut c: u128, mut d: u128) -> Ordering {
    loop {
        let (p, q) = (a / b, c / d);
        if p != q {
            return p.cmp(&q);
        }
        (a, c) = (a % b, c % d);
        match (a, c) {
            (0, 0) => return Ordering::Equal,
            (0, _) => return Ordering::Less,
            (_, 0) => return Ordering::Greater,
            // a/b < c/d exactly where d/c < b/a.
            _ => (a, b, c, d) = (d, c, b, a),
        }
    }
}

/// Calls `f` with every way to put `left` slots on the nodes from `at` on,
/// the earlier nodes' counts already in `counts`; `None` from `f` stops it.
fn outcomes(
    counts: &mut [u32],
    at: usize,
    left: u32,
    f: &mut dyn FnMut(&[u32]) -> Option<()>,
) -> Option<()> {
    if at + 1 == counts.len() {
        counts[at] = left;
        return f(counts);
    }
    for c in 0..=left {
        counts[at] = c;
        outcomes(counts, at + 1, left - c, f)?;
    }
    Some(())
}

/// The ways `counts` of distinct slots fall on their nodes:
/// `(Σ counts)! / Π counts!`.
fn multinomial(counts: &[u32]) -> Option<u128> {
    let mut ways = 1u128;
    let mut n = 0u128;
    for &c in counts {
        for i in 1..=u128::from(c) {
            n += 1;
            ways = ways.checked_mul(n)? / i;
        }
    }
    Some(ways)
}
