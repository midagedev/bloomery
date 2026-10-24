//! `nodes_scale` (`docs/contrib/dgx-spark.md` §9.9): the node plans on
//! synthetic DGX Sparks, N = 1 to 4, against the expected cells that
//! `tools/flow/nodes_scale.py` derives from §9.7's model and §9.8's
//! criteria apart from the planner (`tests/nodes_scale_cells.tsv`). Three
//! synthetic shapes carry §9.7's model terms for V4.1, GLM-5.3-Flash and
//! Qwen3.8, A low and high; no file is read but where a clause reads the
//! Qwen3.8 file's tensor table.

use std::sync::OnceLock;

use gguf::GgmlType;

use super::nodes::{
    Fabric, Fills, FillsApart, IdleWhy, LinkCost, NicBound, NodeBudget, NodeIdle, NodePlan, Nodes,
    NodesError, POST_NS, PoolOver, Terms, Workload, plan_nodes, routed_only,
};
use super::workstation::{
    GB10_NIC, GB10_POOL, GB10_READ, ROW_PAGE, TierBatchBytes, gb10, plan_node_tiers, plan_on,
    spark_nodes,
};
use super::{
    CardFormat, Device, Format, KvBytes, Machine, ModelTensor, ModelTensors, PlacementError, Plan,
    PlanLevers, PoolNeed, Role, RoutedRule, RowTableSource, Violation, plan_routed, plan_with,
    row_table_tier,
};
use crate::accepts::{Program, routed_rule};

/// The head's rows.
const VOCAB: u64 = 151_552;

/// The switch's hop at N = 4 [assumed, §9.7].
const HOP_NS: u64 = 1_500;

/// λ's band [derived, §9.7].
const LAMBDAS: [u64; 2] = [25_000, 70_000];

/// One prompt rate a node, S₁ (tokens a second), at each end of §9.8's pins.
const PROMPT_RATES: [u64; 2] = [1_000, 3_000];

/// Node 0's cache bytes in every cell, over its layers.
const KV: u64 = 2_000_000_000;

/// The context every plan is made at.
const CTX: u64 = 4096;

/// Picoseconds a second.
const PS: u128 = 1_000_000_000_000;

/// One synthetic shape of §9.7's model terms table: the band ends of A and
/// Sh, and the bytes beside them node 0's pool holds (`embedding`, the token
/// embedding's file bytes). Every tensor is Q4_K rows of `n_embd` values, so
/// each term is rounded down to whole rows.
#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    layers: usize,
    n_embd: u64,
    experts: u64,
    used: u64,
    expert: u64,
    shared: [u64; 2],
    serial: [u64; 2],
    embedding: u64,
}

const V41: Shape = Shape {
    name: "V4.1",
    layers: 40,
    n_embd: 5120,
    experts: 384,
    used: 6,
    expert: 16_770_000,
    shared: [800_000_000, 1_500_000_000],
    serial: [3_200_000_000, 5_800_000_000],
    embedding: 4_500_000_000,
};

/// GLM's A and Sh are the file's header read, one value each, so its A low
/// and A high cases are the same plan.
const GLM: Shape = Shape {
    name: "GLM",
    layers: 42,
    n_embd: 4096,
    experts: 288,
    used: 8,
    expert: 15_200_000,
    shared: [1_600_000_000, 1_600_000_000],
    serial: [7_360_000_000, 7_360_000_000],
    embedding: 11_000_000_000,
};

const QWEN38: Shape = Shape {
    name: "Qwen3.8",
    layers: 48,
    n_embd: 2560,
    experts: 512,
    used: 10,
    expert: 3_170_000,
    shared: [140_000_000, 140_000_000],
    serial: [3_400_000_000, 4_400_000_000],
    embedding: 1_000_000_000,
};

const SHAPES: [Shape; 3] = [V41, GLM, QWEN38];

impl Shape {
    /// One Q4_K row's bytes.
    fn row(self) -> u64 {
        self.n_embd / 256 * 144
    }

    /// Rows a layer at A low (`high` false) or high: attention, the shared
    /// expert, and one expert.
    fn rows(self, high: bool) -> (u64, u64, u64) {
        let at = usize::from(high);
        let (row, layers) = (self.row(), self.layers as u64);
        (
            (self.serial[at] / row - VOCAB) / layers,
            self.shared[at] / row / layers,
            self.expert / row,
        )
    }

    /// The bytes a decode token reads of [`Shape::model`] on one node: A
    /// (attention, the head and one embedding row), and a layer's Sh and one
    /// expert.
    fn bytes(self, high: bool) -> (u64, u64, u64) {
        let (attn, shared, expert) = self.rows(high);
        let row = self.row();
        (
            (attn * self.layers as u64 + VOCAB + 1) * row,
            shared * row,
            expert * row,
        )
    }

    /// The shape as a model's tensors at A low (`high` false) or high: per
    /// layer one attention tensor and one shared expert, and one routed stack
    /// of `experts` experts of `expert` bytes; a head and a token embedding.
    /// A's attention part is A less the head.
    fn model(self, high: bool) -> ModelTensors {
        let row = self.row();
        let (attn, shared, expert) = self.rows(high);
        let tensor = |name: String, layer, role, rows: Vec<u64>| {
            let file_bytes = row * rows.iter().product::<u64>();
            ModelTensor {
                name,
                shard: 0,
                layer,
                role,
                ty: GgmlType::Q4_K,
                dims: [self.n_embd].into_iter().chain(rows).collect(),
                file_bytes,
                gathered_rows: (role == Role::TokenEmbedding).then_some(1),
            }
        };
        let mut tensors = vec![tensor(
            "embedding".into(),
            None,
            Role::TokenEmbedding,
            vec![self.embedding / row],
        )];
        for l in 0..self.layers {
            tensors.push(tensor(
                format!("attn {l}"),
                Some(l),
                Role::Attention,
                vec![attn],
            ));
            tensors.push(tensor(
                format!("shexp {l}"),
                Some(l),
                Role::SharedExpert,
                vec![shared],
            ));
            tensors.push(tensor(
                format!("experts {l}"),
                Some(l),
                Role::RoutedExperts,
                vec![expert, self.experts],
            ));
        }
        tensors.push(tensor("head".into(), None, Role::Head, vec![VOCAB]));
        ModelTensors {
            tensors,
            layers: self.layers,
            experts: self.experts,
            experts_used: self.used,
            bf16_words: |_| false,
        }
    }
}

/// §9.9's fabric for `n` nodes at round trip `lambda_ns`: one cable, a ring
/// of three, a switch for four.
fn fabric(n: usize, lambda_ns: u64) -> Fabric {
    let link = LinkCost {
        round_trip_ns: lambda_ns,
    };
    match n {
        1 => Fabric::Alone,
        2 => Fabric::Pair { link },
        3 => Fabric::Ring { link },
        _ => Fabric::Switch {
            link,
            hop_ns: HOP_NS,
        },
    }
}

fn workload(prompt_rate: u64) -> Workload {
    Workload {
        prompt_rate,
        nvme_read: 1_000_000_000,
        whole: false,
        fills: None,
    }
}

/// The same cache bytes on every layer.
struct Kv(u64);

impl KvBytes for Kv {
    fn layer_bytes(&self, _: usize, _: u64) -> u64 {
        self.0
    }
}

/// Today's card plan of `model` on `machine`, one node: node 0's with
/// [`KV`] over the layers, a remote's ([`routed_only`]) with none.
fn today<'a>(model: &'a ModelTensors, machine: &'a Machine, kv: bool) -> Plan<'a> {
    let kv = Kv(if kv { KV / model.layers as u64 } else { 0 });
    plan_with(model, machine, CTX, &kv, None).expect("today's plan")
}

/// A Spark of `free` bytes for `layers`.
fn spark(free: u64, layers: usize) -> Machine {
    plan_on(gb10(free), layers)
}

fn budget_on(model: &ModelTensors, machine: &Machine, node: usize) -> NodeBudget {
    if node == 0 {
        NodeBudget::of(&today(model, machine, true))
    } else {
        NodeBudget::routed(model, machine, RoutedRule::PerType(CardFormat::of_routed))
    }
    .expect("a budget")
}

/// `model`'s node budgets: node 0's on `machines[0]`, each other node's of
/// its routed stacks on its own machine; the nodes of `nodes` take the
/// machines' nodes.
fn budgets_on(model: &ModelTensors, nodes: &mut Nodes, machines: &[Machine]) -> Vec<NodeBudget> {
    let mut budgets = Vec::new();
    for (j, m) in machines.iter().enumerate() {
        nodes.machine.nodes[j] = m.nodes[0].clone();
        budgets.push(budget_on(model, m, j));
    }
    budgets
}

/// The terms of `model` on `machine` as node 0's plan reads them.
fn terms_of(model: &ModelTensors, machine: &Machine, n_embd: u64) -> Terms {
    Terms::of(&today(model, machine, true), n_embd).expect("the terms")
}

#[test]
fn census_budgets_keep_each_node_and_stage_terms() {
    let model = QWEN38.model(false);
    let mut specs = [
        gb10(80_000_000_000),
        gb10(60_000_000_000),
        gb10(90_000_000_000),
    ];
    for (j, spec) in specs.iter_mut().enumerate() {
        spec.device.as_mut().expect("a census identity").uuid = [j as u8; 16];
    }
    let machines = plan_node_tiers(
        model.layers,
        specs[0],
        &specs[1..],
        TierBatchBytes {
            staging: 123,
            scratch: 456,
            host: 789,
        },
    )
    .expect("three independent nodes");
    assert_eq!(machines.len(), specs.len(), "no remote omitted");
    let remote = routed_only(&model);
    let mut budgets = Vec::new();
    for (j, (machine, spec)) in machines.iter().zip(specs).enumerate() {
        assert_eq!(machine.nodes.len(), 1);
        assert!(machine.tiers.is_empty());
        assert_eq!(machine.cards[0].device, spec.device);
        assert_eq!(machine.cards[0].free_bytes, spec.free_bytes);
        assert_eq!(machine.unified(), super::workstation::unified_of(&spec));
        let oracle = if j == 0 {
            today(&model, machine, true)
        } else {
            today(&remote, machine, false)
        };
        let budget = budget_on(&model, machine, j);
        assert_eq!(budget, NodeBudget::of(&oracle).expect("the oracle"));
        assert_eq!(budget.pool(), oracle.pool_need().expect("one pool").pool);
        assert_eq!(budget.node(), &machine.nodes[0]);
        budgets.push(budget);
    }
    assert_ne!(budgets[0].pool(), budgets[1].pool());
    assert_ne!(budgets[1].pool(), budgets[2].pool());
    assert!(budgets[0].fixed() > budgets[1].fixed());
    let stage = today(&model, &machines[0], true);
    let terms = Terms::of(&stage, QWEN38.n_embd).expect("stage terms");
    assert_eq!(terms.n_embd, QWEN38.n_embd);
    assert_eq!(terms.layers.len(), QWEN38.layers);
    assert!(terms.serial > 0, "the full model supplies the serial bytes");
}

/// A shape's terms and its budgets: node 0's, a remote GB10's.
struct Case {
    shape: Shape,
    high: bool,
    terms: Terms,
    budgets: [NodeBudget; 2],
}

/// Every shape at A low and high, planned once.
fn cases() -> &'static [Case] {
    static CASES: OnceLock<Vec<Case>> = OnceLock::new();
    CASES.get_or_init(|| {
        let mut cases = Vec::new();
        for shape in SHAPES {
            let spec = gb10(GB10_POOL);
            let machines = plan_node_tiers(
                shape.layers,
                spec,
                &[spec],
                TierBatchBytes {
                    staging: 0,
                    scratch: 0,
                    host: 0,
                },
            )
            .expect("two nodes with no batch bytes");
            for high in [false, true] {
                let model = shape.model(high);
                let plan = today(&model, &machines[0], true);
                cases.push(Case {
                    shape,
                    high,
                    terms: Terms::of(&plan, shape.n_embd).expect("the terms"),
                    budgets: [
                        budget_on(&model, &machines[0], 0),
                        budget_on(&model, &machines[1], 1),
                    ],
                });
            }
        }
        cases
    })
}

impl Case {
    /// The budgets of `n` Sparks.
    fn budgets(&self, n: usize) -> Vec<NodeBudget> {
        (0..n).map(|j| self.budgets[j.min(1)].clone()).collect()
    }

    /// The plan on `nodes` for `workload`.
    fn plan<'a>(
        &'a self,
        nodes: &'a Nodes,
        budgets: &'a [NodeBudget],
        workload: Workload,
    ) -> Result<NodePlan<'a>, NodesError> {
        plan_nodes(nodes, &self.terms, budgets, workload)
    }
}

/// One cell of §9.8's table: a model, N, λ (µs) and A high.
type Cell = (&'static str, usize, u64, bool);

/// Every cell's plan at `prompt_rate` on nodes of `ruler_permille`, handed to
/// `check` with its cell, N the innermost loop.
fn each_cell(ruler_permille: u64, prompt_rate: u64, mut check: impl FnMut(Cell, &NodePlan<'_>)) {
    for case in cases() {
        for lambda in LAMBDAS {
            for n in 1..=4 {
                let mut nodes = spark_nodes(n, case.shape.layers, fabric(n, lambda));
                nodes.ruler_permille = ruler_permille;
                let budgets = case.budgets(n);
                let plan = case
                    .plan(&nodes, &budgets, workload(prompt_rate))
                    .unwrap_or_else(|e| panic!("{} N = {n}: {e}", case.shape.name));
                check((case.shape.name, n, lambda / 1000, case.high), &plan);
            }
        }
    }
}

/// The case of `cell`.
fn case_of(cell: Cell) -> &'static Case {
    cases()
        .iter()
        .find(|c| c.shape.name == cell.0 && c.high == cell.3)
        .expect("a case")
}

/// The box ruler.
const RULER: u64 = super::workstation::RULER;

/// A one-layer model of `experts` ids of `bytes` each, `used` a token,
/// beside a head of `4 × bytes`: F32 rows of one value.
fn tiny(experts: u64, used: u64, bytes: u64) -> ModelTensors {
    let tensor = |name: &str, layer, role, ty, dims: Vec<u64>, file_bytes| ModelTensor {
        name: name.into(),
        shard: 0,
        layer,
        role,
        ty,
        dims,
        file_bytes,
        gathered_rows: None,
    };
    ModelTensors {
        tensors: vec![
            tensor(
                "head",
                None,
                Role::Head,
                GgmlType::F32,
                vec![1, bytes],
                4 * bytes,
            ),
            tensor(
                "experts 0",
                Some(0),
                Role::RoutedExperts,
                GgmlType::I8,
                vec![1, bytes, experts],
                experts * bytes,
            ),
        ],
        layers: 1,
        experts,
        experts_used: used,
        bf16_words: |_| false,
    }
}

/// One-layer Sparks for `model`, of `id`-byte ids, whose budgets leave
/// room for `rooms` ids: node 0's beside its cache, each other's of the
/// routed stacks alone. Each pool is half an id over, since a smaller
/// pool's own budget is smaller.
fn with_room(model: &ModelTensors, id: u64, rooms: &[u64]) -> Vec<Machine> {
    let wide = spark(GB10_POOL, 1);
    let machines: Vec<Machine> = rooms
        .iter()
        .enumerate()
        .map(|(j, &room)| {
            let fixed = budget_on(model, &wide, j).fixed();
            spark(fixed + room * id + id / 2, 1)
        })
        .collect();
    for (j, (m, &room)) in machines.iter().zip(rooms).enumerate() {
        let b = budget_on(model, m, j);
        assert_eq!((b.pool() - b.fixed()) / id, room, "node {j}");
    }
    machines
}

/// The nodes that own an id, node 0 always.
fn used(plan: &NodePlan<'_>) -> u64 {
    plan.owned()
        .iter()
        .enumerate()
        .filter(|&(j, &o)| j == 0 || o > 0)
        .count() as u64
}

/// The inputs `tools/flow/nodes_scale.py` derives the expected cells from.
const INPUTS: &str = include_str!("../../tests/nodes_scale_inputs.tsv");

/// The expected cells (§9.9 "Expected sets"), as that script writes them.
const CELLS: &str = include_str!("../../tests/nodes_scale_cells.tsv");

/// The value of the inputs row `name`.
fn input(name: &str) -> u64 {
    INPUTS
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .find_map(|l| {
            let mut c = l.split('\t');
            (c.next() == Some(name)).then(|| c.next())
        })
        .flatten()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("nodes_scale_inputs.tsv: no whole number for {name}"))
}

/// A derived value whose band holds its criterion's line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Edge {
    /// The busiest NIC, on criterion 5's half.
    Nic,
    /// Node `node`'s link share in ppm, on criterion 2's tenth.
    Link { node: usize, ppm: u64, tol: u64 },
    /// Node `node`'s gain in ppm, on criterion 7's twice the ruler.
    Gain { node: usize, ppm: u64, tol: u64 },
}

impl Edge {
    fn criterion(self) -> &'static str {
        match self {
            Edge::Nic => "nic",
            Edge::Link { .. } => "link",
            Edge::Gain { .. } => "gain",
        }
    }
}

/// One derived cell: a cell at a ruler and S₁, what the script derives of
/// its plan, and each value's tolerance, the largest change one id moved
/// between two used nodes makes.
#[derive(Debug)]
struct Derived {
    cell: Cell,
    ruler: u64,
    rate: u64,
    used: u64,
    /// The idle nodes and why, `node:pool`, `node:link` or `node:gain`, or
    /// `-`.
    idle: String,
    paged: u64,
    /// Each node's paged ids, to one id.
    paged_nodes: Vec<u64>,
    step_ns: u64,
    step_tol: u64,
    nic: u64,
    nic_tol: u64,
    fires: bool,
    /// Node 0's messages in the layer that reaches the most remotes.
    msgs: u64,
    edges: Vec<Edge>,
}

/// Every derived cell.
fn derived() -> &'static [Derived] {
    static DERIVED: OnceLock<Vec<Derived>> = OnceLock::new();
    DERIVED.get_or_init(|| {
        let mut rows = CELLS.lines().filter(|l| !l.starts_with('#'));
        assert_eq!(
            rows.next(),
            Some(
                "model\ta\tn\tlambda_us\truler\ts1\tused\tidle\tpaged\tpaged_nodes\tstep_ns\tstep_tol\tnic\tnic_tol\tfires\tmsgs\tedge"
            ),
            "nodes_scale_cells.tsv's header"
        );
        rows.map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            let [model, a, n, lambda, ruler, rate, used, idle, paged, per, step, step_tol, nic, nic_tol, fires, msgs, edge] =
                c[..]
            else {
                panic!("nodes_scale_cells.tsv: {l:?} is not 17 cells");
            };
            let num = |v: &str| -> u64 {
                v.parse()
                    .unwrap_or_else(|_| panic!("nodes_scale_cells.tsv: {v:?} in {l:?}"))
            };
            let name = SHAPES
                .iter()
                .map(|s| s.name)
                .find(|&s| s == model)
                .unwrap_or_else(|| panic!("nodes_scale_cells.tsv: no shape {model}"));
            let edges = edge
                .split(',')
                .filter(|&e| e != "-")
                .map(|e| {
                    let f: Vec<&str> = e.split(':').collect();
                    match f[..] {
                        ["nic"] => Edge::Nic,
                        ["link", node, ppm, tol] => Edge::Link {
                            node: num(node) as usize,
                            ppm: num(ppm),
                            tol: num(tol),
                        },
                        ["gain", node, ppm, tol] => Edge::Gain {
                            node: num(node) as usize,
                            ppm: num(ppm),
                            tol: num(tol),
                        },
                        _ => panic!("nodes_scale_cells.tsv: edge {e:?} in {l:?}"),
                    }
                })
                .collect();
            Derived {
                cell: (name, num(n) as usize, num(lambda), a == "high"),
                ruler: num(ruler),
                rate: num(rate),
                used: num(used),
                idle: idle.to_string(),
                paged: num(paged),
                paged_nodes: per.split(',').map(num).collect(),
                step_ns: num(step),
                step_tol: num(step_tol),
                nic: num(nic),
                nic_tol: num(nic_tol),
                fires: fires == "1",
                msgs: num(msgs),
                edges,
            }
        })
        .collect()
    })
}

/// The derived cell of `cell` at `ruler` and S₁ `rate`.
fn derived_at(cell: Cell, ruler: u64, rate: u64) -> &'static Derived {
    derived()
        .iter()
        .find(|d| d.cell == cell && d.ruler == ruler && d.rate == rate)
        .unwrap_or_else(|| panic!("no derived cell {cell:?} at ruler {ruler}, S₁ {rate}"))
}

/// The side of each derived edge, pinned here: a cell, its ruler and S₁, the
/// edge's criterion, and whether that criterion fires there (criterion 2 or
/// 7 leaves the node idle, criterion 5 names the NIC).
const SIDES: [(Cell, u64, u64, &str, bool); 4] = [
    (("V4.1", 2, 25, false), RULER, 3_000, "nic", true),
    (("V4.1", 2, 25, false), 4 * RULER, 3_000, "nic", true),
    (("Qwen3.8", 3, 70, false), RULER, 1_000, "link", false),
    (("Qwen3.8", 3, 70, false), RULER, 3_000, "link", false),
];

/// Holds the derived side of `e` in `d` to its pin in [`SIDES`]; an edge
/// with no pin, or whose derived side is not its pin, panics by name.
fn side(d: &Derived, e: Edge) {
    let fires = match e {
        Edge::Nic => d.fires,
        Edge::Link { node, .. } => d.idle.split(',').any(|i| i == format!("{node}:link")),
        Edge::Gain { node, .. } => d.idle.split(',').any(|i| i == format!("{node}:gain")),
    };
    let (_, _, _, _, pin) = SIDES
        .iter()
        .find(|s| s.0 == d.cell && s.1 == d.ruler && s.2 == d.rate && s.3 == e.criterion())
        .unwrap_or_else(|| {
            panic!(
                "{:?} at ruler {}, S₁ {}: a {} edge with no pinned side",
                d.cell,
                d.ruler,
                d.rate,
                e.criterion()
            )
        });
    assert_eq!(
        fires,
        *pin,
        "{:?} at ruler {}, S₁ {}: the derived {} side is not the pinned one",
        d.cell,
        d.ruler,
        d.rate,
        e.criterion()
    );
}

/// Whether a link of `link_ns` can pass a tenth of a step of `step_ns`,
/// both rounded down to whole nanoseconds: the link is under `link_ns + 1`
/// and the step at least `step_ns`.
fn past_tenth(link_ns: u64, step_ns: u64) -> bool {
    10 * (link_ns + 1) > step_ns
}

/// The idle nodes of `plan` as the derived cells name them.
fn idle_of(plan: &NodePlan<'_>) -> String {
    let idle: Vec<String> = plan
        .idle()
        .iter()
        .map(|i| {
            let why = match i.why {
                IdleWhy::Pool { .. } => "pool",
                IdleWhy::LinkShare { .. } => "link",
                IdleWhy::Gain { .. } => "gain",
            };
            format!("{}:{why}", i.node)
        })
        .collect();
    if idle.is_empty() {
        "-".to_string()
    } else {
        idle.join(",")
    }
}

/// §9.9 "Expected sets": the cells are `tools/flow/nodes_scale.py`'s,
/// derived from `nodes_scale_inputs.tsv`, whose rows are this gate's own
/// machine, shapes and budgets. In every cell the plan uses the derived
/// nodes and leaves the others idle for the derived criterion, pages the
/// derived ids in all and on each node to one id, and its step and busiest
/// NIC are the derived ones to one id's change. Where a derived value's band
/// holds its line, the side is pinned in [`SIDES`], and a link share is
/// held to the band, read from the node's `NodeIdle` or, where the node is
/// the last used, from the plan; so is a gain, from a node's `NodeIdle`.
#[test]
fn expected_sets() {
    for (name, value) in [
        ("read", GB10_READ),
        ("nic", GB10_NIC),
        ("pool", case_of(("V4.1", 1, 25, false)).budgets[0].pool()),
        ("lambda_lo", LAMBDAS[0]),
        ("lambda_hi", LAMBDAS[1]),
        ("hop", HOP_NS),
        ("post", POST_NS),
        ("row_value", 4),
        ("vocab", VOCAB),
        ("ruler", RULER),
        ("ruler_wide", 4 * RULER),
        ("s1_lo", PROMPT_RATES[0]),
        ("s1_hi", PROMPT_RATES[1]),
    ] {
        assert_eq!(input(name), value, "{name}");
    }
    for case in cases() {
        let s = case.shape;
        let end = if case.high { "hi" } else { "lo" };
        let at = usize::from(case.high);
        for (term, value) in [
            ("layers".to_string(), s.layers as u64),
            ("n_embd".to_string(), s.n_embd),
            ("experts".to_string(), s.experts),
            ("top_k".to_string(), s.used),
            ("expert".to_string(), s.expert),
            (format!("shared_{end}"), s.shared[at]),
            (format!("serial_{end}"), s.serial[at]),
            (format!("fixed0_{end}"), case.budgets[0].fixed()),
            ("fixed1".to_string(), case.budgets[1].fixed()),
        ] {
            assert_eq!(
                input(&format!("{}/{term}", s.name)),
                value,
                "{}/{term}",
                s.name
            );
        }
        assert_eq!(case.budgets[1].pool(), case.budgets[0].pool());
    }
    let mut seen = 0;
    let mut pinned = Vec::new();
    for (ruler, rate) in [
        (RULER, PROMPT_RATES[0]),
        (RULER, PROMPT_RATES[1]),
        (4 * RULER, PROMPT_RATES[1]),
    ] {
        each_cell(ruler, rate, |cell, plan| {
            seen += 1;
            let d = derived_at(cell, ruler, rate);
            let at = format!("{cell:?} at ruler {ruler}, S₁ {rate}");
            assert_eq!(
                (used(plan), idle_of(plan)),
                (d.used, d.idle.clone()),
                "{at}: the nodes used and idle"
            );
            let p = plan.predict().expect("a prediction");
            let id: u64 = plan.terms().layers.iter().map(|l| l.expert).sum();
            let paged: Vec<u64> = plan
                .owned()
                .iter()
                .enumerate()
                .map(|(j, &o)| {
                    let held = if o == 0 {
                        0
                    } else {
                        let fixed = case_of(cell).budgets[j.min(1)].fixed();
                        p.per_node[j]
                            .pool
                            .checked_sub(fixed)
                            .expect("a used node's pool holds its budget")
                            / id
                    };
                    o.checked_sub(held).expect("a node holds at most its ids")
                })
                .collect();
            assert!(
                paged.iter().sum::<u64>() == d.paged
                    && paged.len() == d.paged_nodes.len()
                    && paged
                        .iter()
                        .zip(&d.paged_nodes)
                        .all(|(a, b)| a.abs_diff(*b) <= 1),
                "{at}: paged {paged:?}, derived {} as {:?} to one id each",
                d.paged,
                d.paged_nodes
            );
            assert!(
                p.step_ns.abs_diff(d.step_ns) <= d.step_tol,
                "{at}: step {} ns, derived {} ± {} ns",
                p.step_ns,
                d.step_ns,
                d.step_tol
            );
            let busiest = p.per_node.iter().map(|l| l.nic_permille).max().unwrap_or(0);
            assert!(
                busiest.abs_diff(d.nic) <= d.nic_tol,
                "{at}: NIC {busiest} ‰, derived {} ± {} ‰",
                d.nic,
                d.nic_tol
            );
            for &e in &d.edges {
                side(d, e);
                pinned.push((cell, ruler, rate, e.criterion()));
                let (node, want, tol) = match e {
                    Edge::Nic => continue,
                    Edge::Link { node, ppm, tol } | Edge::Gain { node, ppm, tol } => {
                        (node, ppm, tol)
                    }
                };
                let why = plan.idle().iter().find(|i| i.node == node).map(|i| i.why);
                let last = used(plan) == node as u64 + 1;
                let ppm = match (e, why) {
                    (Edge::Link { .. }, Some(IdleWhy::LinkShare { link_ns, step_ns })) => {
                        assert!(past_tenth(link_ns, step_ns), "{at}: {why:?}");
                        i128::from(link_ns) * 1_000_000 / i128::from(step_ns)
                    }
                    (Edge::Link { .. }, None) if last => {
                        i128::from(p.link_ns) * 1_000_000 / i128::from(p.step_ns)
                    }
                    (
                        Edge::Gain { .. },
                        Some(IdleWhy::Gain {
                            without_ns,
                            with_ns,
                        }),
                    ) => {
                        (i128::from(without_ns) - i128::from(with_ns)) * 1_000_000
                            / i128::from(without_ns)
                    }
                    (Edge::Gain { .. }, None) => continue,
                    _ => panic!("{at}: no witness of {e:?} on node {node}, idle {why:?}"),
                };
                assert!(
                    ppm.abs_diff(i128::from(want)) <= u128::from(tol),
                    "{at}: node {node}'s {} {ppm} ppm, derived {want} ± {tol}",
                    e.criterion()
                );
            }
        });
    }
    assert_eq!(seen, derived().len());
    let mut sides: Vec<_> = SIDES.iter().map(|s| (s.0, s.1, s.2, s.3)).collect();
    pinned.sort();
    sides.sort();
    assert_eq!(pinned, sides, "a pinned side with no derived edge");
}

/// Clause 1, `n1_is_today`: one Spark's machine is [`plan_on`] the GB10,
/// and its plan is today's: held whole, node 0's bytes are today's pool sum
/// ([`PoolNeed`]), or both refuse it by the same bytes; paged, node 0 holds
/// today's terms and every id its pool has room for. No link term, no
/// message, no NIC, no idle node, and the step and node 0's window the bytes
/// the shape lays out over the pool's read rate. The terms read the plan:
/// a BF16 head and shared expert in the card's f32, an engram table on the
/// host in its gathered rows, one on the NVMe tier refused by name, and the Qwen3.8
/// file's row width as given, whatever its head tensors' order (its model
/// tensors and first two layers, planned by its program's routed rule). A host
/// reserve on node 0 raises its bytes by itself. A table read from NVMe adds
/// its fill to the step once: twice the read rate takes half the fill off, on
/// one Spark and on two, whose split and node windows stay.
#[test]
fn n1_is_today() {
    for case in cases() {
        let (shape, at) = (case.shape, (case.shape.name, case.high));
        let nodes = spark_nodes(1, shape.layers, Fabric::Alone);
        assert_eq!(nodes.machine, spark(GB10_POOL, shape.layers));
        let model = shape.model(case.high);
        let today = today(&model, &nodes.machine, true);
        let Some(PoolNeed { cards, host, pool }) = today.pool_need() else {
            panic!("{at:?}: a GB10 has a pool");
        };
        let need = cards + host;
        let broken = today.violations();
        assert!(
            broken
                .iter()
                .all(|v| matches!(v, Violation::PoolOver { .. } | Violation::HostOver { .. })),
            "{at:?}: {broken:?}"
        );
        let budgets = case.budgets(1);
        let whole = Workload {
            whole: true,
            ..workload(3_000)
        };
        let (serial, shared, expert) = shape.bytes(case.high);
        let id = expert * shape.layers as u64;
        match case.plan(&nodes, &budgets, whole) {
            Ok(plan) => {
                assert!(broken.is_empty(), "{at:?}: {broken:?}");
                assert_eq!(plan.predict().expect("a prediction").per_node[0].pool, need);
            }
            Err(NodesError::PoolOver(over)) => {
                assert!(broken.contains(&Violation::PoolOver {
                    cards,
                    host,
                    beside: 0,
                    pool
                }));
                assert_eq!(
                    over,
                    PoolOver {
                        node: 0,
                        need,
                        pool
                    },
                    "{at:?}"
                );
            }
            Err(e) => panic!("{at:?}: {e}"),
        }
        let plan = case
            .plan(&nodes, &budgets, workload(3_000))
            .expect("one Spark plans");
        let p = plan.predict().expect("a prediction");
        let held = p.per_node[0].pool;
        assert!(
            held <= pool && (need - held).is_multiple_of(id) && (need == held || pool - held < id),
            "{at:?}: {held} B held, today {need} B in {pool} B"
        );
        let window = (shared + shape.used * expert) * shape.layers as u64;
        let ns = |bytes: u64| u128::from(bytes) * 1_000_000_000 / u128::from(GB10_READ);
        assert_eq!(plan.owned(), [shape.experts], "{at:?}");
        assert!(plan.idle().is_empty(), "{at:?}: {:?}", plan.idle());
        assert_eq!(
            (
                p.link_ns,
                p.per_node[0].msgs_per_layer,
                p.per_node[0].nic_permille
            ),
            (0, 0, 0),
            "{at:?}"
        );
        assert!(
            u128::from(p.step_ns).abs_diff(ns(serial + window)) <= 1
                && u128::from(p.per_node[0].window_ns).abs_diff(ns(window)) <= 1,
            "{at:?}: step {} ns and window {} ns, the shape's {} and {} ns",
            p.step_ns,
            p.per_node[0].window_ns,
            ns(serial + window),
            ns(window)
        );
    }
    let step_of = |plan: &Plan<'_>, n_embd, nvme_read| {
        let nodes = spark_nodes(1, plan.model.layers, Fabric::Alone);
        let terms = Terms::of(plan, n_embd).expect("the terms");
        let budgets = [NodeBudget::of(plan).expect("a budget")];
        let w = Workload {
            nvme_read,
            ..workload(3_000)
        };
        let step = plan_nodes(&nodes, &terms, &budgets, w)
            .and_then(|p| p.predict())
            .expect("one Spark")
            .step_ns;
        (terms, step)
    };
    let (shape, n) = (V41, V41.n_embd);
    let row = shape.row();
    let (attn, sh, _) = shape.rows(false);
    let mut model = shape.model(false);
    for t in model
        .tensors
        .iter_mut()
        .filter(|t| t.name == "head" || t.name == "shexp 0")
    {
        t.ty = GgmlType::BF16;
        t.file_bytes = t.dims.iter().product::<u64>() * 2;
    }
    model.tensors.push(ModelTensor {
        name: "engram 0".into(),
        shard: 0,
        layer: Some(0),
        role: Role::EngramTable,
        ty: GgmlType::Q4_K,
        dims: vec![n, 1000],
        file_bytes: 1000 * row,
        gathered_rows: Some(4),
    });
    let machine = spark(GB10_POOL, shape.layers);
    let mut plan = today(&model, &machine, true);
    assert_eq!(plan.row_tier().expect("one tier"), Some(Device::Nvme));
    let (nvme, filled_step) = step_of(&plan, n, 1_000_000_000);
    assert_eq!(nvme.serial_nvme, 4 * row);
    let (_, faster_step) = step_of(&plan, n, 2_000_000_000);
    assert_eq!(filled_step - faster_step, 2 * row);
    let two_sparks = |nvme_read| {
        let mut nodes = spark_nodes(2, shape.layers, fabric(2, LAMBDAS[0]));
        let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
        let w = Workload {
            nvme_read,
            ..workload(3_000)
        };
        let plan = plan_nodes(&nodes, &nvme, &budgets, w).expect("two Sparks");
        let p = plan.predict().expect("a prediction");
        let windows: Vec<u64> = p.per_node.iter().map(|l| l.window_ns).collect();
        (plan.owned().to_vec(), windows, p.step_ns)
    };
    let (owned, windows, filled_pair) = two_sparks(1_000_000_000);
    assert!(
        owned.iter().all(|&o| o > 0),
        "both Sparks hold ids: {owned:?}"
    );
    let (faster_owned, faster_windows, faster_pair) = two_sparks(2_000_000_000);
    assert_eq!((faster_owned, faster_windows), (owned, windows));
    assert_eq!(filled_pair - faster_pair, 2 * row);
    row_table_tier(&mut plan, u64::MAX, 0, 1, ROW_PAGE, RowTableSource::Split)
        .expect("the table's tier");
    assert_eq!(plan.row_tier().expect("one tier"), Some(Device::Host));
    let (t, step) = step_of(&plan, n, 0);
    let serial = (attn * shape.layers as u64 + 1) * row + VOCAB * n * 4;
    let shared = sh * row * (shape.layers as u64 - 1) + sh * n * 4;
    let shared_of = |t: &Terms| t.layers.iter().map(|l| l.shared).sum::<u64>();
    assert_eq!((t.serial, shared_of(&t)), (serial + 4 * row, shared));
    assert_eq!((nvme.serial, shared_of(&nvme)), (t.serial, shared));
    assert_eq!(t.serial_nvme, 0);
    assert_eq!(filled_step - step, 4 * row);
    let bytes = t.serial + shared + shape.used * shape.layers as u64 * t.layers[0].expert;
    let ns = u128::from(bytes) * 1_000_000_000 / u128::from(GB10_READ);
    assert!(
        u128::from(step).abs_diff(ns) <= 1,
        "step {step} ns, {ns} ns"
    );
    let file = super::qwen38_cards::model(false);
    let file = ModelTensors {
        tensors: file
            .tensors
            .into_iter()
            .filter(|t| t.layer.is_none_or(|l| l < 2))
            .collect(),
        layers: 2,
        ..file
    };
    let machine = spark(GB10_POOL, file.layers);
    let kv = Kv(KV / file.layers as u64);
    let rule = routed_rule(Program::Qwen38Body);
    let mut plan = plan_routed(&file, &machine, CTX, &kv, &PlanLevers::default(), rule)
        .expect("the file's plan");
    let (i, ple) = file
        .tensors
        .iter()
        .enumerate()
        .find(|(_, t)| t.role == Role::EngramTable)
        .expect("the PLE table");
    let on_nvme = Terms::of(&plan, 2560).expect("the file's NVMe terms");
    let mut unread = plan.clone();
    let table = unread
        .rows
        .iter()
        .position(|r| r.tensor == i)
        .expect("the table");
    unread.rows[table].segments[0].format = Format::HostFile;
    assert!(matches!(
        Terms::of(&unread, 2560),
        Err(NodesError::Model(PlacementError::Tensor { .. }))
    ));
    unread.rows[table].segments[0].format = Format::NvmeFile;
    unread.rows[table].read_bytes = 0;
    let unread = Terms::of(&unread, 2560).expect("the file's unread table");
    row_table_tier(&mut plan, u64::MAX, 0, 1, ROW_PAGE, RowTableSource::Split)
        .expect("the PLE table's tier");
    let on_host = Terms::of(&plan, 2560).expect("the file's host terms");
    let gathered = ple.gathered_rows.expect("its rows") * ple.file_bytes
        / ple.dims[1..].iter().product::<u64>();
    assert_eq!(
        (on_host.n_embd, on_host.serial - unread.serial),
        (2560, gathered)
    );
    assert_eq!(on_nvme.serial, on_host.serial);
    assert_eq!(on_nvme.serial_nvme, gathered);
    assert_eq!((on_host.serial_nvme, unread.serial_nvme), (0, 0));
    let model = QWEN38.model(false);
    let reserve = 4 << 30;
    let mut machine = spark(GB10_POOL, QWEN38.layers);
    let base = NodeBudget::of(&today(&model, &machine, true)).expect("a budget");
    machine
        .host_mut()
        .reserves
        .push(("checkpoints".to_string(), reserve));
    let reserved = NodeBudget::of(&today(&model, &machine, true)).expect("a budget");
    let terms = &case_of(("Qwen3.8", 1, 25, false)).terms;
    let held = |machine: Machine, budget| {
        let nodes = Nodes {
            machine,
            ..spark_nodes(1, QWEN38.layers, Fabric::Alone)
        };
        let budgets = [budget];
        let plan = plan_nodes(&nodes, terms, &budgets, workload(3_000)).expect("one Spark");
        plan.predict().expect("a prediction").per_node[0].pool
    };
    assert_eq!(
        held(machine, reserved),
        held(spark(GB10_POOL, QWEN38.layers), base) + reserve
    );
}

/// Clause 2, `pool_per_node`: each used node's planned bytes are held to
/// its own pool, a remote's beside its own card's set-aside and margin, and
/// a node the plan leaves idle holds none. V4.1 whole is refused at N = 2
/// by a node's pool and placed at N = 4; no cell of §9.8 is refused
/// (`each_cell` panics on a refusal). A node whose own budget passes its
/// pool is named idle for it, not refused, whether node 0 holds the model or
/// pages; a node is used while the pools before it page, whatever it gains,
/// unless its own pool has no room; and a node the final split gives no id is
/// named idle, also one the loop took while node 0 paged and a later node's
/// room emptied.
#[test]
fn pool_per_node() {
    each_cell(RULER, 3_000, |cell, plan| {
        let over = plan.over().expect("a plan's sums");
        assert!(over.is_empty(), "{cell:?}: {over:?}");
        let id: u64 = plan.terms().layers.iter().map(|l| l.expert).sum();
        let p = plan.predict().expect("a prediction");
        let card = &plan.nodes().machine.cards[0];
        let remote = &case_of(cell).budgets[1];
        assert!(
            remote.fixed()
                >= card.set_aside_bytes().expect("bounded set-aside") + card.margin_bytes
        );
        for (j, &owned) in plan.owned().iter().enumerate().skip(1) {
            let ids = p.per_node[j].pool.checked_sub(remote.fixed());
            assert!(
                (owned == 0 && p.per_node[j].pool == 0)
                    || ids.is_some_and(|b| b > 0 && b % id == 0),
                "{cell:?}: node {j}"
            );
        }
    });
    let whole = Workload {
        whole: true,
        ..workload(3_000)
    };
    let case = &cases()[0];
    let at = |n| {
        let nodes = spark_nodes(n, V41.layers, fabric(n, LAMBDAS[0]));
        let budgets = case.budgets(n);
        case.plan(&nodes, &budgets, whole)
            .map(|p| p.owned().to_vec())
    };
    assert!(
        matches!(at(2), Err(NodesError::PoolOver(_))),
        "V4.1 whole at N = 2: {:?}",
        at(2)
    );
    let owned = at(4).expect("V4.1 whole at N = 4");
    assert_eq!(owned.iter().sum::<u64>(), V41.experts);

    let model = QWEN38.model(false);
    let mut pinned = spark(GB10_POOL, QWEN38.layers);
    pinned
        .host_mut()
        .reserves
        .push(("pinned".to_string(), 120_000_000_000));
    let mut nodes = spark_nodes(2, QWEN38.layers, fabric(2, LAMBDAS[0]));
    let machines = [spark(GB10_POOL, QWEN38.layers), pinned.clone()];
    let budgets = budgets_on(&model, &mut nodes, &machines);
    assert!(budgets[1].fixed() > budgets[1].pool());
    let plan = cases()[4]
        .plan(&nodes, &budgets, whole)
        .expect("node 0 holds Qwen3.8");
    let over = IdleWhy::Pool {
        fixed: budgets[1].fixed(),
        pool: budgets[1].pool(),
    };
    assert_eq!(plan.owned(), [QWEN38.experts, 0]);
    assert_eq!(plan.idle(), [NodeIdle { node: 1, why: over }]);
    assert_eq!(plan.predict().expect("a prediction").per_node[1].pool, 0);

    let shape = Shape {
        embedding: 40_000_000_000,
        ..QWEN38
    };
    let model = shape.model(false);
    let machine = spark(GB10_POOL, shape.layers);
    let terms = terms_of(&model, &machine, shape.n_embd);
    let mut nodes = spark_nodes(2, shape.layers, fabric(2, LAMBDAS[1]));
    nodes.ruler_permille = 4 * RULER;
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("two Sparks");
    let id = shape.bytes(false).2 * shape.layers as u64;
    let rooms = |budgets: &[NodeBudget]| -> Vec<u64> {
        budgets
            .iter()
            .map(|b| b.pool().saturating_sub(b.fixed()) / id)
            .collect()
    };
    let room = rooms(&budgets);
    assert!(
        room[0] < shape.experts && plan.owned().iter().zip(&room).all(|(o, r)| o <= r),
        "{room:?}: {:?}",
        plan.owned()
    );
    let mut full = machine.clone();
    full.host_mut().reserves.push((
        "full".to_string(),
        budgets[1].pool() - budgets[1].fixed() - id / 2,
    ));
    let mut nodes = spark_nodes(2, shape.layers, fabric(2, LAMBDAS[0]));
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), pinned]);
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("node 0 pages");
    let over = IdleWhy::Pool {
        fixed: budgets[1].fixed(),
        pool: budgets[1].pool(),
    };
    assert_eq!(plan.owned(), [shape.experts, 0]);
    assert_eq!(plan.idle(), [NodeIdle { node: 1, why: over }]);
    let mut nodes = spark_nodes(3, shape.layers, fabric(3, LAMBDAS[0]));
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), full, machine]);
    let room = rooms(&budgets);
    assert!(room[0] < shape.experts && room[1] == 0 && budgets[1].fixed() <= budgets[1].pool());
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("three Sparks");
    assert_eq!(plan.owned()[1], 0, "{room:?}: {:?}", plan.owned());
    assert_eq!(plan.idle().iter().map(|i| i.node).collect::<Vec<_>>(), [1]);

    let shape = Shape {
        name: "two ids",
        layers: 1,
        experts: 2,
        used: 1,
        expert: 40_000_000_000,
        ..QWEN38
    };
    let model = shape.model(false);
    let mut nodes = spark_nodes(3, 1, fabric(3, LAMBDAS[0]));
    let machines: Vec<Machine> = [60_000_000_000, 20_000_000_000, 60_000_000_000]
        .into_iter()
        .map(|free| spark(free, 1))
        .collect();
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], shape.n_embd);
    let id = shape.bytes(false).2;
    let rooms: Vec<u64> = budgets
        .iter()
        .map(|b| (b.pool() - b.fixed()) / id)
        .collect();
    assert_eq!(rooms, [1, 0, 1]);
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("three Sparks");
    assert_eq!(plan.owned(), [1, 0, 1]);
    assert_eq!(plan.idle().iter().map(|i| i.node).collect::<Vec<_>>(), [1]);
    let mut nodes = spark_nodes(2, 1, fabric(2, LAMBDAS[0]));
    nodes.ruler_permille = 1000;
    let budgets = budgets_on(&model, &mut nodes, &machines[..2]);
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("two Sparks");
    assert_eq!(plan.owned(), [2, 0]);
    assert_eq!(plan.idle().iter().map(|i| i.node).collect::<Vec<_>>(), [1]);
}

/// Whether `plan`'s slowest window is at least the rule's optimum and
/// within (1 + 2.5 × ruler) of it, in picoseconds; the windows and the
/// optimum, for the message.
fn balanced(plan: &NodePlan<'_>) -> (bool, Vec<u128>, u128) {
    let windows: Vec<u128> = (0..plan.owned().len())
        .map(|j| plan.window_ps(j).expect("a window"))
        .collect();
    let slowest = windows.iter().copied().max().unwrap_or(0);
    let optimum = plan.balanced_window_ps().expect("the optimum");
    let held = optimum <= slowest && 2000 * slowest <= (2000 + 5 * u128::from(RULER)) * optimum;
    (held, windows, optimum)
}

/// Clause 3, `read_balance`: on every plan the slowest node's window is at
/// least the rule's optimum and within (1 + 2.5 × ruler) of it, also where
/// two pools hold more ids than a u64 counts. The windows are read in
/// picoseconds: three ids of 10⁹ B at top-k 1 read at 10¹⁸ B/s on two nodes
/// of one id's room each split [2, 1], whose windows of 666 and 333 ps are a
/// third past the optimum's 500 ps, which the check refuses.
#[test]
fn read_balance() {
    let check = |cell: Cell, plan: &NodePlan<'_>| {
        let (held, windows, optimum) = balanced(plan);
        assert!(
            held,
            "{cell:?}: windows {windows:?} ps, optimum {optimum} ps"
        );
    };
    each_cell(RULER, 3_000, check);
    let model = tiny(2, 2, 1);
    let machine = spark(3 << 62, 1);
    let link = LinkCost { round_trip_ns: 0 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    nodes.read = 1;
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let terms = Terms::of(&today(&model, &machine, false), 1).expect("the terms");
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(3_000)).expect("two nodes");
    assert_eq!(plan.owned(), [1, 1]);
    check(("u64 pools", 2, 0, false), &plan);
    let id = 1_000_000_000;
    let model = tiny(3, 1, id);
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    nodes.read = 1_000_000_000_000_000_000;
    let machines = with_room(&model, id, &[1, 1]);
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], 1);
    let plan = plan_nodes(&nodes, &terms, &budgets, workload(1)).expect("two nodes");
    assert_eq!(plan.owned(), [2, 1]);
    assert_eq!(balanced(&plan), (false, vec![666, 333], 500));
}

/// Clause 4, `link_share`: the chosen plan's link term is at most a tenth of
/// its step, and a node dropped for it would pass the tenth, read on its
/// whole nanoseconds ([`past_tenth`]). [`expected_sets`] holds the nodes
/// dropped for it to the derived cells where criterion 2 fires; here, a plan
/// that drops one uses two nodes, and Qwen3.8 at N = 4, λ 70, A low is among
/// them. The share is read on the exact steps: three ids of 1,000 B at
/// top-k 1 read at 10¹¹ B/s, λ 17,000 ps, split [2, 1], expose 5,666⅔ of
/// 55,666⅔ ps (10.2 %), and three of 19,199 B at 10¹² B/s, λ 32,000 ps,
/// expose 10,666⅔ of 106,661⅔ ps, past a tenth by ½ ps.
#[test]
fn link_share() {
    let dropped_past = |at: &dyn std::fmt::Debug, plan: &NodePlan<'_>| {
        for i in plan.idle() {
            if let IdleWhy::LinkShare { link_ns, step_ns } = i.why {
                assert!(past_tenth(link_ns, step_ns), "{at:?}: {i:?}");
            }
        }
    };
    let mut fired = Vec::new();
    each_cell(RULER, 3_000, |cell, plan| {
        let step = plan.step_ps().expect("a step");
        let link = plan.exposed_ps(step).expect("the link");
        assert!(
            link.times(10).expect("ten links") <= step,
            "{cell:?}: link {link:?} of step {step:?}"
        );
        dropped_past(&cell, plan);
        if plan
            .idle()
            .iter()
            .any(|i| matches!(i.why, IdleWhy::LinkShare { .. }))
        {
            fired.push(cell);
            assert_eq!(used(plan), 2, "{cell:?}");
        }
    });
    assert!(fired.contains(&("Qwen3.8", 4, 70, false)), "{fired:?}");
    let model = tiny(3, 1, 1_000);
    let machine = spark(GB10_POOL, 1);
    let link = LinkCost { round_trip_ns: 17 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.ruler_permille) = (100_000_000_000, 1);
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let terms = terms_of(&model, &machine, 1);
    let whole = Workload {
        whole: true,
        ..workload(1)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(
        plan.idle(),
        [NodeIdle {
            node: 1,
            why: IdleWhy::LinkShare {
                link_ns: 5,
                step_ns: 55
            }
        }]
    );
    let model = tiny(3, 1, 19_199);
    let link = LinkCost { round_trip_ns: 32 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.ruler_permille) = (1_000_000_000_000, 1);
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let terms = terms_of(&model, &machine, 1);
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(
        plan.idle(),
        [NodeIdle {
            node: 1,
            why: IdleWhy::LinkShare {
                link_ns: 10,
                step_ns: 106
            }
        }]
    );
    dropped_past(&"19,199 B ids", &plan);
}

/// Node 0's bytes a token on a plan of `owned` ids, times the experts: the
/// mean rows of its slots on the remotes, one a slot, kept a fraction.
fn nic_bytes(shape: Shape, owned: &[u64]) -> u128 {
    let layers = shape.layers as u64;
    let k = u128::from(shape.used);
    let remote: u128 = owned[1..].iter().map(|&o| u128::from(o)).sum();
    u128::from(layers * 4 * shape.n_embd) * k * remote
}

/// Node 0's NIC share of a prompt batch on such a plan at S₁ `s1`, a NIC of
/// one byte a second: S₁ times the one-node token's bytes over A and the
/// slowest node's FFN, node 0's with Sh, as a numerator and a denominator.
fn nic_prompt(shape: Shape, owned: &[u64], s1: u64) -> (u128, u128) {
    let (a, sh, e) = shape.bytes(false);
    let layers = shape.layers as u64;
    let (sh, id) = (u128::from(sh * layers), u128::from(e * layers));
    let (k, ex) = (u128::from(shape.used), u128::from(shape.experts));
    let ffn = owned
        .iter()
        .enumerate()
        .map(|(j, &o)| k * u128::from(o) * id + if j == 0 { sh * ex } else { 0 })
        .max()
        .unwrap_or(0);
    let one = u128::from(a) + sh + k * id;
    (
        u128::from(s1) * one * nic_bytes(shape, owned),
        u128::from(a) * ex + ffn,
    )
}

/// Node 0's NIC, in permille, on such a plan: against the decode step and
/// the prompt batch, the larger.
fn nic_of(shape: Shape, owned: &[u64], step_ns: u64, s1: u64) -> u128 {
    let nic = u128::from(GB10_NIC);
    let ex = u128::from(shape.experts);
    let decode = nic_bytes(shape, owned) * PS * 1000 / (u128::from(step_ns) * 1000 * nic * ex);
    let (num, den) = nic_prompt(shape, owned, s1);
    decode.max(num * 1000 / (den * nic))
}

/// Clause 5, `nic_not_max`: a plan whose NIC passes half a decode step or a
/// prompt batch names the node and its busy share, which is the node's reported
/// one and the busiest, exactly in the derived cells where criterion 5 fires on
/// the chosen plan at its split: node 0 takes the rows of the remotes' share of
/// the slots. On the line the side is the one [`expected_sets`] pins: V4.1 at
/// N = 2, λ 25, A low fires at S₁ 3,000. A share of exactly one half reads
/// 500 ‰ and does not fire, at ≤ and not <, and one just past it reads 500 ‰
/// and fires. The prompt counts the split's own shares: a remote that owns
/// every id gains the prompt only Sh. A token's mean bytes stay a fraction:
/// three ids a layer at top-k 1, owned [1, 2], take 8/3 B a token, two thirds
/// of a 4 B row, and node 0's prompt share is 4/7, past the line. The decode
/// line reads the exact step: the same split of ids of 666,666,664 B read at
/// 10¹⁰ B/s, a NIC of 16 B/s and λ 2,000 ps take a step of
/// 333,333,333,333⅓ ps, twice node 0's NIC time of 166,666,666,666⅔ ps, so
/// on the line. Paged, each window's read and fill round down once, together:
/// forty ids of 649,500,000 B at top-k 1 read at 10¹⁹ B/s on two nodes of one
/// id's room each, at a hit rate of 999 ‰ and a miss's fill of 1,000 ps, split
/// [20, 20], take a step of 325.7 ps that reads 324 ps (a read of 64.95 ps and a
/// fill of 19/20 ps a window). A row of 647 values at 8 × 10¹² B/s gives node 0
/// a NIC time of 161.75 ps, half of 323.5 ps, 2.2 ps under the step and so past
/// the 2 ps a one-layer step can read low: no node is named.
#[test]
fn nic_not_max() {
    for rate in PROMPT_RATES {
        let mut fired = Vec::new();
        let mut expected = Vec::new();
        each_cell(RULER, rate, |cell, plan| {
            let bound = plan.nic_bound().expect("a prediction");
            let p = plan.predict().expect("a prediction");
            let busiest = p.per_node.iter().map(|l| l.nic_permille).max().unwrap_or(0);
            match bound {
                Some(b) => {
                    fired.push(cell);
                    assert!(
                        b.busy_permille == p.per_node[b.node].nic_permille
                            && b.busy_permille == busiest
                            && busiest >= 500,
                        "{cell:?}: {b:?}, {:?}",
                        p.per_node
                    );
                }
                None => assert!(busiest <= 500, "{cell:?}: {:?}", p.per_node),
            }
            if derived_at(cell, RULER, rate).fires {
                expected.push(cell);
            }
        });
        assert_eq!(fired, expected, "S₁ {rate}");
    }
    let shape = Shape {
        name: "remote FFN",
        experts: 2,
        used: 2,
        expert: 100_000_000,
        ..V41
    };
    let model = shape.model(false);
    let wide = spark(GB10_POOL, shape.layers);
    let fixed = NodeBudget::of(&today(&model, &wide, true))
        .expect("a budget")
        .fixed();
    let mut nodes = spark_nodes(2, shape.layers, fabric(2, LAMBDAS[0]));
    let machines = [spark(fixed + 2_000_000_000, shape.layers), wide];
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], shape.n_embd);
    let s1 = 6_000;
    let whole = Workload {
        whole: true,
        ..workload(s1)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two Sparks");
    assert_eq!(plan.owned(), [0, 2]);
    let p = plan.predict().expect("a prediction");
    let busy = nic_of(shape, plan.owned(), p.step_ns, s1);
    assert!(
        u128::from(p.per_node[0].nic_permille).abs_diff(busy) <= 1,
        "{} ‰, {busy} ‰",
        p.per_node[0].nic_permille
    );
    assert_eq!(plan.nic_bound().expect("a prediction"), None);
    let (_, den) = nic_prompt(shape, plan.owned(), 1);
    let s1 = u64::try_from(den).expect("an S₁");
    let (num, _) = nic_prompt(shape, plan.owned(), s1);
    let line = u64::try_from(2 * num / den).expect("a NIC rate");
    let at = |nic: u64| {
        let mut nodes = spark_nodes(2, shape.layers, fabric(2, LAMBDAS[0]));
        nodes.nic = nic;
        let budgets = budgets_on(&model, &mut nodes, &machines);
        let whole = Workload {
            prompt_rate: s1,
            ..whole
        };
        let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two Sparks");
        let bound = plan.nic_bound().expect("a prediction");
        (
            plan.predict().expect("a prediction").per_node[0].nic_permille,
            bound.map(|b| b.node),
        )
    };
    assert_eq!(at(line), (500, None), "a NIC at exactly half");
    assert_eq!(at(line - 1), (500, Some(0)), "a NIC just past half");
    let id = 1_000_000_000;
    let model = tiny(3, 1, id);
    let link = LinkCost { round_trip_ns: 0 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.nic, nodes.ruler_permille) = (1, 5, 1);
    let machines = with_room(&model, id, &[1, 2]);
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], 1);
    let whole = Workload {
        whole: true,
        ..workload(1)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(plan.owned(), [1, 2]);
    let p = plan.predict().expect("a prediction");
    assert_eq!(
        (
            plan.nic_bound().expect("a prediction"),
            p.per_node[0].nic_permille
        ),
        (
            Some(NicBound {
                node: 0,
                busy_permille: 571
            }),
            571
        )
    );
    let id = 666_666_664;
    let model = tiny(3, 1, id);
    let link = LinkCost { round_trip_ns: 2 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.nic, nodes.ruler_permille) = (10_000_000_000, 16, 1);
    let machines = with_room(&model, id, &[1, 2]);
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], 1);
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(plan.owned(), [1, 2]);
    let p = plan.predict().expect("a prediction");
    assert_eq!(
        (
            plan.nic_bound().expect("a prediction"),
            p.per_node[0].nic_permille
        ),
        (None, 500)
    );
    let id = 649_500_000;
    let model = tiny(40, 1, id);
    let link = LinkCost { round_trip_ns: 0 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.nic, nodes.ruler_permille) =
        (10_000_000_000_000_000_000, 8_000_000_000_000, 1);
    let machines = with_room(&model, id, &[1, 1]);
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], 647);
    let paged = Workload {
        fills: Some(Fills {
            hit_permille: 999,
            fill_ns: 1,
        }),
        ..workload(1)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, paged).expect("two nodes");
    assert_eq!(
        (plan.owned(), plan.nic_bound().expect("a prediction")),
        (&[20, 20][..], None)
    );
}

/// Clause 6, `messages_o_n`: per layer the busiest node posts
/// 2 × min(k, N − 1) messages over the N nodes the plan uses, never one a
/// slot, and their posts take at most 2 % of the layer. Also where k is
/// under N − 1: one expert a token of seven, two ids a pool, on four Sparks
/// the pools all need, where a token's handoff goes only to the node its
/// slot lands on, and the step and node 0's NIC are that rule's. In every
/// cell node 0's count is the derived cell's.
#[test]
fn messages_o_n() {
    let check = |cell: Cell, plan: &NodePlan<'_>| {
        let p = plan.predict().expect("a prediction");
        let busiest = p
            .per_node
            .iter()
            .map(|l| l.msgs_per_layer)
            .max()
            .unwrap_or(0);
        let k = plan.terms().experts_used;
        let layers = plan.terms().layers.len() as u64;
        assert_eq!(busiest, 2 * k.min(used(plan) - 1), "{cell:?}");
        assert!(
            50 * busiest * POST_NS * layers <= p.step_ns,
            "{cell:?}: {busiest} messages a layer"
        );
    };
    each_cell(RULER, PROMPT_RATES[1], |cell, plan| {
        check(cell, plan);
        let p = plan.predict().expect("a prediction");
        assert_eq!(
            p.per_node[0].msgs_per_layer,
            derived_at(cell, RULER, PROMPT_RATES[1]).msgs,
            "{cell:?}: node 0's messages"
        );
    });
    let shape = Shape {
        name: "k1",
        experts: 7,
        used: 1,
        expert: 1_000_000_000,
        ..V41
    };
    let model = shape.model(false);
    let machine = spark(GB10_POOL, shape.layers);
    let terms = terms_of(&model, &machine, shape.n_embd);
    let mut nodes = spark_nodes(4, shape.layers, fabric(4, LAMBDAS[0]));
    let budgets = budgets_on(&model, &mut nodes, &vec![machine; 4]);
    let whole = Workload {
        whole: true,
        ..workload(3_000)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("k1 on four Sparks");
    assert!(plan.owned().iter().all(|&o| o > 0), "{:?}", plan.owned());
    check(("k1", 4, LAMBDAS[0] / 1000, false), &plan);
    let o: Vec<u128> = plan.owned().iter().map(|&x| u128::from(x)).collect();
    let (a, sh, e) = shape.bytes(false);
    let ps = |b: u64| u128::from(b) * PS / u128::from(GB10_READ);
    let lambda = u128::from(LAMBDAS[0] + HOP_NS) * 1000;
    let layer = o[0] * ps(sh + e) + (o[1] + o[2] + o[3]) * ps(sh).max(lambda + ps(e));
    let p = plan.predict().expect("a prediction");
    assert_eq!(
        u128::from(p.step_ns),
        (ps(a) + shape.layers as u128 * layer / 7) / 1000,
        "{:?}",
        plan.owned()
    );
    let busy = nic_of(shape, plan.owned(), p.step_ns, 3_000);
    assert!(
        u128::from(p.per_node[0].nic_permille).abs_diff(busy) <= 1,
        "{} ‰, {busy} ‰",
        p.per_node[0].nic_permille
    );
}

/// Clause 7, `never_slower`: no chosen plan is slower than the N − 1 plan on
/// the same link: a pair for a ring of three, three switched Sparks for four.
/// [`expected_sets`] holds the nodes named idle for their gain to the derived
/// cells where criterion 7 fires. The ruler is read from the machine: on one
/// whose ruler is 4 %, Qwen3.8 names node 1 idle at N = 2 where it gains
/// under 8 %, at λ 70, and at `RULER` it does not. The gain is read on the
/// exact steps: three ids of 5,000 B at top-k 2 read at 10¹² B/s take
/// 30,000 ps on one node and 27,777⁷⁄₉ ps on two, a gain of 7.4 %, under a
/// ruler of 5 % twice, and read at 10¹⁴ B/s 300 and 277⁷⁄₉ ps, a gain of
/// 7.41 %, under a ruler of 3.8 % twice.
#[test]
fn never_slower() {
    let gain = |plan: &NodePlan<'_>| {
        plan.idle()
            .iter()
            .any(|i| matches!(i.why, IdleWhy::Gain { .. }))
    };
    let mut fired = Vec::new();
    let mut slower = Vec::new();
    each_cell(RULER, 3_000, |cell, plan| {
        if cell.0 == "Qwen3.8" && cell.1 == 2 && gain(plan) {
            fired.push(cell);
        }
        let (_, n, lambda, _) = cell;
        if n == 1 {
            return;
        }
        let case = case_of(cell);
        let mut nodes = spark_nodes(n - 1, case.shape.layers, fabric(n - 1, lambda * 1000));
        if n == 4 {
            nodes.fabric = plan.nodes().fabric;
        }
        let budgets = case.budgets(n - 1);
        let step = |p: &NodePlan<'_>| p.step_ps().expect("a step");
        let before = case
            .plan(&nodes, &budgets, plan.workload())
            .expect("the N − 1 plan");
        if step(plan) > step(&before) {
            slower.push((cell, step(plan), step(&before)));
        }
    });
    let qwen = |n, lambda, high| ("Qwen3.8", n, lambda, high);
    assert_eq!(fired, vec![]);
    assert_eq!(slower, vec![]);
    let mut wide = Vec::new();
    each_cell(4 * RULER, 3_000, |cell, plan| {
        if cell.0 == "Qwen3.8" && cell.1 == 2 && gain(plan) {
            wide.push(cell);
        }
    });
    assert_eq!(wide, vec![qwen(2, 70, false), qwen(2, 70, true)]);
    let model = tiny(3, 2, 5_000);
    let machine = spark(GB10_POOL, 1);
    let link = LinkCost { round_trip_ns: 0 };
    let mut nodes = spark_nodes(2, 1, Fabric::Pair { link });
    (nodes.read, nodes.ruler_permille) = (1_000_000_000_000, 50);
    let budgets = budgets_on(&model, &mut nodes, &[machine.clone(), machine.clone()]);
    let terms = terms_of(&model, &machine, 1);
    let whole = Workload {
        whole: true,
        ..workload(1)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(
        (plan.owned(), plan.idle()),
        (
            &[3, 0][..],
            &[NodeIdle {
                node: 1,
                why: IdleWhy::Gain {
                    without_ns: 30,
                    with_ns: 27
                }
            }][..]
        )
    );
    (nodes.read, nodes.ruler_permille) = (100_000_000_000_000, 38);
    let plan = plan_nodes(&nodes, &terms, &budgets, whole).expect("two nodes");
    assert_eq!(
        (plan.owned(), plan.idle()),
        (
            &[3, 0][..],
            &[NodeIdle {
                node: 1,
                why: IdleWhy::Gain {
                    without_ns: 0,
                    with_ns: 0
                }
            }][..]
        )
    );
}

/// C3 (`docs/research/tierengine.md` (k)), `fills_balance`: the split counts
/// each node's fills and pages no id while a pool has room. V4.1 on three
/// Sparks whose remote pools are 60 GB fits the three pools whole at 115 GB
/// beside them and pages none; at 80 and 60 GB the pools hold fewer than
/// every id, the windows with their fills place the paged ids on both
/// remotes, and the remotes' fills are named apart. Fills under one a token
/// are compared whole: five ids at top-k 1 on three Sparks of one id's room
/// each page one id on node 1 and none on node 2, 0.2 and 0 thousandths of
/// a fill a token at a hit rate of 999 ‰, which are apart.
#[test]
fn fills_balance() {
    let case = &cases()[0];
    let model = V41.model(false);
    let w = Workload {
        fills: Some(Fills {
            hit_permille: 750,
            fill_ns: 430_000,
        }),
        ..workload(3_000)
    };
    let on = |pools: [u64; 2]| {
        let mut nodes: Nodes = spark_nodes(3, V41.layers, fabric(3, LAMBDAS[0]));
        let mut budgets = case.budgets(3);
        for (j, free) in [1, 2].into_iter().zip(pools) {
            let machine = spark(free, V41.layers);
            nodes.machine.nodes[j] = machine.nodes[0].clone();
            budgets[j] = budget_on(&model, &machine, j);
        }
        let plan = case
            .plan(&nodes, &budgets, w)
            .expect("V4.1 on three Sparks");
        let p = plan.predict().expect("a prediction");
        let fills: Vec<u64> = p.per_node.iter().map(|l| l.fills_milli).collect();
        (fills, plan.fills_apart().expect("a prediction"))
    };
    assert_eq!(on([115_000_000_000, 60_000_000_000]), (vec![0, 0, 0], None));
    assert_eq!(
        on([80_000_000_000, 60_000_000_000]),
        (vec![0, 1093, 3125], Some(FillsApart { low: 1, high: 2 }))
    );
    let id = 1_000_000_000;
    let model = tiny(5, 1, id);
    let link = LinkCost { round_trip_ns: 0 };
    let mut nodes = spark_nodes(3, 1, Fabric::Ring { link });
    let machines = with_room(&model, id, &[1, 1, 1]);
    let budgets = budgets_on(&model, &mut nodes, &machines);
    let terms = terms_of(&model, &machines[0], 1);
    let w = Workload {
        fills: Some(Fills {
            hit_permille: 999,
            fill_ns: 1,
        }),
        ..workload(1)
    };
    let plan = plan_nodes(&nodes, &terms, &budgets, w).expect("three Sparks");
    assert_eq!(plan.owned(), [2, 2, 1]);
    assert_eq!(
        plan.fills_apart().expect("a prediction"),
        Some(FillsApart { low: 2, high: 1 })
    );
    let mut terms = terms;
    terms.serial_nvme = 4096;
    let filled = plan_nodes(&nodes, &terms, &budgets, w).expect("three Sparks");
    let faster = plan_nodes(
        &nodes,
        &terms,
        &budgets,
        Workload {
            nvme_read: 2_000_000_000,
            ..w
        },
    )
    .expect("three Sparks with a faster row fill");
    assert_eq!(filled.owned(), [2, 2, 1]);
    assert_eq!(faster.owned(), filled.owned());
    let filled = filled.predict().expect("a prediction");
    let faster = faster.predict().expect("a prediction");
    for (a, b) in filled.per_node.iter().zip(&faster.per_node) {
        assert_eq!(a.window_ns, b.window_ns);
    }
    assert_eq!(filled.step_ns - faster.step_ns, 2048);
}

/// Undefined inputs and sums past their integers are refused by name: a
/// switch whose round trip and hop pass u64, a token whose slots' outcomes
/// pass u128 (512 experts at top-k 15), a ruler outside 1 to 1000 permille,
/// a prompt rate or row width of 0, today's card plan of a machine of two
/// nodes or of a pool whose bytes with its card's context pass u64, and a
/// budget of another node.
#[test]
fn named_refusals() {
    let case = &cases()[4];
    let budgets = case.budgets(2);
    let link = LinkCost {
        round_trip_ns: u64::MAX,
    };
    let far = spark_nodes(2, QWEN38.layers, Fabric::Switch { link, hop_ns: 1 });
    assert!(matches!(
        case.plan(&far, &budgets, workload(3_000)),
        Err(NodesError::Overflow { .. })
    ));
    let mut nodes = spark_nodes(2, QWEN38.layers, fabric(2, LAMBDAS[0]));
    let wide = Terms {
        experts_used: 15,
        ..case.terms.clone()
    };
    assert!(matches!(
        plan_nodes(&nodes, &wide, &budgets, workload(3_000)),
        Err(NodesError::Overflow {
            experts: 512,
            used: 15
        })
    ));
    let flat = Terms {
        n_embd: 0,
        ..case.terms.clone()
    };
    assert!(matches!(
        plan_nodes(&nodes, &flat, &budgets, workload(3_000)),
        Err(NodesError::Undefined("a row width of 0"))
    ));
    assert!(matches!(
        case.plan(&nodes, &budgets, workload(0)),
        Err(NodesError::Undefined("a prompt rate of 0"))
    ));
    let no_drive = Workload {
        nvme_read: 0,
        ..workload(3_000)
    };
    case.plan(&nodes, &budgets, no_drive)
        .expect("no NVMe rows need no drive rate");
    let filled = Terms {
        serial_nvme: 1,
        ..case.terms.clone()
    };
    assert!(matches!(
        plan_nodes(&nodes, &filled, &budgets, no_drive),
        Err(NodesError::Undefined(
            "an NVMe fill rate of 0 for rows read from the NVMe tier"
        ))
    ));
    let slow_drive = Workload {
        nvme_read: 1,
        ..workload(3_000)
    };
    plan_nodes(&nodes, &filled, &budgets, slow_drive)
        .expect("NVMe rows with a positive drive rate");
    for ruler in [0, 1001, u64::MAX] {
        nodes.ruler_permille = ruler;
        assert!(
            matches!(
                case.plan(&nodes, &budgets, workload(3_000)),
                Err(NodesError::Undefined("a ruler outside 1..=1000 permille"))
            ),
            "ruler {ruler}"
        );
    }
    let model = QWEN38.model(false);
    assert!(matches!(
        plan_with(&model, &nodes.machine, CTX, &Kv(0), None),
        Err(super::PlacementError::Nodes { nodes: 2, tiers: 0 })
    ));
    assert!(matches!(
        plan_with(&model, &spark(u64::MAX, QWEN38.layers), CTX, &Kv(0), None),
        Err(super::PlacementError::PoolPastU64 {
            bytes: u64::MAX,
            ..
        })
    ));
    nodes.ruler_permille = RULER;
    assert!(matches!(
        case.plan(&nodes, &case.budgets(1), workload(3_000)),
        Err(NodesError::Budget(1))
    ));
}
