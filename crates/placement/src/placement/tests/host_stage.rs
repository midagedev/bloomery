use super::*;

/// A cache of `LAYER_KV` bytes a layer and `POSITION_ROWS` beside it, for the host stage.
struct HostKv;

const LAYER_KV: u64 = 1_000;
const POSITION_ROWS: u64 = 7;

impl KvBytes for HostKv {
    fn layer_bytes(&self, _layer: usize, _ctx_max: u64) -> u64 {
        LAYER_KV
    }

    fn stage_position_bytes(&self, _ctx_max: u64) -> u64 {
        POSITION_ROWS
    }
}

/// [`layered`] with one dense attention tensor a layer (a q8_0 matrix of 16 rows).
fn with_attention(layers: usize) -> ModelTensors {
    let mut model = layered(layers);
    for l in 0..layers {
        model.tensors.push(ModelTensor {
            name: format!("attn {l}"),
            shard: 0,
            layer: Some(l),
            role: Role::Attention,
            ty: GgmlType::Q8_0,
            dims: vec![256, 16],
            file_bytes: HEAD,
            gathered_rows: None,
        });
    }
    model
}

fn machine_of(host_bytes: u64) -> Machine {
    Machine {
        cards: Vec::new(),
        tiers: Vec::new(),
        nodes: vec![Node {
            host: Host {
                usable_bytes: host_bytes,
                reserves: Vec::new(),
            },
            unified: None,
        }],
    }
}

/// With no stage card the host is the one stage: every layer, the head and
/// the embedding are its, a model with no layer is refused by name, and a card
/// map keeps its own refusals. Mutant: the host arm taken out of
/// `Stages::new` ends the map in "the cards run 0 layers".
#[test]
fn no_card_makes_the_host_the_one_stage() {
    let stages = Stages::new(5, &[]).expect("the host stage");
    assert!(stages.host);
    assert_eq!(stages.of_layer, [0; 5]);
    assert_eq!(stages.embedding, None);
    match Stages::new(0, &[]) {
        Err(PlacementError::Stages(msg)) => assert!(msg.contains("no layer"), "{msg}"),
        other => panic!("{:?}, want a stage-map refusal", other.map(|s| s.host)),
    }
    let cards = [card("card0", 0..2, true, false)];
    match Stages::new(5, &cards) {
        Err(PlacementError::Stages(msg)) => {
            assert_eq!(msg, "the cards run 2 layers, the model has 5");
        }
        other => panic!("{:?}, want a stage-map refusal", other.map(|s| s.host)),
    }
}

/// A zero-card machine plans every layer, the head and the embedding on the
/// host, in the file's bytes: no card segment, every routed expert on the
/// host, the always-read tensors among the host's tables and the cache in the
/// host's need.
#[test]
fn a_zero_card_plan_puts_everything_on_the_host() {
    let model = with_attention(3);
    let machine = machine_of(1 << 40);
    let plan = plan_with(&model, &machine, 4096, &HostKv, None).expect("the host stage plans");
    assert!(plan.cards.is_empty());
    for row in &plan.rows {
        let t = &model.tensors[row.tensor];
        for s in &row.segments {
            assert_eq!(s.device, Device::Host, "{}", t.name);
            assert_eq!(s.format, Format::HostFile, "{}", t.name);
        }
        assert_eq!(row.stage, 0, "{}", t.name);
    }
    assert_eq!(plan.host.experts, 3 * 8);
    assert_eq!(plan.host.expert_bytes, 3 * 8 * EXPERT);
    let dense = 2 * HEAD + 3 * HEAD;
    assert_eq!(plan.host.table_bytes, dense, "embedding, head, attention");
    assert_eq!(plan.host.stage_kv_bytes, 3 * LAYER_KV + POSITION_ROWS);
    assert_eq!(plan.n_l, [0, 0, 0]);
    assert_eq!(plan.violations().len(), 0, "{:?}", plan.violations());
    let need = workstation::HostNeed::of(&plan, 0);
    assert_eq!(
        (need.experts, need.tables, need.shadows),
        (3 * 8 * EXPERT, dense, 3 * LAYER_KV + POSITION_ROWS)
    );
}

/// A host need past the host's room is a `HostOver` that names the host and
/// both figures, the cache among its terms; the plan is not the old
/// "the cards run 0 layers" stage-map refusal.
#[test]
fn a_host_need_past_the_room_is_refused_by_name() {
    let model = with_attention(3);
    let fits = machine_of(1 << 40);
    let plan = plan_with(&model, &fits, 4096, &HostKv, None).expect("the host stage plans");
    let need = workstation::HostNeed::of(&plan, 0).bytes();
    let tight = machine_of(need - 1);
    let plan = plan_with(&model, &tight, 4096, &HostKv, None).expect("the plan, not a refusal");
    let over: Vec<String> = plan
        .violations()
        .iter()
        .filter(|v| matches!(v, Violation::HostOver { .. }))
        .map(ToString::to_string)
        .collect();
    assert_eq!(over.len(), 1, "{over:?}");
    assert!(over[0].starts_with("host: "), "{}", over[0]);
    assert!(
        over[0].contains(&format!("usable {} B", need - 1)),
        "{}",
        over[0]
    );
    assert!(plan.host.headroom_bytes < 0, "{}", plan.host.headroom_bytes);
    // Without the cache the same room holds.
    let bare = plan_with(&model, &tight, 4096, &NoKv, None).expect("the plan");
    assert!(
        bare.violations()
            .iter()
            .all(|v| !matches!(v, Violation::HostOver { .. })),
        "{:?}",
        bare.violations()
    );
}

/// The stage map's head rule: at most one card carries it, and with none
/// the head stays on the host.
#[test]
fn the_stage_map_takes_at_most_one_head() {
    let map = |head: [bool; 2]| {
        Stages::new(
            8,
            &[
                card("card0", 0..5, head[0], false),
                card("card1", 5..8, head[1], false),
            ],
        )
    };
    let held = |head: [bool; 2]| map(head).unwrap_or_else(|e| panic!("{head:?}: {e}")).head;
    assert_eq!(held([false, false]), None);
    assert_eq!(held([false, true]), Some(1));
    match map([true, true]) {
        Err(PlacementError::Stages(msg)) => {
            assert_eq!(msg, "2 cards carry the head, at most one does");
        }
        other => panic!("{:?}, want a stage-map refusal", other.map(|s| s.head)),
    }
}

/// A plan whose cards carry no head reads it on the host in the file's bytes and
/// the card holds the rest: with the head on a card too small for it the plan
/// names the card over its bound, with the card's head off it plans, the head's
/// row one host segment among the host's tables. Mutant: the head always on its
/// stage card (`place_whole`'s head arm taken out) ends the second arm in a
/// `WrongCard` for the head and the same `CardOver` as the first.
#[test]
fn a_plan_with_no_head_card_leaves_the_head_on_the_host() {
    let model = with_attention(2);
    let usable = 2 * HEAD + HEAD / 2;
    let machine = |head: bool| Machine {
        cards: vec![Card {
            head,
            ..bytes_card("stage", usable, 0..2)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let (with, without) = (machine(true), machine(false));
    let plan = plan_with(&model, &with, 4096, &HostKv, None).expect("the plan");
    assert!(
        matches!(
            plan.violations().as_slice(),
            [Violation::CardOver { card, .. }] if card == "stage"
        ),
        "{:?}",
        plan.violations()
    );
    let plan = plan_with(&model, &without, 4096, &HostKv, None).expect("the plan");
    assert_eq!(plan.violations().len(), 0, "{:?}", plan.violations());
    let head = plan
        .rows
        .iter()
        .find(|r| model.tensors[r.tensor].role == Role::Head)
        .expect("the head's row");
    let [segment] = head.segments.as_slice() else {
        panic!("the head is one segment: {:?}", head.segments);
    };
    assert_eq!(
        (segment.device, segment.format, segment.resident_bytes),
        (Device::Host, Format::HostFile, HEAD)
    );
    assert_eq!(plan.host.table_bytes, 2 * HEAD, "embedding and head");
}
