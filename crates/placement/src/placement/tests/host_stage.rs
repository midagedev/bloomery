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
