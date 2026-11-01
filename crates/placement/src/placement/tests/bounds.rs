use super::*;

#[test]
fn a_part_read_probe_covers_short_contexts() {
    use ctx_ask::{Ask, Limit, Probe, Steps};
    use workstation::{HostNeed, least_part_reads, reserve_part_reads};

    struct Kv;
    impl KvBytes for Kv {
        fn layer_bytes(&self, _: usize, ctx: u64) -> u64 {
            ctx
        }
    }
    let model = layered(1);
    let mut machine = Machine {
        cards: vec![bytes_card("stage", HEAD + 1024, 0..1)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let levers = PlanLevers::default();
    let large = plan_host_routed(&model, &machine, 4096, &Kv, &levers).expect("a raw plan");
    assert!(ctx_ask::short_of(&large.violations()).is_some());
    let probed = least_part_reads(|ctx| {
        let probe = plan_host_routed(&model, &machine, ctx, &Kv, &levers).ok()?;
        probe
            .violations()
            .is_empty()
            .then(|| Ok::<_, PlacementError>(probe.cards[0].dense_bytes))
    });
    if let Some(reads) = probed.expect("the probe's reads") {
        reserve_part_reads(&mut machine, reads);
    }
    let chosen = ctx_ask::choose(
        Ask::Set { total: 512 },
        1,
        Limit {
            positions: 4096,
            source: "trained",
        },
        Steps {
            floor: 4096,
            step: 1024,
        },
        |ctx| {
            plan_host_routed(&model, &machine, ctx, &Kv, &levers).map(
                |plan| match ctx_ask::short_of(&plan.violations()) {
                    Some(short) => Probe::Short(short),
                    None => Probe::Holds(plan),
                },
            )
        },
    )
    .expect("an explicit short context fits below the default search floor");
    assert_eq!(chosen.slot_ctx, 512);
    assert_eq!(HostNeed::of(&chosen.plan, 0).read_ahead, HEAD);
}

#[test]
fn a_node_host_projection_keeps_non_routed_rows_on_node_zero() {
    let model = layered(3);
    let machine = Machine {
        cards: vec![bytes_card("stage", HEAD + 5 * EXPERT, 0..3)],
        tiers: vec![bytes_card("tier", 4 * EXPERT, 0..0)],
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("a tier with work");
    let remote = Machine {
        cards: Vec::new(),
        tiers: Vec::new(),
        nodes: machine.nodes.clone(),
    };
    let projected = nodes::routed_host(&model, &plan.rows, 1, &remote, 4)
        .expect("the routed tier's host projection");
    assert!(std::ptr::eq(projected.machine, &remote));
    assert_eq!(projected.host.arena_columns, 4);
    assert_eq!(projected.host.experts, plan.tier_n_l[0].iter().sum::<u64>());
    assert!(projected.host.experts > 0);
    assert_eq!(projected.host.table_bytes, 0);
    assert_eq!(projected.host.pool_card_bytes, 0);
    let mut bytes = 0;
    assert_eq!(projected.rows.len(), model.tensors.len());
    for (index, row) in projected.rows.iter().enumerate() {
        assert_eq!(row.tensor, index);
        let t = &model.tensors[row.tensor];
        if row.segments[0].device == Device::Unused {
            continue;
        }
        assert_eq!(t.role, Role::RoutedExperts);
        let source = plan.rows.iter().find(|r| r.tensor == row.tensor).unwrap();
        let assigned = source
            .segments
            .iter()
            .find(|s| s.device == Device::Card(1))
            .unwrap();
        assert_eq!(row.segments.len(), 1);
        let held = &row.segments[0];
        assert_eq!((held.device, held.format), (Device::Host, Format::HostFile));
        assert_eq!(held.experts, assigned.experts);
        bytes += held.resident_bytes;
    }
    assert_eq!(projected.host.expert_bytes, bytes);
    let stage = nodes::routed_host(&model, &plan.rows, 0, &remote, 1).unwrap();
    assert!(
        stage
            .rows
            .iter()
            .filter(|r| r.segments[0].device != Device::Unused)
            .all(|r| model.tensors[r.tensor].role == Role::RoutedExperts)
    );
    assert_eq!(stage.host.experts, plan.n_l.iter().sum::<u64>());
    let mut changed = plan.rows.clone();
    let mut extra = changed
        .iter()
        .find(|r| r.segments.iter().any(|s| s.device == Device::Card(1)))
        .unwrap()
        .clone();
    let segment = extra
        .segments
        .iter_mut()
        .find(|s| s.device == Device::Card(1))
        .unwrap();
    let ids = segment.experts.as_ref().unwrap();
    let others = (0..model.experts as u32)
        .filter(|id| !ids.ids().contains(id))
        .take(ids.len() as usize)
        .collect();
    segment.experts = Some(ExpertList::new(others, model.experts).unwrap());
    changed.push(extra);
    assert!(
        nodes::routed_host(&model, &changed, 1, &remote, 1)
            .unwrap_err()
            .to_string()
            .contains("different expert lists")
    );
    let empty = nodes::routed_host(&model, &plan.rows, 2, &remote, 1).unwrap();
    assert!(
        empty
            .rows
            .iter()
            .all(|r| r.segments[0].device == Device::Unused)
    );
    assert_eq!(empty.host.expert_bytes, 0);

    let mut pool_machine = machine.clone();
    pool_machine.nodes[0].unified = Some(UnifiedPool { bytes: 1 << 30 });
    pool_machine.tiers[0].scratch_bytes = EXPERT;
    let mut lowered = plan;
    lowered.machine = &pool_machine;
    let half = model.experts / 2;
    nodes::lower_routed(&mut lowered, &[half, model.experts - half]).unwrap();
    assert_eq!(
        lowered.tier_n_l,
        vec![vec![model.experts - half; model.layers]]
    );
    assert!(lowered.n_l.iter().all(|&n| n <= half));
    assert_eq!(lowered.cards[1].expert_bytes, 0);
    assert!(
        lowered.violations().is_empty(),
        "{:?}",
        lowered.violations()
    );
    let stage = &lowered.cards[0];
    assert_eq!(
        lowered.pool_need().unwrap().cards,
        stage.dense_bytes + stage.expert_bytes + stage.rounding_bytes
    );
    assert_eq!(
        lowered.host.pool_card_bytes,
        stage.dense_bytes + stage.expert_bytes + stage.rounding_bytes
    );
    let projected = nodes::routed_host(&model, &lowered.rows, 1, &remote, 4).unwrap();
    assert!(
        projected.host.experts
            > plan_with(&model, &machine, 4096, &NoKv, None)
                .unwrap()
                .cards[1]
                .experts
    );
    for row in &lowered.rows {
        let t = &model.tensors[row.tensor];
        if t.role == Role::RoutedExperts {
            let assigned = row
                .segments
                .iter()
                .find(|s| s.device == Device::Card(1))
                .unwrap();
            assert_eq!(
                assigned.experts.as_ref().unwrap().ids(),
                (half as u32..model.experts as u32).collect::<Vec<_>>()
            );
            assert_eq!(
                (assigned.format, assigned.resident_bytes),
                (Format::HostFile, 0)
            );
        }
    }
    assert!(nodes::lower_routed(&mut lowered, &[model.experts, 0]).is_err());
    assert!(nodes::lower_routed(&mut lowered, &[half, half + 1]).is_err());
    // An explicit ownership seeds the stage alone on node 0, whether
    // the tier shares its pool or sits on a node of its own.
    let mut apart = pool_machine.clone();
    apart.nodes.push(Node {
        host: Host {
            usable_bytes: 1 << 30,
            reserves: Vec::new(),
        },
        unified: Some(UnifiedPool { bytes: 1 << 30 }),
    });
    for m in [&pool_machine, &apart] {
        let owned = nodes::loaded_routed(
            &model,
            m,
            4096,
            &NoKv,
            &PlanLevers::default(),
            RoutedRule::PerType(CardFormat::of_routed),
            0,
            Some(&[half, model.experts - half]),
        )
        .unwrap_or_else(|e| panic!("{} nodes: {e}", m.nodes.len()));
        assert_eq!(owned.tier_n_l, lowered.tier_n_l, "{} nodes", m.nodes.len());
    }

    for ty in [
        GgmlType::Q5_1,
        GgmlType::IQ3_XXS,
        GgmlType::IQ4_XS,
        GgmlType::IQ4_NL,
    ] {
        let mut blocks = model.clone();
        for t in blocks
            .tensors
            .iter_mut()
            .filter(|t| t.role == Role::RoutedExperts)
        {
            t.ty = ty;
            t.file_bytes =
                t.dims.iter().product::<u64>() / ty.blck_size().unwrap() * ty.type_size().unwrap();
        }
        let mut seed = machine.clone();
        seed.tiers.clear();
        for budget in [None, Some(HEAD)] {
            let mut plan = plan_routed(
                &blocks,
                &seed,
                4096,
                &NoKv,
                &PlanLevers {
                    card_budget_bytes: budget.into(),
                },
                RoutedRule::PerType(|_| Some(CardFormat::KQuant)),
            )
            .unwrap();
            if budget.is_some() {
                assert_eq!(plan.n_l, vec![0; blocks.layers]);
            }
            plan.machine = &machine;
            for owned in [[0, model.experts], [half, half]] {
                let mut lowered = plan.clone();
                nodes::lower_routed(&mut lowered, &owned)
                    .unwrap_or_else(|e| panic!("{ty}, {budget:?}, {owned:?}: {e}"));
                assert!(lowered.violations().is_empty());
                for row in &lowered.rows {
                    let t = &blocks.tensors[row.tensor];
                    if t.role != Role::RoutedExperts {
                        let original = &plan.rows[row.tensor];
                        assert_eq!(
                            (row.tensor, row.stage, row.read_bytes),
                            (original.tensor, original.stage, original.read_bytes)
                        );
                        assert_eq!(row.segments.len(), original.segments.len());
                        for (a, b) in row.segments.iter().zip(&original.segments) {
                            assert_eq!(
                                (a.device, a.format, &a.experts, a.resident_bytes),
                                (b.device, b.format, &b.experts, b.resident_bytes)
                            );
                        }
                        continue;
                    }
                    assert_eq!(row.read_bytes, plan.rows[row.tensor].read_bytes);
                    assert_eq!(row.stage, plan.rows[row.tensor].stage);
                    let kept = lowered.n_l[t.layer.unwrap()];
                    let remote = row
                        .segments
                        .iter()
                        .find(|s| s.device == Device::Card(1))
                        .unwrap();
                    assert_eq!(
                        (remote.format, remote.resident_bytes),
                        (Format::HostFile, 0)
                    );
                    assert_eq!(
                        remote.experts,
                        Some(ExpertList::range(owned[0]..model.experts).unwrap())
                    );
                    let card = row.segments.iter().find(|s| s.device == Device::Card(0));
                    assert_eq!(card.is_some(), kept != 0);
                    if let Some(card) = card {
                        assert_eq!(card.format, Format::Card(CardFormat::KQuant));
                    }
                    let host = row.segments.iter().find(|s| s.device == Device::Host);
                    assert_eq!(host.is_some(), kept < owned[0]);
                    if let Some(host) = host {
                        assert_eq!(
                            host.experts,
                            Some(ExpertList::range(kept..owned[0]).unwrap())
                        );
                        assert_eq!(host.format, Format::HostFile);
                        assert_eq!(
                            host.resident_bytes,
                            (owned[0] - kept) * t.file_bytes / model.experts
                        );
                    }
                }
                assert_eq!(lowered.cards[1].expert_bytes, 0);
            }
        }
    }
}

/// An nvfp4 stack loads in its planes under the default expert rule, and
/// its card segment counts one f32 scale an expert it holds after the
/// planes, where a dense nvfp4 tensor's segment holds its one scale. The
/// host's experts of a stack are its file bytes there, which the host's
/// nvfp4 dot reads, whether a card holds the rest as planes (a residency
/// swap stages each plane from the file's blocks) or none. Mutant, the rule
/// before: a split between a card and the host was refused by name.
// PIN(2026-10-31): an NVFP4 stack splits between a card and the host; a swap stages its planes.
#[test]
fn an_nvfp4_stack_holds_a_scale_an_expert_on_a_card_and_its_file_bytes_on_the_host() {
    use GgmlType::NVFP4;
    assert_eq!(CardFormat::of_routed(NVFP4), Some(CardFormat::Nvfp4Planes));
    assert_eq!(CardFormat::of_routed(GgmlType::Q5_K), None);
    let model = legged(&[[NVFP4; 3]]);
    let i = model
        .tensors
        .iter()
        .position(|t| t.role == Role::RoutedExperts)
        .expect("a stack");
    let t = &model.tensors[i];
    let list = |ids: &[u32]| ExpertList::new(ids.to_vec(), 8).expect("a list");
    let row = routed_row_on(
        i,
        t,
        0,
        vec![(0, list(&[6, 1, 4])), (1, list(&[0, 2, 3, 5, 7]))],
        &model,
    )
    .expect("every expert on a card");
    // Three experts of four rows of 256 values: 32 code words and 4 scale
    // words a row, then three scales.
    let held = [12 * 128, 12 * 16, 3 * 4];
    let seg = &row.segments[0];
    assert_eq!(seg.buffer_bytes(t, model.experts).expect("buffers"), held);
    assert_eq!(seg.resident_bytes, held.iter().sum::<u64>());
    let dense = ModelTensor {
        dims: vec![256, 4],
        role: Role::Attention,
        ..t.clone()
    };
    // PIN(2026-10-20): a dense nvfp4 tensor now holds its one scale for the GEMM's store; it was the planes alone.
    assert_eq!(
        card_buffers(&dense, CardFormat::Nvfp4Planes, 4).expect("buffers"),
        [4 * 128, 4 * 16, 4]
    );
    let split = routed_row(i, t, 0, list(&[0, 1]), &model).expect("a card and the host");
    let [card, host] = split.segments.as_slice() else {
        panic!("a card and a host segment, got {:?}", split.segments);
    };
    assert_eq!(
        card.buffer_bytes(t, model.experts).expect("buffers"),
        [8 * 128, 8 * 16, 2 * 4]
    );
    assert_eq!(
        (host.device, host.format, host.resident_bytes),
        (Device::Host, Format::HostFile, t.file_bytes / 8 * 6)
    );
    assert_eq!(
        host.experts.as_ref().map(ToString::to_string).as_deref(),
        Some("2..8")
    );
    let host = routed_row_on(i, t, 0, Vec::new(), &model).expect("every expert on the host");
    let [seg] = host.segments.as_slice() else {
        panic!("one host segment, got {:?}", host.segments);
    };
    assert_eq!(
        (seg.device, seg.format, seg.resident_bytes),
        (Device::Host, Format::HostFile, t.file_bytes)
    );
    assert_eq!(seg.experts.as_ref().map(ExpertList::len), Some(8));
}

/// The host holds a routed stack's experts only where a host kernel runs them
/// ([`crate::kernels::host_stack`]): a BF16 or F16 stack, which no host kernel
/// reads, and NVFP4 rows 48 values wide, whole 16-value groups the host's
/// 32-value rows do not cut, are refused by name with the experts the host
/// would hold, by the same rule whether a split or the planner put them there;
/// NVFP4, MXFP4 and Q4_K stacks at whole rows go to the host. Mutant, the
/// rule before: a host segment was the stack's file bytes whatever the host
/// ran.
#[test]
fn the_host_holds_experts_only_where_a_host_kernel_runs_them() {
    let stack = |ty: GgmlType, k: u64| {
        let mut model = legged(&[[GgmlType::Q4_K; 3]]);
        let i = model
            .tensors
            .iter()
            .position(|t| t.role == Role::RoutedExperts)
            .expect("a stack");
        let blocks = k / ty.blck_size().expect("a sized type");
        let t = &mut model.tensors[i];
        (t.ty, t.dims[0]) = (ty, k);
        t.file_bytes = ty.type_size().expect("a sized type") * blocks * 32;
        (model, i)
    };
    let list = |ids: &[u32]| ExpertList::new(ids.to_vec(), 8).expect("a list");
    for (ty, k) in [
        (GgmlType::BF16, 256),
        (GgmlType::F16, 256),
        (GgmlType::NVFP4_ROWS, 48),
    ] {
        let (model, i) = stack(ty, k);
        let t = &model.tensors[i];
        // The card's planes take no 48-value row, so those rows stay whole on the host.
        let splits = if ty == GgmlType::NVFP4_ROWS { 1 } else { 2 };
        for on_card in [Vec::new(), vec![(0, list(&[0, 1]))]]
            .into_iter()
            .take(splits)
        {
            match routed_row_on(i, t, 0, on_card, &model) {
                Err(e @ PlacementError::NoHostKernel { .. }) => {
                    let e = e.to_string();
                    assert!(
                        e.contains(&t.name) && e.contains(&format!("{ty} rows of {k} values")),
                        "{e}"
                    );
                }
                other => panic!("{ty} at {k}: {other:?}, not the host kernel refusal"),
            }
        }
    }
    for ty in [GgmlType::NVFP4, GgmlType::MXFP4, GgmlType::Q4_K] {
        let (model, i) = stack(ty, 256);
        let row = routed_row_on(i, &model.tensors[i], 0, Vec::new(), &model)
            .unwrap_or_else(|e| panic!("{ty} on the host: {e}"));
        assert_eq!(row.segments[0].device, Device::Host, "{ty}");
    }
    let mut bf16 = legged(&[[GgmlType::Q4_K; 3], [GgmlType::BF16; 3]]);
    bf16.bf16_words = |_| false;
    let machine = Machine {
        cards: vec![bytes_card("stage", HEAD + 64 * EXPERT, 0..2)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let leg = RoutedRule::Leg {
        gate_up: |ty| (ty == GgmlType::Q4_K).then_some(CardFormat::KQuant),
        down: |ty| (ty == GgmlType::Q4_K).then_some(CardFormat::KQuant),
        pair: None,
    };
    match plan_routed(&bf16, &machine, 4096, &NoKv, &PlanLevers::default(), leg) {
        Err(PlacementError::NoHostKernel { ty, experts, .. }) => {
            assert_eq!((ty, experts.as_str()), (GgmlType::BF16, "0..8"));
        }
        other => panic!(
            "a BF16 layer the card leg leaves to the host: {:?}",
            other.err()
        ),
    }
}

/// A card leg judges a layer by its parts: Qwen3-30B's leg (a Q4_K or
/// Q5_K gate·up, a Q4_K, Q5_K or Q6_K down) cards a Q4_K gate and up over
/// a Q6_K down, and keeps on the host a Q6_K gate and up (a type only its
/// down reads) and a gate and up of two types — the layers its load
/// refuses — where the per-type rule over the union of its types cards
/// all four. A routed stack whose name names no part is refused by name
/// under the leg and placed under the per-type rule.
#[test]
fn a_leg_cards_a_layer_by_its_parts() {
    use GgmlType::{Q4_K, Q5_K, Q6_K};
    let gate_up: RoutedFormat = |ty| matches!(ty, Q4_K | Q5_K).then_some(CardFormat::KQuant);
    let down: RoutedFormat = |ty| matches!(ty, Q4_K | Q5_K | Q6_K).then_some(CardFormat::KQuant);
    let leg = RoutedRule::Leg {
        gate_up,
        down,
        pair: None,
    };
    let union = RoutedRule::PerType(down);
    let machine = |layers: usize| Machine {
        cards: vec![bytes_card("stage", 1 << 30, 0..layers)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let (m1, m4) = (machine(1), machine(4));
    let levers = PlanLevers::default();
    let model = legged(&[
        [Q4_K, Q4_K, Q6_K],
        [Q6_K, Q6_K, Q6_K],
        [Q4_K, Q5_K, Q4_K],
        [Q5_K, Q5_K, Q5_K],
    ]);
    let plan = plan_routed(&model, &m4, 4096, &NoKv, &levers, leg).expect("the leg");
    assert_eq!(plan.n_l, vec![8, 0, 0, 8]);
    let plan = plan_routed(&model, &m4, 4096, &NoKv, &levers, union).expect("per type");
    assert_eq!(plan.n_l, vec![8, 8, 8, 8]);
    let unnamed = layered(1);
    match plan_routed(&unnamed, &m1, 4096, &NoKv, &levers, leg) {
        Err(e) => assert_eq!(
            e.to_string(),
            "tensor experts 0: a routed stack whose name names no gate, up or down part, \
             which a card leg's rule places by part"
        ),
        Ok(_) => panic!("the leg placed a stack of no part"),
    }
    let plan = plan_routed(&unnamed, &m1, 4096, &NoKv, &levers, union)
        .expect("the per-type rule places a stack of no part");
    assert_eq!(plan.n_l, vec![8]);
}

#[test]
fn a_union_recounts_uploads_selected_forms_and_capacity() {
    use crate::kernels::{DownAct, DownForms};
    use workstation::{TIER_BATCH_RESERVE, try_tier_leg_batch_bytes_of};

    let mut model = legged(&[
        [GgmlType::IQ3_S, GgmlType::IQ3_S, GgmlType::Q5_K],
        [GgmlType::IQ4_XS, GgmlType::IQ4_XS, GgmlType::Q8_0],
    ]);
    model.experts_used = 6;
    let forms = DownForms::of(DownAct::Q8Act).with(DownAct::Q8Blocks32);
    let batch = try_tier_leg_batch_bytes_of(256, 256, 6, (2, 3), forms, false).unwrap();
    let mut stage = bytes_card("stage", 1 << 30, 0..2);
    stage.tier_leg = batch.leg;
    stage
        .reserves
        .push((TIER_BATCH_RESERVE.into(), batch.card().unwrap()));
    let flat = Machine {
        cards: vec![stage],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let machine = Machine {
        tiers: vec![bytes_card("tier", 1 << 30, 0..0)],
        ..flat.clone()
    };
    let rule = crate::accepts::routed_rule(crate::accepts::Program::FlintBody);
    let (index, iq3) = model
        .tensors
        .iter()
        .enumerate()
        .find(|(_, t)| t.role == Role::RoutedExperts && t.ty == GgmlType::IQ3_S)
        .unwrap();
    assert!(matches!(
        routed_row(
            index,
            iq3,
            0,
            ExpertList::new(vec![0], model.experts).unwrap(),
            &model
        ),
        Err(PlacementError::NoCardFormat { .. })
    ));
    let mut rows = plan_routed(&model, &flat, 1, &NoKv, &PlanLevers::default(), rule)
        .unwrap()
        .rows;
    for row in &mut rows {
        let tensor = &model.tensors[row.tensor];
        if tensor.role == Role::RoutedExperts {
            let device = usize::from(tensor.layer == Some(1));
            *row = routed_row_on_as(
                row.tensor,
                tensor,
                0,
                vec![(device, ExpertList::new(vec![0], model.experts).unwrap())],
                &model,
                rule.format(tensor).unwrap(),
            )
            .unwrap();
        }
    }
    let plan = totals(
        &model,
        &machine,
        1,
        rows,
        &[(0, 0, 0); 2],
        (vec![1, 0], vec![vec![0, 1]]),
        card_budget::CardBudget::default(),
    )
    .unwrap();
    let joined = union_plan(&plan, &flat).unwrap();
    assert_eq!(joined.n_l, [1, 1]);
    assert!(joined.tier_n_l.is_empty());
    assert_eq!(
        joined.cards[0].expert_bytes,
        plan.cards[0].expert_bytes + plan.cards[1].expert_bytes
    );
    assert_eq!(joined.cards[0].experts, 2);
    let heap = card_heaps(&model, &flat, &joined.rows).0.remove(0);
    assert_eq!(
        joined.cards[0].rounding_bytes,
        heap.taken - joined.cards[0].dense_bytes - joined.cards[0].expert_bytes
    );
    let reserve = batch.staging + batch.leg.unwrap().bytes(forms).unwrap();
    assert_eq!(joined.cards[0].reserve_bytes, reserve);
    assert_eq!(
        joined.card_reserve(0, TIER_BATCH_RESERVE).unwrap(),
        Some(reserve)
    );
    assert!(reserve > plan.cards[0].reserve_bytes);
    assert_eq!(joined.cards[0].kv_bytes, plan.cards[0].kv_bytes);
    for (before, after) in plan.rows.iter().zip(&joined.rows) {
        let host = |row: &Row| {
            row.segments
                .iter()
                .filter(|s| !matches!(s.device, Device::Card(_)))
                .map(|s| {
                    (
                        s.device,
                        s.resident_bytes,
                        s.experts.as_ref().map(|e| e.ids().to_vec()),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(host(before), host(after));
    }
    let mut conflicting = plan.clone();
    let row = conflicting
        .rows
        .iter_mut()
        .find(|r| r.segments.iter().any(|s| s.device == Device::Card(1)))
        .unwrap();
    let mut other = row
        .segments
        .iter()
        .find(|s| s.device == Device::Card(1))
        .unwrap()
        .clone();
    other.device = Device::Card(0);
    other.format = Format::Card(CardFormat::F32);
    row.segments.push(other);
    assert!(matches!(
        union_plan(&conflicting, &flat),
        Err(PlacementError::Tensor { .. })
    ));
    let mut short = flat.clone();
    short.cards[0].usable_bytes = joined.cards[0].bytes() - 1;
    assert!(matches!(union_plan(&plan, &short),
        Err(PlacementError::Metadata { key, .. }) if key == "union reference"));
    let mut wrong = flat.clone();
    wrong.cards.clear();
    assert!(matches!(
        union_plan(&plan, &wrong),
        Err(PlacementError::Stages(_))
    ));
}

/// Scratch enters each expert candidate's footprint monotonically.
/// A boundary that admits one down form cannot pay for an unused second
/// form, and adding the first expert of that form pays its exact bytes.
#[test]
fn a_tier_jointly_selects_expert_rows_and_down_forms() {
    use crate::accepts::{Program, routed_rule};
    use crate::kernels::{DownAct, DownForms};
    use GgmlType::{Q5_K, Q8_0};
    use workstation::{TIER_BATCH_RESERVE, try_tier_leg_batch_bytes_of};

    let both = DownForms::of(DownAct::Q8Act).with(DownAct::Q8Blocks32);
    let mixed = try_tier_leg_batch_bytes_of(256, 256, 1, (2, 3), both, false).expect("tier bytes");
    let rule = routed_rule(Program::Qwen35Body);
    for (downs, first, batch) in [
        ([Q5_K, Q8_0], DownAct::Q8Act, mixed),
        ([Q8_0, Q5_K], DownAct::Q8Blocks32, mixed),
        (
            [Q5_K; 2],
            DownAct::Q8Act,
            workstation::tier_leg_batch_bytes(256, 256, 1, 2, 3).expect("valid tier bytes"),
        ),
    ] {
        let scratch = |forms| crate::kernels::leg_bytes(2, 1, 256, 256, forms);
        let selected =
            DownForms::of(first).with(crate::kernels::down(downs[1]).expect("second down").1);
        let mut model = legged(&[[Q5_K, Q5_K, downs[0]], [Q5_K, Q5_K, downs[1]]]);
        model.experts = 1;
        model.experts_used = 1;
        for t in model
            .tensors
            .iter_mut()
            .filter(|t| t.role == Role::RoutedExperts)
        {
            t.dims[1] = 256;
            t.dims[2] = 1;
            t.file_bytes *= 8;
        }
        let expert = |l| {
            model
                .tensors
                .iter()
                .filter(|t| t.role == Role::RoutedExperts && t.layer == Some(l))
                .map(|t| t.file_bytes / model.experts)
                .sum::<u64>()
        };
        let one = batch.staging + scratch(DownForms::of(first)) + expert(0);
        let two = batch.staging + scratch(selected) + expert(0) + expert(1);
        for (room, counts, forms) in [
            (one - 1, [0, 0], DownForms::NONE),
            (one, [1, 0], DownForms::of(first)),
            (two - 1, [1, 0], DownForms::of(first)),
            (two, [1, 1], selected),
        ] {
            let mut tier = bytes_card("tier", room, 0..0);
            tier.reserves
                .push((TIER_BATCH_RESERVE.into(), batch.card().unwrap()));
            tier.tier_leg = batch.leg;
            let machine = Machine {
                cards: vec![bytes_card("stage", HEAD, 0..2)],
                tiers: vec![tier],
                nodes: vec![Node {
                    host: host(),
                    unified: None,
                }],
            };
            let plan = plan_routed(&model, &machine, 1, &NoKv, &PlanLevers::default(), rule)
                .expect("the joint plan");
            assert_eq!(plan.n_l, [0, 0]);
            assert_eq!(plan.tier_n_l[0], counts, "{downs:?}, {room}");
            let selected = batch.staging + scratch(forms);
            assert_eq!(
                plan.card_reserve(1, TIER_BATCH_RESERVE)
                    .expect("selected reserve"),
                Some(selected)
            );
            assert_eq!(plan.cards[1].reserve_bytes, selected);
            let rows = expert(0) * counts[0] + expert(1) * counts[1];
            assert_eq!(plan.cards[1].bytes(), selected + rows);
            assert!(plan.violations().is_empty(), "{downs:?}, {room}");
        }
    }
}

/// A malformed descriptor must refuse a plan and a later selected-reserve
/// read, not wrap to a small scratch or masquerade as a fixed-reserve card.
#[test]
fn a_tier_refuses_invalid_scratch_descriptors_before_placement() {
    use crate::accepts::{Program, routed_rule};
    use crate::kernels::{DownAct, DownForms};
    use workstation::{TIER_BATCH_RESERVE, TierLegScratch, try_tier_leg_batch_bytes_of};

    let forms = DownForms::of(DownAct::Q8Act);
    let batch =
        try_tier_leg_batch_bytes_of(256, 256, 1, (1, 1), forms, false).expect("valid batch");
    let good = batch.leg.expect("a leg descriptor");
    let mut model = legged(&[[GgmlType::Q5_K; 3]]);
    model.experts_used = 1;
    for t in model
        .tensors
        .iter_mut()
        .filter(|t| t.role == Role::RoutedExperts)
    {
        t.dims[1] = 256;
        t.file_bytes *= 64;
    }
    let machine = |leg| {
        let mut tier = bytes_card("tier", 1 << 30, 0..0);
        tier.reserves
            .push((TIER_BATCH_RESERVE.into(), batch.card().unwrap()));
        tier.tier_leg = Some(leg);
        Machine {
            cards: vec![bytes_card("stage", HEAD, 0..1)],
            tiers: vec![tier],
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        }
    };
    let valid = machine(good);
    let rule = routed_rule(Program::Qwen35Body);
    let plan = plan_routed(&model, &valid, 1, &NoKv, &PlanLevers::default(), rule)
        .expect("valid descriptor plans");
    for leg in [
        TierLegScratch {
            chunk: 1 << 61,
            n_used: 8,
            ..good
        },
        TierLegScratch { chunk: 0, ..good },
        TierLegScratch { ff: 0, ..good },
        TierLegScratch { n_embd: 0, ..good },
    ] {
        let bad = machine(leg);
        assert!(
            matches!(
                plan_routed(&model, &bad, 1, &NoKv, &PlanLevers::default(), rule),
                Err(PlacementError::Metadata { key, .. }) if key == "tier leg scratch"
            ),
            "{leg:?}"
        );
        let mut stale = plan.clone();
        stale.machine = &bad;
        assert!(
            matches!(
                stale.card_reserve(1, TIER_BATCH_RESERVE),
                Err(PlacementError::Metadata { key, .. }) if key == "tier leg scratch"
            ),
            "{leg:?}"
        );
        assert!(
            matches!(
                try_tier_leg_batch_bytes_of(
                    leg.n_embd, leg.ff, leg.n_used, (leg.chunk, 1), leg.downs, leg.input_scales,
                ),
                Err(PlacementError::Metadata { key, .. }) if key == "tier leg scratch"
            ),
            "{leg:?}"
        );
    }
    assert!(try_tier_leg_batch_bytes_of(256, 256, 8, (1, 1 << 61), forms, false).is_err());
    let inconsistent = workstation::TierBatchBytes {
        scratch: 0,
        ..batch
    };
    assert!(matches!(
        inconsistent.card(),
        Err(PlacementError::Metadata { key, .. }) if key == "tier leg batch"
    ));
}

// PIN(2026-10-22): the Qwen3.6 leg runs a Q8_0 down and (once the leg has the
// arm) a Q6_K gate and up through the common leg, so the layers that held those
// types hold types it still has no GEMM for (an MXFP4 down, an IQ2_XS gate and
// up); its rows are the ones the test held, and Qwen3.8's gain the MXFP4 down
// its Q8_0 down did not have.
// PIN(2026-10-25): IQ2_XS has a GEMM now, so an MXFP4 gate and up stand for the
// types with none; the rows are unchanged.

#[test]
fn host_only_reasons_follow_the_program_and_part_in_file_order() {
    use crate::accepts::{Program, routed_rule};
    use GgmlType::{MXFP4, Q4_K, Q5_K, Q6_K};
    let model = legged(&[
        [Q4_K, Q4_K, Q6_K],
        [Q4_K, Q4_K, MXFP4],
        [MXFP4, MXFP4, Q6_K],
        [Q4_K, Q5_K, Q4_K],
    ]);
    let rows = |program| {
        routed_rule(program)
            .host_only(&model)
            .unwrap()
            .into_iter()
            .map(|h| (h.layer, h.tensor.name.as_str()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        rows(Program::Qwen35Body),
        [
            (1, "ffn_down_exps.weight"),
            (2, "ffn_gate_exps.weight"),
            (2, "ffn_up_exps.weight"),
            (3, "ffn_up_exps.weight"),
        ]
    );
    assert_eq!(
        rows(Program::Qwen38Body),
        [
            (0, "ffn_down_exps.weight"),
            (1, "ffn_down_exps.weight"),
            (2, "ffn_gate_exps.weight"),
            (2, "ffn_up_exps.weight"),
            (2, "ffn_down_exps.weight"),
            (3, "ffn_up_exps.weight"),
            (3, "ffn_down_exps.weight"),
        ]
    );
    // The reason names the failing kernel contract: an unsupported part
    // is one cause, a gate/up type mismatch another, and neither is empty
    // or collapsed into the other. The wording itself is not pinned.
    let why = |program: Program, layer: usize, tensor: &str| {
        routed_rule(program)
            .host_only(&model)
            .unwrap()
            .into_iter()
            .find(|h| h.layer == layer && h.tensor.name == tensor)
            .map(|h| h.why)
            .expect("the cause is named")
    };
    let unsupported = why(Program::Qwen35Body, 1, "ffn_down_exps.weight");
    let mismatch = why(Program::Qwen35Body, 3, "ffn_up_exps.weight");
    assert!(!unsupported.is_empty(), "{unsupported}");
    assert!(!mismatch.is_empty(), "{mismatch}");
    assert_ne!(
        unsupported, mismatch,
        "an unsupported part and a gate/up mismatch are distinct causes"
    );
    let empty = ModelTensors {
        tensors: Vec::new(),
        ..model
    };
    assert!(
        routed_rule(Program::Qwen35Body)
            .host_only(&empty)
            .unwrap()
            .is_empty()
    );
}

/// The part a routed stack's name names, a draft layer's among them; a
/// fused gate·up stack and any other name name none.
#[test]
fn a_stack_name_names_its_part() {
    for (name, part) in [
        ("ffn_gate_exps.weight", Some(ExpertPart::Gate)),
        ("ffn_up_exps.weight", Some(ExpertPart::Up)),
        ("ffn_down_exps.weight", Some(ExpertPart::Down)),
        ("nextn.ffn_down_exps.weight", Some(ExpertPart::Down)),
        ("ffn_gate_up_exps.weight", None),
        ("ffn_down_shexp.weight", None),
        ("experts 0", None),
    ] {
        assert_eq!(ExpertPart::of_name(name), part, "{name}");
    }
}

/// A card another process holds: the census's free reading caps the
/// expert rule — fewer experts fit than the card's usable bytes hold,
/// the headroom and the violations against the free bytes — and a card
/// budget below the free reading still caps below it.
#[test]
fn a_held_card_plans_within_its_free_bytes() {
    let model = layered(3);
    let usable = HEAD + 24 * EXPERT;
    let machine = |free: Option<u64>| Machine {
        cards: vec![Card {
            free_bytes: free,
            held_by: free.map(|_| "pid 7 (977 MiB)"),
            ..bytes_card("held", usable, 0..3)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let held_machine = machine(Some(HEAD + 12 * EXPERT));
    let plan = plan_with(&model, &held_machine, 4096, &NoKv, None).expect("a held card plans");
    let held: u64 = plan.n_l.iter().sum();
    assert!(held > 0 && held < 24, "a held card keeps fewer than all 24");
    assert_eq!(
        plan.usable_bytes(&plan.machine.cards[0]),
        HEAD + 12 * EXPERT
    );
    assert!(plan.violations().is_empty());
    // A card budget above the free reading leaves the free cap binding;
    // one below it caps below.
    let above = plan_with(&model, &held_machine, 4096, &NoKv, Some(usable))
        .expect("plans under a budget above free");
    assert_eq!(
        above.n_l.iter().sum::<u64>(),
        held,
        "a budget past free changes nothing"
    );
    let below = plan_with(&model, &held_machine, 4096, &NoKv, Some(HEAD + 6 * EXPERT))
        .expect("plans under a budget below free");
    assert_eq!(below.n_l.iter().sum::<u64>(), 6);
    // A full free reading (a quiet card) plans every expert, as the
    // census-free card always did.
    for free in [usable, u64::MAX] {
        let full = machine(Some(free));
        let plan = plan_with(&model, &full, 4096, &NoKv, None).expect("plans");
        assert_eq!(plan.n_l.iter().sum::<u64>(), 24, "free {free}");
        assert!(plan.violations().is_empty());
    }
    let census_free = machine(None);
    let plan = plan_with(&model, &census_free, 4096, &NoKv, None).expect("plans");
    assert_eq!(plan.n_l.iter().sum::<u64>(), 24);
}

/// The host's need by page class: every addend of the need in the class
/// the one table gives it, the OS reserve out, so the classes sum to
/// [`workstation::HostNeed::bytes`] under each way of holding the set. The
/// lock moves the host set, the tables and the churn pool to the locked
/// class and leaves the row cache and the run-ahead window in the page
/// cache; a pinned arena moves from the anonymous class to the pinned one.
/// A folder's part reads are counted once, as its read-ahead (a need that
/// also kept them among its reserves would put them in the run-ahead window
/// too), and the stream's staging is pinned beside them. A reserve with no
/// class, a need whose OS reserve is not the host's and one whose reserves
/// are under the host's own are refused by name.
#[test]
fn a_host_need_is_held_by_page_class() {
    use workstation::{
        CHECKPOINTS_RESERVE, HostNeed, Hosting, NeedClassError, NeedClasses, NeedTerm, OS_RESERVE,
        PART_READS, POINT_ROWS_RESERVE, PageClass, TIER_BATCH_HOST_RESERVE,
        XSTREAM_STAGING_RESERVE,
    };
    let model = layered(3);
    let machine_with = |reserves: Vec<(String, u64)>| Machine {
        cards: vec![bytes_card("c", HEAD + 6 * EXPERT, 0..3)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: Host {
                usable_bytes: 1 << 40,
                reserves,
            },
            unified: None,
        }],
    };
    let machine = machine_with(vec![
        (OS_RESERVE.to_string(), 100),
        (TIER_BATCH_HOST_RESERVE.to_string(), 40),
        (CHECKPOINTS_RESERVE.to_string(), 30),
        (POINT_ROWS_RESERVE.to_string(), 20),
    ]);
    let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("plans");
    assert!(plan.host.experts > 0, "the host holds experts");
    // The NVMe split's arena and run-ahead window, the row cache, the
    // tables and a unified pool's share, none of which the plan above has.
    let (rows, window, arena, cards, tables, shadows, extra) = (8, 5, 7, 11, 13, 17, 6);
    plan.host.row_reserve_bytes = rows;
    plan.host.reserve_bytes += rows + window;
    plan.host.nvme_arena_bytes = arena;
    plan.host.pool_card_bytes = cards;
    plan.host.table_bytes = tables;
    plan.host.shadow_bytes = shadows;
    let need = HostNeed::of(&plan, extra);
    let experts = need.experts;
    assert_eq!(need.reserves, 100 + 40 + 30 + 20 + rows + window);
    let files = experts + tables + extra;
    let cases = [
        (
            Hosting {
                lock: false,
                arena_pinned: false,
            },
            NeedClasses {
                pinned: shadows + 40 + 30 + 20,
                anon: arena,
                cache: files + rows + window,
                locked: 0,
                unified: cards,
            },
        ),
        (
            Hosting {
                lock: true,
                arena_pinned: false,
            },
            NeedClasses {
                pinned: shadows + 40 + 30 + 20,
                anon: arena,
                cache: rows + window,
                locked: files,
                unified: cards,
            },
        ),
        (
            Hosting {
                lock: false,
                arena_pinned: true,
            },
            NeedClasses {
                pinned: shadows + 40 + 30 + 20 + arena,
                anon: 0,
                cache: files + rows + window,
                locked: 0,
                unified: cards,
            },
        ),
    ];
    for (how, want) in cases {
        let got = need.classes(&plan, how).expect("a need of this plan");
        assert_eq!(got, want, "{how:?}");
        assert_eq!(
            got.bytes(),
            need.bytes(),
            "{how:?}: the classes sum to the need"
        );
        // Reads past the churn pool are the peak's, the pool out: the load reads it in after them.
        let reading = HostNeed {
            read_ahead: 17,
            ..need
        };
        let with_reads = reading.classes(&plan, how).expect("a load's read reserve");
        assert_eq!(with_reads.anon, got.anon + 17);
        assert_eq!(with_reads.bytes(), reading.bytes());
        assert_eq!(reading.bytes(), need.bytes() - extra + 17);
    }
    // The table: each term's class under both holdings.
    let locked = Hosting {
        lock: true,
        arena_pinned: true,
    };
    let loose = Hosting {
        lock: false,
        arena_pinned: false,
    };
    let class = |t: NeedTerm| (t.class(loose), t.class(locked));
    assert_eq!(
        NeedTerm::ALL.map(class),
        [
            (PageClass::Cache, PageClass::Locked),
            (PageClass::Cache, PageClass::Locked),
            (PageClass::Pinned, PageClass::Pinned),
            (PageClass::Pinned, PageClass::Pinned),
            (PageClass::Pinned, PageClass::Pinned),
            (PageClass::Pinned, PageClass::Pinned),
            (PageClass::Pinned, PageClass::Pinned),
            (PageClass::Cache, PageClass::Cache),
            (PageClass::Cache, PageClass::Cache),
            (PageClass::Cache, PageClass::Locked),
            (PageClass::Anon, PageClass::Anon),
            (PageClass::Anon, PageClass::Pinned),
            (PageClass::Unified, PageClass::Unified),
        ]
    );
    // A reserve nobody classed is refused by name, not counted nowhere.
    let odd = machine_with(vec![
        (OS_RESERVE.to_string(), 100),
        ("a new reserve".to_string(), 3),
    ]);
    let odd_plan = plan_with(&model, &odd, 4096, &NoKv, None).expect("plans");
    assert_eq!(
        HostNeed::of(&odd_plan, 0).classes(&odd_plan, loose),
        Err(NeedClassError::Reserve("a new reserve".to_string()))
    );
    // A folder's reserved part reads are the need's read-ahead, classed as a load's reads are.
    let bare = machine_with(vec![(OS_RESERVE.to_string(), 100)]);
    let folder = machine_with(vec![
        (OS_RESERVE.to_string(), 100),
        (PART_READS.to_string(), 17),
    ]);
    let bare_plan = plan_with(&model, &bare, 4096, &NoKv, None).expect("plans");
    let folder_plan = plan_with(&model, &folder, 4096, &NoKv, None).expect("plans");
    let reads = HostNeed::of(&folder_plan, 0);
    assert_eq!(reads.read_ahead, 17);
    let without = HostNeed::of(&bare_plan, 0)
        .classes(&bare_plan, loose)
        .expect("a plan's need");
    let with = reads
        .classes(&folder_plan, loose)
        .expect("a folder plan's need");
    assert_eq!(with.bytes(), reads.bytes());
    assert_eq!(
        with,
        NeedClasses {
            anon: without.anon + 17,
            ..without
        }
    );
    // The expert stream's staging is pinned for the load's life: it adds to the need beside the
    // part reads, not inside their peak.
    let streaming = machine_with(vec![
        (OS_RESERVE.to_string(), 100),
        (PART_READS.to_string(), 17),
        (XSTREAM_STAGING_RESERVE.to_string(), 23),
    ]);
    let streaming_plan = plan_with(&model, &streaming, 4096, &NoKv, None).expect("plans");
    let staged = HostNeed::of(&streaming_plan, 0);
    assert_eq!(staged.bytes(), reads.bytes() + 23);
    assert_eq!(
        staged.classes(&streaming_plan, loose),
        Ok(NeedClasses {
            pinned: with.pinned + 23,
            ..with
        })
    );
    // A need that is not the plan's.
    let other_os = HostNeed { os: 1, ..need };
    assert!(matches!(
        other_os.classes(&plan, loose),
        Err(NeedClassError::NotThePlans { .. })
    ));
    let thin = HostNeed {
        reserves: 100 + 40 + 30 + 20 + rows - 1,
        ..need
    };
    assert!(matches!(
        thin.classes(&plan, loose),
        Err(NeedClassError::NotThePlans { .. })
    ));
}

/// A card's bytes are every term the plan counts on it but the headroom.
#[test]
fn a_cards_bytes_sum_its_terms() {
    let model = layered(3);
    let machine = Machine {
        cards: vec![bytes_card("c", HEAD + 24 * EXPERT + 100_000, 0..3)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("plans");
    let c = &plan.cards[0];
    assert_eq!(
        c.bytes(),
        c.dense_bytes
            + c.expert_bytes
            + c.rounding_bytes
            + c.kv_bytes
            + c.stage_rows_bytes
            + c.scratch_bytes
            + c.context_bytes
            + c.reserve_bytes
    );
    assert!(c.bytes() > 0);
    let mut huge = c.clone();
    huge.dense_bytes = u64::MAX;
    huge.expert_bytes = 1;
    assert_eq!(huge.bytes(), u64::MAX, "a sum past u64 reads u64::MAX");
}

/// A unified machine's pool holds the card and the host together: the
/// model's bytes, wherever the fill put an expert, the card's cache,
/// set-aside and margin, the host's ring shadows, reserves and NVMe
/// arena and window, and the bytes the load holds beside the plan, each
/// counted once. A plan whose card and host each fit their own figure
/// plans clean at a pool of exactly that sum, as with no pool at all,
/// and is refused by name one byte under it, term by term; the host's
/// headroom is what the pool leaves past the card and the host's need.
/// The census reads the pool net of the context's own creation cost
/// ([`workstation::CONTEXT_SELF`]), which the card's context term counts
/// again: whole on a context term below that cost, and that cost alone
/// on the workstation's [`workstation::CONTEXT`], which is above it.
#[test]
fn a_unified_pool_holds_card_and_host_together() {
    struct Kv;
    impl KvBytes for Kv {
        fn layer_bytes(&self, _layer: usize, _ctx_max: u64) -> u64 {
            64
        }
        fn shadow_bytes(&self, _layer: usize, _ctx_max: u64) -> u64 {
            32
        }
    }
    let model = layered(3);
    let margin = 200;
    let (kv, shadows, pinned) = (3 * 64, 3 * 32, 50);
    // An NVMe split's arena and run-ahead window, set after the plan.
    let (arena, window) = (7, 5);
    let plan_at = |context: u64, experts: u64, pool: Option<u64>, beside: u64| {
        let usable = HEAD + experts * EXPERT + context + 300 + 100 + margin + kv;
        let machine = Machine {
            cards: vec![Card {
                context_bytes: context,
                scratch_bytes: 300,
                margin_bytes: margin,
                free_bytes: Some(usable - context.min(workstation::CONTEXT_SELF)),
                reserves: vec![("draft".to_string(), 100)],
                ..bytes_card("pool", usable, 0..3)
            }],
            tiers: Vec::new(),
            // On a unified machine the host is the pool
            // (`workstation::host_of`).
            nodes: vec![Node {
                host: Host {
                    usable_bytes: pool.unwrap_or(1 << 40),
                    reserves: vec![("pinned".to_string(), pinned)],
                },
                unified: pool.map(|bytes| UnifiedPool { bytes }),
            }],
        };
        let mut plan = plan_with(&model, &machine, 4096, &Kv, None).expect("plans");
        assert!(plan.host.experts > 0, "the host holds experts");
        let headroom = plan.host.headroom_bytes;
        plan.host.nvme_arena_bytes = arena;
        plan.host.reserve_bytes += window;
        (headroom, plan.violations_beside(beside))
    };
    let model_bytes: u64 = model.tensors.iter().map(|t| t.file_bytes).sum();
    let cases = [1_000, workstation::CONTEXT]
        .into_iter()
        .flat_map(|c| [(c, 6, 0), (c, 12, 9)]);
    for (context, experts, beside) in cases {
        let on_card = HEAD + experts * EXPERT;
        let cards = on_card + kv + context + 300 + 100 + margin;
        let host = model_bytes - on_card + shadows + pinned + arena + window;
        let total = cards + host + beside;
        let own = context.min(workstation::CONTEXT_SELF);
        let (_, v) = plan_at(context, experts, None, beside);
        assert!(
            v.is_empty(),
            "context {context}, {experts} experts on the card, no pool: {v:?}"
        );
        let (headroom, v) = plan_at(context, experts, Some(total - own), beside);
        assert!(
            v.is_empty(),
            "context {context}, {experts} experts on the card, a pool of the sum: {v:?}"
        );
        assert_eq!(
            headroom,
            i128::from(arena + window + beside),
            "context {context}, {experts} experts on the card: the pool past the plan's own \
             terms"
        );
        match plan_at(context, experts, Some(total - own - 1), beside)
            .1
            .as_slice()
        {
            [v @ Violation::PoolOver { .. }] => assert_eq!(
                v.to_string(),
                format!(
                    "unified pool: the cards with their margins {cards} B + the host's need \
                     {host} B + {beside} B the load holds beside the plan = {total} B pass \
                     the pool {} B by 1 B",
                    total - 1
                )
            ),
            v => {
                panic!("context {context}, {experts} experts on the card: {v:?}, not one PoolOver")
            }
        }
    }
}

/// A need past the room its plan was made at is refused there, naming
/// what decided that room: a given room under the machine's caps as
/// given, the unified pool under a reading's plan room as the machine's
/// figure, and the machine's figure for a plan made without a room.
#[test]
fn a_need_is_held_to_the_room_its_plan_was_made_at() {
    use workstation::{HOST_DRIFT, HostNeed, HostRead};
    let model = layered(3);
    let pool = 1 << 30;
    let machine = Machine {
        cards: vec![Card {
            context_bytes: 1_000,
            free_bytes: Some(pool),
            ..bytes_card("pool", 2 * pool, 0..3)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: Some(UnifiedPool { bytes: pool }),
        }],
    };
    let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).unwrap();
    let figure = HostNeed::machine_room(machine.host());
    let base = HostNeed::of(&plan, 0);
    let refused = |plan: &Plan<'_>, room: u64| {
        assert!(base.bytes() < room, "{} >= {room}", base.bytes());
        let fits = HostNeed {
            read_ahead: room - base.bytes(),
            ..base
        };
        assert!(fits.check_planned(plan).is_ok());
        let over = HostNeed {
            read_ahead: fits.read_ahead + 1,
            ..base
        };
        let short = over.check_planned(plan).unwrap_err();
        assert_eq!(short.available, room);
        (over, *short.read)
    };
    assert_eq!(refused(&plan, figure).1, HostRead::Machine);
    let room = plan.plan_at(&(pool - 1000, HostRead::Given));
    assert_eq!(room, pool - 1000);
    let (over, read) = refused(&plan, room);
    assert_eq!(read, HostRead::Given);
    assert!(over.check(&(figure, HostRead::Machine)).is_ok());
    let room = plan.plan_at(&(pool + HOST_DRIFT + 1000, HostRead::MemAvailable));
    assert_eq!(room, pool);
    assert_eq!(refused(&plan, room).1, HostRead::Machine);
}

#[test]
fn plan_room_uses_the_host_and_unified_limits_once() {
    use workstation::{HOST_DRIFT, HostNeed, HostRead, OS_RESERVE};
    let model = layered(3);
    let pool = 1 << 30;
    for context in [1_000, workstation::CONTEXT] {
        for host_room in [pool - 100, pool + 100] {
            let machine = Machine {
                cards: vec![Card {
                    context_bytes: context,
                    free_bytes: Some(pool),
                    ..bytes_card("pool", 2 * pool, 0..3)
                }],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: host_room + 64,
                        reserves: vec![(OS_RESERVE.into(), 64)],
                    },
                    unified: Some(UnifiedPool { bytes: pool }),
                }],
            };
            let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).unwrap();
            for read in [HostRead::Given, HostRead::MemAvailable] {
                for room in [pool - 200, pool + HOST_DRIFT + 200] {
                    let reading = (room, read.clone());
                    let want = workstation::plan_room(&reading).min(host_room).min(pool);
                    assert_eq!(plan.plan_at(&reading), want);
                    let by = if want < workstation::plan_room(&reading) {
                        HostRead::Machine
                    } else {
                        read.clone()
                    };
                    assert_eq!(plan.host.room, Some((room, (want, by))));
                    if read == HostRead::Given {
                        assert_eq!(
                            plan.host.headroom_bytes,
                            HostNeed::of(&plan, 0).headroom(want)
                        );
                    }
                }
            }
        }
    }
}
