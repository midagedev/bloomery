use super::*;

#[test]
fn machine_accessors_use_the_first_node() {
    let nodes = [11, 22].map(|bytes| Node {
        host: Host {
            usable_bytes: bytes,
            ..host()
        },
        unified: Some(UnifiedPool { bytes: bytes + 1 }),
    });
    let mut machine = Machine {
        cards: vec![],
        tiers: vec![],
        nodes: nodes.to_vec(),
    };
    assert_eq!(machine.host(), &nodes[0].host);
    assert_eq!(machine.unified(), nodes[0].unified);
    machine.host_mut().usable_bytes = 33;
    *machine.unified_mut() = Some(UnifiedPool { bytes: 44 });
    assert_eq!(machine.nodes[0].host.usable_bytes, 33);
    assert_eq!(machine.nodes[0].unified, Some(UnifiedPool { bytes: 44 }));
    assert_eq!(machine.nodes[1], nodes[1]);
    machine.nodes.clear();
    for mutable in [false, true] {
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if mutable {
                machine.host_mut();
            } else {
                let _ = machine.host();
            }
        }))
        .unwrap_err();
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied());
        assert_eq!(message, Some("a machine lists this process's node first"));
    }
}

/// Each format's word names it back, every word apart; another word
/// names none.
#[test]
fn a_formats_word_names_it() {
    for (word, format) in [
        ("kquant", Format::Card(CardFormat::KQuant)),
        ("q5_0", Format::Card(CardFormat::Q5_0)),
        ("q5_1", Format::Card(CardFormat::Q5_1)),
        ("q8_0_planes", Format::Card(CardFormat::Q8_0Planes)),
        ("f32", Format::Card(CardFormat::F32)),
        ("bf16_as_f32", Format::Card(CardFormat::Bf16AsF32)),
        ("bf16_raw", Format::Card(CardFormat::Bf16Raw)),
        ("mxfp8_planes", Format::Card(CardFormat::Mxfp8Planes)),
        ("fp8_b128_planes", Format::Card(CardFormat::Fp8B128Planes)),
        ("host_file", Format::HostFile),
        ("nvme_file", Format::NvmeFile),
        ("unused", Format::Unused),
    ] {
        assert_eq!(Format::of_word(word), Some(format), "{word}");
    }
    assert_eq!(Format::of_word("q4_k"), None);
    assert_eq!(Format::of_word(""), None);
}

/// The floor is the larger of a call's `k × columns + lane_slots` and the
/// batched union's two batches and the lane's slots, and the union's term
/// counts the lane's slots whether or not a lane runs: the model side's
/// refusal of a union under it (`TierError::BatchFloor`) never reads the
/// plan's lane slots. A lane's slots are its four open reads and one
/// victim claim.
#[test]
fn the_floor_carries_the_batched_unions_pins() {
    assert_eq!(ARENA_LANE_SLOTS, 4 + 1);
    let union = 2 * UNION_BATCH_IDS + ARENA_LANE_SLOTS;
    let slot_set = 1 << 20;
    // The routed width a call names, one column, a lane: under the union.
    let narrow = ArenaFloor::new(10, 1, ARENA_LANE_SLOTS, slot_set).expect("a floor");
    assert_eq!((narrow.union, narrow.slots), (union, union));
    assert_eq!(narrow.bytes, union * slot_set);
    // No lane runs: the union's term still counts the lane's slots.
    let no_lane = ArenaFloor::new(10, 1, 0, slot_set).expect("a floor");
    assert_eq!(no_lane.slots, union);
    // A call wider than the union's pins: its own term.
    let wide = ArenaFloor::new(16, 3, ARENA_LANE_SLOTS, slot_set).expect("a floor");
    assert_eq!((wide.union, wide.slots), (union, 16 * 3 + ARENA_LANE_SLOTS));
    assert_eq!(wide.bytes, wide.slots * slot_set);
    let text = narrow.to_string();
    assert!(
        text.contains("37 slots a paged layer") && text.contains("37 for a union in batches"),
        "{text}"
    );
    // Past u64 is refused by name, not wrapped.
    assert!(ArenaFloor::new(u64::MAX, 2, 0, 1).is_err());
}

/// PIN(2026-10-12): a lane holds its open reads and its victim claim at
/// once, so the union's floor is `2 × UNION_BATCH_IDS + LANE_THREADS + 1`
/// slots (two batches pinned, four lane threads each pinning the id they
/// stage, one victim claim): 37. The earlier one-slot lane term, 33,
/// refused a claim beside four pins and two pinned batches.
#[test]
fn a_lane_holds_its_open_reads_and_its_victim_claim() {
    let lane_threads = 4;
    assert_eq!(ARENA_LANE_SLOTS, lane_threads + 1);
    let floor = ArenaFloor::new(10, 1, ARENA_LANE_SLOTS, 1 << 20).expect("a floor");
    assert_eq!(floor.slots, 2 * UNION_BATCH_IDS + lane_threads + 1);
    assert_eq!(floor.slots, 37);
    // The call's term moves with the lane's slots too: k ids and the lane.
    let call = ArenaFloor::new(40, 1, ARENA_LANE_SLOTS, 1).expect("a floor");
    assert_eq!(call.slots, 40 + lane_threads + 1);
}

/// [`plan_in_pool`] on a planner whose floor passes the room by
/// `short(budget)` B: on a unified machine with no given budget each
/// round lowers the budget by the shortfall the last refusal named, at
/// least a granule, and the plan that fits is marked as the pool's, so
/// the host's refusal does not name a lever no one set. A refusal other
/// than the floor (on a machine with no node too), a discrete card, a
/// given budget and a shortfall past the card's whole budget are the
/// first plan's result; a lowered budget refused for another reason, a
/// budget that runs out and a floor unmet after [`POOL_ROUNDS`] rounds
/// are refused by name, with the rounds, the budget and the last refusal.
#[test]
fn a_unified_pool_lowers_the_card_budget_to_the_tiers_floor() {
    const ROOM: u64 = 100 << 30;
    const POOL: u64 = 64 << 30;
    let granule = workstation::GRANULE.get();
    let model = ModelTensors {
        tensors: Vec::new(),
        layers: 1,
        experts: 1,
        experts_used: 1,
        bf16_words: |_| false,
    };
    let pool = |unified: bool| {
        let mut c = card("GB10", 0..1, true, true);
        c.usable_bytes = 120 << 30;
        c.free_bytes = Some(POOL);
        Machine {
            cards: vec![c],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: POOL,
                    reserves: Vec::new(),
                },
                unified: unified.then_some(UnifiedPool { bytes: POOL }),
            }],
        }
    };
    let floor = |short: u64| PlacementError::HostRoomFloor {
        room: ROOM,
        floor: ROOM + short,
        base: 0,
        layer: 0,
        need: 0,
        arena: None,
    };
    // Every budget the planner is called with, and its result: refused
    // by `short(budget)` B over the room, or for another reason at and
    // under `other_at`.
    let run = |m: &Machine, levers: &PlanLevers, short: &dyn Fn(u64) -> u64, other_at: u64| {
        let mut calls = Vec::new();
        let got = plan_in_pool(m, levers, |l| {
            let budget = capped(&m.cards[0], l.card_budget_bytes.every());
            calls.push(l.card_budget_bytes.every());
            if budget <= other_at {
                return Err(PlacementError::HostRoom("other".to_string()));
            }
            match short(budget) {
                0 => Ok(budget_plan(&model, m, budget)),
                s => Err(floor(s)),
            }
        });
        (
            got.map(|p| (p.card_budget.every(), p.budget_from_pool)),
            calls,
        )
    };
    let (on, off) = (pool(true), pool(false));
    let unset = PlanLevers::default();
    let full = capped(&on.cards[0], None);
    assert_eq!(full, POOL + on.cards[0].context_in_free());

    // A shortfall that falls by half the bytes the budget gives up: the
    // first round lowers by it, the next by the half left, and a
    // shortfall under a granule by a granule.
    let fit = full - (8 << 30);
    let halves = |b: u64| b.saturating_sub(fit).div_ceil(2);
    let (got, calls) = run(&on, &unset, &halves, 0);
    let mut want = vec![None];
    let mut b = full;
    while halves(b) > 0 {
        b -= halves(b).max(granule);
        want.push(Some(b));
    }
    assert_eq!(calls, want);
    assert!(calls.len() > 3, "{calls:?}");
    assert_eq!(got.expect("the lowered plan"), (Some(b), true));
    let need = |from_pool| {
        let mut p = budget_plan(&model, &on, b);
        p.budget_from_pool = from_pool;
        workstation::HostNeed::of(&p, 0).card_budget
    };
    assert!(!need(true) && need(false));

    // The first plan's result: a discrete card, a given budget, a
    // shortfall past the card's whole budget.
    let given = PlanLevers {
        card_budget_bytes: Some(96 << 30).into(),
    };
    for (m, levers, short) in [(&off, &unset, 1), (&on, &given, 1), (&on, &unset, full + 1)] {
        let (got, calls) = run(m, levers, &|_| short, 0);
        assert!(matches!(got, Err(PlacementError::HostRoomFloor { .. })));
        assert_eq!(calls.len(), 1);
    }
    // A refusal before the floor is the first plan's result, even on a
    // machine with no node to read.
    let none = Machine {
        nodes: Vec::new(),
        ..pool(true)
    };
    let got = plan_in_pool(&none, &unset, |_| {
        Err(PlacementError::Nodes { nodes: 0, tiers: 0 })
    });
    assert!(matches!(got, Err(PlacementError::Nodes { nodes: 0, .. })));

    // Refused by name: another refusal of a lowered budget, a budget
    // that runs out, a floor still unmet after the rounds.
    let pooled = |got: Result<_, PlacementError>| match got {
        Err(PlacementError::PoolBudget {
            rounds,
            budget,
            last,
        }) => (rounds, budget, *last),
        other => panic!("not refused by name: {other:?}"),
    };
    let (rounds, budget, last) = pooled(run(&on, &unset, &|_| 1 << 30, full - (1 << 30)).0);
    assert_eq!((rounds, budget), (1, full - (1 << 30)));
    assert!(matches!(last, PlacementError::HostRoom(_)));
    let (rounds, budget, last) = pooled(run(&on, &unset, &|b| b / 2 + 1, 0).0);
    assert!(rounds > 1 && budget < full, "{rounds} {budget}");
    assert!(matches!(last, PlacementError::HostRoomFloor { .. }));
    let (got, calls) = run(&on, &unset, &|_| 1, 0);
    let (rounds, budget, last) = pooled(got);
    assert_eq!(rounds, POOL_ROUNDS);
    assert_eq!(budget, full - u64::from(POOL_ROUNDS) * granule);
    assert_eq!(calls.len(), 1 + POOL_ROUNDS as usize);
    assert!(matches!(last, PlacementError::HostRoomFloor { .. }));
    let text = PlacementError::PoolBudget {
        rounds,
        budget,
        last: Box::new(last),
    }
    .to_string();
    assert!(text.contains("lowered 64 times"), "{text}");
}

/// The stage map's token-embedding rule: none, or the one card that runs
/// layer 0; a card that does not run it, or a second card, is refused.
#[test]
fn token_embedding_is_the_layer_0_card_alone() {
    let map = |embed: [bool; 2]| {
        Stages::new(
            8,
            &[
                card("card0", 0..5, false, embed[0]),
                card("card1", 5..8, true, embed[1]),
            ],
        )
    };
    let refused = |embed: [bool; 2]| match map(embed) {
        Err(PlacementError::Stages(msg)) => msg,
        Err(e) => panic!("{embed:?}: {e}, want a stage-map refusal"),
        Ok(_) => panic!("{embed:?}: the stage map was accepted"),
    };
    let held = |embed: [bool; 2]| {
        map(embed)
            .unwrap_or_else(|e| panic!("{embed:?}: {e}"))
            .embedding
    };
    assert_eq!(held([false, false]), None);
    assert_eq!(held([true, false]), Some(0));
    assert_eq!(
        refused([false, true]),
        "card card1 holds the token embedding but does not run layer 0"
    );
    assert_eq!(
        refused([true, true]),
        "2 cards hold the token embedding, not one"
    );
}

/// A BF16 head the loader places in MXFP8 planes is the one tensor the plan
/// names as requantized: a head off the planes' 128-value block keeps its
/// words, a Q8_0 head its own planes, a BF16 embedding is widened, and none
/// of those changes the file's values. Mutant: `requantizes` true for every
/// card format other than the file type's own, which lists the words-kept
/// head as well.
#[test]
fn a_bf16_head_in_mxfp8_planes_is_named_as_requantized() {
    let tensor = |name: &str, role, ty, k: u64| ModelTensor {
        name: name.to_string(),
        shard: 0,
        layer: None,
        role,
        ty,
        dims: vec![k, 16],
        file_bytes: 0,
        gathered_rows: (role == Role::TokenEmbedding).then_some(1),
    };
    let model = |tensors: Vec<ModelTensor>| ModelTensors {
        tensors,
        layers: 0,
        experts: 8,
        experts_used: 2,
        bf16_words: |_| false,
    };
    let embed = tensor(
        "token_embd.weight",
        Role::TokenEmbedding,
        GgmlType::BF16,
        256,
    );
    let planes = model(vec![
        embed.clone(),
        tensor(HEAD_TENSOR, Role::Head, GgmlType::BF16, 256),
    ]);
    let listed = planes.requantized();
    assert_eq!(
        listed,
        vec![Requantized {
            name: HEAD_TENSOR.to_string(),
            file: GgmlType::BF16,
            card: CardFormat::Mxfp8Planes,
        }]
    );
    assert_eq!(
        listed[0].to_string(),
        format!("output.weight:{}:mxfp8_planes", GgmlType::BF16)
    );
    for (ty, k) in [(GgmlType::BF16, 320), (GgmlType::Q8_0, 256)] {
        let kept = model(vec![embed.clone(), tensor(HEAD_TENSOR, Role::Head, ty, k)]);
        assert!(kept.requantized().is_empty(), "a {ty} head of {k} values");
    }
}

/// The footprint the expert rule asks once an expert is the allocator's replay of every
/// buffer in order, whatever counts it is asked at and in whatever order: fixed tensors
/// below and above a granule between stacks whose buffers cross it as their counts grow.
#[test]
fn a_footprint_is_the_replay_of_every_buffer_at_any_counts() {
    let tensor = |name: &str, layer, role, ty: GgmlType, dims: Vec<u64>| ModelTensor {
        name: name.into(),
        shard: 0,
        layer,
        role,
        ty,
        dims,
        file_bytes: 0,
        gathered_rows: None,
    };
    let (layers, granule) = (3, NonZeroU64::new(4096).expect("nonzero"));
    let mut tensors = Vec::new();
    for l in 0..layers {
        tensors.push(tensor(
            "norm",
            Some(l),
            Role::Attention,
            GgmlType::F32,
            vec![256],
        ));
        tensors.push(tensor(
            "big",
            Some(l),
            Role::Attention,
            GgmlType::Q8_0,
            vec![256, 20],
        ));
        tensors.push(tensor(
            "small",
            Some(l),
            Role::Attention,
            GgmlType::Q8_0,
            vec![256, 3],
        ));
        tensors.push(tensor(
            "stack",
            Some(l),
            Role::RoutedExperts,
            GgmlType::Q4_K,
            vec![256, 8, 9],
        ));
    }
    let uploads: Vec<_> = tensors
        .iter()
        .map(|t| {
            let format = CardFormat::of(t.ty).expect("a card format");
            let held = match t.role {
                Role::RoutedExperts => Held::Experts {
                    layer: t.layer.expect("a layer"),
                    rows_per_expert: 8,
                },
                _ => Held::Rows(rows_of(t)),
            };
            (t, format, held)
        })
        .collect();
    let replay = |n_l: &[u64]| {
        let buffers = uploads.iter().flat_map(|&(t, format, held)| {
            let rows = match held {
                Held::Rows(rows) => rows,
                Held::Experts {
                    layer,
                    rows_per_expert,
                } => n_l[layer] * rows_per_expert,
            };
            if rows == 0 {
                Vec::new()
            } else {
                card_buffers(t, format, rows).expect("buffers")
            }
        });
        allocator_bytes(granule, buffers)
    };
    let footprint = Footprint::new(granule, &uploads);
    let mut n_l = vec![0u64; layers];
    let mut seed = 7u64;
    for ask in 0..400 {
        if ask < 27 {
            n_l[ask % layers] += 1;
        } else {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            n_l[(seed >> 33) as usize % layers] = (seed >> 40) % 10;
        }
        assert_eq!(
            footprint.at(&n_l).expect("a footprint"),
            replay(&n_l),
            "{n_l:?}"
        );
    }
}

#[test]
fn only_card_uploads_require_the_card_layout() {
    for width in [64, 160] {
        for (role, embedding, device) in [
            (Role::EngramTable, false, Device::Nvme),
            (Role::HashTable, false, Device::Host),
            (Role::TokenEmbedding, false, Device::Host),
            (Role::Unread, false, Device::Unused),
            (Role::Unused, false, Device::Unused),
            (Role::Attention, false, Device::Card(0)),
            (Role::TokenEmbedding, true, Device::Card(0)),
        ] {
            let mut model = layered(1);
            model.tensors[0] = ModelTensor {
                name: "rows.weight".into(),
                shard: 0,
                layer: (role != Role::TokenEmbedding).then_some(0),
                role,
                ty: GgmlType::NVFP4_ROWS,
                dims: vec![width, 5],
                file_bytes: width / 16 * 9 * 5,
                gathered_rows: Some(1),
            };
            let machine = Machine {
                cards: vec![Card {
                    usable_bytes: 1 << 30,
                    ..card("card", 0..1, true, embedding)
                }],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: host(),
                    unified: None,
                }],
            };
            let planned = plan_with(&model, &machine, 4096, &NoKv, None);
            if width == 160 && device == Device::Card(0) {
                let error = planned
                    .expect_err("the upload needs whole card blocks")
                    .to_string();
                assert!(error.contains("rows.weight"), "{error}");
                assert!(error.contains("Nvfp4Planes upload refuses"), "{error}");
            } else {
                let plan = planned.unwrap_or_else(|e| panic!("{role:?} {width}: {e}"));
                assert_eq!(plan.rows[0].segments[0].device, device);
            }
        }
    }
    // A routed stack the host planner keeps whole on the host takes no
    // card layout, while the expert rule's card segments of it are refused.
    let mut model = layered(1);
    model.tensors[1] = ModelTensor {
        ty: GgmlType::NVFP4_ROWS,
        dims: vec![160, 4, 8],
        file_bytes: 160 / 16 * 9 * 4 * 8,
        ..model.tensors[1].clone()
    };
    let machine = Machine {
        cards: vec![Card {
            usable_bytes: 1 << 30,
            ..card("card", 0..1, true, true)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let plan = plan_host_routed(&model, &machine, 4096, &NoKv, &PlanLevers::default())
        .unwrap_or_else(|e| panic!("host-routed 160: {e}"));
    let devices: Vec<Device> = plan.rows[1].segments.iter().map(|s| s.device).collect();
    assert_eq!(devices, vec![Device::Host]);
    let error = plan_with(&model, &machine, 4096, &NoKv, None)
        .expect_err("card experts need whole card blocks")
        .to_string();
    assert!(error.contains("experts 0"), "{error}");
    assert!(error.contains("Nvfp4Planes upload refuses"), "{error}");
}

/// A model of 4 layers with every role's bf16 tensor (an attention
/// matrix, a hyper-connection, a shared expert and its one-row gate, a
/// router, an FFN norm, the head and the token embedding, 256-value rows)
/// beside a routed q4_K stack, its readers' choice `bf16_words`, planned
/// on `n` cards of `4 / n` layers each: every bf16 tensor's card segment
/// is on the card that runs it, in the format `ModelTensors::card_format`
/// names, at its format's bytes. With `bf16_words` the tensors of the
/// dense roles and the router keep their words (the file's bytes); the
/// norm, and every tensor without it, are f32 values (twice the file's
/// bytes): the plan of a model that does not set it is the one it was.
#[test]
fn a_bf16_matrix_keeps_its_words_where_its_readers_read_them() {
    let bf16 = |name: String, layer, role, rows: &[u64]| {
        let mut dims = vec![256];
        dims.extend_from_slice(rows);
        ModelTensor {
            name,
            shard: 0,
            layer,
            role,
            ty: GgmlType::BF16,
            file_bytes: 2 * dims.iter().product::<u64>(),
            dims,
            gathered_rows: (role == Role::TokenEmbedding).then_some(1),
        }
    };
    assert_eq!(
        file_card_format(HEAD_TENSOR, GgmlType::BF16, 320),
        Some(CardFormat::Bf16Raw)
    );
    assert_eq!(
        file_card_format(HEAD_TENSOR, GgmlType::Q8_0, 256),
        Some(CardFormat::Q8_0Planes)
    );
    assert_eq!(
        file_card_format("token_embd.weight", GgmlType::BF16, 256),
        CardFormat::of(GgmlType::BF16)
    );
    let raw = [
        Role::Attention,
        Role::HyperConnection,
        Role::SharedExpert,
        Role::Head,
        Role::TokenEmbedding,
        Role::Router,
    ];
    for bf16_words in [false, true] {
        for n in [1usize, 2, 4] {
            let mut model = layered(4);
            model.bf16_words = if bf16_words { |_| true } else { |_| false };
            for l in 0..4 {
                for (what, role, rows) in [
                    ("attn", Role::Attention, &[16][..]),
                    ("hc", Role::HyperConnection, &[4]),
                    ("shexp", Role::SharedExpert, &[8]),
                    ("shexp gate", Role::SharedExpert, &[]),
                    ("router", Role::Router, &[9]),
                    ("norm", Role::FfnNorm, &[]),
                ] {
                    model
                        .tensors
                        .push(bf16(format!("{what} {l}"), Some(l), role, rows));
                }
            }
            for (name, role) in [
                (HEAD_TENSOR, Role::Head),
                ("output_hc_up.weight", Role::Head),
                ("bf16 embd", Role::TokenEmbedding),
            ] {
                model.tensors.push(bf16(name.into(), None, role, &[16]));
            }
            let per = 4 / n;
            let cards = (0..n)
                .map(|c| Card {
                    usable_bytes: 1 << 30,
                    ..card(
                        &format!("card{c}"),
                        c * per..(c + 1) * per,
                        c == n - 1,
                        c == 0,
                    )
                })
                .collect();
            let machine = Machine {
                cards,
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: host(),
                    unified: None,
                }],
            };
            let plan = plan_with(&model, &machine, 4096, &NoKv, None)
                .unwrap_or_else(|e| panic!("n={n} words={bf16_words}: {e}"));
            let mut seen = 0;
            for r in &plan.rows {
                let t = &model.tensors[r.tensor];
                if t.ty != GgmlType::BF16 {
                    continue;
                }
                seen += 1;
                let (format, bytes) = if t.name == HEAD_TENSOR {
                    (
                        CardFormat::Mxfp8Planes,
                        t.file_bytes / 2 + t.file_bytes / 64,
                    )
                } else if bf16_words && raw.contains(&t.role) {
                    (CardFormat::Bf16Raw, t.file_bytes)
                } else {
                    (CardFormat::Bf16AsF32, 2 * t.file_bytes)
                };
                let card = match (t.layer, t.role) {
                    (Some(l), _) => l / per,
                    (None, Role::Head) => n - 1,
                    (None, _) => 0,
                };
                assert_eq!(model.card_format(t), Some(format), "{}", t.name);
                let got: Vec<_> = r
                    .segments
                    .iter()
                    .map(|s| (s.device, s.format, s.experts.is_none(), s.resident_bytes))
                    .collect();
                assert_eq!(
                    got,
                    [(Device::Card(card), Format::Card(format), true, bytes)],
                    "n={n} words={bf16_words} {}",
                    t.name
                );
            }
            assert_eq!(seen, 4 * 6 + 3, "n={n}");
        }
    }
}

#[test]
fn card_sums_preserve_the_u64_boundary() {
    let boundary = Card {
        context_bytes: 2,
        scratch_bytes: 3,
        reserves: vec![("beside".into(), u64::MAX - 5)],
        ..bytes_card("boundary", u64::MAX, 0..1)
    };
    assert_eq!(boundary.reserve_bytes(), Some(u64::MAX - 5));
    assert_eq!(boundary.set_aside_bytes(), Some(u64::MAX));
    assert_eq!(boundary.floor_bytes(0, 0, 0), Some(u64::MAX));
    for (dense, kv, past) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
        assert_eq!(boundary.floor_bytes(dense, kv, past), None);
    }
    for (context, scratch, reserves, reserve_sum) in [
        (0, 0, vec![u64::MAX, 1], None),
        (1, 0, vec![u64::MAX], Some(u64::MAX)),
        (0, 1, vec![u64::MAX], Some(u64::MAX)),
        (u64::MAX, 1, vec![], Some(0)),
    ] {
        let card = Card {
            context_bytes: context,
            scratch_bytes: scratch,
            reserves: reserves.into_iter().map(|b| ("beside".into(), b)).collect(),
            ..boundary.clone()
        };
        assert_eq!(card.reserve_bytes(), reserve_sum);
        assert_eq!(card.set_aside_bytes(), None);
        assert_eq!(card.floor_bytes(0, 0, 0), None);
    }
}

#[test]
fn host_reserve_sum_preserves_the_u64_boundary() {
    let host = |reserves: &[u64]| Host {
        usable_bytes: u64::MAX,
        reserves: reserves.iter().map(|&b| ("beside".into(), b)).collect(),
    };
    assert_eq!(host(&[]).reserve_bytes(), Some(0));
    assert_eq!(host(&[u64::MAX]).reserve_bytes(), Some(u64::MAX));
    assert_eq!(host(&[u64::MAX - 5, 5]).reserve_bytes(), Some(u64::MAX));
    assert_eq!(host(&[u64::MAX, 1]).reserve_bytes(), None);
    assert_eq!(host(&[1 << 63, 1 << 63]).reserve_bytes(), None);
}

/// A host whose reserves sum past u64 bytes wraps to a small number
/// under unchecked addition (release) or panics (debug): the plan then
/// passes the host bound or never returns. It is a named refusal.
#[test]
fn overflowing_host_reserves_are_a_named_placement_refusal() {
    let model = layered(3);
    for reserves in [
        vec![u64::MAX, 1],
        vec![1 << 63, 1 << 63],
        vec![u64::MAX - 5, 3, 3],
    ] {
        let machine = Machine {
            cards: vec![bytes_card("stage", HEAD + 12 * EXPERT, 0..3)],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: 1 << 40,
                    reserves: reserves.iter().map(|&b| ("beside".into(), b)).collect(),
                },
                unified: None,
            }],
        };
        assert!(
            matches!(
                plan_with(&model, &machine, 4096, &NoKv, None),
                Err(PlacementError::HostBytesPastU64)
            ),
            "{reserves:?}"
        );
    }
}

/// Reserves that fit u64 alone but not beside the plan's host experts:
/// the host's need passes every room, so the plan is a `HostOver`, not a
/// wrapped small need that fits.
#[test]
fn a_host_need_past_u64_is_a_host_over() {
    let model = layered(3);
    let machine = Machine {
        cards: vec![bytes_card("stage", HEAD + 12 * EXPERT, 0..3)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: Host {
                usable_bytes: 1 << 40,
                reserves: vec![("beside".into(), u64::MAX - 100)],
            },
            unified: None,
        }],
    };
    let plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
    assert!(plan.host.expert_bytes > 100, "the host holds experts");
    let over: Vec<_> = plan
        .violations()
        .into_iter()
        .filter(|v| matches!(v, Violation::HostOver { .. }))
        .collect();
    assert_eq!(
        over,
        vec![Violation::HostOver {
            total: u64::MAX,
            usable: 1 << 40
        }]
    );
    assert!(
        workstation::HostNeed::of(&plan, 0)
            .check(&(u64::MAX, workstation::HostRead::Given))
            .is_err()
    );
}

/// The split's room arithmetic is exact where the host's need passes u64
/// bytes: the host keeps what the room leaves past the exact need and the
/// window, within one expert, and the rest moves to the NVMe tier. An
/// unchecked `need + window − room` wraps in release (a small, wrong
/// `keep`) and panics in debug.
#[test]
fn an_nvme_split_of_a_need_past_u64_is_exact() {
    let (model, machine, expert) = host_need_past_u64();
    let w = 8 * expert;
    let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
    let held = plan.host.expert_bytes;
    assert_eq!(held, 8 * w, "every expert is on the host");
    let need = workstation::HostNeed::of(&plan, 0);
    assert!(
        need.exact(true) > u128::from(u64::MAX),
        "the need passes u64"
    );
    let room = u64::MAX - w;
    expert_nvme_tier(&mut plan, room, None).expect("the split");
    let keep = u128::from(held) - (need.exact(true) + u128::from(2 * w) - u128::from(room));
    let kept = u128::from(plan.host.expert_bytes);
    assert!(
        kept <= keep && kept + u128::from(expert) > keep,
        "the host keeps {kept} B of {keep}"
    );
    assert_eq!(plan.host.nvme_expert_bytes, held - plan.host.expert_bytes);
    assert_eq!(plan.nvme_bytes, plan.host.nvme_expert_bytes);
}

/// A room of `u64::MAX` does not hold a need past u64 bytes: the split
/// runs. `HostNeed::bytes` reads `u64::MAX` for that need, which compared
/// to the room would leave the plan whole.
#[test]
fn a_room_of_u64_max_does_not_hold_a_need_past_u64() {
    let (model, machine, _) = host_need_past_u64();
    let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
    let held = plan.host.expert_bytes;
    assert_eq!(workstation::HostNeed::of(&plan, 0).bytes(), u64::MAX);
    expert_nvme_tier(&mut plan, u64::MAX, None).expect("the split");
    assert!(
        plan.host.expert_bytes < held && plan.host.nvme_expert_bytes > 0,
        "the split moved experts to the NVMe tier"
    );
}

#[test]
fn overflowing_card_sums_are_named_placement_refusals() {
    let model = layered(1);
    for (context, scratch, reserves) in [
        (0, 0, vec![u64::MAX, 1]),
        (1, 0, vec![u64::MAX]),
        (0, 1, vec![u64::MAX]),
        (u64::MAX, 1, vec![]),
    ] {
        for free_bytes in [None, Some(u64::MAX - 1)] {
            let machine = Machine {
                cards: vec![Card {
                    context_bytes: context,
                    scratch_bytes: scratch,
                    reserves: reserves.iter().map(|&b| ("beside".into(), b)).collect(),
                    free_bytes,
                    ..bytes_card("overflow", u64::MAX, 0..1)
                }],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: host(),
                    unified: None,
                }],
            };
            for budget in [None, Some(u64::MAX)] {
                assert!(matches!(
                    plan_with(&model, &machine, 4096, &NoKv, budget),
                    Err(PlacementError::CardBytesPastU64 { card }) if card == "overflow"
                ));
            }
            assert!(matches!(
                whole_need(&model, &machine.cards[0], 0, WholeLoad::Unchecked),
                Err(PlacementError::CardBytesPastU64 { card }) if card == "overflow"
            ));
        }
    }
    // The set-aside itself fits; adding the actual uploads does not.
    let machine = Machine {
        cards: vec![Card {
            reserves: vec![("beside".into(), u64::MAX)],
            ..bytes_card("load-overflow", u64::MAX, 0..1)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    assert!(matches!(
        plan_with(&model, &machine, 4096, &NoKv, None),
        Err(PlacementError::CardBytesPastU64 { card }) if card == "load-overflow"
    ));
    assert!(matches!(
        whole_need(&model, &machine.cards[0], 0, WholeLoad::Unchecked),
        Err(PlacementError::CardBytesPastU64 { card }) if card == "load-overflow"
    ));
}

#[test]
fn pooled_card_sums_are_checked_only_for_shared_memory() {
    let model = layered(2);
    for unified in [None, Some(UnifiedPool { bytes: u64::MAX })] {
        let machine = Machine {
            cards: (0..2)
                .map(|i| Card {
                    head: i == 1,
                    reserves: vec![("resident".into(), 1 << 63)],
                    ..bytes_card(&format!("stage{i}"), u64::MAX, i..i + 1)
                })
                .collect(),
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified,
            }],
        };
        let result = plan_with(&model, &machine, 4096, &NoKv, None);
        if unified.is_some() {
            assert!(matches!(
                result,
                Err(PlacementError::CardBytesPastU64 { card }) if card == "stage1"
            ));
        } else {
            let plan = result.expect("separate card memories need no combined card total");
            assert_eq!(plan.n_l, vec![8, 8]);
            assert!(plan.violations().is_empty());
        }
    }
}

/// A tier after a stage card that holds every expert is refused by name;
/// a tier with experts left to take, one whose budget leaves it none
/// while the host keeps the rest, and the host-routed planner, which
/// refuses a tier card by name.
#[test]
fn a_tier_with_nothing_left_is_refused() {
    let model = layered(3);
    let machine = |stage: u64, tier: u64| Machine {
        cards: vec![bytes_card("stage", HEAD + stage * EXPERT, 0..3)],
        tiers: vec![bytes_card("tier", tier * EXPERT, 0..0)],
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let whole = machine(24, 4);
    match plan_with(&model, &whole, 4096, &NoKv, None) {
        Err(e @ PlacementError::IdleTier { .. }) => assert_eq!(
            e.to_string(),
            "tier card tier (tier 0) holds no expert: the cards before it hold every routed \
             expert of the layers whose stacks a card can hold; plan without it"
        ),
        Err(e) => panic!("{e}, not IdleTier"),
        Ok(_) => panic!("the idle tier was planned"),
    }
    let part = machine(5, 4);
    let plan = plan_with(&model, &part, 4096, &NoKv, None).expect("a tier with work");
    assert_eq!(plan.tier_n_l, vec![vec![1, 1, 2]]);
    let small = machine(5, 0);
    let plan = plan_with(&model, &small, 4096, &NoKv, None).expect("a tier too small");
    assert_eq!(plan.tier_n_l, vec![vec![0, 0, 0]]);
    assert!(plan.host.experts > 0);
    match plan_host_routed(&model, &whole, 4096, &NoKv, &PlanLevers::default()) {
        Err(e @ PlacementError::HostRoutedTier { .. }) => assert_eq!(
            e.to_string(),
            "tier card tier (tier 0) holds no expert: this plan puts every routed expert \
             on the host; plan without it"
        ),
        Err(e) => panic!("{e}, not HostRoutedTier"),
        Ok(_) => panic!("the host-routed planner accepted a tier card"),
    }
}

/// A tier card on a node of its own ([`workstation::node_of`])
/// takes nothing from this node's pool: the same plan's pool need is the
/// stage card's and the host's alone, so a pool of exactly that plans clean
/// while the tier on this node's pool passes it by the tier card's bytes.
/// A machine of other than one node, or one more a tier card, is refused
/// by name.
#[test]
fn a_tier_on_its_own_node_takes_nothing_of_this_nodes_pool() {
    let model = layered(3);
    let tier = 4 * EXPERT;
    let node = |bytes: u64| Node {
        host: Host {
            usable_bytes: bytes,
            reserves: Vec::new(),
        },
        unified: Some(UnifiedPool { bytes }),
    };
    let machine = |pool: u64, own: bool| Machine {
        cards: vec![bytes_card("stage", HEAD + 5 * EXPERT, 0..3)],
        tiers: vec![bytes_card("tier", tier, 0..0)],
        nodes: if own {
            vec![node(pool), node(tier)]
        } else {
            vec![node(pool)]
        },
    };
    let planned = |pool: u64, own: bool| {
        let m = machine(pool, own);
        let plan = plan_with(&model, &m, 4096, &NoKv, None).expect("plans");
        assert_eq!(plan.tier_n_l, vec![vec![1, 1, 2]], "own node {own}");
        let need = plan.pool_need().expect("a unified pool");
        (plan.host.pool_card_bytes, need, plan.violations())
    };
    let (shared_cards, shared, _) = planned(1 << 40, false);
    let (own_cards, own, _) = planned(1 << 40, true);
    assert_eq!(shared_cards - own_cards, tier);
    assert_eq!(shared.cards - own.cards, tier);
    assert_eq!(shared.host, own.host);
    let (_, _, v) = planned(own.cards + own.host, true);
    assert!(v.is_empty(), "{v:?}");
    match planned(own.cards + own.host, false).2.as_slice() {
        [Violation::PoolOver { cards, .. }] => assert_eq!(*cards, shared.cards),
        v => panic!("{v:?}, not one PoolOver"),
    }
    // A census tier card's context comes out of its own node's pool.
    for (own, pool) in [(true, 1 << 40), (false, (1 << 40) + 1)] {
        let mut m = machine(1 << 40, own);
        m.tiers[0].context_bytes = 1;
        m.tiers[0].free_bytes = Some(tier);
        assert_eq!(
            m.pool_bytes().expect("a pool"),
            Some(pool),
            "own node {own}"
        );
    }
    // What the caller plans beside each card comes out of the pool of
    // that card's node.
    for (own, cards) in [(true, 1), (false, 2)] {
        let m = machine(1 << 40, own);
        let plan = plan_routed_reserving(
            &model,
            &m,
            4096,
            &NoKv,
            &PlanLevers::default(),
            RoutedRule::PerType(CardFormat::of_routed),
            1,
        )
        .expect("plans");
        let need = plan.pool_need().expect("a unified pool");
        assert_eq!(
            plan.host.pool_card_bytes - need.cards,
            cards,
            "own node {own}"
        );
    }
    let two = Machine {
        tiers: vec![bytes_card("tier", tier, 0..0); 2],
        ..machine(1 << 40, true)
    };
    match plan_with(&model, &two, 4096, &NoKv, None) {
        Err(e @ PlacementError::Nodes { nodes: 2, tiers: 2 }) => assert_eq!(
            e.to_string(),
            "a machine of 2 nodes for 2 tier cards: the card plan plans one node, or one more \
             a tier card; plan the others with placement::nodes"
        ),
        Err(e) => panic!("{e}, not Nodes"),
        Ok(_) => panic!("two tiers on one remote node planned"),
    }
}

/// A layer's host experts are the model's less the stage card's and the
/// tier's, they sum to the plan's host experts, a layer the two cards
/// hold whole hosts none, and a layer past the plan's counts or cards
/// that hold more than the model has are refused by name.
#[test]
fn a_layers_host_experts_are_the_models_less_the_cards() {
    let model = layered(3);
    let machine = |stage: u64, tier: u64| Machine {
        cards: vec![bytes_card("stage", HEAD + stage * EXPERT, 0..3)],
        tiers: vec![bytes_card("tier", tier * EXPERT, 0..0)],
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let hosted = |plan: &Plan<'_>| -> Vec<u64> {
        (0..model.layers)
            .map(|l| plan.host_experts(l).expect("a layer of the plan"))
            .collect()
    };
    let (small, big) = (machine(5, 4), machine(20, 4));
    let part = plan_with(&model, &small, 4096, &NoKv, None).expect("a tier with work");
    let want: Vec<u64> = (0..model.layers)
        .map(|l| model.experts - part.n_l[l] - part.tier_n_l[0][l])
        .collect();
    assert_eq!(hosted(&part), want);
    assert_eq!(hosted(&part).iter().sum::<u64>(), part.host.experts);
    assert!(part.host.experts > 0);
    let whole = plan_with(&model, &big, 4096, &NoKv, None).expect("two cards");
    assert_eq!(hosted(&whole), vec![0; model.layers]);
    assert_eq!(whole.host.experts, 0);
    match part.host_experts(model.layers) {
        Err(e @ PlacementError::LayerPastCounts { .. }) => assert_eq!(
            e.to_string(),
            "layer 3 is past the plan's per-layer expert counts (3 layers)"
        ),
        other => panic!("{other:?}, not LayerPastCounts"),
    }
    let mut over = part.clone();
    over.n_l[1] = 5;
    over.tier_n_l[0][1] = 4;
    match over.host_experts(1) {
        Err(e @ PlacementError::LayerOverHeld { .. }) => assert_eq!(
            e.to_string(),
            "layer 1: the stage card holds 5 experts and the tier cards 4, past the \
             model's 8"
        ),
        other => panic!("{other:?}, not LayerOverHeld"),
    }
}
