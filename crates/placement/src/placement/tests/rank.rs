use std::sync::Arc;

use super::*;

/// A rank file's text over `experts` experts: one row a `(layer, order)`.
fn rank_text(experts: u64, orders: &[(usize, Vec<u32>)]) -> String {
    let mut text =
        format!("# router-rank\tn_expert={experts}\tn_used=2\tsets=prose,code\tpositions=100\n");
    for (layer, order) in orders {
        let ids: Vec<String> = order.iter().map(u32::to_string).collect();
        text.push_str(&format!("{layer}\t{}\n", ids.join(",")));
    }
    text
}

/// [`layered`]`(3)`'s ranks: layer 0 the ids backwards, layer 1 an
/// interleave, layer 2 the evens before the odds.
fn orders() -> Vec<(usize, Vec<u32>)> {
    vec![
        (0, vec![7, 6, 5, 4, 3, 2, 1, 0]),
        (1, vec![3, 1, 5, 7, 0, 2, 4, 6]),
        (2, vec![0, 2, 4, 6, 1, 3, 5, 7]),
    ]
}

fn ranked(orders: &[(usize, Vec<u32>)]) -> PlanLevers {
    let rank = ExpertRank::parse("rank.tsv", &rank_text(8, orders)).expect("a rank");
    PlanLevers {
        expert_rank: Some(Arc::new(rank)),
        ..PlanLevers::default()
    }
}

/// A stage card of 5 of [`layered`]`(3)`'s 24 experts, `tiers` tier cards of
/// `tier` each, and a host that holds the rest.
fn machine(tier: &[u64]) -> Machine {
    Machine {
        cards: vec![bytes_card("stage", HEAD + 5 * EXPERT, 0..3)],
        tiers: tier
            .iter()
            .enumerate()
            .map(|(t, &n)| bytes_card(&format!("tier{t}"), n * EXPERT, 0..0))
            .collect(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    }
}

/// Each routed layer's ids on the stage card, on each tier card in tier
/// order, and on the host.
type Homes = Vec<(Vec<u32>, Vec<Vec<u32>>, Vec<u32>)>;

fn homes(plan: &Plan<'_>, tiers: usize) -> Homes {
    let ids = |row: &Row, device: Device| -> Vec<u32> {
        row.segments
            .iter()
            .filter(|s| s.device == device)
            .flat_map(|s| s.experts.iter().flat_map(|e| e.ids().to_vec()))
            .collect()
    };
    let mut out = vec![(Vec::new(), vec![Vec::new(); tiers], Vec::new()); plan.model.layers];
    for row in &plan.rows {
        let t = &plan.model.tensors[row.tensor];
        let Some(l) = t.layer.filter(|_| t.role == Role::RoutedExperts) else {
            continue;
        };
        out[l] = (
            ids(row, Device::Card(0)),
            (0..tiers).map(|i| ids(row, Device::Card(1 + i))).collect(),
            ids(row, Device::Host),
        );
    }
    out
}

fn refusal(result: Result<Plan<'_>, PlacementError>) -> String {
    match result {
        Err(e @ PlacementError::ExpertRank { .. }) => e.to_string(),
        Err(e) => panic!("{e}, not ExpertRank"),
        Ok(_) => panic!("a rank the plan cannot place was planned"),
    }
}

/// Under a rank the stage card keeps each layer's hottest `n_l` ids, the tier
/// the layer's coldest `own` ids and the host the middle; the counts per
/// layer are the plan's without it, every id has one home, and the plan names
/// the file. Without a rank the stage card keeps the id prefix and the tier
/// the ids after it.
#[test]
fn a_rank_gives_the_stage_the_hottest_and_the_tier_the_coldest() {
    let model = layered(3);
    let machine = machine(&[4]);
    let plain = plan(&model, &machine, 4096, &NoKv, &PlanLevers::default()).expect("a plan");
    assert_eq!(plain.n_l, vec![2, 2, 1]);
    assert_eq!(plain.tier_n_l, vec![vec![1, 1, 2]]);
    assert!(plain.expert_rank.is_none());
    assert_eq!(
        homes(&plain, 1),
        vec![
            (vec![0, 1], vec![vec![2]], vec![3, 4, 5, 6, 7]),
            (vec![0, 1], vec![vec![2]], vec![3, 4, 5, 6, 7]),
            (vec![0], vec![vec![1, 2]], vec![3, 4, 5, 6, 7]),
        ]
    );

    let levers = ranked(&orders());
    let by_rank = plan(&model, &machine, 4096, &NoKv, &levers).expect("a ranked plan");
    assert_eq!(
        (&by_rank.n_l, &by_rank.tier_n_l),
        (&plain.n_l, &plain.tier_n_l)
    );
    assert!(by_rank.violations().is_empty());
    assert_eq!(
        by_rank.expert_rank.as_ref().map(|r| r.path()),
        Some("rank.tsv")
    );
    assert_eq!(
        homes(&by_rank, 1),
        vec![
            // the hottest two are 7 and 6, the coldest one is 0
            (vec![6, 7], vec![vec![0]], vec![1, 2, 3, 4, 5]),
            // the hottest two are 3 and 1, the coldest one is 6
            (vec![1, 3], vec![vec![6]], vec![0, 2, 4, 5, 7]),
            // the hottest one is 0, the coldest two are 5 and 7
            (vec![0], vec![vec![5, 7]], vec![1, 2, 3, 4, 6]),
        ]
    );
}

/// Two tier cards take their blocks from the cold end in tier order: tier 0
/// the coldest `own` ids, tier 1 the next-coldest block, the host what stays.
#[test]
fn tier_cards_take_blocks_from_the_cold_end_in_order() {
    let model = layered(3);
    let machine = machine(&[2, 3]);
    let levers = ranked(&orders());
    let by_rank = plan(&model, &machine, 4096, &NoKv, &levers).expect("a ranked plan");
    let own = &by_rank.tier_n_l;
    assert_eq!(own.len(), 2);
    assert!(own[0].iter().sum::<u64>() > 0 && own[1].iter().sum::<u64>() > 0);
    let found = homes(&by_rank, 2);
    for (l, (_, order)) in orders().into_iter().enumerate() {
        let (stage, tiers, host) = &found[l];
        let n = order.len();
        let (a, b, k) = (
            own[0][l] as usize,
            own[1][l] as usize,
            by_rank.n_l[l] as usize,
        );
        let sorted = |ids: &[u32]| {
            let mut v = ids.to_vec();
            v.sort_unstable();
            v
        };
        assert_eq!(*stage, sorted(&order[..k]), "layer {l} stage");
        assert_eq!(tiers[0], sorted(&order[n - a..]), "layer {l} tier 0");
        assert_eq!(
            tiers[1],
            sorted(&order[n - a - b..n - a]),
            "layer {l} tier 1"
        );
        assert_eq!(*host, sorted(&order[k..n - a - b]), "layer {l} host");
    }
}

/// A rank that disagrees with the model, or that a plan cannot use, is
/// refused by name: a layer the model routes that it does not order (FAIL
/// first: a plan that took it would give the layer a list from nothing), a
/// layer the model has no routed stack in, other experts a layer than the
/// model's, and a plan that keeps every routed expert on the host.
#[test]
fn a_rank_that_disagrees_with_the_model_is_refused_by_name() {
    let model = layered(3);
    let machine = machine(&[4]);
    let mut short = orders();
    short.pop();
    let text = refusal(plan(&model, &machine, 4096, &NoKv, &ranked(&short)));
    assert!(
        text.contains("rank.tsv")
            && text.contains("layer 2 holds routed experts in the model and the rank orders none"),
        "{text}"
    );
    let mut extra = orders();
    extra.push((9, vec![0, 1, 2, 3, 4, 5, 6, 7]));
    let text = refusal(plan(&model, &machine, 4096, &NoKv, &ranked(&extra)));
    assert!(
        text.contains("orders layer 9, which holds no routed experts in the model"),
        "{text}"
    );
    let wide = ExpertRank::parse(
        "wide.tsv",
        &rank_text(
            16,
            &(0..3)
                .map(|l| (l, (0..16).collect()))
                .collect::<Vec<(usize, Vec<u32>)>>(),
        ),
    )
    .expect("a rank over 16 experts");
    let levers = PlanLevers {
        expert_rank: Some(Arc::new(wide)),
        ..PlanLevers::default()
    };
    let text = refusal(plan(&model, &machine, 4096, &NoKv, &levers));
    assert!(
        text.contains("has 8 experts in the model and the rank orders 16 a layer"),
        "{text}"
    );
    let on_host = Machine {
        cards: vec![bytes_card("stage", HEAD, 0..3)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    // the same machine through the ordinary planner: its rule is on, and its
    // fill still puts every expert on the host
    for result in [
        plan_host_routed(&model, &on_host, 4096, &NoKv, &ranked(&orders())),
        plan(&model, &on_host, 4096, &NoKv, &ranked(&orders())),
    ] {
        let text = refusal(result);
        assert!(
            text.contains("places no routed expert on a card, so it has nothing to order"),
            "{text}"
        );
    }
}

/// Explicit node-owned id ranges keep their id blocks, so a rank beside them
/// is refused by name rather than ignored.
#[test]
fn a_rank_beside_node_owned_ranges_is_refused_by_name() {
    let model = layered(3);
    let machine = machine(&[4]);
    let text = refusal(nodes::loaded_routed(
        &model,
        &machine,
        4096,
        &NoKv,
        &ranked(&orders()),
        RoutedRule::PerType(CardFormat::of_routed),
        0,
        Some(&[4, 4]),
    ));
    assert!(text.contains("explicit node-owned id ranges"), "{text}");
}

/// [`layered`]`(5)`'s ranks: layer 0 the ids backwards and each later layer
/// an interleave no other layer has, none the id order.
fn five_orders() -> Vec<(usize, Vec<u32>)> {
    vec![
        (0, vec![7, 6, 5, 4, 3, 2, 1, 0]),
        (1, vec![3, 1, 5, 7, 0, 2, 4, 6]),
        (2, vec![0, 2, 4, 6, 1, 3, 5, 7]),
        (3, vec![5, 0, 6, 1, 7, 2, 3, 4]),
        (4, vec![2, 5, 1, 4, 0, 7, 3, 6]),
    ]
}

/// A stage card and one tier card over [`layered`]`(5)`, the host holding
/// most of every layer's experts.
fn five_machine() -> Machine {
    Machine {
        cards: vec![bytes_card("stage", HEAD + 10 * EXPERT, 0..5)],
        tiers: vec![bytes_card("tier0", 4 * EXPERT, 0..0)],
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    }
}

/// `plan`'s routed ids by home, for every layer: the host segment's and the
/// NVMe tier's, ascending as the lists keep them.
fn split_homes(plan: &Plan<'_>) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut out = vec![(Vec::new(), Vec::new()); plan.model.layers];
    for row in &plan.rows {
        let t = &plan.model.tensors[row.tensor];
        let Some(l) = t.layer.filter(|_| t.role == Role::RoutedExperts) else {
            continue;
        };
        let ids = |device: Device| -> Vec<u32> {
            row.segments
                .iter()
                .filter(|s| s.device == device)
                .flat_map(|s| s.experts.iter().flat_map(|e| e.ids().to_vec()))
                .collect()
        };
        out[l] = (ids(Device::Host), ids(Device::Nvme));
    }
    out
}

/// The room that leaves `keep` experts on the host of a plan whose host leg
/// holds `held` bytes with `heavy` its heaviest layer: the split's own terms
/// ([`expert_nvme_tier`]), so the test fixes how many ids move, not where
/// the dial lands.
fn room_for(plan: &Plan<'_>, keep: u64) -> u64 {
    let need = workstation::HostNeed::of(plan, 0);
    let heavy = split_homes(plan)
        .iter()
        .map(|(host, _)| host.len() as u64 * EXPERT)
        .max()
        .unwrap_or(0);
    u64::try_from(
        u128::from(keep * EXPERT) + need.exact(true) + u128::from(2 * heavy)
            - u128::from(plan.host.expert_bytes),
    )
    .expect("a room under u64")
}

/// Under a rank the NVMe expert tier takes each layer's coldest host ids,
/// not its id tail, and the host keeps the rest: the bytes the split moves
/// are the rank's own, so the arena and the need account nothing new.
/// Without a rank the tier keeps today's id tail. The stage card keeps the
/// layer's hottest ids either way.
#[test]
fn a_rank_gives_the_nvme_tier_the_coldest_host_ids() {
    let model = layered(5);
    let machine = five_machine();
    let orders = five_orders();
    for (levers, coldest) in [(PlanLevers::default(), false), (ranked(&orders), true)] {
        let mut plan = plan(&model, &machine, 4096, &NoKv, &levers).expect("a plan");
        let before = split_homes(&plan);
        let held = plan.host.expert_bytes;
        assert!(
            before
                .iter()
                .all(|(host, nvme)| !host.is_empty() && nvme.is_empty()),
            "the split has not run yet"
        );
        let room = room_for(&plan, 2 * 5);
        expert_nvme_tier(&mut plan, room, Some(0)).expect("the split");
        let after = split_homes(&plan);
        let on_cards = homes(&plan, 1);
        assert_eq!(after.len(), before.len());
        for (l, ((host_before, _), (host, nvme))) in before.iter().zip(&after).enumerate() {
            let order = &orders[l].1;
            let off = host_before.len() - host.len();
            assert_eq!(nvme.len(), off, "layer {l} moves what it does not hold");
            // The expected NVMe list: the layer's coldest `off` host ids
            // under the rank, or its id tail; the host keeps the rest.
            let mut want: Vec<u32> = if coldest {
                order
                    .iter()
                    .rev()
                    .copied()
                    .filter(|id| host_before.binary_search(id).is_ok())
                    .take(off)
                    .collect()
            } else {
                host_before[host_before.len() - off..].to_vec()
            };
            want.sort_unstable();
            assert_eq!(nvme, &want, "layer {l} NVMe ids");
            let mut kept: Vec<u32> = host_before
                .iter()
                .copied()
                .filter(|id| !want.contains(id))
                .collect();
            kept.sort_unstable();
            assert_eq!(host, &kept, "layer {l} host ids");
            let n_l = plan.n_l[l] as usize;
            // The stage card holds the layer's hottest ids under the rank,
            // its id prefix without one.
            let mut card = if coldest {
                order[..n_l].to_vec()
            } else {
                (0..n_l as u32).collect()
            };
            card.sort_unstable();
            assert_eq!(on_cards[l].0, card, "layer {l} card ids");
        }
        // The counts are the plan's, unchanged by the pick: the same bytes
        // move whatever ids they are.
        assert_eq!(plan.host.expert_bytes, 2 * 5 * EXPERT);
        assert_eq!(plan.host.nvme_expert_bytes, held - 2 * 5 * EXPERT);
        assert_eq!(plan.host.nvme_arena_bytes, 0);
    }
}

/// A rank that does not order a layer the split pages is refused by name at
/// the split, the refusal [`ExpertRank::split`] gives.
#[test]
fn a_rank_that_orders_no_layer_of_a_split_is_refused_by_name() {
    let model = layered(5);
    let machine = five_machine();
    let mut plan = plan(&model, &machine, 4096, &NoKv, &PlanLevers::default()).expect("a plan");
    let mut short = five_orders();
    short.pop();
    let rank = ExpertRank::parse("short.tsv", &rank_text(8, &short)).expect("a rank");
    plan.expert_rank = Some(Arc::new(rank));
    let room = room_for(&plan, 2 * 5);
    let text = match expert_nvme_tier(&mut plan, room, Some(0)) {
        Err(e @ PlacementError::ExpertRank { .. }) => e.to_string(),
        Err(e) => panic!("{e}, not ExpertRank"),
        Ok(()) => panic!("a rank that orders no layer of the split paged it"),
    };
    assert!(
        text.contains("short.tsv") && text.contains("layer 4 is not ordered over its 8 experts"),
        "{text}"
    );
}

/// A rank file the reader cannot place is refused by name, never padded or
/// defaulted, and the file it reads gives back its header and orders.
#[test]
fn a_rank_file_the_reader_cannot_place_is_refused_by_name() {
    let good = rank_text(8, &orders());
    let rank = ExpertRank::parse("good.tsv", &good).expect("a rank");
    assert_eq!((rank.sets(), rank.positions()), ("prose,code", 100));
    assert_eq!(rank.layers().collect::<Vec<_>>(), [0, 1, 2]);
    assert_eq!(rank.ranked(1), Some(&[3, 1, 5, 7, 0, 2, 4, 6][..]));
    assert_eq!(rank.ranked(3), None);

    let swap = |from: &str, to: &str| good.replacen(from, to, 1);
    let cases: Vec<(String, &str)> = vec![
        (good.trim_end().to_string(), "no final newline"),
        (
            swap("# router-rank", "# router-rank2"),
            "not a `# router-rank` header",
        ),
        (swap("\tn_used=2", ""), "the header has no n_used"),
        (
            swap("\tn_used=2", "\tn_used=2\tn_used=3"),
            "n_used is given twice",
        ),
        (
            swap("\tsets=prose,code", "\tsets=prose,code\tcolor=red"),
            "color is not one a rank has",
        ),
        (swap("positions=100", "positions=0"), "positions=0"),
        (swap("n_used=2", "n_used=9"), "0 < n_used <= n_expert"),
        (
            swap("7,6,5,4,3,2,1,0", "7,6,5,4,3,2,1"),
            "layer 0 lists 7 ids, not n_expert 8",
        ),
        (
            swap("7,6,5,4,3,2,1,0", "7,6,5,4,3,2,1,0,0"),
            "layer 0 lists id 0 twice",
        ),
        (
            swap("7,6,5,4,3,2,1,0", "7,6,5,4,3,2,1,8"),
            "id 8 is not below n_expert 8",
        ),
        (
            swap("7,6,5,4,3,2,1,0", "7,6,5,4,3,2,1,x"),
            "id \"x\" is not an id",
        ),
        (swap("\n1\t3,1", "\n0\t3,1"), "layer 0 is given twice"),
        (swap("\n1\t3,1", "\n1 3,1"), "is not <layer><TAB><ids>"),
        (
            format!("{}\n", good.lines().next().unwrap()),
            "holds no layer",
        ),
    ];
    for (text, want) in cases {
        let Err(e) = ExpertRank::parse("bad.tsv", &text) else {
            panic!("a rank file with {want:?} was read");
        };
        let said = e.to_string();
        assert!(
            said.starts_with("expert rank bad.tsv: ") && said.contains(want),
            "{want}: {said}"
        );
    }
    let Err(e) = ExpertRank::read(std::path::Path::new("/no/such/rank.tsv")) else {
        panic!("a file that is not there was read");
    };
    assert!(
        e.to_string().contains("/no/such/rank.tsv: cannot be read"),
        "{e}"
    );
}
