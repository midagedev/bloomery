use super::*;
use crate::graph::Graph;
use crate::host::node::{NodeCensus, NodeOpen};
use crate::host::nvtier::{ArenaKind, NvTier};
use crate::host::run::HostWidths;
use crate::host::tier::{TierAct, TierWork};
use crate::weights::Weights;
use bloomery_levers::NvTierEvict;
use cuda_core::DeviceBuffer;
use gguf::write::{Layout, TensorDecl, Writer};
use model::moe::{HostLayer, HostLayerSpec};
use model::placement::{
    Device, ExpertList, Format, ModelTensor, ModelTensors, Role, Row, Segment, nodes, workstation,
};

#[test]
#[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
fn hw_held_node_nvme_host_run_keeps_eager_and_captured_rows() {
    for act in [TierAct::F32, TierAct::Q8] {
        for evict in [NvTierEvict::Tick, NvTierEvict::Heat] {
            node_nvme_host_run(crate::census().unwrap(), true, act, evict);
        }
    }
}

#[test]
fn node_nvme_host_budget_keeps_the_os_reserve() {
    let card = workstation::A6000;
    node_nvme_host_run(
        vec![model::placement::devices::DeviceInfo {
            ordinal: 0,
            name: card.name.into(),
            total_bytes: card.total_bytes,
            free_bytes: card.usable_bytes(),
            integrated: false,
            uuid: [1; 16],
            pci_bus: "0000:01:00.0".into(),
            held_by: None,
            compute_capability: Some((8, 6)),
        }],
        false,
        TierAct::F32,
        NvTierEvict::Tick,
    );
}

fn node_nvme_host_run(
    census: Vec<model::placement::devices::DeviceInfo>,
    hardware: bool,
    act: TierAct,
    evict: NvTierEvict,
) {
    const WIDTH: usize = 256;
    const ROW_BYTES: usize = 34 * WIDTH / 32;
    const LAYERS: usize = 8;
    const EXPERTS: usize = 128;
    let dir = std::env::temp_dir().join(format!(
        "bloomery-node-nvme-{}-{hardware}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("model.gguf");
    let result = std::panic::catch_unwind(|| {
        let names: Vec<[String; 3]> = (0..LAYERS).map(model::arch::names::routed_stacks).collect();
        let tensors: Vec<ModelTensor> = names
            .iter()
            .enumerate()
            .flat_map(|(layer, names)| {
                names.iter().map(move |name| ModelTensor {
                    name: name.clone(),
                    shard: 0,
                    layer: Some(layer),
                    role: Role::RoutedExperts,
                    ty: gguf::GgmlType::Q8_0,
                    dims: vec![WIDTH as u64, WIDTH as u64, EXPERTS as u64],
                    file_bytes: (ROW_BYTES * WIDTH * EXPERTS) as u64,
                    gathered_rows: None,
                })
            })
            .collect();
        let layout = Layout::new(
            &[],
            tensors
                .iter()
                .map(|t| TensorDecl {
                    name: t.name.clone(),
                    dims: t.dims.clone(),
                    type_id: 8,
                    nbytes: t.file_bytes,
                })
                .collect(),
        )
        .unwrap();
        let mut writer = Writer::new(std::fs::File::create_new(&path).unwrap(), layout).unwrap();
        let mut bytes = Vec::new();
        for expert in 0..EXPERTS {
            for _ in 0..WIDTH * (WIDTH / 32) {
                bytes.extend_from_slice(&[0x00, 0x24]);
                bytes.extend((0..32).map(|i| (1 + (expert + i) % 7) as u8));
            }
        }
        for tensor in &tensors {
            writer.tensor(&tensor.name, &bytes).unwrap();
        }
        writer.finish().unwrap();
        let file = Arc::new(gguf::Split::open(&path).unwrap());
        let model = ModelTensors {
            tensors,
            layers: LAYERS,
            experts: EXPERTS as u64,
            experts_used: 2,
            bf16_words: |_| false,
        };
        // Leave expert 0 off the tier so its places are not global ids.
        let assigned: Vec<Row> = model
            .tensors
            .iter()
            .enumerate()
            .map(|(tensor, _)| Row {
                tensor,
                segments: vec![Segment {
                    device: Device::Card(0),
                    format: Format::HostFile,
                    experts: Some(ExpertList::range(1..EXPERTS as u64).unwrap()),
                    resident_bytes: 0,
                }],
                read_bytes: 0,
                stage: 0,
            })
            .collect();
        let mut machine = workstation::plan_on(
            model::placement::devices::spec_of_device(&census[0]),
            LAYERS,
        );
        machine.cards[0].context_bytes = 0;
        machine.cards[0].scratch_bytes = 0;
        machine.cards[0].reserves.clear();
        let mut source = nodes::routed_host(&model, &assigned, 0, &machine, 2).unwrap();
        source.rows = assigned;
        let set = TierSet::new(
            0..LAYERS,
            EXPERTS,
            vec![(1..EXPERTS as u32).collect(); LAYERS],
        )
        .unwrap();
        let shape = TierShape {
            hidden: WIDTH,
            n_used: 2,
            rows: 2,
            act,
        };
        let mut open = NodeOpen {
            nvme: None,
            family: "synthetic".into(),
            name: "synthetic".into(),
            device: Some(model::placement::devices::DeviceId {
                ordinal: census[0].ordinal,
                uuid: census[0].uuid,
            }),
            card: 0,
            rows: NodeOpen::rows_of(&source, &file, 0).unwrap(),
            set: set.clone(),
            shape,
            cols: 2,
            layers: LAYERS,
            block: (0, 0),
        };
        let mut node = NodeCensus {
            devices: census,
            host_room: (0, workstation::HostRead::Given),
            arena_given: Some(72 << 20),
        };
        let mut resident = open.clone();
        for row in &mut resident.rows {
            row.format = Format::Card(model::placement::CardFormat::Q8_0Planes).to_string();
        }
        resident.project(&source, &node).unwrap();
        assert!(resident.nvme.is_none());
        for read in [
            workstation::HostRead::Given,
            workstation::HostRead::MemAvailable,
            workstation::HostRead::VmStats,
            workstation::HostRead::Cgroup {
                path: "/node.scope".into(),
            },
            workstation::HostRead::Machine,
        ] {
            node.host_room.1 = read.clone();
            let refusal = open
                .project(&source, &node)
                .expect_err("the node has no host room");
            assert!(
                refusal
                    .to_string()
                    .contains(&format!("the node's host room 0 B ({read})")),
                "a remote plan refusal must name its own reading: {refusal}"
            );
        }
        node.host_room.1 = workstation::HostRead::Given;
        let mut reconstructed = open.clone();
        reconstructed.nvme = Some(super::super::node::NodeNvme {
            room: u64::MAX,
            arena: workstation::ArenaGrant::whole(0),
            reserve: 0,
        });
        let reconstructed = reconstructed.host_plan(&model, &machine).unwrap();
        let need = workstation::HostNeed::of(&reconstructed, 0);
        let assigned_bytes = (LAYERS * 3 * (EXPERTS - 1) * ROW_BYTES * WIDTH) as u64;
        println!(
            "node host budget: assigned={assigned_bytes} os={} zero-room-refused=true reconstructed={}",
            need.os,
            need.bytes()
        );
        assert_eq!(need.bytes(), assigned_bytes);
        node.host_room.0 = u64::MAX;
        open.project(&source, &node).unwrap();
        assert_eq!(open.nvme.as_ref().unwrap().arena.planned, 0);
        node.host_room.0 = 160 << 20;
        open.project(&source, &node).unwrap();
        resident.nvme = open.nvme.clone();
        node.host_room.0 = 0;
        resident.project(&source, &node).unwrap();
        assert!(resident.nvme.is_none());
        node.host_room.0 = 160 << 20;
        let paging = open
            .nvme
            .as_ref()
            .expect("the constrained remote node must page");
        let mut projected = open.host_plan(&model, &machine).unwrap();
        for row in &mut projected.rows {
            let tensor = &model.tensors[row.tensor];
            if tensor.layer != Some(1) {
                continue;
            }
            let moved: u64 = row
                .segments
                .iter()
                .filter(|s| s.device == Device::Nvme)
                .map(|s| s.resident_bytes)
                .sum();
            row.segments = vec![Segment {
                device: Device::Host,
                format: Format::HostFile,
                experts: Some(ExpertList::range(1..EXPERTS as u64).unwrap()),
                resident_bytes: (ROW_BYTES * WIDTH * (EXPERTS - 1)) as u64,
            }];
            projected.host.expert_bytes += moved;
            projected.host.nvme_expert_bytes -= moved;
            projected.nvme_bytes -= moved;
        }
        let mut missing = open.clone();
        missing.rows.pop();
        assert!(
            missing
                .host_plan(&model, &machine)
                .unwrap_err()
                .to_string()
                .contains("missing routed stack")
        );
        let mut duplicate = open.clone();
        duplicate.rows.push(open.rows[0].clone());
        assert!(
            duplicate
                .host_plan(&model, &machine)
                .unwrap_err()
                .to_string()
                .contains("more than once")
        );
        assert!(
            projected
                .rows
                .iter()
                .all(|r| model.tensors[r.tensor].role == Role::RoutedExperts)
        );
        if !hardware {
            return;
        }
        let gpu = Gpu::new().unwrap();
        let make_tier = || {
            Arc::new(
                NvTier::of_paged(
                    &projected,
                    &file,
                    false,
                    paging.arena,
                    ArenaKind::Anon,
                    evict,
                )
                .unwrap()
                .unwrap(),
            )
        };
        let tier = make_tier();
        assert!(tier.covers(0));
        assert!(!tier.covers(1));
        let make_run = || {
            HostRun::build(
                Arc::clone(&file),
                0,
                false,
                HostWidths {
                    embd: WIDTH,
                    ff: WIDTH,
                    n_used: 2,
                },
                |src| {
                    names
                        .iter()
                        .map(|n| {
                            HostLayer::build(
                                src,
                                &HostLayerSpec {
                                    gate: &n[0],
                                    up: &n[1],
                                    down: &n[2],
                                    n_expert: EXPERTS,
                                    embd: WIDTH,
                                    ff: WIDTH,
                                    swiglu_limit: 0.0,
                                },
                            )
                        })
                        .collect::<Result<Vec<_>, model::ModelError>>()
                },
            )
            .unwrap()
        };
        let mut baseline = make_run();
        let input: Vec<f32> = (0..2 * WIDTH)
            .map(|i| {
                (if i % 128 == 0 {
                    127.0
                } else {
                    (i % 127) as f32 - 63.0
                }) / 256.0
                    * (1u32 << (i / 128)) as f32
            })
            .collect();
        let selected = [
            0,
            (EXPERTS - 2) as u32,
            (EXPERTS - 2) as u32,
            super::super::slots::HOST,
        ];
        let mut expected = vec![123.0f32; 4 * WIDTH];
        for col in 0..2 {
            let x = Tensor2 {
                ne0: WIDTH,
                ne1: 1,
                data: input[col * WIDTH..(col + 1) * WIDTH].to_vec(),
            };
            for slot in 0..2 {
                let at = 2 * col + slot;
                if selected[at] != super::super::slots::HOST {
                    baseline
                        .experts_into(
                            0,
                            &x,
                            &[(selected[at] + 1, 1.0)],
                            &mut expected[at * WIDTH..(at + 1) * WIDTH],
                            1,
                        )
                        .unwrap();
                }
            }
        }
        let mut experts = CpuExperts::new(
            &gpu,
            make_run(),
            Some(Arc::clone(&tier)),
            set.clone(),
            shape,
            2,
        )
        .unwrap();
        let weights = Weights::load_rows(gpu.stream(), &file, &model, &[], 0).unwrap();
        let x = DeviceBuffer::from_host(gpu.stream(), &input).unwrap();
        let mut q8 = crate::Q8Act::with_k(gpu.stream(), 2, WIDTH).unwrap();
        gpu.enqueue_quantize_q8_1(&x, &mut q8).unwrap();
        if act == TierAct::Q8 {
            q8.q4 = DeviceBuffer::zeroed(gpu.stream(), q8.q4.len()).unwrap();
        }
        let input_view = match act {
            TierAct::F32 => TierInput::F32(&x),
            TierAct::Q8 => TierInput::Q8(&q8),
        };
        let sel = DeviceBuffer::from_host(gpu.stream(), &selected).unwrap();
        let mut out =
            DeviceBuffer::from_host(gpu.stream(), &vec![123.0f32; expected.len()]).unwrap();
        let host = DeviceBuffer::from_host(gpu.stream(), &[super::super::slots::HOST; 4]).unwrap();
        for layer in [0, 1] {
            let mut enqueue_host = || {
                experts.enqueue_layer(
                    &gpu,
                    &weights,
                    layer,
                    TierIo {
                        act: input_view,
                        sel: &host,
                        rows: &mut out,
                        cols: 2,
                    },
                )
            };
            enqueue_host().unwrap();
            gpu.stream().synchronize().unwrap();
            let graph = Graph::capture(gpu.stream(), |_| enqueue_host()).unwrap();
            drop(enqueue_host);
            experts.check().unwrap();
            assert_eq!(
                out.to_host_vec(gpu.stream()).unwrap(),
                vec![123.0; 4 * WIDTH]
            );
            graph.launch(gpu.stream()).unwrap();
            gpu.stream().synchronize().unwrap();
            experts.check().unwrap();
            assert_eq!(
                out.to_host_vec(gpu.stream()).unwrap(),
                vec![123.0; 4 * WIDTH]
            );
        }
        let mut enqueue = || {
            experts.enqueue_layer(
                &gpu,
                &weights,
                0,
                TierIo {
                    act: input_view,
                    sel: &sel,
                    rows: &mut out,
                    cols: 2,
                },
            )
        };
        enqueue().unwrap();
        gpu.stream().synchronize().unwrap();
        let check_counts = |counts: [(u32, f64); 3]| {
            for (id, want) in counts {
                match (evict, tier.route_count(0, id)) {
                    (NvTierEvict::Heat, Some(actual)) => assert!(
                        (actual - want).abs() <= 1e-12 * want.abs(),
                        "{act:?} {evict:?}: expert {id}: {actual} != {want}"
                    ),
                    (NvTierEvict::Tick, None) => {}
                    (_, count) => panic!("{evict:?}: unexpected routing count {count:?}"),
                }
            }
        };
        let decay = (-1.0f64 / 32.0).exp2();
        check_counts([(1, 0.0), (2, 0.0), (127, 1.0 + decay)]);
        for _ in 0..64 {
            enqueue().unwrap();
        }
        gpu.stream().synchronize().unwrap();
        let graph = Graph::capture(gpu.stream(), |_| enqueue()).unwrap();
        drop(enqueue);
        let before = [1, 2, 127].map(|id| (id, tier.route_count(0, id).unwrap_or(0.0)));
        experts
            .enqueue_layer(
                &gpu,
                &weights,
                0,
                TierIo {
                    act: input_view,
                    sel: &host,
                    rows: &mut out,
                    cols: 2,
                },
            )
            .unwrap();
        gpu.stream().synchronize().unwrap();
        experts.check().unwrap();
        check_counts(before.map(|(id, count)| (id, count * (-2.0f64 / 32.0).exp2())));
        for round in 0..3 {
            if round != 0 {
                assert!(tier.release(0, (EXPERTS - 1) as u32).unwrap() > 0);
                graph.launch(gpu.stream()).unwrap();
            }
            gpu.stream().synchronize().unwrap();
            experts.check().unwrap();
            let actual = out.to_host_vec(gpu.stream()).unwrap();
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
        assert!(
            tier.stats().fills >= 3,
            "the paged expert must refill after each release"
        );
        assert_eq!(
            experts.host_fills().unwrap(),
            (tier.stats().fills, tier.stats().fill_bytes)
        );
        assert!(
            tier.stats().step_fills > 0,
            "HostRun must use the NVMe step claims"
        );
        drop(graph);
        let mut resident = CpuExperts::new(&gpu, make_run(), None, set.clone(), shape, 2).unwrap();
        resident
            .enqueue_layer(
                &gpu,
                &weights,
                0,
                TierIo {
                    act: input_view,
                    sel: &sel,
                    rows: &mut out,
                    cols: 2,
                },
            )
            .unwrap();
        gpu.stream().synchronize().unwrap();
        resident.check().unwrap();
        assert_eq!(resident.host_fills().unwrap(), (0, 0));
        assert_eq!(
            out.to_host_vec(gpu.stream())
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let mut pending =
            CpuExperts::new(&gpu, make_run(), Some(make_tier()), set.clone(), shape, 2).unwrap();
        let alive = Arc::downgrade(&pending.state);
        let held = crate::checkpoint::hold::Hold::on(gpu.stream()).unwrap();
        pending
            .enqueue_layer(
                &gpu,
                &weights,
                0,
                TierIo {
                    act: input_view,
                    sel: &sel,
                    rows: &mut out,
                    cols: 2,
                },
            )
            .unwrap();
        drop(pending);
        assert!(
            alive.upgrade().is_some(),
            "a queued callback must retain its state"
        );
        held.open();
        gpu.stream().synchronize().unwrap();
        assert_eq!(out.to_host_vec(gpu.stream()).unwrap(), expected);
        let bad = DeviceBuffer::from_host(gpu.stream(), &[(EXPERTS - 1) as u32, 0, 0, 0]).unwrap();
        experts
            .enqueue_layer(
                &gpu,
                &weights,
                0,
                TierIo {
                    act: input_view,
                    sel: &bad,
                    rows: &mut out,
                    cols: 2,
                },
            )
            .unwrap();
        gpu.stream().synchronize().unwrap();
        let work = TierWork::open(gpu, weights, Box::new(experts), shape, 2).unwrap();
        assert!(
            work.fault()
                .unwrap_err()
                .to_string()
                .contains("outside its set")
        );
        let resets = [
            work.clear_fault(),
            work.clear_fault_within(std::time::Duration::from_secs(5), "the callback reset"),
        ];
        println!("failed-callback resets: {resets:?}");
        for reset in resets {
            assert!(reset.unwrap_err().to_string().contains("outside its set"));
        }
        let scoped_gpu = Gpu::new().unwrap();
        let stream = Arc::downgrade(&scoped_gpu.stream_handle());
        let scoped =
            CpuExperts::new(&scoped_gpu, make_run(), Some(make_tier()), set, shape, 2).unwrap();
        scoped_gpu.stream().synchronize().unwrap();
        drop(scoped_gpu);
        if stream.upgrade().is_none() {
            // The stream is gone; do not query its handle while unwinding.
            std::mem::forget(scoped);
            panic!("the host executor must own its stream");
        }
        drop(scoped);
        assert!(
            stream.upgrade().is_none(),
            "a settled executor must release its stream"
        );
    });
    std::fs::remove_dir_all(&dir).unwrap();
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}
