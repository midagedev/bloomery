//! Gate `gate-ds41-load`: the V4.1 body on the gate card. The model is loaded
//! through the entry the engine opens V4.1 with
//! (`bloomery_gpu_deepseek41::body::open`) by the gate placement
//! (`workstation::plan_gate`: the 3090 runs every layer and the head) at the
//! serving context, and checked against the plan this binary makes from the
//! same file. A correctness run: every time and memory figure it prints is a
//! runtime value.
//!
//! - (i) segments: every segment the plan puts on the card is resident at the
//!   plan's buffer bytes, nothing else is, and the total is the card's dense +
//!   expert bytes (`gate_load_v41`'s check 1); the routed stacks hold Σ n_l ×
//!   one expert's card bytes; the slot map read back from the card holds, per
//!   layer, the layer's card experts in slots `0..n_l` in ascending id order
//!   and the host mark on the rest, and equals the host copy the host tier
//!   serves by. The card experts are made here from their source alone — the
//!   hot list `BLOOMERY_HOT_LIST` names (its first `n_l` ids of the layer), or
//!   the id prefix `[0, n_l)` when it is unset — not from the plan's segments.
//! - (ii) state: per layer, the body's window ring, compressed rows, index
//!   keys and compressor state are `KvLayout`'s bytes for the layer, exactly;
//!   its ring shadow is `KvLayout`'s shadow bytes, and the page-locked host
//!   allocation that holds the shadows is the plan's host shadow term.
//! - (iii) image: at the decode-step sets' positions (4, 301 and 1,025) the
//!   host plan's step, with an embedding row and engram rows of a known
//!   pattern, is built into the step image, uploaded by the body's refresh
//!   and read back from the card. Its integers are the plan's; its rope
//!   tables are `RopeTable`'s at the positions the plan names, from specs
//!   made here from the file's keys as the rope gate makes them, and a row
//!   table of a group the step does not complete is zero; the rows come back
//!   in the layout's packing.
//! - (iv) the chain captures (its node count is the step gate's), and
//!   seeding a depth refuses.
//! - (v) reload: a probe context holds the card across two loads; the first
//!   model's drop gives back everything but what its chain capture took from
//!   the context (iv) and, when the body held page-locked shadows, the one
//!   2 MiB mapping granule that capture keeps; the second load takes what the
//!   first took less that granule, and its drop gives back all of it.
//! - (vi) slots: on every layer's routed stacks, at slots 0, n/2 and n − 1,
//!   the stack's `_sel` gemv (`q3k_gemv_sel`, `q4k_gemv_sel`) is bit for bit
//!   the plain gemv of an upload of just that slot's expert's rows from the
//!   file — the contract `gate_p9`/`gate_q4k_sel` pin on a prefix, here on the
//!   loaded stacks, so a list whose gather puts another expert in a slot fails.
//!
//! Before any upload the card is found by name and must have free what the
//! plan puts on it besides the context; a short card refuses the run.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_load: built without the `deepseek41` feature; see `just gate-ds41-load`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_load", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::btree_map::Entry;
    use std::collections::{BTreeMap, BTreeSet};
    use std::time::Instant;

    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::model::ChainBody;
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model, StepInput};
    use bloomery_gpu_deepseek41::chain::attn::join_groups;
    use bloomery_gpu_deepseek41::params::{ImageView, Table};
    use bloomery_gpu_deepseek41::rope::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu_gates::{
        GateError, bits_equal, bytes_to_words, checks_failed, ref_dir_named, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::GgmlType;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::kv::KvLayout;
    use model::arch::deepseek41::place::PlanInputs;
    use model::arch::deepseek41::plan::{Planner, StepPlan};
    use model::placement::{CardFormat, Device, HotList, KvBytes, Plan, Role, workstation};

    /// The decode-step sets whose positions check (iii) builds: 4, where no
    /// csa group completes, and 301 and 1,025, where one does and the window
    /// ring has wrapped.
    const STEP_SETS: [&str; 3] = [
        "ref_deepseek41_step4_every_node",
        "ref_deepseek41_d1_every_node",
        "ref_deepseek41_d2_every_node",
    ];

    /// The port's names of the compressed streams, in the planner's order.
    const STREAMS: [&str; 2] = ["csa", "hca"];

    /// The serving context: the plan's and both loads'.
    const CTX_MAX: u64 = workstation::CTX_MAX;

    /// The device granule a page-locked allocation of the shadows' size takes
    /// from the card (check v).
    const PINNED_GRANULE: i128 = 2 << 20;

    pub fn run() -> Result<(), GateError> {
        let path = workstation::model_v41();
        let split = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let inputs = PlanInputs::read(&split)?;
        let machine = workstation::plan_gate(inputs.model.layers);
        let plan = inputs
            .plan(&machine, CTX_MAX)
            .map_err(|e| format!("the gate plan: {e}"))?;
        let planner = Planner::from_file(&split, &inputs.hp, CTX_MAX)?;
        let specs = rope_specs_from_keys(&split)?;
        drop(split);
        let hot = HotList::from_env()?;
        let lists = card_lists(&plan, hot)?;
        print_plan(&path, &plan, hot);

        let card = &plan.machine.cards[0];
        let probe = Gpu::for_card(&card.name).map_err(|e| {
            format!(
                "refusing before any upload: card {} is not here: {e}",
                card.name
            )
        })?;
        let t = &plan.cards[0];
        let need = t.dense_bytes + t.expert_bytes + t.rounding_bytes + t.kv_bytes + t.scratch_bytes;
        let free0 = free(&probe)?;
        println!(
            "card {} is {}: {free0} B free with the probe's context; the plan puts {need} B on it \
             besides the context (resident + rounding + KV + scratch)",
            card.name,
            probe.device_name()?
        );
        if free0 < need {
            return Err(format!(
                "refusing before any upload: {} has {free0} B free, the plan needs {need} B",
                card.name
            )
            .into());
        }

        let mut m = load(&path, 1)?;
        let free1 = free(&probe)?;
        let mut ok = true;
        let shadow_host;
        {
            let w = m.stages()[0]
                .weights()
                .ok_or("the loaded stage carries no weights")?;
            ok &= check_segments(&plan, w, &inputs.hp);
            ok &= check_expert_bytes(&plan, w);
        }
        {
            let (gpu, w, body) = m.body_parts("gate_deepseek41_load")?;
            ok &= check_slots(&plan, &lists, hot.is_some(), body, gpu.stream())?;
            ok &= check_sel(&plan, &lists, &path, gpu, w)?;
            ok &= check_state(&inputs.kv, &plan, body);
            shadow_host = body.shadow_host().bytes;
            ok &= check_image(&planner, &specs, body, gpu.stream())?;
        }
        let (refusals_ok, captured) = check_refusals(&mut m, &probe)?;
        ok &= refusals_ok;
        drop(m);
        let free2 = free(&probe)?;
        let m = load(&path, 2)?;
        let free3 = free(&probe)?;
        drop(m);
        let free4 = free(&probe)?;
        ok &= check_reload([free0, free1, free2, free3, free4], captured, shadow_host);

        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_deepseek41_load — the gate plan's segments are resident at its bytes and \
             the slot map is its card experts ({}) on the card and in the host tier, each slot's \
             `_sel` gemv its expert's own; every layer's state is KvLayout's bytes; the step image \
             at positions 4, 301 and 1025 reads back as the plan's integers and RopeTable's \
             tables; the chain captures and a synthetic depth refuses; a second load takes what \
             the first took and each drop gives back all but the context's capture (and the \
             page-locked shadows' granule it keeps)",
            hot.map_or("the id prefix".to_string(), |h| format!(
                "hot list {}",
                h.path()
            ))
        );
        Ok(())
    }

    /// The card's free bytes, as the probe's context reads them.
    fn free(probe: &Gpu) -> Result<u64, GateError> {
        Ok(probe.mem_info()?.0 as u64)
    }

    /// Load the model by the gate placement, through the engine's entry.
    fn load(path: &str, n: usize) -> Result<Deepseek41Model, GateError> {
        let file = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let start = Instant::now();
        let m = body::open(file, workstation::plan_gate, CTX_MAX as usize)?;
        println!(
            "load {n}: {} B resident in {:.1} s (runtime value)",
            m.resident_bytes(),
            start.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    /// Each layer's card experts, ascending, from their source alone: the
    /// hot list's first `n_l` ids of the layer, or `0..n_l` without one.
    fn card_lists(plan: &Plan<'_>, hot: Option<&HotList>) -> Result<Vec<Vec<u32>>, GateError> {
        let mut out = Vec::with_capacity(plan.model.layers);
        for (l, &n) in plan.n_l.iter().enumerate() {
            let n = usize::try_from(n)?;
            let mut ids: Vec<u32> = match hot {
                Some(h) => h
                    .ranked(l)
                    .get(..n)
                    .ok_or_else(|| format!("the hot list has fewer than {n} ids on layer {l}"))?
                    .to_vec(),
                None => (0..u32::try_from(n)?).collect(),
            };
            ids.sort_unstable();
            out.push(ids);
        }
        Ok(out)
    }

    fn print_plan(path: &str, plan: &Plan<'_>, hot: Option<&HotList>) {
        println!(
            "gate plan of {path}: {} layers, ctx_max {}",
            plan.model.layers, plan.ctx_max
        );
        for (card, t) in plan.machine.cards.iter().zip(&plan.cards) {
            println!(
                "  {} layers {:?}{}: dense {} experts {} ({}) rounding {} KV {} scratch {} context {} \
                 headroom {}",
                card.name,
                card.layers,
                if card.head { " +head" } else { "" },
                t.dense_bytes,
                t.expert_bytes,
                t.experts,
                t.rounding_bytes,
                t.kv_bytes,
                t.scratch_bytes,
                t.context_bytes,
                t.headroom_bytes
            );
        }
        let mut runs: Vec<(usize, usize, u64)> = Vec::new();
        for (l, &n) in plan.n_l.iter().enumerate() {
            match runs.last_mut() {
                Some((_, end, v)) if *v == n => *end = l + 1,
                _ => runs.push((l, l + 1, n)),
            }
        }
        let runs: Vec<String> = runs
            .iter()
            .map(|(a, b, n)| format!("layers {a}..{b} n_l {n}"))
            .collect();
        println!("  expert prefixes: {}", runs.join(", "));
        match hot {
            None => println!("  card experts: the id prefix (BLOOMERY_HOT_LIST unset)"),
            Some(h) => {
                let about: Vec<String> = h
                    .provenance()
                    .iter()
                    .map(|(k, v)| format!("{k} {v}"))
                    .collect();
                println!(
                    "  card experts: hot list {} ({}), each layer's first n_l",
                    h.path(),
                    about.join("; ")
                );
            }
        }
    }

    /// The window-only and the compressed layers' ropes from the file's keys,
    /// as the rope gate makes them.
    fn rope_specs_from_keys(split: &Split) -> Result<(RopeSpec, RopeSpec), GateError> {
        let f = |s: &str| -> Result<f32, GateError> {
            split
                .arch_get_f32(s)
                .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)).into())
        };
        let u = |s: &str| -> Result<u64, GateError> {
            split
                .arch_get_u64(s)
                .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)).into())
        };
        let n_dims = usize::try_from(u("rope.dimension_count")?)?;
        Ok((
            RopeSpec::window(f("rope.freq_base")?, n_dims),
            RopeSpec::yarn(
                f("attention.compress_rope_freq_base")?,
                f("rope.scaling.factor")?,
                i32::try_from(u("rope.scaling.original_context_length")?)?,
                f("rope.scaling.yarn_beta_fast")?,
                f("rope.scaling.yarn_beta_slow")?,
                n_dims,
            ),
        ))
    }

    /// The device buffers a resident weight holds, in bytes, measured from
    /// the buffers.
    fn buffers(dw: &DevWeight) -> Vec<u64> {
        dw.buffer_bytes().into_iter().map(|b| b as u64).collect()
    }

    /// Check (i), segments: each segment on the card holds the plan's buffers
    /// for it — or, for a projection the load joined into a row stream
    /// (`join_groups`), the joint holds the sum of its parts' segments — nothing
    /// else is resident, and the total is the card's dense + expert bytes.
    fn check_segments(plan: &Plan<'_>, w: &Weights, hp: &Hparams) -> bool {
        let name = &plan.machine.cards[0].name;
        let (mut segments, mut planned, mut bad) = (0usize, BTreeSet::new(), Vec::new());
        // Every part's joint, and every joint's parts, over the file's layers.
        let (mut joint_of, mut parts_of) = (BTreeMap::new(), BTreeMap::new());
        for l in 0..hp.layers.len() {
            match join_groups(hp, l) {
                Ok(groups) => {
                    for g in groups {
                        for p in &g.parts {
                            joint_of.insert(p.clone(), g.joint.clone());
                        }
                        parts_of.insert(g.joint, g.parts);
                    }
                }
                Err(e) => bad.push(format!("layer {l}: {e}")),
            }
        }
        // The bytes the plan gives the parts of each resident joint.
        let mut joint_want: BTreeMap<&str, u64> = BTreeMap::new();
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            for seg in row.segments.iter().filter(|s| s.device == Device::Card(0)) {
                segments += 1;
                planned.insert(t.name.as_str());
                let want = seg.buffer_bytes(t, plan.model.experts);
                let joint = joint_of.get(&t.name).filter(|j| w.get(j).is_some());
                match (w.get(&t.name).map(buffers), want, joint) {
                    (Some(got), Ok(want), _) if got == want => {}
                    (Some(got), want, _) => bad.push(format!(
                        "{}: buffers {got:?} B resident, the plan's segment {want:?}",
                        t.name
                    )),
                    (None, Ok(want), Some(j)) => {
                        *joint_want.entry(j.as_str()).or_default() += want.iter().sum::<u64>();
                    }
                    (None, Err(e), Some(_)) => {
                        bad.push(format!("{}: the plan's segment: {e}", t.name));
                    }
                    (None, _, None) => bad.push(format!("{}: not resident", t.name)),
                }
            }
        }
        for (j, want) in &joint_want {
            let got: u64 = w.get(j).map(buffers).unwrap_or_default().iter().sum();
            if got != *want {
                bad.push(format!(
                    "{j}: {got} B resident, its parts' segments {want} B"
                ));
            }
            if let Some(unplanned) = parts_of[*j].iter().find(|p| !planned.contains(p.as_str())) {
                bad.push(format!(
                    "{j}: resident, and its part {unplanned} has no segment here"
                ));
            }
        }
        for n in w
            .names()
            .filter(|n| !planned.contains(n) && !joint_want.contains_key(n))
        {
            bad.push(format!(
                "{n}: resident, and the plan has no segment of it here"
            ));
        }
        let t = &plan.cards[0];
        let (got, want) = (w.resident_bytes() as u64, t.dense_bytes + t.expert_bytes);
        if got != want {
            bad.push(format!(
                "total {got} B, the plan's dense + experts {want} B"
            ));
        }
        println!(
            "check i {name}: {segments} segments, {got} B resident, plan dense + experts {want} B: {}",
            verdict(bad.is_empty())
        );
        for b in &bad {
            println!("FAIL: check i {name}: {b}");
        }
        bad.is_empty()
    }

    /// Check (i), expert bytes: the routed stacks on the card hold Σ n_l × one
    /// expert's card bytes (one expert's rows of each of the layer's routed
    /// stacks), which is the plan's expert bytes and what is resident.
    fn check_expert_bytes(plan: &Plan<'_>, w: &Weights) -> bool {
        let name = &plan.machine.cards[0].name;
        let experts = plan.model.experts;
        let mut one: BTreeMap<usize, u64> = BTreeMap::new();
        let (mut resident, mut bad) = (0u64, Vec::new());
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            let (Role::RoutedExperts, Some(l)) = (t.role, t.layer) else {
                continue;
            };
            let rows: u64 = t.dims.iter().skip(1).product();
            let k = t.dims.first().copied().unwrap_or(0);
            let bytes =
                CardFormat::of(t.ty).and_then(|f| f.resident_bytes(t.ty, k, rows / experts));
            match bytes {
                Some(b) => *one.entry(l).or_default() += b,
                None if plan.n_l[l] == 0 => {}
                None => bad.push(format!("{}: one expert has no card layout", t.name)),
            }
            if row.segments.iter().any(|s| s.device == Device::Card(0)) {
                resident += w.get(&t.name).map_or(0, |dw| dw.resident_bytes() as u64);
            }
        }
        let want: u64 = one.iter().map(|(&l, &b)| plan.n_l[l] * b).sum();
        let per: BTreeSet<u64> = one
            .iter()
            .filter(|&(&l, _)| plan.n_l[l] > 0)
            .map(|(_, &b)| b)
            .collect();
        let planned = plan.cards[0].expert_bytes;
        let pass = bad.is_empty() && want == planned && resident == want;
        println!(
            "check i {name} expert bytes: Σ n_l × {per:?} B per expert = {want} B, the plan's expert \
             bytes {planned} B, resident routed stacks {resident} B: {}",
            verdict(pass)
        );
        for b in &bad {
            println!("FAIL: check i {name} expert bytes: {b}");
        }
        pass
    }

    /// Check (i), slot map: read back from the card, each layer's row holds
    /// the layer's card experts `lists[l]` in slots `0..n_l`, ascending, and
    /// [`HOST`] on the rest, and the card copy is the host tier's copy entry
    /// for entry.
    fn check_slots(
        plan: &Plan<'_>,
        lists: &[Vec<u32>],
        listed: bool,
        body: &Body,
        stream: &CudaStream,
    ) -> Result<bool, GateError> {
        let name = &plan.machine.cards[0].name;
        let got = body.slots().buf().to_host_vec(stream)?;
        let n = usize::try_from(plan.model.experts)?;
        let layers = body.layers();
        let show = |s: u32| {
            if s == HOST {
                "host".to_string()
            } else {
                format!("slot {s}")
            }
        };
        let mut bad = Vec::new();
        if got.len() != layers.len() * n {
            bad.push(format!(
                "{} entries, {} layers of {n} experts want {}",
                got.len(),
                layers.len(),
                layers.len() * n
            ));
        }
        for (row, l) in got.chunks(n).zip(layers.clone()) {
            let list = &lists[l];
            let want = |e: usize| {
                u32::try_from(e)
                    .ok()
                    .and_then(|e| list.binary_search(&e).ok())
                    .and_then(|s| u32::try_from(s).ok())
                    .unwrap_or(HOST)
            };
            if let Some(e) = (0..row.len()).find(|&e| row[e] != want(e)) {
                bad.push(format!(
                    "layer {l}: expert {e} in {}, the layer's {} card experts put it in {}",
                    show(row[e]),
                    list.len(),
                    show(want(e))
                ));
            }
        }
        let host = body.slot_map().as_slice();
        let same = host == got.as_slice();
        if !same {
            let at = host.iter().zip(&got).position(|(h, c)| h != c);
            bad.push(format!(
                "the host copy ({} entries) differs from the card's ({}) at entry {at:?}",
                host.len(),
                got.len()
            ));
        }
        let on_card = got.iter().filter(|&&s| s != HOST).count();
        let planned: u64 = layers.clone().map(|l| plan.n_l[l]).sum();
        println!(
            "check i {name} slot map: {} layers x {n} experts, {on_card} on the card, the plan's \
             {} hold {planned}, host copy {}: {}",
            layers.len(),
            if listed { "hot lists" } else { "prefixes" },
            if same { "equal" } else { "differs" },
            verdict(bad.is_empty())
        );
        for b in &bad {
            println!("FAIL: check i {name} slot map: {b}");
        }
        Ok(bad.is_empty())
    }

    /// The activation column check (vi) dots, `k` values of a fixed pattern
    /// in about [-1, 1], quantized to q8_1 on the card.
    fn activation(gpu: &Gpu, k: usize) -> Result<Q8Act, GateError> {
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 7919 + 13) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let x_dev = DeviceBuffer::from_host(gpu.stream(), &x)?;
        let mut act = Q8Act::with_k(gpu.stream(), 1, k)?;
        gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        Ok(act)
    }

    /// Check (vi): on every routed stack on the card, at slots 0, n/2 and
    /// n − 1 of the layer's card experts `lists[l]`, the `_sel` gemv of the
    /// resident stack is bit for bit the plain gemv of the listed expert's
    /// rows uploaded alone from the file at `path`.
    fn check_sel(
        plan: &Plan<'_>,
        lists: &[Vec<u32>],
        path: &str,
        gpu: &Gpu,
        w: &Weights,
    ) -> Result<bool, GateError> {
        let name = &plan.machine.cards[0].name;
        let stream = gpu.stream();
        let split = Split::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let mut acts: BTreeMap<usize, Q8Act> = BTreeMap::new();
        let (mut pairs, mut layers, mut bad) = (0usize, BTreeSet::new(), Vec::new());
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            let (Role::RoutedExperts, Some(l)) = (t.role, t.layer) else {
                continue;
            };
            let list = &lists[l];
            if list.is_empty() || !row.segments.iter().any(|s| s.device == Device::Card(0)) {
                continue;
            }
            let Some(DevWeight::KQuant { w: stack, .. }) = w.get(&t.name) else {
                bad.push(format!("{}: not resident as a K-quant stack", t.name));
                continue;
            };
            let (shard, info) = split
                .find(&t.name)
                .ok_or_else(|| format!("{} is not in the split", t.name))?;
            let data = split.shard(shard).ok_or("shard out of range")?.data(info)?;
            let k = usize::try_from(info.dims[0])?;
            let rows = usize::try_from(info.dims[1..].iter().product::<u64>())?;
            let rpe = rows / usize::try_from(plan.model.experts)?;
            let rb = usize::try_from(info.nbytes)? / rows;
            let act: &Q8Act = match acts.entry(k) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(v) => v.insert(activation(gpu, k)?),
            };
            let mut slots = vec![0, list.len() / 2, list.len() - 1];
            slots.dedup();
            for s in slots {
                let id = usize::try_from(list[s])?;
                let own_bytes = data
                    .get(id * rpe * rb..(id + 1) * rpe * rb)
                    .ok_or_else(|| format!("{}: expert {id} runs past the file bytes", t.name))?;
                let words = bytes_to_words(own_bytes);
                let own = DeviceTensor::upload(stream, &words, rpe, words.len() / rpe)?;
                let sel = DeviceBuffer::from_host(stream, &[u32::try_from(s)?])?;
                let mut y_sel = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
                let mut y_own = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
                match info.ty {
                    GgmlType::Q3_K => {
                        gpu.enqueue_gemv_q3k_sel(stack, act, &sel, 1, rpe, &mut y_sel)?;
                        gpu.enqueue_gemv_q3k(&own, act, &mut y_own)?;
                    }
                    GgmlType::Q4_K => {
                        gpu.q4k_sel()
                            .enqueue_gemv_q4k_sel(stream, stack, act, &sel, 1, rpe, &mut y_sel)?;
                        gpu.enqueue_gemv_q4k(&own, act, &mut y_own)?;
                    }
                    ty => {
                        bad.push(format!("{}: {ty} has no `_sel` gemv here", t.name));
                        break;
                    }
                }
                let (a, b) = (y_sel.to_host_vec(stream)?, y_own.to_host_vec(stream)?);
                pairs += 1;
                layers.insert(l);
                if !bits_equal(&a, &b) {
                    let r = a
                        .iter()
                        .zip(&b)
                        .position(|(x, y)| x.to_bits() != y.to_bits());
                    bad.push(format!(
                        "layer {l} {} slot {s}: `_sel` differs from expert {id}'s own gemv first \
                         at row {r:?}",
                        t.name
                    ));
                }
            }
        }
        let pass = bad.is_empty() && pairs > 0;
        println!(
            "check vi {name}: {pairs} (stack, slot) pairs on {} layers, slots 0, n/2 and n-1: the \
             `_sel` gemv at each slot is its listed expert's own gemv bit for bit: {}",
            layers.len(),
            verdict(pass)
        );
        for b in bad.iter().take(12) {
            println!("FAIL: check vi {name}: {b}");
        }
        if bad.len() > 12 {
            println!("FAIL: check vi {name}: … and {} more", bad.len() - 12);
        }
        Ok(pass)
    }

    /// Check (ii): each layer's cache and compressor bytes, measured from the
    /// body's buffers, are `KvLayout`'s for the layer; each layer's ring
    /// shadow is `KvLayout`'s shadow bytes, and the host allocation holding
    /// them all is the plan's host shadow term.
    fn check_state(kv: &KvLayout, plan: &Plan, body: &Body) -> bool {
        let mut ok = check_shadows(kv, plan, body);
        let (mut got_all, mut want_all) = (0u64, 0u64);
        let mut per_buffer: Vec<(&str, u64)> = Vec::new();
        for l in body.layers() {
            let Some(bufs) = body.state_buffers(l) else {
                println!("FAIL: check ii layer {l}: the body holds no buffers for it");
                ok = false;
                continue;
            };
            let got: u64 = bufs.iter().map(|&(_, n)| n as u64).sum();
            let want = kv.layer_bytes(l, CTX_MAX);
            let pass = got == want;
            ok &= pass;
            got_all += got;
            want_all += want;
            let parts: Vec<String> = bufs
                .iter()
                .filter(|&&(_, n)| n > 0)
                .map(|(what, n)| format!("{what} {n}"))
                .collect();
            println!(
                "check ii layer {l:>2}: {} = {got} B, KvLayout {want} B: {}",
                parts.join(" + "),
                verdict(pass)
            );
            if !pass {
                let all: Vec<String> = bufs.iter().map(|(what, n)| format!("{what} {n}")).collect();
                println!("FAIL: check ii layer {l}: {}", all.join(", "));
            }
            for &(what, n) in &bufs {
                match per_buffer.iter_mut().find(|(w, _)| *w == what) {
                    Some((_, sum)) => *sum += n as u64,
                    None => per_buffer.push((what, n as u64)),
                }
            }
        }
        let parts: Vec<String> = per_buffer
            .iter()
            .map(|(what, n)| format!("{what} {n}"))
            .collect();
        println!(
            "check ii: {got_all} B over {} layers ({}), KvLayout {want_all} B: {}",
            body.layers().len(),
            parts.join(", "),
            verdict(ok)
        );
        let step: Vec<String> = body
            .step_buffers()
            .iter()
            .map(|(what, n)| format!("{what} {n}"))
            .collect();
        println!(
            "  besides the state: {}; the body holds {} B",
            step.join(", "),
            body.resident_bytes()
        );
        ok
    }

    /// Check (ii)'s shadow half: per layer, the body's shadow bytes against
    /// `KvLayout`'s, and the page-locked allocation against the plan's host
    /// term.
    fn check_shadows(kv: &KvLayout, plan: &Plan, body: &Body) -> bool {
        let mut bad: Vec<String> = Vec::new();
        let mut want_all = 0u64;
        for l in body.layers() {
            let want = kv.shadow_bytes(l, CTX_MAX);
            want_all += want;
            match body.shadow_bytes(l) {
                Some(got) if got as u64 == want => {}
                got => bad.push(format!("layer {l}: body {got:?} B, KvLayout {want} B")),
            }
        }
        let host = body.shadow_host();
        let host_ok = host.bytes as u64 == plan.host.shadow_bytes && host.bytes as u64 == want_all;
        let pass = bad.is_empty() && host_ok;
        println!(
            "check ii shadows: {} layers at {} B each per KvLayout, {want_all} B; the page-locked \
             allocation {} B (unified_addressing={}), the plan's host term {} B: {}",
            body.layers().len(),
            kv.shadow_bytes(body.layers().start, CTX_MAX),
            host.bytes,
            host.unified_addressing,
            plan.host.shadow_bytes,
            verdict(pass)
        );
        for b in &bad {
            println!("FAIL: check ii shadows {b}");
        }
        pass
    }

    /// The step a decode-step set holds: its sequence (`# tokens`) and the
    /// step's position (`# decode_pos`, its last token's).
    fn step_of(set: &str) -> Result<(Vec<u32>, u32), GateError> {
        let path = ref_dir_named(set).join("MANIFEST.tsv");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let (mut tokens, mut pos) = (None, None);
        for line in text.lines().take_while(|l| l.starts_with('#')) {
            if let Some(v) = line.strip_prefix("# tokens\t") {
                tokens = Some(
                    v.split(',')
                        .map(str::parse)
                        .collect::<Result<Vec<u32>, _>>()?,
                );
            } else if let Some(v) = line.strip_prefix("# decode_pos\t") {
                pos = Some(v.parse::<u32>()?);
            }
        }
        match (tokens, pos) {
            (Some(t), Some(p)) if p as usize + 1 == t.len() => Ok((t, p)),
            (t, p) => Err(format!(
                "{}: {} tokens and decode_pos {p:?}: the step is the last token",
                path.display(),
                t.map_or(0, |t| t.len())
            )
            .into()),
        }
    }

    /// One verdict line of check (iii).
    fn line(set: &str, what: &str, pass: bool) -> bool {
        println!("check iii {set:<32} {what}: {}", verdict(pass));
        pass
    }

    /// `values`' bits.
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Check (iii): each set's step built, refreshed and read back.
    fn check_image(
        planner: &Planner,
        specs: &(RopeSpec, RopeSpec),
        body: &mut Body,
        stream: &CudaStream,
    ) -> Result<bool, GateError> {
        let (window, yarn) = (RopeTable::new(&specs.0)?, RopeTable::new(&specs.1)?);
        let dims = body.image().layout().dims().clone();
        println!(
            "check iii: the step image is {} B: {} token(s), streams of ratio {:?}, {} rope values \
             per table, {} embedding bytes, {} engram bytes per token",
            body.image().layout().bytes(),
            dims.tokens,
            dims.stream_ratios,
            dims.rope_dims,
            dims.embd_bytes,
            dims.engram_bytes
        );
        let mut ok = true;
        let mut plan = StepPlan::default();
        for set in STEP_SETS {
            let (tokens, pos) = step_of(set)?;
            let at = pos as usize;
            planner.plan_into(&tokens[at..], pos, &tokens[..at], &mut plan)?;
            let embd: Vec<u8> = (0..dims.embd_bytes)
                .map(|i| (i as u8).wrapping_mul(0x37) ^ (pos as u8))
                .collect();
            let engram: Vec<u8> = (0..dims.engram_bytes)
                .map(|i| (i as u8).wrapping_mul(151) ^ (pos as u8))
                .collect();
            body.image_mut().build(&plan, &embd, &engram)?;
            body.refresh(stream, &StepInput { pos })?;
            let words = body.params().to_host_vec(stream)?;
            let view = body.image().layout().view(&words)?;
            ok &= image_matches(set, &plan, &view, dims.window, (&window, &yarn));
            ok &= rows_match(set, &view, &embd, &engram);
        }
        Ok(ok)
    }

    /// The image's integers and tables against the plan and `RopeTable`.
    fn image_matches(
        set: &str,
        plan: &StepPlan,
        view: &ImageView<'_>,
        window: u32,
        (wt, yt): (&RopeTable, &RopeTable),
    ) -> bool {
        let mut ok = true;
        let p = plan.pos[0];
        let tok = view.token(0);
        let want = [
            plan.tokens[0],
            p,
            plan.raw_write[0] % window,
            p - plan.raw_first[0] + 1,
        ];
        ok &= line(
            set,
            &format!(
                "token {} pos {} slot {} len {} — the plan's token, position, cell {} mod {window} \
                 and window from cell {}",
                tok.token, tok.pos, tok.slot, tok.len, plan.raw_write[0], plan.raw_first[0]
            ),
            [tok.token, tok.pos, tok.slot, tok.len] == want,
        );
        for (s, st) in plan.streams.iter().enumerate() {
            let f = view.stream(s);
            let name = STREAMS.get(s).copied().unwrap_or("stream");
            let fields: [(&str, bool); 8] = [
                ("groups", f.groups as usize == st.groups()),
                ("persists", f.persists as usize == st.persist_src.len()),
                ("n_visible", f.n_visible == st.n_visible.as_slice()),
                ("state_read", f.state_read == st.state_read.as_slice()),
                (
                    "state_write",
                    f.state_write
                        .iter()
                        .map(|&w| u64::from(w))
                        .eq(st.state_write.iter().copied()),
                ),
                ("write_pos", f.write_pos == st.write_pos.as_slice()),
                ("persist_src", f.persist_src == st.persist_src.as_slice()),
                ("persist_dst", f.persist_dst == st.persist_dst.as_slice()),
            ];
            let bad: Vec<&str> = fields.iter().filter(|f| !f.1).map(|f| f.0).collect();
            ok &= line(
                set,
                &format!(
                    "{name} (ratio {}): groups {} kept {} n_visible {:?} state_read {:?} \
                     state_write {:?} write_pos {:?} persist {:?} -> {:?} — the plan's{}",
                    st.ratio,
                    f.groups,
                    f.persists,
                    f.n_visible,
                    f.state_read,
                    f.state_write,
                    f.write_pos,
                    f.persist_src,
                    f.persist_dst,
                    if bad.is_empty() {
                        String::new()
                    } else {
                        format!(", except {}", bad.join(", "))
                    }
                ),
                bad.is_empty(),
            );
            if st.ratio > 1 {
                let got = view.row_table(s, 0).unwrap_or(&[]);
                let (want, at) = match st.write_pos.first() {
                    Some(&w) => {
                        let mut v = Vec::new();
                        yt.push(w, Direction::Forward, &mut v);
                        (bits(&v), format!("YaRN at write position {w}"))
                    }
                    None => (
                        vec![0; got.len()],
                        "zero: the step completes no group".to_string(),
                    ),
                };
                ok &= line(
                    set,
                    &format!("{name} row table: {at}"),
                    !got.is_empty() && got == want.as_slice(),
                );
            }
        }
        let mut bad = Vec::new();
        for table in Table::ALL {
            let (t, dir) = match table {
                Table::WindowForward => (wt, Direction::Forward),
                Table::WindowBack => (wt, Direction::Back),
                Table::YarnForward => (yt, Direction::Forward),
                Table::YarnBack => (yt, Direction::Back),
            };
            let mut v = Vec::new();
            t.push(p, dir, &mut v);
            if view.table(0, table) != bits(&v).as_slice() {
                bad.push(format!("{table:?}"));
            }
        }
        ok &= line(
            set,
            &format!(
                "{} rope tables at position {p} — RopeTable's bit for bit{}",
                Table::ALL.len(),
                if bad.is_empty() {
                    String::new()
                } else {
                    format!(", except {}", bad.join(", "))
                }
            ),
            bad.is_empty(),
        );
        ok
    }

    /// The caller's rows, read back in the layout's packing: four bytes to a
    /// little-endian word, the last zero-padded.
    fn rows_match(set: &str, view: &ImageView<'_>, embd: &[u8], engram: &[u8]) -> bool {
        let words = |bytes: &[u8]| -> Vec<u32> {
            bytes
                .chunks(4)
                .map(|b| {
                    let mut le = [0u8; 4];
                    le[..b.len()].copy_from_slice(b);
                    u32::from_le_bytes(le)
                })
                .collect()
        };
        let (embd_words, engram_words) = (words(embd), words(engram));
        line(
            set,
            &format!(
                "embedding row of {} bytes and engram rows of {} bytes, back as they went",
                embd.len(),
                engram.len()
            ),
            view.embd(0) == embd_words.as_slice() && view.engram(0) == engram_words.as_slice(),
        )
    }

    /// Check (iv): the chain captures and a synthetic depth refuses. Returns
    /// the device bytes the context's first capture took: the driver keeps
    /// them for the context, not the model, and check (v) counts them out.
    fn check_refusals(m: &mut Deepseek41Model, probe: &Gpu) -> Result<(bool, u64), GateError> {
        let describe = |r: &Result<String, String>| match r {
            Ok(v) => format!("accepted ({v})"),
            Err(e) => format!("refused: {e}"),
        };
        let before = free(probe)?;
        let chain = m
            .capture_step()
            .map(|n| format!("{n} nodes"))
            .map_err(|e| e.to_string());
        let captured = before.saturating_sub(free(probe)?);
        let chain_ok = chain.is_ok();
        println!(
            "check iv: capturing the chain — {} (the capture took {captured} B of the context): {}",
            describe(&chain),
            verdict(chain_ok)
        );
        let seed = m
            .seed_depth(1)
            .map(|()| "depth 1".to_string())
            .map_err(|e| e.to_string());
        let seed_ok = matches!(&seed, Err(e) if e.contains("Body::seed_depth"));
        println!(
            "check iv: seeding a depth of 1 — {}: {}",
            describe(&seed),
            verdict(seed_ok)
        );
        Ok((chain_ok && seed_ok, captured))
    }

    /// Check (v): the card's free bytes before the first load, after it,
    /// after its drop, after the second load and after its drop. The first
    /// model's drop leaves exactly `captured` behind — what its chain capture
    /// took from the context (check iv) — and the second cycle, which
    /// captures nothing, leaves nothing.
    /// PIN(2026-09-23): the first capture in a context takes 8 MiB the model's
    /// drop does not return and a second capture takes 0 (measured in the
    /// b5step round); before the chain captured, the pin was "every drop
    /// returns everything".
    /// PIN(2026-09-24): a page-locked allocation of the shadows' size takes one
    /// 2 MiB device granule, and a device allocation made after it and alive
    /// past its free keeps that granule (probe `pinprobe2.py`, shadowhost
    /// round): with shadows and a capture, the first drop leaves the capture
    /// plus exactly that granule, and the second load finds it mapped.
    fn check_reload(
        [before, first, dropped, second, end]: [u64; 5],
        captured: u64,
        shadow_host: usize,
    ) -> bool {
        let took = |a: u64, b: u64| i128::from(a) - i128::from(b);
        let granule: i128 = if shadow_host > 0 && captured > 0 {
            PINNED_GRANULE
        } else {
            0
        };
        let pass = took(before, dropped) == i128::from(captured) + granule
            && took(dropped, second) == took(before, first) - granule
            && end == dropped;
        println!(
            "check v: free {before} B before, {first} after load 1 (took {}), {dropped} after its drop \
             (gave back {}; the capture's {captured} B stay with the context), {second} after load 2 \
             (took {}), {end} after its drop (gave back {}): {}",
            took(before, first),
            took(dropped, first),
            took(dropped, second),
            took(end, second),
            verdict(pass)
        );
        if !pass {
            println!(
                "FAIL: check v: the first drop left {} B taken besides the capture's {captured} \
                 (allowed {granule}: the shadows' {shadow_host} B page-locked), the second load \
                 took {} B more than the first (allowed {}), the second drop left {} B taken",
                took(before, dropped) - i128::from(captured),
                took(dropped, second) - took(before, first),
                -granule,
                took(dropped, end)
            );
        }
        pass
    }
}
