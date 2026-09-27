//! Gate `gate-gpu-glm5next-card`: GLM-5.3-Flash's routed experts on the card.
//! The model is loaded through the session the engine opens it with, by the
//! gate placement (`workstation::plan_gate`) under the levers parsed at
//! `main` — the hot list `BLOOMERY_HOT_LIST`, or the id prefix without one,
//! and the card budget — and its card experts are checked against sources
//! made here, never against the plan's own segments:
//!
//! - (i) slot map: every routed layer's row of the host tier's slot map
//!   holds the layer's card experts — the hot list's first `n_l` ids of the
//!   layer, or `[0, n_l)`, `n_l` the plan's — in slots `0..n_l` in ascending
//!   id order, and the host mark on the rest; card experts sit only on
//!   layers whose three stacks `card_routed` reads, and those layers' counts
//!   differ by at most one (the expert rule deals one at a time in turn);
//!   without a card budget (`BLOOMERY_CARD_BUDGET`) each of them holds at
//!   least one, so a plan that quietly keeps every expert on the host fails.
//! - (ii) slots: on every layer with card experts, at slots 0, n/2 and
//!   n − 1, each resident stack's `_sel` at the slot is bit for bit the same
//!   entry, or the plain gemv, over that expert's rows uploaded alone from
//!   the file: the Q4_K gate and up by `q4k_gemv_sel` against `q4k_gemv`,
//!   the Q5_K down by `q5k_gemv_sel` against itself on a one-expert stack at
//!   id 0, and the gate·up entry the step runs (`kq_gate_up_act_q4k` or
//!   `_q5k`, the layer's routed limit) against itself on the expert's own
//!   pair. A list whose gather put another expert in a slot fails here.
//! - (iii) card copy: read back from the card, the slot map's copy the
//!   handoff reads holds each routed layer's host row at the map's
//!   `row_offset(l)` — the offset the layer's handoff adds an id to — entry
//!   for entry; under a hot list the rows must differ between layers (a
//!   map of one row repeated would pass a prefix, not a list), so a copy laid
//!   out or indexed by another rule than the host's fails here.
//!
//! A correctness run: the figures it prints are runtime values.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_card: built without the `glm5next` feature; see `just gate-gpu-glm5next-card`."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_card", gate::run())
}

#[cfg(feature = "glm5next")]
mod gate {
    use std::time::Instant;

    use app::arch::glm5next::GlmCfg;
    use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::kquant::{Act, GateUpAct, KquantKernels, SelDown};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use bloomery_gpu_gates::{GateError, bits_equal, bytes_to_words, checks_failed, verdict};
    use bloomery_gpu_glm5next::{Body, Glm5nextModel};
    use bloomery_levers::{CARD_BUDGET, HOT_LIST};
    use cuda_core::DeviceBuffer;
    use gguf::{GgmlType, Split};
    use model::arch::glm5next::hparams::Hparams;
    use model::arch::glm5next::names;
    use model::arch::glm5next::place::{PlanInputs, card_routed};
    use model::placement::{HotList, Machine, Plan, PlanLevers, workstation};
    use refset::arch::glm5next::MODEL;

    /// The serving context the e2e gate loads at.
    const CTX: usize = 1088;

    /// What the plan decided, kept from the open's plan record.
    struct Log {
        t: Instant,
        n_l: Vec<u64>,
        experts: u64,
    }

    impl OpenLog<Body> for Log {
        fn plan(
            &mut self,
            place: &'static str,
            _inputs: &PlanInputs,
            _machine: &Machine,
            plan: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            println!(
                "plan place={place} ctx_max={} host_experts={} card_experts={}",
                plan.ctx_max, plan.host.experts, plan.cards[0].experts
            );
            self.n_l = plan.n_l.clone();
            self.experts = plan.model.experts;
            Ok(true)
        }

        fn load(&mut self, m: &Glm5nextModel) -> Result<(), SessionError> {
            println!(
                "load resident_bytes={} in {:.1} s (runtime value)",
                m.resident_bytes(),
                self.t.elapsed().as_secs_f64()
            );
            Ok(())
        }

        fn capture(&mut self, _nodes: usize) -> Result<(), SessionError> {
            Ok(())
        }

        fn prompt_buffers(&mut self, _m: &Glm5nextModel) -> Result<(), SessionError> {
            Ok(())
        }
    }

    /// Each layer's card experts from their source: the hot list's first
    /// `n_l[l]` ids of the layer, or `[0, n_l[l])`, ascending.
    fn card_lists(n_l: &[u64], hot: Option<&HotList>) -> Result<Vec<Vec<u32>>, GateError> {
        let mut out = Vec::with_capacity(n_l.len());
        for (l, &n) in n_l.iter().enumerate() {
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

    /// Check (i): the host tier's slot map against `lists`, and the layers
    /// the card holds experts on against the stacks' types.
    fn check_map(
        body: &Body,
        split: &Split,
        lists: &[Vec<u32>],
        hp: &Hparams,
        listed: bool,
        budgeted: bool,
    ) -> Result<bool, GateError> {
        let map = body.hybrid().slots();
        let n = map.n_expert();
        let mut bad = Vec::new();
        for l in map.layers() {
            let row = map
                .row(l)
                .ok_or_else(|| format!("no slot-map row for layer {l}"))?;
            let list = &lists[l];
            let want = |e: usize| {
                u32::try_from(e)
                    .ok()
                    .and_then(|e| list.binary_search(&e).ok())
                    .and_then(|s| u32::try_from(s).ok())
                    .unwrap_or(HOST)
            };
            if let Some(e) = (0..n).find(|&e| row[e] != want(e)) {
                bad.push(format!(
                    "layer {l}: expert {e} at {}, the layer's {} card experts put it at {}",
                    row[e],
                    list.len(),
                    want(e)
                ));
            }
        }
        let mut readable = Vec::new();
        let routed = lists
            .iter()
            .enumerate()
            .take(hp.n_trunk)
            .skip(hp.dense_lead);
        for (l, list) in routed {
            let reads = stack_names(l).iter().all(|name| {
                split
                    .find(name)
                    .is_some_and(|(_, info)| card_routed(info.ty).is_some())
            });
            if reads {
                readable.push(l);
            }
            if !reads && !list.is_empty() {
                bad.push(format!(
                    "layer {l}: card experts on stacks the card does not read"
                ));
            }
        }
        let counts: Vec<usize> = readable.iter().map(|&l| lists[l].len()).collect();
        let (lo, hi) = (
            counts.iter().min().copied().unwrap_or(0),
            counts.iter().max().copied().unwrap_or(0),
        );
        if !budgeted && lo == 0 {
            bad.push(format!(
                "no card budget, and a layer the card reads holds no expert ({lo}..{hi} over \
                 {readable:?}): the A6000 and 3090 plans keep some on each"
            ));
        }
        if hi > lo + 1 {
            bad.push(format!(
                "the layers the card reads hold {lo} to {hi} experts: the expert rule deals one \
                 at a time in turn, so they differ by at most one"
            ));
        }
        let on_card: usize = lists.iter().map(Vec::len).sum();
        println!(
            "check i slot map: {} layers x {n} experts, {on_card} on the card from the {}, \
             {lo}..{hi} a layer on the {} layers whose stacks the card reads {readable:?}: {}",
            map.layers().len(),
            if listed { "hot list" } else { "prefix" },
            readable.len(),
            verdict(bad.is_empty())
        );
        for b in &bad {
            println!("FAIL: check i: {b}");
        }
        Ok(bad.is_empty())
    }

    /// Check (iii) (module doc): the card copy against the host map, row by
    /// row at `row_offset`, and a list's rows not all one.
    fn check_copy(
        gpu: &Gpu,
        body: &Body,
        lists: &[Vec<u32>],
        listed: bool,
    ) -> Result<bool, GateError> {
        let map = body.hybrid().slots();
        let copy = body.slot_copy().buf().to_host_vec(gpu.stream())?;
        let n = map.n_expert();
        let mut bad = Vec::new();
        if copy.len() != map.as_slice().len() {
            bad.push(format!(
                "the card copy holds {} entries, the host map {}",
                copy.len(),
                map.as_slice().len()
            ));
        }
        for l in map.layers() {
            let at = map
                .row_offset(l)
                .ok_or_else(|| format!("no row offset for layer {l}"))?;
            let row = map
                .row(l)
                .ok_or_else(|| format!("no slot-map row for layer {l}"))?;
            if copy.get(at..at + n) != Some(row) {
                bad.push(format!(
                    "layer {l}: the card copy at {at} is not the host row"
                ));
            }
        }
        let card: Vec<&Vec<u32>> = lists.iter().filter(|l| !l.is_empty()).collect();
        let distinct = card.windows(2).any(|w| w[0] != w[1]);
        if listed && card.len() > 1 && !distinct {
            bad.push("a hot list put the same experts on every card layer".to_string());
        }
        println!(
            "check iii card copy: {} rows of {n} at their row offsets equal to the host map, \
             {} card layers, rows {}: {}",
            map.layers().len(),
            card.len(),
            if distinct {
                "differ between layers"
            } else {
                "all one"
            },
            verdict(bad.is_empty())
        );
        for b in &bad {
            println!("FAIL: check iii: {b}");
        }
        Ok(bad.is_empty())
    }

    /// Layer `l`'s routed gate, up and down names.
    fn stack_names(l: usize) -> [String; 3] {
        [
            names::ffn_gate_exps(l),
            names::ffn_up_exps(l),
            names::ffn_down_exps(l),
        ]
    }

    /// `k` values of a fixed pattern in about [-1, 1], quantized to q8_1 on
    /// the card.
    fn activation(gpu: &Gpu, k: usize) -> Result<Q8Act, GateError> {
        let x: Vec<f32> = (0..k)
            .map(|i| ((i * 7919 + 13) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let x_dev = DeviceBuffer::from_host(gpu.stream(), &x)?;
        let mut act = Q8Act::with_k(gpu.stream(), 1, k)?;
        gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        Ok(act)
    }

    /// Expert `id`'s rows of stack `name` from the file, uploaded alone:
    /// the stack's type, the upload, its rows and their values.
    fn own_rows(
        gpu: &Gpu,
        split: &Split,
        name: &str,
        experts: usize,
        id: usize,
    ) -> Result<(GgmlType, DeviceTensor<u32>, usize, usize), GateError> {
        let (shard, info) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the split"))?;
        let data = split.shard(shard).ok_or("shard out of range")?.data(info)?;
        let rows = usize::try_from(info.dims[1..].iter().product::<u64>())?;
        let rpe = rows / experts;
        let rb = usize::try_from(info.nbytes)? / rows;
        let bytes = data
            .get(id * rpe * rb..(id + 1) * rpe * rb)
            .ok_or_else(|| format!("{name}: expert {id} runs past the file bytes"))?;
        let words = bytes_to_words(bytes);
        let own = DeviceTensor::upload(gpu.stream(), &words, rpe, words.len() / rpe)?;
        Ok((info.ty, own, rpe, usize::try_from(info.dims[0])?))
    }

    /// The resident stack `name` as K-quant words.
    fn resident<'w>(w: &'w Weights, name: &str) -> Result<&'w DeviceTensor<u32>, GateError> {
        match w.get(name) {
            Some(DevWeight::KQuant { w, .. }) => Ok(w),
            _ => Err(format!("{name}: not resident as a K-quant stack").into()),
        }
    }

    /// One stack's gemv at `slot` of the resident `stack` and at the
    /// expert's own upload: the Q4_K `_sel` against the plain Q4_K gemv, the
    /// Q5_K `_sel` against itself at id 0 of the one-expert upload.
    #[allow(
        clippy::too_many_arguments,
        reason = "the gate's card, kernels, both stacks and the launch's shape (rust-quality R8)"
    )]
    fn gemv_pair(
        gpu: &Gpu,
        kq: &KquantKernels,
        ty: GgmlType,
        stack: &DeviceTensor<u32>,
        own: &DeviceTensor<u32>,
        act: &Q8Act,
        slot: u32,
        rpe: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        let stream = gpu.stream();
        let sel = DeviceBuffer::from_host(stream, &[slot])?;
        let zero = DeviceBuffer::from_host(stream, &[0u32])?;
        let mut y_sel = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
        let mut y_own = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
        match ty {
            GgmlType::Q4_K => {
                gpu.q4k_sel()
                    .enqueue_gemv_q4k_sel(stream, stack, act, &sel, 1, rpe, &mut y_sel)?;
                gpu.enqueue_gemv_q4k(own, act, &mut y_own)?;
            }
            GgmlType::Q5_K => {
                let sink = gpu.unlabelled_sink();
                for (w, sel, y) in [(stack, &sel, &mut y_sel), (own, &zero, &mut y_own)] {
                    let a = SelDown {
                        w,
                        act,
                        sel,
                        n_slots: 1,
                        rows_per_expert: rpe,
                    };
                    kq.enqueue_gemv_q5k_sel(stream, &a, sink, y)?;
                }
            }
            ty => return Err(format!("{ty} has no `_sel` gemv here").into()),
        }
        Ok((y_sel.to_host_vec(stream)?, y_own.to_host_vec(stream)?))
    }

    /// The step's gate·up entry of type `ty` at `slot` of the resident pair
    /// and at id 0 of the expert's own pair, one slot on one column.
    #[allow(
        clippy::too_many_arguments,
        reason = "the gate's card, kernels, both pairs and the launch's shape (rust-quality R8)"
    )]
    fn gate_up_pair(
        gpu: &Gpu,
        kq: &KquantKernels,
        ty: GgmlType,
        pair: [&DeviceTensor<u32>; 2],
        own: [&DeviceTensor<u32>; 2],
        act: &Q8Act,
        slot: u32,
        rpe: usize,
        limit: f32,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        let stream = gpu.stream();
        let sel = DeviceBuffer::from_host(stream, &[slot])?;
        let zero = DeviceBuffer::from_host(stream, &[0u32])?;
        let mut h_sel = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
        let mut h_own = DeviceBuffer::<f32>::zeroed(stream, rpe)?;
        let sink = gpu.unlabelled_sink();
        for ([wg, wu], sel, h) in [(pair, &sel, &mut h_sel), (own, &zero, &mut h_own)] {
            let a = GateUpAct {
                wg,
                wu,
                act,
                sel,
                n_slots: 1,
                rows_per_expert: rpe,
                slots_per_col: 1,
                rule: Act::SwigluClamp { limit },
            };
            match ty {
                GgmlType::Q4_K => kq.enqueue_gate_up_q4k(stream, &a, sink, h)?,
                GgmlType::Q5_K => kq.enqueue_gate_up_q5k(stream, &a, sink, h)?,
                ty => return Err(format!("{ty} has no gate·up entry here").into()),
            }
        }
        Ok((h_sel.to_host_vec(stream)?, h_own.to_host_vec(stream)?))
    }

    /// Check (ii) (module doc) over the layers with card experts in
    /// `lists`.
    fn check_slots(
        gpu: &Gpu,
        w: &Weights,
        split: &Split,
        lists: &[Vec<u32>],
        hp: &Hparams,
        experts: usize,
    ) -> Result<bool, GateError> {
        let kq = KquantKernels::load(gpu.context(), gpu.fault_word())?;
        let (mut pairs, mut layers, mut bad) = (0usize, 0usize, Vec::new());
        for (l, list) in lists.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            if !(hp.dense_lead..hp.n_trunk).contains(&l) {
                bad.push(format!("layer {l}: card experts off the routed layers"));
                continue;
            }
            let limit = hp.limit_exp[l];
            layers += 1;
            let names = stack_names(l);
            let stacks = [
                resident(w, &names[0])?,
                resident(w, &names[1])?,
                resident(w, &names[2])?,
            ];
            let mut slots = vec![0, list.len() / 2, list.len() - 1];
            slots.dedup();
            for s in slots {
                let id = usize::try_from(list[s])?;
                let slot = u32::try_from(s)?;
                let mut own = Vec::with_capacity(3);
                for (name, &stack) in names.iter().zip(&stacks) {
                    let (ty, rows, rpe, k) = own_rows(gpu, split, name, experts, id)?;
                    let act = activation(gpu, k)?;
                    let (a, b) = gemv_pair(gpu, &kq, ty, stack, &rows, &act, slot, rpe)?;
                    pairs += 1;
                    if !bits_equal(&a, &b) {
                        bad.push(format!(
                            "layer {l} {name} slot {s}: `_sel` differs from expert {id}'s own \
                             first at row {:?}",
                            a.iter()
                                .zip(&b)
                                .position(|(x, y)| x.to_bits() != y.to_bits())
                        ));
                    }
                    own.push((ty, rows, rpe, act));
                }
                let (ty, _, rpe, act) = &own[0];
                let (a, b) = gate_up_pair(
                    gpu,
                    &kq,
                    *ty,
                    [stacks[0], stacks[1]],
                    [&own[0].1, &own[1].1],
                    act,
                    slot,
                    *rpe,
                    limit,
                )?;
                pairs += 1;
                if !bits_equal(&a, &b) {
                    bad.push(format!(
                        "layer {l} gate·up slot {s}: the entry differs from expert {id}'s own pair \
                         first at row {:?}",
                        a.iter()
                            .zip(&b)
                            .position(|(x, y)| x.to_bits() != y.to_bits())
                    ));
                }
            }
        }
        let fault = gpu.take_fault()?;
        let pass = bad.is_empty() && pairs > 0 && fault.is_none();
        println!(
            "check ii: {pairs} launches paired on {layers} layers, slots 0, n/2 and n-1: each \
             stack's `_sel` and the gate·up at the slot are its listed expert's own bit for bit, \
             fault word {}: {}",
            fault.map_or_else(|| "clear".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        for b in bad.iter().take(12) {
            println!("FAIL: check ii: {b}");
        }
        if bad.len() > 12 {
            println!("FAIL: check ii: … and {} more", bad.len() - 12);
        }
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[HOT_LIST, CARD_BUDGET])?;
        let place = PlanLevers::from_levers(&levers)?;
        let hot = place.hot.clone();
        let budgeted = place.card_budget_bytes.is_some();
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let inputs = PlanInputs::read(&file)?;
        let hp = inputs.hp.clone();
        let cfg = GlmCfg {
            place,
            host: levers.host(),
        };
        let mut log = Log {
            t: Instant::now(),
            n_l: Vec::new(),
            experts: 0,
        };
        let args = OpenArgs {
            place: "gate",
            machine: workstation::plan_gate,
            ctx: CTX,
            mode: StepMode::Graph,
            cfg,
        };
        let loaded =
            Loaded::<Body>::open(file, args, &mut log)?.ok_or("the open stopped at its plan")?;
        let mut s: Session<Body> = loaded.ready(&mut log)?;
        let lists = card_lists(&log.n_l, hot.as_ref())?;
        let split = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let experts = usize::try_from(log.experts)?;
        let m = s.model_mut();
        let (gpu, w, body) = m.body_parts("gate_glm5next_card")?;
        let mut ok = check_map(body, &split, &lists, &hp, hot.is_some(), budgeted)?;
        ok &= check_copy(gpu, body, &lists, hot.is_some())?;
        ok &= check_slots(gpu, w, &split, &lists, &hp, experts)?;
        if ok {
            println!(
                "PASSED: gate_glm5next_card the slot map holds the {} card experts, only on \
                 layers whose stacks the card reads; the card copy is the host map at its row \
                 offsets; each stack's `_sel` and the gate·up at slots 0, n/2 and n-1 are the \
                 listed expert's own bit for bit",
                if hot.is_some() {
                    "hot list's"
                } else {
                    "prefix's"
                }
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
