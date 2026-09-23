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
//!   expert bytes (`gate_load_v41`'s check 1); the slot map read back from the
//!   card holds, per layer, the plan's experts `[0, n_l)` in slots `0..n_l`
//!   and the host mark on the rest, and equals the host copy the host tier
//!   serves by.
//! - (ii) state: per layer, the body's window ring, compressed rows, index
//!   keys and compressor state are `KvLayout`'s bytes for the layer, exactly.
//! - (iii) image: at the decode-step sets' positions (4, 301 and 1,025) the
//!   host plan's step, with an embedding row and engram rows of a known
//!   pattern, is built into the step image, uploaded by the body's refresh
//!   and read back from the card. Its integers are the plan's; its rope
//!   tables are `RopeTable`'s at the positions the plan names, from specs
//!   made here from the file's keys as the rope gate makes them, and a row
//!   table of a group the step does not complete is zero; the rows come back
//!   in the layout's packing.
//! - (iv) refusals: capturing the chain and seeding a depth both refuse.
//! - (v) reload: a probe context holds the card across two loads; each
//!   model's drop gives back everything its load took, and the second load
//!   takes exactly what the first took.
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
    use std::collections::BTreeSet;
    use std::time::Instant;

    use bloomery_gpu::Gpu;
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::model::ChainBody;
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model, StepInput};
    use bloomery_gpu_deepseek41::params::{ImageView, Table};
    use bloomery_gpu_deepseek41::rope::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_dir_named, verdict};
    use cuda_core::CudaStream;
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::kv::KvLayout;
    use model::arch::deepseek41::plan::{Planner, StepPlan};
    use model::arch::deepseek41::roles;
    use model::placement::{self, Device, KvBytes, Plan, workstation};

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

    pub fn run() -> Result<(), GateError> {
        let path = workstation::model_v41();
        let split = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let hp = Hparams::read(&split)?;
        let model = roles::classify(&split, &hp)?;
        let kv = KvLayout::from_file(&split, &hp)?;
        let machine = workstation::plan_gate(model.layers);
        let plan = placement::plan(&model, &machine, CTX_MAX, &kv)?;
        let broken = plan.violations();
        if !broken.is_empty() {
            let list: Vec<String> = broken.iter().map(ToString::to_string).collect();
            return Err(format!("the gate plan breaks its invariants: {}", list.join("; ")).into());
        }
        let planner = Planner::from_file(&split, &hp, CTX_MAX)?;
        let specs = rope_specs_from_keys(&split)?;
        drop(split);
        print_plan(&path, &plan);

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
        {
            let w = m.stages()[0]
                .weights()
                .ok_or("the loaded stage carries no weights")?;
            ok &= check_segments(&plan, w);
        }
        {
            let (gpu, _, body) = m.body_parts("gate_deepseek41_load")?;
            ok &= check_slots(&plan, body, gpu.stream())?;
            ok &= check_state(&kv, body);
            ok &= check_image(&planner, &specs, body, gpu.stream())?;
        }
        ok &= check_refusals(&mut m);
        drop(m);
        let free2 = free(&probe)?;
        let m = load(&path, 2)?;
        let free3 = free(&probe)?;
        drop(m);
        let free4 = free(&probe)?;
        ok &= check_reload([free0, free1, free2, free3, free4]);

        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_deepseek41_load — the gate plan's segments are resident at its bytes and \
             the slot map is its expert prefix on the card and in the host tier; every layer's \
             state is KvLayout's bytes; the step image at positions 4, 301 and 1025 reads back as \
             the plan's integers and RopeTable's tables; the chain and a synthetic depth refuse; a \
             second load takes what the first took and each drop gives it back"
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

    fn print_plan(path: &str, plan: &Plan<'_>) {
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
    /// for it, nothing else is resident, and the total is the card's dense +
    /// expert bytes.
    fn check_segments(plan: &Plan<'_>, w: &Weights) -> bool {
        let name = &plan.machine.cards[0].name;
        let (mut segments, mut planned, mut bad) = (0usize, BTreeSet::new(), Vec::new());
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            for seg in row.segments.iter().filter(|s| s.device == Device::Card(0)) {
                segments += 1;
                planned.insert(t.name.as_str());
                let want = seg.buffer_bytes(t, plan.model.experts);
                match (w.get(&t.name).map(buffers), want) {
                    (Some(got), Ok(want)) if got == want => {}
                    (Some(got), want) => bad.push(format!(
                        "{}: buffers {got:?} B resident, the plan's segment {want:?}",
                        t.name
                    )),
                    (None, _) => bad.push(format!("{}: not resident", t.name)),
                }
            }
        }
        for n in w.names().filter(|n| !planned.contains(n)) {
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

    /// Check (i), slot map: read back from the card, each layer's row holds
    /// the plan's experts `[0, n_l)` in slots `0..n_l` and [`HOST`] on the
    /// rest, and the card copy is the host tier's copy entry for entry.
    fn check_slots(plan: &Plan<'_>, body: &Body, stream: &CudaStream) -> Result<bool, GateError> {
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
            let n_l = usize::try_from(plan.n_l[l])?;
            let want = |e: usize| u32::try_from(e).ok().filter(|_| e < n_l).unwrap_or(HOST);
            if let Some(e) = (0..row.len()).find(|&e| row[e] != want(e)) {
                bad.push(format!(
                    "layer {l}: expert {e} in {}, the plan's n_l {n_l} puts it in {}",
                    show(row[e]),
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
             prefixes hold {planned}, host copy {}: {}",
            layers.len(),
            if same { "equal" } else { "differs" },
            verdict(bad.is_empty())
        );
        for b in &bad {
            println!("FAIL: check i {name} slot map: {b}");
        }
        Ok(bad.is_empty())
    }

    /// Check (ii): each layer's cache and compressor bytes, measured from the
    /// body's buffers, are `KvLayout`'s for the layer.
    fn check_state(kv: &KvLayout, body: &Body) -> bool {
        let mut ok = true;
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
             per table, {} embedding values, {} engram bytes per token",
            body.image().layout().bytes(),
            dims.tokens,
            dims.stream_ratios,
            dims.rope_dims,
            dims.n_embd,
            dims.engram_bytes
        );
        let mut ok = true;
        let mut plan = StepPlan::default();
        for set in STEP_SETS {
            let (tokens, pos) = step_of(set)?;
            let at = pos as usize;
            planner.plan_into(&tokens[at..], pos, &tokens[..at], &mut plan)?;
            let embd: Vec<u16> = (0..dims.n_embd)
                .map(|i| (i as u16).wrapping_mul(0x9e37) ^ (pos as u16))
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

    /// The caller's rows, read back in the layout's packing: two bf16 to a
    /// word, the first low; four engram bytes to a little-endian word.
    fn rows_match(set: &str, view: &ImageView<'_>, embd: &[u16], engram: &[u8]) -> bool {
        let embd_words: Vec<u32> = embd
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[lo, hi]| u32::from(lo) | (u32::from(hi) << 16))
            .collect();
        let engram_words: Vec<u32> = engram
            .chunks(4)
            .map(|b| {
                let mut le = [0u8; 4];
                le[..b.len()].copy_from_slice(b);
                u32::from_le_bytes(le)
            })
            .collect();
        line(
            set,
            &format!(
                "embedding row of {} bf16 and engram rows of {} bytes, back as they went",
                embd.len(),
                engram.len()
            ),
            view.embd(0) == embd_words.as_slice() && view.engram(0) == engram_words.as_slice(),
        )
    }

    /// Check (iv): the chain and a synthetic depth refuse.
    fn check_refusals(m: &mut Deepseek41Model) -> bool {
        let describe = |r: &Result<String, String>| match r {
            Ok(v) => format!("accepted ({v})"),
            Err(e) => format!("refused: {e}"),
        };
        let chain = m
            .capture_step()
            .map(|n| format!("{n} nodes"))
            .map_err(|e| e.to_string());
        let chain_ok = matches!(&chain, Err(e) if e.contains("not assembled"));
        println!(
            "check iv: capturing the chain — {}: {}",
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
        chain_ok && seed_ok
    }

    /// Check (v): the card's free bytes before the first load, after it,
    /// after its drop, after the second load and after its drop.
    fn check_reload([before, first, dropped, second, end]: [u64; 5]) -> bool {
        let took = |a: u64, b: u64| i128::from(a) - i128::from(b);
        let pass = dropped == before && second == first && end == before;
        println!(
            "check v: free {before} B before, {first} after load 1 (took {}), {dropped} after its drop \
             (gave back {}), {second} after load 2 (took {}), {end} after its drop (gave back {}): {}",
            took(before, first),
            took(dropped, first),
            took(dropped, second),
            took(end, second),
            verdict(pass)
        );
        if !pass {
            println!(
                "FAIL: check v: the first drop left {} B taken, the second load took {} B more than \
                 the first, the second drop left {} B taken",
                took(before, dropped),
                took(dropped, second) - took(before, first),
                took(before, end)
            );
        }
        pass
    }
}
