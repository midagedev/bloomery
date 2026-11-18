//! GPU gate for the DSpark draft's load and its feature-to-KV graph
//! (`bloomery_gpu_deepseek41::draft::{load, kv}`), on the 3090.
//!
//! 1. **Load.** The draft file (`$BLOOMERY_DSPARK_MODEL`) and the target's
//!    borrowed tensors (`$BLOOMERY_REF_MODEL`) go on the card, each tensor in
//!    `dspark::card_format`'s format. Every group's buffer bytes equal the
//!    inventory's card bytes, the borrowed head's equal its Q6_K file bytes;
//!    the `cuMemGetInfo` drop across the load is printed next to the sum and
//!    next to the allocator model's figure.
//! 2. **Feature-to-KV on seeded inputs** ([`BLOCKS`]: block 0 appends the
//!    prompt's [`PROMPT_ROWS`] positions, each later block the rows of one
//!    round, [`ROUND_ROWS`]): seeded feature rows in ±2
//!    (`hc_host::lcg`, one seed per block), each row's rope position and ring
//!    row consecutive from the last block's. The engine's chain (features →
//!    `main_x` → every layer) writes finite rows, and the layers alone on the
//!    chain's own `main_x` (`KvAppend::set_main`, `enqueue_layers`, into rings
//!    of their own) write the chain's rows bit for bit.
//! 3. **The rings.** Every row a feature-to-KV graph has written so far equals
//!    in our ring, bit for bit, what the append left there.
//! 4. **The Markov step's previous token.** One row of the head's Markov step
//!    with `first` in the vocabulary raises nothing; with `first` past it the
//!    step raises `FaultSite::TokenId`, its readback carries the word and the
//!    site mask, and every logit of the row is NaN (finite in the vocabulary).
//!
//! PIN(2026-11-18): the dsref set's comparisons retired with the set (`main_x` and every layer's rows within the q8_1 bound of ik's, chained and alone; ik's K = V and its block and kv views of the rings); it was dumped from the Q3_K_M file and ik cannot read bq3km

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_dspark_kv", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use bloomery_gpu::{Fault, FaultSite, Gpu, LAYER_NONE};
    use bloomery_gpu_deepseek41::draft::kv::{DraftRings, KvAppend};
    use bloomery_gpu_deepseek41::draft::load::DraftWeights;
    use bloomery_gpu_deepseek41::markov::{MarkovKernels, MarkovWeights};
    use bloomery_gpu_gates::hc_host::lcg;
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use model::arch::dspark::{self, Borrow, DraftHparams, Group};
    use refset::arch::deepseek41::dspark_model;

    const NAME: &str = "gate_dspark_kv";
    /// Blocks whose feature-to-KV graph is checked.
    const BLOCKS: [usize; 4] = [0, 1, 2, 3];
    /// The rows block 0's graph appends: the prompt's positions.
    const PROMPT_ROWS: usize = 64;
    /// The rows each later block's graph appends: one round's committed rows.
    const ROUND_ROWS: usize = 3;

    // ------------------------------------------------------------ the load

    /// placement.rs's allocator model: an allocation of a granule or more
    /// takes whole granules; a smaller one its size rounded up to 512 from
    /// the first shared granule with room, else a new one.
    fn heap_bytes(buffers: &[u64], granule: u64) -> u64 {
        let (mut taken, mut left) = (0u64, Vec::<u64>::new());
        for &b in buffers {
            if b >= granule {
                taken += b.div_ceil(granule) * granule;
                continue;
            }
            let room = b.next_multiple_of(512).min(granule);
            match left.iter_mut().find(|l| **l >= room) {
                Some(l) => *l -= room,
                None => {
                    left.push(granule - room);
                    taken += granule;
                }
            }
        }
        taken
    }

    fn load(
        gpu: &Gpu,
        draft: &Split,
        target: &Split,
        ok: &mut bool,
    ) -> Result<DraftWeights, GateError> {
        let hp = DraftHparams::read(draft)?;
        let inv = dspark::inventory(draft, &hp, target)?;
        let groups = [
            Group::Attention,
            Group::HyperConnection,
            Group::Router,
            Group::SharedExpert,
            Group::RoutedExperts,
            Group::Fc,
            Group::Markov,
            Group::Confidence,
            Group::OutputNorm,
        ];
        let mut predicted: BTreeMap<String, u64> = BTreeMap::new();
        for g in groups {
            predicted.insert(
                format!("{g:?}"),
                inv.card_bytes(g).ok_or("a group has no card bytes")?,
            );
        }
        for b in &inv.borrowed {
            let f = b
                .card_format()
                .ok_or("a borrowed tensor has no card format")?;
            let rows = match b.borrow {
                Borrow::Copied => b.dims[1],
                Borrow::RowSource => 1,
            };
            let bytes = f
                .resident_bytes(b.ty, b.dims[0], rows)
                .ok_or("no resident size")?;
            predicted.insert(format!("target {}", b.name), bytes);
        }
        let total_pred: u64 = predicted.values().sum();
        for (k, v) in &predicted {
            println!("{NAME}: predict | {k} | {v} B");
        }
        println!(
            "{NAME}: predict | total | {total_pred} B (inventory card bytes; the mask row is the one resident row source)"
        );

        let (free0, _) = gpu.mem_info()?;
        let w = DraftWeights::load(gpu.stream(), draft, target)?;
        gpu.stream().synchronize()?;
        let (free1, total) = gpu.mem_info()?;

        let mut got: BTreeMap<String, u64> = BTreeMap::new();
        let mut buffers = Vec::new();
        println!("{NAME}: table | tensor | group | card format | buffers B");
        for t in w.table() {
            let key = match t.group {
                Some(g) => format!("{g:?}"),
                None => format!("target {}", t.name),
            };
            *got.entry(key.clone()).or_default() += t.buffers.iter().sum::<u64>();
            buffers.extend_from_slice(&t.buffers);
            println!(
                "{NAME}: table | {} | {key} | {} | {:?}",
                t.name, t.format, t.buffers
            );
        }
        let same = got == predicted;
        println!(
            "{NAME}: load | per-group buffer bytes equal the prediction: {same} {}",
            verdict(same)
        );
        if !same {
            for (k, v) in &got {
                println!(
                    "{NAME}: load | {k} | loaded {v} predicted {:?}",
                    predicted.get(k)
                );
            }
        }
        *ok &= same;
        let sum: u64 = buffers.iter().sum();
        let granule = gpu.allocation_granularity()? as u64;
        let model = heap_bytes(&buffers, granule);
        let delta = (free0 - free1) as u64;
        println!(
            "{NAME}: load | {} buffers, sum {sum} B; allocator model (granule {granule}) {model} B; \
             cuMemGetInfo free {free0} -> {free1} of {total}: delta {delta} B ({:+} vs sum, {:+} vs model) — measured, not pinned",
            buffers.len(),
            delta as i64 - sum as i64,
            delta as i64 - model as i64
        );
        Ok(w)
    }

    struct Cx<'a> {
        gpu: &'a Gpu,
        w: &'a DraftWeights,
        hp: &'a DraftHparams,
        ok: bool,
    }

    /// Block `b`'s seeded graph through our append, both ways; returns the
    /// ring rows it wrote (the ring row and its rope position).
    fn check_block(
        cx: &mut Cx<'_>,
        b: usize,
        kv: &mut KvAppend,
        rings: &mut DraftRings,
        iso: &mut DraftRings,
    ) -> Result<Vec<(usize, i32)>, GateError> {
        let (hp, s) = (cx.hp, cx.gpu.stream());
        let (first, n) = match b {
            0 => (0, PROMPT_ROWS),
            _ => (PROMPT_ROWS + (b - 1) * ROUND_ROWS, ROUND_ROWS),
        };
        let seed = 0x5eed_0000 + u32::try_from(b)?;
        let feats = lcg(n * kv.features(), seed);
        println!(
            "{NAME}: block {b}: {n} seeded rows (seed {seed:#x}), ring rows and rope positions {first}..={}",
            first + n - 1
        );

        // The engine's chain: features -> main_x -> every layer.
        kv.stage(s, &feats, u32::try_from(first)?, u32::try_from(first)?)?;
        kv.enqueue(cx.gpu, cx.w, rings)?;
        s.synchronize()?;
        let rb = kv.to_host(s)?;
        let finite = rb.main.iter().all(|x| x.is_finite())
            && rb.out.iter().all(|l| l.iter().all(|x| x.is_finite()));

        // The layers alone, on the chain's own main_x.
        kv.set_main(s, &rb.main)?;
        kv.enqueue_layers(cx.gpu, cx.w, iso)?;
        s.synchronize()?;
        let rbi = kv.to_host(s)?;
        let same = (0..hp.n_layer)
            .filter(|&l| bits(&rbi.out[l]) == bits(&rb.out[l]))
            .count();
        let pass = finite && same == hp.n_layer;
        println!(
            "{NAME}: block {b}: the chain's main_x and rows finite {finite}; the layers alone on its main_x \
             write its rows bit for bit on {same}/{} layers {}",
            hp.n_layer,
            verdict(pass)
        );
        cx.ok &= pass;
        (first..first + n)
            .map(|r| Ok((r, i32::try_from(r)?)))
            .collect()
    }

    /// The rows the graphs wrote so far: in our ring bit for bit what the
    /// appends left.
    fn check_rings(
        cx: &mut Cx<'_>,
        b: usize,
        written: &BTreeMap<usize, (i32, Vec<Vec<u16>>)>,
        rings: &DraftRings,
    ) -> Result<(), GateError> {
        let (hp, s) = (cx.hp, cx.gpu.stream());
        let hd = hp.head_dim;
        for l in 0..hp.n_layer {
            let ours = rings.ring(l).ok_or("a ring")?.buf().to_host_vec(s)?;
            let own = written
                .iter()
                .filter(|&(&r, (_, rows))| {
                    let slot = r % hp.window;
                    ours[slot * hd..(slot + 1) * hd] == rows[l][..]
                })
                .count();
            let w = written.len();
            let pass = own == w;
            println!(
                "{NAME}: rings after block {b} layer {l}: {w} written rows; ours = our appends {own}/{w} {}",
                verdict(pass)
            );
            cx.ok &= pass;
        }
        Ok(())
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The Markov step's previous token read on the card: one row, `first`
    /// in the vocabulary raises nothing; `first` three past it raises
    /// [`FaultSite::TokenId`] as an unlabelled launch, the step's own
    /// readback (`tok[1..3]`, the argmax's copy of the word and the mask)
    /// carries it, and the row's logits are NaN, every one — no correction a
    /// draft could be read from (finite, every one, in the vocabulary).
    fn markov_first(gpu: &Gpu, w: &DraftWeights) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let mk = MarkovKernels::load(gpu.context())?;
        let (w1, w2) = w.markov();
        let mw = MarkovWeights { w1, w2 };
        let n_vocab = w2.rows();
        let bad = u32::try_from(n_vocab)? + 3;
        let want_fault = Fault::at(LAYER_NONE, FaultSite::TokenId);
        let mut ok = gpu.take_fault()?.is_none();
        for (first_id, want) in [(5u32, None), (bad, Some(want_fault))] {
            let first = DeviceBuffer::from_host(stream, &[first_id])?;
            let mut tok = DeviceBuffer::from_host(stream, &[u32::MAX; 3])?;
            let mut logits = DeviceBuffer::<f32>::zeroed(stream, n_vocab)?;
            mk.enqueue_step(
                stream,
                gpu.elem(),
                &mw,
                &first,
                &mut tok,
                1,
                0,
                gpu.unlabelled_sink(),
                &mut logits,
            )?;
            let t = tok.to_host_vec(stream)?;
            let word = Fault::from_words(t[1], t[2]);
            let fault = gpu.take_fault()?;
            let l = logits.to_host_vec(stream)?;
            let nan = l.iter().filter(|v| v.is_nan()).count();
            let finite = l.iter().filter(|v| v.is_finite()).count();
            let logits_ok = if want.is_some() {
                nan == l.len()
            } else {
                finite == l.len()
            };
            let pass = fault == want && word == want && logits_ok;
            let shown = |f: Option<Fault>| f.map_or_else(|| "none".to_owned(), |f| f.to_string());
            println!(
                "{NAME}: markov first={first_id} (n_vocab {n_vocab}): fault=\"{}\" readback=\"{}\" \
                 want=\"{}\"; logits nan={nan} finite={finite} of {} (want all {}) {}",
                shown(fault),
                shown(word),
                shown(want),
                l.len(),
                if want.is_some() { "NaN" } else { "finite" },
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let path = PathBuf::from(dspark_model()?);
        let draft = Split::open(&path)?;
        let target = Split::open(ref_model_path()?)?;
        let gpu = Gpu::new()?;
        println!("{NAME}: card {}", gpu.device_name()?);
        let mut ok = true;
        let w = load(&gpu, &draft, &target, &mut ok)?;
        ok &= markov_first(&gpu, &w)?;
        let hp = w.hp().clone();

        let mut kv = KvAppend::new(&gpu, &hp)?;
        let mut rings = DraftRings::new(gpu.stream(), &hp)?;
        let mut iso = DraftRings::new(gpu.stream(), &hp)?;
        println!(
            "{NAME}: rings {} B, append buffers {} B, launches per append of 1/4/64 rows {}/{}/{}",
            rings.device_bytes(),
            kv.device_bytes(),
            kv.launches(1),
            kv.launches(4),
            kv.launches(64)
        );
        let mut cx = Cx {
            gpu: &gpu,
            w: &w,
            hp: &hp,
            ok,
        };
        let mut written: BTreeMap<usize, (i32, Vec<Vec<u16>>)> = BTreeMap::new();
        for b in BLOCKS {
            let rows = check_block(&mut cx, b, &mut kv, &mut rings, &mut iso)?;
            let s = gpu.stream();
            let now: Vec<Vec<u16>> = (0..hp.n_layer)
                .map(|l| Ok(rings.ring(l).ok_or("a ring")?.buf().to_host_vec(s)?))
                .collect::<Result<_, GateError>>()?;
            for (r, p) in rows {
                let slot = r % hp.window;
                let per: Vec<Vec<u16>> = now
                    .iter()
                    .map(|ring| ring[slot * hp.head_dim..(slot + 1) * hp.head_dim].to_vec())
                    .collect();
                written.insert(r, (p, per));
            }
            check_rings(&mut cx, b, &written, &rings)?;
        }
        if cx.ok {
            println!(
                "PASSED: {NAME} — the draft on the card at its predicted bytes; seeded blocks 0-3: the chain's rows finite, \
                 the layers alone on its main_x write them bit for bit; the rings hold every append"
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
