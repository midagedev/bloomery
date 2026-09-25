//! GPU gate for the V4.1 prompt batch (`body::prefill`) on the gate
//! placement (`workstation::plan_gate`): a batched prefill of `P` ids leaves
//! the model in the state `P` decode steps over the same ids leave it, bit
//! for bit.
//!
//! The ids are the first [`ORACLE`] + 1 of `$BLOOMERY_DATA/engram/corpus-prose.ids`
//! (V4.1 text ids, one per line). The feature tap is attached with the DSpark
//! draft's `target_layers` (`$BLOOMERY_DSPARK_MODEL`, header only).
//!
//! - **Oracle, one run.** From a reset, the ids decode-stepped one at a time
//!   through the graph, each position's features read after its step. After
//!   the step at each case's position `P − 1` the gate records every layer's
//!   window ring and compressor state, the features of positions `0 .. P` and
//!   the head's logits; after the step at `P`, the logits again. The rows a
//!   position writes once and no later step touches — each layer's shadow
//!   row, compressed row and index key — are read from the run's end, rows
//!   below each case's `P` (`⌊P / ratio⌋` for a stream).
//! - **Cases.** For each `P` of [`CASES`]: reset, `prefill` of `ids[.. P]`,
//!   then every recorded field compared; then one decode step with `ids[P]`,
//!   its logits against the oracle's position `P`. The split case feeds
//!   `ids[.. 1100]` as two calls (700 + 400). One line per case, every
//!   field's verdict and md5.
//!
//! `--cases a,b,…` runs those `P` only (the oracle then stops at the largest
//! one's `P + 1`), and `--split` / `--no-split` turns the split case on or
//! off: the FAIL-first runs use a short subset.
//!
//! `--seams P` is the locator, not a verdict: `P` eager steps with every
//! seam's streams read back (the finite probe's `observed_step`), then a
//! batch of the same ids with every seam read back (`body::prefill_observed`),
//! and one line per seam — engram, attention, MoE of each layer — with the
//! tokens whose streams differ and the largest difference, up to the first
//! seams that differ.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_prefill: built without the `deepseek41` feature; see `just gate-gpu-ds41-prefill`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_prefill", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the draft's header only; the loop half serves generate_ds41"
)]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_finite.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the probe's streams; its non-finite report serves the other bins"
)]
mod finite;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::BTreeMap;
    use std::time::Instant;

    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu::{DeviceTensor, GpuError};
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model};
    use bloomery_gpu_deepseek41::span::span;
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, verdict};
    use cuda_core::DeviceCopy;
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::placement::workstation;

    use crate::{dspark, finite};

    const NAME: &str = "gate_deepseek41_prefill";
    /// Positions the oracle steps at most: the longest case.
    const ORACLE: usize = 4096;
    /// The prompt lengths: one chunk, one ubatch's chunk seams, the ring's
    /// wrap, the ubatch seam, and several ubatches.
    const CASES: [usize; 12] = [1, 2, 5, 127, 128, 129, 511, 512, 513, 1100, 2600, 4096];
    /// The split case: `ids[.. SPLIT.0 + SPLIT.1]` as two prefill calls.
    const SPLIT: (usize, usize) = (700, 400);

    struct Args {
        cases: Vec<usize>,
        split: bool,
        seams: Option<usize>,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str =
            "usage: gate_deepseek41_prefill [--cases P,P,…] [--split|--no-split] [--seams P]";
        let mut a = Args {
            cases: CASES.to_vec(),
            split: true,
            seams: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--cases" => {
                    let v = it.next().ok_or(USAGE)?;
                    a.cases = v
                        .split(',')
                        .map(|p| p.trim().parse::<usize>())
                        .collect::<Result<_, _>>()?;
                }
                "--seams" => a.seams = Some(it.next().ok_or(USAGE)?.parse()?),
                "--split" => a.split = true,
                "--no-split" => a.split = false,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if a.cases.iter().any(|&p| p == 0 || p > ORACLE) {
            return Err(format!("cases {:?}: each in 1..={ORACLE}", a.cases).into());
        }
        Ok(a)
    }

    /// The first `n` ids of `$BLOOMERY_DATA/engram/corpus-prose.ids`.
    fn corpus(n: usize) -> Result<Vec<u32>, GateError> {
        let path = data_dir().join("engram").join("corpus-prose.ids");
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ids = text
            .split_whitespace()
            .take(n)
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?;
        if ids.len() < n {
            return Err(
                format!("{}: {} ids, the gate reads {n}", path.display(), ids.len()).into(),
            );
        }
        Ok(ids)
    }

    /// What one position `P` leaves, as the oracle recorded it or a case
    /// read it.
    #[derive(Default)]
    struct Snap {
        /// Per layer: the ring's md5, the compressor state's (None without).
        ring: Vec<Digest>,
        state: Vec<Option<Digest>>,
        /// Per layer, rows below `P`: the shadow's, the compressed rows' and
        /// the index keys' md5 (None without).
        shadow: Vec<Digest>,
        rows: Vec<Option<Digest>>,
        keys: Vec<Option<Digest>>,
        /// The features of positions `0 .. P`.
        taps: Digest,
        /// The logits after position `P − 1`, and after the step at `P`.
        logits: Vec<u32>,
        next: Option<Vec<u32>>,
    }

    pub fn run() -> Result<(), GateError> {
        let args = parse_args()?;
        let (_, dhp) = dspark::draft_hparams()?;
        let layers = dhp.target_layers.clone();
        let path = workstation::model_v41();
        let hp = Hparams::read(&Split::open(&path).map_err(|e| format!("open {path}: {e}"))?)?;
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let t = Instant::now();
        let mut m = body::open(file, workstation::plan_gate, workstation::CTX_MAX as usize)?;
        body::attach_features(&mut m, &layers)?;
        m.set_mode(StepMode::Graph);
        body::prepare_prefill(&mut m)?;
        if let Some(p) = args.seams {
            let ids = corpus(p)?;
            return seams(&mut m, &ids);
        }
        let top = args
            .cases
            .iter()
            .copied()
            .chain(args.split.then_some(SPLIT.0 + SPLIT.1))
            .max()
            .unwrap_or(1);
        let steps = (top + 1).min(ORACLE);
        let ids = corpus(ORACLE + 1)?;
        println!(
            "{NAME}: loaded in {:.1} s; batch buffers {} B; ids corpus-prose.ids[..{}] first {:?}; \
             tap layers {layers:?}; oracle {steps} steps; cases {:?} split {}",
            t.elapsed().as_secs_f64(),
            m.body(NAME)?.batch_bytes(),
            ORACLE + 1,
            &ids[..4],
            args.cases,
            args.split
        );
        let mut wanted: Vec<usize> = args.cases.clone();
        if args.split {
            wanted.push(SPLIT.0 + SPLIT.1);
        }
        wanted.sort_unstable();
        wanted.dedup();
        let t = Instant::now();
        let oracle = oracle(&mut m, &hp, &ids, steps, &wanted)?;
        println!(
            "{NAME}: oracle {steps} steps in {:.1} s",
            t.elapsed().as_secs_f64()
        );
        let mut pass = true;
        for &p in &args.cases {
            let t = Instant::now();
            m.reset()?;
            let snap = run_case(&mut m, &hp, &ids, &[p])?;
            pass &= compare(&format!("P={p}"), &snap, &oracle[&p], t);
        }
        if args.split {
            let t = Instant::now();
            let p = SPLIT.0 + SPLIT.1;
            m.reset()?;
            let snap = run_case(&mut m, &hp, &ids, &[SPLIT.0, SPLIT.1])?;
            pass &= compare(
                &format!("split {}+{}", SPLIT.0, SPLIT.1),
                &snap,
                &oracle[&p],
                t,
            );
        }
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// A seam of the observed step: the sub-layer kind and the layer.
    type SeamKey = (body::BatchSeamKind, usize);

    /// `--seams` (module doc).
    fn seams(m: &mut Deepseek41Model, ids: &[u32]) -> Result<(), GateError> {
        let p = ids.len();
        m.reset()?;
        let mut head = {
            let (gpu, w, b) = m.body_parts(NAME)?;
            bloomery_gpu::head::Head::new(gpu, w, b.head_eps())?
        };
        // Per (kind, layer), in seam order: each position's streams.
        let mut rec: Vec<(SeamKey, Vec<Vec<f32>>)> = Vec::new();
        // Position 0's attention buffers per layer, for a one-token batch.
        let mut taps0: Vec<Vec<(&'static str, Vec<f32>)>> = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            let mut i = 0;
            finite::observed_step(m, &mut head, id, pos as u32, &mut |gpu, seam, v| {
                if pos == 0
                    && let body::Seam::Attn { taps, .. } = seam
                {
                    taps0.push(taps_host(gpu, taps)?);
                }
                let key = match seam {
                    body::Seam::Engram { layer, .. } => (body::BatchSeamKind::Engram, *layer),
                    body::Seam::Attn { layer, .. } => (body::BatchSeamKind::Attn, *layer),
                    body::Seam::Ffn { layer, .. } => (body::BatchSeamKind::Ffn, *layer),
                };
                if pos == 0 {
                    rec.push((key, Vec::with_capacity(p)));
                }
                match rec.get_mut(i) {
                    Some((k, vs)) if *k == key => vs.push(v.to_vec()),
                    _ => {
                        return Err(GpuError::State {
                            what: NAME,
                            missing: "the same seams at every position",
                        });
                    }
                }
                i += 1;
                Ok(())
            })?;
        }
        m.reset()?;
        let mut i = 0;
        let mut off = 0usize;
        body::prefill_observed(m, ids, &mut |gpu, seam| {
            gpu.stream().synchronize()?;
            let n4 = seam.streams.len() / crate::gate::T_MAX_TOKENS;
            let mut host = vec![0.0f32; seam.streams.len()];
            seam.streams.copy_to_host(gpu.stream(), &mut host)?;
            let Some((key, want)) = rec.get(i) else {
                return Err(GpuError::State {
                    what: NAME,
                    missing: "a recorded seam for every batch seam",
                });
            };
            i += 1;
            let mut bad = Vec::new();
            let mut worst = 0.0f32;
            for t in 0..seam.tokens {
                let pos = seam.first as usize + t;
                let got = &host[t * n4..(t + 1) * n4];
                let w = &want[pos];
                let diff = got
                    .iter()
                    .zip(w)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                if diff > 0 {
                    bad.push((pos, diff));
                    worst = got
                        .iter()
                        .zip(w)
                        .map(|(a, b)| (a - b).abs())
                        .fold(worst, f32::max);
                }
            }
            if !bad.is_empty()
                && off == 0
                && seam.tokens == 1
                && let Some(t) = &seam.attn
            {
                let layer_i = rec[..i]
                    .iter()
                    .filter(|(k, _)| k.0 == body::BatchSeamKind::Attn)
                    .count()
                    - 1;
                for ((name, got), (_, want)) in taps_host(gpu, t)?.iter().zip(&taps0[layer_i]) {
                    let diff = got
                        .iter()
                        .zip(want)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let worst = got
                        .iter()
                        .zip(want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    println!(
                        "{NAME}: seams | attention layer {} {name}: {diff} of {} differ, max |diff| {worst:e}; first values batch {:?} step {:?}",
                        seam.layer,
                        want.len(),
                        &got[..got.len().min(4)],
                        &want[..want.len().min(4)]
                    );
                }
            }
            if !bad.is_empty() && off < 6 {
                off += 1;
                println!(
                    "{NAME}: seams | {:?} layer {} ({:?} layer {}): {} of {} tokens differ {:?}, max |diff| {worst:e}",
                    seam.kind,
                    seam.layer,
                    key.0,
                    key.1,
                    bad.len(),
                    seam.tokens,
                    &bad[..bad.len().min(8)]
                );
            } else if bad.is_empty() && off == 0 {
                println!(
                    "{NAME}: seams | {:?} layer {}: {} tokens bit-equal",
                    seam.kind, seam.layer, seam.tokens
                );
            }
            Ok(())
        })?;
        Ok(())
    }

    /// The attention piece's buffers on the host, by name.
    fn taps_host(
        gpu: &bloomery_gpu::Gpu,
        t: &bloomery_gpu_deepseek41::chain::attn::AttnTaps<'_>,
    ) -> Result<Vec<(&'static str, Vec<f32>)>, GpuError> {
        let bufs: [(&'static str, &cuda_core::DeviceBuffer<f32>); 10] = [
            ("hc", t.hc),
            ("normed", t.normed),
            ("q_a", t.q_a),
            ("q_a_normed", t.q_a_normed),
            ("q", t.q),
            ("kv", t.kv),
            ("kv_row", t.kv_row),
            ("y", t.y),
            ("wo_a", t.wo_a),
            ("out", t.out),
        ];
        bufs.iter()
            .map(|&(name, b)| {
                let mut host = vec![0.0f32; b.len()];
                b.copy_to_host(gpu.stream(), &mut host)?;
                Ok((name, host))
            })
            .collect()
    }

    /// The batch's streams buffer holds this many tokens: [`body::T_MAX`].
    pub(crate) const T_MAX_TOKENS: usize = body::T_MAX;

    /// The oracle run (module doc): a [`Snap`] per position of `wanted`.
    fn oracle(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        ids: &[u32],
        steps: usize,
        wanted: &[usize],
    ) -> Result<BTreeMap<usize, Snap>, GateError> {
        m.reset()?;
        let mut taps = Md5::new();
        let mut out: BTreeMap<usize, Snap> = BTreeMap::new();
        for (p, &id) in ids.iter().enumerate().take(steps) {
            m.step(&[id])?;
            if let Some(s) = out.get_mut(&p) {
                s.next = Some(bits(&m.logits()?));
            }
            {
                let (gpu, _, b) = m.body_parts(NAME)?;
                let f = b.read_features(gpu, 1)?;
                if f.pos as usize != p {
                    return Err(
                        format!("{NAME}: the features after step {p} name {}", f.pos).into(),
                    );
                }
                taps.update(bytes_of(f.values));
            }
            let at = p + 1;
            if wanted.contains(&at) {
                let mut s = live(m)?;
                s.taps = taps.clone().finish();
                s.logits = bits(&m.logits()?);
                out.insert(at, s);
            }
        }
        // The rows a position writes once: from the run's end, below each P.
        for (&p, s) in out.iter_mut() {
            written(m, hp, p, s)?;
        }
        Ok(out)
    }

    /// One case: `parts` prefill calls over `ids[.. Σ parts]`, then one step.
    fn run_case(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        ids: &[u32],
        parts: &[usize],
    ) -> Result<Snap, GateError> {
        let mut taps = Md5::new();
        let mut at = 0;
        for &n in parts {
            let mut feed = |_: u32, rows: &[f32]| -> Result<(), GpuError> {
                taps.update(bytes_of(rows));
                Ok(())
            };
            body::prefill_with(m, &ids[at..at + n], Some(&mut feed))?;
            at += n;
        }
        let mut s = live(m)?;
        s.taps = taps.finish();
        s.logits = bits(&m.logits()?);
        written(m, hp, at, &mut s)?;
        if at < ORACLE {
            m.step(&[ids[at]])?;
            s.next = Some(bits(&m.logits()?));
        }
        Ok(s)
    }

    /// The ring and the compressor state of every layer, as they stand.
    fn live(m: &mut Deepseek41Model) -> Result<Snap, GateError> {
        let (gpu, _, b) = m.body_parts(NAME)?;
        let stream = gpu.stream();
        let mut s = Snap::default();
        for l in b.layers() {
            let st = b
                .state_mut(l)
                .ok_or_else(|| format!("{NAME}: no state for layer {l}"))?;
            s.ring.push(md5_of(stream, st.ring, None)?);
            s.state.push(match (st.values, st.scores) {
                (Some(v), Some(sc)) => {
                    let mut h = Md5::new();
                    h.update(bytes_of(&tensor_host(stream, v, None)?));
                    h.update(bytes_of(&tensor_host(stream, sc, None)?));
                    Some(h.finish())
                }
                _ => None,
            });
        }
        Ok(s)
    }

    /// The rows below position `p` every layer writes once: its shadow rows,
    /// its compressed rows and its index keys (`⌊p / ratio⌋` rows).
    fn written(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        p: usize,
        s: &mut Snap,
    ) -> Result<(), GateError> {
        let (gpu, _, b) = m.body_parts(NAME)?;
        let stream = gpu.stream();
        let (mut shadow, mut rows, mut keys) = (Vec::new(), Vec::new(), Vec::new());
        for l in b.layers() {
            let host = b
                .shadow_rows(gpu, l)?
                .ok_or_else(|| format!("{NAME}: no shadow for layer {l}"))?;
            let mut h = Md5::new();
            h.update(bytes_of(&host[..p * hp.head_dim]));
            shadow.push(h.finish());
            let ratio = hp.layers.get(l).map_or(0, |k| k.ratio() as usize);
            let st = b
                .state_mut(l)
                .ok_or_else(|| format!("{NAME}: no state for layer {l}"))?;
            let n = p.checked_div(ratio).unwrap_or(0);
            rows.push(match st.rows {
                Some(t) if n > 0 => Some(md5_of(stream, t, Some(n))?),
                Some(_) => Some(Md5::new().finish()),
                None => None,
            });
            keys.push(match st.keys {
                Some(t) if n > 0 => Some(md5_of(stream, t, Some(n))?),
                Some(_) => Some(Md5::new().finish()),
                None => None,
            });
        }
        s.shadow = shadow;
        s.rows = rows;
        s.keys = keys;
        Ok(())
    }

    /// Rows `0 .. rows` of `t` (all of them with `None`) on the host.
    fn tensor_host<T: DeviceCopy + Default + Clone>(
        stream: &cuda_core::CudaStream,
        t: &DeviceTensor<T>,
        rows: Option<usize>,
    ) -> Result<Vec<T>, GateError> {
        let n = rows.unwrap_or(t.rows()) * t.cols();
        let w = span(NAME, t.buf(), 0, n)?;
        let mut host = vec![T::default(); n];
        w.copy_to_host(stream, &mut host)?;
        Ok(host)
    }

    fn md5_of<T: DeviceCopy + Default + Clone>(
        stream: &cuda_core::CudaStream,
        t: &DeviceTensor<T>,
        rows: Option<usize>,
    ) -> Result<Digest, GateError> {
        let host = tensor_host(stream, t, rows)?;
        let mut h = Md5::new();
        h.update(bytes_of(&host));
        Ok(h.finish())
    }

    /// Every field of `got` against `want`: one line.
    fn compare(what: &str, got: &Snap, want: &Snap, t: Instant) -> bool {
        let field = |name: &str, g: &[Digest], w: &[Digest]| -> (bool, String) {
            let same = g.iter().zip(w).filter(|(a, b)| a == b).count();
            let ok = g.len() == w.len() && same == g.len();
            let first = g.iter().zip(w).position(|(a, b)| a != b);
            (
                ok,
                format!(
                    "{name} {same}/{} md5 {}{}",
                    w.len(),
                    hex(&fold(g)),
                    first.map_or(String::new(), |i| format!(" (first off at layer {i})"))
                ),
            )
        };
        let opt = |v: &[Option<Digest>]| -> Vec<Digest> { v.iter().flatten().copied().collect() };
        let shape_ok = got
            .state
            .iter()
            .map(Option::is_some)
            .eq(want.state.iter().map(Option::is_some))
            && got
                .rows
                .iter()
                .map(Option::is_some)
                .eq(want.rows.iter().map(Option::is_some))
            && got
                .keys
                .iter()
                .map(Option::is_some)
                .eq(want.keys.iter().map(Option::is_some));
        let parts = [
            field("ring", &got.ring, &want.ring),
            field("state", &opt(&got.state), &opt(&want.state)),
            field("shadow", &got.shadow, &want.shadow),
            field("rows", &opt(&got.rows), &opt(&want.rows)),
            field("keys", &opt(&got.keys), &opt(&want.keys)),
        ];
        let taps = got.taps == want.taps;
        let logits = got.logits == want.logits;
        let next = got.next == want.next;
        let ok = shape_ok && parts.iter().all(|(o, _)| *o) && taps && logits && next;
        let text: Vec<String> = parts.into_iter().map(|(_, s)| s).collect();
        println!(
            "{NAME}: case {what}: {} | taps md5 {} {} | logits[P-1] {} | next step logits {} | {:.1} s: {}",
            text.join(" | "),
            hex(&got.taps),
            verdict(taps),
            verdict(logits),
            match &got.next {
                None => "n/a (P is the oracle's last position)".to_string(),
                Some(_) => verdict(next).to_string(),
            },
            t.elapsed().as_secs_f64(),
            verdict(ok)
        );
        ok
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The bytes of `v`, as the host holds them.
    fn bytes_of<T: DeviceCopy>(v: &[T]) -> &[u8] {
        // SAFETY: `T` is a plain device-copyable value type (no padding in
        // the u16/u32/f32 the gate reads); the slice covers exactly `v`'s
        // initialized bytes and borrows `v`.
        unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
    }

    /// The md5 of the per-layer md5s, in order: a field's one printed value.
    fn fold(v: &[Digest]) -> Digest {
        let mut h = Md5::new();
        for d in v {
            h.update(d);
        }
        h.finish()
    }

    fn hex(d: &Digest) -> String {
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    type Digest = [u8; 16];

    /// MD5 (RFC 1321), streaming.
    #[derive(Clone)]
    struct Md5 {
        state: [u32; 4],
        buf: [u8; 64],
        fill: usize,
        len: u64,
    }

    impl Md5 {
        fn new() -> Md5 {
            Md5 {
                state: [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476],
                buf: [0; 64],
                fill: 0,
                len: 0,
            }
        }

        fn update(&mut self, mut data: &[u8]) {
            self.len = self.len.wrapping_add(data.len() as u64);
            if self.fill > 0 {
                let take = (64 - self.fill).min(data.len());
                self.buf[self.fill..self.fill + take].copy_from_slice(&data[..take]);
                self.fill += take;
                data = &data[take..];
                if self.fill < 64 {
                    return;
                }
                let block = self.buf;
                self.block(&block);
                self.fill = 0;
            }
            let (blocks, rest) = data.as_chunks::<64>();
            for b in blocks {
                self.block(b);
            }
            self.buf[..rest.len()].copy_from_slice(rest);
            self.fill = rest.len();
        }

        fn finish(mut self) -> Digest {
            let bits = self.len.wrapping_mul(8);
            self.update(&[0x80]);
            while self.fill != 56 {
                self.update(&[0]);
            }
            self.update(&bits.to_le_bytes());
            let mut out = [0u8; 16];
            for (o, s) in out.as_chunks_mut::<4>().0.iter_mut().zip(self.state) {
                o.copy_from_slice(&s.to_le_bytes());
            }
            out
        }

        fn block(&mut self, b: &[u8; 64]) {
            const S: [u32; 64] = [
                7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14,
                20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11,
                16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
            ];
            const K: [u32; 64] = [
                0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
                0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
                0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
                0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
                0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
                0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
                0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
                0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
                0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
                0xeb86d391,
            ];
            let mut m = [0u32; 16];
            for (w, c) in m.iter_mut().zip(b.as_chunks::<4>().0) {
                *w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            let [mut a, mut bb, mut c, mut d] = self.state;
            for i in 0..64 {
                let (f, g) = match i / 16 {
                    0 => ((bb & c) | (!bb & d), i),
                    1 => ((d & bb) | (!d & c), (5 * i + 1) % 16),
                    2 => (bb ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (bb | !d), (7 * i) % 16),
                };
                let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
                a = d;
                d = c;
                c = bb;
                bb = bb.wrapping_add(f.rotate_left(S[i]));
            }
            for (s, v) in self.state.iter_mut().zip([a, bb, c, d]) {
                *s = s.wrapping_add(v);
            }
        }
    }
}
