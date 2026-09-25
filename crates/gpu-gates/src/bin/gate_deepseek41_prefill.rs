//! GPU gate for the V4.1 prompt batch (`body::prefill`) on the gate
//! placement (`workstation::plan_gate`): a batched prefill of `P` ids leaves
//! the model where `P` decode steps over the same ids leave it, bit for bit,
//! in everything a later step or cut reads.
//!
//! The ids are the first [`ORACLE`] + 1 of `$BLOOMERY_DATA/engram/corpus-prose.ids`
//! (V4.1 text ids, one per line). The feature tap is attached with the DSpark
//! draft's `target_layers` (`$BLOOMERY_DSPARK_MODEL`, header only), and each
//! call keeps the features of its last `attention.sliding_window` positions
//! of that draft — the rows a draft's window-only layers hold.
//!
//! The batch runs the CED triangle (`body::Need`): above the last layer that
//! owns a compressor, a layer runs only the positions a later reader needs,
//! so it writes the ring shadow rows of those positions alone. The contract
//! is therefore not "every shadow row equals the steps'" but: every row a
//! layer's need says it wrote equals the steps' row, and no cut
//! `Body::keep_point` grants restores a row outside those — what the batch
//! leaves unwritten is never claimed.
//!
//! - **Attention count.** Before anything else, two direct launches of the
//!   attention with a visible count past its source's rows — window keys past
//!   the ring's, compressed rows past the stream's — each raise the fault word
//!   at `attn_count`; a launch within the rows raises nothing.
//! - **Oracle, one run.** From a reset, the ids decode-stepped one at a time
//!   through the graph, each position's features read after its step (an md5
//!   per position). After the step at each case's position `P − 1` the gate
//!   records every layer's window ring and compressor state and the head's
//!   logits; after the step at `P`, the logits again. The rows a position
//!   writes once and no later step touches are read from the run's end: each
//!   layer's shadow rows (an md5 per row), compressed rows and index keys
//!   (rows below each case's `⌊P / ratio⌋`).
//! - **Cases.** For each `P` of [`CASES`] and each split of [`SPLITS`] (its
//!   parts as consecutive prefill calls, each call its own triangle): reset,
//!   the calls, then the ring, state, compressed rows, keys and logits
//!   against the oracle; each layer's shadow rows over the positions the
//!   calls' needs say it wrote; the kept features (positions and rows); every
//!   cut `0 < k <= P` that `keep_point` grants restores only written rows;
//!   then one decode step with `ids[P]`, its logits against the oracle's
//!   position `P`. The splits are 700 + 400, and two whose second call sits
//!   on either side of the length where the triangle starts to cut layer 20's
//!   block (8 + 128 · 19 = 2440 positions from an aligned end): 1800 + 1000
//!   (the total past it, the second call short of it) and 300 + 2700.
//! - **Wide taps.** `P` = 1100 keeping the features of 300 positions: the
//!   tapped layers' blocks widen past the window the triangle gives them.
//! - **Rollback.** After a prefill of 1100 ids: a cut inside the call is
//!   refused by name (`keep_point` grants less), the call's own tail is
//!   granted, and stepping from there gives the oracle's logits at 1099 and
//!   1100.
//! - **Take back.** After a prefill of 700 ids, a call of 400 more whose
//!   feature reader fails once the batch has run: the call is taken back to
//!   700, and the same call again gives the oracle's logits at 1099 and 1100.
//!
//! `--cases a,b,…` runs those `P` only (the oracle then stops at the largest
//! one's `P + 1`); `--split` / `--no-split` turns the splits on or off, and
//! `--no-extra` the wide-taps and rollback cases: the FAIL-first runs use a
//! short subset.
//!
//! `--seams P` is the locator, not a verdict: `P` eager steps with every
//! seam's streams read back (the finite probe's `observed_step`), then a
//! batch of the same ids with every seam read back (`body::prefill_observed`),
//! and one line per seam — engram, attention, MoE of each layer — with the
//! tokens whose streams differ among those the batch ran and the largest
//! difference, up to the first seams that differ.

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
    use std::ops::Range;
    use std::time::Instant;

    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu::{DeviceTensor, FaultSite, GpuError};
    use bloomery_gpu_deepseek41::attn::{self, AttnArgs, AttnKernels, LATENT};
    use bloomery_gpu_deepseek41::body::{self, CedState, Deepseek41Model, Need};
    use bloomery_gpu_deepseek41::span::span;
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, verdict};
    use cuda_core::{DeviceBuffer, DeviceCopy};
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::placement::workstation;

    use crate::{dspark, finite};

    const NAME: &str = "gate_deepseek41_prefill";
    /// Positions the oracle steps at most: the longest case and one more.
    const ORACLE: usize = 4097;
    /// The longest case.
    const P_MAX: usize = ORACLE - 1;
    /// The prompt lengths: one chunk, one ubatch's chunk seams, the ring's
    /// wrap, the ubatch seam, and several ubatches.
    const CASES: [usize; 12] = [1, 2, 5, 127, 128, 129, 511, 512, 513, 1100, 2600, 4096];
    /// The splits: `ids[.. a + b]` as two prefill calls (module doc).
    const SPLITS: [(usize, usize); 3] = [(700, 400), (1800, 1000), (300, 2700)];
    /// The wide-taps case: `P` and the features it keeps.
    const WIDE: (usize, usize) = (1100, 300);
    /// The rollback case's `P` and the refused cut inside it.
    const CUT: (usize, usize) = (1100, 600);

    struct Args {
        cases: Vec<usize>,
        split: bool,
        extra: bool,
        seams: Option<usize>,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_prefill [--cases P,P,…] [--split|--no-split] \
                             [--no-extra] [--seams P]";
        let mut a = Args {
            cases: CASES.to_vec(),
            split: true,
            extra: true,
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
                "--no-extra" => a.extra = false,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if a.cases.iter().any(|&p| p == 0 || p > P_MAX) {
            return Err(format!("cases {:?}: each in 1..={P_MAX}", a.cases).into());
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
        /// Per layer, rows below `P`: the compressed rows' and the index
        /// keys' md5 (None without).
        rows: Vec<Option<Digest>>,
        keys: Vec<Option<Digest>>,
        /// The logits after position `P − 1`, and after the step at `P`.
        logits: Vec<u32>,
        next: Option<Vec<u32>>,
    }

    /// The oracle's per-position records, from the run's end: per layer an
    /// md5 per shadow row, and an md5 per position's features.
    struct Rows {
        shadow: Vec<Vec<Digest>>,
        taps: Vec<Digest>,
    }

    /// What a case's calls wrote and kept: per layer the shadow rows its
    /// needs name, and the kept features, position by position.
    struct Case {
        snap: Snap,
        /// Per layer, the md5 of its written rows' md5s.
        shadow: Vec<Digest>,
        written: Vec<Vec<Range<usize>>>,
        /// A cut `keep_point` grants whose restore reads an unwritten row.
        unclaimed: Option<usize>,
        taps: Vec<(usize, Digest)>,
        taps_want: Vec<usize>,
        needs: Vec<Need>,
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
        let ced = m.body(NAME)?.ced();
        let mut pass = count_cases(&mut m)?;
        let splits: Vec<(usize, usize)> = if args.split { SPLITS.to_vec() } else { vec![] };
        let top = args
            .cases
            .iter()
            .copied()
            .chain(splits.iter().map(|&(a, b)| a + b))
            .chain(args.extra.then_some(WIDE.0.max(CUT.0)))
            .max()
            .unwrap_or(1);
        let steps = (top + 1).min(ORACLE);
        let ids = corpus(ORACLE + 1)?;
        println!(
            "{NAME}: loaded in {:.1} s; ced={ced}; batch buffers {} B; ids corpus-prose.ids[..{}] \
             first {:?}; tap layers {layers:?}, draft window {}; oracle {steps} steps; cases {:?} \
             splits {splits:?} extra {}",
            t.elapsed().as_secs_f64(),
            m.body(NAME)?.batch_bytes(),
            ORACLE + 1,
            &ids[..4],
            dhp.window,
            args.cases,
            args.extra
        );
        if ced != CedState::On {
            println!("{NAME}: the triangle is {ced}; the gate pins it on: FAIL");
            pass = false;
        }
        let mut wanted: Vec<usize> = args.cases.clone();
        wanted.extend(splits.iter().map(|&(a, b)| a + b));
        if args.extra {
            wanted.extend([WIDE.0, CUT.0]);
        }
        wanted.sort_unstable();
        wanted.dedup();
        let t = Instant::now();
        let (oracle, rows) = oracle(&mut m, &hp, &ids, steps, &wanted)?;
        println!(
            "{NAME}: oracle {steps} steps in {:.1} s",
            t.elapsed().as_secs_f64()
        );
        let mut runs: Vec<(String, Vec<usize>, usize)> = args
            .cases
            .iter()
            .map(|&p| (format!("P={p}"), vec![p], dhp.window))
            .collect();
        runs.extend(
            splits
                .iter()
                .map(|&(a, b)| (format!("split {a}+{b}"), vec![a, b], dhp.window)),
        );
        if args.extra {
            runs.push((
                format!("wide taps P={} window={}", WIDE.0, WIDE.1),
                vec![WIDE.0],
                WIDE.1,
            ));
        }
        for (what, parts, window) in &runs {
            let t = Instant::now();
            m.reset()?;
            let case = run_case(&mut m, &hp, &ids, parts, *window, &rows)?;
            let p: usize = parts.iter().sum();
            pass &= compare(what, &case, &oracle[&p], &rows, t);
        }
        if args.extra {
            pass &= rollback_case(&mut m, &ids, &oracle)?;
            pass &= take_back_case(&mut m, &ids, &oracle)?;
        }
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// The attention count cases (module doc): the fault each launch leaves
    /// in the word, cleared after each.
    fn count_cases(m: &mut Deepseek41Model) -> Result<bool, GateError> {
        let (gpu, _, _) = m.body_parts(NAME)?;
        let kernels = AttnKernels::load(gpu.context())?;
        let stream = gpu.stream();
        const HEADS: usize = 16;
        const RING: usize = 8;
        const COMP: usize = 4;
        let q = DeviceBuffer::from_host(stream, &vec![0.01f32; HEADS * LATENT])?;
        let ring: DeviceTensor<u16> = DeviceTensor::zeroed(stream, RING, LATENT)?;
        let comp: DeviceTensor<u16> = DeviceTensor::zeroed(stream, COMP, LATENT)?;
        let sinks = DeviceBuffer::from_host(stream, &[0.0f32; HEADS])?;
        let segs = attn::segments(RING, COMP);
        let mut part_v = DeviceBuffer::zeroed(stream, attn::partials_v_len(HEADS, segs))?;
        let mut part_ms = DeviceBuffer::zeroed(stream, attn::partials_ms_len(HEADS, segs))?;
        let mut y = DeviceBuffer::zeroed(stream, HEADS * LATENT)?;
        let mut ok = true;
        let cases: [(&str, [u32; 2], Option<FaultSite>); 3] = [
            ("within the rows", [RING as u32, COMP as u32], None),
            (
                "window keys past the ring",
                [RING as u32 + 1, 0],
                Some(FaultSite::AttnCount),
            ),
            (
                "compressed rows past the stream",
                [1, COMP as u32 + 1],
                Some(FaultSite::AttnCount),
            ),
        ];
        for (what, counts, want) in cases {
            gpu.clear_fault()?;
            let vis = DeviceBuffer::from_host(stream, &counts)?;
            kernels.enqueue(
                stream,
                AttnArgs {
                    q: &q,
                    window: &ring,
                    compressed: Some(&comp),
                    selected: None,
                    vis: &vis,
                    sinks: &sinks,
                    scale: 1.0,
                    tokens: 1,
                    heads: HEADS,
                    part_v: &mut part_v,
                    part_ms: &mut part_ms,
                    y: &mut y,
                    fault: gpu.unlabelled_sink(),
                },
            )?;
            stream.synchronize()?;
            let got = gpu.fault()?;
            let pass = got.and_then(|f| f.site()) == want && got.is_some() == want.is_some();
            ok &= pass;
            println!(
                "{NAME}: attention count {what} {counts:?} of {RING} ring rows and {COMP} \
                 compressed rows: fault {} (want {}): {}",
                got.map_or("none".to_string(), |f| f.to_string()),
                want.map_or("none", FaultSite::name),
                verdict(pass)
            );
        }
        gpu.clear_fault()?;
        Ok(ok)
    }

    /// The oracle run (module doc): a [`Snap`] per position of `wanted`, and
    /// the per-position records.
    fn oracle(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        ids: &[u32],
        steps: usize,
        wanted: &[usize],
    ) -> Result<(BTreeMap<usize, Snap>, Rows), GateError> {
        m.reset()?;
        let mut taps = Vec::with_capacity(steps);
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
                taps.push(md5(bytes_of(f.values)));
            }
            let at = p + 1;
            if wanted.contains(&at) {
                let mut s = live(m)?;
                s.logits = bits(&m.logits()?);
                out.insert(at, s);
            }
        }
        // The rows a position writes once: from the run's end, below each P.
        for (&p, s) in out.iter_mut() {
            written_rows(m, hp, p, s)?;
        }
        let shadow = shadow_rows(m, hp, steps)?;
        Ok((out, Rows { shadow, taps }))
    }

    /// Per layer, an md5 per shadow row of positions `0 .. n`.
    fn shadow_rows(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        n: usize,
    ) -> Result<Vec<Vec<Digest>>, GateError> {
        let (gpu, _, b) = m.body_parts(NAME)?;
        b.layers()
            .map(|l| {
                let host = b
                    .shadow_rows(gpu, l)?
                    .ok_or_else(|| format!("{NAME}: no shadow for layer {l}"))?;
                Ok(host[..n * hp.head_dim]
                    .chunks_exact(hp.head_dim)
                    .map(|r| md5(bytes_of(r)))
                    .collect())
            })
            .collect()
    }

    /// One case: `parts` prefill calls over `ids[.. Σ parts]`, each keeping
    /// the features of its last `window` positions, then one step.
    fn run_case(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        ids: &[u32],
        parts: &[usize],
        window: usize,
        rows: &Rows,
    ) -> Result<Case, GateError> {
        let mut taps: Vec<(usize, Digest)> = Vec::new();
        let mut needs = Vec::new();
        let mut at = 0;
        let width = m.body(NAME)?.feature_width();
        for &n in parts {
            let mut feed = |first: u32, rows: &[f32]| -> Result<(), GpuError> {
                for (i, r) in rows.chunks_exact(width).enumerate() {
                    taps.push((first as usize + i, md5(bytes_of(r))));
                }
                Ok(())
            };
            let feats = body::FeatureRows {
                window,
                sink: &mut feed,
            };
            body::prefill_with(m, &ids[at..at + n], Some(feats))?;
            needs.push(
                m.body(NAME)?
                    .prefill_need()
                    .cloned()
                    .ok_or_else(|| format!("{NAME}: no needs after a prefill"))?,
            );
            at += n;
        }
        let taps_want: Vec<usize> = needs.iter().flat_map(|n| n.features..n.end).collect();
        let n_layers = rows.shadow.len();
        let written: Vec<Vec<Range<usize>>> = (0..n_layers)
            .map(|i| needs.iter().map(|n| n.written(i)).collect())
            .collect();
        let mut snap = live(m)?;
        snap.logits = bits(&m.logits()?);
        written_rows(m, hp, at, &mut snap)?;
        let shadow = written_md5(&shadow_rows(m, hp, at)?, &written);
        let unclaimed = claimed(m, hp, at, &written)?;
        if at < ORACLE {
            m.step(&[ids[at]])?;
            snap.next = Some(bits(&m.logits()?));
        }
        Ok(Case {
            snap,
            shadow,
            written,
            unclaimed,
            taps,
            taps_want,
            needs,
        })
    }

    /// Per layer, the md5 of its rows' md5s over its `written` positions.
    fn written_md5(rows: &[Vec<Digest>], written: &[Vec<Range<usize>>]) -> Vec<Digest> {
        rows.iter()
            .zip(written)
            .map(|(r, ranges)| {
                let mut h = Md5::new();
                for q in ranges.iter().flat_map(Clone::clone) {
                    h.update(&r[q]);
                }
                h.finish()
            })
            .collect()
    }

    /// The first cut `0 < k <= p` that `keep_point` grants and whose restore
    /// reads a shadow row outside some layer's `written`: the rows of `k + 1
    /// − W ..= k − 1` the ring no longer holds (it holds the last `W`
    /// positions, `W` its rows).
    fn claimed(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        p: usize,
        written: &[Vec<Range<usize>>],
    ) -> Result<Option<usize>, GateError> {
        let b = m.body(NAME)?;
        let w = hp.window.min(workstation::CTX_MAX as usize);
        let mut granted = Vec::new();
        for k in 1..=p {
            if b.keep_point(k) != k {
                continue;
            }
            granted.push(k);
            let restored = (k + 1).saturating_sub(w)..k.min(p.saturating_sub(w));
            for (l, ranges) in written.iter().enumerate() {
                let covered = restored
                    .clone()
                    .all(|q| ranges.iter().any(|r| r.contains(&q)));
                if !covered {
                    println!(
                        "{NAME}: layer {l}: cut {k} restores {restored:?}, written {ranges:?}"
                    );
                    return Ok(Some(k));
                }
            }
        }
        println!(
            "{NAME}: keep_point grants {} of the cuts 1..={p}: {}",
            granted.len(),
            runs_of(&granted)
        );
        Ok(None)
    }

    /// `v` (increasing) as runs `a..=b`, the middle elided past twelve.
    fn runs_of(v: &[usize]) -> String {
        let mut out: Vec<String> = Vec::new();
        let mut i = 0;
        while i < v.len() {
            let mut j = i;
            while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
                j += 1;
            }
            out.push(if i == j {
                v[i].to_string()
            } else {
                format!("{}..={}", v[i], v[j])
            });
            i = j + 1;
        }
        if out.len() > 12 {
            let tail = out.split_off(out.len() - 6);
            out.truncate(3);
            out.push("…".to_string());
            out.extend(tail);
        }
        out.join(",")
    }

    /// The take-back case (module doc): a call that fails after its batch ran
    /// leaves the model where it found it, and the same call again gives the
    /// oracle's logits.
    fn take_back_case(
        m: &mut Deepseek41Model,
        ids: &[u32],
        oracle: &BTreeMap<usize, Snap>,
    ) -> Result<bool, GateError> {
        let t = Instant::now();
        let (a, b) = SPLITS[0];
        m.reset()?;
        body::prefill(m, &ids[..a])?;
        let mut refuse = |_: u32, _: &[f32]| -> Result<(), GpuError> {
            Err(GpuError::State {
                what: NAME,
                missing: "a reader that takes the rows (the take-back case's planted failure)",
            })
        };
        let rows = body::FeatureRows {
            window: 1,
            sink: &mut refuse,
        };
        let failed = body::prefill_with(m, &ids[a..a + b], Some(rows));
        let (pos, len) = (m.pos() as usize, m.body(NAME)?.history().len());
        let back = failed.is_err() && pos == a && len == a;
        body::prefill(m, &ids[a..a + b])?;
        let at_end = bits(&m.logits()?) == oracle[&(a + b)].logits;
        m.step(&[ids[a + b]])?;
        let next = Some(bits(&m.logits()?)) == oracle[&(a + b)].next;
        let ok = back && at_end && next;
        println!(
            "{NAME}: case take back {a}+{b}: the second call failed ({}), the model stands at \
             {pos} with {len} tokens {} | the call again: logits[P-1] {} | next step logits {} | \
             {:.1} s: {}",
            failed
                .err()
                .map_or("no error".to_string(), |e| e.to_string()),
            verdict(back),
            verdict(at_end),
            verdict(next),
            t.elapsed().as_secs_f64(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The rollback case (module doc).
    fn rollback_case(
        m: &mut Deepseek41Model,
        ids: &[u32],
        oracle: &BTreeMap<usize, Snap>,
    ) -> Result<bool, GateError> {
        let t = Instant::now();
        let (p, inside) = CUT;
        m.reset()?;
        body::prefill(m, &ids[..p])?;
        let kept_inside = m.body(NAME)?.keep_point(inside);
        let refused = match m.rollback(inside as u32) {
            Err(e) => {
                let text = e.to_string();
                println!("{NAME}: rollback to {inside} after a prefill of {p}: refused: {text}");
                kept_inside < inside && text.contains("keep_point")
            }
            Ok(()) => {
                println!("{NAME}: rollback to {inside} after a prefill of {p}: granted");
                false
            }
        };
        let tail = m.body(NAME)?.keep_point(p - 1);
        m.rollback(tail as u32)?;
        for &id in &ids[tail..p] {
            m.step(&[id])?;
        }
        let at_end = bits(&m.logits()?) == oracle[&p].logits;
        m.step(&[ids[p]])?;
        let next = Some(bits(&m.logits()?)) == oracle[&p].next;
        let ok = refused && tail > inside && at_end && next;
        println!(
            "{NAME}: case rollback P={p}: cut {inside} refused (keep_point {kept_inside}) {} | \
             tail cut {tail} granted, stepped to {p}: logits[P-1] {} | next step logits {} | \
             {:.1} s: {}",
            verdict(refused),
            verdict(at_end),
            verdict(next),
            t.elapsed().as_secs_f64(),
            verdict(ok)
        );
        Ok(ok)
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

    /// The rows below position `p` every layer writes once: its compressed
    /// rows and its index keys (`⌊p / ratio⌋` rows).
    fn written_rows(
        m: &mut Deepseek41Model,
        hp: &Hparams,
        p: usize,
        s: &mut Snap,
    ) -> Result<(), GateError> {
        let (gpu, _, b) = m.body_parts(NAME)?;
        let stream = gpu.stream();
        let (mut rows, mut keys) = (Vec::new(), Vec::new());
        for l in b.layers() {
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
        s.rows = rows;
        s.keys = keys;
        Ok(())
    }

    /// Every field of `got` against `want` and the oracle's rows: one line.
    fn compare(what: &str, case: &Case, want: &Snap, rows: &Rows, t: Instant) -> bool {
        let got = &case.snap;
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
        let shadow_want = written_md5(&rows.shadow, &case.written);
        let parts = [
            field("ring", &got.ring, &want.ring),
            field("state", &opt(&got.state), &opt(&want.state)),
            field("shadow(written)", &case.shadow, &shadow_want),
            field("rows", &opt(&got.rows), &opt(&want.rows)),
            field("keys", &opt(&got.keys), &opt(&want.keys)),
        ];
        let kept: Vec<usize> = case.taps.iter().map(|&(p, _)| p).collect();
        let taps =
            kept == case.taps_want && case.taps.iter().all(|&(p, d)| rows.taps.get(p) == Some(&d));
        let logits = got.logits == want.logits;
        let next = got.next == want.next;
        let claimed = case.unclaimed.is_none();
        let ok = shape_ok && parts.iter().all(|(o, _)| *o) && taps && claimed && logits && next;
        let text: Vec<String> = parts.into_iter().map(|(_, s)| s).collect();
        let reach = |n: &Need| n.layers.get(20).map_or(0, |l| l.full - n.first);
        println!(
            "{NAME}: case {what}: {} | taps {} rows from {} {} | granted cuts claimed {} | \
             logits[P-1] {} | next step logits {} | layer 20 block cut by {:?} | {:.1} s: {}",
            text.join(" | "),
            kept.len(),
            kept.first().map_or("-".to_string(), ToString::to_string),
            verdict(taps),
            verdict(claimed),
            verdict(logits),
            match &got.next {
                None => "n/a (P is the oracle's last position)".to_string(),
                Some(_) => verdict(next).to_string(),
            },
            case.needs.iter().map(reach).collect::<Vec<_>>(),
            t.elapsed().as_secs_f64(),
            verdict(ok)
        );
        ok
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
                let got = &host[(seam.at + t) * n4..(seam.at + t + 1) * n4];
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

    /// The md5 of `b`.
    fn md5(b: &[u8]) -> Digest {
        let mut h = Md5::new();
        h.update(b);
        h.finish()
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
