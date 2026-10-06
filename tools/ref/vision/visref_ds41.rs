//! `visref_ds41` — the engine's side of the V4.1 vision fork comparison (`tools/ref/vision/`): every case of the
//! fork's set (`refset` family `visref-deepseek41`, written by `just dump-ref-visref`) fed to the engine on the gate
//! placement — the set's ids, and for an image case its span's bf16 rows and kinds (`body::prefill_media`) — then
//! stepped through the fork's greedy answer, teacher-forced. At each of the answer's first `logits` positions the
//! engine's logits row is scored against the fork's: KL(P_fork ‖ Q_engine) over the whole vocabulary in f64, and
//! whether the two argmaxes (lowest id on a tie, the fork harness's rule) agree.
//!
//!     visref_ds41 [--set DIR] [--free N] [--case NAME]...
//!
//! The band is the text controls' (the `text` and `prose` cases: the same two engines, the same harness, no image),
//! taken by case: an image case is inside it when its mean KLD is at most the largest control case's mean and its
//! top-1 agreement at least the smallest control case's. The positions of one answer are not independent draws, so
//! the band does not pool them; the i.i.d. edge (the 99th percentile of the mean of n positions drawn with
//! replacement from the pooled controls, RESAMPLES draws from a fixed seed) is printed beside it as the narrower
//! figure, and judges nothing. `--free N` (default:
//! the fork's own budget, the case's `gen`; 0: none) then decodes each image case greedily from its prompt, the
//! engine's answer beside the fork's `.answer.txt`.
//!
//! Lines: `visref pos` per scored position, `visref case` per case, `visref band` per image case, `visref free` and
//! the answer's text between `visref text … begin` and `… end`. It exits 0 when every case ran: the band lines say
//! inside or outside, and the comparison is read, not gated.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!("visref_ds41: built without the `deepseek41` feature.");
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("visref_ds41", run::run())
}

#[cfg(feature = "deepseek41")]
mod run {
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model, MediaKind, MediaSpan};
    use bloomery_gpu_gates::{GateError, data_dir};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, CED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, PREFILL_GROUP,
        R8, STEP_STATS,
    };
    use gguf::Split;
    use model::placement::workstation;
    use refset::arch::deepseek41v::{VISREF, VISREF_SET};
    use refset::visref::{Case, VisrefSet};

    const NAME: &str = "visref_ds41";
    /// Bootstrap draws of the band.
    const RESAMPLES: usize = 20_000;

    /// One case's scored positions: per position the KLD and whether the argmaxes agree.
    struct Scored {
        kld: Vec<f64>,
        agree: Vec<bool>,
    }

    /// `l`'s log-softmax in f64; a row holding a NaN or an infinity is refused by name.
    fn log_softmax(l: &[f32], what: &str) -> Result<Vec<f64>, GateError> {
        if let Some(i) = l.iter().position(|v| !v.is_finite()) {
            return Err(format!("{what}: logit {i} is {}", l[i]).into());
        }
        let m = l
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
        let s: f64 = l.iter().map(|&v| (f64::from(v) - m).exp()).sum();
        let ln = s.ln();
        Ok(l.iter().map(|&v| f64::from(v) - m - ln).collect())
    }

    /// The first index of the largest value: the fork harness's argmax.
    fn argmax(l: &[f32]) -> u32 {
        let mut best = 0;
        for (i, &v) in l.iter().enumerate() {
            if v > l[best] {
                best = i;
            }
        }
        best as u32
    }

    fn kinds(types: &[u8]) -> Vec<MediaKind> {
        types
            .iter()
            .map(|t| match t {
                0 => MediaKind::Start,
                1 => MediaKind::Image,
                2 => MediaKind::NewLine,
                _ => MediaKind::End,
            })
            .collect()
    }

    /// Reset, feed case `c`'s ids with its span, return the first answer position's logits row.
    fn prompt(m: &mut Deepseek41Model, set: &VisrefSet, c: &Case) -> Result<Vec<f32>, GateError> {
        m.reset()?;
        let ids = set.ids(c)?;
        if c.span_len == 0 {
            body::prefill_media(m, &ids, &[])?;
        } else {
            let rows = set.rows(c)?;
            let kinds = kinds(&set.types(c)?);
            let span = MediaSpan {
                at: c.span_at..c.span_at + c.span_len,
                rows: &rows,
                kinds: &kinds,
            };
            body::prefill_media(m, &ids, &[span])?;
        }
        Ok(m.logits()?)
    }

    /// Case `c` teacher-forced through the fork's answer, each kept position scored.
    fn score(m: &mut Deepseek41Model, set: &VisrefSet, c: &Case) -> Result<Scored, GateError> {
        let answer = set.answer(c)?;
        let fork = set.logits(c)?;
        let v = set.n_vocab;
        let mut row = prompt(m, set, c)?;
        let mut s = Scored {
            kld: Vec::with_capacity(c.logits),
            agree: Vec::with_capacity(c.logits),
        };
        for k in 0..c.logits {
            if k > 0 {
                m.step(&[answer[k - 1]])?;
                row = m.logits()?;
            }
            if row.len() != v {
                return Err(format!(
                    "{}: the engine's logits row holds {} values, the set's {v}",
                    c.name,
                    row.len()
                )
                .into());
            }
            let f = &fork[k * v..(k + 1) * v];
            let lp = log_softmax(f, &format!("{} fork position {k}", c.name))?;
            let lq = log_softmax(&row, &format!("{} engine position {k}", c.name))?;
            let kld: f64 = lp.iter().zip(&lq).map(|(p, q)| p.exp() * (p - q)).sum();
            let entropy: f64 = -lp.iter().map(|p| p.exp() * p).sum::<f64>();
            let (ours, theirs) = (argmax(&row), argmax(f));
            println!(
                "visref pos case={} k={k} fork={theirs} engine={ours} agree={} kld={kld:.6e} fork_entropy={entropy:.4} fork_p={:.4}",
                c.name,
                u8::from(ours == theirs),
                lp[theirs as usize].exp()
            );
            s.kld.push(kld);
            s.agree.push(ours == theirs);
        }
        Ok(s)
    }

    /// The `q` quantile of `v` (nearest rank).
    fn quantile(v: &mut [f64], q: f64) -> f64 {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * q).round() as usize]
    }

    /// The band for a case of `n` positions: the 99th percentile of the mean KLD and the 1st percentile of the
    /// agreement over `n` positions drawn with replacement from the pooled controls.
    fn band(kld: &[f64], agree: &[bool], n: usize) -> (f64, f64) {
        let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ n as u64;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % kld.len() as u64) as usize
        };
        let (mut means, mut agrees) =
            (Vec::with_capacity(RESAMPLES), Vec::with_capacity(RESAMPLES));
        for _ in 0..RESAMPLES {
            let (mut s, mut a) = (0.0, 0usize);
            for _ in 0..n {
                let i = next();
                s += kld[i];
                a += usize::from(agree[i]);
            }
            means.push(s / n as f64);
            agrees.push(a as f64 / n as f64);
        }
        (quantile(&mut means, 0.99), quantile(&mut agrees, 0.01))
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            CED,
            PREFILL_GROUP,
            ENGRAM_HELPER,
            STEP_STATS,
            CARD_BUDGET,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        let mut args = std::env::args().skip(1);
        let (mut dir, mut free, mut only) = (VISREF.path(VISREF_SET), None, Vec::new());
        while let Some(a) = args.next() {
            let mut val = || {
                args.next()
                    .ok_or_else(|| format!("{NAME}: {a} takes a value"))
            };
            match a.as_str() {
                "--set" => dir = data_dir().join(val()?),
                "--free" => free = Some(val()?.parse::<usize>()?),
                "--case" => only.push(val()?),
                _ => {
                    return Err(format!(
                        "usage: {NAME} [--set DIR] [--free N] [--case NAME]...; got {a}"
                    )
                    .into());
                }
            }
        }
        let set = VisrefSet::open(&dir, &VISREF)?;
        for name in &only {
            set.case(name)?;
        }
        let cfg = body::OpenCfg::from_levers(&levers)?;
        let path = workstation::model_v41();
        let tok = tokenizer::Tokenizer::from_gguf(&path).map_err(|e| format!("{path}: {e}"))?;
        let t = std::time::Instant::now();
        let mut m = body::open(
            Split::open(&path).map_err(|e| format!("open {path}: {e}"))?,
            workstation::plan_gate,
            usize::try_from(workstation::CTX_MAX)?,
            &cfg,
        )?;
        m.set_mode(StepMode::Graph);
        body::prepare_prefill(&mut m)?;
        body::prepare_media(&mut m)?;
        m.capture_step()?;
        println!(
            "{NAME}: loaded in {:.1} s; set {} ({} cases, fork {}, rows {}), keep {}",
            t.elapsed().as_secs_f64(),
            dir.display(),
            set.cases.len(),
            set.build.as_deref().unwrap_or("-"),
            set.rows_checkpoint.as_deref().unwrap_or("-"),
            set.keep
        );
        let cases: Vec<&Case> = set
            .cases
            .iter()
            .filter(|c| only.is_empty() || only.contains(&c.name))
            .collect();
        let mut scored = Vec::new();
        for c in &cases {
            let s = score(&mut m, &set, c)?;
            let n = s.kld.len();
            let mean = s.kld.iter().sum::<f64>() / n as f64;
            let agree = s.agree.iter().filter(|&&a| a).count();
            let mut sorted = s.kld.clone();
            println!(
                "visref case={} kind={} positions={n} kld_mean={mean:.6e} kld_p50={:.6e} kld_p90={:.6e} kld_max={:.6e} top1={agree}/{n}",
                c.name,
                c.kind,
                quantile(&mut sorted, 0.5),
                quantile(&mut sorted, 0.9),
                quantile(&mut sorted, 1.0)
            );
            scored.push((*c, s, mean, agree));
        }
        let (mut pool_k, mut pool_a) = (Vec::new(), Vec::new());
        for (c, s, ..) in &scored {
            if c.kind != "image" {
                pool_k.extend_from_slice(&s.kld);
                pool_a.extend_from_slice(&s.agree);
            }
        }
        if pool_k.is_empty() {
            println!("visref band: no control case ran, no band");
        } else {
            let pooled = pool_k.iter().sum::<f64>() / pool_k.len() as f64;
            println!(
                "visref controls positions={} kld_mean={pooled:.6e} top1={}/{}",
                pool_k.len(),
                pool_a.iter().filter(|&&a| a).count(),
                pool_a.len()
            );
            let controls = scored.iter().filter(|(c, ..)| c.kind != "image");
            let hi = controls
                .clone()
                .map(|(_, _, mean, _)| *mean)
                .fold(0.0, f64::max);
            let lo = controls
                .map(|(_, s, _, agree)| *agree as f64 / s.kld.len() as f64)
                .fold(1.0, f64::min);
            println!("visref band kld_mean<={hi:.6e} top1>={lo:.4} (the control cases' extremes)");
            for (c, s, mean, agree) in &scored {
                if c.kind != "image" {
                    continue;
                }
                let n = s.kld.len();
                let (iid_hi, iid_lo) = band(&pool_k, &pool_a, n);
                let top1 = *agree as f64 / n as f64;
                let inside = *mean <= hi && top1 >= lo;
                println!(
                    "visref band case={} positions={n} kld_mean={mean:.6e} band_kld_mean<={hi:.6e} top1={top1:.4} \
                     band_top1>={lo:.4} {} iid_kld_mean<={iid_hi:.6e} iid_top1>={iid_lo:.4} ratio={:.2}",
                    c.name,
                    if inside { "inside" } else { "outside" },
                    mean / hi
                );
            }
        }
        for c in cases.iter().filter(|c| c.kind == "image") {
            let n = free.unwrap_or(c.gen_max);
            if n == 0 {
                continue;
            }
            let mut next = argmax(&prompt(&mut m, &set, c)?);
            let mut ids = vec![next];
            while ids.len() < n && !tok.eog().contains(&next) {
                m.step(&[next])?;
                next = argmax(&m.logits()?);
                ids.push(next);
            }
            let fork = set.answer(c)?;
            let same = ids.iter().zip(&fork).take_while(|(a, b)| a == b).count();
            println!(
                "visref free case={} tokens={} eog={} same_prefix={same} fork_tokens={}",
                c.name,
                ids.len(),
                u8::from(tok.eog().contains(&next)),
                fork.len()
            );
            println!(
                "visref text case={} begin\n{}\nvisref text case={} end",
                c.name,
                tok.decode(&ids),
                c.name
            );
        }
        Ok(())
    }
}
