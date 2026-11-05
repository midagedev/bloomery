//! `lab_prune_ds41` — a lab, not a gate: what DeepSeek-V4.1-Flash loses when the coldest X percent of
//! each routed layer's experts cannot be picked. No engine path reads it and no lever carries it.
//!
//!     lab_prune_ds41 --rank FILE [--corpus prose,code,ko] [--cuts 0,25,40,50] [--n 2048]
//!                    [--text FILE] [--ids-dir DIR] [--plan]
//!
//! The mask. A prune cuts the coldest `cut(X)` experts of a layer by its rank (`FILE`, the rank file
//! `tools/ref/router-prune.py rank` writes, read by `ExpertRank`; hottest first, so the cold end is the
//! tail): `cut(X) = floor(n_expert * X / 100 + 1/2)`, 96, 154, 192 and 230 of 384 for X = 25, 40, 50 and
//! 60 (`prune::cut_count`). The router adds a per-expert bias to the scores for the selection only and
//! renormalizes the six unbiased scores it kept (`router.rs`), so pushing a cold expert's bias to
//! [`MASK`] keeps it out of the top six while the renormalization runs over the remaining picks, as the
//! router always does. The lab writes each layer's `exp_probs_b` row on the card between passes
//! (`patch_bytes`, which returns the bytes that were there): the first write returns the file's own row,
//! and every later one returns the row the lab wrote before it, else the run stops by name, so a mask
//! that writes a buffer the router does not read is not mistaken for one that does. The engine carries no
//! change, and a pass at X = 0 writes the file's own rows back through the same path.
//!
//! What is scored. Per corpus the first `--n` ids (default 2,048) are stepped one id at a time from a
//! cleared model, the logits of every position read back whole (the vocabulary's f32 row). The first pass
//! runs the unpruned model, its rows held in host memory as the truth; every cut then steps the same ids
//! and is scored position by position against it: KL(truth ‖ pruned) over the whole vocabulary in f64
//! (`prune::kld`), whether the pruned model's top logit is the truth's (`kld::top2`), and the change of
//! the negative log probability of the next id. A cut of 0 is the A/A: the file's own rows written
//! through the mask's path, and the run fails unless every position's KL is exactly 0 and every top id is
//! the truth's. A cut above 0 whose mean KL is 0 fails too: the mask did not reach the router.
//!
//! Corpora. `prose` and `code` are the first ids of `$BLOOMERY_DATA/engram/corpus-prose.ids` and
//! `corpus-code.ids` (`--ids-dir` names another directory), the id files `tools/ref/engram-corpus.sh`
//! writes, the same ids the router sets were traced on: the rank's learning slice leaves their first
//! 2,048 positions out (`router-prune.py`). `ko` is `--text` (default `tools/ref/corpus-ko.txt`) tokenized
//! by the model file's own vocabulary (`bloomery_tokenizer`, no special tokens): Kim Yu-jeong, «Dongbaek-
//! kkot» (1936), the text of <https://ko.wikisource.org/wiki/동백꽃> (the author died in 1937, the page
//! carries the `PD-old-70` licence), its page header and licence section cut. A text
//! that holds fewer than `--n` ids is refused by name.
//!
//! Placement. The serving plan `a` (`Place::A`) on the one card the gate runner puts in view, the
//! residency off, the prompt fed as steps; the experts a layer holds on the host tier are cut as the card's
//! are, the mask lives in the router. The routed expert bytes a cut drops are the file's, each routed
//! stack's bytes over its `n_expert`. `--plan` prints everything above that needs no card (the rank's
//! layers, the corpora's ids, the cut table and the predicted wall) and stops before the load.
//!
//! Output is plain text, a `lab prune` line a pass; the run ends non-zero when a check above failed.

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("lab_prune_ds41", lab::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_open.rs"]
mod ds41_open;

#[cfg(feature = "deepseek41")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "deepseek41")]
mod lab {
    use std::collections::BTreeSet;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use app::Session;
    use bloomery_gpu_deepseek41::body::{Body, OpenCfg, PrefillMode};
    use bloomery_gpu_deepseek41::router::{N_EXPERT, N_USED};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::kld::top2;
    use bloomery_gpu_gates::prune::{Stat, cut_count, kld, lse, nll};
    use bloomery_gpu_gates::{
        GateError, checks_failed, data_dir, patch_bytes, ref_model_path, split_f32, verdict,
    };
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, ENGRAM_HELPER, HOST_LOCK, HOST_POPULATE, R8,
    };
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::names;
    use model::placement::ExpertRank;
    use runtime::{Out, Target, Want};
    use tokenizer::Tokenizer;

    use crate::ds41_open::open;

    const NAME: &str = "lab_prune_ds41";

    /// What a cold expert's selection bias is moved by: finite, so the router's finiteness check holds
    /// (an infinite value is refused there), and past any score plus bias a kept expert can have.
    const MASK: f32 = -1.0e30;

    /// One layer's bias row in bytes.
    const ROW_BYTES: usize = N_EXPERT * 4;

    /// The ids a pass steps by default.
    const N_DEFAULT: usize = 2048;

    /// The fewest ids a pass may step: the second-half mean and the percentiles need a sample.
    const N_MIN: usize = 64;

    /// A step's wall as planned, in ms, to price the job before the load: the serving plan's decode step
    /// at depth 512 (the README's V4.1 row) plus the deeper positions, the logits row's read-back and the
    /// f64 sums. It is derived, not measured.
    const STEP_MS: f64 = 40.0;

    /// The model's load as planned, in s: the page cache holds the file.
    const LOAD_S: f64 = 180.0;

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Corpus {
        Prose,
        Code,
        Ko,
    }

    impl Corpus {
        fn name(self) -> &'static str {
            match self {
                Corpus::Prose => "prose",
                Corpus::Code => "code",
                Corpus::Ko => "ko",
            }
        }
    }

    struct Args {
        rank: PathBuf,
        corpora: Vec<Corpus>,
        cuts: Vec<usize>,
        n: usize,
        text: PathBuf,
        ids_dir: Option<PathBuf>,
        plan_only: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: lab_prune_ds41 --rank FILE [--corpus prose,code,ko] \
                             [--cuts 0,25,40,50] [--n 2048] [--text FILE] [--ids-dir DIR] [--plan]";
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut a = Args {
            rank: PathBuf::new(),
            corpora: vec![Corpus::Prose, Corpus::Code, Corpus::Ko],
            cuts: vec![0, 25, 40, 50],
            n: N_DEFAULT,
            text: PathBuf::from("tools/ref/corpus-ko.txt"),
            ids_dir: None,
            plan_only: false,
        };
        let mut seen = BTreeSet::new();
        let mut it = argv.iter().map(String::as_str);
        let mut have_rank = false;
        while let Some(flag) = it.next() {
            if !seen.insert(flag) {
                return Err(format!("{flag} given twice: {USAGE}").into());
            }
            if flag == "--plan" {
                a.plan_only = true;
                continue;
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} takes a value: {USAGE}"))?;
            match flag {
                "--rank" => {
                    a.rank = PathBuf::from(v);
                    have_rank = true;
                }
                "--corpus" => {
                    a.corpora = v
                        .split(',')
                        .map(|c| match c {
                            "prose" => Ok(Corpus::Prose),
                            "code" => Ok(Corpus::Code),
                            "ko" => Ok(Corpus::Ko),
                            _ => Err(format!("--corpus {c:?} is not prose, code or ko")),
                        })
                        .collect::<Result<_, _>>()?;
                }
                "--cuts" => {
                    a.cuts =
                        v.split(',')
                            .map(|c| {
                                c.parse::<usize>().ok().filter(|&x| x < 100).ok_or_else(|| {
                                    format!("--cuts {c:?} is not a percent below 100")
                                })
                            })
                            .collect::<Result<_, _>>()?;
                }
                "--n" => {
                    a.n =
                        v.parse().ok().filter(|&n| n >= N_MIN).ok_or_else(|| {
                            format!("--n {v:?} is not a count of at least {N_MIN}")
                        })?;
                }
                "--text" => a.text = PathBuf::from(v),
                "--ids-dir" => a.ids_dir = Some(PathBuf::from(v)),
                _ => return Err(format!("unknown flag {flag}: {USAGE}").into()),
            }
        }
        if !have_rank {
            return Err(format!("--rank FILE names the experts to cut: {USAGE}").into());
        }
        if a.corpora.is_empty()
            || a.corpora.iter().collect::<BTreeSet<_>>().len() != a.corpora.len()
        {
            return Err(format!("--corpus names no corpus or one twice: {USAGE}").into());
        }
        if a.cuts.is_empty() || a.cuts.iter().collect::<BTreeSet<_>>().len() != a.cuts.len() {
            return Err(format!("--cuts names no cut or one twice: {USAGE}").into());
        }
        Ok(a)
    }

    /// The first `n` ids of an id file (one decimal id a line), each below `vocab`.
    fn read_ids(path: &Path, n: usize, vocab: usize) -> Result<Vec<u32>, GateError> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut ids = Vec::with_capacity(n);
        for (i, line) in BufReader::new(f).lines().take(n).enumerate() {
            let line = line?;
            let id: u32 = line
                .trim()
                .parse()
                .map_err(|_| format!("{} line {}: {line:?} is not an id", path.display(), i + 1))?;
            if id as usize >= vocab {
                return Err(format!(
                    "{} line {}: id {id} is not below the vocabulary {vocab}",
                    path.display(),
                    i + 1
                )
                .into());
            }
            ids.push(id);
        }
        if ids.len() < n {
            return Err(format!("{} holds {} ids, {n} asked", path.display(), ids.len()).into());
        }
        Ok(ids)
    }

    /// The ids of every corpus asked, by the files above.
    fn corpora_ids(
        a: &Args,
        model: &Path,
        vocab: usize,
    ) -> Result<Vec<(Corpus, Vec<u32>, String)>, GateError> {
        let dir = a
            .ids_dir
            .clone()
            .unwrap_or_else(|| data_dir().join("engram"));
        let mut out = Vec::new();
        for &c in &a.corpora {
            let (ids, src) = match c {
                Corpus::Prose | Corpus::Code => {
                    let path = dir.join(format!("corpus-{}.ids", c.name()));
                    (read_ids(&path, a.n, vocab)?, path.display().to_string())
                }
                Corpus::Ko => {
                    let text = std::fs::read_to_string(&a.text)
                        .map_err(|e| format!("{}: {e}", a.text.display()))?;
                    let tok = Tokenizer::from_gguf(model)?;
                    if tok.n_vocab() != vocab {
                        return Err(format!(
                            "the tokenizer's vocabulary {} is not the model's {vocab}",
                            tok.n_vocab()
                        )
                        .into());
                    }
                    let all = tok.encode(&text, false, false);
                    if all.len() < a.n {
                        return Err(format!(
                            "{} holds {} ids once tokenized ({} bytes), {} asked",
                            a.text.display(),
                            all.len(),
                            text.len(),
                            a.n
                        )
                        .into());
                    }
                    let src = format!(
                        "{} ({} bytes, {} ids once tokenized)",
                        a.text.display(),
                        text.len(),
                        all.len()
                    );
                    (all[..a.n].to_vec(), src)
                }
            };
            out.push((c, ids, src));
        }
        Ok(out)
    }

    /// The layers the model routes by its router, checked against the rank: each is ranked, and the rank
    /// names no other layer.
    fn routed_layers(hp: &Hparams, rank: &ExpertRank) -> Result<Vec<usize>, GateError> {
        if rank.n_expert() != N_EXPERT as u64 || rank.n_used() != N_USED as u64 {
            return Err(format!(
                "{}: ranks {} experts and {} picks, the model has {N_EXPERT} and {N_USED}",
                rank.path(),
                rank.n_expert(),
                rank.n_used()
            )
            .into());
        }
        let mut layers = Vec::new();
        for (l, kind) in hp.layers.iter().enumerate() {
            if kind.hash_routed {
                return Err(format!(
                    "layer {l} routes by a token table: its experts are not picked through the \
                     selection bias a prune writes"
                )
                .into());
            }
            if kind.routed {
                if rank.ranked(l).is_none() {
                    return Err(format!("{}: no row for routed layer {l}", rank.path()).into());
                }
                layers.push(l);
            }
        }
        if let Some(extra) = rank.layers().find(|l| !layers.contains(l)) {
            return Err(
                format!("{}: ranks layer {extra}, which does not route", rank.path()).into(),
            );
        }
        Ok(layers)
    }

    /// Per routed layer, the bytes of one expert across its gate, up and down stacks, from the file.
    fn expert_bytes(split: &Split, layers: &[usize]) -> Result<Vec<u64>, GateError> {
        layers
            .iter()
            .map(|&l| {
                let mut sum = 0u64;
                for name in [
                    names::ffn_gate_exps(l),
                    names::ffn_up_exps(l),
                    names::ffn_down_exps(l),
                ] {
                    let (_, t) = split
                        .find(&name)
                        .ok_or_else(|| format!("{name} is not in the model file"))?;
                    if t.nbytes % N_EXPERT as u64 != 0 {
                        return Err(format!(
                            "{name}: {} bytes do not divide over {N_EXPERT} experts",
                            t.nbytes
                        )
                        .into());
                    }
                    sum += t.nbytes / N_EXPERT as u64;
                }
                Ok(sum)
            })
            .collect()
    }

    fn row_bytes(row: &[f32]) -> [u8; ROW_BYTES] {
        let mut b = [0u8; ROW_BYTES];
        for (c, v) in b.chunks_exact_mut(4).zip(row) {
            c.copy_from_slice(&v.to_le_bytes());
        }
        b
    }

    /// The router's selection bias on the card: the file's rows, and the row each layer holds now.
    struct Biases {
        layers: Vec<usize>,
        file: Vec<Vec<f32>>,
        held: Vec<[u8; ROW_BYTES]>,
    }

    impl Biases {
        fn new(split: &Split, layers: &[usize]) -> Result<Biases, GateError> {
            let file = layers
                .iter()
                .map(|&l| split_f32(split, &names::exp_probs_b(l), N_EXPERT))
                .collect::<Result<Vec<_>, _>>()?;
            let held = file.iter().map(|r| row_bytes(r)).collect();
            Ok(Biases {
                layers: layers.to_vec(),
                file,
                held,
            })
        }

        /// Write every routed layer's bias row: the file's own, with each `cold` expert moved by [`MASK`].
        /// The bytes the write replaces must be the row this lab wrote before (the file's, at first).
        fn write(
            &mut self,
            s: &mut Session<Body>,
            cold: Option<&[Vec<bool>]>,
        ) -> Result<(), GateError> {
            let (gpu, w, _) = s.model_mut().body_parts(NAME)?;
            for (i, &l) in self.layers.iter().enumerate() {
                let mut row = self.file[i].clone();
                if let Some(cold) = cold {
                    for (v, &c) in row.iter_mut().zip(&cold[i]) {
                        if c {
                            *v += MASK;
                        }
                    }
                }
                let new = row_bytes(&row);
                let buf = w.f32_buf(NAME, &names::exp_probs_b(l))?;
                let old = patch_bytes(gpu.stream(), buf, 0, new)?;
                if old != self.held[i] {
                    return Err(format!(
                        "layer {l}: the selection bias on the card is not the row this lab wrote \
                         before it: the buffer the lab patches is not the one it thinks"
                    )
                    .into());
                }
                self.held[i] = new;
            }
            Ok(())
        }
    }

    /// The unpruned model's rows of one corpus.
    struct Truth {
        vocab: usize,
        rows: Vec<f32>,
        lse: Vec<f64>,
        top: Vec<usize>,
    }

    /// Steps `ids` from a cleared model and hands every position's logits row to `each`; returns the
    /// pass's wall in s.
    fn pass(
        s: &mut Session<Body>,
        ids: &[u32],
        vocab: usize,
        mut each: impl FnMut(usize, &[f32]) -> Result<(), GateError>,
    ) -> Result<f64, GateError> {
        let t = Instant::now();
        s.clear()?;
        for (pos, &id) in ids.iter().enumerate() {
            let Out::Logits { row, .. } = s.step(id, Want::Logits)? else {
                return Err("a step that asked for its logits read back none".into());
            };
            if row.len() != vocab {
                return Err(format!(
                    "position {pos}: {} logits, the vocabulary {vocab}",
                    row.len()
                )
                .into());
            }
            if let Some(i) = row.iter().position(|v| !v.is_finite()) {
                return Err(format!("position {pos}: logit {i} is not finite").into());
            }
            each(pos, row)?;
        }
        Ok(t.elapsed().as_secs_f64())
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            ENGRAM_HELPER,
            CARD_BUDGET,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        let a = parse_args()?;
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let hp = Hparams::read(&split)?;
        let rank = ExpertRank::read(&a.rank)?;
        let layers = routed_layers(&hp, &rank)?;
        let per_expert = expert_bytes(&split, &layers)?;
        let total: u64 = per_expert.iter().map(|b| b * N_EXPERT as u64).sum();
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;

        println!("{NAME}: model {}", path.display());
        println!(
            "{NAME}: rank {} (sets {}, {} positions), {} routed layers {}..{}, {N_EXPERT} experts, {N_USED} picks",
            rank.path(),
            rank.sets(),
            rank.positions(),
            layers.len(),
            layers[0],
            layers[layers.len() - 1]
        );
        println!(
            "{NAME}: routed expert bytes in the file {:.2} GiB ({} B an expert a layer, layer {}'s)",
            gib(total),
            per_expert[0],
            layers[0]
        );
        let mut cold_sets: Vec<(usize, Vec<Vec<bool>>, u64)> = Vec::new();
        for &x in &a.cuts {
            let cut = cut_count(N_EXPERT, x);
            let mut mask = Vec::new();
            for &l in &layers {
                let order = rank.ranked(l).ok_or("a routed layer lost its rank row")?;
                let mut cold = vec![false; N_EXPERT];
                for &e in &order[N_EXPERT - cut..] {
                    cold[e as usize] = true;
                }
                mask.push(cold);
            }
            let dropped: u64 = per_expert.iter().map(|b| b * cut as u64).sum();
            println!(
                "{NAME}: cut X={x}% = {cut} of {N_EXPERT} experts a layer, {} of them, {:.2} GiB \
                 ({:.1} % of the routed bytes)",
                cut * layers.len(),
                gib(dropped),
                100.0 * dropped as f64 / total as f64
            );
            cold_sets.push((cut, mask, dropped));
        }
        let corpora = corpora_ids(&a, &path, hp.n_vocab)?;
        for (c, ids, src) in &corpora {
            println!(
                "{NAME}: corpus {}: {} ids from {src}, first {:?} last {:?}",
                c.name(),
                ids.len(),
                &ids[..4],
                &ids[ids.len() - 4..]
            );
        }
        let steps = corpora.len() * (1 + a.cuts.len()) * a.n;
        println!(
            "{NAME}: planned wall [derived]: {} corpora x (1 truth + {} cuts) x {} ids = {steps} steps x {STEP_MS} ms + load {LOAD_S} s = {:.0} s ({:.1} min)",
            corpora.len(),
            a.cuts.len(),
            a.n,
            steps as f64 * STEP_MS / 1000.0 + LOAD_S,
            (steps as f64 * STEP_MS / 1000.0 + LOAD_S) / 60.0
        );
        if a.plan_only {
            println!("{NAME}: --plan: stopped before the load");
            return Ok(());
        }

        let mut cfg = OpenCfg::from_levers(&levers)?;
        cfg.body.prefill = PrefillMode::Steps;
        let machine = Place::A.machine_untiered()?;
        let t0 = Instant::now();
        let mut s = open(&path, Place::A, machine, &cfg)?;
        println!(
            "{NAME}: load in {:.1} s, place a",
            t0.elapsed().as_secs_f64()
        );
        let mut biases = Biases::new(&split, &layers)?;
        drop(split);

        let mut pass_ok = true;
        for (corpus, ids, _) in &corpora {
            let vocab = hp.n_vocab;
            biases.write(&mut s, None)?;
            let mut truth = Truth {
                vocab,
                rows: Vec::with_capacity(ids.len() * vocab),
                lse: Vec::with_capacity(ids.len()),
                top: Vec::with_capacity(ids.len()),
            };
            let wall = pass(&mut s, ids, vocab, |_, row| {
                truth.rows.extend_from_slice(row);
                truth.lse.push(lse(row));
                truth.top.push(top2(row).0);
                Ok(())
            })?;
            println!(
                "{NAME}: {} truth: {} positions, unpruned, {wall:.0} s",
                corpus.name(),
                ids.len()
            );
            for (x, (cut, mask, dropped)) in a.cuts.iter().zip(&cold_sets) {
                biases.write(&mut s, Some(mask))?;
                let (mut k, mut d) = (Stat::default(), Stat::default());
                let mut same = 0usize;
                let wall = pass(&mut s, ids, vocab, |pos, row| {
                    let p = &truth.rows[pos * truth.vocab..(pos + 1) * truth.vocab];
                    let q_lse = lse(row);
                    k.push(kld(p, truth.lse[pos], row, q_lse));
                    same += usize::from(top2(row).0 == truth.top[pos]);
                    if let Some(&next) = ids.get(pos + 1) {
                        d.push(nll(row, q_lse, next) - nll(p, truth.lse[pos], next));
                    }
                    Ok(())
                })?;
                let n = k.n();
                println!(
                    "{NAME}: lab prune corpus={} X={x} cut={cut} n={n}: KLD mean {:.6} ± {:.6} p50 {:.6} \
                     p99 {:.6} max {:.6} second-half mean {:.6}; same top {:.2} %; dNLL {:+.5} ± {:.5} \
                     (PPL {:+.3} %); experts dropped {:.2} GiB; {wall:.0} s",
                    corpus.name(),
                    k.mean(),
                    k.se(),
                    k.percentile(50),
                    k.percentile(99),
                    k.max(),
                    k.mean_from(n / 2),
                    100.0 * same as f64 / n as f64,
                    d.mean(),
                    d.se(),
                    100.0 * (d.mean().exp() - 1.0),
                    gib(*dropped),
                );
                if *x == 0 {
                    let zero = k.max() == 0.0 && same == n;
                    println!(
                        "{NAME}: A/A {}: every position's KL is exactly 0 and its top id the truth's: {}",
                        corpus.name(),
                        verdict(zero)
                    );
                    pass_ok &= zero;
                } else {
                    let moved = k.mean() > 0.0;
                    println!(
                        "{NAME}: X={x} {}: the pruned model's mean KL is above 0, so the mask reached the router: {}",
                        corpus.name(),
                        verdict(moved)
                    );
                    pass_ok &= moved;
                }
            }
        }
        biases.write(&mut s, None)?;
        if !pass_ok {
            return Err(checks_failed());
        }
        println!("{NAME}: every check passed");
        Ok(())
    }
}
