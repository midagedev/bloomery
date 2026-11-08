//! `generate_kimi_k3` — Kimi K3's end-to-end decode CLI: a prompt of ids (or text, raw, no chat
//! template) walked through the decode step one id at a time or, on a placed load, as placed
//! passes (`--walk pass`), then `-n` greedy ids, printed.
//!
//!     BLOOMERY_MODEL=kimi-k3 generate_kimi_k3 [--tokens a,b,c | --prompt TEXT] [-n N]
//!                            [--ctx C] [--place a|bp|gate] [--mode eager|graph] [--plan]
//!                            [--granule BYTES] [--walk steps|pass]
//!     BLOOMERY_MODEL=kimi-k3 generate_kimi_k3 --score FILE [--score-top K] [--ctx C] [--place a|bp|gate]
//!                            [--walk steps|pass]
//!     BLOOMERY_MODEL=kimi-k3 generate_kimi_k3 --imatrix OUT --text FILE [--ctx C] [--chunks N]
//!                            [--place a|bp|gate] [--walk steps|pass]
//!
//! Defaults: `-n 8`, `--ctx 512`, `--place a`, `--mode graph`. The model is `ref_model_path()`. The
//! file is read as the plan reads it ([`PlanInputs::read_without_chat_for`]: every part the body
//! does not run is refused by name, the template's tool-call parser excepted; a K3.1 file takes
//! every `--ctx`, past its `attention.indexer.top_k` the latent layers attending the rows the
//! indexer picks) and planned on `--place` before anything loads; `--plan` ends there. A placed plan the stage card's dense trunk alone refuses
//! (`PlacementError::CardFreeFloor`) is made again with the head read on the host (no card carries
//! it); the `head on host` line on stderr says so and carries the refusal.
//!
//! `--place a` loads the whole model on one card ([`bloomery_gpu::GpuModel::open`]): a file whose
//! tensors exceed the card's free bytes is refused by name by the load. Any other placement
//! (`bp`: the stage card, the second card as the expert tier and the host and NVMe tiers behind
//! them, `gate`) loads by the plan ([`bloomery_gpu::GpuModel::open_placed`] through
//! `Body::open_placed`): the dense trunk and each layer's card experts on the stage card, the
//! embedding as a host row, the tier cards' experts under the host tier (a node's, with
//! `BLOOMERY_TIER_NODES`, served by `tier_server`). The experts a plan leaves off the cards run
//! on the host tier (`model::ops::ExpertForm::Situ`), the NVMe tier paging what the room cannot
//! hold. The prompt walk is the body's `Prompt`: its ids stepped one at a time (`--walk steps`, the
//! default), or as placed passes (`--walk pass`, a placed load only;
//! [`bloomery_gpu::GpuModel::prompt_placed`]), a K3.1 file's past
//! its indexer's top-k too. A pass load's plan is made apart from the step's:
//! the widest pass shape ([`bloomery_gpu::arch::kimi_k3::SHAPES`], rows by the latent mixers' inner
//! rows) whose bytes the stage card's room holds as the prompt reserve, the expert tier's batch at
//! its rows and the NVMe arena at its floor (`PlanInputs::with_arena_at_floor`). The
//! `prompt_pass` record after the plan's names the shape, its bytes, the room and the narrowest
//! wider shape refused. `--walk steps` plans as the step does.
//!
//! The plan is made under `BLOOMERY_CARD_BUDGET` and `BLOOMERY_CARD_MARGIN` when they are set.
//! `--granule BYTES` counts each card's uploads at that allocator granule instead of the
//! census's (the workstation's 2 MiB when the flag is absent): a fixture's experts fit the shared
//! granules the dense trunk leaves open at 2 MiB, so no budget splits them between the cards, and
//! a finer granule gives the budget a window. A value that is not a power of two of at least 4 KiB
//! is refused by name. It is a plan input only; the driver allocates as it does.
//!
//! `--score FILE` teacher-forces instead of generating: each line of FILE is
//! `KEY<TAB>prompt ids<TAB>completion ids<TAB>extra ids` (ids comma-separated; the extras one group
//! per completion position, groups joined by `|`, or empty for none). The model runs one line's
//! prompt from an empty model, then for each completion position prints, before feeding that
//! position's token, one JSON line: `key`, `pos`, `teacher` (the completion's id), `lp_teacher`
//! (its log-probability), `top` (the `--score-top` highest `[id, logprob]`, default 20) and `at`
//! (`[id, logprob]` of each extra id). Log-probabilities are the full logits row's log-softmax in
//! f64. Under `--walk pass` a line's prompt and its completion but the last id run as one call of
//! placed passes cut from the end ([`bloomery_gpu::GpuModel::score_placed`]), the completion's
//! positions read from the last pass's rows; a completion past the pass's rows is refused by name.
//! An empty prompt or completion, a malformed line or prompt + completion past `--ctx` is
//! refused by name before the model loads; an id past the vocabulary before the first row runs.
//!
//! `--imatrix OUT --text FILE` collects the importance matrix `llama-imatrix` writes, from this engine's
//! own prompt walk: FILE (plain text) is tokenized as `llama-imatrix` does (special tokens added, none
//! parsed), cut into chunks of `--ctx` ids (the remainder dropped; at least two chunks, as llama
//! requires; `--chunks N` takes the first N), each chunk's first id is the BOS where the vocabulary
//! adds one, and every chunk runs from an empty model through the prompt walk (`--walk pass` on a
//! placed load). OUT is llama's GGUF form (`bloomery_gpu::imatrix`: `<weight>.in_sum2`, `.counts`;
//! the head is not collected, as `llama-imatrix` without `--process-output`). A layer with experts on
//! the host tier is refused by name: the card holds no activation of those. One `imatrix chunk=` line
//! a chunk (its wall) and a closing `imatrix chunks=` line.
//!
//! The lines on stdout: `load` (the resident bytes, the context, seconds), `ids` (the generated
//! ids), `logits` (an FNV-1a over the last row's bits) and, with the vocabulary's pieces, `text`.
//! The plan's records go to stderr.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::record::at_main(
        "generate_kimi_k3",
        bloomery_gpu_gates::record::GENERATE_KIMI_K3,
    );
    bloomery_gpu_gates::exit_with("generate_kimi_k3", cli::run())
}

#[cfg(feature = "gpu")]
mod cli {
    use app::Prompt;
    use bloomery_gpu::arch::kimi_k3::{Body, PassShape};
    use bloomery_gpu::model::{ChainBody, StepMode};
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::generate::{Place, TierRule};
    use bloomery_gpu_gates::host_stats::{Probe, print_stats};
    use bloomery_gpu_gates::kimi_k3_pass::{self, Picked};
    use bloomery_gpu_gates::need;
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::{GateError, fnv_row, gpu_census, ref_model_path};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, CARD_MARGIN, HOST_LOCK, HOST_POPULATE, HostCfg, STEP_STATS,
        TIER_NODES,
    };
    use gguf::Split;
    use model::arch::kimi_k3::place::{PlaceError, PlanInputs, TIER_ROWS, tier_batch_at};
    use model::arch::place_hybrid::reserve_room;
    use model::placement::{Machine, PlacementError, Plan, PlanLevers};
    use tokenizer::Tokenizer;

    /// The argument after flag `name`: `None` when the flag is absent, an error when it is last.
    fn flag(args: &[String], name: &str) -> Result<Option<String>, GateError> {
        args.iter()
            .position(|a| a == name)
            .map(|i| {
                args.get(i + 1).cloned().ok_or_else(|| {
                    GateError::from(format!("generate_kimi_k3: {name} needs a value"))
                })
            })
            .transpose()
    }

    /// The model of `file` on the card this process holds, with `ctx` positions, stepping in `mode`.
    /// The body's own refusals reach the caller by name (a file past the card's free bytes before
    /// any tensor is read).
    fn open(file: Split, ctx: usize, mode: StepMode) -> Result<GpuModel<Body>, GateError> {
        let mut m = GpuModel::<Body>::open(Gpu::new()?, file, ctx)?;
        m.set_mode(mode);
        Ok(m)
    }

    /// The model of `file` as `plan` places it (card 0 the stage), stepping in `mode`, with the
    /// placed pass at `pass`'s shape when it names one. The body's own refusals reach the caller
    /// by name.
    fn open_placed(
        file: Split,
        (plan, inputs): (&Plan<'_>, &PlanInputs),
        host: HostCfg,
        mode: StepMode,
        pass: Option<PassShape>,
    ) -> Result<GpuModel<Body>, GateError> {
        let mut m = match pass {
            Some(shape) => GpuModel::<Body>::open_placed_pass(file, plan, inputs, 0, host, shape)?,
            None => GpuModel::<Body>::open_placed(file, plan, inputs, 0, host)?,
        };
        m.set_mode(mode);
        Ok(m)
    }

    /// `n` greedy ids after `prompt`, from the model's position: the argmax after the prompt
    /// stepped, then each id fed back one a step.
    fn greedy(m: &mut GpuModel<Body>, prompt: &[u32], n: usize) -> Result<Vec<u32>, GateError> {
        let mut ids = vec![Body::prompt(m, prompt)?];
        while ids.len() < n {
            let last = ids[ids.len() - 1];
            ids.push(m.step(&[last])?);
        }
        Ok(ids)
    }

    /// [`greedy`] under `BLOOMERY_STEP_STATS=1` on a placed load: the host tier read before the
    /// first generated step and after each ([`Probe`]), the `stat step` lines and the summary
    /// after the ids, and the tier's `nvtier` record after the prompt and after the steps (its
    /// counters are since load, so the decode's fills are the difference). With the host
    /// profiler on, its table of the steps alone follows on stderr, as `bloomery-decode` prints
    /// it: reset after the prompt, its wall the step loop's.
    fn greedy_probed(
        m: &mut GpuModel<Body>,
        prompt: &[u32],
        n: usize,
    ) -> Result<Vec<u32>, GateError> {
        let mut ids = vec![Body::prompt(m, prompt)?];
        print_nvtier(m)?;
        let mut probes = vec![probe(m, None)?];
        model::profile::reset();
        let t_steps = std::time::Instant::now();
        while ids.len() < n {
            let last = ids[ids.len() - 1];
            ids.push(m.step(&[last])?);
            let p = probe(m, probes.last())?;
            probes.push(p);
        }
        let steps_ns = u64::try_from(t_steps.elapsed().as_nanos()).unwrap_or(u64::MAX);
        print_stats(&probes, 0);
        print_nvtier(m)?;
        if model::profile::enabled() {
            eprintln!("{}", model::profile::report(steps_ns, "decode"));
        }
        Ok(ids)
    }

    /// A [`Probe`] of the placed body's host tier; the whole-card load has none.
    fn probe(m: &GpuModel<Body>, prev: Option<&Probe>) -> Result<Probe, GateError> {
        let placed = m
            .body("generate_kimi_k3")?
            .placed()
            .ok_or("BLOOMERY_STEP_STATS reads a placed load's host tier; --place a has none")?;
        Probe::read(placed.hybrid(), m.gpu(), prev)
    }

    /// The tier's `nvtier` record (counters since load); nothing on a load without an arena.
    fn print_nvtier(m: &GpuModel<Body>) -> Result<(), GateError> {
        let placed = m
            .body("generate_kimi_k3")?
            .placed()
            .ok_or("BLOOMERY_STEP_STATS reads a placed load's host tier; --place a has none")?;
        if let Some(r) = record::nvtier_of(placed.hybrid().nvme_tier().map(|t| &**t)) {
            r.print();
        }
        Ok(())
    }

    /// The tier cards the body serves: one, the stage card's neighbour. No sitting has shown the
    /// tier not slower, so the flag defaults to `a` and the rule never picks.
    const TIER_RULE: TierRule = TierRule {
        tiers: 1,
        break_even: None,
        basis: "",
    };

    /// The levers this CLI acts on: the card budget and margin the plan is made under, the host
    /// set's load and the expert tier's nodes.
    const ACTS_ON: [&str; 7] = [
        CARD_BUDGET,
        CARD_MARGIN,
        HOST_POPULATE,
        HOST_LOCK,
        CARD_DONTNEED,
        TIER_NODES,
        STEP_STATS,
    ];

    /// Every flag this CLI reads; any other argument is refused by name, never ignored.
    const KNOWN: [&str; 14] = [
        "--walk",
        "--tokens",
        "--prompt",
        "-n",
        "--ctx",
        "--place",
        "--mode",
        "--plan",
        "--granule",
        "--score",
        "--score-top",
        "--imatrix",
        "--text",
        "--chunks",
    ];

    /// `--imatrix`: the importance matrix of the text at `text` as `ctx`-id chunks (the first
    /// `chunks` of them, all when `None`), each from an empty model, written to `out`.
    fn imatrix(
        m: &mut GpuModel<Body>,
        tok: &Tokenizer,
        (text, out): (&str, &str),
        (ctx, chunks): (usize, Option<usize>),
        walk_pass: bool,
    ) -> Result<(), GateError> {
        let raw = std::fs::read(text).map_err(|e| format!("--text {text}: read: {e}"))?;
        let ids = tok.encode(&raw, true, false);
        if ids.len() < 2 * ctx {
            return Err(format!(
                "generate_kimi_k3: --text {text} tokenizes to {} ids; at least {} (two chunks of                  --ctx {ctx}) are needed",
                ids.len(),
                2 * ctx
            )
            .into());
        }
        let n = chunks.map_or(ids.len() / ctx, |c| c.min(ids.len() / ctx));
        if n < 2 {
            return Err(format!("generate_kimi_k3: --chunks {chunks:?}: at least 2 chunks").into());
        }
        let bos = match (tok.add_bos(), tok.specials().bos) {
            (true, Some(b)) => Some(b),
            (true, None) => {
                return Err("generate_kimi_k3: the vocabulary adds a BOS and names none".into());
            }
            (false, _) => None,
        };
        // A captured step replays its launches with the host's row counts of the capture: the walk is eager.
        m.set_mode(StepMode::Eager);
        let collector = bloomery_gpu::imatrix::attach();
        for c in 0..n {
            let mut chunk = ids[c * ctx..(c + 1) * ctx].to_vec();
            if let Some(b) = bos {
                chunk[0] = b;
            }
            m.reset()?;
            let t = std::time::Instant::now();
            if walk_pass {
                m.prompt_placed(&chunk)?;
            } else {
                Body::prompt(m, &chunk)?;
            }
            println!(
                "imatrix chunk={}/{n} tokens={ctx} wall_s={:.3}",
                c + 1,
                t.elapsed().as_secs_f64()
            );
        }
        collector.write(std::path::Path::new(out), (ctx as u32, n as u32), text)?;
        println!(
            "imatrix chunks={n} chunk_size={ctx} entries={} wrote={out}",
            collector.names().len()
        );
        Ok(())
    }

    /// One line of the `--score` file: a prompt, the completion forced after it and, per
    /// completion position, the extra ids whose log-probabilities are printed beside the top.
    struct ScoreRow {
        key: String,
        prompt: Vec<u32>,
        done: Vec<u32>,
        extra: Vec<Vec<u32>>,
    }

    /// The comma-separated ids of `field` (`what` names the field in an error).
    fn id_list(field: &str, what: &str, line: usize) -> Result<Vec<u32>, GateError> {
        if field.is_empty() {
            return Ok(Vec::new());
        }
        field
            .split(',')
            .map(|t| {
                t.trim().parse::<u32>().map_err(|e| {
                    GateError::from(format!(
                        "generate_kimi_k3: --score line {line}: {what} {t:?}: {e}"
                    ))
                })
            })
            .collect()
    }

    /// The rows of the `--score` file `path`, each checked against `ctx` (prompt + completion).
    fn read_score_rows(path: &str, ctx: usize) -> Result<Vec<ScoreRow>, GateError> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("--score {path}: read: {e}"))?;
        let mut rows = Vec::new();
        for (i, l) in text.lines().enumerate().filter(|(_, l)| !l.is_empty()) {
            let line = i + 1;
            let f: Vec<&str> = l.split('\t').collect();
            let [key, prompt, done, extra] = f[..] else {
                return Err(format!(
                    "generate_kimi_k3: --score line {line}: {} tab-separated fields, want 4 \
                     (key, prompt ids, completion ids, extra ids)",
                    f.len()
                )
                .into());
            };
            if key.is_empty()
                || !key
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b))
            {
                return Err(format!(
                    "generate_kimi_k3: --score line {line}: key {key:?} is not [A-Za-z0-9_.-]+"
                )
                .into());
            }
            let prompt = id_list(prompt, "prompt id", line)?;
            let done = id_list(done, "completion id", line)?;
            if prompt.is_empty() || done.is_empty() {
                return Err(format!(
                    "generate_kimi_k3: --score line {line} ({key}): an empty prompt or completion"
                )
                .into());
            }
            if prompt.len() + done.len() > ctx {
                return Err(format!(
                    "generate_kimi_k3: --score line {line} ({key}): {} prompt + {} completion ids \
                     exceed --ctx {ctx}",
                    prompt.len(),
                    done.len()
                )
                .into());
            }
            let extra: Vec<Vec<u32>> = if extra.is_empty() {
                vec![Vec::new(); done.len()]
            } else {
                extra
                    .split('|')
                    .map(|g| id_list(g, "extra id", line))
                    .collect::<Result<_, _>>()?
            };
            if extra.len() != done.len() {
                return Err(format!(
                    "generate_kimi_k3: --score line {line} ({key}): {} extra groups for {} \
                     completion ids",
                    extra.len(),
                    done.len()
                )
                .into());
            }
            rows.push(ScoreRow {
                key: key.to_owned(),
                prompt,
                done,
                extra,
            });
        }
        if rows.is_empty() {
            return Err(format!("generate_kimi_k3: --score {path}: no rows").into());
        }
        Ok(rows)
    }

    /// `[[id, logprob],…]` of `ids` under `logits`' log-softmax (`lse` its log-sum-exp).
    fn pairs(ids: impl Iterator<Item = u32>, logits: &[f32], lse: f64) -> String {
        let items: Vec<String> = ids
            .map(|id| format!("[{id},{}]", f64::from(logits[id as usize]) - lse))
            .collect();
        format!("[{}]", items.join(","))
    }

    /// One position's JSON line: the log-softmax over the whole row in f64, the teacher id's
    /// log-probability, the `top` highest ids and the `extra` ids'.
    fn score_line(
        (key, pos): (&str, usize),
        logits: &[f32],
        teacher: u32,
        extra: &[u32],
        top: usize,
    ) -> Result<String, GateError> {
        if let Some(i) = logits.iter().position(|x| !x.is_finite()) {
            return Err(format!(
                "generate_kimi_k3: {key} position {pos}: logit {i} is {} (no log-probability)",
                logits[i]
            )
            .into());
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max = f64::from(max);
        let sum: f64 = logits.iter().map(|&x| (f64::from(x) - max).exp()).sum();
        let lse = max + sum.ln();
        let k = top.min(logits.len());
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        let by_logit = |a: &u32, b: &u32| {
            logits[*b as usize]
                .total_cmp(&logits[*a as usize])
                .then(a.cmp(b))
        };
        idx.select_nth_unstable_by(k - 1, by_logit);
        idx.truncate(k);
        idx.sort_by(by_logit);
        Ok(format!(
            "{{\"key\":\"{key}\",\"pos\":{pos},\"teacher\":{teacher},\"lp_teacher\":{},\"top\":{},\"at\":{}}}",
            f64::from(logits[teacher as usize]) - lse,
            pairs(idx.into_iter(), logits, lse),
            pairs(extra.iter().copied(), logits, lse),
        ))
    }

    /// Refused by name: an id of `rows` past the model's vocabulary of `n_vocab` ids.
    fn check_ids(rows: &[ScoreRow], n_vocab: usize) -> Result<(), GateError> {
        for r in rows {
            let mut ids = r
                .prompt
                .iter()
                .chain(&r.done)
                .chain(r.extra.iter().flatten());
            if let Some(id) = ids.find(|&&id| id as usize >= n_vocab) {
                return Err(format!(
                    "generate_kimi_k3: {}: id {id} is past the vocabulary ({n_vocab} ids)",
                    r.key
                )
                .into());
            }
        }
        Ok(())
    }

    /// Every row of `rows` from an empty model: the prompt stepped, then each completion id fed
    /// after its position's line is printed (the last id is never fed: no line follows it).
    fn score(m: &mut GpuModel<Body>, rows: &[ScoreRow], top: usize) -> Result<(), GateError> {
        check_ids(rows, m.n_vocab()?)?;
        for r in rows {
            m.reset()?;
            Body::prompt(m, &r.prompt)?;
            for (t, &tok) in r.done.iter().enumerate() {
                let line = score_line((&r.key, t), &m.logits()?, tok, &r.extra[t], top)?;
                println!("{line}");
                if t + 1 < r.done.len() {
                    m.step(&[tok])?;
                }
            }
        }
        Ok(())
    }

    /// Every row of `rows` from an empty model as one call of placed passes: the prompt and the
    /// completion but its last id, the last pass's rows read at the completion's positions
    /// ([`GpuModel::score_placed`]); the lines are [`score`]'s.
    fn score_pass(m: &mut GpuModel<Body>, rows: &[ScoreRow], top: usize) -> Result<(), GateError> {
        let n_vocab = m.n_vocab()?;
        check_ids(rows, n_vocab)?;
        for r in rows {
            m.reset()?;
            let n = r.done.len();
            let ids: Vec<u32> = r.prompt.iter().chain(&r.done[..n - 1]).copied().collect();
            let logits = m.score_placed(&ids, n)?;
            for (t, &tok) in r.done.iter().enumerate() {
                let row = &logits[t * n_vocab..(t + 1) * n_vocab];
                println!("{}", score_line((&r.key, t), row, tok, &r.extra[t], top)?);
            }
        }
        Ok(())
    }

    /// `machine` with no card carrying the head, which the host reads.
    fn headless_of(machine: &Machine) -> Machine {
        let mut m = machine.clone();
        m.cards.iter_mut().for_each(|c| c.head = false);
        m
    }

    /// The widest pass shape the plan holds ([`kimi_k3_pass::pick`]), each shape planned as
    /// [`plan_for`] plans: on the machine `build` makes at its rows, and on that machine headless
    /// when the stage card's dense trunk alone refuses it. Every shape refused, and any refusal
    /// the pick does not step down on, ends the run by name.
    fn pick_shape(
        inputs: &PlanInputs,
        build: &dyn Fn(u64) -> Result<Machine, GateError>,
        ctx: usize,
        levers: &PlanLevers,
        name: &str,
    ) -> Result<Picked, GateError> {
        let attempt = |pass: &PlanInputs, machine: &Machine| -> Result<(), PlaceError> {
            match pass.plan(machine, ctx as u64, levers) {
                Err(PlaceError::Placement(PlacementError::CardFreeFloor(_))) => pass
                    .plan(&headless_of(machine), ctx as u64, levers)
                    .map(drop),
                other => other.map(drop),
            }
        };
        kimi_k3_pass::pick(inputs, ctx, build, &attempt)
            .map_err(|e| format!("--place {name} --walk pass: {e}"))?
            .map_err(|no| format!("--place {name} --walk pass: {no}").into())
    }

    /// The plan on `machine` and the machine it is on. A placed plan (`placed` names its place)
    /// the stage card's dense trunk alone refuses (its bytes past the card's free bytes) is made
    /// again on `headless`, the same machine with no card carrying the head, which the host reads.
    fn plan_for<'a>(
        inputs: &'a PlanInputs,
        (machine, headless): (&'a Machine, &'a Machine),
        ctx: u64,
        levers: &PlanLevers,
        placed: Option<&str>,
    ) -> Result<(&'a Machine, Plan<'a>), GateError> {
        match (inputs.plan(machine, ctx, levers), placed) {
            (Err(PlaceError::Placement(PlacementError::CardFreeFloor(floor))), Some(name)) => {
                let on_host = inputs.plan(headless, ctx, levers).map_err(|e| {
                    format!("--place {name}: {floor}; and with the head on the host: {e}")
                })?;
                eprintln!(
                    "head on host: the plan with the head on the stage card was refused ({floor}); \
                     the plan below reads the head on the host"
                );
                Ok((headless, on_host))
            }
            (other, _) => Ok((
                machine,
                other.map_err(|e| format!("--place {}: {e}", placed.unwrap_or("a")))?,
            )),
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&ACTS_ON)?;
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            if !KNOWN.contains(&args[i].as_str()) {
                return Err(format!(
                    "generate_kimi_k3: {} is no flag of this CLI (one of {KNOWN:?})",
                    args[i]
                )
                .into());
            }
            i += if args[i] == "--plan" { 1 } else { 2 };
        }
        let plan_only = args.iter().any(|a| a == "--plan");
        let n: usize = flag(&args, "-n")?.map_or(Ok(8), |s| s.parse())?;
        let ctx: usize = flag(&args, "--ctx")?.map_or(Ok(512), |s| s.parse())?;
        let mode = match flag(&args, "--mode")?.as_deref() {
            None | Some("graph") => StepMode::Graph,
            Some("eager") => StepMode::Eager,
            Some(o) => {
                return Err(format!("generate_kimi_k3: --mode is eager or graph, not {o}").into());
            }
        };
        let place = Place::parse(flag(&args, "--place")?.as_deref().unwrap_or("a"))?;
        let (tokens, text) = (flag(&args, "--tokens")?, flag(&args, "--prompt")?);
        let scoring = flag(&args, "--score")?;
        let score_top: usize = flag(&args, "--score-top")?.map_or(Ok(20), |s| s.parse())?;
        if scoring.is_some() {
            if tokens.is_some() || text.is_some() || args.iter().any(|a| a == "-n") {
                return Err(
                    "generate_kimi_k3: --score takes its prompts from the file: no --tokens, \
                     --prompt or -n"
                        .into(),
                );
            }
            if score_top == 0 {
                return Err("generate_kimi_k3: --score-top is at least 1".into());
            }
        } else if args.iter().any(|a| a == "--score-top") {
            return Err("generate_kimi_k3: --score-top needs --score".into());
        }
        let imat = flag(&args, "--imatrix")?;
        let corpus = flag(&args, "--text")?;
        let chunks: Option<usize> = flag(&args, "--chunks")?.map(|s| s.parse()).transpose()?;
        if imat.is_some() {
            if corpus.is_none() {
                return Err("generate_kimi_k3: --imatrix needs --text FILE".into());
            }
            if scoring.is_some()
                || tokens.is_some()
                || text.is_some()
                || args.iter().any(|a| a == "-n")
            {
                return Err(
                    "generate_kimi_k3: --imatrix takes its prompt from --text: no --score, \
                     --tokens, --prompt or -n"
                        .into(),
                );
            }
        } else if corpus.is_some() || chunks.is_some() {
            return Err("generate_kimi_k3: --text and --chunks need --imatrix".into());
        }
        let free = scoring.is_none() && imat.is_none();
        let walk_pass = match flag(&args, "--walk")?.as_deref() {
            None | Some("steps") => false,
            Some("pass") => true,
            Some(o) => {
                return Err(format!("generate_kimi_k3: --walk is steps or pass, not {o}").into());
            }
        };
        if free && tokens.is_some() == text.is_some() && !plan_only {
            return Err("generate_kimi_k3: one prompt: --tokens a,b,c or --prompt TEXT".into());
        }

        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read_without_chat_for(&split, ctx as u64)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let census = gpu_census::census()?;
        let chosen = Place::choose(Some(place), &census, TIER_RULE, |_| {
            Err("--place unset is never asked: the flag defaults to a".into())
        })?;
        chosen.record().eprint();
        chosen.place.serves("kimi-k3", TIER_RULE.tiers)?;
        let tiered = !chosen.place.tier_cards().is_empty();
        let granule = match flag(&args, "--granule")? {
            Some(g) => {
                let g: u64 = g.parse()?;
                if !g.is_power_of_two() || g < 4 << 10 {
                    return Err(format!(
                        "generate_kimi_k3: --granule {g}: a power of two of at least 4096 bytes \
                         (the default is the workstation's 2 MiB)"
                    )
                    .into());
                }
                Some(std::num::NonZeroU64::new(g).ok_or("--granule is nonzero")?)
            }
            None => None,
        };
        // The machine whose expert tier's batch holds `rows` rows.
        let build = |rows: u64| -> Result<Machine, GateError> {
            let batch = tiered
                .then(|| tier_batch_at(&inputs.hp, rows))
                .transpose()
                .map_err(|e| format!("--place {}: {e}", chosen.place.name()))?;
            let mut machine = chosen.place.machine(None, batch)?(inputs.model.layers);
            if let Some(granule) = granule {
                for card in machine.cards.iter_mut().chain(machine.tiers.iter_mut()) {
                    card.granule_bytes = granule;
                }
            }
            Ok(machine)
        };
        let plan_levers = PlanLevers::from_levers(&levers)?;
        let placed = chosen.place.name() != "a";
        if walk_pass && !placed {
            return Err(
                "generate_kimi_k3: --walk pass runs on a placed load; --place a steps the prompt"
                    .into(),
            );
        }
        let pass = walk_pass
            .then(|| pick_shape(&inputs, &build, ctx, &plan_levers, chosen.place.name()))
            .transpose()?;
        let machine = build(pass.map_or(TIER_ROWS, |p| p.shape.rows as u64))?;
        let inputs = match pass {
            Some(p) => kimi_k3_pass::inputs_for(&inputs, ctx, p.shape)?,
            None => inputs,
        };
        let headless = headless_of(&machine);
        let (machine, plan) = plan_for(
            &inputs,
            (&machine, &headless),
            ctx as u64,
            &plan_levers,
            placed.then(|| chosen.place.name()),
        )?;
        record::plan(chosen.place.name(), machine, &plan).eprint();
        if let Some(Picked { shape, wider }) = pass {
            record::prompt_pass(
                (shape.rows, shape.mla_rows),
                (inputs.prompt_reserve, reserve_room(&plan, 0)),
                wider.map(|(w, short)| ((w.rows, w.mla_rows), short)),
            )
            .eprint();
        }
        need::print(&plan, 0, 0, true)?;
        if plan_only {
            return Ok(());
        }
        let score_rows = scoring
            .as_deref()
            .map(|f| read_score_rows(f, ctx))
            .transpose()?;

        let tok = (text.is_some() || imat.is_some())
            .then(|| Tokenizer::from_gguf(&path))
            .transpose()?;
        let ids: Vec<u32> = match (&tokens, &text) {
            (Some(list), _) => list
                .split(',')
                .map(|t| t.trim().parse::<u32>())
                .collect::<Result<_, _>>()?,
            (_, Some(t)) => tok.as_ref().ok_or("no tokenizer")?.encode(t, true, false),
            _ if !free => Vec::new(),
            _ => unreachable!("one prompt by the check above"),
        };
        if free && (ids.is_empty() || n == 0) {
            return Err(format!(
                "generate_kimi_k3: a prompt of {} ids and -n {n}: each is at least 1",
                ids.len()
            )
            .into());
        }
        if free && ids.len() + n > ctx {
            return Err(format!(
                "generate_kimi_k3: a prompt of {} ids and {n} generated exceed --ctx {ctx}",
                ids.len()
            )
            .into());
        }
        let t = std::time::Instant::now();
        let mut m = if placed {
            open_placed(
                split,
                (&plan, &inputs),
                levers.host(),
                mode,
                pass.map(|p| p.shape),
            )?
        } else {
            open(split, ctx, mode)?
        };
        println!(
            "load arch={} resident_bytes={} ctx={ctx} layers={} load_s={:.1}",
            <Body as ChainBody>::arch().name(),
            m.resident_bytes(),
            m.layers().len(),
            t.elapsed().as_secs_f64()
        );
        if let (Some(out), Some(file)) = (imat.as_deref(), corpus.as_deref()) {
            let tok = tok.as_ref().ok_or("no tokenizer")?;
            return imatrix(&mut m, tok, (file, out), (ctx, chunks), walk_pass);
        }
        if let Some(rows) = score_rows {
            return if walk_pass {
                score_pass(&mut m, &rows, score_top)
            } else {
                score(&mut m, &rows, score_top)
            };
        }
        let out = if levers.step_stats() {
            greedy_probed(&mut m, &ids, n)?
        } else {
            greedy(&mut m, &ids, n)?
        };
        println!("ids {out:?}");
        println!("logits fnv1a64={:016x}", fnv_row(&m.logits()?));
        if let Some(tok) = tok {
            println!("text {:?}", tok.decode(&out));
        }
        Ok(())
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("generate_kimi_k3: built without the `gpu` feature");
    std::process::exit(2);
}
