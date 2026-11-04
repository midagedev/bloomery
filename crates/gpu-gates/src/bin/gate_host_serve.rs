//! `gate_host_serve` — the host seat of `bloomery-serve` against the model it
//! serves, on the DeepSeek-V2-Lite file of the prompt set (`$BLOOMERY_REF_MODEL`,
//! the deepseek2 profile's file), with no card visible: every server and
//! `--plan` run the gate starts holds `CUDA_VISIBLE_DEVICES` empty, and the
//! gate itself opens no card and takes no card lock (it runs under
//! `tools/host-gate.sh`, the lock-free runner).
//!
//!     gate_host_serve --dir <dir> [--qwen36 <file>]
//!
//! The server is `bloomery-serve -m <file> …`, beside this binary with
//! `BLOOMERY_REF_MODEL` removed from its environment: the serving run names no
//! `--model` word (the empty census seats the file on the host seat), the
//! `--plan` runs of the seat's own flags do. Logs per run in `<dir>/<name>/`
//! (`server.err`, the responses).
//!
//! - `plan_names_the_host_stage`: `--model host --plan` exits 0 and prints the
//!   `ctx` line (the trained context), the `plan` record (`stage host`,
//!   `layers 0..<block_count>`, `head host`), the `need` record (the file's
//!   tensor bytes, no card) and the `parallel` line, in that order, and no
//!   `load` or `listening` record. FAIL-first: a plan record that names
//!   another stage, or a layer range off by one, is red.
//! - `ctx_past_the_trained_one_is_refused`: `--ctx <trained + 1> --plan`
//!   exits non-zero naming both counts.
//! - `parallel_above_one_is_refused`: `--parallel 2 --plan` exits non-zero
//!   naming the one sequence, and no `plan` record is printed.
//! - `the_llama_doors_ask_for_the_host`: `-ngl 0` and `-dev none`, each with no
//!   `--model` word and no card visible, print the host seat's plan (`--plan`
//!   exits 0), and `-ngl <block_count>` (a layer left on the CPU) is refused
//!   by name with the count that passes.
//! - `load_and_listen`: the server (no `--model` word) listens; the `load`
//!   record names `deepseek2`, one slot, the file's layers and the bytes the
//!   need named, comes before the `listening` record, and its `ctx` is the ctx
//!   line's.
//! - `completion_ids_are_the_clis`: `/completion` at temperature 0, [`N`]
//!   tokens, of every prompt row of `tools/ref/prompts.tsv` (fed as ids, each
//!   from an empty cache), answers the ids `bloomery-decode` gives for the
//!   same ids — run here through the CLI's own calls (the file's `Gguf`, its
//!   `Derived` weights, one `step` over the prompt, `argmax`, then one `step`
//!   an answer id), bit for bit. FAIL-first: a seat that starts its decode one
//!   position late moves the ids and the clause is red.
//! - `the_first_argmax_matches_ik`: each answer's first token is the argmax
//!   ik answered for that row (`$BLOOMERY_DATA/argmax-ik.tsv`, `just
//!   argmax-ref` writes it), with the divergence set pinned empty.
//! - `qwen36_is_refused_by_name` (only with `--qwen36`, the Qwen3.6 file):
//!   the start with no `--model` word and no card visible seats the file on
//!   the host seat, which ends 65 and lists what the host engine does not run
//!   of it (its delta-rule and GQA mixers, the attention output gate, the
//!   recurrent state, the shared expert's gate).

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_host_serve", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::process::{Command, ExitStatus};
    use std::time::Duration;

    use bloomery_gpu_gates::ctx_ladder::{ctx_line, field};
    use bloomery_gpu_gates::prompts::{PromptRow, read_prompts};
    use bloomery_gpu_gates::record::{self, Fields};
    use bloomery_gpu_gates::serve_client::{
        Answer, SeatGate, Served, agree, check, greedy, plan_run, serve_cmd, server_log, utf8,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, generate, ref_model_path};
    use gguf::{Gguf, Split};
    use model::arch::deepseek2::derived::Derived;
    use model::arch::deepseek2::forward::{argmax, new_cache, step};

    const USAGE: &str = "usage: gate_host_serve --dir <dir> [--qwen36 <file>]";

    /// The tokens a completion asks for.
    const N: usize = 8;

    /// Polls of a second that bound the wait for a server to listen.
    const LISTEN_POLLS: usize = 900;

    /// What the Qwen3.6 refusal lists of the host engine's rows, each as the
    /// words its item prints (`models::Need`'s display).
    const QWEN36_ITEMS: [&str; 5] = [
        "delta rule",
        "GQA flash",
        "attention output gate",
        "recurrent-state",
        "sigmoid gate",
    ];

    /// This gate's seat: the one server's model word and the kinds it prints.
    const SEAT: SeatGate = SeatGate {
        model: "host",
        kinds: record::BLOOMERY_SERVE_HOST,
    };

    /// The gate's `--dir <dir>` and `--qwen36 <file>` (refused when the file
    /// it names is not there: the clause is asked for or absent, never dropped
    /// by a typo).
    fn args() -> Result<(PathBuf, Option<PathBuf>), GateError> {
        let mut dir = None;
        let mut qwen36 = None;
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--dir" => dir = Some(PathBuf::from(v)),
                "--qwen36" => qwen36 = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        let qwen36 = match qwen36 {
            None => None,
            Some(p) if p.exists() => Some(p),
            Some(p) => {
                return Err(format!("--qwen36 {}: no such file", p.display()).into());
            }
        };
        Ok((dir.ok_or(USAGE)?, qwen36))
    }

    /// The prompt rows of `tools/ref/prompts.tsv`, the set the model's own
    /// prompts gate and ik's argmax file are over.
    fn prompts() -> Result<Vec<PromptRow>, GateError> {
        read_prompts(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/prompts.tsv"))
    }

    /// ik's argmax for each prompt row, in row order: one row of
    /// `$BLOOMERY_DATA/argmax-ik.tsv` each (the lead writes it: `just
    /// argmax-ref`), its third column.
    fn read_ik() -> Result<Vec<u32>, GateError> {
        let base = std::env::var("BLOOMERY_DATA").unwrap_or_else(|_| "/root/bloomery-data".into());
        let path = Path::new(&base).join("argmax-ik.tsv");
        let text = std::fs::read_to_string(&path).map_err(|e| {
            format!(
                "no ik answers at {} ({e}). The lead produces them: `just argmax-ref`. Do not \
                 compute them yourself and do not skip this clause.",
                path.display()
            )
        })?;
        text.lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                f.get(2)
                    .and_then(|a| a.trim().parse().ok())
                    .ok_or_else(|| format!("{}: no argmax id in {l:?}", path.display()).into())
            })
            .collect()
    }

    /// `bloomery-serve --model host -m <file>` with no card visible
    /// (`CUDA_VISIBLE_DEVICES` names none), then `args`.
    fn no_card(file: &Path, args: &[&str]) -> Result<Command, GateError> {
        let (mut cmd, seat) = SEAT.command(file)?;
        cmd.env("CUDA_VISIBLE_DEVICES", "").args(seat).args(args);
        Ok(cmd)
    }

    /// `bloomery-serve -m <file>` with no `--model` word (the file and the
    /// flags pick the seat) and no card visible, then `args`.
    fn no_word(file: &Path, args: &[&str]) -> Result<Command, GateError> {
        let mut cmd = serve_cmd()?;
        cmd.env("CUDA_VISIBLE_DEVICES", "")
            .arg("-m")
            .arg(utf8(file)?)
            .args(args);
        Ok(cmd)
    }

    /// One `--plan` run of `cmd`: its exit status, the log of its records and
    /// its stderr's text, the run's logs in `<dir>/<name>/`.
    fn plan(
        dir: &Path,
        name: &str,
        cmd: Command,
    ) -> Result<(ExitStatus, record::Log, String), GateError> {
        let d = dir.join(name);
        std::fs::create_dir_all(&d)?;
        let err = d.join("server.err");
        let (status, lines) = plan_run(cmd, &[], &err)?;
        println!("{name}: --plan: {status}");
        let text = lines.join("\n");
        if !status.success() {
            println!("{name} stderr:\n{text}");
        }
        Ok((status, server_log(&err, SEAT.kinds)?, text))
    }

    /// Whether the `plan` record names the host stage over every layer: stage
    /// `host`, layers `0..<block_count>`, head `host`.
    fn names_host_stage(p: &Fields, layers: u64) -> Result<bool, GateError> {
        Ok(p.word("stage")? == "host"
            && p.word("layers")? == format!("0..{layers}")
            && p.word("head")? == "host")
    }

    /// Whether the `need` record holds the file's bytes and no card.
    fn needs_no_card(n: &Fields, bytes: u64) -> Result<bool, GateError> {
        Ok(n.u64("bytes")? == bytes && n.csv("cards")?.is_empty())
    }

    /// `plan_names_the_host_stage` and `ctx_past_the_trained_one_is_refused`
    /// (the module header); the trained context the plan named.
    fn plan_names_the_host_stage(dir: &Path, file: &Path) -> Result<Option<u64>, GateError> {
        let mut ok = true;
        let (status, log, text) = plan(dir, "plan", no_card(file, &[])?)?;
        let planned = log.first(&record::PLAN_HOST)?;
        let needed = log.first(&record::NEED)?;
        let parallel_at = text
            .lines()
            .position(|l| l.starts_with("parallel "))
            .map(|i| i + 1);
        let in_order = matches!(
            (&planned, &needed, parallel_at),
            (Some(p), Some(n), Some(at)) if p.at() < n.at() && n.at() < at
        );
        let (trained, layers) = trained_and_layers(file)?;
        let host_stage = match &planned {
            Some(p) => names_host_stage(p, layers)?,
            None => false,
        };
        let no_load = log.first(&record::LOAD_HOST)?.is_none()
            && log.first(&record::LISTENING_HOST)?.is_none();
        let exits = status.success();
        println!(
            "plan only: exit 0 {exits}; plan, need, parallel in order {in_order}; stage host and \
             layers 0..{layers} {host_stage}; no load or listening record {no_load}"
        );
        let bytes = resident_bytes(file)?;
        let need_named = match &needed {
            Some(n) => needs_no_card(n, bytes)?,
            None => false,
        };
        println!("plan only: the need names the file's {bytes} B and no card {need_named}");
        let line = ctx_line(&text);
        let holds = line.and_then(|l| field(l, "ctx")) == Some(trained);
        println!("plan only: the ctx line {line:?} holds the trained context {holds}");
        check(
            &mut ok,
            "plan_names_the_host_stage",
            exits && in_order && host_stage && no_load && need_named && holds,
        );
        let above = (trained + 1).to_string();
        let (status, _, text) = plan(dir, "plan-past-trained", no_card(file, &["--ctx", &above])?)?;
        let named = text.contains("passes the file's trained");
        println!("plan only: --ctx {above} is refused and names both counts {named}");
        check(
            &mut ok,
            "ctx_past_the_trained_one_is_refused",
            !status.success() && named,
        );
        Ok(ok.then_some(trained))
    }

    /// `parallel_above_one_is_refused` (the module header).
    fn parallel_above_one_is_refused(dir: &Path, file: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let (status, log, text) =
            plan(dir, "refuse-parallel", no_card(file, &["--parallel", "2"])?)?;
        let named = text.contains("holds one sequence");
        let no_plan = log.first(&record::PLAN_HOST)?.is_none();
        println!(
            "refuse-parallel: exit non-zero {}; names one sequence {named}; no plan record \
             {no_plan}",
            !status.success()
        );
        check(
            &mut ok,
            "parallel_above_one_is_refused",
            !status.success() && named && no_plan,
        );
        Ok(ok)
    }

    /// `the_llama_doors_ask_for_the_host` (the module header): `-ngl 0` and
    /// `-dev none` plan on the host seat with no `--model` word and no card,
    /// and `-ngl <block_count>` keeps its named refusal.
    fn the_llama_doors_ask_for_the_host(dir: &Path, file: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let (_, layers) = trained_and_layers(file)?;
        for (name, flag, value) in [
            ("door-ngl-zero", "-ngl", "0"),
            ("door-dev-none", "-dev", "none"),
        ] {
            let (status, log, _) = plan(dir, name, no_word(file, &[flag, value])?)?;
            let planned = match log.first(&record::PLAN_HOST)? {
                Some(p) => names_host_stage(&p, layers)?,
                None => false,
            };
            println!(
                "{name}: {flag} {value} --plan exits 0 {} and plans the host stage {planned}",
                status.success()
            );
            check(
                &mut ok,
                "the_llama_doors_ask_for_the_host",
                status.success() && planned,
            );
        }
        let at = layers.to_string();
        let (status, _, text) = plan(dir, "door-ngl-layers", no_word(file, &["-ngl", &at])?)?;
        let named = text.contains("runs every layer on its stage card");
        println!(
            "door-ngl-layers: -ngl {at} is refused {} and names the rule {named}",
            !status.success()
        );
        check(
            &mut ok,
            "the_llama_doors_ask_for_the_host",
            !status.success() && named,
        );
        Ok(ok)
    }

    /// `load_and_listen` (the module header) and the answers its requests
    /// gave, one a prompt row; `None` when the server never listened.
    fn load_and_listen(
        dir: &Path,
        file: &Path,
        trained: Option<u64>,
        rows: &[PromptRow],
    ) -> Result<(Option<Vec<Answer>>, bool), GateError> {
        let mut ok = true;
        let d = dir.join("serve");
        std::fs::create_dir_all(&d)?;
        let mut cmd = no_word(file, &[])?;
        cmd.arg("--port").arg("0");
        let mut s = Served::spawn_cmd(cmd, &[], &d)?;
        let err_log = d.join("server.err");
        let addr = match s.address(&err_log, LISTEN_POLLS, Duration::from_secs(1)) {
            Ok(addr) => addr,
            Err(e) => {
                println!("the server never listened: {e}");
                check(&mut ok, "load_and_listen", false);
                return Ok((None, ok));
            }
        };
        let text = std::fs::read_to_string(&err_log).unwrap_or_default();
        let records = server_log(&err_log, SEAT.kinds)?;
        let load = records.one(&record::LOAD_HOST)?;
        let listen = records.one(&record::LISTENING_HOST)?;
        let (_, layers) = trained_and_layers(file)?;
        let bytes = resident_bytes(file)?;
        let loaded = load.word("arch")? == "deepseek2"
            && load.u64("slots")? == 1
            && load.u64("layers")? == layers
            && load.u64("resident_bytes")? == bytes
            && listen.u64("slots")? == 1;
        let named = ctx_line(&text).and_then(|l| field(l, "ctx"));
        let same = trained.is_none_or(|t| named == Some(t));
        let ctx_is = named == Some(load.u64("ctx")?) && named == Some(listen.u64("ctx")?);
        let in_order = load.at() < listen.at();
        println!(
            "serve at {addr}: load {:?}; listening {:?}; names arch, one slot, {layers} layers \
             and {bytes} B {loaded}; ctx is the line's {ctx_is} (the plan run's {same}); load \
             before listening {in_order}",
            load.line(),
            listen.line()
        );
        check(
            &mut ok,
            "load_and_listen",
            loaded && same && ctx_is && in_order,
        );
        let url = |p: &str| format!("http://{addr}{p}");
        let mut answers = Vec::with_capacity(rows.len());
        for row in rows {
            let (a, _) = greedy(
                &url,
                row.tokens.clone(),
                N,
                &d,
                &format!("p{}", row.id),
                false,
            )?;
            answers.push(a);
        }
        println!("server stopped: {}", s.stop()?);
        Ok((Some(answers), ok))
    }

    /// The ids `bloomery-decode` gives for each prompt row, run here through
    /// the CLI's own calls on the same file: the `Gguf`, its `Derived`
    /// weights, one `step` over the prompt (the CLI's prefill), `argmax`, then
    /// one `step` an answer id, each answer fed back — the split the server's
    /// prefill of all but the last id and one step of it gives the same bits
    /// as (`hw_kv_every_split_is_bit_exact`).
    fn cli_ids(file: &Path, rows: &[PromptRow]) -> Result<Vec<Vec<u32>>, GateError> {
        let g = Gguf::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let derived = Derived::new(&g).map_err(|e| format!("{}: {e}", file.display()))?;
        let mut cache = new_cache(&g).map_err(|e| format!("{}: {e}", file.display()))?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            cache.truncate(0);
            let logits = step(&g, &row.tokens, &mut cache, &derived)
                .map_err(|e| format!("prompt {}: {e}", row.id))?;
            let mut answer = vec![argmax(&logits.data)];
            while answer.len() < N {
                let last = *answer.last().ok_or("an empty answer")?;
                let logits = step(&g, &[last], &mut cache, &derived)
                    .map_err(|e| format!("prompt {}: {e}", row.id))?;
                answer.push(argmax(&logits.data));
            }
            out.push(answer);
        }
        Ok(out)
    }

    /// `completion_ids_are_the_clis` and `the_first_argmax_matches_ik` (the
    /// module header), the server stopped.
    fn ids_and_ik(file: &Path, rows: &[PromptRow], answers: &[Answer]) -> Result<bool, GateError> {
        let mut ok = true;
        if answers.len() != rows.len() {
            return Err(format!(
                "the server answered {} prompts of the set's {}",
                answers.len(),
                rows.len()
            )
            .into());
        }
        let ids = cli_ids(file, rows)?;
        let mut mismatched = Vec::new();
        for (row, (answer, want)) in rows.iter().zip(answers.iter().zip(&ids)) {
            if !agree(&answer.tokens, &answer.stop, want) {
                mismatched.push(row.id);
                println!(
                    "prompt {} ({}): the server's {:?} is not the CLI's {want:?}",
                    row.id, row.text, answer.tokens
                );
            }
        }
        println!(
            "completion ids: {} of {} rows answer the CLI's bit for bit",
            rows.len() - mismatched.len(),
            rows.len()
        );
        check(
            &mut ok,
            "completion_ids_are_the_clis",
            mismatched.is_empty(),
        );
        let ik = read_ik()?;
        if ik.len() != rows.len() {
            return Err(format!(
                "ik answered {} prompts, the set holds {}",
                ik.len(),
                rows.len()
            )
            .into());
        }
        let mut diverged: Vec<(usize, u32, Option<u32>)> = Vec::new();
        for ((row, answer), &want) in rows.iter().zip(answers).zip(&ik) {
            let first = answer.tokens.first().copied();
            if first != Some(want) {
                diverged.push((row.id, want, first));
            }
        }
        for (id, want, got) in &diverged {
            println!("prompt {id}: ik {want}, the server's first {got:?}");
        }
        // The divergence set is pinned empty, as the model's own prompts gate
        // pins it: a flip on either side of the same bits is a move someone
        // must own.
        const KNOWN_DIVERGENCE: &[usize] = &[];
        let moved: Vec<usize> = diverged.iter().map(|&(id, ..)| id).collect();
        let none = moved == KNOWN_DIVERGENCE;
        println!(
            "the first argmax: {} of {} rows match ik; divergence set {moved:?} is the pinned \
             {KNOWN_DIVERGENCE:?} {none}",
            rows.len() - diverged.len(),
            rows.len()
        );
        check(&mut ok, "the_first_argmax_matches_ik", none);
        Ok(ok)
    }

    /// `qwen36_is_refused_by_name` (the module header), on the file the recipe
    /// named: the start with no model word, no card visible and no host ask is
    /// the host seat's, and the seat refuses the file with the host engine's
    /// list.
    fn qwen36_is_refused_by_name(dir: &Path, file: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let (status, _, text) = plan(dir, "refuse-qwen36", no_word(file, &[])?)?;
        let code = status.code();
        let named: Vec<&str> = QWEN36_ITEMS
            .iter()
            .filter(|w| text.contains(*w))
            .copied()
            .collect();
        println!(
            "refuse-qwen36: exit 65 {} (code {code:?}); the list names {named:?} of {} host items",
            code == Some(65),
            QWEN36_ITEMS.len()
        );
        check(
            &mut ok,
            "qwen36_is_refused_by_name",
            code == Some(65) && named.len() == QWEN36_ITEMS.len(),
        );
        Ok(ok)
    }

    /// The file's trained context and its `block_count`.
    fn trained_and_layers(file: &Path) -> Result<(u64, u64), GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        let trained = u64::try_from(
            generate::trained_ctx(&split)
                .ok_or_else(|| format!("{}: no context_length", file.display()))?,
        )?;
        let layers = split
            .arch_get_u64("block_count")
            .ok_or_else(|| format!("{}: no block_count", file.display()))?;
        Ok((trained, layers))
    }

    /// The file's tensor bytes, as the header states them.
    fn resident_bytes(file: &Path) -> Result<u64, GateError> {
        let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
        split
            .iter_tensors()
            .map(|(_, t)| t.nbytes)
            .try_fold(0u64, u64::checked_add)
            .ok_or_else(|| "the file's tensor bytes pass u64".into())
    }

    pub fn run() -> Result<(), GateError> {
        let (dir, qwen36) = args()?;
        std::fs::create_dir_all(&dir)?;
        let file = ref_model_path()?;
        let rows = prompts()?;
        let mut ok = true;
        let trained = plan_names_the_host_stage(&dir, &file)?;
        ok &= trained.is_some();
        ok &= parallel_above_one_is_refused(&dir, &file)?;
        ok &= the_llama_doors_ask_for_the_host(&dir, &file)?;
        let (answers, pass) = load_and_listen(&dir, &file, trained, &rows)?;
        ok &= pass;
        if let Some(answers) = answers {
            ok &= ids_and_ik(&file, &rows, &answers)?;
        }
        match qwen36 {
            Some(q) => ok &= qwen36_is_refused_by_name(&dir, &q)?,
            None => println!("qwen36_is_refused_by_name: skipped, no --qwen36 file"),
        }
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_host_serve the host seat plans the file's stage, answers the CLI's ids \
             bit for bit with ik's first argmax, takes the llama doors' host ask, and refuses a \
             file the host engine cannot run by name"
        );
        Ok(())
    }
}
