//! `generate_metal` — the qwen35moe session's decode CLI on Metal: a
//! qwen35moe file opened on this Mac's device, greedy, one token a step, the
//! generated text streamed as ids are picked. The flags spell what the CUDA
//! family's `generate_qwen3moe` spells where the meaning is the same.
//!
//!     generate_metal [-m PATH | --hf REPO]
//!                    (--prompt <text> | --tokens a,b,c) [-n N] [--ctx C]
//!                    [--print-ids] [--ignore-eos] [--repeat N] [--profile [N]]
//!                    [--profile-layers] [--step-times [N]] [--prefill batch|steps]
//!                    [--pass-times U]
//!     generate_metal [-m PATH | --hf REPO] --plan [--ctx C]
//!
//! The model is a GGUF file's first shard or a checkpoint folder (an MLX
//! folder as downloaded, no `prepare`), named by `-m PATH`, by `--hf REPO`
//! (fetched into the cache, `bloomery_metal::model_arg`) or by
//! `$BLOOMERY_REF_MODEL`, as the CUDA CLIs read it. It is named once: a
//! path beside `--hf`, `-m` or `--hf` twice, and a flag that differs from the
//! variable are each a named error and exit 2. The file opens once, for the
//! tokenizer, the plan's check and the session.
//!
//! Defaults: N 32, C the smaller of 4096 and the file's `context_length`. A
//! flag given twice takes its last value, but `-m` and `--hf` (above).
//! `--prompt` tokenizes the text with the file's own vocabulary (`tokenizer`,
//! the specials the header asks for, no chat template — the CUDA CLI runs
//! none either; a server's seat runs the file's). The generation stops at
//! the count, at an end-of-generation id of the vocabulary's own set, or at
//! the cache's last row; the summary lines say which.
//!
//! `--ignore-eos` (off by default) stops nothing at an end-of-generation id:
//! the run goes on to the count or the cache's last row, and its ids hold the
//! end-of-generation ids the model picks. The picks are the device's argmax,
//! as without the flag, not a mask over the vocabulary's logits as the CUDA
//! CLIs' `--ignore-eos` takes them: a run's ids through its first
//! end-of-generation id equal the run without the flag. A decode timed at a
//! depth of real text then is not cut short; the summary prints
//! `ignore_eos=true`.
//!
//! `--repeat N` (default 1) runs the prompt N times on the one open, the
//! session reset before each run after the first. Each run streams its own
//! output; with N above 1 each run's summary block opens with `run=K`, and
//! the open's lines print once, with the first run's.
//!
//! The text streams through the tokenizer's decoder, so a multi-byte
//! character split across pieces prints once, complete; `--print-ids`
//! prints the generated ids as one comma-separated stdout line instead (the
//! `--tokens` spelling, so a run's output feeds back as an input), for
//! gates. The summary lines go to stderr, `key=value` one a line:
//!
//!     model=<path> gemm_path=<simdgroup|tensor: the prompt pass's GEMMs,
//!     the device's matrix-unit rule's pick> open_s=<the open's wall>
//!     open_load_s=<the plan read and the weights copied> open_rows_ms=<the
//!     rows arena allocated> open_residency_ms=<the residency set's commit>
//!     open_pipelines_ms=<the pipelines built> device_allocated_bytes=<the
//!     driver's live device bytes after the open> rows_arena_bytes=<the rows
//!     arena's bytes> prompt_ids=<count>
//!     prompt_s=<the prompt's wall> prompt_tok_s=<rate> decode_tokens=<count>
//!     decode_passes=<count> decode_s=<the loop's wall> decode_tok_s=<rate>
//!     stop=<eog|length|ctx> decode_first_ms=<the loop's first pass>
//!     decode_rest_ms=<the mean of the others> resident_bytes=<the
//!     residency set's bytes> residency_requests=<its requests>
//!
//! The decode rate is the loop's passes over its wall: generated token 0
//! comes from the prompt call and is in `decode_tokens`, not in the rate. A
//! run of one token has no pass, and its rate prints `none`.
//!
//! `--pass-times U[,U…]` times, for each U, the ops of one pass of U prompt
//! ids, each on a command buffer of its own waited before the next, as
//! per-class sums, beside the wall of the same pass as the prompt runs it
//! (`pass_wall`); then the wall of one checkpoint after a pass of the last U
//! and of the whole prompt's call from position 0, both warm.
//!
//! `--prefill steps` feeds the prompt a decode step an id, the same-binary arm
//! of the default (`batch`: the prompt in passes of up to
//! [`Session::pass_rows`] ids, a short one by steps). The session opens its
//! rows arena either way.
//!
//! `--profile [N]` (default N 32; `--profile-layers` implies it) steps N
//! more tokens after the summary, each group of each token on a command
//! buffer of its own with its GPU seconds and its bytes read, and prints a
//! `prof` line a group kind (and a layer, under `--profile-layers`), then
//! the total beside the run's own decode ms a token, so the gap between the
//! groups' GPU time and the real step shows.
//!
//! `--step-times [N]` (default N 32) steps N more tokens in each mode —
//! host-fed, chain, ahead, one id throughout — and prints one `step_times`
//! line a mode: the host's encode µs a token, its GPU start against the host
//! clock at the commit, the GPU's own µs, and the host's µs from the GPU end
//! to the settle's return. What the phases say of a mode the sum against the
//! step's wall does: a host-fed step pays all four serially, a chain or
//! ahead step encodes the next token under the current one's GPU time.
//!
//! The wall and rate lines are development-loop numbers: no lease, no
//! witness, this Mac whatever else it runs. The `prof` lines are too: the
//! profiled token runs one command buffer a group, waited, which the real
//! step's single buffer does not.
//!
//! `--plan` reads the file's header and tensor table alone (no weight byte, no
//! device buffer) and prints, on stdout, what the open would find: the
//! description, the room the device and the host report beside the bytes the
//! open would make buffers of, every refusal the open's checks meet (not the
//! first alone), and a last line, `plan: opens` (exit 0) or `plan: refused, N
//! items` (exit 3). A file whose data section is cut short is read as a header
//! alone. A folder or file the open refuses before the plan can check it is a
//! named error and exit 3; a header the plan cannot read is a named error and
//! exit 64; a repo it cannot fetch is a named error and exit 1. `--hf` fetches
//! what the cache lacks before the plan reads it. It takes no generation flag.
//!
//! A bad flag or a bad value is a named error and exit 2; a refusal of the
//! device, the session or the tokenizer, or a fault the device raised, is a
//! named error and exit 1 — never a partial result presented as success.
//! The body is macOS only; elsewhere it refuses and exits 78.
//!
//! A build with `--features encode-trace` also takes `--encode-trace FILE`:
//! the run after the open (the prompt's passes and the decode steps) writes
//! every dispatch, binding and commit/wait/event it asks of Metal to FILE
//! (`bloomery_metal::trace`), which `metal_scan --compare-trace` compares.

#[cfg(target_os = "macos")]
fn main() -> std::process::ExitCode {
    cli::run()
}

#[cfg(target_os = "macos")]
mod cli {
    use std::io::Write as _;
    #[cfg(feature = "encode-trace")]
    use std::path::PathBuf;
    use std::process::ExitCode;
    use std::time::{Duration, Instant};

    use bloomery_metal::arch::qwen3moe::{
        check,
        count::{GroupKind, KINDS},
        generate::{Generated, greedy},
        plan,
        session::{Op, PassOut, PassSample, ProfileGroup, Session, StateOut, StepMode},
    };
    use bloomery_metal::device::MetalDevice;
    use bloomery_metal::kernels::PASS_ROWS;
    use bloomery_metal::model_arg;
    use hf::source::Source;
    use model_hf::ModelFile;
    use runtime::prompt::PrefillMode;
    use runtime::{Target, Want};

    /// The model file, as the CUDA generate CLIs name it.
    const MODEL_ENV: &str = "BLOOMERY_REF_MODEL";

    /// The model's path flags; `--hf` is the repo's.
    const PATH_FLAGS: &[&str] = &["-m"];

    /// The model the flags and `MODEL_ENV` name, or why there is none; the repo is fetched
    /// by the run, not here.
    fn model_source() -> Result<Source, String> {
        let env = match std::env::var(MODEL_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(e) => return Err(format!("{MODEL_ENV}: {e}")),
        };
        model_arg::named(
            &value_flags()?,
            PATH_FLAGS,
            env.as_deref(),
            &format!("-m PATH or --hf REPO, or set {MODEL_ENV}"),
        )
    }

    /// The cache rows' default cap.
    const CTX_DEFAULT: usize = 4096;

    const USAGE: &str = "\
generate_metal [-m PATH | --hf REPO] (--prompt <text> | --tokens a,b,c) [-n N] [--ctx C] [--print-ids] [--ignore-eos] [--repeat N] [--profile [N]] [--profile-layers] [--step-times [N]] [--prefill batch|steps] [--pass-times U[,U…]]
generate_metal [-m PATH | --hf REPO] --plan [--ctx C]
  the model is a GGUF file (its first shard) or a checkpoint folder, named by -m,
  by --hf (fetched into the cache) or by $BLOOMERY_REF_MODEL; defaults N 32, --ctx the smaller of 4096
  and the file's context_length; --plan reads the header alone and prints
  every refusal the open would make, the bytes it would allocate and the
  room (exit 0 opens, 3 refused, 64 a header it cannot read); --print-ids prints the ids, not the text;
  --ignore-eos stops nothing at an end-of-generation id (the picks stay the device's argmax, unmasked);
  --repeat N runs the prompt N times on one open, each run with its own summary block;
  --profile [N] (default 32) steps N more tokens after the summary, each
  group's GPU time and bytes printed; --profile-layers adds a layer each;
  --step-times [N] (default 32) steps N more tokens in each mode, the step
  clock's phases a mode printed; --prefill steps feeds the prompt a step an
  id, batch (the default) in passes; --pass-times U[,U…] times the ops of one
  pass of the first U prompt ids, one command buffer at a time, beside the
  pass's own wall";

    /// The flags that take a value.
    const VALUE_FLAGS: &[&str] = &[
        "-m",
        "--hf",
        "--prompt",
        "--tokens",
        "-n",
        "--ctx",
        "--prefill",
        "--pass-times",
        "--repeat",
        #[cfg(feature = "encode-trace")]
        "--encode-trace",
    ];
    /// The flags that stand alone.
    const BARE_FLAGS: [&str; 5] = [
        "--print-ids",
        "--ignore-eos",
        "--help",
        "--profile-layers",
        "--plan",
    ];
    /// The profile tokens' default count, and the step-times' per mode.
    const PROFILE_DEFAULT: usize = 32;

    /// What named the prompt: the text the tokenizer encodes, or the ids as
    /// given. The tokenizer's own failures are the run's, not the usage
    /// parse's, so it opens after the flags are read.
    enum Prompt {
        Text(String),
        Ids(Vec<u32>),
    }

    /// What `--profile` asked: how many tokens to step past the summary,
    /// and whether a layer's line prints beside each kind's.
    struct Profile {
        n: usize,
        layers: bool,
    }

    struct Args {
        prompt: Prompt,
        n: usize,
        ctx: Option<usize>,
        print_ids: bool,
        /// `--ignore-eos`: no stop at an end-of-generation id.
        ignore_eos: bool,
        /// `--repeat N`: the runs on one open, at least one.
        repeat: usize,
        profile: Option<Profile>,
        /// `--step-times [N]`: the step clock's tokens a mode.
        step_times: Option<usize>,
        prefill: PrefillMode,
        /// `--pass-times U[,U…]`: the rows of each pass whose ops are timed
        /// alone.
        pass_times: Option<Vec<usize>>,
        /// `--encode-trace FILE`: where the run's encode trace is written.
        #[cfg(feature = "encode-trace")]
        encode_trace: Option<PathBuf>,
        model: Source,
    }

    /// Every value flag with its value, in order. The walk takes each value
    /// flag's next argument as its value, so a value is never read as a flag; a
    /// value flag without its value is refused by name.
    fn value_flags() -> Result<Vec<(String, String)>, String> {
        let mut args = std::env::args().skip(1);
        let mut out = Vec::new();
        while let Some(a) = args.next() {
            if !VALUE_FLAGS.contains(&a.as_str()) {
                continue;
            }
            let value = args.next().ok_or_else(|| format!("{a} needs a value"))?;
            out.push((a, value));
        }
        Ok(out)
    }

    /// Every value of flag `name`, in order.
    fn flags(name: &str) -> Result<Vec<String>, String> {
        Ok(value_flags()?
            .into_iter()
            .filter(|(flag, _)| flag == name)
            .map(|(_, value)| value)
            .collect())
    }

    /// The last value of flag `name`, if it was given.
    fn flag(name: &str) -> Result<Option<String>, String> {
        Ok(flags(name)?.pop())
    }

    /// Comma-separated ids, as the CUDA CLIs parse them.
    fn ids_of(s: &str) -> Result<Vec<u32>, String> {
        s.split(',')
            .map(|v| v.trim().parse::<u32>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("--tokens {s}: {e}"))
    }

    /// The flags that stand alone or take a count.
    const COUNT_FLAGS: [&str; 2] = ["--profile", "--step-times"];

    /// The flag words among the arguments, in order, each with the count it
    /// took. A value flag's value is skipped, never read as a flag; a count
    /// flag eats the next argument only when that one names a count.
    fn flag_words() -> Vec<(String, Option<usize>)> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut words = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if VALUE_FLAGS.contains(&a.as_str()) {
                i += 2;
                continue;
            }
            let count = COUNT_FLAGS
                .contains(&a.as_str())
                .then(|| args.get(i + 1).and_then(|v| v.parse::<usize>().ok()))
                .flatten();
            words.push((a.clone(), count));
            i += if count.is_some() { 2 } else { 1 };
        }
        words
    }

    /// A flag this CLI does not know, if one sits among the arguments.
    fn unknown_flag() -> Option<String> {
        flag_words()
            .into_iter()
            .map(|(a, _)| a)
            .find(|a| !COUNT_FLAGS.contains(&a.as_str()) && !BARE_FLAGS.contains(&a.as_str()))
    }

    /// Whether the bare flag `name` stands among the flag words.
    fn has_flag(name: &str) -> bool {
        flag_words().iter().any(|(a, _)| a == name)
    }

    /// The count flag `name`'s count, its default when it stands alone; the
    /// last one wins, as for every flag.
    fn count_flag(name: &str) -> Option<usize> {
        flag_words()
            .into_iter()
            .rfind(|(a, _)| a == name)
            .map(|(_, n)| n.unwrap_or(PROFILE_DEFAULT))
    }

    fn parse() -> Result<Args, String> {
        if let Some(name) = unknown_flag() {
            return Err(format!("{name} is not a flag of this CLI"));
        }
        if has_flag("--help") {
            print!("{USAGE}");
            std::process::exit(0);
        }
        let text = flag("--prompt")?;
        let tokens = flag("--tokens")?;
        if text.is_some() && tokens.is_some() {
            return Err("give --prompt or --tokens, not both".into());
        }
        let prompt = match (&text, &tokens) {
            (Some(t), _) => Prompt::Text(t.clone()),
            (_, Some(s)) => Prompt::Ids(ids_of(s)?),
            _ => return Err("give one of --prompt or --tokens".to_string()),
        };
        let n_gen: usize = match flag("-n")? {
            None => 32,
            Some(s) => s.parse().map_err(|e| format!("-n {s}: {e}"))?,
        };
        if n_gen == 0 {
            return Err(
                "-n 0: a generation of 0 tokens: token 0 comes out of the prompt call".into(),
            );
        }
        let ctx = ctx_flag()?;
        let print_ids = has_flag("--print-ids");
        let ignore_eos = has_flag("--ignore-eos");
        let repeat: usize = match flag("--repeat")? {
            None => 1,
            Some(s) => s.parse().map_err(|e| format!("--repeat {s}: {e}"))?,
        };
        if repeat == 0 {
            return Err("--repeat 0: no run at all; drop the flag to run once".into());
        }
        #[cfg(feature = "encode-trace")]
        let encode_trace = flag("--encode-trace")?.map(PathBuf::from);
        #[cfg(feature = "encode-trace")]
        if repeat > 1 && encode_trace.is_some() {
            return Err("--repeat beside --encode-trace: the trace holds one run".into());
        }
        // `--profile-layers` alone implies the profile.
        let profile_n = count_flag("--profile");
        let layers = has_flag("--profile-layers");
        let profile = match (profile_n, layers) {
            (Some(n), l) => {
                if n == 0 {
                    return Err(format!(
                        "--profile {n}: a profile of no tokens; drop the flag to run none"
                    ));
                }
                Some(Profile { n, layers: l })
            }
            (None, true) => Some(Profile {
                n: PROFILE_DEFAULT,
                layers: true,
            }),
            (None, false) => None,
        };
        let step_times = count_flag("--step-times");
        if step_times == Some(0) {
            return Err(
                "--step-times 0: a step-times run of no tokens; drop the flag to run none".into(),
            );
        }
        let pass_times = match flag("--pass-times")? {
            None => None,
            Some(s) => {
                let mut us = Vec::new();
                for v in s.split(',') {
                    let u: usize = v
                        .trim()
                        .parse()
                        .map_err(|e| format!("--pass-times {s}: {e}"))?;
                    if u <= PASS_ROWS {
                        return Err(format!(
                            "--pass-times {u}: a pass holds more than {PASS_ROWS} rows"
                        ));
                    }
                    us.push(u);
                }
                Some(us)
            }
        };
        let prefill = match flag("--prefill")? {
            None => PrefillMode::Batch,
            Some(s) => PrefillMode::from_name(&s)
                .ok_or_else(|| format!("--prefill {s}: batch or steps"))?,
        };
        let model = model_source()?;
        Ok(Args {
            prompt,
            n: n_gen,
            ctx,
            print_ids,
            ignore_eos,
            repeat,
            profile,
            step_times,
            prefill,
            pass_times,
            #[cfg(feature = "encode-trace")]
            encode_trace,
            model,
        })
    }

    /// The `--ctx` value: absent, or a count of at least one row.
    fn ctx_flag() -> Result<Option<usize>, String> {
        match flag("--ctx")? {
            None => Ok(None),
            Some(s) => {
                let c: usize = s.parse().map_err(|e| format!("--ctx {s}: {e}"))?;
                if c == 0 {
                    return Err(format!("--ctx {s}: a cache of no rows"));
                }
                Ok(Some(c))
            }
        }
    }

    /// The smaller of the default cap and `split`'s `context_length`.
    fn default_ctx(split: &gguf::Split, model: &std::path::Path) -> Result<usize, String> {
        let train = split
            .arch_get_u64("context_length")
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| {
                format!(
                    "{}: {} is absent or not a count",
                    model.display(),
                    split.arch_key("context_length")
                )
            })?;
        Ok(CTX_DEFAULT.min(train))
    }

    /// The generated text as it completes: the tokenizer's streaming
    /// decoder, flushed a piece.
    struct Stream<'a> {
        decoder: Option<tokenizer::Decoder<'a>>,
    }

    impl Stream<'_> {
        fn id(&mut self, id: u32) {
            let Some(d) = self.decoder.as_mut() else {
                return;
            };
            if let Some(piece) = d.push(id) {
                print!("{piece}");
                let _ = std::io::stdout().flush();
            }
        }

        fn finish(&mut self) {
            if let Some(rest) = self.decoder.as_mut().and_then(|d| d.finish()) {
                print!("{rest}");
            }
            let _ = std::io::stdout().flush();
        }
    }

    /// One accumulator of a kind's (or a kind and layer's) profiled groups.
    #[derive(Clone, Copy, Default)]
    struct Acc {
        groups: u64,
        gpu_s: f64,
        bytes: u64,
    }

    /// The `prof` lines after a run's summary: `p.n` more tokens stepped,
    /// each group on a command buffer of its own, its GPU seconds and its
    /// bytes read, one line a kind and (under `p.layers`) a layer, then the
    /// total beside the run's own decode ms a token. The ids stop nothing
    /// here: the profile steps on, whatever the run stopped at, until the
    /// count or the cache's last row.
    fn profile_run(
        s: &mut Session,
        p: &Profile,
        mut id: u32,
        decode_ms: Option<f64>,
    ) -> Result<(), String> {
        let rows = s.ctx() - s.pos() as usize;
        if rows == 0 {
            return Err(format!(
                "prof: the cache holds no row past position {}; give --ctx more rows",
                s.pos()
            ));
        }
        let steps = p.n.min(rows);
        let layers = s.layer_count();
        // One accumulator a kind and layer (the head's and the embed's sit
        // at `layers`), filled in place: the first token sizes it, the rest
        // allocate nothing.
        let mut acc: Vec<Acc> = vec![Acc::default(); KINDS.len() * (layers + 1)];
        let mut samples: Vec<ProfileGroup> = Vec::new();
        let mut ids: Vec<Vec<u32>> = Vec::with_capacity(layers);
        for _ in 0..steps {
            let pos = s.pos();
            id = s
                .step_profile(id, &mut samples)
                .map_err(|e| e.to_string())?
                .id;
            let probe = s.probe().map_err(|e| e.to_string())?;
            ids.clear();
            for l in 0..layers {
                ids.push(probe.router_ids(l).map_err(|e| e.to_string())?);
            }
            for g in &samples {
                let none: &[u32] = &[];
                let routed = match g.kind {
                    GroupKind::GateUp | GroupKind::Down => &ids[g.layer.unwrap_or(0)],
                    _ => none,
                };
                let bytes = s
                    .group_bytes(g, pos, routed)
                    .map_err(|e| e.to_string())?
                    .total();
                let a = &mut acc[kind_index(g.kind) * (layers + 1) + g.layer.unwrap_or(layers)];
                a.groups += 1;
                a.gpu_s += g.gpu_s;
                a.bytes += bytes;
            }
        }
        eprintln!(
            "prof tokens={steps} at {}..={} (development-loop numbers: one command buffer a \
             group, waited; the real step's single buffer is not this)",
            s.pos() - steps as u32,
            s.pos() - 1
        );
        let kind_total = |kind: GroupKind| -> Acc {
            let mut t = Acc::default();
            for l in 0..=layers {
                let a = &acc[kind_index(kind) * (layers + 1) + l];
                t.groups += a.groups;
                t.gpu_s += a.gpu_s;
                t.bytes += a.bytes;
            }
            t
        };
        let line = |op: &str, layer: Option<usize>, a: &Acc| {
            let n = steps as f64;
            let at = layer.map_or_else(String::new, |l| format!(" layer={l}"));
            eprintln!(
                "prof op={op}{at} groups={:.2} gpu_ms={:.3} bytes={:.0} gbps={:.1}",
                a.groups as f64 / n,
                a.gpu_s / n * 1e3,
                a.bytes as f64 / n,
                if a.gpu_s > 0.0 {
                    a.bytes as f64 / a.gpu_s / 1e9
                } else {
                    0.0
                },
            );
        };
        let mut total = Acc::default();
        for &kind in &KINDS {
            let t = kind_total(kind);
            if t.groups == 0 {
                continue;
            }
            line(kind.name(), None, &t);
            total.groups += t.groups;
            total.gpu_s += t.gpu_s;
            total.bytes += t.bytes;
            if p.layers {
                for l in 0..layers {
                    let a = &acc[kind_index(kind) * (layers + 1) + l];
                    if a.groups != 0 {
                        line(kind.name(), Some(l), a);
                    }
                }
            }
        }
        eprintln!(
            "prof total groups={:.2} gpu_ms={:.3} bytes={:.0} gbps={:.1} decode_ms={}",
            total.groups as f64 / steps as f64,
            total.gpu_s / steps as f64 * 1e3,
            total.bytes as f64 / steps as f64,
            if total.gpu_s > 0.0 {
                total.bytes as f64 / total.gpu_s / 1e9
            } else {
                0.0
            },
            decode_ms.map_or_else(|| "none".to_owned(), |ms| format!("{ms:.3}")),
        );
        Ok(())
    }

    /// `kind`'s place in [`KINDS`].
    fn kind_index(kind: GroupKind) -> usize {
        KINDS
            .iter()
            .position(|k| *k == kind)
            .unwrap_or_else(|| panic!("generate_metal: the kind {kind:?} sits in no kind table"))
    }

    /// Timed runs of each `--pass-times` measurement, after one warm run.
    const PASS_TIMES_RUNS: usize = 3;

    /// The pass over the first `u` ids of the prompt with every op on a
    /// command buffer of its own, waited before the next is committed: the
    /// op's GPU seconds are its own and never a queued buffer's overlap. One
    /// warm pass and `PASS_TIMES_RUNS` timed ones from position 0, each op's
    /// median; the sums per class print as lines, in milliseconds. Then the
    /// same pass as the prompt runs it, its host wall from the call to its
    /// return, the median of `PASS_TIMES_RUNS` after a warm one, beside the
    /// ops' sum: the gap is what the pass pays outside its ops' own GPU
    /// time. The session is reset before each pass and after the last.
    fn pass_times_run(s: &mut Session, prompt: &[u32], u: usize) -> Result<(), String> {
        if prompt.len() < u {
            return Err(format!(
                "--pass-times {u}: the prompt holds {} ids",
                prompt.len()
            ));
        }
        let ids = &prompt[..u];
        let mut runs: Vec<Vec<PassSample>> = Vec::new();
        for run in 0..=PASS_TIMES_RUNS {
            s.reset().map_err(|e| e.to_string())?;
            let mut samples = Vec::new();
            s.pass_profile(ids, PassOut::Last, &mut samples)
                .map_err(|e| e.to_string())?;
            if run > 0 {
                runs.push(samples);
            }
        }
        s.reset().map_err(|e| e.to_string())?;
        let n = runs[0].len();
        if runs.iter().any(|r| r.len() != n) {
            return Err("the timed passes ran different op counts".into());
        }
        let mut classes: Vec<(&str, f64)> = [
            "gemm",
            "route",
            "experts",
            "delta",
            "attention",
            "embed_head",
        ]
        .into_iter()
        .map(|c| (c, 0.0))
        .collect();
        let mut total = 0.0;
        for i in 0..n {
            let mut t: Vec<f64> = runs.iter().map(|r| r[i].gpu_s).collect();
            let med = median(&mut t);
            let class = match runs[0][i].op {
                Op::AttnIn(_) | Op::AttnOut(_) | Op::DeltaIn(_) | Op::DeltaOut(_) => "gemm",
                Op::Route(_) => "route",
                Op::Experts(_) => "experts",
                Op::DeltaCore(_) => "delta",
                Op::AttnCore(_) => "attention",
                Op::Embed | Op::Head => "embed_head",
            };
            if let Some(c) = classes.iter_mut().find(|c| c.0 == class) {
                c.1 += med;
            }
            total += med;
        }
        eprintln!(
            "pass_times rows={u} ops={n} runs={PASS_TIMES_RUNS} (median per op, one buffer at a time)"
        );
        for (c, t) in &classes {
            eprintln!(
                "pass_times class={c} ms={:.2} share={:.1}%",
                t * 1e3,
                t / total * 100.0
            );
        }
        eprintln!(
            "pass_times total_ms={:.2} rows_per_s={:.0}",
            total * 1e3,
            u as f64 / total
        );
        let mut walls = Vec::new();
        for run in 0..=PASS_TIMES_RUNS {
            s.reset().map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            s.pass(ids, PassOut::Last, StateOut::default())
                .map_err(|e| e.to_string())?;
            if run > 0 {
                walls.push(t0.elapsed().as_secs_f64());
            }
        }
        s.reset().map_err(|e| e.to_string())?;
        let wall = median(&mut walls);
        eprintln!(
            "pass_wall rows={u} wall_ms={:.2} ops_gpu_ms={:.2} gap_ms={:.2} gap_per_op_us={:.1}",
            wall * 1e3,
            total * 1e3,
            (wall - total) * 1e3,
            (wall - total) / n as f64 * 1e6
        );
        Ok(())
    }

    /// The median of `v`, which holds at least one value.
    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// The fixed work around a pass, warm: one checkpoint's wall after a
    /// pass of the first `u` ids, and the whole prompt's call from position
    /// 0 (`Target::prompt`: its passes and its checkpoints), each the median
    /// of `PASS_TIMES_RUNS` after a warm one. The session is reset before
    /// each and after the last.
    fn call_times_run(s: &mut Session, prompt: &[u32], u: usize) -> Result<(), String> {
        let (mut takes, mut calls) = (Vec::new(), Vec::new());
        for run in 0..=PASS_TIMES_RUNS {
            s.reset().map_err(|e| e.to_string())?;
            s.pass(&prompt[..u], PassOut::Last, StateOut::default())
                .map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            s.checkpoint().map_err(|e| e.to_string())?;
            let take = t0.elapsed().as_secs_f64();
            s.reset().map_err(|e| e.to_string())?;
            let t0 = Instant::now();
            Target::prompt(s, prompt, Want::Argmax).map_err(|e| e.to_string())?;
            let call = t0.elapsed().as_secs_f64();
            if run > 0 {
                takes.push(take);
                calls.push(call);
            }
        }
        s.reset().map_err(|e| e.to_string())?;
        eprintln!(
            "call_times rows={u} checkpoint_ms={:.2} checkpoint_bytes={} prompt_ids={} prompt_warm_ms={:.2}",
            median(&mut takes) * 1e3,
            s.checkpoint_bytes(),
            prompt.len(),
            median(&mut calls) * 1e3
        );
        Ok(())
    }

    /// The `step_times` lines: `n` tokens in each mode past the run —
    /// host-fed and chain greedy, ahead on one id throughout, so every token
    /// ahead is kept — the step clock's phases averaged a mode and printed
    /// as one line, then the count. The lines are development-loop numbers,
    /// like the summary's: the three runs sit at three depths, a mode's
    /// attention reading a few keys more than the mode before it.
    fn step_times_run(s: &mut Session, n: usize, id: u32) -> Result<(), String> {
        let rows = s.ctx() - s.pos() as usize;
        let per = n.min(rows / 3);
        if per == 0 {
            return Err(format!(
                "step_times: {rows} rows past position {} hold no token of each of the three \
                 modes; give --ctx more rows",
                s.pos()
            ));
        }
        s.start_step_clock();
        let mut pick = id;
        for _ in 0..per {
            pick = s
                .step(pick, StepMode::HostFed)
                .map_err(|e| e.to_string())?
                .id;
        }
        for _ in 0..per {
            pick = s.step(pick, StepMode::Chain).map_err(|e| e.to_string())?.id;
        }
        for _ in 0..per {
            s.step(id, StepMode::Ahead { next: Some(id) })
                .map_err(|e| e.to_string())?;
        }
        let clock = s.step_clock().ok_or("the step clock did not run")?;
        for mode in [
            StepMode::HostFed,
            StepMode::Chain,
            StepMode::Ahead { next: None },
        ] {
            let name = match mode {
                StepMode::HostFed => "host",
                StepMode::Chain => "chain",
                StepMode::Ahead { .. } => "ahead",
            };
            let p = clock.phase(mode);
            eprintln!(
                "step_times mode={name} steps={} encode_us={:.1} start_us={:.1} gpu_us={:.1} \
                 wake_us={:.1}",
                p.steps, p.encode_us, p.start_us, p.gpu_us, p.wake_us,
            );
        }
        eprintln!(
            "step_times {per} tokens each of host, chain and ahead (development-loop numbers: \
             no lease, no witness)"
        );
        Ok(())
    }

    pub(super) fn run() -> ExitCode {
        // No lever row acts on this CLI; startup still refuses a retired
        // name that is set, and a bare `--levers` prints the table.
        if let Err(e) = bloomery_levers::at_main_with_flags(&[], has_flag) {
            eprintln!("generate_metal: {e}");
            return ExitCode::from(1);
        }
        if has_flag("--plan") {
            return plan_run();
        }
        let a = match parse() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("generate_metal: {e}");
                return ExitCode::from(2);
            }
        };
        match generate(&a) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("generate_metal: {e}");
                ExitCode::from(1)
            }
        }
    }

    /// `--plan`: the header's plan on stdout, the code it ends with. It takes
    /// `--ctx`, the model's flags and no other flag but `--help`, which
    /// `parse` answers first in a generation run: a flag that would do
    /// nothing here is refused by name.
    fn plan_run() -> ExitCode {
        if let Some(name) = unknown_flag() {
            eprintln!("generate_metal: {name} is not a flag of this CLI");
            return ExitCode::from(2);
        }
        if has_flag("--help") {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        let other = std::env::args().skip(1).find(|a| {
            (VALUE_FLAGS.contains(&a.as_str()) && !["--ctx", "-m", "--hf"].contains(&a.as_str()))
                || (BARE_FLAGS.contains(&a.as_str()) && a != "--plan")
                || COUNT_FLAGS.contains(&a.as_str())
        });
        if let Some(name) = other {
            eprintln!("generate_metal: --plan reads the header alone; {name} is a generation flag");
            return ExitCode::from(2);
        }
        let ctx = match ctx_flag() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("generate_metal: {e}");
                return ExitCode::from(2);
            }
        };
        let source = match model_source() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("generate_metal: {e}");
                return ExitCode::from(2);
            }
        };
        let model = match model_arg::resolve(&source) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("generate_metal: --plan: {e}");
                return ExitCode::from(1);
            }
        };
        match plan_of(&model.path, ctx) {
            Ok(printed) => {
                for line in &printed.lines {
                    println!("{line}");
                }
                ExitCode::from(printed.code)
            }
            Err(e) => {
                eprintln!("generate_metal: --plan: {e}");
                ExitCode::from(e.code())
            }
        }
    }

    fn plan_of(
        model: &std::path::Path,
        ctx: Option<usize>,
    ) -> Result<plan::Printed, plan::OpenError> {
        let header = plan::open_plan_header(model)?;
        let ctx = match ctx {
            Some(c) => c,
            None => default_ctx(header.split(), model).map_err(plan::OpenError::Unreadable)?,
        };
        let outcome = check::check(header.split(), ctx);
        Ok(plan::render(
            model,
            header.data,
            ctx,
            &outcome,
            &plan::device_room(),
            &plan::HostMemory::probe(),
        ))
    }

    /// The summary lines of one run: the prompt's and the decode's walls and
    /// rates, why the run stopped and the residency's request count. `run` is
    /// the run's place when `--repeat` asked for several, and opens the block.
    fn print_run(
        out: &Generated,
        prompt_ids: usize,
        run: Option<usize>,
        ignore_eos: bool,
        session: &Session,
    ) {
        let prompt_s = out.prompt_wall.as_secs_f64();
        let decode_s = out.decode_wall.as_secs_f64();
        if let Some(run) = run {
            eprintln!("run={run}");
        }
        eprintln!("prompt_s={prompt_s:.3}");
        eprintln!("prompt_tok_s={:.1}", prompt_ids as f64 / prompt_s);
        eprintln!("decode_tokens={}", out.tokens());
        eprintln!("decode_passes={}", out.passes);
        eprintln!("decode_s={decode_s:.3}");
        if out.passes == 0 {
            eprintln!("decode_tok_s=none");
        } else {
            eprintln!("decode_tok_s={:.1}", out.passes as f64 / decode_s);
        }
        if ignore_eos {
            eprintln!("ignore_eos=true");
        }
        eprintln!("stop={}", out.stop.name());
        if let Some((first, rest)) = out.pass_walls.split_first() {
            eprintln!("decode_first_ms={:.2}", first.as_secs_f64() * 1e3);
            if !rest.is_empty() {
                let mean = rest.iter().map(Duration::as_secs_f64).sum::<f64>() / rest.len() as f64;
                eprintln!("decode_rest_ms={:.2}", mean * 1e3);
            }
        }
        eprintln!("residency_requests={}", session.residency().requests);
    }

    fn generate(a: &Args) -> Result<(), String> {
        let model = model_arg::resolve(&a.model)?;
        let file = ModelFile::open(&model.path).map_err(|e| e.to_string())?;
        // The tokenizer decodes what the run streams and owns the
        // end-of-generation set, so every run opens it, whatever named the
        // prompt.
        let tok = file.tokenizer().map_err(|e| e.to_string())?;
        let prompt = match &a.prompt {
            Prompt::Text(t) => tok.encode(t, true, false),
            Prompt::Ids(v) => v.clone(),
        };
        if prompt.is_empty() {
            return Err("the prompt has no ids".into());
        }
        // The cache rows: the asked count, else the smaller of the default cap and the file's
        // `context_length`.
        let ctx = match a.ctx {
            Some(c) => c,
            None => default_ctx(file.header(), &model.path)?,
        };
        // The library is the instance the file's shape selects, by the check `--plan` prints.
        let instance = Session::instance_of_model(&file, ctx).map_err(|e| e.to_string())?;
        if instance.name == bloomery_metal::instance::UNREGISTERED {
            eprintln!(
                "metal: {instance}: no registered instance holds this shape, so no gate checked its output"
            );
        }
        let device = MetalDevice::open(instance).map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        let mut session = Session::open_model(&device, device.library(), &file, ctx)
            .map_err(|e| e.to_string())?;
        // The session holds its own buffers: the file's mapping is not kept for the run.
        drop(file);
        session.set_prefill(a.prefill);
        let open_s = t0.elapsed().as_secs_f64();
        #[cfg(feature = "encode-trace")]
        if a.encode_trace.is_some() {
            bloomery_metal::trace::start();
        }
        // The driver's own `currentAllocatedSize` is the one count of the
        // live device buffers this process holds; the arena keeps no sum of
        // its own.
        let allocated = device.allocated_bytes().map_err(|e| e.to_string())?;
        // The picks stop at the vocabulary's end-of-generation set unless the run ignores it.
        let eog: &[u32] = if a.ignore_eos { &[] } else { tok.eog() };
        // One run of the prompt, its text or ids on stdout as they are picked.
        let run_once = |session: &mut Session| -> Result<Generated, String> {
            let mut stream = Stream {
                decoder: (!a.print_ids).then(|| tokenizer::Decoder::new(&tok, true)),
            };
            let out = greedy(session, &prompt, a.n, eog, |id| stream.id(id))
                .map_err(|e| e.to_string())?;
            #[cfg(feature = "encode-trace")]
            if let Some(path) = &a.encode_trace {
                let mut text = bloomery_metal::trace::finish();
                text.push_str(&format!(
                    "# run prompt_ids={} decode_tokens={} gemm_path={}\n",
                    prompt.len(),
                    out.tokens(),
                    device.library().gemm_path().name()
                ));
                std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
            }
            if a.print_ids {
                let ids: Vec<String> = out.ids.iter().map(u32::to_string).collect();
                println!("{}", ids.join(","));
            } else {
                stream.finish();
                if a.repeat > 1 {
                    println!();
                }
            }
            Ok(out)
        };
        let mut out = run_once(&mut session)?;
        eprintln!("# development-loop numbers: no lease, no witness");
        eprintln!("model={}", model.path.display());
        eprintln!("gemm_path={}", device.library().gemm_path().name());
        eprintln!("open_s={open_s:.3}");
        let split = session.open_split();
        eprintln!("open_load_s={:.3}", split.load.as_secs_f64());
        eprintln!("open_rows_ms={:.2}", split.rows.as_secs_f64() * 1e3);
        eprintln!(
            "open_residency_ms={:.2}",
            split.residency.as_secs_f64() * 1e3
        );
        eprintln!(
            "open_pipelines_ms={:.2}",
            split.pipelines.as_secs_f64() * 1e3
        );
        eprintln!("device_allocated_bytes={allocated}");
        let resident = session.residency();
        eprintln!("resident_bytes={}", resident.bytes);
        eprintln!("rows_arena_bytes={}", session.rows_arena_bytes());
        eprintln!("prompt_ids={}", prompt.len());
        let first = (a.repeat > 1).then_some(1);
        print_run(&out, prompt.len(), first, a.ignore_eos, &session);
        for run in 2..=a.repeat {
            session.reset().map_err(|e| e.to_string())?;
            out = run_once(&mut session)?;
            print_run(&out, prompt.len(), Some(run), a.ignore_eos, &session);
        }
        let decode_s = out.decode_wall.as_secs_f64();
        if let Some(p) = &a.profile {
            let id = *out
                .ids
                .last()
                .ok_or("a profile of no token: the run picked none")?;
            let decode_ms = (out.passes > 0).then(|| decode_s / out.passes as f64 * 1e3);
            profile_run(&mut session, p, id, decode_ms)?;
        }
        if let Some(us) = &a.pass_times {
            for &u in us {
                pass_times_run(&mut session, &prompt, u)?;
            }
            if let Some(&u) = us.last() {
                call_times_run(&mut session, &prompt, u)?;
            }
        }
        if let Some(n) = a.step_times {
            let id = *out
                .ids
                .last()
                .ok_or("a step-times run of no token: the run picked none")?;
            step_times_run(&mut session, n, id)?;
        }
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
fn main() -> std::process::ExitCode {
    eprintln!("generate_metal: refused: the Metal CLI runs on macOS only");
    std::process::ExitCode::from(78)
}
