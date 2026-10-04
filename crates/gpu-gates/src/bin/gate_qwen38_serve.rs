//! `gate_qwen38_serve` — `bloomery-serve-qwen38` on the 3090 (placement
//! gate), driven over HTTP, as one process under the GPU gate lock.
//!
//!     gate_qwen38_serve --gen <generate_qwen3moe log> --prompt <text>
//!                       --ids <a,b,…> --dir <out>
//!
//! Starts the server beside this binary (`--host 127.0.0.1 --port 0 --place
//! gate --ctx-size 4096`), reads its address from its stderr, waits for
//! `/health`, then checks:
//!
//! - `/props`' `engine` object, printed once, against this gate's own plan of
//!   the file the server opens (the gate card, the server's default context,
//!   the expert rule the inherited levers name): the server's name, argv and
//!   pid; the model's architecture (`qwen4exp`), shards, bytes on disk and
//!   counts as the header states them; a card `GPU<n>` and the host `CPU`,
//!   each device's `bytes` the sum of its classes and equal to the plan's
//!   card (dense + experts) and host (experts + tables) bytes; the cards' KV
//!   bytes; no draft; `ctx_verified` the deepest reference set's
//!   positions (`refset::arch::qwen4exp::VERIFIED_POSITIONS`);
//! - `/completion` of `--prompt` at temperature 0 with `return_tokens`: its
//!   ids are `generate_qwen3moe --tokens <--ids> -n 16`'s `tokens` line — all
//!   16, or a prefix ending in the end-of-generation id when the server
//!   stopped there (`generate_qwen3moe` does not stop at it). The prompt must
//!   hold at most eight ids: the passes both engines take below
//!   [`GEMM_FROM`](bloomery_gpu::arch::qwen3moe::Prompt38::GEMM_FROM) are bit
//!   for bit steps, so the server's prompt feed (the prompt less its last id)
//!   and `generate_qwen3moe`'s (the whole prompt) leave the same state, while
//!   past that each runs the prompt's last position through a different arm;
//! - the same `/completion` again, and once more after the other requests
//!   below: the same ids each time (a request that does not extend the held
//!   sequence keeps the checkpoint at its prompt call's end, or under the
//!   draft prefills from a reset — never a wrong state);
//! - `/v1/chat/completions` of one user turn at temperature 0: the streamed
//!   deltas concatenate to the non-streamed content, and the stream ends with
//!   `data: [DONE]`;
//! - `/tokenize` of `--prompt` is `--ids`;
//! - the positions the server serves: `/props`' `n_ctx` is the context it
//!   was started at; a prompt of that many ids is a 400
//!   (`exceed_context_size_error`, naming it) and the server stays up;
//! - `cache` ([`cache`]): two sessions' requests through the prompt cache —
//!   a verbatim resend after the other session keeps every held position
//!   (under the draft the turn's prompt end, its last window having fed
//!   rows past the reply) and gives the ids of the same session run with no
//!   switch; a resend with the reply's reasoning stripped keeps the
//!   checkpoint at the first turn's end and gives a fresh run's ids; under
//!   the draft a cut keeps that checkpoint only at or past the seat's
//!   break-even for the reply (its `draft keep` line), drafting nothing and
//!   saying why by name, and below it resets with the draft on; both sides
//!   of the break-even at the turn's end, by reply length, each with a fresh
//!   run's ids and its `mtp keep` record;
//! - under `BLOOMERY_DRAFT=mtp`, requests that extend the sequence the
//!   server holds (`continued`): each keeps the held prefix (`cache_n`), a
//!   prompt call past it prints one `mtp prompt` record — the draft caught
//!   up at the kept position, nothing skipped — and none prints a skip, and
//!   each drafts, with the plain run's ids (the generated ones); the prompt
//!   resent keeps its end by a cut, after which the extension drafts
//!   nothing, by name, with the same prompt's ids fed fresh; `/props`' `engine.draft` names the draft file
//!   `refset::arch::qwen4exp::mtp::draft_file` picks, by name and path;
//!   and a sampled `/completion` (temperature 0.8, a fixed seed) is served
//!   through plain steps, drafting nothing, and the same request at `top_k`
//!   1 gives this server's greedy ids;
//! - the seats' `parallel` lines ([`parallel_agrees`]): the resident-slot
//!   rule names its slots, and the line's own terms hold them — the slots
//!   are the flag's (`--parallel 2`) or the default's (2, the `ctx` clause's
//!   flagless server), and the split they serve (`slot_ctx`, `total`) is the
//!   `ctx` line's own context times the slots;
//! - the slots clause ([`slots_flow_together`], a server of its own under
//!   `BLOOMERY_DRAFT=mtp`, `BLOOMERY_RESIDENCY=off`): two greedy streamed
//!   requests of distinct prompts on `--parallel 2`, each first run alone
//!   on that same server and then both together —
//!   (v1) each request's together ids are its alone ids, and its
//!   `draft_n`/`draft_n_accepted` are its alone run's: the slots hold
//!   separate sequences, and the draft's host state is parked with its slot
//!   on every switch ([`Seat::select`]) so a slot's passes are its own —
//!   the coverage the swap clause held (a preempted request's draft
//!   rejoining) lives here now: the rejoin is the per-slot park, and this
//!   equality is red without it;
//!   (v2) `swaps_total` is absent or 0 (nothing parks), and over the
//!   together run the deltas of the busy slots booked per engine call
//!   (busy total = `n_busy_slots_per_decode` × `n_decode_total`) sit at ≥
//!   1.5: the worker books one call a round carrying every running slot
//!   (`worker.rs`'s `Stats`), so two drafted requests that decode together
//!   book 2 busy a call — at 96 tokens each and a mean ~2.5 kept ids a
//!   drafted pass, ~40 rounds carry both and a handful of head and tail
//!   rounds one, (2·N + ~4)/(N + ~4) ≈ 1.9 — while a turn-taking server
//!   books exactly 1 busy a call, 1.0;
//!   (v3) the two SSE streams' token arrivals, timestamped by a reader
//!   thread a stream: while both are live, every window of
//!   [`SLOTS_WINDOW`] = 16 consecutive arrivals holds at least
//!   [`SLOTS_EACH`] = 4 of each — one drafted pass a slot a round keeps
//!   1–4 ids (`advance_rows` = 4), so ideal delivery alternates in runs of
//!   at most 4, and batching (the server flushes per event; curl and the
//!   pipe may clump) tolerates up to 12 consecutive same-stream arrivals
//!   before the minority drops under 4, while a turn-taking server's
//!   64-token turns put windows of 16 holding 16 of one stream and none of
//!   the other;
//!   (v4) the `load` line names `slots=2` and `slot_ctx` = the `--ctx-size`
//!   the server was started at over two (2048 of the 4096 total), the
//!   `listening` record the same split, and `/props`' `n_ctx` is the slot
//!   context.
//!   Then one more server of `--parallel 2` under the seat's own defaults
//!   — `--place a`, no `BLOOMERY_DRAFT`, no `BLOOMERY_RESIDENCY` (the
//!   user's case: the A6000 stages, the adaptive residency runs, the draft
//!   runs) — runs the together pair once and holds (v2) and (v3) alone,
//!   no id equality: under the adaptive residency the other stream moves
//!   experts between the host and the card, so a stream's bits need not
//!   equal its alone run's.
//!   The swap clause this replaces is gone: the turn path it held
//!   (`serve::SwapEngine`, the park of a preempted request's state) left
//!   the seat with the resident slots — no load of this seat takes turns
//!   any more (every plan counts its sequences
//!   ([`PlanInputs::plan_with_slots`]), a plan that cannot hold them is
//!   refused by name), so nothing remains for it to check; a preempted
//!   request cannot happen (no slot waits for another), and the draft's
//!   park-and-rejoin under a switch is (v1)'s draft-count equality.
//!   FAIL-first mutants, each red on its line: the seat still building the
//!   turn-taking engine leaves (v2) at ~1.0 and (v3) failing; a select
//!   that parks no draft state leaves (v1)'s draft counts red (a slot's
//!   pass refuses or drafts another's rows); a reply that hands row
//!   answers back in the wrong order scrambles (v1)'s ids;
//!   `slot_drafts` false ends the server at `check_slots`'s named refusal
//!   before it listens; a context search that ignores the slot count loads
//!   a plan of one sequence and makes the second slot's load fail (the
//!   server never listens) or (v4) red.
//!
//! Then the server is killed by the handle this binary spawned it with and
//! waited for, and more servers start on the card, one at a time:
//!
//! - `cache` again on a server with the draft the first did not run
//!   (`BLOOMERY_DRAFT=mtp` when it ran `off`, and the other way);
//! - `ctx` ([`ctx`]): a server with no `--ctx-size` prints its default
//!   (the margin rule's answer — or the largest the card holds when that is
//!   fewer, or the prompt cache's clamp of it), the largest context the card
//!   holds with its card expert bytes and the largest within the plan's
//!   margin, before its load, each the rule's against this gate's own plans
//!   (stopped there), the fit at most the file's serving cap
//!   (`place::serve_ctx`), and one asked for a position past the fit is
//!   refused by name before it listens (the card's rule, or the cap's when
//!   the card holds more);
//! - `residency`: `bloomery-serve-qwen38` with the same arguments under
//!   `BLOOMERY_RESIDENCY=mid-p0-s1` (`BLOOMERY_DRAFT=off`, the MTP levers
//!   removed; the word set explicitly, one the lever takes): it loads and
//!   prints the word as a `residency lever` record (`why=set`) and a
//!   `residency host` record; the greedy `/completion` above ends a prompt
//!   pass and step passes, as `residency pass` records say; `POST
//!   /residency/reset` is a 200 whose `diff` is 0, and the server prints its
//!   `residency reset` record with the same counts; the same `/completion`
//!   after the reset gives the first one's ids. Not the first server's ids: a
//!   flip moves an expert from the host to the card, whose sums round another
//!   way, so a greedy id can move at a near tie; from the seed, one history
//!   lands the same flips at the same passes, so two runs of it are bit for
//!   bit. Its logs are in `<dir>/residency` (mutants: the seat opens with
//!   `Body38::open_placed`, which runs no machine whatever the lever — no
//!   `residency pass` record, and the reset is the server's 501; a reset that
//!   leaves the learned residency — the second request runs on the flips the
//!   first one made).
//!
//! Logs and the raw stream go to `--dir`.
//!
//! The server inherits this binary's environment, so the levers it acts on
//! are the server's (`ACTS_ON`, the same list): one the server would refuse
//! is refused here, at `main`, before the server starts. The gate sets two of
//! them itself on every server it starts, so neither the placement's defaults
//! nor the environment move what a clause means: `BLOOMERY_DRAFT` (`mtp` when
//! this binary's environment names it, else `off`; the sampled clause's plain
//! server `off`) and `BLOOMERY_RESIDENCY` (`off`, the residency clause's
//! word there) — the slots clause's second server alone sets neither: it is
//! the seat under its own defaults (the user's case).
//! `BLOOMERY_RESIDENCY` set in this binary's environment is
//! refused by name.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen38_serve: built without the `deepseek41` feature; see `just gate-gpu-qwen38-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_serve", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::fs::File;
    use std::os::unix::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::mpsc;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{checkpoint_bytes, seq_positional_bytes};
    use bloomery_gpu_gates::record;
    use bloomery_gpu_gates::serve_client::{curl, ids_of, json_of, parse_ids};
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use gguf::Split;
    use model::arch::models::Mixer;
    use model::arch::qwen35moe::head_list::head_rows_of;
    use model::arch::qwen35moe::place::{
        Experts, MtpInputs, PlanInputs, machine_for_experts, serve_ctx,
    };
    use model::placement::PlanLevers;
    use model::placement::workstation::RTX_3090;
    use refset::arch::qwen4exp::VERIFIED_POSITIONS;
    use refset::arch::qwen4exp::mtp::draft_file;
    use serde_json::{Value, json};
    use threads::helper::{Placement, spawn_helper};

    /// The levers the server acts on — the same list
    /// `bloomery_serve_qwen38` parses, kept one with it: this gate starts the
    /// server with its own environment, so a lever the server would refuse is
    /// refused here first.
    const ACTS_ON: &[&str] = &[
        bloomery_levers::QWEN38_EXPERTS,
        bloomery_levers::CARD_BUDGET,
        bloomery_levers::PIN_MAIN,
        bloomery_levers::HOST_POPULATE,
        bloomery_levers::HOST_LOCK,
        bloomery_levers::CARD_DONTNEED,
        bloomery_levers::R8,
        bloomery_levers::DRAFT,
        bloomery_levers::MTP_HEAD_ROWS,
        bloomery_levers::MTP_DRAFT,
        bloomery_levers::RESIDENCY,
    ];

    const USAGE: &str = "usage: gate_qwen38_serve --gen <generate_qwen3moe log> --prompt <text> \
                         --ids <a,b,…> --dir <out>";

    /// The server's arguments after its path; `/props` must echo them. The
    /// context is named: the clauses below count on [`CTX`] positions, and
    /// the default is the `ctx` clause's. The slot actions need a save
    /// directory; the gate asks only for `erase`, which writes nothing. The
    /// one slot is named: the clauses below hold the one-slot path's prompt
    /// cache, whose choreography — which state the slot-0 `erase` drops,
    /// what the cache can serve a resend — the streams of several slots
    /// would move with a request-arrival race (the free slots' LRU); the
    /// slots clause's server runs the two-slot streams on its own, and the
    /// `ctx` clause's flagless server holds the resident default's own
    /// lines.
    const SERVER_ARGS: [&str; 12] = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--place",
        "gate",
        "--ctx-size",
        "4096",
        "--slot-save-path",
        "/tmp",
        "--parallel",
        "1",
    ];
    /// The same arguments with no context named: the default's.
    const DEFAULT_ARGS: [&str; 6] = ["--host", "127.0.0.1", "--port", "0", "--place", "gate"];
    /// The load takes tens of seconds; the bound is the spec's 120 polls × 5 s.
    const POLLS: usize = 120;
    const POLL: Duration = Duration::from_secs(5);
    /// The greedy requests' length, `generate_qwen3moe -n 16`'s.
    const N_PREDICT: usize = 16;
    /// The context the servers are started at (`SERVER_ARGS`), and the one
    /// the default's expert cost is counted against.
    const CTX: usize = 4096;
    /// The default context's multiple.
    const CTX_STEP: usize = 256;
    /// Session A's first turn, session B's, a later turn A resends after its
    /// reply (`cache` clause (a)), and the text A resends instead of its
    /// reply (clause (b)): the last's first id is not the reply's.
    const TURN_A: &str = "Name three rivers that flow through Germany and say in one sentence \
                          which of them is the longest.";
    const TURN_B: &str = "Write a haiku about a lighthouse in winter.";
    const LATER_A: &str = "<|im_end|>\n<|im_start|>user\nAnd which of them reaches the sea \
                           first?<|im_end|>\n<|im_start|>assistant\n";
    const STRIPPED_A: &str = "The Rhine, the Danube and the Elbe; the Danube is the longest.\
                              <|im_end|>\n<|im_start|>user\nAnd which of them reaches the sea \
                              first?<|im_end|>\n<|im_start|>assistant\n";
    /// The break-even clause's two sessions, one a side, each a first turn and
    /// the text it resends in place of its reply: no other clause sends them,
    /// so no state the prompt cache holds shares more than their turn.
    const BREAK_EVEN_TURNS: [(&str, &str); 2] = [
        (
            "Name three mountains of the Alps and say in one sentence which of them is the \
             highest.",
            "Mont Blanc, the Matterhorn and the Eiger; Mont Blanc is the highest.<|im_end|>\n\
             <|im_start|>user\nAnd which of them was climbed first?<|im_end|>\n\
             <|im_start|>assistant\n",
        ),
        (
            "Name three lakes of Italy and say in one sentence which of them is the largest.",
            "Lake Garda, Lake Como and Lake Maggiore; Lake Garda is the largest.<|im_end|>\n\
             <|im_start|>user\nAnd which of them lies furthest north?<|im_end|>\n\
             <|im_start|>assistant\n",
        ),
    ];
    /// The words of the seat's line that says the MTP draft proposes nothing
    /// after a cut or a state put back.
    const DRAFT_OFF: &str = "the MTP draft proposes nothing from position";
    /// The slots clause's second prompt: `TURN_A`'s shape over another
    /// country's rivers — distinct from `TURN_A` from its first content ids,
    /// and its greedy reply runs as long.
    const SLOTS_B: &str = "Name three rivers that flow through Russia and say in one sentence \
                           which of them is the longest.";
    /// The slots clause's requests: long enough that the together run holds
    /// dozens of rounds with both streams live (and `TURN_A`'s greedy reply
    /// runs past it without an end-of-generation id — no `ignore_eos`, which
    /// steps a request plainly).
    const SLOTS_PREDICT: usize = 96;
    /// The slots clause's interleave window and its floor for each stream
    /// (the module header's (v3) derivation).
    const SLOTS_WINDOW: usize = 16;
    const SLOTS_EACH: usize = 4;
    /// The one chat turn the chat clauses send.
    const CHAT: &str = "What is the capital of France? Answer in one word.";
    /// The chat's reply length.
    const CHAT_PREDICT: usize = 32;
    /// The sampled request's temperature (llama-server's default) and seed.
    const SAMPLED_TEMPERATURE: f64 = 0.8;
    const SAMPLED_SEED: u64 = 42;
    /// The residency clause's word: set explicitly, one the lever takes; no
    /// seed expert pinned, one spare a layer.
    const RESIDENCY_WORD: &str = "mid-p0-s1";

    /// `BLOOMERY_QWEN38_EXPERTS` as the plan's expert rule — the server's
    /// reading of the inherited environment.
    fn experts38(levers: &bloomery_levers::Levers) -> Result<Experts, GateError> {
        match levers.qwen38_experts() {
            "host" => Ok(Experts::Host),
            "card" => Ok(Experts::Card),
            other => Err(format!("BLOOMERY_QWEN38_EXPERTS={other}: host or card").into()),
        }
    }

    /// The server this binary started; killed and reaped on every way out.
    /// `serve_client::Served`'s core with this gate's server's name — that
    /// module's `exe` names `bloomery-serve-ds41`.
    struct Served38 {
        child: Child,
    }

    impl Served38 {
        /// The server's path: `bloomery-serve-qwen38` beside this binary.
        fn exe() -> Result<PathBuf, GateError> {
            Ok(std::env::current_exe()?.with_file_name("bloomery-serve-qwen38"))
        }

        /// Starts `bloomery-serve-qwen38` beside this binary with `args` on
        /// `cmd` (the server with the environment the caller set on it),
        /// stdout to `<dir>/server.out` and stderr to `<dir>/server.err`. The
        /// child is killed when this process dies, so a runner's bound that
        /// ends this process does not leave the server holding a card.
        fn spawn_with(args: &[&str], dir: &Path, cmd: &mut Command) -> Result<Served38, GateError> {
            let exe = Self::exe()?;
            cmd.args(args)
                .stdin(Stdio::null())
                .stdout(File::create(dir.join("server.out"))?)
                .stderr(File::create(dir.join("server.err"))?);
            // SAFETY: the closure runs in the child between fork and exec and
            // calls only `prctl`, which is async-signal-safe and touches no
            // memory of ours.
            unsafe {
                cmd.pre_exec(|| {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                });
            }
            let child = cmd
                .spawn()
                .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
            Ok(Served38 { child })
        }

        /// Waits for the `listening on http://<addr>` line in the server's
        /// stderr, `polls` reads `poll` apart.
        fn address(
            &mut self,
            err_log: &Path,
            polls: usize,
            poll: Duration,
        ) -> Result<String, GateError> {
            for _ in 0..polls {
                let text = std::fs::read_to_string(err_log).unwrap_or_default();
                if let Some(addr) = text
                    .lines()
                    .find_map(|l| l.split_once("listening on http://").map(|(_, a)| a.trim()))
                {
                    return Ok(addr.to_owned());
                }
                if let Some(status) = self.child.try_wait()? {
                    return Err(format!(
                        "the server exited ({status}) before listening; {}:\n{text}",
                        err_log.display()
                    )
                    .into());
                }
                std::thread::sleep(poll);
            }
            Err(format!("the server did not listen within {polls} polls").into())
        }

        fn stop(&mut self) -> Result<String, GateError> {
            self.child.kill()?;
            Ok(format!("{}", self.child.wait()?))
        }
    }

    impl Drop for Served38 {
        fn drop(&mut self) {
            if matches!(self.child.try_wait(), Ok(None)) {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    /// The `tokens [..]` line of a `generate_qwen3moe` log.
    fn gen_tokens(log: &Path) -> Result<Vec<u32>, GateError> {
        let text = std::fs::read_to_string(log).map_err(|e| format!("{}: {e}", log.display()))?;
        let line = text
            .lines()
            .find_map(|l| l.strip_prefix("tokens "))
            .ok_or_else(|| format!("{}: no `tokens` line", log.display()))?;
        parse_ids(line)
    }

    /// `ids` agree with `reference`, a run that did not stop at the
    /// end-of-generation id: all of them, or, when `stop` is `eos`, a prefix
    /// ending there.
    fn agree(ids: &[u32], stop: &str, reference: &[u32]) -> bool {
        match stop {
            "eos" => !ids.is_empty() && reference.starts_with(ids),
            _ => ids == reference,
        }
    }

    struct Args {
        gen_log: PathBuf,
        prompt: String,
        ids: Vec<u32>,
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut gen_log, mut prompt, mut ids, mut dir) = (None, None, None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--gen" => gen_log = Some(PathBuf::from(v)),
                "--prompt" => prompt = Some(v),
                "--ids" => ids = Some(parse_ids(&v)?),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (gen_log, prompt, ids, dir) {
            (Some(gen_log), Some(prompt), Some(ids), Some(dir)) => Ok(Args {
                gen_log,
                prompt,
                ids,
                dir,
            }),
            _ => Err(USAGE.into()),
        }
    }

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    /// `/props`' `engine` object (the module header) against the plan of the
    /// file the server opens, made here from its headers the way the server
    /// makes it, under the levers the server inherits; `argv` and `pid` are
    /// the process this gate spawned.
    fn props_engine(
        url: &dyn Fn(&str) -> String,
        argv: &[String],
        pid: u32,
        levers: &bloomery_levers::Levers,
        err_log: &Path,
    ) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let e = json_of("/props", st, &body)?["engine"].clone();
        println!("props engine {e}");
        let mtp = levers.draft() == Some("mtp");
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|err| format!("open {}: {err}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let experts = experts38(levers)?;
        let ub = ubatch_for(CTX)?;
        let mut machine = machine_for_experts(
            RTX_3090,
            inputs.spec.layers.len(),
            u64::try_from(ub)?,
            experts,
        );
        machine.cards[0].free_bytes = Some(server_card_free(err_log)?);
        // Under the draft the plan carries it (its granules, its store, its
        // row map and its program's arena beside the target's card terms),
        // and `/props` files its bytes as the card's `draft` class.
        let levers_plan = PlanLevers::from_levers(levers)?;
        let (draft_path, from) = draft_file(levers.mtp_draft(), &path);
        let terms = |dense: u64, experts_at: u64, host: u64, tables: u64, kv: u64, draft: u64| {
            (dense + experts_at + draft, host + tables, kv, draft)
        };
        let (card_bytes, host_bytes, kv, draft_bytes) = match mtp {
            false => {
                let plan =
                    inputs.plan_with(&machine, u64::try_from(CTX)?, &levers_plan, experts)?;
                let c = plan.cards.first().ok_or("the gate's plan has no card")?;
                terms(
                    c.dense_bytes,
                    c.expert_bytes,
                    plan.host.expert_bytes,
                    plan.host.table_bytes,
                    plan.cards.iter().map(|c| c.kv_bytes).sum(),
                    0,
                )
            }
            true => {
                let rows = head_rows_of(levers.mtp_head_rows(), &split, inputs.spec.vocab)?.rows;
                let draft = Split::open(&draft_path).map_err(|e| {
                    format!(
                        "open the MTP draft {} ({}): {e}",
                        draft_path.display(),
                        from.describe()
                    )
                })?;
                let mtp = MtpInputs::read(&draft, &split, &inputs, rows)?;
                let with = inputs.plan_mtp_with(
                    &machine,
                    u64::try_from(CTX)?,
                    &levers_plan,
                    &mtp,
                    experts,
                )?;
                let bytes = with.draft_card_bytes() + with.arena_bytes;
                let c = with
                    .plan
                    .cards
                    .first()
                    .ok_or("the gate's plan has no card")?;
                terms(
                    c.dense_bytes,
                    c.expert_bytes,
                    with.plan.host.expert_bytes,
                    with.plan.host.table_bytes,
                    with.plan.cards.iter().map(|c| c.kv_bytes).sum(),
                    bytes,
                )
            }
        };
        let mut file_bytes = 0u64;
        for i in 0..split.shard_count() {
            let shard = split.shard_path(i).ok_or("a shard without a path")?;
            file_bytes += std::fs::metadata(shard)?.len();
        }
        println!(
            "plan card bytes {card_bytes} (the draft's {draft_bytes} of it) host bytes \
             {host_bytes} kv {kv}; file {file_bytes} B in {} shards",
            split.shard_count()
        );
        let devices = e["placement"]["devices"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let class_sum = |d: &Value| {
            d["classes"]
                .as_object()
                .map(|c| c.values().filter_map(Value::as_u64).sum::<u64>())
        };
        let layers = format!("{}-{}", 0, inputs.spec.layers.len().saturating_sub(1));
        let m = &e["model"];
        let mut ok = true;
        check(
            &mut ok,
            "props_engine_identity",
            e["name"] == "bloomery"
                && e["version"]
                    .as_str()
                    .is_some_and(|v| !v.is_empty() && !v.ends_with(" mock"))
                && e["args"] == json!(argv)
                && e["server_pid"] == json!(pid),
        );
        check(
            &mut ok,
            "props_engine_model",
            m["format"] == "gguf"
                && m["arch"] == json!(split.architecture())
                && m["files"] == json!(split.shard_count())
                && m["bytes"] == json!(file_bytes)
                && m["n_layers"] == json!(inputs.model.layers)
                && m["n_experts"] == json!(inputs.model.experts)
                && m["n_experts_used"] == json!(inputs.model.experts_used)
                && m["ctx_train"] == json!(split.arch_get_u64("context_length"))
                && m["quant"].as_str().is_some_and(|q| !q.is_empty()),
        );
        check(
            &mut ok,
            "props_engine_devices",
            devices.len() == 2
                && devices[0]["device"]
                    .as_str()
                    .and_then(|d| d.strip_prefix("GPU"))
                    .is_some_and(|n| n.parse::<u32>().is_ok())
                && devices[0]["layers"] == json!(layers)
                && devices[1]["device"] == "CPU",
        );
        check(
            &mut ok,
            "props_engine_bytes_are_the_plans",
            devices
                .iter()
                .all(|d| d["bytes"].as_u64().is_some_and(|b| Some(b) == class_sum(d)))
                && devices
                    .first()
                    .is_some_and(|d| d["bytes"] == json!(card_bytes))
                && devices
                    .get(1)
                    .is_some_and(|d| d["bytes"] == json!(host_bytes)),
        );
        check(
            &mut ok,
            "props_engine_vram_kv",
            e["placement"]["vram_kv_bytes"] == json!(kv),
        );
        check(
            &mut ok,
            "props_engine_ctx_verified",
            e["ctx_verified"] == json!(VERIFIED_POSITIONS),
        );
        match mtp {
            false => check(&mut ok, "props_engine_no_draft", e.get("draft").is_none()),
            true => {
                let d = &e["draft"];
                check(
                    &mut ok,
                    "props_engine_names_the_draft",
                    d["kind"] == json!("mtp")
                        && d["n_max"] == json!(3)
                        && draft_path
                            .file_name()
                            .is_some_and(|n| d["model"] == json!(n.to_string_lossy()))
                        && d["path"] == json!(draft_path.display().to_string())
                        && devices
                            .first()
                            .is_some_and(|d| d["classes"]["draft"].as_u64() == Some(draft_bytes)),
                );
            }
        }
        Ok(ok)
    }

    /// `/props`' `n_ctx` is the server's default context, a prompt of that
    /// many ids is a 400 naming it, and the server answers `/health` after it.
    fn position_limit(url: &dyn Fn(&str) -> String, id: u32) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let n_ctx = json_of("/props", st, &body)?["default_generation_settings"]["n_ctx"].clone();
        let long = vec![id; CTX];
        let (st, body) = curl(
            &url("/completion"),
            Some(&json!({"prompt": long, "n_predict": 1, "temperature": 0})),
            false,
        )?;
        let refused: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let e = &refused["error"];
        println!(
            "positions served {CTX}: /props n_ctx {n_ctx}; a prompt of {CTX} ids: HTTP {st} {e}"
        );
        let (hst, hbody) = curl(&url("/health"), None, false)?;
        let mut ok = true;
        check(
            &mut ok,
            "props_n_ctx_is_the_default_context",
            n_ctx == json!(CTX),
        );
        check(
            &mut ok,
            "prompt_of_the_served_positions_is_a_400",
            st == 400
                && e["type"] == "exceed_context_size_error"
                && e["message"]
                    .as_str()
                    .is_some_and(|m| m.contains(&CTX.to_string())),
        );
        check(
            &mut ok,
            "health_after_the_400",
            hst == 200 && hbody.contains("\"ok\""),
        );
        Ok(ok)
    }

    /// The records on the server's stderr so far whose line starts with
    /// `head` and a space.
    fn records(err_log: &Path, head: &str) -> Result<Vec<String>, GateError> {
        let prefix = format!("{head} ");
        Ok(std::fs::read_to_string(err_log)?
            .lines()
            .filter(|l| l.starts_with(&prefix))
            .map(str::to_owned)
            .collect())
    }

    /// The `mtp prompt` records on the server's stderr so far.
    fn mtp_prompts(err_log: &Path) -> Result<Vec<String>, GateError> {
        records(err_log, "mtp prompt")
    }

    /// The drafted seat's break-even as its `draft keep` line prints it: the
    /// ids a reply token and the most reply tokens it weighs.
    #[derive(Clone, Copy)]
    struct Keep38 {
        per_token: f64,
        nominal: usize,
    }

    impl Keep38 {
        fn read(err_log: &Path) -> Result<Keep38, GateError> {
            let line = record_line(err_log, "draft keep")?
                .ok_or("the drafted server printed no `draft keep` line")?;
            let field = |k: &str| {
                line.split_whitespace()
                    .find_map(|w| w.strip_prefix(k)?.strip_prefix('='))
            };
            match (
                field("per_reply_token").and_then(|v| v.parse::<f64>().ok()),
                field("reply").and_then(|v| v.parse::<usize>().ok()),
            ) {
                (Some(per_token), Some(nominal)) => Ok(Keep38 { per_token, nominal }),
                _ => Err(
                    format!("a `draft keep` line without per_reply_token and reply: {line}").into(),
                ),
            }
        }

        /// The reply a greedy request of `n` tokens is weighed for: the
        /// tokens after its first, at most the nominal.
        fn reply(self, n: usize) -> usize {
            n.saturating_sub(1).min(self.nominal)
        }

        /// The break-even of a greedy request of `n` tokens, the seat's
        /// arithmetic.
        fn at(self, n: usize) -> usize {
            let r = f64::from(u32::try_from(self.reply(n)).unwrap_or(u32::MAX));
            (r * self.per_token).ceil() as usize
        }

        /// The `mtp keep` record a request of `n` tokens that weighed a
        /// prefix of `prefix` positions prints, its branch `kept` or `reset`.
        fn record(self, n: usize, prefix: u64, kept: bool) -> String {
            format!(
                "mtp keep branch={} prefix={prefix} break_even={} reply={}",
                if kept { "kept" } else { "reset" },
                self.at(n),
                self.reply(n)
            )
        }
    }

    /// The seat's lines on the server's stderr so far that say the MTP draft
    /// proposes nothing after a cut or a state put back.
    fn draft_offs(err_log: &Path) -> Result<Vec<String>, GateError> {
        Ok(std::fs::read_to_string(err_log)?
            .lines()
            .filter(|l| l.contains(DRAFT_OFF))
            .map(str::to_owned)
            .collect())
    }

    /// An `mtp prompt` record's join: its start, the rows the draft caught
    /// up and why it skipped (`none` when it drafts; the text runs to the
    /// line's end).
    fn join_of(line: &str) -> Option<(u64, u64, &str)> {
        let key = |name: &str| {
            line.split_whitespace()
                .find_map(|w| w.strip_prefix(name)?.strip_prefix('='))
                .and_then(|v| v.parse().ok())
        };
        let (_, skipped) = line.split_once(" skipped=")?;
        Some((key("start")?, key("caught_up")?, skipped))
    }

    /// One greedy `/completion` of the token array `ids`: its status, ids,
    /// `cache_n` and `draft_n`, and the `mtp prompt` records and draft-off
    /// lines it made the server print. A request the server does not answer is a status of 0
    /// and its error as the ids' line.
    fn greedy(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        ids: &[u32],
        n: usize,
        cache: bool,
    ) -> Result<Greedy, GateError> {
        let before = mtp_prompts(err_log)?.len();
        let offs_before = draft_offs(err_log)?.len();
        let keeps_before = records(err_log, "mtp keep")?.len();
        let body = json!({
            "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
            "cache_prompt": cache,
        });
        let g = match curl(&url("/completion"), Some(&body), false) {
            Ok((200, text)) => {
                let v = json_of("/completion", 200, &text)?;
                Greedy {
                    ok: true,
                    tokens: ids_of(&v["tokens"]),
                    cache_n: v["timings"]["cache_n"].as_u64().unwrap_or(u64::MAX),
                    draft_n: v["timings"]["draft_n"].as_u64().unwrap_or(0),
                    joins: Vec::new(),
                    offs: Vec::new(),
                    keeps: Vec::new(),
                    said: String::new(),
                }
            }
            Ok((st, text)) => Greedy::failed(format!("HTTP {st}: {text}")),
            Err(e) => Greedy::failed(e.to_string()),
        };
        let joins = mtp_prompts(err_log)?.split_off(before);
        let offs = draft_offs(err_log)?.split_off(offs_before);
        let keeps = records(err_log, "mtp keep")?.split_off(keeps_before);
        Ok(Greedy {
            joins,
            offs,
            keeps,
            ..g
        })
    }

    struct Greedy {
        ok: bool,
        tokens: Vec<u32>,
        cache_n: u64,
        draft_n: u64,
        joins: Vec<String>,
        offs: Vec<String>,
        /// The `mtp keep` records the request made the server print.
        keeps: Vec<String>,
        said: String,
    }

    impl Greedy {
        fn failed(said: String) -> Greedy {
            Greedy {
                ok: false,
                tokens: Vec::new(),
                cache_n: u64::MAX,
                draft_n: 0,
                joins: Vec::new(),
                offs: Vec::new(),
                keeps: Vec::new(),
                said,
            }
        }

        /// The request's line.
        fn show(&self, what: &str) {
            if self.ok {
                println!(
                    "{what}: tokens {:?} cache_n={} draft_n={} joins {:?} offs {:?} keeps {:?}",
                    self.tokens, self.cache_n, self.draft_n, self.joins, self.offs, self.keeps
                );
            } else {
                println!("{what}: {} joins {:?}", self.said, self.joins);
            }
        }

        /// Under the draft, a request that kept `cache_n` by a cut: its last
        /// draft-off line names that position, its join there names why the
        /// draft skips, and it drafted nothing.
        fn skipped_by_name(&self) -> bool {
            let at = format!(" from position {} ", self.cache_n);
            self.ok
                && self.draft_n == 0
                && self.offs.last().is_some_and(|l| l.contains(&at))
                && self.joins.iter().any(|l| {
                    join_of(l).is_some_and(|(s, c, k)| s == self.cache_n && c == 0 && k != "none")
                })
        }

        /// Under the draft, a request that reset below the break-even: it
        /// kept nothing, printed `record` (its `mtp keep`) and no draft-off
        /// line, and drafted.
        fn reset_by_rule(&self, record: &str) -> bool {
            self.ok
                && self.cache_n == 0
                && self.draft_n > 0
                && self.offs.is_empty()
                && self.keeps.last().is_some_and(|l| l == record)
        }

        /// One join printed for this request, at `start`, that walked the
        /// draft up to it and skipped nothing.
        fn joined_at(&self, start: u64) -> bool {
            matches!(self.joins.as_slice(), [l] if join_of(l)
                .is_some_and(|(s, c, k)| s == start && c >= 1 && k == "none"))
        }
    }

    /// The drafted server's continuations (`BLOOMERY_DRAFT=mtp`): requests
    /// that extend the sequence the server holds keep its prefix
    /// (`cache_prompt`), and the draft joins them — the rows an earlier
    /// request left waiting walked, the token the new request puts at their
    /// last position — so it drafts from the first window, with the plain
    /// run's ids. Three joins: after a request fed fresh that ended on its
    /// first step (`n_predict` 1), extended by the id it generated: its prompt call is
    /// empty, the join is its first step's, and no `mtp prompt` record says
    /// it skipped; after drafted windows (the held sequence is at most 11
    /// ids past the prompt: 8 generated, the last window's 3 past them),
    /// extended by the plain run's next ids, a prompt call of 1 to 4 ids and
    /// its record; and after a first step again, extended by ids the model
    /// did not generate, against the same prompt fed fresh. That first step
    /// is the prompt resent, which keeps the prompt's end (one short of it)
    /// by a cut: the draft proposes nothing from there, by name, and the
    /// extension, with the draft off, keeps the prompt only at or past the
    /// break-even of its reply (`mtp keep` `kept`, plain steps), else resets
    /// (`reset`, drafted) — with the fresh run's ids either way. Every prompt
    /// call stays below
    /// [`GEMM_FROM`](bloomery_gpu::arch::qwen3moe::Prompt38::GEMM_FROM), so
    /// a continued and a fresh feed leave the same state.
    fn continued(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        ids: &[u32],
        reference: &[u32],
    ) -> Result<bool, GateError> {
        let p = ids.len() as u64;
        let with = |tail: &[u32]| -> Vec<u32> { ids.iter().chain(tail).copied().collect() };
        let mut ok = true;

        // PIN(2026-10-01): fed fresh, so the draft starts on: under the keep rule's checkpoints
        // the prompt kept from an earlier request's cached state is a cut, after which the draft
        // proposes nothing; was `cache_prompt` true.
        let a = greedy(url, err_log, ids, 1, false)?;
        a.show("continued: the prompt, one id");
        check(
            &mut ok,
            "continued_first_request_is_the_plain_first_id",
            a.ok && a.tokens == reference[..1],
        );

        let x1 = greedy(url, err_log, &with(&reference[..1]), 8, true)?;
        x1.show("continued: extended by its id");
        check(
            &mut ok,
            "continued_by_its_own_id_keeps_the_prompt_and_skips_nothing",
            x1.ok && x1.cache_n == p && x1.joins.is_empty(),
        );
        check(
            &mut ok,
            "continued_by_its_own_id_drafts_the_plain_ids",
            x1.ok && x1.draft_n > 0 && x1.tokens == reference[1..9],
        );

        let x2 = greedy(url, err_log, &with(&reference[..13]), 3, true)?;
        x2.show("continued: after windows, extended by the plain ids");
        check(
            &mut ok,
            "continued_after_windows_keeps_them_and_joins_at_their_end",
            x2.ok && x2.cache_n > p + 1 && x2.joined_at(x2.cache_n),
        );
        check(
            &mut ok,
            "continued_after_windows_drafts_the_plain_ids",
            x2.ok && x2.draft_n > 0 && x2.tokens == reference[13..16],
        );

        let ext: Vec<u32> = ids.iter().copied().skip(1).take(3).collect();
        if ext.len() != 3 || ext[0] == reference[0] {
            return Err(format!(
                "--ids {ids:?}: the extension {ext:?} must be three ids, its first not the \
                 generated id {}",
                reference[0]
            )
            .into());
        }
        let b = greedy(url, err_log, ids, 1, true)?;
        b.show("continued: the prompt again, one id");
        let x3 = greedy(url, err_log, &with(&ext), 8, true)?;
        x3.show("continued: extended by other ids");
        let fresh = greedy(url, err_log, &with(&ext), 8, false)?;
        fresh.show("continued: the same ids fed fresh");
        // PIN(2026-10-01): the keep rule grants checkpoints under the draft (lead, round recsave),
        // and the draft's rejoin after a cut is not built (app/src/mtp.rs): `b` keeps p − 1 by a
        // cut, so `x3` drafts nothing and its join names why; was a join at p that drafts.
        // PIN(2026-10-02): the seat's break-even weighs a keep that leaves the draft off (round
        // q38rules): `b`'s reply of one token makes no pass, so it keeps p − 1 by a cut; `x3`,
        // with the draft off, keeps the whole prompt only at or past the break-even of its 8
        // tokens, else it resets and drafts; was the prompt kept and nothing drafted, always.
        let k = Keep38::read(err_log)?;
        let x3_kept = p as usize >= k.at(8);
        let skips_at_p = x3
            .joins
            .iter()
            .any(|l| join_of(l).is_some_and(|(s, c, k)| s == p && c == 0 && k != "none"));
        let x3_ok = if x3_kept {
            x3.ok
                && x3.cache_n == p
                && x3.draft_n == 0
                && x3.offs.is_empty()
                && skips_at_p
                && x3.keeps.last() == Some(&k.record(8, p, true))
        } else {
            // The reset may follow a state the prompt cache put back first,
            // whose prefix the rule then declined: any prefix up to p.
            (1..=p).any(|q| x3.reset_by_rule(&k.record(8, q, false)))
        };
        check(
            &mut ok,
            "continued_by_other_ids_keeps_the_prompt_and_skips_by_name",
            b.cache_n == p - 1
                && b.skipped_by_name()
                && b.keeps.last() == Some(&k.record(1, p - 1, true))
                && x3_ok,
        );
        check(
            &mut ok,
            "continued_by_other_ids_runs_the_fresh_ids",
            x3.ok
                && fresh.ok
                && fresh.cache_n == 0
                && fresh.draft_n > 0
                && x3.tokens == fresh.tokens
                && x3.tokens.len() == 8,
        );
        Ok(ok)
    }

    /// The drafted server's sampled requests, served through plain steps with
    /// the target's logits row read after each step: `sampled` (answered `st`
    /// `body`) is served and counts no draft; `top1` (answered `st1` `body1`),
    /// the same request cut to `top_k` 1, samples the row's own argmax, so its
    /// ids are this server's greedy `first`. A plain server is no reference
    /// for the sampled ids: the drafted load plans fewer target experts on the
    /// card (the draft takes card bytes), and an expert served on the host
    /// rounds its sums another way, so a sampled id can differ at a near tie.
    /// Mutants: the refusal of a sampled request under a draft restored (a
    /// 400); a row read after the draft's walk from a buffer the walk writes
    /// (`top1` then follows the draft's argmax, not `first`).
    fn sampled_served(
        sampled: (u16, &str),
        top1: (u16, &str),
        first: &[u32],
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let (st, body) = sampled;
        let drafted = serde_json::from_str::<Value>(body).unwrap_or(Value::Null);
        let got = ids_of(&drafted["tokens"]);
        println!("sampled on the drafted server: HTTP {st} tokens {got:?}");
        check(
            &mut ok,
            "drafted_serves_a_sampled_request",
            st == 200 && !got.is_empty(),
        );
        check(
            &mut ok,
            "drafted_sampled_request_drafts_nothing",
            drafted["timings"].get("draft_n").is_none(),
        );
        let (st1, body1) = top1;
        let k1 = json_of("/completion", st1, body1)?;
        let got1 = ids_of(&k1["tokens"]);
        println!("top_k 1 on the drafted server: tokens {got1:?}, greedy {first:?}");
        check(
            &mut ok,
            "drafted_top1_sample_is_the_greedy_ids",
            got1 == first && k1["timings"].get("draft_n").is_none(),
        );
        Ok(ok)
    }

    /// The card free bytes the server's own `plan` record
    /// ([`record::PLAN38`]) named: its census reading at its load. The
    /// gate's re-derivations take the same one — a fresh census read beside
    /// the loaded server would see the server's own bytes as taken and size
    /// another plan. A server that printed no `plan` record, or a record
    /// without its `card_free`, is a named error, never an uncapped re-plan:
    /// the seat prints its `ctx` line before its `plan` record, so a reader
    /// that stops the server at the `ctx` line waits for the record too.
    fn server_card_free(err_log: &Path) -> Result<u64, GateError> {
        let kind = &record::PLAN38;
        let line = record_line(err_log, kind.head)?.ok_or_else(|| {
            format!(
                "{}: the server printed no `{}` record",
                err_log.display(),
                kind.name
            )
        })?;
        line.split(" card_free=")
            .nth(1)
            .and_then(|t| t.split(' ').next())
            .and_then(|n| n.parse::<u64>().ok())
            .ok_or_else(|| {
                format!(
                    "the server's `{}` record carries no card_free: {line}",
                    kind.name
                )
                .into()
            })
    }

    /// The first record on the server's stderr at `err_log` whose line
    /// starts with `head` and a space.
    fn record_line(err_log: &Path, head: &str) -> Result<Option<String>, GateError> {
        let prefix = format!("{head} ");
        Ok(std::fs::read_to_string(err_log)?
            .lines()
            .find(|l| l.starts_with(&prefix))
            .map(str::to_owned))
    }

    /// The `residency` clause (the module header): a server under
    /// [`RESIDENCY_WORD`], started alone once the others have stopped,
    /// serves `completion`, its reset is a 200 with its record, and
    /// `completion` after the reset gives the first one's ids.
    fn residency(dir: &Path, completion: &Value) -> Result<bool, GateError> {
        let word = RESIDENCY_WORD;
        println!("residency: {word}");
        let res_dir = dir.join("residency");
        std::fs::create_dir_all(&res_dir)?;
        let err_log = res_dir.join("server.err");
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::RESIDENCY, word)
            .env(bloomery_levers::DRAFT, "off")
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT);
        let mut served = Served38::spawn_with(&SERVER_ARGS, &res_dir, &mut cmd)?;
        println!("residency server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = true;
        let lever = record_line(&err_log, "residency lever")?;
        let host = record_line(&err_log, "residency host")?;
        println!("residency records: {lever:?} {host:?}");
        let named = format!("residency={word}");
        check(
            &mut ok,
            "residency_loads_the_word",
            lever.is_some_and(|l| l.contains(&named) && l.ends_with(" why=set"))
                && host.is_some_and(|l| l.contains(&named)),
        );
        let (st, body) = curl(&url("/completion"), Some(completion), false)?;
        let ids = ids_of(&json_of("/completion", st, &body)?["tokens"]);
        println!("residency completion tokens {ids:?}");
        let passes = |kind: &str| -> Result<bool, GateError> {
            let want = format!("residency pass pass={kind} ");
            Ok(std::fs::read_to_string(&err_log)?
                .lines()
                .any(|l| l.starts_with(&want)))
        };
        check(
            &mut ok,
            "residency_request_ends_prompt_and_step_passes",
            passes("prompt")? && passes("step")?,
        );
        let landed: u64 = std::fs::read_to_string(&err_log)?
            .lines()
            .filter(|l| l.starts_with("residency pass "))
            .filter_map(|l| l.split(' ').find_map(|w| w.strip_prefix("landed=")))
            .filter_map(|n| n.parse::<u64>().ok())
            .sum();
        println!("residency flips landed over the request: {landed}");
        let (st, body) = curl(&url("/residency/reset"), Some(&json!({})), false)?;
        let reset: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let record = record_line(&err_log, "residency reset")?;
        println!("residency reset: HTTP {st} {reset}; record {record:?}");
        let counts = ["cancelled", "copies", "diff"].map(|k| format!("{k}={}", reset[k]));
        check(
            &mut ok,
            "residency_reset_is_a_200_with_its_record",
            st == 200
                && reset["diff"] == json!(0)
                && record.is_some_and(|l| counts.iter().all(|c| l.contains(&format!(" {c} ")))),
        );
        let (st, body) = curl(&url("/completion"), Some(completion), false)?;
        let again = ids_of(&json_of("/completion", st, &body)?["tokens"]);
        println!("residency completion after the reset tokens {again:?}");
        check(
            &mut ok,
            "residency_ids_after_the_reset_are_the_first_requests",
            !ids.is_empty() && again == ids,
        );
        println!("residency server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The ids of `messages` as the server's chat template renders them
    /// (`/apply-template`, then `/tokenize`).
    fn rendered(url: &dyn Fn(&str) -> String, messages: Value) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(
            &url("/apply-template"),
            Some(&json!({ "messages": messages })),
            false,
        )?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        tokenized(url, &text)
    }

    /// `/tokenize` of `text`.
    fn tokenized(url: &dyn Fn(&str) -> String, text: &str) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(&url("/tokenize"), Some(&json!({ "content": text })), false)?;
        Ok(ids_of(&json_of("/tokenize", st, &body)?["tokens"]))
    }

    /// The slot dropped and not saved (`POST /slots/0?action=erase`): the
    /// prompt cache keeps what it held, and nothing more.
    fn erase(url: &dyn Fn(&str) -> String) -> Result<(), GateError> {
        let (st, body) = curl(&url("/slots/0?action=erase"), Some(&json!({})), false)?;
        json_of("/slots/0?action=erase", st, &body)?;
        Ok(())
    }

    /// The `cache` clause on one server (`drafted`: its MTP draft runs):
    /// session A's first turn fed fresh, session B's turn, then A again — its
    /// state saved when B took the slot and put back — and the greedy ids
    /// against a run of A with no switch between:
    /// - (a) A's conversation resent verbatim with a later turn: it keeps
    ///   every position A held (under the draft, whose last window fed rows
    ///   past the reply it returned, the resend shares fewer than held and
    ///   keeps the turn's prompt end, as the same session does with no
    ///   switch), and its ids are those of the same two requests run back to
    ///   back (fresh, then the resend), run first. Not a fresh run
    ///   of the resend: the positions A's generation fed were steps, which a
    ///   fresh prompt call would run through the ubatch walk;
    /// - (b) A's first turn resent with the reasoning-stripped reply in place
    ///   of the reply, so the shared prefix ends at the first turn's end: it
    ///   keeps the checkpoint there (the turn's prompt call's end, one short
    ///   of the turn: its last id is fed by the first step), the draft on
    ///   or off, and its ids are a fresh run's of the same ids (every prompt
    ///   call of at least nine ids is the ubatch walk, whose bits do not
    ///   depend on where a call is cut);
    /// - under the draft, the cuts (the resends with and without the switch,
    ///   the stripped resend) weigh the turn's end against the seat's
    ///   break-even for the reply ([`Keep38`], the `draft keep` line): at or
    ///   past it each keeps the turn's end, drafts nothing and says so by
    ///   name (the seat's draft-off line at the kept position, an `mtp
    ///   prompt` record there with why, a `kept` `mtp keep` record); below it
    ///   each keeps nothing, prints a `reset` record and no draft-off line,
    ///   and drafts. The fresh run after them drafts. Without the draft no
    ///   such line prints. The draft's rejoin after a cut is not built: when
    ///   it is, this clause turns over;
    /// - (c) under the draft, both sides of the break-even at the turn's end
    ///   ([`break_even`]).
    ///
    /// Mutants: the state keeping the current position's recurrent stores
    /// alone ((b) keeps 0); a resume that leaves the PLE history behind (the
    /// resend is refused); a keep rule that grants any position (B's request
    /// cuts where no checkpoint stands, refused); under the draft every
    /// shorter prefix kept as none (the cuts' clause sees no cut).
    fn cache(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        drafted: bool,
        label: &str,
    ) -> Result<bool, GateError> {
        let p1 = rendered(url, json!([{ "role": "user", "content": TURN_A }]))?;
        let pb = rendered(url, json!([{ "role": "user", "content": TURN_B }]))?;
        let later = tokenized(url, LATER_A)?;
        let stripped = tokenized(url, STRIPPED_A)?;
        let gemm = bloomery_gpu::arch::qwen3moe::Prompt38::GEMM_FROM;
        if p1.len() < gemm + 1 || later.len() < gemm || stripped.len() < gemm {
            return Err(format!(
                "the cache clause's turns are {}, {} and {} ids; each needs at least {gemm} past \
                 the last kept position",
                p1.len(),
                later.len(),
                stripped.len()
            )
            .into());
        }
        let n = N_PREDICT;
        let with = |a: &[u32], b: &[u32], c: &[u32]| -> Vec<u32> {
            a.iter().chain(b).chain(c).copied().collect()
        };
        let mut ok = true;

        // The run with no switch first, then the slot dropped unsaved, so no
        // cached state holds the resend when the switched run asks for it.
        let r1 = greedy(url, err_log, &p1, n, false)?;
        r1.show(&format!("cache {label}: A's turn fresh"));
        let resend = with(&p1, &r1.tokens, &later);
        let r2 = greedy(url, err_log, &resend, n, true)?;
        r2.show(&format!("cache {label}: A resent verbatim, no switch"));
        erase(url)?;
        let a1 = greedy(url, err_log, &p1, n, false)?;
        a1.show(&format!("cache {label}: A's turn fresh again"));
        let b1 = greedy(url, err_log, &pb, n, true)?;
        b1.show(&format!("cache {label}: B's turn"));
        let a2 = greedy(url, err_log, &resend, n, true)?;
        a2.show(&format!("cache {label}: A resent verbatim after B"));
        let held = (p1.len() + a1.tokens.len()).saturating_sub(1) as u64;
        let turn_end = p1.len() as u64 - 1;
        // PIN(2026-10-02): under the draft a cut keeps the turn's end only at or past the seat's
        // break-even for the reply (round q38rules: `Q38::draft_keep`), else the request resets
        // and prefills whole with the draft on; was the turn's end, the draft off, always.
        let keep38 = drafted.then(|| Keep38::read(err_log)).transpose()?;
        let cut_kept = keep38.is_none_or(|k| turn_end >= k.at(n) as u64);
        let cut_at = if cut_kept { turn_end } else { 0 };
        let want = if drafted { cut_at } else { held };
        if let Some(k) = keep38 {
            println!(
                "cache {label}: the turn's end {turn_end} against the break-even {} of a reply of \
                 {} tokens: {}",
                k.at(n),
                k.reply(n),
                if cut_kept { "kept" } else { "reset" }
            );
        }
        let kept = a2.cache_n == want && r2.cache_n == want;
        check(
            &mut ok,
            &format!("cache_{label}_a_keeps_every_held_position_after_the_switch"),
            r1.ok && a1.ok && b1.ok && a2.ok && !a1.tokens.is_empty() && kept,
        );
        check(
            &mut ok,
            &format!("cache_{label}_a_ids_are_the_run_with_no_switch"),
            r2.ok && r1.tokens == a1.tokens && !a2.tokens.is_empty() && a2.tokens == r2.tokens,
        );

        if stripped.first() == a1.tokens.first() {
            return Err(format!(
                "the stripped reply's first id {:?} is the reply's: the shared prefix would not \
                 end at the turn",
                stripped.first()
            )
            .into());
        }
        erase(url)?;
        let c1 = greedy(url, err_log, &p1, n, false)?;
        c1.show(&format!("cache {label}: A's turn fresh (b)"));
        let b2 = greedy(url, err_log, &pb, n, true)?;
        b2.show(&format!("cache {label}: B's turn (b)"));
        let strip = with(&p1, &stripped, &[]);
        let a3 = greedy(url, err_log, &strip, n, true)?;
        a3.show(&format!("cache {label}: A resent stripped after B"));
        let f3 = greedy(url, err_log, &strip, n, false)?;
        f3.show(&format!("cache {label}: the same ids fresh"));
        check(
            &mut ok,
            &format!("cache_{label}_b_keeps_the_turns_end"),
            c1.ok && b2.ok && a3.ok && a3.cache_n == if drafted { cut_at } else { turn_end },
        );
        check(
            &mut ok,
            &format!("cache_{label}_b_ids_are_a_fresh_runs"),
            f3.ok && f3.cache_n == 0 && !a3.tokens.is_empty() && a3.tokens == f3.tokens,
        );
        let cuts = [&r2, &a2, &a3];
        let named = match keep38 {
            Some(k) if cut_kept => {
                let rec = k.record(n, turn_end, true);
                cuts.iter()
                    .all(|g| g.skipped_by_name() && g.keeps.last() == Some(&rec))
                    && f3.draft_n > 0
                    && f3.offs.is_empty()
            }
            Some(k) => {
                let rec = k.record(n, turn_end, false);
                cuts.iter().all(|g| g.reset_by_rule(&rec)) && f3.draft_n > 0 && f3.offs.is_empty()
            }
            None => cuts.iter().chain([&f3].iter()).all(|g| g.offs.is_empty()),
        };
        check(
            &mut ok,
            &format!("cache_{label}_cuts_leave_the_draft_off_by_name"),
            named,
        );
        if let Some(k) = keep38 {
            ok &= break_even(url, err_log, k, &pb)?;
        }
        Ok(ok)
    }

    /// The drafted seat's break-even at a turn's end, as the `cache`
    /// clause's (b) reaches it (the turn, B's turn, the turn resent with its
    /// reply stripped: its state put back, the turn's end kept by a cut): a
    /// reply short enough that the turn's end is at or past its break-even
    /// keeps it — the draft off from there, by name, a `kept` record — and
    /// one whose break-even is past it resets — nothing kept, the draft on, a
    /// `reset` record, no draft-off line; each gives the ids of the same
    /// request fed fresh. Each side has a session of its own
    /// ([`BREAK_EVEN_TURNS`]), so the prompt cache holds no state that shares
    /// more than its turn. The replies are the seat's own arithmetic on its
    /// `draft keep` line: the longest of at most [`N_PREDICT`] tokens whose
    /// break-even the turn's end reaches, and [`N_PREDICT`], whose
    /// break-even it must not. Mutants: a break-even of 0 (the short reply
    /// keeps the turn's end) and of `usize::MAX` (the long reply resets).
    fn break_even(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        k: Keep38,
        pb: &[u32],
    ) -> Result<bool, GateError> {
        let mut ok = true;
        for (side, (turn, reply_text)) in ["long", "short"].into_iter().zip(BREAK_EVEN_TURNS) {
            let p1 = rendered(url, json!([{ "role": "user", "content": turn }]))?;
            let stripped = tokenized(url, reply_text)?;
            let turn_end = p1.len() - 1;
            let long = (2..=N_PREDICT)
                .rev()
                .find(|&n| k.at(n) <= turn_end)
                .ok_or_else(|| {
                    format!(
                        "the turn's end {turn_end} is below the break-even of every reply of 2 to \
                     {N_PREDICT} tokens ({} ids a token): the clause needs a longer turn",
                        k.per_token
                    )
                })?;
            if k.at(N_PREDICT) <= turn_end {
                return Err(format!(
                    "the turn's end {turn_end} reaches the break-even {} of a reply of {N_PREDICT} \
                 tokens: the clause needs a shorter turn",
                    k.at(N_PREDICT)
                )
                .into());
            }
            let (n, kept) = if side == "long" {
                (long, true)
            } else {
                (N_PREDICT, false)
            };
            let strip: Vec<u32> = p1.iter().chain(&stripped).copied().collect();
            {
                erase(url)?;
                let a = greedy(url, err_log, &p1, n, false)?;
                a.show(&format!("break-even {side}: A's turn fresh, {n} tokens"));
                if stripped.first() == a.tokens.first() {
                    return Err(format!(
                    "the {side} side's stripped reply's first id {:?} is the reply's: the shared \
                     prefix would not end at the turn",
                    stripped.first()
                )
                .into());
                }
                let b = greedy(url, err_log, pb, n, true)?;
                b.show(&format!("break-even {side}: B's turn"));
                let s = greedy(url, err_log, &strip, n, true)?;
                s.show(&format!("break-even {side}: A resent stripped after B"));
                let f = greedy(url, err_log, &strip, n, false)?;
                f.show(&format!("break-even {side}: the same ids fresh"));
                let rec = k.record(n, turn_end as u64, kept);
                let branch = if kept {
                    s.cache_n == turn_end as u64
                        && s.skipped_by_name()
                        && s.keeps.last() == Some(&rec)
                } else {
                    s.reset_by_rule(&rec)
                };
                let fresh = a.ok && b.ok && f.ok && f.cache_n == 0 && f.offs.is_empty();
                let same = !s.tokens.is_empty() && s.tokens == f.tokens;
                println!(
                    "break-even {side}: the turn's end {turn_end}, a reply of {n} tokens weighed as {} \
                 against the break-even {}: want {rec:?} {}",
                    k.reply(n),
                    k.at(n),
                    verdict(branch && fresh && same)
                );
                check(
                    &mut ok,
                    if kept {
                        "break_even_past_it_keeps_the_prefix_the_draft_off"
                    } else {
                        "break_even_below_it_resets_the_draft_on"
                    },
                    branch && fresh,
                );
                check(
                    &mut ok,
                    if kept {
                        "break_even_kept_ids_are_a_fresh_runs"
                    } else {
                        "break_even_reset_ids_are_a_fresh_runs"
                    },
                    same,
                );
            }
        }
        Ok(ok)
    }

    /// `/metrics`' `llamacpp:<name>` value; `None` when it carries none.
    fn metric(url: &dyn Fn(&str) -> String, name: &str) -> Result<Option<f64>, GateError> {
        let (st, body) = curl(&url("/metrics"), None, false)?;
        if st != 200 {
            return Err(format!("/metrics: HTTP {st}: {body}").into());
        }
        let key = format!("llamacpp:{name} ");
        Ok(body
            .lines()
            .find_map(|l| l.strip_prefix(&key))
            .and_then(|v| v.trim().parse().ok()))
    }

    /// The server's `parallel` line (the seat's own, printed before its
    /// load).
    fn parallel_line(err_log: &Path) -> Result<Option<String>, GateError> {
        Ok(std::fs::read_to_string(err_log)?
            .lines()
            .find(|l| l.starts_with("parallel rule="))
            .map(str::to_owned))
    }

    /// The `parallel` line's rule against its own terms (the module header):
    /// under `slots` the line's slots are the flag's (`--parallel`'s value,
    /// the caller naming it) or the default's (2), and the split the line
    /// names holds — `slot_ctx` the one context the whole round of clauses
    /// pins a slot at (the server's `--ctx-size` total over the slots) and
    /// `total` the slots' sum. `None` a line that is not the seat's;
    /// `Some(false)` a rule whose terms do not hold.
    fn parallel_agrees(line: &str, flag: Option<usize>) -> Option<bool> {
        let field = |k: &str| {
            line.split_whitespace()
                .find_map(|w| w.strip_prefix(k)?.strip_prefix('='))
        };
        let rule = field("rule")?;
        let slots: usize = field("slots")?.parse().ok()?;
        let slot_ctx: usize = field("slot_ctx")?.parse().ok()?;
        let total: usize = field("total")?.parse().ok()?;
        let want = match (rule, flag) {
            ("slots", f) => f.unwrap_or(2),
            _ => return None,
        };
        Some(slots == want && total == slots * slot_ctx && slot_ctx > 0)
    }

    /// One streamed request's answer: the final event's tokens, its draft
    /// counts, and each token event's arrival on its reader thread's clock.
    struct Streamed {
        tokens: Vec<u32>,
        /// The last event's `timings.draft_n`/`draft_n_accepted`.
        drafts: (u64, u64),
        arrivals: Vec<Instant>,
    }

    /// A streamed request's answer channel: what its reader thread sends
    /// when the stream ends.
    type StreamedRx = mpsc::Receiver<Result<Streamed, String>>;

    /// `/completion` of `ids` at temperature 0, streamed: a helper thread of
    /// its own runs `curl -N` and reads the SSE lines as they land, each
    /// token event timestamped the moment it is read (the module header's
    /// (v3) reader thread a stream). The final event carries the tokens and
    /// the request's timings, its draft counts among them.
    fn streamed(
        addr: &str,
        ids: &[u32],
        n: usize,
    ) -> Result<(JoinHandle<()>, StreamedRx), GateError> {
        let (tx, rx) = mpsc::channel();
        let body = json!({
            "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
            "cache_prompt": false, "stream": true,
        });
        let url = format!("http://{addr}/completion");
        let (h, _) = spawn_helper("slots-request", Placement::Float, move || {
            let run = (|| -> Result<Streamed, String> {
                use std::io::{BufRead, BufReader, Write};
                let mut child = Command::new("curl")
                    .args([
                        "-sS",
                        "-N",
                        "--max-time",
                        "600",
                        "-H",
                        "Content-Type: application/json",
                        "--data-binary",
                        "@-",
                    ])
                    .arg(&url)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
                    .map_err(|e| format!("curl {url}: {e}"))?;
                let mut stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| format!("curl {url}: no stdin"))?;
                stdin
                    .write_all(body.to_string().as_bytes())
                    .map_err(|e| format!("curl {url}: {e}"))?;
                drop(stdin);
                let out = child
                    .stdout
                    .take()
                    .ok_or_else(|| format!("curl {url}: no stdout"))?;
                let mut arrivals = Vec::new();
                let mut tokens = Vec::new();
                let mut drafts = (0, 0);
                for line in BufReader::new(out).lines() {
                    let line = line.map_err(|e| format!("curl {url}: {e}"))?;
                    let at = Instant::now();
                    let Some(v) = line
                        .strip_prefix("data: ")
                        .and_then(|d| serde_json::from_str::<Value>(d).ok())
                    else {
                        continue;
                    };
                    if v["stop"] == json!(false) && !v["content"].as_str().unwrap_or("").is_empty()
                    {
                        arrivals.push(at);
                    }
                    if let (Some(n), Some(a)) = (
                        v["timings"]["draft_n"].as_u64(),
                        v["timings"]["draft_n_accepted"].as_u64(),
                    ) {
                        drafts = (n, a);
                    }
                    if v.get("tokens").is_some_and(Value::is_array) {
                        tokens = ids_of(&v["tokens"]);
                    }
                }
                let status = child.wait().map_err(|e| format!("curl {url}: {e}"))?;
                if !status.success() {
                    return Err(format!("curl {url}: {status}"));
                }
                Ok(Streamed {
                    tokens,
                    drafts,
                    arrivals,
                })
            })();
            let _ = tx.send(run);
        })
        .map_err(|e| format!("slots: {}", e.what()))?;
        Ok((h, rx))
    }

    /// One streamed request run to its end on its own thread: its answer, or
    /// `None` a request that failed (a red check, not the clause's end).
    fn run_streamed(
        addr: &str,
        ids: &[u32],
        n: usize,
        what: &str,
    ) -> Result<Option<Streamed>, GateError> {
        let (h, rx) = streamed(addr, ids, n)?;
        h.join()
            .map_err(|_| format!("slots: the {what} request's thread panicked"))?;
        let ran = rx
            .recv()
            .map_err(|_| format!("slots: the {what} request's thread gave no answer"))?;
        match ran {
            Ok(s) => Ok(Some(s)),
            Err(e) => {
                println!("slots: the {what} request failed: {e}");
                Ok(None)
            }
        }
    }

    /// The busy slots the server has booked over its `decodes` engine calls:
    /// `/metrics` carries them as the running mean `n_busy_slots_per_decode`
    /// (`serve::api`'s metrics), so the total is that mean times the calls.
    fn busy_total(url: &dyn Fn(&str) -> String, decodes: f64) -> Result<f64, GateError> {
        Ok(metric(url, "n_busy_slots_per_decode")?.unwrap_or(f64::NAN) * decodes)
    }

    /// The (v3) interleave check (the module header's derivation): while both
    /// streams are live — from the later one's first arrival to the earlier
    /// one's last — every window of [`SLOTS_WINDOW`] consecutive arrivals
    /// holds at least [`SLOTS_EACH`] of each stream. `None` a pair a stream
    /// of which never ran; the check is red on an empty stretch.
    fn interleave(
        ok: &mut bool,
        name: &str,
        together: &[Option<Streamed>],
    ) -> Result<(), GateError> {
        let (Some(a), Some(b)) = (together[0].as_ref(), together[1].as_ref()) else {
            check(ok, name, false);
            return Ok(());
        };
        let mut merged: Vec<(usize, Instant)> = [a, b]
            .iter()
            .enumerate()
            .flat_map(|(i, s)| s.arrivals.iter().map(move |&t| (i, t)))
            .collect();
        merged.sort_by_key(|&(_, t)| t);
        let live_from = [a, b]
            .iter()
            .map(|s| s.arrivals.first().copied())
            .max()
            .flatten();
        let live_to = [a, b]
            .iter()
            .map(|s| s.arrivals.last().copied())
            .min()
            .flatten();
        let tags: Vec<usize> = merged
            .iter()
            .filter(|(_, t)| live_from.is_some_and(|f| *t >= f) && live_to.is_some_and(|u| *t < u))
            .map(|&(i, _)| i)
            .collect();
        let mut thin = 0;
        for w in tags.windows(SLOTS_WINDOW) {
            let of_first = w.iter().filter(|&&i| i == 0).count();
            if of_first < SLOTS_EACH || SLOTS_WINDOW - of_first < SLOTS_EACH {
                thin += 1;
            }
        }
        println!(
            "slots interleave: {} arrivals while both live, {} thin window(s) of {}",
            tags.len(),
            thin,
            SLOTS_WINDOW,
        );
        check(ok, name, !tags.is_empty() && thin == 0);
        Ok(())
    }

    /// The slots clause (the module header) in two parts. The first server,
    /// into `<dir>/slots`, is this gate's own shape at `--parallel 2` with
    /// the draft on and the residency off: the ids and draft counts of two
    /// streamed requests run together are their alone runs'
    /// ((v1) — the per-slot draft park is the rejoin the swap clause held),
    /// a round carries both slots and nothing swaps ((v2)), the streams
    /// interleave ((v3)), and the load, `listening` and `/props` name the
    /// split ((v4)). The second server, into `<dir>/slots-default`, is the
    /// seat under its own defaults at `--parallel 2` — `--place a`, neither
    /// lever set (the user's case: the A6000 stages, the adaptive residency
    /// and the draft run) — and runs the together pair once, holding (v2)
    /// and (v3) alone.
    // PIN(2026-10-05): the swap clause (`swap_rejoins_the_draft`) is removed — the
    /// turn path it checked (`serve::SwapEngine`, the park of a preempted request's
    /// state) left the seat with the resident slots, and its coverage (a preempted
    /// request's draft rejoining) is this clause's (v1) draft-count equality.
    fn slots_flow_together(dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let own = dir.join("slots");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        // SERVER_ARGS ends in the one-slot `--parallel 1`; this clause's
        // server takes the two-slot value, the total context the same 4096.
        let mut args: Vec<&str> = SERVER_ARGS.to_vec();
        let parallel = args.len() - 1;
        args[parallel] = "2";
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, "mtp")
            .env(bloomery_levers::RESIDENCY, "off");
        let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
        println!("slots server pid {}", served.child.id());
        let addr = match served.address(&err_log, POLLS, POLL) {
            Ok(a) => a,
            // A server that never listens names no split: the checks are red
            // by name, and the clause's second server still runs.
            Err(e) => {
                println!(
                    "slots: the server never listened: {e}; it said: {}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                for name in [
                    "slots_parallel_line_names_the_rule",
                    "slots_load_names_the_split",
                    "slots_alone_ids_stay",
                    "slots_alone_draft_counts_stay",
                    "slots_rounds_carry_both_slots",
                    "slots_streams_interleave",
                ] {
                    check(&mut ok, name, false);
                }
                let _ = served.stop();
                return slots_default_residency(dir, &mut ok);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let line = parallel_line(&err_log)?.ok_or("the slots server printed no `parallel` line")?;
        println!("slots {line}");
        check(
            &mut ok,
            "slots_parallel_line_names_the_rule",
            line.contains("rule=slots") && parallel_agrees(&line, Some(2)) == Some(true),
        );
        // (v4) The load names the split, the `listening` record the same,
        // and /props the slot ctx.
        // The `load arch=` line, not the host tier's own `load host_tier …`
        // note that can precede it.
        let load = std::fs::read_to_string(&err_log)?
            .lines()
            .find(|l| l.starts_with("load arch="))
            .unwrap_or("")
            .to_owned();
        let field = |l: &str, k: &str| {
            l.split_whitespace()
                .find_map(|w| w.strip_prefix(&format!("{k}=")))
                .and_then(|v| v.parse::<u64>().ok())
        };
        let listening = std::fs::read_to_string(&err_log)?
            .lines()
            .find(|l| l.starts_with(record::LISTENING38.head))
            .unwrap_or("")
            .to_owned();
        let (st, body) = curl(&url("/props"), None, false)?;
        let props = json_of("/props", st, &body)?;
        let n_ctx = props["n_ctx"].as_u64().unwrap_or(u64::MAX);
        println!(
            "slots: {load}; {listening}; props n_ctx {n_ctx}, load slots={:?} slot_ctx={:?}",
            field(&load, "slots"),
            field(&load, "slot_ctx"),
        );
        check(
            &mut ok,
            "slots_load_names_the_split",
            field(&load, "slots") == Some(2)
                && field(&load, "slot_ctx") == Some(u64::try_from(CTX / 2).unwrap())
                && field(&listening, "slots") == Some(2)
                && field(&listening, "slot_ctx") == Some(u64::try_from(CTX / 2).unwrap())
                && n_ctx == u64::try_from(CTX / 2).unwrap(),
        );
        // Two distinct prompts, each first run alone, then both together.
        let (a_ids, b_ids) = (
            rendered(&url, json!([{ "role": "user", "content": TURN_A }]))?,
            rendered(&url, json!([{ "role": "user", "content": SLOTS_B }]))?,
        );
        let gemm = bloomery_gpu::arch::qwen3moe::Prompt38::GEMM_FROM;
        if a_ids.len() < gemm || b_ids.len() < gemm {
            return Err(format!(
                "the slots clause's prompts are {} and {} ids; each needs at least {gemm}",
                a_ids.len(),
                b_ids.len()
            )
            .into());
        }
        let mut alone = Vec::new();
        for (ids, what) in [(&a_ids, "alone A"), (&b_ids, "alone B")] {
            alone.push(run_streamed(&addr, ids, SLOTS_PREDICT, what)?);
        }
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy0 = busy_total(&url, decode0)?;
        // Both posted at once: the prompts serialize on the engine thread (a
        // round or two of skew), so nearly every round carries both slots —
        // the (v2) bound's shape.
        let first = streamed(&addr, &a_ids, SLOTS_PREDICT)?;
        let second = streamed(&addr, &b_ids, SLOTS_PREDICT)?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "together A"), (second, "together B")] {
            h.join()
                .map_err(|_| format!("slots: the {what} request's thread panicked"))?;
            let ran = rx
                .recv()
                .map_err(|_| format!("slots: the {what} request's thread gave no answer"))?;
            match ran {
                Ok(s) => together.push(Some(s)),
                Err(e) => {
                    println!("slots: the {what} request failed: {e}");
                    together.push(None);
                }
            }
        }
        let decode1 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy1 = busy_total(&url, decode1)?;
        let swaps = metric(&url, "swaps_total")?;
        println!(
            "slots alone: {} and {} ids; together {} and {} ids, drafts {:?} -> {:?} and {:?} \
             -> {:?}; decode {decode0} -> {decode1}, busy {busy0} -> {busy1}, swaps {swaps:?}",
            alone[0].as_ref().map_or(0, |s| s.tokens.len()),
            alone[1].as_ref().map_or(0, |s| s.tokens.len()),
            together[0].as_ref().map_or(0, |s| s.tokens.len()),
            together[1].as_ref().map_or(0, |s| s.tokens.len()),
            alone[0].as_ref().map(|s| s.drafts),
            together[0].as_ref().map(|s| s.drafts),
            alone[1].as_ref().map(|s| s.drafts),
            together[1].as_ref().map(|s| s.drafts),
        );
        // (v1) The slots hold separate sequences and each slot's draft its
        // own state: together ids and draft counts are alone ones (a failed
        // request is red, not the clause's end).
        check(
            &mut ok,
            "slots_alone_ids_stay",
            together[0].as_ref().is_some_and(|t| !t.tokens.is_empty())
                && together[1].as_ref().is_some_and(|t| !t.tokens.is_empty())
                && alone[0]
                    .as_ref()
                    .is_some_and(|a| a.tokens == together[0].as_ref().unwrap().tokens)
                && alone[1]
                    .as_ref()
                    .is_some_and(|a| a.tokens == together[1].as_ref().unwrap().tokens),
        );
        check(
            &mut ok,
            "slots_alone_draft_counts_stay",
            together[0].as_ref().is_some_and(|t| t.drafts.0 > 0)
                && together[1].as_ref().is_some_and(|t| t.drafts.0 > 0)
                && alone[0]
                    .as_ref()
                    .is_some_and(|a| a.drafts == together[0].as_ref().unwrap().drafts)
                && alone[1]
                    .as_ref()
                    .is_some_and(|a| a.drafts == together[1].as_ref().unwrap().drafts),
        );
        // (v2) Nothing parks, and a round carries both slots (the module
        // header's derivation): >= 1.5 against the turns' 1.0.
        let ratio = (busy1 - busy0) / (decode1 - decode0);
        check(
            &mut ok,
            "slots_rounds_carry_both_slots",
            swaps.is_none_or(|v| v == 0.0) && ratio >= 1.5,
        );
        // (v3) While both stream, the arrivals interleave.
        interleave(&mut ok, "slots_streams_interleave", &together)?;
        println!("slots server stopped: {}", served.stop()?);
        slots_default_residency(dir, &mut ok)
    }

    /// The slots clause's second server (the module header): the seat under
    /// its own defaults at `--parallel 2` — `--place a`, neither lever set —
    /// runs the together pair once and holds (v2) and (v3) alone; under the
    /// adaptive residency a stream's bits need not equal its alone run's, so
    /// no id equality is asked. This gate's own card is the 3090; this
    /// server's stage is the A6000 (the user's case), its load waiting out a
    /// timing sitting by the box guard this gate runs under.
    fn slots_default_residency(dir: &Path, ok: &mut bool) -> Result<bool, GateError> {
        let own = dir.join("slots-default");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env_remove(bloomery_levers::DRAFT)
            .env_remove(bloomery_levers::RESIDENCY)
            .env_remove(bloomery_levers::MTP_HEAD_ROWS)
            .env_remove(bloomery_levers::MTP_DRAFT);
        let args = [
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--place",
            "a",
            "--parallel",
            "2",
        ];
        let mut served = Served38::spawn_with(&args, &own, &mut cmd)?;
        println!("slots-default server pid {}", served.child.id());
        let addr = match served.address(&err_log, POLLS, POLL) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "slots-default: the server never listened: {e}; it said: {}",
                    std::fs::read_to_string(&err_log).unwrap_or_default()
                );
                check(ok, "slots_default_rounds_carry_both_slots", false);
                check(ok, "slots_default_streams_interleave", false);
                let _ = served.stop();
                return Ok(*ok);
            }
        };
        let url = |p: &str| format!("http://{addr}{p}");
        let line = parallel_line(&err_log)?
            .ok_or("the slots-default server printed no `parallel` line")?;
        println!("slots-default {line}");
        check(
            ok,
            "slots_default_parallel_line_names_the_rule",
            line.contains("rule=slots") && parallel_agrees(&line, Some(2)) == Some(true),
        );
        let (a_ids, b_ids) = (
            rendered(&url, json!([{ "role": "user", "content": TURN_A }]))?,
            rendered(&url, json!([{ "role": "user", "content": SLOTS_B }]))?,
        );
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy0 = busy_total(&url, decode0)?;
        let first = streamed(&addr, &a_ids, SLOTS_PREDICT)?;
        let second = streamed(&addr, &b_ids, SLOTS_PREDICT)?;
        let mut together = Vec::new();
        for ((h, rx), what) in [(first, "default A"), (second, "default B")] {
            h.join()
                .map_err(|_| format!("slots-default: the {what} request's thread panicked"))?;
            let ran = rx.recv().map_err(|_| {
                format!("slots-default: the {what} request's thread gave no answer")
            })?;
            match ran {
                Ok(s) => together.push(Some(s)),
                Err(e) => {
                    println!("slots-default: the {what} request failed: {e}");
                    together.push(None);
                }
            }
        }
        let decode1 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy1 = busy_total(&url, decode1)?;
        let swaps = metric(&url, "swaps_total")?;
        println!(
            "slots-default together: {} and {} ids; decode {decode0} -> {decode1}, busy {busy0} \
             -> {busy1}, swaps {swaps:?}",
            together[0].as_ref().map_or(0, |s| s.tokens.len()),
            together[1].as_ref().map_or(0, |s| s.tokens.len()),
        );
        let ratio = (busy1 - busy0) / (decode1 - decode0);
        check(
            ok,
            "slots_default_rounds_carry_both_slots",
            swaps.is_none_or(|v| v == 0.0) && ratio >= 1.5,
        );
        interleave(ok, "slots_default_streams_interleave", &together)?;
        println!("slots-default server stopped: {}", served.stop()?);
        Ok(*ok)
    }

    /// The `cache` clause on a server of its own with the draft the main
    /// server did not run (`drafted`), the residency off.
    fn cache_other(dir: &Path, drafted: bool) -> Result<bool, GateError> {
        let label = if drafted { "drafted" } else { "plain" };
        let own = dir.join(format!("cache-{label}"));
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let mut cmd = Command::new(Served38::exe()?);
        cmd.env(bloomery_levers::DRAFT, if drafted { "mtp" } else { "off" })
            .env(bloomery_levers::RESIDENCY, "off");
        if !drafted {
            cmd.env_remove(bloomery_levers::MTP_HEAD_ROWS)
                .env_remove(bloomery_levers::MTP_DRAFT);
        }
        let mut served = Served38::spawn_with(&SERVER_ARGS, &own, &mut cmd)?;
        println!("cache {label} server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let ok = cache(&url, &err_log, drafted, label)?;
        println!("cache {label} server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The plan's card expert bytes on the gate card at a slot's `ctx`, the
    /// two resident slots of this clause's flagless server counted
    /// (`plan_with_slots`), plain, as the server makes it; `None` when no
    /// plan takes the context.
    fn card_at(
        inputs: &PlanInputs,
        levers: &PlanLevers,
        experts: Experts,
        ctx: usize,
        free: Option<u64>,
    ) -> Result<Option<u64>, GateError> {
        let mut machine = machine_for_experts(
            RTX_3090,
            inputs.spec.layers.len(),
            u64::try_from(ubatch_for(ctx)?)?,
            experts,
        );
        machine.cards[0].free_bytes = free;
        Ok(inputs
            .plan_with_slots(&machine, u64::try_from(ctx)?, levers, experts, 2)
            .ok()
            .and_then(|p| p.cards.first().map(|c| c.expert_bytes)))
    }

    /// The `ctx` clause: a plain server with no `--ctx-size` (and no
    /// `--parallel`, so the seat's default two resident slots) prints its
    /// `ctx` line before its load (it is stopped there), against this gate's
    /// own plans of the file — every plan counting the two slots, as the
    /// seat's does: its context is a slot's, the margin rule's answer
    /// (`margin`: the line's own `margin_ctx`, which the clause below holds
    /// to the gate's plans), or the largest the card holds when that is
    /// fewer than [`CTX`] (`card`), or the prompt cache's clamp of the
    /// margin's answer (`cache`, composed the seat's way from the cache
    /// line's own ram — the clamp rides MemAvailable as the fit rides the
    /// census); the line's `fit` is the largest slot context the card holds
    /// up to the file's serving cap (`place::serve_ctx`, its
    /// `context_length`) — its plan taken, and the one past it refused when
    /// the card bounds it, else the cap itself — with that plan's card
    /// expert bytes; its `margin_ctx` holds at most the plan's margin fewer
    /// card expert bytes than the two-slot plan at [`CTX`], and the next
    /// multiple of [`CTX_STEP`] past it more (or it is the fit). A server
    /// asked for a total whose slot share passes the fit is refused by name
    /// before it listens: by the card's rule when the card bounds the fit,
    /// by the cap's (the trained context, YaRN not built) when the cap
    /// does. Mutants: the default ignoring the fit (staying at the base);
    /// the default passing the fit through without the margin rule; the
    /// refusal of a context past the card taken out (the load's own plan
    /// refuses it, not by the rule's name); the cap's refusal taken out
    /// (the card's rule names the clamped fit instead).
    // PIN(2026-10-05): re-derived for the seat's default two resident slots — every
    /// `at` below is `plan_with_slots(2)` at a slot's context, the default `--parallel`
    /// names two (the `parallel` line's check), and the refusal arm asks for the total
    /// whose slot share is one past the fit. Was the one-sequence search (round
    /// q38rules).
    fn ctx(dir: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let own = dir.join("ctx");
        std::fs::create_dir_all(&own)?;
        let err_log = own.join("server.err");
        let spawn = |args: &[&str]| -> Result<Served38, GateError> {
            let mut cmd = Command::new(Served38::exe()?);
            cmd.env(bloomery_levers::DRAFT, "off")
                .env(bloomery_levers::RESIDENCY, "off")
                .env_remove(bloomery_levers::MTP_HEAD_ROWS)
                .env_remove(bloomery_levers::MTP_DRAFT);
            Served38::spawn_with(args, &own, &mut cmd)
        };
        let mut served = spawn(&DEFAULT_ARGS)?;
        let mut line = None;
        for _ in 0..POLLS {
            let text = std::fs::read_to_string(&err_log).unwrap_or_default();
            line = text
                .lines()
                .find(|l| l.starts_with("ctx rule="))
                .map(str::to_owned);
            // The `plan` record carries the free reading the re-derivations
            // below take ([`server_card_free`]), and the seat prints its `ctx`
            // line before it: both must be in the log before the server
            // stops, or the reading is raced away.
            let head = format!("{} ", record::PLAN38.head);
            let plan = text.lines().any(|l| l.starts_with(&head));
            if (line.is_some() && plan) || served.child.try_wait()?.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        if served.child.try_wait()?.is_none() {
            println!("ctx: the default server stopped: {}", served.stop()?);
        }
        let line = line.ok_or("the default server printed no `ctx` line")?;
        println!("ctx: {line}");
        // The resident-slot default's own line (the module header): this
        // flagless server is the one that takes it, [`SERVER_ARGS`] naming
        // the one slot its clauses count on. The slots its rule names are
        // what the line's own terms hold — the default's 2 and the split it
        // names the `ctx` line's own context. FAIL-first: a default that
        // serves another count (or a split off the `ctx` line) turns this
        // red.
        let parallel =
            parallel_line(&err_log)?.ok_or("the default server printed no `parallel` line")?;
        println!("ctx: {parallel}");
        let field = |k: &str| -> Option<String> {
            line.split_whitespace()
                .find_map(|w| w.strip_prefix(k)?.strip_prefix('='))
                .map(str::to_owned)
        };
        let num = |k: &str| field(k).and_then(|v| v.parse::<usize>().ok());
        let (rule, ctx, fit) = (field("rule"), num("ctx"), num("fit"));
        let fit_bytes = field("fit_card_expert_bytes").and_then(|v| v.parse::<u64>().ok());
        let margin_ctx = num("margin_ctx");
        let (Some(rule), Some(ctx), Some(fit), Some(fit_bytes), Some(margin_ctx)) =
            (rule, ctx, fit, fit_bytes, margin_ctx)
        else {
            return Err(format!(
                "a `ctx` line without its rule, ctx, fit, fit_card_expert_bytes and margin_ctx: \
                 {line}"
            )
            .into());
        };
        // The rule's own terms: the default's 2 slots at the `ctx` line's own
        // context (the line's `total` twice it, held by `parallel_agrees`).
        let pfield = |k: &str| {
            parallel
                .split_whitespace()
                .find_map(|w| w.strip_prefix(k)?.strip_prefix('='))
                .and_then(|v| v.parse::<usize>().ok())
        };
        let parallel_ok = parallel_agrees(&parallel, None).unwrap_or(false)
            && pfield("slots") == Some(2)
            && pfield("slot_ctx") == Some(ctx);
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::describe(&split)?;
        let experts = experts38(levers)?;
        let plan_levers = PlanLevers::from_levers(levers)?;
        let free = Some(server_card_free(&err_log)?);
        // The default the seat's rules compose to also answers the prompt
        // cache's bound, whose ram rides MemAvailable as the fit rides the
        // census: the gate takes the server's own `cache` line's ram, not a
        // fresh reading.
        let ram = record_line(&err_log, "cache")?
            .ok_or("the default server printed no `cache` line")?
            .split_whitespace()
            .find_map(|w| w.strip_prefix("ram="))
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or("a `cache` line without its ram")?;
        let at = |c: usize| card_at(&inputs, &plan_levers, experts, c, free);
        let base = at(CTX)?.ok_or("no plan at the base context")?;
        let lost = |c: usize| -> Result<Option<u64>, GateError> {
            Ok(at(c)?.map(|e| base.saturating_sub(e)))
        };
        let margin = model::placement::workstation::MARGIN;
        let here = lost(margin_ctx)?;
        let past = lost(margin_ctx + CTX_STEP)?;
        let (at_fit, past_fit) = (at(fit)?, at(fit + 1)?);
        let cap = usize::try_from(serve_ctx(1, &inputs.hp)?)?;
        // PIN(2026-10-02): the fit is the card's largest context up to the file's serving cap
        // (round q38rules: `serve_ctx`, 262,144 for Qwen3.8); where the card holds more, the fit
        // is the cap and a plan past it is still a plan; was the card's largest, `past_fit` none.
        let card_bounds = fit < cap;
        // One session's whole-context state (`Seq38`), the bound the seat's
        // `cache` rule clamps the default with (`Ctx38::host_bound`).
        let gdn = inputs
            .spec
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, Mixer::DeltaRule(_)))
            .count();
        let qsa = inputs.spec.layers.len() - gdn;
        let state = |n: usize| seq_positional_bytes(qsa, n) + 2 * checkpoint_bytes(gdn);
        let cache_ctx = |from: usize| {
            let mut lo = 1;
            let mut hi = from;
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if state(mid) <= ram {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            (lo / CTX_STEP * CTX_STEP).max(lo.min(CTX_STEP))
        };
        let (want_rule, want_ctx) = if fit < CTX {
            ("card", fit)
        } else if ram == 0 || state(margin_ctx) <= ram {
            ("margin", margin_ctx)
        } else {
            ("cache", cache_ctx(margin_ctx))
        };
        println!(
            "ctx: the gate's plans: lost at {margin_ctx} {here:?}, at {} {past:?}, fit {fit} \
             {at_fit:?}, past it {past_fit:?}; the serving cap {cap} ({}); the default the rules \
             compose to: rule {want_rule} ctx {want_ctx} (the cache line's ram {ram})",
            margin_ctx + CTX_STEP,
            if card_bounds {
                "the card bounds the fit"
            } else {
                "the cap bounds the fit"
            }
        );
        let mut ok = true;
        check(
            &mut ok,
            "ctx_default_is_the_rules",
            rule.as_str() == want_rule && ctx == want_ctx,
        );
        check(
            &mut ok,
            "parallel_default_holds_what_its_rule_names",
            parallel_ok,
        );
        check(
            &mut ok,
            "ctx_line_prints_the_fit_and_the_margin",
            at_fit == Some(fit_bytes)
                && fit <= cap
                && (if card_bounds {
                    past_fit.is_none()
                } else {
                    fit == cap
                })
                && here.is_some_and(|l| l <= margin)
                && margin_ctx <= fit
                && (margin_ctx == fit
                    || (margin_ctx.is_multiple_of(CTX_STEP) && past.is_none_or(|l| l > margin))),
        );
        // The refusal arm asks for a total whose slot share is one past the
        // fit: the server splits `--ctx-size` over its default two slots, so
        // the total the arm names is twice the share it refuses.
        let over = (2 * (fit + 1)).to_string();
        let share = fit + 1;
        let args: Vec<&str> = DEFAULT_ARGS
            .iter()
            .copied()
            .chain(["--ctx-size", &over])
            .collect();
        let mut refused = spawn(&args)?;
        let mut status = None;
        for _ in 0..POLLS {
            status = refused.child.try_wait()?;
            if status.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        let text = std::fs::read_to_string(&err_log).unwrap_or_default();
        let named = if card_bounds {
            format!(
                "--ctx-size {over}: with --parallel 2 each slot takes {share} positions and the \
                 card holds at most {fit} a slot beside the plan's dense weights (`--place gate`)"
            )
        } else {
            format!(
                "a context of {share} positions: the file was trained at {cap} (context_length), \
                 and a load serves at most {cap}; YaRN scaling past the trained context is not \
                 implemented"
            )
        };
        println!(
            "ctx: --ctx-size {over}: exit {status:?}; named {}",
            text.contains(&named)
        );
        if status.is_none() {
            println!("ctx: the refused server stopped: {}", refused.stop()?);
        }
        check(
            &mut ok,
            "ctx_past_the_fit_is_refused_by_name",
            status.is_some_and(|s| !s.success())
                && text.contains(&named)
                && !text.contains("listening on http://"),
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
        if levers.draft() != Some("mtp") && levers.mtp_draft().is_some() {
            return Err(
                "BLOOMERY_MTP_DRAFT names the MTP draft file; it needs BLOOMERY_DRAFT=mtp".into(),
            );
        }
        if let Some(word) = levers.residency() {
            return Err(format!(
                "BLOOMERY_RESIDENCY={word}: the gate sets it on every server it starts (off, and \
                 the residency clause's mid-p0-s1)"
            )
            .into());
        }
        let a = parse_args()?;
        let reference = gen_tokens(&a.gen_log)?;
        std::fs::create_dir_all(&a.dir)?;
        let exe = Served38::exe()?;
        let err_log = a.dir.join("server.err");
        let mut cmd = Command::new(&exe);
        cmd.env(
            bloomery_levers::DRAFT,
            if levers.draft() == Some("mtp") {
                "mtp"
            } else {
                "off"
            },
        )
        .env(bloomery_levers::RESIDENCY, "off");
        let mut served = Served38::spawn_with(&SERVER_ARGS, &a.dir, &mut cmd)?;
        println!("server pid {}", served.child.id());
        let addr = served.address(&err_log, POLLS, POLL)?;
        let url = |p: &str| format!("http://{addr}{p}");
        println!("server listening on {addr}");

        let mut ok = true;
        let (st, body) = curl(&url("/health"), None, false)?;
        println!("health {st} {body}");
        check(&mut ok, "health_ok", st == 200 && body.contains("\"ok\""));
        // The seat's `plan` line names its stage card's free bytes at plan
        // time, at most the device's usable bytes — the census term the
        // expert rule filled within (memguard). FAIL-first: a plan line that
        // drops it, or names it past the usable bytes, turns this red.
        let free_named = record_line(&err_log, "plan")?.is_some_and(|l| {
            let free = l
                .split("card_free=")
                .nth(1)
                .and_then(|t| t.split(' ').next())
                .and_then(|n| n.parse::<u64>().ok());
            let usable = l
                .split("devices=")
                .nth(1)
                .and_then(|d| d.split([',', ' ', ']']).next())
                .and_then(|c| c.rsplit(':').next())
                .and_then(|n| n.parse::<u64>().ok());
            free.is_some_and(|f| usable.is_some_and(|u| f <= u))
        });
        println!("plan names the stage card's free bytes {free_named}");
        check(&mut ok, "plan_names_the_cards_free_bytes", free_named);
        let argv: Vec<String> = std::iter::once(exe.to_string_lossy().into_owned())
            .chain(SERVER_ARGS.iter().map(|s| (*s).to_owned()))
            .collect();
        ok &= props_engine(&url, &argv, served.child.id(), &levers, &err_log)?;

        let completion = json!({
            "prompt": a.prompt, "n_predict": N_PREDICT, "temperature": 0, "return_tokens": true,
        });
        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c1 = json_of("/completion", st, &body)?;
        let first = ids_of(&c1["tokens"]);
        let stop = c1["stop_type"].as_str().unwrap_or("").to_owned();
        println!(
            "completion tokens {first:?} stop_type={stop} content={}",
            c1["content"]
        );
        println!("generate_qwen3moe {reference:?}");
        check(
            &mut ok,
            "completion_ids_are_generate_qwen3moe",
            agree(&first, &stop, &reference),
        );
        if levers.draft() == Some("mtp") {
            // The drafted server's own clause: the greedy ids above are the
            // plain run's (a draft changes which passes run, never a token),
            // and the pass carried its counts.
            let d = &c1["timings"];
            check(
                &mut ok,
                "drafted_timings_carry_the_draft_counts",
                d["draft_n"].as_u64().is_some_and(|n| n > 0)
                    && d["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
            );
        }

        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c2 = json_of("/completion", st, &body)?;
        println!(
            "completion again {:?} cache_n={}",
            ids_of(&c2["tokens"]),
            c2["timings"]["cache_n"]
        );
        check(
            &mut ok,
            "completion_again_identical",
            ids_of(&c2["tokens"]) == first,
        );

        let chat = json!({
            "messages": [{"role": "user", "content": CHAT}],
            "temperature": 0, "max_tokens": CHAT_PREDICT,
        });
        let (st, body) = curl(&url("/v1/chat/completions"), Some(&chat), false)?;
        let plain = json_of("/v1/chat/completions", st, &body)?;
        let plain_content = plain["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_owned();
        println!(
            "chat content={:?} finish_reason={} usage={}",
            plain_content, plain["choices"][0]["finish_reason"], plain["usage"]
        );
        let mut streamed = chat.clone();
        streamed["stream"] = json!(true);
        let (st, sse) = curl(&url("/v1/chat/completions"), Some(&streamed), true)?;
        std::fs::write(a.dir.join("chat-stream.sse"), &sse)?;
        let events: Vec<&str> = sse
            .split("\n\n")
            .filter_map(|e| e.strip_prefix("data: "))
            .collect();
        let mut deltas = String::new();
        for e in &events {
            if let Ok(v) = serde_json::from_str::<Value>(e)
                && let Some(t) = v["choices"][0]["delta"]["content"].as_str()
            {
                deltas.push_str(t);
            }
        }
        println!(
            "chat stream HTTP {st}: {} events, content={deltas:?}, last event {:?}",
            events.len(),
            events.last().copied().unwrap_or("")
        );
        check(
            &mut ok,
            "chat_stream_equals_non_stream",
            st == 200 && deltas == plain_content,
        );
        check(
            &mut ok,
            "chat_stream_ends_with_done",
            events.last() == Some(&"[DONE]"),
        );

        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({"content": a.prompt})),
            false,
        )?;
        let tok = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        println!("tokenize {tok:?} ids {:?}", a.ids);
        check(&mut ok, "tokenize_is_the_ids", tok == a.ids);

        let (st, body) = curl(&url("/completion"), Some(&completion), false)?;
        let c3 = json_of("/completion", st, &body)?;
        println!(
            "completion after the other requests {:?} cache_n={}",
            ids_of(&c3["tokens"]),
            c3["timings"]["cache_n"]
        );
        check(
            &mut ok,
            "completion_after_other_requests_identical",
            ids_of(&c3["tokens"]) == first,
        );

        let id = a.ids.iter().copied().min().ok_or("--ids is empty")?;
        ok &= position_limit(&url, id)?;
        let drafted = levers.draft() == Some("mtp");
        ok &= cache(
            &url,
            &err_log,
            drafted,
            if drafted { "drafted" } else { "plain" },
        )?;
        let sampled = json!({
            "prompt": a.prompt, "n_predict": N_PREDICT, "temperature": SAMPLED_TEMPERATURE,
            "seed": SAMPLED_SEED, "return_tokens": true,
        });
        if levers.draft() == Some("mtp") {
            ok &= continued(&url, &err_log, &a.ids, &reference)?;
            let (st, body) = curl(&url("/completion"), Some(&sampled), false)?;
            let top1 = json!({
                "prompt": a.prompt, "n_predict": N_PREDICT, "temperature": SAMPLED_TEMPERATURE,
                "top_k": 1, "seed": SAMPLED_SEED, "return_tokens": true,
            });
            let (st1, body1) = curl(&url("/completion"), Some(&top1), false)?;
            ok &= sampled_served((st, &body), (st1, &body1), &first)?;
        }

        println!("server stopped: {}", served.stop()?);
        ok &= cache_other(&a.dir, !drafted)?;
        ok &= ctx(&a.dir, &levers)?;
        ok &= residency(&a.dir, &completion)?;
        // The resident slots together (the module header): its own servers,
        // the first with the draft on whatever this binary's lever says.
        ok &= slots_flow_together(&a.dir)?;
        if ok {
            println!("gate-gpu-qwen38-serve: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
