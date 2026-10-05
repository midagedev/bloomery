//! `gate_glm5next_serve` — the GLM seat of `bloomery-serve` (`--model glm`)
//! on the 3090 (placement gate), driven over HTTP, against
//! `generate_glm5next` on the same card, as one process under the GPU gate
//! lock.
//!
//!     gate_glm5next_serve --arm plain|drafted|slots --dir <out>
//!
//! Each arm starts the server beside this binary (`--model glm --host
//! 127.0.0.1 --port 0 --place gate --ctx 2048 --slot-save-path /tmp
//! --parallel 1`, the one-slot server its clauses hold; the levers are set
//! here), reads
//! its address from its stderr, waits for `/health`, takes the prompt of one
//! chat turn ([`CHAT`]) as the server renders it (`/apply-template`, then
//! `/tokenize` of that text without BOS: the ids `/v1/chat/completions`
//! encodes), checks the server, kills it by the handle this binary spawned it
//! with and waits for it, then runs `generate_glm5next --place gate --ctx
//! 2048 --tokens <those ids> -n 16` beside this binary under the same levers
//! and holds the ids the server served against it. The loads run one after
//! the other (four in `plain`, two in `drafted`); the recipe runs the arms
//! as two processes, each under its own bound. `slots` runs the drafted
//! arm's (s5) alone — its server and clauses, nothing else — for a run of
//! those clauses by name; the recipe runs them inside `drafted`.
//!
//! `plain` (`BLOOMERY_DRAFT=off BLOOMERY_RESIDENCY=off`), after three runs of
//! the seat with both levers unset and `--plan`, which print the unset rule's
//! records and the plan and exit 0 before any card is opened: under `--place
//! a` and under `--place bp` (plan (b′), the 3090 an expert tier) a `draft
//! unset draft=mtp` and a `residency unset residency=mid-p0-s1` record and
//! the `residency host` record of the word — its pinned count 0 and the
//! headroom past the churn pool non-negative, the pool inside the plan's
//! host terms — every placement's `ctx` line naming a default context at or
//! past the floor; under `--place gate` both `off`, no `residency host`:
//!
//! - the seat's `--ctx` rule ([`ctx`], on the gate card, one slot): the
//!   default's `ctx` line against this gate's own plans of the file — the
//!   rule's context, its fit and its margin — the flag winning below the
//!   floor, and a context past `place::ORACLE_POSITIONS` refused by name
//!   before it listens;
//!
//! - the load prints `load draft=off (BLOOMERY_DRAFT=off)` and the word as a
//!   `residency lever` record (`why=set`), no `residency host` and no verify
//!   capture; `/props`' `engine` carries no `draft` and no card `draft` class;
//! - the chat turn at temperature 0 is a 200;
//! - `/completion` of the prompt's ids at temperature 0 with `return_tokens`:
//!   its ids are the CLI's `tokens` line — all 16, or a prefix ending in the
//!   end-of-generation id when the server stopped there (the CLI does not
//!   stop at it). Exact: the server feeds the prompt less its last id, then
//!   one step, and so does the CLI (`--last-step`) — the same batches, so
//!   the same bits (a batch past a chunk runs the GEMM, not the steps'
//!   gemvs);
//! - the server's `load` record and the CLI's, both under
//!   `BLOOMERY_PREFILL_GROUP` unset, print `group=` the group the plan
//!   reserves the prompt units for (`place::PROMPT_GROUP`);
//! - `POST /residency/reset` is the server's 501 (no machine);
//! - `cache` ([`cache`]): two sessions through the prompt cache, A's state
//!   saved when B takes the slot and put back when A returns — every held
//!   position kept and the ids of the same requests with no switch, and a
//!   resend that shares only A's turn keeps the checkpoint the state carried
//!   there, with the no-switch run's ids;
//! - `round_trip` ([`round_trip`]): a session's second turn sent as the chat
//!   messages a client sends back — the reply's content, no
//!   `reasoning_content`, the template rendering the think span empty —
//!   keeps the first turn's prompt call's end (`cache_n` its prompt less
//!   one) and saves nothing (no `cache save` line);
//! - then a third load, the server under `BLOOMERY_DRAFT=mtp
//!   BLOOMERY_RESIDENCY=off`: it prints `load draft=mtp` and no `residency
//!   host`, the same `/completion` carries the draft's counts, and its ids
//!   are the plain CLI's, all of them; a sampled `/completion` cut to
//!   `top_k` 1 (temperature 0.8, a fixed seed) is served through plain steps,
//!   drafting nothing, each step's row read before the step is told to the
//!   draft, and its ids are the plain CLI's too. Green-only for the row's
//!   read: the NextN walk writes no row of the target's head, so a `step_row`
//!   left at the default reads the same row; the clause holds the sampled
//!   path's ids end to end; and `cache` again under the draft, where the
//!   draft also rejoins A where A left it, its counts the no-switch run's.
//!   Exact: with no residency nothing
//!   moves between the host and the card, every kept token is the target's
//!   own argmax, and a verify's rows are its steps' bits — the window's
//!   wiring end to end, which the drafted arm's clause holds only up to the
//!   first landing.
//! - a fourth load, a plain server of `--parallel 2` under
//!   `BLOOMERY_STEP_STATS=1` ([`slots_one_pass`]; no draft, no residency —
//!   the bits need no expert moving): its `parallel` line names `pass=one`,
//!   and two rendered turns ([`TURN_A`], [`TURN_C`]), each run alone for
//!   [`SLOTS_PREDICT`] greedy ids (`cache_prompt` off), then both posted at
//!   once, answer each one's alone ids, every `slots round` record of the
//!   together window reading `cmd=step`, `slots=2`, `rows=2` and
//!   `passes=ceil(rows/2)` — the body's two-row bound (FAIL-first: the seat
//!   left on the fallback loop prints `passes=rows`);
//!
//! `drafted` (`BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=mid-p0-s1`, the clip's
//! levers; the word set explicitly, one the gate plan's card slots take):
//!
//! - the load prints `load draft=mtp`, the verify's capture
//!   (`pair_graph_nodes`), the word as a `residency lever` record (`why=set`)
//!   and a `residency host` record of it; `/props`' `engine.draft` is `mtp`
//!   of width 1 naming the target file, and the card's `draft` class is the
//!   bytes this gate's own NextN plan of the file reserves for the layer
//!   (its card terms and arena);
//! - `/completion` of the prompt's ids from the seed (`cache_prompt` off)
//!   carries the draft's counts (`draft_n` and `draft_n_accepted` above 0)
//!   and ends, in its `residency pass` records, a prompt pass, a step pass
//!   (the prompt's last id) and at least one `pair` pass (a window that
//!   proposed) (mutants: the seat drives plain steps on the NextN load — no
//!   counts, no `pair` pass, no `engine.draft`; the seat opens the load
//!   without the word — no `residency pass` record, and the reset below is a
//!   501);
//! - `POST /residency/reset` is a 200 whose `diff` is 0, and the server
//!   prints its `residency reset` record with the same counts; the same
//!   `/completion` after it gives the first one's ids (from the seed, one
//!   history lands the same flips at the same passes, so two runs of it are
//!   bit for bit);
//! - the chat turn at temperature 0 drafts (`draft_n` above 0);
//! - a state put back runs on the slot map as it stands
//!   ([`resume_under_residency`]): kept whole, its passes logged, the draft
//!   rejoining;
//! - (s4) the context split ([`split_refused`]), each a `--plan` run under
//!   `BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=off` that ends, refused by name,
//!   before its `ctx` line and its plan: `--ctx 4 --parallel 2` (two
//!   positions a slot, under the draft's three) and `--parallel 0`;
//! - (s5) resident slots ([`slots_flow_together`]), on a server of
//!   `--parallel 2` under `BLOOMERY_DRAFT=mtp BLOOMERY_RESIDENCY=off` (bits
//!   hold only with no expert moving) and `BLOOMERY_STEP_STATS=1`: its
//!   `parallel` line names `pass=one` — the open runs a drafted round as one
//!   pass where the body's pass of several slots (`SlotRows::MAX_ROWS`)
//!   holds two windows of the seat's depth (one proposal, two rows a
//!   window); a body of fewer rows keeps the turns, which this clause reads
//!   red; `/props`' `total_slots` is 2, its
//!   `n_ctx` and the listening record's `ctx` a slot's half of the `--ctx`,
//!   and its `vram_kv_bytes` the stage card's KV term a plan of two
//!   sequences counts at that context (`PlanInputs::kv_term`: the trunk's
//!   stores for each, the bytes beside them for the second; the next-token
//!   layer's store is the card's `draft` class); two rendered turns
//!   ([`TURN_A`], [`TURN_C`]), each run alone for [`SLOTS_PREDICT`] greedy ids
//!   (`cache_prompt` off), every one of them, the draft proposing; then both
//!   posted at once: each one's ids, `draft_n` and `draft_n_accepted` its
//!   alone run's (a slot's sequence and its draft's side are its own across
//!   every select), `swaps_total` absent or 0, and over the together run the
//!   busy slots booked a call (the deltas of `n_busy_slots_per_decode` ×
//!   `n_decode_total` over `n_decode_total`) at least [`SLOTS_RATIO`]: the
//!   worker books one call a round carrying every running slot, and each
//!   request's prompt (its prompt call and its last id's step, which gives
//!   the first id) one call of one. After it a request runs a pass a round,
//!   each keeping one or two of the 63 ids left — 32 to 63 rounds; with the
//!   two requests' rounds `p₁ ≤ p₂` and the later one's first pass at most
//!   `s` rounds after the earlier's, the calls are at most `p₂ + s + 2` and
//!   the busy `p₁ + p₂ + 2`, so the ratio is at least (32 + 63 + 2)/(63 + s +
//!   2) = 97/(65 + s) — 1.45 at `s` = 2, 1.41 at 4 [derived] — where a server
//!   whose slots take the model in turns books one slot a call, 1.0. Every
//!   `slots round` record of the together window that reads `cmd=pass` (the
//!   alone runs print none, a round of one busy slot being a select and a
//!   pass) reads `slots=2` and `passes=1`, one of them `rows=2`: a two-slot
//!   round's windows, two rows each, fit one pass of such a body.
//!   Mutants: the seat wrapping its engine in `serve::SwapEngine` again (the
//!   turn rules: one busy a call, ratio 1.0, red on the ratio check); a split
//!   that returns the total (`n_ctx`, the listening `ctx` and
//!   `vram_kv_bytes` the total's, red on the first two checks, and (s4)'s
//!   split planned, red there); the open deciding turns (the fallback loop's
//!   `passes=rows`, so `passes=2` on a round of two rows, and `pass=turns`).
//!   Then on the same server, sampled requests (temperature 0.8, top-k 40,
//!   a fixed seed a turn, `cache_prompt` off), which the server steps and
//!   never drafts: each turn alone, then both at once — each one's ids its
//!   alone run's, no draft counts, every `cmd=step` record of the window
//!   `slots=2` and `passes=rows` (one with `rows=2`) and no `cmd=pass`
//!   record; then the greedy [`TURN_A`] beside the sampled [`TURN_C`] — the
//!   greedy one's ids and draft counts its alone run's, the sampled one's
//!   ids its alone run's, every `cmd=pass` record of the window `slots=2`,
//!   `rows=1`, `passes=1` (at least one). These pin today's shape: a
//!   sampled row is a select and a step, its round the server's call apart
//!   from the drafted passes', because the drafted pass cannot carry a
//!   window that proposes nothing (`app::mtp::pass_slots` refuses depth 0,
//!   the body's NextN pass of several slots a slot of one row, and the
//!   drafted row lends no logits row). Mutant: the seat running a NextN
//!   load's step round as one pass — the body refuses it by name, the
//!   server dies, and the sampled clauses go red.
//! - then on the same server the drafted round's skip path
//!   ([`slots_skipping`]): a slot whose draft skips steps alone beside a
//!   drafting one. The first turn greedy leaves its slot holding the turn
//!   and its reply, the draft's waiting rows at the reply's end; a request
//!   that carries the turn with [`STRIPPED_A`] in place of the reply (greedy,
//!   `cache_prompt` on) keeps the checkpoint the turn's prompt call took at
//!   its end and its prompt call starts there, where the waiting rows do not
//!   end — the draft skips, its `mtp prompt` record naming why, and the
//!   request steps plainly (no draft counts). Run alone first; then both
//!   slots dropped, the history built again and the second turn run alone
//!   (`cache_prompt` off — a resend of it with the cache on would cut its own
//!   held end the same way and skip too), and the two posted at once: both
//!   answer, each request's ids — the drafting one's counts too — its alone
//!   run's, the skipping request keeping the turn's end (`cache_n` the turn
//!   less one), and the seat turns the skipping slot off before each one-pass
//!   round (its own `mtp prompt` record naming why), every `cmd=pass` record
//!   of the together window reading `slots=2`, `rows=2` and `passes=2` — the
//!   off slot's plain pass beside the drafting slot's window, where a round
//!   of two drafting slots runs one pass. Mutant: the seat not turning a
//!   skipping slot off — the slot joins the pass as its one plain row, which
//!   the body's NextN pass of several slots refuses by name, the server dies
//!   and the clause goes red.
//! - against the CLI under the same levers: the ids agree, all of them or a
//!   prefix ending in the end-of-generation id — the CLI takes the server's
//!   cut (`--last-step`), so both run the same batches, count the rule's
//!   steps the same and land their flips at the same passes; through which
//!   pass the first landing sits is printed, not held (the reset clause
//!   above holds the seed's determinism).
//!
//! A request that extends the held sequence (the prompt and the ids it
//! generated) is printed with its `cache_n` and the draft's `mtp prompt`
//! records, not held: the checkpoint rule keeps every fed position only for
//! a request that extends all of them.
//!
//! Logs and the CLI's output go to `--dir`. The server and the CLI inherit
//! this binary's environment, so the levers they act on are the seat's
//! (`ACTS_ON`, the same list): one the seat would refuse is refused here, at
//! `main`, before anything starts; `BLOOMERY_DRAFT` and `BLOOMERY_RESIDENCY`
//! set in this binary's environment are refused by name (the arm sets both).

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!(
        "gate_glm5next_serve: built without the `glm5next` feature; see `just gate-gpu-glm5next-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm5next_serve", gate::run())
}

#[cfg(feature = "glm5next")]
mod gate {
    use std::fs::File;
    use std::os::unix::process::CommandExt;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::mpsc;
    use std::thread::JoinHandle;
    use std::time::Duration;

    use bloomery_gpu_gates::record::{self, Log};
    use bloomery_gpu_gates::serve_client::{
        curl, ids_of, json_of, metric, parse_ids, server_log, stage_usable,
    };
    use bloomery_gpu_gates::{GateError, checks_failed, ref_model_path, verdict};
    use gguf::Split;
    use model::arch::glm5next::place::{
        KdaLanes, NextnInputs, ORACLE_POSITIONS, PROMPT_GROUP, PlanInputs,
    };
    use model::placement::{PlanLevers, workstation};
    use serde_json::{Value, json};
    use threads::helper::{Placement, spawn_helper};

    /// The levers the seat acts on — the same list `serve_seats::glm`
    /// parses, kept one with it: this gate starts the server with its own
    /// environment, so a lever the server would refuse is refused here first.
    const ACTS_ON: &[&str] = &[
        bloomery_levers::CARD_BUDGET,
        bloomery_levers::HOST_POPULATE,
        bloomery_levers::HOST_LOCK,
        bloomery_levers::CARD_DONTNEED,
        bloomery_levers::R8,
        bloomery_levers::PIN_MAIN,
        bloomery_levers::DRAFT,
        bloomery_levers::RESIDENCY,
        bloomery_levers::STEP_STATS,
    ];

    const USAGE: &str = "usage: gate_glm5next_serve --arm plain|drafted|slots --dir <out>";

    /// The stores both engines size, the seat's floor and the CLI's default.
    const CTX: usize = 2048;
    /// The seat's default context's multiple (its `CTX_STEP`).
    const CTX_STEP: usize = 256;
    /// The flag arm's context, below the seat's floor: a flag seen to win,
    /// not to coincide with a default the floor already takes.
    const FLAG_CTX: usize = 1536;
    /// A context past `ORACLE_POSITIONS`, the refusal the clause asks for.
    const OVER_CTX: usize = 99_999;
    /// The `ctx` clause's polls for the seat's pre-load lines, 1 s apart:
    /// the lines print at plan time, before the load's first-start JIT.
    const LINE_POLLS: usize = 90;
    const LINE_POLL: Duration = Duration::from_secs(1);
    /// The server's arguments after its path. The slot actions need a save
    /// directory; the gate asks only for `erase`, which writes nothing. One
    /// slot is pinned (`--parallel 1`): this gate's clauses hold the
    /// one-slot path's prompt cache, and the slots clause's server takes two
    /// on its own.
    const SERVER_ARGS: [&str; 14] = [
        "--model",
        "glm",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--place",
        "gate",
        "--ctx",
        "2048",
        "--slot-save-path",
        "/tmp",
        "--parallel",
        "1",
    ];
    /// The load takes a minute or two; the bound is 120 polls × 5 s.
    const POLLS: usize = 120;
    const POLL: Duration = Duration::from_secs(5);
    /// The greedy requests' length, the CLI's `-n`.
    const N_PREDICT: usize = 16;
    /// The slots clause's requests' length, and the least busy slots a call
    /// its together run books (the module header's derivation).
    const SLOTS_PREDICT: usize = 64;
    const SLOTS_RATIO: f64 = 1.4;
    /// The slots clause's sampled requests' seeds, one a turn ([`TURN_A`],
    /// [`TURN_C`]).
    const SEEDS: [u64; 2] = [7, 11];
    /// The one chat turn.
    const CHAT: &str = "What is the capital of France? Answer in one word.";
    /// The `cache` clause's sessions: A's first turn, B's turn, the text A
    /// sends after its reply, and the text A resends in place of its reply
    /// (its first id is not the reply's).
    const TURN_A: &str = "Name three rivers that flow through Germany and say in one sentence \
                          which of them is the longest.";
    const TURN_B: &str = "Write a haiku about a lighthouse in winter.";
    const LATER_A: &str = " And which of them reaches the sea first?";
    const STRIPPED_A: &str = "Rhine, Danube and Elbe; the Danube is the longest. And which of \
                              them reaches the sea first?";
    /// The `round_trip` clause's session: its first turn and its second,
    /// sharing nothing past the template's head with the `cache` clause's.
    const TURN_C: &str = "List the planets of the solar system in order from the Sun and say \
                          which of them is the largest.";
    const LATER_C: &str = "And which of them has the most moons?";
    /// The skipping clause's halves of the `mtp prompt` records that name a
    /// skip: why the draft skips at a prompt call that cut the held sequence
    /// (the join's reason, `app::mtp`'s), and why the seat turns such a slot
    /// off before a round of one pass (`serve_seats::glm`'s turn-off why).
    const SKIP_AT_JOIN: &str = "do not end at the call's first position";
    const SKIP_TURNED_OFF: &str = "a NextN pass of several slots takes each slot's whole verify";
    /// The drafted arm's residency word: no seed expert pinned, one spare a
    /// layer.
    const RESIDENCY_WORD: &str = "mid-p0-s1";

    /// The arms, one a process: `slots` is the drafted arm's (s5) alone,
    /// its server and clauses, which `drafted` runs too.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Arm {
        Plain,
        Drafted,
        Slots,
    }

    impl Arm {
        fn parse(v: &str) -> Result<Arm, GateError> {
            match v {
                "plain" => Ok(Arm::Plain),
                "drafted" => Ok(Arm::Drafted),
                "slots" => Ok(Arm::Slots),
                other => Err(format!("--arm is plain, drafted or slots, not {other}").into()),
            }
        }
    }

    /// The levers a process runs under: `BLOOMERY_DRAFT` and
    /// `BLOOMERY_RESIDENCY` set (both, on the server and the CLI alike), or
    /// both unset ([`UNSET`]).
    type Levers = &'static [(&'static str, &'static str)];
    /// The plain arm's server and CLI.
    const PLAIN: Levers = &[
        (bloomery_levers::DRAFT, "off"),
        (bloomery_levers::RESIDENCY, "off"),
    ];
    /// The plain arm's drafted server: the draft with no residency.
    const DRAFT_ONLY: Levers = &[
        (bloomery_levers::DRAFT, "mtp"),
        (bloomery_levers::RESIDENCY, "off"),
    ];
    /// The drafted arm's slots server: the draft with no residency (the bits
    /// need no expert moving) and the seat's rounds counted, each a `slots
    /// round` record.
    const DRAFT_ONLY_STATS: Levers = &[
        (bloomery_levers::DRAFT, "mtp"),
        (bloomery_levers::RESIDENCY, "off"),
        (bloomery_levers::STEP_STATS, "1"),
    ];
    /// The drafted arm's server and CLI.
    const DRAFTED: Levers = &[
        (bloomery_levers::DRAFT, "mtp"),
        (bloomery_levers::RESIDENCY, RESIDENCY_WORD),
    ];
    /// Both levers unset: the seat's own rule.
    const UNSET: Levers = &[];
    /// The plain arm's slots server: no draft, no residency (the bits need
    /// no expert moving) and the seat's rounds counted, each a `slots round`
    /// record.
    const PLAIN_STATS: Levers = &[
        (bloomery_levers::DRAFT, "off"),
        (bloomery_levers::RESIDENCY, "off"),
        (bloomery_levers::STEP_STATS, "1"),
    ];

    /// The server this binary started; killed and reaped on every way out.
    /// `serve_client::Served`'s core with this gate's server's name — that
    /// module's `exe` names `bloomery-serve-ds41`.
    struct Served {
        child: Child,
    }

    impl Served {
        /// `bloomery-serve` beside this binary with `args` and the arm's
        /// levers, stdout to `<dir>/server.out` and stderr to
        /// `<dir>/server.err`. The child is killed when this process dies, so
        /// a runner's bound that ends this process does not leave the server
        /// holding a card.
        fn spawn(args: &[&str], dir: &Path, levers: Levers) -> Result<Served, GateError> {
            let exe = beside("bloomery-serve")?;
            let mut cmd = Command::new(&exe);
            cmd.args(args)
                .env_remove(bloomery_levers::DRAFT)
                .env_remove(bloomery_levers::RESIDENCY)
                .envs(levers.iter().copied())
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
            Ok(Served { child })
        }

        /// Waits for the `listening on http://<addr>` line in the server's
        /// stderr, `polls` reads `poll` apart.
        fn address(&mut self, err_log: &Path) -> Result<String, GateError> {
            for _ in 0..POLLS {
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
                std::thread::sleep(POLL);
            }
            Err(format!("the server did not listen within {POLLS} polls").into())
        }

        fn stop(&mut self) -> Result<String, GateError> {
            self.child.kill()?;
            Ok(format!("{}", self.child.wait()?))
        }
    }

    impl Drop for Served {
        fn drop(&mut self) {
            if matches!(self.child.try_wait(), Ok(None)) {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    /// The binary `name` beside this one.
    fn beside(name: &str) -> Result<PathBuf, GateError> {
        Ok(std::env::current_exe()?.with_file_name(name))
    }

    struct Args {
        arm: Arm,
        dir: PathBuf,
    }

    fn parse_args() -> Result<Args, GateError> {
        let (mut arm, mut dir) = (None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value: {USAGE}"))?;
            match flag.as_str() {
                "--arm" => arm = Some(Arm::parse(&v)?),
                "--dir" => dir = Some(PathBuf::from(v)),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        match (arm, dir) {
            (Some(arm), Some(dir)) => Ok(Args { arm, dir }),
            _ => Err(USAGE.into()),
        }
    }

    fn check(ok: &mut bool, name: &str, pass: bool) {
        println!("check {name}: {}", verdict(pass));
        *ok &= pass;
    }

    /// The lines of the server's stderr from line `from` on.
    fn lines_from(err_log: &Path, from: usize) -> Result<Vec<String>, GateError> {
        Ok(std::fs::read_to_string(err_log)?
            .lines()
            .skip(from)
            .map(str::to_owned)
            .collect())
    }

    /// `lines` of the server's stderr, read by the seat's record kinds.
    fn seat_log(lines: &[String]) -> Log {
        Log::of(&lines.join("\n"), record::BLOOMERY_SERVE_GLM)
    }

    /// The `group` of the one `load` record in `log`.
    fn load_group(log: &Log) -> Result<Option<u64>, GateError> {
        Ok(log.one(&record::LOAD_GENERATOR)?.opt_u64("group")?)
    }

    /// The value of `key=` in a seat's line that no record kind reads.
    fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
        line.split_whitespace()
            .find_map(|w| w.strip_prefix(key)?.strip_prefix('='))
    }

    /// The `residency pass` records of `log`: each pass's kind, the rows it
    /// kept and the flips that went live at its boundary.
    fn passes(log: &Log) -> Result<Vec<(String, u64, u64)>, GateError> {
        let mut out = Vec::new();
        for f in log.all(&record::RESIDENCY_PASS)? {
            out.push((f.word("pass")?.to_owned(), f.u64("kept")?, f.u64("landed")?));
        }
        Ok(out)
    }

    /// The tokens a run produced through the first pass whose boundary
    /// landed a flip (every token, when none did): `first` the tokens before
    /// the first pass's rows count — with the server's cut both runs' prompt
    /// calls give no token, their step token 0 — then each pass's kept rows,
    /// that pass's included — it ran before its flips went live.
    fn before_landing(passes: &[(String, u64, u64)], first: u64) -> u64 {
        let mut n = first;
        for (_, kept, landed) in passes {
            n += kept;
            if *landed > 0 {
                return n;
            }
        }
        u64::MAX
    }

    /// The prompt of the chat turn as the server renders and encodes it.
    fn chat_ids(url: &dyn Fn(&str) -> String) -> Result<Vec<u32>, GateError> {
        let messages = json!({"messages": [{"role": "user", "content": CHAT}]});
        let (st, body) = curl(&url("/apply-template"), Some(&messages), false)?;
        let text = json_of("/apply-template", st, &body)?["prompt"]
            .as_str()
            .ok_or("/apply-template: no prompt")?
            .to_owned();
        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({"content": text, "add_special": false})),
            false,
        )?;
        let ids = ids_of(&json_of("/tokenize", st, &body)?["tokens"]);
        println!("chat prompt {text:?} ids {ids:?}");
        if ids.is_empty() {
            return Err("the chat prompt tokenizes to no id".into());
        }
        Ok(ids)
    }

    /// One greedy `/completion` of `ids`, `n` tokens: the response.
    fn greedy(
        url: &dyn Fn(&str) -> String,
        ids: &[u32],
        n: usize,
        cache: bool,
    ) -> Result<Value, GateError> {
        let body = json!({
            "prompt": ids, "n_predict": n, "temperature": 0, "return_tokens": true,
            "cache_prompt": cache,
        });
        let (st, text) = curl(&url("/completion"), Some(&body), false)?;
        json_of("/completion", st, &text)
    }

    /// The chat turn at temperature 0, from a reset (`cache_prompt` off: a
    /// prefix kept at a checkpoint below the held sequence's end leaves the
    /// draft nothing to join, and the request would step plainly): its status
    /// and `timings`.
    fn chat(url: &dyn Fn(&str) -> String) -> Result<(u16, Value), GateError> {
        let body = json!({
            "messages": [{"role": "user", "content": CHAT}],
            "temperature": 0, "max_tokens": N_PREDICT, "cache_prompt": false,
        });
        let (st, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
        let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        println!(
            "chat HTTP {st}: message {} finish_reason {} usage {} timings {}",
            v["choices"][0]["message"], v["choices"][0]["finish_reason"], v["usage"], v["timings"]
        );
        Ok((st, v["timings"].clone()))
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

    /// `generate_glm5next --place gate --ctx 2048 --tokens <ids> -n 16`,
    /// with `--last-step` when `last_step` (the server's cut: the prompt less
    /// its last id, then a step), beside this binary under the arm's levers,
    /// stdout to `<dir>/gen.log` and stderr to `<dir>/gen.err`: its `tokens`
    /// line and its stdout read by its record kinds.
    fn cli(
        dir: &Path,
        levers: Levers,
        ids: &[u32],
        last_step: bool,
    ) -> Result<(Vec<u32>, Log), GateError> {
        let exe = beside("generate_glm5next")?;
        let tokens = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let out = dir.join("gen.log");
        let mut args = vec![
            "--place", "gate", "--ctx", "2048", "--tokens", &tokens, "-n", "16",
        ];
        if last_step {
            args.push("--last-step");
        }
        let status = Command::new(&exe)
            .args(&args)
            .envs(levers.iter().copied())
            .stdin(Stdio::null())
            .stdout(File::create(&out)?)
            .stderr(File::create(dir.join("gen.err"))?)
            .status()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        if !status.success() {
            return Err(format!(
                "{} exited {status}; see {}",
                exe.display(),
                dir.join("gen.err").display()
            )
            .into());
        }
        let lines: Vec<String> = std::fs::read_to_string(&out)?
            .lines()
            .map(str::to_owned)
            .collect();
        let line = lines
            .iter()
            .find_map(|l| l.strip_prefix("tokens "))
            .ok_or_else(|| format!("{}: no `tokens` line", out.display()))?;
        let got = parse_ids(line)?;
        let log =
            Log::of(&lines.join("\n"), record::GENERATE_GLM5NEXT).named(out.display().to_string());
        if let Some(s) = log.first(&record::MTP_SUMMARY)? {
            println!("generate_glm5next {}", s.line());
        }
        Ok((got, log))
    }

    /// `/props`' `engine` object: the draft it names, and the card's `draft`
    /// class, against `bytes`, this gate's NextN plan's bytes for the layer
    /// (`None`: no draft).
    fn props_draft(url: &dyn Fn(&str) -> String, bytes: Option<u64>) -> Result<bool, GateError> {
        let (st, body) = curl(&url("/props"), None, false)?;
        let e = json_of("/props", st, &body)?["engine"].clone();
        println!(
            "props engine draft {} placement {}",
            e["draft"], e["placement"]
        );
        let classes: Vec<Option<u64>> = e["placement"]["devices"]
            .as_array()
            .map(|d| {
                d.iter()
                    .filter(|d| d["device"].as_str().is_some_and(|n| n.starts_with("GPU")))
                    .map(|d| d["classes"]["draft"].as_u64())
                    .collect()
            })
            .unwrap_or_default();
        let path = ref_model_path()?.display().to_string();
        let mut ok = true;
        match bytes {
            None => check(
                &mut ok,
                "plain_props_carry_no_draft",
                e.get("draft").is_none()
                    && !classes.is_empty()
                    && classes.iter().all(Option::is_none),
            ),
            Some(b) => check(
                &mut ok,
                "drafted_props_name_the_nextn_draft",
                e["draft"]["kind"] == "mtp"
                    && e["draft"]["n_max"] == json!(1)
                    && e["draft"]["path"].as_str() == Some(path.as_str())
                    && classes == [Some(b)],
            ),
        }
        Ok(ok)
    }

    /// The bytes this gate's NextN plan of the file reserves for the layer
    /// on the gate card at [`CTX`]: its card terms and its arena.
    fn nextn_plan_bytes(levers: &bloomery_levers::Levers) -> Result<u64, GateError> {
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let nextn = NextnInputs::read(&inputs)?;
        let machine = workstation::plan_gate(inputs.model.layers);
        let plan = inputs.plan_nextn(
            &machine,
            u64::try_from(CTX)?,
            &PlanLevers::from_levers(levers)?,
            &nextn,
        )?;
        Ok(plan.nextn_card_bytes() + plan.arena_bytes)
    }

    /// `bloomery-serve --model glm --place <place> --plan` with both levers
    /// unset and no `--ctx`, so the default context rule runs: its exit
    /// status and stderr. It plans and exits before the load, so it opens no
    /// card.
    fn plan_only(dir: &Path, place: &str) -> Result<(bool, Vec<String>), GateError> {
        let exe = beside("bloomery-serve")?;
        let err = dir.join(format!("plan-{place}.err"));
        let status = Command::new(&exe)
            .args(["--model", "glm", "--place", place, "--plan"])
            .env_remove(bloomery_levers::DRAFT)
            .env_remove(bloomery_levers::RESIDENCY)
            .envs(UNSET.iter().copied())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(File::create(&err)?)
            .status()
            .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
        let lines = lines_from(&err, 0)?;
        println!("--place {place} --plan with the levers unset: {status}");
        for l in &lines {
            println!("  {l}");
        }
        Ok((status.success(), lines))
    }

    /// The unset rule's lines (the module header): `--place a` and `bp` draft
    /// and run the residency's default word, `--place gate` neither.
    fn unset_rule(dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let picks = |lines: &[String], draft: &str, residency: &str| -> Result<bool, GateError> {
            let log = seat_log(lines);
            let d = log.one(&record::DRAFT_UNSET_GLM)?;
            let r = log.one(&record::RESIDENCY_UNSET_GLM)?;
            let listening = log.first(&record::LISTENING_GLM)?;
            let host = log.first(&record::RESIDENCY_HOST)?;
            println!(
                "  read: draft unset {}, residency unset {}, residency host {}, listening {}",
                d.word("draft")?,
                r.word("residency")?,
                host.is_some(),
                listening.is_some()
            );
            Ok(d.word("draft")? == draft
                && r.word("residency")? == residency
                && listening.is_none()
                && host.is_some() == (residency != "off"))
        };
        let (a_ok, a) = plan_only(dir, "a")?;
        check(
            &mut ok,
            "unset_place_a_drafts_and_runs_mid_p0_s1",
            a_ok && picks(&a, "mtp", RESIDENCY_WORD)?,
        );
        let (bp_ok, bp) = plan_only(dir, "bp")?;
        check(
            &mut ok,
            "unset_place_bp_drafts_and_runs_mid_p0_s1",
            bp_ok && picks(&bp, "mtp", RESIDENCY_WORD)?,
        );
        let (g_ok, g) = plan_only(dir, "gate")?;
        check(
            &mut ok,
            "unset_place_gate_runs_neither",
            g_ok && picks(&g, "off", "off")?,
        );
        // Every placement's `ctx` line: a rule word of the default rule's,
        // its context at or past the floor while the card holds it (a card
        // that holds less plans fewer, `card`, as qwen38 does). FAIL-first:
        // a seat that keeps the floor silently, or prints no line, turns
        // this red.
        let ctx_line = |lines: &[String]| -> Option<(String, usize)> {
            let l = lines.iter().find(|l| l.starts_with("ctx rule="))?;
            Some((field(l, "rule")?.to_owned(), field(l, "ctx")?.parse().ok()?))
        };
        for (place, lines) in [("a", &a), ("bp", &bp), ("gate", &g)] {
            let named = ctx_line(lines).is_some_and(|(rule, ctx)| {
                matches!(rule.as_str(), "base" | "margin" | "fit" | "card") && ctx >= CTX
            });
            println!("--place {place}: a default ctx line, rule and context in {named}");
            check(&mut ok, "unset_plans_take_the_ctx_default", named);
        }
        // The default word's churn pool inside the plan's host terms: the
        // `residency host` record of a default load carries the word's
        // pinned count (0: no seed expert pinned, the pool every card expert
        // lands in) and the headroom left past it. FAIL-first: a default that
        // pins seed experts, or a pool past the headroom, turns this red.
        let mut pools = Vec::new();
        for lines in [&a, &bp] {
            let host = seat_log(lines).one(&record::RESIDENCY_HOST)?;
            pools.push((host.u64("pinned")?, host.i64("headroom_after")?));
        }
        println!("--place a and bp: residency host (pinned, headroom_after) {pools:?}");
        check(
            &mut ok,
            "unset_default_pool_is_p0_inside_the_headroom",
            pools
                .iter()
                .all(|&(pinned, after)| pinned == 0 && after >= 0),
        );
        // Every plan line names its stage card's free bytes at plan time, at
        // most its usable bytes — the census term the expert rule filled
        // within (memguard). FAIL-first: a plan line that drops it, or names
        // it past the card's usable bytes, turns this red.
        for (place, lines) in [("a", &a), ("bp", &bp), ("gate", &g)] {
            let plan = seat_log(lines).one(&record::PLAN)?;
            let free = plan.opt_u64("card_free")?;
            let usable = stage_usable(&plan)?;
            let named = free.is_some_and(|f| f <= usable);
            println!(
                "--place {place}: card_free named and within usable {named}: card_free={free:?}, \
                 usable {usable}"
            );
            check(&mut ok, "plan_names_the_cards_free_bytes", named);
        }
        Ok(ok)
    }

    /// The stage card's free bytes as a server's own `plan` record read
    /// them (the census reading its rule searched against). A server that
    /// printed no `plan` record or more than one, or a record without its
    /// `card_free`, is a named error, never an uncapped re-plan.
    fn server_card_free(err_log: &Path) -> Result<u64, GateError> {
        let free = server_log(err_log, record::BLOOMERY_SERVE_GLM)?
            .one(&record::PLAN)?
            .u64("card_free")?;
        println!("the server's plan record: card_free={free}");
        Ok(free)
    }

    /// The plan's stage-card expert bytes on the gate card at `ctx`, plain,
    /// as the server makes it; `None` when no plan takes the context.
    fn card_at(
        inputs: &PlanInputs,
        levers: &PlanLevers,
        ctx: usize,
        free: Option<u64>,
    ) -> Result<Option<u64>, GateError> {
        let mut machine = workstation::plan_gate(inputs.model.layers);
        machine.cards[0].free_bytes = free;
        Ok(inputs
            .plan(&machine, u64::try_from(ctx)?, levers)
            .ok()
            .and_then(|p| p.cards.first().map(|c| c.expert_bytes)))
    }

    /// The `ctx` clause (the module header): a plain server with no `--ctx`
    /// prints its `ctx` and `plan` lines before any load and is stopped
    /// there, against this gate's own plans of the file: its context is the
    /// rule's — [`CTX`] under `base`, the largest within the plan's margin
    /// under `margin`, the plan's largest under `fit`, the card's largest
    /// under `card`; the line's `trained` is the file's; its `fit` the
    /// largest context whose plan stands up to the trained context capped to
    /// `ORACLE_POSITIONS` — this gate's own plan taken, the one past it
    /// refused when the card bounds it, else the cap itself — with that
    /// plan's stage-card expert bytes; its `margin_ctx` at most
    /// `workstation::MARGIN` fewer of them than the plan at [`CTX`], and the
    /// next multiple of [`CTX_STEP`] past it more (or it is the fit). A
    /// server asked for [`FLAG_CTX`] takes it (below the floor, a value no
    /// default can coincide with); one asked past the oracle is refused by
    /// name before it listens. Every unset rule word must hold the relation
    /// it claims between the line's `ctx`, its `margin_ctx` and its `fit` —
    /// a default that keeps the floor while its margin reaches further
    /// names the wrong word. Mutants: the default taking the trained
    /// context past the fit (the server's own plan refuses it by name
    /// before any line); the search giving up (the line's `margin_ctx`
    /// keeps the floor while a step past it stays within it).
    fn ctx(dir: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let own = dir.join("ctx");
        std::fs::create_dir_all(&own)?;
        let mut args: Vec<&str> = SERVER_ARGS.to_vec();
        let ctx_at = args
            .iter()
            .position(|&f| f == "--ctx")
            .ok_or("SERVER_ARGS names no --ctx")?;
        args.drain(ctx_at..ctx_at + 2);
        let wait_for = |served: &mut Served,
                        err_log: &Path,
                        want: &[&str]|
         -> Result<Vec<String>, GateError> {
            for _ in 0..LINE_POLLS {
                let text = std::fs::read_to_string(err_log).unwrap_or_default();
                if want.iter().all(|w| text.lines().any(|l| l.starts_with(w))) {
                    return Ok(text.lines().map(str::to_owned).collect());
                }
                if let Some(status) = served.child.try_wait()? {
                    return Err(format!(
                        "the server exited ({status}) before its lines; {}:\n{text}",
                        err_log.display()
                    )
                    .into());
                }
                std::thread::sleep(LINE_POLL);
            }
            Err(format!(
                "the server printed no {want:?} within {LINE_POLLS} polls; {}",
                err_log.display()
            )
            .into())
        };

        // The default arm: the rule's own line, the server stopped before
        // its load. Each arm's server takes its own directory — `spawn`
        // writes `server.err` inside the one it is given.
        let default = own.join("default");
        std::fs::create_dir_all(&default)?;
        let err_log = default.join("server.err");
        let mut served = Served::spawn(&args, &default, PLAIN)?;
        println!("ctx server pid {}", served.child.id());
        let lines = wait_for(&mut served, &err_log, &["ctx rule=", "plan "])?;
        println!("ctx: the default server stopped: {}", served.stop()?);
        let line = lines
            .iter()
            .find(|l| l.starts_with("ctx rule="))
            .map(String::as_str)
            .ok_or("the default server printed no `ctx` line")?
            .to_owned();
        println!("ctx: {line}");
        let num = |k: &str| field(&line, k).and_then(|v| v.parse::<usize>().ok());
        let bytes = |k: &str| field(&line, k).and_then(|v| v.parse::<u64>().ok());
        let (rule, ctx, trained, fit) = (
            field(&line, "rule").map(str::to_owned),
            num("ctx"),
            num("trained"),
            num("fit"),
        );
        let (fit_bytes, margin_ctx) = (bytes("fit_card_expert_bytes"), num("margin_ctx"));
        let (Some(rule), Some(ctx), Some(trained), Some(fit), Some(fit_bytes), Some(margin_ctx)) =
            (rule, ctx, trained, fit, fit_bytes, margin_ctx)
        else {
            return Err(format!(
                "a `ctx` line without its rule, ctx, trained, fit, fit_card_expert_bytes and \
                 margin_ctx: {line}"
            )
            .into());
        };

        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let plan_levers = PlanLevers::from_levers(levers)?;
        let free = Some(server_card_free(&err_log)?);
        let at = |c: usize| card_at(&inputs, &plan_levers, c, free);
        let base = at(CTX)?.ok_or("no plan at the base context")?;
        let lost = |c: usize| -> Result<Option<u64>, GateError> {
            Ok(at(c)?.map(|e| base.saturating_sub(e)))
        };
        let margin = model::placement::workstation::MARGIN;
        let here = lost(margin_ctx)?;
        let past = lost(margin_ctx + CTX_STEP)?;
        let (at_fit, past_fit) = (at(fit)?, at(fit + 1)?);
        let cap = inputs
            .hp
            .n_ctx_train
            .min(usize::try_from(ORACLE_POSITIONS)?);
        let card_bounds = fit < cap;
        println!(
            "ctx: the gate's plans: lost at {margin_ctx} {here:?}, at {} {past:?}, fit {fit} \
             {at_fit:?}, past it {past_fit:?}; the cap {cap} ({})",
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
            match rule.as_str() {
                "set" => false,
                "base" => ctx == CTX && margin_ctx == CTX && CTX <= fit,
                "card" => ctx == fit && margin_ctx == fit && fit < CTX,
                "margin" => ctx == margin_ctx && CTX < margin_ctx && margin_ctx < fit,
                "fit" => ctx == fit && margin_ctx == fit && fit > CTX,
                _ => false,
            },
        );
        check(
            &mut ok,
            "ctx_line_prints_the_fit_and_the_margin",
            trained == inputs.hp.n_ctx_train
                && fit <= cap
                && at_fit == Some(fit_bytes)
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

        // The flag arm: below the floor, the flag wins.
        let flag_ctx = FLAG_CTX.to_string();
        let mut flag = args.clone();
        flag.extend_from_slice(&["--ctx", flag_ctx.as_str()]);
        let flagdir = own.join("flag");
        std::fs::create_dir_all(&flagdir)?;
        let err_log = flagdir.join("server.err");
        let mut served = Served::spawn(&flag, &flagdir, PLAIN)?;
        let lines = wait_for(&mut served, &err_log, &["ctx rule="])?;
        println!("ctx: the flag server stopped: {}", served.stop()?);
        let line = lines
            .iter()
            .find(|l| l.starts_with("ctx rule="))
            .map(String::as_str)
            .ok_or("the flag server printed no `ctx` line")?
            .to_owned();
        println!("ctx: {line}");
        check(
            &mut ok,
            "ctx_flag_wins",
            field(&line, "rule") == Some("set")
                && field(&line, "ctx").and_then(|v| v.parse::<usize>().ok()) == Some(FLAG_CTX),
        );

        // A context past the oracle: refused by name before it listens.
        let over_ctx = OVER_CTX.to_string();
        let err = own.join("oracle.err");
        let mut cmd = Command::new(beside("bloomery-serve")?);
        cmd.args([
            "--model",
            "glm",
            "--place",
            "gate",
            "--ctx",
            over_ctx.as_str(),
            "--parallel",
            "1",
            "--plan",
        ])
        .env_remove(bloomery_levers::DRAFT)
        .env_remove(bloomery_levers::RESIDENCY)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(File::create(&err)?);
        let status = cmd
            .status()
            .map_err(|e| format!("spawn bloomery-serve: {e}"))?;
        let text = std::fs::read_to_string(&err)?;
        let refused = text
            .lines()
            .find(|l| l.contains(&format!("ctx_max {OVER_CTX}")))
            .unwrap_or("")
            .to_owned();
        println!("ctx: --ctx {OVER_CTX} --plan: {status}; refused: {refused:?}");
        check(
            &mut ok,
            "ctx_past_the_oracle_is_refused_by_name",
            !status.success()
                && text.contains(&format!("ctx_max {OVER_CTX}"))
                && text.contains(&ORACLE_POSITIONS.to_string()),
        );
        Ok(ok)
    }

    /// The plain arm's drafted server, `BLOOMERY_DRAFT=mtp` with no
    /// residency (the module header): the prompt's greedy ids and the
    /// draft's counts.
    fn draft_only(dir: &Path, ids: &[u32], reference: &[u32]) -> Result<bool, GateError> {
        let dir = dir.join("mtp");
        std::fs::create_dir_all(&dir)?;
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&SERVER_ARGS, &dir, DRAFT_ONLY)?;
        println!("draft-only server pid {}", served.child.id());
        let addr = served.address(&err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = true;
        let load = server_log(&err_log, record::BLOOMERY_SERVE_GLM)?;
        let draft = load.first(&record::LOAD_DRAFT_GLM)?;
        let lever = load.one(&record::RESIDENCY_LEVER)?;
        let (word, why) = (lever.word("residency")?, lever.word("why")?);
        let host = load.first(&record::RESIDENCY_HOST)?;
        println!(
            "draft-only records: load draft {:?}, residency lever {word} why={why}, residency \
             host {:?}",
            draft.as_ref().map(record::Fields::line),
            host.as_ref().map(record::Fields::line)
        );
        check(
            &mut ok,
            "draft_only_loads_the_draft_and_no_residency",
            draft.is_some() && word == "off" && why == "set" && host.is_none(),
        );
        let c = greedy(&url, ids, N_PREDICT, false)?;
        let got = ids_of(&c["tokens"]);
        let stop = c["stop_type"].as_str().unwrap_or("").to_owned();
        let t = &c["timings"];
        println!(
            "draft-only completion tokens {got:?} stop_type={stop} draft_n={} draft_n_accepted={}",
            t["draft_n"], t["draft_n_accepted"]
        );
        check(
            &mut ok,
            "draft_only_carries_the_draft_counts",
            t["draft_n"].as_u64().is_some_and(|n| n > 0)
                && t["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
        );
        check(
            &mut ok,
            "draft_only_ids_are_the_plain_generate_glm5next",
            agree(&got, &stop, reference),
        );
        let top1 = json!({
            "prompt": ids, "n_predict": N_PREDICT, "temperature": 0.8, "top_k": 1, "seed": 42,
            "return_tokens": true, "cache_prompt": false,
        });
        let (st, text) = curl(&url("/completion"), Some(&top1), false)?;
        let k1 = json_of("/completion", st, &text)?;
        let sampled = ids_of(&k1["tokens"]);
        let k1_stop = k1["stop_type"].as_str().unwrap_or("").to_owned();
        println!(
            "draft-only top_k 1 sample tokens {sampled:?} stop_type={k1_stop} draft_n={}",
            k1["timings"]["draft_n"]
        );
        check(
            &mut ok,
            "draft_only_top1_sample_is_the_plain_generate_glm5next",
            k1["timings"].get("draft_n").is_none() && agree(&sampled, &k1_stop, reference),
        );
        ok &= cache(&url, &err_log, true)?;
        println!("draft-only server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The ids of `messages` as the server's chat template renders them
    /// (`/apply-template`, then `/tokenize` without BOS).
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

    /// `/tokenize` of `text` without BOS.
    fn tokenized(url: &dyn Fn(&str) -> String, text: &str) -> Result<Vec<u32>, GateError> {
        let (st, body) = curl(
            &url("/tokenize"),
            Some(&json!({ "content": text, "add_special": false })),
            false,
        )?;
        Ok(ids_of(&json_of("/tokenize", st, &body)?["tokens"]))
    }

    /// The slot `slot` dropped and not saved (`POST
    /// /slots/<slot>?action=erase`): the prompt cache keeps what it held, and
    /// nothing more.
    fn erase(url: &dyn Fn(&str) -> String, slot: usize) -> Result<(), GateError> {
        let at = format!("/slots/{slot}?action=erase");
        let (st, body) = curl(&url(&at), Some(&json!({})), false)?;
        json_of(&at, st, &body)?;
        Ok(())
    }

    /// One greedy request of the `cache` clause as the clause reads it: its
    /// ids, `cache_n`, the draft's counts and stop, and the server's lines it
    /// printed (`cache` notes, `mtp prompt` records, `residency pass`
    /// records).
    struct Turn {
        tokens: Vec<u32>,
        cache_n: u64,
        draft_n: u64,
        accepted: u64,
        stop: String,
        lines: Vec<String>,
    }

    impl Turn {
        fn run(
            url: &dyn Fn(&str) -> String,
            err_log: &Path,
            ids: &[u32],
            cache: bool,
        ) -> Result<Turn, GateError> {
            let from = lines_from(err_log, 0)?.len();
            let c = greedy(url, ids, N_PREDICT, cache)?;
            let t = &c["timings"];
            Ok(Turn {
                tokens: ids_of(&c["tokens"]),
                cache_n: t["cache_n"].as_u64().unwrap_or(u64::MAX),
                draft_n: t["draft_n"].as_u64().unwrap_or(0),
                accepted: t["draft_n_accepted"].as_u64().unwrap_or(0),
                stop: c["stop_type"].as_str().unwrap_or("").to_owned(),
                lines: lines_from(err_log, from)?,
            })
        }

        fn show(&self, what: &str) {
            let picked: Vec<&String> = self
                .lines
                .iter()
                .filter(|l| l.contains(" cache ") || l.starts_with("mtp prompt "))
                .collect();
            println!(
                "{what}: tokens {:?} stop={} cache_n={} draft_n={} accepted={} lines {picked:?}",
                self.tokens, self.stop, self.cache_n, self.draft_n, self.accepted
            );
        }

        /// The `cache load` note this request printed: a state put back.
        fn loaded(&self) -> Option<&str> {
            self.lines
                .iter()
                .find(|l| l.contains(": cache load "))
                .map(String::as_str)
        }

        /// The `mtp prompt` records this request printed that say the draft
        /// skips.
        fn skips(&self) -> Result<usize, GateError> {
            skips(&seat_log(&self.lines))
        }
    }

    /// The `cache` clause on one server (`drafted`: its MTP draft runs; the
    /// residency off, so a sequence's bits are its own): session A's first
    /// turn fed fresh, B's turn, then A again — its state saved when B took
    /// the slot (`glm5next::seq_save`) and put back (`seq_resume`) — against
    /// the same requests with no switch between, run first:
    /// - (a) A's turn, its reply and a later text: it keeps every position A
    ///   held (a `cache load` note, `cache_n` the run with no switch's), and
    ///   its ids are that run's; under the draft the draft rejoins where A
    ///   left it — no `mtp prompt` record says it skips (the parked rows and
    ///   the NextN store's positions came back), it drafts and accepts, and
    ///   its counts are that run's: counts, which hold the store's and the
    ///   arena rows' bits only as far as a moved proposal moves an accept;
    /// - (b) A's turn with another text in place of its reply, so the shared
    ///   prefix ends at the turn: it keeps the checkpoint at the turn's
    ///   prompt call's end (the point the state carried), and its ids are the
    ///   run with no switch's, which cut to the model's own checkpoint there.
    ///
    /// Mutants: the state carrying no point ((b) keeps nothing); the KDA
    /// stores zeroed after the put-back ((a)'s ids move); the latent rows
    /// zeroed after it ((a)'s and (b)'s ids move); the seat's rule granting
    /// nothing of a saved state (no `cache load`: (a) and (b) keep nothing);
    /// under the draft its side not put back (the draft skips by name).
    fn cache(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        drafted: bool,
    ) -> Result<bool, GateError> {
        let label = if drafted { "drafted" } else { "plain" };
        let p1 = rendered(url, json!([{ "role": "user", "content": TURN_A }]))?;
        let pb = rendered(url, json!([{ "role": "user", "content": TURN_B }]))?;
        let later = tokenized(url, LATER_A)?;
        let stripped = tokenized(url, STRIPPED_A)?;
        let with = |a: &[u32], b: &[u32], c: &[u32]| -> Vec<u32> {
            a.iter().chain(b).chain(c).copied().collect()
        };
        let mut ok = true;

        // (a) with no switch, then the slot dropped unsaved, so no cached
        // state holds A when the switched run asks for it.
        let r1 = Turn::run(url, err_log, &p1, false)?;
        r1.show(&format!("cache {label}: A's turn fresh"));
        if r1.stop != "limit" || r1.tokens.len() != N_PREDICT {
            return Err(format!(
                "the cache clause's A turn stopped at {} tokens ({}): the clause needs {N_PREDICT}",
                r1.tokens.len(),
                r1.stop
            )
            .into());
        }
        let resend = with(&p1, &r1.tokens, &later);
        let r2 = Turn::run(url, err_log, &resend, true)?;
        r2.show(&format!("cache {label}: A resent, no switch"));
        erase(url, 0)?;
        let a1 = Turn::run(url, err_log, &p1, false)?;
        a1.show(&format!("cache {label}: A's turn fresh again"));
        let b1 = Turn::run(url, err_log, &pb, true)?;
        b1.show(&format!("cache {label}: B's turn"));
        let a2 = Turn::run(url, err_log, &resend, true)?;
        a2.show(&format!("cache {label}: A resent after B"));
        let least = (p1.len() + N_PREDICT - 1) as u64;
        check(
            &mut ok,
            &format!("cache_{label}_a_keeps_every_held_position_after_the_switch"),
            a1.tokens == r1.tokens
                && !b1.tokens.is_empty()
                && a2.loaded().is_some()
                && r2.loaded().is_none()
                && a2.cache_n == r2.cache_n
                && a2.cache_n >= least,
        );
        check(
            &mut ok,
            &format!("cache_{label}_a_ids_are_the_run_with_no_switch"),
            !a2.tokens.is_empty() && a2.tokens == r2.tokens,
        );
        if drafted {
            let (a2_skips, r2_skips) = (a2.skips()?, r2.skips()?);
            println!(
                "cache {label}: draft skips A resent after B {a2_skips}, no switch {r2_skips}"
            );
            check(
                &mut ok,
                "cache_drafted_a_draft_rejoins_where_a_left_it",
                a2_skips == 0
                    && r2_skips == 0
                    && a2.draft_n > 0
                    && a2.accepted > 0
                    && (a2.draft_n, a2.accepted) == (r2.draft_n, r2.accepted),
            );
        }

        // (b): the shared prefix ends at the turn, a checkpoint the prompt
        // call took at its end (the turn less its last id, which the first
        // step feeds).
        if stripped.first() == r1.tokens.first() {
            return Err(format!(
                "the replacement text's first id {:?} is the reply's: the shared prefix would not \
                 end at the turn",
                stripped.first()
            )
            .into());
        }
        let strip = with(&p1, &stripped, &[]);
        erase(url, 0)?;
        let c1 = Turn::run(url, err_log, &p1, false)?;
        let s1 = Turn::run(url, err_log, &strip, true)?;
        s1.show(&format!("cache {label}: A's turn replaced, no switch"));
        erase(url, 0)?;
        let c2 = Turn::run(url, err_log, &p1, false)?;
        let b2 = Turn::run(url, err_log, &pb, true)?;
        let s2 = Turn::run(url, err_log, &strip, true)?;
        s2.show(&format!("cache {label}: A's turn replaced after B"));
        let turn_end = p1.len() as u64 - 1;
        check(
            &mut ok,
            &format!("cache_{label}_b_keeps_the_turns_end"),
            c1.tokens == r1.tokens
                && c2.tokens == r1.tokens
                && !b2.tokens.is_empty()
                && s2.loaded().is_some()
                && s1.cache_n == turn_end
                && s2.cache_n == turn_end,
        );
        check(
            &mut ok,
            &format!("cache_{label}_b_ids_are_the_run_with_no_switch"),
            !s2.tokens.is_empty() && s2.tokens == s1.tokens,
        );
        Ok(ok)
    }

    /// The `round_trip` clause on the plain server: a session as a chat
    /// client runs it. Its first turn is a chat request whose reply, the
    /// think span open past its prompt's length, is longer than its prompt;
    /// its second turn is the chat messages the client sends back — the
    /// reply's `content` and no `reasoning_content` — which the template
    /// renders with the think span empty, so the second turn's prompt parts
    /// from what the slot holds at the reply's first id. It keeps the first
    /// turn's prompt call's end (the checkpoint there: `cache_n` the first
    /// prompt less its last id), less than half of what the slot holds, and
    /// the slot's state is not saved (no `cache save` line): the second turn
    /// carries the slot's last prompt whole and drops only that prompt's
    /// reply, which the client never sends back. Mutant: the cache's half
    /// rule alone (llama-server's `f_keep < 0.5`), which saves the state
    /// here.
    fn round_trip(url: &dyn Fn(&str) -> String, err_log: &Path) -> Result<bool, GateError> {
        erase(url, 0)?;
        let first = json!([{ "role": "user", "content": TURN_C }]);
        let p1 = rendered(url, first.clone())?;
        // Past the prompt less one, so the slot then holds more than twice
        // what the second turn keeps.
        let reply = p1.len() + 8;
        let body = json!({ "messages": first, "temperature": 0, "max_tokens": reply });
        let (st, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
        let r1 = json_of("/v1/chat/completions", st, &text)?;
        let m1 = &r1["choices"][0]["message"];
        let finish = r1["choices"][0]["finish_reason"].as_str().unwrap_or("");
        let reasoning = m1["reasoning_content"].as_str().unwrap_or("");
        let content = m1["content"].as_str().unwrap_or("").to_owned();
        println!(
            "round trip: a first turn of {} ids, a reply of {reply} ({finish}): reasoning \
             {reasoning:?} content {content:?} timings {}",
            p1.len(),
            r1["timings"]
        );
        if finish != "length" || reasoning.is_empty() {
            return Err(format!(
                "the round trip's first reply ended {finish:?} with reasoning {reasoning:?}: the \
                 clause needs {reply} ids that open with the think span"
            )
            .into());
        }
        let mut messages = vec![first[0].clone()];
        messages.push(json!({ "role": "assistant", "content": content }));
        messages.push(json!({ "role": "user", "content": LATER_C }));
        let messages = Value::Array(messages);
        let p2 = rendered(url, messages.clone())?;
        if !p2.starts_with(&p1) {
            return Err(format!(
                "the second turn's {} ids do not carry the first turn's {} whole",
                p2.len(),
                p1.len()
            )
            .into());
        }
        let from = lines_from(err_log, 0)?.len();
        let body = json!({ "messages": messages, "temperature": 0, "max_tokens": 4 });
        let (st, text) = curl(&url("/v1/chat/completions"), Some(&body), false)?;
        let t = json_of("/v1/chat/completions", st, &text)?["timings"].clone();
        let saves: Vec<String> = lines_from(err_log, from)?
            .into_iter()
            .filter(|l| l.contains(": cache save "))
            .collect();
        let want = p1.len() as u64 - 1;
        println!(
            "round trip: a second turn of {} ids, the slot holding {}: cache_n={} (want {want}) \
             cache_ms={} saves {saves:?}",
            p2.len(),
            p1.len() + reply - 1,
            t["cache_n"],
            t["cache_ms"]
        );
        let mut ok = true;
        check(
            &mut ok,
            "round_trip_keeps_the_prompt_calls_end_and_saves_nothing",
            t["cache_n"].as_u64() == Some(want) && saves.is_empty(),
        );
        Ok(ok)
    }

    /// Under the residency, a state put back runs on the slot map as it
    /// stands (the state carries none): A's turn, B's, A resent — put back (a
    /// `cache load` note), every held position kept, its passes printing
    /// their `residency pass` records, and the draft rejoining (no `mtp
    /// prompt` record says it skips, and it drafts). Its ids are printed, not
    /// held: B's passes land flips, and a flip moves an expert between the
    /// host and the card, whose sums round another way. Mutant: the draft's
    /// side not put back (the draft skips by name).
    fn resume_under_residency(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
    ) -> Result<bool, GateError> {
        let p1 = rendered(url, json!([{ "role": "user", "content": TURN_A }]))?;
        let pb = rendered(url, json!([{ "role": "user", "content": TURN_B }]))?;
        let later = tokenized(url, LATER_A)?;
        let a1 = Turn::run(url, err_log, &p1, false)?;
        a1.show("residency resume: A's turn fresh");
        let b1 = Turn::run(url, err_log, &pb, true)?;
        b1.show("residency resume: B's turn");
        let resend: Vec<u32> = p1.iter().chain(&a1.tokens).chain(&later).copied().collect();
        let a2 = Turn::run(url, err_log, &resend, true)?;
        a2.show("residency resume: A resent after B");
        let a2_passes = passes(&seat_log(&a2.lines))?;
        let a2_skips = a2.skips()?;
        println!(
            "residency resume: A resent's passes (kind, kept, landed) {a2_passes:?}, draft skips \
             {a2_skips}"
        );
        let mut ok = true;
        check(
            &mut ok,
            "drafted_resume_runs_on_the_map_as_it_stands",
            a1.tokens.len() == N_PREDICT
                && !b1.tokens.is_empty()
                && a2.loaded().is_some()
                && a2.cache_n >= (p1.len() + N_PREDICT - 1) as u64
                && !a2.tokens.is_empty()
                && !a2_passes.is_empty()
                && a2_skips == 0
                && a2.draft_n > 0,
        );
        Ok(ok)
    }

    /// The `mtp prompt` records of `log` that say the draft skips (their
    /// `skipped` not `none`).
    fn skips(log: &Log) -> Result<usize, GateError> {
        let mut n = 0;
        for f in log.all(&record::MTP_PROMPT)? {
            n += usize::from(f.text("skipped")? != "none");
        }
        Ok(n)
    }

    /// (s4) the context split's refusals (module header), each a `--plan` run
    /// under the draft-only levers that must end before the plan: a `--ctx`
    /// whose split leaves a slot fewer positions than the draft's window, and
    /// `--parallel 0`.
    fn split_refused(dir: &Path) -> Result<bool, GateError> {
        let mut ok = true;
        let exe = beside("bloomery-serve")?;
        for (name, flags, want) in [
            (
                "split_under_the_window_is_refused_by_name",
                &["--ctx", "4", "--parallel", "2"][..],
                "--ctx 4: --parallel 2 splits it to 2 positions a slot, under the 3 a slot needs",
            ),
            (
                "parallel_0_is_refused_by_name",
                &["--parallel", "0"][..],
                "--parallel 0: the server serves no slot",
            ),
        ] {
            let err = dir.join(format!("{name}.err"));
            let status = Command::new(&exe)
                .args(["--model", "glm", "--place", "gate"])
                .args(flags)
                .arg("--plan")
                .env_remove(bloomery_levers::DRAFT)
                .env_remove(bloomery_levers::RESIDENCY)
                .envs(DRAFT_ONLY.iter().copied())
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(File::create(&err)?)
                .status()
                .map_err(|e| format!("spawn {}: {e}", exe.display()))?;
            let lines = lines_from(&err, 0)?;
            let named = lines.iter().any(|l| l.contains(want));
            // The `ctx` line no record kind reads; the plan through its own.
            let planned = lines.iter().any(|l| l.starts_with("ctx rule="))
                || seat_log(&lines).first(&record::PLAN)?.is_some();
            println!(
                "slots: {} --plan: {status}; named {named}, planned {planned}",
                flags.join(" ")
            );
            check(&mut ok, name, !status.success() && named && !planned);
        }
        Ok(ok)
    }

    /// (s5) the slots clause (module header) on a server of two resident
    /// slots started into `<dir>/slots`, the draft on, no residency and the
    /// rounds counted: two prompts alone, then both at once.
    fn slots_flow_together(dir: &Path) -> Result<bool, GateError> {
        let dir = dir.join("slots");
        std::fs::create_dir_all(&dir)?;
        // SERVER_ARGS ends in the one-slot `--parallel 1`; this clause's
        // server takes two slots of the same total context.
        let mut args: Vec<&str> = SERVER_ARGS.to_vec();
        let parallel = args.len() - 1;
        args[parallel] = "2";
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&args, &dir, DRAFT_ONLY_STATS)?;
        println!("slots server pid {}", served.child.id());
        let addr = served.address(&err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = true;
        let listening = server_log(&err_log, record::BLOOMERY_SERVE_GLM)?
            .one(&record::LISTENING_GLM)?
            .u64("ctx")?;
        let (st, text) = curl(&url("/props"), None, false)?;
        let props = json_of("/props", st, &text)?;
        let (n_ctx, slots) = (props["n_ctx"].as_u64(), props["total_slots"].as_u64());
        // The stage card's KV term a plan of one and of two sequences counts
        // at a slot's context (`PlanInputs::kv_term`, the next-token layer's
        // store in its own plan, `/props`' `draft` class): the trunk's stores
        // for each, the bytes beside them for the second.
        let path = ref_model_path()?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let nextn = NextnInputs::read(&inputs)?;
        let half = u64::try_from(CTX / 2)?;
        let trunk = 0..inputs.hp.n_trunk;
        let kv = |n| inputs.kv_term(trunk.clone(), half, KdaLanes::Two, Some(&nextn), n);
        let (one, two) = (kv(1)?, kv(2)?);
        let vram = props["engine"]["placement"]["vram_kv_bytes"].as_u64();
        println!(
            "slots: listening ctx {listening}; /props total_slots {slots:?}, n_ctx {n_ctx:?}, \
             vram_kv_bytes {vram:?} (the plan's KV term of two sequences {two}, of one {one})"
        );
        check(
            &mut ok,
            "slots_split_the_context",
            slots == Some(2) && listening == half && n_ctx == Some(half),
        );
        check(
            &mut ok,
            "slots_props_count_every_sequence",
            vram == Some(two),
        );
        // The `parallel` line printed before the load names the round's
        // shape the open chose: one pass, the body's pass of several slots
        // holding both slots' drafted windows.
        let parallel_line = std::fs::read_to_string(&err_log)?
            .lines()
            .find(|l| l.starts_with("parallel "))
            .ok_or("the slots server printed no parallel line")?
            .to_owned();
        println!("slots: {parallel_line}");
        check(
            &mut ok,
            "slots_parallel_line_names_one_pass",
            field(&parallel_line, "pass") == Some("one"),
        );
        // A request's ids and the draft's counts.
        let run = |v: &Value| {
            (
                ids_of(&v["tokens"]),
                v["timings"]["draft_n"].as_u64().unwrap_or(0),
                v["timings"]["draft_n_accepted"].as_u64().unwrap_or(0),
            )
        };
        let a_ids = rendered(&url, json!([{ "role": "user", "content": TURN_A }]))?;
        let c_ids = rendered(&url, json!([{ "role": "user", "content": TURN_C }]))?;
        let (mut alone, mut stops) = (Vec::new(), Vec::new());
        for ids in [&a_ids, &c_ids] {
            let v = greedy(&url, ids, SLOTS_PREDICT, false)?;
            stops.push(v["stop_type"].as_str().unwrap_or("").to_owned());
            alone.push(run(&v));
        }
        let busy = |decodes: f64| -> Result<f64, GateError> {
            Ok(metric(&url, "n_busy_slots_per_decode")?.unwrap_or(f64::NAN) * decodes)
        };
        let decode0 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy0 = busy(decode0)?;
        // The alone runs print no record (a round of one busy slot is a
        // select and a pass), so the window opens empty; taken anyway, so the
        // together run's records are exactly the ones after this point.
        let before = round_count(&err_log)?;
        let together: Vec<Option<(Vec<u32>, u64, u64)>> = both_at_once(
            &url,
            [greedy_body(&a_ids), greedy_body(&c_ids)],
            ["slots A", "slots C"],
        )?
        .iter()
        .map(|v| v.as_ref().map(run))
        .collect();
        let decode1 = metric(&url, "n_decode_total")?.unwrap_or(f64::NAN);
        let busy1 = busy(decode1)?;
        let swaps = metric(&url, "swaps_total")?;
        let ratio = (busy1 - busy0) / (decode1 - decode0);
        let count = |r: &(Vec<u32>, u64, u64)| (r.0.len(), r.1, r.2);
        println!(
            "slots alone (ids, drafts, accepted) {:?} and {:?}, stop types {stops:?}; together \
             {:?} and {:?}; decode {decode0} -> {decode1}, busy {busy0} -> {busy1}, busy a call \
             {ratio:.3}, swaps {swaps:?}",
            count(&alone[0]),
            count(&alone[1]),
            together[0].as_ref().map(count),
            together[1].as_ref().map(count),
        );
        check(
            &mut ok,
            "slots_alone_runs_are_whole_and_draft",
            alone.iter().all(|a| a.0.len() == SLOTS_PREDICT && a.1 > 0),
        );
        check(
            &mut ok,
            "slots_together_ids_and_draft_counts_are_alone",
            together[0].as_ref() == Some(&alone[0]) && together[1].as_ref() == Some(&alone[1]),
        );
        check(
            &mut ok,
            "slots_rounds_carry_both_slots",
            swaps.is_none_or(|v| v == 0.0) && ratio >= SLOTS_RATIO,
        );
        // The together window's drafted rounds (`cmd=pass`): a step round of
        // two rows (both slots at their prompt's last id) keeps the fallback
        // loop on a NextN load, the one-pass step round being the plain
        // load's, and a pass round of one row (the other slot stepping) is
        // one pass in either shape, so the clause reads the pass rounds and
        // needs one of two rows among them.
        let window = rounds_from(&err_log, before)?;
        let pass_rounds = of_cmd(&window, "pass");
        let one_pass = pass_rounds.iter().any(|r| r.u64("rows") == Ok(2))
            && pass_rounds
                .iter()
                .all(|r| r.u64("slots") == Ok(2) && r.u64("passes") == Ok(1));
        print_rounds("slots", &window, &pass_rounds);
        check(&mut ok, "slots_pass_rounds_run_one_pass", one_pass);
        ok &= slots_sampled(&url, &err_log, [&a_ids, &c_ids], &alone[0])?;
        ok &= slots_skipping(&url, &err_log, [&a_ids, &c_ids])?;
        println!("slots server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The (s5) server's sampled clauses (the module header): two sampled
    /// requests ([`sampled_body`], [`SEEDS`]) alone, then both at once, then
    /// the greedy `ids[0]` beside the sampled `ids[1]`, `greedy_alone` the
    /// greedy request's alone run (its ids, draft count and accepted).
    ///
    /// A sampled request steps on the server — a `next` a token, its row
    /// read for the sampler — and its round of steps is the server's call
    /// apart from the drafted passes' (`Engine::step_slots` beside
    /// `Engine::advance_slots`). The default seat runs neither as one pass
    /// with the drafted rows: a sampled row would be a window that proposes
    /// nothing, which `app::mtp::pass_slots` refuses (depth 0, and every
    /// window's draft proposes), the body's NextN pass of several slots
    /// refuses (a slot of one row, `Body::refuse_slots`), and no call of the
    /// server carries (the drafted row lends no logits row, `SlotPassRow`).
    /// So the clauses pin today's shape: two sampled slots' round is a
    /// select and a step a row (`cmd=step`, `passes=rows`), and a round of a
    /// greedy and a sampled slot is two calls — the sampled row's step and
    /// the greedy row's pass of one window (`cmd=pass`, `rows=1`).
    fn slots_sampled(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        ids: [&[u32]; 2],
        greedy_alone: &(Vec<u32>, u64, u64),
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let mut alone = Vec::new();
        for (ids, seed) in ids.iter().zip(SEEDS) {
            let (st, text) = curl(&url("/completion"), Some(&sampled_body(ids, seed)), false)?;
            alone.push(json_of("/completion", st, &text)?);
        }
        let before = round_count(err_log)?;
        let together = both_at_once(
            url,
            [
                sampled_body(ids[0], SEEDS[0]),
                sampled_body(ids[1], SEEDS[1]),
            ],
            ["sampled A", "sampled C"],
        )?;
        let window = rounds_from(err_log, before)?;
        let steps = of_cmd(&window, "step");
        let ids_of_v = |v: &Value| ids_of(&v["tokens"]);
        let plain = |v: &Value| v["timings"].get("draft_n").is_none();
        println!(
            "slots sampled alone {:?} and {:?} ids, together {:?} and {:?}",
            ids_of_v(&alone[0]).len(),
            ids_of_v(&alone[1]).len(),
            together[0].as_ref().map(|v| ids_of_v(v).len()),
            together[1].as_ref().map(|v| ids_of_v(v).len()),
        );
        print_rounds("slots sampled", &window, &steps);
        check(
            &mut ok,
            "slots_sampled_together_ids_are_alone",
            alone.iter().all(|v| !ids_of_v(v).is_empty() && plain(v))
                && together.iter().zip(&alone).all(|(t, a)| {
                    t.as_ref()
                        .is_some_and(|t| ids_of_v(t) == ids_of_v(a) && plain(t))
                }),
        );
        check(
            &mut ok,
            "slots_sampled_rounds_step_in_turn",
            steps.iter().any(|r| r.u64("rows") == Ok(2))
                && steps.iter().all(|r| {
                    r.u64("slots") == Ok(2) && r.u64("passes").is_ok_and(|p| r.u64("rows") == Ok(p))
                })
                && of_cmd(&window, "pass").is_empty(),
        );

        let before = round_count(err_log)?;
        let mixed = both_at_once(
            url,
            [greedy_body(ids[0]), sampled_body(ids[1], SEEDS[1])],
            ["mixed greedy A", "mixed sampled C"],
        )?;
        let window = rounds_from(err_log, before)?;
        let passes = of_cmd(&window, "pass");
        let counts = |v: &Value| {
            (
                ids_of_v(v),
                v["timings"]["draft_n"].as_u64().unwrap_or(0),
                v["timings"]["draft_n_accepted"].as_u64().unwrap_or(0),
            )
        };
        let greedy_mixed = mixed[0].as_ref().map(counts);
        println!(
            "slots mixed greedy (ids, drafts, accepted) {:?} against alone {:?}; sampled {:?} ids",
            greedy_mixed.as_ref().map(|c| (c.0.len(), c.1, c.2)),
            (greedy_alone.0.len(), greedy_alone.1, greedy_alone.2),
            mixed[1].as_ref().map(|v| ids_of_v(v).len()),
        );
        print_rounds("slots mixed", &window, &passes);
        check(
            &mut ok,
            "slots_mixed_together_ids_are_alone",
            greedy_mixed.as_ref() == Some(greedy_alone)
                && mixed[1]
                    .as_ref()
                    .is_some_and(|v| ids_of_v(v) == ids_of_v(&alone[1]) && plain(v)),
        );
        check(
            &mut ok,
            "slots_mixed_rounds_run_apart",
            !passes.is_empty()
                && passes.iter().all(|r| {
                    r.u64("slots") == Ok(2) && r.u64("rows") == Ok(1) && r.u64("passes") == Ok(1)
                }),
        );
        Ok(ok)
    }

    /// The (s5) skipping clause (the module header): a slot whose draft skips
    /// steps alone beside a drafting one. `ids[0]` greedy leaves its slot
    /// holding the turn and its reply, the draft's waiting rows at the
    /// reply's end; the strip — that prompt with [`STRIPPED_A`] in place of
    /// the reply, `cache_prompt` on — keeps the checkpoint the turn's prompt
    /// call took at its end and its prompt call starts there, where the
    /// waiting rows do not end, so the draft skips and the request steps
    /// plainly. The strip alone first; then both slots dropped (a dropped
    /// slot saves no state the together strip could take back in place of
    /// the cut), the history built again, the second turn `ids[1]` run alone
    /// on the other slot (`cache_prompt` off: a resend of it with the cache
    /// on would cut its own held end the same way and skip too), and the
    /// strip beside that turn posted at once: the seat turns the skipping
    /// slot off before each one-pass round, the round running its plain pass
    /// beside the drafting slot's window, and each request's ids — the
    /// drafting one's counts too — are its alone run's. Mutant: the seat not
    /// turning a skipping slot off — the slot joins the pass as its one
    /// plain row, which the body's NextN pass of several slots refuses by
    /// name, the server dies and this clause goes red.
    fn slots_skipping(
        url: &dyn Fn(&str) -> String,
        err_log: &Path,
        ids: [&[u32]; 2],
    ) -> Result<bool, GateError> {
        let mut ok = true;
        let stripped = tokenized(url, STRIPPED_A)?;
        let strip: Vec<u32> = ids[0].iter().chain(&stripped).copied().collect();
        // The fixture: the first turn's reply greedy to the limit, and the
        // replacement text parting at the reply, so the strip shares exactly
        // the turn and cuts to its prompt call's end.
        let first = greedy(url, ids[0], SLOTS_PREDICT, false)?;
        let reply = ids_of(&first["tokens"]);
        let stopped = first["stop_type"].as_str().unwrap_or("");
        if stopped != "limit" || reply.len() != SLOTS_PREDICT {
            return Err(format!(
                "the skipping clause's first turn stopped at {} tokens ({stopped}): the clause \
                 needs {SLOTS_PREDICT}",
                reply.len()
            )
            .into());
        }
        if stripped.first() == reply.first() {
            return Err(format!(
                "the replacement text's first id {:?} is the reply's: the strip would carry the \
                 reply's start, not part at the turn",
                stripped.first()
            )
            .into());
        }
        let turn_end = ids[0].len() as u64 - 1;
        let plain = |v: &Value| v["timings"].get("draft_n").is_none();
        let cache_n = |v: &Value| v["timings"]["cache_n"].as_u64();
        let run_of = |v: &Value| {
            (
                ids_of(&v["tokens"]),
                v["timings"]["draft_n"].as_u64().unwrap_or(0),
                v["timings"]["draft_n_accepted"].as_u64().unwrap_or(0),
            )
        };
        // The `mtp prompt` records of `lines` that say the draft skips, their
        // `skipped` texts.
        let skips_of = |lines: &[String]| -> Result<Vec<String>, GateError> {
            seat_log(lines)
                .all(&record::MTP_PROMPT)?
                .iter()
                .filter(|f| f.text("skipped").is_ok_and(|s| s != "none"))
                .map(|f| Ok(f.text("skipped")?.to_owned()))
                .collect()
        };
        // The strip alone on that history: the checkpoint the turn's prompt
        // call took at its end kept, the draft skipping from its prompt call
        // (its rounds one request's, a select and a plain pass, printing no
        // round record and no turn-off record).
        let from = lines_from(err_log, 0)?.len();
        let alone = greedy(url, &strip, SLOTS_PREDICT, true)?;
        let alone_ids = ids_of(&alone["tokens"]);
        let alone_skips = skips_of(&lines_from(err_log, from)?)?;
        println!(
            "slots skipping: the strip alone {} ids ({}) cache_n {:?} (the turn's end \
             {turn_end}) draft_n {}, skips {alone_skips:?}",
            alone_ids.len(),
            alone["stop_type"].as_str().unwrap_or(""),
            cache_n(&alone),
            alone["timings"]["draft_n"],
        );
        // The history again and the fresh drafting turn alone; then both at
        // once, the strip's slot the one holding the history (the prefix it
        // shares) and the turn's the other.
        erase(url, 0)?;
        erase(url, 1)?;
        greedy(url, ids[0], SLOTS_PREDICT, false)?;
        let turn_alone = run_of(&greedy(url, ids[1], SLOTS_PREDICT, false)?);
        let before = round_count(err_log)?;
        let from = lines_from(err_log, 0)?.len();
        let together = both_at_once(
            url,
            [strip_body(&strip), greedy_body(ids[1])],
            ["slots strip", "slots drafting turn"],
        )?;
        let window = rounds_from(err_log, before)?;
        let passes = of_cmd(&window, "pass");
        let skips = skips_of(&lines_from(err_log, from)?)?;
        let strip_at = together[0].as_ref();
        let turn_at = together[1].as_ref();
        let strip_ids = strip_at.map(|v| ids_of(&v["tokens"]));
        println!(
            "slots skipping: together the strip {:?} ids cache_n {:?} draft_n {}, the turn (ids, \
             drafts, accepted) {:?} of alone {turn_alone:?}; skips {skips:?}",
            strip_ids.as_ref().map(Vec::len),
            strip_at.and_then(cache_n),
            strip_at
                .map(|v| v["timings"]["draft_n"].as_u64().unwrap_or(0))
                .unwrap_or(0),
            turn_at.map(run_of),
        );
        print_rounds("slots skipping", &window, &passes);
        check(
            &mut ok,
            "slots_skipping_slot_steps_alone_beside_a_drafting_one",
            !alone_ids.is_empty()
                && alone_skips.iter().any(|s| s.contains(SKIP_AT_JOIN))
                && plain(&alone)
                && cache_n(&alone) == Some(turn_end)
                && strip_ids.as_deref() == Some(alone_ids.as_slice())
                && strip_at.is_some_and(plain)
                && strip_at.and_then(cache_n) == Some(turn_end)
                && turn_at.map(run_of).as_ref() == Some(&turn_alone)
                && turn_alone.1 > 0,
        );
        check(
            &mut ok,
            "slots_skipping_rounds_step_the_off_slot_alone",
            !passes.is_empty()
                && passes.iter().all(|r| {
                    r.u64("rows") == Ok(2) && r.u64("slots") == Ok(2) && r.u64("passes") == Ok(2)
                }),
        );
        check(
            &mut ok,
            "slots_skipping_records_name_the_skip_and_the_turn_off",
            skips.iter().any(|s| s.contains(SKIP_AT_JOIN))
                && skips.iter().any(|s| s.contains(SKIP_TURNED_OFF)),
        );
        Ok(ok)
    }

    /// A greedy `/completion` body of `ids`: [`SLOTS_PREDICT`] tokens at
    /// temperature 0, `cache_prompt` off.
    fn greedy_body(ids: &[u32]) -> Value {
        json!({
            "prompt": ids, "n_predict": SLOTS_PREDICT, "temperature": 0,
            "return_tokens": true, "cache_prompt": false,
        })
    }

    /// A sampled `/completion` body of `ids`: [`SLOTS_PREDICT`] tokens at
    /// temperature 0.8 among the top 40 with `seed`, `cache_prompt` off —
    /// the server steps it, a `next` a token, and never drafts it.
    fn sampled_body(ids: &[u32], seed: u64) -> Value {
        json!({
            "prompt": ids, "n_predict": SLOTS_PREDICT, "temperature": 0.8, "top_k": 40,
            "seed": seed, "return_tokens": true, "cache_prompt": false,
        })
    }

    /// The skipping clause's request of `ids` (the first turn's prompt with
    /// [`STRIPPED_A`] in place of its reply): greedy, `cache_prompt` on — the
    /// request that keeps the turn's prompt call's end, where the draft's
    /// waiting rows do not end, and so skips.
    fn strip_body(ids: &[u32]) -> Value {
        json!({
            "prompt": ids, "n_predict": SLOTS_PREDICT, "temperature": 0,
            "return_tokens": true, "cache_prompt": true,
        })
    }

    /// Two `/completion` requests of `bodies` posted at once, each on a
    /// helper thread of its own: each one's response, in order, or `None`
    /// (printed under its name) when it failed.
    fn both_at_once(
        url: &dyn Fn(&str) -> String,
        bodies: [Value; 2],
        names: [&str; 2],
    ) -> Result<Vec<Option<Value>>, GateError> {
        type Answer = Result<(u16, String), String>;
        let mut posted: Vec<(JoinHandle<()>, mpsc::Receiver<Answer>)> = Vec::new();
        for (b, name) in bodies.into_iter().zip(names) {
            let u = url("/completion");
            let (tx, rx) = mpsc::channel();
            let (h, _) = spawn_helper("slots-request", Placement::Float, move || {
                let _ = tx.send(curl(&u, Some(&b), false).map_err(|e| e.to_string()));
            })
            .map_err(|e| format!("{name}: {}", e.what()))?;
            posted.push((h, rx));
        }
        let mut out = Vec::new();
        for ((h, rx), name) in posted.into_iter().zip(names) {
            h.join()
                .map_err(|_| format!("{name}: the request's thread panicked"))?;
            let answer = rx
                .recv()
                .map_err(|_| format!("{name}: the request's thread gave no answer"))?;
            match answer
                .map_err(GateError::from)
                .and_then(|(st, text)| json_of("/completion", st, &text))
            {
                Ok(v) => out.push(Some(v)),
                Err(e) => {
                    println!("{name}: the request failed: {e}");
                    out.push(None);
                }
            }
        }
        Ok(out)
    }

    /// The `slots round` records the server has printed so far.
    fn round_count(err_log: &Path) -> Result<usize, GateError> {
        Ok(seat_log(&lines_from(err_log, 0)?)
            .all(&record::SLOTS_ROUND)?
            .len())
    }

    /// The `slots round` records after the first `before`.
    fn rounds_from(err_log: &Path, before: usize) -> Result<Vec<record::Fields>, GateError> {
        let mut rounds = seat_log(&lines_from(err_log, 0)?).all(&record::SLOTS_ROUND)?;
        Ok(rounds.split_off(before.min(rounds.len())))
    }

    /// The records of `window` whose command is `cmd`.
    fn of_cmd<'a>(window: &'a [record::Fields], cmd: &str) -> Vec<&'a record::Fields> {
        window.iter().filter(|r| r.word("cmd") == Ok(cmd)).collect()
    }

    /// A window's record count and the lines of the records a clause reads.
    fn print_rounds(what: &str, window: &[record::Fields], read: &[&record::Fields]) {
        println!(
            "{what}: {} round record(s) in the window, {} read: {}",
            window.len(),
            read.len(),
            read.iter()
                .map(|r| r.line().to_owned())
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }

    /// The plain arm's slots clause (the module header): a plain server of
    /// `--parallel 2` under `BLOOMERY_STEP_STATS=1` — no draft, no
    /// residency, so the bits need no expert moving — whose `parallel` line
    /// names `pass=one`; two rendered turns ([`TURN_A`], [`TURN_C`]), each
    /// run alone for [`SLOTS_PREDICT`] greedy ids (`cache_prompt` off),
    /// then both posted at once: each one's ids its alone run's, and every
    /// `slots round` record of the together window — the alone runs print
    /// none, a round of one row being a select and a `next` — reading
    /// `cmd=step`, `slots=2`, `rows=2` and `passes=ceil(rows/2)`, the
    /// body's two-row bound. FAIL-first: the seat left on the fallback loop
    /// prints `passes=rows`.
    fn slots_one_pass(dir: &Path) -> Result<bool, GateError> {
        let dir = dir.join("slots-plain");
        std::fs::create_dir_all(&dir)?;
        // SERVER_ARGS ends in the one-slot `--parallel 1`; this clause's
        // server takes two slots of the same total context.
        let mut args: Vec<&str> = SERVER_ARGS.to_vec();
        let parallel = args.len() - 1;
        args[parallel] = "2";
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&args, &dir, PLAIN_STATS)?;
        println!("plain slots server pid {}", served.child.id());
        let addr = served.address(&err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = true;
        // The `parallel` line printed before the load names the round's
        // shape the open chose.
        let parallel_line = std::fs::read_to_string(&err_log)?
            .lines()
            .find(|l| l.starts_with("parallel "))
            .ok_or("the plain slots server printed no parallel line")?
            .to_owned();
        println!("plain slots: {parallel_line}");
        check(
            &mut ok,
            "plain_slots_parallel_line_names_one_pass",
            field(&parallel_line, "pass") == Some("one"),
        );
        let a_ids = rendered(&url, json!([{ "role": "user", "content": TURN_A }]))?;
        let c_ids = rendered(&url, json!([{ "role": "user", "content": TURN_C }]))?;
        let mut alone = Vec::new();
        for ids in [&a_ids, &c_ids] {
            alone.push(ids_of(&greedy(&url, ids, SLOTS_PREDICT, false)?["tokens"]));
        }
        check(
            &mut ok,
            "plain_slots_alone_runs_are_whole",
            alone.iter().all(|a| a.len() == SLOTS_PREDICT),
        );
        // The alone runs print no record (a round of one row is a select
        // and a `next`), so the window opens empty; taken anyway, so the
        // together run's records are exactly the ones after this point.
        let before = round_count(&err_log)?;
        let together: Vec<Option<Vec<u32>>> = both_at_once(
            &url,
            [greedy_body(&a_ids), greedy_body(&c_ids)],
            ["plain slots A", "plain slots C"],
        )?
        .iter()
        .map(|v| v.as_ref().map(|v| ids_of(&v["tokens"])))
        .collect();
        let window = rounds_from(&err_log, before)?;
        let every = !window.is_empty()
            && window.iter().all(|r| {
                let rows = r.u64("rows");
                r.word("cmd") == Ok("step")
                    && r.u64("slots") == Ok(2)
                    && rows == Ok(2)
                    && r.u64("passes") == rows.map(|n| n.div_ceil(2))
            });
        println!(
            "plain slots alone {} and {} ids, together {:?} and {:?}",
            alone[0].len(),
            alone[1].len(),
            together[0].as_ref().map(Vec::len),
            together[1].as_ref().map(Vec::len),
        );
        print_rounds("plain slots", &window, &window.iter().collect::<Vec<_>>());
        check(
            &mut ok,
            "plain_slots_together_ids_are_alone",
            together[0].as_ref() == Some(&alone[0]) && together[1].as_ref() == Some(&alone[1]),
        );
        check(&mut ok, "plain_slots_rounds_run_one_pass", every);
        println!("plain slots server stopped: {}", served.stop()?);
        Ok(ok)
    }

    /// The plain arm (the module header).
    fn plain(dir: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let mut ok = unset_rule(dir)?;
        ok &= ctx(dir, levers)?;
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&SERVER_ARGS, dir, PLAIN)?;
        println!("plain server pid {}", served.child.id());
        let addr = served.address(&err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let (st, body) = curl(&url("/health"), None, false)?;
        check(
            &mut ok,
            "plain_health_ok",
            st == 200 && body.contains("\"ok\""),
        );
        let load = server_log(&err_log, record::BLOOMERY_SERVE_GLM)?;
        let off = load.one(&record::LOAD_DRAFT_OFF_GLM)?;
        let lever = load.one(&record::RESIDENCY_LEVER)?;
        let (word, why) = (lever.word("residency")?, lever.word("why")?);
        let host = load.first(&record::RESIDENCY_HOST)?;
        let pair = load.first(&record::CAPTURE_PAIR)?;
        println!(
            "plain records: load draft=off ({}), residency lever {word} why={why}, residency host \
             {:?}, pair capture {:?}",
            off.text("why")?,
            host.as_ref().map(record::Fields::line),
            pair.as_ref().map(record::Fields::line)
        );
        check(
            &mut ok,
            "plain_loads_no_draft_and_no_residency",
            off.text("why")? == "BLOOMERY_DRAFT=off"
                && word == "off"
                && why == "set"
                && host.is_none()
                && pair.is_none(),
        );
        ok &= props_draft(&url, None)?;
        let ids = chat_ids(&url)?;
        let (st, _) = chat(&url)?;
        check(&mut ok, "plain_chat_turn_served", st == 200);
        let c = greedy(&url, &ids, N_PREDICT, false)?;
        let got = ids_of(&c["tokens"]);
        let stop = c["stop_type"].as_str().unwrap_or("").to_owned();
        println!("plain completion tokens {got:?} stop_type={stop}");
        let (st, body) = curl(&url("/residency/reset"), Some(&json!({})), false)?;
        println!("plain residency reset: HTTP {st} {body}");
        check(&mut ok, "plain_residency_reset_is_501", st == 501);
        ok &= cache(&url, &err_log, false)?;
        ok &= round_trip(&url, &err_log)?;
        println!("plain server stopped: {}", served.stop()?);
        let (reference, cli_log) = cli(dir, PLAIN, &ids, true)?;
        println!("generate_glm5next tokens {reference:?}");
        check(
            &mut ok,
            "plain_ids_are_generate_glm5next",
            agree(&got, &stop, &reference),
        );
        let (seat, cli_group) = (load_group(&load)?, load_group(&cli_log)?);
        println!("plain load groups: the seat's {seat:?}, generate_glm5next's {cli_group:?}");
        let want = Some(u64::try_from(PROMPT_GROUP)?);
        check(
            &mut ok,
            "plain_loads_run_the_reserved_group",
            seat == want && cli_group == want,
        );
        ok &= draft_only(dir, &ids, &reference)?;
        ok &= slots_one_pass(dir)?;
        Ok(ok)
    }

    /// The drafted arm (the module header).
    fn drafted(dir: &Path, levers: &bloomery_levers::Levers) -> Result<bool, GateError> {
        let bytes = nextn_plan_bytes(levers)?;
        let err_log = dir.join("server.err");
        let mut served = Served::spawn(&SERVER_ARGS, dir, DRAFTED)?;
        println!("drafted server pid {}", served.child.id());
        let addr = served.address(&err_log)?;
        let url = |p: &str| format!("http://{addr}{p}");
        let mut ok = true;
        let (st, body) = curl(&url("/health"), None, false)?;
        check(
            &mut ok,
            "drafted_health_ok",
            st == 200 && body.contains("\"ok\""),
        );
        let load = server_log(&err_log, record::BLOOMERY_SERVE_GLM)?;
        let draft = load.first(&record::LOAD_DRAFT_GLM)?;
        let pair = load.first(&record::CAPTURE_PAIR)?;
        let lever = load.one(&record::RESIDENCY_LEVER)?;
        let host = load.one(&record::RESIDENCY_HOST)?;
        let (word, why, host_word) = (
            lever.word("residency")?,
            lever.word("why")?,
            host.word("residency")?,
        );
        println!(
            "drafted records: load draft {:?}, pair capture {:?}, residency lever {word} \
             why={why}, residency host {host_word}",
            draft.as_ref().map(record::Fields::line),
            pair.as_ref().map(record::Fields::line)
        );
        check(
            &mut ok,
            "drafted_loads_the_draft_and_the_word",
            draft.is_some()
                && pair.is_some()
                && word == RESIDENCY_WORD
                && why == "set"
                && host_word == RESIDENCY_WORD,
        );
        ok &= props_draft(&url, Some(bytes))?;
        let ids = chat_ids(&url)?;

        let before = lines_from(&err_log, 0)?.len();
        let c = greedy(&url, &ids, N_PREDICT, false)?;
        let first = ids_of(&c["tokens"]);
        let stop = c["stop_type"].as_str().unwrap_or("").to_owned();
        let t = &c["timings"];
        println!(
            "drafted completion tokens {first:?} stop_type={stop} draft_n={} draft_n_accepted={}",
            t["draft_n"], t["draft_n_accepted"]
        );
        let served_passes = passes(&seat_log(&lines_from(&err_log, before)?))?;
        println!("drafted completion passes (kind, kept, landed) {served_passes:?}");
        check(
            &mut ok,
            "drafted_completion_carries_the_draft_counts",
            t["draft_n"].as_u64().is_some_and(|n| n > 0)
                && t["draft_n_accepted"].as_u64().is_some_and(|n| n > 0),
        );
        let kind = |k: &str| served_passes.iter().any(|p| p.0 == k);
        check(
            &mut ok,
            "drafted_completion_ends_prompt_step_and_pair_passes",
            kind("prompt") && kind("step") && kind("pair"),
        );

        let (st, body) = curl(&url("/residency/reset"), Some(&json!({})), false)?;
        let reset: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
        let names = ["cancelled", "copies", "diff"];
        let counts = match server_log(&err_log, record::BLOOMERY_SERVE_GLM)?
            .first(&record::RESIDENCY_RESET)?
        {
            Some(r) => Some(
                names
                    .iter()
                    .map(|k| r.u64(k))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            None => None,
        };
        println!("drafted residency reset: HTTP {st} {reset}; record {names:?} {counts:?}");
        check(
            &mut ok,
            "drafted_reset_is_a_200_with_its_record",
            st == 200
                && reset["diff"] == json!(0)
                && counts.is_some_and(|c| {
                    names
                        .iter()
                        .zip(&c)
                        .all(|(k, v)| reset[*k].as_u64() == Some(*v))
                }),
        );
        let again = ids_of(&greedy(&url, &ids, N_PREDICT, false)?["tokens"]);
        println!("drafted completion after the reset tokens {again:?}");
        check(
            &mut ok,
            "drafted_ids_after_the_reset_are_the_first_requests",
            !first.is_empty() && again == first,
        );

        let (st, timings) = chat(&url)?;
        check(
            &mut ok,
            "drafted_chat_turn_drafts",
            st == 200 && timings["draft_n"].as_u64().is_some_and(|n| n > 0),
        );
        let mut extended = ids.clone();
        extended.extend_from_slice(&first);
        let joins_from = lines_from(&err_log, 0)?.len();
        let x = greedy(&url, &extended, 8, true)?;
        let joins: Vec<String> = seat_log(&lines_from(&err_log, joins_from)?)
            .all(&record::MTP_PROMPT)?
            .iter()
            .map(|f| f.line().to_owned())
            .collect();
        println!(
            "drafted continuation (not held): cache_n={} draft_n={} tokens {:?} joins {joins:?}",
            x["timings"]["cache_n"],
            x["timings"]["draft_n"],
            ids_of(&x["tokens"])
        );
        ok &= resume_under_residency(&url, &err_log)?;

        println!("drafted server stopped: {}", served.stop()?);
        ok &= split_refused(dir)?;
        ok &= slots_flow_together(dir)?;

        let (reference, cli_log) = cli(dir, DRAFTED, &ids, true)?;
        let cli_passes = passes(&cli_log)?;
        println!("generate_glm5next tokens {reference:?}");
        println!("generate_glm5next passes (kind, kept, landed) {cli_passes:?}");
        let landing = before_landing(&served_passes, 0).min(before_landing(&cli_passes, 0));
        let same_passes = served_passes == cli_passes;
        println!(
            "drafted ids: the whole run agrees {}; the first landing sits through pass {} of \
             {}, the passes the same lists {same_passes}",
            agree(&first, &stop, &reference),
            landing,
            first.len()
        );
        check(
            &mut ok,
            "drafted_ids_are_generate_glm5next",
            agree(&first, &stop, &reference),
        );
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
        for (name, set) in [
            (bloomery_levers::DRAFT, levers.draft()),
            (bloomery_levers::RESIDENCY, levers.residency()),
        ] {
            if let Some(word) = set {
                return Err(format!(
                    "{name}={word}: the gate sets it on every process it starts (the plain arm \
                     off, the drafted arm mtp and {RESIDENCY_WORD})"
                )
                .into());
            }
        }
        let a = parse_args()?;
        std::fs::create_dir_all(&a.dir)?;
        let ok = match a.arm {
            Arm::Plain => plain(&a.dir, &levers)?,
            Arm::Drafted => drafted(&a.dir, &levers)?,
            Arm::Slots => slots_flow_together(&a.dir)?,
        };
        if ok {
            println!("gate-gpu-glm5next-serve: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
