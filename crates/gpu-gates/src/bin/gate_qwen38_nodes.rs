//! GPU gate for Qwen3.8's plan (b′) with its expert tier served by a tier
//! node (`docs/contrib/dgx-spark.md` 9.5 items 5 and 6): node 0, a process
//! that loads the stage with `BLOOMERY_TIER_NODES` set, and node 1, a
//! `tier_server` beside this binary, over the loopback channel. Both run on
//! this machine's first card; the tier's page is one `/dev/shm` file both
//! processes map.
//!
//! - Census: node 1's census, read over the channel by the function the
//!   load plans its tier cards from (`node_cards`), is the tier card the
//!   plan sizes: the plan's tier card takes the free bytes node 1 sent. The
//!   plan is plan (b′) on this card (stage) and node 1's (tier) under the
//!   largest whole-GiB card budget up to [`TOP_GIB`] that puts a tier expert
//!   on every layer and a host expert on a layer before the last.
//! - Open: invalid layer or column counts and an overflowing block route
//!   are refused by name. Node 0's open of the plan's tier with one routed
//!   segment's experts swapped for as many others is refused by node 1,
//!   naming the tier's set, and node 1 serves the next connection. A second node 1
//!   serving a copy of the model file whose headers are the file's and
//!   whose one tier segment holds one other byte refuses the plan's tier,
//!   naming the segment's other bytes. The copy is the shard that segment
//!   is in, the smallest such shard, under `$BLOOMERY_DATA`, beside links to
//!   the other shards; it is removed after.
//! - Bind: a symlink under the tier page's name is refused after a valid
//!   open; its target is unchanged after disconnect and the server's drain.
//! - Bits: the history (`q38_history`) on node 0 with the tier on node 1,
//!   every token and logits row bit for bit the same plan's history in one
//!   process, the tier a card of that process. Structure: node 0's stage
//!   graphs hold the
//!   one-process load's nodes; the tier's graphs of the step and of each
//!   verify width, captured on node 1, hold the resident or host executor's
//!   nodes per layer and the fault copy; each chain fed and captured once and replayed by
//!   every later pass. Precondition: node 1 holds the tier's experts and
//!   every tier layer was sent a routed slot.
//! - Lost node 1: node 1 killed after the history, node 0's next step fails
//!   within the go deadline and its grace naming the node and the lost card,
//!   the host tier is poisoned as a lost card, and the next step and a reset
//!   are refused.
//! - Stopped node 1: node 1 stopped before node 0 asks a step of it, node
//!   0's step fails within the bound naming the node's request past its
//!   deadline and the lost card, and node 0 keeps its failed load. Node 1 is
//!   then continued: it exits within [`LOST_BOUND_S`] naming the loss, its
//!   words released and its stream drained.
//! - Lost node 0: node 0 asks a step of node 1 through a relay in this
//!   process that holds the step's launch; node 0's stage runs to its first
//!   wait on the tier or on its own host leg. Node 0 is then killed (its
//!   channel ends) or stopped (its channel holds, and with it the host leg
//!   its stage waits on), and the relay delivers the launch. Node 1 waits
//!   for go words that do not come (precondition: node 1 names fewer layers
//!   served than asked), releases its words, drains its stream and exits
//!   naming the loss within [`LOST_BOUND_S`]. A stopped node 0 is then
//!   continued: its held step fails, naming the node's release and the lost
//!   card, and returns no token from the rows node 1 served after it.
//!
//! `--host` lowers an equal partition of expert ids into the loaded map.
//! The functional gate's loopback server has 20 GiB of host room; measurements
//! use the caller's host room. The same plan's local CPU tier is the byte
//! oracle; the node must have no resident device weights and must report
//! nonzero NVMe fills.
//! `--history <gib> <out>` saves a history for an off-host comparison.
//!
//! Node 0 runs as a child of this binary (`--node0`, `--hold0`) with the
//! lever set; this process holds no load while a child does.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen38_nodes", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/q38_history.rs"]
mod q38_history;

#[cfg(feature = "gpu")]
#[path = "shared/qwen38_place.rs"]
#[allow(
    dead_code,
    reason = "the gate builds plan (b′)'s machine; the module's other half serves the CLI and the seat"
)]
mod q38place;

#[cfg(feature = "gpu")]
mod gate {
    use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
    use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use app::Session;
    use bloomery_gpu::GpuError;
    use bloomery_gpu::arch::qwen3moe::NODE_FAMILY;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{Body38, Qwen38Model, SplitWalk};
    use bloomery_gpu::host::PoisonKind;
    use bloomery_gpu::host::node::wire::{In, MAX_FRAME, Out as Payload, Tag};
    use bloomery_gpu::host::node::{NodeOpen, TierNode, node_cards};
    use bloomery_gpu::host::slots::SlotMap;
    use bloomery_gpu::host::step::Chain;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::host::tier::{TierAct, TierCard, TierOpen, TierSet, TierShape};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::{
        GateError, beside, checks_failed, data_dir, prose_ids, ref_model_path, verdict,
    };
    use bloomery_gpu_gates::{record, tier};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, HostCfg, Levers, TIER_NODES,
    };
    use gguf::Split;
    use model::arch::qwen35moe::place::{
        Experts, PlaceError, PlanInputs, machine_bp_on, tier_batch,
    };
    use model::placement::devices::spec_of_device;
    use model::placement::nodes::{NodeBudget, NodesError, Terms};
    use model::placement::workstation::{self, CardSpec};
    use model::placement::{Machine, PlacementError, Plan, PlanLevers};

    use crate::q38_history::{self, PROMPT, Run, same, tier_graphs_and_chains, tier_layers};
    use crate::q38place;

    const NAME: &str = "gate_qwen38_nodes";
    /// The stores' positions, `gate_qwen38_twocard`'s.
    const CTX: usize = 3072;
    /// The largest card budget the plan search tries, GiB: node 0's stage
    /// and node 1's tier share this machine's first card, so two budgets
    /// must fit that device; smaller cards search lower budgets.
    const TOP_GIB: u64 = 16;
    const GATE_HOST_ROOM: u64 = 20 << 30;
    /// The go deadline and the grace a lost node is named within, plus room;
    /// for a lost node 0, node 1's drain deadline too.
    const LOST_BOUND_S: f64 = 45.0;
    /// How long node 0's stage runs its held step before node 0 is killed or
    /// stopped: room for it to reach its first wait.
    const HOLD: Duration = Duration::from_millis(500);
    /// The line node 0 prints before the step the relay holds.
    const ARMED: &str = "armed: the next step waits for a line on stdin";
    /// The line node 0 prints with the held step's result.
    const HELD: &str = "held step: ";

    enum Mode {
        Gate,
        Measure {
            arm: String,
            gib: u64,
        },
        MeasureNode0 {
            gib: u64,
        },
        /// Node 0 of the bits and lost node 1 clauses: the plan under `gib`,
        /// its run written to `out`, node 1's process `server` killed after.
        Node0 {
            gib: u64,
            out: PathBuf,
            server: u32,
        },
        History {
            gib: u64,
            out: PathBuf,
        },
        Decode {
            gib: u64,
            tokens: usize,
            out: PathBuf,
        },
        /// A node 0 that holds its stage's stream and asks a step.
        Hold0 {
            gib: u64,
        },
    }

    fn parse_args(args: &[String]) -> Result<(Mode, bool), GateError> {
        const USAGE: &str = "usage: gate_qwen38_nodes [--host] (node 0 modes, run by the gate: --node0 \
                             <gib> <out> <server pid> | --hold0 <gib>) or --measure \
                             <local|node> <gib> or --decode0 <gib> <tokens> <out> or --history <gib> <out>";
        let host = args.first().is_some_and(|a| a == "--host");
        let args = if host { &args[1..] } else { args };
        let gib = |s: &str, max: u64| -> Result<u64, GateError> {
            let n = s
                .parse::<u64>()
                .map_err(|e| format!("{s:?}: {e}: {USAGE}"))?;
            if n == 0 || n > max {
                return Err(format!("budget must be in 1..={max} GiB").into());
            }
            Ok(n)
        };
        let mode = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
            [] => Mode::Gate,
            ["--node0", g, out, server] => Mode::Node0 {
                gib: gib(g, TOP_GIB)?,
                out: PathBuf::from(out),
                server: server
                    .parse()
                    .map_err(|e| format!("{server:?}: {e}: {USAGE}"))?,
            },
            ["--history", g, out] => Mode::History {
                gib: gib(g, TOP_GIB)?,
                out: PathBuf::from(out),
            },
            ["--decode0", g, tokens, out] => {
                let tokens: usize = tokens
                    .parse()
                    .map_err(|e| format!("{tokens:?}: {e}: {USAGE}"))?;
                if tokens == 0 || tokens > CTX - PROMPT {
                    return Err(
                        format!("decode tokens must be in 1..={}: {USAGE}", CTX - PROMPT).into(),
                    );
                }
                Mode::Decode {
                    gib: gib(g, u64::MAX >> 30)?,
                    tokens,
                    out: PathBuf::from(out),
                }
            }
            ["--hold0", g] => Mode::Hold0 {
                gib: gib(g, TOP_GIB)?,
            },
            ["--measure", arm @ ("local" | "node"), g] => Mode::Measure {
                arm: arm.to_string(),
                gib: gib(g, TOP_GIB)?,
            },
            ["--measure-node0", g] => Mode::MeasureNode0 {
                gib: gib(g, TOP_GIB)?,
            },
            _ => return Err(USAGE.into()),
        };
        if host && matches!(mode, Mode::Decode { .. }) {
            return Err("--host is not supported with --decode0; use --history".into());
        }
        Ok((mode, host))
    }

    /// What every mode reads: the file, its plan inputs, the ubatch and the
    /// ids.
    struct Common {
        path: PathBuf,
        inputs: PlanInputs,
        host: HostCfg,
        ub: usize,
        ids: Vec<u32>,
        node_host: bool,
    }

    impl Common {
        fn new(levers: &Levers) -> Result<Common, GateError> {
            let path = ref_model_path()?;
            let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
            // The plans' budgets are the gate's own search's ([`plan_under`] up to
            // [`TOP_GIB`]): a caller's budget lever is not read, so one that
            // disagrees with the fixture's header is refused here as every other
            // gate refuses it.
            tier::plan_levers(&split, levers, 0)?;
            let inputs = PlanInputs::describe(&split)?;
            let ub = ubatch_for(CTX)?;
            let ids = prose_ids("qwen4exp", PROMPT + 32)?;
            Ok(Common {
                path,
                inputs,
                host: levers.host(),
                ub,
                ids,
                node_host: false,
            })
        }

        /// Plan (b′)'s machine: this process's first card the stage, `node`
        /// the tier.
        fn machine(&self, node: CardSpec) -> Result<Machine, GateError> {
            let own = bloomery_gpu::census()?;
            let stage = own.first().ok_or("this process sees no card")?;
            let layers = self.inputs.stack();
            let ub = self.ub as u64;
            Ok(machine_bp_on(
                (spec_of_device(stage), node),
                layers,
                ub,
                None,
                tier_batch(&self.inputs.spec, ub),
            ))
        }

        /// Plan (b′)'s machine as a node 0 load lays it out: the placement
        /// `Place::on_host` resolves with `BLOOMERY_TIER_NODES` set, its tier
        /// card on the node the placement plans it on
        /// (`q38place::machine_bp38`).
        fn machine_placed(&self) -> Result<Machine, GateError> {
            let ub = self.ub as u64;
            q38place::machine_bp38(
                Place::parse("0+1")?.on_host()?,
                self.inputs.stack(),
                ub,
                None,
                tier_batch(&self.inputs.spec, ub),
            )
        }

        /// A load of `plan`, its session over [`CTX`] positions.
        fn open(&self, plan: &Plan<'_>) -> Result<Session<Body38>, GateError> {
            let t0 = Instant::now();
            let file = Split::open(&self.path)
                .map_err(|e| format!("open {}: {e}", self.path.display()))?;
            let m = Body38::open_placed_residency(
                file,
                plan,
                &self.inputs,
                0,
                self.host,
                self.ub,
                Residency::Off,
                bloomery_levers::ChurnHosting::Hosted,
                1,
            )?;
            let t = m
                .body(NAME)?
                .hybrid()
                .tiers()
                .first()
                .ok_or("the load holds no tier")?;
            let at = t.node().map_or_else(
                || "this process".to_string(),
                |a| format!("the node at {a}"),
            );
            println!(
                "load in {:.1} s: stage {} B, tier {} experts ({} B) on {at}",
                t0.elapsed().as_secs_f64(),
                m.resident_bytes(),
                t.set().experts(),
                t.resident_bytes()
            );
            let mut s = Session::from_model(m, u32::try_from(CTX)?);
            s.model_mut()
                .body_parts(NAME)?
                .2
                .set_split_walk(SplitWalk::Off)?;
            Ok(s)
        }
    }

    /// The plans' context and prompt ubatch: [`CTX`] and the load's ubatch at it.
    fn plan_shape() -> Result<(u64, u64), GateError> {
        Ok((CTX as u64, u64::try_from(ubatch_for(CTX)?)?))
    }

    /// Plan (b′) on `machine` under `gib` GiB a card: the plan, or why it is
    /// not the gate's: budgets past the shared device's usable bytes, a
    /// layer with no tier expert, an idle tier, or no host leg before the
    /// last layer (a stopped node 0 strands its card only at a host leg, so
    /// the stopped clause needs one ahead of a tier layer).
    fn plan_under<'a>(
        c: &'a Common,
        machine: &'a Machine,
        gib: u64,
    ) -> Result<Result<Plan<'a>, String>, GateError> {
        let levers = PlanLevers {
            card_budget_bytes: Some(gib << 30).into(),
        };
        let inputs = &c.inputs;
        let planned = if c.node_host {
            let half = inputs.model.experts / 2;
            inputs.plan_nodes(
                machine,
                plan_shape()?,
                &levers,
                &[half, inputs.model.experts - half],
            )
        } else {
            inputs.plan_with(machine, plan_shape()?, &levers, Experts::Card)
        };
        match planned {
            Ok(plan) => {
                let tier = plan.tier_n_l.first().ok_or("plan (b′) holds no tier")?;
                if let Some(l) = tier.iter().position(|&n| n == 0) {
                    return Ok(Err(format!("layer {l} holds no tier expert")));
                }
                let host = (0..tier.len())
                    .map(|l| plan.host_experts(l))
                    .collect::<Result<Vec<u64>, _>>()?;
                Ok(
                    if host[..host.len().saturating_sub(1)].iter().all(|&n| n == 0) {
                        Err(format!(
                            "no layer before the last holds a host expert (host experts a layer \
                         {host:?})"
                        ))
                    } else {
                        Ok(plan)
                    },
                )
            }
            Err(e) => Ok(Err(plan_refusal(e, gib)?)),
        }
    }

    fn plan_refusal(error: PlaceError, gib: u64) -> Result<String, GateError> {
        match error {
            PlaceError::Placement(PlacementError::IdleTier { .. }) => {
                Ok("the tier holds no expert".to_string())
            }
            PlaceError::Placement(PlacementError::DeviceTwice { .. }) => Ok(error.to_string()),
            e => Err(format!("plan (b′) under {gib} GiB: {e}").into()),
        }
    }

    /// `run` written to `path` for the gate to read back ([`read_run`]).
    fn write_run(run: &Run, path: &Path) -> Result<(), GateError> {
        let mut o = Payload::default();
        o.u32s(&run.tokens).u64s(
            &run.stage_nodes
                .iter()
                .map(|&n| n as u64)
                .collect::<Vec<_>>(),
        );
        o.len(run.logits.len());
        for row in &run.logits {
            o.u32s(&row.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        std::fs::write(path, o.done()).map_err(|e| format!("{}: {e}", path.display()).into())
    }

    /// The run [`write_run`] wrote to `path`; a file of another form is
    /// refused by name.
    fn read_run(path: &Path) -> Result<Run, GateError> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut i = In::new(&b);
        let run = (|| {
            let tokens = i.u32s("tokens")?;
            let stage_nodes = i
                .u64s("stage nodes")?
                .into_iter()
                .map(|n| usize::try_from(n).map_err(|_| format!("a node count {n}")))
                .collect::<Result<Vec<_>, _>>()?;
            let rows = i.len("logits rows")?;
            let logits = (0..rows)
                .map(|_| {
                    Ok(i.u32s("a logits row")?
                        .into_iter()
                        .map(f32::from_bits)
                        .collect())
                })
                .collect::<Result<Vec<Vec<f32>>, String>>()?;
            i.end()?;
            Ok::<_, String>(Run {
                tokens,
                logits,
                stage_nodes,
                landed: 0,
            })
        })();
        run.map_err(|e| format!("{}: {e}", path.display()).into())
    }

    /// The tier's structure after a history ([`tier_graphs_and_chains`]),
    /// and the precondition that the tier serves experts and every tier
    /// layer was sent a routed slot.
    fn tier_structure(s: &mut Session<Body38>, layers: &[usize]) -> Result<bool, GateError> {
        let body = s.model_mut().body_parts(NAME)?.2;
        let tier = body
            .hybrid_mut()
            .tiers_mut()
            .first_mut()
            .ok_or("the load holds no tier")?;
        let (fills, bytes) = tier.host_fills()?;
        let host = tier.resident_bytes() == 0;
        let paging = !host || tier.node().is_none() || (fills > 0 && bytes > 0);
        println!(
            "paging: host tier {host}, NVMe fills {fills}, file bytes {bytes}: {}",
            verdict(paging)
        );
        let ok = tier_graphs_and_chains(tier, layers.len()) && paging;
        let st = tier.stats();
        let first = tier.set().layers().start;
        let hits = |l: usize| st.layer_hits.get(l - first).copied().unwrap_or(0);
        let missing: Vec<usize> = layers.iter().copied().filter(|&l| hits(l) == 0).collect();
        // PIN(2026-10-17): host tiers hold no device weights; graph and paging checks cover their execution.
        let pre = missing.is_empty() && tier.set().experts() > 0;
        println!(
            "precondition: the tier holds {} experts ({} B) and every tier layer was sent a \
             routed slot (missing {missing:?}): {}",
            tier.set().experts(),
            tier.resident_bytes(),
            verdict(pre)
        );
        Ok(ok && pre)
    }

    /// A child process whose output this process prints, each line under
    /// `tag`, and passes on: stdout's lines through [`Proc::lines`], stderr
    /// kept whole.
    struct Proc {
        child: Child,
        lines: Receiver<String>,
        err: Arc<Mutex<String>>,
        stderr_reader: Option<std::thread::JoinHandle<()>>,
        tag: &'static str,
    }

    impl Proc {
        fn spawn(mut cmd: Command, tag: &'static str) -> Result<Proc, GateError> {
            bloomery_gpu_gates::die_with_parent(&mut cmd);
            let mut child = cmd
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("{tag}: spawn: {e}"))?;
            let (tx, lines) = mpsc::channel();
            let out = child.stdout.take().ok_or("a child with no stdout")?;
            std::thread::spawn(move || {
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    println!("[{tag}] {line}");
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
            let err = Arc::new(Mutex::new(String::new()));
            let mut stderr = child.stderr.take().ok_or("a child with no stderr")?;
            let kept = Arc::clone(&err);
            let stderr_reader = std::thread::spawn(move || {
                let mut b = [0u8; 4096];
                while let Ok(n) = stderr.read(&mut b) {
                    if n == 0 {
                        break;
                    }
                    let text = String::from_utf8_lossy(&b[..n]).into_owned();
                    for l in text.lines() {
                        eprintln!("[{tag}] {l}");
                    }
                    kept.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push_str(&text);
                }
            });
            Ok(Proc {
                child,
                lines,
                err,
                stderr_reader: Some(stderr_reader),
                tag,
            })
        }

        /// The first stdout line starting with `prefix`, waited for within
        /// `within`.
        fn wait_line(&self, prefix: &str, within: Duration) -> Result<String, GateError> {
            let end = Instant::now() + within;
            loop {
                let left = end.saturating_duration_since(Instant::now());
                match self.lines.recv_timeout(left) {
                    Ok(l) if l.starts_with(prefix) => return Ok(l),
                    Ok(_) => {}
                    Err(e) => {
                        return Err(format!("{}: no line {prefix:?} ({e})", self.tag).into());
                    }
                }
            }
        }

        /// The process's exit, waited for within `within`, with its stderr
        /// read to the end; `None` past it.
        fn wait_within(&mut self, within: Duration) -> Result<Option<ExitStatus>, GateError> {
            let end = Instant::now() + within;
            loop {
                if let Some(s) = self.child.try_wait()? {
                    if let Some(reader) = self.stderr_reader.take() {
                        reader
                            .join()
                            .map_err(|_| format!("{}: stderr reader panicked", self.tag))?;
                    }
                    return Ok(Some(s));
                }
                if Instant::now() > end {
                    return Ok(None);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }

        fn signal(&self, sig: &str) -> Result<(), GateError> {
            let pid = self.child.id().to_string();
            let s = Command::new("kill").args([sig, &pid]).status()?;
            if !s.success() {
                return Err(format!("kill {sig} {pid}: {s}").into());
            }
            Ok(())
        }

        /// Write `line` to the process's stdin.
        fn say(&mut self, line: &str) -> Result<(), GateError> {
            let stdin = self.child.stdin.as_mut().ok_or("a child with no stdin")?;
            stdin.write_all(line.as_bytes())?;
            stdin.flush()?;
            Ok(())
        }

        fn stderr(&self) -> String {
            self.err
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
        }
    }

    impl Drop for Proc {
        fn drop(&mut self) {
            if let Ok(None) = self.child.try_wait() {
                drop(self.child.kill());
                drop(self.child.wait());
            }
        }
    }

    fn node1_command(model: Option<&Path>, host_room: Option<u64>) -> Result<Command, GateError> {
        let mut cmd = Command::new(beside("tier_server")?);
        cmd.args(["--listen", "127.0.0.1:0"])
            .env_remove(TIER_NODES)
            .env_remove(CARD_BUDGET)
            .env_remove(HOST_POPULATE)
            .env_remove(HOST_LOCK)
            .env_remove(CARD_DONTNEED);
        if let Some(room) = host_room {
            cmd.env(bloomery_levers::HOST_ROOM, room.to_string());
        }
        if let Some(m) = model {
            cmd.arg("-m").arg(m);
        }
        Ok(cmd)
    }

    /// Node 1: a tier server on a free loopback port, serving `model` when
    /// one is given, and its address.
    fn node1(
        model: Option<&Path>,
        host_room: Option<u64>,
    ) -> Result<(Proc, SocketAddr), GateError> {
        let p = Proc::spawn(node1_command(model, host_room)?, "node 1")?;
        let line = p.wait_line("listening on ", Duration::from_secs(120))?;
        let addr = line["listening on ".len()..].parse()?;
        Ok((p, addr))
    }

    /// Node 0 as a child of this binary in `mode`'s arguments, with the
    /// lever naming `addr`.
    fn node0(args: &[String], addr: SocketAddr, host: bool) -> Result<Proc, GateError> {
        let mut cmd = Command::new(std::env::current_exe()?);
        if host {
            cmd.arg("--host");
        }
        cmd.args(args).env(TIER_NODES, addr.to_string());
        Proc::spawn(cmd, "node 0")
    }

    /// The tier card of the node at `addr` as the load plans it
    /// ([`node_cards`]), and the free bytes its census sent.
    fn node_card(addr: SocketAddr) -> Result<(CardSpec, u64), GateError> {
        let cards = node_cards(&[addr])?;
        let (census, spec) = cards.first().ok_or("no tier card for one node")?;
        for d in &census.devices {
            println!(
                "census of node 1: device {} {:?}, total {} B, free {} B, integrated {}",
                d.ordinal, d.name, d.total_bytes, d.free_bytes, d.integrated
            );
        }
        let free = census.devices.first().map_or(0, |d| d.free_bytes);
        Ok((*spec, free))
    }

    fn census_specs(addr: SocketAddr) -> Result<(CardSpec, CardSpec, u64), GateError> {
        let (spec, free) = node_card(addr)?;
        let own = bloomery_gpu::census()?;
        let stage = spec_of_device(own.first().ok_or("this process sees no card")?);
        Ok((stage, spec, free))
    }

    /// The census clause: node 1's census over the channel, and the plan of
    /// its card; the plan and its budget.
    fn census_plan<'a>(
        c: &'a Common,
        machine: &'a mut Option<Machine>,
        addr: SocketAddr,
    ) -> Result<(Plan<'a>, u64, bool), GateError> {
        let (stage, spec, free) = census_specs(addr)?;
        let machines = workstation::plan_node_tiers(
            c.inputs.model.layers,
            stage,
            &[spec],
            tier_batch(&c.inputs.spec, c.ub as u64),
        )?;
        if machines.len() != 2 {
            return Err("the census plan omitted a node".into());
        }
        let stage_plan = c.inputs.plan_with(
            &machines[0],
            plan_shape()?,
            &PlanLevers::default(),
            Experts::Card,
        )?;
        let terms = Terms::of(&stage_plan, u64::from(c.inputs.spec.hidden));
        let mut budgets = vec![NodeBudget::of(&stage_plan)];
        for remote in &machines[1..] {
            budgets.push(NodeBudget::routed(
                &c.inputs.model,
                remote,
                model::placement::accepts::routed_rule(model::arch::qwen35moe::place::PROGRAM),
            ));
        }
        let mut independent = match (machines[0].unified(), &terms) {
            (Some(_), Ok(terms)) => terms.experts == c.inputs.model.experts,
            (None, Err(NodesError::NotUnified(0))) => true,
            (_, Err(e)) => return Err(format!("the census terms: {e}").into()),
            (None, Ok(_)) => return Err("a discrete node acquired unified terms".into()),
        };
        for ((m, budget), spec) in machines.iter().zip(&budgets).zip([stage, spec]) {
            independent &= m.nodes.len() == 1
                && m.tiers.is_empty()
                && m.cards[0].device == spec.device
                && m.cards[0].free_bytes == spec.free_bytes
                && m.unified() == workstation::unified_of(&spec);
            match (m.unified(), budget) {
                (Some(_), Ok(budget)) => independent &= budget.node() == &m.nodes[0],
                (None, Err(NodesError::NotUnified(0))) => {}
                (_, Err(e)) => return Err(format!("the census budget: {e}").into()),
                (None, Ok(_)) => return Err("a discrete node acquired a unified budget".into()),
            }
        }
        println!(
            "census: independent node budgets {budgets:?}, terms {terms:?}: {}",
            verdict(independent)
        );
        let m = machine.insert(c.machine(spec)?);
        let m: &'a Machine = m;
        let tier = m
            .tiers
            .first()
            .ok_or("plan (b′)'s machine holds no tier card")?;
        let mut ok = independent && tier.free_bytes == Some(free);
        println!(
            "census: the plan's tier card {:?} takes the free bytes node 1 sent ({:?} = {free} B): \
             {}",
            tier.name,
            tier.free_bytes,
            verdict(ok)
        );
        for gib in (1..=TOP_GIB).rev() {
            match plan_under(c, m, gib)? {
                Err(why) => println!("budget {gib} GiB: {why}; next"),
                Ok(plan) => {
                    let held = |v: &[u64]| {
                        (
                            v.iter().copied().min().unwrap_or(0),
                            v.iter().copied().max().unwrap_or(0),
                        )
                    };
                    println!(
                        "plan (b′) under {gib} GiB a card: stage {} experts (a layer {:?}), tier \
                         {} experts (a layer {:?}) on {} layers, host {} experts",
                        plan.cards[0].experts,
                        held(&plan.n_l),
                        plan.cards.get(1).map_or(0, |c| c.experts),
                        held(plan.tier_n_l.first().map_or(&[][..], Vec::as_slice)),
                        tier_layers(&plan).len(),
                        plan.host.experts
                    );
                    ok &= plan.cards.get(1).is_some_and(|c| c.experts > 0);
                    return Ok((plan, gib, ok));
                }
            }
        }
        Err(format!("no budget up to {TOP_GIB} GiB puts a tier expert on every layer").into())
    }

    /// The plan a node 0 child loads: the placement `Place::on_host` resolves
    /// with `BLOOMERY_TIER_NODES` set, its tier node 1's device at `addr`,
    /// under `gib`.
    fn child_plan<'a>(
        c: &'a Common,
        machine: &'a mut Option<Machine>,
        addr: SocketAddr,
        gib: u64,
    ) -> Result<Plan<'a>, GateError> {
        let (spec, _) = node_card(addr)?;
        let m: &'a Machine = machine.insert(c.machine_placed()?);
        let tier = m.tiers.first().ok_or("the placement holds no tier")?;
        if tier.device.map(|d| d.uuid) != spec.device.map(|d| d.uuid) {
            return Err("the placement did not select node 1's device".into());
        }
        plan_under(c, m, gib)?.map_err(|why| format!("under {gib} GiB: {why}").into())
    }

    /// The plan's tier as node 0 opens it on a node, its rows from `file`
    /// and its shape a placeholder: each open clause is refused before the
    /// shape is read.
    fn tier_open(plan: &Plan<'_>, file: &Split) -> Result<NodeOpen, GateError> {
        let t = TierOpen::of_machine(plan.machine)?
            .into_iter()
            .next()
            .ok_or("the plan holds no tier card")?;
        let n_expert = usize::try_from(plan.model.experts)?;
        let map = SlotMap::of_plan_tiers(plan, 0, &[t.card], 0..plan.model.layers, n_expert)?;
        Ok(NodeOpen {
            nvme: None,
            family: NODE_FAMILY.word.to_string(),
            name: t.name,
            device: t.device,
            card: t.card,
            rows: NodeOpen::rows_of(plan, file, t.card)?,
            set: TierSet::of_map(&map, 0)?,
            shape: TierShape {
                hidden: 1,
                n_used: 1,
                rows: 1,
                act: TierAct::F32,
            },
            cols: 1,
            layers: plan.model.layers,
            block: (0, 0),
        })
    }

    /// The open clause: the plan's tier as node 0 opens it on the node at
    /// `addr`: invalid allocation bounds are refused, then one routed
    /// segment's experts swapped for as many others not in the set is
    /// refused naming the set. Each refusal leaves the server accepting.
    fn swapped_open(
        c: &Common,
        plan: &Plan<'_>,
        file: &Split,
        addr: SocketAddr,
    ) -> Result<bool, GateError> {
        let mut open = tier_open(plan, file)?;
        open.shape.hidden = usize::try_from(c.inputs.spec.hidden)?;
        open.shape.n_used = usize::try_from(plan.model.experts_used)?;
        for (field, value) in [
            ("layers", 0),
            ("layers", usize::MAX),
            ("columns", 0),
            ("columns", model::ops::DEFER_MAX_COLS + 1),
            ("block", usize::MAX),
            (
                "block",
                usize::MAX / open.shape.n_used / size_of::<u32>() + 1,
            ),
        ] {
            let mut bad = open.clone();
            match field {
                "layers" => bad.layers = value,
                "columns" => bad.cols = value,
                "block" => bad.block.0 = value,
                _ => unreachable!(),
            }
            bad.rows.clear();
            bad.set = TierSet::new(0..0, open.set.n_expert(), vec![])?;
            let r = TierNode::open(addr, &bad).map(drop);
            let ok = matches!(&r, Err(GpuError::Shape { detail, .. }) if detail.contains(field));
            println!(
                "{} open: node 1 refused {field}={value} with {r:?}",
                if ok { "ok" } else { "FAIL" }
            );
            if !ok {
                return Ok(false);
            }
        }
        let n_expert = u32::try_from(plan.model.experts)?;
        let row = open
            .rows
            .iter_mut()
            .find(|r| r.experts.as_ref().is_some_and(|e| !e.is_empty()))
            .ok_or("the tier card holds no routed segment")?;
        let held = row.experts.take().unwrap_or_default();
        let others: Vec<u32> = (0..n_expert)
            .filter(|e| !held.contains(e))
            .take(held.len())
            .collect();
        println!("open: {} experts {held:?} sent as {others:?}", row.name);
        row.experts = Some(others);
        let r = TierNode::open(addr, &open).map(drop);
        let ok = matches!(&r, Err(e) if e.to_string().contains("its set"));
        println!(
            "{} open: node 1 answered the swapped segment with {r:?}",
            if ok { "ok" } else { "FAIL" }
        );
        Ok(ok)
    }

    /// One server frame, bounded as the node protocol's frames are.
    fn node_frame(s: &mut TcpStream) -> Result<Vec<u8>, GateError> {
        let mut len = [0u8; 4];
        s.read_exact(&mut len)?;
        let len = u32::from_le_bytes(len) as usize;
        if !(1..=MAX_FRAME).contains(&len) {
            return Err(format!("node 1 sent a frame of {len} bytes").into());
        }
        let mut frame = vec![0u8; len];
        s.read_exact(&mut frame)?;
        Ok(frame)
    }

    /// Send one request and read its reply on the clause's bounded stream.
    fn node_reply(s: &mut TcpStream, tag: Tag, body: &[u8]) -> Result<Vec<u8>, GateError> {
        let len = u32::try_from(body.len() + 1)?;
        s.write_all(&len.to_le_bytes())?;
        s.write_all(&[tag as u8])?;
        s.write_all(body)?;
        node_frame(s)
    }

    /// A valid open followed by a bind to a symlink is refused; after the
    /// channel ends without Close, the target's bytes are still its own.
    fn symlink_bind(c: &Common, plan: &Plan<'_>, file: &Split) -> Result<bool, GateError> {
        use std::os::unix::fs::OpenOptionsExt;

        let mut open = tier_open(plan, file)?;
        open.shape.hidden = usize::try_from(c.inputs.spec.hidden)?;
        open.shape.n_used = usize::try_from(plan.model.experts_used)?;
        open.block = (
            1,
            usize::try_from(model::arch::qwen35moe::place::tier_route_scratch_bytes(1))?,
        );
        let (mut n1, addr) = node1(None, c.node_host.then_some(GATE_HOST_ROOM))?;
        open.project(plan, &TierNode::census_of(addr)?)?;
        let target = std::env::temp_dir().join(format!("{NAME}-{}.target", std::process::id()));
        let alias =
            Path::new("/dev/shm").join(format!("bloomery-tier-{}-symlink", std::process::id()));
        let before = vec![0x5Au8; 1 << 20];
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&target)?;
        let mut linked = false;
        let r = (|| -> Result<bool, GateError> {
            f.write_all(&before)?;
            drop(f);
            std::os::unix::fs::symlink(&target, &alias)?;
            linked = true;

            let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(60))?;
            s.set_read_timeout(Some(Duration::from_secs(600)))?;
            s.set_write_timeout(Some(Duration::from_secs(60)))?;
            let census = node_frame(&mut s)?;
            if census[0] != Tag::Census as u8 {
                return Err("bind: node 1 sent no census".into());
            }
            let mut payload = Payload::default();
            open.write(&mut payload)?;
            let opened = node_reply(&mut s, Tag::Open, payload.done())?;
            if opened[0] != Tag::Ok as u8 {
                return Err(format!("bind: node 1 refused the valid open: {opened:?}").into());
            }
            payload.clear();
            bloomery_gpu::host::node::write_file_bind(&alias, before.len(), 1, 0, &mut payload)?;
            let reply = node_reply(&mut s, Tag::Bind, payload.done())?;
            let refused = reply[0] == Tag::Err as u8
                && String::from_utf8_lossy(&reply[1..]).contains(alias.to_string_lossy().as_ref());
            s.shutdown(Shutdown::Both)?;
            drop(s);
            let ended = n1.wait_within(Duration::from_secs(60))?;
            let drained = ended.is_some_and(|s| !s.success())
                && n1
                    .stderr()
                    .contains("the tier held no page and its stream drained");
            if ended.is_none() {
                return Err("bind: node 1 did not end after disconnect".into());
            }
            let unchanged = std::fs::read(&target)? == before;
            let ok = refused && drained && unchanged;
            println!(
                "{} bind: symlink refused {refused}, disconnected server drained {drained}, \
                 target unchanged {unchanged}",
                if ok { "ok" } else { "FAIL" }
            );
            Ok(ok)
        })();
        let mut errors = Vec::new();
        for path in [linked.then_some(alias.as_path()), Some(target.as_path())]
            .into_iter()
            .flatten()
        {
            if let Err(e) = std::fs::remove_file(path) {
                errors.push(format!("remove {}: {e}", path.display()));
            }
        }
        if errors.is_empty() {
            r
        } else {
            if let Err(e) = r {
                errors.insert(0, format!("bind: {e}"));
            }
            Err(errors.join("; ").into())
        }
    }

    /// The open clause on other bytes: a copy of `file` under
    /// `$BLOOMERY_DATA` whose one tier segment's first byte is flipped, its
    /// headers `file`'s, served by a second node 1; the plan's tier opened
    /// on it is refused naming the segment's other bytes.
    fn other_bytes_open(plan: &Plan<'_>, file: &Split, host: bool) -> Result<bool, GateError> {
        let open = tier_open(plan, file)?;
        let hashed = open.rows.iter().try_fold(0u64, |sum, r| {
            let held = match &r.experts {
                Some(ids) => u64::try_from(ids.len())? * (r.file_bytes / plan.model.experts),
                None => r.file_bytes,
            };
            Ok::<_, GateError>(sum + held)
        })?;
        println!(
            "open: the tier's {} rows hold {hashed} file bytes, which each node hashes",
            open.rows.len()
        );
        let size = |s: usize| {
            file.shard_path(s)
                .and_then(|p| std::fs::metadata(p).ok())
                .map_or(u64::MAX, |m| m.len())
        };
        let (row, first) = open
            .rows
            .iter()
            .filter_map(|r| Some((r, *r.experts.as_ref()?.first()?)))
            .min_by_key(|(r, _)| size(r.shard))
            .ok_or("the tier card holds no routed segment")?;
        let (s, info) = file
            .find(&row.name)
            .ok_or("the tier's tensor is not in the file")?;
        let g = file.shard(s).ok_or("the tier's shard is not in the file")?;
        let per = info.nbytes / plan.model.experts;
        let at = g.data_base() + info.offset + u64::from(first) * per;
        let dir = data_dir().join(format!("{NAME}-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let made = (|| -> Result<PathBuf, GateError> {
            let mut paths = Vec::new();
            for i in 0..file.shard_count() {
                let src = file.shard_path(i).ok_or("a shard with no path")?;
                let dst = dir.join(src.file_name().ok_or("a shard path with no name")?);
                if i == s {
                    std::fs::copy(src, &dst)?;
                    let mut f = std::fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(&dst)?;
                    let mut b = [0u8; 1];
                    f.seek(SeekFrom::Start(at))?;
                    f.read_exact(&mut b)?;
                    f.seek(SeekFrom::Start(at))?;
                    f.write_all(&[!b[0]])?;
                } else {
                    std::os::unix::fs::symlink(std::fs::canonicalize(src)?, &dst)?;
                }
                paths.push(dst);
            }
            println!(
                "open: {} expert {first}: byte {at} of shard {s} flipped in a copy of its {} B \
                 under {}",
                row.name,
                size(s),
                dir.display()
            );
            paths
                .into_iter()
                .next()
                .ok_or_else(|| "a file of no shard".into())
        })();
        let r = made.and_then(|copy| {
            let (_n1, addr) = node1(Some(&copy), host.then_some(GATE_HOST_ROOM))?;
            Ok(TierNode::open(addr, &open).map(drop))
        });
        let r = remove_open_copy(&dir, r)?;
        let ok = matches!(&r, Err(e) if e.to_string().contains("hold other bytes")
            && e.to_string().contains(&row.name));
        println!(
            "{} open: node 1 on the copy answered the plan's tier with {r:?}",
            if ok { "ok" } else { "FAIL" }
        );
        Ok(ok)
    }

    /// Remove the copy and keep both failures when the open and cleanup fail.
    fn remove_open_copy(
        dir: &Path,
        r: Result<Result<(), bloomery_gpu::GpuError>, GateError>,
    ) -> Result<Result<(), bloomery_gpu::GpuError>, GateError> {
        if let Err(e) = std::fs::remove_dir_all(dir) {
            let open = match &r {
                Err(e) => Some(e.to_string()),
                Ok(open) => open.as_ref().err().map(ToString::to_string),
            };
            let cleanup = format!("remove {}: {e}", dir.display());
            return Err(match open {
                Some(open) => format!("open: {open}; {cleanup}"),
                None => cleanup,
            }
            .into());
        }
        r
    }

    /// The address `BLOOMERY_TIER_NODES` names, the one a child reads.
    fn lever_addr() -> Result<SocketAddr, GateError> {
        match bloomery_levers::tier_nodes()?[..] {
            [a] => Ok(a),
            ref a => Err(format!("{TIER_NODES} names {} addresses, the gate one", a.len()).into()),
        }
    }

    /// Save the local or remote history, check its structure, and optionally
    /// kill the loopback server for the lost-node clause.
    fn run_history(
        c: &Common,
        gib: u64,
        out: &Path,
        server: Option<u32>,
    ) -> Result<bool, GateError> {
        let nodes = bloomery_levers::tier_nodes()?;
        let addr = match nodes.as_slice() {
            [] => None,
            [addr] => Some(*addr),
            _ => return Err("the history needs exactly one tier".into()),
        };
        let mut machine = None;
        let plan = if let Some(addr) = addr {
            child_plan(c, &mut machine, addr, gib)?
        } else {
            let own = bloomery_gpu::census()?;
            let card = own.first().ok_or("no local tier card")?;
            let m = machine.insert(c.machine(spec_of_device(card))?);
            plan_under(c, m, gib)?.map_err(|why| format!("under {gib} GiB: {why}"))?
        };
        let capacity = if c.node_host {
            let resident = c.inputs.plan_with(
                plan.machine,
                plan_shape()?,
                &PlanLevers {
                    card_budget_bytes: Some(gib << 30).into(),
                },
                Experts::Card,
            )?;
            let tier = &plan.cards[1];
            let before = &resident.cards[1];
            let ok = tier.experts > before.experts && tier.expert_bytes == 0;
            println!(
                "capacity: the node owns {} experts against the resident tier's {}, with {} B \
                 charged as resident tier weights: {}",
                tier.experts,
                before.experts,
                tier.expert_bytes,
                verdict(ok)
            );
            ok
        } else {
            true
        };
        let layers = tier_layers(&plan);
        let mut s = c.open(&plan)?;
        let on = s
            .model()
            .body(NAME)?
            .hybrid()
            .tiers()
            .first()
            .and_then(TierCard::node);
        let mut ok = capacity && on == addr;
        println!(
            "node 0: the tier is served by the node at {on:?}: {}",
            verdict(ok)
        );
        let run = q38_history::history(&mut s, &c.ids, |_| Ok(()))?;
        write_run(&run, out)?;
        ok &= tier_structure(&mut s, &layers)?;
        if let Some(server) = server {
            ok &= lost_node1(
                s.model_mut(),
                &c.ids,
                addr.ok_or("no server address")?,
                server,
            )?;
        }
        Ok(ok)
    }

    /// Node 1 killed: the next step fails naming the node and the lost card
    /// within the bound, the poison is a lost card, the next step and a
    /// reset are refused.
    fn lost_node1(
        m: &mut Qwen38Model,
        ids: &[u32],
        addr: SocketAddr,
        server: u32,
    ) -> Result<bool, GateError> {
        m.reset()?;
        m.step(&ids[..4])?;
        let s = Command::new("kill")
            .args(["-KILL", &server.to_string()])
            .status()?;
        println!("lost node 1: kill -KILL {server}: {s}");
        let t0 = Instant::now();
        let r = m.step(&[ids[4]]);
        let secs = t0.elapsed().as_secs_f64();
        let node = addr.to_string();
        let mut ok = match &r {
            Err(e)
                if e.to_string().contains("the card is lost")
                    && e.to_string().contains(&node)
                    && secs < LOST_BOUND_S =>
            {
                println!("ok lost node 1: the step fails in {secs:.1} s naming the lost node: {e}");
                true
            }
            other => {
                println!(
                    "FAIL lost node 1: after {secs:.1} s the step gave {other:?}, want the lost \
                     card at {node}"
                );
                false
            }
        };
        let kind = m.body(NAME)?.hybrid().last_poison().map(|p| p.mark().kind);
        if kind != Some(PoisonKind::CardLost) {
            println!("FAIL lost node 1: the host tier's poison is {kind:?}, want CardLost");
            ok = false;
        }
        match m.step(&[ids[4]]) {
            Err(e) => println!("ok lost node 1: the next step is refused: {e}"),
            Ok(_) => {
                println!("FAIL lost node 1: the next step ran");
                ok = false;
            }
        }
        match m.reset() {
            Err(e) if e.to_string().contains("the card is lost") => {
                println!("ok lost node 1: a reset is refused by the lost card's poison: {e}");
            }
            other => {
                println!("FAIL lost node 1: a reset gave {other:?}, want the lost card's refusal");
                ok = false;
            }
        }
        Ok(ok)
    }

    /// `--hold0`: node 0, its step chain captured, waits for a line on
    /// stdin and runs one more step, whose launch the relay holds when node
    /// 0 is reached through it; this process is killed or stopped inside
    /// it. It keeps its load after the step until the gate kills it.
    fn run_hold0(c: &Common, gib: u64) -> Result<(), GateError> {
        let addr = lever_addr()?;
        let mut machine = None;
        let plan = child_plan(c, &mut machine, addr, gib)?;
        let mut s = c.open(&plan)?;
        let m = s.model_mut();
        m.reset()?;
        // The step chain's first pass captures the tier's graph and the
        // next replays it: the held step sends node 1 one launch.
        for &id in &c.ids[..3] {
            m.step(&[id])?;
        }
        println!("{ARMED}");
        std::io::stdin().read_line(&mut String::new())?;
        let r = m.step(&[c.ids[3]]);
        println!("{HELD}{r:?}");
        std::io::stdin().read_line(&mut String::new())?;
        Err("the held step returned: a node 0 the gate stops ends by its kill".into())
    }

    /// The tier's progress and the layers asked of it, as node 1 names them
    /// in its loss: "the tier past P of the A layers asked of it".
    fn progress(err: &str) -> Option<(u64, u64)> {
        let at = err.find("the tier past ")? + "the tier past ".len();
        let mut words = err[at..].split_whitespace();
        let past = words.next()?.parse().ok()?;
        let asked = words.nth(2)?.parse().ok()?;
        Some((past, asked))
    }

    /// A relay between node 0 and node 1 that holds node 0's next launch
    /// frame once armed: node 0's stage runs its step to the first wait the
    /// tier or its own host leg must answer, and node 1 is asked the step
    /// only when the gate forwards the frame. Each connection node 0 opens
    /// gets its own connection to node 1; node 1's bytes pass unread.
    struct Relay {
        addr: SocketAddr,
        armed: Arc<AtomicBool>,
        held: Receiver<()>,
        release: Sender<bool>,
    }

    impl Relay {
        fn start(to: SocketAddr) -> Result<Relay, GateError> {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            let addr = listener.local_addr()?;
            let armed = Arc::new(AtomicBool::new(false));
            let (held_tx, held) = mpsc::channel();
            let (release, release_rx) = mpsc::channel();
            let release_rx = Arc::new(Mutex::new(release_rx));
            let flag = Arc::clone(&armed);
            std::thread::spawn(move || {
                for down in listener.incoming() {
                    let (Ok(down), Ok(up)) = (down, TcpStream::connect(to)) else {
                        return;
                    };
                    let (Ok(mut from), Ok(mut back)) = (up.try_clone(), down.try_clone()) else {
                        return;
                    };
                    std::thread::spawn(move || {
                        drop(std::io::copy(&mut from, &mut back));
                        drop(back.shutdown(Shutdown::Both));
                    });
                    let (flag, held, release) =
                        (Arc::clone(&flag), held_tx.clone(), Arc::clone(&release_rx));
                    std::thread::spawn(move || relay_up(down, up, &flag, &held, &release));
                }
            });
            Ok(Relay {
                addr,
                armed,
                held,
                release,
            })
        }
    }

    /// Node 0's frames to node 1, in order; the first launch frame after the
    /// relay is armed is held until the gate releases it, and node 1's side
    /// is closed after it when the release says so (node 0 killed).
    fn relay_up(
        mut down: TcpStream,
        mut up: TcpStream,
        armed: &AtomicBool,
        held: &Sender<()>,
        release: &Mutex<Receiver<bool>>,
    ) {
        loop {
            let Ok(frame) = node_frame(&mut down) else {
                drop(up.shutdown(Shutdown::Write));
                return;
            };
            let len = u32::try_from(frame.len())
                .expect("node_frame bounds its length below u32::MAX")
                .to_le_bytes();
            let mut close = false;
            if frame.first().copied() == Some(Tag::Launch as u8)
                && armed.swap(false, Ordering::SeqCst)
            {
                if held.send(()).is_err() {
                    return;
                }
                close = release
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .recv()
                    .unwrap_or(true);
            }
            if up
                .write_all(&len)
                .and_then(|()| up.write_all(&frame))
                .is_err()
            {
                return;
            }
            if close {
                drop(up.shutdown(Shutdown::Write));
                return;
            }
        }
    }

    /// Node 0 asks a step of node 1 through the relay, which holds the
    /// step's launch; node 0 is then ended by `sig` and the launch
    /// forwarded: node 1 exits within the bound, naming the loss, its go
    /// words released and its stream drained.
    fn lost_node0(gib: u64, sig: &str, what: &str, host: bool) -> Result<bool, GateError> {
        let (mut n1, addr) = node1(None, host.then_some(GATE_HOST_ROOM))?;
        let relay = Relay::start(addr)?;
        let mut n0 = node0(&["--hold0".to_string(), gib.to_string()], relay.addr, host)?;
        n0.wait_line(ARMED, Duration::from_secs(600))?;
        relay.armed.store(true, Ordering::SeqCst);
        n0.say("step\n")?;
        relay
            .held
            .recv_timeout(Duration::from_secs(120))
            .map_err(|e| format!("lost node 0 ({what}): node 0 sent no launch: {e}"))?;
        std::thread::sleep(HOLD);
        n0.signal(sig)?;
        relay
            .release
            .send(sig == "-KILL")
            .map_err(|e| format!("lost node 0 ({what}): the relay ended: {e}"))?;
        let t0 = Instant::now();
        let status = n1.wait_within(Duration::from_secs_f64(LOST_BOUND_S))?;
        let secs = t0.elapsed().as_secs_f64();
        let mut cont = true;
        if sig != "-KILL" {
            n0.signal("-CONT")?;
            let held = n0
                .wait_line(HELD, Duration::from_secs(120))
                .unwrap_or_else(|e| e.to_string());
            cont = held.contains("Err(")
                && held.contains("released the tier's words")
                && held.contains("the card is lost");
            println!(
                "{} lost node 0 ({what}): continued, node 0's held step gave {held}",
                if cont { "ok" } else { "FAIL" }
            );
            n0.signal("-KILL")?;
        }
        n0.wait_within(Duration::from_secs(60))?;
        let err = n1.stderr();
        let named = err.contains("node 0 at") && err.contains("is lost");
        let (past, asked) = progress(&err).unwrap_or((0, 0));
        let mid = past < asked;
        println!(
            "precondition: node 1 was past {past} of the {asked} layers node 0 asked of it when \
             node 0 was {what}: {}",
            verdict(mid)
        );
        let released = err.contains("the tier released its words and its stream drained");
        let ok = status.is_some_and(|s| !s.success()) && named && released && mid;
        println!(
            "{} lost node 0 ({what}): node 1 ended {status:?} {secs:.1} s after, the loss named \
             {named}, its words released and its stream drained {released}",
            if ok { "ok" } else { "FAIL" }
        );
        Ok(ok && cont)
    }

    /// Node 1 stopped before node 0 asks a step of it: node 0's request
    /// passes its deadline and the step fails naming the node and the lost
    /// card, node 0 keeping its failed load. Node 1 is then continued: it
    /// exits within the bound, naming the loss, its words released and its
    /// stream drained.
    fn stopped_node1(gib: u64, host: bool) -> Result<bool, GateError> {
        let (mut n1, addr) = node1(None, host.then_some(GATE_HOST_ROOM))?;
        let mut n0 = node0(&["--hold0".to_string(), gib.to_string()], addr, host)?;
        n0.wait_line(ARMED, Duration::from_secs(600))?;
        n1.signal("-STOP")?;
        n0.say("step\n")?;
        let t0 = Instant::now();
        let held = n0
            .wait_line(HELD, Duration::from_secs_f64(LOST_BOUND_S))
            .unwrap_or_else(|e| e.to_string());
        let secs = t0.elapsed().as_secs_f64();
        let failed = held.contains("Err(")
            && held.contains("passed its deadline")
            && held.contains("the card is lost")
            && held.contains(&addr.to_string());
        println!(
            "{} stopped node 1: node 0's step gave, in {secs:.1} s: {held}",
            if failed { "ok" } else { "FAIL" }
        );
        n1.signal("-CONT")?;
        let t0 = Instant::now();
        let status = n1.wait_within(Duration::from_secs_f64(LOST_BOUND_S))?;
        let secs = t0.elapsed().as_secs_f64();
        let err = n1.stderr();
        let named = err.contains("node 0 at") && err.contains("is lost");
        let released = err.contains("the tier released its words and its stream drained");
        let ended = status.is_some_and(|s| !s.success()) && named && released;
        println!(
            "{} stopped node 1: continued with node 0's failed load held, node 1 ended {status:?} \
             {secs:.1} s after, the loss named {named}, its words released and its stream \
             drained {released}",
            if ended { "ok" } else { "FAIL" }
        );
        n0.signal("-KILL")?;
        n0.wait_within(Duration::from_secs(60))?;
        Ok(failed && ended)
    }

    /// Remove a saved run; an early node failure may have written none.
    fn remove_run(path: &Path) -> std::io::Result<()> {
        match std::fs::remove_file(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            r => r,
        }
    }

    fn gate(c: &Common) -> Result<(), GateError> {
        let mut pass = true;
        tier::sc("census: node 1's census over the channel is the plan's tier card")?;
        let (mut n1, addr) = node1(None, c.node_host.then_some(GATE_HOST_ROOM))?;
        let mut machine = None;
        let (plan, gib, census_ok) = census_plan(c, &mut machine, addr)?;
        pass &= census_ok;
        let layers = tier_layers(&plan);

        let file = Split::open(&c.path).map_err(|e| format!("open {}: {e}", c.path.display()))?;
        tier::sc("open: node 1 refuses a segment of other experts than the tier's set")?;
        pass &= swapped_open(c, &plan, &file, addr)?;
        tier::sc("open: node 1 refuses a copy of the file whose tier segment holds other bytes")?;
        pass &= other_bytes_open(&plan, &file, c.node_host)?;
        tier::sc(
            "bind: node 1 refuses a symlink and leaves its target unchanged after disconnect",
        )?;
        pass &= symlink_bind(c, &plan, &file)?;
        drop(file);

        tier::sc("bits: node 0 with the tier on node 1 = one process, bit for bit")?;
        let mut one = c.open(&plan)?;
        let want = q38_history::history(&mut one, &c.ids, |_| Ok(()))?;
        pass &= tier_structure(&mut one, &layers)?;
        drop(one);
        let out = std::env::temp_dir().join(format!("{NAME}-{}.run", std::process::id()));
        let args = [
            "--node0".to_string(),
            gib.to_string(),
            out.display().to_string(),
            n1.child.id().to_string(),
        ];
        let mut n0 = node0(&args, addr, c.node_host)?;
        let status = n0
            .wait_within(Duration::from_secs(1800))?
            .ok_or("node 0 did not end within 1800 s")?;
        let node0_ok = status.success();
        println!("node 0 ended {status}: its clauses {}", verdict(node0_ok));
        pass &= node0_ok;
        match read_run(&out) {
            Ok(got) => {
                pass &= same("node 0 and node 1", &want, &got);
                let stage = got.stage_nodes == want.stage_nodes;
                println!(
                    "structure: node 0's stage graphs hold {:?} nodes, the one-process load's {:?}: {}",
                    got.stage_nodes,
                    want.stage_nodes,
                    verdict(stage)
                );
                pass &= stage;
            }
            Err(e) => {
                println!("FAIL node 0 and node 1: no run read back: {e}");
                pass = false;
            }
        }
        remove_run(&out).map_err(|e| format!("remove {}: {e}", out.display()))?;
        let killed = n1.wait_within(Duration::from_secs(60))?;
        println!("node 1 ended {killed:?} (killed by node 0's lost node clause)");

        tier::sc("stopped node 1: node 0's step fails; continued, node 1 ends")?;
        pass &= stopped_node1(gib, c.node_host)?;
        tier::sc("lost node 0: killed, node 1 releases its words and drains")?;
        pass &= lost_node0(gib, "-KILL", "killed", c.node_host)?;
        tier::sc("lost node 0: stopped, node 1 releases its words; continued, its step fails")?;
        pass &= lost_node0(gib, "-STOP", "stopped", c.node_host)?;

        let (ran, deferred) = tier::tally();
        println!(
            "clauses: {ran} ran, {deferred} left to the real tier (deferred(real) lines above)"
        );
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    fn measure(c: &Common, gib: u64, decode: Option<(usize, &Path)>) -> Result<(), GateError> {
        let nodes = bloomery_levers::tier_nodes()?;
        let (arm, machine, at) = match nodes.as_slice() {
            [] => {
                let own = bloomery_gpu::census()?;
                let card = own.first().ok_or("no local tier card")?;
                ("local", c.machine(spec_of_device(card))?, None)
            }
            [addr] => ("node", c.machine_placed()?, Some(*addr)),
            _ => return Err("tier cost needs exactly one tier".into()),
        };
        let plan = if decode.is_some() {
            c.inputs.plan_with(
                &machine,
                plan_shape()?,
                &PlanLevers {
                    card_budget_bytes: Some(gib << 30).into(),
                },
                Experts::Card,
            )?
        } else {
            plan_under(c, &machine, gib)?.map_err(|why| format!("under {gib} GiB: {why}"))?
        };
        let identity = format!(
            "ctx={CTX} ub={} stage={:?} tier={:?} host={} host_bytes={}",
            c.ub, plan.n_l, plan.tier_n_l, plan.host.experts, plan.host.expert_bytes
        );
        let mut s = c.open(&plan)?;
        let loaded_at = s
            .model()
            .body(NAME)?
            .hybrid()
            .tiers()
            .first()
            .ok_or("no loaded tier")?
            .node();
        if loaded_at != at {
            return Err(format!("{arm} loaded tier at {loaded_at:?}").into());
        }
        let warm = q38_history::history(&mut s, &c.ids, |_| Ok(()))?;
        if !tier_structure(&mut s, &tier_layers(&plan))? {
            return Err("tier structure or routing precondition failed".into());
        }
        let apps = Command::new("nvidia-smi")
            .args(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
            .output()?;
        if !apps.status.success() {
            return Err(format!("nvidia-smi: {}", String::from_utf8_lossy(&apps.stderr)).into());
        }
        let gpu_processes = std::str::from_utf8(&apps.stdout)?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()?
            .len();
        let expected = if at.is_some() {
            let local = Command::new("nvidia-smi")
                .args(["--query-gpu=uuid", "--format=csv,noheader,nounits"])
                .output()?;
            if !local.status.success() {
                return Err(
                    format!("nvidia-smi: {}", String::from_utf8_lossy(&local.stderr)).into(),
                );
            }
            let device = machine.tiers[0]
                .device
                .ok_or("the tier census has no device UUID")?;
            let uuid = bloomery_gpu_gates::gpu_census::gpu_uuid(&device.uuid);
            if std::str::from_utf8(&local.stdout)?
                .lines()
                .any(|line| line.trim() == uuid.as_str())
            {
                2
            } else {
                1
            }
        } else {
            1
        };
        if gpu_processes != expected {
            return Err(
                format!("{arm} needs {expected} GPU processes, saw {gpu_processes}").into(),
            );
        }
        s.clear()?;
        let n = decode.map_or(96, |(n, _)| n);
        let mut timed = Run {
            tokens: Vec::with_capacity(PROMPT + n),
            ..Run::default()
        };
        let step = |s: &mut Session<Body38>, run: &mut Run, id| -> Result<u32, GateError> {
            if decode.is_none() {
                q38_history::step(s, run, id)
            } else {
                let next = runtime::Target::step(s, id, runtime::Want::Argmax)?.argmax();
                run.tokens.push(next);
                Ok(next)
            }
        };
        let mut next = 0;
        for &id in &c.ids[..PROMPT] {
            next = step(&mut s, &mut timed, id)?;
        }
        let counts = |s: &Session<Body38>| -> Result<(u64, u64, Vec<Vec<u64>>), GateError> {
            Ok(s.model().body(NAME)?.hybrid().tiers().iter().fold(
                (0, 0, Vec::new()),
                |(issued, control, mut hits), t| {
                    let stats = t.stats();
                    hits.push(stats.layer_hits);
                    (
                        issued + stats.issued,
                        control + stats.control_round_trips,
                        hits,
                    )
                },
            ))
        };
        let before = counts(&s)?;
        let start = Instant::now();
        for _ in 0..n {
            next = step(&mut s, &mut timed, next)?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let ms = elapsed * 1000.0;
        let after = counts(&s)?;
        let mut read_terms = None;
        let mut wire = Vec::new();
        if decode.is_some() {
            read_terms = Some(Terms::of(&plan, u64::from(c.inputs.spec.hidden))?);
            let tiers = s.model().body(NAME)?.hybrid().tiers();
            for tier in tiers {
                let layout = tier.page_layout(tiers.len())?;
                wire.push((tier.cols(), layout.wire_bytes(Chain::Step.cols())));
            }
        }
        record::Record::new(&record::TIER_COST)
            .w("arm", arm)
            .w("plan", identity)
            .list("history", &warm.tokens)
            .list("tokens", &timed.tokens)
            .u("prompt", PROMPT)
            .u("decode", n)
            .f("tg_tps", n as f64 * 1000.0 / ms)
            .f("step_ms", ms / n as f64)
            .u("gpu_processes", gpu_processes)
            .u("budget_gib", gib)
            .f("elapsed_s", elapsed)
            .u("control_round_trips", after.1 - before.1)
            .u("handoffs", after.0 - before.0)
            .w("read_terms", format!("{read_terms:?}"))
            .w("wire", format!("{wire:?}"))
            .w("hits_before", format!("{:?}", before.2))
            .w("hits_after", format!("{:?}", after.2))
            .print();
        if let Some((_, out)) = decode {
            write_run(&timed, out)?;
            write_run(&warm, &out.with_extension("history.run"))?;
        }
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            CARD_BUDGET,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
        ])?;
        record::at_main(NAME, record::GATE_QWEN38_NODES);
        let (mode, host) = parse_args(&std::env::args().skip(1).collect::<Vec<_>>())?;
        let mut c = Common::new(&levers)?;
        c.node_host = host;
        match mode {
            Mode::Gate => {
                if !bloomery_levers::tier_nodes()?.is_empty() {
                    return Err(
                        format!("{TIER_NODES} is set: the gate sets it for its node 0").into(),
                    );
                }
                gate(&c)
            }
            Mode::Node0 { gib, out, server } => {
                if run_history(&c, gib, &out, Some(server))? {
                    Ok(())
                } else {
                    Err(checks_failed())
                }
            }
            Mode::History { gib, out } => {
                if run_history(&c, gib, &out, None)? {
                    Ok(())
                } else {
                    Err(checks_failed())
                }
            }
            Mode::Decode { gib, tokens, out } => measure(&c, gib, Some((tokens, &out))),
            Mode::Hold0 { gib } => run_hold0(&c, gib),
            Mode::Measure { arm, gib } => {
                if !bloomery_levers::tier_nodes()?.is_empty() {
                    return Err(
                        format!("unset {TIER_NODES}: the harness owns its loopback node").into(),
                    );
                }
                if arm == "local" {
                    measure(&c, gib, None)
                } else {
                    let room = bloomery_levers::nvtier_levers()?.host_room;
                    let (_server, addr) = node1(None, room)?;
                    let mut child = node0(
                        &["--measure-node0".into(), gib.to_string()],
                        addr,
                        c.node_host,
                    )?;
                    let line = child.wait_line("tier cost ", Duration::from_secs(900))?;
                    if !child
                        .wait_within(Duration::from_secs_f64(LOST_BOUND_S))?
                        .is_some_and(|s| s.success())
                    {
                        return Err("tier cost child did not exit successfully".into());
                    }
                    println!("{line}");
                    Ok(())
                }
            }
            Mode::MeasureNode0 { gib } => {
                lever_addr()?;
                measure(&c, gib, None)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn node_host_room_is_explicit_or_inherited() {
            for room in [None, Some(0), Some(super::GATE_HOST_ROOM), Some(32 << 30)] {
                let cmd = super::node1_command(None, room).unwrap();
                let actual = cmd
                    .get_envs()
                    .find(|(name, _)| *name == bloomery_levers::HOST_ROOM)
                    .map(|(_, value)| value.unwrap().to_str().unwrap());
                let expected = room.map(|bytes| bytes.to_string());
                assert_eq!(actual, expected.as_deref());
            }
        }

        #[test]
        fn host_modes_never_ignore_the_requested_tier() {
            for args in [
                vec![],
                vec!["--node0", "1", "out", "1"],
                vec!["--hold0", "1"],
                vec!["--history", "1", "out"],
                vec!["--measure", "local", "1"],
                vec!["--measure", "node", "1"],
                vec!["--measure-node0", "1"],
                vec!["--decode0", "1", "1", "out"],
            ] {
                let args: Vec<String> = args.into_iter().map(String::from).collect();
                let (resident, host) = super::parse_args(&args).unwrap();
                assert!(!host);
                let mut hosted = vec!["--host".to_string()];
                hosted.extend(args);
                let parsed = super::parse_args(&hosted);
                if matches!(resident, super::Mode::Decode { .. }) {
                    let error = parsed.err().expect("--host --decode0 must be refused");
                    assert_eq!(
                        error.to_string(),
                        "--host is not supported with --decode0; use --history"
                    );
                } else {
                    let (mode, host) = parsed.unwrap();
                    assert!(host);
                    assert_eq!(
                        std::mem::discriminant(&mode),
                        std::mem::discriminant(&resident)
                    );
                }
            }
        }

        use super::{
            Command, Duration, Instant, NAME, PlaceError, PlacementError, Proc, census_plan,
            plan_refusal, remove_open_copy, remove_run,
        };
        use bloomery_gpu::GpuError;
        use bloomery_gpu_gates::GateError;

        #[test]
        #[ignore = "needs a remote tier server and the model on both nodes"]
        fn hw_a_refused_remote_replay_resets_to_its_fresh_history() -> Result<(), GateError> {
            let levers = bloomery_levers::at_main(&[
                super::CARD_BUDGET,
                super::HOST_POPULATE,
                super::HOST_LOCK,
                super::CARD_DONTNEED,
                bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
            ])?;
            let c = super::Common::new(&levers)?;
            let addr = super::lever_addr()?;
            assert!(
                !addr.ip().is_loopback(),
                "the regression needs separate nodes"
            );
            let mut machine = None;
            let (plan, _, independent) = super::census_plan(&c, &mut machine, addr)?;
            assert!(independent, "the node budgets must use their own census");
            let mut s = c.open(&plan)?;
            let mut fresh = super::q38_history::history(&mut s, &c.ids, |_| Ok(()))?;
            assert!(super::tier_structure(&mut s, &super::tier_layers(&plan))?);
            let ready = s.model().body(NAME)?.hybrid().tiers()[0]
                .stats()
                .control_round_trips;
            s.model_mut()
                .body_parts(NAME)?
                .2
                .hybrid_mut()
                .plant_refusal("remote replay refusal");
            let error = s
                .model_mut()
                .step(&[c.ids[0]])
                .expect_err("a planted refusal");
            assert!(
                error.to_string().contains("remote replay refusal"),
                "{error}"
            );
            assert!(!matches!(error, GpuError::Unsettled { .. }), "{error}");
            let before = s.model().body(NAME)?.hybrid().tiers()[0]
                .stats()
                .control_round_trips;
            assert_eq!(
                before - ready,
                2,
                "one replay launch and one remote release"
            );
            s.clear()?;
            let after = s.model().body(NAME)?.hybrid().tiers()[0]
                .stats()
                .control_round_trips;
            assert_eq!(after - before, 4, "three reset phases and one fault clear");
            let mut recovered = super::q38_history::Run::default();
            super::q38_history::step(&mut s, &mut recovered, c.ids[0])?;
            fresh.tokens.truncate(1);
            fresh.logits.truncate(1);
            assert!(super::q38_history::same(
                "first decode after refusal reset",
                &fresh,
                &recovered
            ));
            Ok(())
        }

        // PIN(2026-10-23): the census clause runs with model and corpus fixtures in gate-gpu-qwen38-nodes.
        #[test]
        #[ignore = "needs the Qwen model and corpus files"]
        fn hw_census_clause_requires_the_remote_census() -> Result<(), GateError> {
            let levers = bloomery_levers::at_main(&[
                super::CARD_BUDGET,
                super::HOST_POPULATE,
                super::HOST_LOCK,
                super::CARD_DONTNEED,
                bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
            ])?;
            let c = super::Common::new(&levers)?;
            let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("an address");
            let addr = listener.local_addr().expect("the address");
            drop(listener);
            let Err(err) = census_plan(&c, &mut None, addr) else {
                panic!("the plan requires the remote census");
            };
            assert!(
                err.to_string().contains(&format!("the node at {addr}:")),
                "no local census replaces the remote: {err}"
            );
            Ok(())
        }

        #[test]
        fn children_die_with_the_gate_parent() -> Result<(), GateError> {
            // A nonmatching skip filter distinguishes the helper subprocess.
            const CHILD_MODE: &str = "q38_gate_parent_child";
            if std::env::args().any(|a| a == CHILD_MODE) {
                let mut cmd = Command::new("sleep");
                cmd.arg("120");
                let running = Proc::spawn(cmd, "node 0")?;
                let mut cmd = Command::new("sleep");
                cmd.arg("120");
                let stopped = Proc::spawn(cmd, "node 1")?;
                stopped.signal("-STOP")?;
                println!("children: {} {}", running.child.id(), stopped.child.id());
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                }
            }
            let mut cmd = Command::new(std::env::current_exe()?);
            cmd.args([
                "--exact",
                "gate::tests::children_die_with_the_gate_parent",
                "--nocapture",
                "--format=terse",
                "--skip",
                CHILD_MODE,
            ]);
            let mut parent = Proc::spawn(cmd, "gate parent")?;
            let line = parent.wait_line("children: ", Duration::from_secs(10))?;
            let pids = line["children: ".len()..]
                .split_whitespace()
                .map(str::parse::<u32>)
                .collect::<Result<Vec<_>, _>>()?;
            assert_eq!(pids.len(), 2, "{line}");
            parent.child.kill()?;
            let ended = parent.wait_within(Duration::from_secs(10))?;
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut alive = pids;
            while !alive.is_empty() && Instant::now() < deadline {
                alive.retain(
                    |pid| match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                        Ok(stat) => !stat
                            .rsplit_once(") ")
                            .is_some_and(|(_, tail)| tail.starts_with(['Z', 'X'])),
                        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
                    },
                );
                if !alive.is_empty() {
                    std::thread::sleep(Duration::from_millis(25));
                }
            }
            for pid in &alive {
                let status = Command::new("kill")
                    .args(["-KILL", &pid.to_string()])
                    .status()?;
                assert!(status.success(), "cleanup child {pid}: {status}");
            }
            assert!(ended.is_some(), "the gate parent did not exit");
            assert!(
                alive.is_empty(),
                "children outlived the gate parent: {alive:?}"
            );
            Ok(())
        }

        #[test]
        fn child_never_executes_after_parent_dies_before_registration() -> Result<(), GateError> {
            use std::os::unix::process::CommandExt;

            const CHILD_MODE: &str = "q38_gate_before_registration";
            if std::env::args().any(|a| a == CHILD_MODE) {
                let task = std::fs::read_link("/proc/thread-self")?;
                println!("children file: /proc/{}/children", task.display());
                let mut cmd = Command::new("echo");
                cmd.arg("server executed")
                    .stderr(std::process::Stdio::null());
                let stop = || {
                    // SAFETY: raise stops only this child.
                    if unsafe { libc::raise(libc::SIGSTOP) } != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                };
                // SAFETY: stop calls only async-signal-safe raise and allocates nothing.
                unsafe { cmd.pre_exec(stop) };
                bloomery_gpu_gates::die_with_parent(&mut cmd);
                return Err(format!("server executed: {}", cmd.status()?).into());
            }
            let mut cmd = Command::new(std::env::current_exe()?);
            cmd.args([
                "--exact",
                "gate::tests::child_never_executes_after_parent_dies_before_registration",
                "--nocapture",
                "--format=terse",
                "--skip",
                CHILD_MODE,
            ]);
            let mut parent = Proc::spawn(cmd, "gate parent")?;
            let mut child = None;
            let result = (|| -> Result<(), GateError> {
                let line = parent.wait_line("children file: ", Duration::from_secs(10))?;
                let children = &line["children file: ".len()..];
                let deadline = Instant::now() + Duration::from_secs(5);
                let mut stopped = false;
                while Instant::now() < deadline {
                    let pids = std::fs::read_to_string(children)?;
                    if let Some(pid) = pids.split_whitespace().next() {
                        child = Some(pid.parse::<u32>()?);
                        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
                        if stat
                            .rsplit_once(") ")
                            .is_some_and(|(_, tail)| tail.starts_with('T'))
                        {
                            stopped = true;
                            break;
                        }
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
                if !stopped {
                    return Err("the child did not stop before registration".into());
                }
                parent.child.kill()?;
                if parent.wait_within(Duration::from_secs(10))?.is_none() {
                    return Err("the gate parent did not exit".into());
                }
                let pid = child.ok_or("the gate parent did not fork")?;
                let status = Command::new("kill")
                    .args(["-CONT", &pid.to_string()])
                    .status()?;
                if !status.success() {
                    return Err(format!("continue child {pid}: {status}").into());
                }
                let output = parent.lines.recv_timeout(Duration::from_secs(5));
                if !matches!(output, Err(std::sync::mpsc::RecvTimeoutError::Disconnected)) {
                    return Err(
                        format!("the server executed or kept stdout open: {output:?}").into(),
                    );
                }
                Ok(())
            })();
            if let Some(pid) = child {
                let alive = match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                    Ok(stat) => !stat
                        .rsplit_once(") ")
                        .is_some_and(|(_, tail)| tail.starts_with(['Z', 'X'])),
                    Err(e) => e.kind() != std::io::ErrorKind::NotFound,
                };
                if alive {
                    let status = Command::new("kill")
                        .args(["-KILL", &pid.to_string()])
                        .status()?;
                    assert!(status.success(), "cleanup child {pid}: {status}");
                }
            }
            result
        }

        #[test]
        fn plan_refusals_retry_only_budget_candidates() {
            for (error, retry) in [
                (
                    PlacementError::DeviceTwice {
                        card: "shared".into(),
                        cards: 2,
                        budgets: 32 << 30,
                        usable: 31 << 30,
                    }
                    .into(),
                    true,
                ),
                (
                    PlacementError::IdleTier {
                        card: "tier".into(),
                        tier: 0,
                    }
                    .into(),
                    true,
                ),
                (PlaceError::Positions { ctx_max: 0 }, false),
            ] {
                let why = match &error {
                    PlaceError::Placement(PlacementError::IdleTier { .. }) => {
                        "the tier holds no expert".to_string()
                    }
                    _ => error.to_string(),
                };
                match plan_refusal(error, 16) {
                    Ok(reason) => assert!(retry && reason == why, "{reason}"),
                    Err(e) => {
                        assert!(!retry, "{e}");
                        assert_eq!(e.to_string(), format!("plan (b′) under 16 GiB: {why}"));
                    }
                }
            }
        }

        #[test]
        fn run_cleanup_accepts_only_missing_files() {
            let dir = std::env::temp_dir().join(format!("{NAME}-cleanup-{}", std::process::id()));
            std::fs::create_dir(&dir).expect("the cleanup directory");
            let out = dir.join("node0.run");
            remove_run(&out).expect("an early node failure wrote no run");
            std::fs::write(&out, b"a saved run").expect("the saved run");
            remove_run(&out).expect("the saved run removed");
            assert!(!out.exists());
            assert!(remove_run(&dir).is_err(), "a directory is not a run file");
            std::fs::remove_dir(&dir).expect("the cleanup directory removed");
        }

        #[test]
        fn copy_cleanup_keeps_the_open_error() {
            let path =
                std::env::temp_dir().join(format!("{NAME}-copy-cleanup-{}", std::process::id()));
            std::fs::write(&path, b"a file, not a directory").expect("the cleanup blocker");
            for (r, open) in [
                (Err("copy setup refused".into()), Some("copy setup refused")),
                (
                    Ok(Err(GpuError::Shape {
                        what: "open",
                        detail: "tier refused".into(),
                    })),
                    Some("tier refused"),
                ),
                (Ok(Ok(())), None),
            ] {
                let err = remove_open_copy(&path, r)
                    .expect_err("a file cannot be removed as a directory")
                    .to_string();
                assert!(
                    err.contains(&format!("remove {}:", path.display())),
                    "{err}"
                );
                if let Some(open) = open {
                    assert!(err.contains(open), "{err}");
                }
            }
            std::fs::remove_file(&path).expect("the cleanup blocker removed");
            std::fs::create_dir(&path).expect("the copied model directory");
            let opened = remove_open_copy(
                &path,
                Ok(Err(GpuError::Shape {
                    what: "open",
                    detail: "tier refused".into(),
                })),
            )
            .expect("the copy removed");
            assert!(
                opened
                    .expect_err("the open refusal kept")
                    .to_string()
                    .contains("tier refused")
            );
            assert!(!path.exists());
        }
    }
}
