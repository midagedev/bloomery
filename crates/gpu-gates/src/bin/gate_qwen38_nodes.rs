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
//! - Open: node 0's open of the plan's tier with one routed segment's
//!   experts swapped for as many others is refused by node 1, naming the
//!   tier's set, and node 1 serves the next connection. A second node 1
//!   serving a copy of the model file whose headers are the file's and
//!   whose one tier segment holds one other byte refuses the plan's tier,
//!   naming the segment's other bytes. The copy is the shard that segment
//!   is in, the smallest such shard, under `$BLOOMERY_DATA`, beside links to
//!   the other shards; it is removed after.
//! - Bits: the history (`q38_history`) on node 0 with the tier on node 1,
//!   every token and logits row bit for bit the same plan's history in one
//!   process, the tier a card of that process. Structure: node 0's stage
//!   graphs hold the
//!   one-process load's nodes; the tier's graphs of the step and of each
//!   verify width, captured on node 1, hold [`TIER_LAYER_NODES`] a tier layer
//!   and the fault copy; each chain fed and captured once and replayed by
//!   every later pass. Precondition: node 1 holds the tier's experts and
//!   every tier layer was sent a routed slot.
//! - Lost node 1: node 1 killed after the history, node 0's next step fails
//!   within the go deadline and its grace naming the node and the lost card,
//!   the host tier is poisoned as a lost card, and the next step and a reset
//!   are refused.
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
    use bloomery_gpu::arch::qwen3moe::NODE_FAMILY;
    use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
    use bloomery_gpu::arch::qwen3moe::{Body38, Qwen38Model, SplitWalk};
    use bloomery_gpu::host::PoisonKind;
    use bloomery_gpu::host::node::wire::{In, Out as Payload, Tag};
    use bloomery_gpu::host::node::{NodeOpen, TierNode, node_cards};
    use bloomery_gpu::host::slots::SlotMap;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::host::tier::{TierAct, TierCard, TierOpen, TierSet, TierShape};
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{
        GateError, beside, checks_failed, data_dir, prose_ids, ref_model_path, verdict,
    };
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, HostCfg, Levers, TIER_NODES,
    };
    use gguf::Split;
    use model::arch::qwen35moe::place::{
        Experts, PlaceError, PlanInputs, machine_bp_on, tier_batch,
    };
    use model::placement::devices::spec_of_device;
    use model::placement::workstation::CardSpec;
    use model::placement::{Machine, PlacementError, Plan, PlanLevers};

    use crate::q38_history::{self, PROMPT, Run, same, tier_graphs_and_chains, tier_layers};

    const NAME: &str = "gate_qwen38_nodes";
    /// The stores' positions, `gate_qwen38_twocard`'s.
    const CTX: usize = 3072;
    /// The largest card budget the plan search tries, GiB: node 0's stage
    /// and node 1's tier share this machine's first card, so two budgets
    /// of it fit a 48 GB card.
    const TOP_GIB: u64 = 16;
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
        /// Node 0 of the bits and lost node 1 clauses: the plan under `gib`,
        /// its run written to `out`, node 1's process `server` killed after.
        Node0 {
            gib: u64,
            out: PathBuf,
            server: u32,
        },
        /// A node 0 that holds its stage's stream and asks a step.
        Hold0 {
            gib: u64,
        },
    }

    fn parse_args() -> Result<Mode, GateError> {
        const USAGE: &str = "usage: gate_qwen38_nodes (node 0 modes, run by the gate: --node0 \
                             <gib> <out> <server pid> | --hold0 <gib>)";
        let args: Vec<String> = std::env::args().skip(1).collect();
        let gib = |s: &str| s.parse::<u64>().map_err(|e| format!("{s:?}: {e}: {USAGE}"));
        Ok(
            match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
                [] => Mode::Gate,
                ["--node0", g, out, server] => Mode::Node0 {
                    gib: gib(g)?,
                    out: PathBuf::from(out),
                    server: server
                        .parse()
                        .map_err(|e| format!("{server:?}: {e}: {USAGE}"))?,
                },
                ["--hold0", g] => Mode::Hold0 { gib: gib(g)? },
                _ => return Err(USAGE.into()),
            },
        )
    }

    /// What every mode reads: the file, its plan inputs, the ubatch and the
    /// ids.
    struct Common {
        path: PathBuf,
        inputs: PlanInputs,
        host: HostCfg,
        ub: usize,
        ids: Vec<u32>,
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
            })
        }

        /// Plan (b′)'s machine: this process's first card the stage, `node`
        /// the tier.
        fn machine(&self, node: CardSpec) -> Result<Machine, GateError> {
            let own = bloomery_gpu::census()?;
            let stage = own.first().ok_or("this process sees no card")?;
            let layers = self.inputs.spec.layers.len();
            let ub = self.ub as u64;
            Ok(machine_bp_on(
                (spec_of_device(stage), node),
                layers,
                ub,
                None,
                tier_batch(&self.inputs.hp, ub),
            ))
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

    /// Plan (b′) on `machine` under `gib` GiB a card: the plan, or why it is
    /// not the gate's: a layer with no tier expert, an idle tier, or no host
    /// leg before the last layer (a stopped node 0 strands its card only at
    /// a host leg, so the stopped clause needs one ahead of a tier layer).
    fn plan_under<'a>(
        inputs: &'a PlanInputs,
        machine: &'a Machine,
        gib: u64,
    ) -> Result<Result<Plan<'a>, String>, GateError> {
        let levers = PlanLevers {
            card_budget_bytes: Some(gib << 30),
        };
        match inputs.plan_with(machine, CTX as u64, &levers, Experts::Card) {
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
            Err(PlaceError::Placement(PlacementError::IdleTier { .. })) => {
                Ok(Err("the tier holds no expert".to_string()))
            }
            Err(e) => Err(format!("plan (b′) under {gib} GiB: {e}").into()),
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
    /// and the precondition that the tier holds its experts and every tier
    /// layer was sent a routed slot.
    fn tier_structure(s: &Session<Body38>, layers: &[usize]) -> Result<bool, GateError> {
        let body = s.model().body(NAME)?;
        let tier = body
            .hybrid()
            .tiers()
            .first()
            .ok_or("the load holds no tier")?;
        let ok = tier_graphs_and_chains(tier, layers.len());
        let st = tier.stats();
        let first = tier.set().layers().start;
        let hits = |l: usize| st.layer_hits.get(l - first).copied().unwrap_or(0);
        let missing: Vec<usize> = layers.iter().copied().filter(|&l| hits(l) == 0).collect();
        let pre = missing.is_empty() && tier.set().experts() > 0 && tier.resident_bytes() > 0;
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

    /// Node 1: a tier server on a free loopback port, serving `model` when
    /// one is given, and its address.
    fn node1(model: Option<&Path>) -> Result<(Proc, SocketAddr), GateError> {
        let mut cmd = Command::new(beside("tier_server")?);
        cmd.args(["--listen", "127.0.0.1:0"])
            .env_remove(TIER_NODES)
            .env_remove(CARD_BUDGET)
            .env_remove(HOST_POPULATE)
            .env_remove(HOST_LOCK)
            .env_remove(CARD_DONTNEED);
        if let Some(m) = model {
            cmd.arg("-m").arg(m);
        }
        let p = Proc::spawn(cmd, "node 1")?;
        let line = p.wait_line("listening on ", Duration::from_secs(120))?;
        let addr = line["listening on ".len()..].parse()?;
        Ok((p, addr))
    }

    /// Node 0 as a child of this binary in `mode`'s arguments, with the
    /// lever naming `addr`.
    fn node0(args: &[String], addr: SocketAddr) -> Result<Proc, GateError> {
        let mut cmd = Command::new(std::env::current_exe()?);
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

    /// The census clause: node 1's census over the channel, and the plan of
    /// its card; the plan and its budget.
    fn census_plan<'a>(
        c: &'a Common,
        machine: &'a mut Option<Machine>,
        addr: SocketAddr,
    ) -> Result<(Plan<'a>, u64, bool), GateError> {
        let (spec, free) = node_card(addr)?;
        let m = machine.insert(c.machine(spec)?);
        let m: &'a Machine = m;
        let tier = m
            .tiers
            .first()
            .ok_or("plan (b′)'s machine holds no tier card")?;
        let mut ok = tier.free_bytes == Some(free);
        println!(
            "census: the plan's tier card {:?} takes the free bytes node 1 sent ({:?} = {free} B): \
             {}",
            tier.name,
            tier.free_bytes,
            verdict(ok)
        );
        for gib in (1..=TOP_GIB).rev() {
            match plan_under(&c.inputs, m, gib)? {
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

    /// The plan a node 0 child loads: node 1's census at `addr` and `gib`.
    fn child_plan<'a>(
        c: &'a Common,
        machine: &'a mut Option<Machine>,
        addr: SocketAddr,
        gib: u64,
    ) -> Result<Plan<'a>, GateError> {
        let (spec, _) = node_card(addr)?;
        let m: &'a Machine = machine.insert(c.machine(spec)?);
        plan_under(&c.inputs, m, gib)?.map_err(|why| format!("under {gib} GiB: {why}").into())
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
    /// `addr`, one routed segment's experts swapped for as many others not
    /// in the set, is refused naming the set.
    fn swapped_open(plan: &Plan<'_>, file: &Split, addr: SocketAddr) -> Result<bool, GateError> {
        let mut open = tier_open(plan, file)?;
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

    /// The open clause on other bytes: a copy of `file` under
    /// `$BLOOMERY_DATA` whose one tier segment's first byte is flipped, its
    /// headers `file`'s, served by a second node 1; the plan's tier opened
    /// on it is refused naming the segment's other bytes.
    fn other_bytes_open(plan: &Plan<'_>, file: &Split) -> Result<bool, GateError> {
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
            let (_n1, addr) = node1(Some(&copy))?;
            Ok(TierNode::open(addr, &open).map(drop))
        });
        drop(std::fs::remove_dir_all(&dir));
        let r = r?;
        let ok = matches!(&r, Err(e) if e.to_string().contains("hold other bytes")
            && e.to_string().contains(&row.name));
        println!(
            "{} open: node 1 on the copy answered the plan's tier with {r:?}",
            if ok { "ok" } else { "FAIL" }
        );
        Ok(ok)
    }

    /// The address `BLOOMERY_TIER_NODES` names, the one a child reads.
    fn lever_addr() -> Result<SocketAddr, GateError> {
        match bloomery_levers::tier_nodes()?[..] {
            [a] => Ok(a),
            ref a => Err(format!("{TIER_NODES} names {} addresses, the gate one", a.len()).into()),
        }
    }

    /// `--node0`: the history on node 0 with node 1's tier, written to
    /// `out`; the tier's structure; then node 1 killed and the lost node
    /// clause.
    fn run_node0(c: &Common, gib: u64, out: &Path, server: u32) -> Result<bool, GateError> {
        let addr = lever_addr()?;
        let mut machine = None;
        let plan = child_plan(c, &mut machine, addr, gib)?;
        let layers = tier_layers(&plan);
        let mut s = c.open(&plan)?;
        let on = s
            .model()
            .body(NAME)?
            .hybrid()
            .tiers()
            .first()
            .and_then(TierCard::node);
        let mut ok = on == Some(addr);
        println!(
            "node 0: the tier is served by the node at {on:?}: {}",
            verdict(ok)
        );
        let run = q38_history::history(&mut s, &c.ids, |_| Ok(()))?;
        write_run(&run, out)?;
        ok &= tier_structure(&s, &layers)?;
        ok &= lost_node1(s.model_mut(), &c.ids, addr, server)?;
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
    /// stdin and runs one more step, whose launch the relay holds; this
    /// process is killed or stopped inside it.
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
            let mut len = [0u8; 4];
            let mut frame = Vec::new();
            let read = down.read_exact(&mut len).and_then(|()| {
                frame.resize(u32::from_le_bytes(len) as usize, 0);
                down.read_exact(&mut frame)
            });
            if read.is_err() {
                drop(up.shutdown(Shutdown::Write));
                return;
            }
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
    fn lost_node0(gib: u64, sig: &str, what: &str) -> Result<bool, GateError> {
        let (mut n1, addr) = node1(None)?;
        let relay = Relay::start(addr)?;
        let mut n0 = node0(&["--hold0".to_string(), gib.to_string()], relay.addr)?;
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

    fn gate(c: &Common) -> Result<(), GateError> {
        let mut pass = true;
        tier::sc("census: node 1's census over the channel is the plan's tier card")?;
        let (mut n1, addr) = node1(None)?;
        let mut machine = None;
        let (plan, gib, census_ok) = census_plan(c, &mut machine, addr)?;
        pass &= census_ok;
        let layers = tier_layers(&plan);

        let file = Split::open(&c.path).map_err(|e| format!("open {}: {e}", c.path.display()))?;
        tier::sc("open: node 1 refuses a segment of other experts than the tier's set")?;
        pass &= swapped_open(&plan, &file, addr)?;
        tier::sc("open: node 1 refuses a copy of the file whose tier segment holds other bytes")?;
        pass &= other_bytes_open(&plan, &file)?;
        drop(file);

        tier::sc("bits: node 0 with the tier on node 1 = one process, bit for bit")?;
        let mut one = c.open(&plan)?;
        let want = q38_history::history(&mut one, &c.ids, |_| Ok(()))?;
        pass &= tier_structure(&one, &layers)?;
        drop(one);
        let out = std::env::temp_dir().join(format!("{NAME}-{}.run", std::process::id()));
        let args = [
            "--node0".to_string(),
            gib.to_string(),
            out.display().to_string(),
            n1.child.id().to_string(),
        ];
        let mut n0 = node0(&args, addr)?;
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
        drop(std::fs::remove_file(&out));
        let killed = n1.wait_within(Duration::from_secs(60))?;
        println!("node 1 ended {killed:?} (killed by node 0's lost node clause)");

        tier::sc("lost node 0: killed, node 1 releases its words and drains")?;
        pass &= lost_node0(gib, "-KILL", "killed")?;
        tier::sc("lost node 0: stopped, node 1 releases its words; continued, its step fails")?;
        pass &= lost_node0(gib, "-STOP", "stopped")?;

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

    pub fn run() -> Result<(), GateError> {
        let levers =
            bloomery_levers::at_main(&[CARD_BUDGET, HOST_POPULATE, HOST_LOCK, CARD_DONTNEED])?;
        let c = Common::new(&levers)?;
        match parse_args()? {
            Mode::Gate => {
                if !bloomery_levers::tier_nodes()?.is_empty() {
                    return Err(
                        format!("{TIER_NODES} is set: the gate sets it for its node 0").into(),
                    );
                }
                gate(&c)
            }
            Mode::Node0 { gib, out, server } => {
                if run_node0(&c, gib, &out, server)? {
                    Ok(())
                } else {
                    Err(checks_failed())
                }
            }
            Mode::Hold0 { gib } => run_hold0(&c, gib),
        }
    }
}
