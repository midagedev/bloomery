//! The tensor-parallel V4.1 body (`bloomery_gpu_deepseek41::body::open_rank`) on the
//! original-format fixture folder (`<fixture root>/v41-folder`, written by
//! `model_hf::arch::deepseek41::fixture`) on the one visible card, each rank reading its part of
//! each tensor as the model's cut gives it.
//!
//! Past one rank every rank is a process of its own (this binary under [`CHILD`]), joined as a
//! deployment joins them: a rank group over the rank file's control addresses on the loopback
//! (`bloomery_gpu::host::rank::RankSession`), its gathers on TCP ends (`RemoteWire::Tcp`), its
//! rounds and picks by the group's greedy run (`app::tp::greedy`). A rank's wait for the others'
//! parts is a stream memory op; each process holds a context of its own, whose queues no other
//! rank's work stands in, so one rank's wait never holds another rank's part.
//!
//! - (n1) one rank reads the folder whole and runs [`STEPS`] greedy steps from [`FIRST`]: every
//!   logit is finite. No one-card run is its reference: the one-card body reads a GGUF split and
//!   does not open the folder. What only the folder's formats run is held elsewhere: the shared
//!   gate and up of MXFP8 or BF16 by `chain::ffn`'s `hw_shared_rows_pair_is_two_gemvs_then_swiglu`
//!   (`just gate-gpu-ds41-lib`), the routed stacks' card formats by `body`'s unit test.
//! - (n2), (n4) two and four ranks over the same steps against (n1), within the logits band
//!   (`logits_band::walk`): each sum over the ranks adds the ranks' parts of the same exact integer
//!   block sums in another f32 order than one rank's whole pass, so the logits sit near one rank's,
//!   not on its bits.
//!
#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_tp", gate::run())
}

#[cfg(not(feature = "deepseek41"))]
fn main() {}

#[cfg(feature = "deepseek41")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_rank.rs"]
mod ds41_rank;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::net::{Ipv4Addr, TcpListener};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::Duration;

    use crate::ds41_rank::rank_greedy;
    use bloomery_gpu::Gpu;
    use bloomery_gpu::host::RemoteWire;
    use bloomery_gpu::host::node::wire::{In, Out};
    use bloomery_gpu::host::rank::{Bootstrap, Fingerprint, RankList, RankSession};
    use bloomery_gpu_deepseek41::body::{self, OpenCfg, RankFrom};
    use bloomery_gpu_deepseek41::chain::join;
    use bloomery_gpu_gates::GateError;
    use bloomery_gpu_gates::logits_band::{self, Positions};
    use bloomery_levers::{CARD_BUDGET, ENGRAM_HELPER};
    use gguf::digest::Sha256;
    use model::arch::deepseek41::hparams::Hparams;
    use model_hf::folder::{self, Opened};

    /// The greedy steps each run takes.
    const STEPS: usize = 24;
    /// The token the runs start from.
    const FIRST: u32 = 1;
    /// The positions a load's stores hold.
    const CTX: usize = 64;
    /// The fixture folder under the fixture root.
    const FOLDER: &str = "v41-folder";
    /// The first argument of a rank's process: `CHILD <rank> <rank file> <out file>`.
    const CHILD: &str = "--rank-of";
    /// How long a rank waits for the others at the group's open.
    const WAIT: Duration = Duration::from_secs(120);

    /// One run's steps: each step's token and the ranks' logits laid in rank order.
    type Run = Vec<(u32, Vec<f32>)>;

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[ENGRAM_HELPER, CARD_BUDGET])?;
        crate::gate_card::init()?;
        let cfg = OpenCfg::from_levers(&levers)?;
        let dir = format!("{}/{FOLDER}", refset::fixture::root()?);
        let args: Vec<String> = std::env::args().skip(1).collect();
        if let [child, rank, list, out] = args.as_slice()
            && child == CHILD
        {
            return rank_process(&dir, &cfg, rank.parse()?, Path::new(list), Path::new(out));
        }
        let one = one_rank(&dir, &cfg)?;
        let finite = one.iter().all(|(_, l)| l.iter().all(|v| v.is_finite()));
        println!(
            "(n1) one rank, {STEPS} steps from {FIRST}: tokens {:?}, every logit finite {finite}: {}",
            one.iter().map(|s| s.0).collect::<Vec<_>>(),
            if finite { "PASS" } else { "FAIL" }
        );
        let mut pass = finite;
        for ranks in [2, 4] {
            let got = run_group(ranks)?;
            let walk = logits_band::walk(
                &format!("(n{ranks}) against (n1)"),
                positions(&one).as_view(),
                positions(&got).as_view(),
            );
            pass &= walk.ok;
        }
        if pass {
            Ok(())
        } else {
            Err("gate_deepseek41_tp: a clause failed".into())
        }
    }

    /// A run's tokens and rows, as [`logits_band::walk`] reads them.
    struct Held {
        tokens: Vec<u32>,
        rows: Vec<Vec<f32>>,
    }

    impl Held {
        fn as_view(&self) -> Positions<'_> {
            Positions {
                tokens: &self.tokens,
                rows: &self.rows,
            }
        }
    }

    fn positions(run: &Run) -> Held {
        Held {
            tokens: run.iter().map(|s| s.0).collect(),
            rows: run.iter().map(|s| s.1.clone()).collect(),
        }
    }

    /// [`STEPS`] greedy steps from [`FIRST`] on one rank that reads the folder at `dir` whole.
    fn one_rank(dir: &str, cfg: &OpenCfg) -> Result<Run, GateError> {
        let Opened {
            header,
            source,
            rows,
            ..
        } = folder::open(dir.as_ref()).map_err(|e| format!("open {dir}: {e}"))?;
        let from = RankFrom {
            file: header,
            source: &source,
            stored: Some(&rows),
        };
        let mut m = body::open_rank(Gpu::new()?, from, (0, 1), None, CTX, cfg.body)?;
        let (mut token, mut out) = (FIRST, Vec::with_capacity(STEPS));
        for _ in 0..STEPS {
            token = m.step(&[token])?;
            out.push((token, m.logits()?));
        }
        Ok(out)
    }

    /// [`STEPS`] greedy steps from [`FIRST`] on `ranks` ranks, each a process of its own
    /// ([`rank_process`]) on a rank file of loopback addresses: at each step the group's token and
    /// the ranks' logits laid in rank order. Any rank that fails fails the run, by its exit.
    fn run_group(ranks: usize) -> Result<Run, GateError> {
        let scratch =
            std::env::temp_dir().join(format!("gate_deepseek41_tp-{}-{ranks}", std::process::id()));
        std::fs::create_dir_all(&scratch)?;
        let got = group_in(&scratch, ranks);
        std::fs::remove_dir_all(&scratch)?;
        got
    }

    fn group_in(scratch: &Path, ranks: usize) -> Result<Run, GateError> {
        // A port the system has free for rank 0's control listener; every rank its own loopback
        // address, on which its TCP ends listen.
        let port = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?
            .local_addr()?
            .port();
        let list: String = (0..ranks)
            .map(|r| format!("127.0.0.{}:{port} 127.0.1.{}\n", r + 1, r + 1))
            .collect();
        let list_path = scratch.join("ranks");
        std::fs::write(&list_path, list)?;
        let exe = std::env::current_exe()?;
        let outs: Vec<PathBuf> = (0..ranks)
            .map(|r| scratch.join(format!("rank{r}")))
            .collect();
        let children = outs
            .iter()
            .enumerate()
            .map(|(r, out)| {
                Command::new(&exe)
                    .arg(CHILD)
                    .arg(r.to_string())
                    .arg(&list_path)
                    .arg(out)
                    .spawn()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let failed: Vec<String> = children
            .into_iter()
            .enumerate()
            .filter_map(|(r, mut c)| match c.wait() {
                Ok(s) if s.success() => None,
                Ok(s) => Some(format!("rank {r} of {ranks}: {s}")),
                Err(e) => Some(format!("rank {r} of {ranks}: {e}")),
            })
            .collect();
        if !failed.is_empty() {
            return Err(failed.join("; ").into());
        }
        let mut parts = outs
            .iter()
            .map(|p| read_part(&std::fs::read(p)?))
            .collect::<Result<Vec<_>, GateError>>()?;
        let tokens = std::mem::take(&mut parts[0].0);
        if tokens.len() != STEPS || parts.iter().any(|(_, l)| l.len() != STEPS) {
            return Err(format!("{ranks} ranks: not {STEPS} steps of every rank").into());
        }
        // The merged token is the argmax of the gathered row: a pick or merge that emits another
        // token fails here, not only where the reference is clear.
        tokens
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                let row: Vec<f32> = parts
                    .iter()
                    .flat_map(|(_, l)| l[i].iter().copied())
                    .collect();
                if row.len() < 2 || row.iter().any(|v| !v.is_finite()) {
                    return Err(format!("{ranks} ranks: invalid logits at step {i}").into());
                }
                let want = bloomery_gpu_gates::kld::top2(&row).0;
                if t as usize != want {
                    return Err(
                        format!("{ranks} ranks: token {t}, argmax {want} at step {i}").into(),
                    );
                }
                Ok((t, row))
            })
            .collect()
    }

    /// Rank `rank` of the rank file at `list` over the folder at `dir`: the group's open on TCP
    /// ends, then [`rank_greedy`], rank 0 feeding [`FIRST`]; the rank's
    /// logits at each step, and rank 0's tokens, written to `out` ([`write_part`]).
    fn rank_process(
        dir: &str,
        cfg: &OpenCfg,
        rank: usize,
        list: &Path,
        out: &Path,
    ) -> Result<(), GateError> {
        let list = RankList::read(list)?;
        let ranks = list.len();
        let Opened {
            header,
            source,
            rows,
            ..
        } = folder::open(dir.as_ref()).map_err(|e| format!("open {dir}: {e}"))?;
        let hp = Hparams::read(&header).map_err(|e| e.to_string())?;
        // Every rank reads the same folder: its path is the source's identity.
        let mut h = Sha256::new();
        h.update(dir.as_bytes());
        let fp = Fingerprint {
            source: h.finish(),
            headers: [0; 32],
            form: 0,
            build: bloomery_gpu_gates::rank_build()?,
            n: u16::try_from(ranks)?,
            rank: u16::try_from(rank)?,
        };
        let how = Bootstrap {
            fabric: None,
            wait: WAIT,
            wire: RemoteWire::Tcp,
        };
        let gpu = Gpu::new()?;
        let mut session =
            RankSession::open(&gpu, &list, rank, &how, &fp, &[], &join::shapes(&hp)?)?;
        let mut logits = Vec::with_capacity(STEPS);
        let prompt = if rank == 0 { vec![FIRST] } else { Vec::new() };
        let (tokens, _, _) = rank_greedy(
            gpu,
            &mut session,
            (rank, ranks),
            RankFrom {
                file: header,
                source: &source,
                stored: Some(&rows),
            },
            (CTX, cfg.body),
            (&prompt, STEPS),
            |_, l, _| logits.push(l.to_vec()),
        )?;
        std::fs::write(out, write_part(&tokens, &logits))?;
        Ok(())
    }

    /// A rank's part of a run as its process writes it: the tokens (rank 0's; none on another
    /// rank), then each step's logits as their bits.
    fn write_part(tokens: &[u32], logits: &[Vec<f32>]) -> Vec<u8> {
        let mut o = Out::default();
        o.u32s(tokens).len(logits.len());
        for l in logits {
            o.u32s(&l.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        o.done().to_vec()
    }

    /// [`write_part`]'s tokens and logits.
    fn read_part(b: &[u8]) -> Result<(Vec<u32>, Vec<Vec<f32>>), GateError> {
        let mut r = In::new(b);
        let tokens = r.u32s("tokens")?;
        let logits = (0..r.len("steps")?)
            .map(|_| Ok(r.u32s("logits")?.into_iter().map(f32::from_bits).collect()))
            .collect::<Result<Vec<_>, String>>()?;
        r.end()?;
        Ok((tokens, logits))
    }
}
