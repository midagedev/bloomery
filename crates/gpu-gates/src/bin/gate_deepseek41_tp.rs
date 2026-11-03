//! The tensor-parallel V4.1 body (`bloomery_gpu_deepseek41::body::open_rank`) on the
//! original-format fixture folder (`<fixture root>/v41-folder`, written by
//! `model_hf::arch::deepseek41::fixture`): every rank a thread and a handle of its own on the one
//! visible card, reading its part of each tensor as the model's cut gives it, their sum pages
//! joined by a loopback past one rank.
//!
//! The ranks share the card's primary context, and a rank's wait for the others' parts is a stream
//! memory op the driver's scheduler does not see: two ranks' streams on one hardware queue hold
//! each other, the wait in front of the part it waits for. So the gate runs only under the
//! driver's most queues, `CUDA_DEVICE_MAX_CONNECTIONS=32`, which its recipe sets.
//!
//! - (n1) one rank reads the folder whole and runs [`STEPS`] greedy steps from [`FIRST`]: every
//!   logit is finite.
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
mod gate {
    use std::sync::{Arc, mpsc};

    use bloomery_gpu::head::{local_pick, merge};
    use bloomery_gpu::host::rank::Sums;
    use bloomery_gpu::host::{GatherLoop, Gathers, PostCols};
    use bloomery_gpu::{Gpu, MappedHost};
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model, OpenCfg, RankFrom};
    use bloomery_gpu_deepseek41::chain::join;
    use bloomery_gpu_gates::GateError;
    use bloomery_gpu_gates::logits_band::{self, Positions};
    use bloomery_levers::{CARD_BUDGET, ENGRAM_HELPER};
    use model_hf::folder::{self, Opened};
    use model::arch::deepseek41::hparams::Hparams;

    /// The greedy steps each run takes.
    const STEPS: usize = 24;
    /// The token the runs start from.
    const FIRST: u32 = 1;
    /// The positions a load's stores hold.
    const CTX: usize = 64;
    /// The fixture folder under the fixture root.
    const FOLDER: &str = "v41-folder";
    /// The hardware queues the driver gives the ranks' streams (`CUDA_DEVICE_MAX_CONNECTIONS`).
    const CONNECTIONS: &str = "32";

    /// One run's steps: each step's token and the ranks' logits laid in rank order.
    type Run = Vec<(u32, Vec<f32>)>;

    pub fn run() -> Result<(), GateError> {
        let queues = std::env::var("CUDA_DEVICE_MAX_CONNECTIONS");
        if queues.as_deref() != Ok(CONNECTIONS) {
            return Err(format!(
                "gate_deepseek41_tp: CUDA_DEVICE_MAX_CONNECTIONS is {queues:?}; the ranks' streams \
                 each need a hardware queue, so the gate runs under {CONNECTIONS} (its recipe)"
            )
            .into());
        }
        let levers = bloomery_levers::at_main(&[ENGRAM_HELPER, CARD_BUDGET])?;
        crate::gate_card::init()?;
        let cfg = OpenCfg::from_levers(&levers)?;
        let dir = format!("{}/{FOLDER}", refset::fixture::root()?);
        let one = run_ranks(&dir, &cfg, 1)?;
        let finite = one.iter().all(|(_, l)| l.iter().all(|v| v.is_finite()));
        println!(
            "(n1) one rank, {STEPS} steps from {FIRST}: tokens {:?}, every logit finite {finite}: {}",
            one.iter().map(|s| s.0).collect::<Vec<_>>(),
            if finite { "PASS" } else { "FAIL" }
        );
        let mut pass = finite;
        for ranks in [2, 4] {
            let got = run_ranks(&dir, &cfg, ranks)?;
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

    /// Rank `r` of `ranks` of the folder at `dir` opened on a context of its own, with its sums
    /// on a page of its own past one rank.
    fn open_rank(
        dir: &str,
        cfg: &OpenCfg,
        r: usize,
        ranks: usize,
    ) -> Result<Deepseek41Model, GateError> {
        let Opened {
            header,
            source,
            rows,
            ..
        } = folder::open(dir.as_ref()).map_err(|e| format!("open {dir}: {e}"))?;
        let gpu = Gpu::new()?;
        let sums = (ranks > 1)
            .then(|| {
                let hp = Hparams::read(&header).map_err(|e| e.to_string())?;
                Ok::<_, GateError>(Sums::new(gpu.context(), r, ranks, &join::shapes(&hp)?)?)
            })
            .transpose()?;
        Ok(body::open_rank(
            gpu,
            RankFrom {
                file: header,
                source: &source,
                stored: Some(&rows),
            },
            (r, ranks),
            sums,
            CTX,
            cfg.body,
        )?)
    }

    /// [`STEPS`] greedy steps from [`FIRST`] on `ranks` ranks of the folder at `dir`, each opened
    /// and stepped on a thread of its own ([`open_rank`]), their pages joined by a loopback past one
    /// rank: at each step the merge of the ranks' picks (`bloomery_gpu::head::merge`) and the
    /// ranks' logits laid in rank order.
    fn run_ranks(dir: &str, cfg: &OpenCfg, ranks: usize) -> Result<Run, GateError> {
        type Link = Option<(Gathers, Arc<MappedHost>)>;
        std::thread::scope(|scope| -> Result<_, GateError> {
            let (links_tx, links_rx) = mpsc::channel::<(usize, Result<Link, String>)>();
            let (out_tx, out_rx) = mpsc::channel::<(usize, Result<Vec<f32>, String>)>();
            let mut tokens = Vec::with_capacity(ranks);
            for r in 0..ranks {
                let (tok_tx, tok_rx) = mpsc::channel::<u32>();
                tokens.push(tok_tx);
                let (links_tx, out_tx) = (links_tx.clone(), out_tx.clone());
                scope.spawn(move || {
                    let opened = open_rank(dir, cfg, r, ranks);
                    let mut m = match opened {
                        Ok(m) => m,
                        Err(e) => {
                            let _ = links_tx.send((r, Err(e.to_string())));
                            return;
                        }
                    };
                    let link = m.body("gate_deepseek41_tp").map(|b| {
                        b.rank_link()
                            .map(|(layout, page)| (layout.clone(), Arc::clone(page)))
                    });
                    let _ = links_tx.send((r, link.map_err(|e| e.to_string())));
                    while let Ok(token) = tok_rx.recv() {
                        let logits = m
                            .step(&[token])
                            .and_then(|_| m.logits())
                            .map_err(|e| e.to_string());
                        let failed = logits.is_err();
                        let _ = out_tx.send((r, logits));
                        if failed {
                            return;
                        }
                    }
                });
            }
            drop((links_tx, out_tx));
            let mut links: Vec<Link> = vec![None; ranks];
            for _ in 0..ranks {
                let (r, link) = links_rx.recv().map_err(|e| e.to_string())?;
                links[r] = link.map_err(|e| format!("rank {r} of {ranks}: {e}"))?;
            }
            let ring = if ranks > 1 {
                let (layouts, pages): (Vec<_>, Vec<_>) = links
                    .into_iter()
                    .map(|l| l.ok_or("a rank of several without a sum page"))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .unzip();
                Some(GatherLoop::new(&layouts, &pages, &PostCols::default())?)
            } else {
                None
            };
            let mut token = FIRST;
            let mut out = Vec::with_capacity(STEPS);
            for _ in 0..STEPS {
                for tx in &tokens {
                    tx.send(token).map_err(|e| e.to_string())?;
                }
                let mut parts = vec![Vec::new(); ranks];
                for _ in 0..ranks {
                    let (r, logits) = out_rx.recv().map_err(|e| e.to_string())?;
                    match logits {
                        Ok(l) => parts[r] = l,
                        Err(e) => {
                            // A rank still waiting on the failed rank's part runs on.
                            let released = ring.as_ref().map(GatherLoop::release);
                            drop(tokens);
                            return Err(format!(
                                "rank {r} of {ranks}: {e} (link released: {released:?})"
                            )
                            .into());
                        }
                    }
                }
                let mut picks = Vec::with_capacity(ranks);
                let mut at = 0u32;
                for p in &parts {
                    picks.push(local_pick(p, at)?);
                    at += u32::try_from(p.len())?;
                }
                token = merge(&picks)?;
                out.push((token, parts.concat()));
            }
            drop(tokens);
            Ok(out)
        })
    }
}
