//! The tensor-parallel V4.1 body (`bloomery_gpu_deepseek41::body::open_rank`) on the fixture, every
//! rank a context of its own on the one visible card.
//!
//! - (n1) one rank against the one-card body (`body::open`, the gate plan without a card budget,
//!   so every expert is on the card): the rank reads every tensor whole from the file as a part
//!   source, holds every routed expert on the card and runs no host tier; over [`STEPS`] greedy
//!   steps from [`FIRST`], every step's token and logits are the one-card body's bit for bit.

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
#[path = "shared/ds41_tier.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the model path; the rest serves the step and prefill gates"
)]
mod ds41_tier;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::sync::{Arc, mpsc};

    use bloomery_gpu::host::{GatherLoop, Gathers, PostCols};
    use bloomery_gpu::{Gpu, GpuError, MappedHost};
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model, OpenCfg};
    use bloomery_gpu_gates::GateError;
    use bloomery_levers::{CARD_BUDGET, ENGRAM_HELPER};
    use gguf::{GgmlType, Split};
    use model::part::Take;

    /// The greedy steps each run takes.
    const STEPS: usize = 24;
    /// The token the runs start from.
    const FIRST: u32 = 1;
    /// The positions a load's stores hold.
    const CTX: usize = 64;

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[ENGRAM_HELPER, CARD_BUDGET])?;
        crate::gate_card::init()?;
        let cfg = OpenCfg::from_levers(&levers)?;
        let path = crate::ds41_tier::model_path()?;
        let open = || Split::open(&path).map_err(|e| format!("open {path}: {e}"));

        let whole = {
            let mut m = body::open(open()?, crate::gate_card::plan_gate, CTX, &cfg)?;
            greedy(&mut m)?
        };
        let rank = run_ranks(&path, &cfg, 1, &|_, _, _, _| Ok(Take::Whole))?;
        let tokens = |r: &[(u32, Vec<f32>)]| r.iter().map(|s| s.0).collect::<Vec<_>>();
        let first_off = whole
            .iter()
            .zip(&rank)
            .position(|(a, b)| a.0 != b.0 || !same_bits(&a.1, &b.1));
        let pass = first_off.is_none();
        println!(
            "(n1) one rank against the one-card body, {STEPS} steps from {FIRST}: first step off \
             {first_off:?}; tokens {:?} and {:?}: {}",
            tokens(&whole),
            tokens(&rank),
            if pass { "PASS" } else { "FAIL" }
        );
        if pass {
            Ok(())
        } else {
            Err("gate_deepseek41_tp: a clause failed".into())
        }
    }

    /// What rank `r` reads of a tensor (name, type, dims).
    type TakeOf = dyn Fn(usize, &str, GgmlType, &[u64]) -> Result<Take, GpuError> + Sync;

    /// [`STEPS`] greedy steps from [`FIRST`] on `ranks` ranks of the file at `path`, each opened
    /// and stepped on a thread of its own with a context of its own on the one card, reading what
    /// `take` gives it, their pages joined by a loopback past one rank: at each step the merge of
    /// the ranks' picks ([`merge`]) and the ranks' logits laid in rank order.
    fn run_ranks(
        path: &str,
        cfg: &OpenCfg,
        ranks: usize,
        take: &TakeOf,
    ) -> Result<Vec<(u32, Vec<f32>)>, GateError> {
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
                    let opened = (|| -> Result<Deepseek41Model, GateError> {
                        let open = || Split::open(path).map_err(|e| format!("open {path}: {e}"));
                        let source = open()?;
                        Ok(body::open_rank(
                            Gpu::new()?,
                            open()?,
                            &source,
                            |name, ty, dims| take(r, name, ty, dims),
                            (r, ranks),
                            CTX,
                            cfg.body,
                        )?)
                    })();
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
                let logits = parts.concat();
                token = merge(&logits)?;
                out.push((token, logits));
            }
            drop(tokens);
            Ok(out)
        })
    }

    /// The step's token over logits laid in rank order: the largest value, of equal values the
    /// lowest id.
    fn merge(logits: &[f32]) -> Result<u32, GateError> {
        let mut best: Option<(usize, f32)> = None;
        for (i, &v) in logits.iter().enumerate() {
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((i, v));
            }
        }
        let (i, _) = best.ok_or("no logits to merge")?;
        Ok(u32::try_from(i)?)
    }

    /// [`STEPS`] greedy steps from [`FIRST`]: each step's token and logits.
    fn greedy(m: &mut Deepseek41Model) -> Result<Vec<(u32, Vec<f32>)>, GateError> {
        let mut token = FIRST;
        let mut out = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            token = m.step(&[token])?;
            out.push((token, m.logits()?));
        }
        Ok(out)
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }
}
