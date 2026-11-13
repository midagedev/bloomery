//! `ppl_exaone` — K-EXAONE-2.0's loss tool: the engine's own run of a GGUF over a fixed id file, written as an ik
//! KL-divergence base file (`bloomery_gpu_gates::kld`), so that `kld_diff` prints how far one quantization of the
//! model sits from another on the same text.
//!
//!     BLOOMERY_REF_MODEL=<first shard> ppl_exaone --ids FILE --run CTX[xCHUNKS]=OUT.kld ...
//!                            [--arm a|bp] [--plan-ctx C]
//!     kld_diff A.kld B.kld                    mean KLD ± SE, p50/p99/max, same top, PPL of A and of B
//!
//! `--ids` is one decimal id a line (`corpus-prose.ids`). Each `--run` (repeatable, one load serves them all) takes
//! the first `CTX × CHUNKS` ids as `CHUNKS` chunks of `CTX` (default one chunk) and writes their base file `OUT`. A
//! chunk runs from an empty model one id a step, and a record (ik's quantized log-softmax over the whole
//! vocabulary) is written at every position from `CTX/2` through `CTX-2`, scored against the id after it: the ids as
//! given, no BOS put in a chunk's first place. A file is written whole or not at all (`<OUT>.part`, then a rename),
//! and an existing `OUT` is refused before the load. The model is planned as the serve seat plans it on the card in
//! view (`--arm a`, the gate placement; `bp` the two-card tier plan) at `--plan-ctx` positions (default 8192, so two
//! files of one tool run under one placement); a `CTX` above it is refused. Two files share a base file's meaning only
//! when their ids, `CTX` and `CHUNKS` are the same.
//!
//! The lines on stdout: `load` (the resident bytes, the context, seconds), a progress line every 256 positions and
//! a line a chunk, and a `kld_base` line a run.

#[cfg(feature = "flint")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("ppl_exaone", cli::run())
}

#[cfg(feature = "flint")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "flint")]
#[allow(
    dead_code,
    reason = "the seat serves several gates; each takes the calls it reads"
)]
#[path = "shared/exaone_seat.rs"]
mod seat;

#[cfg(feature = "flint")]
#[path = "shared/exaone_ppl.rs"]
mod ppl;

#[cfg(feature = "flint")]
mod cli {
    use crate::lcpp_e2e::Arm;
    use bloomery_gpu_gates::kld::{read_ids, write_base};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use std::path::PathBuf;

    /// Positions the model is planned for when `--plan-ctx` is not given: the serve seat's context.
    const PLAN_CTX: usize = 8192;

    /// Every flag this CLI reads; any other argument is refused by name, never ignored.
    const KNOWN: [&str; 4] = ["--ids", "--run", "--arm", "--plan-ctx"];

    /// One `--run`: the first `ctx * chunks` ids as `chunks` chunks of `ctx`, written to `out`.
    struct Run {
        ctx: usize,
        chunks: usize,
        out: PathBuf,
    }

    impl Run {
        /// `CTX=OUT` or `CTXxCHUNKS=OUT`; a spec of another shape is refused by name.
        fn parse(spec: &str) -> Result<Self, GateError> {
            let bad = || {
                GateError::from(format!(
                    "ppl_exaone: --run {spec:?} is not CTX[xCHUNKS]=OUT.kld (CTX at least 3, CHUNKS at least 1)"
                ))
            };
            let (shape, out) = spec.split_once('=').ok_or_else(bad)?;
            let (ctx, chunks) = shape.split_once('x').unwrap_or((shape, "1"));
            let ctx: usize = ctx.parse().map_err(|_| bad())?;
            let chunks: usize = chunks.parse().map_err(|_| bad())?;
            if ctx < 3 || chunks == 0 || out.is_empty() {
                return Err(bad());
            }
            Ok(Self {
                ctx,
                chunks,
                out: PathBuf::from(out),
            })
        }

        fn ids(&self) -> usize {
            self.ctx * self.chunks
        }
    }

    /// Every value given to flag `name`, in order: empty when the flag is absent, an error when it is last.
    fn flags(args: &[String], name: &str) -> Result<Vec<String>, GateError> {
        let mut found = Vec::new();
        let mut at = args.iter().enumerate();
        while let Some((i, a)) = at.next() {
            if a == name {
                let v = args
                    .get(i + 1)
                    .ok_or_else(|| GateError::from(format!("ppl_exaone: {name} needs a value")))?;
                found.push(v.clone());
                at.next();
            }
        }
        Ok(found)
    }

    /// The one value of flag `name`: `None` when absent, an error when it is given twice.
    fn flag(args: &[String], name: &str) -> Result<Option<String>, GateError> {
        let mut all = flags(args, name)?;
        match all.len() {
            0 => Ok(None),
            1 => Ok(all.pop()),
            _ => Err(format!("ppl_exaone: {name} is given once").into()),
        }
    }

    pub fn run() -> Result<(), GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        for pair in args.chunks(2) {
            if !KNOWN.contains(&pair[0].as_str()) {
                return Err(format!(
                    "ppl_exaone: {} is no flag of this tool (one of {KNOWN:?})",
                    pair[0]
                )
                .into());
            }
        }
        let ids_path = PathBuf::from(
            flag(&args, "--ids")?.ok_or("ppl_exaone: --ids FILE is required (one id a line)")?,
        );
        let plan_ctx = flag(&args, "--plan-ctx")?.map_or(Ok(PLAN_CTX), |s| {
            s.parse::<usize>()
                .map_err(|_| GateError::from(format!("ppl_exaone: --plan-ctx {s:?} is no count")))
        })?;
        let arm = Arm::parse(flag(&args, "--arm")?.as_deref().unwrap_or("a"))?;
        let runs = flags(&args, "--run")?
            .iter()
            .map(|s| Run::parse(s))
            .collect::<Result<Vec<_>, _>>()?;
        if runs.is_empty() {
            return Err("ppl_exaone: at least one --run CTX[xCHUNKS]=OUT.kld is required".into());
        }
        for (i, r) in runs.iter().enumerate() {
            if r.ctx > plan_ctx {
                return Err(format!(
                    "ppl_exaone: --run context {} is above --plan-ctx {plan_ctx}",
                    r.ctx
                )
                .into());
            }
            if r.out.exists() {
                return Err(format!(
                    "ppl_exaone: {} exists, and a base is never overwritten",
                    r.out.display()
                )
                .into());
            }
            if runs[..i].iter().any(|p| p.out == r.out) {
                return Err(
                    format!("ppl_exaone: {} is the output of two runs", r.out.display()).into(),
                );
            }
        }
        // The count and the format before the load; the vocabulary's bound once the head's width is known.
        let need = runs.iter().map(Run::ids).max().unwrap_or(0);
        read_ids(&ids_path, need, usize::MAX)?;

        let path = ref_model_path()?;
        let t = std::time::Instant::now();
        let mut m = crate::ppl::open(&path, plan_ctx, arm)?;
        let vocab = m.vocab();
        println!(
            "load arm={} model={} resident_bytes={} ctx={plan_ctx} layers={} vocab={vocab} load_s={:.1}",
            arm.word(),
            path.display(),
            m.resident_bytes(),
            m.layers(),
            t.elapsed().as_secs_f64()
        );
        let ids = read_ids(&ids_path, need, vocab)?;
        for r in &runs {
            let w = write_base(&mut m, &ids[..r.ids()], r.ctx, vocab, &r.out)?;
            println!(
                "kld_base path={} n_ctx={} n_chunk={} n_vocab={vocab} positions={} ppl={:.4} wall_s={:.1}",
                w.path.display(),
                r.ctx,
                r.chunks,
                w.positions,
                w.ppl,
                w.secs
            );
        }
        Ok(())
    }
}

#[cfg(not(feature = "flint"))]
fn main() {
    eprintln!("ppl_exaone: built without the `flint` feature");
    std::process::exit(2);
}
