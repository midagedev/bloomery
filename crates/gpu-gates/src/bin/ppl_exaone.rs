//! `ppl_exaone` — K-EXAONE-2.0's loss tool: the engine's own run of a GGUF over a fixed id file, written as an ik
//! KL-divergence base file (`bloomery_gpu_gates::kld`), so that `kld_diff` prints how far one quantization of the
//! model sits from another on the same text.
//!
//!     BLOOMERY_REF_MODEL=<first shard> ppl_exaone (--ids FILE | --ids-of BASE.kld) --run CTX[xCHUNKS]=OUT.kld ...
//!                            [--arm a|bp] [--plan-ctx C]
//!     kld_diff A.kld B.kld                    mean KLD ± SE, p50/p99/max, same top, PPL of A and of B
//!
//! `--ids` is one decimal id a line (`corpus-prose.ids`), `--ids-of` the ids an existing base file holds. The command
//! line's model-free half is `bloomery_gpu_gates::kld::cli`'s. Each `--run` (repeatable, one load serves them all) takes
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
    use bloomery_gpu_gates::kld::cli::{Args, Spec};
    use bloomery_gpu_gates::{GateError, ref_model_path};

    /// This tool's flags beside the loss tools' common ones (`kld::cli`).
    const SPEC: Spec = Spec {
        tool: "ppl_exaone",
        own: &["--arm"],
        switches: &[],
        plan_only: None,
    };

    pub fn run() -> Result<(), GateError> {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let args = Args::parse(&SPEC, &raw)?;
        let plan_ctx = args.plan_ctx();
        let arm = Arm::parse(args.value("--arm").unwrap_or("a"))?;
        // The count and the format before the load; the vocabulary's bound once the head's width is known.
        let ids = args.ids()?;

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
        args.walk(&mut m, &ids.for_vocab(vocab)?, vocab)
    }
}

#[cfg(not(feature = "flint"))]
fn main() {
    eprintln!("ppl_exaone: built without the `flint` feature");
    std::process::exit(2);
}
