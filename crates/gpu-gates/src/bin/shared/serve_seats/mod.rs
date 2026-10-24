//! One server binary's seats: each module is one model the server serves —
//! the flags it takes, the records it prints, its cache and slot rules —
//! behind [`bloomery_gpu_gates::bind::Seat`], with a `run(&args)` a binary
//! calls once. The per-model binaries (`bloomery-serve-ds41`,
//! `bloomery-serve-qwen38`) include their one seat directly; the one-binary
//! server (`bloomery-serve --model ds41|qwen38|glm|qwen3|mimo2|solar2|minimaxm3|nemotron`)
//! includes this registry and is one call into the seat the model names.
//!
//! A seat's module is its contract's owner: read it for the flags, the
//! records, the keep rule and the prompt cache. [`glm`] is the GLM-5.3-Flash
//! seat, opened as `generate_glm5next` opens the model; [`qwen3`] the
//! single-card Qwen seat (qwen3moe, qwen35moe), opened as
//! `generate_qwen3moe` opens them; [`mimo2`] the MiMo-V2.6-Flash seat, whose
//! every routed expert sits on the host, opened as `gate_mimo2_e2e` opens it;
//! [`solar2`] the Solar-Open2 seat, one sequence on one card with the routed
//! experts the card does not hold on the host, opened as `gate_solar2_e2e`
//! opens it; [`minimaxm3`] the MiniMax-M3 seat, one sequence on one card with
//! the routed experts the card does not hold on the host, opened as
//! `gate_minimaxm3_e2e` opens it; [`nemotron`] the Nemotron 3 Nano seat, one
//! sequence on one card, opened as `gate_nemotron_e2e` opens it.
//! [`hybrid`] holds what the seats of one chain on one card ([`solar2`],
//! [`minimaxm3`]) do the same way.
//! `drafted` is the
//! seats' shared MTP draft driver and `served` the owner of the DSpark draft
//! the V4.1 and qwen3 seats serve, each one file a seat's per-model binary
//! includes beside the seat. The seats' shared `--ctx` rules are one owner:
//! the context an unset or set `--ctx` serves is `generate::choose_ctx` over
//! `placement::ctx_ask::choose` (the ladder), and a set value's bound and the
//! slot count it buys (one slot at the whole context while `--parallel` names
//! none) are `placement::ctx`.

#[cfg(feature = "deepseek41")]
mod drafted;
// The seats' one owner of the draft a seat serves besides the MTP one: the
// DSpark draft, its slot rule and the draft word (`served::ServedDraft`).
#[cfg(feature = "deepseek41")]
pub mod ds41;
#[cfg(feature = "glm5next")]
pub mod glm;
#[cfg(feature = "deepseek41")]
mod served;
// The qwen38 seat runs no V4.1 code, but the server surface it sits on —
// `bind` and the `serve` and `sampler` crates — is scoped to `deepseek41`,
// as its per-model binary's doc says.
#[cfg(feature = "deepseek41")]
pub mod qwen38;
// A Qwen3.8 checkpoint folder, which the qwen38 seat opens and [`header`] reads.
#[cfg(feature = "deepseek41")]
pub mod qwen38_folder;
// The qwen3 seat sits behind the same server-surface feature.
#[cfg(feature = "deepseek41")]
pub mod qwen3;
// The mimo2 seat sits behind `deepseek41` too: the server surface it binds
// through is scoped to it, and it runs no V4.1 code.
#[cfg(all(feature = "deepseek41", feature = "mimo2"))]
pub mod mimo2;
// The solar2 seat sits behind `deepseek41` for the same reason: the server
// surface it binds through is scoped to it, and it runs no V4.1 code.
#[cfg(all(feature = "deepseek41", feature = "solar2"))]
pub mod solar2;
// The minimaxm3 seat sits behind `deepseek41` for the same reason: the server
// surface it binds through is scoped to it, and it runs no V4.1 code.
#[cfg(all(feature = "deepseek41", feature = "minimaxm3"))]
pub mod minimaxm3;
// The seats of the bodies that run one chain on one card, fed in prompt
// batches (`solar2`, `minimaxm3`): their open, their load record, their calls
// and their think tags are one owner's.
#[cfg(all(feature = "deepseek41", any(feature = "solar2", feature = "minimaxm3")))]
pub mod hybrid;
// The nemotron seat sits behind `deepseek41` for the same reason: the server
// surface it binds through is scoped to it, and it runs no V4.1 code. The
// body it serves is in the gpu crate, so it needs no feature of its own.
#[cfg(feature = "deepseek41")]
pub mod nemotron;
// The raon1 seat sits behind `deepseek41` for the nemotron seat's reason; its body is
// in the gpu crate.
#[cfg(feature = "deepseek41")]
pub mod raon1;
// The seats' one owner of a round run as passes (`step_rows_one_pass`).
#[cfg(feature = "deepseek41")]
pub mod rounds;
// The tiered seats' one run (`solar2`, `mimo2`): the flags, the placement, the context, the plan
// and the engine thread behind one owner, each seat giving its family's facts.
#[cfg(all(feature = "deepseek41", any(feature = "mimo2", feature = "solar2")))]
pub mod tiered;
// The decide seat: a decision model's backbone on the card, its head on the host.
#[cfg(feature = "clef")]
pub mod decide;

/// `f` of the header of the model at `path`, which the server seats it by: its GGUF split's
/// first shard, or a checkpoint folder's when `path` is a directory (a Qwen3.8 checkpoint,
/// [`qwen38_folder::with_header`], which keeps the folder for the seat's load).
#[cfg(feature = "deepseek41")]
pub fn with_header<R>(
    path: &std::path::Path,
    f: impl FnOnce(&gguf::Split) -> R,
) -> Result<R, bloomery_gpu_gates::GateError> {
    if path.is_dir() {
        return qwen38_folder::with_header(path, f);
    }
    let split = gguf::Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    Ok(f(&split))
}
