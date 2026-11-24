//! One server binary's seats: each module is one model the server serves —
//! the flags it takes, the records it prints, its cache and slot rules —
//! behind [`bloomery_gpu_gates::bind::Seat`], with a `run(&args)` a binary
//! calls once. The per-model binaries (`bloomery-serve-ds41`,
//! `bloomery-serve-qwen38`) include their one seat directly; the one-binary
//! server (`bloomery-serve --model WORD`, the words of [`SEATS`]) includes this registry and is one call into
//! the seat the model names.
//!
//! A seat's module is its contract's owner: read it for the flags, the
//! records, the keep rule and the prompt cache. [`glm`] is the GLM-5.3-Flash
//! seat, opened as `generate_glm5next` opens the model; [`qwen3`] the
//! single-card Qwen seat (qwen3moe, qwen35moe), opened as
//! `generate_qwen3moe` opens them; [`mimo2`] the MiMo-V2.6-Flash seat, whose
//! every routed expert sits on the host, opened as `gate_mimo2_e2e` opens it;
//! [`solar2`] the Solar-Open2 seat, one sequence on one card with the routed
//! experts the card does not hold on the host, opened as `gate_solar2_e2e`
//! opens it; [`minimaxm3`] the MiniMax-M3 seat, `--parallel N` sequences on
//! one card served in turn, with the routed experts the card does not hold on
//! the host; [`mistrallarge4`] the Mistral Large 4 seat, `--parallel N` sequences on
//! one card served in turn (an expert tier card beside it on two cards), with the
//! routed experts the card does not hold on the host and an optional Pixtral image
//! input; [`musespark`] the Muse Spark seat, one sequence on one card with the
//! routed experts the card does not hold on the host, whose chat surface is
//! the recipient protocol; [`nemotron`] the Nemotron 3 Nano seat, one
//! sequence on one card, opened as `gate_nemotron_e2e` opens it.
//! [`hybrid`] holds the open, load record and calls of the one-chain seat
//! ([`solar2`], [`flint`], [`kimi_k3`], [`hy4`], [`ernie55`]). [`host_seat`] is the host seat: the file's own host forward with
//! no card (deepseek2 today), served through `serve::TargetEngine`.
//! `drafted` is the
//! seats' shared MTP draft driver and `served` the owner of the DSpark draft
//! the V4.1 and qwen3 seats serve, each one file a seat's per-model binary
//! includes beside the seat. The seats' shared `--ctx` rules are one owner:
//! the context an unset or set `--ctx` serves is `generate::choose_ctx` over
//! `placement::ctx_ask::choose` (the fit), and a set value's bound and the
//! slot count it buys (one slot at the whole context while `--parallel` names
//! none) are `placement::ctx`.

#[cfg(feature = "deepseek41")]
mod drafted;
// The seats' one owner of the draft a seat serves besides the MTP one: the
// DSpark draft, its slot rule and the draft word (`served::ServedDraft`).
#[cfg(feature = "deepseek41")]
pub mod ds41;
#[cfg(all(feature = "deepseek41", feature = "flint"))]
pub mod flint;
#[cfg(feature = "glm5next")]
pub mod glm;
#[cfg(feature = "deepseek41")]
mod served;
// The qwen38 seat runs no V4.1 code, but the server surface it sits on —
// `bind` and the `serve` and `sampler` crates — is scoped to `deepseek41`,
// as its per-model binary's doc says.
#[cfg(feature = "deepseek41")]
pub mod qwen38;
// The expert stream's rule the qwen38, minimaxm3, musespark and mistrallarge4 seats run;
// `bloomery-serve-qwen38` declares it beside its seat the same way.
#[cfg(feature = "deepseek41")]
#[path = "../xstream_rule.rs"]
pub mod xstream_rule;

/// The elastic expert tenant's record lines, printed by the Qwen3.8 seat's settle.
#[cfg(feature = "deepseek41")]
#[path = "../elastic.rs"]
pub mod elastic;
// The seats' one owner of a checkpoint folder's open (`minimaxm3_folder`, `qwen38_folder`,
// `qwen3_folder`, `nemotron`).
#[cfg(feature = "deepseek41")]
pub mod folder;
// A Qwen3.8 checkpoint folder, which the qwen38 seat opens; [`header`] reads any folder.
#[cfg(feature = "deepseek41")]
pub mod qwen38_folder;
// A dense Qwen3.5-family checkpoint folder, which the qwen3 seat loads whole on the card.
#[cfg(feature = "deepseek41")]
pub mod qwen3_folder;
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
// A MiniMax-M3 model at a path (a GGUF split or a checkpoint folder), which the minimaxm3 seat and its
// gates open.
#[cfg(all(feature = "deepseek41", feature = "minimaxm3"))]
pub mod minimaxm3_folder;
// The musespark seat sits behind `deepseek41` for the same reason: the server
// surface it binds through is scoped to it, and it runs no V4.1 code. The body
// it serves is in the gpu crate, so it needs no feature of its own.
#[cfg(feature = "deepseek41")]
pub mod musespark;
// The mistrallarge4 seat sits behind `deepseek41` for the same reason: the server surface it binds
// through is scoped to it, and it runs no V4.1 code. Its body is in `bloomery-gpu-mistrallarge4`.
#[cfg(all(feature = "deepseek41", feature = "mistrallarge4"))]
pub mod mistrallarge4;
// The seat of a body that runs one chain on one card, fed in prompt batches
// (`solar2`): its open, its load record, its calls and its think tags.
#[cfg(all(
    feature = "deepseek41",
    any(
        feature = "solar2",
        feature = "flint",
        feature = "kimik3",
        feature = "hy4",
        feature = "ernie55"
    )
))]
pub mod hybrid;
// The hybrid seat's in-file MTP draft: one sequence's window behind the object the seat holds.
#[cfg(all(
    feature = "deepseek41",
    any(
        feature = "solar2",
        feature = "flint",
        feature = "kimik3",
        feature = "hy4",
        feature = "ernie55"
    )
))]
pub mod hybrid_draft;
// The Kimi K3 seat, on the hybrid seat's owner: one sequence, its prompt stepped one id at a time.
#[cfg(all(feature = "deepseek41", feature = "kimik3"))]
pub mod kimi_k3;
// The Hy4-flash seat, on the hybrid seat's owner: resident sequences, the prompt stepped one id at
// a time.
#[cfg(all(feature = "deepseek41", feature = "hy4"))]
pub mod hy4;
// The ERNIE 5.5 Flash seat, on the hybrid seat's owner: one sequence of a checkpoint folder, its
// prompt the pass's units where the card holds every expert, else stepped.
#[cfg(all(feature = "deepseek41", feature = "ernie55"))]
pub mod ernie55;
// The nemotron seat sits behind `deepseek41` for the same reason: the server
// surface it binds through is scoped to it, and it runs no V4.1 code. The
// body it serves is in the gpu crate, so it needs no feature of its own.
#[cfg(feature = "deepseek41")]
pub mod nemotron;
// The inkling2 seat sits behind `deepseek41` for the nemotron seat's reason; its body is in the
// gpu crate and its prompt and keep rules in the app crate.
#[cfg(feature = "deepseek41")]
pub mod inkling2;
// The raon1 seat sits behind `deepseek41` for the nemotron seat's reason; its body is
// in the gpu crate.
#[cfg(feature = "deepseek41")]
pub mod raon1;
// The seat side of a plane state (`minimaxm3`, `musespark`): the prompt-cache methods of a seat whose
// body keeps rows of positions alone.
#[cfg(feature = "deepseek41")]
pub mod planes;
// The seats' one owner of a round run as passes (`step_rows_one_pass`).
#[cfg(feature = "deepseek41")]
pub mod rounds;
// The tiered seats' one run (`solar2`, `mimo2`, `minimaxm3`, `musespark`): the flags, the placement, the
// context, the plan and the engine thread behind one owner, each seat giving its family's facts.
#[cfg(feature = "deepseek41")]
pub mod tiered;
// The host seat: the file's own host forward on the CPU, no card in the plan.
// The module is `host_seat`, not `host`: `host` is an identifier the check's
// module-name match reads in many files, so a module of that name is reached
// from every root and `tools/check-records.py` asks for a record group of it.
#[cfg(feature = "deepseek41")]
pub mod host_seat;

// The decide seat: a decision model's backbone on the card, its head on the host.
#[cfg(feature = "clef")]
pub mod decide;

// ---- the generative seats' table ----
//
// `bloomery-serve` reads this table and nothing else about which models it serves: its words, usage lines,
// the seat a file takes, and each seat's `run` and `check`. A model's seat is one `seat!` row beside its
// `pub mod`; the one-binary server's own code names no model.

use bloomery_gpu_gates::{GateError, side_seat};
use gguf::Split;
use model::arch::{Arch, DEEPSEEK4};
use serve::ServeError;

/// A generative seat's `run`: loads the model and serves until the listener or the engine fails.
pub type Run = fn(&[String], &side_seat::Flags) -> Result<ServeError, GateError>;

/// A generative seat's `check` before the fetch: the arguments, whether the levers are read, and whether the
/// model comes from a `--hf` repo.
pub type Check = fn(&[String], bool, bool) -> Result<(), GateError>;

/// How a seat picks its files.
pub enum Serves {
    /// The file's `general.architecture` is one of these strings: an architecture no `model::arch::Arch`
    /// stands for, or one that the registry has no reader for (the host seat's).
    Names(&'static [&'static str]),
    /// The file's `general.architecture` names one of these variants through [`Arch::from_name`] (the
    /// registry's row of the family).
    Arch(&'static [Arch]),
    /// The seat reads the file's description and takes the file the program it runs serves: the answer waits
    /// for the first shard, and this seat is the one that takes a decision row's backbone file as a
    /// generative model.
    File(fn(&Split) -> bool),
}

impl Serves {
    /// Whether a file of the `general.architecture` `arch` is served; `None` when the answer needs the
    /// first shard `file` and none is given.
    pub fn of(&self, arch: &str, file: Option<&Split>) -> Option<bool> {
        Some(match self {
            Serves::Names(names) => names.contains(&arch),
            Serves::Arch(arches) => matches!(Arch::from_name(arch), Ok(a) if arches.contains(&a)),
            Serves::File(serves) => serves(file?),
        })
    }
}

/// One generative seat: the `--model` word, which files it serves, the lines `--help` prints for it, and
/// its two entry points.
pub struct Seat {
    pub word: &'static str,
    pub serves: Serves,
    /// One line a model the seat serves: the word, what it serves and a working example start.
    pub models: &'static [&'static str],
    pub run: Run,
    pub check: Check,
}

/// A seat's row, the module `$m` supplying `run` and `check`; `source` names a seat whose `check` also
/// asks whether the model comes from a `--hf` repo.
macro_rules! seat {
    ($m:ident, $word:literal, $serves:expr, $models:expr) => {
        Seat {
            word: $word,
            serves: $serves,
            models: &$models,
            run: $m::run,
            check: |args, read, _| $m::check(args, read).map(drop),
        }
    };
    ($m:ident, source, $word:literal, $serves:expr, $models:expr) => {
        Seat {
            word: $word,
            serves: $serves,
            models: &$models,
            run: $m::run,
            check: |args, read, from_repo| $m::check(args, read, from_repo).map(drop),
        }
    };
}

/// Every generative seat. The order is the one a file's seat is looked up in and the refusals list the words
/// in; the host seat is last, since a card seat is preferred for a file both serve.
pub const SEATS: &[Seat] = &[
    seat!(
        ds41,
        source,
        "ds41",
        Serves::Names(&["deepseek41", DEEPSEEK4]),
        [
            "  ds41    DeepSeek-V4.1-Flash          --hf midagedev/DeepSeek-V4.1-Flash-1017-GGUF:bq3km"
        ]
    ),
    seat!(
        qwen38,
        source,
        "qwen38",
        Serves::Names(&["qwen4exp"]),
        ["  qwen38  Qwen3.8-Flash-Next           --hf unsloth/Qwen3.8-Flash-Next-GGUF:UD-Q4_K_XL"]
    ),
    seat!(
        glm,
        source,
        "glm",
        Serves::Arch(&[Arch::Glm5next]),
        ["  glm     GLM-5.3-Flash                --hf unsloth/GLM-5.3-Flash-GGUF:UD-Q4_K_XL"]
    ),
    seat!(
        qwen3,
        source,
        "qwen3",
        Serves::File(|file| bloomery_gpu_gates::qwen_seat::Family::of(file).is_ok()),
        [
            "  qwen3   Qwen3-30B-A3B and Qwen3.6    --hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M"
        ]
    ),
    seat!(
        mimo2,
        "mimo2",
        Serves::Names(&["mimo2"]),
        ["  mimo2   MiMo-V2.6-Flash              --hf ggml-org/MiMo-V2.6-Flash-MOPD-GGUF:MXFP4"]
    ),
    seat!(
        solar2,
        "solar2",
        Serves::Arch(&[Arch::SolarOpen2]),
        ["  solar2  Solar-Open2-250B             -m <path>"]
    ),
    seat!(
        minimaxm3,
        "minimaxm3",
        Serves::Arch(&[Arch::MiniMaxM3]),
        ["  minimaxm3 MiniMax-M3-UD-IQ4_XS        -m <path>"]
    ),
    seat!(
        nemotron,
        "nemotron",
        Serves::Arch(&[Arch::NemotronH, Arch::NemotronHMoe]),
        [
            "  nemotron Nemotron 3 Nano             --hf nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:Q4_K_M",
            "  nemotron Nemotron 3.5 Bolt            -m <path>"
        ]
    ),
    seat!(
        raon1,
        "raon1",
        Serves::Arch(&[Arch::Raon1]),
        ["  raon1   Raon-1-mini                  -m <path>"]
    ),
    seat!(
        inkling2,
        "inkling2",
        Serves::Arch(&[Arch::Inkling2]),
        ["  inkling2 Inkling 2 Mini              -m <path>"]
    ),
    seat!(
        musespark,
        "musespark",
        Serves::Arch(&[Arch::MuseSpark]),
        ["  musespark Muse Spark Open 816B       -m <path>"]
    ),
    seat!(
        flint,
        "flint",
        Serves::Arch(&[Arch::Flint, Arch::ExaoneMoe]),
        ["  flint   Flint                       -m <path>"]
    ),
    seat!(
        kimi_k3,
        "kimik3",
        Serves::Arch(&[Arch::KimiK3]),
        ["  kimik3  Kimi K3 and K3.1             -m <path>"]
    ),
    seat!(
        hy4,
        "hy4",
        Serves::Arch(&[Arch::Hy4]),
        ["  hy4     Hy4-flash                    -m <path>"]
    ),
    seat!(
        ernie55,
        "ernie55",
        Serves::Arch(&[Arch::Ernie55]),
        ["  ernie55 ERNIE 5.5 Flash Preview      -m <folder>"]
    ),
    seat!(
        mistrallarge4,
        "mistrallarge4",
        Serves::Arch(&[Arch::MistralLarge4]),
        ["  mistrallarge4 Mistral Large 4 1T-A52B -m <path>"]
    ),
    seat!(
        host_seat,
        "host",
        Serves::Names(&["deepseek2"]),
        [
            "  host    a file on the CPU alone      -m <path>   (no card; -ngl 0 and -dev none ask for it)"
        ]
    ),
];
