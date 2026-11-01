//! A qwen3moe file's `--place`-unset decisions, read before any load and held
//! to what the card runs: the CLI's (`q3place::unplaced_qwen3` at
//! [`CLI_CTX`]) and the serve seat's (its own choice of context and load,
//! `q3place::serve_choice`, for its default slots). Both read the file's header and this card's census
//! only: no tensor byte, no upload. Where the whole load refuses the file
//! (`Qwen3moeModel::whole_load`: the card kernel table names no entry for a
//! layer's routed stacks) the answer must be the placed plan, and every layer
//! that plan seats card experts in (`Plan::n_l`) must hold stacks the table
//! runs ([`card_runs`]): the load refuses by name a layer that holds card
//! experts of stacks the table does not run.
//!
//! The file needs its header and its full length: a header written over a
//! sparse file of the length the header's tensors span opens as the file
//! does.

use bloomery_gpu::Qwen3moeModel;
use bloomery_gpu::kv_format::KvQ8;
use gguf::Split;
use model::placement::PlanLevers;
use model::placement::accepts::{self, Program};
use model::placement::ctx_ask::DEFAULT_SLOTS;

#[path = "qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the decisions plan a qwen3moe file only; the other half serves the CLI and the seat"
)]
pub(super) mod q3place;

/// The CLI's default context for a qwen3moe run (`generate_qwen3moe`'s
/// `--ctx` unset).
pub const CLI_CTX: usize = 4096;

/// The CLI's decision on `split`, as the words of one line, or why it is
/// refused or breaks the rule above.
pub fn cli_default(split: &Split) -> Result<String, String> {
    let whole_refuses = Qwen3moeModel::whole_load(split).is_err();
    let u = q3place::unplaced_qwen3(split, CLI_CTX, KvQ8::F16).map_err(|e| e.to_string())?;
    held(split, &u, CLI_CTX, whole_refuses)
}

/// The serve seat's decision on `split`: its own choice of context and the
/// load at it (`q3place::serve_choice`), as for [`cli_default`].
pub fn serve_default(split: &Split) -> Result<String, String> {
    let whole_refuses = Qwen3moeModel::whole_load(split).is_err();
    let levers = PlanLevers::default();
    let opts = q3place::ServeOpts {
        place: None,
        levers: &levers,
        kv: KvQ8::F16,
        media: false,
        beside: bloomery_gpu_gates::beside::Beside::default(),
        ubatch: None,
    };
    let (chosen, _) = q3place::serve_choice(split, (None, DEFAULT_SLOTS), opts)
        .map_err(|e| format!("the --ctx choice refused: {e}"))?;
    let total = usize::try_from(chosen.total).map_err(|e| e.to_string())?;
    let line = chosen.line();
    let u = match chosen.plan {
        q3place::ServeLoad::Whole(on) => q3place::Unplaced::Whole(on.card),
        q3place::ServeLoad::Placed { q, .. } => q3place::Unplaced::Placed(q),
    };
    let how = held(split, &u, total, whole_refuses)?;
    Ok(format!("{line}, {how}"))
}

/// The answer `u` at `ctx` positions held to the rule: the whole load only
/// where it runs the file; a placed plan with card experts only in layers
/// whose stacks the card runs.
fn held(
    split: &Split,
    u: &q3place::Unplaced,
    ctx: usize,
    whole_refuses: bool,
) -> Result<String, String> {
    match u {
        q3place::Unplaced::Whole(_) if whole_refuses => {
            Err("the whole load refuses the file, yet the answer is the whole load".into())
        }
        q3place::Unplaced::Whole(_) => Ok("whole".into()),
        q3place::Unplaced::Placed(q) => {
            let plan = q
                .plan(ctx, &PlanLevers::default())
                .map_err(|e| format!("the placed plan refused: {e}"))?;
            for (l, &n) in plan.n_l.iter().enumerate() {
                if n > 0 && !card_runs(split, l)? {
                    return Err(format!(
                        "the placed plan seats {n} card experts in layer {l}, whose routed \
                         stacks ({}) the card kernel table does not run",
                        stack_types(split, l)?.map(|t| format!("{t:?}")).join("/")
                    ));
                }
            }
            Ok(format!(
                "placed, {} card experts",
                plan.n_l.iter().sum::<u64>()
            ))
        }
    }
}

/// Layer `l`'s routed gate, up and down types, as the header holds them.
fn stack_types(split: &Split, l: usize) -> Result<[gguf::quant::GgmlType; 3], String> {
    let ty = |part: &str| {
        let name = format!("blk.{l}.ffn_{part}_exps.weight");
        split
            .find(&name)
            .map(|(_, t)| t.ty)
            .ok_or_else(|| format!("{name}: not in the file"))
    };
    Ok([ty("gate")?, ty("up")?, ty("down")?])
}

/// Whether the card leg runs layer `l`'s routed stacks, as the load checks
/// them ([`accepts::leg_runs`] of the qwen3moe body).
fn card_runs(split: &Split, l: usize) -> Result<bool, String> {
    let types = stack_types(split, l)?;
    Ok(accepts::leg_runs(Program::Qwen3moeBody, types) == Some(true))
}
