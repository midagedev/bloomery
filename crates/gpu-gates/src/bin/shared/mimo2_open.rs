//! What the MiMo gates that load the file in process share (`gate_mimo2_e2e`,
//! `gate_mimo2_serve`): the session opened by its placement on the gate card
//! with its prompts fed one step an id ([`open`], the open's records printed
//! by [`Log`]), the vocabulary the logits rows are read at ([`N_VOCAB`]), and
//! the last position's argmax held to ik's, equal or a named tie
//! ([`last_argmax`]). A gate includes it by `#[path]` and holds
//! `shared/gate_card.rs` and `shared/e2e.rs` as modules beside it.

use std::path::Path;
use std::time::Instant;

use app::arch::mimo2::Mimo2Cfg;
use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::flip::tie_allowed;
use bloomery_gpu_mimo2::{Body, Mimo2Model};
use gguf::Split;
use model::arch::mimo2::place::PlanInputs;
use model::placement::{Machine, Plan, PlanLevers};

use crate::e2e::tie_numbers;

/// The model's vocabulary: the width of a logits row.
pub const N_VOCAB: usize = 152_576;

/// No band yet on a named tie's logits row against the head input's distance
/// (the forced arm that measures one is a later round's): the tie is ik's
/// runner-up inside twice our distance at the two ids, and the row's and the
/// last layer's distances are printed.
pub const HEAD_BAND: f64 = f64::INFINITY;

/// What the open decided besides the session: the plan's card experts a
/// layer.
pub struct Opened {
    pub n_l: Vec<u64>,
}

/// The open's records, as the gate prints them: the plan, and the load at the
/// context it was opened for.
pub struct Log {
    t: Instant,
    ctx: usize,
    n_l: Vec<u64>,
}

impl OpenLog<Body> for Log {
    fn plan(
        &mut self,
        place: &'static str,
        _inputs: &PlanInputs,
        _machine: &Machine,
        plan: &Plan<'_>,
    ) -> Result<bool, SessionError> {
        println!(
            "plan place={place} ctx_max={} host_experts={} card_experts={}",
            plan.ctx_max, plan.host.experts, plan.cards[0].experts
        );
        self.n_l = plan.n_l.clone();
        Ok(true)
    }

    fn load(&mut self, m: &Mimo2Model) -> Result<(), SessionError> {
        println!(
            "load resident_bytes={} ctx={} layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            self.ctx,
            m.layers().len(),
            self.t.elapsed().as_secs_f64()
        );
        Ok(())
    }

    fn capture(&mut self, _nodes: usize) -> Result<(), SessionError> {
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &Mimo2Model) -> Result<(), SessionError> {
        Ok(())
    }
}

/// The session of `file` at `ctx` positions on the gate placement
/// (`crate::gate_card::plan_gate`), its prompts fed one step an id.
pub fn open(
    file: &Path,
    levers: &bloomery_levers::Levers,
    ctx: usize,
) -> Result<(Session<Body>, Opened), GateError> {
    let split = Split::open(file).map_err(|e| format!("open {}: {e}", file.display()))?;
    let cfg = Mimo2Cfg {
        place: PlanLevers::from_levers(levers)?,
        host: levers.host(),
    };
    let mut log = Log {
        t: Instant::now(),
        ctx,
        n_l: Vec::new(),
    };
    let args = OpenArgs {
        place: "gate",
        machine: crate::gate_card::plan_gate,
        ctx,
        mode: StepMode::Graph,
        cfg,
    };
    let s = Loaded::<Body>::open(split, args, &mut log)?
        .ok_or("the open stopped at its plan")?
        .ready(&mut log)?;
    Ok((s, Opened { n_l: log.n_l }))
}

/// The last position's argmax against ik's: equal, or a named tie
/// ([`tie_allowed`], no band on the logits row). Prints the numbers; the
/// verdict and whether it was a tie.
pub fn last_argmax(what: &str, (ours, ik): (&[f32], &[f32]), input_rel: f64) -> (bool, bool) {
    let (top, ik_top, ik_2, margin, dist, logits_rel) = tie_numbers(ours, ik);
    let tie = tie_allowed(
        (top, ik_top, ik_2),
        (margin, dist),
        (logits_rel, input_rel),
        HEAD_BAND,
    );
    println!(
        "{what}: argmax ours={top} ik={ik_top} (ik's runner-up {ik_2}, margin {margin:.4}, \
         our distance at the two {dist:.4}, logits_rel {logits_rel:.3e} against the last \
         layer's {input_rel:.3e}{})",
        if tie { ", a named tie" } else { "" }
    );
    (top == ik_top || tie, tie)
}
