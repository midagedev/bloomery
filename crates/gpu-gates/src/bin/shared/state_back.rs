//! The state a load leaves its model in between the configs that share it:
//! `generate_ds41`'s `--arm` list and `bloomery-chat`'s `--then` runs read it
//! once after the load and assert it back before every config after the
//! first — the position, the fault word and each card's free device bytes —
//! so a state that leaks across the boundary ends the process by name
//! instead of quietly feeding the next config.
//!
//! What checks: `generate_ds41`'s one-slot plain path, before each arm after
//! the first of an `--arm` list — before the arm's `arm` record, so
//! `--arm-sync`'s wait never sits behind a check — and `bloomery-chat`,
//! before each run after the first, after its clear. What does not:
//! `--repeat` keeps the residency by design (`Target::reset`, not the
//! session's clear), a clear over several slots (`BLOOMERY_GEN_SLOTS`)
//! leaves the model-wide state the parked slots stand on, and the draft and
//! finite-probe paths run one arm by their own refusal.
//!
//! The cards read: the stage card through [`GpuModel::gpu`] and every expert
//! tier card the load holds through [`Body::hybrid`]'s tiers, each its free
//! device bytes as `Gpu::mem_info` reports them. The stores' contents are
//! not read: hashing them is the expensive half of a state check, which
//! this module does not do.

use bloomery_gpu::Gpu;
use bloomery_gpu::GpuModel;
use bloomery_gpu_deepseek41::body::Body;
use bloomery_gpu_gates::GateError;

/// How far a card's free device bytes may sit under the reading the load
/// left: half the smallest per-slot store set a leaked sequence would hold,
/// so a leaked sequence crosses it, while the driver's own growth on a first
/// graph launch stays under it.
const CARD_FREE_SLACK: usize = 64 * 1024 * 1024;

/// The model's state one config leaves and the next must find again.
pub struct StateBack {
    /// The cache row the next fed token would land in.
    pos: u32,
    /// Whether a fault poisoned the model.
    poisoned: bool,
    /// Each card the load holds device memory on: its name and free bytes.
    cards: Vec<(String, usize)>,
}

impl StateBack {
    /// `m`'s state: its position, its poison word and every card's free
    /// device bytes.
    pub fn read(m: &GpuModel<Body>) -> Result<StateBack, GateError> {
        let body = m.body("state_back")?;
        let mut cards = Vec::with_capacity(1 + body.hybrid().tiers().len());
        cards.push(card("stage", m.gpu())?);
        for (t, tier) in body.hybrid().tiers().iter().enumerate() {
            cards.push(card(&format!("tier {t}"), tier.gpu())?);
        }
        // A tier's read bound its own context; the stage card's is current
        // again.
        m.gpu().context().bind_to_thread()?;
        Ok(StateBack {
            pos: m.pos(),
            poisoned: m.poisoned().is_some(),
            cards,
        })
    }

    /// Whether `now` is the state the load left (`self`) within the rules:
    /// the position and the poison word exactly, each card's free bytes
    /// within [`CARD_FREE_SLACK`] — free bytes above the load's are fine. A
    /// refusal names `at` (the arm or run it guards), the field, the card
    /// and both readings.
    pub fn check(&self, now: &StateBack, at: &str) -> Result<(), GateError> {
        if now.pos != self.pos {
            return Err(format!(
                "state back at {at}: the position is {}, the load left {}",
                now.pos, self.pos
            )
            .into());
        }
        if now.poisoned != self.poisoned {
            return Err(format!(
                "state back at {at}: a fault poisoned the model the load left clear"
            )
            .into());
        }
        for (b, n) in self.cards.iter().zip(&now.cards) {
            if n.1 + CARD_FREE_SLACK < b.1 {
                return Err(format!(
                    "state back at {at}: card {} holds {} B free, the load left {} B on it \
                     (slack {CARD_FREE_SLACK} B)",
                    n.0, n.1, b.1
                )
                .into());
            }
        }
        Ok(())
    }
}

/// One card's name (its role and the driver's name for it) and free device
/// bytes.
fn card(role: &str, gpu: &Gpu) -> Result<(String, usize), GateError> {
    Ok((format!("{role} {}", gpu.device_name()?), gpu.mem_info()?.0))
}
