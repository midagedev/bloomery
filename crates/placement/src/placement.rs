//! Where every tensor of a model lives on this machine: one row per tensor with
//! its device, device format, resident bytes and per-token read, the per-device
//! totals, and a check of the invariants the table must keep.
//!
//! The inputs are header facts and device figures only: the model as its
//! architecture classifies it ([`ModelTensors`], one [`Role`] per tensor), the
//! GPU cards in stage order with the layers each runs and the host with its
//! reserves ([`Machine`]), `ctx_max`, and the architecture's KV size
//! ([`KvBytes`]). This module spells no tensor name and knows no architecture;
//! the classifier that feeds it does.
//!
//! Routed experts follow one rule ([`plan`]): the eligible layers of a card
//! keep the same number of experts on it, give or take one — the count
//! `n_l` — and the rest stay on the host. Which experts a layer keeps is an
//! [`ExpertList`]: the id prefix `[0, n_l)`. A card byte budget
//! (`BLOOMERY_CARD_BUDGET`, [`card_budget`]: one for every card, or one per
//! card) caps a card's usable bytes before the rule runs — the lever comes in
//! as [`PlanLevers`], which a binary parses once — and the census's free
//! reading caps them below that:
//! while another process holds the device, fewer experts fit
//! ([`Card::free_bytes`], [`capped`]).
//!
//! A machine may also carry expert tier cards ([`Machine::tiers`]): cards
//! that run no stage and hold only routed experts beside the host. The stage
//! cards are planned first, exactly as without tiers; then each tier, in
//! order, takes per eligible layer the ids after the stage card's `n_l` and
//! the earlier tiers', in id order, one
//! expert at a time on the layer that holds the fewest so far, so the
//! per-layer totals stay even ([`Plan::tier_n_l`]).

pub mod card_budget;
pub mod churn;
pub mod ctx;
pub mod ctx_ask;
pub mod devices;
pub mod need_hook;
pub mod nodes;
pub mod paged_drop;
pub mod tp;
pub mod workstation;

#[cfg(test)]
mod nodes_scale;
#[cfg(test)]
mod qwen38_cards;

mod machine;
mod planner;
mod pool;
mod refusal;
mod room;
mod rule;
mod table;

#[cfg(test)]
mod tests;

pub use gguf::card::CardFormat;
pub use machine::{
    Card, Device, ExpertList, Format, HEAD_TENSOR, Host, KvBytes, Machine, ModelTensor,
    ModelTensors, Node, Requantized, Reserve, UnifiedPool, allocator_bytes, card_format_of_role,
    file_card_format, stack_experts,
};
pub use models::{Role, Unimplemented};
pub use planner::{
    PlanLevers, plan, plan_host_routed, plan_host_routed_reserving, plan_routed,
    plan_routed_reserving, plan_with,
};
pub use pool::{PoolNeed, plan_in_pool};
pub use refusal::{CardFreeFloor, PlacementError, Violation};
pub use room::{
    ARENA_ALIGN, ARENA_LANE_SLOTS, ArenaDial, ArenaFloor, RowTableSource, STEP_PORT_COLUMNS,
    UNION_BATCH_IDS, WholeLoad, WholeNeed, arena_part_bytes, arena_slots, arena_window_bytes,
    expert_nvme_tier, host_room_given, nvme_arena_of, row_room, row_table_tier, whole_need,
};
pub use rule::{
    ExpertPart, HostOnly, LegPair, RoutedFormat, RoutedRule, per_expert, routed_row, routed_row_on,
    whole_on_card,
};
pub use table::{CardTotals, HostTotals, Plan, Row, Segment, Span, union_plan};

use machine::NoKv;
use planner::{check_devices, check_tiers};
use pool::totals;
use rule::{card_bytes, card_segment_as, in_place, row_bytes};

#[cfg(test)]
use rule::Stages;
