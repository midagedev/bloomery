//! A card byte budget: every card of a plan plans as if it had at most this
//! many usable bytes — `min(usable, budget)` — so one large card can stand in
//! for any smaller one. The expert rule then runs on the same usable − KV −
//! context − scratch − margin arithmetic and keeps fewer experts.
//!
//! The value is decimal bytes with optional `_` separators, or a whole
//! number of MiB or GiB with an `M` or `G` suffix (binary units: `38G` is
//! 40,802,189,312 B).

/// Bytes a value names ([`bloomery_levers::parse_bytes`]): the parse the
/// `BLOOMERY_CARD_BUDGET` lever's kind runs, and the fixture builder's byte
/// flags.
pub use bloomery_levers::parse_bytes as parse;
