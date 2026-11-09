//! The elastic expert tenant's record lines, for the seats that settle it
//! (`bloomery-serve-qwen38`): each move of a settle
//! (`bloomery_gpu::settle_to_margin`) as its record and the tenant's counters
//! after it ([`record::groups::elastic::GROUP`]).

use bloomery_gpu::{Elastic, ElasticStats};
use bloomery_gpu_gates::record::{self, Record};

/// One move of a settle as its record ([`record::ELASTIC_GIVE_BACK`],
/// [`record::ELASTIC_FILL`] or [`record::ELASTIC_HELD`]).
pub fn settle(e: &Elastic) -> Record {
    record::require(record::groups::elastic::GROUP, "elastic::settle");
    match e {
        Elastic::GiveBack { site, asked, freed } => Record::new(&record::ELASTIC_GIVE_BACK)
            .w("site", site)
            .u("asked", asked)
            .u("freed", freed.bytes)
            .u("slots", freed.slots)
            .u("layers", freed.layers),
        Elastic::Fill {
            site,
            boundary,
            free,
            margin,
            filled,
        } => Record::new(&record::ELASTIC_FILL)
            .w("site", site)
            .u("boundary", boundary)
            .u("free", free)
            .u("margin", margin)
            .u("slots", filled.slots)
            .u("layers", filled.layers)
            .u("bytes", filled.bytes),
        Elastic::Held { site, why } => Record::new(&record::ELASTIC_HELD)
            .w("site", site)
            .w("why", why),
    }
}

/// The warm-up's reading as its record ([`record::ELASTIC_WARM`]): the prompt
/// pass's `ids` (and the `ubatch` and `ctx` it is cut from), the card bytes it left taken and the two free readings.
pub fn warm(
    (ids, ubatch, ctx): (usize, usize, usize),
    pass_bytes: u64,
    (free_before, free): (u64, u64),
    margin: u64,
) -> Record {
    record::require(record::groups::elastic::GROUP, "elastic::warm");
    Record::new(&record::ELASTIC_WARM)
        .u("ids", ids)
        .u("ubatch", ubatch)
        .u("ctx", ctx)
        .u("pass_bytes", pass_bytes)
        .u("free_before", free_before)
        .u("free", free)
        .u("margin", margin)
}

/// The tenant's counters since the load as their record ([`record::STAT_ELASTIC`]).
pub fn stat(s: &ElasticStats) -> Record {
    record::require(record::groups::elastic::GROUP, "elastic::stat");
    Record::new(&record::STAT_ELASTIC)
        .u("give_backs", s.give_backs)
        .u("give_back_bytes", s.give_back_bytes)
        .u("fills", s.fills)
        .u("fill_bytes", s.fill_bytes)
        .u("boundary", s.boundary)
}
