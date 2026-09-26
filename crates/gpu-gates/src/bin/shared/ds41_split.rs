//! `stat prefill split`: a V4.1 prompt call's `body::PrefillStats` as its
//! record ([`record::STAT_PREFILL_SPLIT`]), for the binaries that print it —
//! `generate_ds41` after a batched feed, the prefill gate per case.

use bloomery_gpu_deepseek41::body::PrefillStats;
use bloomery_gpu_gates::record::{self, Record};

/// The host time by phase, summed and per layer-batch (the first batches'
/// waits over their own layer-batches), the queue entries and the
/// batch-excluded slots per layer-batch, and each layer-batch's card time
/// when the call was timed on the card.
pub fn split(s: &PrefillStats) -> Record {
    let ms = |ns: u64| ns as f64 / 1e6;
    let over = |v: f64, n: u64| if n == 0 { 0.0 } else { v / n as f64 };
    let per = |v: f64| over(v, s.layer_batches);
    let r = Record::new(&record::STAT_PREFILL_SPLIT)
        .u("group", s.group)
        .u("batches", s.batches)
        .u("layer_batches", s.layer_batches)
        .f("prologue_ms", ms(s.prologue_ns))
        .f("chain_ms", ms(s.chain_ns))
        .f("union_ms", ms(s.union_ns))
        .f("wait_ms", ms(s.wait_ns))
        .f("enqueue_ms", ms(s.enqueue_ns()))
        .f("copy_ms", ms(s.copy_ns))
        .f("union_lb", per(ms(s.union_ns)))
        .f("wait_lb", per(ms(s.wait_ns)))
        .f(
            "wait_first_lb",
            over(ms(s.wait_first_ns), s.first_layer_batches),
        )
        .f("enqueue_lb", per(ms(s.enqueue_ns())))
        .f("copy_lb", per(ms(s.copy_ns)))
        .f("entries_route", per(s.entries_route as f64))
        .f("entries_shadow", per(s.entries_shadow as f64))
        .f("excluded_lb", per(s.excluded_slots as f64));
    if s.card_timed {
        r.f("card_out_ms", s.card_out_ms)
            .f("card_in_ms", s.card_in_ms)
            .f("card_proj_ms", s.card_proj_ms)
            .f("card_out_lb", per(s.card_out_ms))
            .f("card_in_lb", per(s.card_in_ms))
            .f("card_proj_lb", per(s.card_proj_ms))
    } else {
        r.w("card", "untimed")
    }
}
