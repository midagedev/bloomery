//! The host tier's step statistics under `BLOOMERY_STEP_STATS`, the one
//! printer the generate binaries whose body has a host tier and no engram
//! rows share (`generate_qwen3moe`, `generate_glm5next`): a probe is one
//! read of the body's host tier — its counters since load and, beside the
//! probe before, the go waits of the services in between — with the
//! process's page faults and the card's free device bytes, and the records
//! are [`crate::record::STAT_STEP_HOST`] and
//! [`crate::record::STAT_SUMMARY_HOST`]. `generate_ds41` keeps its own
//! probe beside these (its engram fields have no host-tier reader).

use crate::GateError;
use crate::record::{self, Record};
use bloomery_gpu::Gpu;
use bloomery_gpu::host::HostExperts;
use bloomery_gpu::host::HostTier;
use bloomery_gpu::host::step::GapSummary;
use bloomery_gpu::hybrid::HybridStats;

/// The counters one `BLOOMERY_STEP_STATS` read takes: the host tier's since
/// load, the go waits of the services since the probe before, the process's
/// page faults since start, and the card's free device bytes now.
pub struct Probe {
    hybrid: HybridStats,
    /// The go waits of the services since the probe before; zeros at the
    /// first read, and over a span no service ran in.
    gap: GapSummary,
    majflt: u64,
    minflt: u64,
    vram_free: u64,
}

impl Probe {
    /// One read of `tier` with the process's faults and `gpu`'s free device
    /// bytes: the tier's counters since load and the go waits of the
    /// services since `prev`'s read (`None` before the first).
    pub fn read<H: HostExperts>(
        tier: &HostTier<H>,
        gpu: &Gpu,
        prev: Option<&Probe>,
    ) -> Result<Probe, GateError> {
        let hybrid = tier.stats();
        let gap = match prev {
            Some(p) => tier.gap_summary(p.hybrid.gaps, hybrid.gaps)?,
            None => GapSummary::default(),
        };
        let (minflt, majflt) = faults()?;
        Ok(Probe {
            hybrid,
            gap,
            majflt,
            minflt,
            vram_free: u64::try_from(gpu.mem_info()?.0)?,
        })
    }
}

/// The process's minor and major page faults since start: fields 10 and
/// 12 of `/proc/self/stat`, counted past the command's closing
/// parenthesis.
pub fn faults() -> Result<(u64, u64), GateError> {
    let stat = std::fs::read_to_string("/proc/self/stat")?;
    let rest = stat
        .rsplit_once(')')
        .map(|(_, r)| r)
        .ok_or("/proc/self/stat: no command field")?;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let at = |i: usize| -> Result<u64, GateError> {
        Ok(fields
            .get(i)
            .ok_or_else(|| format!("/proc/self/stat: {} fields past the command", fields.len()))?
            .parse::<u64>()?)
    };
    Ok((at(7)?, at(9)?))
}

/// One `stat step` record per generated step from the deltas of `probes`
/// (one read before the first generated step, one after each), then the
/// `stat summary` over the steps past `warm`; nothing without two reads.
/// `leg_us` and `host_slots` are summed over the step's `served` layers;
/// `vram_free` is the read after the step, not a delta; `gap_*_us` are the
/// go waits of the step's services (the probe after the step holds them),
/// zeros when it ran none.
pub fn print_stats(probes: &[Probe], warm: usize) {
    let mut legs: Vec<f64> = Vec::with_capacity(probes.len());
    let (mut straggle_max, mut slots, mut majflt, mut minflt) = (0.0_f64, 0_u64, 0_u64, 0_u64);
    let mut vram_free_min = u64::MAX;
    for (k, w) in probes.windows(2).enumerate() {
        let i = k + 1;
        let (p, q) = (&w[0].hybrid, &w[1].hybrid);
        let served = q.served - p.served;
        let leg_us = (q.leg_ns - p.leg_ns) as f64 / 1e3;
        let straggle_us = (q.straggle_ns - p.straggle_ns) as f64 / 1e3;
        let host_slots = q.host_slots - p.host_slots;
        let host_w2 = if served == 0 {
            0.0
        } else {
            (q.host_w2 - p.host_w2) / served as f64
        };
        let (dmaj, dmin) = (w[1].majflt - w[0].majflt, w[1].minflt - w[0].minflt);
        Record::new(&record::STAT_STEP_HOST)
            .u("i", i)
            .flag("warm", i <= warm)
            .u("served", served)
            .f("leg_us", leg_us)
            .f("straggle_us", straggle_us)
            .f("straggle_max_us", q.straggle_max_ns as f64 / 1e3)
            .u("host_slots", host_slots)
            .f("host_w2", host_w2)
            .u("go_early", q.go_early - p.go_early)
            .u("parks", q.parks_in_service - p.parks_in_service)
            .u("majflt", dmaj)
            .u("minflt", dmin)
            .u("vram_free", w[1].vram_free)
            .f("gap_min_us", w[1].gap.min_ns as f64 / 1e3)
            .f("gap_p50_us", w[1].gap.p50_ns as f64 / 1e3)
            .f("gap_max_us", w[1].gap.max_ns as f64 / 1e3)
            .print();
        if i > warm {
            legs.push(leg_us);
            straggle_max = straggle_max.max(straggle_us);
            slots += host_slots;
            majflt += dmaj;
            minflt += dmin;
            vram_free_min = vram_free_min.min(w[1].vram_free);
        }
    }
    let n = legs.len();
    if n == 0 {
        return;
    }
    let mean = legs.iter().sum::<f64>() / n as f64;
    legs.sort_by(f64::total_cmp);
    Record::new(&record::STAT_SUMMARY_HOST)
        .u("steps", n)
        .f("leg_us_mean", mean)
        .f("leg_us_p50", lower_median(&legs))
        .f("straggle_us_max", straggle_max)
        .f("host_slots_mean", slots as f64 / n as f64)
        .u("majflt", majflt)
        .u("minflt", minflt)
        .u("vram_free_load", probes[0].vram_free)
        .u("vram_free_min", vram_free_min)
        .print();
}

/// The lower median of `sorted` (ascending, at least one value): the value at
/// `(n − 1) / 2`, the convention of the go waits' p50 ([`GapSummary`]), so a
/// summary's `_p50` fields mean one thing.
#[must_use]
pub fn lower_median(sorted: &[f64]) -> f64 {
    assert!(!sorted.is_empty(), "lower_median of no values");
    sorted[(sorted.len() - 1) / 2]
}
