//! `generate_ds41 --preflight`, in a build with the `checkpoint` feature: the
//! original checkpoint folder's headers (`model_hf::folder::open`) read into each
//! rank's `tp need` record.

use crate::{GateError, gpu_census, record};
use model::arch::deepseek41::tp::RunShape;
use model::placement::tp::RankPool;
use model::placement::workstation::{self, DeviceInfo};
use model_hf::arch::deepseek41::{preflight as rank_ledgers, rank_reads};

/// `--preflight <folder> --ranks N[,N...] [--ctx C]`: each rank's `tp need` record.
pub fn preflight(args: &[String]) -> Result<(), GateError> {
    let usage = "usage: generate_ds41 --preflight <folder> --ranks N[,N...] [--ctx C]";
    let (mut dir, mut counts, mut ctx) = (None, Vec::new(), workstation::CTX_MAX);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value: {usage}"))?;
        match flag.as_str() {
            "--preflight" => dir = Some(std::path::PathBuf::from(v)),
            "--ranks" => {
                counts = v
                    .split(',')
                    .map(|n| n.trim().parse::<usize>())
                    .collect::<Result<_, _>>()?;
            }
            "--ctx" => ctx = v.parse()?,
            _ => return Err(format!("{flag} is not a preflight flag: {usage}").into()),
        }
    }
    let dir = dir.ok_or(usage)?;
    if counts.is_empty() {
        return Err(usage.into());
    }
    let census = gpu_census::census()?;
    let [here] = census.as_slice() else {
        return Err(format!(
            "--preflight: this process sees {} devices, not one",
            census.len()
        )
        .into());
    };
    let pool = unified_pool(here)?;
    let opened =
        model_hf::folder::open(&dir).map_err(|e| format!("--preflight {}: {e}", dir.display()))?;
    let run = RunShape {
        ctx,
        slots: 1,
        batch: 1,
    };
    let mut refused = Vec::new();
    for ranks in counts {
        let pools = vec![
            RankPool {
                device: None,
                ..pool.clone()
            };
            ranks
        ];
        let ledgers = match rank_ledgers(&opened, &pools, run) {
            Ok(ledgers) => ledgers,
            Err(e) => {
                refused.push(format!("at {ranks} ranks: {e}"));
                continue;
            }
        };
        for l in &ledgers {
            let reads = rank_reads(&opened, l.rank, ranks)
                .map_err(|e| format!("--preflight rank {} of {ranks}: {e}", l.rank))?;
            record::tp_need(l, ranks, &reads).print();
        }
    }
    if refused.is_empty() {
        return Ok(());
    }
    Err(format!("--preflight refused {}", refused.join("; ")).into())
}

/// The pool every rank is counted against: `here`'s node's, on a device that allocates from host
/// memory. A rank's need sums device allocations with host ones (ring shadows, row pages, read
/// buffers), which only one pool for both holds, so a card of its own memory is refused by name.
fn unified_pool(here: &DeviceInfo) -> Result<RankPool, GateError> {
    let spec = workstation::spec_of_device(here);
    if workstation::unified_of(&spec).is_none() {
        return Err(
            "--preflight requires unified memory; host and device budgets are separate on this card"
                .into(),
        );
    }
    Ok(RankPool::of("local".into(), &spec))
}

#[cfg(test)]
mod tests {
    use model::placement::workstation::DeviceInfo;

    /// A device that allocates from host memory gives its node's pool; a card of its own memory
    /// is refused by name.
    #[test]
    fn preflight_counts_only_a_unified_pool() {
        let device = |integrated| DeviceInfo {
            ordinal: 0,
            name: "card".into(),
            total_bytes: 128 << 30,
            free_bytes: 100 << 30,
            integrated,
            uuid: [1; 16],
            pci_bus: "0000:01:00.0".into(),
            held_by: None,
            compute_capability: Some((12, 1)),
            granule_bytes: None,
        };
        assert_eq!(
            super::unified_pool(&device(true)).unwrap().pool_bytes,
            100 << 30
        );
        let e = super::unified_pool(&device(false)).unwrap_err();
        assert!(e.to_string().contains("requires unified memory"), "{e}");
    }
}
