//! The census a plan resolves its cards against, with what the driver
//! cannot name: [`bloomery_gpu::census`] (behind the `gpu` feature, as
//! [`census`] is) reads each device's identity and its free bytes on the
//! primary context; this module adds the processes holding each device, as
//! `nvidia-smi` names them — one query, best effort. A tool that does not
//! answer names no holder, and a plan never fails for want of the answer:
//! the free bytes alone still cap it.

#[cfg(feature = "gpu")]
use std::process::Command;

#[cfg(feature = "gpu")]
use model::placement::workstation::DeviceInfo;

#[cfg(feature = "gpu")]
use crate::GateError;

/// The visible devices with each card's free bytes and, when nvidia-smi
/// answers, the other processes holding it: the census a plan resolves
/// against ([`Place::on_host`](crate::generate::Place::on_host)). Our own
/// pid is left out of a holder list: the plan sizes what this process is
/// about to take, not what it holds now.
#[cfg(feature = "gpu")]
pub fn census() -> Result<Vec<DeviceInfo>, GateError> {
    let mut census = bloomery_gpu::census()?;
    let Some(rows) = compute_apps() else {
        return Ok(census);
    };
    let mine = std::process::id();
    for d in &mut census {
        let held: Vec<String> = rows
            .iter()
            .filter(|(uuid, pid, _)| *uuid == gpu_uuid(&d.uuid) && *pid != mine)
            .map(|(_, pid, used)| format!("pid {pid} ({used})"))
            .collect();
        if !held.is_empty() {
            d.held_by = Some(held.join(", "));
        }
    }
    Ok(census)
}

/// nvidia-smi's compute-app listing as `(gpu uuid, pid, used text)` rows: a
/// pure read of the tool's `csv,noheader` text. A row that is not a uuid, a
/// pid and a used figure gives `None` — no holder is named, never an error.
#[cfg(any(feature = "gpu", test))]
fn compute_apps_of(text: &str) -> Option<Vec<(String, u32, String)>> {
    let mut rows = Vec::new();
    for line in text.lines() {
        let mut cells = line.split(',');
        let (uuid, pid, used) = (cells.next()?, cells.next()?, cells.next()?);
        let pid = pid.trim().parse::<u32>().ok()?;
        rows.push((uuid.trim().to_string(), pid, used.trim().to_string()));
    }
    Some(rows)
}

/// [`compute_apps_of`] of the tool's answer; a tool that is missing or fails
/// gives `None`.
#[cfg(feature = "gpu")]
fn compute_apps() -> Option<Vec<(String, u32, String)>> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    compute_apps_of(&String::from_utf8_lossy(&out.stdout))
}

/// A device UUID as nvidia-smi spells it: `GPU-` and the sixteen bytes in
/// the driver's 8-4-4-4-12 groups.
#[cfg(any(feature = "gpu", test))]
fn gpu_uuid(bytes: &[u8; 16]) -> String {
    let hex = |b: &[u8]| {
        b.iter()
            .map(|x| format!("{x:02x}"))
            .collect::<Vec<_>>()
            .join("")
    };
    format!(
        "GPU-{}-{}-{}-{}-{}",
        hex(&bytes[0..8]),
        hex(&bytes[8..10]),
        hex(&bytes[10..12]),
        hex(&bytes[12..14]),
        hex(&bytes[14..16])
    )
}

#[cfg(test)]
mod tests {
    use super::{compute_apps_of, gpu_uuid};

    /// The uuid spelling matches nvidia-smi's: the 8-4-4-4-12 groups of the
    /// driver's sixteen bytes.
    #[test]
    fn the_uuid_is_nvidia_smis() {
        let mut bytes = [0u8; 16];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = u8::try_from(0x10 + i).expect("small");
        }
        assert_eq!(gpu_uuid(&bytes), "GPU-1011121314151617-1819-1a1b-1c1d-1e1f");
    }

    /// The listing's rows read as uuid, pid and the used figure as the tool
    /// spelled it; a row that is none of that gives no listing.
    #[test]
    fn the_listing_reads_its_rows() {
        let text = "GPU-1011121314151617-1819-1a1b-1c1d-1e1f, 4321, 977 MiB\n\
                    GPU-0000000000000000-0000-0000-0000-0000, 55, 2 MiB\n";
        assert_eq!(
            compute_apps_of(text),
            Some(vec![
                (
                    "GPU-1011121314151617-1819-1a1b-1c1d-1e1f".to_string(),
                    4321,
                    "977 MiB".to_string()
                ),
                (
                    "GPU-0000000000000000-0000-0000-0000-0000".to_string(),
                    55,
                    "2 MiB".to_string()
                )
            ])
        );
        assert_eq!(compute_apps_of(""), Some(Vec::new()));
        assert_eq!(compute_apps_of("no gpu today\n"), None);
        assert_eq!(compute_apps_of("GPU-x, notapid, 1 MiB\n"), None);
    }
}
