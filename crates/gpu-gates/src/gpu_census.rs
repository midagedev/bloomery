//! The census a plan resolves its cards against, with what the driver
//! cannot name: [`bloomery_gpu::census`] (behind the `gpu` feature, as
//! [`census`] is) reads each device's identity and its free bytes on the
//! primary context; this module adds the processes holding each device, as
//! `nvidia-smi` names them — one query, best effort. A tool that does not
//! answer names no holder, and a plan never fails for want of the answer:
//! the free bytes alone still cap it.

#[cfg(feature = "gpu")]
use std::process::Command;

#[cfg(any(feature = "gpu", test))]
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
    name_holders(&mut census, &rows, std::process::id());
    Ok(census)
}

/// Each device's holders from the listing's `rows` (`compute_apps_of`): the
/// rows on the device's uuid as nvidia-smi spells it ([`gpu_uuid`]), every pid
/// but `mine`, as one named list in [`DeviceInfo::held_by`].
#[cfg(any(feature = "gpu", test))]
fn name_holders(census: &mut [DeviceInfo], rows: &[(String, u32, String)], mine: u32) {
    for d in census {
        let held: Vec<String> = rows
            .iter()
            .filter(|(uuid, pid, _)| *uuid == gpu_uuid(&d.uuid) && *pid != mine)
            .map(|(_, pid, used)| format!("pid {pid} ({used})"))
            .collect();
        if !held.is_empty() {
            d.held_by = Some(held.join(", "));
        }
    }
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
#[must_use]
pub fn gpu_uuid(bytes: &[u8; 16]) -> String {
    let hex = |b: &[u8]| {
        b.iter()
            .map(|x| format!("{x:02x}"))
            .collect::<Vec<_>>()
            .join("")
    };
    format!(
        "GPU-{}-{}-{}-{}-{}",
        hex(&bytes[0..4]),
        hex(&bytes[4..6]),
        hex(&bytes[6..8]),
        hex(&bytes[8..10]),
        hex(&bytes[10..16])
    )
}

#[cfg(test)]
mod tests {
    use super::{DeviceInfo, compute_apps_of, gpu_uuid, name_holders};

    /// The uuid spelling matches nvidia-smi's: the 8-4-4-4-12 groups of the
    /// driver's sixteen bytes.
    #[test]
    fn the_uuid_is_nvidia_smis() {
        let mut bytes = [0u8; 16];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = u8::try_from(0x10 + i).expect("small");
        }
        assert_eq!(gpu_uuid(&bytes), "GPU-10111213-1415-1617-1819-1a1b1c1d1e1f");
        // The box's A6000 as nvidia-smi --query-gpu=uuid prints it.
        let a6000 = [
            0x8c, 0x12, 0x9f, 0xa6, 0x73, 0x82, 0x35, 0xa5, 0x24, 0x64, 0x9f, 0xf0, 0x1d, 0x99,
            0xfc, 0xd4,
        ];
        assert_eq!(gpu_uuid(&a6000), "GPU-8c129fa6-7382-35a5-2464-9ff01d99fcd4");
    }

    /// The listing's rows read as uuid, pid and the used figure as the tool
    /// spelled it; a row that is none of that gives no listing.
    #[test]
    fn the_listing_reads_its_rows() {
        let text = "GPU-10111213-1415-1617-1819-1a1b1c1d1e1f, 4321, 977 MiB\n\
                    GPU-00000000-0000-0000-0000-000000000000, 55, 2 MiB\n";
        assert_eq!(
            compute_apps_of(text),
            Some(vec![
                (
                    "GPU-10111213-1415-1617-1819-1a1b1c1d1e1f".to_string(),
                    4321,
                    "977 MiB".to_string()
                ),
                (
                    "GPU-00000000-0000-0000-0000-000000000000".to_string(),
                    55,
                    "2 MiB".to_string()
                )
            ])
        );
        assert_eq!(compute_apps_of(""), Some(Vec::new()));
        assert_eq!(compute_apps_of("no gpu today\n"), None);
        assert_eq!(compute_apps_of("GPU-x, notapid, 1 MiB\n"), None);
    }
    /// A card the listing names a compute app on gets that holder: the
    /// listing's own text (two cards' rows as nvidia-smi prints them on the
    /// box) against the driver's uuid bytes, so the uuid spelling and the
    /// match cannot drift apart unseen. Our own pid is never a holder.
    #[test]
    fn a_listed_compute_app_names_its_card_holder() {
        let device = |ordinal: u32, uuid: [u8; 16]| DeviceInfo {
            ordinal,
            name: String::new(),
            total_bytes: 0,
            free_bytes: 0,
            uuid,
            pci_bus: String::new(),
            held_by: None,
        };
        let a6000 = [
            0x8c, 0x12, 0x9f, 0xa6, 0x73, 0x82, 0x35, 0xa5, 0x24, 0x64, 0x9f, 0xf0, 0x1d, 0x99,
            0xfc, 0xd4,
        ];
        let rtx3090 = [
            0x30, 0x7f, 0xa0, 0xf6, 0xda, 0xae, 0x24, 0xe5, 0x6f, 0xd3, 0xcd, 0x50, 0x62, 0x0d,
            0xe6, 0xb1,
        ];
        let text = "GPU-307fa0f6-daae-24e5-6fd3-cd50620de6b1, 3182623, 977 MiB\n\
                    GPU-8c129fa6-7382-35a5-2464-9ff01d99fcd4, 3165383, 21540 MiB\n\
                    GPU-8c129fa6-7382-35a5-2464-9ff01d99fcd4, 77, 300 MiB\n";
        let rows = compute_apps_of(text).expect("the listing reads");
        let mut census = vec![device(0, rtx3090), device(1, a6000)];
        name_holders(&mut census, &rows, 77);
        assert_eq!(census[0].held_by.as_deref(), Some("pid 3182623 (977 MiB)"));
        assert_eq!(
            census[1].held_by.as_deref(),
            Some("pid 3165383 (21540 MiB)")
        );
    }
}
