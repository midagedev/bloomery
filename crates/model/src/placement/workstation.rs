//! This workstation's figures, re-exported from `bloomery-placement` — the
//! pure crate holds the specs, the layer maps, the reserve formulas and the
//! host-need arithmetic. What stays here is the one reading of the machine
//! the Mac cannot do.

pub use bloomery_placement::placement::workstation::*;

/// The host's available bytes now: [`mem_available`] of `/proc/meminfo`.
pub fn host_available() -> Result<u64, String> {
    let text =
        std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("read /proc/meminfo: {e}"))?;
    mem_available(&text)
}
