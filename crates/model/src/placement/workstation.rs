//! This workstation's figures, re-exported from `bloomery-placement` — the
//! pure crate holds the specs, the layer maps, the reserve formulas and the
//! host-need arithmetic. What stays here is the one reading of the machine
//! the Mac cannot do.

pub use bloomery_placement::placement::workstation::*;

/// The host's available bytes now ([`host_room`]): the smaller of the
/// platform's own reading of the host's memory — Linux's `MemAvailable` in
/// `/proc/meminfo`, macOS's free, inactive and purgeable pages — and the
/// room under every cgroup v2 memory limit above this process — inside a
/// container (`docker run --memory`), a systemd scope (`MemoryMax`) or a
/// pod, the platform's reading still shows the whole machine, and the
/// limit's room is the reading that binds. `BLOOMERY_HOST_ROOM` set
/// (bytes, `M` and `G` binary units, a parsed lever) replaces both, read
/// [`HostRead::Given`]; a value that is not bytes is refused by name. A
/// base page the row room cannot count in is refused even when the room is
/// supplied by the caller.
pub fn host_available() -> Result<u64, String> {
    host_available_read().map(|(bytes, _)| bytes)
}

/// [`host_available`] with the reading that decided ([`HostRead`]), for
/// every refusal and line that prints the bytes to name.
pub fn host_available_read() -> Result<(u64, HostRead), String> {
    // The base page the row room counts in: a page the platform cannot
    // count in is refused by name, before the room is read.
    os::page_bytes().map_err(|e| e.to_string())?;
    let levers = bloomery_levers::nvtier_levers().map_err(|e| e.to_string())?;
    if let Some(room) = levers.host_room {
        return Ok((room, HostRead::Given));
    }
    let memory = os::host_memory().map_err(|e| e.to_string())?;
    let levels = cgroup_levels()?;
    let texts: Vec<CgroupLevel<'_>> = levels
        .iter()
        .map(|(path, max, current, stat)| CgroupLevel {
            path,
            max,
            current,
            stat,
        })
        .collect();
    host_room(memory.available_bytes, platform_reading(), &texts)
}

/// The platform's own reading of the host's available bytes, as the host
/// room names it.
#[cfg(target_os = "linux")]
fn platform_reading() -> HostRead {
    HostRead::MemAvailable
}

/// The platform's own reading of the host's available bytes, as the host
/// room names it.
#[cfg(target_os = "macos")]
fn platform_reading() -> HostRead {
    HostRead::VmStats
}

/// The arena's bytes `BLOOMERY_NVTIER_BYTES` gives the NVMe expert tier
/// ([`bloomery_placement::placement::expert_nvme_tier`]'s `given`), `None`
/// unset: read once with the room by a family's `describe`, so a gate that
/// plans an arm sets the field of its own. A value that is not bytes is
/// refused by name.
pub fn nvtier_arena_given() -> Result<Option<u64>, String> {
    bloomery_levers::nvtier_levers()
        .map(|levers| levers.bytes)
        .map_err(|e| format!("BLOOMERY_NVTIER_BYTES: {e}"))
}

/// One cgroup v2 level's file, read under `/sys/fs/cgroup`.
fn cgroup_read(path: &str, file: &str) -> std::io::Result<String> {
    std::fs::read_to_string(format!("/sys/fs/cgroup{path}/{file}"))
}

/// The cgroup v2 levels above this process — its own cgroup first, then
/// every ancestor to the root — as each level's `memory.max`,
/// `memory.current` and `memory.stat` texts. The unified hierarchy binds
/// this process's memory only when the `memory` controller rides it
/// (`/sys/fs/cgroup/cgroup.controllers` names it): a system with no cgroup
/// v2 at all, and a hybrid one that keeps memory on cgroup v1 — the
/// unified root then exposes no `memory.max` — can hold no limit this
/// reading sees, so both read "no levels" (`MemAvailable` alone; v1 limits
/// are not read). A `/proc/self/cgroup` with no unified-hierarchy line has
/// none the same way. Where the controller does ride v2, a level with
/// no `memory.max` is one the controller is not enabled on (its parent's
/// `cgroup.subtree_control` leaves `memory` out, and the root never has
/// one), so it sets no limit and is skipped; any other failure to read a
/// level's files is an error by name — the limit it may hold is not
/// silently dropped.
fn cgroup_levels() -> Result<Vec<(String, String, String, String)>, String> {
    let controllers = match std::fs::read_to_string("/sys/fs/cgroup/cgroup.controllers") {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("read /sys/fs/cgroup/cgroup.controllers: {e}")),
    };
    if !controllers.split_whitespace().any(|c| c == "memory") {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string("/proc/self/cgroup")
        .map_err(|e| format!("read /proc/self/cgroup: {e}"))?;
    let Some(mut path) = cgroup_v2_path(&text)? else {
        return Ok(Vec::new());
    };
    let read = |path: &str, file: &str| -> Result<String, String> {
        cgroup_read(path, file).map_err(|e| format!("read /sys/fs/cgroup{path}/{file}: {e}"))
    };
    let mut levels = Vec::new();
    loop {
        match cgroup_read(&path, "memory.max") {
            Ok(max) => levels.push((
                path.clone(),
                max,
                read(&path, "memory.current")?,
                read(&path, "memory.stat")?,
            )),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("read /sys/fs/cgroup{path}/memory.max: {e}")),
        }
        if path == "/" {
            return Ok(levels);
        }
        path = match path.rsplit_once('/') {
            Some((head, _)) if !head.is_empty() => head.to_owned(),
            _ => "/".to_owned(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::host_available_read;

    /// The host's room reads on this platform, named by the platform's own
    /// reading, and the base page the row room counts in is one the
    /// platform can count in.
    #[test]
    fn the_host_room_reads_on_this_platform() {
        let page = crate::placement::host_lock::page_bytes().expect("the base page reads");
        assert!(page >= 4096, "the base page is {page} B");
        assert!(page.is_power_of_two(), "the base page is {page} B");
        let (bytes, read) = host_available_read().expect("the host room reads");
        assert!(bytes > 0, "the host has memory available");
        // Linux: the room is `/proc/meminfo`'s `MemAvailable`. macOS: the
        // host statistics' free, inactive and purgeable pages.
        // A cgroup limit above this process, or `BLOOMERY_HOST_ROOM`, may name
        // the reading in place of the platform's own.
        let word = read.word();
        #[cfg(target_os = "linux")]
        assert!(
            word == "MemAvailable" || word.starts_with("cgroup:") || word == "given",
            "{word}"
        );
        #[cfg(target_os = "macos")]
        assert!(word == "vmstats" || word == "given", "{word}");
    }
}
