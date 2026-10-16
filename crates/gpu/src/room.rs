//! The host's room, read once the cards a plan names are held: the one
//! owner of a reading of this machine's room ([`HeldRoom`]). A device's
//! first open takes host bytes the process keeps for its life — the primary
//! context, held as the anchor, and this crate's bundle, loaded once on it —
//! so a room read before it is a room no load of the plan can find. Every
//! plan's reading and every load's check go through [`HeldRoom::read`], which
//! holds each card first; the reader itself is private here, so no reading
//! skips the hold.

use ::model::placement::Machine;
use ::model::placement::room::{LoadReserves, process_pages};
use ::model::placement::workstation::{
    CgroupLevel, DeviceId, HostRead, device_on_host, host_room, mem_available,
};

use crate::{Gpu, GpuError, census, held};

/// One reading of the host's room, taken once every card a plan names was
/// held ([`HeldRoom::read`]), or a room a caller gave ([`HeldRoom::given`]):
/// the bytes and the reading that decided, with what the process's own pages
/// and `MemAvailable` were at the same moment — the pages the witness that
/// a plan's room is its load's (`room plan` and `room load` records), and
/// `MemAvailable` for the line alone, since every other process moves it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldRoom {
    bytes: u64,
    read: HostRead,
    pages: u64,
    mem_available: u64,
    held: Vec<String>,
}

impl HeldRoom {
    /// Hold every card of `cards` — by the name and device a plan gives
    /// it, resolved against this process's census ([`device_on_host`]) —
    /// then read the host's room: a card this process has not opened is
    /// opened and dropped ([`Gpu::hold_card`]'s hold), one it holds already
    /// opens nothing. `BLOOMERY_HOST_ROOM` set gives the room instead
    /// ([`HostRead::Given`]): nothing of this machine's room is read, so no
    /// card is held for it. No card named, a card the census does not
    /// resolve, a room that cannot be read and a process status that cannot
    /// be read are refused by name.
    pub fn read<'a>(
        cards: impl IntoIterator<Item = (&'a str, Option<DeviceId>)>,
    ) -> Result<HeldRoom, GpuError> {
        const WHAT: &str = "HeldRoom::read";
        let levers = bloomery_levers::nvtier_levers().map_err(|e| GpuError::plan(WHAT, e))?;
        if let Some(bytes) = levers.host_room {
            return HeldRoom::given(bytes);
        }
        let cards: Vec<(&str, Option<DeviceId>)> = cards.into_iter().collect();
        if cards.is_empty() {
            return Err(GpuError::plan(
                WHAT,
                "no card named: a reading of the host's room holds the cards its plan names first",
            ));
        }
        let seen = census()?;
        for &(name, device) in &cards {
            let ordinal =
                device_on_host(name, device, &seen).map_err(|e| GpuError::plan(WHAT, e))?;
            let ordinal = usize::try_from(ordinal).expect("a u32 ordinal fits usize");
            if !held(ordinal) {
                Gpu::hold_card(name, device)?;
            }
        }
        HeldRoom::reading(cards.iter().map(|&(name, _)| name.to_owned()).collect())
    }

    /// [`HeldRoom::read`] of every card `machine` names: its stage cards,
    /// then its tier cards.
    pub fn read_machine(machine: &Machine) -> Result<HeldRoom, GpuError> {
        HeldRoom::read(machine.all_cards().map(|c| (c.name.as_str(), c.device)))
    }

    /// The host's room now, `held` naming the cards held before it: the
    /// reading [`HeldRoom::read`] takes once its cards are held, and the
    /// hold's own test's reading on either side of a hold.
    pub(crate) fn reading(held: Vec<String>) -> Result<HeldRoom, GpuError> {
        const WHAT: &str = "HeldRoom::read";
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| GpuError::plan(WHAT, format!("read /proc/meminfo: {e}")))?;
        let levels = cgroup_levels().map_err(|e| GpuError::plan(WHAT, e))?;
        let texts: Vec<CgroupLevel<'_>> = levels
            .iter()
            .map(|(path, max, current, stat)| CgroupLevel {
                path,
                max,
                current,
                stat,
            })
            .collect();
        let (bytes, read) = host_room(&meminfo, &texts).map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(HeldRoom {
            bytes,
            read,
            pages: own_pages()?,
            mem_available: mem_available(&meminfo).map_err(|e| GpuError::plan(WHAT, e))?,
            held,
        })
    }

    /// The room `bytes` a caller gave ([`HostRead::Given`]): no reading of
    /// this machine's room, so no card is held for it; the process's pages
    /// and `MemAvailable` are read beside it for the line.
    pub fn given(bytes: u64) -> Result<HeldRoom, GpuError> {
        const WHAT: &str = "HeldRoom::given";
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| GpuError::plan(WHAT, format!("read /proc/meminfo: {e}")))?;
        Ok(HeldRoom {
            bytes,
            read: HostRead::Given,
            pages: own_pages()?,
            mem_available: mem_available(&meminfo).map_err(|e| GpuError::plan(WHAT, e))?,
            held: Vec::new(),
        })
    }

    /// The room's bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The reading that gave [`HeldRoom::bytes`].
    #[must_use]
    pub fn read_by(&self) -> &HostRead {
        &self.read
    }

    /// The room whole and its reading, as a plan takes a room
    /// (`PlanInputs::room`) when nothing is allocated between the plan and
    /// its load's check.
    #[must_use]
    pub fn room(&self) -> (u64, HostRead) {
        (self.bytes, self.read.clone())
    }

    /// The room a plan takes when the load allocates `reserves` before its
    /// check reads the room again ([`LoadReserves::room`]), and the reading.
    #[must_use]
    pub fn plan_room(&self, reserves: &LoadReserves) -> (u64, HostRead) {
        (reserves.room(self.bytes, &self.read), self.read.clone())
    }

    /// The process's own anonymous and shared resident pages at the
    /// reading, in bytes ([`process_pages`]).
    #[must_use]
    pub fn pages(&self) -> u64 {
        self.pages
    }

    /// `MemAvailable` at the reading, in bytes: a line's, never a bound.
    #[must_use]
    pub fn mem_available(&self) -> u64 {
        self.mem_available
    }

    /// The cards held before the reading, by the names the caller gave;
    /// empty for a given room.
    #[must_use]
    pub fn held(&self) -> &[String] {
        &self.held
    }
}

/// This process's own pages now ([`process_pages`] of `/proc/self/status`).
fn own_pages() -> Result<u64, GpuError> {
    const WHAT: &str = "HeldRoom";
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|e| GpuError::plan(WHAT, format!("read /proc/self/status: {e}")))?;
    process_pages(&status).map_err(|e| GpuError::plan(WHAT, e))
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
    let Some(mut path) = ::model::placement::workstation::cgroup_v2_path(&text)? else {
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
