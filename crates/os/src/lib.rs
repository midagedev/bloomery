//! bloomery-os: the one owner of the operating-system calls the host chain
//! makes, so that what Linux and macOS do differently is written once.
//!
//! Every call here has a Linux body — the call the kernel has — and a macOS
//! body: the call that platform has, or a named refusal
//! ([`OsError::Unsupported`]) when it has none. A platform with neither is a
//! compile error, never a silent fallback: a caller that gets a refusal
//! handles it by name, as it already handles the kernel's.
//!
//! The Linux bodies are the ones the callers had before this crate existed,
//! moved here unchanged; the macOS bodies are the forms the Mac port map
//! names. A type the two platforms shape differently (a `mincore` vector, a
//! `statvfs` field) is settled here, so every caller sees one signature.

use std::ffi::CString;
use std::fs::File;
use std::os::fd::RawFd;
use std::path::Path;

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!("bloomery-os has a body for Linux and macOS only; add this platform's calls here");

/// A refusal by the platform: a call it does not have, or one the kernel
/// refused.
#[derive(Debug)]
pub enum OsError {
    /// The platform has no such call; `call` names it.
    Unsupported { call: &'static str },
    /// The kernel refused the call.
    Io(std::io::Error),
}

impl OsError {
    /// Whether the call is one the platform does not have, or one the kernel
    /// does not know: the two refusals the populate fallback stands for.
    fn is_no_advice(&self) -> bool {
        match self {
            Self::Unsupported { .. } => true,
            Self::Io(e) => e.raw_os_error() == Some(libc::EINVAL),
        }
    }
}

impl std::fmt::Display for OsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported { call } => write!(f, "unsupported on this platform: {call}"),
            Self::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for OsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Unsupported { .. } => None,
        }
    }
}

impl From<std::io::Error> for OsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<OsError> for std::io::Error {
    fn from(e: OsError) -> Self {
        match e {
            OsError::Unsupported { call } => std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!("unsupported on this platform: {call}"),
            ),
            OsError::Io(e) => e,
        }
    }
}

/// `p` as a C string; a path holding a NUL is refused.
fn c_path(p: &Path) -> Result<CString, OsError> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(p.as_os_str().as_bytes())
        .map_err(|e| OsError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))
}

// ------------------------------------------------------------------- affinity

/// The cpus the calling thread may run on, ascending.
///
/// Linux: the kernel's mask for the calling thread, read into a buffer that
/// grows until the mask fits. macOS: every logical cpu of the machine, for
/// the platform has no per-thread mask.
#[cfg(target_os = "linux")]
pub fn mask_cpus() -> Result<Vec<u32>, OsError> {
    let mut words = 16usize;
    loop {
        let mut set = vec![0u64; words];
        let bytes = words * std::mem::size_of::<u64>();
        // SAFETY: `set` is `bytes` writable bytes, and `sched_getaffinity`
        // writes at most that many.
        let rc = unsafe { libc::sched_getaffinity(0, bytes, set.as_mut_ptr().cast()) };
        if rc == 0 {
            return Ok(set
                .iter()
                .enumerate()
                .flat_map(|(w, &bits)| {
                    (0..64u32)
                        .filter(move |b| bits >> b & 1 == 1)
                        .map(move |b| w as u32 * 64 + b)
                })
                .collect());
        }
        let e = std::io::Error::last_os_error();
        // EINVAL: the mask is wider than the buffer.
        if e.raw_os_error() != Some(libc::EINVAL) || words >= 1 << 14 {
            return Err(OsError::Io(e));
        }
        words *= 2;
    }
}

/// The cpus the calling thread may run on, ascending.
///
/// Linux: the kernel's mask for the calling thread, read into a buffer that
/// grows until the mask fits. macOS: every logical cpu of the machine, for
/// the platform has no per-thread mask.
#[cfg(target_os = "macos")]
pub fn mask_cpus() -> Result<Vec<u32>, OsError> {
    let n = logical_cpus()?;
    Ok((0..n).collect())
}

/// The machine's logical cpu count (`hw.logicalcpu`).
#[cfg(target_os = "macos")]
fn logical_cpus() -> Result<u32, OsError> {
    sysctl_u32("hw.logicalcpu")
}

/// Set the calling thread's mask to `cpus`.
///
/// Linux: the kernel's mask for the calling thread. macOS: the platform has
/// no per-thread mask, so the call is refused by name.
#[cfg(target_os = "linux")]
pub fn set_mask(cpus: &[u32]) -> Result<(), OsError> {
    let named = 8 * std::mem::size_of::<libc::cpu_set_t>();
    for &c in cpus {
        if c as usize >= named {
            return Err(OsError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("cpu {c} is past the {named} a cpu_set_t names"),
            )));
        }
    }
    // SAFETY: `set` is a zeroed cpu_set_t only written through CPU_SET with
    // indices below its own bit count, and sched_setaffinity reads it with
    // the matching size.
    let rc = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &c in cpus {
            libc::CPU_SET(c as usize, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set)
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(OsError::Io(std::io::Error::last_os_error()))
    }
}

/// Set the calling thread's mask to `cpus`.
///
/// Linux: the kernel's mask for the calling thread. macOS: the platform has
/// no per-thread mask, so the call is refused by name.
#[cfg(target_os = "macos")]
pub fn set_mask(_cpus: &[u32]) -> Result<(), OsError> {
    Err(OsError::Unsupported {
        call: "sched_setaffinity",
    })
}

/// Pin the calling thread to one logical cpu.
///
/// Linux: the kernel's mask, set to that one cpu. macOS: the platform has no
/// per-thread mask, so the call is refused by name and the pool floats.
#[cfg(target_os = "linux")]
pub fn pin(cpu: u32) -> Result<(), OsError> {
    set_mask(&[cpu])
}

/// Pin the calling thread to one logical cpu.
///
/// Linux: the kernel's mask, set to that one cpu. macOS: the platform has no
/// per-thread mask, so the call is refused by name and the pool floats.
#[cfg(target_os = "macos")]
pub fn pin(_cpu: u32) -> Result<(), OsError> {
    Err(OsError::Unsupported {
        call: "sched_setaffinity",
    })
}

/// Widen the calling thread's mask to every cpu the set can name.
///
/// Linux: the kernel's mask, set to every cpu it names. macOS: the platform
/// has no per-thread mask to widen — a thread there runs on every cpu — so
/// the call has nothing to do and says so.
#[cfg(target_os = "linux")]
pub fn float_everywhere() -> Result<(), OsError> {
    let max = u32::try_from(8 * std::mem::size_of::<libc::cpu_set_t>())
        .expect("a cpu_set_t names fewer cpus than a u32 holds");
    let cpus: Vec<u32> = (0..max).collect();
    set_mask(&cpus)
}

/// Widen the calling thread's mask to every cpu the set can name.
///
/// Linux: the kernel's mask, set to every cpu it names. macOS: the platform
/// has no per-thread mask to widen — a thread there runs on every cpu — so
/// the call has nothing to do and says so.
#[cfg(target_os = "macos")]
pub fn float_everywhere() -> Result<(), OsError> {
    Ok(())
}

// -------------------------------------------------------------------- rusage

/// The page faults the calling thread has taken, minor and major.
///
/// Linux: the kernel's count for the calling thread. macOS: the platform
/// counts a process, not a thread, so the call is refused by name.
#[cfg(target_os = "linux")]
pub fn thread_faults() -> Result<u64, OsError> {
    // SAFETY: `rusage` is a C struct of integers; all-zero bytes are a value of it.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a live, writable `rusage` for the duration of the call.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) };
    assert_eq!(rc, 0, "getrusage(RUSAGE_THREAD)");
    Ok(u64::try_from(usage.ru_minflt + usage.ru_majflt).expect("a fault count is not negative"))
}

/// The page faults the calling thread has taken, minor and major.
///
/// Linux: the kernel's count for the calling thread. macOS: the platform
/// counts a process, not a thread, so the call is refused by name.
#[cfg(target_os = "macos")]
pub fn thread_faults() -> Result<u64, OsError> {
    Err(OsError::Unsupported {
        call: "getrusage(RUSAGE_THREAD)",
    })
}

// ---------------------------------------------------------------------- file

/// The open flag a direct read carries, where the platform has one: Linux's
/// `O_DIRECT`. macOS turns the cache off with a descriptor command
/// (`F_NOCACHE`), not an open flag, so it has none.
#[cfg(target_os = "linux")]
pub const DIRECT_FLAG: Option<u32> = Some(libc::O_DIRECT as u32);

/// The open flag a direct read carries, where the platform has one: Linux's
/// `O_DIRECT`. macOS turns the cache off with a descriptor command
/// (`F_NOCACHE`), not an open flag, so it has none.
#[cfg(target_os = "macos")]
pub const DIRECT_FLAG: Option<u32> = None;

/// `path` opened for reads that bypass the page cache.
///
/// Linux: the `O_DIRECT` flag. macOS: the flag has no counterpart, so the
/// file is opened and its cache turned off with `F_NOCACHE`.
#[cfg(target_os = "linux")]
pub fn open_direct(path: &Path) -> Result<File, OsError> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .map_err(OsError::Io)
}

/// `path` opened for reads that bypass the page cache.
///
/// Linux: the `O_DIRECT` flag. macOS: the flag has no counterpart, so the
/// file is opened and its cache turned off with `F_NOCACHE`.
#[cfg(target_os = "macos")]
pub fn open_direct(path: &Path) -> Result<File, OsError> {
    use std::os::fd::AsRawFd;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .open(path)
        .map_err(OsError::Io)?;
    // SAFETY: the descriptor is this file's own and lives across the call.
    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if rc != 0 {
        return Err(OsError::Io(std::io::Error::last_os_error()));
    }
    Ok(file)
}

/// Rename `from` to `to`, refused when `to` exists.
///
/// Linux: `renameat2` with `RENAME_NOREPLACE`. macOS: `renamex_np` with
/// `RENAME_EXCL`.
#[cfg(target_os = "linux")]
pub fn rename_noreplace(from: &Path, to: &Path) -> Result<(), OsError> {
    let (a, b) = (c_path(from)?, c_path(to)?);
    // SAFETY: both are NUL-terminated paths that live across the call.
    let rc = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            a.as_ptr(),
            libc::AT_FDCWD,
            b.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(OsError::Io(std::io::Error::last_os_error()))
    }
}

/// Rename `from` to `to`, refused when `to` exists.
///
/// Linux: `renameat2` with `RENAME_NOREPLACE`. macOS: `renamex_np` with
/// `RENAME_EXCL`.
#[cfg(target_os = "macos")]
pub fn rename_noreplace(from: &Path, to: &Path) -> Result<(), OsError> {
    let (a, b) = (c_path(from)?, c_path(to)?);
    // SAFETY: both are NUL-terminated paths that live across the call.
    let rc = unsafe { libc::renamex_np(a.as_ptr(), b.as_ptr(), libc::RENAME_EXCL) };
    if rc == 0 {
        Ok(())
    } else {
        Err(OsError::Io(std::io::Error::last_os_error()))
    }
}

/// Bytes free to this process on the filesystem that holds `dir`.
///
/// Linux: the `statvfs` fields as they are. macOS: the same fields, whose
/// widths that platform sets differently, widened to the same numbers.
#[cfg(target_os = "linux")]
pub fn free_bytes(dir: &Path) -> Result<u64, OsError> {
    let c = c_path(dir)?;
    let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c` is a NUL-terminated path that lives across the call, and
    // `st` is storage for one `statvfs`, which the call fills on success.
    if unsafe { libc::statvfs(c.as_ptr(), st.as_mut_ptr()) } != 0 {
        return Err(OsError::Io(std::io::Error::last_os_error()));
    }
    // SAFETY: the call returned 0, so it wrote every field of `st`.
    let st = unsafe { st.assume_init() };
    Ok(st.f_bavail.saturating_mul(st.f_frsize))
}

/// Bytes free to this process on the filesystem that holds `dir`.
///
/// Linux: the `statvfs` fields as they are. macOS: the same fields, whose
/// widths that platform sets differently, widened to the same numbers.
#[cfg(target_os = "macos")]
pub fn free_bytes(dir: &Path) -> Result<u64, OsError> {
    let c = c_path(dir)?;
    let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: as in the Linux body: `c` is a NUL-terminated path that lives
    // across the call, and `st` is storage for one `statvfs`.
    if unsafe { libc::statvfs(c.as_ptr(), st.as_mut_ptr()) } != 0 {
        return Err(OsError::Io(std::io::Error::last_os_error()));
    }
    // SAFETY: the call returned 0, so it wrote every field of `st`.
    let st = unsafe { st.assume_init() };
    // `f_bavail` is a 32-bit count on macOS and `f_frsize` a 64-bit one: the
    // same numbers Linux's fields hold, widened once.
    Ok(u64::from(st.f_bavail).saturating_mul(st.f_frsize))
}

/// Drop the clean pages of `fd` over `at .. at + len` from the page cache.
///
/// Linux: `posix_fadvise(DONTNEED)`. macOS: the platform has no such call,
/// so it is refused by name.
#[cfg(target_os = "linux")]
pub fn fadvise_dontneed(fd: RawFd, at: libc::off_t, len: libc::off_t) -> Result<(), OsError> {
    // SAFETY: the descriptor is the caller's own and lives across the call;
    // fadvise reads and writes no memory of ours.
    let rc = unsafe { libc::posix_fadvise(fd, at, len, libc::POSIX_FADV_DONTNEED) };
    if rc == 0 {
        Ok(())
    } else {
        Err(OsError::Io(std::io::Error::from_raw_os_error(rc)))
    }
}

/// Drop the clean pages of `fd` over `at .. at + len` from the page cache.
///
/// Linux: `posix_fadvise(DONTNEED)`. macOS: the platform has no such call,
/// so it is refused by name.
#[cfg(target_os = "macos")]
pub fn fadvise_dontneed(_fd: RawFd, _at: libc::off_t, _len: libc::off_t) -> Result<(), OsError> {
    Err(OsError::Unsupported {
        call: "posix_fadvise",
    })
}

/// Write the bytes of `fd` over `at .. at + len` out to the drive.
///
/// Linux: `sync_file_range` over the range. macOS: the platform has no
/// ranged write-out, so the whole file is synced with `F_FULLFSYNC`.
#[cfg(target_os = "linux")]
pub fn sync_range(fd: RawFd, at: libc::off_t, len: libc::off_t) -> Result<(), OsError> {
    let flags = libc::SYNC_FILE_RANGE_WAIT_BEFORE
        | libc::SYNC_FILE_RANGE_WRITE
        | libc::SYNC_FILE_RANGE_WAIT_AFTER;
    // SAFETY: the descriptor is the caller's own and lives across the call; the
    // call reads no memory of ours.
    if unsafe { libc::sync_file_range(fd, at, len, flags) } != 0 {
        return Err(OsError::Io(std::io::Error::last_os_error()));
    }
    Ok(())
}

/// Write the bytes of `fd` over `at .. at + len` out to the drive.
///
/// Linux: `sync_file_range` over the range. macOS: the platform has no
/// ranged write-out, so the whole file is synced with `F_FULLFSYNC`.
#[cfg(target_os = "macos")]
pub fn sync_range(fd: RawFd, _at: libc::off_t, _len: libc::off_t) -> Result<(), OsError> {
    // SAFETY: the descriptor is the caller's own and lives across the call.
    let rc = unsafe { libc::fcntl(fd, libc::F_FULLFSYNC, 0) };
    if rc != 0 {
        return Err(OsError::Io(std::io::Error::last_os_error()));
    }
    Ok(())
}

// --------------------------------------------------------------------- host

/// The smallest base page a row room can count in, in bytes.
const MIN_PAGE: u64 = 4096;

/// The host's base page size in bytes: the unit every page-aligned mapping
/// and every row-room count holds.
///
/// Linux and macOS: `sysconf(_SC_PAGESIZE)`. A host whose base page is not
/// a power of two, or is under [`MIN_PAGE`] bytes, is refused by name: the
/// row room counts pages, and a page it cannot count in is a page it cannot
/// size a room in.
pub fn page_bytes() -> Result<u64, OsError> {
    // SAFETY: `sysconf` reads no memory of ours and writes none.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return Err(OsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "sysconf(_SC_PAGESIZE) is not a page size",
        )));
    }
    counted_page(page as u64)
}

/// A base page size the row room can count in: a power of two of at least
/// [`MIN_PAGE`] bytes.
fn counted_page(page: u64) -> Result<u64, OsError> {
    if page < MIN_PAGE || !page.is_power_of_two() {
        return Err(OsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "the base page is {page} B: a power of two of at least {MIN_PAGE} B is what the \
                 row room counts in"
            ),
        )));
    }
    Ok(page)
}

/// The host's memory: what it has, and what this process may use of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemory {
    /// The machine's total memory in bytes.
    pub total_bytes: u64,
    /// The memory available to this process, in bytes.
    pub available_bytes: u64,
}

/// The host's total and available memory.
///
/// Linux: `/proc/meminfo`'s `MemTotal` and `MemAvailable`. macOS: the
/// machine's `hw.memsize` and its host statistics' free and inactive
/// pages. A field the platform does not report is refused by name, never
/// read as 0.
#[cfg(target_os = "linux")]
pub fn host_memory() -> Result<HostMemory, OsError> {
    let text = std::fs::read_to_string("/proc/meminfo").map_err(|e| {
        OsError::Io(std::io::Error::new(
            e.kind(),
            format!("read /proc/meminfo: {e}"),
        ))
    })?;
    let (total, available) = meminfo_bytes(&text)?;
    Ok(HostMemory {
        total_bytes: total,
        available_bytes: available,
    })
}

/// The host's total and available memory.
///
/// Linux: `/proc/meminfo`'s `MemTotal` and `MemAvailable`. macOS: the
/// machine's `hw.memsize` and its host statistics' free and inactive
/// pages. A field the platform does not report is refused by name, never
/// read as 0.
#[cfg(target_os = "macos")]
pub fn host_memory() -> Result<HostMemory, OsError> {
    let total_bytes = sysctl_u64("hw.memsize")?;
    let page = page_bytes()?;
    let mut statistics = std::mem::MaybeUninit::<libc::vm_statistics64>::uninit();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // The libc crate marks `mach_host_self` deprecated in favour of the
    // `mach2` crate, which this workspace does not carry; the host's
    // statistics come through it.
    #[allow(deprecated)]
    // SAFETY: `mach_host_self` takes no arguments and touches no memory of
    // ours; it returns the host port this task may query.
    let host = unsafe { libc::mach_host_self() };
    // SAFETY: `statistics` is storage for one `vm_statistics64` and `count`
    // the number of fields the call writes; the call fills them on success.
    let rc = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&raw mut statistics).cast(),
            &raw mut count,
        )
    };
    if rc != libc::KERN_SUCCESS {
        return Err(OsError::Io(std::io::Error::other(format!(
            "host_statistics64(HOST_VM_INFO64) returned kern_return {rc}"
        ))));
    }
    // SAFETY: the call returned `KERN_SUCCESS`, so it wrote the statistics.
    let statistics = unsafe { statistics.assume_init() };
    // Purgeable pages sit on the active and inactive queues too, so they are
    // not added again.
    let pages = statistics
        .free_count
        .saturating_add(statistics.inactive_count);
    Ok(HostMemory {
        total_bytes,
        available_bytes: u64::from(pages).saturating_mul(page),
    })
}

/// The `MemTotal` and `MemAvailable` of a `/proc/meminfo` text, in bytes.
#[cfg(target_os = "linux")]
fn meminfo_bytes(text: &str) -> Result<(u64, u64), OsError> {
    let kib = |name: &str| {
        let prefix = format!("{name}:");
        let value = text
            .lines()
            .find_map(|l| l.strip_prefix(&prefix))
            .ok_or_else(|| {
                OsError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("the meminfo text has no {name} line"),
                ))
            })?;
        value
            .trim()
            .strip_suffix("kB")
            .map(str::trim_end)
            .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|d| d.parse::<u64>().ok())
            .and_then(|kb| kb.checked_mul(1024))
            .ok_or_else(|| {
                OsError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("the meminfo text's {name} is {value:?}, not a whole number of kB"),
                ))
            })
    };
    Ok((kib("MemTotal")?, kib("MemAvailable")?))
}

/// One performance level of the machine: its cores, its threads and the
/// cache its cores share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PerfLevel {
    /// The level's physical cores.
    pub physical_cpus: u32,
    /// The level's logical cpus.
    pub logical_cpus: u32,
    /// The cache the level's cores share, in bytes.
    pub l2_bytes: u64,
}

/// The machine's performance levels, the fastest first.
///
/// macOS: the `hw.perflevel*` sysctls, one entry a level. Linux: the
/// platform has no such levels — its cpu topology is read from sysfs — so
/// the call is refused by name.
#[cfg(target_os = "macos")]
pub fn perf_levels() -> Result<Vec<PerfLevel>, OsError> {
    let count = sysctl_u32("hw.nperflevels")?;
    let mut levels = Vec::with_capacity(count as usize);
    for level in 0..count {
        levels.push(PerfLevel {
            physical_cpus: sysctl_u32(&format!("hw.perflevel{level}.physicalcpu"))?,
            logical_cpus: sysctl_u32(&format!("hw.perflevel{level}.logicalcpu"))?,
            l2_bytes: sysctl_u64(&format!("hw.perflevel{level}.l2cachesize"))?,
        });
    }
    Ok(levels)
}

/// The machine's performance levels, the fastest first.
///
/// macOS: the `hw.perflevel*` sysctls, one entry a level. Linux: the
/// platform has no such levels — its cpu topology is read from sysfs — so
/// the call is refused by name.
#[cfg(target_os = "linux")]
pub fn perf_levels() -> Result<Vec<PerfLevel>, OsError> {
    Err(OsError::Unsupported {
        call: "sysctl(hw.perflevel*)",
    })
}

/// This process's peak resident set size, in bytes.
///
/// Linux: `getrusage`'s `ru_maxrss`, in kibibytes, in bytes. macOS: the
/// same field, in bytes as that platform reports it.
pub fn max_rss_bytes() -> Result<u64, OsError> {
    max_rss_of(libc::RUSAGE_SELF)
}

/// The peak resident set size of the largest child this process has waited
/// for, in bytes, read as [`max_rss_bytes`] reads its own.
pub fn children_max_rss_bytes() -> Result<u64, OsError> {
    max_rss_of(libc::RUSAGE_CHILDREN)
}

fn max_rss_of(who: libc::c_int) -> Result<u64, OsError> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `usage` is storage for one `rusage`, which the call fills.
    if unsafe { libc::getrusage(who, usage.as_mut_ptr()) } != 0 {
        return Err(OsError::Io(std::io::Error::last_os_error()));
    }
    // SAFETY: the call returned 0, so it wrote every field of `usage`.
    let usage = unsafe { usage.assume_init() };
    Ok(rss_bytes(usage.ru_maxrss))
}

/// The peak resident set `maxrss` holds, in bytes: Linux counts it in
/// kibibytes, macOS in bytes.
#[cfg(target_os = "linux")]
fn rss_bytes(maxrss: libc::c_long) -> u64 {
    u64::try_from(maxrss)
        .expect("getrusage peak RSS must be nonnegative")
        .checked_mul(1024)
        .expect("getrusage peak RSS in bytes exceeds u64")
}

/// The peak resident set `maxrss` holds, in bytes: Linux counts it in
/// kibibytes, macOS in bytes.
#[cfg(target_os = "macos")]
fn rss_bytes(maxrss: libc::c_long) -> u64 {
    u64::try_from(maxrss).expect("getrusage peak RSS must be nonnegative")
}

/// The `u32` the named sysctl holds.
#[cfg(target_os = "macos")]
fn sysctl_u32(name: &str) -> Result<u32, OsError> {
    let invalid = || {
        OsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{name} is not a count"),
        ))
    };
    let value = i32::try_from(sysctl_u64(name)?).map_err(|_| invalid())?;
    u32::try_from(value).map_err(|_| invalid())
}

/// The `u64` the named sysctl holds.
#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Result<u64, OsError> {
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let c = std::ffi::CString::new(name)
        .map_err(|e| OsError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)))?;
    // SAFETY: `value` is a live `u64` and `len` its size; the call writes at
    // most that many bytes into it.
    let rc = unsafe {
        libc::sysctlbyname(
            c.as_ptr(),
            (&raw mut value).cast(),
            (&raw mut len),
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return Err(sysctl_refused(name));
    }
    // A 4-byte value fills the low half of the zeroed `u64`, which is the
    // same number on this little-endian platform.
    if len != std::mem::size_of::<u64>() && len != std::mem::size_of::<u32>() {
        return Err(sysctl_width(name, len));
    }
    Ok(value)
}

/// The named sysctl's refusal, with the call's errno.
#[cfg(target_os = "macos")]
fn sysctl_refused(name: &str) -> OsError {
    let e = std::io::Error::last_os_error();
    OsError::Io(std::io::Error::new(
        e.kind(),
        format!("sysctlbyname({name}): {e}"),
    ))
}

/// A sysctl whose value has a width the reader does not take.
#[cfg(target_os = "macos")]
fn sysctl_width(name: &str, len: usize) -> OsError {
    OsError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("sysctlbyname({name}) wrote {len} B, not an integer this reader takes"),
    ))
}

// ------------------------------------------------------------------- memory

/// How a span's pages were faulted in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Populated {
    /// The platform's populate advice.
    Advised,
    /// A read of one byte a page.
    Touched,
}

/// The platform's populate advice: fault the pages of `addr .. addr + len` in.
///
/// Linux: `madvise(MADV_POPULATE_READ)`. macOS: the platform has no such
/// advice, so the call is refused by name and the pages are read in instead.
///
/// # Safety
/// `addr .. addr + len` must be whole pages inside one live mapping.
#[cfg(target_os = "linux")]
pub unsafe fn advise_populate(addr: *mut libc::c_void, len: usize) -> Result<(), OsError> {
    // SAFETY: the caller's contract above; MADV_POPULATE_READ faults the
    // pages in and never writes them.
    let rc = unsafe { libc::madvise(addr, len, libc::MADV_POPULATE_READ) };
    if rc == 0 {
        Ok(())
    } else {
        Err(OsError::Io(std::io::Error::last_os_error()))
    }
}

/// The platform's populate advice: fault the pages of `addr .. addr + len` in.
///
/// Linux: `madvise(MADV_POPULATE_READ)`. macOS: the platform has no such
/// advice, so the call is refused by name and the pages are read in instead.
///
/// # Safety
/// `addr .. addr + len` must be whole pages inside one live mapping.
#[cfg(target_os = "macos")]
pub unsafe fn advise_populate(_addr: *mut libc::c_void, _len: usize) -> Result<(), OsError> {
    Err(OsError::Unsupported {
        call: "madvise(MADV_POPULATE_READ)",
    })
}

/// The platform's huge-page advice for `addr .. addr + len`: back it with
/// huge pages where the kernel can, so that its first writes fault once a
/// huge page instead of once a base page.
///
/// Linux: `madvise(MADV_HUGEPAGE)`. macOS: the platform has no such advice,
/// so the call is refused by name.
///
/// # Safety
/// `addr .. addr + len` must be whole pages of private anonymous memory the
/// caller owns.
#[cfg(target_os = "linux")]
pub unsafe fn advise_huge(addr: *mut libc::c_void, len: usize) -> Result<(), OsError> {
    // SAFETY: the caller's contract above; MADV_HUGEPAGE changes how the
    // pages are backed, never what they hold.
    let rc = unsafe { libc::madvise(addr, len, libc::MADV_HUGEPAGE) };
    if rc == 0 {
        Ok(())
    } else {
        Err(OsError::Io(std::io::Error::last_os_error()))
    }
}

/// The platform's huge-page advice for `addr .. addr + len`: back it with
/// huge pages where the kernel can, so that its first writes fault once a
/// huge page instead of once a base page.
///
/// Linux: `madvise(MADV_HUGEPAGE)`. macOS: the platform has no such advice,
/// so the call is refused by name.
///
/// # Safety
/// `addr .. addr + len` must be whole pages of private anonymous memory the
/// caller owns.
#[cfg(target_os = "macos")]
pub unsafe fn advise_huge(_addr: *mut libc::c_void, _len: usize) -> Result<(), OsError> {
    Err(OsError::Unsupported {
        call: "madvise(MADV_HUGEPAGE)",
    })
}

/// The huge page [`advise_huge_within`] asks for.
const HUGE_PAGE: usize = 2 << 20;

/// The size from which a new allocation is memory nothing has written:
/// below it glibc's allocator mostly hands back heap memory it has written
/// before, which the advice does not change.
pub const ADVISE_FROM: usize = 32 << 20;

/// Huge pages for the whole huge pages inside `span`, a new heap allocation
/// of at least [`ADVISE_FROM`] bytes nothing has written yet (a vector's
/// spare capacity, or a new zeroed vector), so that its first writes fault
/// once each huge page instead of once each base page, and its free unmaps
/// huge pages. A refusal (a kernel without transparent huge pages, or a
/// platform without the advice) leaves base pages, which the writes fill
/// the same.
pub fn advise_huge_within<T>(span: &mut [T]) {
    let bytes = size_of_val(span);
    if bytes < ADVISE_FROM {
        return;
    }
    let at = span.as_mut_ptr().addr();
    let (first, end) = (
        at.next_multiple_of(HUGE_PAGE),
        (at + bytes) / HUGE_PAGE * HUGE_PAGE,
    );
    if end > first {
        // SAFETY: `first..end` is whole pages inside the allocation `span`
        // borrows, private anonymous memory its owner holds.
        let advised = unsafe { advise_huge(first as *mut libc::c_void, end - first) };
        drop(advised);
    }
}

/// Fault the pages of `addr .. addr + len` in.
///
/// `advise` is the platform's populate advice call. A refusal that names the
/// advice as unknown (EINVAL) or absent ([`OsError::Unsupported`]) gets the
/// same pages faulted by a read of one byte each `page` bytes instead; any
/// other refusal is the caller's, by name.
///
/// # Safety
/// `addr .. addr + len` must be inside one live mapping: the pages are read
/// one byte a page when the advice is not there.
pub unsafe fn populate_span(
    addr: *mut libc::c_void,
    len: usize,
    page: u64,
    advise: impl Fn() -> Result<(), OsError>,
) -> Result<Populated, OsError> {
    match advise() {
        Ok(()) => Ok(Populated::Advised),
        Err(e) if e.is_no_advice() => {
            // SAFETY: the caller's contract above.
            unsafe { touch(addr, len, page)? };
            Ok(Populated::Touched)
        }
        Err(e) => Err(e),
    }
}

/// The pages of `addr .. addr + len` faulted in by a read of one byte a page.
///
/// # Safety
/// `addr .. addr + len` must be inside one live mapping.
unsafe fn touch(addr: *mut libc::c_void, len: usize, page: u64) -> Result<(), OsError> {
    let step = usize::try_from(page).map_err(|_| {
        OsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("a page of {page} bytes"),
        ))
    })?;
    for at in (0..len).step_by(step) {
        // SAFETY: `at` is below `len`, so the byte is inside the caller's live
        // range; the read is volatile so that it is made.
        let _ = unsafe { std::ptr::read_volatile(addr.cast::<u8>().add(at)) };
    }
    Ok(())
}

/// The kernel's `mincore` vector for `vec`: a byte a page.
#[cfg(target_os = "linux")]
fn mincore_vec(vec: &mut [u8]) -> *mut libc::c_uchar {
    vec.as_mut_ptr()
}

/// The kernel's `mincore` vector for `vec`: a `c_char` a page.
#[cfg(target_os = "macos")]
fn mincore_vec(vec: &mut [u8]) -> *mut libc::c_char {
    vec.as_mut_ptr().cast()
}

/// The address `mincore` takes: a mutable pointer on Linux, a constant one
/// on macOS.
#[cfg(target_os = "linux")]
fn mincore_addr(addr: *mut libc::c_void) -> *mut libc::c_void {
    addr
}

/// The address `mincore` takes: a mutable pointer on Linux, a constant one
/// on macOS.
#[cfg(target_os = "macos")]
fn mincore_addr(addr: *mut libc::c_void) -> *const libc::c_void {
    addr.cast_const()
}

/// The resident pages of `addr .. addr + len`, one byte a page in `vec`.
///
/// Linux: the kernel's `mincore`, whose vector is a byte a page. macOS: the
/// same call, whose vector is a `c_char` a page; the shapes are settled here.
///
/// # Safety
/// `addr .. addr + len` must be inside one live mapping, and `vec` must hold
/// one byte for each of the pages of that range.
pub unsafe fn mincore(addr: *mut libc::c_void, len: usize, vec: &mut [u8]) -> Result<(), OsError> {
    // SAFETY: the caller's contract above; mincore only reads the page tables
    // and writes nothing through `addr`.
    let rc = unsafe { libc::mincore(mincore_addr(addr), len, mincore_vec(vec)) };
    if rc != 0 {
        return Err(OsError::Io(std::io::Error::last_os_error()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{OsError, Populated};
    use std::path::PathBuf;

    /// A scratch file of this process's own making, removed when the test
    /// ends.
    fn scratch(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("bloomery-os-{name}-{}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    /// `len` bytes on a page boundary, zeroed: the mapping the populate
    /// advice and the resident-page count take.
    struct Page {
        ptr: *mut u8,
        len: usize,
        layout: std::alloc::Layout,
    }

    impl Page {
        fn new(len: usize) -> Page {
            let layout =
                std::alloc::Layout::from_size_align(len, 4096).expect("a page-aligned layout");
            // SAFETY: `layout` has a non-zero size; the memory is zeroed below
            // before anything reads it.
            let ptr = unsafe { std::alloc::alloc(layout) };
            assert!(!ptr.is_null(), "the page-aligned memory is there");
            // SAFETY: `ptr` is `len` writable bytes of the allocation above.
            unsafe { std::ptr::write_bytes(ptr, 0, len) };
            Page { ptr, len, layout }
        }

        fn addr(&self) -> *mut std::ffi::c_void {
            self.ptr.cast()
        }
    }

    impl Drop for Page {
        fn drop(&mut self) {
            // SAFETY: `ptr` came from `alloc` with this layout, and nothing
            // uses it after the drop.
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }

    /// The cpus the calling thread may run on: at least one, ascending.
    #[test]
    fn the_mask_names_the_cpus_the_thread_may_run_on() {
        let cpus = super::mask_cpus().expect("the mask reads");
        assert!(!cpus.is_empty(), "the mask names a cpu");
        assert!(
            cpus.windows(2).all(|w| w[0] < w[1]),
            "the mask is ascending: {cpus:?}"
        );
    }

    /// The mask the caller sets is the mask the thread then has; the platform
    /// with no per-thread mask refuses the call by name.
    #[test]
    fn the_mask_is_the_one_the_caller_sets() {
        #[cfg(target_os = "linux")]
        {
            let cpus = super::mask_cpus().expect("the mask reads");
            let one = &cpus[..1];
            super::set_mask(one).expect("the mask is set");
            assert_eq!(super::mask_cpus().expect("the mask reads back"), one);
        }
        #[cfg(target_os = "macos")]
        {
            let e = super::set_mask(&[0]).expect_err("the platform has no mask");
            assert!(
                matches!(e, OsError::Unsupported { call } if call == "sched_setaffinity"),
                "{e}"
            );
        }
    }

    /// A pin of a cpu the mask does not name is refused.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_pin_past_the_mask_is_refused() {
        let e = super::pin(u32::MAX).expect_err("the cpu is past the mask");
        assert!(
            matches!(&e, OsError::Io(e) if e.kind() == std::io::ErrorKind::InvalidInput),
            "{e}"
        );
    }

    /// A pin of a cpu the mask names holds the thread on it; the platform with
    /// no per-thread mask refuses the pin by name.
    #[test]
    fn a_pin_holds_the_thread_on_its_cpu() {
        #[cfg(target_os = "linux")]
        {
            let cpus = super::mask_cpus().expect("the mask reads");
            let cpu = cpus[0];
            super::pin(cpu).expect("the pin holds");
            assert_eq!(super::mask_cpus().expect("the mask reads back"), vec![cpu]);
        }
        #[cfg(target_os = "macos")]
        {
            let e = super::pin(0).expect_err("the platform has no mask");
            assert!(
                matches!(e, OsError::Unsupported { call } if call == "sched_setaffinity"),
                "{e}"
            );
        }
    }

    /// Floating widens a narrow mask back to the cpus the process may run on;
    /// the platform with no mask has nothing to widen and says so.
    #[test]
    fn floating_widens_the_mask() {
        #[cfg(target_os = "linux")]
        {
            let cpus = super::mask_cpus().expect("the mask reads");
            super::set_mask(&cpus[..1]).expect("the mask is set");
            super::float_everywhere().expect("the mask widens");
            let after = super::mask_cpus().expect("the mask reads back");
            assert!(
                after.len() >= cpus.len(),
                "the mask widened: {cpus:?} then {after:?}"
            );
        }
        #[cfg(target_os = "macos")]
        super::float_everywhere().expect("there is no mask to widen");
    }

    /// The page faults of the calling thread are a count that does not go
    /// back; the platform that counts a process, not a thread, refuses the
    /// per-thread count by name.
    #[test]
    fn the_thread_faults_are_counted() {
        #[cfg(target_os = "linux")]
        {
            let before = super::thread_faults().expect("the fault count reads");
            let after = super::thread_faults().expect("the fault count reads");
            assert!(
                after >= before,
                "the count does not go back: {before} then {after}"
            );
        }
        #[cfg(target_os = "macos")]
        {
            let e = super::thread_faults().expect_err("the platform counts a process");
            assert!(
                matches!(e, OsError::Unsupported { call } if call == "getrusage(RUSAGE_THREAD)"),
                "{e}"
            );
        }
    }

    /// A direct open of a file the process wrote is the handle of that file;
    /// the flag the platform has for it is named.
    #[test]
    fn a_direct_open_is_the_handle_of_the_file() {
        let p = scratch("direct");
        std::fs::write(&p, vec![7u8; 4096]).expect("the file is written");
        let file = super::open_direct(&p).expect("the file opens for direct reads");
        let len = file.metadata().expect("the file has a size").len();
        let _ = std::fs::remove_file(&p);
        assert_eq!(len, 4096, "the handle is the file that was written");
        #[cfg(target_os = "linux")]
        assert!(super::DIRECT_FLAG.is_some(), "the platform has the flag");
        #[cfg(target_os = "macos")]
        assert!(super::DIRECT_FLAG.is_none(), "the platform has no flag");
    }

    /// A rename that replaces nothing moves the file; a rename onto an
    /// existing file is refused.
    #[test]
    fn a_rename_onto_an_existing_file_is_refused() {
        let from = scratch("rename-from");
        let to = scratch("rename-to");
        let there = scratch("rename-there");
        std::fs::write(&from, b"a").expect("the file is written");
        std::fs::write(&there, b"b").expect("the file is written");
        super::rename_noreplace(&from, &to).expect("the rename moves the file");
        assert!(to.exists(), "the file is at its new name");
        let e = super::rename_noreplace(&there, &to).expect_err("the target exists");
        assert!(
            matches!(&e, OsError::Io(e) if e.raw_os_error() == Some(libc::EEXIST)),
            "{e}"
        );
        let _ = std::fs::remove_file(&to);
        let _ = std::fs::remove_file(&there);
    }

    /// The free bytes of a directory are its free blocks times their size, so
    /// the count is a whole number of 512-byte sectors (every filesystem's
    /// fragment size is one); a count of blocks without their size is not.
    #[test]
    fn the_free_bytes_of_a_directory_are_counted() {
        let dir = std::env::temp_dir();
        let free = super::free_bytes(&dir).expect("the filesystem answers");
        assert!(free > 0, "the filesystem has room: {free}");
        assert_eq!(
            free % 512,
            0,
            "free_bytes {free} is not blocks times their size"
        );
    }

    /// The page cache drops the clean pages of a file; the platform with no
    /// `posix_fadvise` refuses the drop by name.
    #[test]
    fn the_page_cache_drops_a_files_clean_pages() {
        let p = scratch("fadvise");
        std::fs::write(&p, b"bloomery").expect("the file is written");
        let file = std::fs::File::open(&p).expect("the file opens");
        let fd = {
            use std::os::fd::AsRawFd;
            file.as_raw_fd()
        };
        let dropped = super::fadvise_dontneed(fd, 0, 8);
        let _ = std::fs::remove_file(&p);
        #[cfg(target_os = "linux")]
        {
            dropped.expect("the pages drop");
            // The call reaches the kernel: a closed descriptor is refused.
            let e = super::fadvise_dontneed(-1, 0, 8).expect_err("no file is open at -1");
            assert!(
                matches!(&e, OsError::Io(io) if io.raw_os_error() == Some(libc::EBADF)),
                "{e}"
            );
        }
        #[cfg(target_os = "macos")]
        {
            let e = dropped.expect_err("the platform has no call");
            assert!(
                matches!(e, OsError::Unsupported { call } if call == "posix_fadvise"),
                "{e}"
            );
        }
    }

    /// A file's bytes are written out.
    #[test]
    fn a_files_bytes_are_written_out() {
        let p = scratch("sync");
        std::fs::write(&p, b"bloomery").expect("the file is written");
        let file = std::fs::File::open(&p).expect("the file opens");
        let fd = {
            use std::os::fd::AsRawFd;
            file.as_raw_fd()
        };
        super::sync_range(fd, 0, 8).expect("the bytes are written out");
        let _ = std::fs::remove_file(&p);
        // The call reaches the kernel: a closed descriptor is refused.
        let e = super::sync_range(-1, 0, 8).expect_err("no file is open at -1");
        assert!(
            matches!(&e, OsError::Io(io) if io.raw_os_error() == Some(libc::EBADF)),
            "{e}"
        );
    }

    /// The pages of a span are faulted in: by the platform's advice where it
    /// has one, by a read of one byte a page where it has none.
    #[test]
    fn a_span_is_faulted_in() {
        let span = Page::new(8192);
        let addr = span.addr();
        // SAFETY: the span is the page-aligned allocation above, so its pages
        // are inside one live mapping.
        let got = unsafe {
            super::populate_span(addr, span.len, 4096, || {
                super::advise_populate(addr, span.len)
            })
        }
        .expect("the pages fault in");
        #[cfg(target_os = "linux")]
        assert_eq!(got, Populated::Advised);
        #[cfg(target_os = "macos")]
        assert_eq!(got, Populated::Touched);
    }

    /// Linux takes the huge-page advice for whole pages of anonymous memory;
    /// macOS, which has no such advice, refuses it by name.
    #[test]
    fn the_huge_page_advice_is_taken_where_the_platform_has_it() {
        let span = Page::new(4 << 20);
        // SAFETY: the span is the page-aligned allocation above, whole pages
        // of this process's anonymous memory.
        let got = unsafe { super::advise_huge(span.addr(), span.len) };
        #[cfg(target_os = "linux")]
        {
            assert!(
                got.is_ok()
                    || matches!(&got, Err(OsError::Io(e)) if e.raw_os_error() == Some(libc::EINVAL)),
                "{got:?}"
            );
            if got.is_ok() {
                let smaps = std::fs::read_to_string("/proc/self/smaps").expect("the mapping flags");
                let at = span.addr().addr();
                let flags = smaps
                    .lines()
                    .skip_while(|line| {
                        let Some((first, last)) = line
                            .split_whitespace()
                            .next()
                            .and_then(|range| range.split_once('-'))
                        else {
                            return true;
                        };
                        let first = usize::from_str_radix(first, 16).expect("the mapping start");
                        let last = usize::from_str_radix(last, 16).expect("the mapping end");
                        !(first <= at && at < last)
                    })
                    .find(|line| line.starts_with("VmFlags:"))
                    .expect("the advised mapping's flags");
                assert!(flags.split_whitespace().any(|flag| flag == "hg"), "{flags}");
            }
        }
        #[cfg(target_os = "macos")]
        assert!(
            matches!(&got, Err(OsError::Unsupported { call }) if *call == "madvise(MADV_HUGEPAGE)"),
            "{got:?}"
        );
    }

    /// The resident pages of a span are counted, a byte a page.
    #[test]
    fn a_span_s_resident_pages_are_counted() {
        let span = Page::new(8192);
        let mut vec = [0u8; 2];
        // SAFETY: the span is the page-aligned allocation above, so its pages
        // are inside one live mapping, and `vec` holds a byte for each.
        unsafe { super::mincore(span.addr(), span.len, &mut vec) }.expect("the pages are counted");
        // Linux answers for every page of the range; macOS answers for the
        // first page of it, so the claim both make is that one.
        #[cfg(target_os = "linux")]
        assert!(
            vec.iter().all(|b| b & 1 == 1),
            "the pages are resident: {vec:?}"
        );
        #[cfg(target_os = "macos")]
        assert_eq!(vec[0] & 1, 1, "the first page is resident: {vec:?}");
    }

    /// The base page is a power of two of at least 4 KiB: the row room counts
    /// pages, and a page it cannot count in is a page it cannot size a room
    /// in.
    #[test]
    fn the_base_page_is_a_power_of_two_of_at_least_four_kibibytes() {
        let page = super::page_bytes().expect("the base page reads");
        assert!(page >= 4096, "the base page is {page} B");
        assert!(page.is_power_of_two(), "the base page is {page} B");
        // x86_64 Linux: the base page is 4 KiB (aarch64 kernels may take 16
        // or 64 KiB).
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        assert_eq!(page, 4096, "the kernel's base page");
        // macOS: the kernel's base page is 16 KiB on Apple silicon.
        #[cfg(target_os = "macos")]
        assert!(page >= 16384, "the base page is {page} B");
        // A page the row room cannot count in is refused by name.
        assert_eq!(super::counted_page(4096).expect("4 KiB counts"), 4096);
        assert_eq!(super::counted_page(16384).expect("16 KiB counts"), 16384);
        for bad in [0u64, 1, 3000, 4095, 5000] {
            let why = super::counted_page(bad).expect_err("the page cannot be counted in");
            assert!(why.to_string().contains(&bad.to_string()), "{why}");
        }
    }

    /// The host's total and available memory, in bytes: the platform's own
    /// reading, with the total the machine has and the available no more
    /// than it.
    #[test]
    fn the_host_memory_names_its_total_and_its_available_bytes() {
        let memory = super::host_memory().expect("the host memory reads");
        assert!(memory.total_bytes > 0, "the host has memory");
        assert!(memory.available_bytes > 0, "the host has memory available");
        // Linux: the reading is `/proc/meminfo`'s own lines.
        #[cfg(target_os = "linux")]
        {
            // `MemAvailable` moves between two reads; `MemTotal` does not.
            assert!(memory.available_bytes <= memory.total_bytes);
            let text = std::fs::read_to_string("/proc/meminfo").expect("meminfo reads");
            let (total, _) = super::meminfo_bytes(&text).expect("the meminfo lines read");
            assert_eq!(memory.total_bytes, total);
        }
        // macOS: the total is the machine's `hw.memsize`.
        #[cfg(target_os = "macos")]
        assert_eq!(
            memory.total_bytes,
            super::sysctl_u64("hw.memsize").expect("hw.memsize reads")
        );
    }

    /// The `MemTotal` and `MemAvailable` of a `/proc/meminfo` text, in
    /// bytes; a text without the line, or with a value that is not a whole
    /// number of kB, is refused.
    #[cfg(target_os = "linux")]
    #[test]
    fn meminfo_reads_its_lines_in_kb() {
        let text = "MemTotal:       263741212 kB\nMemFree:         1234567 kB\n\
                    MemAvailable:   257300180 kB\nBuffers:            1234 kB\n";
        assert_eq!(
            super::meminfo_bytes(text).expect("the lines read"),
            (263_741_212 * 1024, 257_300_180 * 1024)
        );
        assert!(super::meminfo_bytes("MemTotal: 1 kB\n").is_err());
        assert!(super::meminfo_bytes("MemAvailable: 1 kB\n").is_err());
        for value in ["12x kB", "12 MB", "kB"] {
            let text = format!("MemTotal: 1 kB\nMemAvailable: {value}\n");
            assert!(super::meminfo_bytes(&text).is_err(), "{value}");
        }
    }

    /// The machine's performance levels: their cores, their threads and the
    /// cache they share. The platform with no such levels refuses the call
    /// by name.
    #[test]
    fn the_machine_names_its_performance_levels() {
        // Linux: the platform has no performance levels.
        #[cfg(target_os = "linux")]
        {
            let e = super::perf_levels().expect_err("the platform has no performance levels");
            assert!(
                matches!(e, OsError::Unsupported { call } if call == "sysctl(hw.perflevel*)"),
                "{e}"
            );
        }
        // macOS: the machine names its levels, their cores and their cache.
        #[cfg(target_os = "macos")]
        {
            let levels = super::perf_levels().expect("the performance levels read");
            assert!(!levels.is_empty(), "the machine names a level");
            for level in levels {
                assert!(level.physical_cpus > 0, "the level has cores");
                assert!(level.logical_cpus >= level.physical_cpus);
                assert!(level.l2_bytes > 0, "the level has a cache");
            }
        }
    }

    /// This process's peak resident set, in bytes: the platform's own unit
    /// widened to bytes.
    #[test]
    fn the_peak_resident_set_is_in_bytes() {
        let bytes = super::max_rss_bytes().expect("the peak resident set reads");
        assert!(bytes > 0, "the process holds memory");
        // Linux: `ru_maxrss` counts kibibytes, so the bytes are a whole
        // number of them.
        #[cfg(target_os = "linux")]
        assert_eq!(bytes % 1024, 0, "the bytes are whole kibibytes: {bytes}");
        // Both: a test process holds more than a mebibyte and less than the
        // machine, so a count left in kibibytes, or multiplied once more, is
        // out of range.
        let total = super::host_memory()
            .expect("the host memory reads")
            .total_bytes;
        assert!(
            (1 << 20..=total).contains(&bytes),
            "the peak is {bytes} B on a {total} B host"
        );
        #[cfg(target_os = "linux")]
        assert_eq!(super::rss_bytes(2048), 2_097_152);
        #[cfg(target_os = "macos")]
        assert_eq!(super::rss_bytes(2048), 2048);
        let children = super::children_max_rss_bytes().expect("the children's peak reads");
        assert!(children <= total, "the children's peak is {children} B");
    }
}
