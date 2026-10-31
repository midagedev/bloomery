//! Helper threads' placement: the one owner of where a thread the engine
//! spawns beside its step thread runs.
//!
//! A thread inherits its spawner's affinity mask. A binary that pins its step
//! thread to one cpu ([`crate::Pool::pin_caller`]) would hand every helper it
//! spawns later that one cpu, and the helper's work would take turns with the
//! step on it. [`spawn_helper`] places the new thread before its body runs, as
//! its [`Placement`] says: on a cpu, on the SMT sibling of a cpu (the pool
//! runs one thread a physical core, so each core's other logical cpu idles),
//! or floating. A pin the kernel refuses, or a sibling a core does not have,
//! floats instead, and the list says so; floating off an inherited one-cpu
//! mask widens the mask to every cpu (the kernel keeps it inside the
//! process's cpuset). A mask that cannot be read or widened is a named error
//! and the body never runs. Every placed helper is listed ([`helpers`]) with
//! what it asked for, where it runs and its mask's size, for the binaries'
//! `helper` records.

use std::sync::mpsc;
use std::sync::{Mutex, PoisonError};
use std::thread::JoinHandle;

/// Why a helper was not placed.
#[derive(Debug)]
pub enum HelperError {
    /// The thread could not be spawned.
    Spawn(std::io::Error),
    /// The kernel would not report the new thread's own mask.
    ReadMask,
    /// The kernel refused to widen an inherited one-cpu mask: the helper
    /// would share its spawner's cpu.
    Widen,
    /// The thread ended before it reported its placement.
    Exited,
}

impl HelperError {
    /// The error in words, for callers whose errors carry a static string.
    #[must_use]
    pub fn what(&self) -> &'static str {
        match self {
            HelperError::Spawn(_) => "the helper thread could not be spawned",
            HelperError::ReadMask => "sched_getaffinity refused the helper thread its own mask",
            HelperError::Widen => {
                "sched_setaffinity refused to widen the helper's inherited one-cpu mask: it would \
                 share its spawner's cpu"
            }
            HelperError::Exited => "the helper thread exited before it placed itself",
        }
    }
}

impl std::fmt::Display for HelperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HelperError::Spawn(e) => write!(f, "{}: {e}", self.what()),
            _ => f.write_str(self.what()),
        }
    }
}

impl std::error::Error for HelperError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HelperError::Spawn(e) => Some(e),
            _ => None,
        }
    }
}

/// Where a helper asks to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Pinned to this cpu.
    Pin(usize),
    /// Pinned to the SMT sibling of this cpu.
    Sibling(usize),
    /// Floating.
    Float,
}

impl std::fmt::Display for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Placement::Pin(c) => write!(f, "pin:{c}"),
            Placement::Sibling(c) => write!(f, "sibling:{c}"),
            Placement::Float => f.write_str("float"),
        }
    }
}

/// The threads a helper splits one large read over, itself included, and
/// where that count comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parts {
    pub threads: usize,
    pub from: PartsFrom,
}

/// Where a helper's [`Parts`] count comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartsFrom {
    /// The worker pool's participants on this machine ([`crate::report`]):
    /// the cores the process's cpu mask leaves, or `BLOOMERY_THREADS`.
    Pool,
    /// The caller named the count.
    Asked,
}

impl Parts {
    /// One thread: the helper reads every job alone.
    pub const ALONE: Parts = Parts::asked(1);

    /// `threads` threads, as the caller names them.
    #[must_use]
    pub const fn asked(threads: usize) -> Parts {
        Parts {
            threads,
            from: PartsFrom::Asked,
        }
    }

    /// As many threads as the worker pool has participants on this machine
    /// ([`crate::report`]): a read the step thread waits for runs over the
    /// cpus its own pool would.
    ///
    /// # Panics
    ///
    /// Panics by name when the worker pool's plan cannot be read.
    #[must_use]
    pub fn pool() -> Parts {
        let plan = match crate::report() {
            Ok(plan) => plan,
            Err(error) => panic!("helper parts: cannot read the worker pool plan: {error}"),
        };
        Parts {
            threads: plan.threads,
            from: PartsFrom::Pool,
        }
    }
}

impl std::fmt::Display for PartsFrom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            PartsFrom::Pool => "pool",
            PartsFrom::Asked => "asked",
        })
    }
}

/// A placed helper as [`helpers`] lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Helper {
    pub name: String,
    /// What it asked for.
    pub asked: Placement,
    /// The cpu it is pinned to; `None` when it floats.
    pub pinned: Option<usize>,
    /// Cpus in its mask once placed.
    pub cpus: usize,
    /// The threads it splits one read over ([`spawn_parted`]).
    pub parts: Parts,
}

static HELPERS: Mutex<Vec<Helper>> = Mutex::new(Vec::new());

/// Every helper [`spawn_helper`] placed in this process, in spawn order.
#[must_use]
pub fn helpers() -> Vec<Helper> {
    HELPERS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// Spawn thread `name`, place it as `asked` (a pin the kernel refuses or a
/// sibling the core lacks floats, off any one-cpu mask it inherited), then
/// run `body` on it. Returns once the thread is placed: its handle and the
/// cpu it is pinned to (`None` when it floats). A placement error ends the
/// thread before `body` runs and is returned once it has ended.
pub fn spawn_helper<F>(
    name: &str,
    asked: Placement,
    body: F,
) -> Result<(JoinHandle<()>, Option<usize>), HelperError>
where
    F: FnOnce() + Send + 'static,
{
    spawn_parted(name, asked, Parts::ALONE, body)
}

/// [`spawn_helper`] for a helper whose body splits one read over `parts`
/// threads, listed with the helper.
pub fn spawn_parted<F>(
    name: &str,
    asked: Placement,
    parts: Parts,
    body: F,
) -> Result<(JoinHandle<()>, Option<usize>), HelperError>
where
    F: FnOnce() + Send + 'static,
{
    let (placed_tx, placed_rx) = mpsc::channel::<Result<(Option<usize>, usize), HelperError>>();
    let handle = std::thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || {
            let cpu = match asked {
                Placement::Pin(c) => Some(c),
                Placement::Sibling(c) => sibling_of(c),
                Placement::Float => None,
            };
            let placed = place(cpu).and_then(|p| Ok((p, mask()?.len())));
            let ok = placed.is_ok();
            if placed_tx.send(placed).is_ok() && ok {
                body();
            }
        })
        .map_err(HelperError::Spawn)?;
    let placed = placed_rx.recv().map_err(|_| HelperError::Exited);
    match placed {
        Ok(Ok((pinned, cpus))) => {
            HELPERS
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(Helper {
                    name: name.to_owned(),
                    asked,
                    pinned,
                    cpus,
                    parts,
                });
            Ok((handle, pinned))
        }
        Ok(Err(e)) | Err(e) => {
            let _ = handle.join();
            Err(e)
        }
    }
}

/// Place the calling thread as a [`Placement::Float`] helper, for a thread a
/// caller spawned itself, a scoped one, which [`spawn_helper`]'s `'static`
/// body cannot be: a one-cpu mask inherited from its spawner is widened to
/// every cpu, so the thread does not take turns with the spawner on its cpu.
/// A mask that cannot be read or widened is a named error.
pub fn float_caller() -> Result<(), HelperError> {
    place(None).map(|_| ())
}

/// The calling thread's SMT sibling when it is pinned to one cpu. `None` for
/// a floating caller or a core without SMT.
#[must_use]
pub fn caller_sibling() -> Option<usize> {
    sibling_of(sole_cpu().ok()??)
}

/// The other logical cpu of `cpu`'s core
/// (`/sys/devices/system/cpu/cpu<c>/topology/thread_siblings_list`); `None`
/// for a core without SMT.
#[cfg(target_os = "linux")]
#[must_use]
pub fn sibling_of(cpu: usize) -> Option<usize> {
    let list = std::fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    ))
    .ok()?;
    cpu_list(&list).find(|&c| c != cpu)
}

/// No sysfs SMT sibling is available on macOS.
#[cfg(target_os = "macos")]
#[must_use]
pub fn sibling_of(_cpu: usize) -> Option<usize> {
    None
}

/// The cpus of the calling thread's mask.
pub fn mask() -> Result<Vec<usize>, HelperError> {
    crate::mask_cpus()
        .map(|cpus| cpus.into_iter().map(|c| c as usize).collect())
        .map_err(|_| HelperError::ReadMask)
}

/// The one cpu the calling thread's mask holds; `None` for a mask of more
/// than one cpu.
fn sole_cpu() -> Result<Option<usize>, HelperError> {
    let m = mask()?;
    Ok(match m.as_slice() {
        &[cpu] => Some(cpu),
        _ => None,
    })
}

/// Place the calling thread: pinned to `cpu` when one is asked for and the
/// kernel allows it (`Some`), else floating (`None`), a one-cpu mask widened.
fn place(cpu: Option<usize>) -> Result<Option<usize>, HelperError> {
    if let Some(c) = cpu
        && pin_to(c)
    {
        return Ok(Some(c));
    }
    if sole_cpu()?.is_some() && !float_everywhere() {
        return Err(HelperError::Widen);
    }
    Ok(None)
}

/// Widen the calling thread's affinity to every cpu the set can name; false
/// when the kernel refuses.
fn float_everywhere() -> bool {
    os::float_everywhere().is_ok()
}

/// Pin the calling thread to one logical cpu; false when the kernel refuses
/// (a container, a cpu outside the process's set).
fn pin_to(cpu: usize) -> bool {
    u32::try_from(cpu).is_ok_and(|c| os::pin(c).is_ok())
}

/// The cpus of a sysfs cpu list (`0,32`, `0-1`, `0-3,8`); a malformed entry
/// ends the list.
#[cfg(target_os = "linux")]
fn cpu_list(text: &str) -> impl Iterator<Item = usize> + '_ {
    text.trim()
        .split(',')
        .map_while(|part| match part.split_once('-') {
            Some((a, b)) => Some(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?),
            None => part.parse::<usize>().ok().map(|c| c..=c),
        })
        .flatten()
}
