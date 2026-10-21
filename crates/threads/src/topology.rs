//! Where the pool's threads can sit: the machine's L3 groups read from sysfs
//! ([`detect_topology`]), cut to the cpus the process may run on
//! ([`Shape::of`], the one owner of the pool's placement rule), and the
//! report of the result ([`PoolPlan`]) a plan prints before any thread is
//! spawned. Nothing here calls the kernel's affinity calls: the mask is an
//! argument, so a test names the mask it cuts by.

use std::fmt;

#[cfg(any(target_os = "linux", test))]
fn parse_cpu_list(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        if let Some((a, b)) = part.split_once('-') {
            for c in a.trim().parse::<u32>().unwrap()..=b.trim().parse::<u32>().unwrap() {
                out.push(c);
            }
        } else {
            out.push(part.trim().parse().unwrap());
        }
    }
    out
}

/// One shared L3 as the topology reads it: its cores, each core's logical
/// cpus (the lowest first), and its size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CcdGroup {
    pub(crate) cores: Vec<Vec<u32>>,
    /// The L3's size in bytes; 0 where sysfs does not say.
    pub(crate) l3_bytes: u64,
}

/// What sysfs said of the machine's cpus.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Topology {
    pub(crate) groups: Vec<CcdGroup>,
    /// Sysfs listed the cpus' siblings. `false`: a machine with no topology
    /// files, whose `groups` are empty; [`Shape::of`] takes the cpus of the
    /// process's mask as the cores of one group.
    pub(crate) read: bool,
}

/// `cache/index3/size` as bytes: `32768K`, `32M`, `1G` or a bare count;
/// `None` for anything else.
#[cfg(any(target_os = "linux", test))]
fn parse_cache_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (digits, unit) = match s.char_indices().last()? {
        (i, 'K' | 'k') => (&s[..i], 1u64 << 10),
        (i, 'M' | 'm') => (&s[..i], 1 << 20),
        (i, 'G' | 'g') => (&s[..i], 1 << 30),
        (_, c) if c.is_ascii_digit() => (s, 1),
        _ => return None,
    };
    digits.parse::<u64>().ok()?.checked_mul(unit)
}

/// The L3 size `read` found at `path`, a `cache/index3/size`: 0 where there is
/// no such file (the topology does not say — a container, a VM). A file that is
/// there and holds no size, or cannot be read, panics by name: a cap taken from
/// it must not be a 0 that looks like an absent L3.
#[cfg(any(target_os = "linux", test))]
fn l3_size(path: &str, read: std::io::Result<String>) -> u64 {
    match read {
        Ok(text) => parse_cache_size(&text).unwrap_or_else(|| {
            panic!("{path} holds {text:?}, not a cache size (K, M, G or a byte count)")
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => 0,
        Err(e) => panic!("{path}: {e}"),
    }
}

/// Detect the L3 (CCD) topology: one group per shared L3 — its cores' cpus
/// and its size — sorted by primary core id. The primary hyperthread of each
/// core is its lowest sibling id. Ported from `crates/q3k-cpu/src/main.rs:324`
/// (`ccd_topology`), restructured so the physical-core count is recoverable
/// and a core's cpus stay together.
#[cfg(target_os = "linux")]
pub(crate) fn detect_topology() -> Topology {
    detect_with(&|path| std::fs::read_to_string(path))
}

/// The machine's cpu topology: the performance level's cores and the cache
/// they share. The platform names no cpu's level, so the performance level
/// takes the machine's lowest logical cpu ids — one core a physical cpu of
/// the level — and the cache it shares stands where the L3 stands. The pool
/// floats on this platform (it has no per-thread mask), so the ids shape
/// the plan's report and the threads' placement among the level's cores.
#[cfg(target_os = "macos")]
pub(crate) fn detect_topology() -> Topology {
    let levels = os::perf_levels().unwrap_or_else(|e| panic!("the performance levels: {e}"));
    let Some(performance) = levels.first() else {
        return Topology {
            groups: Vec::new(),
            read: false,
        };
    };
    let cores = (0..performance.physical_cpus)
        .map(|cpu| vec![cpu])
        .collect::<Vec<Vec<u32>>>();
    Topology {
        groups: vec![CcdGroup {
            cores,
            l3_bytes: performance.l2_bytes,
        }],
        read: true,
    }
}

/// [`detect_topology`] over `read`, the reader of a sysfs file.
#[cfg(any(target_os = "linux", test))]
fn detect_with(read: &dyn Fn(&str) -> std::io::Result<String>) -> Topology {
    let read_sys = |path: &str| read(path).ok();
    let mut groups: Vec<(String, Vec<u32>)> = Vec::new(); // (l3 key, primaries)
    let mut cpu = 0u32;
    while let Some(sib) = read_sys(&format!(
        "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    )) {
        let siblings = parse_cpu_list(&sib);
        let primary = *siblings.iter().min().unwrap();
        if primary == cpu {
            let l3 = read_sys(&format!(
                "/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list"
            ))
            .unwrap_or_default();
            match groups.iter_mut().find(|(k, _)| *k == l3) {
                Some((_, v)) => v.push(cpu),
                None => groups.push((l3, vec![cpu])),
            }
        }
        cpu += 1;
    }
    if groups.is_empty() {
        // No sysfs topology (container?): no groups. The mask's cpus stand in
        // for them ([`Shape::of`]), so the pool still has a default thread
        // count of the cpus there are.
        return Topology {
            groups: Vec::new(),
            read: false,
        };
    }
    groups.sort_by_key(|(_, v)| v[0]);
    let groups = groups
        .into_iter()
        .map(|(_, primaries)| {
            let path = format!(
                "/sys/devices/system/cpu/cpu{}/cache/index3/size",
                primaries[0]
            );
            let l3_bytes = l3_size(&path, read(&path));
            let cores = primaries
                .into_iter()
                .map(|p| {
                    let sib = parse_cpu_list(
                        &read_sys(&format!(
                            "/sys/devices/system/cpu/cpu{p}/topology/thread_siblings_list"
                        ))
                        .unwrap_or_default(),
                    );
                    std::iter::once(p)
                        .chain(sib.into_iter().filter(|&s| s != p))
                        .collect()
                })
                .collect();
            CcdGroup { cores, l3_bytes }
        })
        .collect();
    Topology { groups, read: true }
}

/// What the process's cpu mask took from the topology.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cut {
    /// Cpus the topology lists, and of them the mask allows.
    pub topology_cpus: usize,
    pub kept_cpus: usize,
    /// Physical cores, and of them those with an allowed cpu.
    pub cores: usize,
    pub kept_cores: usize,
    /// L3 groups, and of them those with an allowed core.
    pub ccds: usize,
    pub kept_ccds: usize,
    /// Cpus the mask allows that the topology does not list.
    pub unlisted: usize,
    /// The mask shares no cpu with the topology, so the pool takes the mask's
    /// own cpus as one group of one-cpu cores.
    pub from_mask: bool,
}

impl fmt::Display for Cut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the process's cpu mask cuts the topology: {} of {} cpus allowed ({} of {} cores, \
             {} of {} L3 groups)",
            self.kept_cpus,
            self.topology_cpus,
            self.kept_cores,
            self.cores,
            self.kept_ccds,
            self.ccds
        )?;
        if self.unlisted > 0 {
            write!(
                f,
                "; {} allowed cpus are not in the topology",
                self.unlisted
            )?;
        }
        if self.from_mask {
            write!(
                f,
                "; none of the topology's cpus is allowed, so the pool takes the mask's own \
                 cpus as one group"
            )?;
        }
        Ok(())
    }
}

/// The pool's placement rule applied: which cpus its threads can take, and
/// how many threads it runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Shape {
    /// One Vec per CCD (shared L3), logical cpu ids, primaries before
    /// siblings, every one allowed by the mask.
    pub(crate) topo: Vec<Vec<u32>>,
    /// Each CCD's L3 in bytes, parallel to `topo`; 0 where sysfs does not say.
    pub(crate) l3: Vec<u64>,
    /// Cores with an allowed cpu: the default thread count.
    pub(crate) physical: usize,
    /// The pool's participants, the calling thread among them.
    pub(crate) threads: usize,
    /// The cpus the mask allows, ascending.
    pub(crate) allowed: Vec<u32>,
    /// `Some` when the mask took cpus from the topology.
    pub(crate) cut: Option<Cut>,
    /// Sysfs listed the topology ([`Topology::read`]).
    pub(crate) topology_read: bool,
}

impl Shape {
    /// `topology` cut to `mask` (the cpus the process may run on), with
    /// `threads` participants or, `None`, one per allowed core. A core whose
    /// lowest cpu the mask leaves out is represented by its lowest allowed
    /// one, so a mask of siblings alone still places one thread a core; a
    /// core with no allowed cpu, and a group with no allowed core, are out.
    /// A mask that shares no cpu with the topology takes its own cpus as one
    /// group of one-cpu cores, and so does a topology sysfs did not list
    /// (`read` false), which has no cpus to cut and so no cut to name.
    pub(crate) fn of(topology: Topology, mask: &[u32], threads: Option<usize>) -> Shape {
        let mut allowed = mask.to_vec();
        allowed.sort_unstable();
        allowed.dedup();
        assert!(!allowed.is_empty(), "the process's cpu mask holds no cpu");
        let listed: usize = topology
            .groups
            .iter()
            .flat_map(|g| &g.cores)
            .map(Vec::len)
            .sum();
        let cores: usize = topology.groups.iter().map(|g| g.cores.len()).sum();
        let ccds = topology.groups.len();
        let mut topo = Vec::new();
        let mut l3 = Vec::new();
        let mut kept_cpus = 0;
        let mut kept_cores = 0;
        for g in &topology.groups {
            let mut primaries = Vec::new();
            let mut siblings = Vec::new();
            for core in &g.cores {
                let mut on = core
                    .iter()
                    .copied()
                    .filter(|c| allowed.binary_search(c).is_ok());
                if let Some(first) = on.next() {
                    primaries.push(first);
                    siblings.extend(on);
                }
            }
            if primaries.is_empty() {
                continue;
            }
            kept_cores += primaries.len();
            kept_cpus += primaries.len() + siblings.len();
            primaries.extend(siblings);
            topo.push(primaries);
            l3.push(g.l3_bytes);
        }
        let unlisted = allowed.len().saturating_sub(kept_cpus);
        let from_mask = topo.is_empty();
        if from_mask {
            kept_cores = allowed.len();
            kept_cpus = allowed.len();
            topo.push(allowed.clone());
            l3.push(0);
        }
        let physical = kept_cores;
        let cut =
            (topology.read && (from_mask || kept_cpus < listed || unlisted > 0)).then_some(Cut {
                topology_cpus: listed,
                kept_cpus: if from_mask { 0 } else { kept_cpus },
                cores,
                kept_cores: if from_mask { 0 } else { kept_cores },
                ccds,
                kept_ccds: if from_mask { 0 } else { topo.len() },
                unlisted,
                from_mask,
            });
        Shape {
            topo,
            l3,
            physical,
            threads: threads.unwrap_or(physical).max(1),
            allowed,
            cut,
            topology_read: topology.read,
        }
    }

    /// The cpus the topology offers the pool's threads.
    pub(crate) fn cpus(&self) -> usize {
        self.topo.iter().map(Vec::len).sum()
    }

    /// Participants past the cpus there are: they share the last slot of
    /// their CCD ([`Shape::cpu_for`]).
    pub(crate) fn oversubscribed(&self) -> usize {
        self.threads.saturating_sub(self.cpus())
    }

    /// Participant `t`'s CCD `t % ccds` and its slot `t / ccds` in that CCD's
    /// cpu list (physical cores first). Beyond the cpus there are the last
    /// slot is shared rather than panicking.
    pub(crate) fn cpu_for(&self, t: usize) -> u32 {
        let (ccd, slot) = ccd_slot(t, self.topo.len());
        let g = &self.topo[ccd];
        g[slot.min(g.len() - 1)]
    }

    /// What building a pool of this shape says, a line each, none when the
    /// pool is the topology as sysfs listed it: the mask's cut, a topology
    /// sysfs did not list, and participants past the cpus there are.
    pub(crate) fn lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(cut) = &self.cut {
            lines.push(format!("bloomery-threads: {cut}; {} threads", self.threads));
        }
        if !self.topology_read {
            lines.push(format!(
                "bloomery-threads: sysfs lists no cpu topology: the pool takes the cpus of the \
                 process's mask as cores of one group ({} threads)",
                self.threads
            ));
        }
        if self.oversubscribed() > 0 {
            lines.push(format!(
                "bloomery-threads: {} threads over {} cpus: the last {} share the last slot of \
                 their L3 group",
                self.threads,
                self.cpus(),
                self.oversubscribed()
            ));
        }
        lines
    }

    /// The shape as a plan prints it.
    pub(crate) fn plan(&self) -> PoolPlan {
        PoolPlan {
            threads: self.threads,
            workers: (0..self.threads - 1).map(|t| self.cpu_for(t)).collect(),
            caller: self.cpu_for(self.threads - 1),
            allowed: self.allowed.clone(),
            cut: self.cut.clone(),
        }
    }
}

/// Participant `t`'s place when the pool spreads over `nccd` CCDs: its CCD
/// `t % nccd` and its slot `t / nccd` in that CCD's cpu list. The one owner of
/// the rule: the pool pins by it ([`crate::Pool`]) and [`crate::CcdMap`] answers by it.
pub(crate) const fn ccd_slot(t: usize, nccd: usize) -> (usize, usize) {
    (t % nccd, t / nccd)
}

/// Where the pool's threads sit, before or after it is built: what a plan
/// prints so a broker can check every task of the process against its grant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolPlan {
    /// The pool's participants, the calling thread among them.
    pub threads: usize,
    /// Worker `t`'s cpu, `t` in `0..threads - 1`. Empty for a built pool
    /// whose pins failed: it floats over `allowed`.
    pub workers: Vec<u32>,
    /// The cpu of the dispatcher's slot, where `Pool::pin_caller` pins the
    /// calling thread. The pool never pins the caller on its own, nor the
    /// helper threads the engine spawns: those float over `allowed`.
    pub caller: u32,
    /// Every cpu the process may run on (its affinity mask), ascending.
    pub allowed: Vec<u32>,
    /// `Some` when the mask took cpus from the topology.
    pub cut: Option<Cut>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A machine of `ccds` L3 groups of `cores` cores of two cpus each, laid
    /// out as the box does: core c of group g is cpu `g*cores + c`, its
    /// sibling `ccds*cores + g*cores + c`.
    fn machine(ccds: u32, cores: u32) -> Topology {
        let smt = ccds * cores;
        Topology {
            groups: (0..ccds)
                .map(|g| CcdGroup {
                    cores: (0..cores)
                        .map(|c| vec![g * cores + c, smt + g * cores + c])
                        .collect(),
                    l3_bytes: 32 << 20,
                })
                .collect(),
            read: true,
        }
    }

    fn all(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    /// An unconstrained mask keeps the topology as sysfs listed it: the
    /// primaries then the siblings of each group, a thread per core, no cut,
    /// and the pins the rule gave before the mask was read.
    #[test]
    fn a_mask_of_every_cpu_changes_nothing() {
        let t = machine(4, 8);
        let s = Shape::of(t.clone(), &all(64), None);
        assert!(s.cut.is_none());
        assert_eq!((s.physical, s.threads), (32, 32));
        assert_eq!(s.topo.len(), 4);
        for (g, cpus) in s.topo.iter().enumerate() {
            let g = g as u32;
            let want: Vec<u32> = (0..8)
                .map(|c| g * 8 + c)
                .chain((0..8).map(|c| 32 + g * 8 + c))
                .collect();
            assert_eq!(cpus, &want, "group {g}");
        }
        assert_eq!(s.l3, vec![32 << 20; 4]);
        // Worker t sits on CCD t % 4 as that CCD's t / 4-th cpu.
        let plan = s.plan();
        assert_eq!(plan.workers.len(), 31);
        assert_eq!(&plan.workers[..6], &[0, 8, 16, 24, 1, 9]);
        assert_eq!(plan.caller, 31);
        assert_eq!(plan.allowed, all(64));
        // The mask's order and repeats do not matter.
        let mut shuffled = all(64);
        shuffled.reverse();
        shuffled.extend(all(64));
        assert_eq!(Shape::of(t, &shuffled, None), s);
    }

    /// A mask of the first CCD's cpus: one group, its eight cores, no
    /// sibling dropped from it, and the cut says what the mask took.
    #[test]
    fn a_mask_of_one_ccd_cuts_the_others() {
        let mask: Vec<u32> = (0..8).chain(32..40).collect();
        let s = Shape::of(machine(4, 8), &mask, None);
        assert_eq!(s.topo, vec![(0..8).chain(32..40).collect::<Vec<u32>>()]);
        assert_eq!((s.physical, s.threads), (8, 8));
        let cut = s.cut.clone().expect("the mask cuts the topology");
        assert_eq!(
            (cut.topology_cpus, cut.kept_cpus, cut.cores, cut.kept_cores),
            (64, 16, 32, 8)
        );
        assert_eq!((cut.ccds, cut.kept_ccds, cut.unlisted), (4, 1, 0));
        assert!(!cut.from_mask);
        let line = cut.to_string();
        assert!(
            line.contains("16 of 64 cpus")
                && line.contains("8 of 32 cores")
                && line.contains("1 of 4 L3 groups"),
            "{line}"
        );
        // The workers of a one-group pool take that group's cpus in order.
        let plan = s.plan();
        assert_eq!(plan.workers, (0..7).collect::<Vec<u32>>());
        assert_eq!(plan.caller, 7);
        assert_eq!(plan.allowed, mask);
    }

    /// A mask of the primaries alone keeps a thread a core with no sibling to
    /// fall back on; a mask of the siblings alone represents each core by its
    /// sibling, so the pool still runs a thread a core.
    #[test]
    fn a_core_is_kept_by_the_cpu_the_mask_leaves_it() {
        let primaries: Vec<u32> = (0..32).collect();
        let s = Shape::of(machine(4, 8), &primaries, None);
        assert_eq!((s.physical, s.threads, s.cpus()), (32, 32, 32));
        assert_eq!(s.cut.as_ref().map(|c| c.kept_cpus), Some(32));
        let siblings: Vec<u32> = (32..64).collect();
        let s = Shape::of(machine(4, 8), &siblings, None);
        assert_eq!((s.physical, s.threads, s.cpus()), (32, 32, 32));
        assert_eq!(s.topo[1], (40..48).collect::<Vec<u32>>());
        // A core with one of its two cpus allowed.
        let s = Shape::of(machine(1, 2), &[1, 2], None);
        assert_eq!(s.topo, vec![vec![2, 1]], "cores in the group's order");
        assert_eq!(s.physical, 2);
        let s = Shape::of(machine(1, 2), &[1, 3], None);
        assert_eq!(s.topo, vec![vec![1, 3]], "one core, its two cpus");
        assert_eq!(s.physical, 1);
    }

    /// A thread count the lever gave stands over the cores there are, and
    /// the participants past the allowed cpus are counted: they share a slot.
    #[test]
    fn a_set_thread_count_past_the_cpus_is_counted() {
        let mask: Vec<u32> = vec![0, 1, 2, 3];
        let s = Shape::of(machine(1, 4), &mask, Some(6));
        assert_eq!((s.physical, s.threads, s.cpus()), (4, 6, 4));
        assert_eq!(s.oversubscribed(), 2);
        assert_eq!(s.cpu_for(5), 3);
        let s = Shape::of(machine(1, 4), &mask, Some(3));
        assert_eq!(s.oversubscribed(), 0);
        // Zero threads is one: the pool always has a participant.
        assert_eq!(Shape::of(machine(1, 4), &mask, Some(0)).threads, 1);
    }

    /// A mask that shares no cpu with the topology (a topology that does not
    /// list the cpus the process has) takes its own cpus as one group, the
    /// cut says so, and the pool is never empty.
    #[test]
    fn a_mask_outside_the_topology_is_its_own_group() {
        let s = Shape::of(machine(1, 2), &[10, 11, 12], None);
        assert_eq!(s.topo, vec![vec![10, 11, 12]]);
        assert_eq!((s.physical, s.threads), (3, 3));
        let cut = s.cut.expect("a cut");
        assert!(
            cut.from_mask && cut.kept_cpus == 0 && cut.unlisted == 3,
            "{cut:?}"
        );
        assert!(
            cut.to_string().contains("none of the topology's cpus"),
            "{cut}"
        );
        // Allowed cpus the topology does not list are named beside a real cut.
        let s = Shape::of(machine(1, 2), &[0, 1, 99], None);
        assert_eq!(s.topo, vec![vec![0, 1]]);
        assert_eq!(s.cut.as_ref().map(|c| c.unlisted), Some(1));
    }

    /// A machine with no topology files runs the cpus of the process's mask
    /// as the cores of one group, whichever cpus they are, and says so in the
    /// pool's line instead of naming a cut.
    #[test]
    fn a_machine_with_no_topology_files_runs_the_masks_cpus() {
        let none = |_: &str| Err(std::io::ErrorKind::NotFound.into());
        let standin = detect_with(&none);
        assert!(!standin.read && standin.groups.is_empty());
        // PIN(2026-10-21): three threads for a mask of three cpus holding cpu
        // 0 (was one: the stand-in was the single cpu 0 cut by the mask).
        let s = Shape::of(standin.clone(), &[0, 1, 2], None);
        assert!(!s.topology_read);
        assert_eq!((s.physical, s.threads), (3, 3));
        assert_eq!(s.topo, vec![vec![0, 1, 2]]);
        assert_eq!(s.cut, None);
        let lines = s.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(
            lines[0].contains("as cores of one group (3 threads)"),
            "{lines:?}"
        );
        // A mask without cpu 0 follows the same rule.
        let s = Shape::of(standin, &(8..16).collect::<Vec<u32>>(), None);
        assert_eq!((s.physical, s.threads), (8, 8));
        assert_eq!(s.topo, vec![(8..16).collect::<Vec<u32>>()]);
        assert_eq!(s.cut, None);
    }

    /// A sysfs that lists its cpus is read whole: one L3 group of two cores
    /// of two cpus each, the lowest cpu of a core first.
    #[test]
    fn a_listed_topology_is_read_as_listed() {
        let sysfs = |path: &str| -> std::io::Result<String> {
            let file = path.rsplit('/').next().unwrap_or_default();
            let cpu = path
                .split("/cpu/cpu")
                .nth(1)
                .and_then(|r| r.split('/').next())
                .and_then(|n| n.parse::<u32>().ok());
            match (cpu, file) {
                (Some(c @ 0..=3), "thread_siblings_list") => Ok(if c % 2 == 0 {
                    format!("{c},{}", c + 1)
                } else {
                    format!("{},{c}", c - 1)
                }),
                (Some(0..=3), "shared_cpu_list") => Ok("0-3".to_string()),
                (Some(0..=3), "size") => Ok("32M\n".to_string()),
                _ => Err(std::io::ErrorKind::NotFound.into()),
            }
        };
        let t = detect_with(&sysfs);
        assert!(t.read);
        assert_eq!(t.groups.len(), 1);
        assert_eq!(t.groups[0].cores, vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(t.groups[0].l3_bytes, 32 << 20);
    }

    /// A pool says nothing when the mask leaves the topology whole, and one
    /// line for each way it does not: the cut, a topology sysfs did not list,
    /// threads past the cpus.
    #[test]
    fn a_pool_names_what_the_mask_took() {
        let whole = Shape::of(machine(4, 8), &all(64), None);
        assert_eq!(whole.lines(), Vec::<String>::new());
        let one = Shape::of(machine(4, 8), &all(8), None);
        let lines = one.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(
            lines[0].starts_with("bloomery-threads: the process's cpu mask cuts the topology")
                && lines[0].ends_with("; 8 threads"),
            "{lines:?}"
        );
        let over = Shape::of(machine(1, 4), &[0, 1], Some(5));
        let lines = over.lines();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(
            lines[1].contains("5 threads over 2 cpus: the last 3 share"),
            "{lines:?}"
        );
        let standin = Topology {
            groups: vec![CcdGroup {
                cores: vec![vec![0]],
                l3_bytes: 0,
            }],
            read: false,
        };
        let lines = Shape::of(standin, &[0], None).lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(
            lines[0].contains("sysfs lists no cpu topology"),
            "{lines:?}"
        );
    }

    /// `cache/index3/size` reads K, M and G suffixes and a bare count, and
    /// refuses a unit or digits it does not know.
    #[test]
    fn cache_sizes_parse() {
        assert_eq!(parse_cache_size("32768K\n"), Some(32 << 20));
        assert_eq!(parse_cache_size("32M"), Some(32 << 20));
        assert_eq!(parse_cache_size("1G"), Some(1 << 30));
        assert_eq!(parse_cache_size("4096"), Some(4096));
        for bad in ["", "K", "12T", "x32M", "-1K", "1.5M"] {
            assert_eq!(parse_cache_size(bad), None, "{bad:?}");
        }
        assert_eq!(parse_cache_size("18446744073709551615K"), None);
    }

    /// An L3 size reads as written, is 0 only where the file is absent, and
    /// a file that is there and holds no size or cannot be read is refused by
    /// name.
    #[test]
    fn an_l3_size_is_absent_or_read_or_refused() {
        let no_file = || Err(std::io::ErrorKind::NotFound.into());
        assert_eq!(l3_size("p", Ok("32768K\n".into())), 32 << 20);
        assert_eq!(l3_size("p", no_file()), 0);
        for (read, named) in [
            (Ok("lots\n".to_string()), "\"lots\\n\""),
            (Ok(String::new()), "\"\""),
            (Err(std::io::ErrorKind::PermissionDenied.into()), "denied"),
        ] {
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                l3_size("sys/index3/size", read)
            }));
            let msg = caught
                .expect_err("a size that is not one is refused")
                .downcast::<String>()
                .expect("a formatted panic");
            assert!(
                msg.contains("sys/index3/size") && msg.contains(named),
                "{msg}"
            );
        }
    }

    /// A sysfs cpu list reads ranges and singles in order.
    #[test]
    fn a_cpu_list_reads_ranges_and_singles() {
        assert_eq!(parse_cpu_list("0,32\n"), vec![0, 32]);
        assert_eq!(parse_cpu_list("0-3,8"), vec![0, 1, 2, 3, 8]);
    }

    /// The machine's cpu topology: on the platform that reads sysfs, the
    /// shared L3 groups it lists; on the platform that names its performance
    /// levels, the performance level's cores and the cache they share, and
    /// the pool's default thread count is that level's core count.
    #[test]
    fn the_topology_is_the_machines_groups() {
        #[cfg(target_os = "linux")]
        {
            let topology = detect_topology();
            assert_eq!(topology.read, !topology.groups.is_empty());
        }
        #[cfg(target_os = "macos")]
        {
            let topology = detect_topology();
            assert!(topology.read, "the performance levels read");
            assert_eq!(topology.groups.len(), 1, "one performance level");
            let group = &topology.groups[0];
            assert!(group.l3_bytes > 0, "the level shares a cache");
            let cpus: Vec<u32> = group.cores.iter().flat_map(|c| c.iter().copied()).collect();
            assert_eq!(cpus, (0..group.cores.len() as u32).collect::<Vec<u32>>());
            // The pool's default thread count is the level's cores.
            let cores = group.cores.len();
            let mask: Vec<u32> = (0..cores as u32 + 2).collect();
            let shape = Shape::of(topology, &mask, None);
            assert_eq!(shape.threads, cores);
            assert_eq!(shape.physical, cores);
        }
    }
}
