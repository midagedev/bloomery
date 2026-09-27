//! Each architecture's families, and the table of all of them. An
//! architecture's families live in its own module; shared code reaches them
//! by the architecture's name — the value its sets' `# arch` line carries,
//! which `model::arch::Arch::name` spells — through [`families`] and
//! [`node_dumps`], never by module path.

use crate::family::{Family, Identity};

pub mod deepseek41;
pub mod deepseek41v;
pub mod glm5next;
pub mod qwen35moe;

/// Each architecture's families, by the architecture's name.
static BY_ARCH: &[(&str, &[&Family])] = &[
    (deepseek41::ARCH, deepseek41::FAMILIES),
    (deepseek41v::ARCH, deepseek41v::FAMILIES),
    (qwen35moe::ARCH, qwen35moe::FAMILIES),
    (glm5next::ARCH, glm5next::FAMILIES),
];

/// The families of architecture `arch`; none for one the table does not hold.
#[must_use]
pub fn families(arch: &str) -> &'static [&'static Family] {
    BY_ARCH
        .iter()
        .find(|(a, _)| *a == arch)
        .map(|&(_, f)| f)
        .unwrap_or(&[])
}

/// The node-dump family of architecture `arch` ([`Identity::Manifest`]):
/// every set of it opens through that family's check.
#[must_use]
pub fn node_dumps(arch: &str) -> Option<&'static Family> {
    families(arch)
        .iter()
        .copied()
        .find(|f| f.identity == Identity::Manifest)
}

/// Every family of the table.
pub fn all() -> impl Iterator<Item = &'static Family> {
    BY_ARCH.iter().flat_map(|(_, f)| f.iter().copied())
}

/// The family named `name`.
#[must_use]
pub fn named(name: &str) -> Option<&'static Family> {
    all().find(|f| f.name == name)
}

#[cfg(test)]
mod tests {
    use super::all;

    /// Every family's sets in place pass their family's check: each states
    /// the file the tree runs, names its family's ik build where the family
    /// pins one, and is complete. One line per set, then the verdict.
    #[test]
    #[ignore = "hw: needs the reference sets in $BLOOMERY_DATA and BLOOMERY_DSPARK_MODEL on the box"]
    fn hw_every_set_in_place_is_its_familys() {
        let mut failed = 0usize;
        let mut sets = 0usize;
        for f in all() {
            for path in f.in_place() {
                sets += 1;
                match f.check_set(&path) {
                    Ok(p) => println!(
                        "hw_refset: {} {}: dumped from {}{} — PASS",
                        f.name,
                        path.display(),
                        p.dumped_from,
                        p.build.map_or_else(String::new, |b| format!(" build {b}"))
                    ),
                    Err(e) => {
                        failed += 1;
                        println!("hw_refset: {} {}: FAIL {e}", f.name, path.display());
                    }
                }
            }
        }
        println!("hw_refset: {sets} set(s) in place, {failed} failed");
        assert_eq!(
            failed, 0,
            "hw_refset: {failed} set(s) in place fail their family's check"
        );
    }
}
