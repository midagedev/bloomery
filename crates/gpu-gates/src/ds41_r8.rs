//! The V4.1 host tier's r8 pick for a load whose plan pages routed experts through the NVMe tier's
//! arena: the one rule `bloomery-serve-ds41` and `generate_ds41` take. The arena's fills read the
//! source's bytes and cannot read an r8 sidecar (`bloomery_gpu::model::nvme_tier` refuses that pair
//! once an arena is granted), so where the plan reserves an arena, `BLOOMERY_R8` is unset and a
//! sidecar sits beside the file, the host tier reads the source, named on stderr; a set `on` stays
//! the user's and meets the refusal. A plan with no arena — every id the cards do not hold read
//! through the file mappings, the paged ones too — reads the sidecar as any other load does.

use std::path::{Path, PathBuf};

use bloomery_levers::{HostCfg, Levers};

use crate::GateError;

/// The r8 sidecar beside the model file, `None` when no file is there
/// ([`model::r8file::sidecar_path`]).
pub fn sidecar_beside(path: &Path) -> Result<Option<PathBuf>, GateError> {
    let sidecar = model::r8file::sidecar_path(path)?;
    Ok(sidecar.try_exists()?.then_some(sidecar))
}

/// The line `who` prints when an unset `BLOOMERY_R8` reads the source instead of `sidecar`, the
/// plan reserving `arena` bytes of the NVMe tier's arena.
fn source_note(who: &str, sidecar: &Path, arena: u64) -> String {
    format!(
        "{who}: BLOOMERY_R8 is unset and the plan pages routed experts through an NVMe tier arena \
         of {arena} B, whose fills read the source's bytes: the host tier reads the source, not \
         the sidecar at {} (the `load host_tier r8=off` line below is this choice; BLOOMERY_R8=on \
         refuses the pair)",
        sidecar.display()
    )
}

/// The host tier's settings for the load of the file at `path` that `who` makes, whose plan
/// reserves `arena` bytes of the NVMe tier's arena (`HostTotals::nvme_arena_bytes`; 0 on a plan
/// that pages nothing or pages through the file mappings): [`Levers::host_for`] with the sidecar
/// beside `path` under a plan with an arena as its `unfit`, and the choice named on stderr when it
/// moved `r8`. The load's grant is never more than the plan's arena, so a pick that keeps `r8`
/// never meets the tier's refusal.
pub fn host_at_plan(
    levers: &Levers,
    who: &str,
    path: &Path,
    arena: u64,
) -> Result<HostCfg, GateError> {
    let sidecar = if arena > 0 {
        sidecar_beside(path)?
    } else {
        None
    };
    let host = levers.host_for(sidecar.is_some());
    if let Some(sidecar) = sidecar.filter(|_| levers.host().r8 && !host.r8) {
        eprintln!("{}", source_note(who, &sidecar, arena));
    }
    Ok(host)
}

#[cfg(test)]
mod tests {
    use super::{host_at_plan, sidecar_beside, source_note};
    use bloomery_levers::Levers;
    use std::path::Path;

    /// A model file's first shard in a directory of its own, and its sidecar's place beside it.
    fn model_in(dir: &Path) -> std::path::PathBuf {
        dir.join("src").join("m-00001-of-00002.gguf")
    }

    /// The sidecar is the file `r8file::sidecar_path` names, and `None` where none is there.
    #[test]
    fn the_sidecar_is_beside_the_file_only_where_it_is_written() {
        let dir = std::env::temp_dir().join(format!("bloomery-ds41-r8-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).expect("the source directory");
        let model = model_in(&dir);
        assert_eq!(sidecar_beside(&model).expect("a path"), None);
        let side = dir.join("src-r8");
        std::fs::create_dir_all(&side).expect("the sidecar directory");
        std::fs::write(side.join("m-r8.gguf"), b"").expect("the sidecar");
        assert_eq!(
            sidecar_beside(&model).expect("a path"),
            Some(side.join("m-r8.gguf"))
        );
        std::fs::remove_dir_all(&dir).expect("the directory");
    }

    /// A load whose plan reserves no NVMe tier arena, or finds no sidecar, keeps the levers' own
    /// host settings; one whose plan reserves an arena with a sidecar beside the file reads the
    /// source unless the user set `BLOOMERY_R8` (then the set value stands, `on` meeting the load's
    /// refusal). The process's own `BLOOMERY_R8`, set or not, is what the last two assertions
    /// compare with. Mutants: `arena` ignored (every sidecar load reads the source, the default
    /// paged plan with no arena among them), `host_for` skipped (a load with an arena keeps r8
    /// under a sidecar).
    #[test]
    fn an_arena_load_under_a_sidecar_reads_the_source_where_r8_is_unset() {
        let levers = Levers::from_env().expect("the process's levers");
        let dir =
            std::env::temp_dir().join(format!("bloomery-ds41-r8-pick-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).expect("the source directory");
        let model = model_in(&dir);
        let plain = host_at_plan(&levers, "test", &model, 1).expect("no sidecar");
        assert_eq!(plain, levers.host(), "no sidecar: nothing moves");
        let side = dir.join("src-r8");
        std::fs::create_dir_all(&side).expect("the sidecar directory");
        std::fs::write(side.join("m-r8.gguf"), b"").expect("the sidecar");
        let mapped = host_at_plan(&levers, "test", &model, 0).expect("no arena");
        assert_eq!(
            mapped,
            levers.host(),
            "no arena: the mappings carry the sidecar, nothing moves"
        );
        let paged = host_at_plan(&levers, "test", &model, 1).expect("an arena");
        assert_eq!(
            paged.r8,
            levers.r8_set().unwrap_or(false),
            "an arena under a sidecar: unset reads the source, a set value stands"
        );
        assert_eq!(
            paged,
            bloomery_levers::HostCfg {
                r8: paged.r8,
                ..levers.host()
            },
            "only r8 moves"
        );
        std::fs::remove_dir_all(&dir).expect("the directory");
    }

    /// The note names its binary, the sidecar's path, the plan's arena and the one setting that
    /// refuses the pair.
    #[test]
    fn the_note_names_its_binary_and_the_sidecar() {
        let text = source_note("generate_ds41", Path::new("/m/src-r8/m-r8.gguf"), 4096);
        assert!(
            text.starts_with("generate_ds41: BLOOMERY_R8 is unset"),
            "{text}"
        );
        assert!(text.contains("/m/src-r8/m-r8.gguf"), "{text}");
        assert!(text.contains("arena of 4096 B"), "{text}");
        assert!(text.contains("BLOOMERY_R8=on refuses the pair"), "{text}");
    }
}
